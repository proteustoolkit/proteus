"""Build and run Proteus problems from an advection-diffusion-reaction form.

The input is the plain-data problem that ``ymf.symbolic.adr_problem``
produces from a YMF specification: for each scalar equation, the mass,
advective flux, diffusion tensors, reaction and Hamiltonian as numpy code
strings, with their derivatives and Proteus's dependence flags; plus the
geometry, Dirichlet data, initial conditions and the exact solution.

That is the strong form, already sorted into the slots of Proteus's

    dm_i/dt + div(f_i - sum_k a_ik grad(phi_k)) + r_i + H_i = 0

Two transformations then make it a discrete problem, and both are kept
separate here because each can map one input to several outputs:

- strong -> continuous weak: :class:`ADRCoefficients` gives the equations
  to Proteus's transport machinery, which integrates the flux terms by
  parts; the boundary treatment (strong Dirichlet, outflow fluxes) is set
  by :func:`physics`.
- continuous weak -> discrete weak: :func:`numerics` chooses the finite
  element spaces per unknown, quadrature, stabilization and time stepping.

Nothing here imports sympy: the symbolic work happened in the front end,
and this module only compiles and evaluates the code strings it was given.
"""

import os

import numpy

from proteus import (FemTools, LinearAlgebraTools, LinearSolvers, MeshTools,
                     NonlinearSolvers, NumericalFlux, Quadrature, StepControl,
                     TimeIntegration, defaults)
from proteus.TransportCoefficients import TC_base
from proteus.csparsity import CSR_INDEX_DTYPE

__all__ = ["ADRCoefficients", "physics", "numerics", "run", "emit_pn", "describe",
           "l2_errors", "ELEMENTS"]


def _compile(code):
    return compile(code, "<adr>", "eval")


#: names the code strings may use, beyond coordinates, time and the unknowns
_GLOBALS = {"__builtins__": {"abs": abs}, "numpy": numpy}


class _Evaluator(object):
    """Compiled code strings, evaluated against a quadrature dictionary."""

    def __init__(self, components, dim):
        self.components = list(components)
        self.dim = dim

    def namespace(self, t, c, with_gradients):
        x = c['x']
        ns = {"t": t, "x": x[..., 0]}
        if self.dim > 1:
            ns["y"] = x[..., 1]
        if self.dim > 2:
            ns["z"] = x[..., 2]
        for j, name in enumerate(self.components):
            if ('u', j) in c:
                ns[name] = c[('u', j)]
            if with_gradients and ('grad(u)', j) in c:
                g = c[('grad(u)', j)]
                for k in range(self.dim):
                    ns["grad_%s_%d" % (name, k)] = g[..., k]
        return ns


class ADRCoefficients(TC_base):
    """A TC_base whose coefficients are given as numpy code strings.

    ``adr`` is the ``"adr"`` part of a ymf ADR problem: ``components`` (the
    scalar unknowns, in order) and one entry per equation.
    """

    def __init__(self, adr):
        self.adr = adr
        self.nd = adr["dim"]
        names = list(adr["components"])
        index = {name: i for i, name in enumerate(names)}
        self._eval = _Evaluator(names, self.nd)
        self._needs_gradients = False

        mass, advection, diffusion, potential, reaction, hamiltonian = {}, {}, {}, {}, {}, {}
        sdInfo = {}
        self._terms = []   # (key, compiled code, slice, source) to fill on evaluate
        for i, eq in enumerate(adr["equations"]):
            def flags(block, constant_key=i):
                # ymf says how a coefficient depends on each unknown, and
                # nothing for a constant; Proteus wants 'constant' keyed by
                # some component, by convention the equation's own.
                deps = {index[c]: f for c, f in block["depends_on"].items()}
                return deps or {constant_key: 'constant'}

            if "mass" in eq:
                b = eq["mass"]
                mass[i] = flags(b)
                self._add(('m', i), b["m"])
                for c, code in b["dm"].items():
                    self._add(('dm', i, index[c]), code)
            if "advection" in eq:
                b = eq["advection"]
                advection[i] = flags(b)
                for k, code in enumerate(b["f"]):
                    self._add(('f', i), code, k)
                for c, codes in b["df"].items():
                    for k, code in enumerate(codes):
                        self._add(('df', i, index[c]), code, k)
            if "diffusion" in eq:
                diffusion[i] = {}
                for phi, b in eq["diffusion"].items():
                    ck = index[phi]
                    # Proteus's diffusion flags are 'constant' or 'nonlinear'
                    diffusion[i][ck] = ({index[c]: 'nonlinear' for c in b["depends_on"]}
                                        or {ck: 'constant'})
                    potential[ck] = {ck: 'u'}
                    sdInfo[(i, ck)] = (numpy.array(b["rowptr"], CSR_INDEX_DTYPE),
                                       numpy.array(b["colind"], CSR_INDEX_DTYPE))
                    for n, code in enumerate(b["a"]):
                        self._add(('a', i, ck), code, n)
                    for c, codes in b["da"].items():
                        for n, code in enumerate(codes):
                            self._add(('da', i, ck, index[c]), code, n)
            if "reaction" in eq:
                b = eq["reaction"]
                reaction[i] = flags(b)
                self._add(('r', i), b["r"])
                for c, code in b["dr"].items():
                    self._add(('dr', i, index[c]), code)
            if "hamiltonian" in eq:
                b = eq["hamiltonian"]
                hamiltonian[i] = {index[c]: f for c, f in b["depends_on"].items()}
                self._needs_gradients = True
                self._add(('H', i), b["H"])
                for c, codes in b["dH"].items():
                    for k, code in enumerate(codes):
                        self._add(('dH', i, index[c]), code, k)

        TC_base.__init__(self, nc=len(names), mass=mass, advection=advection,
                         diffusion=diffusion, potential=potential, reaction=reaction,
                         hamiltonian=hamiltonian, variableNames=names,
                         sparseDiffusionTensors=sdInfo)
        vectors = [i for i, n in enumerate(names) if n.endswith("_0")]
        if vectors:
            stem = names[vectors[0]][:-2]
            self.vectorComponents = [index["%s_%d" % (stem, k)] for k in range(self.nd)]
            self.vectorName = stem

    def _add(self, key, code, last=None):
        self._terms.append((key, _compile(code), last, code))

    def evaluate(self, t, c):
        ns = self._eval.namespace(t, c, self._needs_gradients)
        for key, code, last, _ in self._terms:
            if key not in c:
                continue
            value = eval(code, _GLOBALS, ns)
            if last is None:
                c[key][...] = value
            else:
                c[key][..., last] = value


class _Field(object):
    """uOfXT for one component, from a code string (x is one point or many)."""

    def __init__(self, code, dim):
        self.code = _compile(code)
        self.dim = dim

    def uOfXT(self, x, t):
        x = numpy.asarray(x)
        ns = {"t": t, "x": x[..., 0]}
        if self.dim > 1:
            ns["y"] = x[..., 1]
        if self.dim > 2:
            ns["z"] = x[..., 2]
        value = eval(self.code, _GLOBALS, ns)
        return value + 0.0 * x[..., 0] if numpy.ndim(x) > 1 else value

    def uOfX(self, x):
        return self.uOfXT(x, 0.0)


def _selector(entry):
    """``select(ns, flag)``: whether one condition claims a boundary point.

    A region found by position carries ``where``, boolean code over the
    point (None: the whole boundary); a region found by the mesh's boundary
    tags carries ``flags`` instead, and no ``where``, and claims the points
    whose boundary flag is one of them.
    """
    if "flags" in entry:
        flags = frozenset(int(f) for f in entry["flags"])
        return lambda ns, flag: flag in flags
    if entry["where"] is None:
        return lambda ns, flag: True
    code = _compile(entry["where"])
    return lambda ns, flag: bool(eval(code, _GLOBALS, ns))


def _test_text(entry):
    """The same test as Python source, for :func:`emit_pn`'s files."""
    if "flags" in entry:
        return "flag in %r" % (sorted(int(f) for f in entry["flags"]),)
    return "True" if entry["where"] is None else entry["where"]


def _dirichlet(entries, dim, tol):
    """getDBC(x, flag) for one component.

    Proteus calls it for every node, interior ones included, with flag 0
    for the interior; it must return None there, and for boundary nodes
    no entry claims.
    """
    tests = [(_selector(e), _value_of(e, dim)) for e in entries]

    def getDBC(x, flag):
        if flag == 0:
            return None
        ns = {"tol": tol, "x": x[0]}
        if dim > 1:
            ns["y"] = x[1]
        if dim > 2:
            ns["z"] = x[2]
        for select, value in tests:
            if select(ns, flag):
                return value
        return None
    return getDBC


def _robin(entries, dim, tol):
    """``get(x, flag)`` for one component's Robin conditions.

    Returns None where no condition claims the point, else a callable
    ``(x, t) -> (alpha, beta, g)`` for ``alpha*u + beta*(q.n) = g``.
    NumericalFlux.RobinFactory turns that into the flux and its exact
    derivative; see there for why this cannot be a flux boundary condition.
    """
    tests = [(_selector(e), _Field(e["alpha"], dim), _Field(e["beta"], dim),
              _Field(e["g"], dim))
             for e in entries]

    def get(x, flag):
        ns = {"tol": tol, "x": x[0]}
        if dim > 1:
            ns["y"] = x[1]
        if dim > 2:
            ns["z"] = x[2]
        for select, alpha, beta, g in tests:
            if select(ns, flag):
                return lambda x, t, a=alpha, b=beta, gg=g: (
                    a.uOfXT(x, t), b.uOfXT(x, t), gg.uOfXT(x, t))
        return None
    return get


def _neumann_targets(problem, adr, names):
    """``{component index: (potential index, entries)}`` for the neumann conditions.

    A neumann condition prescribes the outward diffusive flux, so it attaches
    to a (component, potential) pairing of the equation's diffusion term.
    With one diffusion term the pairing is unambiguous; with several it is
    not, and guessing would put the flux silently on the wrong potential.

    Shared by :func:`physics` and :func:`emit_pn` so the two cannot disagree
    about which pairing carries the condition.
    """
    by_component = {}
    for e in problem.get("neumann") or []:
        by_component.setdefault(e["component"], []).append(e)
    targets = {}
    for i, eq in enumerate(adr["equations"]):
        here = by_component.pop(names[i], [])
        if not here:
            continue
        potentials = list(eq.get("diffusion", {}))
        if not potentials:
            raise ValueError(
                "neumann condition on %s, which has no diffusion term: there "
                "is no diffusive flux to prescribe" % (names[i],))
        if len(potentials) > 1 and names[i] not in potentials:
            raise ValueError(
                "neumann condition on %s, whose equation diffuses %s: say "
                "which potential the flux belongs to"
                % (names[i], ", ".join(sorted(potentials))))
        phi = names[i] if names[i] in potentials else potentials[0]
        targets[i] = (names.index(phi), here)
    if by_component:
        raise ValueError("neumann condition on %s, which is not an unknown"
                         % ", ".join(sorted(by_component)))
    return targets


def _value_of(entry, dim):
    """The ``(x, t)`` value one condition supplies: a formula or a table.

    How a value is supplied is orthogonal to what kind of condition it is, so
    this serves Dirichlet and flux conditions alike: a table under a flux is
    a discharge hydrograph, under a Dirichlet condition a stage hydrograph or
    a measured inlet value.

    ``numpy.interp`` is linear between samples and holds the end values
    outside the range, because extrapolating a record past its end is not
    something to do quietly.
    """
    if entry.get("series") is not None:
        times = numpy.array([t for t, _ in entry["series"]], dtype=float)
        values = numpy.array([v for _, v in entry["series"]], dtype=float)
        return lambda x, t: float(numpy.interp(t, times, values))
    field = _Field(entry["value"], dim)
    return lambda x, t: field.uOfXT(x, t)


def _neumann(entries, dim, tol):
    """getDFBC(x, flag) for one (component, potential) pair.

    The value is the OUTWARD diffusive flux q.n with q = -a grad u, which is
    what Proteus's diffusiveFluxBoundaryConditions wants: see
    test/poisson_2d/poisson_het_2d_p.py, whose getDiffFluxBC2 returns
    dot(-A grad u, n) and is checked against an exact solution.

    Unlike getDBC this is called only at exterior boundary quadrature
    points, so there is no interior sentinel to guard against. Returning
    None leaves the face to the natural condition, which for a diffusion
    term is zero flux.
    """
    tests = [(_selector(e), _value_of(e, dim)) for e in entries]

    def getDFBC(x, flag):
        ns = {"tol": tol, "x": x[0]}
        if dim > 1:
            ns["y"] = x[1]
        if dim > 2:
            ns["z"] = x[2]
        for select, value in tests:
            if select(ns, flag):
                return value
        return None
    return getDFBC


def _periodic(axes, lower, upper, tol, dirichlet):
    """getPDBC(x, flag): the key point that identifies periodic copies of a node.

    A node on either face of a periodic axis maps to the same key -- its
    coordinates with that axis snapped to the lower face -- and Proteus
    gives nodes with equal keys one degree of freedom.

    Proteus lets a periodic match override a Dirichlet value at the same
    node, which would silently drop, say, a wall velocity at a corner or a
    pressure datum on a periodic face. ``dirichlet`` is this component's
    getDBC; where it claims a node, the node is not periodic.
    """
    def getPDBC(x, flag):
        if dirichlet(x, flag if flag else 1) is not None:
            return None
        on = [a for a in axes
              if abs(x[a] - lower[a]) <= tol or abs(x[a] - upper[a]) <= tol]
        if not on:
            return None
        key = numpy.array([round(float(v), 8) for v in x[:3]])
        for a in on:
            key[a] = lower[a]
        return key
    return getPDBC


def _zero_on_periodic_faces(axes, lower, upper, tol):
    def getAFBC(x, flag):
        if any(abs(x[a] - lower[a]) <= tol or abs(x[a] - upper[a]) <= tol for a in axes):
            return lambda x, t: 0.0
        return None
    return getAFBC


def physics(problem, name):
    """The continuous problem: domain, coefficients, boundary and initial data."""
    adr = problem["adr"]
    dim = adr["dim"]
    names = adr["components"]
    lower, upper = problem["domain"]["lower"], problem["domain"]["upper"]
    p = defaults.Physics_base(nd=dim, name=name)
    # The old-style box route (domain None, L and x0) is the one that hands
    # the numerics' nnx/nny/nnz to the mesh generator. A piecewise_linear
    # region is a PSLG or PLC domain, meshed with triangle or tetgen; its
    # bounding box still sets L, the scale of the boundary tolerance.
    region = problem["domain"].get("region")
    if region is None:
        p.domain = None
    else:
        from proteus import DomainYMF
        p.domain = DomainYMF.from_ymf(region)
        p.domain.name = name
    p.L = tuple(u - l for u, l in zip(upper, lower)) + (1.0,) * (3 - dim)
    p.x0 = tuple(lower) + (0.0,) * (3 - dim)
    p.coefficients = ADRCoefficients(adr)
    constant = problem.get("up_to_constant") or []
    if constant:
        # determined only up to a constant: the linear solver keeps it in
        # the null space of the operator (see numerics)
        p.coefficients.nullSpace = "ConstantComponentsNullSpace"
        p.coefficients.nullSpaceComponents = [names.index(c) for c in constant]

    tol = 1e-8 * max(p.L[:dim])
    # A Robin point is not a Dirichlet point and carries no flux boundary
    # condition: both are applied after the numerical flux runs (see
    # OneLevelTransport.calculateExteriorElementBoundaryCoefficients) and
    # would overwrite it. The Robin class is the only thing that may write
    # the flux there.
    p.robinConditions = {}
    robin = {}
    for e in problem.get("robin") or []:
        robin.setdefault(e["component"], []).append(e)
    for name in robin:
        if name not in names:
            raise ValueError("robin condition on %s, which is not an unknown" % name)
    by_component = {}
    for e in problem["dirichlet"]:
        by_component.setdefault(e["component"], []).append(e)
    p.dirichletConditions = {i: _dirichlet(by_component.get(n, []), dim, tol)
                             for i, n in enumerate(names)}
    periodic = {e["component"]: e["axes"] for e in problem.get("periodic", [])}
    if periodic:
        p.periodicDirichletConditions = {
            i: (_periodic(periodic[n], lower, upper, tol, p.dirichletConditions[i])
                if n in periodic else (lambda x, flag: None))
            for i, n in enumerate(names)}
    # On a periodic face the boundary flux of each equation cancels against
    # the opposite face's, so it is set to zero there, on both faces.
    # Computing it from the trace instead (as on every other boundary) is
    # what Proteus does by default, and when the flux couples components --
    # the pressure in the momentum flux -- the two faces do not cancel and
    # the solution is garbage. Elsewhere the flux comes from the trace.
    p.advectiveFluxBoundaryConditions = {
        i: _zero_on_periodic_faces(periodic.get(n, []), lower, upper, tol)
        for i, n in enumerate(names)}
    p.robinConditions = {i: _robin(robin[n], dim, tol)
                         for i, n in enumerate(names) if n in robin}
    targets = _neumann_targets(problem, adr, names)
    p.diffusiveFluxBoundaryConditions = {i: {} for i in range(len(names))}
    for i, eq in enumerate(adr["equations"]):
        target, entries = targets.get(i, (None, []))
        for phi in eq.get("diffusion", {}):
            ck = names.index(phi)
            p.diffusiveFluxBoundaryConditions[i][ck] = (
                _neumann(entries, dim, tol) if ck == target
                else (lambda x, flag: None))
    # Advective fluxes through the boundary are computed from the solution's
    # trace -- on Dirichlet boundaries, the boundary data -- so inflow and
    # outflow are both handled; 'noFlow' would drop the term.
    p.fluxBoundaryConditions = {i: ('outFlow' if "advection" in eq else 'noFlow')
                                for i, eq in enumerate(adr["equations"])}

    exact = {names.index(n): _Field(code, dim) for n, code in problem.get("exact", {}).items()}
    initial = {names.index(n): _Field(code, dim) for n, code in problem.get("initial", {}).items()}
    # No initial condition means a steady problem's starting guess: zero,
    # never the exact solution, which would hand Newton the answer.
    for i in range(len(names)):
        if i not in initial:
            initial[i] = _Field("0.0", dim)
    p.initialConditions = initial
    p.analyticalSolution = exact
    return p


#: (family, order) -> Proteus finite element space on simplices
ELEMENTS = {
    ("CG", 1): FemTools.C0_AffineLinearOnSimplexWithNodalBasis,
    ("CG", 2): FemTools.C0_AffineQuadraticOnSimplexWithNodalBasis,
    ("DG", 0): FemTools.DG_Constants,
    ("DG", 1): FemTools.DG_AffineLinearOnSimplexWithNodalBasis,
    ("DG", 2): FemTools.DG_AffineQuadraticOnSimplexWithNodalBasis,
}


def reorder(problem, first):
    """The same problem with the components (and their equations) reordered.

    Equation i belongs to component i, so the two are permuted together;
    everything else refers to components by name. ``first`` lists the
    components to put first, in order.
    """
    import copy
    adr = problem["adr"]
    names = list(adr["components"])
    order = list(first) + [n for n in names if n not in first]
    out = copy.deepcopy(problem)
    out["adr"]["components"] = order
    out["adr"]["equations"] = [copy.deepcopy(adr["equations"][names.index(n)]) for n in order]
    return out


#: weak-form stabilization methods numerics() can install
STABILIZATIONS = ("none", "supg/pspg")


def velocity_pressure(problem):
    """(velocity components, pressure component) of a velocity-pressure system."""
    vectors = [c for c in problem["unknowns"].values() if len(c) > 1]
    scalars = [c[0] for c in problem["unknowns"].values() if len(c) == 1]
    if len(vectors) != 1 or len(scalars) != 1:
        raise ValueError("SUPG/PSPG is wired for velocity-pressure systems (one vector "
                         "and one scalar unknown), not %s" % list(problem["unknowns"]))
    return vectors[0], scalars[0]


def check_stabilization(problem, stabilization):
    """The stabilization, normalized, if Proteus can install it for this problem.

    Raises ValueError saying why not otherwise. Shared by :func:`numerics`
    and ``pro weak``, which offers a formulation only where it would run.
    """
    stabilization = (stabilization or "none").lower()
    if stabilization not in STABILIZATIONS:
        raise ValueError("stabilization %r is not wired up (have %s)"
                         % (stabilization, ", ".join(STABILIZATIONS)))
    if stabilization == "supg/pspg":
        velocity, _ = velocity_pressure(problem)
        # The stabilization reads the density from the momentum equations'
        # mass coefficient (dm), so they need their ρ ∂v/∂t term even when
        # the solve is steady (which drops it).
        missing = [eq["component"] for eq in problem["adr"]["equations"]
                   if eq["component"] in velocity and "mass" not in eq]
        if missing:
            raise ValueError("SUPG/PSPG takes the density from the time derivative "
                             "ρ ∂v/∂t, which the equations for %s lack; include it "
                             "(a steady solve drops it)" % ", ".join(missing))
    return stabilization


#: PETSc options for a singular operator (a constant in its null space)
NULL_SPACE_SOLVER = {"ksp_type": "gmres", "pc_type": "lu",
                     "pc_factor_mat_solver_type": "mumps", "mat_mumps_icntl_24": "1",
                     "ksp_rtol": "1e-14", "ksp_atol": "1e-14", "ksp_max_it": "50"}

#: a direct solve through PETSc: LU by MUMPS, applied once
MUMPS_SOLVER = {"ksp_type": "preonly", "pc_type": "lu",
                "pc_factor_mat_solver_type": "mumps"}

#: the spec's ``linear_solver`` values Proteus has a solver for (see
#: :func:`linear_solver_choice`); ``umfpack`` is the schema's fourth, and has none
LINEAR_SOLVERS = ("superlu", "mumps", "petsc")


def _ranks():
    """The number of MPI ranks of a Proteus that has started; 1 if none has
    (this never starts MPI: a linter on a login node must not)."""
    from proteus import Comm
    return Comm.comm.size() if Comm.comm is not None else 1


def parallel_periodic(problem, spaces, ranks=None):
    """Whether a run's periodic conditions go through Proteus's parallel
    periodic numbering (``n.parallelPeriodic``): on more than one rank,
    for a periodic problem. Raises ValueError where it cannot.

    That numbering merges each periodic pair of mesh nodes into one unknown
    across ranks, on an element partition without overlap, and gives every
    component component 0's map: so every component must be in CG1, whose
    degrees of freedom are the mesh nodes. On one rank the serial path
    merges each component's own pairs, in any space.

    Advection is allowed: the faces between ranks are exterior faces of each
    rank's mesh, but the weak upwind fluxes ADRProblem uses give the two
    ranks equal and opposite interface fluxes, which cancel when the
    shared dofs are summed (test_adr_problem, advection across a rank cut).
    """
    if not problem.get("periodic"):
        return False
    if ranks is None:
        ranks = _ranks()
    if ranks == 1:
        return False
    if set(tuple(s) for s in spaces.values()) != {("CG", 1)}:
        raise ValueError(
            "periodic conditions on more than one rank need every component in CG1 "
            "(this run has %d ranks, and %s): Proteus's parallel periodic numbering is "
            "nodal and shared by all components; run on one process"
            % (ranks, ", ".join("%s %s%d" % (c, f, o) for c, (f, o) in spaces.items())))
    return True


def linear_solver_choice(problem, spaces, linear_solver=None, petsc_options=None,
                         ranks=None):
    """``(solver, petsc options)`` for a spec's ``linear_solver`` and ``petsc_options``.

    * ``superlu``: Proteus's own LU (SuperLU), serial; refused on more than
      one rank.
    * ``mumps``: a direct solve through PETSc, LU by MUMPS.
    * ``petsc``: a PETSc Krylov solve, set by ``petsc_options``; PETSc's
      defaults without them.
    * omitted: what Proteus picked before it read the field -- its serial
      LU, or MUMPS when the operator has a null space. On more than one
      rank: MUMPS, or ``petsc`` where this PETSc has no MUMPS.

    ``ranks`` is the run's number of MPI ranks (default: the started
    Proteus's, else 1).

    A singular operator (``problem["up_to_constant"]``) needs a PETSc
    solver, which removes the null space, and MUMPS's null-pivot detection
    when it is ``mumps``. ``petsc_options`` (without the leading dash) are
    laid over ``mumps``'s or applied as ``petsc``'s, and refused for
    anything else rather than ignored.
    """
    null_space = bool(problem.get("up_to_constant"))
    solver = None if linear_solver is None else str(linear_solver).strip().lower()
    if solver is not None and solver not in LINEAR_SOLVERS:
        raise ValueError("linear_solver %r has no Proteus solver (have %s)"
                         % (linear_solver, ", ".join(LINEAR_SOLVERS)))
    if petsc_options and solver not in ("petsc", "mumps"):
        raise ValueError("petsc_options apply to linear_solver petsc or mumps, not %s"
                         % ("the default" if solver is None else solver))
    if ranks is None:
        ranks = _ranks()
    if solver is None:
        if ranks > 1:
            from petsc4py import PETSc
            solver = "mumps" if PETSc.Sys.hasExternalPackage("mumps") else "petsc"
        else:
            solver = "mumps" if null_space else "superlu"
    if solver == "superlu" and ranks > 1:
        raise ValueError("linear_solver superlu is serial, and this run has %d ranks; "
                         "give mumps or petsc, or run on one process" % (ranks,))
    if solver == "superlu":
        if null_space:
            raise ValueError(
                "the operator is singular (%s determined up to a constant), so it "
                "needs linear_solver mumps or petsc, which remove the null space; "
                "superlu does not" % ", ".join(problem["up_to_constant"]))
        return solver, None
    if null_space and any(f == "DG" for f, _ in spaces.values()):
        raise ValueError("a null space with DG spaces is not wired up")
    if solver == "mumps":
        from petsc4py import PETSc
        if not PETSc.Sys.hasExternalPackage("mumps"):
            raise ValueError("linear_solver mumps, but this PETSc was built without MUMPS")
        options = dict(NULL_SPACE_SOLVER if null_space else MUMPS_SOLVER)
    else:
        options = {}
    options.update({str(k): str(v) for k, v in (petsc_options or {}).items()})
    return solver, options


def keeps_dirichlet_dofs(n, spaces):
    """Whether strong Dirichlet DOFs stay in the system, rows replaced by u = g.

    The layout :class:`LinearSolvers.KSP_petsc4py` expects, so it follows
    the linear solver. DG spaces impose Dirichlet data weakly, through their
    numerical flux, and have no such DOFs.
    """
    return (n.levelLinearSolver is LinearSolvers.KSP_petsc4py
            and not any(f == "DG" for f, _ in spaces.values()))


def mesh_constraint(h, dim, boundary_points=True):
    """triangle's (2-D) or tetgen's (3-D) options for element size ``h``.

    Proteus's own convention, as in its models: a maximum triangle area of
    h**2/2 or tetrahedron volume of h**3/6 (the right simplex with legs h).
    ``boundary_points=False`` adds ``Y``: the generator puts no points on the
    boundary beyond those it was given, which a periodic region needs (see
    :func:`periodic_region`).
    """
    y = "" if boundary_points else "Y"
    if dim == 2:
        return "VApq30Dena%8.8f%s" % (h ** 2 / 2.0, y)
    if dim == 3:
        return "VApq1.35q12feena%e%s" % (h ** 3 / 6.0, y)
    raise ValueError("an unstructured mesh needs dimension 2 or 3, not %d" % dim)


#: file formats whose file is a mesh Proteus reads; a geometry (gmsh_geo) is
#: meshed by h. SMS/ADH 2dm and 3dm meshes are converted to ymf first
#: (proteustoolkit/ymf-import-sms): Proteus no longer reads them
MESH_FILE_FORMATS = ("hex", "tetgen", "ymf")
CONVERTED_FORMATS = ("2dm", "3dm")


def _is_file(region):
    return region is not None and region.get("kind") == "file"


def _is_discrete(region):
    """A ``discrete`` region that is a mesh to solve on (a triangulated surface is not)."""
    if region is None or region.get("kind") != "discrete":
        return False
    from proteus import DomainYMF
    return not DomainYMF._is_surface(region)


def _discrete_dimension(region):
    return 2 if region["topology"]["type"] == "Triangle" else 3


def file_bounds(region):
    """``(lower, upper)``: the bounding box of a mesh file's nodes.

    A file domain without ``bounds`` is posed with none, and the solver takes
    them from the mesh (C27 Phase 5, D2): here, from the node coordinates
    the file holds, before anything is built from it. Read for tetgen
    (``<path>.node``) and ymf (the mesh's node array); other formats, and a
    geometry that is not a mesh yet, need ``bounds`` in the spec.
    """
    fmt, path = region["format"], region["path"]
    rows = []
    if fmt == "tetgen":
        with open(path + ".node") as f:
            lines = [l.split("#")[0].split() for l in f]
        lines = [l for l in lines if l]
        count, dim = int(lines[0][0]), int(lines[0][1])
        rows = [[float(v) for v in l[1:1 + dim]] for l in lines[1:1 + count]]
    elif fmt == "ymf":
        prefix = path[:-4] if path.endswith(".ymf") else path
        dim = int(region.get("dimension", 3))
        rows = numpy.asarray(MeshTools.readMeshYmf(prefix).nodeArray)[:, :dim].tolist()
    else:
        raise ValueError("a %s file domain gives its bounds in the spec: Proteus reads the "
                         "bounds only from tetgen and ymf files" % (fmt,))
    if not rows:
        raise ValueError("no nodes in the %s file %s" % (fmt, path))
    nodes = numpy.array(rows)
    return nodes.min(axis=0).tolist(), nodes.max(axis=0).tolist()


def periodic_region(region, h):
    """A 2-D region with every boundary segment cut into pieces of length <= h.

    A periodic region (ymf poses one only when its two faces are
    translations of each other) needs a mesh in which every node on one face
    has its translated partner on the other. triangle with ``Y`` adds no
    boundary points (see :func:`mesh_constraint`), so the boundary nodes are
    exactly the region's vertices; cutting each segment into equal pieces
    gives the two faces translated nodes, and the boundary the resolution
    ``h`` would otherwise have given it. A new vertex takes its segment's flag.
    A 3-D region is returned unchanged: tetgen's ``Y`` keeps the facets'
    own vertices, and the partner check (:func:`periodic_partners`) says if
    that is not enough.
    """
    if int(region.get("dimension", 2)) != 2:
        return region
    import math
    vertices = [list(map(float, v)) for v in region["vertices"]]
    flags = list(region.get("vertex_flags") or [])
    segment_flags = list(region.get("segment_flags") or [])
    segments, new_flags = [], []
    for k, (a, b) in enumerate(region["segments"]):
        pa, pb = vertices[a], vertices[b]
        pieces = max(1, int(math.ceil(math.dist(pa, pb) / float(h) - 1e-12)))
        chain = [a]
        for j in range(1, pieces):
            vertices.append([pa[i] + (pb[i] - pa[i]) * j / pieces for i in range(2)])
            if flags:
                flags.append(segment_flags[k] if segment_flags else flags[a])
            chain.append(len(vertices) - 1)
        chain.append(b)
        for u, v in zip(chain[:-1], chain[1:]):
            segments.append([u, v])
            if segment_flags:
                new_flags.append(segment_flags[k])
    out = dict(region, vertices=vertices, segments=segments)
    if segment_flags:
        out["segment_flags"] = new_flags
    if flags:
        out["vertex_flags"] = flags
    return out


def periodic_partners(nodes, axes, lower, upper, tol):
    """``{axis: unmatched}``: nodes on a periodic face with no partner on the other.

    The key function (:func:`_periodic`) gives a node and its translated
    copy one degree of freedom; a node without a copy is simply not merged,
    and the solution is then not periodic there, silently. So after meshing
    every node on the lower face must have one on the upper face at the
    same position along the other axes (to 1e-8, as the keys are rounded),
    and the other way round.
    """
    nodes = numpy.asarray(nodes)
    out = {}
    for a in axes:
        others = [i for i in range(nodes.shape[1]) if i != a]

        def keys(on):
            return {tuple(round(float(v), 8) for v in p[others]) for p in nodes[on]}
        low = keys(numpy.abs(nodes[:, a] - lower[a]) <= tol)
        high = keys(numpy.abs(nodes[:, a] - upper[a]) <= tol)
        out[a] = len(low ^ high)
    return out


def numerics(problem, spaces, cells, time=None, quadrature_order=None,
             stabilization="none", coefficients=None, levels=1,
             tolerance=None, max_iterations=None, h=None,
             linear_solver=None, petsc_options=None):
    """The discrete problem.

    ``spaces`` maps each scalar component to a (family, order) pair;
    ``cells`` is the number of mesh cells along each axis of a box; a
    piecewise_linear region gives ``h``, its element size, instead (see
    :func:`mesh_constraint`); ``time`` is None
    for a steady problem or a dict with ``dt``. ``stabilization`` is one of
    :data:`STABILIZATIONS`; ``supg/pspg`` installs Proteus's residual-based
    velocity-pressure stabilization (ASGS), which needs the problem in
    pressure-first order (see :func:`reorder`) and its ``coefficients``.
    ``tolerance`` and ``max_iterations`` set Newton's absolute residual
    tolerance (default 1e-10) and iteration limit (default 25).
    ``linear_solver`` and ``petsc_options`` are the spec's; see
    :func:`linear_solver_choice`.
    """
    adr = problem["adr"]
    dim = adr["dim"]
    names = adr["components"]
    n = defaults.Numerics_base()
    n.femSpaces = {}
    orders = []
    for i, name in enumerate(names):
        family, order = spaces[name]
        if (family, order) not in ELEMENTS:
            raise ValueError("no Proteus space for %s%d on simplices" % (family, order))
        n.femSpaces[i] = ELEMENTS[(family, order)]
        orders.append(order)
    q = quadrature_order or min(2 * max(orders) + 2, 5 if dim == 3 else 8)
    n.elementQuadrature = Quadrature.SimplexGaussQuadrature(dim, q)
    n.elementBoundaryQuadrature = Quadrature.SimplexGaussQuadrature(dim - 1, q)
    region = problem["domain"].get("region")
    if region is None:
        if cells is None or h is not None:
            raise ValueError("a box is meshed by cells per side, not h")
        n.nnx = n.nny = cells + 1
        if dim == 3:
            n.nnz = cells + 1
    elif _is_file(region) and region["format"] in CONVERTED_FORMATS:
        raise ValueError("Proteus does not read %s meshes: convert the file to a ymf mesh with "
                         "proteustoolkit/ymf-import-sms and pose format: ymf" % (region["format"],))
    elif _is_discrete(region):
        # a mesh is a domain (D4): read as it is, refined by levels
        if problem.get("periodic"):
            raise ValueError("periodic conditions need a piecewise_linear region; a discrete "
                             "domain cannot pair its faces")
        if h is not None or cells is not None:
            raise ValueError("a discrete domain is already a mesh: give its levels only, not "
                             "cells or h")
    elif _is_file(region) and problem.get("periodic"):
        raise ValueError("periodic conditions need a piecewise_linear region; a %s file domain "
                         "cannot pair its faces" % (region["format"],))
    elif _is_file(region) and region["format"] in MESH_FILE_FORMATS:
        if h is not None or cells is not None:
            raise ValueError("a %s file is already a mesh: give its levels only, not "
                             "cells or h" % (region["format"],))
    elif _is_file(region):
        if h is None or cells is not None:
            raise ValueError("a %s file is meshed by h, its element size"
                             % (region["format"],))
    else:
        if h is None or cells is not None:
            raise ValueError("a piecewise_linear region is meshed by h, its "
                             "element size, not by cells")
        n.triangleOptions = mesh_constraint(float(h), dim,
                                            boundary_points=not problem.get("periodic"))
    # nLevels > 1: the box mesh above is the coarsest, refined uniformly
    # levels-1 times; the solution of record is on the finest.
    n.nLevels = int(levels)
    if any(f == "DG" for f, _ in spaces.values()):
        n.numericalFluxType = NumericalFlux.Advection_DiagonalUpwind_Diffusion_IIPG_exterior \
            if all(f == "CG" for f, _ in spaces.values()) else \
            NumericalFlux.Advection_DiagonalUpwind_Diffusion_IIPG
        # A DG face term couples an element to its face neighbour, so every
        # face neighbour of an owned element has to be on the rank. Node
        # partitioning with one layer of overlap (the default) does not do
        # that; the missing neighbours' faces are assembled as domain
        # boundary and the answer depends on the rank count (1.0e-3 serial,
        # 1.3e-1 on 2 ranks, 16x16 Poisson). An element partition with one
        # layer does.
        n.parallelPartitioningType = MeshTools.MeshParallelPartitioningTypes.element
        n.nLayersOfOverlapForParallel = 1
    stabilization = check_stabilization(problem, stabilization)
    if stabilization == "supg/pspg":
        from proteus import SubgridError
        velocity, pressure = velocity_pressure(problem)
        if names[0] != pressure or names[1:1 + dim] != velocity:
            raise ValueError("SUPG/PSPG needs the pressure first, then the velocity "
                             "(reorder the problem); got %s" % names)
        n.subgridError = SubgridError.NavierStokesASGS_velocity_pressure(
            coefficients, dim, lag=False)
    if time is None:
        n.timeIntegration = TimeIntegration.NoIntegration
    else:
        n.timeIntegration = TimeIntegration.BackwardEuler
        n.stepController = StepControl.FixedStep
        n.DT = float(time["dt"])
    if problem.get("periodic"):
        n.periodicDirichletConditions = None   # set from the physics in run()
        # Serial: each component's own periodic pairs merged, in any space.
        # Parallel: Proteus's parallel periodic numbering, CG1 only, on an
        # element partition without overlap (see parallel_periodic).
        n.parallelPeriodic = parallel_periodic(problem, spaces)
        if n.parallelPeriodic:
            n.parallelPartitioningType = MeshTools.MeshParallelPartitioningTypes.element
            n.nLayersOfOverlapForParallel = 0
    n.multilevelNonlinearSolver = NonlinearSolvers.Newton
    n.levelNonlinearSolver = NonlinearSolvers.Newton
    n.fullNewtonFlag = True
    n.maxNonlinearIts = int(max_iterations or 25)
    n.maxLineSearches = 0
    n.tolFac = 0.0
    n.nl_atol_res = float(tolerance or 1.0e-10)
    n.matrix = LinearAlgebraTools.SparseMatrix
    solver, options = linear_solver_choice(problem, spaces, linear_solver, petsc_options)
    if solver == "superlu":
        n.multilevelLinearSolver = n.levelLinearSolver = LinearSolvers.LU
        return n
    # Through PETSc. The petsc4py solver wants the Dirichlet DOFs kept in
    # the system with their rows replaced, not eliminated; setup() sets
    # that flux (see keeps_dirichlet_dofs).
    n.multilevelLinearSolver = n.levelLinearSolver = LinearSolvers.KSP_petsc4py
    n.linearSmoother = None
    n.linear_solver_options_prefix = "adr_"
    if solver == "mumps":
        # exact, so as tight as the arithmetic allows; with a null space,
        # MUMPS LU (null pivot detection) inside GMRES, which removes it
        n.linTolFac = 0.0
        n.l_atol_res = 1.0e-14
    else:
        # inexact: Proteus's usual pairing, relative 1e-3 and an absolute
        # floor well under Newton's, so Newton still meets its tolerance
        n.linTolFac = 0.001
        n.l_atol_res = 0.001 * n.nl_atol_res
    # data, not a write to PETSc's options database: NS_base sets them
    # while it builds and runs the solvers, then puts the database back
    n.petscOptions = options
    return n


def _name(obj):
    return obj.__name__ if isinstance(obj, type) else type(obj).__name__


def describe(problem, n):
    """The discrete problem ``n`` (from :func:`numerics`), as plain data.

    Names the Proteus classes each choice resolved to, so that a record of
    the run says exactly what was solved, not only what was asked for.
    """
    names = problem["adr"]["components"]
    out = {
        "spaces": {names[i]: _name(space) for i, space in sorted(n.femSpaces.items())},
        "quadrature": {"element": int(n.elementQuadrature.order),
                       "boundary": int(n.elementBoundaryQuadrature.order)},
        "mesh": ({"nodes_per_side": int(n.nnx), "levels": int(n.nLevels)}
                 if problem["domain"].get("region") is None else
                 {"generator_options": n.triangleOptions, "levels": int(n.nLevels)}),
        "time_integration": ("none" if n.timeIntegration is TimeIntegration.NoIntegration
                             else _name(n.timeIntegration)),
        "nonlinear_solver": {"type": _name(n.levelNonlinearSolver),
                             "atol": float(n.nl_atol_res),
                             "max_iterations": int(n.maxNonlinearIts)},
        "linear_solver": _name(n.levelLinearSolver),
    }
    if getattr(n, "petscOptions", None):
        out["petsc_options"] = {n.linear_solver_options_prefix + k: v
                                for k, v in n.petscOptions.items()}
    if problem.get("up_to_constant"):
        out["null_space"] = {"constant": list(problem["up_to_constant"])}
    if getattr(n, "subgridError", None) is not None:
        out["subgrid_error"] = _name(n.subgridError)
    if getattr(n, "numericalFluxType", None) is not None:
        out["numerical_flux"] = _name(n.numericalFluxType)
    if n.timeIntegration is not TimeIntegration.NoIntegration:
        out["time_step"] = float(n.DT)
    return out


def l2_errors(model, problem, t):
    """L2 error of each component with an exact solution, by quadrature.

    For a component the problem determines only up to a constant
    (``problem["up_to_constant"]``), the error modulo constants: its mean
    is removed first.

    On several ranks each sum is over the elements a rank owns (the rest of
    its mesh is other ranks' elements, held for coupling), then over the
    ranks, so every rank returns the error over the whole domain. On one
    rank it is the plain sum, unchanged.
    """
    names = problem["adr"]["components"]
    dim = problem["adr"]["dim"]
    q = model.q
    owned = model.mesh.nElements_owned
    from proteus import Comm
    comm = Comm.comm if Comm.comm is not None and Comm.comm.size() > 1 else None

    def total(values):
        local = numpy.sum(values[:owned])
        if comm is None:
            return local
        from mpi4py import MPI
        return comm.comm.tompi4py().allreduce(float(local), op=MPI.SUM)

    errors = {}
    for name, code in problem.get("exact", {}).items():
        i = names.index(name)
        exact = _Field(code, dim).uOfXT(q['x'], t)
        error = q[('u', i)] - exact
        if name in (problem.get("up_to_constant") or []):
            # modulo constants: the problem does not determine one
            error = error - total(error * q[('dV_u', i)]) / total(q[('dV_u', i)])
        errors[name] = float(numpy.sqrt(total(error ** 2 * q[('dV_u', i)])))
    return errors


def with_kept_dof_diagonals(problem):
    """The problem, with a zero self-coupling where a kept Dirichlet DOF needs one.

    When Dirichlet DOFs stay in the system (:func:`keeps_dirichlet_dofs`),
    Proteus replaces each one's row by u = g, which needs the row's
    diagonal entry to exist in the sparsity pattern. An equation that does
    not involve its own unknown has none: the incompressibility equation
    of a Taylor-Hood velocity-pressure system, say, once a pressure datum
    is pinned. So such an equation gets a reaction term that is zero, and
    has zero derivative, but is declared to depend on its own unknown --
    a structural diagonal and nothing else. Equations that need none are
    left alone, so a problem that already ran solves bitwise as before.
    """
    import copy
    adr = problem["adr"]
    pinned = {e["component"] for e in problem.get("dirichlet") or []}

    def involves_itself(eq, name):
        if name in eq.get("diffusion", {}):
            return True
        return any(name in eq[term].get("depends_on", {})
                   for term in ("mass", "advection", "reaction", "hamiltonian") if term in eq)

    missing = [k for k, eq in enumerate(adr["equations"])
               if eq["component"] in pinned and not involves_itself(eq, eq["component"])]
    if not missing:
        return problem
    problem = copy.deepcopy(problem)
    for k in missing:
        eq = problem["adr"]["equations"][k]
        name = eq["component"]
        reaction = eq.setdefault("reaction", {"r": "0.0", "dr": {}, "depends_on": {}})
        reaction["dr"][name] = "0.0"
        reaction["depends_on"][name] = "linear"
    return problem


def _refuse_positions_on_untagged_walls(problem):
    """Refuse Dirichlet conditions found by position on a ymf mesh with untagged exterior walls.

    A facet list that leaves exterior walls untagged gives them flag 0, and Proteus passes flag 0 for the
    interior too, so getDBC cannot tell those walls from the interior: a condition selected by ``where``
    would silently not be imposed there, and the wall would get zero flux. Until the flags carry
    exteriority, such a condition has to select its walls by their tags.
    """
    region = problem["domain"].get("region")
    if _is_discrete(region):
        source, label = region, "the discrete domain"
    elif _is_file(region) and region.get("format") == "ymf":
        path = region["path"]
        source, label = (path[:-4] if path.endswith(".ymf") else path), region["path"]
    else:
        return
    by_position = sorted(set(e["region"] for e in problem.get("dirichlet") or [] if "flags" not in e))
    if not by_position:
        return
    untagged = MeshTools.ymfUntaggedExteriorFacets(source)
    if untagged:
        raise ValueError(
            "Dirichlet regions %s are found by position, but %s leaves %d exterior facets untagged (flag 0), "
            "which Proteus cannot tell from the interior: the condition would not be imposed on them. Select "
            "those walls by their tags (boundary_regions: tags), or tag every exterior facet in the mesh"
            % (", ".join(by_position), label, untagged))


def setup(problem, spaces, cells, time=None, name="adr", stabilization="none",
          levels=1, tolerance=None, max_iterations=None, h=None,
          linear_solver=None, petsc_options=None, mesh=None):
    """The problem as solved, its physics and numerics, and the system settings.

    Returns ``(problem, p, n, system)``: the problem (reordered pressure
    first for SUPG/PSPG), the Physics_base and Numerics_base, and a dict of
    the System_base settings (``tnList``, and for a time-dependent solve
    ``dt_system_fixed``). :func:`run` solves this; :func:`emit_pn` writes it
    out as p, n and so files.
    """
    if (stabilization or "none").lower() == "supg/pspg":
        velocity, pressure = velocity_pressure(problem)
        problem = reorder(problem, [pressure] + velocity)
    solver, _ = linear_solver_choice(problem, spaces, linear_solver, petsc_options)
    if solver != "superlu" and not any(f == "DG" for f, _ in spaces.values()):
        # Dirichlet DOFs will be kept (see keeps_dirichlet_dofs)
        problem = with_kept_dof_diagonals(problem)
    region = problem["domain"].get("region")
    if _is_file(region) and problem["domain"].get("lower") is None:
        # posing follows meshing for this kind alone: the bounds come from
        # the mesh file, before the tolerance and anything else needs them
        lower, upper = file_bounds(region)
        problem = dict(problem, domain=dict(problem["domain"], lower=lower, upper=upper))
    elif _is_discrete(region) and problem["domain"].get("lower") is None:
        # nodes in an HDF5 file: ymf posed it without bounds, the mesh gives them
        lower, upper = MeshTools.ymfMeshBounds(region, _discrete_dimension(region))
        problem = dict(problem, domain=dict(problem["domain"], lower=lower, upper=upper))
    _refuse_positions_on_untagged_walls(problem)
    if problem.get("periodic") and problem["domain"].get("region") is not None and h \
            and not _is_file(problem["domain"]["region"]) and not _is_discrete(problem["domain"]["region"]):
        # no boundary points from the mesher, so cut the faces to h here
        problem = dict(problem, domain=dict(
            problem["domain"], region=periodic_region(problem["domain"]["region"], h)))
    p = physics(problem, name)
    n = numerics(problem, spaces, cells, time, stabilization=stabilization,
                 coefficients=p.coefficients, levels=levels,
                 tolerance=tolerance, max_iterations=max_iterations, h=h,
                 linear_solver=linear_solver, petsc_options=petsc_options)
    if p.domain is not None:
        # the region's own mesh options: generated by triangle or tetgen
        # from <name>.poly (see write_poly), then refined levels-1 times;
        # or, given ``mesh``, read from the files with that prefix, which a
        # caller made beforehand (proteus.cli.ymfmesh), and refined the same
        options = MeshTools.getMeshOptions(p.domain)
        options.triangleOptions = n.triangleOptions
        options.nLevels = n.nLevels
        # partitioned as the solvers will assume (see parallel_periodic)
        options.parallelPartitioningType = n.parallelPartitioningType
        options.nLayersOfOverlapForParallel = n.nLayersOfOverlapForParallel
        options.genMesh = mesh is None
        p.domain.polyfile = name if mesh is None else mesh
        if _is_file(region):
            # read from (or, for a geometry, made beside) the file itself
            path = os.path.abspath(region["path"])
            p.domain.polyfile = path[:-4] if path.endswith(".geo") else path
            if region["format"] not in MESH_FILE_FORMATS:
                options.he = float(h)
    if problem.get("periodic"):
        n.periodicDirichletConditions = p.periodicDirichletConditions
    keep = keeps_dirichlet_dofs(n, spaces)
    if keep:
        n.numericalFluxType = NumericalFlux.StrongDirichletFactory(p.fluxBoundaryConditions,
                                                                   keepDOFs=True)
    if p.robinConditions:
        # alpha*u + beta*(q.n) = g needs the trace in the flux, so it is the
        # numerical flux's job rather than a boundary condition's.
        # Built on the same strongly-Dirichlet base the rest of this module
        # uses, so the Dirichlet faces keep being eliminated and the outflow
        # advective flux keeps being computed. Any numerical flux suppresses
        # the one Transport would compute by default.
        n.numericalFluxType = NumericalFlux.RobinFactory(
            p.robinConditions,
            base=NumericalFlux.StrongDirichletFactory(
                p.fluxBoundaryConditions,
                keepDOFs=keep))
    if time is None:
        system = {"tnList": [0.0, 1.0]}
    else:
        if not problem.get("time"):
            raise ValueError("a time step was given for a problem with no time interval")
        t0, t1 = (float(v) for v in problem["time"])
        outputs = int(time.get("outputs", 1))
        p.T = t1
        # The system-level controller sets each step's size; the model's
        # own DT alone does not, and the run would take one step per output.
        system = {"tnList": [t0 + (t1 - t0) * k / outputs for k in range(outputs + 1)],
                  "dt_system_fixed": float(time["dt"])}
    return problem, p, n, system


def run(problem, spaces, cells, time=None, name="adr", stabilization="none",
        opts=None, levels=1, tolerance=None, max_iterations=None, h=None,
        linear_solver=None, petsc_options=None, mesh=None):
    """Solve; return (NS_base, the finest level model, L2 errors at the end).

    ``time`` is None for a steady solve, or ``{"dt": ...}`` to step through
    the problem's own time interval (``problem["time"]``); ``outputs`` (1 by
    default) sets how many evenly spaced archive frames to write. The
    archive Proteus writes holds the numerical approximation only; what
    produced it is the caller's to record (see :func:`describe`).

    A region domain is meshed by triangle or tetgen on rank 0 as part of the
    solve, unless ``mesh`` names the prefix of mesh files already made for
    it (``<mesh>.ele``, ``.node``...), which are read instead.
    """
    from proteus import NumericalSolution, SplitOperator, default_s
    from proteus import runtime
    if opts is None:
        opts = runtime.RunOptions()
    runtime.start(log=None)
    problem, p, n, system = setup(problem, spaces, cells, time, name, stabilization,
                                  levels, tolerance, max_iterations, h,
                                  linear_solver, petsc_options, mesh)
    if mesh is None:
        write_poly(p)
    so = defaults.System_base(name=name, pnList=[(p, n)], sList=[default_s])
    so.tnList = system["tnList"]
    if "dt_system_fixed" in system:
        so.systemStepControllerType = SplitOperator.Sequential_FixedStep
        so.dt_system_fixed = system["dt_system_fixed"]
    ns = NumericalSolution.NS_base(so, [p], [n], so.sList, opts)
    if problem.get("periodic") and problem["domain"].get("region") is not None:
        # a node without its partner is simply not merged, silently
        lower, upper = problem["domain"]["lower"], problem["domain"]["upper"]
        axes = sorted({a for e in problem["periodic"] for a in e["axes"]})
        tol = 1e-8 * max(u - l for u, l in zip(upper, lower))
        mesh = ns.mlMesh_nList[0].meshList[-1]
        nodes = mesh.nodeArray[:mesh.nNodes_global, :len(lower)]
        unmatched = {a: k for a, k in periodic_partners(nodes, axes, lower, upper, tol).items()
                     if k}
        if unmatched:
            raise ValueError(
                "periodic region: %s node%s on the periodic faces %s no partner on the "
                "opposite face (%s); the mesh does not pair the faces"
                % (sum(unmatched.values()), "" if sum(unmatched.values()) == 1 else "s",
                   "has" if sum(unmatched.values()) == 1 else "have",
                   ", ".join("axis %s: %d" % ("xyz"[a], k) for a, k in unmatched.items())))
    ns.calculateSolution(name)
    model = ns.modelList[0].levelModelList[-1]
    return ns, model, l2_errors(model, problem, so.tnList[-1])


def write_poly(p):
    """Write a region domain's <polyfile>.poly, the mesh generator's input,
    on rank 0 (a box needs none, nor does a file domain: it is read)."""
    if p.domain is None or not hasattr(p.domain, "writePoly") or getattr(
            p.domain, "meshfile", None) is not None or hasattr(p.domain, "geofile"):
        return
    from proteus import Comm
    comm = Comm.get()
    if comm.isMaster():
        p.domain.writePoly(p.domain.polyfile)
    comm.barrier()


# --- writing a problem out as p, n and so files ---------------------------------


def _supplied_value(entry, dim):
    """The ``return`` lines for one condition, written out: formula or table."""
    if entry.get("series") is not None:
        # linear between samples, end values held outside the range
        return ["        T = %r" % ([t for t, _ in entry["series"]],),
                "        Q = %r" % ([v for _, v in entry["series"]],),
                "        return lambda X, t: float(numpy.interp(t, T, Q))"]
    return ["        return lambda X, t: Field(%s).uOfXT(X, t)"
            % _function(dim, entry["value"])]


def _spelled(module, obj):
    """``Module.Name`` for a class, preferring an alias to a ``*_base`` name
    (StepControl.FixedStep, not StepControl.SC_base)."""
    names = sorted(k for k, v in vars(module).items() if v is obj)
    better = [k for k in names if not k.endswith("_base")]
    return "%s.%s" % (module.__name__.rsplit(".", 1)[-1], (better or names or [obj.__name__])[0])


def _coordinates(dim):
    return ["x", "y", "z"][:dim]


def _function(dim, code):
    """A lambda of the coordinates and t, around one code string."""
    return "lambda %s, t: %s" % (", ".join(_coordinates(dim)), code)


def emit_pn(problem, spaces, cells, time=None, name="adr", stabilization="none",
            levels=1, tolerance=None, max_iterations=None, directory=".", header=None,
            h=None, linear_solver=None, petsc_options=None):
    """Write the problem as a classic Proteus ``<name>_p.py``, ``_n.py`` and ``_so.py``.

    The files say in Python exactly what :func:`run` builds in memory: the
    settings are read from the objects :func:`setup` returns, and the
    coefficients from the same code strings, so ``parun <name>_so.py``
    solves bitwise the same problem. They are written to be read: one
    function per boundary condition and region, the coefficients one term
    per line. The system's name is ``<name>_pn``, so a run of the files
    does not overwrite the archive of the run they describe.

    ``header`` is a line or two of text for each file's docstring (say, the
    output it reproduces). Returns the three paths.
    """
    import os
    import pprint
    from proteus import SubgridError
    problem, p, n, system = setup(problem, spaces, cells, time, name, stabilization,
                                  levels, tolerance, max_iterations, h,
                                  linear_solver, petsc_options)
    keep = keeps_dirichlet_dofs(n, spaces)
    c = p.coefficients
    adr = problem["adr"]
    dim = adr["dim"]
    names = list(adr["components"])
    xyz = _coordinates(dim)
    lower, upper = list(problem["domain"]["lower"]), list(problem["domain"]["upper"])
    tol = 1e-8 * max(p.L[:dim])
    about = [header] if header else []

    def doc(what):
        return ['"""%s for %s.' % (what, name), ""] + about + [
            "Written by proteus.ADRProblem.emit_pn; run with parun %s_so.py." % name,
            "For bitwise the same result, run it as the original was: one thread,",
            "OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 parun ...",
            "Components: %s." % ", ".join("%d = %s" % (i, nm) for i, nm in enumerate(names)),
            '"""']

    # ---- p ----
    P = doc("Physics") + [
        "import numpy",
        "from proteus.default_p import *",
        "from proteus.TransportCoefficients import TC_base",
        "",
        "nd = %d" % dim,
        "name = %r" % name,
    ]
    region = problem["domain"].get("region")
    if region is None:
        P += ["# the old-style box: domain None, with L and x0, hands nnx/nny/nnz to the mesher",
              "domain = None"]
    elif _is_discrete(region):
        # a mesh held in the document, read in memory as setup builds it
        options = MeshTools.getMeshOptions(p.domain)
        P += ["# a discrete domain: the mesh the document holds, read in memory",
              "from proteus import DomainYMF, MeshTools",
              "domain = DomainYMF.from_ymf(%s)" % pprint.pformat(region, width=72).replace(
                  "\n", "\n" + " " * len("domain = DomainYMF.from_ymf(")),
              "domain.name = name",
              "meshOptions = MeshTools.getMeshOptions(domain)",
              "meshOptions.nLevels = %d" % options.nLevels,
              "meshOptions.genMesh = %r" % options.genMesh,
              "# the bounding box: the scale of the boundary tolerance"]
    elif _is_file(region):
        # as setup builds it: the file's absolute path, read (or, for a geometry, meshed by he) where it lies;
        # nothing is written beside it
        options = MeshTools.getMeshOptions(p.domain)
        absolute = dict(region, path=os.path.abspath(region["path"]))
        P += ["# a %s file domain, %s from %s" % (
                  region["format"], "read" if region["format"] in MESH_FILE_FORMATS else "meshed",
                  absolute["path"]),
              "from proteus import DomainYMF, MeshTools",
              "domain = DomainYMF.from_ymf(%s)" % pprint.pformat(absolute, width=72).replace(
                  "\n", "\n" + " " * len("domain = DomainYMF.from_ymf(")),
              "domain.name = name",
              "domain.polyfile = %r" % p.domain.polyfile,
              "meshOptions = MeshTools.getMeshOptions(domain)",
              "meshOptions.nLevels = %d" % options.nLevels,
              "meshOptions.genMesh = %r" % options.genMesh]
        if region["format"] not in MESH_FILE_FORMATS:
            P.append("meshOptions.he = %r" % options.he)
        P.append("# the bounding box: the scale of the boundary tolerance")
    else:
        options = MeshTools.getMeshOptions(p.domain)
        P += ["# a piecewise_linear region: %s meshes it from %s.poly, written here"
              % ("triangle" if dim == 2 else "tetgen", p.domain.polyfile),
              "from proteus import Comm, DomainYMF, MeshTools",
              "domain = DomainYMF.from_ymf(%s)" % pprint.pformat(region, width=72).replace(
                  "\n", "\n" + " " * len("domain = DomainYMF.from_ymf(")),
              "domain.name = name",
              "domain.polyfile = %r" % p.domain.polyfile,
              "meshOptions = MeshTools.getMeshOptions(domain)",
              "meshOptions.triangleOptions = %r" % options.triangleOptions,
              "meshOptions.nLevels = %d" % options.nLevels,
              "meshOptions.genMesh = True",
              "if Comm.get().isMaster():",
              "    domain.writePoly(domain.polyfile)",
              "Comm.get().barrier()",
              "# the bounding box: the scale of the boundary tolerance"]
    P += ["L = %r" % (tuple(p.L),),
          "x0 = %r" % (tuple(p.x0),)]
    if time is not None:
        P.append("T = %r" % p.T)
    P += ["", "",
          "class Coefficients(TC_base):",
          '    """dm_i/dt + div(f_i - sum_k a_ik grad(phi_k)) + r_i + H_i = 0"""',
          "",
          "    def __init__(self):",
          "        TC_base.__init__(",
          "            self, nc=%d, variableNames=%r," % (c.nc, list(c.variableNames))]
    for key in ("mass", "advection", "diffusion", "potential", "reaction", "hamiltonian"):
        P.append("            %s=%r," % (key, getattr(c, key)))
    P.append("            sparseDiffusionTensors={%s})" % ", ".join(
        "%r: (numpy.array(%r, 'i'), numpy.array(%r, 'i'))" % (k, list(map(int, r)), list(map(int, cl)))
        for k, (r, cl) in c.sdInfo.items()))
    if getattr(c, "nullSpaceComponents", None):
        P += ["        # %s: determined only up to a constant, which the linear solver"
              % ", ".join(names[k] for k in c.nullSpaceComponents),
              "        # keeps in the null space of the operator",
              "        self.nullSpace = %r" % c.nullSpace,
              "        self.nullSpaceComponents = %r" % list(c.nullSpaceComponents)]
    if getattr(c, "vectorComponents", None) is not None:
        P += ["        self.vectorComponents = %r" % list(c.vectorComponents),
              "        self.vectorName = %r" % c.vectorName]
    P += ["", "    def evaluate(self, t, c):"]
    P += ["        %s = c['x'][..., %d]" % (v, k) for k, v in enumerate(xyz)]
    for j, nm in enumerate(names):
        P.append("        %s = c[('u', %d)]" % (nm, j))
    if c._needs_gradients:
        for j, nm in enumerate(names):
            P.append("        if ('grad(u)', %d) in c:" % j)
            P += ["            grad_%s_%d = c[('grad(u)', %d)][..., %d]" % (nm, k, j, k)
                  for k in range(dim)]
    for key, _, last, code in c._terms:
        target = "c[%r][...]" % (key,) if last is None else "c[%r][..., %d]" % (key, last)
        P.append("        if %r in c: %s = %s" % (key, target, code))
    P += ["", "", "coefficients = Coefficients()", "", "",
          "class Field(object):",
          '    """A function of the coordinates and t, as initial or boundary data."""',
          "",
          "    def __init__(self, f):",
          "        self.f = f",
          "",
          "    def uOfXT(self, X, t):",
          "        X = numpy.asarray(X)",
          "        value = self.f(%s, t)" % ", ".join("X[..., %d]" % k for k in range(dim)),
          "        return value + 0.0 * X[..., 0] if numpy.ndim(X) > 1 else value",
          "",
          "    def uOfX(self, X):",
          "        return self.uOfXT(X, 0.0)",
          ""]
    exact = problem.get("exact", {})
    initial = problem.get("initial", {})
    P += ["", "analyticalSolution = {"]
    P += ["    %d: Field(%s),   # %s" % (i, _function(dim, exact[nm]), nm)
          for i, nm in enumerate(names) if nm in exact]
    P += ["}", "# a steady problem with no initial condition starts Newton from zero,",
          "# never from the exact solution", "initialConditions = {"]
    P += ["    %d: Field(%s),   # %s" % (i, _function(dim, initial.get(nm, "0.0")), nm)
          for i, nm in enumerate(names)]
    P += ["}", "", "tol = %r" % tol]

    by_component = {}
    for e in problem["dirichlet"]:
        by_component.setdefault(e["component"], []).append(e)
    for i, nm in enumerate(names):
        entries = by_component.get(nm, [])
        P += ["", "",
              "def getDBC_%s(X, flag):" % nm,
              "    if flag == 0:     # an interior node",
              "        return None"]
        if entries:
            P.append("    %s = %s" % (", ".join(xyz), ", ".join("X[%d]" % k for k in range(dim))))
        for e in entries:
            test = _test_text(e)
            P.append("    if %s:   # region %s" % (test, e["region"]))
            P += _supplied_value(e, dim)
        P.append("    return None")
    P += ["", "", "dirichletConditions = {%s}" % ", ".join(
        "%d: getDBC_%s" % (i, nm) for i, nm in enumerate(names))]

    periodic = {e["component"]: list(e["axes"]) for e in problem.get("periodic", [])}
    if periodic:
        P += ["", "",
              "# Periodic: a node on either face of a periodic axis is keyed by its",
              "# coordinates with that axis snapped to the lower face; equal keys share",
              "# a degree of freedom. A node a Dirichlet condition claims is not periodic.",
              "LOWER = %r" % (lower,),
              "UPPER = %r" % (upper,),
              "",
              "",
              "def on_faces(X, axes):",
              "    return [a for a in axes",
              "            if abs(X[a] - LOWER[a]) <= tol or abs(X[a] - UPPER[a]) <= tol]",
              "",
              "",
              "def periodic(getDBC, axes):",
              "    def getPDBC(X, flag):",
              "        if getDBC(X, flag if flag else 1) is not None:",
              "            return None",
              "        on = on_faces(X, axes)",
              "        if not on:",
              "            return None",
              "        key = numpy.array([round(float(v), 8) for v in X[:3]])",
              "        for a in on:",
              "            key[a] = LOWER[a]",
              "        return key",
              "    return getPDBC",
              "",
              "",
              "def zero_on_faces(axes):",
              "    # the two faces' fluxes cancel, so each is set to zero",
              "    def getAFBC(X, flag):",
              "        if on_faces(X, axes):",
              "            return lambda X, t: 0.0",
              "        return None",
              "    return getAFBC",
              "",
              "",
              "periodicDirichletConditions = {%s}" % ", ".join(
                  "%d: periodic(getDBC_%s, %r)" % (i, nm, periodic[nm]) if nm in periodic
                  else "%d: lambda X, flag: None" % i for i, nm in enumerate(names)),
              "advectiveFluxBoundaryConditions = {%s}" % ", ".join(
                  "%d: zero_on_faces(%r)" % (i, periodic[nm]) if nm in periodic
                  else "%d: lambda X, flag: None" % i for i, nm in enumerate(names))]
    else:
        P += ["advectiveFluxBoundaryConditions = {%s}" % ", ".join(
            "%d: lambda X, flag: None" % i for i in range(len(names)))]
    targets = _neumann_targets(problem, adr, names)
    for i, (ck, entries) in sorted(targets.items()):
        P += ["", "",
              "# the OUTWARD diffusive flux q.n, with q = -a grad %s" % names[i],
              "def getDFBC_%s(X, flag):" % names[i],
              "    %s = %s" % (", ".join(xyz), ", ".join("X[%d]" % k for k in range(dim)))]
        for e in entries:
            test = _test_text(e)
            P.append("    if %s:   # region %s" % (test, e["region"]))
            P += _supplied_value(e, dim)
        P += ["    # elsewhere the natural condition: zero diffusive flux",
              "    return None"]
    if targets:
        P.append("")
    P += ["diffusiveFluxBoundaryConditions = {%s}" % ", ".join(
              "%d: {%s}" % (i, ", ".join(
                  ("%d: getDFBC_%s" % (k, names[i])) if targets.get(i, (None,))[0] == k
                  else "%d: lambda X, flag: None" % k for k in sorted(d)))
              for i, d in sorted(p.diffusiveFluxBoundaryConditions.items())),
          "# advective fluxes through the boundary come from the trace",
          "fluxBoundaryConditions = %r" % (p.fluxBoundaryConditions,)]

    # ---- n ----
    from proteus import default_n
    N = doc("Numerics") + [
        "from proteus.default_n import *",
        "from proteus import (FemTools, LinearAlgebraTools, LinearSolvers, NonlinearSolvers,",
        "                     NumericalFlux, Quadrature, StepControl, SubgridError, TimeIntegration)",
        "from %s_p import %s" % (name, ", ".join(["coefficients", "nd"] + (
            ["periodicDirichletConditions"] if periodic else []) + (
            ["fluxBoundaryConditions"] if keep else []))),
        "",
        "femSpaces = {"]
    N += ["    %d: %s,   # %s" % (i, _spelled(FemTools, n.femSpaces[i]), names[i])
          for i in sorted(n.femSpaces)]
    N += ["}",
          "elementQuadrature = Quadrature.SimplexGaussQuadrature(nd, %d)" % n.elementQuadrature.order,
          "elementBoundaryQuadrature = Quadrature.SimplexGaussQuadrature(nd - 1, %d)"
          % n.elementBoundaryQuadrature.order]
    if region is None:
        N.append("%s = %d   # %d cells per side" % (" = ".join(["nnx", "nny", "nnz"][:dim]), n.nnx, cells))
    else:
        N.append("triangleOptions = %r   # h = %r; the p file's meshOptions carry it"
                 % (n.triangleOptions, h))
    N.append("nLevels = %d   # uniform refinements of that mesh, plus one" % n.nLevels)
    if keep:
        N += ["# strong Dirichlet conditions with the DOFs kept in the system (rows",
              "# replaced by u = g), the layout KSP_petsc4py expects",
              "numericalFluxType = NumericalFlux.StrongDirichletFactory(fluxBoundaryConditions,",
              "                                                         keepDOFs=True)"]
    elif n.numericalFluxType is not default_n.numericalFluxType:
        N.append("numericalFluxType = %s" % _spelled(NumericalFlux, n.numericalFluxType))
    if n.subgridError is not None:
        if not isinstance(n.subgridError, SubgridError.NavierStokesASGS_velocity_pressure):
            raise ValueError("emit_pn does not know how to write %s" % type(n.subgridError).__name__)
        N.append("subgridError = SubgridError.NavierStokesASGS_velocity_pressure("
                 "coefficients, nd, lag=False)")
    if n.timeIntegration is TimeIntegration.NoIntegration:
        N.append("timeIntegration = TimeIntegration.NoIntegration   # steady")
    else:
        N += ["timeIntegration = %s" % _spelled(TimeIntegration, n.timeIntegration),
              "stepController = %s" % _spelled(StepControl, n.stepController),
              "DT = %r" % n.DT]
    if periodic:
        N += ["# the serial periodic path: the parallel one numbers every component's",
              "# periodic DOFs with component 0's space, which breaks mixed spaces",
              "parallelPeriodic = False"]
    N += ["multilevelNonlinearSolver = %s" % _spelled(NonlinearSolvers, n.multilevelNonlinearSolver),
          "levelNonlinearSolver = %s" % _spelled(NonlinearSolvers, n.levelNonlinearSolver),
          "fullNewtonFlag = %r" % n.fullNewtonFlag,
          "maxNonlinearIts = %d" % n.maxNonlinearIts,
          "maxLineSearches = %d" % n.maxLineSearches,
          "tolFac = %r" % n.tolFac,
          "nl_atol_res = %r" % n.nl_atol_res,
          "matrix = %s" % _spelled(LinearAlgebraTools, n.matrix),
          "multilevelLinearSolver = %s" % _spelled(LinearSolvers, n.multilevelLinearSolver),
          "levelLinearSolver = %s" % _spelled(LinearSolvers, n.levelLinearSolver)]
    if n.levelLinearSolver is LinearSolvers.KSP_petsc4py:
        N += (["# a singular operator: MUMPS LU (null pivot detection) inside GMRES,",
               "# which removes the null space the coefficients declare"]
              if problem.get("up_to_constant") else
              ["# solved through PETSc (the spec's linear_solver)"])
        N += ["linearSmoother = None",
              "linear_solver_options_prefix = %r" % n.linear_solver_options_prefix,
              "linTolFac = %r" % n.linTolFac,
              "l_atol_res = %r" % n.l_atol_res,
              "# options for this model's solvers, relative to the prefix; NS_base sets",
              "# them while it builds and runs the solvers, then restores the database",
              "petscOptions = {"]
        N += ["    %r: %r," % (k, v) for k, v in n.petscOptions.items()]
        N += ["}"]

    # ---- so ----
    S = doc("The system") + [
        "from proteus.default_so import *",
        "from proteus import default_s",
        "",
        "name = %r   # not %r: a run of these files keeps clear of the original's archive"
        % (name + "_pn", name),
        "pnList = [(%r, %r)]" % (name + "_p", name + "_n"),
        "sList = [default_s]",
        "tnList = %s" % pprint.pformat(system["tnList"])]
    if "dt_system_fixed" in system:
        S += ["# the system's controller sets each step; the model's DT alone does not",
              "systemStepControllerType = Sequential_FixedStep",
              "dt_system_fixed = %r" % system["dt_system_fixed"]]

    paths = []
    for suffix, lines in (("_p", P), ("_n", N), ("_so", S)):
        path = os.path.join(directory, name + suffix + ".py")
        with open(path, "w") as f:
            f.write("\n".join(lines) + "\n")
        paths.append(path)
    return paths
