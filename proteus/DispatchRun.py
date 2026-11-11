"""Run a split discretization from its dispatch, through the decks' own machinery (C5 step 2).

:func:`proteus.Dispatch.dispatch` maps each sub-problem of a split discretization onto an mprans module and its
flags. This module builds the run from that result and from ymf's plain data of the split
(:func:`ymf.symbolic.split.split_problem`: the domain, the time interval, initial data and the closed boundary
conditions), through the machinery a deck uses, not a parallel path (YMF_MODEL_FAMILIES_C42 section 5.2):

* two-phase level-set flow: a ``TwoPhaseFlowProblem`` (:func:`two_phase`), its models added in the dispatched
  order with TwoPhaseFlow's Parameters classes, its domain a SpatialTools tank with ``BC_RANS`` setters;
* shallow water: an ``SWFlowProblem`` (:func:`shallow_water`), its p and n modules SWFlow's own (``sw_p``,
  ``sw_n``), loaded as ``parun --SWEs`` loads them.

After the machinery has built the p and n modules, :func:`check_built` compares every attribute the dispatch gives
with what was built; a difference is refused (:class:`RealizeError`) before anything is solved, so the run is the
dispatched discretization or nothing. What the dispatch does not cover is refused by
:class:`proteus.Dispatch.DispatchError` before this module is reached.

This module is the front end's: it reads the model's coefficients through ymf.symbolic (sympy), as ymf_run does.
"""
import contextlib
import os

import numpy

from proteus import DispatchTable as table
from proteus.Dispatch import DispatchError, resolve
from ymf import vocabulary as vocab

__all__ = ["RealizeError", "Realized", "realize", "precheck", "tank_sides", "two_phase", "shallow_water",
           "check_built", "run",
           "petsc_options_restored"]


class RealizeError(DispatchError):
    """A spec the dispatch covers that the deck machinery cannot build as dispatched."""


class Realized:
    """A built run: the system (``so``), its p and n modules, the problem object the deck machinery made, and the
    records of what was taken from where (``record``)."""

    def __init__(self, so, pList, nList, problem, record):
        self.so = so
        self.pList = pList
        self.nList = nList
        self.problem = problem
        self.record = record


def realize(doc, found, split, realization, name, outdir, ranks=1):
    """Build the run of a dispatched discretization (``found``) on one realization (``{"h": ...}`` or
    ``{"cells": ...}``), with ymf's plain data of the split (``split``). Files (the PSLG) go to ``outdir``."""
    if found.family not in table.REALIZERS:
        raise RealizeError("%s dispatches (family %s), and nothing realizes that family yet; the table realizes %s"
                           % (found.discretization, found.family, ", ".join(sorted(table.REALIZERS))))
    builder = resolve(table.REALIZERS[found.family])
    return builder(doc, found, split, realization, name, outdir, ranks)


# ---------------------------------------------------------------------------------------------------------------
# Shared pieces
# ---------------------------------------------------------------------------------------------------------------

def _entry(doc, name):
    return next(d for d in doc["solution_paths"]["discretizations"] if d["name"] == name)


def _field(code, dim):
    from proteus.ADRProblem import _Field
    return _Field(code, dim)


class _Constant(object):
    def __init__(self, value):
        self.value = float(value)

    def uOfXT(self, x, t):
        return self.value


class _Regularized(object):
    """``f(width, φ0(x))``: a regularized function of another field's initial data, with Proteus's own ``f``."""

    def __init__(self, function, width, inner):
        self.function, self.width, self.inner = function, width, inner

    def uOfXT(self, x, t):
        return self.function(self.width, self.inner.uOfXT(x, t))


def _time(split, entry):
    """(final time, number of outputs) of a transient split; the outputs are ``discretization.outputs`` (1 by
    default), evenly spaced, as ymf_run's ADR path reads them."""
    if not split.get("time"):
        raise RealizeError("%s: the model's domain gives no time interval (', t ∈ (t0, t1]')" % entry["name"])
    t0, t1 = (float(t) for t in split["time"])
    if t0 != 0.0:
        raise RealizeError("%s: the deck machinery starts at t = 0, and the domain's interval starts at %r"
                           % (entry["name"], t0))
    block = entry.get("discretization") or {}
    outputs = int(block.get("outputs", 1)) if isinstance(block, dict) else 1
    return t1, outputs


def _linear_solver(entry, ranks):
    """``superlu``: a direct solve, TwoPhaseFlow's useSuperlu (SuperLU on one rank, SuperLU_DIST through PETSc on
    several); ``petsc``: the machinery's iterative solvers and their options per model prefix."""
    solver = (entry.get("solver") or {}).get("linear_solver")
    if solver not in (None, "superlu", "petsc"):
        raise RealizeError("%s: linear_solver %r; the deck machinery takes superlu or petsc"
                           % (entry["name"], solver))
    solver = solver or "petsc"
    if solver == "superlu":
        return solver, "SuperLU" if ranks == 1 else "SuperLU_DIST"
    return solver, "PETSc KSP"


def _where(condition, dim, tol):
    from proteus.ADRProblem import _GLOBALS, _compile
    if condition.get("where") is None:
        return lambda x: True
    code = _compile(condition["where"])

    def test(x):
        ns = {"x": x[0], "tol": tol}
        if dim > 1:
            ns["y"] = x[1]
        if dim > 2:
            ns["z"] = x[2]
        return bool(eval(code, _GLOBALS, ns))
    return test


def _number(value):
    if isinstance(value, dict):
        return tuple(_number(value[k]) for k in sorted(value))
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


def _signature(conditions):
    """{(variable, type, component, only, value)} of one region's closed conditions."""
    return {(c["variable"], c["type"], c.get("component"), c.get("only"), _number(c.get("value")))
            for c in conditions}


def _matches(pattern, have):
    if len(pattern) != len(have):
        return False
    for p in pattern:
        hit = False
        for h in have:
            if p[:4] != h[:4]:
                continue
            if p[4] == "any" or p[4] == h[4] or (
                    p[4] == "zero_vector" and isinstance(h[4], tuple) and all(v == 0.0 for v in h[4])):
                hit = True
                break
        if not hit:
            return False
    return True


# ---------------------------------------------------------------------------------------------------------------
# Two-phase level-set flow: TwoPhaseFlowProblem
# ---------------------------------------------------------------------------------------------------------------

def _by_role(doc, role):
    """The name of the model's coefficient with ``role`` (``ymf.vocabulary.coefficient_by_role``); refused, naming the
    role and how to state it, when the model has none."""
    name = vocab.coefficient_by_role(doc, role)
    if name is None:
        raise RealizeError("the realizer needs the model's %s, and no coefficient states role: %s (for example "
                           "``%s: {formula: ..., role: %s}``)" % (role.replace("_", " "), role, "ρ", role))
    return name


def _two_phase_values(doc, dim):
    """SystemPhysics' rho_0, nu_0, rho_1, nu_1 and gravity from the model's coefficients: the density and dynamic
    viscosity at phase indicator 0 and 1, and nu = mu / rho, simplified symbolically before any number enters, so
    that a viscosity written rho_w nu_w gives nu_w itself."""
    import sympy
    from proteus.Dispatch import _Model
    from ymf.symbolic.language import Space
    from ymf.symbolic.notation import Parser
    model = _Model(doc)
    if len(model.heaviside_of) != 1:
        raise RealizeError("two-phase flow needs one phase indicator (a coefficient heaviside(level set)); the "
                           "model has %d" % len(model.heaviside_of))
    (indicator,) = model.heaviside_of
    spec = model.coefficients
    names = {role: _by_role(doc, role) for role in table.TWO_PHASE_COEFFICIENT_ROLES}
    symbols = {n: sympy.Symbol("c_%d" % i) for i, n in enumerate(spec)}
    numbers = {}
    formulas = {}
    for n, entry in spec.items():
        value = entry.get("value") if isinstance(entry, dict) else entry
        formula = entry.get("formula") if isinstance(entry, dict) else None
        if formula is not None:
            formulas[n] = formula
        elif value is not None:
            numbers[symbols[n]] = float(value)
    space = Space(dim, {}, dict(symbols))
    parse = Parser(space).expression

    def at(name, chi):
        if name in formulas:
            return parse(formulas[name]).subs(symbols[indicator], chi)
        if symbols[name] in numbers:               # a constant: the same in both phases
            return symbols[name]
        raise RealizeError("coefficient %s has neither a formula nor a value, and the two-phase realizer needs a "
                           "number for the %s at each phase" % (name, [r for r, n in names.items() if n == name][0]))

    out = {}
    for phase in (0, 1):
        rho = at(names["density"], phase)
        mu = at(names["dynamic_viscosity"], phase)
        nu = sympy.simplify(mu / rho)
        for key, expr in (("rho_%d" % phase, rho), ("nu_%d" % phase, nu)):
            if not set(expr.free_symbols) <= set(numbers):
                raise RealizeError("%s at phase indicator %d is %s, which is not a number"
                                   % (key, phase, expr))
            if expr in numbers:                # a coefficient's value itself, untouched by arithmetic
                out[key] = numbers[expr]
            else:
                out[key] = float(expr.subs(numbers))
    g = parse(formulas.get(names["gravity"]) or str(spec[names["gravity"]].get("value")))
    g = [float(v) for v in (g if isinstance(g, sympy.MatrixBase) else [g])]
    out["gravity"] = g + [0.0] * (3 - len(g))
    out["indicator"] = indicator
    return out


def _box(split):
    """(lower, upper) of a 2-D box domain; refused otherwise."""
    domain_data = split["domain"]
    if domain_data.get("region") is not None:
        raise RealizeError("the realizer builds a box domain; this one is a %s region"
                           % domain_data["region"].get("kind"))
    lower, upper = [float(v) for v in domain_data["lower"]], [float(v) for v in domain_data["upper"]]
    if len(lower) != 2:
        raise RealizeError("the realizer builds 2-D boxes so far; this domain is %d-D" % len(lower))
    return lower, upper


def tank_sides(split):
    """{side: (region, BC_RANS setter, its arguments)} for the four sides of a SpatialTools Tank2D of the domain's
    box, from the closed conditions of every sub-problem (one region per side, one setter per region's
    conditions); refused (:class:`RealizeError`) otherwise. Data only: nothing is built."""
    lower, upper = _box(split)
    size = [u - l for u, l in zip(upper, lower)]
    dim = 2
    tol = 1e-8 * max(size)
    conditions = [c for sub in split["subproblems"] for c in sub.get("boundary_conditions") or []]
    # each side's points that tell which region it is in: its midpoint and quarter points (the corners belong to
    # two sides, so they are not asked)
    sides = {"x-": lambda s: (lower[0], lower[1] + s * size[1]), "x+": lambda s: (upper[0], lower[1] + s * size[1]),
             "y-": lambda s: (lower[0] + s * size[0], lower[1]), "y+": lambda s: (lower[0] + s * size[0], upper[1])}
    regions = {}
    for c in conditions:
        regions.setdefault(c["region"], {"where": _where(c, dim, tol), "conditions": []})
        key = (c["variable"], c["type"], c.get("component"), c.get("only"), repr(c.get("value")))
        if key not in [k for k, _ in regions[c["region"]]["conditions"]]:
            regions[c["region"]]["conditions"].append((key, c))
    out = {}
    for side, point in sides.items():
        found = sorted(r for r, data in regions.items()
                       if all(data["where"](point(s)) for s in (0.25, 0.5, 0.75)))
        partial = sorted(r for r, data in regions.items() if r not in found
                         and any(data["where"](point(s)) for s in (0.25, 0.5, 0.75)))
        if partial or len(found) != 1:
            raise RealizeError("side %s of the tank is in %s; the realizer sets one boundary condition per side"
                               % (side, "no boundary region" if not found and not partial else
                                  "regions %s" % ", ".join(found + partial)))
        (region,) = found
        have = [c for _, c in regions[region]["conditions"]]
        signature = _signature(have)
        setter = next((s for s in table.TWO_PHASE_BOUNDARY_SETTERS if _matches(s["signature"], signature)), None)
        if setter is None:
            raise RealizeError("region %s (side %s): no BC_RANS setter imposes its closed conditions %s; the table "
                               "knows %s" % (region, side, sorted(map(str, signature)),
                                             ", ".join(s["name"] for s in table.TWO_PHASE_BOUNDARY_SETTERS)))
        kwargs = {}
        for argument, (variable, kind) in (setter.get("kwargs") or {}).items():
            value = next(c["value"] for c in have if (c["variable"], c["type"]) == (variable, kind))
            kwargs[argument] = float(value)
        out[side] = (region, setter["setter"], kwargs)
    return out


def _tank(split, he, name, outdir, comm):
    """The SpatialTools tank of the domain's box, its sides' BC_RANS set by :func:`tank_sides`, meshed by triangle
    at element size ``he`` (Proteus's convention, ADRProblem.mesh_constraint)."""
    from proteus import Domain
    from proteus.mprans import SpatialTools as st
    from proteus.ADRProblem import mesh_constraint
    sides = tank_sides(split)
    lower, upper = _box(split)
    size = [u - l for u, l in zip(upper, lower)]
    domain = Domain.PlanarStraightLineGraphDomain()
    coords = None if all(v == 0.0 for v in lower) else [(u + l) / 2.0 for u, l in zip(upper, lower)]
    tank = st.Tank2D(domain, tuple(size), coords)
    record = {}
    for side, (region, setter, kwargs) in sides.items():
        getattr(tank.BC[side], setter)(**kwargs)
        record[side] = {"region": region, "setter": setter, **({"kwargs": kwargs} if kwargs else {})}
    st.assembleDomain(domain)
    domain.MeshOptions.he = he
    domain.MeshOptions.triangleOptions = mesh_constraint(he, 2)
    domain.MeshOptions.triangleFlag = 0
    domain.MeshOptions.genMesh = True
    domain.polyfile = os.path.join(outdir, name)
    if comm.isMaster():
        cwd = os.getcwd()
        os.chdir(outdir)
        try:
            domain.writePoly(name)
        finally:
            os.chdir(cwd)
    comm.barrier()
    return domain, record


def _set(obj, path, value):
    *owners, attribute = path.split(".")
    for owner in owners:
        obj = getattr(obj, owner)
    setattr(obj, attribute, value)


def _initial_conditions(param, model, sub, split, dim, he, entry):
    """The model's initial data, per Proteus variable: the model unknowns' (ymf's split ``initial``), a
    sub-problem's own initial condition, or zero for a correction the sub-problem introduces."""
    variables = list(param.p.coefficients.variableNames)
    components = []
    unknowns = sub.get("unknowns") or {}
    for slot in model.components:
        if slot == "velocity":
            vector = [u for u, comps in unknowns.items() if len(comps) > 1]
            components += unknowns[vector[0]] if vector else []
        elif slot == "pressure":
            scalar = [u for u, comps in unknowns.items() if len(comps) == 1]
            components += unknowns[scalar[0]] if scalar else []
        else:
            components += [c for comps in unknowns.values() for c in comps]
    if len(components) != len(variables):
        raise RealizeError("%s: %s has variables %s, and the sub-problem's unknowns are %s"
                           % (model.subproblem, model.module, variables, components))
    record = {}
    spec_sub = next(s for s in entry["subproblems"] if s["name"] == model.subproblem)
    for variable, component in zip(variables, components):
        own = (sub.get("initial") or {}).get(component)
        if component in split["initial"]:
            value, source = _field(split["initial"][component], dim), "model"
        elif own is not None:
            value, source = _regularized_initial(spec_sub, sub, split, dim, he, entry, model)
        elif (spec_sub.get("introduces") or {}).get("correction_of"):
            value, source = _Constant(0.0), "zero (a correction starts at zero)"
        else:
            raise RealizeError("%s: no initial condition for %s (%s)" % (model.subproblem, component, variable))
        setattr(param.p.initialConditions, variable, value)
        record[variable] = source
    return record


def _regularized_initial(spec_sub, sub, split, dim, he, entry, model):
    """A sub-problem's initial condition ``f(φ)`` with ``f`` regularized: Proteus's own implementation of the
    regularization named, at the width the discretization binds, of the model's initial ``φ``."""
    from proteus.Dispatch import _HEAVISIDE, _width
    text = spec_sub.get("initial_condition") or ""
    m = _HEAVISIDE.match(text)
    by = ((sub.get("regularized") or {}).get("heaviside") or {}).get("by")
    if not m or by not in table.REGULARIZED_FUNCTIONS or m.group(1) not in split["initial"]:
        raise RealizeError("%s: initial condition %r; the realizer takes heaviside(<unknown>) regularized by %s"
                           % (model.subproblem, text, ", ".join(table.REGULARIZED_FUNCTIONS)))
    factor = _width(spec_sub, "heaviside", entry)
    function = resolve(table.REGULARIZED_FUNCTIONS[by])
    return (_Regularized(function, factor * he, _field(split["initial"][m.group(1)], dim)),
            "%s(%r h, %s0)" % (by, factor, m.group(1)))


def two_phase(doc, found, split, realization, name, outdir, ranks=1):
    """A ``TwoPhaseFlowProblem`` built from the dispatch, initialized (its p and n modules built) and checked."""
    from proteus import Comm
    import proteus.TwoPhaseFlow.TwoPhaseFlowProblem as TpFlow
    entry = _entry(doc, found.discretization)
    if "h" not in realization:
        raise RealizeError("%s: the two-phase realizer meshes the tank by element size; give mesh: {h: ...}"
                           % found.discretization)
    he = float(realization["h"])
    if int(realization.get("levels", 1)) != 1:
        raise RealizeError("%s: mesh levels > 1 are not wired up for a split discretization" % found.discretization)
    comm = Comm.get()
    dim = split["dim"]
    domain, sides = _tank(split, he, name, outdir, comm)
    final_time, outputs = _time(split, entry)
    solver, direct = _linear_solver(entry, ranks)
    values = _two_phase_values(doc, dim)

    problem = TpFlow.TwoPhaseFlowProblem()
    problem.domain = domain
    problem.outputStepping.final_time = final_time
    problem.outputStepping.dt_output = final_time / outputs
    problem.outputStepping.systemStepExact = bool(found.system.get("systemStepExact", False))
    physics = problem.SystemPhysics
    for key in ("rho_0", "nu_0", "rho_1", "nu_1"):
        setattr(physics, key, values[key])
    physics.gravity = numpy.array(values["gravity"])
    # the family assumes no surface tension (C42 D6); RANS2P's kernels do not read it either way
    physics.surf_tension_coeff = 0.0
    for model in found:
        if model.module not in table.TWO_PHASE_PARAMETERS:
            raise RealizeError("%s: no TwoPhaseFlow Parameters class builds %s" % (model.subproblem, model.module))
        physics.addModel(resolve(table.TWO_PHASE_PARAMETERS[model.module]), model.subproblem)
    problem.SystemNumerics.cfl = found.system["runCFL"]
    problem.SystemNumerics.useSuperlu = solver == "superlu"

    subs = {s["name"]: s for s in split["subproblems"]}
    record = {"boundary": sides, "initial": {}, "system": {
        "final_time": final_time, "outputs": outputs, "linear_solver": "%s (%s)" % (solver, direct),
        **{k: values[k] for k in ("rho_0", "nu_0", "rho_1", "nu_1", "gravity")}}}
    for model in found:
        param = physics.modelDict[model.subproblem]
        for flag, value in model.coefficients.items():
            setattr(param.p.coefficients, flag, value)
        for key, value in model.numerics.items():
            if key in table.TWO_PHASE_NUMERICS_KNOBS:
                _set(param, table.TWO_PHASE_NUMERICS_KNOBS[key], value)
            elif key == "timeIntegration" and hasattr(param, "timeDiscretization"):
                if value not in table.TWO_PHASE_TIME_DISCRETIZATION:
                    raise RealizeError("%s: %s takes %s, not %s" % (
                        model.subproblem, model.module, ", ".join(table.TWO_PHASE_TIME_DISCRETIZATION), value))
                param.timeDiscretization = table.TWO_PHASE_TIME_DISCRETIZATION[value]
        record["initial"][model.subproblem] = _initial_conditions(param, model, subs[model.subproblem], split, dim,
                                                                  he, entry)
    problem.initializeAll()
    check_built(found, [(model, physics.modelDict[model.subproblem].p, physics.modelDict[model.subproblem].n)
                        for model in found], dim)
    so = problem.so
    so.name = name
    if found.system.get("systemStepControllerType") and not issubclass(
            so.systemStepControllerType, resolve(found.system["systemStepControllerType"])):
        raise RealizeError("the system step controller is %s, the dispatch says %s"
                           % (so.systemStepControllerType.__name__, found.system["systemStepControllerType"]))
    pList = [p for p, _ in so.pnList]
    nList = [n for _, n in so.pnList]
    return Realized(so, pList, nList, problem, record)


# ---------------------------------------------------------------------------------------------------------------
# Shallow water: SWFlowProblem
# ---------------------------------------------------------------------------------------------------------------

class _OfPoint(object):
    """A coefficient of the coordinates as SWFlow calls it, ``f(X)`` with ``X[0]``, ``X[1]`` the coordinates (of
    one point, or arrays of them), as its decks write ``bathymetry_function(X)``."""

    def __init__(self, code, dim):
        from proteus.ADRProblem import _compile
        self.code = _compile(code)
        self.dim = dim

    def __call__(self, X):
        from proteus.ADRProblem import _GLOBALS
        ns = {"x": X[0], "y": X[1], "t": 0.0}
        if self.dim > 2:
            ns["z"] = X[2]
        return eval(self.code, _GLOBALS, ns) + 0.0 * X[0]


def _coefficient_code(doc, name, dim):
    """Code (over x, y, z) of a model coefficient that is a number or a function of the coordinates."""
    from ymf.symbolic.adr import to_code
    from ymf.symbolic.problem import _coefficients
    spec = doc["model"]["strong_form"].get("coefficients") or {}
    values = _coefficients(spec, dim, strict=False)
    if name not in values:
        raise RealizeError("the shallow-water realizer reads coefficient %s, which the model does not resolve"
                           % name)
    return to_code(values[name])


def _shallow_water_value(doc, name, dim):
    """A number for a constant coefficient, else a callable f([x, y])."""
    code = _coefficient_code(doc, name, dim)
    try:
        return float(code)
    except ValueError:
        return _OfPoint(code, dim)


def shallow_water_boundary(split):
    """SWFlow's boundary treatment (an entry of the table's SHALLOW_WATER_BOUNDARY) for the closed conditions,
    which must hold on the whole boundary; refused otherwise. Data only."""
    (sub,) = split["subproblems"]
    conditions = sub.get("boundary_conditions") or []
    if any(c.get("where") is not None for c in conditions):
        raise RealizeError("the shallow-water realizer takes one boundary treatment on the whole boundary (∂Ω)")
    signature = _signature(conditions)
    boundary = next((b for b in table.SHALLOW_WATER_BOUNDARY if _matches(b["signature"], signature)), None)
    if boundary is None:
        raise RealizeError("the boundary conditions %s are none SWFlow imposes; the table knows %s"
                           % (sorted(map(str, signature)),
                              ", ".join(b["name"] for b in table.SHALLOW_WATER_BOUNDARY)))
    return boundary


def precheck(doc, found, split):
    """What :func:`realize` would refuse that the data alone decides (the family, the domain, the boundary, the
    two-phase physical values), so that a runner refuses before anything runs. Raises :class:`RealizeError`."""
    if found.family not in table.REALIZERS:
        raise RealizeError("nothing realizes family %s yet (the table realizes %s)"
                           % (found.family, ", ".join(sorted(table.REALIZERS))))
    if found.family == "two_phase_level_set":
        tank_sides(split)
        _two_phase_values(doc, split["dim"])
    elif found.family == "shallow_water":
        _box(split)
        shallow_water_boundary(split)
        for role in list(table.SHALLOW_WATER_COEFFICIENT_ROLES) + [table.SHALLOW_WATER_BED_ROLE]:
            _by_role(doc, role)


def shallow_water(doc, found, split, realization, name, outdir, ranks=1):
    """An ``SWFlowProblem`` built from the dispatch, its p and n modules loaded as ``parun --SWEs`` loads them
    (proteus/SWFlow/models, reading the problem from Context), and checked."""
    import types
    import proteus
    from proteus import Comm, Context, Domain, default_s, defaults
    from proteus.ADRProblem import mesh_constraint
    from proteus.SWFlow import SWFlowProblem as SWF
    (model,) = found.models
    if model.module not in table.SHALLOW_WATER_MODELS:
        raise RealizeError("%s: SWFlow has no p and n modules for %s" % (model.subproblem, model.module))
    models = table.SHALLOW_WATER_MODELS[model.module]
    entry = _entry(doc, found.discretization)
    if "h" not in realization or int(realization.get("levels", 1)) != 1:
        raise RealizeError("%s: the shallow-water realizer meshes the region by element size, one level; give "
                           "mesh: {h: ...}" % found.discretization)
    he = float(realization["h"])
    dim = split["dim"]
    if split["domain"].get("region") is not None or dim != 2:
        raise RealizeError("the shallow-water realizer builds a 2-D box (RectangularDomain)")
    comm = Comm.get()
    final_time, outputs = _time(split, entry)

    (sub,) = split["subproblems"]
    boundary = shallow_water_boundary(split)

    # the domain: SWFlow's unstructured path, a RectangularDomain's PSLG meshed by triangle at h
    lower, upper = [float(v) for v in split["domain"]["lower"]], [float(v) for v in split["domain"]["upper"]]
    rectangle = Domain.RectangularDomain(L=[u - l for u, l in zip(upper, lower)], x=lower)
    prefix = os.path.join(outdir, name)
    if comm.isMaster():
        rectangle.writePoly(prefix)
    comm.barrier()
    domain = Domain.PlanarStraightLineGraphDomain(fileprefix=prefix)
    domain.MeshOptions.triangleOptions = mesh_constraint(he, dim)

    # initial data and the coefficients
    unknowns = sub["unknowns"]
    depth = [u for u, comps in unknowns.items() if len(comps) == 1]
    discharge = [u for u, comps in unknowns.items() if len(comps) > 1]
    if len(depth) != 1 or len(discharge) != 1:
        raise RealizeError("%s: SWFlow solves a depth and a discharge; the sub-problem's unknowns are %s"
                           % (model.subproblem, sorted(unknowns)))
    components = unknowns[depth[0]] + unknowns[discharge[0]]
    variables = ["water_height", "x_mom", "y_mom"]
    initial = {v: _field(split["initial"][c], dim) for v, c in zip(variables, components)}
    physical = dict(SWF.default_physical_parameters)
    for role, key in table.SHALLOW_WATER_COEFFICIENT_ROLES.items():
        physical[key] = _shallow_water_value(doc, _by_role(doc, role), dim)
    bathymetry = _OfPoint(_coefficient_code(doc, _by_role(doc, table.SHALLOW_WATER_BED_ROLE), dim), dim)

    problem = SWF.SWFlowProblem(
        sw_model=models["sw_model"], cfl=found.system["runCFL"],
        outputStepping=SWF.OutputStepping(final_time, dt_output=final_time / outputs),
        structured=False, he=he, domain=domain, initialConditions=initial,
        boundaryConditions={v: (lambda x, flag: None) for v in variables},
        reflectingBCs=boundary["reflectingBCs"], bathymetry=bathymetry)
    # copies: SWFlowProblem hands out its module's default dicts, which a deck then edits in place
    problem.physical_parameters = physical
    problem.swe_parameters = dict(SWF.default_swe_parameters)
    for flag in table.SHALLOW_WATER_FLAGS:
        if flag in model.coefficients:
            problem.swe_parameters[flag] = model.coefficients[flag]
    problem.swe_parameters["SSPOrder"] = model.numerics["timeOrder"]
    problem.swe_parameters["cfl"] = found.system["runCFL"]

    # the p and n modules, as parun --SWEs loads them: they read the problem from Context
    case = types.ModuleType("ymf_" + name)
    case.mySWFlowProblem = problem
    Context.setFromModule(case)
    path = os.path.join(os.path.dirname(proteus.__file__), "SWFlow", "models")
    p = defaults.load_physics(models["p"], path=path)
    n = defaults.load_numerics(models["n"], path=path)
    p.name = models["p"]
    check_built(found, [(model, p, n)], dim)
    so = defaults.System_base()
    so.name = name
    so.pnList = [(p, n)]
    so.sList = [default_s]
    stepping = problem.outputStepping
    so.tnList = [0.0, stepping["dt_init"]] + [float(k) * stepping["final_time"] / float(stepping["nDTout"])
                                              for k in range(1, stepping["nDTout"] + 1)]
    if not issubclass(so.systemStepControllerType, resolve(found.system["systemStepControllerType"])):
        raise RealizeError("the system step controller is %s, the dispatch says %s"
                           % (so.systemStepControllerType.__name__, found.system["systemStepControllerType"]))
    record = {"boundary": {"whole boundary": boundary["name"], "reflectingBCs": boundary["reflectingBCs"]},
              "initial": dict(zip(variables, components)),
              "system": {"final_time": final_time, "outputs": outputs, "mesh": "triangle, %s" % (
                  domain.MeshOptions.triangleOptions,),
                  "linear_solver": "SWFlow's n module: LU on one rank, KSP_petsc4py on several",
                  **{k: (v if isinstance(v, float) else "f(x, y)") for k, v in physical.items()
                     if k in ("gravity", "mannings", "porosity", "drag_alpha", "drag_beta")}}}
    return Realized(so, [p], [n], problem, record)


# ---------------------------------------------------------------------------------------------------------------
# The check: what the machinery built is what the dispatch says
# ---------------------------------------------------------------------------------------------------------------

def _equal(actual, expected):
    if isinstance(expected, type) or callable(expected):
        return actual is expected
    if isinstance(expected, bool) or isinstance(actual, bool):
        return bool(actual) == bool(expected)
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        return float(actual) == float(expected)
    return actual == expected


def _resolved(value, model):
    if isinstance(value, str) and (value.startswith("proteus.") or value.startswith("{module}")):
        return model.resolve(value)
    if isinstance(value, dict):
        return {k: _resolved(v, model) for k, v in value.items()}
    return value


def differences(model, p, n, nd):
    """What the built p and n modules of one dispatched model differ in from the dispatch (empty if nothing):
    every coefficient flag, physics attribute and numerics attribute the dispatch gives, and the spaces."""
    out = []
    if type(p.coefficients) is not model.resolve(model.coefficients_class):
        out.append("coefficients are %s" % type(p.coefficients).__name__)
    for flag, value in model.coefficients.items():
        if not _equal(getattr(p.coefficients, flag, None), value):
            out.append("coefficients.%s is %r, dispatch says %r" % (flag, getattr(p.coefficients, flag, None),
                                                                     value))
    for attribute, value in model.physics.items():
        if not _equal(getattr(p, attribute, None), _resolved(value, model)):
            out.append("p.%s" % attribute)
    for attribute, value in model.numerics.items():
        if attribute == "quadrature_order":
            if n.elementQuadrature.order != value or n.elementBoundaryQuadrature.order != value:
                out.append("quadrature order %s, dispatch says %s" % (n.elementQuadrature.order, value))
        elif "." in attribute:
            owner, field = attribute.split(".")
            obj = getattr(n, owner)
            actual = obj.lagRequested if field == "lag" else getattr(obj, field)
            if not _equal(actual, value):
                out.append("n.%s is %r, dispatch says %r" % (attribute, actual, value))
        elif attribute in ("subgridError", "shockCapturing"):
            obj = getattr(n, attribute)
            if (obj is not None) if value is None else (type(obj) is not model.resolve(value)):
                out.append("n.%s is %s, dispatch says %s" % (attribute, type(obj).__name__, value))
        else:
            actual = getattr(n, attribute, None)
            if not _equal(actual, _resolved(value, model)):
                out.append("n.%s is %r, dispatch says %r" % (attribute, actual, value))
    expected = {i: resolve(b) for i, b in model.femSpaces(nd).items()}
    if n.femSpaces != expected:
        out.append("n.femSpaces")
    return out


def check_built(found, built, nd):
    """Refuse a run whose built modules differ from the dispatch. ``built`` is [(model, p, n)] in order."""
    problems = []
    for model, p, n in built:
        problems += ["%s (%s): %s" % (model.subproblem, model.cell, d) for d in differences(model, p, n, nd)]
        if getattr(n, "runCFL", found.system.get("runCFL")) != found.system.get("runCFL"):
            problems.append("%s: runCFL %r, dispatch says %r" % (model.subproblem, n.runCFL,
                                                                 found.system["runCFL"]))
    if problems:
        raise RealizeError("the deck machinery built something other than the dispatch:\n  " + "\n  ".join(problems))


# ---------------------------------------------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------------------------------------------

@contextlib.contextmanager
def petsc_options_restored():
    """Leave PETSc's options database as it was: the deck machinery sets each model's solver options by prefix
    (rans2p_ksp_type, ...), which a later run in the same process would otherwise inherit."""
    from petsc4py import PETSc
    options = PETSc.Options()
    before = options.getAll()
    try:
        yield
    finally:
        after = options.getAll()
        for key in after:
            if key not in before:
                options.delValue(key)
        for key, value in before.items():          # changed, or deleted inside the context
            if after.get(key) != value:
                options.setValue(key, value)


def run(realized, name, opts=None):
    """Solve a realized discretization; return the NS_base."""
    from proteus import NumericalSolution, default_s, runtime
    if opts is None:
        opts = runtime.RunOptions()
    so = realized.so
    sList = so.sList or [default_s for _ in so.pnList]
    ns = NumericalSolution.NS_base(so, realized.pList, realized.nList, sList, opts)
    ns.calculateSolution(name)
    return ns
