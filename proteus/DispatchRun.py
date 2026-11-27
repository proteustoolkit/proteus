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
import ast
import contextlib
import os
import warnings

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


class _ExactSquares(ast.NodeTransformer):
    """``u**2`` as ``numpy.square(u)``."""

    def visit_BinOp(self, node):
        self.generic_visit(node)
        if (isinstance(node.op, ast.Pow) and isinstance(node.right, ast.Constant)
                and isinstance(node.right.value, (int, float)) and not isinstance(node.right.value, bool)
                and node.right.value == 2):
            call = ast.Call(func=ast.Attribute(value=ast.Name(id="numpy", ctx=ast.Load()), attr="square", ctx=ast.Load()),
                            args=[node.left], keywords=[])
            return ast.copy_location(call, node)
        return node


def exact_squares(code):
    """ymf's code string with every ``u**2`` replaced by ``numpy.square(u)``, i.e. ``u*u`` rounded once.

    The decks square arrays (``(d**2).sum()``), and numpy squares an array by multiplying. The same ``**2`` on one
    point (an initial condition is evaluated a node at a time, so ``x`` is a ``numpy.float64``) is the C library's
    ``pow(u, 2.0)``, which is not ``u*u`` for about one double in 800 (measured on macOS arm64, macOS x86_64 and
    Linux glibc alike, 356 to 517 of 400000 draws): the last bit of a signed distance, and a run that is no longer
    bitwise the deck's. Whether a mesh has such a node is a matter of its coordinates (the damBreakLevelSet3D
    comparison failed on the one node of a 274-node TetGen mesh that has it, and passed on 278-node ones)."""
    return ast.unparse(_ExactSquares().visit(ast.parse(code, mode="eval")))


def _field(code, dim):
    from proteus.ADRProblem import _Field
    return _Field(exact_squares(code), dim)


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


#: the packages ymf's ``linear_solver`` names that the deck machinery selects, by the kind they request; ``direct`` and
#: ``iterative`` are the kinds themselves (ymf.vocabulary.LINEAR_SOLVERS)
_MACHINERY_PACKAGES = {"superlu": "direct", "petsc": "iterative"}


def _linear_request(entry, family):
    """The kind (``direct`` or ``iterative``) of the discretization's ``linear_solver``, or None when it names none.
    A package the machinery does not select is refused: asking for what is not done is not met by doing something
    else. (The solver's ``petsc_options`` are read per model: :func:`model_petsc_options`.)"""
    solver = entry.get("solver") or {}
    word = solver.get("linear_solver")
    if word is None:
        return None
    try:
        kind = vocab.LINEAR_SOLVERS[str(word)]["kind"]
    except KeyError:
        raise RealizeError("%s: linear_solver %r; ymf knows %s" % (entry["name"], word, ", ".join(vocab.LINEAR_SOLVERS)))
    if str(word) not in ("direct", "iterative") and str(word) not in _MACHINERY_PACKAGES:
        raise RealizeError("%s: linear_solver %s is a package the deck machinery does not select (it takes direct "
                           "or iterative, or the packages %s)" % (entry["name"], word, ", ".join(_MACHINERY_PACKAGES)))
    return kind


def _linear_solver(entry, ranks):
    """``(kind, how)`` for TwoPhaseFlow: ``direct`` is useSuperlu (SuperLU on one rank, SuperLU_DIST through PETSc on
    several), ``iterative`` the machinery's PETSc KSP solvers and their options per model prefix. Not stated: iterative."""
    kind = _linear_request(entry, "two-phase") or "iterative"
    if kind == "direct":
        return kind, "SuperLU" if ranks == 1 else "SuperLU_DIST"
    return kind, "PETSc KSP"


def _shallow_water_linear_solver(entry, ranks):
    """What SWFlow's n module solves with, by the number of ranks (LU on one rank, KSP_petsc4py on several), against
    what the spec asks: a different kind is refused rather than overridden."""
    kind = _linear_request(entry, "shallow-water")
    solves = "direct" if ranks == 1 else "iterative"
    if kind is not None and kind != solves:
        raise RealizeError(
            "%s: linear_solver asks for a %s solve, and SWFlow's n module solves with %s on %d rank%s; it cannot be "
            "told otherwise (leave linear_solver out to take what it does)"
            % (entry["name"], kind, "LU" if ranks == 1 else "KSP_petsc4py", ranks, "" if ranks == 1 else "s"))
    return solves, "LU" if ranks == 1 else "KSP_petsc4py"


def model_petsc_options(found, split):
    """{sub-problem: {option: value}}: the PETSc options (``solver.petsc_options``, without the leading dash) each
    dispatched model's linear solve is given. A sub-problem's own options are its model's; the discretization's
    (which ymf's split merges into every sub-problem) reach every model. (A sub-problem the discretization has no
    model for is refused by its name in ``dispatch``, before this.) Data only."""
    out = {m.subproblem: {} for m in found}
    for sub in split["subproblems"]:
        options = (sub.get("solver") or {}).get("petsc_options") or {}
        for key, value in options.items():
            out[sub["name"]][key] = str(value)
    return {name: options for name, options in out.items() if options}


def _serial_lu(found, entry, ranks):
    """Whether the discretization's linear solves are LU (no PETSc KSP to read options): two-phase ``direct`` on one
    rank (useSuperlu), SWFlow on one rank."""
    if found.family == "two_phase_level_set":
        return _linear_solver(entry, ranks)[0] == "direct" and ranks == 1
    return found.family in SHALLOW_WATER_FAMILIES and ranks == 1


def _unused_petsc_options(found, options, why):
    """Warn that options the spec states are not read (the solve they name does not go through PETSc's KSP), and
    return them for the run record. Nothing is dropped silently."""
    warnings.warn("%s: petsc_options for %s are not read: %s" % (found.discretization, ", ".join(sorted(options)), why),
                  RuntimeWarning, stacklevel=3)
    return {m: dict(o) for m, o in options.items()}


def _check_petsc_options(found, split, entry, ranks):
    """The options to apply; ``({}, unused)`` of them when the solve is serial LU, which reads none (a warning)."""
    options = model_petsc_options(found, split)
    if options and _serial_lu(found, entry, ranks):
        return {}, _unused_petsc_options(found, options, "on one rank this discretization solves with LU (ask for "
                                                          "iterative, or run on several ranks, to have them read)")
    return options, {}


def _apply_petsc_options(options, prefixes):
    """Set each model's options under its solver prefix in PETSc's options database, after the machinery has set its
    own (the spec's win where they name the same option)."""
    from petsc4py import PETSc
    database = PETSc.Options()
    for name, mine in options.items():
        for key, value in mine.items():
            database.setValue(prefixes[name] + key, value)


def _names(x, tol, dim):
    ns = {"x": x[0], "tol": tol}
    if dim > 1:
        ns["y"] = x[1]
    if dim > 2:
        ns["z"] = x[2]
    return ns


def _where(condition, dim, tol):
    from proteus.ADRProblem import _GLOBALS, _compile
    if condition.get("where") is None:
        return lambda x: True
    code = _compile(condition["where"])
    return lambda x: bool(eval(code, _GLOBALS, _names(x, tol, dim)))


def _holds(condition, dim, tol):
    """``f(x, flag)``: does a boundary condition hold at the boundary point ``x`` (on the boundary piece of mesh tag
    ``flag``)? As ymf states it (ymf round 7): where the region is (``where``, or ``flags``, the region's mesh
    tags) AND on its ``part`` (if it has one) AND NOT on any of the parts it is ``excluding``. Each code is read as
    ``where`` is, over x, y, z and ``tol``."""
    from proteus.ADRProblem import _GLOBALS, _compile
    region = _where(condition, dim, tol)
    flags = None if condition.get("flags") is None else {int(f) for f in condition["flags"]}
    part = None if condition.get("part") is None else _compile(condition["part"])
    excluded = [_compile(code) for code in condition.get("excluding") or []]

    def holds(x, flag=None):
        if flags is not None and (flag is None or int(flag) not in flags):
            return False
        if not region(x):
            return False
        ns = _names(x, tol, dim)
        if part is not None and not eval(part, _GLOBALS, ns):
            return False
        return not any(eval(code, _GLOBALS, ns) for code in excluded)
    return holds


def _number(value):
    if isinstance(value, dict):
        return tuple(_number(value[k]) for k in sorted(value))
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


def _value_of(condition):
    """A condition's value for a signature: its number(s), or ``sign:1`` for a signed condition without one (the
    level set's inflow at an atmosphere, whose value is the phase indicator's, not its own)."""
    if condition.get("value") is None and condition.get("sign") is not None:
        return "sign:%d" % int(condition["sign"])
    return _number(condition.get("value"))


def _signature(conditions):
    """{(variable, type, component, only, value)} of one region's closed conditions."""
    return {(c["variable"], c["type"], c.get("component"), c.get("only"), _value_of(c)) for c in conditions}


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


def _two_phase_values(doc, dim, found=None):
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
    out["surface_tension"] = _surface_tension(doc, spec, found)
    return out


def _surface_tension(doc, spec, found):
    """σ of the coefficient with role ``surface_tension`` (a number), else 0 (C42 D6: the family assumes none). Only
    RANS3PF's kernels read it (RANS2P computes no surface tension), so a nonzero σ with another flow solver is
    refused, rather than run without the force the spec states."""
    name = vocab.coefficient_by_role(doc, "surface_tension")
    if name is None:
        return 0.0
    entry = spec[name]
    value = entry.get("value") if isinstance(entry, dict) else entry
    if value is None:
        raise RealizeError("the surface tension %s must be a number (value: ...) for the realizer" % name)
    sigma = float(value)
    if sigma != 0.0:
        flow = [m for m in found if m.module == "proteus.kernels.RANS2P"] \
            if found is not None else []
        if flow:
            raise RealizeError("the model states surface tension %s = %r, and %s computes none (only RANS3PF's "
                               "kernels read it): state 0 or choose a RANS3PF discretization"
                               % (name, sigma, ", ".join(m.module.rsplit(".", 1)[1] for m in flow)))
    return sigma


#: the families SWFlow's decks solve: the hydrostatic model and the dispersive one by hyperbolic relaxation
SHALLOW_WATER_FAMILIES = ("shallow_water", "dispersive_shallow_water")

#: SpatialTools.Tank2D's boundary flags: the flags a stored mesh of a tank carries (the sides of a box mesh)
TANK2D_FLAGS = {"y-": 1, "x+": 2, "y+": 3, "x-": 4}
#: and Tank3D's
TANK3D_FLAGS = {"z-": 1, "x-": 2, "y+": 3, "x+": 4, "y-": 5, "z+": 6}


def _stored(split):
    """The prefix of the stored mesh the domain names (``kind: file, format: tetgen``: ``<prefix>.node``, ``.ele``,
    ``.edge``, as triangle writes them and as the decks read them with ``genMesh=False``), or None for a box
    that is meshed. Refused for any other region."""
    region = split["domain"].get("region")
    if region is None:
        return None
    if region.get("kind") != "file" or region.get("format") != "tetgen":
        raise RealizeError("the realizer builds a box domain, or reads a stored triangle/tetgen mesh (kind: file, "
                           "format: tetgen); this one is a %s region%s"
                           % (region.get("kind"), " in format %s" % region["format"] if region.get("format") else ""))
    prefix = region["path"]
    if prefix.endswith((".node", ".ele")):
        prefix = os.path.splitext(prefix)[0]
    if split["domain"].get("lower") is None:
        raise RealizeError("a stored mesh is realized as the tank of its box: state the file domain's bounds "
                           "(bounds: {lower, upper})")
    return prefix


def stored_element_size(prefix):
    """The element size of a stored triangle mesh: the longest edge of its elements, which is what Proteus's
    ``mesh.h`` is. (A mesh that is already made has no h to state; the run's tolerances and regularization widths
    read the size, so it is measured.)"""
    import numpy

    def rows(suffix):
        with open(prefix + suffix) as f:
            lines = [l.split("#")[0].split() for l in f]
        lines = [l for l in lines if l]
        return lines[0], lines[1:]
    head, nodes = rows(".node")
    dim = int(head[1])
    x = numpy.array([[float(v) for v in l[1:1 + dim]] for l in nodes[:int(head[0])]])
    first = int(nodes[0][0])
    head, elements = rows(".ele")
    cells = numpy.array([[int(v) - first for v in l[1:1 + int(head[1])]] for l in elements[:int(head[0])]])
    longest = 0.0
    for a in range(cells.shape[1]):
        for b in range(a + 1, cells.shape[1]):
            longest = max(longest, float(numpy.linalg.norm(x[cells[:, a]] - x[cells[:, b]], axis=1).max()))
    return longest


def _box(split):
    """(lower, upper) of a 2-D box domain, or of the bounds of a stored mesh's file domain; refused otherwise."""
    _stored(split)
    domain_data = split["domain"]
    lower, upper = [float(v) for v in domain_data["lower"]], [float(v) for v in domain_data["upper"]]
    if len(lower) not in (2, 3):
        raise RealizeError("the realizer builds 2-D and 3-D boxes; this domain is %d-D" % len(lower))
    if len(lower) == 3 and _stored(split) is not None:
        raise RealizeError("a stored mesh is read for a 2-D tank so far (the flags of Tank2D); this domain is 3-D")
    return lower, upper


def tank_sides(split):
    """{side: (region, BC_RANS setter, its arguments, a note on what it does not impose)} for the sides of a
    SpatialTools Tank2D or Tank3D of the domain's box, from the closed conditions of every sub-problem (one region per
    side, one setter per region's conditions); refused (:class:`RealizeError`) otherwise. Data only: nothing is
    built."""
    import itertools
    lower, upper = _box(split)
    size = [u - l for u, l in zip(upper, lower)]
    dim = len(lower)
    tol = 1e-8 * max(size)
    conditions = [c for sub in split["subproblems"] for c in sub.get("boundary_conditions") or []]
    for c in conditions:
        for word, why in (("part", "a state on part of a side"), ("excluding", "a condition less a part"),
                          ("trace", "a boundary value from another sub-problem's trace (the penalty form is the "
                                    "generator's, C19)")):
            if c.get(word) is not None:
                raise RealizeError("%s of %s (%s: %r): the realizer sets one BC_RANS setter per side of the tank, "
                                   "not %s" % (c["variable"], c["region"], word, c[word], why))
    # each side's points that tell which region it is in: its midpoint and quarter points (the corners belong to
    # two sides, so they are not asked)
    names = "xyz"[:dim]

    def face(axis, end):
        value = upper[axis] if end == "+" else lower[axis]
        others = [a for a in range(dim) if a != axis]

        def points():
            for fractions in itertools.product((0.25, 0.5, 0.75), repeat=dim - 1):
                point = [0.0] * dim
                point[axis] = value
                for a, f in zip(others, fractions):
                    point[a] = lower[a] + f * size[a]
                yield tuple(point)
        return points
    sides = {"%s%s" % (names[axis], end): face(axis, end) for axis in range(dim) for end in "-+"}
    sides = {side: sides[side] for side in (("x-", "x+", "y-", "y+") if dim == 2 else
                                            ("x-", "x+", "y-", "y+", "z-", "z+"))}
    stored = _stored(split) is not None
    flag_of = TANK2D_FLAGS if dim == 2 else TANK3D_FLAGS
    regions = {}
    for c in conditions:
        regions.setdefault(c["region"], {"where": _where(c, dim, tol), "flags": c.get("flags"), "conditions": []})
        key = (c["variable"], c["type"], c.get("component"), c.get("only"), repr(c.get("value")))
        if key not in [k for k, _ in regions[c["region"]]["conditions"]]:
            regions[c["region"]]["conditions"].append((key, c))
    if any(data["flags"] is not None for data in regions.values()) and not stored:
        raise RealizeError("a region named by mesh tags needs a stored mesh to carry them (domain: kind: file)")

    def claims(data, side, point):
        """(all of the side, part of it) for a region."""
        if data["flags"] is not None:
            hit = flag_of[side] in {int(f) for f in data["flags"]}
            return hit, False
        hits = [data["where"](x) for x in point()]
        return all(hits), any(hits) and not all(hits)

    out = {}
    for side, point in sides.items():
        found = sorted(r for r, data in regions.items() if claims(data, side, point)[0])
        partial = sorted(r for r, data in regions.items() if r not in found and claims(data, side, point)[1])
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
        for argument, (variable, kind, *path) in (setter.get("kwargs") or {}).items():
            value = next(c for c in have if (c["variable"], c["type"]) == (variable, kind))
            for key in (path[0] if path else "value").split("."):
                value = value[key]
            kwargs[argument] = float(value)
        out[side] = (region, setter["setter"], kwargs, setter.get("note", "") % kwargs)
    return out


def _tank(split, he, name, outdir, comm):
    """The SpatialTools tank of the domain's box (Tank2D meshed by triangle, Tank3D by TetGen), its sides' BC_RANS set by
    :func:`tank_sides`, at element size ``he`` (Proteus's convention, ADRProblem.mesh_constraint), or its stored mesh."""
    from proteus import Domain
    from proteus import SpatialTools as st
    from proteus.ADRProblem import mesh_constraint
    sides = tank_sides(split)
    lower, upper = _box(split)
    size = [u - l for u, l in zip(upper, lower)]
    dim = len(lower)
    domain = Domain.PlanarStraightLineGraphDomain() if dim == 2 else Domain.PiecewiseLinearComplexDomain()
    coords = None if all(v == 0.0 for v in lower) else [(u + l) / 2.0 for u, l in zip(upper, lower)]
    tank = (st.Tank2D if dim == 2 else st.Tank3D)(domain, tuple(size), coords)
    record = {}
    for side, (region, setter, kwargs, note) in sides.items():
        getattr(tank.BC[side], setter)(**kwargs)
        record[side] = {"region": region, "setter": setter, **({"kwargs": kwargs} if kwargs else {}),
                        **({"note": note} if note else {})}
    st.assembleDomain(domain)
    domain.MeshOptions.he = he
    domain.MeshOptions.triangleOptions = mesh_constraint(he, dim)
    if dim == 2:
        domain.MeshOptions.triangleFlag = 0
    stored = _stored(split)
    if stored is not None:
        # a stored mesh, read as the decks read theirs with genMesh=False; its flags are the tank's (tank_sides
        # chose each side's setter from them), he the element size it was made at
        for suffix in (".node", ".ele"):
            if not os.path.exists(stored + suffix):
                raise RealizeError("the stored mesh %s%s does not exist" % (stored, suffix))
        domain.MeshOptions.genMesh = False
        domain.polyfile = os.path.abspath(stored)
        return domain, record
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


def model_components(model, sub):
    """The sub-problem's unknown components in the order of the model's Proteus variables (the cell's slots:
    pressure first, then velocity, for the flow models)."""
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
    return components


def _initial_conditions(param, model, sub, split, dim, he, entry):
    """The model's initial data, per Proteus variable: the model unknowns' (ymf's split ``initial``), a
    sub-problem's own initial condition, or zero for a correction the sub-problem introduces."""
    variables = list(param.p.coefficients.variableNames)
    unknowns = sub.get("unknowns") or {}
    components = model_components(model, sub)
    if not unknowns and model.role in table.SCHEME_AUXILIARY_ROLES:
        # the projection scheme's pressure increment and its spin-up are the scheme's own variables: the spec has
        # no unknown for them, and the decks start them at zero
        for variable in variables:
            setattr(param.p.initialConditions, variable, _Constant(0.0))
        return {variable: "zero (the scheme's own variable, no unknown of the spec)" for variable in variables}
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


def _padded(point):
    point = [float(v) for v in point]
    if len(point) > 3:
        raise RealizeError("a gauge point has at most three coordinates: %s" % (point,))
    return tuple(point + [0.0] * (3 - len(point)))


def gauge_objects(doc, owners):
    """{sub-problem: [Gauges auxiliary variables]} for the spec's ``model.gauges`` (point, line and line_integral kinds),
    each built as a deck builds its own (PointGauges, LineGauges, LineIntegralGauges) and attached to the model that
    solves the fields it samples. ``owners`` is {sub-problem name: (sub-problem plain data, [(component, Proteus
    variable name), ...])}. A field is an unknown of the sub-problem (a vector's gauge samples every component), or one
    of its components. Refused: a gauge of the boundary-integral kind (forces and moments are the flow model's kernel
    sums, which the gauge only reads: not wired), fields of two sub-problems in one gauge (it is one file), a field no
    sub-problem solves."""
    from proteus import Gauges
    out = {}
    for index, gauge in enumerate(doc["model"].get("gauges") or []):
        name = gauge.get("name") or "gauge%d" % index
        kind = gauge["kind"]
        if kind not in ("point", "line", "line_integral"):
            raise RealizeError("gauge %s: the %s kind is not run through dispatch yet (the force and moment sums are "
                               "the flow kernels', read by a boundary-integral gauge); point, line and line_integral "
                               "gauges are" % (name, kind))
        chosen = {}
        for field in gauge["fields"]:
            hits = [(sub, [v for c, v in pairs if c in (sub_data.get("unknowns") or {}).get(field, [field])])
                    for sub, (sub_data, pairs) in owners.items()
                    if field in (sub_data.get("unknowns") or {}) or field in [c for c, _ in pairs]]
            if not hits:
                raise RealizeError("gauge %s: no sub-problem solves %s (they solve %s)" % (name, field, ", ".join(
                    sorted(u for _, (d, _) in owners.items() for u in (d.get("unknowns") or {})))))
            sub, variables = hits[0]
            chosen.setdefault(sub, []).extend(variables)
        if len(chosen) != 1:
            raise RealizeError("gauge %s samples fields of %s: one gauge is one file on one model; split it by "
                               "sub-problem" % (name, " and ".join(sorted(chosen))))
        (sub, variables), = chosen.items()
        common = {"sampleRate": float(gauge.get("sample_rate") or 0.0),
                  "fileName": gauge.get("file") or "%s.csv" % name}
        if kind == "point":
            if not gauge.get("points"):
                raise RealizeError("gauge %s: a point gauge needs points" % name)
            item = Gauges.PointGauges(gauges=((tuple(variables), tuple(_padded(p) for p in gauge["points"])),),
                                      **common)
        else:
            if not gauge.get("lines"):
                raise RealizeError("gauge %s: a %s gauge needs lines" % (name, kind))
            lines = tuple(tuple(_padded(p) for p in line) for line in gauge["lines"])
            make = Gauges.LineGauges if kind == "line" else Gauges.LineIntegralGauges
            item = make(gauges=((tuple(variables), lines),), **common)
        _bind_by_position(item, [v for _, v in owners[sub][1]], name)
        out.setdefault(sub, []).append(item)
    return out


def _bind_by_position(item, declared, name):
    """The names a gauge samples are the model's variables, which a level model may rename (CLSVOF's coefficients say
    ``clsvof``, its level model ``phi``, and Gauges reads the level model's). ``declared`` are the names of the
    coefficients; when the model is attached the gauge's field names are translated, position by position, to the
    names the level model has."""
    original = item.attachModel

    def attach(model, ar):
        actual = list(model.levelModelList[-1].coefficients.variableNames)
        if actual != declared:
            if len(actual) != len(declared):
                raise RealizeError("gauge %s: the model has %d variables %s, the dispatch declared %s"
                                   % (name, len(actual), actual, declared))
            mapping = dict(zip(declared, actual))
            item.fields = [mapping.get(f, f) for f in item.fields]
            for point in item.points.values():
                point["fields"] = {mapping.get(f, f) for f in point["fields"]}
            item.lines = [(mapping.get(f, f), line) for f, line in item.lines]
        return original(model, ar)
    item.attachModel = attach


def _gauge_owners(model_list, subs, variables_of):
    """The ``owners`` of :func:`gauge_objects`: each model's sub-problem and its (component, variable) pairs."""
    owners = {}
    for model in model_list:
        sub = subs[model.subproblem]
        variables = variables_of(model)
        owners[model.subproblem] = (sub, list(zip(model_components(model, sub), variables)))
    return owners


def two_phase(doc, found, split, realization, name, outdir, ranks=1):
    """A ``TwoPhaseFlowProblem`` built from the dispatch, initialized (its p and n modules built) and checked."""
    from proteus import Comm, LinearSolvers
    import proteus.TwoPhaseFlow.TwoPhaseFlowProblem as TpFlow
    entry = _entry(doc, found.discretization)
    stored = _stored(split)
    if stored is None and "h" not in realization:
        raise RealizeError("%s: the two-phase realizer meshes the tank by element size; give mesh: {h: ...}"
                           % found.discretization)
    he = stored_element_size(stored) if stored is not None else float(realization["h"])
    if int(realization.get("levels", 1)) != 1:
        raise RealizeError("%s: mesh levels > 1 are not wired up for a split discretization" % found.discretization)
    comm = Comm.get()
    dim = split["dim"]
    domain, sides = _tank(split, he, name, outdir, comm)
    final_time, outputs = _time(split, entry)
    solver, direct = _linear_solver(entry, ranks)      # the kind, and what does it
    values = _two_phase_values(doc, dim, found)

    problem = TpFlow.TwoPhaseFlowProblem()
    problem.domain = domain
    problem.outputStepping.final_time = final_time
    problem.outputStepping.dt_output = final_time / outputs
    problem.outputStepping.systemStepExact = bool(found.system.get("systemStepExact", False))
    physics = problem.SystemPhysics
    for key in ("rho_0", "nu_0", "rho_1", "nu_1"):
        setattr(physics, key, values[key])
    physics.gravity = numpy.array(values["gravity"])
    # sigma: the spec's coefficient with role surface_tension, else none (C42 D6)
    physics.surf_tension_coeff = values["surface_tension"]
    for model in found:
        if model.module not in table.TWO_PHASE_PARAMETERS:
            raise RealizeError("%s: no TwoPhaseFlow Parameters class builds %s" % (model.subproblem, model.module))
        physics.addModel(resolve(table.TWO_PHASE_PARAMETERS[model.module]), model.subproblem)
    problem.SystemNumerics.cfl = found.system["runCFL"]
    problem.SystemNumerics.useSuperlu = solver == "direct"

    subs = {s["name"]: s for s in split["subproblems"]}
    record = {"boundary": sides, "initial": {}, "system": {
        "mesh": ("stored %s, h = %g" % (_stored(split), he)) if _stored(split) is not None else
        "triangle, h = %g" % he,
        "final_time": final_time, "outputs": outputs, "linear_solver": "%s (%s)" % (solver, direct),
        **{k: values[k] for k in ("rho_0", "nu_0", "rho_1", "nu_1", "gravity", "surface_tension")}}}
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
    gauges = gauge_objects(doc, _gauge_owners(
        found, subs, lambda m: list(physics.modelDict[m.subproblem].p.coefficients.variableNames)))
    for sub_name, items in gauges.items():
        physics.modelDict[sub_name].auxiliaryVariables = items
    if gauges:
        record["system"]["gauges"] = {s: [g.fileName for g in items] for s, items in gauges.items()}
    problem.initializeAll()
    petsc_options, unused = _check_petsc_options(found, split, entry, ranks)
    prefixes = {}
    for model in found:
        n = physics.modelDict[model.subproblem].n
        prefixes[model.subproblem] = n.linear_solver_options_prefix or ""
        if model.subproblem in petsc_options and not (isinstance(n.multilevelLinearSolver, type) and issubclass(
                n.multilevelLinearSolver, LinearSolvers.KSP_petsc4py)):
            unused.update(_unused_petsc_options(found, {model.subproblem: petsc_options.pop(model.subproblem)},
                                                "%s solves with %s, which reads no PETSc options" % (
                                                    model.module, n.multilevelLinearSolver.__name__)))
    _apply_petsc_options(petsc_options, prefixes)
    if petsc_options:
        record["system"]["petsc_options"] = {m: {prefixes[m] + k: v for k, v in o.items()}
                                             for m, o in petsc_options.items()}
    if unused:
        record["system"]["petsc_options_unused"] = unused
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

class _AuxiliaryInitial(object):
    """The initial value of a relaxation auxiliary: ymf's expression of the depth's initial value (``h^2``, ``0``),
    called as SWFlow calls an initial condition, ``uOfXT(X, t)``."""

    def __init__(self, text, depth_name, depth, dim):
        import sympy
        from sympy.parsing.sympy_parser import (convert_xor, implicit_multiplication, parse_expr,
                                                standard_transformations)
        symbols = {name: sympy.Symbol(name) for name in (depth_name, "x", "y", "z", "t")}
        try:
            expression = parse_expr(text, local_dict=symbols, transformations=standard_transformations + (
                convert_xor, implicit_multiplication))
        except (sympy.SympifyError, SyntaxError, TypeError, NameError) as exc:
            raise RealizeError("an auxiliary's initial value %r is not an expression of %s, x, y, t: %s"
                               % (text, depth_name, exc)) from None
        extra = sorted(str(s) for s in expression.free_symbols - set(symbols.values()))
        if extra:
            raise RealizeError("an auxiliary's initial value %r reads %s: only the depth %s, x, y, t"
                               % (text, ", ".join(extra), depth_name))
        self.function = sympy.lambdify([symbols[depth_name], symbols["x"], symbols["y"], symbols["t"]], expression,
                                       "numpy")
        self.depth = depth

    def uOfXT(self, X, t):
        return float(self.function(self.depth.uOfXT(X, t), X[0], X[1], t))


class _OfPoint(object):
    """A coefficient of the coordinates as SWFlow calls it, ``f(X)`` with ``X[0]``, ``X[1]`` the coordinates (of
    one point, or arrays of them), as its decks write ``bathymetry_function(X)``."""

    def __init__(self, code, dim):
        from proteus.ADRProblem import _compile
        self.code = _compile(exact_squares(code))
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


#: SWFlow's boundary data, by the variable of the problem the model's depth and discharge components fill
SHALLOW_WATER_VARIABLES = ("water_height", "x_mom", "y_mom")


def _is_state(condition, unknowns):
    """A condition that states the unknown on the boundary (a Dirichlet condition on a depth or discharge component,
    Cartesian or whole), as against a boundary treatment of the whole boundary (a wall, an outflow)."""
    return (condition["type"] == "dirichlet" and condition["variable"] in unknowns
            and condition.get("component") not in ("normal", "tangential"))


def _state_fields(condition, unknowns):
    """{SWFlow variable: number} a state gives: the components it acts on (``components``, the names ymf gives them;
    else the Cartesian ``component``; else all of the variable's), each with its value."""
    comps = unknowns[condition["variable"]]
    if condition.get("components"):
        where = list(condition["components"])
    elif condition.get("component") in tuple("xyz"):
        axis = "xyz".index(condition["component"])
        if axis >= len(comps):
            raise RealizeError("%s has no %s component (its components are %s)"
                               % (condition["variable"], condition["component"], ", ".join(comps)))
        where = [comps[axis]]
    else:
        where = list(comps)
    value = condition.get("value")
    if isinstance(value, dict):
        values = {c: value[c] for c in where if c in value}
    elif value is not None and len(where) == 1:
        values = {where[0]: value}
    else:
        values = {}
    if set(values) != set(where):
        raise RealizeError("the state of %s on %s has value %r: the shallow-water realizer reads one number per "
                           "component it acts on (%s)" % (condition["variable"], condition.get("part") or "its region",
                                                        condition.get("value"), ", ".join(where)))
    out = {}
    for name, v in values.items():
        try:
            number = float(v)
        except (TypeError, ValueError):
            raise RealizeError("the state of %s on %s is %r, which is not a number (the shallow-water realizer "
                               "imposes constants)" % (name, condition.get("part") or "its region", v)) from None
        out[name] = number
    return out


def _state_function(variable, states):
    """SWFlow's Dirichlet function ``(x, flag) -> None | (x, t) -> value`` of one variable: the number of the first
    state of ``variable`` that holds at the boundary node, else None (no data there)."""
    mine = [st for st in states if st["field"] == variable]

    def dirichlet(x, flag):
        for st in mine:
            if st["holds"](x, flag):
                value = st["value"]
                return lambda x, t: value
        return None
    return dirichlet


def auxiliary_states(states):
    """The states of the dispersive family's auxiliary unknowns, from those of the depth and the discharge, by the rule
    the decks state (obstacle_flow.py, santos_step.py): h eta and h w take the depth's boundary data (h eta = h^2,
    h w = 0 where the depth is given), h beta the discharge's x component's. The spec states the model's own
    unknowns; an auxiliary has no state of its own."""
    out = []
    for st in states:
        if st["field"] == "water_height":
            out.append(dict(st, field="h_times_eta", value=st["value"] ** 2, what="h eta = (%s)^2" % st["what"]))
            out.append(dict(st, field="h_times_w", value=0.0, what="h w = 0 where %s" % st["what"]))
        elif st["field"] == "x_mom":
            out.append(dict(st, field="h_times_beta", what="h beta, as %s" % st["what"]))
    return out


def shallow_water_boundary(split):
    """SWFlow's boundary data for the closed conditions. Data only: ``{"name", "reflectingBCs", "states"}``.

    * The boundary treatment of the whole boundary (an entry of the table's SHALLOW_WATER_BOUNDARY: the walls, the
      outflow), from the conditions that are not states; it may be ``excluding`` the parts the states hold on, and
      holds nowhere else but on the whole boundary (refused otherwise). With states and no treatment the rest of the
      boundary is free: the outflow's ``reflectingBCs`` off.
    * Each state (a Dirichlet condition on the depth or on a discharge component) as the SWFlow variable it sets, its
      number and where it holds (:func:`_holds`: its region, ``part`` and ``excluding``): ymf's round 7 words. Where
      two states of one variable both hold, the first stated wins."""
    (sub,) = split["subproblems"]
    conditions = sub.get("boundary_conditions") or []
    unknowns = sub["unknowns"]
    region = split["domain"].get("region")
    lower, upper = _box(split)
    tol = 1e-8 * max(u - l for u, l in zip(upper, lower))
    for c in conditions:
        if c.get("trace") is not None:
            raise RealizeError("the shallow-water realizer cannot take the boundary value of %s from the trace of "
                               "another sub-problem (trace: %r): a shallow-water discretization has one sub-problem"
                               % (c["variable"], c["trace"]))
    states, treatment = [], []
    for c in conditions:
        (states if _is_state(c, unknowns) else treatment).append(c)
    for c in states:
        if c.get("kind", "strong") != "strong":
            raise RealizeError("SWFlow imposes a state strongly (at the boundary nodes); the state of %s is %s"
                               % (c["variable"], c["kind"]))
    for c in treatment:
        if c.get("where") is not None or c.get("part") is not None:
            raise RealizeError("the shallow-water realizer takes one boundary treatment on the whole boundary (∂Ω): "
                               "a wall or an outflow holds on the boundary less the parts a state is stated on "
                               "(excluding), not on a part")
    if any(c.get("flags") is not None for c in treatment):
        tags = {int(v) for v in (region or {}).get("boundary_tags", {}).values()}
        if any(not tags <= {int(f) for f in c["flags"]} for c in treatment if c.get("flags") is not None):
            raise RealizeError("the shallow-water realizer takes one boundary treatment on the whole boundary: "
                               "name every boundary tag of the mesh (%s) in each condition's region"
                               % ", ".join(sorted(map(str, tags))))
    stated = {c["part"] for c in states if c.get("part") is not None}
    for c in treatment:
        loose = [code for code in c.get("excluding") or [] if code not in stated]
        if loose:
            raise RealizeError("a boundary treatment excludes %s, where no state is stated" % ", ".join(loose))
    if treatment:
        signature = _signature(treatment)
        boundary = next((b for b in table.SHALLOW_WATER_BOUNDARY if _matches(b["signature"], signature)), None)
        if boundary is None:
            raise RealizeError("the boundary conditions %s are none SWFlow imposes; the table knows %s"
                               % (sorted(map(str, signature)),
                                  ", ".join(b["name"] for b in table.SHALLOW_WATER_BOUNDARY)))
        boundary = dict(boundary)
    elif states:
        boundary = {"name": "free (the states only)", "reflectingBCs": False}
    else:
        raise RealizeError("the boundary conditions are none SWFlow imposes; the table knows %s"
                           % ", ".join(b["name"] for b in table.SHALLOW_WATER_BOUNDARY))
    depth = [u for u, comps in unknowns.items() if len(comps) == 1]
    discharge = [u for u, comps in unknowns.items() if len(comps) > 1]
    if depth and discharge:
        variable_of = dict(zip(unknowns[depth[0]] + unknowns[discharge[0]], SHALLOW_WATER_VARIABLES))
    else:
        variable_of = {}
    boundary["states"] = []
    for c in states:
        for name, number in _state_fields(c, unknowns).items():
            if name not in variable_of:
                raise RealizeError("%s: SWFlow solves a depth and a discharge; the sub-problem's unknowns are %s"
                                   % (sub.get("name", "?"), sorted(unknowns)))
            boundary["states"].append({"field": variable_of[name], "value": number, "holds": _holds(c, 2, tol),
                                       "what": "%s = %r%s" % (name, number, "" if c.get("part") is None else
                                                              " on %s" % c["part"])})
    return boundary


def precheck(doc, found, split, ranks=None):
    """What :func:`realize` would refuse that the data alone decides (the family, the domain, the boundary, the
    two-phase physical values, the linear solver asked for against the number of ``ranks`` -- the started Proteus's,
    else 1), so that a runner refuses before anything runs. Raises :class:`RealizeError`."""
    if ranks is None:
        from proteus.ADRProblem import _ranks
        ranks = _ranks()
    if found.family not in table.REALIZERS:
        raise RealizeError("nothing realizes family %s yet (the table realizes %s)"
                           % (found.family, ", ".join(sorted(table.REALIZERS))))
    if found.family == "two_phase_level_set":
        tank_sides(split)
        _two_phase_values(doc, split["dim"], found)
        _linear_solver(_entry(doc, found.discretization), ranks)
    elif found.family in SHALLOW_WATER_FAMILIES:
        _box(split)
        shallow_water_boundary(split)
        for role in list(table.SHALLOW_WATER_COEFFICIENT_ROLES) + [table.SHALLOW_WATER_BED_ROLE]:
            _by_role(doc, role)
        _shallow_water_linear_solver(_entry(doc, found.discretization), ranks)


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
    stored = _stored(split)
    if stored is not None:
        if "h" in realization or "cells" in realization or int(realization.get("levels", 1)) != 1:
            raise RealizeError("%s: a stored mesh is read as it is, one level (mesh: {levels: 1})"
                               % found.discretization)
    elif ("h" in realization) == ("cells" in realization) or int(realization.get("levels", 1)) != 1:
        raise RealizeError("%s: the shallow-water realizer meshes the box by cells per side (SWFlow's structured "
                           "mesh) or by element size (triangle), one level; give mesh: {cells: ...} or "
                           "mesh: {h: ...}" % found.discretization)
    structured = "cells" in realization
    dim = split["dim"]
    if dim != 2 or (split["domain"].get("region") is not None and stored is None):
        raise RealizeError("the shallow-water realizer builds a 2-D box (RectangularDomain) or reads a stored mesh")
    comm = Comm.get()
    final_time, outputs = _time(split, entry)
    solves, how = _shallow_water_linear_solver(entry, ranks)

    (sub,) = split["subproblems"]
    boundary = shallow_water_boundary(split)
    states = boundary["states"]
    if found.family == "dispersive_shallow_water":
        states = states + auxiliary_states(states)

    # the domain: SWFlow's unstructured path, a RectangularDomain's PSLG meshed by triangle at h
    lower, upper = [float(v) for v in split["domain"]["lower"]], [float(v) for v in split["domain"]["upper"]]
    rectangle = Domain.RectangularDomain(L=[u - l for u, l in zip(upper, lower)], x=lower)
    if stored is not None:
        for suffix in (".node", ".ele"):
            if not os.path.exists(stored + suffix):
                raise RealizeError("the stored mesh %s%s does not exist" % (stored, suffix))
        he = stored_element_size(stored)
        nnx = nny = None
        domain = Domain.PlanarStraightLineGraphDomain(fileprefix=os.path.abspath(stored))
        domain.MeshOptions.genMesh = False
        domain.polyfile = os.path.abspath(stored)
    elif structured:
        # SWFlow's structured path (the decks' structured=True): the rectangle itself, cells per side as the decks'
        # nnx - 1 = nny - 1, he the deck's L[0]/(nnx - 1)
        from ymf.discretizations import cells_per_axis
        nnx, nny = [c + 1 for c in cells_per_axis(realization["cells"], dim)]
        he = (upper[0] - lower[0]) / float(nnx - 1)
        domain = rectangle
    else:
        he = float(realization["h"])
        nnx = nny = None
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
    auxiliary_record = {}
    if found.family == "dispersive_shallow_water":
        # the relaxation's auxiliaries (h eta, h w, h beta) start from the spec's initial values: their defaults are
        # the roles' (h^2, 0, 0), a function of the depth's initial value
        dispersion = sub.get("dispersion") or {}
        by_role = {a["role"]: a for a in dispersion.get("auxiliaries") or []}
        for role, variable in table.SHALLOW_WATER_AUXILIARIES:
            if role not in by_role:
                raise RealizeError("%s: the dispersive family needs an auxiliary unknown with role %s"
                                   % (model.subproblem, role))
            variables.append(variable)
            auxiliary_record[variable] = "%s = %s" % (by_role[role]["name"], by_role[role]["initial"])
            initial[variable] = _AuxiliaryInitial(by_role[role]["initial"], depth[0], initial["water_height"], dim)
    physical = dict(SWF.default_physical_parameters)
    for role, key in table.SHALLOW_WATER_COEFFICIENT_ROLES.items():
        physical[key] = _shallow_water_value(doc, _by_role(doc, role), dim)
    bathymetry = _OfPoint(_coefficient_code(doc, _by_role(doc, table.SHALLOW_WATER_BED_ROLE), dim), dim)

    problem = SWF.SWFlowProblem(
        sw_model=models["sw_model"], cfl=found.system["runCFL"],
        outputStepping=SWF.OutputStepping(final_time, dt_output=final_time / outputs),
        structured=structured, he=he, nnx=nnx, nny=nny, domain=domain, initialConditions=initial,
        boundaryConditions={v: _state_function(v, states) for v in variables},
        reflectingBCs=boundary["reflectingBCs"], bathymetry=bathymetry, genMesh=stored is None)
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
    gauges = gauge_objects(doc, _gauge_owners(found, {model.subproblem: sub},
                                              lambda m: list(p.coefficients.variableNames)))
    if gauges:
        n.auxiliaryVariables = gauges[model.subproblem]
    petsc_options, unused = _check_petsc_options(found, split, entry, ranks)
    _apply_petsc_options(petsc_options, {model.subproblem: n.linear_solver_options_prefix or ""})
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
    record = {"boundary": {"whole boundary": boundary["name"], "reflectingBCs": boundary["reflectingBCs"],
                           **({"states": [st["what"] for st in states]} if states else {})},
              "initial": {**dict(zip(variables, components)), **auxiliary_record},
              "system": {"final_time": final_time, "outputs": outputs, "mesh": ("stored %s, h = %g" % (stored, he) if stored is not None else
                       "structured, %d x %d nodes" % (nnx, nny) if structured else "triangle, %s" % (
                  domain.MeshOptions.triangleOptions,)),
                  "linear_solver": "%s (%s)" % (solves, how),
                  **{k: (v if isinstance(v, float) else "f(x, y)") for k, v in physical.items()
                     if k in ("gravity", "mannings", "porosity", "drag_alpha", "drag_beta")}}}
    if gauges:
        record["system"]["gauges"] = {s: [g.fileName for g in items] for s, items in gauges.items()}
    if petsc_options:
        record["system"]["petsc_options"] = {m: dict(o) for m, o in petsc_options.items()}
    if unused:
        record["system"]["petsc_options_unused"] = unused
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
