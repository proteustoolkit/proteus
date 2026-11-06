"""From a closed ymf spec to the Proteus modules and flags that solve it (C5 step 1).

A ymf spec states the model (physics), the formulation (a ``weak_forms`` entry the discretization names with
``from_weak_form``) and the discretization: spaces, time integration, stabilization, the split into sub-problems,
the symbols it binds (``h``, ``ε = 1.5 h``) and how it regularizes the model's sharp functions. :func:`dispatch` reads
one discretization of such a spec and returns, for each sub-problem in solve order, the mprans module and the
attributes of its physics and numerics modules, using the table in :mod:`proteus.DispatchTable`
(YMF_MODEL_FAMILIES_C42 section 6: D1, D8).

* A sub-problem no cell covers is refused by name (:class:`DispatchError`), with what each candidate cell differs
  in. It is never mapped onto the nearest module: that would solve a different discretization under this one's name.
* An interim ``proteus:`` hint on a sub-problem (D8) is checked against what the dispatcher infers: a disagreement
  is refused, a hint on a flag the spec does not state replaces the table's default, and a hint the cell does not
  know is refused.
* Every value says where it came from (``sources``): ``cell`` (the table's cell), ``spec`` (a typed key of the
  spec), ``default`` (the table's default, which the spec left open), or ``hint``.

The result is a description (dotted names, numbers), not Proteus objects; :meth:`Dispatch.resolve` imports a
name. Building a ``pnList`` from it and running the solve is the next step (through ``ymf_run``).
"""
import copy
import importlib
import re

from proteus import DispatchTable as table

__all__ = ["DispatchError", "Dispatched", "Dispatch", "dispatch", "dispatch_file", "role_of", "family_of"]


class DispatchError(ValueError):
    """A spec the table does not cover, or a hint that disagrees with it. ``subproblem`` and ``cell`` name where."""

    def __init__(self, message, subproblem=None, cell=None):
        super().__init__(message)
        self.subproblem = subproblem
        self.cell = cell


def resolve(name, module=None):
    """Import a dotted name; ``{module}`` stands for ``module``. None and non-strings are returned as they are."""
    if not isinstance(name, str):
        return name
    if "{module}" in name:
        name = name.replace("{module}", module)
    path, _, attribute = name.rpartition(".")
    return getattr(importlib.import_module(path), attribute)


class Dispatched:
    """One sub-problem's module and attributes.

    ``coefficients``: flags of the module's ``Coefficients``; ``physics``: other attributes of the physics module
    (``LevelModelType``, weak Dirichlet conditions); ``numerics``: attributes of the numerics module;
    ``spaces``: the element space of each slot (``velocity``, ``pressure``, ``fields``) and ``components`` the slot
    of each component in order (``velocity`` stands for one component per space dimension).
    """

    def __init__(self, index, subproblem, role, cell):
        self.index = index
        self.subproblem = subproblem
        self.role = role
        self.cell = cell["cell"]
        self.module = cell["module"]
        self.name = cell["name"]
        self.coefficients_class = cell["physics"]["coefficients"]
        self.coefficients = {}
        self.physics = {}
        self.numerics = {}
        self.spaces = {}
        self.components = list(cell["components"])
        self.sources = {}

    def femSpaces(self, nd):
        """``{component: basis name}`` in a space of ``nd`` dimensions."""
        slots = []
        for slot in self.components:
            slots += [slot] * (nd if slot == "velocity" else 1)
        return {i: table.SPACES[self.spaces[slot]] for i, slot in enumerate(slots)}

    def resolve(self, name):
        return resolve(name, self.module)

    def as_dict(self):
        return {"index": self.index, "subproblem": self.subproblem, "role": self.role, "cell": self.cell,
                "module": self.module, "name": self.name, "coefficients_class": self.coefficients_class,
                "coefficients": dict(self.coefficients), "physics": dict(self.physics),
                "numerics": dict(self.numerics),
                "spaces": {slot: "%s%d" % space for slot, space in self.spaces.items()},
                "components": list(self.components), "sources": dict(self.sources)}

    def __repr__(self):
        return "Dispatched(%s -> %s [%s])" % (self.subproblem, self.module, self.cell)


class Dispatch:
    """A discretization dispatched: its family, formulation, the sub-problems' modules in solve order, and the
    system's step control (``system``)."""

    def __init__(self, discretization, family, formulation):
        self.discretization = discretization
        self.family = family
        self.formulation = formulation
        self.models = []
        self.system = {}
        self.sources = {}

    def __iter__(self):
        return iter(self.models)

    def __getitem__(self, subproblem):
        for model in self.models:
            if model.subproblem == subproblem:
                return model
        raise KeyError(subproblem)

    @property
    def modules(self):
        return [model.module for model in self.models]

    resolve = staticmethod(resolve)

    def as_dict(self):
        return {"discretization": self.discretization, "family": self.family, "formulation": self.formulation,
                "models": [m.as_dict() for m in self.models], "system": dict(self.system),
                "sources": dict(self.sources)}


# ---------------------------------------------------------------------------------------------------------------
# What the model says
# ---------------------------------------------------------------------------------------------------------------

_HEAVISIDE = re.compile(r"^\s*heaviside\(\s*([^\s()]+)\s*\)\s*$")


class _Model:
    """The facts about the model the roles are read from."""

    def __init__(self, doc):
        model = doc.get("model") or {}
        strong = model.get("strong_form") or {}
        self.physical = model.get("physical_model") or {}
        self.unknowns = {}
        for u in strong.get("unknowns") or []:
            if isinstance(u, str):
                self.unknowns[u] = 0
            else:
                self.unknowns[u["name"]] = int(u.get("rank") or 0)
        self.equations = {}
        for i, e in enumerate(strong.get("equations") or []):
            if isinstance(e, dict):
                self.equations[e.get("name", i)] = e
        self.coefficients = strong.get("coefficients") or {}
        #: {coefficient: unknown} for each coefficient that is heaviside(unknown): χ -> φ
        self.heaviside_of = {}
        for name, entry in self.coefficients.items():
            formula = entry.get("formula") if isinstance(entry, dict) else entry
            m = _HEAVISIDE.match(str(formula or ""))
            if m and m.group(1) in self.unknowns:
                self.heaviside_of[name] = m.group(1)
        self.level_sets = set(self.heaviside_of.values())
        self.weak_forms = [w.get("label") for w in model.get("weak_forms") or []]

    def solved_for(self, sub):
        return [self.equations[e].get("for") for e in sub.get("solves") or [] if e in self.equations]


def family_of(doc):
    """The table's family of the spec's model, by its physical_model; refused if none or several match."""
    physical = _Model(doc).physical
    have = {"processes": set(physical.get("processes") or []),
            "assumptions": set(physical.get("assumptions") or [])}
    found = [name for name, needs in table.FAMILIES.items()
             if all(set(needs.get(key) or []) <= have[key] for key in ("processes", "assumptions"))]
    if len(found) != 1:
        known = "; ".join("%s (%s)" % (name, ", ".join("%s %s" % (key, "+".join(v)) for key, v in needs.items()))
                          for name, needs in table.FAMILIES.items())
        raise DispatchError(
            "the model's physical_model (processes %s, assumptions %s) %s; the table knows %s"
            % (sorted(have["processes"]), sorted(have["assumptions"]),
               "matches no family" if not found else "matches several families (%s)" % ", ".join(found), known))
    return found[0]


def _normal(text):
    return re.sub(r"\s+", "", str(text or ""))


def _is_eikonal(enforces, model):
    text = _normal(enforces)
    return any(text in ("norm(∇%s)=1" % phi, "|∇%s|=1" % phi) for phi in model.level_sets)


def _is_divergence_free(enforces, model):
    text = _normal(enforces)
    return any(rank == 1 and text == "∇·%s=0" % name for name, rank in model.unknowns.items())


def role_of(sub, model, family):
    """What one sub-problem does, in the table's vocabulary (``DispatchTable.ROLES``); refused if none fits."""
    solved = model.solved_for(sub)
    vectors = [u for u in solved if model.unknowns.get(u) == 1]
    scalars = [u for u in solved if model.unknowns.get(u) == 0]
    new = sub.get("introduces") or {}
    enforces = sub.get("enforces")
    role = None
    if family == "two_phase_level_set":
        if solved:
            if len(vectors) == 1 and len(scalars) == 1 and not set(scalars) & model.level_sets:
                role = "velocity_pressure"
            elif len(vectors) == 1 and not scalars:
                role = "velocity"
            elif not vectors and len(scalars) == 1 and scalars[0] in model.level_sets:
                role = "level_set_redistanced" if _is_eikonal(enforces, model) else "level_set"
            elif not vectors and len(scalars) == 1:
                role = "pressure"
        elif new.get("approximates") in model.heaviside_of:
            role = "volume_fraction"
        elif new.get("correction_of") in model.level_sets:
            role = "mass_correction"
        elif _is_eikonal(enforces, model):
            role = "redistancing"
        elif _is_divergence_free(enforces, model):
            role = "pressure_increment"
        elif sub.get("spin_up"):
            role = "pressure_initial"
    elif family in ("shallow_water", "dispersive_shallow_water"):
        if solved and set(solved) == set(model.unknowns):
            role = "shallow_water"
    if role is None:
        said = ", ".join("%s %s" % (key, sub[key]) for key in ("solves", "enforces", "introduces", "spin_up")
                         if sub.get(key)) or "nothing the table reads"
        raise DispatchError(
            "sub-problem %s (%s) has no role the table knows in family %s; roles: %s"
            % (sub.get("name"), said, family, ", ".join(table.ROLES[family])), subproblem=sub.get("name"))
    return role


# ---------------------------------------------------------------------------------------------------------------
# What the discretization says
# ---------------------------------------------------------------------------------------------------------------

_FACTOR = re.compile(r"^\s*([0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)\s*\*?\s*(\S+)\s*$")


def _bindings(entry, sub):
    out = dict(entry.get("binds") or {})
    out.update(sub.get("binds") or {})
    return out


def _element_size_factor(symbol, entry, sub, what):
    """``k`` for a bound symbol whose formula is ``k h`` with ``h`` the element size."""
    binds = _bindings(entry, sub)
    if symbol not in binds:
        raise DispatchError("%s: %s's width %s is not bound by the discretization" % (sub["name"], what, symbol),
                            subproblem=sub["name"])
    binding = binds[symbol] or {}
    if binding.get("meaning") == "element_size":
        return 1.0
    m = _FACTOR.match(str(binding.get("formula") or ""))
    if m and (binds.get(m.group(2)) or {}).get("meaning") == "element_size":
        return float(m.group(1))
    raise DispatchError("%s: %s = %r; the table reads a width as a number times the element size (a symbol bound "
                        "with meaning: element_size)" % (sub["name"], symbol, binding.get("formula")),
                        subproblem=sub["name"])


def _width(sub, function, entry):
    """The width factor of a regularized function (closed specs merge the entry's regularize into each
    sub-problem's)."""
    rule = (sub.get("regularize") or entry.get("regularize") or {}).get(function)
    if not rule or not rule.get("width"):
        raise DispatchError(
            "%s: the model's %s is sharp, and the discretization does not regularize it (regularize: {%s: {by: ..., "
            "width: ...}})" % (sub["name"], function, function), subproblem=sub["name"])
    return _element_size_factor(rule["width"], entry, sub, function)


def _stabilization(sub):
    s = sub.get("stabilization")
    if s is None:
        return {}
    if isinstance(s, str):
        return {"method": s}
    return _value(dict(s))


def _time_of(sub):
    if sub.get("time_integration") is not None:
        return str(sub["time_integration"])
    return None


def _scheme(entry):
    block = entry.get("discretization")
    return _value(dict(block)) if isinstance(block, dict) else {}


def _options(sub, entry, formulation):
    """What the sub-problem states, for matching and for the refusal message."""
    fe = sub.get("finite_element") or {}
    spaces = {slot: (str(v.get("family")), int(v.get("order"))) for slot, v in fe.items() if isinstance(v, dict)}
    return {"formulation": formulation, "spaces": spaces, "time_integration": _time_of(sub),
            "stabilization": _stabilization(sub).get("method"), "scheme": _scheme(entry).get("scheme")}


def _describe(options):
    parts = ["formulation %s" % options["formulation"]]
    parts += ["%s %s%d" % ((slot,) + space) for slot, space in sorted(options["spaces"].items())]
    for key in ("time_integration", "stabilization", "scheme"):
        if options[key] is not None:
            parts.append("%s %s" % (key.replace("_", " "), options[key]))
    return ", ".join(parts)


def _mismatch(cell, options):
    """Why ``cell`` does not cover ``options``, or None."""
    if cell["formulation"] != table.AS_WRITTEN and options["formulation"] != cell["formulation"]:
        return "formulation %s, the cell's is %s" % (options["formulation"], cell["formulation"])
    for slot, space in options["spaces"].items():
        if slot not in cell["spaces"]:
            return "the spec gives a %s space, the cell has %s" % (slot, ", ".join(cell["spaces"]))
        if space not in cell["spaces"][slot]["options"]:
            return "%s is %s%d, the cell has %s" % (
                slot, space[0], space[1], ", ".join("%s%d" % s for s in cell["spaces"][slot]["options"]))
    t = options["time_integration"]
    if t is not None and t not in cell["time_integration"]["options"]:
        return "time integration %s, the cell has %s" % (t, ", ".join(cell["time_integration"]["options"]))
    s = options["stabilization"]
    if s is not None and s not in (cell.get("stabilization") or {}).get("options", []):
        return "stabilization %s, the cell has %s" % (
            s, ", ".join((cell.get("stabilization") or {}).get("options", [])) or "none")
    k = options["scheme"]
    if k is not None and "scheme" in cell and k not in cell["scheme"]["options"]:
        return "scheme %s, the cell has %s" % (k, ", ".join(cell["scheme"]["options"]))
    return None


def _cell(family, role, sub, options):
    candidates = [c for c in table.CELLS if c["family"] == family and c["role"] == role]
    if not candidates:
        raise DispatchError("no Proteus module covers %s: the table has no cell for role %s in family %s"
                            % (sub["name"], role, family), subproblem=sub["name"])
    reasons = [(c["cell"], _mismatch(c, options)) for c in candidates]
    found = [c for c, (_, why) in zip(candidates, reasons) if why is None]
    if not found:
        raise DispatchError(
            "no Proteus module covers %s (%s, %s; %s). Cells for this role: %s"
            % (sub["name"], family, role, _describe(options),
               "; ".join("%s: %s" % (name, why) for name, why in reasons)),
            subproblem=sub["name"], cell=(family, role, _describe(options)))
    if len(found) > 1:
        raise RuntimeError("DispatchTable: cells %s all cover %s; the table is ambiguous"
                           % (", ".join(c["cell"] for c in found), sub["name"]))
    return found[0]


# ---------------------------------------------------------------------------------------------------------------
# Building one sub-problem's attributes
# ---------------------------------------------------------------------------------------------------------------

def _set(d, where, key, value, source):
    where[key] = value
    d.sources[key] = source


def _build(index, sub, role, cell, entry, model, options):
    d = Dispatched(index, sub["name"], role, cell)
    physics, numerics = cell["physics"], cell["numerics"]
    stab = _stabilization(sub)
    stab_cell = cell.get("stabilization") or {}

    # spaces: stated or the cell's default
    for slot, spec in cell["spaces"].items():
        stated = options["spaces"].get(slot)
        d.spaces[slot] = stated or spec["default"]
        d.sources["space:" + slot] = "spec" if stated else "default"

    # coefficients
    for key, value in (physics.get("fixed") or {}).items():
        _set(d, d.coefficients, key, value, "cell")
    for key, value in (physics.get("default") or {}).items():
        _set(d, d.coefficients, key, value, "default")
    if stab_cell.get("flags"):
        method = stab.get("method") or stab_cell["default"]
        for key, value in stab_cell["flags"][method].items():
            _set(d, d.coefficients, key, value, "spec" if stab.get("method") else "default")
    for key in physics.get("from_heaviside_width") or []:
        _set(d, d.coefficients, key, _width(sub, "heaviside", entry), "spec")
    for key in physics.get("from_sign_width") or []:
        _set(d, d.coefficients, key, _width(sub, "sign", entry), "spec")
    for symbol, key in (physics.get("from_binding") or {}).items():
        _set(d, d.coefficients, key, _element_size_factor(symbol, entry, sub, key), "spec")
    if physics.get("from_carrier_lag"):
        lags = sub.get("lags") or {}
        carried = any(name in model.heaviside_of or (isinstance(lag, dict) and lag.get("of") in model.heaviside_of)
                      for name, lag in lags.items())
        _set(d, d.coefficients, physics["from_carrier_lag"], 1.0 if carried else 0.0, "spec")
    sc_spec = stab.get("shock_capturing") or {}
    sc_cell = stab_cell.get("shock_capturing") or {}
    for key in physics.get("from_shock_capturing") or []:
        if key in sc_spec:
            _set(d, d.coefficients, key, float(sc_spec[key]), "spec")
        elif key in sc_cell:
            _set(d, d.coefficients, key, sc_cell[key], "default")
    scheme = _scheme(entry)
    for key, (flag, kind) in (physics.get("from_scheme") or {}).items():
        if key in scheme:
            _set(d, d.coefficients, flag, kind(scheme[key]), "spec")
    if "scheme" in cell:
        # the scheme's block is free-form in ymf: an option the cell does not read must not be dropped silently
        # (outputs is the runner's, as in ymf_run's ADR path: how many evenly spaced archive frames)
        unread = sorted(set(scheme) - set(physics.get("from_scheme") or {})
                        - {"scheme", "time_integration", "outputs"})
        if unread:
            raise DispatchError("%s: cell %s does not read the scheme option(s) %s; it reads %s"
                                % (sub["name"], cell["cell"], ", ".join(unread),
                                   ", ".join(sorted(physics.get("from_scheme") or {})) or "none"),
                                subproblem=sub["name"], cell=cell["cell"])
        stated = scheme.get("time_integration")
        if stated is not None and stated != (options["time_integration"] or cell["time_integration"]["default"]):
            raise DispatchError("%s: the scheme says time integration %s and the sub-problem %s"
                                % (sub["name"], stated, options["time_integration"]),
                                subproblem=sub["name"], cell=cell["cell"])

    # other attributes of the physics module
    if physics.get("LevelModelType"):
        _set(d, d.physics, "LevelModelType", physics["LevelModelType"], "cell")
    for key, value in (physics.get("attributes") or {}).items():
        _set(d, d.physics, key, value, "cell")

    # numerics
    for key, value in (numerics.get("fixed") or {}).items():
        _set(d, d.numerics, key, value, "cell")
    for key, value in (numerics.get("default") or {}).items():
        _set(d, d.numerics, key, value, "default")
    time = options["time_integration"] or cell["time_integration"]["default"]
    for key, value in cell["time_integration"]["options"][time].items():
        _set(d, d.numerics, key, value, "spec" if options["time_integration"] else "default")
    if d.numerics.get("subgridError"):
        lag = stab.get("lag", stab_cell.get("subgrid_lag"))
        if lag is not None:
            _set(d, d.numerics, "subgridError.lag", bool(lag), "spec" if "lag" in stab else "default")
        if "velocity" in d.spaces:
            _set(d, d.numerics, "subgridError.hFactor", table.H_FACTOR[d.spaces["velocity"][1]], "spec")
    if d.numerics.get("shockCapturing"):
        for key, attribute in (("factor", "shockCapturingFactor"), ("lag", "lag")):
            if key in sc_spec:
                _set(d, d.numerics, "shockCapturing." + attribute, sc_spec[key], "spec")
            elif key in sc_cell:
                _set(d, d.numerics, "shockCapturing." + attribute, sc_cell[key], "default")
    solver = sub.get("solver") or {}
    if solver.get("max_iterations") is not None:
        _set(d, d.numerics, "maxNonlinearIts", int(solver["max_iterations"]), "spec")
    for flag, by_value in (numerics.get("by_flag") or {}).items():
        for key, value in by_value[d.coefficients[flag]].items():
            _set(d, d.numerics, key, value, d.sources[flag])

    _apply_hint(d, sub, cell)
    # a hint can change a flag a by_flag rule reads
    for flag, by_value in (numerics.get("by_flag") or {}).items():
        for key, value in by_value[d.coefficients[flag]].items():
            if d.numerics.get(key) != value:
                _set(d, d.numerics, key, value, d.sources[flag])
    return d


def _value(text):
    """A hint's value: strictyaml reads a free-form (``Any``) block as strings."""
    if isinstance(text, dict):
        return {_value(k): _value(v) for k, v in text.items()}
    if isinstance(text, list):
        return [_value(v) for v in text]
    if not isinstance(text, str):
        return text
    if text.lower() in ("true", "false"):
        return text.lower() == "true"
    for kind in (int, float):
        try:
            return kind(text)
        except ValueError:
            pass
    return text


def _same(a, b):
    if isinstance(a, bool) or isinstance(b, bool):
        return bool(a) == bool(b)
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return float(a) == float(b)
    return a == b


def _apply_hint(d, sub, cell):
    """D8: the sub-problem's ``proteus:`` block, checked against what dispatch inferred."""
    hint = sub.get("proteus")
    if not hint:
        return
    if not isinstance(hint, dict):
        raise DispatchError("%s: the proteus: hint must be a map, not %r" % (sub["name"], hint),
                            subproblem=sub["name"], cell=cell["cell"])
    short = cell["module"].rpartition(".")[2]
    for key, value in _value(dict(hint)).items():
        if key == "model":
            if str(value) != short:
                raise DispatchError(
                    "%s: the proteus: hint says model %s, and dispatch infers %s (cell %s, role %s); the hint "
                    "disagrees with the spec" % (sub["name"], value, short, cell["cell"], d.role),
                    subproblem=sub["name"], cell=cell["cell"])
            d.sources["model"] = "spec, hint agrees"
            continue
        if key not in cell["hints"]:
            raise DispatchError(
                "%s: the proteus: hint sets %s, which cell %s (%s) does not know; it takes %s"
                % (sub["name"], key, cell["cell"], short, ", ".join(cell["hints"]) or "no hints"),
                subproblem=sub["name"], cell=cell["cell"])
        where = d.coefficients if key in d.coefficients or key not in d.numerics else d.numerics
        source = d.sources.get(key)
        if source in ("cell", "spec"):
            if not _same(where[key], value):
                raise DispatchError(
                    "%s: the proteus: hint sets %s = %r, and dispatch infers %r from the %s (cell %s); the hint "
                    "disagrees" % (sub["name"], key, value, where[key],
                                   "spec" if source == "spec" else "table", cell["cell"]),
                    subproblem=sub["name"], cell=cell["cell"])
            d.sources[key] = source + ", hint agrees"
        else:
            _set(d, where, key, value, "hint")


# ---------------------------------------------------------------------------------------------------------------
# The discretization
# ---------------------------------------------------------------------------------------------------------------

def _entry(doc, name):
    entries = [e for e in (doc.get("solution_paths") or {}).get("discretizations") or [] if isinstance(e, dict)]
    if name is None:
        if len(entries) != 1:
            raise DispatchError("the spec has %d discretizations (%s); name one"
                                % (len(entries), ", ".join(e.get("name", "?") for e in entries)))
        return entries[0]
    for entry in entries:
        if entry.get("name") == name:
            return entry
    raise DispatchError("the spec has no discretization %r (it has %s)"
                        % (name, ", ".join(e.get("name", "?") for e in entries)))


def _formulation(entry, model):
    label = entry.get("from_weak_form")
    if label is None and len(model.weak_forms) == 1:
        label = model.weak_forms[0]
    return str(label) if label is not None else table.AS_WRITTEN


def _subproblems(entry, model):
    subs = entry.get("subproblems")
    if subs:
        return [dict(s) for s in subs]
    # one system: a single sub-problem that solves every named equation
    sub = {"name": entry["name"], "solves": [e for e in model.equations if isinstance(e, str)]}
    for key in ("finite_element", "solver", "regularize", "binds", "stabilization", "proteus"):
        if entry.get(key) is not None:
            sub[key] = entry[key]
    return [sub]


def dispatch(doc, discretization=None):
    """Dispatch one discretization of a ymf spec (``doc``, composed; closed or not: it is closed here, which is
    idempotent). Returns a :class:`Dispatch`; raises :class:`DispatchError` for what the table does not cover."""
    from ymf.explicit import explicit
    doc = explicit(copy.deepcopy(doc))
    entry = _entry(doc, discretization)
    model = _Model(doc)
    family = family_of(doc)
    formulation = _formulation(entry, model)
    if formulation != table.AS_WRITTEN and formulation not in table.FORMULATIONS:
        raise DispatchError(
            "discretization %s solves weak form %s, a formulation the table does not know (it knows %s)"
            % (entry["name"], formulation, ", ".join(table.FORMULATIONS)), cell=(family, formulation))
    result = Dispatch(entry["name"], family, formulation)
    cells = []
    subs = _subproblems(entry, model)
    for index, sub in enumerate(subs):
        role = role_of(sub, model, family)
        options = _options(sub, entry, formulation)
        cell = _cell(family, role, sub, options)
        cells.append(cell)
        result.models.append(_build(index, sub, role, cell, entry, model, options))

    # what a cell needs beside it (RANS3PF's pressure solves)
    roles = {m.role for m in result.models}
    for cell in cells:
        missing = [r for r in cell.get("requires") or [] if r not in roles]
        if missing:
            raise DispatchError("%s (%s) needs sub-problems with roles %s beside it, and %s has none"
                                % (cell["cell"], cell["module"], ", ".join(missing), entry["name"]),
                                cell=cell["cell"])

    # quadrature: from the highest order of the discretization's spaces, or cut-cell integration
    order = max(space[1] for m in result.models for space in m.spaces.values())
    exact = any(m.coefficients.get("useExact") for m in result.models)
    for m in result.models:
        _set(m, m.numerics, "quadrature_order",
             table.QUADRATURE_ORDER_EXACT if exact else table.QUADRATURE_ORDER[order], "spec")

    # the system
    control = entry.get("step_control") or {}
    kind = control.get("type", "min_cfl")
    if (family, kind) not in table.STEP_CONTROL:
        raise DispatchError("step_control type %s: the table has no system step controller for it in family %s"
                            % (kind, family), cell=(family, kind))
    result.system["systemStepControllerType"] = table.STEP_CONTROL[(family, kind)]
    result.sources["systemStepControllerType"] = "spec" if "type" in control else "default"
    if "cfl" in control:
        result.system["runCFL"] = float(control["cfl"])
        result.sources["runCFL"] = "spec"
    if "step_exact" in control:
        result.system["systemStepExact"] = bool(control["step_exact"])
        result.sources["systemStepExact"] = "spec"
    spin_up = [m.index for m, sub in zip(result.models, subs) if sub.get("spin_up")]
    if spin_up:
        # spin-up solves run before the first step and are left out of each step's sequence
        result.system["modelSpinUpList"] = spin_up
        result.sources["modelSpinUpList"] = "spec"
    return result


def dispatch_file(path, discretization=None):
    """:func:`dispatch` a spec file (ymf.closure.load: composed, validated)."""
    from ymf.closure import load
    spec, _, _ = load(path)
    return dispatch(spec, discretization)
