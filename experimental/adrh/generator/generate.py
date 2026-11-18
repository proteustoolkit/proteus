#!/usr/bin/env python3
"""Emit ADRH engine coefficient functors from ymf's plain-data sub-problem records (generator spike 2026-11-16, slices round 2026-11-17).

Input: JSON, one file per kernel and dimension, written by spec.py from ``ymf.symbolic.split_problem``'s sub-problem record
(no sympy and no ymf needed here: the trust boundary of ymf.symbolic is kept). The generator reads what the spec states and
chooses nothing by the model's name:

    "kernel"       the functor's name, <kernel>Coefficients          "dim"   the space dimension
    "adr"          one scalar equation as ymf's ADR form: m, dm, f, df, H, dH, r, dr, diffusion a, da (numpy code strings)
    "reads"        the fields other sub-problems supply, in order    -> per-point inputs fld[i] (their components)
    "mesh_symbols" the spec's mesh symbols (h: element_size)         -> per-point inputs after the reads (the driver says which size)
    "stabilization"  the resolved block (ymf.stabilization.resolve)   -> template, tau form, shock capturing constants
    "boundary_conditions"  the conditions with their `kind`           -> the face kernel the functor selects
    "coefficient_roles", "coefficients", "named", "lagged_coefficients"   -> a lagged state coefficient (dual numbers)

Output: C++ with, for each kernel and dimension, a functor of one of the engine's two forms:

* a stabilization is stated (SUPG): `template <int nS> struct <K>Coefficients` for scalarSUPGElement and the scalar face kernels
  (vof.h): nSpace, nFields, hasF/hasH/hasR/hasDR/hasA/hasDDH, tauForm, face, the shock-capturing constants, evaluate(fld, u, gu, x, T);
* none is stated (Galerkin): the engine.h functor of MCorr's kind: nc, nSpace, nFields, fields/nQ pointers, bindElement/bindPoint,
  evaluate(t, x, u, gu, Terms).

What the engine's templates cannot take is refused (GeneratorError): every blocker is collected and named at once, see
GENERATOR_SPIKE_REPORT_2026-11-16.md and HANDOFF_2026-11-17_c19-slices.md. Usage: generate.py out.h in1.json [in2.json ...]
"""
import ast
import json
import re
import sys


class GeneratorError(Exception):
    pass


_FUNCS = {"sqrt": "fn::sqrt", "sin": "fn::sin", "cos": "fn::cos", "tan": "fn::tan", "exp": "fn::exp",
          "log": "fn::log", "absolute": "fn::fabs", "abs": "fn::fabs", "tanh": "fn::tanh", "sinh": "fn::sinh",
          "cosh": "fn::cosh"}
_CMP = {"less": "<", "greater": ">", "less_equal": "<=", "greater_equal": ">="}
_COORDINATES = ("x", "y", "z")


def _attr(node):
    return node.attr if isinstance(node, ast.Attribute) else None


class _CSE:
    """Common subexpressions of one functor's expressions: a function call (sqrt, sin, a select, ...) that occurs more than once in
    the code strings of one evaluate is computed once into a temporary. ymf prints every slot as an independent string (RDLS's
    smoothed sign appears in H, dH x2 and r, the norm of grad u six times); the compiler does not merge libm calls. The value of
    each is the same expression evaluated once, so the result does not change."""

    def __init__(self, texts):
        self.count, self.names, self.decls = {}, {}, []
        for text in texts:
            self._count(ast.parse(text, mode="eval").body)

    def _count(self, n):
        # a call inside a select's branches is not hoisted: the branch is taken at some points only (RDLS's sin is evaluated only
        # inside the smoothing band), so the select itself is the unit
        if self.heavy(n):
            key = ast.dump(n)
            self.count[key] = self.count.get(key, 0) + 1
            if _attr(n.func) == "select":
                return
        for c in ast.iter_child_nodes(n):
            self._count(c)

    @staticmethod
    def heavy(n):
        return isinstance(n, ast.Call) and _attr(n.func) in set(_FUNCS) | {"select"}

    def hoist(self, n):
        return self.heavy(n) and self.count.get(ast.dump(n), 0) > 1


class _Translator:
    """numpy source (the strings ymf.symbolic.adr.to_code prints) -> a C++ expression with the same association order."""

    def __init__(self, names, where, cse=None):
        self.names, self.where, self.cse = names, where, cse
        self.in_select = 0

    def __call__(self, text):
        return self.visit(ast.parse(text, mode="eval").body)

    def visit(self, n):
        is_select = isinstance(n, ast.Call) and _attr(n.func) == "select"
        if self.cse is not None and not self.in_select and self.cse.hoist(n):
            key = ast.dump(n)
            if key not in self.cse.names:
                self.in_select += is_select
                expr = self.visit_(n)
                self.in_select -= is_select
                self.cse.names[key] = "t_%d" % len(self.cse.decls)
                self.cse.decls.append("const double %s = %s;" % (self.cse.names[key], expr))
            return self.cse.names[key]
        self.in_select += is_select
        try:
            return self.visit_(n)
        finally:
            self.in_select -= is_select

    def fail(self, what):
        raise GeneratorError("%s: %s" % (self.where, what))

    def visit_(self, n):
        if isinstance(n, ast.Constant):
            if isinstance(n.value, bool) or not isinstance(n.value, (int, float)):
                self.fail("constant %r" % (n.value,))
            return repr(float(n.value))
        if isinstance(n, ast.Name):
            if n.id not in self.names:
                self.fail("the name %s is not an unknown, a gradient, a field the sub-problem reads, a coordinate or a mesh "
                          "symbol of the spec" % n.id)
            return self.names[n.id]
        if isinstance(n, ast.Attribute) and n.attr == "pi":
            return "3.14159265358979323846"
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.USub):
            return "(-%s)" % self.visit(n.operand)
        if isinstance(n, ast.BinOp):
            a, b = self.visit(n.left), self.visit(n.right)
            if isinstance(n.op, ast.Pow):
                e = n.right
                if isinstance(e, ast.Constant) and e.value == 2:
                    return "(%s * %s)" % (a, a) if re.fullmatch(r"[\w\[\]\.]+", a) else "fn::sq(%s)" % a
                if isinstance(e, ast.Constant) and e.value == 0.5:
                    return "fn::sqrt(%s)" % a
                return "fn::pow(%s, %s)" % (a, b)
            op = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/"}.get(type(n.op))
            if op is None:
                self.fail("operator %s" % type(n.op).__name__)
            return "(%s %s %s)" % (a, op, b)
        if isinstance(n, ast.Call):
            f = _attr(n.func)
            if f == "reduce" and _attr(n.func.value) in ("logical_or", "logical_and"):
                joiner = " || " if _attr(n.func.value) == "logical_or" else " && "
                return "(%s)" % joiner.join(self.visit(a) for a in n.args[0].elts)
            if f in ("logical_or", "logical_and"):
                return "(%s)" % (" || " if f == "logical_or" else " && ").join(self.visit(a) for a in n.args)
            if f in _FUNCS:
                return "%s(%s)" % (_FUNCS[f], ", ".join(self.visit(a) for a in n.args))
            if f in _CMP:
                return "(%s %s %s)" % (self.visit(n.args[0]), _CMP[f], self.visit(n.args[1]))
            if f in ("minimum", "maximum"):
                a, b = (self.visit(x) for x in n.args)
                return "(%s %s %s ? %s : %s)" % (a, "<" if f == "minimum" else ">", b, a, b)
            if f == "select":
                conds, vals = n.args[0].elts, n.args[1].elts
                out = "0.0"
                for c, v in reversed(list(zip(conds, vals))):
                    out = "(%s ? %s : %s)" % (self.visit(c) if not (isinstance(c, ast.Constant) and c.value is True) else "true",
                                              self.visit(v), out)
                return out
            self.fail("the function %s" % ast.unparse(n.func))
        self.fail("the expression %s" % ast.unparse(n))


# ---------------------------------------------------------------------------------------------------------------------
# what the spec's words mean to the engine

# the engine's own values for what a stabilization block leaves unstated (proteus's ResGradQuad_SC defaults)
_SC_DEFAULTS = {"beta": 1.0, "reference": 1.0, "lagged": True}


def _stabilization(stab, problems):
    """-> the constants of the SUPG template, and the keys the engine defaulted."""
    method = stab.get("method")
    if method not in ("SUPG", None):
        problems.append("stabilization %s: the template implements SUPG (the other methods are other templates: ASGS/VMS, "
                        "entropy viscosity and graph viscosity are the edge-based paths)" % method)
    if stab.get("lagged"):
        problems.append("stabilization lagged: true (the subgrid error from the previous step): the template evaluates it at "
                        "the current iterate (only shock capturing can be lagged)")
    if stab.get("tau", "metric") != "metric":
        problems.append("tau: %s: the template implements the metric tau (proteus useMetrics = 1)" % stab.get("tau"))
    sc = stab.get("shock_capturing")
    defaulted = []
    if not isinstance(sc, dict):
        return {"factor": 0.0, "beta": 1.0, "reference": 1.0, "lagged": True}, ["shock_capturing (none stated: factor 0)"]
    out = {"factor": float(sc.get("factor", 0.0))}
    if "factor" not in sc:
        defaulted.append("factor")
    for k, d in _SC_DEFAULTS.items():
        if k in sc:
            out[k] = sc[k]
        else:
            out[k] = d
            defaulted.append(k)
    return out, defaulted


def _face_kind(entry, conditions, problems, where):
    """The face kernel the equation and its boundary words select: a conservative flux upwinds (strong Dirichlet on inflow); a
    transport Hamiltonian (linear in grad u, no flux) penalizes the inflow; a nonlinear Hamiltonian (eikonal) has no boundary term."""
    for bc in conditions or []:
        kind = bc.get("kind")
        if kind == "weak":
            problems.append("a weak Dirichlet condition on %s (kind: weak): the scalar template's faces have no a grad u / "
                            "penalty term (the generic exteriorFaceKernel has)" % bc.get("variable"))
        elif kind == "strong" and "f" in (entry.get("advection") or {}) and bc.get("only") != "inflow":
            problems.append("a strong Dirichlet condition on the whole boundary of %s: only the inflow part ('only: inflow') is "
                            "an upwind face term; the rest is the driver removing the unknown" % bc.get("variable"))
    if "advection" in entry:
        return "Upwind"
    ham = entry.get("hamiltonian")
    if ham:
        return "InflowPenalty" if all(v != "nonlinear" for v in ham["depends_on"].values()) else "None"
    return "None"


# ---------------------------------------------------------------------------------------------------------------------

class _Inputs:
    """The names an expression may use, and where each comes from."""

    def __init__(self, spec):
        self.dim = spec["dim"]
        self.unknown = spec["adr"]["equations"][0]["component"]
        self.fields = [c for r in (spec.get("reads") or {}).values() for c in r.get("components", [])]
        self.mesh = list(spec.get("mesh_symbols") or [])
        self.step_fields = []                       # a coefficient lagged to the previous step is a driver-supplied field
        self.names = {self.unknown: "u"}
        for k in range(self.dim):
            self.names["grad_%s_%d" % (self.unknown, k)] = "gu[%d]" % k
        for k, c in enumerate(_COORDINATES[:self.dim]):
            self.names.setdefault(c, "X[%d]" % k)
        self.slot = {}

    def finish(self):
        layout = self.fields + self.step_fields + self.mesh
        for i, f in enumerate(layout):
            self.names[f] = "fld[%d]" % i
            self.slot[f] = i
        self.layout = layout
        return layout


def _code(inputs, problems, cse=None):
    def tr(text, what, names=None, plain=False):
        try:
            return _Translator(names or inputs.names, what, None if plain else cse)(text)
        except GeneratorError as exc:
            problems.append(str(exc))
            return "0.0"
    return tr


def _collect(entry):
    """Every code string of an ADR equation's slots (m, dm, f, df, H, dH, r, dr, a, da)."""
    out = []

    def walk(v, key=None):
        if isinstance(v, str):
            if key not in ("component",):
                out.append(v)
        elif isinstance(v, dict):
            for k, x in v.items():
                if k not in ("depends_on", "rowptr", "colind"):
                    walk(x, k)
        elif isinstance(v, (list, tuple)):
            for x in v:
                walk(x, key)
    walk(entry)
    return out


def _isotropic(diff, dim, unknown, problems, where):
    """The entry of ADR's diffusion {potential: {rowptr, colind, a, da}} as one scalar a (a_II all the same code), or a problem."""
    ent = diff.get(unknown)
    if ent is None or len(diff) != 1:
        problems.append("%s: a diffusion potential other than the unknown" % where)
        return None
    rows = ent["rowptr"]
    entries = {}
    for i in range(dim):
        for k in range(rows[i], rows[i + 1]):
            entries[(i, ent["colind"][k])] = ent["a"][k]
    if any(i != j for i, j in entries) or len({entries.get((i, i)) for i in range(dim)}) != 1 or len(entries) != dim:
        problems.append("%s: a diffusion tensor that is not a scalar times the identity (the template's slot is isotropic)" % where)
        return None
    da = (ent.get("da") or {}).get(unknown)           # the derivative of each CSR value; the values are all the same code
    return entries[(0, 0)], (da[rows[0]] if da else None)


def _lagged(spec, problems):
    """The coefficients the sub-problem lags: {name: when}; none unless the spec states it."""
    return dict(spec.get("lagged_coefficients") or {})


def _lagged_lines(spec, inputs, lagged, problems):
    """The slots of a sub-problem that lags state coefficients (ymf's `lagged_coefficients`), from the record's named form: the
    coefficients are evaluated once each (`inline`), then m, f, H, r, a over those names, all in forward-mode dual numbers seeded on
    u (0) and grad u (1..nS). The derivatives of the slots are the dual parts, so a coefficient evaluated as a plain double does not
    carry its dependence on u into the Jacobian:
        previous_iterate  evaluated from the current u as a double: the residual is the full one, the Jacobian is Picard's
        previous_step     a per-point input of the driver (the coefficient's value at the previous step): no derivative
    A coefficient that is not lagged is evaluated as a dual number (its dependence on u enters the Jacobian, as ymf's dm, df, dr do)."""
    dim, unknown = inputs.dim, inputs.unknown
    named, coefficients = spec.get("named"), spec.get("coefficients")
    if not named or not coefficients:
        problems.append("lagged_coefficients is stated but the record has no named form (coefficients, named): ymf round 5's "
                        "split_problem writes it for the state coefficients a sub-problem uses")
        return None
    D = "Dual<%d>" % (dim + 1)
    entry = named["equations"][0]
    if "hamiltonian" in entry:
        problems.append("a Hamiltonian together with lagged coefficients: d(dH)/d(grad u) of a lagged form is not generated")
    typ, var = {}, {}
    base = {k: v for k, v in inputs.names.items() if k != unknown and not k.startswith("grad_%s_" % unknown)}
    lines = ["const %s uD = %s::seed(u, 0);" % (D, D), "%s guD[%d];" % (D, dim)]
    lines += ["guD[%d] = %s::seed(gu[%d], %d);" % (k, D, k, k + 1) for k in range(dim)]

    def names(kind):
        n = dict(base)
        n[unknown] = "uD" if kind == "D" else "u"
        for k in range(dim):
            n["grad_%s_%d" % (unknown, k)] = ("guD[%d]" if kind == "D" else "gu[%d]") % k
        for c, v in var.items():
            n[c] = v if (kind == "D" or typ[c] == "double") else v + ".v"
        return n
    for i, c in enumerate(named["order"]):
        when = lagged.get(c)
        if when == "previous_step":
            typ[c], var[c] = "double", inputs.names[c]
            continue
        kind = "double" if when == "previous_iterate" else "D"
        code = _Translator(names("double" if kind == "double" else "D"), "coefficient %s" % c)(coefficients[c]["inline"])
        typ[c], var[c] = kind, "c_%d" % i
        lines.append("const %s c_%d = %s;" % (D if kind == "D" else "double", i, code))
    nD = names("D")

    def slot(text, label):
        try:
            return _Translator(nD, label)(text)
        except GeneratorError as exc:
            problems.append(str(exc))
            return "0.0"
    has = {"F": False, "H": False, "R": False, "DR": False, "A": False, "DDH": False}
    if "mass" in entry:
        lines += ["const %s mD = %s;" % (D, slot(entry["mass"]["m"], "mass")), "T.m = mD.v;", "T.dm = mD.d[0];"]
    else:
        lines += ["T.m = 0.0;", "T.dm = 0.0;"]
    if "advection" in entry:
        has["F"] = True
        for k in range(dim):
            lines += ["{ const %s fD = %s; T.f[%d] = fD.v; T.df[%d] = fD.d[0]; }" % (D, slot(entry["advection"]["f"][k], "f[%d]" % k), k, k)]
    if "reaction" in entry:
        has["R"] = has["DR"] = True
        lines += ["{ const %s rD = %s; T.r = rD.v; T.dr = rD.d[0]; }" % (D, slot(entry["reaction"]["r"], "r")), "T.nuExtra = 0.0;"]
    if "diffusion" in entry:
        iso = _isotropic({unknown: dict(entry["diffusion"][unknown], da={})}, dim, unknown, problems, "%s" % spec["kernel"])
        if iso:
            has["A"] = True
            lines += ["{ const %s aD = %s; T.ad = aD.v; T.dad = aD.d[0]; }" % (D, slot(iso[0], "a"))]
    return lines, has


def _specialization_supg(spec):
    dim, adr = spec["dim"], spec["adr"]
    kernel = spec["kernel"]
    where = "%s (%dD)" % (kernel, dim)
    problems = []
    if len(adr["equations"]) != 1:
        raise GeneratorError("%s: the scalar template takes one equation, not %d (a vector problem needs the n-component Terms "
                             "path, which has no SUPG/VMS/shock-capturing template)" % (where, len(adr["equations"])))
    entry = adr["equations"][0]
    unknown = entry["component"]
    inputs = _Inputs(spec)
    lagged = _lagged(spec, problems)
    named = spec.get("named") if lagged else None
    if lagged:
        for name, when in lagged.items():
            if when == "previous_step":
                inputs.step_fields.append(name)
    layout = inputs.finish()
    cse = None if lagged else _CSE(_collect(entry))
    tr = _code(inputs, problems, cse)
    sc, defaulted = _stabilization(spec["stabilization"], problems)
    face = _face_kind(entry, spec.get("boundary_conditions"), problems, where)
    lines, has = [], {"F": False, "H": False, "R": False, "DR": False, "A": False, "DDH": False}
    dh_exprs = None
    if lagged:
        got = _lagged_lines(spec, inputs, lagged, problems)
        if got:
            lines, has = got
    else:
        if "mass" in entry:
            m = entry["mass"]
            lines.append("T.m = %s;" % tr(m["m"], "mass"))
            lines.append("T.dm = %s;" % tr(m["dm"].get(unknown, "0"), "dm"))
        else:
            lines += ["T.m = 0.0;", "T.dm = 0.0;"]
        if "advection" in entry:
            a = entry["advection"]
            has["F"] = True
            for k in range(dim):
                lines.append("T.f[%d] = %s;" % (k, tr(a["f"][k], "f[%d]" % k)))
            df = a["df"].get(unknown)
            for k in range(dim):
                lines.append("T.df[%d] = %s;" % (k, tr(df[k], "df[%d]" % k) if df else "0.0"))
        if "hamiltonian" in entry:
            h = entry["hamiltonian"]
            has["H"] = True
            lines.append("T.H = %s;" % tr(h["H"], "H"))
            dH = [tr(h["dH"][unknown][k], "dH[%d]" % k) for k in range(dim)]
            dh_exprs = [tr(h["dH"][unknown][k], "dH[%d]" % k, plain=True) for k in range(dim)]    # over Dual: no temporaries
            lines += ["T.dH[%d] = %s;" % (k, e) for k, e in enumerate(dH)]
            if h["depends_on"].get(unknown) == "nonlinear":
                has["DDH"] = True
        if "reaction" in entry:
            r = entry["reaction"]
            has["R"] = True
            lines.append("T.r = %s;" % tr(r["r"], "r"))
            lines.append("T.nuExtra = 0.0;")
            if r.get("dr", {}).get(unknown):
                has["DR"] = True
                lines.append("T.dr = %s;" % tr(r["dr"][unknown], "dr"))
        if "diffusion" in entry:
            iso = _isotropic(entry["diffusion"], dim, unknown, problems, where)
            if iso:
                has["A"] = True
                lines.append("T.ad = %s;" % tr(iso[0], "a"))
                lines.append("T.dad = %s;" % (tr(iso[1], "da") if iso[1] else "0.0"))
    if problems:
        raise GeneratorError("%s cannot be generated:\n  - %s" % (where, "\n  - ".join(problems)))
    # the tau form follows the equation: a time term gives the metric tau with dm_t, none gives the steady form (proteus's
    # useMetrics = 1 tau of VOF/NCLS and of RDLS)
    tau_form = 0 if "mass" in entry else 1
    if cse is not None:
        lines = cse.decls + lines
    return dict(kernel=kernel, dim=dim, flavor="supg", has=has, tau=tau_form, lines=lines, dh=dh_exprs, sc=sc, defaulted=defaulted,
                face=face, layout=layout, uses_x=any("X[" in l for l in lines), uses_u=any(re.search(r"\bu\b", l) for l in lines),
                ddh=has["DDH"])


def _specialization_terms(spec):
    """No stabilization is stated: the problem is Galerkin and its functor is the engine.h form (Terms, one component, the point's
    inputs bound by bindElement/bindPoint): every slot of the ADR form, with the diffusion tensor in full."""
    dim, adr = spec["dim"], spec["adr"]
    kernel = spec["kernel"]
    where = "%s (%dD)" % (kernel, dim)
    problems = []
    if len(adr["equations"]) != 1:
        raise GeneratorError("%s: the generated Galerkin functor takes one component, not %d" % (where, len(adr["equations"])))
    if spec.get("lagged_coefficients"):
        problems.append("lagged_coefficients on a Galerkin problem: only the SUPG template's functor reads them")
    entry = adr["equations"][0]
    unknown = entry["component"]
    inputs = _Inputs(spec)
    inputs.finish()
    names = {k: v.replace("gu[", "gu[0][") if v.startswith("gu[") else ("u[0]" if v == "u" else v) for k, v in inputs.names.items()}
    cse = _CSE(_collect(entry))
    tr = _code(inputs, problems, cse)
    t = lambda text, what: tr(text, what, names)
    lines, has = [], {"mass": False, "advection": False, "diffusion": False, "hamiltonian": False, "reaction": False}
    da_nz = False
    if "mass" in entry:
        has["mass"] = True
        lines += ["T.m[0] = %s;" % t(entry["mass"]["m"], "mass"), "T.dm[0][0] = %s;" % t(entry["mass"]["dm"].get(unknown, "0"), "dm")]
    if "advection" in entry:
        has["advection"] = True
        a = entry["advection"]
        for k in range(dim):
            lines.append("T.f[0][%d] = %s;" % (k, t(a["f"][k], "f[%d]" % k)))
            lines.append("T.df[0][0][%d] = %s;" % (k, t(a["df"][unknown][k], "df[%d]" % k) if a["df"].get(unknown) else "0.0"))
    if "hamiltonian" in entry:
        has["hamiltonian"] = True
        h = entry["hamiltonian"]
        lines.append("T.H[0] = %s;" % t(h["H"], "H"))
        for k in range(dim):
            lines.append("T.dH[0][0][%d] = %s;" % (k, t(h["dH"][unknown][k], "dH[%d]" % k)))
    if "reaction" in entry:
        has["reaction"] = True
        r = entry["reaction"]
        lines.append("T.r[0] = %s;" % t(r["r"], "r"))
        lines.append("T.dr[0][0] = %s;" % (t(r["dr"][unknown], "dr") if r.get("dr", {}).get(unknown) else "0.0"))
    if "diffusion" in entry:
        has["diffusion"] = True
        if list(entry["diffusion"]) != [unknown]:
            problems.append("a diffusion potential other than the unknown")
        else:
            ent = entry["diffusion"][unknown]
            lines += ["T.phi[0] = u[0];", "T.dphi[0][0] = 1.0;"]
            da = (ent.get("da") or {}).get(unknown)
            for i in range(dim):
                for k in range(ent["rowptr"][i], ent["rowptr"][i + 1]):
                    j = ent["colind"][k]
                    lines.append("T.a[0][0][%d][%d] = %s;" % (i, j, t(ent["a"][k], "a[%d][%d]" % (i, j))))
                    if da:
                        da_nz = True
                        lines.append("T.da[0][0][0][%d][%d] = %s;" % (i, j, t(da[k], "da[%d][%d]" % (i, j))))
    if problems:
        raise GeneratorError("%s cannot be generated:\n  - %s" % (where, "\n  - ".join(problems)))
    lines = cse.decls + lines
    return dict(kernel=kernel, dim=dim, flavor="terms", has=has, lines=lines, layout=inputs.layout, da_nz=da_nz)


def _emit_terms(s):
    dim, kernel, has = s["dim"], s["kernel"], s["has"]
    flags = ", ".join("has_%s = %s" % (k, str(v).lower()) for k, v in has.items())
    nf = len(s["layout"])
    return ["  template <>", "  struct %sCoefficients<%d> : DenseMasks" % (kernel, dim), "  {",
            "    // per-point inputs fld[]: %s" % (", ".join(s["layout"]) or "none"),
            "    static constexpr int nc = 1, nSpace = %d, nFields = %d;" % (dim, nf),
            "    static constexpr bool %s;" % flags,
            "    static constexpr bool da_nz(int, int, int) { return %s; }" % str(s["da_nz"]).lower(),
            "    const double *fields = nullptr;   // the whole mesh's point inputs (nE, nQ, nFields)",
            "    int nQ = 0;",
            "    const double *fq = nullptr;       // this point's inputs, set by bindElement + bindPoint",
            "    ADRH_FN %sCoefficients bindElement(int eN) const" % kernel,
            "    { %sCoefficients c = *this; c.fq = fields + (long)eN * nQ * nFields; return c; }" % kernel,
            "    ADRH_FN %sCoefficients bindPoint(int q) const" % kernel,
            "    { %sCoefficients c = *this; c.fq = fq + (long)q * nFields; return c; }" % kernel,
            "    ADRH_FN void evaluate(double, const double X[3], const double u[1], const double gu[1][%d], Terms<1, %d> &T) const" % (dim, dim),
            "    {", "      const double *fld = fq;", "      (void)fld; (void)X; (void)u; (void)gu;"] + \
           ["      " + l for l in s["lines"]] + ["    }", "  };", ""]


# ---------------------------------------------------------------------------------------------------------------------

def _emit_supg(s):
    dim, kernel = s["dim"], s["kernel"]
    has = s["has"]
    flags = ", ".join("%s = %s" % (n, str(has[k]).lower()) for n, k in (("hasF", "F"), ("hasH", "H"), ("hasR", "R")))
    extra = ", ".join("%s = true" % n for n, k in (("hasDR", "DR"), ("hasA", "A"), ("hasDDH", "DDH")) if has[k])
    sc = s["sc"]
    out = ["  template <>", "  struct %sCoefficients<%d>" % (kernel, dim), "  {",
           "    // per-point inputs fld[]: %s" % (", ".join("%s" % f for f in s["layout"]) or "none"),
           "    static constexpr int nSpace = %d, nFields = %d;" % (dim, len(s["layout"])),
           "    static constexpr bool %s;" % flags]
    if extra:
        out.append("    static constexpr bool %s;" % extra)
    out += ["    static constexpr bool usesPoint = %s;" % str(s["uses_x"]).lower(),
            "    static constexpr int tauForm = %d;" % s["tau"],
            "    static constexpr FaceKind face = FaceKind::%s;" % s["face"],
            "    // what the spec states of the stabilization (%s)" % ("the engine's defaults for: " + ", ".join(s["defaulted"])
                                                                      if s["defaulted"] else "all stated"),
            "    static constexpr double scFactor = %r, scBeta = %r, scReference = %r;" % (sc["factor"], float(sc["beta"]), float(sc["reference"])),
            "    static constexpr bool scLagged = %s;" % str(bool(sc["lagged"])).lower(),
            "    ADRH_FN static SUPGParameters supgParameters(double alpha, bool exactJacobian = false)",
            "    {",
            "      SUPGParameters P{alpha, scFactor, scReference, scBeta, 0.0, 4.0};",
            "      P.lag_sc = scLagged ? 1 : 0;",
            "      P.exactJacobian = exactJacobian ? 1 : 0;",
            "      return P;",
            "    }"]
    if s["ddh"]:
        out += ["    // dH_I (the expressions of ymf's dH) over a scalar type S: double for the Hamiltonian, Dual for d(dH)/d(grad u)",
                "    template <class S>",
                "    ADRH_FN ADRH_FORCE_INLINE static void dHExpressions(const double *fld, const double *X, S u, const S *gu, S *dH)",
                "    {", "      (void)fld; (void)X; (void)u;"]
        out += ["      dH[%d] = %s;" % (k, e) for k, e in enumerate(s["dh"])]
        out += ["    }"]
    xarg = "X" if s["uses_x"] else "nullptr"
    sig = "const double *fld, double u, const double *gu, %sScalarTerms<%d> &T" % ("const double *X, " if s["uses_x"] else "", dim)
    out += ["    ADRH_FN ADRH_FORCE_INLINE void evaluate(%s) const" % sig, "    {", "      (void)fld; (void)u; (void)gu;"]
    body = list(s["lines"])
    out += ["      " + l for l in body]
    out += ["    }"]
    if s["ddh"]:
        # d(dH)/d(grad u): only the exact Jacobian needs it, so the kernel asks for it (exactJacobian) and nothing else pays
        out += ["    ADRH_FN ADRH_FORCE_INLINE void evaluateDDH(const double *fld, double u, const double *gu, %sScalarTerms<%d> &T) const"
                % ("const double *X, " if s["uses_x"] else "", dim), "    {",
                "      Dual<%d> g[%d], dh[%d];" % (dim, dim, dim),
                "      for (int K = 0; K < %d; ++K) g[K] = Dual<%d>::seed(gu[K], K);" % (dim, dim),
                "      dHExpressions<Dual<%d>>(fld, %s, Dual<%d>(u), g, dh);" % (dim, xarg, dim),
                "      for (int I = 0; I < %d; ++I)" % dim,
                "        for (int K = 0; K < %d; ++K) T.ddH[I][K] = dh[I].d[K];" % dim,
                "    }"]
    out += ["  };", ""]
    return out


def emit(specs):
    by = {}
    for s in specs:
        by.setdefault(s["kernel"], []).append(s)
    out = ["// GENERATED by experimental/adrh/generator/generate.py from ymf's plain-data sub-problem records; do not edit.",
           "// Inputs: " + ", ".join("%s %dD" % (s["kernel"], s["dim"]) for s in specs),
           "#pragma once", '#include <cmath>', '#include "portable.h"', '#include "dual.h"', '#include "vof.h"', '#include "masked.h"', "",
           "namespace adrh", "{", "namespace generated", "{"]
    for kernel, group in by.items():
        out.append("  template <int nS_>\n  struct %sCoefficients { static constexpr bool available = false; };" % kernel)
        for spec in group:
            s = _specialization_supg(spec) if spec.get("stabilization") else _specialization_terms(spec)
            body = _emit_supg(s) if s["flavor"] == "supg" else _emit_terms(s)
            body[2] = body[2]       # struct line kept
            # `available` marks a dimension the spec stated
            body.insert(4, "    static constexpr bool available = true;")
            out += body
    out += ["} // namespace generated", "} // namespace adrh", ""]
    return "\n".join(out)


def main(argv):
    out, ins = argv[0], argv[1:]
    specs = [json.load(open(p, encoding="utf-8")) for p in ins]
    try:
        text = emit(specs)
    except GeneratorError as exc:
        print("generate.py: %s" % exc, file=sys.stderr)
        return 1
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
