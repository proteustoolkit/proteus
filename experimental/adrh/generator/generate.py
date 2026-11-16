#!/usr/bin/env python3
"""Emit an ADRH engine coefficient functor from ymf's plain-data ADR form (generator spike, 2026-11-16).

Input: JSON (no sympy needed here; the trust boundary of ymf.symbolic is kept)

    {"kernel": "VOF",                       # the functor's name: <kernel>Coefficients
     "dim": 2,
     "adr": {...},                          # ymf.symbolic.adr_form / split_problem's sub-problem["adr"], one scalar equation
     "fields": ["v_0", "v_1", "ε"],         # the known-field components the driver supplies per point, in slot order
     "stabilization": {"method": "SUPG", "tau": "metric"}}   # chooses the engine template's tauForm

Output: a C++ header with `template <int nS> struct <kernel>Coefficients` specialized for each dimension given, in the
form scalarSUPGElement / scalarUpwindFace / scalarInflowPenaltyFace consume (vof.h's `VOFCoefficients` is the hand-written
original): nSpace, nFields, hasF, hasH, hasR, tauForm and evaluate(fld, u, gu, T).

What the generator refuses (GeneratorError) is the list of what the weak form states and the engine's scalar template cannot
take, see GENERATOR_SPIKE_REPORT_2026-11-16.md. Usage: generate.py out.h in1.json [in2.json ...]
"""
import ast
import json
import re
import sys


class GeneratorError(Exception):
    pass


_FUNCS = {"sqrt": "std::sqrt", "sin": "std::sin", "cos": "std::cos", "tan": "std::tan", "exp": "std::exp",
          "log": "std::log", "absolute": "std::fabs", "abs": "std::fabs", "tanh": "std::tanh", "sinh": "std::sinh",
          "cosh": "std::cosh", "hypot": "std::hypot"}
_CMP = {"less": "<", "greater": ">", "less_equal": "<=", "greater_equal": ">="}


def _attr(node):
    return node.attr if isinstance(node, ast.Attribute) else None


class _Translator:
    """numpy source (the strings ymf.symbolic.adr.to_code prints) -> a C++ expression with the same association order."""

    def __init__(self, names, where):
        self.names, self.where = names, where

    def __call__(self, text):
        return self.visit(ast.parse(text, mode="eval").body)

    def fail(self, what):
        raise GeneratorError("%s: %s" % (self.where, what))

    def visit(self, n):
        if isinstance(n, ast.Constant):
            if isinstance(n.value, bool) or not isinstance(n.value, (int, float)):
                self.fail("constant %r" % (n.value,))
            return repr(float(n.value))
        if isinstance(n, ast.Name):
            if n.id not in self.names:
                self.fail("the name %s is not an unknown, a gradient or a declared field (a coordinate, time, a mesh "
                          "symbol or a coefficient the driver does not supply)" % n.id)
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
                    return "(%s * %s)" % (a, a) if re.fullmatch(r"[\w\[\]\.]+", a) else self.fail("square of a compound")
                if isinstance(e, ast.Constant) and e.value == 0.5:
                    return "std::sqrt(%s)" % a
                return "std::pow(%s, %s)" % (a, b)
            op = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/"}.get(type(n.op))
            if op is None:
                self.fail("operator %s" % type(n.op).__name__)
            return "(%s %s %s)" % (a, op, b)
        if isinstance(n, ast.Call):
            f = _attr(n.func)
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


def _flag_terms(entry):
    return [k for k in ("mass", "advection", "hamiltonian", "reaction", "diffusion") if k in entry]


def _specialization(spec):
    dim, adr = spec["dim"], spec["adr"]
    kernel, fields = spec["kernel"], list(spec.get("fields", []))
    if len(adr["equations"]) != 1:
        raise GeneratorError("%s: the scalar template takes one equation, not %d (a vector problem needs the n-component "
                             "Terms path, which has no SUPG/VMS template)" % (kernel, len(adr["equations"])))
    entry = adr["equations"][0]
    problems = []
    unknown = entry["component"]
    where = "%s (%dD)" % (kernel, dim)
    if "diffusion" in entry:
        problems.append("a diffusive flux: scalarSUPGElement has no a(u) grad u slot (ScalarTerms carries no a); the "
                        "generic Terms path has it but not the SUPG/shock-capturing template")
    names = {unknown: "u"}
    for k in range(dim):
        names["grad_%s_%d" % (unknown, k)] = "gu[%d]" % k
    for i, f in enumerate(fields):
        names[f] = "fld[%d]" % i
    def tr(text, what):
        try:
            return _Translator(names, what)(text)
        except GeneratorError as exc:
            problems.append(str(exc))
            return "0.0"
    lines, has = [], {"F": False, "H": False, "R": False}
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
        if a["depends_on"].get(unknown) == "nonlinear" and False:
            pass
    if "hamiltonian" in entry:
        h = entry["hamiltonian"]
        has["H"] = True
        if h["depends_on"].get(unknown) == "nonlinear":
            problems.append("a Hamiltonian nonlinear in grad u: the template's ScalarTerms returns dH but no d(dH)/d(grad u) "
                            "(RDLS's S|grad u|); the Jacobian would be inexact")
        lines.append("T.H = %s;" % tr(h["H"], "H"))
        for k in range(dim):
            lines.append("T.dH[%d] = %s;" % (k, tr(h["dH"][unknown][k], "dH[%d]" % k)))
    if "reaction" in entry:
        r = entry["reaction"]
        if r["dr"]:
            problems.append("a reaction that depends on u: ScalarTerms carries r(x) but no dr/du")
        has["R"] = True
        lines.append("T.r = %s;" % tr(r["r"], "r"))
        lines.append("T.nuExtra = 0.0;")
    stab = spec.get("stabilization", {})
    if stab.get("method") not in ("SUPG", None):
        problems.append("stabilization %s: the template implements SUPG with residual shock capturing" % (stab.get("method"),))
    tau_form = {"metric": 0, "no_time_term": 1}[stab.get("tau", "metric")]
    if problems:
        raise GeneratorError("%s cannot be generated:\n  - %s" % (where, "\n  - ".join(problems)))
    return dim, has, tau_form, lines, uses_u(lines)


def uses_u(lines):
    return any(re.search(r"\bu\b", l) for l in lines)


def emit(specs):
    by = {}
    for s in specs:
        by.setdefault(s["kernel"], []).append(s)
    out = ["// GENERATED by experimental/adrh/generator/generate.py from ymf's plain-data ADR form; do not edit.",
           "// Inputs: " + ", ".join("%s %dD" % (s["kernel"], s["dim"]) for s in specs),
           "#pragma once", '#include <cmath>', '#include "portable.h"', '#include "vof.h"', "",
           "namespace adrh", "{", "namespace generated", "{"]
    for kernel, group in by.items():
        out.append("  template <int nS_>\n  struct %sCoefficients;" % kernel)
        for s in group:
            dim, has, tau, lines, used_u = _specialization(s)
            out += ["  template <>", "  struct %sCoefficients<%d>" % (kernel, dim), "  {",
                    "    static constexpr int nSpace = %d, nFields = %d;" % (dim, len(s.get("fields", []))),
                    "    static constexpr bool hasF = %s, hasH = %s, hasR = %s;" % tuple(str(has[k]).lower() for k in "FHR"),
                    "    static constexpr int tauForm = %d;" % tau,
                    "    ADRH_FN ADRH_FORCE_INLINE void evaluate(const double *fld, double u, const double *gu,"
                    " ScalarTerms<%d> &T) const" % dim,
                    "    {", "      (void)fld; (void)u; (void)gu;"]
            out += ["      " + l for l in lines]
            out += ["    }", "  };", ""]
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
