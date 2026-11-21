#!/usr/bin/env python3
"""Write the generator's JSON inputs from ymf (needs ymf[symbolic] and sympy):

    PYTHONPATH=<ymf>/src:. python make_inputs.py <ymf checkout>/tests/fixtures/two_phase

Two specs are read, ymf's two-phase dam break as DispatchTable states it (dambreak_formulation.ymf: no porosity, so the kernels
compare with proteus's own fixtures) and the same with a porous block (dambreak_porous.ymf, 2D and 3D). Every sub-problem of
the discretization becomes ``inputs/<prefix><name>_<dim>d.json`` (spec.kernel_input): there is no list of which kernels to
build, no per-model field declaration, no stabilization written here. The porous spec is re-dimensioned to 3D by adding the
box's third axis and the gravity's third component (the equations are written over coordinates, so nothing else changes).
Which of the inputs the engine can take is the generator's refusal (regenerate.sh).
"""
import copy
import json
import os
import sys

from ymf.schema import load_ymf
from ymf.symbolic.split import split_problem

from spec import kernel_input

HERE = os.path.dirname(os.path.abspath(__file__))
DISCRETIZATION = "level_set_five"


def write(name, spec):
    os.makedirs(os.path.join(HERE, "inputs"), exist_ok=True)
    with open(os.path.join(HERE, "inputs", name + ".json"), "w", encoding="utf-8") as fh:
        json.dump(spec, fh, ensure_ascii=False, indent=1)
        fh.write("\n")


def redimension(doc, dim, depth=0.6):
    """The same spec in ``dim`` = 3: one more axis of the box (depth), a 0 for it in the vector coefficient with the gravity role."""
    doc = copy.deepcopy(doc)
    form = doc["model"]["strong_form"]
    if dim == 3:
        domain = form["domain"]
        cut = domain.rindex("],") + 1            # the box ends at the last "]," (the time interval follows)
        form["domain"] = "%s × [0, %g]%s" % (domain[:cut], depth, domain[cut:])
        for entry in form["coefficients"].values():
            if isinstance(entry, dict) and entry.get("role") == "gravity":
                entry["formula"] = entry["formula"].rstrip(") ") + ", 0)"
    return doc


SPECS = (("dambreak_formulation.ymf", "", (2,)), ("dambreak_porous.ymf", "Porous", (2, 3)))


def constructed(doc, equation, lagged=None):
    """No fixture states a lagged coefficient, so the engine reads it from this constructed spec: the dam break's level-set
    sub-problem solving ``equation`` with the state coefficient kappa = 1 + phi^2 (ymf's own split_problem classifies the equation
    and writes the named form), no shock capturing, and optionally kappa lagged (ymf round 5's `lagged_coefficients`)."""
    doc = copy.deepcopy(doc)
    entry = next(e for e in doc["solution_paths"]["discretizations"] if e["name"] == DISCRETIZATION)
    sub = next(s for s in entry["subproblems"] if s["name"] == "ncls")
    doc["model"]["strong_form"]["coefficients"]["κ"] = {"formula": "1 + φ*φ", "units": "1"}
    sub.pop("solves")
    sub["for"] = "φ"
    sub["equation"] = equation
    sub["stabilization"] = {"method": "SUPG", "tau": "metric", "lagged": False, "shock_capturing": "none"}
    if lagged:
        sub["lagged_coefficients"] = {"κ": lagged}
    return doc


#: the constructed problems: name -> equation; each is written unlagged, lagged to the previous iterate and to the previous step
CONSTRUCTED = {"transport": "∂φ/∂t + ∇·(κ v φ) = 0"}
#: constructed problems with no lagging, for the slots of C19 slice 2: a reaction and a diffusion that depend on u (kappa(u)), linear
#: advection (so the Jacobian is exactly what the slots give)
SLOTS = {"reacdiff": "∂φ/∂t + ∇·(v φ) − ∇·(κ ∇φ) + κ φ = 0"}


def without_shock_capturing(doc, subproblem):
    """The spec's sub-problem with `shock_capturing: none` (nu = 0), so that the Jacobian of a nonlinear Hamiltonian can be checked
    against finite differences of the residual: the viscosity is not differentiated by the template (it is lagged, or proteus's)."""
    doc = copy.deepcopy(doc)
    entry = next(e for e in doc["solution_paths"]["discretizations"] if e["name"] == DISCRETIZATION)
    sub = next(s for s in entry["subproblems"] if s["name"] == subproblem)
    stab = dict(sub.get("stabilization") or entry.get("stabilization") or {"method": "SUPG", "tau": "metric"})
    stab["shock_capturing"] = "none"
    sub["stabilization"] = stab
    return doc


def main(fixtures, only=None):
    base = load_ymf(os.path.join(fixtures, SPECS[0][0]))
    for name, equation in CONSTRUCTED.items():
        for suffix, lagged in (("", None), ("_lagged", "previous_iterate"), ("_step", "previous_step")):
            split = split_problem(constructed(base, equation, lagged), DISCRETIZATION)
            sub = next(s for s in split["subproblems"] if s["name"] == "ncls")
            write("%s%s_2d" % (name, suffix), kernel_input(split, sub, (name + suffix).title().replace("_", "")))
    for name, equation in SLOTS.items():
        split = split_problem(constructed(base, equation), DISCRETIZATION)
        sub = next(s for s in split["subproblems"] if s["name"] == "ncls")
        write("%s_2d" % name, kernel_input(split, sub, name.title()))
    split = split_problem(without_shock_capturing(base, "rdls"), DISCRETIZATION)
    write("rdls_nosc_2d", kernel_input(split, next(s for s in split["subproblems"] if s["name"] == "rdls"), "RdlsNosc"))
    if only == "variants":
        return
    for name, prefix, dims in SPECS:
        doc = load_ymf(os.path.join(fixtures, name))
        for dim in dims:
            split = split_problem(redimension(doc, dim), DISCRETIZATION)
            assert split["dim"] == dim
            for sub in split["subproblems"]:
                spec = kernel_input(split, sub, prefix + sub["name"].upper())
                if spec is not None and not (prefix and len(spec["adr"]["equations"]) > 1):   # the vector problem is kept once, unprefixed
                    write("%s%s_%dd" % (prefix.lower() + "_" if prefix else "", sub["name"], dim), spec)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
