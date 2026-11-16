#!/usr/bin/env python3
"""Write the generator's JSON inputs from ymf (needs ymf[symbolic], sympy): `python make_inputs.py <ymf checkout>/tests/fixtures/two_phase/dambreak.ymf`.

VOF and NCLS 2D come from the dam-break spec itself (ymf.symbolic.split_problem, sub-problems `vof` and `ncls`); VOF carries a
porosity because proteus's VOF multiplies m and f by it and the spec's carrier χh does not state one, so the porosity form is
written with ymf's Space (known fields v, ε) and the equation text below. 3D is the same text in a 3D Space (one spec, two
dimensions). `fields` is the order of the per-point inputs the driver supplies (vof.h's convention).
"""
import json
import os
import sys

from ymf.schema import load_ymf
from ymf.symbolic import Space, adr_form, parse_equation
from ymf.symbolic.split import split_problem

HERE = os.path.dirname(os.path.abspath(__file__))
STAB = {"method": "SUPG", "tau": "metric", "shock_capturing": {"lagged": True}}


def write(name, spec):
    os.makedirs(os.path.join(HERE, "inputs"), exist_ok=True)
    with open(os.path.join(HERE, "inputs", name + ".json"), "w", encoding="utf-8") as fh:
        json.dump(spec, fh, ensure_ascii=False, indent=1)
        fh.write("\n")


def space_adr(dim, unknown, text):
    sp = Space(dim, {unknown: 0}, {}, fields={"v": 1, "ε": 0})
    return adr_form(parse_equation(text, sp), sp)


def main(path):
    split = split_problem(load_ymf(path), "level_set_five")
    sub = {s["name"]: s for s in split["subproblems"]}
    v = lambda dim: ["v_%d" % k for k in range(dim)]
    # what the spec states (no porosity): VOF's carrier transport and NCLS's Hamiltonian, 2D, as split_problem classifies them
    write("vof_deck_2d", {"kernel": "VOFDeck", "dim": 2, "adr": sub["vof"]["adr"], "fields": v(2), "stabilization": STAB})
    write("ncls_2d", {"kernel": "NCLS", "dim": 2, "adr": sub["ncls"]["adr"], "fields": v(2), "stabilization": STAB})
    for dim in (1, 2, 3):      # the engine's dispatch instantiates every dimension it supports
        write("vof_%dd" % dim, {"kernel": "VOF", "dim": dim, "fields": v(dim) + ["ε"], "stabilization": STAB,
                                "adr": space_adr(dim, "χ", "∂(ε*χ)/∂t + ∇·(ε*χ*v) = 0")})
        if dim != 2:
            sp = Space(dim, {"φ": 0}, {}, fields={"v": 1})
            write("ncls_%dd" % dim, {"kernel": "NCLS", "dim": dim, "fields": v(dim), "stabilization": STAB,
                                     "adr": adr_form(parse_equation("∂φ/∂t + v·∇φ = 0", sp), sp)})
    # what the generator must refuse today: RDLS (nonlinear H, a field only the driver knows), MCorr (a reaction in u, a
    # regularization width from the mesh), the flow problem (a vector, VMS); read from the same spec
    for name in ("rdls", "mcorr", "flow"):
        s = sub[name]
        comps = [c for r in s["reads"].values() for c in r["components"]]
        write("refused_%s_2d" % name, {"kernel": name.upper(), "dim": 2, "adr": s["adr"], "fields": comps,
                                       "stabilization": STAB})
    # the 2D Space route and the split route agree on NCLS
    sp2 = Space(2, {"φ": 0}, {}, fields={"v": 1})
    assert adr_form(parse_equation("∂φ/∂t + v·∇φ = 0", sp2), sp2) == sub["ncls"]["adr"]


if __name__ == "__main__":
    main(sys.argv[1])
