#!/usr/bin/env python3
"""Errors of Table tab:henry_results: FCT and low-order runs against the semi-analytical
reference (henry_semianalytical.py, evaluated at the run's nodes).

For each run, at its last archived step:
  L2       discrete lumped-mass norm (sum_i ML_i e_i^2)^(1/2), ML_i = sum of |T|/3 over the
           triangles at node i (the manuscript's \\|e\\|_{L^2})
  L2_x1.8  the same over nodes with x <= 1.8 m (excludes the upper seaward corner)
  Linf     max_i |e_i|, and where it occurs
  rms      the unweighted nodal RMS that plot_henry_comparison.py prints
  toe      x where the 10/50/90 % isochlors meet the bottom (y = 0), computed vs reference,
           by linear interpolation along the bottom nodes
Writes JSON (--out) and prints a summary.
"""
import argparse
import json
import re

import h5py
import numpy as np


def last(h5, var="u"):
    pat = re.compile(rf"^{var}_t(\d+)$")
    with h5py.File(h5, "r") as f:
        k = max(int(m.group(1)) for key in f.keys() if (m := pat.match(key)))
        return (f[f"{var}_t{k}"][:], f["nodesSpatial_Domain0"][:],
                f["elementsSpatial_Domain0"][:], k)


def lumped_mass(nodes, elems):
    p = nodes[:, :2]
    a, b, c = p[elems[:, 0]], p[elems[:, 1]], p[elems[:, 2]]
    area = 0.5*np.abs((b[:, 0]-a[:, 0])*(c[:, 1]-a[:, 1]) - (c[:, 0]-a[:, 0])*(b[:, 1]-a[:, 1]))
    ML = np.zeros(len(p))
    for k in range(3):
        np.add.at(ML, elems[:, k], area/3.0)
    return ML


def toe(x, c, level):
    """first x (from the sea side, x decreasing) where c falls through level"""
    o = np.argsort(x)[::-1]
    x, c = x[o], c[o]
    for k in range(len(x)-1):
        if (c[k]-level)*(c[k+1]-level) <= 0.0 and c[k] != c[k+1]:
            return float(x[k] + (level-c[k])*(x[k+1]-x[k])/(c[k+1]-c[k]))
    return float("nan")


def metrics(h5, ref):
    c, nodes, elems, step = last(h5)
    x, y = nodes[:, 0], nodes[:, 1]
    if np.allclose(x, ref["x"]) and np.allclose(y, ref["y"]):
        ca = ref["c"]
    else:
        from scipy.spatial import cKDTree
        _, idx = cKDTree(np.column_stack([ref["x"], ref["y"]])).query(nodes[:, :2])
        ca = ref["c"][idx]
    e = c - ca
    ML = lumped_mass(nodes, elems)
    near = x <= 1.8 + 1e-12
    k = int(np.argmax(np.abs(e)))
    bot = np.abs(y - y.min()) < 1e-9
    out = dict(h5=h5, step=int(step), nodes=int(len(x)), elements=int(len(elems)),
               c_min=float(c.min()), c_max=float(c.max()),
               L2=float(np.sqrt((ML*e*e).sum())),
               L2_x1p8=float(np.sqrt((ML[near]*e[near]**2).sum())),
               Linf=float(abs(e[k])), Linf_at=[float(x[k]), float(y[k])],
               rms=float(np.sqrt(np.mean(e*e))), toe={})
    for lvl in (0.1, 0.5, 0.9):
        xc, xr = toe(x[bot], c[bot], lvl), toe(x[bot], ca[bot], lvl)
        out["toe"][f"{lvl:.1f}"] = dict(computed=xc, reference=xr, diff_mm=1000.0*(xc-xr))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fct", required=True)
    p.add_argument("--lo", required=True)
    p.add_argument("--npz", required=True)
    p.add_argument("--out", default="henry_metrics.json")
    a = p.parse_args()
    ref = np.load(a.npz)
    res = {"fct": metrics(a.fct, ref), "lo": metrics(a.lo, ref)}
    r1, r2 = res["fct"]["L2"], res["lo"]["L2"]
    res["L2_rel_diff_fct_vs_lo"] = (r1 - r2)/r2
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    for k in ("lo", "fct"):
        m = res[k]
        print("%-4s L2 %.3e  L2(x<=1.8) %.3e  Linf %.3f at (%.3f, %.3f)  rms %.3e  c in [%.3g, %.3g]  toes(mm) %s"
              % (k, m["L2"], m["L2_x1p8"], m["Linf"], *m["Linf_at"], m["rms"], m["c_min"], m["c_max"],
                 ", ".join("%s: %+.1f" % (l, t["diff_mm"]) for l, t in m["toe"].items())))


if __name__ == "__main__":
    main()
