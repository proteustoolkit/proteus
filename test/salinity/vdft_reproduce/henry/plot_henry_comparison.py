#!/usr/bin/env python3
"""
Compare Proteus numerical concentration with the Henry semi-analytical
reference. Reads:
    ls_CCS_so.h5          : Proteus output (nodes, elements, u_t<N>)
    henry_reference.npz   : output of henry_semianalytical.py

Produces a 3-panel figure:
    [analytical c]  [numerical c]  [overlaid 50% isochlors]
"""
from __future__ import annotations

import argparse
import re
import sys

import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri


def latest_step(h5, varname):
    """Return the highest N for which '<varname>_t<N>' exists."""
    pat = re.compile(rf"^{re.escape(varname)}_t(\d+)$")
    steps = []
    with h5py.File(h5, "r") as f:
        for k in f.keys():
            m = pat.match(k)
            if m:
                steps.append(int(m.group(1)))
    if not steps:
        raise RuntimeError(f"No datasets matching {varname}_t<N> in {h5}")
    return max(steps)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--h5", default="ls_CCS_so.h5")
    p.add_argument("--npz", default="henry_reference.npz")
    p.add_argument("--var", default="u", help="Proteus variable name (default 'u')")
    p.add_argument("--step", type=int, default=None,
                   help="time step (default: last)")
    p.add_argument("--out", default="henry_comparison.png")
    p.add_argument("--levels", type=int, default=21)
    p.add_argument("--isochlor", type=float, default=0.5)
    args = p.parse_args()

    # --- analytical ---
    ref = np.load(args.npz)
    x = ref["x"]
    y = ref["y"]
    c_ana = ref["c"]
    psi_ana = ref["psi"]
    print(f"analytical:  a={ref['a']}, b={ref['b']}, xi={ref['xi']}",
          file=sys.stderr)
    print(f"  c range: [{c_ana.min():.3f}, {c_ana.max():.3f}]",
          file=sys.stderr)

    # --- numerical ---
    step = args.step if args.step is not None else latest_step(args.h5, args.var)
    field = f"{args.var}_t{step}"
    print(f"numerical:   reading '{field}' from {args.h5}", file=sys.stderr)
    with h5py.File(args.h5, "r") as f:
        c_num = f[field][:]
        nodes = f["nodesSpatial_Domain0"][:]
        elems = f["elementsSpatial_Domain0"][:]
    print(f"  c range: [{c_num.min():.3f}, {c_num.max():.3f}]", file=sys.stderr)

    # Sanity: the npz coordinates should match the h5 nodes
    if not (np.allclose(nodes[:, 0], x) and np.allclose(nodes[:, 1], y)):
        print("WARNING: npz node order does not match h5 — reordering analytical "
              "to h5 order via nearest match", file=sys.stderr)
        # naive O(N^2) match (3200 nodes is fine)
        from scipy.spatial import cKDTree
        tree = cKDTree(np.column_stack([x, y]))
        _, idx = tree.query(nodes[:, :2])
        c_ana = c_ana[idx]
        psi_ana = psi_ana[idx]
        x = nodes[:, 0]
        y = nodes[:, 1]

    # --- plot ---
    triang = mtri.Triangulation(x, y, elems)
    levels = np.linspace(0.0, 1.0, args.levels)

    fig, axs = plt.subplots(1, 3, figsize=(15, 4.2),
                            gridspec_kw=dict(width_ratios=[1, 1, 1.05]))

    cf0 = axs[0].tricontourf(triang, c_ana, levels=levels, cmap="viridis", extend="both")
    axs[0].tricontour(triang, c_ana, levels=[args.isochlor], colors="white", linewidths=1.2)
    axs[0].set_title(f"semi-analytical c   (a={float(ref['a']):.3g}, b={float(ref['b']):.3g})")
    axs[0].set_aspect("equal")
    axs[0].set_xlabel("x"); axs[0].set_ylabel("y")

    cf1 = axs[1].tricontourf(triang, c_num, levels=levels, cmap="viridis", extend="both")
    axs[1].tricontour(triang, c_num, levels=[args.isochlor], colors="white", linewidths=1.2)
    axs[1].set_title(f"Proteus c   (step {step})")
    axs[1].set_aspect("equal")
    axs[1].set_xlabel("x")

    # overlay isochlors
    axs[2].tricontour(triang, c_ana, levels=[args.isochlor],
                      colors="C0", linewidths=2.0, linestyles="-")
    axs[2].tricontour(triang, c_num, levels=[args.isochlor],
                      colors="C3", linewidths=2.0, linestyles="--")
    axs[2].set_title(f"{int(100*args.isochlor)}% isochlor: analytical (blue) vs Proteus (red)")
    axs[2].set_aspect("equal")
    axs[2].set_xlim(x.min(), x.max())
    axs[2].set_ylim(y.min(), y.max())
    axs[2].set_xlabel("x")
    axs[2].grid(alpha=0.3)

    cbar = fig.colorbar(cf1, ax=axs[:2].tolist(), fraction=0.025, pad=0.02)
    cbar.set_label("c")

    fig.suptitle("Henry problem: numerical vs semi-analytical")
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"wrote {args.out}", file=sys.stderr)

    # error metrics
    err = c_num - c_ana
    L2 = np.sqrt(np.mean(err ** 2))
    Linf = np.abs(err).max()
    print(f"\n  ||c_num - c_ana||_2  = {L2:.4e}", file=sys.stderr)
    print(f"  ||c_num - c_ana||_inf = {Linf:.4e}", file=sys.stderr)


if __name__ == "__main__":
    main()
