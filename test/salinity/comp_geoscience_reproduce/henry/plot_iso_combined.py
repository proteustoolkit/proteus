#!/usr/bin/env python3
"""
Single figure overlaying 10%, 50%, 90% isochlors for analytical and FCT runs.
"""
from __future__ import annotations

import argparse, re, sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri


def latest_step(h5_path, varname="u"):
    pat = re.compile(rf"^{re.escape(varname)}_t(\d+)$")
    with h5py.File(h5_path, "r") as f:
        steps = [int(m.group(1)) for k in f.keys() if (m := pat.match(k)) is not None]
    if not steps:
        raise RuntimeError(f"no {varname}_t<N> in {h5_path}")
    return max(steps)


def read_h5(h5_path, step, varname="u"):
    with h5py.File(h5_path, "r") as f:
        c = f[f"{varname}_t{step}"][:]
        nodes = f["nodesSpatial_Domain0"][:]
        elems = f["elementsSpatial_Domain0"][:]
    return c, nodes, elems


def reorder_analytical(x_ana, y_ana, c_ana, nodes):
    if (np.allclose(nodes[:, 0], x_ana) and np.allclose(nodes[:, 1], y_ana)):
        return c_ana
    from scipy.spatial import cKDTree
    tree = cKDTree(np.column_stack([x_ana, y_ana]))
    _, idx = tree.query(nodes[:, :2])
    return c_ana[idx]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fct_h5", default="ls_CCS_so.h5")
    p.add_argument("--npz", default="henry_reference.npz")
    p.add_argument("--var", default="u")
    p.add_argument("--levels", type=float, nargs="+", default=[0.1, 0.5, 0.9])
    p.add_argument("--out", default="iso_10_50_90_combined.png")
    p.add_argument("--tag", default="right_dbc/ref_1")
    p.add_argument("--fct-label", default="FCT")
    p.add_argument("--clip", action="store_true",
                   help="clip analytical c to [0,1]")
    args = p.parse_args()

    ref = np.load(args.npz)
    a, b = float(ref["a"]), float(ref["b"])
    x_ana, y_ana, c_ana_raw = ref["x"], ref["y"], ref["c"]
    if args.clip:
        c_ana_raw = np.clip(c_ana_raw, 0.0, 1.0)

    fct_step = latest_step(args.fct_h5, args.var)
    c_fct, nodes, elems = read_h5(args.fct_h5, fct_step, args.var)
    c_ana = reorder_analytical(x_ana, y_ana, c_ana_raw, nodes)

    print(f"FCT step {fct_step}: c in [{c_fct.min():.3e}, {c_fct.max():.3e}]",
          file=sys.stderr)

    x = nodes[:, 0]; y = nodes[:, 1]
    triang = mtri.Triangulation(x, y, elems)

    # color per level
    palette = {0.1: "#1f77b4", 0.5: "#2ca02c", 0.9: "#d62728"}

    fig, ax = plt.subplots(figsize=(10, 5.5))
    from matplotlib.lines import Line2D
    legend = []

    for lvl in args.levels:
        col = palette.get(round(lvl, 2), "C0")
        ax.tricontour(triang, c_ana, levels=[lvl],
                      colors=col, linewidths=2.4, linestyles="-")
        ax.tricontour(triang, c_fct, levels=[lvl],
                      colors=col, linewidths=2.0, linestyles="--")
        pct = int(round(100*lvl))
        legend.append(Line2D([0], [0], color=col, lw=2.4, ls="-",
                             label=f"{pct}% analytical"))
        legend.append(Line2D([0], [0], color=col, lw=2.0, ls="--",
                             label=f"{pct}% {args.fct_label}"))

    ax.legend(handles=legend, loc="upper left", framealpha=0.95, ncol=3)
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(y.min(), y.max())
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_title(f"Henry isochlors at $t=T$: analytical (solid), {args.fct_label} (dashed)")
    ax.grid(alpha=0.3)

    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
