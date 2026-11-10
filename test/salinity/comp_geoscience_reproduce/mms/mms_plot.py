"""Figures for the coupled MMS subsection.

    python mms_plot.py [runs dir]        (default: ./runs)

Writes into ./figures:
  mms_fields.pdf       manufactured vs numerical c and psi at t = T, and their
                       difference (T3, FCT, N = 80)
  mms_contours.pdf     iso-lines of the numerical (red) and analytical (blue,
                       dashed) psi and c at t = T/2 and T (T3, FCT, N = 80)
  mms_initial.pdf      initial psi and c (T3, N = 80)
  mms_profiles.pdf     vertical profiles through x = L/2 at t = T (T3, FCT)
  mms_convergence.pdf  L2 errors of c and psi against h, low order and FCT (T3),
                       at Co = 0.5 (dt) and Co = 0.25 (dt/2)

Colors follow the data-viz reference palette: one-hue sequential ramps for the
fields, a blue/red diverging pair around a gray zero for the errors, and the
first two categorical slots for the two schemes.
"""

import glob
import importlib
import os
import re
import sys
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

HERE = Path(__file__).resolve().parent
RUNS = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "runs"
OUT = HERE / "figures"

INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SERIES = {"FCT": "#2a78d6", "low order": "#eb6834"}
BLUE = LinearSegmentedColormap.from_list(
    "blue", ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"])
ORANGE = LinearSegmentedColormap.from_list(
    "orange", ["#fde4d8", "#f6b190", "#eb6834", "#c24e1e", "#8f3613"])
NUMERICAL, ANALYTICAL = "#b5302f", "#2a78d6"
DIVERGING = LinearSegmentedColormap.from_list(
    "diverging", ["#184f95", "#6da7ec", "#f0efec", "#ee8f8e", "#b5302f"])

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.linewidth": 0.6,
    "savefig.dpi": 200, "savefig.bbox": "tight",
})


def run_fields(run_dir, frac=1.0):
    """Nodes, triangles, c_h, psi_h and the exact fields of one run at the
    archived step closest to t = frac * T (the final step by default)."""
    run_dir = Path(run_dir)
    for line in open(run_dir / "run.env"):
        k, v = line.strip().split("=", 1)
        os.environ[k] = v
    sys.path.insert(0, str(run_dir))
    for name in ("mms_exact", "domain_mms"):
        sys.modules.pop(name, None)
    exact = importlib.import_module("mms_exact")
    domain = importlib.import_module("domain_mms")
    sys.path.pop(0)
    f = h5py.File(glob.glob(str(run_dir / "*.h5"))[0], "r")
    steps = [int(k.split("_t")[-1]) for k in f if k.startswith("u_t")]
    T = domain.tnList[max(steps)]
    last = min(steps, key=lambda k: abs(domain.tnList[k] - frac * T))
    X = np.asarray(f["nodesSpatial_Domain%d" % last])
    tri = mtri.Triangulation(X[:, 0], X[:, 1], np.asarray(f["elementsSpatial_Domain%d" % last]))
    t = domain.tnList[last]
    return dict(X=X, tri=tri, t=t,
                c=np.asarray(f["u_t%d" % last]), psi=np.asarray(f["pressure_head_t%d" % last]),
                c_e=exact.c_e(X, t), psi_e=exact.psi_e(X, t))


def fields_figure(run_dir, out):
    d = run_fields(run_dir)
    rows = [("c", "c", "$c_h - c_e$", BLUE, 1.0),
            ("psi", r"$\psi$ [cm]", r"$\psi_h - \psi_e$ [cm]", ORANGE, 100.0)]
    fig, axes = plt.subplots(2, 3, figsize=(7.6, 5.0), constrained_layout=True)
    for r, (key, label, elabel, cmap, scale) in enumerate(rows):
        ex, nu = d[key + "_e"] * scale, d[key] * scale
        levels = np.linspace(min(ex.min(), nu.min()), max(ex.max(), nu.max()), 13)
        err = nu - ex
        emax = np.abs(err).max()
        for c, vals in enumerate((ex, nu)):
            cs = axes[r][c].tricontourf(d["tri"], vals, cmap=cmap, levels=levels)
        cb = fig.colorbar(cs, ax=axes[r][:2], shrink=0.9, pad=0.02,
                          ticks=matplotlib.ticker.MaxNLocator(5))
        cb.set_label(label, fontsize=8)
        ce = axes[r][2].tricontourf(d["tri"], err, cmap=DIVERGING,
                                    levels=np.linspace(-emax, emax, 13),
                                    norm=TwoSlopeNorm(0.0, -emax, emax))
        cbe = fig.colorbar(ce, ax=axes[r][2], shrink=0.9, pad=0.02,
                           ticks=matplotlib.ticker.MaxNLocator(5))
        cbe.formatter.set_powerlimits((-2, 2))
        cbe.update_ticks()
        cbe.set_label(elabel, fontsize=8)
        for cbar in (cb, cbe):
            cbar.ax.tick_params(labelsize=7)
            cbar.outline.set_linewidth(0.4)
        for c in range(3):
            ax = axes[r][c]
            ax.set_aspect("equal")
            ax.set_xticks([0, 0.5, 1])
            ax.set_yticks([0, 0.5, 1])
            if r == 1:
                ax.set_xlabel("x [m]")
        axes[r][0].set_ylabel("%s\n\nz [m]" % label.split(" [")[0])
    for c, title in enumerate(("manufactured", "numerical", "difference")):
        axes[0][c].set_title(title, fontsize=9, color=INK)
    fig.savefig(out)
    plt.close(fig)


def contours_figure(run_dir, out, fracs=(0.5, 1.0)):
    """Per time t = frac * T (one row): iso-lines of the numerical (mesh) and
    analytical (fine grid) flow solution psi (left) and transport solution c
    (right), on the same levels in every row."""
    # three levels per field that exist at both times: c spans (0.2, 0.8);
    # psi_e spans [-12.9, -8.6] cm at T/2 and [-17.2, -8.6] cm at T
    levels = {"c": [0.35, 0.5, 0.65], "psi": [-12.5, -11.0, -9.5]}
    # analytical field on a grid much finer than any mesh, not its nodal interpolant
    g = np.linspace(0.0, 1.0, 401)
    gx, gz = np.meshgrid(g, g)
    Xf = np.stack([gx, gz, np.zeros_like(gx)], axis=-1)
    fig, axes = plt.subplots(len(fracs), 2, figsize=(6.6, 3.1 * len(fracs)),
                             constrained_layout=True, squeeze=False)
    for r, frac in enumerate(fracs):
        d = run_fields(run_dir, frac)
        exact = importlib.import_module("mms_exact")
        label = {0.25: "T/4", 0.5: "T/2", 1.0: "T"}.get(frac, "%gT" % frac)
        panels = ((axes[r][0], "psi", exact.psi_e, 100.0, r"$\psi$ [cm]", "%.1f"),
                  (axes[r][1], "c", exact.c_e, 1.0, "c", "%.2f"))
        for ax, key, fe, scale, title, fmt in panels:
            ex = fe(Xf, d["t"]) * scale
            lv = [v for v in levels[key] if ex.min() < v < ex.max()]
            ca = ax.contour(gx, gz, ex, levels=lv, colors=ANALYTICAL,
                            linewidths=1.6, linestyles="--")
            ax.tricontour(d["tri"], d[key] * scale, levels=lv, colors=NUMERICAL,
                          linewidths=1.2, linestyles="-")
            ax.clabel(ca, fmt=fmt, fontsize=7, colors=INK2, inline_spacing=3)
            ax.set_aspect("equal")
            ax.set_xticks([0, 0.5, 1])
            ax.set_yticks([0, 0.5, 1])
            ax.set_title(r"%s, $t = %s = %.0f$ s" % (title, label, d["t"]),
                         fontsize=9, color=INK)
        axes[r][0].set_ylabel("z [m]")
    for ax in axes[-1]:
        ax.set_xlabel("x [m]")
    handles = [matplotlib.lines.Line2D([], [], color=NUMERICAL, lw=1.0, ls="-"),
               matplotlib.lines.Line2D([], [], color=ANALYTICAL, lw=1.2, ls="--")]
    fig.legend(handles, ["numerical", "analytical"], loc="outside lower center",
               ncol=2, frameon=False, fontsize=8)
    fig.savefig(out)
    plt.close(fig)


def setup_figure(out):
    """Problem sketch: the unit square with the boundary data on each side and
    the initial data inside.  psi_e = psi_0 on the boundary because b = 0 there."""
    sys.path.insert(0, str(HERE))
    exact = importlib.import_module("mms_exact")
    sys.path.pop(0)
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    ax.add_patch(matplotlib.patches.Rectangle((0, 0), 1, 1, facecolor="#f7f6f3",
                                              edgecolor=INK, lw=1.6))
    fs = 9
    side = dict(fontsize=fs, color=INK, ha="center", va="center")
    flow = r"$\psi=\psi_0$ (weak)"
    ax.text(0.5, 1.10, flow + "\n" + r"$c=c_e(x,L,t)$", **side)
    ax.text(0.5, -0.10, flow + "\n" + r"$c=c_e(x,0,t)$", **side)
    ax.text(-0.13, 0.5, flow + "\n" + r"$c=c_e(0,z,t)$", rotation=90, **side)
    ax.text(1.13, 0.5, flow + "\n" + r"$c=c_e(L,z,t)$", rotation=90, **side)
    ic = (r"Initial data, $t=0$:" "\n\n"
          r"$\psi(x,z,0)=\psi_0=-1/\alpha$" "\n\n"
          r"$c(x,z,0)=\bar c + A_c\,\tanh(\xi_0/\delta)$" "\n"
          r"$\xi_0=z-z_0+a_f\cos(2\pi x/L)$")
    ax.text(0.5, 0.62, ic, fontsize=fs + 1, color=INK, ha="center", va="center",
            linespacing=1.2)
    vals = (r"$\psi_0=%.2f$ cm,  $\bar c=%g$,  $A_c=%g$" "\n"
            r"$z_0=%g$ m,  $\delta=%g$ m" "\n"
            r"$a_f=%g$ m,  $L=%g$ m"
            % (100 * exact.PSI0, exact.CBAR, exact.A_C, exact.Z0, exact.DELTA,
               exact.A_F, exact.L))
    ax.text(0.5, 0.27, vals, fontsize=fs, color=INK2, ha="center", va="center",
            linespacing=1.5)
    ax.text(0.04, 0.95, r"$\Omega=(0,L)^2$", fontsize=fs, color=INK2, va="top")
    # coordinate axes at the lower-left corner, gravity beside the right side
    arrow = dict(arrowstyle="-|>", color=INK, lw=1.0)
    ax.annotate("", (0.14, 0.04), (0.04, 0.04), arrowprops=arrow)
    ax.annotate("", (0.04, 0.14), (0.04, 0.04), arrowprops=arrow)
    ax.text(0.155, 0.04, "x", fontsize=fs, color=INK, va="center")
    ax.text(0.04, 0.155, "z", fontsize=fs, color=INK, ha="center")
    ax.annotate("", (1.28, 0.62), (1.28, 0.82), arrowprops=arrow)
    ax.text(1.31, 0.72, r"$\mathbf{g}$", fontsize=fs + 1, color=INK, va="center")
    ax.set_xlim(-0.25, 1.38)
    ax.set_ylim(-0.22, 1.22)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.savefig(out)
    plt.close(fig)


def initial_figure(run_dir, out):
    """Initial data at t = 0: psi (left) and c (right) of the archived first
    step, with the analytical and numerical c iso-lines of contours_figure."""
    d = run_fields(run_dir, 0.0)
    exact = importlib.import_module("mms_exact")
    g = np.linspace(0.0, 1.0, 401)
    gx, gz = np.meshgrid(g, g)
    Xf = np.stack([gx, gz, np.zeros_like(gx)], axis=-1)
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.2), constrained_layout=True)
    # psi_e(x, 0) = psi_0 = -1/alpha on the whole domain (b is scaled by t/T),
    # so the panel is a uniform fill with no iso-lines
    psi0 = 100.0 * d["psi"]
    ax = axes[0]
    ax.set_facecolor("#fde4d8")
    ax.text(0.5, 0.5, r"$\psi = \psi_0 = -1/\alpha = %.2f$ cm" % psi0.mean()
            + "\n" + r"(max $|\psi_h - \psi_0|$ = %.0e cm)" % np.abs(psi0 - 100.0 * exact.PSI0).max(),
            ha="center", va="center", fontsize=8, color=INK,
            bbox=dict(facecolor="white", edgecolor="none", pad=3.0))
    ax.set_title(r"$\psi$ [cm], $t = 0$", fontsize=9, color=INK)
    ax = axes[1]
    cs = ax.tricontourf(d["tri"], d["c"], cmap=BLUE, levels=np.linspace(0.2, 0.8, 13),
                        alpha=0.55)
    lv = [0.35, 0.5, 0.65]
    ca = ax.contour(gx, gz, exact.c_e(Xf, 0.0), levels=lv, colors=ANALYTICAL,
                    linewidths=1.6, linestyles="--")
    ax.tricontour(d["tri"], d["c"], levels=lv, colors=NUMERICAL, linewidths=1.2)
    ax.clabel(ca, fmt="%.2f", fontsize=7, colors=INK2, inline_spacing=3)
    cb = fig.colorbar(cs, ax=ax, shrink=0.85, pad=0.02, ticks=[0.2, 0.4, 0.6, 0.8])
    cb.ax.tick_params(labelsize=7)
    cb.outline.set_linewidth(0.4)
    ax.set_title(r"c, $t = 0$", fontsize=9, color=INK)
    for ax in axes:
        ax.set_aspect("equal")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([0, 0.5, 1])
        ax.set_yticks([0, 0.5, 1])
        ax.set_xlabel("x [m]")
    axes[0].set_ylabel("z [m]")
    handles = [matplotlib.lines.Line2D([], [], color=NUMERICAL, lw=1.0, ls="-"),
               matplotlib.lines.Line2D([], [], color=ANALYTICAL, lw=1.2, ls="--")]
    fig.legend(handles, ["numerical", "analytical"], loc="outside lower center",
               ncol=2, frameon=False, fontsize=8)
    fig.savefig(out)
    plt.close(fig)


def profiles_figure(run_dirs, out):
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.0), constrained_layout=True)
    colors = ["#2a78d6", "#eb6834"]
    for n, (label, rd) in enumerate(run_dirs):
        d = run_fields(rd)
        line = np.isclose(d["X"][:, 0], 0.5)
        order = np.argsort(d["X"][line, 1])
        z = d["X"][line, 1][order]
        for ax, key, scale in ((axes[0], "c", 1.0), (axes[1], "psi", 100.0)):
            if n == 0:
                zf = np.linspace(0, 1, 401)
                Xf = np.column_stack([np.full_like(zf, 0.5), zf, np.zeros_like(zf)])
                exact = importlib.import_module("mms_exact")
                fe = exact.c_e if key == "c" else exact.psi_e
                ax.plot(fe(Xf, d["t"]) * scale, zf, color=INK, lw=1.0, label="manufactured")
            ax.plot(d[key][line][order] * scale, z, color=colors[n], lw=1.5,
                    marker="o", ms=3.5, mfc=colors[n], mec="white", mew=0.6, label=label)
    axes[0].set_xlabel("c"); axes[1].set_xlabel(r"$\psi$ [cm]")
    axes[0].set_ylabel("z [m]")
    for ax in axes:
        ax.grid(True, color=GRID, lw=0.5)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")
    fig.savefig(out)
    plt.close(fig)


def convergence_figure(runs, out):
    sys.path.insert(0, str(HERE))
    import mms_table
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.0), constrained_layout=True)
    data = {}
    # Co = 0.5 ladder (solid, filled) and its dt/2 partner (dashed, open)
    for scheme, fct in (("FCT", 1), ("low order", 0)):
        for suffix, ls, mfc in (("", "-", SERIES[scheme]), ("_dt2", "--", "white")):
            dirs = [d for d in glob.glob(str(runs / ("a_T3_fct%d_N*" % fct)))
                    if re.fullmatch(r"a_T3_fct%d_N\d+%s" % (fct, suffix), Path(d).name)
                    and open(Path(d) / "run.out").read().strip().endswith("exit 0")]
            rows = sorted((mms_table.read_run(d) for d in dirs), key=lambda r: r["N"])
            data[(scheme, suffix)] = rows
            h = np.array([1.0 / r["N"] for r in rows])
            label = scheme + (r", $\Delta t/2$" if suffix else r", $\Delta t$")
            for ax, key in ((axes[0], "E2c"), (axes[1], "E2p")):
                e = np.array([r[key] for r in rows])
                ax.loglog(h, e, color=SERIES[scheme], lw=1.5 if not suffix else 1.1, ls=ls,
                          marker="o", ms=5 if not suffix else 4, mfc=mfc,
                          mec="white" if not suffix else SERIES[scheme], mew=0.8, label=label)
    for ax, key in ((axes[0], "E2c"), (axes[1], "E2p")):
        # reference slopes over one refinement level, anchored below the data
        h_fine = min(1.0 / r["N"] for r in data[("FCT", "")])
        e_fine = min(min(r[key] for r in rows) for rows in data.values())
        h_lo, h_hi = h_fine, 2.0 * h_fine
        y0 = 0.3 * e_fine
        for p, ls in ((1, "--"), (2, ":")):
            ax.loglog([h_lo, h_hi], [y0, y0 * 2.0 ** p], color=INK2, lw=0.8, ls=ls)
            ax.annotate("order %d" % p, (h_hi, y0 * 2.0 ** p), xytext=(3, 0),
                        textcoords="offset points", fontsize=7, color=INK2,
                        ha="left", va="center")
        ax.set_xlabel("h [m]")
        ax.grid(True, which="major", color=GRID, lw=0.5)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel(r"$\|e\|_{L^2}(c)$")
    axes[1].set_ylabel(r"$\|e\|_{L^2}(\psi)$")
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.savefig(out)
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    fields_figure(RUNS / "a_T3_fct1_N80", OUT / "mms_fields.pdf")
    fields_figure(RUNS / "a_T3_fct1_N80", OUT / "mms_fields.png")
    contours_figure(RUNS / "a_T3_fct1_N80", OUT / "mms_contours.pdf")
    contours_figure(RUNS / "a_T3_fct1_N80", OUT / "mms_contours.png")
    setup_figure(OUT / "mms_setup.pdf")
    setup_figure(OUT / "mms_setup.png")
    initial_figure(RUNS / "a_T3_fct1_N80", OUT / "mms_initial.pdf")
    initial_figure(RUNS / "a_T3_fct1_N80", OUT / "mms_initial.png")
    profiles_figure([("FCT, N = 20", RUNS / "a_T3_fct1_N20"),
                     ("FCT, N = 80", RUNS / "a_T3_fct1_N80")], OUT / "mms_profiles.pdf")
    profiles_figure([("FCT, N = 20", RUNS / "a_T3_fct1_N20"),
                     ("FCT, N = 80", RUNS / "a_T3_fct1_N80")], OUT / "mms_profiles.png")
    convergence_figure(RUNS, OUT / "mms_convergence.pdf")
    convergence_figure(RUNS, OUT / "mms_convergence.png")
    print("wrote", sorted(p.name for p in OUT.iterdir()))
