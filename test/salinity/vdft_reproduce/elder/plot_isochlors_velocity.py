"""Elder: 20/40/60% isochlors with Darcy streamlines at 2, 4, 10 and 20 yr
(Voss & Souza 1987 Fig. 9a times), 2 x 2 panels.

Velocity is not archived, so it is rebuilt from psi and u as Richards forms it
(saturated, K_r = 1, rho_0 = 1; see ../plot_u_vel_panels_20yr.py):
    vx = -K dpsi/dx,   vz = -K (dpsi/dz + 1 + eps*u)
with the per-triangle P1 gradient area-averaged to the nodes.

Vertical axis is elevation z (0 = base, 150 m = source surface).
"""
import sys
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np

BASE = Path(__file__).resolve().parent
# `python plot_isochlors_velocity.py colour` draws u as a colour field, with the
# isochlors in white, into its own file
COLOUR = len(sys.argv) > 1 and sys.argv[1] == "colour"
OUT_PATH = BASE / ("elder_u_colour_velocity_20yr.png" if COLOUR
                   else "elder_isochlors_velocity_20yr.png")

YEARS = [2, 4, 10, 20]                 # 2 x 2 grid, row-major
LEVELS = [0.2, 0.4, 0.6]
COLOURS = ["#0090ff", "#00c000", "#ff1a1a"]
CONTOUR_LW = 2.2
V_EXAG = 1.0                            # vertical exaggeration of the panels
H, K, EPS = 150.0, 0.4107, 0.2          # m, m/day, (rho_s - rho_f)/rho_f
DT_DAYS = 30.4375                       # tnList = [0, 1e-8, 2e-8] + [i*DT]


def nodal_gradient(nodes, el, f):
    x, z = nodes[:, 0], nodes[:, 1]
    a, b, c = el.T
    x1, x2, z1, z2 = x[b] - x[a], x[c] - x[a], z[b] - z[a], z[c] - z[a]
    f1, f2 = f[b] - f[a], f[c] - f[a]
    det = x1 * z2 - x2 * z1
    gx, gz = (z2 * f1 - z1 * f2) / det, (x1 * f2 - x2 * f1) / det
    w = np.abs(det)
    Gx, Gz, W = (np.zeros(len(x)) for _ in range(3))
    for col in (a, b, c):
        np.add.at(Gx, col, w * gx)
        np.add.at(Gz, col, w * gz)
        np.add.at(W, col, w)
    return Gx / W, Gz / W


def main():
    with h5py.File(BASE / "ls_CCS_so.h5", "r") as h5:
        last = max(int(k[3:]) for k in h5 if k.startswith("u_t") and k[3:].isdigit())
        nodes = h5[f"nodesSpatial_Domain{last}"][:, :2]
        el = h5[f"elementsSpatial_Domain{last}"][:]
        frames = []
        for y in YEARS:
            k = int(round(y * 365.25 / DT_DAYS)) + 2
            if k <= last:
                frames.append((y, h5[f"u_t{k}"][:], h5[f"pressure_head_t{k}"][:]))

    tri = mtri.Triangulation(nodes[:, 0], nodes[:, 1], el)
    xg, zg = np.linspace(0, 600, 301), np.linspace(0, H, 76)
    XG, ZG = np.meshgrid(xg, zg)

    # height from the panel shape: 2 rows of ~7 in wide panels at 600 x 150*V_EXAG
    # (~6.3 in when the colourbar takes its share of the width)
    panel_w = 6.3 if COLOUR else 7.0
    fig, axes = plt.subplots(2, 2, figsize=(15, 1.2 + 2 * panel_w * H * V_EXAG / 600.0),
                             constrained_layout=True,
                             sharex=True, sharey=True)
    for ax, (y, u, psi) in zip(axes.flat, frames):
        gx, gz = nodal_gradient(nodes, el, psi)
        vx, vz = -K * gx, -K * (gz + 1.0 + EPS * u)

        VX = mtri.LinearTriInterpolator(tri, vx)(XG, ZG)
        VZ = mtri.LinearTriInterpolator(tri, vz)(XG, ZG)
        sp = np.hypot(VX, VZ)
        if COLOUR:
            field = ax.tripcolor(tri, u, cmap="viridis", vmin=0.0, vmax=1.0,
                                 shading="gouraud", zorder=0, rasterized=True)
        ax.streamplot(xg, zg, VX, VZ, color="0.85" if COLOUR else "0.55",
                      density=1.2, arrowsize=0.6,
                      linewidth=0.2 + 1.2 * np.clip(sp / np.nanmax(sp), 0, 1), zorder=1)
        # isochlors drawn last so they sit on top of the streamlines
        cs = ax.tricontour(tri, u, levels=LEVELS, colors="w" if COLOUR else COLOURS,
                           linewidths=CONTOUR_LW, zorder=3)
        if COLOUR:
            ax.clabel(cs, fmt="%.1f", fontsize=8, inline=True, inline_spacing=2)
        ax.set_xlim(0, 600)
        ax.set_ylim(0, H)
        ax.set_aspect(V_EXAG)
        ax.set_yticks([0, 50, 100, 150])
        ax.set_title(f"{y} year" + ("s" if y >= 2 else ""), fontsize=11)

    for ax in axes[:, 0]:
        ax.set_ylabel("z (m)")
    for ax in axes[-1, :]:
        ax.set_xlabel("x (m)")
    if COLOUR:
        fig.colorbar(field, ax=axes, shrink=0.8, pad=0.01,
                     label="u = C / C$_{max}$  (isochlors 0.2 / 0.4 / 0.6 in white)")
    else:
        handles = [plt.Line2D([], [], color=c, lw=CONTOUR_LW, label=f"{int(l*100)}%")
                   for l, c in zip(LEVELS, COLOURS)]
        handles.append(plt.Line2D([], [], color="0.55", lw=1.0, label="Darcy streamlines"))
        fig.legend(handles=handles, loc="outside lower center", fontsize=9, ncol=4,
                   frameon=False)
    fig.savefig(OUT_PATH, dpi=150)
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
