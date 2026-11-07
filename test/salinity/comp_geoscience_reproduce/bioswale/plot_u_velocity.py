"""Bioswale: u as a colour field with 0.2/0.4/0.6 isochlors and Darcy streamlines
at t = 0.1219, 0.2488 and 0.5025 d, 1 x 3 panels (the layout of
../elder/plot_isochlors_velocity.py colour).

Velocity is not archived, so it is rebuilt from psi as Richards forms it here
(unsaturated, no density coupling -- Raingarden_p.py sets no density_model):
    q = -Ks k_r(psi) (grad psi + e_z)
per triangle, with that triangle's van Genuchten-Mualem soil and k_r at the
triangle mean of psi, then area-averaged to the nodes.
"""
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np

BASE = Path(__file__).resolve().parent
OUT_PATH = BASE / "u_colour_velocity_bioswale.png"

FRAMES = [51, 102, 204]                 # tnList = [0, 1e-8, 2e-8] + [i*DT], t = (k-2)*DT
DT = 0.5 / 201                          # ls_CCS_so.py: T = 0.5 d, nDTout = 201
LEVELS = [0.2, 0.4, 0.6]
CONTOUR_LW = 2.2
L = (3.0, 5.0)                          # m
DRAIN = ((1.45, 1.65), 0.25)            # domain_rg.py: drain_center, drain_radius
# Raingarden_p.py, per elementMaterialTypes flag: Ks [m/d], alpha [1/m], n
SOIL = {1: (1.06, 7.5, 1.89),           # base
        2: (7.128, 14.5, 2.68),         # swale trapezoid
        3: (5.0, 8.0, 2.4)}             # upper layer


def kr_vgm(psi, alpha, n):
    m = 1.0 - 1.0 / n
    se = np.where(psi < 0.0, (1.0 + (alpha * np.abs(psi)) ** n) ** (-m), 1.0)
    return np.sqrt(se) * (1.0 - (1.0 - se ** (1.0 / m)) ** m) ** 2


def nodal_darcy(nodes, el, mat, psi):
    x, z = nodes[:, 0], nodes[:, 1]
    a, b, c = el.T
    x1, x2, z1, z2 = x[b] - x[a], x[c] - x[a], z[b] - z[a], z[c] - z[a]
    f1, f2 = psi[b] - psi[a], psi[c] - psi[a]
    det = x1 * z2 - x2 * z1
    gx, gz = (z2 * f1 - z1 * f2) / det, (x1 * f2 - x2 * f1) / det
    Ks, alpha, n = (np.array([SOIL[t][i] for t in mat]) for i in range(3))
    K = Ks * kr_vgm((psi[a] + psi[b] + psi[c]) / 3.0, alpha, n)
    qx, qz = -K * gx, -K * (gz + 1.0)
    w = np.abs(det)
    Qx, Qz, W = (np.zeros(len(x)) for _ in range(3))
    for col in (a, b, c):
        np.add.at(Qx, col, w * qx)
        np.add.at(Qz, col, w * qz)
        np.add.at(W, col, w)
    return Qx / W, Qz / W


def main():
    with h5py.File(BASE / "ls_CCS_so.h5", "r") as h5:
        last = max(int(k[3:]) for k in h5 if k.startswith("u_t") and k[3:].isdigit())
        nodes = h5[f"nodesSpatial_Domain{last}"][:, :2]
        el = h5[f"elementsSpatial_Domain{last}"][:]
        mat = h5[f"elementMaterialTypes_t{last}"][:]
        frames = [(k, h5[f"u_t{k}"][:], h5[f"pressure_head_t{k}"][:]) for k in FRAMES if k <= last]

    tri = mtri.Triangulation(nodes[:, 0], nodes[:, 1], el)
    xg, zg = np.linspace(0, L[0], 61), np.linspace(0, L[1], 101)
    XG, ZG = np.meshgrid(xg, zg)

    fig, axes = plt.subplots(1, 3, figsize=(13, 7.2), constrained_layout=True,
                             sharex=True, sharey=True)
    for ax, (k, u, psi) in zip(axes.flat, frames):
        qx, qz = nodal_darcy(nodes, el, mat, psi)
        # the interpolators mask points in the drain hole, so no streamline crosses it
        QX = mtri.LinearTriInterpolator(tri, qx)(XG, ZG)
        QZ = mtri.LinearTriInterpolator(tri, qz)(XG, ZG)
        sp = np.hypot(QX, QZ)
        field = ax.tripcolor(tri, u, cmap="viridis", vmin=0.0, vmax=1.0,
                             shading="gouraud", zorder=0, rasterized=True)
        ax.streamplot(xg, zg, QX, QZ, color="0.85", density=1.2, arrowsize=0.6,
                      linewidth=0.2 + 1.2 * np.clip(sp / np.nanmax(sp), 0, 1), zorder=1)
        # isochlors drawn last so they sit on top of the streamlines
        cs = ax.tricontour(tri, u, levels=LEVELS, colors="w", linewidths=CONTOUR_LW, zorder=3)
        ax.clabel(cs, fmt="%.1f", fontsize=8, inline=True, inline_spacing=2)
        ax.add_patch(plt.Circle(*DRAIN, facecolor="white", edgecolor="0.3", lw=1.0, zorder=4))
        ax.set_xlim(0, L[0])
        ax.set_ylim(0, L[1])
        ax.set_aspect(1.0)
        ax.set_title(f"t = {(k - 2) * DT:.4f} d", fontsize=11)
        ax.set_xlabel("x (m)")
    axes[0].set_ylabel("z (m)")
    fig.colorbar(field, ax=axes, shrink=0.8, pad=0.01,
                 label="u = C / C$_{in}$  (isochlors 0.2 / 0.4 / 0.6 in white)")
    fig.savefig(OUT_PATH, dpi=150)
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
