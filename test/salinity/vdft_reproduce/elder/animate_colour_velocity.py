"""Elder: movie of elder_u_colour_velocity_20yr.png -- u as a colour field, Darcy
streamlines and white 20/40/60% isochlors -- over every archived step to 20 yr.

Velocity is rebuilt from psi and u exactly as in plot_isochlors_velocity.py.
"""
from pathlib import Path
import re

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from matplotlib.animation import FFMpegWriter

from plot_isochlors_velocity import nodal_gradient, LEVELS, CONTOUR_LW, H, K, EPS, DT_DAYS

BASE = Path(__file__).resolve().parent
OUT_PATH = BASE / "elder_u_colour_velocity_20yr.mov"
FPS = 12
DPI = 150


def main():
    with h5py.File(BASE / "ls_CCS_so.h5", "r") as h5:
        # steps 0..2 are t = 0, 1e-8, 2e-8; keep step 2 as the t = 0 frame
        steps = sorted(int(k[3:]) for k in h5 if re.fullmatch(r"u_t\d+", k))
        steps = [s for s in steps if s >= 2]
        nodes = h5[f"nodesSpatial_Domain{steps[-1]}"][:, :2]
        el = h5[f"elementsSpatial_Domain{steps[-1]}"][:]
        frames = [(h5[f"u_t{s}"][:], h5[f"pressure_head_t{s}"][:]) for s in steps]

    tri = mtri.Triangulation(nodes[:, 0], nodes[:, 1], el)
    xg, zg = np.linspace(0, 600, 301), np.linspace(0, H, 76)
    XG, ZG = np.meshgrid(xg, zg)

    fig, ax = plt.subplots(figsize=(12, 3.6), constrained_layout=True)
    fig.colorbar(plt.cm.ScalarMappable(plt.Normalize(0.0, 1.0), cmap="viridis"),
                 ax=ax, shrink=0.9, pad=0.01,
                 label="u = C / C$_{max}$\n(isochlors 0.2 / 0.4 / 0.6)")

    writer = FFMpegWriter(fps=FPS, codec="libx264",
                          extra_args=["-pix_fmt", "yuv420p", "-crf", "18"])
    with writer.saving(fig, OUT_PATH, dpi=DPI):
        for s, (u, psi) in zip(steps, frames):
            ax.clear()
            gx, gz = nodal_gradient(nodes, el, psi)
            vx, vz = -K * gx, -K * (gz + 1.0 + EPS * u)
            VX = mtri.LinearTriInterpolator(tri, vx)(XG, ZG)
            VZ = mtri.LinearTriInterpolator(tri, vz)(XG, ZG)
            sp = np.hypot(VX, VZ)
            ax.tripcolor(tri, u, cmap="viridis", vmin=0.0, vmax=1.0,
                         shading="gouraud", zorder=0, rasterized=True)
            ax.streamplot(xg, zg, VX, VZ, color="0.85", density=1.2, arrowsize=0.6,
                          linewidth=0.2 + 1.2 * np.clip(sp / max(np.nanmax(sp), 1e-30), 0, 1),
                          zorder=1)
            if u.max() > LEVELS[0]:
                cs = ax.tricontour(tri, u, levels=LEVELS, colors="w",
                                   linewidths=CONTOUR_LW, zorder=3)
                ax.clabel(cs, fmt="%.1f", fontsize=8, inline=True, inline_spacing=2)
            ax.set_xlim(0, 600)
            ax.set_ylim(0, H)
            ax.set_aspect(1.0)
            ax.set_yticks([0, 50, 100, 150])
            ax.set_xlabel("x (m)")
            ax.set_ylabel("z (m)")
            t = (s - 2) * DT_DAYS
            ax.set_title(f"t = {t / 365.25:5.2f} yr  ({t:6.0f} d)", fontsize=11)
            writer.grab_frame()
    print(f"wrote {OUT_PATH}  ({len(steps)} frames at {FPS} fps)")


if __name__ == "__main__":
    main()
