"""Zalesak's disk after one revolution as 3D surfaces, (a) FCT and (b) low
order, height and colour both u.

    python plot_disc.py <run dir>        (the run dir of run.sh; has low/ and fct/)

Reads the last archived step of <run dir>/{low,fct}/tadr_level_*.h5.  The deck's
analyticalSolution is the initial condition, which after one full rotation is
also the exact solution, so u_analytical is the reference.  Prints min, max and
the lumped L1 error of each scheme; writes <run dir>/disc_r.png.
"""
import sys
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np

RUN = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent / "run"
CMAP = "coolwarm"
ELEV, AZIM = 18.0, -100.0               # low side view, degrees


def load(d):
    with h5py.File(next(d.glob("tadr_level_*.h5")), "r") as h5:
        last = max(int(k[3:]) for k in h5 if k.startswith("u_t") and k[3:].isdigit())
        nodes = h5[f"nodesSpatial_Domain{last}"][:, :2]
        el = h5[f"elementsSpatial_Domain{last}"][:]
        return nodes, el, h5[f"u_t{last}"][:], h5[f"u_analytical_t{last}"][:]


def lumped_mass(nodes, el):
    x, y = nodes[:, 0], nodes[:, 1]
    a, b, c = el.T
    area = 0.5 * np.abs((x[b] - x[a]) * (y[c] - y[a]) - (x[c] - x[a]) * (y[b] - y[a]))
    m = np.zeros(len(nodes))
    for k in (a, b, c):
        np.add.at(m, k, area / 3.0)
    return m


fields = {}
for name, label in (("low", "Low order"), ("fct", "FCT")):
    nodes, el, u, ue = load(RUN / name)
    m = lumped_mass(nodes, el)
    print("%-9s u_min %+.3e  u_max %.6f  L1 error %.4e"
          % (label, u.min(), u.max(), np.sum(m * np.abs(u - ue))))
    # each run keeps its own mesh: under MPI the archive numbers nodes by that run's
    # partition, so one run's u does not line up with the other's nodes
    fields[name] = (mtri.Triangulation(nodes[:, 0], nodes[:, 1], el), el, u)
panels = [("(a) FCT", fields["fct"]), ("(b) Low order", fields["low"])]

norm = matplotlib.colors.Normalize(0.0, 1.0)
plt.rcParams["font.size"] = 18
fig = plt.figure(figsize=(16, 7))
for i, (label, (tri, el, u)) in enumerate(panels):
    # each panel a little wider than half the figure, so the zoomed surface is
    # not clipped; the 3D axes background is transparent where they overlap
    ax = fig.add_axes([-0.03 + 0.45 * i, -0.08, 0.5, 1.16], projection="3d")
    ax.set_facecolor("none")
    surf = ax.plot_trisurf(tri, u, linewidth=0.0, antialiased=False)
    # matplotlib colours a triangle by its vertex mean, which stripes the disk's
    # vertical walls (triangles spanning 0..1); the vertex max paints them as the
    # disk top, and on the smooth parts differs from the mean by one cell's change
    surf.set_facecolor(plt.get_cmap(CMAP)(norm(u[el].max(axis=1))))
    ax.set_zlim(0.0, 1.0)
    ax.view_init(elev=ELEV, azim=AZIM)
    ax.set_box_aspect((1.0, 1.0, 0.55), zoom=1.25)
    ax.set_axis_off()
    ax.set_title(label, y=0.88)
fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=CMAP),
             cax=fig.add_axes([0.95, 0.25, 0.015, 0.5]), label="u")
fig.savefig(RUN / "disc_r.png", dpi=200, bbox_inches="tight", pad_inches=0.05)
print("wrote", RUN / "disc_r.png")
