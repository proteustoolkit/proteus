"""Bioswale domain sketch: soil zones, drain, boundary and initial conditions.

    python plot_domain_bioswale.py [run_dir]

The geometry is read from the Triangle files the simulation was meshed with,
bio2d_N.node / bio2d_N.ele / bio2d_N.poly in run_dir (default: this script's
directory, where run.sh copies it): the zones are the mesh elements coloured
by their region attribute, the outlines are the PSLG segments, the drain is
the PSLG hole.  The boundary and initial conditions are those of
Raingarden_p.py and tranport_p.py.

Drawn at the printed width (0.45 of the 174 mm text width in sn-article.tex)
with 8 pt lettering, no transparency, so it stays vector in EPS.
Writes bioswale_contamination_domain.png next to this script.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np

BASE = Path(__file__).resolve().parent
RUN = Path(sys.argv[1]) if len(sys.argv) > 1 else BASE
OUT_PATH = BASE / "bioswale_contamination_domain.png"

WIDTH_MM = 0.45 * 174.0   # \includegraphics width in sn-article.tex

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
    "font.size": 8,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def rows(path):
    # Triangle file: whitespace-separated numbers, '#' starts a comment
    out = []
    for line in Path(path).read_text().splitlines():
        tok = line.split("#", 1)[0].split()
        if tok:
            out.append([float(t) for t in tok])
    return out


node = rows(RUN / "bio2d_N.node")
xy = np.array([r[1:3] for r in node[1:1 + int(node[0][0])]])
ele = rows(RUN / "bio2d_N.ele")
ne = int(ele[0][0])
tris = np.array([r[1:4] for r in ele[1:1 + ne]], dtype=int) - 1
region = np.array([int(r[4]) for r in ele[1:1 + ne]])

poly = rows(RUN / "bio2d_N.poly")
nv = int(poly[0][0])
pv = np.array([r[1:3] for r in poly[1:1 + nv]])
ns = int(poly[1 + nv][0])
seg = np.array([r[1:3] for r in poly[2 + nv:2 + nv + ns]], dtype=int) - 1
seg_mark = np.array([int(r[3]) for r in poly[2 + nv:2 + nv + ns]])
k = 2 + nv + ns
nh = int(poly[k][0])
holes = np.array([r[1:3] for r in poly[k + 1:k + 1 + nh]])

# boundaryTags of domain_rg.py: bottom 1, top 2, left 3, right 4, swale 5,
# drain 6; region flags: 1 base soil, 2 bioswale zone, 3 storage zone
DRAIN = 6
LX, LZ = pv[:, 0].max(), pv[:, 1].max()
print("elements per region:",
      {r: int((region == r).sum()) for r in np.unique(region)})

ZONE = {1: ("Base soil", "#e3cfae"),
        2: ("Bioswale zone", "#cfe2bf"),
        3: ("Storage zone", "#f2e6d2")}
WATER = "#2a6fdb"

fig_w = WIDTH_MM / 25.4
fig, ax = plt.subplots(figsize=(fig_w, 1.25 * fig_w))

tri = mtri.Triangulation(xy[:, 0], xy[:, 1], tris)
ax.tripcolor(tri, facecolors=region, vmin=0.5, vmax=3.5,
             cmap=matplotlib.colors.ListedColormap([ZONE[r][1] for r in (1, 2, 3)]),
             edgecolors="face", linewidth=0.3, zorder=1)

# PSLG segments: outer boundary, zone interfaces and the drain wall
for (a, b), m in zip(seg, seg_mark):
    ax.plot(pv[[a, b], 0], pv[[a, b], 1],
            color=WATER if m == DRAIN else "k",
            lw=1.2 if m == DRAIN else 0.7, zorder=3, solid_capstyle="round")
# the drain is the PSLG hole: left unfilled (white)
drain_v = pv[np.unique(seg[seg_mark == DRAIN])]
centre = holes[0]
order = np.argsort(np.arctan2(drain_v[:, 1] - centre[1], drain_v[:, 0] - centre[0]))
ax.fill(drain_v[order, 0], drain_v[order, 1], color="white", zorder=2)

# Ponded strip on the top boundary, L/3 <= x <= 2L/3 (Raingarden_p.py,
# tranport_p.py): psi = 0.5 m, c = 1
x0, x1 = LX / 3.0, 2.0 * LX / 3.0
ax.fill([x0, x1, x1, x0], [LZ, LZ, LZ + 0.12, LZ + 0.12], color="#a9cdf2",
        zorder=2, lw=0)
ax.plot([x0, x1], [LZ, LZ], color=WATER, lw=2.0, zorder=4, solid_capstyle="butt")
ax.text(0.5 * (x0 + x1), LZ + 0.2, "Ponding: ψ = 0.5 m, c = 1",
        ha="center", va="bottom", color=WATER)

# zone names in the open part of each zone, clear of the drain
ax.text(0.5 * LX, 4.15, ZONE[3][0], ha="center", va="center")
ax.text(0.5 * LX, 2.55, ZONE[2][0], ha="center", va="center")
ax.text(0.5 * LX, 0.65, ZONE[1][0], ha="center", va="center")
ax.text(0.5 * LX, 0.3, "Initial: ψ = −3 m, c = 0", ha="center", va="center")

# drain label outside the domain, with a leader to the wall
ax.annotate("Drain:\nseepage face", xy=(centre[0] + 0.25, centre[1]),
            xytext=(LX + 0.12, centre[1]), ha="left", va="center", color=WATER,
            arrowprops=dict(arrowstyle="-", color=WATER, lw=0.6),
            annotation_clip=False)
ax.text(LX + 0.12, 4.0, "Other\nboundaries:\nno flow", ha="left", va="center",
        clip_on=False)

ax.set_xlim(0.0, LX)
ax.set_ylim(0.0, LZ + 0.75)
ax.set_aspect("equal")
ax.set_xticks(np.arange(0.0, LX + 0.01, 1.0))
ax.set_yticks(np.arange(0.0, LZ + 0.01, 1.0))
ax.set_xlabel("x (m)")
ax.set_ylabel("z (m)")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.spines["left"].set_bounds(0.0, LZ)

fig.savefig(OUT_PATH, dpi=600, bbox_inches="tight", pad_inches=0.02)
print(f"wrote {OUT_PATH}")
