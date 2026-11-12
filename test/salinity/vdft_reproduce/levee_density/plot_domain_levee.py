"""Levee domain sketch: cross section, initial water table, boundary conditions.

    python plot_domain_levee.py [run_dir]

The geometry is read from levee_case_d.poly, the PSLG the simulation was
meshed from, in run_dir/density (default: the current directory, from which
run.sh calls this script).  The river-side Dirichlet face is the set of
segments tagged leftTop; the boundary and initial conditions are those of
density/Raingarden_p.py and density/tranport_p.py.

Drawn at the printed width (0.8 of the 174 mm text width in sn-article.tex)
with 8 pt lettering, no transparency, so it stays vector in EPS.
Writes levee_river_side_no_plume.png in the current directory.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RUN = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd()
POLY = RUN / "density" / "levee_case_d.poly"
OUT_PATH = Path("levee_river_side_no_plume.png")

WIDTH_MM = 0.8 * 174.0    # \includegraphics width in sn-article.tex

# Raingarden_p.py: initial water table and river stage (m)
WATER_TABLE = 45.0
RIVER_STAGE = 48.0
# boundaryTags of density/domain_rg.py: bottom 1, top 2, left 3, right 4,
# leftTop 5 (the wetted river face)
LEFT_TOP = 5

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


poly = rows(POLY)
nv = int(poly[0][0])
pv = np.array([r[1:3] for r in poly[1:1 + nv]])
ns = int(poly[1 + nv][0])
seg = np.array([r[1:3] for r in poly[2 + nv:2 + nv + ns]], dtype=int) - 1
seg_mark = np.array([int(r[3]) for r in poly[2 + nv:2 + nv + ns]])

# the PSLG is one closed loop: walk it for the domain outline
nxt = {a: b for a, b in seg}
loop = [seg[0][0]]
while nxt[loop[-1]] != loop[0]:
    loop.append(nxt[loop[-1]])
outline = pv[loop]
LX, ZMAX = pv[:, 0].max(), pv[:, 1].max()

# river water: between the wetted face and the river stage
wet = pv[np.unique(seg[seg_mark == LEFT_TOP])]
wet = wet[np.argsort(wet[:, 0])]
river = np.vstack([wet, [[wet[0, 0], RIVER_STAGE]]])

WATER = "#2a6fdb"
fig_w = WIDTH_MM / 25.4
fig, ax = plt.subplots(figsize=(fig_w, 0.5 * fig_w))

# unsaturated soil, then the saturated part below the initial water table;
# the ground is at or above the water table everywhere, so that part is the
# full-width rectangle
ax.fill(outline[:, 0], outline[:, 1], color="#f2e6d2", lw=0, zorder=1)
ax.fill([0, LX, LX, 0], [0, 0, WATER_TABLE, WATER_TABLE], color="#d6e8f7",
        lw=0, zorder=1)
ax.fill(river[:, 0], river[:, 1], color="#a9cdf2", lw=0, zorder=2)
ax.plot([0, LX], [WATER_TABLE, WATER_TABLE], "--", color=WATER, lw=0.8, zorder=3)

for (a, b), m in zip(seg, seg_mark):
    ax.plot(pv[[a, b], 0], pv[[a, b], 1],
            color=WATER if m == LEFT_TOP else "k",
            lw=2.0 if m == LEFT_TOP else 0.7, zorder=4, solid_capstyle="round")

crest = pv[pv[:, 1] == ZMAX]
ax.text(crest[:, 0].mean(), ZMAX + 1.0, "Levee", ha="center", va="bottom")
# above the river, clear of the levee slope (z = 52 m at x = 51 m)
ax.text(2.0, RIVER_STAGE + 5.5, "River: ψ = %g − z, c = 0.7" % RIVER_STAGE,
        ha="left", va="bottom", color=WATER)
ax.text(0.5 * LX, 22.0, "Saturated zone", ha="center", va="center")
ax.text(0.5 * LX, 16.0, "Initial: ψ = %g − z, c = 0" % WATER_TABLE,
        ha="center", va="center")
ax.text(132.0, 49.0, "Unsaturated zone", ha="center", va="center")
ax.text(132.0, WATER_TABLE - 1.0, "Initial water table, z = %g m" % WATER_TABLE,
        ha="center", va="top", color=WATER)
ax.text(LX + 3.0, 0.5 * ZMAX, "Other\nboundaries:\nno flow", ha="left",
        va="center", clip_on=False)

ax.set_xlim(0.0, LX)
ax.set_ylim(0.0, ZMAX + 6.0)
ax.set_aspect("equal")
ax.set_xticks(np.arange(0.0, LX + 0.1, 20.0))
ax.set_yticks([0, 15, 30, WATER_TABLE, ZMAX])
ax.set_xlabel("x (m)")
ax.set_ylabel("z (m)")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.spines["left"].set_bounds(0.0, ZMAX)

fig.savefig(OUT_PATH, dpi=600, bbox_inches="tight", pad_inches=0.02)
print(f"wrote {OUT_PATH.resolve()}")
