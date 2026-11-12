"""Henry problem domain sketch: boundary and initial conditions.

    python plot_domain_henry_minimal.py

Domain size from domain_henry.py, boundary and initial conditions from
flow_p.py and tranport_p.py: freshwater inflow q_n = -6.6e-5 m/s with c = 0
on the left, saltwater hydrostatic head with c = 1 on the right, no flow on
the top and bottom; initially fresh water at hydrostatic head.

Drawn at the printed width (one 84 mm column) with 8 pt lettering, no
transparency, so it stays vector in EPS.
Writes domain_henry_minimal.png next to this script.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT_PATH = Path(__file__).resolve().parent / "domain_henry_minimal.png"

WIDTH_MM = 84.0           # one column of the large-format layout
LX, LZ = 2.0, 1.0         # domain_henry.py, L = (2.0, 1.0) m

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

FRESH = "#2a6fdb"
SALT = "#b23a48"

fig_w = WIDTH_MM / 25.4
fig, ax = plt.subplots(figsize=(fig_w, 0.62 * fig_w))

ax.fill([0, LX, LX, 0], [0, 0, LZ, LZ], color="#eef3f8", lw=0, zorder=1)
ax.plot([0, LX, LX, 0, 0], [0, 0, LZ, LZ, 0], color="k", lw=0.7, zorder=3)
# the Dirichlet/flux boundaries, coloured by the water they carry
ax.plot([0, 0], [0, LZ], color=FRESH, lw=2.0, zorder=4)
ax.plot([LX, LX], [0, LZ], color=SALT, lw=2.0, zorder=4)

for z in np.linspace(0.15, 0.85, 4):
    ax.annotate("", xy=(0.0, z), xytext=(-0.2, z),
                arrowprops=dict(arrowstyle="-|>", color=FRESH, lw=0.8),
                annotation_clip=False)

ax.text(0.5 * LX, LZ - 0.04, "No flow", ha="center", va="top")
ax.text(0.5 * LX, 0.04, "No flow", ha="center", va="bottom")
# \mathregular keeps the exponent in the text font, so every glyph is in the
# one sans-serif font; L_z = 1 m and rho_s/rho_0 = 1.025 (tranport_p.py) are
# written out.  Boundary labels sit inside the box beside their wall, so the
# sketch fits one column.
ax.text(0.06, 0.62, "Fresh water\nq = −6.6×10$^{\\mathregular{-5}}$ m/s\nc = 0",
        ha="left", va="center", color=FRESH)
ax.text(LX - 0.06, 0.62, "Seawater\nψ = 1.025 (1 m − z)\nc = 1",
        ha="right", va="center", color=SALT)
ax.text(0.5 * LX, 0.3, "Initial: ψ = 1 m − z, c = 0", ha="center", va="center")
ax.text(0.5 * LX, -0.1, "2 m × 1 m", ha="center", va="top", color="0.35")

ax.set_xlim(0.0, LX)
ax.set_ylim(0.0, LZ)
ax.set_aspect("equal")
ax.set_axis_off()

fig.savefig(OUT_PATH, dpi=600, bbox_inches="tight", pad_inches=0.02)
print(f"wrote {OUT_PATH}")
