from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, Polygon, Rectangle


# ============================================================
# 2D bioswale domain schematic
# Ponded saline inflow on the top strip with an internal drain
# ============================================================

# -------------------------
# Domain dimensions
# -------------------------
Lx = 3.0
z_bottom = 0.0
z_top = 5.0
ponding_depth = 0.18

# Region geometry
swale_poly = np.array(
    [
        [0.5, 3.0],
        [2.5, 3.0],
        [2.0, 1.0],
        [1.0, 1.0],
    ]
)
drain_center = np.array([1.45, 1.65])
drain_radius = 0.25

# Ponding strip on the top boundary
pond_x0 = 1.0
pond_x1 = 2.0
pond_z0 = z_top
pond_z1 = z_top + ponding_depth

# -------------------------
# Figure setup
# -------------------------
fig, ax = plt.subplots(figsize=(11, 7), dpi=300)

# Base soil region
ax.add_patch(
    Rectangle(
        (0.0, z_bottom),
        Lx,
        3.0 - z_bottom,
        facecolor="#e6cfaa",
        edgecolor="black",
        linewidth=1.4,
        zorder=1,
    )
)

# Storage zone above the swale
ax.add_patch(
    Rectangle(
        (0.0, 3.0),
        Lx,
        z_top - 3.0,
        facecolor="#f1ddbe",
        edgecolor="black",
        linewidth=1.4,
        zorder=2,
    )
)

# Bioswale media
ax.add_patch(
    Polygon(
        swale_poly,
        closed=True,
        facecolor="#98c98b",
        edgecolor="black",
        linewidth=1.6,
        zorder=4,
    )
)

# Internal drain / seepage face
drain_patch = Circle(
    drain_center,
    drain_radius,
    facecolor="#6f7d86",
    edgecolor="black",
    linewidth=1.2,
    zorder=6,
)
ax.add_patch(drain_patch)

# Ponded saline water
ax.add_patch(
    Rectangle(
        (pond_x0, pond_z0),
        pond_x1 - pond_x0,
        ponding_depth,
        facecolor="#5dade2",
        edgecolor="#0057d9",
        linewidth=1.2,
        alpha=0.95,
        zorder=7,
    )
)

xw = np.linspace(pond_x0, pond_x1, 250)
yw = pond_z1 + 0.03 * np.sin(2 * np.pi * (xw - pond_x0) / 0.20)
ax.plot(xw, yw, color="#0057d9", linewidth=1.5, zorder=8)

# Soil speckles to match the levee-style schematic
rng = np.random.default_rng(8)
for x0, y0, width, height, color, npts, zorder in [
    (0.0, 0.0, 3.0, 3.0, "#9a6b3a", 900, 3),
    (0.0, 3.0, 3.0, 2.0, "#c9a77a", 700, 3),
]:
    xs = rng.uniform(x0, x0 + width, npts)
    zs = rng.uniform(y0, y0 + height, npts)
    ax.scatter(xs, zs, s=4.5, color=color, alpha=0.18, zorder=zorder, linewidths=0)

xs = rng.uniform(0.9, 2.1, 350)
zs = rng.uniform(1.0, 3.0, 350)
swale_line_left = 3.0 - 2.0 * (xs - 0.5)
swale_line_right = 3.0 - 2.0 * (2.5 - xs)
in_swale = np.logical_and.reduce(
    [
        xs >= 1.0,
        xs <= 2.0,
        zs >= 1.0,
        zs <= 3.0,
    ]
)
in_swale |= np.logical_and.reduce(
    [
        xs >= 0.5,
        xs < 1.0,
        zs >= swale_line_left,
        zs <= 3.0,
    ]
)
in_swale |= np.logical_and.reduce(
    [
        xs > 2.0,
        xs <= 2.5,
        zs >= swale_line_right,
        zs <= 3.0,
    ]
)
outside_drain = (
    (xs - drain_center[0]) ** 2 + (zs - drain_center[1]) ** 2
    >= drain_radius ** 2
)
mask = np.logical_and(in_swale, outside_drain)
ax.scatter(xs[mask], zs[mask], s=4.5, color="#487d3b", alpha=0.16, zorder=5, linewidths=0)

# Boundary markers
for xpos in np.linspace(0.18, 0.82, 4):
    ax.plot([xpos, xpos], [z_bottom, z_bottom + 0.15], color="gray", linewidth=0.7, zorder=9)
for xpos in np.linspace(0.0, Lx - 0.16, 22):
    ax.plot([xpos, xpos + 0.12], [z_bottom - 0.10, z_bottom], color="gray", linewidth=0.7, zorder=9)

for xpos in [0.35, 0.70, 2.30, 2.65]:
    ax.plot([xpos, xpos], [z_top, z_top + 0.12], color="green", linewidth=1.0, zorder=9)
    ax.plot([xpos, xpos - 0.04], [z_top + 0.05, z_top + 0.13], color="green", linewidth=1.0, zorder=9)
    ax.plot([xpos, xpos + 0.04], [z_top + 0.05, z_top + 0.13], color="green", linewidth=1.0, zorder=9)

for xpos in [1.2, 1.5, 1.8]:
    ax.add_patch(
        FancyArrowPatch(
            (xpos, 5.15),
            (xpos, 4.58 if xpos == 1.5 else 4.70),
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=1.1,
            color="#0057d9",
            zorder=9,
        )
    )

# -------------------------
# Labels
# -------------------------
ax.text(
    1.50,
    5.56,
    "Small ponded water on top DBC",
    color="#003fb3",
    fontsize=12,
    ha="center",
    va="bottom",
    fontweight="bold",
)
# ax.text(
#     1.50,
#     5.44,
#     "Constant head  $\\psi=0.5$ m\nHigh concentration  $c=c_{in}=3.0$",
#     color="#003fb3",
#     fontsize=9.5,
#     ha="center",
#     va="top",
# )

ax.text(
    0.44,
    4.20,
    "Storage zone",
    fontsize=11,
    ha="left",
    va="center",
    fontweight="bold",
)
ax.text(
    0.44,
    3.88,
    "Region 3\nHydrostatic flow IC\n$\\psi=-z$, $c=0$",
    fontsize=8.8,
    ha="left",
    va="top",
)

ax.text(
    1.50,
    2.50,
    "Bioswale zone",
    fontsize=11,
    ha="center",
    va="center",
    fontweight="bold",
    zorder=10,
)
ax.text(
    1.50,
    2.18,
    "Region 2\nbioswale media",
    fontsize=8.4,
    ha="center",
    va="top",
    zorder=10,
)

ax.text(
    0.42,
    0.92,
    "Base soil",
    fontsize=11,
    ha="left",
    va="center",
    fontweight="bold",
)
ax.text(
    0.42,
    0.60,
    "Region 1\nHydrostatic flow IC\n$\\psi=-z$, $c=0$",
    fontsize=8.8,
    ha="left",
    va="top",
)

ax.text(
    3.56,
    1.72,
    "Internal drain / seepage face\nnatural free drainage",
    fontsize=8.8,
    color="#23323a",
    ha="left",
    va="center",
)

ax.text(
    3.62,
    4.25,
    "No-flux top boundary\noutside ponding strip\n$\\mathbf{q}\\cdot\\mathbf{n}=0$",
    fontsize=8.7,
    ha="center",
    va="center",
    color="green",
)

ax.text(
    3.66,
    2.48,
    "Lateral boundaries\nno-flow",
    fontsize=8.7,
    ha="center",
    va="center",
    color="#6d4c1a",
)

ax.text(
    1.50,
    -0.36,
    "Impermeable base / no-flow bottom boundary",
    fontsize=8.8,
    ha="center",
)

# Axis arrows
ax.annotate(
    "",
    xy=(4.18, 0.0),
    xytext=(0.0, 0.0),
    arrowprops=dict(arrowstyle="->", linewidth=1.3, color="black"),
)
ax.annotate(
    "",
    xy=(0.0, 5.82),
    xytext=(0.0, 0.0),
    arrowprops=dict(arrowstyle="->", linewidth=1.3, color="black"),
)

# -------------------------
# Axis formatting
# -------------------------
ax.set_xlim(-0.08, 4.28)
ax.set_ylim(-0.42, 5.72)
ax.set_aspect("equal", adjustable="box")

ax.set_xlabel("$x$ (m)", fontsize=11)
ax.set_ylabel("$z$ (m)", fontsize=11)

ax.set_xticks(np.arange(0.0, 3.1, 0.5))
ax.set_yticks(np.arange(0.0, 5.1, 1.0))

ax.set_title(
    "Bioswale Domain — Ponded Saline Inflow with Internal Drain",
    fontsize=13,
    pad=10,
)

for spine in ax.spines.values():
    spine.set_visible(False)

ax.grid(False)

output_path = Path(__file__).resolve().with_name("bioswale_contamination_domain.png")
plt.tight_layout()
plt.savefig(output_path, dpi=300, bbox_inches="tight")
plt.close()
