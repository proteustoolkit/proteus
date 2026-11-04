import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle, FancyArrowPatch

# ============================================================
# 2D levee domain schematic
# River-side water only, no country-side infiltration, no plume
# ============================================================

# -------------------------
# Domain dimensions
# -------------------------
Lx = 160.0
z_bottom = 0.0
z_top = 60.0
water_table_z = 45.0

# River water level
river_level = 48.0

# Levee geometry
# Points are approximate and can be modified
levee_x = np.array([20, 35, 55, 70, 92, 105])
levee_z = np.array([45, 48, 53, 57, 57, 53])

# Country-side ground surface
country_x = np.array([105, 160])
country_z = np.array([53, 53])

# -------------------------
# Figure setup
# -------------------------
fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

# Saturated zone
ax.add_patch(
    Rectangle(
        (0, z_bottom),
        Lx,
        water_table_z - z_bottom,
        facecolor="#c7e8ff",
        edgecolor="none",
        alpha=0.95,
        zorder=1,
    )
)

# Unsaturated/domain material polygon
domain_poly = np.array(
    [
        [0, z_bottom],
        [160, z_bottom],
        [160, 53],
        [105, 53],
        [92, 57],
        [70, 57],
        [55, 53],
        [35, 48],
        [20, 45],
        [0, 45],
    ]
)

ax.add_patch(
    Polygon(
        domain_poly,
        closed=True,
        facecolor="#f4e4ca",
        edgecolor="black",
        linewidth=1.4,
        zorder=2,
    )
)

# Re-draw saturated overlay only inside rectangle for clarity
ax.add_patch(
    Rectangle(
        (0, z_bottom),
        Lx,
        water_table_z - z_bottom,
        facecolor="#c7e8ff",
        edgecolor="none",
        alpha=0.85,
        zorder=3,
    )
)

# Add light soil speckles in unsaturated zone
rng = np.random.default_rng(10)
xs = rng.uniform(0, Lx, 1800)
zs = rng.uniform(water_table_z, 58, 1800)

def ground_elevation(x):
    if x <= 20:
        return 45.0
    elif x <= 35:
        return 45.0 + (48.0 - 45.0) * (x - 20) / (35 - 20)
    elif x <= 55:
        return 48.0 + (53.0 - 48.0) * (x - 35) / (55 - 35)
    elif x <= 70:
        return 53.0 + (57.0 - 53.0) * (x - 55) / (70 - 55)
    elif x <= 92:
        return 57.0
    elif x <= 105:
        return 57.0 + (53.0 - 57.0) * (x - 92) / (105 - 92)
    else:
        return 53.0

mask = np.array([z <= ground_elevation(x) for x, z in zip(xs, zs)])
ax.scatter(xs[mask], zs[mask], s=0.25, color="#9a6b3a", alpha=0.20, zorder=4)

# River water polygon
river_poly = np.array(
    [
        [0, water_table_z],
        [20, water_table_z],
        [35, 48],
        [50, river_level],
        [0, river_level],
    ]
)

ax.add_patch(
    Polygon(
        river_poly,
        closed=True,
        facecolor="#5dade2",
        edgecolor="#0057d9",
        linewidth=1.2,
        alpha=0.95,
        zorder=5,
    )
)

# River surface wiggle
xw = np.linspace(0, 50, 250)
yw = river_level + 0.08 * np.sin(2 * np.pi * xw / 4.5)
ax.plot(xw, yw, color="#0057d9", linewidth=1.5, zorder=6)

# Water table
ax.plot(
    [0, Lx],
    [water_table_z, water_table_z],
    "--",
    color="#0057d9",
    linewidth=2.0,
    zorder=7,
)

# Bottom impermeable hatch
ax.plot([0, Lx], [z_bottom, z_bottom], color="black", linewidth=1.6, zorder=8)
for x in np.arange(0, Lx, 4):
    ax.plot([x, x + 3], [z_bottom - 1.2, z_bottom], color="gray", linewidth=0.7, zorder=8)

# Right boundary dashed
ax.plot([Lx, Lx], [z_bottom, 53], "--", color="black", linewidth=1.4, zorder=8)

# Small grass on country side
for x in np.arange(112, 158, 10):
    ax.plot([x, x], [53, 54.4], color="green", linewidth=1.0, zorder=9)
    ax.plot([x, x - 1.0], [53.5, 54.5], color="green", linewidth=1.0, zorder=9)
    ax.plot([x, x + 1.0], [53.5, 54.5], color="green", linewidth=1.0, zorder=9)

# River water marker
ax.scatter([6], [river_level + 1.2], marker="v", s=90, color="#0057d9", zorder=10)

# -------------------------
# Labels
# -------------------------
ax.text(15, 53.2, "RIVER\n(River-side water)", color="#003fb3", fontsize=10,
        ha="center", va="bottom", fontweight="bold")
ax.text(14, 50.0, "Constant head\n$\\psi=\\psi_r$\nHigh concentration\n$c=c_r$",
        color="#003fb3", fontsize=8.5, ha="center", va="top")

ax.text(79, 58.4, "Levee", fontsize=11, ha="center", va="bottom", fontweight="bold")

ax.text(135, 57.2, "Country side\n(No infiltration)", color="green",
        fontsize=10, ha="center", va="bottom", fontweight="bold")
ax.text(135, 55.0, "No-flux top boundary\n$\\mathbf{q}\\cdot\\mathbf{n}=0$",
        fontsize=8.5, ha="center", va="top")

ax.text(82, 50.0, "Unsaturated zone\n(Vadose zone)", fontsize=10,
        ha="center", va="center")
ax.text(82, 24.0, "Saturated zone\n(Groundwater)", fontsize=10,
        ha="center", va="center")

ax.text(136, water_table_z + 1.3, "Initial water table\n$z=45$ m",
        color="#0057d9", fontsize=9, ha="center", va="bottom")

ax.text(165.5, 28, "Outflow boundary\n(free drainage)\n$\\mathbf{q}\\cdot\\mathbf{n}=0$\n$c$: natural outflow",
        fontsize=8.5, ha="center", va="center", color="#003fb3")

ax.text(80, -4.4, "Impermeable base / no-flow bottom boundary",
        fontsize=8.5, ha="center")

# Axis arrows
ax.annotate("", xy=(168, 0), xytext=(0, 0),
            arrowprops=dict(arrowstyle="->", linewidth=1.3, color="black"))
ax.annotate("", xy=(0, 64), xytext=(0, 0),
            arrowprops=dict(arrowstyle="->", linewidth=1.3, color="black"))

# -------------------------
# Axis formatting
# -------------------------
ax.set_xlim(-2, 172)
ax.set_ylim(-6, 65)
ax.set_aspect("equal", adjustable="box")

ax.set_xlabel("$x$ (m)", fontsize=11)
ax.set_ylabel("$z$ (m)", fontsize=11)

ax.set_xticks(np.arange(0, 161, 20))
ax.set_yticks([0, 15, 30, 45, 60])

ax.set_title(
    "Levee Domain — River-Side Water, No Country-Side Infiltration",
    fontsize=13,
    pad=12,
)

# Remove default spines because axis arrows are drawn manually
for spine in ax.spines.values():
    spine.set_visible(False)

ax.grid(False)

plt.tight_layout()
plt.savefig("levee_river_side_no_plume.png", dpi=300, bbox_inches="tight")
plt.close()
