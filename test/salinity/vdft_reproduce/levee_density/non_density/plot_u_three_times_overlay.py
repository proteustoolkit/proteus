from pathlib import Path
import xml.etree.ElementTree as ET

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from matplotlib.patches import Polygon, Rectangle


BASE = Path(__file__).resolve().parent
H5_PATH = BASE / "ls_CCS_so.h5"
XMF_PATH = BASE / "ls_CCS_so.xmf"
OUT_PATH = BASE / "u_three_times_overlay.png"

# Simulation domain (true data coordinates of the field).
X_LIMITS = (0.0, 160.0)
Z_LIMITS = (0.0, 60.0)

# Transparency of the concentration overlay (0 = invisible, 1 = opaque).
OVERLAY_ALPHA = 0.65

# Mask out u values at or below this threshold so the schematic shows through
# in regions with no plume.
ZERO_MASK_THRESHOLD = 0.1


def draw_levee_schematic(ax) -> None:
    """Draw the same schematic as plot_domain_levee.py directly in data
    coordinates, so the simulation field's (0, 0) maps exactly onto the
    bottom-left of the schematic axes."""
    Lx = 160.0
    z_bottom = 0.0
    water_table_z = 45.0
    river_level = 48.0

    # Saturated zone (under water table).
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

    # Unsaturated/domain material polygon (the levee body and ground).
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

    # Saturated overlay (re-draw on top of domain polygon for clarity).
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

    # River water polygon.
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

    # River surface wiggle.
    xw = np.linspace(0, 50, 250)
    yw = river_level + 0.08 * np.sin(2 * np.pi * xw / 4.5)
    ax.plot(xw, yw, color="#0057d9", linewidth=1.5, zorder=6)

    # Water table dashed line.
    ax.plot(
        [0, Lx],
        [water_table_z, water_table_z],
        "--",
        color="#0057d9",
        linewidth=1.6,
        zorder=7,
    )

    # Bottom impermeable hatch.
    ax.plot([0, Lx], [z_bottom, z_bottom], color="black", linewidth=1.6, zorder=8)
    for x in np.arange(0, Lx, 4):
        ax.plot(
            [x, x + 3],
            [z_bottom - 1.2, z_bottom],
            color="gray",
            linewidth=0.7,
            zorder=8,
        )

    # Right boundary dashed.
    ax.plot([Lx, Lx], [z_bottom, 53], "--", color="black", linewidth=1.4, zorder=8)

    # Small grass on country side.
    for x in np.arange(112, 158, 10):
        ax.plot([x, x], [53, 54.4], color="green", linewidth=1.0, zorder=9)
        ax.plot([x, x - 1.0], [53.5, 54.5], color="green", linewidth=1.0, zorder=9)
        ax.plot([x, x + 1.0], [53.5, 54.5], color="green", linewidth=1.0, zorder=9)


def available_steps(h5_file) -> list[int]:
    return sorted(
        int(k[len("u_t"):]) for k in h5_file.keys() if k.startswith("u_t")
    )


def load_times(xmf_path: Path) -> dict[int, float]:
    if not xmf_path.exists():
        return {}
    root = ET.parse(xmf_path).getroot()
    out: dict[int, float] = {}
    for grid in root.findall('.//Grid[@GridType="Uniform"]'):
        time_el = grid.find("Time")
        if time_el is None:
            continue
        name = time_el.attrib.get("Name")
        if name is None:
            continue
        try:
            out[int(name)] = float(time_el.attrib["Value"])
        except (KeyError, ValueError):
            continue
    return out


def choose_steps(steps: list[int]) -> list[int]:
    last = steps[-1]
    requested = [100, 400, last]
    return [s for s in requested if s <= last]


def main() -> None:
    xmf_times = load_times(XMF_PATH)

    with h5py.File(H5_PATH, "r") as h5_file:
        steps_in_h5 = available_steps(h5_file)
        if not steps_in_h5:
            raise RuntimeError(f"no u_t* datasets found in {H5_PATH}")
        steps = choose_steps(steps_in_h5)

        n = len(steps)
        fig, axes = plt.subplots(
            nrows=n,
            ncols=1,
            figsize=(10, 4.3 * n + 0.5),
            sharex=True,
            sharey=True,
            constrained_layout=True,
        )
        if n == 1:
            axes = [axes]

        mappable = None
        for ax, step in zip(axes, steps):
            # Schematic background drawn in data coordinates -> automatic alignment.
            draw_levee_schematic(ax)

            # Concentration field.
            nodes = h5_file[f"nodesSpatial_Domain{step}"][:]
            elements = h5_file[f"elementsSpatial_Domain{step}"][:]
            u = h5_file[f"u_t{step}"][:]

            tri = mtri.Triangulation(nodes[:, 0], nodes[:, 1], elements)

            u_masked = np.ma.masked_less_equal(u, ZERO_MASK_THRESHOLD)
            cmap = plt.get_cmap("turbo").copy()
            cmap.set_bad(color=(1.0, 1.0, 1.0, 0.0))

            # Mask whole triangles whose three vertices are all below the
            # threshold; gouraud shading otherwise bleeds non-zero colour into
            # neighbouring zero-only cells.
            triangle_mask = np.all(u[elements] <= ZERO_MASK_THRESHOLD, axis=1)
            tri.set_mask(triangle_mask)

            mappable = ax.tripcolor(
                tri,
                u_masked,
                shading="gouraud",
                cmap=cmap,
                vmin=0.0,
                vmax=0.7,
                rasterized=True,
                alpha=OVERLAY_ALPHA,
                zorder=10,
                edgecolors="none",
                linewidth=0,
                antialiased=False,
            )

            ax.set_xlim(*X_LIMITS)
            ax.set_ylim(*Z_LIMITS)
            ax.set_aspect("equal")
            ax.set_ylabel("z [m]")
            t_str = f"t ≈ {xmf_times[step]:.1f} d, " if step in xmf_times else ""
            ax.set_title(
                f"time  {t_str}"
                #f"u ∈ [{max(0.0, float(u.min())):.3f}, {float(u.max()):.3f}]"
            )

    axes[-1].set_xlabel("x [m]")
    colorbar = fig.colorbar(mappable, ax=axes, fraction=0.06, pad=0.04)
    colorbar.set_label("u (normalized concentration)")
    fig.suptitle(
        "Contamination profile at Levee without density coupling",
        fontsize=18,
    )
    fig.savefig(OUT_PATH, dpi=200, bbox_inches="tight")

    print(f"wrote {OUT_PATH}")
    print(f"steps {steps}")


if __name__ == "__main__":
    main()
