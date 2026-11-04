from pathlib import Path
import xml.etree.ElementTree as ET

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np


BASE = Path(__file__).resolve().parent
H5_PATH = BASE / "ls_CCS_so.h5"
XMF_PATH = BASE / "ls_CCS_so.xmf"
OUT_PATH = BASE / "u_three_times.png"
U_MIN = 0.0
U_MAX = 1.0


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
    if last >= 400:
        requested = [100, 400, last]
    else:
        requested = [max(1, last // 4), max(1, last // 2), last]

    chosen = []
    for step in requested:
        nearest = min(steps, key=lambda s: (abs(s - step), s))
        if nearest not in chosen:
            chosen.append(nearest)
    return chosen


def main() -> None:
    xmf_times = load_times(XMF_PATH)

    with h5py.File(H5_PATH, "r") as h5_file:
        steps_in_h5 = available_steps(h5_file)
        if not steps_in_h5:
            raise RuntimeError(f"no u_t* datasets found in {H5_PATH}")
        steps = choose_steps(steps_in_h5)

        all_nodes = [h5_file[f"nodesSpatial_Domain{step}"][:] for step in steps]
        stacked_nodes = np.vstack(all_nodes)
        x_limits = (
            float(stacked_nodes[:, 0].min()) - 0.05,
            float(stacked_nodes[:, 0].max()) + 0.05,
        )
        z_limits = (
            float(stacked_nodes[:, 1].min()) - 0.05,
            float(stacked_nodes[:, 1].max()) + 0.05,
        )

        n = len(steps)
        x_span = x_limits[1] - x_limits[0]
        z_span = z_limits[1] - z_limits[0]
        panel_width = 4.0
        panel_height = panel_width * z_span / max(x_span, 1e-9)
        fig, axes = plt.subplots(
            nrows=1,
            ncols=n,
            figsize=(panel_width * n + 1.2, panel_height + 0.8),
            sharex=True,
            sharey=True,
            constrained_layout=True,
        )
        if n == 1:
            axes = [axes]

        mappable = None
        for ax, step in zip(axes, steps):
            nodes = h5_file[f"nodesSpatial_Domain{step}"][:]
            elements = h5_file[f"elementsSpatial_Domain{step}"][:]
            u = h5_file[f"u_t{step}"][:]
            u_plot = np.clip(u, U_MIN, U_MAX)

            tri = mtri.Triangulation(nodes[:, 0], nodes[:, 1], elements)
            mappable = ax.tripcolor(
                tri,
                u_plot,
                shading="gouraud",
                cmap="turbo",
                vmin=U_MIN,
                vmax=U_MAX,
                rasterized=True,
            )
            ax.set_xlim(*x_limits)
            ax.set_ylim(*z_limits)
            ax.set_aspect("equal")
            ax.set_xlabel("x [m]")
            t_str = f"t approx {xmf_times[step]:.4f} d, " if step in xmf_times else ""
            ax.set_title(
                f"time, {t_str}"
                #f"u shown on [{U_MIN:.1f}, {U_MAX:.1f}]"
            )

    axes[0].set_ylabel("z [m]")
    colorbar = fig.colorbar(mappable, ax=axes, fraction=0.04, pad=0.02)
    colorbar.set_label("u (normalized concentration)")
    fig.suptitle(
        "Bioswale_contamination",
        fontsize=17,
    )
    fig.savefig(OUT_PATH, dpi=300, bbox_inches="tight")

    print(f"wrote {OUT_PATH}")
    print(f"steps {steps}")


if __name__ == "__main__":
    main()
