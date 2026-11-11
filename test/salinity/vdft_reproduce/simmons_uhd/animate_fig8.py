"""Movie of plot_fig8.py --colormap's numerical panel over every archived step.

    python animate_fig8.py <run_dir>

One frame per archived output time (the t = 1e-8 / 2e-8 bookkeeping steps are
skipped), written as fig8_<case>_colormap.mov in the run directory.  The lab
photos exist only at the Fig. 8 times, so there is no lab column.
"""

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from matplotlib.animation import FFMpegWriter

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cases
from plot_fig8 import BANDS, DYE_CMAP

FPS = 2
DPI = 200


def main(run_dir):
    d = Path(run_dir).resolve()
    base = [p for p in d.glob("*.h5") if p.stat().st_size > 0][0].with_suffix("")
    case_name = d.name.split("_")[1].upper()
    root = ET.parse(str(base) + ".xmf").getroot()
    times = np.array([float(g.find("Time").attrib["Value"])
                      for g in root.findall('.//Grid[@GridType="Uniform"]')])
    steps = [s for s, t in enumerate(times) if s == 0 or t > 1.0]

    fig, ax = plt.subplots(figsize=(5.6, 5.0), constrained_layout=True)
    levels = BANDS[:-1] + [1.0]
    cbar = None

    out = d / ("fig8_%s_colormap.mov" % case_name.lower())
    writer = FFMpegWriter(fps=FPS, codec="libx264",
                          extra_args=["-pix_fmt", "yuv420p", "-crf", "18"])
    with h5py.File(str(base) + ".h5", "r") as f, writer.saving(fig, str(out), dpi=DPI):
        for s in steps:
            nodes = f["nodesSpatial_Domain%d" % s][:]
            elements = f["elementsSpatial_Domain%d" % s][:]
            u = f["u_t%d" % s][:]
            hit = u >= cases.PENETRATION_THRESHOLD
            depth = cases.LZ - (float(nodes[hit, 1].min()) if hit.any() else cases.LZ)

            ax.clear()
            tri = mtri.Triangulation(nodes[:, 0], nodes[:, 1], elements)
            field = ax.tricontourf(tri, np.clip(u, 0.0, 1.0), levels=levels, cmap=DYE_CMAP)
            # built from the first frame's bands so the bar matches the field;
            # the levels are fixed, so it stays valid for every frame
            if cbar is None:
                cbar = fig.colorbar(field, ax=ax, shrink=0.8,
                                    label=r"$u = \omega / \omega_{\max}$")
            ax.plot([cases.SOURCE_X0, cases.SOURCE_X1], [cases.LZ, cases.LZ],
                    color="k", lw=3, solid_capstyle="butt")
            ax.set_xlim(0.0, cases.LX)
            ax.set_ylim(0.0, cases.LZ)
            ax.set_aspect("equal")
            ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.18])
            ax.set_xlabel("x [m]")
            ax.set_ylabel("z [m]")
            ax.set_title("%s numerical, t = %d min" % (case_name, round(times[s] / 60.0)),
                         fontsize=10)
            ax.text(0.985, 0.04, "%.3f m" % depth, transform=ax.transAxes,
                    fontsize=8, ha="right", color="0.25")
            writer.grab_frame()
            print("t = %5.0f min  depth %.3f m" % (times[s] / 60.0, depth))
    print("wrote %s  (%d frames at %d fps)" % (out, len(steps), FPS))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    main(sys.argv[1])
