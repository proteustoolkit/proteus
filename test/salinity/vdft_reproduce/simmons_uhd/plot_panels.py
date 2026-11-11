"""Multi-time isochlor panels for one Simmons benchmark run.

    python plot_panels.py <run_dir> [minutes,minutes,...]

Draws the 0.2 and 0.7 isochlors over a light concentration field at the
requested reporting times.  0.7 is the benchmark's own penetration-height
threshold ("Definition of the new benchmark problem"); 0.2 shows the dilute
leading edge, which is where fingering appears first.

Reads nodes, elements and u from the SAME archived step -- the h5 field
ordering is a permutation of the CCS ordering, so mixing sources scrambles the
geometry -- and copies the archive before opening it, so a live run is safe.
"""

import shutil
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cases

LEVELS = [0.2, cases.PENETRATION_THRESHOLD]


def main(run_dir, minutes=None):
    d = Path(run_dir)
    h5s = [p for p in d.glob("*.h5") if p.stat().st_size > 0]
    if not h5s:
        raise SystemExit("no non-empty archive in %s" % d)
    base = h5s[0].with_suffix("")

    root = ET.parse(str(base) + ".xmf").getroot()
    times = np.array([float(g.find("Time").attrib["Value"])
                      for g in root.findall('.//Grid[@GridType="Uniform"]')])

    if minutes:
        want = [float(m) * 60.0 for m in minutes.split(",")]
    else:                       # skip the 1e-8 s startup kicks
        want = [t for t in times if t > 1.0][::3][:6]
    steps = [int(np.argmin(np.abs(times - w))) for w in want]

    tmp = Path(tempfile.mkdtemp()) / "snapshot.h5"
    shutil.copy2(str(base) + ".h5", str(tmp))

    fig, axes = plt.subplots(len(steps), 1, figsize=(8.5, 2.7 * len(steps)),
                             sharex=True, constrained_layout=True)
    axes = np.atleast_1d(axes)

    print("%5s %9s %10s %10s %12s" % ("step", "t(min)", "u_min", "u_max",
                                      "depth(m)"))
    with h5py.File(str(tmp), "r") as f:
        for ax, s in zip(axes, steps):
            nodes = f["nodesSpatial_Domain%d" % s][:]
            elements = f["elementsSpatial_Domain%d" % s][:]
            u = f["u_t%d" % s][:]

            hit = u >= cases.PENETRATION_THRESHOLD
            z_pen = float(nodes[hit, 1].min()) if hit.any() else cases.LZ
            depth = cases.LZ - z_pen
            print("%5d %9.1f %10.2e %10.6f %12.4f"
                  % (s, times[s] / 60.0, u.min(), u.max(), depth))

            tri = mtri.Triangulation(nodes[:, 0], nodes[:, 1], elements)
            ax.tripcolor(tri, u, shading="gouraud", cmap="Blues",
                         vmin=0.0, vmax=1.0, rasterized=True)
            ax.tricontour(tri, u, levels=LEVELS, colors=["0.35", "crimson"],
                          linewidths=1.3)
            ax.plot([cases.SOURCE_X0, cases.SOURCE_X1], [cases.LZ, cases.LZ],
                    color="k", lw=4, solid_capstyle="butt")
            ax.axhline(z_pen, color="crimson", ls=":", lw=0.8)
            ax.set_xlim(0.0, cases.LX)
            ax.set_ylim(0.0, cases.LZ)
            ax.set_aspect("equal")
            ax.set_ylabel("z [m]")
            ax.text(0.012, 0.06, "t = %d min" % round(times[s] / 60.0),
                    transform=ax.transAxes, fontsize=10)
            ax.text(0.99, 0.06, "penetration depth %.3f m" % depth,
                    transform=ax.transAxes, fontsize=8, ha="right",
                    color="0.3")

    axes[-1].set_xlabel("x [m]")
    handles = [plt.Line2D([], [], color="0.35", lw=1.3),
               plt.Line2D([], [], color="crimson", lw=1.3)]
    axes[0].legend(handles, ["u = 0.2", "u = 0.7 (penetration)"],
                   loc="lower left", bbox_to_anchor=(0.0, 0.14), fontsize=9,
                   framealpha=0.9)
    fig.suptitle("%s" % base.name)

    out = d / ("%s_panels.png" % base.name)
    fig.savefig(str(out), dpi=150, bbox_inches="tight")
    print("wrote %s" % out)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
