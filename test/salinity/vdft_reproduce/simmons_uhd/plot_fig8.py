"""Fig. 8 / Fig. 6 style fingering-pattern panels for one Simmons run.

    python plot_fig8.py <run_dir> [<run_dir> ...]
    python plot_fig8.py --turbo <run_dir> [<run_dir> ...]
    python plot_fig8.py --rainbow <run_dir> [<run_dir> ...]
    python plot_fig8.py --colormap <run_dir> [<run_dir> ...]

With --turbo the run panels are the continuous field on the turbo colour map,
with --rainbow the field in evenly distributed 0.1 bands of the rainbow
colour map; both with a shared colour bar and no lab outline, written as
fig8_<case>_turbo.png / fig8_<case>_rainbow.png beside the band version.
With --colormap the run panels are the paper's 0.05/0.2/0.7 bands on a
beige-to-dye colour map matched to the lab photo, with a shared colour bar, as
fig8_<case>_colormap.png.

Draws the case's Fig. 8 comparison times (SHD 40/60, UMD 30/50, SMD 90/120,
and UHD 32/40 for Fig. 6) with the paper's rendering convention: discrete
filled concentration bands on a beige ground, plume core in red, equal aspect,
x in [0, 1.18] and z in [0, 1.06].  Several run directories are drawn as
extra columns, so realisations sit side by side the way the paper's do.

This is the FIGURE-8 view -- pattern, not penetration curve.  Use
plot_panels.py when you want the isochlors and the penetration number.
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
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cases

#: The paper renders the plume as a few discrete bands, not a continuous ramp:
#: beige ground, a pale dilute halo, then the red core.  0.7 is the benchmark's
#: own penetration threshold, so the red boundary IS the reported contour.
BANDS = [0.0, 0.05, 0.20, 0.70, 1.01]
COLORS = ["#f2e2cf", "#f7c9ae", "#f08a6a", "#d81f18"]

#: --colormap: beige sand to the lab's purple dye, so the run reads like the photo.
DYE_CMAP = LinearSegmentedColormap.from_list(
    "beige_dye", ["#f2e2cf", "#e9a8cf", "#841369"])

#: Fig. 8's comparison times per case, in minutes (UHD is Fig. 6).
FIG_TIMES = {"UHD": (32, 40), "SHD": (40, 60), "UMD": (30, 50),
             "SMD": (90, 120)}


def _load(run_dir):
    d = Path(run_dir)
    h5s = [p for p in d.glob("*.h5") if p.stat().st_size > 0]
    if not h5s:
        raise SystemExit("no non-empty archive in %s" % d)
    base = h5s[0].with_suffix("")
    root = ET.parse(str(base) + ".xmf").getroot()
    times = np.array([float(g.find("Time").attrib["Value"])
                      for g in root.findall('.//Grid[@GridType="Uniform"]')])
    tmp = Path(tempfile.mkdtemp()) / "snapshot.h5"
    shutil.copy2(str(base) + ".h5", str(tmp))
    # Title columns by the RUN DIRECTORY, not the archive name: two runs of the
    # same case/level/seed differing only in a numerics setting produce byte-
    # identical archive names, so the archive name cannot tell them apart.
    return d.name, base, times, tmp


def _paper_panel(case_name):
    """The published Fig. 8 strip for this case, if it has been saved.

    Drop a crop of the paper's figure at data/fig8_<case>_paper.png and it is
    drawn as the leftmost column.  It is a bitmap of someone else's figure, so
    it is shown as-is with no axes -- never resampled onto our grid and never
    treated as data.

    A data/fig8_<case>_paper_frame.png is preferred when present: that is the
    same strip cut to the INSIDE of the paper's axes frames (x in [0, LX],
    z in [0, LZ]), so it can be drawn with that physical extent on axes
    identical to the run panels and the two columns share one scale.
    """
    d = Path(__file__).resolve().parent / "data"
    p = d / ("fig8_%s_paper_frame.png" % case_name.lower())
    if p.exists():
        return p, True
    p = d / ("fig8_%s_paper.png" % case_name.lower())
    return (p, False) if p.exists() else None


def _lab_outline(case_name, minutes):
    """(x, z, plume) of the digitised lab dye mask at this time, if present.

    Written by digitize_uhd_photos.py at data/fig8_<case>_lab_photo_t<T>_mask.npz.
    Drawn on the run panels as the lab plume's border line.  Not x-shifted:
    the UHD photo's dyed span is centred on the source to within 1 cm.
    """
    p = (Path(__file__).resolve().parent / "data"
         / ("fig8_%s_lab_photo_t%d_mask.npz" % (case_name.lower(), minutes)))
    if not p.exists():
        return None
    m = np.load(str(p))
    return m["x"], m["z"], m["plume"]


def main(run_dirs, case_name=None, style=None):
    case_name = (case_name or cases.CASES and
                 Path(run_dirs[0]).name.split("_")[1].upper())
    if case_name not in FIG_TIMES:
        case_name = "SMD"
    want = [m * 60.0 for m in FIG_TIMES[case_name]]

    loaded = [_load(r) for r in run_dirs]
    paper = _paper_panel(case_name)
    ncol = len(loaded) + (1 if paper else 0)
    nrow = len(want)

    fig, axes = plt.subplots(nrow, ncol, squeeze=False,
                             figsize=(4.6 * ncol, 4.4 * nrow),
                             constrained_layout=True)

    col0 = 0
    if paper:
        col0 = 1
        paper, framed = paper
        img = plt.imread(str(paper))
        # The saved crop is the paper's own two-row strip, so split it in
        # half vertically and put each half beside the matching time.
        h = img.shape[0] // nrow
        for i in range(nrow):
            ax = axes[i][0]
            if framed:
                ax.imshow(img[i * h:(i + 1) * h],
                          extent=[0.0, cases.LX, 0.0, cases.LZ])
                ax.set_xlim(0.0, cases.LX)
                ax.set_ylim(0.0, cases.LZ)
                ax.set_aspect("equal")
                ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.18])
                if i == nrow - 1:
                    ax.set_xlabel("x [m]")
            else:
                ax.imshow(img[i * h:(i + 1) * h])
                ax.set_axis_off()
            if i == 0:
                ax.set_title("Guevara Morel & Graf (2023), Fig. %s (lab)"
                             % ("6a" if case_name == "UHD" else "8"),
                             fontsize=9)

    print("%-28s %8s %10s %10s %10s" % ("run", "t(min)", "u_min", "u_max",
                                        "depth(m)"))
    for j, (label, base, times, tmp) in enumerate(loaded):
        with h5py.File(str(tmp), "r") as f:
            for i, w in enumerate(want):
                s = int(np.argmin(np.abs(times - w)))
                ax = axes[i][col0 + j]
                nodes = f["nodesSpatial_Domain%d" % s][:]
                elements = f["elementsSpatial_Domain%d" % s][:]
                u = f["u_t%d" % s][:]

                hit = u >= cases.PENETRATION_THRESHOLD
                z_pen = float(nodes[hit, 1].min()) if hit.any() else cases.LZ
                depth = cases.LZ - z_pen
                print("%-28s %8.0f %10.2e %10.6f %10.4f"
                      % (label[:28], times[s] / 60.0, u.min(), u.max(),
                         depth))

                tri = mtri.Triangulation(nodes[:, 0], nodes[:, 1], elements)
                if style == "turbo":
                    field = ax.tripcolor(tri, np.clip(u, 0.0, 1.0), cmap="turbo",
                                         vmin=0.0, vmax=1.0, shading="gouraud")
                elif style == "rainbow":
                    field = ax.tricontourf(tri, np.clip(u, 0.0, 1.0),
                                           levels=np.linspace(0.0, 1.0, 11),
                                           cmap="rainbow")
                elif style == "colormap":
                    field = ax.tricontourf(tri, np.clip(u, 0.0, 1.0),
                                           levels=BANDS[:-1] + [1.0], cmap=DYE_CMAP)
                else:
                    ax.tricontourf(tri, u, levels=BANDS, colors=COLORS,
                                   extend="neither")
                    ax.set_facecolor(COLORS[0])
                ax.plot([cases.SOURCE_X0, cases.SOURCE_X1],
                        [cases.LZ, cases.LZ], color="k", lw=3,
                        solid_capstyle="butt")
                lab = None if style else _lab_outline(case_name, FIG_TIMES[case_name][i])
                if lab is not None:
                    ax.contour(lab[0], lab[1], lab[2], levels=[0.5],
                               colors="#1f3a93", linewidths=1.0)
                    if i == 0:
                        ax.plot([], [], color="#1f3a93", lw=1.0,
                                label="lab dye outline (digitised)")
                        ax.legend(loc="lower left", fontsize=7, frameon=False)
                ax.set_xlim(0.0, cases.LX)
                ax.set_ylim(0.0, cases.LZ)
                ax.set_aspect("equal")
                ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.18])
                if col0 + j == 0:
                    ax.set_ylabel("t = %d min\n\nz [m]"
                                  % round(times[s] / 60.0))
                elif col0 and j == 0:
                    # The paper column is leftmost, so label the row there and
                    # keep a bare axis label on the first run column.
                    if framed:
                        axes[i][0].set_ylabel("t = %d min\n\nz [m]"
                                              % round(times[s] / 60.0))
                    else:
                        axes[i][0].set_ylabel("t = %d min"
                                              % round(times[s] / 60.0))
                        axes[i][0].axis("on")
                        axes[i][0].set_xticks([])
                        axes[i][0].set_yticks([])
                    ax.set_ylabel("z [m]")
                if i == nrow - 1:
                    ax.set_xlabel("x [m]")
                if i == 0:
                    ax.set_title("Numerical" if style == "colormap" else label,
                                 fontsize=9)
                # white reads on turbo/rainbow; the beige ground needs grey
                ax.text(0.985, 0.04, "%.3f m" % depth, transform=ax.transAxes,
                        fontsize=8, ha="right",
                        color="w" if style in ("turbo", "rainbow") else "0.25")

    if style:
        fig.colorbar(field, ax=[axes[i][col0 + j] for i in range(nrow)
                                for j in range(len(loaded))],
                     shrink=0.7, label=r"$u = \omega / \omega_{\max}$")
    if style != "colormap":
        fig.suptitle("%s -- Fig. 8 comparison times" % case_name)
    out = Path(run_dirs[0]) / ("fig8_%s%s.png" % (case_name.lower(),
                                                  "_" + style if style else ""))
    fig.savefig(str(out), dpi=150, bbox_inches="tight")
    print("wrote %s" % out)
    return out


if __name__ == "__main__":
    flags = {"--turbo": "turbo", "--rainbow": "rainbow", "--colormap": "colormap"}
    dirs = [a for a in sys.argv[1:] if a not in flags]
    if not dirs:
        raise SystemExit(__doc__)
    style = next((s for f, s in flags.items() if f in sys.argv), None)
    main(dirs, style=style)
