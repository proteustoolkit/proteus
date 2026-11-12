"""Write the manuscript figures as EPS.

    python make_eps.py                       # all figures, into this folder
    python make_eps.py elder                 # only the jobs whose name contains "elder"
    python make_eps.py out=DIR henry=RUN ... # write to DIR, read henry's run from RUN

Each figure is redrawn by its own plotting script, unchanged, from the run
directory of its case.  The run directories default to where each case's
run.sh puts them (RUNS below) and are overridden as CASE=DIR.  The low-order
UHD run (uhd_lo) is not made by simmons_uhd/run.sh, which sets SIMMONS_FCT=1;
give its directory explicitly.  Figure.savefig is replaced while a script
runs, so the figure the script would save as PNG/PDF is written as EPS
instead; the PNGs in the run directories are not touched.

The figure is drawn to PDF first and converted with pdftops.  EPS has no
transparency, and matplotlib's own EPS writer turns translucent artists
opaque (the levee overlays become white boxes over the sketch); pdftops
flattens transparent regions into an image instead, and compresses images.

Settings follow the Computational Geosciences artwork guidelines:
- Arial/Helvetica lettering.  Neither is installed here, so Liberation Sans
  (metric-compatible with Arial) is the fallback.
- fonts embedded as TrueType, not Type 3.
- 600 dpi at the printed width for every rasterized part, the guideline
  minimum for combination art (halftone + lettering).  The scripts draw their
  figures larger than they are printed, so the raster resolution is set from
  the width at which the manuscript includes each figure, not from the
  figure's own size.
Figure sizes and font sizes are those of each script.
"""
import os
import runpy
import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

OUT = Path(__file__).resolve().parent     # out=DIR overrides
VR = OUT.parent                           # vdft_reproduce

# Default run directory of each case: the one its run.sh writes with no
# argument.  CASE=DIR on the command line overrides it.
RUNS = {
    "zalesak": VR / "zalesak/run",            # low/ and fct/
    "mms": VR / "mms/runs",                   # a_T3_* (mms/run.sh)
    "henry": VR / "henry/run",                # fct/, lo/, henry_reference.npz
    "elder": VR / "elder/run",
    "uhd_fct": VR / "simmons_uhd/run_uhd",    # plot_fig8.py needs run_uhd* names
    "uhd_lo": VR / "simmons_uhd/run_uhd_lo",
    "bioswale": VR / "bioswale/run",
    "levee": VR / "levee_density/run",        # density/ and non_density/
}

RC = {
    "ps.fonttype": 42,
    "pdf.fonttype": 42,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "Nimbus Sans",
                        "DejaVu Sans"],
    "mathtext.fontset": "custom",
    "mathtext.rm": "Liberation Sans",
    "mathtext.it": "Liberation Sans:italic",
    "mathtext.bf": "Liberation Sans:bold",
}
DPI = 600         # guideline minimum for combination art, at printed size
BIG = 5000        # collections with more paths/points than this are rasterized
# Guideline lettering is 8-12 pt at the final size.  The scripts draw their
# figures larger than they are printed, so their text shrinks when the
# manuscript scales the figure down; every text is enlarged by one factor so
# that the smallest prints at MIN_PT, and none prints above MAX_PT.
MIN_PT, MAX_PT = 8.0, 12.0
# The guidelines put titles in the caption, not the figure.  A title wider
# than the plot also grows the saved page as fast as the text, so it can never
# print at MIN_PT.  Figure titles (suptitle) are dropped everywhere; these
# single-plot figures also have their axes title dropped.
NO_AXES_TITLE = {"henry_iso_fct", "henry_iso_lo"}

# Printed widths (mm) from the \includegraphics widths in sn-article.tex, for
# the large-format two-column layout: column 84 mm, text width 174 mm.
COL, TXT = 84.0, 174.0

_savefig = Figure.savefig

def jobs(runs):
    # (job name, script, argv,
    #  {substring of the file the script saves: (EPS name, printed width mm)}).
    # Scripts that run.sh copies into a run directory, and that read their data
    # from their own directory, are run from that copy.
    r = {k: str(v) for k, v in runs.items()}
    return [
        # drawn for one column; sn-article.tex includes it at \columnwidth
        ("henry_domain", VR / "henry/plot_domain_henry_minimal.py", [],
         {"domain_henry_minimal": ("domain_henry_minimal", COL)}),
        ("zalesak", VR / "zalesak/plot_disc.py", [r["zalesak"]],
         {"disc_r": ("disc_r", 0.9 * TXT)}),
        ("mms", VR / "mms/mms_plot.py", [r["mms"]],
         {"mms_setup.pdf": ("mms_setup", COL),
          "mms_contours.pdf": ("mms_contours", COL),
          "mms_convergence.pdf": ("mms_convergence", COL)}),
        ("henry_iso_fct", VR / "henry/plot_iso_combined.py",
         ["--fct_h5", r["henry"] + "/fct/ls_CCS_so.h5",
          "--npz", r["henry"] + "/henry_reference.npz",
          "--out", "henry_iso_fct.png", "--fct-label", "FCT"],
         {"henry_iso_fct": ("henry_iso_fct", 0.45 * TXT)}),
        ("henry_iso_lo", VR / "henry/plot_iso_combined.py",
         ["--fct_h5", r["henry"] + "/lo/ls_CCS_so.h5",
          "--npz", r["henry"] + "/henry_reference.npz",
          "--out", "henry_iso_lo.png", "--fct-label", "Low order"],
         {"henry_iso_lo": ("henry_iso_lo", 0.45 * TXT)}),
        ("elder", runs["elder"] / "plot_isochlors_velocity.py", ["colour"],
         {"elder_u_colour_velocity_20yr": ("elder_u_colour_velocity_20yr", TXT)}),
        # lab column: the digitised dye extent (simmons_uhd/data/*_mask.npz)
        ("uhd_fig8", VR / "simmons_uhd/plot_fig8.py",
         ["--colormap", "FCT=" + r["uhd_fct"], "Low order=" + r["uhd_lo"]],
         {"fig8_": ("fig8_uhd_lab_fct_lo", 0.9 * TXT)}),
        ("bioswale_domain", VR / "bioswale/plot_domain_bioswale.py", [r["bioswale"]],
         {"bioswale_contamination_domain":
          ("bioswale_contamination_domain", 0.45 * TXT)}),
        ("bioswale_three_times", runs["bioswale"] / "plot_u_three_times.py", [],
         {"u_three_times.": ("u_three_times_bioswale", 0.9 * TXT)}),
        ("levee_domain", VR / "levee_density/plot_domain_levee.py", [r["levee"]],
         {"levee_river_side_no_plume": ("levee_river_side_no_plume", 0.8 * TXT)}),
        ("levee_density", runs["levee"] / "density/plot_u_three_times_overlay.py", [],
         {"u_three_times_overlay": ("u_three_times_overlay_density", 0.49 * TXT)}),
        ("levee_non_density",
         runs["levee"] / "non_density/plot_u_three_times_overlay.py", [],
         {"u_three_times_overlay":
          ("u_three_times_overlay_non_density", 0.49 * TXT)}),
    ]


# Layout fixes for figures whose text, once enlarged to print at >= 8 pt,
# collides with something.  Each runs before the text is enlarged.

def _left_column_y_only(axes):
    # the columns share one z scale: y tick labels and label on the leftmost
    # column only, so the repeated labels do not take width from the plots
    x0 = min(ax.get_position().x0 for ax in axes)
    for ax in axes:
        if ax.get_position().x0 > x0 + 0.01:
            ax.set_ylabel("")
            ax.tick_params(axis="y", labelleft=False)
    # the end tick labels of neighbouring columns ("1.0" | "0.0") would touch;
    # 0.06 of the figure width between columns keeps them apart at 8 pt
    engine = axes[0].figure.get_layout_engine()
    if engine is not None and hasattr(engine, "set"):
        engine.set(wspace=0.06)


def _fig8(fig):
    _left_column_y_only([ax for ax in fig.axes if 1.1 < ax.get_xlim()[1] < 1.2])
    # seven x ticks (0.00 ... 1.18) do not fit a 45 mm panel at 8 pt
    for ax in fig.axes:
        if 1.1 < ax.get_xlim()[1] < 1.2:
            ax.set_xticks([0.0, 0.5, 1.0])
            ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.1f"))
        leg = ax.get_legend()
        if leg is not None:   # lab legend: lift it clear of the depth label
            leg.set_bbox_to_anchor((0.0, 0.1), transform=ax.transAxes)


def _mms_contours(fig):
    from matplotlib.contour import ContourSet
    _left_column_y_only(fig.axes)
    for ax in fig.axes:
        t = ax.title
        if t.get_text():      # clear of the "1.0" tick label below it
            ax.set_title(t.get_text(), fontsize=t.get_fontsize(),
                         color=t.get_color(), pad=12)
        for cs in ax.collections:
            if isinstance(cs, ContourSet):
                # the inline gaps were cut for the original label size
                for lab in getattr(cs, "labelTexts", []):
                    lab.set_bbox(dict(facecolor="white", edgecolor="none", pad=0.3))


def _mms_convergence(fig):
    # Both panels are L2 errors over a similar range (1e-4 to 2e-2): one shared
    # y axis, labelled once, with the variable as the panel title, so the
    # second panel's y labels do not take width from the plots.
    a, b = fig.axes[:2]
    # taller panels (about square at the printed column width): the plot is
    # larger without needing more than one column
    w, h = fig.get_size_inches()
    fig.set_size_inches(w, 1.4 * h)
    lo = min(a.get_ylim()[0], b.get_ylim()[0])
    hi = max(a.get_ylim()[1], b.get_ylim()[1])
    size = a.yaxis.label.get_fontsize()
    for ax, var in ((a, r"$c$"), (b, r"$\psi$")):
        ax.set_ylim(lo, hi)
        ax.set_title(var, fontsize=size)
    # y label and the tick labels (10^-4 ...) of both axes at the figure's
    # smallest text size (the order annotations and legend), so they print at
    # MIN_PT
    smallest = min(t.get_fontsize() for t in fig.findobj(matplotlib.text.Text)
                   if t.get_visible() and t.get_text().strip())
    a.set_ylabel(r"$\|e\|_{L^2}$", fontsize=smallest)
    for ax in (a, b):
        ax.tick_params(axis="both", which="both", labelsize=smallest)
    b.set_ylabel("")
    b.tick_params(axis="y", labelleft=False)
    # The four-entry legend covers the data wherever it sits in a panel at
    # 8 pt.  Colour encodes the scheme and line style the time step, so it is
    # split into two two-entry legends in the empty lower right of each panel
    # (small errors at coarse h).
    leg = a.get_legend()
    handles, labels = a.get_legend_handles_labels()
    size = leg.get_texts()[0].get_fontsize()
    leg.remove()
    by = dict(zip(labels, handles))
    Line2D = matplotlib.lines.Line2D
    scheme = [Line2D([], [], color=by[k].get_color(), lw=1.5)
              for k in ("FCT, $\\Delta t$", "low order, $\\Delta t$")]
    step = [Line2D([], [], color="0.35", lw=by[k].get_linewidth(), ls=by[k].get_linestyle(),
                   marker="o", ms=by[k].get_markersize(), mfc=mfc, mec="0.35")
            for k, mfc in (("FCT, $\\Delta t$", "0.35"), ("FCT, $\\Delta t/2$", "white"))]
    opts = dict(loc="lower right", fontsize=size, frameon=False, borderaxespad=0.2,
                handlelength=1.6, labelspacing=0.25)
    a.add_artist(a.legend(scheme, ["FCT", "Low order"], **opts))
    b.legend(step, ["$\\Delta t$", "$\\Delta t/2$"], **opts)


def _henry_iso(fig):
    # the three-column legend covers the curves; the left half (x < 1) is
    # empty.  The grid's alpha would flatten the whole page to an image.
    ax = fig.axes[0]
    leg = ax.get_legend()
    handles, labels = leg.legend_handles, [t.get_text() for t in leg.get_texts()]
    size = leg.get_texts()[0].get_fontsize()
    leg.remove()
    ax.legend(handles, labels, loc="center left", ncol=1, fontsize=size,
              framealpha=1.0, edgecolor="0.8")
    ax.grid(True, color="0.88", alpha=1.0)


def _elder(fig):
    # the colour-bar label is longer than the colour bar; the isochlors are
    # named in the caption
    for ax in fig.axes:
        lab = ax.yaxis.label
        if "(isochlors" in lab.get_text():
            lab.set_text(lab.get_text().split("(isochlors")[0].strip())


TWEAKS = {"fig8_uhd_lab_fct_lo": _fig8, "mms_contours": _mms_contours,
          "mms_convergence": _mms_convergence, "henry_iso_fct": _henry_iso,
          "henry_iso_lo": _henry_iso, "elder_u_colour_velocity_20yr": _elder}


def rasterize_large(fig):
    # 3D surfaces and meshes with many cells are far smaller as an image than
    # as one vector path per cell (disc_r: ~200k triangles)
    for ax in fig.axes:
        for c in ax.collections:
            n = max(len(c.get_paths()), len(c.get_offsets()))
            if ax.name == "3d" or n > BIG:
                c.set_rasterized(True)


def saved_width(fig, kwargs):
    # width in inches of the page savefig writes
    fig.canvas.draw()
    if kwargs.get("bbox_inches") == "tight" or (
            kwargs.get("bbox_inches") is None
            and plt.rcParams["savefig.bbox"] == "tight"):
        pad = kwargs.get("pad_inches", plt.rcParams["savefig.pad_inches"])
        return fig.get_tightbbox(fig.canvas.get_renderer()).width + 2 * float(pad)
    return fig.get_size_inches()[0]


def enlarge_text(fig, shrink):
    # shrink = saved width / printed width: a font of s pt prints at s/shrink
    texts = [t for t in fig.findobj(matplotlib.text.Text)
             if t.get_visible() and t.get_text().strip()]
    if not texts:
        return
    f = max(1.0, MIN_PT * shrink / min(t.get_fontsize() for t in texts))
    new = lambda size: min(size * f, MAX_PT * shrink)
    # tick labels are re-created at draw time from the axis' tick settings, so
    # set those from the original tick size (read before any text is changed)
    ticks = []
    for ax in fig.axes:
        for name, axis in (("x", ax.xaxis), ("y", ax.yaxis)):
            for which, tl in (("major", axis.get_major_ticks()),
                              ("minor", axis.get_minor_ticks())):
                if tl:
                    ticks.append((ax, name, which, tl[0].label1.get_fontsize()))
    for t, size in [(t, t.get_fontsize()) for t in texts]:
        t.set_fontsize(new(size))
    for ax, name, which, size in ticks:
        ax.tick_params(axis=name, which=which, labelsize=new(size))


def plot_sizes(fig, kwargs, printed_in):
    # (width, height) of every axes in printed inches, i.e. scaled by the
    # printed width over the width of the page savefig writes
    s = printed_in / saved_width(fig, kwargs)
    r = fig.canvas.get_renderer()
    return [(b.width / fig.dpi * s, b.height / fig.dpi * s)
            for b in (ax.get_window_extent(r) for ax in fig.axes)]


def keep_plot_size(fig, kwargs, printed_in, before):
    # Enlarged text takes room from the plots inside a figure of fixed size.
    # Grow the figure's height until every axes prints at least as tall as it
    # did before (aspect-locked axes regain their width with it); stop when a
    # step no longer helps, which is when the width is the limit.
    def lack(sizes):
        return max(b[1] / a[1] for a, b in zip(sizes, before) if a[1] > 0)
    current = lack(plot_sizes(fig, kwargs, printed_in))
    for _ in range(12):
        if current <= 1.01:
            break
        w, h = fig.get_size_inches()
        fig.set_size_inches(w, h * min(current, 1.5))
        new = lack(plot_sizes(fig, kwargs, printed_in))
        if new > 0.99 * current:     # no gain: undo the step
            fig.set_size_inches(w, h)
            break
        current = new


def run_script(script, argv, names):
    written = []

    def savefig(self, fname, *args, **kwargs):
        base = Path(str(fname)).name
        hit = next((v for key, v in names.items() if key in base), None)
        if hit is None or hit[0] in written:
            return
        eps, width_mm = hit
        rasterize_large(self)
        printed_in = width_mm / 25.4
        before = plot_sizes(self, kwargs, printed_in)
        if self.get_suptitle():
            self.suptitle("")
        if eps in NO_AXES_TITLE:
            for ax in self.axes:
                ax.set_title("")
        if eps in TWEAKS:
            TWEAKS[eps](self)
        # enlarging the text widens the saved page, which LaTeX then shrinks
        # a little more: repeat until the smallest prints at MIN_PT (to 1%).
        # Growing the figure to give the plots back their size can change the
        # page width too, so the two alternate.
        for _ in range(3):
            for _ in range(10):
                shrink = saved_width(self, kwargs) / printed_in
                smallest = min(t.get_fontsize() for t in self.findobj(matplotlib.text.Text)
                               if t.get_visible() and t.get_text().strip()) / shrink
                if smallest >= 0.99 * MIN_PT:
                    break
                enlarge_text(self, shrink)
            keep_plot_size(self, kwargs, printed_in, before)
        after = plot_sizes(self, kwargs, printed_in)
        w_in = saved_width(self, kwargs)
        # raster resolution so that it is DPI at the printed width
        dpi = DPI * printed_in / w_in
        kwargs.update(format="pdf", dpi=dpi)
        with tempfile.TemporaryDirectory() as tmp:
            pdf = Path(tmp) / f"{eps}.pdf"
            _savefig(self, pdf, *args, **kwargs)
            # -rasterize whenneeded: only pages with transparency are
            # flattened, at the same resolution as the rasterized artists
            subprocess.run(["pdftops", "-eps", "-level3", "-rasterize",
                            "whenneeded", "-r", str(int(round(dpi))), str(pdf),
                            str(OUT / f"{eps}.eps")], check=True)
        sizes = [t.get_fontsize() * printed_in / w_in
                 for t in self.findobj(matplotlib.text.Text)
                 if t.get_visible() and t.get_text().strip()]
        print(f"    {eps}: {w_in:.2f} in drawn -> {width_mm:.0f} mm printed, "
              f"raster {dpi:.0f} dpi on the page = {DPI} dpi printed, "
              f"text {min(sizes):.1f}-{max(sizes):.1f} pt printed, "
              f"plots {min(a[0] / b[0] for a, b in zip(after, before) if b[0] > 0):.2f}"
              f" x width, {min(a[1] / b[1] for a, b in zip(after, before) if b[1] > 0):.2f}"
              f" x height of the original", flush=True)
        written.append(eps)

    plt.rcdefaults()
    plt.rcParams.update(RC)
    Figure.savefig = savefig
    cwd, saved_argv = os.getcwd(), sys.argv
    with tempfile.TemporaryDirectory() as tmp:
        # scripts that write relative paths write into a scratch directory
        os.chdir(tmp)
        sys.argv = [str(script)] + argv
        # run_path does not put the script's directory on sys.path, which
        # scripts importing their sibling modules (cases.py, ...) need
        sys.path.insert(0, str(Path(script).parent))
        try:
            runpy.run_path(str(script), run_name="__main__")
        finally:
            sys.path.pop(0)
            os.chdir(cwd)
            sys.argv = saved_argv
            Figure.savefig = _savefig
            plt.close("all")
    return written


if __name__ == "__main__":
    only, runs = "", dict(RUNS)
    for arg in sys.argv[1:]:
        key, sep, value = arg.partition("=")
        if not sep:
            only = arg
        elif key == "out":
            OUT = Path(value).resolve()
            OUT.mkdir(parents=True, exist_ok=True)
        elif key in runs:
            runs[key] = Path(value).resolve()
        else:
            raise SystemExit(f"unknown run {key!r}; known: out, {', '.join(runs)}")
    for name, script, argv, names in jobs(runs):
        if only not in name:
            continue
        try:
            done = run_script(script, argv, names)
            expected = [v[0] for v in names.values()]
            missing = [n for n in expected if n not in done]
            print(f"[{name}] wrote {done}" + (f"  MISSING {missing}" if missing else ""),
                  flush=True)
        except Exception as exc:
            print(f"[{name}] FAILED: {type(exc).__name__}: {exc}", flush=True)
