"""
Povich et al. (2013) Fig. 2: mass fraction along x = 10 m at the last archived time for the 2-D box,
comparing velocity treatments (CV/NoCV x PE/LN) against the density-free reference.

usage: python plot_fig2.py [--step N --days D] ref_dir case_dir [case_dir ...]
Each directory holds one run's archive (*.h5). Prints the 10-90% transition thickness and the
difference from the reference profile, and writes fig2_profiles.png beside this script's caller.
"""
import glob
import os
import re
import sys

import h5py
import numpy as np
from scipy.special import erfc

X_PROFILE = 10.0
D_MOL = 1.0e-8


STEP = None  # archive step to read; None = last


def profile(run_dir):
    path = glob.glob(os.path.join(run_dir, "*.h5"))[0]
    with h5py.File(path, "r") as f:
        steps = sorted(int(m.group(1)) for k in f if (m := re.match(r"mf_t(\d+)$", k)))
        s = steps[-1] if STEP is None else STEP
        mf = f["mf_t%d" % s][:]
        nodes = f["nodesSpatial_Domain0"][:]
        elements = f["elementsSpatial_Domain0"][:]
    import matplotlib.tri as mtri
    tri = mtri.Triangulation(nodes[:, 0], nodes[:, 1], elements)
    y = np.unique(nodes[:, 1])  # the structured mesh's node rows
    c = mtri.LinearTriInterpolator(tri, mf)(np.full_like(y, X_PROFILE), y).filled(np.nan)
    return y, c, os.path.basename(os.path.normpath(run_dir))


def thickness(y, c, lo=0.1, hi=0.9):
    """Distance between the lo and hi crossings of a profile decreasing upward."""
    def cross(level):
        i = np.where((c[:-1] - level) * (c[1:] - level) <= 0)[0]
        if i.size == 0:
            return np.nan
        i = i[0]
        return y[i] + (level - c[i]) * (y[i + 1] - y[i]) / (c[i + 1] - c[i])
    return cross(lo) - cross(hi)


def main(ref_dir, case_dirs, t_seconds=86400.0 * 1000.0):
    y_ref, c_ref, _ = profile(ref_dir)
    y0 = 0.5 * (y_ref[np.argmax(c_ref < 0.5) - 1] + y_ref[np.argmax(c_ref < 0.5)])
    c_exact = 0.5 * erfc((y_ref - y0) / (2.0 * np.sqrt(D_MOL * t_seconds)))
    print("%-12s %12s %14s %14s" % ("run", "10-90% [m]", "max|c-c_ref|", "L2(c-c_ref)"))
    print("%-12s %12.3f %14s %14s" % ("analytic", thickness(y_ref, c_exact), "-", "-"))
    print("%-12s %12.3f %14.3e %14.3e" % ("reference", thickness(y_ref, c_ref),
                                          np.abs(c_ref - c_exact).max(), 0.0))
    rows = [(y_ref, c_exact, "analytic (eps=0)"), (y_ref, c_ref, "eps=0 run")]
    for d in case_dirs:
        y, c, name = profile(d)
        assert np.allclose(y, y_ref)
        e = c - c_ref
        print("%-12s %12.3f %14.3e %14.3e" % (name, thickness(y, c), np.abs(e).max(),
                                              np.sqrt(np.trapezoid(e**2, y))))
        rows.append((y, c, name))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(4.5, 6))
        for y, c, name in rows:
            ax.plot(c, y, label=name, lw=1.4 if "eps=0" in name or "analytic" in name else 1.0)
        ax.set_xlabel("mass fraction c")
        ax.set_ylabel("y [m]")
        ax.set_ylim(12, 28)
        ax.legend(fontsize=8)
        ax.set_title("2-D box, x = 10 m, t = %g d" % (t_seconds / 86400.0))
        fig.tight_layout()
        fig.savefig("fig2_profiles.png", dpi=150)
    except ImportError:
        pass


if __name__ == "__main__":
    args, days = sys.argv[1:], 1000.0
    while args and args[0].startswith("--"):
        if args[0] == "--step":
            STEP = int(args[1])
        elif args[0] == "--days":
            days = float(args[1])
        args = args[2:]
    main(args[0], args[1:], t_seconds=86400.0 * days)
