"""Plume penetration height from a finished Simmons benchmark archive.

    python penetration_height.py simmons_shd_l0_t0_s0

"Plume penetration height z_num is the z-position of the deepest fingertip that
has the solute mass fraction of 70% of omega_max" -- so in the normalised
variable the diagnostic is the minimum z over all nodes with u >= 0.70.

Reads the h5 archive the way the working Elder/Liu post-processing does: nodes,
elements and the solution are all taken from the SAME archived step
(`nodesSpatial_Domain{n}` / `u_t{n}`), which is what keeps the node ordering
consistent -- the h5 field ordering is a permutation of the CCS ordering, so
mixing sources silently scrambles the geometry.

Also computes the two goodness-of-fit measures the paper reports, once
`data/z_lab_<CASE>.csv` exists (gap G6):

    r        correlation coefficient, Eq. (19); acceptable if > 0.90
    eps_z    L2 relative deviation,   Eq. (20)
"""

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import h5py
import numpy as np

try:
    from . import cases
except ImportError:
    import cases

# Archives live where the run was launched, not necessarily beside this script
# (runs are commonly done in a scratch subdirectory of symlinks).
HERE = Path.cwd()


def load_times(run_name):
    """Output times.  A serial run only writes <name>.xmf; the _complete.xmf is
    produced by the archive gather, so try both."""
    for suffix in ("_complete.xmf", ".xmf"):
        path = HERE / (run_name + suffix)
        if not path.exists():
            continue
        try:
            root = ET.parse(str(path)).getroot()
            times = [float(g.find("Time").attrib["Value"])
                     for g in root.findall('.//Grid[@GridType="Uniform"]')]
            if times:
                return np.array(times)
        except ET.ParseError:
            continue
    return None


def penetration_series(h5_path, threshold=cases.PENETRATION_THRESHOLD):
    """(steps, z_num) -- deepest z at which u >= threshold, per archived step.

    A step in which the plume has not yet formed (no node above threshold)
    yields the top of the domain, which is the physically right answer for a
    penetration depth measured from the surface.
    """
    steps, depths = [], []
    with h5py.File(str(h5_path), "r") as f:
        keys = sorted(int(k[len("u_t"):]) for k in f.keys()
                      if k.startswith("u_t"))
        for n in keys:
            u = f["u_t%d" % n][:]
            nodes = f["nodesSpatial_Domain%d" % n][:]
            hit = u >= threshold
            steps.append(n)
            depths.append(float(nodes[hit, 1].min()) if hit.any() else cases.LZ)
    return np.array(steps), np.array(depths)


def correlation(z_lab, z_num):
    """Eq. (19).  Plain Pearson r between observed and simulated."""
    a = z_lab - z_lab.mean()
    b = z_num - z_num.mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))


def eps_z(z_lab, z_num):
    """Eq. (20).  L2 relative deviation."""
    return float(np.linalg.norm(z_num - z_lab) / np.linalg.norm(z_lab))


def main(run_name):
    h5_path = HERE / (run_name + ".h5")

    if not h5_path.exists():
        raise SystemExit("no archive %s -- has the run finished?" % h5_path)

    steps, z_num = penetration_series(h5_path)
    times = load_times(run_name)

    print("%-8s %-12s %s" % ("step", "t [s]", "z_num [m]"))
    for i, (n, z) in enumerate(zip(steps, z_num)):
        t = times[n] if times is not None and n < len(times) else float("nan")
        print("%-8d %-12.4g %.5f" % (n, t, z))

    # Depth below the surface is what Figs. 7/9 plot (axis inverted, 0 at top).
    print("\nfinal penetration depth below surface: %.4f m"
          % (cases.LZ - z_num[-1]))

    case_name = run_name.split("_")[1].upper()
    lab_path = HERE / "data" / ("z_lab_%s.csv" % case_name)
    if lab_path.exists() and times is not None:
        lab = np.loadtxt(str(lab_path), delimiter=",", skiprows=1)
        t_lab, z_lab = lab[:, 0], lab[:, 1]
        z_at_lab = np.interp(t_lab, times[steps], z_num)
        print("\nr      = %.4f   (acceptable > 0.90, Hill 1998)"
              % correlation(z_lab, z_at_lab))
        print("eps_z  = %.4f" % eps_z(z_lab, z_at_lab))
    else:
        print("\n(no %s yet -- gap G6, digitise Figs. 7/9 to score the run)"
              % lab_path.name)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    main(sys.argv[1])
