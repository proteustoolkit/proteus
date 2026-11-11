"""Digitise Guevara Morel & Graf (2023) Fig. 7 -- the UHD calibration curve.

    python digitize_fig7.py <page-1914.png> [--out UMD/data]

Fig. 7 is a vector plot of depth below the sand surface versus time: the lab
penetration heights z_lab of Simmons et al. (2002) as green dots, the HGS
30-realisation mean as a black line, and mean +/- sigma as red dotted lines.
This turns the rendered page (pdftoppm, 300 dpi) into

    z_lab_UHD.csv   t [min], depth [m]        the green dots
    z_hgs_UHD.csv   t [min], depth [m], sigma [m]   the black line, every minute

which is what penetration_height.py scores a run against (gap G6), and what
the paper's step-5 pass test (mean inside z_hgs +/- sigma) needs.

Calibration is from the dotted GRIDLINES, not from the frame or the tick
labels: the major ticks sit at t = 10..40 and depth = 0.2..1.0, so the four
vertical and five horizontal gridlines give a two-point-per-axis linear map
that needs no character recognition and no assumption about the axis limits.
The left frame and top frame should then come out at t = 0 and depth = 0,
which is printed as the check.

The black curve is traced column by column from the origin, taking in each
column the dark run nearest the previous row.  That is what keeps the trace
off the black text inside the axes ("r = 0.9984") and off the legend box,
both of which sit far from the curve in every column they share with it.

This is a digitisation of a printed figure: one pixel is about 0.07 min by
0.0015 m, and the dots are 6 px across, so the lab depths carry roughly
+/- 2 mm from the rendering alone, on top of whatever Simmons' own reading
of the photographs carried.
"""

import os
import sys

import numpy as np
from PIL import Image
from scipy import ndimage

# Fig. 7's bounding box on the 300-dpi rendering of p. 1914 (2481 x 3296 px).
CROP = (slice(1100, 2000), slice(1250, 2320))
T_OUT = [6, 9, 12, 15, 18, 20, 26, 32, 38, 40, 44, 46, 48]


def frame(dark):
    """Pixel rows/cols of the axes frame: the lines that are dark end to end."""
    cols = np.where(dark.mean(0) > 0.5)[0]
    rows = np.where(dark.mean(1) > 0.5)[0]
    return rows[0], rows[-1], cols[0], cols[-1]


def gridlines(grey, axis, lo, hi):
    """Pixel positions of the dotted gridlines along one axis: the columns (or
    rows) inside the frame with the most grey pixels, merged when adjacent."""
    count = grey.sum(axis)
    inside = np.zeros_like(count, dtype=bool)
    inside[lo + 4:hi - 3] = True
    hit = inside & (count > 0.25 * count[inside].max())
    labels, n = ndimage.label(hit)
    return np.array(ndimage.center_of_mass(hit, labels, range(1, n + 1))).ravel()


def main(page, out_dir):
    im = np.asarray(Image.open(page).convert("RGB")).astype(int)[CROP]
    R, G, B = im[..., 0], im[..., 1], im[..., 2]
    dark = (R < 100) & (G < 100) & (B < 100)
    grey = (abs(R - G) < 20) & (abs(G - B) < 20) & (R > 120) & (R < 225)
    green = (G > 150) & (R < 130) & (B < 130)
    red = (R > 170) & (G < 120) & (B < 120)

    top, bottom, left, right = frame(dark)
    gx = gridlines(grey, 0, left, right)
    gz = gridlines(grey, 1, top, bottom)
    assert len(gx) == 4 and len(gz) == 5, (gx, gz)
    # least-squares linear maps pixel -> (t, depth)
    at, bt = np.polyfit(gx, [10, 20, 30, 40], 1)
    az, bz = np.polyfit(gz, [0.2, 0.4, 0.6, 0.8, 1.0], 1)
    t_of = lambda x: at * x + bt
    z_of = lambda y: az * y + bz
    print("frame check: left edge t = %.2f min, top edge depth = %.3f m, "
          "right edge t = %.1f min" % (t_of(left), z_of(top), t_of(right)))
    print("resolution: %.3f min/px, %.4f m/px" % (at, az))

    # -- green dots -----------------------------------------------------------
    labels, n = ndimage.label(green)
    pts = []
    for (cy, cx), size in zip(ndimage.center_of_mass(green, labels, range(1, n + 1)),
                              ndimage.sum(green, labels, range(1, n + 1))):
        t, z = t_of(cx), z_of(cy)
        if size < 10:
            continue
        if z < 0.12 and t > 20:        # the legend's sample dot
            continue
        pts.append((t, z))
    pts.sort()
    lab = np.array(pts)
    print("\nlab dots (t [min], depth [m]):")
    for t, z in lab:
        print("  %5.1f  %.3f" % (t, z))

    # -- black mean line, traced from the origin -------------------------------
    interior = dark.copy()
    interior[:top + 3] = interior[bottom - 2:] = False
    interior[:, :left + 3] = interior[:, right - 2:] = False
    cols = np.arange(left + 3, right - 2)
    rows = np.full(len(cols), np.nan)
    prev = float(top)
    for i, x in enumerate(cols):
        hit = np.where(interior[:, x])[0]
        hit = hit[abs(hit - prev) < 25]
        if len(hit) == 0:
            continue
        # the contiguous run containing the hit nearest the previous row
        near = hit[np.argmin(abs(hit - prev))]
        run = hit[abs(hit - near) <= 3]
        rows[i] = prev = run.mean()
    ok = ~np.isnan(rows)
    t_line = t_of(cols[ok])
    z_line = z_of(rows[ok])
    print("\nmean line traced on %d of %d columns" % (ok.sum(), len(cols)))

    # -- red dotted +/- sigma: nearest red run above and below the line --------
    up, dn = [], []
    for i, x in enumerate(cols):
        if np.isnan(rows[i]):
            continue
        hit = np.where(red[:, x])[0]
        hit = hit[abs(hit - rows[i]) < 60]
        a = hit[hit < rows[i] - 2]
        b = hit[hit > rows[i] + 2]
        if len(a):
            up.append((t_of(x), z_of(a.max())))
        if len(b):
            dn.append((t_of(x), z_of(b.min())))
    up, dn = np.array(up), np.array(dn)
    print("sigma band: %d upper and %d lower red samples, first at t = %.1f min"
          % (len(up), len(dn), min(up[0, 0], dn[0, 0])))

    t_hgs = np.arange(0, int(t_line.max()) + 1)
    z_hgs = np.interp(t_hgs, t_line, z_line)
    # the band is invisible under the black line early on, i.e. sigma below
    # about half a line width (~1 mm); report 0 there rather than extrapolate
    def band(side):
        s = np.interp(t_hgs, side[:, 0], side[:, 1], left=np.nan, right=np.nan)
        return s
    sig = 0.5 * np.abs(band(dn) - band(up))
    sig = np.where(np.isnan(sig), 0.0, sig)

    print("\nHGS mean +/- sigma at the output times:")
    for t in T_OUT:
        print("  %2d  %.3f  +/- %.3f" % (t, np.interp(t, t_hgs, z_hgs),
                                        np.interp(t, t_hgs, sig)))

    os.makedirs(out_dir, exist_ok=True)
    np.savetxt(os.path.join(out_dir, "z_lab_UHD.csv"), lab, delimiter=",",
               fmt="%.2f,%.4f", header="t_min,depth_m  (Guevara Morel & Graf 2023 "
               "Fig. 7 green dots = Simmons et al. 2002 UHD; digitize_fig7.py)",
               comments="# ")
    np.savetxt(os.path.join(out_dir, "z_hgs_UHD.csv"),
               np.c_[t_hgs, z_hgs, sig], delimiter=",", fmt="%d,%.4f,%.4f",
               header="t_min,depth_mean_m,sigma_m  (Guevara Morel & Graf 2023 "
               "Fig. 7 HGS 30-realisation mean and +/- sigma; digitize_fig7.py)",
               comments="# ")

    # -- check image: what was read, drawn over the crop -----------------------
    chk = np.asarray(Image.open(page).convert("RGB"))[CROP].copy()
    for t, z in lab:
        x, y = int((t - bt) / at), int((z - bz) / az)
        chk[y - 12:y + 13, x - 1:x + 2] = (0, 0, 255)
        chk[y - 1:y + 2, x - 12:x + 13] = (0, 0, 255)
    for x, y in zip(cols[ok], rows[ok].astype(int)):
        chk[y + 8:y + 10, x] = (0, 0, 255)
    for side in (up, dn):
        for t, z in side:
            chk[int((z - bz) / az), int((t - bt) / at)] = (0, 180, 0)
    Image.fromarray(chk).save(os.path.join(out_dir, "fig7_uhd_digitisation_check.png"))
    print("\nwrote z_lab_UHD.csv, z_hgs_UHD.csv, fig7_uhd_digitisation_check.png in", out_dir)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    out = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else "UMD/data"
    main(args[0], out)
