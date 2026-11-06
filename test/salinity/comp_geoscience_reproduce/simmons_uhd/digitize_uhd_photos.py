"""Digitise the UHD laboratory photographs (Simmons et al. 2002, as printed in
Guevara Morel & Graf 2023) into plume masks and penetration fronts.

    python digitize_uhd_photos.py <photos.png> [--out UMD/data]

The input is the lab-photo column of the UHD figure: a stack of panels, one
per time (t = 32 and 40 min), each a photograph pasted inside a framed axes
box.  Per panel this writes

    fig8_uhd_lab_photo_t<T>_front.csv   x [m], penetration depth [m]  (deepest
                                        dyed pixel in every pixel column)
    fig8_uhd_lab_photo_t<T>_mask.npz    x, z, plume (bool) -- the dye mask in
                                        physical coordinates, for finger
                                        counting / overlays
    fig8_uhd_lab_digitised_summary.csv  max depth, mean depth under the source,
                                        dyed area
    fig8_uhd_lab_digitisation_check.png what was read, drawn over the panels

No finger counts: the pale dye halo between the dark fingers passes the dye
threshold too, so the envelope mask is one connected lobe (1-3 "fingers"
where the eye sees 6-8), and a darker-core threshold gives counts that move
with the threshold at this 3.7 mm/px resolution.  Count on the saved mask by
hand if a count is needed.

Calibration is from the panel FRAMES, which are drawn at the axis limits
x in [0, LX], z in [0, LZ]; the frames come out at LX/LZ aspect to 0.2%, so
the frame rectangle is a two-point calibration with no character recognition.
Frames are the rows/columns that are black end to end -- the photograph has
no such row or column (sand is mid-grey, the dye is dark magenta, and the
tank's black base is a few rows high but not full width).

Depth is measured from the TOP FRAME, i.e. the tank top / sand surface, the
same datum as digitize_fig8.py and the paper's Fig. 7 curve.  As for the SMD
photographs, the ruler shows the photo is not x-registered to the box (the
ruler at the tank wall sits at x ~ 1.01-1.06 of a 1.178 box), so depths and
finger counts are usable and absolute x positions carry an offset of up to
~0.15 m.  One pixel is 3.7 mm; quote depths to +/- 1 cm at best.

Reuses digitize_fig8.py's dye threshold and ruler cut so the UHD numbers are
read exactly as the SMD ones were.
"""

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

import cases
from digitize_fig8 import mask_of, ruler_column

ROW_MINUTES = [32, 40]

#: A frame line is black (grey below this) over at least this fraction of the
#: image width (rows) or of the panel height (columns).
FRAME_BLACK = 0.15
FRAME_FRAC = 0.6


def black_lines(gray, frac):
    """Indices along axis 0 of gray whose black fraction exceeds frac, merged
    when adjacent (a printed line can be 2 px wide)."""
    hit = np.where((gray < FRAME_BLACK).mean(axis=1) > frac)[0]
    runs = []
    for i in hit:
        if runs and i <= runs[-1][-1] + 1:
            runs[-1].append(i)
        else:
            runs.append([i])
    return [int(round(np.mean(r))) for r in runs]


def panel_boxes(gray):
    """[(c0, c1, r0, r1)] of the stacked panels' frames, top to bottom."""
    rows = black_lines(gray, FRAME_FRAC)
    if len(rows) != 2 * len(ROW_MINUTES):
        raise SystemExit("expected %d frame rows, got %s" % (2 * len(ROW_MINUTES), rows))
    boxes = []
    for r0, r1 in zip(rows[0::2], rows[1::2]):
        cols = black_lines(gray[r0 + 2:r1 - 1].T, 0.9)
        if len(cols) != 2:
            raise SystemExit("expected 2 frame columns in rows %d-%d, got %s" % (r0, r1, cols))
        boxes.append((cols[0], cols[1], r0, r1))
    return boxes


def main(fig_path, out_dir):
    img = mpimg.imread(str(fig_path))[:, :, :3].astype(float)
    if img.max() > 1.001:
        img /= 255.0
    boxes = panel_boxes(img.mean(axis=2))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, len(ROW_MINUTES), figsize=(5.0 * len(ROW_MINUTES), 4.8),
                             squeeze=False, constrained_layout=True)
    summary = []
    for i, minutes in enumerate(ROW_MINUTES):
        c0, c1, r0, r1 = boxes[i]
        print("t = %d min: frame cols %d-%d rows %d-%d, aspect %.3f (LX/LZ %.3f)"
              % (minutes, c0, c1, r0, r1, (c1 - c0) / (r1 - r0), cases.LX / cases.LZ))
        sub = img[r0 + 1:r1, c0 + 1:c1]
        plume, _ = mask_of(sub, True)

        # Pixel centres -> physical coordinates from the frame rectangle.
        x = (np.arange(sub.shape[1]) + 1.0) / (c1 - c0) * cases.LX
        z = cases.LZ - (np.arange(sub.shape[0]) + 1.0) / (r1 - r0) * cases.LZ
        cut = ruler_column(sub.mean(axis=2))
        if cut is not None:
            x, plume = x[:cut], plume[:, :cut]
            print("   ruler at x = %.3f m, mask cut there" % x[-1])
        # Paler patches inside the dyed region drop below the threshold; they
        # are inside the plume, so fill them and keep one outer border.
        plume = ndimage.binary_fill_holes(plume)

        front = np.full(x.size, np.nan)
        for k in range(x.size):
            hit = np.nonzero(plume[:, k])[0]
            if hit.size:
                front[k] = cases.LZ - z[hit.max()]
        depth = float(np.nanmax(front))
        under = (x >= cases.SOURCE_X0) & (x <= cases.SOURCE_X1)
        mean_depth = float(np.nanmean(front[under]))
        dx, dz = cases.LX / (c1 - c0), cases.LZ / (r1 - r0)
        area = float(plume.sum()) * dx * dz

        name = "fig8_uhd_lab_photo_t%d" % minutes
        with open(out_dir / (name + "_front.csv"), "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["x_m", "penetration_depth_m"])
            for xv, fv in zip(x, front):
                w.writerow(["%.4f" % xv, "" if not np.isfinite(fv) else "%.4f" % fv])
        np.savez_compressed(out_dir / (name + "_mask.npz"), x=x, z=z, plume=plume)
        summary.append((minutes, depth, mean_depth, area))

        ax = axes[0][i]
        ax.imshow(sub, extent=[x[0] - 0.5 * dx, cases.LX, 0.0, cases.LZ],
                  aspect="equal", origin="upper")
        ax.contour(x, z, plume, levels=[0.5], colors="cyan", linewidths=0.8)
        ax.plot(x, cases.LZ - front, color="lime", lw=1.0)
        ax.set_xlim(0.0, cases.LX)
        ax.set_ylim(0.0, cases.LZ)
        ax.set_title("lab, t = %d min" % minutes, fontsize=10)
        ax.set_xlabel("x [m]")
        if i == 0:
            ax.set_ylabel("z [m]")
        ax.text(0.98, 0.04, "%.3f m" % depth, transform=ax.transAxes,
                ha="right", fontsize=8, color="0.2")

    fig.suptitle("UHD lab photographs digitised -- cyan: dye mask, "
                 "green: penetration front")
    qc = out_dir / "fig8_uhd_lab_digitisation_check.png"
    fig.savefig(str(qc), dpi=150, bbox_inches="tight")

    with open(out_dir / "fig8_uhd_lab_digitised_summary.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["t_min", "max_depth_m", "mean_depth_source_m", "plume_area_m2"])
        print("%-6s %10s %10s %10s" % ("t(min)", "max_depth", "mean_depth", "area"))
        for minutes, depth, mean_depth, area in summary:
            w.writerow([minutes, "%.4f" % depth, "%.4f" % mean_depth, "%.5f" % area])
            print("%-6d %10.3f %10.3f %10.4f" % (minutes, depth, mean_depth, area))
    print("wrote %s and per-panel CSV/npz in %s" % (qc, out_dir))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("figure")
    ap.add_argument("--out", default="UMD/data")
    a = ap.parse_args()
    main(a.figure, a.out)
