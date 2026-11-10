"""Per-case verification tables for the comp_geoscience runs.

usage: python make_tables.py          (from this folder)

For every test problem writes <case>.tex and <case>.md here.  The run directories
are listed in RUNS below; change them to point at other runs.  Each number is
taken from the run's own log or archive:

  solute mass balance   [TADR-CONT2] INT total = time-integrated solute created by
                        the coupling (continuity-defect source + fluid-mass history
                        jumps), divided by the solute mass sum(ML M u) at the end
                        ([TADR-MASS]).  With the cancelled form this is the only way
                        the transport scheme can create or destroy solute.
  fluid mass balance    [TADR-EDGE] max over steps of |sum ML (M - m_flow)| /
                        sum ML m_flow: the fluid mass TADR carries from Richards'
                        edge fluxes against Richards' own theta(psi) rho.  Interior
                        residual: [RICHARDS-EDGE] max |B_interior| / dt (the Newton
                        residual at interior nodes), where the run logged it.
  bounds                u min / max over every archived frame.
  accuracy              Henry: semi-analytical solution (henry_error_*.txt);
                        Zalesak: L1 error after one revolution; MMS: [MMS-ERR].
All log quantities above are global MPI reductions.
"""
import glob
import re
import sys
from pathlib import Path

import h5py
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
MMS_RUNS = ROOT.parent / "mms_coupled" / "runs"

RUNS = {
    "zalesak": [("ref 4, low order", ROOT / "zalesak/run_clip/low"),
                ("ref 4, FCT", ROOT / "zalesak/run_clip/fct"),
                # serial: the 64-rank ref 5 run predates the MPI fix of TADR's limiter
                ("ref 5, low order", ROOT / "zalesak/run_stab5_ref5/low"),
                ("ref 5, FCT", ROOT / "zalesak/run_stab5_ref5/fct")],
    "henry": [("FCT", ROOT / "henry/run_clip/fct", "henry/run_clip/henry_error_fct.txt"),
              ("low order", ROOT / "henry/run_clip/lo", "henry/run_clip/henry_error_lo.txt")],
    "elder": [("FCT", ROOT / "elder/run_dK")],
    "bioswale": [("FCT", ROOT / "bioswale/run_beta0_weak")],
    "levee": [("density", ROOT / "levee_density/run_dK/density"),
              ("non-density", ROOT / "levee_density/run_dK/non_density")],
    "uhd": [("FCT", ROOT / "simmons_uhd/run_dK")],
}
BOUNDS = {"zalesak": (0.0, 1.0), "henry": (0.0, 1.0), "elder": (0.0, 1.0),
          "bioswale": (0.0, 1.0), "levee": (0.0, 0.7), "mms": (0.2, 0.8), "uhd": (0.0, 1.0)}
MMS_N = [10, 20, 40, 80, 160, 320]

F = r"([-+]?[\d.]+(?:e[-+]?\d+)?)"
PAT = {
    "cont2": re.compile(r"\[TADR-CONT2\].*?total=" + F),
    "mass": re.compile(r"\[TADR-MASS\].*?sum\(ML\*M\*u\)=" + F),
    "edge": re.compile(r"\[TADR-EDGE\].*?max\|M - m_flow\|/max\|m_flow\|=" + F
                       + r" sum ML\(M - m_flow\)/sum ML m_flow=" + F),
    "rich": re.compile(r"\[RICHARDS-EDGE\] t=" + F + r" dt=" + F + r".*?max\|B_interior\|=" + F),
    "err": re.compile(r"\[MMS-ERR\] t=" + F + r" c: E1=" + F + r" E2=" + F + r" Einf=" + F
                      + r" psi: E1=" + F + r" E2=" + F + r" Einf=" + F),
}


def scan_log(d):
    log = max(glob.glob(f"{d}/*.log"), key=lambda p: Path(p).stat().st_size)
    out = {k: [] for k in PAT}
    for line in open(log, errors="replace"):
        for k, p in PAT.items():
            m = p.search(line)
            if m:
                out[k].append([float(g) for g in m.groups()])
    r = {"steps": len(out["mass"])}
    if out["cont2"] and out["mass"]:
        r["solute"] = abs(out["cont2"][-1][0]) / abs(out["mass"][-1][0])
    if out["edge"]:
        r["drift"] = max(abs(v[1]) for v in out["edge"])
    if out["rich"]:
        r["resid"] = max(v[2] / v[1] for v in out["rich"])
    if out["mass"]:
        r["mass_change"] = out["mass"][-1][0] / out["mass"][0][0] - 1.0
    if out["err"]:
        r["err"] = out["err"][-1]
    return r


def scan_h5(d, final=False):
    """u min/max over every archived frame (final=True: the last frame only), frame count, L1 error."""
    lo, hi = np.inf, -np.inf
    with h5py.File(glob.glob(f"{d}/*.h5")[0], "r") as f:
        ks = sorted((int(k[3:]) for k in f if k.startswith("u_t") and k[3:].isdigit()))
        for k in (ks[-1:] if final else ks):
            u = f[f"u_t{k}"][:]
            lo, hi = min(lo, u.min()), max(hi, u.max())
        l1 = None
        if f"u_analytical_t{ks[-1]}" in f:
            x = f[f"nodesSpatial_Domain{ks[-1]}"][:, :2]
            el = f[f"elementsSpatial_Domain{ks[-1]}"][:]
            a, b, c = el.T
            area = 0.5 * np.abs((x[b, 0] - x[a, 0]) * (x[c, 1] - x[a, 1]) - (x[c, 0] - x[a, 0]) * (x[b, 1] - x[a, 1]))
            ml = np.zeros(len(x))
            for k in (a, b, c):
                np.add.at(ml, k, area / 3.0)
            l1 = float(np.sum(ml * np.abs(f[f"u_t{ks[-1]}"][:] - f[f"u_analytical_t{ks[-1]}"][:])))
    return lo, hi, len(ks), l1


def henry_err(path):
    t = (ROOT / path).read_text()
    return tuple(float(re.search(r"\|\|c_num - c_ana\|\|_" + k + r"\s*=\s*" + F, t).group(1)) for k in ("2", "inf"))


e = lambda v: "--" if v is None else f"{v:.1e}"
# bounds hold to round-off: 1e-10 of the bound range covers the solver's floating-point
# noise (smallest violation seen in these runs: -2.7e-12) and nothing physical
ok = lambda lo, hi, b: "yes" if lo >= b[0] - 1e-10 * (b[1] - b[0]) and hi <= b[1] + 1e-10 * (b[1] - b[0]) else "NO"


def write(case, title, head, rows, notes):
    tex = [r"\begin{table}[h]", r"\centering", r"\caption{%s}" % title,
           r"\begin{tabular}{%s}" % ("l" + "r" * (len(head) - 1)), r"\hline",
           " & ".join(head) + r" \\", r"\hline"]
    tex += [" & ".join(str(c) for c in r) + r" \\" for r in rows]
    tex += [r"\hline", r"\end{tabular}", r"\end{table}"]
    (HERE / f"{case}.tex").write_text("\n".join(tex).replace("%", r"\%").replace("_", r"\_") + "\n")
    md = [f"## {title}", "", "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    md += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    md += [""] + [f"- {n}" for n in notes]
    (HERE / f"{case}.md").write_text("\n".join(md) + "\n")
    print("\n".join(md) + "\n")


def coupled(case, title, extra=None):
    b = BOUNDS[case]
    head = ["run", "steps", "solute MB error", "fluid mass drift", "interior fluid residual",
            "u min", "u max", f"in [{b[0]:g}, {b[1]:g}]"]
    if extra:
        head += ["L2 error", "Linf error"]
    rows = []
    for spec in RUNS[case]:
        label, d = spec[0], spec[1]
        r = scan_log(d)
        lo, hi, n, _ = scan_h5(d)
        row = [label, r["steps"], e(r.get("solute")), e(r.get("drift")), e(r.get("resid")),
               f"{lo:.1e}", f"{hi:.6f}", ok(lo, hi, b)]
        if extra:
            row += [f"{v:.3e}" for v in henry_err(spec[2])]
        rows.append(row)
    return head, rows


notes_common = ["solute MB error: time-integrated solute created by the coupling / solute mass at T",
                "fluid mass drift: max over steps of |sum ML (M - m_flow)| / sum ML m_flow",
                "interior fluid residual: max |B_interior|/dt (Newton residual); -- = not logged in this run"]

# 1. Zalesak (transport only)
rows = []
for label, d in RUNS["zalesak"]:
    r = scan_log(d)
    lo, hi, n, l1 = scan_h5(d)
    loT, hiT, _, _ = scan_h5(d, final=True)
    rows.append([label, e(r.get("mass_change")), f"{loT:.1e}", f"{hiT:.6f}", ok(lo, hi, BOUNDS["zalesak"]), f"{l1:.4e}"])
write("zalesak", "Zalesak's disk after one revolution",
      ["run", "solute mass change", "u min at T", "u max at T", "in [0, 1], all frames", "L1 error"], rows,
      ["solute mass change: sum(ML u) at T / at t = 0 - 1, including what leaves through the weakly imposed u = 0 boundary",
       "L1 error against the exact solution (the initial condition)",
       "ref 4: current build, serial; ref 5: serial (the 64-rank ref 5 run predates the MPI fix of TADR's limiter)"])

# 2. MMS
def mms(name):
    d = MMS_RUNS / name
    return scan_log(d), scan_h5(d)

rows = []
for fct in (0, 1):
    for N in MMS_N:
        r, (lo, hi, n, _) = mms(f"a_T3_fct{fct}_N{N}")
        rows.append(["FCT" if fct else "low order", N, e(r.get("solute")), e(r.get("drift")),
                     f"{lo:.4f}", f"{hi:.4f}", ok(lo, hi, BOUNDS["mms"])])
r, (lo, hi, n, _) = mms("a_T1_fct1_N20")
rows.append(["T1 (c = c0), FCT", 20, e(r.get("solute")), e(r.get("drift")), f"{lo:.4f}", f"{hi:.4f}",
             "max|c-c0| = %.1e" % r["err"][3]])
write("mms_conservation", "Coupled manufactured solution (T3): conservation and bounds",
      ["scheme", "N", "solute MB error", "fluid mass drift", "c min", "c max", "in [0.2, 0.8]"], rows,
      notes_common[:2] + ["T1: constant concentration c0 = 0.5; the last column is max over the run of |c - c0|"])

rows = []
for scheme, fct, dt2 in (("low order", 0, ""), ("FCT", 1, ""), ("FCT, dt/2", 1, "_dt2")):
    prev = None
    for N in MMS_N:
        r, _ = mms(f"a_T3_fct{fct}_N{N}{dt2}")
        E2c, E2p = r["err"][2], r["err"][5]
        oc = f"{np.log2(prev[0] / E2c):.2f}" if prev else "--"
        op = f"{np.log2(prev[1] / E2p):.2f}" if prev else "--"
        rows.append([scheme, N, f"{E2c:.3e}", oc, f"{E2p:.3e}", op])
        prev = (E2c, E2p)
write("mms_accuracy", "Coupled manufactured solution (T3): L2 errors and observed orders at T",
      ["scheme", "N", "E2(c)", "order", "E2(psi)", "order"], rows,
      ["order = log2(E_N / E_2N); h halves between rows",
       "dt/2: the same runs with half the time step"])

# 3-6. coupled cases
h, rows = coupled("henry", "Henry", extra=True)
write("henry", "Henry problem: conservation, bounds and error against the semi-analytical solution", h, rows,
      notes_common + ["L2, Linf: nodal error against the semi-analytical solution (Zidane et al. 2012)"])
for case, title in (("elder", "Elder problem (20 years)"),
                    ("bioswale", "Bioswale, bio2d set-up (T = 0.5 d)"),
                    ("levee", "Levee, density and non-density"),
                    ("uhd", "Simmons UHD, level 4 (48 min)")):
    h, rows = coupled(case, title + ": conservation and bounds")
    write(case, title + ": conservation and bounds", h, rows, notes_common)

# 7. UHD penetration depth: depth below the surface (LZ = 1.06 m) of the deepest node
# with u >= 0.7 (cases.PENETRATION_THRESHOLD), at the times of the lab record
sys.path.insert(0, str(ROOT / "simmons_uhd"))
import cases
import penetration_height
d = RUNS["uhd"][0][1]
steps, z = penetration_height.penetration_series(glob.glob(f"{d}/*.h5")[0])
t_num = penetration_height.load_times(str(d / Path(glob.glob(f"{d}/*.h5")[0]).stem)) / 60.0
lab = np.loadtxt(ROOT / "simmons_uhd/data/z_lab_UHD.csv", delimiter=",", comments="#")
hgs = np.loadtxt(ROOT / "simmons_uhd/data/z_hgs_UHD.csv", delimiter=",", comments="#")
lab = lab[lab[:, 0] > 0]
# the lab record is read to 0.01 min off the 3-min output grid (digitize_fig7.py)
k = [int(np.argmin(np.abs(t_num - t))) for t in lab[:, 0]]
d_num = cases.LZ - z[k]
d_hgs = np.interp(lab[:, 0], hgs[:, 0], hgs[:, 1])
rows = [[f"{t:.0f}", f"{a:.3f}", f"{b:.3f}", f"{c:.3f}"] for t, a, b, c in zip(lab[:, 0], lab[:, 1], d_hgs, d_num)]
r_l, r_h = penetration_height.correlation(lab[:, 1], d_num), penetration_height.correlation(d_hgs, d_num)
rows.append(["r", "%.4f" % r_l, "%.4f" % r_h, ""])
rows.append(["eps_z", "%.4f" % penetration_height.eps_z(lab[:, 1], d_num),
             "%.4f" % penetration_height.eps_z(d_hgs, d_num), ""])
write("uhd_depth", "Simmons UHD: penetration depth of the u = 0.7 front",
      ["t (min)", "lab (m)", "HGS (m)", "this work (m)"], rows,
      ["depth below the surface of the deepest node with u >= 0.7",
       "lab: Simmons et al. (2002); HGS: 30-realisation mean of Guevara Morel & Graf (2023), Fig. 7",
       "r, eps_z: Eqs. (19)-(20) of Guevara Morel & Graf (2023), this work against lab / HGS"])
