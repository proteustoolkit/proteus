"""Collect the coupled-MMS results into convergence and sweep tables.

    python mms_table.py ladder <run dir> ...   errors and observed orders vs N
    python mms_table.py sweep  <run dir> ...   coupling error e_c vs eps
    python mms_table.py blocks c|psi <run dir> ...
                                               LaTeX table of E2 and its order
                                               vs h, low order | FCT, one block
                                               per time t = T/4, T/2, T
    python mms_table.py dt2 <run dir> ...      LaTeX table at t = T of the dt/2
                                               partner <run dir>_dt2 of each
                                               run: E2, its order vs h, and
                                               r = E2(dt)/E2(dt/2), c and psi

Each run dir is one run_mms.sh case.  ladder and sweep use the LAST [MMS-ERR]
line of its mms_so.log (t = T); the fluid budget is the last [TADR-CONT2] total.
They print a readable table and the LaTeX rows for tab:mms_t3 / tab:mms_sweep.
"""

import math
import re
import sys
from pathlib import Path

_ERR = re.compile(r"\[MMS-ERR\] t=(\S+) c: E1=(\S+) E2=(\S+) Einf=(\S+) "
                  r"psi: E1=(\S+) E2=(\S+) Einf=(\S+)")
_CFG = re.compile(r"\[MMS-CFG\] .*eps=(\S+) omega_max=(\S+) .* N=(\d+) dt=(\S+) "
                  r"nsteps=(\d+) FCT=(\d)")
_CONT = re.compile(r"\[TADR-CONT2\].*total=(\S+)")


def read_run(run_dir):
    log = (Path(run_dir) / "mms_so.log").read_text()
    err = _ERR.findall(log)
    cfg = _CFG.search(log)
    if not err or not cfg:
        raise RuntimeError("%s: no [MMS-ERR]/[MMS-CFG] line" % run_dir)
    cont = _CONT.findall(log)
    e = [float(v) for v in err[-1]]
    return {
        "dir": str(run_dir), "t": e[0],
        "E1c": e[1], "E2c": e[2], "Einfc": e[3],
        "E1p": e[4], "E2p": e[5], "Einfp": e[6],
        "eps": float(cfg.group(1)), "omega": float(cfg.group(2)),
        "N": int(cfg.group(3)), "dt": float(cfg.group(4)), "FCT": int(cfg.group(6)),
        "fluid": float(cont[-1]) if cont else float("nan"),
    }


def error_at(run_dir, t_target):
    """E2(c) and E2(psi) from the [MMS-ERR] line closest to t_target."""
    rows = [[float(v) for v in m] for m in
            _ERR.findall((Path(run_dir) / "mms_so.log").read_text())]
    row = min(rows, key=lambda r: abs(r[0] - t_target))
    return {"t": row[0], "c": row[2], "psi": row[5]}


def _sci(v):
    m, x = ("%.3e" % v).split("e")
    return r"$%s\times10^{%d}$" % (m, int(x))


def blocks(quantity, run_dirs, fractions=(0.25, 0.5, 1.0)):
    """Print the LaTeX table: E2 and order vs h_e, low order | FCT, per time."""
    runs = [read_run(d) for d in run_dirs]
    T = max(r["t"] for r in runs)
    Ns = sorted({r["N"] for r in runs})
    by = {(r["N"], r["FCT"]): r["dir"] for r in runs}
    name = {"c": "concentration $c$", "psi": r"pressure head $\psi$"}[quantity]
    unit = ""
    out = [r"\begin{table}[h]", r"\centering", r"\begin{tabular}{c|cc|cc}", r"\hline",
           r"\multirow{2}{*}{$h_e$ [m]}",
           r"& \multicolumn{2}{c|}{\textbf{Low Order}}",
           r"& \multicolumn{2}{c}{\textbf{FCT}} \\",
           r"\cline{2-5}",
           r"& $\|e\|_{L^2}$%s & Order" % unit,
           r"& $\|e\|_{L^2}$%s & Order \\" % unit,
           r"\hline", r"\hline"]
    for b, frac in enumerate(fractions):
        label = {0.25: "T/4", 0.5: "T/2", 1.0: "T"}.get(frac, "%gT" % frac)
        out += ["", r"\multicolumn{5}{c}{\textbf{$t = %s = %.1f$ s}} \\" % (label, frac * T),
                r"\hline"]
        prev = {}
        for N in Ns:
            cells = ["$%g$" % (1.0 / N)]
            for fct in (0, 1):
                d = by.get((N, fct))
                if d is None:
                    cells += ["--", "--"]
                    continue
                e = error_at(d, frac * T)[quantity]
                o = "--" if fct not in prev else "$%.3f$" % _order(prev[fct], e)
                cells += [_sci(e), o]
                prev[fct] = e
            out.append("%-12s & %s & %s\n           & %s & %s \\\\" % tuple(cells))
        out.append(r"\hline\hline" if b < len(fractions) - 1 else r"\hline")
    out += [r"\end{tabular}",
            r"\caption{Two-way coupling (T3), $\Delta t\propto h$ at $\mathrm{Co}=0.5$: "
            r"discrete $L^2$ error of the %s against the manufactured solution.}" % name,
            r"\label{tab:mms_t3_%s}" % quantity, r"\end{table}"]
    print("\n".join(out))


def dt2(run_dirs):
    """Print the LaTeX table at t = T: E2 of the dt/2 partner <run dir>_dt2 of
    each run, its order vs h_e, and r = E2(dt)/E2(dt/2) on the same mesh, low
    order | FCT, one block for c and one for psi.  With e = e_h + e_dt and e_dt
    proportional to dt, r = 1 when the spatial error dominates and r = 2 when
    the backward Euler error does."""
    full = {(r["N"], r["FCT"]): r for r in (read_run(d) for d in run_dirs)}
    half = {k: read_run(r["dir"] + "_dt2") for k, r in full.items()}
    Ns = sorted({N for N, _ in full})
    out = [r"\begin{table*}[t]", r"\centering", r"\begin{tabular}{c|ccc|ccc}", r"\hline",
           r"\multirow{2}{*}{$h_e$ [m]}",
           r"& \multicolumn{3}{c|}{\textbf{Low Order}}",
           r"& \multicolumn{3}{c}{\textbf{FCT}} \\",
           r"\cline{2-7}",
           r"& $\|e\|_{L^2}$ & Order & $r$",
           r"& $\|e\|_{L^2}$ & Order & $r$ \\",
           r"\hline", r"\hline"]
    parts = (("E2c", r"concentration $c$"), ("E2p", r"pressure head $\psi$"))
    for b, (key, name) in enumerate(parts):
        out += ["", r"\multicolumn{7}{c}{\textbf{%s}} \\" % name, r"\hline"]
        prev = {}
        for N in Ns:
            cells = ["$%g$" % (1.0 / N)]
            for fct in (0, 1):
                if (N, fct) not in full:
                    cells += ["--", "--", "--"]
                    continue
                e = half[(N, fct)][key]
                o = "--" if fct not in prev else "$%.3f$" % _order(prev[fct], e)
                cells += [_sci(e), o, "$%.3f$" % (full[(N, fct)][key] / e)]
                prev[fct] = e
            out.append("%-12s & %s & %s & %s\n           & %s & %s & %s \\\\" % tuple(cells))
        out.append(r"\hline\hline" if b < len(parts) - 1 else r"\hline")
    out += [r"\end{tabular}",
            r"\caption{Two-way coupling (T3) with $\Delta t/2$ ($\mathrm{Co}=0.25$), "
            r"$t=T$: discrete $L^2$ error against the manufactured solution, its "
            r"order, and the ratio $r=\|e\|_{L^2}(\Delta t)/\|e\|_{L^2}(\Delta t/2)$ "
            r"to the error at $\mathrm{Co}=0.5$ on the same mesh. $r=1$ when the "
            r"spatial error dominates, $r=2$ when the backward Euler error does.}",
            r"\label{tab:mms_dt2}", r"\end{table*}"]
    print("\n".join(out))


def _order(coarse, fine):
    return math.log2(coarse / fine) if coarse > 0 and fine > 0 else float("nan")


def ladder(runs):
    runs = sorted(runs, key=lambda r: r["N"])
    keys = ("E1c", "E2c", "Einfc", "E2p")
    print("%5s %10s " % ("N", "dt") + " ".join("%11s %5s" % (k, "r") for k in keys)
          + " %10s" % "fluid")
    rows = []
    for k, r in enumerate(runs):
        rates = [_order(runs[k - 1][q], r[q]) if k else float("nan") for q in keys]
        print("%5d %10.3e " % (r["N"], r["dt"])
              + " ".join("%11.4e %5.2f" % (r[q], rr) for q, rr in zip(keys, rates))
              + " %10.2e" % r["fluid"])
        cells = ["%d" % r["N"], "%.3g" % r["dt"]]
        for q, rr in zip(keys, rates):
            cells += ["%.2e" % r[q], "--" if math.isnan(rr) else "%.2f" % rr]
        cells.append("%.1e" % r["fluid"])
        rows.append("    & " + " & ".join(cells) + r" \\")
    print("\nLaTeX rows (tab:mms_t3):")
    print("\n".join(rows))


def sweep(runs):
    by_dt = {}
    for r in runs:
        by_dt.setdefault(round(r["dt"], 12), []).append(r)
    dts = sorted(by_dt, reverse=True)
    table = {}
    for dt in dts:
        ref = [r for r in by_dt[dt] if r["eps"] == 0.0 and r["omega"] == 0.0]
        if not ref:
            raise RuntimeError("dt=%g: no eps=0, omega_max=0 run (T2) to reference" % dt)
        for r in sorted(by_dt[dt], key=lambda r: (r["omega"], r["eps"])):
            e_c = r["E2c"] - ref[0]["E2c"]
            print("dt=%.4e eps=%-6g omega_max=%-6g E2(c)=%.6e e_c=%+.4e"
                  % (dt, r["eps"], r["omega"], r["E2c"], e_c))
            table.setdefault((r["omega"], r["eps"]), []).append((r["E2c"], e_c))
    print("\nLaTeX rows (tab:mms_sweep), columns dt = %s:"
          % ", ".join("%.3g" % d for d in dts))
    for (omega, eps), cols in sorted(table.items()):
        cells = ["%g" % eps] + ["%.3e & %+.2e" % col for col in cols]
        print("    " + " & ".join(cells) + r" \\")


if __name__ == "__main__":
    if len(sys.argv) < 3 or sys.argv[1] not in ("ladder", "sweep", "blocks", "dt2"):
        sys.exit(__doc__)
    if sys.argv[1] == "blocks":
        blocks(sys.argv[2], sys.argv[3:])
    elif sys.argv[1] == "dt2":
        dt2(sys.argv[2:])
    else:
        runs = [read_run(d) for d in sys.argv[2:]]
        (ladder if sys.argv[1] == "ladder" else sweep)(runs)
