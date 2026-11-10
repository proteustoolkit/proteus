"""Build Table tab:uhd_coupling of the paper from the run logs.

Run each directory's run.sh first; it writes simmons_so.log there. In each
log, the last [TADR-CONT2] line gives the solute created by the coupling,
integrated over the run and split into the flux mismatch ("flux") and the
storage jump ("history"); the last [TADR-MASS] line gives the final solute
mass sum(ML*M*c). Both are printed by TADR._reportContinuityDefect at every
postStep. Runs without a log are skipped.

Usage:  python coupling_table.py
"""
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))

# (row label in the paper, run directory)
RUNS = [
    ("Velocity-based, Co = 0.5", "velocity_co0.5"),
    ("Velocity-based, Co = 0.25", "velocity_co0.25"),
    ("Velocity-based, lagged density", "velocity_lagged_density"),
    ("  same, flow limiter off", "velocity_lagged_density_flowfct_off"),
    ("Compatible (reconstructed flux)", "compatible"),
]

CONT2 = re.compile(r"\[TADR-CONT2\].*INT: flux=(\S+) history=(\S+) total=(\S+)")
MASS = re.compile(r"\[TADR-MASS\].*sum\(ML\*M\*u\)=(\S+)")

print("%-34s %12s %12s %12s" % ("Coupling", "Flux (%)", "Storage (%)", "Total (%)"))
for label, run in RUNS:
    flux = hist = total = mass = None
    log = os.path.join(HERE, run, "simmons_so.log")
    if not os.path.exists(log):
        print("%-34s (no simmons_so.log; run %s/run.sh)" % (label, run))
        continue
    with open(log) as f:
        for line in f:
            m = CONT2.search(line)
            if m:
                flux, hist, total = (float(v) for v in m.groups())
            m = MASS.search(line)
            if m:
                mass = float(m.group(1))
    pct = lambda v: 100.0 * v / mass
    print("%-34s %12.3g %12.3g %12.3g" % (label, pct(flux), pct(hist), pct(total)))
