"""Boundedness and mass-conservation table for a set of runs.

    python bounds_mass_table.py LABEL:RUN_DIR:LO:HI [LABEL:RUN_DIR:LO:HI ...]

RUN_DIR holds one run (its *.h5 archive and *.log); [LO, HI] is the admissible
range of c for that case (the min and max of the initial and boundary data).
Prints one markdown row per run:

  nodal min/max   over every node of every archived output (u_t*)
  n out           node-outputs below LO or above HI by more than TOL
  step min/max    quadrature-point u range of every time step, from the
                  [Coupling rho q] log lines (global over ranks); catches
                  excursions between outputs, but P1 quadrature values are
                  averages of nodal ones, so it can only under-report
  spurious solute time-integrated solute created by the flow/transport
                  incompatibility, [TADR-CONT2] "INT total", over the final
                  solute mass sum ML*M*u ([TADR-MASS]); the transport itself is
                  in flux form and its FCT correction antisymmetric, so this is
                  the coupling's only conservation error up to the Newton
                  tolerance
  max carried-M   max over steps of max|M - m_flow|/max|m_flow|, [TADR-EDGE]:
                  how far TADR's carried fluid mass drifts from Richards' m
  dmass/mass      transport-only runs (no coupling log): lumped solute mass of
                  the last output against the first, sum_i ML_i u_i.  The
                  domain is open (outflow carries u out), so this is outflow
                  plus any conservation error, not a conservation error alone
"""
import re
import sys
from pathlib import Path

import h5py
import numpy as np

# Absolute tolerance for counting a node out of bounds: c is O(1) here, so
# 1e-12 is round-off level for the linear solves (direct LU).
TOL = 1.0e-12
NUM = r"([-+0-9.eE]+)"


def lumped(nodes, el):
    x, y = nodes[:, 0], nodes[:, 1]
    a, b, c = el.T
    area = 0.5 * np.abs((x[b] - x[a]) * (y[c] - y[a]) - (x[c] - x[a]) * (y[b] - y[a]))
    m = np.zeros(len(nodes))
    for k in (a, b, c):
        np.add.at(m, k, area / 3.0)
    return m


def nodal(h5name, lo, hi):
    with h5py.File(h5name, "r") as h5:
        ks = sorted((int(k[3:]) for k in h5 if k.startswith("u_t") and k[3:].isdigit()))
        umin, umax, nout = np.inf, -np.inf, 0
        for k in ks:
            u = h5[f"u_t{k}"][:].ravel()
            umin, umax = min(umin, u.min()), max(umax, u.max())
            nout += int((u < lo - TOL).sum() + (u > hi + TOL).sum())
        mass = []
        for k in (ks[0], ks[-1]):
            m = lumped(h5[f"nodesSpatial_Domain{k}"][:, :2], h5[f"elementsSpatial_Domain{k}"][:])
            mass.append(np.sum(m * h5[f"u_t{k}"][:].ravel()))
    return umin, umax, nout, len(ks), mass


def logged(logname):
    qlo, qhi, cont, solute, edge, nstep = np.inf, -np.inf, None, None, None, 0
    pq = re.compile(r"\[Coupling rho q\].*?u \(min,max,mean\)=\(%s,%s," % (NUM, NUM))
    # the log's "% of mass" has 4 decimals, so take the absolute total instead
    pc = re.compile(r"\[TADR-CONT2\].*INT:.* total=%s" % NUM)
    pm = re.compile(r"\[TADR-MASS\].*sum\(ML\*M\*u\)=%s" % NUM)
    pe = re.compile(r"\[TADR-EDGE\].*max\|M - m_flow\|/max\|m_flow\|=%s" % NUM)
    with open(logname, errors="replace") as f:
        for line in f:
            if (m := pq.search(line)):
                qlo, qhi, nstep = min(qlo, float(m[1])), max(qhi, float(m[2])), nstep + 1
            elif (m := pc.search(line)):
                cont = float(m[1])
            elif (m := pm.search(line)):
                solute = float(m[1])
            elif (m := pe.search(line)):
                edge = max(edge or 0.0, float(m[1]))
    if cont is not None:
        cont /= solute
    return qlo, qhi, nstep, cont, edge


print("| case | [lo, hi] | nodal min | nodal max | n out (outputs) | step min | step max "
      "| steps | spurious solute / mass | max carried-M drift | dmass/mass |")
print("|---|---|---|---|---|---|---|---|---|---|---|")
for spec in sys.argv[1:]:
    label, d, lo, hi = spec.rsplit(":", 3)
    lo, hi, d = float(lo), float(hi), Path(d)
    h5name = next(p for p in sorted(d.glob("*.h5")) if "u_t0" in h5py.File(p, "r"))
    umin, umax, nout, nk, mass = nodal(h5name, lo, hi)
    logs = [p for p in d.glob("*.log") if p.stat().st_size > 0]
    qlo, qhi, nstep, cont, edge = logged(max(logs, key=lambda p: p.stat().st_size))
    f = lambda v, fmt="%.3e": "-" if v is None or not np.isfinite(v) else fmt % v
    dm = (mass[1] - mass[0]) / mass[0] if cont is None else None
    print(f"| {label} | [{lo:g}, {hi:g}] | {umin:+.2e} | {umax:.5f} | {nout} / {nk} "
          f"| {f(qlo, '%+.2e')} | {f(qhi, '%.5f')} | {nstep or '-'} | {f(cont, '%+.1e')} "
          f"| {f(edge, '%.1e')} | {f(dm, '%+.1e')} |")
