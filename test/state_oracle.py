"""The state-difference oracle: which numerical state does a hot start leave different from the run it continues?

C14 design note (C14_DESIGN_NOTE_2026-11-2x.md, 3.6): the 84 "unmeasured" rows of state_contract_inventory.txt and the
owners outside it are predictions from reading until a hot start measures them, and each measurement by hand cost a
session. This is the measurement as a tool.

For a deck, run it three times with test/step_retry_mpi.py (STATEDUMP=<prefix> STATEDUMPT=<t>):
  full  the uninterrupted run, to a time past t; its state is dumped at the start of the output interval beginning at t
  stop  the run to t (its final frame carries the solver state)
  hot   `-H -t t` on the stop run's archive (continue, or HOTSTARTMODE=restart); dumped at the same instant, after the
        restore, before the first step
and diff the dumps. Every array and scalar the models hold (`snapshot.collect`: the dofs and their history, the quadrature
and face arrays, the coefficients, time integration, shock capturing, subgrid error, numerical flux, step controllers and
the system step controller; lists and dicts of arrays included) is compared by attribute path. A path that differs is
either state some owner has not restored, or a value that is derived and recomputed before it is read (it then matters
only if the first solves differ); the allowlist (state_oracle_allowlist.txt) says which, per deck, with the reason, so a
new `_old` array or a new owner that a hot start leaves different fails test_state_oracle.py the way a new class fails
test_state_contract.py.

    python state_oracle.py TRUE.pkl HOT.pkl [relative floor]
"""
import fnmatch
import os
import pickle
import sys

import numpy as np

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(here, "hotstart_measure"))


def dump(ns, path):
    """Pickle {attribute path: array or scalar} of the numerical state of ``ns`` (see snapshot.collect)."""
    import snapshot
    state = {key: value[0] for key, value in snapshot.collect(ns).items()}
    with open(path, "wb") as f:
        pickle.dump(state, f)
    return state


def load(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def diff(true, hot, floor=0.0):
    """{path: (max abs difference, relative to max(max|true|, 1)), note}: the values that differ by more than
    ``floor`` (relative), a shape change, and paths that exist in one state only (``note`` says which)."""
    out = {}
    for path in sorted(set(true) | set(hot)):
        if " at 0x" in path:  # a dict keyed by objects (SO_base.exitModelStep): the key is the object's address
            continue
        if path not in hot:
            out[path] = (np.inf, np.inf, "only in the uninterrupted run")
            continue
        if path not in true:
            out[path] = (np.inf, np.inf, "only in the hot start")
            continue
        x, y = np.asarray(true[path], dtype=float), np.asarray(hot[path], dtype=float)
        if x.shape != y.shape:
            out[path] = (np.inf, np.inf, "shape %s vs %s" % (x.shape, y.shape))
            continue
        if x.size == 0:
            continue
        with np.errstate(invalid="ignore"):
            d = np.where(np.isnan(x) & np.isnan(y) | (x == y), 0.0, np.abs(x - y))
        d = float(np.nanmax(d)) if d.size else 0.0
        scale = max(float(np.nanmax(np.abs(np.where(np.isnan(x), 0.0, x)))), 1.0)
        if d > floor * scale:
            out[path] = (d, d / scale, "")
    return out


def owner(path):
    """(model, owner group) of an attribute path: 'ls.timeIntegration.m_last[0]' -> ('ls', 'timeIntegration')."""
    if path.startswith("system."):
        return "system", "systemStepController"
    model, _, rest = path.partition(".")
    group = rest.split(".")[0].split("[")[0]
    return model, group


def read_allowlist(path=None):
    """[(deck, fnmatch pattern over attribute paths, owner row, reason)]: differences accepted for a deck, each tied to
    the inventory row or section-3.5 owner it belongs to. The deck field is an fnmatch pattern over deck names (``*`` matches every deck, ``damBreakLevelSet*`` the family)."""
    path = path or os.path.join(here, "state_oracle_allowlist.txt")
    rows = []
    if os.path.exists(path):
        for line in open(path):
            line = line.strip()
            if line and not line.startswith("#"):
                deck, pattern, row, reason = [f.strip() for f in line.split("|", 3)]
                rows.append((deck, pattern, row, reason))
    return rows


def unaccounted(deck, differences, allowlist):
    """(paths that differ and no allowlist row of this deck covers, allowlist rows of this deck that matched nothing; a
    ``noise`` row may match nothing: uninitialised memory differs from run to run)."""
    mine = [r for r in allowlist if fnmatch.fnmatchcase(deck, r[0])]
    hit = set()
    left = []
    for path in differences:
        rows = [i for i, r in enumerate(mine) if fnmatch.fnmatchcase(path, r[1])]
        hit.update(rows)
        if not rows:
            left.append(path)
    stale = [r for i, r in enumerate(mine) if i not in hit and r[0] != "*" and r[2] not in ("noise", "decided")]
    return left, stale


def report(differences, true, stream=sys.stdout):
    groups = {}
    for path, (d, rel, note) in differences.items():
        g = groups.setdefault(owner(path), [0, 0.0, path])
        g[0] += 1
        if rel >= g[1]:
            g[1], g[2] = rel, path
    stream.write("%d of %d values differ\n" % (len(differences), len(true)))
    for (model, group), (n, rel, worst) in sorted(groups.items(), key=lambda kv: -kv[1][1]):
        stream.write("  %-14s %-22s %4d paths, worst %.2e at %s\n" % (model, group, n, rel, worst))


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--deck=")]
    deck = ([a[7:] for a in sys.argv[1:] if a.startswith("--deck=")] or [None])[0]
    a, b = load(args[0]), load(args[1])
    differences = diff(a, b, float(args[2]) if len(args) > 2 else 0.0)
    if deck:
        # --deck=NAME: classify with the allowlist rows of NAME (or `*`) and print only what no row covers
        left, stale = unaccounted(deck, differences, read_allowlist())
        differences = {p: differences[p] for p in left}
        print("not covered by the allowlist rows of %s: %d (stale rows: %d)" % (deck, len(left), len(stale)))
    report(differences, a)
    for path, (d, rel, note) in sorted(differences.items(), key=lambda kv: -kv[1][1])[:int(os.environ.get("LIST", "60"))]:
        print("    %-90s %.3e %s" % (path[:90], rel, note))
