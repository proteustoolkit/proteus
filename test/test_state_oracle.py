"""The state oracle (state_oracle.py) on decks: what a hot start leaves different from the run it continues.

Each deck is run three ways by state_oracle_run.sh through test/step_retry_mpi.py (full; stopped at T0; hot started at T0
in continue mode), the numerical state dumped at the restore (dump 0) and an output interval later (dump 1), and the
differences at the last dump classified by state_oracle_allowlist.txt. The tests:
  - an exact deck has no difference at the last dump except the allowlist's (counters, memory nobody initialised, the
    decided dof_last_last, owners nothing reads, values recomputed before they are read);
  - a deck with a known gap (a `gap` row) still differs in a dof, and the test fails when the gap closes, so the row is
    removed and the C14 note's owner list shrinks;
  - a restart (hotStartMode=restart, which rebuilds the history) is NOT exact on an exact deck: the oracle can fail.
"""
import os
import subprocess

import pytest

import state_oracle

pytest.importorskip("h5py")
here = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(here, "state_oracle_run.sh")

# name: (DECKENV for step_retry_mpi.py, output times: the hot start time, then the next output time)
DECKS = {
    "damBreakLevelSet": ('DECK=damBreakLevelSet.py OPTS="he=0.1" FT=0.1 DTOUT=0.02', "0.04,0.06"),
    "damBreakLevelSet_turbulence": ('DECK=damBreakLevelSet.py OPTS="he=0.1 turbulence=1" FT=0.1 DTOUT=0.02', "0.04,0.06"),
    # RANS2P's Smagorinsky closure (CLOSURE=1 in step_retry_mpi.py; the TwoPhaseFlow decks leave it at 0): the kernel lags the
    # eddy viscosity (C14 note 3.5 row g)
    "damBreakLevelSet_smagorinsky": ('DECK=damBreakLevelSet.py OPTS="he=0.1" FT=0.1 DTOUT=0.02 CLOSURE=1', "0.04,0.06"),
    "ladr_2d": ('DIR=ci SO="ladr_2d_p.py ladr_2d_n.py"', "0.1,0.125"),
    "ladr_2d_sglag": ('DIR=ci SO="ladr_2d_p.py ladr_2d_sglag_n.py"', "0.1,0.125"),
    # density-coupled Compatible TADR on Richards (C14 note 3.5 rows a-d): 41x16 stored mesh, 10 s output intervals
    "levee_small": ('DIR=salinity/levee_small SO="ls_CCS_so.py"', "20.0,30.0"),
    # the same deck with the transport on an edge-based scheme (rows a, b, d, and Richards' rho_nodal_old): 4 = explicit
    # entropy viscosity with FCT (SSP order 1), 5 = ImplicitEV
    "levee_small_ev4": ('DIR=salinity/levee_small SO="ls_CCS_ev_so.py" LEVEE_STAB=4', "20.0,30.0"),
    "levee_small_ev5": ('DIR=salinity/levee_small SO="ls_CCS_ev_so.py" LEVEE_STAB=5', "20.0,30.0"),
    # the flow model's flux correction (Richards FCT: the limited dofs differ from the Newton vector, C14 note 3.5 row f)
    "levee_small_fct": ('DIR=salinity/levee_small SO="ls_CCS_so.py" LEVEE_FLOW_FCT=semi-implicit', "20.0,30.0"),
    "levee_small_ev4_fct": ('DIR=salinity/levee_small SO="ls_CCS_ev_so.py" LEVEE_STAB=4 LEVEE_FLOW_FCT=semi-implicit', "20.0,30.0"),
    # (ImplicitEV + implicit flow FCT is bitwise on Linux CI and 4e-16 off on macOS arm64: last-bit platform noise, not gated;
    # run it with DECKENV='DIR=salinity/levee_small SO=ls_CCS_ev_so.py LEVEE_STAB=5 LEVEE_FLOW_FCT=implicit')
    "circular_damBreak_sw2dcv": ('DIR=SWFlow SO="--SWEs circular_damBreak.py" OPTS="sw_model=0 final_time=0.3 dt_output=0.1"',
                                 "0.1,0.2"),
}


def _run(tmp_path, deck, mode="continue"):
    deckenv, times = DECKS[deck]
    env = dict(os.environ, DECKENV=deckenv, HEAD="1")
    subprocess.run(["sh", SCRIPT, str(tmp_path), times, mode], env=env, check=True, capture_output=True, text=True,
                   timeout=1800)
    last = len(times.split(",")) - 1
    return (state_oracle.load(str(tmp_path / "full" / ("full_%d_p0.pkl" % last))),
            state_oracle.load(str(tmp_path / "hot" / ("hot_%d_p0.pkl" % last))))


def _is_dof(path):
    return ".u[" in path and ".dof" in path and not path.endswith("dof_last_last")


@pytest.mark.parametrize("deck", sorted(DECKS))
def test_a_hot_start_leaves_only_the_accepted_differences(tmp_path, deck):
    true, hot = _run(tmp_path, deck)
    differences = state_oracle.diff(true, hot)
    allowlist = state_oracle.read_allowlist()
    gaps = [r for r in allowlist if r[0] == deck and r[2] == "gap"]
    if gaps:
        assert any(_is_dof(p) for p in differences), \
            "%s: no dof differs after the hot start: the gap %s is closed; remove its row from " \
            "state_oracle_allowlist.txt" % (deck, [g[3] for g in gaps])
        return
    left, stale = state_oracle.unaccounted(deck, differences, allowlist)
    assert not left, "after a hot start of %s these values still differ and no allowlist row says why (an owner a hot " \
                     "start does not restore, or a value recomputed before it is read; say which in " \
                     "state_oracle_allowlist.txt):\n  %s" % (deck, "\n  ".join(
                         "%s %.2e" % (p, differences[p][1]) for p in left[:30]))
    assert not stale, "allowlist rows of %s that matched nothing: %s" % (deck, stale)


def test_a_restart_is_not_exact_and_the_oracle_sees_it(tmp_path):
    """The check that the oracle can fail: a restart rebuilds the history from the solution."""
    true, hot = _run(tmp_path, "damBreakLevelSet", mode="restart")
    differences = state_oracle.diff(true, hot)
    assert any(_is_dof(p) and rel > 1e-6 for p, (_, rel, _) in differences.items())
