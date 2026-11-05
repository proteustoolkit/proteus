"""Hot start of the level-set family (test/TwoPhaseFlow damBreakLevelSet: RANS2P + VOF + NCLS + RDLS + MCorr,
he=0.1) continues the uninterrupted run bit for bit.

What the archive's solver state now carries for it (C17/C14 round 1; largest relative difference over every
model's dofs at the end, hot start vs the uninterrupted run, with that piece left out; macOS arm64, serial):
  the fields other models read from a model at the quadrature points (NumericalSolution's coupledFields
  owner): the values of its last residual, not of its dofs, since MCorr and RDLS change VOF's and NCLS's
  dofs after their solves                                                            6.6e-2
  BackwardEuler's m_last, the mass the next step differences against (VOF's is its uncorrected mass)
  and BackwardEuler_cfl's dtLast                                                      5.0e-1 (MCorr)
  RANS2P's lagged shock capturing and subgrid error (DelayedLagState, v_last)         3.2e-3, 6.7e-4
  the solver's unknowns, the next Newton's initial guess (unknowns owner)            5.4e-14
  the next system step (systemStep owner): restarted at 0.005 the first step went
  to 0.01 instead of 0.0075                                                           5.4e-1
Before all of it: 5.1e-1 (velocity 7.3e-2, VOF 1.4e-2, forces ~1%). After: 0.0, forces 0.0.
"""
import os
import subprocess
import sys

import numpy
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
HELPER = os.path.join(HERE, "step_retry_mpi.py")


def _run(cwd, **overrides):
    os.makedirs(str(cwd), exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS="1", DECK="damBreakLevelSet.py")
    env.setdefault("FI_PROVIDER", "tcp")
    env["PYTHONPATH"] = os.pathsep.join(
        [p for p in sys.path if p] + [env.get("PYTHONPATH", "")]).strip(os.pathsep)
    env.update({k: str(v) for k, v in overrides.items()})
    out = subprocess.run([sys.executable, HELPER], capture_output=True, text=True, timeout=1800, env=env, cwd=str(cwd))
    assert out.returncode == 0, out.stdout[-2000:] + out.stderr[-2000:]
    return numpy.loadtxt(os.path.join(str(cwd), "u.csv"), delimiter=",", skiprows=1)


def _worst(a, b):
    assert a.shape == b.shape and numpy.abs(a[:, :4] - b[:, :4]).max() < 1e-10
    return max(numpy.abs(a[a[:, 0] == k, 4] - b[b[:, 0] == k, 4]).max() / (numpy.abs(a[a[:, 0] == k, 4]).max() + 1e-300)
               for k in numpy.unique(a[:, 0]))


# (archive interval, hot start time, final time): 0.04 is past the switch to lagged shock capturing and
# subgrid error; at 0.005 the step after the restart is capped by the history of the steps before it
CASES = {"lagged": (0.02, 0.04, 0.1), "early": (0.005, 0.005, 0.03)}


@pytest.mark.parametrize("case", sorted(CASES))
def test_hot_start_continues_the_uninterrupted_run(tmp_path, case):
    dtout, t, ft = CASES[case]
    full = _run(tmp_path / "full", FT=ft, DTOUT=dtout)
    _run(tmp_path / "hot", FT=t, DTOUT=dtout)
    hot = _run(tmp_path / "hot", FT=ft, DTOUT=dtout, HOTSTART=t)
    assert _worst(full, hot) == 0.0
    # every accepted step after the restart is the uninterrupted run's, and so is every force row
    after = numpy.loadtxt(str(tmp_path / "full" / "accepted.txt"))
    numpy.testing.assert_array_equal(numpy.loadtxt(str(tmp_path / "hot" / "accepted.txt")), after[after > t + 1e-12])
    numpy.testing.assert_array_equal(numpy.loadtxt(str(tmp_path / "hot" / "forceHistory_p.txt")),
                                     numpy.loadtxt(str(tmp_path / "full" / "forceHistory_p.txt")))


def test_a_restart_that_rebuilds_the_history_differs(tmp_path):
    """The check that the test above can fail: --hotStartMode restart rebuilds the history from the restart
    state, as every hot start did before the solver state carried it."""
    dtout, t, ft = CASES["lagged"]
    full = _run(tmp_path / "full", FT=ft, DTOUT=dtout)
    _run(tmp_path / "hot", FT=t, DTOUT=dtout)
    hot = _run(tmp_path / "hot", FT=ft, DTOUT=dtout, HOTSTART=t, HOTSTARTMODE="restart")
    assert _worst(full, hot) > 1e-3
