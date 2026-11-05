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
import shutil
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


def test_a_continue_on_another_partition_warns_once(tmp_path):
    """A continue hot start whose archived solver state another partition wrote (the global mesh path with an
    unpinned graph partitioner) goes on as a restart. It said so per model in the log only; now one
    RuntimeWarning on the first rank names the models and the reason."""
    dtout, t, ft = CASES["lagged"]
    _run(tmp_path / "hot", FT=t, DTOUT=dtout)
    env = dict(os.environ, OMP_NUM_THREADS="1", DECK="damBreakLevelSet.py", FT=str(ft), DTOUT=str(dtout),
               HOTSTART=str(t), OTHERPARTITION="1")
    env.setdefault("FI_PROVIDER", "tcp")
    env["PYTHONPATH"] = os.pathsep.join([p for p in sys.path if p] + [env.get("PYTHONPATH", "")]).strip(os.pathsep)
    out = subprocess.run([sys.executable, HELPER], capture_output=True, text=True, timeout=1800, env=env,
                         cwd=str(tmp_path / "hot"))
    assert out.returncode == 0, out.stderr[-2000:]
    warned = [line for line in out.stderr.splitlines() if "RuntimeWarning: hot start (continue)" in line]
    assert len(warned) == 1, out.stderr[-3000:]
    assert "different partition" in warned[0] and all(m in warned[0] for m in ("rans2p", "vof", "mcorr"))


def _mpiexec():
    beside = os.path.join(os.path.dirname(sys.executable), "mpiexec")
    return beside if os.path.exists(beside) else shutil.which("mpiexec")


def _run_ranks(cwd, ranks, **overrides):
    os.makedirs(str(cwd), exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS="1", DECK="damBreakLevelSet.py",
               PETSC_OPTIONS="-mat_partitioning_type current")
    env.setdefault("FI_PROVIDER", "tcp")
    env["PYTHONPATH"] = os.pathsep.join(
        [p for p in sys.path if p] + [env.get("PYTHONPATH", "")]).strip(os.pathsep)
    env.update({k: str(v) for k, v in overrides.items()})
    out = subprocess.run([_mpiexec(), "-n", str(ranks), sys.executable, HELPER], capture_output=True, text=True,
                         timeout=1800, env=env, cwd=str(cwd))
    assert out.returncode == 0, out.stdout[-2000:] + out.stderr[-2000:]
    return out


@pytest.mark.skipif(_mpiexec() is None, reason="needs mpiexec")
def test_a_hot_start_on_another_number_of_ranks_is_a_restart(tmp_path):
    """Decided by cekees 2026-11-05: a hot start on another partition is legitimate; with the solution archived
    through MPI-IO and a global mesh it restarts from the archived step's solution, without the solver state.
    Archived on 2 ranks, hot-started (continue) on 3: the same as --hotStartMode restart on 3 ranks (and one warning
    saying so), to the run-to-run noise of 3 ranks (0.0 and 9.9e-14 in two runs; C7.1 saw 2.1e-12 on vortex2D). Measured against a serial restart from the same step: 8.2e-9 (rank count), where
    the uninterrupted run differs from a restart by 3.6e-1, so it restarts from 0.04, not from the initial state."""
    dtout, t, ft = CASES["lagged"]
    _run_ranks(tmp_path / "stop", 2, FT=t, DTOUT=dtout)
    shutil.copytree(str(tmp_path / "stop"), str(tmp_path / "continue"))
    shutil.copytree(str(tmp_path / "stop"), str(tmp_path / "restart"))
    out = _run_ranks(tmp_path / "continue", 3, FT=ft, DTOUT=dtout, HOTSTART=t)
    _run_ranks(tmp_path / "restart", 3, FT=ft, DTOUT=dtout, HOTSTART=t, HOTSTARTMODE="restart")
    assert sum("RuntimeWarning: hot start (continue)" in line for line in out.stderr.splitlines()) == 1
    numpy.testing.assert_array_equal(numpy.loadtxt(str(tmp_path / "continue" / "accepted.txt")),
                                     numpy.loadtxt(str(tmp_path / "restart" / "accepted.txt")))
    for rank in range(3):
        a, b = (numpy.loadtxt(str(tmp_path / d / ("u%d.csv" % rank)), delimiter=",", skiprows=1)
                for d in ("restart", "continue"))
        assert _worst(a, b) < 1e-10


def test_a_driver_without_hot_start_mode_continues(tmp_path):
    """Options without ``hotStartMode`` (a parun installed before the option existed: setup.py copies the script
    at install time, editable install or not) take the default, continue, instead of failing in the restore."""
    dtout, t, ft = CASES["lagged"]
    full = _run(tmp_path / "full", FT=ft, DTOUT=dtout)
    _run(tmp_path / "hot", FT=t, DTOUT=dtout, NOHOTSTARTMODE=1)
    hot = _run(tmp_path / "hot", FT=ft, DTOUT=dtout, HOTSTART=t, NOHOTSTARTMODE=1)
    assert _worst(full, hot) == 0.0


def test_the_hot_start_time_meets_an_archived_time_in_floating_point():
    """-H -t 0.06 with archives every 0.02: the archived step is 0.06000000000000001 (a sum of steps), and the
    "latest archived step at or before t" rule used to restart from 0.04. An archived time counts as at t
    within a millionth of the smallest interval between archived times (NumericalSolution.hot_start_step)."""
    from proteus.NumericalSolution import hot_start_step
    times = [0.0, 0.02, 0.04, 0.06000000000000001, 0.08]
    assert hot_start_step(times, 0.06) == 3
    assert hot_start_step(times, 0.059) == 2
    assert hot_start_step(times, 0.0600001) == 3
    assert hot_start_step(times, float("inf")) == 4
    assert hot_start_step(times, -1.0) == 0
    assert hot_start_step([0.0], 1.0) == 0


def test_hot_start_at_an_archived_time_given_in_decimal(tmp_path):
    """The same rule end to end: damBreakLevelSet archived every 0.02 to 0.1, hot-started again from -t 0.06 (the
    archive holds 0.06000000000000001). It continues from 0.06, a restart (only the last archived step has its
    solver state), and its first step ends at 0.08; the old rule restarted from 0.04 and stepped to 0.06."""
    _run(tmp_path / "hot", FT=0.1, DTOUT=0.02)
    archived = numpy.loadtxt(str(tmp_path / "hot" / "accepted.txt"))
    at = archived[numpy.argmin(numpy.abs(archived - 0.06))]
    assert at != 0.06 and abs(at - 0.06) < 1e-15  # the case: the archived step is not 0.06 in floating point
    _run(tmp_path / "hot", FT=0.1, DTOUT=0.02, HOTSTART=0.06)
    first = numpy.atleast_1d(numpy.loadtxt(str(tmp_path / "hot" / "accepted.txt")))[0]
    assert abs(first - 0.08) < 1e-12, "the hot start did not continue from 0.06 (first step ended at %r)" % first
