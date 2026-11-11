"""An exception that one rank raises and nothing catches ends the whole parun or pro run job, with a nonzero exit.

Before, the rank that raised printed its traceback and exited, and the others waited in their next collective until the
job's wall time ran out: on LONI QB (job 1173802, 4096 ranks) one rank's Archiver raised a ValueError in the first
archive write and the other ranks hung for 2.7 h. parun and pro now install proteus.runtime's
abort_on_uncaught_exception, a sys.excepthook that writes the traceback with the rank to stderr and the log and calls
MPI Abort(1). Measured (macOS arm64, MPICH, fillingTank at 2 ranks; step_retry_mpi.py
with RAISERANK=1, the first archive write raising on rank 1 only):

    before  no exit: rank 0 waits in a collective until the 300 s timeout kills it
    after   exit 1 in 1.3 s from launch, "parun: uncaught exception on rank 1 of 2; aborting the MPI job"

and pro run on test/ci's ladr_2d at 2 ranks (parun_exit_status/ladr_2d_raise_n.py, RAISERANK=1):

    before  no exit: rank 0 waits until the 300 s timeout kills it
    after   exit 1 in 0.7 s ("pro: uncaught exception on rank 1 of 2; aborting the MPI job")

A StepTimeError, which every rank raises together, is still caught and exits 1 without an abort, by parun
(test_step_time_check.py::test_a_nan_on_one_rank_stops_every_rank) and now by pro run too, which before let it
reach the top level on every rank.
"""
import glob
import os
import shutil
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
HELPER = os.path.join(HERE, "step_retry_mpi.py")
PARUN = os.path.join(HERE, os.pardir, "scripts", "parun")
DRIVERS = {
    # the source tree's parun when there is one: an installed parun is a copy made at install time
    "parun": [sys.executable, PARUN if os.path.isfile(PARUN) else shutil.which("parun")],
    "pro": [sys.executable, "-m", "proteus.cli", "run"],
}
# a run that aborts exits in seconds; one that hangs would wait for the CI job's timeout instead
TIMEOUT = 180


def _env(**overrides):
    env = dict(os.environ, OMP_NUM_THREADS="1", PETSC_OPTIONS="-mat_partitioning_type current")
    env.setdefault("FI_PROVIDER", "tcp")
    env["PYTHONPATH"] = os.pathsep.join(
        [p for p in sys.path if p] + [env.get("PYTHONPATH", "")]).strip(os.pathsep)
    env.update({k: str(v) for k, v in overrides.items()})
    return env


def _mpiexec():
    beside = os.path.join(os.path.dirname(sys.executable), "mpiexec")
    return beside if os.path.exists(beside) else shutil.which("mpiexec")


def _run(cwd, ranks=None, **overrides):
    cmd = [sys.executable, HELPER] if ranks is None else [_mpiexec(), "-n", str(ranks), sys.executable, HELPER]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT, env=_env(**overrides),
                             cwd=str(cwd))
    except subprocess.TimeoutExpired:
        pytest.fail("parun did not exit within %d s: the ranks that did not raise are hung" % TIMEOUT)
    log = "".join(open(f).read() for f in sorted(glob.glob(os.path.join(str(cwd), "*.log"))))
    return out, log


@pytest.mark.skipif(_mpiexec() is None, reason="needs mpiexec")
@pytest.mark.parametrize("rank", [1, 0])
def test_an_exception_on_one_rank_aborts_every_rank(tmp_path, rank):
    out, log = _run(tmp_path, ranks=2, RAISERANK=rank)
    report = out.stdout[-3000:] + out.stderr[-3000:] + log[-3000:]
    assert out.returncode != 0, report
    assert "parun: uncaught exception on rank %d of 2; aborting the MPI job" % rank in out.stderr, report
    assert "ValueError: RAISERANK: rank %d fails its first archive write" % rank in out.stderr, report
    if rank == 0:
        # only rank 0 writes the log (without -a): the traceback is there too
        assert "uncaught exception on rank 0 of 2" in log, log[-3000:]


def test_an_exception_in_a_serial_run_exits_one(tmp_path):
    out, log = _run(tmp_path, RAISERANK=0)
    report = out.stdout[-3000:] + out.stderr[-3000:] + log[-3000:]
    assert out.returncode == 1, report
    assert "parun: uncaught exception on rank 0 of 1\n" in out.stderr, report
    assert "uncaught exception on rank 0 of 1" in log, log[-3000:]


def _run_driver(cwd, driver, ranks=None, **overrides):
    """Run test/ci's ladr_2d under ``driver`` with parun_exit_status/ladr_2d_raise_n.py."""
    for name in ("adr.py", "ladr_2d_p.py", "ladr_2d_n.py"):
        shutil.copy(os.path.join(HERE, "ci", name), str(cwd))
    shutil.copy(os.path.join(HERE, "parun_exit_status", "ladr_2d_raise_n.py"), str(cwd))
    cmd = DRIVERS[driver] + ["ladr_2d_p.py", "ladr_2d_raise_n.py"]
    if ranks is not None:
        cmd = [_mpiexec(), "-n", str(ranks)] + cmd
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT, env=_env(**overrides),
                              cwd=str(cwd))
    except subprocess.TimeoutExpired:
        pytest.fail("%s did not exit within %d s: the ranks that did not raise are hung" % (driver, TIMEOUT))


@pytest.mark.skipif(_mpiexec() is None, reason="needs mpiexec")
@pytest.mark.parametrize("driver", sorted(DRIVERS))
def test_each_driver_aborts_on_an_exception_on_one_rank(tmp_path, driver):
    out = _run_driver(tmp_path, driver, ranks=2, RAISERANK=1)
    report = out.stdout[-3000:] + out.stderr[-3000:]
    assert out.returncode != 0, report
    assert "%s: uncaught exception on rank 1 of 2; aborting the MPI job" % driver in out.stderr, report


@pytest.mark.skipif(_mpiexec() is None, reason="needs mpiexec")
@pytest.mark.parametrize("driver", sorted(DRIVERS))
def test_each_driver_exits_one_without_an_abort_on_a_step_time_error(tmp_path, driver):
    """Every rank raises the StepTimeError together: the driver catches it and every rank exits 1."""
    out = _run_driver(tmp_path, driver, ranks=2, RAISERANK="all")
    report = out.stdout[-3000:] + out.stderr[-3000:]
    assert out.returncode == 1, report
    assert "uncaught exception" not in out.stderr and "MPI_Abort" not in out.stderr, report


@pytest.mark.skipif(_mpiexec() is None, reason="needs mpiexec")
@pytest.mark.parametrize("driver", sorted(DRIVERS))
def test_each_driver_runs_the_deck_without_raising(tmp_path, driver):
    out = _run_driver(tmp_path, driver, ranks=2)
    assert out.returncode == 0, out.stdout[-3000:] + out.stderr[-3000:]
