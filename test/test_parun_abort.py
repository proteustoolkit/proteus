"""An exception that one rank raises and nothing catches ends the whole parun job, with a nonzero exit status.

Before, the rank that raised printed its traceback and exited, and the others waited in their next collective until the
job's wall time ran out: on LONI QB (job 1173802, 4096 ranks) one rank's Archiver raised a ValueError in the first
archive write and the other ranks hung for 2.7 h. parun's sys.excepthook now writes the traceback with the rank to
stderr and the log and calls MPI Abort(1). Measured (macOS arm64, MPICH, fillingTank at 2 ranks; step_retry_mpi.py
with RAISERANK=1, the first archive write raising on rank 1 only):

    before  no exit: rank 0 waits in a collective until the 300 s timeout kills it
    after   exit 1 in 1.3 s from launch, "parun: uncaught exception on rank 1 of 2; aborting the MPI job"

A StepTimeError, which every rank raises together, is still caught and exits 1 without an abort
(test_step_time_check.py::test_a_nan_on_one_rank_stops_every_rank).
"""
import glob
import os
import shutil
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
HELPER = os.path.join(HERE, "step_retry_mpi.py")
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
