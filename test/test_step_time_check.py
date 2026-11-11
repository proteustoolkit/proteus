"""A non-finite time, or a system time that does not move forward, stops the run on every rank with exit status 1.

Before, the stepping loop went on: ``while t_system_last < tn`` is False for a nan time, so a run whose system time
went nan left the loop as if it had reached the final time, and parun exited 0. NS_base._checkStepTimes now checks
the system controller's t_system_last, t_system and dt_system before every system step and after a retry cuts it,
and each model's t_model_last, t_model, dt_model and substeps before its steps; the ranks gather what they found and
all raise StepTimeError, which parun reports before exiting 1. Measured (macOS arm64; step_retry_mpi.py):

  fillingTank, the 3rd system step the controller chooses made nan (NANDT=3), final time 0.05:
      before  exit 0 under each controller, the last accepted time nan (Sequential_MinAdaptiveModelStep 0.001,
              0.003, 0.00352, nan; Sequential_MinModelStep 0.001, 0.002, 0.00279, nan; _SS 0.001, 0.003, 0.0052, nan)
      after   exit 1 at system step 4: "t_system=nan; dt_system=nan", every model's dt_model nan
  the same on 2 ranks with the nan on rank 1 only:
      before  hangs (rank 1 steps to nan, rank 0 goes on into the next collective)
      after   exit 1 on both ranks, the message naming rank 1
  rising_bubble_rans3p under Sequential_MinModelStep, pressure's 10th solve reported failed, the retry not putting back
  each model's t_model_last (NOTMODELRESTORE, the c17-r1 state before 83ac62c9):
      before  exit 0 after 20 min: 3541 retries over 1781 accepted steps, the system time walking back from
              0.0095 (0.00925, 0.008875, ...) to -1.6e308, -inf and nan
      after   exit 1 in 2 s, at the 2nd retry of system step 11: "the system step does not move forward:
              t_system=0.00925, t_system_last=0.0095, dt_system=-0.00025"
"""
import glob
import os
import shutil
import subprocess
import sys

import numpy
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
HELPER = os.path.join(HERE, "step_retry_mpi.py")
TIMEOUT = 900


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
    """Run step_retry_mpi.py in ``cwd``; the exit status, the accepted times and the run's log."""
    os.makedirs(str(cwd), exist_ok=True)
    cmd = [sys.executable, HELPER] if ranks is None else [_mpiexec(), "-n", str(ranks), sys.executable, HELPER]
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT, env=_env(**overrides), cwd=str(cwd))
    log = "".join(open(f).read() for f in sorted(glob.glob(os.path.join(str(cwd), "*.log"))))
    accepted = numpy.atleast_1d(numpy.loadtxt(os.path.join(str(cwd), "accepted.txt")))
    return out, accepted, log


def _stopped(out, accepted, log):
    assert out.returncode == 1, out.stdout[-2000:] + out.stderr[-2000:] + log[-2000:]
    assert "Stopping: a time step went non-finite or backwards" in log, log[-3000:]
    assert "run number 0 failed" in log, log[-3000:]
    assert numpy.all(numpy.isfinite(accepted)) and numpy.all(numpy.diff(accepted) > 0), accepted
    return [line for line in log.splitlines() if "Stopping:" in line][0]


CONTROLLERS = ["Sequential_MinAdaptiveModelStep", "Sequential_MinModelStep", "Sequential_MinAdaptiveModelStep_SS"]


@pytest.mark.parametrize("controller", CONTROLLERS)
def test_a_nan_system_step_stops_the_run(tmp_path, controller):
    out, accepted, log = _run(tmp_path, NANDT=3, SYSTEMCONTROLLER=controller)
    message = _stopped(out, accepted, log)
    assert "system step 4" in message and "dt_system=nan" in message, message
    assert "clsvof t_model_last=" in message, message
    assert len(accepted) == 3


def test_without_the_check_a_nan_system_step_ends_the_run_with_exit_zero(tmp_path):
    """The old behaviour, the check turned off: the run 'finishes' at a nan time and exits 0."""
    out, accepted, log = _run(tmp_path, NANDT=3, NOTIMECHECK=1)
    assert out.returncode == 0, out.stdout[-2000:] + out.stderr[-2000:]
    assert len(accepted) == 4 and numpy.isnan(accepted[-1]) and numpy.all(numpy.isfinite(accepted[:-1])), accepted


@pytest.mark.skipif(_mpiexec() is None, reason="needs mpiexec")
def test_a_nan_on_one_rank_stops_every_rank(tmp_path):
    out, accepted, log = _run(tmp_path, ranks=2, NANDT=3, NANDTRANK=1)
    message = _stopped(out, accepted, log)
    assert "rank 1: system step 4" in message and "rank 0:" not in message, message
    # every rank raised the StepTimeError together and parun caught it: no MPI abort (test_parun_abort.py)
    assert "uncaught exception" not in out.stderr and "MPI_Abort" not in out.stderr, out.stderr[-2000:]


def test_a_retry_that_walks_the_time_backwards_stops_the_run(tmp_path):
    out, accepted, log = _run(tmp_path, DIR="surface_tension/rising_bubble_rans3p", SO="risingBubble_so.py",
                              OPTS="nd=2 USE_SUPG_NS=1 ARTIFICIAL_VISCOSITY_NS=1", ATOL=1e-10, MAXITS=200,
                              FAILMODEL="pressure0", FAILAT=10, SYSTEMCONTROLLER="Sequential_MinModelStep",
                              NOTMODELRESTORE=1)
    message = _stopped(out, accepted, log)
    assert "the system step does not move forward" in message and "(retry" in message, message
    assert accepted[-1] < 0.0101
