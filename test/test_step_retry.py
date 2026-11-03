"""A retried system step gives the answer of a run that took the same steps without failing.

When a model's solve fails, the system step is cut and retried. The retry used to start from the failed attempt:
the models' dofs were the failed iterate (CLSVOF builds its normals "at t_n" from that initial guess), the dof
history was shifted a second time, and a model that had finished in the failed attempt kept the lags its postStep had
moved forward (CLSVOF's previous solution and velocity, RANS3PF's boundary velocity). NumericalSolution now restores
every model to the start of the system step (dofs and the state() of its lagged objects) before the retry.

fillingTank, t=0.05, one forced failure at the 10th solve of the named model; largest relative difference against a
failure-free run that lands on the same accepted times (measured, macOS arm64):
                      before (velocity)   after
  CLSVOF fails        4.7e-3              1.6e-13
  PresInc fails       (see the PR)        1.6e-13
(The 16% first reported was mostly the different step sequence: the same steps without a failure differ from the
default run by 15-17% in velocity at t=0.05; fillingTank is that sensitive to dt.)
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
    env = dict(os.environ, OMP_NUM_THREADS="1")
    env.setdefault("FI_PROVIDER", "tcp")
    env["PYTHONPATH"] = os.pathsep.join(
        [p for p in sys.path if p] + [env.get("PYTHONPATH", "")]).strip(os.pathsep)
    env.update({k: str(v) for k, v in overrides.items()})
    out = subprocess.run([sys.executable, HELPER], capture_output=True, text=True, timeout=1800, env=env, cwd=str(cwd))
    assert out.returncode == 0, out.stdout[-2000:] + out.stderr[-2000:]
    return (numpy.loadtxt(os.path.join(str(cwd), "u.csv"), delimiter=",", skiprows=1),
            os.path.join(str(cwd), "accepted.txt"))


def _worst(a, b):
    assert a.shape == b.shape and numpy.abs(a[:, :3] - b[:, :3]).max() < 1e-10
    worst = 0.0
    for key in numpy.unique(a[:, 0]):
        k = a[:, 0] == key
        worst = max(worst, numpy.abs(a[k, 3] - b[k, 3]).max() / (numpy.abs(a[k, 3]).max() + 1e-300))
    return worst


@pytest.mark.parametrize("model", ["clsvof", "pressureIncrement"])
def test_a_retried_step_is_the_step_taken_without_failing(tmp_path, model):
    forced, accepted = _run(tmp_path / "forced", FAILMODEL=model, FAILAT=10)
    replay, _ = _run(tmp_path / "replay", REPLAY=accepted)
    assert _worst(forced, replay) < 1e-10


def test_without_the_restore_the_retry_differs(tmp_path):
    forced, accepted = _run(tmp_path / "forced", FAILMODEL="clsvof", FAILAT=10, NORESTORE=1)
    replay, _ = _run(tmp_path / "replay", REPLAY=accepted)
    assert _worst(forced, replay) > 1e-5
