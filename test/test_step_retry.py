"""A retried system step gives the answer of a run that took the same steps without failing.

When a model's solve fails, the system step is cut and retried. The retry used to start from the failed attempt:
the models' dofs were the failed iterate (CLSVOF builds its normals "at t_n" from that initial guess), the dof
history was shifted a second time, and a model that had finished in the failed attempt kept the lags its postStep had
moved forward (CLSVOF's previous solution and velocity, RANS3PF's boundary velocity). NumericalSolution now restores
every model to the start of the system step (dofs and the state() of its lagged objects) before the retry.

One forced failure of the named model; largest relative difference, over every model, against a failure-free run
that lands on the same accepted times (measured, macOS arm64; velocity before):
  fillingTank         CLSVOF 5.7e-3, RANS3PF 1.3e-1, PresInc 1.3e-1, Pres 4.5e-1   after: 5.6e-13
  damBreakLevelSet    VOF, NCLS, RDLS or MCorr 6.1e-2 (RANS2P 2.6e-15)             after: 3.8e-13
The retry also read the failed attempt's fields at the quadrature points (RANS2P reads VOF's and NCLS's), and RANS2P
and RANS3PF wrote a history row for the failed attempt; both are covered here.
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
    assert a.shape == b.shape and numpy.abs(a[:, :4] - b[:, :4]).max() < 1e-10
    worst = 0.0
    for key in numpy.unique(a[:, 0]):
        k = a[:, 0] == key
        worst = max(worst, numpy.abs(a[k, 4] - b[k, 4]).max() / (numpy.abs(a[k, 4]).max() + 1e-300))
    return worst


CASES = {
    "fillingTank-clsvof": dict(DECK="fillingTank.py", ATOL=1e-9, FAILMODEL="clsvof", FAILAT=10),
    "fillingTank-pressureIncrement": dict(DECK="fillingTank.py", ATOL=1e-9, FAILMODEL="pressureIncrement", FAILAT=10),
    "damBreakLevelSet-vof": dict(DECK="damBreakLevelSet.py", FAILMODEL="vof", FAILAT=5),
}


def _replay_options(options):
    """The case's options without the forced failure."""
    return {k: v for k, v in options.items() if k not in ("FAILMODEL", "FAILAT", "FAILSTEP", "NORESTORE")}


@pytest.mark.parametrize("case", sorted(CASES))
def test_a_retried_step_is_the_step_taken_without_failing(tmp_path, case):
    options = CASES[case]
    forced, accepted = _run(tmp_path / "forced", **options)
    replay, _ = _run(tmp_path / "replay", REPLAY=accepted, **_replay_options(options))
    assert int(open(os.path.join(str(tmp_path / "forced"), "retries.txt")).read()) >= 1, "the failure was not retried"
    assert _worst(forced, replay) < 1e-10
    # the failed attempt leaves no row in the histories RANS2P/RANS3PF write
    history = os.path.join(str(tmp_path / "forced"), "timeHistory.txt")
    if os.path.exists(history):
        times = numpy.loadtxt(history)
        numpy.testing.assert_array_equal(times, numpy.loadtxt(accepted))
        forces = numpy.loadtxt(os.path.join(str(tmp_path / "forced"), "forceHistory_p.txt"))
        assert forces.shape[0] % len(times) == 0


@pytest.mark.parametrize("case", ["fillingTank-clsvof", "damBreakLevelSet-vof"])
def test_without_the_restore_the_retry_differs(tmp_path, case):
    options = CASES[case]
    forced, accepted = _run(tmp_path / "forced", NORESTORE=1, **options)
    replay, _ = _run(tmp_path / "replay", REPLAY=accepted, **_replay_options(options))
    assert _worst(forced, replay) > 1e-5
