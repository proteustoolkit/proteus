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
Round 4 (same measure; before is main 9fca0b95):
  rising_bubble_rans3p  MCorr3P (or RANS3PF, PresInc) failure 7.6e-1   after: 2.7e-13   the solver's unknowns (uList)
                        were rebuilt from the dofs; RDLS's are not its dofs at the start of a step
                        Pres failure                 1.1e-5    after: 2.7e-13   RANS3PF's ebqe_velocity_last, which
                        PresInc's postStep sets, was not in RANS3PF's state()
  damBreakLevelSet interface=1 (RANS2P + CLSVOF), CLSVOF failure 1.1e-3   after: 0.0   CLSVOF's interface locator
  SWFlow circular_damBreak, SW2DCV failing in a middle SSP stage: crashed (stage counter past the order)   after: 0.0
                        the integrator's stage counter, stage values, the transport's *_dof_old and dLow are its state()
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
    "fillingTank-clsvof": dict(DECK="fillingTank.py", ATOL=1e-9, FAILMODEL="clsvof", FAILAT=10, BIGAUGES="1,2,3,4"),
    "fillingTank-pressureIncrement": dict(DECK="fillingTank.py", ATOL=1e-9, FAILMODEL="pressureIncrement", FAILAT=10),
    "damBreakLevelSet-vof": dict(DECK="damBreakLevelSet.py", FAILMODEL="vof", FAILAT=5),
    "rising_bubble_rans3p-mcorr3p": dict(DIR="surface_tension/rising_bubble_rans3p", SO="risingBubble_so.py",
                                         OPTS="nd=2 USE_SUPG_NS=1 ARTIFICIAL_VISCOSITY_NS=1", ATOL=1e-10,
                                         MAXITS=200, FAILMODEL="ls_consrv_p0", FAILAT=4),
    "rising_bubble_rans3p-pressure": dict(DIR="surface_tension/rising_bubble_rans3p", SO="risingBubble_so.py",
                                          OPTS="nd=2 USE_SUPG_NS=1 ARTIFICIAL_VISCOSITY_NS=1", ATOL=1e-10,
                                          MAXITS=200, FAILMODEL="pressure0", FAILAT=4),
    # FAILAT=3 fails a first stage of the TwoStageNewton (NCLS3P, RDLS elliptic): main ran the second stage, which
    # reset the failure, and took the step
    "rising_bubble_rans3p-ncls3p-first-stage": dict(DIR="surface_tension/rising_bubble_rans3p", SO="risingBubble_so.py",
                                                    OPTS="nd=2 USE_SUPG_NS=1 ARTIFICIAL_VISCOSITY_NS=1", ATOL=1e-10,
                                                    MAXITS=200, FAILMODEL="ls_p", FAILAT=3),
    "rising_bubble_rans3p-rdls-first-stage": dict(DIR="surface_tension/rising_bubble_rans3p", SO="risingBubble_so.py",
                                                  OPTS="nd=2 USE_SUPG_NS=1 ARTIFICIAL_VISCOSITY_NS=1", ATOL=1e-10,
                                                  MAXITS=200, FAILMODEL="redist_p", FAILAT=3),
    "rising_bubble_rans3p-vof3p-first-stage": dict(DIR="surface_tension/rising_bubble_rans3p", SO="risingBubble_so.py",
                                                   OPTS="nd=2 USE_SUPG_NS=1 ARTIFICIAL_VISCOSITY_NS=1", ATOL=1e-10,
                                                   MAXITS=200, FAILMODEL="vof_p", FAILAT=3),
    "damBreakLevelSet-rans2p-clsvof": dict(DECK="damBreakLevelSet.py", OPTS="he=0.1 interface=1", FT=0.1, ATOL=1e-10,
                                           MAXITS=1000, FAILMODEL="clsvof", FAILAT=6),
    "circular_damBreak-sw2dcv": dict(DIR="SWFlow", SO="--SWEs circular_damBreak.py",
                                     OPTS="sw_model=0 final_time=0.1 dt_output=0.1", FAILMODEL="sw", FAILAT=6),
    "circular_damBreak-gn_sw2dcv": dict(DIR="SWFlow", SO="--SWEs circular_damBreak.py",
                                        OPTS="sw_model=1 final_time=0.1 dt_output=0.1", FAILMODEL="GN", FAILSTEP=4),
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
    # a boundary integral gauge writes one sample per accepted step, the replay's values (measured 2.4e-13)
    gauge = os.path.join(str(tmp_path / "forced"), "boundary_integrals.csv")
    if os.path.exists(gauge):
        rows = [line.split(",") for line in open(gauge).read().splitlines()[1:]]
        replayed = [line.split(",") for line in open(os.path.join(str(tmp_path / "replay"),
                                                                  "boundary_integrals.csv")).read().splitlines()[1:]]
        assert [r[:5] for r in rows] == [r[:5] for r in replayed]
        assert len(set(float(r[0]) for r in rows)) == len(numpy.atleast_1d(numpy.loadtxt(accepted)))
        values, values_replayed = (numpy.array([float(r[5]) for r in x]) for x in (rows, replayed))
        assert numpy.abs(values - values_replayed).max() <= 1e-10 * max(numpy.abs(values).max(), 1.0)


@pytest.mark.parametrize("case", ["fillingTank-clsvof", "damBreakLevelSet-vof"])
def test_without_the_restore_the_retry_differs(tmp_path, case):
    options = CASES[case]
    forced, accepted = _run(tmp_path / "forced", NORESTORE=1, **options)
    replay, _ = _run(tmp_path / "replay", REPLAY=accepted, **_replay_options(options))
    assert _worst(forced, replay) > 1e-5
