"""RANS2P_IB (RANS2P with immersed Euler-Bernoulli beams): the beam channel deck, serial against several ranks.

Until 2026-11-05 no deck ran RANS2P_IB, and it could not have: its Jacobian raised NameError (``elf.coefficients``),
its Coefficients passed RANS2P the default arguments instead of its own (any rho, nu, g, nd or LS_model was dropped:
every RANS2P_IB run was 3D single-phase water), and the beam loads were integrated over ghost elements too before the
sum over ranks, so on 2 ranks the drag on a rigid beam was 40% too large and a flexible beam bent 70% further.
"""
import os
import shutil
import subprocess
import sys

import pytest

from proteus.mprans import RANS2P_IB

HERE = os.path.dirname(os.path.abspath(__file__))
HELPER = os.path.join(HERE, "beam_channel_mpi.py")
MPIEXEC = shutil.which("mpiexec")


def test_coefficients_pass_their_arguments_to_rans2p():
    c = RANS2P_IB.Coefficients(nu_0=1.0e-3, rho_0=1000.0, nd=2, g=[0.0, -9.81, 0.0], LS_model=2, useMetrics=1.0,
                               beamLocation=[], beamLength=[], beamRadius=[], EI=[], GJ=[])
    assert (c.nu_0, c.rho_0, c.nd, list(c.g), c.LS_model, c.useMetrics) == (1.0e-3, 1000.0, 2, [0.0, -9.81, 0.0], 2,
                                                                             1.0)


def _run(launch_policy, tmp_path, ranks, opts):
    env = dict(os.environ, OPTS=opts, OMP_NUM_THREADS="1")
    env.setdefault("PETSC_OPTIONS", "-mat_partitioning_type current")
    out = launch_policy.retrying_run(subprocess.run)([MPIEXEC, "-n", str(ranks), sys.executable, HELPER],
                                                     cwd=str(tmp_path), env=env, capture_output=True, text=True,
                                                     timeout=600)
    assert out.returncode == 0, (out.stdout + out.stderr)[-4000:]
    line = [l for l in out.stdout.splitlines() if l.startswith("RESULT")][-1]
    return {k: float(v) for k, v in (item.split("=") for item in line.split()[2:])}


@pytest.mark.skipif(MPIEXEC is None, reason="mpiexec not available")
@pytest.mark.parametrize("opts", ["beamRigid=True", "beamRigid=False"])
def test_beam_channel_on_two_ranks_is_the_serial_answer(tmp_path, launch_policy, opts):
    """Sums of p, u, v, w (and squares) over owned nodes, the beam drag, the beam loads q1-q3 and the beam's vertex
    coordinates. Measured: equal to serial within 4e-16 relative on 2 and 3 ranks (drag1, itself 4e-6, within 4e-14);
    before the owned-element fix the drag was 1.12e-2 against 7.99e-3 serial (rigid), and the flexible beam's summed x
    coordinates 1.0331 against 1.0195."""
    (tmp_path / "1").mkdir()
    (tmp_path / "2").mkdir()
    serial = _run(launch_policy, tmp_path / "1", 1, opts)
    parallel = _run(launch_policy, tmp_path / "2", 2, opts)
    assert serial["drag0"] < -1e-3 and serial["q1"] > 0.5   # the beam is loaded
    if opts == "beamRigid=False":
        assert serial["xv"] > 1.01                          # and bends (1.0 is upright)
    worst = {k: abs(serial[k] - parallel[k]) / abs(serial[k]) for k in serial}
    assert max(worst.values()) < 1e-11, worst


STEP_RETRY = os.path.join(os.path.dirname(HERE), "step_retry_mpi.py")
_LU = " ".join("-%s_ksp_type preonly -%s_pc_type lu -%s_pc_factor_mat_solver_type superlu" % (p, p, p)
               for p in ("rans2p", "dye"))


def _retry_run(cwd, **overrides):
    import numpy
    cwd.mkdir()
    env = dict(os.environ, OMP_NUM_THREADS="1", DIR="rans2p_ib", SO="beam_channel_so.py", PETSC_OPTIONS=_LU)
    env.setdefault("FI_PROVIDER", "tcp")
    env["PYTHONPATH"] = os.pathsep.join([p for p in sys.path if p] + [env.get("PYTHONPATH", "")]).strip(os.pathsep)
    env.update(overrides)
    out = subprocess.run([sys.executable, STEP_RETRY], capture_output=True, text=True, timeout=600, env=env,
                         cwd=str(cwd))
    assert out.returncode == 0, (out.stdout + out.stderr)[-4000:]
    return numpy.loadtxt(str(cwd / "u.csv"), delimiter=",", skiprows=1), str(cwd / "accepted.txt")


@pytest.mark.parametrize("opts", ["beamRigid=True", "beamRigid=False"])
def test_a_later_models_failure_is_retried_from_the_step_start(tmp_path, opts):
    """The beam channel with a dye after RANS2P_IB (beam_channel_so.py, Sequential_MinModelStep): the dye's third
    solve is reported failed, after RANS2P_IB has taken its step, so the system step is cut and retried. The retry
    must give the run that takes the same steps without failing (step_retry_mpi.py's REPLAY), in both models.

    Measured, largest relative difference over each model's dofs (rigid and flexible alike):
      main: 9.2e-2. The retry skipped RANS2P_IB (Sequential_MinModelStep had already moved its last step's end to
            the failed attempt's end), which then ran one step behind the dye and never reached T.
      RANS2P_IB stepped again, without its state(): 6.4e-6, the lagged beam drag of the failed attempt.
      With RANS2P_IB.Coefficients.state(): 0.0. (Restoring the drag but not the beam: 6.6e-7, and the flexible
      beam's deflection at the retried step 1.8e-3 off.)
    A flexible beam's deflection history has no row from the failed attempt and is the replay's."""
    import numpy
    forced, accepted = _retry_run(tmp_path / "forced", OPTS=opts, FAILMODEL="dye", FAILAT="3")
    replay, _ = _retry_run(tmp_path / "replay", OPTS=opts, REPLAY=accepted)
    assert int(open(str(tmp_path / "forced" / "retries.txt")).read()) == 1
    times = numpy.loadtxt(accepted)
    assert 0.025 in numpy.round(times, 12) and abs(times[-1] - 0.04) < 1e-12, times
    assert forced.shape == replay.shape and numpy.abs(forced[:, :4] - replay[:, :4]).max() == 0.0
    for key in numpy.unique(forced[:, 0]):
        k = forced[:, 0] == key
        worst = numpy.abs(forced[k, 4] - replay[k, 4]).max() / numpy.abs(replay[k, 4]).max()
        assert worst < 1e-12, (key, worst)
    deflection = tmp_path / "forced" / "deflectionHistory.txt"
    if opts == "beamRigid=False":
        numpy.testing.assert_array_equal(numpy.loadtxt(str(deflection)),
                                         numpy.loadtxt(str(tmp_path / "replay" / "deflectionHistory.txt")))
        numpy.testing.assert_array_equal(numpy.loadtxt(str(deflection))[:, 0], times)
