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
