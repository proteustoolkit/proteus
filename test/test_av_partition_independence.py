"""RANS3PF edge-based artificial viscosity (ARTIFICIAL_VISCOSITY 3 and 4) must not depend on the partition.

The dissipation of an owned dof uses the smoothness indicator of its graph neighbours. With one overlap layer
the neighbours across a rank cut are ghosts whose graph rows are truncated, so their indicator differs from the
serial one: damBreak, the TwoPhaseFlow default, drifted from serial by 2e-2 in gauge pressure (scale 23) by
t=0.1 with the cut at x=1.6. The indicator is now computed in Python on the owned dofs and scattered to the ghosts;
the control runs without that scatter and differs.
"""
import os
import shutil
import subprocess
import sys

import numpy
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
HELPER = os.path.join(HERE, "av_partition_mpi.py")


def _mpiexec():
    bindir = os.path.dirname(sys.executable)
    for name in ("mpiexec", "mpirun"):
        candidate = os.path.join(bindir, name)
        if os.path.exists(candidate):
            return candidate
    return shutil.which("mpiexec") or shutil.which("mpirun")


MPIEXEC = _mpiexec()


def _pressure(ranks, cwd, **overrides):
    env = dict(os.environ, OMP_NUM_THREADS="1")
    env.setdefault("FI_PROVIDER", "tcp")
    env.setdefault("OMPI_MCA_rmaps_base_oversubscribe", "1")
    env.setdefault("PRTE_MCA_rmaps_default_mapping_policy", ":oversubscribe")
    env["PYTHONPATH"] = os.pathsep.join(
        [p for p in sys.path if p] + [env.get("PYTHONPATH", "")]).strip(os.pathsep)
    env.update({k: str(v) for k, v in overrides.items()})
    argv = [sys.executable, HELPER] if ranks == 1 else [MPIEXEC, "-n", str(ranks), sys.executable, HELPER]
    out = subprocess.run(argv, capture_output=True, text=True, timeout=1800, env=env, cwd=str(cwd))
    assert out.returncode == 0, out.stdout[-2000:] + out.stderr[-2000:]
    return numpy.loadtxt(os.path.join(str(cwd), "pressure.csv"), delimiter=",", skiprows=1)


@pytest.mark.skipif(MPIEXEC is None, reason="mpiexec not available")
def test_damBreak_default_artificial_viscosity_on_two_ranks_is_the_serial_answer(tmp_path):
    for d in ("1", "2", "3"):
        (tmp_path / d).mkdir()
    serial = _pressure(1, tmp_path / "1")
    parallel = _pressure(2, tmp_path / "2")
    assert numpy.abs(serial - parallel).max() < 1e-8
    control = _pressure(2, tmp_path / "3", NOFIX=1)
    assert numpy.abs(serial - control).max() > 1e-3
