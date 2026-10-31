"""VOS3P edge-based FCT (STABILIZATION_TYPE=4) must not depend on the partition.

The limiter of an owned dof reads uLow of its graph neighbours. With one overlap layer those across a rank cut are
ghosts whose uLow was built from a truncated graph row, so 1D advection of test/VOS on 2 ranks kept a pulse peak of
0.044 where serial keeps 0.281. uLow is now scattered from the owners before the FCT step; the control runs without
that scatter and differs.
"""
import os
import shutil
import subprocess
import sys

import numpy
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
HELPER = os.path.join(HERE, "vos_partition_mpi.py")


def _mpiexec():
    bindir = os.path.dirname(sys.executable)
    for name in ("mpiexec", "mpirun"):
        candidate = os.path.join(bindir, name)
        if os.path.exists(candidate):
            return candidate
    return shutil.which("mpiexec") or shutil.which("mpirun")


MPIEXEC = _mpiexec()


def _field(ranks, cwd, **overrides):
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
    return numpy.loadtxt(os.path.join(str(cwd), "u.csv"), delimiter=",", skiprows=1)


@pytest.mark.skipif(MPIEXEC is None, reason="mpiexec not available")
def test_vos3p_fct_on_two_ranks_is_the_serial_answer(tmp_path):
    for d in ("1", "2", "3"):
        (tmp_path / d).mkdir()
    serial = _field(1, tmp_path / "1")
    parallel = _field(2, tmp_path / "2")
    assert numpy.abs(serial - parallel).max() < 1e-8
    control = _field(2, tmp_path / "3", NOFIX=1)
    assert numpy.abs(serial - control).max() > 1e-3
