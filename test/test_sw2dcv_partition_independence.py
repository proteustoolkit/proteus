"""SW2DCV edge-based limiting (circular_damBreak, lumped mass matrix) must not depend on the partition.

Three things made a 2-rank run drift from serial (1.6e-4 in h at t=0.5): the relaxation of the local bounds used a
domain size summed over the ranks, which counts every overlap element twice; the second differences behind the
bounds and the limiter coefficients R, hLow/huLow and the limited solution of an owned dof are read at ghost
neighbours whose graph rows are truncated. The first is now the trace of the lumped mass matrix; the others are
computed on the owners and scattered, the limiting iterations being driven from Python. The controls run without
the scatters and with the old domain size, and differ.

The time step is chosen from the edge based cfl of the owned dofs only, since the cfl of a ghost dof comes from a
truncated row and can exceed the global maximum; the ghost-cfl control puts garbage there, which the step must ignore.
"""
import os
import shutil
import subprocess
import sys

import numpy
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
HELPER = os.path.join(HERE, "sw2dcv_partition_mpi.py")


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
def test_sw2dcv_circular_damBreak_on_two_ranks_is_the_serial_answer(tmp_path):
    for d in ("1", "2", "3", "4", "5", "6", "7", "8", "9"):
        (tmp_path / d).mkdir()
    serial = _field(1, tmp_path / "1")
    parallel = _field(2, tmp_path / "2")
    assert numpy.abs(serial - parallel).max() < 1e-10
    assert numpy.abs(serial - _field(2, tmp_path / "3", NOSCAT=1)).max() > 1e-9
    assert numpy.abs(serial - _field(2, tmp_path / "4", OLDVOL=1)).max() > 1e-6
    # the time step reads the cfl of the dofs it owns only: garbage in the ghosts' must not matter
    ghosts = _field(2, tmp_path / "5", GHOSTCFL=1)
    assert numpy.abs(serial - ghosts).max() < 1e-10
    reads_ghosts = _field(2, tmp_path / "6", GHOSTCFL=1, OLDMAX=1)
    assert numpy.abs(serial - reads_ghosts).max() > 1e-6
    # the consistent mass matrix: a linear solve per stage, whose tolerance decides how close the runs are
    serial_c = _field(1, tmp_path / "7", CONSISTENT=1, TIGHT=1)
    assert numpy.abs(serial_c - _field(2, tmp_path / "8", CONSISTENT=1, TIGHT=1)).max() < 1e-8
    assert numpy.abs(serial_c - _field(2, tmp_path / "9", CONSISTENT=1)).max() > 1e-6
