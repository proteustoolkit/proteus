"""VOS3P edge-based FCT (STABILIZATION_TYPE=4) must not depend on the partition.

The FCT limiter of an owned dof reads uLow of its graph neighbours. With one overlap layer the neighbours across a
rank cut are ghosts whose graph rows are truncated: uLow is now scattered from the owners. Without it a 2-rank run
kept a pulse peak of 0.044 where serial keeps 0.281. The time step is chosen from the edge based cfl of the owned
dofs only (the cfl of a ghost, like its lumped mass, comes from a truncated row and can exceed the global maximum;
that made the unit-square run differ by 1e-4): the ghost-cfl control puts garbage there, which the step must
ignore. (Scattering the limiter coefficients and the limited solution as well, as VOF needs, changed nothing here.)
"""
import importlib.util
import os
import shutil
import sys

import numpy
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
# a failed launch keeps its whole output (loaded by path: --import-mode=importlib)
_spec = importlib.util.spec_from_file_location("launch_output", os.path.join(HERE, "launch_output.py"))
launch_output = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(launch_output)
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
    out = launch_output.run(argv, cwd=cwd, env=env, timeout=1800)
    return numpy.loadtxt(os.path.join(str(cwd), "u.csv"), delimiter=",", skiprows=1)


@pytest.mark.skipif(MPIEXEC is None, reason="mpiexec not available")
def test_vos3p_fct_on_two_ranks_is_the_serial_answer(tmp_path):
    for d in ("1", "2", "3", "4", "5"):
        (tmp_path / d).mkdir()
    serial = _field(1, tmp_path / "1")
    parallel = _field(2, tmp_path / "2")
    assert numpy.abs(serial - parallel).max() < 1e-10
    assert numpy.abs(serial - _field(2, tmp_path / "3", NOFIX=1)).max() > 1e-3
    ghosts = _field(2, tmp_path / "4", GHOSTCFL=1)
    assert numpy.abs(serial - ghosts).max() < 1e-10
    reads_ghosts = _field(2, tmp_path / "5", GHOSTCFL=1, OLDMAX=1)
    assert numpy.abs(serial - reads_ghosts).max() > 1e-6
