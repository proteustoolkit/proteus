"""VOF and VOF3P edge-based stabilization with FCT (STABILIZATION_TYPE=3, 2 and 4) must not depend on the partition.

An owned dof reads the smoothness indicator, uLow and the limiter coefficients R of its graph neighbours. With one
overlap layer those across a rank cut are ghosts whose graph rows are truncated, so a blob advected across the cut
on 2 ranks differed from serial by 5e-3 (peak 0.63). psi and R are now scattered from the owners (R between the two
phases of the FCT kernel), as are uLow and the limited solution; with entropy viscosity the normalized entropy
residual is scattered between the two phases of the residual kernel. The controls run without the scatters and differ.

The run uses the adaptive Min_dt_controller. Its step is chosen from the edge based cfl of the owned dofs only, since
the cfl of a ghost is built from a truncated row and can exceed the global maximum; the ghost-cfl control puts
garbage there, which the step must ignore.
"""
import os
import shutil
import subprocess
import sys

import numpy
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
HELPER = os.path.join(HERE, "vof_fct_partition_mpi.py")


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
@pytest.mark.parametrize("model", ["VOF", "VOF3P"])
@pytest.mark.parametrize("stabilization", [3, 2, 4], ids=["smoothness", "entropy_viscosity", "dk_fct"])
def test_vof_edge_based_fct_on_two_ranks_is_the_serial_answer(tmp_path, model, stabilization):
    for d in ("1", "2", "3", "4", "5"):
        (tmp_path / d).mkdir()
    serial = _field(1, tmp_path / "1", STAB=stabilization, MODEL=model)
    parallel = _field(2, tmp_path / "2", STAB=stabilization, MODEL=model)
    assert numpy.abs(serial - parallel).max() < 1e-8
    control = _field(2, tmp_path / "3", NOFIX=1, STAB=stabilization, MODEL=model)
    assert numpy.abs(serial - control).max() > 1e-3
    # the time step reads the cfl of the dofs it owns only: garbage in the ghosts' must not matter
    ghosts = _field(2, tmp_path / "4", GHOSTCFL=1, STAB=stabilization, MODEL=model)
    assert numpy.abs(serial - ghosts).max() < 1e-8
    reads_ghosts = _field(2, tmp_path / "5", GHOSTCFL=1, OLDMAX=1, STAB=stabilization, MODEL=model)
    assert numpy.abs(serial - reads_ghosts).max() > 1e-6


@pytest.mark.skipif(MPIEXEC is None, reason="mpiexec not available")
@pytest.mark.parametrize("model", ["VOF", "VOF3P"])
def test_vof_taylor_galerkin_entropy_viscosity_on_two_ranks_is_the_serial_answer(tmp_path, model):
    """STABILIZATION_TYPE=1 (Taylor-Galerkin with entropy viscosity, two-stage implicit, consistent mass, no FCT):
    the entropy viscosity is normalized by the entropy's deviation over the whole domain (EntropyViscosity), so each
    rank's kernel output is renormalized after the solve. Measured 3.5e-10 (2 ranks), 3.9e-10 (4) relative; the
    control keeps each rank's own normalization and the 2-rank run blows up (8e5)."""
    for d in ("1", "2", "3"):
        (tmp_path / d).mkdir()
    serial = _field(1, tmp_path / "1", STAB=1, MODEL=model)
    assert numpy.abs(serial - _field(2, tmp_path / "2", STAB=1, MODEL=model)).max() < 1e-8
    assert numpy.abs(serial - _field(2, tmp_path / "3", STAB=1, NOREDUCE=1, MODEL=model)).max() > 1e-3
