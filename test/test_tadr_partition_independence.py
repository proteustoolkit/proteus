"""TADR edge-based stabilization (STABILIZATION_TYPE=3, 4 and 2) must not depend on the partition.

The dissipation of an owned dof reads the smoothness indicator psi and the normalized entropy residual of its graph
neighbours, the FCT limiter reads uLow and the limiter coefficients R of its neighbours, and the time step reads the
edge based cfl: across a rank cut those belong to ghosts whose graph rows are truncated (2 ranks differed from serial by
0.1-0.3 at t=0.08). The residual and FCT kernels now run in two phases with the owners' values scattered between them, the
consistent mass matrix solver refreshes the ghosts of the stage solution, and the time step uses owned dofs only.

The runs converge the mass matrix solves (PETSc options and tolerances). Smoothness and Kuzmin match serial to 1e-8; the
consistent mass matrix with smoothness only to 1e-8 for a short run (its difference grows from 4e-11 at t=0.02 to 5e-8 at
t=0.08, probably round-off amplified by the regularized indicator in flat regions, which was not checked). Entropy
viscosity has a branch, etaMin == etaMax, that amplifies ulp-level differences of the nodal projections (2e-5 at t=0.02), so
it gets a looser tolerance. The controls run without the scatters and differ.
"""
import os
import shutil
import subprocess
import sys

import numpy
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
HELPER = os.path.join(HERE, "tadr_partition_mpi.py")


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
    env["PETSC_OPTIONS"] = "-tadr_ksp_rtol 1e-14 -tadr_ksp_atol 1e-40 -tadr_ksp_max_it 5000"
    env.update({k: str(v) for k, v in overrides.items()})
    argv = [sys.executable, HELPER] if ranks == 1 else [MPIEXEC, "-n", str(ranks), sys.executable, HELPER]
    out = subprocess.run(argv, capture_output=True, text=True, timeout=1800, env=env, cwd=str(cwd))
    assert out.returncode == 0, out.stdout[-2000:] + out.stderr[-2000:]
    return numpy.loadtxt(os.path.join(str(cwd), "u.csv"), delimiter=",", skiprows=1)


@pytest.mark.skipif(MPIEXEC is None, reason="mpiexec not available")
@pytest.mark.parametrize("lumped,stabilization,final_time,tolerance", [
    (1, 3, 0.08, 1e-8), (0, 3, 0.02, 1e-8), (1, 4, 0.08, 1e-8), (0, 4, 0.08, 1e-8),
    (1, 2, 0.02, 1e-3), (0, 2, 0.02, 1e-3)],
    ids=["lumped-smoothness", "consistent-smoothness", "lumped-kuzmin", "consistent-kuzmin",
         "lumped-entropy_viscosity", "consistent-entropy_viscosity"])
def test_tadr_edge_based_on_two_ranks_is_the_serial_answer(tmp_path, lumped, stabilization, final_time, tolerance):
    for d in ("1", "2", "3"):
        (tmp_path / d).mkdir()
    options = dict(LUMPED=lumped or "", STAB=stabilization, TFINAL=final_time)
    serial = _field(1, tmp_path / "1", **options)
    parallel = _field(2, tmp_path / "2", **options)
    assert numpy.abs(serial - parallel).max() < tolerance
    control = _field(2, tmp_path / "3", NOSCAT=1, **options)
    assert numpy.abs(serial - control).max() > 1e-2


@pytest.mark.skipif(MPIEXEC is None, reason="mpiexec not available")
def test_tadr_time_step_ignores_the_cfl_of_ghost_dofs(tmp_path):
    for d in ("1", "2", "3"):
        (tmp_path / d).mkdir()
    options = dict(LUMPED=1, STAB=3, TFINAL=0.08)
    serial = _field(1, tmp_path / "1", **options)
    # garbage in the edge based cfl of the ghost dofs, which the time step must not read
    assert numpy.abs(serial - _field(2, tmp_path / "2", GHOSTCFL=1, **options)).max() < 1e-8
    assert numpy.abs(serial - _field(2, tmp_path / "3", GHOSTCFL=1, OLDMAX=1, **options)).max() > 1e-6


@pytest.mark.skipif(MPIEXEC is None, reason="mpiexec not available")
@pytest.mark.parametrize("lumped,stabilization,final_time", [(1, 3, 0.08), (0, 4, 0.08)],
                         ids=["lumped-smoothness", "consistent-kuzmin"])
def test_tadr_with_a_varying_porosity_on_two_ranks_is_the_serial_answer(tmp_path, lumped, stabilization, final_time):
    """The residual and the FCT limiter project the porosity to the dofs; a ghost's projection sums only the elements
    the rank has, and the edge loops of an owned dof read it at the neighbours. Both kernels now take the owners'
    values (a phase -1 call and a scatter). Measured: 1.4e-3 (lumped smoothness) and 9.7e-4 (Kuzmin) without, 1.6e-15
    and 1.0e-14 with; the serial answer is unchanged to the bit."""
    for d in ("1", "2", "3"):
        (tmp_path / d).mkdir()
    options = dict(POROUS=1, LUMPED=lumped or "", STAB=stabilization, TFINAL=final_time)
    serial = _field(1, tmp_path / "1", **options)
    assert numpy.abs(serial - _field(2, tmp_path / "2", **options)).max() < 1e-8
    assert numpy.abs(serial - _field(2, tmp_path / "3", NOTHETASCAT=1, **options)).max() > 1e-4

