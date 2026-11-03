"""NCLS edge-based stabilization (STABILIZATION_TYPE=2 and 1) must not depend on the partition.

The dissipation of an owned dof reads the smoothness indicator psi and, with entropy viscosity, the normalized entropy
residual of its graph neighbours. With one overlap layer those across a rank cut are ghosts whose graph rows are
truncated, so the residual of the first stage differed from serial by up to 30% of its size. The residual kernel now
runs in two phases: the first computes psi and the entropy residual, Python scatters them from the owners, and the
second takes the scattered values. The solver also refreshes the ghost entries of the stage solution after each mass
matrix solve; without that the second stage read stale ghosts.

The comparison is of the solution at the final time with the mass matrix solves converged far enough (PETSc options
and tolerances) for 1e-8: with the default tolerances 2 ranks differ from serial by 1e-2. Entropy viscosity has a
discontinuous branch (etaMin == etaMax) that amplifies round-off differences step by step, so it stops earlier and
its tolerance is 1e-6: on CI the 2-rank answer differed from serial by 6e-8 on linux-64, 2.4e-7 on linux-aarch64 and
9e-16 on macOS arm64, the same on every run of a host, so it is round-off through that branch, not a partition effect.
The control runs without the scatter of psi and of the entropy residual and differs.
"""
import os
import shutil
import subprocess
import sys

import numpy
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
HELPER = os.path.join(HERE, "ncls_edge_partition_mpi.py")


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
    env["PETSC_OPTIONS"] = "-ncls_ksp_rtol 1e-14 -ncls_ksp_atol 1e-40 -ncls_ksp_max_it 5000"
    env.update({k: str(v) for k, v in overrides.items()})
    argv = [sys.executable, HELPER] if ranks == 1 else [MPIEXEC, "-n", str(ranks), sys.executable, HELPER]
    out = subprocess.run(argv, capture_output=True, text=True, timeout=1800, env=env, cwd=str(cwd))
    assert out.returncode == 0, out.stdout[-2000:] + out.stderr[-2000:]
    return numpy.loadtxt(os.path.join(str(cwd), "u.csv"), delimiter=",", skiprows=1)


@pytest.mark.skipif(MPIEXEC is None, reason="mpiexec not available")
@pytest.mark.parametrize("stabilization,final_time,tolerance", [(2, 0.08, 1e-8), (1, 0.04, 1e-6)],
                         ids=["smoothness", "entropy_viscosity"])
def test_ncls_edge_based_on_two_ranks_is_the_serial_answer(tmp_path, stabilization, final_time, tolerance):
    for d in ("1", "2", "3"):
        (tmp_path / d).mkdir()
    serial = _field(1, tmp_path / "1", STAB=stabilization, TFINAL=final_time)
    parallel = _field(2, tmp_path / "2", STAB=stabilization, TFINAL=final_time)
    assert numpy.abs(serial - parallel).max() < tolerance
    # one ghosted vector per model, not one per array scattered
    assert int(open(os.path.join(str(tmp_path / "2"), "dofvecs.txt")).read()) <= 1
    control = _field(2, tmp_path / "3", NOSCAT=1, STAB=stabilization, TFINAL=final_time)
    assert numpy.abs(serial - control).max() > 1e-4
