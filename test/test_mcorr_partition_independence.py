"""The edge-based mass correction with FCT (MCorr) must run in parallel and must not depend on the partition.

Its solver, NewtonWithL2ProjectionForMassCorrection, swapped the SparseFactor of the LU linear solver between the
Jacobian and the mass matrix, so it could not run with KSP, which every parallel run uses. With KSP it now puts the mass
matrix values in the matrix of the solver and prepares it again, solves the L2 projection with a fixed tight tolerance
(not an Eisenstat-Walker one, which also overwrote the Newton residual norm), and refreshes the ghosts of the projection.
The FCT kernel reads R of the neighbours, solH and the limited solution, which are written into the VOF dofs, ghosts
included; it runs in two phases with the owners' values scattered between them.

The redistancing norm of NCLS was a per-rank norm, so the ranks did different numbers of redistancing iterations and
deadlocked in the collective calls; it is a global norm over the owned dofs now.

The run couples NCLS, VOF and MCorr. 2 ranks match serial to 1e-8 in all three; the control without the MCorr scatters
differs; the KSP path agrees with the LU path in serial.
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
HELPER = os.path.join(HERE, "mcorr_partition_mpi.py")


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
    env["PETSC_OPTIONS"] = "-ncls_ksp_rtol 1e-14 -ncls_ksp_atol 1e-40 -vof_ksp_rtol 1e-14 -vof_ksp_atol 1e-40 -mcorr_ksp_rtol 1e-14 -mcorr_ksp_atol 1e-40 -mcorr_ksp_max_it 5000"
    env.update({k: str(v) for k, v in overrides.items()})
    argv = [sys.executable, HELPER] if ranks == 1 else [MPIEXEC, "-n", str(ranks), sys.executable, HELPER]
    out = launch_output.run(argv, cwd=cwd, env=env, timeout=1800)
    result = numpy.load(os.path.join(str(cwd), "final.npz"))
    return {key: result[key][:, 2] for key in result.files}


def _difference(a, b):
    return max(numpy.abs(a[key] - b[key]).max() for key in a)


@pytest.mark.skipif(MPIEXEC is None, reason="mpiexec not available")
def test_edge_based_mass_correction_with_fct_on_two_ranks_is_the_serial_answer(tmp_path):
    for d in ("1", "2", "3", "4"):
        (tmp_path / d).mkdir()
    serial_lu = _field(1, tmp_path / "1", PAR="False")
    serial = _field(1, tmp_path / "2")
    # the KSP path of the solver (new) is the LU one (the only one there was) at the tolerance of the solves
    assert _difference(serial_lu, serial) < 1e-8
    parallel = _field(2, tmp_path / "3")
    assert _difference(serial, parallel) < 1e-8
    control = _field(2, tmp_path / "4", NOFIX=1)
    assert _difference(serial, control) > 1e-5
