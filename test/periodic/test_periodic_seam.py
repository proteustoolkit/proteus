"""The periodic seam of a C0 space with a weak-Dirichlet exterior flux (Advection_DiagonalUpwind_Diffusion_IIPG_exterior).

Until 2026-11-08 the serial DOF map kept two DOFs at each periodic pair for a weak-Dirichlet flux, coupled only by
the flux's weak Dirichlet condition with the partner's trace as the value: one-sided -a grad(u).n on each side and a
penalty whose partner half the Jacobian lags. That fixed point is inconsistent (L2 1.06e-1, 1.12e-1, 1.15e-1 on
8x8, 16x16, 32x32 cells, after 620, 2225, 8324 Newton iterations). The parallel periodic numbering did merge the
pairs, but the flux kept applying its weak Dirichlet flux on the now interior seam (8.2e-2 on 16x16 on 2 and 4 ranks,
Newton not converging) and crashed (IndexError) where a seam point's partner was on another rank. Now the pairs are one
DOF on both paths and the seam carries no boundary term (NF_base.seamIsInterior).
"""
import importlib.util
import os
import shutil
import sys

import numpy
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
HELPER = os.path.join(HERE, "seam_poisson_mpi.py")
_spec = importlib.util.spec_from_file_location("launch_output", os.path.join(HERE, "..", "launch_output.py"))
launch_output = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(launch_output)


def _mpiexec():
    bindir = os.path.dirname(sys.executable)
    for name in ("mpiexec", "mpirun"):
        candidate = os.path.join(bindir, name)
        if os.path.exists(candidate):
            return candidate
    return shutil.which("mpiexec") or shutil.which("mpirun")


MPIEXEC = _mpiexec()


def _direct_solver():
    from petsc4py import PETSc
    for package in ("superlu_dist", "mumps"):
        if PETSc.Sys.hasExternalPackage(package):
            return package
    return None


def _run(cwd, ranks=1, partitioner="current", **seam):
    env = dict(os.environ, OMP_NUM_THREADS="1")
    env.setdefault("FI_PROVIDER", "tcp")
    env.setdefault("OMPI_MCA_rmaps_base_oversubscribe", "1")
    env.setdefault("PRTE_MCA_rmaps_default_mapping_policy", ":oversubscribe")
    env["PYTHONPATH"] = os.pathsep.join(
        [p for p in sys.path if p] + [env.get("PYTHONPATH", "")]).strip(os.pathsep)
    options = "-mat_partitioning_type %s" % partitioner
    if seam.get("PARALLEL"):
        options += (" -seam_ksp_type preonly -seam_pc_type lu -seam_pc_factor_mat_solver_type %s"
                    % _direct_solver())
    env["PETSC_OPTIONS"] = options
    env.update({"SEAM_" + k: str(v) for k, v in seam.items()})
    argv = [sys.executable, HELPER]
    if MPIEXEC:   # even for one rank (test_archive_parallel_metadata.run_ranks says why)
        argv = [MPIEXEC, "-n", str(ranks)] + argv
    out = launch_output.run(argv, cwd, env, timeout=600)
    line = [l for l in out.stdout.splitlines() if l.startswith("RESULT")][0]
    return {k: float(v) for k, v in (item.split("=") for item in line.split()[1:])}


@pytest.mark.parametrize("bx", [0.0, 1.0])
def test_a_periodic_seam_is_interior_and_converges_at_second_order(bx, tmp_path):
    errors = []
    for nn in (9, 17, 33):
        periodic = _run(tmp_path, NN=nn, BX=bx)
        assert periodic["newton"] == 1, periodic       # a linear problem: an exact Jacobian
        reference = _run(tmp_path, NN=nn, BX=bx, PERIODIC=0)
        # the same solution with its values at x = 0, 1 imposed instead: the same order and size of error
        assert periodic["L2"] < 1.5*reference["L2"], (nn, periodic, reference)
        errors.append(periodic["L2"])
    rates = numpy.log2(numpy.array(errors[:-1])/numpy.array(errors[1:]))
    # 4.0e-2, 1.08e-2, 2.81e-3 (bx = 0); 1.06e-1, 1.12e-1, 1.15e-1 before
    assert numpy.all(rates > 1.8), (errors, rates)


@pytest.mark.skipif(MPIEXEC is None, reason="mpiexec not available")
@pytest.mark.parametrize("ranks,partitioner", [(1, "current"), (2, "current"), (4, "current"), (4, "parmetis")])
def test_the_parallel_periodic_numbering_solves_as_the_serial_merge(ranks, partitioner, tmp_path):
    """Serial 1.074283e-02 (16x16, bx = 1); the parallel path agreed to every digit printed on 1, 2, 4 ranks with
    'current', 3 and 4 with parmetis and ptscotch (with parmetis on 4 ranks every seam point's partner is on another
    rank)."""
    if _direct_solver() is None:
        pytest.skip("needs superlu_dist or mumps")
    if partitioner != "current":
        from petsc4py import PETSc
        if not PETSc.Sys.hasExternalPackage(partitioner):
            pytest.skip("needs %s" % partitioner)
    serial = _run(tmp_path, NN=17, BX=1.0)
    parallel = _run(tmp_path, ranks=ranks, partitioner=partitioner, NN=17, BX=1.0, PARALLEL=1)
    assert parallel["newton"] == 1, parallel
    numpy.testing.assert_allclose(parallel["L2"], serial["L2"], rtol=1e-9)
