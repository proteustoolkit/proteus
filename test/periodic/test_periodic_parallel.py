"""The periodic duct on two ranks.

It segfaulted in object construction: duct.py asked n for an element partition
without overlap (what the parallel periodic numbering needs) but left the
domain's MeshOptions at the default node partition, so the mesh was generated in
pieces with no global copy and the global trial space got None for its mesh
arrays. Then SimpleNavierStokes2D, the Schur complement preconditioner it uses
with KSP_petsc4py, was refused as a linear smoother in parallel.
"""
import os
import shutil
import subprocess
import sys

import numpy
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
HELPER = os.path.join(HERE, "duct_partition_mpi.py")
OPTIONS = os.path.join(HERE, "petsc", "petsc.options.schur.selfp_petsc.superlu")


def _mpiexec():
    bindir = os.path.dirname(sys.executable)
    for name in ("mpiexec", "mpirun"):
        candidate = os.path.join(bindir, name)
        if os.path.exists(candidate):
            return candidate
    return shutil.which("mpiexec") or shutil.which("mpirun")


MPIEXEC = _mpiexec()


def _run(ranks, cwd, **overrides):
    options = OPTIONS
    if ranks > 1:
        # the serial direct solver does not run on several ranks
        options = os.path.join(str(cwd), "petsc.options.superlu_dist")
        with open(OPTIONS) as f, open(options, "w") as g:
            g.write(f.read().replace("superlu\n", "superlu_dist\n"))
    env = dict(os.environ, OMP_NUM_THREADS="1", SOLVER_OPTIONS=options,
               PETSC_OPTIONS="-mat_partitioning_type current")
    env.setdefault("FI_PROVIDER", "tcp")
    env.setdefault("OMPI_MCA_rmaps_base_oversubscribe", "1")
    env.setdefault("PRTE_MCA_rmaps_default_mapping_policy", ":oversubscribe")
    env["PYTHONPATH"] = os.pathsep.join(
        [p for p in sys.path if p] + [env.get("PYTHONPATH", "")]).strip(os.pathsep)
    env.update({k: str(v) for k, v in overrides.items()})
    # even for one rank (test_archive_parallel_metadata.run_ranks says why)
    argv = [sys.executable, HELPER]
    if MPIEXEC:
        argv = [MPIEXEC, "-n", str(ranks)] + argv
    out = subprocess.run(argv, capture_output=True, text=True, timeout=900, env=env, cwd=str(cwd))
    assert out.returncode == 0, out.stdout[-2000:] + out.stderr[-2000:]
    line = [l for l in out.stdout.splitlines() if l.startswith("RESULT")][0]
    return {k: float(v) for k, v in (item.split("=") for item in line.split()[1:])}


@pytest.mark.skipif(MPIEXEC is None, reason="mpiexec not available")
@pytest.mark.parametrize("tight", [0, 1])
def test_duct_on_two_ranks_matches_serial_at_the_same_newton_tolerance(tight, tmp_path):
    """tight=0 is the file's Newton tolerance (1e-8), where neither run is converged: the
    serial answer is 10% off the tight one, and the two ranks must stop where one does.
    tight=1 is 1e-13, where both must reach the same converged answer.

    Before the Newton fix the tight two-rank run did not: the first residual of each solve was
    handed to the linear solver before the ranks' contributions at shared dofs were summed, so
    every first step was off by 20-50%, the pressure came out as +-1e-10 noise where the serial
    one is 1e-24, and the run ended after three steps. (The loose run only looked right: its
    extra Newton iterations happened to land on the converged answer.)
    """
    from petsc4py import PETSc
    if not PETSc.Sys.hasExternalPackage("superlu_dist"):
        pytest.skip("needs superlu_dist")
    kw = {"TIGHT": 1} if tight else {}
    serial = _run(1, tmp_path, **kw)
    parallel = _run(2, tmp_path, **kw)
    rtol = 1e-5 if tight else 3e-2
    numpy.testing.assert_allclose(parallel["u2"], serial["u2"], rtol=rtol)
    numpy.testing.assert_allclose(parallel["us"], serial["us"], rtol=rtol)
    # the pressure is ~1e-20 in the sum of squares; the broken run had 1e-12
    assert parallel["p2"] < 1e-18, parallel
