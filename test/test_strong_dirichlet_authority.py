"""A strong-Dirichlet dof shared between ranks is one identity row, whatever each rank's dictionary says.

With ``forceStrongConditions`` each rank replaces the rows of the dofs in its own dictionary by ``u - g`` and
an identity row, and the reverse add sums the copies of a shared dof. A node on the rank cut is Dirichlet on
a rank only if that rank's own elements carry a face that says so, so the sum could be an identity plus a
partial PDE row (and where both ranks hold it, a diagonal of 2). The step2d corner (5, -1) is tagged as an
outflow node (p = 0 by the tag) while v = 0 comes from the bottom-wall face: with the wall-face element on
rank 0 and the outflow-face element on rank 1, rank 1 holds the corner without the v condition. Before the
fix 2 ranks gave p2 = 3.43 against 2.51 and Newton ran to its 100-iteration cap; now every rank count gives
the serial answer to rounding.
"""
import os
import shutil
import subprocess
import sys

import numpy
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
HELPER = os.path.join(HERE, "strong_dirichlet_partition_mpi.py")
POISSON = os.path.join(HERE, "strong_dirichlet_poisson_mpi.py")


def _mpiexec():
    bindir = os.path.dirname(sys.executable)
    for name in ("mpiexec", "mpirun"):
        candidate = os.path.join(bindir, name)
        if os.path.exists(candidate):
            return candidate
    return shutil.which("mpiexec") or shutil.which("mpirun")


MPIEXEC = _mpiexec()


def _run(ranks, cwd, helper=None, **env_extra):
    env = dict(os.environ, OMP_NUM_THREADS="1", **{k: str(v) for k, v in env_extra.items()})
    env.setdefault("FI_PROVIDER", "tcp")
    env.setdefault("OMPI_MCA_rmaps_base_oversubscribe", "1")
    env.setdefault("PRTE_MCA_rmaps_default_mapping_policy", ":oversubscribe")
    env["PYTHONPATH"] = os.pathsep.join([p for p in sys.path if p] + [env.get("PYTHONPATH", "")]).strip(os.pathsep)
    argv = [sys.executable, helper or HELPER]
    if MPIEXEC:
        argv = [MPIEXEC, "-n", str(ranks)] + argv
    out = subprocess.run(argv, capture_output=True, text=True, timeout=900, env=env, cwd=str(cwd))
    assert out.returncode == 0, out.stdout[-2000:] + out.stderr[-2000:]
    line = [l for l in out.stdout.splitlines() if l.startswith("RESULT")][0]
    return {k: float(v) for k, v in (item.split("=") for item in line.split()[1:])}


def _need_superlu_dist():
    pytest.importorskip("petsc4py")
    from petsc4py import PETSc
    if not PETSc.Sys.hasExternalPackage("superlu_dist"):
        pytest.skip("needs superlu_dist")


@pytest.mark.skipif(MPIEXEC is None, reason="mpiexec not available")
@pytest.mark.parametrize("ranks", [2, 4])
@pytest.mark.parametrize("trap", [0, 1])
def test_rans2p_step_with_the_cut_through_a_dirichlet_corner_gives_the_serial_answer(ranks, trap, tmp_path):
    _need_superlu_dist()
    serial = _run(1, tmp_path)
    parallel = _run(ranks, tmp_path, **({"TRAP": 1} if trap else {}))
    for key in ("p2", "ps", "u2", "us", "v2", "vs"):
        numpy.testing.assert_allclose(parallel[key], serial[key], rtol=1e-9, atol=1e-12, err_msg=key)
    assert parallel["newton"] == serial["newton"] < 20


@pytest.mark.skipif(MPIEXEC is None, reason="mpiexec not available")
@pytest.mark.parametrize("ranks", [2, 4])
@pytest.mark.parametrize("trap", [0, 1])
def test_poisson_with_flag_selected_strong_dirichlet_dofs_gives_the_serial_answer(ranks, trap, tmp_path):
    """The same defect in a scalar model of the base Transport class (kept-DOF strong Dirichlet on the left face
    only, chosen by flag): before the fix the trapped corner gave s = -173.293 on 2 and 4 ranks against -171.802 on one."""
    _need_superlu_dist()
    serial = _run(1, tmp_path, POISSON)
    parallel = _run(ranks, tmp_path, POISSON, **({"TRAP": 1} if trap else {}))
    for key in ("s", "s2"):
        numpy.testing.assert_allclose(parallel[key], serial[key], rtol=1e-9, atol=1e-12, err_msg=key)
