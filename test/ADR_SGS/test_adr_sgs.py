#!/usr/bin/env python
"""Convergence rates of ADR's subgrid-error stabilization with a variable diffusion coefficient.

The case is test/ci/ladr_sgs_2d_p.py: div(b u) - div(a(x) grad u) = f on the unit square, a = 1 + 0.5 x y,
u = sin(2.3 x + 0.4) cos(1.7 y), structured triangles with nnx = 5, 9, 17, 33.

The subgrid error is -tau * (b.grad(u) - div(a grad u) - f) and needs the diffusion term in both the
residual and tau: with it, L2/H1 rates are 2/1 (P1) and 3/2 (P2); without it (sge=0, the old behaviour)
they are about 1.1/1.0 (P1) and 1.0/1.0 (P2).
"""
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
HELPER = os.path.join(HERE, "adr_sgs_case.py")


def _env(**extra):
    env = dict(os.environ, OMP_NUM_THREADS="1")
    env.setdefault("FI_PROVIDER", "tcp")
    env.setdefault("OMPI_MCA_rmaps_base_oversubscribe", "1")
    env.setdefault("PRTE_MCA_rmaps_default_mapping_policy", ":oversubscribe")
    env["PYTHONPATH"] = os.pathsep.join(
        [q for q in sys.path if q] + [env.get("PYTHONPATH", "")]).strip(os.pathsep)
    env.update(extra)
    return env


def _run(ctx, ranks=1, cwd=None, **extra_env):
    cmd = [sys.executable, HELPER, ctx]
    if ranks > 1:
        cmd = [MPIEXEC, "-n", str(ranks)] + cmd
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=600, env=_env(**extra_env), cwd=str(cwd))
    assert out.returncode == 0, out.stdout[-2000:] + out.stderr[-2000:]
    line = [l for l in out.stdout.splitlines() if l.startswith("RESULT")][0]
    return {k: float(v) for k, v in (item.split("=") for item in line.split()[1:])}


def _mpiexec():
    beside = os.path.join(os.path.dirname(sys.executable), "mpiexec")
    return beside if os.path.exists(beside) else shutil.which("mpiexec")


def _direct_solver():
    try:
        import petsc4py
        petsc4py.init()
        from petsc4py import PETSc
    except ImportError:
        return None
    for pkg in ("mumps", "superlu_dist"):
        if PETSc.Sys.hasExternalPackage(pkg):
            return pkg
    return None


MPIEXEC = _mpiexec()
DIRECT_SOLVER = _direct_solver()

# errors on nnx = 5, 9, 17, 33, measured on 2026-11-12 (macOS arm64)
L2 = {1: [3.199e-02, 8.321e-03, 2.104e-03, 5.278e-04],
      2: [1.058e-03, 1.277e-04, 1.579e-05, 1.968e-06]}
H1 = {1: [3.369e-01, 1.701e-01, 8.525e-02, 4.265e-02],
      2: [3.608e-02, 8.988e-03, 2.243e-03, 5.604e-04]}
RATE_L2 = {1: 1.9, 2: 2.9}
RATE_H1 = {1: 0.95, 2: 1.9}


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    cwd = tmp_path_factory.mktemp("adrsgs")
    return {order: [_run("refinement={0} pOrder={1}".format(r, order), cwd=cwd) for r in range(4)]
            for order in (1, 2)}


@pytest.mark.parametrize("order", [1, 2])
def test_variable_diffusion_converges_at_the_optimal_rate(runs, order):
    l2 = [r["L2"] for r in runs[order]]
    h1 = [r["H1"] for r in runs[order]]
    rl2 = [np.log2(l2[i] / l2[i + 1]) for i in range(3)]
    rh1 = [np.log2(h1[i] / h1[i + 1]) for i in range(3)]
    assert min(rl2[1:]) >= RATE_L2[order], "L2 {0}, rates {1}".format(l2, rl2)
    assert min(rh1[1:]) >= RATE_H1[order], "H1 {0}, rates {1}".format(h1, rh1)
    np.testing.assert_allclose(l2, L2[order], rtol=2e-3)
    np.testing.assert_allclose(h1, H1[order], rtol=2e-3)


@pytest.mark.parametrize("order", [1, 2])
def test_the_old_residual_is_first_order(order, tmp_path):
    """sge=0 keeps the advection-only residual and tau: the L2 rate drops to ~1 (the defect this fixes)."""
    l2 = [_run("refinement={0} pOrder={1} sge=0".format(r, order), cwd=tmp_path)["L2"] for r in (1, 2, 3)]
    assert np.log2(l2[1] / l2[2]) < 1.3, l2


@pytest.mark.skipif(MPIEXEC is None, reason="mpiexec not available")
@pytest.mark.skipif(DIRECT_SOLVER is None, reason="PETSc has no parallel direct solver")
@pytest.mark.parametrize("order", [1, 2])
def test_two_ranks_match_serial(order, tmp_path):
    serial = _run("refinement=1 pOrder={0}".format(order), cwd=tmp_path)
    petsc = ("-mat_partitioning_type current -ksp_type preonly -pc_type lu "
             "-pc_factor_mat_solver_type " + str(DIRECT_SOLVER))
    par = _run("refinement=1 pOrder={0} usePetsc=True".format(order), ranks=2, cwd=tmp_path, PETSC_OPTIONS=petsc)
    assert par["ranks"] == 2
    np.testing.assert_allclose([par["L2"], par["H1"]], [serial["L2"], serial["H1"]], rtol=1e-8)
