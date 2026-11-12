#!/usr/bin/env python
"""ADR in 3D without cuts (C44 phase 0), and a clear error for every cut-cell path.

Before this, ADR could not set up a 3D problem at all (it told P1 from P2 by a triangle's dof count and
stopped at setup), and its 2D-only cut-cell code would have run on garbage in 3D, because assert() is
compiled out (-DNDEBUG).

The case is test/ci/ladr_ss_3d_p.py on structured tetrahedra (nnx = 3, 5, 9; no mesher): a manufactured
solution with Dirichlet data on the six faces.

* "smooth": diffusion with a variable a(x). L2 rates 2 (P1) and 3 (P2). It has no advection because
  ADR's subgrid-error term is inconsistent with diffusion plus advection (see the p module).
* "quadratic": advection with constant a. Lies in P2, so P2 reproduces it to solver tolerance.
* two ranks give the serial errors, and a case that asks for any cut-cell path stops with a message that
  names the feature and the C44 phase.

Rates are checked against the finest-pair error ratio, with the coarsest-mesh errors pinned too.
"""
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest

from proteus import Context, NumericalSolution, default_s, defaults

HERE = os.path.dirname(os.path.abspath(__file__))
CI_DIR = os.path.join(HERE, os.pardir, "ci")
HELPER = os.path.join(HERE, "adr3d_case.py")


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

# L2 error on nnx = 3, 5, 9, measured on 2026-11-12 (Linux/macOS agree to ~1e-10).
SMOOTH_L2 = {1: [4.3409021547330694e-02, 1.0907883251453155e-02, 2.7306060032508e-03],
             2: [1.691565412850317e-03, 2.1456241309774965e-04, 2.6928183055828742e-05]}
RATE_MIN = {1: 1.95, 2: 2.9}


@pytest.fixture(scope="module")
def smooth_runs(tmp_path_factory):
    cwd = tmp_path_factory.mktemp("adr3d")
    return {order: [_run("refinement={0} pOrder={1}".format(r, order), cwd=cwd)["L2"] for r in range(3)]
            for order in (1, 2)}


@pytest.mark.parametrize("order", [1, 2])
def test_plain_3d_adr_converges_at_the_optimal_rate(smooth_runs, order):
    l2 = smooth_runs[order]
    rates = [np.log2(l2[i] / l2[i + 1]) for i in range(2)]
    assert all(r >= RATE_MIN[order] for r in rates), "L2 errors {0}, rates {1}".format(l2, rates)
    np.testing.assert_allclose(l2, SMOOTH_L2[order], rtol=1e-5)


# L2 error with a planar embedded solid (n.x = 0.37, Nitsche terms, no ghost penalty), P1, nnx = 3, 5, 9
PLANE_L2 = [3.90776936002934089e-02, 9.81589909475118899e-03, 2.54488780303339327e-03]


def test_p1_plane_cut_embedded_boundary_converges_at_rate_two(tmp_path):
    """C44 phase 1, first piece: 3D CutFEM volume and Nitsche terms on P1 tetrahedra (fit degree 3)."""
    l2 = [_run("refinement={0} pOrder=1 cut=\"plane\"".format(r), cwd=tmp_path)["L2"] for r in range(3)]
    rates = [np.log2(l2[i] / l2[i + 1]) for i in range(2)]
    assert all(r >= 1.9 for r in rates), "L2 errors {0}, rates {1}".format(l2, rates)
    np.testing.assert_allclose(l2, PLANE_L2, rtol=1e-5)


def test_p2_reproduces_a_quadratic_with_advection(tmp_path):
    out = _run("refinement=1 pOrder=2 exact=\"quadratic\"", cwd=tmp_path)
    assert out["L2"] < 1e-10 and out["Linfty"] < 1e-10, out


@pytest.mark.skipif(MPIEXEC is None, reason="mpiexec not available")
@pytest.mark.skipif(DIRECT_SOLVER is None, reason="PETSc has no parallel direct solver")
@pytest.mark.parametrize("order", [1, 2])
def test_two_ranks_match_serial(order, tmp_path):
    serial = _run("refinement=1 pOrder={0}".format(order), cwd=tmp_path)
    petsc = ("-mat_partitioning_type current -ksp_type preonly -pc_type lu "
             "-pc_factor_mat_solver_type " + str(DIRECT_SOLVER))
    par = _run("refinement=1 pOrder={0} usePetsc=True".format(order), ranks=2, cwd=tmp_path, PETSC_OPTIONS=petsc)
    assert par["ranks"] == 2
    np.testing.assert_allclose([par["L2"], par["Linfty"]], [serial["L2"], serial["Linfty"]], rtol=1e-8)


@pytest.mark.skipif(MPIEXEC is None, reason="mpiexec not available")
@pytest.mark.skipif(DIRECT_SOLVER is None, reason="PETSc has no parallel direct solver")
def test_plane_cut_two_ranks_match_serial(tmp_path):
    serial = _run("refinement=1 pOrder=1 cut=\"plane\"", cwd=tmp_path)
    petsc = ("-mat_partitioning_type current -ksp_type preonly -pc_type lu "
             "-pc_factor_mat_solver_type " + str(DIRECT_SOLVER))
    par = _run("refinement=1 pOrder=1 cut=\"plane\" usePetsc=True", ranks=2, cwd=tmp_path, PETSC_OPTIONS=petsc)
    assert par["ranks"] == 2
    np.testing.assert_allclose([par["L2"], par["Linfty"]], [serial["L2"], serial["Linfty"]], rtol=1e-8)


CUT_PHASE = {"embedded": 1, "robin": 1, "immersed": 2, "scifem": 2, "pg": 2, "both": 3}


def _setup(cut):
    from proteus import runtime
    runtime.start()
    Context.contextOptionsString = "refinement=0 cut=\"{0}\"".format(cut)
    try:
        p = defaults.load_physics("ladr_ss_3d_p", CI_DIR)
        n = defaults.load_numerics("ladr_ss_3d_n", CI_DIR)
    finally:
        Context.contextOptionsString = None
    so = defaults.System_base()
    so.name = p.name
    so.pnList = [("ladr_ss_3d_p", "ladr_ss_3d_n")]
    so.sList = [default_s]
    return so, p, n, runtime.RunOptions()


@pytest.mark.parametrize("cut", sorted(CUT_PHASE))
def test_cut_cell_features_stop_at_setup_in_3d(cut, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    so, p, n, opts = _setup(cut)
    with pytest.raises(NotImplementedError, match=r"2D only.*C44 phase {0}".format(CUT_PHASE[cut])):
        NumericalSolution.NS_base(so, [p], [n], so.sList, opts)


@pytest.mark.parametrize("cut", sorted(CUT_PHASE))
def test_kernel_refuses_cut_cell_features_in_3d(cut, tmp_path, monkeypatch):
    """The C++ kernel makes the same check, for a caller that skips the Python one."""
    from proteus import ADR
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(ADR, "_requireSupportedDimension", lambda *a: None)
    so, p, n, opts = _setup(cut)
    with pytest.raises(RuntimeError, match=r"2D only.*C44 phase {0}".format(CUT_PHASE[cut])):
        ns = NumericalSolution.NS_base(so, [p], [n], so.sList, opts)
        ns.calculateSolution("adr3d_cut")
