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
    l2 = [_run("refinement={0} pOrder=1 cut=\"plane_nogp\"".format(r), cwd=tmp_path)["L2"] for r in range(3)]
    rates = [np.log2(l2[i] / l2[i + 1]) for i in range(2)]
    assert all(r >= 1.9 for r in rates), "L2 errors {0}, rates {1}".format(l2, rates)
    np.testing.assert_allclose(l2, PLANE_L2, rtol=1e-5)


PLANE_GP_L2 = [4.01164436024943061e-02, 9.51733028228005963e-03, 2.45584691996369574e-03]


def test_p1_plane_cut_with_ghost_penalty_converges_at_rate_two(tmp_path):
    """The ghost penalty on triangular faces (default gamma 0.1), P1."""
    l2 = [_run("refinement={0} pOrder=1 cut=\"plane\"".format(r), cwd=tmp_path)["L2"] for r in range(3)]
    rates = [np.log2(l2[i] / l2[i + 1]) for i in range(2)]
    assert all(r >= 1.9 for r in rates), "L2 errors {0}, rates {1}".format(l2, rates)
    np.testing.assert_allclose(l2, PLANE_GP_L2, rtol=1e-5)


SPHERE_L2 = [3.79973288609295468e-02, 8.63172377002737953e-03, 2.23261167258554305e-03, 5.62128889801431710e-04]


def test_p1_sphere_embedded_boundary_converges_at_rate_two(tmp_path):
    """Curved solid (r = 0.3 at the cube centre), P1, ghost penalty on: nnx = 3, 5, 9, 17."""
    l2 = [_run("refinement={0} pOrder=1 cut=\"embedded\"".format(r), cwd=tmp_path)["L2"] for r in range(4)]
    rates = [np.log2(l2[i] / l2[i + 1]) for i in range(3)]
    assert all(r >= 1.9 for r in rates), "L2 errors {0}, rates {1}".format(l2, rates)
    np.testing.assert_allclose(l2, SPHERE_L2, rtol=1e-5)


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
    serial = _run("refinement=1 pOrder=1 cut=\"plane_nogp\"", cwd=tmp_path)
    petsc = ("-mat_partitioning_type current -ksp_type preonly -pc_type lu "
             "-pc_factor_mat_solver_type " + str(DIRECT_SOLVER))
    par = _run("refinement=1 pOrder=1 cut=\"plane_nogp\" usePetsc=True", ranks=2, cwd=tmp_path, PETSC_OPTIONS=petsc)
    assert par["ranks"] == 2
    np.testing.assert_allclose([par["L2"], par["Linfty"]], [serial["L2"], serial["Linfty"]], rtol=1e-8)


@pytest.mark.skipif(MPIEXEC is None, reason="mpiexec not available")
@pytest.mark.skipif(DIRECT_SOLVER is None, reason="PETSc has no parallel direct solver")
def test_plane_cut_ghost_penalty_two_ranks_match_serial(tmp_path):
    serial = _run("refinement=1 pOrder=1 cut=\"plane\"", cwd=tmp_path)
    petsc = ("-mat_partitioning_type current -ksp_type preonly -pc_type lu "
             "-pc_factor_mat_solver_type " + str(DIRECT_SOLVER))
    par = _run("refinement=1 pOrder=1 cut=\"plane\" usePetsc=True", ranks=2, cwd=tmp_path, PETSC_OPTIONS=petsc)
    assert par["ranks"] == 2
    np.testing.assert_allclose([par["L2"], par["Linfty"]], [serial["L2"], serial["Linfty"]], rtol=1e-8)


IFEM_PLANE_L2 = [3.50335078291571511e-02, 8.62255956802715669e-03, 2.55608379622826150e-03]


def test_p1_immersed_plane_two_material_solution(tmp_path):
    """C44 phase 2, P1: the two-sided IFEM basis across a plane that is not aligned with the mesh (mua = 1,
    mub = 10, u continuous, mu du/dn continuous). The P1 IFE space is nonconforming across faces of cut elements
    and ADR has no 3D SCIFEM face terms yet, so the rate falls below 2 as the mesh is refined; this pins the
    values (and the first-pair rate), it does not assert the optimal rate. The 2D code does the same on a
    generic-slope plane (test/IFEM test=13.0 without SCIFEM)."""
    l2 = [_run("refinement={0} pOrder=1 cut=\"ifem_plane\" mub=10.0".format(r), cwd=tmp_path)["L2"] for r in range(3)]
    assert np.log2(l2[0] / l2[1]) >= 1.9, l2
    assert l2[2] < l2[1] < l2[0], l2
    np.testing.assert_allclose(l2, IFEM_PLANE_L2, rtol=1e-5)


def test_p1_immersed_equal_coefficients_reproduce_a_linear_solution(tmp_path):
    """With mua = mub the IFE basis is the ordinary linear one and the piecewise-linear solution is exact."""
    out = _run("refinement=1 pOrder=1 cut=\"ifem_pwl\" mua=1.0 mub=1.0", cwd=tmp_path)
    assert out["L2"] < 1e-12 and out["Linfty"] < 1e-12, out


IFEM_SPHERE_L2 = [1.63184160987282501e-02, 6.89258384492486692e-03, 2.40914151097162705e-03, 6.12520897829881808e-04]


def test_p1_immersed_sphere_two_material_solution(tmp_path):
    """Curved interface (r = 0.3, mua = 1, mub = 10), radial solution: the P1 IFE solution converges, at rate
    1.98 between the two finest meshes (nnx = 9, 17)."""
    l2 = [_run("refinement={0} pOrder=1 cut=\"ifem_sphere\" mub=10.0".format(r), cwd=tmp_path)["L2"] for r in range(4)]
    assert np.log2(l2[2] / l2[3]) >= 1.9, l2
    np.testing.assert_allclose(l2, IFEM_SPHERE_L2, rtol=1e-5)


def test_p1_immersed_flux_jump_lift_and_solution_jump(tmp_path):
    """The prescribed flux jump ([mu du/dn] = 2 through the Dirac load and the unit flux-jump lift) and a
    constant solution jump ([u] = u(+) - u(-) = 0.7, through the nodal lift) give the errors of the same problem
    without jump data: a constant solution jump is exactly in the space, the flux jump to within the lift's
    consistency (about 2% of the error)."""
    base = _run("refinement=1 pOrder=1 cut=\"ifem_plane\" mub=10.0", cwd=tmp_path)["L2"]
    flux = _run("refinement=1 pOrder=1 cut=\"ifem_plane\" mub=10.0 fluxJump=2.0", cwd=tmp_path)["L2"]
    sol = _run("refinement=1 pOrder=1 cut=\"ifem_plane\" mub=10.0 solJump=0.7", cwd=tmp_path)["L2"]
    assert abs(sol - base) < 1e-9 * base, (sol, base)
    assert abs(flux - base) < 0.05 * base, (flux, base)


@pytest.mark.skipif(MPIEXEC is None, reason="mpiexec not available")
@pytest.mark.skipif(DIRECT_SOLVER is None, reason="PETSc has no parallel direct solver")
def test_p1_immersed_plane_two_ranks_match_serial(tmp_path):
    ctx = "refinement=1 pOrder=1 cut=\"ifem_plane\" mub=10.0 fluxJump=2.0"
    serial = _run(ctx, cwd=tmp_path)
    petsc = ("-mat_partitioning_type current -ksp_type preonly -pc_type lu "
             "-pc_factor_mat_solver_type " + str(DIRECT_SOLVER))
    par = _run(ctx + " usePetsc=True", ranks=2, cwd=tmp_path, PETSC_OPTIONS=petsc)
    assert par["ranks"] == 2
    np.testing.assert_allclose([par["L2"], par["Linfty"]], [serial["L2"], serial["Linfty"]], rtol=1e-8)


SCIFEM = "scifem=1.0 scifemPenalty=10.0"
IFEM_SCIFEM_PLANE_L2 = {10.0: [3.66983450782109102e-02, 9.03573535578436164e-03, 2.34186492160958545e-03, 6.17740313962773400e-04],
                        1000.0: [3.37869233267889812e-02, 8.79494815923208099e-03, 2.63388253074005926e-03, 1.17265913688247752e-03]}
IFEM_SCIFEM_SPHERE_L2 = {10.0: [1.61503769780517101e-02, 8.47620060428019503e-03, 2.59021131275304203e-03, 6.78877414369735070e-04],
                         1000.0: [8.19725414710156634e-03, 8.75073991203536954e-03, 2.64772024283712682e-03, 6.88329959911526245e-04]}


@pytest.mark.parametrize("mub", [10.0, 1000.0])
def test_p1_immersed_scifem_plane(mub, tmp_path):
    """SCIFEM consistency terms + interior penalty (gamma = 10) on the cut triangular faces, plane n.x = 0.45.
    mub = 10 reaches rate 2 (2.02, 1.95, 1.92); mub = 1000, which does not converge without the face terms,
    now converges (rates 1.94, 1.74, 1.17: it flattens at the finest pair, where the plane leaves the cube and
    the exterior cut faces carry no consistency term, as in 2D)."""
    l2 = [_run("refinement={0} pOrder=1 cut=\"ifem_plane\" mub={1} {2}".format(r, mub, SCIFEM), cwd=tmp_path)["L2"] for r in range(4)]
    rates = [np.log2(l2[i] / l2[i + 1]) for i in range(3)]
    if mub == 10.0:
        assert all(r >= 1.9 for r in rates), (l2, rates)
    else:
        assert l2[3] < l2[2] < l2[1] < l2[0] and rates[0] >= 1.9 and rates[1] >= 1.7, (l2, rates)
    np.testing.assert_allclose(l2, IFEM_SCIFEM_PLANE_L2[mub], rtol=1e-5)


@pytest.mark.parametrize("mub", [10.0, 1000.0])
def test_p1_immersed_scifem_sphere(mub, tmp_path):
    """Sphere r = 0.3 (the interface does not touch the boundary): rate 1.9-2 on the finest pair at both contrasts."""
    l2 = [_run("refinement={0} pOrder=1 cut=\"ifem_sphere\" mub={1} {2}".format(r, mub, SCIFEM), cwd=tmp_path)["L2"] for r in range(4)]
    assert np.log2(l2[2] / l2[3]) >= 1.9, l2
    np.testing.assert_allclose(l2, IFEM_SCIFEM_SPHERE_L2[mub], rtol=1e-5)


@pytest.mark.skipif(MPIEXEC is None, reason="mpiexec not available")
@pytest.mark.skipif(DIRECT_SOLVER is None, reason="PETSc has no parallel direct solver")
def test_p1_immersed_scifem_two_ranks_match_serial(tmp_path):
    """The face terms need both elements of every cut face; two layers of overlap (the case sets them) give them."""
    ctx = "refinement=1 pOrder=1 cut=\"ifem_plane\" mub=10.0 fluxJump=2.0 " + SCIFEM
    serial = _run(ctx, cwd=tmp_path)
    petsc = ("-mat_partitioning_type current -ksp_type preonly -pc_type lu "
             "-pc_factor_mat_solver_type " + str(DIRECT_SOLVER))
    par = _run(ctx + " usePetsc=True", ranks=2, cwd=tmp_path, PETSC_OPTIONS=petsc)
    assert par["ranks"] == 2
    np.testing.assert_allclose([par["L2"], par["Linfty"]], [serial["L2"], serial["Linfty"]], rtol=1e-8)


CUT_PHASE = {"robin": 1, "pg": 2, "both": 3}


def _setup(cut, extra=""):
    from proteus import runtime
    runtime.start()
    Context.contextOptionsString = "refinement=0 cut=\"{0}\" {1}".format(cut, extra)
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


def test_p2_immersed_stops_at_setup_in_3d(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    so, p, n, opts = _setup("ifem_plane", "pOrder=2")
    with pytest.raises(NotImplementedError, match=r"2D only.*immersedBoundary with P2.*C44 phase 2"):
        NumericalSolution.NS_base(so, [p], [n], so.sList, opts)


def test_kernel_refuses_p2_immersed_in_3d(tmp_path, monkeypatch):
    from proteus import ADR
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(ADR, "_requireSupportedDimension", lambda *a: None)
    so, p, n, opts = _setup("ifem_plane", "pOrder=2")
    with pytest.raises(RuntimeError, match=r"2D only.*immersedBoundary with P2.*C44 phase 2"):
        ns = NumericalSolution.NS_base(so, [p], [n], so.sList, opts)
        ns.calculateSolution("adr3d_cut")


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
