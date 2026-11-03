#!/usr/bin/env python
"""
Regression tests for the three-phase hybrid CutFEM-IFEM driver, test/ci/ladr_3p_2d_p.py.

test_ifem.py pins the two-phase ladr_ss cases. This module pins the cases where a solid
(embeddedBoundary, CutFEM/Nitsche) straddles the fluid-fluid interface (immersedBoundary,
IFEM), so some cells are cut by both: the triply-cut logic, the jump lifts, the Robin
condition on Gamma_s and the P2 orientation guard in equivalent_polynomials.h
(p2_cell_flipped) only run here.

Every case is the coarsest mesh of the corresponding manuscript table, with the manuscript's
settings (paper=True in the driver; qOrder 6 for p=1, 8 for p=2):

    table1      test=2.0, mu_b/mu_a=5, nCells=10             (Table 1, structured)
    table2      test=2.1, solution jump 0.2, refinement=1    (Table 2)
    table3      test=2.1, flux jump 0.2, refinement=1        (Table 3)
    table4      test=3.0, Robin on Gamma_s, refinement=1     (Table 4)
    exact       test=1.0, piecewise-linear solution, refinement=1

plus table1 with paper=False, which keeps the earlier driver defaults (masked-DOF pinning on,
no SCIFEM terms) under test.

To regenerate the saved values after an intentional change, from the repository root:

    IFEM_SAVE_COMPARISON=1 pytest test/IFEM/test_ifem_3p.py
"""
import os
import shutil
import subprocess
import sys
import warnings

import numpy as np
import pytest

from proteus import ADR, Comm, Context, NumericalSolution, Profiling, default_s, defaults
from proteus.iproteus import opts

comm = Comm.get()
Profiling.logLevel = 2
Profiling.verbose = False

CI_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "ci")
COMPARISON_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "comparison_files")

CASES = {
    "table1":        "paper=True test=2.0 nCells=10",
    "table2":        "paper=True test=2.1 test21solutionJump=0.2 test21fluxJump=0.0 refinement=1",
    "table3":        "paper=True test=2.1 test21solutionJump=0.0 test21fluxJump=0.2 refinement=1",
    "table4":        "paper=True test=3.0 refinement=1",
    "exact":         "paper=True test=1.0 refinement=1",
    "table1_legacy": "paper=False test=2.0 nCells=10",
}

# Same tolerances as test_ifem.py: a change in the formulation moves these by percent.
RTOL = 1.0e-8
ATOL = 1.0e-10

SAVE = os.environ.get("IFEM_SAVE_COMPARISON", "") not in ("", "0")


def case_id(name, order):
    return "{0}_p{1}".format(name, order)


def run_case(name, order):
    """Solve one case; return array([L2, Linfty])."""
    Context.contextOptionsString = "{0} pOrder={1} unstructured=False".format(CASES[name], order)
    try:
        p = defaults.load_physics("ladr_3p_2d_p", CI_DIR)
        # The n module's own "from ladr_3p_2d_p import *" is an ordinary import, cached in
        # sys.modules with the first case's options; drop it so it re-executes with this case's.
        sys.modules.pop("ladr_3p_2d_p", None)
        n = defaults.load_numerics("ladr_3p_2d_c0pk_n", CI_DIR)
    finally:
        Context.contextOptionsString = None

    so = defaults.System_base()
    so.name = p.name
    so.pnList = [("ladr_3p_2d_p", "ladr_3p_2d_c0pk_n")]
    so.sList = [default_s]
    for attr in ("tnList", "systemStepControllerType", "systemStepExact", "archiveFlag"):
        if hasattr(n, attr):
            setattr(so, attr, getattr(n, attr))

    ns = NumericalSolution.NS_base(so, [p], [n], so.sList, opts)
    ns.calculateSolution("ifem3p_" + case_id(name, order))

    lm = ns.modelList[0].levelModelList[-1]
    return np.array([float(lm.L2_error[0] ** 0.5), float(lm.Linfty_error[0])])


PARAMS = [(c, o) for c in CASES for o in (1, 2)]


@pytest.mark.parametrize("name,order", PARAMS, ids=[case_id(*p) for p in PARAMS])
def test_ifem_3p(name, order):
    """L2/Linfty errors must match the saved reference."""
    cid = case_id(name, order)
    actual = run_case(name, order)
    assert np.all(np.isfinite(actual)), "{0}: non-finite error {1}".format(cid, actual)

    expected_path = os.path.join(COMPARISON_DIR, "ifem3p_" + cid + ".csv")
    if SAVE:
        actual.tofile(expected_path, sep=",")
        warnings.warn("IFEM_SAVE_COMPARISON: wrote {0} (L2={1:.10e} Linfty={2:.10e})"
                      .format(os.path.basename(expected_path), actual[0], actual[1]))
    if not os.path.exists(expected_path):
        raise AssertionError(
            "no saved reference at {0}.\nRun from the repository root:\n"
            "    IFEM_SAVE_COMPARISON=1 pytest test/IFEM/test_ifem_3p.py".format(expected_path))

    expected = np.fromfile(expected_path, sep=",")
    np.testing.assert_allclose(
        actual, expected, rtol=RTOL, atol=ATOL,
        err_msg=("{0}: error moved, which means the three-phase formulation changed.\n"
                 "  expected L2={1:.10e} Linfty={2:.10e}\n  actual   L2={3:.10e} Linfty={4:.10e}")
        .format(cid, expected[0], expected[1], actual[0], actual[1]))


def _coefficients():
    return ADR.Coefficients(aOfX={0: lambda x: np.eye(2)}, fOfX={0: lambda x: 0.0})


def test_triply_cut_options_default():
    """Without arguments the triply-cut choices are the pre-option behaviour."""
    c = _coefficients()
    assert (c.triplyCut_pinMaskedDofs, c.triplyCut_edgeFluidSide, c.triplyCut_edgeBoundaryOnly) \
        == (True, False, False)


@pytest.mark.parametrize("env,value,attr,expected", [
    ("ADR_DISABLE_TC", "1", "triplyCut_pinMaskedDofs", False),
    ("ADR_TC_EDGE_FLUID", "1", "triplyCut_edgeFluidSide", True),
    ("ADR_TC_EDGE_BOUNDARY_ONLY", "1", "triplyCut_edgeBoundaryOnly", True),
])
def test_triply_cut_environment_variables_still_work(monkeypatch, env, value, attr, expected):
    """The old environment variables override the arguments and say they are deprecated."""
    monkeypatch.setenv(env, value)
    with pytest.warns(DeprecationWarning, match=attr):
        c = _coefficients()
    assert getattr(c, attr) == expected


# --- the same cases on two ranks -----------------------------------------------------------

MPI_HELPER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ifem_3p_partition_mpi.py")


def _mpiexec():
    beside = os.path.join(os.path.dirname(sys.executable), "mpiexec")
    return beside if os.path.exists(beside) else shutil.which("mpiexec")


MPIEXEC = _mpiexec()


def _direct_solver():
    """A PETSc package that factors a distributed matrix, or None."""
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


DIRECT_SOLVER = _direct_solver()


def _run_ranks(ranks, ctx, cwd, **extra_env):
    env = dict(os.environ, OMP_NUM_THREADS="1",
               PETSC_OPTIONS=("-mat_partitioning_type current -ksp_type preonly -pc_type lu "
                              "-pc_factor_mat_solver_type " + str(DIRECT_SOLVER)))
    env.setdefault("FI_PROVIDER", "tcp")
    env.setdefault("OMPI_MCA_rmaps_base_oversubscribe", "1")
    env.setdefault("PRTE_MCA_rmaps_default_mapping_policy", ":oversubscribe")
    env["PYTHONPATH"] = os.pathsep.join(
        [q for q in sys.path if q] + [env.get("PYTHONPATH", "")]).strip(os.pathsep)
    env.update({k: str(v) for k, v in extra_env.items()})
    return subprocess.run([MPIEXEC, "-n", str(ranks), sys.executable, MPI_HELPER, ctx],
                          capture_output=True, text=True, timeout=600, env=env, cwd=str(cwd))


PARALLEL_CASES = ["table1_p1", "table1_p2", "table2_p2", "table4_p2"]


@pytest.mark.skipif(MPIEXEC is None, reason="mpiexec not available")
@pytest.mark.skipif(DIRECT_SOLVER is None, reason="PETSc has no parallel direct solver")
@pytest.mark.parametrize("cid", PARALLEL_CASES)
def test_ifem_3p_two_ranks_match_serial(cid, tmp_path):
    """Two ranks give the serial pins.

    Three defects made them differ: the ghost-penalty faces were taken on owned faces only
    (full-overlap assembly needs every local face), the face terms need two layers of overlap
    (one dropped the faces whose far element sat on the other rank), and for P2 the element
    mapping read vertex coordinates and node diameters through node numbers where the kernel
    holds them by dof (NaN residual). Each moved these errors by 1e-3 to 1e-1 relative.
    """
    name, order = cid.rsplit("_p", 1)
    out = _run_ranks(2, "{0} pOrder={1}".format(CASES[name], order), tmp_path)
    assert out.returncode == 0, out.stdout[-2000:] + out.stderr[-2000:]
    line = [l for l in out.stdout.splitlines() if l.startswith("RESULT")][0]
    result = dict(item.split("=") for item in line.split()[1:])
    expected = np.fromfile(os.path.join(COMPARISON_DIR, "ifem3p_" + cid + ".csv"), sep=",")
    np.testing.assert_allclose([float(result["L2"]), float(result["Linfty"])], expected,
                               rtol=RTOL, atol=ATOL, err_msg=cid + " on 2 ranks")


@pytest.mark.skipif(MPIEXEC is None, reason="mpiexec not available")
@pytest.mark.skipif(DIRECT_SOLVER is None, reason="PETSc has no parallel direct solver")
def test_ifem_3p_one_layer_of_overlap_is_refused(tmp_path):
    """With one layer of overlap the face terms are silently wrong, so ADR refuses to run."""
    out = _run_ranks(2, CASES["table1"] + " pOrder=1", tmp_path, NLAYERS=1)
    assert out.returncode != 0
    assert "two layers of mesh overlap" in out.stdout + out.stderr
