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
