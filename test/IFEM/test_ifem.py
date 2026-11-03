#!/usr/bin/env python
"""
Regression tests for the immersed finite element method (IFEM) path in ADR.

test_equivalent_polynomials.py exercises the equivalent-polynomial integration
rules on their own -- it checks that the H/ImH/D surrogates integrate correctly
over a cut simplex, but never solves a PDE.  This module covers the other half:
it runs the *full* solve for every analytical solution used by the convergence
study in test/ci/run_convergence.sh and pins the resulting errors.

Why the errors and not the solution vector: the observed convergence rates are
what the IFEM work is judged on, and a rate is a property of the error sequence.
Pinning the coarsest-mesh error therefore guards the rates directly, and unlike
a DOF vector it does not change meaning when mesh numbering or partitioning
changes.  The mesh is the structured one (unstructured=False) so the comparison
never depends on the triangle library's version-to-version output.

Cases: every test number ladr_ss_2d_p.py defines, for both P1 and P2.  The six
with a diffusion jump across the interface (mua != mub) are additionally run
with the SCIFEM interface-consistency terms enabled; for the rest those terms
are structurally zero, since va == vb makes the jump [[u]] vanish identically.

To regenerate the saved values after an intentional change, from the
repository root (the paths below are relative to it):

    IFEM_SAVE_COMPARISON=1 pytest test/IFEM/test_ifem.py

The cases still run and pass in that mode -- each value is read back after being
written -- and every file written is listed in pytest's warnings summary.

Inspect the resulting diff before committing it -- a change here means the
convergence behaviour moved.

Known issues
------------
Some cases pin behaviour that is known to be wrong; they stay pinned rather than
skipped so movement is still noticed, and KNOWN_ISSUES says which and why.
"""
import os
import warnings

import numpy as np
import pytest

from proteus.iproteus import opts
from proteus import Comm, Context, NumericalSolution, Profiling, default_s, defaults

comm = Comm.get()
Profiling.logLevel = 2
Profiling.verbose = False

# The p/n modules under test are the same ones the convergence script drives;
# importing rather than copying them keeps the two from drifting apart.
CI_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "ci")
COMPARISON_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "comparison_files")

# Coarsest mesh of the convergence sweep: nnx = 4*2**refinement + 1 = 9.
REFINEMENT = 1

ALL_TESTS = ["1.0", "2.0", "2.1", "3.0", "4.0", "4.1", "5.0", "6.0",
             "7.0", "8.0", "9.0", "10.0", "11.0", "12.0", "13.0"]

# mua != mub in ladr_ss_2d_p.py, so the SCIFEM terms are not identically zero.
JUMP_TESTS = ["2.0", "2.1", "8.0", "11.0", "12.0", "13.0"]

ORDERS = [1, 2]

# Cases whose saved value records a defect rather than correct behaviour.  Keyed
# by case id; surfaced in the failure message so nobody "fixes" a reference file
# without realising what it encodes.
KNOWN_ISSUES = {
    "test11.0_p2": (
        "KNOWN DEFECT: P1 reproduces this piecewise-linear solution exactly "
        "(1.08e-14) but P2, whose space contains P1's, gives 6.37e-03 unless "
        "SCIFEM is enabled."),
    "test12.0_p2": (
        "Converges erratically at fine meshes, reaching rate 0.77 at refinement "
        "5 and rate -2.22 at refinement 6 with SCIFEM on."),
    "test13.0_p1": (
        "test=13.0 is not fully working yet: convergence is sub-optimal and the "
        "saved values pin current behaviour rather than assert correctness."),
    "test13.0_p1_scifem": "See test13.0_p1: test=13.0 is not fully working yet.",
    "test13.0_p2": "See test13.0_p1: test=13.0 is not fully working yet.",
    "test13.0_p2_scifem": "See test13.0_p1: test=13.0 is not fully working yet.",
}

# Tight enough that any change to the formulation shows up (a real change moves
# these by percent, not by parts per billion), loose enough to survive a
# different BLAS/compiler.  atol dominates only for test=11.0, whose solution
# the FE space reproduces exactly and whose "error" is pure roundoff; there the
# meaningful assertion is precisely that it stays at roundoff.
RTOL = 1.0e-8
ATOL = 1.0e-10

SAVE = os.environ.get("IFEM_SAVE_COMPARISON", "") not in ("", "0")


def case_id(test, order, scifem):
    return "test{0}_p{1}{2}".format(test, order, "_scifem" if scifem else "")


def run_case(test, order, scifem):
    """Solve one case on the coarsest mesh; return array([L2, Linfty])."""
    Context.contextOptionsString = (
        "test={0} unstructured=False refinement={1} "
        "immersedSCIFEM_switch={2}".format(test, REFINEMENT, 1.0 if scifem else 0.0))

    p = defaults.load_physics("ladr_ss_2d_p", CI_DIR)
    n = defaults.load_numerics("ladr_ss_2d_c0p{0}_n".format(order), CI_DIR)

    so = defaults.System_base()
    so.name = p.name
    so.pnList = [("ladr_ss_2d_p", "ladr_ss_2d_c0p{0}_n".format(order))]
    so.sList = [default_s]
    for attr in ("tnList", "systemStepControllerType", "systemStepExact", "archiveFlag"):
        if hasattr(n, attr):
            setattr(so, attr, getattr(n, attr))

    ns = NumericalSolution.NS_base(so, [p], [n], so.sList, opts)
    ns.calculateSolution(case_id(test, order, scifem))

    lm = ns.modelList[0].levelModelList[-1]
    # both are length-1 arrays; L2_error holds the *squared* norm
    return np.array([float(lm.L2_error[0] ** 0.5), float(lm.Linfty_error[0])])


PARAMS = ([(t, o, False) for t in ALL_TESTS for o in ORDERS] +
          [(t, o, True) for t in JUMP_TESTS for o in ORDERS])


@pytest.mark.parametrize("test,order,scifem", PARAMS,
                         ids=[case_id(*p) for p in PARAMS])
def test_ifem(test, order, scifem):
    """Coarsest-mesh L2/Linfty errors must match the saved reference."""
    actual = run_case(test, order, scifem)

    assert np.all(np.isfinite(actual)), \
        "{0}: non-finite error {1}".format(case_id(test, order, scifem), actual)

    expected_path = os.path.join(COMPARISON_DIR,
                                 "ifem_" + case_id(test, order, scifem) + ".csv")
    if SAVE:
        actual.tofile(expected_path, sep=",")
        # Read straight back rather than skipping: the case still has to pass on
        # its own terms, and this catches any precision lost in the round trip.
        warnings.warn("IFEM_SAVE_COMPARISON: wrote {0} (L2={1:.10e} Linfty={2:.10e})"
                      .format(os.path.basename(expected_path), actual[0], actual[1]))

    if not os.path.exists(expected_path):
        raise AssertionError(
            "no saved reference at {0}.\nRun from the repository root:\n"
            "    IFEM_SAVE_COMPARISON=1 pytest test/IFEM/test_ifem.py"
            .format(expected_path))

    cid = case_id(test, order, scifem)
    note = KNOWN_ISSUES.get(cid)

    expected = np.fromfile(expected_path, sep=",")
    np.testing.assert_allclose(
        actual, expected, rtol=RTOL, atol=ATOL,
        err_msg=("{0}: coarse-mesh error moved, which means the convergence "
                 "behaviour changed.\n  expected L2={1:.10e} Linfty={2:.10e}"
                 "\n  actual   L2={3:.10e} Linfty={4:.10e}{5}").format(
                     cid, expected[0], expected[1], actual[0], actual[1],
                     "\n\n  " + note if note else ""))
