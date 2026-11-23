#!/usr/bin/env python
"""The cut geometry of ADR (Unfitted.h CutGeometry) is built once and read by the residual and the Jacobian alike.

Before it, the lists of faces the ghost penalty and the SCIFEM terms are integrated on were filled only inside
calculateResidual, so a Jacobian assembled before the first residual (a fresh model object) silently left those
terms out. Here a model that has never seen a residual call assembles its Jacobian at the converged state and must
give the matrix the converged model has. The cases are the two-phase test=8 (circle, jump coefficients, SCIFEM on)
and the three-phase table1 (solid straddling the fluid interface: triply cut cells, composite fits, ghost penalty),
each P1 and P2.
"""
import os

import numpy as np
import pytest

from proteus.kernels import ADR
from proteus import Comm, Context, NumericalSolution, Profiling, default_s, defaults

comm = Comm.get()
Profiling.logLevel = 2
Profiling.verbose = False

CI_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "ci")

CASES = {
    "twophase_scifem": ("ladr_ss_2d_p", "ladr_ss_2d_c0p{0}_n",
                        "test=8.0 unstructured=False refinement=1 immersedSCIFEM_switch=1.0"),
    "threephase": ("ladr_3p_2d_p", "ladr_3p_2d_c0pk_n",
                   "paper=True test=2.0 nCells=10 unstructured=False pOrder={0}"),
}


def solved_model(opts, name, order):
    pname, nname, ctx = CASES[name]
    nname = nname.format(order)
    Context.contextOptionsString = ctx.format(order)
    try:
        p = defaults.load_physics(pname, CI_DIR)
        import sys
        sys.modules.pop(pname, None)
        n = defaults.load_numerics(nname, CI_DIR)
    finally:
        Context.contextOptionsString = None
    so = defaults.System_base()
    so.name = p.name
    so.pnList = [(pname, nname)]
    so.sList = [default_s]
    for attr in ("tnList", "systemStepControllerType", "systemStepExact", "archiveFlag"):
        if hasattr(n, attr):
            setattr(so, attr, getattr(n, attr))
    ns = NumericalSolution.NS_base(so, [p], [n], so.sList, opts)
    ns.calculateSolution("cutgeom_{0}_p{1}".format(name, order))
    return ns.modelList[0].levelModelList[-1]


@pytest.mark.parametrize("order", [1, 2])
@pytest.mark.parametrize("name", sorted(CASES))
def test_jacobian_before_any_residual_equals_the_converged_jacobian(opts, name, order):
    lm = solved_model(opts, name, order)
    u = lm.u[0].dof.copy()
    r = np.zeros_like(u)
    lm.getResidual(u.copy(), r)
    lm.getJacobian(lm.jacobian)
    expected = np.array(lm.nzval).copy()
    assert np.abs(expected).max() > 0.0
    # a C++ model object that has never run a residual
    fresh = ADR.cADR_base(lm.nSpace_global, lm.nQuadraturePoints_element,
                          lm.u[0].femSpace.elementMaps.localFunctionSpace.dim,
                          lm.u[0].femSpace.referenceFiniteElement.localFunctionSpace.dim,
                          lm.testSpace[0].referenceFiniteElement.localFunctionSpace.dim,
                          lm.nElementBoundaryQuadraturePoints_elementBoundary, 0)
    lm.adr = fresh
    lm.setUnknowns(u.copy())
    lm.recompute_ifem_geometry = False
    lm.getJacobian(lm.jacobian)
    np.testing.assert_allclose(np.array(lm.nzval), expected, rtol=0.0, atol=1e-12 * np.abs(expected).max())
