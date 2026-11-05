"""Miscible displacement: p1-nc Darcy flow coupled to C0 P1 advective-dispersive
transport (md_so.py's two models), on a regular triangular mesh for a short time.

The transport model's coefficients (SubsurfaceTransportCoefficients.
GroundwaterTransportCoefficients) evaluate through a Cython function stored as a
class attribute and called as self.f(...). Under Cython 3 that call bound
``self`` as a 13th argument and the deck stopped at the first evaluate with a
TypeError (C47).
"""
import os
import sys
import types

import numpy as np

from proteus.iproteus import opts
from proteus import Context, NumericalSolution, default_s, default_so

HERE = os.path.dirname(os.path.abspath(__file__))
# bare-name deck modules; 1.9.x has no conftest that drops them after a test
DECK_MODULES = ("miscible_displacement", "md_flow_p", "md_flow_ncp1_n",
                "md_trans_p", "md_trans_c0p1_n")


def _so_module(name, pnList, tnList):
    so = types.ModuleType(name + "_so")
    vars(so).update({k: v for k, v in vars(default_so).items() if not k.startswith("__")})
    so.name = name
    so.pnList = pnList
    so.tnList = tnList
    so.sList = [default_s for _ in pnList]
    so.systemStepControllerType = default_so.Sequential_MinAdaptiveModelStep
    so.needEBQ_GLOBAL = True
    so.needEBQ = True
    return so


def test_miscible_displacement(tmp_path):
    cwd = os.getcwd()
    saved_context = Context.contextOptionsString
    os.chdir(tmp_path)
    Context.contextOptionsString = ("unstructured_mesh=False refinement_level=8 "
                                    "ndays=500.0 nDTout=2")
    sys.path.insert(0, HERE)
    for name in DECK_MODULES:
        sys.modules.pop(name, None)
    try:
        import md_flow_p, md_flow_ncp1_n, md_trans_p, md_trans_c0p1_n
        pList = [md_flow_p, md_trans_p]
        nList = [md_flow_ncp1_n, md_trans_c0p1_n]
        so = _so_module("md_test", [("md_flow_p", "md_flow_ncp1_n"),
                                    ("md_trans_p", "md_trans_c0p1_n")],
                        [i * md_flow_p.T / float(md_flow_p.nDTout)
                         for i in range(md_flow_p.nDTout + 1)])
        ns = NumericalSolution.NS_base(so, pList, nList, so.sList, opts)
        failed = ns.calculateSolution("md_test")
    finally:
        sys.path.remove(HERE)
        for name in DECK_MODULES:
            sys.modules.pop(name, None)
        Context.contextOptionsString = saved_context
        os.chdir(cwd)
    assert not failed
    flow = ns.modelList[0].levelModelList[-1]
    trans = ns.modelList[1].levelModelList[-1]
    # 9x9 nodes, 128 triangles
    assert trans.mesh.nElements_global == 128
    # head between the outflow (0 m) and inflow (100 m) values
    h = flow.u[0].dof
    assert h.min() > -1.0e-8 and h.max() < 100.0 + 1.0e-8
    # Darcy velocity from the p1-nc post-processing is nonzero and points from
    # the inflow (left, bottom) to the outflow (right, top)
    q = flow.q[("velocity", 0)]
    assert np.isfinite(q).all()
    assert q[..., 0].mean() > 0.0 and q[..., 1].mean() > 0.0
    # the injected tracer has entered. SUPG with shock capturing undershoots at
    # the front on coarse meshes (min c measured -0.155 on this 8x8 mesh, -0.29
    # on 16x16, -3e-5 on 32x32), so the lower bound is loose
    c = trans.u[0].dof
    assert np.isfinite(c).all()
    assert c.max() <= 1.0 + 1.0e-8 and c.min() > -0.5
    # tracer mass at t = 500 d (theta c integrated), measured 26191.0447689
    mass = (trans.q[("m", 0)] * trans.q[("dV_u", 0)]).sum()
    assert abs(mass - 26191.0447689) < 1.0e-4 * 26191.0447689
