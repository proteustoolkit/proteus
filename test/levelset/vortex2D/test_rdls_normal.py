"""RDLS records the unit normal of the level set, grad(phi)/|grad(phi)|, in q[('grad(u)', 0)] ("q_n" in its kernel).

Until 2026-11-08 the element loop of RDLS.h's calculateResidual wrote q_n from a local, dir, whose assignment was
commented out: whatever the stack held. The exterior loop and the EV kernel write the normal (C19 round 3's finding).
"""
from importlib import reload

import numpy as np

from proteus import NumericalSolution
from . import (ls_vortex_2d_p, redist_vortex_2d_p, vof_vortex_2d_p, ls_consrv_vortex_2d_p, ls_vortex_2d_n,
               redist_vortex_2d_n, vof_vortex_2d_n, ls_consrv_vortex_2d_n, ls_vortex_2d_so)


def _p1_gradients(model):
    """The gradient of the P1 level set on each triangle, from its three nodes (independent of the kernel)."""
    mesh = model.mesh
    l2g = model.u[0].femSpace.dofMap.l2g
    dof = model.u[0].dof
    X = mesh.nodeArray[mesh.elementNodesArray][:, :, :2]          # (nElements, 3, 2)
    E = np.stack([X[:, 1] - X[:, 0], X[:, 2] - X[:, 0]], axis=1)   # rows: edge vectors
    U = dof[l2g]
    dU = np.stack([U[:, 1] - U[:, 0], U[:, 2] - U[:, 0]], axis=1)
    return np.linalg.solve(E, dU[..., None])[..., 0]               # (nElements, 2)


def test_rdls_records_the_unit_normal(opts, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from proteus import default_s
    reload(default_s)
    so = ls_vortex_2d_so
    tnList = so.tnList
    so.tnList = tnList[:3]
    try:
        ns = NumericalSolution.NS_base(so,
                                       [ls_vortex_2d_p, redist_vortex_2d_p, vof_vortex_2d_p, ls_consrv_vortex_2d_p],
                                       [ls_vortex_2d_n, redist_vortex_2d_n, vof_vortex_2d_n, ls_consrv_vortex_2d_n],
                                       [default_s]*len(so.pnList), opts)
        ns.calculateSolution(so.name + "_rdls_normal")
    finally:
        so.tnList = tnList
    rdls = [m.levelModelList[-1] for m in ns.modelList if type(m.levelModelList[-1]).__module__.endswith("RDLS")][0]
    # the last residual is evaluated at the converged dofs
    rdls.getResidual(rdls.u[0].dof.copy(), np.zeros(rdls.u[0].dof.shape))
    g = _p1_gradients(rdls)
    expected = g/np.sqrt(1.0e-8 + (g**2).sum(axis=1))[:, None]
    actual = rdls.q[('grad(u)', 0)]
    assert np.all(np.isfinite(actual))
    np.testing.assert_allclose(actual, np.broadcast_to(expected[:, None, :], actual.shape), atol=1e-10)
