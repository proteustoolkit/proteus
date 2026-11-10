"""The sharp-Heaviside MCorr Jacobian and the quadrature order (the rotation deck integrates the correction with a quadrature of order 3).

The Jacobian of the sharp Heaviside is  sum_q w_q (D_q / |grad phi|) N_i N_j  with the equivalent-polynomial Dirac D_q (a SURFACE Dirac: it integrates to the
interface measure, so dH/dphi = D/|grad phi|).  Equivalent polynomials of order nP = 2 (MCorr.h) reproduce a polynomial moment of degree 2 only with a
quadrature of degree nP + 2 = 4: at order 3 the analytic Jacobian is inexact (measured 2e-4), so 'analytic' refuses, 'auto' falls back to the
finite-difference Jacobian (exact for the discrete residual at any order) and 'fd' is exact.  (levelset/vortex2D/test_mcorr_jacobian.py covers order 5.)
"""
import os
import sys
from importlib import reload

import numpy as np
import pytest

from proteus import NumericalSolution
from . import (ls_rotation_2d_p, redist_rotation_2d_p, vof_rotation_2d_p, ls_consrv_rotation_2d_p, ls_rotation_2d_n,
               redist_rotation_2d_n, vof_rotation_2d_n, ls_consrv_rotation_2d_n, ls_rotation_2d_so)

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import fd_jacobian  # noqa: E402  (test/fd_jacobian.py)

H_FD = 1.0e-6
TOL = 1.0e-8


def _mcorr_after_two_steps(opts):
    from proteus import default_s
    reload(default_s)
    so = ls_rotation_2d_so
    tnList = so.tnList
    so.tnList = tnList[:3]
    try:
        ns = NumericalSolution.NS_base(so, [ls_rotation_2d_p, redist_rotation_2d_p, vof_rotation_2d_p, ls_consrv_rotation_2d_p],
                                       [ls_rotation_2d_n, redist_rotation_2d_n, vof_rotation_2d_n, ls_consrv_rotation_2d_n],
                                       [default_s] * len(so.pnList), opts)
        ns.calculateSolution(so.name + "_mcorr_jacobian")
    finally:
        so.tnList = tnList
    mc = [m.levelModelList[-1] for m in ns.modelList if type(m.levelModelList[-1]).__module__.endswith("MCorr")][0]
    mc.coefficients.useExact = True
    mc.coefficients.flowCoefficients.useExact = True
    phi = mc.coefficients.lsModel.u[0].dof + mc.u[0].dof
    assert (np.abs(phi) < 0.3 * mc.elementDiameter.mean()).sum() > 3, "no cut elements: the check would not test the sharp Jacobian"
    return mc


def _error(mc):
    return fd_jacobian.levelmodel_error(mc, H_FD, kink_distance=np.abs(mc.coefficients.lsModel.u[0].dof + mc.u[0].dof).min())


def test_the_analytic_sharp_jacobian_refuses_a_quadrature_below_degree_four(opts, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    mc = _mcorr_after_two_steps(opts)
    assert mc.quadrature_order == 3
    mc.coefficients.exactJacobian = "analytic"
    with pytest.raises(ValueError, match="degree 4"):
        mc.getJacobian(mc.jacobian)


@pytest.mark.parametrize("mode", ["auto", "fd"])
def test_the_finite_difference_sharp_jacobian_is_exact_at_quadrature_order_three(opts, tmp_path, monkeypatch, mode):
    monkeypatch.chdir(tmp_path)
    mc = _mcorr_after_two_steps(opts)
    mc.coefficients.exactJacobian = mode
    assert mc._exact_jacobian_fd() == 1
    assert _error(mc) < TOL
