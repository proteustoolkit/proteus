"""The MCorr Jacobian is the derivative of its residual (finite-difference check).

Newton's quadratic convergence, and its convergence to a tight absolute tolerance at all, needs J = dR/du.  The check compares J.d with the
central difference (R(u + h d) - R(u - h d)) / (2 h) for a random direction d and perturbations so small that no node of the level set
phi + u changes sign (the interface only moves inside its cells), so the residual is smooth along the line.  For an exact Jacobian the relative
error falls like h^2 until round-off; an inexact one (a term missing from J, or a Dirac of another width than the Heaviside it differentiates)
shows as an error that does not fall with h.

Three things this test pins (each failed before it was written):
  * the smoothed Heaviside, whatever epsFactDirac is: J differentiates r, so its Dirac must use the Heaviside's width (epsFactHeaviside);
  * the sharp (equivalent-polynomial) Heaviside, useExact: on cut elements the Dirac weights times N_i N_j integrate a quadratic exactly only
    if the Dirac weights are exact for quadratics, which they are not; J uses the derivative of the Heaviside weights instead (error 2e-4 -> 1e-10).
"""
from importlib import reload

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import fd_jacobian  # noqa: E402  (test/fd_jacobian.py)

from proteus import NumericalSolution
from . import (ls_vortex_2d_p, redist_vortex_2d_p, vof_vortex_2d_p, ls_consrv_vortex_2d_p, ls_vortex_2d_n,
               redist_vortex_2d_n, vof_vortex_2d_n, ls_consrv_vortex_2d_n, ls_vortex_2d_so)

H_FD = 1.0e-6          # perturbation of the correction dofs (the level set moves by <= ~4e-6, the cells are 0.05 wide)
TOL = 1.0e-8           # observed: 5e-13 (smooth), 3e-11 (sharp); an inexact Jacobian shows 1e-4 and does not fall with h


def _mcorr_after_two_steps(opts):
    from proteus import default_s
    reload(default_s)
    so = ls_vortex_2d_so
    tnList = so.tnList
    so.tnList = tnList[:3]
    try:
        ns = NumericalSolution.NS_base(so, [ls_vortex_2d_p, redist_vortex_2d_p, vof_vortex_2d_p, ls_consrv_vortex_2d_p],
                                       [ls_vortex_2d_n, redist_vortex_2d_n, vof_vortex_2d_n, ls_consrv_vortex_2d_n],
                                       [default_s] * len(so.pnList), opts)
        ns.calculateSolution(so.name + "_mcorr_jacobian")
    finally:
        so.tnList = tnList
    return [m.levelModelList[-1] for m in ns.modelList if type(m.levelModelList[-1]).__module__.endswith("MCorr")][0]


def _relative_error(mc, h):
    u0 = mc.u[0].dof.copy()
    phi = mc.coefficients.lsModel.u[0].dof
    return fd_jacobian.levelmodel_error(mc, h, kink_distance=np.abs(phi + u0).min())


@pytest.mark.parametrize("dirac_over_heaviside", [1.0, 0.5, 2.0])
def test_smoothed_heaviside_jacobian_is_exact_whatever_the_dirac_width(opts, tmp_path, monkeypatch, dirac_over_heaviside):
    monkeypatch.chdir(tmp_path)
    mc = _mcorr_after_two_steps(opts)
    mc.coefficients.epsFactDirac = dirac_over_heaviside * mc.coefficients.epsFactHeaviside
    assert not mc.coefficients.useExact
    assert _relative_error(mc, H_FD) < TOL


def _sharp(mc, mode):
    mc.coefficients.useExact = True
    mc.coefficients.flowCoefficients.useExact = True
    mc.coefficients.exactJacobian = mode
    phi = mc.coefficients.lsModel.u[0].dof + mc.u[0].dof
    assert (np.abs(phi) < 0.3 * mc.elementDiameter.mean()).sum() > 10, "no cut elements: the check would not test the sharp Jacobian"


@pytest.mark.parametrize("mode", ["analytic", "fd", "auto"])
def test_sharp_heaviside_jacobian_is_exact_on_cut_elements(opts, tmp_path, monkeypatch, mode):
    """This deck integrates with a quadrature of order 5 (>= nP + 2 = 4), so the analytic Jacobian D/|grad phi| is exact; the finite-difference mode is exact at any order."""
    monkeypatch.chdir(tmp_path)
    mc = _mcorr_after_two_steps(opts)
    assert mc.quadrature_order == 5
    _sharp(mc, mode)
    assert mc._exact_jacobian_fd() == (1 if mode == "fd" else 0)      # auto picks the analytic Jacobian when the quadrature allows it
    assert _relative_error(mc, H_FD) < TOL
