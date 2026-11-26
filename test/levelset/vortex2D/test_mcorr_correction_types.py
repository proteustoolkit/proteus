"""MCorr correctionType 'dg', 'dgp0', 'global' and 'none' of the vortex2D deck: the four level solvers that were never run.

ls_consrv_vortex_2d_n.py picks the level solver of the mass corrector from ``correctionType``: ``MCorr.ElementNewton``
('dg': an element-by-element solve), ``ElementConstantNewton`` ('dgp0': one constant per element), ``GlobalConstantNewton``
('global': one constant for the domain) and ``DummyNewton`` ('none': no correction). 'cg' (Newton) and 'bisect'
(test_mcorr_bisection.py) are the other two. Run in-process (as test_mcorr_bisection.py) so the corrector's Python is measured by the
coverage job; two steps of the deck each.

Until the C0 kernel-bugs round only 'none' ran: MCorr.elementSolve, elementConstantSolve and globalConstantRJ did not pass
``r_l2g`` and the cut-cell masks (``isActiveR``, ``isActiveDOF``, ``isActiveElement``, ``phi_solid``, and for elementSolve
``phi_solid_nodes``, ``useExact_s``, ``ebqe_phi_s``, ``elementBoundariesArray``) that the C++ reads ("RuntimeError: key r_l2g
not found"), and globalConstantRJ shifted four arguments by one (``stride_u`` = the residual, ``globalResidual`` = the
porosity). Now:

- 'global' conserves the mass as Newton ('cg') does: measured 9.4e-11 serial, 4.4e-11 on 2 ranks (Newton 8.0e-11);
- 'dg' and 'dgp0' run but their element Newton diverges in elements the interface does not cross (there H(phi + u) does
  not depend on u, so no local root exists and du = -R/J with J ~ 0): |q_u| reaches ~1e12 ('dg') and ~1e5 ('dgp0').
  Pinned below so that a fix is noticed; the variants are on the dead-code list (HANDOFF_2026-11-23_c49-r3-done.md).
"""
from importlib import reload

import numpy as np
import pytest

from proteus import NumericalSolution
from . import (vortex2D, ls_vortex_2d_p, redist_vortex_2d_p, vof_vortex_2d_p, ls_consrv_vortex_2d_p, ls_vortex_2d_n,
               redist_vortex_2d_n, vof_vortex_2d_n, ls_consrv_vortex_2d_n, ls_vortex_2d_so)


def _decks(correction):
    reload(vortex2D)
    vortex2D.correctionType = correction
    reload(ls_consrv_vortex_2d_p)
    reload(ls_consrv_vortex_2d_n)


@pytest.fixture
def correction_decks(request):
    _decks(request.param)
    yield request.param
    _decks("cg")      # the other vortex2D tests import these modules: put the defaults back


def _run_two_steps(opts, name):
    from proteus import default_s
    reload(default_s)
    so = ls_vortex_2d_so
    tnList, soname = so.tnList, so.name
    so.tnList = tnList[:3]
    try:
        ns = NumericalSolution.NS_base(so, [ls_vortex_2d_p, redist_vortex_2d_p, vof_vortex_2d_p, ls_consrv_vortex_2d_p],
                                       [ls_vortex_2d_n, redist_vortex_2d_n, vof_vortex_2d_n, ls_consrv_vortex_2d_n],
                                       [default_s] * len(so.pnList), opts)
        ns.calculateSolution(soname + "_" + name)
    finally:
        so.tnList = tnList
    return ns


@pytest.mark.parametrize("correction_decks, solver", [("dg", "ElementNewton"), ("dgp0", "ElementConstantNewton"),
                                                      ("global", "GlobalConstantNewton"), ("none", "DummyNewton")],
                         indirect=["correction_decks"])
def test_the_level_solver_the_deck_names_runs(opts, tmp_path, monkeypatch, correction_decks, solver):
    from proteus.kernels import MCorr
    monkeypatch.chdir(tmp_path)
    ns = _run_two_steps(opts, correction_decks)
    m = [m for m in ns.modelList if type(m.levelModelList[-1]).__module__.endswith("MCorr")][0]
    mc = m.levelModelList[-1]
    assert type(m.solver.solverList[-1]) is getattr(MCorr, solver)
    u = mc.u[0].dof
    assert np.isfinite(u).all()
    assert np.isfinite(mc.coefficients.massConservationError)
    if correction_decks == "none":
        assert np.all(mc.q[('u', 0)] == 0.0) and np.all(mc.q[('r', 0)] == 0.0)
    if correction_decks == "global":
        assert np.all(u == u[0]), "the global correction is one constant for every dof"
        assert mc.coefficients.massConservationError < 1e-8, "the global constant does not conserve the mass"
    if correction_decks in ("dg", "dgp0"):
        # the element solves keep the correction in q_u (the dofs stay 0) and diverge where the interface is absent;
        # when this fails they no longer do: replace it with a test that each element conserves its mass
        assert np.all(u == 0.0)
        assert np.abs(mc.q[('u', 0)]).max() > 1e3
