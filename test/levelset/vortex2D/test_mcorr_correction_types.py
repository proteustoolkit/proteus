"""MCorr correctionType 'dg', 'dgp0', 'global' and 'none' of the vortex2D deck: the four level solvers that were never run.

ls_consrv_vortex_2d_n.py picks the level solver of the mass corrector from ``correctionType``: ``MCorr.ElementNewton``
('dg': an element-by-element solve), ``ElementConstantNewton`` ('dgp0': one constant per element), ``GlobalConstantNewton``
('global': one constant for the domain) and ``DummyNewton`` ('none': no correction). 'cg' (Newton) and 'bisect'
(test_mcorr_bisection.py) are the other two. Run in-process (as test_mcorr_bisection.py) so the corrector's Python is measured by the
coverage job; two steps of the deck each.

Only 'none' runs. MCorr.elementSolve, elementConstantSolve and globalConstantRJ (so 'dg', 'dgp0' and 'global') do not put
``r_l2g`` in the arguments of the C++ call (residual and Jacobian do, MCorr.py:703 and :854), which the C++ side reads: all three raise
"RuntimeError: key r_l2g not found". The tests are strict xfails: they run the Python up to the call, and when the kernel is fixed
they pass and the marks have to go. The report for the kernel's owner is in HANDOFF_2026-11-23_c49-r3-done.md.
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


_R_L2G = pytest.mark.xfail(strict=True, raises=RuntimeError, reason="MCorr does not pass r_l2g to elementSolve, elementConstantSolve, globalConstantRJ")


@pytest.mark.parametrize("correction_decks, solver", [pytest.param("dg", "ElementNewton", marks=_R_L2G),
                                                      pytest.param("dgp0", "ElementConstantNewton", marks=_R_L2G),
                                                      pytest.param("global", "GlobalConstantNewton", marks=_R_L2G),
                                                      ("none", "DummyNewton")],
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
