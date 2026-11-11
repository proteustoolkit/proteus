"""MCorr correctionType 'bisect': a constant mass-conserving shift of the level set found by bracketing and bisection.

With the sharp (equivalent-polynomial) Heaviside the mass of phi + c is piecewise smooth in c, with kinks where the zero level set crosses a node;
Newton on the global constant can stall at a kink, a bracketing method cannot.  Run in-process (unlike the subprocess Jacobian tests) so the
corrector's Python is measured by the coverage job.  Checked here: the shift conserves the VOF mass to the deck's absolute tolerance, it is one
constant for every dof, and it is the root of the mass defect (the mass evaluated independently at c, c - 10 atol and c + 10 atol brackets the target),
starting from too much and from too little mass.
"""
from importlib import reload

import numpy as np
import pytest

from proteus import NumericalSolution, Norms
from proteus.Comm import globalSum
from . import (vortex2D, ls_vortex_2d_p, redist_vortex_2d_p, vof_vortex_2d_p, ls_consrv_vortex_2d_p, ls_vortex_2d_n,
               redist_vortex_2d_n, vof_vortex_2d_n, ls_consrv_vortex_2d_n, ls_vortex_2d_so)


def _decks(use_exact, correction):
    reload(vortex2D)
    vortex2D.useExact = use_exact
    vortex2D.correctionType = correction
    reload(ls_consrv_vortex_2d_p)
    reload(ls_consrv_vortex_2d_n)


@pytest.fixture
def bisect_decks():
    _decks(True, "bisect")
    yield
    _decks(False, "cg")      # the other vortex2D tests import these modules: put the defaults back


def _run_two_steps(opts):
    from proteus import default_s
    reload(default_s)
    so = ls_vortex_2d_so
    tnList, name = so.tnList, so.name
    so.tnList = tnList[:3]
    try:
        ns = NumericalSolution.NS_base(so, [ls_vortex_2d_p, redist_vortex_2d_p, vof_vortex_2d_p, ls_consrv_vortex_2d_p],
                                       [ls_vortex_2d_n, redist_vortex_2d_n, vof_vortex_2d_n, ls_consrv_vortex_2d_n],
                                       [default_s] * len(so.pnList), opts)
        ns.calculateSolution(name + "_bisect")
    finally:
        so.tnList = tnList
    return ns


def test_the_bisection_shift_conserves_the_vof_mass(opts, tmp_path, monkeypatch, bisect_decks):
    from proteus.mprans import MCorr
    monkeypatch.chdir(tmp_path)
    ns = _run_two_steps(opts)
    m = [m for m in ns.modelList if type(m.levelModelList[-1]).__module__.endswith("MCorr")][0]
    mc = m.levelModelList[-1]
    assert isinstance(m.solver.solverList[-1], MCorr.GlobalConstantBisection)
    assert mc.coefficients.useExact
    atol = ls_consrv_vortex_2d_n.nl_atol_res
    assert mc.atol == atol
    # the last solve (step 2) conserved mass to the tolerance, with at least one mass evaluation beyond f(0)
    assert mc.coefficients.massConservationError <= atol
    assert mc.bisection_its > 0
    # one constant for every dof
    assert np.all(mc.u[0].dof == mc.u[0].dof[0])


def test_the_bisection_brackets_from_either_side(opts, tmp_path, monkeypatch, bisect_decks):
    """Start from too much and from too little mass (the level set shifted by +-h/4): the corrector conserves mass from both sides, its shift is
    the root of the mass defect (evaluated independently at c and c -+ 10 atol), and phi + c is the same level set from both sides."""
    monkeypatch.chdir(tmp_path)
    ns = _run_two_steps(opts)
    mc = [m.levelModelList[-1] for m in ns.modelList if type(m.levelModelList[-1]).__module__.endswith("MCorr")][0]
    ls = mc.coefficients.lsModel
    h = float(np.max(mc.elementDiameter))
    co = mc.coefficients
    target = globalSum(Norms.scalarDomainIntegral(co.vofModel.q['dV'], co.vofModel.q[('m', 0)], mc.mesh.nElements_owned))
    mc.atol = ls_consrv_vortex_2d_n.nl_atol_res
    shifts = []
    for offset in (0.25 * h, -0.25 * h):
        ls.u[0].dof += offset
        mc.coefficients.q_u_ls += offset
        try:
            mc.globalConstantBisection(mc.u[0].dof, None)
            assert mc.coefficients.massConservationError <= mc.atol and mc.bisection_its > 1
            c = mc.u[0].dof[0]
            assert np.all(mc.u[0].dof == c)
            phi0, qphi0 = np.array(ls.u[0].dof), np.array(mc.coefficients.q_u_ls)

            def defect(s):
                return mc.calculateMass(None, phi_dof=phi0 + s, q_phi_values=qphi0 + s) - target

            assert abs(defect(c)) <= mc.atol
            assert defect(c - 10 * mc.atol) * defect(c + 10 * mc.atol) <= 0.0
            shifts.append(c + offset)
        finally:
            ls.u[0].dof -= offset
            mc.coefficients.q_u_ls -= offset
    # the conserving level set phi + c is the same from both sides, to the bracket width the tolerance allows
    assert shifts[0] == pytest.approx(shifts[1], abs=1.0e-3 * h)
