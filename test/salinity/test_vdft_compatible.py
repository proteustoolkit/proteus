"""Compatible flow -> transport coupling (Barua & Kees, Sec. 2.2; TADR STABILIZATION_TYPE='Compatible').

The transport is driven by the flow step's realized edge exchanges E_ij and carries its storage
M^{n+1} = M^n + (m^{n+1} - m^n), so the discrete fluid balance it sees is the one the flow solved
(Dawson, Sun & Wheeler 2004). Three consequences are tested:

* a constant concentration stays constant, whatever the flow does (levee deck, c0 = cin = 0.5);
* a stably stratified hydrostatic column stays at rest;
* the fluid and solute budgets close to the flow and transport solver tolerances, and the
  concentration stays within [0, cin] when the inflow is denser (density-coupled levee deck).
"""
import importlib
import os
import sys

import numpy as np
import pytest

from proteus.iproteus import *  # noqa: F401,F403  (opts, default_s, NumericalSolution)
from proteus import LinearSolvers, NonlinearSolvers, StepControl
from proteus.mprans import TADR

HERE = os.path.dirname(os.path.abspath(__file__))
CASE_D = os.path.join(HERE, 'Levee_contamination_Density', 'Case_D')
CASE_D_MODULES = ['domain_rg', 'Raingarden_p', 'Raingarden_n', 'tranport_p', 'tranport_n', 'ls_CCS_so']
COLUMN = os.path.join(HERE, 'stratified_column')
COLUMN_MODULES = ['column_domain', 'column_flow_p', 'column_flow_n', 'column_transport_p',
                  'column_transport_n', 'column_so']


def _load(directory, names):
    """Import a deck's modules by their own (absolute) names, fresh."""
    for name in names:
        sys.modules.pop(name, None)
    sys.path.insert(0, directory)
    try:
        return [importlib.import_module(name) for name in names]
    finally:
        sys.path.remove(directory)


def _unload(names):
    for name in names:
        sys.modules.pop(name, None)


def _direct_solvers(*nModules):
    for n in nModules:
        n.multilevelLinearSolver = LinearSolvers.LU
        n.levelLinearSolver = LinearSolvers.LU


def _run(so, pList, nList, name):
    so.name = name
    for p in pList:
        p.name = p.__name__
    so.sList = [default_s for p in pList]
    ns = NumericalSolution.NS_base(so, pList, nList, so.sList, opts)
    ns.calculateSolution(name)
    return ns


def _compatible_levee_transport(tp, tn, c0, cin):
    """The levee deck's transport, switched to the compatible scheme."""
    tp.coefficients = tp.MyCoefficients(tp.aOfX,
                                        alpha_L=tp.alpha_L,
                                        alpha_T=tp.alpha_T,
                                        Dm=tp.Dm,
                                        porosity=0.41,
                                        V_model=0,
                                        specified_velocity=False,
                                        checkMass=False,
                                        FCT=True,
                                        LUMPED_MASS_MATRIX=False,
                                        STABILIZATION_TYPE='Compatible',
                                        diagonal_conductivity=True,
                                        rho_f=tp.coefficients.rho_f,
                                        rho_s=tp.coefficients.rho_s)
    tp.coefficients.variableNames = ['u']
    tp.initialConditions = {0: tp.ConstantIC(c0)}

    def getLeftConcDirichletBCs(x, tag):
        if tag == tp.boundaryTags['leftTop']:
            return lambda x, t: cin
    tp.dirichletConditions = {0: getLeftConcDirichletBCs}
    tn.timeIntegration = TADR.CompatibleBackwardEuler
    tn.stepController = StepControl.Min_dt_controller
    tn.multilevelNonlinearSolver = NonlinearSolvers.Newton
    tn.levelNonlinearSolver = NonlinearSolvers.Newton
    tn.fullNewtonFlag = True
    tn.updateJacobian = True
    tn.shockCapturing = None
    tn.tolFac = 0.0
    tn.nl_atol_res = 1.0e-10
    tn.maxNonlinearIts = 20
    tn.maxLineSearches = 0


def _check_budgets(transport, flow_atol, transport_atol):
    """Every step's fluid and solute budget closes to its solver's tolerance.

    Newton stops at ||r||_2 <= nl_atol_res (tolFac = 0), and the budget errors are dt sum(r)
    over the flow and transport residuals, so |error| <= dt sqrt(N) nl_atol_res, plus round-off.
    """
    sqrtN = np.sqrt(transport.u[0].dof.shape[0])
    worst_fluid, worst_solute = 0.0, 0.0
    for b in transport.compatibleBudget:
        fluid_tol = b['dt'] * sqrtN * flow_atol + 1.0e-12 * b['fluid_mass']
        solute_tol = b['dt'] * sqrtN * transport_atol + 1.0e-12 * max(1.0, b['solute_mass'])
        assert abs(b['fluid_budget_error']) <= fluid_tol, b
        assert abs(b['solute_budget_error']) <= solute_tol, b
        worst_fluid = max(worst_fluid, abs(b['fluid_budget_error']) / fluid_tol)
        worst_solute = max(worst_solute, abs(b['solute_budget_error']) / solute_tol)
    return worst_fluid, worst_solute


@pytest.fixture
def levee_deck(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    modules = _load(CASE_D, CASE_D_MODULES)
    yield modules
    _unload(CASE_D_MODULES)


@pytest.mark.parametrize('flow_fct', [None, 'semi-implicit', 'implicit'])
def test_levee_constant_concentration(levee_deck, flow_fct):
    """T1: c0 = cin = 0.5 on the levee deck stays 0.5 while the vadose zone wets up, with the
    flow's low-order scheme and with either flux correction (whose realized corrections enter E_ij).

    1.9.0 (pointwise velocity_couple, recomputed theta rho(c) c storage) reaches
    max|c - 0.5| = 0.96 by 10 d on this deck.
    """
    domain, rp, rn, tp, tn, so = levee_deck
    if flow_fct is not None:
        rp.coefficients.FCT = True
        rp.coefficients.FCT_TYPE = flow_fct
    _compatible_levee_transport(tp, tn, c0=0.5, cin=0.5)
    _direct_solvers(rn, tn)
    so.tnList = [0.0, 1.0e-8, 2.0e-8] + [10.0 * i for i in range(1, 4)]
    ns = _run(so, [rp, tp], [rn, tn], 'vdft_levee_t1')
    transport = ns.modelList[1].levelModelList[-1]
    budget = transport.compatibleBudget
    assert len(budget) > 10
    assert budget[-1]['t'] == pytest.approx(30.0)
    deviation = max(max(abs(b['c_max'] - 0.5), abs(b['c_min'] - 0.5)) for b in budget)
    assert deviation < 1.0e-10
    assert np.abs(transport.u[0].dof - 0.5).max() < 1.0e-10
    _check_budgets(transport, rn.nl_atol_res, tn.nl_atol_res)


def test_levee_density_budgets_and_bounds(levee_deck):
    """Denser inflow (cin = 0.7, eps = 0.025) into fresh water: c stays in [0, cin] and both
    budgets close to solver tolerance. 1.9.0 overshoots to c = 0.740 by 500 d on this deck."""
    domain, rp, rn, tp, tn, so = levee_deck
    cin = 0.7
    _compatible_levee_transport(tp, tn, c0=0.0, cin=cin)
    _direct_solvers(rn, tn)
    so.tnList = [0.0, 1.0e-8, 2.0e-8] + [10.0 * i for i in range(1, 4)]
    ns = _run(so, [rp, tp], [rn, tn], 'vdft_levee_density')
    transport = ns.modelList[1].levelModelList[-1]
    budget = transport.compatibleBudget
    assert len(budget) > 10
    assert budget[-1]['solute_mass'] > 1.0  # solute has entered
    assert min(b['c_min'] for b in budget) >= -1.0e-12
    assert max(b['c_max'] for b in budget) <= cin + 1.0e-12
    _check_budgets(transport, rn.nl_atol_res, tn.nl_atol_res)
    # the dispersion flux q_h satisfies int q_h . grad w_i = sum_j Espatial_ij (P1 triangles)
    mesh = transport.mesh
    x = mesh.nodeArray[:, :2][mesh.elementNodesArray]
    J = np.stack([x[:, 1] - x[:, 0], x[:, 2] - x[:, 0]], axis=2)  # columns are edge vectors
    gradRef = np.array([[-1.0, -1.0], [1.0, 0.0], [0.0, 1.0]])
    grads = np.einsum('kr,erI->ekI', gradRef, np.linalg.inv(J))
    area = 0.5 * np.abs(np.linalg.det(J))
    moments = np.zeros(transport.u[0].dof.shape)
    np.add.at(moments, mesh.elementNodesArray, area[:, None] * np.einsum('ekI,eI->ek', grads, transport._element_q))
    ex = ns.modelList[0].levelModelList[-1].transportExport
    rowSum = np.bincount(transport._exportRows, ex['Espatial'], moments.shape[0])
    assert np.abs(moments - rowSum).max() <= 1.0e-10 * max(1.0, np.abs(rowSum).max())


def test_stratified_column_at_rest(tmp_path, monkeypatch):
    """Hydrostatic column with c linear in z (dense at the bottom) stays stationary.

    The deck's previous transport (explicit edge-based EV with the pointwise velocity_couple)
    moves c by 0.11 in 10 d and overshoots to c = 1.11.
    """
    monkeypatch.chdir(tmp_path)
    domain, fp, fn, tp, tn, so = _load(COLUMN, COLUMN_MODULES)
    try:
        ns = _run(so, [fp, tp], [fn, tn], 'vdft_stratified_column')
        flow = ns.modelList[0].levelModelList[-1]
        transport = ns.modelList[1].levelModelList[-1]
        z = transport.mesh.nodeArray[:transport.u[0].dof.shape[0], 1]
        assert len(transport.compatibleBudget) > 5
        assert transport.compatibleBudget[-1]['t'] == pytest.approx(so.T)
        assert np.abs(transport.u[0].dof - fp.concentration(z)).max() < 1.0e-10
        assert np.abs(flow.u[0].dof - fp.hydrostatic_pressure_head(z)).max() < 1.0e-10
        assert np.abs(flow.transportExport['E']).max() < 1.0e-12
        _check_budgets(transport, fn.nl_atol_res, tn.nl_atol_res)
    finally:
        _unload(COLUMN_MODULES)
