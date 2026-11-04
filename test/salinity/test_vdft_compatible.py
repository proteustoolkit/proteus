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
        # the decks inherit genMesh from proteus.default_p, which other test modules set to False
        # at collection (test/post_processing/import_modules/bdm_tests_3Dtemplate.py)
        p.genMesh = True
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


# --------------------------------------------------------------------------
# the edge-flux coupling of the edge-based TADR schemes (types 2-5) to Richards
# --------------------------------------------------------------------------


def _edge_flux_levee_transport(tp, tn, stab, order, c0, cin):
    """The levee deck's transport on an edge-based TADR scheme that advects with Richards' edge fluxes:
    2-4 explicit (SSP stages of order `order`, lumped mass), 5 ImplicitEV (backward Euler, Newton)."""
    explicit = stab in (2, 3, 4)
    tp.coefficients = tp.MyCoefficients(tp.aOfX,
                                        alpha_L=tp.alpha_L,
                                        alpha_T=tp.alpha_T,
                                        Dm=tp.Dm,
                                        porosity=0.41,
                                        V_model=0,
                                        specified_velocity=False,
                                        checkMass=False,
                                        FCT=True,
                                        LUMPED_MASS_MATRIX=explicit,
                                        STABILIZATION_TYPE=stab,
                                        diagonal_conductivity=True,
                                        rho_f=tp.coefficients.rho_f,
                                        rho_s=tp.coefficients.rho_s)
    tp.coefficients.variableNames = ['u']
    tp.initialConditions = {0: tp.ConstantIC(c0)}

    def getLeftConcDirichletBCs(x, tag):
        if tag == tp.boundaryTags['leftTop']:
            return lambda x, t: cin
    tp.dirichletConditions = {0: getLeftConcDirichletBCs}
    tn.stepController = StepControl.Min_dt_controller
    tn.multilevelNonlinearSolver = NonlinearSolvers.Newton
    tn.tolFac = 0.0
    tn.nl_atol_res = 1.0e-10
    tn.maxLineSearches = 0
    # the edge-based kernels read the shock-capturing lag and factor even when the factor is 0
    tn.shockCapturing = TADR.ShockCapturing(tp.coefficients, tp.nd, shockCapturingFactor=0.0, lag=True)
    if explicit:
        tn.timeIntegration = TADR.RKEV
        tn.timeOrder = tn.nStagesTime = order
        tn.runCFL = 0.5
        tn.levelNonlinearSolver = NonlinearSolvers.ExplicitLumpedMassMatrix
        tn.fullNewtonFlag = False
        tn.updateJacobian = False
    else:
        # as test/tadr_partition_mpi.py runs ImplicitEV: CFL-limited backward Euler, Newton, FCT after the solve
        from proteus.TimeIntegration import BackwardEuler_cfl
        tn.timeIntegration = BackwardEuler_cfl
        tn.timeOrder = tn.nStagesTime = 1
        tn.runCFL = 0.5
        tn.levelNonlinearSolver = NonlinearSolvers.Newton
        tn.fullNewtonFlag = True
        tn.updateJacobian = True
        tn.maxNonlinearIts = 100


EDGE_FLUX_CASES = [(2, 1), (3, 1), (4, 1), (4, 2), (4, 3), (5, 1)]


@pytest.mark.parametrize('stab,order', EDGE_FLUX_CASES)
def test_levee_edge_flux_constant_concentration(levee_deck, stab, order):
    """c0 = cin = 0.5 stays 0.5 when the edge-based TADR schemes advect with Richards' realized edge
    fluxes (Richards STABILIZATION_TYPE 2 exports them), at every SSP order: the stage blends act on
    the masses M c, not on c."""
    domain, rp, rn, tp, tn, so = levee_deck
    _edge_flux_levee_transport(tp, tn, stab, order, c0=0.5, cin=0.5)
    _direct_solvers(rn, tn)
    so.tnList = [0.0, 1.0e-8, 2.0e-8] + [10.0 * i for i in range(1, 3)]
    ns = _run(so, [rp, tp], [rn, tn], 'vdft_levee_edge_flux_%d_%d' % (stab, order))
    transport = ns.modelList[1].levelModelList[-1]
    assert transport._use_edge_flux == 1
    deviation = np.abs(transport.u[0].dof - 0.5).max()
    assert deviation < 1.0e-10


DIAGNOSTIC_CASES = [
    # (TADR type, TADR diagnostics, Richards FCT type, Richards diagnostics, log lines that must appear)
    (4, ("boundary_flux", "dmp", "rowsum"), "semi-implicit", ("boundary_flux", "edge_flux", "fct"),
     ("[TADR-BC]", "TADR_DMP t=", "TADR_ROWSUM", "[RICHARDS-BC]", "[RICHARDS-EDGE]", "[FCT-CONS]", "[FCT] ")),
    (5, ("fct",), None, (),
     ("TADR FCT sharpness",)),
]


@pytest.mark.parametrize('stab,tadr_diag,flow_fct,richards_diag,expected', DIAGNOSTIC_CASES)
def test_levee_edge_flux_diagnostics_run(levee_deck, monkeypatch, stab, tadr_diag, flow_fct, richards_diag,
                                         expected):
    """Every run-time diagnostic, turned on through the Coefficients options, runs and logs without a probe failure
    (the probes catch their own exceptions, so a broken one would only say so in the log), and leaves the
    solution unchanged: c0 = cin = 0.5 stays 0.5."""
    from proteus.richards import Richards
    logged = []

    def record(msg, level=1, *args, **kwargs):
        logged.append(str(msg))

    monkeypatch.setattr(TADR, 'logEvent', record)
    monkeypatch.setattr(Richards, 'logEvent', record)
    domain, rp, rn, tp, tn, so = levee_deck
    rp.coefficients.diagnostics = Richards._diagnostics(richards_diag)
    if flow_fct is not None:
        rp.coefficients.FCT = True
        rp.coefficients.FCT_TYPE = flow_fct
    _edge_flux_levee_transport(tp, tn, stab, 1, c0=0.5, cin=0.5)
    tp.coefficients.diagnostics = TADR._diagnostics(tadr_diag)
    _direct_solvers(rn, tn)
    so.tnList = [0.0, 1.0e-8, 2.0e-8, 10.0]
    ns = _run(so, [rp, tp], [rn, tn], 'vdft_levee_diagnostics_%d' % stab)
    transport = ns.modelList[1].levelModelList[-1]
    assert transport._use_edge_flux == 1
    assert np.abs(transport.u[0].dof - 0.5).max() < 1.0e-10
    assert not [m for m in logged if 'probe failed' in m]
    for line in expected:
        assert any(m.startswith(line) for m in logged), line


def test_levee_edge_velocity(levee_deck):
    """EDGE_VELOCITY: the transport advects with the velocity Richards rebuilds from its realized edge fluxes
    (Richards STABILIZATION_TYPE 2), and a constant concentration still stays constant."""
    domain, rp, rn, tp, tn, so = levee_deck
    _edge_flux_levee_transport(tp, tn, 5, 1, c0=0.5, cin=0.5)
    tp.coefficients.EDGE_VELOCITY = True
    _direct_solvers(rn, tn)
    so.tnList = [0.0, 1.0e-8, 2.0e-8, 10.0]
    ns = _run(so, [rp, tp], [rn, tn], 'vdft_levee_edge_velocity')
    transport = ns.modelList[1].levelModelList[-1]
    flow = ns.modelList[0].levelModelList[-1]
    assert (transport._use_edge_flux, transport._use_edge_velocity) == (1, 1)
    assert np.isfinite(flow.q[('velocity_edge', 0)]).all()
    assert np.abs(transport.u[0].dof - 0.5).max() < 1.0e-10


@pytest.mark.parametrize('model', ['TADR', 'Richards'])
def test_an_unknown_diagnostic_is_refused(model):
    from proteus.richards import Richards
    module = {'TADR': TADR, 'Richards': Richards}[model]
    with pytest.raises(ValueError, match="unknown %s diagnostics" % model):
        module._diagnostics(("boundary_flux", "no_such_probe"))
    assert module._diagnostics("boundary_flux") == frozenset(["boundary_flux"])


@pytest.mark.xfail(strict=True, reason="ImplicitEV (TADR STABILIZATION_TYPE 5) with Richards' semi-implicit FCT "
                                       "does not keep a constant concentration: c = 0.5 reaches about 550 by 10 d "
                                       "(1748 with FCT_BND_DIFF); types 2-4 keep it to round-off")
def test_levee_implicit_ev_with_flow_fct(levee_deck):
    domain, rp, rn, tp, tn, so = levee_deck
    rp.coefficients.FCT = True
    rp.coefficients.FCT_TYPE = 'semi-implicit'
    _edge_flux_levee_transport(tp, tn, 5, 1, c0=0.5, cin=0.5)
    _direct_solvers(rn, tn)
    so.tnList = [0.0, 1.0e-8, 2.0e-8, 10.0]
    ns = _run(so, [rp, tp], [rn, tn], 'vdft_levee_implicit_ev_flow_fct')
    transport = ns.modelList[1].levelModelList[-1]
    assert transport._use_edge_flux == 1
    assert np.abs(transport.u[0].dof - 0.5).max() < 1.0e-10
