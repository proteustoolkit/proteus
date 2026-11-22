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
import shutil
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
        p.genMesh = not getattr(p, 'stored_mesh', False)
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
    # The mesh is stored, not generated: the external `triangle` of each environment (conda-forge on the Mac, on
    # Linux, the PETSc-built one the fleet nightly puts first on PATH) makes a different mesh from the same .poly
    # and options (26411, 26366 and 26312 elements), and the first dt of the explicit transport follows the mesh
    # (0.192 s on the first, 0.249 s on the last). The files are triangle's output on the Mac, .1. removed.
    for ext in ('node', 'ele', 'edge'):
        shutil.copy(os.path.join(CASE_D, 'stored_mesh', 'levee_case_d.' + ext), tmp_path)
    modules = _load(CASE_D, CASE_D_MODULES)
    domain, rp, rn, tp, tn, so = modules
    rp.stored_mesh = tp.stored_mesh = True   # read by _run
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


def test_levee_implicit_ev_with_flow_fct(levee_deck):
    """ImplicitEV (type 5) with Richards' semi-implicit FCT keeps c0 = cin = 0.5. Before a8508bdc the phase split
    re-formed m_dof = theta rho(u^n) u^n while M carried rho(u^{n-1}): c reached about 550 by 10 d here."""
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


# --------------------------------------------------------------------------
# which flow models the edge-flux coupling takes (review of the 2026-11-04 trains)
# --------------------------------------------------------------------------


class _Pattern(object):
    def __init__(self, n):
        self.rowptr, self.colind = np.array([0, n], 'i'), np.arange(n, dtype='i')

    def getCSRrepresentation(self):
        return self.rowptr, self.colind, None


def _transport_stub(stab, vModel, n=4):
    """The attributes TADR.LevelModel._refreshFluidMassFromFlow reads, on one rank."""
    from types import SimpleNamespace
    return SimpleNamespace(coefficients=SimpleNamespace(STABILIZATION_TYPE=stab, vModel=vModel, EDGE_VELOCITY=False,
                                                       VELOCITY_COUPLING=None),
                           edgeFlux=np.zeros(n), MC_global=_Pattern(n), M_dof=np.zeros(n), theta_nodal=np.full(n, -1.0),
                           _M_dof_initialized=True, _edge_flux_checked=False, _use_edge_flux=0, _use_edge_velocity=0)


def test_a_galerkin_flow_model_does_not_turn_on_edge_fluxes():
    """Richards allocates its edge-flux arrays for every type, but only its edge-based kernel fills them. A
    Galerkin Richards (type 0, the bioswale deck) left the arrays at zero and TADR advected nothing; it now keeps
    the transport on its own velocity, and a type-2 Richards still turns the coupling on."""
    from types import SimpleNamespace
    n = 4
    for flow_type, expected in [(0, 0), (2, 1)]:
        vModel = SimpleNamespace(coefficients=SimpleNamespace(STABILIZATION_TYPE=flow_type),
                                 edgeFluxInterior=np.zeros(n), MC_global=_Pattern(n), mn=np.zeros(n))
        stub = _transport_stub(4, vModel, n)
        try:
            TADR.LevelModel._refreshFluidMassFromFlow(stub)
        except Exception:
            pass  # past the gate, the edge-flux branch needs a whole flow model; only the decision is checked
        assert stub._use_edge_flux == expected, flow_type


def test_a_flow_model_without_nodal_theta_leaves_it_to_the_projection():
    """A flow model with a positive mn and no thetaW_dof (m_comp_co2) raised AttributeError; theta_nodal now stays
    at zero, which tells the kernel to keep its own projection, and M is taken from mn."""
    from types import SimpleNamespace
    n = 4
    vModel = SimpleNamespace(mn=np.full(n, 2.0))
    stub = _transport_stub(5, vModel, n)
    stub._edge_flux_checked = True
    assert TADR.LevelModel._refreshFluidMassFromFlow(stub) == 0
    np.testing.assert_array_equal(stub.theta_nodal, 0.0)
    np.testing.assert_array_equal(stub.M_dof, 2.0)


# The fleet's pip venvs on Linux (spark2 aarch64, ceds-model x86_64; nightly 33659c3a) crash this test; the conda builds
# (linux-64 CI) and the macOS venvs (chewbacca, darcy) pass it. It is not a test fault: with Richards' implicit FCT the
# Newton iteration of one of the two models (n = 13419) produces five Jacobians whose every entry is NaN, then one
# with entries of 1.2e16, and the serial SuperLU factorization (dgstrf -> dpivotL, superluWrappers) dies with SIGSEGV
# whatever the final time of the first long step (0.1, 1, 10 s all crash after ~50 s). Reproduction: this test, on a
# Linux pip build (about 50 s). Owner: the Richards implicit FCT and its Newton fallback (kernel Richards.h: C0;
# the solver's non-finite guard before the factorization), reported in HANDOFF_2026-11-15_c3-round9.md. The mark
# keeps the nightly's failure list to what is new; remove it when the divergence is fixed.
_LINUX_PIP_VENV = (sys.platform.startswith('linux') and sys.prefix != sys.base_prefix
                   and not os.path.exists(os.path.join(sys.prefix, 'conda-meta')))


@pytest.mark.xfail(_LINUX_PIP_VENV, run=False, reason="known: Richards implicit FCT diverges (NaN Jacobian) and "
                   "SuperLU dgstrf segfaults on Linux pip builds (C3 round 9 handoff)")
def test_levee_edge_flux_with_implicit_flow_fct(levee_deck):
    """Richards' implicit FCT sets fctCorrectionApplied but leaves the accepted mass in mLow (its final solve);
    the edge-flux coupling read mAccepted, all zeros, and drained its carried fluid mass every step. c0 = cin =
    0.5 stays 0.5.

    The run has to take its steps: with the first dt TADR used to propose (2.66e-14, from a CFL formed at M_low = 0)
    the flow model never accepts the first step (the residual's round-off floor, eps m_i ML_i / dt, is 5e-3 to 0.5
    against nl_atol_res 1e-6 at the dt = 2.7e-14 .. 5e-17 the step controller tries) and the run ends at t = 5e-17
    with c untouched, which is why this test passed for months. The first dt is now the deck's output interval,
    1e-8 (RKEV.initialize_dt); the run must reach t = 10 with every Newton solve of the flow converged and the head
    inside the range of the initial and boundary heads (-12 .. river stage)."""
    domain, rp, rn, tp, tn, so = levee_deck
    rp.coefficients.FCT = True
    rp.coefficients.FCT_TYPE = 'implicit'
    _edge_flux_levee_transport(tp, tn, 4, 1, c0=0.5, cin=0.5)
    _direct_solvers(rn, tn)
    so.tnList = [0.0, 1.0e-8, 2.0e-8, 10.0]
    failures = []
    newton_solve = NonlinearSolvers.Newton.solve

    def counting_solve(self, *args, **kwargs):
        out = newton_solve(self, *args, **kwargs)
        if type(self.F).__module__.endswith('Richards'):
            failures.append(bool(self.failedFlag))
        return out

    NonlinearSolvers.Newton.solve = counting_solve
    try:
        ns = _run(so, [rp, tp], [rn, tn], 'vdft_levee_edge_flux_implicit_fct')
    finally:
        NonlinearSolvers.Newton.solve = newton_solve
    flow = ns.modelList[0].levelModelList[-1]
    transport = ns.modelList[1].levelModelList[-1]
    assert transport._use_edge_flux == 1
    assert flow.acceptedMass() is flow.mLow
    assert np.abs(transport.u[0].dof - 0.5).max() < 1.0e-10
    # the flow stepped to the final time (the measured run: 3 output intervals, ~200 Newton solves, none failed)
    assert flow.timeIntegration.tLast == pytest.approx(so.tnList[-1])
    assert len(failures) > 20 and not any(failures), (len(failures), sum(failures))
    head = flow.u[0].dof
    assert np.all(np.isfinite(head))
    assert -12.0 - 1.0e-4 <= head.min() and head.max() <= rp.river_stage + 1.0e-4


def test_first_dt_is_not_set_by_an_uninitialized_mass(levee_deck, monkeypatch):
    """The first system dt, which is the minimum over the models, has to be one the flow model's Newton solve can
    take: the round-off of the stored mass over dt, eps m_i ML_i / dt, is the floor of ||r||, and nl_atol_res is
    1e-6, so dt > eps / nl_atol_res = 2e-10 (the deck's first output interval is 1e-8). The transport's RKEV
    used to propose 0.5 / 1.9e13 = 2.66e-14 because its edge-based CFL is formed from M_low = 0 (TADR.h floors it
    at 1e-14). Stops at the first step's preStep, so it needs no time stepping."""
    from proteus.richards import Richards
    domain, rp, rn, tp, tn, so = levee_deck
    rp.coefficients.FCT = True
    rp.coefficients.FCT_TYPE = 'implicit'
    _edge_flux_levee_transport(tp, tn, 4, 1, c0=0.5, cin=0.5)
    _direct_solvers(rn, tn)
    so.tnList = [0.0, 1.0e-8, 2.0e-8, 10.0]

    class Seen(Exception):
        pass

    seen = {}
    preStep = Richards.Coefficients.preStep

    def checked_preStep(self, t, firstStep=False):
        if firstStep:
            seen['dt'] = self.model.timeIntegration.dt
            raise Seen
        return preStep(self, t, firstStep)

    monkeypatch.setattr(Richards.Coefficients, 'preStep', checked_preStep)
    with pytest.raises(Seen):
        _run(so, [rp, tp], [rn, tn], 'vdft_levee_first_dt')
    assert seen['dt'] > np.finfo(float).eps / rn.nl_atol_res, seen


@pytest.mark.parametrize('order', [1, 3])
def test_first_dt_of_an_explicit_transport_respects_its_cfl(levee_deck, monkeypatch, order):
    """A long first output interval does not make the explicit transport's first step the interval. The first
    dt cannot be read from the CFL of the initial state with its carried mass of 0 (the test above), so
    RKEV.initialize_dt takes min(first interval, floored CFL dt * M_min / 1e-14), M_min the flow model's smallest
    initial mass: the dispersive CFL of the real mass, 0.19 s here (the flow's fluxes are 0 until it steps, so the
    advective CFL is not known). With the first interval of 1000 s as the first dt the run does not survive it:
    c = -2.6e8 .. 9.5e7, 20 failed Newton solves of the flow, dead at t = 1000 (measured, SSP33). Stops at the
    flow's second step, i.e. after the first system step."""
    domain, rp, rn, tp, tn, so = levee_deck
    rp.coefficients.FCT = True
    rp.coefficients.FCT_TYPE = 'implicit'
    _edge_flux_levee_transport(tp, tn, 4, order, c0=0.0, cin=1.0)
    _direct_solvers(rn, tn)
    so.tnList = [0.0, 1.0e3, 2.0e3, 3.0e3]

    class Seen(Exception):
        pass

    seen = {'calls': 0}
    preStep = TADR.Coefficients.preStep

    def second_preStep(self, t, firstStep=False):
        seen['calls'] += 1
        if seen['calls'] == 2:
            seen['dt'] = self.model.timeIntegration.tLast   # the first step's dt
            seen['c'] = self.model.u[0].dof.copy()
            raise Seen
        return preStep(self, t, firstStep)

    monkeypatch.setattr(TADR.Coefficients, 'preStep', second_preStep)
    with pytest.raises(Seen):
        _run(so, [rp, tp], [rn, tn], 'vdft_levee_first_dt_long')
    assert np.finfo(float).eps / rn.nl_atol_res < seen['dt'] < 1.0e3, seen['dt']
    c = seen['c']
    # the first step is not bound-preserving: it takes the wetting front's fluxes that the CFL of the initial state
    # could not see. Measured on the stored mesh (26411 elements, first dt 0.192 s): c in [0, 4.49] (SSP11),
    # [-2.31, 9.51] (SSP33). The first dt follows the mesh (0.249 s on the 26312-element mesh of the PETSc-built
    # triangle: c in [-7.33, 16.07] for SSP33), so the bound is set above both; the failure it guards against, the
    # interval as the first dt, is c ~ 1e8.
    assert -20.0 <= c.min() and c.max() <= 20.0, (c.min(), c.max(), seen['dt'])


def test_first_step_old_time_mass_is_at_the_coupled_density(levee_deck, monkeypatch):
    """The flow model's history is stored at initialization, at rho_f, before the transport model has set
    the density rho_f (1 + eps c0); preStep copies that density in at the first step. The old-time mass
    m_last has to follow it: with c0 = 0.5, eps = 0.025 the stale one differs from m(u^0) by 1.25 %, and
    (m(u) - m_last)/dt, at the dt = 2.7e-14 the transport's CFL gives, is 1e11-1e13 where the head cannot
    change the mass: Newton's first step goes to |u| ~ 1e14, the iterates grow by 2 per iteration to 1e189
    and the linear solve of the diverged step makes the all-NaN Jacobian of the nightly. Stops at the first
    step's preStep, so it needs no time stepping."""
    from proteus.richards import Richards
    domain, rp, rn, tp, tn, so = levee_deck
    rp.coefficients.FCT = True
    rp.coefficients.FCT_TYPE = 'implicit'
    _edge_flux_levee_transport(tp, tn, 4, 1, c0=0.5, cin=0.5)
    _direct_solvers(rn, tn)
    so.tnList = [0.0, 1.0e-8, 2.0e-8, 10.0]

    class Seen(Exception):
        pass

    seen = {}
    preStep = Richards.Coefficients.preStep

    def checked_preStep(self, t, firstStep=False):
        out = preStep(self, t, firstStep)
        if firstStep:
            flow = self.model
            ti = flow.timeIntegration
            seen['rho'] = (flow.q['rho'].min(), flow.q['rho'].max(), self.rho)
            seen['dof'] = flow.u[0].dof.copy()
            scratch = np.zeros_like(flow.u[0].dof)
            flow.getResidual(flow.u_dof_old.copy(), scratch)
            seen['mismatch'] = np.abs(ti.m_tmp[0] - ti.m_last[0]).max()
            seen['mass'] = np.abs(ti.m_tmp[0]).max()
            # the first residual is the flux terms only: the storage term has nothing to remove
            seen['r'] = np.abs(scratch).max()
            raise Seen
        return out

    monkeypatch.setattr(Richards.Coefficients, 'preStep', checked_preStep)
    with pytest.raises(Seen):
        _run(so, [rp, tp], [rn, tn], 'vdft_levee_first_step_mass')
    lo, hi, rho_f = seen['rho']
    assert hi > 1.01 * rho_f and lo > 1.01 * rho_f      # the coupled density is in: 1 + 0.025 * 0.5
    assert seen['mismatch'] < 1.0e-12 * seen['mass'], seen
    assert seen['r'] < 1.0e3, seen                       # 6e5 to 5e7 per unit dt = 1e-8 before, 1e11-1e13 at 2.7e-14


def test_the_conservative_boundary_flux_is_opt_in():
    """BC_CONS changes the boundary velocity a coupled transport sees, so existing decks keep the pointwise trace
    unless they ask for it."""
    from proteus.richards import Richards
    import inspect
    assert inspect.signature(Richards.Coefficients.__init__).parameters['BC_CONS'].default is False
