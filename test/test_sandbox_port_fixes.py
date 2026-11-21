"""Defects found while porting the proteus-sandbox decks to current proteus.

Each test builds the object the old decks build and checks the behaviour that failed.
"""
import numpy
import pytest

from proteus import MeshTools, SplitOperator, SubgridError, TransportCoefficients


def test_edge_mesh_is_a_simplex_mesh():
    # PostProcessingTools asks every mesh for its type (conservativeFlux in 1D)
    assert MeshTools.EdgeMesh().meshType() == 'simplex'


@pytest.mark.parametrize("cls", [TransportCoefficients.LevelSetCurvatureCoefficients,
                                 TransportCoefficients.LevelSetConservation])
def test_level_set_coefficients_take_the_dimension_from_transport(cls):
    c = cls()  # the decks gave no nd: this asserted
    assert (0, 0) not in c.sdInfo
    c.initializeSparseDiffusion(3)  # Transport calls it with the space dimension
    rowptr, colind = c.sdInfo[(0, 0)]
    assert list(rowptr) == [0, 1, 2, 3] and list(colind) == [0, 1, 2]
    # an explicit nd is kept, and the hook leaves it alone
    d = cls(nd=2)
    d.initializeSparseDiffusion(3)
    assert list(d.sdInfo[(0, 0)][1]) == [0, 1]


def test_level_set_normal_diffusion_is_eps_on_the_diagonal():
    c = TransportCoefficients.LevelSetNormalCoefficients(epsFact=1.0)
    c.eps = 0.5
    nd = 2
    # Transport's full layout: nd*nd entries per point, row-major
    c.sdInfo[(0, 0)] = (numpy.arange(0, nd**2 + 1, nd, dtype='i'),
                        numpy.array([list(range(nd))] * nd, dtype='i').flatten())
    a = numpy.ones((3, 4, nd * nd))  # elements, points, entries; ebq adds one more axis
    c._setDiffusionToEps(a)
    assert numpy.all(a[..., [0, 3]] == 0.5) and numpy.all(a[..., [1, 2]] == 0.0)
    ebq = numpy.ones((3, 2, 4, nd * nd))
    c._setDiffusionToEps(ebq)
    assert numpy.all(ebq[..., [0, 3]] == 0.5) and numpy.all(ebq[..., [1, 2]] == 0.0)
    # dense (sd=False): [..., nd, nd]
    c.sd = False
    dense = numpy.ones((3, 4, nd, nd))
    c._setDiffusionToEps(dense)
    assert numpy.all(dense == 0.5 * numpy.eye(nd))


def test_stokes_subgrid_error_reads_a_full_tensor_with_the_dense_kernel():
    coefficients = TransportCoefficients.Stokes(g=[0.0, -9.8], nd=2)
    sge = SubgridError.StokesASGS_velocity(coefficients, 2)
    # Stokes declares no layout: Transport gives it the full one, nd*nd entries per point
    assert not sge._diagonalDiffusion(numpy.zeros((5, 3, 4)))
    # a diagonal layout, nd entries per point, is what the *_sd kernels index
    assert sge._diagonalDiffusion(numpy.zeros((5, 3, 2)))
    assert not sge._diagonalDiffusion(numpy.zeros((5, 3, 2, 2)))


class _Controller(object):
    maxSolverFailures = 7
    dt_model = 1.0


class _Model(object):
    def __init__(self):
        self.stepController = _Controller()


def test_min_flcbdf_system_controller_without_flcbdf_models_has_maxFailures():
    # a deck that keeps Sequential_MinFLCBDFModelStep with another integrator: the first
    # failure raised AttributeError: maxFailures in retrySequence_modelStepFailure
    so = SplitOperator.Sequential_MinFLCBDFModelStep([_Model(), _Model()])
    assert so.maxFailures == 7
    so.retry = False
    assert so.retrySequence_modelStepFailure() is False


def _threephase(**kw):
    return TransportCoefficients.ThreephaseNavierStokes_ST_LS_SO(nd=2, **kw)


def test_threephase_exterior_boundary_without_fluid_profile():
    c = _threephase()
    cebqe = {('u', 0): numpy.zeros((2, 3)), ('f', 0): numpy.zeros((2, 3, 2)), 'x': numpy.zeros((2, 3, 3))}
    c.initializeGlobalExteriorElementBoundaryQuadrature(0.0, cebqe)  # called None: TypeError
    assert numpy.all(c.ebqe_phi == 1.0)


def test_threephase_exterior_boundary_uses_the_fluid_profile():
    def profile(t, x, phi, n):
        phi.flat[:] = x[..., 0].flat
        n.flat[:] = 0.0
    c = _threephase(defaultFluidProfile=profile)
    x = numpy.zeros((2, 3, 3))
    x[..., 0] = numpy.arange(6.0).reshape(2, 3)
    cebqe = {('u', 0): numpy.zeros((2, 3)), ('f', 0): numpy.zeros((2, 3, 2)), 'x': x}
    c.initializeGlobalExteriorElementBoundaryQuadrature(0.0, cebqe)  # was skipped
    assert numpy.all(c.ebqe_phi == x[..., 0])


class _Integrator(object):
    """A time integrator whose solver state carries its step, as FLCBDF's does."""
    def __init__(self):
        self.dt = None

    def state(self):
        return {"dt": self.dt}

    def set_state(self, state):
        self.dt = state["dt"]


class _LevelModel(object):
    def __init__(self):
        self.timeIntegration = _Integrator()
        self.coefficients = type("C", (), {"nc": 0})()
        self.u = {}
        self.q = {}


class _StepController(object):
    def __init__(self, model):
        self.model = model
        self.t_model_last = 0.0
        self.dt_model = None

    def set_dt_allLevels(self):
        self.t_model = self.t_model_last + self.dt_model
        for lm in self.model.levelModelList:
            lm.timeIntegration.dt = self.dt_model


class _TransportModel(object):
    def __init__(self):
        self.levelModelList = [_LevelModel()]
        self.uList = [numpy.zeros(1)]
        self.stepController = _StepController(self)


def test_a_retry_keeps_the_step_the_controller_cut_to():
    from proteus.NumericalSolution import NS_base
    ns = NS_base.__new__(NS_base)
    model = _TransportModel()
    ns.modelList = [model]
    model.stepController.dt_model = 1.0
    model.stepController.set_dt_allLevels()
    ns.saveSystemStepState()          # the start of the system step, dt = 1
    # the attempt fails; the controller cuts the step (choose_dt_system) before the restore
    model.stepController.dt_model = 0.25
    model.stepController.set_dt_allLevels()
    ns.restoreSystemStepState()
    # the restored solver state carried dt = 1: the retry repeated the failed step
    assert model.levelModelList[0].timeIntegration.dt == 0.25
    assert model.stepController.t_model == 0.25


def test_parun_rejects_an_old_sso_file_with_a_message(tmp_path):
    import shutil
    import subprocess
    import sys
    import os
    # the repository's script (an installed copy is made at install time and can be older)
    parun = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "parun")
    if not os.path.exists(parun):
        parun = shutil.which("parun")
    if parun is None:
        pytest.skip("parun is not on PATH")
    sso = tmp_path / "x.sso"
    sso.write_text("x_p.py x_n.py\n")
    out = subprocess.run([sys.executable, parun, str(sso)], capture_output=True, text=True,
                         cwd=str(tmp_path), timeout=600)
    # it died with IndexError on sList[0]
    assert out.returncode != 0
    assert ".sso text files are no longer read" in out.stderr


def test_scalar_heaviside_domain_integral_calls_its_kernel_with_the_counts_first():
    from proteus import Norms
    dV = numpy.full((3, 2), 0.5)
    u = numpy.array([[1.0, -1.0], [2.0, 3.0], [-1.0, -2.0]])
    # the kernel adds dV where u > 0 (TypeError before: the arguments were in the old order)
    assert Norms.scalarHeavisideDomainIntegral(dV, u) == pytest.approx(1.5)
    assert Norms.scalarHeavisideDomainIntegral(dV, u, nElements=1) == pytest.approx(0.5)


def test_redistance_level_set_sandf_can_be_built():
    c = TransportCoefficients.RedistanceLevelSetSandF(epsFact=1.5, nModelId=1, rdModelId=0)
    assert c.nModelId == 1 and c.rdModelId == 0


def test_missing_subgrid_error_coefficients_alias_the_coefficients():
    from proteus.Transport import OneLevelTransport
    da = numpy.ones((2, 3, 1))
    lagged = numpy.zeros((2, 3, 1))
    cq = {('da', 0, 0, 0): da, ('dH', 0, 0): numpy.ones((2, 3, 1)), ('dH_sge', 0, 0): lagged}
    OneLevelTransport._aliasMissingSubgridErrorCoefficients(OneLevelTransport.__new__(OneLevelTransport), cq)
    assert cq[('da_sge', 0, 0, 0)] is da          # missing: aliased, as a subgrid error that does not lag
    assert cq[('dH_sge', 0, 0)] is lagged         # set up by a subgrid error: left alone


def test_exterior_pressure_integrals_by_boundary_flag():
    # AuxiliaryVariables.BoundaryPressure calls it; the wrapper was lost when cfemIntegrals became Cython
    from proteus import cfemIntegrals, cmeshTools
    flags = numpy.array([0, 1, 0, 2], 'i')                        # by element boundary
    exterior = numpy.array([1, 3], cmeshTools.INDEX_DTYPE)        # exterior boundaries 1 (flag 1) and 3 (flag 2)
    p = numpy.array([[2.0, 4.0], [1.0, 1.0]])                    # pressure at 2 points on each
    dS = numpy.array([[0.5, 0.5], [0.25, 0.25]])
    P = numpy.zeros(3)
    A = numpy.zeros(3)
    cfemIntegrals.accumulateExteriorElementPressureIntegrals(flags, exterior, p, dS, P, A)
    assert list(P) == [0.0, 3.0, 0.5] and list(A) == [0.0, 1.0, 0.5]


def test_nc_level_set_kernel_takes_the_space_dimension_from_the_shape():
    # the wrappers passed dH[dH.ndim-1] (an element) where they meant dH.shape[dH.ndim-1]
    from proteus.ctransportCoefficients import ncLevelSetCoefficientsEvaluate
    nE, nq, nd = 2, 3, 2
    v = numpy.ones((nE, nq, nd))
    u = numpy.linspace(0.0, 1.0, nE * nq).reshape(nE, nq)
    grad_u = numpy.zeros((nE, nq, nd))
    grad_u[..., 0] = 2.0
    m, dm, H = (numpy.zeros((nE, nq)) for _ in range(3))
    dH = numpy.zeros((nE, nq, nd))
    ncLevelSetCoefficientsEvaluate(v, u, grad_u, m, dm, H, dH)
    assert numpy.allclose(m, u) and numpy.allclose(dm, 1.0)
    assert numpy.allclose(H, 2.0) and numpy.allclose(dH, v)


def test_hamilton_jacobi_shock_capturing_passes_its_flag_as_a_c_char():
    # cshockCapturing.calculateNumericalDiffusionHJ takes a C char: a str raised "an integer is required"
    from proteus import ShockCapturing
    coefficients = TransportCoefficients.NCLevelSetCoefficients()
    sc = ShockCapturing.HamiltonJacobi_SC(coefficients, 1, shockCapturingFlag='2')
    assert sc.shockCapturingFlag == ord('2')


@pytest.mark.parametrize("name", ["ConservationHistoryLS", "ConservationHistoryMC"])
def test_conservation_history_attach_model_returns_itself(name, tmp_path, monkeypatch):
    # NumericalSolution keeps what attachModel returns: these returned None
    from proteus import AuxiliaryVariables, Profiling
    monkeypatch.setattr(Profiling, "logDir", str(tmp_path))
    av = getattr(AuxiliaryVariables, name)("square")
    assert av.attachModel(object(), None) is av


def test_conservation_history_writes_what_the_history_gained(tmp_path, monkeypatch):
    # calculate_init runs before attachModels starts the coefficients' history: nothing to write
    # then (AttributeError before); the first call after it writes the initial entry too
    from proteus import AuxiliaryVariables, Profiling
    monkeypatch.setattr(Profiling, "logDir", str(tmp_path))
    coefficients = type("C", (), {})()
    model = type("M", (), {"levelModelList": [type("L", (), {"coefficients": coefficients})()]})()
    av = AuxiliaryVariables.ConservationHistoryLS("h").attachModel(model, None)
    av.calculate_init()
    for name in ("lsGlobalMassArray", "lsGlobalMassErrorArray", "fluxArray", "timeArray"):
        setattr(coefficients, name, [0.0, 1.0])
    av.calculate()
    coefficients.timeArray.append(2.0)
    for name in ("lsGlobalMassArray", "lsGlobalMassErrorArray", "fluxArray"):
        getattr(coefficients, name).append(3.0)
    av.calculate()
    times = [float(x) for x in open(str(tmp_path / "htime")).read().split()]
    assert times == [0.0, 1.0, 2.0]


def test_two_phase_ns_on_element_boundaries_without_level_set_values_raises():
    # the kernel was handed None for phi (SIGSEGV): the so module must ask for needEBQ
    c = TransportCoefficients.TwophaseNavierStokes_ST_LS_SO(nd=3, g=[0.0, 0.0, -9.8], LS_model=1)
    c.q_phi = numpy.zeros((2, 3))
    c.ebqe_phi = numpy.zeros((4, 2))
    c.ebq_phi = c.ebq_n = c.ebq_kappa = None
    with pytest.raises(RuntimeError, match="needEBQ"):
        c.evaluate(0.0, {('u', 0): numpy.zeros((2, 3, 3))})


def test_level_set_conservation_counts_no_flux_for_vof_coefficients_without_one():
    c = TransportCoefficients.LevelSetConservation(nd=2)
    c.vofModel = type("M", (), {"coefficients": object()})()   # CLevelSetCoefficients keeps no fluxIntegral
    assert c._vofFluxIntegral() == 0.0
    c.vofModel.coefficients = type("V", (), {"fluxIntegral": 2.5})()
    assert c._vofFluxIntegral() == 2.5


def test_exterior_lesaint_raviart_flux_writes_only_the_exterior_boundaries():
    # the wrapper took the number of all element boundaries for the exterior ones: heap corruption
    from proteus import cnumericalFlux, cmeshTools
    nEB, nExt, nq, nd = 5, 2, 1, 1
    exterior = numpy.array([0, 4], cmeshTools.INDEX_DTYPE)
    ebElements = numpy.zeros((nEB, 2), cmeshTools.INDEX_DTYPE)
    ebLocal = numpy.zeros((nEB, 2), 'i')
    isDOF = numpy.zeros((nExt, nq), 'i')
    inflow = numpy.zeros((nExt, nq), 'i')
    n = numpy.ones((nExt, nq, nd))
    bc_u, bc_H, u, H = (numpy.zeros((nExt, nq)) for _ in range(4))
    bc_dH, dH = numpy.zeros((nExt, nq, nd)), numpy.ones((nExt, nq, nd))
    buf = numpy.full(nEB * nq, -7.0)
    flux = buf[:nExt * nq].reshape(nExt, nq)
    dbuf = numpy.full(nEB * nq, -7.0)
    dflux = dbuf[:nExt * nq].reshape(nExt, nq)
    cnumericalFlux.calculateExteriorLesaintRaviartNumericalFlux(0, exterior, ebElements, ebLocal, isDOF, inflow, n,
                                                               bc_u, bc_H, bc_dH, u, H, dH, flux, dflux)
    assert numpy.all(buf[nExt * nq:] == -7.0) and numpy.all(dbuf[nExt * nq:] == -7.0)


def test_fixed_step_controller_without_a_fixed_step_steps_to_the_output_time():
    # Sequential_FixedStep (SO_base) took dt_system_fixed, None by default: TypeError float + None
    from proteus import SplitOperator

    class _Controller(object):
        def set_dt_allLevels(self):
            pass

        def setSubsteps(self, tList):
            self.substeps = tList
    so = SplitOperator.SO_base.__new__(SplitOperator.SO_base)
    so.dt_system_fixed = None
    so.stepSchedule = None
    so.system = type("S", (), {"name": "s"})()
    so.modelList = [type("M", (), {"stepController": _Controller()})()]
    so.initialize_dt_system(0.0, 0.5)
    assert so.dt_system == 0.5 and so.t_system == 0.5


def test_two_phase_ns_ls_so_calls_its_kernel():
    # evaluate called self.TwophaseNavierStokes_LS_SO_2D_Evaluate, a module function: AttributeError
    c = TransportCoefficients.TwophaseNavierStokes_LS_SO(nd=2, g=[0.0, -9.8], LS_model=None)
    c.eps = 0.1
    shape = (2, 3)
    c.initializeElementQuadrature(0.0, {('u', 0): numpy.zeros(shape)})
    q = {('u', i): numpy.zeros(shape) for i in range(3)}
    q[('grad(u)', 0)] = numpy.zeros(shape + (2,))
    for k in [('m', 1), ('dm', 1, 1), ('m', 2), ('dm', 2, 2), ('r', 1), ('r', 2), ('H', 1), ('H', 2)]:
        q[k] = numpy.zeros(shape)
    for k in [('f', 0), ('df', 0, 1), ('df', 0, 2), ('f', 1), ('df', 1, 1), ('df', 1, 2), ('f', 2), ('df', 2, 1),
              ('df', 2, 2), ('dH', 1, 0), ('dH', 2, 0)]:
        q[k] = numpy.zeros(shape + (2,))
    q[('a', 1, 1)] = numpy.zeros(shape + (2, 2))
    q[('a', 2, 2)] = numpy.zeros(shape + (2, 2))
    c.evaluate(0.0, q)
    assert numpy.all(q[('dm', 1, 1)] > 0.0)   # the kernel ran and wrote the mass derivative


def test_poly_file_comments_may_follow_a_value(tmp_path):
    # triangle reads '#' anywhere on a line as a comment; the reader took '0#enforce' for a flag
    from proteus import Domain
    (tmp_path / "c.poly").write_text("# a square\n4 2 0 1\n1 0 0 1#corner\n2 1 0 1\n3 1 1 1\n4 0 1 0#enforce\n"
                                     "4 1\n1 1 2 1 #bottom\n2 2 3 1\n3 3 4 1\n4 4 1 1\n0 #holes\n0 #regions\n")
    d = Domain.PlanarStraightLineGraphDomain(fileprefix=str(tmp_path / "c"), name="c")
    assert d.vertexFlags == [1, 1, 1, 0] and len(d.segments) == 4


def test_volume_averaged_ns_boundary_porosity_without_solution_values_on_ebq_global():
    c = TransportCoefficients.VolumeAveragedNavierStokesFullDevStress()
    cebq = {('u', 0): numpy.zeros((3, 3, 2)), 'x': numpy.zeros((3, 3, 2, 3))}
    cebq_global = {'x': numpy.zeros((5, 2, 3))}   # no ('u',0) there now: KeyError
    c.initializeElementBoundaryQuadrature(0.0, cebq, cebq_global)
    assert c.ebq_global_porosity.shape == (5, 2) and numpy.all(c.ebq_global_porosity == 1.0)


def test_l2_projection_of_a_sparse_layout_tensor_averages_each_entry():
    # rank 2 read [element, point, nd, nd] and took the entry count for nd: heap corruption
    from proteus.ctransportCoefficients import L2projectEvaluate
    dV = numpy.array([[1.0, 3.0]])
    buf = numpy.full(2 * 4 + 5, -7.0)
    r = buf[:8].reshape(1, 2, 4)
    r[0, 0] = [1.0, 2.0, 3.0, 4.0]
    r[0, 1] = [5.0, 6.0, 7.0, 8.0]
    L2projectEvaluate(2, dV, r)
    assert numpy.allclose(r[0, 0], [4.0, 5.0, 6.0, 7.0]) and numpy.allclose(r[0, 1], r[0, 0])
    assert numpy.all(buf[8:] == -7.0)


def _twophase_stokes_q(shape, nd=2, mass=True):
    q = {('u', i): numpy.zeros(shape) for i in range(nd + 1)}
    q[('grad(u)', 0)] = numpy.zeros(shape + (nd,))
    for ci in range(1, nd + 1):
        if mass:
            q[('m', ci)] = numpy.zeros(shape)
            q[('dm', ci, ci)] = numpy.zeros(shape)
        q[('a', ci, ci)] = numpy.zeros(shape + (nd * nd,))
        q[('r', ci)] = numpy.zeros(shape)
        q[('H', ci)] = numpy.zeros(shape)
        q[('dH', ci, 0)] = numpy.zeros(shape + (nd,))
    q[('f', 0)] = numpy.zeros(shape + (nd,))
    for cj in range(1, nd + 1):
        q[('df', 0, cj)] = numpy.zeros(shape + (nd,))
    return q


def test_steady_two_phase_stokes_evaluates_without_mass_terms():
    # steady=True leaves out the mass terms; evaluate read c[('m',1)]: KeyError
    c = TransportCoefficients.TwophaseStokes_LS_SO(g=[0.0, -9.8], nd=2, steady=True)
    c.eps = 0.1
    shape = (2, 3)
    c.q_phi = numpy.ones(shape)
    c.ebqe_phi = numpy.ones((4, 2))
    q = _twophase_stokes_q(shape, mass=False)
    c.evaluate(0.0, q)
    assert numpy.all(q[('a', 1, 1)][..., 0] > 0.0)   # the kernel ran


def test_two_phase_stokes_without_level_set_values_at_its_points_raises():
    # a level set on another quadrature (or with mass lumping) gave the kernel None or a short array: SIGSEGV
    c = TransportCoefficients.TwophaseStokes_LS_SO(g=[0.0, -9.8], nd=2)
    c.eps = 0.1
    c.q_phi = numpy.ones((2, 7))
    c.ebqe_phi = numpy.ones((4, 2))
    c.ebq_phi = None
    with pytest.raises(RuntimeError, match="quadrature"):
        c.evaluate(0.0, _twophase_stokes_q((2, 4)))


def test_moving_domain_wrappers_lost_in_the_move_to_cython():
    # a moving-mesh model (threep_cylinder_md_2d) stopped at each of these: AttributeError
    from proteus import cfemIntegrals, cmeshTools
    # copy the mesh velocity on an interior boundary from its left element to its right one
    xt = numpy.zeros((2, 2, 1, 3))
    xt[0, 0, 0] = [1.0, 2.0, 3.0]
    cfemIntegrals.copyLeftElementBoundaryInfo_movingDomain(numpy.array([[0, 1]], cmeshTools.INDEX_DTYPE),
                                                           numpy.array([[0, 1]], 'i'),
                                                           numpy.array([], cmeshTools.INDEX_DTYPE),
                                                           numpy.array([0], cmeshTools.INDEX_DTYPE), xt)
    assert list(xt[1, 1, 0]) == [1.0, 2.0, 3.0]
    # df -= dm xt
    x = numpy.zeros((1, 2, 3))
    x[..., 0], x[..., 1] = 1.0, 2.0
    df = numpy.zeros((1, 2, 2))
    cfemIntegrals.update_df_movingDomain(x, numpy.full((1, 2), 3.0), df)
    assert numpy.allclose(df, [[[-3.0, -6.0], [-3.0, -6.0]]])


def test_moving_domain_boundary_maps_match_the_fixed_ones_when_the_mesh_is_still():
    from proteus import cfemIntegrals, cmeshTools
    nodes = numpy.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    l2g = numpy.array([[0, 1, 2]], cmeshTools.INDEX_DTYPE)
    nq = 2
    grad_psi = numpy.zeros((3, nq, 3, 2))
    grad_psi[:, :, 0] = [-1.0, -1.0]
    grad_psi[:, :, 1] = [1.0, 0.0]
    grad_psi[:, :, 2] = [0.0, 1.0]
    normals = numpy.array([[0.70710678, 0.70710678], [-1.0, 0.0], [0.0, -1.0]])
    jacobians = numpy.array([[[-1.0], [1.0]], [[0.0], [-1.0]], [[1.0], [0.0]]])

    def maps(moving):
        out = [numpy.zeros((1, 3, nq, 2, 2)), numpy.zeros((1, 3, nq, 1, 1)), numpy.zeros((1, 3, nq)),
               numpy.zeros((1, 3, nq, 2))]
        if moving:
            cfemIntegrals.parametricMaps_getJacobianValuesTrace_movingDomain(numpy.zeros((1, 3, nq, 3)), grad_psi,
                                                                             normals, jacobians, l2g, nodes, *out)
        else:
            cfemIntegrals.parametricMaps_getJacobianValuesTrace(grad_psi, normals, jacobians, l2g, nodes, *out)
        return out
    for a, b in zip(maps(True), maps(False)):
        assert numpy.allclose(a, b)
    assert numpy.all(maps(True)[2] > 0.0)
