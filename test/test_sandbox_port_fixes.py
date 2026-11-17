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
