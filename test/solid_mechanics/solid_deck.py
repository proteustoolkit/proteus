"""Shared set-up of proteus.elastoplastic problems on one-element-thick tetrahedral slabs (plane strain)."""
import types
import numpy as np
from proteus import default_p, default_n, default_s, default_so
from proteus import Domain, FemTools, Quadrature, NumericalFlux, StepControl, TimeIntegration, NonlinearSolvers
from proteus import LinearSolvers, LinearAlgebraTools, MeshTools, NumericalSolution
from proteus.elastoplastic import ElastoPlastic

MCDP, VON_MISES, TRESCA, DRUCKER_PRAGER, MOHR_COULOMB = 1, 2, 3, 4, 5


def zero(x, t):
    return 0.0


def _module(name, base):
    m = types.ModuleType(name)
    m.__dict__.update({k: v for k, v in base.__dict__.items() if not k.startswith('__')})
    return m


def material(E, nu, cohesion, model=MCDP, phi=0.0, psi=0.0, density=0.0):
    """One material: (model types, model parameters) for ElastoPlastic.Coefficients. P_a = 0, so the modulus is constant.

    For von Mises and Tresca the uniaxial yield stress is 2*cohesion."""
    params = np.zeros((1, 14), 'd')
    D = E/((1.0 + nu)*(1.0 - 2.0*nu))
    params[0, :5] = (E, nu, D*(1.0 - nu), D*nu, 0.5*D*(1.0 - 2.0*nu))
    params[0, 5:8] = (phi, cohesion, psi)
    params[0, 13] = density
    return np.array([model], 'i'), params


def physics(name, fileprefix, boundaryFlags, mat, g=(0.0, 0.0, 0.0), loadStepping=False):
    p = _module(name, default_p)
    p.nd = 3
    p.domain = Domain.MeshTetgenDomain(fileprefix)
    p.domain.boundaryFlags = boundaryFlags
    p.genMesh = False
    p.LevelModelType = ElastoPlastic.LevelModel
    p.coefficients = ElastoPlastic.Coefficients(modelType_block=mat[0], modelParams_block=mat[1],
                                                g=list(g), rhow=0.0, pa=0.0, nd=3, SRF=1.0,
                                                loadStepping=loadStepping)
    p.initialConditions = None
    p.analyticalSolution = None
    p.T = 1.0
    return p


def boundary_conditions(p, dirichlet):
    """dirichlet: {component: {boundary flag: function(x, t)}}. Every other (component, flag) pair is traction free."""
    def setter(ci):
        def getDBC(x, flag):
            return dirichlet.get(ci, {}).get(flag)
        return getDBC

    def flux(ci):
        def getStress(x, flag):
            if flag in dirichlet.get(ci, {}):
                return None
            return zero
        return getStress

    p.dirichletConditions = dict((ci, setter(ci)) for ci in range(3))
    p.stressFluxBoundaryConditions = dict((ci, flux(ci)) for ci in range(3))


def numerics(name, order=1, atol=1.0e-5, tnList=(0.0, 1.0)):
    """atol: with E ~ 2e5 the residual floor is about 1e-7, so smaller absolute tolerances fail and retry the step."""
    n = _module(name, default_n)
    n.timeIntegration = TimeIntegration.NoIntegration
    n.stepControl = StepControl.Newton_controller
    n.tnList = list(tnList)
    if order == 1:
        space = FemTools.C0_AffineLinearOnSimplexWithNodalBasis
        n.elementQuadrature = Quadrature.SimplexGaussQuadrature(3, 3)
        n.elementBoundaryQuadrature = Quadrature.SimplexGaussQuadrature(2, 3)
    else:
        space = FemTools.C0_AffineQuadraticOnSimplexWithNodalBasis
        n.elementQuadrature = Quadrature.SimplexGaussQuadrature(3, 6)
        n.elementBoundaryQuadrature = Quadrature.SimplexGaussQuadrature(2, 6)
    n.femSpaces = {0: space, 1: space, 2: space}
    n.subgridError = None
    n.massLumping = False
    n.numericalFluxType = NumericalFlux.Stress_IIPG_exterior
    n.shockCapturing = None
    n.multilevelNonlinearSolver = NonlinearSolvers.Newton
    n.levelNonlinearSolver = NonlinearSolvers.Newton
    n.fullNewtonFlag = True
    n.maxNonlinearIts = 25
    n.maxLineSearches = 0
    n.tolFac = 0.0
    n.atol = atol
    n.nl_rtol_res = 0.0
    n.nl_atol_res = atol
    n.matrix = LinearAlgebraTools.SparseMatrix
    n.multilevelLinearSolver = LinearSolvers.LU
    n.levelLinearSolver = LinearSolvers.LU
    n.linTolFac = 0.0
    n.conservativeFlux = None
    n.parallelPartitioningType = MeshTools.MeshParallelPartitioningTypes.node
    n.nLayersOfOverlapForParallel = 1
    return n


def run(p, n, name, opts):
    """Solve and return (numerical solution, finest level model)."""
    so = _module(name + '_so', default_so)
    so.name = p.name = name
    so.sList = [default_s]
    so.tnList = n.tnList
    ns = NumericalSolution.NS_base(so, [p], [n], so.sList, opts)
    ns.calculateSolution(so.name)
    return ns, ns.modelList[0].levelModelList[-1]
