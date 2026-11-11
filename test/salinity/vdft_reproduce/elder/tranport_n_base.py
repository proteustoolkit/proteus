from __future__ import print_function
from __future__ import absolute_import
from builtins import range

from proteus import *
from proteus.default_n import *
from proteus.mprans import TADR

from domain_elder import *
from tranport_p import *

parallel = True

multilevelNonlinearSolver = Newton
if coefficients.STABILIZATION_TYPE == 5:
    # ImplicitEV: backward-Euler + Newton on the edge-based upwind operator.
    # TADR rejects SSP time integration and the explicit lumped-mass solvers
    # for this type.  The low-order operator is an M-matrix, so dt is limited
    # by accuracy and Newton convergence rather than by the CFL condition.
    levelNonlinearSolver = Newton
    maxLineSearches = 10
    fullNewtonFlag = True
    updateJacobian = True
    timeIntegration = BackwardEuler
elif coefficients.STABILIZATION_TYPE in [0, 1]:
    levelNonlinearSolver = Newton
    maxLineSearches = 0
    fullNewtonFlag = True
    updateJacobian = True
    timeIntegration = BackwardEuler_cfl
elif coefficients.STABILIZATION_TYPE == 1:
    levelNonlinearSolver = TwoStageNewton
    fullNewtonFlag = False
    updateJacobian = False
    timeIntegration = BackwardEuler_cfl
else:
    fullNewtonFlag = False
    updateJacobian = False
    timeIntegration = TADR.RKEV
    if coefficients.LUMPED_MASS_MATRIX is True:
        levelNonlinearSolver = ExplicitLumpedMassMatrix
    else:
        levelNonlinearSolver = ExplicitConsistentMassMatrixForVOF

SSPOrder = 1
if coefficients.STABILIZATION_TYPE == 5:
    # dt is not CFL-bound here; grow/shrink it on the Newton iteration count
    # like the flow model does.
    stepController = HeuristicNL_dt_controller
    # The inexact Newton (frozen a_ij) converges linearly and needs ~4 iters,
    # so a floor of 4 would sit exactly in the controller's no-op band
    # (grow iff nnl < floor, shrink iff nnl > ceil) and pin dt forever.
    nonlinearIterationsFloor = 10
    nonlinearIterationsCeil = 20
    dtNLgrowFactor = 2.0
    dtNLreduceFactor = 0.50
    dtNLfailureReduceFactor = 0.25
    useInitialGuessPredictor = True
    stepExact = True
    tolFac = 0.0
    nl_atol_res = 1.0e-7
    maxNonlinearIts = 30
else:
    stepController = Min_dt_controller
    runCFL = 0.5
timeOrder = SSPOrder
nStagesTime = SSPOrder
pDegree_tadr = 1

numericalFluxType = TADR.NumericalFlux
shockCapturing = TADR.ShockCapturing(
    coefficients,
    nd,
    shockCapturingFactor=shockCapturingFactor_tadr,
    lag=lag_shockCapturing_tadr,
)

matrix = SparseMatrix
if parallel:
    multilevelLinearSolver = KSP_petsc4py
    levelLinearSolver = KSP_petsc4py
    linear_solver_options_prefix = "tadr_"
    linearSolverConvergenceTest = "r-true"
else:
    multilevelLinearSolver = LU
    levelLinearSolver = LU

if checkMass:
    auxiliaryVariables = [MassOverRegion()]
