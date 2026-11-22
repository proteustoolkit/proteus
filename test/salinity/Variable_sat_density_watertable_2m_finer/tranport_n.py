from __future__ import print_function
from __future__ import absolute_import
from builtins import range

from proteus import *
from proteus.default_n import *
from proteus.kernels import TADR

from domain_liu import *
from tranport_p import *

parallel = True

multilevelNonlinearSolver = Newton
if coefficients.STABILIZATION_TYPE in [0, 1]:
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
stepController = Min_dt_controller
runCFL = 0.2
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
