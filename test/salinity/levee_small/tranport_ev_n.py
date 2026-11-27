import os

from tranport_n import *  # noqa: F401,F403
from tranport_ev_p import coefficients, STAB  # the scheme chosen by LEVEE_STAB (not the p module's defaults: they reset numericalFluxType)
from proteus import StepControl, NonlinearSolvers, TimeIntegration

stepController = StepControl.Min_dt_controller
multilevelNonlinearSolver = NonlinearSolvers.Newton
tolFac = 0.0
nl_atol_res = 1.0e-10
maxLineSearches = 0
# the edge-based kernels read the shock-capturing lag and factor even when the factor is 0
shockCapturing = TADR.ShockCapturing(coefficients, nd, shockCapturingFactor=0.0, lag=True)
timeOrder = nStagesTime = 1
runCFL = 0.5
if STAB in (2, 3, 4):
    timeIntegration = TADR.RKEV
    levelNonlinearSolver = NonlinearSolvers.ExplicitLumpedMassMatrix
    fullNewtonFlag = False
    updateJacobian = False
else:
    timeIntegration = TimeIntegration.BackwardEuler_cfl
    levelNonlinearSolver = NonlinearSolvers.Newton
    fullNewtonFlag = True
    updateJacobian = True
    maxNonlinearIts = 100
