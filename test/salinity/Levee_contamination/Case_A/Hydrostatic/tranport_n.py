from __future__ import print_function
from __future__ import absolute_import
from builtins import range
from proteus import *
from proteus.default_n import *
from domain_rg import *
from tranport_p import *

# Compatible transport (STABILIZATION_TYPE='Compatible'): one backward-Euler step per flow step,
# solved to the residual tolerance so the solute budget closes; serial only for now.
parallel = False

timeIntegration = TADR.CompatibleBackwardEuler
stepController = Min_dt_controller
timeOrder = 1
nStagesTime = 1

multilevelNonlinearSolver = Newton
levelNonlinearSolver = Newton
fullNewtonFlag = True
updateJacobian = True
tolFac = 0.0
nl_atol_res = 1.0e-10
maxNonlinearIts = 20
maxLineSearches = 0

numericalFluxType = TADR.NumericalFlux

shockCapturing = None

matrix = SparseMatrix
# direct solver: the tadr_-prefixed KSP would ignore a global PETSC_OPTIONS
multilevelLinearSolver = LU
levelLinearSolver = LU

if checkMass:
    auxiliaryVariables = [MassOverRegion()]
