from proteus.default_n import *
from proteus.mprans import TADR

try:
    from .column_transport_p import *
except ImportError:
    from column_transport_p import *

# compatible coupling: backward Euler + Newton, one transport step per flow step
timeIntegration = TADR.CompatibleBackwardEuler
stepController = Min_dt_controller
numericalFluxType = TADR.NumericalFlux
shockCapturing = None
multilevelNonlinearSolver = Newton
levelNonlinearSolver = Newton
fullNewtonFlag = True
tolFac = 0.0
nl_atol_res = 1.0e-10
maxNonlinearIts = 20
maxLineSearches = 0

matrix = SparseMatrix
multilevelLinearSolver = LU
levelLinearSolver = LU
