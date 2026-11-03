from proteus.default_n import *
from proteus.richards import Richards

try:
    from .column_flow_p import *
except ImportError:
    from column_flow_p import *

timeIntegration = Richards.ThetaScheme
timeOrder = 1
stepController = HeuristicNL_dt_controller
nonlinearIterationsFloor = 4
nonlinearIterationsCeil = 8
dtNLgrowFactor = 2
dtNLreduceFactor = 0.5
dtNLfailureReduceFactor = 0.5
useInitialGuessPredictor = True
stepExact = True
atol_u[0] = 1.0e-3
rtol_u[0] = 1.0e-3

numericalFluxType = Advection_DiagonalUpwind_Diffusion_IIPG_exterior
shockCapturing = None
multilevelNonlinearSolver = Newton
levelNonlinearSolver = Newton
fullNewtonFlag = True
tolFac = 0.0
nl_atol_res = 1.0e-8
maxNonlinearIts = 30
maxLineSearches = 10

matrix = SparseMatrix
multilevelLinearSolver = LU
levelLinearSolver = LU
linTolFac = 0.001
parallelPartitioningType = MeshParallelPartitioningTypes.node
