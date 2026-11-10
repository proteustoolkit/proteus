from proteus import *
from proteus.default_n import *
from proteus.default_so import *

from flow_p import *
from domain_elder import *

timeIntegrator = ForwardIntegrator
rtol_u[0] = 1.0e-3
atol_u[0] = 1.0e-3
timeIntegration = BackwardEuler
stepController = HeuristicNL_dt_controller

if not galerkin:
    timeIntegration = Richards.ThetaScheme

timeOrder = 1
nonlinearIterationsFloor = 8
nonlinearIterationsCeil = 16

dtNLgrowFactor = 1.25 #2
dtNLreduceFactor = 0.25
dtNLfailureReduceFactor = 0.25
useInitialGuessPredictor = True
stepExact = True

massLumping = False

numericalFluxType = Advection_DiagonalUpwind_Diffusion_IIPG_exterior
shockCapturing = None

multilevelNonlinearSolver = Newton
levelNonlinearSolver = Newton
nonlinearSmoother = NLJacobi
fullNewtonFlag = True

tolFac = 0.0
nl_atol_res = 1.0e-7
maxNonlinearIts = 30
maxLineSearches = 10

matrix = SparseMatrix
multilevelLinearSolver = KSP_petsc4py
levelLinearSolver = KSP_petsc4py
linearSmoother = None
linTolFac = 0.001

parallelPartitioningType = MeshParallelPartitioningTypes.element
