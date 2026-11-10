from proteus import *
from proteus.default_n import *
from Raingarden_p import *
from proteus.default_so import *
from domain_rg import *


#timeIntegration = BackwardEuler
timeIntegrator = ForwardIntegrator
#timeIntegration = FLCBDF
#stepController  = FLCBDF_controller
#systemStepControllerType = SplitOperator.Sequential_MinFLCBDFModelStep
rtol_u[0] = 1.0e-3
atol_u[0] = 1.0e-3
timeIntegration = BackwardEuler
stepController = HeuristicNL_dt_controller#FixedStep
#stepController = FixedStep
#nDTout = 1#int(T/DT)#int(T/DT) #100#int(T/DT)
#for controlling time stepping
if not galerkin:
    timeIntegration = Richards.ThetaScheme
timeOrder = 1
#stepController = Min_dt_controller
nonlinearIterationsFloor = 4
nonlinearIterationsCeil  = 8 # 8
dtNLgrowFactor = 2
dtNLreduceFactor = 0.5
dtNLfailureReduceFactor = 0.5
useInitialGuessPredictor= True
stepExact = True
#nDTout = 200
#DT = T/nDTout
#tnList = [0.0,1.0e-8]+[i*DT for i  in range(1,nDTout+1)]
atol_u[0] = 1.0e-3
rtol_u[0] = 1.0e-3



massLumping = False

numericalFluxType = None
numericalFluxType = Advection_DiagonalUpwind_Diffusion_IIPG_exterior #need weak for parallel and global conservation

shockCapturing = None
#shockCapturing = ResGradQuadDelayLag_SC(coefficients,nd,shockCapturingFactor=0.75,lag=True,nStepsToDelay=5)

#multilevelNonlinearSolver  = NLStarILU
#multilevelNonlinearSolver  = NLGaussSeidel
#multilevelNonlinearSolver  = NLJacobi
#multilevelNonlinearSolver  = NLNI
#multilevelNonlinearSolver  = FAS
multilevelNonlinearSolver = Newton

#levelNonlinearSolver = NLStarILU
#levelNonlinearSolver = FAS
levelNonlinearSolver = Newton
#levelNonlinearSolver = ExplicitLumpedMassMatrixForRichards
#levelNonlinearSolver = ExplicitConsistentMassMatrixForRichards

#levelNonlinearSolver = NLGaussSeidel
#levelNonlinearSolver = NLJacobi

#nonlinearSmoother = NLStarILU
#nonlinearSmoother = NLGaussSeidel
nonlinearSmoother = NLJacobi

fullNewtonFlag = True

tolFac = 0.0

nl_atol_res = 1.0e-6

maxNonlinearIts = 30#1001
maxLineSearches =0

matrix = SparseMatrix

#multilevelLinearSolver = LU
#multilevelLinearSolver = PETSc
#multilevelLinearSolver = NI
multilevelLinearSolver = KSP_petsc4py


#levelLinearSolver = LU
#levelLinearSolver = PETSc
#levelLinearSolver = MGM
levelLinearSolver = KSP_petsc4py

linearSmoother = Jacobi
linearSmoother = GaussSeidel
linearSmoother = StarILU
linearSmoother = None

linTolFac = 0.001

#conservativeFlux = {0:'pwl-bdm'}
#parallelPartitioningType = MeshParallelPartitioningTypes.element

parallelPartitioningType = MeshParallelPartitioningTypes.node
#default number of layers to use > 1 with element partition means
#C0P1 methods don't need to do communication in global element assembly
#nodal partitioning does not need communication for C0P1 (has overlap 1) regardless
#nLayersOfOverlapForParallel = 0
