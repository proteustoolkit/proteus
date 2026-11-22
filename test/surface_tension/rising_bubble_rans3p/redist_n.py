from proteus import *
try:
    from .redist_p import *
    from .risingBubble import *
except:
    from redist_p import *
    from risingBubble import *
from proteus.kernels import RDLS
tolFac = 0.0
nl_atol_res = rd_nl_atol_res

linTolFac = 0.01
l_atol_res = 0.01*rd_nl_atol_res

if redist_Newton:
    timeIntegration = NoIntegration
    stepController = Newton_controller
    maxNonlinearIts = 50
    maxLineSearches = 0
    nonlinearSolverConvergenceTest = 'rits'
    levelNonlinearSolverConvergenceTest = 'rits'
    linearSolverConvergenceTest = 'r-true'
    useEisenstatWalker = False
else:
    timeIntegration = BackwardEuler_cfl
    stepController = RDLS.PsiTC
    runCFL=2.0
    psitc['nStepsForce']=3
    psitc['nStepsMax']=50
    psitc['reduceRatio']=10.0
    psitc['startRatio']=1.0
    rtol_res[0] = 0.0
    atol_res[0] = rd_nl_atol_res
    useEisenstatWalker = False#True
    maxNonlinearIts = 1
    maxLineSearches = 0
    nonlinearSolverConvergenceTest = 'rits'
    levelNonlinearSolverConvergenceTest = 'rits'
    linearSolverConvergenceTest = 'r-true'

femSpaces = {0:pbasis}

massLumping       = False
numericalFluxType = DoNothing
conservativeFlux  = None
subgridError      = RDLS.SubgridError(coefficients,nd)
shockCapturing    = RDLS.ShockCapturing(coefficients,nd,shockCapturingFactor=rd_shockCapturingFactor,lag=rd_lag_shockCapturing)

fullNewtonFlag = True
multilevelNonlinearSolver  = Newton
levelNonlinearSolver       = TwoStageNewton

nonlinearSmoother = NLGaussSeidel
linearSmoother    = None

matrix = SparseMatrix

if useOldPETSc:
    multilevelLinearSolver = PETSc
    levelLinearSolver      = PETSc
else:
    multilevelLinearSolver = KSP_petsc4py
    levelLinearSolver      = KSP_petsc4py

if useSuperlu:
    multilevelLinearSolver = LU
    levelLinearSolver      = LU

linear_solver_options_prefix = 'rdls_'
