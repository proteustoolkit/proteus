from proteus import *
from proteus.default_n import *
from pressure_p import *


mesh = domain.MeshOptions
triangleOptions = triangleOptions
nLayersOfOverlapForParallel = mesh.nLayersOfOverlapForParallel

femSpaces = {0:ct.pbasis}


# stepController  = StepControl.Min_dt_cfl_controller
# runCFL= 0.99
# runCFL= 0.5

stepController=FixedStep

#matrix type
numericalFluxType = NumericalFlux.ConstantAdvection_exterior
#numericalFluxType = NumericalFlux
matrix = LinearAlgebraTools.SparseMatrix

linear_solver_options_prefix = 'pressure_'

if ct.useSuperlu:
    multilevelLinearSolver = LinearSolvers.LU
    levelLinearSolver      = LinearSolvers.LU
else:
    multilevelLinearSolver = KSP_petsc4py
    levelLinearSolver      = KSP_petsc4py
    parallelPartitioningType    = parallelPartitioningType
    nLayersOfOverlapForParallel = nLayersOfOverlapForParallel
    nonlinearSmoother = None
    linearSmoother    = None

multilevelNonlinearSolver = NonlinearSolvers.Newton
levelNonlinearSolver      = NonlinearSolvers.Newton

#linear solve rtolerance

linTolFac = 0.0
l_atol_res = 0.1*ct.pressure_nl_atol_res
tolFac = 0.0
nl_atol_res = ct.pressure_nl_atol_res
maxLineSearches = 0
nonlinearSolverConvergenceTest      = 'r'
levelNonlinearSolverConvergenceTest = 'r'
linearSolverConvergenceTest         = 'r-true'

periodicDirichletConditions=None

conservativeFlux=None
