from proteus import *
from proteus.default_n import *
from pressureInitial_p import *


mesh = domain.MeshOptions
triangleOptions = triangleOptions
nLayersOfOverlapForParallel = mesh.nLayersOfOverlapForParallel

femSpaces = {0:ct.pbasis}

stepController=FixedStep

#numericalFluxType = NumericalFlux.ConstantAdvection_Diffusion_SIPG_exterior #weak boundary conditions (upwind ?)
matrix = LinearAlgebraTools.SparseMatrix

#linearSmoother    = LinearSolvers.NavierStokesPressureCorrection # pure neumann laplacian solver
#multilevelLinearSolver = LinearSolvers.KSP_petsc4py
#levelLinearSolver = LinearSolvers.KSP_petsc4py
#linearSmoother    = None
#multilevelLinearSolver = LinearSolvers.LU
#levelLinearSolver      = LinearSolvers.LU
numericalFluxType = NumericalFlux.ConstantAdvection_exterior
#numericalFluxType = NumericalFlux

linear_solver_options_prefix = 'pinit_'

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
l_atol_res = 0.01*ct.phi_nl_atol_res
tolFac = 0.0
nl_atol_res = ct.phi_nl_atol_res
nonlinearSolverConvergenceTest = 'r'
levelNonlinearSolverConvergenceTest = 'r'
linearSolverConvergenceTest             = 'r-true'
maxLineSearches=0
periodicDirichletConditions=None

conservativeFlux=None
