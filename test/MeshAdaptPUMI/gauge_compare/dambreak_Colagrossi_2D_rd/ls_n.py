from proteus.default_n import *
import ls_p as physics
from proteus import (StepControl,
                     TimeIntegration,
                     NonlinearSolvers,
                     LinearSolvers,
                     LinearAlgebraTools)
from proteus.kernels import NCLS
from proteus import Context

ct = Context.get()
domain = ct.domain
nd = ct.domain.nd
mesh = domain.MeshOptions

if ct.timeDiscretization=='vbdf':
    timeIntegration = TimeIntegration.VBDF
    timeOrder=2
    stepController  = StepControl.Min_dt_cfl_controller
elif ct.timeDiscretization=='flcbdf':
    timeIntegration = TimeIntegration.FLCBDF
    #stepController = FLCBDF_controller
    stepController  = StepControl.Min_dt_cfl_controller
    time_tol = 10.0*ct.ls_nl_atol_res
    atol_u = {0:time_tol}
    rtol_u = {0:time_tol}
else:
    timeIntegration = TimeIntegration.BackwardEuler_cfl
    stepController  = StepControl.Min_dt_cfl_controller

# time stepping
runCFL = ct.runCFL

# mesh options
nLevels = ct.nLevels
parallelPartitioningType = mesh.parallelPartitioningType
nLayersOfOverlapForParallel = mesh.nLayersOfOverlapForParallel
restrictFineSolutionToAllMeshes = mesh.restrictFineSolutionToAllMeshes
triangleOptions = mesh.triangleOptions



elementQuadrature = ct.elementQuadrature
elementBoundaryQuadrature = ct.elementBoundaryQuadrature

femSpaces = {0: ct.basis}

massLumping       = False
conservativeFlux  = None
numericalFluxType = NCLS.NumericalFlux

subgridError      = NCLS.SubgridError(coefficients=physics.coefficients,
                                      nd=ct.domain.nd)
shockCapturing    = NCLS.ShockCapturing(physics.coefficients,
                                        ct.domain.nd,
                                        shockCapturingFactor=ct.ls_shockCapturingFactor,
                                        lag=ct.ls_lag_shockCapturing)

fullNewtonFlag  = True
multilevelNonlinearSolver = NonlinearSolvers.Newton
levelNonlinearSolver      = NonlinearSolvers.Newton

nonlinearSmoother = None
linearSmoother    = None

matrix = LinearAlgebraTools.SparseMatrix

if ct.useOldPETSc:
    multilevelLinearSolver = LinearSolvers.PETSc
    levelLinearSolver      = LinearSolvers.PETSc
else:
    multilevelLinearSolver = LinearSolvers.KSP_petsc4py
    levelLinearSolver      = LinearSolvers.KSP_petsc4py
if ct.useSuperlu:
    multilevelLinearSolver = LinearSolvers.LU
    levelLinearSolver      = LinearSolvers.LU

linear_solver_options_prefix = 'ncls_'
nonlinearSolverConvergenceTest = 'rits'
levelNonlinearSolverConvergenceTest = 'r'
linearSolverConvergenceTest         = 'r-true'

tolFac = 0.0
linTolFac = 0.0
l_atol_res = 0.1*ct.ls_nl_atol_res
nl_atol_res = ct.ls_nl_atol_res
useEisenstatWalker = False#True

maxNonlinearIts = 50
maxLineSearches = 0

auxiliaryVariables = ct.domain.auxiliaryVariables['ls']