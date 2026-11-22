from proteus.default_n import *
from proteus import (StepControl,
                     TimeIntegration,
                     NonlinearSolvers,
                     LinearSolvers,
                     LinearAlgebraTools)
from . import vof_p as physics
from proteus.kernels import VOF
from .multiphase import *

if timeDiscretization=='vbdf':
    timeIntegration = TimeIntegration.VBDF
    timeOrder=2
    stepController  = StepControl.Min_dt_cfl_controller
elif timeDiscretization=='flcbdf':
    timeIntegration = TimeIntegration.FLCBDF
    #stepController = FLCBDF_controller
    stepController  = StepControl.Min_dt_cfl_controller
    time_tol = 10.0*vof_nl_atol_res
    atol_u = {0:time_tol}
    rtol_u = {0:time_tol}
else:
    timeIntegration = TimeIntegration.BackwardEuler_cfl
    stepController  = StepControl.Min_dt_cfl_controller

femSpaces = {0: basis}

massLumping       = False
numericalFluxType = VOF.NumericalFlux
conservativeFlux  = None
subgridError      = VOF.SubgridError(coefficients=physics.coefficients,
                                     nd=domain.nd)
shockCapturing    = VOF.ShockCapturing(coefficients=physics.coefficients,
                                       nd=domain.nd,
                                       shockCapturingFactor=vof_shockCapturingFactor,
                                       lag=vof_lag_shockCapturing)

fullNewtonFlag = True
multilevelNonlinearSolver = NonlinearSolvers.Newton
levelNonlinearSolver      = NonlinearSolvers.Newton

nonlinearSmoother = None
linearSmoother    = None

matrix = LinearAlgebraTools.SparseMatrix

if useOldPETSc:
    multilevelLinearSolver = LinearSolvers.PETSc
    levelLinearSolver      = LinearSolvers.PETSc
else:
    multilevelLinearSolver = LinearSolvers.KSP_petsc4py
    levelLinearSolver      = LinearSolvers.KSP_petsc4py

if useSuperlu:
    multilevelLinearSolver = LinearSolvers.LU
    levelLinearSolver      = LinearSolvers.LU

linear_solver_options_prefix = 'vof_'
levelNonlinearSolverConvergenceTest = 'rits'
linearSolverConvergenceTest         = 'r-true'

tolFac      = 0.0
nl_atol_res = vof_nl_atol_res

linTolFac   = 0.001
l_atol_res = 0.001*vof_nl_atol_res

useEisenstatWalker = False

maxNonlinearIts = 50
maxLineSearches = 0

auxiliaryVariables = domain.auxiliaryVariables['vof']
