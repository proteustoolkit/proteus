from proteus import (StepControl,
                     TimeIntegration,
                     NonlinearSolvers,
                     LinearSolvers,
                     LinearAlgebraTools)
from proteus.default_n import *
from . import twp_navier_stokes_p as physics
from proteus.kernels import RANS2P
from .multiphase import *

#time stepping
if timeDiscretization=='vbdf':
    timeIntegration = TimeIntegration.VBDF
    timeOrder=2
    stepController  = StepControl.Min_dt_cfl_controller
elif timeDiscretization=='flcbdf':
    timeIntegration = TimeIntegration.FLCBDF
    #stepController = FLCBDF_controller_sys
    stepController  = StepControl.Min_dt_cfl_controller
    time_tol = 10.0*ns_nl_atol_res
    atol_u = {1:time_tol,2:time_tol}
    rtol_u = {1:time_tol,2:time_tol}
else:
    timeIntegration = TimeIntegration.BackwardEuler_cfl
    stepController  = StepControl.Min_dt_cfl_controller

femSpaces = {0:basis,
             1:basis,
             2:basis}

massLumping       = False
numericalFluxType = RANS2P.NumericalFlux
subgridError = RANS2P.SubgridError(coefficients=physics.coefficients,
                                   nd=nd,
                                   lag=ns_lag_subgridError,
                                   hFactor=hFactor)
shockCapturing = RANS2P.ShockCapturing(coefficients=physics.coefficients,
                                       nd=nd,
                                       shockCapturingFactor=ns_shockCapturingFactor,
                                       lag=ns_lag_shockCapturing)

fullNewtonFlag = True
multilevelNonlinearSolver = NonlinearSolvers.Newton
levelNonlinearSolver      = NonlinearSolvers.Newton

nonlinearSmoother = None

linearSmoother    = LinearSolvers.SimpleNavierStokes2D

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

linear_solver_options_prefix = 'rans2p_'
levelNonlinearSolverConvergenceTest = 'r'
linearSolverConvergenceTest             = 'r-true'

tolFac = 0.0
linTolFac = 0.01
l_atol_res = 0.01*vof_nl_atol_res
nl_atol_res = ns_nl_atol_res
useEisenstatWalker = False
maxNonlinearIts = 50
maxLineSearches = 0
if useHex:
    pass
else:
    conservativeFlux = {0:'pwl-bdm-opt'}
