from proteus import *
from proteus import (StepControl,
                     TimeIntegration,
                     NonlinearSolvers,
                     LinearSolvers,
                     LinearAlgebraTools)
from twp_navier_stokes_p import *
from tank import *

if timeDiscretization=='vbdf':
    timeIntegration = VBDF
    timeOrder=2
    stepController  = Min_dt_cfl_controller
elif timeDiscretization=='flcbdf':
    timeIntegration = FLCBDF
    #stepController = FLCBDF_controller_sys
    stepController  = Min_dt_cfl_controller
    time_tol = 10.0*ns_nl_atol_res
    atol_u = {0:time_tol,1:time_tol}
    rtol_u = {0:time_tol,1:time_tol}
else:
    timeIntegration = BackwardEuler_cfl
    stepController  = Min_dt_cfl_controller

femSpaces = {0:basis,
	     1:basis}
#femSpaces = {0:C0_AffineQuadraticOnSimplexWithNodalBasis,
#             1:C0_AffineQuadraticOnSimplexWithNodalBasis}

massLumping       = False
numericalFluxType = None
conservativeFlux  = None

numericalFluxType = RANS3PF.NumericalFlux
subgridError = RANS3PF.SubgridError(coefficients,nd,lag=ns_lag_subgridError,hFactor=hFactor)
shockCapturing = RANS3PF.ShockCapturing(coefficients,nd,ns_shockCapturingFactor,lag=ns_lag_shockCapturing)

fullNewtonFlag = True
multilevelNonlinearSolver = NonlinearSolvers.Newton
levelNonlinearSolver      = NonlinearSolvers.Newton

nonlinearSmoother = None

linearSmoother    = None #SimpleNavierStokes2D

matrix = SparseMatrix

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
nonlinearSolverConvergenceTest = 'rits'
levelNonlinearSolverConvergenceTest = 'rits'
linearSolverConvergenceTest             = 'r-true'

tolFac = 0.0
linTolFac = 0.01
l_atol_res = 0.01*ns_nl_atol_res
nl_atol_res = ns_nl_atol_res
useEisenstatWalker = False
maxNonlinearIts = 50
maxLineSearches = 0
conservativeFlux = {0:'point-eval'}
#conservativeFlux = {0:'pwl-bdm-opt'}
#auxiliaryVariables=[pointGauges,lineGauges]
auxiliaryVariables = ct.domain.auxiliaryVariables['twp']+[ct.v_output]
