from proteus.default_n import *
from proteus import (StepControl,
                     TimeIntegration,
                     NonlinearSolvers,
                     LinearSolvers,
                     LinearAlgebraTools)
import vof_p as physics
from proteus.kernels import VOF
from proteus import Context

ct = Context.get()
domain = ct.domain
nd = ct.domain.nd
periodicDirichletConditions = physics.periodicDirichletConditions
parallelPeriodic=True

# time stepping
runCFL = ct.runCFL
if ct.timeIntegration == "VBDF":
    timeIntegration = TimeIntegration.VBDF
    timeOrder = 2
else:
    timeIntegration = TimeIntegration.BackwardEuler_cfl
stepController  = StepControl.Min_dt_controller

elementQuadrature = ct.elementQuadrature
elementBoundaryQuadrature = ct.elementBoundaryQuadrature

femSpaces = {0: ct.basis}

massLumping       = False
numericalFluxType = VOF.NumericalFlux
conservativeFlux  = None
subgridError      = VOF.SubgridError(coefficients=physics.coefficients,
                                     nd=ct.domain.nd)
shockCapturing    = VOF.ShockCapturing(physics.coefficients,
                                       ct.domain.nd,
                                       shockCapturingFactor=ct.vof_shockCapturingFactor,
                                       lag=ct.vof_lag_shockCapturing)

fullNewtonFlag = True
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

linear_solver_options_prefix = 'vof_'
levelNonlinearSolverConvergenceTest = 'r'
linearSolverConvergenceTest         = 'r-true'

tolFac      = 0.0
linTolFac   = 0.001
l_atol_res = 0.001*ct.vof_nl_atol_res
nl_atol_res = ct.vof_nl_atol_res
useEisenstatWalker = False#True

maxNonlinearIts = 50
maxLineSearches = 0
