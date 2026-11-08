from proteus import StepControl
from proteus.default_n import *
from proteus import (StepControl,
                     TimeIntegration,
                     NonlinearSolvers,
                     LinearSolvers,
                     LinearAlgebraTools)
import vof_p as physics
from proteus import Context
from proteus.mprans import VOS3P

ct = Context.get()
domain = ct.domain
nd = ct.domain.nd
mesh = domain.MeshOptions


# time stepping
runCFL = ct.runCFL
timeIntegration = VOS3P.RKEV#BackwardEuler_cfl
stepController  = StepControl.Min_dt_controller

# mesh options
nLevels = ct.nLevels
parallelPartitioningType = mesh.parallelPartitioningType
nLayersOfOverlapForParallel = mesh.nLayersOfOverlapForParallel
restrictFineSolutionToAllMeshes = mesh.restrictFineSolutionToAllMeshes
triangleOptions = mesh.triangleOptions



elementQuadrature = ct.elementQuadrature
elementBoundaryQuadrature = ct.elementBoundaryQuadrature


femSpaces = {0:ct.basis}

massLumping       = False
numericalFluxType = VOS3P.NumericalFlux
conservativeFlux  = None
subgridError      = VOS3P.SubgridError(coefficients=physics.coefficients,nd=nd)
shockCapturing    = VOS3P.ShockCapturing(physics.coefficients,nd,shockCapturingFactor=ct.vos_shockCapturingFactor,lag=ct.vos_lag_shockCapturing)

fullNewtonFlag = True
multilevelNonlinearSolver = NonlinearSolvers.Newton
levelNonlinearSolver      = NonlinearSolvers.ExplicitLumpedMassMatrix

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

linear_solver_options_prefix = 'vos_'
nonlinearSolverConvergenceTest = 'rits'
levelNonlinearSolverConvergenceTest = 'rits'
linearSolverConvergenceTest         = 'r-true'

tolFac      = 0.0
nl_atol_res = ct.vos_nl_atol_res

linTolFac   = 0.0
l_atol_res = 0.1*ct.vos_nl_atol_res

useEisenstatWalker = False

maxNonlinearIts = 50
maxLineSearches = 0

#auxiliaryVariables = ct.domain.auxiliaryVariables['vos']
auxiliaryVariables = ct.domain.auxiliaryVariables['vos']+[ct.vos_output]
