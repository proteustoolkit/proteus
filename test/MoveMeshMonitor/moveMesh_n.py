from proteus.default_n import *
from proteus import (FemTools,
                     Quadrature,
                     TimeIntegration,
                     NumericalFlux,
                     NonlinearSolvers,
                     LinearSolvers,
                     StepControl,
                     LinearAlgebraTools)
try:
    from . import moveMesh_p as physics
except:
    import moveMesh_p as physics

from proteus import Context
from proteus.kernels import AddedMass

ct = Context.get()
domain = ct.domain
nd = ct.domain.nd
mesh = domain.MeshOptions

# time stepping
timeIntegration = TimeIntegration.NoIntegration

# mesh options
nLevels = ct.nLevels
restrictFineSolutionToAllMeshes = mesh.restrictFineSolutionToAllMeshes
nn = ct.nn
mesh.nn = nn
mesh.triangleFlag = 0

from petsc4py import PETSc
OptDB = PETSc.Options()
OptDB.setValue("ksp_constant_null_space", 1)
OptDB.setValue("info", 1)

elementQuadrature = ct.elementQuadrature
elementBoundaryQuadrature = ct.elementBoundaryQuadrature

femSpaces = {0: ct.basis}

#stepController=StepControl.FixedStep

T = ct.T

#massLumping       = False
numericalFluxType = NumericalFlux.Diffusion_SIPG_exterior
#numericalFluxType = NumericalFlux.DoNothing
#numericalFluxType = AddedMass.NumericalFlux
#conservativeFlux = {0: 'pwl-bdm'}

matrix = LinearAlgebraTools.SparseMatrix

#subgridError = None
#shockCapturing = None

#fullNewtonFlag = True

multilevelLinearSolver = LinearSolvers.KSP_petsc4py
levelLinearSolver      = LinearSolvers.KSP_petsc4py
parallelPartitioningType = mesh.parallelPartitioningType
nLayersOfOverlapForParallel = mesh.nLayersOfOverlapForParallel
nonlinearSmoother = NonlinearSolvers.MoveMeshMonitorNewton
linearSmoother = LinearSolvers.NavierStokesPressureCorrection

linear_solver_options_prefix = 'mesh2_'

multilevelNonlinearSolver  = NonlinearSolvers.MoveMeshMonitorNewton
levelNonlinearSolver       = NonlinearSolvers.MoveMeshMonitorNewton

linTolFac = 0.0
l_atol_res = 1.0e-8
tolFac = 0.0
nl_atol_res = 1.0e-8

nonlinearSolverConvergenceTest = 'r'
levelNonlinearSolverConvergenceTest = 'r'
linearSolverConvergenceTest = 'r-true'
maxNonLinearIts = 1
maxLineSearches = 0
periodicDirichletCOnditions = None
conservativeFlux = None
