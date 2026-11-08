import os
from proteus.default_n import *
from proteus import NumericalFlux
from seam_poisson_p import *

timeIntegration = NoIntegration
femSpaces = {0: C0_AffineLinearOnSimplexWithNodalBasis}
elementQuadrature = SimplexGaussQuadrature(nd, 4)
elementBoundaryQuadrature = SimplexGaussQuadrature(nd - 1, 4)
nLevels = 1
numericalFluxType = getattr(NumericalFlux, os.environ.get("SEAM_FLUX", "Advection_DiagonalUpwind_Diffusion_IIPG_exterior"))
multilevelNonlinearSolver = Newton
levelNonlinearSolver = Newton
fullNewtonFlag = True
nl_atol_res = 1e-11
tolFac = 0.0
maxNonlinearIts = int(os.environ.get("SEAM_MAXITS", "20"))
maxLineSearches = 0
matrix = SparseMatrix
periodicDirichletConditions = {0: getPDBC} if PERIODIC else None
parallelPeriodic = False
if PARALLEL:
    parallelPeriodic = True
    parallelPartitioningType = MeshParallelPartitioningTypes.element
    nLayersOfOverlapForParallel = 0
    multilevelLinearSolver = KSP_petsc4py
    levelLinearSolver = KSP_petsc4py
    linear_solver_options_prefix = 'seam_'
    linearSmoother = None
    l_atol_res = 1e-14
    linTolFac = 0.0
else:
    multilevelLinearSolver = LU
    levelLinearSolver = LU
