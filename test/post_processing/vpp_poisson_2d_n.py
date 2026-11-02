"""Numerics for vpp_poisson_2d_p: C0 P1 or P2 (VPP_ORDER), the post-processing variant from VPP_VARIANT.

Dirichlet conditions are weak (SIPG exterior flux) unless VPP_STRONG=1.
"""
import os
from proteus.default_n import *
try:
    from .vpp_poisson_2d_p import *
except ImportError:
    from vpp_poisson_2d_p import *

timeIntegration = NoIntegration
nDTout = 1

order = int(os.environ.get("VPP_ORDER", "1"))
if order == 1:
    femSpaces = {0: C0_AffineLinearOnSimplexWithNodalBasis}
else:
    femSpaces = {0: C0_AffineQuadraticOnSimplexWithNodalBasis}
elementQuadrature = SimplexGaussQuadrature(nd, 4)
elementBoundaryQuadrature = SimplexGaussQuadrature(nd - 1, 4)

nnx = int(os.environ.get("VPP_NN", "13"))
nny = nnx
nLevels = 1
subgridError = None
shockCapturing = None

multilevelNonlinearSolver = Newton
levelNonlinearSolver = Newton
maxNonlinearIts = 1
maxLineSearches = 0
fullNewtonFlag = True
nl_atol_res = 1.0e-10
tolFac = 0.0

matrix = SparseMatrix
multilevelLinearSolver = KSP_petsc4py
levelLinearSolver = KSP_petsc4py
linearSolverConvergenceTest = 'r-true'
linearSmoother = None
linTolFac = 0.0
l_atol_res = 1.0e-13
nLayersOfOverlapForParallel = int(os.environ.get("VPP_OVERLAP", "1"))
parallelPartitioningType = MeshParallelPartitioningTypes.node
numericalFluxType = ConstantAdvection_Diffusion_SIPG_exterior
if os.environ.get("VPP_STRONG"):
    # strong Dirichlet conditions (no weak boundary flux), serial only: the direct solver takes the reduced system
    numericalFluxType = StrongDirichlet
    multilevelLinearSolver = LU
    levelLinearSolver = LU

conservativeFlux = {0: os.environ.get("VPP_VARIANT", "pwl")}
