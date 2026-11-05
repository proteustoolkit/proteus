"""Numerics for vpp_poisson_2d_p: P1 or P2 (VPP_ORDER), the post-processing variant from VPP_VARIANT.

The space is C0 unless VPP_SPACE is dg (interior penalty, SIPG) or ldg (local DG, the mixed form whose velocity the
transport computes itself); the discontinuous spaces are partitioned by element. C0 Dirichlet conditions are weak (SIPG
exterior flux) unless VPP_STRONG=1.
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
space = os.environ.get("VPP_SPACE", "c0")
if space == "c0":
    femSpaces = {0: C0_AffineLinearOnSimplexWithNodalBasis if order == 1 else C0_AffineQuadraticOnSimplexWithNodalBasis}
else:
    femSpaces = {0: DG_AffineLinearOnSimplexWithNodalBasis if order == 1 else DG_AffineQuadraticOnSimplexWithNodalBasis}
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
# LDG reads the velocity of a face neighbour, solved from that element's faces: two layers
nLayersOfOverlapForParallel = int(os.environ.get("VPP_OVERLAP", "2" if space == "ldg" else "1"))
parallelPartitioningType = MeshParallelPartitioningTypes.node
numericalFluxType = ConstantAdvection_Diffusion_SIPG_exterior
if space == "dg":
    numericalFluxType = Advection_DiagonalUpwind_Diffusion_SIPG
elif space == "ldg":
    numericalFluxType = Diffusion_LDG
if space != "c0":
    # a discontinuous space couples elements only through face fluxes, so each rank owns whole elements
    parallelPartitioningType = MeshParallelPartitioningTypes.element
if os.environ.get("VPP_ELEMENT_PARTITION"):
    parallelPartitioningType = MeshParallelPartitioningTypes.element
if os.environ.get("VPP_STRONG"):
    # strong Dirichlet conditions (no weak boundary flux), serial only: the direct solver takes the reduced system
    numericalFluxType = StrongDirichlet
    multilevelLinearSolver = LU
    levelLinearSolver = LU

conservativeFlux = {0: os.environ.get("VPP_VARIANT", "pwl")}
