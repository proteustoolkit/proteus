from proteus import *
from proteus.default_n import *
try:
    from .ladr_sgs_2d_p import *
except ImportError:
    from ladr_sgs_2d_p import *

timeIntegration = NoIntegration

elementQuadrature = SimplexGaussQuadrature(nd, opts.qOrder)
elementBoundaryQuadrature = SimplexGaussQuadrature(nd-1, opts.qOrder)
femSpaces = {0: {1: C0_AffineLinearOnSimplexWithNodalBasis,
                 2: C0_AffineQuadraticOnSimplexWithNodalBasis}[opts.pOrder]}

# structured triangles (the mesher is not involved)
nnx = 2**(opts.refinement+2) + 1
nny = nnx
genMesh = True
nLevels = 1
nLayersOfOverlapForParallel = 2

subgridError = ADR.SubgridError(coefficients, nd)
shockCapturing = ADR.ShockCapturing(coefficients, nd, shockCapturingFactor=0.0, lag=False)
numericalFluxType = ADR.NumericalFlux

multilevelNonlinearSolver = Newton
levelNonlinearSolver = Newton
fullNewtonFlag = True
tolFac = 0.0
nl_atol_res = 1.0e-10
maxNonlinearIts = 5  # linear problem: one Newton step

matrix = SparseMatrix
if opts.usePetsc:
    multilevelLinearSolver = KSP_petsc4py
    levelLinearSolver = KSP_petsc4py
else:
    multilevelLinearSolver = LU
    levelLinearSolver = LU
linearSmoother = None
linTolFac = 0.0
l_atol_res = 1.0e-12
conservativeFlux = None
