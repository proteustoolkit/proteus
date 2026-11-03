from proteus import *
from proteus.default_n import *
try:
    from .ladr_ss_2d_p import *
except:
    from ladr_ss_2d_p import *

timeIntegration = NoIntegration

runCFL = 0.9

femSpaces = {0:C0_AffineLinearOnSimplexWithNodalBasis}
elementQuadrature = SimplexGaussQuadrature(nd,opts.qOrder)
elementBoundaryQuadrature = SimplexGaussQuadrature(nd-1,opts.qOrder)
#femSpaces = {0:C0_AffineQuadraticOnSimplexWithNodalBasis}
#femSpaces = {0:DG_AffineP0_OnSimplexWithMonomialBasis}
#femSpaces = {0:DG_AffineP1_OnSimplexWithMonomialBasis}
#femSpaces = {0:DG_AffineP5_OnSimplexWithMonomialBasis}
#femSpaces = {0:DG_Constants}
#femSpaces = {0:DG_AffineLinearOnSimplexWithNodalBasis}
#femSpaces = {0:DG_AffineQuadraticOnSimplexWithNodalBasis}

# femSpaces = {0:DG_AffineP1_OnSimplexWithMonomialBasis}
# elementQuadrature = SimplexGaussQuadrature(nd,4)
# elementBoundaryQuadrature = SimplexGaussQuadrature(nd-1,4)

#femSpaces = {0:DG_AffineP1_OnSimplexWithMonomialBasis}
#femSpaces = {0:DG_AffineLinearOnSimplexWithNodalBasis}
# elementQuadrature = SimplexGaussQuadrature(nd,4)
# elementBoundaryQuadrature = SimplexGaussQuadrature(nd-1,4)

#femSpaces = {0:DG_AffineP0_OnSimplexWithMonomialBasis}
#femSpaces = {0:DG_AffineP1_OnSimplexWithMonomialBasis}
#femSpaces = {0:DG_AffineP2_OnSimplexWithMonomialBasis}
#femSpaces = {0:DG_AffineQuadraticOnSimplexWithNodalBasis}
#elementQuadrature = SimplexGaussQuadrature(nd,3)
#elementBoundaryQuadrature = SimplexGaussQuadrature(nd-1,3)

#femSpaces = {0:DG_AffineP3_OnSimplexWithMonomialBasis}
#femSpaces = {0:DG_AffineP0_OnSimplexWithMonomialBasis}
#femSpaces = {0:DG_AffineP1_OnSimplexWithMonomialBasis}
#femSpaces = {0:DG_AffineP2_OnSimplexWithMonomialBasis}
#femSpaces = {0:DG_AffineP3_OnSimplexWithMonomialBasis}
#elementQuadrature = SimplexGaussQuadrature(nd,6)
#elementBoundaryQuadrature = SimplexGaussQuadrature(nd-1,6)

# femSpaces = {0:DG_AffineP4_OnSimplexWithMonomialBasis}
# elementQuadrature = SimplexGaussQuadrature(nd,5)
# elementBoundaryQuadrature = SimplexGaussQuadrature(nd-1,5)

#femSpaces = {0:DG_AffineP3_OnSimplexWithMonomialBasis}
#elementQuadrature = SimplexGaussQuadrature(nd,6)
#elementBoundaryQuadrature = SimplexGaussQuadrature(nd-1,6)


nnx=4*2**opts.refinement+1
nny = nnx
genMesh = True
triangleOptions = "VApq30Dena%8.8f" % (((L[0]/(nnx-1))**2)/2.0,)
nLevels = 1
nLayersOfOverlapForParallel = 2  # face terms; matches the p module's MeshOptions

subgridError = None
subgridError = ADR.SubgridError(coefficients,nd)
shockCapturing = ADR.ShockCapturing(coefficients,nd,shockCapturingFactor=0.0,lag=False)
#massLumping = True

numericalFluxType = ADR.NumericalFlux
multilevelNonlinearSolver  = Newton
levelNonlinearSolver = Newton
nonlinearSmoother = None#NLStarILU

fullNewtonFlag = True

tolFac = 0.0#1.0e-8

nl_atol_res = 1.0e-8

maxNonlinearIts =1001

matrix = SparseMatrix
multilevelLinearSolver = LU
levelLinearSolver = LU
linearSmoother = None#GaussSeidel

linTolFac = 0.001

conservativeFlux = None
