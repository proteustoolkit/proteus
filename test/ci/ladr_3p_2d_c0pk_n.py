from proteus import *
from proteus.default_n import *
try:
    from .ladr_3p_2d_p import *
except:
    from ladr_3p_2d_p import *

timeIntegration = NoIntegration

runCFL = 0.9

if opts.pOrder == 1:
    femSpaces = {0:C0_AffineLinearOnSimplexWithNodalBasis}
elif opts.pOrder == 2:
    femSpaces = {0:C0_AffineQuadraticOnSimplexWithNodalBasis}
else:
    raise ValueError("pOrder must be 1 or 2")
elementQuadrature = SimplexGaussQuadrature(nd,opts.qOrder)
elementBoundaryQuadrature = SimplexGaussQuadrature(nd-1,opts.qOrder)

nnx=4*2**opts.refinement+1
nny = nnx
genMesh = True
triangleOptions = "VApq30Dena%8.8f" % (((L[0]/(nnx-1))**2)/2.0,)
nLevels = 1

subgridError = None
subgridError = ADR.SubgridError(coefficients,nd)
shockCapturing = ADR.ShockCapturing(coefficients,nd,shockCapturingFactor=0.0,lag=False)

numericalFluxType = ADR.NumericalFlux
multilevelNonlinearSolver  = Newton
levelNonlinearSolver = Newton
nonlinearSmoother = None

fullNewtonFlag = True

tolFac = 0.0

nl_atol_res = 1.0e-8

maxNonlinearIts =1001

matrix = SparseMatrix
if opts.usePetsc:
    multilevelLinearSolver = KSP_petsc4py
    levelLinearSolver = KSP_petsc4py
else:
    multilevelLinearSolver = LU
    levelLinearSolver = LU
linearSmoother = None

linTolFac = 0.001

conservativeFlux = None
