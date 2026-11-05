from proteus import *
from proteus.default_n import *
from beam_channel_p import *

timeIntegration = BackwardEuler_cfl
stepController = FixedStep
runCFL = 0.9
DT = 0.01
nDTout = 4

femSpaces = {0: C0_AffineLinearOnSimplexWithNodalBasis,
             1: C0_AffineLinearOnSimplexWithNodalBasis,
             2: C0_AffineLinearOnSimplexWithNodalBasis,
             3: C0_AffineLinearOnSimplexWithNodalBasis}
elementQuadrature = SimplexGaussQuadrature(nd, 3)
elementBoundaryQuadrature = SimplexGaussQuadrature(nd - 1, 3)

nnx, nny, nnz = 9, 14, 5
nLevels = 1

numericalFluxType = RANS2P_IB.RANS2P.NumericalFlux
subgridError = RANS2P_IB.RANS2P.SubgridError(coefficients, nd, lag=False, hFactor=1.0)
shockCapturing = RANS2P_IB.RANS2P.ShockCapturing(coefficients, nd, 0.0, lag=False)

fullNewtonFlag = True
multilevelNonlinearSolver = Newton
levelNonlinearSolver = Newton
nonlinearSmoother = None
linearSmoother = None
matrix = SparseMatrix
multilevelLinearSolver = KSP_petsc4py
levelLinearSolver = KSP_petsc4py
linear_solver_options_prefix = "rans2p_"
levelNonlinearSolverConvergenceTest = "r"
linearSolverConvergenceTest = "r-true"
tolFac = 0.0
nl_atol_res = 1.0e-10
l_atol_res = 1.0e-12
maxNonlinearIts = 20
maxLineSearches = 0
parallelPartitioningType = MeshParallelPartitioningTypes.node
nLayersOfOverlapForParallel = 0
