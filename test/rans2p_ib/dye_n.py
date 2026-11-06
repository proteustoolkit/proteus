from proteus.default_n import *
from proteus.mprans import VOF
import dye_p as physics
from beam_channel_n import nnx, nny, nnz, nLevels, DT, nDTout, femSpaces as _flow_spaces

timeIntegration = BackwardEuler_cfl
stepController = FixedStep
runCFL = 0.9

femSpaces = {0: C0_AffineLinearOnSimplexWithNodalBasis}
elementQuadrature = SimplexGaussQuadrature(physics.nd, 3)
elementBoundaryQuadrature = SimplexGaussQuadrature(physics.nd - 1, 3)

numericalFluxType = VOF.NumericalFlux
subgridError = VOF.SubgridError(coefficients=physics.coefficients, nd=physics.nd)
shockCapturing = VOF.ShockCapturing(coefficients=physics.coefficients, nd=physics.nd, shockCapturingFactor=0.2,
                                    lag=True)

fullNewtonFlag = True
multilevelNonlinearSolver = Newton
levelNonlinearSolver = Newton
nonlinearSmoother = None
linearSmoother = None
matrix = SparseMatrix
multilevelLinearSolver = KSP_petsc4py
levelLinearSolver = KSP_petsc4py
linear_solver_options_prefix = "dye_"
levelNonlinearSolverConvergenceTest = "rits"
linearSolverConvergenceTest = "r-true"
tolFac = 0.0
nl_atol_res = 1.0e-10
linTolFac = 0.0
l_atol_res = 1.0e-12
maxNonlinearIts = 20
maxLineSearches = 0
parallelPartitioningType = MeshParallelPartitioningTypes.node
nLayersOfOverlapForParallel = 0
