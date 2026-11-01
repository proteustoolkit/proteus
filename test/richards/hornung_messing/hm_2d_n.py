from proteus import *
from proteus.default_n import *
from hm_2d_p import *

timeIntegrator = ForwardIntegrator
timeIntegration = Richards.ThetaScheme
timeIntegrationTheta = ct.theta_time
stepController = FixedStep
timeOrder = 1
DT = T / ct.nDT
tnList = [i * DT for i in range(ct.nDT + 1)]
nDTout = ct.nDT

nnx = nny = ct.nn
nLevels = 1
if ct.quad:
    # Q1 on squares, h = 1/(nn-1), 2x2 Gauss points (the kernel's Q1 build)
    quad = True
    femSpaces = {0: C0_AffineLinearOnCubeWithNodalBasis}
    elementQuadrature = CubeGaussQuadrature(nd, 2)
    elementBoundaryQuadrature = CubeGaussQuadrature(nd - 1, 2)
else:
    # structured right-triangle P1 mesh; all off-diagonal stiffness entries
    # are <= 0, so the edge-based low order needs no M-matrix clipping
    femSpaces = {0: C0_AffineLinearOnSimplexWithNodalBasis}
    elementQuadrature = SimplexGaussQuadrature(nd, 4)
    elementBoundaryQuadrature = SimplexGaussQuadrature(nd - 1, 4)
    triangleFlag = 0

subgridError = None
massLumping = False
shockCapturing = None
numericalFluxType = Richards_IIPG_exterior  # weak boundary flux, as in the WRR decks

multilevelNonlinearSolver = Newton
levelNonlinearSolver = Newton
nonlinearSmoother = None
fullNewtonFlag = True
tolFac = 0.0
nl_atol_res = 1.0e-10
maxNonlinearIts = ct.nl_max_its
# increment test ||du|| < nl_rtol_du ||u|| (+ nl_atol_du), e.g. nl_test='du' nl_unorm='L2' for the JNMA criterion
levelNonlinearSolverConvergenceTest = ct.nl_test
nonlinearSolverUNorm = None if ct.nl_unorm == 'l2' else ct.nl_unorm
nl_rtol_du = ct.nl_rtol_du
nl_atol_du = 0.0
maxLineSearches = 20

matrix = SparseMatrix
multilevelLinearSolver = LU
levelLinearSolver = LU
linearSmoother = None
linTolFac = 0.001
conservativeFlux = None
if ct.par:
    # e.g. mpiexec -n 4 parun -p hm_2d_p.py hm_2d_n.py -C "par=True ..." \
    #        -P "-ksp_type preonly -pc_type lu -pc_factor_mat_solver_type superlu_dist"
    multilevelLinearSolver = KSP_petsc4py
    levelLinearSolver = KSP_petsc4py
    parallelPartitioningType = getattr(MeshParallelPartitioningTypes, ct.par_part)
    nLayersOfOverlapForParallel = ct.par_overlap
