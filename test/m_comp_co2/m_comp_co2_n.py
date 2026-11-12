from proteus import *
from proteus.default_n import *
from m_comp_co2_p import *

# Numerics deck for the m_comp_co2 regression cases; the case comes from
# m_comp_co2_p.py (Context option "case").  Run with direct LU (see there).
timeIntegrator = ForwardIntegrator
timeIntegration = m_comp_co2.ThetaScheme
timeOrder = 1
nLevels = 1

femSpaces = {0: C0_AffineLinearOnSimplexWithNodalBasis,
             1: C0_AffineLinearOnSimplexWithNodalBasis}
# Vertex (Lobatto) quadrature; per_rock_closure requires it on the boundary.
elementQuadrature = SimplexLobattoQuadrature(nd, 1)
elementBoundaryQuadrature = SimplexLobattoQuadrature(nd - 1, 1)

massLumping = False
numericalFluxType = Advection_DiagonalUpwind_Diffusion_IIPG_exterior
shockCapturing = None

rtol_u[0] = 1.0e-3
atol_u[0] = 1.0e-3
rtol_u[1] = 1.0e-3
atol_u[1] = 1.0e-3

stepController = HeuristicNL_dt_controller
dtNLreduceFactor = 0.5
dtNLfailureReduceFactor = 0.5
stepExact = True

multilevelNonlinearSolver = Newton
levelNonlinearSolver = Newton
nonlinearSmoother = NLJacobi
fullNewtonFlag = True
tolFac = 0.0
maxNonlinearIts = 25

matrix = SparseMatrix
multilevelLinearSolver = KSP_petsc4py
levelLinearSolver = KSP_petsc4py
linearSmoother = None
linTolFac = 1.0e-3
l_atol_res = 1.0e-3

parallelPartitioningType = MeshParallelPartitioningTypes.node

if CASE in REST_CASES:
    # Regular triangular mesh on the box.
    nnx = 5
    nny = 61 if CASE == "segregation" else 41
    # (DT [s], T [s], nDTout): a transient that must decay or move gets the
    # smaller DT; the exact rest states hold at 1e3 s.
    DT, T, nDTout = {
        "static":         (1.0e3, 2.0e4, 20),
        "hydrostatic":    (1.0e2, 1.0e4, 50),
        "segregation":    (1.0e2, 5.0e4, 50),
        "open_top":       (1.0e3, 2.0e4, 20),
        "open_top_stab2": (1.0e3, 2.0e4, 20),
    }[CASE]
    nonlinearIterationsFloor = 7
    nonlinearIterationsCeil = 14
    dtNLgrowFactor = 1.2
    useInitialGuessPredictor = True
    nl_atol_res = 1.0e-8
    maxLineSearches = 5
else:
    # rock_column: first step DT = 1 s (default_n), outputs every 300 s.
    T = opts.T
    nDTout = max(1, int(round(T / 300.0)))
    nonlinearIterationsFloor = 10
    nonlinearIterationsCeil = 20
    dtNLgrowFactor = 1.5
    useInitialGuessPredictor = False
    nl_atol_res = 1.0e-10    # per-unit-depth molar fluxes on a 0.02 m strip are small
    maxLineSearches = 20
    maxSolverFailures = 1000
