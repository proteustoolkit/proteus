"""Numerics for the TADR half of the Simmons benchmark.

STABILIZATION_TYPE 2 runs explicit RKEV at runCFL.  That is the right thing at
the coarse end of the ladder, but see README.md: at spatial level 4
(dx ~ 2.4e-3 m) with pore velocities of order 3e-3 m/s it costs order 5e4 steps
per realisation, times 30 realisations.  STABILIZATION_TYPE 5 (implicit
entropy viscosity) exists for exactly this and is what the Elder case uses.
Make that choice on the coarse ladder, not at 250,000 elements.
"""

import os

from proteus import *
from proteus.default_n import *
from proteus.mprans import TADR

# Import the domain BEFORE the physics module.  domain_simmons does
# `from proteus.default_p import *`, which re-exports `coefficients = None`;
# importing it second would silently clobber the real coefficients object and
# fail a few lines down with an AttributeError on None.
try:
    from .domain_simmons import *
    from .transport_p import *
except ImportError:
    from domain_simmons import *
    from transport_p import *

nnx = nnx
nny = nny
triangleFlag = triangleFlag
nLevels = 1

multilevelNonlinearSolver = Newton

if coefficients.STABILIZATION_TYPE == 5:
    # ImplicitEV: backward Euler + Newton on the edge-based upwind operator.
    # TADR rejects SSP time integration and the explicit lumped-mass solvers
    # for this type.  The low-order operator is an M-matrix, so dt is limited
    # by accuracy and Newton convergence, not by the CFL condition.
    levelNonlinearSolver = Newton
    maxLineSearches = 10
    fullNewtonFlag = True
    updateJacobian = True
    timeIntegration = BackwardEuler
elif coefficients.STABILIZATION_TYPE in [0, 1]:
    # NOTE: BackwardEuler_cfl reads q[('cfl',0)] -- the CELL-based CFL, which
    # is not normalised by density.  Harmless here only because this deck now
    # runs dimensionless density; in SI it would drive dt ~1000x too small and
    # look like a hang.
    levelNonlinearSolver = Newton
    maxLineSearches = 0
    fullNewtonFlag = True
    updateJacobian = True
    timeIntegration = BackwardEuler_cfl
else:
    fullNewtonFlag = False
    updateJacobian = False
    timeIntegration = TADR.RKEV
    if coefficients.LUMPED_MASS_MATRIX is True:
        levelNonlinearSolver = ExplicitLumpedMassMatrix
    else:
        levelNonlinearSolver = ExplicitConsistentMassMatrixForVOF

SSPOrder = 1
if coefficients.STABILIZATION_TYPE == 5:
    # dt is not CFL-bound here; grow/shrink it on the Newton iteration count
    # like the flow model does.  The inexact Newton (frozen a_ij) converges
    # linearly in ~4 iterations, so a floor of 4 would sit inside the
    # controller's no-op band (grow iff nnl < floor, shrink iff nnl > ceil)
    # and pin dt forever -- hence the floor of 10.
    stepController = CappedNL_dt_controller  # dt_max from domain_simmons (G7)
    nonlinearIterationsFloor = 10
    nonlinearIterationsCeil = 20
    dtNLgrowFactor = 2.0
    dtNLreduceFactor = 0.50
    dtNLfailureReduceFactor = 0.25
    useInitialGuessPredictor = True
    stepExact = True
    tolFac = 0.0
    nl_atol_res = 1.0e-7
    maxNonlinearIts = 30
else:
    stepController = Min_dt_controller
    # SIMMONS_RUNCFL: edge-based CFL number for the explicit RKEV paths
    # (STAB 2/3/4).  0.2 is the value every archived explicit run used; the
    # STAB=2+FCT discriminating run of 2026-09-22 asked for 0.5.
    runCFL = float(os.environ.get("SIMMONS_RUNCFL", "0.2"))
timeOrder = SSPOrder
nStagesTime = SSPOrder
pDegree_tadr = 1

numericalFluxType = TADR.NumericalFlux
shockCapturing = TADR.ShockCapturing(
    coefficients,
    nd,
    shockCapturingFactor=shockCapturingFactor_tadr,
    lag=lag_shockCapturing_tadr,
)

matrix = SparseMatrix
multilevelLinearSolver = KSP_petsc4py
levelLinearSolver = KSP_petsc4py
linear_solver_options_prefix = "tadr_"
linearSolverConvergenceTest = "r-true"

parallelPartitioningType = MeshParallelPartitioningTypes.element
