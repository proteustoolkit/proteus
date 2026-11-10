"""Numerics for the TADR half of the coupled MMS.

The STABILIZATION_TYPE=5 branch of benchmark_problem/transport_n.py, with a
fixed time step (FixedDT_controller in domain_mms) and a nonlinear tolerance
well below the discretization error.
"""

import os

from proteus import *
from proteus.default_n import *
from proteus.mprans import TADR

# Domain first: domain_mms re-exports default_p's `coefficients = None`.
try:
    from .domain_mms import *
    from .transport_p import *
except ImportError:
    from domain_mms import *
    from transport_p import *

nnx = nnx
nny = nny
triangleFlag = triangleFlag
nLevels = 1

# ImplicitEV: backward Euler + Newton on the edge-based upwind operator.
multilevelNonlinearSolver = Newton
levelNonlinearSolver = Newton
maxLineSearches = 10
fullNewtonFlag = True
updateJacobian = True
timeIntegration = BackwardEuler

SSPOrder = 1
timeOrder = SSPOrder
nStagesTime = SSPOrder
stepController = FixedDT_controller
stepExact = True
# Same relative + absolute Newton test as flow_n.py, for the same reason.
tolFac = float(os.environ.get("MMS_NL_RTOL", "1.0e-5"))
nl_atol_res = float(os.environ.get("MMS_NL_ATOL", "1.0e-9"))
maxNonlinearIts = 30
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
