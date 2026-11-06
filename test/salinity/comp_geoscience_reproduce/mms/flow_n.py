"""Numerics for the Richards half of the coupled MMS.

As benchmark_problem/flow_n.py, except for a fixed time step
(FixedDT_controller in domain_mms) and a nonlinear tolerance well below the
discretization error.
"""

import os

from proteus import *
from proteus.default_n import *
from proteus.default_so import *
from proteus.richards import Richards

# Domain first: domain_mms re-exports default_p's `coefficients = None`, so
# importing it after flow_p would clobber the Richards coefficients.
try:
    from .domain_mms import *
    from .flow_p import *
except ImportError:
    from domain_mms import *
    from flow_p import *

nnx = nnx
nny = nny
triangleFlag = triangleFlag
nLevels = 1

timeIntegrator = ForwardIntegrator
timeIntegration = Richards.ThetaScheme
timeOrder = 1
stepController = FixedDT_controller
stepExact = True

rtol_u[0] = 1.0e-3
atol_u[0] = 1.0e-3

massLumping = False
numericalFluxType = Advection_DiagonalUpwind_Diffusion_IIPG_exterior
shockCapturing = None

multilevelNonlinearSolver = Newton
levelNonlinearSolver = Newton
nonlinearSmoother = NLJacobi
fullNewtonFlag = True

# Newton stops at norm(r) < MMS_NL_RTOL*norm(r0) + MMS_NL_ATOL.  In the 1e-8 s
# startup steps the storage term ML*m/dt is ~3e5, so round-off holds the
# residual at 1e-10..3e-10 whatever norm(r0) is; the absolute part sits above
# that floor.  On a real step a 1e-9 residual moves psi by ~3e-5 m at N=80
# (ML dm/dpsi / dt ~ 4e-5), below the discretization error there (>= 4e-4 m).
tolFac = float(os.environ.get("MMS_NL_RTOL", "1.0e-5"))
nl_atol_res = float(os.environ.get("MMS_NL_ATOL", "1.0e-9"))
maxNonlinearIts = 30
maxLineSearches = 10

matrix = SparseMatrix
multilevelLinearSolver = KSP_petsc4py
levelLinearSolver = KSP_petsc4py
linear_solver_options_prefix = "richards_"
linearSmoother = None
linTolFac = 0.001

parallelPartitioningType = MeshParallelPartitioningTypes.element
