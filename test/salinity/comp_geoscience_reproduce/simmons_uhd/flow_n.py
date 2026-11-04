"""Numerics for the Richards half of the Simmons benchmark."""

from proteus import *
from proteus.default_n import *
from proteus.default_so import *
from proteus.richards import Richards

# Domain first: domain_simmons re-exports default_p's `coefficients = None`,
# so importing it after flow_p would clobber the Richards coefficients.
try:
    from .domain_simmons import *
    from .flow_p import *
except ImportError:
    from domain_simmons import *
    from flow_p import *

# Structured mesh, built from the rectangle by NumericalSolution because
# flow_p leaves domain = polyfile = None.  nnx/nny/triangleFlag come from
# domain_simmons and follow Eqs. (10)/(11) at the selected spatial level.
nnx = nnx
nny = nny
triangleFlag = triangleFlag
nLevels = 1

timeIntegrator = ForwardIntegrator
timeIntegration = Richards.ThetaScheme
timeOrder = 1

stepController = CappedNL_dt_controller     # dt_max from domain_simmons (G7)
nonlinearIterationsFloor = 10
nonlinearIterationsCeil = 20
# Matched to the STAB=5 block in transport_n.py.  The system dt is the min over
# both models, so the slower ramp sets the pace for the whole run: tnList opens
# with the 1e-8 s startup kicks and the first reporting time is 600 s, so the
# controller has ~11 decades to climb before any physical time is covered.  At
# the old 1.25 that took ~110 growth events -- measured on SMD level 2, 56% of
# all steps went into reaching t = 1 s.  At 2.0 it is ~36.
dtNLgrowFactor = 2.0
dtNLreduceFactor = 0.50
dtNLfailureReduceFactor = 0.25
useInitialGuessPredictor = True
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

tolFac = 0.0
nl_atol_res = 1.0e-7
maxNonlinearIts = 30
maxLineSearches = 10

matrix = SparseMatrix
multilevelLinearSolver = KSP_petsc4py
levelLinearSolver = KSP_petsc4py
linear_solver_options_prefix = "richards_"
linearSmoother = None
linTolFac = 0.001

parallelPartitioningType = MeshParallelPartitioningTypes.element
