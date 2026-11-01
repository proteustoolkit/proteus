"""
The default values for numerics modules

.. todo::

  Clean up default_n.py, finish documenting, decide on tolerance scheme
"""
try:
    from importlib import reload
except:
    pass
from .TimeIntegration import *
from .Quadrature import *
from .FemTools import *
from .SubgridError import *
from .ShockCapturing import *
from .NumericalFlux import *
from .NonlinearSolvers import *
from .LinearAlgebraTools import *
from .LinearSolvers import *
from .clapack import *
from .StepControl import *
from .AuxiliaryVariables import *
from .MeshTools import *
## \todo clean up default_n module

stepController = FixedStep
"""The step controller class derived from :class:`proteus.StepControl.SC_base`"""

timeIntegration = NoIntegration
"""The time integration class derived from :class:`proteus.TimeIntegraction.TI_base"""

timeIntegrator  = ForwardIntegrator
"""Deprecated, the time integrator class"""

runCFL = 0.9
"""The maximum CFL for the time step"""

nStagesTime = 1
"""The number of stages for the time discretization"""

timeOrder= 1
"""The order of the time discretization"""

DT = 1.0
"""The time step"""

nDTout = 1
"""The number of output time steps"""

rtol_u = {0:1.0e-4}
"""A dictionary of relative time integration tolerances for the components"""

atol_u = {0:1.0e-4}
"""A dictionary of absolute time integration tolerances for the components"""

nltol_u = 0.33
"""The nonlinear tolerance factor for the component error"""

ltol_u = 0.05
"""The linear tolerance factor for the component error"""

rtol_res = {0:1.0e-4}
"""A dictionary of relative tolerances for the weak residuals"""

atol_res = {0:1.0e-4}
"""A dictionary of absolute tolerances for the weak residuals"""

nl_atol_res = 1.0
"""The nonlinear residual tolerance"""

l_atol_res = 1.0
"""The linear residual tolerance"""

femSpaces = {}
r"""A dictionary of the finite element classes for each component

The classes should be of type :class:`proteus.FemTools.ParametricFiniteElementSpace` """

elementQuadrature = None
"""A quadrature object for element integrals"""

elementBoundaryQuadrature = None
"""A quadrature object for element boundary integrals"""

nn = 3
"""Number of nodes in each direction for regular grids"""

nnx = None
"""Number of nodes in the x-direction for regular grids"""

nny = None
"""Number of nodes in the y-direction for regular grids"""

nnz = None
"""Number of nodes in the z-direction for regular grids"""

triangleOptions="q30DenA"
"""Options string for triangle or tetGen"""

triangleFlag=0
"""Set the diagonal direction when triangulating a quadrilateral mesh

0 - right leaning
1 - alternating 'union jack'
2 - left leaning
"""

nLevels = 1
"""Number of levels for multilevel mesh"""

subgridError = None
"""The subgrid error object of a type derived from :class:`proteus.SubgridError.SGE_base`"""

massLumping = False
"""Boolean to lump mass matrix"""

reactionLumping = False
"""Boolean to lump reaction term"""

shockCapturing = None
"""The shock capturing diffusion object of a type derived from :class:`proteus.ShockCapturing.ShockCapturing_base`"""

numericalFluxType = None
"""A numerical flux class of type :class:`proteus.NumericalFlux.NF_base`"""

multilevelNonlinearSolver  = NLNI
"""A multilevel nonlinear solver class of type :class:`proteus.NonlinearSolvers.MultilevelNonlinearSolver`"""

levelNonlinearSolver = Newton
"""A nonlinear solver class of type :class:`proteus.NonlinearSolvers.NonlinearSolver`"""

nonlinearSmoother = None
"""A nonlinear solver class of type :class:`proteus.NonlinearSolvers.NonlinearSolver`"""

fullNewtonFlag = True
"""Boolean to do full Newton or modified Newton"""

nonlinearSolverNorm = l2Norm
"""Norm to use for nonlinear algebraic residual"""

nonlinearSolverUNorm = None
"""Norm for the Newton increment and iterate in the 'u', 'du' and 'r_and_du' tests: a callable
(as nonlinearSolverNorm), 'L2' (finite element L2 norm, sqrt(x^T M x), consistent mass) or
'L2_lumped'. None uses nonlinearSolverNorm. The 'L2' options need model support (femL2Norm)."""

nl_rtol_du = 1.0e-4
"""Relative tolerance of the 'du' and 'r_and_du' tests: ||du|| < nl_rtol_du*||u|| + nl_atol_du"""

nl_atol_du = 1.0e-16
"""Absolute tolerance of the 'du' and 'r_and_du' tests"""

tolFac = 0.01

atol = 1.0e-8

maxNonlinearIts =10

maxLineSearches =10

psitc = {'nStepsForce':3,'nStepsMax':100}

matrix = SparseMatrix

multilevelLinearSolver = LU

levelLinearSolver = LU

computeEigenvalues = False

computeEigenvectors = None#'left','right'

linearSmoother = None

linearSmootherOptions = ()

linTolFac = 0.001

conservativeFlux = None

checkMass = False

multigridCycles = 2

preSmooths = 2

postSmooths = 2

computeLinearSolverRates = False

printLinearSolverInfo = False

computeLevelLinearSolverRates = False

printLevelLinearSolverInfo = False

computeLinearSmootherRates = False

printLinearSmootherInfo = False

linearSolverMaxIts = 1000

linearWCycles = 3

linearPreSmooths = 3

linearPostSmooths = 3

computeNonlinearSolverRates=True

printNonlinearSolverInfo=False

computeNonlinearLevelSolverRates=False

printNonlinearLevelSolverInfo=False

computeNonlinearSmootherRates=False

printNonlinearSmootherInfo=False

nonlinearPreSmooths=3

nonlinearPostSmooths=3

nonlinearWCycles=3

useEisenstatWalker=False

maxErrorFailures=10

maxSolverFailures=10

needEBQ_GLOBAL = False

needEBQ = False

auxiliaryVariables=[]

periodicDirichletConditions=None

restrictFineSolutionToAllMeshes=False
parallelPartitioningType = MeshParallelPartitioningTypes.node
#default number of layers to use > 1 with element partition means
#C0P1 methods don't need to do communication in global element assembly
#nodal partitioning does not need communication for C0P1 (has overlap 1) regardless
nLayersOfOverlapForParallel = 1

parallelPeriodic=False#set this to true and use element,0 overlap to use periodic BC's in parallel

nonlinearSolverConvergenceTest = 'r'
levelNonlinearSolverConvergenceTest = 'r'

linearSolverConvergenceTest = 'r' #r,its,r-true for true residual
#we can add this if desired for setting solver specific options in petsc
linear_solver_options_prefix= None #

bcsTimeDependent = True
"""Allow optimizations if boundary conditions are not time dependent"""

adaptMesh = False
"""Adaptively refine the mesh in space"""

adaptMesh_nSteps = 10
"""Adapt the mesh every nSteps"""

adaptMesh_numIter = 2
"""If the mesh adaption  algorithm is iterative, do this many iterates"""

quad = None
hex = None
