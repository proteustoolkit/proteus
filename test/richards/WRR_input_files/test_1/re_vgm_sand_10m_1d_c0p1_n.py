from proteus import *
from proteus.default_n import *
from re_vgm_sand_10m_1d_p import *


timeIntegrator = ForwardIntegrator
useFLCBDF = False#True#False
useGustafsson = False#True#
atol_u[0] = 1.0e-5
rtol_u[0] = 1.0e-5
#DT = 1.0e-6#None#0.025#1.0e-1/timeScale
tnList = [0.0]; nDTout = ct.nDT if ct.nDT > 0 else 1000#cek don't have stability worked out yet, blows up for large time steps
for i in range(nDTout):
    tnList.append(0.0+(i+1)*(T)/float(nDTout))
if useFLCBDF:
    timeIntegration = FLCBDF
    stepController  = FLCBDF_controller
    systemStepControllerType = SplitOperator.Sequential_MinFLCBDFModelStep
elif useGustafsson:
    #timeIntegration = BackwardEuler
    timeIntegration = VBDF
    timeOrder = 2
    stepController = GustafssonFullNewton_dt_controller#FixedStep
    #nDTout = 1#int(T/DT)#int(T/DT) #100#int(T/DT)
    #for controlling time stepping
    nonlinearConvergenceRate_ref = 0.2#0.3#0.2
    useInitialGuessPredictor= True
    stepExact = True
    systemStepControllerType = SplitOperator.Sequential_MinAdaptiveModelStep
else:
    timeIntegration = BackwardEuler
    stepController = HeuristicNL_dt_controller#FixedStep
    #nDTout = 1#int(T/DT)#int(T/DT) #100#int(T/DT)
    #for controlling time stepping
    nonlinearIterationsFloor = 8
    nonlinearIterationsCeil  = 16
    dtNLgrowFactor = 2
    dtNLreduceFactor = 0.5
    dtNLfailureReduceFactor = 0.5
    useInitialGuessPredictor= True
    stepExact = True

if not galerkin:
    timeIntegration = Richards.ThetaScheme
#    timeIntegration = Richards.RKEV
timeOrder = 1
stepController = FixedStep
#systemStepControllerType = SplitOperator.Sequential_FixedStep
#dt_system_fixed = 0.001
#nDTout = 1#int(T/DT)#int(T/DT) #100#int(T/DT)
#for controlling time stepping
nonlinearIterationsFloor = 4
nonlinearIterationsCeil  = 8
dtNLgrowFactor = 2
dtNLreduceFactor = 0.5
dtNLfailureReduceFactor = 0.5
useInitialGuessPredictor= True
stepExact = True

femSpaces = {0:C0_AffineLinearOnSimplexWithNodalBasis}

elementQuadrature = SimplexGaussQuadrature(nd,3)

elementBoundaryQuadrature = SimplexGaussQuadrature(nd-1,1)

#elementQuadrature = SimplexLobattoQuadrature(nd,1)
#
#elementBoundaryQuadrature = SimplexLobattoQuadrature(nd-1,1)

#nn=101
#nLevels = 4
nn= 41#3*2**2
if ct.nn > 0:
    nn = ct.nn
nLevels = 1#10-2
#nn=3
#nLevels = 10

subgridError = None
#subgridError = AdvectionDiffusionReaction_ASGS(coefficients,nd,stabFlag='2',lag=True)
#subgridError = AdvectionDiffusionReaction_ASGS(coefficients,nd,stabFlag='2',lag=False)
#subgridError = AdvectionDiffusionReactionTransientSubscales_ASGS(coefficients,nd,stabFlag='1',lag=True,trackSubScales=True,useHarariDirectly=False,
#                                                                 limit_tau_t=True,tau_t_limit_max=0.9)
#subgridError = AdvectionDiffusionReactionHaukeSangalliInterpolant_ASGS(coefficients,nd,stabFlag='2',lag=True,interpolationFemSpaceType=femSpaces[0])

#masslumping = False
massLumping = True

shockCapturing = None
#shockCapturing = ResGradQuad_SC(coefficients,nd,shockCapturingFactor=0.99,lag=True)
#shockCapturing = ResGradQuadDelayLag_SC(coefficients,nd,shockCapturingFactor=0.99,lag=True,nStepsToDelay=10)
#shockCapturing = JaffreGradU_SC(coefficients,nd,shockCapturingFactor=0.9,lag=True,betaPower=0.1)
#shockCapturing = ResGrad_SC(coefficients,nd,shockCapturingFactor=0.99,lag=True)
numericalFluxType = Richards_IIPG_exterior #need weak for parallel and global conservation 
#numericalFluxType = Advection_DiagonalUpwind_Diffusion_IIPG_exterior #need weak for parallel and global conservation

#multilevelNonlinearSolver  = NLStarILU
#multilevelNonlinearSolver  = NLGaussSeidel
#multilevelNonlinearSolver  = NLJacobi
#multilevelNonlinearSolver  = NLNI
#multilevelNonlinearSolver  = FAS
multilevelNonlinearSolver = Newton

#levelNonlinearSolver = FAS
if galerkin:
    levelNonlinearSolver = Newton
else:
    levelNonlinearSolver = Newton
    #levelNonlinearSolver = ExplicitLumpedMassMatrixForRichards
#levelNonlinearSolver = ExplicitConsistentMassMatrixForRichards
#levelNonlinearSolver = NLStarILU
#levelNonlinearSolver = NLGaussSeidel
#levelNonlinearSolver = NLJacobi

nonlinearSmoother = NLStarILU
#nonlinearSmoother = NLGaussSeidel
#nonlinearSmoother = NLJacobi

fullNewtonFlag = True

tolFac = 0.0

nl_atol_res = 1.0e-8

maxNonlinearIts = 100#1001
if ct.nl_solver == 'lscheme':
    maxNonlinearIts = 1000  # linear convergence
maxLineSearches =100
if ct.nl_solver != 'newton':
    maxLineSearches = 0  # lscheme: contracts in energy, not monotonically in r; hybrid: replaces it

matrix = SparseMatrix
#matrix = Numeric.array

multilevelLinearSolver = LU
#multilevelLinearSolver = NI
#multilevelLinearSolver = Jacobi
#multilevelLinearSolver = GaussSeidel
#multilevelLinearSolver = StarILU

#levelLinearSolver = LU
computeEigenvalues = False#True
#computeEigenvalues = False
levelLinearSolver = LU#MGM

#linearSmoother = Jacobi
#linearSmoother = GaussSeidel
linearSmoother = StarILU

linTolFac = 0.01

#conservativeFlux = {0:'pwl'}
