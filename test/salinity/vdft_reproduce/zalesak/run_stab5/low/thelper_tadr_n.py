from proteus import *
from proteus.default_n import *
try:
    from .thelper_tadr_p import *
    from .thelper_tadr import *
except:
    from thelper_tadr_p import *
    from thelper_tadr import *

nd = ct.nd

multilevelNonlinearSolver  = Newton
if ct.STABILIZATION_TYPE==0: #SUPG
    levelNonlinearSolver = Newton
    fullNewtonFlag = True
    updateJacobian = True
    timeIntegration = BackwardEuler_cfl
elif ct.STABILIZATION_TYPE==1: #TaylorGalerkin
    levelNonlinearSolver = TwoStageNewton
    fullNewtonFlag = False
    updateJacobian = False
    timeIntegration = BackwardEuler_cfl
elif ct.STABILIZATION_TYPE==5: #ImplicitEV -- backward Euler, NOT CFL limited
    # STAB=5 is implicit: it must use Newton + BackwardEuler, not the explicit
    # SSP/RKEV path the other edge-based types take.  Without this branch it
    # falls into the `else` below and is silently run by an explicit solver,
    # which is why it has never had test coverage.
    levelNonlinearSolver = Newton
    fullNewtonFlag = True
    updateJacobian = True
    timeIntegration = BackwardEuler_cfl
else:
    fullNewtonFlag = False
    updateJacobian = False
    timeIntegration = TADR.RKEV # SSP33
    if ct.LUMPED_MASS_MATRIX==True:
        levelNonlinearSolver = ExplicitLumpedMassMatrix
    else:
        levelNonlinearSolver = ExplicitConsistentMassMatrixForVOF

stepController = Min_dt_controller
runCFL = ct.cfl
timeOrder = ct.SSPOrder
nStagesTime = ct.SSPOrder

if useHex:
    hex=True
    quad=True
    if pDegree_tadr == 1:
        femSpaces = {0:C0_AffineLinearOnCubeWithNodalBasis}
    elif pDegree_tadr == 2:
        if useBernstein:
            femSpaces = {0:C0_AffineBernsteinOnCube}
        else:
            femSpaces = {0:C0_AffineLagrangeOnCubeWithNodalBasis}
    else:
        print("pDegree = %s not recognized " % pDegree_tadr)
    elementQuadrature = CubeGaussQuadrature(nd,tadr_quad_order)
    elementBoundaryQuadrature = CubeGaussQuadrature(nd-1,tadr_quad_order)
else:
    if pDegree_tadr == 1:
        femSpaces = {0:C0_AffineLinearOnSimplexWithNodalBasis}
    elif pDegree_tadr == 2:
        femSpaces = {0:C0_AffineQuadraticOnSimplexWithNodalBasis}
    else:
        print("pDegree = %s not recognized " % pDegree_tadr)
    elementQuadrature = SimplexGaussQuadrature(nd,tadr_quad_order)
    elementBoundaryQuadrature = SimplexGaussQuadrature(nd-1,tadr_quad_order)

#numericalFluxType = TADR.NumericalFlux
#numericalFluxType = DoNothing
numericalFluxType = Advection_DiagonalUpwind_IIPG_exterior # PERIODIC

shockCapturing = TADR.ShockCapturing(coefficients,nd,shockCapturingFactor=shockCapturingFactor_tadr,lag=lag_shockCapturing_tadr)

matrix = SparseMatrix
if parallel:
    multilevelLinearSolver = KSP_petsc4py
    levelLinearSolver = KSP_petsc4py
    linear_solver_options_prefix = 'tadr_'
    linearSolverConvergenceTest = 'r-true'
else:
    multilevelLinearSolver = LU
    levelLinearSolver = LU

if checkMass:
    auxiliaryVariables = [MassOverRegion()]

tnList=[0.,1E-6]+[float(n)*ct.T/float(ct.nDTout) for n in range(1,ct.nDTout+1)]
