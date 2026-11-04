from __future__ import print_function
from __future__ import absolute_import
from builtins import range
from proteus import *
from proteus.default_n import *
from domain_rg import *
from tranport_p import *
#from thelper_tadr import *
parallel= True

# try:
#     from .thelper_tadr_p import *
#     from .thelper_tadr import *
# except:
#     from thelper_tadr_p import *
#     from thelper_tadr import *


multilevelNonlinearSolver  = Newton
#if coefficients.STABILIZATION_TYPE  ['Galerkin', 'VMS']: #SUPG
if coefficients.STABILIZATION_TYPE in  [0,1]: #SUPG
    levelNonlinearSolver = Newton
    maxLineSearches = 0
    fullNewtonFlag = True
    updateJacobian = True
    timeIntegration = BackwardEuler_cfl
elif coefficients.STABILIZATION_TYPE== 1: #'TaylorGalerkinEV':
    levelNonlinearSolver = TwoStageNewton
    fullNewtonFlag = False
    updateJacobian = False
    timeIntegration = BackwardEuler_cfl
else:
    fullNewtonFlag = False
    updateJacobian = False
    timeIntegration = TADR.RKEV # SSP33
    if coefficients.LUMPED_MASS_MATRIX==True:
        levelNonlinearSolver = ExplicitLumpedMassMatrix
    else:
        levelNonlinearSolver = ExplicitConsistentMassMatrixForVOF
SSPOrder = 1
stepController = Min_dt_controller
runCFL = 0.5
timeOrder = SSPOrder
nStagesTime = SSPOrder

# if useHex:
#     hex=True
#     quad=True
#     if pDegree_tadr == 1:
#         femSpaces = {0:C0_AffineLinearOnCubeWithNodalBasis}
#     elif pDegree_tadr == 2:
#         if useBernstein:
#             femSpaces = {0:C0_AffineBernsteinOnCube}
#         else:
#             femSpaces = {0:C0_AffineLagrangeOnCubeWithNodalBasis}
#     else:
#         print("pDegree = %s not recognized " % pDegree_tadr)
#     elementQuadrature = CubeGaussQuadrature(nd,tadr_quad_order)
#     elementBoundaryQuadrature = CubeGaussQuadrature(nd-1,tadr_quad_order)
# else:
#     if pDegree_tadr == 1:
#         femSpaces = {0:C0_AffineLinearOnSimplexWithNodalBasis}
#     elif pDegree_tadr == 2:
#         femSpaces = {0:C0_AffineQuadraticOnSimplexWithNodalBasis}
#     else:
#         print("pDegree = %s not recognized " % pDegree_tadr)
#     elementQuadrature = SimplexGaussQuadrature(nd,tadr_quad_order)
#     elementBoundaryQuadrature = SimplexGaussQuadrature(nd-1,tadr_quad_order)

numericalFluxType = TADR.NumericalFlux
#numericalFluxType = DoNothing
#numericalFluxType = Advection_DiagonalUpwind_IIPG_exterior # PERIODIC

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

#tnList=[0.,1E-6]+[float(n)*ct.T/float(ct.nDTout) for n in range(1,ct.nDTout+1)]
