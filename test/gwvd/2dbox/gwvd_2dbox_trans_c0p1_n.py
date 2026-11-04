from proteus import *
from proteus.default_n import *
from proteus.gwvd import GwvdCoefficients, Gwvd_consolidated_classes
from gwvd_2dbox_trans_p_cg import *
from gwvd_2dbox_cg import *

# Time Integration and Step control
timeIntegration =Gwvd_consolidated_classes.NonConservativeBackwardEuler#_cfl# TimeIntegration.BackwardEuler
stepController = Gwvd_consolidated_classes.Fixed_dt_controller_systemit#
stepController= Min_dt_cfl_controller
DT =DT
#runCFL =1.0# 0.1

#FEM Spaces and Quadrature Rules

femSpaces = {0:C0_AffineLinearOnSimplexWithNodalBasis}
#femSpaces = {0:C0_AffineQuadraticOnSimplexWithNodalBasis}
elementQuadrature = SimplexGaussQuadrature(nd,gw_quad_order)
elementBoundaryQuadrature = SimplexGaussQuadrature(nd-1,gw_quad_order)
#elementQuadrature = SimplexLobattoQuadrature(nd,gw_quad_order)
#elementBoundaryQuadrature = SimplexLobattoQuadrature(nd-1,gw_quad_order)

nLevels = 1 #1

# Stabilization and Shock Capturing

massLumping=False

subgridError =SubgridError.HamiltonJacobi_ASGS(coefficients,nd,stabFlag='2',lag=True)

#subgridError = AdvectionDiffusionReaction_ASGS(coefficients,nd,lag=False)

shockCapturing =ShockCapturing.ResGrad_SC(coefficients,nd,lag=True) #None

# Non-linear Solvers

multilevelNonlinearSolver  = Newton
levelNonlinearSolver = Newton
nonlinearSolverConvergenceTest ='rits'
levelNonlinearSolverConvergenceTest ='rits'    
fullNewtonFlag = True
maxNonlinearIts =1
maxLineSearches =0

# Solver Tolerance Factors
atol=1.0-8
tolFac = 1.0e-8 # 1.0e-8
nl_atol_res = 1.0e-8 #1.0e-8
a_tol_res=1.0e-8
r_tol_res=1.0e-8
linTolFac = 1.0e-10

# Parallel Implementation

if (parallel):
    multilevelLinearSolver = PETSc
    levelLinearSolver = PETSc
    numericalFluxType = NumericalFlux.Diffusion_IIPG_exterior
    nLayersOfOverlapForParallel = 2
    conservativeFlux =  None

else:
    multilevelLinearSolver = LU
    levelLinearSolver = LU
    numericalFluxType =None#GwvdNumericalFlux.HamiltonJacobi_DiagonalLesaintRaviart_Diffusion_NIPG_Inflow_BCs_exterior# None#Diffusion_IIPG_exterior
    conservativeFlux = None 

matrix = SparseMatrix
#smoother = Jacobi
