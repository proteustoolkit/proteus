from proteus import *
from proteus.default_n import *
from proteus.gwvd import GwvdCoefficients, Gwvd_consolidated_classes
from gwvd_2dbox_cg import *
from gwvd_2dbox_flow_p_cg import *

# Time Integration and Stepcontrol
if (transient):
    timeIntegration =Gwvd_consolidated_classes.NonConservativeBackwardEuler
    DT=DT
else:
    timeIntegration=None

stepController = Gwvd_consolidated_classes.Fixed_dt_controller_systemit# Min_dt_cfl_controller


# FEM Spaces and Quadrature Rules

femSpaces =  {0:C0_AffineLinearOnSimplexWithNodalBasis} 
#femSpaces =  {0:C0_AffineQuadraticOnSimplexWithNodalBasis}

elementQuadrature = SimplexGaussQuadrature(nd,gw_quad_order)
elementBoundaryQuadrature = SimplexGaussQuadrature(nd-1,gw_quad_order)
#elementQuadrature = SimplexLobattoQuadrature(nd,gw_quad_order)
#elementBoundaryQuadrature = SimplexLobattoQuadrature(nd-1,gw_quad_order)

nLevels =1 

# Stabilization and Shockcapturing

subgridError = None
shockCapturing = None

# Non-linear solvers 

multilevelNonlinearSolver  = Newton
levelNonlinearSolver = Newton
fullNewtonFlag = True
nonlinearSolverConvergenceTest = 'rits'
levelNonlinearSolverConvergenceTest = 'rits'    
# Solver Tolerance Factors

tolFac = 1.0e-8 
nl_atol_res = 1.0e-8 
linTolFac = 1.0e-10

matrix = SparseMatrix
#matrix = Numeric.array


if (parallel):
    multilevelLinearSolver = PETSc
    levelLinearSolver = PETSc
    nLayersOfOverlapForParallel = 2
    numericalFluxType = NumericalFlux.Diffusion_IIPG_exterior
    if (vel_pp==1):
        conservativeFlux ={0:'pwl-opt'} #{0:'point-eval'} #{0:'pwl-opt'} # #{0:'pwc'}
    else:
        conservativeFlux ={0:'point-eval'}#{0:'pwl-opt'} #{0:'point-eval'} #{0:'pwl-opt'} # #{0:'pwc'}
else:
    multilevelLinearSolver = LU
    levelLinearSolver = LU
    numericalFluxType =None# Diffusion_IIPG_exterior
    if (vel_pp==0):
        conservativeFlux ={0:'point-eval'}#  {0:'pwl'} # {0:'pwc'}
    else:
        conservativeFlux ={0:'pwl'} #{0:'point-eval'} #{0:'pwl-opt'} # #{0:'pwc'}

# Nonlinear Solve Info
nonlinearSolverConvergenceTest = 'rits'
levelNonlinearSolverConvergenceTest = 'rits'
maxNonlinearIts =1
tolFac = 1.0e-8 # 1.0e-8
atol=1.0e-8
nl_atol_res = 1.0e-10 #1.0e-8
matrix=SparseMatrix
linTolFac = 1.0e-10
maxLineSearches =0

