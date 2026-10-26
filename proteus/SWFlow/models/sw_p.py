from proteus import *
from proteus.default_p import *
from proteus.mprans import SW2DCV
from proteus.Domain import RectangularDomain
import numpy as np
from proteus import Context

# *********************************************** #
# ********** READ FROM myTpFlowProblem ********** #
# *********************************************** #
ct = Context.get()
mySWFlowProblem = ct.mySWFlowProblem
genMesh=mySWFlowProblem.genMesh
physical_parameters = mySWFlowProblem.physical_parameters
numerical_parameters = mySWFlowProblem.swe_parameters
initialConditions = mySWFlowProblem.initialConditions
boundaryConditions = mySWFlowProblem.boundaryConditions
bathymetry = mySWFlowProblem.bathymetry
reflecting_BCs = mySWFlowProblem.reflectingBCs
analyticalSolution = mySWFlowProblem.analyticalSolution

# DOMAIN #
nd = 2
domain = mySWFlowProblem.domain
if domain is None:
    meshfile = mySWFlowProblem.AdH_file

# ******************************** #
# ********** PARAMETERS ********** #
# ******************************** #
# PHYSICAL PARAMETERS #
g = physical_parameters['gravity']
LINEAR_FRICTION = physical_parameters['LINEAR_FRICTION']
mannings = physical_parameters['mannings']
porosity = physical_parameters.get('porosity', None)
drag_alpha = physical_parameters.get('drag_alpha', None)
drag_beta = physical_parameters.get('drag_beta', None)

# NUMERICAL PARAMETERS #
cE = numerical_parameters['cE']
LUMPED_MASS_MATRIX = numerical_parameters['LUMPED_MASS_MATRIX']
INTEGRATE_BY_PARTS_WALLS = numerical_parameters.get('INTEGRATE_BY_PARTS_WALLS', 1)
BOUNDS_RELAXATION = numerical_parameters.get('BOUNDS_RELAXATION', 1)
BOUNDS_EXCLUDE_SELF = numerical_parameters.get('BOUNDS_EXCLUDE_SELF', 0)
WALL_RIEMANN = numerical_parameters.get('WALL_RIEMANN', 0)
LOW_ORDER_ONLY = numerical_parameters.get('LOW_ORDER_ONLY', 0)
NODAL_H_BOUNDS = numerical_parameters.get('NODAL_H_BOUNDS', 0)

# ********************************** #
# ********** COEFFICIENTS ********** #
# ********************************** #
LevelModelType = SW2DCV.LevelModel
coefficients = SW2DCV.Coefficients(g=g,
                                   bathymetry=bathymetry,
                                   cE=cE,
                                   LUMPED_MASS_MATRIX=LUMPED_MASS_MATRIX,
                                   LINEAR_FRICTION=LINEAR_FRICTION,
                                   mannings=mannings,
                                   INTEGRATE_BY_PARTS_WALLS=INTEGRATE_BY_PARTS_WALLS,
                                   BOUNDS_RELAXATION=BOUNDS_RELAXATION,
                                   BOUNDS_EXCLUDE_SELF=BOUNDS_EXCLUDE_SELF,
                                   WALL_RIEMANN=WALL_RIEMANN,
                                   LOW_ORDER_ONLY=LOW_ORDER_ONLY,
                                   NODAL_H_BOUNDS=NODAL_H_BOUNDS,
                                   # optional: {'where': f([x, y]) -> bool, 'stage': f(x, y, t)}
                                   tailwater=boundaryConditions.get('tailwater', None),
                                   porosity=porosity,
                                   drag_alpha=drag_alpha,
                                   drag_beta=drag_beta,
                                   # optional: walls([x, y]) -> bool marks the
                                   # boundary nodes that are walls; the rest are open
                                   wall_function=boundaryConditions.get('walls', None),
                                   # optional: drag_law(state) -> nodal beta, re-evaluated every step
                                   drag_law=physical_parameters.get('drag_law', None))

# **************************************** #
# ********** INITIAL CONDITIONS ********** #
# **************************************** #
initialConditions = {0: initialConditions['water_height'],
                     1: initialConditions['x_mom'],
                     2: initialConditions['y_mom']}

# ***************************************** #
# ********** BOUNDARY CONDITIONS ********** #
# ***************************************** #
dirichletConditions = {0: boundaryConditions['water_height'],
                       1: boundaryConditions['x_mom'],
                       2: boundaryConditions['y_mom']}

fluxBoundaryConditions = {0: 'outFlow',
                          1: 'outFlow',
                          2: 'outFlow'}
advectiveFluxBoundaryConditions = {0: lambda x, flag: lambda x, t: 0.0,
                                   1: lambda x, flag: lambda x, t: 0.0,
                                   2: lambda x, flag: lambda x, t: 0.0}
diffusiveFluxBoundaryConditions = {0: {},
                                   1: {1: lambda x, flag: lambda x, t: 0.0},
                                   2: {2: lambda x, flag: lambda x, t: 0.0}}
# **************************************** #
# ********** ANALYTICAL SOLUTION ********* #
# **************************************** #
if (mySWFlowProblem.analyticalSolution != None):
    analyticalSolution = {0: analyticalSolution['h_exact'],
                          1: analyticalSolution['hu_exact'],
                          2: analyticalSolution['hv_exact']}
