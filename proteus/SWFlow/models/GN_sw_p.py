from proteus import *
from proteus.default_p import *
from proteus.mprans import GN_SW2DCV
from proteus.Domain import RectangularDomain
import numpy as np
from proteus import Context

# *********************************************** #
# ********** READ FROM myTpFlowProblem ********** #
# *********************************************** #
ct = Context.get()
mySWFlowProblem = ct.mySWFlowProblem
genMesh = mySWFlowProblem.genMesh
physical_parameters = mySWFlowProblem.physical_parameters
numerical_parameters = mySWFlowProblem.swe_parameters
initialConditions = mySWFlowProblem.initialConditions
boundaryConditions = mySWFlowProblem.boundaryConditions
bathymetry = mySWFlowProblem.bathymetry
reflecting_BCs = mySWFlowProblem.reflectingBCs
analyticalSolution = mySWFlowProblem.analyticalSolution
waveConditions = mySWFlowProblem.waveConditions

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
gen_length = physical_parameters['gen_length']
gen_start = physical_parameters['gen_start']
abs_length = physical_parameters['abs_length']
abs_start = physical_parameters['abs_start']

# NUMERICAL PARAMETERS #
LUMPED_MASS_MATRIX = numerical_parameters['LUMPED_MASS_MATRIX']
INTEGRATE_BY_PARTS_WALLS = numerical_parameters.get('INTEGRATE_BY_PARTS_WALLS', 1)

# ********************************** #
# ********** COEFFICIENTS ********** #
# ********************************** #
LevelModelType = GN_SW2DCV.LevelModel
coefficients = GN_SW2DCV.Coefficients(g=g,
                                      bathymetry=bathymetry,
                                      LUMPED_MASS_MATRIX=LUMPED_MASS_MATRIX,
                                      LINEAR_FRICTION=LINEAR_FRICTION,
                                      mannings=mannings,
                                      gen_length=gen_length,
                                      gen_start=gen_start,
                                      abs_length=abs_length,
                                      abs_start=abs_start,
                                      waveConditions=waveConditions,
                                      INTEGRATE_BY_PARTS_WALLS=INTEGRATE_BY_PARTS_WALLS)

# **************************************** #
# ********** INITIAL CONDITIONS ********** #
# **************************************** #
initialConditions = {0: initialConditions['water_height'],
                     1: initialConditions['x_mom'],
                     2: initialConditions['y_mom'],
                     3: initialConditions['h_times_eta'],
                     4: initialConditions['h_times_w'],
                     5: initialConditions['h_times_beta']}

# ***************************************** #
# ********** BOUNDARY CONDITIONS ********** #
# ***************************************** #
dirichletConditions = {0: boundaryConditions['water_height'],
                       1: boundaryConditions['x_mom'],
                       2: boundaryConditions['y_mom'],
                       3: boundaryConditions['h_times_eta'],
                       4: boundaryConditions['h_times_w'],
                       5: boundaryConditions['h_times_beta']}
fluxBoundaryConditions = {0: 'outFlow',
                          1: 'outFlow',
                          2: 'outFlow',
                          3: 'outFlow',
                          4: 'outFlow',
                          5: 'outFlow'}
advectiveFluxBoundaryConditions = {0: lambda x, flag: None,
                                   1: lambda x, flag: None,
                                   2: lambda x, flag: None,
                                   3: lambda x, flag: None,
                                   4: lambda x, flag: None,
                                   5: lambda x, flag: None}
diffusiveFluxBoundaryConditions = {0: {},
                                   1: {1: lambda x, flag: None},
                                   2: {2: lambda x, flag: None},
                                   3: {3: lambda x, flag: None},
                                   4: {4: lambda x, flag: None},
                                   5: {5: lambda x, flag: None}}
# **************************************** #
# ********** ANALYTICAL SOLUTION ********* #
# **************************************** #
if (mySWFlowProblem.analyticalSolution is not None):
    analyticalSolution = {0: analyticalSolution['h_exact'],
                          1: analyticalSolution['hu_exact'],
                          2: analyticalSolution['hv_exact'],
                          3: analyticalSolution['heta_exact'],
                          4: analyticalSolution['hw_exact'],
                          5: analyticalSolution['hbeta_exact']}
# **************************************** #
# **********   WAVE CONDITIONS   ********* #
# **************************************** #
if (mySWFlowProblem.waveConditions is not None):
    waveConditions = {0: waveConditions['h'],
                          1: waveConditions['h_u'],
                          2: waveConditions['h_v'],
                          3: waveConditions['h_eta'],
                          4: waveConditions['h_w'],
                          5: waveConditions['h_beta']}
