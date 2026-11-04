from proteus import *
from proteus.default_p import *
from proteus.richards import Richards
import numpy
import numpy as np

try:
    from .domain_rg import *
except ImportError:
    from domain_rg import *

nd = 2
galerkin = False

analyticalSolution = None

viscosity = 8.9e-4
density = 998.2
gravity = 9.8
lengthScale = 1.0
timeScale = 1.0  # days

# Homogeneous sandy foundation/levee properties used in the levee cases.
permeability = 1.06 * viscosity / (gravity * density)
thetaS = 0.41
thetaR = 0.065
mvg_alpha = 7.5
mvg_n = 1.89
mvg_m = 1.0 - 1.0 / mvg_n
hydraulic_conductivity = timeScale * density * gravity * permeability / (viscosity * lengthScale)

dimensionless_density = 1.0
dimensionless_gravity = np.array([0.0, -1.0, 0.0])

nMediaTypes = 1
alphaVGtypes = numpy.zeros((nMediaTypes + 1,), "d")
nVGtypes = numpy.zeros((nMediaTypes + 1,), "d")
thetaStypes = numpy.zeros((nMediaTypes + 1,), "d")
thetaRtypes = numpy.zeros((nMediaTypes + 1,), "d")
thetaSRtypes = numpy.zeros((nMediaTypes + 1,), "d")
KsTypes = numpy.zeros((nMediaTypes + 1, 2), "d")

for i in range(nMediaTypes + 1):
    alphaVGtypes[i] = mvg_alpha
    nVGtypes[i] = mvg_n
    thetaStypes[i] = thetaS
    thetaRtypes[i] = thetaR
    thetaSRtypes[i] = thetaStypes[i] - thetaRtypes[i]
    KsTypes[i, :] = [hydraulic_conductivity, hydraulic_conductivity]

LevelModelType = Richards.LevelModel
coefficients = Richards.Coefficients(
    nd,
    KsTypes,
    nVGtypes,
    alphaVGtypes,
    thetaRtypes,
    thetaSRtypes,
    gravity=dimensionless_gravity,
    density=dimensionless_density,
    beta=0.0,
    density_model=1,
    diagonal_conductivity=True,
    STABILIZATION_TYPE=2,
    ENTROPY_TYPE=1,
    LUMPED_MASS_MATRIX=False,
    FCT=False,
    num_fct_iter=0,
    cE=1.0,
    uL=0.0,
    uR=1.0,
    cK=1.0,
    outputQuantDOFs=False,
)

water_table_elevation = water_table_z
river_stage = river_level


def hydrostatic_pressure_head(x, water_level):
    return water_level - x[1]


class HydrostaticIC:
    def uOfXT(self, x, t):
        return hydrostatic_pressure_head(x, water_table_elevation)


def getHydrostaticHeadDirichletBCs(x, tag):
    if tag == boundaryTags["leftTop"]:
        return lambda x, t: hydrostatic_pressure_head(x, river_stage)


dirichletConditions = {0: getHydrostaticHeadDirichletBCs}
initialConditions = {0: HydrostaticIC()}


def getFlowAdvectiveFluxBCs(x, tag):
    if tag == boundaryTags["right"]:
        return None
    if tag != boundaryTags["leftTop"]:
        return lambda x, t: 0.0


fluxBoundaryConditions = {0: "setFlow"}
advectiveFluxBoundaryConditions = {0: getFlowAdvectiveFluxBCs}
diffusiveFluxBoundaryConditions = {0: {}}
