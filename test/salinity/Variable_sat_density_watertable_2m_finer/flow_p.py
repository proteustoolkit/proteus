from proteus import *
from proteus.default_p import *
from proteus.kernels import Richards
import numpy
import numpy as np

try:
    from .domain_liu import *
except ImportError:
    from domain_liu import *

nd = 2
galerkin = False

analyticalSolution = None

# BASE case values from the literature setup.
viscosity = 1.0e-3
density = 1000.0
gravity = 9.8
lengthScale = 1.0
timeScale = 1.0  # days

# Homogeneous soil properties for the BASE case.
intrinsic_permeability = 1.15e-11
Kx = 3.0  # m/day
Kz = Kx
thetaS1 = 0.285
thetaR1 = 0.076
mvg_alpha1 = 3.1
mvg_n1 = 5.65
mvg_m1 = 1.0 - 1.0 / mvg_n1

dimensionless_conductivity_x = Kx
dimensionless_conductivity_z = Kz
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
    alphaVGtypes[i] = mvg_alpha1
    nVGtypes[i] = mvg_n1
    thetaStypes[i] = thetaS1
    thetaRtypes[i] = thetaR1
    thetaSRtypes[i] = thetaStypes[i] - thetaRtypes[i]
    KsTypes[i, :] = [dimensionless_conductivity_x, dimensionless_conductivity_z]

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
    beta=0.000,
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

rho_fw = 1.0
rho_sw = rho_fw * 1.025
mf_fw = 0.0
mf_sw = 1.0
epsilon = (rho_sw - rho_fw) / rho_fw
drho_dmf = epsilon * rho_fw

# BASE case: 2 m unsaturated thickness in a 10 m deep domain.
water_table_depth = 5.0
water_table_elevation = L[1] - water_table_depth
side_total_head = water_table_elevation  # H = 8 m

# Centered 20 m surface strip, matching the reference geometry.
infiltration_x0 = 40.0
infiltration_x1 = 60.0

# Express day-based rates directly in the input deck.
# Richards uses outward
# normal flux, so infiltration into the domain is negative.
top_flux = -1.e-2  # m/day  -> q/Ks = 0.0167; front advances ~35 m in 200 d, stops ~15 m above bottom


def hydrostatic_pressure_head(x):
    # Constant total head H = 8 m on the side boundaries, written as
    # pressure head psi = H - z.
    return side_total_head - x[1]


class HydrostaticIC:
    def uOfXT(self, x, t):
        return hydrostatic_pressure_head(x)


def getHydrostaticHeadDirichletBCs(x, tag):
    if tag in (boundaryTags["left"], boundaryTags["right"]):
        return lambda x, t: hydrostatic_pressure_head(x)


dirichletConditions = {0: getHydrostaticHeadDirichletBCs}
initialConditions = {0: HydrostaticIC()}


def on_top_infiltration_strip(x, tag, tol=1.0e-8):
    return (
        tag == boundaryTags["top"]
        and abs(x[1] - L[1]) <= tol
        and infiltration_x0 - tol <= x[0] <= infiltration_x1 + tol
    )


def getFlowAdvectiveFluxBCs(x, tag):
    if tag == boundaryTags["top"]:
        if infiltration_x0 <= x[0] <= infiltration_x1:
            return lambda x, t: top_flux
        return lambda x, t: 0.0
    if tag == boundaryTags["bottom"]:
        return lambda x, t: 0.0


fluxBoundaryConditions = {0: "setFlow"}
advectiveFluxBoundaryConditions = {0: getFlowAdvectiveFluxBCs}
diffusiveFluxBoundaryConditions = {0: {}}
