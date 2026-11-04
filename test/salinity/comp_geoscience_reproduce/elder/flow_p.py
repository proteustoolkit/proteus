from proteus import *
from proteus.default_p import *
from proteus.richards import Richards
import numpy
import numpy as np

try:
    from .domain_elder import *
except ImportError:
    from domain_elder import *

nd = 2
galerkin = False

analyticalSolution = None

# BASE case values from the literature setup.
viscosity = 1.0e-3
density = 1000.0
gravity = 9.81
lengthScale = 1.0
timeScale = 1.0  # days

# Elder problem soil properties (Voss & Souza 1987).
# k = 4.845e-13 m^2 -> K = k*rho*g/mu = 4.845e-13*1000*9.81/1e-3
#                        = 4.753e-6 m/s = 0.4107 m/day.
intrinsic_permeability = 4.845e-13
Kx = 0.4107  # m/day
Kz = Kx
thetaS1 = 0.1  # porosity epsilon = 0.1
thetaR1 = 0.0
# Voss & Souza's Elder box is a fully saturated confined aquifer: their (4) has
# no retention curve and no k_rw.  alpha=3.1 put the air entry at 1/alpha=0.32 m,
# and psi along the top runs down to -1.17 m (the corners are pinned at 0 and
# rho0 is the lightest fluid, so the surface over any denser column sits at
# negative gauge head).  That desaturated ~1400 top nodes to theta ~ 7e-4 and
# k_r ~ 3e-7 -- sealing the top row, collapsing TADR's aliased porosity, and
# leaving near-empty rows in the STAB=2 Jacobian.  alpha=1e-5 moves the air
# entry to 1e5 m, so theta=thetaS and k_r=1 to machine precision over the whole
# psi range the box visits (exact out to psi=-10 m, k_r=0.99999999 at -1000 m).
# n stays at 5.65: with alpha*psiC << 1 a LARGE n is what kills (alpha*psiC)^n
# fastest -- n=1.2 would still give k_r=0.80 at psi=-1.2 m.
mvg_alpha1 = 1.0e-5
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
    FCT=True, FCT_BND_DIFF=True,  # Barua edge-flux runs: boundary routing on (default off since #173)
      # Richards FCT on, as in the UHD L4 runs
    num_fct_iter=0,
    cE=1.0,
    uL=0.0,
    uR=1.0,
    cK=1.0,
    outputQuantDOFs=False,
)

rho_fw = 1.0
rho_sw = rho_fw * 1.2  # rho = 1000 + 200*C  ->  rho_max/rho_fw = 1.2
mf_fw = 0.0
mf_sw = 1.0
epsilon = (rho_sw - rho_fw) / rho_fw
drho_dmf = epsilon * rho_fw

# Fully saturated Elder case: pressure is initially hydrostatic with
# psi = 0 at the top surface (z = L[1]) and hydrostatic at the bottom.
side_total_head = L[1]  # H = L[1] -> psi = L[1] - z


def hydrostatic_pressure_head(x):
    # Hydrostatic pressure head psi = H - z with H = L[1].
    return side_total_head - x[1]


class HydrostaticIC:
    def uOfXT(self, x, t):
        return hydrostatic_pressure_head(x)


# Elder: the box is closed to flow; the only pressure constraint is
# p = 0 pinned at the two upper corners (psi = 0 there since z = L[1]).
def getFlowDirichletBCs(x, tag, tol=1.0e-6):
    if tag == boundaryTags["top"] and (
        abs(x[0]) <= tol or abs(x[0] - L[0]) <= tol
    ):
        return lambda x, t: 0.0


dirichletConditions = {0: getFlowDirichletBCs}
initialConditions = {0: HydrostaticIC()}


def getFlowAdvectiveFluxBCs(x, tag):
    # Closed box: no flow across any boundary.
    return lambda x, t: 0.0


fluxBoundaryConditions = {0: "setFlow"}
advectiveFluxBoundaryConditions = {0: getFlowAdvectiveFluxBCs}
diffusiveFluxBoundaryConditions = {0: {}}
