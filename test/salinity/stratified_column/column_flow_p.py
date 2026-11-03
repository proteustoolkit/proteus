from proteus.default_p import *
from proteus.richards import Richards
import numpy as np

try:
    from .column_domain import *
except ImportError:
    from column_domain import *

# soil and density as in the levee case (Levee_contamination_Density/Case_D), days and metres
viscosity = 8.9e-4
density = 998.2
gravity = 9.8
permeability = 1.06 * viscosity / (gravity * density)
hydraulic_conductivity = density * gravity * permeability / viscosity
thetaS = 0.41
thetaR = 0.065
mvg_alpha = 7.5
mvg_n = 1.89

rho_f = 1.0
rho_s = 1.025
c_bottom = 1.0

nMediaTypes = 1
alphaVGtypes = np.full((nMediaTypes + 1,), mvg_alpha)
nVGtypes = np.full((nMediaTypes + 1,), mvg_n)
thetaRtypes = np.full((nMediaTypes + 1,), thetaR)
thetaSRtypes = np.full((nMediaTypes + 1,), thetaS - thetaR)
KsTypes = np.full((nMediaTypes + 1, 2), hydraulic_conductivity)

LevelModelType = Richards.LevelModel
coefficients = Richards.Coefficients(nd,
                                     KsTypes,
                                     nVGtypes,
                                     alphaVGtypes,
                                     thetaRtypes,
                                     thetaSRtypes,
                                     gravity=np.array([0.0, -1.0, 0.0]),
                                     density=rho_f,
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
                                     outputQuantDOFs=False)


def concentration(z):
    return c_bottom * (1.0 - z / L[1])


def hydrostatic_pressure_head(z):
    # psi(z) = -int_{z_wt}^{z} rho_T(c(s)) / rho_f ds, rho_T = rho_f (1 + eps c)
    eps = (rho_s - rho_f) / rho_f
    return -((1.0 + eps * c_bottom) * (z - water_table_z)
             - eps * c_bottom * (z ** 2 - water_table_z ** 2) / (2.0 * L[1]))


class HydrostaticIC:
    def uOfXT(self, x, t):
        return hydrostatic_pressure_head(x[1])


initialConditions = {0: HydrostaticIC()}
dirichletConditions = {0: lambda x, flag: None}
fluxBoundaryConditions = {0: "setFlow"}
advectiveFluxBoundaryConditions = {0: lambda x, flag: lambda x, t: 0.0}
diffusiveFluxBoundaryConditions = {0: {}}
