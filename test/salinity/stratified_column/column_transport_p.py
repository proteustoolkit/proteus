from proteus.default_p import *
from proteus.mprans import TADR
import numpy as np

try:
    from .column_domain import *
    from .column_flow_p import rho_f, rho_s, concentration
except ImportError:
    from column_domain import *
    from column_flow_p import rho_f, rho_s, concentration

LevelModelType = TADR.LevelModel

# no molecular diffusion: the mechanical dispersion vanishes with the velocity, so the
# stratified state is an exact steady solution
coefficients = TADR.Coefficients(aOfX={0: lambda x: np.zeros((2, 2))},
                                 alpha_L=0.1,
                                 alpha_T=0.01,
                                 Dm=0.0,
                                 V_model=0,
                                 specified_velocity=False,
                                 checkMass=False,
                                 FCT=True,
                                 LUMPED_MASS_MATRIX=False,
                                 STABILIZATION_TYPE="Compatible",
                                 diagonal_conductivity=True,
                                 rho_f=rho_f,
                                 rho_s=rho_s)
coefficients.variableNames = ["u"]


class StratifiedIC:
    def uOfXT(self, x, t):
        return concentration(x[1])


initialConditions = {0: StratifiedIC()}
dirichletConditions = {0: lambda x, flag: None}
fluxBoundaryConditions = {0: "setFlow"}
advectiveFluxBoundaryConditions = {0: lambda x, flag: lambda x, t: 0.0}
diffusiveFluxBoundaryConditions = {0: {0: lambda x, flag: lambda x, t: 0.0}}
