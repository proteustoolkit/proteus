from __future__ import absolute_import

import numpy as np
from proteus import *
from proteus.default_p import *
from proteus import Profiling
from proteus.mprans import TADR

try:
    from .domain_elder import *
except ImportError:
    from domain_elder import *

physicalDiffusion = 0.0
refinement = 0
shockCapturingFactor_tadr = 0.0 #0.2
lag_shockCapturing_tadr = True

parallel = True
linearSmoother = None
checkMass = False


class MyCoefficients(TADR.Coefficients):
    pass


LevelModelType = TADR.LevelModel
logEvent = Profiling.logEvent
a0 = 0.30802  # m^2/day  (Elder Dm = 3.565e-6 m^2/s = 0.30802 m^2/day)

def a(x):
    return np.array([[a0, 0.0], [0.0, a0]])


aOfX = {0: a}
alpha_L = 0.0   # Elder: no mechanical dispersion
alpha_T = 0.0
Dm = 0.30802  # m^2/day  (Elder Dm = 3.565e-6 m^2/s)

coefficients = MyCoefficients(
    aOfX,
    alpha_L=alpha_L,
    alpha_T=alpha_T,
    Dm=Dm,
    porosity=0.1,
    V_model=0,
    specified_velocity=False,
    checkMass=checkMass,
    FCT=True,  # TADR FCT on, as in the UHD L4 runs
    LUMPED_MASS_MATRIX=True,
    forceStrongConditions=True,
    STABILIZATION_TYPE=5,
    diagonal_conductivity=True,
    ENTROPY_TYPE='POWER',
    cE=0.1,
    cK=1.0,
    physicalDiffusion=0.0,
    rho_f=1.0,
    rho_s=1.2,
)
coefficients.variableNames = ["u"]

# Elder solute source: C = 1 over the central 300 m of the top surface,
# C = 0 over the entire base.
cin = 1.0
conc_x0 = 150.0
conc_x1 = 450.0


def in_top_source_strip(x, tag, tol=1.0e-6):
    return tag == boundaryTags["top"] and conc_x0 - tol <= x[0] <= conc_x1 + tol


def getConcDirichletBCs(x, tag):
    if x[1] == L[1]:
        if (x[0] >= conc_x0 and
            x[0] <= conc_x1):
            return lambda x, t: cin
    if x[1] == 0.0:
        return lambda x, t: 0.0


dirichletConditions = {0: getConcDirichletBCs}


class ShockIC:
    def uOfXT(self, x, t):
        bc = getConcDirichletBCs(x, 0)
        if bc != None:
            return 0.0 #bc(x, t)
        else:
            return 0.0

    def uOfX(self, x):
        return self.uOfXT(x, 0.0)


initialConditions = {0: ShockIC()}


def getZeroMassDiffusiveFluxBCs(x, tag):
    # Dirichlet nodes (top source strip, base) are governed by the strong
    # BC; every other boundary is a closed, zero-diffusive-flux wall.
    if in_top_source_strip(x, tag):
        return None
    if tag == boundaryTags["bottom"]:
        return None
    return lambda x, t: 0.0


def getZeroMassAdvectiveFluxBCs(x, tag):
    if in_top_source_strip(x, tag):
        return None
    if tag == boundaryTags["bottom"]:
        return None
    # Closed box: no advective flux across the remaining walls.
    return lambda x, t: 0.0


fluxBoundaryConditions = {0: "setFlow"}
advectiveFluxBoundaryConditions = {0: getZeroMassAdvectiveFluxBCs}
diffusiveFluxBoundaryConditions = {0: {0: getZeroMassDiffusiveFluxBCs}}
