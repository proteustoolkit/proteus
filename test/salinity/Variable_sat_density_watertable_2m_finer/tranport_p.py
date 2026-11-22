from __future__ import absolute_import

import numpy as np
from proteus import *
from proteus.default_p import *
from proteus import Profiling
from proteus.kernels import TADR

try:
    from .domain_liu import *
except ImportError:
    from domain_liu import *

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
a0 = 8.64e-6  # m^2/day  (Liu et al. Dm=1e-9 m²/s)

def a(x):
    return np.array([[a0, 0.0], [0.0, a0]])


aOfX = {0: a}
alpha_L = 0.1   
alpha_T = 0.01
Dm = 8.64e-6  # m^2/day  (Liu et al. Dm=1e-9 m²/s)

coefficients = MyCoefficients(
    aOfX,
    alpha_L=alpha_L,
    alpha_T=alpha_T,
    Dm=Dm,
    porosity=0.285,
    V_model=0,
    specified_velocity=False,
    checkMass=checkMass,
    FCT=True,
    LUMPED_MASS_MATRIX=True,
    forceStrongConditions=True,
    STABILIZATION_TYPE=2,
    diagonal_conductivity=True,
    ENTROPY_TYPE='POWER',
    cE=0.1,
    cK=1.0,
    physicalDiffusion=0.0,
    rho_f=1.0,
    rho_s=1.025,
)
coefficients.variableNames = ["u"]

# Use a normalized inlet concentration so the density law spans the
# literature density ratio from rho_f to rho_s.
cin = 1.0
infiltration_x0 = 40.0
infiltration_x1 = 60.0


def on_top_infiltration_strip(x, tag, tol=1.0e-8):
    return (
        tag == boundaryTags["top"]
        and abs(x[1] - L[1]) <= tol
        and infiltration_x0 - tol <= x[0] <= infiltration_x1 + tol
    )


def getTopConcDirichletBCs(x, tag):
    if on_top_infiltration_strip(x, tag):
        return lambda x, t: cin 


dirichletConditions = {0: getTopConcDirichletBCs}


class ConstantIC:
    def __init__(self, cval=0.0):
        self.cval = cval

    def uOfXT(self, x, t):
        return self.cval

    def uOfX(self, x):
        return self.cval


initialConditions = {0: ConstantIC(0.0)}


def getZeroMassDiffusiveFluxBCs(x, tag):
    if tag == boundaryTags["top"]:
        if infiltration_x0 <= x[0] <= infiltration_x1:
            return None
        return lambda x, t: 0.0
    return lambda x, t: 0.0


def getZeroMassAdvectiveFluxBCs(x, tag):
    if tag == boundaryTags["top"]:
        if infiltration_x0 <= x[0] <= infiltration_x1:
            return None
        return lambda x, t: 0.0
    if tag == boundaryTags["bottom"]:
        return lambda x, t: 0.0
    if tag in (boundaryTags["left"], boundaryTags["right"]):
        # Leave the side boundaries as natural/outflow for transport.
        return None


fluxBoundaryConditions = {0: "setFlow"}
advectiveFluxBoundaryConditions = {0: getZeroMassAdvectiveFluxBCs}
diffusiveFluxBoundaryConditions = {0: {0: getZeroMassDiffusiveFluxBCs}}
