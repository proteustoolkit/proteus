from __future__ import absolute_import

import numpy as np
from proteus import *
from proteus.default_p import *
from proteus import Profiling
from proteus.mprans import TADR

try:
    from .domain_rg import *
except ImportError:
    from domain_rg import *

physicalDiffusion = 0.0
refinement = 0
shockCapturingFactor_tadr = 0.0
lag_shockCapturing_tadr = True

parallel = True
linearSmoother = None
checkMass = False


class MyCoefficients(TADR.Coefficients):
    pass


LevelModelType = TADR.LevelModel
logEvent = Profiling.logEvent
a0 = 18.8571e-3


def a(x):
    return np.array([[a0, 0.0], [0.0, a0]])


aOfX = {0: a}
alpha_L = 0.1
alpha_T = 0.01
Dm = 18.8571e-3

coefficients = MyCoefficients(
    aOfX,
    alpha_L=alpha_L,
    alpha_T=alpha_T,
    Dm=Dm,
    porosity=0.41,
    V_model=0,
    specified_velocity=False,
    checkMass=checkMass,
    FCT=True,
    LUMPED_MASS_MATRIX=True,
    STABILIZATION_TYPE=2,
    diagonal_conductivity=True,
    ENTROPY_TYPE="POWER",
    cE=0.1,
    cK=1.0,
    physicalDiffusion=0.0,
    rho_f=1.0,
    rho_s=1.025,
)
coefficients.variableNames = ["u"]

cin = 0.7


def getLeftConcDirichletBCs(x, tag):
    if tag == boundaryTags["leftTop"]:
        return lambda x, t: cin


dirichletConditions = {0: getLeftConcDirichletBCs}


class ConstantIC:
    def __init__(self, cval=0.0):
        self.cval = cval

    def uOfXT(self, x, t):
        return self.cval

    def uOfX(self, x):
        return self.cval


initialConditions = {0: ConstantIC(0.0)}


def getZeroMassDiffusiveFluxBCs(x, tag):
    if tag == boundaryTags["leftTop"]:
        return None
    if tag == boundaryTags["right"]:
        return None
    return lambda x, t: 0.0


def getZeroMassAdvectiveFluxBCs(x, tag):
    if tag == boundaryTags["leftTop"]:
        return None
    if tag == boundaryTags["right"]:
        return None
    return lambda x, t: 0.0


fluxBoundaryConditions = {0: "setFlow"}
advectiveFluxBoundaryConditions = {0: getZeroMassAdvectiveFluxBCs}
diffusiveFluxBoundaryConditions = {0: {0: getZeroMassDiffusiveFluxBCs}}
