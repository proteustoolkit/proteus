from math import *
from proteus import *
from proteus.default_p import *
try:
    from .cylinder import *
except:
    from cylinder import *
from proteus.kernels import Pres

name = "pressure"

LevelModelType = Pres.LevelModel
coefficients=Pres.Coefficients(modelIndex=PRESSURE_model,
                               fluidModelIndex=V_model,
                               pressureIncrementModelIndex=PINC_model,
                               useRotationalForm=False)


def getDBC_p(x,flag):
    if flag in [boundaryTags['right']]:
        return lambda x,t: 0.0

def getFlux(x,flag):
    return None

class getIBC_p(object):
    def __init__(self,waterLevel):
        self.waterLevel=waterLevel
    def uOfXT(self,x,t):
        return 0.0

initialConditions = {0:getIBC_p(waterLine_z)}

dirichletConditions = {0:getDBC_p } # pressure bc are explicitly set
advectiveFluxBoundaryConditions = {0:getFlux}
