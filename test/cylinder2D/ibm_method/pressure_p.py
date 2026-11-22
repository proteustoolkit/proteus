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
    if not(flag == boundaryTags['right']):
        return lambda x,t: 0.0

class getIBC_p(object):
    def __init__(self,waterLevel):
        self.waterLevel=waterLevel
    def uOfXT(self,x,t):
        return 0.0
#        if signedDistance(x) < 0:
#            return -(L[1] - self.waterLevel)*rho_1*g[1] - (self.waterLevel - x[1])*rho_0*g[1]
#        else:
#            return -(L[1] - x[1])*rho_1*g[1]

initialConditions = {0:getIBC_p(waterLine_z)}

dirichletConditions = {0:getDBC_p } # pressure bc are explicitly set
advectiveFluxBoundaryConditions = {0:getFlux}
