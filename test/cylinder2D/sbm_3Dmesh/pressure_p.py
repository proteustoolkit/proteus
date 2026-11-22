from math import *
from proteus import *
from proteus.default_p import *
from cylinder import *
from proteus.kernels import Pres

name = "pressure"

LevelModelType=Pres.LevelModel
coefficients=Pres.Coefficients(modelIndex=PRESSURE_model,
                               fluidModelIndex=V_model,
                               pressureIncrementModelIndex=PINC_model,
                               useRotationalForm=False)


def getDBC_p(x,flag):
    if flag in [boundaryTags['right']]:# without boundaryTags['front'],boundaryTags['back'] becuase of slip-bc 
        return lambda x,t: 0.0

def getFlux(x,flag):
    return None

class getIBC_p(object):
    def uOfXT(self,x,t):
       return 0.0

initialConditions = {0:getIBC_p()}

dirichletConditions = {0:getDBC_p }
advectiveFluxBoundaryConditions = {0:getFlux}
