from math import *
from proteus import *
from proteus.default_p import *
from NS_hotstart import *
from proteus.kernels import Pres

name = "pressure"

coefficients=Pres.Coefficients(modelIndex=2,
                               fluidModelIndex=0,
                               pressureIncrementModelIndex=1,
                               useRotationalForm=False)
LevelModelType = Pres.LevelModel

def getDBC_p(x,flag):
    None

def getFlux(x,flag):
    None

class getIBC_p(object):
    def __init__(self):
        pass
    def uOfXT(self,x,t):
        return np.cos(x[0])*np.sin(x[1])

initialConditions = {0:getIBC_p()}
dirichletConditions = {0:getDBC_p} # pressure bc are explicitly set
advectiveFluxBoundaryConditions = {0:getFlux}
