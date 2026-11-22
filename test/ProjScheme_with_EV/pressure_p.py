from math import *
from proteus import *
from proteus.default_p import *
from .NS_convergence import *
from proteus.kernels import Pres

name = "pressure"
LevelModelType = Pres.LevelModel
coefficients=Pres.Coefficients(modelIndex=PRESSURE_model,
                               fluidModelIndex=V_model,
                               pressureIncrementModelIndex=PINC_model,
                               useRotationalForm=False)

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
