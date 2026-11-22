from math import *
from proteus import *
from proteus.default_p import *
try:
    from .risingBubble import *
except:
    from risingBubble import *
from proteus.kernels import Pres

name = "pressure"
LevelModelType = Pres.LevelModel
coefficients=Pres.Coefficients(modelIndex=PRESSURE_model,
                               fluidModelIndex=V_model,
                               pressureIncrementModelIndex=PINC_model,
                               useRotationalForm=False)

def getDBC_p(x,flag):
    if flag == boundaryTags['top'] and openTop:
        return lambda x,t: 0.0

def getFlux(x,flag):
    if not(flag == boundaryTags['top'] and openTop):
        return lambda x,t: 0.0

class getIBC_p(object):
    def __init__(self):
        pass
    def uOfXT(self,x,t):
        return -(L[1] - x[1])*rho_0*g[1]

initialConditions = {0:getIBC_p()}
dirichletConditions = {0:getDBC_p} # pressure bc are explicitly set
advectiveFluxBoundaryConditions = {0:getFlux}
