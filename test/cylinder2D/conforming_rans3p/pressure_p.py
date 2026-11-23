from math import *
from proteus import *
from proteus.default_p import *
try:
    from .cylinder3p import *
except:
    from cylinder3p import *
    
from proteus.kernels import Pres

name = "pressure"

coefficients=Pres.Coefficients(modelIndex=PRESSURE_model,
                               fluidModelIndex=V_model,
                               pressureIncrementModelIndex=PINC_model,
                               useRotationalForm=False)


LevelModelType = Pres.LevelModel

def getDBC_p(x,flag):
    if flag in [boundaryTags['right']]:
        return lambda x,t: 0.0

def getFlux(x,flag):
    return None

class getIBC_p(object):
    def uOfXT(self,x,t):
       return 0.0

initialConditions = {0:getIBC_p()}

dirichletConditions = {0:getDBC_p } # pressure bc are explicitly set
advectiveFluxBoundaryConditions = {0:getFlux}
