from math import *
from proteus import *
from proteus.default_p import *
try:
    from .multiphase import *
except:
    from multiphase import *
from proteus.kernels import PresInit

name = "pressureInitial"
#LevelModelType = PresInit.LevelModel
coefficients=PresInit.Coefficients(nd=nd,
                                   modelIndex=PINIT_model,
                                   fluidModelIndex=V_model,
                                   pressureModelIndex=PRESSURE_model)

#pressure increment should be zero on any pressure dirichlet boundaries
def getDBC_pInit(x,flag):
    if flag == boundaryTags['top']:
        return lambda x,t: 0.0

#the advectiveFlux should be zero on any no-flow  boundaries
def getAdvectiveFlux_pInit(x,flag):
    if flag != boundaryTags['top']:
        return lambda x,t: 0.0

def getDiffusiveFlux_pInit(x,flag):
    if flag != boundaryTags['top']:
        return lambda x,t: 0.0

class getIBC_pInit(object):
    def __init__(self):
        pass
    def uOfXT(self,x,t):
        return 0.0

initialConditions = {0:getIBC_pInit()}
dirichletConditions = {0:getDBC_pInit}
advectiveFluxBoundaryConditions = {0:getAdvectiveFlux_pInit}
diffusiveFluxBoundaryConditions = {0:{0:getDiffusiveFlux_pInit}}
