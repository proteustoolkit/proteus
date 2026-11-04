from math import *
from proteus import *
from proteus.default_p import *
try:
    from .risingBubble import *
except:
    from risingBubble import *

name = "pressureincrement"

from proteus.mprans import PresInc
LevelModelType = PresInc.LevelModel
coefficients=PresInc.Coefficients(rho_f_min = (1.0-1.0e-8)*rho_1,
                                  rho_s_min = (1.0-1.0e-8)*rho_s,
                                  nd = nd,
                                  modelIndex=PINC_model,
                                  fluidModelIndex=V_model,
                                  # the increment is determined up to a constant only in a closed tank: with
                                  # the top open its Dirichlet condition fixes it, and a pin would change it
                                  fixNullSpace=not openTop,
                                  nullSpace='NoNullSpace')#'ConstantNullSpace')

#pressure increment should be zero on any pressure dirichlet boundaries
def getDBC_phi(x,flag):
    if flag == boundaryTags['top'] and openTop:
        return lambda x,t: 0.0

#the advectiveFlux should be zero on any no-flow  boundaries
def getAdvectiveFlux_qt(x,flag):
    if not (flag == boundaryTags['top'] and openTop):
        return lambda x,t: 0.0

def getDiffusiveFlux_phi(x,flag):
    if not (flag == boundaryTags['top'] and openTop):
        return lambda x,t: 0.0

class getIBC_phi(object):
    def __init__(self):
        pass
    def uOfXT(self,x,t):
        return 0.0
        
initialConditions = {0:getIBC_phi()}
dirichletConditions = {0:getDBC_phi }
advectiveFluxBoundaryConditions = {0:getAdvectiveFlux_qt}
diffusiveFluxBoundaryConditions = {0:{0:getDiffusiveFlux_phi}}
