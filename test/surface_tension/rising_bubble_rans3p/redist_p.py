from proteus import *
from proteus.default_p import *
from math import *
try:
    from .risingBubble import *
except:
    from risingBubble import *
from proteus.kernels import RDLS
"""
The redistancing equation in the sloshbox test problem.
"""

LevelModelType = RDLS.LevelModel
coefficients = RDLS.Coefficients(applyRedistancing=applyRedistancing,
                                 epsFact=epsFact_redistance,
                                 nModelId=LS_model,
                                 rdModelId=RD_model,
                                 useMetrics=useMetrics,
                                 ELLIPTIC_REDISTANCING=ELLIPTIC_REDISTANCING,
                                 alpha=alpha_REDISTANCING,
                                 copyList=rdls_copyList)

def getDBC_rd(x,flag):
    pass

dirichletConditions     = {0:getDBC_rd}
weakDirichletConditions = {0:RDLS.setZeroLSweakDirichletBCsSimple}

advectiveFluxBoundaryConditions =  {}
diffusiveFluxBoundaryConditions = {0:{}}

class PHI_IC(object):
    def uOfXT(self,x,t):
        return signedDistanceToBubble(x)

initialConditions  = {0:PHI_IC()}
