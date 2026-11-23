from proteus import *
from proteus.default_p import *
from proteus.ctransportCoefficients import smoothedHeaviside
from math import *
try:
    from .multiphase import *
except:
    from multiphase import *
from proteus.kernels import CLSVOF

LevelModelType = CLSVOF.LevelModel
coefficients = CLSVOF.Coefficients(V_model=V_model,
                                   ME_model=CLSVOF_model,
                                   useMetrics=useMetrics,
                                   epsFactHeaviside=epsFactHeaviside_clsvof,
                                   epsFactDirac=epsFactHeaviside_clsvof,
                                   lambdaFact=lambdaFact_clsvof,
                                   outputQuantDOFs=True,
                                   computeMetrics=computeMetrics_clsvof)
coefficients.variableNames=['phi']
name="clsvof"

#####################
# INITIAL CONDITION #
#####################
class init_cond(object):
    def uOfXT(self,x,t):
        return signedDistance(x)
initialConditions  = {0:init_cond()}

#######################
# BOUNDARY CONDITIONS #
#######################
def getDBC_vof(x,flag):
    if flag == boundaryTags['top'] and openTop:
        return lambda x,t: 1.0
#
def getAFBC_vof(x,flag):
    if flag != boundaryTags['top'] or not openTop:
        return lambda x,t: 0.0

dirichletConditions = {0:getDBC_vof}
advectiveFluxBoundaryConditions = {0:getAFBC_vof}
diffusiveFluxBoundaryConditions = {0:{}}
