from proteus import *
from proteus.default_p import *
from proteus.ctransportCoefficients import smoothedHeaviside
from math import *
from .multiphase import *
from proteus.kernels import CLSVOF

LevelModelType = CLSVOF.LevelModel
coefficients = CLSVOF.Coefficients(V_model=0,
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
dirichletConditions = {0: lambda x, flag: domain.bc[flag].vof_dirichlet.init_cython()}
advectiveFluxBoundaryConditions = {0: lambda x, flag: domain.bc[flag].vof_advective.init_cython()}
diffusiveFluxBoundaryConditions = {0:{}}

