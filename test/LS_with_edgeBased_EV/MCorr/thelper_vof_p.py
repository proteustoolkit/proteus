from proteus import *
from proteus.default_p import *
from proteus.ctransportCoefficients import smoothedHeaviside
from .thelper_cons_ls import *
from proteus.kernels import VOF
name=soname+"_vof"

"""
The non-conservative level set description of a bubble in a two-phase flow
"""

LevelModelType = VOF.LevelModel
coefficients = VOF.Coefficients(LS_model=LS_model, #0
                                V_model=V_model, #0
                                RD_model=RD_model,
                                ME_model=VOF_model,
                                checkMass=checkMass,
                                epsFact=epsFact_vof,
                                # edge based parameters
                                STABILIZATION_TYPE=ct.STABILIZATION_TYPE_vof,
                                ENTROPY_TYPE=ct.ENTROPY_TYPE_vof,
                                FCT=ct.FCT,
                                cE=ct.cE_vof,
                                cK=ct.cK)

#####################
# INITIAL CONDITION #
#####################
class init_cond(object):
    def __init__(self,L):
        self.radius = 0.15
        self.xc=0.5
        self.yc=0.75
    def uOfXT(self,x,t):
        r = math.sqrt((x[0]-self.xc)**2 + (x[1]-self.yc)**2)
        return smoothedHeaviside(epsFactHeaviside*he,self.radius - r)
        
initialConditions  = {0:init_cond(L)}
analyticalSolution = {0:init_cond(L)}

#######################
# BOUNDARY CONDITIONS #
#######################
def getDBC(x,flag):
    None
dirichletConditions = {0:getDBC}

#Periodic Boundary Conditions
eps=1.0e-8
def getPDBC(x,flag):
    if x[0] <= eps or x[0] >= (L[0]-eps):
        return numpy.array([0.0,x[1],0.0])
#periodicDirichletConditions = {0:getPDBC}

def zeroadv(x,flag):
    return lambda x,t: 0.
advectiveFluxBoundaryConditions =  {0:zeroadv}

fluxBoundaryConditions = {0:'outFlow'}
diffusiveFluxBoundaryConditions = {0:{}}
