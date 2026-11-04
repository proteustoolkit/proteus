from __future__ import absolute_import
from builtins import object
import numpy as np
from proteus import *
from proteus.default_p import *
from proteus.ctransportCoefficients import smoothedHeaviside

from proteus import Domain
from proteus import Norms
from proteus import Profiling
from proteus import Context
from proteus.mprans import TADR

from proteus.Profiling import logEvent

from math import *
try:
    from .domain_henry import *
except:
    from domain_henry import *

import numpy as np

################################## Inputs ########################################

physicalDiffusion = 0.0 #1.0e-8 # m2/s, molecular diffusion coefficient
refinement = 0

# SHOCK CAPTURING PARAMETERS #
shockCapturingFactor_tadr=0.0 #0.2
lag_shockCapturing_tadr= False #True



# General parameters #
parallel = True # if True use PETSc solvers
linearSmoother = None
checkMass = False


class MyCoefficients(TADR.Coefficients):
    pass


LevelModelType = TADR.LevelModel
logEvent = Profiling.logEvent
#name=soname
a0= 6.6e-6 #18.8571e-6
def a(x):
    return np.array([[a0,0.0],[0.0,a0]])
aOfX = {0:a}
alpha_L= 0.0
alpha_T= 0.0
Dm=  18.86e-6  # Matches Henry reference b = Dm/|q_in| = 0.1
coefficients = MyCoefficients(
    aOfX,
    alpha_L=alpha_L,
    alpha_T=alpha_T,
    Dm=Dm,
    porosity=.35,
    V_model = 0,
    specified_velocity=False,
    checkMass=checkMass,
    FCT=False,
    LUMPED_MASS_MATRIX=True,
    STABILIZATION_TYPE=5,  # ImplicitEV (the paper figure's deck: 2); 4=Kuzmin FCT (DMP-preserving at high Pe)
    diagonal_conductivity= True,
    ENTROPY_TYPE='POWER',
    cE=0.1, cK=1.0, physicalDiffusion=0.0,
    rho_f=1.0, rho_s=1.025) #ct.physicalDiffusion,)
coefficients.variableNames=['u']

#####################
# INITIAL CONDITION #
#####################
###########################
# Defining Initial Conditions Functions
# Initially will start at all freshwater
# and then go to a steady state
###########################


#cin= 10.00
mf_fw = 0.0 # saline mass fraction, freshwater
mf_sw = 1.0  # saline mass fraction, saltwater
boundaryTags = domain.boundaryTags


def getHenryConcDirichletBCs(x, tag):
    # Pure Dirichlet: c = 0 on the freshwater inflow boundary,
    # c = 1 on the entire seaward (right) boundary.  Matches Henry's
    # analytical formulation, which assumes c = 1 everywhere on x = L.
    if tag == boundaryTags['left']:
        return lambda x, t: mf_fw
    elif tag == boundaryTags['right']:
        return lambda x, t: mf_sw

dirichletConditions = {0:getHenryConcDirichletBCs}

class constantIC:
    def __init__(self,cval=0.0):
        self.cval = cval
    def uOfXT(self,x,t):
        return self.cval
    def uOfX(self,x):
        return self.cval

initialConditions ={0:constantIC(0.0)}
# Henry Problem Mass Flux Boundary Conditions

def getzeroMassDiffusiveFluxBCs(x,tag):
    if tag in [boundaryTags['top'], boundaryTags['bottom']]:
        return lambda x,t: 0.0
    else:
        pass

def getzeroBCs(x,tag):
    return lambda x,t: 0.0

def getzeroMassAdvectiveFluxBCs(x,tag):
    if tag in [boundaryTags['top'], boundaryTags['bottom']]:
        return lambda x,t: 0.0

fluxBoundaryConditions = {0:'setFlow'}
advectiveFluxBoundaryConditions ={0:getzeroMassAdvectiveFluxBCs}
diffusiveFluxBoundaryConditions ={0:{0:getzeroMassDiffusiveFluxBCs}}#{0:{0:getzeroBCs}}#{0:{0:getzeroMassDiffusiveFluxBCs}}
