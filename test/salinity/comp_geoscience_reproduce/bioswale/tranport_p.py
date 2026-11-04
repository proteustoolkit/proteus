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
from domain_rg import *
import numpy as np

################################## Inputs ########################################

physicalDiffusion = 0.0 #1.0e-8 # m2/s, molecular diffusion coefficient
refinement = 0

# SHOCK CAPTURING PARAMETERS #
shockCapturingFactor_tadr=0.2
lag_shockCapturing_tadr=True

# General parameters #
parallel = False # if True use PETSc solvers
linearSmoother = None
checkMass = False


class MyCoefficients(TADR.Coefficients):
    pass


LevelModelType = TADR.LevelModel
logEvent = Profiling.logEvent
#name=soname
a0= 18.8571e-4 #e-6
def a(x):
    return np.array([[a0,0.0],[0.0,a0]])
aOfX = {0:a}
alpha_L= 0.1
alpha_T= 0.1*alpha_L
Dm= 18.86e-4 #m2/s, dispersion coefficient
coefficients = MyCoefficients(
    aOfX,
    alpha_L=alpha_L,
    alpha_T=alpha_T,
    Dm=Dm,
    V_model = 0,
    specified_velocity=False,
    checkMass=checkMass,
    FCT=True,
    LUMPED_MASS_MATRIX=True,
    forceStrongConditions=True,
    STABILIZATION_TYPE=2,
    diagonal_conductivity= True,
    ENTROPY_TYPE='LOG',
    cE=0.1, cK=1.0, physicalDiffusion=0.0,
    rho_f=1.0,
    rho_s=1.025) #ct.physicalDiffusion)
coefficients.variableNames=['u']

#####################
# INITIAL CONDITION #
#####################
###########################
# Defining Initial Conditions Functions
# Initially will start at all freshwater
# and then go to a steady state
###########################


cin= 1.00
def getRGConcDirichletBCs(x,tag):
    if x[1] == L[1]:
        if (x[0] >= L[0]/3.0 and
            x[0] <= 2.0*L[0]/3.0):
            return lambda x,t: cin
    else:
        return None

dirichletConditions = {0:getRGConcDirichletBCs}

class constantIC:
    def __init__(self,cval=0.0):
        self.cval = cval
    def uOfXT(self,x,t):
        bc= getRGConcDirichletBCs(x,0)
        if bc != None:
            return bc(x,t)
        else:
            return self.cval
    def uOfX(self,x):
        bc= getRGConcDirichletBCs(x,0)
        if bc != None:
            return bc(x,t)
        else:
            return self.cval

initialConditions ={0:constantIC(0.0)}

def getZeroDiffusiveFlux(x, flag):
    if flag == boundaryTags['top'] and (L[0]/3.0 <= x[0] <= 2.0*L[0]/3.0):
        return None  # Dirichlet already specified on the ponded strip
    return lambda x, t: 0.0
fluxBoundaryConditions = {0:'setFlow'}

advectiveFluxBoundaryConditions = {0: getZeroDiffusiveFlux}  # use 'setFlow' behavior
diffusiveFluxBoundaryConditions  = {0: {0: getZeroDiffusiveFlux}}
