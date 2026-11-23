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
from proteus.kernels import TADR

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
shockCapturingFactor_tadr=0.2
lag_shockCapturing_tadr=True



# General parameters #
parallel = True # if True use PETSc solvers
linearSmoother = None
checkMass = False


class MyCoefficients(TADR.Coefficients):
    pass


LevelModelType = TADR.LevelModel
logEvent = Profiling.logEvent
#name=soname
a0= 18.8571e-6
def a(x):
    return np.array([[a0,0.0],[0.0,a0]])
aOfX = {0:a}
alpha_L= 0.1
alpha_T= 0.01
Dm= 18.8571e-6
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
    STABILIZATION_TYPE=2,
    diagonal_conductivity= True,
    ENTROPY_TYPE='LOG',
    cE=0.1, cK=1.0, physicalDiffusion=0.0,
    rho_f=1000.0, rho_s=1025.0) #ct.physicalDiffusion,)
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


def _is_right_boundary(x, tol=1.0e-8):
    return abs(x[0] - L[0]) <= tol


class HenryNumericalFlux(TADR.NumericalFlux):
    def setDirichletValues(self, ebqe):
        for ci in range(self.nc):
            self.ebqe[('u', ci)].flat[:] = ebqe[('u', ci)].flat[:]
            self.isDOFBoundary[ci][:] = 0

            for (ebNE, k), g, x in zip(list(self.DOFBoundaryConditionsDictList[ci].keys()),
                                       list(self.DOFBoundaryConditionsDictList[ci].values()),
                                       list(self.DOFBoundaryPointDictList[ci].values())):
                if _is_right_boundary(x):
                    if hasattr(self.vt.coefficients, 'ebqe_v'):
                        velocity = self.vt.coefficients.ebqe_v[ebNE, k, :self.vt.nSpace_global]
                        flow = float(velocity[0])
                    else:
                        flow = -1.0
                    if flow < 0.0:
                        self.ebqe[('u', ci)][ebNE, k] = mf_sw
                        self.isDOFBoundary[ci][ebNE, k] = 1
                        ebqe[('diffusiveFlux_bc_flag', 0, 0)][ebNE, k] = 0
                        ebqe[('diffusiveFlux_bc', 0, 0)][ebNE, k] = 0.0
                    else:
                        self.isDOFBoundary[ci][ebNE, k] = 0
                        ebqe[('diffusiveFlux_bc_flag', 0, 0)][ebNE, k] = 1
                        ebqe[('diffusiveFlux_bc', 0, 0)][ebNE, k] = 0.0
                else:
                    self.ebqe[('u', ci)][ebNE, k] = g(x, self.vt.timeIntegration.t)
                    self.isDOFBoundary[ci][ebNE, k] = 1

        for ci in range(self.nc):
            for bci in list(self.periodicBoundaryConditionsDictList[ci].values()):
                self.ebqe[('u', ci)][bci[0]] = ebqe[('u', ci)][bci[1]]
                self.ebqe[('u', ci)][bci[1]] = ebqe[('u', ci)][bci[0]]

        if self.vt.movingDomain:
            self.vt.coefficients.updateToMovingDomain(self.vt.timeIntegration.t, self.ebqe)

def getHenryConcDirichletBCs(x,tag):
    if tag == boundaryTags['left']:
        return lambda x,t: mf_fw
    elif tag == boundaryTags['right']:
        return lambda x,t: mf_sw

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
