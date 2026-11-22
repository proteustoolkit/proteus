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

#T =4800.0 #100.0 #12000 # 6.0e2*6..,1.0e4 #time scale, s
#nDTout =10 #0  # output timesteps to use
ct=Context.Options([
    # General parameters #
    ("physicalDiffusion", 0.0, "isotropic diffusion coefficient"),
    #("problem",0,"0: 1D problem with periodic BCs. 1: 2D rotation of zalesak disk"),
    #("nd",2,"space dimension"),
    #("T",100,"Final time"),
    #("nDTout",100,"Number of time steps to archive"),
    ("refinement",0,"Level of refinement"),
    ("unstructured",False,"Use unstructured mesh. Set to false for periodic BCs"),
    # Choice of numerical method #
    ("STABILIZATION_TYPE","Galerkin","Galerkin, VMS, TaylorGalerkinEV, EntropyViscosity, SmoothnessIndicator, Kuzmin, "),
    ("LUMPED_MASS_MATRIX",False,"Flag to lumped the mass matrix"),
    ("ENTROPY_TYPE","POWER","POWER, LOG"),
    ("FCT",True,"Use Flux Corrected Transport"),
    # Numerical parameters #
    ("cE",0.1,"Entropy viscosity constant"),
    ("cK",1.0,"Artificial compression constant"),
    ("cfl",0.5,"Target cfl"),
    ("SSPOrder",2,"SSP method of order 1, 2 or 3")
],mutable=True)


# SHOCK CAPTURING PARAMETERS #
shockCapturingFactor_tadr=0.2
lag_shockCapturing_tadr=True

# number of space dimensions #
#nd=ct.nd

# General parameters #
parallel = False # if True use PETSc solvers
linearSmoother = None
checkMass = False

# Finite element sapce #
pDegree_tadr=1
useBernstein=False
useHex=False

# quadrature order #
tadr_quad_order = 2*pDegree_tadr+1
#from gw_cl_domain import *

class MyCoefficients(TADR.Coefficients):
    def attachModels(self, modelList):
        """
        Attach the Richards model to TADR and pass the velocity field (grad(u)).
        """
        # Attach Richards model
        self.model = modelList[0]  # Richards model
        self.rdModel = self.model  # Alias for convenience
            # Log modelList for inspection
        logEvent(f"Model list contains {len(modelList)} models:")
        for i, model in enumerate(modelList):
            logEvent(f"Model {i}: {model.name}, Type: {type(model)}")

        # Reference velocity fields directly from Richards model
        if ('grad(u_v)', 0) in self.model.q:
            self.q_v = self.model.q[('grad(u_v)', 0)]  # Reference gradient of u (velocity)
            #self.q_v *= 1.241e-9
        else:
            raise KeyError("Richards model q[('grad(u_v)', 0)] not found!")

        if ('grad(u)', 0) in self.model.ebqe:
            self.ebqe_v = self.model.ebqe[('grad(u)', 0)]  # Reference boundary gradient of u (velocity)
        else:
            raise KeyError("Richards model ebqe[('grad(u)', 0)] not found!")

        # Ensure quadrature volumes exist
        if 'dV' not in self.model.q:
            self.model.q['dV'] = np.ones(self.model.q[('u', 0)].shape, 'd')

        if 'dV_last' not in self.model.q:
            self.model.q['dV_last'] = np.zeros_like(self.model.q['dV'])

        # Log for debugging
        logEvent(f"Richards velocity (q_v): mean={self.q_v.mean()}, min={self.q_v.min()}, max={self.q_v.max()}")
        logEvent(f"Richards boundary velocity (ebqe_v): mean={self.ebqe_v.mean()}, min={self.ebqe_v.min()}, max={self.ebqe_v.max()}")

LevelModelType = TADR.LevelModel
logEvent = Profiling.logEvent
#name=soname
a0= 18.8571e-6 #e-6 
def a(x):
    return np.array([[a0,0.0],[0.0,a0]])
aOfX = {0:a}
alpha_L= 0.1
alpha_T= 0.1*alpha_L
Dm= 18.86e-6
coefficients = MyCoefficients(
    alpha_L=alpha_L,
    alpha_T=alpha_T,
    Dm=Dm,
    V_model=0,  
    checkMass=checkMass,
    FCT=ct.FCT,
    LUMPED_MASS_MATRIX=ct.LUMPED_MASS_MATRIX, 
    STABILIZATION_TYPE=ct.STABILIZATION_TYPE,
    diagonal_conductivity= True, 
    ENTROPY_TYPE=ct.ENTROPY_TYPE, 
    cE=ct.cE, cK=ct.cK, physicalDiffusion=ct.physicalDiffusion) 
coefficients.variableNames=['u']

# Density Relation
rho_fw =  1000.0 # density of freshwater
rho_sw = 1025.0 # 1026.0 density of saltwater
mf_fw = 0.0 # saline mass fraction, freshwater
mf_sw = 1.0# .0357 # saline mass fraction, saltwater
epsilon = (rho_sw - rho_fw)/rho_fw 
drho_dmf =epsilon*rho_fw #700. #epsilon*rho_fw # density change with mass fraction

# Dynamic Viscosity Relation
nu = 1.0e-3

#####################
# INITIAL CONDITION #
#####################
###########################
# Defining Initial Conditions Functions
# Initially will start at all freshwater
# and then go to a steady state
###########################


class constantIC:
    def __init__(self,cval=0.0):
        self.cval = cval
    def uOfXT(self,x,t):
        return self.cval
    def uOfX(self,x):
        return self.cval
#        return self.cval


def getHenryConcDirichletBCs(x,tag):
    if x[0]<1.e-8:     
        return lambda x,t: mf_fw 
    elif abs(x[0]-L[0])<1.e-8:
        return lambda x,t: mf_sw

# Henry Problem Mass Flux Boundary Conditions

def getzeroMassDiffusiveFluxBCs(x,tag):
    if ((x[0] > 1.e-8 and abs(x[0] -L[0])> 1.e-8) and 
        (x[1] < 1.0e-8 or  abs(x[1]- L[1]) < 1.0e-8) ):
        return lambda x,t: 0.0
    else: 
        pass

def getzeroBCs(x,tag):
    return lambda x,t: 0.0
   
def getzeroMassAdvectiveFluxBCs(x,tag):
    if ((x[0] > 1.0e-8 and x[0] < L[0]-1.0e-8) and 
       ( x[1] < 1.0e-8 or x[1] > L[1]-1.0e-8)):
        return lambda x,t: 0.0
    

initialConditions ={0:constantIC(mf_fw)} 
dirichletConditions ={0:getHenryConcDirichletBCs} 
fluxBoundaryConditions = {0:'setFlow'}
advectiveFluxBoundaryConditions ={0:getzeroMassDiffusiveFluxBCs}
diffusiveFluxBoundaryConditions ={0:{0:getzeroMassDiffusiveFluxBCs}}#{0:{0:getzeroBCs}}#{0:{0:getzeroMassDiffusiveFluxBCs}}

