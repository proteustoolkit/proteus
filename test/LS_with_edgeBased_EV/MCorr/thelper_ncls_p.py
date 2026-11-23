from proteus import *
from proteus.default_p import *
from proteus.ctransportCoefficients import smoothedHeaviside
from math import *
from .thelper_cons_ls import *
from proteus.kernels import NCLS
#import Profiling

LevelModelType = NCLS.LevelModel
logEvent = Profiling.logEvent
name=soname+"_ls"

nd=2

class MyCoefficients(NCLS.Coefficients):
    def attachModels(self,modelList):
        self.model = modelList[0]
        self.q_v = np.zeros(self.model.q[('dH',0,0)].shape,'d')
        self.ebqe_v = np.zeros(self.model.ebqe[('dH',0,0)].shape,'d')
        self.rdModel = self.model
        self.ebqe_rd_u = self.rdModel.ebqe[('u',0)]
        # Define a 'velocity' field to be read by VOF
        self.model.q[('velocity',0)]=self.q_v
        self.model.ebqe[('velocity',0)]=self.ebqe_v
        self.ghost_penalty_constant=0.0
        self.phi_s = numpy.ones(self.model.mesh.nodeArray.shape[0], 'd')*1e10#
        self.useExact=False
        self.ebqe_phi_s        = numpy.ones((self.model.ebqe['x'].shape[0],self.model.ebqe['x'].shape[1]),'d') * 1e10
        self.q_phi_solid = numpy.ones(self.model.q[('u', 0)].shape, 'd')
coefficients = MyCoefficients(
    checkMass=checkMass,
    RD_model=RD_model,
    epsFact=epsFactHeaviside, 
    # Choice of numerical method #
    STABILIZATION_TYPE=ct.STABILIZATION_TYPE_ncls,
    COUPEZ=ct.COUPEZ,
    DO_REDISTANCING=ct.DO_REDISTANCING,
    SATURATED_LEVEL_SET=ct.SATURATED_LEVEL_SET,
    ENTROPY_TYPE=ct.ENTROPY_TYPE_ncls, 
    # Coupez and re-distancing parameters #
    epsCoupez=epsCoupez*he, 
    epsFactRedistancing=epsFactRedistance, #for signed function
    redistancing_tolerance=redist_tolerance,
    lambda_coupez=lambda_coupez,
    # Entropy viscosity parameter #
    cE=ct.cE_ncls)

coefficients.variableNames=['u']

##################
# VELOCITY FIELD #
##################
def velx(X,t):
    x=X[0]
    y=X[1]
    #    return -2*pi*(X[1]-0.5)
    return -2*np.sin(pi*x)**2*np.sin(pi*y)*np.cos(pi*y)*np.sin(2*pi*t/ct.T)

def vely(X,t):
    x=X[0]
    y=X[1]
    #    return 2*pi*(X[0]-0.5)
    return 2*np.sin(pi*y)**2*np.sin(pi*x)*np.cos(pi*x)*np.sin(2*pi*t/ct.T)
    
velocityField={0:velx, 
               1:vely}

#####################
# INITIAL CONDITION #
#####################
class init_cond(object):
    def __init__(self,L,scaling=0.25):
        self.radius=0.15
        self.xc=0.5
        self.yc=0.75
        self.scaling=scaling
    def uOfXT(self,x,t):
        import numpy as np
        beta = epsCoupez*he
        r = math.sqrt((x[0]-self.xc)**2 + (x[1]-self.yc)**2)
        if ct.STABILIZATION_TYPE_ncls==0 or ct.level_set_function==0: #use dist function
            return self.radius - r
        else: #use saturated distance function
            return beta*(2*smoothedHeaviside(beta, self.radius - r)-1)    
    
analyticalSolution = {0:init_cond(L,scaling=1.0)}
initialConditions  = {0:init_cond(L,scaling=1.0)}

#######################
# BOUNDARY CONDITIONS #
#######################
def getDBC(x,flag):
    pass
dirichletConditions = {0:getDBC}

#Periodic Boundary Conditions
eps=1.0e-8
def getPDBC(x,flag):
    if x[0] <= eps or x[0] >= (L[0]-eps):
        return numpy.array([0.0,x[1],0.0])
#periodicDirichletConditions = {0:getPDBC}
    
def zeroInflow(x):
    return lambda x,t: 0.0
fluxBoundaryConditions = {0:'outFlow'}

def zeroadv(x):
    return lambda x,t: 0.0

advectiveFluxBoundaryConditions =  {0:zeroadv}
diffusiveFluxBoundaryConditions = {0:{}}
