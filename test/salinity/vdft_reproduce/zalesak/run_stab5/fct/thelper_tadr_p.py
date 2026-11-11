from proteus import *
from proteus.default_p import *
from proteus.ctransportCoefficients import smoothedHeaviside
from math import *
try:
    from .thelper_tadr import *
except:
    from thelper_tadr import *

LevelModelType = TADR.LevelModel
logEvent = Profiling.logEvent
name=soname

nd=ct.nd

coefficients = MyCoefficients(
    alpha_L=0.0,
    alpha_T=0.0,
    Dm=0.0,
    checkMass=checkMass,
    FCT=ct.FCT,
    LUMPED_MASS_MATRIX=ct.LUMPED_MASS_MATRIX, 
    STABILIZATION_TYPE=ct.STABILIZATION_TYPE, 
    ENTROPY_TYPE=ct.ENTROPY_TYPE, 
    cE=ct.cE, cK=ct.cK, physicalDiffusion=ct.physicalDiffusion) 
coefficients.variableNames=['u']

##################
# VELOCITY FIELD #
##################
def velx(X,t):
    if ct.problem in [0,2]:
        return 1.0
    else:        
        return -2*pi*(X[1]-0.5)

def vely(X,t):
    if ct.problem in [0,2]:
        return 0.0
    else:
        return 2*pi*(X[0]-0.5)

velocityFieldAsFunction={0:velx,
                         1:vely}

#####################
# INITIAL CONDITION #
#####################
class init_cond(object):
    def __init__(self,L):
        self.radius = 0.15
        self.xc=0.5
        self.yc=0.75
        self.xc2=0.5
        self.yc2=0.25
        self.xc3=0.25
        self.yc3=0.5
    def uOfXT(self,x,t):
        if ct.problem==0:
            if (x[0] >= 0.3 and x[0] <= 0.7):
                return 1.0
            else:
                return 0.0
        elif ct.problem==1: 
            r  = math.sqrt((x[0]-self.xc )**2 + (x[1]-self.yc )**2)
            r2 = math.sqrt((x[0]-self.xc2)**2 + (x[1]-self.yc2)**2)
            r3 = math.sqrt((x[0]-self.xc3)**2 + (x[1]-self.yc3)**2)
            slit = x[0] > self.xc-0.025 and x[0] < self.xc+0.025 and x[1] < self.yc+0.1125
            if (r <= self.radius and slit == False):
                return 1.
            elif (r <= self.radius and slit == True):
                return 0.
            elif (r2 <= self.radius):
                return (1.0 - r2/self.radius)
            elif (r3 <= self.radius):
                return (1.0 + math.cos(math.pi*r3/self.radius))/4.0
            else:
                return 0.0
        elif ct.problem == 2:
            if ct.nd == 1:
                a = 0.15
                b = 1.0
                if fabs(x[0] - 0.5) <= a:
                    return b/a*math.sqrt(a**2 - (x[0] - 0.5)**2)
                else:
                    return 0.0

initialConditions  = {0:init_cond(L)}
analyticalSolution = {0:init_cond(L)}

#######################
# BOUNDARY CONDITIONS #
#######################
def getDBC(x,flag):
    if ct.problem in [0,2]:
        None
    else:
        return lambda x,t: 0.0
dirichletConditions = {0:getDBC}

#Periodic Boundary Conditions
eps=1.0e-8
def getPDBC(x,flag):
    if x[0] <= eps or x[0] >= (L[0]-eps):
        return numpy.array([0.0,x[1],0.0])
if ct.problem in [0,2]:
    periodicDirichletConditions = {0:getPDBC}

def zeroadv(x,flag):
    None
advectiveFluxBoundaryConditions =  {0:zeroadv}

fluxBoundaryConditions = {0:'outFlow'}
diffusiveFluxBoundaryConditions = {0:{}}
