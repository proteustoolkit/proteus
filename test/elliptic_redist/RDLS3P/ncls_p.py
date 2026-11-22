from proteus import *
from proteus.default_p import *
from math import *
from .vortex2D import *
from proteus.kernels import NCLS

LevelModelType = NCLS.LevelModel
logEvent = Profiling.logEvent
name=soname+"_ls"

nd=2

RD_model=1
coefficients = MyCoefficients(
    checkMass=False,
    RD_model=RD_model,
    epsFact=epsFactHeaviside)
coefficients.variableNames=['u']

##################
# VELOCITY FIELD #
##################
def velx(X,t):
    x=X[0]
    y=X[1]
    return 0.0
    #return -2*np.sin(pi*x)**2*np.sin(pi*y)*np.cos(pi*y)*np.sin(2*pi*t/8.0)

def vely(X,t):
    x=X[0]
    y=X[1]
    return 0.0
    #return 2*np.sin(pi*y)**2*np.sin(pi*x)*np.cos(pi*x)*np.sin(2*pi*t/8.0)

velocityField={0:velx, 
               1:vely}

#####################
# INITIAL CONDITION #
#####################
class init_cond(object):
    def __init__(self,L,scaling=0.75):
        self.radius=0.15
        self.xc=0.5
        self.yc=0.75
        self.scaling=scaling
    def uOfXT(self,x,t):
        import numpy as np
        return self.scaling*(self.radius - math.sqrt((x[0]-self.xc)**2 + (x[1]-self.yc)**2))

class zalesak_disk(object):
    def __init__(self,L,scaling=0.75):
        self.radius=0.15
        self.xc=0.5
        self.yc=0.75
        self.scaling=scaling
    def uOfXT(self,X,t):
        import numpy as np
        x = X[0]
        y = X[1]
        # distance to center of the disk
        r = math.sqrt((x-self.xc)**2 + (y-self.yc)**2)
        dist_circle = self.radius - r # distance to circle
        # coordinates to slot coorners
        xslot1 = self.xc-0.025 
        xslot2 = self.xc+0.025
        yslot1 = 0.75 - np.sqrt(self.radius**2-0.025**2)
        yslot2 = 0.85
        
        #distance to the boundary of the slot
        aux1 = np.abs(x-xslot1)
        aux2 = np.abs(x-xslot2)
        aux3 = np.abs(y-yslot2)
        aux4 = np.abs(y-yslot1)
        
        if (y > yslot1): #above disk
            if (xslot1 < x and x <= xslot2 and y <= yslot2): #inside slot
                dist = -np.min([aux1,aux2,aux3])
            else: #Not inside slot
                if x <= xslot1: #left of slot
                    if y <= yslot2: #down top boundary of slot 
                        dist = np.min([dist_circle,aux1])
                    else: #above top boundary of slot
                        dist = np.min([dist_circle,np.sqrt(aux1**2+aux3**2)])
                elif x >= xslot2: #right of slot
                    if y <= yslot2: #down top boundary of slot
                        dist = np.min([dist_circle,aux2])
                    else: #above top boundary of slot
                        dist = np.min([dist_circle,np.sqrt(aux2**2+aux3**2)])
                else: #x-coordiante is within slot
                    dist = np.min([dist_circle,aux3])
        else: #below disk
            if x > xslot1 and x < xslot2: #x-coordinate is within slot
                dist = -np.min([np.sqrt(aux1**2 + aux4**2), np.sqrt(aux2**2 + aux4**2)])
            else:
                dist = dist_circle
        return self.scaling*dist
    
analyticalSolution = {0:init_cond(L,scaling=1.0)}
initialConditions  = {0:init_cond(L,scaling=0.9)}

# BOUNDARY CONDITIONS #
def getDBC(x,flag):
    pass

def zeroInflow(x):
    return lambda x,t: 0.0

dirichletConditions = {0:getDBC}
fluxBoundaryConditions = {0:'outFlow'}

def zeroadv(x):
    return lambda x,t: 0.0
advectiveFluxBoundaryConditions =  {}
diffusiveFluxBoundaryConditions = {0:{}}
