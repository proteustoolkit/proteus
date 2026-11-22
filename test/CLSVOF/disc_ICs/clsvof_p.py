from proteus import *
from proteus.default_p import *
from proteus.ctransportCoefficients import smoothedHeaviside
from proteus import Context
from proteus.kernels import CLSVOF
from math import *
try:
    from .clsvof import *
except:
    from clsvof import *

LevelModelType = CLSVOF.LevelModel
logEvent = Profiling.logEvent
name=soname

nd=2

epsFactHeaviside=1.5
coefficients = MyCoefficients(useMetrics=useMetrics,
                              doSpinUpStep=doSpinUpStep,
                              epsFactHeaviside=epsFactHeaviside,
                              epsFactDirac=epsFactHeaviside,
                              lambdaFact=lambdaFact,
                              outputQuantDOFs=True,
                              computeMetrics=computeMetrics,
                              disc_ICs=True)
coefficients.variableNames=['u']

##################
# VELOCITY FIELD #
##################
def velx(X,t):
    x=X[0]
    y=X[1]
    return 0*x

def vely(X,t):
    x=X[0]
    y=X[1]
    return 0*x

velocityFieldAsFunction={0:velx,
                         1:vely}
        
##################
# EXACT SOLUTION #
##################
def zalesak_disk_per_point(X,t):
    xc = 0.5
    yc = 0.75
    radius=0.15
    x=X[0]
    y=X[1]
    # distance to center of the disk
    r = np.sqrt((x-xc)**2 + (y-yc)**2)
    dist = radius - r # distance to circle
    # coordinates to slot coorners
    xslot1 = xc-0.025
    xslot2 = xc+0.025
    yslot1 = 0.75 - np.sqrt(radius**2-0.025**2)
    yslot2 = 0.85

    #distance to the boundary of the slot
    aux1 = np.abs(x-xslot1)
    aux2 = np.abs(x-xslot2)
    aux3 = np.abs(y-yslot2)
    aux4 = np.abs(y-yslot1)

    if (y > yslot1): #above disk
        if (xslot1 < x and x <= xslot2 and y <= yslot2): #inside slot
            return -np.min([aux1,aux2,aux3])
        else: #Not inside slot
            if x <= xslot1: #left of slot
                if y <= yslot2: #down top boundary of slot
                    return np.min([dist,aux1])
                else: #above top boundary of slot
                    return np.min([dist,np.sqrt(aux1**2+aux3**2)])
            elif x >= xslot2: #right of slot
                if y <= yslot2: #down top boundary of slot
                    return np.min([dist,aux2])
                else: #above top boundary of slot
                    return np.min([dist,np.sqrt(aux2**2+aux3**2)])
            else: #x-coordiante is within slot
                return np.min([dist,aux3])
    else: #below disk
        if x > xslot1 and x < xslot2: #x-coordinate is within slot
            return  -np.min([np.sqrt(aux1**2 + aux4**2), np.sqrt(aux2**2 + aux4**2)])
        else:
            return dist

def zalesak_disk(X,t):
    radius = 0.15
    xc = 0.5
    yc = 0.75
    if type(X) != dict:
        return zalesak_disk_per_point(X,0)
    else:
        x = X[0]
        y = X[1]
        z=np.zeros(x.shape,'d')
        for i in range(len(x)):
            j=0
            for xq,yq in zip(x[i],y[i]):
                XX = {0:xq,1:yq}
                z[i,j] = zalesak_disk_per_point(XX,t)
                j = j+1
        return z

class exact_solution_zalesak():
    def uOfXT(self,X,t):
        return zalesak_disk(X,0)
    
class exact_solution_circle():
    def uOfXT(self,X,t):
        xc = 0.5
        yc = 0.5
        radius=0.25
        x=X[0]
        y=X[1]
        # distance to center of the disk
        r = np.sqrt((x-xc)**2 + (y-yc)**2)
        dist = radius - r # distance to circle
        return dist

if ct.test_case==1:
    exact_solution = exact_solution_circle
else:
    exact_solution = exact_solution_zalesak
analyticalSolution={0:exact_solution()}

#####################
# INITIAL CONDITION #
#####################
class init_cond(object):
    def __init__(self):
        pass
    def uOfXT(self,x,t):
        dist = exact_solution().uOfXT(x,0)
        if dist >= 0:
            return 0. if dist==0 else 1
        else:
            return -1
initialConditions  = {0:init_cond()}

#######################
# BOUNDARY CONDITIONS #
#######################
def getDBC(x,flag):
    return lambda x,t: 0.
dirichletConditions = {0:getDBC}

def zeroadv(x,flag):
    None
advectiveFluxBoundaryConditions =  {0:zeroadv}

fluxBoundaryConditions = {0:'outFlow'}
diffusiveFluxBoundaryConditions = {0:{}}
