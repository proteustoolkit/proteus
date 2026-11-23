import numpy
from proteus import *
from proteus.default_p import *
from math import *
from proteus.kernels import VOF, NCLS, VOS3P
"""
Linear advection of a guassian with periodic bcs
"""

nd = 2#number of space dimensions

#constant velocity
velocity = numpy.array([1.0,1.0])
if nd == 3:
    velocity = numpy.array([1.0,1.0,1.0])
#where gaussian starts
center = numpy.array([0.5,0.5])#numpy.array([0.25,0.25])#numpy.array([0.25,0.5])
if nd == 3:
    velocity = numpy.array([1.0,1.0,1.0])
    center = numpy.array([0.5,0.5,0.5])#numpy.array([0.25,0.25])#numpy.array([0.25,0.5])
#size
sigma  = 1./16.
#quadrature
space_quad_order = 4#6 for dgp3
#form of level set equation
useHJ=False#True#False
#number nodes in each direction
nn=11#81#161#41
#number of meshes in multilevel mesh
nLevels = 1
#end time of simulation
T = 1.0
#number of output time steps, ignored for adaptive/CFL based runs
nDTout = 100
#max CFL
runCFL = 0.185#0.3,0.185,0.125 for dgp1,dgp2,dgpk(3)

name = 'la_periodicGauss_{0}d'.format(nd)

##  \page Tests Test Problems 
# \ref la_periodicGauss_2d_p.py "Linear advection of a Gaussian"
# \addtogroup test
#
#  \file la_periodicGauss_2d_p.py
# @{
#  

##\ingroup test
#  \brief Conservative linear advection of a cone in a constant
#  velocity field.
#


class ConstantVelocityGaussian2D:
    def __init__(self,sigma=1./8.,b=[1.0,0.0],xc=0.25,yc=0.5):
        self.sigma = sigma
        self.xc= xc
        self.yc= yc
        self.b = b
    def uOfXT(self,x,t):
        centerX = (self.xc + self.b[0]*t)%1.0
        centerY = (self.yc + self.b[1]*t)%1.0
        d2 = (x[0]-centerX)**2 + (x[1]-centerY)**2
        return exp(-0.5*d2/self.sigma**2)
class ConstantVelocityGaussian3D:
    def __init__(self,sigma=1./8.,b=[1.0,0.0,0.0],xc=0.25,yc=0.5,zc=0.5):
        self.sigma = sigma
        self.xc= xc
        self.yc= yc
        self.zc= zc
        self.b = b
    def uOfXT(self,x,t):
        centerX = (self.xc + self.b[0]*t)%1.0
        centerY = (self.yc + self.b[1]*t)%1.0
        centerZ = (self.zc + self.b[2]*t)%1.0
        d2 = (x[0]-centerX)**2 + (x[1]-centerY)**2 + (x[2]-centerZ)**2
        return exp(-0.5*d2/self.sigma**2)

#
analyticalSolution = {0:ConstantVelocityGaussian2D(sigma=sigma,
                                                   b=velocity,
                                                   xc=center[0],
                                                   yc=center[1])}
if nd == 3:
    analyticalSolution = {0:ConstantVelocityGaussian3D(sigma=sigma,
                                                       b=velocity,
                                                       xc=center[0],
                                                       yc=center[1],
                                                       zc=center[2])}
M = {0:1.0}
A = {0:numpy.zeros((nd,nd),'d')}
B = {0:velocity}
C = {0:0.0}

useVOS=True
useVOF=False
useNCLS=False
if useVOS:
    LevelModelType = VOS3P.LevelModel
    coefficients = VOS3P.Coefficients(LS_model=None,V_model=None,RD_model=None,ME_model=0,checkMass=False,
                                      epsFact=0.0,useMetrics=1.0,
                                      STABILIZATION_TYPE=2,
                                      LUMPED_MASS_MATRIX=False,
                                      ENTROPY_TYPE=2,
                                      FCT=True,
                                      num_fct_iter=1)
elif useNCLS:
    LevelModelType = NCLS.LevelModel
    coefficients = NCLS.Coefficients(V_model=None,RD_model=None,ME_model=0,checkMass=False,
                                     epsFact=0.0,useMetrics=1.0)
elif useVOF:
    LevelModelType = VOF.LevelModel
    coefficients = VOF.Coefficients(LS_model=None,V_model=None,RD_model=None,ME_model=0,checkMass=False,
                                    epsFact=0.0,useMetrics=1.0,FCT=False)
elif useHJ:
    coefficients = ConstantVelocityLevelSet(b=velocity)
else:
    coefficients = LinearVADR_ConstantCoefficients(nc=1,M=M,A=A,B=B,C=C)

coefficients.variableNames=['u']

#now define the Dirichlet boundary conditions

def getDBC(x,tag):
    return None

#and periodic boundary conditions
eps=1.0e-8
def getPDBC(x,tag):
    if (x[0] == 0.0 or x[0] == 1.0) and (x[1] == 0.0 or x[1] == 1.0):
        return numpy.array([0.0,0.0,0.0])
    elif x[0] == 0.0 or x[0] == 1.0:
        return numpy.array([0.0,round(x[1],5),0.0])
    elif (x[1] == 0.0 or x[1] == 1.0):# and (0.0 < x[0] and x[0] < 1):
        return numpy.array([round(x[0],5),0.0,0.0])
if nd == 3:
    def getPDBC(x,tag):
        if (x[0] == 0.0 or x[0] == 1.0) and (x[1] == 0.0 or x[1] == 1.0) and (x[2] == 0.0 or x[2] == 1.0):
            return numpy.array([0.0,0.0,0.0])
        elif (x[0] == 0.0 or x[0] == 1.0) and (x[1] == 0.0 or x[1] == 1.0):
            return numpy.array([0.0,0.0,round(x[2],5)])
        elif (x[0] == 0.0 or x[0] == 1.0) and (x[2] == 0.0 or x[2] == 1.0):
            return numpy.array([0.0,round(x[1],5),0.0])
        elif (x[1] == 0.0 or x[1] == 1.0) and (x[2] == 0.0 or x[2] == 1.0):
            return numpy.array([round(x[0],5),0.0,0.0])
        elif x[0] == 0.0 or x[0] == 1.0:
            return numpy.array([0.0,round(x[1],5),round(x[2],5)])
        elif (x[1] == 0.0 or x[1] == 1.0):# and (0.0 < x[0] and x[0] < 1):
            return numpy.array([round(x[0],5),0.0,round(x[2],5)])
        elif (x[2] == 0.0 or x[2] == 1.0):# and (0.0 < x[0] and x[0] < 1):
            return numpy.array([round(x[0],5),round(x[1],5),0.0])
    
periodicDirichletConditions = {0:getPDBC}
parallelPeriodic=True    
dirichletConditions = {0:getDBC}

initialConditions  = {0:analyticalSolution[0]}


def getFluxBC(x, flag):
    return lambda x,t: 0.0
advectiveFluxBoundaryConditions =  {0:getFluxBC}

diffusiveFluxBoundaryConditions = {0:{}}