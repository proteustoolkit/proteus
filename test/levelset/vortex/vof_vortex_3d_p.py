from proteus import *
from proteus.default_p import *
from proteus.ctransportCoefficients import smoothedHeaviside
from .vortex import *
from proteus.kernels import VOF
name=soname+"_vof"

"""
The non-conservative level set description of a bubble in a two-phase flow
"""

LevelModelType = VOF.LevelModel

##\ingroup test
#\file vof_vortex_2d_p.py
#
# \todo finish vof_vortex_2d_p.py

if applyRedistancing:
    coefficients = VOF.Coefficients(LS_model=0,V_model=0,RD_model=1,ME_model=2,checkMass=checkMass,
                                   epsFact=epsFact_vof)
else:
    coefficients = VOF.Coefficients(LS_model=0,V_model=0,RD_model=-1,ME_model=1,checkMass=checkMass,
                                   epsFact=epsFact_vof)

def Heaviside(phi):
    if phi > 0:
        return 1.0
    elif phi < 0:
        return 0.0
    else:
        return 0.5

class Vortex_phi(object):
    def __init__(self,center=[0.5,0.75,0.5],radius=0.15):
        self.radius  = radius
        self.center  = center
    def uOfX(self,X):
        dx = X[0]-self.center[0]; dy = X[1]-self.center[1]; dz = X[2]-self.center[2]
        dBubble = self.radius - sqrt(dx**2 + dy**2 + dz**2)
        return smoothedHeaviside(epsFactHeaviside*he,dBubble)#Heaviside(dBubble)
    #end
    def uOfXT(self,X,t):
        return self.uOfX(X)
    #end
#end Vortex_phi

class Vortex_phi_cylinder(object):
    def __init__(self,center=[0.5,0.75,0.5],radius=0.15):
        self.radius  = radius
        self.center  = center
    def uOfX(self,X):
        dx = X[0]-self.center[0]; dy = X[1]-self.center[1]; dz = X[2]-self.center[2]
        dBubble = self.radius - sqrt(dx**2 + dy**2)
        return smoothedHeaviside(epsFactHeaviside*he,dBubble)#Heaviside(dBubble)
    #end
    def uOfXT(self,X,t):
        return self.uOfX(X)
    #end
#end Vortex_phi

analyticalSolutions = None

def getDBC(x,flag):
    pass

dirichletConditions = {0:getDBC}

#cek changed to put sphere inside arbitrary box with dim L
if pseudo2D:
    initialConditions  = {0:Vortex_phi_cylinder(center=[0.5*L[0],0.75*L[1],0.5*L[2]],radius=0.15*L[0])}
else:
    initialConditions  = {0:Vortex_phi(center=[0.5*L[0],0.75*L[1],0.5*L[2]],radius=0.15*L[0])}

fluxBoundaryConditions = {0:'outFlow'}

#cek made no flux since v.n = 0 for this v
def getAFBC(x,flag):
   return lambda x,t: 0.0

advectiveFluxBoundaryConditions =  {0:getAFBC}

diffusiveFluxBoundaryConditions = {0:{}}
