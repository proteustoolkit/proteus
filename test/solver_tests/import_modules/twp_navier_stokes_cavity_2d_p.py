"""
Incompressible Navier-Stokes flow around a cylinder in 2D.
"""
from proteus import *
from proteus.default_p import *
from proteus import defaults
defaults.reset_default_p()
try:
    from .cavity2d import *
except:
    from cavity2d import *
from proteus.kernels import RANS2P

bcsTimeDependent = True
LevelModelType = RANS2P.LevelModel
LS_model = None
name = 'twp_mprans_cavity'

coefficients = RANS2P.Coefficients(epsFact=epsFact_viscosity,
                                   rho_0=rho_0,
                                   nu_0=nu_0,
                                   rho_1=rho_1,
                                   nu_1=nu_1,
                                   g=g,
                                   nd=nd,
                                   LS_model=None,
                                   epsFact_density=epsFact_density,
                                   stokes=False,#useStokes,
                                   useVF=useVF,
                                   useRBLES=0.0,
                                   useMetrics=1.0,
                                   forceStrongDirichlet=ns_forceStrongDirichlet,
                                   NONCONSERVATIVE_FORM=1.0,
                                   MOMENTUM_SGE=0.0,
                                   PRESSURE_PROJECTION_STABILIZATION=1.0,
                                   nullSpace='NavierStokesConstantPressure')

class uTrue(object):
    def __init__(self):
        pass
    def uOfX(self,x):
        return 1.
    def uOfXT(self,x,t):
        return 1. - x[0]**4

class vTrue(object):
    def __init__(self):
        pass
    def vOfX(self,x):
        return 0.0
    def uOfXT(self,x,t):
        return self.vOfX(x)

def getDBC_p(x,flag):
    pass

def getDBC_u(x,flag):
    if flag in [boundaryTags['left'], boundaryTags['bottom'], boundaryTags['right']]:
        return lambda x,t : 0.0
    if flag in [boundaryTags['top']]:
        return lambda x,t: uTrue().uOfXT(x,t)

def getDBC_v(x,flag):
    if flag in [boundaryTags['left'], boundaryTags['top'], boundaryTags['bottom'], boundaryTags['right']]:
        return lambda x,t: 0.


dirichletConditions = {0:getDBC_p,
                       1:getDBC_u,
                       2:getDBC_v}

def getAFBC_p(x,flag):
    pass

def getAFBC_u(x,flag):
    pass

def getAFBC_v(x,flag):
    pass

def getDFBC_u(x,flag):
    pass
    
def getDFBC_v(x,flag):
    pass


advectiveFluxBoundaryConditions =  {0:getAFBC_p,
                                    1:getAFBC_u,
                                    2:getAFBC_v}

diffusiveFluxBoundaryConditions = {0:{},
                                   1:{1:getDFBC_u},
                                   2:{2:getDFBC_v}}

class Steady_p(object):
    def __init__(self):
        pass
    def uOfXT(self,x,t):
        return 0.0

class Steady_u(object):
    def __init__(self):
        pass
    def uOfXT(self,x,t):
        return 0.0

class Steady_v(object):
    def __init__(self):
        pass
    def uOfXT(self,x,t):
        return 0.0



initialConditions = {0:Steady_p(),
                     1:Steady_u(),
                     2:Steady_v()}
