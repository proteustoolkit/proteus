from proteus import *
from proteus.default_p import *
from couette import *
from proteus.kernels import RANS2P
from decimal import *

from proteus import Context
ct = Context.get()

LevelModelType = RANS2P.LevelModel
if useOnlyVF:
    LS_model = None
else:
    LS_model = 2

dragAlphaTypes = numpy.array([0.0,
                              0.0,
                              0.0,
                              0.0])
dragBetaTypes = numpy.array([0.0,0.0,0.0,0.0])

coefficients = RANS2P.Coefficients(epsFact=epsFact_viscosity,
                                   sigma=0.0,
                                   rho_0 = rho_0,
                                   nu_0 = nu_0,
                                   rho_1 = rho_1,
                                   nu_1 = nu_1,
                                   g=g,
                                   nd=nd,
                                   VF_model=1,
                                   LS_model=LS_model,
                                   epsFact_density=epsFact_density,
                                   stokes=False,
                                   useVF=useVF,
				   useRBLES=useRBLES,
				   useMetrics=useMetrics,
                                   eb_adjoint_sigma=1.0,
                                   dragAlphaTypes=dragAlphaTypes,
                                   dragBetaTypes=dragAlphaTypes,
                                   forceStrongDirichlet=ns_forceStrongDirichlet,
                                   turbulenceClosureModel=ns_closure)

Uinf = 0.002

def getDBC_p(x,flag):
    if flag==boundaryTags['back']:
        return lambda x,t: 0.0

def getDBC_u(x,flag):
    if flag in [boundaryTags['front'], boundaryTags['bottom'], boundaryTags['top'], boundaryTags['back']]:
        return lambda x,t: 0.0

def getDBC_v(x,flag):
    if flag in [boundaryTags['front'], boundaryTags['bottom'], boundaryTags['top'], boundaryTags['back']]:
        return lambda x,t: Uinf*x[2]*1.0/L[2]

def getDBC_w(x,flag):
    if flag in [boundaryTags['front'], boundaryTags['bottom'], boundaryTags['top'], boundaryTags['back']]:
        return lambda x,t: 0.0

dirichletConditions = {0:getDBC_p,
                       1:getDBC_u,
                       2:getDBC_v,
                       3:getDBC_w}

def getAFBC_p(x,flag):
    if flag==boundaryTags['front']:
        return lambda x,t: -Uinf*x[2]*1.0/L[2]
    elif flag==boundaryTags['back']:
        return None
    else:
        return lambda x,t: 0.0

def getAFBC_u(x,flag):
    if flag not in [boundaryTags['front'],boundaryTags['top'],boundaryTags['bottom'],boundaryTags['back']]:
        return lambda x,t: 0.0

def getAFBC_v(x,flag):
    if flag==boundaryTags['back']:
        return None
    elif flag not in [boundaryTags['front'],boundaryTags['top'],boundaryTags['bottom']]:
        return lambda x,t: 0.0

def getAFBC_w(x,flag):
    if flag not in [boundaryTags['front'],boundaryTags['top'],boundaryTags['bottom'],boundaryTags['back']]:
        return lambda x,t: 0.0

def getDFBC_u(x,flag):
    if flag not in [boundaryTags['front'],boundaryTags['back'], boundaryTags['bottom'], boundaryTags['top']]:
        return lambda x,t: 0.0

def getDFBC_v(x,flag):
    if flag not in [boundaryTags['front'],boundaryTags['back'], boundaryTags['bottom'], boundaryTags['top']]:
        return lambda x,t: 0.0

def getDFBC_w(x,flag):
    if flag == boundaryTags['back']:
        return None 
    elif flag not in [boundaryTags['front'], boundaryTags['bottom'], boundaryTags['top']]:
        return lambda x,t: 0.0


advectiveFluxBoundaryConditions =  {0:getAFBC_p,
                                    1:getAFBC_u,
                                    2:getAFBC_v,
                                    3:getAFBC_w}

diffusiveFluxBoundaryConditions = {0:{},
                                   1:{1:getDFBC_u},
                                   2:{2:getDFBC_v},
                                   3:{3:getDFBC_w}}

class PerturbedSurface_p(object):
    def __init__(self,waterLevel):
        self.waterLevel=waterLevel
    def uOfXT(self,x,t):
        return hydrostatic_pressure(x)

class AtRest(object):
    def __init__(self):
        pass
    def uOfXT(self,x,t):
        return 0.0

class CouetteProfile(object):
    def __init__(self):
        pass
    def uOfXT(self,x,t):
        v = Uinf*(x[2])/L[2]
        return v

class Constant(object):
    def __init__(self):
        pass
    def uOfXT(self,x,t):
        v = 1.0
        return v

initialConditions = {0:AtRest(),#0:PerturbedSurface_p(waterLine_z),
                     1:AtRest(),
                     2:AtRest(),
                     3:AtRest()}
