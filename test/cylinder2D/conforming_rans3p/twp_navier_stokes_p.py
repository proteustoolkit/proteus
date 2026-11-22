from proteus import *
from proteus.default_p import *
try:
    from .cylinder3p import *
except:
    from cylinder3p import *
from proteus.kernels import RANS3PF

LevelModelType = RANS3PF.LevelModel
if useOnlyVF:
    LS_model = None
else:
    if sedimentDynamics:
        LS_model = 2
    else:
        LS_model = None
if useRANS >= 1:
    Closure_0_model = 5; Closure_1_model=6
    if useOnlyVF:
        Closure_0_model=2; Closure_1_model=3
    if movingDomain:
        Closure_0_model += 1; Closure_1_model += 1
else:
    Closure_0_model = None
    Closure_1_model = None
name = "momentum"
coefficients = RANS3PF.Coefficients(epsFact=epsFact_viscosity,
                                    sigma=0.0,
                                    rho_0 = rho_0,
                                    nu_0 = nu_0,
                                    rho_1 = rho_1,
                                    nu_1 = nu_1,
                                    g=g,
                                    nd=nd,
                                    ME_model=V_model,
                                    PRESSURE_model=PRESSURE_model,
                                    SED_model=SED_model,
                                    VOF_model=VOF_model,
                                    VOS_model=VOS_model,
                                    LS_model=LS_model,
                                    Closure_0_model=Closure_0_model,
                                    Closure_1_model=Closure_1_model,
                                    epsFact_density=epsFact_density,
                                    stokes=False,
                                    useVF=useVF,
                                    useRBLES=useRBLES,
                                    useMetrics=useMetrics,
                                    eb_adjoint_sigma=1.0,
                                    eb_penalty_constant=weak_bc_penalty_constant,
                                    forceStrongDirichlet=ns_forceStrongDirichlet,
                                    turbulenceClosureModel=ns_closure,
                                    movingDomain=movingDomain,
                                    dragAlpha=dragAlpha,
                                    PSTAB=0.0,
                                    USE_SUPG=0.0,
                                    ARTIFICIAL_VISCOSITY=1)
def vel(x,t):
    U = Um*x[1]*(fl_H-x[1])/(fl_H/2.0)**2
    if t < 2.0:
        return t*U/2.0
    else:
        return U
    return U

def getDBC_u(x,flag):
    if flag in[ boundaryTags['left']]:
        return vel
    elif flag in [boundaryTags['front'],boundaryTags['back'],boundaryTags['top'],boundaryTags['bottom'],boundaryTags['obstacle']]:
        return lambda x,t: 0.0
    elif ns_forceStrongDirichlet==False and flag == boundaryTags['right']:
        return vel

def getDBC_v(x,flag):
    if flag in[boundaryTags['left']]:
        return lambda x,t: 0.0
    elif flag in [boundaryTags['front'],boundaryTags['back'],boundaryTags['top'],boundaryTags['bottom'],boundaryTags['obstacle']]:
        return lambda x,t: 0.0
    elif ns_forceStrongDirichlet==False and flag == boundaryTags['right']:
        return lambda x,t: 0.0

dirichletConditions = {0:getDBC_u,
                       1:getDBC_v}

def getAFBC_u(x,flag):
    if flag in [boundaryTags['left'],boundaryTags['front'],boundaryTags['back'],boundaryTags['right'],boundaryTags['top'],boundaryTags['bottom'],boundaryTags['obstacle']]:
        return None
    else:
        return lambda x,t: 0.0

def getAFBC_v(x,flag):
      if flag in [boundaryTags['left'],boundaryTags['front'],boundaryTags['back'],boundaryTags['right'],boundaryTags['top'],boundaryTags['bottom'],boundaryTags['obstacle']]:
          return None
      else:
          return lambda x,t: 0.0

def getDFBC_u(x,flag):
  if flag in [boundaryTags['left'],boundaryTags['front'],boundaryTags['back'],boundaryTags['top'],boundaryTags['bottom'],boundaryTags['obstacle']]:
      return None
  else:
      return lambda x,t: 0.0

def getDFBC_v(x,flag):
  if flag in [boundaryTags['left'],boundaryTags['front'],boundaryTags['back'],boundaryTags['top'],boundaryTags['bottom'],boundaryTags['obstacle']]:
      return None
  else:
      return lambda x,t: 0.0

advectiveFluxBoundaryConditions =  {0:getAFBC_u,
                                    1:getAFBC_v}

diffusiveFluxBoundaryConditions = {0:{0:getDFBC_u},
                                   1:{1:getDFBC_v}}

class AtRest(object):
    def __init__(self):
        pass
    def uOfXT(self,x,t):
        return 0.0

initialConditions = {0:AtRest(),
                     1:AtRest()}
