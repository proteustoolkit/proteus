from proteus import *
from proteus.default_p import *
from cylinder import *
from proteus.kernels import RANS3PF

name = "momentum"
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
                                    PSTAB=1.0,
                                    nParticles=1,
                                    particle_epsFact=1.5,
                                    particle_alpha=1e6,
                                    particle_beta=1e6,
                                    particle_penalty_constant=1e16,
                                    particle_sdfList=[particle_sdf],
                                    particle_velocityList=[particle_vel],
                                    use_sbm=USE_SBM,
                                    use_ball_as_particle = 0)


def getDBC_u(x,flag):
    if flag in [boundaryTags['left']]:
        return lambda x,t: velRamp(t,x)
    elif flag in [boundaryTags['top'],boundaryTags['bottom']]:
        return lambda x,t: 0.0
    else:
        return None

def getDBC_v(x,flag):
    if flag in[boundaryTags['left'],boundaryTags['top'],boundaryTags['bottom']]:
        return lambda x,t: 0.0
    else:
        return None

def getDBC_w(x,flag):
    if flag in[boundaryTags['left'],boundaryTags['top'],boundaryTags['bottom']]:
        return lambda x,t: 0.0
    else:
        return None
    
dirichletConditions = {0:getDBC_u,
                       1:getDBC_v,
                       2:getDBC_w}

def getAFBC_u(x,flag):
  if flag in [boundaryTags['left'],boundaryTags['top'],boundaryTags['bottom']]:
      return None
  else:
      return lambda x,t: 0.0

def getAFBC_v(x,flag):
  if flag in [boundaryTags['left'],boundaryTags['top'],boundaryTags['bottom']]:
      return None
  else:
      return lambda x,t: 0.0
def getAFBC_w(x,flag):
  if flag in [boundaryTags['left'],boundaryTags['top'],boundaryTags['bottom']]:
      return None
  else:
      return lambda x,t: 0.0
      
def getDFBC_u(x,flag):
  if flag in [boundaryTags['left'],boundaryTags['top'],boundaryTags['bottom']]:
      return None
  else:
      return lambda x,t: 0.0

def getDFBC_v(x,flag):
  if flag in [boundaryTags['left'],boundaryTags['top'],boundaryTags['bottom']]:
      return None
  else:
      return lambda x,t: 0.0
  
def getDFBC_w(x,flag):
  if flag in [boundaryTags['left'],boundaryTags['top'],boundaryTags['bottom']]:
      return None
  else:
      return lambda x,t: 0.0

advectiveFluxBoundaryConditions =  {0:getAFBC_u,
                                    1:getAFBC_v,
                                    2:getAFBC_w}

diffusiveFluxBoundaryConditions = {0:{0:getDFBC_u},
                                   1:{1:getDFBC_v},
                                   2:{2:getDFBC_w}}

class AtRest(object):
    def __init__(self):
        pass
    def uOfXT(self,x,t):
        return 0.0

initialConditions = {0:AtRest(),
                     1:AtRest(),
                     2:AtRest()}
