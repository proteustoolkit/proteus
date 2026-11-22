from proteus import *
from proteus.default_p import *
try:
    from .multiphase import *
except:
    from multiphase import *
from proteus.kernels import RANS3PF

#No RANS
Closure_0_model = None
Closure_1_model = None

LevelModelType = RANS3PF.LevelModel
coefficients = RANS3PF.Coefficients(epsFact=epsFact_viscosity,
                                    sigma=sigma_01,
                                    rho_0 = rho_0,
                                    nu_0 = nu_0,
                                    rho_1 = rho_1,
                                    nu_1 = nu_1,
                                    g=g,
                                    nd=nd,
                                    ME_model=V_model,
                                    PRESSURE_model=PRESSURE_model,
                                    SED_model=SED_model,
                                    CLSVOF_model=CLSVOF_model,                                    
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
                                    USE_SUPG=USE_SUPG_NS,
                                    ARTIFICIAL_VISCOSITY=ARTIFICIAL_VISCOSITY_NS,
                                    cE=1.0, cMax=1.0)

# ----- DIRICHLET BOUNDARY CONDITIONS ----- #
def getDBC_u(x,flag):
    if flag == boundaryTags['top'] and openTop:
        return lambda x,t: 0.0

def getDBC_v(x,flag):
    if flag == boundaryTags['top'] and openTop:
        return lambda x,t: 0.0

def getDBC_w(x,flag):
    if flag == boundaryTags['top'] and openTop:
        return lambda x,t: 0.0

# ----- ADVECTIVE FLUX BOUNDARY CONDITIONS ----- #
def getAFBC_u(x,flag):
    if flag != boundaryTags['top'] or not openTop:
        return lambda x,t: 0.0

def getAFBC_v(x,flag):
    if flag != boundaryTags['top'] or not openTop:
        return lambda x,t: 0.0

def getAFBC_w(x,flag):
    if flag != boundaryTags['top'] or not openTop:
        return lambda x,t: 0.0

# ----- DIFFUSIVE FLUX BOUNDARY CONDITIONS ----- #
def getDFBC_u(x,flag):
    return lambda x,t: 0.0

def getDFBC_v(x,flag):
    return lambda x,t: 0.0

def getDFBC_w(x,flag):
    return lambda x,t: 0.0

class AtRest(object):
    def __init__(self):
        pass
    def uOfXT(self,x,t):
        return 0.0
    
if nd==2:
    # BOUNDARY CONDITIONS #
    dirichletConditions = {0:getDBC_u,
                           1:getDBC_v}    
    advectiveFluxBoundaryConditions =  {0:getAFBC_u,
                                        1:getAFBC_v}
    diffusiveFluxBoundaryConditions = {0:{0:getDFBC_u},
                                       1:{1:getDFBC_v}}
    # INITIAL CONDITIONS #
    initialConditions = {0:AtRest(),
                         1:AtRest()}
else:
    # BOUNDARY CONDITIONS #
    dirichletConditions = {0:getDBC_u,
                           1:getDBC_v,
                           2:getDBC_w}    
    advectiveFluxBoundaryConditions =  {0:getAFBC_u,
                                        1:getAFBC_v,
                                        2:getAFBC_w}
    diffusiveFluxBoundaryConditions = {0:{0:getDFBC_u},
                                       1:{1:getDFBC_v},
                                       2:{2:getDFBC_w}}
    # INITIAL CONDITIONS #
    initialConditions = {0:AtRest(),
                         1:AtRest(),
                         2:AtRest()}
