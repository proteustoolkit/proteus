from proteus import *
from proteus.default_p import *
from tank import *
from proteus.mprans import RANS3PF
from proteus import Context

ct = Context.get()

LevelModelType = RANS3PF.LevelModel

coefficients = RANS3PF.Coefficients(epsFact=epsFact_viscosity,
                                    sigma=0.0,
                                    rho_0 = rho_0,
                                    nu_0 = nu_0,
                                    rho_1 = rho_1,
                                    nu_1 = nu_1,
                                    g=g,
                                    nd=nd,
                                    ME_model=ct.V_model,
                                    PRESSURE_model=ct.P_model,
                                    SED_model=ct.SED_model,
                                    VOF_model=ct.VOF_model,
                                    VOS_model=ct.VOS_model,
                                    LS_model=ct.LS_model,
                                    Closure_0_model=ct.K_model,
                                    Closure_1_model=ct.EPS_model,
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
                                    PSTAB=ct.opts.PSTAB,
                                    aDarcy = sedClosure.aDarcy,
                                    betaForch = sedClosure.betaForch,
                                    grain = sedClosure.grain,
                                    packFraction = sedClosure.packFraction,
                                    maxFraction = sedClosure.maxFraction,
                                    frFraction = sedClosure.frFraction,
                                    sigmaC = sedClosure.sigmaC,
                                    C3e = sedClosure.C3e,
                                    C4e = sedClosure.C4e,
                                    eR = sedClosure.eR,
                                    fContact = sedClosure.fContact,
                                    mContact = sedClosure.mContact,
                                    nContact = sedClosure.nContact,
                                    angFriction = sedClosure.angFriction,
                                    vos_function = ct.vos_function,
                                    vos_limiter = ct.sedClosure.vos_limiter,
                                    mu_fr_limiter = ct.sedClosure.mu_fr_limiter,
                                    )

dirichletConditions = {0: lambda x, flag: domain.bc[flag].u_dirichlet.init_cython(),
                       1: lambda x, flag: domain.bc[flag].v_dirichlet.init_cython()}

advectiveFluxBoundaryConditions = {0: lambda x, flag: domain.bc[flag].u_advective.init_cython(),
                                   1: lambda x, flag: domain.bc[flag].v_advective.init_cython()}

diffusiveFluxBoundaryConditions = {0: {0: lambda x, flag: domain.bc[flag].u_diffusive.init_cython()},
                                   1: {1: lambda x, flag: domain.bc[flag].v_diffusive.init_cython()}}

if nd == 3:
    dirichletConditions[2] = lambda x, flag: domain.bc[flag].w_dirichlet.init_cython()
    advectiveFluxBoundaryConditions[2] = lambda x, flag: domain.bc[flag].w_advective.init_cython()
    diffusiveFluxBoundaryConditions[2] = {2: lambda x, flag: domain.bc[flag].w_diffusive.init_cython()}

class AtRest:
    def __init__(self):
        pass
    def uOfXT(self,x,t):
        return 0.0


smoothing = 3*he


class U_IC:


    def uOfXT(self,x,t):
        if ct.signedDistance(x) <= -0.28:
            H = 1
            speed = 0
        elif -0.278 < ct.signedDistance(x) < -0.28 + smoothing:
            H = 1-(smoothedHeaviside(smoothing/2. , ct.signedDistance(x)+0.28 - smoothing/2.))
            speed = ct.opts.inflow_vel

        elif  (ct.signedDistance(x) <= 0): 
        #and ct.signedDistance(x)>= -0.27+smoothing):
            H = 0
            speed = ct.opts.inflow_vel
        
        elif 0 < ct.signedDistance(x) <= smoothing:
            H = smoothedHeaviside(smoothing/2. , ct.signedDistance(x) - smoothing/2.)
            speed = ct.opts.inflow_vel
        else:
            H = 1
            speed = 0
        
        u = H * 0.0 + (1-H)*speed

        return u


class V_IC:
    def uOfXT(self,x,t):
                
        v = 0.0

        return v




#if ct.opts.current:
#    initialConditions = {0: U_IC(),
#                         1: V_IC()}
#
#else:    
#    initialConditions = {0:AtRest(),
#                         1:AtRest()}

initialConditions = {0:AtRest(),
                     1:AtRest()}
