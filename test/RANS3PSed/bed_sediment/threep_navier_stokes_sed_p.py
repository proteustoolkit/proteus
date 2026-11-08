from proteus import *
from proteus.default_p import *
from tank import *
from proteus.mprans import RANS3PSed
from proteus import Context

ct = Context.get()

LevelModelType = RANS3PSed.LevelModel


coefficients = RANS3PSed.Coefficients(epsFact=epsFact_viscosity,
                                      sigma=0.0,
                                      rho_0 = rho_0,
                                      nu_0 = nu_0, 
                                      rho_1 = rho_1,
                                      nu_1 = nu_1,
                                      rho_s = rho_s,
                                      g=g,
                                      nd=nd,
                                      ME_model=ct.SED_model,
                                      PRESSURE_model=ct.P_model,
                                      FLUID_model=ct.V_model,
                                      VOS_model=ct.VOS_model,
                                      VOF_model=ct.VOF_model,
                                      LS_model=ct.LS_model,
                                      Closure_0_model=ct.K_model,
                                      Closure_1_model=ct.EPS_model,
                                      epsFact_density=epsFact_density,
                                      stokes=False,
                                      useVF=True,
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
                                    staticSediment = False,
                                    vos_limiter = ct.sedClosure.vos_limiter,
                                    mu_fr_limiter = ct.sedClosure.mu_fr_limiter)

dirichletConditions = {0: lambda x, flag: domain.bc[flag].us_dirichlet.init_cython(),
                       1: lambda x, flag: domain.bc[flag].vs_dirichlet.init_cython()}

advectiveFluxBoundaryConditions = {0: lambda x, flag: domain.bc[flag].us_advective.init_cython(),
                                   1: lambda x, flag: domain.bc[flag].vs_advective.init_cython()}

diffusiveFluxBoundaryConditions = {0: {0: lambda x, flag: domain.bc[flag].us_diffusive.init_cython()},
                                   1: {1: lambda x, flag: domain.bc[flag].vs_diffusive.init_cython()}}

if nd == 3:
    dirichletConditions[2] = lambda x, flag: domain.bc[flag].ws_dirichlet.init_cython()
    advectiveFluxBoundaryConditions[2] = lambda x, flag: domain.bc[flag].ws_advective.init_cython()
    diffusiveFluxBoundaryConditions[2] = {2: lambda x, flag: domain.bc[flag].ws_diffusive.init_cython()}

class AtRest:
    def __init__(self):
        pass
    def uOfXT(self,x,t):
        return 0.0

initialConditions = {0:AtRest(),
                     1:AtRest()}
