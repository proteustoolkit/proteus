from proteus.default_p import *
from proteus.kernels import RANS2P
import numpy as np
from proteus import Context
from proteus import defaults
defaults.reset_default_p()

ct = Context.get()
domain = ct.domain
nd = domain.nd
mesh = domain.MeshOptions
name = 'twp_navier_stokes'

genMesh = mesh.genMesh
movingDomain = ct.movingDomain
T = ct.T  # might not be necessaryd

LevelModelType = RANS2P.LevelModel
if ct.useOnlyVF:
    LS_model = None
else:
    LS_model = 2
if ct.useRANS >= 1:
    Closure_0_model = 5
    Closure_1_model = 6
    if ct.useOnlyVF:
        Closure_0_model = 2
        Closure_1_model = 3
    if ct.movingDomain:
        Closure_0_model += 1
        Closure_1_model += 1
else:
    Closure_0_model = None
    Closure_1_model = None

if ct.stabilization=='pressure_projection':
    mom_sge = 0.; pre_sge=0.; vel_sge=0.;pre_proj=1.0
elif ct.stabilization=='proteus_full':
    mom_sge=1.0; pre_sge=1.0; vel_sge=1.;pre_proj=0.0
    
coefficients = RANS2P.Coefficients(epsFact=ct.epsFact_viscosity,
                                   sigma=0.0,
                                   rho_0=ct.rho_0,
                                   nu_0=ct.nu_0,
                                   rho_1=ct.rho_1,
                                   nu_1=ct.nu_1,
                                   g=ct.g,
                                   nd=nd,
                                   # ME_model=int(ct.movingDomain)+0,
                                   # VF_model=int(ct.movingDomain)+1,
                                   # LS_model=int(ct.movingDomain)+LS_model,
                                   ME_model = 0,
                                   VF_model = 1,
                                   LS_model = LS_model,
                                   Closure_0_model=Closure_0_model,
                                   Closure_1_model=Closure_1_model,
                                   epsFact_density=ct.epsFact_density,
                                   stokes=False,
                                   useVF=ct.useVF,
                                   useRBLES=ct.useRBLES,
                                   useMetrics=ct.useMetrics,
                                   eb_adjoint_sigma=1.0,
                                   eb_penalty_constant=ct.weak_bc_penalty_constant,
                                   forceStrongDirichlet=ct.ns_forceStrongDirichlet,
                                   turbulenceClosureModel=ct.ns_closure,
                                   NONCONSERVATIVE_FORM=1.0,
                                   MOMENTUM_SGE=mom_sge,
                                   PRESSURE_SGE=pre_sge,
                                   VELOCITY_SGE=vel_sge,
                                   PRESSURE_PROJECTION_STABILIZATION=pre_proj,
                                   movingDomain=ct.movingDomain)

v_walls = [ct.domain.boundaryTags[f] for f in ['tank2D1_y+', 'tank2D1_y-']]
h_walls = [ct.domain.boundaryTags[f] for f in ['tank2D1_x+', 'tank2D1_x-']]
walls = [ct.domain.boundaryTags[f] for f in ['tank2D1_y+', 'tank2D1_y-', 'tank2D1_x+', 'tank2D1_x-']]

boundaryCreatesNullSpace = True

if ct.boundary_condition_type=='fs':
    def getDBC_p(x,flag):
        pass

    def getDBC_u(x,flag):
        # Could also return None here?
        if flag in h_walls:
            return lambda x,t: 0.0

    def getDBC_v(x,flag):
        # Could also return None here?
        if flag in v_walls:
            return lambda x,t: 0.0

    def getAFBC_p(x,flag):
        if flag in walls:
            return lambda x,t: 0.0

    def getAFBC_u(x,flag):
        if flag in walls:
            return lambda x,t: 0.0

    def getAFBC_v(x,flag):
        if flag in walls:
            return lambda x,t: 0.0

    def getDFBC_u(x,flag):
        if flag in walls or flag == 0:
            return lambda x,t: 0.0

    def getDFBC_v(x,flag):
        if flag in walls or flag == 0:
            return lambda x,t: 0.0

if ct.boundary_condition_type=='ns':

    def getDBC_p(x,flag):
        return None

    def getDBC_u(x,flag):
        if flag in walls:
            return lambda x,t: 0.0

    def getDBC_v(x,flag):
        if flag in walls:
            return lambda x,t: 0.0

    def getAFBC_p(x,flag):
        if flag in walls:
            return lambda x,t: 0.0

    def getAFBC_u(x,flag):
        if flag in walls:
            return lambda x,t: 0.0

    def getAFBC_v(x,flag):
        if flag in walls:
            return lambda x,t: 0.0

    def getDFBC_u(x,flag):
        if flag in walls:
            return lambda x,t: 0.0

    def getDFBC_v(x,flag):
        if flag in walls:
            return lambda x,t: 0.0

if ct.boundary_condition_type=='ns' or ct.boundary_condition_type=='fs':

    dirichletConditions = {0:getDBC_p,
                           1:getDBC_u,
                           2:getDBC_v}

    advectiveFluxBoundaryConditions =  {0:getAFBC_p,
                                        1:getAFBC_u,
                                        2:getAFBC_v}

    diffusiveFluxBoundaryConditions = {0:{},
                                       1:{1:getDFBC_u},
                                       2:{2:getDFBC_v}}

if ct.boundary_condition_type=='ot':

    dirichletConditions = {0: lambda x, flag: domain.bc[flag].p_dirichlet.init_cython(),
                           1: lambda x, flag: domain.bc[flag].u_dirichlet.init_cython(),
                           2: lambda x, flag: domain.bc[flag].v_dirichlet.init_cython()}

    advectiveFluxBoundaryConditions = {0: lambda x, flag: domain.bc[flag].p_advective.init_cython(),
                                       1: lambda x, flag: domain.bc[flag].u_advective.init_cython(),
                                       2: lambda x, flag: domain.bc[flag].v_advective.init_cython()}

    diffusiveFluxBoundaryConditions = {0: {},
                                       1: {1: lambda x, flag: domain.bc[flag].u_diffusive.init_cython()},
                                       2: {2: lambda x, flag: domain.bc[flag].v_diffusive.init_cython()}}

class PerturbedSurface_p:
    def __init__(self,waterLevel):
        self.waterLevel=waterLevel
    def uOfXT(self,x,t):
        if ct.signedDistance(x) < 0:
            return -(L[1] - self.waterLevel)*ct.rho_1*ct.g[1] - (self.waterLevel - x[1])*ct.rho_0*ct.g[1]
        else:
            return -(L[1] - self.waterLevel)*ct.rho_1*ct.g[1]

class AtRest:
    def __init__(self):
        pass
    def uOfXT(self,x,t):
        return 0.0

initialConditions = {0:PerturbedSurface_p(ct.waterLine_z),
                     1:AtRest(),
                     2:AtRest()}