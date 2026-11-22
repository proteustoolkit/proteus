from proteus.default_p import *
from proteus.kernels import RANS2P
from .multiphase import *
import numpy as np


Closure_0_model = None
Closure_1_model = None
LevelModelType = RANS2P.LevelModel
coefficients = RANS2P.Coefficients(epsFact=epsFact_viscosity,
                                   sigma=0.0,
                                   rho_0=rho_0,
                                   nu_0=nu_0,
                                   rho_1=rho_1,
                                   nu_1=nu_1,
                                   g=g,
                                   nd=nd,
                                   ME_model=0,
                                   CLSVOF_model=CLSVOF_model,
                                   VF_model=VF_model,
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
                                   movingDomain=movingDomain)
    
dirichletConditions = {0: lambda x, flag: domain.bc[flag].p_dirichlet.init_cython(),
                       1: lambda x, flag: domain.bc[flag].u_dirichlet.init_cython(),
                       2: lambda x, flag: domain.bc[flag].v_dirichlet.init_cython()}

advectiveFluxBoundaryConditions = {0: lambda x, flag: domain.bc[flag].p_advective.init_cython(),
                                   1: lambda x, flag: domain.bc[flag].u_advective.init_cython(),
                                   2: lambda x, flag: domain.bc[flag].v_advective.init_cython()}

diffusiveFluxBoundaryConditions = {0:{},
                                   1: {1: lambda x, flag: domain.bc[flag].u_diffusive.init_cython()},
                                   2: {2: lambda x, flag: domain.bc[flag].v_diffusive.init_cython()}}

class AtRest(object):
    def __init__(self):
        pass
    def uOfXT(self,x,t):
        return 0.0

initialConditions = {0:AtRest(),
                     1:AtRest(),
                     2:AtRest()}
