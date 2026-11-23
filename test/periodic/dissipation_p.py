from proteus import *
from proteus.default_p import *
from proteus.kernels import Dissipation
from proteus import Context
ct = Context.get()
domain = ct.domain
nd = domain.nd

LevelModelType = Dissipation.LevelModel
if ct.useOnlyVF:
    RD_model = None
    LS_model = None
    dissipation_model = 3
    ME_model = 2
else:
    RD_model = 3
    LS_model = 2
    ME_model = 6
    kappa_model = 5
if ct.movingDomain:
    kappa_model += 1
    ME_model += 1
    LS_model += 1
    RD_model += 1
#
dissipation_model_flag = 1
if ct.useRANS == 2:
    dissipation_model_flag=2
elif ct.useRANS == 3:
    dissipation_model_flag=3

coefficients = Dissipation.Coefficients(V_model=int(ct.movingDomain)+0,
                                        ME_model=ME_model,
                                        LS_model=LS_model,
                                        RD_model=RD_model,
                                        kappa_model=kappa_model,
                                        dissipation_model_flag=dissipation_model_flag,#1 -- K-epsilon, 2 -- K-omega 1998, 3 -- K-omega 1988
                                        useMetrics=ct.useMetrics,
                                        rho_0=ct.rho_0,nu_0=ct.nu_0,
                                        rho_1=ct.rho_1,nu_1=ct.nu_1,
                                        g=ct.g,
                                        c_mu=0.09,sigma_e=1.0,
                                        sc_uref=ct.dissipation_sc_uref,
                                        sc_beta=ct.dissipation_sc_beta)


dirichletConditions = {0: lambda x, flag: domain.bc[flag].dissipation_dirichlet.init_cython()}
advectiveFluxBoundaryConditions = {0: lambda x, flag: domain.bc[flag].dissipation_advective.init_cython()}
diffusiveFluxBoundaryConditions = {0: {0: lambda x, flag: domain.bc[flag].dissipation_diffusive.init_cython()}}



class ConstantIC:
    def __init__(self, cval=0.):
        self.cval = cval
    def uOfXT(self, x, t):
        return self.cval

kInflow = 0.
dissipationInflow = coefficients.c_mu*kInflow**(1.5)/(0.03*ct.tank_dim[nd-1])
if ct.useRANS >= 2:
    dissipationInflow = dissipationInflow/(kInflow+1.0e-12)

initialConditions = {0: ConstantIC(cval=dissipationInflow*0.001)}