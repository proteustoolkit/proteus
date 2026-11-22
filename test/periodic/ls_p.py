from proteus.default_p import *
from proteus.kernels import NCLS
from proteus import Context

ct = Context.get()
domain = ct.domain
nd = domain.nd
movingDomain = ct.movingDomain
T = ct.T

LevelModelType = NCLS.LevelModel

coefficients = NCLS.Coefficients(V_model=int(ct.movingDomain)+0,
                                 RD_model=int(ct.movingDomain)+3,
                                 ME_model=int(ct.movingDomain)+2,
                                 checkMass=True,
                                 useMetrics=ct.useMetrics,
                                 epsFact=ct.epsFact_consrv_heaviside,
                                 sc_uref=ct.ls_sc_uref,
                                 sc_beta=ct.ls_sc_beta,
                                 movingDomain=ct.movingDomain)

dirichletConditions = {0: lambda x, flag: None}

advectiveFluxBoundaryConditions = {}

diffusiveFluxBoundaryConditions = {0: {}}

periodicDirichletConditions = {0:ct.getPDBC}

class PHI_SOL:
    def uOfXT(self, x, t):
        return (x[nd-1] - (max(ct.wave.eta(x, t%(ct.tank_dim[0]/ct.wave.c)),
                               ct.wave.eta(x+ct.tank_dim[0], t%(ct.tank_dim[0]/ct.wave.c)))
                           +
                           ct.opts.water_level))

initialConditions = {0: PHI_SOL()}

analyticalSolution = {0: PHI_SOL()}