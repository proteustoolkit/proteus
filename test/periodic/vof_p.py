from proteus.default_p import *
from proteus.ctransportCoefficients import smoothedHeaviside
from proteus.kernels import VOF
from proteus import Context

ct = Context.get()
domain = ct.domain
nd = domain.nd
parallelPeriodic=True

movingDomain = ct.movingDomain
T = ct.T

LevelModelType = VOF.LevelModel
if ct.useOnlyVF:
    RD_model = None
    LS_model = None
else:
    RD_model = 3
    LS_model = 2

coefficients = VOF.Coefficients(LS_model=int(ct.movingDomain)+LS_model,
                                V_model=int(ct.movingDomain)+0,
                                RD_model=int(ct.movingDomain)+RD_model,
                                ME_model=int(ct.movingDomain)+1,
                                checkMass=True,
                                useMetrics=ct.useMetrics,
                                epsFact=ct.epsFact_vof,
                                sc_uref=ct.vof_sc_uref,
                                sc_beta=ct.vof_sc_beta,
                                movingDomain=ct.movingDomain)

def zero(x, t):
    return 0.0

dirichletConditions = {0: lambda x, flag: None}

advectiveFluxBoundaryConditions = {0: lambda x, flag: zero}

diffusiveFluxBoundaryConditions = {0: {}}

periodicDirichletConditions = {0:ct.getPDBC}

class VF_SOL:
    def uOfXT(self, x, t):
        return smoothedHeaviside(ct.epsFact_consrv_heaviside*ct.opts.he,
                                 (x[nd-1] - (max(ct.wave.eta(x, t%(ct.tank_dim[0]/ct.wave.c)),
                                                 ct.wave.eta(x+ct.tank_dim[0], t%(ct.tank_dim[0]/ct.wave.c)))
                                             +
                                             ct.opts.water_level)))
initialConditions = {0: VF_SOL()}
analyticalSolution = {0: VF_SOL()}