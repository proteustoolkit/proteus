from proteus import StepControl
from proteus import *
from proteus.default_p import *
from proteus.mprans import NCLS3P
from proteus import Context

ct = Context.get()
domain = ct.domain
nd = ct.domain.nd
mesh = domain.MeshOptions


genMesh = mesh.genMesh
movingDomain = ct.movingDomain
T = ct.T

LevelModelType = NCLS3P.LevelModel

coefficients = NCLS3P.Coefficients(V_model=ct.V_model,
                                   RD_model=ct.RD_model,
                                   ME_model=ct.LS_model,
                                   checkMass=False,
                                   useMetrics=ct.useMetrics,
                                   epsFact=ct.epsFact_consrv_heaviside,
                                   sc_uref=ct.ls_sc_uref,
                                   sc_beta=ct.ls_sc_beta,
                                   movingDomain=ct.movingDomain)

dirichletConditions = {0: lambda x, flag: None}

advectiveFluxBoundaryConditions = {}

diffusiveFluxBoundaryConditions = {0: {}}

class PHI_IC:
    def uOfXT(self, x, t):
        return x[nd-1] - ct.waterLevel

initialConditions = {0: PHI_IC()}
