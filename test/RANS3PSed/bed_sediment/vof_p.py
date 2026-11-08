from proteus.default_p import *
from proteus.ctransportCoefficients import smoothedHeaviside
from proteus.mprans import VOF3P
from proteus import Context

ct = Context.get()
domain = ct.domain
nd = domain.nd
mesh = domain.MeshOptions


genMesh = mesh.genMesh
movingDomain = ct.movingDomain
T = ct.T
LevelModelType = VOF3P.LevelModel

coefficients = VOF3P.Coefficients(LS_model=ct.LS_model,
                                  V_model=ct.V_model,
                                  RD_model=ct.RD_model,
                                  ME_model=ct.VOF_model,
                                  VOS_model=ct.VOS_model,
                                  checkMass=True,
                                  useMetrics=ct.useMetrics,
                                  epsFact=ct.epsFact_vof,
                                  sc_uref=ct.vof_sc_uref,
                                  sc_beta=ct.vof_sc_beta,
                                  movingDomain=ct.movingDomain)

dirichletConditions = {0: lambda x, flag: domain.bc[flag].vof_dirichlet.init_cython()}

advectiveFluxBoundaryConditions = {0: lambda x, flag: domain.bc[flag].vof_advective.init_cython()}

diffusiveFluxBoundaryConditions = {0: {}}

class VF_IC:
    def uOfXT(self, x, t):
        return smoothedHeaviside(ct.epsFact_consrv_heaviside*mesh.he,x[nd-1]-ct.waterLevel)

initialConditions = {0: VF_IC()}
