from proteus.default_p import *
from proteus.kernels import NCLS
from proteus import Context

ct = Context.get()
domain = ct.domain
nd = domain.nd
mesh = domain.MeshOptions


genMesh = mesh.genMesh
movingDomain = ct.movingDomain
T = ct.T

LevelModelType = NCLS.LevelModel

coefficients = NCLS.Coefficients(V_model=int(ct.movingDomain)+0,
                                 RD_model=int(ct.movingDomain)+3,
                                 ME_model=int(ct.movingDomain)+2,
                                 checkMass=False,
                                 useMetrics=ct.useMetrics,
                                 epsFact=ct.ecH,
                                 sc_uref=ct.ls_sc_uref,
                                 sc_beta=ct.ls_sc_beta,
                                 movingDomain=ct.movingDomain)

dirichletConditions = {0: lambda x, flag: None}
advectiveFluxBoundaryConditions = {}
diffusiveFluxBoundaryConditions = {0: {}}

class PerturbedSurface_phi:
    def uOfXT(self,x,t):
        return ct.signedDistance(x)

initialConditions  = {0:PerturbedSurface_phi()}