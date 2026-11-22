from proteus.default_p import *
from proteus.kernels import MCorr
from proteus import Context

ct = Context.get()
domain = ct.domain
nd = ct.domain.nd
mesh = domain.MeshOptions


genMesh = mesh.genMesh
movingDomain = ct.movingDomain
T = ct.T

LevelModelType = MCorr.LevelModel

coefficients = MCorr.Coefficients(LSModel_index=int(ct.movingDomain)+2,
                                  V_model=int(ct.movingDomain)+0,
                                  me_model=int(ct.movingDomain)+4,
                                  VOFModel_index=int(ct.movingDomain)+1,
                                  applyCorrection=ct.applyCorrection,
                                  nd=nd,
                                  checkMass=True,
                                  useMetrics=ct.useMetrics,
                                  epsFactHeaviside=ct.epsFact_consrv_heaviside,
                                  epsFactDirac=ct.epsFact_consrv_dirac,
                                  epsFactDiffusion=ct.epsFact_consrv_diffusion)

class zero_phi:
    def __init__(self):
        pass
    def uOfX(self, X):
        return 0.0
    def uOfXT(self, X, t):
        return 0.0

initialConditions = {0: zero_phi()}

periodicDirichletConditions = {0:ct.getPDBC}