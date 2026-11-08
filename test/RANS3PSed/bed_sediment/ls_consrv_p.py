from proteus import StepControl
from proteus import *
from proteus.default_p import *
from proteus.mprans import MCorr3P
from proteus import Context

ct = Context.get()
domain = ct.domain
nd = ct.domain.nd
mesh = domain.MeshOptions


genMesh = mesh.genMesh
movingDomain = ct.movingDomain
T = ct.T

LevelModelType = MCorr3P.LevelModel

coefficients = MCorr3P.Coefficients(LS_model=ct.LS_model,
                                    V_model=ct.V_model,
                                    ME_model=ct.MCORR_model,
                                    VOF_model=ct.VOF_model,
                                    VOS_model=ct.VOS_model,
                                    applyCorrection=ct.applyCorrection,
                                    nd=nd,
                                    checkMass=False,
                                    useMetrics=ct.useMetrics,
                                    epsFactHeaviside=ct.epsFact_consrv_heaviside,
                                    epsFactDirac=ct.epsFact_consrv_dirac,
                                    epsFactDiffusion=ct.epsFact_consrv_diffusion)

class zero_phi:
    def __init__(self):
        pass
    def uOfX(self,X):
        return 0.0
    def uOfXT(self,X,t):
        return 0.0

initialConditions  = {0:zero_phi()}
