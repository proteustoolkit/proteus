from proteus import *
from proteus.default_p import *
try:
    from .risingBubble import *
except:
    from risingBubble import *
from proteus.kernels import MCorr3P

LevelModelType = MCorr3P.LevelModel
coefficients = MCorr3P.Coefficients(LS_model=LS_model,
                                    V_model=V_model,
                                    ME_model=MCORR_model,
                                    VOF_model=VOF_model,
                                    VOS_model=VOS_model,
                                    applyCorrection=applyCorrection,
                                    nd=nd,
                                    checkMass=False,
                                    useMetrics=useMetrics,
                                    epsFactHeaviside=epsFact_consrv_heaviside,
                                    epsFactDirac=epsFact_consrv_dirac,
                                    epsFactDiffusion=epsFact_consrv_diffusion)

class zero_phi(object):
    def __init__(self):
        pass
    def uOfX(self,X):
        return 0.0
    def uOfXT(self,X,t):
        return 0.0

initialConditions  = {0:zero_phi()}
