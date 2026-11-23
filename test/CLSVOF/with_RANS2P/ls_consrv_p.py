from proteus.default_p import *
from proteus.kernels import MCorr
from .multiphase import *

LevelModelType = MCorr.LevelModel
coefficients = MCorr.Coefficients(LSModel_index=int(movingDomain)+2,
                                  V_model=int(movingDomain)+0,
                                  me_model=int(movingDomain)+4,
                                  VOFModel_index=int(movingDomain)+1,
                                  applyCorrection=applyCorrection,
                                  nd=nd,
                                  checkMass=False,
                                  useMetrics=useMetrics,
                                  epsFactHeaviside=ecH,
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



