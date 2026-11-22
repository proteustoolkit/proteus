from proteus import *
from proteus.default_p import *
from couette import *
from proteus.kernels import MCorr

LevelModelType = MCorr.LevelModel

coefficients = MCorr.Coefficients(LSModel_index=2,V_model=0,me_model=4,VOFModel_index=1,
                                  applyCorrection=applyCorrection,nd=nd,checkMass=False,useMetrics=useMetrics,
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



