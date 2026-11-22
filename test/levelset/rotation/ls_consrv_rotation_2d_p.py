from proteus import *
from proteus.default_p import *
try:
    from .rotation2D import *
except:
    from rotation2D import *
name=soname+"_phicor"
from proteus.kernels import MCorr


LevelModelType = MCorr.LevelModel
coefficients = MCorr.Coefficients(applyCorrection=applyCorrection,
                                  epsFactHeaviside=epsFactHeaviside,
                                  epsFactDirac=epsFactDirac,
                                  epsFactDiffusion=epsFactDiffusion,
                                  LSModel_index=0,
                                  V_model=0,
                                  me_model=3,
                                  VOFModel_index=2,
                                  checkMass=checkMass,
                                  nd=nd,
                                  useMetrics=useMetrics)

class zero_phi(object):
    def __init__(self):
        pass
    def uOfX(self,X):
        return 0.0
    def uOfXT(self,X,t):
        return self.uOfX(X)

analyticalSolutions = None

def getDBC_cnsrv(x,flag):
    pass

dirichletConditions = {0:getDBC_cnsrv}

initialConditions  = {0:zero_phi()}

fluxBoundaryConditions = {0:'noFlow'}


def getAFBC_cnsrv(x):
    return lambda x,t: 0.0
def getDFBC_cnsrv(x):
    return lambda x,t: 0.0

advectiveFluxBoundaryConditions =  {}

diffusiveFluxBoundaryConditions = {0:{0:getDFBC_cnsrv}}
