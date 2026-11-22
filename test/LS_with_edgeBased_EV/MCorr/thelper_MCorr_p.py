from proteus import *
from proteus.default_p import *
from .thelper_cons_ls import *
name=soname+"_phicor"
from proteus.kernels import MCorr


LevelModelType = MCorr.LevelModel
coefficients = MCorr.Coefficients(applyCorrection=applyCorrection,
                                  epsFactHeaviside=epsFactHeaviside,
                                  epsFactDirac=epsFactDirac,
                                  epsFactDiffusion=epsFactDiffusion,
                                  LSModel_index=LS_model, #0
                                  V_model=V_model, #0
                                  VOFModel_index=VOF_model,
                                  me_model=MCorr_model,
                                  checkMass=checkMass,                                  
                                  nd=nd,
                                  useQuadraticRegularization=ct.STABILIZATION_TYPE_vof>0,
                                  edgeBasedStabilizationMethods=ct.STABILIZATION_TYPE_vof>0)

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
