from proteus.default_p import *
from proteus.kernels import NCLS
from .multiphase import *

LevelModelType = NCLS.LevelModel
coefficients = NCLS.Coefficients(V_model=int(movingDomain)+0,
                                 RD_model=int(movingDomain)+3,
                                 ME_model=int(movingDomain)+2,
                                 checkMass=False,
                                 useMetrics=useMetrics,
                                 epsFact=ecH,
                                 sc_uref=ls_sc_uref,
                                 sc_beta=ls_sc_beta,
                                 movingDomain=movingDomain)
 
dirichletConditions = {0: lambda x, flag: None}
advectiveFluxBoundaryConditions =  {}
diffusiveFluxBoundaryConditions = {0:{}}

class PerturbedSurface_phi(object):       
    def uOfXT(self,x,t):
        return signedDistance(x)
    
initialConditions  = {0:PerturbedSurface_phi()}
