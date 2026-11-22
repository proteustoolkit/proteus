from proteus.default_p import *
from proteus.ctransportCoefficients import smoothedHeaviside
from proteus.kernels import VOF
from .multiphase import *

LevelModelType = VOF.LevelModel
if useOnlyVF:
    RD_model = None
    LS_model = None
else:
    RD_model = 3
    LS_model = 2

coefficients = VOF.Coefficients(LS_model=int(movingDomain)+LS_model,
                                V_model=int(movingDomain)+0,
                                RD_model=int(movingDomain)+RD_model,
                                ME_model=int(movingDomain)+1,
                                checkMass=False,
                                useMetrics=useMetrics,
                                epsFact=epsFact_vof,
                                sc_uref=vof_sc_uref,
                                sc_beta=vof_sc_beta,
                                movingDomain=movingDomain)

dirichletConditions = {0: lambda x, flag: domain.bc[flag].vof_dirichlet.init_cython()}

advectiveFluxBoundaryConditions = {0: lambda x, flag: domain.bc[flag].vof_advective.init_cython()}

diffusiveFluxBoundaryConditions = {0:{}}

class PerturbedSurface_H(object):
    def uOfXT(self,x,t):
        return smoothedHeaviside(ecH*he,signedDistance(x))
	    
initialConditions  = {0:PerturbedSurface_H()}
