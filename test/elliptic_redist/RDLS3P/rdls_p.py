from proteus import *
from proteus.default_p import *
from math import *
from .vortex2D import *
from proteus.kernels import RDLS

from . import ncls_p
name = soname+"_rdls"
LevelModelType = RDLS.LevelModel

coefficients = RDLS.Coefficients(applyRedistancing=True,
                                 epsFact=epsFactRedistance,
                                 nModelId=0,
                                 rdModelId=1,
                                 useMetrics=useMetrics,
                                 ELLIPTIC_REDISTANCING=ct.ELLIPTIC_REDISTANCING,
                                 backgroundDissipationEllipticRedist=1.0,
                                 alpha=1E9)
                                 
#now define the Dirichlet boundary conditions
def getDBC(x,flag):
    pass

dirichletConditions = {0:getDBC}

if LevelModelType == RDLS.LevelModel:
    weakDirichletConditions = {0:RDLS.setZeroLSweakDirichletBCsSimple}
else:
    weakDirichletConditions = {0:coefficients.setZeroLSweakDirichletBCs}

#weakDirichletConditions = {0:coefficients.setZeroLSweakDirichletBCs2}
#weakDirichletConditions = None

initialConditions  = ncls_p.initialConditions
fluxBoundaryConditions = {0:'noFlow'}
advectiveFluxBoundaryConditions =  {}
diffusiveFluxBoundaryConditions = {0:{}}