from proteus import *
from proteus.default_p import *
from math import *
from .thelper_cons_ls import *
from proteus.kernels import RDLS
from . import thelper_ncls_p

LevelModelType = RDLS.LevelModel

coefficients = RDLS.Coefficients(applyRedistancing=True,
                                 epsFact=epsFactRedistance,
                                 nModelId=0,
                                 rdModelId=1)

#now define the Dirichlet boundary conditions

def getDBC(x,flag):
    pass

dirichletConditions = {0:getDBC}

if LevelModelType == RDLS.LevelModel:
    #weakDirichletConditions = {0:RDLS.setZeroLSweakDirichletBCs}
    weakDirichletConditions = {0:RDLS.setZeroLSweakDirichletBCsSimple}
else:
    weakDirichletConditions = {0:coefficients.setZeroLSweakDirichletBCs}

#weakDirichletConditions = {0:coefficients.setZeroLSweakDirichletBCs2}
#weakDirichletConditions = None

initialConditions  = thelper_ncls_p.initialConditions

fluxBoundaryConditions = {0:'noFlow'}

advectiveFluxBoundaryConditions =  {}

diffusiveFluxBoundaryConditions = {0:{}}
