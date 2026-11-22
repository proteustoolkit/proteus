from proteus import *
from proteus.default_p import *
from math import *
try:
    from .vortex2D import *
    from . import ncls_p
except:
    from vortex2D import *
    import ncls_p
from proteus.kernels import RDLS

name = soname+"_rdls"
LevelModelType = RDLS.LevelModel

coefficients = RDLS.Coefficients(applyRedistancing=True,
                                 epsFact=epsFactRedistance,
                                 nModelId=0,
                                 rdModelId=1,
                                 useMetrics=useMetrics,
                                 ELLIPTIC_REDISTANCING=ct.ELLIPTIC_REDISTANCING,
                                 backgroundDissipationEllipticRedist=1.0,
                                 backgroundDiffusionFactor=0.01,
                                 alpha=1E9,
                                 useExact=useExact,
                                 weakDirichletFactor=weakDirichletFactor)
                                 
#now define the Dirichlet boundary conditions
def getDBC(x,flag):
    pass

dirichletConditions = {0:getDBC}

def setNoZeroLSweakDirichletBCs(RDLSvt):
    assert hasattr(RDLSvt, 'freezeLevelSet')
    assert hasattr(RDLSvt, 'u_dof_last')
    assert hasattr(RDLSvt, 'weakDirichletConditionFlags')
    assert hasattr(RDLSvt.coefficients, 'epsFact')
    assert hasattr(RDLSvt, 'dofFlag_element')
    RDLSvt.freezeLevelSet = 0

if not useExact:
    if LevelModelType == RDLS.LevelModel:
        weakDirichletConditions = {0:RDLS.setZeroLSweakDirichletBCsSimple}
    else:
        weakDirichletConditions = {0:coefficients.setZeroLSweakDirichletBCs}
else:
    weakDirichletConditions = {0:setNoZeroLSweakDirichletBCs}

#weakDirichletConditions = {0:coefficients.setZeroLSweakDirichletBCs2}
#weakDirichletConditions = None

initialConditions  = ncls_p.initialConditions
fluxBoundaryConditions = {0:'noFlow'}
advectiveFluxBoundaryConditions =  {}
diffusiveFluxBoundaryConditions = {0:{}}
