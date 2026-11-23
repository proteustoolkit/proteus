from proteus import *
from proteus.default_p import *
from math import *
try:
    from .vortex2D import *
    from . import ls_vortex_2d_p
except:
    from vortex2D import *
    import ls_vortex_2d_p
    
from proteus.kernels import RDLS
name = soname+"_rdls"
LevelModelType = RDLS.LevelModel

coefficients = RDLS.Coefficients(applyRedistancing=applyRedistancing,
                                 epsFact=epsFactRedistance,
                                 nModelId=0,
                                 rdModelId=1,
                                 useMetrics=useMetrics,
                                 backgroundDiffusionFactor=0.01,
                                 weakDirichletFactor=1.0e3,
                                 useExact=useExact)

#now define the Dirichlet boundary conditions

def getDBC(x,flag):
    pass

dirichletConditions = {0:getDBC}

if True:#not useExact:
    if LevelModelType == RDLS.LevelModel:
        #weakDirichletConditions = {0:RDLS.setZeroLSweakDirichletBCs}
        weakDirichletConditions = {0:RDLS.setZeroLSweakDirichletBCsSimple}
    else:
        weakDirichletConditions = {0:coefficients.setZeroLSweakDirichletBCs}
else:
    def setNoZeroLSweakDirichletBCs(RDLSvt):
        assert hasattr(RDLSvt, 'freezeLevelSet')
        assert hasattr(RDLSvt, 'u_dof_last')
        assert hasattr(RDLSvt, 'weakDirichletConditionFlags')
        assert hasattr(RDLSvt.coefficients, 'epsFact')
        assert hasattr(RDLSvt, 'dofFlag_element')
        RDLSvt.freezeLevelSet = 0
    weakDirichletConditions = {0: setNoZeroLSweakDirichletBCs}


initialConditions  = ls_vortex_2d_p.initialConditions

fluxBoundaryConditions = {0:'noFlow'}

advectiveFluxBoundaryConditions =  {}

diffusiveFluxBoundaryConditions = {0:{}}
