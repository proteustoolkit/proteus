from proteus.default_p import *
from proteus.kernels import RDLS
from proteus import Context

ct = Context.get()
domain = ct.domain
nd = domain.nd
mesh = domain.MeshOptions


genMesh = mesh.genMesh
movingDomain = ct.movingDomain
T = ct.T

"""
The redistancing equation in the sloshbox test problem.
"""

LevelModelType = RDLS.LevelModel

coefficients = RDLS.Coefficients(applyRedistancing=ct.applyRedistancing,
                                 epsFact=ct.epsFact_redistance,
                                 nModelId=int(ct.movingDomain)+2,
                                 rdModelId=int(ct.movingDomain)+3,
                                 useMetrics=ct.useMetrics,
                                 backgroundDiffusionFactor=ct.backgroundDiffusionFactor)

def getDBC_rd(x, flag):
    pass

dirichletConditions     = {0: getDBC_rd}
weakDirichletConditions = {0: RDLS.setZeroLSweakDirichletBCsSimple}

advectiveFluxBoundaryConditions = {}
diffusiveFluxBoundaryConditions = {0: {}}

class PerturbedSurface_phi:
    def uOfXT(self,x,t):
        return ct.signedDistance(x)

initialConditions  = {0:PerturbedSurface_phi()}