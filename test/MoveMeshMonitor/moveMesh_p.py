from proteus.default_p import *
from proteus import MoveMeshMonitor
import numpy as np
from proteus import Context

ct = Context.get()
domain = ct.domain
nd = ct.domain.nd
mesh = domain.MeshOptions


genMesh = mesh.genMesh
movingDomain = ct.movingDomain
T = ct.T

initialConditions = None

analyticalSolution = {}

coefficients = MoveMeshMonitor.Coefficients(ct.my_func,
                                            nd=ct.domain.nd,
                                            he_max=ct.he_max,
                                            he_min=ct.he_min,
                                            LS_MODEL=None,
                                            ME_MODEL=0,
                                            fixedNodeMaterialTypes=ct.fixedNodeMaterialTypes,
                                            fixedElementMaterialTypes=None,
                                            nSmoothOut=ct.nSmoothOut,
                                            epsTimeStep=ct.epsTimeStep,
                                            epsFact_density=0.,
                                            grading=0.,
                                            do_firstStep=True)

def getDBC(x,flag):
    if flag != 0:
        return None

dirichletConditions = {0:getDBC}

def getDFBC(x, flag):
    return lambda x, t: 0.

diffusiveFluxBoundaryConditions = {0:{0:getDFBC}}