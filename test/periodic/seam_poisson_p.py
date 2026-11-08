"""C0 P1 periodic advection-diffusion on the unit square: u = sin(2 pi x) sin(pi y), periodic in x, u = 0 at y = 0, 1.

eps (-Laplace u) + bx du/dx = f, with the numerical flux named by SEAM_FLUX (Advection_DiagonalUpwind_Diffusion_IIPG_
exterior by default). SEAM_PERIODIC=0 replaces the periodic pair by Dirichlet conditions at x = 0, 1, the reference
the periodic run is compared with. SEAM_PARALLEL=1 uses the parallel periodic numbering (element partition without
overlap), which every run on more than one rank needs. Read by seam_poisson_mpi.py.
"""
import os
import numpy as np
from proteus.default_p import *
from proteus.TransportCoefficients import TC_base
from proteus import Domain

nd = 2
L = (1.0, 1.0, 1.0)
name = "seam_poisson"
NN = int(os.environ.get("SEAM_NN", "9"))
EPS = float(os.environ.get("SEAM_EPS", "1.0"))
BX = float(os.environ.get("SEAM_BX", "0.0"))
PERIODIC = os.environ.get("SEAM_PERIODIC", "1") == "1"
PARALLEL = os.environ.get("SEAM_PARALLEL", "0") == "1"

domain = Domain.RectangularDomain(L=L[:2])
domain.MeshOptions.nnx = domain.MeshOptions.nny = NN
domain.MeshOptions.triangleFlag = 0
if PARALLEL:
    domain.MeshOptions.setParallelPartitioningType('element')
    domain.MeshOptions.nLayersOfOverlapForParallel = 0


def exact(x):
    return np.sin(2*np.pi*x[0])*np.sin(np.pi*x[1])


def source(x):
    return (EPS*5*np.pi**2*np.sin(2*np.pi*x[0])*np.sin(np.pi*x[1])
            + BX*2*np.pi*np.cos(2*np.pi*x[0])*np.sin(np.pi*x[1]))


class ADR(TC_base):
    def __init__(self):
        TC_base.__init__(self, nc=1, mass={}, advection={0: {0: 'linear'}}, diffusion={0: {0: {0: 'constant'}}},
                         potential={0: {0: 'u'}}, reaction={0: {0: 'constant'}}, hamiltonian={},
                         useSparseDiffusion=False)

    def evaluate(self, t, c):
        u = c[('u', 0)]
        c[('f', 0)][..., 0] = BX*u
        c[('f', 0)][..., 1] = 0.0
        c[('df', 0, 0)][..., 0] = BX
        c[('df', 0, 0)][..., 1] = 0.0
        a = c[('a', 0, 0)]
        a[...] = 0.0
        a[..., 0] = EPS
        a[..., 3] = EPS
        c[('r', 0)][...] = -source(np.moveaxis(c['x'], -1, 0))
        c[('dr', 0, 0)][...] = 0.0


coefficients = ADR()
eps = 1e-8


def getDBC(x, flag):
    if x[1] < eps or x[1] > 1 - eps or (not PERIODIC and (x[0] < eps or x[0] > 1 - eps)):
        return lambda x, t: 0.0


dirichletConditions = {0: getDBC}


def getPDBC(x, flag):
    if x[0] < eps or x[0] > 1 - eps:
        return np.array([0.0, round(x[1], 8), 0.0])


periodicDirichletConditions = {0: getPDBC} if PERIODIC else None
advectiveFluxBoundaryConditions = {}
diffusiveFluxBoundaryConditions = {0: {}}
