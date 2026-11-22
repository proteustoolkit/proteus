from proteus import Domain
from proteus import Norms
from proteus import Profiling
from proteus import Context
from proteus.kernels import CLSVOF
import numpy as np
import math
try:
    from .parameters import *
except:
    from parameters import *

#----- TEST CASE ----- #
# 1: CIRCLE
# 2: ZALESAK DISK

# ----- PARAMETERS FOR CLSVOF ----- #
doSpinUpStep=False
lambdaFact=1.0
computeMetrics=0 
eps_tolerance_clsvof=True

# ----- REFINEMENT ----- #
triangleFlag=1
unstructured=False
refinement=2

# ----- NUMERICAL PARAMETERS ----- #
cfl=0.33
useMetrics=0

# ----- number of space dimensions ----- #
nd=2
T=1.0
nDTout=1

# ----- General parameters ----- #
parallel = False
linearSmoother = None

# ----- Finite element sapce ----- #
pDegree_clsvof=1
useBernstein=False
useHex=False

# ----- quadrature order ----- #
clsvof_quad_order = 2*pDegree_clsvof+1

# parallel partitioning info #
from proteus import MeshTools
partitioningType = MeshTools.MeshParallelPartitioningTypes.node

# create mesh #
nn=nnx=nny=(2**refinement)*10+1
nnz=1
L=[1.0,1.0]
# definition of he
he=1.0/(nnx-1.0)
clsvof_nl_atol_res = 1.0e-10#max(1.0e-10, 0.01 * he ** 2)

unstructured=unstructured #True for tetgen, false for tet or hex from rectangular grid
box=Domain.RectangularDomain(L)
box.writePoly("box")
if unstructured:
    domain=Domain.PlanarStraightLineGraphDomain(fileprefix="box")
    domain.boundaryTags = box.boundaryTags
    bt = domain.boundaryTags
    triangleOptions="pAq30Dena%8.8f"  % (0.5*he**2,)
else:
    domain = box

domain.MeshOptions.nn = domain.MeshOptions.nnx = domain.MeshOptions.nny = nn
domain.MeshOptions.nnz = nnz

soname="clsvof_level_"+repr(refinement)

class MyCoefficients(CLSVOF.Coefficients):
    def attachModels(self,modelList):
        self.model = modelList[0]
        self.q_v = np.zeros(self.model.q[('grad(u)',0)].shape,'d')
        self.ebqe_v = np.zeros(self.model.ebqe[('grad(u)',0)].shape,'d')
        self.q_v_old = np.zeros(self.model.q[('grad(u)',0)].shape,'d')
        self.q_v_tStar = np.zeros(self.model.q[('grad(u)',0)].shape,'d')
