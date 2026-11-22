from proteus import Domain
from proteus import Norms
from proteus import Profiling
from proteus import Context
from proteus.kernels import CLSVOF
import numpy as np
import math
import os
try:
    from .parameters import *
except:
    from parameters import *

AUTOMATED_TEST=True

#----- TEST CASE ----- #
# 1: 2D periodic vortex
# 2: 2D solid rotation
# 3: 3D solid rotation
# 4: 3D LeVeque test
# see parameters.py

# ----- PARAMETERS FOR CLSVOF ----- #
doSpinUpStep=False
lambdaFact=1.0
computeMetrics=0 #0: don't compute metrics, 1: compute at EOS, 2: compute at ETS
eps_tolerance_clsvof=True #Set tol on nonlinear solver to machine zero?
#clsvof_nl_atol_res # see below

# ----- REFINEMENT ----- #
triangleFlag=1
unstructured=False
refinement=3 #default: 3

# ----- NUMERICAL PARAMETERS ----- #
cfl=0.33
useMetrics=0

# ----- number of space dimensions ----- #
nd=2
if ct.test_case>2:
    nd=3

if ct.test_case==1:
    T=8.0
    nDTout=80
elif ct.test_case==2 or ct.test_case==3:
    T=1.0
    nDTout=10
else:
    T=3.0
    nDTout=30

refinementMultiplier=12.5
if AUTOMATED_TEST==True:
    refinementMultiplier=3#5
    refinement=1
    T=0.1
    nDTout=1

if ct.test_case==3:
    refinementMultiplier=4#5


# ----- General parameters ----- #
parallel = False # if True use PETSc solvers
linearSmoother = None

# ----- Finite element sapce ----- #
pDegree_clsvof=1
useBernstein=True
useHex=False

# ----- quadrature order ----- #
clsvof_quad_order = 2*pDegree_clsvof+1

# parallel partitioning info #
from proteus import MeshTools
partitioningType = MeshTools.MeshParallelPartitioningTypes.node

# create mesh #
if nd==2:
    nn=nnx=nny=(2**refinement)*10+1
    nnz=1
    L=[1.0,1.0]
else:
    if ct.test_case==3:
        nn=nnx=nny=int((2**refinement)*refinementMultiplier+1)
        nnz=int((nnx-1)//2+1)
        L=[1.0,1.0,0.5]
    else:
        nn=nnx=nny=nnz=int((2**refinement)*refinementMultiplier+1)
        L=[1.0,1.0,1.0]
# definition of he
he=1.0/(nnx-1.0)
clsvof_nl_atol_res = max(1.0e-10, 0.01 * he ** 2)

unstructured=unstructured #True for tetgen, false for tet or hex from rectangular grid
box=Domain.RectangularDomain(L)
#box.writePoly("box")
if unstructured:
    domain=Domain.PlanarStraightLineGraphDomain(fileprefix="box")
    domain.boundaryTags = box.boundaryTags
    bt = domain.boundaryTags
    triangleOptions="pAq30Dena%8.8f"  % (0.5*he**2,)
else:
    domain = box
    meshname = os.path.dirname(os.path.abspath(__file__))+"/"+"mesh_"+str(ct.test_case)
    #domain.writePoly(meshname)
    domain.polyfile = meshname
    domain.MeshOptions.genMesh=False

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
