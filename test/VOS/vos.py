from proteus import Domain
from proteus import Norms
from proteus import Profiling 
from proteus import Context 
from proteus.kernels import VOS3P
import numpy as np
import math 

ct=Context.Options([
    # General parameters #
    ("problem",0,"0: 1D advection, 1: implosion with circle, 2: implosion with ring"),
    ("T",0.2,"Final time"),
    ("nDTout",10,"Number of time steps to archive"),
    ("refinement",3,"Level of refinement"),
    # Choice of numerical method #
    ("STABILIZATION_TYPE",2,"0: SUPG, 1: EV, 2: smoothness based indicator"),
    ("LUMPED_MASS_MATRIX",True,"Flag to lumped the mass matrix"),
    # Numerical parameters #
    ("cfl",0.1,"Target cfl"),
    ("SSPOrder",1,"SSP method of order 1, 2 or 3")
],mutable=True)

if ct.problem==0:
    ct.T=1.0
    ct.nDTout=20
elif ct.problem==1:    
    ct.T=0.40
    ct.nDTout=40
else:    
    ct.T=1.0
    ct.nDTout=100
#
# SHOCK CAPTURING PARAMETERS #
shockCapturingFactor_vos=0.2
lag_shockCapturing_vos=True

# number of space dimensions #
nd=2

# General parameters #
parallel = False # if True use PETSc solvers
linearSmoother = None
checkMass = False

# Finite element sapce #
pDegree_vos=1
useBernstein=False
useHex=False

# quadrature order #
vos_quad_order = 2*pDegree_vos+1

# parallel partitioning info #
from proteus import MeshTools
partitioningType = MeshTools.MeshParallelPartitioningTypes.node

# create mesh #
nn=nnx=(2**ct.refinement)*10+1
nny=nnx
if ct.problem==0:
    nny=int((nnx-1)/10+1)
nnz=1
he=1.0/(nnx-1.0)

unstructured=False
box=Domain.RectangularDomain(L=(1.0,0.1 if ct.problem==0 else 1.0),
                             x=(0.0,0.0),
                             name="box");
box.writePoly("box")
if unstructured:
    domain=Domain.PlanarStraightLineGraphDomain(fileprefix="box")
    domain.boundaryTags = box.boundaryTags
    bt = domain.boundaryTags
    triangleOptions="pAq30Dena%8.8f"  % (0.5*he**2,)
else:
    domain = box
soname="vos_level_"+repr(ct.refinement)

class MyCoefficients(VOS3P.Coefficients):
    def attachModels(self,modelList):
        self.model = modelList[0]
        self.rdModel = self.model
        self.q_v = np.zeros(self.model.q[('grad(u)',0)].shape,'d')
        self.ebqe_v = np.zeros(self.model.ebqe[('grad(u)',0)].shape,'d')
        self.ebqe_phi = np.zeros(self.model.ebqe[('u',0)].shape,'d')#cek hack, we don't need this 

        if self.model.hasVelocityFieldAsFunction:
            self.model.updateVelocityFieldAsFunction()
