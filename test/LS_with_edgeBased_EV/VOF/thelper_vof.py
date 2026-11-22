from proteus import Domain
from proteus import Norms
from proteus import Profiling 
from proteus import Context 
from proteus.kernels import VOF
import numpy as np
import math 

ct=Context.Options([
    # General parameters #
    ("problem",0,"0: 1D problem with periodic BCs. 1: 2D rotation of zalesak disk"),
    ("nd",2,"space dimension"),
    ("T",0.1,"Final time"),
    ("nDTout",1,"Number of time steps to archive"),
    ("refinement",0,"Level of refinement"),
    ("unstructured",False,"Use unstructured mesh. Set to false for periodic BCs"),
    # Choice of numerical method #
    ("STABILIZATION_TYPE",1,"0: SUPG, 1: EV, 2: smoothness based indicator"),
    ("LUMPED_MASS_MATRIX",False,"Flag to lumped the mass matrix"),
    ("ENTROPY_TYPE",1,"1: quadratic, 2: logarithmic"),
    ("FCT",True,"Use Flux Corrected Transport"),
    # Numerical parameters #
    ("cE",0.1,"Entropy viscosity constant"),
    ("cK",1.0,"Artificial compression constant"),
    ("cfl",0.5,"Target cfl"),
    ("SSPOrder",2,"SSP method of order 1, 2 or 3")
],mutable=True)

assert ct.problem==0 or ct.problem==1 or ct.problem == 2, "problem must be set to 0 or 1"
# SHOCK CAPTURING PARAMETERS #
shockCapturingFactor_vof=0.2
lag_shockCapturing_vof=True

# number of space dimensions #
nd=ct.nd

# General parameters #
parallel = False # if True use PETSc solvers
linearSmoother = None
checkMass = False

# Finite element sapce #
pDegree_vof=1
useBernstein=False
useHex=False

# quadrature order #
vof_quad_order = 2*pDegree_vof+1

# parallel partitioning info #
from proteus import MeshTools
partitioningType = MeshTools.MeshParallelPartitioningTypes.node

# create mesh #
nn=nnx=(2**ct.refinement)*10+1
nny=1
if nd == 2:
    nny=(nnx-1)//10+1 if ct.problem==0 else nnx
nnz=1
he=1.0/(nnx-1.0)

unstructured=ct.unstructured #True for tetgen, false for tet or hex from rectangular grid
if nd == 1:
    box=Domain.RectangularDomain(L=(2.0,),
                                 x=(0.0,),
                                 name="box");
    domain = box
elif nd == 2:
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

domain.MeshOptions.nn = nn
domain.MeshOptions.nnx = nnx
domain.MeshOptions.nny = nny
domain.MeshOptions.nnz = nnz
domain.MeshOptions.triangleFlag=0

soname="vof_level_"+repr(ct.refinement)

class MyCoefficients(VOF.Coefficients):
    def attachModels(self,modelList):
        self.model = modelList[0]
        self.rdModel = self.model
        self.q_v = np.zeros(self.model.q[('grad(u)',0)].shape,'d')
        self.ebqe_v = np.zeros(self.model.ebqe[('grad(u)',0)].shape,'d')
        self.ebqe_phi = np.zeros(self.model.ebqe[('u',0)].shape,'d')#cek hack, we don't need this
        class MyFlowCoefficients(VOF.Coefficients):
            def __init__(self,model):
                self.model=model
                self.ghost_penalty_constant=0.0
                self.phi_s = np.ones(self.model.mesh.nodeArray.shape[0], 'd')*1e10#
                self.useExact = False
                self.ebqe_phi_s = np.ones((self.model.ebqe['x'].shape[0],self.model.ebqe['x'].shape[1]),'d') * 1e10
                self.q_phi_solid = np.ones(self.model.q[('u', 0)].shape, 'd') * 1e10
        self.flowCoefficients = MyFlowCoefficients(self.model)