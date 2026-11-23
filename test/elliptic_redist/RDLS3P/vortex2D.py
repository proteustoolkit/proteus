from proteus import Domain
from proteus import Context
from proteus.kernels import NCLS
import numpy as np

ct=Context.Options([
    # General parameters #
    ("T",0.01,"Final time"),
    ("nDTout",1,"Number of time steps to archive"),
    ("refinement",1,"Level of refinement"),
    ("unstructured",False,"Use unstructured mesh. Set to false for periodic BCs"),
    # parameters for elliptic re-distancing    
    ("ELLIPTIC_REDISTANCING",0, "Type of elliptic re-distancing"),
],mutable=True)

# ELLIPTIC_REDISTANCING #
#0: no elliptic re-distancing
#1: nolinear via single or double potential, with(out) |grad(u)| reconstructed
#2: linear via C0 normal reconstruction
#3: nolninear via C0 normal reconstruction
runCFL=0.33

#number of space dimensions
nd=2

# General parameters 
parallel = True
linearSmoother = None
checkMass=False

# Finite element space
pDegree_ls=1 
useBernstein=True
useHex=False#True

# quadrature order
vortex_quad_order = 2*pDegree_ls+1

#parallel partitioning info
from proteus import MeshTools
partitioningType = MeshTools.MeshParallelPartitioningTypes.node

# create mesh #
nn=nnx=nny=(2**ct.refinement)*10+1
nnz=1
he=1.0/(nnx-1.0)

unstructured=ct.unstructured #True for tetgen, false for tet or hex from rectangular grid
box=Domain.RectangularDomain(L=(1.0,1.0),
                             x=(0.0,0.0),
                             name="box");
box.writePoly("box")
if unstructured:
    domain=Domain.PlanarStraightLineGraphDomain(fileprefix="box")
    domain.boundaryTags = box.boundaryTags
    bt = domain.boundaryTags
    domain.MeshOptions.triangleOptions="pAq30Dena%8.8f"  % (0.5*he**2,)
else:
    domain = box
domain.MeshOptions.nn = nn
domain.MeshOptions.nnx = nnx
domain.MeshOptions.nny = nny
domain.MeshOptions.nnz = nnz
domain.MeshOptions.triangleFlag=0

epsFactHeaviside=epsFactDirac=1.5
epsFactRedistance=0.33

useMetrics=0.0
if useMetrics:
    shockCapturingFactor_ls=0.5
    shockCapturingFactor_rd=0.5
    lag_shockCapturing_ls=True
    lag_shockCapturing_rd=False
else:
    shockCapturingFactor_ls=0.0
    shockCapturingFactor_rd=0.9
    lag_shockCapturing_ls=True
    lag_shockCapturing_rd=False

#use absolute tolerances on al models
atolRedistance = max(1.0e-12,0.1*he)
atolLevelSet     = max(1.0e-12,0.001*he**2)

linearSolverConvergenceTest = 'r-true' 
fmmFlag=0

soname="vortex_c0p"+repr(pDegree_ls)+"_level_"+repr(ct.refinement)

class MyCoefficients(NCLS.Coefficients):
    def attachModels(self,modelList):
        self.model = modelList[0]
        self.q_v = np.zeros(self.model.q[('dH',0,0)].shape,'d')
        self.ebqe_v = np.zeros(self.model.ebqe[('dH',0,0)].shape,'d')
        self.rdModel = self.model
        self.ebqe_rd_u = self.rdModel.ebqe[('u',0)]
