from proteus import Domain
from proteus import Context
from proteus.kernels import NCLS
from proteus import Norms
from proteus import Profiling 
import numpy as np
import math 

ct=Context.Options([
    # General parameters #
    ("problem",0,"0: 1D advection with PBCs. 1: 2D rotation of disk"),
    ("level_set_function",1,"0: distance function, 1: saturated distance function"),
    ("T",0.1,"Final time"),
    ("nDTout",1,"Number of time steps to archive"),
    ("refinement",3,"Level of refinement"),
    ("unstructured",False,"Use unstructured mesh. Set to false for periodic BCs"),
    # Choice of numerical method #
    ("STABILIZATION_TYPE",1,"0: SUPG, 1: EV, 2: smoothness based indicator"),
    ("SATURATED_LEVEL_SET",True,"Saturate the distance function or not?"),
    ("ENTROPY_TYPE",2,"1: parabolic, 2: logarithmic"),
    ("COUPEZ",True,"Flag to turn on/off the penalization term in Coupez approach"), 
    ("DO_REDISTANCING",True,"Solve Eikonal type equation after transport?"),    
    # Numerical parameters #
    ("pure_redistancing",False,"To test pure redistancing; i.e., no advection"),
    ("cE",1.0,"Entropy viscosity constant"),
    ("cfl",0.5,"Target cfl"),
    ("SSPOrder",2,"SSP method of order 1, 2 or 3")
],mutable=True)

assert ct.problem==0 or ct.problem==1, "Problem must be either 0 or 1"
assert ct.level_set_function== 0 or ct.level_set_function==1, "Leve set function must be set to 0 or 1"

# NUMERICAL PARAMETERS #
epsCoupez=3
redist_tolerance=0.1
epsFactRedistance=0.33 #For signed dist function
lambda_coupez = 0.1 #Strength of redistancing and coupez force
epsFactHeaviside=epsFactDirac=1.5

if ct.level_set_function==0: #dist function
    assert ct.SATURATED_LEVEL_SET==False, "If level_set_function=0, use SATURATED_LEVEL_SET=False"
    assert ct.ENTROPY_TYPE==1, "If level_set_function=0, use ENTROPY_VISCOSITY=1"
else: #saturated distance function
    assert ct.SATURATED_LEVEL_SET==True, "If level_set_function=1, use SATURATED_LEVEL_SET=True"
    assert ct.ENTROPY_TYPE==2, "If level_set_function=1, use ENTROPY_VISCOSITY=2"

# SHOCK CAPTURING PARAMETERS #
shockCapturingFactor_ncls=0.2
lag_shockCapturing_ncls=True

# number of space dimensions #
nd=2

# General parameters #
parallel = False
linearSmoother = None
checkMass=False

# Finite element spaces #
pDegree_ncls=1
useBernstein=False
useHex=False

# Quadrature order # 
ncls_quad_order = 2*pDegree_ncls+1

# parallel partitioning info #
from proteus import MeshTools
partitioningType = MeshTools.MeshParallelPartitioningTypes.node

# create mesh # 
nn=nnx=(2**ct.refinement)*10+1
nny=(nnx-1)//10+1 if ct.problem==0 else nnx
nnz=1
he=1.0/(nnx-1.0)

unstructured=ct.unstructured #True for tetgen, false for tet or hex from rectangular grid
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

soname="ncls_level_"+repr(ct.refinement)

class MyCoefficients(NCLS.Coefficients):
    def attachModels(self,modelList):
        self.model = modelList[0]
        self.q_v = np.zeros(self.model.q[('dH',0,0)].shape,'d')
        self.ebqe_v = np.zeros(self.model.ebqe[('dH',0,0)].shape,'d')
        self.rdModel = self.model
        
