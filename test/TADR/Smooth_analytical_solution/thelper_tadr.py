from proteus import Domain
from proteus import Norms
from proteus import Profiling
from proteus import Context
from proteus.kernels import TADR
import numpy as np
import math

ct=Context.Options([
    # General parameters #
    ("physicalDiffusion", 0.0, "isotropic diffusion coefficient"),
    ("nd",1,"space dimension (benchmark is 1D)"),
    ("T",1.0,"Final time"),
    ("nDTout",1000,"Number of time steps to archive"),
    ("refinement",0,"Level of refinement"),
    ("unstructured",False,"Use unstructured mesh"),
    # Analytical benchmark parameters (smooth Gaussian pulse) #
    ("C0",1.0,"Initial Gaussian peak concentration"),
    ("v",0.3,"Constant advection velocity"),
    ("D",0.007,"Constant diffusion coefficient"),
    ("x0",0.5,"Initial Gaussian center"),
    ("sigma0",0.08,"Initial Gaussian standard deviation"),
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
# SHOCK CAPTURING PARAMETERS #
shockCapturingFactor_tadr=0.2
lag_shockCapturing_tadr=True

# number of space dimensions #
nd = ct.nd
assert nd == 1, "Step_analytical_solution benchmark is configured for nd=1 only."

# General parameters #
parallel = True #False # if True use PETSc solvers
linearSmoother = None
checkMass = False

# Finite element sapce #
pDegree_tadr=1
useBernstein=False
useHex=False

# quadrature order #
tadr_quad_order = 2*pDegree_tadr+1

# parallel partitioning info #
from proteus import MeshTools
partitioningType = MeshTools.MeshParallelPartitioningTypes.node

# create mesh #
nn = nnx = (2**ct.refinement)*10 + 1
nny = 1
nnz = 1
he = 2.0/(nnx - 1.0)

box = Domain.RectangularDomain(L=(2.0,),
                               x=(0.0,),
                               name="box")
domain = box

domain.MeshOptions.nn = nn
domain.MeshOptions.nnx = nnx
domain.MeshOptions.nny = nny
domain.MeshOptions.nnz = nnz
domain.MeshOptions.triangleFlag=0

soname = "tadr_level_" + repr(ct.refinement)

class MyCoefficients(TADR.Coefficients):
    def attachModels(self,modelList):
        self.model = modelList[0]
        self.rdModel = self.model
        self.q_v = np.zeros(self.model.q[('grad(u)',0)].shape,'d')
        self.ebqe_v = np.zeros(self.model.ebqe[('grad(u)',0)].shape,'d')
        self.ebqe_phi = np.zeros(self.model.ebqe[('u',0)].shape,'d')#cek hack, we don't need this 
        self.q_theta = np.ones(self.model.q[('u',0)].shape,'d')
        self.ebqe_theta = np.ones(self.model.ebqe[('u',0)].shape,'d')
        self.q_v_old = self.q_v.copy()
        self.q_theta_old = self.q_theta.copy()
        self.flowCoefficients = None
        
