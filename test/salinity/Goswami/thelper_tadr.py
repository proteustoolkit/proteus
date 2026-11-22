from proteus import Domain
from proteus import Norms
from proteus import Profiling 
from proteus import Context 
from proteus.kernels import TADR
import numpy as np
import math 
from proteus.Profiling import logEvent
#T =4800.0 #100.0 #12000 # 6.0e2*6..,1.0e4 #time scale, s
#nDTout =10 #0  # output timesteps to use
ct=Context.Options([
    # General parameters #
    ("physicalDiffusion", 0.0, "isotropic diffusion coefficient"),
    #("problem",0,"0: 1D problem with periodic BCs. 1: 2D rotation of zalesak disk"),
    #("nd",2,"space dimension"),
    ("T",100,"Final time"),
    #("nDTout",100,"Number of time steps to archive"),
    ("refinement",0,"Level of refinement"),
    ("unstructured",False,"Use unstructured mesh. Set to false for periodic BCs"),
    # Choice of numerical method #
    ("STABILIZATION_TYPE","Galerkin","Galerkin, VMS, TaylorGalerkinEV, EntropyViscosity, SmoothnessIndicator, Kuzmin, "),
    ("LUMPED_MASS_MATRIX",False,"Flag to lumped the mass matrix"),
    ("ENTROPY_TYPE","POWER","POWER, LOG"),
    ("FCT",True,"Use Flux Corrected Transport"),
    # Numerical parameters #
    ("cE",0.1,"Entropy viscosity constant"),
    ("cK",1.0,"Artificial compression constant"),
    ("cfl",0.5,"Target cfl"),
    ("SSPOrder",2,"SSP method of order 1, 2 or 3")
],mutable=True)

#assert ct.problem==0 or ct.problem==1 or ct.problem == 2, "problem must be set to 0 or 1"
# SHOCK CAPTURING PARAMETERS #
shockCapturingFactor_tadr=0.2
lag_shockCapturing_tadr=True

# number of space dimensions #
#nd=ct.nd

# General parameters #
parallel = True # if True use PETSc solvers
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
L=(.530,.26,1)#.26 
nd = 2

nnx =107#213#107
nny =53#105#53#53# 105

unstructured=ct.unstructured #True for tetgen, false for tet or hex from rectangular grid


domain = Domain.RectangularDomain(L=[L[0],L[1]],name="gwvd_gc_domain",units="m")

polyfile="gwvd_gc_domain_2d"
domain.writePoly(polyfile)
triangleOptions = "q30Dena0.00001"


#domain.MeshOptions.nn = nn
domain.MeshOptions.nnx = nnx
domain.MeshOptions.nny = nny
#domain.MeshOptions.nnz = nnz
domain.MeshOptions.triangleFlag=0

soname="tadr_level_"+repr(ct.refinement)

    
# class MyCoefficients(TADR.Coefficients):
#     def attachModels(self,modelList):
#         self.model = modelList[0]
#         self.rdModel = self.model
#         self.q_v = np.zeros(self.model.q[('grad(u)',0)].shape,'d')
#         self.ebqe_v = np.zeros(self.model.ebqe[('grad(u)',0)].shape,'d')

# class MyCoefficients(TADR.Coefficients):
#     def attachModels(self,modelList):
#         self.model = modelList[0]
#         self.rdModel = self.model
#         self.q_v = np.copy(self.model.q[('grad(u)', 0)])  # Copy element-level velocity
#         self.ebqe_v = np.copy(self.model.ebqe[('grad(u)', 0)])
#         if 'dV' not in self.model.q:
#             self.model.q['dV'] = np.ones(self.model.q[('u', 0)].shape, 'd')
#         # Ensure 'dV_last' exists in the Richards model quadrature
#         if 'dV_last' not in self.model.q:
#             self.model.q['dV_last'] = np.zeros_like(self.model.q['dV'])
#         logEvent(f"Richards velocity (q_v): mean={self.q_v.mean()}, min={self.q_v.min()}, max={self.q_v.max()}")
#         logEvent(f"Richards boundary velocity (ebqe_v): mean={self.ebqe_v.mean()}, min={self.ebqe_v.min()}, max={self.ebqe_v.max()}")

        # logEvent("Richards velocity (q_v):", self.q_v)
        # logEvent("Richards boundary velocity (ebqe_v):", self.ebqe_v)
        # #self.q_v = np.zeros(self.model.q[('grad(u)',0)].shape,'d')
        #self.ebqe_v = np.zeros(self.model.ebqe[('grad(u)',0)].shape,'d')
class MyCoefficients(TADR.Coefficients):
    def attachModels(self, modelList):
        """
        Attach the Richards model to TADR and pass the velocity field (grad(u)).
        """
        # Attach Richards model
        self.model = modelList[0]  # Richards model
        self.rdModel = self.model  # Alias for convenience
            # Log modelList for inspection
        logEvent(f"Model list contains {len(modelList)} models:")
        for i, model in enumerate(modelList):
            logEvent(f"Model {i}: {model.name}, Type: {type(model)}")

        # Reference velocity fields directly from Richards model
        if ('grad(u_v)', 0) in self.model.q:
            self.q_v = self.model.q[('grad(u_v)', 0)]  # Reference gradient of u (velocity)
            self.q_v *= 1.241e-9
        else:
            raise KeyError("Richards model q[('grad(u_v)', 0)] not found!")

        if ('grad(u)', 0) in self.model.ebqe:
            self.ebqe_v = self.model.ebqe[('grad(u)', 0)]  # Reference boundary gradient of u (velocity)
        else:
            raise KeyError("Richards model ebqe[('grad(u)', 0)] not found!")

        # Ensure quadrature volumes exist
        if 'dV' not in self.model.q:
            self.model.q['dV'] = np.ones(self.model.q[('u', 0)].shape, 'd')

        if 'dV_last' not in self.model.q:
            self.model.q['dV_last'] = np.zeros_like(self.model.q['dV'])

        # Log for debugging
        logEvent(f"Richards velocity (q_v): mean={self.q_v.mean()}, min={self.q_v.min()}, max={self.q_v.max()}")
        logEvent(f"Richards boundary velocity (ebqe_v): mean={self.ebqe_v.mean()}, min={self.ebqe_v.min()}, max={self.ebqe_v.max()}")