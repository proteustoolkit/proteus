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
    ("problem",0,"0: 1D problem with periodic BCs. 1: 2D rotation of zalesak disk"),
    ("nd",2,"space dimension"),
    ("T",0.1,"Final time"),
    ("nDTout",1,"Number of time steps to archive"),
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

assert ct.problem==0 or ct.problem==1 or ct.problem == 2, "problem must be set to 0 or 1"
# SHOCK CAPTURING PARAMETERS #
shockCapturingFactor_tadr=0.2
lag_shockCapturing_tadr=True

# number of space dimensions #
nd=ct.nd

# General parameters #
parallel = False # if True use PETSc solvers
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
nn=nnx= 21 #(2**ct.refinement)*10+1
#nny=1
if nd == 2:
    #nny=(nnx-1)//10+1 if ct.problem==0 else nnx
    nny=nnx
nnz=1
he=1.0/(nnx-1.0)

unstructured=ct.unstructured #True for tetgen, false for tet or hex from rectangular grid
if nd == 1:
    box=Domain.RectangularDomain(L=(2.0,),
                                 x=(0.0,),
                                 name="box");
    domain = box
elif nd == 2:
    domain = Domain.PlanarStraightLineGraphDomain()
    boundaries = [f'marker_{i}' for i in range(1, 17)]
    boundaryTags=dict([(key,i+1) for (i,key) in enumerate(boundaries)])
    file_path = 'points_array_try.txt'  # Replace with your actual file path
    with open(file_path, 'r') as file:
        lines = file.readlines()
    selected_values = [[float(line.split(',')[0]), float(line.split(',')[2])] for line in lines]
    # Create a NumPy array from the selected values
    filtered_points = np.array(selected_values)
    vertices= filtered_points.tolist()
    vertexFlags = []
    marker= np.loadtxt('vertexflags.txt', dtype=int).tolist()
    for value in marker:
        formatted_value = f"marker_{value}"
        vertexFlags.append(boundaryTags[formatted_value])

    segments=np.loadtxt('segments.txt', dtype=int).tolist()

    segmentFlags = []
    flags_segments= np.loadtxt('segmentsFlags.txt', dtype=int).tolist()

    for value in flags_segments:
        marker_number= int(value)
        formatted_value = f"marker_{marker_number}"
        segmentFlags.append(boundaryTags[formatted_value])
    regions= [[0.8,1.4], #1
          [0.8,1.15], #2
           [0.6,1.0],[1.2,0.9],[1.7,1.0], #3
           [0.4,0.9],[1.1,0.8],[2.2,0.95], #4
           [0.5,0.85],[1.2,0.75],[1.7,0.9], #5
           [0.42,0.75],[1.20,0.65],[1.62,0.75], #6
           [2.61,0.70], #7
           [0.18,0.55],[0.60,0.60], #8
           [0.30,0.65], #9
           [0.21,0.60],[0.60,0.65], #10
           [0.15,0.35],[0.60,0.45], #11
           [0.15,0.20],[0.81,0.25], #12
           [1.83,0.05], #13
           [0.36,0.25], #14
           [0.96,0.90], #15
           [1.44,0.85], #16 
           ]
#regionFlags=[1,2,3]
    regionFlags=[1,
            2,
             3,3,3,
             4,4,4,
             5,5,5,
             6,6,6,
             7,
             8,8,
             9,
             10,10,
             11,11,
             12,12,
             13,
             14,
             15,16]


    domain = Domain.PlanarStraightLineGraphDomain(vertices= vertices,
                                              vertexFlags=vertexFlags,
                                              segments=segments,
                                              segmentFlags=segmentFlags,
                                              regions = regions,
                                              regionFlags = regionFlags,)
                                               #holes= holes)
    domain.writePoly('CCS_rescaled')
    G=[2.799999952316284,1.4950000047683716]
    triangleOptions="pAnea%f"% (0.5*(G[0]/(nnx-1))**2,)
#    triangleOptions="pAq30Dena%8.8f"  % 0.001

    # box=Domain.RectangularDomain(L=(1.0,1.0 if ct.problem==0 else 1.0),
    #                              x=(0.0,0.0),
    #                              name="box");
    # box.writePoly("box")
    
    # if unstructured:
    #     domain=Domain.PlanarStraightLineGraphDomain(fileprefix="box")
    #     domain.boundaryTags = box.boundaryTags
    #     bt = domain.boundaryTags
    #     triangleOptions="pAq30Dena%8.8f"  % (0.5*he**2,)
    # else:
    #     domain = box
# import pdb
# pdb.set_trace()

domain.MeshOptions.nn = nn
domain.MeshOptions.nnx = nnx
domain.MeshOptions.nny = nny
domain.MeshOptions.nnz = nnz
domain.MeshOptions.triangleFlag=0

soname="tadr_level_"+repr(ct.refinement)

class MyCoefficients(TADR.Coefficients):
    def attachModels(self,modelList):
        self.model = modelList[0]
        self.rdModel = self.model
        self.q_v = np.zeros(self.model.q[('grad(u)',0)].shape,'d')
        self.ebqe_v = np.zeros(self.model.ebqe[('grad(u)',0)].shape,'d')
    
