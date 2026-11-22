from proteus import *
from proteus.default_p import *
"""
Heterogeneous Poisson's equation, -div(a(x)u) = f(x), on unit domain [0,1]x[0,1]x[0,1]
"""

from proteus.kernels import ADR

LevelModelType = ADR.LevelModel


##\page Tests Test Problems
# \ref poisson_3d_p.py "Heterogeneous Poisson's equation, -div(a(x)u) = f(x), on unit domain [0,1]x[0,1]x[0,1]"
#

##\ingroup test
#\file poisson_3d_p.py
#
#\brief Heterogenous Poisson's equations in 3D unit domain [0,1]x[0,1]x[0,1]

#----------------------------------------------------
# Domain - mesh - quadrature
#----------------------------------------------------
#space dimension
nd = 3

hull_length = 0.5
hull_beam   = 0.5
hull_draft  = 0.5

L=(2.0*hull_length,
   2.0*hull_beam, 
   2.0*hull_draft)

x_ll = (0.0,
        0.0,
        0.0)

hull_center = (0.5*hull_length,
               0.5*hull_beam,
               0.5*hull_draft)

nLevels = 1

he = L[0]/10.0
#he*=0.5
#he = hull_draft/1.0
#he = hull_draft/6.0
genMesh=True#False
vessel = None
#vessel = 'cube'
#vessel = 'wigley'
boundaryTags = { 'bottom': 1, 'front':2, 'right':3, 'back': 4, 'left':5, 'top':6, 'obstacle':7}
if vessel == 'wigley-gmsh':
    domain = Domain.MeshTetgenDomain(fileprefix="mesh")
    domain.boundaryTags = boundaryTags
else:
    vertices=[[x_ll[0],x_ll[1],x_ll[2]],#0
              [x_ll[0]+L[0],x_ll[1],x_ll[2]],#1
              [x_ll[0]+L[0],x_ll[1]+L[1],x_ll[2]],#2
              [x_ll[0],x_ll[1]+L[1],x_ll[2]],#3
              [x_ll[0],x_ll[1],x_ll[2]+L[2]],#4
              [x_ll[0]+L[0],x_ll[1],x_ll[2]+L[2]],#5
              [x_ll[0]+L[0],x_ll[1]+L[1],x_ll[2]+L[2]],#6
              [x_ll[0],x_ll[1]+L[1],x_ll[2]+L[2]]]#7
    vertexFlags=[boundaryTags['left'],
                 boundaryTags['right'],
                 boundaryTags['right'],
                 boundaryTags['left'],
                 boundaryTags['left'],
                 boundaryTags['right'],
                 boundaryTags['right'],
                 boundaryTags['left']]
    facets=[[[0,1,2,3]],
            [[0,1,5,4]],
            [[1,2,6,5]],
            [[2,3,7,6]],
            [[3,0,4,7]],
            [[4,5,6,7]]]
    facetFlags=[boundaryTags['bottom'],
                boundaryTags['front'],
                boundaryTags['right'],
                boundaryTags['back'],
                boundaryTags['left'],
                boundaryTags['top']]
    regions=[[x_ll[0]+0.5*L[0],x_ll[1]+0.5*L[1],x_ll[2]+0.5*L[2]]]
    regionFlags=[1.0]
    holes=[]
    if vessel == 'wigley':
        from math import log
        he_hull = log(64.0*he+1.0)/64.0
        #print he,he_hull
        #he_hull = he
        n_points_length = int(ceil(hull_length/he_hull))+1
        n_points_draft  = 2*int(ceil(hull_draft/he_hull))+1
        #print "points",n_points_length,n_points_draft
        dx = hull_length/float(n_points_length-1)
        dz = 2.0*hull_draft/float(n_points_draft-1)
        #print "he",he,dx,dz
        #grid on right half of hull
        for i in range(n_points_length):
            for j in range(n_points_draft):
                x = i*dx - 0.5*hull_length
                z = j*dz - hull_draft
                zStar = min(0.0,z)
                y = 0.5*hull_beam*(1.0 - 4.0*(x/hull_length)**2) * (1.0 - (zStar/hull_draft)**2)
                vertices.append([x+hull_center[0],
                                 y+hull_center[1],
                                 z+hull_center[2]])
                vertexFlags.append(boundaryTags['obstacle'])
        def vN_right(i,j):
            return 8 + i*n_points_draft+j
        for i in range(n_points_length-1):
            for j in range(n_points_draft-1):
                if i < n_points_length//2:
                    facets.append([[vN_right(i,j),vN_right(i+1,j+1),vN_right(i+1,j)]])
                    facetFlags.append(boundaryTags['obstacle'])
                    facets.append([[vN_right(i,j),vN_right(i,j+1),vN_right(i+1,j+1)]])
                    facetFlags.append(boundaryTags['obstacle'])
                else:
                    facets.append([[vN_right(i,j),vN_right(i,j+1),vN_right(i+1,j)]])
                    facetFlags.append(boundaryTags['obstacle'])
                    facets.append([[vN_right(i,j+1),vN_right(i+1,j+1),vN_right(i+1,j)]])
                    facetFlags.append(boundaryTags['obstacle'])                
        #grid on left half of hull
        for i in range(1,n_points_length-1):
            for j in range(1,n_points_draft):
                x = i*dx - 0.5*hull_length
                z = j*dz - hull_draft
                zStar = min(0.0,z)
                y = 0.5*hull_beam*(1.0 - 4.0*(x/hull_length)**2) * (1.0 - (zStar/hull_draft)**2)
                vertices.append([x+hull_center[0],
                                 hull_center[1] - y,
                                 z+hull_center[2]])
                vertexFlags.append(boundaryTags['obstacle'])
        def vN_left(i,j):
            if i== 0 or j==0:
                return vN_right(i,j)
            if i == (n_points_length-1):# or j==(n_points_draft-1):
                return vN_right(i,j)
            else:
                return 8 + n_points_length*n_points_draft+(i-1)*(n_points_draft-1)+j-1
        for i in range(n_points_length-1):
            for j in range(n_points_draft-1):
                if i < n_points_length/2:
                    facets.append([[vN_left(i,j),vN_left(i+1,j+1),vN_left(i+1,j)]])
                    facetFlags.append(boundaryTags['obstacle'])
                    facets.append([[vN_left(i,j),vN_left(i,j+1),vN_left(i+1,j+1)]])
                    facetFlags.append(boundaryTags['obstacle'])
                else:
                    facets.append([[vN_left(i,j),vN_left(i,j+1),vN_left(i+1,j)]])
                    facetFlags.append(boundaryTags['obstacle'])
                    facets.append([[vN_left(i,j+1),vN_left(i+1,j+1),vN_left(i+1,j)]])
                    facetFlags.append(boundaryTags['obstacle'])
        topFacet=[]
        for i in range(n_points_length):
            topFacet.append(vN_right(i,n_points_draft-1))
        for i in range(n_points_length-2,0,-1):
            topFacet.append(vN_left(i,n_points_draft-1))
        facets.append([topFacet])
        facetFlags.append(boundaryTags['obstacle'])
        #for v in vertices: print v
        #for f in facets: print f
        holes.append(hull_center)
    if vessel == 'cube':
        nStart = len(vertices)
        vertices.append([hull_center[0] - 0.5*hull_length,
                         hull_center[1] - 0.5*hull_beam,
                         hull_center[2] - 0.5*hull_draft])
        vertexFlags.append(boundaryTags['obstacle'])
        vertices.append([hull_center[0] - 0.5*hull_length,
                         hull_center[1] + 0.5*hull_beam,
                         hull_center[2] - 0.5*hull_draft])
        vertexFlags.append(boundaryTags['obstacle'])
        vertices.append([hull_center[0] + 0.5*hull_length,
                         hull_center[1] + 0.5*hull_beam,
                         hull_center[2] - 0.5*hull_draft])
        vertexFlags.append(boundaryTags['obstacle'])
        vertices.append([hull_center[0] + 0.5*hull_length,
                         hull_center[1] - 0.5*hull_beam,
                         hull_center[2] - 0.5*hull_draft])
        vertexFlags.append(boundaryTags['obstacle'])
        vertices.append([hull_center[0] - 0.5*hull_length,
                         hull_center[1] - 0.5*hull_beam,
                         hull_center[2] + 0.5*hull_draft])
        vertexFlags.append(boundaryTags['obstacle'])
        vertices.append([hull_center[0] - 0.5*hull_length,
                         hull_center[1] + 0.5*hull_beam,
                         hull_center[2] + 0.5*hull_draft])
        vertexFlags.append(boundaryTags['obstacle'])
        vertices.append([hull_center[0] + 0.5*hull_length,
                         hull_center[1] + 0.5*hull_beam,
                         hull_center[2] + 0.5*hull_draft])
        vertexFlags.append(boundaryTags['obstacle'])
        vertices.append([hull_center[0] + 0.5*hull_length,
                         hull_center[1] - 0.5*hull_beam,
                         hull_center[2] + 0.5*hull_draft])
        vertexFlags.append(boundaryTags['obstacle'])
        facets.append([[nStart,nStart+1,nStart+2,nStart+3]])#1
        facetFlags.append(boundaryTags['obstacle'])
        facets.append([[nStart,nStart+1,nStart+5,nStart+4]])#2
        facetFlags.append(boundaryTags['obstacle'])
        facets.append([[nStart+1,nStart+2,nStart+6,nStart+5]])#3
        facetFlags.append(boundaryTags['obstacle'])
        facets.append([[nStart+2,nStart+3,nStart+7,nStart+6]])#4
        facetFlags.append(boundaryTags['obstacle'])
        facets.append([[nStart+3,nStart,nStart+4,nStart+7]])#5
        facetFlags.append(boundaryTags['obstacle'])
        facets.append([[nStart+4,nStart+5,nStart+6,nStart+7]])#6
        facetFlags.append(boundaryTags['obstacle'])
        holes.append(hull_center)
    domain = Domain.PiecewiseLinearComplexDomain(vertices=vertices,
                                                 vertexFlags=vertexFlags,
                                                 facets=facets,
                                                 facetFlags=facetFlags,
                                                 regions=regions,
                                                 regionFlags=regionFlags,
                                                 holes=holes)
    #go ahead and add a boundary tags member 
    domain.boundaryTags = boundaryTags
    if vessel:
        domain.writePoly("mesh_"+vessel)
    else:
        domain.writePoly("meshNoVessel")
restrictFineSolutionToAllMeshes=False
parallelPartitioningType = MeshTools.MeshParallelPartitioningTypes.node
nLayersOfOverlapForParallel = 0

#steady-state so no initial conditions
initialConditions = None
#use sparse diffusion representation
sd=True

velocity=1.0
diffusion=0.01

def a(x):
    return numpy.array([[diffusion,0.0,0.0],[0.0,diffusion,0.0],[0.0,0.0,diffusion]],'d')
def f(x):
    return 0.0

from proteus import AnalyticalSolutions
class LinearAD_SteadyState(AnalyticalSolutions.SteadyState):
    from math import exp,sqrt
    r"""
    The solution of the steady linear advection-diffusion equation

    The boundary value problem is

    .. math: (bu - au_x)_x = 0 \quad u(0) = 1 \quad u(1) = 0
    """
    def __init__(self,b=[1.0,0,0],a=5.0e-1):
        self.b_ = b
        self.a_ = a
        bn = sqrt(b[0]**2 + b[1]**2 + b[2]**2)
        self.bn=bn
        if bn!=0.0:
            self.D_ = 1.0/(exp(bn/a)-1.0)
        else:
            self.D_ = 0.0
        self.C_ = -self.D_*exp(bn/a)
    def uOfX(self,X):
        x=X
        if self.D_ !=0.0:
            return -self.D_*exp((self.b_[0]*x[0]+self.b_[1]*x[1]+self.b_[2]*x[2])/self.a_) - self.C_
        else:
            return 1.0-(self.b_[0]*x[0]+self.b_[1]*x[1]+self.b_[2]*x[2])/self.bn

import numpy as np
sol = LinearAD_SteadyState(b=np.array([velocity,0.0,0.0]),a=diffusion)

#dirichlet boundary condition functions on (x=0,y,z), (x,y=0,z), (x,y=1,z), (x,y,z=0), (x,y,z=1)
def getDBC(x,flag):
    if flag in [boundaryTags['left'],boundaryTags['right']]:
        return lambda x,t: sol.uOfXT(x,t)

def getAdvFluxBC(x,flag):
    if flag not in [boundaryTags['left'],boundaryTags['right']]:
        return lambda x,t: 0.0

def getDiffFluxBC(x,flag):
    if flag not in [boundaryTags['left'],boundaryTags['right']]:
        return lambda x,t: 0.0

nc = 1
#load analytical solution, dirichlet conditions, flux boundary conditions into the expected variables
analyticalSolution = {0:sol}
#
dirichletConditions = {0:getDBC}
advectiveFluxBoundaryConditions =  {0:getAdvFluxBC}
diffusiveFluxBoundaryConditions = {0:{0:getDiffFluxBC}}
fluxBoundaryConditions = {0:'setFlow'} #options are 'setFlow','noFlow','mixedFlow'

coefficients = ADR.Coefficients(aOfX={0:a},fOfX={0:f},velocity=np.array([velocity,0.0,0.0]),nc=nc,nd=nd)
coefficients.variableNames=['u0']
