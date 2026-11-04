from proteus import *
from proteus.default_p import *
from proteus.richards import Richards
import math

nd = 2

L=[3.0,5.0,1.0]


regularGrid=False

domain = Domain.PlanarStraightLineGraphDomain()
boundaries=['bottom','top','left','right',
            'swale',
           'drain']

boundaryTags=dict([(key,i+1) for (i,key) in enumerate(boundaries)])

vertices=[[0.0,0.0], #0
          [0.0,3.0],  #1
          [3.0,3.0 ], #2
          [3.0, 0.0], #3
          [3.0,5.0], # 4
          [0.0 ,5.0], #5
          [0.5, 3.0], #6
          [2.5, 3.0], #7
          [2.0, 1.0],#8
          [1.0, 1.0],#9
         ]

# Proteus PSLG domains do not accept a true circle primitive, so the
# drain is represented by a polygonal approximation of a circle.
drain_center = (1.45, 1.65)
drain_radius = 0.25
drain_n_vertices = 24

for i in range(drain_n_vertices):
    theta = 2.0*math.pi*float(i)/float(drain_n_vertices)
    vertices.append([drain_center[0] + drain_radius*math.cos(theta),
                     drain_center[1] + drain_radius*math.sin(theta)])

vertexFlags=[boundaryTags['bottom'], #0
             boundaryTags['left'], #1
             boundaryTags['right'], #2
             boundaryTags['bottom'], #3
             boundaryTags['top'], #4
             boundaryTags['left'], #5
             boundaryTags['swale'], #6
             boundaryTags['swale'], #7
             boundaryTags['swale'], #8
             boundaryTags['swale'], #9
             ] + [boundaryTags['drain']]*drain_n_vertices


segments =[[0,1], #0
           [1,2], #1
           [2,3], #2
           [3,0], #3
           [2,4], #4
           [4,5], #5
           [5,1], #6
           [6,7], #7
           [7,8], #8
           [8,9], #9
           [9,6]] #10

drain_start = 10
for i in range(drain_n_vertices):
    segments.append([drain_start + i,
                     drain_start + ((i + 1) % drain_n_vertices)])


segmentFlags=[boundaryTags['left'], #0
             boundaryTags['swale'], #1
             boundaryTags['right'], #2
             boundaryTags['bottom'], #3
             boundaryTags['right'], #4
             boundaryTags['top'], #5
             boundaryTags['left'], #6

             boundaryTags['swale'], #7
             boundaryTags['swale'], #8
             boundaryTags['swale'], #9
             boundaryTags['swale'], #10
             ] + [boundaryTags['drain']]*drain_n_vertices

holes= [[drain_center[0],drain_center[1]]]
regions=[[0.1,0.1], [1.5, 2.5], [1.5, 4.5]]#, [1.5,0.5]]
regionFlags=[1,2,3]

domain = Domain.PlanarStraightLineGraphDomain(vertices= vertices,
                                              vertexFlags=vertexFlags,
                                              segments=segments,
                                              segmentFlags=segmentFlags,
                                              regions = regions,
                                              regionFlags = regionFlags,
                                               holes= holes)


#dplt.plot_pslg_domain(polygon)

domain.writePoly('bio2d_N')

femSpaces = {0:C0_AffineLinearOnSimplexWithNodalBasis}
#femSpaces = {0:C0_AffineQuadraticOnSimplexWithNodalBasis}

#elementQuadrature = SimplexGaussQuadrature(nd,4)

#elementBoundaryQuadrature = SimplexGaussQuadrature(nd-1,4)

elementQuadrature = SimplexLobattoQuadrature(nd,1)
#
elementBoundaryQuadrature = SimplexLobattoQuadrature(nd-1,1)

nnx=21
nny=21
nLevels = 1
triangleFlag = 0
triangleOptions="pAq30Dena%f" % (0.5*(L[0]/(nnx-1))**2,)

#he= 0.4

#triangleOptions="pa0.02"


subgridError = None
