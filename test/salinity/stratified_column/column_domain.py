from proteus import Domain
from proteus.default_n import *

nd = 2

# 1 m x 2 m closed column, water table at mid height
L = [1.0, 2.0, 1.0]
water_table_z = 1.0

boundaries = ["bottom", "right", "top", "left"]
boundaryTags = dict((key, i + 1) for (i, key) in enumerate(boundaries))

vertices = [[0.0, 0.0], [L[0], 0.0], [L[0], L[1]], [0.0, L[1]]]
vertexFlags = [boundaryTags["bottom"], boundaryTags["bottom"], boundaryTags["top"], boundaryTags["top"]]
segments = [[0, 1], [1, 2], [2, 3], [3, 0]]
segmentFlags = [boundaryTags["bottom"], boundaryTags["right"], boundaryTags["top"], boundaryTags["left"]]
regions = [[0.5 * L[0], 0.5 * L[1]]]
regionFlags = [1]

domain = Domain.PlanarStraightLineGraphDomain(vertices=vertices,
                                              vertexFlags=vertexFlags,
                                              segments=segments,
                                              segmentFlags=segmentFlags,
                                              regions=regions,
                                              regionFlags=regionFlags)
domain.writePoly("stratified_column")

femSpaces = {0: C0_AffineLinearOnSimplexWithNodalBasis}
elementQuadrature = SimplexGaussQuadrature(nd, 3)
elementBoundaryQuadrature = SimplexGaussQuadrature(nd - 1, 3)

he = 0.1
nLevels = 1
triangleFlag = 0
triangleOptions = "pAq30Dena%f" % (0.5 * he ** 2,)
