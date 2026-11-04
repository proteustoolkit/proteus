from proteus import Domain
from proteus import *
from proteus.default_n import *

nd = 2

# Levee geometry taken from the Case_D schematic.
x_left = 0.0
x_right = 160.0
z_bottom = 0.0
water_table_z = 45.0
river_level = 48.0
crest_level = 57.0
country_level = 53.0

L = [x_right - x_left, crest_level - z_bottom, 1.0]

regularGrid = False

boundaries = ["bottom", "top", "left", "right", "leftTop"]
boundaryTags = dict((key, i + 1) for (i, key) in enumerate(boundaries))

vertices = [
    [0.0, 0.0],
    [160.0, 0.0],
    [160.0, 53.0],
    [105.0, 53.0],
    [92.0, 57.0],
    [70.0, 57.0],
    [55.0, 53.0],
    [35.0, 48.0],
    [20.0, 45.0],
    [0.0, 45.0],
]

vertexFlags = [
    boundaryTags["left"],
    boundaryTags["bottom"],
    boundaryTags["right"],
    boundaryTags["top"],
    boundaryTags["top"],
    boundaryTags["top"],
    boundaryTags["top"],
    boundaryTags["leftTop"],
    boundaryTags["leftTop"],
    boundaryTags["left"],
]

segments = [
    [0, 1],  # bottom
    [1, 2],  # right
    [2, 3],  # dry country-side top
    [3, 4],
    [4, 5],
    [5, 6],
    [6, 7],
    [7, 8],  # wetted left-top boundary
    [8, 9],  # wetted left-top boundary
    [9, 0],  # vertical left side
]

segmentFlags = [
    boundaryTags["bottom"],
    boundaryTags["right"],
    boundaryTags["top"],
    boundaryTags["top"],
    boundaryTags["top"],
    boundaryTags["top"],
    boundaryTags["top"],
    boundaryTags["leftTop"],
    boundaryTags["leftTop"],
    boundaryTags["left"],
]

regions = [[80.0, 30.0]]
regionFlags = [1]

domain = Domain.PlanarStraightLineGraphDomain(
    vertices=vertices,
    vertexFlags=vertexFlags,
    segments=segments,
    segmentFlags=segmentFlags,
    regions=regions,
    regionFlags=regionFlags,
)

domain.writePoly("levee_case_d")

femSpaces = {0: C0_AffineLinearOnSimplexWithNodalBasis}

elementQuadrature = SimplexLobattoQuadrature(nd, 1)
elementBoundaryQuadrature = SimplexLobattoQuadrature(nd - 1, 1)

nnx = 161
nny = 61
nLevels = 1
triangleFlag = 0
triangleOptions = "pAq30Dena%f" % (0.5 * (L[0] / (nnx - 1)) ** 2,)

subgridError = None
