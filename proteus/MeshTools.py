"""
Tools for creating and manipulating 1,2, and 3D meshes.

.. inheritance-diagram:: proteus.MeshTools
   :parts: 1
"""
from .EGeometry import *
import numpy as np
import array
import inspect
import h5py
import os
from .Archiver import *
from .LinearAlgebraTools import ParVec_petsc4py
from .Profiling import logEvent,memory, memHardLimit
from . import Domain
from . import Comm
from subprocess import run, check_output

class Node(object):
    """A numbered point in 3D Euclidean space

    :ivar N: node number
    :ivar p: Euclidean coordinates

    Comparison operators and a hash value are defined using the 3-tuple of
    coordinates.  This allows using Node objects and tuples of node objects as
    dictionary keys, but in that use case one should be careful not to modify
    the node coordinates.

    >>> n0 = Node(nodeNumber=0,x=0.0,y=0.0,z=0.0)
    >>> n1 = Node(nodeNumber=1,x=1.0,y=1.0,z=1.0)
    >>> n1 >= n0
    True
    """
    xUnitVector = EVec(1.0,0.0,0.0)
    yUnitVector = EVec(0.0,1.0,0.0)
    zUnitVector = EVec(0.0,0.0,1.0)
    def __init__(self,nodeNumber=0,x=0.0,y=0.0,z=0.0):
        self.N=nodeNumber
        self.p=EVec(x,y,z)
        self.basis = [Node.xUnitVector,
                      Node.yUnitVector,
                      Node.zUnitVector]
        self.elementBoundaries=[]
        self.barycenter = self.p
        self.length = 1.0
        self.diameter=self.length
        self.innerDiameter=self.length
        self.hasGeometricInfo = True
        self.unitNormal = Node.xUnitVector
        self.nodes=(self,)
    def computeGeometricInfo(self):
        pass
    def __str__(self):
        return str(self.N)+":"+str(self.p)
    def __hash__(self):
        return hash((self.p[X],self.p[Y],self.p[Z]))
    def __lt__(self,other):
        return  (self.p[X],self.p[Y],self.p[Z]) < \
               (other.p[X],other.p[Y],other.p[Z])
    def __le__(self,other):
        return  (self.p[X],self.p[Y],self.p[Z]) <= \
               (other.p[X],other.p[Y],other.p[Z])
    def __eq__(self,other):
        return  (self.p[X],self.p[Y],self.p[Z]) == \
               (other.p[X],other.p[Y],other.p[Z])
    def __ne__(self,other):
        return  (self.p[X],self.p[Y],self.p[Z]) != \
               (other.p[X],other.p[Y],other.p[Z])
    def __gt__(self,other):
        return  (self.p[X],self.p[Y],self.p[Z]) > \
               (other.p[X],other.p[Y],other.p[Z])
    def __ge__(self,other):
        return  (self.p[X],self.p[Y],self.p[Z]) >= \
               (other.p[X],other.p[Y],other.p[Z])

class Element(object):
    """An numbered polytope in R^n

    :ivar N: element number
    :ivar nodes: sorted tuple of nodes defining the polytope

    The nodes data member can be used as a dictionary key for the polytope as
    long as the nodes aren't later modified.
    """
    def __init__(self,elementNumber=0,nodes=[]):
        self.N = elementNumber
        nodeList = nodes
        nodeList.sort()
        self.nodes = tuple(nodeList)
        self.elementBoundaries=[]

class Edge(Element):
    xUnitVector = EVec(1.0,1.0,0.0)
    yUnitVector = EVec(0.0,1.0,0.0)
    zUnitVector = EVec(0.0,0.0,1.0)
    """
    1D Element--a line connecting two Nodes

    The nodes are stored as a lexicographically sorted node list.
    """
    def __init__(self,edgeNumber=0,nodes=[]):
        #Element.__init__(self,edgeNumber,nodes)
        #inline Element.__init__
        self.N = edgeNumber
        nodeList = nodes
        nodeList.sort()
        self.nodes = tuple(nodeList)
        #self.nodes=nodes
        #self.nodes=nodes[:]
        #self.nodes.sort()
        self.elementBoundaries = [self.nodes[1],self.nodes[0]]
        self.hasGeometricInfo = False

    def computeGeometricInfo(self):
        if not self.hasGeometricInfo:
            self.basis = [self.nodes[1].p - self.nodes[0].p,
                          Edge.yUnitVector,
                          Edge.zUnitVector]
            self.barycenter = (self.nodes[0].p+self.nodes[1].p)/2.0
            self.length = enorm(self.basis[0])
            self.normal = EVec(-self.basis[0][Y], self.basis[0][X],0.0)
            norm = enorm(self.normal)
            if  norm:
                self.unitNormal = self.normal/norm
            else:
                #in 3D edge normals don't make sense in general so above
                #may divide by zero if edge has zero projection onto x-y plane
                self.normal = EVec(0.0, -self.basis[0][Z], self.basis[0][Y])
                self.unitNormal = self.normal/enorm(self.normal)
            self.diameter=self.length
            self.innerDiameter = self.length
            self.hasGeometricInfo = True
            self.nodeUnitNormalList=[]
            self.nodeUnitNormalList.append(-self.basis[0]/self.length)
            self.nodeUnitNormalList.append(self.basis[0]/self.length)
            self.elementBoundaryUnitNormalList=self.nodeUnitNormalList
            self.elementBoundaryJacobianList=[Edge.xUnitVector,Edge.xUnitVector]
def getNodesFromEdges(edges):
    """Extract the subset of nodes from a list of edges."""
    nodes={}
    for e in edges:
        for n in e.nodes:
            nodes[n]=n
    return list(nodes.values())

class Polygon(Element):
    """An abstract 2D element--a closed set of Edges connecting a set of Nodes.

    The nodes and edges are stored as lexicographically sorted lists."""
    def __init__(self,polygonNumber=0,nodes=[]):
        Element.__init__(self,polygonNumber,nodes)
        #the edges have to be set up by the specific polygon
        self.edges=[]

def getEdgesFromPolygons(polygons):
    """Extract the subset of edges from a list of polygons"""
    edges={}
    for p in polygons:
        for e in p.edges:
            edges[e.nodes] = e
    return list(edges.values())

class Triangle(Polygon):
    """A 2D triangular element"""
    edgeMap = {(1,2):0,(0,2):1,(0,1):2}
    zUnitVector = EVec(0.0,0.0,1.0)
    def __init__(self,triangleNumber=0,nodes=[],edgeDict=None):
        #Polygon.__init__(self,triangleNumber,nodes)
        #inline
        self.edges=[]
        #Element.__init__
        #inline
        self.N = triangleNumber
        nodeList = nodes
        nodeList.sort()
        self.nodes = tuple(nodeList)
        #self.nodes=nodes[:]
        #self.nodes.sort()
        #
        edgeNodeList = [(self.nodes[1],self.nodes[2]),
                        (self.nodes[0],self.nodes[2]),
                        (self.nodes[0],self.nodes[1])]
        if edgeDict is None:
            self.edges = [Edge(eN,list(edgeNodes)) for \
                          eN,edgeNodes in enumerate(edgeNodeList)]
        else:
            self.edges = [edgeDict[edgeNodes] for edgeNodes in edgeNodeList]
        self.hasGeometricInfo=False
        self.elementBoundaries=self.edges

    def computeGeometricInfo(self):
        if not self.hasGeometricInfo:
            self.barycenter = (self.nodes[0].p+self.nodes[1].p+self.nodes[2].p)/3.0
            self.basis = [ n.p - self.nodes[0].p for n in self.nodes[1:]]
            self.basis.append(Triangle.zUnitVector)
            self.linearMap = ETen(self.basis[0],self.basis[1],self.basis[2])
            self.normal = ecross(self.basis[0],self.basis[1])
            normNormal = enorm(self.normal)
            self.unitNormal = self.normal/normNormal
            self.area = 0.5*normNormal
            for e in self.edges: e.computeGeometricInfo()
            self.diameter = max([e.length for e in self.edges])
            self.innerDiameter = 4.0*self.area/sum(
                [e.length for e in self.edges])
            self.edgeUnitNormalList=[]
            for nNt,eN in Triangle.edgeMap.items():
                unitNormal = self.edges[eN].unitNormal
                if edot(unitNormal,self.nodes[nNt[0]].p - self.nodes[eN].p) < 0:
                    unitNormal*=-1.0
                self.edgeUnitNormalList.append(unitNormal)
            self.elementBoundaryUnitNormalList = self.edgeUnitNormalList
            self.hasGeometricInfo=True

class Quadrilateral(Polygon):
    """A 2D quadrilateral element"""
    def __init__(self,quadrilateralNumber=0,edges=[],simple=True):
        Polygon.__init__(self,quadrilateralNumber)
        self.edges = edges
        nodeList = getNodesFromEdges(self.edges)
        nodeList = self.sortNodes(nodeList)
        self.nodes = tuple(nodeList)
        self.hasGeometricInfo = False
        self.elementBoundaries = self.edges
        # This boolean flags whether the quadrilateral is simple
        # (eg. a rectangle).  Certain features are more difficult
        # to implement if this is not the case.
        self.simple = True

    def sortNodes(self,nodeList):
        newList = [None] * 4
        coordinate_list = [1,1,1]

        # initialize coordinate mins and maxs
        xMin = nodeList[0].p[X]
        xMax = nodeList[0].p[X]
        yMin = nodeList[0].p[Y]
        yMax = nodeList[0].p[Y]
        zMin = nodeList[0].p[Z]
        zMax = nodeList[0].p[Z]
        for node in nodeList:
            if xMin > node.p[X]:
                xMin = node.p[X]
            if xMax < node.p[X]:
                xMax = node.p[X]
            if yMin > node.p[Y]:
                yMin = node.p[Y]
            if yMax < node.p[Y]:
                yMax = node.p[Y]
            if zMin > node.p[Z]:
                zMin = node.p[Z]
            if zMax < node.p[Z]:
                zMax = node.p[Z]

        # indentify degenerate coordinate space.
        # NOTE - this is not entirely accurate, but assumes
        # 2D quadrilateral objects are orthogonal to one of
        # the cononical coordinate axes

        if xMin==xMax:
            coordinate_list[0] = 0
        if yMin==yMax:
            coordinate_list[1] = 0
        if zMin==zMax:
            coordinate_list[2] = 0
        if sum(coordinate_list) !=2:
            assert 0, 'Invalid 2D quadrilateral object'

        for i, t in enumerate(coordinate_list):
            if t == 0:
                case = i

        # x is degenerate variable
        if case == 0:
            var1 = 1        # y marked as first node
            var2 = 2        # z marked as second
            var1_min = yMin
            var1_max = yMax
            var2_min = zMin
            var2_max = zMax
        # y is degenerate variable
        elif case == 1:
            var1 = 0        # x marked as first node
            var2 = 2        # z marked as second
            var1_min = xMin
            var1_max = xMax
            var2_min = zMin
            var2_max = zMax
        # z is degenerate variable
        elif case == 2:
            var1 = 0        # x marked as first node
            var2 = 1        # y marked as second
            var1_min = xMin
            var1_max = xMax
            var2_min = yMin
            var2_max = yMax
        else:
            assert 0, 'Invalide Quadrilateral Mesh Case'

        for node in nodeList:
            if node.p[var1]==var1_min and node.p[var2]==var2_min:
                newList[0] = node
            elif node.p[var1]==var1_min and node.p[var2]==var2_max:
                newList[1] = node
            elif node.p[var1]==var1_max and node.p[var2]==var2_max:
                newList[2] = node
            elif node.p[var1]==var1_max and node.p[var2]==var2_min:
                newList[3] = node

        for i,item in enumerate(newList):
            if not newList[i]:
                assert 0,'Quadrialteral Mesh Generation Error '+str(newList)+" i = "+str(i)
        return newList

    def computeGeometricInfo(self):
        if not self.hasGeometricInfo:
            for e in self.edges: e.computeGeometricInfo()
            #the nodes must lie in a plane
            #use triangles to compute area
            #grab one triangle
            t0 = Triangle(0,list(self.nodes[0:3]))
            t0.computeGeometricInfo()
            #find the nodes  that lie on the new edge,diagonal0
            for et in t0.edges:
                edgeIsNew=True
                for e in self.edges:
                    if e.nodes == et.nodes:
                        edgeIsNew=False
                if edgeIsNew:
                    break
            diagonal0=et
            t1 = Triangle(0,[self.nodes[3],
                             diagonal0.nodes[0],
                             diagonal0.nodes[1]])
            t1.computeGeometricInfo()
            #get normal from one of the triangles
            self.unitNormal = t0.unitNormal
            self.area = t0.area + t1.area
            #find the long diagonal
            diagonalNode=0
            for n in self.nodes[0:3]:
                if n != diagonal0.nodes[0] and n != diagonal0.nodes[1]:
                    diagonalNode=n
                    break;
            diagonal1 = Edge(0,[n,self.nodes[3]])
            diagonal1.computeGeometricInfo()
            self.diameter = max(diagonal1.length,diagonal0.length)
            self.innerDiameter = 4.0*self.area/sum(
                [e.length for e in self.edges])
            # Calculate the coordinate of a simple quad
            if self.simple==True:
                self.xmin = self.nodes[0].p[X]
                self.ymin = self.nodes[0].p[Y]
                self.xmax = self.nodes[0].p[X]
                self.ymax = self.nodes[0].p[Y]
                for node in self.nodes:
                    if node.p[X] < self.xmin:
                        self.xmin = node.p[X]
                    elif node.p[X] > self.xmax:
                        self.xmax = node.p[X]
                    else:
                        pass
                    if node.p[Y] < self.ymin:
                        self.ymin = node.p[Y]
                    elif node.p[Y] > self.ymax:
                        self.ymax = node.p[Y]
                    else:
                        pass
                self.xmid = (self.xmin+self.xmax)/2.
                self.ymid = (self.ymin+self.ymax)/2.
                self.zmid = 0.

class Polyhedron(Element):
    """
    An abstract 3D Element--a closed set of Polygons connecting a set
    of Edges.

    The nodes and edges are stored as lexicographically sorted lists.
    """
    def __init__(self,polyhedronNumber=0,nodes=[]):
        Element.__init__(self,polyhedronNumber,nodes)
        self.edges=[]
        self.polygons=[]
    def __cmp__(self,other):
        return compareNodes(self.nodes,other.nodes)

class Tetrahedron(Polyhedron):
    """A 3D tetrahedral element"""
    triangleMap = {(1,2,3):0,(0,2,3):1,(0,1,3):2,(0,1,2):3}
    edgeMap = {(0,1): 0,
               (0,2): 1,
               (0,3): 2,
               (1,2): 3,
               (1,3): 4,
               (2,3): 5}

    def __init__(self,tetrahedronNumber,nodes,edgeDict=None,triangleDict=None):
        #Polyhedron.__init__(self,tetrahedronNumber,nodes)
        #inline
        #Element.__init__
        #inline
        self.N = tetrahedronNumber
        nodeList = nodes
        nodeList.sort()
        self.nodes = tuple(nodeList)
        #self.nodes=nodes[:]
        #self.nodes.sort()
        #
        triangleNodeList = [(self.nodes[1],
                             self.nodes[2],
                             self.nodes[3]),
                            (self.nodes[0],
                             self.nodes[2],
                             self.nodes[3]),
                            (self.nodes[0],
                             self.nodes[1],
                             self.nodes[3]),
                            (self.nodes[0],
                             self.nodes[1],
                             self.nodes[2])]
        if triangleDict is None:
            self.triangles = [Triangle(triangleNumber=tN,
                                       nodes=list(triangleNodes))
                              for tN,triangleNodes in
                              enumerate(triangleNodeList)]
        else:
            self.triangles = [triangleDict[triangleNodes] for triangleNodes in
                              triangleNodeList]
        self.polygons=self.triangles
        edgeNodeList = [(self.nodes[0],self.nodes[1]),
                        (self.nodes[0],self.nodes[2]),
                        (self.nodes[0],self.nodes[3]),
                        (self.nodes[1],self.nodes[2]),
                        (self.nodes[1],self.nodes[3]),
                        (self.nodes[2],self.nodes[3])]
        if edgeDict is None:
            self.edges = [Edge(edgeNumber=eN,nodes=list(edgeNodes)) for
                          eN,edgeNodes in enumerate(edgeNodeList)]
        else:
            self.edges = [edgeDict[edgeNodes] for edgeNodes in edgeNodeList]
        self.hasGeometricInfo=False
        self.elementBoundaries = self.triangles

    def computeGeometricInfo(self):
        if not self.hasGeometricInfo:
            for t in self.triangles: t.computeGeometricInfo()
            self.barycenter =(self.nodes[0].p+self.nodes[1].p+self.nodes[2].p+self.nodes[3].p)/4.0
            self.basis = [n.p - self.nodes[0].p for n in self.nodes[1:]]
            self.linearMap = ETen(self.basis[0],self.basis[1],self.basis[2])
            self.volume = abs(edet(self.linearMap))/6.0
            self.diameter = max([t.diameter for t in self.triangles])
             #Zhang's formula for rho=innerDiameter of a simplex
            self.innerDiameter = 6.0*self.volume/sum([t.area for t in
                                                      self.triangles])
            self.triangleUnitNormalList=[]
            for nNt,tN in Tetrahedron.triangleMap.items():
                unitNormal = self.triangles[tN].unitNormal
                if edot(unitNormal,self.nodes[nNt[0]].p - self.nodes[tN].p) < 0:
                    unitNormal *= -1.0
                self.triangleUnitNormalList.append(unitNormal)
            self.elementBoundaryUnitNormalList = self.triangleUnitNormalList
            self.hasGeometricInfo=True

class Hexahedron(Polyhedron):
    """A 3D hexahedral element"""
    def __init__(self,HN,quadrilaterals):
        Polyhedron.__init__(self,HN)
        self.N = HN
        self.quadrilaterals = quadrilaterals
        self.polygons = self.quadrilaterals
        self.edges = getEdgesFromPolygons(quadrilaterals)
        #self.nodes = getNodesFromEdges(self.edges)
        #self.nodes.sort()
        nodeList = getNodesFromEdges(self.edges)
        nodeList.sort()
        self.nodes = tuple(nodeList)
        self.hasGeometricInfo=False
        self.elementBoundaries = self.quadrilaterals

#todo add enum34 and replace with real Python enum
#: Material type of a face (or node) that the partitioner cut: interior to the
#: global mesh, exterior to a subdomain. Reserved: domains use 0 for "no tag" and
#: positive values for conditions.
RANK_INTERFACE_FLAG = -1

class MeshParallelPartitioningTypes(object):
    """
    fake an enum for parallel partitioning options
    """
    element = 0 ;  node     = 1

def _cartesianChunkStarts(nn, p):
    """Where each of p near-equal chunks of 0..nn-1 starts, plus nn: the first
    nn % p chunks are one longer, as in the partitioners' default partition."""
    return np.cumsum([0] + [nn // p + (nn % p > a) for a in range(p)])

def _cartesianBrickStarts(nn, p, level=0):
    """Where each of p bricks along an axis of nn coarsest-level nodes starts, as a node index on the
    grid refined `level` times, plus the node count.

    The bricks are cut on coarse *cells*: the nn-1 cells are split among the p ranks as evenly
    as the partitioners' default chunks, a node belongs to the brick of the cell it starts, and
    the last node to the last brick. Each rank then owns whole coarse cells' worth of fine nodes
    -- nodes 0..19 and 20..40 for 10 cells over 2 ranks at 3 levels, where cutting on coarse
    *nodes* gave 24 and 17 (the 6 and 5 coarse nodes, scaled up, with the trailing fine layers
    going to the lower brick), a factor 2.8 in 3D. It is exactly balanced when the cells divide
    evenly and off by at most one coarse cell otherwise. Every parent of a fine node is a
    node of the coarse cell the fine node lies in, so it is the owner's or its neighbour's.
    An axis with fewer cells than ranks cannot be cut that way and keeps the node chunks.
    """
    if nn == 1:
        return _cartesianChunkStarts(nn, p)
    if nn - 1 < p:
        return np.minimum(_cartesianChunkStarts(nn, p) << level, ((nn - 1) << level) + 1)
    cells = _cartesianChunkStarts(nn - 1, p)
    return np.append(cells[:-1] << level, ((nn - 1) << level) + 1)

def weightedBisection(points, weights, nParts):
    """Recursive coordinate bisection: the part (0..nParts-1) of each point, parts of equal weight.

    Each step splits the points of a part along the longest side of their bounding box, at the
    weighted quantile that gives each side the share of the weight that its share of the parts
    asks for (nParts need not be a power of two: 7 parts split 3|4 and the cut falls at 3/7).
    The parts are boxes, so their surface is as small as the point set allows -- which is what a
    graph partitioner cutting a *coarse* level into ~30 nodes per rank does not deliver, and what
    shows up eight times larger once the cut is scaled up to the finest level. Deterministic:
    ties in a coordinate go by index. A part is empty only when there are more parts than points.
    """
    points = np.asarray(points, dtype=float)
    weights = np.asarray(weights, dtype=float)
    owner = np.zeros(len(points), dtype=np.intc)
    def split(index, first, n):
        if n == 1 or len(index) == 0:
            owner[index] = first
            return
        pts = points[index]
        extent = pts.max(axis=0) - pts.min(axis=0)
        axis = int(np.argmax(extent))
        order = index[np.lexsort((index, points[index, axis]))]
        nLow = n // 2
        cumulative = np.cumsum(weights[order])
        target = cumulative[-1] * nLow / float(n)
        k = int(np.searchsorted(cumulative, target))
        # the cut falls on whichever side of the target is closer
        if k > 0 and abs(cumulative[k - 1] - target) < abs(cumulative[min(k, len(order) - 1)] - target):
            k -= 1
        k = min(max(k + 1, 1 if len(order) > 1 else 0), len(order) - (1 if len(order) > 1 else 0))
        split(order[:k], first, nLow)
        split(order[k:], first + nLow, n - nLow)
    split(np.arange(len(points)), 0, int(nParts))
    return owner

def cartesianRankGrid(nnx, nny, nnz, size):
    """The rank grid (px, py, pz), px*py*pz == size, with the least ghost surface.

    Cutting an nnx x nny x nnz node grid into px x py x pz bricks puts
    (px-1)*nny*nnz + (py-1)*nnx*nnz + (pz-1)*nnx*nny nodes on the cuts; the
    factorization of size with the fewest wins. An axis with one node is never
    cut. Ties go to the smaller largest brick, then to more cuts along x, then
    along y, so the choice is deterministic.
    """
    best = None
    for px in range(1, size + 1):
        if size % px:
            continue
        for py in range(1, size // px + 1):
            if (size // px) % py:
                continue
            pz = size // (px * py)
            if (nnx == 1 and px > 1) or (nny == 1 and py > 1) or (nnz == 1 and pz > 1):
                continue
            surface = (px - 1) * nny * nnz + (py - 1) * nnx * nnz + (pz - 1) * nnx * nny
            biggest = -(-nnx // px) * -(-nny // py) * -(-nnz // pz)
            key = (surface, biggest, -px, -py)
            if best is None or key < best[0]:
                best = (key, (px, py, pz))
    if best is None:
        raise ValueError("a %d x %d x %d node grid cannot be cut among %d ranks" % (nnx, nny, nnz, size))
    return best[1]

def cartesianSubdomain(kind, nn, L, rankGrid=None, nLayersOfOverlap=1, triangleFlag=1, level=0,
                       origin=(0.0, 0.0, 0.0)):
    """This rank's piece of a regular grid, generated without the global mesh.

    kind is 'quad', 'hex', 'tri' or 'tet', nn the node counts (nnx, nny[, nnz])
    and L the lengths, as for generateQuadrilateralMeshFromRectangularGrid,
    generateHexahedralMeshFromRectangularGrid,
    generateTriangularMeshFromRectangularGrid (with triangleFlag, whose
    default this shares) and generateTetrahedralMeshFromRectangularGrid.
    Returns a mesh that stands for the global one -- the global counts only, no
    arrays -- with subdomainMesh and the partition's offsets and
    subdomain-to-global maps set exactly as partitionMesh sets them for that
    global mesh and the node owners cartesianNodeOwners gives for rankGrid
    (default cartesianRankGrid). nodeOld2New and elementOld2New map generator
    numbers to the partition's, in place of the nodeNumbering_global_old2new
    and elementNumbering_global_old2new arrays.

    level > 0 generates the grid refined that many times, cut by the level-0
    bricks scaled (cartesianNodeOwners(level=...)); nn stays the coarsest
    counts. origin shifts the nodes as MeshTools shifts a generated mesh.
    """
    from . import cmeshTools, cpartitioning, Comm
    comm = Comm.get()
    flat = kind in ('quad', 'tri')
    nn = tuple(nn) + (1,) * (3 - len(nn))
    L = tuple(L) + (1.0,) * (3 - len(L))
    if rankGrid is None:
        rankGrid = cartesianRankGrid(nn[0], nn[1], 1 if flat else nn[2], comm.size())
    rankGrid = tuple(rankGrid) + (1,) * (3 - len(rankGrid))
    mesh = {'quad': QuadrilateralMesh, 'hex': HexahedralMesh, 'tri': TriangularMesh, 'tet': TetrahedralMesh}[kind]()
    mesh.comm = comm
    mesh.cmesh = cmeshTools.CMesh()
    mesh.subdomainMesh = mesh.__class__()
    mesh.subdomainMesh.globalMesh = mesh
    mesh.subdomainMesh.cmesh = cmeshTools.CMesh()
    mesh.nLayersOfOverlap = nLayersOfOverlap
    mesh.parallelPartitioningType = MeshParallelPartitioningTypes.node
    (mesh.elementOffsets_subdomain_owned,
     mesh.elementNumbering_subdomain2global,
     mesh.nodeOffsets_subdomain_owned,
     mesh.nodeNumbering_subdomain2global,
     mesh.elementBoundaryOffsets_subdomain_owned,
     mesh.elementBoundaryNumbering_subdomain2global,
     mesh.edgeOffsets_subdomain_owned,
     mesh.edgeNumbering_subdomain2global) = cpartitioning.generateCartesianSubdomain(
         comm.comm.tompi4py(), kind, triangleFlag, nn, L, rankGrid, nLayersOfOverlap,
         mesh.cmesh, mesh.subdomainMesh.cmesh, level, origin)
    (mesh.nNodes_global, mesh.nElements_global,
     mesh.nElementBoundaries_global, mesh.nEdges_global) = cpartitioning.cartesianGlobalCounts(kind, triangleFlag, nn, level)
    #the per-element counts and the largest node star, as the C++ set them
    (_, _, mesh.nNodes_element, mesh.nNodes_elementBoundary, mesh.nElementBoundaries_element,
     _, _, _, _, _, mesh.max_nNodeNeighbors_node) = cmeshTools.buildPythonMeshInterfaceNoArrays(mesh.cmesh)[:11]
    mesh.rankGrid = rankGrid
    mesh.cartesianLevel = level
    mesh.nodeNumbering_global_old2new = None
    mesh.elementNumbering_global_old2new = None
    mesh.nodeOld2New = lambda old: cpartitioning.cartesianNumbering(kind, triangleFlag, nn, rankGrid, False, old, level)
    mesh.elementOld2New = lambda old: cpartitioning.cartesianNumbering(kind, triangleFlag, nn, rankGrid, True, old, level)
    mesh._finishPartitioning(comm)
    mesh.subdomainMesh.rankInterfacesMarked = True     #generateCartesianSubdomain flags the cut faces -1
    #the global scalars a mesh reports, from the pieces
    mpi = comm.comm.tompi4py()
    from mpi4py import MPI
    sd = mesh.subdomainMesh
    mesh.h = mpi.allreduce(sd.h, op=MPI.MAX)
    mesh.hMin = mpi.allreduce(sd.hMin if sd.nElements_global > 0 else np.inf, op=MPI.MIN)
    mesh.sigmaMax = mpi.allreduce(sd.sigmaMax, op=MPI.MAX)
    mesh.volume = float(np.prod(L[:3 if kind in ('hex', 'tet') else 2]))
    if comm.size() == 1:
        #On one rank the piece is the whole grid, in the global numbering, so the
        #global mesh's arrays are the subdomain's (views, not copies) and code
        #that reads the global mesh in serial -- the serial-only dof maps, say --
        #sees a complete mesh.
        for name in _cartesianSerialGlobalArrays:
            setattr(mesh, name, getattr(sd, name))
    return mesh


#the subdomain arrays a one-rank generated mesh shares as its global ones
_cartesianSerialGlobalArrays = (
    'nInteriorElementBoundaries_global', 'nExteriorElementBoundaries_global', 'max_nElements_node',
    'elementNodesArray', 'nodeElementsArray', 'nodeElementOffsets', 'elementNeighborsArray',
    'elementBoundariesArray', 'elementBoundaryNodesArray', 'elementBoundaryElementsArray',
    'elementBoundaryLocalElementBoundariesArray', 'interiorElementBoundariesArray',
    'exteriorElementBoundariesArray', 'edgeNodesArray', 'nodeStarArray', 'nodeStarOffsets',
    'elementMaterialTypes', 'elementBoundaryMaterialTypes', 'nodeMaterialTypes', 'nodeArray',
    'elementDiametersArray', 'elementInnerDiametersArray', 'elementBoundaryDiametersArray',
    'elementBarycentersArray', 'elementBoundaryBarycentersArray', 'nodeDiametersArray',
    'nodeSupportArray')

def cartesianNodeOwners(nnx, nny, nnz, rankGrid=None, size=None, fastest='x', level=0):
    """The rank that owns each node of a regular nnx x nny x nnz node grid cut into bricks.

    Rank (a, b, c) of the px x py x pz rankGrid, which is rank a + px*(b + py*c),
    owns the nodes with indices in its brick [i0,i1) x [j0,j1) x [k0,k1). Along
    each axis the bricks are the near-equal chunks the partitioners use for
    their default partition, so they may be uneven, and a brick is empty when an
    axis has more ranks than nodes. rankGrid defaults to
    cartesianRankGrid(nnx, nny, nnz, size), size to the number of ranks.

    The owners are in the generator's own node numbering, which fastest names:
    'x' for i + nnx*(j + nny*k), as the triangle, tetrahedron and hexahedron
    generators number them, or 'y' for j + nny*i, as the (2D) quadrilateral
    generator does. Passed to Mesh.partitionMesh as nodeOwners, they replace
    the graph partitioner, and the partitioner then needs no PETSc.

    level > 0 gives the owners on the grid refined that many times (every cell
    halved along each axis), cut by the same bricks scaled -- the partition of
    each level of a generated hierarchy (cartesianSubdomain(level=...)).
    nnx, nny, nnz stay the coarsest level's counts.
    """
    if rankGrid is None:
        if size is None:
            from . import Comm
            size = Comm.get().size()
        rankGrid = cartesianRankGrid(nnx, nny, nnz, size)
    px, py, pz = rankGrid
    def starts(nn, p):
        return _cartesianBrickStarts(nn, p, level)
    a, b, c = (np.repeat(np.arange(p), np.diff(starts(nn, p)))
               for nn, p in ((nnx, px), (nny, py), (nnz, pz)))
    if fastest == 'x':
        owners = a[None, None, :] + px * (b[None, :, None] + py * c[:, None, None])
    elif fastest == 'y':
        assert nnz == 1 and pz == 1, "the y-fastest (quadrilateral) numbering is 2D"
        owners = a[:, None] + px * b[None, :]
    else:
        raise ValueError("fastest must be 'x' or 'y', not %r" % (fastest,))
    return np.ascontiguousarray(owners.ravel(), dtype=np.intc)

class Mesh(object):
    """A partition of a domain in R^n into elements.

    This is the base class for meshes. Contains routines for
    plotting the edges of the mesh in Matlab

    Attributes
    ----------
    elementBoundariesArray : array type
        This array lists the global edge number associated with every
        edge or face of an element.
    """
    #cek adding parallel support
    def __init__(self):
        # check if plex is used
        self.plexMesh = False
        #array interface
        self.nSubdomains_global=1
        self.sN = 0
        #node coordinates indexed by node number
        self.nNodes_global=0
        self.nNodes_subdomain=0
        self.nodeArray=None
        self.nodeVelocityArray=None
        self.nNodes_element=0
        #element node numbers, indexed by element number
        self.nElements_global=0
        self.nElements_proc=0
        self.elementNodesArray=None
        self.max_nElements_node=0
        self.nElements_node=None #mwf warning not calculated in buildPythonFromC
        self.nodeElementsArray=None
        self.nodeElementOffsets=None
        #element boundary numbers, indexed by element number
        self.nElementBoundaries_element=0
        self.elementBoundariesArray=None
        #element numbers, indexed by element boundary number and left(0) and right(1) element
        self.nElementBoundaries_global=0
        self.elementBoundaryElementsArray=None
        #local element boundary numbers, indexed by element boundary number and left(0) and right(1) element
        self.elementBoundaryLocalElementBoundariesArray=None
        #neighboring element  numbers, indexed by local element boundary number
        self.elementNeighborsArray=None
        #node numbers, indexed by element boundary number
        self.elementBoundaryNodesArray=None
        #element boundary numbers, indexed by interior/exterior
        #element boundary number
        self.interiorElementBoundariesArray=None
        self.nInteriorElementBoundaries_global=0
        self.exteriorElementBoundariesArray=None
        self.nExteriorElementBoundaries_global=0
        #edge node numbers, indexed by edge number
        self.nEdges_global=0
        self.edgeNodesArray=None
        self.nodeStarArray=None
        self.nodeStarOffsets=None
        self.h=0.0
        self.hMin=0.0
        self.hasGeometricInfo=False
        self.boundaryMesh=None
        #physical coordinates of element barycenters and elementBoundary barycenters
        self.elementBarycentersArray=None
        self.elementBoundaryBarycentersArray=None
        self.nodeDiametersArray=None
        self.nodeSupportArray=None
        #unique labels for classes of elements, elementBoundaries, nodes,
        self.elementMaterialTypes=None
        self.elementBoundaryMaterialTypes=None
        self.nodeMaterialTypes=None
        #parallel stuff
        self.nElements_owned=self.nElements_global
        self.nNodes_owned=self.nNodes_global
        self.nElementBoundaries_owned=self.nElementBoundaries_global
        self.nEdges_owned=self.nEdges_global
        self.elementOffsets_subdomain_owned=[0,self.nElements_global]
        self.elementNumbering_subdomain2global=np.arange(self.nElements_global,dtype='i')
        self.nodeOffsets_subdomain_owned=[0,self.nNodes_global]
        self.nodeNumbering_subdomain2global=np.arange(self.nNodes_global,dtype='i')
        self.elementBoundaryOffsets_subdomain_owned=[0,self.nElementBoundaries_global]
        self.elementBoundaryNumbering_subdomain2global=np.arange(self.nElementBoundaries_global,dtype='i')
        self.edgeOffsets_subdomain_owned=[0,self.nEdges_global]
        self.edgeNumbering_subdomain2global=np.arange(self.nEdges_global,dtype='i')
        self.subdomainMesh=self
        self.globalMesh = None
        self.arGridCollection=None
        self.arGrid=None
        self.nLayersOfOverlap = None
        self.parallelPartitioningType = MeshParallelPartitioningTypes.element
    @property
    def elementBoundaryUserFlags(self):
        """The element boundary flags as user code and the kernels see them.

        ``elementBoundaryMaterialTypes`` carries the partitioner's reserved
        ``-1`` on the faces it cut. A user's boundary-condition function, or a
        kernel that indexes an array by the flag, must see 0 there ("no tag"),
        as it always has; this is that array. It is the flags themselves when
        nothing was marked.
        """
        flags = self.elementBoundaryMaterialTypes
        if self.__dict__.get('rankInterfacesMarked') and flags is not None:
            cached = self.__dict__.get('_elementBoundaryUserFlags')
            if cached is None or cached.shape != flags.shape:
                cached = np.ascontiguousarray(np.maximum(flags, 0), dtype=flags.dtype)
                self.__dict__['_elementBoundaryUserFlags'] = cached
            return cached
        return flags

    def partitionMesh(self,nLayersOfOverlap=1,parallelPartitioningType=MeshParallelPartitioningTypes.node,nodeOwners=None,elementOwners=None,nodeWeights=None):
        from . import cmeshTools
        from . import Comm
        from . import cpartitioning
        comm = Comm.get()
        self.comm=comm
        logEvent(memory("partitionMesh 1","MeshTools"),level=4)
        logEvent("Partitioning mesh among %d processors using partitioningType = %d" % (comm.size(),parallelPartitioningType))
        self.subdomainMesh=self.__class__()
        self.subdomainMesh.globalMesh = self
        self.subdomainMesh.cmesh=cmeshTools.CMesh()
        self.nLayersOfOverlap = nLayersOfOverlap; self.parallelPartitioningType = parallelPartitioningType
        if comm.size() == 1:
            #one rank owns everything, which is what the graph partitioner would
            #say; prescribing it keeps a serial run from initializing PETSc
            if parallelPartitioningType == MeshParallelPartitioningTypes.node and nodeOwners is None:
                nodeOwners = np.zeros(self.nNodes_global,dtype=np.intc)
            elif parallelPartitioningType != MeshParallelPartitioningTypes.node and elementOwners is None:
                elementOwners = np.zeros(self.nElements_global,dtype=np.intc)
        logEvent(memory("partitionMesh 2","MeshTools"),level=4)
        #the global exterior faces, by node set, before partitioning rearranges the global arrays
        realBoundary = set(map(tuple, np.sort(self.elementBoundaryNodesArray[self.exteriorElementBoundariesArray],
                                              axis=1).tolist())) if comm.size() > 1 else None
        nodesBefore = np.array(self.nodeArray, copy=True) if comm.size() > 1 else None
        if parallelPartitioningType == MeshParallelPartitioningTypes.node:
            #mwf for now always gives 1 layer of overlap
            (self.elementOffsets_subdomain_owned,
             self.elementNumbering_subdomain2global,
             self.nodeOffsets_subdomain_owned,
             self.nodeNumbering_subdomain2global,
             self.elementBoundaryOffsets_subdomain_owned,
             self.elementBoundaryNumbering_subdomain2global,
             self.edgeOffsets_subdomain_owned,
             self.edgeNumbering_subdomain2global) = cpartitioning.partitionNodes(comm.comm.tompi4py(),
                                                                                 nLayersOfOverlap,
                                                                                 self.cmesh,
                                                                                 self.subdomainMesh.cmesh,
                                                                                 nodeOwners,
                                                                                 nodeWeights)
        else:
            assert nodeOwners is None and nodeWeights is None, "prescribed owners and node weights need node partitioning"
            (self.elementOffsets_subdomain_owned,
             self.elementNumbering_subdomain2global,
             self.nodeOffsets_subdomain_owned,
             self.nodeNumbering_subdomain2global,
             self.elementBoundaryOffsets_subdomain_owned,
             self.elementBoundaryNumbering_subdomain2global,
             self.edgeOffsets_subdomain_owned,
             self.edgeNumbering_subdomain2global) = cpartitioning.partitionElements(comm.comm.tompi4py(),
                                                                                    nLayersOfOverlap,
                                                                                    self.cmesh,
                                                                                    self.subdomainMesh.cmesh,
                                                                                    elementOwners)
        #
        logEvent(memory("partitionMesh 3","MeshTools"),level=4)
        #Partitioning renumbered the global nodes; keep the permutation so that
        #anything built against the original numbering -- the multilevel
        #hierarchy's node parents, for one -- can be read in the numbering
        #nodeNumbering_subdomain2global refers to.
        self.nodeNumbering_global_old2new = cpartitioning.getNodeNumbering_global_old2new(self.cmesh)
        self.elementNumbering_global_old2new = cpartitioning.getElementNumbering_global_old2new(self.cmesh)
        self._finishPartitioning(comm)
        if realBoundary is not None:
            self._markRankInterfaces(realBoundary, nodesBefore)

    def _nodePermutationFromCoordinates(self, nodesBefore):
        """old2new for the nodes, by matching rows of the node array before and after.

        None if two nodes share coordinates (then the match is ambiguous)."""
        after = np.asarray(self.nodeArray)
        if after.shape != nodesBefore.shape:
            return None
        before_order = np.lexsort(nodesBefore.T[::-1])
        after_order = np.lexsort(after.T[::-1])
        a, b = nodesBefore[before_order], after[after_order]
        if not np.array_equal(a, b):
            return None
        if len(a) > 1 and np.any(np.all(a[1:] == a[:-1], axis=1)):
            return None
        old2new = np.empty(len(a), dtype=np.int64)
        old2new[before_order] = after_order
        return old2new

    def _markRankInterfaces(self, realBoundary, nodesBefore):
        """Give the faces a rank shares with its neighbours the reserved flag -1.

        A rank's exterior faces are the real boundary of the domain plus the
        faces cut by the partition. The domain's flags are copied to the
        subdomain unchanged, and a face that is interior to the global mesh
        carries 0 there, so a cut face and an untagged real boundary face look
        the same. The global mesh tells them apart: a subdomain exterior face
        is a real boundary face exactly when its nodes, in the global
        numbering, are the nodes of an exterior face of the global mesh.
        ``realBoundary`` is that set, taken before partitioning (which
        rearranges the global arrays): sorted original node numbers per face.

        Only faces whose flag is 0 are marked; a positive flag on a cut face is
        a condition the domain put there and is kept. ``-1`` is reserved for the
        partitioner. Nothing else sets it.
        """
        sub = self.subdomainMesh
        exterior = sub.exteriorElementBoundariesArray
        old2new = self.nodeNumbering_global_old2new
        if old2new is None:
            #An element partition renumbers the global nodes in place without
            #saying how; recover the permutation from the coordinates.
            old2new = self._nodePermutationFromCoordinates(nodesBefore)
            if old2new is None:
                logEvent("Rank interfaces not marked: the nodes of this mesh do not "
                         "have distinct coordinates, so the partitioner's renumbering "
                         "cannot be recovered", level=2)
                return
        old2new = np.asarray(old2new)
        new2old = np.empty_like(old2new)
        new2old[old2new] = np.arange(len(old2new), dtype=old2new.dtype)
        nodes = new2old[np.asarray(self.nodeNumbering_subdomain2global)[sub.elementBoundaryNodesArray[exterior]]]
        keys = np.sort(nodes, axis=1).tolist()
        cut = np.fromiter((tuple(k) not in realBoundary for k in keys), dtype=bool, count=len(keys))
        flags = sub.elementBoundaryMaterialTypes
        faces = exterior[cut]
        flags[faces[flags[faces] == 0]] = RANK_INTERFACE_FLAG
        sub.__dict__.pop('_elementBoundaryUserFlags', None)
        sub.rankInterfacesMarked = True

    def _finishPartitioning(self,comm):
        """What every partitioner does once the C++ has built the subdomain mesh
        and the maps: the Python view of the subdomain, the owned counts, and
        the node diameters on the ghosts."""
        self.subdomainMesh.buildFromC(self.subdomainMesh.cmesh)
        self.subdomainMesh.nElements_owned = self.elementOffsets_subdomain_owned[comm.rank()+1] - self.elementOffsets_subdomain_owned[comm.rank()]
        self.subdomainMesh.nNodes_owned = self.nodeOffsets_subdomain_owned[comm.rank()+1] - self.nodeOffsets_subdomain_owned[comm.rank()]
        self.subdomainMesh.nElementBoundaries_owned = self.elementBoundaryOffsets_subdomain_owned[comm.rank()+1] - self.elementBoundaryOffsets_subdomain_owned[comm.rank()]
        self.subdomainMesh.nEdges_owned = self.edgeOffsets_subdomain_owned[comm.rank()+1] - self.edgeOffsets_subdomain_owned[comm.rank()]

        comm.barrier()
        logEvent(memory("partitionMesh 4","MeshTools"),level=4)
        logEvent("Number of Subdomain Elements Owned= "+str(self.subdomainMesh.nElements_owned))
        logEvent("Number of Subdomain Elements = "+str(self.subdomainMesh.nElements_global))
        logEvent("Number of Subdomain Nodes Owned= "+str(self.subdomainMesh.nNodes_owned))
        logEvent("Number of Subdomain Nodes = "+str(self.subdomainMesh.nNodes_global))
        logEvent("Number of Subdomain elementBoundaries Owned= "+str(self.subdomainMesh.nElementBoundaries_owned))
        logEvent("Number of Subdomain elementBoundaries = "+str(self.subdomainMesh.nElementBoundaries_global))
        logEvent("Number of Subdomain Edges Owned= "+str(self.subdomainMesh.nEdges_owned))
        logEvent("Number of Subdomain Edges = "+str(self.subdomainMesh.nEdges_global))
        comm.barrier()
        logEvent("Finished partitioning")
        par_nodeDiametersArray = ParVec_petsc4py(self.subdomainMesh.nodeDiametersArray,
                                                 bs=1,
                                                 n=self.subdomainMesh.nNodes_owned,
                                                 N=self.nNodes_global,
                                                 nghosts=self.subdomainMesh.nNodes_global - self.subdomainMesh.nNodes_owned,
                                                 subdomain2global=self.nodeNumbering_subdomain2global)
        par_nodeDiametersArray.scatter_forward_insert()
        # comm.beginSequential()
        # from Profiling import memory
        # memory()
        # logEvent(memory("Partitioning Mesh","Mesh"),level=1)
        # del self.cmesh
        # #cmeshTools.deleteMeshDataStructures(self.cmesh)
        # logEvent(memory("Without global mesh","Mesh"),level=1)
        # comm.endSequential()
    def partitionMeshFromFiles(self,filebase,base,nLayersOfOverlap=1,parallelPartitioningType=MeshParallelPartitioningTypes.node):
        from . import cmeshTools
        from . import Comm
        from . import cpartitioning
        comm = Comm.get()
        self.comm=comm
        logEvent(memory("partitionMesh 1","MeshTools"),level=4)
        logEvent("Partitioning mesh among %d processors using partitioningType = %d" % (comm.size(),parallelPartitioningType))
        self.subdomainMesh=self.__class__()
        self.subdomainMesh.globalMesh = self
        self.subdomainMesh.cmesh=cmeshTools.CMesh()
        self.nLayersOfOverlap = nLayersOfOverlap; self.parallelPartitioningType = parallelPartitioningType
        logEvent(memory("partitionMesh 2","MeshTools"),level=4)
        
        if parallelPartitioningType == MeshParallelPartitioningTypes.node:
            logEvent("Starting nodal partitioning")#mwf for now always gives 1 layer of overlap
            logEvent("filebase {0:s}".format(filebase))
            logEvent("base {0:d}".format(base))
            logEvent("nLayersOfOverlap {0:d}".format(nLayersOfOverlap))
            logEvent("parallelPartitioningType {0:d}".format(parallelPartitioningType))
            if isinstance(self,TetrahedralMesh):
                    (self.elementOffsets_subdomain_owned,
                     self.elementNumbering_subdomain2global,
                     self.nodeOffsets_subdomain_owned,
                     self.nodeNumbering_subdomain2global,
                     self.elementBoundaryOffsets_subdomain_owned,
                     self.elementBoundaryNumbering_subdomain2global,
                     self.edgeOffsets_subdomain_owned,
                     self.edgeNumbering_subdomain2global) = cpartitioning.partitionNodesFromTetgenFiles(comm.comm.tompi4py(),
                                                                                                                filebase,
                                                                                                                base,
                                                                                                                nLayersOfOverlap,
                                                                                                                self.cmesh,
                                                                                                                self.subdomainMesh.cmesh,
                                                                                                                memHardLimit)
            elif isinstance(self,TriangularMesh):
                    (self.elementOffsets_subdomain_owned,
                     self.elementNumbering_subdomain2global,
                     self.nodeOffsets_subdomain_owned,
                     self.nodeNumbering_subdomain2global,
                     self.elementBoundaryOffsets_subdomain_owned,
                     self.elementBoundaryNumbering_subdomain2global,
                     self.edgeOffsets_subdomain_owned,
                     self.edgeNumbering_subdomain2global) = cpartitioning.partitionNodesFromTriangleFiles(comm.comm.tompi4py(),
                                                                                                                filebase,
                                                                                                                base,
                                                                                                                nLayersOfOverlap,
                                                                                                                self.cmesh,
                                                                                                                self.subdomainMesh.cmesh)                    
            else:
                assert 0,"can't partition non-simplex mesh"

        else:
            logEvent("Starting element partitioning")
            (self.elementOffsets_subdomain_owned,
             self.elementNumbering_subdomain2global,
             self.nodeOffsets_subdomain_owned,
             self.nodeNumbering_subdomain2global,
             self.elementBoundaryOffsets_subdomain_owned,
             self.elementBoundaryNumbering_subdomain2global,
             self.edgeOffsets_subdomain_owned,
             self.edgeNumbering_subdomain2global) = cpartitioning.partitionElementsFromTetgenFiles(comm.comm.tompi4py(),
                                                                                                   filebase,
                                                                                                   base,
                                                                                                   nLayersOfOverlap,
                                                                                                   self.cmesh,
                                                                                                   self.subdomainMesh.cmesh,
                                                                                                   memHardLimit)
        #
        logEvent(memory("partitionMesh 3","MeshTools"),level=4)
        self.buildFromCNoArrays(self.cmesh)
        #Partitioning renumbered the global nodes; keep the permutation so that
        #anything built against the original numbering -- the multilevel
        #hierarchy's node parents, for one -- can be read in the numbering
        #nodeNumbering_subdomain2global refers to.
        self.nodeNumbering_global_old2new = cpartitioning.getNodeNumbering_global_old2new(self.cmesh)
        self.subdomainMesh.buildFromC(self.subdomainMesh.cmesh)
        self.subdomainMesh.nElements_owned = self.elementOffsets_subdomain_owned[comm.rank()+1] - self.elementOffsets_subdomain_owned[comm.rank()]
        self.subdomainMesh.nNodes_owned = self.nodeOffsets_subdomain_owned[comm.rank()+1] - self.nodeOffsets_subdomain_owned[comm.rank()]
        self.subdomainMesh.nElementBoundaries_owned = self.elementBoundaryOffsets_subdomain_owned[comm.rank()+1] - self.elementBoundaryOffsets_subdomain_owned[comm.rank()]
        self.subdomainMesh.nEdges_owned = self.edgeOffsets_subdomain_owned[comm.rank()+1] - self.edgeOffsets_subdomain_owned[comm.rank()]

        comm.barrier()
        logEvent(memory("partitionMesh 4","MeshTools"),level=4)
        logEvent("Number of Subdomain Elements Owned= "+str(self.subdomainMesh.nElements_owned))
        logEvent("Number of Subdomain Elements = "+str(self.subdomainMesh.nElements_global))
        logEvent("Number of Subdomain Nodes Owned= "+str(self.subdomainMesh.nNodes_owned))
        logEvent("Number of Subdomain Nodes = "+str(self.subdomainMesh.nNodes_global))
        logEvent("Number of Subdomain elementBoundaries Owned= "+str(self.subdomainMesh.nElementBoundaries_owned))
        logEvent("Number of Subdomain elementBoundaries = "+str(self.subdomainMesh.nElementBoundaries_global))
        logEvent("Number of Subdomain Edges Owned= "+str(self.subdomainMesh.nEdges_owned))
        logEvent("Number of Subdomain Edges = "+str(self.subdomainMesh.nEdges_global))
        comm.barrier()
        logEvent("Finished partitioning")
        
        par_nodeDiametersArray = ParVec_petsc4py(self.subdomainMesh.nodeDiametersArray,
                                                 bs=1,
                                                 n=self.subdomainMesh.nNodes_owned,
                                                 N=self.nNodes_global,
                                                 nghosts=self.subdomainMesh.nNodes_global - self.subdomainMesh.nNodes_owned,
                                                 subdomain2global=self.nodeNumbering_subdomain2global)
        par_nodeDiametersArray.scatter_forward_insert()
        # comm.beginSequential()
        # from Profiling import memory
        # memory()
        # logEvent(memory("Partitioning Mesh","Mesh"),level=1)
        # del self.cmesh
        # #cmeshTools.deleteMeshDataStructures(self.cmesh)
        # logEvent(memory("Without global mesh","Mesh"),level=1)
        # comm.endSequential()
    def writeMeshXdmf(self,ar,name='',t=0.0,init=False,meshChanged=False,Xdmf_ElementTopology="Triangle",tCount=0, EB=False):
        """Describe this mesh in the archive at time ``t``; with ``EB`` its element boundaries too.

        The grid -- a ymf grid dict in the collection ``Mesh <name>`` (and
        ``EBMesh <name>``) -- is built on every pass through the archive's
        builders; the arrays behind it are written only when ``init or
        meshChanged``. The material types are written on every pass, as
        they were; the local-to-global maps, which ``collect.py`` and a
        hot start from a global archive read, only with the arrays.
        """
        if self.arGridCollection is not None:
            init = False
        elif not init:
            self.arGridCollection = ar.domain[0]
            if EB:
                assert(len(ar.domain) > 1)
                self.arEBGridCollection = ar.domain[1]
        if init:
            self.arGridCollection = ar.domain.collection("Mesh "+name)
            if EB:
                self.arEBGridCollection = ar.domain.collection("EBMesh "+name)
        if not self.arGridCollection.is_current(t):
            rank = ar.comm.rank()
            #
            #topology and geometry
            #
            if ar.global_sync:
                assert ar.writesDatasets, "global_sync not  supported with text heavy data"
                n_elements       = self.globalMesh.nElements_global
                nodes_dims       = [self.globalMesh.nNodes_global, 3]
                elements_dataset = 'elements'+name+str(tCount)
                nodes_dataset    = 'nodes'+name+str(tCount)
                elements_dtype   = self.globalMesh.nodeNumbering_subdomain2global
            else:
                n_elements       = self.nElements_owned
                nodes_dims       = [self.nNodes_global, 3]
                elements_dataset = 'elements'+str(rank)+name+str(tCount)
                nodes_dataset    = 'nodes'+str(rank)+name+str(tCount)
                elements_dtype   = self.elementNodesArray
            self.arGrid = ar.write_grid(self.arGridCollection, None, t, tCount)
            ar.write_topology(self.arGrid, Xdmf_ElementTopology, n_elements,
                              [n_elements, self.nNodes_element],
                              elements_dataset, 'elements'+name, dtype=elements_dtype)
            ar.write_geometry(self.arGrid, nodes_dims, nodes_dataset, 'nodes'+name,
                              dtype=self.nodeArray)
            if init or meshChanged:
                if ar.global_sync:
                    ar.create_dataset_sync(elements_dataset,
                                           offsets=self.globalMesh.elementOffsets_subdomain_owned,
                                           data=self.globalMesh.nodeNumbering_subdomain2global[self.elementNodesArray[:self.nElements_owned]])
                    ar.create_dataset_sync(nodes_dataset,
                                           offsets=self.globalMesh.nodeOffsets_subdomain_owned,
                                           data=self.nodeArray[:self.nNodes_owned])
                elif ar.writesDatasets:
                    ar.create_dataset_async(elements_dataset,data=self.elementNodesArray[:self.nElements_owned])
                    ar.create_dataset_async(nodes_dataset,data=self.nodeArray)
                else:
                    numpy.savetxt(ar.textDataDir+"/elements"+name+".txt",self.elementNodesArray[:self.nElements_owned],fmt='%d')
                    numpy.savetxt(ar.textDataDir+"/nodes"+name+".txt",self.nodeArray)
            #
            #element boundary topology and geometry
            #
            if EB:
                Xdmf_ElementEBTopology = "Triangle" #cek hack
                self.arEBGrid = ar.write_grid(self.arEBGridCollection, None, t, tCount)
                #the boundary grid shares the mesh's nodes dataset
                ar.write_topology(self.arEBGrid, Xdmf_ElementEBTopology, self.nElementBoundaries_global,
                                  [self.nElementBoundaries_global, self.nNodes_elementBoundary],
                                  'elementBoundaries'+str(rank)+name+str(tCount),
                                  'elementBoundaries'+name, dtype=self.elementBoundaryNodesArray)
                ar.write_geometry(self.arEBGrid, [self.nNodes_global, 3],
                                  'nodes'+str(rank)+name+str(tCount), 'nodes'+name,
                                  dtype=self.nodeArray)
                if init or meshChanged:
                    if ar.writesDatasets:
                        ar.create_dataset_async('elementBoundaries'+str(rank)+name+str(tCount), data = self.elementBoundaryNodesArray)
                    else:
                        np.savetxt(ar.textDataDir+"/elementBoundaries"+name+".txt",self.elementBoundaryNodesArray,fmt='%d')

            # Add the local->global index maps for collect.py and for
            # reverse mapping in hotstarts from a global archive.
            if self.globalMesh is not None and not ar.global_sync:
                ar.write_field(self.arGrid, "NodeMapL2G",
                               self.globalMesh.nodeNumbering_subdomain2global, tCount,
                               dimensions=[self.nNodes_global],
                               dataset='nodeMapL2G'+str(rank)+name+str(tCount),
                               text_stem='nodeMapL2G'+name,
                               write_data=init or meshChanged)
                ar.write_field(self.arGrid, "CellMapL2G",
                               self.globalMesh.elementNumbering_subdomain2global[:self.nElements_owned], tCount,
                               center="Cell", dimensions=[self.nElements_owned],
                               dataset='cellMapL2G'+str(rank)+name+str(tCount),
                               text_stem='cellMapL2G'+name,
                               write_data=init or meshChanged)
            #
            #material types
            #
            if ar.global_sync:
                ar.write_field(self.arGrid, "nodeMaterialTypes", self.nodeMaterialTypes, tCount,
                               dimensions=[self.globalMesh.nNodes_global],
                               sync_offsets=self.globalMesh.nodeOffsets_subdomain_owned,
                               sync_data=self.nodeMaterialTypes[:self.nNodes_owned])
                ar.write_field(self.arGrid, "elementMaterialTypes", self.elementMaterialTypes, tCount,
                               center="Cell", dimensions=[self.globalMesh.nElements_global],
                               sync_offsets=self.globalMesh.elementOffsets_subdomain_owned,
                               sync_data=self.elementMaterialTypes[:self.nElements_owned])
                if EB:
                    #the boundary grid's node types are the mesh's, just written
                    ar.write_field(self.arEBGrid, "ebnodeMaterialTypes", self.nodeMaterialTypes, tCount,
                                   dimensions=[self.globalMesh.nNodes_global],
                                   dataset="nodeMaterialTypes_t"+str(tCount), write_data=False)
                    ar.write_field(self.arEBGrid, "elementBoundaryMaterialTypes",
                                   self.elementBoundaryMaterialTypes, tCount,
                                   center="Cell", dimensions=[self.globalMesh.nElementBoundaries_global],
                                   sync_offsets=self.globalMesh.elementBoundaryOffsets_subdomain_owned,
                                   sync_data=self.elementBoundaryMaterialTypes[:self.nElementBoundaries_owned])
            else:
                ar.write_field(self.arGrid, "nodeMaterialTypes", self.nodeMaterialTypes, tCount,
                               dimensions=[self.nNodes_global])
                ar.write_field(self.arGrid, "elementMaterialTypes",
                               self.elementMaterialTypes[:self.nElements_owned], tCount,
                               center="Cell", dimensions=[self.nElements_owned])
                if EB:
                    ar.write_field(self.arEBGrid, "ebnodeMaterialTypes", self.nodeMaterialTypes, tCount,
                                   dimensions=[self.nNodes_global],
                                   dataset="nodeMaterialTypes_p"+str(rank)+"_t"+str(tCount),
                                   write_data=False)
                    ar.write_field(self.arEBGrid, "elementBoundaryMaterialTypes",
                                   self.elementBoundaryMaterialTypes, tCount,
                                   center="Cell", dimensions=[self.nElementBoundaries_global])
            #done with material types
    def buildFromC(self,cmesh):
        from . import cmeshTools
        #
        logEvent(memory("buildFromC","MeshTools"),level=4)
        self.cmesh = cmesh
        (self.nElements_global,
         self.nNodes_global,
         self.nNodes_element,
         self.nNodes_elementBoundary,
         self.nElementBoundaries_element,
         self.nElementBoundaries_global,
         self.nInteriorElementBoundaries_global,
         self.nExteriorElementBoundaries_global,
         self.max_nElements_node,
         self.nEdges_global,
         self.max_nNodeNeighbors_node,
         self.elementNodesArray,
         self.nodeElementsArray,
         self.nodeElementOffsets,
         self.elementNeighborsArray,
         self.elementBoundariesArray,
         self.elementBoundaryNodesArray,
         self.elementBoundaryElementsArray,
         self.elementBoundaryLocalElementBoundariesArray,
         self.interiorElementBoundariesArray,
         self.exteriorElementBoundariesArray,
         self.edgeNodesArray,
         self.nodeStarArray,
         self.nodeStarOffsets,
         self.elementMaterialTypes,
         self.elementBoundaryMaterialTypes,
         self.nodeMaterialTypes,
         self.nodeArray,
         self.nx,self.ny, self.nz,      #NURBS
         self.px,self.py, self.pz,      #NURBS
         self.elementIJK, #NURBS
         self.weights,    #NURBS
         self.U_KNOT,     #NURBS
         self.V_KNOT,     #NURBS
         self.W_KNOT,     #NURBS
         self.elementDiametersArray,
         self.elementInnerDiametersArray,
         self.elementBoundaryDiametersArray,
         self.elementBarycentersArray,
         self.elementBoundaryBarycentersArray,
         self.nodeDiametersArray,
         self.nodeSupportArray,
         self.h,
         self.hMin,
         self.sigmaMax,
         self.volume) = cmeshTools.buildPythonMeshInterface(self.cmesh)
        # print("from C")
        # print("nElements_global = ", self.nElements_global),
        # print("nNodes_global = ", self.nNodes_global),
        # print("nNodes_element = ", self.nNodes_element),
        # print("nNodes_elementBoundary = ", self.nNodes_elementBoundary),
        # print("nElementBoundaries_element = ", self.nElementBoundaries_element),
        # print("nElementBoundaries_global = ", self.nElementBoundaries_global),
        # print("nInteriorElementBoundaries_global = ", self.nInteriorElementBoundaries_global),
        # print("nExteriorElementBoundaries_global = ", self.nExteriorElementBoundaries_global),
        # print("max_nElements_node = ", self.max_nElements_node),
        # print("nEdges_global = ", self.nEdges_global),
        # print("max_nNodeNeighbors_node = ", self.max_nNodeNeighbors_node),
        # print("elementNodesArray = ", self.elementNodesArray),
        # print("nodeElementsArray = ", self.nodeElementsArray),
        # print("nodeElementOffsets = ", self.nodeElementOffsets),
        # print("elementNeighborsArray = ", self.elementNeighborsArray),
        # print("elementBoundariesArray = ", self.elementBoundariesArray),
        # print("elementBoundaryNodesArray = ", self.elementBoundaryNodesArray),
        # print("elementBoundaryElementsArray = ", self.elementBoundaryElementsArray),
        # print("elementBoundaryLocalElementBoundariesArray = ", self.elementBoundaryLocalElementBoundariesArray),
        # print("interiorElementBoundariesArray = ", self.interiorElementBoundariesArray),
        # print("exteriorElementBoundariesArray = ", self.exteriorElementBoundariesArray),
        # print("edgeNodesArray = ", self.edgeNodesArray),
        # print("nodeStarArray = ", self.nodeStarArray),
        # print("nodeStarOffsets = ", self.nodeStarOffsets),
        # print("elementMaterialTypes = ", self.elementMaterialTypes),
        # print("elementBoundaryMaterialTypes = ", self.elementBoundaryMaterialTypes),
        # print("nodeMaterialTypes = ", self.nodeMaterialTypes),
        # print("nodeArray = ", self.nodeArray),
        # print("elementDiametersArray = ", self.elementDiametersArray),
        # print("elementInnerDiametersArray = ", self.elementInnerDiametersArray),
        # print("elementBoundaryDiametersArray = ", self.elementBoundaryDiametersArray),
        # print("elementBarycentersArray = ", self.elementBarycentersArray),
        # print("elementBoundaryBarycentersArray = ", self.elementBoundaryBarycentersArray),
        # print("nodeDiametersArray = ", self.nodeDiametersArray),
        # print("nodeSupportArray = ", self.nodeSupportArray),
        # print("h = ", self.h),
        # print("hMin = ", self.hMin),
        # print("volume = ", self.volume)
        self.hasGeometricInfo = True
        #default to single processor
        self.nNodes_owned = self.nNodes_global
        self.nElements_owned = self.nElements_global
        self.nElementBoundaries_owned = self.nElementBoundaries_global
        self.nEdges_owned = self.nEdges_global
        logEvent(memory("buildFromC","MeshTools"),level=4)
    def buildCMeshFromPlex(self, cmesh):
        from . import cmeshTools
        logEvent("Passing DMPlex to cMeshTools to generate CMesh")
        self.cmesh = cmesh
        cmeshTools.generateCMeshFromDMPlex(self, self.cmesh)
        self.hasGeometricInfo = True
        #default to single processor
        self.nNodes_owned = self.nNodes_global
        self.nElements_owned = self.nElements_global
        self.nElementBoundaries_owned = self.nElementBoundaries_global
        self.nEdges_owned = self.nEdges_global
        logEvent("Done generating CMesh from DMPlex")
    def buildFromCNoArrays(self,cmesh):
        from . import cmeshTools
        #
        logEvent(memory("buildFromC","MeshTools"),level=4)
        self.cmesh = cmesh
        (self.nElements_global,
         self.nNodes_global,
         self.nNodes_element,
         self.nNodes_elementBoundary,
         self.nElementBoundaries_element,
         self.nElementBoundaries_global,
         self.nInteriorElementBoundaries_global,
         self.nExteriorElementBoundaries_global,
         self.max_nElements_node,
         self.nEdges_global,
         self.max_nNodeNeighbors_node,
         self.h,
         self.hMin,
         self.sigmaMax,
         self.volume) = cmeshTools.buildPythonMeshInterfaceNoArrays(self.cmesh)
        self.hasGeometricInfo = False
        logEvent(memory("buildFromCNoArrays","MeshTools"),level=4)
    def buildNodeStarArrays(self):
        import itertools
        if self.nodeStarArray is None:
            self.nodeStarList=[]
            for n in range(self.nNodes_global):
                self.nodeStarList.append(set())
            for i_ele in range(self.nElements_global): #: is this OK for parallel mesh?
                for n1,n2 in itertools.permutations(self.elementNodesArray[i_ele],2):#: works for combination of triangle and quadrilateral 
                    #: if n1<self.nNodes_global: #: Saving only locally owned node is not enough; should include ghost node
                    self.nodeStarList[n1].add(n2) #: does not contain itself; use set() instead of list since each pair is visited 1 or 2 times for 2D mesh
            self.nodeStarOffsets = np.zeros((self.nNodes_global+1,),'i')
            lenNodeStarArray=0
            for nN in range(1,self.nNodes_global+1):
                self.nodeStarOffsets[nN] = self.nodeStarOffsets[nN-1] + len(self.nodeStarList[nN-1])
            self.nodeStarArray =np.fromiter(itertools.chain.from_iterable(self.nodeStarList),'i')
            del self.nodeStarList
    def buildArraysFromLists(self):
        #nodes
        self.nNodes_global = len(self.nodeList)
        self.nodeArray = np.zeros((self.nNodes_global,3),'d')
        nodeElementsList=[]
        for nN,n in enumerate(self.nodeList):
            self.nodeArray[nN][:] = n.p
            nodeElementsList.append([])
        #elements
        self.nNodes_element = len(self.elementList[0].nodes)
        self.nElements_global = len(self.elementList)
        self.elementNodesArray = np.zeros((self.nElements_global,
                                                self.nNodes_element),
                                               'i')
        for en,e in enumerate(self.elementList):
            for nN_element,n in enumerate(e.nodes):
                self.elementNodesArray[en,nN_element]=n.N
                nodeElementsList[n.N].append(en)
        #elements per  node
        nodeElementsDict={}
        for eN in range(self.nElements_global):
            for nN_element in range(self.nNodes_element):
                nN = self.elementNodesArray[eN,nN_element]
                if nN in nodeElementsDict:
                    nodeElementsDict[nN].append(eN)
                else:
                    nodeElementsDict[nN] = [eN]
        self.max_nElements_node = max(len(nodeElementsDict[nN]) for  nN in range(self.nNodes_global))
        self.nElements_node = np.zeros((self.nNodes_global),'i')
        #mwf make a 1d array now
        #self.nodeElementsArrayOld = np.zeros((self.nNodes_global,self.max_nElements_node),'i')
        self.nodeElementOffsets = np.zeros((self.nNodes_global+1,),'i')
        for nN in range(len(nodeElementsDict)):
            elementList = nodeElementsDict[nN]
            self.nElements_node[nN] = len(elementList)
            self.nodeElementOffsets[nN+1] = self.nodeElementOffsets[nN]+self.nElements_node[nN]
            #for eN_element,eN in enumerate(elementList):
            #    self.nodeElementsArrayOld[nN,eN_element]=eN
        self.nodeElementsArray = np.zeros((self.nodeElementOffsets[self.nNodes_global],),'i')
        for nN,elementList in nodeElementsDict.items():
            for eN_element,eN in enumerate(elementList):
                self.nodeElementsArray[self.nodeElementOffsets[nN]+eN_element]=eN
            #
        #
        #elementBoundariesArray
        self.nElementBoundaries_element = len(
            self.elementList[0].elementBoundaries)
        self.elementBoundariesArray = np.zeros(
            (self.nElements_global,self.nElementBoundaries_element),
            'i')
        #collect set of element boundaries while we're looping
        elementBoundaryNumbers=set()
        for eN,e in enumerate(self.elementList):
            for ebN_element,eb in enumerate(e.elementBoundaries):
                self.elementBoundariesArray[eN,ebN_element]=eb.N
                elementBoundaryNumbers.add(eb.N)
        self.nElementBoundaries_global=len(elementBoundaryNumbers)
        #elementBoundaryElementsArray
        self.elementBoundaryElementsArray=np.ones(
            (self.nElementBoundaries_global,2),'i')
        self.elementBoundaryElementsArray*=-1
        self.elementBoundaryLocalElementBoundariesArray=np.zeros(
            (self.nElementBoundaries_global,2),'i')
        elementBoundaryElementsCardArray =np.zeros(
            (self.nElementBoundaries_global),'i')
        for eN in range(self.nElements_global):
            for ebN_element in range(self.nElementBoundaries_element):
                ebN = self.elementBoundariesArray[eN,ebN_element]
                elementBoundaryElementsCardArray[ebN]+=1
                eN_boundaryElement=elementBoundaryElementsCardArray[ebN]-1
                self.elementBoundaryElementsArray[ebN,eN_boundaryElement]=eN
                self.elementBoundaryLocalElementBoundariesArray[ebN,eN_boundaryElement]=ebN_element
                if elementBoundaryElementsCardArray[ebN] > 2:
                    logEvent("WARNING, element neighbors of boundary element > 2")
                    elementBoundaryElementsCardArray[ebN]=2
        #interior and exterior
        self.nExteriorElementBoundaries_global=2*self.nElementBoundaries_global\
                                               - np.sum(
            elementBoundaryElementsCardArray)
        self.nInteriorElementBoundaries_global= self.nElementBoundaries_global-\
                                               self.nExteriorElementBoundaries_global
        self.exteriorElementBoundariesArray=np.zeros(
            (self.nExteriorElementBoundaries_global,),'i')
        self.interiorElementBoundariesArray=np.zeros(
            (self.nInteriorElementBoundaries_global,),'i')
        interior=0
        exterior=0
        for ebN in range(self.nElementBoundaries_global):
            if elementBoundaryElementsCardArray[ebN]==1:
                self.exteriorElementBoundariesArray[exterior]=ebN
                exterior+=1
            else:
                self.interiorElementBoundariesArray[interior]=ebN
                interior+=1
        del elementBoundaryElementsCardArray
        self.nNodes_elementBoundary = len(self.elementBoundaryList[0].nodes)
        self.elementBoundaryNodesArray = np.zeros((self.nElementBoundaries_global,
                                                        self.nNodes_elementBoundary),
                                                       'i')
        for ebN,eb in enumerate(self.elementBoundaryList):
            for nN_element,n in enumerate(eb.nodes):
                self.elementBoundaryNodesArray[ebN,nN_element]=n.N
        #element  neighbors
        self.elementNeighborsArray = np.zeros((self.nElements_global,self.nElementBoundaries_element),'i')
        for eN in range(self.nElements_global):
            for ebN_element in range(self.nElementBoundaries_element):
                ebN = self.elementBoundariesArray[eN,ebN_element]
                eN_left = self.elementBoundaryElementsArray[ebN,0]
                eN_right = self.elementBoundaryElementsArray[ebN,1]
                if eN == eN_left:
                    self.elementNeighborsArray[eN,ebN_element] = eN_right
                elif eN == eN_right:
                    self.elementNeighborsArray[eN,ebN_element] = eN_left
                else:
                    self.elementNeighborsArray[eN,ebN_element] = -1
        #edges
        self.edgeNodesArray = np.zeros(
            (len(self.edgeList),2),'i')
        for en,e in enumerate(self.edgeList):
            self.edgeNodesArray[en,0]=e.nodes[0].N
            self.edgeNodesArray[en,1]=e.nodes[1].N
        #geometric info
        self.computeGeometricInfo()
        self.elementDiametersArray = np.zeros((self.nElements_global,),'d')
        self.elementInnerDiametersArray = np.zeros((self.nElements_global,),'d')
        for en in range(self.nElements_global):
            self.elementDiametersArray[en] = self.elementList[en].diameter
            self.elementInnerDiametersArray[en]=self.elementList[en].innerDiameter
        self.elementBoundaryDiametersArray = np.zeros((self.nElementBoundaries_global,),'d')
        for eN,e in enumerate(self.elementList):
            for ebN_element,eb in enumerate(e.elementBoundaries):
                self.elementBoundaryDiametersArray[self.elementBoundariesArray[eN,ebN_element]] = eb.diameter
        self.elementMaterialTypes = np.zeros((self.nElements_global,),'i')
        self.elementBoundaryMaterialTypes = np.zeros((self.nElementBoundaries_global,),'i')
        self.nodeMaterialTypes = np.zeros((self.nNodes_global,),'i')
        #
        self.elementBarycentersArray         = np.zeros((self.nElements_global,3),'d')
        self.elementBoundaryBarycentersArray = np.zeros((self.nElementBoundaries_global,3),'d')
        for eN in range(self.nElements_global):
            self.elementBarycentersArray[eN,:] = 0.0
            for ebN in range(self.nNodes_element):
                self.elementBarycentersArray[eN,:] += self.nodeArray[self.elementNodesArray[eN,ebN],:]
            self.elementBarycentersArray[eN,:] /= float(self.nNodes_element)
        for ebN in range(self.nElementBoundaries_global):
            self.elementBoundaryBarycentersArray[ebN,:] = 0.0
            for nN in range(self.nNodes_elementBoundary):
                self.elementBoundaryBarycentersArray[ebN,:] += self.nodeArray[self.elementBoundaryNodesArray[ebN,nN],:]
            self.elementBoundaryBarycentersArray[ebN,:] /= float(self.nNodes_elementBoundary)
        #
        #now get rid of lists
        del self.nodeList
        del self.elementList
        del self.elementBoundaryList
        del self.edgeList
        #self.partitionMesh()
    def computeGeometricInfo(self):
        self.elementList[0].computeGeometricInfo()
        self.h=self.elementList[0].diameter
        self.hMin=self.h
        for e in self.elementList[1:]:
            e.computeGeometricInfo()
            self.h = max(self.h,e.diameter)
            self.hMin=min(self.hMin,e.diameter)
            for eb in e.elementBoundaries:
                e.computeGeometricInfo()
        self.hasGeometricInfo=True

    def buildMatlabMeshDataStructures(self,meshFileBase='meshMatlab',writeToFile=True):
        """
        build array data structures for matlab finite element mesh
        representation and write to a file to view and play with in
        matlatb. The current matlab support is mostly for 2d, but this
        will return basic arrays for 1d and 3d too

        in matlab can then print mesh with

        pdemesh(p,e,t)

        if one has pdetoolbox
        where

          p is the vertex or point matrix
          e is the edge matrix, and
          t is the element matrix

        e will be the elementBoundary matrix in 1d and 3d, but perhaps
        should remain the edge array?

        points matrix is [nd x num vertices]
          format :
             row 1 = x coord,
             row 2 = y coord for nodes in mesh
             row 3 = z coord for nodes in mesh ...

        edge matrix is [2*nd+3 x num faces]
          format:
             row 1  = start vertex number
             ...
             row nd   = end vertex number
             row nd+1 = start value in edge parameterization, should be 0
             row nd+2 = next value in edge parameterization, should be 1 or 2
             row nd+nd= end value in edge parameterization, should be 2 or 1
             row 2*nd+1 = global face id, base 1
             row 2*nd+2 = subdomain on left? always 1 for now
             row 2*nd+3 = subdomain on right? always 0 for now

        element matrix is [nd+2 x num elements]
            row 1 = vertex 1 global number
            row 2 = vertex 2 global number
            ...
            row nd+1 = vertex 3 global number
            row 4 = triangle subdomain number

         where 1,2,3 is a local counter clockwise numbering of vertices in
           triangle

        """
        matlabBase = 1
        nd = self.nNodes_element-1
        p = np.zeros((nd,self.nNodes_global),'d')
        e = np.zeros((2*nd+3,self.nElementBoundaries_global),'d')
        t = np.zeros((nd+2,self.nElements_global),'d')

        #load p,e,t and write file
        if writeToFile:
            mfile = open(meshFileBase+'.m','w')
        else:
            mfile = open('/dev/null','w')
        #
        if writeToFile:
            mfile.write('p = [ ... \n')
        for nN in range(self.nNodes_global):
            for I in range(nd):
                p[I,nN]=self.nodeArray[nN,I]
                if writeToFile:
                    mfile.write('%g ' % p[I,nN])
            mfile.write('\n')
        if writeToFile:
            mfile.write(']; \n')
            mfile.write("p = p\';\n")  #need transpose for matlab

        if writeToFile:
            mfile.write('e = [ ... \n')
        for ebN in range(self.nElementBoundaries_global):
            eN_left = self.elementBoundaryElementsArray[ebN,0]
            eN_right= self.elementBoundaryElementsArray[ebN,1]#-1 --> exterior
            for nN in range(self.nNodes_elementBoundary):
                e[nN,ebN]=self.elementBoundaryNodesArray[ebN,nN] + matlabBase #global node number of start node base 1
            #assume for now existing parameterization ok
            for nN in range(self.nNodes_elementBoundary):
                e[self.nNodes_elementBoundary+nN,ebN]=nN #edge param. is 0 to 1
            e[2*self.nNodes_elementBoundary+1,ebN] = ebN+matlabBase
            e[2*self.nNodes_elementBoundary+1,ebN] = self.elementMaterialTypes[eN_left] #subdomain to left
            if eN_right >= 0:
                e[2*self.nNodes_elementBoundary+2,ebN]= self.elementMaterialTypes[eN_right] #subdomain to right
            else:
                e[2*self.nNodes_elementBoundary+2,ebN]= -1
            if writeToFile:
                for i in range(e.shape[0]):
                    mfile.write(' %g ' % e[i,ebN])
                mfile.write(' \n ')
        if writeToFile:
            mfile.write(']; \n')
            mfile.write("e = e\';\n")  #need transpose for matlab

        #write triangles last
        if writeToFile:
            mfile.write('t = [ ... \n')
        for eN in range(self.nElements_global):
            for nN in range(self.nNodes_element):
                t[nN,eN]=self.elementNodesArray[eN,nN]+matlabBase    #global node number for vertex nN
            t[self.nNodes_element,eN]=self.elementMaterialTypes[eN]                     #subdomain id
            if writeToFile:
                for i in range(t.shape[0]):
                    mfile.write('%g ' % t[i,eN])
                mfile.write('\n')
        if writeToFile:
            mfile.write(']; \n');
            mfile.write("t = t\';\n") #need transpose for matlab


        mfile.close()
        return p,e,t

    def writeEdgesMatlab(self,filename):
        """store coordinates in files formatted for Matlab"""
        xfile=filename+'_x.grf'
        yfile=filename+'_y.grf'
        zfile=filename+'_z.grf'
        print('Storing edge information in %s, %s, and %s' % \
              (xfile,yfile,zfile))
        xOut = open(xfile,'w')
        yOut = open(yfile,'w')
        zOut = open(zfile,'w')
        for edge in self.edgeList:
            xOut.write('%14.8e ' % edge.nodes[0].p[X] )
            yOut.write('%14.8e ' % edge.nodes[0].p[Y] )
            zOut.write('%14.8e ' % edge.nodes[0].p[Z] )
        xOut.write('\n')
        yOut.write('\n')
        zOut.write('\n')
        for edge in self.edgeList:
            xOut.write('%14.8e ' % edge.nodes[1].p[X])
            yOut.write('%14.8e ' % edge.nodes[1].p[Y])
            zOut.write('%14.8e ' % edge.nodes[1].p[Z])
        xOut.write('\n')
        yOut.write('\n')
        zOut.write('\n')
        xOut.close()
        yOut.close()
        zOut.close()

    def viewTetrahedraMatlab(self,filename):
        """plot the edges"""
        cmdfile = filename +'.m'
        xfile=filename+'_x.grf'
        yfile=filename+'_y.grf'
        zfile=filename+'_z.grf'
        xedges=filename+'_x'
        yedges=filename+'_y'
        zedges=filename+'_z'
        #the following is for debugging: plot each tet seperately
        nT = len(self.edgeList)//6
        plotcommand = "-r \"load " + xfile + \
                      ", load " + yfile + \
                      ", load " + zfile
        plots=''
        for i in range(nT):
            plots = plots + \
                    ", figure(" +str(i+1)+")" \
                    ", axis([0 1 0 1 0 1]), plot3("+xedges+\
                    "(:,"+str(i)+"*6+1:("+str(i)+"+1)*6),"+yedges+\
                    "(:,"+str(i)+"*6+1:("+str(i)+"+1)*6),"+zedges+\
                    "(:,"+str(i)+"*6+1:("+str(i)+"+1)*6),\'b-\') "
        plotcommand = plotcommand + plots +'\"'
        cmdOut = open(cmdfile,'w')
        cmdOut.write(plotcommand)
        cmdOut.close()
        print('Calling matlab to view mesh')
        os.execlp('matlab',
                  'matlab',
                  '-nodesktop',
                  '-nosplash',
                  '-r',
                  filename)

    def viewMeshMatlab(self,filename):
        """plot the edges"""
        cmdfile = filename +'.m'
        xfile=filename+'_x.grf'
        yfile=filename+'_y.grf'
        zfile=filename+'_z.grf'
        xedges=filename+'_x'
        yedges=filename+'_y'
        zedges=filename+'_z'
        plotcommand = "load " + xfile + \
                      ", load " + yfile + \
                      ", load " + zfile + \
                      ", figure " + \
                      ", axis([0 1 0 1 0 1]), plot3("+xedges+\
                      ","+yedges+\
                      ","+zedges+\
                      ",\'b-\')"
        print(plotcommand)
        cmdOut = open(cmdfile,'w')
        cmdOut.write(plotcommand)
        cmdOut.close()
        print('Calling matlab to view mesh')
        os.execlp('matlab',
                  'matlab',
                  '-nodesktop',
                  '-nosplash',
                  '-r',
                  filename)
#          from os import popen
#          matlab = popen('matlab','w')
#          matlab.write(plotcommand+'\n')
#          matlab.flush()
#          raw_input('Please press return to continue...\n')

    def writeEdgesGnuplot(self,filename):
        """store coordinates in files formatted for Matlab"""
        datfile=filename+'.dat'
        print('Storing edge information in %s' % datfile)
        edgesOut = open(datfile,'w')
        for edge in self.edgeList:
            dataline = '%14.8e %14.8e %14.8e \n' % \
                       (edge.nodes[0].p[X],
                        edge.nodes[0].p[Y],
                        edge.nodes[0].p[Z])
            edgesOut.write(dataline)
            dataline = '%14.8e %14.8e %14.8e \n \n \n' % \
                       (edge.nodes[1].p[X],
                        edge.nodes[1].p[Y],
                        edge.nodes[1].p[Z])
            edgesOut.write(dataline)
        edgesOut.close()
    def writeEdgesGnuplot2(self,filename):
        """store coordinates in files formatted for Matlab"""
        datfile=filename+'.dat'
        print('Storing edge information in %s' % datfile)
        edgesOut = open(datfile,'w')
        for n0,n1 in self.edgeNodesArray:
            dataline = '%14.8e %14.8e %14.8e \n' % \
                       (self.nodeArray[n0][0],
                        self.nodeArray[n0][1],
                        self.nodeArray[n0][2])
            edgesOut.write(dataline)
            dataline = '%14.8e %14.8e %14.8e \n \n \n' % \
                       (self.nodeArray[n1][0],
                        self.nodeArray[n1][1],
                        self.nodeArray[n1][2])
            edgesOut.write(dataline)
        edgesOut.close()
    def viewMeshGnuplot(self,filename):
        cmdfile = filename +'.cmd'
        datfile = filename +'.dat'
        cmd = "set pointsize 2.5 \n set term x11 \n splot \'"+datfile+"\' with linespoints pointsize 2.5 pt 2\n"+\
              "set xlabel \'x\' \n set ylabel \'y\' \n set zlabel \'z\' \n "
        cmdOut = open(cmdfile,'w')
        cmdOut.write(cmd)
        cmdOut.close()
        from os import execlp
        print('Calling gnuplot to view mesh')
        execlp('gnuplot','gnuplot',cmdfile,'-')
    def viewMeshGnuplotPipe(self,filename):
        cmdfile = filename +'.cmd'
        datfile = filename +'.dat'
        cmd = "set pointsize 1.5 \n set term x11 \n splot \'"+datfile+"\' with linespoints pointsize 2.5 pt 2 \n"+\
              "set xlabel \'x\' \n set ylabel \'y\' \n set zlabel \'z\' \n "
        cmdOut = open(cmdfile,'w')
        cmdOut.write(cmd)
        cmdOut.close()
        from os import execlp
        print('Calling gnuplot to view mesh')
        from os import popen
        gnuplot = popen('gnuplot','w')
        gnuplot.write(cmd+'\n')
        gnuplot.flush()
        input('Please press return to continue... \n')
    def viewMeshGnuplotPipePar(self,filenames):
        from os import popen
        gnuplot = popen('gnuplot','w')
        for i,filename in enumerate(filenames):
            cmdfile = filename +'.cmd'
            datfile = filename +'.dat'
            cmd = ("set term x11 %i \n splot \'" % (i,))+datfile+"\' with linespoints \n"+\
                  "set xlabel \'x\' \n set ylabel \'y\' \n set zlabel \'z\'"
            cmdOut = open(cmdfile,'w')
            cmdOut.write(cmd)
            cmdOut.close()
            from os import execlp
            print('Calling gnuplot to view mesh')
            gnuplot.write(cmd+'\n')
            gnuplot.flush()
        input('Please press return to continue... \n')

    def convertFromPUMI(self, domain, MeshAdapt, faceList,regList, parallel=False, dim=3):
        from . import cmeshTools
        from . import MeshAdaptPUMI
        from . import cpartitioning
        from . import Comm
        comm = Comm.get()
        self.cmesh = cmeshTools.CMesh()
        if parallel:
          self.subdomainMesh=self.__class__()
          self.subdomainMesh.globalMesh = self
          self.subdomainMesh.cmesh = cmeshTools.CMesh()
          MeshAdapt.constructFromParallelPUMIMesh(self.cmesh,
              self.subdomainMesh.cmesh)
          if(domain.AdaptManager.reconstructedFlag==1):
            logEvent("Material arrays updating based on reconstructed model.\n")
            MeshAdapt.updateMaterialArrays(self.subdomainMesh.cmesh);
          elif(domain.AdaptManager.reconstructedFlag==2):
            logEvent("Material arrays updating based on better reconstructed model.\n")
            MeshAdapt.updateMaterialArrays2(self.subdomainMesh.cmesh);
          else:
              logEvent("Material arrays updating based on geometric model.\n")
              for i in range(len(faceList)):
                for j in range(len(faceList[i])):
                  #MeshAdapt.updateMaterialArrays(self.subdomainMesh.cmesh,(dim-1), i+1,
                  #    faceList[i][j])
                  MeshAdapt.updateMaterialArrays(self.subdomainMesh.cmesh,(dim-1), domain.boundaryLabels[i], faceList[i][j])
              for i in range(len(regList)):
                for j in range(len(regList[i])):
                  MeshAdapt.updateMaterialArrays(self.subdomainMesh.cmesh,dim, i+1, regList[i][j])
          if dim == 3:
            cmeshTools.allocateGeometricInfo_tetrahedron(self.subdomainMesh.cmesh)
            cmeshTools.computeGeometricInfo_tetrahedron(self.subdomainMesh.cmesh)
          if dim == 2:
            cmeshTools.allocateGeometricInfo_triangle(self.subdomainMesh.cmesh)
            cmeshTools.computeGeometricInfo_triangle(self.subdomainMesh.cmesh)
          self.buildFromCNoArrays(self.cmesh)
          (self.elementOffsets_subdomain_owned,
           self.elementNumbering_subdomain2global,
           self.nodeOffsets_subdomain_owned,
           self.nodeNumbering_subdomain2global,
           self.elementBoundaryOffsets_subdomain_owned,
           self.elementBoundaryNumbering_subdomain2global,
           self.edgeOffsets_subdomain_owned,
           self.edgeNumbering_subdomain2global) = (
               cpartitioning.convertPUMIPartitionToPython(comm.comm.tompi4py(),
                                                          self.cmesh,
                                                          self.subdomainMesh.cmesh))
          self.subdomainMesh.buildFromC(self.subdomainMesh.cmesh)
          self.subdomainMesh.nElements_owned = (
              self.elementOffsets_subdomain_owned[comm.rank()+1] -
              self.elementOffsets_subdomain_owned[comm.rank()])
          self.subdomainMesh.nNodes_owned = (
              self.nodeOffsets_subdomain_owned[comm.rank()+1] -
              self.nodeOffsets_subdomain_owned[comm.rank()])
          self.subdomainMesh.nElementBoundaries_owned = (
              self.elementBoundaryOffsets_subdomain_owned[comm.rank()+1] -
              self.elementBoundaryOffsets_subdomain_owned[comm.rank()])
          self.subdomainMesh.nEdges_owned = (
              self.edgeOffsets_subdomain_owned[comm.rank()+1] -
              self.edgeOffsets_subdomain_owned[comm.rank()])
          comm.barrier()
          par_nodeDiametersArray = (
              ParVec_petsc4py(self.subdomainMesh.nodeDiametersArray,
                              bs=1,
                              n=self.subdomainMesh.nNodes_owned,
                              N=self.nNodes_global,
                              nghosts = self.subdomainMesh.nNodes_global -
                                        self.subdomainMesh.nNodes_owned,
                              subdomain2global = 
                                  self.nodeNumbering_subdomain2global))
          par_nodeDiametersArray.scatter_forward_insert()
          comm.barrier()
        else:
          MeshAdapt.constructFromSerialPUMIMesh(self.cmesh)
          if(domain.AdaptManager.reconstructedFlag==1):
            logEvent("Material arrays updating based on reconstructed model.\n")
            MeshAdapt.updateMaterialArrays(self.cmesh);
          elif(domain.AdaptManager.reconstructedFlag==2):
            logEvent("Material arrays updating based on better reconstructed model.\n")
            MeshAdapt.updateMaterialArrays2(self.cmesh);
          else:
              for i in range(len(faceList)):
                for j in range(len(faceList[i])):
                  #MeshAdapt.updateMaterialArrays(self.cmesh,(dim-1), i+1, faceList[i][j])
                  MeshAdapt.updateMaterialArrays(self.cmesh,(dim-1), domain.boundaryLabels[i], faceList[i][j])
              for i in range(len(regList)):
                for j in range(len(regList[i])):
                  MeshAdapt.updateMaterialArrays(self.cmesh,dim, i+1, regList[i][j])
          if dim == 3:
            cmeshTools.allocateGeometricInfo_tetrahedron(self.cmesh)
            cmeshTools.computeGeometricInfo_tetrahedron(self.cmesh)
          if dim == 2:
            cmeshTools.allocateGeometricInfo_triangle(self.cmesh)
            cmeshTools.computeGeometricInfo_triangle(self.cmesh)
          self.buildFromC(self.cmesh)
        logEvent("meshInfo says : \n"+self.meshInfo())

class _SubdomainHierarchy(object):
    """Parent/child relations over subdomain meshes; see MultilevelMesh.subdomainHierarchy."""
    def hasVertexInterpolation(self):
        #the vertex route reads global node parents, which this does not have;
        #the geometric construction needs only the element relations
        return False


class MultilevelMesh(Mesh):
    """A hierchical multilevel mesh"""
    def __init__(self,levels=1):
        self.meshList=[]
        self.elementParents=None
        self.nodeParentsArrayList=None
    def buildFromC(self,cmultilevelMesh):
        from . import cmeshTools
        self.cmultilevelMesh = cmultilevelMesh
        (self.nLevels,
         self.cmeshList,
         self.elementParentsArrayList,
         self.elementChildrenArrayList,
         self.elementChildrenOffsetsList) = cmeshTools.buildPythonMultilevelMeshInterface(cmultilevelMesh)
    #Whether a finer level inherits the partition of the coarser one. Turn it
    #off to partition every level independently, as proteus always used to.
    levelConsistentPartitioning = True
    #Whether the coarsest level's graph partition weighs each node by the work of everything
    #below it that inherits its owner (descendantWeights) instead of counting nodes.
    weightedCoarsePartition = True
    #What cuts the coarsest level when it is not prescribed: 'graph' (the graph partitioner, PETSc's
    #MatPartitioning) or 'rcb' (weightedBisection on the node coordinates: boxes, compact, no PETSc).
    coarsePartitioner = 'graph'
    #A function f(mesh, parallelPartitioningType) of the coarsest level's mesh
    #returning the rank that owns each of its nodes (node partitioning) or
    #elements (element partitioning), in place of the graph partitioner -- for a
    #regular grid, cartesianNodeOwners. With level-consistent partitioning the
    #finer levels inherit from it, so the whole hierarchy is partitioned without
    #PETSc. Like the switch above it is read while the constructor partitions,
    #so set it on the class.
    coarsestLevelOwners = None
    #How a regular grid (and its hierarchy) is built: 'generated' -- every rank
    #generates only its own pieces (cartesianSubdomain), with no global mesh,
    #no graph partitioner and no global-sized array -- or 'global', the global
    #mesh partitioned by the graph partitioner. MeshOptions.regularGrid selects
    #it per run (and the constructors' regularGrid argument); this is the
    #default for constructors called without one.
    regularGrid = 'generated'
    generated = False

    def generatesInPieces(self, kind, triangleFlag, nLayersOfOverlap, parallelPartitioningType,
                          regularGrid=None):
        """Whether this regular grid's hierarchy is generated in pieces.

        The cases the generated path does not cover take the global path and
        say so: union-jack and left-leaning triangles (they do not refine into
        themselves), overlap other than one layer, and element partitioning
        (which DG uses).
        """
        if regularGrid is None:
            regularGrid = self.regularGrid
        if regularGrid not in ('generated', 'global'):
            raise ValueError("regularGrid is 'generated' or 'global', not %r" % (regularGrid,))
        if regularGrid == 'global':
            logEvent("regular grid: the global mesh, partitioned (regularGrid='global')", level=2)
            return False
        reasons = []
        if kind == 'tri' and triangleFlag != 0:
            reasons.append("triangleFlag %d (only right-leaning triangles, triangleFlag 0, are generated)"
                           % triangleFlag)
        if nLayersOfOverlap != 1:
            reasons.append("%d layers of overlap (only one is generated)" % nLayersOfOverlap)
        if parallelPartitioningType != MeshParallelPartitioningTypes.node:
            reasons.append("element partitioning (only node partitioning is generated)")
        if reasons:
            logEvent("regular grid: the global mesh, partitioned, because of " + "; ".join(reasons), level=1)
            return False
        logEvent("regular grid: generated in pieces (regularGrid='generated')", level=2)
        return True

    def generateCartesianHierarchy(self, kind, counts, origin, L, nLevels, triangleFlag=0):
        """A hierarchy of regular grids, each rank generating only its pieces.

        Level l is the coarsest grid refined l times (cells halved along each
        axis), generated by cartesianSubdomain(level=l): quadrilaterals and
        hexahedra as generateStructuredHierarchy makes them, triangles
        (triangleFlag 0) and tetrahedra as the generators' Kuhn splits at each
        level's own resolution, which are nested. Every level is cut into the
        coarsest level's bricks scaled, so the partition is level-consistent by
        construction: every parent of a fine node a rank owns is in its coarse
        subdomain. meshList holds meshes that stand for the global levels by
        their counts only; the inter-level operators come from
        partitionedVertexInterpolation and subdomainHierarchy, computed per rank.
        """
        from . import Comm
        flat = kind in ('quad', 'tri')
        dim = 2 if flat else 3
        counts = tuple(counts[:dim])
        rankGrid = cartesianRankGrid(counts[0], counts[1], 1 if flat else counts[2], Comm.get().size())
        self.cartesian = dict(kind=kind, counts=counts, triangleFlag=triangleFlag, rankGrid=rankGrid)
        self.generated = True
        self.meshList = []
        for l in range(nLevels):
            mesh = cartesianSubdomain(kind, counts, tuple(L)[:dim], rankGrid=rankGrid,
                                      triangleFlag=triangleFlag, level=l, origin=tuple(origin))
            mesh.levelConsistent = l > 0
            self.meshList.append(mesh)
        self.nLevels = nLevels
        self.cmultilevelMesh = None
        #a diagnostic for test runs: which hierarchies were generated in pieces
        log = os.environ.get('PROTEUS_GENERATED_HIERARCHY_LOG')
        if log and Comm.get().rank() == 0:
            with open(log, 'a') as f:
                f.write("%s %s %d levels %d ranks\n" % (kind, 'x'.join(map(str, counts)), nLevels, Comm.get().size()))
        #there are no global parent/child arrays: see subdomainHierarchy
        self.elementParentsArrayList = None
        self.elementChildrenArrayList = None
        self.elementChildrenOffsetsList = None

    def hasPartitionedNumbering(self):
        """Whether the vertex route can read the hierarchy in the partitioned
        numbering: node partitioning kept each level's permutation, or the
        hierarchy was generated in that numbering."""
        return self.generated or all(getattr(mesh, 'nodeNumbering_global_old2new', None) is not None
                                     for mesh in self.meshList)

    def partitionLevels(self,nLayersOfOverlap=1,
                        parallelPartitioningType=MeshParallelPartitioningTypes.node):
        """Partition every level, each finer one inheriting the coarser one's.

        Partitioning the levels independently gives each its own graph cut, so
        the rank that owns a fine node is unrelated to the ranks that own its
        coarse parents and a transfer between levels is an all-to-all. Here the
        coarsest level is cut by the graph partitioner and every finer level is
        *assigned* from it.

        Node partitioning: a node the refinement inherited stays where it was,
        and a node it added goes to the owner of one of its parents. Every
        parent of a fine node then lies in the node star of an owned coarse
        node, so it is in this rank's subdomain -- owned or ghost -- and a
        prolongation row needs only data the rank already has. Which parent a
        new node takes is balanced rather than "the first": the k-th row picks
        parent k mod (number of parents), so edge midpoints split between their
        two endpoints' owners instead of piling onto one.

        Element partitioning: a fine element goes to the owner of its parent.

        mesh.levelConsistent records that a level inherited. Falls back to
        independent partitioning, and says so, when the hierarchy does not
        support it: a single rank, or -- for node partitioning -- levels that
        are not nodally nested (weakly non-hierarchical refinement, coarsened
        fine meshes), which have no vertex interpolation to read the parents
        from.
        """
        from . import Comm
        byNodes = parallelPartitioningType == MeshParallelPartitioningTypes.node
        consistent = (self.levelConsistentPartitioning and
                      Comm.get().size() > 1 and len(self.meshList) > 1)
        if consistent and byNodes and not self.hasVertexInterpolation():
            logEvent("levels are partitioned independently: the hierarchy is not "
                     "nodally nested, so a finer level has no parents to inherit from",level=2)
            consistent = False
        #a plain function set on the class would otherwise come back bound
        coarsestLevelOwners = inspect.getattr_static(self,'coarsestLevelOwners')
        for l,mesh in enumerate(self.meshList):
            owners = None
            weights = None
            if l == 0 and coarsestLevelOwners is not None:
                owners = np.asarray(coarsestLevelOwners(mesh,parallelPartitioningType),dtype=np.intc)
            elif l == 0 and consistent and byNodes and self.coarsePartitioner == 'rcb':
                from . import Comm as _Comm
                owners = weightedBisection(mesh.nodeArray,self.descendantWeights(),_Comm.get().size())
            elif l == 0 and consistent and byNodes and self.weightedCoarsePartition:
                #the graph partitioner cuts the coarsest level; weigh its nodes by what
                #inherits from them
                weights = self.descendantWeights()
            if consistent and l > 0:
                owners = (self.inheritedNodeOwners(l-1) if byNodes
                          else self.inheritedElementOwners(l-1))
            mesh.partitionMesh(nLayersOfOverlap=nLayersOfOverlap,
                               parallelPartitioningType=parallelPartitioningType,
                               nodeOwners=owners if byNodes else None,
                               elementOwners=None if byNodes else owners,
                               nodeWeights=weights)
            mesh.levelConsistent = owners is not None

    def descendantWeights(self):
        """What each node of the coarsest level will cost its owner, over the whole hierarchy.

        Every finer level inherits the coarsest level's partition (inheritedNodeOwners), so
        the rank that owns a coarse node also owns -- and does the work for -- the fine nodes
        that descend from it under the same parent rule. This counts them: for each level, the
        number of its nodes whose chain of parents ends at each coarsest-level node, summed
        over levels (a node is a node of every level from the one that creates it down to the
        finest, and each level's smoothing and assembly touch it). Handing these to the graph
        partitioner makes it balance the finest level's work, which dominates, instead of
        counting coarse nodes: a coarse node on the boundary has about half the descendants
        of an interior one, and with ~30 coarse nodes per rank that is not noise.

        Returns an int array over the coarsest level's nodes, all >= 1.
        """
        n0 = self.meshList[0].nNodes_global
        ancestor = np.arange(n0)
        weights = np.ones(n0,dtype=np.int64)
        for l in range(len(self.meshList)-1):
            rowptr,colind,nzval = self.vertexInterpolation()[l+1]
            ancestor = ancestor[self._parentPick(rowptr,colind)]
            weights += np.bincount(ancestor,minlength=n0)
        return weights.astype('i')

    def inheritedElementOwners(self,l):
        """The rank that owns each element of level l+1, in level l+1's own numbering:
        the owner of its parent on the partitioned level l."""
        coarse = self.meshList[l]
        old2new = np.asarray(coarse.elementNumbering_global_old2new)
        offsets = np.asarray(coarse.elementOffsets_subdomain_owned)
        coarseOwner = np.searchsorted(offsets[1:],old2new,side='right')
        return coarseOwner[np.asarray(self.elementParentsArrayList[l+1])].astype('i')

    def globalElementsOfSubdomain(self,l):
        """For every element of level l's subdomain mesh, its index on level l's global mesh.

        Partitioning keeps the permutation it applied to a level's elements, so
        this is a table lookup. Where it was not kept (a partitioner that does
        not record it) an element is matched through its nodes instead: each
        subdomain element's nodes are taken back to the global (original) node
        numbers and compared with the global mesh's elements by sorted node
        tuple. Either way exact, or it raises: a silent mismatch here would hand
        a transfer operator the wrong parent.
        """
        mesh = self.meshList[l]
        sd = mesh.subdomainMesh
        elementOld2New = getattr(mesh,'elementNumbering_global_old2new',None)
        if elementOld2New is not None:
            elementOld2New = np.asarray(elementOld2New)
            elementNew2Old = np.empty_like(elementOld2New)
            elementNew2Old[elementOld2New] = np.arange(elementOld2New.size,dtype=elementOld2New.dtype)
            return elementNew2Old[np.asarray(mesh.elementNumbering_subdomain2global)].astype('i')
        old2new = np.asarray(mesh.nodeNumbering_global_old2new)
        new2old = np.empty_like(old2new)
        new2old[old2new] = np.arange(old2new.size,dtype=old2new.dtype)
        def keyOf(nodes):
            tuples = np.ascontiguousarray(np.sort(nodes,axis=1),dtype=np.int64)
            return tuples.view(np.dtype((np.void,tuples.dtype.itemsize*tuples.shape[1]))).ravel()
        sdOld = new2old[np.asarray(mesh.nodeNumbering_subdomain2global)[np.asarray(sd.elementNodesArray)]]
        sdKeys = keyOf(sdOld)
        globalKeys = keyOf(np.asarray(mesh.elementNodesArray))
        order = np.argsort(globalKeys)
        pos = np.searchsorted(globalKeys[order],sdKeys)
        pos[pos == order.size] = 0
        found = order[pos]
        if not np.array_equal(globalKeys[found],sdKeys):
            raise ValueError("level %i: a subdomain element has no counterpart on the global mesh" % l)
        if np.unique(found).size != found.size:
            raise ValueError("level %i: subdomain elements are not distinct on the global mesh" % l)
        return found.astype('i')

    def subdomainHierarchy(self):
        """The parent/child relations of the levels *as each rank sees them*.

        The global relations (elementParentsArrayList, elementChildren...) are
        numbered over the global meshes. This is the same structure over the
        subdomain meshes: for each coarse subdomain element, the fine subdomain
        elements that are its children. A fine element whose parent is not on
        this rank's coarse subdomain has parent -1 and is nobody's child.
        Because a finer level inherits the coarser one's partition
        (partitionLevels), every fine element touching an owned fine node has
        its parent here, and so operators built from this structure are
        complete on the rows a rank owns -- which is all that is kept.

        Returns an object with the attributes MultilevelProjectionOperators
        reads: meshList (the subdomain meshes), elementParentsArrayList,
        elementChildrenArrayList and elementChildrenOffsetsList.
        """
        hierarchy = _SubdomainHierarchy()
        hierarchy.meshList = [mesh.subdomainMesh for mesh in self.meshList]
        hierarchy.elementParentsArrayList = [[]]
        hierarchy.elementChildrenArrayList = []
        hierarchy.elementChildrenOffsetsList = []
        if not self.generated:
            globalOf = [self.globalElementsOfSubdomain(l) for l in range(len(self.meshList))]
        for l in range(len(self.meshList)-1):
            nCoarseSd = hierarchy.meshList[l].nElements_global
            nFineSd = hierarchy.meshList[l+1].nElements_global
            if self.generated:
                from . import cpartitioning, Comm
                c = self.cartesian
                parents = cpartitioning.cartesianSubdomainParents(c['kind'], c['triangleFlag'], c['counts'],
                                                                  c['rankGrid'], l+1, Comm.get().rank())
            else:
                subdomainOfGlobal = -np.ones((self.meshList[l].nElements_global,),'i')
                subdomainOfGlobal[globalOf[l]] = np.arange(nCoarseSd,dtype='i')
                parents = subdomainOfGlobal[np.asarray(self.elementParentsArrayList[l+1])[globalOf[l+1]]]
            hierarchy.elementParentsArrayList.append(parents.astype('i'))
            kids = np.nonzero(parents >= 0)[0]
            order = kids[np.argsort(parents[kids],kind='stable')].astype('i')
            offsets = np.zeros((nCoarseSd+1,),'i')
            offsets[1:] = np.cumsum(np.bincount(parents[kids],minlength=nCoarseSd))
            hierarchy.elementChildrenArrayList.append(order)
            hierarchy.elementChildrenOffsetsList.append(offsets)
        return hierarchy

    def partitionBalance(self):
        """How evenly each level's partition spreads the work, one dict per level.

        Collective. For every level it reports, as max over ranks / mean over ranks (1.0 is
        perfect): ``nodes`` (owned nodes), ``elements`` (subdomain elements, owned and ghost --
        an element loop assembles all of them, so this is the assembly work) and ``nodesSubdomain``
        (owned plus ghost nodes, i.e. memory and the vector length); plus ``ghostFraction``
        (the largest ghost-to-owned node ratio on any rank, the surface-to-volume cost of the cut).
        For an element-partitioned hierarchy the same for elements: ``ownedElements`` (max/mean of
        the elements a rank owns) and ``ghostElementFraction``.
        Meant for asserting that a partitioner has not traded balance for locality and for
        reading off where the cost of a hierarchy is going.
        """
        from . import Comm
        from mpi4py import MPI
        comm = Comm.get().comm.tompi4py()
        out = []
        for mesh in self.meshList:
            sd = mesh.subdomainMesh
            off = np.asarray(mesh.nodeOffsets_subdomain_owned)
            owned = int(off[comm.rank+1]-off[comm.rank])
            eoff = np.asarray(mesh.elementOffsets_subdomain_owned)
            ownedElements = int(eoff[comm.rank+1]-eoff[comm.rank])
            local = np.array([owned,sd.nElements_global,sd.nNodes_global,ownedElements],dtype='d')
            allv = np.array(comm.allgather(local))
            mean = np.maximum(allv.mean(axis=0),1.0e-300)
            ghosts = (allv[:,2]-allv[:,0])/np.maximum(allv[:,0],1.0)
            ghostElements = (allv[:,1]-allv[:,3])/np.maximum(allv[:,3],1.0)
            out.append({'nodes':float(allv[:,0].max()/mean[0]),
                        'elements':float(allv[:,1].max()/mean[1]),
                        'nodesSubdomain':float(allv[:,2].max()/mean[2]),
                        'ghostFraction':float(ghosts.max()),
                        'ownedElements':float(allv[:,3].max()/mean[3]),
                        'ghostElementFraction':float(ghostElements.max())})
        return out

    def inheritedNodeOwners(self,l):
        """The rank that owns each node of level l+1, in level l+1's own numbering.

        Reads the owner of every coarse node from the partitioned level l (its
        node renumbering and offsets), and gives each fine node the owner of
        one parent. Needs level l already partitioned.
        """
        coarse = self.meshList[l]
        old2new = np.asarray(coarse.nodeNumbering_global_old2new)
        offsets = np.asarray(coarse.nodeOffsets_subdomain_owned)
        coarseOwner = np.searchsorted(offsets[1:],old2new,side='right')
        rowptr,colind,nzval = self.vertexInterpolation()[l+1]
        return coarseOwner[self._parentPick(rowptr,colind)].astype('i')

    #Which parent a new node takes its owner from: 'lowest' (the parent with the smallest node
    #number: deterministic, independent of the partition, and on a numbered grid always the
    #neighbour on the same side, so the cut stays a flat plane) or 'alternate' (the k-th row
    #takes parent k mod (number of parents): splits the nodes on a cut between its two sides
    #evenly, but at every refinement the interface becomes a checkerboard, and the ghost layer
    #of a rank several times as large).
    parentRule = 'lowest'

    def _parentPick(self,rowptr,colind):
        """For every fine node, the coarse node whose owner it takes: the column index into colind."""
        nFine = rowptr.shape[0]-1
        counts = np.diff(rowptr)
        if self.parentRule == 'alternate':
            return colind[rowptr[:-1]+(np.arange(nFine) % np.maximum(counts,1))]
        if self.parentRule != 'lowest':
            raise ValueError("parentRule is 'lowest' or 'alternate', not %r" % (self.parentRule,))
        lowest = np.minimum.reduceat(colind,rowptr[:-1])
        return lowest

    def refine(self):
        pass
    def locallyRefine(self,elementTagArray):
        pass
    def buildArrayLists(self):
        self.nLevels = len(self.meshList)
        self.calculateElementParents()
        self.elementParentsArrayList=[[]]
        self.elementChildrenArrayList=[]
        self.elementChildrenOffsetsList=[]
        for l in range(1,self.nLevels):
            self.elementParentsArrayList.append(self.elementParents[l])
            len_children=0
            for children in list(self.elementChildren[l-1].values()):
                len_children += len(children)
            self.elementChildrenArrayList.append(np.zeros((len_children,),'i'))
            self.elementChildrenOffsetsList.append(np.zeros((self.meshList[l-1].nElements_global+1,),'i'))
            index=0
            for eN_p,children in enumerate(self.elementChildren[l-1].values()):
                self.elementChildrenOffsetsList[l-1][eN_p] = index
                for ec in children:
                    self.elementChildrenArrayList[l-1][index] = ec.N
                    index += 1
            self.elementChildrenOffsetsList[l-1][-1] = index
    def generateStructuredHierarchy(self,newMesh,generate,counts,origin,dim,
                                    nLayersOfOverlap=1,
                                    parallelPartitioningType=None):
        """Build a hierarchy of regular quadrilateral or hexahedral grids directly.

        Each level is generated at its own resolution from the grid generator --
        level l+1 has twice as many cells as level l along every axis -- rather
        than by refining the level below. A regular grid's nesting is known in
        closed form, so there is nothing to refine, and the levels share one
        coordinate frame by construction. It is also what lets a parallel run
        start from just the coarse grid's dimensions: no mesh file and no
        external generator or partitioner is involved.

        Parameters
        ----------
        newMesh : callable
            Returns an empty mesh of the right type.
        generate : callable
            ``generate(mesh, counts_l)`` fills the mesh with the regular grid
            that has ``counts_l`` nodes along each axis.
        counts : sequence of tuples
            Nodes per axis on the coarsest level, as (nx, ny, nz).
        origin : (x, y, z)
            Added to every level before anything is derived from its geometry.
        dim : int
            Space dimension, 2 or 3.

        Element parents and children are recorded in the same array lists the C
        refinement fills, so everything that reads them is indifferent to how the
        hierarchy was made.
        """
        from . import Comm
        nLevels = self._structuredLevels
        if parallelPartitioningType is None:
            parallelPartitioningType = MeshParallelPartitioningTypes.node
        levelCounts = []
        n = list(counts)
        for l in range(nLevels):
            levelCounts.append(tuple(n))
            n = [2*(k-1)+1 if k > 1 else 1 for k in n]
        for l in range(nLevels):
            mesh = newMesh()
            generate(mesh,levelCounts[l])
            #Translate before deriving anything from the geometry. Barycenters
            #are computed in C when the grid is generated, so they move too.
            for axis in range(3):
                mesh.nodeArray[:,axis] += origin[axis]
                if getattr(mesh,'elementBarycentersArray',None) is not None:
                    mesh.elementBarycentersArray[:,axis] += origin[axis]
            self.meshList.append(mesh)
        self.cmultilevelMesh = None
        self.nLevels = nLevels
        self.elementParentsArrayList = [[]]
        self.elementChildrenArrayList = []
        self.elementChildrenOffsetsList = []
        self.elementParents = {0: -np.ones((self.meshList[0].nElements_global,),'i')}
        for l in range(nLevels-1):
            parents = self._structuredElementParents(self.meshList[l],self.meshList[l+1],
                                                     levelCounts[l],dim)
            nCoarse = self.meshList[l].nElements_global
            order = np.argsort(parents,kind='stable').astype('i')
            offsets = np.zeros((nCoarse+1,),'i')
            offsets[1:] = np.cumsum(np.bincount(parents,minlength=nCoarse))
            self.elementParentsArrayList.append(parents)
            self.elementChildrenArrayList.append(order)
            self.elementChildrenOffsetsList.append(offsets)
            self.elementParents[l+1] = parents
        self.partitionLevels(nLayersOfOverlap,parallelPartitioningType)

    @staticmethod
    def _structuredElementParents(coarse,fine,coarseCounts,dim):
        """The coarse element containing each fine element of a regular grid.

        Found from the element barycenters and the grid spacing alone, so it does
        not depend on how either level numbers its nodes or elements -- the
        quadrilateral generator numbers nodes y-fastest and the hexahedral one
        x-fastest, and neither is assumed here.
        """
        nodes = np.asarray(coarse.nodeArray)[:,:dim]
        lo = nodes.min(axis=0)
        ncell = np.array([k-1 for k in coarseCounts[:dim]],dtype=int)
        h = (nodes.max(axis=0)-lo)/ncell
        cc = np.asarray(coarse.nodeArray)[np.asarray(coarse.elementNodesArray)][:,:,:dim].mean(axis=1)
        cf = np.asarray(fine.nodeArray)[np.asarray(fine.elementNodesArray)][:,:,:dim].mean(axis=1)
        ic = np.rint((cc-lo)/h-0.5).astype(int)
        key_c = np.ravel_multi_index(ic.T,tuple(ncell))
        lookup = -np.ones((int(np.prod(ncell)),),dtype=int)
        lookup[key_c] = np.arange(cc.shape[0])
        if_ = np.minimum(np.floor((cf-lo)/h).astype(int),ncell-1)
        parents = lookup[np.ravel_multi_index(if_.T,tuple(ncell))]
        assert (parents >= 0).all(), "a fine element lies outside every coarse element"
        return parents.astype('i')

    def calculateElementParents(self,recalculate=False):
        """
        get array elementParents[l,e] = e_c, where element e_c is the parent of element e
            elementParents[0,:] = -1
        """
        if (self.elementParents is None or recalculate):
            self.elementParents = {}
            nLevels = len(self.meshList)
            for l in range(nLevels):
                nE   = self.meshList[l].nElements_global
                self.elementParents[l] = np.ones((nE,),'i')
                self.elementParents[l][:]=-1
            for l in range(0,nLevels-1):
                nEc = self.meshList[l].nElements_global
                for ec in range(nEc):
                    for ef in self.elementChildren[l][ec]:
                        #print """l=%s ec= %s ef.N= %s """ % (l,ec,ef.N)
                        self.elementParents[l+1][ef.N] = ec
                    #ef
                #ec
            #l
    def calculateNodeParents(self,recalculate=False):
        """Record which coarse nodes each fine node descends from.

        Uniform refinement is nested in two ways that the rest of proteus does
        not currently exploit. Coarse node ``i`` keeps its number on the fine
        level, because the refinement routines start numbering new nodes at the
        coarse node count -- so ``fineMesh.nodeArray[:nNodes_coarse]`` is
        exactly ``coarseMesh.nodeArray``. And every node the refinement adds is
        the midpoint of a coarse element edge.

        So each fine node is described by a pair of coarse nodes: ``(i,i)`` for
        a node inherited from the coarse mesh, and the two endpoints of a coarse
        edge for one that was added. That is enough to write down the transfer
        operators for a nodal space without any geometry -- a fine node that is
        a coarse node takes its value, and a midpoint takes the average of its
        two parents -- which is both cheaper than evaluating coarse basis
        functions at fine interpolation points and local to whichever rank owns
        the fine node.

        Sets ``nodeParentsArrayList[l]``, of shape (nNodes_global(l), 2), in the
        node numbering of level ``l-1``. Entry 0 is None: the coarsest level has
        no parent.
        """
        if self.nodeParentsArrayList is not None and not recalculate:
            return self.nodeParentsArrayList
        nLevels = len(self.meshList)
        #accumulate locally and publish only on success: a hierarchy that turns
        #out not to be nested raises partway through, and a half-built list
        #cached on self would be returned intact by the next call.
        nodeParentsArrayList = [None]
        for l in range(nLevels-1):
            coarseMesh = self.meshList[l]
            fineMesh = self.meshList[l+1]
            nNodes_coarse = coarseMesh.nNodes_global
            nNodes_fine = fineMesh.nNodes_global
            coarseNodes = np.asarray(coarseMesh.nodeArray)
            fineNodes = np.asarray(fineMesh.nodeArray)
            elementNodes = np.asarray(coarseMesh.elementNodesArray)
            nNodes_element = elementNodes.shape[1]

            #the identity block below assumes the fine level kept the coarse
            #numbering. That is what globallyRefine* does, but a hierarchy built
            #any other way -- by coarsening a given fine mesh, say -- need not,
            #and then every entry here would be quietly wrong rather than
            #missing, so check it rather than assume it.
            if not np.array_equal(fineNodes[:nNodes_coarse],coarseNodes):
                raise ValueError(
                    "level %i's nodes are not a prefix of level %i's, so the "
                    "levels do not share a node numbering and a fine node "
                    "cannot be named by a pair of coarse nodes. This hierarchy "
                    "needs the geometric transfer operators."
                    % (l,l+1))

            parents = np.zeros((nNodes_fine,2),'i')
            inherited = np.arange(nNodes_coarse,dtype='i')
            parents[:nNodes_coarse,0] = inherited
            parents[:nNodes_coarse,1] = inherited

            #every added node is the midpoint of some coarse element edge, so
            #index the edge midpoints and look the added nodes up by position.
            #the midpoints are formed as 0.5*(a+b) on both sides, so this is an
            #exact match rather than a tolerance test.
            pairs = [(i,j)
                     for i in range(nNodes_element)
                     for j in range(i+1,nNodes_element)]
            midpointOf = {}
            for (i,j) in pairs:
                a = elementNodes[:,i]
                b = elementNodes[:,j]
                mid = 0.5*(coarseNodes[a] + coarseNodes[b])
                for eN in range(mid.shape[0]):
                    midpointOf.setdefault(mid[eN].tobytes(),(a[eN],b[eN]))
            missing = []
            for nN in range(nNodes_coarse,nNodes_fine):
                parentPair = midpointOf.get(fineNodes[nN].tobytes())
                if parentPair is None:
                    missing.append(nN)
                else:
                    parents[nN,0],parents[nN,1] = parentPair
            if missing:
                raise ValueError(
                    "level %i: %i of %i added nodes are not the midpoint of any "
                    "coarse element edge, so this hierarchy is not a uniform "
                    "refinement and the nodal transfer operators do not apply "
                    "(first few: %s)"
                    % (l+1,len(missing),nNodes_fine-nNodes_coarse,missing[:5]))
            nodeParentsArrayList.append(parents)
        self.nodeParentsArrayList = nodeParentsArrayList
        return self.nodeParentsArrayList

    def isNodallyNested(self):
        """Whether the combinatorial nodal transfer operators apply here.

        They need two things, both of which uniform refinement of a single
        coarse mesh gives and neither of which holds in general:

        * each level's nodes are a prefix of the next level's, so the levels
          share a node numbering;
        * every node a level adds is the midpoint of an edge of the level
          below, so it can be named by a pair of coarser nodes.

        Hierarchies that fail this are not unusual and are worth supporting --
        a weakly non-hierarchical refinement whose new vertices do not sit on
        parent edges, or a hierarchy built by coarsening a given fine mesh,
        where the levels need not share any node numbering at all. For those
        the transfer operators have to be built the general way, by evaluating
        coarse basis functions at the fine space's interpolation conditions,
        which is what MultilevelProjectionOperators does. That path stays: this
        predicate exists so the cheaper one is only ever taken when it is
        actually valid, rather than assumed.

        Note that this is a question about the *mesh*. Whether the nodal
        operators apply also depends on the finite element space -- see
        nodalTransfersApply.
        """
        try:
            self.calculateNodeParents()
        except ValueError:
            return False
        return True

    def vertexInterpolation(self,recalculate=False):
        """The vertex-to-vertex prolongations of a hierarchy, as CSR matrices.

        Entry ``l+1`` is ``(rowptr, colind, nzval)`` with one row per node of
        level ``l+1`` and columns that are nodes of level ``l``, in the numbering
        the hierarchy was built with: row i holds the weights with which the
        level-``l`` vertex basis functions interpolate to fine node i. For a
        vertex-based Lagrange space (P1 on simplices, Q1 on quadrilaterals and
        hexahedra) that is exactly the coarse-to-fine transfer operator, and it
        needs no basis evaluation.

        Two constructions, tried in this order:

        * node parents (calculateNodeParents): an inherited node copies its
          coarse value and a node added at an edge midpoint averages that edge's
          two ends. Cheap, but it assumes uniform refinement of simplices.
        * regular grids: every cell is an axis-aligned box, and a fine node's
          weights on the corners of the coarse cell containing it are the
          products of one-dimensional hat functions -- bilinear or trilinear
          interpolation. A quadrilateral's cell-centre node has four parents and
          a hexahedron's face and body centres have four and eight, which node
          parents cannot express. No node numbering is assumed.

        Raises ValueError if neither applies, so the caller can fall back to the
        general geometric operators rather than take a result that is wrong.
        """
        if getattr(self,'_vertexInterpolation',None) is not None and not recalculate:
            return self._vertexInterpolation
        if self.generated:
            raise ValueError("a hierarchy generated in pieces has no global vertex "
                             "interpolation; see partitionedVertexInterpolation")
        nLevels = len(self.meshList)
        try:
            self.calculateNodeParents(recalculate=recalculate)
            result = [None]+[self.nodalProlongation(l) for l in range(nLevels-1)]
        except ValueError as pairsFailure:
            if self.elementParentsArrayList is None or len(self.elementParentsArrayList) < nLevels:
                raise
            try:
                result = [None]+[self._boxInterpolation(l) for l in range(nLevels-1)]
            except ValueError as boxFailure:
                raise ValueError(
                    "no vertex interpolation for this hierarchy. As uniform "
                    "refinement of simplices: %s As a regular grid: %s"
                    % (pairsFailure,boxFailure))
        self._vertexInterpolation = result
        return result

    def hasVertexInterpolation(self):
        """Whether vertexInterpolation can describe this hierarchy.

        A question about the mesh only; whether the operators apply also depends
        on the finite element space -- see FemTools.nodalTransfersApply.
        A hierarchy generated in pieces has one by construction, held per rank.
        """
        if self.generated:
            return True
        try:
            self.vertexInterpolation()
        except ValueError:
            return False
        return True

    def _boxInterpolation(self,l):
        """Level ``l`` to ``l+1`` for a hierarchy of axis-aligned box cells."""
        coarse,fine = self.meshList[l],self.meshList[l+1]
        parents = np.asarray(self.elementParentsArrayList[l+1])
        cn = np.asarray(coarse.nodeArray)
        fn = np.asarray(fine.nodeArray)
        ce = np.asarray(coarse.elementNodesArray)
        fe = np.asarray(fine.elementNodesArray)
        nv = ce.shape[1]
        dim = {4:2,8:3}.get(nv)
        if dim is None:
            raise ValueError("cells with %i vertices are not boxes" % nv)
        X = cn[ce][:,:,:dim]                           # (nE, nv, dim)
        lo = X.min(axis=1)
        hi = X.max(axis=1)
        H = hi-lo
        if (H <= 0.0).any():
            raise ValueError("a coarse cell has zero extent")
        #each cell must be the box spanned by its own extremes: its vertices are
        #exactly the 2^dim corners, however the generator happens to order them
        tol = 1.0e-12*max(1.0,float(np.abs(cn).max()))
        atHi = np.abs(X-hi[:,None,:]) <= tol
        atLo = np.abs(X-lo[:,None,:]) <= tol
        if not (atHi | atLo).all():
            raise ValueError("a coarse cell is not an axis-aligned box")
        code = (atHi*(1 << np.arange(dim))[None,None,:]).sum(axis=2)
        if not (np.sort(code,axis=1) == np.arange(nv)[None,:]).all():
            raise ValueError("a coarse cell does not have one vertex at each corner")
        #a fine node lies in every fine element that contains it, and each of
        #those lies in some coarse cell; any such cell gives the same
        #interpolation because the transfer is continuous across cell faces
        nFine = fn.shape[0]
        first = -np.ones((nFine,),dtype=int)
        flat = fe.ravel()
        owner = np.repeat(np.arange(fe.shape[0]),fe.shape[1])
        first[flat[::-1]] = owner[::-1]
        if (first < 0).any():
            raise ValueError("a fine node belongs to no element")
        P = parents[first]
        if (P < 0).any():
            raise ValueError("a fine element has no coarse parent")
        Y = fn[:,:dim]
        w = np.ones((nFine,nv))
        for d in range(dim):
            w *= np.clip(1.0-np.abs(Y[:,None,d]-X[P][:,:,d])/H[P][:,None,d],0.0,1.0)
        if np.abs(w.sum(axis=1)-1.0).max() > 1.0e-10:
            raise ValueError("a fine node lies outside its coarse cell")
        keep = w > 1.0e-14
        rowptr = np.zeros((nFine+1,),'i')
        rowptr[1:] = np.cumsum(keep.sum(axis=1))
        colind = ce[P][keep].astype('i')
        nzval = w[keep].astype('d')
        return rowptr,colind,nzval

    def partitionedVertexInterpolation(self,l):
        """vertexInterpolation()[l+1], in the partitioned global numbering.

        For a hierarchy generated in pieces, only the rows this rank owns
        (LinearAlgebraTools.vertexProlongationMat takes either).

        The same translation partitionedNodeParents makes, for a matrix: rows
        are fine nodes and columns coarse nodes, and partitioning renumbers each
        level's nodes independently so that every rank's owned ones are
        contiguous. An unpartitioned hierarchy has no permutation and the CSR is
        returned unchanged.
        """
        if self.generated:
            #only the rows of the fine nodes this rank owns, computed here
            from . import cpartitioning, Comm
            c = self.cartesian
            return cpartitioning.cartesianVertexInterpolation(c['kind'], c['triangleFlag'], c['counts'],
                                                              c['rankGrid'], l+1, Comm.get().rank())
        rowptr,colind,nzval = self.vertexInterpolation()[l+1]
        coarseOld2New = getattr(self.meshList[l],'nodeNumbering_global_old2new',None)
        fineOld2New = getattr(self.meshList[l+1],'nodeNumbering_global_old2new',None)
        if coarseOld2New is None or fineOld2New is None:
            return rowptr,colind,nzval
        coarseOld2New = np.asarray(coarseOld2New)
        fineOld2New = np.asarray(fineOld2New)
        nFine = rowptr.shape[0]-1
        #row r of the result is the old row that maps to r
        oldOfNew = np.empty((nFine,),dtype=int)
        oldOfNew[fineOld2New] = np.arange(nFine)
        lengths = np.diff(rowptr)[oldOfNew]
        newptr = np.zeros((nFine+1,),'i')
        newptr[1:] = np.cumsum(lengths)
        gather = np.repeat(rowptr[oldOfNew]-newptr[:-1],lengths)+np.arange(int(newptr[-1]))
        return newptr,coarseOld2New[colind[gather]].astype('i'),nzval[gather].copy()

    def useHierarchicalGeometry(self):
        """Keep node coordinates only on the finest level.

        Because refinement numbers the nodes it adds after the ones it
        inherited, every coarser level's node coordinates are literally a
        prefix of the finest level's -- so each coarse level's nodeArray can be
        a view into the finest array instead of its own copy.

        The saving is the tail of a geometric series and so is bounded --
        measured at 14% of the coordinate memory for a 3D hierarchy, and about
        a third in 2D. The point is less the bytes than that the levels then
        cannot disagree about where a shared node is, which is the property the
        nodal transfer operators above are built on.

        This is opt-in rather than the default because it makes the levels
        alias: writing to a coarse nodeArray would move the fine mesh too, and
        proteus does translate a mesh in place (MultilevelTetrahedralMesh
        shifts level 0 by x/y/z while constructing). The views are handed back
        read-only so that any such write raises instead of silently corrupting
        the finer levels; move the finest level and re-call this instead.

        Returns the number of bytes no longer duplicated.
        """
        if len(self.meshList) < 2:
            return 0
        finest = self.meshList[-1]
        fineNodes = np.asarray(finest.nodeArray)
        saved = 0
        for l in range(len(self.meshList)-1):
            mesh = self.meshList[l]
            nNodes = mesh.nNodes_global
            own = np.asarray(mesh.nodeArray)
            if not np.array_equal(own,fineNodes[:nNodes]):
                raise ValueError(
                    "level %i's nodes are not a prefix of the finest level's, "
                    "so this hierarchy cannot share geometry; it is not a "
                    "uniform refinement of a single coarse mesh" % (l,))
            view = fineNodes[:nNodes]
            view.flags.writeable = False
            mesh.nodeArray = view
            saved += own.nbytes
        return saved

    def partitionedNodeParents(self,l):
        """calculateNodeParents()[l+1], in the partitioned global numbering.

        calculateNodeParents works in the numbering the hierarchy was built
        with. Partitioning renumbers the global nodes so each rank's owned ones
        are contiguous, and it is that numbering -- the one
        nodeNumbering_subdomain2global refers to -- that the solvers' vectors
        and matrices use. The two are not the same, so a parent pair taken
        straight from calculateNodeParents indexes the wrong unknowns once the
        mesh has been partitioned.

        An unpartitioned hierarchy has no permutation to apply and the parents
        are returned unchanged, which is why this is safe to call either way.
        """
        parents = self.calculateNodeParents()[l+1]
        coarseOld2New = getattr(self.meshList[l],'nodeNumbering_global_old2new',None)
        fineOld2New = getattr(self.meshList[l+1],'nodeNumbering_global_old2new',None)
        if coarseOld2New is None or fineOld2New is None:
            return parents
        coarseOld2New = np.asarray(coarseOld2New)
        fineOld2New = np.asarray(fineOld2New)
        partitioned = np.empty_like(parents)
        partitioned[fineOld2New] = coarseOld2New[parents]
        return partitioned

    def nodalProlongation(self,l):
        """The prolongation from level ``l`` to ``l+1`` for a C0 nodal space.

        Returned as (rowptr, colind, nzval) in CSR, with one row per fine node:
        a fine node inherited from the coarse mesh copies its coarse value, and
        a node the refinement added takes the average of the two coarse nodes
        whose edge it bisects. For an affine-linear nodal space this is exactly
        the interpolation of the coarse basis onto the fine nodes, which is what
        makes it a drop-in for the geometrically-constructed operator.
        """
        parents = self.calculateNodeParents()[l+1]
        nNodes_fine = parents.shape[0]
        inherited = parents[:,0] == parents[:,1]
        nnz = int(np.where(inherited,1,2).sum())
        rowptr = np.zeros((nNodes_fine+1,),'i')
        rowptr[1:] = np.cumsum(np.where(inherited,1,2))
        colind = np.zeros((nnz,),'i')
        nzval = np.zeros((nnz,),'d')
        single = np.flatnonzero(inherited)
        double = np.flatnonzero(~inherited)
        colind[rowptr[single]] = parents[single,0]
        nzval[rowptr[single]] = 1.0
        colind[rowptr[double]] = parents[double,0]
        colind[rowptr[double]+1] = parents[double,1]
        nzval[rowptr[double]] = 0.5
        nzval[rowptr[double]+1] = 0.5
        return rowptr,colind,nzval

class PointMesh(Mesh):
    #Elements=Nodes
    """
    0D mesh
    """
    def __init__(self,points):
        self.nodeArray=points
        self.nNodes_global = points.shape[0]
        self.elementNodesArray=np.arange(self.nNodes_global,dtype='i')
        self.nElements_global = self.nNodes_global

MX=0
MY=1
MZ=2
I=0
J=1
K=1

class EdgeGrid(Mesh):
    """A 1D regular grid on an interval"""
    def __init__(self,nx=2,Lx=1.0):
        Mesh.__init__(self)
        #dimensions and ranges
        self.nx=nx
        self.ex=nx-1
        self.nRange_x = list(range(nx))
        self.eRange_x = list(range(self.ex))
        #lengths
        self.Lx=Lx
        self.dx = Lx/self.ex
        #node coordinates
        self.nodeGridArray = np.zeros((self.nx,3),'d')
        for i in self.nRange_x:
            self.nodeGridArray[i,MX] = i*self.dx
        #edge node numbers
        self.edgeNodesArray=np.zeros((self.ex,2),'i')
        #try to do this like we'll do 2d and 3d
        #edge nodes
        en=2
        edgeNodeNumbers = np.zeros((en,),'i')
        #reference edge
        eI=1
        refEdge_nodeIndeces = [-eI,eI]
        refEdge_NodeDict={}
        for rn,rnii in enumerate(refEdge_nodeIndeces):
            refEdge_NodeDict[rnii] = rn
        for i in self.eRange_x:
            #edge number
            eN=i
            #fine grid index of edge
            ii = 2*i + 1
            #fine grid index of edge nodes
            for rn,rnii in enumerate(refEdge_nodeIndeces):
                nii = rnii + ii
                edgeNodeNumbers[rn]=nii//2
            self.edgeNodesArray[eN,:]=edgeNodeNumbers
        #Mesh interface
        self.nNodes_global=self.nx
        self.nEdges_global=self.ex
        self.nElements_global=self.ex
        self.nElementBoundaries_global=self.nx
        self.nodeArray=self.nodeGridArray
        self.elementNodesArray=self.edgeNodesArray
        self.elementBoundariesArray=self.nodeArray
        self.boundaryMesh=PointMesh(np.array([self.nodeArray[0],
                                              self.nodeArray[-1]],dtype='d'))

class QuadrilateralGrid(Mesh):
    """A 2D regular grid of quadrilateral cells"""
    def __init__(self,nx=2,ny=2,Lx=1.0,Ly=1.0):
        Mesh.__init__(self)
        #nodes
        self.nx=nx
        self.ny=ny
        self.nxy=nx*ny
        #edges
        self.eXx=nx-1
        self.eXy=ny
        self.eXxy=self.eXx*self.eXy
        self.eYx=nx
        self.eYy=ny-1
        self.eYxy = self.eYx*self.eYy
        self.eXYx = self.eXx + self.eYx
        self.eXYy = self.eXy + self.eYy
        self.eXYxy = self.eXxy + self.eYxy
        #quads
        self.qx = nx-1
        self.qy = ny-1
        self.qxy = self.qx*self.qy
        #ranges
        self.nRange_x = list(range(self.nx))
        self.nRange_y = list(range(self.ny))
        self.qRange_x = list(range(self.qx))
        self.qRange_y = list(range(self.qx))
        #lengths
        self.Lx=Lx
        self.Ly=Ly
        self.dx = Lx/self.eXx
        self.dy = Ly/self.eYy
        #node coordinates
        self.nodeGridArray=np.zeros((nx,ny,3),'d')
        for i in self.nRange_x:
            for j in self.nRange_y:
                self.nodeGridArray[i,j,MX]=i*self.dx
                self.nodeGridArray[i,j,MY]=j*self.dy
        #edge node numbers
        en=2
        edgeNodeNumbers = np.zeros((en,),'i')
        self.edgeNodesArray=np.zeros((self.eXYxy,en),'i')
        #quad node numbers
        qn=4
        quadNodeNumbers = np.zeros((qn,),'i')
        self.quadrilateralNodesArray=np.zeros((self.qxy,qn),'i')
        #quad edge numbers
        qe=4
        quadEdgeNumbers = np.zeros((qe,),'i')
        self.quadrilateralEdgesArray=np.zeros((self.qxy,qe),'i')
        #reference quad
        refQuad_NodeIndeces = [(-1,-1),
                               (-1, 1),
                               ( 1,-1),
                               ( 1, 1)]
        refQuad_NodeIndeces.sort()
        #a map between reference node indeces and numbers
        refQuad_NodeDict={}
        for rn,rniijj in enumerate(refQuad_NodeIndeces):
            refQuad_NodeDict[rniijj]=rn
        refQuad_EdgeIndeces = [(-1,0),
                               ( 0,-1),
                               ( 0, 1),
                               ( 1, 0)]
        refQuad_EdgeIndeces.sort()
        refQuad_EdgeNodes=[]
        #use the map between indeces and numbers to
        #map edge indeces to the edge's node numbers
        for reiijj in refQuad_EdgeIndeces:
            if reiijj[I] == 0:
                refQuad_EdgeNodes.append([
                    refQuad_NodeDict[(-1,reiijj[J])],
                    refQuad_NodeDict[( 1,reiijj[J])]])
            else:
                refQuad_EdgeNodes.append([
                    refQuad_NodeDict[(reiijj[I],-1)],
                    refQuad_NodeDict[(reiijj[I], 1)]])
        for i in self.qRange_x:
            for j in self.qRange_y:
                #quad number
                qN = i*self.qy + j
                #fine grid indeces of quad
                ii = 2*i + 1
                jj = 2*j + 1
                #nodes
                for rn,rniijj in enumerate(refQuad_NodeIndeces):
                    nii = rniijj[I] + ii
                    njj = rniijj[J] + jj
                    nN = (nii//2)*self.ny + njj//2
                    quadNodeNumbers[rn]=nN
                self.quadrilateralNodesArray[qN][:]=quadNodeNumbers
                #edges
                for re,reiijj in enumerate(refQuad_EdgeIndeces):
                    eii = reiijj[I] + ii
                    ejj = reiijj[J] + jj
                    eN = (eii//2)*self.eXYy + (eii%2)*self.eYy + ejj//2
                    quadEdgeNumbers[re]=eN
                    #nodes
                    for n,rn in enumerate(refQuad_EdgeNodes[re]):
                        self.edgeNodesArray[eN][n] = quadNodeNumbers[rn]
                self.quadrilateralEdgesArray[qN][:]=quadEdgeNumbers
        #Mesh interface (dimensions)
        self.nNodes_global=self.nxy
        self.nEdges_global=self.eXYxy
        self.nElements_global=self.qxy
        self.nElementBoundaries_global=self.eXYxy
        self.nodeArray=np.reshape(self.nodeGridArray,(self.nxy,3))
        self.elementNodesArray=self.quadrilateralNodesArray
        self.elementBoundariesArray=self.edgeNodesArray
        #todo extract boundary mesh

class RectangularGrid(Mesh):
    """A regular partition into rectangles.

    Nodes, edges, and faces can be indexed by (i,j,k) as follows.
    The edges and faces are divided according to orientation (i.e. x-edge...).
    An (i,j,k) index is associated with the type of edge or face
    having node (i,j,k) as the first node in a lexicographically sorted
    list of nodes corresponding to the edge or face."""
    def __init__(self,nx=1,ny=1,nz=1,Lx=1.0,Ly=1.0,Lz=1.0):
        Mesh.__init__(self)
        self.Lx = Lx
        self.Ly = Ly
        self.Lz = Lz
        #nodes
        self.nx=nx
        self.ny=ny
        self.nz=nz
        self.nxy = nx*ny;
        self.nxyz = nx*ny*nz;

        #edges
        self.nXex=nx-1 #number of x-edges in the x dimension
        self.nXey=ny
        self.nXez=nz

        self.nYex=nx
        self.nYey=ny-1
        self.nYez=nz

        self.nZex=nx
        self.nZey=ny
        self.nZez=nz-1

        #number of edges of all types associated with a row of nodes
        self.nXYZex = self.nXex + self.nYex + self.nZex

        #number of edges associated with an xy plane of nodes
        self.nXexy=self.nXex*self.nXey
        self.nYexy=self.nYex*self.nYey
        self.nZexy=self.nZex*self.nZey
        self.nXYZexy = self.nXexy + self.nYexy + self.nZexy

        #number of edges of each type in the grid
        self.nXexyz = self.nXexy*self.nXez
        self.nYexyz = self.nYexy*self.nYez
        self.nZexyz = self.nZexy*self.nZez

        #total number of edges
        self.nXYZexyz = self.nXexyz + self.nYexyz + self.nZexyz

        #quadrilaterals
        self.nXYhx=nx-1 #number of XY quadrilaterals in x-dimension
        self.nXYhy=ny-1
        self.nXYhz=nz

        self.nXZhx=nx-1
        self.nXZhy=ny
        self.nXZhz=nz-1

        self.nYZhx=nx
        self.nYZhy=ny-1
        self.nYZhz=nz-1

        #number of quadrilaterals of all types associate with a row of nodes
        self.nXY_XZ_YZhx =self.nXYhx + self.nXZhx + self.nYZhx

        #number of quadrilaterals associated with an xy plane of nodes
        self.nXYhxy=self.nXYhx*self.nXYhy
        self.nXZhxy=self.nXZhx*self.nXZhy
        self.nYZhxy=self.nYZhx*self.nYZhy
        self.nXY_XZ_YZhxy =self.nXYhxy + self.nXZhxy + self.nYZhxy

        #number of quadrilaterals of each type in the grid
        self.nXYhxyz = self.nXYhxy*self.nXYhz
        self.nXZhxyz = self.nXZhxy*self.nXZhz
        self.nYZhxyz = self.nYZhxy*self.nYZhz

        #total number of quadrilaterals
        self.nXY_XZ_YZhxyz =self.nXYhxyz + self.nXZhxyz + self.nYZhxyz

        #hexahedra
        self.nHx=nx-1
        self.nHy=ny-1
        self.nHz=nz-1
        self.nHxy = self.nHx*self.nHy
        self.nHxyz = self.nHxy*self.nHz

        #encode red and black
        self.black=0
        self.red=1

        #dimensions of hexahedra
        if self.nHx>0:
            hx = float(Lx)/(nx-1)
        else:
            hx = 1.0
        if self.nHy>0:
            hy = float(Ly)/(ny-1)
        else:
            hy=1.0
        if self.nHz>0:
            hz = float(Lz)/(nz-1)
        else:
            hz=1.0
        self.nodeDict={}
        self.xedgeDict={}
        self.yedgeDict={}
        self.zedgeDict={}
        self.xedgeList=[]
        self.yedgeList=[]
        self.zedgeList=[]
        self.XYQuadrilateralDict={}
        self.XZQuadrilateralDict={}
        self.YZQuadrilateralDict={}
        self.XYQuadrilateralList=[]
        self.XZQuadrilateralList=[]
        self.YZQuadrilateralList=[]
        self.hexahedronDict={}
        self.hexahedronList=[]
        self.nodeList=[]
        for k in range(self.nz):
            for j in range(self.ny):
                for i in range(self.nx):
                    n = self.getNodeNumber(i,j,k)
                    x = i*hx
                    y = j*hy
                    z = k*hz
                    self.nodeDict[(i,j,k)]=Node(n,x,y,z)
                    self.nodeList.append(self.nodeDict[(i,j,k)])
        for k in range(self.nXez):
            for j in range(self.nXey):
                for i in range(self.nXex):
                    en = self.getXEdgeNumber(i,j,k)
                    self.xedgeDict[(i,j,k)] = Edge(en,
                                                   [self.getNode(i,j,k),
                                                    self.getNode(i+1,j,k)])
                    self.xedgeList.append(self.xedgeDict[(i,j,k)])
        for k in range(self.nYez):
            for j in range(self.nYey):
                for i in range(self.nYex):
                    en = self.getYEdgeNumber(i,j,k)
                    self.yedgeDict[(i,j,k)] = Edge(en,
                                                   [self.getNode(i,j,k),
                                                    self.getNode(i,j+1,k)])
                    self.yedgeList.append(self.yedgeDict[(i,j,k)])
        for k in range(self.nZez):
            for j in range(self.nZey):
                for i in range(self.nZex):
                    en = self.getZEdgeNumber(i,j,k)
                    self.zedgeDict[(i,j,k)] = Edge(en,
                                                   [self.getNode(i,j,k),
                                                    self.getNode(i,j,k+1)])
                    self.zedgeList.append(self.zedgeDict[(i,j,k)])
        for k in range(self.nXYhz):
            for j in range(self.nXYhy):
                for i in range(self.nXYhx):
                    qn = self.getXYQuadrilateralNumber(i,j,k)
                    edges = [self.getXEdge(i,j,k),
                             self.getXEdge(i,j+1,k),
                             self.getYEdge(i,j,k),
                             self.getYEdge(i+1,j,k)]
                    self.XYQuadrilateralDict[(i,j,k)] = Quadrilateral(qn,edges)
                    self.XYQuadrilateralList.append(
                        self.XYQuadrilateralDict[(i,j,k)])
        for k in range(self.nXZhz):
            for j in range(self.nXZhy):
                for i in range(self.nXZhx):
                    qn = self.getXZQuadrilateralNumber(i,j,k)
                    edges = [self.getXEdge(i,j,k),
                             self.getXEdge(i,j,k+1),
                             self.getZEdge(i,j,k),
                             self.getZEdge(i+1,j,k)]
                    self.XZQuadrilateralDict[(i,j,k)] = Quadrilateral(qn,edges)
                    self.XZQuadrilateralList.append(
                        self.XZQuadrilateralDict[(i,j,k)])
        for k in range(self.nYZhz):
            for j in range(self.nYZhy):
                for i in range(self.nYZhx):
                    qn = self.getYZQuadrilateralNumber(i,j,k)
                    edges = [self.getYEdge(i,j,k),
                             self.getYEdge(i,j,k+1),
                             self.getZEdge(i,j,k),
                             self.getZEdge(i,j+1,k)]
                    self.YZQuadrilateralDict[(i,j,k)] = Quadrilateral(qn,edges)
                    self.YZQuadrilateralList.append(
                        self.YZQuadrilateralDict[(i,j,k)])
        for  k in range(self.nHz):
            for j in range(self.nHy):
                for i in range(self.nHx):
                    Hn = self.getHexahedronNumber(i,j,k)
                    quadrilaterals = [self.getXYQuadrilateral(i,j,k),
                                      self.getXYQuadrilateral(i,j,k+1),
                                      self.getXZQuadrilateral(i,j,k),
                                      self.getXZQuadrilateral(i,j+1,k),
                                      self.getYZQuadrilateral(i,j,k),
                                      self.getYZQuadrilateral(i+1,j,k)]
                    self.hexahedronDict[(i,j,k)] = Hexahedron(Hn,
                                                              quadrilaterals)
                    self.hexahedronList.append(self.hexahedronDict[(i,j,k)])
        #build lists for mesh base class
        self.edgeList = self.xedgeList + \
                        self.yedgeList + \
                        self.zedgeList
        #figure out if this is a 1D,2D, or 3D grid
        if self.nz > 1:
            self.elementList = self.hexahedronList
            self.elementDict = self.hexahedronDict
        elif self.ny > 1:
            self.elementList = self.XYQuadrilateralList
            self.elementDict = self.XYQuadrilateralDict
        else:
            self.elementList = self.xedgeList
            self.elementDict = self.xedgeDict
        #self.buildArraysFromLists()
        #todo: extract boundary mesh

    def getNodeNumber(self,i,j,k):
        return i + j*self.nx + k*self.nxy

    def getNode(self,i,j,k):
        return self.nodeDict[(i,j,k)]

    def getXEdgeNumber(self,ie,je,ke):
        return ie + je*self.nXex + ke*self.nXexy

    def getYEdgeNumber(self,ie,je,ke):
        return ie + je*self.nYex + ke*self.nYexy

    def getZEdgeNumber(self,ie,je,ke):
        return ie + je*self.nZex + ke*self.nZexy

    def getXEdge(self,ie,je,ke):
        return self.xedgeDict[(ie,je,ke)]

    def getYEdge(self,ie,je,ke):
        return self.yedgeDict[(ie,je,ke)]

    def getZEdge(self,ie,je,ke):
        return self.zedgeDict[(ie,je,ke)]

    def getXYQuadrilateralNumber(self,ih,jh,kh):
        return ih + jh*self.nXYhx + kh*self.nXYhxy

    def getXZQuadrilateralNumber(self,ih,jh,kh):
        return ih + jh*self.nXZhx + kh*self.nXZhxy

    def getYZQuadrilateralNumber(self,ih,jh,kh):
        return ih + jh*self.nYZhx + kh*self.nYZhxy

    def getXYQuadrilateral(self,ih,jh,kh):
        return self.XYQuadrilateralDict[(ih,jh,kh)]

    def getXZQuadrilateral(self,ih,jh,kh):
        return self.XZQuadrilateralDict[(ih,jh,kh)]

    def getYZQuadrilateral(self,ih,jh,kh):
        return self.YZQuadrilateralDict[(ih,jh,kh)]

    def getHexahedronNumber(self,iH,jH,kH):
        return iH + jH*self.nHx + kH*self.nHxy

    def getHexahedron(self,iH,jH,kH):
        return self.hexahedronDict[(iH,jH,kH)]

    def getColor(self,i,j,k):
        return (i%2 + j%2 + k%2)%2

    def refine(self,oldMesh,refineFactorX=2,refineFactorY=2,refineFactorZ=2):
        NX = oldMesh.nx
        NY = oldMesh.ny
        NZ = oldMesh.nz
        if NX > 1:
            NX = (NX-1)*refineFactorX + 1
        else:
            refineFactorX=1
        if NY > 1:
            NY = (NY-1)*refineFactorY + 1
        else:
            refineFactorY=1
        if NZ > 1:
            NZ = (NZ-1)*refineFactorZ + 1
        else:
            refineFactorZ=1
        RectangularGrid.__init__(self,NX,NY,NZ,
                                 oldMesh.Lx,oldMesh.Ly,oldMesh.Lz)
        childrenDict={}
        for IJK,e in oldMesh.elementDict.items():
            I = IJK[0]
            J = IJK[1]
            K = IJK[2]
            childrenDict[e.N]=[]
            for xOffset in range(refineFactorX):
                for yOffset in range(refineFactorY):
                    for zOffset in range(refineFactorZ):
                        i = I*refineFactorX + xOffset
                        j = J*refineFactorY + yOffset
                        k = K*refineFactorZ + zOffset
                        childrenDict[e.N].append(self.elementDict[(i,j,k)])
        return childrenDict

class MultilevelRectangularGrid(MultilevelMesh):
    """A hierarchical multilevel grid"""
    def __init__(self,levels,nx,ny=1,nz=1,
                 Lx=1.0,Ly=1.0,Lz=1.0,
                 refinementLevels=1):
        MultilevelMesh.__init__(self)
        self.refineFactorList=[EVec(0,0,0)]
        self.meshList.append(RectangularGrid(nx,ny,nz,Lx,Ly,Lz))
        self.elementChildren = []
        logEvent(self.meshList[0].meshInfo())
        for l in range(1,refinementLevels+1):
            self.refine()
            logEvent(self.meshList[-1].meshInfo())

    def refine():
        self.meshList.append(RectangularMesh())
        childrenDict = self.meshList[-1].refine(self.meshList[-2])
        self.elementChildren.append(childrenDict)


class TetrahedralMesh(Mesh):
    """A mesh of tetrahedra.

    The nodes, edges, triangles, and tetrahedra are indexed by their
    node tuples. The corresponding lists are derived from the dictionaries, and
    sorted lexicographically. The global node numbers are redefined to
    give a lexicographic ordering.

    The mesh can be generated from a rectangular grid and refined using either
    4T or Freudenthal-Bey global refinement.

    Attributes
    ----------
    elementNodesArray : array_like
        A list of lists storing the node values associated with each element 
        in the triangulation.  The first index refers to the element number,
        while the second index refers to the global node value.
    nodeArray : array_like
        A list of lists storing node coordinates.  The first index referes
        to the global node number, while the second index refers to the x, y
        and z coordinates of the node respectively.
    """

    def __init__(self):
        Mesh.__init__(self)
        self.nodeDict={}
        self.edgeDict={}
        self.triangleDict={}
        self.triangleList=[]
        self.tetrahedronDict={}
        self.tetrahedronList=[]
        self.oldToNewNode=[]
        self.boundaryMesh=TriangularMesh()
        self.nNodes_elementBoundary = 0
        self.rank = 0
        self.size = 1
        self.max_nNodeNeighbors_node = 0
    def meshType(self):
        return 'simplex'
    def computeGeometricInfo(self):
        from . import cmeshTools
        cmeshTools.computeGeometricInfo_tetrahedron(self.cmesh)
    def generateTetrahedralMeshFromRectangularGrid(self,nx,ny,nz,Lx,Ly,Lz):
        from . import cmeshTools
        self.cmesh = cmeshTools.CMesh()
        logEvent("Generating grid and mesh")
        cmeshTools.generateTetrahedralMeshFromRectangularGrid(nx,ny,nz,Lx,Ly,Lz,self.cmesh)
        logEvent("Allocating geometric info")
        cmeshTools.allocateGeometricInfo_tetrahedron(self.cmesh)
        logEvent("Computing geometric info")
        cmeshTools.computeGeometricInfo_tetrahedron(self.cmesh)
        self.buildFromC(self.cmesh)

    def rectangularToTetrahedral6T(self,grid):
        #copy the nodes from the rectangular mesh
        #I want to be able to renumber later without
        #changing the grid nodes, so I do deep copies here
        self.nodeList = [Node(n.N,n.p[X],n.p[Y],n.p[Z]) for n in grid.nodeList]
        self.nodeDict = dict([(n,n) for n in self.nodeList])
        for i in range(grid.nHx):
            for j in range(grid.nHy):
                for k in range(grid.nHz):
                    #associate the element (i,j,k) with the
                    #left, front, bottom node
                    #do a top down numbering to match Ong's dissertation
                    n1 = self.nodeList[grid.getNodeNumber(i,j,k+1)]
                    n2 = self.nodeList[grid.getNodeNumber(i,j+1,k+1)]
                    n3 = self.nodeList[grid.getNodeNumber(i+1,j+1,k+1)]
                    n4 = self.nodeList[grid.getNodeNumber(i+1,j,k+1)]
                    n5 = self.nodeList[grid.getNodeNumber(i,j,k)]
                    n6 = self.nodeList[grid.getNodeNumber(i,j+1,k)]
                    n7 = self.nodeList[grid.getNodeNumber(i+1,j+1,k)]
                    n8 = self.nodeList[grid.getNodeNumber(i+1,j,k)]
                    self.newTetrahedron(nodes=[n1,n2,n3,n6])
                    self.newTetrahedron(nodes=[n1,n3,n5,n6])
                    self.newTetrahedron(nodes=[n3,n5,n6,n7])
                    self.newTetrahedron(nodes=[n1,n3,n4,n5])
                    self.newTetrahedron(nodes=[n3,n4,n5,n7])
                    self.newTetrahedron(nodes=[n4,n5,n7,n8])
        self.finalize()

    def rectangularToTetrahedral5T(self,grid):
        #copy the nodes from the rectangular mesh
        #I want to be able to renumber later without
        #changing the grid nodes, so I do deep copies here
        self.nodeList = [Node(n.N,n.p[X],n.p[Y],n.p[Z]) for n in grid.nodeList]
        self.nodeDict = dict([(n,n) for n in self.nodeList])
        for i in range(grid.nHx):
            for j in range(grid.nHy):
                for k in range(grid.nHz):
                    #associate the element (i,j,k) with the
                    #left, front, bottom node
                    #get the left,front,bottom,node and its color
                    if (grid.getColor(i,j,k) == grid.black):
                        b0 = self.nodeList[grid.getNodeNumber(i,j,k)]
                        rx = self.nodeList[grid.getNodeNumber(i+1,j,k)]
                        ry = self.nodeList[grid.getNodeNumber(i,j+1,k)]
                        rz = self.nodeList[grid.getNodeNumber(i,j,k+1)]
                        r0 = self.nodeList[grid.getNodeNumber(i+1,j+1,k+1)]
                        bx = self.nodeList[grid.getNodeNumber(i,j+1,k+1)]
                        by = self.nodeList[grid.getNodeNumber(i+1,j,k+1)]
                        bz = self.nodeList[grid.getNodeNumber(i+1,j+1,k)]
                    else:
                        r0 = self.nodeList[grid.getNodeNumber(i,j,k)]
                        bx = self.nodeList[grid.getNodeNumber(i+1,j,k)]
                        by = self.nodeList[grid.getNodeNumber(i,j+1,k)]
                        bz = self.nodeList[grid.getNodeNumber(i,j,k+1)]
                        b0 = self.nodeList[grid.getNodeNumber(i+1,j+1,k+1)]
                        rx = self.nodeList[grid.getNodeNumber(i,j+1,k+1)]
                        ry = self.nodeList[grid.getNodeNumber(i+1,j,k+1)]
                        rz = self.nodeList[grid.getNodeNumber(i+1,j+1,k)]
                    self.newTetrahedron(nodes=[rx,by,bz,b0])
                    self.newTetrahedron(nodes=[ry,bz,bx,b0])
                    self.newTetrahedron(nodes=[rz,b0,bx,by])
                    self.newTetrahedron(nodes=[r0,bx,by,bz])
                    self.newTetrahedron(nodes=[b0,bx,by,bz])
        self.finalize()

    rectangularToTetrahedral = rectangularToTetrahedral6T

    def fixLocalNumbering(self):
        for TN in range(len(self.tetrahedronList)):
            self.tetrahedronList[TN].computeGeometricInfo()
            if edet(self.tetrahedronList[TN].linearMap) < 0:
                newNodes = list(self.tetrahedronList[TN].nodes)
                newNodes[2] = self.tetrahedronList[TN].nodes[1]
                newNodes[1] = self.tetrahedronList[TN].nodes[2]
                self.tetrahedronList[TN].nodes = newNodes
    def finalize(self):
        self.buildLists()
        #self.fixLocalNumbering()
        self.buildBoundaryMaps()
        self.buildArraysFromLists()
        self.hMax = 0.0
        self.hMin = 1.0e16
        self.sigmaMax = 0.0
        self.totalVolume = 0.0
        for T in self.tetrahedronList:
            T.computeGeometricInfo()
            self.hMax = max(T.diameter,self.hMax)
            self.hMin = min(T.diameter,self.hMin)
            self.sigmaMax = max(T.diameter/T.innerDiameter,self.sigmaMax)
            self.totalVolume += T.volume
    def buildLists(self):
        self.buildListsNodes()
        self.buildListsEdges()
        self.buildListsTriangles()
        self.buildListsTetrahedra()
        self.elementList = self.tetrahedronList
        self.elementBoundaryList = self.triangleList
    def buildListsNodes(self):
        keyList = list(self.nodeDict.keys())
        keyList.sort()
        self.nodeList=[]
        self.oldToNewNode=list(range(len(self.nodeDict)))
        for nN,k in enumerate(keyList):
            self.oldToNewNode[self.nodeDict[k].N]=nN
            self.nodeDict[k].N = nN
            self.nodeList.append(self.nodeDict[k])

    def buildListsEdges(self):
        keyList = list(self.edgeDict.keys())
        keyList.sort()
        self.edgeList=[]
        for eN,k in enumerate(keyList):
            self.edgeDict[k].N = eN
            self.edgeList.append(self.edgeDict[k])

    def buildListsTriangles(self):
        keyList = list(self.triangleDict.keys())
        keyList.sort()
        self.triangleList=[]
        for tN,k in enumerate(keyList):
            self.triangleDict[k].N = tN
            self.triangleList.append(self.triangleDict[k])
        self.polygonList = self.triangleList

    def buildListsTetrahedra(self):
        keyList = list(self.tetrahedronDict.keys())
        keyList.sort()
        self.tetrahedronList=[]
        for TN,k in enumerate(keyList):
            self.tetrahedronDict[k].N = TN
            self.tetrahedronList.append(self.tetrahedronDict[k])
        self.polyhedronList = self.tetrahedronList

    def buildBoundaryMaps(self):
        """
        Extract a mapping tn -> list((TN,tnLocal)) that
        provides all elements with the boundary face (triangle) tn
        and the local triangle number for that triangle.
        Likewise build mappings for edges and nodes
        Also extract a list of the triangles with only one associate
        element; these are the external boundary triangles. Then extract
        the edges and nodes from the boundary triangles.
        """
        self.triangleMap=[[] for t in self.triangleList]
        self.edgeMap=[[] for e in self.edgeList]
        self.nodeMap=[[] for n in self.nodeList]
        self.boundaryTriangles=set()
        self.interiorTriangles=set()
        self.boundaryEdges=set()
        self.boundaryNodes=set()
        self.interiorEdges=set()
        self.interiorNodes=set()
        logEvent("Building triangle,edge, and node maps")
        for T in self.tetrahedronList:
            for localTriangleNumber,t in enumerate(T.triangles):
                self.triangleMap[t.N].append((T.N,localTriangleNumber))
            for localEdgeNumber,e in enumerate(T.edges):
                self.edgeMap[e.N].append((T.N,localEdgeNumber))
            for localNodeNumber,n in enumerate(T.nodes):
                self.nodeMap[n.N].append((T.N,localNodeNumber))
        logEvent("Extracting boundary and interior triangles")
        for tN,etList in enumerate(self.triangleMap):
            if len(etList) == 1:
                self.boundaryTriangles.add(self.triangleList[tN])
            else:
                self.interiorTriangles.add(self.triangleList[tN])
        logEvent("Extracting boundary edges and nodes")
        for t in self.boundaryTriangles:
            self.boundaryEdges.update(t.edges)
            self.boundaryNodes.update(t.nodes)
        logEvent("Extracting interior edges and nodes")
        for t in self.interiorTriangles:
            self.interiorEdges.update(t.edges)
            self.interiorNodes.update(t.nodes)
        self.boundaryMesh.buildFromSets(self.boundaryTriangles,
                                        self.boundaryEdges,self.boundaryNodes)

    def newTetrahedron(self,nodes):
        T = Tetrahedron(tetrahedronNumber=len(self.tetrahedronDict),
                        nodes=nodes)
        self.tetrahedronDict[T.nodes] = T
        self.registerTriangles(T)
        return T

    def registerEdges(self,t):
        for en,e in enumerate(t.edges):
            if e.nodes in self.edgeDict:
                t.edges[en]=self.edgeDict[e.nodes]
            else:
                eN=len(self.edgeDict)
                e.N=eN
                self.edgeDict[e.nodes]=e

    def registerTriangles(self,T):
        for tn,t in enumerate(T.triangles):
            if t.nodes in self.triangleDict:
                T.triangles[tn]=self.triangleDict[t.nodes]
            else:
                t.N=len(self.triangleDict)
                self.triangleDict[t.nodes]=t
                self.registerEdges(t)

    def registerNode(self,node):
        if node in self.nodeDict:
            node = self.nodeDict[node]
        else:
            node.N = len(self.nodeDict)
            self.nodeDict[node] = node
        return node

    def readMeshADH(self,filename,adhBase=1):
        meshIn = open(filename+'.3dm','r')
        firstLine = meshIn.readline()
        firstWords = firstLine.split()
        logEvent("Reading object=%s from file=%s" % (firstWords[0],filename))
        line = meshIn.readline()
        columns = line.split()
        tets = []
        tetEdges=set()
        tetTriangles=set()
        logEvent("Reading "+str(filename)+" and building node lists for tetrahedra,triangles, and edges")
        #assume test are ordered by tet number
        while (columns[0] == 'E4T'):
            nodeNumbers = [int(c) - adhBase for c in columns[2:6]]
            nodeNumbers.sort()
            tets.append(array.array('i',nodeNumbers))
            tetTriangles.update([(nodeNumbers[1],nodeNumbers[2],nodeNumbers[3]),
                                 (nodeNumbers[0],nodeNumbers[2],nodeNumbers[3]),
                                 (nodeNumbers[0],nodeNumbers[1],nodeNumbers[3]),
                                 (nodeNumbers[0],nodeNumbers[1],nodeNumbers[2])])
            tetEdges.update([(nodeNumbers[0],nodeNumbers[1]),
                             (nodeNumbers[0],nodeNumbers[2]),
                             (nodeNumbers[0],nodeNumbers[3]),
                             (nodeNumbers[1],nodeNumbers[2]),
                             (nodeNumbers[1],nodeNumbers[3]),
                             (nodeNumbers[2],nodeNumbers[3])])
            line = meshIn.readline()
            columns = line.split()
        print("Building node list and dict")
        #assume nodes are ordered by node number
        while (len(columns) == 5):
            newNode = Node(int(columns[1]) - adhBase,
                           float(columns[2]),
                           float(columns[3]),
                           float(columns[4]))
            self.nodeList.append(newNode)
            self.nodeDict[newNode]=newNode
            line = meshIn.readline()
            columns = line.split()
        print("Number of tetrahedra:"+str(len(tets)))
        print("Number of triangles :"+str(len(tetTriangles)))
        print("Number of edges     :"+str(len(tetEdges)))
        print("Number of nodes     :"+str(len(self.nodeList)))
        print("Number of objects   :"+str(len(tetEdges)+len(tetTriangles)+len(tets)+len(self.nodeList)))
        print("Building edge list")
        self.edgeList =[Edge(edgeNumber=eN,nodes=[self.nodeList[nN[0]],self.nodeList[nN[1]]]) \
                        for eN,nN in enumerate(tetEdges)]
        print("Building edge dict")
        self.edgeDict = dict([(e.nodes,e) for e in self.edgeList])
        print("Building triangle list")
        self.triangleList =[Triangle(triangleNumber=tN,nodes=[self.nodeList[nN[0]],self.nodeList[nN[1]],self.nodeList[nN[2]]],edgeDict=self.edgeDict) \
                            for tN,nN in enumerate(tetTriangles)]
        print("Building triangle dict")
        self.triangleDict = dict([(t.nodes,t) for t in self.triangleList])
        print("Building tetredron list")
        self.tetrahedronList = [Tetrahedron(tetrahedronNumber=TN,
                                            nodes=[self.nodeList[nN[0]],self.nodeList[nN[1]],self.nodeList[nN[2]],self.nodeList[nN[3]]],
                                            edgeDict=self.edgeDict,
                                            triangleDict=self.triangleDict) \
                                for TN,nN in enumerate(tets)]
        self.elementList = self.tetrahedronList
        self.elementBoundaryList = self.triangleList
        print("Building tetrahedron dict")
        self.tetrahedronDict = dict([(T.nodes,T) for T in self.tetrahedronList])
        print("Building boundary maps")
        self.buildBoundaryMaps()
    def writeMeshXdmf(self,ar,name='',t=0.0,init=False,meshChanged=False,tCount=0, EB=False):
        #print "Warning mwf hack for EB printing for tet writeMeshXdmf for now"
        #EB = True
        Mesh.writeMeshXdmf(self,ar,name,t,init,meshChanged,"Tetrahedron",tCount,EB=EB)
    def writeMeshEnsight(self,filename,description=None):
        base=1
        #write the casefile
        caseOut=open(filename+'.case','w')
        caseOut.write('FORMAT\n'+'type: ensight gold\n')
        caseOut.write('GEOMETRY\n'+'model: '+filename+'.geo\n')
        caseOut.close()
        meshOut=open(filename+'.geo','w')
        meshOut.write('Ensight Gold\n')
        meshOut.write('Unstructured Tetrahedral Mesh\n')
        meshOut.write('node id given\n')
        meshOut.write('element id given\n')
        #extents = 'extents\n %12.5E %12.5E\n %12.5E %12.5E\n %12.5E %12.5E\n' % (self.xmin,self.xmax,self.ymin,self.ymax,self.zmin,self.zmax)
        #meshOut.write('extents\n'+`self.xmin`+' '+`self.xmax`+'\n')
        meshOut.write('part \n'+'%10i\n' % 1)
        if description:
            meshOut.write(description+'\n')
        else:
            meshOut.write('A Mesh\n')
        meshOut.write('coordinates\n'+'%10i\n' % self.nNodes_global)
        for nN in range(self.nNodes_global):
            meshOut.write('%10i\n' % (nN+base))
        for nN in range(self.nNodes_global):
            meshOut.write('%12.5E\n' % self.nodeArray[nN,0])
        for nN in range(self.nNodes_global):
            meshOut.write('%12.5E\n' % self.nodeArray[nN,1])
        for nN in range(self.nNodes_global):
            meshOut.write('%12.5E\n' % self.nodeArray[nN,2])
        meshOut.write('tetra4\n'+'%10i\n' % self.nElements_global)
        for eN in range(self.nElements_global):
            meshOut.write('%10i\n' % (eN+base))
        for eN in range(self.nElements_global):
            meshOut.write('%10i%10i%10i%10i\n' % tuple((nN+base) for nN in self.elementNodesArray[eN,:]))
        meshOut.close()

    def appendMeshEnsight(self,filename,description=None):
        base=1
        #write the casefile
        caseOut=open(filename+'.case','a')
        caseOut.write('GEOMETRY\n'+'model: '+filename+'.geo\n')
        caseOut.close()
        meshOut=open(filename+'.geo','w')
        meshOut.write('Unstructured Tetrahedral Mesh\n\n')
        meshOut.write('node id given\n')
        meshOut.write('element id given\n')
        #extents = 'extents\n %12.5E %12.5E\n %12.5E %12.5E\n %12.5E %12.5E\n' % (self.xmin,self.xmax,self.ymin,self.ymax,self.zmin,self.zmax)
        #meshOut.write('extents\n'+`self.xmin`+' '+`self.xmax`+'\n')
        meshOut.write('part \n'+'%10i\n' % 1)
        if description:
            meshOut.write(description+'\n')
        else:
            meshOut.write("A Mesh")
        meshOut.write('coordinates\n'+'%10i\n' % len(self.nodeList))
        for n in self.nodeList:
            nN = n.N+base
            meshOut.write('%10i\n' % nN)
        for n in self.nodeList:
            meshOut.write('%12.5E\n' % n.p[X])
        for n in self.nodeList:
            meshOut.write('%12.5E\n' % n.p[Y])
        for n in self.nodeList:
            meshOut.write('%12.5E\n' % n.p[Z])
        meshOut.write('tetra4\n'+'%10i\n' % len(self.elementList))
        for e in self.elementList:
            eN = e.N + base
            meshOut.write('%10i\n' % eN)
        for e in self.elementList:
            meshOut.write('%10i%10i%10i%10i\n' % tuple(n.N+base for n in e.nodes))
        meshOut.close()

    def writeMeshADH(self,filename,adhBase=1):
        from . import cmeshTools
        cmeshTools.write3dmFiles(self.cmesh,filename,adhBase)

    def writeBoundaryFacesADH(self,filename,adhBase=1):
        boundaryFacesOut=open(filename,'w')
        for t in self.boundaryTriangles:
            TN = self.triangleMap[t.N][0][0]
            T = self.tetrahedronList[TN]
            localFaceNumber = self.triangleMap[t.N][0][1]
            T.computeGeometricInfo()
            DJ = edet(T.linearMap)
            if DJ < 0:
                #print "Negative determinant ="+`DJ`+" Swapping two nodes"
                newNodes = list(T.nodes)
                newNodes[3] = T.nodes[2]
                newNodes[2] = T.nodes[3]
                newBasis = [n - newNodes[0] for n in newNodes[1:]]
                newMap = ETen(newBasis[0],newBasis[1],newBasis[2])
                #print "New Determinant "+`edet(newMap)`
                if localFaceNumber == T.nodes[2]:
                    localFaceNumber = T.nodes[3]
                elif localFaceNumber == T.nodes[3]:
                    localFaceNumber = T.nodes[2]
            line = 'FCS %5i %5i %5i' % \
                          (T.N + adhBase,
                           localFaceNumber + adhBase,
                           1)
            #print line
            boundaryFacesOut.write(line+'\n')
        boundaryFacesOut.close()

    def writeBoundaryNodesADH(self,filename,adhBase=1):
        boundaryNodesOut=open(filename,'w')
        for n in self.boundaryNodes:
            line = 'NDS %5i %5i' % \
                          (n.N + adhBase,
                           1)
            #print line
            boundaryNodesOut.write(line+'\n')
        boundaryNodesOut.close()

    def refine4T(self,oldMesh):
        childrenDict={}
        for T in oldMesh.tetrahedronList:
            #deep copy old nodes because we'll renumber
            TNodes = [Node(eN,n.p[X],n.p[Y],n.p[Z]) for eN,n in enumerate(T.nodes)]
            for lnN,n in enumerate(TNodes): TNodes[lnN]=self.registerNode(n)
            #add new node
            T.computeGeometricInfo()
            newNode = Node(len(self.nodeDict),
                           T.barycenter[X],
                           T.barycenter[Y],
                           T.barycenter[Z])
            newNode = self.registerNode(newNode)
            T1=self.newTetrahedron([TNodes[0],TNodes[1],TNodes[2],newNode])
            T2=self.newTetrahedron([TNodes[1],TNodes[2],TNodes[3],newNode])
            T3=self.newTetrahedron([TNodes[2],TNodes[3],TNodes[0],newNode])
            T4=self.newTetrahedron([TNodes[3],TNodes[0],TNodes[1],newNode])
            childrenDict[T.N]=[T1,T2,T3,T4]
        self.finalize()
        return childrenDict

    def refineFreudenthalBey(self,oldMesh):
        logEvent("Refining the mesh using Freudenthal-Bey refinement")
        childrenDict={}
        for T in list(oldMesh.tetrahedronDict.values()):
            #deep copy old nodes because we'll renumber
            TNodes = [Node(nN,n.p[X],n.p[Y],n.p[Z]) for nN,n in \
                      enumerate(T.nodes)]
            for lnN,n in enumerate(TNodes): TNodes[lnN]=self.registerNode(n)
            #add new nodes (midpoints of edges)
            #use local edge tuples as keys
            newNodes={}
            for et,en in T.edgeMap.items():
                T.edges[en].computeGeometricInfo()
                p = T.edges[en].barycenter
                newNodes[et] = Node(en,p[X],p[Y],p[Z])

            #set the global node numbers
            for k,n in newNodes.items(): newNodes[k]=self.registerNode(n)
            #add corner tets
            T1=self.newTetrahedron([TNodes[0],
                                 newNodes[(0,1)],
                                 newNodes[(0,2)],
                                 newNodes[(0,3)]])
            T2=self.newTetrahedron([TNodes[1],
                                 newNodes[(0,1)],
                                 newNodes[(1,2)],
                                 newNodes[(1,3)]])
            T3=self.newTetrahedron([TNodes[2],
                                 newNodes[(0,2)],
                                 newNodes[(1,2)],
                                 newNodes[(2,3)]])
            T4=self.newTetrahedron([TNodes[3],
                                 newNodes[(0,3)],
                                 newNodes[(1,3)],
                                 newNodes[(2,3)]])
            #add center tets
            #choose the shortest diagonal of the octahedron
            dLengths = [enorm(newNodes[(0,1)].p-newNodes[(2,3)].p),
                        enorm(newNodes[(0,2)].p-newNodes[(1,3)].p),
                        enorm(newNodes[(0,3)].p-newNodes[(1,2)].p)]
            shortestEdgeLength = min(dLengths)
            if shortestEdgeLength == dLengths[0]:
                #diagonal (0,1)(2,3)
                T5=self.newTetrahedron([newNodes[(0,1)],
                                        newNodes[(2,3)],
                                        newNodes[(0,3)],
                                        newNodes[(1,3)]])
                T6=self.newTetrahedron([newNodes[(0,1)],
                                        newNodes[(2,3)],
                                        newNodes[(0,3)],
                                        newNodes[(0,2)]])
                T7=self.newTetrahedron([newNodes[(0,1)],
                                        newNodes[(2,3)],
                                        newNodes[(0,2)],
                                        newNodes[(1,2)]])
                T8=self.newTetrahedron([newNodes[(0,1)],
                                        newNodes[(2,3)],
                                        newNodes[(1,2)],
                                        newNodes[(1,3)]])
            elif shortestEdgeLength == dLengths[1]:
                #diagonal (0,2)(1,3)
                T5=self.newTetrahedron([newNodes[(0,2)],
                                        newNodes[(1,3)],
                                        newNodes[(0,3)],
                                        newNodes[(2,3)]])
                T6=self.newTetrahedron([newNodes[(0,2)],
                                        newNodes[(1,3)],
                                        newNodes[(2,3)],
                                        newNodes[(1,2)]])
                T7=self.newTetrahedron([newNodes[(0,2)],
                                        newNodes[(1,3)],
                                        newNodes[(1,2)],
                                        newNodes[(0,1)]])
                T8=self.newTetrahedron([newNodes[(0,2)],
                                        newNodes[(1,3)],
                                        newNodes[(0,1)],
                                        newNodes[(0,3)]])
            else:
                #diagonal (0,3)(1,2)
                T5=self.newTetrahedron([newNodes[(0,3)],
                                        newNodes[(1,2)],
                                        newNodes[(0,1)],
                                        newNodes[(1,3)]])
                T6=self.newTetrahedron([newNodes[(0,3)],
                                        newNodes[(1,2)],
                                        newNodes[(1,3)],
                                        newNodes[(2,3)]])
                T7=self.newTetrahedron([newNodes[(0,3)],
                                        newNodes[(1,2)],
                                        newNodes[(2,3)],
                                        newNodes[(0,2)]])
                T8=self.newTetrahedron([newNodes[(0,3)],
                                        newNodes[(1,2)],
                                        newNodes[(0,2)],
                                        newNodes[(0,1)]])
            childrenDict[T.N]=[T1,T2,T3,T4,T5,T6,T7,T8]
        self.finalize()
        return childrenDict
        #for debugging: print each tet
        #self.edgeList=[]
        #Tlist = self.tetrahedronDict.values()
        #for T in Tlist:
        #    self.edgeList = self.edgeList + T.edges

    def refine(self,oldMesh):
        return self.refineFreudenthalBey(oldMesh)

    def generateFromPlex(self, plex, parallel=False):
        # from petsc4py import PETSc
        from . import Comm
        comm = Comm.comm.mpi4py_comm
        self.rank = comm.rank
        self.size = comm.size
        if self.rank == 0: 
            self.plexMesh = True

            # PETSc numbering convension in 3D : cells -> vertices -> faces -> edges
            # But the point range can be accessed using HeightStratum; height 0 = cells, 1 = faces, 2 = edges, and 3 = vertices.
            
            # Proteus convension: 
            # nNodes             = vertices
            # nElements          = cells
            # nelementBoundaries = faces
            # nEdges             = edges
            dim = plex.getDimension()
            
            # ToDo - conditions for 1D and 2D         
            if dim == 3:
                self.nNodes_element = 4 # num nodes per cell
                self.nNodes_elementBoundary = 3 # num nodes per face
                self.nElementBoundaries_element = 4 # num faces per cell
            
            cStart, cEnd = plex.getHeightStratum(0) # Total cells
            fStart, fEnd = plex.getHeightStratum(1) # Total faces 
            eStart, eEnd = plex.getHeightStratum(2) # Total edges
            nStart, nEnd = plex.getHeightStratum(3) # Total nodes


            self.nElements_global           = cEnd - cStart
            self.nElementBoundaries_global  = fEnd - fStart
            self.nEdges_global              = eEnd - eStart
            self.nNodes_global              = nEnd - nStart
            
            # get coordinate array from PETSc Vec TODO - get this working for -dm_refine 0 and -dm_refine >0
            # coordinatesVec = plex.getCoordinatesLocal() # Works with -dm_refine >0 only
            coordinatesVec = plex.getCoordinates() # Works with -dm_refine 0 only
            
            self.nodeArray = coordinatesVec.array.reshape(self.nNodes_global, dim)

            # All the entities have global numbering starting from 0
            # So to convert PETSc numering to Proteus numbering we subtract the total number of entities below in DAG from the numbering of the required entity. For instance, In the cube with 6 tetrahedrons, we convert the node numering from plex [6,14) by subtracting total numer of enetities below, which is 6 cells (Recall DAG sequence mentioned above).
            
            
            self.elementNodesArray = np.empty((self.nElements_global, self.nNodes_element), dtype=np.int32)
            # for i in range(self.nElements_global):
            #     self.elementNodesArray[i] = plex.getTransitiveClosure(i)[0][-self.nNodes_element:] - self.nElements_global
            
            # Loop over each node and get a transitive closure without cone and get the number of entries less than the number of total cells in the array to get the nodeElementOffsets and use those entry to populate  nodeElementsArray 
            elementOffsetList = [0]
            starOffsetList = [0]
            self.nodeElementsArray = np.array([], dtype=np.int32)
            self.nodeStarArray = np.array([], dtype=np.int32)
            self.nodeMaterialTypes = np.array([], dtype=np.int32)
            self.max_nNodeNeighbors_node = 0
            elementOffset = 0
            starOffset = 0
            logEvent("Loop 1")
            for i in np.arange(nStart, nEnd):
                self.nodeMaterialTypes = np.append(self.nodeMaterialTypes, plex.getCellTypeLabel().getValue(i))
                
                nAdjacentElements = int(sum(plex.getAdjacency(i) < self.nElements_global))
                nAdjacentNodes = int(sum((plex.getAdjacency(i)<nEnd) &  (plex.getAdjacency(i)>=nStart)) - 1)
                
                # Update nodeElements
                self.nodeElementsArray = np.concatenate((self.nodeElementsArray, plex.getTransitiveClosure(i, useCone=False)[0][-nAdjacentElements:]))
                # Update node Star
                nodeAdjacencyBoolArray = (plex.getAdjacency(i)<nEnd) &  (plex.getAdjacency(i)>=nStart)
                self.nodeStarArray = np.concatenate((self.nodeStarArray, plex.getAdjacency(i)[nodeAdjacencyBoolArray][1:] - nStart))
                # Update max_nNodeNeighbors_node
                self.max_nNodeNeighbors_node = nAdjacentNodes if nAdjacentNodes > self.max_nNodeNeighbors_node else self.max_nNodeNeighbors_node
                
                # Update offset
                elementOffset += nAdjacentElements
                starOffset += nAdjacentNodes
                elementOffsetList.append(elementOffset)
                starOffsetList.append(starOffset)        
            self.nodeElementOffsets = np.array(elementOffsetList, dtype=np.int32)
            self.nodeStarOffsets = np.array(starOffsetList, dtype=np.int32)

            # Get the cone of each cell to get the faces
            self.elementBoundariesArray = np.empty((self.nElements_global, self.nElementBoundaries_element), dtype=np.int32)
            self.elementMaterialTypes = np.array([], dtype=np.int32)
            logEvent("Loop 2")
            for i in np.arange(cStart, cEnd):
                elementNodesArrayUnsorted = np.array(plex.getTransitiveClosure(i)[0][-self.nNodes_element:] - self.nElements_global)
                # self.elementNodesArray[i] = plex.getTransitiveClosure(i)[0][-self.nNodes_element:] - self.nElements_global
                self.elementMaterialTypes = np.append(self.elementMaterialTypes, plex.getCellTypeLabel().getValue(i))
                # Adjust the ordering of faces to match the Proteus convention. 
                
                # print("elementNodesArrayUnsorted", elementNodesArrayUnsorted)
                self.elementBoundariesArray[i] = plex.getCone(i) - (self.nNodes_global + self.nElements_global)
                elementNodeList = []
                elementBoundaryNodesArrayTemp = np.empty((self.nElementBoundaries_element, self.nNodes_elementBoundary), dtype=np.int32)
                j=0
                for face in plex.getCone(i):
                    elementBoundaryNodesArrayTemp[j] = plex.getTransitiveClosure(face)[0][- self.nNodes_elementBoundary:] - self.nElements_global
                    # print(face, elementBoundaryNodesArrayTemp[j])
                    mask = np.where(~np.isin(elementNodesArrayUnsorted, elementBoundaryNodesArrayTemp[j]))[0]
                    index = int(elementNodesArrayUnsorted[mask])
                    # print(mask, index)
                    elementNodeList.append(index)
                    j += 1
                # print("elementNodeList", elementNodeList)
                self.elementNodesArray[i] = np.array(elementNodeList, dtype=np.int32)
                
            
            # Loop over each face to get the transitive closure and get the nodes
            # self.elementBoundaryNodesArray = np.empty((self.nElementBoundaries_global, self.nNodes_elementBoundary), dtype=np.int32)
            # self.elementBoundaryMaterialTypes = np.array([], dtype=np.int32)
            # for i in np.arange(fStart, fEnd):
                # self.elementBoundaryMaterialTypes = np.append(self.elementBoundaryMaterialTypes, plex.getCellTypeLabel().getValue(i))
                # self.elementBoundaryNodesArray[i - fStart] = plex.getTransitiveClosure(i)[0][- self.nNodes_elementBoundary:]  - self.nElements_global
            
            # loop over faces to get the support. Add -1 as a second entry gfor the boundary faces.
            self.elementBoundaryElementsArray = np.empty((self.nElementBoundaries_global, 2), dtype=np.int32)
            self.interiorElementBoundariesArray = np.array([], dtype=np.int32)
            self.exteriorElementBoundariesArray = np.array([], dtype=np.int32)
            self.elementBoundaryMaterialTypes = np.array([], dtype=np.int32)
            logEvent("Loop 3")
            for i in np.arange(fStart, fEnd):
                self.elementBoundaryMaterialTypes = np.append(self.elementBoundaryMaterialTypes, plex.getCellTypeLabel().getValue(i))
                if np.size(plex.getSupport(i)) == 1:
                    self.elementBoundaryElementsArray[i - fStart] = np.insert(plex.getSupport(i), 1, -1)
                    self.exteriorElementBoundariesArray = np.concatenate((self.exteriorElementBoundariesArray, np.array([i - fStart])))
                else:
                    self.elementBoundaryElementsArray[i - fStart] = plex.getSupport(i)
                    self.interiorElementBoundariesArray = np.concatenate((self.interiorElementBoundariesArray, np.array([i - fStart])))
            
            self.nInteriorElementBoundaries_global = np.size(self.interiorElementBoundariesArray)
            self.nExteriorElementBoundaries_global = np.size(self.exteriorElementBoundariesArray)
            
            # Get the cone of each edge to get the list of nodes 
            self.edgeNodesArray = np.empty((self.nEdges_global, 2), dtype=np.int32)
            logEvent("Loop 4")
            for i in np.arange(eStart, eEnd):
                self.edgeNodesArray[i - eStart] = plex.getCone(i) - self.nElements_global
            
            
            self.elementNeighborsArray = np.empty((self.nElements_global, self.nElementBoundaries_global), dtype=np.int32)
            
            logEvent("Done generating mesh from DMPlex")

            # ToDo stuff
            self.h = 0.0
            self.hMin = 0.0
            self.sigmaMax = 0.0
            self.volume   = 0.0            
        else:
            pass
        
        # bcast data to all processes
        self.nElements_global = comm.bcast(self.nElements_global, root=0)
        self.nNodes_global = comm.bcast(self.nNodes_global, root=0)
        self.nNodes_element = comm.bcast(self.nNodes_element, root=0)
        self.elementNodesArray = comm.bcast(self.elementNodesArray, root=0)
        self.nodeMaterialTypes = comm.bcast(self.nodeMaterialTypes, root=0)
        self.nodeArray = comm.bcast(self.nodeArray, root=0)
        
        self.nNodes_elementBoundary = comm.bcast(self.nNodes_elementBoundary, root=0)
        self.nEdges_global = comm.bcast(self.nEdges_global, root=0)
        self.nElementBoundaries_element = comm.bcast(self.nElementBoundaries_element, root=0)
        self.nElementBoundaries_global = comm.bcast(self.nElementBoundaries_global, root=0)
        self.nInteriorElementBoundaries_global = comm.bcast(self.nInteriorElementBoundaries_global, root=0)
        self.nExteriorElementBoundaries_global = comm.bcast(self.nExteriorElementBoundaries_global, root=0)
        self.max_nNodeNeighbors_node = comm.bcast(self.max_nNodeNeighbors_node, root=0)
        # self.max_nElements_node = comm.bcast(self.max_nElements_node, root=0)
        
        self.nodeElementsArray = comm.bcast(self.nodeElementsArray, root=0)
        self.nodeElementOffsets = comm.bcast(self.nodeElementOffsets, root=0)
        self.elementBoundariesArray = comm.bcast(self.elementBoundariesArray, root=0)
        self.elementBoundaryNodesArray = comm.bcast(self.elementBoundaryNodesArray, root=0)
        self.elementBoundaryElementsArray = comm.bcast(self.elementBoundaryElementsArray, root=0)
        self.interiorElementBoundariesArray = comm.bcast(self.interiorElementBoundariesArray, root=0)
        self.exteriorElementBoundariesArray = comm.bcast(self.exteriorElementBoundariesArray, root=0)
        self.edgeNodesArray = comm.bcast(self.edgeNodesArray, root=0)
        self.nodeStarArray = comm.bcast(self.nodeStarArray, root=0)
        self.nodeStarOffsets = comm.bcast(self.nodeStarOffsets, root=0)
        
        
        self.elementMaterialTypes = comm.bcast(self.elementMaterialTypes, root=0)
        self.elementBoundaryMaterialTypes = comm.bcast(self.elementBoundaryMaterialTypes, root=0)
        
        # self.h = comm.bcast(self.h, root=0)
        # self.hMin = comm.bcast(self.hMin, root=0)
        # self.sigmaMax = comm.bcast(self.sigmaMax, root=0)
        # self.volume = comm.bcast(self.volume, root=0)
        comm.barrier()
        logEvent("Passing DMPlex to cMeshTools to generate CMesh")
        from . import cmeshTools
        self.cmesh = cmeshTools.CMesh()
        self.buildCMeshFromPlex(self.cmesh)
    
        
        
    def generateFromTetgenFiles(self,filebase,base,skipGeometricInit=False,parallel=False):
        from . import cmeshTools
        logEvent(memory("declaring CMesh"),level=4)
        self.cmesh = cmeshTools.CMesh()
        logEvent(memory("Initializing CMesh"),level=4)
        if parallel:
            cmeshTools.generateFromTetgenFilesParallel(self.cmesh,filebase,base)
        else:
            cmeshTools.generateFromTetgenFiles(self.cmesh,filebase,base)
        logEvent(memory("calling cmeshTools.generateFromTetgenFiles","cmeshTools"),level=4)
        if skipGeometricInit == False:
            cmeshTools.allocateGeometricInfo_tetrahedron(self.cmesh)
            cmeshTools.computeGeometricInfo_tetrahedron(self.cmesh)
        self.buildFromC(self.cmesh)
        logEvent(memory("calling buildFromC"),level=4)
    def generateFrom3DMFile(self,filebase,base=1):
        from . import cmeshTools
        self.cmesh = cmeshTools.CMesh()
        cmeshTools.generateFrom3DMFile(self.cmesh,filebase,base)
        cmeshTools.allocateGeometricInfo_tetrahedron(self.cmesh)
        cmeshTools.computeGeometricInfo_tetrahedron(self.cmesh)
        self.buildFromC(self.cmesh)
    def writeTetgenFiles(self,filebase,base):
        from . import cmeshTools
        cmeshTools.writeTetgenFiles(self.cmesh,filebase,base)
    def meshInfo(self):
        minfo = """Number of tetrahedra : %d
Number of triangles  : %d
Number of edges      : %d
Number of nodes      : %d
max(sigma_k)         : %f
min(h_k)             : %f\n""" % (self.nElements_global,
                                  self.nElementBoundaries_global,
                                  self.nEdges_global,
                                  self.nNodes_global,
                                  self.sigmaMax,
                                  self.hMin)
        if self.subdomainMesh != self:
            sinfo = self.subdomainMesh.meshInfo()
            info = "*** Global ***\n" + minfo + "\n*** Local ***\n" + sinfo
            return info
        return minfo

class HexahedralMesh(Mesh):
    """A mesh of hexahedra.

    """

    def __init__(self):
        Mesh.__init__(self)
        self.nodeDict={}
        self.edgeDict={}
        self.faceDict={}
        self.faceList=[]
        self.elemDict={}
        self.elemList=[]
        self.oldToNewNode=[]
        self.boundaryMesh=QuadrilateralMesh()
    def meshType(self):
        return 'cuboid'
    def computeGeometricInfo(self):
        from . import cmeshTools
        print("no info yet for hexahedral mesh")
        #cmeshTools.computeGeometricInfo_tetrahedron(self.cmesh)
    def generateHexahedralMeshFromRectangularGrid(self,nx,ny,nz,Lx,Ly,Lz):
        from . import cmeshTools
        self.cmesh = cmeshTools.CMesh()
        cmeshTools.generateHexahedralMeshFromRectangularGrid(nx,ny,nz,0,0,0,Lx,Ly,Lz,self.cmesh)
        cmeshTools.allocateGeometricInfo_hexahedron(self.cmesh)
        cmeshTools.computeGeometricInfo_hexahedron(self.cmesh)
        self.buildFromC(self.cmesh)

    def finalize(self):
        self.buildLists()
        #self.fixLocalNumbering()
        self.buildBoundaryMaps()
        self.buildArraysFromLists()
        self.hMax = 0.0
        self.hMin = 1.0e16
        self.sigmaMax = 0.0
        self.totalVolume = 0.0
        for T in self.tetrahedronList:
            T.computeGeometricInfo()
            self.hMax = max(T.diameter,self.hMax)
            self.hMin = min(T.diameter,self.hMin)
            self.sigmaMax = max(T.diameter/T.innerDiameter,self.sigmaMax)
            self.totalVolume += T.volume

    def buildLists(self):
        self.buildListsNodes()
        self.buildListsEdges()
        self.buildListsFaces()
        self.buildListsElems()
        self.elementList = self.elemList
        self.elementBoundaryList = self.faceList
    def buildListsNodes(self):
        keyList = list(self.nodeDict.keys())
        keyList.sort()
        self.nodeList=[]
        self.oldToNewNode=list(range(len(self.nodeDict)))
        for nN,k in enumerate(keyList):
            self.oldToNewNode[self.nodeDict[k].N]=nN
            self.nodeDict[k].N = nN
            self.nodeList.append(self.nodeDict[k])

    def buildListsEdges(self):
        keyList = list(self.edgeDict.keys())
        keyList.sort()
        self.edgeList=[]
        for eN,k in enumerate(keyList):
            self.edgeDict[k].N = eN
            self.edgeList.append(self.edgeDict[k])

    def buildListsFaces(self):
        keyList = list(self.faceDict.keys())
        keyList.sort()
        self.triangleList=[]
        for tN,k in enumerate(keyList):
            self.faceDict[k].N = tN
            self.faceList.append(self.faceDict[k])
        self.polygonList = self.faceList

    def buildListsElems(self):
        keyList = list(self.elemDict.keys())
        keyList.sort()
        self.elemList=[]
        for TN,k in enumerate(keyList):
            self.elemDict[k].N = TN
            self.elemList.append(self.elemDict[k])
        self.polyhedronList = self.elemList

    def buildBoundaryMaps(self):
        """
        Extract a mapping tn -> list((TN,tnLocal)) that
        provides all elements with the boundary face  tn
        and the local triangle number for that face
        Likewise build mappings for edges and nodes
        Also extract a list of the triangles with only one associate
        element; these are the external boundary triangles. Then extract
        the edges and nodes from the boundary triangles.
        """
        self.faceMap=[[] for t in self.faceList]
        self.edgeMap=[[] for e in self.edgeList]
        self.nodeMap=[[] for n in self.nodeList]
        self.boundaryTriangles=set()
        self.interiorTriangles=set()
        self.boundaryEdges=set()
        self.boundaryNodes=set()
        self.interiorEdges=set()
        self.interiorNodes=set()
        logEvent("Building triangle,edge, and node maps")
        for T in self.elemList:
            for localFaceNumber,t in enumerate(T.faces):
                self.faceMap[t.N].append((T.N,localFaceNumber))
            for localEdgeNumber,e in enumerate(T.edges):
                self.edgeMap[e.N].append((T.N,localEdgeNumber))
            for localNodeNumber,n in enumerate(T.nodes):
                self.nodeMap[n.N].append((T.N,localNodeNumber))
        logEvent("Extracting boundary and interior triangles")
        for tN,etList in enumerate(self.faceMap):
            if len(etList) == 1:
                self.boundaryFaces.add(self.faceList[tN])
            else:
                self.interiorFaces.add(self.faceList[tN])
        logEvent("Extracting boundary edges and nodes")
        for t in self.boundaryTriangles:
            self.boundaryEdges.update(t.edges)
            self.boundaryNodes.update(t.nodes)
        logEvent("Extracting interior edges and nodes")
        for t in self.interiorTriangles:
            self.interiorEdges.update(t.edges)
            self.interiorNodes.update(t.nodes)
        self.boundaryMesh.buildFromSets(self.boundaryFaces,
                                        self.boundaryEdges,self.boundaryNodes)

    def registerEdges(self,t):
        for en,e in enumerate(t.edges):
            if e.nodes in self.edgeDict:
                t.edges[en]=self.edgeDict[e.nodes]
            else:
                eN=len(self.edgeDict)
                e.N=eN
                self.edgeDict[e.nodes]=e

    def registerFaces(self,T):
        for tn,t in enumerate(T.faces):
            if t.nodes in self.faceDict:
                T.faces[tn]=self.faceDict[t.nodes]
            else:
                t.N=len(self.faceDict)
                self.faceDict[t.nodes]=t
                self.registerEdges(t)

    def registerNode(self,node):
        if node in self.nodeDict:
            node = self.nodeDict[node]
        else:
            node.N = len(self.nodeDict)
            self.nodeDict[node] = node
        return node

#    def refine(self,oldMesh):
#        return self.refineFreudenthalBey(oldMesh)

    def meshInfo(self):
        minfo = """Number of hexahedra  : %d
Number of faces      : %d
Number of edges      : %d
Number of nodes      : %d
max(sigma_k)         : %d
min(h_k)             : %d\n""" % (self.nElements_global,
                                  self.nElementBoundaries_global,
                                  self.nEdges_global,
                                  self.nNodes_global,
                                  self.sigmaMax,
                                  self.hMin)
        if self.subdomainMesh != self:
            sinfo = self.subdomainMesh.meshInfo()
            info = "*** Global ***\n" + minfo + "\n*** Local ***\n" + sinfo
            return info
        return minfo

    def writeMeshXdmf(self,ar,name='',t=0.0,init=False,meshChanged=False,tCount=0,EB=False):
        Mesh.writeMeshXdmf(self,ar,name,t,init,meshChanged,"Hexahedron",tCount,EB=EB)

    def generateFromHexFile(self,filebase,base=0):
        from . import cmeshTools
        self.cmesh = cmeshTools.CMesh()
        cmeshTools.generateFromHexFile(self.cmesh,filebase,base)
        cmeshTools.allocateGeometricInfo_hexahedron(self.cmesh)
        cmeshTools.computeGeometricInfo_hexahedron(self.cmesh)
        self.buildFromC(self.cmesh)

class Mesh2DM(Mesh):
    """A triangular mesh based on an ADH 3dm file"""
    def __init__(self,filename,adhBase=1):
        meshIn = open(filename+'.3dm','r')
        firstLine = meshIn.readline()
        firstWords = firstLine.split()
        logEvent("Reading object=%s from file=%s" % (firstWords[0],filename))
        line = meshIn.readline()
        columns = line.split()
        #read in the tetrahedra and nodes as memory-efficiently as possible
        tn0 = array.array('i')
        tn1 = array.array('i')
        tn2 = array.array('i')
        material = array.array('i')
        nx  = array.array('d')
        ny  = array.array('d')
        nz  = array.array('d')
        print("Reading "+str(filename))
        #assume tets are ordered by tet number
        while (len(columns) > 0 and (columns[0] == 'E3T' or columns[0] == 'GE3')):
            tn0.append(int(columns[2]))
            tn1.append(int(columns[3]))
            tn2.append(int(columns[4]))
            material.append(int(columns[5]))
            line = meshIn.readline()
            columns = line.split()
        #allow for missing lines
        while (len(columns) == 0):
            line = meshIn.readline()
            columns = line.split()
        #assume nodes are ordered by node number
        while (len(columns) == 5):
            nx.append(float(columns[2]))
            ny.append(float(columns[3]))
            nz.append(float(columns[4]))
            line = meshIn.readline()
            columns = line.split()
        meshIn.close()
        print("Allocating node and element arrays")
        self.nTriangles_global = len(tn0)
        self.triangleArray = np.zeros(
            (self.nTriangles_global,3),'i')
        tA = self.triangleArray
        self.triangleMaterialArray = np.zeros(
            (self.nTriangles_global,),'i')
        tMA = self.triangleMaterialArray
        self.nNodes_global = len(nx)
        self.nodeArray = np.zeros((self.nNodes_global,3),'d')
        for tN in range(self.nTriangles_global):
            tA[tN,0] = tn0[tN] - adhBase
            tA[tN,1] = tn1[tN] - adhBase
            tA[tN,2] = tn2[tN] - adhBase
            tMA[tN]  = material[tN] - adhBase
        for nN in range(self.nNodes_global):
            self.nodeArray[nN,0]= nx[nN]
            self.nodeArray[nN,1]= ny[nN]
            self.nodeArray[nN,2]= nz[nN]
        print("Deleting temporary storage")
        del tn0,tn1,tn2,nx,ny,nz
        self.nElements_global = self.nTriangles_global
        self.elementNodesArray = self.triangleArray
        self.elementMaterialTypes = self.triangleMaterialArray
        print("Number of triangles:"+str(self.nElements_global))
        print("Number of nodes     :"+str(self.nNodes_global))
        #archive with Xdmf
        self.nNodes_element = 3
        self.arGridCollection = None
        self.arGrid = None; self.arTime = None

    def buildEdgeArrays(self):
        print("Extracting edges triangles dictionary")
        edges_triangles={}
        t=self.triangleArray
        self.nInteriorEdges_global=0
        for N in range(self.nTriangles_global):
            #sort node numbers so the nodes can
            #uniquely identify the triangles/edges
            n = list(t[N,:])
            n.sort()
            edges = [(n[0],n[1]),
                     (n[0],n[2]),
                     (n[1],n[2])]
            for t in triangles:
                if t in edges_triangles:
                    edges_triangles[t].append(N)
                    self.nInteriorTriangles_global+=1
                else:
                    edges_triangles[t]=[N]
        print("Building edge and exterior arrays")
        self.nEdges_global = len(edges_triangles)
        self.edgeArray = np.zeros(
            (self.nEdges_global,2),'i')
        self.edgeMaterialArray = np.zeros(
            (self.nEdges_global,2),'i')
        self.interiorEdgeArray = np.zeros(
            (self.nInteriorEdges_global,),'i')
        self.nExteriorEdges_global = self.nEdges_global - \
                                     self.nInteriorEdges_global
        self.exteriorEdgeArray = np.zeros(
            (self.nExteriorEdges_global,),'i')
        eN=0
        ieN=0
        eeN=0
        exteriorNodes=set()
        eA = self.edgeArray
        eMA = self.edgeMaterialArray
        tMA = self.triangleMaterialArray
        for eNodes,tlist in edges_triangles.items():
            eA[eN,0]=eNodes[0]
            eA[eN,1]=eNodes[1]
            if len(tlist)==2:
                self.interiorEdgeArray[ieN]=eN
                eMA[eN][0]= tMA[tlist[0]]
                eMA[eN][1]= tMA[Tlist[1]]
                ieN+=1
            else:
                exteriorNodes.update(tNodes)
                self.exteriorEdgeArray[eeN]=eN
                eMA[eN][0]=tMA[tlist[0]]
                eeN+=1
            eN+=1
        self.nExteriorNodes_global = len(exteriorNodes)
        self.exteriorNodeArray = np.zeros(
            (self.nExteriorNodes_global,),'i')
        self.globalToExteriorNodeArray = np.zeros(
            (self.nNodes_global,),'i')
        for nExtN,nN in enumerate(exteriorNodes):
            self.exteriorNodeArray[nExtN]=nN
            self.globalToExteriorNodeArray[nN]=nExtN
        print("Number of edges         :"+str(self.nEdges_global))
        print("Number on interior      :"+str(self.nInteriorEdges_global))
        print("Number on exterior      :"+str(self.nExteriorEdges_global))
        print("Number of exterior nodes:"+str(self.nExteriorNodes_global))
        #at this point we can easily build a boundary mesh by renumbering using
        #exteriorNodeArray and exteriorEdgeArray to renumber
        #and the info in nodeArray and edgeArray

    def writeBoundaryMeshADH(self,filename,adhBase=1):
        #I'll print it using node numbers from the 3D mesh
        meshOut = open(filename+'Boundary.3dm','w')
        meshOut.write('MESH1D\n')
        for eeN in range(self.nExteriorEdges_global):
            eN = self.exteriorEdgeArray[eeN]
            n0 = self.edgeArray[eN][0] + adhBase
            n1 = self.edgeArray[eN][1] + adhBase
            m  = self.edgeMaterialArray[eN][0] + adhBase
            line = 'E3T %5i %5i %5i %5i' % \
                          (tN+adhBase,n0,n1,m)
            meshOut.write(line+'\n')
        meshOut.close()

    def writeMeshEnsight(self,filename,description=None):
        base=1
        #write the casefile
        caseOut=open(filename+'.case','w')
        caseOut.write('FORMAT\n'+'type: ensight gold\n')
        caseOut.write('GEOMETRY\n'+'model: '+filename+'.geo\n')
        caseOut.close()
        meshOut=open(filename+'.geo','w')
        meshOut.write('Ensight Gold\n')
        meshOut.write('Unstructured Triangular Mesh\n')
        meshOut.write('node id given\n')
        meshOut.write('element id given\n')
        #extents = 'extents\n %12.5E %12.5E\n %12.5E %12.5E\n %12.5E %12.5E\n' % (self.xmin,self.xmax,self.ymin,self.ymax,self.zmin,self.zmax)
        #meshOut.write('extents\n'+`self.xmin`+' '+`self.xmax`+'\n')
        meshOut.write('part \n'+'%10i\n' % 1)
        if description:
            meshOut.write(description+'\n')
        else:
            meshOut.write('A Mesh\n')
        meshOut.write('coordinates\n'+'%10i\n' % self.nNodes_global)
        for nN in range(self.nNodes_global):
            ensightNodeNumber = (nN+base)
            meshOut.write('%10i\n' % ensightNodeNumber)
        for nN in range(self.nNodes_global):
            meshOut.write('%12.5E\n' % self.nodeArray[nN,0])
        for nN in range(self.nNodes_global):
            meshOut.write('%12.5E\n' % self.nodeArray[nN,1])
        for nN in range(self.nNodes_global):
            meshOut.write('%12.5E\n' % self.nodeArray[nN,2])
        meshOut.write('tria3\n'+'%10i\n' % self.nTriangles_global)
        for tN in range(self.nTriangles_global):
            ensightElementNumber = tN + base
            meshOut.write('%10i\n' % ensightElementNumber)
        tA = self.triangleArray
        for tN in range(self.nTriangles_global):
            meshOut.write('%10i%10i%10i\n' % (tA[tN,0]+base,tA[tN,1]+base,tA[tN,2]+base))
        meshOut.close()

    def writeMeshXdmf(self,ar,name='',t=0.0,init=False,meshChanged=False,Xdmf_ElementTopology="Triangle",tCount=0):
        if self.arGridCollection is not None:
            init = False
        elif not init:
            self.arGridCollection = ar.domain[0]
        if init:
            self.arGridCollection = ar.domain.collection("Mesh "+name)
        if not self.arGridCollection.is_current(t):
            rank = ar.comm.rank()
            #
            #topology and geometry
            #
            if ar.global_sync:
                assert ar.writesDatasets, "global_sync with text heavy data not supported"
                n_elements       = self.globalMesh.nElements_global
                nodes_dims       = [self.globalMesh.nNodes_global, 3]
                elements_dataset = 'elements'+name+str(tCount)
                nodes_dataset    = 'nodes'+name+str(tCount)
                elements_dtype   = self.globalMesh.nodeNumbering_subdomain2global
            else:
                n_elements       = self.nElements_owned
                nodes_dims       = [self.nNodes_global, 3]
                elements_dataset = 'elements'+str(rank)+name+str(tCount)
                nodes_dataset    = 'nodes'+str(rank)+name+str(tCount)
                elements_dtype   = self.elementNodesArray
            self.arGrid = ar.write_grid(self.arGridCollection, None, t, tCount)
            ar.write_topology(self.arGrid, Xdmf_ElementTopology, n_elements,
                              [n_elements, self.nNodes_element],
                              elements_dataset, 'elements'+name, dtype=elements_dtype)
            ar.write_geometry(self.arGrid, nodes_dims, nodes_dataset, 'nodes'+name,
                              dtype=self.nodeArray)
            #material types
            if ar.global_sync:
                ar.write_field(self.arGrid, "elementMaterialTypes", self.elementMaterialTypes, tCount,
                               center="Cell", dimensions=[self.globalMesh.nElements_global],
                               sync_offsets=self.globalMesh.elementOffsets_subdomain_owned,
                               sync_data=self.elementMaterialTypes[:self.nElements_owned],
                               write_data=init or meshChanged)
            else:
                ar.write_field(self.arGrid, "elementMaterialTypes",
                               self.elementMaterialTypes[:self.nElements_owned], tCount,
                               center="Cell", dimensions=[self.nElements_owned],
                               write_data=init or meshChanged)
            if init or meshChanged:
                if ar.global_sync:
                    ar.create_dataset_sync(elements_dataset,
                                           offsets = self.globalMesh.elementOffsets_subdomain_owned,
                                           data = self.globalMesh.nodeNumbering_subdomain2global[self.elementNodesArray[:self.nElements_owned]])
                    ar.create_dataset_sync(nodes_dataset,
                                           offsets = self.globalMesh.nodeOffsets_subdomain_owned,
                                           data = self.nodeArray[:self.nNodes_owned])
                elif ar.writesDatasets:
                    ar.create_dataset_async(elements_dataset, data = self.elementNodesArray[:self.nElements_owned])
                    ar.create_dataset_async(nodes_dataset, data = self.nodeArray)
                else:
                    np.savetxt(ar.textDataDir+"/elements"+name+".txt",self.elementNodesArray[:self.nElements_owned],fmt='%d')
                    np.savetxt(ar.textDataDir+"/nodes"+name+".txt",self.nodeArray)


class Mesh3DM(Mesh):
    """
    A Mesh for reading in tetrahedral meshes in the .3dm format
    """
    def __init__(self,filename,adhBase=1):
        meshIn = open(filename+'.3dm','r')
        firstLine = meshIn.readline()
        firstWords = firstLine.split()
        print("Reading object=%s from file=%s" % (firstWords[0],filename))
        line = meshIn.readline()
        columns = line.split()
        #read in the tetrahedra and nodes as memory-efficiently as possible
        Tn0 = array.array('i')
        Tn1 = array.array('i')
        Tn2 = array.array('i')
        Tn3 = array.array('i')
        material = array.array('i')
        nx  = array.array('d')
        ny  = array.array('d')
        nz  = array.array('d')
        print("Reading "+str(filename))
        #assume tets are ordered by tet number
        while (len(columns) > 0 and (columns[0] == 'E4T' or columns[0] == 'GE4')):
            Tn0.append(int(columns[2]))
            Tn1.append(int(columns[3]))
            Tn2.append(int(columns[4]))
            Tn3.append(int(columns[5]))
            material.append(int(columns[6]))
            line = meshIn.readline()
            columns = line.split()
        #assume nodes are ordered by node number
        while (len(columns) == 5):
            nx.append(float(columns[2]))
            ny.append(float(columns[3]))
            nz.append(float(columns[4]))
            line = meshIn.readline()
            columns = line.split()
        meshIn.close()
        print("Allocating node and element arrays")
        self.nTetrahedra_global = len(Tn0)
        self.tetrahedronArray = np.zeros(
            (self.nTetrahedra_global,4),'i')
        TA = self.tetrahedronArray
        self.tetrahedronMaterialArray = np.zeros(
            (self.nTetrahedra_global,),'i')
        TMA = self.tetrahedronMaterialArray
        self.nNodes_global = len(nx)
        self.nodeArray = np.zeros((self.nNodes_global,3),'d')
        for TN in range(self.nTetrahedra_global):
            TA[TN,0] = Tn0[TN] - adhBase
            TA[TN,1] = Tn1[TN] - adhBase
            TA[TN,2] = Tn2[TN] - adhBase
            TA[TN,3] = Tn3[TN] - adhBase
            TMA[TN]  = material[TN] - adhBase
        for nN in range(self.nNodes_global):
            self.nodeArray[nN,0]= nx[nN]
            self.nodeArray[nN,1]= ny[nN]
            self.nodeArray[nN,2]= nz[nN]
        print("Deleting temporary storage")
        del Tn0,Tn1,Tn2,Tn3,nx,ny,nz
        self.nElements_global = self.nTetrahedra_global
        self.elementNodesArray = self.tetrahedronArray
        self.elementMaterialTypes = self.tetrahedronMaterialArray
        self.arGridCollection=None
        print("Number of tetrahedra:"+str(self.nElements_global))
        print("Number of nodes     :"+str(self.nNodes_global))

    def buildTriangleArrays(self):
        print("Extracting triangles tetrahedra dictionary")
        triangles_tetrahedra={}
        T=self.tetrahedronArray
        self.nInteriorTriangles_global=0
        for N in range(self.nTetrahedra_global):
            #sort node numbers so the nodes can
            #uniquely identify the triangles/edges
            n = list(T[N,:])
            n.sort()
            triangles = [(n[0],n[1],n[2]),
                         (n[0],n[1],n[3]),
                         (n[0],n[2],n[3]),
                         (n[1],n[2],n[3])]
            for t in triangles:
                if t in triangles_tetrahedra:
                    triangles_tetrahedra[t].append(N)
                    self.nInteriorTriangles_global+=1
                else:
                    triangles_tetrahedra[t]=[N]
        print("Building triangle and exterior arrays")
        self.nTriangles_global = len(triangles_tetrahedra)
        self.triangleArray = np.zeros(
            (self.nTriangles_global,3),'i')
        self.triangleMaterialArray = np.zeros(
            (self.nTriangles_global,2),'i')
        self.interiorTriangleArray = np.zeros(
            (self.nInteriorTriangles_global,),'i')
        self.nExteriorTriangles_global = self.nTriangles_global - \
                                         self.nInteriorTriangles_global
        self.exteriorTriangleArray = np.zeros(
            (self.nExteriorTriangles_global,),'i')
        tN=0
        itN=0
        etN=0
        exteriorNodes=set()
        tA = self.triangleArray
        tMA = self.triangleMaterialArray
        TMA = self.tetrahedronMaterialArray
        for tNodes,Tlist in triangles_tetrahedra.items():
            tA[tN,0]=tNodes[0]
            tA[tN,1]=tNodes[1]
            tA[tN,2]=tNodes[2]
            if len(Tlist)==2:
                self.interiorTriangleArray[itN]=tN
                tMA[tN][0]= TMA[Tlist[0]]
                tMA[tN][1]= TMA[Tlist[1]]
                itN+=1
            else:
                exteriorNodes.update(tNodes)
                self.exteriorTriangleArray[etN]=tN
                tMA[tN][0]=TMA[Tlist[0]]
                etN+=1
            tN+=1
        self.nExteriorNodes_global = len(exteriorNodes)
        self.exteriorNodeArray = np.zeros(
            (self.nExteriorNodes_global,),'i')
        self.globalToExteriorNodeArray = np.zeros(
            (self.nNodes_global,),'i')
        for nExtN,nN in enumerate(exteriorNodes):
            self.exteriorNodeArray[nExtN]=nN
            self.globalToExteriorNodeArray[nN]=nExtN
        print("Number of triangles     :"+str(self.nTriangles_global))
        print("Number on interior      :"+str(self.nInteriorTriangles_global))
        print("Number on exterior      :"+str(self.nExteriorTriangles_global))
        print("Number of exterior nodes:"+str(self.nExteriorNodes_global))
        #at this point we can easily build a boundary mesh by renumbering using
        #exteriorNodeArray and exteriorTriangleArray to renumber
        #and the info in nodeArray and triangleArray

    def buildEdgeArray(self):
        print("Extracting set of edges")
        edges = set()
        t=self.triangleArray
        for N in range(self.nTriangles_global):
            #triangle nodes are assumed sorted
            edges.update([(t[N,0],t[N,1]),
                          (t[N,0],t[N,2]),
                          (t[N,1],t[N,2])])
        print("Building edgeArray")
        self.nEdges_global = len(edges)
        self.edgeArray = np.zeros(
            (self.nEdges_global,2),'i')
        eN=0
        for e in edges:
            self.edgeArray[eN][0] = e[0]
            self.edgeArray[eN][1] = e[1]
        del edges
        print("Number of edges     :"+str(self.nEdges_global))

    def writeBoundaryMeshADH(self,filename,adhBase=1):
        #I'll print it using node numbers from the 3D mesh
        meshOut = open(filename+'Boundary.3dm','w')
        meshOut.write('MESH2D\n')
        for tN in self.exteriorTriangleArray:
            n0 = self.triangleArray[tN][0] + adhBase
            n1 = self.triangleArray[tN][1] + adhBase
            n2 = self.triangleArray[tN][2] + adhBase
            m  = self.triangleMaterialArray[tN][0] + adhBase
            line = 'E3T %5i %5i %5i %5i %5i' % \
                          (tN+adhBase,n0,n1,n2,m)
            meshOut.write(line+'\n')
        for nN in self.exteriorNodeArray:
            n  = self.nodeArray[nN]
            line = 'ND %5i %14.8e %14.8e %14.8e' % \
                          (nN + adhBase,n[0],n[1],n[2])
            #print line
            meshOut.write(line+'\n')
        meshOut.close()

    def writeMeshEnsight(self,filename,description=None):
        base=1
        #write the casefile
        caseOut=open(filename+'.case','w')
        caseOut.write('FORMAT\n'+'type: ensight gold\n')
        caseOut.write('GEOMETRY\n'+'model: '+filename+'.geo\n')
        caseOut.close()
        meshOut=open(filename+'.geo','w')
        meshOut.write('Ensight Gold\n')
        meshOut.write('Unstructured Tetrahedral Mesh\n')
        meshOut.write('node id given\n')
        meshOut.write('element id given\n')
        #extents = 'extents\n %12.5E %12.5E\n %12.5E %12.5E\n %12.5E %12.5E\n' % (self.xmin,self.xmax,self.ymin,self.ymax,self.zmin,self.zmax)
        #meshOut.write('extents\n'+`self.xmin`+' '+`self.xmax`+'\n')
        meshOut.write('part \n'+'%10i\n' % 1)
        if description:
            meshOut.write(description+'\n')
        else:
            meshOut.write('A Mesh\n')
        meshOut.write('coordinates\n'+'%10i\n' % self.nNodes_global)
        for nN in range(self.nNodes_global):
            ensightNodeNumber = (nN+base)
            meshOut.write('%10i\n' % ensightNodeNumber)
        for nN in range(self.nNodes_global):
            meshOut.write('%12.5E\n' % self.nodeArray[nN,0])
        for nN in range(self.nNodes_global):
            meshOut.write('%12.5E\n' % self.nodeArray[nN,1])
        for nN in range(self.nNodes_global):
            meshOut.write('%12.5E\n' % self.nodeArray[nN,2])
        meshOut.write('tetra4\n'+'%10i\n' % self.nTetrahedra_global)
        for TN in range(self.nTetrahedra_global):
            ensightElementNumber = TN + base
            meshOut.write('%10i\n' % ensightElementNumber)
        TA = self.tetrahedronArray
        for TN in range(self.nTetrahedra_global):
            meshOut.write('%10i%10i%10i%10i\n' % (TA[TN,0]+base,
                                                  TA[TN,1]+base,
                                                  TA[TN,2]+base,
                                                  TA[TN,3]+base))
        meshOut.close()

    def writeBoundaryMeshEnsight(self,filename,description=None):
        base=1
        #write the casefile
        caseOut=open(filename+'Boundary.case','w')
        caseOut.write('FORMAT\n'+'type: ensight gold\n')
        caseOut.write('GEOMETRY\n'+'model: '+filename+'Boundary.geo\n')
        caseOut.close()
        meshOut=open(filename+'Boundary.geo','w')
        meshOut.write('Unstructured Triangular Surface Mesh\n\n')
        meshOut.write('node id given\n')
        meshOut.write('element id given\n')
        meshOut.write('part \n'+'%10i\n' % 1)
        if description:
            meshOut.write(description+'\n')
        else:
            meshOut.write('A Mesh\n')
        meshOut.write('coordinates\n'+'%10i\n' % self.nExteriorNodes_global)
        for nN in range(self.nExteriorNodes_global):
            ensightNodeNumber = (nN+base)
            meshOut.write('%10i\n' % ensightNodeNumber)
        for nN in range(self.nExteriorNodes_global):
            meshOut.write('%12.5E\n' %
                          self.nodeArray[self.exteriorNodeArray[nN],0])
        for nN in range(self.nExteriorNodes_global):
            meshOut.write('%12.5E\n' %
                          self.nodeArray[self.exteriorNodeArray[nN],1])
        for nN in range(self.nExteriorNodes_global):
            meshOut.write('%12.5E\n' %
                          self.nodeArray[self.exteriorNodeArray[nN],2])
        meshOut.write('tria3\n'+'%10i\n' % self.nExteriorTriangles_global)
        for tN in range(self.nExteriorTriangles_global):
            ensightElementNumber = tN + base
            meshOut.write('%10i\n' % ensightElementNumber)
        tA = self.triangleArray
        for tN in self.exteriorTriangleArray:
            meshOut.write('%10i%10i%10i\n' %
                          (self.globalToExteriorNodeArray[tA[tN,0]]+base,
                           self.globalToExteriorNodeArray[tA[tN,1]]+base,
                           self.globalToExteriorNodeArray[tA[tN,2]]+base))
        meshOut.close()

    def writeMeshXdmf(self,ar,name='',t=0.0,init=False,meshChanged=False,Xdmf_ElementTopology="Tetrahedron",tCount=0):
        if self.arGridCollection is not None:
            init = False
        elif not init:
            self.arGridCollection = ar.domain[0]
        if init:
            self.arGridCollection = ar.domain.collection("Mesh "+name)
        if not self.arGridCollection.is_current(t):
            rank = ar.comm.rank()
            #
            #topology and geometry
            #
            if ar.global_sync:
                assert ar.writesDatasets, "global_sync with text heavy data not supported"
                n_elements       = self.globalMesh.nElements_global
                nodes_dims       = [self.globalMesh.nNodes_global, 3]
                elements_dataset = 'elements'+name+str(tCount)
                nodes_dataset    = 'nodes'+name+str(tCount)
                elements_dtype   = self.globalMesh.nodeNumbering_subdomain2global
            else:
                n_elements       = self.nElements_owned
                nodes_dims       = [self.nNodes_global, 3]
                elements_dataset = 'elements'+str(rank)+name+str(tCount)
                nodes_dataset    = 'nodes'+str(rank)+name+str(tCount)
                elements_dtype   = self.elementNodesArray
            self.arGrid = ar.write_grid(self.arGridCollection, None, t, tCount)
            ar.write_topology(self.arGrid, Xdmf_ElementTopology, n_elements,
                              [n_elements, self.nNodes_element],
                              elements_dataset, 'elements'+name, dtype=elements_dtype)
            ar.write_geometry(self.arGrid, nodes_dims, nodes_dataset, 'nodes'+name,
                              dtype=self.nodeArray)
            #material types
            if ar.global_sync:
                ar.write_field(self.arGrid, "elementMaterialTypes", self.elementMaterialTypes, tCount,
                               center="Cell", dimensions=[self.globalMesh.nElements_global],
                               sync_offsets=self.globalMesh.elementOffsets_subdomain_owned,
                               sync_data=self.elementMaterialTypes[:self.nElements_owned],
                               write_data=init or meshChanged)
            else:
                ar.write_field(self.arGrid, "elementMaterialTypes",
                               self.elementMaterialTypes[:self.nElements_owned], tCount,
                               center="Cell", dimensions=[self.nElements_owned],
                               write_data=init or meshChanged)
            if init or meshChanged:
                if ar.global_sync:
                    ar.create_dataset_sync(elements_dataset,
                                           offsets = self.globalMesh.elementOffsets_subdomain_owned,
                                           data = self.globalMesh.nodeNumbering_subdomain2global[self.elementNodesArray[:self.nElements_owned]])
                    ar.create_dataset_sync(nodes_dataset,
                                           offsets = self.globalMesh.nodeOffsets_subdomain_owned,
                                           data = self.nodeArray[:self.nNodes_owned])
                elif ar.writesDatasets:
                    ar.create_dataset_async(elements_dataset, data = self.elementNodesArray[:self.nElements_owned])
                    ar.create_dataset_async(nodes_dataset, data = self.nodeArray)
                else:
                    np.savetxt(ar.textDataDir+"/elements"+name+".txt",self.elementNodesArray[:self.nElements_owned],fmt='%d')
                    np.savetxt(ar.textDataDir+"/nodes"+name+".txt",self.nodeArray)


class MultilevelTetrahedralMesh(MultilevelMesh):
    """A hierarchical multilevel mesh with tetrahedral cells"""
    def __init__(self,
                 nx, ny, nz,
                 x=0.0, y=0.0, z=0.0,
                 Lx=1.0, Ly=1.0, Lz=1.0,
                 refinementLevels=1,
                 skipInit=False,
                 nLayersOfOverlap=1,
                 parallelPartitioningType=MeshParallelPartitioningTypes.node,
                 regularGrid=None):
        from . import cmeshTools
        from . import Comm
        MultilevelMesh.__init__(self)
        self.useC = True
        self.nLayersOfOverlap = nLayersOfOverlap; self.parallelPartitioningType = parallelPartitioningType
        logEvent("Generating tetrahedral mesh")
        if not skipInit:
            if self.generatesInPieces('tet', 0, nLayersOfOverlap, parallelPartitioningType, regularGrid):
                self.generateCartesianHierarchy('tet', (nx,ny,nz), (x,y,z), (Lx,Ly,Lz), refinementLevels)
            elif self.useC:
                self.meshList.append(TetrahedralMesh())
                self.meshList[0].generateTetrahedralMeshFromRectangularGrid(nx,ny,nz,Lx,Ly,Lz)
                #Translate the coarse mesh *before* refining it. The C refinement copies
                #the coarse nodes into every finer level, so shifting level 0 afterwards
                #left the finer levels at the origin -- a different coordinate frame from
                #level 0, so geometric transfers between them were wrong and the finest
                #mesh, the one that is solved on, was not where the domain is.
                self.meshList[0].nodeArray[:,0] += x
                self.meshList[0].nodeArray[:,1] += y
                self.meshList[0].nodeArray[:,2] += z
                self.cmultilevelMesh = cmeshTools.CMultilevelMesh(self.meshList[0].cmesh,refinementLevels)
                self.buildFromC(self.cmultilevelMesh)
                for l in range(1,refinementLevels):
                    self.meshList.append(TetrahedralMesh())
                    self.meshList[l].cmesh = self.cmeshList[l]
                    self.meshList[l].buildFromC(self.cmeshList[l])
                self.partitionLevels(nLayersOfOverlap,parallelPartitioningType)
            else:
                grid=RectangularGrid(nx,ny,nz,Lx,Ly,Lz)
                self.meshList.append(TetrahedralMesh())
                self.meshList[0].rectangularToTetrahedral(grid)
                self.meshList[0].nodeArray[:,0] += x
                self.meshList[0].nodeArray[:,1] += y
                self.meshList[0].nodeArray[:,2] += z
                self.elementChildren=[]
                logEvent(self.meshList[0].meshInfo())
                for l in range(1,refinementLevels):
                    self.refine()
                    logEvent(self.meshList[-1].meshInfo())
                self.buildArrayLists()
    def generateFromExistingCoarseMesh(self,mesh0,refinementLevels,nLayersOfOverlap=1,
                                       parallelPartitioningType=MeshParallelPartitioningTypes.node):
        from . import cmeshTools
        #blow away or just trust garbage collection
        self.nLayersOfOverlap=nLayersOfOverlap;self.parallelPartitioningType=parallelPartitioningType
        self.meshList = []
        self.elementParents = None
        self.cmultilevelMesh = None
        if self.useC:
            self.meshList.append(mesh0)
            logEvent("cmeshTools.CMultilevelMesh")
            self.cmultilevelMesh = cmeshTools.CMultilevelMesh(self.meshList[0].cmesh,refinementLevels)
            # if not mesh0.plexMesh:
            logEvent("buildFromC")
            self.buildFromC(self.cmultilevelMesh)
            logEvent("partitionMesh")
            for l in range(1,refinementLevels):
                self.meshList.append(TetrahedralMesh())
                self.meshList[l].cmesh = self.cmeshList[l]
                self.meshList[l].buildFromC(self.meshList[l].cmesh)
            self.partitionLevels(nLayersOfOverlap,parallelPartitioningType)
        else:
            grid=RectangularGrid(nx,ny,nz,Lx,Ly,Lz)
            self.meshList.append(TetrahedralMesh())
            self.meshList[0].rectangularToTetrahedral(grid)
            self.meshList[0].subdomainMesh = self.meshList[0]
            self.elementChildren=[]
            logEvent(self.meshList[0].meshInfo())
            for l in range(1,refinementLevels):
                self.refine()
                self.meshList[l].subdomainMesh = self.meshList[l]
                logEvent(self.meshList[-1].meshInfo())
            self.buildArrayLists()

    def generatePartitionedMeshFromPUMI(self,mesh0,refinementLevels,nLayersOfOverlap=1):
        from . import cmeshTools
        self.meshList = []
        self.meshList.append(mesh0)
        self.cmultilevelMesh = cmeshTools.CMultilevelMesh(self.meshList[0].cmesh,refinementLevels)
        self.buildFromC(self.cmultilevelMesh)
        self.elementParents = None
        self.elementChildren=[]

    def generatePartitionedMeshFromTetgenFiles(self,filebase,base,mesh0,refinementLevels,nLayersOfOverlap=1,
                                               parallelPartitioningType=MeshParallelPartitioningTypes.node):
        from . import cmeshTools
        if filebase==None:
            filebase="mesh"
        assert(refinementLevels==1)
        assert(parallelPartitioningType==MeshParallelPartitioningTypes.node)
        assert(nLayersOfOverlap<=1)
        mesh0.cmesh = cmeshTools.CMesh()
        #blow away or just trust garbage collection
        self.nLayersOfOverlap=nLayersOfOverlap;self.parallelPartitioningType=parallelPartitioningType
        self.meshList = []
        self.elementParents = None
        self.cmultilevelMesh = None
        self.meshList.append(mesh0)
        logEvent("cmeshTools.CMultilevelMesh")
        self.cmultilevelMesh = cmeshTools.CMultilevelMesh(self.meshList[0].cmesh,refinementLevels)
        logEvent("buildFromC")
        self.buildFromC(self.cmultilevelMesh)
        logEvent("partitionMesh")
        self.meshList[0].partitionMeshFromFiles(filebase,base,nLayersOfOverlap=nLayersOfOverlap,parallelPartitioningType=parallelPartitioningType)

    def refine(self):
        self.meshList.append(TetrahedralMesh())
        childrenDict = self.meshList[-1].refine(self.meshList[-2])
        self.elementChildren.append(childrenDict)
    def computeGeometricInfo(self):
        for m in self.meshList:
            m.computeGeometricInfo()

class MultilevelHexahedralMesh(MultilevelMesh):
    """A hierarchical multilevel mesh with hexahedral cells"""
    def __init__(self,
                 nx, ny, nz,
                 px=0, py=0, pz=0,
                 x=0.0, y=0.0, z=0.0,
                 Lx=1.0, Ly=1.0, Lz=1.0,
                 refinementLevels=1,
                 skipInit=False,
                 nLayersOfOverlap=1,
                 parallelPartitioningType=MeshParallelPartitioningTypes.node,
                 regularGrid=None):
        from . import cmeshTools
        from . import Comm
        MultilevelMesh.__init__(self)
        if refinementLevels == 1:
            self.useC = True
        else:
            self.useC = False
        #Hexahedral refinement was never implemented (refine() raises), but a
        #regular grid does not need it: see generateStructuredHierarchy.
        structured = refinementLevels > 1
        self.nLayersOfOverlap = nLayersOfOverlap; self.parallelPartitioningType = parallelPartitioningType
        logEvent("Generating hexahedral mesh")
        if not skipInit:
            if self.generatesInPieces('hex', 0, nLayersOfOverlap, parallelPartitioningType, regularGrid):
                self.generateCartesianHierarchy('hex', (nx,ny,nz), (x,y,z), (Lx,Ly,Lz), refinementLevels)
            elif structured:
                self._structuredLevels = refinementLevels
                self.generateStructuredHierarchy(
                    HexahedralMesh,
                    lambda mesh,c: mesh.generateHexahedralMeshFromRectangularGrid(c[0],c[1],c[2],Lx,Ly,Lz),
                    (nx,ny,nz),(x,y,z),3,
                    nLayersOfOverlap=nLayersOfOverlap,
                    parallelPartitioningType=parallelPartitioningType)
            elif self.useC:
                self.meshList.append(HexahedralMesh())
                self.meshList[0].generateHexahedralMeshFromRectangularGrid(nx,ny,nz,Lx,Ly,Lz)
                self.meshList[0].nodeArray[:,0] += x
                self.meshList[0].nodeArray[:,1] += y
                self.meshList[0].nodeArray[:,2] += z
                self.cmultilevelMesh = cmeshTools.CMultilevelMesh(self.meshList[0].cmesh,refinementLevels)
                self.buildFromC(self.cmultilevelMesh)
                for l in range(1,refinementLevels):
                    self.meshList.append(HexahedralMesh())
                    self.meshList[l].cmesh = self.cmeshList[l]
                    self.meshList[l].buildFromC(self.cmeshList[l])
                self.partitionLevels(nLayersOfOverlap,parallelPartitioningType)
            else:
                grid=RectangularGrid(nx,ny,nz,Lx,Ly,Lz)
                self.meshList.append(HexahedralMesh())
                self.elementChildren=[]
                self.meshList[0].sigmaMax=0.0
                logEvent(self.meshList[0].meshInfo())
                for l in range(1,refinementLevels):
                    self.refine()
                    self.meshList[-1].sigmaMax=0.0
                    logEvent(self.meshList[-1].meshInfo())
                self.buildArrayLists()
    def generateFromExistingCoarseMesh(self,mesh0,refinementLevels,nLayersOfOverlap=1,
                                       parallelPartitioningType=MeshParallelPartitioningTypes.node):
        from . import cmeshTools
        #blow away or just trust garbage collection
        self.nLayersOfOverlap=nLayersOfOverlap;self.parallelPartitioningType=parallelPartitioningType
        self.meshList = []
        self.elementParents = None
        self.cmultilevelMesh = None

        self.meshList.append(mesh0)
        self.cmultilevelMesh = cmeshTools.CMultilevelMesh(self.meshList[0].cmesh,refinementLevels)
        self.buildFromC(self.cmultilevelMesh)
        for l in range(1,refinementLevels):
            self.meshList.append(HexahedralMesh())
            self.meshList[l].cmesh = self.cmeshList[l]
            self.meshList[l].buildFromC(self.meshList[l].cmesh)
        self.partitionLevels(nLayersOfOverlap,parallelPartitioningType)


    def refine(self):
        raise NotImplementedError
        self.meshList.append(HexahedralMesh())
        childrenDict = self.meshList[-1].refine(self.meshList[-2])
        self.elementChildren.append(childrenDict)
    def computeGeometricInfo(self):
        for m in self.meshList:
            m.computeGeometricInfo()

def buildReferenceSimplex(nd=2):
    """
    Create and return a Proteus mesh object for the reference 
    element.

    Parameters
    ----------
    nd : int
        Dimension of reference element

    Returns
    -------
    mesh : :class:`proteus.MeshTools.TriangularMesh`
        Simplex mesh
    """
    from proteus import Domain

    assert(nd in [1,2,3])

    if nd==1:
        pass # Note sure what needs to go here?!
    
    unit_simplex_domain = Domain.unitSimplex(nd)
    polyfile = "reference_element"
    unit_simplex_domain.writePoly(polyfile)

    if nd==2:
        runTriangle(polyfile,
                    "Yp")
        mesh = genMeshWithTriangle(polyfile,
                                   nbase=1)
        mesh.partitionMesh()
        mesh.globalMesh = mesh
        return mesh
    if nd==3:
        runTetgen(polyfile,
                  "Yp")
        mesh = genMeshWithTetgen(polyfile,
                                 nbase = 1)
        return mesh

class TriangularMesh(Mesh):
    """A mesh of triangles

    The nodes, edges, and triangles are indexed by their
    node tuples. The corresponding lists are derived from the dictionaries, and
    sorted lexicographically. The global node numbers are redefined to
    give a lexicographic ordering.

    The mesh can be generated from a rectangular grid and refined using either
    3t or Freudenthal-Bey global refinement.
    """

    def __init__(self):
        Mesh.__init__(self)
        self.nodeDict={}
        self.edgeDict={}
        self.triangleDict={}
        self.triangleList=[]
        self.oldToNewNode=[]
    def meshType(self):
        return 'simplex'
    def computeGeometricInfo(self):
        from . import cmeshTools
        cmeshTools.computeGeometricInfo_triangle(self.cmesh)
    def generateTriangularMeshFromRectangularGrid(self,nx,ny,Lx,Ly,triangleFlag=1):
        from . import cmeshTools
        self.cmesh = cmeshTools.CMesh()
        cmeshTools.generateTriangularMeshFromRectangularGrid(nx,ny,Lx,Ly,self.cmesh,triangleFlag)
        cmeshTools.allocateGeometricInfo_triangle(self.cmesh)
        cmeshTools.computeGeometricInfo_triangle(self.cmesh)
        self.buildFromC(self.cmesh)
    def rectangularToTriangularOriented(self,grid):
        #copy the nodes from the rectangular mesh
        #I want to be able to renumber latter without
        #changing the grid nodes, so I do deep copies here
        self.nodeList = [Node(n.N,n.p[X],n.p[Y],n.p[Z]) for n in grid.nodeList]
        self.nodeDict = dict([(n,n) for n in self.nodeList])
        for i in range(grid.nHx):
            for j in range(grid.nHy):
                k=0
                n0 = self.nodeList[grid.getNodeNumber(i,j,k)]
                n1 = self.nodeList[grid.getNodeNumber(i,j+1,k)]
                n2 = self.nodeList[grid.getNodeNumber(i+1,j,k)]
                n3 = self.nodeList[grid.getNodeNumber(i+1,j+1,k)]
                self.newTriangle([n0,n1,n3])
                self.newTriangle([n0,n2,n3])
        self.finalize()
        #self.buildListsEdges()
        #self.buildListsTriangles()
    def rectangularToTriangularOrientedOtherWay(self,grid):
        #copy the nodes from the rectangular mesh
        #I want to be able to renumber latter without
        #changing the grid nodes, so I do deep copies here
        self.nodeList = [Node(n.N,n.p[X],n.p[Y],n.p[Z]) for n in grid.nodeList]
        self.nodeDict = dict([(n,n) for n in self.nodeList])
        for i in range(grid.nHx):
            for j in range(grid.nHy):
                k=0
                n0 = self.nodeList[grid.getNodeNumber(i,j,k)]
                n1 = self.nodeList[grid.getNodeNumber(i,j+1,k)]
                n2 = self.nodeList[grid.getNodeNumber(i+1,j,k)]
                n3 = self.nodeList[grid.getNodeNumber(i+1,j+1,k)]
                self.newTriangle([n0,n2,n1])
                self.newTriangle([n2,n3,n1])
        self.finalize()
        #self.buildListsEdges()
        #self.buildListsTriangles()

    def rectangularToTriangularRedBlack(self,grid):
        #copy the nodes from the rectangular mesh
        #I want to be able to renumber latter without
        #changing the grid nodes, so I do deep copies here
        self.nodeList = [Node(n.N,n.p[X],n.p[Y],n.p[Z]) for n in grid.nodeList]
        self.nodeDict = dict([(n,n) for n in self.nodeList])
        self.triangleDict={}
        for i in range(grid.nHx):
            for j in range(grid.nHy):
                k=0
                #associate the element (i,j,k) with the
                #left, front, bottom node
                #get the left,front,bottom,node and its color
                if (grid.getColor(i,j,k) == grid.black):
                    b0 = self.nodeList[grid.getNodeNumber(i,j,k)]
                    r0 = self.nodeList[grid.getNodeNumber(i+1,j,k)]
                    r1 = self.nodeList[grid.getNodeNumber(i,j+1,k)]
                    b1 = self.nodeList[grid.getNodeNumber(i+1,j+1,k)]
                else:
                    r0 = self.nodeList[grid.getNodeNumber(i,j,k)]
                    b0 = self.nodeList[grid.getNodeNumber(i+1,j,k)]
                    b1 = self.nodeList[grid.getNodeNumber(i,j+1,k)]
                    r1 = self.nodeList[grid.getNodeNumber(i+1,j+1,k)]
                self.newTriangle([b0,r0,r1])
                self.newTriangle([b1,r0,r1])
        self.finalize()
        #self.buildListsEdges()
        #self.buildListsTriangles()
    #mwf debug switch to redblac
    rectangularToTriangular = rectangularToTriangularOrientedOtherWay#rectangularToTriangularOriented
    def generateFromTriangleMesh(self,ctrirep,base):
        from .import cmeshTools
        self.cmesh = cmeshTools.CMesh()
        cmeshTools.generateFromTriangleMesh(self.cmesh,ctrirep,base)
        cmeshTools.allocateGeometricInfo_triangle(self.cmesh)
        cmeshTools.computeGeometricInfo_triangle(self.cmesh)
        self.buildFromC(self.cmesh)
    def generateFromTriangleFiles(self,filebase,base):
        from .import cmeshTools
        self.cmesh = cmeshTools.CMesh()
        cmeshTools.generateFromTriangleFiles(self.cmesh,filebase,base)
        cmeshTools.allocateGeometricInfo_triangle(self.cmesh)
        cmeshTools.computeGeometricInfo_triangle(self.cmesh)
        self.buildFromC(self.cmesh)
    def writeTriangleFiles(self,filebase,base):
        from .import cmeshTools
        cmeshTools.writeTriangleFiles(self.cmesh,filebase,base)
        
    def generateFromArrays(self,nodes,elements,nodeMaterialTypes=None,elementMaterialTypes=None):
        """Build the mesh from arrays of node coordinates and zero-based element nodes"""
        from .import cmeshTools
        self.cmesh = cmeshTools.CMesh()
        cmeshTools.generateFromArrays(self.cmesh,nodes,elements,nodeMaterialTypes,elementMaterialTypes)
        cmeshTools.allocateGeometricInfo_triangle(self.cmesh)
        cmeshTools.computeGeometricInfo_triangle(self.cmesh)
        self.buildFromC(self.cmesh)

    def generateFromYmfFile(self,prefix):
        """Build the mesh from the ymf mesh ``prefix.ymf``, ``prefix.h5``"""
        info = readMeshYmf(prefix)
        assert info.elementTopologyName == 'Triangle', info.elementTopologyName
        self.generateFromArrays(info.nodeArray,info.elementNodesArray,
                                info.nodeMaterialTypes,info.elementMaterialTypes)

    def generateFrom2DMFile(self,filebase,base=1):
        from .import cmeshTools
        self.cmesh = cmeshTools.CMesh()
        cmeshTools.generateFrom2DMFile(self.cmesh,filebase,base)
        cmeshTools.allocateGeometricInfo_triangle(self.cmesh)
        cmeshTools.computeGeometricInfo_triangle(self.cmesh)
        self.buildFromC(self.cmesh)

    def constructTriangularMeshOnRectangle(self,Lx,Ly,nx,ny,writeMesh=0,
                                           meshFileBase='mesh2d'):
        """
        wrapper function for making a triangular mesh on the rectangle
        [0,Lx] x [0,Ly].

        """
        nz = 1
        Lz = 1.0
        grid2d = RectangularGrid(nx,ny,nz,Lx,Ly,Lz)
        #grid2d.writeEdgesGnuplot('grid2d')
        #grid2d.viewMeshGnuplotPipe('grid2d')

        self.rectangularToTriangular(grid2d)

        if writeMesh == 1:
            #print mesh in gnuplot format
            self.writeEdgesGnuplot(meshFileBase)
            #can view with
            #self.viewMeshGnuplotPipe(meshFileBase)
        elif writeMesh == 2:
            self.writeEdgesMatlab(meshFileBase)
            #view in matlab with meshFileBase.m
        #end else

        return self

    def buildFromSets(self,triangleSet,edgeSet,nodeSet):
        self.nodeList = list(nodeSet)
        self.nodeDict = dict([(n,n) for n in self.nodeList])
        self.edgeList = list(edgeSet)
        self.edgeDict = dict([(e.nodes,e) for e in self.edgeList])
        self.triangleList = list(triangleSet)
        self.triangleDict = dict([(t.nodes,t) for t in self.triangleList])
        self.elementList = self.triangleList
        self.elementBoundaryList = self.edgeList
    def fixLocalNumbering(self):
        for tN in range(len(self.triangleList)):
            self.triangleList[tN].computeGeometricInfo()
            if edet(self.triangleList[tN].linearMap) < 0:
                newNodes = list(self.triangleList[tN].nodes)
                newNodes[2] = self.triangleList[tN].nodes[1]
                newNodes[1] = self.triangleList[tN].nodes[2]
                self.triangleList[tN].nodes = newNodes
    def finalize(self):
        self.buildLists()
        #self.fixLocalNumbering()
        self.buildArraysFromLists()
        #todo: build boundary mesh

    def buildLists(self):
        self.buildListsNodes()
        self.buildListsEdges()
        self.buildListsTriangles()
        self.elementList = self.triangleList
        self.elementBoundaryList = self.edgeList

    def buildListsNodes(self):
        keyList = list(self.nodeDict.keys())
        keyList.sort()
        self.nodeList=[]
        self.oldToNewNode=list(range(len(self.nodeDict)))
        for nN,k in enumerate(keyList):
            self.oldToNewNode[self.nodeDict[k].N]=nN
            self.nodeDict[k].N = nN
            self.nodeList.append(self.nodeDict[k])

    def buildListsEdges(self):
        keyList = list(self.edgeDict.keys())
        keyList.sort()
        self.edgeList=[]
        for eN,k in enumerate(keyList):
            self.edgeDict[k].N = eN
            self.edgeList.append(self.edgeDict[k])

    def buildListsTriangles(self):
        keyList = list(self.triangleDict.keys())
        keyList.sort()
        self.triangleList=[]
        for tN,k in enumerate(keyList):
            self.triangleDict[k].N = tN
            self.triangleList.append(self.triangleDict[k])
        self.polygonList = self.triangleList

    def newTriangle(self,nodes):
        t = Triangle(len(self.triangleDict),nodes)
        self.triangleDict[t.nodes] = t
        self.registerEdges(t)
        return t

    def registerEdges(self,t):
        for en,e in enumerate(t.edges):
            if e.nodes in self.edgeDict:
                t.edges[en]=self.edgeDict[e.nodes]
            else:
                eN=len(self.edgeDict)
                e.N=eN
                self.edgeDict[e.nodes]=e

    def registerNode(self,node):
        if node in self.nodeDict:
            node = self.nodeDict[node]
        else:
            node.N = len(self.nodeDict)
            self.nodeDict[node] = node
        return node
    def buildLevelSetMesh(self,value,nodalValues):
        levelSetMesh = EdgeMesh()
        self.levelSetNodeNumbers = set()
        for t in self.triangleList:
            nodes={}
            for e in t.edges:
                nl = e.nodes[0]
                vl = nodalValues[nl.N]
                nr = e.nodes[1]
                vr = nodalValues[nr.N]
                if ((vl >= value and value >= vr) or
                    (vl <= value and value <= vr)):
                    if vl == vr:
                        newNl = Node(len(levelSetMesh.nodeDict),
                                     nl.p[X],
                                     nl.p[Y],
                                     nl.p[Z])
                        newNl = levelSetMesh.registerNode(newNl)
                        newNr = Node(len(levelSetMesh.nodeDict),
                                     nr.p[X],
                                     nr.p[Y],
                                     nr.p[Z])
                        newNr = levelSetMesh.registerNode(newNr)
                        levelSetMesh.newEdge([newNl,newNr])
                        self.levelSetNodeNumbers.add(nl.N)
                        self.levelSetNodeNumbers.add(nr.N)
                    elif value == vl:
                        newNode = Node(len(levelSetMesh.nodeDict),
                                       nl.p[X],
                                       nl.p[Y],
                                       nl.p[Z])
                        nodes[newNode] = newNode
                        self.levelSetNodeNumbers.add(nl.N)
                    elif value == vr and len(nodes) < 2:
                        newNode = Node(len(levelSetMesh.nodeDict),
                                       nr.p[X],
                                       nr.p[Y],
                                       nr.p[Z])
                        nodes[newNode] = newNode
                        self.levelSetNodeNumbers.add(nr.N)
                    else:
                        wr = (value-vl)/(vr-vl)
                        wl = (value-vr)/(vl-vr)
                        newPoint = nl.p*wl + nr.p*wr
                        newNode = Node(len(levelSetMesh.nodeDict),
                                       newPoint[X],
                                       newPoint[Y],
                                       newPoint[Z])
                        nodes[newNode] = newNode
                        self.levelSetNodeNumbers.add(nl.N)
                        self.levelSetNodeNumbers.add(nr.N)
                elif vl < value:
                    self.levelSetNodeNumbers.add(nl.N)
                elif vr < value:
                    self.levelSetNodeNumbers.add(nr.N)
            if len(nodes) == 0:
                pass
            elif len(nodes) == 1:
                print("singleton")
            elif len(nodes) == 2:
                newNodes=[]
                for n in list(nodes.values()):
                    newNodes.append(levelSetMesh.registerNode(n))
                levelSetMesh.newEdge(newNodes)
            else:
                print("unexpected case in buildLevelSetMesh")
                print(t.N)
                for e in t.edges:
                    print(e.N)
                    for n in e.nodes:
                        print(n.N)
                        print(n.p)
                print("level set triangle")
                for n in list(nodes.values()):
                    print(n.p)
        if len(levelSetMesh.edgeDict) == 0:
            print("level set does not cross any edges")
            return None
        else:
            levelSetMesh.finalize()
        return levelSetMesh
    def refine3t(self,oldMesh):
        childrenDict={}
        for t in oldMesh.triangleList:
            #deep copy old nodes because we'll renumber
            tNodes = [Node(eN,n.p[X],n.p[Y],n.p[Z])
                      for eN,n in enumerate(t.nodes)]
            for lnN,n in enumerate(tNodes): tNodes[lnN]=self.registerNode(n)
            #add new node
            t.computeGeometricInfo()
            newNode = Node(len(self.nodeDict),
                           t.barycenter[X],
                           t.barycenter[Y],
                           t.barycenter[Z])
            newNode = self.registerNode(newNode)
            t1=self.newTriangle([tNodes[0],tNodes[1],newNode])
            t2=self.newTriangle([tNodes[1],tNodes[2],newNode])
            t3=self.newTriangle([tNodes[2],tNodes[0],newNode])
            childrenDict[t.N]=[t1,t2,t3]
        self.finalize()
        return childrenDict

    def refineFreudenthalBey(self,oldMesh):
        logEvent("Refining the mesh using Freudenthal-Bey refinement")
        childrenDict={}
        for t in list(oldMesh.triangleDict.values()):
            #deep copy old nodes because we'll renumber
            tNodes = [Node(nN,n.p[X],n.p[Y],n.p[Z])
                      for nN,n in enumerate(t.nodes)]
            for lnN,n in enumerate(tNodes): tNodes[lnN]=self.registerNode(n)
            #add new nodes (midpoints of edges)
            #use local edge tuples as keys
            newNodes={}
            for et,en in t.edgeMap.items():
                t.edges[en].computeGeometricInfo()
                p = t.edges[en].barycenter
                newNodes[et] = Node(en,p[X],p[Y],p[Z])

            #set the global node numbers
            for k,n in newNodes.items(): newNodes[k]=self.registerNode(n)
            #add corner triangles
            t1=self.newTriangle([tNodes[0],
                              newNodes[(0,1)],
                              newNodes[(0,2)]])
            t2=self.newTriangle([tNodes[1],
                              newNodes[(0,1)],
                              newNodes[(1,2)]])
            t3=self.newTriangle([tNodes[2],
                              newNodes[(0,2)],
                              newNodes[(1,2)]])
            #add center triangle
            t4=self.newTriangle([newNodes[(0,1)],
                              newNodes[(1,2)],
                              newNodes[(0,2)]])
            childrenDict[t.N]=[t1,t2,t3,t4]
        self.finalize()
        return childrenDict
        #for debugging: print each tet
        #self.edgeList=[]
        #Tlist = self.tetrahedronDict.values()
        #for T in Tlist:
        #    self.edgeList = self.edgeList + T.edges

    def refine(self,oldMesh):
        return self.refineFreudenthalBey(oldMesh)

    def meshInfo(self):
        minfo = """Number of triangles  : %d
Number of edges : %d
Number of nodes : %d\n""" % (self.nElements_global,
                             self.nElementBoundaries_global,
                             self.nNodes_global)
        if self.subdomainMesh != self:
            sinfo = self.subdomainMesh.meshInfo()
            info = "*** Global ***\n" + minfo + "\n*** Local ***\n" + sinfo
            return info
        return minfo

    def readMeshADH(self,filename,adhBase=1,suffix='3dm'):
        meshIn = open(filename+'.'+suffix,'r')
        firstLine = meshIn.readline()
        firstWords = firstLine.split()
        print("Reading object=%s from file=%s" % (firstWords[0],filename))
        line = meshIn.readline()
        columns = line.split()
        triangles = []
        triangleEdges=set()
        logEvent("Reading "+str(filename)+ \
                " and building node lists for triangles, and edges")
        #assume triangles are ordered by triangle number
        while (columns[0] == 'E3T'):
            nodeNumbers = [int(c) - adhBase for c in columns[2:5]]
            nodeNumbers.sort()
            triangles.append(array.array('i',nodeNumbers))
            triangleEdges.update([(nodeNumbers[0],nodeNumbers[1]),
                                  (nodeNumbers[0],nodeNumbers[2]),
                                  (nodeNumbers[1],nodeNumbers[2])])
            line = meshIn.readline()
            columns = line.split()
        print("Building node list and dict")
        #assume nodes are ordered by node number
        while (len(columns) == 5):
            newNode = Node(int(columns[1]) - adhBase,
                           float(columns[2]),
                           float(columns[3]),
                           float(columns[4]))
            self.nodeList.append(newNode)
            self.nodeDict[newNode]=newNode
            line = meshIn.readline()
            columns = line.split()
        print("Number of triangles :"+str(len(triangles)))
        print("Number of edges     :"+str(len(triangleEdges)))
        print("Number of nodes     :"+str(len(self.nodeList)))
        print("Number of objects   :"+\
              str(len(triangleEdges)+len(triangles)+len(self.nodeList)))
        print("Building edge list")
        self.edgeList =[Edge(edgeNumber=eN,nodes=[self.nodeList[nN[0]],
                                                  self.nodeList[nN[1]]])
                        for eN,nN in enumerate(triangleEdges)]
        print("Building edge dict")
        self.edgeDict = dict([(e.nodes,e) for e in self.edgeList])
        print("Building triangle list")
        self.triangleList =[Triangle(triangleNumber=tN,
                                     nodes=[self.nodeList[nN[0]],
                                            self.nodeList[nN[1]],
                                            self.nodeList[nN[2]]],
                                     edgeDict=self.edgeDict)
                            for tN,nN in enumerate(triangles)]
        print("Building triangle dict")
        self.triangleDict = dict([(t.nodes,t) for t in self.triangleList])
        self.elementList = self.triangleList
        self.elementBoundaryList = self.edgeList

    def writeMeshXdmf(self,ar,name='',t=0.0,init=False,meshChanged=False,tCount=0,EB=False):
        Mesh.writeMeshXdmf(self,ar,name,t,init,meshChanged,"Triangle",tCount,EB=EB)
    def writeMeshEnsight(self,filename,description=None):
        base=1
        #write the casefile
        caseOut=open(filename+'.case','w')
        caseOut.write('FORMAT\n'+'type: ensight gold\n')
        caseOut.write('GEOMETRY\n'+'model: '+filename+'.geo\n')
        caseOut.close()
        meshOut=open(filename+'.geo','w')
        meshOut.write('Ensight Gold\n')
        meshOut.write('Unstructured Triangular Mesh\n')
        meshOut.write('node id given\n')
        meshOut.write('element id given\n')
        meshOut.write('part \n'+'%10i\n' % 1)
        if description:
            meshOut.write(description+'\n')
        else:
            meshOut.write('A Mesh\n')
        meshOut.write('coordinates\n'+'%10i\n' % self.nNodes_global)
        for nN in range(self.nNodes_global):
            meshOut.write('%10i\n' % (nN+base))
        for nN in range(self.nNodes_global):
            meshOut.write('%12.5E\n' % self.nodeArray[nN,0])
        for nN in range(self.nNodes_global):
            meshOut.write('%12.5E\n' % self.nodeArray[nN,1])
        for nN in range(self.nNodes_global):
            meshOut.write('%12.5E\n' % self.nodeArray[nN,2])
        meshOut.write('tria3\n'+'%10i\n' % self.nElements_global)
        for eN in range(self.nElements_global):
            meshOut.write('%10i\n' % (eN+base))
        for eN in range(self.nElements_global):
            meshOut.write('%10i%10i%10i\n' % tuple((nN+base) for nN in self.elementNodesArray[eN,:]))
        meshOut.close()

    def appendMeshEnsight(self,filename,description=None):
        base=1
        #write the casefile
        caseOut=open(filename+'.case','a')
        caseOut.write('GEOMETRY\n'+'model: '+filename+'.geo\n')
        caseOut.close()
        meshOut=open(filename+'.geo','w')
        meshOut.write('Unstructured Triangular Mesh\n\n')
        meshOut.write('node id given\n')
        meshOut.write('element id given\n')
        meshOut.write('part \n'+'%10i\n' % 1)
        if description:
            meshOut.write(description+'\n')
        else:
            meshOut.write('The whole mesh\n')
        meshOut.write('coordinates\n'+'%10i\n' % len(self.nodeList))
        for n in self.nodeList:
            nN = n.N+base
            meshOut.write('%10i\n' % nN)
        for n in self.nodeList:
            meshOut.write('%12.5E\n' % n.p[X])
        for n in self.nodeList:
            meshOut.write('%12.5E\n' % n.p[Y])
        for n in self.nodeList:
            meshOut.write('%12.5E\n' % n.p[Z])
        meshOut.write('tria3\n'+'%10i\n' % len(self.elementList))
        for e in self.elementList:
            eN = e.N + base
            meshOut.write('%10i\n' % eN)
        for e in self.elementList:
            meshOut.write('%10i%10i%10i\n' % tuple(n.N+base for n in e.nodes))
        meshOut.close()

    def writeMeshADH(self,filename,adhBase=1):
        from .import cmeshTools
        cmeshTools.write2dmFiles(self.cmesh,filename,adhBase)
    def writeAsymptote(self,fileprefix,L,x,units="m"):
        """
        Write a representation of the triangular mesh in the Asymptote vector graphics language
        """
        unitsize=4.0/L[0]
        f = open(fileprefix+".asy",'w')
        fileString="""
unitsize(4.0 inches / %(Lx)f);
size(5 inches);
real Lx=%(Lx)f;
real Ly=%(Ly)f;
real offset=0.0125Lx;
real x=%(x)f;
real y=%(y)f;
string strx="$%(Lx)2.2f\\mbox{%(units)s}$";
string stry="$%(Ly)2.2f\\mbox{%(units)s}$";
draw(strx,(x,y-offset)--(x+Lx,y-offset),S,black,Bars,Arrows,PenMargins);
draw(stry,(x-offset,y)--(x-offset,y+Ly),W,black,Bars,Arrows,PenMargins);
import graph;
import palette;
pen[] regionPens = Rainbow(NColors=%(nRegionFlags)d);
pen[] boundaryPens = Rainbow(NColors=%(nBoundaryFlags)d);
""" % {'Lx':L[0],'Ly':L[1],'x':x[0],'y':x[1],'units':units,
       'nRegionFlags':(max(self.elementMaterialTypes) - min(self.elementMaterialTypes)),
       'nBoundaryFlags':(max(self.elementBoundaryMaterialTypes)-min(self.elementBoundaryMaterialTypes))}
        #now draw triangles
        for t,tFlag in zip(self.elementNodesArray,self.elementMaterialTypes):
            fileString+="fill((%f,%f)--(%f,%f)--(%f,%f)--cycle,regionPens[%d]);\n" % (self.nodeArray[t[0]][0],self.nodeArray[t[0]][1],
                                                                                     self.nodeArray[t[1]][0],self.nodeArray[t[1]][1],
                                                                                     self.nodeArray[t[2]][0],self.nodeArray[t[2]][1],
                                                                                     tFlag-min(self.elementMaterialTypes))
        for eb,ebFlag in zip(self.elementBoundaryNodesArray,self.elementBoundaryMaterialTypes):
            if True:#ebFlag > 0:
                fileString+="draw((%f,%f)--(%f,%f),boundaryPens[%d]+linewidth(0.01));\n" % (self.nodeArray[eb[0]][0],self.nodeArray[eb[0]][1],
                                                                                           self.nodeArray[eb[1]][0],self.nodeArray[eb[1]][1],
                                                                                           ebFlag-min(self.elementBoundaryMaterialTypes))
        f.write(fileString)
        f.close()

#     def buildMatlabMeshDataStructures(self,meshFileBase='meshMatlab',writeToFile=True):
#         """
#         build array data structures for matlab finite element mesh representation
#         and write to a file to view and play with in matlatb

#         in matlab can then print mesh with

#         pdemesh(p,e,t)

#         where

#           p is the vertex or point matrix
#           e is the edge matrix, and
#           t is the element matrix

#         points matrix is [2 x num vertices]
#           format :
#              row 1 = x coord,
#              row 2 = y coord for nodes in mesh

#         edge matrix is [7 x num edges]
#           format:
#              row 1 = start vertex number
#              row 2 = end vertex number
#              row 3 = start value in edge parameterization, should be 0
#              row 4 = end   value in edge parameterization, should be 1
#              row 5 = global edge id, base 1
#              row 6 = subdomain on left? always 1 for now
#              row 7 = subdomain on right? always 0 for now

#         element matrix is [4 x num elements]
#             row 1 = vertex 1 global number
#             row 2 = vertex 2 global number
#             row 3 = vertex 3 global number
#             row 4 = triangle subdomain number
#          where 1,2,3 is a local counter clockwise numbering of vertices in
#            triangle

#          """
#         matlabBase = 1
#         p = np.zeros((2,self.nNodes_global),'d')
#         e = np.zeros((7,self.nElementBoundaries_global),'d')
#         t = np.zeros((4,self.nElements_global),'d')

#         #load p,e,t and write file
#         if writeToFile:
#             mfile = open(meshFileBase+'.m','w')
#         else:
#             mfile = open('/dev/null','w')
#         #
#         if writeToFile:
#             mfile.write('p = [ ... \n')
#         for nN in range(self.nNodes_global):
#             p[0,nN]=self.nodeArray[nN,0]
#             p[1,nN]=self.nodeArray[nN,1]
#             if writeToFile:
#                 mfile.write('%g %g \n' % tuple(p[:,nN]))
#         if writeToFile:
#             mfile.write(']; \n')
#             mfile.write("p = p\';\n")  #need transpose for matlab

#         if writeToFile:
#             mfile.write('e = [ ... \n')
#         for ebN in range(self.nElementBoundaries_global):
#             e[0,ebN]=self.elementBoundaryNodesArray[ebN,0] + matlabBase #global node number of start node base 1
#             e[1,ebN]=self.elementBoundaryNodesArray[ebN,1] + matlabBase #global node number of end node base 1
#             e[2,ebN]=0.0 #edge param. is 0 to 1
#             e[3,ebN]=1.0
#             e[4,ebN]=ebN + matlabBase  #global edge number base 1
#             e[5,ebN]=0 #subdomain to left
#             e[6,ebN]=1 #subdomain to right
#             if writeToFile:
#                 mfile.write('%g %g %g %g %g %g %g \n' % tuple(e[:,ebN]))
#         if writeToFile:
#             mfile.write(']; \n')
#             mfile.write("e = e\';\n")  #need transpose for matlab

#         #write triangles last
#         if writeToFile:
#             mfile.write('t = [ ... \n')
#         for eN in range(self.nElements_global):
#             t[0,eN]=self.elementNodesArray[eN,0]+matlabBase    #global node number for vertex 0
#             t[1,eN]=self.elementNodesArray[eN,1]+matlabBase    #global node number for vertex 0
#             t[2,eN]=self.elementNodesArray[eN,2]+matlabBase    #global node number for vertex 0
#             t[3,eN]=1                     #subdomain id
#             if writeToFile:
#                 mfile.write('%g %g %g %g \n' % tuple(t[:,eN]))
#         if writeToFile:
#             mfile.write(']; \n');
#             mfile.write("t = t\';\n") #need transpose for matlab



class QuadrilateralMesh(Mesh):
    """A mesh of quads

    The nodes, edges, and triangles are indexed by their
    node tuples. The corresponding lists are derived from the dictionaries, and
    sorted lexicographically. The global node numbers are redefined to
    give a lexicographic ordering.

    The mesh can be generated from a rectangular grid and refined using either
    3t or Freudenthal-Bey global refinement.
    """

    def __init__(self):
        Mesh.__init__(self)
        self.nodeDict={}
        self.edgeDict={}
        self.quadDict={}
        self.quadList=[]
        self.oldToNewNode=[]
        # tempoaray
        self.max_nNodeNeighbors_node = 4


    def buildFromSets(self,faceSet,edgeSet,nodeSet):
        self.nodeList = list(nodeSet)
        self.nodeDict = dict([(n,n) for n in self.nodeList])
        self.edgeList = list(edgeSet)
        self.edgeDict = dict([(e.nodes,e) for e in self.edgeList])
        self.quadList = list(faceSet)
        self.quadDict = dict([(t.nodes,t) for t in self.faceList])
        self.elementList = self.triangleList
        self.elementBoundaryList = self.edgeList

    def rectangularToQuadrilateral(self,grid,x=0.0,y=0.0,z=0.0):
        ''' WIP - I think this is the first function that needs to be
            written so that MultilevelQuadrilateralMesh can work.  This
            function does not call C functions.
        '''
        self.nodeList = [Node(n.N,n.p[X]+x,n.p[Y]+y,n.p[Z]+z) for n in grid.nodeList]
        # Is the following line necessary?
        self.nodeDict = dict([(n,n) for n in self.nodeList])
        for i in range(grid.nHx):
            for j in range(grid.nHy):
                k=0
                n0 = self.nodeList[grid.getNodeNumber(i,j,k)]
                n1 = self.nodeList[grid.getNodeNumber(i,j+1,k)]
                n2 = self.nodeList[grid.getNodeNumber(i+1,j+1,k)]
                n3 = self.nodeList[grid.getNodeNumber(i+1,j,k)]
                e0 = Edge(nodes=[n0,n1])
                e1 = Edge(nodes=[n1,n2])
                e2 = Edge(nodes=[n2,n3])
                e3 = Edge(nodes=[n3,n0])
                self.newQuadrilateral([e0,e1,e2,e3])
        self.finalize()
        self.buildNodeDiameterArray()

    def generateQuadrilateralMeshFromRectangularGrid(self,nx,ny,Lx,Ly):
        from .import cmeshTools
        self.cmesh = cmeshTools.CMesh()
        cmeshTools.generateQuadrilateralMeshFromRectangularGrid(nx,ny,0,0,Lx,Ly,self.cmesh)
        cmeshTools.allocateGeometricInfo_quadrilateral(self.cmesh)
        cmeshTools.computeGeometricInfo_quadrilateral(self.cmesh)
        self.buildFromC(self.cmesh)
        
    def generateFromQuadFileIFISS(self,meshfile):
        ''' WIP - read a matlab.mat file containing IFISS vertices
        and elements
        '''
        import scipy.io
        griddata = scipy.io.loadmat(meshfile+'.mat')
        self.nodeList = [Node(nN,n[0],n[1],0.0) for nN,n in enumerate(griddata['vertices'])]
        # Is the following line necessary?
        self.nodeDict = dict([(n,n) for n in self.nodeList])
        for q in griddata['quads']:
            n0,n3,n2,n1 = q # clockwise ordering needed
            e0 = Edge(nodes=[self.nodeList[n0],self.nodeList[n1]])
            e1 = Edge(nodes=[self.nodeList[n1],self.nodeList[n2]])
            e2 = Edge(nodes=[self.nodeList[n2],self.nodeList[n3]])
            e3 = Edge(nodes=[self.nodeList[n3],self.nodeList[n0]])
            self.newQuadrilateral([e0,e1,e2,e3])
        self.finalize()
        for F,nN in griddata['bdyflags']:
            self.nodeMaterialTypes[nN] = F
        for ebNE in range(self.nExteriorElementBoundaries_global):
            ebN = self.exteriorElementBoundariesArray[ebNE]
            n0,n1 = self.elementBoundaryNodesArray[ebN]
            self.elementBoundaryMaterialTypes[ebN]=max(self.nodeMaterialTypes[n0],
                                                       self.nodeMaterialTypes[n1])

    def meshType(self):
        return 'cuboid'

    def meshInfo(self):
        minfo = """Number of quadrilaterals  : %d
Number of edges : %d
Number of nodes : %d\n""" % (self.nElements_global,
                             self.nElementBoundaries_global,
                             self.nNodes_global)
        if self.subdomainMesh != self:
            sinfo = self.subdomainMesh.meshInfo()
            info = "*** Global ***\n" + minfo + "\n*** Local ***\n" + sinfo
            return info
        return minfo

    def newQuadrilateral(self,edges):
        q = Quadrilateral(len(self.quadDict),edges)
        self.quadDict[q.nodes] = q
        self.registerEdges(q)
        return q

    def registerEdges(self,q):
        '''check if an edge is in the mesh dictionary
           if it is, point to existing entry
           otherwise, create a new entry
        '''
        for en,e in enumerate(q.edges):
            if e.nodes in self.edgeDict:
                q.edges[en]=self.edgeDict[e.nodes]
            else:
                eN=len(self.edgeDict)
                e.N=eN
                self.edgeDict[e.nodes]=e

    def registerNode(self,node):
        ''' check if a node is in the mesh dictionary
            if it is, point to existing entry
            otherwise, create a new entry
        '''
        if node in self.nodeDict:
            node = self.nodeDict[node]
        else:
            node.N = len(self.nodeDict)
            self.nodeDict[node] = node
        return node

    def refine(self,oldMesh):
        logEvent("Refining Using Standard Quadrilateral Refinement")
        childrenDict={}
        for q in list(oldMesh.quadDict.values()):
            qNodes = [Node(nN,n.p[X],n.p[Y],n.p[Z]) for nN,n in enumerate(q.nodes)]
            for lnN,n in enumerate(qNodes): qNodes[lnN] = self.registerNode(n)
            q.computeGeometricInfo()

            newNodeLeft = Node(len(self.nodeDict),q.xmin,q.ymid,q.zmid)
            newNodeLeft = self.registerNode(newNodeLeft)
            newNodeTop = Node(len(self.nodeDict),q.xmid,q.ymax,q.zmid)
            newNodeTop = self.registerNode(newNodeTop)
            newNodeRight = Node(len(self.nodeDict),q.xmax,q.ymid,q.zmid)
            newNodeRight = self.registerNode(newNodeRight)
            newNodeBottom = Node(len(self.nodeDict),q.xmid,q.ymin,q.zmid)
            newNodeBottom = self.registerNode(newNodeBottom)
            newNodeMid = Node(len(self.nodeDict),q.xmid,q.ymid,q.zmid)
            newNodeMid = self.registerNode(newNodeMid)

            e1 = Edge(nodes=[qNodes[0],newNodeLeft])
            e2 = Edge(nodes=[newNodeLeft,newNodeMid])
            e3 = Edge(nodes=[newNodeMid,newNodeBottom])
            e4 = Edge(nodes=[newNodeBottom,qNodes[0]])
            e5 = Edge(nodes=[newNodeLeft,qNodes[1]])
            e6 = Edge(nodes=[qNodes[1],newNodeTop])
            e7 = Edge(nodes=[newNodeTop,newNodeMid])
            e8 = Edge(nodes=[newNodeTop,qNodes[2]])
            e9 = Edge(nodes=[qNodes[2], newNodeRight])
            e10 = Edge(nodes=[newNodeRight, newNodeMid])
            e11 = Edge(nodes=[qNodes[3],newNodeBottom])
            e12 = Edge(nodes=[newNodeRight,qNodes[3]])

            q1 = self.newQuadrilateral([e1,e2,e3,e4])
            self.registerEdges(q1)
            q2 = self.newQuadrilateral([e5,e6,e7,e2])
            self.registerEdges(q2)
            q3 = self.newQuadrilateral([e3,e10,e12,e11])
            self.registerEdges(q3)
            q4 = self.newQuadrilateral([e7,e8,e9,e10])
            self.registerEdges(q4)

            childrenDict[q.N]=[q1,q2,q3,q4]
        self.finalize()
        return childrenDict

    def finalize(self):
        ''' WIP '''
        self.buildLists()
        self.buildArraysFromLists()

    def buildLists(self):
        ''' WIP '''
        self.buildListsNodes()
        self.buildListsEdges()
        self.buildListsQuadrilaterals()
        self.elementList = self.quadList
        self.elementBoundaryList = self.edgeList

    def buildListsNodes(self):
        keyList = list(self.nodeDict.keys())
        keyList.sort()
        self.nodeList=[]
        self.oldToNewNode=list(range(len(self.nodeDict)))
        for nN,k in enumerate(keyList):
            self.oldToNewNode[self.nodeDict[k].N]=nN
            self.nodeDict[k].N = nN
            self.nodeList.append(self.nodeDict[k])

    def buildListsEdges(self):
        keyList = list(self.edgeDict.keys())
        keyList.sort()
        self.edgeList=[]
        for eN,k in enumerate(keyList):
            self.edgeDict[k].N = eN
            self.edgeList.append(self.edgeDict[k])

    def buildListsQuadrilaterals(self):
        keyList = list(self.quadDict.keys())
        keyList.sort()
        self.quadList = []
        for qN,q in enumerate(keyList):
            self.quadDict[q].N = qN
            self.quadList.append(self.quadDict[q])
        self.polygonList = self.quadList

    def writeMeshXdmf(self,ar,name='',t=0.0,init=False,meshChanged=False,tCount=0,EB=False):
        Mesh.writeMeshXdmf(self,ar,name,t,init,meshChanged,"Quadrilateral",tCount,EB=EB)

    def buildNodeDiameterArray(self):
        nNodes = len(self.nodeArray)
        self.nodeDiametersArray = np.zeros(nNodes)
        self.nodeSupportArray = np.zeros(nNodes)
        self.volume = 0.

        for eN in range(self.nElements_global):
            area = self._calc_quad_area(eN)
            self.volume += area
            hMax = self.elementDiametersArray[eN]

            for nN in range(self.nNodes_element):
                nodeDiameter = hMax*area
                idx = self.elementNodesArray[eN][nN]
                self.nodeDiametersArray[idx]+=nodeDiameter
                self.nodeSupportArray[idx]+=area

        for nN in range(nNodes):
            self.nodeDiametersArray[nN] /= self.nodeSupportArray[nN]

    @staticmethod
    def _calc_pt_distance(pt1,pt2):
        """ Calculate the distance between two points.

        Arguments
        ---------
        pt1: lst
            Coordinates of the first point
        pt2: lst
            Coordinates of the second point

        Returns
        -------
        distance : float
        """
        d = 0.
        for i,j in zip(pt1,pt2):
            d += (i-j)**2
        return math.sqrt(d)

    def _calc_hmax(self,i):
        """ Find the largest edge length of an element.

        Arguments
        ---------
        i : int
            Element number

        Returns
        -------
        hmax : float
            The largest edge length of element i
        """
        hMax = 0.
        element_nodes = self.nodeArray[self.elementNodesArray[i]]
        for j, nN_L in enumerate(element_nodes):
            print('nN_L = ' + str(nN_L))
            for nN_R in element_nodes[j+1:]:
                print('nN_R = ' + str(nN_R))
                hMax = max(hMax,self._calc_pt_distance(nN_L,nN_R))
        return hMax

    def _calc_quad_area(self,i):
        """ Calculates the area of a quadrilateral.

        Arguments
        ---------
        i : int
            The quadrilateral whose volume is being calculated.

        Returns
        -------
        A : float
            The quadrilateral's area
        """
        n = [n0,n1,n2,n3] = self.nodeArray[self.elementNodesArray[0]]
        d = [self._calc_pt_distance(n0,n[1]),
             self._calc_pt_distance(n0,n[-1])]
        A = d[0]*d[1]
        return A

class MultilevelTriangularMesh(MultilevelMesh):
    """A hierarchical  multilevel mesh of triangular cells"""
    from .import cmeshTools
    def __init__(self,
                 nx, ny, nz,
                 x=0.0, y=0.0, z=0.0,
                 Lx=1.0, Ly=1.0, Lz=1.0,
                 refinementLevels=1,
                 skipInit=False,
                 nLayersOfOverlap=1,
                 parallelPartitioningType=MeshParallelPartitioningTypes.node,
                 triangleFlag=0,
                 regularGrid=None):
        from . import cmeshTools
        MultilevelMesh.__init__(self)
        self.useC = True
        self.nLayersOfOverlap=nLayersOfOverlap; self.parallelPartitioningType = parallelPartitioningType
        #self.useC = False
        if not skipInit:
            if self.generatesInPieces('tri', triangleFlag, nLayersOfOverlap, parallelPartitioningType, regularGrid):
                self.generateCartesianHierarchy('tri', (nx,ny), (x,y,z), (Lx,Ly,Lz), refinementLevels,
                                                triangleFlag=triangleFlag)
            elif self.useC:
                self.meshList.append(TriangularMesh())
                self.meshList[0].generateTriangularMeshFromRectangularGrid(nx,ny,Lx,Ly,triangleFlag=triangleFlag)
                #Translate the coarse mesh *before* refining it. The C refinement copies
                #the coarse nodes into every finer level, so shifting level 0 afterwards
                #left the finer levels at the origin -- a different coordinate frame from
                #level 0, so geometric transfers between them were wrong and the finest
                #mesh, the one that is solved on, was not where the domain is.
                self.meshList[0].nodeArray[:,0] += x
                self.meshList[0].nodeArray[:,1] += y
                self.meshList[0].nodeArray[:,2] += z
                self.cmultilevelMesh = cmeshTools.CMultilevelMesh(self.meshList[0].cmesh,refinementLevels)
                self.buildFromC(self.cmultilevelMesh)
                for l in range(1,refinementLevels):
                    self.meshList.append(TriangularMesh())
                    self.meshList[l].cmesh = self.cmeshList[l]
                    self.meshList[l].buildFromC(self.meshList[l].cmesh)
                self.partitionLevels(nLayersOfOverlap,parallelPartitioningType)
            else:
                grid=RectangularGrid(nx,ny,nz,Lx,Ly,Lz)
                self.meshList.append(TriangularMesh())
                self.meshList[0].rectangularToTriangular(grid)
                self.meshList[0].nodeArray[:,0] += x
                self.meshList[0].nodeArray[:,1] += y
                self.meshList[0].nodeArray[:,2] += z
                self.meshList[0].subdomainMesh = self.meshList[0]
                self.elementChildren=[]
                logEvent(self.meshList[0].meshInfo())
                for l in range(1,refinementLevels):
                    self.refine()
                    self.meshList[l].subdomainMesh = self.meshList[l]
                    logEvent(self.meshList[-1].meshInfo())
                self.buildArrayLists()
    #
    #mwf what's the best way to build from an existing mesh
    def generateFromExistingCoarseMesh(self,mesh0,refinementLevels,nLayersOfOverlap=1,
                                       parallelPartitioningType=MeshParallelPartitioningTypes.node):
        from .import cmeshTools
        #blow away or just trust garbage collection
        
        self.nLayersOfOverlap = nLayersOfOverlap; self.parallelPartitioningType = parallelPartitioningType
        self.meshList = []
        self.elementParents = None
        self.cmultilevelMesh = None
        if self.useC:
            self.meshList.append(mesh0)
            self.cmultilevelMesh = cmeshTools.CMultilevelMesh(self.meshList[0].cmesh,refinementLevels)
            self.buildFromC(self.cmultilevelMesh)
            for l in range(1,refinementLevels):
                self.meshList.append(TriangularMesh())
                self.meshList[l].cmesh = self.cmeshList[l]
                self.meshList[l].buildFromC(self.meshList[l].cmesh)
            self.partitionLevels(nLayersOfOverlap,parallelPartitioningType)
        else:
            grid=RectangularGrid(nx,ny,nz,Lx,Ly,Lz)
            self.meshList.append(TriangularMesh())
            self.meshList[0].rectangularToTriangular(grid)
            self.meshList[0].subdomainMesh = self.meshList[0]
            self.elementChildren=[]
            logEvent(self.meshList[0].meshInfo())
            for l in range(1,refinementLevels):
                self.refine()
                self.meshList[l].subdomainMesh = self.meshList[l]
                logEvent(self.meshList[-1].meshInfo())
            self.buildArrayLists()
    def generatePartitionedMeshFromPUMI(self,mesh0,refinementLevels,nLayersOfOverlap=1):
        from .import cmeshTools
        self.meshList = []
        self.meshList.append(mesh0)
        self.cmultilevelMesh = cmeshTools.CMultilevelMesh(self.meshList[0].cmesh,refinementLevels)
        self.buildFromC(self.cmultilevelMesh)
        self.elementParents = None
        self.elementChildren=[]

    def generatePartitionedMeshFromTriangleFiles(self,filebase,base,mesh0,refinementLevels,nLayersOfOverlap=1,
                                               parallelPartitioningType=MeshParallelPartitioningTypes.node):
        from . import cmeshTools
        if filebase==None:
            filebase="mesh"
        assert(refinementLevels==1)
        assert(parallelPartitioningType==MeshParallelPartitioningTypes.node)
        assert(nLayersOfOverlap<=1)
        mesh0.cmesh = cmeshTools.CMesh()
        #blow away or just trust garbage collection
        self.nLayersOfOverlap=nLayersOfOverlap;self.parallelPartitioningType=parallelPartitioningType
        self.meshList = []
        self.elementParents = None
        self.cmultilevelMesh = None
        self.meshList.append(mesh0)
        logEvent("cmeshTools.CMultilevelMesh")
        self.cmultilevelMesh = cmeshTools.CMultilevelMesh(self.meshList[0].cmesh,refinementLevels)
        logEvent("buildFromC")
        self.buildFromC(self.cmultilevelMesh)
        logEvent("partitionMesh")
        self.meshList[0].partitionMeshFromFiles(filebase,base,nLayersOfOverlap=nLayersOfOverlap,parallelPartitioningType=parallelPartitioningType)
        
    def refine(self):
        self.meshList.append(TriangularMesh())
        childrenDict = self.meshList[-1].refine(self.meshList[-2])
        self.elementChildren.append(childrenDict)
        
    def computeGeometricInfo(self):
        for m in self.meshList:
            m.computeGeometricInfo()
    def locallyRefine(self,elementTagArray,flagForRefineType=0):
        """
        simple local refinement assuming elementTagArray[eN]=1 --> bisect

        flagForRefineType = 0 -- newest node, 1 -- 4T, 2 -- U4T
        """
        logEvent("MultilevelTriangularMesh:locallyRefine")
        if flagForRefineType == 0:
            logEvent("MultilevelTriangularMesh: calling cmeshTools.setNewestNodeBases")
            self.cmeshTools.setNewestNodeBases(2,self.cmultilevelMesh)
        if self.useC:
            logEvent("MultilevelTriangularMesh: calling locallRefineMultilevelMesh")
            self.cmeshTools.locallyRefineMultilevelMesh(2,self.cmultilevelMesh,elementTagArray,flagForRefineType)
            logEvent("MultilevelTriangularMesh: calling buildFromC")
            self.buildFromC(self.cmultilevelMesh)
            self.meshList.append(TriangularMesh())
            self.meshList[self.nLevels-1].cmesh = self.cmeshList[self.nLevels-1]
            self.meshList[self.nLevels-1].buildFromC(self.meshList[self.nLevels-1].cmesh)
            self.meshList[self.nLevels-1].partitionMesh(nLayersOfOverlap=self.nLayersOfOverlap,parallelPartitioningType=self.parallelPartitioningType)
        else:
            print("""locallyRefine not implemented for self.useC= %s """ % (self.useC))
        #

class MultilevelQuadrilateralMesh(MultilevelMesh):
    """ A heirarchical multilevel mesh of quadrilaterals
       WIP """
    def __init__(self,
                 nx,ny,nz,
                 x=0.0,y=0.0,z=0.0,
                 Lx=1.0,Ly=1.0,Lz=1.0,
                 refinementLevels=1,
                 skipInit=False,
                 nLayersOfOverlap=1,
                 parallelPartitioningType=MeshParallelPartitioningTypes.node,
                 triangleFlag=0,
                 useC=True,
                 regularGrid=None):
        from .import cmeshTools
        MultilevelMesh.__init__(self)
        self.useC = useC  # Implementing with C will take a bit more work. Disabling for now.
        #A regular grid needs no refinement at all: every level can be generated
        #at its own resolution (see generateStructuredHierarchy). That is the
        #default; asking for useC=False keeps the old Python refinement.
        structured = (refinementLevels > 1 and useC)
        if refinementLevels > 1 and not structured:
            logEvent("Quad refinement is not supported in C routines, switching off c-mesh");
            self.useC = False  # Currently quad refinement is not supported in C routines.
        self.nLayersOfOverlap=nLayersOfOverlap ; self.parallelPartitioningType = parallelPartitioningType
        if not skipInit:
            if self.generatesInPieces('quad', 0, nLayersOfOverlap, parallelPartitioningType, regularGrid):
                self.generateCartesianHierarchy('quad', (nx,ny), (x,y,z), (Lx,Ly,Lz), refinementLevels)
            elif structured:
                self._structuredLevels = refinementLevels
                self.generateStructuredHierarchy(
                    QuadrilateralMesh,
                    lambda mesh,c: mesh.generateQuadrilateralMeshFromRectangularGrid(c[0],c[1],Lx,Ly),
                    (nx,ny,1),(x,y,z),2,
                    nLayersOfOverlap=nLayersOfOverlap,
                    parallelPartitioningType=parallelPartitioningType)
            elif self.useC:
                self.meshList.append(QuadrilateralMesh())
                self.meshList[0].generateQuadrilateralMeshFromRectangularGrid(nx,ny,Lx,Ly)
                self.meshList[0].nodeArray[:,0] += x
                self.meshList[0].nodeArray[:,1] += y
                self.cmultilevelMesh = cmeshTools.CMultilevelMesh(self.meshList[0].cmesh,refinementLevels)
                self.buildFromC(self.cmultilevelMesh)
                for l in range(1,refinementLevels):
                    self.meshList.append(QuadrilateralMesh())
                    self.meshList[l].cmesh = self.cmeshList[l]
                    self.meshList[l].buildFromC(self.cmeshList[l])
                self.partitionLevels(nLayersOfOverlap,parallelPartitioningType)
            else:
                grid=RectangularGrid(nx,ny,nz,Lx,Ly,Lz)
                self.meshList.append(QuadrilateralMesh())
                self.meshList[0].rectangularToQuadrilateral(grid,x,y,z)
                self.meshList[0].subdomainMesh = self.meshList[0]
                self.elementChildren=[]
                logEvent(self.meshList[0].meshInfo())
                self.meshList[0].globalMesh = self.meshList[0]

                # The following four lines should be called elsewhere...Most of this is don in
                # the c-function calls that are not implemented yet for 2D quads
                self.meshList[0].nElements_owned = self.meshList[0].nElements_global
                self.meshList[0].nodeNumbering_subdomain2global = np.zeros((self.meshList[0].nNodes_global,), 'd')
                self.meshList[0].elementNumbering_subdomain2global = np.zeros((self.meshList[0].nElements_global,), 'd')
                self.meshList[0].nodeOffsets_subdomain_owned[-1] = self.meshList[0].nNodes_global
                self.meshList[0].nNodes_owned = self.meshList[0].nNodes_global
                self.meshList[0].elementOffsets_subdomain_owned[-1] = self.meshList[0].nElements_global

                for node in range(self.meshList[0].nNodes_global):
                    self.meshList[0].nodeNumbering_subdomain2global[node] = node
                for element in range(self.meshList[0].nElements_global):
                    self.meshList[0].elementNumbering_subdomain2global[element] = element

                self.meshList[0].buildNodeStarArrays()
                for l in range(1,refinementLevels):
                    self.refine()
                    self.meshList[l].subdomainMesh = self.meshList[l]
                    logEvent(self.meshList[-1].meshInfo())
                    self.meshList[l].buildNodeStarArrays()
                self.buildArrayLists()
                # print("from Python")
                # print  (self.meshList[0].nElements_global,
                #         self.meshList[0].nNodes_global,
                #         self.meshList[0].nNodes_element,
                #         self.meshList[0].nNodes_elementBoundary,
                #         self.meshList[0].nElementBoundaries_element,
                #         self.meshList[0].nElementBoundaries_global,
                #         self.meshList[0].nInteriorElementBoundaries_global,
                #         self.meshList[0].nExteriorElementBoundaries_global,
                #         self.meshList[0].max_nElements_node,
                #         self.meshList[0].nEdges_global,
                #         self.meshList[0].max_nNodeNeighbors_node,
                #         self.meshList[0].elementNodesArray,
                #         self.meshList[0].nodeElementsArray,
                #         self.meshList[0].nodeElementOffsets,
                #         self.meshList[0].elementNeighborsArray,
                #         self.meshList[0].elementBoundariesArray,
                #         self.meshList[0].elementBoundaryNodesArray,
                #         self.meshList[0].elementBoundaryElementsArray,
                #         self.meshList[0].elementBoundaryLocalElementBoundariesArray,
                #         self.meshList[0].interiorElementBoundariesArray,
                #         self.meshList[0].exteriorElementBoundariesArray,
                #         self.meshList[0].edgeNodesArray,
                #         self.meshList[0].nodeStarArray,
                #         self.meshList[0].nodeStarOffsets,
                #         self.meshList[0].elementMaterialTypes,
                #         self.meshList[0].elementBoundaryMaterialTypes,
                #         self.meshList[0].nodeMaterialTypes,
                #         self.meshList[0].nodeArray,
                #         self.meshList[0].elementDiametersArray,
                #         self.meshList[0].elementInnerDiametersArray,
                #         self.meshList[0].elementBoundaryDiametersArray,
                #         self.meshList[0].elementBarycentersArray,
                #         self.meshList[0].elementBoundaryBarycentersArray,
                #         self.meshList[0].nodeDiametersArray,
                #         self.meshList[0].nodeSupportArray,
                #         self.meshList[0].h,
                #         self.meshList[0].hMin,
                #         self.meshList[0].volume)

    def refine(self):
        self.meshList.append(QuadrilateralMesh())
        self.meshList[-1].globalMesh = self.meshList[-1]
        childrenDict = self.meshList[-1].refine(self.meshList[-2])

        # The following four lines should be called elsewhere...Most of this is don in
        # the c-function calls that are not implemented yet for 2D quads
        self.meshList[-1].nElements_owned = self.meshList[-1].nElements_global
        self.meshList[-1].nodeNumbering_subdomain2global = np.zeros((self.meshList[-1].nNodes_global,), 'i')
        self.meshList[-1].elementNumbering_subdomain2global = np.zeros((self.meshList[-1].nElements_global,), 'i')
        self.meshList[-1].nodeOffsets_subdomain_owned[-1] = self.meshList[-1].nNodes_global
        self.meshList[-1].nNodes_owned = self.meshList[-1].nNodes_global
        self.meshList[-1].elementOffsets_subdomain_owned[-1] = self.meshList[-1].nElements_global

        for node in range(self.meshList[-1].nNodes_global):
            self.meshList[-1].nodeNumbering_subdomain2global[node]=node
        for element in range(self.meshList[-1].nElements_global):
            self.meshList[-1].elementNumbering_subdomain2global[element]=element
        self.elementChildren.append(childrenDict)


class InterpolatedBathymetryMesh(MultilevelTriangularMesh):
    """A triangular mesh that interpolates bathymetry from a point cloud"""
    def __init__(self,
                 domain,
                 triangleOptions,
                 gmshOption=False,
                 atol=1.0e-4,
                 rtol=1.0e-4,
                 maxElementDiameter=None,
                 maxLevels=20,
                 maxNodes=100000,
                 bathyType="points",#"grid"
                 bathyAssignmentScheme="interpolation",#"localAveraging","L2-projection","H1-projection"
                 errorNormType="L2", #L1,Linfty
                 refineType=0,
                 ):
        from scipy import interpolate as scipy_interpolate
        if maxElementDiameter:
            self.maxElementDiameter = maxElementDiameter
        else:
            self.maxElementDiameter = np.inf
        self.atol = atol
        self.rtol = rtol
        self.maxLevels=maxLevels
        self.maxNodes=maxNodes
        self.domain = domain
        self.triangleOptions = triangleOptions
        self.bathyType=bathyType
        self.bathyAssignmentScheme=bathyAssignmentScheme
        self.errorNormType = errorNormType
        self.gmshOption = gmshOption
        if gmshOption==True:
            logEvent("InterpolatedBathymetryMesh: Calling gmsh to generate 2D coarse mesh for"+self.domain.name)
            runGmsh(domain.polyfile)
            logEvent("InterpolatedBathymetryMesh: Converting to Proteus Mesh")
            self.coarseMesh= TriangularMesh()            
        else:
            logEvent("InterpolatedBathymetryMesh: Calling Triangle to generate 2D coarse mesh for "+self.domain.name)
            runTriangle(domain.polyfile,
                    self.triangleOptions)       

            logEvent("InterpolatedBathymetryMesh: Converting to Proteus Mesh")
            self.coarseMesh = TriangularMesh()

        self.coarseMesh.generateFromTriangleFiles(filebase=domain.polyfile,base=1)
        
        MultilevelTriangularMesh.__init__(self,0,0,0,skipInit=True,nLayersOfOverlap=0,
                                          parallelPartitioningType=MeshParallelPartitioningTypes.node)
        self.generateFromExistingCoarseMesh(self.coarseMesh,1,
                                            parallelPartitioningType=MeshParallelPartitioningTypes.node)
        self.computeGeometricInfo()
        
        #allocate some arrays based on the bathymetry data
        logEvent("InterpolatedBathymetryMesh:Allocating data structures for bathymetry interpolation algorithm")
        if bathyType == "points":
            self.nPoints_global = self.domain.bathy.shape[0]
            self.pointElementsArray_old = -np.ones((self.nPoints_global,),'i')
            self.pointElementsArray = -np.ones((self.nPoints_global,),'i')
            self.pointNodeWeightsArray = np.zeros((self.nPoints_global,3),'d')
            self.bathyInterpolant = scipy_interpolate.LinearNDInterpolator(self.domain.bathy[:,:2],self.domain.bathy[:,2])
            self.bathyNearestNeighbor = scipy_interpolate.NearestNDInterpolator(self.domain.bathy[:,:2], self.domain.bathy[:,2])
        elif bathyType == "grid":
            self.nPoints_global = self.domain.bathy.shape[0]
            self.pointElementsArray_old = -np.ones((self.nPoints_global,),'i')
            self.pointElementsArray = -np.ones((self.nPoints_global,),'i')
            self.pointNodeWeightsArray = np.zeros((self.nPoints_global,3),'d')
            x = self.domain.bathy[:self.domain.bathyGridDim[1],0]
            y = self.domain.bathy[:self.domain.bathyGridDim[0]*self.domain.bathyGridDim[1]:self.domain.bathyGridDim[1],1]
            z = self.domain.bathy[:,2].reshape(self.domain.bathyGridDim).transpose()
            self.bathyInterpolant = scipy_interpolate.RectBivariateSpline(x,y,z,kx=1,ky=1)
            #self.bathyInterpolant = scipy_interpolate.interp2d(x,y,z)
        
        logEvent("InterpolatedBathymetryMesh: Locating points on initial mesh")
        self.locatePoints_initial(self.meshList[-1])
        
        logEvent("InterpolatedBathymetryMesh:setting mesh bathymetry from data")
        self.setMeshBathymetry(self.meshList[-1])
        logEvent("InterpolatedBathymetryMesh: tagging elements for refinement")
        self.tagElements(self.meshList[-1])
        
        if gmshOption==True:
            levels=0
            error= 0.5 #self.tagElements(self.meshList[-1])
        else:
            levels = 0
            error = 1.0  #self.tagElements
        while error >= 1.0 and self.meshList[-1].nNodes_global < self.maxNodes and levels < self.maxLevels:
            levels += 1
            logEvent("InterpolatedBathymetryMesh: Locally refining, level = %i" % (levels,))
            self.locallyRefine(self.meshList[-1].elementTags,flagForRefineType=refineType)
            logEvent("InterpolatedBathymetryMesh: interpolating bathymetry from parent mesh to refined mesh")
            self.interpolateBathymetry()
            logEvent("InterpolatedBathymetryMesh: Locating points on child mesh")
            self.locatePoints_refined(self.meshList[-1])
            logEvent("InterpolatedBathymetryMesh: setting mesh bathmetry from data")
            self.setMeshBathymetry(self.meshList[-1])
            logEvent("InterpolatedBathymetryMesh: tagging elements for refinement")
            error = self.tagElements(self.meshList[-1])
            logEvent("InterpolatedBathymetryMesh: error = %f atol = %f rtol = %f number of elements tagged = %i" % (error,self.atol,self.rtol,self.meshList[-1].elementTags.sum()))
        
    def setMeshBathymetry(self,mesh):
        if self.bathyAssignmentScheme == "interpolation":
            self.setMeshBathymetry_interpolate(mesh)
        elif self.bathyAssignmentScheme == "localAveraging":
            self.setMeshBathymetry_localAveraging(mesh)
        elif self.bathyAssignmentScheme == "L2-projection":
            raise NotImplementedError
        elif self.bathyAssignmentScheme == "H1-projection":
            raise NotImplementedError

    def setMeshBathymetry_interpolate(self,mesh):
        if self.bathyType == 'grid':
            mesh.nodeArray[:,2] = self.bathyInterpolant.ev(mesh.nodeArray[:,0],mesh.nodeArray[:,1])
        else:
            mesh.nodeArray[:,2] = self.bathyInterpolant(mesh.nodeArray[:,0],mesh.nodeArray[:,1])
            nI = np.isnan(mesh.nodeArray[:,2])
            mesh.nodeArray[nI,2] = self.bathyNearestNeighbor(mesh.nodeArray[nI,0],mesh.nodeArray[nI,1])

    def setMeshBathymetry_localAveraging(self,mesh):
        """
        calculate the arithmetic mean bathymetry of points inside each triangle and then assign the area-weighted average of the element means to each node
        """
        from .FemTools import AffineMaps,ReferenceSimplex,LinearOnSimplexWithNodalBasis
        interpolationSpace = LinearOnSimplexWithNodalBasis(nd=2)
        #maps = AffineMaps(mesh,interpolationSpace.referenceElement,interpolationSpace)
        #maps.useC = True
        #calculate mean element height for each element
        #uses arithmetic mean, so it assumes the "patch" associated with each point the same size (weight)
        mesh.elementMeanZ = np.zeros((mesh.nElements_global,),'d')
        for pN in range(self.nPoints_global):
            eN = self.pointElementsArray[pN]
            if eN >= 0:
                if mesh.nPoints_element[eN] > 0:
                    mesh.elementMeanZ[eN] += self.domain.bathy[pN,2]/float(mesh.nPoints_element[eN])
                    mesh.nodeArray[mesh.elementNodesArray[eN,0],2] = 0.0
                    mesh.nodeArray[mesh.elementNodesArray[eN,1],2] = 0.0
                    mesh.nodeArray[mesh.elementNodesArray[eN,2],2] = 0.0
        #now assign the mesh node bathmetry as an area weighted average of the element mean
        sumArray = mesh.nodeArray[:,2].copy()
        sumArray[:]=0.0
        for eN in range(mesh.nElements_global):
            if mesh.nPoints_element[eN] > 0:#only calculate a contribution if this element contains a point
                #calculate triangle area and assign weighted average of element means to node
                xiArray = np.zeros((2,),'d')
                #
                grad_psi = np.zeros((interpolationSpace.dim,
                                        interpolationSpace.referenceElement.dim),
                                       'd')
                dx = np.zeros((interpolationSpace.referenceElement.dim),
                                 'd')
                jacobian = np.zeros((interpolationSpace.referenceElement.dim,
                                        interpolationSpace.referenceElement.dim),
                                       'd')
                inverseJacobian = np.zeros((interpolationSpace.referenceElement.dim,
                                               interpolationSpace.referenceElement.dim),
                                              'd')
                for j in interpolationSpace.range_dim:
                    grad_psi[j,:] = interpolationSpace.basisGradients[j](xiArray)#evaluate at zero because we can (psi is linear)
                jacobian.flat[:]=0.0
                inverseJacobian.flat[:]=0.0
                for j in interpolationSpace.range_dim:
                    J = mesh.elementNodesArray[eN,j]
                    for m in interpolationSpace.referenceElement.range_dim:
                        for n in interpolationSpace.referenceElement.range_dim:
                            jacobian[m,n] += mesh.nodeArray[J,m]*grad_psi[j,n]
                J = mesh.elementNodesArray[eN,0]
                inverseJacobian = inv(jacobian)
                area = 0.5*det(jacobian)
                sumArray[mesh.elementNodesArray[eN,0]] += area/mesh.nodeSupportArray[mesh.elementNodesArray[eN,0]]
                sumArray[mesh.elementNodesArray[eN,1]] += area/mesh.nodeSupportArray[mesh.elementNodesArray[eN,1]]
                sumArray[mesh.elementNodesArray[eN,2]] += area/mesh.nodeSupportArray[mesh.elementNodesArray[eN,2]]
                mesh.nodeArray[mesh.elementNodesArray[eN,0],2] += area*mesh.elementMeanZ[eN]/mesh.nodeSupportArray[mesh.elementNodesArray[eN,0]]
                mesh.nodeArray[mesh.elementNodesArray[eN,1],2] += area*mesh.elementMeanZ[eN]/mesh.nodeSupportArray[mesh.elementNodesArray[eN,1]]
                mesh.nodeArray[mesh.elementNodesArray[eN,2],2] += area*mesh.elementMeanZ[eN]/mesh.nodeSupportArray[mesh.elementNodesArray[eN,2]]
        #cek debug
        #print "sum of a nodes element areas divided by node support shoudl be 1 ",sumArray
    def locatePoints(self,mesh):
        """
        locate the element containing each point

        this should only be used on very coarse meshes
        """
        from .FemTools import AffineMaps,ReferenceSimplex,LinearOnSimplexWithNodalBasis
        interpolationSpace = LinearOnSimplexWithNodalBasis(nd=2)
        #maps = AffineMaps(mesh,interpolationSpace.referenceElement,interpolationSpace)
        #maps.useC = False
        #find the elements that contain bathymetry points and calculate:
        # - for each element, the number of bathmetry points in that element
        # - for each node, the total area of the nodes elements that containing bathmetry points
        # - the area of each element
        # - the total area covered by elements containing bathmetry points
        mesh.nPoints_element = np.zeros((mesh.nElements_global,),'i')
        mesh.nodeSupportArray = np.zeros((mesh.nNodes_global,),'d')
        mesh.area_element =  np.zeros((mesh.nElements_global,),'d')
        self.pointElementsArray[:] = -1
        self.totalArea = 0.0
        for eN in range(mesh.nElements_global):
            #map points to reference space and test if it lies in the reference triangle
            xiArray = np.zeros((2,),'d')
            xiArray[:] = 0.0
            #
            grad_psi = np.zeros((interpolationSpace.dim,
                                    interpolationSpace.referenceElement.dim),
                                   'd')
            dx = np.zeros((interpolationSpace.referenceElement.dim),
                             'd')
            jacobian = np.zeros((interpolationSpace.referenceElement.dim,
                                    interpolationSpace.referenceElement.dim),
                                   'd')
            inverseJacobian = np.zeros((interpolationSpace.referenceElement.dim,
                                           interpolationSpace.referenceElement.dim),
                                          'd')
            for j in interpolationSpace.range_dim:
                grad_psi[j,:] = interpolationSpace.basisGradients[j](xiArray[0])#evalute at zero because we can (psi is linear)
            jacobian.flat[:]=0.0
            inverseJacobian.flat[:]=0.0
            for j in interpolationSpace.range_dim:
                J = mesh.elementNodesArray[eN,j]
                for m in interpolationSpace.referenceElement.range_dim:
                    for n in interpolationSpace.referenceElement.range_dim:
                        jacobian[m,n] += mesh.nodeArray[J,m]*grad_psi[j,n]
            J = mesh.elementNodesArray[eN,0]
            inverseJacobian = inv(jacobian)
            area = 0.5*det(jacobian)
            mesh.area_element[eN] = area
            self.totalArea += area
            for pN in range(self.nPoints_global):#can optimize by skipping previously found points
                xiArray[:] = 0.0
                dx[:]=self.domain.bathy[pN,:2]
                for m in interpolationSpace.referenceElement.range_dim:
                    dx[m]-=mesh.nodeArray[J,m]
                for m in interpolationSpace.referenceElement.range_dim:
                    for n in interpolationSpace.referenceElement.range_dim:
                        xiArray[m] += inverseJacobian[m,n]*dx[n]
                #barycentric coordinates are non-negative so we're in this element
                if xiArray[0] >=0.0 and xiArray[1] >= 0.0 and 1.0 - xiArray[0] - xiArray[1] >= 0.0:
                    self.pointElementsArray[pN] = eN
                    self.pointNodeWeightsArray[pN,0] = interpolationSpace.basis[0](xiArray)
                    self.pointNodeWeightsArray[pN,1] = interpolationSpace.basis[1](xiArray)
                    self.pointNodeWeightsArray[pN,2] = interpolationSpace.basis[2](xiArray)
        #count the number of points inside each element
        for pN in range(self.nPoints_global):
            if self.pointElementsArray[pN] >= 0:
                mesh.nPoints_element[self.pointElementsArray[pN]] += 1
        #add up the support area for each node
        for eN in range(mesh.nElements_global):
            if mesh.nPoints_element[eN] > 0:
                mesh.nodeSupportArray[mesh.elementNodesArray[eN,0]] += mesh.area_element[eN]
                mesh.nodeSupportArray[mesh.elementNodesArray[eN,1]] += mesh.area_element[eN]
                mesh.nodeSupportArray[mesh.elementNodesArray[eN,2]] += mesh.area_element[eN]
    def locatePoints_refined(self,mesh):
        """
        locate the element containing each point

        this should only be used on very coarse meshes
        """
        from .FemTools import AffineMaps,ReferenceSimplex,LinearOnSimplexWithNodalBasis
        interpolationSpace = LinearOnSimplexWithNodalBasis(nd=2)
        #maps = AffineMaps(mesh,interpolationSpace.referenceElement,interpolationSpace)
        #maps.useC = False
        #find the elements that contain bathymetry points and calculate:
        # - for each element, the number of bathmetry points in that element
        # - for each node, the total area of the nodes elements that containing bathmetry points
        # - the area of each element
        # - the total area covered by elements containing bathmetry points
        mesh.nPoints_element = np.zeros((mesh.nElements_global,),'i')
        mesh.nodeSupportArray = np.zeros((mesh.nNodes_global,),'d')
        mesh.area_element =  np.zeros((mesh.nElements_global,),'d')
        self.totalArea = 0.0
        self.pointElementsArray_old[:] = self.pointElementsArray
        self.pointElementsArray[:] = -1
        for pN in range(self.nPoints_global):
            eN_parent = self.pointElementsArray_old[pN]
            for eN in self.elementChildrenArrayList[-1][self.elementChildrenOffsetsList[-1][eN_parent]:self.elementChildrenOffsetsList[-1][eN_parent+1]]:
                xiArray = np.zeros((2,),'d')
                xiArray[:] = 0.0
                grad_psi = np.zeros((interpolationSpace.dim,
                                        interpolationSpace.referenceElement.dim),
                                       'd')
                dx = np.zeros((interpolationSpace.referenceElement.dim),
                                 'd')
                jacobian = np.zeros((interpolationSpace.referenceElement.dim,
                                        interpolationSpace.referenceElement.dim),
                                       'd')
                inverseJacobian = np.zeros((interpolationSpace.referenceElement.dim,
                                               interpolationSpace.referenceElement.dim),
                                              'd')
                for j in interpolationSpace.range_dim:
                    grad_psi[j,:] = interpolationSpace.basisGradients[j](xiArray[0])#evalute at zero because we can (psi is linear)
                jacobian.flat[:]=0.0
                inverseJacobian.flat[:]=0.0
                for j in interpolationSpace.range_dim:
                    J = mesh.elementNodesArray[eN,j]
                    for m in interpolationSpace.referenceElement.range_dim:
                        for n in interpolationSpace.referenceElement.range_dim:
                            jacobian[m,n] += mesh.nodeArray[J,m]*grad_psi[j,n]
                J = mesh.elementNodesArray[eN,0]
                inverseJacobian = inv(jacobian)
                area = 0.5*det(jacobian)
                mesh.area_element[eN] = area
                self.totalArea += area
                xiArray[:] = 0.0
                dx[:]=self.domain.bathy[pN,:2]
                for m in interpolationSpace.referenceElement.range_dim:
                    dx[m]-=mesh.nodeArray[J,m]
                for m in interpolationSpace.referenceElement.range_dim:
                    for n in interpolationSpace.referenceElement.range_dim:
                        xiArray[m] += inverseJacobian[m,n]*dx[n]
                #barycentric coordinates are non-negative so we're in this element
                if xiArray[0] >=0.0 and xiArray[1] >= 0.0 and 1.0 - xiArray[0] - xiArray[1] >= 0.0:
                    self.pointElementsArray[pN] = eN
                    self.pointNodeWeightsArray[pN,0] = interpolationSpace.basis[0](xiArray)
                    self.pointNodeWeightsArray[pN,1] = interpolationSpace.basis[1](xiArray)
                    self.pointNodeWeightsArray[pN,2] = interpolationSpace.basis[2](xiArray)
        #count the number of points inside each element
        for pN in range(self.nPoints_global):
            if self.pointElementsArray[pN] >= 0:
                mesh.nPoints_element[self.pointElementsArray[pN]] += 1
        #add up the support area for each node
        for eN in range(mesh.nElements_global):
            if mesh.nPoints_element[eN] > 0:
                mesh.nodeSupportArray[mesh.elementNodesArray[eN,0]] += mesh.area_element[eN]
                mesh.nodeSupportArray[mesh.elementNodesArray[eN,1]] += mesh.area_element[eN]
                mesh.nodeSupportArray[mesh.elementNodesArray[eN,2]] += mesh.area_element[eN]
    def locatePoints_initial(self,mesh):
        """
        locate the element containing each point

        first find the nearest node, then loop over that node's  elements
        """
        from scipy.spatial import cKDTree
        from .FemTools import AffineMaps,ReferenceSimplex,LinearOnSimplexWithNodalBasis
        interpolationSpace = LinearOnSimplexWithNodalBasis(nd=2)
        #find the elements that contain bathymetry points and calculate:
        # - for each element, the number of bathmetry points in that element
        # - for each node, the total area of the nodes elements that contain bathmetry points
        # - the area of each element
        # - the total area covered by elements containing bathmetry points
        mesh.nPoints_element = np.zeros((mesh.nElements_global,),'i')
        mesh.nodeSupportArray = np.zeros((mesh.nNodes_global,),'d')
        mesh.area_element =  np.zeros((mesh.nElements_global,),'d')
        self.totalArea = 0.0
        self.pointElementsArray[:] = -1
        tree = cKDTree(mesh.nodeArray[:,:2])
        xiArray = np.zeros((2,),'d')
        grad_psi = np.zeros((interpolationSpace.dim,
                                interpolationSpace.referenceElement.dim),
                               'd')
        dx = np.zeros((interpolationSpace.referenceElement.dim),
                         'd')
        jacobian = np.zeros((interpolationSpace.referenceElement.dim,
                                interpolationSpace.referenceElement.dim),
                               'd')
        inverseJacobian = np.zeros((interpolationSpace.referenceElement.dim,
                                       interpolationSpace.referenceElement.dim),
                                      'd')
        for pN in range(self.nPoints_global):
            (distance,nN) = tree.query(self.domain.bathy[pN,:2])
            for eN in mesh.nodeElementsArray[mesh.nodeElementOffsets[nN]:mesh.nodeElementOffsets[nN+1]]:
                xiArray[:] = 0.0
                for j in interpolationSpace.range_dim:
                    grad_psi[j,:] = interpolationSpace.basisGradients[j](xiArray[0])#evalute at zero because we can (psi is linear)
                jacobian.flat[:]=0.0
                inverseJacobian.flat[:]=0.0
                for j in interpolationSpace.range_dim:
                    J = mesh.elementNodesArray[eN,j]
                    for m in interpolationSpace.referenceElement.range_dim:
                        for n in interpolationSpace.referenceElement.range_dim:
                            jacobian[m,n] += mesh.nodeArray[J,m]*grad_psi[j,n]
                J = mesh.elementNodesArray[eN,0]
                inverseJacobian = inv(jacobian)
                area = 0.5*det(jacobian)
                mesh.area_element[eN] = area
                xiArray[:] = 0.0
                dx[:]=self.domain.bathy[pN,:2]
                for m in interpolationSpace.referenceElement.range_dim:
                    dx[m]-=mesh.nodeArray[J,m]
                for m in interpolationSpace.referenceElement.range_dim:
                    for n in interpolationSpace.referenceElement.range_dim:
                        xiArray[m] += inverseJacobian[m,n]*dx[n]
                #if the barycentric coordinates are non-negative we're in this element
                if xiArray[0] >=0.0 and xiArray[1] >= 0.0 and 1.0 - xiArray[0] - xiArray[1] >= 0.0:
                    self.pointElementsArray[pN] = eN
                    self.pointNodeWeightsArray[pN,0] = interpolationSpace.basis[0](xiArray)
                    self.pointNodeWeightsArray[pN,1] = interpolationSpace.basis[1](xiArray)
                    self.pointNodeWeightsArray[pN,2] = interpolationSpace.basis[2](xiArray)
        self.totalArea += mesh.area_element.sum()
        #count the number of points inside each element
        for pN in range(self.nPoints_global):
            if self.pointElementsArray[pN] >= 0:
                mesh.nPoints_element[self.pointElementsArray[pN]] += 1
        #add up the support area for each node
        for eN in range(mesh.nElements_global):
            if mesh.nPoints_element[eN] > 0:
                mesh.nodeSupportArray[mesh.elementNodesArray[eN,0]] += mesh.area_element[eN]
                mesh.nodeSupportArray[mesh.elementNodesArray[eN,1]] += mesh.area_element[eN]
                mesh.nodeSupportArray[mesh.elementNodesArray[eN,2]] += mesh.area_element[eN]
    def interpolateBathymetry(self):
        """
        interpolate bathymetry for the refinement from the parent  mesh
        """
        from proteus.FemTools import C0_AffineLinearOnSimplexWithNodalBasis,DOFBoundaryConditions,MultilevelProjectionOperators
        mlMeshTemp = MultilevelMesh(levels=2)
        mlMeshTemp.meshList = self.meshList[-2:]
        mlMeshTemp.nLevels=2
        mlMeshTemp.cmeshList = self.cmeshList[-2:]
        mlMeshTemp.elementParentsArrayList = self.elementParentsArrayList[-2:]
        mlMeshTemp.elementChildrenArrayList = self.elementChildrenArrayList[-1:]
        mlMeshTemp.elementChildrenOffsetsList = self.elementChildrenOffsetsList[-1:]
        nd=2
        TrialSpaceTypeDict = {0:C0_AffineLinearOnSimplexWithNodalBasis}
        trialSpaceDictParent = dict([ (cj,TrialSpaceType(mlMeshTemp.meshList[0],nd)) for (cj,TrialSpaceType) in TrialSpaceTypeDict.items()])
        trialSpaceDictChild = dict([ (cj,TrialSpaceType(mlMeshTemp.meshList[1],nd)) for (cj,TrialSpaceType) in TrialSpaceTypeDict.items()])
        trialSpaceDictList  = [trialSpaceDictParent,trialSpaceDictChild]
        offsetListList=[[0],[0]]
        strideListList=[[1],[1]]
        def getDBC(x,flag):
            return None
        bcDictList=[dict([(0,DOFBoundaryConditions(trialSpaceDictParent[0],getPointwiseBoundaryConditions=getDBC,weakDirichletConditions=False))]),
                    dict([(0,DOFBoundaryConditions(trialSpaceDictChild[0],getPointwiseBoundaryConditions=getDBC,weakDirichletConditions=False))])]
        self.meshTransfers = MultilevelProjectionOperators(
            mlMeshTemp,
            trialSpaceDictList,
            offsetListList,
            strideListList,
            bcDictList)
        zParent = self.meshList[-2].nodeArray[:,2].copy()
        zChild = self.meshList[-1].nodeArray[:,2].copy()
        self.meshTransfers.prolongList[-1].matvec(zParent,zChild)
        self.meshList[-1].nodeArray[:,2] = zChild
    def tagElements(self,mesh):
        """
        loop over points and calculate whether the interpolation error is within the tolerance

        this should only be used on very coarse meshes
        """
        mesh.elementTags = np.zeros((mesh.nElements_global,),'i')
        mesh.errorAverage_element =  np.zeros((mesh.nElements_global,),'d')
        errorInfty = 0.0
        mesh.elementTags[mesh.elementDiametersArray > self.maxElementDiameter ] = 1
        for pN in range(self.nPoints_global):
            eN = self.pointElementsArray[pN]
            if eN >= 0:
                zInterp = self.pointNodeWeightsArray[pN,0]*mesh.nodeArray[mesh.elementNodesArray[eN,0],2] +  \
                          self.pointNodeWeightsArray[pN,1]*mesh.nodeArray[mesh.elementNodesArray[eN,1],2] +  \
                          self.pointNodeWeightsArray[pN,2]*mesh.nodeArray[mesh.elementNodesArray[eN,2],2]
                errorPointwise = fabs(zInterp - self.domain.bathy[pN,2])/(fabs(self.domain.bathy[pN,2])*self.rtol + self.atol)
                errorInfty = max(errorPointwise,errorInfty)
                mesh.errorAverage_element[eN] += (errorPointwise/float(mesh.nPoints_element[eN]))
                #print "error average",mesh.errorAverage_element[eN]
                if errorPointwise >= 1.0:
                    mesh.elementTags[eN] = 1
        if self.errorNormType == "L1":
            mesh.elementTags[:] = 0
            errorL1 = 0.0
            for eN in range(mesh.nElements_global):
                errorL1 += mesh.errorAverage_element[eN]*mesh.area_element[eN]
                if mesh.errorAverage_element[eN] >= 1.0:
                    mesh.elementTags[eN] = 1
            errorL1 /= self.totalArea#normalize by domain error to make error have units of length
            return errorL1
        if self.errorNormType == "L2":
            mesh.elementTags[:] = 0
            errorL2 = 0.0
            for eN in range(mesh.nElements_global):
                errorL2 += (mesh.errorAverage_element[eN])**2 * mesh.area_element[eN]
                if mesh.errorAverage_element[eN] >= 1.0:
                    mesh.elementTags[eN] = 1
            errorL2 = sqrt(errorL2)/self.totalArea#normalize by domain error to make error have units of length
            return errorL2
        else:
            print("Interpolation Error, L_infty ",errorInfty)
            return errorInfty

class EdgeMesh(Mesh):
    """A mesh of edges

    The nodes, and edges are indexed by their node tuples. The
    corresponding lists are derived from the dictionaries, and sorted
    lexicographically. The global node numbers are redefined to give a
    lexicographic ordering.
    """

    def __init__(self):
        Mesh.__init__(self)
        self.nodeDict={}
        self.edgeDict={}
        self.oldToNewNode=[]
    def computeGeometricInfo(self):
        from .import cmeshTools
        cmeshTools.computeGeometricInfo_edge(self.cmesh)
    def generateEdgeMeshFromRectangularGrid(self,nx,Lx):
        from .import cmeshTools
        self.cmesh = cmeshTools.CMesh()
        cmeshTools.generateEdgeMeshFromRectangularGrid(nx,Lx,self.cmesh)
        cmeshTools.allocateGeometricInfo_edge(self.cmesh)
        cmeshTools.computeGeometricInfo_edge(self.cmesh)
        self.buildFromC(self.cmesh)
        #mwf debug
        #print "EdgeMesh rect->edge after build nodes=%s " % (self.nodeArray)
    def rectangularToEdge(self,grid):
        #copy the nodes from the rectangular mesh
        #I want to be able to renumber latter without
        #changing the grid nodes, so I do deep copies here
        self.nodeList = [Node(n.N,n.p[X],n.p[Y],n.p[Z]) for n in grid.nodeList]
        self.nodeDict = dict([(n,n) for n in self.nodeList])
        for e in grid.edgeList:
            self.newEdge([self.nodeDict[e.nodes[0]],self.nodeDict[e.nodes[1]]])
        self.finalize()
        #self.buildListsEdges()

    def finalize(self):
        self.buildLists()
        self.buildArraysFromLists()
        #todo: build boundary mesh

    def buildLists(self):
        self.buildListsNodes()
        self.buildListsEdges()
        self.elementList = self.edgeList
        self.elementBoundaryList = self.nodeList

    def buildListsNodes(self):
        keyList = list(self.nodeDict.keys())
        keyList.sort()
        self.nodeList=[]
        self.oldToNewNode=list(range(len(self.nodeDict)))
        for nN,k in enumerate(keyList):
            self.oldToNewNode[self.nodeDict[k].N]=nN
            self.nodeDict[k].N = nN
            self.nodeList.append(self.nodeDict[k])

    def buildListsEdges(self):
        keyList = list(self.edgeDict.keys())
        keyList.sort()
        self.edgeList=[]
        for eN,k in enumerate(keyList):
            self.edgeDict[k].N = eN
            self.edgeList.append(self.edgeDict[k])

    def newEdge(self,nodes):
        e = Edge(len(self.edgeDict),nodes)
        self.edgeDict[e.nodes] = e
        return e

    def registerNode(self,node):
        if node in self.nodeDict:
            node = self.nodeDict[node]
        else:
            node.N = len(self.nodeDict)
            self.nodeDict[node] = node
        return node

    def refine2e(self,oldMesh):
        childrenDict={}
        for e in oldMesh.edgeList:
            #deep copy old nodes because we'll renumber
            eNodes = [Node(eN,n.p[X],n.p[Y],n.p[Z])
                      for eN,n in enumerate(e.nodes)]
            for lnN,n in enumerate(eNodes): eNodes[lnN]=self.registerNode(n)
            #add new node
            e.computeGeometricInfo()
            newNode = Node(len(self.nodeDict),
                           e.barycenter[X],
                           e.barycenter[Y],
                           e.barycenter[Z])
            newNode = self.registerNode(newNode)
            e1=self.newEdge([eNodes[0],newNode])
            e2=self.newEdge([newNode,eNodes[1]])
            childrenDict[e.N]=[e1,e2]
        self.finalize()
        return childrenDict

    def refine(self,oldMesh):
        return self.refine2e(oldMesh)

    def meshInfo(self):
        minfo = """Number of edges : %d
Number of nodes : %d\n""" % (self.nElements_global,self.nNodes_global)
        if self.subdomainMesh != self:
            sinfo = self.subdomainMesh.meshInfo()
            info = "*** Global ***\n" + minfo + "\n*** Local ***\n" + sinfo
            return info
        return minfo

    def writeMeshADH(self,filename):
        pass
    def writeMeshXdmf(self,ar,name='',t=0.0,init=False,meshChanged=False,tCount=0):
        Mesh.writeMeshXdmf(self,ar,name,t,init,meshChanged,"Polyline",tCount)
    def writeMeshEnsight(self,filename,description=None):
        base=1
        #write the casefile
        caseOut=open(filename+'.case','w')
        caseOut.write('FORMAT\n'+'type: ensight gold\n')
        caseOut.write('GEOMETRY\n'+'model: '+filename+'.geo\n')
        caseOut.close()
        meshOut=open(filename+'.geo','w')
        meshOut.write('Ensight Gold\n')
        meshOut.write('Unstructured Edge Mesh\n')
        meshOut.write('node id given\n')
        meshOut.write('element id given\n')
        meshOut.write('part \n'+'%10i\n' % 1)
        if description:
            meshOut.write(description+'\n')
        else:
            meshOut.write('A Mesh\n')
        meshOut.write('coordinates\n'+'%10i\n' % self.nNodes_global)
        for nN in range(self.nNodes_global):
            meshOut.write('%10i\n' % (nN+base))
        for nN in range(self.nNodes_global):
            meshOut.write('%12.5E\n' % self.nodeArray[nN,0])
        for nN in range(self.nNodes_global):
            meshOut.write('%12.5E\n' % self.nodeArray[nN,1])
        for nN in range(self.nNodes_global):
            meshOut.write('%12.5E\n' % self.nodeArray[nN,2])
        meshOut.write('bar2\n'+'%10i\n' % self.nElements_global)
        for eN in range(self.nElements_global):
            meshOut.write('%10i\n' % (eN+base))
        for eN in range(self.nElements_global):
            meshOut.write('%10i%10i\n' % tuple((nN+base) for nN in self.elementNodesArray[eN,:]))
        meshOut.close()

class MultilevelEdgeMesh(MultilevelMesh):
    """A hierarchical multilevel mesh of intervals (edges)"""
    from .import cmeshTools
    def __init__(self,
                 nx, ny, nz,
                 x=0.0, y=0.0, z=0.0,
                 Lx=1.0, Ly=1.0, Lz=1.0,
                 refinementLevels=1,
                 nLayersOfOverlap=1,
                 parallelPartitioningType=MeshParallelPartitioningTypes.node):
        from . import cmeshTools
        MultilevelMesh.__init__(self)
        self.useC=True
        self.nLayersOfOverlap=nLayersOfOverlap; self.parallelPartitioningType = parallelPartitioningType
        if self.useC:
            self.meshList.append(EdgeMesh())
            self.meshList[0].generateEdgeMeshFromRectangularGrid(nx,Lx)
            #Translate the coarse mesh *before* refining it. The C refinement copies
            #the coarse nodes into every finer level, so shifting level 0 afterwards
            #left the finer levels at the origin -- a different coordinate frame from
            #level 0, so geometric transfers between them were wrong and the finest
            #mesh, the one that is solved on, was not where the domain is.
            self.meshList[0].nodeArray[:,0] += x
            self.meshList[0].nodeArray[:,1] += y
            self.meshList[0].nodeArray[:,2] += z
            self.cmultilevelMesh = cmeshTools.CMultilevelMesh(self.meshList[0].cmesh,refinementLevels)
            self.buildFromC(self.cmultilevelMesh)
            for l in range(1,refinementLevels):
                self.meshList.append(EdgeMesh())
                self.meshList[l].cmesh = self.cmeshList[l]
                self.meshList[l].buildFromC(self.meshList[l].cmesh)
            self.partitionLevels(nLayersOfOverlap,parallelPartitioningType)
        else:
            grid=RectangularGrid(nx,ny,nz,Lx,Ly,Lz)
            self.meshList.append(EdgeMesh())
            self.meshList[0].rectangularToEdge(grid)
            self.meshList[0].nodeArray[:,0] += x
            self.meshList[0].nodeArray[:,1] += y
            self.meshList[0].nodeArray[:,2] += z
            self.elementChildren=[]
            print(self.meshList[0].meshInfo())
            for l in range(1,refinementLevels):
                self.refine()
                print(self.meshList[-1].meshInfo())
    def refine(self):
        self.meshList.append(EdgeMesh())
        childrenDict = self.meshList[-1].refine(self.meshList[-2])
        self.elementChildren.append(childrenDict)
    def computeGeometricInfo(self):
        for m in self.meshList:
            m.computeGeometricInfo()
    def locallyRefine(self,elementTagArray):
        """
        simple local refinement assuming elementTagArray[eN]=1 --> bisect
        """
        if self.useC:
            self.cmeshTools.locallyRefineMultilevelMesh(1,self.cmultilevelMesh,elementTagArray)
            self.buildFromC(self.cmultilevelMesh)
            self.meshList.append(EdgeMesh())
            self.meshList[self.nLevels-1].cmesh = self.cmeshList[self.nLevels-1]
            self.meshList[self.nLevels-1].buildFromC(self.meshList[self.nLevels-1].cmesh)
            self.meshList[self.nLevels-1].partitionMesh(nLayersOfOverlap=self.nLayersOfOverlap,parallelPartitioningType=self.parallelPartitioningType)
        else:
            print("""locallyRefine not implemented for self.useC= %s """ % (self.useC))
        #
    #
class MultilevelSimplicialMesh(MultilevelMesh):
    """A wrapper for all the simplicial hierarchical meshes in 1,2, and 3D"""
    def __init__(self,nd,nx,ny=1,nz=1,Lx=1.0,Ly=1.0,Lz=1.0,refinementLevels=1):
        if nd==1:
            MultilevelEdgeMesh.__init__(self,nx,ny,nz,
                                        Lx,Ly,Lz,
                                        refinementLevels)
        elif nd==2:
            MultilevelTriangularMesh.__init__(self,nx,ny,nz,
                                              Lx,Ly,Lz,
                                              refinementLevels)
        elif nd==3:
            MultilevelTetrahedralMesh.__init__(self,nx,ny,nz,
                                               Lz,Ly,Lz,
                                               refineMentLevels)
    def refine(self):
        if nd==1:
            MultilevelEdgeMesh.refine(self)
        elif nd==2:
            MultilevelTrianglularMesh.refine(self)
        elif nd==3:
            MultilevelTetrahedralMesh.refine(self)

## @}

###utility functions for reading meshes back from an archive

def _dataitem_dataset(node, label):
    """The HDF5 dataset name a topology, geometry or attribute's data item refers to.

    A data item reference is ``<file>:/<dataset>``, so splitting on ``:/``
    keeps a dataset path that itself contains a colon intact.
    """
    item = node.get("data_item") if node is not None else None
    assert item is not None, "no data_item under %s" % (label,)
    text = (item.get("data") or "").strip()
    assert text, "the data_item under %s has no HDF5 reference" % (label,)
    return text.split(':/')[-1]


def findArchiveGrid(domain, MeshTag='Spatial_Domain', id_in_collection=-1, verbose=0):
    """One grid of an archive's domain: a step of the collection named by ``MeshTag``.

    ``domain`` is the ymf domain dict (:func:`proteus.Archiver.readArchiveDomain`
    reads one from an archive's ``.h5``). The collection is the first whose
    name contains ``MeshTag``, or the first of all when none does;
    ``id_in_collection`` selects the step, -1 being the last written. A
    per-subdomain step holds one grid per rank, of which the first is
    returned, which is what a serial reader can use.
    """
    from proteus.Archiver import archive_collections
    collections = archive_collections(domain)
    assert collections, "the archive has no time collections"
    collection = None
    for c in collections:
        if MeshTag in c.get("name", ""):
            collection = c
            break
    if collection is None:
        collection = collections[0]
        logEvent("No grid collection named %r; using %r"
                 % (MeshTag, collection.get("name", "<unnamed>")), 3)
    steps = collection["data"]
    assert steps, ("grid collection %r holds no grids"
                   % (collection.get("name", "<unnamed>"),))
    step = steps[id_in_collection]
    if "spatial_collection" in step:
        subdomains = step["spatial_collection"]
        assert subdomains, "spatial collection holds no subdomain grids"
        return subdomains[0]
    return step


def extractPropertiesFromGrid(grid):
    """Unpack the topology, geometry, nodeMaterialTypes and elementMaterialTypes of a grid dict."""
    Topology = grid.get("topology")
    Geometry = grid.get("geometry")
    NodeMaterials = ElementMaterials = None
    for attribute in grid.get("attributes", []):
        name = attribute.get("name")
        if name == 'nodeMaterialTypes':
            NodeMaterials = attribute
        elif name == 'elementMaterialTypes':
            ElementMaterials = attribute
    for label, node in (("Topology", Topology), ("Geometry", Geometry),
                        ("NodeMaterials", NodeMaterials),
                        ("ElementMaterials", ElementMaterials)):
        logEvent("%s %s" % (label, "found" if node is not None else "absent"), 4)

    return Topology,Geometry,NodeMaterials,ElementMaterials

def readUniformElementTopologyFromXdmf(elementTopologyName,Topology,hdf5,topologyid2name,topology2nodes):
    """
    Read the element topology when there are uniform elements in the mesh
    Type of element given by elementTopologyName
    Heavy data stored in hdf5
    topologyid2name -- lookup for number of nodes in a given element type

    returns

    nElements_global  -- the number of elements in the mesh
    nNodes_element    -- number of nodes per element
    elementNodesArray -- element --> node connectivity stored as flattened array accessed using elementNodes_offset
    elementNodes_offset -- offsets into the elementNodesArray storage for element connectivity,
    element eN nodes are in elementNodesArray[elementNodes_offset[eN]:elementNodes_offset[eN+1]]

    """

    nNodes_element = topology2nodes[elementTopologyName]
    entry = _dataitem_dataset(Topology, "Topology")
    logEvent("Reading  elementNodesArray from %s " % entry,3)

    elementNodesArray = hdf5["/"+entry][:]
    assert elementNodesArray.shape[1] == nNodes_element
    nElements_global = elementNodesArray.shape[0]
    logEvent("nElements_global,nNodes_element= (%d,%d) " % (nElements_global,nNodes_element),3)

    elementNodes_offset = np.arange(nElements_global*nNodes_element+1,step=nNodes_element,dtype='i')

    return nElements_global, nNodes_element, elementNodesArray, elementNodes_offset

def readMixedElementTopologyFromXdmf(elementTopologyName,Topology,hdf5,topologyid2name,topology2nodes):
    """
    Read the element topology when there are mixed elements in the mesh
    Heavy data stored in hdf5
    topologyid2name -- lookup for number of nodes in a given element type

    returns

    nElements_global  -- the number of elements in the mesh
    elementNodesArray -- element --> node connectivity stored as flattened
    array accessed using elementNodes_offset
    elementNodes_offset -- offsets into the elementNodesArray storage for element
    connectivity, element eN nodes are
    inelementNodesArray[elementNodes_offset[eN]:elementNodes_offset[eN+1]]

    """
    assert elementTopologyName == 'Mixed'

    entry = _dataitem_dataset(Topology, "Topology")
    logEvent("Reading xdmf_topology from %s " % entry,3)

    xdmf_topology = hdf5["/"+entry][:]
    #build elementNodesArray and offsets now
    nElements_global = 0
    i = 0
    while i < len(xdmf_topology):
        nElements_global += 1
        nNodes_local = topology2nodes[topologyid2name[xdmf_topology[i]]]
        i += nNodes_local+1
    #
    logEvent("Mixed topology found %s elements " % nElements_global,3)
    elementNodes_offset = np.zeros((nElements_global+1,),'i')

    i = 0; eN = 0
    while i < len(xdmf_topology):
        nNodes_local = topology2nodes[topologyid2name[xdmf_topology[i]]]
        elementNodes_offset[eN+1] = elementNodes_offset[eN] + nNodes_local
        eN += 1; i += nNodes_local+1
    elementNodesArray = np.zeros((elementNodes_offset[nElements_global],),'i')
    i = 0; eN = 0
    #this read self.xdmf_topology, which does not exist in a module function
    while i < len(xdmf_topology):
        nNodes_local = topology2nodes[topologyid2name[xdmf_topology[i]]]
        elementNodesArray[elementNodes_offset[eN]:elementNodes_offset[eN+1]][:] = xdmf_topology[i+1:i+1+nNodes_local][:]
        eN += 1; i += nNodes_local+1

    return nElements_global, elementNodesArray, elementNodes_offset

def readMeshFromArchive(archive_base,MeshTag="Spatial_Domain",verbose=0):
    """Read a mesh back from a proteus archive, from its hdf5 file alone

    ``archive_base`` is the archive's path without the ``.h5`` extension.
    Its metadata (version 1, the XDMF fragments proteus <= 1.9.x stored, or
    version 2) describes every step; the last step of the collection named
    by ``MeshTag`` is read, so no ``.xmf`` is needed. Serial only: a
    per-subdomain archive yields its first subdomain.

    :return: a BasicMeshInfo object with the minimal information read

    """
    dataDir, filename = os.path.split(archive_base)
    dataDir = dataDir or '.'
    h5_path = os.path.join(dataDir, filename+'.h5')
    assert os.path.isfile(h5_path)
    from proteus.Archiver import archive_collections
    domain = readArchiveDomain(filename, dataDir)
    assert domain is not None and archive_collections(domain), \
        ("%s holds no proteus archive metadata; a mesh in an XDMF file written "
         "by another tool is read with readMeshXdmf" % (h5_path,))
    return _readMeshFromDomain(domain, h5_path, MeshTag=MeshTag, verbose=verbose)

def readMeshXdmf(xmf_archive_base,heavy_file_base,MeshTag="Spatial_Domain",hasHDF5=True,verbose=0):
    """Read a mesh from an XDMF file, assuming heavy data is in hdf5

    For XDMF written by other tools -- an OpenFOAM mesh converted to XDMF,
    say -- whose ``.h5`` holds the arrays and nothing else. The XDMF is
    parsed by ``ymf.xdmf``, so proteus itself imports no XML parser. A
    proteus archive needs no ``.xmf``: :func:`readMeshFromArchive` reads it
    from its ``.h5``.

    :return: a BasicMeshInfo object with the minimal information read

    """
    from ymf.xdmf import read_xdmf
    assert hasHDF5, "a mesh is read with its heavy data in an hdf5 file"
    assert os.path.isfile(xmf_archive_base+'.xmf')
    assert os.path.isfile(heavy_file_base+'.h5')
    domain, _ = read_xdmf(xmf_archive_base+'.xmf')
    return _readMeshFromDomain(domain, heavy_file_base+'.h5', MeshTag=MeshTag, verbose=verbose)

def _readMeshFromDomain(domain, h5_path, MeshTag="Spatial_Domain", verbose=0):
    """The mesh described by one grid of ``domain``, its arrays read from ``h5_path``."""

    ###information about allowed Xdmf topologies
    #Xdmf cell type id to Name
    topologyid2name = {2:'Polyline',4:'Triangle',5:'Quadrilateral',6:'Tetrahedron',8:'Wedge',9:'Hexahedron',
                       112:'Mixed'} #Mixed isn't actually used 0x070
    #Topology name to number of local nodes
    topology2nodes = {'Polyline':2,'Triangle':3,'Quadrilateral':4,'Tetrahedron':4,'Wedge':6,'Hexahedron':8}

    #for output
    class BasicMeshInfo(object):
        def __init__(self):
            self.nNodes_global     = None
            self.nodeArray         = None
            self.nodeMaterialTypes = None
            self.nNodes_element    = None
            self.nElements_global  = None
            self.elementTopologyName = None
            self.elementNodesArray = None
            self.elementNodes_offset  = None
            self.elementMaterialTypes = None
            self.nNodes_owned         = None
            self.nElements_owned      = None
        #
    #
    MeshInfo = BasicMeshInfo()

    hdf5= h5py.File(h5_path,"r")

    Grid = findArchiveGrid(domain,MeshTag,id_in_collection=-1,verbose=verbose)

    Topology,Geometry,NodeMaterials,ElementMaterials = extractPropertiesFromGrid(Grid)

    assert Geometry is not None
    entry = _dataitem_dataset(Geometry, "Geometry")
    logEvent("Reading nodeArray from %s " % entry,3)

    MeshInfo.nodeArray = hdf5["/"+entry][:]
    MeshInfo.nNodes_global = MeshInfo.nodeArray.shape[0]

    if NodeMaterials is not None:
        entry = _dataitem_dataset(NodeMaterials, "nodeMaterialTypes")
        logEvent("Reading nodeMaterialTypes from %s " % entry,4)
        MeshInfo.nodeMaterialTypes = hdf5["/"+entry][:]
    else:
        MeshInfo.nodeMaterialTypes = np.zeros((MeshInfo.nNodes_global,),'i')

    assert Topology is not None
    MeshInfo.elementTopologyName = Topology.get("type")
    assert MeshInfo.elementTopologyName is not None

    logEvent("elementTopologyName= %s " % MeshInfo.elementTopologyName,3)
    assert MeshInfo.elementTopologyName in list(topologyid2name.values())

    if MeshInfo.elementTopologyName != 'Mixed':
        MeshInfo.nElements_global, MeshInfo.nNodes_element, \
            MeshInfo.elementNodesArray, MeshInfo.elementNodes_offset = readUniformElementTopologyFromXdmf(MeshInfo.elementTopologyName,Topology,
                                                                                                          hdf5,topologyid2name,topology2nodes)

    else:
        MeshInfo.nElements_global, MeshInfo.elementNodesArray, \
            MeshInfo.elementNodes_offset = readMixedElementTopologyFromXdmf(MeshInfo.elementTopologyName,Topology,hdf5,topologyid2name,topology2nodes)

    #
    if ElementMaterials is not None:
        entry = _dataitem_dataset(ElementMaterials, "elementMaterialTypes")
        logEvent("Reading elementMaterialTypes from %s " % entry,3)
        MeshInfo.elementMaterialTypes = hdf5["/"+entry][:]

    else:
        MeshInfo.elementMaterialTypes = np.zeros((MeshInfo.nElements_global,),'i')
    #
    ###only serial for now
    MeshInfo.nNodes_owned = MeshInfo.nNodes_global
    MeshInfo.nElements_owned = MeshInfo.nElements_global
    hdf5.close()

    return MeshInfo
#

#: the space of a ymf mesh file that holds the mesh, as in a proteus archive
YMF_MESH_SPACE = "Spatial_Domain"

def writeMeshYmf(prefix, nodeArray, elementNodesArray, nodeMaterialTypes=None,
                 elementMaterialTypes=None, topologyName='Triangle'):
    """Write a mesh as a ymf file ``prefix.ymf`` and its arrays ``prefix.h5``

    The mesh is one uniform space, :data:`YMF_MESH_SPACE`, at time 0 of a
    format 2 archive, so what reads a mesh from a proteus archive reads it:
    :func:`readMeshYmf` is :func:`readMeshFromArchive` for a file that holds
    a mesh and nothing else. ``elementNodesArray`` is (nElements, nNodes)
    and zero-based.
    """
    from ymf import archive as ymf_archive
    nodeArray = np.ascontiguousarray(nodeArray, dtype='d')
    elementNodesArray = np.ascontiguousarray(elementNodesArray, dtype='i')
    if nodeMaterialTypes is None:
        nodeMaterialTypes = np.zeros((nodeArray.shape[0],), 'i')
    if elementMaterialTypes is None:
        elementMaterialTypes = np.zeros((elementNodesArray.shape[0],), 'i')
    arrays = [("nodes", nodeArray),
              ("elements", elementNodesArray),
              ("nodeMaterialTypes", np.ascontiguousarray(nodeMaterialTypes, dtype='i')),
              ("elementMaterialTypes", np.ascontiguousarray(elementMaterialTypes, dtype='i'))]
    h5_name = os.path.basename(prefix) + '.h5'
    with h5py.File(prefix + '.h5', 'w') as f:
        for name, array in arrays:
            f.create_dataset(name, data=array)
    items = {name: ymf_archive.data_item_for(array, "%s:/%s" % (h5_name, name))
             for name, array in arrays}
    grid = ymf_archive.uniform_grid(
        ymf_archive.topology(topologyName, elementNodesArray.shape[0], items["elements"]),
        ymf_archive.geometry(items["nodes"]),
        [ymf_archive.attribute("nodeMaterialTypes", items["nodeMaterialTypes"], center="Node"),
         ymf_archive.attribute("elementMaterialTypes", items["elementMaterialTypes"], center="Cell")])
    domain = ymf_archive.new_domain()
    ymf_archive.add_instant(domain, 0.0, {YMF_MESH_SPACE: grid})
    ymf_archive.write_ymf(domain, prefix + '.ymf')

def readMeshYmf(prefix, MeshTag=YMF_MESH_SPACE, verbose=0):
    """Read the mesh in ``prefix.ymf`` and ``prefix.h5``, as written by :func:`writeMeshYmf`

    :return: a BasicMeshInfo object, as :func:`readMeshFromArchive` returns
    """
    from ymf import archive as ymf_archive
    domain, _ = ymf_archive.read_ymf(prefix + '.ymf')
    return _readMeshFromDomain(domain, prefix + '.h5', MeshTag=MeshTag, verbose=verbose)

def convert2DMToYmf(filebase, prefix=None, base=1):
    """Convert the XMS 2DM triangle mesh ``filebase.3dm`` to a ymf mesh, ``prefix.ymf`` and ``prefix.h5``

    ``base`` is the index base of the node numbers, and of the element
    materials, as in the file readers; ``prefix`` is ``filebase`` by default.
    Node materials are zero unless ``filebase.bc`` gives them, as ``NDS``
    lines. Its ``FCS`` lines tag element boundaries, which a ymf mesh has no
    place for yet, so they are refused rather than dropped.

    :return: the prefix written
    """
    prefix = filebase if prefix is None else prefix
    elements, elementMaterials, nodes = [], [], []
    with open(filebase + '.3dm') as f:
        for line in f:
            words = line.split()
            if not words:
                continue
            if words[0] == 'E3T':
                if nodes:
                    raise ValueError("%s.3dm: an E3T element follows the nodes" % (filebase,))
                elements.append([int(w) - base for w in words[2:5]])
                elementMaterials.append(int(words[5]) - base)
            elif words[0] == 'ND':
                nodes.append([float(w) for w in words[2:5]])
            elif words[0] in ('E2L', 'E4Q', 'E6T', 'E8Q', 'E9Q'):
                raise NotImplementedError(
                    "%s.3dm: %s elements are not triangles" % (filebase, words[0]))
    nodeMaterials = np.zeros((len(nodes),), 'i')
    if os.path.exists(filebase + '.bc'):
        with open(filebase + '.bc') as f:
            for line in f:
                words = line.split()
                if words and words[0] == 'NDS':
                    nodeMaterials[int(words[1]) - base] = int(words[2])
                elif words and words[0] == 'FCS':
                    raise NotImplementedError(
                        "%s.bc tags element boundaries (FCS), which a ymf mesh cannot hold yet"
                        % (filebase,))
    writeMeshYmf(prefix,
                 np.array(nodes, dtype='d').reshape(-1, 3),
                 np.array(elements, dtype='i').reshape(-1, 3),
                 nodeMaterialTypes=nodeMaterials,
                 elementMaterialTypes=np.array(elementMaterials, dtype='i'),
                 topologyName='Triangle')
    return prefix

def writeHexMesh(mesh_info,hexfile_base,index_base=0):
    """
    Write a hex mesh in Ido's format with base numbering index_base
    HEX
    nNodes_global nElements_global
    x0 y0 z0
    x1 y1 z1
    ...
    xN yN zN
    [n0 n1 n2 n3 n4 n5 n6 n7 mat0]
    [n0 n1 n2 n3 n4 n5 n6 n7 mat1]
    """
    assert mesh_info.elementTopologyName=='Hexahedron'

    header="""HEX
{nNodes_global} {nElements_global}
""".format(nNodes_global=mesh_info.nNodes_global,nElements_global=mesh_info.nElements_global)

    with open(hexfile_base+'.mesh','w') as mout:
        mout.write(header)
        np.savetxt(mout,mesh_info.nodeArray)
        #format the elements, appending element material type
        elems_with_mat = np.append(mesh_info.elementNodesArray,mesh_info.elementMaterialTypes.reshape(mesh_info.nElements_global,1),axis=1)
        elems_with_mat[:,:-1] += index_base
        np.savetxt(mout,elems_with_mat,fmt='%d')


class MultilevelNURBSMesh(MultilevelMesh):
    def __init__(self,
                 nx, ny, nz,
                 x=0.0, y=0.0, z=0.0,
                 px=1, py=1, pz=1,
                 Lx=1.0, Ly=1.0, Lz=1.0,
                 refinementLevels=1,
                 skipInit=False,
                 nLayersOfOverlap=1,
                 parallelPartitioningType=MeshParallelPartitioningTypes.node):
        from .import cmeshTools
        from . import Comm
        MultilevelMesh.__init__(self)
        self.useC = True
        self.nLayersOfOverlap = nLayersOfOverlap; self.parallelPartitioningType = parallelPartitioningType
        logEvent("Generating NURBS mesh")
        if not skipInit:
            self.meshList.append(NURBSMesh())
            self.meshList[0].generateNURBSMeshFromRectangularGrid(nx,ny,nz,px,py,pz,Lx,Ly,Lz)
            self.cmultilevelMesh = cmeshTools.CMultilevelMesh(self.meshList[0].cmesh,refinementLevels)
            self.buildFromC(self.cmultilevelMesh)
            self.meshList[0].nodeArray[:,0] += x
            self.meshList[0].nodeArray[:,1] += y
            self.meshList[0].nodeArray[:,2] += z
            for l in range(1,refinementLevels):
                self.meshList.append(NURBSMesh())
                self.meshList[l].cmesh = self.cmeshList[l]
                self.meshList[l].buildFromC(self.cmeshList[l])
            self.partitionLevels(nLayersOfOverlap,parallelPartitioningType)

    def generateFromExistingCoarseMesh(self,mesh0,refinementLevels,nLayersOfOverlap=1,
                                       parallelPartitioningType=MeshParallelPartitioningTypes.node):
        from .import cmeshTools
        #blow away or just trust garbage collection
        self.nLayersOfOverlap=nLayersOfOverlap;self.parallelPartitioningType=parallelPartitioningType
        self.meshList = []
        self.elementParents = None
        self.cmultilevelMesh = None

        self.meshList.append(mesh0)
        self.cmultilevelMesh = cmeshTools.CMultilevelMesh(self.meshList[0].cmesh,refinementLevels)
        self.buildFromC(self.cmultilevelMesh)
        for l in range(1,refinementLevels):
            self.meshList.append(NURBSMesh())
            self.meshList[l].cmesh = self.cmeshList[l]
            self.meshList[l].buildFromC(self.meshList[l].cmesh)
        self.partitionLevels(nLayersOfOverlap,parallelPartitioningType)


    def refine(self):
        self.meshList.append(NURBSMesh())
        childrenDict = self.meshList[-1].refine(self.meshList[-2])
        self.elementChildren.append(childrenDict)
    def computeGeometricInfo(self):
        for m in self.meshList:
            m.computeGeometricInfo()





class NURBSMesh(HexahedralMesh):
    """A mesh consisting of NURBS.

    """
    def __init__(self):
        HexahedralMesh.__init__(self)

    def generateHexahedralMeshFromRectangularGrid(self,nx,ny,nz,Lx,Ly,Lz):
        generateNURBSMeshFromRectangularGrid(self,nx,ny,nz,1,1,1,Lx,Ly,Lz)

    def generateNURBSMeshFromRectangularGrid(self,nx,ny,nz,px,py,pz,Lx,Ly,Lz):
        from .import cmeshTools
        self.cmesh = cmeshTools.CMesh()
        cmeshTools.generateNURBSMeshFromRectangularGrid(nx,ny,nz,px,py,pz,Lx,Ly,Lz,self.cmesh)
        cmeshTools.allocateGeometricInfo_NURBS(self.cmesh)
        cmeshTools.computeGeometricInfo_NURBS(self.cmesh)
        self.buildFromC(self.cmesh)


def distance(a, b):
    norm = np.linalg.norm
    return norm(b - a)

def triangleVerticesToNormals(elementVertices):
    """
    Given a set of vertices to a triangle, return normals and a point corresponding to each normal
    """
    norm = np.linalg.norm

    elementVertices = np.asarray(elementVertices)

    if norm(elementVertices[:,2]) > 0:
        raise ValueError("Expected triangles in 2D plane, got something else")
    sets = ((0, 1), (0, 2), (1, 2))
    outs = (2, 1, 0)
    faces = []
    rotate = np.asarray(((0., -1., 0.),
                         (1., 0., 0.),
                         (0., 0., 0.)))

    for seti, out in zip(sets, outs):
        vertices = elementVertices[np.array(seti,'i')]
        ab = vertices[1] - vertices[0]
        v_out = vertices[0] - elementVertices[out]
        normal = rotate.dot(ab)
        # normal should point *away* from remaining point
        if normal.dot(v_out) < 0:
            normal = -1*normal
        faces.append((normal, vertices[0]))
    return faces

def tetrahedronVerticesToNormals(elementVertices):
    """
    Given a set of vertices to a tetrahedron, return normals and a point corresponding to each normal
    """

    elementVertices = np.asarray(elementVertices)

    sets = ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3))
    outs = (3, 2, 1, 0)

    faces = []

    for seti, out in zip(sets, outs):
        vertices = elementVertices[np.array(seti,'i')]
        ab = vertices[1] - vertices[0]
        ac = vertices[2] - vertices[0]
        normal = np.cross(ab, ac)
        v_out = vertices[0] - elementVertices[out]
        # normal should point *away* from remaining point
        if normal.dot(v_out) < 0:
            normal = -1*normal
        faces.append((normal, vertices[0]))
    return faces


def cubeVerticesToNormals(elementVertices):
    """
    Given the vertices of a quadrilateral (in the plane z = 0) or a hexahedron, return a normal
    and a point for each face, as triangleVerticesToNormals does, from the convex hull of the vertices
    whatever the order they are listed in.

    The element must be convex with straight edges and, for a hexahedron, planar faces: a
    quadrilateral always is, and so is a hexahedron that is a box or a parallelepiped. A
    hexahedron with a warped (non-planar) face is not the polyhedron the hull encloses, and
    ValueError is raised for it.
    """
    from scipy.spatial import ConvexHull
    elementVertices = np.asarray(elementVertices, dtype=np.double)
    nVertices = len(elementVertices)
    if nVertices == 4:
        nd, nFaces = 2, 4
        if np.linalg.norm(elementVertices[:, 2]) > 0:
            raise ValueError("Expected quadrilaterals in 2D plane, got something else")
    elif nVertices == 8:
        nd, nFaces = 3, 6
    else:
        raise ValueError("Expected 4 or 8 vertices, got %d" % nVertices)
    hull = ConvexHull(elementVertices[:, :nd])
    # a facet is a triangle in 3D, so a planar quadrilateral face is two with the same plane
    planes = {}
    for equation in hull.equations:
        key = tuple(np.round(equation, 8))
        planes.setdefault(key, equation)
    if len(planes) != nFaces:
        raise ValueError("A %s with %d distinct face planes (expected %d) is not planar-faced: "
                         "line gauges cannot clip a warped element"
                         % ("quadrilateral" if nd == 2 else "hexahedron", len(planes), nFaces))
    faces = []
    for equation in planes.values():
        normal = np.zeros(3)
        normal[:nd] = equation[:nd]
        point = np.zeros(3)
        # n . x + offset = 0 for x on the face, n a unit vector pointing out
        point[:nd] = -equation[nd] * equation[:nd]
        faces.append((normal, point))
    return faces


def intersectPoints(line, points):
    """
    Given a line segment (defined as two points), identify all points that the line segment intersects.

    This hasn't been vectorized.
    """

    a, b = line
    a = np.asarray(a)
    b = np.asarray(b)
    distanceAB = distance(a, b)

    def onAB(p):
        p = np.asarray(p)
        eps = 2*np.max((np.max(np.spacing(a)), np.max(np.spacing(b)), np.max(np.spacing(p))))
        distancePA = distance(a, p)
        distancePB = distance(p, b)
        return p if abs(distancePA + distancePB - distanceAB) < eps else None

    return [onAB(p) for p in points]


def intersectEdges(line, edges):
    """
    Given a line segment (defined as two points), identify the locations of its intersections with all
    given edges (defined as line segments).  If the line and an edge overlap, the *furthest* point
    along the line (closest to the second point) that is still on each edge is returned.

    This hasn't been vectorized.
    """
    norm = np.linalg.norm

    def cross2d(u, v):
        # np.cross() dropped support for 2D vectors in numpy>=2.0 (it used to return
        # the scalar z-component); intersectEdge is used for both 2D and 3D points,
        # so only substitute the version-independent 2D formula for genuinely 2D
        # inputs and fall back to the real (still numpy>=2.0-supported) cross product
        # for 3D ones.
        if len(u) == 2 and len(v) == 2:
            return u[0]*v[1] - u[1]*v[0]
        return np.cross(u, v)

    def intersectEdge(line, edge):

        line = np.asarray(line)
        edge = np.asarray(edge)
        a, b = line
        c, d = edge
        v_l = b - a
        v_e = d - c

        vl_cross_ve = cross2d(v_l, v_e)
        mag_vl_cross_ve = norm(vl_cross_ve)

        if mag_vl_cross_ve == 0:
            # lines are parallel, check for overlap
            intersects = intersectPoints(line, edge) + intersectPoints(edge, line)
            # test for an intersect in intersectPoints
            intersect = next((i for i in intersects if i is not None), None)
            if intersect is not None:
                # farthest endpoint is a, so start from there
                closest_endpoint = a
                closest_distance = distance(closest_endpoint, b)
                # could reuse iterator from above, but it's confusing enough as it is :)
                for intersect in intersects:
                    if intersect is None:
                        continue
                    intersect_distance = distance(intersect, b)
                    if intersect_distance < closest_distance:
                        closest_endpoint = intersect
                        closest_distance = intersect_distance
                return closest_endpoint
            else:
                return None

        # lines are not parallel, check for intersection
        vl_cross_ve = cross2d(v_l, v_e)

        # if v_l and v_e intersect, then there is an x that satisfies
        x_vl_cross_ve = cross2d((c - a), v_e)

        # but the two above vectors must be parallel
        # NOTE: vl_cross_ve/x_vl_cross_ve are scalars here (2D cross product), so
        # np.cross() on them was never valid regardless of numpy version -- this
        # branch isn't exercised by any current test, so left as pre-existing,
        # unverified logic rather than guessing at a redesign.
        if norm(np.cross(vl_cross_ve, x_vl_cross_ve)) > 1e-8:
            return None

        # two lines are parallel, solve for x
        x = norm(x_vl_cross_ve)/norm(vl_cross_ve)

        intersect = a + x*(b-a)

        # and verify intersection is on the line
        points = intersectPoints(line, [intersect])
        assert(len(points) == 1)
        return points[0]

    return [intersectEdge(line, edge) for edge in edges]


def intersectPolyhedron(line, polyhedron):
    """
    Given a line (defined as two points), identify the locations that it enters and exits the
    polyhedron (defined as a collection of half-planes in three-space in normal, vertex form)

    If the facets of the polyhedron are in edge form, the normal can be computed by taking the cross product of any
    two non-parallel edges of the facet (in three-space).  Any vertex of the facet will work.

    Implementation of algorithm described here: http://geomalgorithms.com/a13-_intersect-4.html

    This hasn't been vectorized.
    """

    a, b = line
    a, b = np.asarray(a), np.asarray(b)

    if distance(a, b) == 0:
        raise ValueError("Line segment must not have length 0")

    v_l = b - a
    t_e = 0  # location along line entering polyhedron (initial value 0)
    t_l = 1  # location along line leaving polyhedron (initial value 1)

    for plane in polyhedron:
        n, v = plane
        n, v = np.asarray(n), np.asarray(v)
        ndotba = -n.dot(a - v)
        d = n.dot(v_l)
        if d == 0:
            # the line segment is parallel to this face
            if ndotba < 0:
                # the line is outside the face
                return None
            else:
                # the line is in or on the face, ignore this face
                continue
        t = ndotba/float(d)
        if d < 0:
            # segment is entering polyhedron across this facet
            t_e = max(t_e, t)
            if t_e > t_l:
                # segment enters polyhedron after leaving, no intersection
                return None
        else:
            # segment is exiting polyhedron across this facet
            t_l = min(t_l, t)
            if t_l < t_e:
                # segment exits polyhedron before entering, no intersection
                return None

    assert(t_e <= t_l)

    return [a + t_e*v_l, a + t_l*v_l]


def segmentMayCrossElement(mesh, endpoints):
    """
    Return a boolean mask of the elements whose bounding box the segment touches.

    A slab test of the segment against every element's bounding box, vectorized,
    so that only the few elements near a line are clipped one at a time in Python.
    The boxes are padded by a roundoff tolerance, so an element the segment
    touches is never dropped.
    """
    a, b = np.asarray(endpoints[0], np.double), np.asarray(endpoints[1], np.double)
    nodes = mesh.nodeArray[mesh.elementNodesArray]  # (nElements, nVertices, 3)
    low, high = nodes.min(axis=1), nodes.max(axis=1)
    pad = 1.0e-9*max(float(np.max(high - low)), float(np.max(np.abs(b - a))), 1.0e-300)
    low, high = low - pad, high + pad
    direction = b - a
    t_in = np.zeros(len(low))
    t_out = np.ones(len(low))
    hit = np.ones(len(low), dtype=bool)
    for axis in range(3):
        if direction[axis] == 0.0:
            hit &= (a[axis] >= low[:, axis]) & (a[axis] <= high[:, axis])
        else:
            t0 = (low[:, axis] - a[axis])/direction[axis]
            t1 = (high[:, axis] - a[axis])/direction[axis]
            t_in = np.maximum(t_in, np.minimum(t0, t1))
            t_out = np.minimum(t_out, np.maximum(t0, t1))
    return hit & (t_in <= t_out)


def getMeshIntersections(mesh, toPolyhedron, endpoints):
    """
    Return all intersections between a line segment and a Proteus mesh


    :param mesh - a Proteus mesh
    :param toPolyhedron - a method for converting Proteus element vertices to polyhedra in normal/point form
    :param endpoints - a pair of points in 3-space defining the line segment

    :return a list of pairs of intersections through the mesh
    """

    intersections = set()
    for element in mesh.elementNodesArray[segmentMayCrossElement(mesh, endpoints)]:
        # map nodes to physical vertices
        elementVertices = mesh.nodeArray[element]
        # get plane normals
        polyhedron = toPolyhedron(elementVertices)
        elementIntersections = intersectPolyhedron(endpoints, polyhedron)
        if elementIntersections:
            if np.array_equal(elementIntersections[0], elementIntersections[1]):
                continue
            intersections.update(((tuple(elementIntersections[0]), tuple(elementIntersections[1])),),)
    return intersections

def runGmsh(polyfile,
               baseFlags="Yp",  #change baseflags for gmsh file
               name = ""):
    """
    Generate msh files from a polyfile.
    Arguments
    ---------
    polyfile : str
        Filename with appropriate data for gmsh.
    baseFlags : str
        Standard Tetgen options for generation
    name : str
    """
    #rom subprocess import check_call
    #gmshcmd = "gmsh -%s -e %s.poly" % (baseFlags, polyfile)  #change this to accept gmsh command. 
    # 
    #check_call(gmshcmd,shell=True)
    
    logEvent("Done running gmsh")
    elefile = "%s.ele" % polyfile
    nodefile = "%s.node" % polyfile
    edgefile = "%s.edge" % polyfile
    assert os.path.exists(elefile), "no 1.ele"
    tmp = "%s.ele" % polyfile
    os.rename(elefile,tmp)
    assert os.path.exists(tmp), "no .ele"
    assert os.path.exists(nodefile), "no 1.node"
    tmp = "%s.node" % polyfile
    os.rename(nodefile,tmp)
    assert os.path.exists(tmp), "no .node"
    if os.path.exists(edgefile):
        tmp = "%s.edge" % polyfile
        os.rename(edgefile,tmp)
        assert os.path.exists(tmp), "no 1.edge"
        
def runTriangle(polyfile,
               baseFlags="Yp",
               name = ""):
    """
    Generate tetgen files from a polyfile.

    Arguments
    ---------
    polyfile : str
        Filename with appropriate data for tengen.
    baseFlags : str
        Standard Tetgen options for generation
    name : str
    """
    tricmd = "triangle -%s -e %s.poly" % (baseFlags, polyfile)

    run(tricmd,shell=True)

    logEvent("Done running triangle")
    elefile = "%s.1.ele" % polyfile
    nodefile = "%s.1.node" % polyfile
    edgefile = "%s.1.edge" % polyfile
    assert os.path.exists(elefile), "no 1.ele"
    tmp = "%s.ele" % polyfile
    os.rename(elefile,tmp)
    assert os.path.exists(tmp), "no .ele"
    assert os.path.exists(nodefile), "no 1.node"
    tmp = "%s.node" % polyfile
    os.rename(nodefile,tmp)
    assert os.path.exists(tmp), "no .node"
    if os.path.exists(edgefile):
        tmp = "%s.edge" % polyfile
        os.rename(edgefile,tmp)
        assert os.path.exists(tmp), "no .edge"

def runTetgen(polyfile,
              baseFlags="Yp",
              name = ""):
    """
    Generate tetgen files from a polyfile.

    Arguments
    ---------
    polyfile : str
        Filename with appropriate data for tengen.
    baseFlags : str
        Standard Tetgen options for generation
    name : str
        

    """
    from subprocess import run, check_output
    tetcmd = "tetgen - %s %s.poly" % (baseFlags, polyfile)

    logEvent(run(tetcmd,shell=True,capture_output=True,text=True).stdout)
    
    logEvent("Done running tetgen")
    elefile = "%s.1.ele" % polyfile
    nodefile = "%s.1.node" % polyfile
    facefile = "%s.1.face" % polyfile
    edgefile = "%s.1.edge" % polyfile
    assert os.path.exists(elefile), "no 1.ele"
    tmp = "%s.ele" % polyfile
    os.rename(elefile,tmp)
    assert os.path.exists(tmp), "no .ele"
    assert os.path.exists(nodefile), "no 1.node"
    tmp = "%s.node" % polyfile
    os.rename(nodefile,tmp)
    assert os.path.exists(tmp), "no .node"
    if os.path.exists(facefile):
        tmp = "%s.face" % polyfile
        os.rename(facefile,tmp)
        assert os.path.exists(tmp), "no .face"
    if os.path.exists(edgefile):
        tmp = "%s.edge" % polyfile
        os.rename(edgefile,tmp)
        assert os.path.exists(tmp), "no .edge"

def genMeshWithTriangle(polyfile,
                      nbase=1):
   """
   Generate a mesh from a set of triangle files.

   Arguments
   ---------
   polyfile : str
       Filename base for triangle files
   nbase : int

   Returns
   --------
   mesh : :class:`proteus.MeshTools.TriangularMesh`
       Simplex mesh
   """
   elefile = "%s.ele" % polyfile
   nodefile = "%s.node" % polyfile
   edgefile = "%s.edge" % polyfile
   assert os.path.exists(elefile), "no .ele file"
   assert os.path.exists(nodefile), "no  .node file"
   assert os.path.exists(edgefile), "no .edge"
   mesh = TriangularMesh()
   mesh.generateFromTriangleFiles(polyfile,
                                  base=nbase)
   
   return mesh

def genMeshWithTetgen(polyfile,
                      nbase=1):
   """
   Generate a mesh from a set of tetgen files.

   Arguments
   ---------
   polyfile : str
       Filename base for tetgen files
   nbase : int

   Returns
   --------
   mesh : :class:`proteus.MeshTools.TetrahedralMesh`
       Simplex mesh
   """
   elefile = "%s.ele" % polyfile
   nodefile = "%s.node" % polyfile
   facefile = "%s.face" % polyfile
   edgefile = "%s.edge" % polyfile
   assert os.path.exists(elefile), "no .ele file"
   assert os.path.exists(nodefile), "no  .node file"
   assert os.path.exists(facefile), "no .face file"
   mesh=TetrahedralMesh()
   mesh.generateFromTetgenFiles(polyfile,
                                base=nbase)
   return mesh

class MeshOptions(object):
    """
    Mesh options for the domain

    Parameters
    ----------
    nd: 2 for 2D, 3 for 3D
    """
    def __init__(self, nd=None):
        self.nd = nd
        self.he = 1.
        self.use_gmsh = False
        self.use_plex = False
        self.genMesh = True
        # max area/volume per region, parallel to domain.regions
        self.regionConstraints = []
        self.outputFiles_name = 'mesh'
        self.outputFiles = {'poly': True,     
                            'ply': False,        
                            'asymptote': False,
                            'geo': False}
        self.restrictFineSolutionToAllMeshes = False
        self.parallelPartitioningType = MeshParallelPartitioningTypes.node
        self.nLayersOfOverlapForParallel = 1
        self.triangleOptions = None # defined when setTriangleOptions called
        self.nLevels = 1
        self.structured = False
        self.nn = None
        self.nnx = None
        self.nny = None
        self.nnz = None
        self.triangleFlag = 1
        #'generated': a rectangular domain's regular grid is generated in
        #pieces, every rank its own, with no global mesh or graph partitioner
        #(MeshTools.cartesianSubdomain); 'global': the global mesh, partitioned.
        #Cases the generated path does not cover take the global path either way
        #(MultilevelMesh.generatesInPieces).
        self.regularGrid = 'generated'
        self.nd = nd
        if nd is not None:
            if nd == 2:
                self.triangle_string = 'VApq30Dena'
            if nd == 3:
                self.triangle_string = 'VApq1.35q12feena'
        else:
            self.triangle_string = None

    def setElementSize(self, he):
        """
        Sets element size for uniform mesh.

        Parameters
        ----------
        he: float
            mesh characteristic element size
        """
        self.he = he

    def setParallelPartitioningType(self, partitioning_type='node', layers_overlap=0):
        """
        Changes parallel partitioning type

        Parameters
        ----------
        partitioning_type: Optional[str, int]
            parallel partitioning type (default: 'node' (1))
        layers: int
            layers of overlap for paralllel (default: 0)
        """
        if partitioning_type == 'element' or partitioning_type == 0:
            self.parallelPartitioningType = MeshParallelPartitioningTypes.element
        if partitioning_type == 'node' or partitioning_type == 1:
            self.parallelPartitioningType = MeshParallelPartitioningTypes.node
        self.nLayersOfOverlapForParallel = layers_overlap

    def setTriangleOptions(self, triangleOptions=None):
        """
        Sets the trangle options

        Parameters
        ----------
        triangle_options: Optional[str]
            string for triangle options. If not passed, it will be
            set with triangle_string attribute and 'he' value, with
            default for 2D: he**2/2; default for 3D: he**3/6
        """
        if triangleOptions is not None:
            self.triangleOptions = triangleOptions
        else:
            if self.triangleOptions is None:
                assert self.he is not None, 'Element size (he) must be set before setting triangle options'
                assert self.triangle_string is not None, 'triangle_string must be set before setting triangle options'
                if self.nd == 2:
                    self.triangleOptions = self.triangle_string + '%8.8f' \
                                        % (self.he**2/2.,)
                elif self.nd == 3:
                    self.triangleOptions = self.triangle_string + '%21.16e' \
                                        % (self.he**3/6.,)

    def setMeshGenerator(self, generator):
        """
        Indicates mesh generator to use

        Parameters
        ----------
        generator: str
            options: 'gmsh', 'triangle', 'tetgen'

        (!) Only has an effect when setting to 'gmsh' in current 
        implementation (triangle is default for 2D, tetgen for 3D)
        """
        generators = ['gmsh', 'triangle', 'tetgen']
        assert generator in generators, 'Unknown mesh generator'
        if generator == 'gmsh':
            self.use_gmsh = True
        else:
            self.use_gmsh = False

    def setOutputFiles(self, name='mesh', poly=True, ply=False, asymptote=False, geo=False):
        """
        Output files to be created 

        Parameters
        ----------
        name: Optional[str]
            name of the mesh files (prefix) (default: 'mesh')
        poly: Optional[bool]
            create a poly file
        ply: Optional[bool]
            create a ply file
        asymptote: Optional[bool]
            create an asymptote file
        geo:
            create a geofile
        """
        self.outputFiles_name = name
        self.outputFiles['poly'] = poly
        self.outputFiles['ply'] = ply
        self.outputFiles['asymptote'] = asymptote
        self.outputFiles['geo'] = geo


def rankInterfaceExteriorMask(mesh):
    """Which of ``mesh.exteriorElementBoundariesArray`` are rank interfaces.

    On a partitioned mesh the faces between ranks are exterior faces of each
    rank's subdomain even though they are interior to the global mesh. A
    boundary term applied there is applied to the inside of the domain.

    The partitioner already marks them: it copies the domain's flags onto the
    real boundary and leaves ``elementBoundaryMaterialTypes`` at 0 on the
    interfaces. Measured on a 16x16 unit square over 2 ranks, element
    partition, no overlap: the real boundary came back as material types
    1/2/3/4, 64 faces in all, which is the whole boundary, and every one of
    the 16 interface faces on each rank as 0.

    That mark is why Dirichlet conditions are already safe -- ``getDBC(x,
    flag)`` returns None for flag 0 -- and why the diffusive flux is safe,
    since it is driven by ``isDOFBoundary``. The advective flux consults
    neither, so it needs this.

    The result is a boolean array **indexed the way the ``ebqe`` arrays are**,
    by position in ``exteriorElementBoundariesArray``, not by element boundary
    number. Mask with it after the flux kernel has run; do not hand a shortened
    face list to the kernel instead. The kernels index ``n``, ``f``, ``df``,
    ``flux`` and ``inflowFlag`` by the loop counter, so a shorter face list
    silently pairs each face with another face's quadrature data and leaves
    the skipped entries holding the previous evaluation's values.

    ``None`` means there is nothing to mask. Serial runs always get ``None``,
    deliberately: there are no interfaces to remove, and material type 0 is
    not reserved -- a
    :class:`proteus.Domain.PlanarStraightLineGraphDomain` built without
    segment flags numbers them ``[0, 1, 2, ...]``, so flag 0 is a real side of
    the domain there, and masking it in serial would silently drop its
    outflow. The same ambiguity remains for a *parallel* run on a mesh whose
    real boundary is unflagged; there is no way to tell the two apart from the
    material type alone, and the fix for that is to flag the boundary.
    """
    comm = Comm.get()
    if comm is None or comm.size() == 1:
        return None
    exterior = mesh.exteriorElementBoundariesArray
    mask = mesh.__dict__.get('_rankInterfaceExteriorMask')
    if mask is not None and len(mask) != len(exterior):
        mask = None             # adapted since: the cached mask is the old mesh's
    if mask is None:
        flags = np.asarray(mesh.elementBoundaryMaterialTypes)[exterior]
        if mesh.__dict__.get('rankInterfacesMarked'):
            mask = flags == RANK_INTERFACE_FLAG
        else:
            mask = flags == 0   # a partitioner that does not mark: 0 stands for both
        mesh.__dict__['_rankInterfaceExteriorMask'] = mask
        masked = int(mask.sum())
        if masked:
            logEvent("Advective boundary flux: masking %d of %d exterior faces "
                     "on rank %d; they are rank interfaces (material type 0), "
                     "interior to the global mesh"
                     % (masked, len(mask), comm.rank()), level=2)
    return mask


#: Whether :class:`proteus.Domain.D_base`'s forwarding properties --
#: ``domain.MeshOptions``, ``use_gmsh``, ``use_plex``, ``regionConstraints`` --
#: warn when they are used. **Off by default, deliberately.** They are
#: deprecated in 2.0 and go away in 2.1, but proteus still reaches for them in
#: 347 places of its own (62 in ``proteus/``, 285 in ``test/``), so warning
#: unconditionally today would mean warning at ourselves a few hundred times a
#: test run and would make ``-W error`` unusable for everyone else.
#:
#: Turn it on to find the call sites that still need moving::
#:
#:     MeshTools.warnOnDomainMeshForwarding = True
#:
#: and turn it on in CI once ``proteus/`` is clean, which is what makes the
#: 2.1 removal safe to do rather than hopeful.
warnOnDomainMeshForwarding = False


def _warnDomainMeshForwarding(name):
    """Warn that a Domain meshing property is forwarded, if asked to."""
    if warnOnDomainMeshForwarding:
        import warnings
        use = ("MeshTools.getMeshOptions(domain)" if name == "MeshOptions"
               else "MeshTools.getMeshOptions(domain).%s" % name)
        warnings.warn(
            "Domain.%s is deprecated since 2.0 and will be removed in 2.1; "
            "meshing policy belongs to MeshTools, so use %s instead"
            % (name, use),
            DeprecationWarning, stacklevel=3)


def getMeshOptions(domain):
    """
    The MeshOptions governing how a domain gets discretized.

    Meshing policy is not part of what a domain *is*, so a domain does not
    carry it as description; it is attached here, on first use, and owned by
    this module.  Everything that decides how a region becomes a mesh --
    element size, generator choice, partitioning, refinement levels, region
    size constraints -- belongs on the object this returns.

    Parameters
    ----------
    domain: proteus.Domain.D_base
        the domain to be meshed

    Returns
    -------
    MeshOptions
        the options for this domain, created with the domain's dimension if
        none have been attached yet
    """
    meshOptions = domain.__dict__.get('_meshOptions')
    if meshOptions is None:
        meshOptions = MeshOptions(domain.nd)
        domain.__dict__['_meshOptions'] = meshOptions
    return meshOptions


def setMeshOptions(domain, meshOptions):
    """
    Attach mesh options to a domain, replacing any already attached.

    Parameters
    ----------
    domain: proteus.Domain.D_base
        the domain to be meshed
    meshOptions: MeshOptions
        the options to use for it
    """
    domain.__dict__['_meshOptions'] = meshOptions
    return meshOptions


def msh2simplex(fileprefix, nd):
    """
    Converts a .msh file (Gmsh) to .ele .edge .node files (triangle).
    (!) Works only with triangle elements in 2D and tetrahedral elements in 3D.

    Parameters
    ----------
    fileprefix: str
        prefix of the .msh file (e.g. 'mesh' if file called 'mesh.msh')

    """
    assert nd == 2 or nd == 3, 'nd must be 2 or 3'
    mshfile = open(fileprefix+'.msh', 'r')
    nodes = []
    edges_msh = []
    triangles = []
    tetrahedra = []
    tetrahedron_nb = 0
    triangle_nb = 0
    edge_nb = 0
    switch = None
    switch_count = -1
    logEvent('msh2simplex: getting nodes and elements')
    for i, line in enumerate(mshfile):
        if 'Nodes' in line:
            switch = 'nodes'
            switch_count = -1
        if 'Elements' in line:
            switch = 'elements'
            switch_count = -1
        if switch == 'nodes' and switch_count >= 0:
            words = line.split()
            if switch_count == 0:
                node_nb = int(words[0])
            else:
                nid = int(words[0])
                if nd == 2:
                    x, y, z = float(words[1]), float(words[2]), 0
                elif nd == 3:
                    x, y, z = float(words[1]), float(words[2]), float(words[3])
                nodes += [[nid, x, y, z, 0]]
        if switch == 'elements' and switch_count >= 0:
            words = line.split()
            if switch_count == 0:
                el_nb = int(words[0])
            else:
                el_id = int(words[0])
                el_type = int(words[1])
                nb_tags = int(words[2])
                if nb_tags == 2:
                    flag = int(words[3])
                else:
                    flag = 0
                s = 3+nb_tags # starting index on words for element info
                if el_type == 1: # segment
                    edge_nb += 1
                    edges_msh += [[edge_nb, int(words[s]), int(words[s+1]), flag]]
                elif el_type == 2: # triangle
                    triangle_nb += 1
                    triangles += [[triangle_nb, int(words[s]), int(words[s+1]), int(words[s+2]), flag]]
                    # update nodes flags
                    if nd == 3:
                        for i in range(3):
                            if nodes[int(words[s+i])-1][4] == 0:
                                nodes[int(words[s+i])-1][4] = flag
                elif el_type == 4: # tetrahedron 
                    tetrahedron_nb += 1
                    tetrahedra += [[tetrahedron_nb, int(words[s]), int(words[s+1]), int(words[s+2]), int(words[s+3]), flag]]
                elif el_type == 15: # node
                    nodes[el_id-1][4] = flag
        switch_count += 1
    mshfile.close()

    # construct ALL edges with flags and add flags to nodes
    edges_dict = {}
    triangles = np.array(triangles)
    edge_nb = 0
    edges = []

    logEvent('msh2simplex: constructing edges')
    for triangle in triangles[:,1:4]:  # take only vertices index
        for i in range(len(triangle)):
            edge = Edge(edgeNumber=edge_nb, nodes=[triangle[i-1], triangle[i]])
            edge_exist = bool(edges_dict.get(edge.nodes))
            if not edge_exist:
                edge_nb += 1
                edges_dict[edge.nodes] = edge
                edges += [[edge_nb, edge.nodes[0], edge.nodes[1], 0]]
    logEvent('msh2simplex: updating edges and nodes flags')
    edges = np.array(edges)
    for edge in edges_msh:
        edge_nodes = [edge[1], edge[2]]
        edge_nodes.sort()
        edge_nodes = tuple(edge_nodes)
        edge_class = edges_dict.get(edge_nodes)
        edges[edge_class.N, 3] = edge[3]
        # ! edge nodes are indexed from 1 with gmsh
        if nodes[edge[1]-1][-1] == 0:  # update node flags
            nodes[edge[1]-1][-1] = edge[3]
        if nodes[edge[2]-1][-1] == 0:  # update node flags
            nodes[edge[1]-1][-1] = edge[3]

    if nd == 2:
        logEvent('msh2simplex: writing .node .ele .edge files')
    elif nd == 3:
        logEvent('msh2simplex: writing .node .ele .edge .face files')
    header = '{0:d} {1:d} 0 1'.format(node_nb, nd)

    if nd == 2:
        nodes = np.array(nodes)
        nodes = np.delete(nodes, 3, 1)
        fmt = ['%d', '%f', '%f', '%d']
    elif nd == 3:
       fmt = ['%d', '%f', '%f', '%f', '%d']
    np.savetxt(fileprefix+'.node', nodes, fmt=fmt, header=header, comments='')

    header = '{0:d} 1'.format(edge_nb)
    np.savetxt(fileprefix+'.edge', edges, fmt='%d', header=header, comments='')

    if nd == 2:
        header = '{0:d} 3 1'.format(triangle_nb)
        np.savetxt(fileprefix+'.ele', triangles, fmt='%d', header=header, comments='')
    elif nd == 3:
        header = '{0:d} 3 1'.format(triangle_nb)
        np.savetxt(fileprefix+'.face', triangles, fmt='%d', header=header, comments='')
        header = '{0:d} 4 1'.format(tetrahedron_nb)
        np.savetxt(fileprefix+'.ele', tetrahedra, fmt='%d', header=header, comments='')

    logEvent('msh2simplex: finished converting .msh to simplex files')

def generateMesh(physics,numerics,generatePartitionedMeshFromFiles=False):
    try:
        meshOptions = getMeshOptions(physics.domain)
        assert(not (meshOptions.genMesh == True and
                    meshOptions.nn == None and 
                    meshOptions.nnx  == None and
                    meshOptions.nny == None and
                    meshOptions.nnz == None and
                    meshOptions.triangleOptions == None and
                    meshOptions.use_plex == False))
        mlMesh = _generateMesh(physics.domain, meshOptions, generatePartitionedMeshFromFiles, numerics)
    except:
        meshOptions = numerics
        meshOptions.genMesh = physics.genMesh
        mlMesh = _generateMesh(physics.domain, meshOptions, generatePartitionedMeshFromFiles, numerics)
    return mlMesh

def chooseMeshOptions(physics, numerics):
    """The options :func:`generateMesh` meshes with.

    The domain's :func:`getMeshOptions`, unless they say nothing about
    element size or count; then the n module, with the p module's genMesh
    (old-style input). This is generateMesh's choice, without generating.
    """
    try:
        meshOptions = getMeshOptions(physics.domain)
        assert(not (meshOptions.genMesh == True and
                    meshOptions.nn == None and
                    meshOptions.nnx  == None and
                    meshOptions.nny == None and
                    meshOptions.nnz == None and
                    meshOptions.triangleOptions == None and
                    meshOptions.use_plex == False))
        return meshOptions
    except:
        numerics.genMesh = physics.genMesh
        return numerics

def fullOverlapAssembly(partitioningType, nLayersOfOverlap):
    """Whether the residual is assembled as if every rank held its whole neighbourhood.

    The solvers decide this from n (NumericalSolution's ``parallelUsesFullOverlap``):
    any overlap, or a node partition. Without it, contributions to shared dofs
    are added across ranks.
    """
    return nLayersOfOverlap > 0 or partitioningType == MeshParallelPartitioningTypes.node


def checkPartitionMatchesMesh(p, n):
    """Raise, on more than one rank, if n and the domain's mesh options disagree
    about how the residual is assembled.

    A p module with a domain meshes from ``domain.MeshOptions``; n's
    ``parallelPartitioningType`` and ``nLayersOfOverlapForParallel`` then only
    choose the assembly. If one says no overlap and the other says overlap,
    shared-node contributions are counted twice, or a mesh with no global copy
    is asked for one (test/periodic/duct segfaulted there: n asked for an
    element partition with no overlap for the parallel periodic numbering,
    the mesh was a node partition generated in pieces, and the trial space
    built from the global mesh got None for its arrays). Set both, e.g.
    ``p.domain.MeshOptions.parallelPartitioningType = n.parallelPartitioningType``
    and the same for ``nLayersOfOverlapForParallel``.
    """
    from . import Comm
    comm = Comm.get()
    if comm is None or comm.size() == 1:
        return
    options = getMeshOptions(p.domain)
    n_type = getattr(n, 'parallelPartitioningType', options.parallelPartitioningType)
    n_layers = getattr(n, 'nLayersOfOverlapForParallel', options.nLayersOfOverlapForParallel)
    if (fullOverlapAssembly(n_type, n_layers) !=
        fullOverlapAssembly(options.parallelPartitioningType, options.nLayersOfOverlapForParallel)):
        raise ValueError(
            "the numerics for %s ask for parallelPartitioningType=%s, nLayersOfOverlapForParallel=%s "
            "but its domain's MeshOptions say %s, %s, and the two disagree about whether the "
            "residual is assembled with overlap; set both. For example "
            "p.domain.MeshOptions.parallelPartitioningType = n.parallelPartitioningType and "
            "p.domain.MeshOptions.nLayersOfOverlapForParallel = n.nLayersOfOverlapForParallel."
            % (getattr(p, 'name', p), n_type, n_layers,
               options.parallelPartitioningType, options.nLayersOfOverlapForParallel))


def setDomainFromPhysics(p, n):
    """Give an old-style p module (no ``domain``) a domain built from its
    ``nd``, ``L``, ``x0``, ``polyfile`` or ``meshfile``, with mesh options
    from the n module. A p module that has a domain is left alone, so
    calling this twice is the same as calling it once."""
    if p.domain is not None:
        checkPartitionMatchesMesh(p, n)
        return
    if p.nd == 1:
        p.domain = Domain.RectangularDomain(L=p.L[:1],
                                            x=p.x0[:1],
                                            name=p.name)
    elif p.nd == 2:
        if p.polyfile is not None:
            p.domain = Domain.PlanarStraightLineGraphDomain(fileprefix=p.polyfile,name=p.polyfile)
        elif p.meshfile != None:
            p.domain = Domain.Mesh2DMDomain(p.meshfile)
        else:
            p.domain = Domain.RectangularDomain(L=p.L[:2],
                                                x=p.x0[:2],
                                                name=p.name)
    elif p.nd == 3:
        if p.polyfile is not None:
            p.domain = Domain.PiecewiseLinearComplexDomain(fileprefix=p.polyfile,name=p.polyfile)
        elif p.meshfile is not None:
            p.domain = Domain.Mesh3DMDomain(p.meshfile)
        else:
            p.domain = Domain.RectangularDomain(L=p.L[:3],
                                                x=p.x0[:3],
                                                name=p.name)
    else:
        raise RuntimeError("No support for domains in more than three dimensions")

    #Ensuring backwards compatibility with isolated mesh generation routines
    meshOptions = getMeshOptions(p.domain)
    meshOptions.nnx, meshOptions.nny, meshOptions.nnz = n.nnx,n.nny,n.nnz
    meshOptions.nn = n.nn
    meshOptions.nLevels = n.nLevels
    meshOptions.hex = n.hex
    meshOptions.quad = n.quad
    meshOptions.triangleFlag=n.triangleFlag
    meshOptions.triangleOptions=n.triangleOptions
    meshOptions.genMesh=p.genMesh
    # The solvers read the partition from n (parallelUsesFullOverlap), so
    # the mesh must be partitioned the same way. Without these two lines an
    # element partition without overlap asked for in n was meshed as the
    # default node partition with one layer, and assembled as if it had
    # none: every overlapping element counted twice.
    meshOptions.parallelPartitioningType = getattr(
        n, 'parallelPartitioningType', meshOptions.parallelPartitioningType)
    meshOptions.nLayersOfOverlapForParallel = getattr(
        n, 'nLayersOfOverlapForParallel', meshOptions.nLayersOfOverlapForParallel)
    meshOptions.regularGrid = getattr(n, 'regularGrid', 'generated')

def meshFiles(domain, meshOptions):
    """What :func:`writeMeshFiles` would do for this domain, or None.

    A dict: ``generator`` ('triangle', 'tetgen' or 'gmsh'), ``prefix``,
    ``command`` (what runs, including the generator's options),
    ``inputs`` (files whose contents determine the mesh), ``outputs``
    (files it must write) and ``optional`` (files it may write). None for
    domains whose mesh is built in memory or read from files they name.
    """
    domainMeshOptions = getMeshOptions(domain)
    if isinstance(domain, Domain.PlanarStraightLineGraphDomain):
        if domainMeshOptions.use_gmsh is True:
            prefix = domain.geofile
            return {"generator": "gmsh", "prefix": prefix,
                    "command": "gmsh {0}.geo -2 -format msh2; msh2simplex nd=2".format(prefix),
                    "inputs": [prefix + ".geo"],
                    "outputs": [prefix + e for e in (".msh", ".ele", ".node", ".edge")],
                    "optional": []}
        prefix = domain.polyfile
        return {"generator": "triangle", "prefix": prefix,
                "command": "triangle -{0} -e {1}.poly".format(meshOptions.triangleOptions, prefix),
                "inputs": [prefix + ".poly"],
                "outputs": [prefix + e for e in (".ele", ".node", ".edge")],
                "optional": []}
    if isinstance(domain, Domain.PiecewiseLinearComplexDomain):
        if domainMeshOptions.use_gmsh is True:
            prefix = domain.geofile
            return {"generator": "gmsh", "prefix": prefix,
                    "command": "gmsh {0}.geo -3 -format msh2; msh2simplex nd=3; tetgen -Vfeen {0}.ele".format(prefix),
                    "inputs": [prefix + ".geo"],
                    "outputs": [prefix + e for e in (".msh", ".ele", ".node", ".face")],
                    "optional": [prefix + e for e in (".neigh", ".edge")]}
        prefix = domain.polyfile
        return {"generator": "tetgen", "prefix": prefix,
                "command": "tetgen -{0} {1}.poly".format(meshOptions.triangleOptions, prefix),
                "inputs": [prefix + ".poly"],
                "outputs": [prefix + e for e in (".ele", ".node", ".face")],
                "optional": [prefix + e for e in (".neigh", ".edge")]}
    return None

def writeMeshFiles(domain, meshOptions):
    """Run the mesh generator for a PSLG or PLC domain.

    The serial step of :func:`_generateMesh`, which runs it on rank 0 and
    then reads the files collectively. Writes the files
    :func:`meshFiles` lists.
    """
    name = domain.name
    domainMeshOptions = getMeshOptions(domain)
    if isinstance(domain, Domain.PlanarStraightLineGraphDomain):
        if domainMeshOptions.use_gmsh is True:
            fileprefix = domain.geofile
            if not os.path.exists(fileprefix+".msh"):
                logEvent("Running gmsh to generate 2D mesh for "+name, level=1)
                gmsh_cmd = "time gmsh {0:s} -v 10 -2 -o {1:s} -format msh2".format(fileprefix+".geo", fileprefix+".msh")
                logEvent("Calling gmsh on rank 0 with command %s" % (gmsh_cmd,))
                run(gmsh_cmd, shell=True)
                logEvent("Done running gmsh; converting to triangle")
            else:
                logEvent("Using "+fileprefix+".msh to convert to triangle")
            # convert gmsh to triangle format
            msh2simplex(fileprefix=fileprefix, nd=2)
        else:
            fileprefix = domain.polyfile
            logEvent("Calling Triangle to generate 2D mesh for "+name)
            tricmd = "triangle -{0} -e {1}.poly".format(meshOptions.triangleOptions, fileprefix)
            logEvent("Calling triangle on rank 0 with command %s" % (tricmd,))
            logEvent(run(tricmd, shell=True,capture_output=True, encoding='utf-8').stdout)
            logEvent("Done running triangle")
            run("mv {0:s}.1.ele {0:s}.ele".format(fileprefix), shell=True)
            run("mv {0:s}.1.node {0:s}.node".format(fileprefix), shell=True)
            run("mv {0:s}.1.edge {0:s}.edge".format(fileprefix), shell=True)
    elif isinstance(domain, Domain.PiecewiseLinearComplexDomain):
        if domainMeshOptions.use_gmsh is True:
            fileprefix = domain.geofile
        else:
            fileprefix = domain.polyfile
        if domainMeshOptions.use_gmsh is True:
            if not os.path.exists(fileprefix+".msh"):
                logEvent("Running gmsh to generate 3D mesh for "+name, level=1)
                gmsh_cmd = "time gmsh {0:s} -v 10 -3 -o {1:s} -format msh2".format(fileprefix+'.geo', domain.geofile+'.msh')
                logEvent("Calling gmsh on rank 0 with command %s" % (gmsh_cmd,))
                run(gmsh_cmd, shell=True)
                logEvent("Done running gmsh; converting to tetgen")
            else:
                logEvent("Using "+domain.geofile+".msh to convert to tetgen")
            msh2simplex(fileprefix=fileprefix, nd=3)
            logEvent(run("tetgen -Vfeen {0:s}.ele".format(fileprefix), shell=True,capture_output=True,text=True).stdout)
        else:
            logEvent("Running tetgen to generate 3D mesh for "+name, level=1)
            run("rm -f {0:s}.ele".format(fileprefix), shell=True)
            run("rm -f {0:s}.node".format(fileprefix), shell=True)
            run("rm -f {0:s}.face".format(fileprefix), shell=True)
            run("rm -f {0:s}.neigh".format(fileprefix), shell=True)
            run("rm -f {0:s}.edge".format(fileprefix), shell=True)
            tetcmd = "tetgen -{0} {1}.poly".format(meshOptions.triangleOptions, fileprefix)
            logEvent("Calling tetgen on rank 0 with command %s" % (tetcmd,))
            logEvent(run(tetcmd, shell=True,capture_output=True, encoding='utf-8').stdout)
            logEvent("Done running tetgen")
        run("mv {0:s}.1.ele {0:s}.ele".format(fileprefix), shell=True)
        run("mv {0:s}.1.node {0:s}.node".format(fileprefix), shell=True)
        run("mv {0:s}.1.face {0:s}.face".format(fileprefix), shell=True)
        try:
            run("mv {0:s}.1.neigh {0:s}.neigh".format(fileprefix), shell=True)
        except:
            logEvent("Warning: couldn't move {0:s}.1.neigh".format(fileprefix))
            pass
        try:
            run("mv {0:s}.1.edge {0:s}.edge".format(fileprefix), shell=True)
        except:
            logEvent("Warning: couldn't move {0:s}.1.edge".format(fileprefix))
            pass
    else:
        raise TypeError("writeMeshFiles: no mesh generator for a " + type(domain).__name__)

def _generateMesh(domain,meshOptions,generatePartitionedMeshFromFiles=False,numerics=None):
    # convenience function to generate a mesh using triangle/tetgen/gmsh
    comm = Comm.get()
    name = domain.name
    # meshOptions may be an old-style numerics module rather than a
    # MeshOptions (see the fallback in generateMesh).  The generator choice
    # and the characteristic length have only ever come from the options
    # attached to the domain, so read those two from there.
    domainMeshOptions = getMeshOptions(domain)
    # this is the perfect place to create a factory function which takes in an instance and outputs a corresponding mesh
    # support for old-style domain input

    # now generate meshes, could move to Domain and use polymorphism or MeshTools
    if isinstance(domain, Domain.RectangularDomain):
        if domain.nd == 1:
                mlMesh = MultilevelEdgeMesh(meshOptions.nn, 1, 1,
                                            domain.x[0], 0.0, 0.0,
                                            domain.L[0], 1.0, 1.0,
                                            refinementLevels=meshOptions.nLevels,
                                            nLayersOfOverlap=meshOptions.nLayersOfOverlapForParallel,
                                            parallelPartitioningType=meshOptions.parallelPartitioningType)
        elif domain.nd == 2:
            if (meshOptions.nnx == meshOptions.nny is None):
                nnx = nny = meshOptions.nn
            else:
                nnx = meshOptions.nnx
                nny = meshOptions.nny
            logEvent("Building %i x %i rectangular mesh for %s" % (nnx, nny,name))

            if not hasattr(meshOptions, 'quad'):
                meshOptions.quad = False

            if (meshOptions.quad):
                mlMesh = MultilevelQuadrilateralMesh(nnx, nny,1,
                                                                domain.x[0], domain.x[1], 0.0,
                                                                domain.L[0], domain.L[1],1,
                                                                refinementLevels=meshOptions.nLevels,
                                                                nLayersOfOverlap=meshOptions.nLayersOfOverlapForParallel,
                                                                parallelPartitioningType=meshOptions.parallelPartitioningType,
                                                                regularGrid=getattr(meshOptions, 'regularGrid', None))
            else:
                fileprefix = domain.polyfile
                if fileprefix is None:
                    fileprefix = "regular_{0}x{1}_grid".format(nnx,nny)
                nbase = 1
                if hasattr(meshOptions,'triangleFlag') ==True:
                    triangleFlag = meshOptions.triangleFlag
                else:
                    triangleFlag = 0
                if generatePartitionedMeshFromFiles:
                    if comm.isMaster():
                        globalMesh = TriangularMesh()
                        logEvent(Profiling.memory("Before Generating Mesh", className="NumericalSolution", memSaved=Profiling.memLast))
                        memBeforeMesh = Profiling.memLast
                        logEvent("Generating triangular mesh from regular grid")
                        globalMesh.generateTriangularMeshFromRectangularGrid(nnx, nny,domain.L[0],domain.L[1],triangleFlag=triangleFlag)
                        logEvent("Writing triangle files to {0:s}.ele, etc.".format(fileprefix))
                        globalMesh.writeTriangleFiles(fileprefix, nbase)
                        globalMesh.cmesh.deleteCMesh()
                        del globalMesh
                        import gc
                        gc.collect()
                        logEvent(Profiling.memory("After Generating Mesh", className="NumericalSolution", memSaved=memBeforeMesh))
                        memAfterMesh = Profiling.memLast
                        logEvent(Profiling.memory("After deleting mesh", className="NumericalSolution", memSaved=memAfterMesh))
                    comm.barrier()
                    logEvent(Profiling.memory("Before partitioning", className="NumericalSolution"))
                    memBeforePart = Profiling.memLast
                    logEvent("Generating partitioned mesh from Triangle files")
                    mesh = TriangularMesh()
                    mlMesh = MultilevelTriangularMesh(0,0,0,skipInit=True,
                                                      nLayersOfOverlap=meshOptions.nLayersOfOverlapForParallel,
                                                      parallelPartitioningType=meshOptions.parallelPartitioningType)
                    mlMesh.generatePartitionedMeshFromTriangleFiles(fileprefix, nbase,mesh,meshOptions.nLevels,
                                                                    nLayersOfOverlap=meshOptions.nLayersOfOverlapForParallel,
                                                                    parallelPartitioningType=meshOptions.parallelPartitioningType)
                    mlMesh.meshList[0].subdomainMesh.nodeArray[:, 0] += domain.x[0]
                    mlMesh.meshList[0].subdomainMesh.nodeArray[:, 1] += domain.x[1]
                    logEvent(Profiling.memory("After partitioning", className="NumericalSolution", memSaved=memBeforePart))
                else:
                    mlMesh = MultilevelTriangularMesh(nnx, nny,1,
                                                      domain.x[0], domain.x[1], 0.0,
                                                      domain.L[0], domain.L[1],1,
                                                      refinementLevels=meshOptions.nLevels,
                                                      nLayersOfOverlap=meshOptions.nLayersOfOverlapForParallel,
                                                      parallelPartitioningType=meshOptions.parallelPartitioningType,
                                                      triangleFlag=triangleFlag,
                                                      regularGrid=getattr(meshOptions, 'regularGrid', None))
        elif domain.nd == 3:
            if (meshOptions.nnx == meshOptions.nny == meshOptions.nnz is None):
                nnx = nny = nnz = meshOptions.nn
            else:
                nnx = meshOptions.nnx
                nny = meshOptions.nny
                nnz = meshOptions.nnz
            logEvent("Building %i x %i x %i rectangular mesh for %s" % (nnx, nny,nnz,name))
            
            if not hasattr(meshOptions,'NURBS'):
                meshOptions.NURBS = False
            if not hasattr(meshOptions, 'hex'):
                meshOptions.hex = False
            
            if (meshOptions.NURBS):
                mlMesh = MultilevelNURBSMesh(nnx,nny,nnz,
                                                               meshOptions.px,meshOptions.py,meshOptions.pz,
                                                               domain.x[0], domain.x[1], domain.x[2],
                                                               domain.L[0], domain.L[1], domain.L[2],
                                                               refinementLevels=meshOptions.nLevels,
                                                               nLayersOfOverlap=meshOptions.nLayersOfOverlapForParallel,
                                                               parallelPartitioningType=meshOptions.parallelPartitioningType)
            elif (meshOptions.hex):
                if not hasattr(meshOptions, 'px'):
                    meshOptions.px = 0
                    meshOptions.py = 0
                    meshOptions.pz = 0
                mlMesh = MultilevelHexahedralMesh(nnx, nny, nnz,
                                                            meshOptions.px, meshOptions.py,meshOptions.pz,
                                                            domain.x[0], domain.x[1], domain.x[2],
                                                            domain.L[0], domain.L[1], domain.L[2],
                                                            refinementLevels=meshOptions.nLevels,
                                                            nLayersOfOverlap=meshOptions.nLayersOfOverlapForParallel,
                                                            parallelPartitioningType=meshOptions.parallelPartitioningType,
                                                            regularGrid=getattr(meshOptions, 'regularGrid', None))
            else:
                fileprefix = domain.polyfile
                if fileprefix is None:
                    fileprefix = "regular_{0}x{1}x{2}_grid".format(nnx,nny,nnz)
                nbase = 1
                if meshOptions.genMesh:
                    if generatePartitionedMeshFromFiles:
                        if comm.isMaster():
                            globalMesh = TetrahedralMesh()
                            logEvent(Profiling.memory("Before Generating Mesh", className="NumericalSolution", memSaved=Profiling.memLast))
                            memBeforeMesh = Profiling.memLast
                            logEvent("Generating tetrahedral mesh from regular grid")
                            globalMesh.generateTetrahedralMeshFromRectangularGrid(nnx, nny,nnz,domain.L[0],domain.L[1],domain.L[2])
                            logEvent("Writing tetgen files to {0:s}.ele, etc.".format(fileprefix))
                            globalMesh.writeTetgenFiles(fileprefix, nbase)
                            globalMesh.cmesh.deleteCMesh()
                            del globalMesh
                            import gc
                            gc.collect()
                            logEvent("Writing tetgen edge files to {0:s}.edge".format(fileprefix))
                            run("rm -f {0:s}.1.edge {0:s}.edge".format(fileprefix), shell=True)
                            logEvent(run("tetgen -Vfeen {0:s}.ele".format(fileprefix), shell=True,capture_output=True,text=True).stdout)
                            run("mv -f {0:s}.1.ele {0:s}.ele".format(fileprefix), shell=True)
                            run("mv -f {0:s}.1.node {0:s}.node".format(fileprefix), shell=True)
                            run("mv -f {0:s}.1.face {0:s}.face".format(fileprefix), shell=True)
                            run("mv -f {0:s}.1.neigh {0:s}.neigh".format(fileprefix), shell=True)
                            run("mv -f {0:s}.1.edge {0:s}.edge".format(fileprefix), shell=True)
                            logEvent(Profiling.memory("After Generating Mesh", className="NumericalSolution", memSaved=memBeforeMesh))
                            memAfterMesh = Profiling.memLast
                            logEvent(Profiling.memory("After deleting mesh", className="NumericalSolution", memSaved=memAfterMesh))
                        comm.barrier()
                        logEvent(Profiling.memory("Before partitioning", className="NumericalSolution"))
                        memBeforePart = Profiling.memLast
                        logEvent("Generating partitioned mesh from Tetgen files")
                        mesh = TetrahedralMesh()
                        mlMesh = MultilevelTetrahedralMesh(0, 0,0,skipInit=True,
                                                                        nLayersOfOverlap=meshOptions.nLayersOfOverlapForParallel,
                                                                        parallelPartitioningType=meshOptions.parallelPartitioningType)
                        mlMesh.generatePartitionedMeshFromTetgenFiles(fileprefix, nbase,mesh,meshOptions.nLevels,
                                                                        nLayersOfOverlap=meshOptions.nLayersOfOverlapForParallel,
                                                                        parallelPartitioningType=meshOptions.parallelPartitioningType)

                        mlMesh.meshList[0].subdomainMesh.nodeArray[:, 0] += domain.x[0]
                        mlMesh.meshList[0].subdomainMesh.nodeArray[:, 1] += domain.x[1]
                        mlMesh.meshList[0].subdomainMesh.nodeArray[:, 2] += domain.x[2]
                        logEvent(Profiling.memory("After partitioning", className="NumericalSolution", memSaved=memBeforePart))
                    else:
                        mlMesh = MultilevelTetrahedralMesh(nnx, nny, nnz,
                                                                        domain.x[0], domain.x[1], domain.x[2],
                                                                        domain.L[0], domain.L[1], domain.L[2],
                                                                        refinementLevels=meshOptions.nLevels,
                                                                        nLayersOfOverlap=meshOptions.nLayersOfOverlapForParallel,
                                                                        parallelPartitioningType=meshOptions.parallelPartitioningType,
                                                                        regularGrid=getattr(meshOptions, 'regularGrid', None))
                else:
                    mesh = TetrahedralMesh()
                    mlMesh = MultilevelTetrahedralMesh(0, 0,0,skipInit=True,
                                                                    nLayersOfOverlap=meshOptions.nLayersOfOverlapForParallel,
                                                                    parallelPartitioningType=meshOptions.parallelPartitioningType)
                    if generatePartitionedMeshFromFiles:
                        logEvent("Generating partitioned mesh from Tetgen files")
                        mlMesh.generatePartitionedMeshFromTetgenFiles(fileprefix, nbase,mesh,meshOptions.nLevels,
                                                                        nLayersOfOverlap=meshOptions.nLayersOfOverlapForParallel,
                                                                        parallelPartitioningType=meshOptions.parallelPartitioningType)
                    else:
                        logEvent("Generating coarse global mesh from Tetgen files")
                        mesh.generateFromTetgenFiles(fileprefix, nbase,parallel = comm.size() > 1)
                        logEvent("Generating partitioned %i-level mesh from coarse global Tetgen mesh" % (meshOptions.nLevels,))
                        mlMesh.generateFromExistingCoarseMesh(mesh, meshOptions.nLevels,
                                                                nLayersOfOverlap=meshOptions.nLayersOfOverlapForParallel,
                                                                parallelPartitioningType=meshOptions.parallelPartitioningType)

    elif isinstance(domain, Domain.PUMIDomain):
        import sys
        if(comm.size() >1 and meshOptions.parallelPartitioningType!=MeshParallelPartitioningTypes.element):
            sys.exit("The mesh must be partitioned by elements and NOT nodes for adaptivity functionality. Do this with: `meshOptions.setParallelPartitioningType('element')'.")
        if comm.size() > 1 and getattr(numerics, 'conservativeFlux', None) is not None:
            sys.exit("ERROR: Element based partitions don't have a functioning conservative flux calculation. Set conservativeFlux to None in twp_navier_stokes")
        # ibaned: PUMI conversion #1
        if domain.nd == 3:
            mesh = TetrahedralMesh()
        else:
            mesh = TriangularMesh()
        logEvent("Converting PUMI mesh to Proteus")
        mesh.convertFromPUMI(domain, domain.AdaptManager.PUMIAdapter, domain.faceList,
                             domain.regList,
                             parallel = comm.size() > 1, dim = domain.nd)
        if domain.nd == 3:
            mlMesh = MultilevelTetrahedralMesh(
                0, 0,0,skipInit=True,
                nLayersOfOverlap=meshOptions.nLayersOfOverlapForParallel,
                parallelPartitioningType=meshOptions.parallelPartitioningType)
        if domain.nd == 2:
            mlMesh = MultilevelTriangularMesh(
                0, 0,0,skipInit=True,
                nLayersOfOverlap=meshOptions.nLayersOfOverlapForParallel,
                parallelPartitioningType=meshOptions.parallelPartitioningType)
        logEvent("Generating %i-level mesh from PUMI mesh" % (meshOptions.nLevels,))
        if comm.size() ==1:
            mlMesh.generateFromExistingCoarseMesh(
                mesh, meshOptions.nLevels,
                nLayersOfOverlap=meshOptions.nLayersOfOverlapForParallel,
                parallelPartitioningType=meshOptions.parallelPartitioningType)
        else:
            mlMesh.generatePartitionedMeshFromPUMI(
                mesh, meshOptions.nLevels,
                nLayersOfOverlap=meshOptions.nLayersOfOverlapForParallel)
    elif isinstance(domain, Domain.MeshTetgenDomain):
        nbase = 1
        mesh = TetrahedralMesh()
        logEvent("Reading coarse mesh from tetgen file")
        mlMesh = MultilevelTetrahedralMesh(0, 0,0,skipInit=True,
                                                     nLayersOfOverlap=meshOptions.nLayersOfOverlapForParallel,
                                                     parallelPartitioningType=meshOptions.parallelPartitioningType)
        if generatePartitionedMeshFromFiles:
            logEvent("Generating partitioned mesh from Tetgen files")
            mlMesh.generatePartitionedMeshFromTetgenFiles(domain.meshfile, nbase,mesh,meshOptions.nLevels,
                                                          nLayersOfOverlap=meshOptions.nLayersOfOverlapForParallel,
                                                          parallelPartitioningType=meshOptions.parallelPartitioningType)
        else:
            logEvent("Generating coarse global mesh from Tetgen files")
            mesh.generateFromTetgenFiles(domain.polyfile, nbase,parallel = comm.size() > 1)
            logEvent("Generating partitioned %i-level mesh from coarse global Tetgen mesh" % (meshOptions.nLevels,))
            mlMesh.generateFromExistingCoarseMesh(mesh, meshOptions.nLevels,
                                                  nLayersOfOverlap=meshOptions.nLayersOfOverlapForParallel,
                                                  parallelPartitioningType=meshOptions.parallelPartitioningType)
    elif isinstance(domain, Domain.Mesh3DMDomain):
        mesh = TetrahedralMesh()
        logEvent("Reading coarse mesh from 3DM file")
        mesh.generateFrom3DMFile(domain.meshfile)
        mlMesh = MultilevelTetrahedralMesh(0, 0,0,skipInit=True,
                                                     nLayersOfOverlap=meshOptions.nLayersOfOverlapForParallel,
                                                     parallelPartitioningType=meshOptions.parallelPartitioningType)
        logEvent("Generating %i-level mesh from coarse 3DM mesh" % (meshOptions.nLevels,))
        mlMesh.generateFromExistingCoarseMesh(mesh, meshOptions.nLevels,
                                              nLayersOfOverlap=meshOptions.nLayersOfOverlapForParallel,
                                              parallelPartitioningType=meshOptions.parallelPartitioningType)
    elif isinstance(domain, Domain.Mesh2DMDomain):
        mesh = TriangularMesh()
        logEvent("Reading coarse mesh from 2DM file, converted to a ymf mesh")
        ymf_prefix = os.path.basename(domain.meshfile) + "_2dm"
        if comm.rank() == 0:
            convert2DMToYmf(domain.meshfile, ymf_prefix)
        comm.barrier()
        mesh.generateFromYmfFile(ymf_prefix)
        mlMesh = MultilevelTriangularMesh(0, 0,0,skipInit=True,
                                                    nLayersOfOverlap=meshOptions.nLayersOfOverlapForParallel,
                                                    parallelPartitioningType=meshOptions.parallelPartitioningType)
        logEvent("Generating %i-level mesh from coarse 2DM mesh" % (meshOptions.nLevels,))
        mlMesh.generateFromExistingCoarseMesh(mesh, meshOptions.nLevels,
                                              nLayersOfOverlap=meshOptions.nLayersOfOverlapForParallel,
                                              parallelPartitioningType=meshOptions.parallelPartitioningType)
    elif isinstance(domain, Domain.MeshHexDomain):
        mesh = HexahedralMesh()
        logEvent("Reading coarse mesh from file")
        mesh.generateFromHexFile(domain.meshfile)
        mlMesh = MultilevelHexahedralMesh(0, 0,0,skipInit=True,
                                                    nLayersOfOverlap=meshOptions.nLayersOfOverlapForParallel,
                                                    parallelPartitioningType=meshOptions.parallelPartitioningType)
        logEvent("Generating %i-level mesh from coarse mesh" % (meshOptions.nLevels,))
        mlMesh.generateFromExistingCoarseMesh(mesh, meshOptions.nLevels,
                                              nLayersOfOverlap=meshOptions.nLayersOfOverlapForParallel,
                                              parallelPartitioningType=meshOptions.parallelPartitioningType)
    elif isinstance(domain, Domain.GMSH_3D_Domain):
        from subprocess import call
        import sys
        if comm.rank() == 0 and (meshOptions.genMesh or not (os.path.exists(domain.polyfile+".ele") and
                                                   os.path.exists(domain.polyfile+".node") and
                                                   os.path.exists(domain.polyfile+".face"))):
            logEvent("Running gmsh to generate 3D mesh for "+name, level=1)
            gmsh_cmd = "time gmsh {0:s} -v 10 -3 -o {1:s}  -format mesh  -clmax {2:f}".format(domain.geofile, domain.name+".mesh", 0.5*domainMeshOptions.he)

            logEvent("Calling gmsh on rank 0 with command %s" % (gmsh_cmd,))

            run(gmsh_cmd, shell=True)

            logEvent("Done running gmsh; converting to tetgen")

            gmsh2tetgen_cmd = "gmsh2tetgen {0} {1:f} {2:d} {3:d} {4:d}".format(
                domain.name+".mesh",
                domain.length_scale,
                domain.permute_dims[0]+1,  # switch to base 1 index...
                domain.permute_dims[1]+1,
                domain.permute_dims[2]+1)

            run(gmsh2tetgen_cmd, shell=True)
            fileprefix = "mesh"
            run("rm -f {0:s}.1.ele {0:s}.ele".format(fileprefix), shell=True)
            run("rm -f {0:s}.1.node {0:s}.node".format(fileprefix), shell=True)
            run("rm -f {0:s}.1.face {0:s}.face".format(fileprefix), shell=True)
            run("rm -f {0:s}.1.neigh {0:s}.neigh".format(fileprefix), shell=True)
            run("rm -f {0:s}.1.edge {0:s}.edge".format(fileprefix), shell=True)
            logEvent(run("tetgen -Vfeen %s.ele" % ("mesh",), shell=True,capture_output=True,text=True).stdout)
            run("mv %s.1.ele %s.ele" % ("mesh", "mesh"), shell=True)
            run("mv %s.1.node %s.node" % ("mesh", "mesh"), shell=True)
            run("mv %s.1.face %s.face" % ("mesh", "mesh"), shell=True)
            run("mv %s.1.neigh %s.neigh" % ("mesh", "mesh"), shell=True)
            run("mv %s.1.edge %s.edge" % ("mesh", "mesh"), shell=True)
            elefile = "mesh.ele"
            nodefile = "mesh.node"
            facefile = "mesh.face"
            edgefile = "mesh.edge"
            assert os.path.exists(elefile), "no mesh.ele"
            tmp = "%s.ele" % domain.polyfile
            os.rename(elefile, tmp)
            assert os.path.exists(tmp), "no .ele"
            assert os.path.exists(nodefile), "no mesh.node"
            tmp = "%s.node" % domain.polyfile
            os.rename(nodefile, tmp)
            assert os.path.exists(tmp), "no .node"
            if os.path.exists(facefile):
                tmp = "%s.face" % domain.polyfile
                os.rename(facefile, tmp)
                assert os.path.exists(tmp), "no .face"
            if os.path.exists(edgefile):
                tmp = "%s.edge" % domain.polyfile
                os.rename(edgefile, tmp)
                assert os.path.exists(tmp), "no .edge"
        comm.barrier()
        logEvent("Initializing mesh and MultilevelMesh")
        nbase = 1
        mesh = TetrahedralMesh()
        mlMesh = MultilevelTetrahedralMesh(0, 0,0,skipInit=True,
                                                     nLayersOfOverlap=meshOptions.nLayersOfOverlapForParallel,
                                                     parallelPartitioningType=meshOptions.parallelPartitioningType)
        if generatePartitionedMeshFromFiles:
            logEvent("Generating partitioned mesh from Tetgen files")
            mlMesh.generatePartitionedMeshFromTetgenFiles(domain.polyfile, nbase,mesh,meshOptions.nLevels,
                                                          nLayersOfOverlap=meshOptions.nLayersOfOverlapForParallel,
                                                          parallelPartitioningType=meshOptions.parallelPartitioningType)
        else:
            logEvent("Generating coarse global mesh from Tetgen files")
            mesh.generateFromTetgenFiles(domain.polyfile, nbase,parallel = comm.size() > 1)
            logEvent("Generating partitioned %i-level mesh from coarse global Tetgen mesh" % (meshOptions.nLevels,))
            mlMesh.generateFromExistingCoarseMesh(mesh, meshOptions.nLevels,
                                                  nLayersOfOverlap=meshOptions.nLayersOfOverlapForParallel,
                                                  parallelPartitioningType=meshOptions.parallelPartitioningType)

    elif isinstance(domain, Domain.PlanarStraightLineGraphDomain):
        fileprefix = None
        # run mesher
        if domainMeshOptions.use_gmsh is True:
            fileprefix = domain.geofile
            if comm.isMaster() and (not (os.path.exists(fileprefix+".ele") and
                                         os.path.exists(fileprefix+".node") and
                                         os.path.exists(fileprefix+".edge"))):
                writeMeshFiles(domain, meshOptions)
        else:
            fileprefix = domain.polyfile
            if comm.isMaster() and meshOptions.genMesh:
                writeMeshFiles(domain, meshOptions)
        comm.barrier()
        assert fileprefix is not None, 'did not find mesh file name'
        # convert mesh to proteus format
        nbase = 1
        mesh = TriangularMesh()
        mlMesh = MultilevelTriangularMesh(0, 0,0,skipInit=True,
                                          nLayersOfOverlap=meshOptions.nLayersOfOverlapForParallel,
                                          parallelPartitioningType=meshOptions.parallelPartitioningType)
        logEvent("Generating %i-level mesh from coarse Triangle mesh" % (meshOptions.nLevels,))
        if generatePartitionedMeshFromFiles:
            logEvent("Generating partitioned mesh from Triangle files")
            mlMesh.generatePartitionedMeshFromTriangleFiles(fileprefix, nbase,mesh,meshOptions.nLevels,
                                                          nLayersOfOverlap=meshOptions.nLayersOfOverlapForParallel,
                                                          parallelPartitioningType=meshOptions.parallelPartitioningType)
        else:
            mesh.generateFromTriangleFiles(filebase=fileprefix,
                                       base=1)
            mlMesh.generateFromExistingCoarseMesh(mesh, meshOptions.nLevels,
                                              nLayersOfOverlap=meshOptions.nLayersOfOverlapForParallel,
                                              parallelPartitioningType=meshOptions.parallelPartitioningType)
    elif isinstance(domain, Domain.PiecewiseLinearComplexDomain):
        import sys
        if domainMeshOptions.use_gmsh is True:
            fileprefix = domain.geofile
        else:
            fileprefix = domain.polyfile
        if comm.rank() == 0 and (meshOptions.genMesh or (
                not (os.path.exists(fileprefix+".ele") and
                     os.path.exists(fileprefix+".node") and
                     os.path.exists(fileprefix+".face")))):
            writeMeshFiles(domain, meshOptions)
        comm.barrier()
        logEvent("Initializing mesh and MultilevelMesh")
        nbase = 1
        mesh = TetrahedralMesh()
        mlMesh = MultilevelTetrahedralMesh(0, 0,0,skipInit=True,
                                           nLayersOfOverlap=meshOptions.nLayersOfOverlapForParallel,
                                           parallelPartitioningType=meshOptions.parallelPartitioningType)
        if generatePartitionedMeshFromFiles:
            logEvent("Generating partitioned mesh from Tetgen files")
            if("f" not in meshOptions.triangleOptions or "ee" not in meshOptions.triangleOptions):
                sys.exit("ERROR: Remake the mesh with the `f` flag and `ee` flags in triangleOptions.")
            mlMesh.generatePartitionedMeshFromTetgenFiles(fileprefix, nbase,mesh,meshOptions.nLevels,
                                                          nLayersOfOverlap=meshOptions.nLayersOfOverlapForParallel,
                                                          parallelPartitioningType=meshOptions.parallelPartitioningType)
        else:
            logEvent("Generating coarse global mesh from Tetgen files")
            mesh.generateFromTetgenFiles(fileprefix, nbase,parallel = comm.size() > 1)
            logEvent("Generating partitioned %i-level mesh from coarse global Tetgen mesh" % (meshOptions.nLevels,))
            mlMesh.generateFromExistingCoarseMesh(mesh, meshOptions.nLevels,
                                                  nLayersOfOverlap=meshOptions.nLayersOfOverlapForParallel,
                                                  parallelPartitioningType=meshOptions.parallelPartitioningType)

    elif isinstance(domain, Domain.DMPlexDomain):
        import sys
        
        # Here we want to create a mesh using DMPlex. 
        logEvent("Initializing plex mesh")
        nbase = 1
        mesh = TetrahedralMesh()
        mlMesh = MultilevelTetrahedralMesh(0, 0,0,skipInit=True,
                                           nLayersOfOverlap=meshOptions.nLayersOfOverlapForParallel,
                                           parallelPartitioningType=meshOptions.parallelPartitioningType)
        if generatePartitionedMeshFromFiles:
            logEvent("Generating partitioned mesh from Tetgen files")
            if("f" not in meshOptions.triangleOptions or "ee" not in meshOptions.triangleOptions):
                sys.exit("ERROR: Remake the mesh with the `f` flag and `ee` flags in triangleOptions.")
            mlMesh.generatePartitionedMeshFromTetgenFiles(fileprefix, nbase,mesh,meshOptions.nLevels,
                                                          nLayersOfOverlap=meshOptions.nLayersOfOverlapForParallel,
                                                          parallelPartitioningType=meshOptions.parallelPartitioningType)
        else:
            logEvent("Generating coarse global mesh from Plex")
            # mesh.generateFromTetgenFiles(fileprefix, nbase,parallel = comm.size() > 1)
            # generate using Plex
            mesh.generateFromPlex(domain.plex)
            logEvent("Generating partitioned %i-level mesh from Plex mesh" % (meshOptions.nLevels,))
            # generate from Existing plex and ask DM to do refinement process? or ask CMesh to do it? 
            # mlMesh.generatePartitionedMeshFromPlex(mesh, meshOptions.nLevels)
            mlMesh.generateFromExistingCoarseMesh(mesh, meshOptions.nLevels,
                                                  nLayersOfOverlap=meshOptions.nLayersOfOverlapForParallel,
                                                  parallelPartitioningType=meshOptions.parallelPartitioningType)

    meshOptions.genMesh = False
    return mlMesh
