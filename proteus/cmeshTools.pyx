# A type of -*- python -*- file
# cython: language_level=3
from os.path import exists
import cython
cimport cython
import numpy as np
cimport numpy as np
from libcpp cimport bool
from mpi4py import MPI
cimport mesh as cppm
from proteus import Comm

#: the dtype of every mesh entity number, count and offset array, read from
#: proteus_index_t in index_type.h so the two cannot disagree
INDEX_DTYPE = np.dtype('i%d' % sizeof(cppm.proteus_index_t))

cdef object _indexArray(cppm.proteus_index_t* p, cppm.proteus_index_t n0, cppm.proteus_index_t n1=-1):
    """A numpy view of the n0 (x n1) mesh indices at p, as _intArray."""
    if n0 == 0 or n1 == 0:
        return np.empty([n0] if n1 < 0 else [n0, n1], dtype=INDEX_DTYPE)
    if n1 < 0:
        return np.asarray(<cppm.proteus_index_t[:n0]> p)
    return np.asarray(<cppm.proteus_index_t[:n0, :n1]> p)

cdef object _intArray(int* p, int n0, int n1=-1):
    """A numpy view of the n0 (x n1) ints at p. A subdomain mesh is empty on a
    rank that owns nothing, and a typed memoryview cannot describe an empty
    array."""
    if n0 == 0 or n1 == 0:
        return np.empty([n0] if n1 < 0 else [n0, n1], dtype=np.intc)
    if n1 < 0:
        return np.asarray(<int[:n0]> p)
    return np.asarray(<int[:n0, :n1]> p)

cdef object _doubleArray(double* p, int n0, int n1=-1):
    """As _intArray, for doubles."""
    if n0 == 0 or n1 == 0:
        return np.empty([n0] if n1 < 0 else [n0, n1])
    if n1 < 0:
        return np.asarray(<double[:n0]> p)
    return np.asarray(<double[:n0, :n1]> p)

cdef class CMesh:
    def __init__(self):
        self.mesh = cppm.Mesh()
        cppm.initializeMesh(self.mesh)

    def deleteCMesh(self):
        cppm.deleteMesh(self.mesh)

    def buildPythonMeshInterface(self):
        cdef int dim1
        self.nElements_global = self.mesh.nElements_global
        self.nNodes_global = self.mesh.nNodes_global
        self.nNodes_element = self.mesh.nNodes_element
        self.nNodes_elementBoundary = self.mesh.nNodes_elementBoundary
        self.nElementBoundaries_element = self.mesh.nElementBoundaries_element
        self.nElementBoundaries_global = self.mesh.nElementBoundaries_global
        self.nInteriorElementBoundaries_global = self.mesh.nInteriorElementBoundaries_global
        self.nExteriorElementBoundaries_global = self.mesh.nExteriorElementBoundaries_global
        self.max_nElements_node = self.mesh.max_nElements_node
        self.nEdges_global = self.mesh.nEdges_global
        self.max_nNodeNeighbors_node = self.mesh.max_nNodeNeighbors_node
        self.elementNodesArray = _indexArray(self.mesh.elementNodesArray, self.mesh.nElements_global, self.mesh.nNodes_element)
        if self.mesh.nodeElementOffsets != NULL:
            self.nodeElementsArray = _indexArray(self.mesh.nodeElementsArray, self.mesh.nodeElementOffsets[self.mesh.nNodes_global])
            self.nodeElementOffsets = _indexArray(self.mesh.nodeElementOffsets, self.mesh.nNodes_global+1)
        else:
            self.nodeElementsArray = np.empty(0, dtype=INDEX_DTYPE)
            self.nodeElementOffsets = np.empty(0, dtype=INDEX_DTYPE)

        self.elementNeighborsArray = _indexArray(self.mesh.elementNeighborsArray, self.mesh.nElements_global, self.mesh.nElementBoundaries_element)
        self.elementBoundariesArray = _indexArray(self.mesh.elementBoundariesArray, self.mesh.nElements_global, self.mesh.nElementBoundaries_element)
        self.elementBoundaryNodesArray = _indexArray(self.mesh.elementBoundaryNodesArray, self.mesh.nElementBoundaries_global, self.mesh.nNodes_elementBoundary)
        self.elementBoundaryElementsArray = _indexArray(self.mesh.elementBoundaryElementsArray, self.mesh.nElementBoundaries_global, 2)
        self.elementBoundaryLocalElementBoundariesArray = _intArray(self.mesh.elementBoundaryLocalElementBoundariesArray, self.mesh.nElementBoundaries_global, 2)
        if self.mesh.nInteriorElementBoundaries_global:
            self.interiorElementBoundariesArray = _indexArray(self.mesh.interiorElementBoundariesArray, self.mesh.nInteriorElementBoundaries_global)
        else:
            self.interiorElementBoundariesArray = np.empty(0, dtype=INDEX_DTYPE)
        self.exteriorElementBoundariesArray = _indexArray(self.mesh.exteriorElementBoundariesArray, self.mesh.nExteriorElementBoundaries_global)
        self.edgeNodesArray = _indexArray(self.mesh.edgeNodesArray, self.mesh.nEdges_global, 2)
        if self.mesh.nodeStarOffsets != NULL:
            self.nodeStarArray = _indexArray(self.mesh.nodeStarArray, self.mesh.nodeStarOffsets[self.mesh.nNodes_global])
        else:
            self.nodeStarArray = np.empty(0, dtype=INDEX_DTYPE)
        self.nodeStarOffsets = _indexArray(self.mesh.nodeStarOffsets, self.mesh.nNodes_global+1)
        self.elementMaterialTypes = _intArray(self.mesh.elementMaterialTypes, self.mesh.nElements_global)
        self.elementBoundaryMaterialTypes = _intArray(self.mesh.elementBoundaryMaterialTypes, self.mesh.nElementBoundaries_global)
        self.nodeMaterialTypes = _intArray(self.mesh.nodeMaterialTypes, self.mesh.nNodes_global)
        self.nodeArray = _doubleArray(self.mesh.nodeArray, self.mesh.nNodes_global, 3)
        self.nx = self.mesh.nx
        self.ny = self.mesh.ny
        self.nz = self.mesh.nz
        self.px = self.mesh.px
        self.py = self.mesh.py
        self.pz = self.mesh.pz
        if self.mesh.elementIJK != NULL:
            self.elementIJK = _intArray(self.mesh.elementIJK, self.mesh.nElements_global*3)
        else:
            self.elementIJK = np.empty(0, dtype=np.int32)
        if self.mesh.weights != NULL:
            self.weights = _doubleArray(self.mesh.weights, self.mesh.nElements_global)
        else:
            self.weights = np.empty(0)
        if self.mesh.U_KNOT != NULL:
            self.U_KNOT = _doubleArray(self.mesh.U_KNOT, self.mesh.nx+self.mesh.px+1)
        else:
            self.U_KNOT = np.empty(0)
        if self.mesh.V_KNOT != NULL:
            self.V_KNOT = _doubleArray(self.mesh.V_KNOT, self.mesh.ny+self.mesh.py+1)
        else:
            self.V_KNOT = np.empty(0)
        if self.mesh.W_KNOT != NULL:
            self.W_KNOT = _doubleArray(self.mesh.W_KNOT, self.mesh.nz+self.mesh.pz+1)
        else:
            self.W_KNOT = np.empty(0)
        self.elementDiametersArray = _doubleArray(self.mesh.elementDiametersArray, self.mesh.nElements_global)
        self.elementInnerDiametersArray = _doubleArray(self.mesh.elementInnerDiametersArray, self.mesh.nElements_global)
        self.elementBoundaryDiametersArray = _doubleArray(self.mesh.elementBoundaryDiametersArray, self.mesh.nElementBoundaries_global)
        if self.mesh.elementBarycentersArray:
            self.elementBarycentersArray = _doubleArray(self.mesh.elementBarycentersArray, self.mesh.nElements_global, 3)
        if self.mesh.elementBoundaryBarycentersArray:
            self.elementBoundaryBarycentersArray = _doubleArray(self.mesh.elementBoundaryBarycentersArray, self.mesh.nElementBoundaries_global, 3)
        if self.mesh.nodeDiametersArray:
            self.nodeDiametersArray = _doubleArray(self.mesh.nodeDiametersArray, self.mesh.nNodes_global)
        if self.mesh.nodeSupportArray:
            self.nodeSupportArray = _doubleArray(self.mesh.nodeSupportArray, self.mesh.nNodes_global)
        self.h = self.mesh.h
        self.hMin = self.mesh.hMin
        self.sigmaMax = self.mesh.sigmaMax
        self.volume = self.mesh.volume

    def buildPythonMeshInterfaceNoArrays(self):
        cdef int dim1
        self.nElements_global = self.mesh.nElements_global
        self.nNodes_global = self.mesh.nNodes_global
        self.nNodes_element = self.mesh.nNodes_element
        self.nNodes_elementBoundary = self.mesh.nNodes_elementBoundary
        self.nElementBoundaries_element = self.mesh.nElementBoundaries_element
        self.nElementBoundaries_global = self.mesh.nElementBoundaries_global
        self.nInteriorElementBoundaries_global = self.mesh.nInteriorElementBoundaries_global
        self.nExteriorElementBoundaries_global = self.mesh.nExteriorElementBoundaries_global
        self.max_nElements_node = self.mesh.max_nElements_node
        self.nEdges_global = self.mesh.nEdges_global
        self.max_nNodeNeighbors_node = self.mesh.max_nNodeNeighbors_node
        self.h = self.mesh.h
        self.hMin = self.mesh.hMin
        self.sigmaMax = self.mesh.sigmaMax
        self.volume = self.mesh.volume

def buildPythonMeshInterface(cmesh):
    """
    function to be conform to old module, and to calls from MeshTools
    """
    cmesh.buildPythonMeshInterface()
    return (cmesh.nElements_global,
            cmesh.nNodes_global,
            cmesh.nNodes_element,
            cmesh.nNodes_elementBoundary,
            cmesh.nElementBoundaries_element,
            cmesh.nElementBoundaries_global,
            cmesh.nInteriorElementBoundaries_global,
            cmesh.nExteriorElementBoundaries_global,
            cmesh.max_nElements_node,
            cmesh.nEdges_global,
            cmesh.max_nNodeNeighbors_node,
            cmesh.elementNodesArray,
            cmesh.nodeElementsArray,
            cmesh.nodeElementOffsets,
            cmesh.elementNeighborsArray,
            cmesh.elementBoundariesArray,
            cmesh.elementBoundaryNodesArray,
            cmesh.elementBoundaryElementsArray,
            cmesh.elementBoundaryLocalElementBoundariesArray,
            cmesh.interiorElementBoundariesArray,
            cmesh.exteriorElementBoundariesArray,
            cmesh.edgeNodesArray,
            cmesh.nodeStarArray,
            cmesh.nodeStarOffsets,
            cmesh.elementMaterialTypes,
            cmesh.elementBoundaryMaterialTypes,
            cmesh.nodeMaterialTypes,
            cmesh.nodeArray,
            cmesh.nx,
            cmesh.ny,
            cmesh.nz,
            cmesh.px,
            cmesh.py,
            cmesh.pz,
            cmesh.elementIJK,
            cmesh.weights,
            cmesh.U_KNOT,
            cmesh.V_KNOT,
            cmesh.W_KNOT,
            cmesh.elementDiametersArray,
            cmesh.elementInnerDiametersArray,
            cmesh.elementBoundaryDiametersArray,
            cmesh.elementBarycentersArray,
            cmesh.elementBoundaryBarycentersArray,
            cmesh.nodeDiametersArray,
            cmesh.nodeSupportArray,
            cmesh.h,
            cmesh.hMin,
            cmesh.sigmaMax,
            cmesh.volume)

def buildPythonMeshInterfaceNoArrays(cmesh):
    """
    function to be conform to old module, and to calls from MeshTools
    """
    cmesh.buildPythonMeshInterfaceNoArrays()
    return (cmesh.nElements_global,
            cmesh.nNodes_global,
            cmesh.nNodes_element,
            cmesh.nNodes_elementBoundary,
            cmesh.nElementBoundaries_element,
            cmesh.nElementBoundaries_global,
            cmesh.nInteriorElementBoundaries_global,
            cmesh.nExteriorElementBoundaries_global,
            cmesh.max_nElements_node,
            cmesh.nEdges_global,
            cmesh.max_nNodeNeighbors_node,
            cmesh.h,
            cmesh.hMin,
            cmesh.sigmaMax,
            cmesh.volume)

cdef CMesh CMesh_FromMesh(cppm.Mesh mesh):
    CMeshnew = CMesh()
    CMeshnew.mesh = mesh
    return CMeshnew

cdef class CMultilevelMesh:
    cdef cppm.MultilevelMesh multilevelMesh
    cdef public:
        int nLevels
        list cmeshList
        list elementParentsArrayList
        list elementChildrenArrayList
        list elementChildrenOffsetsList
    def __init__(self, CMesh cmesh, nLevels):
        self.multilevelMesh = cppm.MultilevelMesh()
        if cmesh.mesh.nNodes_element == 2:
            cppm.globallyRefineEdgeMesh(nLevels,
                                        cmesh.mesh,
                                        self.multilevelMesh,
                                        False);
            for i in range(1, nLevels):
                cppm.constructElementBoundaryElementsArray_edge(self.multilevelMesh.meshArray[i]);
                cppm.allocateGeometricInfo_edge(self.multilevelMesh.meshArray[i]);
                cppm.computeGeometricInfo_edge(self.multilevelMesh.meshArray[i]);
                cppm.assignElementBoundaryMaterialTypesFromParent(self.multilevelMesh.meshArray[i-1],
                                                                  self.multilevelMesh.meshArray[i],
                                                                  self.multilevelMesh.elementParentsArray[i],
                                                                  1);
        elif cmesh.mesh.nNodes_element == 3:
            cppm.globallyRefineTriangularMesh(nLevels,
                                              cmesh.mesh,
                                              self.multilevelMesh,
                                              False);
            for i in range(1, nLevels):
                cppm.constructElementBoundaryElementsArray_triangle(self.multilevelMesh.meshArray[i]);
                cppm.allocateGeometricInfo_triangle(self.multilevelMesh.meshArray[i]);
                cppm.computeGeometricInfo_triangle(self.multilevelMesh.meshArray[i]);
                cppm.assignElementBoundaryMaterialTypesFromParent(self.multilevelMesh.meshArray[i-1],
						                                     self.multilevelMesh.meshArray[i],
						                                     self.multilevelMesh.elementParentsArray[i],
						                                     2);
        elif cmesh.mesh.nNodes_element == 4 and cmesh.mesh.nNodes_elementBoundary == 2:
            cppm.globallyRefineQuadrilateralMesh(nLevels,
                                                 cmesh.mesh,
                                                 self.multilevelMesh,
                                                 False);
            for i in range(1, nLevels):
                cppm.constructElementBoundaryElementsArray_quadrilateral(self.multilevelMesh.meshArray[i]);
                cppm.allocateGeometricInfo_quadrilateral(self.multilevelMesh.meshArray[i]);
                cppm.computeGeometricInfo_quadrilateral(self.multilevelMesh.meshArray[i]);
                cppm.assignElementBoundaryMaterialTypesFromParent(self.multilevelMesh.meshArray[i-1],
						                                          self.multilevelMesh.meshArray[i],
						                                          self.multilevelMesh.elementParentsArray[i],
						                                          2);
        elif cmesh.mesh.nNodes_element == 4 and cmesh.mesh.nNodes_elementBoundary == 3:
            cppm.globallyRefineTetrahedralMesh(nLevels,
                                               cmesh.mesh,
                                               self.multilevelMesh,
                                               False);
            for i in range(1, nLevels):
                cppm.constructElementBoundaryElementsArray_tetrahedron(self.multilevelMesh.meshArray[i]);
                cppm.allocateGeometricInfo_tetrahedron(self.multilevelMesh.meshArray[i]);
                cppm.computeGeometricInfo_tetrahedron(self.multilevelMesh.meshArray[i]);
                cppm.assignElementBoundaryMaterialTypesFromParent(self.multilevelMesh.meshArray[i-1],
						                                     self.multilevelMesh.meshArray[i],
						                                     self.multilevelMesh.elementParentsArray[i],
						                                     3);
        elif cmesh.mesh.nNodes_element == 8 and cmesh.mesh.nNodes_elementBoundary == 4:
            cppm.globallyRefineHexahedralMesh(nLevels,
                                         cmesh.mesh,
                                         self.multilevelMesh,
                                         False);
            for i in range(1, nLevels):
                cppm.constructElementBoundaryElementsArray_hexahedron(self.multilevelMesh.
                                                                      meshArray[i]);
                cppm.allocateGeometricInfo_hexahedron(self.multilevelMesh.meshArray[i]);
                cppm.computeGeometricInfo_hexahedron(self.multilevelMesh.meshArray[i]);
                cppm.assignElementBoundaryMaterialTypesFromParent(self.multilevelMesh.meshArray[i-1],
						                                          self.multilevelMesh.meshArray[i],
						                                          self.multilevelMesh.elementParentsArray[i],
						                                          4);
        else:
            assert nLevels == 1, 'wrong nLevels'
            cppm.globallyRefineTetrahedralMesh(nLevels,
                                               cmesh.mesh,
                                               self.multilevelMesh,
                                               False);

    def buildPythonMultilevelMeshInterface(self):
        cdef int dim
        self.cmeshList = [CMesh_FromMesh(self.multilevelMesh.meshArray[0])]
        self.elementParentsArrayList = [np.zeros(0)]
        self.elementChildrenArrayList = []
        self.elementChildrenOffsetsList = []
        for n in range(1, self.multilevelMesh.nLevels):
            self.cmeshList += [CMesh_FromMesh(self.multilevelMesh.meshArray[n])]
            dim = self.multilevelMesh.meshArray[n].nElements_global
            self.elementParentsArrayList += [np.asarray(<int[:dim]> self.multilevelMesh.elementParentsArray[n])]
            dim = self.multilevelMesh.elementChildrenOffsets[n-1][self.multilevelMesh.meshArray[n-1].nElements_global]
            self.elementChildrenArrayList += [np.asarray(<int[:dim]> self.multilevelMesh.elementChildrenArray[n-1])]
            dim = self.multilevelMesh.meshArray[n-1].nElements_global+1;
            self.elementChildrenOffsetsList += [np.asarray(<int[:dim]> self.multilevelMesh.elementChildrenOffsets[n-1])]

def buildPythonMultilevelMeshInterface(cmultilevelMesh):
    cmultilevelMesh.buildPythonMultilevelMeshInterface()
    return (cmultilevelMesh.nLevels,
            cmultilevelMesh.cmeshList,
            cmultilevelMesh.elementParentsArrayList,
            cmultilevelMesh.elementChildrenArrayList,
            cmultilevelMesh.elementChildrenOffsetsList)

cpdef void generateTetrahedralMeshFromRectangularGrid(int nx,
                                                      int ny,
                                                      int nz,
                                                      double Lx,
                                                      double Ly,
                                                      double Lz,
                                                      CMesh cmesh):
    cppm.regularHexahedralToTetrahedralMeshElements(nx,ny,nz,cmesh.mesh);
    cppm.regularHexahedralToTetrahedralMeshNodes(nx,ny,nz,Lx,Ly,Lz,cmesh.mesh);
    cppm.constructElementBoundaryElementsArray_tetrahedron(cmesh.mesh);
    cppm.regularHexahedralToTetrahedralElementBoundaryMaterials(Lx,Ly,Lz,cmesh.mesh);

cpdef void cmeshToolsComputeGeometricInfo_tetrahedron(CMesh cmesh):
    cppm.computeGeometricInfo_tetrahedron(cmesh.mesh)

# cdef static void cmeshToolsLocallyRefineMultilevelMesh(CMultilevelMesh cmesh,


cpdef void generateFromTriangleFiles(CMesh cmesh,
                                    unicode filebase,
                                    int base):

    cdef int failed
    failed = cppm.readTriangleMesh(cmesh.mesh,filebase.encode('utf8'),base);
    cppm.constructElementBoundaryElementsArray_triangle(cmesh.mesh);
    failed = cppm.readTriangleElementBoundaryMaterialTypes(cmesh.mesh,filebase.encode('utf8'),base);

cpdef void writeTriangleFiles(CMesh cmesh,
                             unicode filebase,
                             int base):
    cdef int failed
    failed = cppm.writeTriangleMesh(cmesh.mesh,filebase.encode('utf8'),base);

cpdef void generateCMeshFromDMPlex(self, CMesh cmesh):
    cdef int failed
    failed = cppm.readDMPlexMesh(self, cmesh.mesh);

    # int
    cmesh.nElements_global                           = self.nElements_global
    cmesh.nNodes_global                              = self.nNodes_global
    cmesh.nNodes_element                             = self.nNodes_element
    cmesh.elementNodesArray                          = self.elementNodesArray
    cmesh.nodeMaterialTypes                          = self.nodeMaterialTypes
    cmesh.elementMaterialTypes                       = self.elementMaterialTypes
    cmesh.nodeArray                                  = self.nodeArray

    # cppm.constructElementBoundaryElementsArray_tetrahedron(cmesh.mesh);

    cmesh.nNodes_elementBoundary                     = self.nNodes_elementBoundary
    cmesh.nEdges_global                              = self.nEdges_global
    cmesh.nElementBoundaries_element                 = self.nElementBoundaries_element
    cmesh.nElementBoundaries_global                  = self.nElementBoundaries_global
    cmesh.nInteriorElementBoundaries_global          = self.nInteriorElementBoundaries_global
    cmesh.nExteriorElementBoundaries_global          = self.nExteriorElementBoundaries_global
    cmesh.max_nNodeNeighbors_node                    = self.max_nNodeNeighbors_node

    # int*
    cmesh.nodeElementsArray                          = self.nodeElementsArray
    
    cmesh.nodeElementOffsets                         = self.nodeElementOffsets
    cmesh.elementBoundariesArray                     = self.elementBoundariesArray
    cmesh.elementBoundaryNodesArray                  = self.elementBoundaryNodesArray
    cmesh.elementBoundaryElementsArray               = self.elementBoundaryElementsArray
    cmesh.interiorElementBoundariesArray             = self.interiorElementBoundariesArray
    cmesh.exteriorElementBoundariesArray             = self.exteriorElementBoundariesArray
    cmesh.edgeNodesArray                             = self.edgeNodesArray
    cmesh.nodeStarArray                              = self.nodeStarArray
    cmesh.nodeStarOffsets                            = self.nodeStarOffsets

cpdef void generateFromTetgenFiles(CMesh cmesh,
                                  unicode filebase,
                                  int base):
    cdef int failed
    failed = cppm.readTetgenMesh(cmesh.mesh,filebase.encode('utf8'),base);
    cppm.constructElementBoundaryElementsArray_tetrahedron(cmesh.mesh);
    failed = cppm.readTetgenElementBoundaryMaterialTypes(cmesh.mesh,filebase.encode('utf8'),base);

cpdef void generateFromTetgenFilesParallel(CMesh cmesh,
                                          unicode filebase,
                                          int base):
    cdef int failed
    failed = cppm.readTetgenMesh(cmesh.mesh,filebase.encode('utf8'),base);
    cppm.constructElementBoundaryElementsArray_tetrahedron(cmesh.mesh);
    failed = cppm.readTetgenElementBoundaryMaterialTypes(cmesh.mesh,filebase.encode('utf8'),base);

cpdef void writeTetgenFiles(CMesh cmesh,
                           unicode filebase,
                           int base):
    cdef int failed
    cppm.reorientTetrahedralMesh(cmesh.mesh);
    failed = cppm.writeTetgenMesh(cmesh.mesh,filebase.encode('utf8'),base);

cpdef void write3dmFiles(CMesh cmesh,
                        unicode filebase,
                        int base):
    cdef int failed
    failed = cppm.write3dmMesh(cmesh.mesh,filebase.encode('utf8'),base);

cpdef void write2dmFiles(CMesh cmesh,
                        unicode filebase,
                        int base):
    cdef int failed
    failed = cppm.write2dmMesh(cmesh.mesh,filebase.encode('utf8'),base);

cpdef void generateFromHexFile(CMesh cmesh,
                              unicode filebase,
                              int base):
    cdef int failed
    failed = cppm.readHex(cmesh.mesh,filebase.encode('utf8'),base);
    cppm.constructElementBoundaryElementsArray_hexahedron(cmesh.mesh);

cpdef void generateFrom3DMFile(CMesh cmesh,
                              unicode filebase,
                              int base):
    cdef int failed
    failed = cppm.read3DM(cmesh.mesh,filebase.encode('utf8'),base);
    cppm.constructElementBoundaryElementsArray_tetrahedron(cmesh.mesh);
    if exists(filebase+".bc"):
        failed = cppm.readBC(cmesh.mesh,filebase.encode('utf8'),base);
    else:
        failed = False;
    
cpdef void generateFrom2DMFile(CMesh cmesh,
                              unicode filebase,
                              int base):
    cdef int failed
    failed = cppm.read2DM(cmesh.mesh,filebase.encode('utf8'),base);
    cppm.constructElementBoundaryElementsArray_triangle(cmesh.mesh);
    if exists(filebase+".bc"):
        failed = cppm.readBC(cmesh.mesh,filebase.encode('utf8'),base);
    else:
        failed = False

def generateFromArrays(CMesh cmesh, nodes, elements, nodeMaterialTypes=None,
                       elementMaterialTypes=None):
    """Fill a CMesh of simplices from arrays, the in-memory counterpart of the file readers.

    ``nodes`` is (nNodes, 3) coordinates, ``elements`` (nElements, 3 or 4)
    zero-based node numbers; the material types default to zero. The element
    boundaries are constructed as every file reader does, so the caller goes
    on to allocate and compute the geometric info as after a read.
    """
    cdef double[:, ::1] x = np.ascontiguousarray(nodes, dtype=np.float64)
    cdef cppm.proteus_index_t[:, ::1] e = np.ascontiguousarray(elements, dtype=INDEX_DTYPE)
    cdef int[::1] nmt, emt
    cdef int i, j
    cdef int nNodes = x.shape[0]
    cdef int nElements = e.shape[0]
    cdef int nNodes_element = e.shape[1]
    if x.shape[1] != 3:
        raise ValueError("nodes must have shape (nNodes, 3), not (%d, %d)" % (x.shape[0], x.shape[1]))
    if nNodes_element not in (3, 4):
        raise ValueError("elements must be triangles or tetrahedra, not %d nodes each" % nNodes_element)
    nmt = np.zeros(nNodes, dtype=np.intc) if nodeMaterialTypes is None else \
        np.ascontiguousarray(nodeMaterialTypes, dtype=np.intc)
    emt = np.zeros(nElements, dtype=np.intc) if elementMaterialTypes is None else \
        np.ascontiguousarray(elementMaterialTypes, dtype=np.intc)
    if nmt.shape[0] != nNodes or emt.shape[0] != nElements:
        raise ValueError("material types must match the nodes and elements")
    if nElements > 0 and (np.min(e) < 0 or np.max(e) >= nNodes):
        raise ValueError("an element refers to a node outside 0..%d" % (nNodes - 1))
    cppm.allocateNodeAndElementNodeDataStructures(cmesh.mesh, nElements, nNodes, nNodes_element)
    for i in range(nNodes):
        for j in range(3):
            cmesh.mesh.nodeArray[3*i + j] = x[i, j]
        cmesh.mesh.nodeMaterialTypes[i] = nmt[i]
    for i in range(nElements):
        for j in range(nNodes_element):
            cmesh.mesh.elementNodesArray[nNodes_element*i + j] = e[i, j]
        cmesh.mesh.elementMaterialTypes[i] = emt[i]
    if nNodes_element == 3:
        cppm.constructElementBoundaryElementsArray_triangle(cmesh.mesh)
    else:
        cppm.constructElementBoundaryElementsArray_tetrahedron(cmesh.mesh)

cpdef void computeGeometricInfo_tetrahedron(CMesh cmesh):
    cppm.computeGeometricInfo_tetrahedron(cmesh.mesh);

cpdef void allocateGeometricInfo_tetrahedron(CMesh cmesh):
    cppm.allocateGeometricInfo_tetrahedron(cmesh.mesh);

cpdef void allocateNodeAndElementNodeDataStructures(CMesh cmesh,
                                                    int nElements_global,
                                                    int nNodes_global,
                                                    int nNodes_element):
    cppm.allocateNodeAndElementNodeDataStructures(cmesh.mesh,nElements_global,nNodes_global,nNodes_element);

cpdef void constructElementBoundaryElementsArray(CMesh cmesh):
    if cmesh.mesh.nNodes_element == 4:
        cppm.constructElementBoundaryElementsArray_tetrahedron(cmesh.mesh);
    elif cmesh.mesh.nNodes_element == 3:
        cppm.constructElementBoundaryElementsArray_triangle(cmesh.mesh);
    else:
        cppm.constructElementBoundaryElementsArray_edge(cmesh.mesh);

cpdef void generateTriangularMeshFromRectangularGrid(int nx,
                                                     int ny,
                                                     double Lx,
                                                     double Ly,
                                                     CMesh cmesh,
                                                     int triangleFlag):
    cppm.regularRectangularToTriangularMeshElements(nx,ny,cmesh.mesh,triangleFlag);
    cppm.regularRectangularToTriangularMeshNodes(nx,ny,Lx,Ly,cmesh.mesh);
    cppm.constructElementBoundaryElementsArray_triangle(cmesh.mesh);
    cppm.regularRectangularToTriangularElementBoundaryMaterials(Lx,Ly,cmesh.mesh);

cpdef void generateHexahedralMeshFromRectangularGrid(int nx,
                                                     int ny,
                                                     int nz,
                                                     int px,
                                                     int py,
                                                     int pz,
                                                     double Lx,
                                                     double Ly,
                                                     double Lz,
                                                     CMesh cmesh):
    cppm.regularHexahedralMeshElements(nx,ny,nz,px,py,pz,cmesh.mesh);
    cppm.regularMeshNodes(nx,ny,nz,Lx,Ly,Lz,cmesh.mesh);
    cppm.constructElementBoundaryElementsArray_hexahedron(cmesh.mesh);
    cppm.regularHexahedralMeshElementBoundaryMaterials(Lx,Ly,Lz,cmesh.mesh);

cpdef void generateQuadrilateralMeshFromRectangularGrid(int nx,
                                                        int ny,
                                                        int px,
                                                        int py,
                                                        double Lx,
                                                        double Ly,
                                                        CMesh cmesh):
    cppm.regularQuadrilateralMeshElements(nx,ny,cmesh.mesh);
    cppm.regularMeshNodes2D(nx,ny,Lx,Ly,cmesh.mesh);
    cppm.constructElementBoundaryElementsArray_quadrilateral(cmesh.mesh);
    cppm.regularQuadrilateralMeshElementBoundaryMaterials(Lx,Ly,cmesh.mesh);

cpdef void computeGeometricInfo_triangle(CMesh cmesh):
    cppm.computeGeometricInfo_triangle(cmesh.mesh);

cpdef void allocateGeometricInfo_triangle(CMesh cmesh):
    cppm.allocateGeometricInfo_triangle(cmesh.mesh);

cpdef void generateEdgeMeshFromRectangularGrid(int nx,
                                               double Lx,
                                               CMesh cmesh):
    cppm.edgeMeshElements(nx,cmesh.mesh);
    cppm.regularEdgeMeshNodes(nx,Lx,cmesh.mesh);
    cppm.constructElementBoundaryElementsArray_edge(cmesh.mesh);

cpdef void computeGeometricInfo_edge(CMesh cmesh):
    cppm.computeGeometricInfo_edge(cmesh.mesh);

cpdef void allocateGeometricInfo_edge(CMesh cmesh):
    cppm.allocateGeometricInfo_edge(cmesh.mesh);


cpdef void computeGeometricInfo_hexahedron(CMesh cmesh):
    cppm.computeGeometricInfo_hexahedron(cmesh.mesh);

cpdef void computeGeometricInfo_quadrilateral(CMesh cmesh):
    cppm.computeGeometricInfo_quadrilateral(cmesh.mesh);

cpdef void allocateGeometricInfo_hexahedron(CMesh cmesh):
    cppm.allocateGeometricInfo_hexahedron(cmesh.mesh);

cpdef void allocateGeometricInfo_quadrilateral(CMesh cmesh):
    cppm.allocateGeometricInfo_quadrilateral(cmesh.mesh);

cpdef void computeGeometricInfo_NURBS(CMesh cmesh):
    cppm.computeGeometricInfo_NURBS(cmesh.mesh);

cpdef void allocateGeometricInfo_NURBS(CMesh cmesh):
    cppm.allocateGeometricInfo_NURBS(cmesh.mesh);

def locallyRefineMultilevelMesh(int nSpace,
                                          CMultilevelMesh cmultilevelMesh,
                                          np.ndarray elementTagArray,
                                          int refineTypeFlag=0):
    cdef int failed,finestLevel
    if nSpace == 1:
        failed = cppm.locallyRefineEdgeMesh(cmultilevelMesh.multilevelMesh,
                                            <int*>(elementTagArray.data))
        finestLevel = cmultilevelMesh.multilevelMesh.nLevels
        cppm.constructElementBoundaryElementsArray_edge(cmultilevelMesh.multilevelMesh.meshArray[finestLevel-1])
        cppm.allocateGeometricInfo_edge(cmultilevelMesh.multilevelMesh.meshArray[finestLevel-1])
        cppm.computeGeometricInfo_edge(cmultilevelMesh.multilevelMesh.meshArray[finestLevel-1])
        if finestLevel > 1:
            cppm.assignElementBoundaryMaterialTypesFromParent(cmultilevelMesh.multilevelMesh.meshArray[finestLevel-2],
                                                              cmultilevelMesh.multilevelMesh.meshArray[finestLevel-1],
                                                              cmultilevelMesh.multilevelMesh.elementParentsArray[finestLevel-1],
                                                              1)
    elif nSpace == 2:
        if refineTypeFlag==1:
            failed = cppm.locallyRefineTriangleMesh_4T(cmultilevelMesh.multilevelMesh,
                                                       <int*>(elementTagArray.data))
        elif refineTypeFlag==2:
            failed = cppm.locallyRefineTriangleMesh_redGreen(cmultilevelMesh.multilevelMesh,<int*>(elementTagArray.data))
        else:
            failed = cppm.locallyRefineTriangleMesh(cmultilevelMesh.multilevelMesh,
                                                    <int*>(elementTagArray.data))
        finestLevel = cmultilevelMesh.multilevelMesh.nLevels
        cppm.constructElementBoundaryElementsArray_triangle(cmultilevelMesh.multilevelMesh.meshArray[finestLevel-1])
        cppm.allocateGeometricInfo_triangle(cmultilevelMesh.multilevelMesh.meshArray[finestLevel-1])
        cppm.computeGeometricInfo_triangle(cmultilevelMesh.multilevelMesh.meshArray[finestLevel-1])
        if finestLevel > 1:
            cppm.assignElementBoundaryMaterialTypesFromParent(cmultilevelMesh.multilevelMesh.meshArray[finestLevel-2],
                                                              cmultilevelMesh.multilevelMesh.meshArray[finestLevel-1],
                                                              cmultilevelMesh.multilevelMesh.elementParentsArray[finestLevel-1],
                                                              2)
    else:
        print("locallyRefine nSpace= {0:d} not implemented! Returning.".format(nSpace))

def setNewestNodeBases(int nSpace, CMultilevelMesh cmultilevelMesh):
    if nSpace == 2:
        failed = cppm.setNewestNodeBasesToLongestEdge(cmultilevelMesh.multilevelMesh)
    else:
        print("setNewestNodeBases= {0:d} not implemented! Returning".format(nSpace))
