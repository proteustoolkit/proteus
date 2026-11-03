# A type of -*- python -*- file
# cython: language_level = 3
from proteus import Comm as proteus_Comm
import numpy as np
cimport numpy as np
cdef extern from *:
    """
    #include <mpi.h>
    
    #if (MPI_VERSION < 3) && !defined(PyMPI_HAVE_MPI_Message)
    typedef void *PyMPI_MPI_Message;
    #define MPI_Message PyMPI_MPI_Message
    #endif
    
    #if (MPI_VERSION < 4) && !defined(PyMPI_HAVE_MPI_Session)
    typedef void *PyMPI_MPI_Session;
    #define MPI_Session PyMPI_MPI_Session
    #endif"
    """

from mpi4py.MPI cimport (Comm,
                         Op)
from libcpp.vector cimport vector
from mpi4py.libmpi cimport MPI_Comm
cimport mesh
from proteus cimport cmeshTools
from proteus.cmeshTools import INDEX_DTYPE
from proteus.partitioning cimport (c_petscPartitioners,
                                   c_ensurePetscInitialized,
                                   c_petscIsInitialized,
                                   c_generateCartesianSubdomain,
                                   c_cartesianNumbering,
                                   c_cartesianGlobalCounts,
                                   c_cartesianVertexInterpolation,
                                   c_cartesianSubdomainParents,
                                   c_partitionElements,
                                   c_partitionElementsLocal,
                                   c_partitionNodes,
                                   c_partitionNodesLocal,
                                   c_nodeGraphRows,
                                   c_elementGraphRows,
                                   c_elementOwnersFromNodeOwners,
                                   c_elementBoundaryOwnersFromNodeOwners,
                                   c_edgeOwnersFromNodeOwners,
                                   c_partitionNodesFromTetgenFiles,
                                   c_partitionNodesFromTriangleFiles,
                                   buildQuadraticSubdomain2GlobalMappings_1d,
                                   buildQuadraticSubdomain2GlobalMappings_2d,
                                   buildQuadraticSubdomain2GlobalMappings_3d,
                                   buildQuadraticCubeSubdomain2GlobalMappings_3d,
                                   buildDiscontinuousGalerkinSubdomain2GlobalMappings)

def petscPartitioners():
    """The graph partitioners PETSc was built with ('parmetis', 'chaco',
    'party', 'ptscotch'), best first. Empty means PETSc falls back to
    'current', which keeps the initial partition."""
    return c_petscPartitioners().decode().split()

cdef object _indexArray(mesh.proteus_index_t* p, mesh.proteus_index_t n):
    """A numpy view of n mesh indices at p; a rank that owns nothing has empty
    arrays, which a typed memoryview cannot describe."""
    if n == 0:
        return np.empty(0, dtype=INDEX_DTYPE)
    return np.asarray(<mesh.proteus_index_t[:n]> p)

cdef _requireGlobalMesh(cmeshTools.CMesh cmesh, object caller):
    """A global mesh with nothing in it is not a partitioning problem but a mesh
    that was never built: the Triangle and TetGen readers report a file they
    cannot open and return an empty mesh, which happens when the mesher did not
    run (an environment whose bin/ is not on PATH has no ``triangle``). Say so
    here instead of failing on the first owner of no nodes (IndexError)."""
    if cmesh.mesh.nNodes_global == 0 or cmesh.mesh.nElements_global == 0:
        raise ValueError("%s: the global mesh is empty (%d nodes, %d elements). Was it generated? "
                         "A mesh read from files the mesher never wrote is empty."
                         % (caller, cmesh.mesh.nNodes_global, cmesh.mesh.nElements_global))

def partitionElements(Comm comm, int nLayersOfOverlap, cmeshTools.CMesh cmesh, cmeshTools.CMesh subdomain_cmesh,
                      object elementOwners=None):
    """Partition by elements. elementOwners, if given, is an int array with the
    rank that owns each element (in the mesh's current numbering) and replaces
    the graph partitioner -- see MultilevelMesh.partitionLevels."""
    _requireGlobalMesh(cmesh, "partitionElements")
    #only the graph partitioner needs the PETSc C library, which petsc4py would
    #have initialized; given the owners this partitions without PETSc
    if elementOwners is None:
        c_ensurePetscInitialized()
    cmesh.mesh.subdomainp = &subdomain_cmesh.mesh
    cdef int[::1] owners
    cdef const int* ownersp = NULL
    if elementOwners is not None:
        owners = np.ascontiguousarray(elementOwners, dtype=np.intc)
        assert owners.shape[0] == cmesh.mesh.nElements_global, (owners.shape[0], cmesh.mesh.nElements_global)
        ownersp = &owners[0]
    c_partitionElements(comm.ob_mpi,
                        cmesh.mesh,
                        nLayersOfOverlap,
                        ownersp)
    return (
        _indexArray(cmesh.mesh.elementOffsets_subdomain_owned, comm.size+1),
        _indexArray(cmesh.mesh.elementNumbering_subdomain2global, cmesh.mesh.subdomainp.nElements_global),
        _indexArray(cmesh.mesh.nodeOffsets_subdomain_owned, comm.size+1),
        _indexArray(cmesh.mesh.nodeNumbering_subdomain2global, cmesh.mesh.subdomainp.nNodes_global),
        _indexArray(cmesh.mesh.elementBoundaryOffsets_subdomain_owned, comm.size+1),
        _indexArray(cmesh.mesh.elementBoundaryNumbering_subdomain2global, cmesh.mesh.subdomainp.nElementBoundaries_global),
        _indexArray(cmesh.mesh.edgeOffsets_subdomain_owned, comm.size+1),
        _indexArray(cmesh.mesh.edgeNumbering_subdomain2global, cmesh.mesh.subdomainp.nEdges_global)
    )

def partitionNodes(Comm comm, int nLayersOfOverlap, cmeshTools.CMesh cmesh, cmeshTools.CMesh subdomain_cmesh,
                   object nodeOwners=None, object nodeWeights=None):
    """Partition by nodes. nodeOwners, if given, is an int array with the rank
    that owns each node (in the mesh's current numbering) and replaces the
    graph partitioner -- see MultilevelMesh.partitionLevels."""
    _requireGlobalMesh(cmesh, "partitionNodes")
    #only the graph partitioner needs the PETSc C library, which petsc4py would
    #have initialized; given the owners this partitions without PETSc
    if nodeOwners is None:
        c_ensurePetscInitialized()
    cmesh.mesh.subdomainp = &subdomain_cmesh.mesh
    cdef int[::1] owners
    cdef const int* ownersp = NULL
    if nodeOwners is not None:
        owners = np.ascontiguousarray(nodeOwners, dtype=np.intc)
        assert owners.shape[0] == cmesh.mesh.nNodes_global, (owners.shape[0], cmesh.mesh.nNodes_global)
        ownersp = &owners[0]
    cdef int[::1] weights
    cdef const int* weightsp = NULL
    if nodeWeights is not None:
        assert nodeOwners is None, "node weights steer the graph partitioner; prescribed owners replace it"
        weights = np.ascontiguousarray(nodeWeights, dtype=np.intc)
        assert weights.shape[0] == cmesh.mesh.nNodes_global, (weights.shape[0], cmesh.mesh.nNodes_global)
        weightsp = &weights[0]
    c_partitionNodes(comm.ob_mpi,
                     cmesh.mesh,
                     nLayersOfOverlap,
                     ownersp,
                     weightsp)
    return (
        _indexArray(cmesh.mesh.elementOffsets_subdomain_owned, comm.size+1),
        _indexArray(cmesh.mesh.elementNumbering_subdomain2global, cmesh.mesh.subdomainp.nElements_global),
        _indexArray(cmesh.mesh.nodeOffsets_subdomain_owned, comm.size+1),
        _indexArray(cmesh.mesh.nodeNumbering_subdomain2global, cmesh.mesh.subdomainp.nNodes_global),
        _indexArray(cmesh.mesh.elementBoundaryOffsets_subdomain_owned, comm.size+1),
        _indexArray(cmesh.mesh.elementBoundaryNumbering_subdomain2global, cmesh.mesh.subdomainp.nElementBoundaries_global),
        _indexArray(cmesh.mesh.edgeOffsets_subdomain_owned, comm.size+1),
        _indexArray(cmesh.mesh.edgeNumbering_subdomain2global, cmesh.mesh.subdomainp.nEdges_global)
    )

def partitionNodesLocal(int rank, int size, int nLayersOfOverlap, cmeshTools.CMesh cmesh,
                        cmeshTools.CMesh subdomain_cmesh, object nodeOwners):
    """Rank rank's part of the partition of cmesh among size ranks, given the
    owner of every node: what partitionNodes computes on that rank once the
    owners are known, without MPI. One process can compute every rank's
    subdomain this way (a fresh cmesh per rank: the maps are written to it).
    Returns the arrays partitionNodes returns."""
    _requireGlobalMesh(cmesh, "partitionNodesLocal")
    owners = np.ascontiguousarray(nodeOwners, dtype=np.intc)
    assert owners.shape[0] == cmesh.mesh.nNodes_global, (owners.shape[0], cmesh.mesh.nNodes_global)
    assert owners.min() >= 0 and owners.max() < size, (owners.min(), owners.max(), size)
    cdef vector[int] cowners = owners
    cmesh.mesh.subdomainp = &subdomain_cmesh.mesh
    if c_partitionNodesLocal(cmesh.mesh, rank, size, nLayersOfOverlap, cowners) != 0:
        raise RuntimeError("partitionNodesLocal: an element or element boundary is reached by no rank")
    return (
        _indexArray(cmesh.mesh.elementOffsets_subdomain_owned, size+1),
        _indexArray(cmesh.mesh.elementNumbering_subdomain2global, cmesh.mesh.subdomainp.nElements_global),
        _indexArray(cmesh.mesh.nodeOffsets_subdomain_owned, size+1),
        _indexArray(cmesh.mesh.nodeNumbering_subdomain2global, cmesh.mesh.subdomainp.nNodes_global),
        _indexArray(cmesh.mesh.elementBoundaryOffsets_subdomain_owned, size+1),
        _indexArray(cmesh.mesh.elementBoundaryNumbering_subdomain2global, cmesh.mesh.subdomainp.nElementBoundaries_global),
        _indexArray(cmesh.mesh.edgeOffsets_subdomain_owned, size+1),
        _indexArray(cmesh.mesh.edgeNumbering_subdomain2global, cmesh.mesh.subdomainp.nEdges_global)
    )

def partitionElementsLocal(int rank, int size, int nLayersOfOverlap, cmeshTools.CMesh cmesh,
                           cmeshTools.CMesh subdomain_cmesh, object elementOwners):
    """Rank rank's part of the element partition of cmesh among size ranks,
    given the owner of every element: what partitionElements computes on that
    rank once the owners are known, without MPI. It renumbers cmesh in place,
    so use a fresh cmesh for each rank. Returns the arrays partitionElements
    returns."""
    _requireGlobalMesh(cmesh, "partitionElementsLocal")
    owners = np.ascontiguousarray(elementOwners, dtype=np.intc)
    assert owners.shape[0] == cmesh.mesh.nElements_global, (owners.shape[0], cmesh.mesh.nElements_global)
    assert owners.min() >= 0 and owners.max() < size, (owners.min(), owners.max(), size)
    cdef vector[int] cowners = owners
    cmesh.mesh.subdomainp = &subdomain_cmesh.mesh
    if c_partitionElementsLocal(cmesh.mesh, rank, size, nLayersOfOverlap, cowners) != 0:
        raise RuntimeError("partitionElementsLocal: a node, element boundary or edge is reached by no rank")
    return (
        _indexArray(cmesh.mesh.elementOffsets_subdomain_owned, size+1),
        _indexArray(cmesh.mesh.elementNumbering_subdomain2global, cmesh.mesh.subdomainp.nElements_global),
        _indexArray(cmesh.mesh.nodeOffsets_subdomain_owned, size+1),
        _indexArray(cmesh.mesh.nodeNumbering_subdomain2global, cmesh.mesh.subdomainp.nNodes_global),
        _indexArray(cmesh.mesh.elementBoundaryOffsets_subdomain_owned, size+1),
        _indexArray(cmesh.mesh.elementBoundaryNumbering_subdomain2global, cmesh.mesh.subdomainp.nElementBoundaries_global),
        _indexArray(cmesh.mesh.edgeOffsets_subdomain_owned, size+1),
        _indexArray(cmesh.mesh.edgeNumbering_subdomain2global, cmesh.mesh.subdomainp.nEdges_global)
    )

def nodeGraphRows(cmeshTools.CMesh cmesh, int begin, int end):
    """Rows begin..end-1 of the node graph partitionNodes hands the graph
    partitioner, in CSR form (offsets, neighbors): each node's star, sorted."""
    _requireGlobalMesh(cmesh, "nodeGraphRows")
    assert 0 <= begin <= end <= cmesh.mesh.nNodes_global, (begin, end)
    cdef vector[int] offsets
    cdef vector[int] neighbors
    c_nodeGraphRows(cmesh.mesh, begin, end, offsets, neighbors)
    return np.array(offsets, dtype=np.intc), np.array(neighbors, dtype=np.intc)

def elementGraphRows(cmeshTools.CMesh cmesh, int begin, int end):
    """Rows begin..end-1 of the element graph partitionElements hands the graph
    partitioner, in CSR form (offsets, neighbors): each element's neighbours
    across its element boundaries, sorted."""
    _requireGlobalMesh(cmesh, "elementGraphRows")
    assert 0 <= begin <= end <= cmesh.mesh.nElements_global, (begin, end)
    cdef vector[int] offsets
    cdef vector[int] neighbors
    c_elementGraphRows(cmesh.mesh, begin, end, offsets, neighbors)
    return np.array(offsets, dtype=np.intc), np.array(neighbors, dtype=np.intc)

def ownersFromNodeOwners(cmeshTools.CMesh cmesh, object nodeOwners, int size):
    """partitionNodes' ownership rules on cmesh, given the owner of every node:
    the owners of the elements, element boundaries and edges, and the node
    old-to-new numbering the edge rule uses. No MPI. An entity no rank reaches
    has the owner size."""
    _requireGlobalMesh(cmesh, "ownersFromNodeOwners")
    owners = np.ascontiguousarray(nodeOwners, dtype=np.intc)
    assert owners.shape[0] == cmesh.mesh.nNodes_global, (owners.shape[0], cmesh.mesh.nNodes_global)
    cdef vector[int] cowners = owners
    #each rank's owned nodes contiguous, in their old order (numberByOwner)
    old2new = np.empty(owners.shape[0], dtype=np.intc)
    old2new[np.argsort(owners, kind='stable')] = np.arange(owners.shape[0], dtype=np.intc)
    cdef vector[int] cold2new = old2new
    return (np.array(c_elementOwnersFromNodeOwners(cmesh.mesh, cowners, size), dtype=np.intc),
            np.array(c_elementBoundaryOwnersFromNodeOwners(cmesh.mesh, cowners, size), dtype=np.intc),
            np.array(c_edgeOwnersFromNodeOwners(cmesh.mesh, cowners, cold2new), dtype=np.intc),
            old2new)

CARTESIAN_KINDS = {'quad': 0, 'hex': 1, 'tri': 2, 'tet': 3}

def _grid3(object values, object fill):
    return [values[d] if d < len(values) else fill for d in range(3)]

def generateCartesianSubdomain(Comm comm, object kind, int triangleFlag, object nn, object L, object rankGrid,
                               int nLayersOfOverlap, cmeshTools.CMesh cmesh, cmeshTools.CMesh subdomain_cmesh,
                               int level=0, object origin=(0.0, 0.0, 0.0)):
    """This rank's subdomain of a regular grid of kind 'quad', 'hex', 'tri'
    (split by triangleFlag) or 'tet' with nn nodes per axis on origin + [0,L],
    generated directly for the rank grid rankGrid (px, py, pz): what
    partitionNodes gives for the global grid and MeshTools.cartesianNodeOwners,
    without the global mesh. level > 0 is that grid refined level times, cut
    into the level-0 bricks scaled. Fills subdomain_cmesh; cmesh keeps only the
    global counts. Returns the arrays partitionNodes returns."""
    cdef int[3] cnn
    cdef int[3] cgrid
    cdef double[3] cL
    cdef double[3] cx0
    for d in range(3):
        cnn[d] = _grid3(nn, 1)[d]
        cL[d] = _grid3(L, 1.0)[d]
        cgrid[d] = _grid3(rankGrid, 1)[d]
        cx0[d] = _grid3(origin, 0.0)[d]
    cmesh.mesh.subdomainp = &subdomain_cmesh.mesh
    c_generateCartesianSubdomain(comm.ob_mpi, CARTESIAN_KINDS[kind], triangleFlag, cnn, cL, cgrid,
                                 nLayersOfOverlap, cmesh.mesh, level, cx0)
    return (
        _indexArray(cmesh.mesh.elementOffsets_subdomain_owned, comm.size+1),
        _indexArray(cmesh.mesh.elementNumbering_subdomain2global, cmesh.mesh.subdomainp.nElements_global),
        _indexArray(cmesh.mesh.nodeOffsets_subdomain_owned, comm.size+1),
        _indexArray(cmesh.mesh.nodeNumbering_subdomain2global, cmesh.mesh.subdomainp.nNodes_global),
        _indexArray(cmesh.mesh.elementBoundaryOffsets_subdomain_owned, comm.size+1),
        _indexArray(cmesh.mesh.elementBoundaryNumbering_subdomain2global, cmesh.mesh.subdomainp.nElementBoundaries_global),
        _indexArray(cmesh.mesh.edgeOffsets_subdomain_owned, comm.size+1),
        _indexArray(cmesh.mesh.edgeNumbering_subdomain2global, cmesh.mesh.subdomainp.nEdges_global)
    )

def cartesianNumbering(object kind, int triangleFlag, object nn, object rankGrid, bint elements, object oldNumbers,
                       int level=0):
    """The new numbers of nodes (elements=True: elements) with these old
    numbers, in the partition generateCartesianSubdomain generates: the
    permutation partitionNodes keeps as nodeNumbering_global_old2new (or
    elementNumbering_global_old2new), as a function. Elements that need not
    contain their cell's lowest corner (triangleFlag 1 or 2) are counted over
    the whole grid: fine for checking, not for production runs."""
    cdef int[3] cnn
    cdef int[3] cgrid
    for d in range(3):
        cnn[d] = _grid3(nn, 1)[d]
        cgrid[d] = _grid3(rankGrid, 1)[d]
    cdef int[::1] old = np.ascontiguousarray(oldNumbers, dtype=np.intc)
    new = np.empty(old.shape[0], dtype=np.intc)
    cdef int[::1] newView = new
    if old.shape[0] > 0:
        c_cartesianNumbering(CARTESIAN_KINDS[kind], triangleFlag, cnn, cgrid, level, elements,
                             old.shape[0], &old[0], &newView[0])
    return new

def cartesianGlobalCounts(object kind, int triangleFlag, object nn, int level=0):
    """The numbers of nodes, elements, element boundaries and edges of the
    global grid generateCartesianSubdomain partitions."""
    cdef int[3] cnn
    cdef int[4] counts
    for d in range(3):
        cnn[d] = _grid3(nn, 1)[d]
    c_cartesianGlobalCounts(CARTESIAN_KINDS[kind], triangleFlag, cnn, level, counts)
    return tuple(counts[i] for i in range(4))

def cartesianVertexInterpolation(object kind, int triangleFlag, object nn, object rankGrid, int level, int rank):
    """The rows rank owns of the vertex prolongation from level-1 to level of a
    generated hierarchy, as (rowptr, colind, nzval), columns in level-1's
    partitioned numbering."""
    cdef int[3] cnn
    cdef int[3] cgrid
    cdef vector[int] rowptr, colind
    cdef vector[double] nzval
    for d in range(3):
        cnn[d] = _grid3(nn, 1)[d]
        cgrid[d] = _grid3(rankGrid, 1)[d]
    c_cartesianVertexInterpolation(CARTESIAN_KINDS[kind], triangleFlag, cnn, cgrid, level, rank, rowptr, colind, nzval)
    return (np.array(rowptr, dtype=np.intc), np.array(colind, dtype=np.intc), np.array(nzval, dtype=np.float64))

def cartesianSubdomainParents(object kind, int triangleFlag, object nn, object rankGrid, int level, int rank):
    """For each element of rank's subdomain on level of a generated hierarchy,
    its parent's index in rank's subdomain on level-1, or -1."""
    cdef int[3] cnn
    cdef int[3] cgrid
    cdef vector[int] parents
    for d in range(3):
        cnn[d] = _grid3(nn, 1)[d]
        cgrid[d] = _grid3(rankGrid, 1)[d]
    c_cartesianSubdomainParents(CARTESIAN_KINDS[kind], triangleFlag, cnn, cgrid, level, rank, parents)
    return np.array(parents, dtype=np.intc)

def petscInitialized():
    """Whether the PETSc C library has been initialized in this process.

    Partitioning with prescribed owners does not need PETSc; a run with
    petsc4py blocked that partitions only that way should leave this False.
    """
    return c_petscIsInitialized()

def getNodeNumbering_global_old2new(cmeshTools.CMesh cmesh):
    """The permutation partitioning applied to the global node numbering.

    Partitioning renumbers the nodes so that each rank's owned nodes are
    contiguous, and nodeNumbering_subdomain2global refers to that new
    numbering. Arrays built against the mesh's original numbering -- the
    parent/child relations of a multilevel hierarchy, for instance -- have to
    be read through this map before they can be used alongside the partitioned
    vectors.

    Returns None if the mesh has not been partitioned.
    """
    if cmesh.mesh.nodeNumbering_global_old2new == NULL:
        return None
    return _indexArray(cmesh.mesh.nodeNumbering_global_old2new, cmesh.mesh.nNodes_global)

def getElementNumbering_global_old2new(cmeshTools.CMesh cmesh):
    """The permutation partitioning applied to the global element numbering
    (see getNodeNumbering_global_old2new). None if the mesh has not been
    partitioned by this code."""
    if cmesh.mesh.elementNumbering_global_old2new == NULL:
        return None
    return _indexArray(cmesh.mesh.elementNumbering_global_old2new, cmesh.mesh.nElements_global)

def convertPUMIPartitionToPython(Comm comm, cmeshTools.CMesh cmesh, cmeshTools.CMesh subdomain_cmesh):
#need to have Python handles for the cmesh arrays created from apf/PUMI mesh 
    cmesh.mesh.subdomainp = &subdomain_cmesh.mesh
    return (
        _indexArray(cmesh.mesh.elementOffsets_subdomain_owned, comm.size+1),
        _indexArray(cmesh.mesh.elementNumbering_subdomain2global, cmesh.mesh.subdomainp.nElements_global),
        _indexArray(cmesh.mesh.nodeOffsets_subdomain_owned, comm.size+1),
        _indexArray(cmesh.mesh.nodeNumbering_subdomain2global, cmesh.mesh.subdomainp.nNodes_global),
        _indexArray(cmesh.mesh.elementBoundaryOffsets_subdomain_owned, comm.size+1),
        _indexArray(cmesh.mesh.elementBoundaryNumbering_subdomain2global, cmesh.mesh.subdomainp.nElementBoundaries_global),
        _indexArray(cmesh.mesh.edgeOffsets_subdomain_owned, comm.size+1),
        _indexArray(cmesh.mesh.edgeNumbering_subdomain2global, cmesh.mesh.subdomainp.nEdges_global)
    )

def partitionNodesFromTetgenFiles(Comm comm, object filebase, int indexBase, int nLayersOfOverlap, cmeshTools.CMesh cmesh, cmeshTools.CMesh subdomain_cmesh, double memHardLimit):
    #the partitioners need the PETSc C library, which petsc4py would have initialized
    c_ensurePetscInitialized()
    cmesh.mesh.subdomainp = &subdomain_cmesh.mesh
    if not isinstance(filebase, bytes):
        filebase = filebase.encode()
    c_partitionNodesFromTetgenFiles(comm.ob_mpi,
                                    <const char*>(<char*>filebase),
                                    indexBase,
                                    cmesh.mesh,
                                    nLayersOfOverlap,
                                    memHardLimit)
    return (
        _indexArray(cmesh.mesh.elementOffsets_subdomain_owned, comm.size+1),
        _indexArray(cmesh.mesh.elementNumbering_subdomain2global, cmesh.mesh.subdomainp.nElements_global),
        _indexArray(cmesh.mesh.nodeOffsets_subdomain_owned, comm.size+1),
        _indexArray(cmesh.mesh.nodeNumbering_subdomain2global, cmesh.mesh.subdomainp.nNodes_global),
        _indexArray(cmesh.mesh.elementBoundaryOffsets_subdomain_owned, comm.size+1),
        _indexArray(cmesh.mesh.elementBoundaryNumbering_subdomain2global, cmesh.mesh.subdomainp.nElementBoundaries_global),
        _indexArray(cmesh.mesh.edgeOffsets_subdomain_owned, comm.size+1),
        _indexArray(cmesh.mesh.edgeNumbering_subdomain2global, cmesh.mesh.subdomainp.nEdges_global)
    )

def partitionNodesFromTriangleFiles(Comm comm, object filebase, int indexBase, int nLayersOfOverlap, cmeshTools.CMesh cmesh, cmeshTools.CMesh subdomain_cmesh):
    #the partitioners need the PETSc C library, which petsc4py would have initialized
    c_ensurePetscInitialized()
    cmesh.mesh.subdomainp = &subdomain_cmesh.mesh
    if not isinstance(filebase, bytes):
        filebase = filebase.encode()
    c_partitionNodesFromTriangleFiles(comm.ob_mpi,
                                    <const char*>(<char*>filebase),
                                    indexBase,
                                    cmesh.mesh,
                                    nLayersOfOverlap)
    return (
        _indexArray(cmesh.mesh.elementOffsets_subdomain_owned, comm.size+1),
        _indexArray(cmesh.mesh.elementNumbering_subdomain2global, cmesh.mesh.subdomainp.nElements_global),
        _indexArray(cmesh.mesh.nodeOffsets_subdomain_owned, comm.size+1),
        _indexArray(cmesh.mesh.nodeNumbering_subdomain2global, cmesh.mesh.subdomainp.nNodes_global),
        _indexArray(cmesh.mesh.elementBoundaryOffsets_subdomain_owned, comm.size+1),
        _indexArray(cmesh.mesh.elementBoundaryNumbering_subdomain2global, cmesh.mesh.subdomainp.nElementBoundaries_global),
        _indexArray(cmesh.mesh.edgeOffsets_subdomain_owned, comm.size+1),
        _indexArray(cmesh.mesh.edgeNumbering_subdomain2global, cmesh.mesh.subdomainp.nEdges_global)
    )

def buildQuadraticLocal2GlobalMappings(Comm comm,
                                       int nSpace,
                                       cmeshTools.CMesh cmesh,
                                       cmeshTools.CMesh subdomain_cmesh,
                                       np.ndarray elementOffsets_subdomain_owned,
                                       np.ndarray nodeOffsets_subdomain_owned,
                                       np.ndarray elementBoundaryOffsets_subdomain_owned,
                                       np.ndarray edgeOffsets_subdomain_owned,
                                       np.ndarray elementNumbering_subdomain2global,
                                       np.ndarray nodeNumbering_subdomain2global,
                                       np.ndarray elementBoundaryNumbering_subdomain2global,
                                       np.ndarray edgeNumbering_subdomain2global,
                                       np.ndarray quadratic_dof_offsets_subdomain_owned,
                                       np.ndarray quadratic_subdomain_l2g,
                                       np.ndarray quadraticNumbering_subdomain2global,
                                       np.ndarray quadratic_lagrangeNodes):
    cdef int nDOF_all_processes=0
    cdef int nDOF_subdomain=0
    cdef int max_dof_neighbors=0
    if nSpace == 1:
        buildQuadraticSubdomain2GlobalMappings_1d(comm.ob_mpi,
                                                  cmesh.mesh,
                                                  <int*>(elementOffsets_subdomain_owned.data),
                                                  <int*>(nodeOffsets_subdomain_owned.data),
                                                  <int*>(elementNumbering_subdomain2global.data),
                                                  <int*>(nodeNumbering_subdomain2global.data),
                                                  nDOF_all_processes,
                                                  nDOF_subdomain,
                                                  max_dof_neighbors,
                                                  <int*>(quadratic_dof_offsets_subdomain_owned.data),
                                                  <int*>(quadratic_subdomain_l2g.data),
                                                  <int*>(quadraticNumbering_subdomain2global.data),
                                                  <double*>(quadratic_lagrangeNodes.data));
    elif nSpace == 2:
        buildQuadraticSubdomain2GlobalMappings_2d(comm.ob_mpi,
                                                  cmesh.mesh,
                                                  <int*>(elementBoundaryOffsets_subdomain_owned.data),
                                                  <int*>(nodeOffsets_subdomain_owned.data),
                                                  <int*>(elementBoundaryNumbering_subdomain2global.data),
                                                  <int*>(nodeNumbering_subdomain2global.data),
                                                  nDOF_all_processes,
                                                  nDOF_subdomain,
                                                  max_dof_neighbors,
                                                  <int*>(quadratic_dof_offsets_subdomain_owned.data),
                                                  <int*>(quadratic_subdomain_l2g.data),
                                                  <int*>(quadraticNumbering_subdomain2global.data),
                                                  <double*>(quadratic_lagrangeNodes.data))
    else:
        buildQuadraticSubdomain2GlobalMappings_3d(comm.ob_mpi,
                                                  cmesh.mesh,
                                                  <int*>(edgeOffsets_subdomain_owned.data),
                                                  <int*>(nodeOffsets_subdomain_owned.data),
                                                  <int*>(edgeNumbering_subdomain2global.data),
                                                  <int*>(nodeNumbering_subdomain2global.data),
                                                  nDOF_all_processes,
                                                  nDOF_subdomain,
                                                  max_dof_neighbors,
                                                  <int*>(quadratic_dof_offsets_subdomain_owned.data),
                                                  <int*>(quadratic_subdomain_l2g.data),
                                                  <int*>(quadraticNumbering_subdomain2global.data),
                                                  <double*>(quadratic_lagrangeNodes.data))
    return (nDOF_all_processes,
            nDOF_subdomain,
            max_dof_neighbors)
    
def buildQuadraticCubeLocal2GlobalMappings(Comm comm,
                                           int nSpace,
                                           cmeshTools.CMesh cmesh,
                                           cmeshTools.CMesh subdomain_cmesh,
                                           np.ndarray elementOffsets_subdomain_owned,
                                           np.ndarray nodeOffsets_subdomain_owned,
                                           np.ndarray elementBoundaryOffsets_subdomain_owned,
                                           np.ndarray edgeOffsets_subdomain_owned,
                                           np.ndarray elementNumbering_subdomain2global,
                                           np.ndarray nodeNumbering_subdomain2global,
                                           np.ndarray elementBoundaryNumbering_subdomain2global,
                                           np.ndarray edgeNumbering_subdomain2global,
                                           np.ndarray quadratic_dof_offsets_subdomain_owned,
                                           np.ndarray quadratic_subdomain_l2g,
                                           np.ndarray quadraticNumbering_subdomain2global,
                                           np.ndarray quadratic_lagrangeNodes):
    cdef int nDOF_all_processes=0
    cdef int nDOF_subdomain=0
    cdef int max_dof_neighbors=0
    if nSpace == 1:
        assert(False),"buildQuadraticCubeSubdomain2GlobalMappings_1d not implemented!!"
    elif nSpace == 2:
        assert(False),"buildQuadraticCubeSubdomain2GlobalMappings_2d not implemented!!"
    else:
        buildQuadraticCubeSubdomain2GlobalMappings_3d(comm.ob_mpi,
                                                      cmesh.mesh,
                                                      <int*>(edgeOffsets_subdomain_owned.data),
                                                      <int*>(nodeOffsets_subdomain_owned.data),
                                                      <int*>(edgeNumbering_subdomain2global.data),
                                                      <int*>(nodeNumbering_subdomain2global.data),
                                                      nDOF_all_processes,
                                                      nDOF_subdomain,
                                                      max_dof_neighbors,
                                                      <int*>(quadratic_dof_offsets_subdomain_owned.data),
                                                      <int*>(quadratic_subdomain_l2g.data),
                                                      <int*>(quadraticNumbering_subdomain2global.data),
                                                      <double*>(quadratic_lagrangeNodes.data))
    return (nDOF_all_processes,
            nDOF_subdomain,
            max_dof_neighbors)

def buildDiscontinuousGalerkinLocal2GlobalMappings(Comm comm,
                                                   int nDOF_element,
                                                   cmeshTools.CMesh cmesh,
                                                   cmeshTools.CMesh subdomain_cmesh,
                                                   np.ndarray elementOffsets_subdomain_owned,
                                                   np.ndarray elementNumbering_subdomain2global,
                                                   np.ndarray dg_dof_offsets_subdomain_owned,
                                                   np.ndarray dg_subdomain_l2g,
                                                   np.ndarray dgNumbering_subdomain2global):
    cdef int nDOF_all_processes=0
    cdef int nDOF_subdomain=0
    cdef int max_dof_neighbors=0
    buildDiscontinuousGalerkinSubdomain2GlobalMappings(comm.ob_mpi,
                                                       cmesh.mesh,
                                                       <int*>(elementOffsets_subdomain_owned.data),
                                                       <int*>(elementNumbering_subdomain2global.data),
                                                       nDOF_element,
                                                       nDOF_all_processes,
                                                       nDOF_subdomain,
                                                       max_dof_neighbors,
                                                       <int*>(dg_dof_offsets_subdomain_owned.data),
                                                       <int*>(dg_subdomain_l2g.data),
                                                       <int*>(dgNumbering_subdomain2global.data))
    return (nDOF_all_processes,
            nDOF_subdomain,
            max_dof_neighbors)
