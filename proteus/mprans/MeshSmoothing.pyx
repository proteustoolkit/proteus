#!python
#cython: wraparound=True, boundscheck=False, initializedcheck=False, cdivision=True

import numpy as np
cimport numpy as np
from libcpp cimport bool
from libc.math cimport sin, cos, acos, exp, sqrt, fabs, M_PI, abs
from mpi4py import MPI
from proteus.Profiling import logEvent

# dtype and MPI datatype of proteus_index_t (index_type.h); counts and displacements stay int, as MPI requires
INDEX_DTYPE = np.dtype("i%d" % sizeof(proteus_index_t))
MPI_INDEX = MPI.INT if sizeof(proteus_index_t) == 4 else MPI.INT64_T

def smoothNodesLaplace(double[:,:] nodeArray_,
                       proteus_index_t[:] nodeStarOffsets,
                       proteus_index_t[:] nodeStarArray,
                       int[:] nodeMaterialTypes,
                       proteus_index_t nNodes_owned,
                       int nd,
                       bool simultaneous=False,
                       bool smoothBoundaries=True,
                       int[:] fixedNodesBoolArray=None,
                       double alpha=0.):
    """
    Laplace Smoothing:
    Mesh nodes are displaced to the centroid of neighbouring nodes

    Parameters
    ----------
    nodeArray_: double[:,:]
        array of mesh nodes
        (!) will be modified with smoothing
    nodeStarOffsets: int[:]
        array to get offsets for neighbouring node numbers
    nodeStarArray: int[:]
        array of neighbouring nodes
    nodeMaterialTypes: int[:]
        array to know which node is interior node (materialType=0)
    nNodes_owned: int
        number of nodes owned
    nd: int
        number of dimensions (needed to find direction to smooth boundaries)
    simultaneous: bool
        if True: simultaneous smoothing
        if False: sequential smoothing
    smoothBoundaries: bool
        if True: boundaries are smoothed (only with surrounding boundary nodes)
        if False: boudnary nodes are fixed
    fixedNodesBoolArray: int[:]
        array of same length as nodeArray with:
        fixedNodesBoolArray[node] = 1 -> node is fixed
        fixedNodesBoolArray[node] = 0 -> node is not fixed
    alpha: double
        penalty term related to original position of nodes (0 <= alpha <= 1)
    """
    cySmoothNodesLaplace(nodeArray_=nodeArray_,
                         nodeStarOffsets=nodeStarOffsets,
                         nodeStarArray=nodeStarArray,
                         nodeMaterialTypes=nodeMaterialTypes,
                         nNodes_owned=nNodes_owned,
                         nd=nd,
                         simultaneous=simultaneous,
                         smoothBoundaries=smoothBoundaries,
                         fixedNodesBoolArray=fixedNodesBoolArray,
                         alpha=alpha)

def smoothNodesCentroid(double[:,:] nodeArray_,
                        proteus_index_t[:] nodeElementOffsets,
                        proteus_index_t[:] nodeElementsArray,
                        int[:] nodeMaterialTypes,
                        double[:,:] elementBarycentersArray,
                        double[:] elementVolumesArray,
                        proteus_index_t[:,:] elementNodesArray,
                        proteus_index_t nNodes_owned,
                        bool simultaneous=True,
                        bool smoothBoundaries=True,
                        int[:] fixedNodesBoolArray=None,
                        double alpha=0.):
    """
    Centroid Smoothing:
    Mesh nodes are displaced to the centroid of the polygon/volume formed by
    neighbouring nodes

    Parameters
    ----------
    nodeArray_: double[:,:]
        array of mesh nodes
        (!) will be modified with smoothing
    nodeElementsOffsets: int[:]
        array to get offsets for neighbouring element numbers
    nodeElementsArray: int[:]
        array to get element number from nodeElementOffsets
    nodeMaterialTypes: int[:]
        array to know which node is interior node (materialType=0)
    elementBarycentersArray: double[:,:]
        barycenters of elements
    elementVolumesArray: double[:]
        volume of elements
    elementNodesArray: int[:,:]
        list of nodes per elements
    nNodes_owned: int
        number of nodes owned
    simultaneous: bool
        if True: simultaneous smoothing
        if False: sequential smoothing
                  (!) will update areas/volumes and barycenters in this case
    smoothBoundaries: bool
        if True: boundaries are smoothed (only with surrounding boundary nodes)
        if False: boudnary nodes are fixed
    fixedNodesBoolArray: int[:]
        array of same length as nodeArray with:
        fixedNodesBoolArray[node] = 1 -> node is fixed
        fixedNodesBoolArray[node] = 0 -> node is not fixed
    alpha: double
        penalty term related to original position of nodes (0 <= alpha <= 1)
    """
    cySmoothNodesCentroid(nodeArray_=nodeArray_,
                          nodeElementOffsets=nodeElementOffsets,
                          nodeElementsArray=nodeElementsArray,
                          nodeMaterialTypes=nodeMaterialTypes,
                          elementBarycentersArray=elementBarycentersArray,
                          elementVolumesArray=elementVolumesArray,
                          elementNodesArray=elementNodesArray,
                          nNodes_owned=nNodes_owned,
                          fixedNodesBoolArray=fixedNodesBoolArray,
                          simultaneous=simultaneous,
                          smoothBoundaries=smoothBoundaries,
                          alpha=alpha)

def updateDilationElements(double[:] elementDilationArray_,
                           double[:] elementVolumeArray,
                           double[:] elementVolumeTargetArray,
                           proteus_index_t nElements):
    cyUpdateDilationElements(elementDilationArray_=elementDilationArray_,
                             elementVolumeArray=elementVolumeArray,
                             elementVolumeTargetArray=elementVolumeTargetArray,
                             nElements=nElements)

def getDilationElements(double[:] elementVolumeArray,
                        double[:] elementVolumeTargetArray):
    cdef proteus_index_t nElements = len(elementVolumeArray)
    elementDilationArray_ = np.zeros(nElements)
    cyUpdateDilationElements(elementDilationArray_=elementDilationArray_,
                             elementVolumeArray=elementVolumeArray,
                             elementVolumeTargetArray=elementVolumeTargetArray,
                             nElements=nElements)
    return elementDilationArray_

def updateDistortionElements(double[:] elementDistortionArray_,
                             double[:,:,:,:] J_array,
                             double[:,:] detJ_array,
                             int nd,
                             proteus_index_t nElements):
    cyUpdateDistortionElements(elementDistortionArray_=elementDistortionArray_,
                               J_array=J_array,
                               detJ_array=detJ_array,
                               nd=nd,
                               nElements=nElements)

def getDistortionElements(double[:,:,:,:] J_array,
                          double[:,:] detJ_array,
                          int nd):
    nElements = len(detJ_array)
    elementDistortionArray_ = np.zeros(nElements)
    cyUpdateDistortionElements(elementDistortionArray_=elementDistortionArray_,
                               J_array=J_array,
                               detJ_array=detJ_array,
                               nd=nd,
                               nElements=nElements)
    return elementDistortionArray_

def updateInverseMeanRatioTriangleElements(double[:] IMRElementsArray_,
                                           double[:,:] nodeArray,
                                           proteus_index_t[:,:] elementNodesArray,
                                           proteus_index_t nElements):
    cyUpdateInverseMeanRatioTriangleElements(IMRElementsArray_=IMRElementsArray_,
                                             nodeArray=nodeArray,
                                             elementNodesArray=elementNodesArray,
                                             nElements=nElements)

def getInverseMeanRatioTriangleElements(double[:,:] nodeArray,
                                        proteus_index_t[:,:] elementNodesArray):
    cdef proteus_index_t nElements = len(elementNodesArray)
    IMRElementsArray_ = np.zeros(nElements)
    cyUpdateInverseMeanRatioTriangleElements(IMRElementsArray_=IMRElementsArray_,
                                             nodeArray=nodeArray,
                                             elementNodesArray=elementNodesArray,
                                             nElements=nElements)
    return IMRElementsArray_

def updateInverseMeanRatioTriangleNodes(double[:] IMRNodesArray_,
                                        double[:,:] nodeArray,
                                        proteus_index_t[:,:] elementNodesArray,
                                        proteus_index_t[:] nodeElementOffsets,
                                        proteus_index_t[:] nodeElementsArray,
                                        proteus_index_t nNodes,
                                        proteus_index_t nElements,
                                        bool el_average=False):
    cyUpdateInverseMeanRatioTriangleNodes(IMRNodesArray_=IMRNodesArray_,
                                          nodeArray=nodeArray,
                                          elementNodesArray=elementNodesArray,
                                          nodeElementOffsets=nodeElementOffsets,
                                          nodeElementsArray=nodeElementsArray,
                                          el_average=el_average,
                                          nNodes=nNodes,
                                          nElements=nElements)

def getInverseMeanRatioTriangleNodes(double[:,:] nodeArray,
                                     proteus_index_t[:,:] elementNodesArray,
                                     proteus_index_t[:] nodeElementOffsets,
                                     proteus_index_t[:] nodeElementsArray,
                                     bool el_average=False):
    cdef proteus_index_t nNodes = len(nodeArray)
    cdef proteus_index_t nElements = len(elementNodesArray)
    IMRNodesArray_ = np.zeros(nNodes)
    cyUpdateInverseMeanRatioTriangleNodes(IMRNodesArray_=IMRNodesArray_,
                                          nodeArray=nodeArray,
                                          elementNodesArray=elementNodesArray,
                                          nodeElementOffsets=nodeElementOffsets,
                                          nodeElementsArray=nodeElementsArray,
                                          el_average=el_average,
                                          nNodes=nNodes,
                                          nElements=nElements)
    return IMRNodesArray_

def getInverseMeanRatioSingleTriangle(proteus_index_t node0,
                                      double[:,:] nodeArray,
                                      proteus_index_t[:,:] elementNodesArray,
                                      proteus_index_t[:] nodeElementOffsets,
                                      proteus_index_t[:] nodeElementsArray):
    return cyGetInverseMeanRatioSingleTriangle(node0=node0,
                                               nodeArray=nodeArray,
                                               elementNodesArray=elementNodesArray,
                                               nodeElementOffsets=nodeElementOffsets,
                                               nodeElementsArray=nodeElementsArray)

def smoothNodesQuality(double[:] distortion,
                       double[:] dilation,
                       double[:,:] nodeArray,
                       proteus_index_t nNodes_owned,
                       int[:] nodeMaterialTypes,
                       proteus_index_t[:] nodeElementOffsets,
                       proteus_index_t[:] nodeElementsArray,
                       proteus_index_t[:,:] elementNodesArray,
                       bool apply_directly=False):
    assert 1>2, 'smoothNodesQuality is work in progress, do not use'
    return cySmoothNodesQuality(distortion=distortion,
                                dilation=dilation,
                                nodeArray=nodeArray,
                                nNodes_owned=nNodes_owned,
                                nodeMaterialTypes=nodeMaterialTypes,
                                nodeElementOffsets=nodeElementOffsets,
                                nodeElementsArray=nodeElementsArray,
                                elementNodesArray=elementNodesArray,
                                apply_directly=False)

def getLocalNearestNode(double[:] coords,
                        double[:,:] nodeArray,
                        proteus_index_t[:] nodeStarOffsets,
                        proteus_index_t[:] nodeStarArray,
                        proteus_index_t node):
    """Finds nearest node to coordinates (local)

    Parameters
    ----------
    coords: array_like
        coordinates from which to find nearest node
    nodeArray: array_like
        array of fluid mesh node coordinates
    nodeStarOffsets: array_like
        array of offsets from nodes (range)
    nodeStarArray: array_like
        array of neighbouring nodes
    node: int
        first guess for nearest node

    Returns
    -------
    node: int
        nearest node index
    """
    return pyxGetLocalNearestNode(coords,
                                  nodeArray,
                                  nodeStarOffsets,
                                  nodeStarArray,
                                  node)

def getLocalNearestElement(double[:] coords,
                           double[:,:] elementBarycentersArray,
                           proteus_index_t[:,:] elementNeighborsArray,
                           proteus_index_t eN):
    """Finds nearest element to coordinates (local)

    Parameters
    ----------
    coords: double[:]
        coordinates from which to find nearest element
    elementBarycentersArray: double[:,:]
        array of mesh cell barycenter coordinates
    elementNeighborsArray: double[:,:]
        array of element neighbors
    eN: int
        first guess for nearest element

    Returns
    -------
    eN: int
        nearest element index
    """
    return pyxGetLocalNearestElement(coords,
                                     elementBarycentersArray,
                                     elementNeighborsArray,
                                     eN)

def getLocalNearestElementAroundNode(double[:] coords,
                                     proteus_index_t[:] nodeElementOffsets,
                                     proteus_index_t[:] nodeElementsArray,
                                     double[:,:] elementBarycentersArray,
                                     proteus_index_t node):
    """Finds nearest neighbouring element of node to coordinates (local)

    Parameters
    ----------
    coords: double[:]
        coordinates from which to find nearest element
    nodeElementOffsets: int[:]
        element offsets from nodes
    nodeElementsArray: int[:]
        elements array from nodeElementOffsets
    elementBarycentersArray: int[:]
        array of mesh cell barycenter coordinates
    node: proteus_index_t
        node from which to search

    Returns
    -------
    eN: int
        nearest element index
    """
    return pyxGetLocalNearestElementAroundNode(coords=coords,
                                               nodeElementOffsets=nodeElementOffsets,
                                               nodeElementsArray=nodeElementsArray,
                                               elementBarycentersArray=elementBarycentersArray,
                                               node=node)

def getLocalNearestElementIntersection(double[:] coords,
                                       double[:] starting_coords,
                                       double[:,:,:] elementBoundaryNormalsArray,
                                       proteus_index_t[:,:] elementBoundariesArray,
                                       double[:,:] elementBoundaryBarycentersArray,
                                       proteus_index_t[:,:] elementBoundaryElementsArray,
                                       int[:] exteriorElementBoundariesBoolArray,
                                       proteus_index_t eN):
    """Find element nearest or containing coords through element boundary intersection

    Parameters
    ----------
    coords: double[:]
        coordinates of point for which a containing element must be found
    starting_coords: double[:]
        starting coords to look for coords
    elementBoundaryNormals: double[:,:,:]
        normals of the element boundaries
    elementBoundariesArray: int[:,:]
        index of boundaries per elements
    elementBoundaryBarycentersArray: int[:,:]
        barycenters of element boundaries
    elementBoundaryElementsArray: int[:,:]
        array of elements shared by boundaries
    exteriorElementBoundariesBoolArray: int[:]
        boolean array of exterior element boundaries (1: is exterior boundary)
        must be same length as the number of element boundaries
    eN: first guess of element

    Returns
    -------
    nearest_eN: int
        nearest element to coords (-1 if element at border)
    b_i_last: int
        last element boundary crossed
    """
    return pyxGetLocalNearestElementIntersection(coords=coords,
                                                 starting_coords=starting_coords,
                                                 elementBoundaryNormalsArray=elementBoundaryNormalsArray,
                                                 elementBoundariesArray=elementBoundariesArray,
                                                 elementBoundaryBarycentersArray=elementBoundaryBarycentersArray,
                                                 elementBoundaryElementsArray=elementBoundaryElementsArray,
                                                 exteriorElementBoundariesBoolArray=exteriorElementBoundariesBoolArray,
                                                 eN=eN)

def getLocalElement(femSpace,
                    coords,
                    node):
    """Given coordinates and its nearest node, determine if it is on a
    local element.
    (!) old implementation -> slow

    Parameters
    ----------
    femSpace: object
        finite element space
    coords: array_like
        coordinates from which to element
    node: int
        nearest node index

    Returns
    -------
    eN: int or None
        local index of element (None if not found)
    """
    patchBoundaryNodes=set()
    checkedElements=[]
    # nodeElementOffsets give the indices to get the elements sharing the node
    #log Profiling.logEvent("Getting Local Element")
    if node+1 < len(femSpace.mesh.nodeElementOffsets):
        for eOffset in range(femSpace.mesh.nodeElementOffsets[node], femSpace.mesh.nodeElementOffsets[node + 1]):
            eN = femSpace.mesh.nodeElementsArray[eOffset]
            checkedElements.append(eN)
            # union of set
            patchBoundaryNodes|=set(femSpace.mesh.elementNodesArray[eN])
            # evaluate the inverse map for element eN (global to local)
            xi = femSpace.elementMaps.getInverseValue(eN, coords)
            #J = femSpace.elementMaps.getJacobianValues(eN, )
            # query whether xi lies within the reference element
            if femSpace.elementMaps.referenceElement.onElement(xi):
                return eN
    else:
        for eOffset in range(femSpace.mesh.nodeElementOffsets[node]):
            eN = femSpace.mesh.nodeElementsArray[eOffset]
            checkedElements.append(eN)
            # union of set
            patchBoundaryNodes|=set(femSpace.mesh.elementNodesArray[eN])
            # evaluate the inverse map for element eN (global to local)
            xi = femSpace.elementMaps.getInverseValue(eN, coords)
            #J = femSpace.elementMaps.getJacobianValues(eN, )
            # query whether xi lies within the reference element
            if femSpace.elementMaps.referenceElement.onElement(xi):
                return eN
    # extra loop if case coords is in neighbour element
    for node in patchBoundaryNodes:
        for eOffset in range(femSpace.mesh.nodeElementOffsets[node], femSpace.mesh.nodeElementOffsets[node + 1]):
            eN = femSpace.mesh.nodeElementsArray[eOffset]
            if eN not in checkedElements:
                checkedElements.append(eN)
                # evaluate the inverse map for element eN
                xi = femSpace.elementMaps.getInverseValue(eN, coords)
                # query whether xi lies within the reference element
                if femSpace.elementMaps.referenceElement.onElement(xi):
                    return eN
    # no elements found
    return None


def updateElementBoundaryNormalsTriangle(double[:,:,:] elementBoundaryNormalsArray_,
                                         double[:,:] nodeArray,
                                         proteus_index_t[:,:] elementBoundariesArray,
                                         proteus_index_t[:,:] elementBoundaryNodesArray,
                                         double[:,:] elementBoundaryBarycentersArray,
                                         double[:,:] elementBarycentersArray,
                                         proteus_index_t nElements):
    pyxUpdateElementBoundaryNormalsTriangle(elementBoundaryNormalsArray_=elementBoundaryNormalsArray_,
                                            nodeArray=nodeArray,
                                            elementBoundariesArray=elementBoundariesArray,
                                            elementBoundaryNodesArray=elementBoundaryNodesArray,
                                            elementBoundaryBarycentersArray=elementBoundaryBarycentersArray,
                                            elementBarycentersArray=elementBarycentersArray,
                                            nElements=nElements)

def getElementBoundaryNormalsTriangle(double[:,:] nodeArray,
                                      proteus_index_t[:,:] elementBoundariesArray,
                                      proteus_index_t[:,:] elementBoundaryNodesArray,
                                      double[:,:] elementBoundaryBarycentersArray,
                                      double[:,:] elementBarycentersArray):
    elementBoundaryNormalsArray_ = np.zeros(elementBoundariesArray.shape[0], 3, 3)
    cdef proteus_index_t nElements = len(elementBoundaryNormalsArray_)
    pyxUpdateElementBoundaryNormalsTriangle(elementBoundaryNormalsArray_=elementBoundaryNormalsArray_,
                                            nodeArray=nodeArray,
                                            elementBoundariesArray=elementBoundariesArray,
                                            elementBoundaryNodesArray=elementBoundaryNodesArray,
                                            elementBoundaryBarycentersArray=elementBoundaryBarycentersArray,
                                            elementBarycentersArray=elementBarycentersArray,
                                            nElements=nElements)
    return elementBoundaryNormalsArray_

def updateElementBoundaryNormalsTetra(double[:,:,:] elementBoundaryNormalsArray_,
                                      double[:,:] nodeArray,
                                      proteus_index_t[:,:] elementBoundariesArray,
                                      proteus_index_t[:,:] elementBoundaryNodesArray,
                                      double[:,:] elementBoundaryBarycentersArray,
                                      double[:,:] elementBarycentersArray,
                                      proteus_index_t nElements):
    pyxUpdateElementBoundaryNormalsTetra(elementBoundaryNormalsArray_=elementBoundaryNormalsArray_,
                                         nodeArray=nodeArray,
                                         elementBoundariesArray=elementBoundariesArray,
                                         elementBoundaryNodesArray=elementBoundaryNodesArray,
                                         elementBoundaryBarycentersArray=elementBoundaryBarycentersArray,
                                         elementBarycentersArray=elementBarycentersArray,
                                         nElements=nElements)

def getElementBoundaryNormalsTetra(double[:,:] nodeArray,
                                   proteus_index_t[:,:] elementBoundariesArray,
                                   proteus_index_t[:,:] elementBoundaryNodesArray,
                                   double[:,:] elementBoundaryBarycentersArray,
                                   double[:,:] elementBarycentersArray):
    elementBoundaryNormalsArray_ = np.zeros(elementBoundariesArray.shape[0], 4, 3)
    cdef proteus_index_t nElements = len(elementBoundaryNormalsArray_)
    pyxUpdateElementBoundaryNormalsTetra(elementBoundaryNormalsArray_=elementBoundaryNormalsArray_,
                                         nodeArray=nodeArray,
                                         elementBoundariesArray=elementBoundariesArray,
                                         elementBoundaryNodesArray=elementBoundaryNodesArray,
                                         elementBoundaryBarycentersArray=elementBoundaryBarycentersArray,
                                         elementBarycentersArray=elementBarycentersArray,
                                         nElements=nElements)
    return elementBoundaryNormalsArray_

def updateElementVolumesTriangle(double[:] elementVolumesArray_,
                                 proteus_index_t[:,:] elementNodesArray,
                                 double[:,:] nodeArray,
                                 proteus_index_t nElements):
    cyUpdateElementVolumesTriangle(elementVolumesArray_=elementVolumesArray_,
                                   elementNodesArray=elementNodesArray,
                                   nodeArray=nodeArray,
                                   nElements=nElements)

def getElementVolumeTriangle(double[:] nA,
                             double[:] nB,
                             double[:] nC):
    return cyGetElementVolumeTriangle(nA=nA,
                                      nB=nB,
                                      nC=nC)

def updateElementVolumesTetra(double[:] elementVolumesArray_,
                              proteus_index_t[:,:] elementNodesArray,
                              double[:,:] nodeArray,
                              proteus_index_t nElements):
    cyUpdateElementVolumesTetra(elementVolumesArray_=elementVolumesArray_,
                                elementNodesArray=elementNodesArray,
                                nodeArray=nodeArray,
                                nElements=nElements)

def updateElementBarycenters(double[:,:] elementBarycentersArray_,
                             proteus_index_t[:,:] elementNodesArray,
                             double[:,:] nodeArray,
                             proteus_index_t nElements):
    cyUpdateElementBarycenters(elementBarycentersArray_=elementBarycentersArray_,
                               elementNodesArray=elementNodesArray,
                               nodeArray=nodeArray,
                               nElements=nElements)

def getCornerNodesTriangle(double[:,:] nodeArray,
                           proteus_index_t[:] nodeStarArray,
                           proteus_index_t[:] nodeStarOffsets,
                           int[:] nodeMaterialTypes,
                           proteus_index_t nNodes):
    return cyGetCornerNodesTriangle(nodeArray=nodeArray,
                                    nodeStarArray=nodeStarArray,
                                    nodeStarOffsets=nodeStarOffsets,
                                    nodeMaterialTypes=nodeMaterialTypes,
                                    nNodes=nNodes)

def getNonOwnedNodeValues(args_,
                          proteus_index_t nNodes_owned,
                          proteus_index_t nNodes_global,
                          proteus_index_t[:] nodeNumbering_subdomain2global,
                          proteus_index_t[:] nodeOffsets_subdomain_owned):
        nodeNumbering_subdomain2global = np.array(nodeNumbering_subdomain2global, dtype=INDEX_DTYPE)
        nodeOffsets_subdomain_owned = np.array(nodeOffsets_subdomain_owned, dtype=INDEX_DTYPE)
        from proteus import Comm
        cdef object comm = Comm.get().comm.tompi4py()
        cdef int comm_size = comm.size
        cdef int my_rank = comm.rank
        cdef dict arg_2rank = {}
        cdef dict nodes_2rank = {}
        cdef proteus_index_t[:] result
        # the counts and displacements for nodes coming in from other processors
        cdef int[:] counts_in = np.zeros(comm_size, dtype=np.int32)
        cdef int[:] displacements_in = np.zeros(comm_size, dtype=np.int32)
        # the counts and displacements args_ coming back from other processors
        cdef int[:] counts_out = np.zeros(comm_size, dtype=np.int32)
        cdef int[:] displacements_out = np.zeros(comm_size, dtype=np.int32)
        # shape of the argument
        cdef int[:] arg_shape = np.array(args_.shape, dtype=np.int32)
        cdef int[:] arg_shape_copy = np.array(args_.shape, dtype=np.int32)
        cdef int arg_shape_len = len(arg_shape)
        cdef int shape_factor = 1
        cdef int disp = 0
        cdef int rank, rank_recv, ii, ir, new_rank
        cdef proteus_index_t iN, node_new_rank
        cdef int sumtot
        cdef proteus_index_t[:] nodes_2rank_values
        cdef proteus_index_t[:] nodes_2doArray = np.zeros(0, dtype=INDEX_DTYPE)
        if arg_shape_len > 1:
            for i in range(1, arg_shape_len):
                shape_factor = shape_factor*arg_shape[i]
        if comm_size > 1:
            for rank in range(comm_size):
                nodes_2rank[rank] = np.zeros(0, dtype=INDEX_DTYPE)
            for node in range(nNodes_owned, nNodes_global):
                result = cyCheckOwnedVariable(variable_nb_local=node,
                                              rank=my_rank,
                                              nVariables_owned=nNodes_owned,
                                              variableNumbering_subdomain2global=nodeNumbering_subdomain2global,
                                              variableOffsets_subdomain_owned=nodeOffsets_subdomain_owned)
                node_new_rank = result[0]
                new_rank = result[1]
                nodes_2rank[new_rank] = np.append(nodes_2rank[new_rank], node_new_rank)
            # SEND THOSE NODES TO RELEVANT PROCESSORS
            for rank_recv in range(comm_size):
                # -----
                # find length of array to do on processor rank_recv
                nodes_2rank_values = (nodes_2rank[rank_recv][:]).astype(INDEX_DTYPE)
                nodes_2rank_len = len(nodes_2rank_values)
                array_size = comm.allreduce(nodes_2rank_len, op=MPI.SUM)
                # initialise node_2doArray on receiving processor
                if rank_recv == my_rank:
                    nodes_2doArray = np.zeros(array_size, dtype=INDEX_DTYPE)
                # -----
                # get count and disp info for retrieving info from rank_recv later
                # counts_out[rank_recv] = nodes_2rank_len
                counts_out[rank_recv] = nodes_2rank_len
                if rank_recv > 0:
                    displacements_out[rank_recv] = (displacements_out[rank_recv-1]+counts_out[rank_recv-1])
                # -----
                # get count and disp info for receiving array on rank_recv
                my_size = np.array([nodes_2rank_len], dtype=np.int32)
                comm.Gatherv(my_size,
                             [counts_in,
                             tuple(1 for i in range(comm_size)),
                             tuple(i for i in range(comm_size)),
                             MPI.INT],
                             root=rank_recv
                               )
                if rank_recv == my_rank:
                    sumtot = 0
                    for ir in range(comm_size):
                        if ir > 0:
                            sumtot += counts_in[ir-1]
                            if counts_in[ir] == 0:
                                displacements_in[ir] = 0
                            else:
                                displacements_in[ir] = sumtot
                # -----
                # get the nodes_2doArray (nodes where to retrieve values for arg)
                # datatype = MPI.INT.Create_contiguous(2).Commit() 
                comm.Gatherv(nodes_2rank_values,
                             [nodes_2doArray,
                              counts_in,
                              displacements_in,
                              MPI_INDEX],
                              root=rank_recv)
            # SEND VALUE BACK TO ORIGINAL PROCESSORS
            arg_shape_copy[0] = 0
            for rank in range(comm_size):
                # create empty arrays
                arg_shape_copy[0] = counts_in[rank]
                arg_2rank[rank] = np.zeros(arg_shape_copy)
            # build array of arg to send back
            for rank in range(comm_size):
                arg_2rank_values = arg_2rank[rank]
                disp = displacements_in[rank]
                for iN in range(counts_in[rank]):
                    if arg_shape_len > 1:
                        for ii in range(arg_shape[1]):
                            arg_2rank_values[iN, ii] = args_[nodes_2doArray[iN+disp], ii]
                    else:
                        arg_2rank_values[iN] = args_[nodes_2doArray[iN+disp]]
            # retrieve solution
            arg_shape_copy[0] = nNodes_global-nNodes_owned
            arg_2doArray = np.zeros(arg_shape_copy, dtype=np.double)
            for rank_recv in range(comm_size):
                arg_2rank_values = arg_2rank[rank_recv].astype(np.double)
                if arg_shape_len > 1:
                    datatype = MPI.DOUBLE.Create_contiguous(shape_factor).Commit() 
                else:
                    datatype = MPI.DOUBLE
                comm.Gatherv(arg_2rank_values,
                             [arg_2doArray,
                              counts_out,
                              displacements_out,
                              datatype],
                             root=rank_recv)
            # FINALLY APPLY FINAL POSITION OF NON-OWNED NODES
            for iN in range(nNodes_owned, nNodes_global):
                if arg_shape_len > 1:
                    for ii in range(arg_shape[1]):
                        args_[iN, ii] = arg_2doArray[iN-nNodes_owned, ii]
                else:
                    args_[iN] = arg_2doArray[iN-nNodes_owned]
            # logEvent('Non-owned values recovered with {comm_total} communication steps ({comm_pp}*{nproc})'.format(comm_total=str(comm_size*4),
            #                                                                                                        comm_pp=str(4),
            #                                                                                                        nproc=str(comm_size)))

def checkOwnedVariable(proteus_index_t variable_nb_local,
                       int rank,
                       proteus_index_t nVariables_owned,
                       proteus_index_t[:] variableNumbering_subdomain2global,
                       proteus_index_t[:] variableOffsets_subdomain_owned):
    return cyCheckOwnedVariable(variable_nb_local=variable_nb_local,
                                rank=rank,
                                nVariables_owned=nVariables_owned,
                                variableNumbering_subdomain2global=variableNumbering_subdomain2global,
                                variableOffsets_subdomain_owned=variableOffsets_subdomain_owned)

def pyScalarRecoveryAtNodes(double[:] scalars,
                            proteus_index_t[:] nodeElementsArray,
                            proteus_index_t[:] nodeElementOffsets):
    return cyScalarRecoveryAtNodes(scalars=scalars,
                                   nodeElementsArray=nodeElementsArray,
                                   nodeElementOffsets=nodeElementOffsets)

def pyScalarRecoveryAtNodesWeighted(double[:] scalars,
                                    proteus_index_t[:] nodeElementsArray,
                                    proteus_index_t[:] nodeElementOffsets,
                                    double[:] detJ_array,
                                    proteus_index_t nNodes):
    return cyScalarRecoveryAtNodesWeighted(scalars=scalars,
                                           nodeElementsArray=nodeElementsArray,
                                           nodeElementOffsets=nodeElementOffsets,
                                           detJ_array=detJ_array,
                                           nNodes=nNodes)

def pyVectorRecoveryAtNodes(vectors,
                            nodeElementsArray,
                            nodeElementOffsets,
                            nd):
    return cyVectorRecoveryAtNodes(vectors=vectors,
                                   nodeElementsArray=nodeElementsArray,
                                   nodeElementOffsets=nodeElementOffsets,
                                   nd=nd)

def pyVectorRecoveryAtNodesWeighted(double[:,:] vectors,
                                    proteus_index_t[:] nodeElementsArray,
                                    proteus_index_t[:] nodeElementOffsets,
                                    double[:,:] detJ_array,
                                    int nd):
    return cyVectorRecoveryAtNodesWeighted(vectors=vectors,
                                           nodeElementsArray=nodeElementsArray,
                                           nodeElementOffsets=nodeElementOffsets,
                                           detJ_array=detJ_array,
                                           nd=nd)

### Cython implementation of functions above

cdef void cySmoothNodesLaplace(double[:,:] nodeArray_,
                               proteus_index_t[:] nodeStarOffsets,
                               proteus_index_t[:] nodeStarArray,
                               int[:] nodeMaterialTypes,
                               proteus_index_t nNodes_owned,
                               int nd,
                               bool simultaneous=False,
                               bool smoothBoundaries=True,
                               int[:] fixedNodesBoolArray=None,
                               double alpha=0.):
    cdef double[:,:] nodeArray0
    if simultaneous or alpha!= 0:
        nodeArray0 = nodeArray_.copy()
    cdef double[3] sum_star
    cdef double dot
    cdef proteus_index_t nNodeInStar
    cdef proteus_index_t nNodes = 0
    cdef bool fixed_node = False
    cdef double[:] fixed_dir = np.zeros(3)
    cdef double fixed_dir_dist
    cdef proteus_index_t nOffset
    cdef proteus_index_t node
    for node in range(nNodes_owned):
        sum_star[0] = 0.
        sum_star[1] = 0.
        sum_star[2] = 0.
        nNodes = 0
        if nodeMaterialTypes[node] == 0:
            for nOffset in range(nodeStarOffsets[node],
                                 nodeStarOffsets[node+1]):
                if simultaneous is True:
                    sum_star[0] += nodeArray0[nodeStarArray[nOffset], 0]
                    sum_star[1] += nodeArray0[nodeStarArray[nOffset], 1]
                    sum_star[2] += nodeArray0[nodeStarArray[nOffset], 2]
                else:
                    sum_star[0] += nodeArray_[nodeStarArray[nOffset], 0]
                    sum_star[1] += nodeArray_[nodeStarArray[nOffset], 1]
                    sum_star[2] += nodeArray_[nodeStarArray[nOffset], 2]
                nNodes += 1
            nodeArray_[node, 0] = sum_star[0]/nNodes
            nodeArray_[node, 1] = sum_star[1]/nNodes
            nodeArray_[node, 2] = sum_star[2]/nNodes
        # boundary smoothing not ready yet
        elif smoothBoundaries is True:
            # smooth on boundary only
            fixed_node = False
            if fixedNodesBoolArray is not None:
                if fixedNodesBoolArray[node] == 1:
                    sum_star[0] = nodeArray0[node, 0]
                    sum_star[1] = nodeArray0[node, 1]
                    sum_star[2] = nodeArray0[node, 2]
                    nNodes = 1
                    fixed_node = True
            if fixed_node is False:
                if nd == 2:
                    cyFindBoundaryDirectionTriangle(dir_=fixed_dir,
                                                    node=node,
                                                    nodeArray=nodeArray_,
                                                    nodeStarOffsets=nodeStarOffsets,
                                                    nodeStarArray=nodeStarArray,
                                                    nodeMaterialTypes=nodeMaterialTypes)
                if nd == 3:
                    cyFindBoundaryDirectionTetra(dir_=fixed_dir,
                                                 node=node,
                                                 nodeArray=nodeArray_,
                                                 nodeStarOffsets=nodeStarOffsets,
                                                 nodeStarArray=nodeStarArray,
                                                 nodeMaterialTypes=nodeMaterialTypes)
                for nOffset in range(nodeStarOffsets[node],
                                    nodeStarOffsets[node+1]):
                    if simultaneous is True:
                        sum_star[0] += nodeArray0[nodeStarArray[nOffset], 0]
                        sum_star[1] += nodeArray0[nodeStarArray[nOffset], 1]
                        sum_star[2] += nodeArray0[nodeStarArray[nOffset], 2]
                    else:
                        sum_star[0] += nodeArray_[nodeStarArray[nOffset], 0]
                        sum_star[1] += nodeArray_[nodeStarArray[nOffset], 1]
                        sum_star[2] += nodeArray_[nodeStarArray[nOffset], 2]
                    nNodes += 1
            nodeArray_[node, 0] += (sum_star[0]/nNodes-nodeArray_[node, 0])*fixed_dir[0]
            nodeArray_[node, 1] += (sum_star[1]/nNodes-nodeArray_[node, 1])*fixed_dir[1]
            nodeArray_[node, 2] += (sum_star[2]/nNodes-nodeArray_[node, 2])*fixed_dir[2]
        else:
            sum_star[0] = nodeArray0[node, 0]
            sum_star[1] = nodeArray0[node, 1]
            sum_star[2] = nodeArray0[node, 2]
            nNodes = 1
            fixed_node = True
        # if alpha != 0:
        #     nodeArray_[node, 0] = alpha*nodeArray0[node, 0]+(1-alpha)*sum_star[0]/nNodes
        #     nodeArray_[node, 1] = alpha*nodeArray0[node, 0]+(1-alpha)*sum_star[1]/nNodes
        #     nodeArray_[node, 2] = alpha*nodeArray0[node, 0]+(1-alpha)*sum_star[2]/nNodes

cdef void cySmoothNodesCentroid(double[:,:] nodeArray_,
                                proteus_index_t[:] nodeElementOffsets,
                                proteus_index_t[:] nodeElementsArray,
                                int[:] nodeMaterialTypes,
                                double[:] elementVolumesArray,
                                double[:,:] elementBarycentersArray,
                                proteus_index_t[:,:] elementNodesArray,
                                proteus_index_t nNodes_owned,
                                int[:] fixedNodesBoolArray,
                                bool simultaneous=False,
                                bool smoothBoundaries = True,
                                double alpha=0.):
    cdef double[:,:] nodeArray0
    if simultaneous or alpha != 0:
        nodeArray0 = nodeArray_.copy()
    cdef double[3] sum_star
    cdef proteus_index_t nNodeInStar
    cdef double areas = 0.
    cdef int nNodes_star = 0
    cdef double[:] nodeOffset0
    cdef double[:] nodeOffset1
    cdef double var = 0.
    cdef double[:] centroid_cell
    cdef double[:] nA
    cdef double[:] nB
    cdef double[:] nC
    cdef double dot
    cdef double vec_dist
    cdef proteus_index_t node
    cdef proteus_index_t eOffset
    cdef proteus_index_t eN
    for node in range(nNodes_owned):
        sum_star[0] = 0.
        sum_star[1] = 0.
        sum_star[2] = 0.
        areas = 0.
        if nodeMaterialTypes[node] == 0:
            for eOffset in range(nodeElementOffsets[node],
                                    nodeElementOffsets[node+1]):
                eN = nodeElementsArray[eOffset]
                sum_star[0] += elementBarycentersArray[eN,0]*elementVolumesArray[eN]
                sum_star[1] += elementBarycentersArray[eN,1]*elementVolumesArray[eN]
                sum_star[2] += elementBarycentersArray[eN,2]*elementVolumesArray[eN]
                areas += elementVolumesArray[eN]
            if alpha != 0:
                nodeArray_[node, 0] = alpha*nodeArray0[node, 0]+(1-alpha)*sum_star[0]/areas
                nodeArray_[node, 1] = alpha*nodeArray0[node, 0]+(1-alpha)*sum_star[1]/areas
                nodeArray_[node, 2] = alpha*nodeArray0[node, 0]+(1-alpha)*sum_star[2]/areas
            else:
                nodeArray_[node, 0] = sum_star[0]/areas
                nodeArray_[node, 1] = sum_star[1]/areas
                nodeArray_[node, 2] = sum_star[2]/areas
            if not simultaneous:
                for eOffset in range(nodeElementOffsets[node],
                                     nodeElementOffsets[node+1]):
                    eN = nodeElementsArray[eOffset]
                    nA = nodeArray_[elementNodesArray[eN, 0]]
                    nB = nodeArray_[elementNodesArray[eN, 1]]
                    nC = nodeArray_[elementNodesArray[eN, 2]]
                    elementVolumesArray[eN] = cyGetElementVolumeTriangle(nA, nB, nC)
                    elementBarycentersArray[eN, 0] = (nA[0]+nB[0]+nC[0])/3.
                    elementBarycentersArray[eN, 1] = (nA[1]+nB[1]+nC[1])/3.
                    elementBarycentersArray[eN, 2] = (nA[2]+nB[2]+nC[2])/3.

# cdef tuple cyGetDistortionDilation(double[:,:,:,:] J_array,
#                                    double[:,:] detJ_array,
#                                    double[:] target_area_array,
#                                    int nd):
#     #W = np.array([[1., 0.5],
#     #              [0., 0.866]]) # [0., np.sqrt(3)/2.]
#     cdef double[:,:] J
#     cdef double[:,:] JT
#     cdef double detJ
#     cdef double trJTJ
#     cdef double[:,:] JTJ
#     cdef double dilation
#     cdef double distortion
#     cdef double[:] distortion_array = np.zeros(len(target_area_array))
#     cdef double[:] dilation_array = np.zeros(len(target_area_array))
#     for eN in range(len(target_area_array)):
#         detJ = detJ_array[eN, 0]  # index 0 as it is the same for all quadrature points
#         dilation = target_area_array[eN]/detJ
#         if dilation < 1.:
#             dilation = 1/dilation
#         dilation_array[eN] = dilation-1
#         J = J_array[eN][0]
#         JT = np.transpose(J)
#         JTJ = np.zeros_like(J)
#         JTJ = np.dot(J, JT)
#         trJTJ = 0.
#         for i in range(len(J)):
#             trJTJ += JTJ[i,i]
#         distortion = (1./nd*trJTJ)**(nd/2.)/detJ
#         distortion = 1-1./distortion
#         distortion_array[eN] = distortion
#         dilation_array[eN] = dilation
#     return distortion_array, dilation_array

cdef void cyUpdateDilationElements(double[:] elementDilationArray_,
                                   double[:] elementVolumeArray,
                                   double[:] elementVolumeTargetArray,
                                   proteus_index_t nElements):
    cdef double dilation
    cdef proteus_index_t eN
    for eN in range(nElements):
        dilation = elementVolumeArray[eN]/elementVolumeTargetArray[eN]
        if dilation < 1.:
            dilation = 1/dilation
        elementDilationArray_[eN] = dilation

cdef void cyUpdateDistortionElements(double[:] elementDistortionArray_,
                                     double[:,:,:,:] J_array,
                                     double[:,:] detJ_array,
                                     int nd,
                                     proteus_index_t nElements):
    cdef double[:,:] J
    cdef double[:,:] JT
    cdef double detJ
    cdef double trJTJ
    cdef double[:,:] JTJ
    cdef proteus_index_t eN
    cdef int iJ
    for eN in range(len(nElements)):
        detJ = detJ_array[eN, 0]  # index 0 as it is the same for all quadrature points
        J = J_array[eN][0]
        JT = np.transpose(J)
        JTJ = np.zeros_like(J)
        JTJ = np.dot(J, JT)
        trJTJ = 0.
        for iJ in range(len(J)):
            trJTJ += JTJ[iJ,iJ]
        elementDistortionArray_[eN] = (1./nd*trJTJ)**(nd/2.)/detJ

cdef void cyUpdateInverseMeanRatioTriangleNodes(double[:] IMRNodesArray_,
                                                double[:,:] nodeArray,
                                                proteus_index_t[:,:] elementNodesArray,
                                                proteus_index_t[:] nodeElementOffsets,
                                                proteus_index_t[:] nodeElementsArray,
                                                proteus_index_t nElements,
                                                proteus_index_t nNodes,
                                                bool el_average=False):
    cdef double[:,:] W = np.array([[1., 0.5],
                                   [0., sqrt(3)/2.]])
    cdef double[:,:] A = np.zeros((2,2))
    cdef double[:,:] AW = np.zeros((2,2))
    cdef int[:] nElementsArray = np.zeros(nElements)
    cdef double[:] IMR_elements
    if el_average:
        IMR_elements = np.zeros(nNodes)
    cdef proteus_index_t nNel = elementNodesArray.shape[1]
    cdef double[:] vec_a
    cdef double[:] vec_b
    cdef double[:] vec_c
    cdef double IMR
    cdef proteus_index_t nEl = 0
    cdef proteus_index_t iN
    cdef proteus_index_t eN
    cdef proteus_index_t node
    cdef proteus_index_t eOffset
    for eN in range(nElements):
        for iN in range(nNel):
            node = elementNodesArray[eN, iN]
            vec_a = nodeArray[elementNodesArray[eN, iN]]
            vec_b = nodeArray[elementNodesArray[eN, iN-1]]
            vec_c = nodeArray[elementNodesArray[eN, iN-2]]
            A[0,0] = vec_b[0]-vec_a[0]
            A[1,0] = vec_b[1]-vec_a[1]
            A[0,1] = vec_c[0]-vec_a[0]
            A[1,1] = vec_c[1]-vec_a[1]
            AW = np.dot(A, np.linalg.inv(W))
            IMR = (AW[0,0]**2+AW[0,1]**2+AW[1,0]**2+AW[1,1]**2)/(2*abs(AW[0,0]*AW[1,1]-AW[0,1]*AW[1,0]))
            IMRNodesArray_[node] += IMR
            nElementsArray[node] += 1
            if el_average:
                IMR_elements[eN] += IMR
        if el_average:
            IMR_elements[eN] = IMR_elements[eN]/3.
    for node in range(nNodes):
        if not el_average:
            IMRNodesArray_[node] = IMRNodesArray_[node]/nElementsArray[node]
        else:
            nEl = 0
            IMRNodesArray_[node] = 0.
            for eOffset in range(nodeElementOffsets[node],
                                 nodeElementOffsets[node+1]):
                eN = nodeElementsArray[eOffset]
                nEl += 1
                IMRNodesArray_[node] += IMR_elements[eN]
            IMRNodesArray_[node] = IMRNodesArray_[node]/nEl

cdef void cyUpdateInverseMeanRatioTriangleElements(double[:] IMRElementsArray_,
                                                   double[:,:] nodeArray,
                                                   proteus_index_t[:,:] elementNodesArray,
                                                   proteus_index_t nElements):
    cdef double[:,:] W = np.array([[1., 0.5],
                                   [0., sqrt(3)/2.]])
    cdef double[:,:] A = np.zeros((2,2))
    cdef double[:,:] AW = np.zeros((2,2))
    cdef double[:] vec_a
    cdef double[:] vec_b
    cdef double[:] vec_c
    cdef double IMR
    cdef proteus_index_t nEl = 0
    cdef proteus_index_t eN
    cdef proteus_index_t iN
    cdef proteus_index_t node
    for eN in range(nElements):
        for iN, node in enumerate(elementNodesArray[eN]):
            vec_a = nodeArray[elementNodesArray[eN, iN]]
            vec_b = nodeArray[elementNodesArray[eN, iN-1]]
            vec_c = nodeArray[elementNodesArray[eN, iN-2]]
            A[0,0] = vec_b[0]-vec_a[0]
            A[1,0] = vec_b[1]-vec_a[1]
            A[0,1] = vec_c[0]-vec_a[0]
            A[1,1] = vec_c[1]-vec_a[1]
            AW = np.dot(A, np.linalg.inv(W))
            IMR = (AW[0,0]**2+AW[0,1]**2+AW[1,0]**2+AW[1,1]**2)/(2*abs(AW[0,0]*AW[1,1]-AW[0,1]*AW[1,0]))
            IMRElementsArray_[eN] += IMR
        IMRElementsArray_[eN] = IMRElementsArray_[eN]/3.

cdef double cyGetInverseMeanRatioSingleTriangle(proteus_index_t node0,
                                                double[:,:] nodeArray,
                                                proteus_index_t[:,:] elementNodesArray,
                                                proteus_index_t[:] nodeElementOffsets,
                                                proteus_index_t[:] nodeElementsArray,
                                                bool el_average=False):
    cdef double[:,:] W = np.array([[1., 0.5],
                                   [0., sqrt(3)/2.]])
    cdef double[:,:] A = np.zeros((2,2))
    cdef double[:,:] AW = np.zeros((2,2))
    cdef double IMR_node = 0.
    cdef proteus_index_t nEl = 0
    cdef double[:] vec_a
    cdef double[:] vec_b
    cdef double[:] vec_c
    cdef double IMR
    cdef proteus_index_t eOffset
    cdef proteus_index_t iN
    cdef proteus_index_t node
    cdef proteus_index_t eN
    cdef proteus_index_t nNel = elementNodesArray.shape[1]
    for eOffset in range(nodeElementOffsets[node0],
                         nodeElementOffsets[node0+1]):
        eN = nodeElementsArray[eOffset]
        nEl += 1
        ################
        for iN in range(nNel):
            node = elementNodesArray[eN, iN]
            if el_average:
                vec_a = nodeArray[elementNodesArray[eN, iN]]
                vec_b = nodeArray[elementNodesArray[eN, iN-1]]
                vec_c = nodeArray[elementNodesArray[eN, iN-2]]
                A[0,0] = vec_b[0]-vec_a[0]
                A[1,0] = vec_b[1]-vec_a[1]
                A[0,1] = vec_c[0]-vec_a[0]
                A[1,1] = vec_c[1]-vec_a[1]
                AW = np.dot(A, np.linalg.inv(W))
                IMR = (AW[0,0]**2+AW[0,1]**2+AW[1,0]**2+AW[1,1]**2)/(2*abs(AW[0,0]*AW[1,1]-AW[0,1]*AW[1,0]))
                IMR_node += IMR/3.  # /3. because 3 nodes in element
            else:
                if node == node0:
                    vec_a = nodeArray[elementNodesArray[eN, iN]]
                    vec_b = nodeArray[elementNodesArray[eN, iN-1]]
                    vec_c = nodeArray[elementNodesArray[eN, iN-2]]
                    A[0,0] = vec_b[0]-vec_a[0]
                    A[1,0] = vec_b[1]-vec_a[1]
                    A[0,1] = vec_c[0]-vec_a[0]
                    A[1,1] = vec_c[1]-vec_a[1]
                    AW = np.dot(A, np.linalg.inv(W))
                    IMR = (AW[0,0]**2+AW[0,1]**2+AW[1,0]**2+AW[1,1]**2)/(2*abs(AW[0,0]*AW[1,1]-AW[0,1]*AW[1,0]))
                    IMR_node += IMR
    IMR_node = IMR_node/nEl
    return IMR_node

cdef double[:,:] cySmoothNodesQuality(double[:] distortion,
                                      double[:] dilation,
                                      double[:,:] nodeArray,
                                      proteus_index_t nNodes_owned,
                                      int[:] nodeMaterialTypes,
                                      proteus_index_t[:] nodeElementOffsets,
                                      proteus_index_t[:] nodeElementsArray,
                                      proteus_index_t[:,:] elementNodesArray,
                                      bool apply_directly=False):
    cdef double[:,:] disp = nodeArray.copy()
    cdef double[:,:] nodeArrayMod
    if not apply_directly:
        nodeArrayMod = nodeArray.copy()
    else:
        nodeArrayMod = nodeArray
    cdef double[3] weighted_pos
    cdef proteus_index_t nNel = elementNodesArray.shape[1]
    cdef double weight = 0
    cdef double weights = 0
    cdef proteus_index_t node
    cdef proteus_index_t eOffset
    cdef proteus_index_t eN
    cdef int eNnode
    cdef proteus_index_t iN
    for node in range(nNodes_owned):
        if nodeMaterialTypes[node] == 0:
            weights = 0
            weighted_pos[0] = 0
            weighted_pos[1] = 0
            weighted_pos[2] = 0
            for eOffset in range(nodeElementOffsets[node],
                                 nodeElementOffsets[node+1]):
                eN = nodeElementsArray[eOffset]
                for iN in range(nNel):
                    eNnode = elementNodesArray[eN, iN]
                    if eNnode != node:
                        weight = distortion[eN]
                        weighted_pos[0] += nodeArrayMod[eNnode, 0]*weight
                        weighted_pos[1] += nodeArrayMod[eNnode, 1]*weight
                        weighted_pos[2] += nodeArrayMod[eNnode, 2]*weight
                        weights += weight
                nodeArrayMod[node, 0] = weighted_pos[0]/weights
                nodeArrayMod[node, 1] = weighted_pos[1]/weights
                nodeArrayMod[node, 2] = weighted_pos[2]/weights
        disp[node, 0] = disp[node, 0]-nodeArray[node, 0]
        disp[node, 1] = disp[node, 1]-nodeArray[node, 1]
        disp[node, 2] = disp[node, 2]-nodeArray[node, 2]
    return disp

cdef int pyxGetLocalNearestNode(double[:] coords,
                                double[:,:] nodeArray,
                                proteus_index_t[:] nodeStarOffsets,
                                proteus_index_t[:] nodeStarArray,
                                proteus_index_t node):
    # determine local nearest node distance
    cdef proteus_index_t nearest_node = node
    cdef int nearest_node0 = node
    cdef double dist
    cdef double min_dist
    cdef double[:] node_coords = nodeArray[nearest_node]
    cdef bool found_node = False
    min_dist = (node_coords[0]-coords[0])*(node_coords[0]-coords[0])+\
               (node_coords[1]-coords[1])*(node_coords[1]-coords[1])+\
               (node_coords[2]-coords[2])*(node_coords[2]-coords[2])
    cdef int i = 0
    cdef proteus_index_t nOffset
    while found_node is False:
        nearest_node0 = nearest_node
        for nOffset in range(nodeStarOffsets[nearest_node0],
                             nodeStarOffsets[nearest_node0+1]):
            node = nodeStarArray[nOffset]
            node_coords = nodeArray[node]
            dist = (node_coords[0]-coords[0])*(node_coords[0]-coords[0])+\
                   (node_coords[1]-coords[1])*(node_coords[1]-coords[1])+\
                   (node_coords[2]-coords[2])*(node_coords[2]-coords[2])
            if dist < min_dist:
                min_dist = dist
                nearest_node = node
        if nearest_node0 == nearest_node:
            found_node = True
        i += 1
    return nearest_node


cdef int pyxGetLocalNearestElement(double[:] coords,
                                   double[:,:] elementBarycentersArray,
                                   proteus_index_t[:,:] elementNeighborsArray,
                                   proteus_index_t eN):
    # determine local nearest node distance
    cdef proteus_index_t nearest_eN = eN
    cdef proteus_index_t nearest_eN0 = eN
    cdef double dist
    cdef double min_dist
    cdef double[3] eN_coords
    cdef bool found_eN = False
    cdef proteus_index_t nOffset
    cdef proteus_index_t eN_
    eN_coords[0] = elementBarycentersArray[eN, 0]
    eN_coords[1] = elementBarycentersArray[eN, 1]
    eN_coords[2] = elementBarycentersArray[eN, 2]
    cdef int nEneig = elementNeighborsArray.shape[1]
    cdef proteus_index_t iEn
    min_dist = (eN_coords[0]-coords[0])*(eN_coords[0]-coords[0])+\
               (eN_coords[1]-coords[1])*(eN_coords[1]-coords[1])+\
               (eN_coords[2]-coords[2])*(eN_coords[2]-coords[2])
    cdef int i = 0
    while found_eN is False:
        nearest_eN0 = nearest_eN

        for iEn in range(nEneig):
            eN_ = elementNeighborsArray[nearest_eN0, iEn]
            eN_coords[0] = elementBarycentersArray[eN_, 0]
            eN_coords[1] = elementBarycentersArray[eN_, 1]
            eN_coords[2] = elementBarycentersArray[eN_, 2]
            dist = (eN_coords[0]-coords[0])*(eN_coords[0]-coords[0])+\
                   (eN_coords[1]-coords[1])*(eN_coords[1]-coords[1])+\
                   (eN_coords[2]-coords[2])*(eN_coords[2]-coords[2])
            if dist < min_dist:
                min_dist = dist
                nearest_eN = eN_
        if nearest_eN0 == nearest_eN:
            found_eN = True
        i += 1
    return nearest_eN

cdef proteus_index_t[:] pyxGetLocalNearestElementIntersection(double[:] coords,
                                                  double[:] starting_coords,
                                                  double[:,:,:] elementBoundaryNormalsArray,
                                                  proteus_index_t[:,:] elementBoundariesArray,
                                                  double[:,:] elementBoundaryBarycentersArray,
                                                  proteus_index_t[:,:] elementBoundaryElementsArray,
                                                  int[:] exteriorElementBoundariesBoolArray,
                                                  proteus_index_t eN):
    # determine local nearest node distance
    cdef proteus_index_t[2] result
    # cdef int[:] result = np.zeros(2, dtype=np.int32)
    result[0] = -1
    result[1] = -1
    cdef double[3] eN_coords
    eN_coords[0] = starting_coords[0]
    eN_coords[1] = starting_coords[1]
    eN_coords[2] = starting_coords[2]
    cdef double min_dist = sqrt((eN_coords[0]-coords[0])*(eN_coords[0]-coords[0])+\
                                (eN_coords[1]-coords[1])*(eN_coords[1]-coords[1])+\
                                (eN_coords[2]-coords[2])*(eN_coords[2]-coords[2]))
    cdef double[3] direction
    direction[0] = (coords[0]-eN_coords[0])/min_dist
    direction[1] = (coords[1]-eN_coords[1])/min_dist
    direction[2] = (coords[2]-eN_coords[2])/min_dist
    cdef int maxit = 5*elementBoundariesArray.shape[0]
    cdef proteus_index_t nEbn = elementBoundariesArray.shape[1]  # nb boundaries per element
    cdef proteus_index_t nearest_eN = eN
    cdef proteus_index_t nearest_eN0 = eN
    cdef int nOffsets
    cdef double dist
    cdef bool found_eN = False
    cdef int i = 0
    cdef double[:] bound_bar  # barycenter of boundary
    cdef double alpha  # distance
    cdef proteus_index_t b_i  # boundary index element
    cdef int b_i_last = -1
    cdef double[:] normal  # element boundary normal
    cdef double dot  # dot product result
    cdef double dot2  # dot product 2 result
    cdef int it = 0
    cdef proteus_index_t eN_
    cdef int j
    cdef int k
    while found_eN is False and it < maxit:
        nearest_eN0 = nearest_eN
        alpha_min = 1e12
        for j in range(nEbn):
            b_i = elementBoundariesArray[nearest_eN0, j]
            # if b_i != b_i_last:
            normal = elementBoundaryNormalsArray[nearest_eN0, j]
            bound_bar = elementBoundaryBarycentersArray[b_i]
            dot = normal[0]*direction[0]+normal[1]*direction[1]+normal[2]*direction[2]
            if dot > 0.:
                dot2 = (bound_bar[0]-eN_coords[0])*normal[0]+(bound_bar[1]-eN_coords[1])*normal[1]+(bound_bar[2]-eN_coords[2])*normal[2]
                if dot2 >= 0.:
                    alpha = dot2/dot
                    if 0. <= alpha < alpha_min:
                        alpha_min = alpha
                        for k in range(2):
                            eN_ = elementBoundaryElementsArray[b_i, k]
                            if eN_ != nearest_eN0:
                                nearest_eN = eN_
                        # nearest_eN = elementBoundaryElementsArray[nearest_eN0]
                        b_i_last = b_i
        if nearest_eN != nearest_eN0:
            if min_dist-alpha_min >=0:
                eN_coords[0] += alpha_min*direction[0]
                eN_coords[1] += alpha_min*direction[1]
                eN_coords[2] += alpha_min*direction[2]
                min_dist -= alpha_min
            else:  # going too far
                nearest_eN = nearest_eN0
        if nearest_eN0 == nearest_eN or nearest_eN == -1:
            found_eN = True
        i += 1
        it += 1
    if it >= maxit:
        assert 1>2, 'could not find element! (element {eN}: {x}, {y}, {z}), nearest_eN {nearest_eN}: closest coords: {x2}, {y2}, {z2}, after {maxit} iterations'.format(eN=eN,
                                                                                                                                                                        x=coords[0],
                                                                                                                                                                        y=coords[1],
                                                                                                                                                                        z=coords[2],
                                                                                                                                                                        nearest_eN=nearest_eN,
                                                                                                                                                                        x2=eN_coords[0],
                                                                                                                                                                        y2=eN_coords[1],
                                                                                                                                                                        z2=eN_coords[2],
                                                                                                                                                                        maxit=maxit)
    result[0] = nearest_eN
    result[1] = b_i_last
    return result

cdef int pyxGetLocalNearestElementAroundNode(double[:] coords,
                                             proteus_index_t[:] nodeElementOffsets,
                                             proteus_index_t[:] nodeElementsArray,
                                             double[:,:] elementBarycentersArray,
                                             proteus_index_t node):
    cdef double dist
    cdef double min_dist
    cdef double[:] eN_coords
    cdef int i = 0
    cdef proteus_index_t eN
    cdef proteus_index_t iEn
    cdef int rmin = nodeElementOffsets[node]
    cdef int rmax = nodeElementOffsets[node+1]
    for iEn in range(rmin, rmax):
        eN = nodeElementsArray[iEn]
        eN_coords = elementBarycentersArray[eN]
        dist = (eN_coords[0]-coords[0])*(eN_coords[0]-coords[0])+\
               (eN_coords[1]-coords[1])*(eN_coords[1]-coords[1])+\
               (eN_coords[2]-coords[2])*(eN_coords[2]-coords[2])
        if i == 0 or dist < min_dist:
            min_dist = dist
            nearest_eN = eN
        i += 1
    # determine local nearest node distance
    return nearest_eN

cdef void pyxUpdateElementBoundaryNormalsTetra(double[:,:,:] elementBoundaryNormalsArray_,
                                               double[:,:] nodeArray,
                                               proteus_index_t[:,:] elementBoundariesArray,
                                               proteus_index_t[:,:] elementBoundaryNodesArray,
                                               double[:,:] elementBoundaryBarycentersArray,
                                               double[:,:] elementBarycentersArray,
                                               proteus_index_t nElements):
    cdef double[3] normal_check
    cdef double[3] U
    cdef double[3] V
    cdef proteus_index_t b_i
    cdef double[:] node0
    cdef double[:] node1
    cdef double[:] node2
    cdef int i
    cdef int j
    cdef double dot
    for i in range(nElements):
        for j in range(4):
            b_i = elementBoundariesArray[i, j]
            node0 = nodeArray[elementBoundaryNodesArray[elementBoundariesArray[i, j],0]]
            node1 = nodeArray[elementBoundaryNodesArray[elementBoundariesArray[i, j],1]]
            node2 = nodeArray[elementBoundaryNodesArray[elementBoundariesArray[i, j],2]]
            U[0] = node1[0]-node0[0]
            U[1] = node1[1]-node0[1]
            U[2] = node1[2]-node0[2]
            V[0] = node2[0]-node0[0]
            V[1] = node2[1]-node0[1]
            V[2] = node2[2]-node0[2]
            elementBoundaryNormalsArray_[i,j,0] = U[1]*V[2]-U[2]*V[1]
            elementBoundaryNormalsArray_[i,j,1] = U[2]*V[0]-U[0]*V[2]
            elementBoundaryNormalsArray_[i,j,2] = U[0]*V[1]-U[1]*V[0]
            lenghtn = sqrt(elementBoundaryNormalsArray_[i,j,0]**2+elementBoundaryNormalsArray_[i,j,1]**2+elementBoundaryNormalsArray_[i,j,2]**2)
            elementBoundaryNormalsArray_[i,j,0] /= lenghtn
            elementBoundaryNormalsArray_[i,j,1] /= lenghtn
            elementBoundaryNormalsArray_[i,j,2] /= lenghtn
            normal_check[0] = elementBoundaryBarycentersArray[b_i, 0]-elementBarycentersArray[i, 0]
            normal_check[1] = elementBoundaryBarycentersArray[b_i, 1]-elementBarycentersArray[i, 1]
            normal_check[2] = elementBoundaryBarycentersArray[b_i, 2]-elementBarycentersArray[i, 2]
            dot = elementBoundaryNormalsArray_[i,j,0]*normal_check[0]+\
                  elementBoundaryNormalsArray_[i,j,1]*normal_check[1]+\
                  elementBoundaryNormalsArray_[i,j,2]*normal_check[2]
            if dot < 0:
                elementBoundaryNormalsArray_[i,j,0] = -elementBoundaryNormalsArray_[i,j,0]
                elementBoundaryNormalsArray_[i,j,1] = -elementBoundaryNormalsArray_[i,j,1]
                elementBoundaryNormalsArray_[i,j,2] = -elementBoundaryNormalsArray_[i,j,2]

cdef void pyxUpdateElementBoundaryNormalsTriangle(double[:,:,:] elementBoundaryNormalsArray_,
                                                  double[:,:] nodeArray,
                                                  proteus_index_t[:,:] elementBoundariesArray,
                                                  proteus_index_t[:,:] elementBoundaryNodesArray,
                                                  double[:,:] elementBoundaryBarycentersArray,
                                                  double[:,:] elementBarycentersArray,
                                                  proteus_index_t nElements):
    cdef double[2] normal_check
    cdef double[2] U
    cdef proteus_index_t b_i
    cdef double[:] node0
    cdef double[:] node1
    cdef int i
    cdef int j
    cdef double dot
    for i in range(nElements):
        for j in range(3):
            b_i = elementBoundariesArray[i, j]
            node0 = nodeArray[elementBoundaryNodesArray[elementBoundariesArray[i, j],0]]
            node1 = nodeArray[elementBoundaryNodesArray[elementBoundariesArray[i, j],1]]
            U[0] = node1[0]-node0[0]
            U[1] = node1[1]-node0[1]
            elementBoundaryNormalsArray_[i,j,0] = -U[1]
            elementBoundaryNormalsArray_[i,j,1] = U[0]
            lenghtn = sqrt(elementBoundaryNormalsArray_[i,j,0]**2+elementBoundaryNormalsArray_[i,j,1]**2)
            elementBoundaryNormalsArray_[i,j,0] /= lenghtn
            elementBoundaryNormalsArray_[i,j,1] /= lenghtn
            normal_check[0] = elementBoundaryBarycentersArray[b_i, 0]-elementBarycentersArray[i, 0]
            normal_check[1] = elementBoundaryBarycentersArray[b_i, 1]-elementBarycentersArray[i, 1]
            dot = elementBoundaryNormalsArray_[i,j,0]*normal_check[0]+\
                  elementBoundaryNormalsArray_[i,j,1]*normal_check[1]
            if dot < 0:
                elementBoundaryNormalsArray_[i,j,0] = -elementBoundaryNormalsArray_[i,j,0]
                elementBoundaryNormalsArray_[i,j,1] = -elementBoundaryNormalsArray_[i,j,1]

cdef void cyUpdateElementVolumesTriangle(double[:] elementVolumesArray_,
                                         proteus_index_t[:,:] elementNodesArray,
                                         double[:,:] nodeArray,
                                         proteus_index_t nElements):
    cdef double[:] nA
    cdef double[:] nB
    cdef double[:] nC
    cdef double base
    cdef double height
    cdef proteus_index_t eN
    for eN in range(nElements):
        nA = nodeArray[elementNodesArray[eN, 0]]
        nB = nodeArray[elementNodesArray[eN, 1]]
        nC = nodeArray[elementNodesArray[eN, 2]]
        base = sqrt((nB[1]-nA[1])**2+(nB[0]-nA[0])**2)
        height = abs((nB[1]-nA[1])*nC[0]-(nB[0]-nA[0])*nC[1]+nB[0]*nA[1]-nB[1]*nA[0])/base
        elementVolumesArray_[eN] = 0.5*base*height

cdef double cyGetElementVolumeTriangle(double[:] nA,
                                       double[:] nB,
                                       double[:] nC):
    cdef double base = sqrt((nB[1]-nA[1])**2+(nB[0]-nA[0])**2)
    cdef double height = abs((nB[1]-nA[1])*nC[0]-(nB[0]-nA[0])*nC[1]+nB[0]*nA[1]-nB[1]*nA[0])/base
    return 0.5*base*height

cdef void cyUpdateElementVolumesTetra(double[:] elementVolumesArray_,
                                      proteus_index_t[:,:] elementNodesArray,
                                      double[:,:] nodeArray,
                                      proteus_index_t nElements):
    cdef double[:] nA
    cdef double[:] nB
    cdef double[:] nC
    cdef double[:] nD
    cdef double base_tri
    cdef double height_tri
    cdef double area_tri
    cdef double height_tetra
    cdef proteus_index_t eN
    for eN in range(nElements):
        nA = nodeArray[elementNodesArray[eN, 0]]
        nB = nodeArray[elementNodesArray[eN, 1]]
        nC = nodeArray[elementNodesArray[eN, 2]]
        nD = nodeArray[elementNodesArray[eN, 3]]
        base_tri = sqrt((nB[1]-nA[1])**2+(nB[0]-nA[0])**2)
        height_tri = abs((nB[1]-nA[1])*nC[0]-(nB[0]-nA[0])*nC[1]+nB[0]*nA[1]-nB[1]*nA[0])/base_tri
        area_tri = 0.5*base_tri*height_tri
        height_tetra = 0.
        elementVolumesArray_[eN] = 1./3.*area_tri*height_tetra

cdef void cyUpdateElementBarycenters(double[:,:] elementBarycentersArray_,
                                     proteus_index_t[:,:] elementNodesArray,
                                     double[:,:] nodeArray,
                                     proteus_index_t nElements):
    cdef proteus_index_t eN
    cdef proteus_index_t iN
    cdef proteus_index_t node
    cdef proteus_index_t nNel = elementNodesArray.shape[1]
    for eN in range(nElements):
        elementBarycentersArray_[eN, 0] = 0.
        elementBarycentersArray_[eN, 1] = 0.
        elementBarycentersArray_[eN, 2] = 0.
        for iN in range(nNel):
            node = elementNodesArray[eN, iN]
            elementBarycentersArray_[eN, 0] += nodeArray[node, 0]
            elementBarycentersArray_[eN, 1] += nodeArray[node, 1]
            elementBarycentersArray_[eN, 2] += nodeArray[node, 2]
        elementBarycentersArray_[eN, 0] = elementBarycentersArray_[eN, 0]/(iN+1)
        elementBarycentersArray_[eN, 1] = elementBarycentersArray_[eN, 1]/(iN+1)
        elementBarycentersArray_[eN, 2] = elementBarycentersArray_[eN, 2]/(iN+1)

cdef np.ndarray cyGetCornerNodesTriangle(double[:,:] nodeArray,
                                         proteus_index_t[:] nodeStarArray,
                                         proteus_index_t[:] nodeStarOffsets,
                                         int[:] nodeMaterialTypes,
                                         proteus_index_t nNodes):
    cdef np.ndarray cornerNodesArray = np.array([], dtype=np.int32)
    cdef double[3] vec
    cdef double[3] vec2
    cdef double vec_dist
    cdef double dot
    cdef proteus_index_t node
    cdef proteus_index_t nOffset
    for node in range(nNodes):
        if nodeMaterialTypes[node] != 0:
            vec[0] = 0.
            vec[1] = 0.
            vec[2] = 0.
            for nOffset in range(nodeStarOffsets[node],
                                 nodeStarOffsets[node+1]):
                if nodeMaterialTypes[nodeStarArray[nOffset]] != 0:
                    if vec[0] == 0. and vec[1] == 0. and vec[2] == 0.:
                        # initialize first vector
                        vec[0] = nodeArray[node, 0]-nodeArray[nodeStarArray[nOffset], 0]
                        vec[1] = nodeArray[node, 1]-nodeArray[nodeStarArray[nOffset], 1]
                        vec[2] = nodeArray[node, 2]-nodeArray[nodeStarArray[nOffset], 2]
                        vec_dist = sqrt(vec[0]**2+vec[1]**2+vec[2]**2)
                        vec[0] = vec[0]/vec_dist
                        vec[1] = vec[1]/vec_dist
                        vec[2] = vec[2]/vec_dist
                    else:
                        vec2[0] = nodeArray[node, 0]-nodeArray[nodeStarArray[nOffset], 0]
                        vec2[1] = nodeArray[node, 1]-nodeArray[nodeStarArray[nOffset], 1]
                        vec2[2] = nodeArray[node, 2]-nodeArray[nodeStarArray[nOffset], 2]
                        vec_dist = sqrt(vec2[0]**2+vec2[1]**2+vec2[2]**2)
                        vec2[0] = vec2[0]/vec_dist
                        vec2[1] = vec2[1]/vec_dist
                        vec2[2] = vec2[2]/vec_dist
                        dot = vec[0]*vec2[0]+vec[1]*vec2[1]+vec[2]*vec2[2]
                        if dot == 1. or dot == -1.:
                            dot = 1
                        else:
                            cornerNodesArray = np.append(cornerNodesArray, node)
    return cornerNodesArray

cdef proteus_index_t[:] cyCheckOwnedVariable(proteus_index_t variable_nb_local,
                                 int rank,
                                 proteus_index_t nVariables_owned,
                                 proteus_index_t[:] variableNumbering_subdomain2global,
                                 proteus_index_t[:] variableOffsets_subdomain_owned):
    cdef int nSubdomains = variableOffsets_subdomain_owned.shape[0]-1
    cdef proteus_index_t variable_nb_global
    cdef proteus_index_t new_variable_nb_local
    cdef int new_rank = -2  # initialised as fake rank
    cdef int i
    cdef proteus_index_t[2] result = [-1000, -1000]
    # cdef int[:] result = np.zeros(2, dtype=np.int32)
    result[0] = variable_nb_local
    result[1] = new_rank
    if variable_nb_local >= nVariables_owned:
        # change rank ownership
        variable_nb_global = variableNumbering_subdomain2global[variable_nb_local]
        if not variableOffsets_subdomain_owned[rank] <= variable_nb_global < variableOffsets_subdomain_owned[rank+1]:
            for i in range(nSubdomains+1):
                if variableOffsets_subdomain_owned[i] > variable_nb_global:
                    # changing processor
                    if new_rank == -2:
                        new_rank = i-1
    # getting nearest variable number on new rank
    if new_rank >= 0:
        new_variable_nb_local = variable_nb_global-variableOffsets_subdomain_owned[new_rank]
    else:
        new_rank = rank
        new_variable_nb_local = variable_nb_local
    result[0] = new_variable_nb_local
    result[1] = new_rank
    return result

cdef void cyFindBoundaryDirectionTriangle(
    double[:] dir_,
    proteus_index_t node,
    double[:,:] nodeArray,
    proteus_index_t[:] nodeStarOffsets,
    proteus_index_t[:] nodeStarArray,
    int[:] nodeMaterialTypes,
):
    cdef double dir_dist
    cdef proteus_index_t nOffset
    for nOffset in range(nodeStarOffsets[node],
                            nodeStarOffsets[node+1]):
        if nodeMaterialTypes[nodeStarArray[nOffset]] != 0:
            dir_[0] = nodeArray[node, 0]-nodeArray[nodeStarArray[nOffset], 0]
            dir_[1] = nodeArray[node, 1]-nodeArray[nodeStarArray[nOffset], 1]
            dir_[2] = nodeArray[node, 2]-nodeArray[nodeStarArray[nOffset], 2]
            dir_dist = sqrt(dir_[0]**2+dir_[1]**2+dir_[2]**2)
            dir_[0] = abs(dir_[0])/dir_dist
            dir_[1] = abs(dir_[1])/dir_dist
            dir_[2] = abs(dir_[2])/dir_dist

cdef void cyFindBoundaryDirectionTetra(
    double[:] dir_,
    proteus_index_t node,
    double[:,:] nodeArray,
    proteus_index_t[:] nodeStarOffsets,
    proteus_index_t[:] nodeStarArray,
    int[:] nodeMaterialTypes,
):
    cdef double[3] U
    cdef double[3] V
    cdef double dir_dist
    cdef proteus_index_t b_i
    cdef double[:] node0
    cdef double[:] node1
    cdef double[:] node2
    cdef double nNode = 0
    cdef proteus_index_t nOffset
    # get normal
    for nOffset in range(nodeStarOffsets[node],
                         nodeStarOffsets[node+1]):
        if nodeMaterialTypes[nodeStarArray[nOffset]] != 0:
            nNode += 1
            if nNode == 1:
                node0 = nodeArray[nodeStarArray[nOffset]]
            elif nNode == 2:
                node1 = nodeArray[nodeStarArray[nOffset]]
            elif nNode == 3:
                node2 = nodeArray[nodeStarArray[nOffset]]
    assert nNode > 3, 'error looking for dir_'
    U[0] = node1[0]-node0[0]
    U[1] = node1[1]-node0[1]
    U[2] = node1[2]-node0[2]
    V[0] = node2[0]-node0[0]
    V[1] = node2[1]-node0[1]
    V[2] = node2[2]-node0[2]
    dir_[0] = U[1]*V[2]-U[2]*V[1]
    dir_[1] = U[2]*V[0]-U[0]*V[2]
    dir_[2] = U[0]*V[1]-U[1]*V[0]
    dir_dist = sqrt(dir_[0]**2+dir_[1]**2+dir_[2]**2)
    dir_[0] /= dir_dist
    dir_[1] /= dir_dist
    dir_[2] /= dir_dist
    dir_[0] = abs(1-dir_[0])
    dir_[1] = abs(1-dir_[1])
    dir_[2] = abs(1-dir_[2])

cdef proteus_index_t[:] cyGetGlobalVariable(proteus_index_t variable_nb_local,
                                proteus_index_t nVariables_owned,
                                proteus_index_t[:] variableNumbering_subdomain2global,
                                proteus_index_t[:] variableOffsets_subdomain_owned):
    cdef int nSubdomains = len(variableOffsets_subdomain_owned)-1
    cdef proteus_index_t variable_nb_global
    cdef int new_rank = -2  # initialised as fake rank
    # change rank ownership
    variable_nb_global = variableNumbering_subdomain2global[variable_nb_local]
    cdef int i
    cdef proteus_index_t[2] result
    # cdef int[:] result = np.zeros(2, dtype=np.int32)
    result[0] = variable_nb_global
    result[1] = new_rank
    for i in range(nSubdomains+1):
        if variableOffsets_subdomain_owned[i] > variable_nb_global:
            # changing processor
            if new_rank == -2:
                new_rank = i-1
    variable_nb_global = variableNumbering_subdomain2global[variable_nb_local]
    result[0] = variable_nb_global
    result[1] = new_rank
    return result

cdef int cyGetLocalVariable(proteus_index_t variable_nb_global,
                            int rank,
                            proteus_index_t nVariables_owned,
                            proteus_index_t[:] variableNumbering_subdomain2global,
                            proteus_index_t[:] variableOffsets_subdomain_owned):
    cdef proteus_index_t new_variable_nb_local
    if not variableOffsets_subdomain_owned[rank] <= variable_nb_global < variableOffsets_subdomain_owned[rank+1]:
        if variable_nb_global < nVariables_owned:
            new_variable_nb_local = variable_nb_global
    else:
        new_variable_nb_local = variable_nb_global-variableOffsets_subdomain_owned[rank]
    return new_variable_nb_local

cdef double[:] cyScalarRecoveryAtNodes(double[:] scalars,
                                       proteus_index_t[:] nodeElementsArray,
                                       proteus_index_t[:] nodeElementOffsets):
    cdef double[:] recovered_scalars = np.zeros(len(nodeElementOffsets)-1)
    cdef proteus_index_t nb_el
    cdef double var_sum
    cdef proteus_index_t node
    cdef proteus_index_t eOffset
    cdef proteus_index_t nNodes = nodeElementOffsets.shape[0]-1
    for node in range(nNodes):
        nb_el = 0
        var_sum = 0
        for eOffset in range(nodeElementOffsets[node],
                             nodeElementOffsets[node+1]):
            nb_el += 1
            var_sum += scalars[nodeElementsArray[eOffset]]
        recovered_scalars[node] = var_sum/nb_el
    return recovered_scalars

cdef double[:] cyScalarRecoveryAtNodesWeighted(double[:] scalars,
                                               proteus_index_t[:] nodeElementsArray,
                                               proteus_index_t[:] nodeElementOffsets,
                                               double[:] detJ_array,
                                               proteus_index_t nNodes):
    cdef double[:] recovered_scalars = np.zeros(nNodes)
    cdef double detJ_patch = 0.
    cdef double scalar_sum = 0.
    cdef proteus_index_t nb_el
    cdef proteus_index_t node
    cdef proteus_index_t eOffset
    cdef proteus_index_t eN
    for node in range(nNodes):
        nb_el = 0
        detJ_patch = 0.
        scalar_sum = 0.
        for eOffset in range(nodeElementOffsets[node],
                             nodeElementOffsets[node+1]):
            nb_el += 1
            eN = nodeElementsArray[eOffset]
            # for k in range(n_quad):
            #     scalar_k = gradrads[eN, k]
            #     scalar_eN_av += gradrad_k
            # scalar_eN_av /= n_quad
            detJ_patch += detJ_array[eN]
            scalar_sum += detJ_array[eN]*scalars[eN]  # same value at all quad points
        recovered_scalars[node] = scalar_sum/detJ_patch
    return recovered_scalars

cdef double[:,:] cyVectorRecoveryAtNodes(double[:,:] vectors,
                                         proteus_index_t[:] nodeElementsArray,
                                         proteus_index_t[:] nodeElementOffsets,
                                         int nd):
    cdef double[:, :] recovered_vectors = np.zeros((nodeElementOffsets.shape[0]-1, nd))
    cdef double[:] vector_sum = np.zeros(nd)
    cdef proteus_index_t nNodes = nodeElementOffsets.shape[0]-1
    cdef proteus_index_t nb_el
    cdef proteus_index_t eOffset
    cdef proteus_index_t eN
    cdef int ndi
    cdef proteus_index_t node
    for node in range(nNodes):
        nb_el = 0
        for ndi in range(nd):
            vector_sum[ndi] = 0.
        for eOffset in range(nodeElementOffsets[node],
                             nodeElementOffsets[node+1]):
            nb_el += 1
            eN = nodeElementsArray[eOffset]
            # for k in range(n_quad):
            #     vector_k = vectors[eN, k]
            #     vector_eN_av += vector_k
            # vector_eN_av /= n_quad
            for ndi in range(nd):
                vector_sum[ndi] += vectors[eN, ndi]  # same value at all quad points
        for ndi in range(nd):
            recovered_vectors[node, ndi] = vector_sum[ndi]/nb_el
    return recovered_vectors

cdef double[:,:] cyVectorRecoveryAtNodesWeighted(double[:,:] vectors,
                                                 proteus_index_t[:] nodeElementsArray,
                                                 proteus_index_t[:] nodeElementOffsets,
                                                 double[:,:] detJ_array,
                                                 int nd):
    cdef double[:, :] recovered_vectors = np.zeros((len(nodeElementOffsets)-1, nd))
    cdef double[:] vector_sum = np.zeros(nd)
    cdef proteus_index_t nNodes = nodeElementOffsets.shape[0]-1
    cdef double detJ_patch = 0.
    cdef proteus_index_t nb_el
    cdef proteus_index_t eN
    cdef int ndi
    cdef proteus_index_t node
    cdef proteus_index_t eOffset
    for node in range(nNodes):
        nb_el = 0
        detJ_patch = 0.
        for ndi in range(nd):
            vector_sum[ndi] = 0.
        for eOffset in range(nodeElementOffsets[node],
                             nodeElementOffsets[node+1]):
            nb_el += 1
            eN = nodeElementsArray[eOffset]
            # for k in range(n_quad):
            #     vector_k = vectors[eN, k]
            #     vector_eN_av += vector_k
            # vector_eN_av /= n_quad
            detJ_patch += detJ_array[eN,0]
            for ndi in range(nd):
                vector_sum[ndi] += detJ_array[eN,0]*vectors[eN, ndi]  # same value at all quad points
        for ndi in range(nd):
            recovered_vectors[node, ndi] = vector_sum[ndi]/detJ_patch
    return recovered_vectors
