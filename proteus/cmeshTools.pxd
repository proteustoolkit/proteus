# A type of -*- python -*- file
# cython: language_level=3
cimport numpy as np
cimport mesh as cppm

cdef class CMesh:
    cdef cppm.Mesh mesh
    cdef public:
        cdef cppm.proteus_index_t nElements_global
        cdef cppm.proteus_index_t nNodes_global
        cdef int nNodes_element
        cdef int nNodes_elementBoundary
        cdef int nElementBoundaries_element,
        cdef cppm.proteus_index_t nElementBoundaries_global,
        cdef cppm.proteus_index_t nInteriorElementBoundaries_global,
        cdef cppm.proteus_index_t nExteriorElementBoundaries_global,
        cdef cppm.proteus_index_t max_nElements_node,
        cdef cppm.proteus_index_t nEdges_global,
        cdef cppm.proteus_index_t max_nNodeNeighbors_node,
        # cdef int[:,:] elementNodesArray,
        # cdef int[:] nodeElementsArray,
        # cdef int[:] nodeElementOffsets,
        # cdef int[:,:] elementNeighborsArray,
        # cdef int[:,:] elementBoundariesArray,
        # cdef int[:,:] elementBoundaryNodesArray,
        # cdef int[:,:] elementBoundaryElementsArray,
        # cdef int[:,:] elementBoundaryLocalElementBoundariesArray,
        # cdef int[:] interiorElementBoundariesArray,
        # cdef int[:] exteriorElementBoundariesArray,
        # cdef int[:,:] edgeNodesArray,
        # cdef int[:] nodeStarArray,
        # cdef int[:] nodeStarOffsets,
        # cdef int[:] elementMaterialTypes,
        # cdef int[:] elementBoundaryMaterialTypes,
        # cdef int[:] nodeMaterialTypes,
        # cdef double[:,:] nodeArray,

        cdef np.ndarray elementNodesArray,
        cdef np.ndarray nodeElementsArray,
        cdef np.ndarray nodeElementOffsets,
        cdef np.ndarray elementNeighborsArray,
        cdef np.ndarray elementBoundariesArray,
        cdef np.ndarray elementBoundaryNodesArray,
        cdef np.ndarray elementBoundaryElementsArray,
        cdef np.ndarray elementBoundaryLocalElementBoundariesArray,
        cdef np.ndarray interiorElementBoundariesArray,
        cdef np.ndarray exteriorElementBoundariesArray,
        cdef np.ndarray edgeNodesArray,
        cdef np.ndarray nodeStarArray,
        cdef np.ndarray nodeStarOffsets,
        cdef np.ndarray elementMaterialTypes,
        cdef np.ndarray elementBoundaryMaterialTypes,
        cdef np.ndarray nodeMaterialTypes,
        cdef object edgeMaterialTypes,
        cdef np.ndarray nodeArray,

        cdef int nx
        cdef int ny
        cdef int nz      #NURBS
        cdef int px
        cdef int py
        cdef int pz,      #NURBS
        # cdef int[:] elementIJK, #NURBS
        # cdef double[:] weights,    #NURBS
        # cdef double[:] U_KNOT,     #NURBS
        # cdef double[:] V_KNOT,     #NURBS
        # cdef double[:] W_KNOT,     #NURBS
        # cdef double[:] elementDiametersArray,
        # cdef double[:] elementInnerDiametersArray,
        # cdef double[:] elementBoundaryDiametersArray,
        # cdef double[:,:] elementBarycentersArray,
        # cdef double[:,:] elementBoundaryBarycentersArray,
        # cdef double[:] nodeDiametersArray,
        # cdef double[:] nodeSupportArray,

        cdef np.ndarray elementIJK, #NURBS
        cdef np.ndarray weights,    #NURBS
        cdef np.ndarray U_KNOT,     #NURBS
        cdef np.ndarray V_KNOT,     #NURBS
        cdef np.ndarray W_KNOT,     #NURBS
        cdef np.ndarray elementDiametersArray,
        cdef np.ndarray elementInnerDiametersArray,
        cdef np.ndarray elementBoundaryDiametersArray,
        cdef np.ndarray elementBarycentersArray,
        cdef np.ndarray elementBoundaryBarycentersArray,
        cdef np.ndarray nodeDiametersArray,
        cdef np.ndarray nodeSupportArray,

        cdef double h,
        cdef double hMin,
        cdef double sigmaMax,
        cdef double volume
