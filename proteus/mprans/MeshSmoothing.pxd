cdef extern from "index_type.h":
    ctypedef int proteus_index_t

# cython: language_level=3
cimport cython
import numpy as np
cimport numpy as np
from libcpp cimport bool

cdef void cySmoothNodesLaplace(double[:,:] nodeArray_,
                               proteus_index_t[:] nodeStarOffsets,
                               proteus_index_t[:] nodeStarArray,
                               int[:] nodeMaterialTypes,
                               proteus_index_t nNodes_owned,
                               int nd,
                               bool simultaneous=*,
                               bool smoothBoundaries=*,
                               int[:] fixedNodesBoolArray=*,
                               double alpha=*)

cdef void cySmoothNodesCentroid(double[:,:] nodeArray_,
                                proteus_index_t[:] nodeElementOffsets,
                                proteus_index_t[:] nodeElementsArray,
                                int[:] nodeMaterialTypes,
                                double[:] elementVolumesArray,
                                double[:,:] elementBarycentersArray,
                                proteus_index_t[:,:] elementNodesArray,
                                proteus_index_t nNodes_owned,
                                int[:] fixedNodesBoolArray,
                                bool simultaneous=*,
                                bool smoothBoundaries=*,
                                double alpha=*)

cdef void cyUpdateDilationElements(double[:] elementDilationArray_,
                                   double[:] elementVolumeArray,
                                   double[:] elementVolumeTargetArray,
                                   proteus_index_t nElements)

cdef void cyUpdateDistortionElements(double[:] elementDistortionArray_,
                                     double[:,:,:,:] J_array,
                                     double[:,:] detJ_array,
                                     int nd,
                                     proteus_index_t nElements)

cdef void cyUpdateInverseMeanRatioTriangleNodes(double[:] IMRNodesArray_,
                                                double[:,:] nodeArray,
                                                proteus_index_t[:,:] elementNodesArray,
                                                proteus_index_t[:] nodeElementOffsets,
                                                proteus_index_t[:] nodeElementsArray,
                                                proteus_index_t nElements,
                                                proteus_index_t nNodes,
                                                bool el_average=*)

cdef void cyUpdateInverseMeanRatioTriangleElements(double[:] IMRElementsArray_,
                                                   double[:,:] nodeArray,
                                                   proteus_index_t[:,:] elementNodesArray,
                                                   proteus_index_t nElements)

cdef double cyGetInverseMeanRatioSingleTriangle(proteus_index_t node0,
                                                double[:,:] nodeArray,
                                                proteus_index_t[:,:] elementNodesArray,
                                                proteus_index_t[:] nodeElementOffsets,
                                                proteus_index_t[:] nodeElementsArray,
                                                bool el_average=*)

cdef double[:,:] cySmoothNodesQuality(double[:] distortion,
                                      double[:] dilation,
                                      double[:,:] nodeArray,
                                      proteus_index_t nNodes_owned,
                                      int[:] nodeMaterialTypes,
                                      proteus_index_t[:] nodeElementOffsets,
                                      proteus_index_t[:] nodeElementsArray,
                                      proteus_index_t[:,:] elementNodesArray,
                                      bool apply_directly=*)

cdef int pyxGetLocalNearestNode(double[:] coords,
                                double[:,:] nodeArray,
                                proteus_index_t[:] nodeStarOffsets,
                                proteus_index_t[:] nodeStarArray,
                                proteus_index_t node)

cdef int pyxGetLocalNearestElement(double[:] coords,
                                   double[:,:] elementBarycentersArray,
                                   proteus_index_t[:,:] elementNeighborsArray,
                                   proteus_index_t eN)

cdef int[:] pyxGetLocalNearestElementIntersection(double[:] coords,
                                                  double[:] starting_coords,
                                                  double[:,:,:] elementBoundaryNormalsArray,
                                                  proteus_index_t[:,:] elementBoundariesArray,
                                                  double[:,:] elementBoundaryBarycentersArray,
                                                  proteus_index_t[:,:] elementBoundaryElementsArray,
                                                  int[:] exteriorElementBoundariesBoolArray,
                                                  proteus_index_t eN)

cdef int pyxGetLocalNearestElementAroundNode(double[:] coords,
                                             proteus_index_t[:] nodeElementOffsets,
                                             proteus_index_t[:] nodeElementsArray,
                                             double[:,:] elementBarycentersArray,
                                             proteus_index_t node)

cdef void pyxUpdateElementBoundaryNormalsTetra(double[:,:,:] elementBoundaryNormalsArray_,
                                               double[:,:] nodeArray,
                                               proteus_index_t[:,:] elementBoundariesArray,
                                               proteus_index_t[:,:] elementBoundaryNodesArray,
                                               double[:,:] elementBoundaryBarycentersArray,
                                               double[:,:] elementBarycentersArray,
                                               proteus_index_t nElements)

cdef void pyxUpdateElementBoundaryNormalsTriangle(double[:,:,:] elementBoundaryNormalsArray_,
                                                  double[:,:] nodeArray,
                                                  proteus_index_t[:,:] elementBoundariesArray,
                                                  proteus_index_t[:,:] elementBoundaryNodesArray,
                                                  double[:,:] elementBoundaryBarycentersArray,
                                                  double[:,:] elementBarycentersArray,
                                                  proteus_index_t nElements)

cdef void cyUpdateElementVolumesTriangle(double[:] elementVolumesArray_,
                                         proteus_index_t[:,:] elementNodesArray,
                                         double[:,:] nodeArray,
                                         proteus_index_t nElements)

cdef void cyUpdateElementVolumesTetra(double[:] elementVolumesArray_,
                                      proteus_index_t[:,:] elementNodesArray,
                                      double[:,:] nodeArray,
                                      proteus_index_t nElements)

cdef void cyUpdateElementBarycenters(double[:,:] elementBarycentersArray_,
                                     proteus_index_t[:,:] elementNodesArray,
                                     double[:,:] nodeArray,
                                     proteus_index_t nElements)

cdef np.ndarray cyGetCornerNodesTriangle(double[:,:] nodeArray,
                                         proteus_index_t[:] nodeStarArray,
                                         proteus_index_t[:] nodeStarOffsets,
                                         int[:] nodeMaterialTypes,
                                         proteus_index_t nNodes)

cdef int[:] cyCheckOwnedVariable(proteus_index_t variable_nb_local,
                                 int rank,
                                 proteus_index_t nVariables_owned,
                                 proteus_index_t[:] variableNumbering_subdomain2global,
                                 proteus_index_t[:] variableOffsets_subdomain_owned)

cdef int[:] cyGetGlobalVariable(proteus_index_t variable_nb_local,
                                proteus_index_t nVariables_owned,
                                proteus_index_t[:] variableNumbering_subdomain2global,
                                proteus_index_t[:] variableOffsets_subdomain_owned)

cdef int cyGetLocalVariable(proteus_index_t variable_nb_global,
                            int rank,
                            proteus_index_t nVariables_owned,
                            proteus_index_t[:] variableNumbering_subdomain2global,
                            proteus_index_t[:] variableOffsets_subdomain_owned)

cdef double[:] cyScalarRecoveryAtNodes(double[:] scalars,
                                       proteus_index_t[:] nodeElementsArray,
                                       proteus_index_t[:] nodeElementOffsets)

cdef double[:] cyScalarRecoveryAtNodesWeighted(double[:] scalars,
                                               proteus_index_t[:] nodeElementsArray,
                                               proteus_index_t[:] nodeElementOffsets,
                                               double[:] detJ_array,
                                               proteus_index_t nNodes)

cdef double[:,:] cyVectorRecoveryAtNodesWeighted(double[:,:] vectors,
                                                 proteus_index_t[:] nodeElementsArray,
                                                 proteus_index_t[:] nodeElementOffsets,
                                                 double[:,:] detJ_array,
                                                 int nd)

cdef double[:,:] cyVectorRecoveryAtNodes(double[:,:] vectors,
                                         proteus_index_t[:] nodeElementsArray,
                                         proteus_index_t[:] nodeElementOffsets,
                                         int nd)

cdef void cyFindBoundaryDirectionTriangle(double[:] dir_,
                                          proteus_index_t node,
                                          double[:,:] nodeArray,
                                          proteus_index_t[:] nodeStarOffsets,
                                          proteus_index_t[:] nodeStarArray,
                                          int[:] nodeMaterialTypes)

cdef void cyFindBoundaryDirectionTetra(double[:] dir_,
                                       proteus_index_t node,
                                       double[:,:] nodeArray,
                                       proteus_index_t[:] nodeStarOffsets,
                                       proteus_index_t[:] nodeStarArray,
                                       int[:] nodeMaterialTypes)
