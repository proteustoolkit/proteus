from libcpp cimport bool

cdef extern from "index_type.h":
    ctypedef int proteus_index_t

cdef extern from "mesh.h":
    cdef struct Mesh:
      proteus_index_t nElements_global
      proteus_index_t nNodes_global
      proteus_index_t nNodes_element
      proteus_index_t nNodes_elementBoundary
      proteus_index_t nElementBoundaries_element
      proteus_index_t nElementBoundaries_global
      proteus_index_t nInteriorElementBoundaries_global
      proteus_index_t nExteriorElementBoundaries_global
      proteus_index_t max_nElements_node
      proteus_index_t nEdges_global
      proteus_index_t max_nNodeNeighbors_node

      proteus_index_t *elementNodesArray
      proteus_index_t *nodeElementsArray
      proteus_index_t *nodeElementOffsets
      proteus_index_t *elementNeighborsArray
      proteus_index_t *elementBoundariesArray
      proteus_index_t *elementBoundaryNodesArray
      proteus_index_t *elementBoundaryElementsArray
      int *elementBoundaryLocalElementBoundariesArray
      proteus_index_t *interiorElementBoundariesArray
      proteus_index_t *exteriorElementBoundariesArray
      proteus_index_t *edgeNodesArray
      proteus_index_t *nodeStarArray
      proteus_index_t *nodeStarOffsets
      int *elementMaterialTypes
      int *elementBoundaryMaterialTypes
      int *nodeMaterialTypes
      int *edgeMaterialTypes

      int *elementIJK
      double *weights
      double *U_KNOT
      double *V_KNOT
      double *W_KNOT
      int nx,ny,nz
      int px,py,pz

      double *nodeArray,
      double *elementDiametersArray
      double *elementInnerDiametersArray
      double *elementBoundaryDiametersArray
      double *elementBarycentersArray
      double *elementBoundaryBarycentersArray
      double *nodeDiametersArray
      double *nodeSupportArray
      double h,hMin,sigmaMax,volume
      int * newestNodeBases

      proteus_index_t *elementOffsets_subdomain_owned
      proteus_index_t *elementNumbering_subdomain2global
      proteus_index_t *nodeOffsets_subdomain_owned
      proteus_index_t *nodeNumbering_subdomain2global
      proteus_index_t *nodeNumbering_global_old2new
      proteus_index_t *elementNumbering_global_old2new
      proteus_index_t *elementBoundaryOffsets_subdomain_owned
      proteus_index_t *elementBoundaryNumbering_subdomain2global
      proteus_index_t *edgeOffsets_subdomain_owned
      proteus_index_t *edgeNumbering_subdomain2global
      Mesh* subdomainp

    cdef void initializeMesh(Mesh& mesh)

    cdef void deleteMesh(Mesh& mesh)

    cdef struct MultilevelMesh:
        int nLevels
        Mesh* meshArray
        int **elementParentsArray
        int **elementChildrenArray
        int **elementChildrenOffsets

    cdef void initializeMultilevelMesh(MultilevelMesh& multilevelMesh)

    cdef void deleteMultilevelMesh(MultilevelMesh& multilevelMesh)

    cdef int edgeMeshElements(const int& nx,
                              Mesh& mesh)
    cdef int regularEdgeMeshNodes(const int& nx,
                                  const double& Lx,
                                  Mesh& mesh)
    cdef int globallyRefineEdgeMesh(const int& nLevels,
                                    Mesh& mesh,
                                    MultilevelMesh& multilevelMesh)
    cdef int globallyRefineEdgeMesh(const int& nLevels,
                                    Mesh& mesh,
                                    MultilevelMesh& multilevelMesh,
                                    bool averageNewNodeFlags)
    cdef int locallyRefineEdgeMesh(MultilevelMesh& multilevelMesh,
			                       int * elementTagArray)
    cdef int locallyRefineTriangleMesh(MultilevelMesh& multilevelMesh,
				                       int * elementTagArray)
    cdef int locallyRefineTriangleMesh_4T(MultilevelMesh& multilevelMesh,
				                          int * elementTagArray)
    cdef int locallyRefineTriangleMesh_redGreen(MultilevelMesh& multilevelMesh,
				                              int * elementTagArray)
    cdef int setNewestNodeBasesToLongestEdge(MultilevelMesh& multilevelMesh)

    cdef int regularRectangularToTriangularMeshElements(const int& nx,
                                                        const int& ny,
                                                        Mesh& mesh,
                                                        int triangleFlag)
    cdef int regularRectangularToTriangularMeshNodes(const int& nx,
                                                     const int& ny,
                                                     const double& Lx,
                                                     const double& Ly,
                                                     Mesh& mesh)
    cdef int regularRectangularToTriangularElementBoundaryMaterials(const double& Lx,
                                                                    const double& Ly,
                                                                    Mesh& mesh)
    cdef int globallyRefineTriangularMesh(const int& nLevels,
                                          Mesh& mesh,
                                          MultilevelMesh& multilevelMesh)
    cdef int globallyRefineTriangularMesh(const int& nLevels,
                                          Mesh& mesh,
                                          MultilevelMesh& multilevelMesh,
                                          bool averageNewNodeFlags)

    cdef int regularQuadrilateralMeshElements(const int& nx,
                                              const int& ny,
                                              Mesh& mesh)
    cdef int regularQuadrilateralMeshElementBoundaryMaterials(const double& Lx,
                                                              const double& Ly,
                                                              Mesh& mesh)
    cdef int globallyRefineQuadrilateralMesh(const int& nLevels,
                                             Mesh& mesh,
                                             MultilevelMesh& multilevelMesh)
    cdef int globallyRefineQuadrilateralMesh(const int& nLevels,
                                             Mesh& mesh,
                                             MultilevelMesh& multilevelMesh,
                                             bool averageNewNodeFlags)

    cdef int regularMeshNodes(const int& nx,
                              const int& ny,
                              const int& nz,
                              const double& Lx,
                              const double& Ly,
                              const double& Lz,
                              Mesh& mesh)
    cdef int regularMeshNodes2D(const int& nx,
                                const int& ny,
                                const double& Lx,
                                const double& Ly,
                                Mesh& mesh)
    cdef int regularHexahedralMeshElementBoundaryMaterials(const double& Lx,
                                                           const double& Ly,
                                                           const double& Lz,
                                                           Mesh& mesh)
    cdef int regularHexahedralToTetrahedralMeshNodes(const int& nx,
                                                     const int& ny,
                                                     const int& nz,
                                                     const double& Lx,
                                                     const double& Ly,
                                                     const double& Lz,
                                                     Mesh& mesh)
    cdef int regularHexahedralToTetrahedralMeshElements(const int& nx,
                                                        const int& ny,
                                                        const int& nz,
                                                        Mesh& mesh)
    cdef int regularHexahedralToTetrahedralElementBoundaryMaterials(const double& Lx,
                                                                    const double& Ly,
                                                                    const double& Lz,
                                                                    Mesh& mesh)
    cdef int reorientTetrahedralMesh(Mesh& mesh)
    cdef int regularHexahedralMeshElements(const int& nx,
                                           const int& ny,
                                           const int& nz,
                                           const int& px,
                                           const int& py,
                                           const int& pz,
                                           Mesh& mesh)
    cdef int regularNURBSMeshElements(const int& nx,
                                      const int& ny,
                                      const int& nz,
                                      const int& px,
                                      const int& py,
                                      const int& pz,
                                      Mesh& mesh)
    cdef int globallyRefineHexahedralMesh(const int& nLevels,
                                          Mesh& mesh,
                                          MultilevelMesh& multilevelMesh)
    cdef int globallyRefineHexahedralMesh(const int& nLevels,
                                          Mesh& mesh,
                                          MultilevelMesh& multilevelMesh,
                                          bool averageNewNodeFlags)

    cdef int globallyRefineTetrahedralMesh(const int& nLevels,
                                           Mesh& mesh,
                                           MultilevelMesh& multilevelMesh)
    cdef int globallyRefineTetrahedralMesh(const int& nLevels,
                                           Mesh& mesh,
                                           MultilevelMesh& multilevelMesh,
                                           bool averageNewNodeFlags)

    cdef int constructElementBoundaryElementsArray_edge(Mesh& mesh)
    cdef int constructElementBoundaryElementsArray_triangle(Mesh& mesh)
    cdef int constructElementBoundaryElementsArray_quadrilateral(Mesh& mesh)
    cdef int constructElementBoundaryElementsArray_tetrahedron(Mesh& mesh)
    cdef int constructElementBoundaryElementsArray_hexahedron(Mesh& mesh)
    cdef int constructElementBoundaryElementsArray_NURBS(Mesh& mesh)

    cdef int constructElementBoundaryElementsArrayWithGivenElementBoundaryNumbers_edge(Mesh& mesh)
    cdef int constructElementBoundaryElementsArrayWithGivenElementBoundaryNumbers_triangle(Mesh& mesh)
    cdef int constructElementBoundaryElementsArrayWithGivenElementBoundaryNumbers_quadrilateral(Mesh& mesh)
    cdef int constructElementBoundaryElementsArrayWithGivenElementBoundaryNumbers_tetrahedron(Mesh& mesh)

    cdef int constructElementBoundaryElementsArrayWithGivenElementBoundaryAndEdgeNumbers_edge(Mesh& mesh)
    cdef int constructElementBoundaryElementsArrayWithGivenElementBoundaryAndEdgeNumbers_triangle(Mesh& mesh)
    cdef int constructElementBoundaryElementsArrayWithGivenElementBoundaryAndEdgeNumbers_quadrilateral(Mesh& mesh)
    cdef int constructElementBoundaryElementsArrayWithGivenElementBoundaryAndEdgeNumbers_tetrahedron(Mesh& mesh)
    cdef int constructElementBoundaryElementsArrayWithGivenElementBoundaryAndEdgeNumbers_hexahedron(Mesh& mesh)
    cdef int constructElementBoundaryElementsArrayWithGivenElementBoundaryAndEdgeNumbers_NURBS(Mesh& mesh)

    # cdef int writeElements(std::ostream& meshFile,
    #                        const Mesh& mesh)
    # cdef int writeNodes(std::ostream& meshFile,
    #                     const Mesh& mesh)
    # cdef int readElements(std::istream& meshFile,
    #                       Mesh& mesh)


    cdef int allocateGeometricInfo_tetrahedron(Mesh& mesh)
    cdef int allocateGeometricInfo_triangle(Mesh& mesh)
    cdef int allocateGeometricInfo_edge(Mesh& mesh)
    cdef int allocateGeometricInfo_quadrilateral(Mesh& mesh)
    cdef int allocateGeometricInfo_hexahedron(Mesh& mesh)
    cdef int allocateGeometricInfo_NURBS(Mesh& mesh)

    cdef int computeGeometricInfo_tetrahedron(Mesh& mesh)
    cdef int computeGeometricInfo_triangle(Mesh& mesh)
    cdef int computeGeometricInfo_edge(Mesh& mesh)
    cdef int computeGeometricInfo_hexahedron(Mesh& mesh)
    cdef int computeGeometricInfo_quadrilateral(Mesh& mesh)
    cdef int computeGeometricInfo_NURBS(Mesh& mesh)

    cdef int assignElementBoundaryMaterialTypesFromParent(Mesh& parentMesh,
                                                          Mesh& childMesh,
                                                          const int* levelElementParentsArray,
						                                  const int& nSpace_global)
    cdef int allocateNodeAndElementNodeDataStructures(Mesh& mesh,
                                                      int nElements_global,
                                                      int nNodes_global,
                                                      int nNodes_element)
    cdef struct triangulateio

    cdef int setFromTriangleElements(triangulateio* trimesh,
                                     Mesh& mesh,
                                     int base)
    cdef int setFromTriangleNodes(triangulateio* trimesh,
                                  Mesh& mesh,
                                  int base)
    cdef int readTriangleMesh(Mesh& mesh,
                              const char* filebase,
                              int base)
    cdef int readTriangleElementBoundaryMaterialTypes(Mesh& mesh,
                                                      const char* filebase,
                                                      int base)
    cdef int writeTriangleMesh(Mesh& mesh,
                               const char* filebase,
                               int base)
    cdef int readDMPlexMesh(object dmplexMesh,
                               Mesh& mesh)
    cdef int readTetgenMesh(Mesh& mesh,
                            const char* filebase,
                            int base)
    cdef int readTetgenElementBoundaryMaterialTypes(Mesh& mesh,
                                                    const char* filebase,
                                                    int base)
    cdef int readEdgeMaterialTypes(Mesh& mesh,
                                   const char* filebase,
                                   int base)
    cdef int allocateEdgeMaterialTypes(Mesh& mesh)
    cdef int writeTetgenMesh(Mesh& mesh,
                             const char* filebase,
                             int base)
    cdef int readHex(Mesh& mesh,
                     const char* filebase,
                     int indexBase)
    cdef int write3dmMesh(Mesh& mesh,
                          const char* filebase,
                          int base)
    cdef int write2dmMesh(Mesh& mesh,
                          const char* filebase,
                          int base)
    cdef int copyElementBoundaryMaterialTypesFromTriangle(triangulateio* trimesh,
						                                  Mesh& mesh,
                                                          int base)
