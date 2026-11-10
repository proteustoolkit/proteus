# A type of -*- python -*- file
from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp.set cimport set as cppset
from libcpp.map cimport map as cppmap
from libcpp.utility cimport pair
from libcpp cimport bool as cppbool
from mpi4py.libmpi cimport MPI_Comm
cimport mesh

cdef extern from "partitioning.h" namespace "proteus":
    extern string c_petscPartitioners "proteus::petscPartitioners" ()
    extern int c_ensurePetscInitialized "proteus::ensurePetscInitialized" ()
    extern bint c_petscIsInitialized "proteus::petscIsInitialized" ()
    extern int c_generateCartesianSubdomain "proteus::generateCartesianSubdomain" (const MPI_Comm& PROTEUS_COMM_WORLD,
                                                                                   int kind,
                                                                                   int triangleFlag,
                                                                                   const int* nn,
                                                                                   const double* L,
                                                                                   const int* rankGrid,
                                                                                   int nLayersOfOverlap,
                                                                                   mesh.Mesh& mesh,
                                                                                   int level,
                                                                                   const double* origin)
    extern int c_cartesianNumbering "proteus::cartesianNumbering" (int kind, int triangleFlag, const int* nn, const int* rankGrid,
                                                                   int level, int elements, int n, const int* oldNumbers, int* newNumbers)
    extern int c_cartesianVertexInterpolation "proteus::cartesianVertexInterpolation" (int kind, int triangleFlag, const int* nn,
                                                                                       const int* rankGrid, int level, int rank,
                                                                                       vector[int]& rowptr, vector[int]& colind,
                                                                                       vector[double]& nzval)
    extern int c_cartesianSubdomainParents "proteus::cartesianSubdomainParents" (int kind, int triangleFlag, const int* nn,
                                                                                 const int* rankGrid, int level, int rank,
                                                                                 vector[int]& parents)
    extern int c_cartesianGlobalCounts "proteus::cartesianGlobalCounts" (int kind, int triangleFlag, const int* nn, int level, int* counts)
    extern int c_partitionNodes "proteus::partitionNodes" (const MPI_Comm& PROTEUS_COMM_WORLD,
                                                           mesh.Mesh& mesh,
                                                           int nNodes_overlap,
                                                           const int* nodeOwners,
                                                           const int* nodeWeights);
    extern int c_partitionNodesLocal "proteus::partitionNodesLocal" (mesh.Mesh& mesh,
                                                                     int rank,
                                                                     int size,
                                                                     int nNodes_overlap,
                                                                     const vector[int]& nodeOwner)
    extern void c_nodeGraphRows "proteus::nodeGraphRows" (const mesh.Mesh& mesh, mesh.proteus_index_t begin, mesh.proteus_index_t end,
                                                          vector[mesh.proteus_index_t]& offsets, vector[mesh.proteus_index_t]& neighbors)
    extern void c_elementGraphRows "proteus::elementGraphRows" (const mesh.Mesh& mesh, mesh.proteus_index_t begin, mesh.proteus_index_t end,
                                                                vector[mesh.proteus_index_t]& offsets, vector[mesh.proteus_index_t]& neighbors)
    extern vector[int] c_nodeOwnersFromElementOwners "proteus::nodeOwnersFromElementOwners" (const mesh.Mesh& mesh, const vector[int]& elementOwner, int size)
    extern vector[int] c_elementBoundaryOwnersFromElementOwners "proteus::elementBoundaryOwnersFromElementOwners" (const mesh.Mesh& mesh, const vector[int]& elementOwner, int size)
    extern vector[int] c_edgeOwnersFromElementOwners "proteus::edgeOwnersFromElementOwners" (const mesh.Mesh& mesh, const vector[int]& elementOwner, int size)
    extern vector[int] c_elementOwnersFromNodeOwners "proteus::elementOwnersFromNodeOwners" (const mesh.Mesh& mesh,
                                                                                             const vector[int]& nodeOwner,
                                                                                             int size)
    extern vector[int] c_elementBoundaryOwnersFromNodeOwners "proteus::elementBoundaryOwnersFromNodeOwners" (const mesh.Mesh& mesh,
                                                                                                             const vector[int]& nodeOwner,
                                                                                                             int size)
    extern vector[int] c_edgeOwnersFromNodeOwners "proteus::edgeOwnersFromNodeOwners" (const mesh.Mesh& mesh,
                                                                                       const vector[int]& nodeOwner,
                                                                                       const vector[mesh.proteus_index_t]& nodeNumbering_global_old2new)
    extern int c_partitionNodesFromTetgenFiles "proteus::partitionNodesFromTetgenFiles" (const MPI_Comm& PROTEUS_COMM_WORLD,
                                                                                         const char* filebase,
                                                                                         int indexBase, mesh.Mesh& newMesh,
                                                                                         int nNodes_overlap,
                                                                                         double memHardLimit,
                                                                                         vector[mesh.proteus_index_t]* nodeNumbering_subdomain2input,
                                                                                         vector[mesh.proteus_index_t]* elementNumbering_subdomain2input);
    extern cppbool c_readSimplexNodeStars "proteus::readSimplexNodeStars" (const string& elementFileName, const char* format,
                                                                           int simplexDim, int indexBase,
                                                                           mesh.proteus_index_t nNodes_global, mesh.proteus_index_t begin, mesh.proteus_index_t end,
                                                                           mesh.proteus_index_t& nElements_global, vector[cppset[mesh.proteus_index_t]]& nodeStar)
    extern cppbool c_readSimplexElementsTouching "proteus::readSimplexElementsTouching" (const string& elementFileName, const char* format,
                                                                                         int simplexDim, int indexBase, mesh.proteus_index_t nNodes_global,
                                                                                         const cppmap[mesh.proteus_index_t,mesh.proteus_index_t]& nodes,
                                                                                         mesh.proteus_index_t& nElements_global, int& hasElementMarkers,
                                                                                         vector[mesh.proteus_index_t]& elements, vector[mesh.proteus_index_t]& elementNodes,
                                                                                         vector[double]& elementIds)
    extern cppbool c_readSimplexSubentitiesTouching "proteus::readSimplexSubentitiesTouching" (const string& fileName, const char* format,
                                                                                               const char* what, int nNodesPerEntity,
                                                                                               int indexBase, cppbool headerHasMarkers,
                                                                                               const cppmap[mesh.proteus_index_t,mesh.proteus_index_t]& nodes,
                                                                                               mesh.proteus_index_t& nEntities_global, int& hasMarkers,
                                                                                               vector[mesh.proteus_index_t]& numbers, vector[mesh.proteus_index_t]& entityNodes,
                                                                                               vector[int]& ids)
    extern cppmap[mesh.proteus_index_t,mesh.proteus_index_t] c_ownedNodesByFileNumber "proteus::ownedNodesByFileNumber" (const MPI_Comm& comm,
                                                                                       const vector[mesh.proteus_index_t]& nodeOffsets_old,
                                                                                       const vector[mesh.proteus_index_t]& nodeOffsets_new,
                                                                                       const vector[mesh.proteus_index_t]& old2new_chunk)
    extern cppmap[mesh.proteus_index_t,mesh.proteus_index_t] c_newNumbersOfFileNodes "proteus::newNumbersOfFileNodes" (const MPI_Comm& comm,
                                                                                     const vector[mesh.proteus_index_t]& nodeOffsets_old,
                                                                                     const vector[mesh.proteus_index_t]& old2new_chunk,
                                                                                     const cppset[mesh.proteus_index_t]& wanted)
    extern vector[pair[mesh.proteus_index_t,mesh.proteus_index_t]] c_subdomainNumbersByDirectory "proteus::subdomainNumbersByDirectory" (const MPI_Comm& comm,
                                                                                                       mesh.proteus_index_t nGlobal, mesh.proteus_index_t ownedBegin,
                                                                                                       const cppset[mesh.proteus_index_t]& owned,
                                                                                                       const vector[mesh.proteus_index_t]& subdomain)
    extern cppbool c_readSubdomainVertices "proteus::readSubdomainVertices" (const string& vertexFileName, const char* format,
                                                                             int nSpace, int indexBase, int hasVertexMarkers,
                                                                             const cppmap[mesh.proteus_index_t,mesh.proteus_index_t]& nodes_old2new,
                                                                             mesh.proteus_index_t ownedBegin, mesh.proteus_index_t ownedEnd,
                                                                             const cppmap[mesh.proteus_index_t,mesh.proteus_index_t]& nodes_overlap_global2subdomain,
                                                                             mesh.proteus_index_t* nodeNumbering_subdomain2global,
                                                                             cppmap[mesh.proteus_index_t,mesh.proteus_index_t]& nodeNumbering_global2subdomainMap,
                                                                             double* nodeArray, int* nodeMaterialTypes)
    extern int c_partitionNodesFromTriangleFiles "proteus::partitionNodesFromTriangleFiles" (const MPI_Comm& PROTEUS_COMM_WORLD,
                                                                                         const char* filebase,
                                                                                         int indexBase, mesh.Mesh& newMesh,
                                                                                         int nNodes_overlap,
                                                                                         vector[mesh.proteus_index_t]* nodeNumbering_subdomain2input,
                                                                                         vector[mesh.proteus_index_t]* elementNumbering_subdomain2input);

    extern int c_partitionElementsLocal "proteus::partitionElementsLocal" (mesh.Mesh& mesh,
                                                                           int rank,
                                                                           int size,
                                                                           int nElements_overlap,
                                                                           const vector[int]& elementOwner)
    extern int c_partitionElements "proteus::partitionElements" (const MPI_Comm& PROTEUS_COMM_WORLD,
                                                                 mesh.Mesh& mesh,
                                                                 int nElements_overlap,
                                                                 const int* elementOwners);
    extern int buildQuadraticSubdomain2GlobalMappings_1d(const MPI_Comm& PROTEUS_COMM_WORLD, mesh.Mesh& mesh,
                                                         const mesh.proteus_index_t *elementOffsets_subdomain_owned,
                                                         const mesh.proteus_index_t *nodeOffsets_subdomain_owned,
                                                         const mesh.proteus_index_t *elementNumbering_subdomain2global,
                                                         const mesh.proteus_index_t *nodeNumbering_subdomain2global,
                                                         int& nDOF_all_processes,#total number of dofs in whole domain
                                                         int& nDOF_subdomain,#total number of dofs in sub-domain
                                                         int& max_dof_neighbors,#maximum number of neighbors for connectivity of dofs
                                                         int *offsets_subdomain_owned, #starting point of local dofs on each processor (nProcs+1)
                                                         int * subdomain_l2g, #local to global dof mapping on subdomain
                                                         int* subdomain2global,#subdomain dof to global (parallel) numbering
                                                         double * lagrangeNodesArray);#location of nodes corresponding to dofs
    extern int buildQuadraticSubdomain2GlobalMappings_2d(const MPI_Comm& PROTEUS_COMM_WORLD, mesh.Mesh& mesh,
                                                         const mesh.proteus_index_t *elementBoundaryOffsets_subdomain_owned,
                                                         const mesh.proteus_index_t *nodeOffsets_subdomain_owned,
                                                         const mesh.proteus_index_t *elementBoundaryNumbering_subdomain2global,
                                                         const mesh.proteus_index_t *nodeNumbering_subdomain2global,
                                                         int& nDOF_all_processes,#total number of dofs in whole domain
                                                         int& nDOF_subdomain,#total number of dofs in sub-domain
                                                         int& max_dof_neighbors,#maximum number of neighbors for connectivity of dofs
                                                         int *offsets_subdomain_owned, #starting point of local dofs on each processor (nProcs+1)
                                                         int *subdomain_l2g, #local to global dof mapping on subdomain
                                                         int *subdomain2global,#subdomain dof to global (parallel) numbering
                                                         double * lagrangeNodesArray);#location of nodes corresponding to dofs
    extern int buildQuadraticSubdomain2GlobalMappings_3d(const MPI_Comm& PROTEUS_COMM_WORLD, mesh.Mesh& mesh,
                                                         const mesh.proteus_index_t *edgeOffsets_subdomain_owned,
                                                         const mesh.proteus_index_t *nodeOffsets_subdomain_owned,
                                                         const mesh.proteus_index_t *edgeNumbering_subdomain2global,
                                                         const mesh.proteus_index_t *nodeNumbering_subdomain2global,
                                                         int& nDOF_all_processes,#total number of dofs in whole domain
                                                         int& nDOF_subdomain,#total number of dofs in sub-domain
                                                         int& max_dof_neighbors,#maximum number of neighbors for connectivity of dofs
                                                         int *offsets_subdomain_owned, #starting point of local dofs on each processor (nProcs+1)
                                                         int *subdomain_l2g, #local to global dof mapping on subdomain
                                                         int *subdomain2global,#subdomain dof to global (parallel) numbering
                                                         double * lagrangeNodesArray);#location of nodes corresponding to dofs
    extern int buildQuadraticCubeSubdomain2GlobalMappings_3d(const MPI_Comm& PROTEUS_COMM_WORLD, mesh.Mesh& mesh,
                                                             const mesh.proteus_index_t *edgeOffsets_subdomain_owned,
                                                             const mesh.proteus_index_t *nodeOffsets_subdomain_owned,
                                                             const mesh.proteus_index_t *edgeNumbering_subdomain2global,
                                                             const mesh.proteus_index_t *nodeNumbering_subdomain2global,
                                                             int& nDOF_all_processes,#total number of dofs in whole domain
                                                             int& nDOF_subdomain,#total number of dofs in sub-domain
                                                             int& max_dof_neighbors,#maximum number of neighbors for connectivity of dofs
                                                             int *offsets_subdomain_owned, #starting point of local dofs on each processor (nProcs+1)
                                                             int *subdomain_l2g, #local to global dof mapping on subdomain
                                                             int *subdomain2global,#subdomain dof to global (parallel) numbering
                                                             double * lagrangeNodesArray);#location of nodes corresponding to dofs
    extern int buildDiscontinuousGalerkinSubdomain2GlobalMappings(const MPI_Comm& PROTEUS_COMM_WORLD, mesh.Mesh& mesh,
                                                                  const mesh.proteus_index_t *elementOffsets_subdomain_owned,
                                                                  const mesh.proteus_index_t *elementNumbering_subdomain2global,
                                                                  int nDOF_element,
                                                                  int& nDOF_all_processes,#total number of dofs in whole domain
                                                                  int& nDOF_subdomain,#total number of dofs in sub-domain
                                                                  int& max_dof_neighbors,#maximum number of neighbors for connectivity of dofs
                                                                  int *offsets_subdomain_owned, #starting point of local dofs on each processor (nProcs+1)
                                                                  int * subdomain_l2g, #local to global dof mapping on subdomain
                                                                  int* subdomain2global);#subdomain dof to global (parallel) numbering
