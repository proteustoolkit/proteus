# A type of -*- python -*- file
from libcpp.vector cimport vector
from libcpp.string cimport string
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
    extern void c_nodeGraphRows "proteus::nodeGraphRows" (const mesh.Mesh& mesh, int begin, int end,
                                                          vector[int]& offsets, vector[int]& neighbors)
    extern vector[int] c_elementOwnersFromNodeOwners "proteus::elementOwnersFromNodeOwners" (const mesh.Mesh& mesh,
                                                                                             const vector[int]& nodeOwner,
                                                                                             int size)
    extern vector[int] c_elementBoundaryOwnersFromNodeOwners "proteus::elementBoundaryOwnersFromNodeOwners" (const mesh.Mesh& mesh,
                                                                                                             const vector[int]& nodeOwner,
                                                                                                             int size)
    extern vector[int] c_edgeOwnersFromNodeOwners "proteus::edgeOwnersFromNodeOwners" (const mesh.Mesh& mesh,
                                                                                       const vector[int]& nodeOwner,
                                                                                       const vector[int]& nodeNumbering_global_old2new)
    extern int c_partitionNodesFromTetgenFiles "proteus::partitionNodesFromTetgenFiles" (const MPI_Comm& PROTEUS_COMM_WORLD,
                                                                                         const char* filebase,
                                                                                         int indexBase, mesh.Mesh& newMesh,
                                                                                         int nNodes_overlap,
                                                                                         double memHardLimit);
    extern int c_partitionNodesFromTriangleFiles "proteus::partitionNodesFromTriangleFiles" (const MPI_Comm& PROTEUS_COMM_WORLD,
                                                                                         const char* filebase,
                                                                                         int indexBase, mesh.Mesh& newMesh,
                                                                                         int nNodes_overlap);

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
                                                         const int *elementOffsets_subdomain_owned,
                                                         const int *nodeOffsets_subdomain_owned,
                                                         const int *elementNumbering_subdomain2global,
                                                         const int *nodeNumbering_subdomain2global,
                                                         int& nDOF_all_processes,#total number of dofs in whole domain
                                                         int& nDOF_subdomain,#total number of dofs in sub-domain
                                                         int& max_dof_neighbors,#maximum number of neighbors for connectivity of dofs
                                                         int *offsets_subdomain_owned, #starting point of local dofs on each processor (nProcs+1)
                                                         int * subdomain_l2g, #local to global dof mapping on subdomain
                                                         int* subdomain2global,#subdomain dof to global (parallel) numbering
                                                         double * lagrangeNodesArray);#location of nodes corresponding to dofs
    extern int buildQuadraticSubdomain2GlobalMappings_2d(const MPI_Comm& PROTEUS_COMM_WORLD, mesh.Mesh& mesh,
                                                         const int *elementBoundaryOffsets_subdomain_owned,
                                                         const int *nodeOffsets_subdomain_owned,
                                                         const int *elementBoundaryNumbering_subdomain2global,
                                                         const int *nodeNumbering_subdomain2global,
                                                         int& nDOF_all_processes,#total number of dofs in whole domain
                                                         int& nDOF_subdomain,#total number of dofs in sub-domain
                                                         int& max_dof_neighbors,#maximum number of neighbors for connectivity of dofs
                                                         int *offsets_subdomain_owned, #starting point of local dofs on each processor (nProcs+1)
                                                         int *subdomain_l2g, #local to global dof mapping on subdomain
                                                         int *subdomain2global,#subdomain dof to global (parallel) numbering
                                                         double * lagrangeNodesArray);#location of nodes corresponding to dofs
    extern int buildQuadraticSubdomain2GlobalMappings_3d(const MPI_Comm& PROTEUS_COMM_WORLD, mesh.Mesh& mesh,
                                                         const int *edgeOffsets_subdomain_owned,
                                                         const int *nodeOffsets_subdomain_owned,
                                                         const int *edgeNumbering_subdomain2global,
                                                         const int *nodeNumbering_subdomain2global,
                                                         int& nDOF_all_processes,#total number of dofs in whole domain
                                                         int& nDOF_subdomain,#total number of dofs in sub-domain
                                                         int& max_dof_neighbors,#maximum number of neighbors for connectivity of dofs
                                                         int *offsets_subdomain_owned, #starting point of local dofs on each processor (nProcs+1)
                                                         int *subdomain_l2g, #local to global dof mapping on subdomain
                                                         int *subdomain2global,#subdomain dof to global (parallel) numbering
                                                         double * lagrangeNodesArray);#location of nodes corresponding to dofs
    extern int buildQuadraticCubeSubdomain2GlobalMappings_3d(const MPI_Comm& PROTEUS_COMM_WORLD, mesh.Mesh& mesh,
                                                             const int *edgeOffsets_subdomain_owned,
                                                             const int *nodeOffsets_subdomain_owned,
                                                             const int *edgeNumbering_subdomain2global,
                                                             const int *nodeNumbering_subdomain2global,
                                                             int& nDOF_all_processes,#total number of dofs in whole domain
                                                             int& nDOF_subdomain,#total number of dofs in sub-domain
                                                             int& max_dof_neighbors,#maximum number of neighbors for connectivity of dofs
                                                             int *offsets_subdomain_owned, #starting point of local dofs on each processor (nProcs+1)
                                                             int *subdomain_l2g, #local to global dof mapping on subdomain
                                                             int *subdomain2global,#subdomain dof to global (parallel) numbering
                                                             double * lagrangeNodesArray);#location of nodes corresponding to dofs
    extern int buildDiscontinuousGalerkinSubdomain2GlobalMappings(const MPI_Comm& PROTEUS_COMM_WORLD, mesh.Mesh& mesh,
                                                                  const int *elementOffsets_subdomain_owned,
                                                                  const int *elementNumbering_subdomain2global,
                                                                  int nDOF_element,
                                                                  int& nDOF_all_processes,#total number of dofs in whole domain
                                                                  int& nDOF_subdomain,#total number of dofs in sub-domain
                                                                  int& max_dof_neighbors,#maximum number of neighbors for connectivity of dofs
                                                                  int *offsets_subdomain_owned, #starting point of local dofs on each processor (nProcs+1)
                                                                  int * subdomain_l2g, #local to global dof mapping on subdomain
                                                                  int* subdomain2global);#subdomain dof to global (parallel) numbering
