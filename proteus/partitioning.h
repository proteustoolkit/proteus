#ifndef PARTITIONING_H
#define PARTITIONING_H
#include <iostream>
#include <string>
#include <valarray>
#include <vector>
#include <set>
#include <map>
#include <cstdlib>
#include <iterator>
#include "mpi.h"
#include "hdf5.h"
#include "petsc.h"
#include "petscsys.h"
#include "mesh.h"
#include "meshio.h"

namespace proteus
{
  //--memory profiling
  /*
   * Author:  David Robert Nadeau
   * Site:    http://NadeauSoftware.com/
   * License: Creative Commons Attribution 3.0 Unported License
   *          http://creativecommons.org/licenses/by/3.0/deed.en_US
   */

#if defined(_WIN32)
#include <windows.h>
#include <psapi.h>

#elif defined(__unix__) || defined(__unix) || defined(unix) || (defined(__APPLE__) && defined(__MACH__))
#include <unistd.h>
#include <sys/resource.h>

#if defined(__APPLE__) && defined(__MACH__)
#include <mach/mach.h>

#elif (defined(_AIX) || defined(__TOS__AIX__)) || (defined(__sun__) || defined(__sun) || defined(sun) && (defined(__SVR4) || defined(__svr4__)))
#include <fcntl.h>
#include <procfs.h>

#elif defined(__linux__) || defined(__linux) || defined(linux) || defined(__gnu_linux__)
#include <stdio.h>

#endif

#else
#error "Cannot define getPeakRSS( ) or getCurrentRSS( ) for an unknown OS."
#endif

  /**
   * Returns the peak (maximum so far) resident set size (physical
   * memory use) measured in bytes, or zero if the value cannot be
   * determined on this OS.
   */
  inline size_t getPeakRSS( )
  {
#if defined(_WIN32)
    /* Windows -------------------------------------------------- */
    PROCESS_MEMORY_COUNTERS info;
    GetProcessMemoryInfo( GetCurrentProcess( ), &info, sizeof(info) );
    return (size_t)info.PeakWorkingSetSize;

#elif (defined(_AIX) || defined(__TOS__AIX__)) || (defined(__sun__) || defined(__sun) || defined(sun) && (defined(__SVR4) || defined(__svr4__)))
    /* AIX and Solaris ------------------------------------------ */
    struct psinfo psinfo;
    int fd = -1;
    if ( (fd = open( "/proc/self/psinfo", O_RDONLY )) == -1 )
      return (size_t)0L;/* Can't open? */
    if ( read( fd, &psinfo, sizeof(psinfo) ) != sizeof(psinfo) )
      {
        close( fd );
        return (size_t)0L;/* Can't read? */
      }
    close( fd );
    return (size_t)(psinfo.pr_rssize * 1024L);

#elif defined(__unix__) || defined(__unix) || defined(unix) || (defined(__APPLE__) && defined(__MACH__))
    /* BSD, Linux, and OSX -------------------------------------- */
    struct rusage rusage;
    getrusage( RUSAGE_SELF, &rusage );
#if defined(__APPLE__) && defined(__MACH__)
    return (size_t)rusage.ru_maxrss;
#else
    return (size_t)(rusage.ru_maxrss * 1024L);
#endif

#else
    /* Unknown OS ----------------------------------------------- */
    return (size_t)0L;/* Unsupported. */
#endif
  }

  /**
   * Returns the current resident set size (physical memory use) measured
   * in bytes, or zero if the value cannot be determined on this OS.
   */
  inline size_t getCurrentRSS( )
  {
#if defined(_WIN32)
    /* Windows -------------------------------------------------- */
    PROCESS_MEMORY_COUNTERS info;
    GetProcessMemoryInfo( GetCurrentProcess( ), &info, sizeof(info) );
    return (size_t)info.WorkingSetSize;

#elif defined(__APPLE__) && defined(__MACH__)
    /* OSX ------------------------------------------------------ */
    struct mach_task_basic_info info;
    mach_msg_type_number_t infoCount = MACH_TASK_BASIC_INFO_COUNT;
    if ( task_info( mach_task_self( ), MACH_TASK_BASIC_INFO,
                    (task_info_t)&info, &infoCount ) != KERN_SUCCESS )
      return (size_t)0L;/* Can't access? */
    return (size_t)info.resident_size;

#elif defined(__linux__) || defined(__linux) || defined(linux) || defined(__gnu_linux__)
    /* Linux ---------------------------------------------------- */
    long rss = 0L;
    FILE* fp = NULL;
    if ( (fp = fopen( "/proc/self/statm", "r" )) == NULL )
      return (size_t)0L;/* Can't open? */
    if ( fscanf( fp, "%*s%ld", &rss ) != 1 )
      {
        fclose( fp );
        return (size_t)0L;/* Can't read? */
      }
    fclose( fp );
    return (size_t)rss * (size_t)sysconf( _SC_PAGESIZE);

#else
    /* AIX, BSD, Solaris, and Unknown OS ------------------------ */
    return (size_t)0L;/* Unsupported. */
#endif
  }
  //--memory profiling
  inline int enforceMemoryLimit(const MPI_Comm& PROTEUS_COMM_WORLD, int rank, double max_rss_gb,const char* msg)
  {
    if (max_rss_gb <= 0.0)
      return 0;
    double current, current_global,gb(1.0e-9);
    PetscBarrier(NULL);
    current = double(getCurrentRSS())*gb;
    PetscBarrier(NULL);
    current_global=0.0;
    MPI_Allreduce(&current,&current_global,1,MPI_DOUBLE,MPI_MAX,PROTEUS_COMM_WORLD);
    if (current > max_rss_gb)
      {
        std::cout<<"Raising PETSC_ERR_MEM, Memory usage  on rank "<<rank<<'\t'<<current<<"GB"<<'\t'<<"limit "<<max_rss_gb<<std::endl;
        SETERRABORT(PROTEUS_COMM_WORLD,PETSC_ERR_MEM,"Exceeded Proteus memory limit");
      }
    if (rank ==  0)
      std::cout<<msg<<std::endl
               <<"Max memory usage per core "<<current_global<<"GB"<<std::endl;
    return 0;
  }

  /* The graph partitioner (MatPartitioning) and the partitioners that read
     tetgen/triangle files use the PETSc C library. petsc4py initializes it when
     proteus runs with petsc4py; without petsc4py nothing else does, so those
     entry points call this first. partitionNodes and partitionElements given
     their owners use no PETSc and do not. MPI is already initialized by then
     (mpi4py), so PETSc runs on MPI_COMM_WORLD. */
  inline bool petscIsInitialized()
  {
    PetscBool initialized = PETSC_FALSE;
    PetscInitialized(&initialized);
    return initialized == PETSC_TRUE;
  }

  //space-separated names of the graph partitioners PETSc was built with
  std::string petscPartitioners();

  inline int ensurePetscInitialized()
  {
    PetscBool initialized = PETSC_FALSE;
    PetscInitialized(&initialized);
    if (initialized)
      return 0;
    return PetscInitializeNoArguments();
  }

  /* Wall-clock time of each stage of a partitioner, reported (the maximum over
     ranks) on rank 0 when PROTEUS_PARTITION_TIMING is set in the environment.
     This replaces what PetscLogEvent would record, without PETSc. */
  class StageTimer
  {
  public:
    StageTimer(const MPI_Comm& comm, const char* name)
      : comm_(comm), name_(name), last_(MPI_Wtime()), on_(std::getenv("PROTEUS_PARTITION_TIMING") != NULL) {}
    void mark(const char* stage)
    {
      const double now = MPI_Wtime();
      stages_.push_back(stage);
      seconds_.push_back(now - last_);
      last_ = now;
    }
    void report()
    {
      if (!on_)
        return;
      int rank;
      MPI_Comm_rank(comm_,&rank);
      std::vector<double> worst(seconds_.size());
      MPI_Reduce(seconds_.data(),worst.data(),int(seconds_.size()),MPI_DOUBLE,MPI_MAX,0,comm_);
      if (rank == 0)
        {
          std::cout<<"PARTITION_TIMING "<<name_;
          for (size_t i = 0; i < stages_.size(); i++)
            std::cout<<" "<<stages_[i]<<"="<<worst[i];
          std::cout<<std::endl;
        }
    }
  private:
    MPI_Comm comm_;
    const char* name_;
    double last_;
    bool on_;
    std::vector<const char*> stages_;
    std::vector<double> seconds_;
  };

  /* A regular grid of nn[0] x nn[1] (x nn[2]) nodes on [0,L] -- kind 0
     quadrilaterals, 1 hexahedra, 2 triangles (split by triangleFlag as
     generateTriangularMeshFromRectangularGrid splits them), 3 tetrahedra --
     generated directly as this rank's subdomain of the rankGrid partition: the
     subdomain mesh (mesh.subdomainp), offsets and subdomain-to-global maps that
     partitionNodes gives for the global grid from the rectangular-grid
     generators and MeshTools.cartesianNodeOwners, without the global mesh. mesh
     keeps only the global counts. See cartesianPartitioning.cpp. */
  extern int generateCartesianSubdomain(const MPI_Comm& PROTEUS_COMM_WORLD, int kind, int triangleFlag, const int* nn,
                                        const double* L, const int* rankGrid, int nLayersOfOverlap, Mesh& mesh,
                                        int level, const double* origin);
  /* The new numbers of nodes (elements != 0: elements) of that partition,
     from their old ones: the permutation partitionNodes keeps as
     nodeNumbering_global_old2new, as a function. level > 0 is the grid nn
     refined that many times, partitioned by the coarsest level's bricks
     scaled. */
  extern int cartesianNumbering(int kind, int triangleFlag, const int* nn, const int* rankGrid, int level, int elements,
                                int n, const int* oldNumbers, int* newNumbers);
  /* Global numbers of nodes, elements, element boundaries and edges. */
  extern int cartesianGlobalCounts(int kind, int triangleFlag, const int* nn, int level, int* counts);
  /* The rows this rank owns of the vertex prolongation from level-1 to level
     of a generated hierarchy, in CSR, with columns in level-1's partitioned
     numbering. */
  extern int cartesianVertexInterpolation(int kind, int triangleFlag, const int* nn, const int* rankGrid, int level, int rank,
                                          std::vector<proteus_index_t>& rowptr, std::vector<proteus_index_t>& colind, std::vector<double>& nzval);
  /* For each element of this rank's subdomain on level, its parent's index in
     the subdomain on level-1, or -1. */
  extern int cartesianSubdomainParents(int kind, int triangleFlag, const int* nn, const int* rankGrid, int level, int rank,
                                       std::vector<proteus_index_t>& parents);

  extern int partitionNodes(const MPI_Comm& PROTEUS_COMM_WORLD, Mesh& mesh, int nNodes_overlap,
                            const int* nodeOwners=NULL, const int* nodeWeights=NULL);

  /* The local part of partitionNodes: rank's part of the partition given the
     owner of every node. No MPI, so any rank's subdomain can be computed in one
     process. See partitioning.cpp. */
  /* The default partition of n entities: contiguous near-equal chunks in input
     order, as offsets (size+1). Plain C++. */
  extern std::valarray<proteus_index_t> defaultChunkOffsets(proteus_index_t n, int size);

  /* Rows begin..end-1 of the node graph (each node's sorted star) in CSR form:
     a rank's slice of what the graph partitioner partitions. Plain C++. */
  extern void nodeGraphRows(const Mesh& mesh, proteus_index_t begin, proteus_index_t end, std::vector<proteus_index_t>& offsets, std::vector<proteus_index_t>& neighbors);

  /* MPI and PETSc: every node's owner from MatPartitioning, gathered on every
     rank. Collective. */
  extern std::vector<int> graphNodeOwners(const MPI_Comm& PROTEUS_COMM_WORLD, const Mesh& mesh,
                                          const std::valarray<proteus_index_t>& nodeOffsets_old, const int* nodeWeights);

  extern int partitionNodesLocal(Mesh& mesh, int rank, int size, int nNodes_overlap,
                                 const std::vector<int>& nodeOwner, StageTimer* timer=NULL);

  /* partitionNodesLocal's ownership rules, given the owner of every node:
     an element and an element boundary belong to the lowest rank owning one of
     their nodes, an edge to the owner of its end with the lower new number.
     Entities no rank reaches get the owner size. */
  extern std::vector<int> elementOwnersFromNodeOwners(const Mesh& mesh, const std::vector<int>& nodeOwner, int size);
  extern std::vector<int> elementBoundaryOwnersFromNodeOwners(const Mesh& mesh, const std::vector<int>& nodeOwner, int size);
  extern std::vector<int> edgeOwnersFromNodeOwners(const Mesh& mesh, const std::vector<int>& nodeOwner,
                                                   const std::vector<proteus_index_t>& nodeNumbering_global_old2new);

  /* The parun -F partitioners. When given, nodeNumbering_subdomain2input and
     elementNumbering_subdomain2input get the file number (0-based) of each
     node and element of this rank's subdomain, owned and overlap: the
     partition's input numbering, O(subdomain) per rank. */
  extern int partitionNodesFromTetgenFiles(const MPI_Comm& PROTEUS_COMM_WORLD, const char* filebase, int indexBase, Mesh& newMesh, int nNodes_overlap, double memHardLimit,
                                           std::vector<proteus_index_t>* nodeNumbering_subdomain2input=nullptr,
                                           std::vector<proteus_index_t>* elementNumbering_subdomain2input=nullptr);

  /* I/O, no MPI: stream a Triangle or TetGen .ele file (simplexDim nodes per
     element) and return the star of each node in [begin,end) and the element
     count. False if the file cannot be opened. */
  extern bool readSimplexNodeStars(const std::string& elementFileName, const char* format, int simplexDim,
                                   int indexBase, proteus_index_t nNodes_global, int begin, int end, proteus_index_t& nElements_global,
                                   std::vector<std::set<proteus_index_t> >& nodeStar);

  /* I/O, local: the vertices of one subdomain from a Triangle (nSpace 2) or
     TetGen (nSpace 3) node file, in the subdomain's numbering: owned new global
     numbers [ownedBegin,ownedEnd) first, then the overlap (the keys of
     nodes_overlap_global2subdomain, at its values). nodes_old2new maps the file
     nodes the subdomain touches to new global numbers. nodeMaterialTypes is
     written only when hasVertexMarkers. False if the file cannot be opened. */
  extern bool readSubdomainVertices(const std::string& vertexFileName, const char* format, int nSpace, int indexBase,
                                    int hasVertexMarkers, const std::map<proteus_index_t,proteus_index_t>& nodes_old2new,
                                    proteus_index_t ownedBegin, proteus_index_t ownedEnd,
                                    const std::map<proteus_index_t,proteus_index_t>& nodes_overlap_global2subdomain,
                                    proteus_index_t* nodeNumbering_subdomain2global,
                                    std::map<proteus_index_t,proteus_index_t>& nodeNumbering_global2subdomainMap,
                                    double* nodeArray, int* nodeMaterialTypes);

  /* I/O, local: stream a Triangle or TetGen .ele file and keep, in file order,
     the elements with a node among the keys of `nodes` (0-based file
     numbers): their file numbers, their file nodes (simplexDim each, in
     elementNodes) and, when the header announces attributes, the first
     (elementIds). False if the file cannot be opened. */
  extern bool readSimplexElementsTouching(const std::string& elementFileName, const char* format, int simplexDim,
                                          int indexBase, proteus_index_t nNodes_global, const std::map<proteus_index_t,proteus_index_t>& nodes,
                                          proteus_index_t& nElements_global, int& hasElementMarkers,
                                          std::vector<proteus_index_t>& elements, std::vector<proteus_index_t>& elementNodes,
                                          std::vector<double>& elementIds);

  /* I/O, local: stream a Triangle/TetGen .face or .edge file (nNodesPerEntity
     nodes per line) and keep, in file order, the entities with a node among
     the keys of `nodes`: file numbers, file nodes and (headerHasMarkers and the
     header says so) markers. With headerHasMarkers false the header gives only
     the count and markers are not read. False if the file cannot be opened. */
  extern bool readSimplexSubentitiesTouching(const std::string& fileName, const char* format, const char* what,
                                             int nNodesPerEntity, int indexBase, bool headerHasMarkers,
                                             const std::map<proteus_index_t,proteus_index_t>& nodes, proteus_index_t& nEntities_global, int& hasMarkers,
                                             std::vector<proteus_index_t>& numbers, std::vector<proteus_index_t>& entityNodes,
                                             std::vector<int>& ids);

  /* MPI: file number -> new number of the nodes this rank owns after
     repartitioning, sent by the ranks whose default chunk (nodeOffsets_old)
     computed them (old2new_chunk). One exchange. */
  extern std::map<proteus_index_t,proteus_index_t> ownedNodesByFileNumber(const MPI_Comm& comm, const std::vector<proteus_index_t>& nodeOffsets_old,
                                                  const std::vector<proteus_index_t>& nodeOffsets_new,
                                                  const std::vector<proteus_index_t>& old2new_chunk);

  /* MPI: the new numbers of the file nodes `wanted`, from the ranks whose
     default chunk holds them. Collective: every rank calls it. */
  extern std::map<proteus_index_t,proteus_index_t> newNumbersOfFileNodes(const MPI_Comm& comm, const std::vector<proteus_index_t>& nodeOffsets_old,
                                                 const std::vector<proteus_index_t>& old2new_chunk, const std::set<proteus_index_t>& wanted);

  /* MPI: file number -> new number of the subdomain's entities (elements,
     faces, edges), the owned ones numbered from ownedBegin in ascending file
     order and the others asked of a directory on the default chunks of
     [0,nGlobal). owned is a subset of subdomain (ascending). Returns (old, new)
     sorted by old. Collective. An entity of subdomain that no rank owns ends
     the run on every rank, naming `what` (element, face, edge), its number
     and the rank that asked. */
  extern std::vector<std::pair<proteus_index_t,proteus_index_t> > subdomainNumbersByDirectory(const MPI_Comm& comm, proteus_index_t nGlobal, proteus_index_t ownedBegin,
                                                                      const std::set<proteus_index_t>& owned,
                                                                      const std::vector<proteus_index_t>& subdomain,
                                                                      const char* what = "entity");

  /* MPI: flag -1 the subdomain faces the partition cut (exterior to the
     subdomain, interior to the mesh) whose flag is 0, asking each face's owner
     whether it is exterior to the mesh. For partitioners without the global
     mesh (parun -F). Returns the number marked on this rank. Collective.
     flagsFromFile false: the face flags are the builder's exterior 1 /
     interior 0, so every exterior cut face is marked, whatever its flag. */
  extern int markRankInterfacesByOwner(const MPI_Comm& comm, Mesh& mesh, bool flagsFromFile=true);

  extern int partitionNodesFromTriangleFiles(const MPI_Comm& PROTEUS_COMM_WORLD, const char* filebase, int indexBase, Mesh& newMesh, int nNodes_overlap,
                                             std::vector<proteus_index_t>* nodeNumbering_subdomain2input=nullptr,
                                             std::vector<proteus_index_t>* elementNumbering_subdomain2input=nullptr);

  extern int partitionElements(const MPI_Comm& PROTEUS_COMM_WORLD, Mesh& mesh, int nElements_overlap,
                               const int* elementOwners=NULL);

  /* partitionElementsLocal's ownership rules, given the owner of every
     element: a node, an element boundary and an edge belong to the lowest rank
     owning an element that contains them (for an edge, both its nodes).
     Entities no rank reaches get the owner size. */
  extern std::vector<int> nodeOwnersFromElementOwners(const Mesh& mesh, const std::vector<int>& elementOwner, int size);
  extern std::vector<int> elementBoundaryOwnersFromElementOwners(const Mesh& mesh, const std::vector<int>& elementOwner, int size);
  extern std::vector<int> edgeOwnersFromElementOwners(const Mesh& mesh, const std::vector<int>& elementOwner, int size);

  /* Rows begin..end-1 of the element graph (each element's neighbours across its
     boundaries, sorted) in CSR form. Plain C++. */
  extern void elementGraphRows(const Mesh& mesh, proteus_index_t begin, proteus_index_t end, std::vector<proteus_index_t>& offsets, std::vector<proteus_index_t>& neighbors);

  /* MPI and PETSc: every element's owner from MatPartitioning, gathered on every
     rank. Collective. */
  extern std::vector<int> graphElementOwners(const MPI_Comm& PROTEUS_COMM_WORLD, const Mesh& mesh,
                                             const std::valarray<proteus_index_t>& elementOffsets_old);

  /* The local part of partitionElements: rank's part of the partition given the
     owner of every element. No MPI. Renumbers the global mesh in place, as
     partitionElements does. See partitioning.cpp. */
  extern int partitionElementsLocal(Mesh& mesh, int rank, int size, int nElements_overlap,
                                    const std::vector<int>& elementOwner, StageTimer* timer=NULL);

  extern int buildQuadraticSubdomain2GlobalMappings_1d(const MPI_Comm& PROTEUS_COMM_WORLD, Mesh& mesh,
                                                       const int *elementOffsets_subdomain_owned,
                                                       const int *nodeOffsets_subdomain_owned,
                                                       const int *elementNumbering_subdomain2global,
                                                       const int *nodeNumbering_subdomain2global,
                                                       int& nDOF_all_processes,//total number of dofs in whole domain
                                                       int& nDOF_subdomain,//total number of dofs in sub-domain
                                                       int& max_dof_neighbors,//maximum number of neighbors for connectivity of dofs
                                                       int *offsets_subdomain_owned, //starting point of local dofs on each processor (nProcs+1)
                                                       int * subdomain_l2g, //local to global dof mapping on subdomain
                                                       int* subdomain2global,//subdomain dof to global (parallel) numbering
                                                       double * lagrangeNodesArray);//location of nodes corresponding to dofs

  extern int buildQuadraticSubdomain2GlobalMappings_2d(const MPI_Comm& PROTEUS_COMM_WORLD, Mesh& mesh,
                                                       const int *elementBoundaryOffsets_subdomain_owned,
                                                       const int *nodeOffsets_subdomain_owned,
                                                       const int *elementBoundaryNumbering_subdomain2global,
                                                       const int *nodeNumbering_subdomain2global,
                                                       int& nDOF_all_processes,//total number of dofs in whole domain
                                                       int& nDOF_subdomain,//total number of dofs in sub-domain
                                                       int& max_dof_neighbors,//maximum number of neighbors for connectivity of dofs
                                                       int *offsets_subdomain_owned, //starting point of local dofs on each processor (nProcs+1)
                                                       int *subdomain_l2g, //local to global dof mapping on subdomain
                                                       int *subdomain2global,//subdomain dof to global (parallel) numbering
                                                       double * lagrangeNodesArray);//location of nodes corresponding to dofs

  extern int buildQuadraticSubdomain2GlobalMappings_3d(const MPI_Comm& PROTEUS_COMM_WORLD, Mesh& mesh,
                                                       const int *edgeOffsets_subdomain_owned,
                                                       const int *nodeOffsets_subdomain_owned,
                                                       const int *edgeNumbering_subdomain2global,
                                                       const int *nodeNumbering_subdomain2global,
                                                       int& nDOF_all_processes,//total number of dofs in whole domain
                                                       int& nDOF_subdomain,//total number of dofs in sub-domain
                                                       int& max_dof_neighbors,//maximum number of neighbors for connectivity of dofs
                                                       int *offsets_subdomain_owned, //starting point of local dofs on each processor (nProcs+1)
                                                       int *subdomain_l2g, //local to global dof mapping on subdomain
                                                       int *subdomain2global,//subdomain dof to global (parallel) numbering
                                                       double * lagrangeNodesArray);//location of nodes corresponding to dofs

  extern int buildQuadraticCubeSubdomain2GlobalMappings_3d(const MPI_Comm& PROTEUS_COMM_WORLD, Mesh& mesh,
                                                           const int *edgeOffsets_subdomain_owned,
                                                           const int *nodeOffsets_subdomain_owned,
                                                           const int *edgeNumbering_subdomain2global,
                                                           const int *nodeNumbering_subdomain2global,
                                                           int& nDOF_all_processes,//total number of dofs in whole domain
                                                           int& nDOF_subdomain,//total number of dofs in sub-domain
                                                           int& max_dof_neighbors,//maximum number of neighbors for connectivity of dofs
                                                           int *offsets_subdomain_owned, //starting point of local dofs on each processor (nProcs+1)
                                                           int *subdomain_l2g, //local to global dof mapping on subdomain
                                                           int *subdomain2global,//subdomain dof to global (parallel) numbering
                                                           double * lagrangeNodesArray);//location of nodes corresponding to dofs

  extern int buildDiscontinuousGalerkinSubdomain2GlobalMappings(const MPI_Comm& PROTEUS_COMM_WORLD, Mesh& mesh,
                                                                const int *elementOffsets_subdomain_owned,
                                                                const int *elementNumbering_subdomain2global,
                                                                int nDOF_element,
                                                                int& nDOF_all_processes,//total number of dofs in whole domain
                                                                int& nDOF_subdomain,//total number of dofs in sub-domain
                                                                int& max_dof_neighbors,//maximum number of neighbors for connectivity of dofs
                                                                int *offsets_subdomain_owned, //starting point of local dofs on each processor (nProcs+1)
                                                                int * subdomain_l2g, //local to global dof mapping on subdomain
                                                                int* subdomain2global);//subdomain dof to global (parallel) numbering
}//proteus
#endif
