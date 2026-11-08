#include "partitioning.h"
#include "PyEmbeddedFunctions.h"
#include <cstdlib>
#include <vector>
#include <string>
#include <sys/stat.h>
#include <cstring>
#include <cstdio>
#include <cerrno>
#include <unistd.h>
#include <algorithm>
#include <iterator>
#include <sstream>
#include <limits>
#include <stdexcept>

namespace proteus
{
std::string petscPartitioners()
{
  // What PETSc was configured with, in the order PETSc's own default picks one
  std::string have;
#ifdef PETSC_HAVE_PARMETIS
  have += "parmetis ";
#endif
#ifdef PETSC_HAVE_CHACO
  have += "chaco ";
#endif
#ifdef PETSC_HAVE_PARTY
  have += "party ";
#endif
#ifdef PETSC_HAVE_PTSCOTCH
  have += "ptscotch ";
#endif
  if (!have.empty())
    have.pop_back();
  return have;
}

namespace {
// PETSc falls back to MATPARTITIONINGCURRENT, which keeps the input
// partition, when it was built with none of ParMETIS, Chaco, Party or
// PT-Scotch. Say so once instead of silently running on a poor partition.
void warnIfNotRepartitioning(const MPI_Comm& comm, MatPartitioning part)
{
  static bool warned = false;
  int size, rank;
  MPI_Comm_size(comm, &size);
  MPI_Comm_rank(comm, &rank);
  MatPartitioningType type;
  MatPartitioningGetType(part, &type);
  if (!warned && size > 1 && type && std::strcmp(type, MATPARTITIONINGCURRENT) == 0)
    {
      warned = true;
      if (rank == 0)
        std::cerr<<"PROTEUS WARNING: PETSc mesh partitioner is '"<<type
                 <<"', which keeps the initial partition. Build PETSc with PT-Scotch or ParMETIS, or set -mat_partitioning_type."<<std::endl;
    }
}

/* partitionNodes and partitionElements once ran on PETSc index sets and bit
   arrays: ISPartitioningCount, ISPartitioningToNumbering and ISAllGather to
   number the entities each rank owns, and PetscBT masks passed from rank r-1
   to rank r+1 to decide who owns what. What they compute needs neither, and
   with prescribed owners these partitioners now touch no PETSc at all; only
   the graph partitioner (MatPartitioning) remains, on the path without owners.
   The helpers below are that replacement. */

/* The numbering every entity type gets once each has an owner: a rank's owned
   entities are contiguous, ranks in order, and within a rank they keep their
   old order. That is ISPartitioningToNumbering's numbering of a partition laid
   out in contiguous chunks, and the concatenation of each rank's sorted
   std::set of owned entities that ISAllGather used to assemble. */
void numberByOwner(const std::vector<int>& owner, int size,
                   std::valarray<int>& count, std::valarray<int>& offsets,
                   std::vector<int>& old2new, std::vector<int>& new2old)
{
  const int n = int(owner.size());
  count.resize(size);
  count = 0;
  for (int i = 0; i < n; i++)
    count[owner[i]]++;
  offsets.resize(size+1);
  offsets[0] = 0;
  for (int r = 0; r < size; r++)
    offsets[r+1] = offsets[r] + count[r];
  std::vector<int> next(&offsets[0], &offsets[0] + size);
  old2new.resize(n);
  new2old.resize(n);
  for (int i = 0; i < n; i++)
    {
      const int i_new = next[owner[i]]++;
      old2new[i] = i_new;
      new2old[i_new] = i;
    }
}

/* A whole message line to stderr in one write(2): a run about to end in
   MPI_Abort must not have its message split by other output. std::cerr is
   unit-buffered, so `cerr << a << b` is several writes, and under load
   Open MPI's MPI_ABORT banner from another rank landed inside one (C3 r7,
   2026-11-08). One write of less than PIPE_BUF bytes to a pipe is atomic. */
void writeErrorLine(const std::string& line)
{
  std::cerr.flush();
  const std::string text = line + "\n";
  const char* p = text.data();
  size_t left = text.size();
  while (left > 0)
    {
      const ssize_t n = ::write(STDERR_FILENO, p, left);
      if (n <= 0)
        break;
      p += n;
      left -= size_t(n);
    }
}

/* The message, then MPI_Abort on comm. */
[[noreturn]] void abortWith(const MPI_Comm& comm, const std::string& message)
{
  writeErrorLine(message);
  MPI_Abort(comm, 1);
  std::abort();
}

/* Every rank computes the owners of all entities from the same global data, so
   the rule the old mask chain applied in rank order -- an entity belongs to the
   lowest rank that reaches it -- is a minimum, taken locally. Entities start at
   `size', meaning "nobody reached it". */
inline void lowerOwner(std::vector<int>& owner, int i, int rank)
{
  if (rank < owner[i])
    owner[i] = rank;
}

bool everyEntityOwned(const std::vector<int>& owner, int size, const char* what)
{
  for (size_t i = 0; i < owner.size(); i++)
    if (owner[i] >= size)
      {
        std::ostringstream message;
        message<<"partitioning: "<<what<<" "<<i<<" was reached by no rank";
        writeErrorLine(message.str());
        return false;
      }
  return true;
}

void checkEveryEntityOwned(const MPI_Comm& comm, const std::vector<int>& owner, int size, const char* what)
{
  if (!everyEntityOwned(owner,size,what))
    MPI_Abort(comm,1);
}

/* Check prescribed owners and copy them. */
std::vector<int> prescribedOwners(const MPI_Comm& comm, const int* owners, int n, int size, const char* caller, const char* what)
{
  std::vector<int> owner(owners, owners + n);
  for (int i = 0; i < n; i++)
    if (owner[i] < 0 || owner[i] >= size)
      {
        std::ostringstream message;
        message<<caller<<": "<<what<<" "<<i<<" is assigned to rank "<<owner[i]
               <<" but there are "<<size<<" ranks";
        abortWith(comm, message.str());
      }
  return owner;
}

/* Every rank's n ints, concatenated in rank order on every rank: ISAllGather of
   an index set made from them, without PETSc. */
std::vector<int> allGatherInts(const MPI_Comm& comm, const int* mine, int n)
{
  int size;
  MPI_Comm_size(comm,&size);
  std::vector<int> counts(size), displs(size+1,0);
  MPI_Allgather(&n,1,MPI_INT,counts.data(),1,MPI_INT,comm);
  for (int r = 0; r < size; r++)
    displs[r+1] = displs[r] + counts[r];
  std::vector<int> all(displs[size]);
  MPI_Allgatherv(mine,n,MPI_INT,all.data(),counts.data(),displs.data(),MPI_INT,comm);
  return all;
}

/* The graph partitioner's answer -- an index set over this rank's chunk of the
   default contiguous partition, one target rank per entity -- gathered into
   the owner of every entity. */
std::vector<int> gatherPartition(const MPI_Comm& comm, IS partitioning, const std::valarray<int>& offsets_old)
{
  int size;
  MPI_Comm_size(comm,&size);
  PetscInt nLocal;
  ISGetLocalSize(partitioning,&nLocal);
  const PetscInt* local;
  ISGetIndices(partitioning,&local);
  std::vector<int> mine(local, local + nLocal);
  ISRestoreIndices(partitioning,&local);
  std::vector<int> counts(size), displs(size);
  for (int r = 0; r < size; r++)
    {
      counts[r] = offsets_old[r+1] - offsets_old[r];
      displs[r] = offsets_old[r];
    }
  std::vector<int> owner(offsets_old[size]);
  MPI_Allgatherv(mine.data(), int(nLocal), MPI_INT,
                 owner.data(), counts.data(), displs.data(), MPI_INT, comm);
  return owner;
}
}

/* The ownership rules of partitionNodes, which follow from the owner of every
   node. Every rank computes them over the whole mesh, so they are the same on
   every rank and need no communication. Plain C++. An entity no rank reaches
   is given the owner size. */

//An element belongs to the lowest rank that owns one of its nodes.
std::vector<int> elementOwnersFromNodeOwners(const Mesh& mesh, const std::vector<int>& nodeOwner, int size)
{
  std::vector<int> elementOwner(mesh.nElements_global,size);
  for (int nN_global_old = 0; nN_global_old < mesh.nNodes_global; nN_global_old++)
    for (int eN_star_offset = mesh.nodeElementOffsets[nN_global_old];
         eN_star_offset < mesh.nodeElementOffsets[nN_global_old+1]; eN_star_offset++)
      lowerOwner(elementOwner,mesh.nodeElementsArray[eN_star_offset],nodeOwner[nN_global_old]);
  return elementOwner;
}

//An element boundary belongs to the lowest rank that owns one of its nodes,
//reached through the elements in the node's star.
std::vector<int> elementBoundaryOwnersFromNodeOwners(const Mesh& mesh, const std::vector<int>& nodeOwner, int size)
{
  std::vector<int> elementBoundaryOwner(mesh.nElementBoundaries_global,size);
  if (mesh.nNodes_element == 8)
    {

      int lface[6][4] = {{0,1,2,3},
                         {0,1,5,4},
                         {1,2,6,5},
                         {2,3,7,6},
                         {3,0,4,7},
                         {4,5,6,7}};

      for (int nN_global_old = 0; nN_global_old < mesh.nNodes_global; nN_global_old++)
        {
          const int nN_rank = nodeOwner[nN_global_old];
          //now get elements in node star
          for (int offset_old = mesh.nodeElementOffsets[nN_global_old];
               offset_old < mesh.nodeElementOffsets[nN_global_old+1]; offset_old++)
            {
              int eN_star_old = mesh.nodeElementsArray[offset_old];

              //the element boundaries of the element that contain the node
              for (int ebN=0; ebN<mesh.nElementBoundaries_element;ebN++)
                {
                  bool foundNode = false;
                  for (int nNl=0; nNl<mesh.nNodes_elementBoundary ;nNl++)
                    {
                      int nN_global_old_across = mesh.elementNodesArray[eN_star_old*mesh.nNodes_element + lface[ebN][nNl]];
                      if (nN_global_old_across == nN_global_old) foundNode = true;
                    }
                  if (foundNode)
                    lowerOwner(elementBoundaryOwner,
                               mesh.elementBoundariesArray[eN_star_old*mesh.nElementBoundaries_element+ebN],
                               nN_rank);
                }

            }
        }

    }
  else if (mesh.nNodes_element == 4 && mesh.nNodes_elementBoundary == 2)
    {
      //quadrilateral (a tetrahedron also has four nodes and four boundaries,
      //but its boundaries have three nodes): edge j joins local nodes j and j+1. The simplex rule below
      //(every face but the one across from the node) takes an edge to contain the
      //node when it does not, which gave an edge to a rank that held neither of
      //its end points -- and so not both of its cells, leaving its row of a
      //quadratic space incomplete there.
      for (int nN_global_old = 0; nN_global_old < mesh.nNodes_global; nN_global_old++)
        {
          const int nN_rank = nodeOwner[nN_global_old];
          for (int offset_old = mesh.nodeElementOffsets[nN_global_old];
               offset_old < mesh.nodeElementOffsets[nN_global_old+1]; offset_old++)
            {
              int eN_star_old = mesh.nodeElementsArray[offset_old];
              for (int ebN=0; ebN<mesh.nElementBoundaries_element;ebN++)
                {
                  const int n0 = mesh.elementNodesArray[eN_star_old*mesh.nNodes_element + ebN];
                  const int n1 = mesh.elementNodesArray[eN_star_old*mesh.nNodes_element + (ebN+1)%4];
                  if (n0 == nN_global_old || n1 == nN_global_old)
                    lowerOwner(elementBoundaryOwner,
                               mesh.elementBoundariesArray[eN_star_old*mesh.nElementBoundaries_element+ebN],
                               nN_rank);
                }
            }
        }
    }
  else
    {
      for (int nN_global_old = 0; nN_global_old < mesh.nNodes_global; nN_global_old++)
        {
          const int nN_rank = nodeOwner[nN_global_old];
          //now get elements in node star
          for (int offset_old = mesh.nodeElementOffsets[nN_global_old];
               offset_old < mesh.nodeElementOffsets[nN_global_old+1]; offset_old++)
            {
              int eN_star_old = mesh.nodeElementsArray[offset_old];

              //loop through element boundaries on each element, want but want to skip
              //the element boundary across from the owned node
              for (int ebN=0; ebN<mesh.nElementBoundaries_element;ebN++)
                {
                  int nN_global_old_across = mesh.elementNodesArray[eN_star_old*mesh.nNodes_element+ebN];
                  if (nN_global_old_across != nN_global_old)
                    lowerOwner(elementBoundaryOwner,
                               mesh.elementBoundariesArray[eN_star_old*mesh.nElementBoundaries_element+ebN],
                               nN_rank);
                }
            }
        }
    }
  return elementBoundaryOwner;
}

//An edge belongs to the owner of its end with the lower new number.
std::vector<int> edgeOwnersFromNodeOwners(const Mesh& mesh, const std::vector<int>& nodeOwner,
                                          const std::vector<int>& nodeNumbering_global_old2new)
{
  std::vector<int> edgeOwner(mesh.nEdges_global);
  for (int ig = 0; ig < mesh.nEdges_global; ig++)
    {
      const int nN0_global_old = mesh.edgeNodesArray[2*ig];
      const int nN1_global_old = mesh.edgeNodesArray[2*ig+1];
      edgeOwner[ig] = nodeOwner[nodeNumbering_global_old2new[nN0_global_old] < nodeNumbering_global_old2new[nN1_global_old] ?
                                nN0_global_old : nN1_global_old];
    }
  return edgeOwner;
}

/* The default partition of n entities among size ranks: contiguous chunks of
   near-equal size, in input order (the first n%size ranks get one more). */
std::valarray<int> defaultChunkOffsets(int n, int size)
{
  std::valarray<int> offsets(size+1);
  offsets[0] = 0;
  for (int sdN=0; sdN < size; sdN++)
    offsets[sdN+1] = offsets[sdN] + n/size + (n%size > sdN);
  return offsets;
}

/* Rows begin..end-1 of the node graph the graph partitioner is given: each
   node's star (its edge neighbours), sorted, in CSR form. */
void nodeGraphRows(const Mesh& mesh, int begin, int end, std::vector<int>& offsets, std::vector<int>& neighbors)
{
  offsets.assign(1,0);
  neighbors.clear();
  for (int nN_global = begin; nN_global < end; nN_global++)
    {
      for (int offset_global = mesh.nodeStarOffsets[nN_global];
           offset_global < mesh.nodeStarOffsets[nN_global+1]; offset_global++)
        neighbors.push_back(mesh.nodeStarArray[offset_global]);
      offsets.push_back(int(neighbors.size()));
      std::sort(neighbors.begin()+offsets[offsets.size()-2],neighbors.end());
    }
}

/* MPI: the owner of every node from the graph partitioner. Each rank hands
   MatPartitioning its chunk of the default partition (nodeOffsets_old) as
   rows of the node graph, with the nodes' weights if given, and the answer is
   gathered so every rank holds every owner. Collective. */
std::vector<int> graphNodeOwners(const MPI_Comm& PROTEUS_COMM_WORLD, const Mesh& mesh,
                                 const std::valarray<int>& nodeOffsets_old, const int* nodeWeights)
{
  using namespace std;
  int ierr,rank;
  MPI_Comm_rank(PROTEUS_COMM_WORLD,&rank);
  //Determine nodal connectivity on local processor, (local node star array)
  int nNodes_subdomain = (nodeOffsets_old[rank+1] - nodeOffsets_old[rank]);
  std::vector<int> rowOffsets, rowNeighbors;
  nodeGraphRows(mesh,nodeOffsets_old[rank],nodeOffsets_old[rank+1],rowOffsets,rowNeighbors);
  //MatCreateMPIAdj takes ownership of PetscMalloc'd arrays
  PetscInt *nodeNeighborsOffsets_subdomain,*nodeNeighbors_subdomain;
  PetscMalloc(sizeof(PetscInt)*(nNodes_subdomain+1),&nodeNeighborsOffsets_subdomain);
  PetscMalloc(sizeof(PetscInt)*(nNodes_subdomain*mesh.max_nNodeNeighbors_node),&nodeNeighbors_subdomain);
  for (int nN = 0; nN < nNodes_subdomain+1; nN++)
    nodeNeighborsOffsets_subdomain[nN] = rowOffsets[nN];
  for (size_t k = 0; k < rowNeighbors.size(); k++)
    nodeNeighbors_subdomain[k] = rowNeighbors[k];
  //Generate new nodal partition using PETSc interface
  IS nodePartitioningIS_new;
  Mat petscAdjacency;
  //   MatCreateMPIAdj(PROTEUS_COMM_WORLD,
  //              nNodes_subdomain, mesh.nNodes_global,
  //              &nodeNeighborsOffsets_subdomain[0], &nodeNeighbors_subdomain[0],
  //              &weights_subdomain[0],//PETSC_NULL,//ignore weighting for now
  //              &petscAdjacency);
  ierr = MatCreateMPIAdj(PROTEUS_COMM_WORLD,
                         nNodes_subdomain,
                         mesh.nNodes_global,
                         nodeNeighborsOffsets_subdomain,
                         nodeNeighbors_subdomain,
                         PETSC_NULL,//weights_subdomain,
                         &petscAdjacency);CHKERRABORT(PROTEUS_COMM_WORLD, ierr);
  MatPartitioning petscPartition;
  MatPartitioningCreate(PROTEUS_COMM_WORLD,&petscPartition);
  MatPartitioningSetAdjacency(petscPartition,petscAdjacency);
  MatPartitioningSetFromOptions(petscPartition);
  warnIfNotRepartitioning(PROTEUS_COMM_WORLD, petscPartition);
  if (nodeWeights != NULL)
    {
      /* The caller says what each node costs -- for the coarsest level of a hierarchy, the
         work of everything below it that will inherit its owner -- so the cut balances
         that instead of the node count. The array is this rank's slice of the default
         chunks, and PETSc takes ownership of it. */
      PetscInt *vertex_weights;
      ierr = PetscMalloc(sizeof(PetscInt)*nNodes_subdomain,&vertex_weights);CHKERRABORT(PROTEUS_COMM_WORLD, ierr);
      for (int nN = 0; nN < nNodes_subdomain; nN++)
        vertex_weights[nN] = nodeWeights[nodeOffsets_old[rank]+nN];
      ierr = MatPartitioningSetVertexWeights(petscPartition,vertex_weights);CHKERRABORT(PROTEUS_COMM_WORLD, ierr);
    }

  //get petsc index set that has the new subdomain number for each node
  MatPartitioningApply(petscPartition,&nodePartitioningIS_new);
  MatPartitioningDestroy(&petscPartition); //gets petscAdjacency too I believe
  std::vector<int> nodeOwner = gatherPartition(PROTEUS_COMM_WORLD,nodePartitioningIS_new,nodeOffsets_old);
  ISDestroy(&nodePartitioningIS_new);
  return nodeOwner;
}

int partitionNodes(const MPI_Comm& PROTEUS_COMM_WORLD,  Mesh& mesh, int nNodes_overlap,
                   const int* nodeOwners, const int* nodeWeights)
{
  using namespace std;
  int ierr,size,rank;

  ierr = MPI_Comm_size(PROTEUS_COMM_WORLD,&size);
  ierr = MPI_Comm_rank(PROTEUS_COMM_WORLD,&rank);
  StageTimer timer(PROTEUS_COMM_WORLD,"partitionNodes");
  /***********************************************************************
    partition domain based on nodes rather than elements, basically repeats
    partitionElements with this one modification

    right now generates equivalent of 1 layer of overlap regardless of input

    1. Partition nodes in the default partition of contiguous chunks with
       input ordering

    2. Determine nodal connectivity on local processor

    3. Generate new nodal partition using PETSc interface

    4. Collect elements containing locally owned nodes and assign ownership

    5. Generate global element numbering for new subdomain ownership

    6. Create overlap (ghost) information for nodes and elements

    7. March through additional layers of overlap if requested,

    8. Build subdomain meshes in new numbering
  ***********************************************************************/
  //
  //1. Build default nodal partition
  //
  //compute offsets to build processor (local) to global ordering for nodes
  //in default partitioning
  // printf("rank %d partitionNodes nNodes_global %d\n", rank, mesh.nNodes_global);
  valarray<int> nodeOffsets_old = defaultChunkOffsets(mesh.nNodes_global,size);
  //
  //2.-3. The owner of every node: prescribed, or from the graph partitioner
  //
  std::vector<int> nodeOwner;
  if (nodeOwners != NULL)
    {
      /* The caller dictates the partition: nodeOwners[n] is the rank that owns
         node n, in the numbering the mesh has now. This is how a finer level of
         a multilevel hierarchy inherits its coarser level's partition, so that
         the transfer between them is local instead of an all-to-all, and how a
         regular grid is cut into bricks (MeshTools.cartesianNodeOwners). Nothing
         on this path uses PETSc. */
      nodeOwner = prescribedOwners(PROTEUS_COMM_WORLD,nodeOwners,mesh.nNodes_global,size,"partitionNodes","node");
    }
  else
    nodeOwner = graphNodeOwners(PROTEUS_COMM_WORLD,mesh,nodeOffsets_old,nodeWeights);

  timer.mark("node_owners");
  //4.-8. Everything else follows from the owners without communication
  if (partitionNodesLocal(mesh, rank, size, nNodes_overlap, nodeOwner, &timer) != 0)
    MPI_Abort(PROTEUS_COMM_WORLD,1);
  timer.report();
  return 0;
}

namespace
{
/* The ghost entities of one rank, in the new numbering, as sorted sets. */
struct OverlapSets
{
  std::set<int> nodes, elements, elementBoundaries, edges;
};

/* Steps 6-7b of partitionNodesLocal: the overlap (ghost) nodes, elements,
   element boundaries and edges of `rank`, given the global numberings and
   offsets of every entity type (new numbering: each rank's owned entities
   contiguous), the elements' element boundaries in the new numbering
   (elementBoundariesArray_new), and the map (new node 0, new node 1) -> new
   edge. The first layer is everything in the stars of the owned nodes that
   another rank owns; each further layer (nNodes_overlap > 1) is the same for
   the nodes the previous layer added. A last pass adds every node of every
   owned or ghost element, which is a no-op for simplices (see 7b below).
   Reads the mesh and the arguments only. */
OverlapSets nodeOverlapSets(const Mesh& mesh, int rank, int nNodes_overlap,
                            const std::vector<int>& nodeNumbering_global_old2new,
                            const std::vector<int>& nodeNumbering_global_new2old,
                            const std::vector<int>& elementNumbering_global_old2new,
                            const std::vector<int>& elementNumbering_global_new2old,
                            const std::valarray<int>& nodeOffsets_new,
                            const std::valarray<int>& elementOffsets_new,
                            const std::valarray<int>& elementBoundaryOffsets_new,
                            const std::valarray<int>& edgeOffsets_new,
                            const std::valarray<int>& elementBoundariesArray_new,
                            const std::map<NodeTuple<2>, int>& nodesEdgeMap_global_new,
                            const std::set<int>& elements_subdomain_owned)
{
  using namespace std;
  OverlapSets overlap;
  //what the stars of one node (in the old numbering) add to the sets given
  auto addNodeStar = [&](int nN_global_old, OverlapSets& into)
  {
    //nodes in the node star, if another rank owns them
    for (int offset_old = mesh.nodeStarOffsets[nN_global_old];
         offset_old < mesh.nodeStarOffsets[nN_global_old+1]; offset_old++)
      {
        int nN_neig_old = mesh.nodeStarArray[offset_old];
        int nN_neig_new = nodeNumbering_global_old2new[nN_neig_old];
        bool offproc = nN_neig_new >=  nodeOffsets_new[rank+1] || nN_neig_new < nodeOffsets_new[rank];
        if (offproc)
          into.nodes.insert(nN_neig_new);
      }
    //now the elements, elementBoundaries, and edges in the node star
    for (int offset_old = mesh.nodeElementOffsets[nN_global_old];
         offset_old < mesh.nodeElementOffsets[nN_global_old+1]; offset_old++)
      {
        int eN_star_old = mesh.nodeElementsArray[offset_old];
        int eN_star_new = elementNumbering_global_old2new[eN_star_old];
        bool offproc = eN_star_new >= elementOffsets_new[rank+1] || eN_star_new < elementOffsets_new[rank];
        if (offproc)
          into.elements.insert(eN_star_new);
        for (int ebN=0; ebN<mesh.nElementBoundaries_element;ebN++)
          {
            int ebN_global=elementBoundariesArray_new[eN_star_new*mesh.nElementBoundaries_element+ebN];
            if (ebN_global < elementBoundaryOffsets_new[rank] || ebN_global >= elementBoundaryOffsets_new[rank+1])
              into.elementBoundaries.insert(ebN_global);
          }
        for (int nN0=0;nN0<mesh.nNodes_element;nN0++)
          for (int nN1=nN0+1; nN1<mesh.nNodes_element;nN1++)
            {
              const int nN0_global_old = mesh.elementNodesArray[eN_star_old*mesh.nNodes_element+nN0];
              const int nN1_global_old = mesh.elementNodesArray[eN_star_old*mesh.nNodes_element+nN1];
              int nodes[2];
              nodes[0] = nodeNumbering_global_old2new[nN0_global_old];
              nodes[1] = nodeNumbering_global_old2new[nN1_global_old];
              NodeTuple<2> et(nodes);
              //Only some pairs of an element's nodes are edges: every pair in a
              //simplex, but not the face and body diagonals of a quadrilateral
              //or hexahedron. operator[] would insert a default entry for a
              //diagonal and report it as edge 0.
              map<NodeTuple<2>, int>::const_iterator edge_it = nodesEdgeMap_global_new.find(et);
              if (edge_it == nodesEdgeMap_global_new.end())
                continue;
              const int edge_global = edge_it->second;
              if (edge_global < edgeOffsets_new[rank] || edge_global >= edgeOffsets_new[rank+1])
                into.edges.insert(edge_global);
            }//edges
      }//elements in node star
  };
  //
  //6. Elements, nodes, element boundaries and edges in the stars of the owned
  //   nodes that are not locally owned: the ghosts
  //
  for (int nN = nodeOffsets_new[rank]; nN < nodeOffsets_new[rank+1]; nN++)
    addNodeStar(nodeNumbering_global_new2old[nN], overlap);
  //
  //7. If we want more layers of overlap, march out from the nodes the last
  //   layer added, in the old numbering for connectivity
  //
  int overlap_remaining = nNodes_overlap -1; //default gives 1 layer overlap
  set<int> last_nodes_added2overlap = overlap.nodes;
  while (overlap_remaining > 0)
    {
      OverlapSets layer;
      for (set<int>::const_iterator nN_p = last_nodes_added2overlap.begin();
           nN_p != last_nodes_added2overlap.end(); nN_p++)
        addNodeStar(nodeNumbering_global_new2old[*nN_p], layer);
      last_nodes_added2overlap.clear();
      set_difference(layer.nodes.begin(),layer.nodes.end(),
                     overlap.nodes.begin(),overlap.nodes.end(),
                     std::inserter(last_nodes_added2overlap,last_nodes_added2overlap.begin()));
      overlap.nodes.insert(layer.nodes.begin(),layer.nodes.end());
      overlap.elements.insert(layer.elements.begin(),layer.elements.end());
      overlap.elementBoundaries.insert(layer.elementBoundaries.begin(),layer.elementBoundaries.end());
      overlap.edges.insert(layer.edges.begin(),layer.edges.end());
      overlap_remaining--;
    }//overlap loop
  //
  //7b. Every node of every element in the subdomain has to be in the subdomain.
  //    The overlap nodes above come from the node star, which is built from
  //    edges, but the elements come from node-to-element membership. For a
  //    simplex every node of an element is an edge neighbor of any other, so the
  //    two agree. For a quadrilateral or hexahedron they do not: the face and
  //    body diagonal nodes of an element are not edge neighbors, so an element
  //    in the star of an owned node could reference nodes the subdomain never
  //    received. nodeNumbering_global2subdomain is a std::map read with
  //    operator[], so a missing node did not fail -- it silently became
  //    subdomain node 0, giving distorted elements and, on more ranks, crashes.
  //    This adds nothing for simplices.
  //
  const int nOwnedNodesBegin = nodeOffsets_new[rank];
  const int nOwnedNodesEnd = nodeOffsets_new[rank+1];
  for (set<int>::const_iterator eN_oldp = elements_subdomain_owned.begin();
       eN_oldp != elements_subdomain_owned.end(); eN_oldp++)
    for (int nN = 0; nN < mesh.nNodes_element; nN++)
      {
        const int nN_new = nodeNumbering_global_old2new[mesh.elementNodesArray[(*eN_oldp)*mesh.nNodes_element+nN]];
        if (nN_new < nOwnedNodesBegin || nN_new >= nOwnedNodesEnd)
          overlap.nodes.insert(nN_new);
      }
  for (set<int>::const_iterator eN_newp = overlap.elements.begin();
       eN_newp != overlap.elements.end(); eN_newp++)
    {
      const int eN_old = elementNumbering_global_new2old[*eN_newp];
      for (int nN = 0; nN < mesh.nNodes_element; nN++)
        {
          const int nN_new = nodeNumbering_global_old2new[mesh.elementNodesArray[eN_old*mesh.nNodes_element+nN]];
          if (nN_new < nOwnedNodesBegin || nN_new >= nOwnedNodesEnd)
            overlap.nodes.insert(nN_new);
        }
    }
  return overlap;
}

//the subdomain's edge flags from the global mesh's (the .edge markers a
//reader kept), through the subdomain's edges' new global numbers and the
//global edges' new-to-old permutation; 0 when the global mesh has none
void subdomainEdgeMaterialTypes(const Mesh& mesh, Mesh& subdomain,
                                const std::valarray<int>& edgeNumbering_subdomain2global,
                                const std::vector<int>& edgeNumbering_global_new2old)
{
  if (subdomain.edgeMaterialTypes != NULL)
    delete [] subdomain.edgeMaterialTypes;
  subdomain.edgeMaterialTypes = new int[subdomain.nEdges_global]();
  if (mesh.edgeMaterialTypes == NULL)
    return;
  for (int i = 0; i < subdomain.nEdges_global; i++)
    subdomain.edgeMaterialTypes[i] = mesh.edgeMaterialTypes[edgeNumbering_global_new2old[edgeNumbering_subdomain2global[i]]];
}
}//namespace

/* Steps 4-8 of partitionNodes: given the owner of every node, which every rank
   holds, this rank's part of the partition. Plain C++ -- no MPI, no PETSc, no
   files -- so one process can compute any rank's subdomain (rank of size).
   Writes the subdomain mesh to mesh.subdomainp and the offsets, the
   subdomain-to-global maps and the old-to-new permutations to mesh, as
   partitionNodes does. Returns nonzero, having said why on stderr, if some
   element or element boundary is reached by no rank. */
int partitionNodesLocal(Mesh& mesh, int rank, int size, int nNodes_overlap,
                        const std::vector<int>& nodeOwner, StageTimer* timer)
{
  using namespace std;
  //number the nodes: each rank's owned nodes contiguous, in their old order
  valarray<int> nNodes_subdomain_new, nodeOffsets_new;
  std::vector<int> nodeNumbering_global_old2new, nodeNumbering_global_new2old;
  numberByOwner(nodeOwner,size,nNodes_subdomain_new,nodeOffsets_new,
                nodeNumbering_global_old2new,nodeNumbering_global_new2old);
  /* Keep this permutation on the mesh. Partitioning renumbers the global nodes
     so each rank's owned ones are contiguous, and it is that numbering which
     nodeNumbering_subdomain2global refers to. Anything built against the
     original numbering -- a multilevel hierarchy's node parents, say -- has to
     be read through this map before it can be used alongside the partitioned
     vectors, and nothing else records the way back. */
  if (mesh.nodeNumbering_global_old2new != NULL)
    delete [] mesh.nodeNumbering_global_old2new;
  mesh.nodeNumbering_global_old2new = new int[mesh.nNodes_global];
  for (int nN=0;nN<mesh.nNodes_global;nN++)
    {
      mesh.nodeNumbering_global_old2new[nN] = nodeNumbering_global_old2new[nN];
    }


  //   //mwf debug
  //   if (rank == 0)
  //     {
  //       for (int sdN = 0; sdN < size+1; sdN++)
  //    std::cout<<"partitionNodes rank= "<<rank<<" nodeOffset["<<sdN<<"]= "<<nodeOffsets_new[sdN]<<std::endl;
  //       for (int nN = 0; nN < mesh.nNodes_global; nN++)
  //    {
  //      std::cout<<"partitionNodes rank= "<<rank<<" nN= "<<nN<<" old2new= "<<nodeNumbering_global_old2new[nN]<<" new2old= "<<nodeNumbering_global_new2old[nN]<<std::endl;
  //    }
  //       for (int sdN = 0; sdN < size; sdN++)
  //    {
  //      std::cout<<"============"<<std::endl;
  //      std::cout<<"partitionNodes rank= "<<sdN<<" nNodes_owned= "<<nodeOffsets_new[sdN+1]-nodeOffsets_new[sdN]<<" = "<<std::endl;
  //      for (int nN = nodeOffsets_new[sdN]; nN < nodeOffsets_new[sdN+1]; nN++)
  //        {
  //          std::cout<<"new number= "<<nN<<" <--> old number "<<nodeNumbering_global_new2old[nN]<<" x,y,z= "
  //                   <<mesh.nodeArray[nodeNumbering_global_new2old[nN]*3+0]<<" , "
  //                   <<mesh.nodeArray[nodeNumbering_global_new2old[nN]*3+1]<<" , "
  //                   <<mesh.nodeArray[nodeNumbering_global_new2old[nN]*3+2]<<std::endl;
  //        }
  //    }

  //     }
  if (timer) timer->mark("node_numbering");
  //
  //4. To build subdomain meshes, go through and collect elements containing
  //   the locally owned nodes. Assign processor ownership of elements
  //
  //   An element belongs to the lowest rank that owns one of its nodes. This
  //   used to be decided by a mask of elements passed from rank to rank, each
  //   marking the unmarked elements in its owned nodes' stars; the same
  //   traversal over every node, keeping the lowest rank, gives the same owners
  //   with no communication.
  std::vector<int> elementOwner = elementOwnersFromNodeOwners(mesh,nodeOwner,size);
  if (!everyEntityOwned(elementOwner,size,"element"))
    return 1;
  //store owned element numbers (in old numbering)
  set<int> elements_subdomain_owned;
  for (int eN = 0; eN < mesh.nElements_global; eN++)
    if (elementOwner[eN] == rank)
      elements_subdomain_owned.insert(elements_subdomain_owned.end(),eN);

  //
  //5. Generate global element numbering corresponding to new subdomain ownership
  //
  valarray<int> nElements_subdomain_new, elementOffsets_new;
  std::vector<int> elementNumbering_global_old2new, elementNumbering_global_new2old;
  numberByOwner(elementOwner,size,nElements_subdomain_new,elementOffsets_new,
                elementNumbering_global_old2new,elementNumbering_global_new2old);
  if (mesh.elementNumbering_global_old2new != NULL)
    delete [] mesh.elementNumbering_global_old2new;
  mesh.elementNumbering_global_old2new = new int[mesh.nElements_global];
  for (int eN = 0; eN < mesh.nElements_global; eN++)
    mesh.elementNumbering_global_old2new[eN] = elementNumbering_global_old2new[eN];



  if (timer) timer->mark("elements");
  //4b,5b. repeat process to build global face numbering
  //first get element --> element boundaries array for new element but old element boundary numbering
  valarray<int> elementBoundariesArray_new(mesh.nElements_global*mesh.nElementBoundaries_element);
  for (int eN=0; eN < mesh.nElements_global; eN++)
    for (int ebN=0; ebN < mesh.nElementBoundaries_element; ebN++)
      {
        elementBoundariesArray_new[eN*mesh.nElementBoundaries_element+ebN] =
          mesh.elementBoundariesArray[elementNumbering_global_new2old[eN]*mesh.nElementBoundaries_element+ebN];
          // printf("rank %d eN %d ebN %d elementBoundariesArray_new[%d] = %d\n", rank, eN, ebN, eN*mesh.nElementBoundaries_element+ebN, elementBoundariesArray_new[eN*mesh.nElementBoundaries_element+ebN]);
      }
  //A face belongs to the lowest rank that reaches it from one of its owned
  //nodes: going through the elements in an owned node's star picks up the faces
  //of those elements selected below, which can lie on the "outside" of the
  //owned nodes' stars. As for elements, this is the rank-ordered mask chain's
  //rule computed over every node at once.
  std::vector<int> elementBoundaryOwner = elementBoundaryOwnersFromNodeOwners(mesh,nodeOwner,size);
  if (!everyEntityOwned(elementBoundaryOwner,size,"element boundary"))
    return 1;
  //Now as with elements and nodes build a global face numbering
  //resetting the face based information is a little different since much of this is currently built below based
  //on the element and node information
  //
  valarray<int> nElementBoundaries_subdomain_new, elementBoundaryOffsets_new;
  std::vector<int> elementBoundaryNumbering_old2new_global, elementBoundaryNumbering_global_new2old;
  numberByOwner(elementBoundaryOwner,size,nElementBoundaries_subdomain_new,elementBoundaryOffsets_new,
                elementBoundaryNumbering_old2new_global,elementBoundaryNumbering_global_new2old);
  for (int eN=0;eN < mesh.nElements_global; eN++)
    {
      // printf("rank %d eN %d nElementBoundaries_element %d\n", rank, eN, mesh.nElementBoundaries_element);
      int ebN_old;
      for (int ebN=0;ebN < mesh.nElementBoundaries_element; ebN++)
        {
          ebN_old = elementBoundariesArray_new[eN*mesh.nElementBoundaries_element+ebN];
          // printf("rank %d eN %d ebN %d old ebN %d elementBoundariesArray_new %d \n", rank, eN, ebN, ebN_old, elementBoundaryNumbering_old2new_global[ebN_old]);
          elementBoundariesArray_new[eN*mesh.nElementBoundaries_element+ebN] = elementBoundaryNumbering_old2new_global[ebN_old];
        }
    }

  if (timer) timer->mark("element_boundaries");
  //4c,5c. Build global edge numbering as well ownership is determined by who owns the left (0) node
  //    of the edge: the one with the lower new number
  std::vector<int> edgeOwner = edgeOwnersFromNodeOwners(mesh,nodeOwner,nodeNumbering_global_old2new);
  valarray<int> nEdges_subdomain_new, edgeOffsets_new;
  std::vector<int> edgeNumbering_old2new_global, edgeNumbering_global_new2old;
  numberByOwner(edgeOwner,size,nEdges_subdomain_new,edgeOffsets_new,
                edgeNumbering_old2new_global,edgeNumbering_global_new2old);

  //create  array with (new edge) --> (new node 0, new node 1)
  //and map from (new node 0, new node 1) --> (new global edge)
  valarray<int> edgeNodesArray_newNodesAndEdges(2*mesh.nEdges_global);
  map<NodeTuple<2>, int > nodesEdgeMap_global_new;
  for (int ig = 0; ig < mesh.nEdges_global; ig++)
    {
      const int nN0_global_old = mesh.edgeNodesArray[2*ig];
      const int nN1_global_old = mesh.edgeNodesArray[2*ig+1];
      const int nN0_global     = nodeNumbering_global_old2new[nN0_global_old];
      const int nN1_global     = nodeNumbering_global_old2new[nN1_global_old];

      const int edge_new = edgeNumbering_old2new_global[ig];
      edgeNodesArray_newNodesAndEdges[edge_new*2+0] = nN0_global;
      edgeNodesArray_newNodesAndEdges[edge_new*2+1] = nN1_global;
      int nodes[2];
      nodes[0] = nN0_global;
      nodes[1] = nN1_global;
      NodeTuple<2> et(nodes);
      nodesEdgeMap_global_new[et] = edge_new;
    }






  if (timer) timer->mark("edges");
  //
  //6-7b. The ghost nodes, elements, element boundaries and edges, with the extra layers of overlap if asked for
  //
  OverlapSets overlapSets = nodeOverlapSets(mesh, rank, nNodes_overlap,
                                            nodeNumbering_global_old2new, nodeNumbering_global_new2old,
                                            elementNumbering_global_old2new, elementNumbering_global_new2old,
                                            nodeOffsets_new, elementOffsets_new, elementBoundaryOffsets_new, edgeOffsets_new,
                                            elementBoundariesArray_new, nodesEdgeMap_global_new,
                                            elements_subdomain_owned);
  const set<int>& nodes_overlap = overlapSets.nodes;
  const set<int>& elements_overlap = overlapSets.elements;
  const set<int>& elementBoundaries_overlap = overlapSets.elementBoundaries;
  const set<int>& edges_overlap = overlapSets.edges;

  if (timer) timer->mark("overlap");
  //
  //8. Build subdomain meshes in new numbering, assumes memory not allocated in subdomain mesh
  //
  if (mesh.subdomainp == NULL)
    mesh.subdomainp = new Mesh();
  mesh.subdomainp->nElements_global = nElements_subdomain_new[rank] + elements_overlap.size();
  mesh.subdomainp->nNodes_global    = nNodes_subdomain_new[rank] + nodes_overlap.size();
  mesh.subdomainp->nElementBoundaries_global = nElementBoundaries_subdomain_new[rank]+elementBoundaries_overlap.size();
  mesh.subdomainp->nEdges_global   = nEdges_subdomain_new[rank]+edges_overlap.size();
  mesh.subdomainp->nNodes_element   = mesh.nNodes_element;
  mesh.subdomainp->nNodes_elementBoundary = mesh.nNodes_elementBoundary;
  mesh.subdomainp->nElementBoundaries_element = mesh.nElementBoundaries_element;
  //subdomain 2 global mappings (including ghost info)
  valarray<int> nodeNumbering_subdomain2global(mesh.subdomainp->nNodes_global);
  valarray<int> elementNumbering_subdomain2global(mesh.subdomainp->nElements_global);
  valarray<int> elementBoundaryNumbering_subdomain2global(mesh.subdomainp->nElementBoundaries_global);
  valarray<int> edgeNumbering_subdomain2global(mesh.subdomainp->nEdges_global);
  map<int,int> nodeNumbering_global2subdomain;
  map<int,int> elementBoundaryNumbering_global2subdomain;
  mesh.subdomainp->nodeArray = new double[mesh.subdomainp->nNodes_global*3];
  mesh.subdomainp->nodeMaterialTypes = new int[mesh.subdomainp->nNodes_global];
  //locally owned
  // printf("rank %d nNodes_subdomain_new = %d\n", rank, nNodes_subdomain_new[rank]);
  for (int nN = 0; nN < nNodes_subdomain_new[rank]; nN++)
    {
      int nN_global_new = nN + nodeOffsets_new[rank];
      int nN_global_old = nodeNumbering_global_new2old[nN_global_new];
      nodeNumbering_subdomain2global[nN] = nN_global_new;
      nodeNumbering_global2subdomain[nN_global_new] = nN;
      mesh.subdomainp->nodeArray[nN*3+0] = mesh.nodeArray[nN_global_old*3+0];
      mesh.subdomainp->nodeArray[nN*3+1] = mesh.nodeArray[nN_global_old*3+1];
      mesh.subdomainp->nodeArray[nN*3+2] = mesh.nodeArray[nN_global_old*3+2];
      // printf("rank %d nN %d nN_global_new %d nN_global_old %d nodeArray[%d] = (%f,%f,%f) nodeMaterialTypes[%d] = %d\n",
        // rank, nN, nN_global_new, nN_global_old, nN*3,
        // mesh.subdomainp->nodeArray[nN*3+0],
        // mesh.subdomainp->nodeArray[nN*3+1],
        // mesh.subdomainp->nodeArray[nN*3+2],
        // mesh.subdomainp->nodeMaterialTypes[nN]);
      
      mesh.subdomainp->nodeMaterialTypes[nN] = mesh.nodeMaterialTypes[nN_global_old];
    }
  //printf("Hello 10\n");
  //ghost
  //note: sets in C++ are sorted so the overlap is laid out in
  //contiguous chunks corresponding to the partitions
  set<int>::iterator nN_p = nodes_overlap.begin();
  for (int nN = nNodes_subdomain_new[rank]; nN < nNodes_subdomain_new[rank] + int(nodes_overlap.size()); nN++)
  {
    int nN_global_new = *nN_p++;
    int nN_global_old = nodeNumbering_global_new2old[nN_global_new];
    nodeNumbering_subdomain2global[nN] = nN_global_new;
    nodeNumbering_global2subdomain[nN_global_new] = nN;
    mesh.subdomainp->nodeArray[nN*3+0] = mesh.nodeArray[nN_global_old*3+0];
    mesh.subdomainp->nodeArray[nN*3+1] = mesh.nodeArray[nN_global_old*3+1];
    mesh.subdomainp->nodeArray[nN*3+2] = mesh.nodeArray[nN_global_old*3+2];
    mesh.subdomainp->nodeMaterialTypes[nN] = mesh.nodeMaterialTypes[nN_global_old];
  }
  mesh.subdomainp->elementNodesArray = new int[mesh.subdomainp->nElements_global*mesh.subdomainp->nNodes_element];
  mesh.subdomainp->elementMaterialTypes = new int[mesh.subdomainp->nElements_global];
  //locally owned
  for (int eN = 0; eN < nElements_subdomain_new[rank]; eN++)
  {
    int eN_global_new = elementOffsets_new[rank] + eN;
    int eN_global_old = elementNumbering_global_new2old[eN_global_new];
    elementNumbering_subdomain2global[eN] = eN_global_new;
    mesh.subdomainp->elementMaterialTypes[eN] = mesh.elementMaterialTypes[eN_global_old];
    for (int nN =  0; nN < mesh.subdomainp->nNodes_element; nN++)
    {
      int nN_global_old = mesh.elementNodesArray[eN_global_old*mesh.nNodes_element + nN];
      int nN_global_new = nodeNumbering_global_old2new[nN_global_old];
      int nN_subdomain  = nodeNumbering_global2subdomain[nN_global_new];
      mesh.subdomainp->elementNodesArray[eN*mesh.subdomainp->nNodes_element + nN]= nN_subdomain;
    }
    
  }
  //ghost
  set<int>::iterator eN_p = elements_overlap.begin();
  for (int eN = nElements_subdomain_new[rank]; eN < nElements_subdomain_new[rank] + int(elements_overlap.size()); eN++)
  {
    int eN_global_new = *eN_p++;
    int eN_global_old = elementNumbering_global_new2old[eN_global_new];
    elementNumbering_subdomain2global[eN] = eN_global_new;
    mesh.subdomainp->elementMaterialTypes[eN] = mesh.elementMaterialTypes[eN_global_old];
    for (int nN =  0; nN < mesh.subdomainp->nNodes_element; nN++)
    {
      int nN_global_old = mesh.elementNodesArray[eN_global_old*mesh.nNodes_element + nN];
      int nN_global_new = nodeNumbering_global_old2new[nN_global_old];
      int nN_subdomain  = nodeNumbering_global2subdomain[nN_global_new];
      mesh.subdomainp->elementNodesArray[eN*mesh.subdomainp->nNodes_element + nN]= nN_subdomain;
    }
  }
  //element boundaries
  //locally owned
  //printf("Hello 11\n");
  for (int ebN=0; ebN < nElementBoundaries_subdomain_new[rank]; ebN++)
  {
    int ebN_global = ebN + elementBoundaryOffsets_new[rank];
    elementBoundaryNumbering_subdomain2global[ebN]=ebN_global;
    elementBoundaryNumbering_global2subdomain[ebN_global] = ebN;
  }
  //ghost
  set<int>::iterator ebN_p = elementBoundaries_overlap.begin();
  for(int ebN=nElementBoundaries_subdomain_new[rank];ebN < nElementBoundaries_subdomain_new[rank] + int(elementBoundaries_overlap.size()); ebN++)
  {
    int ebN_global = *ebN_p++;
    elementBoundaryNumbering_subdomain2global[ebN] = ebN_global;
    elementBoundaryNumbering_global2subdomain[ebN_global] = ebN;
  }
  //need elementBoundariesArray to assign consistent numbering on subdomain
  mesh.subdomainp->elementBoundariesArray =
  new int[mesh.subdomainp->nElements_global*mesh.subdomainp->nElementBoundaries_element];
  for (int eN=0;eN<nElements_subdomain_new[rank];eN++)
  {
    int eN_global = eN+elementOffsets_new[rank];
    for (int ebN=0;ebN<mesh.subdomainp->nElementBoundaries_element;ebN++)
    mesh.subdomainp->elementBoundariesArray[eN*mesh.subdomainp->nElementBoundaries_element+ebN] =
  elementBoundaryNumbering_global2subdomain[elementBoundariesArray_new[eN_global*mesh.nElementBoundaries_element + ebN]];
}
//ghost elements
set<int>::iterator eN_p2 = elements_overlap.begin();
for (int eN = nElements_subdomain_new[rank]; eN < nElements_subdomain_new[rank] + int(elements_overlap.size()); eN++)
{
  int eN_global_new = *eN_p2++;
  for (int ebN=0;ebN<mesh.subdomainp->nElementBoundaries_element;ebN++)
  {
    // printf("rank %d eN %d ebN %d eN_global_new %d ebN_global %d\n", rank, eN, ebN, eN_global_new, elementBoundariesArray_new[eN_global_new*mesh.nElementBoundaries_element + ebN]);
    mesh.subdomainp->elementBoundariesArray[eN*mesh.subdomainp->nElementBoundaries_element+ebN] =
    elementBoundaryNumbering_global2subdomain[elementBoundariesArray_new[eN_global_new*mesh.nElementBoundaries_element + ebN]];
  }
  
}

//printf("Hello 12\n");
//edges
mesh.subdomainp->edgeNodesArray = new int[mesh.subdomainp->nEdges_global*2];
//locally owned
for (int i=0; i < nEdges_subdomain_new[rank]; i++)
{
  const int ig = i+edgeOffsets_new[rank];
  const int nN0_global = edgeNodesArray_newNodesAndEdges[ig*2+0];
  const int nN1_global = edgeNodesArray_newNodesAndEdges[ig*2+1];
  //mwf todo double check can always count on having nodes on this processor
  const int nN0_subdomain = nodeNumbering_global2subdomain[nN0_global];
  const int nN1_subdomain = nodeNumbering_global2subdomain[nN1_global];
  mesh.subdomainp->edgeNodesArray[2*i+0]=nN0_subdomain;
  mesh.subdomainp->edgeNodesArray[2*i+1]=nN1_subdomain;
  edgeNumbering_subdomain2global[i] = ig;
}
//ghost
set<int>::iterator edge_p = edges_overlap.begin();
for (int i=nEdges_subdomain_new[rank]; i < nEdges_subdomain_new[rank] + int(edges_overlap.size()); i++)
{
  const int ig =*edge_p++;
  const int nN0_global = edgeNodesArray_newNodesAndEdges[ig*2+0];
  const int nN1_global = edgeNodesArray_newNodesAndEdges[ig*2+1];
  //mwf todo make sure always have nodes for the edge on this processor
  const int nN0_subdomain = nodeNumbering_global2subdomain[nN0_global];
  const int nN1_subdomain = nodeNumbering_global2subdomain[nN1_global];
  mesh.subdomainp->edgeNodesArray[2*i+0]=nN0_subdomain;
  mesh.subdomainp->edgeNodesArray[2*i+1]=nN1_subdomain;
  edgeNumbering_subdomain2global[i] = ig;
  
}
subdomainEdgeMaterialTypes(mesh, *mesh.subdomainp, edgeNumbering_subdomain2global, edgeNumbering_global_new2old);

//now build rest of subdomain mesh connectivity information etc
mesh.subdomainp->px = mesh.px;
mesh.subdomainp->py = mesh.py;
mesh.subdomainp->pz = mesh.pz;

if (mesh.subdomainp->px != 0)
{
  //constructElementBoundaryElementsArray_tetrahedron(*mesh.subdomainp);
  //constructElementBoundaryElementsArrayWithGivenElementBoundaryNumbers_tetrahedron(*mesh.subdomainp);
  constructElementBoundaryElementsArrayWithGivenElementBoundaryAndEdgeNumbers_NURBS(*mesh.subdomainp);
  allocateGeometricInfo_NURBS(*mesh.subdomainp);
  computeGeometricInfo_NURBS(*mesh.subdomainp);
}
else if (mesh.subdomainp->nNodes_element == 2)
{
  //constructElementBoundaryElementsArray_edge(*mesh.subdomainp);
  //constructElementBoundaryElementsArrayWithGivenElementBoundaryNumbers_edge(*mesh.subdomainp);
  constructElementBoundaryElementsArrayWithGivenElementBoundaryAndEdgeNumbers_edge(*mesh.subdomainp);
  allocateGeometricInfo_edge(*mesh.subdomainp);
  computeGeometricInfo_edge(*mesh.subdomainp);
}
else if (mesh.subdomainp->nNodes_element == 3)
{
  //constructElementBoundaryElementsArray_triangle(*mesh.subdomainp);
  //constructElementBoundaryElementsArrayWithGivenElementBoundaryNumbers_triangle(*mesh.subdomainp);
  constructElementBoundaryElementsArrayWithGivenElementBoundaryAndEdgeNumbers_triangle(*mesh.subdomainp);
  allocateGeometricInfo_triangle(*mesh.subdomainp);
  computeGeometricInfo_triangle(*mesh.subdomainp);
}
else if (mesh.subdomainp->nNodes_element == 4 && mesh.subdomainp->nNodes_elementBoundary == 2)
{
  //constructElementBoundaryElementsArray_tetrahedron(*mesh.subdomainp);
  //constructElementBoundaryElementsArrayWithGivenElementBoundaryNumbers_tetrahedron(*mesh.subdomainp);
  constructElementBoundaryElementsArrayWithGivenElementBoundaryAndEdgeNumbers_quadrilateral(*mesh.subdomainp);
  allocateGeometricInfo_quadrilateral(*mesh.subdomainp);
  computeGeometricInfo_quadrilateral(*mesh.subdomainp);
}
else if (mesh.subdomainp->nNodes_element == 4 && mesh.subdomainp->nNodes_elementBoundary == 3)
{
  //constructElementBoundaryElementsArray_tetrahedron(*mesh.subdomainp);
  //constructElementBoundaryElementsArrayWithGivenElementBoundaryNumbers_tetrahedron(*mesh.subdomainp);
  constructElementBoundaryElementsArrayWithGivenElementBoundaryAndEdgeNumbers_tetrahedron(*mesh.subdomainp);
  allocateGeometricInfo_tetrahedron(*mesh.subdomainp);
  computeGeometricInfo_tetrahedron(*mesh.subdomainp);
}
else if (mesh.subdomainp->nNodes_element == 8)
{
  constructElementBoundaryElementsArrayWithGivenElementBoundaryAndEdgeNumbers_hexahedron(*mesh.subdomainp);
  allocateGeometricInfo_hexahedron(*mesh.subdomainp);
  computeGeometricInfo_hexahedron(*mesh.subdomainp);
}
else
{
  assert(false);
}

if (mesh.elementBoundaryMaterialTypes != NULL)
{
  assert(mesh.elementBoundariesArray != NULL);
  assert(mesh.subdomainp->elementBoundariesArray != NULL);
  for (int eN=0;eN<mesh.subdomainp->nElements_global;eN++)
  {
    int eN_global_new = elementNumbering_subdomain2global[eN];
    int eN_global_old = elementNumbering_global_new2old[eN_global_new];
    for (int ebN_element = 0; ebN_element < mesh.nElementBoundaries_element; ebN_element++)
    {
      int ebN_global_old = mesh.elementBoundariesArray[eN_global_old*mesh.nElementBoundaries_element+ebN_element];
      int ebN_subdomain = mesh.subdomainp->elementBoundariesArray[eN*mesh.nElementBoundaries_element+ebN_element];
      // printf("ebN_global_old = %d \t ebN_subdomain = %d \t elementBoundaryMaterialType = %d mesh.subdomainp->elementBoundaryMaterialTypes = %d\n",
        // ebN_global_old, ebN_subdomain, mesh.elementBoundaryMaterialTypes[ebN_global_old], mesh.subdomainp->elementBoundaryMaterialTypes[ebN_subdomain]);
      // printf("en = %d eN_global_new %d eN_global_old %d ebN_element %d ebN_global_old %d elementBoundariesArray = %d \n",eN,eN_global_new,eN_global_old,ebN_element,ebN_global_old, mesh.elementBoundariesArray[eN_global_old*mesh.nElementBoundaries_element+ebN_element]);
      // printf("rank %d eN %d eN_global_new %d eN_global_old %d ebN_element %d ebN_global_old %d ebN_subdomain %d elementBoundaryMaterialType = %d\n",
      //  rank,eN,eN_global_new,eN_global_old,ebN_element,ebN_global_old,ebN_subdomain, mesh.elementBoundaryMaterialTypes[ebN_global_old]);
      mesh.subdomainp->elementBoundaryMaterialTypes[ebN_subdomain] = mesh.elementBoundaryMaterialTypes[ebN_global_old];
    }
  }
}
//transfer information about owned nodes and elements to mesh
if (mesh.nodeOffsets_subdomain_owned)
delete [] mesh.nodeOffsets_subdomain_owned;
if (mesh.elementOffsets_subdomain_owned)
delete [] mesh.elementOffsets_subdomain_owned;
if (mesh.elementBoundaryOffsets_subdomain_owned)
delete [] mesh.elementBoundaryOffsets_subdomain_owned;
  if (mesh.edgeOffsets_subdomain_owned)
    delete [] mesh.edgeOffsets_subdomain_owned;
  mesh.nodeOffsets_subdomain_owned    = new int[size+1];
  mesh.elementOffsets_subdomain_owned = new int[size+1];
  mesh.elementBoundaryOffsets_subdomain_owned = new int[size+1];
  mesh.edgeOffsets_subdomain_owned = new int[size+1];
  for (int sdN = 0; sdN < size+1; sdN++)
    {
      mesh.nodeOffsets_subdomain_owned[sdN]    = nodeOffsets_new[sdN];
      mesh.elementOffsets_subdomain_owned[sdN] = elementOffsets_new[sdN];
      mesh.elementBoundaryOffsets_subdomain_owned[sdN] = elementBoundaryOffsets_new[sdN];
      mesh.edgeOffsets_subdomain_owned[sdN] = edgeOffsets_new[sdN];
    }
  if (mesh.nodeNumbering_subdomain2global)
    delete [] mesh.nodeNumbering_subdomain2global;
  mesh.nodeNumbering_subdomain2global = new int[mesh.subdomainp->nNodes_global];
  for (int nN = 0; nN < mesh.subdomainp->nNodes_global; nN++)
    mesh.nodeNumbering_subdomain2global[nN] = nodeNumbering_subdomain2global[nN];
  if (mesh.elementNumbering_subdomain2global)
    delete [] mesh.elementNumbering_subdomain2global;
  mesh.elementNumbering_subdomain2global = new int[mesh.subdomainp->nElements_global];
  for (int eN = 0; eN < mesh.subdomainp->nElements_global; eN++)
    mesh.elementNumbering_subdomain2global[eN] = elementNumbering_subdomain2global[eN];
  //
  if (mesh.elementBoundaryNumbering_subdomain2global)
    delete [] mesh.elementBoundaryNumbering_subdomain2global;
  mesh.elementBoundaryNumbering_subdomain2global = new int[mesh.subdomainp->nElementBoundaries_global];
  for (int ebN = 0; ebN < mesh.subdomainp->nElementBoundaries_global; ebN++)
    mesh.elementBoundaryNumbering_subdomain2global[ebN] = elementBoundaryNumbering_subdomain2global[ebN];
  //
  if (mesh.edgeNumbering_subdomain2global)
    delete [] mesh.edgeNumbering_subdomain2global;
  mesh.edgeNumbering_subdomain2global = new int[mesh.subdomainp->nEdges_global];
  for (int i=0; i< mesh.subdomainp->nEdges_global; i++)
    mesh.edgeNumbering_subdomain2global[i] = edgeNumbering_subdomain2global[i];


  if (timer) timer->mark("subdomain_mesh");
  return 0;
}


/* I/O, local: stream a Triangle or TetGen element file (simplexDim nodes per
   element; format names it in the error) and keep what a rank whose default
   chunk is nodes [begin,end) needs to partition them: the star (the other
   nodes of every element containing it) of each of its nodes. Reads every
   element and keeps O(chunk) data. Returns false, having said why, if the file
   cannot be opened. */
bool readSimplexNodeStars(const std::string& elementFileName, const char* format, int simplexDim,
                          int indexBase, int nNodes_global, int begin, int end, int& nElements_global,
                          std::vector<std::set<int> >& nodeStar)
{
  using namespace std;
  using namespace IOutils;
  std::ifstream elementFile(elementFileName.c_str());
  if (!elementFile.good())
    {
      writeErrorLine(std::string("cannot open ")+format+" element file "+elementFileName);
      return false;
    }
  //read elements
  int nNodesPerSimplex(simplexDim),hasElementMarkers = 0;
  elementFile >> eatcomments >> nElements_global >> nNodesPerSimplex >> hasElementMarkers >> eatline;
  assert(nElements_global > 0);
  assert(nNodesPerSimplex == simplexDim);
  vector<int> element_nodes_old(simplexDim);
  nodeStar.assign(end-begin,set<int>());
  for (int ie = 0; ie < nElements_global; ie++)
    {
      int ne, nv;
      elementFile >> eatcomments >> ne;
      ne -= indexBase;
      assert(0 <= ne && ne < nElements_global && elementFile.good());
      for (int iv = 0; iv < simplexDim; iv++)
        {
          elementFile >> nv;
          nv -= indexBase;
          assert(0 <= nv && nv < nNodes_global);
          element_nodes_old[iv] = nv;
        }
      //for each node on the element
      for (int iv = 0; iv < simplexDim; iv++)
        {
          //check if the node is owned by the subdomain
          int nN_star = element_nodes_old[iv];
          if (nN_star >= begin && nN_star < end)
            {
              //this node is owned by the subdomain so
              for (int jv = 0; jv < simplexDim; jv++)
                {
                  if (iv != jv)
                    {
                      int nN_star_subdomain = nN_star-begin;
                      nodeStar[nN_star_subdomain].insert(element_nodes_old[jv]);
                    }
                }
            }
        }
      elementFile >> eatline;
    }//end ie
  elementFile.close();
  return true;
}

/* I/O, local: stream a Triangle (nSpace 2) or TetGen (nSpace 3) node file
   (format names it in the error) and keep the vertices of one subdomain, in
   its numbering: owned nodes are the new global numbers [ownedBegin,ownedEnd),
   at subdomain index n-ownedBegin; overlap nodes are the keys of
   nodes_overlap_global2subdomain, at its values. nodes_old2new gives the new
   global number of each file node the subdomain touches; other file nodes are
   skipped. Fills nodeNumbering_subdomain2global, its inverse map, nodeArray
   (x,y,z, z=0 in 2D) and, when hasVertexMarkers (the caller's reading of the
   header: a marker, or an attribute taken as one), nodeMaterialTypes from the
   value after the coordinates. Reads every vertex and keeps O(subdomain) data.
   Returns false, having said why, if the file cannot be opened. */
bool readSubdomainVertices(const std::string& vertexFileName, const char* format, int nSpace, int indexBase,
                           int hasVertexMarkers, const std::map<int,int>& nodes_old2new, int ownedBegin, int ownedEnd,
                           const std::map<int,int>& nodes_overlap_global2subdomain,
                           int* nodeNumbering_subdomain2global,
                           std::map<int,int>& nodeNumbering_global2subdomainMap,
                           double* nodeArray, int* nodeMaterialTypes)
{
  using namespace IOutils;
  const int vertexDim = 3;
  std::ifstream vertexFile(vertexFileName.c_str());
  if (!vertexFile.good())
    {
      writeErrorLine(std::string("cannot open ")+format+" node file "+vertexFileName);
      return false;
    }
  int nSpace_file(nSpace),nNodes_global,hasVertexAttributes_file,hasVertexMarkers_file;
  vertexFile >> eatcomments >> nNodes_global >> nSpace_file >> hasVertexAttributes_file >> hasVertexMarkers_file >> eatline ;
  assert(nNodes_global > 0);
  assert(nSpace_file == nSpace);
  for (int iv = 0; iv < nNodes_global; iv++)
    {
      int nv; double x[3] = {0.0, 0.0, 0.0}; int nodeId(0);
      vertexFile >> eatcomments >> nv;
      for (int d = 0; d < nSpace; d++)
        vertexFile >> x[d];
      if (hasVertexMarkers > 0)
        vertexFile >> nodeId;
      nv -= indexBase;
      assert(0 <= nv && nv < nNodes_global && vertexFile.good());
      int nN_global_new = -1;
      std::map<int,int>::const_iterator old2new = nodes_old2new.find(nv);
      if (old2new != nodes_old2new.end())
        nN_global_new = old2new->second;
      //local
      if (nN_global_new >= ownedBegin && nN_global_new < ownedEnd)
        {
          int nv_subdomain_new = nN_global_new - ownedBegin;
          nodeNumbering_subdomain2global[nv_subdomain_new] = nN_global_new;
          nodeNumbering_global2subdomainMap[nN_global_new] = nv_subdomain_new;
          for (int d = 0; d < vertexDim; d++)
            nodeArray[vertexDim*nv_subdomain_new + d] = x[d];
          if (hasVertexMarkers > 0)
            nodeMaterialTypes[nv_subdomain_new] = nodeId;
        }
      //overlap
      std::map<int,int>::const_iterator overlap = nodes_overlap_global2subdomain.find(nN_global_new);
      if (overlap != nodes_overlap_global2subdomain.end())
        {
          int nv_subdomain_new = overlap->second;
          nodeNumbering_subdomain2global[nv_subdomain_new] = nN_global_new;
          nodeNumbering_global2subdomainMap[nN_global_new] = nv_subdomain_new;
          for (int d = 0; d < vertexDim; d++)
            nodeArray[vertexDim*nv_subdomain_new + d] = x[d];
          if (hasVertexMarkers > 0)
            nodeMaterialTypes[nv_subdomain_new] = nodeId;
        }
      vertexFile >> eatline;
    }//end iv
  vertexFile.close();
  return true;
}

/* I/O, local: stream a Triangle or TetGen element file (simplexDim nodes per
   element) and keep, in file order, every element with a node in `nodes`
   (file numbering, the keys): its file number, its file nodes (simplexDim per
   element, appended to elementNodes) and, when the header announces
   attributes, its first attribute (elementIds). Reads every element and keeps
   O(kept) data. Returns false, having said why, if the file cannot be opened. */
bool readSimplexElementsTouching(const std::string& elementFileName, const char* format, int simplexDim,
                                 int indexBase, int nNodes_global, const std::map<int,int>& nodes,
                                 int& nElements_global, int& hasElementMarkers,
                                 std::vector<int>& elements, std::vector<int>& elementNodes,
                                 std::vector<double>& elementIds)
{
  using namespace IOutils;
  std::ifstream elementFile(elementFileName.c_str());
  if (!elementFile.good())
    {
      writeErrorLine(std::string("cannot open ")+format+" elements file "+elementFileName);
      return false;
    }
  int nNodesPerSimplex(simplexDim);
  hasElementMarkers = 0;
  elementFile >> eatcomments >> nElements_global >> nNodesPerSimplex >> hasElementMarkers >> eatline;
  assert(nElements_global > 0);
  assert(nNodesPerSimplex == simplexDim);
  elements.clear(); elementNodes.clear(); elementIds.clear();
  std::vector<int> element_nodes_old(simplexDim);
  for (int ie = 0; ie < nElements_global; ie++)
    {
      int ne, nv;
      long double elementId_double(0);
      elementFile >> eatcomments >> ne;
      ne -= indexBase;
      assert(0 <= ne && ne < nElements_global && elementFile.good());
      bool touches = false;
      for (int iv = 0; iv < simplexDim; iv++)
        {
          elementFile >> nv;
          nv -= indexBase;
          assert(0 <= nv && nv < nNodes_global);
          element_nodes_old[iv] = nv;
          if (nodes.find(nv) != nodes.end())
            touches = true;
        }
      //read the attribute unconditionally: the stream position must not
      //depend on whether this element is kept
      if (hasElementMarkers > 0)
        elementFile >> elementId_double;
      elementFile >> eatline;
      if (touches)
        {
          elements.push_back(ne);
          elementNodes.insert(elementNodes.end(), element_nodes_old.begin(), element_nodes_old.end());
          if (hasElementMarkers > 0)
            elementIds.push_back(static_cast<double>(elementId_double));
        }
    }
  elementFile.close();
  return true;
}

/* I/O, local: stream a Triangle or TetGen face (.face) or edge (.edge) file,
   nNodesPerEntity nodes per line, and keep, in file order, the entities with a
   node among the keys of `nodes` (0-based file numbers): their file numbers,
   file nodes and, when headerHasMarkers and the header announces them,
   markers. With headerHasMarkers false only the count is read from the header
   and markers are never read (what both -F partitioners did with .edge files
   until 2026-11; they now read them as the subdomain's edgeMaterialTypes).
   Entities with no such node are skipped: the -F partitioners do nothing with
   them. Returns false, having said why, if the file cannot be opened. */
bool readSimplexSubentitiesTouching(const std::string& fileName, const char* format, const char* what,
                                    int nNodesPerEntity, int indexBase, bool headerHasMarkers,
                                    const std::map<int,int>& nodes, int& nEntities_global, int& hasMarkers,
                                    std::vector<int>& numbers, std::vector<int>& entityNodes, std::vector<int>& ids)
{
  using namespace IOutils;
  std::ifstream file(fileName.c_str());
  if (!file.good())
    {
      writeErrorLine(std::string("cannot open ")+format+" "+what+" file "+fileName);
      return false;
    }
  hasMarkers = 0;
  if (headerHasMarkers)
    file >> eatcomments >> nEntities_global >> hasMarkers >> eatline;
  else
    file >> eatcomments >> nEntities_global >> eatline;
  assert(nEntities_global > 0);
  numbers.clear(); entityNodes.clear(); ids.clear();
  std::vector<int> entity_nodes(nNodesPerEntity);
  for (int i = 0; i < nEntities_global; i++)
    {
      int n, nv, id(0);
      file >> eatcomments >> n;
      bool touches = false;
      for (int iv = 0; iv < nNodesPerEntity; iv++)
        {
          file >> nv;
          nv -= indexBase;
          entity_nodes[iv] = nv;
          if (nodes.find(nv) != nodes.end())
            touches = true;
        }
      if (hasMarkers > 0)
        file >> id;
      n -= indexBase;
      assert(0 <= n && n < nEntities_global && file.good());
      file >> eatline;
      if (touches)
        {
          numbers.push_back(n);
          entityNodes.insert(entityNodes.end(), entity_nodes.begin(), entity_nodes.end());
          if (hasMarkers > 0)
            ids.push_back(id);
        }
    }
  file.close();
  return true;
}

namespace
{
  //the rank whose range [offsets[r],offsets[r+1]) holds n
  inline int rankOfNumber(const std::vector<int>& offsets, int n)
  {
    return int(std::upper_bound(offsets.begin(), offsets.end(), n) - offsets.begin()) - 1;
  }

  //one MPI_Alltoallv of ints whose counts both sides already know: sendBuffers[r]
  //goes to rank r, and recvCounts[r] ints come from rank r; returns what each
  //rank sent here, concatenated in rank order
  std::vector<int> exchangeIntsCounted(const MPI_Comm& comm, const std::vector<std::vector<int> >& sendBuffers,
                                       const std::vector<int>& recvCounts)
  {
    const int size = int(sendBuffers.size());
    std::vector<int> sendCounts(size), sendDispls(size+1, 0), recvDispls(size+1, 0);
    for (int r = 0; r < size; r++)
      sendCounts[r] = int(sendBuffers[r].size());
    for (int r = 0; r < size; r++)
      {
        sendDispls[r+1] = sendDispls[r] + sendCounts[r];
        recvDispls[r+1] = recvDispls[r] + recvCounts[r];
      }
    std::vector<int> send(sendDispls[size]), recv(recvDispls[size]);
    for (int r = 0; r < size; r++)
      std::copy(sendBuffers[r].begin(), sendBuffers[r].end(), send.begin() + sendDispls[r]);
    MPI_Alltoallv(send.data(), sendCounts.data(), sendDispls.data(), MPI_INT,
                  recv.data(), recvCounts.data(), recvDispls.data(), MPI_INT, comm);
    return recv;
  }

  //one round of MPI_Alltoall (the counts) and MPI_Alltoallv of ints:
  //sendBuffers[r] goes to rank r; returns what each rank sent here,
  //concatenated in rank order, and, in recvCounts if given, how many came
  //from each rank
  std::vector<int> exchangeInts(const MPI_Comm& comm, const std::vector<std::vector<int> >& sendBuffers,
                                std::vector<int>* recvCounts = nullptr)
  {
    const int size = int(sendBuffers.size());
    std::vector<int> sendCounts(size), counts(size);
    for (int r = 0; r < size; r++)
      sendCounts[r] = int(sendBuffers[r].size());
    MPI_Alltoall(sendCounts.data(), 1, MPI_INT, counts.data(), 1, MPI_INT, comm);
    std::vector<int> recv = exchangeIntsCounted(comm, sendBuffers, counts);
    if (recvCounts)
      *recvCounts = std::move(counts);
    return recv;
  }
}

/* MPI: the nodes this rank owns after repartitioning, file number -> new
   number. Every rank holds old2new for its default chunk of file nodes
   [nodeOffsets_old[rank], nodeOffsets_old[rank+1]); each pair goes to the rank
   whose new range (nodeOffsets_new) holds it. One exchange, no file. */
std::map<int,int> ownedNodesByFileNumber(const MPI_Comm& comm, const std::vector<int>& nodeOffsets_old,
                                         const std::vector<int>& nodeOffsets_new, const std::vector<int>& old2new_chunk)
{
  int rank, size;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);
  std::vector<std::vector<int> > pairs(size);
  for (int nN_old = nodeOffsets_old[rank]; nN_old < nodeOffsets_old[rank+1]; nN_old++)
    {
      const int nN_new = old2new_chunk[nN_old - nodeOffsets_old[rank]];
      std::vector<int>& to = pairs[rankOfNumber(nodeOffsets_new, nN_new)];
      to.push_back(nN_old);
      to.push_back(nN_new);
    }
  std::vector<int> received = exchangeInts(comm, pairs);
  std::map<int,int> owned;
  for (size_t i = 0; i < received.size(); i += 2)
    owned[received[i]] = received[i+1];
  assert(int(owned.size()) == nodeOffsets_new[rank+1] - nodeOffsets_new[rank]);
  return owned;
}

/* MPI: the new numbers of the file nodes `wanted`, asked of the ranks whose
   default chunk (nodeOffsets_old) holds them. Two exchanges (requests,
   answers), no file. Every rank calls it, wanted or not. */
namespace
{
  /* MPI: ask, for every key, the rank whose range of offsets holds it, and
     return key -> that rank's answer(key). Two exchanges; collective.
     answer(key) is a number >= 0, or -1 when the asked rank holds none for
     key (no rank owns it: an inconsistent set of mesh files, or a partition
     that left an entity unowned). Neither that nor a key outside
     [offsets[0], offsets[size]) has a meaning for any caller, and the
     asserts are compiled out, so either one ends the run on every rank
     (MPI_Abort), naming the `what`, the number and the rank that asked. */
  template<class Answer>
  std::map<int,int> askOwningRanks(const MPI_Comm& comm, const std::vector<int>& offsets,
                                   const std::set<int>& keys, Answer answer, const char* what)
  {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    if (!keys.empty() && (*keys.begin() < offsets[0] || *keys.rbegin() >= offsets[size]))
      {
        const int key = *keys.begin() < offsets[0] ? *keys.begin() : *keys.rbegin();
        std::ostringstream message;
        message<<"partitioning: "<<what<<" "<<key<<" (asked by rank "<<rank<<") is outside ["
               <<offsets[0]<<","<<offsets[size]<<"), the numbers the ranks hold"
               <<" (are the mesh files consistent?)";
        abortWith(comm, message.str());
      }
    std::vector<std::vector<int> > requests(size);
    for (std::set<int>::const_iterator k = keys.begin(); k != keys.end(); k++)
      requests[rankOfNumber(offsets, *k)].push_back(*k);
    //which ranks asked what (the request exchange's own counts), so the
    //answers go back in the same order; each rank answers as many keys as it
    //was asked, so the answer exchange needs no counts of its own
    std::vector<int> requestCounts(size), askedCounts;
    for (int r = 0; r < size; r++)
      requestCounts[r] = int(requests[r].size());
    std::vector<int> asked = exchangeInts(comm, requests, &askedCounts);
    std::vector<std::vector<int> > answers(size);
    for (int r = 0, i = 0; r < size; r++)
      for (int k = 0; k < askedCounts[r]; k++, i++)
        answers[r].push_back(answer(asked[i]));
    std::vector<int> answered = exchangeIntsCounted(comm, answers, requestCounts);
    std::map<int,int> result;
    int unanswered = 0;
    std::ostringstream message;
    for (int r = 0, i = 0; r < size; r++)
      for (int k = 0; k < requestCounts[r]; k++, i++)
        {
          if (answered[i] < 0 && unanswered++ == 0)
            message<<"partitioning: "<<what<<" "<<requests[r][k]<<" (asked by rank "<<rank<<", of rank "<<r
                   <<") has no owner: no rank holds a number for it (are the mesh files consistent?)";
          result[requests[r][k]] = answered[i];
        }
    if (unanswered)
      {
        message<<"; rank "<<rank<<" asked for "<<unanswered<<" "<<what<<"(s) with no owner";
        abortWith(comm, message.str());
      }
    return result;
  }
}

std::map<int,int> newNumbersOfFileNodes(const MPI_Comm& comm, const std::vector<int>& nodeOffsets_old,
                                        const std::vector<int>& old2new_chunk, const std::set<int>& wanted)
{
  int rank;
  MPI_Comm_rank(comm, &rank);
  const int begin = nodeOffsets_old[rank];
  return askOwningRanks(comm, nodeOffsets_old, wanted,
                        [&](int nN_old) { return old2new_chunk[nN_old - begin]; }, "node");
}

/* MPI: file number -> new number for the entities `subdomain` (elements,
   faces or edges of this rank's subdomain, file numbers, ascending), given the
   ones this rank owns. The owned entities, in ascending file order, take the
   new numbers ownedBegin, ownedBegin+1, ... (ownedBegin = the count owned by
   lower ranks). The others' new numbers come from a directory: every rank
   pushes its (old, new) pairs to the rank whose default chunk of [0,nGlobal)
   holds the old number (one exchange), then asks that rank for the entities
   it needs (askOwningRanks, two exchanges). Memory per rank is O(subdomain +
   nGlobal/size); no file, no global array. Collective. `owned` must be a
   subset of `subdomain`; the result holds exactly the entries of `subdomain`,
   as (old, new) pairs sorted by old. */
std::vector<std::pair<int,int> > subdomainNumbersByDirectory(const MPI_Comm& comm, int nGlobal, int ownedBegin,
                                                             const std::set<int>& owned,
                                                             const std::vector<int>& subdomain, const char* what)
{
  int rank, size;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);
  const std::valarray<int> chunks = defaultChunkOffsets(nGlobal, size);
  const std::vector<int> offsets(std::begin(chunks), std::end(chunks));
  std::vector<std::pair<int,int> > mine;
  mine.reserve(owned.size());
  std::vector<std::vector<int> > pairs(size);
  int nNew = ownedBegin;
  for (std::set<int>::const_iterator it = owned.begin(); it != owned.end(); it++, nNew++)
    {
      mine.push_back(std::make_pair(*it, nNew));
      std::vector<int>& to = pairs[rankOfNumber(offsets, *it)];
      to.push_back(*it);
      to.push_back(nNew);
    }
  std::vector<int> received = exchangeInts(comm, pairs);
  pairs.clear();
  const int begin = offsets[rank];
  std::vector<int> directory(offsets[rank+1] - begin, -1);
  for (size_t i = 0; i < received.size(); i += 2)
    directory[received[i] - begin] = received[i+1];
  received.clear();
  std::set<int> wanted;
  for (std::vector<int>::const_iterator it = subdomain.begin(); it != subdomain.end(); it++)
    if (owned.find(*it) == owned.end())
      wanted.insert(*it);
  std::map<int,int> ghosts = askOwningRanks(comm, offsets, wanted,
                                            [&](int n_old) { return directory[n_old - begin]; }, what);
  std::vector<std::pair<int,int> > old2new;
  old2new.reserve(mine.size() + ghosts.size());
  //by old number only: ghosts' elements are pair<const int,int>, mine's
  //pair<int,int>, and libstdc++ before 15 has no operator< across the two
  //(the keys are distinct, so this is the order pair's operator< gave)
  std::merge(mine.begin(), mine.end(), ghosts.begin(), ghosts.end(), std::back_inserter(old2new),
             [](const auto& a, const auto& b) { return a.first < b.first; });
  assert(old2new.size() == subdomain.size());
  return old2new;
}

namespace
{
  /* One numbering of a subdomain's entities (old -> new, or new -> old), as a
     table sorted by key: the lookups the -F partitioners make of it, in 8
     bytes an entry rather than the ~48 of a std::map node (at 4 ranks the
     std::maps took more memory than the global arrays they replaced). */
  class SubdomainNumbering
  {
  public:
    typedef std::vector<std::pair<int,int> >::const_iterator const_iterator;
    explicit SubdomainNumbering(std::vector<std::pair<int,int> > sorted) : table(std::move(sorted)) {}
    const_iterator find(int key) const
    {
      const_iterator it = std::lower_bound(table.begin(), table.end(), std::make_pair(key, std::numeric_limits<int>::min()));
      return (it != table.end() && it->first == key) ? it : table.end();
    }
    const_iterator end() const { return table.end(); }
    size_t size() const { return table.size(); }
    //the number of key, which must be in the table: a missing key is an
    //inconsistent set of mesh files, not a default (a std::map gave 0)
    int operator[](int key) const
    {
      const_iterator it = find(key);
      if (it == table.end())
        throw std::out_of_range("parun -F: entity " + std::to_string(key) +
                                " is not in this subdomain's numbering (are the mesh files consistent?)");
      return it->second;
    }
    SubdomainNumbering inverse() const
    {
      std::vector<std::pair<int,int> > inverted;
      inverted.reserve(table.size());
      for (const_iterator it = table.begin(); it != table.end(); it++)
        inverted.push_back(std::make_pair(it->second, it->first));
      std::sort(inverted.begin(), inverted.end());
      return SubdomainNumbering(std::move(inverted));
    }
  private:
    std::vector<std::pair<int,int> > table;
  };

  //the keys of a map, ascending
  template<class Map>
  std::vector<int> keysOf(const Map& m)
  {
    std::vector<int> keys;
    keys.reserve(m.size());
    for (typename Map::const_iterator it = m.begin(); it != m.end(); it++)
      keys.push_back(it->first);
    return keys;
  }
}

/* MPI: give the faces the partition cut the reserved flag -1, for a
   partitioner that has no global mesh to compare with (the parun -F path; the
   in-memory partitioners are marked by MeshTools._markRankInterfaces). A
   subdomain face is cut when it is exterior to the subdomain and interior to
   the mesh. The rank that owns a face owns one of its nodes, so it holds every
   element of that node, both neighbours included: its subdomain knows whether
   the face is exterior to the mesh. Every other rank asks it, for the faces
   exterior to its own subdomain. Like _markRankInterfaces, only flag 0 is
   replaced: a positive flag on a cut face is a condition the mesh put there.
   Needs the subdomain mesh, elementBoundaryOffsets_subdomain_owned and
   elementBoundaryNumbering_subdomain2global. Collective. Returns the number
   of faces marked on this rank. flagsFromFile false says the face flags did
   not come from the mesh files but from the subdomain builder (exterior 1,
   interior 0), so a cut face reads 1 there; then every exterior face is
   asked about, and a cut face reads -1 as on the in-memory path (where such a
   mesh gives an interior face 0). */
int markRankInterfacesByOwner(const MPI_Comm& comm, Mesh& mesh, bool flagsFromFile)
{
  int rank, size;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);
  Mesh& sub = *mesh.subdomainp;
  const std::vector<int> offsets(mesh.elementBoundaryOffsets_subdomain_owned,
                                 mesh.elementBoundaryOffsets_subdomain_owned + size + 1);
  std::map<int,int> global2subdomain;
  std::set<int> askAbout;
  for (int ebN = 0; ebN < sub.nElementBoundaries_global; ebN++)
    {
      const int ebN_global = mesh.elementBoundaryNumbering_subdomain2global[ebN];
      global2subdomain[ebN_global] = ebN;
      const bool exterior = sub.elementBoundaryElementsArray[ebN*2+1] == -1;
      const bool owned = ebN_global >= offsets[rank] && ebN_global < offsets[rank+1];
      if (exterior && !owned && (!flagsFromFile || sub.elementBoundaryMaterialTypes[ebN] == 0))
        askAbout.insert(ebN_global);
    }
  //the owner of a face holds it in its subdomain; -1 (not there) ends the run in askOwningRanks
  std::map<int,int> meshExterior = askOwningRanks(comm, offsets, askAbout, [&](int ebN_global) {
      std::map<int,int>::const_iterator it = global2subdomain.find(ebN_global);
      if (it == global2subdomain.end())
        return -1;
      return int(sub.elementBoundaryElementsArray[it->second*2+1] == -1);
    }, "face");
  int marked = 0;
  for (std::map<int,int>::const_iterator it = meshExterior.begin(); it != meshExterior.end(); it++)
    if (!it->second)
      {
        sub.elementBoundaryMaterialTypes[global2subdomain[it->first]] = -1;
        marked++;
      }
  return marked;
}

/* Local: what the -F partitioners take from the elements of one subdomain
   (those with a node whose new number is in [ownedBegin,ownedEnd)), given
   in file order with their file nodes and the new number of every one of
   those nodes (old2new). Fills, as partitionNodesFromTriangleFiles and
   partitionNodesFromTetgenFiles always have: the owned elements (node 1 of
   the sorted new node tuple is owned), each owned node's element star and
   node star, every subdomain element's new nodes and attribute, the
   faces/edges of the subdomain elements with their elements, and old->new for
   every node of a subdomain element. Element numbers stay file numbers. */
template<int simplexDim, class NodeVec>
void collectSubdomainElements(int ownedBegin, int ownedEnd, const std::vector<int>& elements,
                              const std::vector<int>& elementNodes, const std::vector<double>& elementIds,
                              const std::map<int,int>& old2new,
                              std::set<int>& elements_subdomain_owned,
                              std::vector<std::set<int> >& nodeElementsStar,
                              std::vector<std::set<int> >& nodeStarNew,
                              std::map<int,NodeVec>& elementNodesArrayMap,
                              std::map<int,long int>& elementMaterialTypesMap,
                              std::map<NodeTuple<simplexDim-1>,ElementNeighbors>& elementBoundaryElementsMap,
                              std::map<NodeTuple<2>,std::set<std::pair<int,int> > >& edgeElementsMap,
                              std::map<int,int>& nodes_old2new_subdomain_map)
{
  NodeVec element_nodes_new(simplexDim);
  int element_nodes_new_array[simplexDim];
  for (size_t eN_c = 0; eN_c < elements.size(); eN_c++)
    {
      const int ne = elements[eN_c];
      const int* element_nodes_old = &elementNodes[eN_c*simplexDim];
      for (int iv = 0; iv < simplexDim; iv++)
        {
          std::map<int,int>::const_iterator it = old2new.find(element_nodes_old[iv]);
          assert(it != old2new.end());
          element_nodes_new[iv] = it->second;
          element_nodes_new_array[iv] = it->second;
        }
      NodeTuple<simplexDim> nodeTuple(element_nodes_new_array);
      bool inSubdomain = false;
      for (int iv = 0; iv < simplexDim; iv++)
        {
          int nN_star_new = element_nodes_new[iv];
          if (nN_star_new >= ownedBegin && nN_star_new < ownedEnd)
            {
              inSubdomain = true;
              //add all the element boundaries of this element
              for (int ebN = 0; ebN < simplexDim; ebN++)
                {
                  int nodes[simplexDim-1];
                  for (int k = 0; k < simplexDim-1; k++)
                    nodes[k] = element_nodes_new[(ebN+1+k) % simplexDim];
                  NodeTuple<simplexDim-1> ebNodeTuple(nodes);
                  typename std::map<NodeTuple<simplexDim-1>,ElementNeighbors>::iterator eb = elementBoundaryElementsMap.find(ebNodeTuple);
                  if (eb != elementBoundaryElementsMap.end())
                    {
                      if (eb->second.right == -1 && ne != eb->second.left)
                        {
                          eb->second.right = ne;
                          eb->second.right_ebN_element = ebN;
                        }
                    }
                  else
                    elementBoundaryElementsMap[ebNodeTuple] = ElementNeighbors(ne,ebN);
                }
              //add all the edges of this element
              for (int nNL = 0, edN = 0; nNL < simplexDim; nNL++)
                for (int nNR = nNL+1; nNR < simplexDim; nNR++, edN++)
                  {
                    int nodes[2] = {element_nodes_new[nNL], element_nodes_new[nNR]};
                    NodeTuple<2> edNodeTuple(nodes);
                    edgeElementsMap[edNodeTuple].insert(std::pair<int,int>(ne,edN));
                  }
              //add all the nodes to the node star
              int nN_star_new_subdomain = nN_star_new - ownedBegin;
              nodeElementsStar[nN_star_new_subdomain].insert(ne);
              for (int jv = 0; jv < simplexDim; jv++)
                if (iv != jv)
                  nodeStarNew[nN_star_new_subdomain].insert(element_nodes_new[jv]);
            }
        }
      if (!inSubdomain)
        continue;
      elementNodesArrayMap[ne] = element_nodes_new;
      //the later .face/.edge/.node readers need old->new for exactly these nodes
      for (int jv = 0; jv < simplexDim; jv++)
        nodes_old2new_subdomain_map[element_nodes_old[jv]] = element_nodes_new[jv];
      if (nodeTuple.nodes[1] >= ownedBegin && nodeTuple.nodes[1] < ownedEnd)
        elements_subdomain_owned.insert(ne);
      if (!elementIds.empty())
        elementMaterialTypesMap[ne] = static_cast<long int>(elementIds[eN_c]);
    }
}

/* The -F partitioners' second read of the element file: the elements of this
   rank's new subdomain, read locally, with the new numbers of their nodes
   exchanged with the ranks that computed them (no mappings-file read). */
template<int simplexDim, class NodeVec>
bool readSubdomainElements(const MPI_Comm& comm, const std::string& elementFileName, const char* format,
                           int indexBase, int nNodes_global,
                           const std::valarray<int>& nodeOffsets_old, const std::valarray<int>& nodeOffsets_new,
                           const PetscInt* old2new_chunk, int& nElements_global,
                           std::set<int>& elements_subdomain_owned,
                           std::vector<std::set<int> >& nodeElementsStar,
                           std::vector<std::set<int> >& nodeStarNew,
                           std::map<int,NodeVec>& elementNodesArrayMap,
                           std::map<int,long int>& elementMaterialTypesMap,
                           std::map<NodeTuple<simplexDim-1>,ElementNeighbors>& elementBoundaryElementsMap,
                           std::map<NodeTuple<2>,std::set<std::pair<int,int> > >& edgeElementsMap,
                           std::map<int,int>& nodes_old2new_subdomain_map)
{
  int rank;
  MPI_Comm_rank(comm, &rank);
  const std::vector<int> offsets_old(std::begin(nodeOffsets_old), std::end(nodeOffsets_old)),
    offsets_new(std::begin(nodeOffsets_new), std::end(nodeOffsets_new)),
    chunk(old2new_chunk, old2new_chunk + (offsets_old[rank+1] - offsets_old[rank]));
  std::map<int,int> old2new = ownedNodesByFileNumber(comm, offsets_old, offsets_new, chunk);
  std::vector<int> elements, elementNodes;
  std::vector<double> elementIds;
  int hasElementMarkers(0);
  bool read = readSimplexElementsTouching(elementFileName, format, simplexDim, indexBase, nNodes_global, old2new,
                                          nElements_global, hasElementMarkers, elements, elementNodes, elementIds);
  //every rank takes part in the exchange, read or not, so none is left waiting
  std::set<int> wanted;
  for (size_t i = 0; i < elementNodes.size(); i++)
    if (old2new.find(elementNodes[i]) == old2new.end())
      wanted.insert(elementNodes[i]);
  std::map<int,int> halo = newNumbersOfFileNodes(comm, offsets_old, chunk, wanted);
  if (!read)
    return false;
  old2new.insert(halo.begin(), halo.end());
  collectSubdomainElements<simplexDim,NodeVec>(nodeOffsets_new[rank], nodeOffsets_new[rank+1],
                                               elements, elementNodes, elementIds, old2new,
                                               elements_subdomain_owned, nodeElementsStar, nodeStarNew,
                                               elementNodesArrayMap, elementMaterialTypesMap,
                                               elementBoundaryElementsMap, edgeElementsMap,
                                               nodes_old2new_subdomain_map);
  return true;
}

/* Steps 4b-8 of the parun -F partitioners, after the second element read
   (readSubdomainElements): read the faces and edges of the subdomain
   elements (I/O, local), number elements, faces and edges by owner
   (subdomainNumbersByDirectory, MPI), collect the ghosts of the minimal
   overlap, read the subdomain's vertices (I/O, local) and build
   newMesh.subdomainp with its subdomain-to-global maps, then flag the cut
   faces -1 (markRankInterfacesByOwner). One body for triangles (simplexDim 3,
   Triangle .edge files as faces) and tetrahedra (simplexDim 4, TetGen .face);
   format names the file kind in messages. The element maps given are
   consumed. max_rss_gb <= 0 skips the memory checks. Returns true on a
   failed read (every rank reads; a missing file fails on every rank).
   Collective. */
template<int simplexDim>
bool buildSubdomainFromFiles(const MPI_Comm& PROTEUS_COMM_WORLD, int rank, int size, const char* format,
                             const std::string& elementBoundaryFileName, const std::string& edgeFileName,
                             const std::string& vertexFileName, int indexBase, int hasVertexMarkers,
                             int nNodes_overlap, double max_rss_gb, int nElements_global,
                             const std::valarray<int>& nNodes_subdomain_new, const std::valarray<int>& nodeOffsets_new,
                             std::set<int>& elements_subdomain_owned,
                             std::vector<std::set<int> >& nodeElementsStar,
                             std::vector<std::set<int> >& nodeStarNew,
                             std::map<int,std::vector<int> >& elementNodesArrayMap,
                             std::map<int,long int>& elementMaterialTypesMap,
                             std::map<NodeTuple<simplexDim-1>,ElementNeighbors>& elementBoundaryElementsMap,
                             std::map<NodeTuple<2>,std::set<std::pair<int,int> > >& edgeElementsMap,
                             std::map<int,int>& nodes_old2new_subdomain_map,
                             Mesh& newMesh)
{
  using namespace std;
  PetscErrorCode ierr;
  bool failed = false;
  const int nNodes_face = simplexDim-1, nEdges_element = simplexDim*(simplexDim-1)/2;
  char log_buffer[100];
  int build_subdomains_send_marked_elements_event;
  PetscLogEventRegister("Mark/send eles",0,&build_subdomains_send_marked_elements_event);
  PetscLogEventBegin(build_subdomains_send_marked_elements_event,0,0,0,0);
  ierr = enforceMemoryLimit(PROTEUS_COMM_WORLD, rank, max_rss_gb,"Done marking elements");CHKERRABORT(PROTEUS_COMM_WORLD, ierr);
  //cek todo: fix PETsc and memory log messages
  //
  //done with the element file
  //
  //construct compact nodeElementsArray
  valarray<int> nodeElementOffsets(nNodes_subdomain_new[rank]+1);
  nodeElementOffsets[0] = 0;
  for (int nN = 0; nN < nNodes_subdomain_new[rank]; nN++)
    nodeElementOffsets[nN+1] = nodeElementOffsets[nN]+nodeElementsStar[nN].size();
  valarray<int> nodeElementsArray(nodeElementOffsets[nNodes_subdomain_new[rank]]);
  for (int nN=0,offset=0; nN < nNodes_subdomain_new[rank]; nN++)
    {
      for (set<int>::iterator eN_star = nodeElementsStar[nN].begin(); eN_star != nodeElementsStar[nN].end();
           eN_star++,offset++)
        {
          nodeElementsArray[offset] = *eN_star;
        }
    }
  nodeElementsStar.clear();
  //construct compact nodeStarArray
  valarray<int> nodeStarOffsetsNew(nNodes_subdomain_new[rank]+1);
  nodeStarOffsetsNew[0] = 0;
  for (int nN=1;nN<nNodes_subdomain_new[rank]+1;nN++)
    nodeStarOffsetsNew[nN] = nodeStarOffsetsNew[nN-1] + nodeStarNew[nN-1].size();
  valarray<int> nodeStarArrayNew(nodeStarOffsetsNew[nNodes_subdomain_new[rank]]);
  for (int nN=0,offset=0;nN<nNodes_subdomain_new[rank];nN++)
    {
      for (set<int>::iterator nN_star=nodeStarNew[nN].begin();nN_star!=nodeStarNew[nN].end();nN_star++,offset++)
        {
          nodeStarArrayNew[offset] = *nN_star;
        }
    }
  nodeStarNew.clear();
  PetscLogEventEnd(build_subdomains_send_marked_elements_event,0,0,0,0);
  int build_subdomains_global_numbering_elements_event;
  PetscLogEventRegister("Global ele nmbr",0,&build_subdomains_global_numbering_elements_event);
  PetscLogEventBegin(build_subdomains_global_numbering_elements_event,0,0,0,0);
  //
  //5. Generate global element numbering corresponding to new subdomain ownership
  //
  logEvent("Generating global element numbering corresponding to new subdomain ownership",5);
  valarray<int> nElements_subdomain_new(size),
    elementOffsets_new(size+1);
  for (int sdN = 0; sdN < size; sdN++)
    {
      if (sdN == rank)
        nElements_subdomain_new[sdN] = int(elements_subdomain_owned.size());
      else
        nElements_subdomain_new[sdN] = 0;
    }
  valarray<int> nElements_subdomain_new_send = nElements_subdomain_new;
  MPI_Allreduce(&nElements_subdomain_new_send[0],&nElements_subdomain_new[0],size,MPI_INT,MPI_SUM,PROTEUS_COMM_WORLD);
  sprintf(log_buffer,"Final partitioning: Max nElements_subdomain %d nElements_global %d",nElements_subdomain_new.max(),nElements_global);
  logEvent(log_buffer,5);

  //construct new offsets for elements
  elementOffsets_new[0] = 0;
  for (int sdN = 0; sdN < size; sdN++)
    elementOffsets_new[sdN+1] = elementOffsets_new[sdN] + nElements_subdomain_new[sdN];
  //
  //the new numbers of every element of the subdomain, owned or not, from a
  //directory on the default chunks of the file numbers (no mappings-file read)
  //
  const SubdomainNumbering elementNumbering_old2new_subdomain_map(
    subdomainNumbersByDirectory(PROTEUS_COMM_WORLD, nElements_global, elementOffsets_new[rank],
                                elements_subdomain_owned, keysOf(elementNodesArrayMap), "element"));
  const SubdomainNumbering elementNumbering_new2old_subdomain_map = elementNumbering_old2new_subdomain_map.inverse();
  PetscLogEventEnd(build_subdomains_global_numbering_elements_event,0,0,0,0);
  ierr = enforceMemoryLimit(PROTEUS_COMM_WORLD, rank, max_rss_gb,"Done allocating element numbering new2old/old2new");CHKERRABORT(PROTEUS_COMM_WORLD, ierr);
  int build_subdomains_faces_event;
  PetscLogEventRegister("Subd faces",0,&build_subdomains_faces_event);
  PetscLogEventBegin(build_subdomains_faces_event,0,0,0,0);
  //
  //4b,5b. repeat process to build global face (elementBoundary) numbering
  //
  //first read element boundaries to create nodeElementBoundariesArray
  //for all element boundaries on this subdomain, which we'll use to
  //grab element boundaries from the bit array
  logEvent("Generating global face numbering and ownership corresponding to new subdomain ownership",5);
  //the faces with a node of a subdomain element; the others play no part below
  bool hasElementBoundaryMarkers = false;
  int nElementBoundaries_global;
  int ihasElementBoundaryMarkers(0);
  vector<int> ebNumbers, ebNodes, ebIds;
  if (!readSimplexSubentitiesTouching(elementBoundaryFileName, format, simplexDim == 4 ? "face" : "edge", nNodes_face, indexBase, true,
                                      nodes_old2new_subdomain_map, nElementBoundaries_global,
                                      ihasElementBoundaryMarkers, ebNumbers, ebNodes, ebIds))
    {
      failed = true;
      return failed;
    }
  assert(nElementBoundaries_global > 0);
  if (ihasElementBoundaryMarkers > 0)
    {
      hasElementBoundaryMarkers = true;
    }
  newMesh.nElementBoundaries_global = nElementBoundaries_global;
  //note, these will be in the new element numbering
  set<int> elementBoundaries_subdomain_owned;
  vector<set<int>> nodeElementBoundariesStar(nNodes_subdomain_new[rank]);
  map<int,int> elementBoundaryMaterialTypesMap;
  map<int,vector<int> > elementBoundariesMap;
  set<int> supportedElementBoundaries;
  set<int> elementBoundaries_subdomain;
  for (size_t k = 0; k < ebNumbers.size(); k++)
    {
      const int neb = ebNumbers[k];
      const int ebId = ebIds.empty() ? 0 : ebIds[k];
      //grab the element boundaries for the node if the node is owned by the subdomain
      //this will miss the element boundaries on the "outside boundary" of the star, which we'll grab later
      //a node of no subdomain element keeps a distinct negative number, so the
      //face's tuple matches no subdomain face and the check on nodes[1] fails
      int nodes[nNodes_face];
      for (int i = 0; i < nNodes_face; i++)
        {
          const int nn = ebNodes[nNodes_face*k+i];
          nodes[i] = -1-i;
          map<int,int>::const_iterator it = nodes_old2new_subdomain_map.find(nn);
          if (it != nodes_old2new_subdomain_map.end())
            nodes[i] = it->second;
          if (nodes[i] >= nodeOffsets_new[rank] && nodes[i] < nodeOffsets_new[rank+1])
            {
              nodeElementBoundariesStar[nodes[i]-nodeOffsets_new[rank]].insert(neb);
              supportedElementBoundaries.insert(neb);
            }
        }
      NodeTuple<nNodes_face> nodeTuple(nodes);
      if (elementBoundaryElementsMap.find(nodeTuple) != elementBoundaryElementsMap.end())//this element boundary is on an element in the subdomain
        {
          elementBoundaries_subdomain.insert(neb);
          if (nodeTuple.nodes[1] >= nodeOffsets_new[rank] && nodeTuple.nodes[1] < nodeOffsets_new[rank+1])
            elementBoundaries_subdomain_owned.insert(neb);
          if (ihasElementBoundaryMarkers > 0)
            elementBoundaryMaterialTypesMap[neb]=ebId;
          int eN_left = elementNumbering_old2new_subdomain_map[elementBoundaryElementsMap[nodeTuple].left];
          if (elementBoundariesMap.find(eN_left) != elementBoundariesMap.end())
            {
              elementBoundariesMap[eN_left][elementBoundaryElementsMap[nodeTuple].left_ebN_element] = neb;
            }
          else
            {
              //initialize
              vector<int> elementBoundaries_element(simplexDim,-1);
              elementBoundariesMap[eN_left] = elementBoundaries_element;
              //assign
              elementBoundariesMap[eN_left][elementBoundaryElementsMap[nodeTuple].left_ebN_element] = neb;
            }
          if (elementBoundaryElementsMap[nodeTuple].right >= 0)
            {
              int eN_right = elementNumbering_old2new_subdomain_map[elementBoundaryElementsMap[nodeTuple].right];
              if (elementBoundariesMap.find(eN_right) != elementBoundariesMap.end())
                {
                  elementBoundariesMap[eN_right][elementBoundaryElementsMap[nodeTuple].right_ebN_element] = neb;
                }
              else
                {
                  //initialize
                  vector<int> elementBoundaries_element(simplexDim,-1);
                  elementBoundariesMap[eN_right] = elementBoundaries_element;
                  //assign
                  elementBoundariesMap[eN_right][elementBoundaryElementsMap[nodeTuple].right_ebN_element] = neb;
                }
            }
        }
    }
  elementBoundaryElementsMap.clear();
  //done reading element boundaries
  ierr = enforceMemoryLimit(PROTEUS_COMM_WORLD, rank, max_rss_gb,"Done reading element boundaries");CHKERRABORT(PROTEUS_COMM_WORLD, ierr);
  int nElementBoundaries_owned_subdomain=elementBoundaries_subdomain_owned.size(),
    nElementBoundaries_owned_new=0;
  MPI_Allreduce(&nElementBoundaries_owned_subdomain,&nElementBoundaries_owned_new,1,MPI_INT,MPI_SUM,PROTEUS_COMM_WORLD);
  assert(nElementBoundaries_owned_new == nElementBoundaries_global);
  //now get the element boundaries on the outside of the star
  for (auto elementBoundariesp=elementBoundariesMap.begin(); 
       elementBoundariesp!=elementBoundariesMap.end();
       elementBoundariesp++)
    {
      //loop over the nodes of this element for the owned nodes
      for (int iv=0;iv<simplexDim;iv++)
        {
          //the elementNodesArrayMap is in the old element numbering while the elementBoundariesMap is in the new element numbering       
          int nN_global = elementNodesArrayMap[elementNumbering_new2old_subdomain_map[elementBoundariesp->first]][iv];
          if (nN_global >= nodeOffsets_new[rank] && nN_global < nodeOffsets_new[rank+1])
            {
              //add all the faces to this node star
              for(int eb=0;eb<simplexDim;eb++)
                {
                  nodeElementBoundariesStar[nN_global-nodeOffsets_new[rank]].insert(elementBoundariesp->second[eb]);
                }
            }
        }
    }
  //build compact structures for nodeElementBoundariesArray
  valarray<int> nodeElementBoundaryOffsets(nNodes_subdomain_new[rank]+1);
  nodeElementBoundaryOffsets[0] = 0;
  for (int nN = 0; nN < nNodes_subdomain_new[rank]; nN++)
    nodeElementBoundaryOffsets[nN+1] = nodeElementBoundaryOffsets[nN]+nodeElementBoundariesStar[nN].size();
  valarray<int> nodeElementBoundariesArray(nodeElementBoundaryOffsets[nNodes_subdomain_new[rank]]);
  for (int nN=0,offset=0; nN < nNodes_subdomain_new[rank]; nN++)
    {
      for (set<int>::iterator ebN_star = nodeElementBoundariesStar[nN].begin(); ebN_star != nodeElementBoundariesStar[nN].end();
           ebN_star++,offset++)
        {
          nodeElementBoundariesArray[offset] = *ebN_star;
        }
    }

  //get the number of elementBoundaries owned on each processor
  valarray<int> nElementBoundaries_subdomain_new(size),
    elementBoundaryOffsets_new(size+1);
  for (int sdN=0;sdN<size;sdN++)
    if (sdN == rank)
      nElementBoundaries_subdomain_new[sdN] = elementBoundaries_subdomain_owned.size();
    else
      nElementBoundaries_subdomain_new[sdN] = 0;
  valarray<int> nElementBoundaries_subdomain_new_send=nElementBoundaries_subdomain_new;
  MPI_Allreduce(&nElementBoundaries_subdomain_new_send[0],&nElementBoundaries_subdomain_new[0],
                size,MPI_INT,MPI_SUM,PROTEUS_COMM_WORLD);
  elementBoundaryOffsets_new[0] = 0;
  for (int sdN=0;sdN<size;sdN++)
    elementBoundaryOffsets_new[sdN+1] = elementBoundaryOffsets_new[sdN]+nElementBoundaries_subdomain_new[sdN];
  //the new numbers of every face of the subdomain, owned or not, from the
  //directory (no mappings-file read)
  const SubdomainNumbering elementBoundaryNumbering_old2new_subdomain_map(
    subdomainNumbersByDirectory(PROTEUS_COMM_WORLD, nElementBoundaries_global, elementBoundaryOffsets_new[rank],
                                elementBoundaries_subdomain_owned,
                                vector<int>(elementBoundaries_subdomain.begin(), elementBoundaries_subdomain.end()),
                                "face"));
  PetscLogEventEnd(build_subdomains_faces_event,0,0,0,0);
  ierr = enforceMemoryLimit(PROTEUS_COMM_WORLD, rank, max_rss_gb,"Done allocating elementBoudnary old2new/new2old");CHKERRABORT(PROTEUS_COMM_WORLD, ierr);
  int build_subdomains_edges_event;
  PetscLogEventRegister("Subd edges",0,&build_subdomains_edges_event);
  PetscLogEventBegin(build_subdomains_edges_event,0,0,0,0);

  //
  //4c,5c. Repeate the process for edges
  //
  logEvent("Generating global edge numbering and ownership corresponding to new subdomain ownership",5);
  //the edges with a node of a subdomain element; the others play no part below.
  //Their markers become the subdomain's edgeMaterialTypes (in 2D the .edge
  //file is also the face file, read above for the face flags).
  int nEdges_global;
  int ihasEdgeMarkers(0);
  vector<int> edNumbers, edNodes, edIds;
  if (!readSimplexSubentitiesTouching(edgeFileName, format, "edge", 2, indexBase, true,
                                      nodes_old2new_subdomain_map, nEdges_global,
                                      ihasEdgeMarkers, edNumbers, edNodes, edIds))
    {
      failed = true;
      return failed;
    }
  assert(nEdges_global > 0);
  newMesh.nEdges_global = nEdges_global;
  set<int> edges_subdomain_owned;
  vector<set<int> > nodeEdgesStar(nNodes_subdomain_new[rank]);
  map<int,int> edgeMaterialTypesMap;
  map<int,vector<int> > elementEdgesMap;
  map<int,pair<int,int> > edgeNodesMap;
  set<int> supportedEdges;
  for (size_t k = 0; k < edNumbers.size(); k++)
    {
      const int ned = edNumbers[k], nn0 = edNodes[2*k], nn1 = edNodes[2*k+1];
      const int edId = edIds.empty() ? 0 : edIds[k];
      int nn0_new = -1;
      if (nodes_old2new_subdomain_map.find(nn0) != nodes_old2new_subdomain_map.end())
        nn0_new = nodes_old2new_subdomain_map[nn0];
      if (nn0_new >= nodeOffsets_new[rank] && nn0_new < nodeOffsets_new[rank+1])
        {
          nodeEdgesStar.at(nn0_new-nodeOffsets_new[rank]).insert(ned);
          supportedEdges.insert(ned);
        }
      int nn1_new = -2;
      if(nodes_old2new_subdomain_map.find(nn1) != nodes_old2new_subdomain_map.end())
        nn1_new = nodes_old2new_subdomain_map[nn1];
      if (nn1_new >= nodeOffsets_new[rank] && nn1_new < nodeOffsets_new[rank+1])
        {
          nodeEdgesStar.at(nn1_new-nodeOffsets_new[rank]).insert(ned);
          supportedEdges.insert(ned);
        }
      int nodes[2] = {nn0_new,nn1_new};
      NodeTuple<2> nodeTuple(nodes);
      if (edgeElementsMap.find(nodeTuple) != edgeElementsMap.end())//this edge is on an element in the subdomain
        {
          if (nodeTuple.nodes[0] >= nodeOffsets_new[rank] && nodeTuple.nodes[0] < nodeOffsets_new[rank+1])
            edges_subdomain_owned.insert(ned);
          //pick up all the edges on the subdomain and store their nodes
          edgeNodesMap[ned].first = nodeTuple.nodes[0];
          edgeNodesMap[ned].second = nodeTuple.nodes[1];
          //an edge flag is its marker or 0, never -1 (TetGen's marker for
          //the edges of segments), as in readEdgeMaterialTypes
          if (ihasEdgeMarkers > 0)
            edgeMaterialTypesMap[ned] = (edId == -1) ? 0 : edId;
          for (auto elementp=edgeElementsMap[nodeTuple].begin();
               elementp != edgeElementsMap[nodeTuple].end();
               elementp++)
            {
              int eN = elementNumbering_old2new_subdomain_map[elementp->first];
              if (elementEdgesMap.find(eN) != elementEdgesMap.end())
                {
                  elementEdgesMap[eN][elementp->second] = ned;
                }
              else
                {
                  std::vector<int> init(nEdges_element,-1);
                  elementEdgesMap[eN] = init;
                  elementEdgesMap[eN][elementp->second] = ned;
                }
            }
        }
    }//end iv
  edgeElementsMap.clear();
  int nEdges_owned_subdomain=edges_subdomain_owned.size(),
    nEdges_owned_new=0;
  MPI_Allreduce(&nEdges_owned_subdomain,&nEdges_owned_new,1,MPI_INT,MPI_SUM,PROTEUS_COMM_WORLD);
  assert(nEdges_owned_new == nEdges_global);
  ierr = enforceMemoryLimit(PROTEUS_COMM_WORLD, rank, max_rss_gb,"Done reading edges");CHKERRABORT(PROTEUS_COMM_WORLD, ierr);
  //done with edge file
  //
  //just as with faces, we need to add edges along outer boundaries of star
  //not sure if we need to collect nodeEdges star above anymore, since we're doing this
  //
  for (auto edgesp=elementEdgesMap.begin();
       edgesp!=elementEdgesMap.end();
       edgesp++)
    {
      //loop over the nodes of this element for the owned nodes
      for (int iv=0;iv<simplexDim;iv++)
        {
          //the elementNodesArrayMap is in the old element numbering while the elementEdgesMap is in the new element numbering
          int nN_global = -1;
          if (elementNumbering_new2old_subdomain_map.find(edgesp->first) != elementNumbering_new2old_subdomain_map.end())
            nN_global = elementNodesArrayMap[elementNumbering_new2old_subdomain_map[edgesp->first]][iv];
          if (nN_global >= nodeOffsets_new[rank] && nN_global < nodeOffsets_new[rank+1])
            {
              //add all the edges to this node star
              for(int ed=0;ed<nEdges_element;ed++)
                {
                  nodeEdgesStar.at(nN_global-nodeOffsets_new[rank]).insert(edgesp->second[ed]);
                }
            }
        }
    }
  //build compact data structures for nodeEdgesArray
  valarray<int> nodeEdgeOffsets(nNodes_subdomain_new[rank]+1);
  nodeEdgeOffsets[0] = 0;
  for (int nN = 0; nN < nNodes_subdomain_new[rank]; nN++)
    nodeEdgeOffsets[nN+1] = nodeEdgeOffsets[nN]+nodeEdgesStar.at(nN).size();
  valarray<int> nodeEdgesArray(nodeEdgeOffsets[nNodes_subdomain_new[rank]]);
  for (int nN=0,offset=0; nN < nNodes_subdomain_new[rank]; nN++)
    {
      for (auto edN_star = nodeEdgesStar.at(nN).begin();
           edN_star != nodeEdgesStar.at(nN).end();
           edN_star++,offset++)
        {
          nodeEdgesArray[offset] = *edN_star;
        }
    }
  newMesh.subdomainp->nEdges_global = edgeNodesMap.size();
  //get the number of edges on each processor
  valarray<int> nEdges_subdomain_new(size),
    edgeOffsets_new(size+1);
  for (int sdN=0;sdN<size;sdN++)
    if (sdN == rank)
      nEdges_subdomain_new[sdN] = edges_subdomain_owned.size();
    else
      nEdges_subdomain_new[sdN] = 0;
  valarray<int> nEdges_subdomain_new_send=nEdges_subdomain_new;
  MPI_Allreduce(&nEdges_subdomain_new_send[0],&nEdges_subdomain_new[0],size,MPI_INT,MPI_SUM,PROTEUS_COMM_WORLD);
  //
  //construct new offsets for owned edges
  //
  edgeOffsets_new[0] = 0;
  for (int sdN=0;sdN<size;sdN++)
    edgeOffsets_new[sdN+1] = edgeOffsets_new[sdN]+nEdges_subdomain_new[sdN];
  //
  //the new numbers of every edge of the subdomain, owned or not, from the
  //directory (no mappings-file read)
  const SubdomainNumbering edgeNumbering_old2new_subdomain_map(
    subdomainNumbersByDirectory(PROTEUS_COMM_WORLD, nEdges_global, edgeOffsets_new[rank],
                                edges_subdomain_owned, keysOf(edgeNodesMap), "edge"));
  ierr = enforceMemoryLimit(PROTEUS_COMM_WORLD, rank, max_rss_gb,"Done allocating edgeNumering old2new/new2old");CHKERRABORT(PROTEUS_COMM_WORLD, ierr);
  //
  //6. Figure out what is in the node stars but not locally owned, create ghost information
  //
  logEvent("Creating ghost information",5);
  set<int> elements_overlap,nodes_overlap,elementBoundaries_overlap,edges_overlap;
  for (int nN = 0; nN < nNodes_subdomain_new[rank]; nN++)
    {
      //nodes
      for (int offset = nodeStarOffsetsNew[nN];offset<nodeStarOffsetsNew[nN+1];offset++)
        {
          int nN_point_global = nodeStarArrayNew[offset];
          bool offproc = nN_point_global < nodeOffsets_new[rank] || nN_point_global >= nodeOffsets_new[rank+1];
          if (offproc)
            nodes_overlap.insert(nN_point_global);
        }
      //elements
      for (int eN_star_offset = nodeElementOffsets[nN];
           eN_star_offset < nodeElementOffsets[nN+1]; eN_star_offset++)
        {
          int eN_star_old = nodeElementsArray[eN_star_offset];
          int eN_star_new = elementNumbering_old2new_subdomain_map[eN_star_old];
          bool offproc = eN_star_new >= elementOffsets_new[rank+1] || eN_star_new < elementOffsets_new[rank];
          if (offproc)
            elements_overlap.insert(eN_star_new);
        }
      //element boundaries
      for (int ebN_star_offset = nodeElementBoundaryOffsets[nN];
           ebN_star_offset < nodeElementBoundaryOffsets[nN+1]; ebN_star_offset++)
        {
          int ebN_star_old = nodeElementBoundariesArray[ebN_star_offset];
          int ebN_star_new = elementBoundaryNumbering_old2new_subdomain_map[ebN_star_old];
          bool offproc = ebN_star_new >= elementBoundaryOffsets_new[rank+1] || ebN_star_new < elementBoundaryOffsets_new[rank];
          if (offproc)
            elementBoundaries_overlap.insert(ebN_star_new);
        }
      //edges in overlap
      for (int edN_star_offset = nodeEdgeOffsets[nN];
           edN_star_offset < nodeEdgeOffsets[nN+1]; edN_star_offset++)
        {
          int edN_star_old = nodeEdgesArray[edN_star_offset];
          int edN_star_new = edgeNumbering_old2new_subdomain_map[edN_star_old];
          bool offproc = edN_star_new >= edgeOffsets_new[rank+1] || edN_star_new < edgeOffsets_new[rank];
          if (offproc)
            edges_overlap.insert(edN_star_new);
        }
    }//nodes on this processor
  //cek debugging, edge overlap seems to be messed up. Check global node tuples of edges vs global edge numbers
  assert(edges_overlap.size() + nEdges_subdomain_new[rank] == edgeNodesMap.size());
  //
  //enumerate the overlap
  //
  int nN_subdomain = nNodes_subdomain_new[rank];
  map<int,int> nodes_overlap_global2subdomainMap;
  for (auto nN_globalp=nodes_overlap.begin();nN_globalp != nodes_overlap.end(); nN_globalp++,nN_subdomain++)
    nodes_overlap_global2subdomainMap[*nN_globalp] = nN_subdomain;
  //
  //7. add any addtional overlap, skip for now
  //
  //cek todo: re-enable additional overlap with out of core global maps
  assert(nNodes_overlap<2);
  logEvent("Skipping additiona of addtional overlap--only minimal overlap supported by this partioning function for now",5);
  //
  PetscLogEventEnd(build_subdomains_edges_event,0,0,0,0);
  int build_subdomains_renumber_event;
  PetscLogEventRegister("Subd's renumber",0,&build_subdomains_renumber_event);
  PetscLogEventBegin(build_subdomains_renumber_event,0,0,0,0);
  //
  //8. Build subdomain meshes in new numbering, assumes memory not allocated in subdomain mesh
  //
  if (simplexDim == 4)
    {
      logEvent("USER WARNING: You are partioning in parallel directly from Tetgen files (parun -F ...). ",5);
      logEvent("USER WARNING: You must use the 'f' and 'ee' flags in the triangleOptions input to avoid errors.",5);
      logEvent("USER WARNING: You may have done so, but the tetgen options are not stored in the Tetgen files, so we can't know.",5);
    }
  else if (rank==0)
    {
      std::cerr<<"USER WARNING: In order to avoid a segmentation fault, you need to have supplied the 'f' flag to the triangleOptions input."<<std::endl;
      std::cerr<<"USER WARNING: In order to avoid an edge assertion error, you need to have supplied the 'ee' flag to the triangleOptions input."<<std::endl;
    }
  logEvent("Creating final subdomain mesh data structures",5);
  if (newMesh.subdomainp == NULL)
    newMesh.subdomainp = new Mesh();
  newMesh.subdomainp->nElements_global = nElements_subdomain_new[rank] + elements_overlap.size();
  newMesh.subdomainp->nNodes_global    = nNodes_subdomain_new[rank] + nodes_overlap.size();
  newMesh.subdomainp->nElementBoundaries_global = nElementBoundaries_subdomain_new[rank]+elementBoundaries_overlap.size();
  assert(int(edges_subdomain_owned.size()+edges_overlap.size()) == newMesh.subdomainp->nEdges_global);
  //newMesh.subdomainp->nEdges_global   = edges_subdomain_owned.size()+edges_overlap.size();
  newMesh.subdomainp->nNodes_element   = newMesh.nNodes_element;
  newMesh.subdomainp->nNodes_elementBoundary = newMesh.nNodes_elementBoundary;
  newMesh.subdomainp->nElementBoundaries_element = newMesh.nElementBoundaries_element;
  //subdomain 2 global mappings (including ghost info)
  valarray<int> nodeNumbering_subdomain2global(newMesh.subdomainp->nNodes_global);
  valarray<int> elementNumbering_subdomain2global(newMesh.subdomainp->nElements_global);
  valarray<int> elementBoundaryNumbering_subdomain2global(newMesh.subdomainp->nElementBoundaries_global);
  valarray<int> edgeNumbering_subdomain2global(newMesh.subdomainp->nEdges_global);
  map<int,int> nodeNumbering_global2subdomainMap;
  map<int,int> elementBoundaryNumbering_global2subdomainMap;
  map<int,int> edgeNumbering_global2subdomainMap;
  newMesh.subdomainp->nodeArray = new double[newMesh.subdomainp->nNodes_global*3];
  //a node's flag is its vertex marker (or the attribute taken as one) when the
  //.node file gives one, and 0 ("no tag") otherwise: never derived from the
  //faces, which made it depend on the partition (the face visited last), never
  //-1 (rank interfaces are a face property). Without markers
  //readSubdomainVertices leaves the flags alone, so they stay this 0.
  newMesh.subdomainp->nodeMaterialTypes = new int[newMesh.subdomainp->nNodes_global]();
  //
  //now finally finish reading node coordinates and node flags
  //
  if (!readSubdomainVertices(vertexFileName, format, simplexDim-1, indexBase, hasVertexMarkers, nodes_old2new_subdomain_map,
                             nodeOffsets_new[rank], nodeOffsets_new[rank+1], nodes_overlap_global2subdomainMap,
                             &nodeNumbering_subdomain2global[0], nodeNumbering_global2subdomainMap,
                             newMesh.subdomainp->nodeArray, newMesh.subdomainp->nodeMaterialTypes))
    MPI_Abort(PROTEUS_COMM_WORLD, 1);
  //done with vertex file (and all file reads at this point)
  ierr = enforceMemoryLimit(PROTEUS_COMM_WORLD, rank, max_rss_gb,"Done reading vertices");CHKERRABORT(PROTEUS_COMM_WORLD, ierr);

  newMesh.subdomainp->elementNodesArray = new int[newMesh.subdomainp->nElements_global*newMesh.subdomainp->nNodes_element];
  newMesh.subdomainp->elementMaterialTypes = new int[newMesh.subdomainp->nElements_global];
  //
  //elements
  //
  //locally owned
  //
  for (int eN = 0; eN < nElements_subdomain_new[rank]; eN++)
    {
      int eN_global_new = elementOffsets_new[rank] + eN;
      int eN_global_old = elementNumbering_new2old_subdomain_map[eN_global_new];
      elementNumbering_subdomain2global[eN] = eN_global_new;
      newMesh.subdomainp->elementMaterialTypes[eN] = elementMaterialTypesMap[eN_global_old];
      for (int nN =  0; nN < newMesh.subdomainp->nNodes_element; nN++)
        {
          int nN_global_new = elementNodesArrayMap[eN_global_old][nN];
          int nN_subdomain  = nodeNumbering_global2subdomainMap[nN_global_new];
          newMesh.subdomainp->elementNodesArray[eN*newMesh.subdomainp->nNodes_element + nN]= nN_subdomain;
        }
    }
  //
  //ghost
  //
  set<int>::iterator eN_p = elements_overlap.begin();
  for (int eN = nElements_subdomain_new[rank]; eN < nElements_subdomain_new[rank] + int(elements_overlap.size()); eN++,eN_p++)
    {
      int eN_global_new = *eN_p;
      int eN_global_old = elementNumbering_new2old_subdomain_map[eN_global_new];
      elementNumbering_subdomain2global[eN] = eN_global_new;
      newMesh.subdomainp->elementMaterialTypes[eN] = elementMaterialTypesMap[eN_global_old];
      for (int nN =  0; nN < newMesh.subdomainp->nNodes_element; nN++)
        {
          int nN_global_new = elementNodesArrayMap[eN_global_old][nN];
          int nN_subdomain  = nodeNumbering_global2subdomainMap[nN_global_new];
          newMesh.subdomainp->elementNodesArray[eN*newMesh.subdomainp->nNodes_element + nN]= nN_subdomain;
        }
    }
  elementNodesArrayMap.clear();
  elementMaterialTypesMap.clear();
  //
  //element boundaries
  //
  //owned
  //
  for (int ebN=0; ebN < nElementBoundaries_subdomain_new[rank]; ebN++)
    {
      int ebN_global = ebN + elementBoundaryOffsets_new[rank];
      elementBoundaryNumbering_subdomain2global[ebN]=ebN_global;
      elementBoundaryNumbering_global2subdomainMap[ebN_global] = ebN;
    }
  //
  //ghost
  //
  set<int>::iterator ebN_p = elementBoundaries_overlap.begin();
  for(int ebN=nElementBoundaries_subdomain_new[rank];ebN < nElementBoundaries_subdomain_new[rank] + int(elementBoundaries_overlap.size()); ebN++,ebN_p++)
    {
      int ebN_global = *ebN_p;
      elementBoundaryNumbering_subdomain2global[ebN] = ebN_global;
      elementBoundaryNumbering_global2subdomainMap[ebN_global] = ebN;
    }
  //
  //need elementBoundariesArray to assign consistent numbering on subdomain
  //
  //local
  //
  newMesh.subdomainp->elementBoundariesArray =
    new int[newMesh.subdomainp->nElements_global*newMesh.subdomainp->nElementBoundaries_element];
  for (int eN=0;eN<nElements_subdomain_new[rank];eN++)
    {
      int eN_global = eN+elementOffsets_new[rank];
      for (int ebN=0;ebN<newMesh.subdomainp->nElementBoundaries_element;ebN++)
        {
          newMesh.subdomainp->elementBoundariesArray[eN*newMesh.subdomainp->nElementBoundaries_element+ebN] =
            elementBoundaryNumbering_global2subdomainMap[elementBoundaryNumbering_old2new_subdomain_map[elementBoundariesMap[eN_global][ebN]]];
        }
    }
  //
  //ghost elements
  //
  eN_p = elements_overlap.begin();
  for (int eN = nElements_subdomain_new[rank]; eN < nElements_subdomain_new[rank] + int(elements_overlap.size()); eN++,eN_p++)
    {
      int eN_global_new = *eN_p;
      for (int ebN=0;ebN<newMesh.subdomainp->nElementBoundaries_element;ebN++)
        {
          newMesh.subdomainp->elementBoundariesArray[eN*newMesh.subdomainp->nElementBoundaries_element+ebN] =
            elementBoundaryNumbering_global2subdomainMap[elementBoundaryNumbering_old2new_subdomain_map[elementBoundariesMap[eN_global_new][ebN]]];
        }
    }
  //
  //edges
  //
  //local
  //
  for (int edN=0; edN < nEdges_subdomain_new[rank]; edN++)
    {
      int edN_global = edN + edgeOffsets_new[rank];
      edgeNumbering_subdomain2global[edN]=edN_global;
      edgeNumbering_global2subdomainMap[edN_global] = edN;
    }
  //
  //ghost
  //
  set<int>::iterator edN_p = edges_overlap.begin();
  for(int edN=nEdges_subdomain_new[rank];edN < nEdges_subdomain_new[rank] + int(edges_overlap.size()); edN++,edN_p++)
    {
      int edN_global = *edN_p;
      edgeNumbering_subdomain2global[edN] = edN_global;
      edgeNumbering_global2subdomainMap[edN_global] = edN;
    }
  //
  //now build edgeNodes array in new numberings
  //
  newMesh.subdomainp->edgeNodesArray = new int[newMesh.subdomainp->nEdges_global*2];
  for (int i=0;i<newMesh.subdomainp->nEdges_global*2;i++)
    newMesh.subdomainp->edgeNodesArray[i] = -1;
  for (map<int,pair<int,int> >::iterator edgep=edgeNodesMap.begin();
       edgep!=edgeNodesMap.end();
       edgep++)
    {
      int edN_global_old = edgep->first;
      int edN_global_new = edgeNumbering_old2new_subdomain_map[edN_global_old];
      assert(edgeNumbering_global2subdomainMap.find(edN_global_new) != edgeNumbering_global2subdomainMap.end());
      int edN_subdomain  = edgeNumbering_global2subdomainMap[edN_global_new];
      newMesh.subdomainp->edgeNodesArray[edN_subdomain*2+0] = nodeNumbering_global2subdomainMap[edgep->second.first];
      newMesh.subdomainp->edgeNodesArray[edN_subdomain*2+1] = nodeNumbering_global2subdomainMap[edgep->second.second];
    }
  //
  //end edges
  //

  //now build rest of subdomain mesh connectivity information etc
  if constexpr (simplexDim == 4)
    constructElementBoundaryElementsArrayWithGivenElementBoundaryAndEdgeNumbers_tetrahedron(*newMesh.subdomainp);
  else
    constructElementBoundaryElementsArrayWithGivenElementBoundaryAndEdgeNumbers_triangle(*newMesh.subdomainp);
  //build local geometric info
  if constexpr (simplexDim == 4)
    {
      allocateGeometricInfo_tetrahedron(*newMesh.subdomainp);
      computeGeometricInfo_tetrahedron(*newMesh.subdomainp);
    }
  else
    {
      allocateGeometricInfo_triangle(*newMesh.subdomainp);
      computeGeometricInfo_triangle(*newMesh.subdomainp);
    }
  if (hasElementBoundaryMarkers)
    {
      assert(newMesh.subdomainp->elementBoundariesArray != NULL);
      for (auto ebmp = elementBoundaryMaterialTypesMap.begin(); ebmp != elementBoundaryMaterialTypesMap.end();ebmp++)
        {
          int ebN_global_new = elementBoundaryNumbering_old2new_subdomain_map[ebmp->first];
          assert(elementBoundaryNumbering_global2subdomainMap.find(ebN_global_new) != elementBoundaryNumbering_global2subdomainMap.end());
          int ebN_subdomain  = elementBoundaryNumbering_global2subdomainMap[ebN_global_new];
          newMesh.subdomainp->elementBoundaryMaterialTypes[ebN_subdomain] = ebmp->second;
        }
    }
  //the .edge markers, 0 where the file has none
  if (newMesh.subdomainp->edgeMaterialTypes != NULL)
    delete [] newMesh.subdomainp->edgeMaterialTypes;
  newMesh.subdomainp->edgeMaterialTypes = new int[newMesh.subdomainp->nEdges_global]();
  for (auto edmp = edgeMaterialTypesMap.begin(); edmp != edgeMaterialTypesMap.end(); edmp++)
    newMesh.subdomainp->edgeMaterialTypes[edgeNumbering_global2subdomainMap[edgeNumbering_old2new_subdomain_map[edmp->first]]]
      = edmp->second;
  ierr = enforceMemoryLimit(PROTEUS_COMM_WORLD, rank, max_rss_gb,"Done with material types");CHKERRABORT(PROTEUS_COMM_WORLD, ierr);
  PetscLogEventEnd(build_subdomains_renumber_event,0,0,0,0);
  int build_subdomains_cleanup_event;
  PetscLogEventRegister("Cleanup",0,&build_subdomains_cleanup_event);
  PetscLogEventBegin(build_subdomains_cleanup_event,0,0,0,0);
  //transfer information about owned nodes and elements to mesh
  logEvent("Cleaning up after partitioning",5);
  if (newMesh.nodeOffsets_subdomain_owned)
    delete [] newMesh.nodeOffsets_subdomain_owned;
  if (newMesh.elementOffsets_subdomain_owned)
    delete [] newMesh.elementOffsets_subdomain_owned;
  if (newMesh.elementBoundaryOffsets_subdomain_owned)
    delete [] newMesh.elementBoundaryOffsets_subdomain_owned;
  if (newMesh.edgeOffsets_subdomain_owned)
    delete [] newMesh.edgeOffsets_subdomain_owned;
  newMesh.nodeOffsets_subdomain_owned    = new int[size+1];
  newMesh.elementOffsets_subdomain_owned = new int[size+1];
  newMesh.elementBoundaryOffsets_subdomain_owned = new int[size+1];
  newMesh.edgeOffsets_subdomain_owned = new int[size+1];
  for (int sdN = 0; sdN < size+1; sdN++)
    {
      newMesh.nodeOffsets_subdomain_owned[sdN]    = nodeOffsets_new[sdN];
      newMesh.elementOffsets_subdomain_owned[sdN] = elementOffsets_new[sdN];
      newMesh.elementBoundaryOffsets_subdomain_owned[sdN] = elementBoundaryOffsets_new[sdN];
      newMesh.edgeOffsets_subdomain_owned[sdN] = edgeOffsets_new[sdN];
    }
  if (newMesh.nodeNumbering_subdomain2global)
    delete [] newMesh.nodeNumbering_subdomain2global;
  newMesh.nodeNumbering_subdomain2global = new int[newMesh.subdomainp->nNodes_global];
  for (int nN = 0; nN < newMesh.subdomainp->nNodes_global; nN++)
    newMesh.nodeNumbering_subdomain2global[nN] = nodeNumbering_subdomain2global[nN];
  if (newMesh.elementNumbering_subdomain2global)
    delete [] newMesh.elementNumbering_subdomain2global;
  newMesh.elementNumbering_subdomain2global = new int[newMesh.subdomainp->nElements_global];
  for (int eN = 0; eN < newMesh.subdomainp->nElements_global; eN++)
    newMesh.elementNumbering_subdomain2global[eN] = elementNumbering_subdomain2global[eN];
  //
  if (newMesh.elementBoundaryNumbering_subdomain2global)
    delete [] newMesh.elementBoundaryNumbering_subdomain2global;
  newMesh.elementBoundaryNumbering_subdomain2global = new int[newMesh.subdomainp->nElementBoundaries_global];
  for (int ebN = 0; ebN < newMesh.subdomainp->nElementBoundaries_global; ebN++)
    newMesh.elementBoundaryNumbering_subdomain2global[ebN] = elementBoundaryNumbering_subdomain2global[ebN];
  //
  if (newMesh.edgeNumbering_subdomain2global)
    delete [] newMesh.edgeNumbering_subdomain2global;
  newMesh.edgeNumbering_subdomain2global = new int[newMesh.subdomainp->nEdges_global];
  for (int i=0; i< newMesh.subdomainp->nEdges_global; i++)
    newMesh.edgeNumbering_subdomain2global[i] = edgeNumbering_subdomain2global[i];
  //the faces the partition cut read -1 (rank interfaces), as on the in-memory paths
  markRankInterfacesByOwner(PROTEUS_COMM_WORLD, newMesh, hasElementBoundaryMarkers);
  //cleanup
  PetscLogEventEnd(build_subdomains_cleanup_event,0,0,0,0);
  return false;
}

/* Steps 0-4 of the parun -F partitioners, then steps 4b-8
   (buildSubdomainFromFiles). One body for triangles (simplexDim 3: Triangle
   .node/.ele, its .edge file as both faces and edges) and tetrahedra
   (simplexDim 4: TetGen .node/.ele/.face/.edge). The global mesh is never
   read on one rank:

    1. Partition the nodes into the default contiguous chunks of the file
       ordering (nodeOffsets_old)
    2. Read the stars of the chunk's nodes from the element file and build
       the node graph (readSimplexNodeStars)
    3. Repartition the nodes with the PETSc graph partitioner
       (MatPartitioning), number them by owner, and gather the permutation
       into newMesh.nodeNumbering_global_old2new (SW2DCV and GN_SW2DCV read
       it; no mappings.h5 since 2026-11)
    4. Reread the element file for the elements in the support of the owned
       nodes, with the new numbers of their nodes from the ranks that
       computed them (readSubdomainElements)
    4b-8. buildSubdomainFromFiles

   max_rss_gb <= 0 skips the memory checks. Returns nonzero on a failed read
   (every rank reads; a missing file fails on every rank). Collective. */
template<int simplexDim>
int partitionNodesFromFiles(const MPI_Comm& PROTEUS_COMM_WORLD, const char* filebase, int indexBase,
                            Mesh& newMesh, int nNodes_overlap, double max_rss_gb)
{
  using namespace std;
  using namespace IOutils;
  static_assert(simplexDim == 3 || simplexDim == 4, "triangles or tetrahedra");
  const char* format = (simplexDim == 4) ? "Tetgen" : "triangle";
  PetscErrorCode ierr;
  PetscMPIInt size,rank;
  ierr = MPI_Comm_size(PROTEUS_COMM_WORLD,&size);CHKERRABORT(PROTEUS_COMM_WORLD, ierr);
  ierr = MPI_Comm_rank(PROTEUS_COMM_WORLD,&rank);CHKERRABORT(PROTEUS_COMM_WORLD, ierr);
  //cek todo: decide whether to remove PETSc logging and just use proteus logging
  PetscLogStage partitioning_stage;
  PetscLogStageRegister("Mesh Partition",&partitioning_stage);
  PetscLogStagePush(partitioning_stage);
  //
  //0. File names. Tetgen should have been run with -feen (all faces and
  //   edges), triangle with -en (all edges and neighbors)
  //
  const std::string vertexFileName  = std::string(filebase) + ".node" ;
  const std::string elementFileName = std::string(filebase) + ".ele" ;
  const std::string elementBoundaryFileName = std::string(filebase) + ((simplexDim == 4) ? ".face" : ".edge");
  const std::string edgeFileName  = std::string(filebase) + ".edge" ;
  //
  //1. Build default nodal partition
  //
  //read the number of nodes and whether there are node tags; the node
  //coordinates are read after partitioning (readSubdomainVertices), which
  //just requires topology information
  logEvent("Building default partitioning",5);
  int read_elements_event;
  PetscLogEventRegister("Read eles",0,&read_elements_event);
  PetscLogEventBegin(read_elements_event,0,0,0,0);
  std::ifstream vertexFile(vertexFileName.c_str());
  if (!vertexFile.good())
    {
      writeErrorLine(std::string("cannot open ")+format+" node file "+vertexFileName);
      return 1;
    }
  int hasVertexMarkers(0),hasVertexAttributes(0),nSpace(simplexDim-1),nNodes_global;
  vertexFile >> eatcomments >> nNodes_global >> nSpace >> hasVertexAttributes >> hasVertexMarkers >> eatline ;
  assert(nNodes_global > 0);
  assert(nSpace == simplexDim-1);
  vertexFile.close();
  newMesh.nNodes_global = nNodes_global;
  newMesh.nNodes_element = simplexDim;
  newMesh.nNodes_elementBoundary = simplexDim-1;
  newMesh.nElementBoundaries_element = simplexDim;
  //cek todo: review the situation with tetgen and triangle atribs and markers
  //every rank reads the attribute as the flag (as the in-memory readers do);
  //only the warning is rank 0's
  if (hasVertexAttributes > 0)
    {
      if (rank == 0)
        std::cerr<<"WARNING "<<format<<" nodes hasAttributes= "<<hasVertexAttributes
                 <<" > 0 will treat first value as integer id for boundary!!"<<std::endl;
      hasVertexMarkers = 1;
    }
  //offsets provide the lower and upper bounds for the global numbering
  //first we just partition the nodes among the MPI ranks, approximately equally, ignoring connectivity
  valarray<int> nodeOffsets_old(size+1);
  nodeOffsets_old[0] = 0;
  for (int sdN=0; sdN < size; sdN++)
    {
      nodeOffsets_old[sdN+1] = nodeOffsets_old[sdN] +
        int(nNodes_global)/size + (int(nNodes_global)%size > sdN);
    }
  int nNodes_subdomain_old = nodeOffsets_old[rank+1] - nodeOffsets_old[rank];
  int nNodes_subdomain_max = int(nNodes_global)/size + (int(nNodes_global)%size > 0);
  char log_buffer[100];
  sprintf(log_buffer,"First partitioning: Max nNodes_subdomain %d nNodes_global %d",nNodes_subdomain_max,nNodes_global);
  logEvent(log_buffer,5);
  //
  //2. Determine nodal connectivity (nodeStar) for nodes on subdomain
  //
  //connectivity comes from the topology (elements) file. We just grab the elements
  //that contain currently owned nodes
  logEvent("Determining nodal connectivity to construct new partitioning",5);
  int nElements_global;
  vector<set<int> > nodeStar;
  if (!readSimplexNodeStars(elementFileName,format,simplexDim,indexBase,nNodes_global,
                            nodeOffsets_old[rank],nodeOffsets_old[rank+1],
                            nElements_global,nodeStar))
    return 1;
  newMesh.nElements_global = nElements_global;
  PetscLogEventEnd(read_elements_event,0,0,0,0);
  int repartition_nodes_event;
  PetscLogEventRegister("Repart nodes",0,&repartition_nodes_event);
  PetscLogEventBegin(repartition_nodes_event,0,0,0,0);
  //compact graph for PETSc: the star sets are sorted, so each row's
  //neighbors are ascending; nodes and edges are weighted by the star size
  int max_nNodeNeighbors_node=0;
  for (int nN=0;nN<nNodes_subdomain_old;nN++)
    max_nNodeNeighbors_node=max(max_nNodeNeighbors_node,static_cast<int>(nodeStar[nN].size()));
  PetscInt *nodeNeighborsOffsets_subdomain,*nodeNeighbors_subdomain,*weights_subdomain,*vertex_weights_subdomain;
  PetscReal *partition_weights;
  PetscMalloc(sizeof(PetscInt)*(nNodes_subdomain_old+1),&nodeNeighborsOffsets_subdomain);
  PetscMalloc(sizeof(PetscInt)*(nNodes_subdomain_old*max_nNodeNeighbors_node),&nodeNeighbors_subdomain);
  PetscMalloc(sizeof(PetscInt)*(nNodes_subdomain_old*max_nNodeNeighbors_node),&weights_subdomain);
  PetscMalloc(sizeof(PetscInt)*(nNodes_subdomain_old),&vertex_weights_subdomain);
  PetscMalloc(sizeof(PetscReal)*(size),&partition_weights);
  for (int sd=0;sd<size;sd++)
    partition_weights[sd] = 1.0/double(size);
  nodeNeighborsOffsets_subdomain[0] = 0;
  for (int nN = 0,offset=0; nN < nNodes_subdomain_old; nN++)
    {
      for (auto nN_star=nodeStar[nN].begin(); nN_star!=nodeStar[nN].end(); nN_star++,offset++)
        nodeNeighbors_subdomain[offset] = *nN_star;
      nodeNeighborsOffsets_subdomain[nN+1]=offset;
      int weight= (nodeNeighborsOffsets_subdomain[nN+1] - nodeNeighborsOffsets_subdomain[nN]);
      vertex_weights_subdomain[nN] = weight;
      for (int k=nodeNeighborsOffsets_subdomain[nN];k<nodeNeighborsOffsets_subdomain[nN+1];k++)
        weights_subdomain[k] = weight;
    }
  nodeStar.clear();
  //
  //3. Generate new nodal partition using PETSc interface to graph partitioners
  //
  logEvent("Constructing new nodal partition",5);
  Mat petscAdjacency;
  ierr = MatCreateMPIAdj(PROTEUS_COMM_WORLD,
                         nNodes_subdomain_old,
                         nNodes_global,
                         nodeNeighborsOffsets_subdomain,
                         nodeNeighbors_subdomain,
                         weights_subdomain,
                         &petscAdjacency);CHKERRABORT(PROTEUS_COMM_WORLD, ierr);
  ierr = enforceMemoryLimit(PROTEUS_COMM_WORLD, rank, max_rss_gb,"Done allocating MPIAdj");CHKERRABORT(PROTEUS_COMM_WORLD, ierr);
  MatPartitioning petscPartition;
  ierr = MatPartitioningCreate(PROTEUS_COMM_WORLD,&petscPartition);CHKERRABORT(PROTEUS_COMM_WORLD, ierr);
  ierr = MatPartitioningSetAdjacency(petscPartition,petscAdjacency);CHKERRABORT(PROTEUS_COMM_WORLD, ierr);
  ierr = MatPartitioningSetFromOptions(petscPartition);CHKERRABORT(PROTEUS_COMM_WORLD, ierr);
  warnIfNotRepartitioning(PROTEUS_COMM_WORLD, petscPartition);
  ierr = MatPartitioningSetVertexWeights(petscPartition,vertex_weights_subdomain);CHKERRABORT(PROTEUS_COMM_WORLD, ierr);
  ierr = MatPartitioningSetPartitionWeights(petscPartition,partition_weights);CHKERRABORT(PROTEUS_COMM_WORLD, ierr);
  //get petsc index set that has the new subdomain number for each node
  IS nodePartitioningIS_new;
  ierr = MatPartitioningApply(petscPartition,&nodePartitioningIS_new);CHKERRABORT(PROTEUS_COMM_WORLD, ierr);
  ierr = MatPartitioningDestroy(&petscPartition);CHKERRABORT(PROTEUS_COMM_WORLD, ierr); //gets petscAdjacency too I believe
  ierr = enforceMemoryLimit(PROTEUS_COMM_WORLD, rank, max_rss_gb,"Done applying partition");CHKERRABORT(PROTEUS_COMM_WORLD, ierr);
  //determine the number of nodes per subdomain in new partitioning
  valarray<int> nNodes_subdomain_new(size);
  ISPartitioningCount(nodePartitioningIS_new,size,&nNodes_subdomain_new[0]);
  //need new offsets for subdomain to global numbering
  valarray<int> nodeOffsets_new(size+1);
  nodeOffsets_new[0] = 0;
  nNodes_subdomain_max = 0;
  for (int sdN = 0; sdN < size; sdN++)
    {
      nodeOffsets_new[sdN+1] = nodeOffsets_new[sdN] + nNodes_subdomain_new[sdN];
      nNodes_subdomain_max=max(nNodes_subdomain_max, nNodes_subdomain_new[sdN]);
    }
  sprintf(log_buffer,"Final partitioning: Max nNodes_subdomain %d nNodes_global %d",nNodes_subdomain_max,nNodes_global);
  logEvent(log_buffer,5);
  //get the new node numbers for nodes on this subdomain
  IS nodeNumberingIS_subdomain_old2new;
  ISPartitioningToNumbering(nodePartitioningIS_new,&nodeNumberingIS_subdomain_old2new);
  ierr=ISDestroy(&nodePartitioningIS_new);CHKERRABORT(PROTEUS_COMM_WORLD, ierr);
  const PetscInt* nodes_subdomain_old2new_array;
  ISGetIndices(nodeNumberingIS_subdomain_old2new, &nodes_subdomain_old2new_array);
  //
  //the mesh's nodeNumbering_global_old2new, the whole permutation on every
  //rank, from each rank's default chunk: what partitionNodes keeps, and what
  //SW2DCV and GN_SW2DCV read to permute a bathymetry given in the file's node
  //order (they read it from mappings.h5, written here until 2026-11, which a
  //run without -F never wrote and so could find stale). One MPI_Allgatherv;
  //nNodes_global ints per rank.
  //
  logEvent("Gathering the node numbering",5);
  {
    std::vector<int> chunk(nodes_subdomain_old2new_array, nodes_subdomain_old2new_array+nNodes_subdomain_old);
    std::vector<int> counts(size), displs(size);
    for (int r = 0; r < size; r++)
      {
        counts[r] = nodeOffsets_old[r+1] - nodeOffsets_old[r];
        displs[r] = nodeOffsets_old[r];
      }
    if (newMesh.nodeNumbering_global_old2new != NULL)
      delete [] newMesh.nodeNumbering_global_old2new;
    newMesh.nodeNumbering_global_old2new = new proteus_index_t[nNodes_global];
    static_assert(sizeof(proteus_index_t) == sizeof(int), "MPI_INT carries proteus_index_t");
    MPI_Allgatherv(chunk.data(), nNodes_subdomain_old, MPI_INT,
                   newMesh.nodeNumbering_global_old2new, counts.data(), displs.data(), MPI_INT,
                   PROTEUS_COMM_WORLD);
  }
  PetscLogEventEnd(repartition_nodes_event,0,0,0,0);
  ierr = enforceMemoryLimit(PROTEUS_COMM_WORLD, rank, max_rss_gb,"Done with masks");CHKERRABORT(PROTEUS_COMM_WORLD, ierr);
  //
  //4. Reread the element file for the elements in the support of the owned
  //   nodes (OLD element numbers), and assign their ownership. The new
  //   numbers of their nodes come from the ranks that computed them
  //   (readSubdomainElements answers other ranks' old->new requests from
  //   nodes_subdomain_old2new_array).
  //
  logEvent("Collecting elements containing locally owned nodes (element support of the subdomain nodes)",5);
  int build_subdomains_reread_elements_event;
  PetscLogEventRegister("Reread eles",0,&build_subdomains_reread_elements_event);
  PetscLogEventBegin(build_subdomains_reread_elements_event,0,0,0,0);
  set<int> elements_subdomain_owned;
  vector<set<int> > nodeElementsStar(nNodes_subdomain_new[rank]);
  vector<set<int> > nodeStarNew(nNodes_subdomain_new[rank]);
  map<int,vector<int> > elementNodesArrayMap;
  map<int,long int> elementMaterialTypesMap;
  map<NodeTuple<simplexDim-1>,ElementNeighbors> elementBoundaryElementsMap;
  map<NodeTuple<2>,set<pair<int,int> > > edgeElementsMap;
  map<int,int> nodes_old2new_subdomain_map;
  const bool readElements =
    readSubdomainElements<simplexDim,vector<int> >(PROTEUS_COMM_WORLD, elementFileName, format, indexBase, nNodes_global,
                                                   nodeOffsets_old, nodeOffsets_new, nodes_subdomain_old2new_array,
                                                   nElements_global, elements_subdomain_owned,
                                                   nodeElementsStar, nodeStarNew, elementNodesArrayMap,
                                                   elementMaterialTypesMap, elementBoundaryElementsMap,
                                                   edgeElementsMap, nodes_old2new_subdomain_map);
  ISRestoreIndices(nodeNumberingIS_subdomain_old2new, &nodes_subdomain_old2new_array);
  ierr=ISDestroy(&nodeNumberingIS_subdomain_old2new);CHKERRABORT(PROTEUS_COMM_WORLD, ierr);
  if (!readElements)
    return 1;
  int nElements_owned_subdomain(elements_subdomain_owned.size()),
    nElements_owned_new=0;
  MPI_Allreduce(&nElements_owned_subdomain,&nElements_owned_new,1,MPI_INT,MPI_SUM,PROTEUS_COMM_WORLD);
  assert(nElements_owned_new == nElements_global);
  PetscLogEventEnd(build_subdomains_reread_elements_event,0,0,0,0);
  if (buildSubdomainFromFiles<simplexDim>(PROTEUS_COMM_WORLD, rank, size, format,
                                          elementBoundaryFileName, edgeFileName, vertexFileName,
                                          indexBase, hasVertexMarkers, nNodes_overlap, max_rss_gb, nElements_global,
                                          nNodes_subdomain_new, nodeOffsets_new, elements_subdomain_owned,
                                          nodeElementsStar, nodeStarNew, elementNodesArrayMap,
                                          elementMaterialTypesMap, elementBoundaryElementsMap,
                                          edgeElementsMap, nodes_old2new_subdomain_map, newMesh))
    return 1;
  PetscLogStagePop();
  PetscLogView(PETSC_VIEWER_STDOUT_WORLD);
  ierr = enforceMemoryLimit(PROTEUS_COMM_WORLD, rank, max_rss_gb,"Done with partitioning!");CHKERRABORT(PROTEUS_COMM_WORLD, ierr);
  return 0;
}

int partitionNodesFromTetgenFiles(const MPI_Comm& PROTEUS_COMM_WORLD, const char* filebase, int indexBase,
                                  Mesh& newMesh, int nNodes_overlap, double memHardLimit)
{
  //memHardLimit is in MB (parun's memHardLimit); the checks take GB
  return partitionNodesFromFiles<4>(PROTEUS_COMM_WORLD, filebase, indexBase, newMesh, nNodes_overlap,
                                    memHardLimit/1024.0);
}

int partitionNodesFromTriangleFiles(const MPI_Comm& PROTEUS_COMM_WORLD, const char* filebase, int indexBase,
                                    Mesh& newMesh, int nNodes_overlap)
{
  //no memory checks (the Triangle path never had a limit argument)
  return partitionNodesFromFiles<3>(PROTEUS_COMM_WORLD, filebase, indexBase, newMesh, nNodes_overlap, 0.0);
}
//todo add overlap for element based partitions
/* The ownership rules of partitionElements, which follow from the owner of
   every element. Every rank computes them over the whole (not yet renumbered)
   mesh, so they need no communication. Plain C++. An entity no rank reaches
   is given the owner size. */

//A node belongs to the lowest rank that owns an element containing it.
std::vector<int> nodeOwnersFromElementOwners(const Mesh& mesh, const std::vector<int>& elementOwner, int size)
{
  std::vector<int> nodeOwner(mesh.nNodes_global,size);
  for (int eN=0;eN<mesh.nElements_global;eN++)
    for (int nN=0;nN<mesh.nNodes_element;nN++)
      lowerOwner(nodeOwner,mesh.elementNodesArray[eN*mesh.nNodes_element+nN],elementOwner[eN]);
  return nodeOwner;
}

//An element boundary belongs to the lowest rank that owns an element it bounds.
std::vector<int> elementBoundaryOwnersFromElementOwners(const Mesh& mesh, const std::vector<int>& elementOwner, int size)
{
  std::vector<int> elementBoundaryOwner(mesh.nElementBoundaries_global,size);
  for (int eN=0;eN<mesh.nElements_global;eN++)
    for (int ebN=0;ebN<mesh.nElementBoundaries_element;ebN++)
      lowerOwner(elementBoundaryOwner,mesh.elementBoundariesArray[eN*mesh.nElementBoundaries_element+ebN],elementOwner[eN]);
  return elementBoundaryOwner;
}

//An edge belongs to the lowest rank that owns an element with both its nodes.
std::vector<int> edgeOwnersFromElementOwners(const Mesh& mesh, const std::vector<int>& elementOwner, int size)
{
  std::vector<int> edgeOwner(mesh.nEdges_global,size);
  for (int ig = 0; ig < mesh.nEdges_global; ig++)
    {
      const int nN0_old = mesh.edgeNodesArray[2*ig], nN1_old = mesh.edgeNodesArray[2*ig+1];
      for (int offset=mesh.nodeElementOffsets[nN0_old];offset<mesh.nodeElementOffsets[nN0_old+1];offset++)
        {
          const int eN = mesh.nodeElementsArray[offset];
          for (int nN=0;nN<mesh.nNodes_element;nN++)
            if (mesh.elementNodesArray[eN*mesh.nNodes_element+nN] == nN1_old)
              lowerOwner(edgeOwner,ig,elementOwner[eN]);
        }
    }
  return edgeOwner;
}

/* Rows begin..end-1 of the element graph the graph partitioner is given: each
   element's neighbours across its element boundaries, sorted, in CSR form. */
void elementGraphRows(const Mesh& mesh, int begin, int end, std::vector<int>& offsets, std::vector<int>& neighbors)
{
  offsets.assign(1,0);
  neighbors.clear();
  for (int eN_global = begin; eN_global < end; eN_global++)
    {
      for (int ebN=0; ebN< mesh.nElementBoundaries_element;ebN++)
        {
          int eN_neighbor_global = mesh.elementNeighborsArray[eN_global*mesh.nElementBoundaries_element + ebN];
          if (eN_neighbor_global >= 0 )
            neighbors.push_back(eN_neighbor_global);
        }
      offsets.push_back(int(neighbors.size()));
      std::sort(neighbors.begin()+offsets[offsets.size()-2],neighbors.end());
    }
}

/* MPI: the owner of every element from the graph partitioner, each rank
   handing MatPartitioning its default chunk of the element graph; the answer is
   gathered so every rank holds every owner. Collective. */
std::vector<int> graphElementOwners(const MPI_Comm& PROTEUS_COMM_WORLD, const Mesh& mesh,
                                    const std::valarray<int>& elementOffsets_old)
{
  using namespace std;
  int ierr,rank;
  MPI_Comm_rank(PROTEUS_COMM_WORLD,&rank);
  //2. Extract subdomain element adjacency information could read
  //only the required portion from a file
  int nElements_subdomain = (elementOffsets_old[rank+1] - elementOffsets_old[rank]);
  std::vector<int> rowOffsets, rowNeighbors;
  elementGraphRows(mesh,elementOffsets_old[rank],elementOffsets_old[rank+1],rowOffsets,rowNeighbors);
  //MatCreateMPIAdj takes ownership of PetscMalloc'd arrays
  PetscInt *elementNeighborsOffsets_subdomain,*elementNeighbors_subdomain;
  PetscMalloc(sizeof(PetscInt)*(nElements_subdomain+1),&elementNeighborsOffsets_subdomain);
  PetscMalloc(sizeof(PetscInt)*(nElements_subdomain*mesh.nElementBoundaries_element),&elementNeighbors_subdomain);
  for (int eN = 0; eN < nElements_subdomain+1; eN++)
    elementNeighborsOffsets_subdomain[eN] = rowOffsets[eN];
  for (size_t k = 0; k < rowNeighbors.size(); k++)
    elementNeighbors_subdomain[k] = rowNeighbors[k];
  //3. Generate the  new partitiong using PETSc, this is done in parallel using parmetis
  Mat petscAdjacency;
  ierr = MatCreateMPIAdj(PROTEUS_COMM_WORLD,
                         nElements_subdomain,
                         mesh.nElements_global,
                         elementNeighborsOffsets_subdomain,
                         elementNeighbors_subdomain,
                         PETSC_NULL,
                         &petscAdjacency);CHKERRABORT(PROTEUS_COMM_WORLD, ierr);
  IS elementPartitioningIS_new;
  MatPartitioning petscPartition;
  MatPartitioningCreate(PROTEUS_COMM_WORLD,&petscPartition);
  MatPartitioningSetAdjacency(petscPartition,petscAdjacency);
  MatPartitioningSetFromOptions(petscPartition);
  warnIfNotRepartitioning(PROTEUS_COMM_WORLD, petscPartition);

  //get a petsc index set that has the new submdomain number for each element
  MatPartitioningApply(petscPartition,&elementPartitioningIS_new);
  MatPartitioningDestroy(&petscPartition);
  MatDestroy(&petscAdjacency);
  std::vector<int> elementOwner = gatherPartition(PROTEUS_COMM_WORLD,elementPartitioningIS_new,elementOffsets_old);
  ISDestroy(&elementPartitioningIS_new);
  return elementOwner;
}

int partitionElements(const MPI_Comm& PROTEUS_COMM_WORLD, Mesh& mesh, int nElements_overlap,
                      const int* elementOwners)
{
  using namespace std;
  int ierr,size,rank;

  ierr = MPI_Comm_size(PROTEUS_COMM_WORLD,&size);
  ierr = MPI_Comm_rank(PROTEUS_COMM_WORLD,&rank);
  StageTimer timer(PROTEUS_COMM_WORLD,"partitionElements");

  //Contents
  //
  //1. Partition the elements in the "default" partition (contiguous
  //chunks in given ordering)
  //
  //2. Partition the elementNeighbors based on this partition
  //
  //3. Pass to Parmetis to build a better partition of the elements
  //
  //4. Tag a subset of the nodes and faces on the subdomain elements as owned
  //using a mark and pass approach.**
  //
  //5. Extract the nodes and faces in the
  //overlapping elements.**
  //
  //6. Build the subdomain mesh from the
  //subdomain elements
  //
  //**To be more general we could get all the support (i.e. faces
  //and edges) and partitiong them, but the main reason for
  //partitioning is to keep track of a global numbering for degrees
  //of freedom that live on each type of geometric entity. We only
  //have node and element based DOFs so I just rebuild the other
  //information once we have elements and nodes partitioned.
  //
  // \todo check that I restore all data that PETSc expects to have
  // back, add PETSc error checking macros
  //
  //1. Build default partitioning
  //
  //get offsets so we can calculate the processor to global mapping
  //for elements in the old (default) partitioning
  valarray<int> elementOffsets_old = defaultChunkOffsets(mesh.nElements_global,size);

  //2.-3. The owner of every element: prescribed, or from the graph partitioner
  std::vector<int> elementOwner;
  if (elementOwners != NULL)
    {
      /* The caller dictates the partition -- see partitionNodes. elementOwners[e]
         is the rank that owns element e in the current numbering; a finer level
         of a multilevel hierarchy gives each element its parent's owner. Nothing
         on this path uses PETSc. */
      elementOwner = prescribedOwners(PROTEUS_COMM_WORLD,elementOwners,mesh.nElements_global,size,"partitionElements","element");
    }
  else
    elementOwner = graphElementOwners(PROTEUS_COMM_WORLD,mesh,elementOffsets_old);
  timer.mark("element_owners");
  //4.-6. Everything else follows from the owners without communication
  if (partitionElementsLocal(mesh, rank, size, nElements_overlap, elementOwner, &timer) != 0)
    MPI_Abort(PROTEUS_COMM_WORLD,1);
  timer.report();
  return 0;
}

/* Steps 4-6 of partitionElements: given the owner of every element, which
   every rank holds, this rank's part of the partition. Plain C++ -- no MPI,
   no PETSc, no files -- so one process can compute any rank's subdomain. Like
   partitionElements it renumbers the global mesh in place (each rank's owned
   entities contiguous) and writes the subdomain mesh, the offsets, the
   subdomain-to-global maps and the element permutation to mesh. Returns
   nonzero, having said why on stderr, if a node, element boundary or edge is
   reached by no rank. */
int partitionElementsLocal(Mesh& mesh, int rank, int size, int nElements_overlap,
                           const std::vector<int>& elementOwner, StageTimer* timer)
{
  using namespace std;
  //number the elements: each rank's owned elements contiguous, in their old order
  valarray<int> nElements_subdomain_new, elementOffsets_new;
  std::vector<int> elementNumbering_global_old2new, elementNumbering_global_new2old;
  numberByOwner(elementOwner,size,nElements_subdomain_new,elementOffsets_new,
                elementNumbering_global_old2new,elementNumbering_global_new2old);
  /* keep the permutation, as for nodes: a multilevel hierarchy's parent/child
     relations are numbered before partitioning and need it to be read in the
     numbering elementNumbering_subdomain2global refers to */
  if (mesh.elementNumbering_global_old2new != NULL)
    delete [] mesh.elementNumbering_global_old2new;
  mesh.elementNumbering_global_old2new = new int[mesh.nElements_global];
  for (int eN=0;eN<mesh.nElements_global;eN++)
    mesh.elementNumbering_global_old2new[eN] = elementNumbering_global_old2new[eN];

  if (timer) timer->mark("element_numbering");
  //Sort element based arrays, maybe I don't need to do this, maybe
  //I just need to start writing into the subdomain mesh here and
  //preserve subdomain2old and subdomain2global mappings
  valarray<int> elementNodesArray_new(mesh.nElements_global*mesh.nNodes_element),
    elementNeighborsArray_new(mesh.nElements_global*mesh.nElementBoundaries_element),
    elementMaterialTypes_new(mesh.nElements_global),
    elementBoundariesArray_new(mesh.nElements_global*mesh.nElementBoundaries_element),
    elementBoundaryElementsArray_new(mesh.nElementBoundaries_global*2),
    elementBoundaryLocalElementBoundariesArray_new(mesh.nElementBoundaries_global*2),
    elementBoundaryNodesArray_new(mesh.nElementBoundaries_global*mesh.nNodes_elementBoundary),
    edgeNodesArray_new(mesh.nEdges_global*2);
  valarray<int> elementBoundaryMaterialTypes_new(mesh.nElementBoundaries_global);
  for (int eN=0;eN<mesh.nElements_global;eN++)
    {
      for (int nN=0;nN<mesh.nNodes_element;nN++)
        elementNodesArray_new[eN*mesh.nNodes_element + nN] =
          mesh.elementNodesArray[elementNumbering_global_new2old[eN]*mesh.nNodes_element+nN];
      for (int ebN=0;ebN<mesh.nElementBoundaries_element;ebN++)
        {
          elementNeighborsArray_new[eN*mesh.nElementBoundaries_element+ebN] =
            mesh.elementNeighborsArray[elementNumbering_global_new2old[eN]*mesh.nElementBoundaries_element+ebN];
          //need new elements --> old element boundary numbers for now
          elementBoundariesArray_new[eN*mesh.nElementBoundaries_element+ebN] =
            mesh.elementBoundariesArray[elementNumbering_global_new2old[eN]*mesh.nElementBoundaries_element+ebN];
        }
      elementMaterialTypes_new[eN] = mesh.elementMaterialTypes[elementNumbering_global_new2old[eN]];

    }
  //renumber references to element numbers
  for (int eN=0;eN<mesh.nElements_global;eN++)
    {
      for (int ebN=0;ebN<mesh.nElementBoundaries_element;ebN++)
        {
          int eN_ebN = elementNeighborsArray_new[eN*mesh.nElementBoundaries_element+ebN];
          if (eN_ebN >= 0)
            elementNeighborsArray_new[eN*mesh.nElementBoundaries_element+ebN] =
              elementNumbering_global_old2new[eN_ebN];
        }
    }
  //4. now we need to build a new node ordering with better data locality for C0 finite elements
  //otherwise we could just grab the nodes on the subdomain and not worry about ownership
  //in the long run it wouldn't be bad to do a global repartition of faces and edges for mixed hybrid
  //and non-conforming finite elements
  //A node belongs to the lowest rank that owns an element containing it. The
  //rank-ordered mask chain that used to decide it marked the unmarked nodes of
  //each rank's owned elements; the minimum over all elements is the same rule.
  std::vector<int> nodeOwner = nodeOwnersFromElementOwners(mesh,elementOwner,size);
  if (!everyEntityOwned(nodeOwner,size,"node"))
    return 1;
  set<int> nodes_subdomain_owned;
  for (int nN=0;nN<mesh.nNodes_global;nN++)
    if (nodeOwner[nN] == rank)
      nodes_subdomain_owned.insert(nodes_subdomain_owned.end(),nN);

  //Now as with elements build a global node numbering, sort node
  //based information, and renumber references to node numbers
  valarray<int> nNodes_subdomain_new, nodeOffsets_new;
  std::vector<int> nodeNumbering_old2new_global, nodeNumbering_global_new2old;
  numberByOwner(nodeOwner,size,nNodes_subdomain_new,nodeOffsets_new,
                nodeNumbering_old2new_global,nodeNumbering_global_new2old);
  for (int eN=0;eN < mesh.nElements_global; eN++)
    {
      int nN_old;
      for (int nN=0;nN < mesh.nNodes_element; nN++)
        {
          nN_old = elementNodesArray_new[eN*mesh.nNodes_element+nN];
          elementNodesArray_new[eN*mesh.nNodes_element+nN] = nodeNumbering_old2new_global[nN_old];
        }
    }
  //mwf postpone until after element boundary renumbering
  //     for (int i=0;i<mesh.nElementBoundaries_global*mesh.nNodes_elementBoundary;i++)
  //       {
  //         int nN_old = mesh.elementBoundaryNodesArray[i];
  //         elementBoundaryNodesArray_new[i] = nodeNumbering_old2new_global[nN_old];
  //       }
  for (int i=0;i<mesh.nEdges_global*2;i++)
    {
      int nN_old = mesh.edgeNodesArray[i];
      edgeNodesArray_new[i] = nodeNumbering_old2new_global[nN_old];
    }
  valarray<int> nodeStarArray_new(mesh.nodeStarOffsets[mesh.nNodes_global]);
  for (int i=0;i<mesh.nodeStarOffsets[mesh.nNodes_global];i++)
    {
      int nN_old = mesh.nodeStarArray[i];
      nodeStarArray_new[i] = nodeNumbering_old2new_global[nN_old];
    }
  valarray<double> nodeArray_new(mesh.nNodes_global*3);
  valarray<int>  nodeMaterialTypes_new(mesh.nNodes_global);
  for (int nN=0;nN<mesh.nNodes_global;nN++)
    {
      int nN_new = nodeNumbering_old2new_global[nN];
      nodeArray_new[nN_new*3+0] = mesh.nodeArray[nN*3+0];
      nodeArray_new[nN_new*3+1] = mesh.nodeArray[nN*3+1];
      nodeArray_new[nN_new*3+2] = mesh.nodeArray[nN*3+2];
      nodeMaterialTypes_new[nN_new] = mesh.nodeMaterialTypes[nN];
    }

  if (timer) timer->mark("nodes");
  //4b. repeat process to build global face numbering: a face belongs to the
  //lowest rank that owns an element it bounds
  std::vector<int> elementBoundaryOwner = elementBoundaryOwnersFromElementOwners(mesh,elementOwner,size);
  if (!everyEntityOwned(elementBoundaryOwner,size,"element boundary"))
    return 1;
  //Now as with elements and nodes build a global face numbering
  //resetting the face based information is a little different since much of this is currently built below based
  //on the element and node information
  //
  valarray<int> nElementBoundaries_subdomain_new, elementBoundaryOffsets_new;
  std::vector<int> elementBoundaryNumbering_old2new_global, elementBoundaryNumbering_global_new2old;
  numberByOwner(elementBoundaryOwner,size,nElementBoundaries_subdomain_new,elementBoundaryOffsets_new,
                elementBoundaryNumbering_old2new_global,elementBoundaryNumbering_global_new2old);
  for (int eN=0;eN < mesh.nElements_global; eN++)
    {
      int ebN_old;
      for (int ebN=0;ebN < mesh.nElementBoundaries_element; ebN++)
        {
          ebN_old = elementBoundariesArray_new[eN*mesh.nElementBoundaries_element+ebN];
          elementBoundariesArray_new[eN*mesh.nElementBoundaries_element+ebN] = elementBoundaryNumbering_old2new_global[ebN_old];
        }
    }
  //redo
  for (int ebN=0;ebN<mesh.nElementBoundaries_global;ebN++)
    {
      int ebN_old=elementBoundaryNumbering_global_new2old[ebN];
      for (int nN=0; nN<mesh.nNodes_elementBoundary; nN++)
        {
          int nN_old = mesh.elementBoundaryNodesArray[ebN_old*mesh.nNodes_elementBoundary+nN];
          elementBoundaryNodesArray_new[ebN*mesh.nNodes_elementBoundary+nN]=nodeNumbering_old2new_global[nN_old];
        }
      int eN_L_old = mesh.elementBoundaryElementsArray[ebN_old*2+0],
        eN_R_old = mesh.elementBoundaryElementsArray[ebN_old*2+1];
      elementBoundaryElementsArray_new[ebN*2+0] = elementNumbering_global_old2new[eN_L_old];
      if(eN_R_old >= 0)
        elementBoundaryElementsArray_new[ebN*2+1] = elementNumbering_global_old2new[eN_R_old];

      elementBoundaryMaterialTypes_new[ebN] = mesh.elementBoundaryMaterialTypes[ebN_old];
    }
  //     //mwf debug check constistency
  //     for (int ebN=0; ebN<mesh.nElementBoundaries_global;ebN++)
  //       {
  //    int eN_left=elementBoundaryElementsArray_new[ebN*2+0];
  //    int eN_right=elementBoundaryElementsArray_new[ebN*2+1];
  //    assert(eN_left>=0);
  //    bool found_ebN_left=false;
  //    for (int ebN_element=0; ebN_element<mesh.nElementBoundaries_element; ebN_element++)
  //      {
  //        if (ebN == elementBoundariesArray_new[eN_left*mesh.nElementBoundaries_element+ebN_element])
  //          {
  //            assert(ebN_element==elementBoundaryLocalElementBoundariesArray_new[ebN*2+0]);
  //            found_ebN_left=true;
  //          }
  //      }
  //    assert(found_ebN_left);
  //    if (eN_right>=0)
  //      {
  //        bool found_ebN_right=false;
  //        for (int ebN_element=0; ebN_element<mesh.nElementBoundaries_element; ebN_element++)
  //          {
  //            if (ebN == elementBoundariesArray_new[eN_right*mesh.nElementBoundaries_element+ebN_element])
  //              {
  //                assert(ebN_element==elementBoundaryLocalElementBoundariesArray_new[ebN*2+1]);
  //                found_ebN_right=true;
  //              }
  //          }
  //        assert(found_ebN_right);
  //      }
  //       }

  //do not renumber interior and exterior element boundary arrays yet

  //write partitioned mesh to view with "showme"
  //     std::ofstream nodeout("mesh.node"),eleout("mesh.ele"),partout("mesh.part");
  //     eleout<<mesh.nElements_global<<" 3 0"<<std::endl;
  //     partout<<mesh.nElements_global<<"\t"<<size<<std::endl;
  //     for (int eN=0;eN<mesh.nElements_global;eN++)
  //       {
  //    partout<<(eN+1)<<"\t"<<(1+epart[elementNumbering_global_new2old[eN]])<<std::endl;
  //    //partout<<(eN+1)<<"\t"<<(1+epart[eN])<<std::endl;
  //    eleout<<(eN+1)<<"\t"<<(1+elementNodesArray_new[eN*3+0])
  //          <<"\t"<<(1+elementNodesArray_new[eN*3+1])
  //          <<"\t"<<(1+elementNodesArray_new[eN*3+2])
  //          <<std::endl;
  //       }
  //     nodeout<<mesh.nNodes_global<<" 2 0 0"<<std::endl;
  //     for (int nN=0;nN<mesh.nNodes_global;nN++)
  //       {
  //    nodeout<<(nN+1)<<"\t"<<nodeArray_new[nN*3+0]<<"\t"<<nodeArray_new[nN*3+1]<<std::endl;
  //       }
  //     eleout.close();
  //     partout.close();

  if (timer) timer->mark("element_boundaries");
  //4c. Build global edge numbering as well
  // ownership is determined by the edges on owned elements: an edge belongs to
  // the lowest rank that owns an element with both its nodes, and those
  // elements are all in the node star of either one
  std::vector<int> edgeOwner = edgeOwnersFromElementOwners(mesh,elementOwner,size);
  if (!everyEntityOwned(edgeOwner,size,"edge"))
    return 1;
  valarray<int> nEdges_subdomain_new, edgeOffsets_new;
  std::vector<int> edgeNumbering_old2new_global, edgeNumbering_global_new2old;
  numberByOwner(edgeOwner,size,nEdges_subdomain_new,edgeOffsets_new,
                edgeNumbering_old2new_global,edgeNumbering_global_new2old);

  //create  array with (new edge) --> (new node 0, new node 1)
  //and map from (new node 0, new node 1) --> (new global edge)
  valarray<int> edgeNodesArray_newNodesAndEdges(2*mesh.nEdges_global);
  map<NodeTuple<2>,int > nodesEdgeMap_global_new;
  for (int ig = 0; ig < mesh.nEdges_global; ig++)
    {
      int nodes[2];
      const int edge_old = edgeNumbering_global_new2old[ig];
      nodes[0] = edgeNodesArray_new[edge_old*2+0];
      nodes[1] = edgeNodesArray_new[edge_old*2+1];
      NodeTuple<2> et(nodes);
      edgeNodesArray_newNodesAndEdges[ig*2+0] = edgeNodesArray_new[edge_old*2+0];
      edgeNodesArray_newNodesAndEdges[ig*2+1] = edgeNodesArray_new[edge_old*2+1];
      assert(nodesEdgeMap_global_new.find(et) == nodesEdgeMap_global_new.end());
      nodesEdgeMap_global_new[et] = ig;
    }

  if (timer) timer->mark("edges");
  //5. At this point we have new, renumbered and sorted the global element, and node based information, and
  //we have it all on each processor so we can add overlap
  set<int> elements_overlap,nodes_overlap,elementBoundaries_overlap,edges_overlap;
  for(int eN=elementOffsets_new[rank];eN<elementOffsets_new[rank+1];eN++)
    {
      for (int nN=0;nN<mesh.nNodes_element;nN++)
        {
          int nN_global = elementNodesArray_new[eN*mesh.nNodes_element+nN];
          if (nN_global < nodeOffsets_new[rank] || nN_global >= nodeOffsets_new[rank+1])
            nodes_overlap.insert(nN_global);
        }
      for (int ebN=0; ebN<mesh.nElementBoundaries_element;ebN++)
        {
          int ebN_global=elementBoundariesArray_new[eN*mesh.nElementBoundaries_element+ebN];
          if (ebN_global < elementBoundaryOffsets_new[rank] || ebN_global >= elementBoundaryOffsets_new[rank+1])
            elementBoundaries_overlap.insert(ebN_global);
        }
      for (int nN0=0;nN0<mesh.nNodes_element;nN0++)//assume all nodes in element may be edges
        for (int nN1=nN0+1; nN1<mesh.nNodes_element;nN1++)
          {
            int nodes[2];
            nodes[0] = elementNodesArray_new[eN*mesh.nNodes_element+nN0];
            nodes[1] = elementNodesArray_new[eN*mesh.nNodes_element+nN1];
            NodeTuple<2> et(nodes);
            const int nN0_global = elementNodesArray_new[eN*mesh.nNodes_element+nN0];
            const int nN1_global = elementNodesArray_new[eN*mesh.nNodes_element+nN1];
            if (nN0_global < nodeOffsets_new[rank] || nN0_global >= nodeOffsets_new[rank+1])
              assert(nodes_overlap.find(nN0_global) != nodes_overlap.end());
            if (nN1_global < nodeOffsets_new[rank] || nN1_global >= nodeOffsets_new[rank+1])
              assert(nodes_overlap.find(nN1_global) != nodes_overlap.end());
            if (nodesEdgeMap_global_new.find(et) != nodesEdgeMap_global_new.end())
              {
                const int edge_global = nodesEdgeMap_global_new[et];
                if (edge_global < edgeOffsets_new[rank] || edge_global >= edgeOffsets_new[rank+1])
                  edges_overlap.insert(edge_global);
              }
          }//edges
    }

  if (nElements_overlap > 0)
    {
      //get all elements in the node stars and their nodes
      // for (int nN_new=nodeOffsets_new[rank]; nN_new < nodeOffsets_new[rank+1]; nN_new++)
      //   {
      //     int nN = nodeNumbering_global_new2old[nN_new];
      //     for (int offset =mesh.nodeElementOffsets[nN];offset<mesh.nodeElementOffsets[nN+1];offset++)
      //       {
      //         int eN = mesh.nodeElementsArray[offset];
      //         int eN_new = elementNumbering_global_old2new[eN];
      //         if (eN_new < elementOffsets_new[rank] or eN_new >= elementOffsets_new[rank+1])
      //           {
      //             elements_overlap.insert(eN_new);
      //             for (int nN_element=0;nN_element<mesh.nNodes_element;nN_element++)
      //               {
      //                 int nN_global = elementNodesArray_new[eN_new*mesh.nNodes_element+nN_element];
      //                 if (nN_global < nodeOffsets_new[rank] or nN_global >= nodeOffsets_new[rank+1])
      //                   nodes_overlap.insert(nN_global);
      //               }
      //           }
      //       }
      //   }
      //get all elements in the node stars of owned nodes and those elements' nodes and faces and edges
      for (set<int>::iterator nN=nodes_subdomain_owned.begin();nN != nodes_subdomain_owned.end();nN++)
        {
          for (int offset =mesh.nodeElementOffsets[*nN];offset<mesh.nodeElementOffsets[(*nN)+1];offset++)
            {
              int eN = mesh.nodeElementsArray[offset];
              int eN_new = elementNumbering_global_old2new[eN];
              if (eN_new < elementOffsets_new[rank] or eN_new >= elementOffsets_new[rank+1])
                {
                  elements_overlap.insert(eN_new);
                  for (int nN_element=0;nN_element<mesh.nNodes_element;nN_element++)
                    {
                      int nN_global = elementNodesArray_new[eN_new*mesh.nNodes_element+nN_element];
                      if (nN_global < nodeOffsets_new[rank] or nN_global >= nodeOffsets_new[rank+1])
                        nodes_overlap.insert(nN_global);
                    }
                  for (int ebN=0; ebN<mesh.nElementBoundaries_element;ebN++)
                    {
                      int ebN_global=elementBoundariesArray_new[eN_new*mesh.nElementBoundaries_element+ebN];
                      if (ebN_global < elementBoundaryOffsets_new[rank] || ebN_global >= elementBoundaryOffsets_new[rank+1])
                        elementBoundaries_overlap.insert(ebN_global);
                    }
                  //edges too
                  for (int nN0=0;nN0<mesh.nNodes_element;nN0++)
                    for (int nN1=nN0+1; nN1<mesh.nNodes_element;nN1++)
                      {
                        int nodes[2];
                        nodes[0] = elementNodesArray_new[eN_new*mesh.nNodes_element+nN0];
                        nodes[1] = elementNodesArray_new[eN_new*mesh.nNodes_element+nN1];
                        NodeTuple<2> et(nodes);
                        const int nN0_global = elementNodesArray_new[eN_new*mesh.nNodes_element+nN0];
                        const int nN1_global = elementNodesArray_new[eN_new*mesh.nNodes_element+nN1];
                        if (nodesEdgeMap_global_new.find(et) != nodesEdgeMap_global_new.end())
                          {
                            const int edge_global = nodesEdgeMap_global_new[et];
                            if (edge_global < edgeOffsets_new[rank] || edge_global >= edgeOffsets_new[rank+1])
                              edges_overlap.insert(edge_global);
                          }
                      }//edges
                }
            }
        }
      //get all the element neighbors
      for(int eN=elementOffsets_new[rank];eN<elementOffsets_new[rank+1];eN++)
        for(int ebN=0;ebN<mesh.nElementBoundaries_element;ebN++)
          {
            int eN_ebN = elementNeighborsArray_new[eN*mesh.nElementBoundaries_element+ebN];
            if (eN_ebN >= 0 &&
                (eN_ebN < elementOffsets_new[rank] || eN_ebN >= elementOffsets_new[rank+1]))
              {
                elements_overlap.insert(eN_ebN);
                for (int nN=0;nN<mesh.nNodes_element;nN++)
                  {
                    int nN_global = elementNodesArray_new[eN_ebN*mesh.nNodes_element+nN];
                    if (nN_global < nodeOffsets_new[rank] || nN_global >= nodeOffsets_new[rank+1])
                      nodes_overlap.insert(nN_global);
                  }
                for (int ebN_element=0; ebN_element<mesh.nElementBoundaries_element;ebN_element++)
                  {
                    int ebN_global=elementBoundariesArray_new[eN_ebN*mesh.nElementBoundaries_element+ebN_element];
                    if (ebN_global < elementBoundaryOffsets_new[rank] || ebN_global >= elementBoundaryOffsets_new[rank+1])
                      elementBoundaries_overlap.insert(ebN_global);
                  }
                //edges too
                for (int nN0=0;nN0<mesh.nNodes_element;nN0++)
                  for (int nN1=nN0+1; nN1<mesh.nNodes_element;nN1++)
                    {
                      int nodes[2];
                      nodes[0] = elementNodesArray_new[eN_ebN*mesh.nNodes_element+nN0];
                      nodes[1] = elementNodesArray_new[eN_ebN*mesh.nNodes_element+nN1];
                      NodeTuple<2> et(nodes);
                      const int nN0_global = elementNodesArray_new[eN_ebN*mesh.nNodes_element+nN0];
                      const int nN1_global = elementNodesArray_new[eN_ebN*mesh.nNodes_element+nN1];
                      bool foundEdge = false;
                      const int edge_global = nodesEdgeMap_global_new[et];
                      if (edge_global < edgeOffsets_new[rank] || edge_global >= edgeOffsets_new[rank+1])
                        edges_overlap.insert(edge_global);
                    }//edges
              }
          }
    }
  for (int layer=1;layer<nElements_overlap;layer++)
    {
      for (set<int>::iterator eN_p=elements_overlap.begin();eN_p != elements_overlap.end();eN_p++)
        {
          int eN_global = *eN_p;
          for(int ebN=0;ebN<mesh.nElementBoundaries_element;ebN++)
            {
              int eN_ebN = elementNeighborsArray_new[eN_global*mesh.nElementBoundaries_element+ebN];
              if (eN_ebN >= 0 &&
                  (eN_ebN < elementOffsets_new[rank] || eN_ebN >= elementOffsets_new[rank+1]))
                {
                  elements_overlap.insert(eN_ebN);
                  for (int nN=0;nN<mesh.nNodes_element;nN++)
                    {
                      int nN_global = elementNodesArray_new[eN_ebN*mesh.nNodes_element+nN];
                      if (nN_global < nodeOffsets_new[rank] || nN_global >= nodeOffsets_new[rank+1])
                        nodes_overlap.insert(nN_global);
                    }
                  for (int ebN_element=0; ebN_element<mesh.nElementBoundaries_element;ebN_element++)
                    {
                      int ebN_global=elementBoundariesArray_new[eN_ebN*mesh.nElementBoundaries_element+ebN_element];
                      if (ebN_global < elementBoundaryOffsets_new[rank] || ebN_global >= elementBoundaryOffsets_new[rank+1])
                        elementBoundaries_overlap.insert(ebN_global);
                    }
                  //edges too
                  for (int nN0=0;nN0<mesh.nNodes_element;nN0++)
                    for (int nN1=nN0+1; nN1<mesh.nNodes_element;nN1++)
                      {
                        int nodes[2];
                        nodes[0] = elementNodesArray_new[eN_ebN*mesh.nNodes_element+nN0];
                        nodes[1] = elementNodesArray_new[eN_ebN*mesh.nNodes_element+nN1];
                        NodeTuple<2> et(nodes);
                        const int nN0_global = elementNodesArray_new[eN_ebN*mesh.nNodes_element+nN0];
                        const int nN1_global = elementNodesArray_new[eN_ebN*mesh.nNodes_element+nN1];
                        bool foundEdge = false;
                        if (nodesEdgeMap_global_new.find(et) != nodesEdgeMap_global_new.end())
                          {
                            const int edge_global = nodesEdgeMap_global_new[et];
                            if (edge_global < edgeOffsets_new[rank] || edge_global >= edgeOffsets_new[rank+1])
                              edges_overlap.insert(edge_global);
                          }
                      }//edges
                }
            }
        }
    }
  if (timer) timer->mark("overlap");
  //
  //6. Now build subdomain mesh
  //
  //set what we know
  if (mesh.subdomainp == NULL)
    mesh.subdomainp = new Mesh();
  mesh.subdomainp->nElements_global = nElements_subdomain_new[rank] + elements_overlap.size();
  mesh.subdomainp->nNodes_global = nNodes_subdomain_new[rank] + nodes_overlap.size();
  //better be true ... check below
  mesh.subdomainp->nElementBoundaries_global = nElementBoundaries_subdomain_new[rank]+elementBoundaries_overlap.size();
  mesh.subdomainp->nEdges_global = nEdges_subdomain_new[rank] + edges_overlap.size();

  mesh.subdomainp->nNodes_element = mesh.nNodes_element;
  mesh.subdomainp->nNodes_elementBoundary = mesh.nNodes_elementBoundary;
  mesh.subdomainp->nElementBoundaries_element = mesh.nElementBoundaries_element;
  //load the elements, nodes. and elementBoundaries
  valarray<int> nodeNumbering_subdomain2global(mesh.subdomainp->nNodes_global);
  map<int,int> nodeNumbering_global2subdomain;
  mesh.subdomainp->nodeArray = new double[mesh.subdomainp->nNodes_global*3];
  mesh.subdomainp->nodeMaterialTypes = new int[mesh.subdomainp->nNodes_global];
  for(int nN=0;nN<nNodes_subdomain_new[rank];nN++)
    {
      int nN_global = nN + nodeOffsets_new[rank];
      nodeNumbering_subdomain2global[nN] = nN_global;
      nodeNumbering_global2subdomain[nN_global] = nN;
      mesh.subdomainp->nodeArray[nN*3+0] = nodeArray_new[nN_global*3+0];
      mesh.subdomainp->nodeArray[nN*3+1] = nodeArray_new[nN_global*3+1];
      mesh.subdomainp->nodeArray[nN*3+2] = nodeArray_new[nN_global*3+2];
      mesh.subdomainp->nodeMaterialTypes[nN]= nodeMaterialTypes_new[nN_global];
    }
  //note: sets in C++ are sorted so the overlap is laid out in
  //contiguous chunks corresponding to the partitions
  set<int>::iterator nN_p=nodes_overlap.begin();
  for(int nN=nNodes_subdomain_new[rank];nN < nNodes_subdomain_new[rank] + int(nodes_overlap.size()); nN++)
    {
      int nN_global = *nN_p++;
      nodeNumbering_subdomain2global[nN] = nN_global;
      nodeNumbering_global2subdomain[nN_global] = nN;
      mesh.subdomainp->nodeArray[nN*3+0] = nodeArray_new[nN_global*3+0];
      mesh.subdomainp->nodeArray[nN*3+1] = nodeArray_new[nN_global*3+1];
      mesh.subdomainp->nodeArray[nN*3+2] = nodeArray_new[nN_global*3+2];
      mesh.subdomainp->nodeMaterialTypes[nN]= nodeMaterialTypes_new[nN_global];
    }

  //see what we must/can set for element boundaries here
  valarray<int> elementBoundaryNumbering_subdomain2global(mesh.subdomainp->nElementBoundaries_global);
  map<int,int> elementBoundaryNumbering_global2subdomain;
  for (int ebN=0; ebN < nElementBoundaries_subdomain_new[rank]; ebN++)
    {
      int ebN_global = ebN + elementBoundaryOffsets_new[rank];
      elementBoundaryNumbering_subdomain2global[ebN]=ebN_global;
      elementBoundaryNumbering_global2subdomain[ebN_global] = ebN;
    }
  set<int>::iterator ebN_p = elementBoundaries_overlap.begin();
  for(int ebN=nElementBoundaries_subdomain_new[rank];ebN < nElementBoundaries_subdomain_new[rank] + int(elementBoundaries_overlap.size()); ebN++)
    {
      int ebN_global = *ebN_p++;
      elementBoundaryNumbering_subdomain2global[ebN] = ebN_global;
      elementBoundaryNumbering_global2subdomain[ebN_global] = ebN;
    }


  mesh.subdomainp->elementNodesArray = new int[mesh.subdomainp->nElements_global*mesh.subdomainp->nNodes_element];
  mesh.subdomainp->elementMaterialTypes = new int[mesh.subdomainp->nElements_global];
  //try to use elementBoundariesArray to set unique element boundary id below
  mesh.subdomainp->elementBoundariesArray =
    new int[mesh.subdomainp->nElements_global*mesh.subdomainp->nElementBoundaries_element];
  valarray<int> elementNumbering_subdomain2global(mesh.subdomainp->nElements_global);
  for (int eN=0;eN<nElements_subdomain_new[rank];eN++)
    {
      int eN_global = eN+elementOffsets_new[rank];
      elementNumbering_subdomain2global[eN] = eN_global;
      mesh.subdomainp->elementMaterialTypes[eN] = elementMaterialTypes_new[eN_global];
      for (int nN=0;nN<mesh.subdomainp->nNodes_element;nN++)
        {
          mesh.subdomainp->elementNodesArray[eN*mesh.subdomainp->nNodes_element+nN] =
            nodeNumbering_global2subdomain[elementNodesArray_new[eN_global*mesh.nNodes_element + nN]];
        }
      for (int ebN=0;ebN<mesh.subdomainp->nElementBoundaries_element;ebN++)
        mesh.subdomainp->elementBoundariesArray[eN*mesh.subdomainp->nElementBoundaries_element+ebN] =
          elementBoundaryNumbering_global2subdomain[elementBoundariesArray_new[eN_global*mesh.nElementBoundaries_element + ebN]];
    }
  set<int>::iterator eN_p=elements_overlap.begin();
  for(int eN=nElements_subdomain_new[rank];eN < nElements_subdomain_new[rank]+int(elements_overlap.size());eN++)
    {
      int eN_global = *eN_p++;
      mesh.subdomainp->elementMaterialTypes[eN] = elementMaterialTypes_new[eN_global];
      elementNumbering_subdomain2global[eN] = eN_global;
      for (int nN=0;nN<mesh.subdomainp->nNodes_element;nN++)
        {
          mesh.subdomainp->elementNodesArray[eN*mesh.subdomainp->nNodes_element+nN] =
            nodeNumbering_global2subdomain[elementNodesArray_new[eN_global*mesh.nNodes_element + nN]];
        }
      for (int ebN=0;ebN<mesh.subdomainp->nElementBoundaries_element;ebN++)
        mesh.subdomainp->elementBoundariesArray[eN*mesh.subdomainp->nElementBoundaries_element+ebN] =
          elementBoundaryNumbering_global2subdomain[elementBoundariesArray_new[eN_global*mesh.nElementBoundaries_element + ebN]];
    }

  //set edge information, main piece necessary for setting subdomain data structures consistently is a map
  //from (nN0,nN1) --> edge_subdomain
  valarray<int> edgeNumbering_subdomain2global(mesh.subdomainp->nEdges_global);
  mesh.subdomainp->edgeNodesArray = new int[mesh.subdomainp->nEdges_global*2];
  for (int i=0; i < nEdges_subdomain_new[rank]; i++)
    {
      const int ig = i+edgeOffsets_new[rank];
      const int nN0_global = edgeNodesArray_newNodesAndEdges[ig*2+0];
      const int nN1_global = edgeNodesArray_newNodesAndEdges[ig*2+1];
      //mwf todo double check can always count on having nodes on this processor
      assert(nodeNumbering_global2subdomain.find(nN0_global) != nodeNumbering_global2subdomain.end());
      assert(nodeNumbering_global2subdomain.find(nN1_global) != nodeNumbering_global2subdomain.end());
      const int nN0_subdomain = nodeNumbering_global2subdomain[nN0_global];
      const int nN1_subdomain = nodeNumbering_global2subdomain[nN1_global];
      mesh.subdomainp->edgeNodesArray[2*i+0]=nN0_subdomain;
      mesh.subdomainp->edgeNodesArray[2*i+1]=nN1_subdomain;
      edgeNumbering_subdomain2global[i] = ig;
    }
  set<int>::iterator edge_p = edges_overlap.begin();
  for (int i=nEdges_subdomain_new[rank]; i < nEdges_subdomain_new[rank] + int(edges_overlap.size()); i++)
    {
      const int ig =*edge_p++;
      const int nN0_global = edgeNodesArray_newNodesAndEdges[ig*2+0];
      const int nN1_global = edgeNodesArray_newNodesAndEdges[ig*2+1];
      //mwf todo make sure always have nodes for the edge on this processor
      const int nN0_subdomain = nodeNumbering_global2subdomain[nN0_global];
      const int nN1_subdomain = nodeNumbering_global2subdomain[nN1_global];
      mesh.subdomainp->edgeNodesArray[2*i+0]=nN0_subdomain;
      mesh.subdomainp->edgeNodesArray[2*i+1]=nN1_subdomain;
      edgeNumbering_subdomain2global[i] = ig;
    }
  //(the global mesh is still in its old edge numbering here; renumbered below)
  subdomainEdgeMaterialTypes(mesh, *mesh.subdomainp, edgeNumbering_subdomain2global, edgeNumbering_global_new2old);
  //now fill in rest of boundary information, etc
  mesh.subdomainp->px = mesh.px;
  mesh.subdomainp->py = mesh.py;
  mesh.subdomainp->pz = mesh.pz;

  if (mesh.px != 0)
    {
      //constructElementBoundaryElementsArrayWithGivenElementBoundaryNumbers_tetrahedron(*mesh.subdomainp);
      constructElementBoundaryElementsArrayWithGivenElementBoundaryAndEdgeNumbers_NURBS(*mesh.subdomainp);
      allocateGeometricInfo_NURBS(*mesh.subdomainp);
      computeGeometricInfo_NURBS(*mesh.subdomainp);

    }
  else if (mesh.subdomainp->nNodes_element == 2)
    {
      //constructElementBoundaryElementsArrayWithGivenElementBoundaryNumbers_edge(*mesh.subdomainp);
      constructElementBoundaryElementsArrayWithGivenElementBoundaryAndEdgeNumbers_edge(*mesh.subdomainp);
      allocateGeometricInfo_edge(*mesh.subdomainp);
      computeGeometricInfo_edge(*mesh.subdomainp);
    }

  else if (mesh.subdomainp->nNodes_element == 3)
    {
      //constructElementBoundaryElementsArrayWithGivenElementBoundaryNumbers_triangle(*mesh.subdomainp);
      constructElementBoundaryElementsArrayWithGivenElementBoundaryAndEdgeNumbers_triangle(*mesh.subdomainp);
      allocateGeometricInfo_triangle(*mesh.subdomainp);
      computeGeometricInfo_triangle(*mesh.subdomainp);
    }
  else if (mesh.subdomainp->nNodes_element == 4 && mesh.subdomainp->nNodes_elementBoundary == 2)
    {
      //constructElementBoundaryElementsArrayWithGivenElementBoundaryNumbers_tetrahedron(*mesh.subdomainp);
      constructElementBoundaryElementsArrayWithGivenElementBoundaryAndEdgeNumbers_quadrilateral(*mesh.subdomainp);
      allocateGeometricInfo_quadrilateral(*mesh.subdomainp);
      computeGeometricInfo_quadrilateral(*mesh.subdomainp);
    }
  else if (mesh.subdomainp->nNodes_element == 4)
    {
      //constructElementBoundaryElementsArrayWithGivenElementBoundaryNumbers_tetrahedron(*mesh.subdomainp);
      constructElementBoundaryElementsArrayWithGivenElementBoundaryAndEdgeNumbers_tetrahedron(*mesh.subdomainp);
      allocateGeometricInfo_tetrahedron(*mesh.subdomainp);
      computeGeometricInfo_tetrahedron(*mesh.subdomainp);
    }
  else if (mesh.subdomainp->nNodes_element == 8)
    {
      constructElementBoundaryElementsArrayWithGivenElementBoundaryAndEdgeNumbers_hexahedron(*mesh.subdomainp);
      allocateGeometricInfo_hexahedron(*mesh.subdomainp);
      computeGeometricInfo_hexahedron(*mesh.subdomainp);
    }
  else
    {
      assert(false);
    }
  if (mesh.elementBoundaryMaterialTypes != NULL)
    {
      assert(mesh.elementBoundariesArray != NULL);
      //todo need to check that local element boundary numbering for
      //element boundaries stays the same
      for (int eN=0;eN<mesh.subdomainp->nElements_global;eN++)
        {
          int eN_global_new = elementNumbering_subdomain2global[eN];
          int eN_global_old = elementNumbering_global_new2old[eN_global_new];
          for (int ebN_element = 0; ebN_element < mesh.nElementBoundaries_element; ebN_element++)
            {
              int ebN_global_old = mesh.elementBoundariesArray[eN_global_old*mesh.nElementBoundaries_element+ebN_element];
              int ebN_subdomain = mesh.subdomainp->elementBoundariesArray[eN*mesh.nElementBoundaries_element+ebN_element];
              mesh.subdomainp->elementBoundaryMaterialTypes[ebN_subdomain] = mesh.elementBoundaryMaterialTypes[ebN_global_old];
            }
        }
    }
  //now we've got the old mesh in the old ordering and the subdomain mesh in the new ordering
  //the first chunk of nodes and elements are the owned elements so we need to know how many of those there are
  //and the offset of the first one so we can compute subdomain2global
  mesh.nodeOffsets_subdomain_owned = new int[size+1];
  mesh.elementOffsets_subdomain_owned = new int[size+1];
  mesh.elementBoundaryOffsets_subdomain_owned = new int[size+1];
  mesh.edgeOffsets_subdomain_owned = new int[size+1];
  for (int sdN=0;sdN<size+1;sdN++)
    {
      mesh.nodeOffsets_subdomain_owned[sdN] = nodeOffsets_new[sdN];
      mesh.elementOffsets_subdomain_owned[sdN] = elementOffsets_new[sdN];
      mesh.elementBoundaryOffsets_subdomain_owned[sdN] = elementBoundaryOffsets_new[sdN];
      mesh.edgeOffsets_subdomain_owned[sdN] = edgeOffsets_new[sdN];
    }
  //we also need the subdomain 2 new global mappings
  mesh.nodeNumbering_subdomain2global = new int[mesh.subdomainp->nNodes_global];
  for (int nN=0;nN<mesh.subdomainp->nNodes_global;nN++)
    mesh.nodeNumbering_subdomain2global[nN] = nodeNumbering_subdomain2global[nN];
  mesh.elementNumbering_subdomain2global = new int[mesh.subdomainp->nElements_global];
  for (int eN=0;eN<mesh.subdomainp->nElements_global;eN++)
    mesh.elementNumbering_subdomain2global[eN] = elementNumbering_subdomain2global[eN];
  mesh.elementBoundaryNumbering_subdomain2global = new int[mesh.subdomainp->nElementBoundaries_global];
  for (int ebN=0;ebN<mesh.subdomainp->nElementBoundaries_global;ebN++)
    mesh.elementBoundaryNumbering_subdomain2global[ebN] = elementBoundaryNumbering_subdomain2global[ebN];
  mesh.edgeNumbering_subdomain2global = new int[mesh.subdomainp->nEdges_global];
  for (int i=0; i< mesh.subdomainp->nEdges_global; i++)
    mesh.edgeNumbering_subdomain2global[i] = edgeNumbering_subdomain2global[i];

  //
  //go ahead and renumber global mesh
  //
  //std::cout<<"nodeArray"<<std::endl;
  for (int i=0;i<mesh.nNodes_global*3;i++)
    mesh.nodeArray[i] = nodeArray_new[i];
  //std::cout<<"elementNodesArray"<<std::endl;
  for (int i=0;i<mesh.nElements_global*mesh.nNodes_element;i++)
    mesh.elementNodesArray[i] = elementNodesArray_new[i];
  //std::cout<<"elementBoundaryNodesArray"<<std::endl;
  for (int i=0;i<mesh.nElementBoundaries_global*mesh.nNodes_elementBoundary;i++)
    mesh.elementBoundaryNodesArray[i] = elementBoundaryNodesArray_new[i];
  //std::cout<<"edgeNodesArray"<<std::endl;
  for (int i=0;i<mesh.nEdges_global*2;i++)
    mesh.edgeNodesArray[i] = edgeNodesArray_newNodesAndEdges[i];
  if (mesh.edgeMaterialTypes != NULL)
    {
      std::vector<int> edgeMaterialTypes_old(mesh.edgeMaterialTypes, mesh.edgeMaterialTypes + mesh.nEdges_global);
      for (int ig = 0; ig < mesh.nEdges_global; ig++)
        mesh.edgeMaterialTypes[ig] = edgeMaterialTypes_old[edgeNumbering_global_new2old[ig]];
    }
  //std::cout<<"nodeStarArray"<<std::endl;
  for (int i=0;i<mesh.nodeStarOffsets[mesh.nNodes_global];i++)
    mesh.nodeStarArray[i] = nodeStarArray_new[i];
  //std::cout<<"elementNeighborsArray"<<std::endl;
  for (int i=0; i<mesh.nElements_global*mesh.nElementBoundaries_element; i++)
    mesh.elementNeighborsArray[i] = elementNeighborsArray_new[i];
  //std::cout<<"elementBoundariesArray"<<std::endl;
  for (int i=0; i<mesh.nElements_global*mesh.nElementBoundaries_element; i++)
    mesh.elementBoundariesArray[i] = elementBoundariesArray_new[i];
  //std::cout<<"elementBoundaryElementsArray"<<std::endl;
  for (int i=0; i<mesh.nElementBoundaries_global*2; i++)
    mesh.elementBoundaryElementsArray[i] = elementBoundaryElementsArray_new[i];
  //std::cout<<"elementBoundaryLocalElementBoundariesArray"<<std::endl;
  for (int i=0; i<mesh.nElementBoundaries_global*2; i++)
    mesh.elementBoundaryLocalElementBoundariesArray[i] = elementBoundaryLocalElementBoundariesArray_new[i];
  //
  //need material properties too
  //
  if (mesh.elementMaterialTypes != NULL)
    {
      //std::cout<<"elementMaterialTypes"<<std::endl;
      for (int i=0; i < mesh.nElements_global; i++)
        mesh.elementMaterialTypes[i] = elementMaterialTypes_new[i];
    }
  if (mesh.elementBoundaryMaterialTypes != NULL)
    {
      //std::cout<<"elementBoundaryMaterialTypes"<<std::endl;
      for (int i=0; i < mesh.nElementBoundaries_global; i++)
        mesh.elementBoundaryMaterialTypes[i] = elementBoundaryMaterialTypes_new[i];
    }


  if (timer) timer->mark("subdomain_mesh");
  return 0;
}

int buildQuadraticSubdomain2GlobalMappings_1d(const MPI_Comm& PROTEUS_COMM_WORLD, Mesh& mesh,
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
                                              double * lagrangeNodesArray)//location of nodes corresponding to dofs
{
  using namespace std;
  int ierr,size,rank;

  ierr = MPI_Comm_size(PROTEUS_COMM_WORLD,&size);
  ierr = MPI_Comm_rank(PROTEUS_COMM_WORLD,&rank);


  //In 1d the quadratic dofs can be associated with nodes and elements
  //assuming have ownership info and consistent local/global mappings for nodes, elements
  //build a mapping from local quadratic dofs to global quadratic dofs for petsc
  //assume a processor owns a dof if it owns that node or element
  assert(elementOffsets_subdomain_owned);
  assert(nodeOffsets_subdomain_owned);
  assert(elementNumbering_subdomain2global);
  assert(nodeNumbering_subdomain2global);
  assert(mesh.subdomainp);

  const int nNodes_owned    = nodeOffsets_subdomain_owned[rank+1]-nodeOffsets_subdomain_owned[rank];
  const int nElements_owned = elementOffsets_subdomain_owned[rank+1]-elementOffsets_subdomain_owned[rank];

  //start with a logical global ordering of dofs as
  //[global nodes, global elements]
  //want to create global numbering
  //[nodes proc 0,elements proc 0,nodes proc 1, elements proc 1,...]
  valarray<int> quadraticNumbering_new2old(nNodes_owned + nElements_owned);
  for (int nN = 0; nN < nNodes_owned; nN++)
    {
      int dof_global = nodeNumbering_subdomain2global[nN];
      quadraticNumbering_new2old[nN] = dof_global;
    }
  for (int eN = 0; eN < nElements_owned; eN++)
    {
      int dof_global = mesh.nNodes_global + elementNumbering_subdomain2global[eN];
      quadraticNumbering_new2old[nNodes_owned + eN] = dof_global;
    }

  //every rank's new numbering, gathered
  std::vector<int> quadraticNumbering_global_new2old =
    allGatherInts(PROTEUS_COMM_WORLD,std::begin(quadraticNumbering_new2old),nNodes_owned + nElements_owned);
  //get old 2 new mapping for dofs
  valarray<int> quadraticNumbering_old2new_global(mesh.nNodes_global + mesh.nElements_global);
  for (int id = 0; id < mesh.nNodes_global + mesh.nElements_global; id++)
    quadraticNumbering_old2new_global[quadraticNumbering_global_new2old[id]] = id;
  //   //mwf debug
  //   for (int id = 0; id < mesh.nNodes_global + mesh.nElements_global; id++)
  //     {
  //       std::cout<<" rank= "<<rank<<" build c0p2 mappings new2old["<<id<<"]= "<<quadraticNumbering_global_new2old[id]
  //           <<" old2new["<<quadraticNumbering_global_new2old[id]<<"]= "<<quadraticNumbering_old2new_global[quadraticNumbering_global_new2old[id]]
  //           <<std::endl;
  //     }
  assert(offsets_subdomain_owned);
  assert(subdomain2global);
  assert(subdomain_l2g);
  assert(lagrangeNodesArray);
  nDOF_all_processes = mesh.nNodes_global+mesh.nElements_global;
  nDOF_subdomain = mesh.subdomainp->nNodes_global+mesh.subdomainp->nElements_global;
  max_dof_neighbors = 2*mesh.max_nNodeNeighbors_node;

  for (int sdN=0; sdN < size+1; sdN++)
    offsets_subdomain_owned[sdN] = nodeOffsets_subdomain_owned[sdN]+elementOffsets_subdomain_owned[sdN];

  //loop through owned and ghost dofs build subdomain mapping by
  //going from old --> new
  int localOffset = 0;
  for (int nN = 0; nN < nNodes_owned; nN++)
    {
      int dof_global_old = nodeNumbering_subdomain2global[nN];
      int dof_global_new = quadraticNumbering_old2new_global[dof_global_old];
      subdomain2global[localOffset + nN] = dof_global_new;
      //mwf debug
      //std::cout<<" rank= "<<rank<<" nN= "<<nN<<" local dof= "<<localOffset+nN<<" nNodes_owned= "<<nNodes_owned<<" dof_old= "<<dof_global_old<<" new= "<<dof_global_new<<std::endl;
    }
  localOffset += nNodes_owned;
  for (int eN = 0; eN < nElements_owned; eN++)
    {
      int dof_global_old = mesh.nNodes_global + elementNumbering_subdomain2global[eN];
      int dof_global_new = quadraticNumbering_old2new_global[dof_global_old];
      subdomain2global[localOffset + eN] = dof_global_new;
      //mwf debug
      //std::cout<<" rank= "<<rank<<" eN= "<<eN<<" local dof= "<<localOffset+eN<<" nElements_owned= "<<nElements_owned<<" dof_old= "<<dof_global_old<<" new= "<<dof_global_new<<std::endl;
    }
  localOffset += nElements_owned;
  for (int nN = nNodes_owned; nN < mesh.subdomainp->nNodes_global; nN++)
    {
      int dof_global_old = nodeNumbering_subdomain2global[nN];
      int dof_global_new = quadraticNumbering_old2new_global[dof_global_old];
      subdomain2global[localOffset + nN -nNodes_owned] = dof_global_new;
      //mwf debug
      //std::cout<<" rank= "<<rank<<" nN= "<<nN<<" local dof= "<<localOffset+nN-nNodes_owned<<" nNodes_owned= "<<nNodes_owned<<" dof_old= "<<dof_global_old<<" new= "<<dof_global_new<<std::endl;
    }
  localOffset += mesh.subdomainp->nNodes_global - nNodes_owned;
  for (int eN = nElements_owned; eN < mesh.subdomainp->nElements_global; eN++)
    {
      int dof_global_old = mesh.nNodes_global + elementNumbering_subdomain2global[eN];
      int dof_global_new = quadraticNumbering_old2new_global[dof_global_old];
      subdomain2global[localOffset + eN-nElements_owned] = dof_global_new;
      //mwf debug
      //std::cout<<" rank= "<<rank<<" eN= "<<eN<<" local dof= "<<localOffset+eN-nElements_owned<<" nElements_owned= "<<nElements_owned<<" dof_old= "<<dof_global_old<<" new= "<<dof_global_new<<std::endl;
    }
  //setup local to global mapping on the subdomain for finite elements
  const int nDOF_element = 3;
  const int ghostNodeOffset = nNodes_owned + nElements_owned;
  const int ghostElementOffset = ghostNodeOffset + mesh.subdomainp->nNodes_global-nNodes_owned;

  //for lagrange nodes
  int nN_global_subdomain[2];
  for (int eN=0; eN < mesh.subdomainp->nElements_global; eN++)
    {
      for (int nN=0; nN < mesh.subdomainp->nNodes_element; nN++)
        {
          nN_global_subdomain[nN] = mesh.subdomainp->elementNodesArray[eN*mesh.subdomainp->nNodes_element + nN];
          if (nN_global_subdomain[nN] < nNodes_owned)
            subdomain_l2g[eN*nDOF_element+nN] = nN_global_subdomain[nN];
          else
            subdomain_l2g[eN*nDOF_element+nN] = nN_global_subdomain[nN]-nNodes_owned + ghostNodeOffset;
          //vertex dof
          for (int eI=0; eI < 3; eI++)
            lagrangeNodesArray[subdomain_l2g[eN*nDOF_element+nN]*3+eI] = mesh.subdomainp->nodeArray[3*nN_global_subdomain[nN]+eI];
        }
      if (eN < nElements_owned)
        subdomain_l2g[eN*nDOF_element+2] = nNodes_owned + eN;
      else
        subdomain_l2g[eN*nDOF_element+2] = eN - nElements_owned + ghostElementOffset;
      //vertex dof
      for (int eI=0; eI < 3; eI++)
        lagrangeNodesArray[subdomain_l2g[eN*nDOF_element+2]*3+eI] = 0.5*(mesh.subdomainp->nodeArray[3*nN_global_subdomain[0]+eI]+mesh.subdomainp->nodeArray[3*nN_global_subdomain[1]+eI]);
    }


  return 0;
}

int buildQuadraticSubdomain2GlobalMappings_2d(const MPI_Comm& PROTEUS_COMM_WORLD, Mesh& mesh,
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
                                              double * lagrangeNodesArray)//location of nodes corresponding to dofs
{
  using namespace std;
  int ierr,size,rank;

  ierr = MPI_Comm_size(PROTEUS_COMM_WORLD,&size);
  ierr = MPI_Comm_rank(PROTEUS_COMM_WORLD,&rank);

  //In 2d the quadratic dofs can be associated with nodes and element Boundaries
  //assuming have ownership info and consistent local/global mappings for nodes, elementBoundaries
  //build a mapping from local quadratic dofs to global quadratic dofs for petsc
  //assume a processor owns a dof if it owns that node or element
  assert(elementBoundaryOffsets_subdomain_owned);
  assert(nodeOffsets_subdomain_owned);
  assert(elementBoundaryNumbering_subdomain2global);
  assert(nodeNumbering_subdomain2global);
  assert(mesh.subdomainp);

  const int nNodes_owned    = nodeOffsets_subdomain_owned[rank+1]-nodeOffsets_subdomain_owned[rank];
  const int nElementBoundaries_owned = elementBoundaryOffsets_subdomain_owned[rank+1]-elementBoundaryOffsets_subdomain_owned[rank];
  const int nDOFs_owned = nNodes_owned + nElementBoundaries_owned;
  const int nDOFs_global= mesh.nNodes_global + mesh.nElementBoundaries_global;
  //start with a logical global ordering of dofs as
  //[global nodes, global elementBoundaries]
  //want to create global numbering
  //[nodes proc 0,elementBoundaries proc 0,nodes proc 1, elementBoundaries proc 1,...]
  valarray<int> quadraticNumbering_new2old(nDOFs_owned);
  for (int nN = 0; nN < nNodes_owned; nN++)
    {
      int dof_global = nodeNumbering_subdomain2global[nN];
      quadraticNumbering_new2old[nN] = dof_global;
    }
  for (int ebN = 0; ebN < nElementBoundaries_owned; ebN++)
    {
      int dof_global = mesh.nNodes_global + elementBoundaryNumbering_subdomain2global[ebN];
      quadraticNumbering_new2old[nNodes_owned + ebN] = dof_global;
    }

  //every rank's new numbering, gathered
  std::vector<int> quadraticNumbering_global_new2old =
    allGatherInts(PROTEUS_COMM_WORLD,std::begin(quadraticNumbering_new2old),nDOFs_owned);
  //get old 2 new mapping for dofs
  valarray<int> quadraticNumbering_old2new_global(nDOFs_global);
  for (int id = 0; id < nDOFs_global; id++)
    quadraticNumbering_old2new_global[quadraticNumbering_global_new2old[id]] = id;
  //mwf debug
  //for (int id = 0; id < nDOFs_global; id++)
  //  {
  //    std::cout<<" rank= "<<rank<<" build 2d c0p2 mappings new2old["<<id<<"]= "<<quadraticNumbering_global_new2old[id]
  //           <<" old2new["<<quadraticNumbering_global_new2old[id]<<"]= "<<quadraticNumbering_old2new_global[quadraticNumbering_global_new2old[id]]
  //           <<std::endl;
  //  }
  assert(offsets_subdomain_owned);
  assert(subdomain2global);
  assert(subdomain_l2g);
  assert(lagrangeNodesArray);
  for (int sdN=0; sdN < size+1; sdN++)
    offsets_subdomain_owned[sdN] = nodeOffsets_subdomain_owned[sdN]+elementBoundaryOffsets_subdomain_owned[sdN];
  nDOF_all_processes = mesh.nNodes_global+mesh.nElementBoundaries_global;
  nDOF_subdomain = mesh.subdomainp->nNodes_global+mesh.subdomainp->nElementBoundaries_global;
  max_dof_neighbors = 2*mesh.max_nNodeNeighbors_node;

  //loop through owned and ghost dofs build subdomain mapping by
  //going from old --> new
  int localOffset = 0;
  for (int nN = 0; nN < nNodes_owned; nN++)
    {
      int dof_global_old = nodeNumbering_subdomain2global[nN];
      int dof_global_new = quadraticNumbering_old2new_global[dof_global_old];
      subdomain2global[localOffset + nN] = dof_global_new;
      //mwf debug
      //std::cout<<" rank= "<<rank<<" nN= "<<nN<<" local dof= "<<localOffset+nN<<" nNodes_owned= "<<nNodes_owned<<" dof_old= "<<dof_global_old<<" new= "<<dof_global_new<<std::endl;
    }
  localOffset += nNodes_owned;
  for (int ebN = 0; ebN < nElementBoundaries_owned; ebN++)
    {
      int dof_global_old = mesh.nNodes_global + elementBoundaryNumbering_subdomain2global[ebN];
      int dof_global_new = quadraticNumbering_old2new_global[dof_global_old];
      subdomain2global[localOffset + ebN] = dof_global_new;
      //mwf debug
      //std::cout<<" rank= "<<rank<<" ebN= "<<ebN<<" local dof= "<<localOffset+ebN<<" nElementBoundaries_owned= "<<nElementBoundaries_owned
      //               <<" dof_old= "<<dof_global_old<<" new= "<<dof_global_new<<std::endl;
    }
  localOffset += nElementBoundaries_owned;
  for (int nN = nNodes_owned; nN < mesh.subdomainp->nNodes_global; nN++)
    {
      int dof_global_old = nodeNumbering_subdomain2global[nN];
      int dof_global_new = quadraticNumbering_old2new_global[dof_global_old];
      subdomain2global[localOffset + nN -nNodes_owned] = dof_global_new;
      //mwf debug
      //std::cout<<" rank= "<<rank<<" nN= "<<nN<<" local dof= "<<localOffset+nN-nNodes_owned<<" nNodes_owned= "<<nNodes_owned<<" dof_old= "<<dof_global_old<<" new= "<<dof_global_new<<std::endl;
    }
  localOffset += mesh.subdomainp->nNodes_global - nNodes_owned;
  for (int ebN = nElementBoundaries_owned; ebN < mesh.subdomainp->nElementBoundaries_global; ebN++)
    {
      int dof_global_old = mesh.nNodes_global + elementBoundaryNumbering_subdomain2global[ebN];
      int dof_global_new = quadraticNumbering_old2new_global[dof_global_old];
      subdomain2global[localOffset + ebN-nElementBoundaries_owned] = dof_global_new;
      //mwf debug
      //std::cout<<" rank= "<<rank<<" ebN= "<<ebN<<" local dof= "<<localOffset+ebN-nElementBoundaries_owned<<" nElementBoundaries_owned= "<<nElementBoundaries_owned<<" dof_old= "<<dof_global_old<<" new= "<<dof_global_new<<std::endl;
    }
  //setup local to global mapping on the subdomain for finite elements
  const int nDOF_element = 6;
  const int ghostNodeOffset = nNodes_owned + nElementBoundaries_owned;
  const int ghostElementBoundaryOffset = ghostNodeOffset + mesh.subdomainp->nNodes_global-nNodes_owned;

  //for lagrange nodes
  int nN_global_subdomain[3];
  for (int eN=0; eN < mesh.subdomainp->nElements_global; eN++)
    {
      for (int nN=0; nN < mesh.subdomainp->nNodes_element; nN++)
        {
          nN_global_subdomain[nN] = mesh.subdomainp->elementNodesArray[eN*mesh.subdomainp->nNodes_element + nN];
          if (nN_global_subdomain[nN] < nNodes_owned)
            subdomain_l2g[eN*nDOF_element+nN] = nN_global_subdomain[nN];
          else
            subdomain_l2g[eN*nDOF_element+nN] = nN_global_subdomain[nN]-nNodes_owned + ghostNodeOffset;
          //vertex dof
          for (int eI=0; eI < 3; eI++)
            lagrangeNodesArray[subdomain_l2g[eN*nDOF_element+nN]*3+eI] = mesh.subdomainp->nodeArray[3*nN_global_subdomain[nN]+eI];
        }
      for (int ebN=0; ebN < mesh.subdomainp->nElementBoundaries_element; ebN++)
        {
          //take into account numbering of edges according to
          //vertex they are across from
          int ebN_global = mesh.subdomainp->elementBoundariesArray[eN*mesh.subdomainp->nElementBoundaries_element+((ebN+2)%3)];
          if (ebN_global < nElementBoundaries_owned)
            subdomain_l2g[eN*nDOF_element+3+ebN] = nNodes_owned + ebN_global;
          else
            subdomain_l2g[eN*nDOF_element+3+ebN] = ebN_global - nElementBoundaries_owned + ghostElementBoundaryOffset;
          //center of edge dof
          for (int eI=0; eI < 3; eI++)
            lagrangeNodesArray[subdomain_l2g[eN*nDOF_element+3+ebN]*3+eI] = 0.5*(mesh.subdomainp->nodeArray[3*nN_global_subdomain[(ebN+0)%3]+eI]+mesh.subdomainp->nodeArray[3*nN_global_subdomain[(ebN+1)%3]+eI]);
        }
    }//eN


  return 0;
}


int buildQuadraticSubdomain2GlobalMappings_3d(const MPI_Comm& PROTEUS_COMM_WORLD, Mesh& mesh,
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
                                              double * lagrangeNodesArray)//location of nodes corresponding to dofs
{
  using namespace std;
  int ierr,size,rank;

  ierr = MPI_Comm_size(PROTEUS_COMM_WORLD,&size);
  ierr = MPI_Comm_rank(PROTEUS_COMM_WORLD,&rank);

  //In 2d the quadratic dofs can be associated with nodes and element Boundaries
  //assuming have ownership info and consistent local/global mappings for nodes, elementBoundaries
  //build a mapping from local quadratic dofs to global quadratic dofs for petsc
  //assume a processor owns a dof if it owns that node or element
  assert(edgeOffsets_subdomain_owned);
  assert(nodeOffsets_subdomain_owned);
  assert(edgeNumbering_subdomain2global);
  assert(nodeNumbering_subdomain2global);
  assert(mesh.subdomainp);

  const int nNodes_owned = nodeOffsets_subdomain_owned[rank+1]-nodeOffsets_subdomain_owned[rank];
  const int nEdges_owned = edgeOffsets_subdomain_owned[rank+1]-edgeOffsets_subdomain_owned[rank];
  const int nDOFs_owned = nNodes_owned + nEdges_owned;
  const int nDOFs_global= mesh.nNodes_global + mesh.nEdges_global;
  //start with a logical global ordering of dofs as
  //[global nodes, global edges]
  //want to create global numbering
  //[nodes proc 0,edges proc 0,nodes proc 1, edges proc 1,...]
  valarray<int> quadraticNumbering_new2old(nDOFs_owned);
  for (int nN = 0; nN < nNodes_owned; nN++)
    {
      int dof_global = nodeNumbering_subdomain2global[nN];
      quadraticNumbering_new2old[nN] = dof_global;
    }
  for (int i = 0; i < nEdges_owned; i++)
    {
      int dof_global = mesh.nNodes_global + edgeNumbering_subdomain2global[i];
      quadraticNumbering_new2old[nNodes_owned + i] = dof_global;
    }

  //every rank's new numbering, gathered
  std::vector<int> quadraticNumbering_global_new2old =
    allGatherInts(PROTEUS_COMM_WORLD,std::begin(quadraticNumbering_new2old),nDOFs_owned);
  //get old 2 new mapping for dofs
  valarray<int> quadraticNumbering_old2new_global(nDOFs_global);
  for (int id = 0; id < nDOFs_global; id++)
    quadraticNumbering_old2new_global[quadraticNumbering_global_new2old[id]] = id;
  //mwf debug
  //   if (rank == 0)
  //     {
  //       for (int id = 0; id < nDOFs_global; id++)
  //    {
  //      std::cout<<" rank= "<<rank<<" build 2d c0p2 mappings new2old["<<id<<"]= "<<quadraticNumbering_global_new2old[id]
  //               <<" old2new["<<quadraticNumbering_global_new2old[id]<<"]= "<<quadraticNumbering_old2new_global[quadraticNumbering_global_new2old[id]]
  //               <<std::endl;
  //    }
  //     }
  assert(offsets_subdomain_owned);
  assert(subdomain2global);
  assert(subdomain_l2g);
  assert(lagrangeNodesArray);
  for (int sdN=0; sdN < size+1; sdN++)
    offsets_subdomain_owned[sdN] = nodeOffsets_subdomain_owned[sdN]+edgeOffsets_subdomain_owned[sdN];
  nDOF_all_processes = mesh.nNodes_global+mesh.nEdges_global;
  nDOF_subdomain = mesh.subdomainp->nNodes_global+mesh.subdomainp->nEdges_global;
  max_dof_neighbors = 2*mesh.max_nNodeNeighbors_node;

  //loop through owned and ghost dofs build subdomain mapping by
  //going from old --> new
  int localOffset = 0;
  for (int nN = 0; nN < nNodes_owned; nN++)
    {
      int dof_global_old = nodeNumbering_subdomain2global[nN];
      int dof_global_new = quadraticNumbering_old2new_global[dof_global_old];
      subdomain2global[localOffset + nN] = dof_global_new;
      //mwf debug
      //      if (rank == 0)
      //        std::cout<<" rank= "<<rank<<" nN= "<<nN<<" local dof= "<<localOffset+nN<<" nNodes_owned= "<<nNodes_owned<<" dof_old= "<<dof_global_old<<" new= "<<dof_global_new<<std::endl;
    }
  localOffset += nNodes_owned;
  for (int i = 0; i < nEdges_owned; i++)
    {
      int dof_global_old = mesh.nNodes_global + edgeNumbering_subdomain2global[i];
      int dof_global_new = quadraticNumbering_old2new_global[dof_global_old];
      subdomain2global[localOffset + i] = dof_global_new;
      //mwf debug
      //       if (rank == 0)
      //          std::cout<<" rank= "<<rank<<" i= "<<i<<" local dof= "<<localOffset+i<<" nEdges_owned= "<<nEdges_owned
      //                   <<" dof_old= "<<dof_global_old<<" new= "<<dof_global_new<<std::endl;
    }
  localOffset += nEdges_owned;
  for (int nN = nNodes_owned; nN < mesh.subdomainp->nNodes_global; nN++)
    {
      int dof_global_old = nodeNumbering_subdomain2global[nN];
      int dof_global_new = quadraticNumbering_old2new_global[dof_global_old];
      subdomain2global[localOffset + nN -nNodes_owned] = dof_global_new;
      //mwf debug
      //       if (rank == 0)
      //        std::cout<<" rank= "<<rank<<" nN= "<<nN<<" local dof= "<<localOffset+nN-nNodes_owned<<" nNodes_owned= "<<nNodes_owned<<" dof_old= "<<dof_global_old<<" new= "<<dof_global_new<<std::endl;
    }
  localOffset += mesh.subdomainp->nNodes_global - nNodes_owned;
  for (int i = nEdges_owned; i < mesh.subdomainp->nEdges_global; i++)
    {
      int dof_global_old = mesh.nNodes_global + edgeNumbering_subdomain2global[i];
      int dof_global_new = quadraticNumbering_old2new_global[dof_global_old];
      subdomain2global[localOffset + i-nEdges_owned] = dof_global_new;
      //       //mwf debug
      //       if (rank == 0)
      //        std::cout<<" rank= "<<rank<<" i= "<<i<<" local dof= "<<localOffset+i-nEdges_owned<<" nEdges_owned= "<<nEdges_owned<<" dof_old= "<<dof_global_old<<" new= "<<dof_global_new<<std::endl;
    }
  //setup local to global mapping on the subdomain for finite elements
  const int nDOF_element = 10;
  const int ghostNodeOffset = nNodes_owned + nEdges_owned;
  const int ghostEdgeOffset = ghostNodeOffset + mesh.subdomainp->nNodes_global-nNodes_owned;
  //need mapping from nodes to edge to setup element based relationship
  map<NodeTuple<2>, int> nodesEdgeMap_subdomain;
  for (int i=0; i < mesh.subdomainp->nEdges_global; i++)
    {
      int nodes[2];
      nodes[0] = mesh.subdomainp->edgeNodesArray[2*i];
      nodes[1] = mesh.subdomainp->edgeNodesArray[2*i+1];
      NodeTuple<2> et(nodes);
      const int nN0 = mesh.subdomainp->edgeNodesArray[2*i];
      const int nN1 = mesh.subdomainp->edgeNodesArray[2*i+1];
      nodesEdgeMap_subdomain[et] = i;
    }
  for (int eN=0; eN < mesh.subdomainp->nElements_global; eN++)
    {
      int local_offset = 0;
      for (int nN=0; nN < mesh.subdomainp->nNodes_element; nN++)
        {
          //node_i --> unique subdomain node  0 <= i <= 3
          const int nN_global_subdomain = mesh.subdomainp->elementNodesArray[eN*mesh.subdomainp->nNodes_element + nN];
          //assign unique subdomain id based on |owned nodes|owned edges|ghost nodes|ghost edges|
          if (nN_global_subdomain < nNodes_owned)
            {
              subdomain_l2g[eN*nDOF_element + local_offset + nN] = nN_global_subdomain;
            }
          else
            {
              subdomain_l2g[eN*nDOF_element + local_offset + nN] = ghostNodeOffset + nN_global_subdomain - nNodes_owned;
            }
          //unique subdomain id --> unique cross processor id
          //subdomain2global[subdomain_l2g[eN*nDOF_element + local_offset + nN]] = nodeNumbering_subdomain2global[nN_global_subdomain];
          //mwf debug
          //      if (rank == 0)
          //        {
          //          std::cout<<"build loc2glob c0p2 3d eN= "<<eN<<" node= "<<nN<<" sub_dof= "<< subdomain_l2g[eN*nDOF_element + local_offset + nN]
          //                   <<" glob_dof= "<<subdomain2global[subdomain_l2g[eN*nDOF_element + local_offset + nN]]<<std::endl;
          //        }
          for (int eI=0; eI < 3; eI++)
            lagrangeNodesArray[subdomain_l2g[eN*nDOF_element + local_offset +nN]*3+eI] = mesh.subdomainp->nodeArray[nN_global_subdomain*3+eI];
        }
      local_offset += mesh.subdomainp->nNodes_element;
      //(node_i,node_{i+1}) --> unique subdomain edge, 0 <= i < 3
      for (int nN = 0; nN < mesh.subdomainp->nNodes_element-1; nN++)
        {
          const int nN_neig = (nN+1)%mesh.subdomainp->nNodes_element;
          const int nN_global_subdomain     = mesh.subdomainp->elementNodesArray[eN*mesh.subdomainp->nNodes_element + nN];
          const int nN_neig_global_subdomain= mesh.subdomainp->elementNodesArray[eN*mesh.subdomainp->nNodes_element + nN_neig];
          //unique edge subdomain id and global id
          int edge_subdomain = -1,edge_global=-1;
          //see if edge is (nN,nN_neig) or vice versa
          int nodes[2];
          nodes[0] = nN_global_subdomain;
          nodes[1] = nN_neig_global_subdomain;
          NodeTuple<2> et(nodes);
          edge_subdomain = nodesEdgeMap_subdomain[et];
          edge_global    = edgeNumbering_subdomain2global[edge_subdomain];
          assert(edge_subdomain >= 0 && edge_global >= 0);

          //assign unique subdomain id based on |owned nodes|owned edges|ghost nodes|ghost edges|
          if (edge_subdomain < nEdges_owned)
            subdomain_l2g[eN*nDOF_element + local_offset + nN] = nNodes_owned + edge_subdomain;
          else
            subdomain_l2g[eN*nDOF_element + local_offset + nN] = ghostEdgeOffset + edge_subdomain - nEdges_owned;
          //unique subdomain id --> unique cross processor id
          //set above could do here subdomain2global[subdomain_l2g[eN*nDOF_element + local_offset + nN]] = mesh.nNodes_global + edge_global;
          //mwf debug
          //      if (rank == 0)
          //        {
          //          std::cout<<"build loc2glob c0p2 3d eN= "<<eN<<" edge("<<nN<<","<<nN_neig<<") sub_dof= "<< subdomain_l2g[eN*nDOF_element + local_offset + nN]
          //                   <<" glob_dof= "<<subdomain2global[subdomain_l2g[eN*nDOF_element + local_offset + nN]]<<std::endl;
          //        }
          for (int eI=0; eI < 3; eI++)
            lagrangeNodesArray[subdomain_l2g[eN*nDOF_element+local_offset+nN]*3+eI] = 0.5*(mesh.subdomainp->nodeArray[nN_global_subdomain*3+eI]+
                                                                                           mesh.subdomainp->nodeArray[nN_neig_global_subdomain*3+eI]);
        }
      local_offset +=  mesh.subdomainp->nNodes_element-1;
      //(node_i,node_{i+2}) --> unique subdomain edge, 0 <= i < 2
      for (int nN = 0; nN < mesh.subdomainp->nNodes_element-2; nN++)
        {
          const int nN_neig = (nN+2)%mesh.subdomainp->nNodes_element;
          const int nN_global_subdomain     = mesh.subdomainp->elementNodesArray[eN*mesh.subdomainp->nNodes_element + nN];
          const int nN_neig_global_subdomain= mesh.subdomainp->elementNodesArray[eN*mesh.subdomainp->nNodes_element + nN_neig];
          //unique edge subdomain id and global id
          int edge_subdomain = -1, edge_global = -1;
          //see if edge is (nN,nN_neig) or vice versa
          int nodes[2];
          nodes[0] = nN_global_subdomain;
          nodes[1] = nN_neig_global_subdomain;
          NodeTuple<2> et(nodes);
          edge_subdomain = nodesEdgeMap_subdomain[et];
          edge_global =    edgeNumbering_subdomain2global[edge_subdomain];
          assert(edge_subdomain >= 0 && edge_global >= 0);
          //assign unique subdomain id based on |owned nodes|owned edges|ghost nodes|ghost edges|
          if (edge_subdomain < nEdges_owned)
            subdomain_l2g[eN*nDOF_element + local_offset + nN] = nNodes_owned +  edge_subdomain;
          else
            subdomain_l2g[eN*nDOF_element + local_offset + nN] = ghostEdgeOffset + edge_subdomain - nEdges_owned;
          //unique subdomain id --> unique cross processor id
          //subdomain2global[subdomain_l2g[eN*nDOF_element + local_offset + nN]] = mesh.nNodes_global + edge_global;
          //mwf debug
          //      if (rank == 0)
          //        {
          //          std::cout<<"build loc2glob c0p2 3d eN= "<<eN<<" edge("<<nN<<","<<nN_neig<<") sub_dof= "<< subdomain_l2g[eN*nDOF_element + local_offset + nN]
          //                   <<" glob_dof= "<<subdomain2global[subdomain_l2g[eN*nDOF_element + local_offset + nN]]<<std::endl;
          //        }
          for (int eI=0; eI < 3; eI++)
            lagrangeNodesArray[subdomain_l2g[eN*nDOF_element+local_offset+nN]*3+eI] = 0.5*(mesh.subdomainp->nodeArray[nN_global_subdomain*3+eI]+
                                                                                           mesh.subdomainp->nodeArray[nN_neig_global_subdomain*3+eI]);
        }
      local_offset += mesh.subdomainp->nNodes_element-2;
      //(node_i,node_{i+3}) --> unique subdomain edge, 0 = i
      for (int nN = 0; nN < mesh.subdomainp->nNodes_element-3; nN++)
        {
          const int nN_neig = (nN+3)%mesh.subdomainp->nNodes_element;
          const int nN_global_subdomain     = mesh.subdomainp->elementNodesArray[eN*mesh.subdomainp->nNodes_element + nN];
          const int nN_neig_global_subdomain= mesh.subdomainp->elementNodesArray[eN*mesh.subdomainp->nNodes_element + nN_neig];
          //unique edge subdomain id and global id
          int edge_subdomain = -1, edge_global = -1;
          //see if edge is (nN,nN_neig) or vice versa
          int nodes[2];
          nodes[0] = nN_global_subdomain;
          nodes[1] = nN_neig_global_subdomain;
          NodeTuple<2> et(nodes);
          edge_subdomain = nodesEdgeMap_subdomain[et];
          edge_global    = edgeNumbering_subdomain2global[edge_subdomain];
          assert(edge_subdomain >= 0 && edge_global >= 0);
          //assign unique subdomain id based on |owned nodes|owned edges|ghost nodes|ghost edges|
          if (edge_subdomain < nEdges_owned)
            subdomain_l2g[eN*nDOF_element + local_offset + nN] = nNodes_owned +  edge_subdomain;
          else
            subdomain_l2g[eN*nDOF_element + local_offset + nN] = ghostEdgeOffset + edge_subdomain - nEdges_owned;
          //unique subdomain id --> unique cross processor id
          //subdomain2global[subdomain_l2g[eN*nDOF_element + local_offset + nN]] = mesh.nNodes_global +  edge_global;
          //mwf debug
          //      if (rank == 0)
          //        {
          //          std::cout<<"build loc2glob c0p2 3d eN= "<<eN<<" edge("<<nN<<","<<nN_neig<<") sub_dof= "<< subdomain_l2g[eN*nDOF_element + local_offset + nN]
          //                   <<" glob_dof= "<<subdomain2global[subdomain_l2g[eN*nDOF_element + local_offset + nN]]<<std::endl;
          //        }
          for (int eI=0; eI < 3; eI++)
            lagrangeNodesArray[subdomain_l2g[eN*nDOF_element+local_offset+nN]*3+eI] = 0.5*(mesh.subdomainp->nodeArray[nN_global_subdomain*3+eI]+
                                                                                           mesh.subdomainp->nodeArray[nN_neig_global_subdomain*3+eI]);
        }

    }//eN


  return 0;
}

int buildQuadraticCubeSubdomain2GlobalMappings_3d(const MPI_Comm& PROTEUS_COMM_WORLD, Mesh& mesh,
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
                                                  double * lagrangeNodesArray)//location of nodes corresponding to dofs
{
  using namespace std;
  int ierr,size,rank;

  ierr = MPI_Comm_size(PROTEUS_COMM_WORLD,&size);
  ierr = MPI_Comm_rank(PROTEUS_COMM_WORLD,&rank);

  //In 2d the quadratic dofs can be associated with nodes and element Boundaries
  //assuming have ownership info and consistent local/global mappings for nodes, elementBoundaries
  //build a mapping from local quadratic dofs to global quadratic dofs for petsc
  //assume a processor owns a dof if it owns that node or element
  assert(edgeOffsets_subdomain_owned);
  assert(nodeOffsets_subdomain_owned);
  assert(edgeNumbering_subdomain2global);
  assert(nodeNumbering_subdomain2global);
  assert(mesh.subdomainp);

  const int *elementBoundaryOffsets_subdomain_owned=mesh.elementBoundaryOffsets_subdomain_owned;
  const int *elementOffsets_subdomain_owned=mesh.elementOffsets_subdomain_owned;
  const int *elementBoundaryNumbering_subdomain2global=mesh.elementBoundaryNumbering_subdomain2global;
  const int *elementNumbering_subdomain2global=mesh.elementNumbering_subdomain2global;

  const int nNodes_owned      = nodeOffsets_subdomain_owned[rank+1]-nodeOffsets_subdomain_owned[rank];
  const int nEdges_owned      = edgeOffsets_subdomain_owned[rank+1]-edgeOffsets_subdomain_owned[rank];
  const int nBoundaries_owned = elementBoundaryOffsets_subdomain_owned[rank+1]-elementBoundaryOffsets_subdomain_owned[rank];
  const int nElements_owned   = elementOffsets_subdomain_owned[rank+1]-elementOffsets_subdomain_owned[rank];

  const int nDOFs_owned = nNodes_owned + nEdges_owned + nBoundaries_owned + nElements_owned;
  const int nDOFs_global= mesh.nNodes_global + mesh.nEdges_global + mesh.nElementBoundaries_global + mesh.nElements_global;
  //start with a logical global ordering of dofs as
  //[global nodes, global edges]
  //want to create global numbering
  //[nodes proc 0,edges proc 0,nodes proc 1, edges proc 1,...]
  valarray<int> quadraticNumbering_new2old(nDOFs_owned);
  for (int nN = 0; nN < nNodes_owned; nN++)
    {
      int dof_global = nodeNumbering_subdomain2global[nN];
      quadraticNumbering_new2old[nN] = dof_global;
    }
  for (int i = 0; i < nEdges_owned; i++)
    {
      int dof_global = mesh.nNodes_global + edgeNumbering_subdomain2global[i];
      quadraticNumbering_new2old[nNodes_owned + i] = dof_global;
    }

  //-------------------
  for (int i = 0; i < nBoundaries_owned; i++)
    {
      int dof_global =  mesh.nNodes_global + mesh.nEdges_global + elementBoundaryNumbering_subdomain2global[i];
      quadraticNumbering_new2old[nNodes_owned + nEdges_owned + i] = dof_global;
    }
  for (int i = 0; i < nElements_owned; i++)
    {
      int dof_global = mesh.nNodes_global + mesh.nEdges_global + mesh.nElementBoundaries_global  + elementNumbering_subdomain2global[i];
      quadraticNumbering_new2old[nNodes_owned + nEdges_owned + nBoundaries_owned + i] = dof_global;
    }
  //-------------------


  //every rank's new numbering, gathered
  std::vector<int> quadraticNumbering_global_new2old =
    allGatherInts(PROTEUS_COMM_WORLD,std::begin(quadraticNumbering_new2old),nDOFs_owned);
  //get old 2 new mapping for dofs
  valarray<int> quadraticNumbering_old2new_global(nDOFs_global);
  for (int id = 0; id < nDOFs_global; id++)
    quadraticNumbering_old2new_global[quadraticNumbering_global_new2old[id]] = id;
  //mwf debug
  //   if (rank == 0)
  //     {
  //       for (int id = 0; id < nDOFs_global; id++)
  //    {
  //      std::cout<<" rank= "<<rank<<" build 2d c0p2 mappings new2old["<<id<<"]= "<<quadraticNumbering_global_new2old[id]
  //               <<" old2new["<<quadraticNumbering_global_new2old[id]<<"]= "<<quadraticNumbering_old2new_global[quadraticNumbering_global_new2old[id]]
  //               <<std::endl;
  //    }
  //     }
  assert(offsets_subdomain_owned);
  assert(subdomain2global);
  assert(subdomain_l2g);
  assert(lagrangeNodesArray);
  for (int sdN=0; sdN < size+1; sdN++)
    offsets_subdomain_owned[sdN] = nodeOffsets_subdomain_owned[sdN]+edgeOffsets_subdomain_owned[sdN]+elementBoundaryOffsets_subdomain_owned[sdN]+elementOffsets_subdomain_owned[sdN];
  nDOF_all_processes = mesh.nNodes_global+mesh.nEdges_global+ mesh.nElementBoundaries_global+mesh.nElements_global;
  nDOF_subdomain = mesh.subdomainp->nNodes_global+mesh.subdomainp->nEdges_global+mesh.subdomainp->nElementBoundaries_global+mesh.subdomainp->nElements_global;
  max_dof_neighbors = 2*mesh.max_nNodeNeighbors_node;//8*27?

  //loop through owned and ghost dofs build subdomain mapping by
  //going from old --> new
  int localOffset = 0;
  for (int nN = 0; nN < nNodes_owned; nN++)
    {
      int dof_global_old = nodeNumbering_subdomain2global[nN];
      int dof_global_new = quadraticNumbering_old2new_global[dof_global_old];
      subdomain2global[localOffset + nN] = dof_global_new;
      //mwf debug
      //      if (rank == 0)
      //        std::cout<<" rank= "<<rank<<" nN= "<<nN<<" local dof= "<<localOffset+nN<<" nNodes_owned= "<<nNodes_owned<<" dof_old= "<<dof_global_old<<" new= "<<dof_global_new<<std::endl;
    }

  localOffset += nNodes_owned;
  for (int i = 0; i < nEdges_owned; i++)
    {
      int dof_global_old = mesh.nNodes_global + edgeNumbering_subdomain2global[i];
      int dof_global_new = quadraticNumbering_old2new_global[dof_global_old];
      subdomain2global[localOffset + i] = dof_global_new;
      //mwf debug
      //       if (rank == 0)
      //          std::cout<<" rank= "<<rank<<" i= "<<i<<" local dof= "<<localOffset+i<<" nEdges_owned= "<<nEdges_owned
      //                   <<" dof_old= "<<dof_global_old<<" new= "<<dof_global_new<<std::endl;
    }


  localOffset += nEdges_owned;
  for (int i = 0; i < nBoundaries_owned; i++)
    {
      int dof_global_old = mesh.nNodes_global + mesh.nEdges_global +  elementBoundaryNumbering_subdomain2global[i];
      int dof_global_new = quadraticNumbering_old2new_global[dof_global_old];
      subdomain2global[localOffset + i] = dof_global_new;
      //mwf debug
      //       if (rank == 0)
      //          std::cout<<" rank= "<<rank<<" i= "<<i<<" local dof= "<<localOffset+i<<" nEdges_owned= "<<nEdges_owned
      //                   <<" dof_old= "<<dof_global_old<<" new= "<<dof_global_new<<std::endl;
    }

  localOffset += nBoundaries_owned;
  for (int i = 0; i < nElements_owned; i++)
    {
      int dof_global_old = mesh.nNodes_global + mesh.nEdges_global +  mesh.nElementBoundaries_global + elementNumbering_subdomain2global[i];
      int dof_global_new = quadraticNumbering_old2new_global[dof_global_old];
      subdomain2global[localOffset + i] = dof_global_new;
      //mwf debug
      //       if (rank == 0)
      //          std::cout<<" rank= "<<rank<<" i= "<<i<<" local dof= "<<localOffset+i<<" nEdges_owned= "<<nEdges_owned
      //                   <<" dof_old= "<<dof_global_old<<" new= "<<dof_global_new<<std::endl;
    }

  localOffset += nElements_owned;
  for (int nN = nNodes_owned; nN < mesh.subdomainp->nNodes_global; nN++)
    {
      int dof_global_old = nodeNumbering_subdomain2global[nN];
      int dof_global_new = quadraticNumbering_old2new_global[dof_global_old];
      subdomain2global[localOffset + nN - nNodes_owned] = dof_global_new;
      //mwf debug
      //       if (rank == 0)
      //        std::cout<<" rank= "<<rank<<" nN= "<<nN<<" local dof= "<<localOffset+nN-nNodes_owned<<" nNodes_owned= "<<nNodes_owned<<" dof_old= "<<dof_global_old<<" new= "<<dof_global_new<<std::endl;
    }

  localOffset += mesh.subdomainp->nNodes_global - nNodes_owned;
  for (int i = nEdges_owned; i < mesh.subdomainp->nEdges_global; i++)
    {
      int dof_global_old = mesh.nNodes_global + edgeNumbering_subdomain2global[i];
      int dof_global_new = quadraticNumbering_old2new_global[dof_global_old];
      subdomain2global[localOffset + i - nEdges_owned] = dof_global_new;
      //       //mwf debug
      //       if (rank == 0)
      //        std::cout<<" rank= "<<rank<<" i= "<<i<<" local dof= "<<localOffset+i-nEdges_owned<<" nEdges_owned= "<<nEdges_owned<<" dof_old= "<<dof_global_old<<" new= "<<dof_global_new<<std::endl;
    }

  localOffset += mesh.subdomainp->nEdges_global - nEdges_owned;
  for (int i = nBoundaries_owned; i < mesh.subdomainp->nElementBoundaries_global; i++)
    {
      int dof_global_old = mesh.nNodes_global + mesh.nEdges_global + elementBoundaryNumbering_subdomain2global[i];
      int dof_global_new = quadraticNumbering_old2new_global[dof_global_old];
      subdomain2global[localOffset + i - nBoundaries_owned] = dof_global_new;
      //       //mwf debug
      //       if (rank == 0)
      //        std::cout<<" rank= "<<rank<<" i= "<<i<<" local dof= "<<localOffset+i-nEdges_owned<<" nEdges_owned= "<<nEdges_owned<<" dof_old= "<<dof_global_old<<" new= "<<dof_global_new<<std::endl;
    }

  localOffset += mesh.subdomainp->nElementBoundaries_global - nBoundaries_owned;
  for (int i = nElements_owned; i < mesh.subdomainp->nElements_global; i++)
    {
      int dof_global_old = mesh.nNodes_global + mesh.nEdges_global + mesh.nElementBoundaries_global + elementNumbering_subdomain2global[i];
      int dof_global_new = quadraticNumbering_old2new_global[dof_global_old];
      subdomain2global[localOffset + i - nElements_owned] = dof_global_new;
      //       //mwf debug
      //       if (rank == 0)
      //        std::cout<<" rank= "<<rank<<" i= "<<i<<" local dof= "<<localOffset+i-nEdges_owned<<" nEdges_owned= "<<nEdges_owned<<" dof_old= "<<dof_global_old<<" new= "<<dof_global_new<<std::endl;
    }

  //setup local to global mapping on the subdomain for finite elements
  const int nDOF_element = 27;
  const int ghostNodeOffset = nNodes_owned + nEdges_owned + nBoundaries_owned + nElements_owned;
  const int ghostEdgeOffset = ghostNodeOffset + mesh.subdomainp->nNodes_global-nNodes_owned;
  const int ghostBoundaryOffset = ghostEdgeOffset + mesh.subdomainp->nEdges_global-nEdges_owned;
  const int ghostElementOffset = ghostBoundaryOffset + mesh.subdomainp->nElementBoundaries_global-nBoundaries_owned;

  int ledge[12][2] = {{0,1},{1,2},{2,3},{3,0},
                      {0,4},{1,5},{2,6},{3,7},
                      {4,5},{5,6},{6,7},{7,4}};

  int nEdges_element = 12;

  int lface[6][4] = {{0,1,2,3},
                     {0,1,5,4},
                     {1,2,6,5},
                     {2,3,7,6},
                     {3,0,4,7},
                     {4,5,6,7}};

  assert(mesh.subdomainp->nElementBoundaries_element == 6);
  //need mapping from nodes to edge to setup element based relationship
  map<NodeTuple<2>, int> nodesEdgeMap_subdomain;
  for (int i=0; i < mesh.subdomainp->nEdges_global; i++)
    {
      int nodes[2];
      nodes[0] = mesh.subdomainp->edgeNodesArray[2*i+0];
      nodes[1] = mesh.subdomainp->edgeNodesArray[2*i+1];
      NodeTuple<2> nt(nodes);
      nodesEdgeMap_subdomain[nt] = i;
    }

  map<NodeTuple<4>, int> nodesBoundaryMap_subdomain;
  for (int i=0; i < mesh.subdomainp->nElementBoundaries_global; i++)
    {

      int nodes[4];
      nodes[0] = mesh.subdomainp->elementBoundaryNodesArray[i*4+0];
      nodes[1] = mesh.subdomainp->elementBoundaryNodesArray[i*4+1];
      nodes[2] = mesh.subdomainp->elementBoundaryNodesArray[i*4+2];
      nodes[3] = mesh.subdomainp->elementBoundaryNodesArray[i*4+3];
      NodeTuple<4> ebt(nodes);
      assert(nodesBoundaryMap_subdomain.find(ebt) == nodesBoundaryMap_subdomain.end());
      nodesBoundaryMap_subdomain[ebt] = i;
    }

  for (int eN=0; eN < mesh.subdomainp->nElements_global; eN++)
    {
      int local_offset = 0;
      for (int nN=0; nN < mesh.subdomainp->nNodes_element; nN++)
        {
          //node_i --> unique subdomain node  0 <= i <= 3
          const int nN_global_subdomain = mesh.subdomainp->elementNodesArray[eN*mesh.subdomainp->nNodes_element + nN];
          //assign unique subdomain id based on |owned nodes|owned edges|ghost nodes|ghost edges|
          if (nN_global_subdomain < nNodes_owned)
            {
              subdomain_l2g[eN*nDOF_element + local_offset + nN] = nN_global_subdomain;
            }
          else
            {
              subdomain_l2g[eN*nDOF_element + local_offset + nN] = ghostNodeOffset + nN_global_subdomain - nNodes_owned;
            }
          //unique subdomain id --> unique cross processor id
          //subdomain2global[subdomain_l2g[eN*nDOF_element + local_offset + nN]] = nodeNumbering_subdomain2global[nN_global_subdomain];
          //mwf debug
          //      if (rank == 0)
          //        {
          //          std::cout<<"build loc2glob c0p2 3d eN= "<<eN<<" node= "<<nN<<" sub_dof= "<< subdomain_l2g[eN*nDOF_element + local_offset + nN]
          //                   <<" glob_dof= "<<subdomain2global[subdomain_l2g[eN*nDOF_element + local_offset + nN]]<<std::endl;
          //        }
          for (int eI=0; eI < 3; eI++)
            lagrangeNodesArray[subdomain_l2g[eN*nDOF_element + local_offset +nN]*3+eI] = mesh.subdomainp->nodeArray[nN_global_subdomain*3+eI];
        }
      local_offset += mesh.subdomainp->nNodes_element;
      for (int nN = 0; nN < nEdges_element; nN++)
        {
          int nodes[2];
          nodes[0] = mesh.subdomainp->elementNodesArray[eN*mesh.subdomainp->nNodes_element + ledge[nN][0]];
          nodes[1] = mesh.subdomainp->elementNodesArray[eN*mesh.subdomainp->nNodes_element + ledge[nN][1]];
          NodeTuple<2> nt(nodes);
          int edge_subdomain = nodesEdgeMap_subdomain[nt];
          int edge_global    = edgeNumbering_subdomain2global[edge_subdomain];
          assert(edge_subdomain >= 0 && edge_global >= 0);
          //assign unique subdomain id based on |owned nodes|owned edges|ghost nodes|ghost edges|
          if (edge_subdomain < nEdges_owned)
            subdomain_l2g[eN*nDOF_element + local_offset + nN] = nNodes_owned + edge_subdomain;
          else
            subdomain_l2g[eN*nDOF_element + local_offset + nN] = ghostEdgeOffset + edge_subdomain - nEdges_owned;
          //unique subdomain id --> unique cross processor id
          //set above could do here subdomain2global[subdomain_l2g[eN*nDOF_element + local_offset + nN]] = mesh.nNodes_global + edge_global;
          //mwf debug
          //      if (rank == 0)
          //        {
          //          std::cout<<"build loc2glob c0p2 3d eN= "<<eN<<" edge("<<nN<<","<<nN_neig<<") sub_dof= "<< subdomain_l2g[eN*nDOF_element + local_offset + nN]
          //                   <<" glob_dof= "<<subdomain2global[subdomain_l2g[eN*nDOF_element + local_offset + nN]]<<std::endl;
          //        }
          for (int eI=0; eI < 3; eI++)
            lagrangeNodesArray[subdomain_l2g[eN*nDOF_element+local_offset+nN]*3+eI] = 0.5*(mesh.subdomainp->nodeArray[nodes[0]*3+eI]+
                                                                                           mesh.subdomainp->nodeArray[nodes[1]*3+eI]);
        }

      local_offset += nEdges_element;
      for (int nN = 0; nN <  mesh.subdomainp->nElementBoundaries_element; nN++)
        {

          int nodes[4];
          nodes[0] = mesh.subdomainp->elementNodesArray[eN*mesh.subdomainp->nNodes_element+lface[nN][0]];
          nodes[1] = mesh.subdomainp->elementNodesArray[eN*mesh.subdomainp->nNodes_element+lface[nN][1]];
          nodes[2] = mesh.subdomainp->elementNodesArray[eN*mesh.subdomainp->nNodes_element+lface[nN][2]];
          nodes[3] = mesh.subdomainp->elementNodesArray[eN*mesh.subdomainp->nNodes_element+lface[nN][3]];
          NodeTuple<4> ebt(nodes);


          int boundary_subdomain = nodesBoundaryMap_subdomain[ebt];
          int boundary_global    = elementBoundaryNumbering_subdomain2global[boundary_subdomain];

          assert(boundary_subdomain >= 0 && boundary_global >= 0);

          //assign unique subdomain id based on |owned nodes|owned edges|ghost nodes|ghost edges|
          if (boundary_subdomain < nBoundaries_owned)
            subdomain_l2g[eN*nDOF_element + local_offset + nN] = nNodes_owned + nEdges_owned + boundary_subdomain;
          else
            subdomain_l2g[eN*nDOF_element + local_offset + nN] = ghostBoundaryOffset + boundary_subdomain - nBoundaries_owned;
          //unique subdomain id --> unique cross processor id
          //set above could do here subdomain2global[subdomain_l2g[eN*nDOF_element + local_offset + nN]] = mesh.nNodes_global + edge_global;
          //mwf debug
          //      if (rank == 0)
          //        {
          //          std::cout<<"build loc2glob c0p2 3d eN= "<<eN<<" edge("<<nN<<","<<nN_neig<<") sub_dof= "<< subdomain_l2g[eN*nDOF_element + local_offset + nN]
          //                   <<" glob_dof= "<<subdomain2global[subdomain_l2g[eN*nDOF_element + local_offset + nN]]<<std::endl;
          //        }
          for (int eI=0; eI < 3; eI++)
            lagrangeNodesArray[subdomain_l2g[eN*nDOF_element+local_offset+nN]*3+eI] = 0.25*(mesh.subdomainp->nodeArray[nodes[0]*3+eI]+
                                                                                            mesh.subdomainp->nodeArray[nodes[1]*3+eI]+
                                                                                            mesh.subdomainp->nodeArray[nodes[2]*3+eI]+
                                                                                            mesh.subdomainp->nodeArray[nodes[3]*3+eI]);
        }

      local_offset += mesh.subdomainp->nElementBoundaries_element;

      // Interior node!!!

      if (eN < nElements_owned)
        subdomain_l2g[eN*nDOF_element + local_offset] = nNodes_owned + nEdges_owned + nBoundaries_owned + eN;
      else
        subdomain_l2g[eN*nDOF_element + local_offset] = ghostElementOffset + eN - nElements_owned;
      //unique subdomain id --> unique cross processor id
      //set above could do here subdomain2global[subdomain_l2g[eN*nDOF_element + local_offset + nN]] = mesh.nNodes_global + edge_global;
      //mwf debug
      //          if (rank == 0)
      //            {
      //              std::cout<<"build loc2glob c0p2 3d eN= "<<eN<<" edge("<<nN<<","<<nN_neig<<") sub_dof= "<< subdomain_l2g[eN*nDOF_element + local_offset + nN]
      //                       <<" glob_dof= "<<subdomain2global[subdomain_l2g[eN*nDOF_element + local_offset + nN]]<<std::endl;
      //            }
      for (int eI=0; eI < 3; eI++)
        lagrangeNodesArray[subdomain_l2g[eN*nDOF_element+local_offset]*3+eI] = 0.125*(mesh.subdomainp->nodeArray[mesh.subdomainp->elementNodesArray[eN*mesh.subdomainp->nNodes_element + 0]*3+eI]+
                                                                                      mesh.subdomainp->nodeArray[mesh.subdomainp->elementNodesArray[eN*mesh.subdomainp->nNodes_element + 1]*3+eI]+
                                                                                      mesh.subdomainp->nodeArray[mesh.subdomainp->elementNodesArray[eN*mesh.subdomainp->nNodes_element + 2]*3+eI]+
                                                                                      mesh.subdomainp->nodeArray[mesh.subdomainp->elementNodesArray[eN*mesh.subdomainp->nNodes_element + 3]*3+eI]+
                                                                                      mesh.subdomainp->nodeArray[mesh.subdomainp->elementNodesArray[eN*mesh.subdomainp->nNodes_element + 4]*3+eI]+
                                                                                      mesh.subdomainp->nodeArray[mesh.subdomainp->elementNodesArray[eN*mesh.subdomainp->nNodes_element + 5]*3+eI]+
                                                                                      mesh.subdomainp->nodeArray[mesh.subdomainp->elementNodesArray[eN*mesh.subdomainp->nNodes_element + 6]*3+eI]+
                                                                                      mesh.subdomainp->nodeArray[mesh.subdomainp->elementNodesArray[eN*mesh.subdomainp->nNodes_element + 7]*3+eI]);


    }//eN


  return 0;
}





int buildDiscontinuousGalerkinSubdomain2GlobalMappings(const MPI_Comm& PROTEUS_COMM_WORLD, Mesh& mesh,
                                                       const int *elementOffsets_subdomain_owned,
                                                       const int *elementNumbering_subdomain2global,
                                                       int nDOF_element,
                                                       int& nDOF_all_processes,//total number of dofs in whole domain
                                                       int& nDOF_subdomain,//total number of dofs in sub-domain
                                                       int& max_dof_neighbors,//maximum number of neighbors for connectivity of dofs
                                                       int *offsets_subdomain_owned, //starting point of local dofs on each processor (nProcs+1)
                                                       int * subdomain_l2g, //local to global dof mapping on subdomain
                                                       int* subdomain2global)//subdomain dof to global (parallel) numbering

{
  using namespace std;
  int ierr,size,rank;
  ierr = MPI_Comm_size(PROTEUS_COMM_WORLD,&size);
  ierr = MPI_Comm_rank(PROTEUS_COMM_WORLD,&rank);

  //DG dofs stored element wise
  //[...,eN_0,eN_1,..,eN_ndof_local,...]
  //assume a processor owns a dof if it owns that element
  assert(elementOffsets_subdomain_owned);
  assert(elementNumbering_subdomain2global);
  assert(mesh.subdomainp);

  const int nElements_owned = elementOffsets_subdomain_owned[rank+1]-elementOffsets_subdomain_owned[rank];

  assert(offsets_subdomain_owned);
  assert(subdomain2global);
  assert(subdomain_l2g);
  nDOF_all_processes = mesh.nElements_global*nDOF_element;
  nDOF_subdomain = mesh.subdomainp->nElements_global*nDOF_element;
  max_dof_neighbors = mesh.nElementBoundaries_element*nDOF_element;

  for (int sdN=0; sdN < size+1; sdN++)
    offsets_subdomain_owned[sdN] = elementOffsets_subdomain_owned[sdN]*nDOF_element;

  //loop through owned and ghost dofs build subdomain mapping
  for (int eN = 0; eN < mesh.subdomainp->nElements_global; eN++)
    {
      for (int i = 0; i < nDOF_element; i++)
        {
          subdomain2global[eN*nDOF_element + i] = elementNumbering_subdomain2global[eN]*nDOF_element + i;
          //mwf debug
          //std::cout<<" rank= "<<rank<<" eN= "<<eN<<" local dof= "<<localOffset+eN<<" nElements_owned= "<<nElements_owned<<" dof_old= "<<dof_global_old<<" new= "<<dof_global_new<<std::endl;
          subdomain_l2g[eN*nDOF_element+i] = eN*nDOF_element + i;
        }
    }

  return 0;
}
}
