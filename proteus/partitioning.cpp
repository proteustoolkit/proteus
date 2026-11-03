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
        std::cerr<<"partitioning: "<<what<<" "<<i<<" was reached by no rank"<<std::endl;
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
        std::cerr<<caller<<": "<<what<<" "<<i<<" is assigned to rank "<<owner[i]
                 <<" but there are "<<size<<" ranks"<<std::endl;
        MPI_Abort(comm,1);
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
//todo add overlap for element based partitions
int partitionElementsOriginal(const MPI_Comm& PROTEUS_COMM_WORLD, Mesh& mesh, int nElements_overlap)
{
  using namespace std;
  int ierr,size,rank;

  ierr = MPI_Comm_size(PROTEUS_COMM_WORLD,&size);
  ierr = MPI_Comm_rank(PROTEUS_COMM_WORLD,&rank);

  //Contents
  //
  //1. Partition the elements in the "default" partition (contiguous
  //chunks in given ordering)
  //
  //2. Partition the elementNeighbors based on this partition
  //
  //3. Pass to Parmetis to build a better partition of the elements
  //
  //4. Tag a subset of the nodes on the subdomain elements as owned
  //using a mark and pass approach.**
  //
  //5. Extract the nodes in the
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
  valarray<int> elementOffsets_old(size+1);
  elementOffsets_old[0] = 0;
  for(int sdN=0;sdN<size;sdN++)
    elementOffsets_old[sdN+1] = elementOffsets_old[sdN] +
      int(mesh.nElements_global)/size +
      (int(mesh.nElements_global)%size > sdN);

  //2. Extract subdomain element adjacency information could read
  //only the required portion from a file
  int nElements_subdomain = (elementOffsets_old[rank+1] - elementOffsets_old[rank]);
  PetscInt *elementNeighborsOffsets_subdomain,*elementNeighbors_subdomain,*weights_subdomain;
  PetscMalloc(sizeof(PetscInt)*(nElements_subdomain+1),&elementNeighborsOffsets_subdomain);
  PetscMalloc(sizeof(PetscInt)*(nElements_subdomain*mesh.nElementBoundaries_element),&elementNeighbors_subdomain);
  PetscMalloc(sizeof(PetscInt)*(nElements_subdomain*mesh.nElementBoundaries_element),&weights_subdomain);
  //this wastes a little space
  elementNeighborsOffsets_subdomain[0] = 0;
  for (int eN=0,offset=0; eN < nElements_subdomain; eN++)
    {
      int eN_global = elementOffsets_old[rank] + eN;
      int offsetStart=offset;
      for (int ebN=0; ebN< mesh.nElementBoundaries_element;ebN++)
        {
          int eN_neighbor_global = mesh.elementNeighborsArray[eN_global*mesh.nElementBoundaries_element + ebN];
          if (eN_neighbor_global >= 0 )
            elementNeighbors_subdomain[offset++] = eN_neighbor_global;
        }
      elementNeighborsOffsets_subdomain[eN+1]=offset;
      sort(&elementNeighbors_subdomain[offsetStart],&elementNeighbors_subdomain[offset]);
      int weight = (elementNeighborsOffsets_subdomain[eN+1] - elementNeighborsOffsets_subdomain[eN]);
      for (int k=elementNeighborsOffsets_subdomain[eN];k < elementNeighborsOffsets_subdomain[eN+1];k++)
        weights_subdomain[k] = weight;
      // for (int o = offsetStart; o < offset; o++)
      //   {
      //     std::cout<<elementNeighbors_subdomain[o]<<'\t';
      //   }
      // std::cout<<std::endl;
    }
  //3. Generate the  new partitiong using PETSc, this is done in parallel using parmetis
  Mat petscAdjacency;
  //     MatCreateMPIAdj(PROTEUS_COMM_WORLD,
  //                     nElements_subdomain, mesh.nElements_global,
  //                     &elementNeighborsOffsets_subdomain[0], &elementNeighbors_subdomain[0],
  //                     &weights_subdomain[0],
  //                     &petscAdjacency);
  ierr = MatCreateMPIAdj(PROTEUS_COMM_WORLD,
                         nElements_subdomain,
                         mesh.nElements_global,
                         elementNeighborsOffsets_subdomain,
                         elementNeighbors_subdomain,
                         PETSC_NULL,//weights_subdomain,
                         &petscAdjacency);CHKERRABORT(PROTEUS_COMM_WORLD, ierr);
  MatPartitioning petscPartition;
  MatPartitioningCreate(PROTEUS_COMM_WORLD,&petscPartition);
  MatPartitioningSetAdjacency(petscPartition,petscAdjacency);
  MatPartitioningSetFromOptions(petscPartition);
  warnIfNotRepartitioning(PROTEUS_COMM_WORLD, petscPartition);

  //get a petsc index set that has the new submdomain number for each element
  IS elementPartitioningIS_new;
  MatPartitioningApply(petscPartition,&elementPartitioningIS_new);
  MatPartitioningDestroy(&petscPartition);
  //MatDestroy(petscAdjacency);
  //ISView(elementPartitioningIS_new,PETSC_VIEWER_STDOUT_WORLD);

  //experiment with metis
  //mwf set some defaults and not call if size == 1 since metis crashes
  //cek commenting out for now
  //int etype=1,edgecut=0,base=0;
  //epart assign everything to processor zero by default
  //valarray<int> epart(0,mesh.nElements_global),npart(mesh.nNodes_global);
  //if (size > 1)
  //  METIS_PartMeshNodal(&mesh.nElements_global,&mesh.nNodes_global,mesh.elementNodesArray,&etype,&base,&size,&edgecut,&epart[0],&npart[0]);
  //ISCreateGeneralWithArray(PETSC_COMM_SELF,mesh.nElements_global,&epart[0],&elementPartitioningIS_new);
  //write mesh to view with showme
  //     std::ofstream nodeout("mesh.node"),eleout("mesh.ele"),partout("mesh.part");
  //     eleout<<mesh.nElements_global<<" 3 0"<<std::endl;
  //     partout<<mesh.nElements_global<<"\t"<<size<<std::endl;
  //     for (int eN=0;eN<mesh.nElements_global;eN++)
  //       {
  //    partout<<(eN+1)<<"\t"<<(1+epart[eN])<<std::endl;
  //    eleout<<(eN+1)<<"\t"<<(1+mesh.elementNodesArray[eN*3+0])
  //          <<"\t"<<(1+mesh.elementNodesArray[eN*3+1])
  //          <<"\t"<<(1+mesh.elementNodesArray[eN*3+2])
  //          <<std::endl;
  //       }
  //     nodeout<<mesh.nNodes_global<<" 2 0 0"<<std::endl;
  //     for (int nN=0;nN<mesh.nNodes_global;nN++)
  //       {
  //    nodeout<<(nN+1)<<"\t"<<mesh.nodeArray[nN*3+0]<<"\t"<<mesh.nodeArray[nN*3+1]<<std::endl;
  //       }
  //     eleout.close();
  //     partout.close();
  //count the new number of elements on each subdomain
  valarray<int> nElements_subdomain_new(size);
  ISPartitioningCount(elementPartitioningIS_new,size,&nElements_subdomain_new[0]);

  //get the new offsets for the subdomain to global numbering
  valarray<int> elementOffsets_new(size+1);
  elementOffsets_new[0] = 0;
  for (int sdN=0;sdN<size;sdN++)
    elementOffsets_new[sdN+1] = elementOffsets_new[sdN] + nElements_subdomain_new[sdN];

  //get the new element numbers for the elements on this  subdomain
  IS elementNumberingIS_subdomain_old2new;
  ISPartitioningToNumbering(elementPartitioningIS_new,&elementNumberingIS_subdomain_old2new);

  //now get the new element numbers for the whole mesh so that we
  //can just read this processors elements, reorder, and renumber**
  //
  //**We could do this in parallel by scattering all the element
  //information
  IS elementNumberingIS_global_old2new;
  ISAllGather(elementNumberingIS_subdomain_old2new,&elementNumberingIS_global_old2new);
  //ISView(elementNumberingIS_global_old2new,PETSC_VIEWER_STDOUT_SELF);
  const PetscInt *elementNumbering_global_old2new;
  ISGetIndices(elementNumberingIS_global_old2new,&elementNumbering_global_old2new);
  valarray<int> elementNumbering_global_new2old(mesh.nElements_global);
  for(int eN=0;eN<mesh.nElements_global;eN++)
    elementNumbering_global_new2old[elementNumbering_global_old2new[eN]] = eN;

  //Sort element based arrays, maybe I don't need to do this, maybe
  //I just need to start writing into the subdomain mesh here and
  //preserve subdomain2old and subdomain2global mappings
  valarray<int> elementNodesArray_new(mesh.nElements_global*mesh.nNodes_element),
    elementNeighborsArray_new(mesh.nElements_global*mesh.nElementBoundaries_element),
    elementMaterialTypes_new(mesh.nElements_global),
    elementBoundaryElementsArray_new(mesh.nElementBoundaries_global*2),
    elementBoundaryNodesArray_new(mesh.nElementBoundaries_global*mesh.nNodes_elementBoundary),
    edgeNodesArray_new(mesh.nEdges_global*2);
  valarray<int> elementBoundaryMaterialTypes_new(mesh.nElementBoundaries_global);
  for (int eN=0;eN<mesh.nElements_global;eN++)
    {
      for (int nN=0;nN<mesh.nNodes_element;nN++)
        elementNodesArray_new[eN*mesh.nNodes_element + nN] =
          mesh.elementNodesArray[elementNumbering_global_new2old[eN]*mesh.nNodes_element+nN];
      for (int ebN=0;ebN<mesh.nElementBoundaries_element;ebN++)
        elementNeighborsArray_new[eN*mesh.nElementBoundaries_element+ebN] =
          mesh.elementNeighborsArray[elementNumbering_global_new2old[eN]*mesh.nElementBoundaries_element+ebN];
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
  for(int ebN=0;ebN<mesh.nElementBoundaries_global;ebN++)
    {
      int eN_L_old = mesh.elementBoundaryElementsArray[ebN*2+0],
        eN_R_old = mesh.elementBoundaryElementsArray[ebN*2+1];
      elementBoundaryElementsArray_new[ebN*2+0] = elementNumbering_global_old2new[eN_L_old];
      if(eN_R_old >= 0)
        elementBoundaryElementsArray_new[ebN*2+1] = elementNumbering_global_old2new[eN_R_old];
      //mwf assume same numbering scheme for element boundaries for now?
      elementBoundaryMaterialTypes_new[ebN] = mesh.elementBoundaryMaterialTypes[ebN];
    }
  //4. now we need to build a new node ordering with better data locality for C0 finite elements
  //otherwise we could just grab the nodes on the subdomain and not worry about ownership
  //in the long run it wouldn't be bad to do a global repartition of faces and edges for mixed hybrid
  //and non-conforming finite elements
  MPI_Status status;
  PetscBT nodeMask;
  PetscBTCreate(mesh.nNodes_global,&nodeMask);
  if (rank > 0)
    {
      MPI_Recv(nodeMask,PetscBTLength(mesh.nNodes_global),MPI_CHAR,rank-1,0,PROTEUS_COMM_WORLD,&status);
    }
  //mark the unmarked nodes on this subdomain and store the node numbers
  set<int> nodes_subdomain_owned;
  for(int eN=elementOffsets_new[rank];eN<elementOffsets_new[rank+1];eN++)
    for(int nN=0;nN<mesh.nNodes_element;nN++)
      {
        int nN_global = elementNodesArray_new[eN*mesh.nNodes_element+nN];
        if (!PetscBTLookupSet(nodeMask,nN_global))
          nodes_subdomain_owned.insert(nN_global);
      }
  //ship off the mask
  if (rank < size-1)
    MPI_Send(nodeMask,PetscBTLength(mesh.nNodes_global),MPI_CHAR,rank+1,0,PROTEUS_COMM_WORLD);
  ierr = PetscBTDestroy(&nodeMask);
  if (ierr)
    cerr<<"Error in PetscBTDestroy"<<endl;
  //get the number of nodes on each processor
  valarray<int> nNodes_subdomain_new(size),
    nodeOffsets_new(size+1);
  for (int sdN=0;sdN<size;sdN++)
    if (sdN == rank)
      nNodes_subdomain_new[sdN] = nodes_subdomain_owned.size();
    else
      nNodes_subdomain_new[sdN] = 0;
  valarray<int> nNodes_subdomain_new_send=nNodes_subdomain_new;
  MPI_Allreduce(&nNodes_subdomain_new_send[0],&nNodes_subdomain_new[0],size,MPI_INT,MPI_SUM,PROTEUS_COMM_WORLD);
  nodeOffsets_new[0] = 0;
  for (int sdN=0;sdN<size;sdN++)
    nodeOffsets_new[sdN+1] = nodeOffsets_new[sdN]+nNodes_subdomain_new[sdN];
  //Now as with elements build a global node numbering, sort node
  //based information, and renumber references to node numbers
  valarray<int> nodeNumbering_new2old(nodes_subdomain_owned.size());
  set<int>::iterator nN_ownedp=nodes_subdomain_owned.begin();
  for (int nN=0;nN<int(nodes_subdomain_owned.size());nN++)
    {
      nodeNumbering_new2old[nN] = *nN_ownedp++;
    }
  IS nodeNumberingIS_new2old;
  ISCreateGeneral(PROTEUS_COMM_WORLD,nodes_subdomain_owned.size(),&nodeNumbering_new2old[0],PETSC_COPY_VALUES,&nodeNumberingIS_new2old);
  IS nodeNumberingIS_global_new2old;
  ISAllGather(nodeNumberingIS_new2old,&nodeNumberingIS_global_new2old);
  const PetscInt *nodeNumbering_global_new2old;
  valarray<int> nodeNumbering_old2new_global(mesh.nNodes_global);
  ISGetIndices(nodeNumberingIS_global_new2old,&nodeNumbering_global_new2old);
  for (int nN=0;nN<mesh.nNodes_global;nN++)
    {
      nodeNumbering_old2new_global[nodeNumbering_global_new2old[nN]] = nN;
    }
  for (int eN=0;eN < mesh.nElements_global; eN++)
    {
      int nN_old;
      for (int nN=0;nN < mesh.nNodes_element; nN++)
        {
          nN_old = elementNodesArray_new[eN*mesh.nNodes_element+nN];
          elementNodesArray_new[eN*mesh.nNodes_element+nN] = nodeNumbering_old2new_global[nN_old];
        }
    }
  for (int i=0;i<mesh.nElementBoundaries_global*mesh.nNodes_elementBoundary;i++)
    {
      int nN_old = mesh.elementBoundaryNodesArray[i];
      elementBoundaryNodesArray_new[i] = nodeNumbering_old2new_global[nN_old];
    }
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
  //5. At this point we have new, renumbered and sorted the global element and node based information, and
  //we have it all on each processor so we can add overlap
  set<int> elements_overlap,nodes_overlap;
  for(int eN=elementOffsets_new[rank];eN<elementOffsets_new[rank+1];eN++)
    for (int nN=0;nN<mesh.nNodes_element;nN++)
      {
        int nN_global = elementNodesArray_new[eN*mesh.nNodes_element+nN];
        if (nN_global < nodeOffsets_new[rank] || nN_global >= nodeOffsets_new[rank+1])
          nodes_overlap.insert(nN_global);
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
      //get all elements in the node stars of owned nodes and those elements' nodes
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
                }
            }
        }
    }
  //6. Now build subdomain mesh
  //
  //set what we know
  if (mesh.subdomainp == NULL)
    mesh.subdomainp = new Mesh();
  mesh.subdomainp->usePlex = mesh.usePlex;
  mesh.subdomainp->nElements_global = nElements_subdomain_new[rank] + elements_overlap.size();
  mesh.subdomainp->nNodes_global = nNodes_subdomain_new[rank] + nodes_overlap.size();
  mesh.subdomainp->nNodes_element = mesh.nNodes_element;
  mesh.subdomainp->nNodes_elementBoundary = mesh.nNodes_elementBoundary;
  mesh.subdomainp->nElementBoundaries_element = mesh.nElementBoundaries_element;
  //load the elements and nodes
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
  mesh.subdomainp->elementNodesArray = new int[mesh.subdomainp->nElements_global*mesh.subdomainp->nNodes_element];
  mesh.subdomainp->elementMaterialTypes = new int[mesh.subdomainp->nElements_global];
  valarray<int> elementNumbering_subdomain2global(mesh.subdomainp->nElements_global);
  for (int eN=0;eN<nElements_subdomain_new[rank];eN++)
    {
      int eN_global = eN+elementOffsets_new[rank];
      elementNumbering_subdomain2global[eN] = eN_global;
      mesh.subdomainp->elementMaterialTypes[eN] = elementMaterialTypes_new[eN_global];
      for (int nN=0;nN<mesh.subdomainp->nNodes_element;nN++)
        mesh.subdomainp->elementNodesArray[eN*mesh.subdomainp->nNodes_element+nN] =
          nodeNumbering_global2subdomain[elementNodesArray_new[eN_global*mesh.nNodes_element + nN]];
    }
  set<int>::iterator eN_p=elements_overlap.begin();
  for(int eN=nElements_subdomain_new[rank];eN < nElements_subdomain_new[rank]+int(elements_overlap.size());eN++)
    {
      int eN_global = *eN_p++;
      mesh.subdomainp->elementMaterialTypes[eN] = elementMaterialTypes_new[eN_global];
      elementNumbering_subdomain2global[eN] = eN_global;
      for (int nN=0;nN<mesh.subdomainp->nNodes_element;nN++)
        mesh.subdomainp->elementNodesArray[eN*mesh.subdomainp->nNodes_element+nN] =
          nodeNumbering_global2subdomain[elementNodesArray_new[eN_global*mesh.nNodes_element + nN]];
    }

  if (mesh.subdomainp->nNodes_element == 2)
    {
      constructElementBoundaryElementsArray_edge(*mesh.subdomainp);
      allocateGeometricInfo_edge(*mesh.subdomainp);
      computeGeometricInfo_edge(*mesh.subdomainp);
    }
  else if (mesh.subdomainp->nNodes_element == 3)
    {
      constructElementBoundaryElementsArray_triangle(*mesh.subdomainp);
      allocateGeometricInfo_triangle(*mesh.subdomainp);
      computeGeometricInfo_triangle(*mesh.subdomainp);
    }
  else if (mesh.subdomainp->nNodes_element == 4)
    {
      constructElementBoundaryElementsArray_tetrahedron(*mesh.subdomainp);
      allocateGeometricInfo_tetrahedron(*mesh.subdomainp);
      computeGeometricInfo_tetrahedron(*mesh.subdomainp);
    }
  if (mesh.elementBoundaryMaterialTypes != NULL)
    {
      assert(mesh.elementBoundariesArray != NULL);
      //mwftodo need to copy over elementBoundaryMaterialTypes now
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
  for (int sdN=0;sdN<size+1;sdN++)
    {
      mesh.nodeOffsets_subdomain_owned[sdN] = nodeOffsets_new[sdN];
      mesh.elementOffsets_subdomain_owned[sdN] = elementOffsets_new[sdN];
    }
  //we also need the subdomain 2 new global mappings
  mesh.nodeNumbering_subdomain2global = new int[mesh.subdomainp->nNodes_global];
  for (int nN=0;nN<mesh.subdomainp->nNodes_global;nN++)
    mesh.nodeNumbering_subdomain2global[nN] = nodeNumbering_subdomain2global[nN];
  mesh.elementNumbering_subdomain2global = new int[mesh.subdomainp->nElements_global];
  for (int eN=0;eN<mesh.subdomainp->nElements_global;eN++)
    mesh.elementNumbering_subdomain2global[eN] = elementNumbering_subdomain2global[eN];

  ISRestoreIndices(elementNumberingIS_global_old2new,&elementNumbering_global_old2new);

  ISDestroy(&elementPartitioningIS_new);
  ISDestroy(&elementNumberingIS_subdomain_old2new);
  ISDestroy(&elementNumberingIS_global_old2new);

  ISRestoreIndices(nodeNumberingIS_global_new2old,&nodeNumbering_global_new2old);

  ISDestroy(&nodeNumberingIS_new2old);
  ISDestroy(&nodeNumberingIS_global_new2old);

  return 0;
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
  //6. Figure out which elements are in node stars but are not locally owned, create ghost information
  //   for these, do the same for elements
  set<int> elements_overlap,nodes_overlap,elementBoundaries_overlap,edges_overlap;
  for (int nN = nodeOffsets_new[rank]; nN < nodeOffsets_new[rank+1]; nN++)
    {
      int nN_global_old = nodeNumbering_global_new2old[nN];
      //nodes in node star
      for (int offset_old = mesh.nodeStarOffsets[nN_global_old];
           offset_old < mesh.nodeStarOffsets[nN_global_old+1]; offset_old++)
        {
          int nN_neig_old = mesh.nodeStarArray[offset_old];
          int nN_neig_new = nodeNumbering_global_old2new[nN_neig_old];

          bool offproc = nN_neig_new >=  nodeOffsets_new[rank+1] || nN_neig_new < nodeOffsets_new[rank];
          if (offproc)
            nodes_overlap.insert(nN_neig_new);
        }
      //now get elements, elementBoundaries, and edges in node star
      for (int offset_old = mesh.nodeElementOffsets[nN_global_old];
           offset_old < mesh.nodeElementOffsets[nN_global_old+1]; offset_old++)
        {
          int eN_star_old = mesh.nodeElementsArray[offset_old];
          int eN_star_new = elementNumbering_global_old2new[eN_star_old];
          bool offproc = eN_star_new >= elementOffsets_new[rank+1] || eN_star_new < elementOffsets_new[rank];
          if (offproc)
            elements_overlap.insert(eN_star_new);
          for (int ebN=0; ebN<mesh.nElementBoundaries_element;ebN++)
            {
              int ebN_global=elementBoundariesArray_new[eN_star_new*mesh.nElementBoundaries_element+ebN];
              //              //mwf debug
                          //  std::cout<<"partitionNode default overlap rank= "<<rank<<" nN_new= "<<nN<<" eN_star_new= "<<eN_star_new<<" ebN= "<<ebN
                          //           <<" ebN_global= "<<ebN_global<<" ghost= "<<(ebN_global < elementBoundaryOffsets_new[rank] || ebN_global >= elementBoundaryOffsets_new[rank+1])
                          //           <<" offsets= ["<<elementBoundaryOffsets_new[rank]<<","<<elementBoundaryOffsets_new[rank+1]<<"]"<<std::endl;
              if (ebN_global < elementBoundaryOffsets_new[rank] || ebN_global >= elementBoundaryOffsets_new[rank+1])
                {
                  elementBoundaries_overlap.insert(ebN_global);
                }
            }
          for (int nN0=0;nN0<mesh.nNodes_element;nN0++)
            for (int nN1=nN0+1; nN1<mesh.nNodes_element;nN1++)
              {
                const int nN0_global_old = mesh.elementNodesArray[eN_star_old*mesh.nNodes_element+nN0];
                const int nN1_global_old = mesh.elementNodesArray[eN_star_old*mesh.nNodes_element+nN1];
                const int nN0_global     = nodeNumbering_global_old2new[nN0_global_old];
                const int nN1_global     = nodeNumbering_global_old2new[nN1_global_old];
                int nodes[2];
                nodes[0] = nN0_global;
                nodes[1] = nN1_global;
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
                  edges_overlap.insert(edge_global);
              }//edges
        }//elements in node star
    }//nodes on this processor

  //
  //7. If we want more layers of overlap, do we have to build connectivity info in new numbering then march out
  //   or can we just march through nodes in node_overlap and grab all of their elements that aren't owned?
  //
  int overlap_remaining = nNodes_overlap -1; //default gives 1 layer overlap
  //last set of overlap nodes added
  set<int> last_nodes_added2overlap = nodes_overlap;
  while (overlap_remaining > 0)
    {
      set<int> new_nodes_overlap,new_elements_overlap,new_elementBoundaries_overlap,new_edges_overlap;
      set<int>::iterator nN_p = last_nodes_added2overlap.begin();
      while (nN_p != last_nodes_added2overlap.end())
        {
          int nN_global_new = *nN_p;
          int nN_global_old = nodeNumbering_global_new2old[nN_global_new];//need old numbering for connectivity
          for (int offset_old = mesh.nodeStarOffsets[nN_global_old];
               offset_old < mesh.nodeStarOffsets[nN_global_old+1]; offset_old++)
            {
              int nN_neig_old = mesh.nodeStarArray[offset_old];
              int nN_neig_new = nodeNumbering_global_old2new[nN_neig_old];
              //just need to check if neighbor is offprocessor, may already be in overlap
              //but set merge will take care of that
              bool offproc = nN_neig_new >=  nodeOffsets_new[rank+1] || nN_neig_new < nodeOffsets_new[rank];
              if (offproc)
                new_nodes_overlap.insert(nN_neig_new);
            }//node neighbor loop
          nN_p++;
        }//loop adding new nodes

      //loop through added nodes, grab elements only check if not on processor
      set<int>::iterator nN_newp = last_nodes_added2overlap.begin();
      while (nN_newp != last_nodes_added2overlap.end())
        {
          int nN_global_new = *nN_newp;
          int nN_global_old = nodeNumbering_global_new2old[nN_global_new];//need old numbering for connectivity
          for (int offset_old = mesh.nodeElementOffsets[nN_global_old];
               offset_old < mesh.nodeElementOffsets[nN_global_old+1]; offset_old++)
            {
              int eN_star_old = mesh.nodeElementsArray[offset_old];
              int eN_star_new = elementNumbering_global_old2new[eN_star_old];
              bool offproc = eN_star_new >= elementOffsets_new[rank+1] || eN_star_new < elementOffsets_new[rank];
              if (offproc)
                new_elements_overlap.insert(eN_star_new);
              //element boundaries too
              for (int ebN=0; ebN<mesh.nElementBoundaries_element;ebN++)
                {
                  int ebN_global=elementBoundariesArray_new[eN_star_new*mesh.nElementBoundaries_element+ebN];
                  //              //mwf debug
                  //              std::cout<<"partitionNode overlap_remaining= "<<overlap_remaining<<" rank= "<<rank<<" nN_global_new= "<<nN_global_new<<" eN_star_new= "<<eN_star_new<<" ebN= "<<ebN
                  //                       <<" ebN_global= "<<ebN_global<<" ghost= "<<(ebN_global < elementBoundaryOffsets_new[rank] || ebN_global >= elementBoundaryOffsets_new[rank+1])
                  //                       <<" offsets= ["<<elementBoundaryOffsets_new[rank]<<","<<elementBoundaryOffsets_new[rank+1]<<std::endl;
                  if (ebN_global < elementBoundaryOffsets_new[rank] || ebN_global >= elementBoundaryOffsets_new[rank+1])
                    new_elementBoundaries_overlap.insert(ebN_global);
                }//element boundaries
              //edges
              for (int nN0=0;nN0<mesh.nNodes_element;nN0++)
                for (int nN1=nN0+1; nN1<mesh.nNodes_element;nN1++)
                  {
                    const int nN0_global_old = mesh.elementNodesArray[eN_star_old*mesh.nNodes_element+nN0];
                    const int nN1_global_old = mesh.elementNodesArray[eN_star_old*mesh.nNodes_element+nN1];
                    const int nN0_global     = nodeNumbering_global_old2new[nN0_global_old];
                    const int nN1_global     = nodeNumbering_global_old2new[nN1_global_old];
                    int nodes[2];
                    nodes[0] = nN0_global;
                    nodes[1] = nN1_global;
                    NodeTuple<2> et(nodes);
                    //see the identical lookup in step 6
                    map<NodeTuple<2>, int>::const_iterator edge_it = nodesEdgeMap_global_new.find(et);
                    if (edge_it == nodesEdgeMap_global_new.end())
                      continue;
                    const int edge_global = edge_it->second;
                    if (edge_global < edgeOffsets_new[rank] || edge_global >= edgeOffsets_new[rank+1])
                      new_edges_overlap.insert(edge_global);
                  }//edges
            }//elements in node star
          nN_newp++;
        }//new nodes
      last_nodes_added2overlap.clear();
      set_difference(new_nodes_overlap.begin(),new_nodes_overlap.end(),
                     nodes_overlap.begin(),nodes_overlap.end(),
		     std::inserter(last_nodes_added2overlap,last_nodes_added2overlap.begin()));

      //could do a set_merge
      for (set<int>::iterator nN_addedp = new_nodes_overlap.begin();
           nN_addedp != new_nodes_overlap.end();
           nN_addedp++)
        {
          nodes_overlap.insert(*nN_addedp);
        }
      for (set<int>::iterator eN_addedp = new_elements_overlap.begin();
           eN_addedp != new_elements_overlap.end();
           eN_addedp++)
        {
          elements_overlap.insert(*eN_addedp);
        }
      for (set<int>::iterator ebN_addedp = new_elementBoundaries_overlap.begin();
           ebN_addedp != new_elementBoundaries_overlap.end();
           ebN_addedp++)
        {
          elementBoundaries_overlap.insert(*ebN_addedp);
        }
      for (set<int>::iterator edge_addedp = new_edges_overlap.begin();
           edge_addedp != new_edges_overlap.end();
           edge_addedp++)
        {
          edges_overlap.insert(*edge_addedp);
        }
      //example calls

      overlap_remaining--;
    }//ovelap loop

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
  {
    const int nOwnedNodesBegin = nodeOffsets_new[rank];
    const int nOwnedNodesEnd = nodeOffsets_new[rank+1];
    for (set<int>::iterator eN_oldp = elements_subdomain_owned.begin();
         eN_oldp != elements_subdomain_owned.end(); eN_oldp++)
      for (int nN = 0; nN < mesh.nNodes_element; nN++)
        {
          const int nN_new = nodeNumbering_global_old2new[mesh.elementNodesArray[(*eN_oldp)*mesh.nNodes_element+nN]];
          if (nN_new < nOwnedNodesBegin || nN_new >= nOwnedNodesEnd)
            nodes_overlap.insert(nN_new);
        }
    for (set<int>::iterator eN_newp = elements_overlap.begin();
         eN_newp != elements_overlap.end(); eN_newp++)
      {
        const int eN_old = elementNumbering_global_new2old[*eN_newp];
        for (int nN = 0; nN < mesh.nNodes_element; nN++)
          {
            const int nN_new = nodeNumbering_global_old2new[mesh.elementNodesArray[eN_old*mesh.nNodes_element+nN]];
            if (nN_new < nOwnedNodesBegin || nN_new >= nOwnedNodesEnd)
              nodes_overlap.insert(nN_new);
          }
      }
  }

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

namespace {
  //Reopening the mappings file read-only.
  //
  //The numbering datasets are written collectively through the MPI-IO driver
  //and then reopened read-only by every rank. Nothing used to check the
  //reopen: on Carpenter at >=384 ranks some ranks saw
  //
  //  H5F__super_read(): truncated file:
  //      eof = 115343360, sblock->base_addr = 0, stored_eof = 116070216
  //
  //(93 of 1536 in one run), H5Dopen1 returned -1, H5Dread failed on an invalid
  //id, and the ZERO-initialised destination valarray became the numbering --
  //every old id mapping to new id 0. That is the origin of the degenerate
  //face numbering and of edgeNodesArray left at its -1 initialiser.
  //
  //Closing the writer before every reopen (beb3ba81) is necessary but NOT
  //sufficient. A standalone reproduction on Carpenter (HDF5 1.14.6, Lustre,
  //7680 ranks on 40 nodes, production-sized datasets; torino_flume
  //debug/stale_eof) showed what the failures depend on:
  //  - NOT time: waiting 5, 50 or 500 ms after the barrier fails as often as
  //    not waiting, and a stat() right after a failed open always shows the
  //    correct size;
  //  - concurrency on one node: a node's first opener never failed, while
  //    ranks opening at the same moment as many others on their node did
  //    (up to ~100 of 7680 per reopen);
  //  - one real H5Fopen per node first fixes it: with the node leader opening
  //    before the rest, 4 failures in 61,440 opens. A leader that only
  //    fstat()s the file and sees the right size does not help.
  //The likely cause is the Lustre client handing some of many concurrent
  //openers its stale cached size for a file another client just changed.
  //
  //So the reopen is LOCAL and node-leader-first: each rank opens the file
  //with HDF5's default (sec2) driver, one rank per node before the others,
  //and a failed open is retried with a short jittered backoff. Reads are
  //ordinary per-rank POSIX reads whose sieve buffer coalesces the point
  //selections, and the close is not collective, so one slow rank no longer
  //holds every other rank in an MPI_File_close barrier. Same build, mesh and
  //ranks (7680, 14.7M nodes): "Collecting elements" 704 s vs 1215 s with the
  //world MPI-IO reopen, partitioning 882 s vs 1426 s; one MPI-IO run took
  //8981 s when a single straggler rank held the other 7679 in that barrier.
  //
  //PROTEUS_MAPPINGS_READ_DRIVER=mpio restores the world MPI-IO reopen
  //(collective, not retried) as a fallback.
  bool mappingsReadUsesMPIIO()
  {
    static const bool use_mpio = []() {
      const char* driver = std::getenv("PROTEUS_MAPPINGS_READ_DRIVER");
      return driver != NULL && std::string(driver) == "mpio";
    }();
    return use_mpio;
  }

  //Transfer mode for a read from a file opened by openMappingsReadOnly. HDF5
  //rejects collective transfer on a non-MPI driver ("collective access for
  //MPI-based drivers only"), so collective is only requested on the MPI-IO
  //path; on the local path the read is an ordinary independent read.
  herr_t setMappingsReadXfer(hid_t dxpl)
  {
    if (mappingsReadUsesMPIIO())
      return H5Pset_dxpl_mpio(dxpl, H5FD_MPIO_COLLECTIVE);
    return 0;
  }

  //Ranks sharing this rank's node (and its Lustre client), split once.
  MPI_Comm mappingsNodeComm(const MPI_Comm& comm)
  {
    static MPI_Comm node = MPI_COMM_NULL;
    static MPI_Comm split_from = MPI_COMM_NULL;
    if (node == MPI_COMM_NULL || split_from != comm)
      {
        int rank = 0;
        MPI_Comm_rank(comm, &rank);
        MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &node);
        split_from = comm;
      }
    return node;
  }

  //A local read-only open, retried: 10 quick attempts with 10-60 ms of
  //jitter (the failure is a collision with the node's other openers, not a
  //window that closes with time), then up to 5 more 1 s apart. Returns the
  //file id or -1, and the number of attempts used.
  hid_t openLocalWithRetry(const char* name, hid_t fapl, int rank, int& attempts)
  {
    unsigned int seed = 2654435761u * (unsigned int)(rank + 1);
    const int quick = 10, slow = 5;
    hid_t fid = -1;
    attempts = 0;
    while (attempts < quick + slow)
      {
        H5E_BEGIN_TRY {
          fid = H5Fopen(name, H5F_ACC_RDONLY, fapl);
        } H5E_END_TRY;
        attempts++;
        if (fid >= 0 || attempts == quick + slow)
          break;
        if (attempts < quick)
          usleep(10000 + rand_r(&seed) % 50000);
        else
          sleep(1);
      }
    return fid;
  }

  //A rank that exhausts its retries writes what it saw to
  //mappings_open_fail.<rank>.txt before the caller aborts: the abort can
  //drop that rank's stderr (it did on Carpenter, job 3180113).
  void reportMappingsOpenFailure(int rank, const char* name, int attempts)
  {
    char path[64];
    snprintf(path, sizeof(path), "mappings_open_fail.%d.txt", rank);
    FILE* f = fopen(path, "w");
    if (f == NULL)
      f = stderr;
    struct stat st;
    int sr = stat(name, &st);
    fprintf(f, "rank %d: H5Fopen(%s, RDONLY) failed %d times\n", rank, name, attempts);
    if (sr == 0)
      fprintf(f, "stat: st_size %lld\n", (long long)st.st_size);
    else
      fprintf(f, "stat failed: %s\n", strerror(errno));
    H5Eprint2(H5E_DEFAULT, f);
    if (f != stderr)
      fclose(f);
  }

  //Collective: every rank must call this (barrier and the retry count reduction).
  hid_t openMappingsReadOnly(const MPI_Comm& comm, const char* name)
  {
    const bool use_mpio = mappingsReadUsesMPIIO();
    int rank = 0;
    MPI_Comm_rank(comm, &rank);
    static bool reported = false;
    if (!reported)
      {
        if (use_mpio)
          logEvent("mappings file read-only reopen: MPI-IO driver (PROTEUS_MAPPINGS_READ_DRIVER=mpio)",3);
        else
          logEvent("mappings file read-only reopen: local sec2 driver",3);
        reported = true;
      }
    MPI_Barrier(comm);
    hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
    if (fapl < 0) return fapl;
    if (use_mpio)
      H5Pset_fapl_mpio(fapl, comm, MPI_INFO_NULL);
    hid_t fid = -1;
    int attempts = 1;
    int node_rank = 0;
    if (use_mpio)
      {
        //the world MPI-IO open is collective and must not be retried per rank
        H5E_BEGIN_TRY {
          fid = H5Fopen(name, H5F_ACC_RDONLY, fapl);
        } H5E_END_TRY;
      }
    else
      {
        //node leader first, then the rest of the node
        MPI_Comm node = mappingsNodeComm(comm);
        MPI_Comm_rank(node, &node_rank);
        if (node_rank == 0)
          fid = openLocalWithRetry(name, fapl, rank, attempts);
        MPI_Barrier(node);
        if (node_rank != 0)
          fid = openLocalWithRetry(name, fapl, rank, attempts);
      }
    if (fid < 0)
      {
        //reopen once more with error reporting on, to capture the HDF5 stack
        fid = H5Fopen(name, H5F_ACC_RDONLY, fapl);
        if (fid < 0)
          reportMappingsOpenFailure(rank, name, attempts + 1);
      }
    H5Pclose(fapl);
    int counts[4] = {(attempts > 1 && node_rank == 0) ? 1 : 0,
                     (attempts > 1 && node_rank != 0) ? 1 : 0,
                     (fid < 0) ? 1 : 0,
                     attempts};
    int totals[3] = {0, 0, 0}, maxAttempts = 0;
    MPI_Allreduce(counts, totals, 3, MPI_INT, MPI_SUM, comm);
    MPI_Allreduce(&counts[3], &maxAttempts, 1, MPI_INT, MPI_MAX, comm);
    if (totals[0] + totals[1] + totals[2] > 0)
      {
        char msg[256];
        snprintf(msg, sizeof(msg), "mappings file reopen: %d node leader(s) and %d other rank(s) needed a retry (max %d attempts), %d failed (stale size on a parallel filesystem)",
                 totals[0], totals[1], maxAttempts, totals[2]);
        logEvent(msg,3);
      }
    return fid;
  }
}

int partitionNodesFromTetgenFiles(const MPI_Comm& PROTEUS_COMM_WORLD, const char* filebase, int indexBase, 
                                  Mesh& newMesh, int nNodes_overlap, double memHardLimit)
{
  using namespace std;
  PetscErrorCode ierr;
  PetscMPIInt size,rank;

  ierr = MPI_Comm_size(PROTEUS_COMM_WORLD,&size);CHKERRABORT(PROTEUS_COMM_WORLD, ierr);
  ierr = MPI_Comm_rank(PROTEUS_COMM_WORLD,&rank);CHKERRABORT(PROTEUS_COMM_WORLD, ierr);
  //cek todo: decide whether to remove PETSc logging and just use proteus logging
  PetscLogStage partitioning_stage;
  PetscLogStageRegister("Mesh Partition",&partitioning_stage);
  PetscLogStagePush(partitioning_stage);

  /***********************************************************************
    partition domain based on the nodes without reading in the global mesh.

    1. Partition nodes in the default partition of contiguous chunks with
       input ordering

    2. Determine nodal connectivity on local processor

    3. Generate new nodal partition using PETSc interface

    4. Collect subdomain elements: any that contain owned nodes; tag ownership

    4a,b. Collect element boundaries and edges on the subdomain elements and tag ownership

    5. Generate global element, element boundary, and edge  numbering for new subdomain ownership

    6. Create overlap (ghost) information for nodes and elements

    7. March through additional layers of overlap if requested,

    8. Build subdomain meshes in new numbering
  ***********************************************************************/
  //
  //0. Set up tetgen files. Note, tetgen should have been run with -feen to get alll the faces and elements
  //
  bool failed = false;
  const int simplexDim = 4;
  const int vertexDim  = 3;
  using namespace IOutils;
  std::string vertexFileName  = std::string(filebase) + ".node" ;
  std::string elementFileName = std::string(filebase) + ".ele" ;
  std::string elementBoundaryFileName  = std::string(filebase) + ".face" ;
  std::string edgeFileName  = std::string(filebase) + ".edge" ;
  //
  //1. Build default nodal partition
  //
  //compute offsets to build subdomain2global ordering for nodes
  //in default partitioning of the domain into subdomains. All
  //we need is the number of nodes in the global mesh
  //
  //read nodes for tetgen format
  //first just read the number of nodes and whether or not there are node tags
  logEvent("Building default partitioning",5);
  int read_elements_event;
  PetscLogEventRegister("Read eles",0,&read_elements_event);
  PetscLogEventBegin(read_elements_event,0,0,0,0);
  std::ifstream vertexFile(vertexFileName.c_str());
  if (!vertexFile.good())
    {
      std::cerr<<"cannot open Tetgen node file "
               <<vertexFileName<<std::endl;
      failed = true;
      return failed;
    }
  int hasVertexMarkers(0),hasVertexAttributes(0),nSpace(3),nNodes_global;
  //read number of vertices and whether node flags are provided
  vertexFile >> eatcomments >> nNodes_global >> nSpace >> hasVertexAttributes >> hasVertexMarkers >> eatline ;
  assert(nNodes_global > 0);
  assert(nSpace == 3);
  newMesh.nNodes_global = nNodes_global;
  newMesh.nNodes_element = simplexDim;
  newMesh.nNodes_elementBoundary = simplexDim-1;
  newMesh.nElementBoundaries_element = simplexDim;
  //cek todo: review the situation with tetgen and triangle atribs and markers
  if (hasVertexAttributes > 0 && rank==0)
    {
      std::cerr<<"WARNING Tetgen nodes hasAttributes= "<<hasVertexAttributes
               <<" > 0 will treat first value as integer id for boundary!!"<<std::endl;
      hasVertexMarkers = 1;
    }
  //don't need to read anymore from .nodes for now
  //leaving file open because we'll pick up here to read the node coordinates
  //after partioning, which just requires topology information
  //
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
  //2. Determine nodal connectivity (nodeStarArray) for nodes on subdomain
  //
  //connectivty commes from the topology (elements) file. We just grab the elements
  //that contain currently owned nodes
  logEvent("Determining nodal connectivity to construct new partitioning",5);
  std::ifstream elementFile(elementFileName.c_str());
  if (!elementFile.good())
    {
      std::cerr<<"cannot open Tetgen element file "
               <<elementFileName<<std::endl;
      failed = true;
      return failed;
    }
  //read elements
  int nNodesPerSimplex(simplexDim),hasElementMarkers = 0,nElements_global;
  elementFile >> eatcomments >> nElements_global >> nNodesPerSimplex >> hasElementMarkers >> eatline;
  assert(nElements_global > 0);
  assert(nNodesPerSimplex == simplexDim);
  newMesh.nElements_global = nElements_global;
  valarray<int> element_nodes_old(4);
  vector<set<int>> nodeStar(nNodes_subdomain_old);
  map<int,valarray<int>> elements_old;
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
          bool inSubdomain=false;
          if (nN_star >= nodeOffsets_old[rank] && nN_star < nodeOffsets_old[rank+1])
            {
              inSubdomain = true;
              for (int jv = 0; jv < simplexDim; jv++)
                {
                  if (iv != jv)
                    {
                      int nN_star_subdomain = nN_star-nodeOffsets_old[rank];
                      nodeStar[nN_star_subdomain].insert(element_nodes_old[jv]);
                    }
                }
            }
          if (inSubdomain)
            elements_old[ie] = element_nodes_old;
        }
      elementFile >> eatline;
    }//end ie
  elements_old.clear();
  elementFile.close();
  //cek todo: track c++ containers and put inside scope to reduce dynamic memory usage
  PetscLogEventEnd(read_elements_event,0,0,0,0);
  int repartition_nodes_event;
  PetscLogEventRegister("Repart nodes",0,&repartition_nodes_event);
  PetscLogEventBegin(repartition_nodes_event,0,0,0,0);
  //done reading element file for first time; will need to read again after node partitioning
  //build compact data structure for nodeStar, and
  //find maximum number of nodes in any star
  int max_nNodeNeighbors_node=0;
  for (int nN=0;nN<nNodes_subdomain_old;nN++)
    max_nNodeNeighbors_node=max(max_nNodeNeighbors_node,static_cast<int>(nodeStar[nN].size()));
  //build connectivity data structures for PETSc
  PetscBool isInitialized;
  PetscInitialized(&isInitialized);
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
  bool test=false;
  for (int nN = 0,offset=0; nN < nNodes_subdomain_old; nN++)
    {
      for (auto nN_star=nodeStar[nN].begin(); nN_star!=nodeStar[nN].end(); nN_star++,offset++)
        {
          nodeNeighbors_subdomain[offset] = *nN_star;
        }
      nodeNeighborsOffsets_subdomain[nN+1]=offset;
      if (test)//make sure nodeNeighbors are sorted in ascending order ()
      {
        valarray<int> neighbors(nodeNeighborsOffsets_subdomain[nN+1]-nodeNeighborsOffsets_subdomain[nN]);
        for (int i=0;i<neighbors.size();i++)
          neighbors[i] = nodeNeighbors_subdomain[nodeNeighborsOffsets_subdomain[nN]+i];
        sort(&nodeNeighbors_subdomain[nodeNeighborsOffsets_subdomain[nN]],&nodeNeighbors_subdomain[nodeNeighborsOffsets_subdomain[nN+1]]);
        for (int i=0;i<neighbors.size();i++)
          assert(neighbors[i] == nodeNeighbors_subdomain[nodeNeighborsOffsets_subdomain[nN]+i]);
      }
      //weight nodes by size of star
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
  const double max_rss_gb(memHardLimit/1024.0);
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
  // out of core creation of global old2new mapping
  //
  //create the mappings file
  logEvent("Writing/reading node numberings to hdf5",5);
  hid_t mappings_plist_id = H5Pcreate(H5P_FILE_ACCESS);
  H5Pset_fapl_mpio(mappings_plist_id, PROTEUS_COMM_WORLD, MPI_INFO_NULL);
  const char* H5FILE_NAME("mappings.h5");
  hid_t file_id = H5Fcreate(H5FILE_NAME, H5F_ACC_TRUNC, H5P_DEFAULT, mappings_plist_id);
  assert(file_id != H5I_INVALID_HID);
  H5Pclose(mappings_plist_id);
  //
  //write old2new
  //
  hsize_t nodes_global_count[]={static_cast<hsize_t>(nNodes_global)};
  const hsize_t ARRAY_RANK(1);
  hid_t nodes_old2new_filespace_id = H5Screate_simple(ARRAY_RANK, nodes_global_count, NULL);
  hid_t nodes_old2new_dataset_id = H5Dcreate(file_id, "nodeNumbering_old2new", H5T_NATIVE_INT, 
                                             nodes_old2new_filespace_id,
                                             H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  hsize_t nodes_subdomain_count[]={static_cast<hsize_t>(nNodes_subdomain_old)};
  hsize_t nodes_subdomain_offset[]={static_cast<hsize_t>(nodeOffsets_old[rank])};
  hid_t nodes_old2new_memspace_id = H5Screate_simple(ARRAY_RANK, nodes_subdomain_count, NULL);
  H5Sselect_hyperslab(nodes_old2new_filespace_id, H5S_SELECT_SET, 
                      nodes_subdomain_offset, NULL, 
                      nodes_subdomain_count, NULL);
  hid_t nodes_old2new_plist_id = H5Pcreate(H5P_DATASET_XFER);
  herr_t status = H5Pset_dxpl_mpio(nodes_old2new_plist_id, H5FD_MPIO_COLLECTIVE);
  status = H5Dwrite(nodes_old2new_dataset_id, H5T_NATIVE_INT, 
                    nodes_old2new_memspace_id, nodes_old2new_filespace_id,
		                nodes_old2new_plist_id, nodes_subdomain_old2new_array);
  H5Pclose(nodes_old2new_plist_id);
  H5Dclose(nodes_old2new_dataset_id);
  H5Sclose(nodes_old2new_memspace_id);
  H5Sclose(nodes_old2new_filespace_id);
  //
  //write new2old
  //
  hid_t nodes_new2old_filespace_id = H5Screate_simple(ARRAY_RANK, nodes_global_count, NULL);
  hid_t nodes_new2old_dataset_id = H5Dcreate(file_id, "nodeNumbering_new2old", H5T_NATIVE_INT, 
                                              nodes_new2old_filespace_id,
	          			                            H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  hid_t nodes_new2old_memspace_id = H5Screate_simple(ARRAY_RANK, nodes_subdomain_count, NULL);
  //new2old is not a hyperslab, unlike old2new, so we have to use H5Sselect_elements
  //create a node list to write to for the global new2old mapping
  valarray<hsize_t> new_node_indices(nNodes_subdomain_old);
  valarray<int> old_node_indices(nNodes_subdomain_old);
  for (int i=0;i<nNodes_subdomain_old;i++)
    {
      new_node_indices[i] = static_cast<hsize_t>(nodes_subdomain_old2new_array[i]);//new node indices where we are writing to
      old_node_indices[i] = nodeOffsets_old[rank]+i;//old node numbers at those locations
    }
  status = H5Sselect_elements(nodes_new2old_filespace_id, H5S_SELECT_SET, 
                              nNodes_subdomain_old, &new_node_indices[0]);
  hid_t nodes_new2old_plist_id = H5Pcreate(H5P_DATASET_XFER);
  status = H5Pset_dxpl_mpio(nodes_new2old_plist_id, H5FD_MPIO_COLLECTIVE);
  status = H5Dwrite(nodes_new2old_dataset_id, H5T_NATIVE_INT, 
                    nodes_new2old_memspace_id, nodes_new2old_filespace_id, 
                    nodes_new2old_plist_id, &old_node_indices[0]);
  H5Pclose(nodes_new2old_plist_id);
  H5Dclose(nodes_new2old_dataset_id);
  H5Sclose(nodes_new2old_memspace_id);
  H5Sclose(nodes_new2old_filespace_id);
  H5Fclose(file_id);
  //
  //test out of core
  //
  IS nodeNumberingIS_global_old2new;
  const PetscInt *nodeNumbering_global_old2new;
  valarray<int> nodeNumbering_global_new2old;
  if (test)
    {
      ISAllGather(nodeNumberingIS_subdomain_old2new,&nodeNumberingIS_global_old2new);
      ISGetIndices(nodeNumberingIS_global_old2new,&nodeNumbering_global_old2new);
      nodeNumbering_global_new2old.resize(nNodes_global);
      for (int nN = 0; nN < nNodes_global; nN++)
        nodeNumbering_global_new2old[nodeNumbering_global_old2new[nN]] = nN;
      herr_t status;
      valarray<int> nodeNumbering_old2new_read(nNodes_global);
      file_id = openMappingsReadOnly(PROTEUS_COMM_WORLD, H5FILE_NAME);
      assert(file_id != H5I_INVALID_HID);
      hid_t nodeNumbering_old2new_dataset_id = H5Dopen1(file_id, "/nodeNumbering_old2new");
      status = H5Dread(nodeNumbering_old2new_dataset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                       &nodeNumbering_old2new_read[0]);
      status = H5Dclose(nodeNumbering_old2new_dataset_id);
      for (int i=0;i<nNodes_global;i++)
        {
          assert(nodeNumbering_global_old2new[i] == nodeNumbering_old2new_read[i]);
        }
      std::cout<<"==================out of core old2new nodes is correct!===================="<<std::endl;
      hid_t nodeNumbering_new2old_dataset_id = H5Dopen2(file_id, "/nodeNumbering_new2old", H5P_DEFAULT);
      valarray<int> nodeNumbering_new2old_read(nNodes_global);
      status = H5Dread(nodeNumbering_new2old_dataset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                       &nodeNumbering_new2old_read[0]);
      status = H5Dclose(nodeNumbering_new2old_dataset_id);
      H5Fclose(file_id);
      for (int i=0;i<nNodes_global;i++)
      {
        assert(nodeNumbering_global_new2old[i] == nodeNumbering_new2old_read[i]);
      }
      std::cout<<"==================out of core new2old nodes is correct!===================="<<std::endl;
      nodeNumbering_global_new2old.resize(0);
    }
  //cek todo: put this in the right place and use integers not doubles
  Vec nodeNumbering_old2new_petsc;
  ierr = VecCreateMPI(PROTEUS_COMM_WORLD, nNodes_subdomain_old, nNodes_global, &nodeNumbering_old2new_petsc);
  CHKERRABORT(PROTEUS_COMM_WORLD, ierr);
  valarray<PetscInt> indices(nNodes_subdomain_old);
  valarray<PetscInt> ix(nNodes_subdomain_old);
  valarray<PetscScalar> y(nNodes_subdomain_old);
  for (int i=0;i<nNodes_subdomain_old;i++)
    {
      indices[i] = i+nodeOffsets_old[rank];
      ix[i]=i;
      y[i]=static_cast<PetscScalar>(nodes_subdomain_old2new_array[i]);
    }
  ISLocalToGlobalMapping mapping;
  PetscInt bs(1);
  ierr = ISLocalToGlobalMappingCreate(PROTEUS_COMM_WORLD, bs, nNodes_subdomain_old, &indices[0], PETSC_USE_POINTER, &mapping);
  CHKERRABORT(PROTEUS_COMM_WORLD, ierr);
  ierr = VecSetLocalToGlobalMapping(nodeNumbering_old2new_petsc, mapping);
  CHKERRABORT(PROTEUS_COMM_WORLD, ierr);  
  ierr = VecSetValuesLocal(nodeNumbering_old2new_petsc, nNodes_subdomain_old, &ix[0], &y[0], INSERT_VALUES);
  CHKERRABORT(PROTEUS_COMM_WORLD, ierr);
  ierr = VecAssemblyBegin(nodeNumbering_old2new_petsc);
  CHKERRABORT(PROTEUS_COMM_WORLD, ierr);
  ierr = VecAssemblyEnd(nodeNumbering_old2new_petsc);
  CHKERRABORT(PROTEUS_COMM_WORLD, ierr);
  if (test)
    {
      PetscScalar* test_local;
      VecGetArray(nodeNumbering_old2new_petsc,&test_local);
      for (int i=0;i<nNodes_subdomain_old;i++)
	{
	  assert(test_local[i] == static_cast<PetscScalar>(nodes_subdomain_old2new_array[i]));
	}
      VecRestoreArray(nodeNumbering_old2new_petsc,&test_local);
    }
  //ISRestoreIndices(nodeNumberingIS_subdomain_old2new, &nodes_subdomain_old2new_array);
  //ierr=ISDestroy(&nodeNumberingIS_subdomain_old2new);CHKERRABORT(PROTEUS_COMM_WORLD, ierr);

  PetscLogEventEnd(repartition_nodes_event,0,0,0,0);
  int receive_element_mask_event;
  PetscLogEventRegister("Recv. ele mask",0,&receive_element_mask_event);
  PetscLogEventBegin(receive_element_mask_event,0,0,0,0);
  //
  //4. To build subdomain meshes, go through and collect elements containing
  //   the locally owned nodes. Assign processor ownership of elements
  //
  logEvent("Collecting elements containing locally owned nodes (element support of the subdomain nodes)",5);
  PetscLogEventEnd(receive_element_mask_event,0,0,0,0);
  ierr = enforceMemoryLimit(PROTEUS_COMM_WORLD, rank, max_rss_gb,"Done with masks");CHKERRABORT(PROTEUS_COMM_WORLD, ierr);
  int build_subdomains_reread_elements_event;
  PetscLogEventRegister("Reread eles",0,&build_subdomains_reread_elements_event);
  PetscLogEventBegin(build_subdomains_reread_elements_event,0,0,0,0);
  //
  //mark the unmarked elements on this subdomain and store element numbers (in old numbering)
  //
  //We need to find the elements in the support of nodes in the new
  //partitioning, so we'll need to re-read the elements file to get
  //the the elements for the nodes in the new partitioning. We will be collecting OLD element numbers.
  std::ifstream elementFile2(elementFileName.c_str());
  if (!elementFile2.good())
    {
      std::cerr<<"cannot open Tetgen elements file"
               <<elementFileName<<std::endl;
      failed = true;
      return failed;
    }
  elementFile2 >> eatcomments >> nElements_global >> nNodesPerSimplex >> hasElementMarkers >> eatline;
  assert(nElements_global > 0);
  assert(nNodesPerSimplex == simplexDim);
  //cek todo: check that these are all necessary
  set<int> elements_subdomain_owned;
  valarray<int> element_nodes_new(4);
  vector<set<int>> nodeElementsStar(nNodes_subdomain_new[rank]);
  vector<set<int>> nodeStarNew(nNodes_subdomain_new[rank]);
  map<int,valarray<int>> elementNodesArrayMap;
  map<int,long int> elementMaterialTypesMap;
  map<NodeTuple<3>,ElementNeighbors> elementBoundaryElementsMap;
  map<NodeTuple<2>,set<pair<int,int>>> edgeElementsMap;
  set<int> node_collection;
  vector<valarray<int>> element_old_nodes_collection;
  vector<int> elements_collection;
  vector<double> elementId_collection;
  map<int, int> nodes_old2new_subdomain_map;
  //note any element index containers are in the old element numbering
  int eN_c_start = 0;//start of the collection of elements we are currently processing  
  //Chunk this re-read by a FIXED node-collection size. Sizing the chunk as
  //nElements_global/size makes the NUMBER of chunks equal the number of ranks,
  //so every rank added costs another full read cycle, and each cycle has more
  //participants. Restored from 77c99396^.
  const int nNodes_collection_max=10000;
  //Hold the mappings file open across the whole loop. Opening and closing it
  //once per chunk had every rank open the same Lustre file on every chunk.
  file_id = openMappingsReadOnly(PROTEUS_COMM_WORLD, H5FILE_NAME);
  if (file_id < 0)
    {
      H5Eprint2(H5E_DEFAULT, stderr);
      SETERRABORT(PROTEUS_COMM_WORLD, PETSC_ERR_FILE_OPEN,
                  "could not reopen the mappings file; see the HDF5 stack above");
    }
  assert(file_id != H5I_INVALID_HID);
  for (int ie = 0; ie < nElements_global; ie++)
    {
      //
      //Accumulate a collection of elements in the old numbering of sufficient length, 
      //then get the new node numbers for those old nodes, renumber, take ownership of elements,
      //and then release the memory for the collection and continue
      //basic idea is to use only a part of old2new at a time.
      //doing with hdf5 for now, but could do with scatter
      //
      int ne, nv, elementId(0);
      long double elementId_double;
      elementFile2 >> eatcomments >> ne;
      ne -= indexBase;
      assert(0 <= ne && ne < nElements_global && elementFile.good());
      elements_collection.push_back(ne);
      for (int iv = 0; iv < simplexDim; iv++)
        {
          elementFile2 >> nv ;
          nv -= indexBase;
          assert(0 <= nv && nv < nNodes_global);
          node_collection.insert(nv);
          element_nodes_old[iv] = nv;
        }
      if (hasElementMarkers > 0)
        {
          elementFile2 >> elementId_double;
          elementId_collection.push_back(elementId_double);
        }
      element_old_nodes_collection.push_back(element_nodes_old);
      if (node_collection.size() >= nNodes_collection_max || ie == nElements_global-1)
        {
          //create a node list to read from old2new mapping
	        valarray<int> nodes_old2new_subset(node_collection.size());
          map<int,int> nodes_old2new_subset_map;
	        bool test_scatter=false;
	        if (test_scatter)
	        {
	          Vec node_old2new_collection_petsc;
	          ierr = VecCreateMPI(PROTEUS_COMM_WORLD, node_collection.size(), PETSC_DETERMINE,&node_old2new_collection_petsc);
	          CHKERRABORT(PROTEUS_COMM_WORLD, ierr);
	          PetscInt collection_low, collection_high;
	          ierr = VecGetOwnershipRange(node_old2new_collection_petsc, &collection_low, &collection_high);
	          CHKERRABORT(PROTEUS_COMM_WORLD, ierr);
	          valarray<PetscInt> node_collection_array(node_collection.size());
	          valarray<PetscInt> iy_collection(node_collection.size());
            int i_nc=0;
	          for (auto nv=node_collection.begin();nv!=node_collection.end();nv++,i_nc++)
		        {
		          node_collection_array[i_nc] = static_cast<PetscInt>(*nv);
		          iy_collection[i_nc] = i_nc + collection_low;
		        }
	          IS old_ix,iy;
	          ierr = ISCreateGeneral(PROTEUS_COMM_WORLD, node_collection.size(), &node_collection_array[0],
				    PETSC_USE_POINTER, &old_ix);
	          CHKERRABORT(PROTEUS_COMM_WORLD, ierr);
	          ierr = ISCreateGeneral(PROTEUS_COMM_WORLD, node_collection.size(), &iy_collection[0],
				                           PETSC_USE_POINTER, &iy);
	          CHKERRABORT(PROTEUS_COMM_WORLD, ierr);
	          VecScatter old2new_collection_scatter;
	          ierr = VecScatterCreate(nodeNumbering_old2new_petsc, old_ix,
				                            node_old2new_collection_petsc, iy,
				                            &old2new_collection_scatter);
	          CHKERRABORT(PROTEUS_COMM_WORLD, ierr);
	          ierr = VecScatterBegin(old2new_collection_scatter,
				                           nodeNumbering_old2new_petsc, node_old2new_collection_petsc,
				                           INSERT_VALUES, SCATTER_FORWARD);
	          CHKERRABORT(PROTEUS_COMM_WORLD, ierr);

	          ierr = VecScatterEnd(old2new_collection_scatter,
				                         nodeNumbering_old2new_petsc, node_old2new_collection_petsc,
				                         INSERT_VALUES, SCATTER_FORWARD);
	          CHKERRABORT(PROTEUS_COMM_WORLD, ierr);
	          PetscScalar* nodes_old2new_subset_array;
	          ierr = VecGetArray(node_old2new_collection_petsc, &nodes_old2new_subset_array);
	          CHKERRABORT(PROTEUS_COMM_WORLD, ierr);
	          for (int i=0;i<node_collection.size();i++)
		        {
		          nodes_old2new_subset[i] = nodes_old2new_subset_array[i];
		        }
	          ierr = VecRestoreArray(node_old2new_collection_petsc, &nodes_old2new_subset_array);
	          CHKERRABORT(PROTEUS_COMM_WORLD, ierr);
	          ierr = VecDestroy(&node_old2new_collection_petsc);
	          CHKERRABORT(PROTEUS_COMM_WORLD, ierr);
	          ierr = VecScatterDestroy(&old2new_collection_scatter);
	          CHKERRABORT(PROTEUS_COMM_WORLD, ierr);
	          ierr = ISDestroy(&old_ix);
	          CHKERRABORT(PROTEUS_COMM_WORLD, ierr);
	          ierr = ISDestroy(&iy);
	          CHKERRABORT(PROTEUS_COMM_WORLD, ierr);
	          for (int i=0;i<node_collection.size();i++)
		        {
		          nodes_old2new_subset_map[node_collection_array[i]] = nodes_old2new_subset[i];
		        }
	          if (test)
		        {
		          for (int i=0;i<node_collection.size();i++)
		            {
		              assert(nodes_old2new_subset_map[node_collection_array[i]] == nodeNumbering_global_old2new[node_collection_array[i]]);
		            }
		        }
	        }
	      else
	        {
	          herr_t status;
	          valarray<hsize_t> node_collection_array(node_collection.size());
	          int i_nc=0;
	          for (auto nv=node_collection.begin();nv!=node_collection.end();nv++,i_nc++)
		        {
		          node_collection_array[i_nc] = static_cast<hsize_t>(*nv);
		        }
	          nodes_old2new_dataset_id = H5Dopen1(file_id,"/nodeNumbering_old2new");
	          nodes_old2new_filespace_id = H5Dget_space(nodes_old2new_dataset_id);
	          status = H5Sselect_elements(nodes_old2new_filespace_id, H5S_SELECT_SET, 
					  node_collection.size(), &node_collection_array[0]);
	          hsize_t dims[] = {static_cast<hsize_t>(node_collection.size())};
	          hid_t nodes_old2new_subset_memspace_id = H5Screate_simple(1, dims, NULL);
	          nodes_old2new_plist_id = H5Pcreate(H5P_DATASET_XFER);
#ifdef H5_HAVE_PARALLEL
	          status = H5Pset_dxpl_mpio(nodes_old2new_plist_id, H5FD_MPIO_INDEPENDENT);
#endif
	          status = H5Dread(nodes_old2new_dataset_id, H5T_NATIVE_INT, 
			                       nodes_old2new_subset_memspace_id, nodes_old2new_filespace_id, 
			                       H5P_DEFAULT, &nodes_old2new_subset[0]);
	          H5Pclose(nodes_old2new_plist_id);
	          H5Sclose(nodes_old2new_subset_memspace_id);
	          H5Sclose(nodes_old2new_filespace_id);
	          H5Dclose(nodes_old2new_dataset_id);
	          for (int i=0;i<node_collection.size();i++)
		        {
		          nodes_old2new_subset_map[node_collection_array[i]] = nodes_old2new_subset[i];
		        }
	          if (test)
		        {
		          for (int i=0;i<node_collection.size();i++)
		            {
		              assert(nodes_old2new_subset_map[node_collection_array[i]] == nodeNumbering_global_old2new[node_collection_array[i]]);
		            }
		        }
	        }
          for (int eN_c = eN_c_start; eN_c < ie+1;eN_c++)
            {
              ne = elements_collection[eN_c-eN_c_start];
              for (int iv = 0; iv < simplexDim; iv++)
                {
                  element_nodes_old[iv] = element_old_nodes_collection[eN_c-eN_c_start][iv];
                  element_nodes_new[iv] = nodes_old2new_subset_map[element_nodes_old[iv]];
                }
              NodeTuple<4> nodeTuple(&element_nodes_new[0]);
              for (int iv = 0; iv < simplexDim; iv++)
                {
                  int nN_star_new = element_nodes_new[iv];
                  bool inSubdomain=false;
                  if (nN_star_new >= nodeOffsets_new[rank] && nN_star_new < nodeOffsets_new[rank+1])
                    {
                      inSubdomain = true;
                      for (int iv=0; iv < simplexDim; iv++)
                        nodes_old2new_subdomain_map[element_nodes_old[iv]] = element_nodes_new[iv];
                      //add all the element boundaries of this element
                      for (int ebN=0;ebN < 4 ; ebN++)
                        {
                          int nodes[3] = { element_nodes_new[(ebN+1) % 4],
                            element_nodes_new[(ebN+2) % 4],
                            element_nodes_new[(ebN+3) % 4]};
                          NodeTuple<3> nodeTuple(nodes);
                          if(elementBoundaryElementsMap.find(nodeTuple) != elementBoundaryElementsMap.end())
                            {
                              if (elementBoundaryElementsMap[nodeTuple].right == -1 && ne != elementBoundaryElementsMap[nodeTuple].left)
                                {
                                  elementBoundaryElementsMap[nodeTuple].right=ne;
                                  elementBoundaryElementsMap[nodeTuple].right_ebN_element=ebN;
                                }
                            }
                          else
                            {
                              elementBoundaryElementsMap[nodeTuple] = ElementNeighbors(ne,ebN);
                            }
                        }
                      //add all the edges of this element
                      for (int nNL=0,edN=0;nNL < 4 ; nNL++)
                        for(int nNR=nNL+1;nNR < 4;nNR++,edN++)
                          {
                            int nodes[2] = { element_nodes_new[nNL],
                              element_nodes_new[nNR]};
                            NodeTuple<2> nodeTuple(nodes);
                            edgeElementsMap[nodeTuple].insert(pair<int,int>(ne,edN));
                          }
                      //add all the nodes to the node star
                      int nN_star_new_subdomain = nN_star_new - nodeOffsets_new[rank];
                      nodeElementsStar[nN_star_new_subdomain].insert(ne);
                      for (int jv = 0; jv < simplexDim; jv++)
                        {
                          if (iv != jv)
                            {
                              int nN_point_new = element_nodes_new[jv];
                              nodeStarNew[nN_star_new_subdomain].insert(nN_point_new);
                            }
                        }
                    }
                  if (inSubdomain)
                    {
                      elementNodesArrayMap[ne] = element_nodes_new;
                    }
                }
              //if this element contains any node owned by this subdomain, check node 1
              if (elementNodesArrayMap.find(ne) != elementNodesArrayMap.end())
                {
                  if (nodeTuple.nodes[1] >= nodeOffsets_new[rank] && nodeTuple.nodes[1] < nodeOffsets_new[rank+1])
                    elements_subdomain_owned.insert(ne);
                  if (hasElementMarkers > 0)
                    {
                      elementId = static_cast<long int>(elementId_collection[eN_c-eN_c_start]);
                      elementMaterialTypesMap[ne] = elementId;
                    }
                }
            }
          eN_c_start = ie+1;
          node_collection.clear();
          elements_collection.clear();
          elementId_collection.clear();
          element_old_nodes_collection.clear();
        }
      elementFile2 >> eatline;
    }
  elementFile2.close();
  H5Fclose(file_id);
  int nElements_owned_subdomain(elements_subdomain_owned.size()),
    nElements_owned_new=0;
  MPI_Allreduce(&nElements_owned_subdomain,&nElements_owned_new,1,MPI_INT,MPI_SUM,PROTEUS_COMM_WORLD);
  assert(nElements_owned_new == nElements_global);
  PetscLogEventEnd(build_subdomains_reread_elements_event,0,0,0,0);
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
  //map to old element numbering
  valarray<int> elementNumbering_subdomain_new2old(elements_subdomain_owned.size());
  set<int>::iterator eN_ownedp = elements_subdomain_owned.begin();
  map<int, int> elementNumbering_old2new_map;
  for (int eN = 0; eN < int(elements_subdomain_owned.size()); eN++,eN_ownedp++)
    {
      //only for owned elements on the subdomain
      elementNumbering_subdomain_new2old[eN] = *eN_ownedp;
      elementNumbering_old2new_map[*eN_ownedp] = eN;
    }
  //
  //write this subdomain to the mapping file
  //
  ierr = enforceMemoryLimit(PROTEUS_COMM_WORLD, rank, max_rss_gb,"Writing element numberings to hdf5");CHKERRABORT(PROTEUS_COMM_WORLD, ierr);
  logEvent("Writing element numberings to hdf5",5);
  mappings_plist_id = H5Pcreate(H5P_FILE_ACCESS);
  status = H5Pset_fapl_mpio(mappings_plist_id, PROTEUS_COMM_WORLD, MPI_INFO_NULL);
  assert(status >= 0);
  file_id = H5Fopen(H5FILE_NAME, H5F_ACC_RDWR, mappings_plist_id);
  assert(file_id != H5I_INVALID_HID);
  H5Pclose(mappings_plist_id);
  hsize_t e_dims[]={static_cast<hsize_t>(nElements_global)};
  hid_t e_new2old_filespace_id = H5Screate_simple(ARRAY_RANK, e_dims, NULL);
  hid_t e_new2old_dataset_id = H5Dcreate(file_id, "elementNumbering_new2old", H5T_NATIVE_INT, 
                                         e_new2old_filespace_id,
                                         H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  hsize_t e_count[]={static_cast<hsize_t>(nElements_subdomain_new[rank])};
  hsize_t e_offset[]={static_cast<hsize_t>(elementOffsets_new[rank])};
  hid_t e_new2old_memspace_id = H5Screate_simple(ARRAY_RANK, e_count, NULL);
  H5Sselect_hyperslab(e_new2old_filespace_id, H5S_SELECT_SET, e_offset, NULL, e_count, NULL);
  hid_t e_new2old_plist_id = H5Pcreate(H5P_DATASET_XFER);
  H5Pset_dxpl_mpio(e_new2old_plist_id, H5FD_MPIO_COLLECTIVE);
  status = H5Dwrite(e_new2old_dataset_id, H5T_NATIVE_INT, 
                    e_new2old_memspace_id, e_new2old_filespace_id,
                    e_new2old_plist_id, &elementNumbering_subdomain_new2old[0]);
  H5Dflush(e_new2old_dataset_id);
  H5Pclose(e_new2old_plist_id);
  H5Dclose(e_new2old_dataset_id);
  H5Sclose(e_new2old_memspace_id);
  H5Sclose(e_new2old_filespace_id);
  valarray<hsize_t> old_element_indices(static_cast<hsize_t>(nElements_subdomain_new[rank]));
  valarray<int> new_element_indices(nElements_subdomain_new[rank]);
  ierr = enforceMemoryLimit(PROTEUS_COMM_WORLD, rank, max_rss_gb,"new2old done; now old2new");CHKERRABORT(PROTEUS_COMM_WORLD, ierr);
  //do old2new map too, can't use hyperslab
  ierr = enforceMemoryLimit(PROTEUS_COMM_WORLD, rank, max_rss_gb,"old2new index valarrays done");CHKERRABORT(PROTEUS_COMM_WORLD, ierr);
  for (int i=0;i<nElements_subdomain_new[rank];i++)
    {
      old_element_indices[i] = static_cast<hsize_t>(elementNumbering_subdomain_new2old[i]);
      new_element_indices[i] = elementOffsets_new[rank]+i;
    }
  hid_t e_old2new_filespace_id = H5Screate_simple(ARRAY_RANK, e_dims, NULL);
  hid_t e_old2new_dataset_id = H5Dcreate(file_id, "elementNumbering_old2new", H5T_NATIVE_INT, 
                                         e_old2new_filespace_id,
                                         H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  hid_t e_old2new_memspace_id = H5Screate_simple(ARRAY_RANK, e_count, NULL);
  status = H5Sselect_elements(e_old2new_filespace_id, H5S_SELECT_SET, 
                              nElements_subdomain_new[rank], &old_element_indices[0]);
  hid_t e_old2new_plist_id = H5Pcreate(H5P_DATASET_XFER);
  H5Pset_dxpl_mpio(e_old2new_plist_id, H5FD_MPIO_COLLECTIVE);
  status = H5Dwrite(e_old2new_dataset_id, H5T_NATIVE_INT, 
                    e_old2new_memspace_id, e_old2new_filespace_id, 
                    e_old2new_plist_id, &new_element_indices[0]);
  H5Pclose(e_old2new_plist_id);
  H5Sclose(e_old2new_memspace_id);
  H5Dclose(e_old2new_dataset_id);
  H5Sclose(e_old2new_filespace_id);
  H5Fclose(file_id);
  //
  //now get maps for all the elements on the subdomain, not just owned elements
  //
  ierr = enforceMemoryLimit(PROTEUS_COMM_WORLD, rank, max_rss_gb,"Reading numberings from hdf5 to get subdomain map");CHKERRABORT(PROTEUS_COMM_WORLD, ierr);
  logEvent("Reading element numberings from hdf5 to get subdomain map",5);

  valarray<hsize_t> old_element_indices_subdomain(static_cast<hsize_t>(elementNodesArrayMap.size()));
  valarray<int> new_element_indices_subdomain(elementNodesArrayMap.size());
  int eN_old_subdomain = 0;
  for (auto it = elementNodesArrayMap.begin(); it != elementNodesArrayMap.end(); it++, eN_old_subdomain++)
    {
      old_element_indices_subdomain[eN_old_subdomain] = it->first;
    }
  file_id = openMappingsReadOnly(PROTEUS_COMM_WORLD, H5FILE_NAME);
  if (file_id < 0)
    {
      H5Eprint2(H5E_DEFAULT, stderr);
      SETERRABORT(PROTEUS_COMM_WORLD, PETSC_ERR_FILE_OPEN,
                  "could not reopen the mappings file; see the HDF5 stack above");
    }
  e_old2new_dataset_id = H5Dopen1(file_id, "/elementNumbering_old2new");
  e_old2new_filespace_id = H5Dget_space(e_old2new_dataset_id);
  status = H5Sselect_elements(e_old2new_filespace_id, H5S_SELECT_SET, 
                              elementNodesArrayMap.size(), &old_element_indices_subdomain[0]);
  hsize_t e_subdomain_count[]={static_cast<hsize_t>(elementNodesArrayMap.size())};
  hid_t e_old2new_subdomain_memspace_id = H5Screate_simple(ARRAY_RANK, e_subdomain_count, NULL);
  hid_t e_old2new_subdomain_plist_id = H5Pcreate(H5P_DATASET_XFER);
  status = setMappingsReadXfer(e_old2new_subdomain_plist_id);
  status = H5Dread(e_old2new_dataset_id, H5T_NATIVE_INT, 
                   e_old2new_subdomain_memspace_id, e_old2new_filespace_id, 
                   H5P_DEFAULT, &new_element_indices_subdomain[0]);
  H5Pclose(e_old2new_subdomain_plist_id);
  H5Sclose(e_old2new_subdomain_memspace_id);
  H5Dclose(e_old2new_dataset_id);
  H5Sclose(e_old2new_filespace_id);
  H5Fclose(file_id);
  ierr = enforceMemoryLimit(PROTEUS_COMM_WORLD, rank, max_rss_gb,"Constructing element subdomain maps");CHKERRABORT(PROTEUS_COMM_WORLD, ierr);
  map<int,int> elementNumbering_old2new_subdomain_map;
  map<int,int> elementNumbering_new2old_subdomain_map;
  for (int i=0;i<elementNodesArrayMap.size();i++)
    {
      elementNumbering_old2new_subdomain_map[old_element_indices_subdomain[i]] = new_element_indices_subdomain[i];
      elementNumbering_new2old_subdomain_map[new_element_indices_subdomain[i]] = old_element_indices_subdomain[i];
    }
  if (test)
  {
    //these were needed inside the last test section, which was in a loop, so release now
    ISRestoreIndices(nodeNumberingIS_global_old2new,&nodeNumbering_global_old2new);
    ierr=ISDestroy(&nodeNumberingIS_global_old2new);CHKERRABORT(PROTEUS_COMM_WORLD, ierr);
    //set up global element numbering
    IS elementNumberingIS_subdomain_new2old;
    const PetscInt *elementNumbering_global_new2old;//needs to be restored
    valarray<int> elementNumbering_global_old2new(nElements_global);
    ISCreateGeneral(PROTEUS_COMM_WORLD,elements_subdomain_owned.size(),
                    &elementNumbering_subdomain_new2old[0],PETSC_COPY_VALUES,
                    &elementNumberingIS_subdomain_new2old);
    IS elementNumberingIS_global_new2old;
    ISAllGather(elementNumberingIS_subdomain_new2old,&elementNumberingIS_global_new2old);
    ISGetIndices(elementNumberingIS_global_new2old,&elementNumbering_global_new2old);
    //construct reverse mapping
    for (int eN = 0; eN < nElements_global; eN++)
      {
        elementNumbering_global_old2new[elementNumbering_global_new2old[eN]] = eN;
      }
    //check old2new
    valarray<int> elementNumbering_global_old2new_read(nElements_global);
    file_id = openMappingsReadOnly(PROTEUS_COMM_WORLD, H5FILE_NAME);
    hid_t e_old2new_dataset_id = H5Dopen1(file_id, "/elementNumbering_old2new");
    status = H5Dread(e_old2new_dataset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                      &elementNumbering_global_old2new_read[0]);
    status = H5Dclose(e_old2new_dataset_id);
    for (int i=0;i<nElements_global;i++)
    {
      assert(elementNumbering_global_old2new[i] == elementNumbering_global_old2new_read[i]);
    }
    std::cout<<"==================out of core elements old2new is correct!===================="<<std::endl;
    for (auto it = elementNodesArrayMap.begin(); it != elementNodesArrayMap.end(); it++)
      assert(elementNumbering_old2new_subdomain_map[it->first] == 
             elementNumbering_global_old2new_read[it->first]);
    std::cout<<"==================out of core elements old2new map is correct!===================="<<std::endl;
    //check new2old
    valarray<int> elementNumbering_global_new2old_read(nElements_global);
    hid_t e_new2old_dataset_id = H5Dopen1(file_id, "/elementNumbering_new2old");
    status = H5Dread(e_new2old_dataset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                     &elementNumbering_global_new2old_read[0]);
    status = H5Dclose(e_new2old_dataset_id);
    status = H5Fclose(file_id);
    //assert() is compiled out by -DNDEBUG, so this verification has to
    //count and report rather than assert, or the whole block is cost
    //with no signal.
    {
      long bad_=0, first_=-1;
      for (int i=0;i<nElements_global;i++)
        if (elementNumbering_global_new2old[i] != elementNumbering_global_new2old_read[i]) { if (first_<0) first_=i; bad_++; }
      if (bad_)
        {
          std::cerr<<"PROTEUS numbering check failed: elementNumbering_global_new2old vs elementNumbering_global_new2old_read, "
                   <<bad_<<" of "<<nElements_global<<" differ, first at "<<first_<<std::endl;
          SETERRABORT(PROTEUS_COMM_WORLD, PETSC_ERR_PLIB,
                      "out-of-core numbering does not match in-core");
        }
    }
    std::cout<<"==================out of core elements new2old is correct!===================="<<std::endl;
    for (size_t i=0;i< new_element_indices_subdomain.size(); i++)
    {
      int eN = new_element_indices_subdomain[i];
      assert(elementNumbering_new2old_subdomain_map[eN] == elementNumbering_global_new2old_read[eN]);
    }
    std::cout<<"==================out of core elements new2old map is correct!===================="<<std::endl;
    ISRestoreIndices(elementNumberingIS_global_new2old,&elementNumbering_global_new2old);
    ierr=ISDestroy(&elementNumberingIS_global_new2old);CHKERRABORT(PROTEUS_COMM_WORLD, ierr);
    ierr=ISDestroy(&elementNumberingIS_subdomain_new2old);CHKERRABORT(PROTEUS_COMM_WORLD, ierr);
  }
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
  std::ifstream elementBoundaryFile(elementBoundaryFileName.c_str());

  if (!elementBoundaryFile.good())
    {
      std::cerr<<"cannot open Tetgen face file "
               <<elementBoundaryFileName<<std::endl;
      failed = true;
      return failed;
    }

  bool hasElementBoundaryMarkers = false;
  int nElementBoundaries_global;
  int ihasElementBoundaryMarkers(0);
  elementBoundaryFile >> eatcomments >> nElementBoundaries_global >> ihasElementBoundaryMarkers >> eatline ;
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
  map<int,valarray<int>> elementBoundariesMap;
  set<int> supportedElementBoundaries;
  set<int> elementBoundaries_subdomain;
  for (int ieb = 0; ieb < nElementBoundaries_global; ieb++)
    {
      int neb,nn0,nn1,nn2; int ebId(0);
      elementBoundaryFile >> eatcomments >> neb >> nn0 >> nn1 >> nn2;
      if (ihasElementBoundaryMarkers > 0)
        elementBoundaryFile >> ebId;
      neb -= indexBase;
      nn0 -= indexBase;
      nn1 -= indexBase;
      nn2 -= indexBase;
      assert(0 <= neb && neb < nElementBoundaries_global && elementBoundaryFile.good());
      //grab the element boundaries for the node if the node is owned by the subdomain
      //this will miss the element boundaries on the "outside boundary" of the star, which we'll grab later
      //
      int nn0_new = -1;
      if (nodes_old2new_subdomain_map.find(nn0) != nodes_old2new_subdomain_map.end())
        nn0_new = nodes_old2new_subdomain_map[nn0];
      if (nn0_new >= nodeOffsets_new[rank] && nn0_new < nodeOffsets_new[rank+1])
        {
          nodeElementBoundariesStar[nn0_new-nodeOffsets_new[rank]].insert(neb);
          supportedElementBoundaries.insert(neb);
        }
      int nn1_new = -2;
      if (nodes_old2new_subdomain_map.find(nn1) != nodes_old2new_subdomain_map.end())
        nn1_new = nodes_old2new_subdomain_map[nn1];
      if (nn1_new >= nodeOffsets_new[rank] && nn1_new < nodeOffsets_new[rank+1])
        {
          nodeElementBoundariesStar[nn1_new-nodeOffsets_new[rank]].insert(neb);
          supportedElementBoundaries.insert(neb);
        }
      int nn2_new = -3;
      if (nodes_old2new_subdomain_map.find(nn2) != nodes_old2new_subdomain_map.end())
        nn2_new = nodes_old2new_subdomain_map[nn2];
      if (nn2_new >= nodeOffsets_new[rank] && nn2_new < nodeOffsets_new[rank+1])
        {
          nodeElementBoundariesStar[nn2_new-nodeOffsets_new[rank]].insert(neb);
          supportedElementBoundaries.insert(neb);
        }
      //if any of the new nodes is negative, then this face does not belong to an element in the subdomain
      //so the check on nodes[1] will always be false
      const int nodes[3] = {nn0_new,nn1_new,nn2_new};
      NodeTuple<3> nodeTuple(nodes);
      elementBoundaryFile >> eatline;
      if (elementBoundaryElementsMap.find(nodeTuple) != elementBoundaryElementsMap.end())//this element boundary is on an element in the subdomain
        {
          elementBoundaries_subdomain.insert(neb);
          assert(nn0_new >= 0 && nn1_new >= 0 && nn2_new >= 0);
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
              valarray<int> elementBoundaries_element(-1,4);
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
                  valarray<int> elementBoundaries_element(-1, 4);
                  elementBoundariesMap[eN_right] = elementBoundaries_element;
                  //assign
                  elementBoundariesMap[eN_right][elementBoundaryElementsMap[nodeTuple].right_ebN_element] = neb;
                }
            }
        }
    }
  //done reading element boundaries
  elementBoundaryFile.close();
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
      for (int iv=0;iv<4;iv++)
        {
          //the elementNodesArrayMap is in the old element numbering while the elementBoundariesMap is in the new element numbering       
          int nN_global = elementNodesArrayMap[elementNumbering_new2old_subdomain_map[elementBoundariesp->first]][iv];
          if (nN_global >= nodeOffsets_new[rank] && nN_global < nodeOffsets_new[rank+1])
            {
              //add all the faces to this node star
              for(int eb=0;eb<4;eb++)
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
  //
  //Now as with elements and nodes build a global face numbering
  //
  valarray<int> elementBoundaryNumbering_new2old(elementBoundaries_subdomain_owned.size());
  map<int,int> elementBoundaryNumbering_old2new_map;
  set<int>::iterator ebN_ownedp=elementBoundaries_subdomain_owned.begin();
  for (int ebN=0;ebN<int(elementBoundaries_subdomain_owned.size());ebN++, ebN_ownedp++)
    {
      elementBoundaryNumbering_new2old[ebN] = *ebN_ownedp;
      elementBoundaryNumbering_old2new_map[*ebN_ownedp] = ebN+elementBoundaryOffsets_new[rank];
    }
  //
  //write this subdomain to the mapping file
  //
  logEvent("Writing/reading elementBoundary (face) numberings to hdf5",5);
  hsize_t eb_dims[]={static_cast<hsize_t>(nElementBoundaries_global)};
  hid_t eb_new2old_filespace_id = H5Screate_simple(ARRAY_RANK, eb_dims, NULL);
  mappings_plist_id = H5Pcreate(H5P_FILE_ACCESS);
  H5Pset_fapl_mpio(mappings_plist_id, PROTEUS_COMM_WORLD, MPI_INFO_NULL);
  file_id = H5Fopen(H5FILE_NAME, H5F_ACC_RDWR, mappings_plist_id);
  H5Pclose(mappings_plist_id);
  hid_t eb_new2old_dataset_id = H5Dcreate(file_id, "elementBoundaryNumbering_new2old", H5T_NATIVE_INT, 
                                          eb_new2old_filespace_id,
                                          H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  hsize_t eb_count[] = {static_cast<hsize_t>(nElementBoundaries_subdomain_new[rank])};
  hsize_t eb_offset[] = {static_cast<hsize_t>(elementBoundaryOffsets_new[rank])};
  hid_t eb_new2old_memspace_id = H5Screate_simple(ARRAY_RANK, eb_count, NULL);
  H5Sselect_hyperslab(eb_new2old_filespace_id, H5S_SELECT_SET, eb_offset, NULL, eb_count, NULL);
  hid_t eb_new2old_plist_id = H5Pcreate(H5P_DATASET_XFER);
  H5Pset_dxpl_mpio(eb_new2old_plist_id, H5FD_MPIO_COLLECTIVE);
  status = H5Dwrite(eb_new2old_dataset_id, H5T_NATIVE_INT, eb_new2old_memspace_id, eb_new2old_filespace_id,
                    eb_new2old_plist_id, &elementBoundaryNumbering_new2old[0]);
  H5Pclose(eb_new2old_plist_id);
  H5Sclose(eb_new2old_memspace_id);
  H5Sclose(eb_new2old_filespace_id);
  H5Dclose(eb_new2old_dataset_id);
  //do old2new map too, can't use hyperslab
  valarray<hsize_t> old_elementBoundary_indices(nElementBoundaries_subdomain_new[rank]);
  valarray<int> new_elementBoundary_indices(nElementBoundaries_subdomain_new[rank]);
  for (int i=0;i<nElementBoundaries_subdomain_new[rank];i++)
    {
      old_elementBoundary_indices[i] = elementBoundaryNumbering_new2old[i];
      new_elementBoundary_indices[i] = elementBoundaryOffsets_new[rank]+i;
    }
  hid_t eb_old2new_filespace_id = H5Screate_simple(ARRAY_RANK, eb_dims, NULL);
  hid_t eb_old2new_dataset_id = H5Dcreate(file_id, "elementBoundaryNumbering_old2new", H5T_NATIVE_INT, 
                                            eb_old2new_filespace_id,
                                            H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  hid_t eb_old2new_memspace_id = H5Screate_simple(ARRAY_RANK, eb_count, NULL);
  status = H5Sselect_elements(eb_old2new_filespace_id, H5S_SELECT_SET, 
                              nElementBoundaries_subdomain_new[rank], &old_elementBoundary_indices[0]);
  hid_t eb_old2new_plist_id = H5Pcreate(H5P_DATASET_XFER);
  H5Pset_dxpl_mpio(eb_old2new_plist_id, H5FD_MPIO_COLLECTIVE);  
  status = H5Dwrite(eb_old2new_dataset_id, H5T_NATIVE_INT, 
                    eb_old2new_memspace_id, eb_old2new_filespace_id, 
                    eb_old2new_plist_id, &new_elementBoundary_indices[0]);
  H5Pclose(eb_new2old_plist_id);
  H5Sclose(eb_old2new_memspace_id);
  H5Dclose(eb_old2new_dataset_id);
  H5Sclose(eb_old2new_filespace_id);
  H5Fclose(file_id);
  //now get maps for all the elements on the subdomain, not just owned
  //
  valarray<hsize_t> old_elementBoundary_indices_subdomain(elementBoundaries_subdomain.size());
  valarray<int> new_elementBoundary_indices_subdomain(elementBoundaries_subdomain.size());
  int ebN_subdomain = 0;
  for (auto it = elementBoundaries_subdomain.begin(); 
       it != elementBoundaries_subdomain.end(); 
       it++, ebN_subdomain++)
    {
      old_elementBoundary_indices_subdomain[ebN_subdomain] = static_cast<hsize_t>(*it);
    }
  file_id = openMappingsReadOnly(PROTEUS_COMM_WORLD, H5FILE_NAME);
  if (file_id < 0)
    {
      H5Eprint2(H5E_DEFAULT, stderr);
      SETERRABORT(PROTEUS_COMM_WORLD, PETSC_ERR_FILE_OPEN,
                  "could not reopen the mappings file; see the HDF5 stack above");
    }
  eb_old2new_filespace_id = H5Screate_simple(ARRAY_RANK, eb_dims, NULL);
  eb_old2new_dataset_id = H5Dopen1(file_id, "/elementBoundaryNumbering_old2new");
  status = H5Sselect_elements(eb_old2new_filespace_id, H5S_SELECT_SET, 
                              elementBoundaries_subdomain.size(), 
                              &old_elementBoundary_indices_subdomain[0]);
  hsize_t eb_subdomain_count[]={static_cast<hsize_t>(elementBoundaries_subdomain.size())};
  hid_t eb_old2new_subdomain_memspace_id = H5Screate_simple(ARRAY_RANK, eb_subdomain_count, NULL);
  hid_t eb_old2new_subdomain_plist_id = H5Pcreate(H5P_DATASET_XFER);
  setMappingsReadXfer(eb_old2new_subdomain_plist_id);  
  status = H5Dread(eb_old2new_dataset_id, H5T_NATIVE_INT, 
                   eb_old2new_subdomain_memspace_id, eb_old2new_filespace_id, 
                   H5P_DEFAULT, &new_elementBoundary_indices_subdomain[0]);
  H5Pclose(eb_old2new_subdomain_plist_id);
  H5Sclose(eb_old2new_subdomain_memspace_id);
  H5Sclose(eb_old2new_filespace_id);
  H5Dclose(eb_old2new_dataset_id);
  H5Fclose(file_id);
  map<int,int> elementBoundaryNumbering_old2new_subdomain_map;
  map<int,int> elementBoundaryNumbering_new2old_subdomain_map;
  for (int i=0;i<elementBoundaries_subdomain.size();i++)
    {
      elementBoundaryNumbering_old2new_subdomain_map[old_elementBoundary_indices_subdomain[i]] = new_elementBoundary_indices_subdomain[i];
      elementBoundaryNumbering_new2old_subdomain_map[new_elementBoundary_indices_subdomain[i]] = old_elementBoundary_indices_subdomain[i];
    }
  IS elementBoundaryNumberingIS_subdomain_new2old;
  ISCreateGeneral(PROTEUS_COMM_WORLD,elementBoundaries_subdomain_owned.size(),&elementBoundaryNumbering_new2old[0],PETSC_COPY_VALUES,&elementBoundaryNumberingIS_subdomain_new2old);
  if (test)
  {
    IS elementBoundaryNumberingIS_global_new2old;
    const PetscInt *elementBoundaryNumbering_global_new2old;
    valarray<int> elementBoundaryNumbering_global_old2new(nElementBoundaries_global);
    ISAllGather(elementBoundaryNumberingIS_subdomain_new2old,&elementBoundaryNumberingIS_global_new2old);
    elementBoundaryNumbering_global_old2new.resize(newMesh.nElementBoundaries_global);
    ierr = enforceMemoryLimit(PROTEUS_COMM_WORLD, rank, max_rss_gb,"Allocating elementBoudnary old2new/new2old");CHKERRABORT(PROTEUS_COMM_WORLD, ierr);
    ISGetIndices(elementBoundaryNumberingIS_global_new2old,&elementBoundaryNumbering_global_new2old);
    for (int i=0;i<nElementBoundaries_global;i++)
    {
      elementBoundaryNumbering_global_old2new[elementBoundaryNumbering_global_new2old[i]]=i;
    }
    
    herr_t status;
    valarray<int> elementBoundaryNumbering_old2new_read(nElementBoundaries_global);
    file_id = openMappingsReadOnly(PROTEUS_COMM_WORLD, H5FILE_NAME);
    eb_old2new_dataset_id = H5Dopen1(file_id, "/elementBoundaryNumbering_old2new");
    status = H5Dread(eb_old2new_dataset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                     &elementBoundaryNumbering_old2new_read[0]);
    status = H5Dclose(eb_old2new_dataset_id);
    //assert() is compiled out by -DNDEBUG, so this verification has to
    //count and report rather than assert, or the whole block is cost
    //with no signal.
    {
      long bad_=0, first_=-1;
      for (int i=0;i<nElementBoundaries_global;i++)
        if (elementBoundaryNumbering_global_old2new[i] != elementBoundaryNumbering_old2new_read[i]) { if (first_<0) first_=i; bad_++; }
      if (bad_)
        {
          std::cerr<<"PROTEUS numbering check failed: elementBoundaryNumbering_global_old2new vs elementBoundaryNumbering_old2new_read, "
                   <<bad_<<" of "<<nElementBoundaries_global<<" differ, first at "<<first_<<std::endl;
          SETERRABORT(PROTEUS_COMM_WORLD, PETSC_ERR_PLIB,
                      "out-of-core numbering does not match in-core");
        }
    }
    std::cout<<"==================out of core old2new elementBoundaries is correct!===================="<<std::endl;
    hid_t eb_new2old_dataset_id = H5Dopen1(file_id, "/elementBoundaryNumbering_new2old");
    valarray<int> elementBoundaryNumbering_new2old_read(nElementBoundaries_global);
    status = H5Dread(eb_new2old_dataset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                     &elementBoundaryNumbering_new2old_read[0]);
    status = H5Dclose(eb_new2old_dataset_id);
    status = H5Fclose(file_id);
    //assert() is compiled out by -DNDEBUG, so this verification has to
    //count and report rather than assert, or the whole block is cost
    //with no signal.
    {
      long bad_=0, first_=-1;
      for (int i=0;i<nElementBoundaries_global;i++)
        if (elementBoundaryNumbering_global_new2old[i] != elementBoundaryNumbering_new2old_read[i]) { if (first_<0) first_=i; bad_++; }
      if (bad_)
        {
          std::cerr<<"PROTEUS numbering check failed: elementBoundaryNumbering_global_new2old vs elementBoundaryNumbering_new2old_read, "
                   <<bad_<<" of "<<nElementBoundaries_global<<" differ, first at "<<first_<<std::endl;
          SETERRABORT(PROTEUS_COMM_WORLD, PETSC_ERR_PLIB,
                      "out-of-core numbering does not match in-core");
        }
    }
    std::cout<<"==================out of core new2old elementBoundaries is correct!===================="<<std::endl;
    for (int i=0;i<elementBoundaries_subdomain.size();i++)
    {
      int ebN = old_elementBoundary_indices_subdomain[i];
      assert(elementBoundaryNumbering_global_old2new[ebN] == elementBoundaryNumbering_old2new_subdomain_map[ebN]);
    }
    std::cout<<"==================out of core elementBoundaries old2new subdomain_map is correct!===================="<<std::endl;
    for (int i=0;i<elementBoundaries_subdomain.size();i++)
    {
      int ebN = new_elementBoundary_indices_subdomain[i];
      assert(elementBoundaryNumbering_global_new2old[ebN] == elementBoundaryNumbering_new2old_subdomain_map[ebN]);
    }
    std::cout<<"==================out of core elementBoundaries new2old subdomain_map is correct!===================="<<std::endl;
    ISRestoreIndices(elementBoundaryNumberingIS_global_new2old,&elementBoundaryNumbering_global_new2old);
    ierr=ISDestroy(&elementBoundaryNumberingIS_subdomain_new2old);CHKERRABORT(PROTEUS_COMM_WORLD, ierr);
    ierr=ISDestroy(&elementBoundaryNumberingIS_global_new2old);CHKERRABORT(PROTEUS_COMM_WORLD, ierr);
  } 
  PetscLogEventEnd(build_subdomains_faces_event,0,0,0,0);
  ierr = enforceMemoryLimit(PROTEUS_COMM_WORLD, rank, max_rss_gb,"Done allocating elementBoudnary old2new/new2old");CHKERRABORT(PROTEUS_COMM_WORLD, ierr);
  int build_subdomains_edges_event;
  PetscLogEventRegister("Subd edges",0,&build_subdomains_edges_event);
  PetscLogEventBegin(build_subdomains_edges_event,0,0,0,0);

  //
  //4c,5c. Repeate the process for edges
  //
  logEvent("Generating global edge numbering and ownership corresponding to new subdomain ownership",5);
  std::ifstream edgeFile(edgeFileName.c_str());

  if (!edgeFile.good())
    {
      std::cerr<<"cannot open Tetgen edge file"
               <<edgeFileName<<std::endl;
      failed = true;
      return failed;
    }

  bool hasEdgeMarkers = false;
  int nEdges_global;
  int ihasEdgeMarkers(0);
  edgeFile >> eatcomments >> nEdges_global >> eatline;// edge file doesn currently contain markers >> ihasEdgeMarkers >> eatline ;
  assert(nEdges_global > 0);
  if (ihasEdgeMarkers > 0)
    {
      hasEdgeMarkers = true;
    }
  newMesh.nEdges_global = nEdges_global;
  set<int> edges_subdomain_owned;
  vector<set<int> > nodeEdgesStar(nNodes_subdomain_new[rank]);
  map<int,int> edgeMaterialTypesMap;
  map<int,valarray<int> > elementEdgesMap;
  map<int,pair<int,int> > edgeNodesMap;
  set<int> supportedEdges;
  for (int ied = 0; ied < nEdges_global; ied++)
    {
      int ned,nn0,nn1; int edId(0);
      edgeFile >> eatcomments >> ned >> nn0 >> nn1;
      if (ihasEdgeMarkers > 0)
        edgeFile >> edId;
      ned -= indexBase;
      nn0 -= indexBase;
      nn1 -= indexBase;
      assert(0 <= ned && ned < nEdges_global && edgeFile.good());
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
      edgeFile >> eatline;
      if (edgeElementsMap.find(nodeTuple) != edgeElementsMap.end())//this edge is on an element in the subdomain
        {
          if (nodeTuple.nodes[0] >= nodeOffsets_new[rank] && nodeTuple.nodes[0] < nodeOffsets_new[rank+1])
            edges_subdomain_owned.insert(ned);
          //pick up all the edges on the subdomain and store their nodes
          edgeNodesMap[ned].first = nodeTuple.nodes[0];
          edgeNodesMap[ned].second = nodeTuple.nodes[1];
          if (ihasEdgeMarkers > 0)
            edgeMaterialTypesMap[ned]=edId;
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
                  std::valarray<int> init(-1,6);
                  elementEdgesMap[eN] = init;
                  elementEdgesMap[eN][elementp->second] = ned;
                }
            }
        }
    }//end iv
  edgeFile.close();
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
      for (int iv=0;iv<4;iv++)
        {
          //the elementNodesArrayMap is in the old element numbering while the elementEdgesMap is in the new element numbering
          int nN_global = -1;
          if (elementNumbering_new2old_subdomain_map.find(edgesp->first) != elementNumbering_new2old_subdomain_map.end())
            nN_global = elementNodesArrayMap[elementNumbering_new2old_subdomain_map[edgesp->first]][iv];
          if (nN_global >= nodeOffsets_new[rank] && nN_global < nodeOffsets_new[rank+1])
            {
              //add all the edges to this node star
              for(int ed=0;ed<6;ed++)
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
  //Now as with elementBoundaries, build a global face numbering
  //resetting the edge based information is a little different since much of this is currently built below based
  //on the element and node information
  //
  valarray<int> edgeNumbering_new2old(edges_subdomain_owned.size());
  map<int,int> edgeNumbering_old2new_map;
  set<int>::iterator edN_ownedp=edges_subdomain_owned.begin();
  for (int edN=0;edN<int(edges_subdomain_owned.size());edN++,edN_ownedp++)
    {
      edgeNumbering_new2old[edN] = *edN_ownedp;
      edgeNumbering_old2new_map[*edN_ownedp] = edN+edgeOffsets_new[rank];
    }
  //
  //write this subdomain to the mapping file
  //
  logEvent("Writing/reading edge numberings to hdf5",5);
  hsize_t ed_dims[]={static_cast<hsize_t>(nEdges_global)};
  hid_t ed_new2old_filespace_id = H5Screate_simple(ARRAY_RANK, ed_dims, NULL);
  mappings_plist_id = H5Pcreate(H5P_FILE_ACCESS);   
  H5Pset_fapl_mpio(mappings_plist_id, PROTEUS_COMM_WORLD, MPI_INFO_NULL);
  file_id = H5Fopen(H5FILE_NAME, H5F_ACC_RDWR, mappings_plist_id);
  H5Pclose(mappings_plist_id);
  hid_t ed_new2old_dataset_id = H5Dcreate(file_id, "edgeNumbering_new2old", H5T_NATIVE_INT, 
                                          ed_new2old_filespace_id,
                                          H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  hsize_t ed_count[]={static_cast<hsize_t>(nEdges_subdomain_new[rank])};
  hsize_t ed_offset[]={static_cast<hsize_t>(edgeOffsets_new[rank])};
  hid_t ed_new2old_memspace_id = H5Screate_simple(ARRAY_RANK, ed_count, NULL);
  H5Sselect_hyperslab(ed_new2old_filespace_id, H5S_SELECT_SET, ed_offset, NULL, ed_count, NULL);
  hid_t ed_new2old_plist_id = H5Pcreate(H5P_DATASET_XFER);
  H5Pset_dxpl_mpio(ed_new2old_plist_id, H5FD_MPIO_COLLECTIVE);
  status = H5Dwrite(ed_new2old_dataset_id, H5T_NATIVE_INT, 
                    ed_new2old_memspace_id, ed_new2old_filespace_id,
                    ed_new2old_plist_id, &edgeNumbering_new2old[0]);
  H5Pclose(ed_new2old_plist_id);
  H5Sclose(ed_new2old_memspace_id);
  H5Sclose(ed_new2old_filespace_id);
  H5Dclose(ed_new2old_dataset_id);
  //do old2new map too, can't use hyperslab
  valarray<hsize_t> old_edge_indices(static_cast<hsize_t>(nEdges_subdomain_new[rank]));
  valarray<int> new_edge_indices(nEdges_subdomain_new[rank]);
  for (int i=0;i<nEdges_subdomain_new[rank];i++)
    {
      old_edge_indices[i] = static_cast<hsize_t>(edgeNumbering_new2old[i]);
      new_edge_indices[i] = edgeOffsets_new[rank]+i;
    }
  hid_t ed_old2new_filespace_id = H5Screate_simple(ARRAY_RANK, ed_dims, NULL);
  hid_t ed_old2new_dataspace_id = H5Dcreate(file_id, "edgeNumbering_old2new", H5T_NATIVE_INT, 
                                            ed_old2new_filespace_id,
                                            H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  hid_t ed_old2new_memspace_id = H5Screate_simple(ARRAY_RANK, ed_count, NULL);
  ed_old2new_filespace_id = H5Dget_space(ed_old2new_dataspace_id);
  status = H5Sselect_elements(ed_old2new_filespace_id, H5S_SELECT_SET, 
                              nEdges_subdomain_new[rank], &old_edge_indices[0]);
  hid_t ed_old2new_plist_id = H5Pcreate(H5P_DATASET_XFER);
  H5Pset_dxpl_mpio(ed_old2new_plist_id, H5FD_MPIO_COLLECTIVE);
  status = H5Dwrite(ed_old2new_dataspace_id, H5T_NATIVE_INT, 
                    ed_old2new_memspace_id, ed_old2new_filespace_id, 
                    ed_old2new_plist_id, &new_edge_indices[0]);
  H5Pclose(ed_old2new_plist_id);
  H5Sclose(ed_old2new_memspace_id);
  H5Dclose(ed_old2new_dataspace_id);
  H5Sclose(ed_old2new_filespace_id);
  H5Fclose(file_id);
  //
  //now get maps for all the edges on the subdomain, not just owned
  valarray<hsize_t> old_edge_indices_subdomain(edgeNodesMap.size());
  valarray<int> new_edge_indices_subdomain(edgeNodesMap.size());
  int edN_old_subdomain = 0;
  for (auto it = edgeNodesMap.begin(); it != edgeNodesMap.end(); it++, edN_old_subdomain++)
    {
      old_edge_indices_subdomain[edN_old_subdomain] = static_cast<hsize_t>(it->first);
    }
  file_id = openMappingsReadOnly(PROTEUS_COMM_WORLD, H5FILE_NAME);
  if (file_id < 0)
    {
      H5Eprint2(H5E_DEFAULT, stderr);
      SETERRABORT(PROTEUS_COMM_WORLD, PETSC_ERR_FILE_OPEN,
                  "could not reopen the mappings file; see the HDF5 stack above");
    }
  ed_old2new_dataspace_id = H5Dopen1(file_id, "/edgeNumbering_old2new");
  ed_old2new_filespace_id = H5Screate_simple(ARRAY_RANK, ed_dims, NULL);
  status = H5Sselect_elements(ed_old2new_filespace_id, H5S_SELECT_SET, 
                              edgeNodesMap.size(), &old_edge_indices_subdomain[0]);
  hsize_t ed_subdomain_count[]={static_cast<hsize_t>(edgeNodesMap.size())};
  hid_t ed_old2new_subdomain_memspace_id = H5Screate_simple(ARRAY_RANK, ed_subdomain_count, NULL);
  hid_t ed_old2new_subdomain_plist_id = H5Pcreate(H5P_DATASET_XFER);
  setMappingsReadXfer(ed_old2new_subdomain_plist_id);
  status = H5Dread(ed_old2new_dataspace_id, H5T_NATIVE_INT, ed_old2new_subdomain_memspace_id, 
                   ed_old2new_filespace_id, 
                   H5P_DEFAULT, &new_edge_indices_subdomain[0]);
  H5Pclose(ed_old2new_subdomain_plist_id);
  H5Sclose(ed_old2new_subdomain_memspace_id);
  H5Sclose(ed_old2new_filespace_id);
  H5Dclose(ed_old2new_dataspace_id);
  H5Fclose(file_id);
  map<int,int> edgeNumbering_old2new_subdomain_map;
  map<int,int> edgeNumbering_new2old_subdomain_map;
  for (int i=0;i<edgeNodesMap.size();i++)
    {
      edgeNumbering_old2new_subdomain_map[old_edge_indices_subdomain[i]] = new_edge_indices_subdomain[i];
      edgeNumbering_new2old_subdomain_map[new_edge_indices_subdomain[i]] = old_edge_indices_subdomain[i];
    }
  if (test)
  {
    IS edgeNumberingIS_subdomain_new2old;
    ISCreateGeneral(PROTEUS_COMM_WORLD,edges_subdomain_owned.size(),&edgeNumbering_new2old[0],PETSC_COPY_VALUES,&edgeNumberingIS_subdomain_new2old);
    IS edgeNumberingIS_global_new2old;
    ISAllGather(edgeNumberingIS_subdomain_new2old,&edgeNumberingIS_global_new2old);
    const PetscInt *edgeNumbering_global_new2old;
    ISGetIndices(edgeNumberingIS_global_new2old,&edgeNumbering_global_new2old);
    valarray<int> edgeNumbering_global_old2new(newMesh.nEdges_global);
    ierr = enforceMemoryLimit(PROTEUS_COMM_WORLD, rank, max_rss_gb,"Setting edgeNumering old2new/new2old");CHKERRABORT(PROTEUS_COMM_WORLD, ierr);
    for (int edN=0;edN<newMesh.nEdges_global;edN++)
      {
        edgeNumbering_global_old2new[edgeNumbering_global_new2old[edN]] = edN;
      }

    herr_t status;
    valarray<int> edgeNumbering_old2new_read(nEdges_global);
    file_id = openMappingsReadOnly(PROTEUS_COMM_WORLD, H5FILE_NAME);
    hid_t ed_old2new_dataset_id = H5Dopen1(file_id, "/edgeNumbering_old2new");
    status = H5Dread(ed_old2new_dataset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                     &edgeNumbering_old2new_read[0]);
    status = H5Dclose(ed_old2new_dataset_id);
    //assert() is compiled out by -DNDEBUG, so this verification has to
    //count and report rather than assert, or the whole block is cost
    //with no signal.
    {
      long bad_=0, first_=-1;
      for (int i=0;i<nEdges_global;i++)
        if (edgeNumbering_global_old2new[i] != edgeNumbering_old2new_read[i]) { if (first_<0) first_=i; bad_++; }
      if (bad_)
        {
          std::cerr<<"PROTEUS numbering check failed: edgeNumbering_global_old2new vs edgeNumbering_old2new_read, "
                   <<bad_<<" of "<<nEdges_global<<" differ, first at "<<first_<<std::endl;
          SETERRABORT(PROTEUS_COMM_WORLD, PETSC_ERR_PLIB,
                      "out-of-core numbering does not match in-core");
        }
    }
    std::cout<<"==================out of core old2new edges is correct!===================="<<std::endl;
    hid_t ed_new2old_dataset_id = H5Dopen1(file_id, "/edgeNumbering_new2old");
    valarray<int> edgeNumbering_new2old_read(nEdges_global);
    status = H5Dread(ed_new2old_dataset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                     &edgeNumbering_new2old_read[0]);
    status = H5Dclose(ed_new2old_dataset_id);
    status = H5Fclose(file_id);
    //assert() is compiled out by -DNDEBUG, so this verification has to
    //count and report rather than assert, or the whole block is cost
    //with no signal.
    {
      long bad_=0, first_=-1;
      for (int i=0;i<nEdges_global;i++)
        if (edgeNumbering_global_new2old[i] != edgeNumbering_new2old_read[i]) { if (first_<0) first_=i; bad_++; }
      if (bad_)
        {
          std::cerr<<"PROTEUS numbering check failed: edgeNumbering_global_new2old vs edgeNumbering_new2old_read, "
                   <<bad_<<" of "<<nEdges_global<<" differ, first at "<<first_<<std::endl;
          SETERRABORT(PROTEUS_COMM_WORLD, PETSC_ERR_PLIB,
                      "out-of-core numbering does not match in-core");
        }
    }
    std::cout<<"==================out of core new2old edges is correct!===================="<<std::endl;

    for (auto it = edgeNodesMap.begin(); it != edgeNodesMap.end(); ++it)
    {
      assert(edgeNumbering_global_old2new[it->first] == edgeNumbering_old2new_subdomain_map[it->first]);
    }
    std::cout<<"==================out of core edgeNumbering old2new subdomain is correct!===================="<<std::endl;
    for (int i = 0; i < edgeNodesMap.size(); i++)
    {
      int edN = new_edge_indices_subdomain[i];
      assert(edgeNumbering_global_new2old[edN] == edgeNumbering_new2old_subdomain_map[edN]);
    }
    std::cout<<"==================out of core edgeNumbering new2old subdomain is correct!===================="<<std::endl;
    ISRestoreIndices(edgeNumberingIS_global_new2old,&edgeNumbering_global_new2old);
    ierr=ISDestroy(&edgeNumberingIS_subdomain_new2old);CHKERRABORT(PROTEUS_COMM_WORLD, ierr);
    ierr=ISDestroy(&edgeNumberingIS_global_new2old);CHKERRABORT(PROTEUS_COMM_WORLD, ierr);
  }
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
  logEvent("USER WARNING: You are partioning in parallel directly from Tetgen files (parun -F ...). ",5);
  logEvent("USER WARNING: You must use the 'f' and 'ee' flags in the triangleOptions input to avoid errors.",5);
  logEvent("USER WARNING: You may have done so, but the tetgen options are not stored in the Tetgen files, so we can't know.",5);
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
  newMesh.subdomainp->nodeMaterialTypes = new int[newMesh.subdomainp->nNodes_global];
  //
  //now finally finish reading node coordinates and node flags
  //
  for (int iv = 0; iv < nNodes_global; iv++)
    {
      int nv; double x,y,z; int nodeId(0);
      vertexFile >> eatcomments >> nv >> x >> y >> z;
      if (hasVertexMarkers > 0)
        vertexFile >> nodeId;
      nv -= indexBase;
      assert(0 <= nv && nv < nNodes_global && vertexFile.good());
      int nN_global_new = -1;
      if (nodes_old2new_subdomain_map.find(nv) != nodes_old2new_subdomain_map.end())
        nN_global_new = nodes_old2new_subdomain_map[nv];
      //local
      if (nN_global_new >= nodeOffsets_new[rank] && nN_global_new < nodeOffsets_new[rank+1])
        {
          int nv_subdomain_new = nN_global_new - nodeOffsets_new[rank];
          nodeNumbering_subdomain2global[nv_subdomain_new] = nN_global_new;
          nodeNumbering_global2subdomainMap[nN_global_new] = nv_subdomain_new;
          newMesh.subdomainp->nodeArray[vertexDim*nv_subdomain_new + 0] = x;
          newMesh.subdomainp->nodeArray[vertexDim*nv_subdomain_new + 1] = y;
          newMesh.subdomainp->nodeArray[vertexDim*nv_subdomain_new + 2] = z;
          if (hasVertexMarkers > 0)
            newMesh.subdomainp->nodeMaterialTypes[nv_subdomain_new] = nodeId;
        }
      //overlap
      if (nodes_overlap.count(nN_global_new) == 1)
        {
          int nv_subdomain_new = nodes_overlap_global2subdomainMap[nN_global_new];
          nodeNumbering_subdomain2global[nv_subdomain_new] = nN_global_new;
          nodeNumbering_global2subdomainMap[nN_global_new] = nv_subdomain_new;
          newMesh.subdomainp->nodeArray[vertexDim*nv_subdomain_new + 0] = x;
          newMesh.subdomainp->nodeArray[vertexDim*nv_subdomain_new + 1] = y;
          newMesh.subdomainp->nodeArray[vertexDim*nv_subdomain_new + 2] = z;
          if (hasVertexMarkers > 0)
            newMesh.subdomainp->nodeMaterialTypes[nv_subdomain_new] = nodeId;
        }
      vertexFile >> eatline;
    }//end iv
  vertexFile.close();
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
      int eN_global_old = elementNumbering_subdomain_new2old[eN];
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
  constructElementBoundaryElementsArrayWithGivenElementBoundaryAndEdgeNumbers_tetrahedron(*newMesh.subdomainp);
  //build local geometric info
  allocateGeometricInfo_tetrahedron(*newMesh.subdomainp);
  computeGeometricInfo_tetrahedron(*newMesh.subdomainp);
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
      if (!hasVertexMarkers)
        for (int ebNE = 0; ebNE < newMesh.subdomainp->nExteriorElementBoundaries_global; ebNE++)
          {
            int ebN = newMesh.subdomainp->exteriorElementBoundariesArray[ebNE];
            for (int nN_local = 0; nN_local < newMesh.subdomainp->nNodes_elementBoundary; nN_local++)
              {
                int nN = newMesh.subdomainp->elementBoundaryNodesArray[ebN*newMesh.subdomainp->nNodes_elementBoundary+nN_local];
                newMesh.subdomainp->nodeMaterialTypes[nN] = newMesh.subdomainp->elementBoundaryMaterialTypes[ebN];
              }
          }
    }
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
  //cleanup
  PetscLogEventEnd(build_subdomains_cleanup_event,0,0,0,0);
  PetscLogStagePop();
  PetscLogView(PETSC_VIEWER_STDOUT_WORLD);
  ierr = enforceMemoryLimit(PROTEUS_COMM_WORLD, rank, max_rss_gb,"Done with partitioning!");CHKERRABORT(PROTEUS_COMM_WORLD, ierr);
  return 0;
}
/* I/O, local: stream the Triangle element file and keep what a rank whose
   default chunk is nodes [begin,end) needs to partition them: the star (the
   other nodes of every element containing it) of each of its nodes. Reads
   every element and keeps O(chunk) data. Returns false, having said why, if
   the file cannot be opened. */
bool readTriangleNodeStars(const std::string& elementFileName, int indexBase, int nNodes_global,
                           int begin, int end, int& nElements_global,
                           std::vector<std::set<int> >& nodeStar)
{
  using namespace std;
  using namespace IOutils;
  const int simplexDim = 3;
  std::ifstream elementFile(elementFileName.c_str());
  if (!elementFile.good())
    {
      std::cerr<<"cannot open triangle element file "
               <<elementFileName<<std::endl;
      return false;
    }
  //read elements
  int nNodesPerSimplex(simplexDim),hasElementMarkers = 0;
  elementFile >> eatcomments >> nElements_global >> nNodesPerSimplex >> hasElementMarkers >> eatline;
  assert(nElements_global > 0);
  assert(nNodesPerSimplex == simplexDim);
  vector<int> element_nodes_old(3);
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

int partitionNodesFromTriangleFiles(const MPI_Comm& PROTEUS_COMM_WORLD, const char* filebase, int indexBase, Mesh& newMesh, int nNodes_overlap)
{
  using namespace std;
  PetscErrorCode ierr;
  PetscMPIInt size,rank;

  ierr = MPI_Comm_size(PROTEUS_COMM_WORLD,&size);CHKERRABORT(PROTEUS_COMM_WORLD, ierr);
  ierr = MPI_Comm_rank(PROTEUS_COMM_WORLD,&rank);CHKERRABORT(PROTEUS_COMM_WORLD, ierr);
  PetscLogStage partitioning_stage;
  PetscLogStageRegister("Mesh Partition",&partitioning_stage);
  PetscLogStagePush(partitioning_stage);

  /***********************************************************************
    partition domain based on the nodes without reading in the global mesh.

    1. Partition nodes in the default partition of contiguous chunks with
       input ordering

    2. Determine nodal connectivity on local processor

    3. Generate new nodal partition using PETSc interface

    4. Collect subdomain elements: any that contain owned nodes; tag ownership

    4a,b. Collect element boundaries and edges on the subdomain elements and tag ownership

    5. Generate global element and edge numbering for new subdomain ownership

    6. Create overlap (ghost) information for nodes and elements

    7. March through additional layers of overlap if requested,

    8. Build subdomain meshes in new numbering
  ***********************************************************************/
  //
  //0. Set up triangle files. Note, triangle should have been run with -en to get all edges and neighbors
  //
  bool failed = false;
  const int simplexDim = 3;
  const int vertexDim  = 3;
  using namespace IOutils;
  std::string vertexFileName  = std::string(filebase) + ".node" ;
  std::string elementFileName = std::string(filebase) + ".ele" ;
  std::string elementBoundaryFileName  = std::string(filebase) + ".edge" ;
  std::string edgeFileName  = std::string(filebase) + ".edge" ;

  //
  //1. Build default nodal partition
  //
  //compute offsets to build subdomain2global ordering for nodes
  //in default partitioning of the domain into subdomains. All
  //we need is the number of nodes in the global mesh
  //
  //read nodes for triangle format
  //first just read the number of nodes and whether or not there are node tags
  int read_elements_event;
  PetscLogEventRegister("Read eles",0,&read_elements_event);
  PetscLogEventBegin(read_elements_event,0,0,0,0);
  std::ifstream vertexFile(vertexFileName.c_str());
  if (!vertexFile.good())
    {
      std::cerr<<"cannot open triangle node file "
               <<vertexFileName<<std::endl;
      failed = true;
      return failed;
    }
  int hasVertexMarkers(0),hasVertexAttributes(0),nSpace(2),nNodes_global;
  //read number of vertices and whether node flags are provided
  vertexFile >> eatcomments >> nNodes_global >> nSpace >> hasVertexAttributes >> hasVertexMarkers >> eatline ;
  assert(nNodes_global > 0);
  assert(nSpace == 2);
  newMesh.nNodes_global = nNodes_global;
  newMesh.nNodes_element = simplexDim;
  newMesh.nNodes_elementBoundary = simplexDim-1;
  newMesh.nElementBoundaries_element = simplexDim;
  if (hasVertexAttributes > 0)
    {
      std::cerr<<"WARNING triangle nodes hasAttributes= "<<hasVertexAttributes
               <<" > 0 will treat first value as integer id for boundary!!"<<std::endl;
      hasVertexMarkers = 1;
    }
  //don't need to read anymore from nodes for now

  //offsets provide the lower and upper bounds for the global numbering
  //first we just partition the nodes approximately equally ignoring connectivity
  valarray<int> nodeOffsets_old(size+1);
  nodeOffsets_old[0] = 0;
  for (int sdN=0; sdN < size; sdN++)
    {
      nodeOffsets_old[sdN+1] = nodeOffsets_old[sdN] +
        int(nNodes_global)/size + (int(nNodes_global)%size > sdN);
    }
  int nNodes_subdomain_old = nodeOffsets_old[rank+1] - nodeOffsets_old[rank];

  //
  //2. Determine nodal connectivity (nodeStarArray) for nodes on subdomain
  //
  //connectivty commes from the topology (elements) file. We just grab the elements
  //that contain currently owned nodes
  int nNodesPerSimplex(simplexDim),hasElementMarkers = 0,nElements_global;
  vector<int> element_nodes_old(3);
  vector<set<int> > nodeStar;
  if (!readTriangleNodeStars(elementFileName,indexBase,nNodes_global,
                             nodeOffsets_old[rank],nodeOffsets_old[rank+1],
                             nElements_global,nodeStar))
    {
      failed = true;
      return failed;
    }
  newMesh.nElements_global = nElements_global;
  PetscLogEventEnd(read_elements_event,0,0,0,0);
  int repartition_nodes_event;
  PetscLogEventRegister("Repart nodes",0,&repartition_nodes_event);
  PetscLogEventBegin(repartition_nodes_event,0,0,0,0);
  //done reading element file for first time; will need to read again after node partitioning
  //build compact data structure for nodeStar
  valarray<int> nodeStarOffsets(nNodes_subdomain_old+1);
  nodeStarOffsets[0] = 0;
  for (int nN=1;nN<nNodes_subdomain_old+1;nN++)
    nodeStarOffsets[nN] = nodeStarOffsets[nN-1] + nodeStar[nN-1].size();
  valarray<int> nodeStarArray(nodeStarOffsets[nNodes_subdomain_old]);
  for (int nN=0,offset=0;nN<nNodes_subdomain_old;nN++)
    for (set<int>::iterator nN_star=nodeStar[nN].begin();nN_star!=nodeStar[nN].end();nN_star++,offset++)
      nodeStarArray[offset] = *nN_star;
  //find maximum number of nodes in any star
  int max_nNodeNeighbors_node=0;
  for (int nN=0;nN<nNodes_subdomain_old;nN++)
    max_nNodeNeighbors_node=max(max_nNodeNeighbors_node,nodeStarOffsets[nN+1]-nodeStarOffsets[nN]);
  //build connectivity data structures for PETSc
  PetscBool isInitialized;
  PetscInitialized(&isInitialized);
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
  //I think we can simplify this now that nodeStarArray is local to the subdomain, could just use nodeStar instead of nodeStarArray
  for (int nN = 0,offset=0; nN < nNodes_subdomain_old; nN++)
    {
      for (int offset_subdomain = nodeStarOffsets[nN];
           offset_subdomain < nodeStarOffsets[nN+1];
           offset_subdomain++)
        {
          nodeNeighbors_subdomain[offset++] = nodeStarArray[offset_subdomain];
        }
      nodeNeighborsOffsets_subdomain[nN+1]=offset;
      sort(&nodeNeighbors_subdomain[nodeNeighborsOffsets_subdomain[nN]],&nodeNeighbors_subdomain[nodeNeighborsOffsets_subdomain[nN+1]]);
      //weight nodes by size of star
      int weight= (nodeNeighborsOffsets_subdomain[nN+1] - nodeNeighborsOffsets_subdomain[nN]);
      vertex_weights_subdomain[nN] = weight;
      for (int k=nodeNeighborsOffsets_subdomain[nN];k<nodeNeighborsOffsets_subdomain[nN+1];k++)
        weights_subdomain[k] = weight;
    }
  //
  //3. Generate new nodal partition using PETSc interface
  //
  Mat petscAdjacency;
  int nNodes_subdomain_max=0;
  MPI_Allreduce(&nNodes_subdomain_old,
                &nNodes_subdomain_max,
                1,
                MPI_INT,
                MPI_MAX,
                PROTEUS_COMM_WORLD);
  if (rank ==  0)
    std::cout<<"Max nNodes_subdomain "<<nNodes_subdomain_max<<" nNodes_global "<<nNodes_global<<std::endl;
  ierr = MatCreateMPIAdj(PROTEUS_COMM_WORLD,
                         nNodes_subdomain_old,
                         nNodes_global,
                         nodeNeighborsOffsets_subdomain,
                         nodeNeighbors_subdomain,
                         weights_subdomain,
                         &petscAdjacency);CHKERRABORT(PROTEUS_COMM_WORLD, ierr);
  //const double max_rss_gb(0.75*3.25);//half max mem per  core  on topaz
  const double max_rss_gb(10.0);
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
  //ierr = enforceMemoryLimit(PROTEUS_COMM_WORLD, rank, max_rss_gb,"Done applying partition");CHKERRABORT(PROTEUS_COMM_WORLD, ierr);

  //determine the number of nodes per subdomain in new partitioning
  valarray<int> nNodes_subdomain_new(size);
  ISPartitioningCount(nodePartitioningIS_new,size,&nNodes_subdomain_new[0]);

  //need new offsets for subdomain to global numbering
  valarray<int> nodeOffsets_new(size+1);
  nodeOffsets_new[0] = 0;
  for (int sdN = 0; sdN < size; sdN++)
    {
      nodeOffsets_new[sdN+1] = nodeOffsets_new[sdN] + nNodes_subdomain_new[sdN];
    }

  //get the new node numbers for nodes on this subdomain
  IS nodeNumberingIS_subdomain_old2new;
  ISPartitioningToNumbering(nodePartitioningIS_new,&nodeNumberingIS_subdomain_old2new);
  //
  //try out of core
  //
  /*
   * Set up file access property list with parallel I/O access
   */
  hid_t plist_id = H5Pcreate(H5P_FILE_ACCESS);
#ifdef H5_HAVE_PARALLEL
  H5Pset_fapl_mpio(plist_id, PROTEUS_COMM_WORLD, MPI_INFO_NULL);
#endif

  /*
   * Create a new file collectively and release property list identifier.
   */
  const char* H5FILE_NAME("mappings.h5");
  hid_t file_id = H5Fcreate(H5FILE_NAME, H5F_ACC_TRUNC, H5P_DEFAULT, plist_id);
  H5Pclose(plist_id);


  /*
   * Create the dataspace for the dataset.
   */
  hsize_t     dimsf[1];
  dimsf[0] = nNodes_global;
#define RANK   1
  hid_t filespace = H5Screate_simple(RANK, dimsf, NULL);

  /*
   * Create the dataset with default properties and close filespace.
   */
  hid_t dset_id = H5Dcreate(file_id, "nodeNumbering_old2new", H5T_NATIVE_INT, filespace,
                            H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  H5Sclose(filespace);

  /*
   * Each process defines dataset in memory and writes it to the hyperslab
   * in the file.
   */
  hsize_t       count[1];                 /* hyperslab selection parameters */
  hsize_t       offset[1];
  count[0] = nNodes_subdomain_old;
  offset[0] = nodeOffsets_old[rank];
  hid_t memspace = H5Screate_simple(RANK, count, NULL);

  /*
   * Select hyperslab in the file.
   */
  filespace = H5Dget_space(dset_id);
  H5Sselect_hyperslab(filespace, H5S_SELECT_SET, offset, NULL, count, NULL);

  /*
   * Initialize data buffer
   */
  // data = (int *) malloc(sizeof(int)*count[0]*count[1]);
  // for (i=0; i < count[0]*count[1]; i++) {
  //   data[i] = mpi_rank + 10;
  // }
  const PetscInt* data;
  ISGetIndices(nodeNumberingIS_subdomain_old2new, &data);

  /*
   * Create property list for collective dataset write.
   */
  plist_id = H5Pcreate(H5P_DATASET_XFER);
#ifdef H5_HAVE_PARALLEL
  H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);
#endif

  herr_t status = H5Dwrite(dset_id, H5T_NATIVE_INT, memspace, filespace,
                           plist_id, data);
  //free(data);
  ISRestoreIndices(nodeNumberingIS_subdomain_old2new, &data);
  /*
   * Close/release resources.
   */
  H5Dclose(dset_id);
  //Close the write handle here. It used to stay open until the end of the
  //function, so the read-only reopen below met a file that was still open
  //for write and H5Fopen refused it -- PETSC_ERR_FILE_OPEN, MPI_Abort(65) on
  //every rank within 20 s. Closing also flushes, which the read needs.
  H5Fclose(file_id);
  //
  //end try out of core
  //
  //collect new node numbers for whole mesh so that subdomain reordering and renumbering
  //can be done easily

  //The ISAllGather that used to build nodeNumbering_global_old2new here is
  //gone: it put the whole global node numbering on every rank. The element
  //loop below now reads old->new in chunks and keeps only
  //nodes_old2new_subdomain_map, the nodes this rank's elements actually touch.
  //
  //test out of core
  //
  //(the scratch out-of-core comparison block is gone with the gather it
  //compared against)
  //
  //end test out of core
  //
  //reverse mapping for node numbers too
  //cek hack, not needed
  /*
    valarray<int> nodeNumbering_global_new2old(nNodes_global);
    for (int nN = 0; nN < nNodes_global; nN++)
    nodeNumbering_global_new2old[nodeNumbering_global_old2new[nN]] = nN;
  */
  PetscLogEventEnd(repartition_nodes_event,0,0,0,0);
  int receive_element_mask_event;
  PetscLogEventRegister("Recv. ele mask",0,&receive_element_mask_event);
  PetscLogEventBegin(receive_element_mask_event,0,0,0,0);
  //
  //4. To build subdomain meshes, go through and collect elements containing
  //   the locally owned nodes. Assign processor ownership of elements
  //
  PetscLogEventEnd(receive_element_mask_event,0,0,0,0);
  //ierr = enforceMemoryLimit(PROTEUS_COMM_WORLD, rank, max_rss_gb,"Done with masks");CHKERRABORT(PROTEUS_COMM_WORLD, ierr);
  int build_subdomains_reread_elements_event;
  PetscLogEventRegister("Reread eles",0,&build_subdomains_reread_elements_event);
  PetscLogEventBegin(build_subdomains_reread_elements_event,0,0,0,0);
  //
  //mark the unmarked elements on this subdomain and store element numbers (in old numbering)
  //
  //We need to find the elements in the support of nodes in the new
  //partitioning, so we'll need to re-read the elements file to get
  //the the elements for the nodes in the new partitioning. We will be collecting OLD element numbers.
  std::ifstream elementFile2(elementFileName.c_str());

  if (!elementFile2.good())
    {
      std::cerr<<"cannot open triangle elements file"
               <<elementFileName<<std::endl;
      failed = true;
      return failed;
    }
  elementFile2 >> eatcomments >> nElements_global >> nNodesPerSimplex >> hasElementMarkers >> eatline;
  assert(nElements_global > 0);
  assert(nNodesPerSimplex == simplexDim);
  set<int> elements_subdomain_owned;
  vector<int> element_nodes_new(3);
  int element_nodes_new_array[3];
  vector<set<int> > nodeElementsStar(nNodes_subdomain_new[rank]);
  vector<set<int> > nodeStarNew(nNodes_subdomain_new[rank]);
  map<int,vector<int> > elementNodesArrayMap;
  map<int,long int> elementMaterialTypesMap;
  map<NodeTuple<2>,ElementNeighbors> elementBoundaryElementsMap;
  map<NodeTuple<2>,set<pair<int,int> > > edgeElementsMap;

  //note any element index containers are in the old element numbering
  //
  //Out-of-core node numbering, mirroring partitionNodesFromTetgenFiles.
  //
  //This loop used to index nodeNumbering_global_old2new, a global array that
  //ISAllGather had put on EVERY rank. For a 2D mesh with N nodes that is N
  //ints per rank here, and the element, elementBoundary and edge numberings
  //add roughly 13N more -- so per-rank memory did not fall as ranks were
  //added. Measured on damBreak (569508 nodes): peak per core was FLAT at
  //0.26 GB from 384 to 768 ranks while the subdomain halved.
  //
  //Instead, accumulate a chunk of elements, read old->new for just the nodes
  //that chunk touches, process it, and release. The chunk is capped by NODE
  //COUNT, not by nElements_global/size, so the number of chunks does not grow
  //with the rank count -- the mistake 77c99396 made in the tetgen path.
  const int nNodes_collection_max=10000;
  set<int> node_collection;
  vector<int> elements_collection;
  vector<valarray<int> > element_old_nodes_collection;
  vector<double> elementId_collection;
  map<int,int> nodes_old2new_subdomain_map;
  int eN_c_start = 0;
  file_id = openMappingsReadOnly(PROTEUS_COMM_WORLD, H5FILE_NAME);
  if (file_id < 0)
    {
      H5Eprint2(H5E_DEFAULT, stderr);
      SETERRABORT(PROTEUS_COMM_WORLD, PETSC_ERR_FILE_OPEN,
                  "could not reopen the mappings file; see the HDF5 stack above");
    }
  for (int ie = 0; ie < nElements_global; ie++)
    {
      int ne, nv;
      long double elementId_double;
      elementFile2 >> eatcomments >> ne;
      ne -= indexBase;
      assert(0 <= ne && ne < nElements_global && elementFile2.good());
      elements_collection.push_back(ne);
      valarray<int> enodes_PROBE(simplexDim);
      for (int iv = 0; iv < simplexDim; iv++)
        {
          elementFile2 >> nv;
          nv -= indexBase;
          assert(0 <= nv && nv < nNodes_global);
          node_collection.insert(nv);
          enodes_PROBE[iv] = nv;
        }
      element_old_nodes_collection.push_back(enodes_PROBE);
      //read the marker unconditionally: the stream position must not depend on
      //whether this element turns out to be in the subdomain
      if (hasElementMarkers > 0)
        {
          elementFile2 >> elementId_double;
          elementId_collection.push_back(static_cast<double>(elementId_double));
        }
      elementFile2 >> eatline;

      if (node_collection.size() >= (size_t)nNodes_collection_max || ie == nElements_global-1)
        {
          //one point-selection read for the whole chunk
          valarray<hsize_t> node_collection_array(node_collection.size());
          valarray<int> nodes_old2new_subset(node_collection.size());
          int i_nc=0;
          for (auto nvp=node_collection.begin();nvp!=node_collection.end();nvp++,i_nc++)
            node_collection_array[i_nc] = static_cast<hsize_t>(*nvp);
          hid_t dset_id_c = H5Dopen1(file_id,"/nodeNumbering_old2new");
          if (dset_id_c < 0)
            {
              H5Eprint2(H5E_DEFAULT, stderr);
              SETERRABORT(PROTEUS_COMM_WORLD, PETSC_ERR_FILE_READ,
                          "could not open /nodeNumbering_old2new");
            }
          hid_t fspace_c = H5Dget_space(dset_id_c);
          herr_t st_c = H5Sselect_elements(fspace_c, H5S_SELECT_SET,
                                           node_collection.size(), &node_collection_array[0]);
          hsize_t dims_c[] = {static_cast<hsize_t>(node_collection.size())};
          hid_t mspace_c = H5Screate_simple(1, dims_c, NULL);
          herr_t rd_c = H5Dread(dset_id_c, H5T_NATIVE_INT, mspace_c, fspace_c,
                                H5P_DEFAULT, &nodes_old2new_subset[0]);
          if (st_c < 0 || rd_c < 0)
            {
              H5Eprint2(H5E_DEFAULT, stderr);
              SETERRABORT(PROTEUS_COMM_WORLD, PETSC_ERR_FILE_READ,
                          "could not read /nodeNumbering_old2new for this chunk");
            }
          H5Sclose(mspace_c); H5Sclose(fspace_c); H5Dclose(dset_id_c);
          map<int,int> nodes_old2new_subset_map;
          for (size_t i=0;i<node_collection.size();i++)
            nodes_old2new_subset_map[node_collection_array[i]] = nodes_old2new_subset[i];

          for (int eN_c = eN_c_start; eN_c <= ie; eN_c++)
            {
              const int ne = elements_collection[eN_c-eN_c_start];
              long int elementId(0);
              for (int iv = 0; iv < simplexDim; iv++)
                {
                  element_nodes_old[iv] = element_old_nodes_collection[eN_c-eN_c_start][iv];
                  element_nodes_new[iv] = nodes_old2new_subset_map[element_nodes_old[iv]];
                  element_nodes_new_array[iv] = element_nodes_new[iv];
                }
            NodeTuple<simplexDim> nodeTuple(element_nodes_new_array);
            for (int iv = 0; iv < simplexDim; iv++)
              {
                int nN_star_new = element_nodes_new[iv];
                bool inSubdomain=false;
                if (nN_star_new >= nodeOffsets_new[rank] && nN_star_new < nodeOffsets_new[rank+1])
                  {
                    inSubdomain = true;
                    //add all the element boundaries of this element
                    for (int ebN=0;ebN < simplexDim ; ebN++)
                      {
                        int nodes[simplexDim-1] = { element_nodes_new[(ebN+1) % simplexDim],
                                         element_nodes_new[(ebN+2) % simplexDim]};
                        NodeTuple<simplexDim-1> nodeTuple(nodes);
                        if(elementBoundaryElementsMap.find(nodeTuple) != elementBoundaryElementsMap.end())
                          {
                            if (elementBoundaryElementsMap[nodeTuple].right == -1 && ne != elementBoundaryElementsMap[nodeTuple].left)
                              {
                                elementBoundaryElementsMap[nodeTuple].right=ne;
                                elementBoundaryElementsMap[nodeTuple].right_ebN_element=ebN;
                              }
                          }
                        else
                          {
                            elementBoundaryElementsMap[nodeTuple] = ElementNeighbors(ne,ebN);
                          }
                      }
                    //add all the edges of this element
                    for (int nNL=0,edN=0;nNL < simplexDim ; nNL++)
                      for(int nNR=nNL+1;nNR < simplexDim;nNR++,edN++)
                        {
                          int nodes[2] = { element_nodes_new[nNL],
                                           element_nodes_new[nNR]};
                          NodeTuple<2> nodeTuple(nodes);
                          edgeElementsMap[nodeTuple].insert(pair<int,int>(ne,edN));
                        }
                    //add all the nodes to the node star
                    int nN_star_new_subdomain = nN_star_new - nodeOffsets_new[rank];
                    nodeElementsStar[nN_star_new_subdomain].insert(ne);
                    for (int jv = 0; jv < simplexDim; jv++)
                      {
                        if (iv != jv)
                          {
                            int nN_point_new = element_nodes_new[jv];
                            nodeStarNew[nN_star_new_subdomain].insert(nN_point_new);
                          }
                      }
                  }
                if (inSubdomain)
                  {
                    elementNodesArrayMap[ne] = element_nodes_new;
                    //the later .poly/.edge readers need old->new for exactly these
                    //nodes, which is what replaces the global ISAllGather
                    for (int jv = 0; jv < simplexDim; jv++)
                      nodes_old2new_subdomain_map[element_nodes_old[jv]] = element_nodes_new[jv];
                  }
              }
            if (elementNodesArrayMap.find(ne) != elementNodesArrayMap.end())//this element contains a node owned by this subdomain
              {
                if (nodeTuple.nodes[1] >= nodeOffsets_new[rank] && nodeTuple.nodes[1] < nodeOffsets_new[rank+1])
                  elements_subdomain_owned.insert(ne);
                if (hasElementMarkers > 0)
                  {
                    elementId = static_cast<long int>(elementId_collection[eN_c-eN_c_start]);
                    elementMaterialTypesMap[ne] = elementId;
                  }
              }
            }
          eN_c_start = ie+1;
          node_collection.clear();
          elements_collection.clear();
          elementId_collection.clear();
          element_old_nodes_collection.clear();
        }
    }
  H5Fclose(file_id);
  elementFile2.close();
  int nElements_owned_subdomain(elements_subdomain_owned.size()),
    nElements_owned_new=0;
  MPI_Allreduce(&nElements_owned_subdomain,&nElements_owned_new,1,MPI_INT,MPI_SUM,PROTEUS_COMM_WORLD);
  assert(nElements_owned_new == nElements_global);
  PetscLogEventEnd(build_subdomains_reread_elements_event,0,0,0,0);
  int build_subdomains_send_marked_elements_event;
  PetscLogEventRegister("Mark/send eles",0,&build_subdomains_send_marked_elements_event);
  PetscLogEventBegin(build_subdomains_send_marked_elements_event,0,0,0,0);
  //ierr = enforceMemoryLimit(PROTEUS_COMM_WORLD, rank, max_rss_gb,"Done marking elements");CHKERRABORT(PROTEUS_COMM_WORLD, ierr);
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
  PetscLogEventEnd(build_subdomains_send_marked_elements_event,0,0,0,0);
  int build_subdomains_global_numbering_elements_event;
  PetscLogEventRegister("Global ele nmbr",0,&build_subdomains_global_numbering_elements_event);
  PetscLogEventBegin(build_subdomains_global_numbering_elements_event,0,0,0,0);
  //
  //5. Generate global element numbering corresponding to new subdomain ownership
  //
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
  //construct new offsets for elements
  elementOffsets_new[0] = 0;
  for (int sdN = 0; sdN < size; sdN++)
    elementOffsets_new[sdN+1] = elementOffsets_new[sdN] + nElements_subdomain_new[sdN];
  //map to old element numbering
  valarray<int> elementNumbering_subdomain_new2old(elements_subdomain_owned.size());
  set<int>::iterator eN_ownedp = elements_subdomain_owned.begin();
  for (int eN = 0; eN < int(elements_subdomain_owned.size()); eN++,eN_ownedp++)
    {
      elementNumbering_subdomain_new2old[eN] = *eN_ownedp;
    }
  //use Petsc IS to get global new2old numbering
  IS elementNumberingIS_subdomain_new2old;
  ISCreateGeneral(PROTEUS_COMM_WORLD,elements_subdomain_owned.size(),&elementNumbering_subdomain_new2old[0],PETSC_COPY_VALUES,
                  &elementNumberingIS_subdomain_new2old);
  IS elementNumberingIS_global_new2old;
  ISAllGather(elementNumberingIS_subdomain_new2old,&elementNumberingIS_global_new2old);

  const PetscInt *elementNumbering_global_new2old;//needs to be restored
  ISGetIndices(elementNumberingIS_global_new2old,&elementNumbering_global_new2old);
  //construct reverse mapping
  valarray<int> elementNumbering_global_old2new(nElements_global);
  for (int eN = 0; eN < nElements_global; eN++)
    {
      elementNumbering_global_old2new[elementNumbering_global_new2old[eN]] = eN;
    }
  PetscLogEventEnd(build_subdomains_global_numbering_elements_event,0,0,0,0);
  //ierr = enforceMemoryLimit(PROTEUS_COMM_WORLD, rank, max_rss_gb,"Done allocating element numbering new2old/old2new");CHKERRABORT(PROTEUS_COMM_WORLD, ierr);
  int build_subdomains_faces_event;
  PetscLogEventRegister("Subd faces",0,&build_subdomains_faces_event);
  PetscLogEventBegin(build_subdomains_faces_event,0,0,0,0);
  //
  //4b,5b. repeat process to build global face (elementBoundary) numbering
  //
  //first read element boundaries to create nodeElementBoundariesArray
  //for all element boundaries on this subdomain, which we'll use to
  //grab element boundaries from the bit array


  //In 2D, the element boundaries are edges and not faces.
  //nodeElementBoundariesArray maps an element

  std::ifstream elementBoundaryFile(elementBoundaryFileName.c_str());

  if (!elementBoundaryFile.good())
    {
      std::cerr<<"cannot open triangle edge file "
               <<elementBoundaryFileName<<std::endl;
      failed = true;
      return failed;
    }

  bool hasElementBoundaryMarkers = false;
  int nElementBoundaries_global;
  int ihasElementBoundaryMarkers(0);

  elementBoundaryFile >> eatcomments >> nElementBoundaries_global >>ihasElementBoundaryMarkers >> eatline ;
  assert(nElementBoundaries_global > 0);
  if (ihasElementBoundaryMarkers > 0)
    {
      hasElementBoundaryMarkers = true;
    }
  newMesh.nElementBoundaries_global = nElementBoundaries_global;
  //note, these will be in the new element numbering
  set<int> elementBoundaries_subdomain_owned;
  vector<set<int> > nodeElementBoundariesStar(nNodes_subdomain_new[rank]);
  map<int,int> elementBoundaryMaterialTypesMap;
  map<int,vector<int> > elementBoundariesMap;
  set<int> supportedElementBoundaries;
  for (int ieb = 0; ieb < nElementBoundaries_global; ieb++)
    {
      int neb,nn0,nn1; int ebId(0);
      elementBoundaryFile >> eatcomments >> neb >> nn0 >> nn1;
      if (ihasElementBoundaryMarkers > 0)
        elementBoundaryFile >> ebId;
      neb -= indexBase;
      nn0 -= indexBase;
      nn1 -= indexBase;
      assert(0 <= neb && neb < nElementBoundaries_global && elementBoundaryFile.good());
      //grab the element boundaries for the node if the node is owned by the subdomain
      //this will miss the element boundaries on the "outside boundary" of the star, which will grab later
      int nn0_new = -1;
      if (nodes_old2new_subdomain_map.find(nn0) != nodes_old2new_subdomain_map.end())
        nn0_new = nodes_old2new_subdomain_map[nn0];
      if (nn0_new >= nodeOffsets_new[rank] && nn0_new < nodeOffsets_new[rank+1])
        {
          nodeElementBoundariesStar[nn0_new-nodeOffsets_new[rank]].insert(neb);
          supportedElementBoundaries.insert(neb);
        }
      int nn1_new = -1;
      if (nodes_old2new_subdomain_map.find(nn1) != nodes_old2new_subdomain_map.end())
        nn1_new = nodes_old2new_subdomain_map[nn1];
      if (nn1_new >= nodeOffsets_new[rank] && nn1_new < nodeOffsets_new[rank+1])
        {
          nodeElementBoundariesStar[nn1_new-nodeOffsets_new[rank]].insert(neb);
          supportedElementBoundaries.insert(neb);
        }
      int nodes[2] = {nn0_new,nn1_new};
      NodeTuple<2> nodeTuple(nodes);
      elementBoundaryFile >> eatline;
      if (elementBoundaryElementsMap.find(nodeTuple) != elementBoundaryElementsMap.end())//this element boundary is on an element in the subdomain
        {
          if (nodeTuple.nodes[1] >= nodeOffsets_new[rank] && nodeTuple.nodes[1] < nodeOffsets_new[rank+1])
            elementBoundaries_subdomain_owned.insert(neb);
          if (ihasElementBoundaryMarkers > 0)
            elementBoundaryMaterialTypesMap[neb]=ebId;
          int eN_left = elementNumbering_global_old2new[elementBoundaryElementsMap[nodeTuple].left];
          if (elementBoundariesMap.find(eN_left) != elementBoundariesMap.end())
            {
              elementBoundariesMap[eN_left][elementBoundaryElementsMap[nodeTuple].left_ebN_element] = neb;
            }
          else
            {
              //initialize
              vector<int> elementBoundaries_element(3,-1);
              elementBoundariesMap[eN_left] = elementBoundaries_element;
              //assign
              elementBoundariesMap[eN_left][elementBoundaryElementsMap[nodeTuple].left_ebN_element] = neb;
            }
          if (elementBoundaryElementsMap[nodeTuple].right >= 0)
            {
              int eN_right = elementNumbering_global_old2new[elementBoundaryElementsMap[nodeTuple].right];
              if (elementBoundariesMap.find(eN_right) != elementBoundariesMap.end())
                {
                  elementBoundariesMap[eN_right][elementBoundaryElementsMap[nodeTuple].right_ebN_element] = neb;
                }
              else
                {
                  //initialize
                  vector<int> elementBoundaries_element(3,-1);
                  elementBoundariesMap[eN_right] = elementBoundaries_element;
                  //assign
                  elementBoundariesMap[eN_right][elementBoundaryElementsMap[nodeTuple].right_ebN_element] = neb;
                }
            }
        }
    }
  elementBoundaryElementsMap.clear();
  //done reading element boundaries
  elementBoundaryFile.close();
  //ierr = enforceMemoryLimit(PROTEUS_COMM_WORLD, rank, max_rss_gb,"Done reading element boundaries");CHKERRABORT(PROTEUS_COMM_WORLD, ierr);
  int nElementBoundaries_owned_subdomain=elementBoundaries_subdomain_owned.size(),
    nElementBoundaries_owned_new=0;
  MPI_Allreduce(&nElementBoundaries_owned_subdomain,&nElementBoundaries_owned_new,1,MPI_INT,MPI_SUM,PROTEUS_COMM_WORLD);
  assert(nElementBoundaries_owned_new == nElementBoundaries_global);

  //now get the element boundaries on the outside of the star
  for (map<int,vector<int> >::iterator elementBoundariesp=elementBoundariesMap.begin();
       elementBoundariesp!=elementBoundariesMap.end();
       elementBoundariesp++)
    {
      //loop over the nodes of this element for the owned nodes
      for (int iv=0;iv<simplexDim;iv++)
        {
          //the elementNodesArrayMap is in the old element numbering while the elementBoundariesMap is in the new element numbering
          int nN_global = elementNodesArrayMap[elementNumbering_global_new2old[elementBoundariesp->first]][iv];
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
  MPI_Allreduce(&nElementBoundaries_subdomain_new_send[0],&nElementBoundaries_subdomain_new[0],size,MPI_INT,MPI_SUM,PROTEUS_COMM_WORLD);
  elementBoundaryOffsets_new[0] = 0;
  for (int sdN=0;sdN<size;sdN++)
    elementBoundaryOffsets_new[sdN+1] = elementBoundaryOffsets_new[sdN]+nElementBoundaries_subdomain_new[sdN];
  //
  //Now as with elements and nodes build a global face numbering
  //resetting the face-based information is a little different since much of this is currently built below based
  //on the element and node information
  //
  valarray<int> elementBoundaryNumbering_new2old(elementBoundaries_subdomain_owned.size());
  set<int>::iterator ebN_ownedp=elementBoundaries_subdomain_owned.begin();
  for (int ebN=0;ebN<int(elementBoundaries_subdomain_owned.size());ebN++)
    {
      elementBoundaryNumbering_new2old[ebN] = *ebN_ownedp++;
    }
  IS elementBoundaryNumberingIS_subdomain_new2old;
  ISCreateGeneral(PROTEUS_COMM_WORLD,elementBoundaries_subdomain_owned.size(),&elementBoundaryNumbering_new2old[0],PETSC_COPY_VALUES,&elementBoundaryNumberingIS_subdomain_new2old);
  IS elementBoundaryNumberingIS_global_new2old;
  ISAllGather(elementBoundaryNumberingIS_subdomain_new2old,&elementBoundaryNumberingIS_global_new2old);
  const PetscInt *elementBoundaryNumbering_global_new2old;
  valarray<int> elementBoundaryNumbering_global_old2new(newMesh.nElementBoundaries_global);
  ISGetIndices(elementBoundaryNumberingIS_global_new2old,&elementBoundaryNumbering_global_new2old);
  //ierr = enforceMemoryLimit(PROTEUS_COMM_WORLD, rank, max_rss_gb,"Allocating elementBoudnary old2new/new2old");CHKERRABORT(PROTEUS_COMM_WORLD, ierr);
  for (int ebN=0;ebN<newMesh.nElementBoundaries_global;ebN++)
    {
      elementBoundaryNumbering_global_old2new[elementBoundaryNumbering_global_new2old[ebN]] = ebN;
    }
  ISRestoreIndices(elementBoundaryNumberingIS_global_new2old,&elementBoundaryNumbering_global_new2old);
  ISDestroy(&elementBoundaryNumberingIS_subdomain_new2old);
  ISDestroy(&elementBoundaryNumberingIS_global_new2old);
  PetscLogEventEnd(build_subdomains_faces_event,0,0,0,0);
  //ierr = enforceMemoryLimit(PROTEUS_COMM_WORLD, rank, max_rss_gb,"Done allocating elementBoudnary old2new/new2old");CHKERRABORT(PROTEUS_COMM_WORLD, ierr);
  int build_subdomains_edges_event;
  PetscLogEventRegister("Subd edges",0,&build_subdomains_edges_event);
  PetscLogEventBegin(build_subdomains_edges_event,0,0,0,0);

  //
  //4c,5c. Repeate the process for edges
  //
  std::ifstream edgeFile(edgeFileName.c_str());

  if (!edgeFile.good())
    {
      std::cerr<<"cannot open Triangle edge file"
               <<edgeFileName<<std::endl;
      failed = true;
      return failed;
    }

  bool hasEdgeMarkers = false;
  int nEdges_global;
  int ihasEdgeMarkers(0);
  edgeFile >> eatcomments >> nEdges_global >> eatline;// edge file doesn currently contain markers >> ihasEdgeMarkers >> eatline ;
  assert(nEdges_global > 0);
  if (ihasEdgeMarkers > 0)
    {
      hasEdgeMarkers = true;
    }

  newMesh.nEdges_global = nEdges_global;
  set<int> edges_subdomain_owned;
  vector<set<int> > nodeEdgesStar(nNodes_subdomain_new[rank]);
  map<int,int> edgeMaterialTypesMap;
  map<int,vector<int> > elementEdgesMap;
  map<int,pair<int,int> > edgeNodesMap;
  set<int> supportedEdges;
  for (int ied = 0; ied < nEdges_global; ied++)
    {
      int ned,nn0,nn1; int edId(0);
      edgeFile >> eatcomments >> ned >> nn0 >> nn1;
      if (ihasEdgeMarkers > 0)
        edgeFile >> edId;
      ned -= indexBase;
      nn0 -= indexBase;
      nn1 -= indexBase;
      assert(0 <= ned && ned < nEdges_global && edgeFile.good());
      int nn0_new = -1;
      if (nodes_old2new_subdomain_map.find(nn0) != nodes_old2new_subdomain_map.end())
        nn0_new = nodes_old2new_subdomain_map[nn0];
      if (nn0_new >= nodeOffsets_new[rank] && nn0_new < nodeOffsets_new[rank+1])
        {
          nodeEdgesStar.at(nn0_new-nodeOffsets_new[rank]).insert(ned);
          supportedEdges.insert(ned);
        }
      int nn1_new = -1;
      if (nodes_old2new_subdomain_map.find(nn1) != nodes_old2new_subdomain_map.end())
        nn1_new = nodes_old2new_subdomain_map[nn1];
      if (nn1_new >= nodeOffsets_new[rank] && nn1_new < nodeOffsets_new[rank+1])
        {
          nodeEdgesStar.at(nn1_new-nodeOffsets_new[rank]).insert(ned);
          supportedEdges.insert(ned);
        }
      int nodes[2] = {nn0_new,nn1_new};
      NodeTuple<2> nodeTuple(nodes);
      edgeFile >> eatline;
      if (edgeElementsMap.find(nodeTuple) != edgeElementsMap.end())//this edge is on an element in the subdomain
        {
          if (nodeTuple.nodes[0] >= nodeOffsets_new[rank] && nodeTuple.nodes[0] < nodeOffsets_new[rank+1])
            edges_subdomain_owned.insert(ned);
          //pick up all the edges on the subdomain and store their nodes
          edgeNodesMap[ned].first = nodeTuple.nodes[0];
          edgeNodesMap[ned].second = nodeTuple.nodes[1];
          if (ihasEdgeMarkers > 0)
            edgeMaterialTypesMap[ned]=edId;
          for (set<pair<int,int> >::iterator elementp=edgeElementsMap[nodeTuple].begin();
               elementp != edgeElementsMap[nodeTuple].end();
               elementp++)
            {
              int eN = elementNumbering_global_old2new[elementp->first];
              if (elementEdgesMap.find(eN) != elementEdgesMap.end())
                {
                  elementEdgesMap[eN][elementp->second] = ned;
                }
              else
                {
                  std::vector<int> init(3,-1);
                  elementEdgesMap[eN] = init;
                  elementEdgesMap[eN][elementp->second] = ned;
                }
            }
        }
    }//end iv
  edgeElementsMap.clear();
  edgeFile.close();
  int nEdges_owned_subdomain=edges_subdomain_owned.size(),
    nEdges_owned_new=0;
  MPI_Allreduce(&nEdges_owned_subdomain,&nEdges_owned_new,1,MPI_INT,MPI_SUM,PROTEUS_COMM_WORLD);
  assert(nEdges_owned_new == nEdges_global);
  //ierr = enforceMemoryLimit(PROTEUS_COMM_WORLD, rank, max_rss_gb,"Done reading edges");CHKERRABORT(PROTEUS_COMM_WORLD, ierr);
  //done with edge file
  //
  //just as with faces, we need to add edges along outer boundaries of star
  //not sure if we need to collect nodeEdges star above anymore, since we're doing this
  //
  for (map<int,vector<int> >::iterator edgesp=elementEdgesMap.begin();
       edgesp!=elementEdgesMap.end();
       edgesp++)
    {
      //loop over the nodes of this element for the owned nodes
      for (int iv=0;iv<simplexDim;iv++)
        {
          //the elementNodesArrayMap is in the old elemetn numbering while the elementEdgesMap is in the new element numbering
          int nN_global = elementNodesArrayMap[elementNumbering_global_new2old[edgesp->first]][iv];
          if (nN_global >= nodeOffsets_new[rank] && nN_global < nodeOffsets_new[rank+1])
            {
              //add all the edges to this node star
              for(int ed=0;ed<3;ed++)
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
      for (set<int>::iterator edN_star = nodeEdgesStar.at(nN).begin();
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
  //Now as with elementBoundaries, build a global face numbering
  //resetting the edge based information is a little different since much of this is currently built below based
  //on the element and node information
  //
  valarray<int> edgeNumbering_new2old(edges_subdomain_owned.size());
  set<int>::iterator edN_ownedp=edges_subdomain_owned.begin();
  for (int edN=0;edN<int(edges_subdomain_owned.size());edN++,edN_ownedp++)
    {
      edgeNumbering_new2old[edN] = *edN_ownedp;
    }
  IS edgeNumberingIS_subdomain_new2old;
  ISCreateGeneral(PROTEUS_COMM_WORLD,edges_subdomain_owned.size(),&edgeNumbering_new2old[0],PETSC_COPY_VALUES,&edgeNumberingIS_subdomain_new2old);
  IS edgeNumberingIS_global_new2old;
  ISAllGather(edgeNumberingIS_subdomain_new2old,&edgeNumberingIS_global_new2old);
  const PetscInt *edgeNumbering_global_new2old;
  valarray<int> edgeNumbering_global_old2new(newMesh.nEdges_global);
  ISGetIndices(edgeNumberingIS_global_new2old,&edgeNumbering_global_new2old);
  //ierr = enforceMemoryLimit(PROTEUS_COMM_WORLD, rank, max_rss_gb,"Setting edgeNumering old2new/new2old");CHKERRABORT(PROTEUS_COMM_WORLD, ierr);
  for (int edN=0;edN<newMesh.nEdges_global;edN++)
    {
      edgeNumbering_global_old2new[edgeNumbering_global_new2old[edN]] = edN;
    }
  ISRestoreIndices(edgeNumberingIS_global_new2old,&edgeNumbering_global_new2old);
  ISDestroy(&edgeNumberingIS_subdomain_new2old);
  ISDestroy(&edgeNumberingIS_global_new2old);
  //ierr = enforceMemoryLimit(PROTEUS_COMM_WORLD, rank, max_rss_gb,"Done allocating edgeNumering old2new/new2old");CHKERRABORT(PROTEUS_COMM_WORLD, ierr);
  //
  //6. Figure out what is in the node stars but not locally owned, create ghost information
  //

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
          int eN_star_new = elementNumbering_global_old2new[eN_star_old];
          bool offproc = eN_star_new >= elementOffsets_new[rank+1] || eN_star_new < elementOffsets_new[rank];
          if (offproc)
            elements_overlap.insert(eN_star_new);
        }
      //element boundaries
      for (int ebN_star_offset = nodeElementBoundaryOffsets[nN];
           ebN_star_offset < nodeElementBoundaryOffsets[nN+1]; ebN_star_offset++)
        {
          int ebN_star_old = nodeElementBoundariesArray[ebN_star_offset];
          int ebN_star_new = elementBoundaryNumbering_global_old2new[ebN_star_old];
          bool offproc = ebN_star_new >= elementBoundaryOffsets_new[rank+1] || ebN_star_new < elementBoundaryOffsets_new[rank];
          if (offproc)
            elementBoundaries_overlap.insert(ebN_star_new);
        }
      //edges in overlap
      for (int edN_star_offset = nodeEdgeOffsets[nN];
           edN_star_offset < nodeEdgeOffsets[nN+1]; edN_star_offset++)
        {
          int edN_star_old = nodeEdgesArray[edN_star_offset];
          int edN_star_new = edgeNumbering_global_old2new[edN_star_old];
          bool offproc = edN_star_new >= edgeOffsets_new[rank+1] || edN_star_new < edgeOffsets_new[rank];
          if (offproc)
            edges_overlap.insert(edN_star_new);
        }
    }//nodes on this processor
  elementNumbering_global_old2new.resize(0);
  MPI_Barrier(PROTEUS_COMM_WORLD);
  assert(edges_overlap.size() + nEdges_subdomain_new[rank] == edgeNodesMap.size());
  //
  //enumerate the overlap
  //
  int nN_subdomain = nNodes_subdomain_new[rank];
  map<int,int> nodes_overlap_global2subdomainMap;
  for (set<int>::iterator nN_globalp=nodes_overlap.begin();nN_globalp != nodes_overlap.end(); nN_globalp++,nN_subdomain++)
    nodes_overlap_global2subdomainMap[*nN_globalp] = nN_subdomain;

  // map<int,int> elements_overlap_global2subdomainMap;
  // for (set<int>::iterator eN_globalp=elements_overlap.begin();eN_globalp != elements_overlap.end(); eN_globalp++,eN_subdomain++)
  //   elements_overlap_global2subdomainMap[*eN_globalp] = eN_subdomain;

  // int ebN_subdomain = nElementBoundaries_subdomain_new[rank];
  // map<int,int> elementBoundaries_overlap_global2subdomainMap;
  // for (set<int>::iterator ebN_globalp=elementBoundaries_overlap.begin();ebN_globalp != elementBoundaries_overlap.end(); ebN_globalp++,ebN_subdomain++)
  //   elementBoundaries_overlap_global2subdomainMap[*ebN_globalp] = ebN_subdomain;

  // int edN_subdomain = nEdges_subdomain_new[rank];
  // map<int,int> edges_overlap_global2subdomainMap;
  // for (set<int>::iterator edN_globalp=edges_overlap.begin();edN_globalp != edges_overlap.end(); edN_globalp++,edN_subdomain++)
  //   edges_overlap_global2subdomainMap[*edN_globalp] = edN_subdomain;
  //
  //7. add any addtional overlap, skip for now
  //

  PetscLogEventEnd(build_subdomains_edges_event,0,0,0,0);
  int build_subdomains_renumber_event;
  PetscLogEventRegister("Subd's renumber",0,&build_subdomains_renumber_event);
  PetscLogEventBegin(build_subdomains_renumber_event,0,0,0,0);
  //
  //8. Build subdomain meshes in new numbering, assumes memory not allocated in subdomain mesh
  //
  if(rank==0){
    std::cerr<<"USER WARNING: In order to avoid a segmentation fault, you need to have supplied the 'f' flag to the triangleOptions input."<<std::endl;
    std::cerr<<"USER WARNING: In order to avoid an edge assertion error, you need to have supplied the 'ee' flag to the triangleOptions input."<<std::endl;
  }

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
  newMesh.subdomainp->nodeMaterialTypes = new int[newMesh.subdomainp->nNodes_global];
  //
  //now finally finish reading node coordinates and node flags
  //
  for (int iv = 0; iv < nNodes_global; iv++)
    {
      int nv; double x,y; int nodeId(0);
      vertexFile >> eatcomments >> nv >> x >> y;
      if (hasVertexMarkers > 0)
        vertexFile >> nodeId;
      nv -= indexBase;
      assert(0 <= nv && nv < nNodes_global && vertexFile.good());
      int nN_global_new = -1;
      if (nodes_old2new_subdomain_map.find(nv) != nodes_old2new_subdomain_map.end())
        nN_global_new = nodes_old2new_subdomain_map[nv];
      //local
      if (nN_global_new >= nodeOffsets_new[rank] && nN_global_new < nodeOffsets_new[rank+1])
        {
          int nv_subdomain_new = nN_global_new - nodeOffsets_new[rank];
          nodeNumbering_subdomain2global[nv_subdomain_new] = nN_global_new;
          nodeNumbering_global2subdomainMap[nN_global_new] = nv_subdomain_new;
          newMesh.subdomainp->nodeArray[vertexDim*nv_subdomain_new + 0] = x;
          newMesh.subdomainp->nodeArray[vertexDim*nv_subdomain_new + 1] = y;
          newMesh.subdomainp->nodeArray[vertexDim*nv_subdomain_new + 2] = 0.0;
          if (hasVertexMarkers > 0)
            newMesh.subdomainp->nodeMaterialTypes[nv_subdomain_new] = nodeId;
        }
      //overlap
      if (nodes_overlap.count(nN_global_new) == 1)
        {
          int nv_subdomain_new = nodes_overlap_global2subdomainMap[nN_global_new];
          nodeNumbering_subdomain2global[nv_subdomain_new] = nN_global_new;
          nodeNumbering_global2subdomainMap[nN_global_new] = nv_subdomain_new;
          newMesh.subdomainp->nodeArray[vertexDim*nv_subdomain_new + 0] = x;
          newMesh.subdomainp->nodeArray[vertexDim*nv_subdomain_new + 1] = y;
          newMesh.subdomainp->nodeArray[vertexDim*nv_subdomain_new + 2] = 0.0;
          if (hasVertexMarkers > 0)
            newMesh.subdomainp->nodeMaterialTypes[nv_subdomain_new] = nodeId;
        }
      vertexFile >> eatline;
    }//end iv
  vertexFile.close();
  ISDestroy(&nodePartitioningIS_new);
  ISDestroy(&nodeNumberingIS_subdomain_old2new);
  //done with vertex file (and all file reads at this point)
  //ierr = enforceMemoryLimit(PROTEUS_COMM_WORLD, rank, max_rss_gb,"Done reading vertices");CHKERRABORT(PROTEUS_COMM_WORLD, ierr);

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
      int eN_global_old = elementNumbering_global_new2old[eN_global_new];
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
      int eN_global_old = elementNumbering_global_new2old[eN_global_new];
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
  ISRestoreIndices(elementNumberingIS_global_new2old,&elementNumbering_global_new2old);
  ISDestroy(&elementNumberingIS_subdomain_new2old);
  ISDestroy(&elementNumberingIS_global_new2old);
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
            elementBoundaryNumbering_global2subdomainMap[elementBoundaryNumbering_global_old2new[elementBoundariesMap[eN_global][ebN]]];
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
            elementBoundaryNumbering_global2subdomainMap[elementBoundaryNumbering_global_old2new[elementBoundariesMap[eN_global_new][ebN]]];
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
      int edN_global_new = edgeNumbering_global_old2new[edN_global_old];
      assert(edgeNumbering_global2subdomainMap.find(edN_global_new) != edgeNumbering_global2subdomainMap.end());
      int edN_subdomain  = edgeNumbering_global2subdomainMap[edN_global_new];
      newMesh.subdomainp->edgeNodesArray[edN_subdomain*2+0] = nodeNumbering_global2subdomainMap[edgep->second.first];
      newMesh.subdomainp->edgeNodesArray[edN_subdomain*2+1] = nodeNumbering_global2subdomainMap[edgep->second.second];
    }
  edgeNumbering_global_old2new.resize(0);
  //
  //end edges
  //

  //now build rest of subdomain mesh connectivity information etc
  bool callOld=true;
  if(callOld)
    constructElementBoundaryElementsArrayWithGivenElementBoundaryAndEdgeNumbers_triangle(*newMesh.subdomainp);
  else
    {
      //const int DEFAULT_ELEMENT_MATERIAL=0;
      const int DEFAULT_NODE_MATERIAL=-1;
      const int INTERIOR_NODE_MATERIAL=0;
      const int EXTERIOR_NODE_MATERIAL=1;
      const int INTERIOR_ELEMENT_BOUNDARY_MATERIAL=0;
      const int EXTERIOR_ELEMENT_BOUNDARY_MATERIAL=1;

      newMesh.subdomainp->nNodes_elementBoundary = 3;
      newMesh.subdomainp->nElementBoundaries_element = 4;
      assert(newMesh.subdomainp->elementBoundariesArray);
      using namespace std;
      //double start,stop;
      map<NodeTuple<3>,
          ElementNeighbors> elementBoundaryElements;
      map<NodeTuple<3>,
          int> elementBoundaryIds;
      //start=CurrentTime();
      //cout<<"Extracting boundary elements"<<endl;
      for(int eN=0;eN<newMesh.subdomainp->nElements_global;eN++)
        for(int ebN=0;ebN<newMesh.subdomainp->nElementBoundaries_element;ebN++)
          {
            int ebN_global = newMesh.subdomainp->elementBoundariesArray[eN*newMesh.subdomainp->nElementBoundaries_element+ebN];
            int nodes[3];
            nodes[0] = newMesh.subdomainp->elementNodesArray[eN*4+((ebN+1)%4)];
            nodes[1] = newMesh.subdomainp->elementNodesArray[eN*4+((ebN+2)%4)];
            nodes[2] = newMesh.subdomainp->elementNodesArray[eN*4+((ebN+3)%4)];
            NodeTuple<3> ebt(nodes);
            if(elementBoundaryElements.find(ebt) != elementBoundaryElements.end())
              {
                elementBoundaryElements[ebt].right=eN;
                elementBoundaryElements[ebt].right_ebN_element=ebN;
                assert(elementBoundaryIds[ebt] == ebN_global);
              }
            else
              {
                elementBoundaryElements.insert(elementBoundaryElements.end(),make_pair(ebt,ElementNeighbors(eN,ebN)));
                elementBoundaryIds.insert(elementBoundaryIds.end(),make_pair(ebt,ebN_global));
              }
          }
      //stop = CurrentTime();
      //cout<<"Elapsed time for building element boundary elements map= "<<(stop-start)<<"s"<<endl;
      newMesh.subdomainp->nElementBoundaries_global = elementBoundaryElements.size();
      //cout<<"nElementBoundaries_global = "<<newMesh.subdomainp->nElementBoundaries_global<<endl;

      //cout<<"Allocating Arrays"<<endl;
      //start = CurrentTime();
      set<int> interiorElementBoundaries,exteriorElementBoundaries;
      newMesh.subdomainp->elementBoundaryNodesArray =  new int[newMesh.subdomainp->nElementBoundaries_global*newMesh.subdomainp->nNodes_elementBoundary];
      newMesh.subdomainp->elementBoundaryElementsArray = new int[newMesh.subdomainp->nElementBoundaries_global*2];
      newMesh.subdomainp->elementBoundaryLocalElementBoundariesArray = new int[newMesh.subdomainp->nElementBoundaries_global*2];
      newMesh.subdomainp->elementNeighborsArray = new int[newMesh.subdomainp->nElements_global*newMesh.subdomainp->nElementBoundaries_element];
      //stop = CurrentTime();
      //cout<<"Elapsed time for allocating arrays = "<<(stop-start)<<"s"<<endl;

      //cout<<"Generating elementBoundaryElementsArray and elementBoundaryNodesArray"<<endl;
      //start = CurrentTime();
      for(map<NodeTuple<3>,ElementNeighbors>::iterator eb=elementBoundaryElements.begin();
          eb != elementBoundaryElements.end();
          eb++)
        {
          int ebN = elementBoundaryIds[eb->first];
          newMesh.subdomainp->elementBoundaryNodesArray[ebN*3 + 0] = eb->first.nodes[0];
          newMesh.subdomainp->elementBoundaryNodesArray[ebN*3 + 1] = eb->first.nodes[1];
          newMesh.subdomainp->elementBoundaryNodesArray[ebN*3 + 2] = eb->first.nodes[2];

          newMesh.subdomainp->elementBoundaryElementsArray[ebN*2 + 0] = eb->second.left;
          newMesh.subdomainp->elementBoundaryLocalElementBoundariesArray[ebN*2 + 0] = eb->second.left_ebN_element;
          newMesh.subdomainp->elementBoundaryElementsArray[ebN*2 + 1] = eb->second.right;
          newMesh.subdomainp->elementBoundaryLocalElementBoundariesArray[ebN*2 + 1] = eb->second.right_ebN_element;
          newMesh.subdomainp->elementNeighborsArray[eb->second.left*newMesh.subdomainp->nElementBoundaries_element + eb->second.left_ebN_element] = eb->second.right;
          if(eb->second.right != -1)
            {
              interiorElementBoundaries.insert(ebN);
              newMesh.subdomainp->elementNeighborsArray[eb->second.right*newMesh.subdomainp->nElementBoundaries_element + eb->second.right_ebN_element] = eb->second.left;
            }
          else
            exteriorElementBoundaries.insert(ebN);
          assert(newMesh.subdomainp->elementBoundariesArray[eb->second.left*newMesh.subdomainp->nElementBoundaries_element + eb->second.left_ebN_element] == ebN);
          if (eb->second.right != -1)
            {
              assert(newMesh.subdomainp->elementBoundariesArray[eb->second.right*newMesh.subdomainp->nElementBoundaries_element + eb->second.right_ebN_element] == ebN);
            }
        }
      newMesh.subdomainp->nInteriorElementBoundaries_global = interiorElementBoundaries.size();
      newMesh.subdomainp->interiorElementBoundariesArray = new int[newMesh.subdomainp->nInteriorElementBoundaries_global];
      newMesh.subdomainp->nExteriorElementBoundaries_global = exteriorElementBoundaries.size();
      newMesh.subdomainp->exteriorElementBoundariesArray = new int[newMesh.subdomainp->nExteriorElementBoundaries_global];
      int ebNI=0,ebNE=0;
      for (set<int>::iterator ebN=interiorElementBoundaries.begin();ebN != interiorElementBoundaries.end(); ebN++,ebNI++)
        newMesh.subdomainp->interiorElementBoundariesArray[ebNI] = *ebN;
      for (set<int>::iterator ebN=exteriorElementBoundaries.begin();ebN != exteriorElementBoundaries.end(); ebN++,ebNE++)
        newMesh.subdomainp->exteriorElementBoundariesArray[ebNE] = *ebN;
      set<NodeTuple<2> > edges;
      for (int eN=0;eN<newMesh.subdomainp->nElements_global;eN++)
        {
          int nodes[2];
          for (int nN_L=0;nN_L<newMesh.subdomainp->nNodes_element;nN_L++)
            for (int nN_R=nN_L+1;nN_R<newMesh.subdomainp->nNodes_element;nN_R++)
              {
                nodes[0] = newMesh.subdomainp->elementNodesArray[eN*4+nN_L];
                nodes[1] = newMesh.subdomainp->elementNodesArray[eN*4+nN_R];
                edges.insert(NodeTuple<2>(nodes));
              }
        }
      assert(newMesh.subdomainp->nEdges_global == int(edges.size()));
      vector<set<int> > nodeStar(newMesh.subdomainp->nNodes_global);
      for (int edgeN=0;edgeN<newMesh.subdomainp->nEdges_global;edgeN++)
        {
          nodeStar[newMesh.subdomainp->edgeNodesArray[edgeN*2+0]].insert(newMesh.subdomainp->edgeNodesArray[edgeN*2+1]);
          nodeStar[newMesh.subdomainp->edgeNodesArray[edgeN*2+1]].insert(newMesh.subdomainp->edgeNodesArray[edgeN*2+0]);
        }
      newMesh.subdomainp->nodeStarOffsets = new int[newMesh.subdomainp->nNodes_global+1];
      newMesh.subdomainp->nodeStarOffsets[0] = 0;
      for (int nN=1;nN<newMesh.subdomainp->nNodes_global+1;nN++)
        newMesh.subdomainp->nodeStarOffsets[nN] = newMesh.subdomainp->nodeStarOffsets[nN-1] + nodeStar[nN-1].size();
      newMesh.subdomainp->nodeStarArray = new int[newMesh.subdomainp->nodeStarOffsets[newMesh.subdomainp->nNodes_global]];
      for (int nN=0,offset=0;nN<newMesh.subdomainp->nNodes_global;nN++)
        for (set<int>::iterator nN_star=nodeStar[nN].begin();nN_star!=nodeStar[nN].end();nN_star++,offset++)
          newMesh.subdomainp->nodeStarArray[offset] = *nN_star;
      //stop = CurrentTime();
      newMesh.subdomainp->max_nNodeNeighbors_node=0;
      for (int nN=0;nN<newMesh.subdomainp->nNodes_global;nN++)
        newMesh.subdomainp->max_nNodeNeighbors_node=max(newMesh.subdomainp->max_nNodeNeighbors_node,newMesh.subdomainp->nodeStarOffsets[nN+1]-newMesh.subdomainp->nodeStarOffsets[nN]);
      //mwf repeat for node-->elements arrays
      vector<set<int> > nodeElementsStar(newMesh.subdomainp->nNodes_global);
      for (int eN = 0; eN < newMesh.subdomainp->nElements_global; eN++)
        {
          for (int nN = 0; nN < newMesh.subdomainp->nNodes_element; nN++)
            nodeElementsStar[newMesh.subdomainp->elementNodesArray[eN*newMesh.subdomainp->nNodes_element+nN]].insert(eN);
        }
      newMesh.subdomainp->nodeElementOffsets = new int[newMesh.subdomainp->nNodes_global+1];
      newMesh.subdomainp->nodeElementOffsets[0] = 0;
      for (int nN = 0; nN < newMesh.subdomainp->nNodes_global; nN++)
        newMesh.subdomainp->nodeElementOffsets[nN+1] = newMesh.subdomainp->nodeElementOffsets[nN]+nodeElementsStar[nN].size();
      newMesh.subdomainp->nodeElementsArray  = new int[newMesh.subdomainp->nodeElementOffsets[newMesh.subdomainp->nNodes_global]];
      for (int nN=0,offset=0; nN < newMesh.subdomainp->nNodes_global; nN++)
        {
          for (set<int>::iterator eN_star = nodeElementsStar[nN].begin(); eN_star != nodeElementsStar[nN].end();
               eN_star++,offset++)
            {
              newMesh.subdomainp->nodeElementsArray[offset] = *eN_star;
            }
        }
      //mwf end node-->elements construction
      newMesh.subdomainp->elementBoundaryMaterialTypes = new int[newMesh.subdomainp->nElementBoundaries_global];
      //if nodeMaterial is DEFAULT, go ahead and set to interior or exterior
      //depending on which boundary node belongs to.
      //If node on at least one exterior boundary then it's exterior
      for (int ebNE = 0; ebNE < newMesh.subdomainp->nExteriorElementBoundaries_global; ebNE++)
        {
          int ebN = newMesh.subdomainp->exteriorElementBoundariesArray[ebNE];
          newMesh.subdomainp->elementBoundaryMaterialTypes[ebN] = EXTERIOR_ELEMENT_BOUNDARY_MATERIAL;
          for (int nN_local = 0; nN_local < newMesh.subdomainp->nNodes_elementBoundary; nN_local++)
            {
              int nN = newMesh.subdomainp->elementBoundaryNodesArray[ebN*newMesh.subdomainp->nNodes_elementBoundary+nN_local];
              if (newMesh.subdomainp->nodeMaterialTypes[nN] == DEFAULT_NODE_MATERIAL)
                newMesh.subdomainp->nodeMaterialTypes[nN] = EXTERIOR_NODE_MATERIAL;
            }
        }
      for (int ebNI = 0; ebNI < newMesh.subdomainp->nInteriorElementBoundaries_global; ebNI++)
        {
          int ebN = newMesh.subdomainp->interiorElementBoundariesArray[ebNI];
          newMesh.subdomainp->elementBoundaryMaterialTypes[ebN] = INTERIOR_ELEMENT_BOUNDARY_MATERIAL;
          for (int nN_local = 0; nN_local < newMesh.subdomainp->nNodes_elementBoundary; nN_local++)
            {
              int nN = newMesh.subdomainp->elementBoundaryNodesArray[ebN*newMesh.subdomainp->nNodes_elementBoundary+nN_local];
              if (newMesh.subdomainp->nodeMaterialTypes[nN] == DEFAULT_NODE_MATERIAL)
                newMesh.subdomainp->nodeMaterialTypes[nN] = INTERIOR_NODE_MATERIAL;
            }
        }
      //cout<<"Elapsed time for populating arrays = "<<(stop-start)<<"s"<<endl;
    }
  //build local geometric info
  allocateGeometricInfo_triangle(*newMesh.subdomainp);
  computeGeometricInfo_triangle(*newMesh.subdomainp);

  if (hasElementBoundaryMarkers)
    {
      assert(newMesh.subdomainp->elementBoundariesArray != NULL);
      for (map<int,int>::iterator ebmp = elementBoundaryMaterialTypesMap.begin(); ebmp != elementBoundaryMaterialTypesMap.end();ebmp++)
        {
          int ebN_global_new = elementBoundaryNumbering_global_old2new[ebmp->first];
          assert(elementBoundaryNumbering_global2subdomainMap.find(ebN_global_new) != elementBoundaryNumbering_global2subdomainMap.end());
          int ebN_subdomain  = elementBoundaryNumbering_global2subdomainMap[ebN_global_new];
          newMesh.subdomainp->elementBoundaryMaterialTypes[ebN_subdomain] = ebmp->second;
        }
      if (!hasVertexMarkers)
        for (int ebNE = 0; ebNE < newMesh.subdomainp->nExteriorElementBoundaries_global; ebNE++)
          {
            int ebN = newMesh.subdomainp->exteriorElementBoundariesArray[ebNE];
            for (int nN_local = 0; nN_local < newMesh.subdomainp->nNodes_elementBoundary; nN_local++)
              {
                int nN = newMesh.subdomainp->elementBoundaryNodesArray[ebN*newMesh.subdomainp->nNodes_elementBoundary+nN_local];
                newMesh.subdomainp->nodeMaterialTypes[nN] = newMesh.subdomainp->elementBoundaryMaterialTypes[ebN];
              }
          }
    }
  elementBoundaryNumbering_global_old2new.resize(0);
  //ierr = enforceMemoryLimit(PROTEUS_COMM_WORLD, rank, max_rss_gb,"Done with material types");CHKERRABORT(PROTEUS_COMM_WORLD, ierr);
  PetscLogEventEnd(build_subdomains_renumber_event,0,0,0,0);
  int build_subdomains_cleanup_event;
  PetscLogEventRegister("Cleanup",0,&build_subdomains_cleanup_event);
  PetscLogEventBegin(build_subdomains_cleanup_event,0,0,0,0);
  //transfer information about owned nodes and elements to mesh
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
  //cleanup
  /* out of core*/
  H5Sclose(filespace);
  H5Sclose(memspace);
  H5Pclose(plist_id);
  //(the mappings file is closed right after the numbering write and again
  //after the chunked read loop; nothing is open here any more)
  /* out of core */
  PetscLogEventEnd(build_subdomains_cleanup_event,0,0,0,0);
  PetscLogStagePop();
  PetscLogView(PETSC_VIEWER_STDOUT_WORLD);
  //ierr = enforceMemoryLimit(PROTEUS_COMM_WORLD, rank, max_rss_gb,"Done with partitioning!");CHKERRABORT(PROTEUS_COMM_WORLD, ierr);
  return 0;
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
