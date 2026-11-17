#include <algorithm>
#include <cstdio>
#include <memory>
#include <vector>
#include <valarray>
#include "MeshAdaptPUMI.h"
#include "PCU.h"
#include "mesh.h"
#include "apfConvert.h"
#include "apfShape.h"

#include <sstream>

static apf::Numbering* numberOwnedEntitiesFirst(apf::Mesh* m, int dimension,int initialReconstructed)
{
  std::stringstream ss;
  ss << "proteus_number_" << dimension;
  std::string s = ss.str();
  apf::FieldShape* shape;
  if (dimension) /* this switch is just to help rendering */
    shape = apf::getConstant(dimension);
  else
    shape = m->getShape();
  apf::Numbering* n = createNumbering(m, s.c_str(), shape, 1);
  apf::MeshEntity* e;
  apf::MeshIterator* it = m->begin(dimension);
  int i = 0;
  if(initialReconstructed){
    while ((e = m->iterate(it)))
      apf::number(n, e, 0, 0, i++);
    m->end(it);
  }
  else{
    while ((e = m->iterate(it)))
      if (m->isOwned(e))
        apf::number(n, e, 0, 0, i++);
    m->end(it);
    it = m->begin(dimension);
    while ((e = m->iterate(it)))
      if (!m->isOwned(e))
        apf::number(n, e, 0, 0, i++);
    m->end(it);
  }
  return n;
}

//Main API to construct a serial pumi mesh,
//what we do is contruct the global mesh when working with serial
//and let Proteus populate the subdomain data structures
//(though it will be exactly same)
int MeshAdaptPUMIDrvr::constructFromSerialPUMIMesh(Mesh& mesh)
{
  assert(m != 0);
  logEvent("Constructing global data structures",4);

  int dim = m->getDimension();
  mesh.nElements_global = m->count(dim);

  mesh.nNodes_global = m->count(0);

  mesh.nElementBoundaries_global = m->count(dim - 1);

  mesh.nEdges_global = m->count(1);

  //nNodes_element for now is constant for the entire mesh, Ask proteus about using mixed meshes
  switch (dim) {
  case 2:
    mesh.nNodes_element = 3;
    mesh.nNodes_elementBoundary = 2;
    mesh.nElementBoundaries_element = 3;
    break;
  case 3:
    mesh.nNodes_element = 4;
    mesh.nNodes_elementBoundary = 3;
    mesh.nElementBoundaries_element = 4;
    break;
  default:
    apf::fail("dimension is not 2 or 3\n");
    break;
  }
#ifdef MESH_INFO
  std::cerr << "*******Proteus Mesh Stats*********\n";
  std::cerr << "Number of elements " << mesh.nElements_global << "\n";
  std::cerr << "Number of nodes " << mesh.nNodes_global << "\n";
  std::cerr << "Number of boundaries " << mesh.nElementBoundaries_global << "\n";
  std::cerr << "Number of edges " << mesh.nEdges_global << "\n";
#endif
  
  numberLocally();
  constructNodes(mesh);
  constructElements(mesh);
  constructBoundaries(mesh);
  constructEdges(mesh);
  constructMaterialArrays(mesh);

  return 0;
}

void MeshAdaptPUMIDrvr::numberLocally()
{
  for (int d = 0; d <= m->getDimension(); ++d) {
    freeNumbering(local[d]);
    local[d] = numberOwnedEntitiesFirst(m, d,initialReconstructed);
  }
  if(initialReconstructed) 
    initialReconstructed = 0;
}

int MeshAdaptPUMIDrvr::localNumber(apf::MeshEntity* e)
{
  return getNumber(local[apf::getDimension(m, e)], e, 0, 0);
}

int MeshAdaptPUMIDrvr::constructNodes(Mesh& mesh)
{
  mesh.nodeArray = new double[mesh.nNodes_global * 3];
  apf::MeshIterator* it = m->begin(0);
  apf::MeshEntity* e;
  while ((e = m->iterate(it))) {
    int i = localNumber(e);
    apf::Vector3 x;
    m->getPoint(e, 0, x);
    for(int j=0; j<3; j++)
      mesh.nodeArray[i * 3 + j]= x[j];
  }
  m->end(it);
  return 0;
} 

int MeshAdaptPUMIDrvr::constructElements(Mesh& mesh)
{
  mesh.elementNodesArray = new proteus_index_t[mesh.nElements_global*mesh.nNodes_element];
  apf::MeshIterator* it = m->begin(m->getDimension());
  apf::MeshEntity* e;
  while ((e = m->iterate(it))) {
    int i = localNumber(e);
    apf::Downward v;
    int iNumVtx = m->getDownward(e, 0, v);
    for (int j = 0; j < iNumVtx; ++j) {
      int vtxID = localNumber(v[j]);
      mesh.elementNodesArray[i * mesh.nNodes_element + j] = vtxID;
    }
  }
  m->end(it);
  return 0;
}

/* following is going to look adhoc and arbitrary but it is needed to
   resolve a conflict between the way proteus handles faces and we handle faces.
   The order in which we retrieve faces from adjacency is different to what
   proteus expects them to be in their elementBoundaries array.
   To resolve this we need to change the local number of the face of an element
   to what proteus expects it to be and then it works. Whussh. magic.

   original comment above preserved for entertainment.
   This maps SCOREC's tet face numbering to that of proteus.
*/

int getProteusBoundaryIdx(apf::Mesh* m, apf::MeshEntity* e, apf::MeshEntity* f)
{
  apf::Downward fs;
  int dim = m->getDimension();
  int nfs = m->getDownward(e, dim - 1, fs);
  int idx_apf = apf::findIn(fs, nfs, f);
  /* Proteus convention is that the face index equals the vertex index
     of the vertex opposite to the face.
     Proteus and PUMI should have consistent vertex orderings for
     simplices, but the above rule makes the side orderings different */
  static int const tet_boundary_map[4] = {3,2,0,1};
  static int const tri_boundary_map[3] = {2,0,1};
  static int const* const boundary_maps[4] = {
    0,
    0,
    tri_boundary_map,
    tet_boundary_map
  };
  return boundary_maps[dim][idx_apf];
}

int MeshAdaptPUMIDrvr::constructBoundaries(Mesh& mesh)
{
  //build face list (elementBoundary and nodeBoundary arrays)
  //Enter at your own peril for those who stray will be lost
  std::set<int> interiorElementBoundaries;
  std::set<int> exteriorElementBoundaries;

  mesh.elementBoundaryNodesArray =
    new proteus_index_t[mesh.nElementBoundaries_global * mesh.nNodes_elementBoundary];
  mesh.elementBoundaryElementsArray =
    new proteus_index_t[mesh.nElementBoundaries_global * 2];
  mesh.elementBoundaryLocalElementBoundariesArray =
    new int[mesh.nElementBoundaries_global * 2];
  mesh.elementNeighborsArray =
    new proteus_index_t[mesh.nElements_global * mesh.nElementBoundaries_element];
  mesh.elementBoundariesArray =
    new proteus_index_t[mesh.nElements_global * mesh.nElementBoundaries_element];
  //exteriorGlobaltoLocalElementBoundariesArray = 
  //  new int[mesh.nElementBoundaries_global];
  //int exterior_count = 0; //counter for external boundaries

  int dim = m->getDimension();
  apf::MeshIterator* it = m->begin(dim - 1);
  apf::MeshEntity* f;
  while ((f = m->iterate(it))) {
    int i = localNumber(f);
    // get vertices from adjacency
    apf::Downward vs;
    int iNumVtx = m->getDownward(f, 0, vs);
    for (int iVtx = 0; iVtx < iNumVtx; ++iVtx) {
      int vtxID = localNumber(vs[iVtx]);
      mesh.elementBoundaryNodesArray[
				     i * mesh.nNodes_elementBoundary + iVtx] = vtxID;
    }
    //get regions from adjacency
    apf::Up rs;
    m->getUp(f, rs);
    int iNumRgn = rs.n;
    int RgnID[2] = {-1,-1}; 
    int localBoundaryNumber[2] = {-1,-1};
    for (int iRgn = 0; iRgn < iNumRgn; ++iRgn) {
      RgnID[iRgn] = localNumber(rs.e[iRgn]);
      mesh.elementBoundaryElementsArray[i * 2 + iRgn]= RgnID[iRgn];
      localBoundaryNumber[iRgn] = getProteusBoundaryIdx(m, rs.e[iRgn], f);
      assert(localBoundaryNumber[iRgn] != -1);
      mesh.elementBoundaryLocalElementBoundariesArray[
						      i * 2 + iRgn] = localBoundaryNumber[iRgn];
    }
    //left and right regions are shared by this face we are currntly on
    int leftRgnID = RgnID[0]; int leftLocalBoundaryNumber = localBoundaryNumber[0];
    int rightRgnID = RgnID[1]; int rightLocalBoundaryNumber = localBoundaryNumber[1];
    /*left region is always there, so either rightRgnID will have
      an actual ID if this face is shared, or will contain -1
      if it is an exterior face */
    mesh.elementNeighborsArray[
			       leftRgnID * mesh.nElementBoundaries_element + leftLocalBoundaryNumber]
      = rightRgnID;
    mesh.elementBoundariesArray[
				leftRgnID * mesh.nElementBoundaries_element + leftLocalBoundaryNumber]
      = i;

    /* if only 1 region is adjacent to this face,
       that means it is an exterior face */
    if(iNumRgn==1) {
      assert(RgnID[1]==-1);
      assert(localBoundaryNumber[1]==-1); //last 2 checks are only for sanity
      mesh.elementBoundaryElementsArray[i * 2 + 1] = -1;
      mesh.elementBoundaryLocalElementBoundariesArray[i * 2 + 1] = -1;
      //exterior face as only 1 region adjacent
      exteriorElementBoundaries.insert(i);

      //construct inverse mapping 
      //exteriorGlobaltoLocalElementBoundariesArray[i] = exterior_count;
      //exterior_count++;

    } else { //2 regions are shared by this face so interior face
      mesh.elementNeighborsArray[
				 rightRgnID * mesh.nElementBoundaries_element + rightLocalBoundaryNumber]
        = leftRgnID;
      mesh.elementBoundariesArray[
				  rightRgnID * mesh.nElementBoundaries_element + rightLocalBoundaryNumber]
        = i;
      interiorElementBoundaries.insert(i);
    }
  }
  m->end(it);
 
  //construct interior and exterior element boundaries array 
  mesh.nInteriorElementBoundaries_global = interiorElementBoundaries.size();
  mesh.interiorElementBoundariesArray = new proteus_index_t[mesh.nInteriorElementBoundaries_global];
  mesh.nExteriorElementBoundaries_global = exteriorElementBoundaries.size();
  mesh.exteriorElementBoundariesArray = new proteus_index_t[mesh.nExteriorElementBoundaries_global];

  int ebNI=0,ebNE=0;
  for (std::set<int>::iterator ebN=interiorElementBoundaries.begin();ebN != interiorElementBoundaries.end(); ebN++,ebNI++)
    mesh.interiorElementBoundariesArray[ebNI] = *ebN;
  for (std::set<int>::iterator ebN=exteriorElementBoundaries.begin();ebN != exteriorElementBoundaries.end(); ebN++,ebNE++)
    mesh.exteriorElementBoundariesArray[ebNE] = *ebN;
  
  return 0;
}

/* these algorithms are totally independent of SCOREC;
   they form the proteus
   vertex to vertex and vertex to element adjacency tables
   from the edge to vertex and element to vertex tables */
static void createStars(Mesh& mesh)
{
  std::vector<std::set<proteus_index_t> > nodeStar(mesh.nNodes_global);
  for (int edgeN = 0; edgeN < mesh.nEdges_global; edgeN++) {
    nodeStar[mesh.edgeNodesArray[edgeN * 2 + 0]].insert(
							mesh.edgeNodesArray[edgeN * 2 + 1]);
    nodeStar[mesh.edgeNodesArray[edgeN * 2 + 1]].insert(
							mesh.edgeNodesArray[edgeN * 2 + 0]);
  }

  mesh.nodeStarOffsets = new proteus_index_t[mesh.nNodes_global + 1];
  mesh.nodeStarOffsets[0] = 0;
  for (int nN = 1; nN <= mesh.nNodes_global; nN++)
    mesh.nodeStarOffsets[nN] =
      mesh.nodeStarOffsets[nN - 1] + nodeStar[nN - 1].size();
  
  mesh.nodeStarArray = new proteus_index_t[mesh.nodeStarOffsets[mesh.nNodes_global]];
  for (int nN = 0, offset = 0; nN < mesh.nNodes_global; nN++)
    for (std::set<proteus_index_t>::iterator nN_star=nodeStar[nN].begin();
         nN_star!=nodeStar[nN].end();
         nN_star++, offset++)
      mesh.nodeStarArray[offset] = *nN_star;

  mesh.max_nNodeNeighbors_node = 0;
  for (int nN = 0; nN < mesh.nNodes_global; nN++)
    mesh.max_nNodeNeighbors_node =
      std::max(mesh.max_nNodeNeighbors_node,
	       mesh.nodeStarOffsets[nN + 1] - mesh.nodeStarOffsets[nN]);

  std::vector<std::set<int> > nodeElementsStar(mesh.nNodes_global);
  for (int eN = 0; eN < mesh.nElements_global; eN++)
    for (int nN = 0; nN < mesh.nNodes_element; nN++)
      nodeElementsStar[
		       mesh.elementNodesArray[eN * mesh.nNodes_element + nN]
		       ].insert(eN);
  mesh.nodeElementOffsets = new proteus_index_t[mesh.nNodes_global + 1];
  mesh.nodeElementOffsets[0] = 0;
  for (int nN = 0; nN < mesh.nNodes_global; nN++)
    mesh.nodeElementOffsets[nN + 1] =
      mesh.nodeElementOffsets[nN] + nodeElementsStar[nN].size();
  mesh.nodeElementsArray = new proteus_index_t[mesh.nodeElementOffsets[mesh.nNodes_global]];
  for (int nN=0,offset=0; nN < mesh.nNodes_global; nN++)
    for (std::set<int>::iterator eN_star = nodeElementsStar[nN].begin();
         eN_star != nodeElementsStar[nN].end();
         eN_star++, offset++)
      mesh.nodeElementsArray[offset] = *eN_star;
}

int MeshAdaptPUMIDrvr::constructEdges(Mesh& mesh)
{
  mesh.edgeNodesArray = new proteus_index_t[mesh.nEdges_global * 2];
  apf::MeshIterator* it = m->begin(1);
  apf::MeshEntity* e;
  while ((e = m->iterate(it))) {
    int i = localNumber(e);
    apf::MeshEntity* v[2];
    m->getDownward(e, 0, v);
    for(int iVtx=0; iVtx < 2; ++iVtx) {
      int vtxID = localNumber(v[iVtx]);
      mesh.edgeNodesArray[i * 2 + iVtx] = vtxID;
    }
  }
  m->end(it);
  createStars(mesh);
  return 0;
}

#define INTERIOR_MATERIAL 0
#define EXTERIOR_MATERIAL 1
#define DEFAULT_ELEMENT_MATERIAL INTERIOR_MATERIAL

static int getInOutMaterial(apf::Mesh* m, apf::MeshEntity* e)
{
  if (m->getModelType(m->toModel(e)) == m->getDimension())
    return INTERIOR_MATERIAL;
  else
    return EXTERIOR_MATERIAL;
}

/* This function builds the element, elementBoundary, and node
 * Material arrays and fills them in with zero or one depending
 * one whether the entity is classified on the geometric boundary
 * or not.
 * This forms a baseline material tagging for all entities,
 * which later gets overwritten for some entities with more
 * specific values in MeshAdaptPUMIDrvr::UpdateMaterialArrays.
 */
int MeshAdaptPUMIDrvr::constructMaterialArrays(Mesh& mesh)
{
  mesh.elementBoundaryMaterialTypes = new int[mesh.nElementBoundaries_global];
  mesh.nodeMaterialTypes = new int[mesh.nNodes_global];
  mesh.elementMaterialTypes = new int[mesh.nElements_global];
  for (int i = 0; i < mesh.nElements_global; ++i)
    mesh.elementMaterialTypes[i] = DEFAULT_ELEMENT_MATERIAL;
  int dim = m->getDimension();
  apf::MeshIterator* it = m->begin(dim - 1);
  apf::MeshEntity* f;
  while ((f = m->iterate(it))) {
    int i = localNumber(f);
    mesh.elementBoundaryMaterialTypes[i] = getInOutMaterial(m, f);
  }
  m->end(it);
  it = m->begin(0);
  apf::MeshEntity* v;
  while ((v = m->iterate(it))) {
    int i = localNumber(v);
    mesh.nodeMaterialTypes[i] = getInOutMaterial(m, v);
  }
  m->end(it);
  return 0;
}

/* Given a geometric model face identified by the integer
 * (scorec_tag), get all nodes and element boundaries
 * classified on the closure of that model face and
 * put the (proteus_material) integer in their material
 * array slot.
 */
int MeshAdaptPUMIDrvr::updateMaterialArrays(Mesh& mesh,
					    int dim,
					    int proteus_material,
					    int scorec_tag)
{
  //int dim = m->getDimension();
  apf::ModelEntity* geomEnt = m->findModelEntity(dim, scorec_tag);
  apf::MeshIterator* it = m->begin(dim);
  apf::MeshEntity* f;
  if(dim==m->getDimension()){
    while ((f = m->iterate(it))) {
      if (m->toModel(f) == geomEnt) {
	int i = localNumber(f);
	mesh.elementMaterialTypes[i] = proteus_material;
      }
    }
  }
  else{
    while ((f = m->iterate(it))) {
      if (m->toModel(f) == geomEnt) {
        int i = localNumber(f);
        mesh.elementBoundaryMaterialTypes[i] = proteus_material;
      }
    }
    apf::DynamicArray<apf::Node> nodes;
    apf::getNodesOnClosure(m, geomEnt, nodes);
    for (size_t i = 0; i < nodes.getSize(); ++i) {
      int vtxId = localNumber(nodes[i].entity);
      mesh.nodeMaterialTypes[vtxId] = proteus_material;
    }
  }
  m->end(it);
  return 0;
}

/* Overload updateMaterialArray for reconstructed SCOREC meshes.
 * The material types are stored based on derived model entity.
 * We can recover the material of a mesh entity by looking at its classified
 * model entity
 */
int MeshAdaptPUMIDrvr::updateMaterialArrays(Mesh& mesh)
{
  int geomTag;
  apf::ModelEntity* geomEnt;
  apf::MeshIterator* it;
  apf::MeshEntity* f;

  int dim = 0;
  it = m->begin(dim);
  while(f = m->iterate(it)){
    int i = localNumber(f);
    geomEnt = m->toModel(f);
    geomTag = m->getModelTag(geomEnt);
    if(m->getModelType(geomEnt) == dim){
      mesh.nodeMaterialTypes[i] =modelVertexMaterial[geomTag];
    }
    else if(m->getModelType(geomEnt)==(m->getDimension()-1)){ //on the boundary entity
      mesh.nodeMaterialTypes[i] =modelBoundaryMaterial[geomTag];
    }
    //If 3D and is on an exterior edge
    else if(m->getDimension()==3 && m->getModelType(geomEnt)==1){
      apf::Adjacent vert_adjFace;
      m->getAdjacent(f,2,vert_adjFace);
      for(int j=0;j<vert_adjFace.getSize();j++){
        apf::ModelEntity* adjEnt = m->toModel(vert_adjFace[j]);
        if(m->getModelType(adjEnt) == 2){
          mesh.nodeMaterialTypes[i] = modelBoundaryMaterial[m->getModelTag(adjEnt)];
        }
      }
    }
    else{
      mesh.nodeMaterialTypes[i] = 0; //This assumes that all vertices on the boundary are model vertices
    }
  }
  m->end(it);
  if(m->getDimension()==2)
    dim = 1;
  else
    dim = 2;
  it = m->begin(dim);
  while(f = m->iterate(it)){
    geomEnt = m->toModel(f);
    int i = localNumber(f);
    if(m->getModelType(geomEnt) == dim){
      geomTag = m->getModelTag(m->toModel(f));
      mesh.elementBoundaryMaterialTypes[i] =modelBoundaryMaterial[geomTag];
    }
    else{
      geomTag = m->getModelTag(m->toModel(f));
      //Interior boundaries and entities have a material type of zero
      mesh.elementBoundaryMaterialTypes[i] = 0; 
    }
  }
  m->end(it);

  dim = dim+1; //the regions are necessarily one dimension higher than previous dim
  it = m->begin(dim);
  while(f = m->iterate(it)){
    geomEnt = m->toModel(f);
    int i = localNumber(f);
    assert(m->getModelType(geomEnt)==dim);
    geomTag = m->getModelTag(m->toModel(f));
    //The geomTag is actually the right material for region entities
    mesh.elementMaterialTypes[i] = geomTag; //modelRegionMaterial[geomTag];
  }
  m->end(it);
  return 0;
}

int MeshAdaptPUMIDrvr::updateMaterialArrays2(Mesh& mesh)
{
  logEvent("Starting to update material arrays",4);
  int geomTag;
  apf::ModelEntity* geomEnt;
  apf::MeshIterator* it;
  apf::MeshEntity* f;

  //first associate all nodes with a material tag and synchronize fields to avoid mismatches 
  //The procedure is to have each vertex look for its classification.
  //If it is classified in the region, then it is interior.
  //Else, loop over adjacent faces and stop at first instance of mesh face classified on model boundary and take tag.
  //If there are no such adjacent mesh faces, then set the value to be -1. This should only happen if the vertex is a shared entity.
  //If the vertex is shared, communicate value to remote copies.
  //When receiving values, if the current value is -1, write to field the received value. Otherwise, do nothing. 

  apf::Field* nodeMaterials = apf::createLagrangeField(m, "nodeMaterials", apf::SCALAR, 1);
  it = m->begin(0);
  PCU_Comm_Begin(PCUObj);
  while(f = m->iterate(it))
    {
      geomEnt = m->toModel(f);
      //if classified in a region
      if(m->getModelType(geomEnt) == m->getDimension())
	{
	  apf::setScalar(nodeMaterials,f,0,0); 
	}
      else
	{
	  apf::Adjacent vert_adjFace;
	  m->getAdjacent(f,m->getDimension()-1,vert_adjFace);
	  apf::MeshEntity* face;
	  for(int i =0; i<vert_adjFace.getSize();i++)
	    {
	      face=vert_adjFace[i];
	      geomEnt = m->toModel(face);

	      //IF mesh face is classified on boundary
	      if(m->getModelType(geomEnt) == m->getDimension()-1)
		{
		  geomTag = m->getModelTag(geomEnt);
		  apf::setScalar(nodeMaterials,f,0,geomTag);
		  if(m->isShared(f))
		    {
		      apf::Copies remotes;
		      m->getRemotes(f,remotes);
		      for(apf::Copies::iterator iter = remotes.begin(); iter != remotes.end(); ++iter)
			{
			  PCU_COMM_PACK(PCUObj, iter->first,iter->second);
			  PCU_COMM_PACK(PCUObj, iter->first,geomTag);
			}
		    }  
		  break;
		}
	      if(i == vert_adjFace.getSize()-1 )
		apf::setScalar(nodeMaterials,f,0,-1);
	    }
	}
    }
  m->end(it);
  PCU_Comm_Send(PCUObj);
  while(PCU_Comm_Receive(PCUObj))
    {
      PCU_COMM_UNPACK(PCUObj, f);
      PCU_COMM_UNPACK(PCUObj, geomTag);
      int currentTag = apf::getScalar(nodeMaterials,f,0);
      int newTag = std::min(currentTag,geomTag);
      //if vertex is not interior and had no adjacent faces, take received values
      //else take minimum value of all tags
      if(currentTag==-1)
	apf::setScalar(nodeMaterials,f,0,geomTag);
      else
	apf::setScalar(nodeMaterials,f,0,newTag);
    }
  //Ensure there are no mismatches across parts and then assign node materials
  apf::synchronize(nodeMaterials);
  it = m->begin(0);
  while(f=m->iterate(it))
    {
      int vID = localNumber(f);
      mesh.nodeMaterialTypes[vID] = apf::getScalar(nodeMaterials,f,0);
    }
  m->end(it);

  //First iterate over all faces in 3D, get the model tag and apply to all downward adjacencies
  int dim = m->getDimension()-1;
  it = m->begin(dim);
  while(f = m->iterate(it))
    {
      int i = localNumber(f);
      geomEnt = m->toModel(f);
      geomTag = m->getModelTag(geomEnt);
      if(m->getModelType(geomEnt) == dim)
	{
	  mesh.elementBoundaryMaterialTypes[i] = geomTag;
	}
    }
  m->end(it);

  apf::destroyField(nodeMaterials);

  //Loop over regions
  dim = m->getDimension();
  it = m->begin(dim);
  while( f = m->iterate(it)){
    int i = localNumber(f);
    geomEnt = m->toModel(f);
    geomTag = m->getModelTag(geomEnt);
    if(m->getModelType(geomEnt) == dim){
      mesh.elementMaterialTypes[i] = 0;//geomTag;
    }
  }
  m->end(it);
  return 0;
}


/**************************************************************************/

/*This section of code is a modified version of the apf::construct() function available in 
 * scorec/core. This may be added into scorec/core eventually and removed.
 */

//#include <PCU.h>
#include "apfConvert.h"
#include "apfMesh2.h"
#include "apf.h"
#include "apfConvert.h"
#include "apfNumbering.h"
#include <map>

namespace apf {

  static void constructVerts(
			     Mesh2* m, int nverts,
			     proteus_index_t* local2globalMap,
			     GlobalToVert& result)
  {
    ModelEntity* interior = m->findModelEntity(m->getDimension(), 0);
    for (int i = 0; i < nverts; ++i)
      result[local2globalMap[i]] = m->createVert_(interior);
  }


  static void constructBoundaryElements(
					Mesh2* m, const apf::Gid* conn_b, int nelem_b, int etype_b,
					GlobalToVert& globalToVert)
  {
    ModelEntity* interior = m->findModelEntity(m->getDimension(), 0);
    int nev = apf::Mesh::adjacentCount[etype_b][0];
    for (int i = 0; i < nelem_b; ++i) {
      Downward verts;
      int offset = i * nev;
      for (int j = 0; j < nev; ++j){
	verts[j] = globalToVert[conn_b[j + offset]];
      }
      //We only care about how boundary elements are created
      //The intermediate entities need to inherit the classifications
      if(m->getDimension()==2)
	m->createEntity(etype_b,interior,verts);
      else
	apf::buildElement(m,interior,2,verts);
    }
  }
  static void constructElements(
				Mesh2* m, const Gid* conn, int nelem, int etype,
				GlobalToVert& globalToVert)
  {
    ModelEntity* interior = m->findModelEntity(m->getDimension(), 0);
    int nev = apf::Mesh::adjacentCount[etype][0];
    for (int i = 0; i < nelem; ++i) {
      Downward verts;
      int offset = i * nev;
      for (int j = 0; j < nev; ++j)
	verts[j] = globalToVert[conn[j + offset]];
      buildElement(m, interior, etype, verts);
    }
  }

  static apf::Gid getMax(const GlobalToVert& globalToVert, PCU_t PCUObj)
  {
    apf::Gid max = -1;
    APF_CONST_ITERATE(GlobalToVert, globalToVert, it)
      max = std::max(max, it->first);
    return PCU_Max_Int(PCUObj, max); // this is type-dependent
  }


  /* algorithm courtesy of Sebastian Rettenberger:
     use brokers/routers for the vertex global ids.
     Although we have used this trick before (see mpas/apfMPAS.cc),
     I didn't think to use it here, so credit is given. */
  static void constructResidence(Mesh2* m, GlobalToVert& globalToVert, PCU_t PCUObj)
  {
    Gid max = getMax(globalToVert, PCUObj);
    Gid total = max + 1;
    int peers = PCU_Comm_Peers(PCUObj);
    int quotient = total / peers;
    int remainder = total % peers;
    int mySize = quotient;
    int self = PCU_Comm_Self(PCUObj);
    if (self == (peers - 1))
      mySize += remainder;
    typedef std::vector< std::vector<int> > TmpParts;
    TmpParts tmpParts(mySize);
    /* if we have a vertex, send its global id to the
       broker for that global id */
    PCU_Comm_Begin(PCUObj);
    APF_ITERATE(GlobalToVert, globalToVert, it) {
      int gid = it->first;
      int to = std::min(peers - 1, gid / quotient);
      PCU_COMM_PACK(PCUObj, to, gid);
    }
    PCU_Comm_Send(PCUObj);
    int myOffset = self * quotient;
    /* brokers store all the part ids that sent messages
       for each global id */
    while (PCU_Comm_Receive(PCUObj)) {
      int gid;
      PCU_COMM_UNPACK(PCUObj, gid);
      int from = PCU_Comm_Sender(PCUObj);
      tmpParts.at(gid - myOffset).push_back(from);
    }
    /* for each global id, send all associated part ids
       to all associated parts */
    PCU_Comm_Begin(PCUObj);
    for (int i = 0; i < mySize; ++i) {
      std::vector<int>& parts = tmpParts[i];
      for (size_t j = 0; j < parts.size(); ++j) {
	int to = parts[j];
	int gid = i + myOffset;
	int nparts = parts.size();
	PCU_COMM_PACK(PCUObj, to, gid);
	PCU_COMM_PACK(PCUObj, to, nparts);
	for (size_t k = 0; k < parts.size(); ++k)
	  PCU_COMM_PACK(PCUObj, to, parts[k]);
      }
    }
    PCU_Comm_Send(PCUObj);
    /* receiving a global id and associated parts,
       lookup the vertex and classify it on the partition
       model entity for that set of parts */
    while (PCU_Comm_Receive(PCUObj)) {
      int gid;
      PCU_COMM_UNPACK(PCUObj, gid);
      int nparts;
      PCU_COMM_UNPACK(PCUObj, nparts);
      Parts residence;
      for (int i = 0; i < nparts; ++i) {
	int part;
	PCU_COMM_UNPACK(PCUObj, part);
	residence.insert(part);
      }
      MeshEntity* vert = globalToVert[gid];
      m->setResidence(vert, residence);
    }
  }

  /* given correct residence from the above algorithm,
     negotiate remote copies by exchanging (gid,pointer)
     pairs with parts in the residence of the vertex */
  static void constructRemotes(Mesh2* m, GlobalToVert& globalToVert, PCU_t PCUObj)
  {
    int self = PCU_Comm_Self(PCUObj);
    PCU_Comm_Begin(PCUObj);
    APF_ITERATE(GlobalToVert, globalToVert, it) {
      int gid = it->first;
      MeshEntity* vert = it->second;
      Parts residence;
      m->getResidence(vert, residence);
      APF_ITERATE(Parts, residence, rit)
	if (*rit != self) {
	  PCU_COMM_PACK(PCUObj, *rit, gid);
	  PCU_COMM_PACK(PCUObj, *rit, vert);
	}
    }
    PCU_Comm_Send(PCUObj);
    while (PCU_Comm_Receive(PCUObj)) {
      int gid;
      PCU_COMM_UNPACK(PCUObj, gid);
      MeshEntity* remote;
      PCU_COMM_UNPACK(PCUObj, remote);
      int from = PCU_Comm_Sender(PCUObj);
      MeshEntity* vert = globalToVert[gid];
      m->addRemote(vert, from, remote);
    }
  }

  void construct(Mesh2* m, const Gid* conn, const Gid* conn_b, int nelem,
		 int nelem_b, int nverts,int etype, int etype_b, proteus_index_t* local2globalMap,
		 GlobalToVert& globalToVert, PCU_t PCUObj)
  {
    constructVerts(m, nverts,local2globalMap,globalToVert);
    constructBoundaryElements(m, conn_b, nelem_b, etype_b, globalToVert);
    constructElements(m, conn, nelem, etype, globalToVert);
    constructResidence(m, globalToVert, PCUObj);
    constructRemotes(m, globalToVert, PCUObj);
    stitchMesh(m);
    m->acceptChanges();
  }

}

/**************************************************************************/


//The following functions are used to facilitate and perform a reconstruction
//of the proteus mesh into a SCOREC mesh to enable adaptivity features. 
//Currently, only 2D mesh reconstruction is supported.
//The basic strategy is to assume each exterior entity is a model entity since
//no geometric model is given. In Proteus, part boundary mesh entities are 
//considered exterior and so there needs to be logic to avoid classifying those.
//Each model entity should be unique and is associated with a material type.
//These material types are kept track via material arrays and the size of such 
//arrays are based on the total number of owned entities on each rank.
//
//There are some currently obsolete functionality for 2D model entity detection
//for mesh entities which will likely be developed/completed at a later time.
//
//To use, add the following to your case.py file (for example):
//
/*
  adaptMesh = True
  adaptMesh_nSteps = 10
  adaptMesh_numIter = 2
  MeshAdaptMesh=MeshAdaptPUMI.MeshAdaptPUMI(hmax=1.0, hmin=0.001, numIter=adaptMesh_numIter,sfConfig="ERM",logType="off",targetError=100,targetElementCount=8000)
  useModel=False
*/

#include "apf.h"
#include "gmi_null.h"
#include "gmi_mesh.h"
#include "gmi.h"
#include "apfMDS.h"
#include "apfMesh2.h"
#include "apfConvert.h"
#include "ma.h"
#include "PCU.h"

#include <cassert>
#include "gmi_lookup.h"

//Function to transfer some model information from NumericalSolution into the 
//MeshAdaptPUMIDrvr class.

int MeshAdaptPUMIDrvr::transferModelInfo(int* numGeomEntities, int* edges, int* faces, int* mVertex2Model, int*mEdge2Model, int*mBoundary2Model,int nMaxSegments){
  numModelEntities[0] = numGeomEntities[0];
  numModelEntities[1] = numGeomEntities[1];
  numModelEntities[2] = numGeomEntities[2];
  numModelEntities[3] = numGeomEntities[3];
  edgeList = edges;
  faceList = faces;
  meshVertex2Model = mVertex2Model;
  meshEdge2Model = mEdge2Model;
  meshBoundary2Model = mBoundary2Model;
  numSegments = nMaxSegments;
  return 0;
}

//Actual function to prompt recontruction and takes in the subodomain mesh and 
//the global mesh

int MeshAdaptPUMIDrvr::reconstructFromProteus(Mesh& mesh, Mesh& globalMesh,int hasModel)
{
  if(PCU_Comm_Self(PCUObj)==0)
    std::cout<<"STARTING RECONSTRUCTION\n";
  isReconstructed = 1; //True

  /************Preliminaries**************/
  comm_size = PCU_Comm_Peers(PCUObj);
  comm_rank = PCU_Comm_Self(PCUObj);

  int numModelNodes;
  int numModelEdges;
  int numModelBoundaries;
  int numModelRegions;

  int nBoundaryNodes=0; //number of total boundary nodes regardless of ownership
  proteus_index_t nNodes_owned = globalMesh.nodeOffsets_subdomain_owned[PCU_Comm_Self(PCUObj)+1]-globalMesh.nodeOffsets_subdomain_owned[PCU_Comm_Self(PCUObj)];

  for(int i =0;i<mesh.nNodes_global;i++){
    if(mesh.nodeMaterialTypes[i]>0){
      nBoundaryNodes++;
    }    
  }

  int numDim;
  if(mesh.nNodes_element==3)
    numDim = 2;
  else
    numDim = 3;

  //Depending on the dimension of the problem, exterior boundaries may refer to 
  //edges or faces
  if(hasModel){
    numModelNodes=numModelEntities[0];
    numModelEdges=numModelEntities[1];
    numModelBoundaries=numModelEntities[2];
    numModelRegions=numModelEntities[3];
    if(numDim=2){
      //should add some sort of assertion statement here
      numModelBoundaries = numModelEdges;
    }
  }
  else{
    numModelNodes = nBoundaryNodes;
    numModelEdges = static_cast<int>(mesh.nEdges_global);
    numModelBoundaries = static_cast<int>(mesh.nExteriorElementBoundaries_global);
    numModelRegions = numModelEntities[3];
  }
  
  assert(numModelRegions>0);

  numModelTotals[0] = numModelNodes;
  numModelTotals[1] = numModelEdges;
  numModelTotals[2] = numModelBoundaries;
  numModelTotals[3] = 0;//The total number of regions is known so no need to set a value
  PCU_Add_Ints(PCUObj, &numModelTotals[0],4); //get all offsets at the same time
  numModelTotals[3] = numModelRegions;

  /************Model Allocation**************/
  //This section starts the process to derive the geometric
  //model associated with the mesh
  
  gmi_model* gMod;

  struct gmi_base* gMod_base;
  gMod_base = (gmi_base*)malloc(sizeof(*gMod_base));
  gMod_base->model.ops = &gmi_base_ops;
  gmi_base_init(gMod_base);

  struct agm_ent e;
  struct agm_bdry b;
  struct agm_ent d;

  //gvertices
  gmi_base_reserve(gMod_base,AGM_VERTEX,numModelTotals[0]);

  if(numDim==2){
    //gedges
    gmi_base_reserve(gMod_base,AGM_EDGE,numModelTotals[2]);
    //gfaces
    gmi_base_reserve(gMod_base,AGM_FACE,numModelTotals[3]);
    //gregions
    gmi_base_reserve(gMod_base,AGM_REGION,0);
  }
  else if(numDim==3){
    //gedges
    gmi_base_reserve(gMod_base,AGM_EDGE,numModelTotals[1]);
    //gfaces
    gmi_base_reserve(gMod_base,AGM_FACE,numModelTotals[2]);
    //gregions
    gmi_base_reserve(gMod_base,AGM_REGION,numModelTotals[3]);
  }

  gMod = &gMod_base->model;

  /************Mesh Creation**************/
  //We can use apf::construct() which takes in a mapping of the elements
  //to their global vertices as well as boundary elements to their global 
  //vertices and outputs a topologically correct mesh. 
  //
  m = apf::makeEmptyMdsMesh(gMod,numDim,false,&pcuObj_);

  int etype,etype_b;
  int boundaryDim = numDim-1;
  apf::GlobalToVert outMap;
  if(numDim == 2){
    etype = apf::Mesh::TRIANGLE;
    etype_b = apf::Mesh::EDGE;
  }
  else{
    etype = apf::Mesh::TET;
    etype_b = apf::Mesh::TRIANGLE;
  }


  //create the mappings from proteus data structures
  apf::Gid* local2global_elementBoundaryNodes;
  local2global_elementBoundaryNodes = (apf::Gid*) malloc(sizeof(apf::Gid)*mesh.nElementBoundaries_global*apf::Mesh::adjacentCount[etype_b][0]);
  for(int i=0;i<mesh.nElementBoundaries_global*apf::Mesh::adjacentCount[etype_b][0];i++){ //should use adjacent count function from core
    local2global_elementBoundaryNodes[i] = globalMesh.nodeNumbering_subdomain2global[mesh.elementBoundaryNodesArray[i]];
  }
  apf::Gid* local2global_elementNodes;
  local2global_elementNodes = (apf::Gid*) malloc(sizeof(apf::Gid)*mesh.nElements_global*apf::Mesh::adjacentCount[etype][0]);
  for(int i=0;i<mesh.nElements_global*apf::Mesh::adjacentCount[etype][0];i++){ //should use adjacent count function from core
    local2global_elementNodes[i] = globalMesh.nodeNumbering_subdomain2global[mesh.elementNodesArray[i]];
  }
  
  //construct the mesh
  apf::construct(m,local2global_elementNodes,local2global_elementBoundaryNodes,
		 static_cast<int>(mesh.nElements_global),static_cast<int>(mesh.nElementBoundaries_global),static_cast<int>(mesh.nNodes_global),etype,etype_b,
		 globalMesh.nodeNumbering_subdomain2global,outMap,PCUObj);

  //Get the global model offsets after the mesh has been created
  //Need to get the number of owned element boundaries on the current rank
  //Also need to get the number of owned exterior entities for proper processor communication
  //This is necessary because a shared mesh entity should point to the same model entity

  nBoundaryNodes = 0;
  apf::MeshIterator* entIter=m->begin(0);
  apf::MeshEntity* ent;
  int idx = 0;
  while((ent=m->iterate(entIter))){
    if(m->isOwned(ent)){
      if(mesh.nodeMaterialTypes[idx]>0)
        nBoundaryNodes++;
    }
    idx++;
  }
  m->end(entIter);

  entIter=m->begin(boundaryDim);
  idx=0;
  int nExteriorElementBoundaries_owned = 0;
  while((ent=m->iterate(entIter))){
    if(m->isOwned(ent)){
      if(mesh.elementBoundaryMaterialTypes[idx]>0)
        nExteriorElementBoundaries_owned++;  
    }
    idx++;
  }
  m->end(entIter);

  if(hasModel){
    numModelNodes=numModelEntities[0];
    numModelEdges=numModelEntities[1];
    numModelBoundaries=numModelEntities[2];
    numModelRegions=numModelEntities[3];
    if(numDim=2){
      numModelBoundaries = numModelEdges;
    }
  }
  else{
    numModelNodes = nBoundaryNodes;
    numModelEdges = static_cast<int>(mesh.nEdges_global);
    numModelBoundaries = nExteriorElementBoundaries_owned;
    numModelRegions = numModelEntities[3]; 
  }

  numModelOffsets[0] = numModelNodes;
  numModelOffsets[1] = numModelEdges;
  numModelOffsets[2] = numModelBoundaries;
  numModelOffsets[3] = 0;
  
  numModelTotals[0] = numModelNodes;
  numModelTotals[1] = numModelEdges;
  numModelTotals[2] = numModelBoundaries;
  numModelTotals[3] = 0;

  //Get Region starting material
  entIter = m->begin(numDim);
  int regStartMaterial = 100;
  int rID = 0;
  while(ent = m->iterate(entIter)){
    if(mesh.elementMaterialTypes[rID] < regStartMaterial)
      regStartMaterial = mesh.elementMaterialTypes[rID];
    rID++;
  }
  m->end(entIter);


  //get all offsets at the same time
  PCU_Exscan_Ints(PCUObj, &numModelOffsets[0],4);
  PCU_Add_Ints(PCUObj, &numModelTotals[0],4); 
  numModelTotals[3] = numModelRegions;

  //classify mesh entities on model entities

  apf::Vector3 pt;

  apf::ModelEntity* g_vertEnt;
  apf::ModelEntity* g_edgeEnt;
  apf::ModelEntity* g_faceEnt;
  apf::MeshEntity* vertEnt;
  
  modelVertexMaterial = (int*)calloc(numModelTotals[0],sizeof(int));
  modelBoundaryMaterial = (int*)calloc(numModelTotals[2],sizeof(int));
  modelRegionMaterial = (int*)calloc(numModelTotals[3],sizeof(int));

  //gmi set entities
  //more entities were reserved than necessary, but that's okay

  gmi_unfreeze_lookups(gMod_base->lookup);
  for(int i=0;i<numModelTotals[0];i++){
    e = agm_add_ent(gMod_base->topo, AGM_VERTEX);
    gmi_set_lookup(gMod_base->lookup, e, i);
  }
  gmi_freeze_lookup(gMod_base->lookup, (agm_ent_type)0);

  for(int i=0;i<numModelTotals[2];i++){
    if(numDim==2)
      e = agm_add_ent(gMod_base->topo, AGM_EDGE);
    else
      e = agm_add_ent(gMod_base->topo, AGM_FACE);
    gmi_set_lookup(gMod_base->lookup, e, i);
  }
  gmi_freeze_lookup(gMod_base->lookup, (agm_ent_type)boundaryDim);

  for(int i=0;i<numModelRegions;i++){
    if(numDim == 2)
      e = agm_add_ent(gMod_base->topo, AGM_FACE);
    else
      e = agm_add_ent(gMod_base->topo, AGM_REGION);

    //assumes material types are enumerated starting from 1
    gmi_set_lookup(gMod_base->lookup, e, i+regStartMaterial); 
    if(hasModel){
      b = agm_add_bdry(gMod_base->topo, e);
      for(int k=0;k<numSegments;k++){
        if(faceList[(i)*numSegments+k]==-1) break;
        else{
          d = gmi_look_up(gMod_base->lookup,AGM_EDGE,faceList[(i)*numSegments+k]);
          agm_add_use(gMod_base->topo,b,d);
        }
      }
    }
  }
  gmi_freeze_lookup(gMod_base->lookup, (agm_ent_type)numDim);

  if(numDim==3){
    for(int i=0;i<numModelTotals[1];i++){
      e = agm_add_ent(gMod_base->topo, AGM_EDGE);
      gmi_set_lookup(gMod_base->lookup, e, i);
    }
    gmi_freeze_lookup(gMod_base->lookup, (agm_ent_type)1);
  }

  int matTag; 
  apf::ModelEntity* gEnt; 
  int vertCounter = numModelOffsets[0];

  //Iterate over the vertices and set the coordinates and model classification
  int vID = 0;
  entIter = m->begin(0);
  PCU_Comm_Begin(PCUObj);
  while(ent = m->iterate(entIter)){
    pt[0]=mesh.nodeArray[vID*3+0];
    pt[1]=mesh.nodeArray[vID*3+1];
    pt[2]=mesh.nodeArray[vID*3+2];
    m->setPoint(ent,0,pt);
    if(m->isOwned(ent)){
      matTag = mesh.nodeMaterialTypes[vID];
      if(hasModel){
        gEnt = m->findModelEntity(meshVertex2Model[2*vID+1],meshVertex2Model[2*vID]);
        //if entity is a model vertex
        if(meshVertex2Model[2*vID+1]==0) 
          modelVertexMaterial[meshVertex2Model[2*vID]] = matTag;
      }
      else{
        //if entity is interior, it should be classified on a region
        if(matTag==0){
          matTag = mesh.elementMaterialTypes[mesh.nodeElementsArray[mesh.nodeElementOffsets[vID]]];
          gEnt = m->findModelEntity(numDim,matTag);
        }
        //else there is an associated model entity
        else{
          gEnt = m->findModelEntity(0,vertCounter);
          modelVertexMaterial[vertCounter] = matTag;
          vertCounter++;  
        }
      }
      m->setModelEntity(ent,gEnt);
      //if the owner and entity is shared, share the model classification with other entities
      if(m->isShared(ent)){
        apf::Copies remotes;
        m->getRemotes(ent,remotes);
        for(apf::Copies::iterator it = remotes.begin(); it != remotes.end(); ++it){
          PCU_COMM_PACK(PCUObj, it->first,it->second);
          PCU_COMM_PACK(PCUObj, it->first,gEnt);
        }
      }

    } //endif owned
    vID++;
  }
  PCU_Comm_Send(PCUObj);
  //receive model entity classification from owning nodes
  while(PCU_Comm_Receive(PCUObj)){
    PCU_COMM_UNPACK(PCUObj, ent);
    PCU_COMM_UNPACK(PCUObj, gEnt);
    m->setModelEntity(ent,gEnt); 
  }
  PCU_Barrier(PCUObj);
  m->end(entIter);

  //Classify the mesh boundary entities
  //If the edge is on a model edge, it should have a material tag greater than 0.
  //If the edge is on a partition boundary, the material tag should be 0.
  //There is no direct control over ownership when constructing the mesh, so it
  //must be left general.
  int boundaryID = 0; //this is a counter for the set of boundary elements
  int boundaryCounter = 0; //this is a counter for the set of exterior boundary elements
  int boundaryMaterialCounter = numModelOffsets[2]; //this is a counter for the storage array used to map material types to the new mesh
  int edgCounter = numModelOffsets[1]; //this is a counter for the set of exterior edge entities
  apf::ModelEntity* edg_gEnt;
  entIter=m->begin(boundaryDim);
  PCU_Comm_Begin(PCUObj);
  while(ent = m->iterate(entIter)){
    if(hasModel){
      gEnt = m->findModelEntity(meshBoundary2Model[2*boundaryID+1],meshBoundary2Model[2*boundaryID]);
      //if entity is a on a model boundary
      if(meshBoundary2Model[2*boundaryID+1]==1) 
        modelBoundaryMaterial[meshBoundary2Model[2*boundaryID]] = mesh.elementBoundaryMaterialTypes[boundaryID];
    }
    else{
      if(mesh.exteriorElementBoundariesArray[boundaryCounter]==boundaryID && (mesh.elementBoundaryMaterialTypes[boundaryID]!=0)){
        gEnt = m->findModelEntity(boundaryDim,boundaryMaterialCounter);
        modelBoundaryMaterial[boundaryMaterialCounter] = mesh.elementBoundaryMaterialTypes[boundaryID]; 
        boundaryCounter++;
        boundaryMaterialCounter++;

        //If 3D, need to set exterior edge classification
        if(numDim==3){      
          apf::Adjacent adj_edges;
          m->getAdjacent(ent,1,adj_edges);
          for(int i=0;i<adj_edges.getSize();i++){
	    //If the edge is classified on a higher order entity than gEnt or if the edge hasn't been classified yet
            if(m->getModelType(m->toModel(adj_edges[i]))>m->getModelType(gEnt) || (m->getModelType(m->toModel(adj_edges[i]))==0)){
              edg_gEnt = m->findModelEntity(1,edgCounter);
              m->setModelEntity(adj_edges[i],edg_gEnt);
              edgCounter++; 
              //if the owner and entity is shared, share the model classification with other entities
              if(m->isOwned(adj_edges[i]) && m->isShared(adj_edges[i])){
                apf::Copies remotes;
                m->getRemotes(ent,remotes);
                for(apf::Copies::iterator it = remotes.begin(); it != remotes.end(); ++it){
                  PCU_COMM_PACK(PCUObj, it->first,it->second);
                  PCU_COMM_PACK(PCUObj, it->first,edg_gEnt);
                }
              }

            }
          }
        }

      }
      else {
        //If the current exterior entity is an edge on a partition boundary, need to check material and
        //get to the next item in the exterior array
        if(m->isShared(ent) && mesh.elementBoundaryMaterialTypes[mesh.exteriorElementBoundariesArray[boundaryCounter]]==0) 
          boundaryCounter++; 
        //assert((mesh.elementBoundaryMaterialTypes[boundaryID]==0 || numModelTotals[3]>1) && "If working with multi-region cases, turn this assertion off");
        //There are always two entities adjacent to an element boundary
        //Pick one and take that as the material type for classification
        matTag = mesh.elementMaterialTypes[mesh.elementBoundaryElementsArray[2*boundaryID]];
        gEnt = m->findModelEntity(numDim,matTag);
        //If 3D, need to set edge classification
        if(numDim==3){      
          apf::Adjacent adj_edges;
          m->getAdjacent(ent,1,adj_edges);
          for(int i=0;i<adj_edges.getSize();i++){
	    //If the edge is classified on a higher order entity than gEnt or if the edge hasn't been classified yet
            if(m->getModelType(m->toModel(adj_edges[i]))>m->getModelType(gEnt) || (m->getModelType(m->toModel(adj_edges[i]))==0)){
              m->setModelEntity(adj_edges[i],gEnt);
	      //if the owner and entity is shared, share the model classification with other entities
              if(m->isOwned(adj_edges[i]) && m->isShared(adj_edges[i])){
                apf::Copies remotes;
                m->getRemotes(ent,remotes);
                for(apf::Copies::iterator it = remotes.begin(); it != remotes.end(); ++it){
                  PCU_COMM_PACK(PCUObj, it->first,it->second);
                  PCU_COMM_PACK(PCUObj, it->first,gEnt);
                }
              }
            }
          }

        }
      }
    }
    m->setModelEntity(ent,gEnt);
    boundaryID++;
  }
  PCU_Comm_Send(PCUObj);
  //receive model entity classification from owning edges
  while(PCU_Comm_Receive(PCUObj)){
    PCU_COMM_UNPACK(PCUObj, ent);
    PCU_COMM_UNPACK(PCUObj, gEnt);
    m->setModelEntity(ent,gEnt); 
  }
  PCU_Barrier(PCUObj);

  m->end(entIter);

  //Iterate over regions

  //Populate the region materials
  //Assumes that the regions are numbered sequentially from 1 onward
  for(int i=0;i<numModelRegions;i++)
    modelRegionMaterial[i] = i+regStartMaterial;
  
  rID=0;
  entIter = m->begin(numDim);
  while(ent = m->iterate(entIter)){
    gEnt = m->findModelEntity(numDim,mesh.elementMaterialTypes[rID]);
    m->setModelEntity(ent,gEnt);
    rID++;
  }
  m->end(entIter);

  //Sum all of the material arrays to get the model-material mapping across all
  //ranks
  
  PCU_Add_Ints(PCUObj, modelVertexMaterial,numModelTotals[0]);
  PCU_Add_Ints(PCUObj, modelBoundaryMaterial,numModelTotals[2]);
  PCU_Add_Ints(PCUObj, modelRegionMaterial,numModelTotals[3]);

  //check that the mesh is consistent
  m->acceptChanges();
  apf::alignMdsRemotes(m);
  m->verify();
  initialReconstructed = 1;
  //renumber for compatibility with Proteus
  numberLocally();
  m->verify();

  //free mappings
  free(local2global_elementBoundaryNodes);
  free(local2global_elementNodes);

  if(PCU_Comm_Self(PCUObj)==0)
    std::cout<<"FINISHING RECONSTRUCTION\n";
}

int MeshAdaptPUMIDrvr::reconstructFromProteus2(Mesh& mesh,int* isModelVert,int* bFaces){

  //This function only applies for 3D meshes

  int dim;
  int elementType;
  if(mesh.nNodes_element == 3){
    dim = 2;
    elementType = apf::Mesh::TRIANGLE;
  }
  else{
    dim = 3;
    elementType = apf::Mesh::TET;
  }

  isReconstructed = 2;
  proteus_index_t nBFaces = mesh.nExteriorElementBoundaries_global;
  bool isModelVert_bool[mesh.nNodes_global];
  for(int i=0;i<mesh.nNodes_global;i++){
    isModelVert_bool[i] = isModelVert[i] != 0;
  }
  static int numEntries = 2+dim;

  int bEdges_1D[nBFaces][4];    
  int bFaces_2D[nBFaces][5];

  if(dim==2){
    for(int i=0;i<nBFaces;i++){
      int idx = i*numEntries;
      for(int j=0;j<numEntries;j++)
	bEdges_1D[i][j] = bFaces[idx+j];
    }
  }
  if(dim==3){
    for(int i=0;i<nBFaces;i++){
      int idx = i*numEntries;
      for(int j=0;j<numEntries;j++)
	bFaces_2D[i][j] = bFaces[idx+j];
    }
  }

  /*
    bFaces_2D[i][0] = bFaces[idx+0];
    bFaces_2D[i][1] = bFaces[idx+1];
    bFaces_2D[i][2] = bFaces[idx+2];
    bFaces_2D[i][3] = bFaces[idx+3];
    bFaces_2D[i][4] = bFaces[idx+4];
  */

  apf::GlobalToVert outMap;

  gmi_model* tempModel  = gmi_load(".null");
  m = apf::makeEmptyMdsMesh(tempModel,dim,false,&pcuObj_);
  std::valarray<apf::Gid> elementNodesArray(mesh.nElements_global*apf::Mesh::adjacentCount[elementType][0]);
  for(int i=0;i<mesh.nElements_global*apf::Mesh::adjacentCount[elementType][0];i++){
    elementNodesArray[i] = mesh.elementNodesArray[i];
  }
  apf::construct(m,&elementNodesArray[0],static_cast<int>(mesh.nElements_global),elementType,outMap);

  apf::setCoords(m,mesh.nodeArray,static_cast<int>(mesh.nNodes_global),outMap);

  std::map<int,apf::MeshEntity*> globalToRegion;
  apf::MeshIterator* it = m->begin(dim);
  apf::MeshEntity* ent;
  int counter = 0;
  while( ent = m->iterate(it) ){
    globalToRegion.insert(std::pair<int,apf::MeshEntity*> (counter,ent ));
    counter++;
  }

  if(dim == 2)
    apf::derive2DMdlFromManifold(m,isModelVert_bool,nBFaces,bEdges_1D,outMap,globalToRegion);
  else
    apf::deriveMdlFromManifold(m,isModelVert_bool,nBFaces,bFaces_2D,outMap,globalToRegion);
  m->writeNative("Reconstructed.smb");
  gmi_write_dmg(m->getModel(),"Reconstructed.dmg");
  std::cout<<"Finished Reconstruction, terminating program. Rerun with PUMI workflow\n";
  std::exit(0);

}


int MeshAdaptPUMIDrvr::reconstructFromProteusGlobal(int dim, int nNodes, double* nodes, int nElements, int* elements,
                                                    int* elementOwner, int* isModelVert, int nBFaces, int* bFaces){
  // reconstructFromProteus2 builds the PUMI mesh and derives the geometric model
  // from the whole boundary, so it cannot run on a subdomain: a partition
  // boundary would become a model face. Build on rank 0 alone, from the global
  // mesh, in a PCU of one rank; then spread it over the ranks with
  // apf::repeatMdsMesh, the way SCOREC's split tool does, by Proteus's own
  // element partition. The model is written to Reconstructed.dmg and loaded on
  // every rank so they all classify against the same one.
  const int nVertsPerElement = (dim == 2) ? 3 : 4;
  const int elementType = (dim == 2) ? apf::Mesh::TRIANGLE : apf::Mesh::TET;
  const bool isRoot = (PCU_Comm_Self(PCUObj) == 0);
  const int nRanks = PCU_Comm_Peers(PCUObj);
  int nGlobalElements = 0;

  std::unique_ptr<pcu::PCU> selfPCU;
  apf::Mesh2* ms = NULL;
  if(isRoot){
    selfPCU.reset(new pcu::PCU(MPI_COMM_SELF));
    gmi_model* tempModel = gmi_load(".null");
    ms = apf::makeEmptyMdsMesh(tempModel, dim, false, selfPCU.get());
    std::valarray<apf::Gid> elementNodesArray(nElements*nVertsPerElement);
    for(int i=0;i<nElements*nVertsPerElement;i++)
      elementNodesArray[i] = elements[i];
    apf::GlobalToVert outMap;
    apf::construct(ms, &elementNodesArray[0], nElements, elementType, outMap);
    apf::setCoords(ms, nodes, nNodes, outMap);

    std::map<int,apf::MeshEntity*> globalToRegion;
    apf::MeshIterator* it = ms->begin(dim);
    apf::MeshEntity* ent;
    int counter = 0;
    while( (ent = ms->iterate(it)) ){
      globalToRegion.insert(std::pair<int,apf::MeshEntity*>(counter, ent));
      counter++;
    }
    ms->end(it);

    std::unique_ptr<bool[]> isModelVert_bool(new bool[nNodes]);
    for(int i=0;i<nNodes;i++)
      isModelVert_bool[i] = isModelVert[i] != 0;

    const int numEntries = 2+dim;
    if(dim == 2){
      std::vector<int> bEdges(nBFaces*4);
      std::copy(bFaces, bFaces+nBFaces*numEntries, bEdges.begin());
      int (*b)[4] = reinterpret_cast<int(*)[4]>(bEdges.data());
      apf::derive2DMdlFromManifold(ms, isModelVert_bool.get(), nBFaces, b, outMap, globalToRegion);
    }
    else{
      std::vector<int> bFaces3(nBFaces*5);
      std::copy(bFaces, bFaces+nBFaces*numEntries, bFaces3.begin());
      int (*b)[5] = reinterpret_cast<int(*)[5]>(bFaces3.data());
      apf::deriveMdlFromManifold(ms, isModelVert_bool.get(), nBFaces, b, outMap, globalToRegion);
    }
    nGlobalElements = ms->count(dim);
    gmi_write_dmg(ms->getModel(), "Reconstructed.dmg");
  }
  // plain MPI: the root's PCU has its own communicator
  MPI_Allreduce(MPI_IN_PLACE, &nGlobalElements, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);

  gmi_model* model = gmi_load("Reconstructed.dmg");
  apf::Migration* plan = NULL;
  if(isRoot){
    plan = new apf::Migration(ms);
    apf::MeshIterator* it = ms->begin(dim);
    apf::MeshEntity* ent;
    int i = 0;
    while( (ent = ms->iterate(it)) ){
      plan->send(ent, elementOwner[i]);
      i++;
    }
    ms->end(it);
  }
  if(isRoot)
    ms->switchPCU(&pcuObj_);
  m = apf::repeatMdsMesh(ms, model, plan, nRanks, &pcuObj_);
  m->verify();
  m->writeNative("Reconstructed.smb");
  isReconstructed = 2;
  return nGlobalElements;
}
