#ifndef MESH_H
#define MESH_H
#include <set>
#include <vector>
#include <algorithm>
#include <map>
#include <list>
#include <iostream>
#include <iomanip>
#include <fstream>
/** #include <mach/mach_time.h> */
#include <cassert>
#include <cmath>
#include "index_type.h"

 
/**
   \file mesh.h
   \defgroup mesh mesh
   \brief A library of mesh generation and manipulation functions.
   
   @{
*/

extern "C"
{
  /* The simple array based interface to meshes */
  //forward declaration

  struct Mesh
  {
    //if you change this make sure to update  initialize and delete below
    proteus_index_t nElements_global,
      nNodes_global,
      nNodes_element,
      nNodes_elementBoundary,
      nElementBoundaries_element,
      nElementBoundaries_global,
      nInteriorElementBoundaries_global,
      nExteriorElementBoundaries_global,
      max_nElements_node,
      nEdges_global,
      max_nNodeNeighbors_node;
    
    proteus_index_t *elementNodesArray, //the nodes numbers for each element
      *nodeElementsArray,        //the element numbers for each node
      *nodeElementOffsets,       //offsets for indexing into nodeElementsArray,
      *elementNeighborsArray,    //the elment numbers for each neighboring element
      *elementBoundariesArray,   //the global elementBoundary numbers for each element
      *elementBoundaryNodesArray,
      *elementBoundaryElementsArray, //the element numbers for each element boundary
      *interiorElementBoundariesArray, //the element boundary numbers of the interior element boundaries
      *exteriorElementBoundariesArray, //the element boundary numbers of the exterior element boundaries
      *edgeNodesArray,
      *nodeStarArray,
      *nodeStarOffsets;

    //local numbers and tags, not mesh-sized: see index_type.h
    int *elementBoundaryLocalElementBoundariesArray, //the local element boundary number for the left and right neighbors of the element boundary
      *elementMaterialTypes,           //ids for classifying elements,element boundaries,nodes
      *elementBoundaryMaterialTypes,
      *nodeMaterialTypes;
    
    // NURBS
    int    *elementIJK;                // Cartesian indices of element 
    double *weights;                   // NURBS weight
    double *U_KNOT,*V_KNOT,*W_KNOT;    // Knot vectors 
    int    nx,ny,nz;
    int    px,py,pz;           
    // NURBS  
      
    double *nodeArray,*elementDiametersArray,*elementInnerDiametersArray,*elementBoundaryDiametersArray;
    double *elementBarycentersArray, *elementBoundaryBarycentersArray;
    double *nodeDiametersArray,*nodeSupportArray;
    double h,hMin,sigmaMax,volume;
    //for adaptive mesh refinement
    int * newestNodeBases;

    //for parallel computations
    
    proteus_index_t *elementOffsets_subdomain_owned,
      *elementNumbering_subdomain2global,
      *nodeOffsets_subdomain_owned,
      *nodeNumbering_subdomain2global,
      //permutation taking this mesh's original global node numbers to the
      //ones partitioning assigned, so arrays built before partitioning can be
      //read in the numbering subdomain2global refers to
      *nodeNumbering_global_old2new,
      *elementNumbering_global_old2new,
      *elementBoundaryOffsets_subdomain_owned,
      *elementBoundaryNumbering_subdomain2global,
      *edgeOffsets_subdomain_owned,
      *edgeNumbering_subdomain2global;

    // DMPlex mesh flag
    bool usePlex = false;
    
    Mesh* subdomainp;
  };
  
  inline void initializeMesh(Mesh& mesh)
  {
    //dimensions
    mesh.nElements_global=0;
    mesh.nNodes_global=0;
    mesh.nNodes_element=0;
    mesh.nNodes_elementBoundary=0;
    mesh.nElementBoundaries_element=0;
    mesh.nElementBoundaries_global=0;
    mesh.nInteriorElementBoundaries_global=0;
    mesh.nExteriorElementBoundaries_global=0;
    mesh.max_nElements_node=0;
    mesh.nEdges_global=0;
    //arrays
    mesh.elementNodesArray=NULL;
    mesh.nodeElementsArray=NULL;
    mesh.nodeElementOffsets=NULL;
    mesh.elementNeighborsArray=NULL;
    mesh.elementBoundariesArray=NULL;
    mesh.elementBoundaryNodesArray=NULL;
    mesh.elementBoundaryElementsArray=NULL;
    mesh.elementBoundaryLocalElementBoundariesArray=NULL;
    mesh.interiorElementBoundariesArray=NULL;
    mesh.exteriorElementBoundariesArray=NULL;
    mesh.edgeNodesArray=NULL;
    mesh.nodeStarArray=NULL;
    mesh.nodeStarOffsets=NULL;
    mesh.elementMaterialTypes=NULL;
    mesh.elementBoundaryMaterialTypes=NULL;
    mesh.nodeMaterialTypes=NULL;
    mesh.nodeArray=NULL;
    mesh.elementBarycentersArray=NULL;
    mesh.elementBoundaryBarycentersArray=NULL;
    mesh.nodeDiametersArray=NULL;
    mesh.nodeSupportArray=NULL;
    mesh.newestNodeBases=NULL;

    //parallel
    mesh.elementOffsets_subdomain_owned=NULL;
    mesh.elementNumbering_subdomain2global=NULL;
    mesh.nodeOffsets_subdomain_owned=NULL;
    mesh.nodeNumbering_subdomain2global=NULL;
    mesh.nodeNumbering_global_old2new=NULL;
    mesh.elementNumbering_global_old2new=NULL;
    mesh.elementBoundaryOffsets_subdomain_owned=NULL;
    mesh.elementBoundaryNumbering_subdomain2global=NULL;
    mesh.edgeOffsets_subdomain_owned=NULL;
    mesh.edgeNumbering_subdomain2global=NULL;

    // NURBS
    mesh.nx=mesh.ny=mesh.nz=0;
    mesh.px=mesh.py=mesh.pz=0;    
    mesh.elementIJK=NULL;              
    mesh.weights=NULL;               
    mesh.U_KNOT=NULL;
    mesh.V_KNOT=NULL;
    mesh.W_KNOT=NULL;     
    // NURBS 

    //geometry
    mesh.elementDiametersArray=NULL;
    mesh.h=0.0;
    mesh.hMin=0.0;
    mesh.sigmaMax=0.0;
    mesh.volume=0.0;
   
    //parallel
    mesh.elementOffsets_subdomain_owned=NULL;
    mesh.elementNumbering_subdomain2global=NULL;
    mesh.nodeOffsets_subdomain_owned=NULL;
    mesh.nodeNumbering_subdomain2global=NULL;
    mesh.nodeNumbering_global_old2new=NULL;
    mesh.elementNumbering_global_old2new=NULL;
    mesh.elementBoundaryOffsets_subdomain_owned=NULL;
    mesh.elementBoundaryNumbering_subdomain2global=NULL;
    mesh.edgeOffsets_subdomain_owned=NULL;
    mesh.edgeNumbering_subdomain2global=NULL;
    mesh.subdomainp=NULL;

  }

  inline void deleteMesh(Mesh& mesh)
  {
  	 	
    //dimensions
    mesh.nElements_global=0;
    mesh.nNodes_global=0;
    mesh.nNodes_element=0;
    mesh.nNodes_elementBoundary=0;
    mesh.nElementBoundaries_element=0;
    mesh.nElementBoundaries_global=0;
    mesh.nInteriorElementBoundaries_global=0;
    mesh.nExteriorElementBoundaries_global=0;
    mesh.max_nElements_node=0;
    mesh.nEdges_global=0;
    
    //arrays
    if(mesh.elementNodesArray!=NULL) delete [] mesh.elementNodesArray;
    if(mesh.nodeElementsArray!=NULL) delete [] mesh.nodeElementsArray;
    if(mesh.nodeElementOffsets!=NULL) delete [] mesh.nodeElementOffsets;
    if(mesh.elementNeighborsArray!=NULL) delete [] mesh.elementNeighborsArray;
    if(mesh.elementBoundariesArray!=NULL) delete [] mesh.elementBoundariesArray;
    if(mesh.elementBoundaryNodesArray!=NULL) delete [] mesh.elementBoundaryNodesArray;
    if(mesh.elementBoundaryElementsArray!=NULL) delete [] mesh.elementBoundaryElementsArray;
    if(mesh.elementBoundaryLocalElementBoundariesArray!=NULL) delete [] mesh.elementBoundaryLocalElementBoundariesArray;
    if(mesh.interiorElementBoundariesArray!=NULL) delete [] mesh.interiorElementBoundariesArray;
    if(mesh.exteriorElementBoundariesArray!=NULL) delete [] mesh.exteriorElementBoundariesArray;
    if(mesh.edgeNodesArray!=NULL) delete [] mesh.edgeNodesArray;
    if(mesh.nodeStarArray!=NULL) delete [] mesh.nodeStarArray;
    if(mesh.nodeStarOffsets!=NULL) delete [] mesh.nodeStarOffsets;
    if(mesh.elementMaterialTypes!=NULL) delete [] mesh.elementMaterialTypes;
    if(mesh.elementBoundaryMaterialTypes!=NULL) delete [] mesh.elementBoundaryMaterialTypes;
    if(mesh.nodeMaterialTypes!=NULL) delete [] mesh.nodeMaterialTypes;
    if(mesh.nodeArray!=NULL) delete [] mesh.nodeArray;
    if(mesh.elementDiametersArray!=NULL) delete [] mesh.elementDiametersArray;
    if(mesh.elementBarycentersArray!=NULL) delete [] mesh.elementBarycentersArray;
    if(mesh.elementBoundaryBarycentersArray!=NULL) delete [] mesh.elementBoundaryBarycentersArray;
    if(mesh.nodeDiametersArray!=NULL) delete [] mesh.nodeDiametersArray;
    if(mesh.nodeSupportArray!=NULL) delete [] mesh.nodeSupportArray;
    if(mesh.newestNodeBases!=NULL) delete [] mesh.newestNodeBases;
   
    // NURBS
    mesh.nx=mesh.ny=mesh.nz=0;
    mesh.px=mesh.py=mesh.pz=0;     
    if(mesh.elementIJK!=NULL) delete [] mesh.elementIJK;              
    if(mesh.weights!=NULL) delete [] mesh.weights;               
    if(mesh.U_KNOT!=NULL) delete [] mesh.U_KNOT;
    if(mesh.V_KNOT!=NULL) delete [] mesh.V_KNOT;
    if(mesh.W_KNOT!=NULL) delete [] mesh.W_KNOT;     
    // NURBS 
   
    //parallel
    if(mesh.elementOffsets_subdomain_owned!=NULL) delete [] mesh.elementOffsets_subdomain_owned;
    if(mesh.elementNumbering_subdomain2global!=NULL) delete [] mesh.elementNumbering_subdomain2global;
    if(mesh.nodeOffsets_subdomain_owned!=NULL) delete [] mesh.nodeOffsets_subdomain_owned;
    if(mesh.nodeNumbering_subdomain2global!=NULL) delete [] mesh.nodeNumbering_subdomain2global;
    if(mesh.nodeNumbering_global_old2new!=NULL) delete [] mesh.nodeNumbering_global_old2new;
    if(mesh.elementNumbering_global_old2new!=NULL) delete [] mesh.elementNumbering_global_old2new;
    if(mesh.elementBoundaryOffsets_subdomain_owned!=NULL) delete [] mesh.elementBoundaryOffsets_subdomain_owned;
    if(mesh.elementBoundaryNumbering_subdomain2global!=NULL) delete [] mesh.elementBoundaryNumbering_subdomain2global;
    if(mesh.edgeOffsets_subdomain_owned!=NULL) delete [] mesh.edgeOffsets_subdomain_owned;
    if(mesh.edgeNumbering_subdomain2global!=NULL) delete [] mesh.edgeNumbering_subdomain2global; 
    
    mesh.elementNodesArray=NULL;
    mesh.nodeElementsArray=NULL;
    mesh.nodeElementOffsets=NULL;
    mesh.elementNeighborsArray=NULL;
    mesh.elementBoundariesArray=NULL;
    mesh.elementBoundaryNodesArray=NULL;
    mesh.elementBoundaryElementsArray=NULL;
    mesh.elementBoundaryLocalElementBoundariesArray=NULL;
    mesh.interiorElementBoundariesArray=NULL;
    mesh.exteriorElementBoundariesArray=NULL;
    mesh.edgeNodesArray=NULL;
    mesh.nodeStarArray=NULL;
    mesh.nodeStarOffsets=NULL;
    mesh.elementMaterialTypes=NULL;
    mesh.elementBoundaryMaterialTypes=NULL;
    mesh.nodeMaterialTypes=NULL;
    mesh.nodeArray=NULL;
    mesh.elementBarycentersArray=NULL;
    mesh.elementBoundaryBarycentersArray=NULL;

    //parallel
    mesh.elementOffsets_subdomain_owned=NULL;
    mesh.elementNumbering_subdomain2global=NULL;
    mesh.nodeOffsets_subdomain_owned=NULL;
    mesh.nodeNumbering_subdomain2global=NULL;
    mesh.nodeNumbering_global_old2new=NULL;
    mesh.elementNumbering_global_old2new=NULL;
    mesh.elementBoundaryOffsets_subdomain_owned=NULL;
    mesh.elementBoundaryNumbering_subdomain2global=NULL;
    mesh.edgeOffsets_subdomain_owned=NULL;
    mesh.edgeNumbering_subdomain2global=NULL;
    mesh.subdomainp=NULL;
    
  }
  
  struct MultilevelMesh
  {
    int nLevels;
    Mesh* meshArray;
    int **elementParentsArray, //nLevels  x nElements_global_level
      **elementChildrenArray,
      **elementChildrenOffsets; //nLevels x nElements_global_level x nElementChildren_element
  };
  
  inline void initializeMultilevelMesh(MultilevelMesh& multilevelMesh)
  {
    multilevelMesh.nLevels=0;
    multilevelMesh.meshArray=NULL;
    multilevelMesh.elementParentsArray=NULL;
    multilevelMesh.elementChildrenArray=NULL;
    multilevelMesh.elementChildrenOffsets=NULL;
  }

  inline void deleteMultilevelMesh(MultilevelMesh& multilevelMesh)
  {
    for(int i=0;i<multilevelMesh.nLevels;i++)
      {
        deleteMesh(multilevelMesh.meshArray[i]);
        if(i>0)
          if (multilevelMesh.elementParentsArray[i] != NULL) delete [] multilevelMesh.elementParentsArray[i];
        if(i<multilevelMesh.nLevels-1)
          {
            if (multilevelMesh.elementChildrenArray[i] != NULL) delete [] multilevelMesh.elementChildrenArray[i];
            if (multilevelMesh.elementChildrenOffsets[i] != NULL) delete [] multilevelMesh.elementChildrenOffsets[i];
          }
      }
    if (multilevelMesh.meshArray != NULL) delete [] multilevelMesh.meshArray;
    if (multilevelMesh.elementParentsArray != NULL) delete [] multilevelMesh.elementParentsArray;
    if (multilevelMesh.elementChildrenArray != NULL) delete [] multilevelMesh.elementChildrenArray;
    if (multilevelMesh.elementChildrenOffsets != NULL) delete [] multilevelMesh.elementChildrenOffsets;
    
    multilevelMesh.nLevels=0;
    multilevelMesh.meshArray=NULL;
    multilevelMesh.elementParentsArray=NULL;
    multilevelMesh.elementChildrenArray=NULL;
    multilevelMesh.elementChildrenOffsets=NULL;
  }

  int edgeMeshElements(const int& nx, Mesh& mesh);
  int regularEdgeMeshNodes(const int& nx, const double& Lx, Mesh& mesh);
  int globallyRefineEdgeMesh(const int& nLevels, Mesh& mesh, MultilevelMesh& multilevelMesh, bool averageNewNodeFlags=false);
  int locallyRefineEdgeMesh(MultilevelMesh& multilevelMesh, 
			    int * elementTagArray);
  int locallyRefineTriangleMesh(MultilevelMesh& multilevelMesh, 
				int * elementTagArray);
  int locallyRefineTriangleMesh_4T(MultilevelMesh& multilevelMesh, 
				   int * elementTagArray);
  int locallyRefineTriangleMesh_redGreen(MultilevelMesh& multilevelMesh, 
				int * elementTagArray);
  int setNewestNodeBasesToLongestEdge(MultilevelMesh& multilevelMesh);

  int regularRectangularToTriangularMeshElements(const int& nx,const int& ny,Mesh& mesh, int triangleFlag);
  int regularRectangularToTriangularMeshNodes(const int& nx, const int& ny, const double& Lx, const double& Ly, Mesh& mesh);
  int regularRectangularToTriangularElementBoundaryMaterials(const double& Lx, const double& Ly, Mesh& mesh);
  int globallyRefineTriangularMesh(const int& nLevels, Mesh& mesh, MultilevelMesh& multilevelMesh, bool averageNewNodeFlags=false);

  int regularQuadrilateralMeshElements(const int& nx,const int& ny,Mesh& mesh);
  int regularQuadrilateralMeshElementBoundaryMaterials(const double& Lx, const double& Ly, Mesh& mesh);
  int globallyRefineQuadrilateralMesh(const int& nLevels, Mesh& mesh, MultilevelMesh& multilevelMesh, bool averageNewNodeFlags=false);


  int regularMeshNodes(const int& nx,const int& ny,const int& nz, const double& Lx, const double& Ly, const double& Lz, Mesh& mesh);
  int regularMeshNodes2D(const int& nx,const int& ny, const double& Lx, const double& Ly, Mesh& mesh);
  int regularHexahedralMeshElementBoundaryMaterials(const double& Lx, const double& Ly, const double& Lz, Mesh& mesh);
  int regularHexahedralToTetrahedralMeshNodes(const int& nx,const int& ny,const int& nz, const double& Lx, const double& Ly, const double& Lz, Mesh& mesh);
  int regularHexahedralToTetrahedralMeshElements(const int& nx,const int& ny,const int& nz,Mesh& mesh);
  int regularHexahedralToTetrahedralElementBoundaryMaterials(const double& Lx, const double& Ly, const double& Lz, Mesh& mesh);
  int regularHexahedralMeshElements(const int& nx,const int& ny,const int& nz,const int& px,const int& py,const int& pz, Mesh& mesh);
  int reorientTetrahedralMesh(Mesh& mesh);
  int regularNURBSMeshElements(const int& nx,const int& ny,const int& nz,const int& px,const int& py,const int& pz,Mesh& mesh);
  int globallyRefineHexahedralMesh(const int& nLevels, Mesh& mesh, MultilevelMesh& multilevelMesh, bool averageNewNodeFlags=false);

  int globallyRefineTetrahedralMesh(const int& nLevels, Mesh& mesh, MultilevelMesh& multilevelMesh, bool averageNewNodeFlags=false);

  int constructElementBoundaryElementsArray_edge(Mesh& mesh);
  int constructElementBoundaryElementsArray_triangle(Mesh& mesh);
  int constructElementBoundaryElementsArray_quadrilateral(Mesh& mesh);
  int constructElementBoundaryElementsArray_tetrahedron(Mesh& mesh);
  int constructElementBoundaryElementsArray_hexahedron(Mesh& mesh);
  int constructElementBoundaryElementsArray_NURBS(Mesh& mesh);

  int constructElementBoundaryElementsArrayWithGivenElementBoundaryNumbers_edge(Mesh& mesh);
  int constructElementBoundaryElementsArrayWithGivenElementBoundaryNumbers_triangle(Mesh& mesh);
  int constructElementBoundaryElementsArrayWithGivenElementBoundaryNumbers_quadrilateral(Mesh& mesh);
  int constructElementBoundaryElementsArrayWithGivenElementBoundaryNumbers_tetrahedron(Mesh& mesh);
    
  int constructElementBoundaryElementsArrayWithGivenElementBoundaryAndEdgeNumbers_edge(Mesh& mesh);
  int constructElementBoundaryElementsArrayWithGivenElementBoundaryAndEdgeNumbers_triangle(Mesh& mesh);
  int constructElementBoundaryElementsArrayWithGivenElementBoundaryAndEdgeNumbers_quadrilateral(Mesh& mesh);
  int constructElementBoundaryElementsArrayWithGivenElementBoundaryAndEdgeNumbers_tetrahedron(Mesh& mesh);
  int constructElementBoundaryElementsArrayWithGivenElementBoundaryAndEdgeNumbers_hexahedron(Mesh& mesh);
  int constructElementBoundaryElementsArrayWithGivenElementBoundaryAndEdgeNumbers_NURBS(Mesh& mesh);

  int writeElements(std::ostream& meshFile, const Mesh& mesh);
  int writeNodes(std::ostream& meshFile, const Mesh& mesh);
  int readElements(std::istream& meshFile, Mesh& mesh);
  
  int allocateGeometricInfo_tetrahedron(Mesh& mesh);
  int allocateGeometricInfo_triangle(Mesh& mesh);
  int allocateGeometricInfo_edge(Mesh& mesh);
  int allocateGeometricInfo_quadrilateral(Mesh& mesh);
  int allocateGeometricInfo_hexahedron(Mesh& mesh);
  int allocateGeometricInfo_NURBS(Mesh& mesh);

  int computeGeometricInfo_tetrahedron(Mesh& mesh);
  int computeGeometricInfo_triangle(Mesh& mesh);
  int computeGeometricInfo_edge(Mesh& mesh);
  int computeGeometricInfo_hexahedron(Mesh& mesh);   
  int computeGeometricInfo_quadrilateral(Mesh& mesh);
  int computeGeometricInfo_NURBS(Mesh& mesh);

  int assignElementBoundaryMaterialTypesFromParent(Mesh& parentMesh, Mesh& childMesh, const int* levelElementParentsArray,
						   const int& nSpace_global);
  int allocateNodeAndElementNodeDataStructures(Mesh& mesh, int nElements_global, int nNodes_global, int nNodes_element);
  //mwf added for converting from triangle data structure
  struct triangulateio;

  int setFromTriangleElements(triangulateio* trimesh, Mesh& mesh, int base);
  int setFromTriangleNodes(triangulateio* trimesh, Mesh& mesh, int base);
  int readTriangleMesh(Mesh& mesh, const char* filebase, int base);
  int readTriangleElementBoundaryMaterialTypes(Mesh& mesh, const char* filebase, int base);
  int writeTriangleMesh(Mesh& mesh, const char* filebase, int base); 
  int readTetgenMesh(Mesh& mesh, const char* filebase, int base);
  int readTetgenElementBoundaryMaterialTypes(Mesh& mesh, const char* filebase, int base);
  int writeTetgenMesh(Mesh& mesh, const char* filebase, int base);
  int readHex(Mesh& mesh, const char* filebase, int indexBase);
  int write3dmMesh(Mesh& mesh, const char* filebase, int base);
  int write2dmMesh(Mesh& mesh, const char* filebase, int base);
  int copyElementBoundaryMaterialTypesFromTriangle(triangulateio* trimesh, 
						   Mesh& mesh, int base);
              }//extern "C"
  
  // DMPlex mesh reader function declaration.
  // #include <pybind11/pybind11.h>
  // namespace py = pybind11;
  #include <Python.h>
  int readDMPlexMesh(PyObject* dmplexMesh, Mesh& mesh);

/* A sorted tuple of node numbers to use as a key in maps */
template<int nNodes>
class NodeTuple
{
public:
  inline NodeTuple(const int* nodesIn)
  {
    //sort
    int position;
    for(int i=0;i<nNodes;i++)
      {
        position=0;
        //dumb sorting seems fast enough for now
        for(int j=0;j<nNodes;j++)
          if(nodesIn[i] > nodesIn[j])
            position++;
        nodes[position]=nodesIn[i];
        nodes_unsorted[i] = nodesIn[i];
      }
//     for(int i=0;i<nNodes;i++)    
//       nodes[i] = nodesIn[i];
//     std::sort(nodes,nodes+nNodes);
  }
  inline NodeTuple(const NodeTuple<nNodes>& nt)
  {
    for(int i=0;i<nNodes;i++)
      {
        nodes[i]=nt.nodes[i]; 
        nodes_unsorted[i]=nt.nodes_unsorted[i];
      }
  }
  int nodes[nNodes];
  int nodes_unsorted[nNodes];
};

template<int nNodes>
inline bool operator<(const NodeTuple<nNodes>& left, const NodeTuple<nNodes>& right)
{
  for (int i=0;i<nNodes;i++)
    {
      if(left.nodes[i]<right.nodes[i])
        return true;
      else if(left.nodes[i]>right.nodes[i])
        return false;
    }
  return false;
}

class Node
{
public:
  int nN;
  double  x,y,z;
  int flag;
};

inline void midpoint(const double* left, const double* right, Node& midpoint)
{
  midpoint.x = 0.5*(left[0]+right[0]);
  midpoint.y = 0.5*(left[1]+right[1]);
  midpoint.z = 0.5*(left[2]+right[2]);
}

inline double edgeLength(const Node& left, const Node& right)
{
  return sqrt( (right.x - left.x)*(right.x - left.x) +
               (right.y - left.y)*(right.y - left.y) +
               (right.z - left.z)*(right.z - left.z) );
}

inline double edgeLengthFromNodeNumbers(double* nodeArray,const int& left, const int& right)
{
  return sqrt( (nodeArray[right*3+0] - nodeArray[left*3+0])*(nodeArray[right*3+0] - nodeArray[left*3+0]) +
               (nodeArray[right*3+1] - nodeArray[left*3+1])*(nodeArray[right*3+1] - nodeArray[left*3+1]) +
               (nodeArray[right*3+2] - nodeArray[left*3+2])*(nodeArray[right*3+2] - nodeArray[left*3+2]) );
}

inline bool operator<(const Node& left, const Node& right)
{
  if(left.x < right.x) 
    return true;
  else if (left.x > right.x)
    return false;
  else if (left.y < right.y)
    return true;
  else if (left.y > right.y)
    return false;
  else if (left.z < right.z)
    return true;
  else if (left.z > left.z)
    return false;
  return false;
}

class ElementNeighbors
{
public:
  inline ElementNeighbors(int leftIn,int left_ebN_elementIn){left=leftIn;left_ebN_element=left_ebN_elementIn;right=-1;right_ebN_element=-1;}
  inline ElementNeighbors(){left=-1;left_ebN_element=-1;right=-1;right_ebN_element=-1;}
  int left,left_ebN_element,right,right_ebN_element;
};

inline int newEdge(int eN,int* nodes,int n0,int n1)
{
  nodes[eN*2+0] = n0;
  nodes[eN*2+1] = n1;
  eN++;
  return eN;
}
inline int newTriangle(int eN,int* nodes,int n0,int n1,int n2)
{
  nodes[eN*3+0] = n0;
  nodes[eN*3+1] = n1;
  nodes[eN*3+2] = n2;
  eN++;
  return eN;
}

inline int newTetrahedron(int eN,int* nodes,int n0,int n1,int n2,int n3)
{
  nodes[eN*4+0] = n0;
  nodes[eN*4+1] = n1;
  nodes[eN*4+2] = n2;
  nodes[eN*4+3] = n3;
  eN++;
  return eN;
}

inline int newQuadrilateral(int eN,int* nodes,int n0,int n1,int n2,int n3)
{
  nodes[eN*4+0] = n0;
  nodes[eN*4+1] = n1;
  nodes[eN*4+2] = n2;
  nodes[eN*4+3] = n3;
  eN++;
  return eN;
}

inline int newHexahedron(int eN,int* nodes,int n0,int n1,int n2,int n3,int n4,int n5,int n6,int n7)
{
  nodes[eN*8+0] = n0;
  nodes[eN*8+1] = n1;
  nodes[eN*8+2] = n2;
  nodes[eN*8+3] = n3;
  nodes[eN*8+4] = n4;
  nodes[eN*8+5] = n5;
  nodes[eN*8+6] = n6;
  nodes[eN*8+7] = n7;
  eN++;
  return eN;
}

 
bool newestNodeBisect(int eN,
		      int& nElements_global,
		      int& nNodes_global,
		      std::vector<double>& nodeArray,
		      std::vector<int>& elementNodesArray,
		      std::vector<int>& elementNeighborsArray,
		      std::vector<std::list<int> >& childrenList,
		      std::vector<int>& elementParentsArray,
		      std::vector<int>& bases,
		      std::vector<bool>& refined);


bool add4TnodesForRefinement2d(int eN,//element to be refined
			       int& nNodes_global,//number of nodes in mesh, will grow as refine
			       std::vector<bool>& refined,  //is an element to be refined or not?
			       std::vector<int>& edgeMidNodesArray,//edge--> new node from bisection (-1 = none)
			       const int* elementNodesArray,    //parent mesh representation
			       const int* elementBoundariesArray,
			       const int* elementNeighborsArray,
			       const double * nodeArray);

bool add4TnodesForConformity2d(int eN, int ebN_longest, 
			     int ebN_neig,int eN_neig,
			       int& nNodes_global,
			       std::vector<bool>& refined,
			       std::vector<int>& edgeMidNodesArray,
			       const int* elementNodesArray,    //parent mesh representation
			       const int* elementBoundariesArray,
			       const int* elementNeighborsArray,
			       const double * nodeArray);

int findLocalLongestEdge2d(int eN,
			   const int* elementNodesArray,
			   const double * nodeArray);

bool subdivideTriangle4T(int eN_parent,
			 int& eN_new,
			 int* elementParentsArray,
			 int* elementChildrenOffsets,
			 int* elementChildrenArray,
			 int* elementNodesArray_child,
			 const std::vector<int>& edgeMidNodesArray,
			 const std::vector<bool>& refined,
			 const int* elementNodesArray_parent,
			 const int* elementBoundariesArray_parent,
			 const double* nodeArray_parent);


/** @} */
  
#endif
