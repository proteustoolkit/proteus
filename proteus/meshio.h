#ifndef MESH_IO_H
#define MESH_IO_H

#include <iostream>
#include "index_type.h"
#include <vector>
/***********************************************************************
  some basic utilities for reading formatted files
 **********************************************************************/
namespace IOutils
{
std::istream& eatline(std::istream& s);
std::istream& eatchar(std::istream& s);
std::istream& eatcomments(std::istream& s);
bool iswhitespace(const char& c);
} //end IOutils

namespace meshIO
{
/***********************************************************************
 \brief simple routine for reading in the vertex and element 
    information for a triangular mesh. Should be compatible with Triangle.
    Heavily borrowed from p2mesh

  @param filebase, IN. the base name for mesh files. The routine tries
     to open filebase.ele, filebase.node 
  @param indexBase, IN. The base numbering convetion (e.g., 0 or 1) 
    for the mesh

  @param nElements, OUT. The number of elements in the mesh
  @param nNodes, OUT. The number of nodes/vertices in the mesh
  
  @param nodeArray, OUT. physical coordinates of each node in mesh
    nodeArray[I,j] = x^j_I, coordinate j of global node I
    Dim = Nn x 3.

  @param elementNodesArray, OUT. Map from each element to its global
         node numbers elementNodesArray[I,j] = J, the global node
         number J of global element I's local node j 
         Dim: nElements x 3

  @param nodeMaterialTypes, OUT. optional flag for nodes (0 by default)
    nodeMaterialTypesArray[I] = f_I, 
    Dim = Nn .
  @param elementMaterialTypes, OUT. optional flag for elements (0 by default)
    elementMaterialTypesArray[I] = f_I, 
    Dim = nElements .
 **********************************************************************/

bool 
readTriangleMeshNodesAndElements(const char * filebase, 
				 const int& indexBase,
				 int& nElements, int& nNodes,
				 std::vector<double>& nodeArray,
				 std::vector<int>& 
				 elementNodesArray,
				 std::vector<int>& elementNodeTypesArray,
				 std::vector<int>& elementMaterialTypesArray,
				 const int& defaultElementMaterialType = 0,
				 const int& defaultNodeMaterialType = 0);

/***********************************************************************
 \brief simple routine for writing the vertex and element 
    information for a triangular mesh. Should be compatible with Triangle

  @param filebase, IN. the base name for mesh files. The routine tries
     to open filebase.ele, filebase.node 
  @param indexBase, IN. The base numbering convetion (e.g., 0 or 1) 
    for the mesh
    
  @param nElements, IN. The number of elements in the mesh
  @param nNodes, IN. The number of nodes/vertices in the mesh
  
  @param nodeArray, IN. physical coordinates of each node in mesh
    nodeArray[I,j] = x^j_I, coordinate j of global node I
    Dim = Nn x 3.

  @param elementNodesArray, IN. Map from each element to its global
         node numbers elementNodesArray[I,j] = J, the global node
         number J of global element I's local node j 
         Dim: nElements x 3

  @param nodeMaterialTypes, IN. optional flag for nodes (0 by default)
    nodeMaterialTypesArray[I] = f_I, 
    Dim = Nn .
  @param elementMaterialTypes, IN. optional flag for elements (0 by default)
    elementMaterialTypesArray[I] = f_I, 
    Dim = nElements .
 **********************************************************************/

bool 
writeTriangleMeshNodesAndElements(const char * filebase, 
				  const int& indexBase,
				  const proteus_index_t& nElements, const proteus_index_t& nNodes,
				  const double* nodeArray,
				  const proteus_index_t* elementNodesArray,
				  const int* nodeMaterialTypes = 0,
				  const int* elementMaterialTypes = 0);

/***********************************************************************
 \brief simple routine for reading in the element Boundary 
    information for a triangular mesh. Should be compatible with Triangle.

  @param filebase, IN. the base name for mesh files. The routine tries
     to open filebase.edge,
  @param indexBase, IN. The base numbering convetion (e.g., 0 or 1) 
    for the mesh

  @param nElementBoundaries, OUT. The number of elementBoundaries in the mesh
  
  @param elementBoundaryNodesArray, OUT. map from element boundary to global id of nodes defining it
    elementBoundaryNodesArray[I,j] = eb_I->node_j, local node j on edge I 
    Dim = nElementBoundaries x 2.

  @param elementBoundaryMaterialTypes, OUT. optional flag for elementBoundaries 
    elementBoundaryMaterialTypes[I] = f_I, 
    Dim = nElementBoundaries .
 **********************************************************************/

bool 
readTriangleElementBoundaries(const char * filebase, 
			      const int& indexBase,
			      bool& hasMarkers,
			      int& nElementBoundaries,
			      std::vector<int>& elementBoundaryNodesArray,
			      std::vector<int>& elementBoundaryMaterialTypesArray,
			      const int& defaultBoundaryMaterialType = 0);

/***********************************************************************
  write out element boundary nodes array and optionally element
  boundary identifiers. 
 ***********************************************************************/
bool 
writeTriangleElementBoundaryNodes(const char * filebase, 
				  const int& indexBase,
				  const proteus_index_t& nElementBoundaries,
				  const proteus_index_t* elementBoundaryNodesArray,
				  const int* elementBoundaryMaterialTypesArray = 0);


/***********************************************************************
 \brief simple routine for reading in the vertex and element 
    information for a tetrahedral mesh. Should be compatible with tetgen.

  @param filebase, IN. the base name for mesh files. The routine tries
     to open filebase.ele, filebase.node 
  @param indexBase, IN. The base numbering convetion (e.g., 0 or 1) 
    for the mesh

  @param nElements, OUT. The number of elements in the mesh
  @param nNodes, OUT. The number of nodes/vertices in the mesh
  
  @param nodeArray, OUT. physical coordinates of each node in mesh
    nodeArray[I,j] = x^j_I, coordinate j of global node I
    Dim = Nn x 3.

  @param elementNodesArray, OUT. Map from each element to its global
         node numbers elementNodesArray[I,j] = J, the global node
         number J of global element I's local node j 
         Dim: nElements x 4

  @param nodeMaterialTypes, OUT. optional flag for nodes (0 by default)
    nodeMaterialTypesArray[I] = f_I, 
    Dim = Nn .
  @param elementMaterialTypes, OUT. optional flag for elements (0 by default)
    elementMaterialTypesArray[I] = f_I, 
    Dim = nElements .
  **********************************************************************/

bool 
readTetgenMeshNodesAndElements(const char * filebase, 
			       const int& indexBase,
			       int& nElements, int& nNodes,
			       std::vector<double>& nodeArray,
			       std::vector<int>& elementNodesArray,
			       std::vector<int>& elementNodeTypesArray,
			       std::vector<int>& elementMaterialTypesArray,
			       const int& defaultElementMaterialType = 0,
			       const int& defaultNodeMaterialType = 0);



/***********************************************************************
 \brief simple routine for reading in the element Boundary 
    information for a triangular mesh. Should be compatible with Triangle.

  @param filebase, IN. the base name for mesh files. The routine tries
     to open filebase.edge,
  @param indexBase, IN. The base numbering convetion (e.g., 0 or 1) 
    for the mesh

  @param nElementBoundaries, OUT. The number of elementBoundaries in the mesh
  
  @param elementBoundaryNodesArray, OUT. map from element boundary to global id of nodes defining it
    elementBoundaryNodesArray[I,j] = eb_I->node_j, local node j on edge I 
    Dim = nElementBoundaries x 3.

  @param elementBoundaryMaterialTypes, OUT. optional flag for elementBoundaries 
    elementBoundaryMaterialTypes[I] = f_I, 
    Dim = nElementBoundaries .
 **********************************************************************/

bool 
readTetgenElementBoundaries(const char * filebase, 
			    const int& indexBase,
			    bool& hasMarkers,
			    int& nElementBoundaries,
			    std::vector<int>& elementBoundaryNodesArray,
			    std::vector<int>& elementBoundaryMaterialTypesArray,
			    const int& defaultBoundaryMaterialType = 0);


/***********************************************************************
 \brief simple routine for writing the vertex and element 
    information for a tetrahedral mesh. Should be compatible with Tetgen

  @param filebase, IN. the base name for mesh files. The routine tries
     to open filebase.ele, filebase.node 
  @param indexBase, IN. The base numbering convetion (e.g., 0 or 1) 
    for the mesh
    
  @param nElements, IN. The number of elements in the mesh
  @param nNodes, IN. The number of nodes/vertices in the mesh
  
  @param nodeArray, IN. physical coordinates of each node in mesh
    nodeArray[I,j] = x^j_I, coordinate j of global node I
    Dim = Nn x 3.

  @param elementNodesArray, IN. Map from each element to its global
         node numbers elementNodesArray[I,j] = J, the global node
         number J of global element I's local node j 
         Dim: nElements x 4

  @param nodeMaterialTypes, OUT. optional flag for nodes (0 by default)
    nodeMaterialTypesArray[I] = f_I, 
    Dim = Nn .
  @param elementMaterialTypes, OUT. optional flag for elements (0 by default)
    elementMaterialTypesArray[I] = f_I, 
    Dim = nElements .
  **********************************************************************/

bool 
writeTetgenMeshNodesAndElements(const char * filebase, 
				const int& indexBase,
				const proteus_index_t& nElements, const proteus_index_t& nNodes,
				const double* nodeArray,
				const proteus_index_t* elementNodesArray,
				const int* nodeMaterialTypes = 0,
				const int* elementMaterialTypes = 0);

/***********************************************************************
  write out element boundary nodes array and optionally element
  boundary identifiers. 
  flag writeExteriorBoundariesOnly determines if write only exterior
    boundaries or write them all.
  In nElementBoundaries should be nExteriorElementBoundaries_global if
    only writing out exterior boundaries and nElementBoundaries_global
    otherwise.
 ***********************************************************************/
bool 
writeTetgenElementBoundaryNodes(const char * filebase, 
				const int& indexBase,
				const proteus_index_t& nElementBoundaries,
				const proteus_index_t* elementBoundaryNodesArray,
				const int* elementBoundaryMaterialTypesArray = 0,
				const bool& writeExteriorBoundariesOnly = false,
				const proteus_index_t* exteriorElementBoundariesArray = 0);

/***********************************************************************
 \brief simple routine for writing the vertex and element 
    information for a tetrahedral mesh in a 3dm. 

  @param filebase, IN. the base name for mesh files. The routine tries
     to open filebase.3dm
  @param indexBase, IN. The base numbering convetion (e.g., 0 or 1) 
    for the mesh
    
  @param nElements, IN. The number of elements in the mesh
  @param nNodes, IN. The number of nodes/vertices in the mesh
  
  @param nodeArray, IN. physical coordinates of each node in mesh
    nodeArray[I,j] = x^j_I, coordinate j of global node I
    Dim = Nn x 3.

  @param elementNodesArray, IN. Map from each element to its global
         node numbers elementNodesArray[I,j] = J, the global node
         number J of global element I's local node j 
         Dim: nElements x 4
  **********************************************************************/

bool 
write3dmMeshNodesAndElements(const char * filebase, 
			     const int& indexBase,
			     const proteus_index_t& nElements, const proteus_index_t& nNodes,
			     const double* nodeArray,
			     const proteus_index_t* elementNodesArray,
			     const int* elementMaterialTypes = 0);



/***********************************************************************
 \brief simple routine for writing the vertex and element 
    information for a triangular mesh in a 2dm. 

  @param filebase, IN. the base name for mesh files. The routine tries
     to open filebase.2dm
  @param indexBase, IN. The base numbering convetion (e.g., 0 or 1) 
    for the mesh
    
  @param nElements, IN. The number of elements in the mesh
  @param nNodes, IN. The number of nodes/vertices in the mesh
  
  @param nodeArray, IN. physical coordinates of each node in mesh
    nodeArray[I,j] = x^j_I, coordinate j of global node I
    Dim = Nn x 3.

  @param elementNodesArray, IN. Map from each element to its global
         node numbers elementNodesArray[I,j] = J, the global node
         number J of global element I's local node j 
         Dim: nElements x 3
  **********************************************************************/

bool 
write2dmMeshNodesAndElements(const char * filebase, 
			     const int& indexBase,
			     const proteus_index_t& nElements, const proteus_index_t& nNodes,
			     const double* nodeArray,
			     const proteus_index_t* elementNodesArray,
			     const int* elementMaterialTypes = 0);


}

#endif
