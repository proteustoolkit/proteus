#include "meshio.h"
#include <string>
#include <fstream>
#include <cassert>
#include <cmath>
#include <iomanip>
#include <iostream>

namespace IOutils
{
std::istream& eatline(std::istream& s)
{
  while (s.get() != '\n' && s.good()) {}
  return s;
}

std::istream & eatchar(std::istream & s) 
{ s.get() ; return s ; }
bool iswhitespace(const char & c)
{ return c == '\n' || c == ' ' || c == '\t' || c == '\r';}
std::istream & eatcomments(std::istream & s) 
{
  char c = s.peek() ;
  //mwf add whitespace too, or just blank lines?
  while ( ( c == '\n' || c == '!' || c == '%' || c == '#' || c == ';' || c == '$')
	  && s.good() ) 
    { s >> eatline ; c = s.peek() ; }
  return s ;
}

}//IOUtils

namespace meshIO
{

/***********************************************************************

 **********************************************************************/

bool 
readTriangleMeshNodesAndElements(const char * filebase,
				 const int& indexBase,
				 int& nElements, int& nNodes,
				 std::vector<double>& nodeArray,
				 std::vector<int>& 
				 elementNodesArray,
				 std::vector<int>& nodeMaterialTypes,
				 std::vector<int>& elementMaterialTypes,
				 const int& defaultElementMaterialType,
				 const int& defaultNodeMaterialType)
{
  bool failed = false;
  const int simplexDim = 2+1;
  const int vertexDim  =3; //always
  using namespace IOutils;
  std::string vertexFileName  = std::string(filebase) + ".node" ;
  std::string elementFileName = std::string(filebase) + ".ele" ;

  std::ifstream vertexFile(vertexFileName.c_str());
  std::ifstream elementFile(elementFileName.c_str());
  
  if (!vertexFile.good())
    {
      std::cerr<<"readTriangleMeshNodeAndElements cannot open file "
	  <<vertexFileName<<std::endl;
      failed = true;
      return failed;
    }
  if (!elementFile.good())
    {
      std::cerr<<"readTriangleMeshNodeAndElements cannot open file "
	  <<elementFileName<<std::endl;
      failed = true;
      return failed;
    }

  int hasMarkers(0),hasAttributes(0),nSpace(2);
  //read vertices
  vertexFile >> eatcomments >> nNodes >> nSpace >> hasAttributes 
	     >> hasMarkers >> eatline ;
  assert(nNodes > 0);
  assert(nSpace == 2);
  if (hasAttributes > 0)
    {
      std::cerr<<"WARNING readTriangle nodes hasAttributes= "<<hasAttributes
	       <<" > 0 will treat first value as integer id for boundary!!"<<std::endl;
      hasMarkers = 1;
    }
  nodeArray.resize(vertexDim*nNodes);
  nodeMaterialTypes.resize(nNodes,defaultNodeMaterialType);

  for (int iv = 0; iv < nNodes; iv++)
    {
      int nv; double x,y; int nodeId(0);
      vertexFile >> eatcomments >> nv >> x >> y;
      if (hasMarkers > 0)
	vertexFile >> nodeId;
      nv -= indexBase;
      assert(0 <= nv && nv < nNodes && vertexFile.good());
      nodeArray[vertexDim*nv + 0] = x;
      nodeArray[vertexDim*nv + 1] = y;
      nodeArray[vertexDim*nv + 2] = 0.0;//could use a default
      if (hasMarkers > 0)
	nodeMaterialTypes[nv] = nodeId;
      vertexFile >> eatline;
    }//end iv
  vertexFile.close();
  
  //read elements
  int nNodesPerSimplex(simplexDim); hasMarkers = 0;
  elementFile >> eatcomments >> nElements >> nNodesPerSimplex >> hasMarkers  >> eatline;
  assert(nElements > 0);
  assert(nNodesPerSimplex == simplexDim); //not allow midface nodes yet
 
  elementNodesArray.resize(simplexDim*nElements);
  elementMaterialTypes.resize(nElements,defaultElementMaterialType);
  
  for (int ie = 0; ie < nElements; ie++)
    {
      int ne, nv, elementId(0);
      long double elementId_double;
      elementFile >> eatcomments >> ne;
      ne -= indexBase;
      assert(0 <= ne && ne < nElements && elementFile.good());
      for (int iv = 0; iv < simplexDim; iv++)
	{
	  elementFile >> nv ; nv -= indexBase;
	  assert(0 <= nv && nv < nNodes);
	  elementNodesArray[simplexDim*ne + iv] = nv;
#ifdef DEBUG_LEVEL_2
	  std::cout<<"readNodesAndMesh ie= "<<ie<<" "<<iv<<" ---> "
		   <<nv<<std::endl;
#endif
	}//end iv
      if (hasMarkers > 0)
	{
	  elementFile >> elementId_double;
          //elementId = lrintl(elementId_double);
          elementId = static_cast<long int>(elementId_double);
	  elementMaterialTypes[ne] = elementId;
	}
      elementFile >> eatline;
    }//end ie
  elementFile.close();

  return failed;
}//end readTriangleMesh

bool 
writeTriangleMeshNodesAndElements(const char * filebase,
				  const int& indexBase,
				  const int& nElements, const int& nNodes,
				  const double* nodeArray,
				  const int* elementNodesArray,
				  const int* nodeMaterialTypes,
				  const int* elementMaterialTypes)
{
  bool failed = false;
  const int vertexDim=3; const int simplexDim = 2+1;

  std::string vertexFileName  = std::string(filebase) + ".node" ;
  std::string elementFileName = std::string(filebase) + ".ele" ;

  std::ofstream vertexFile(vertexFileName.c_str());
  std::ofstream elementFile(elementFileName.c_str());
  
  if (!vertexFile.good())
    {
      std::cerr<<"writeTriangleMeshNodesAndElements cannot open file "
	  <<vertexFileName<<std::endl;
      failed = true;
      return failed;
    }
  if (!elementFile.good())
    {
      std::cerr<<"writeTriangleMeshNodeAndElements cannot open file "
	  <<elementFileName<<std::endl;
      failed = true;
      return failed;
    }

  int hasNodeMarkers(0);
  if (nodeMaterialTypes)
    hasNodeMarkers = 1;
  //write vertices, with every digit a double has (the stream's default, 6, moved nodes by up to 5e-7)
  vertexFile << std::setprecision(17);
  vertexFile << nNodes <<" 2 0 " << hasNodeMarkers <<" #number of vertices, dim nAttributes has Markers"<<std::endl;
  vertexFile <<"#vertex id x y [propid], base= "<<indexBase<<std::endl;
  for (int iv = 0; iv < nNodes; iv++)
    {
      int nv = iv + indexBase;
      double x = nodeArray[vertexDim*iv + 0];
      double y = nodeArray[vertexDim*iv + 1];
      //don't right z
      vertexFile << nv <<" " <<x <<" "<< y;
      if (hasNodeMarkers > 0)
	vertexFile <<" "<<nodeMaterialTypes[iv];
      vertexFile << std::endl;
    }//end iv
  vertexFile.close();
  
  //write elements
  int hasElementMarkers(0);
  if (elementMaterialTypes)
    hasElementMarkers = 1;
  elementFile << nElements <<" 3 "<< hasElementMarkers <<" #number of elements, nodes per triangle has Markers "<<std::endl;
  elementFile <<"#element id nodes 0 1 2, [propid]  base= "<<indexBase<<std::endl;
   for (int ie = 0; ie < nElements; ie++)
    {
      int ne = ie + indexBase;
      elementFile << ne;
      for (int iv = 0; iv < simplexDim; iv++)
	{
	  int nv = elementNodesArray[simplexDim*ie + iv];
	  nv += indexBase;
	  elementFile <<" "<< nv; 
	}//end iv
      if (hasElementMarkers)
	elementFile<<" "<<elementMaterialTypes[ie];
      elementFile << std::endl;
    }//end ie
  elementFile.close();

  return failed;
}//end writeTriangleMesh

/***********************************************************************
   just read in triangle edge file, in case we need element boundary
    identifiers
 **********************************************************************/

bool 
readTriangleElementBoundaries(const char * filebase,
			      const int& indexBase,
			      bool& hasMarkers,
			      int& nElementBoundaries,
			      std::vector<int>& elementBoundaryNodesArray,
			      std::vector<int>& elementBoundaryMaterialTypesArray,
			      const int& defaultBoundaryMaterialType)

{
  bool failed = false;
  const int elementBoundaryDim = 2;
  using namespace IOutils;
  std::string elementBoundaryFileName  = std::string(filebase) + ".edge" ;

  std::ifstream elementBoundaryFile(elementBoundaryFileName.c_str());
  
  if (!elementBoundaryFile.good())
    {
      std::cerr<<"readTriangleElementBoundaries cannot open file "
	  <<elementBoundaryFileName<<std::endl;
      failed = true;
      return failed;
    }

  hasMarkers = false;
  int ihasMarkers(0);
  //read vertices
  elementBoundaryFile >> eatcomments >> nElementBoundaries >> ihasMarkers >> eatline ;
  assert(nElementBoundaries > 0);
  if (ihasMarkers > 0)
    hasMarkers = true;
  elementBoundaryNodesArray.resize(elementBoundaryDim*nElementBoundaries);
  elementBoundaryMaterialTypesArray.resize(nElementBoundaries,defaultBoundaryMaterialType);

  for (int ieb = 0; ieb < nElementBoundaries; ieb++)
    {
      int neb,nn0,nn1; int ebId(0);
      elementBoundaryFile >> eatcomments >> neb >> nn0 >> nn1;
      if (ihasMarkers > 0)
	elementBoundaryFile >> ebId;
      neb -= indexBase;
      nn0 -= indexBase;
      nn1 -= indexBase;
      assert(0 <= neb && neb < nElementBoundaries && elementBoundaryFile.good());
      elementBoundaryNodesArray[elementBoundaryDim*neb + 0] = nn0;
      elementBoundaryNodesArray[elementBoundaryDim*neb + 1] = nn1;
      if (ihasMarkers > 0)
	elementBoundaryMaterialTypesArray[neb] = ebId;
      elementBoundaryFile >> eatline;
    }//end iv
  elementBoundaryFile.close();
  

  return failed;
}//end readTriangleElementBoundaries


bool 
writeTriangleElementBoundaryNodes(const char * filebase,
				  const int& indexBase,
				  const int& nElementBoundaries,
				  const int* elementBoundaryNodesArray,
				  const int* elementBoundaryMaterialTypes)
{
  bool failed = false;
  const int elementBoundaryDim=2;

  std::string elementBoundaryFileName  = std::string(filebase) + ".edge" ;

  std::ofstream elementBoundaryFile(elementBoundaryFileName.c_str());
  
  if (!elementBoundaryFile.good())
    {
      std::cerr<<"writeTriangleElementBoundaryNodes cannot open file "
	  <<elementBoundaryFileName<<std::endl;
      failed = true;
      return failed;
    }

  int hasElementBoundaryMarkers(0);
  if (elementBoundaryMaterialTypes)
    hasElementBoundaryMarkers = 1;
  //
  elementBoundaryFile << nElementBoundaries <<" " << hasElementBoundaryMarkers <<" #number of elementBoundaries, has Markers"<<std::endl;
  elementBoundaryFile <<"#elementBoundary node0 node1 [id], base= "<<indexBase<<std::endl;
  for (int ieb = 0; ieb < nElementBoundaries; ieb++)
    {
      int neb = ieb + indexBase;
      int nn0 = elementBoundaryNodesArray[elementBoundaryDim*ieb + 0];
      nn0 += indexBase;
      int nn1 = elementBoundaryNodesArray[elementBoundaryDim*ieb + 1];
      nn1 += indexBase;
      elementBoundaryFile << neb <<" " <<nn0 <<" "<< nn1;
      if (hasElementBoundaryMarkers > 0)
	elementBoundaryFile <<" "<<elementBoundaryMaterialTypes[ieb];
      elementBoundaryFile << std::endl;
    }//end iv
  elementBoundaryFile.close();

  return failed;
}//end writeTriangleElementBoundaryNodes

/***********************************************************************

 **********************************************************************/

bool 
readTetgenMeshNodesAndElements(const char * filebase,
			       const int& indexBase,
			       int& nElements, int& nNodes,
			       std::vector<double>& nodeArray,
			       std::vector<int>& 
			       elementNodesArray,
			       std::vector<int>& nodeMaterialTypes,
			       std::vector<int>& elementMaterialTypes,
			       const int& defaultElementMaterialType,
			       const int& defaultNodeMaterialType)
{
  bool failed = false;
  const int simplexDim = 3+1;
  const int vertexDim  = 3; //always
  using namespace IOutils;
  std::string vertexFileName  = std::string(filebase) + ".node" ;
  std::string elementFileName = std::string(filebase) + ".ele" ;

  std::ifstream vertexFile(vertexFileName.c_str());
  std::ifstream elementFile(elementFileName.c_str());
  
  if (!vertexFile.good())
    {
      std::cerr<<"readTetgenMeshNodeAndElements cannot open file "
	  <<vertexFileName<<std::endl;
      failed = true;
      return failed;
    }
  if (!elementFile.good())
    {
      std::cerr<<"readTetgenMeshNodeAndElements cannot open file "
	  <<elementFileName<<std::endl;
      failed = true;
      return failed;
    }

  int hasMarkers(0),hasAttributes(0),nSpace(3);
  //read vertices
  vertexFile >> eatcomments >> nNodes >> nSpace >> hasAttributes >> hasMarkers >> eatline ;
  assert(nNodes > 0);
  assert(nSpace == 3);
  if (hasAttributes > 0)
    {
      std::cerr<<"WARNING readTriangle nodes hasAttributes= "<<hasAttributes
	       <<" > 0 will treat first value as integer id for boundary!!"<<std::endl;
      hasMarkers = 1;
    }
  
  nodeArray.resize(vertexDim*nNodes);
  nodeMaterialTypes.resize(nNodes,defaultNodeMaterialType);

  for (int iv = 0; iv < nNodes; iv++)
    {
      int nv; double x,y,z; int nodeId(0);
      vertexFile >> eatcomments >> nv >> x >> y >> z;
      if (hasMarkers > 0)
	vertexFile >> nodeId;
      nv -= indexBase;
      assert(0 <= nv && nv < nNodes && vertexFile.good());
      nodeArray[vertexDim*nv + 0] = x;
      nodeArray[vertexDim*nv + 1] = y;
      nodeArray[vertexDim*nv + 2] = z;
      if (hasMarkers > 0)
	nodeMaterialTypes[nv] = nodeId;
      vertexFile >> eatline;
    }//end iv
  vertexFile.close();
  
  //read elements
  int nNodesPerSimplex(simplexDim); hasMarkers = 0;
  elementFile >> eatcomments >> nElements >> nNodesPerSimplex >> hasMarkers >> eatline;
  assert(nElements > 0);
  assert(nNodesPerSimplex == simplexDim); //not allow midface nodes yet
  
  elementNodesArray.resize(simplexDim*nElements);
  elementMaterialTypes.resize(nElements,defaultElementMaterialType);

  for (int ie = 0; ie < nElements; ie++)
    {
      int ne, nv, elementId(0);
      long double elementId_double;
      elementFile >> eatcomments >> ne;
      ne -= indexBase;
      assert(0 <= ne && ne < nElements && elementFile.good());
      for (int iv = 0; iv < simplexDim; iv++)
	{
	  elementFile >> nv ; nv -= indexBase;
	  assert(0 <= nv && nv < nNodes);
	  elementNodesArray[simplexDim*ne + iv] = nv;
#ifdef DEBUG_LEVEL_2
	  std::cout<<"readNodesAndMesh ie= "<<ie<<" "<<iv<<" ---> "
		   <<nv<<std::endl;
#endif
	}//end iv
      if (hasMarkers > 0)
	{
	  elementFile >> elementId_double;
          //elementId = lrintl(elementId_double);
          elementId = static_cast<long int>(elementId_double);
	  elementMaterialTypes[ne] = elementId;
	}
      elementFile >> eatline;
    }//end ie
  elementFile.close();

  return failed;
}//end readTetgenMesh

bool 
readTetgenElementBoundaries(const char * filebase,
			    const int& indexBase,
			    bool& hasMarkers,
			    int& nElementBoundaries,
			    std::vector<int>& elementBoundaryNodesArray,
			    std::vector<int>& elementBoundaryMaterialTypesArray,
			    const int& defaultBoundaryMaterialType)

{
  bool failed = false;
  const int elementBoundaryDim = 3;
  using namespace IOutils;
  std::string elementBoundaryFileName  = std::string(filebase) + ".face" ;

  std::ifstream elementBoundaryFile(elementBoundaryFileName.c_str());
  
  if (!elementBoundaryFile.good())
    {
      std::cerr<<"readTetgenElementBoundaries cannot open file "
	  <<elementBoundaryFileName<<std::endl;
      failed = true;
      return failed;
    }

  hasMarkers = false;
  int ihasMarkers(0);
  //read vertices
  elementBoundaryFile >> eatcomments >> nElementBoundaries >> ihasMarkers >> eatline ;
  assert(nElementBoundaries > 0);
  if (ihasMarkers > 0)
    hasMarkers = true;
  elementBoundaryNodesArray.resize(elementBoundaryDim*nElementBoundaries);
  elementBoundaryMaterialTypesArray.resize(nElementBoundaries,defaultBoundaryMaterialType);

  for (int ieb = 0; ieb < nElementBoundaries; ieb++)
    {
      int neb,nn0,nn1,nn2; int ebId(0);
      elementBoundaryFile >> eatcomments >> neb >> nn0 >> nn1 >> nn2;
      if (ihasMarkers > 0)
	elementBoundaryFile >> ebId;
      neb -= indexBase;
      nn0 -= indexBase;
      nn1 -= indexBase;
      nn2 -= indexBase;
      assert(0 <= neb && neb < nElementBoundaries && elementBoundaryFile.good());
      elementBoundaryNodesArray[elementBoundaryDim*neb + 0] = nn0;
      elementBoundaryNodesArray[elementBoundaryDim*neb + 1] = nn1;
      elementBoundaryNodesArray[elementBoundaryDim*neb + 2] = nn2;
      if (ihasMarkers > 0)
	elementBoundaryMaterialTypesArray[neb] = ebId;
      elementBoundaryFile >> eatline;
    }//end iv
  elementBoundaryFile.close();
  

  return failed;
}//end readTetgenElementBoundaries


bool 
writeTetgenMeshNodesAndElements(const char * filebase,
				const int& indexBase,
				const int& nElements, const int& nNodes,
				const double* nodeArray,
				const int* elementNodesArray,
				const int* nodeMaterialTypes,
				const int* elementMaterialTypes)
{
  bool failed = false;
  const int vertexDim=3; const int simplexDim = 3+1;

  std::string vertexFileName  = std::string(filebase) + ".node" ;
  std::string elementFileName = std::string(filebase) + ".ele" ;

  std::ofstream vertexFile(vertexFileName.c_str());
  std::ofstream elementFile(elementFileName.c_str());
  
  if (!vertexFile.good())
    {
      std::cerr<<"writeTetgenMeshNodesAndElements cannot open file "
	  <<vertexFileName<<std::endl;
      failed = true;
      return failed;
    }
  if (!elementFile.good())
    {
      std::cerr<<"writeTetgenMeshNodeAndElements cannot open file "
	  <<elementFileName<<std::endl;
      failed = true;
      return failed;
    }

  //write vertices
  int hasNodeMarkers(0);
  if (nodeMaterialTypes)
    hasNodeMarkers = 1;
  vertexFile << std::setprecision(17);   //every digit a double has (see the triangle writer)
  vertexFile << nNodes <<" 3 0 " << hasNodeMarkers <<" #number of vertices, dim nAttributes has Markers"<<std::endl;
  vertexFile <<"#vertex id x y z [propid], base= "<<indexBase<<std::endl;
  for (int iv = 0; iv < nNodes; iv++)
    {
      int nv = iv + indexBase;
      double x = nodeArray[vertexDim*iv + 0];
      double y = nodeArray[vertexDim*iv + 1];
      double z = nodeArray[vertexDim*iv + 2];
      //don't right z
      vertexFile << nv <<" " <<x <<" "<< y <<" " << z;
      if (hasNodeMarkers > 0)
	vertexFile <<" "<<nodeMaterialTypes[iv];
      vertexFile << std::endl;
    }//end iv
  vertexFile.close();
  
  //write elements
  int hasElementMarkers(0);
  if (elementMaterialTypes)
    hasElementMarkers = 1;
  elementFile << nElements <<" 4 "<< hasElementMarkers <<" #number of elements, nodes per simplex has Markers "<<std::endl;
  elementFile <<"#element id nodes 0 1 2 3, [propid]  base= "<<indexBase<<std::endl;
  for (int ie = 0; ie < nElements; ie++)
    {
      int ne = ie + indexBase;
      elementFile << ne;
      for (int iv = 0; iv < simplexDim; iv++)
	{
	  int nv = elementNodesArray[simplexDim*ie + iv];
	  nv += indexBase;
	  elementFile <<" "<< nv; 
	}//end iv
      if (hasElementMarkers)
	elementFile<<" "<<elementMaterialTypes[ie];
      elementFile << std::endl;
    }//end ie
  elementFile.close();

  return failed;
}//end writeTetgenMesh

bool 
writeTetgenElementBoundaryNodes(const char * filebase,
				const int& indexBase,
				const int& nElementBoundariesToWrite,
				const int* elementBoundaryNodesArray,
				const int* elementBoundaryMaterialTypes,
				const bool& writeExteriorElementBoundariesOnly,
				const int* exteriorElementBoundariesArray)
{
  bool failed = false;
  const int elementBoundaryDim=3;

  std::string elementBoundaryFileName  = std::string(filebase) + ".face" ;

  std::ofstream elementBoundaryFile(elementBoundaryFileName.c_str());
  
  if (!elementBoundaryFile.good())
    {
      std::cerr<<"writeTetgenElementBoundaryNodes cannot open file "
	  <<elementBoundaryFileName<<std::endl;
      failed = true;
      return failed;
    }

  int hasElementBoundaryMarkers(0);
  if (elementBoundaryMaterialTypes)
    hasElementBoundaryMarkers = 1;
  //
  if (writeExteriorElementBoundariesOnly)
    elementBoundaryFile << nElementBoundariesToWrite <<" " << hasElementBoundaryMarkers 
			<<" #number of exterior elementBoundaries, has Markers"<<std::endl;
  else
    elementBoundaryFile << nElementBoundariesToWrite <<" " << hasElementBoundaryMarkers 
			<<" #number of elementBoundaries, has Markers"<<std::endl;

  elementBoundaryFile <<"#elementBoundary node0 node1 node2 [id], base= "<<indexBase<<std::endl;
  if (writeExteriorElementBoundariesOnly)
    {
      assert(exteriorElementBoundariesArray);
      for (int iebe = 0; iebe < nElementBoundariesToWrite; iebe++)
	{
	  int niebe = iebe + indexBase; //number output by exterior boundary number
	  int ieb = exteriorElementBoundariesArray[iebe];
	  int nn0 = elementBoundaryNodesArray[elementBoundaryDim*ieb + 0];
	  nn0 += indexBase;
	  int nn1 = elementBoundaryNodesArray[elementBoundaryDim*ieb + 1];
	  nn1 += indexBase;
	  int nn2 = elementBoundaryNodesArray[elementBoundaryDim*ieb + 2];
	  nn2 += indexBase;
	  
	  elementBoundaryFile << niebe <<" " <<nn0 <<" "<< nn1 <<" "<< nn2;
	  if (hasElementBoundaryMarkers > 0)
	    elementBoundaryFile <<" "<<elementBoundaryMaterialTypes[ieb];
	  elementBoundaryFile << std::endl;
	}
    }
  else
    {
      for (int ieb = 0; ieb < nElementBoundariesToWrite; ieb++)
	{
	  int neb = ieb + indexBase;
	  int nn0 = elementBoundaryNodesArray[elementBoundaryDim*ieb + 0];
	  nn0 += indexBase;
	  int nn1 = elementBoundaryNodesArray[elementBoundaryDim*ieb + 1];
	  nn1 += indexBase;
	  int nn2 = elementBoundaryNodesArray[elementBoundaryDim*ieb + 2];
	  nn2 += indexBase;
	  elementBoundaryFile << neb <<" " <<nn0 <<" "<< nn1 <<" "<< nn2;
	  if (hasElementBoundaryMarkers > 0)
	    elementBoundaryFile <<" "<<elementBoundaryMaterialTypes[ieb];
	  elementBoundaryFile << std::endl;
	}//end iv
    }
  elementBoundaryFile.close();

  return failed;
}//end writeTriangleElementBoundaryNodes


/*************************************************************

************************************************************/
bool
write3dmMeshNodesAndElements(const char * filebase,
			     const int& indexBase,
			     const int& nElements, const int& nNodes,
			     const double* nodeArray,
			     const int* elementNodesArray,
			     const int* elementMaterialTypes)
{

  bool failed = false;
  const int vertexDim=3; const int simplexDim = 3+1;
  
  std::string meshFileName = std::string(filebase) + ".3dm";
  std::ofstream meshFile(meshFileName.c_str());

  if (!meshFile.good())
    {
      std::cerr<<"write3dmMeshNodesAndElements cannot open file "
	  <<meshFileName<<std::endl;
      failed = true;
      return failed;
    }
  meshFile <<"MESH3D"<<std::endl;
  for (int eN = 0; eN < nElements; eN++)
    {
      meshFile << "E4T "<<eN + indexBase 
	       <<" " <<elementNodesArray[eN*simplexDim + 0] + indexBase
	       <<" " <<elementNodesArray[eN*simplexDim + 1] + indexBase
	       <<" " <<elementNodesArray[eN*simplexDim + 2] + indexBase
	       <<" " <<elementNodesArray[eN*simplexDim + 3] + indexBase;
      if (!elementMaterialTypes)
	meshFile <<" 1 "<<std::endl;
      else
	meshFile <<" "<<elementMaterialTypes[eN]+indexBase<<std::endl;
    }
  meshFile << std::endl;
  meshFile << std::setiosflags(std::ios::scientific) << std::setprecision(8); 
  for (int nN = 0; nN < nNodes; nN++)
    {
      meshFile <<"ND "<<nN + indexBase
	       <<"  "<<nodeArray[nN*vertexDim + 0]
	       <<"  "<<nodeArray[nN*vertexDim + 1]
	       <<"  "<<nodeArray[nN*vertexDim + 2]
	       <<std::endl;
    }
  meshFile <<"END"<<std::endl;
  meshFile.close();
  return false;
}
			     
bool
write2dmMeshNodesAndElements(const char * filebase,
			     const int& indexBase,
			     const int& nElements, const int& nNodes,
			     const double* nodeArray,
			     const int* elementNodesArray,
			     const int* elementMaterialTypes)
{

  bool failed = false;
  const int vertexDim=3; const int simplexDim = 2+1;
  
  std::string meshFileName = std::string(filebase) + ".3dm";
  std::ofstream meshFile(meshFileName.c_str());

  if (!meshFile.good())
    {
      std::cerr<<"write2dmMeshNodesAndElements cannot open file "
	  <<meshFileName<<std::endl;
      failed = true;
      return failed;
    }
  meshFile <<"MESH2D"<<std::endl;
  for (int eN = 0; eN < nElements; eN++)
    {
      meshFile << "E3T"<<std::setw(11)<<eN + indexBase 
	       <<std::setw(11)<<elementNodesArray[eN*simplexDim + 0] + indexBase
	       <<std::setw(11)<<elementNodesArray[eN*simplexDim + 1] + indexBase
	       <<std::setw(11)<<elementNodesArray[eN*simplexDim + 2] + indexBase;
      if (!elementMaterialTypes)
	meshFile <<std::setw(11)<<1<<std::endl;
      else
	meshFile <<std::setw(11)<<elementMaterialTypes[eN]<<std::endl;
    }
  meshFile << std::setiosflags(std::ios::scientific) << std::setprecision(8); 
  for (int nN = 0; nN < nNodes; nN++)
    {
      meshFile <<"ND"<<std::setw(11)<<nN + indexBase
	       <<std::setw(17)<<nodeArray[nN*vertexDim + 0]
	       <<std::setw(17)<<nodeArray[nN*vertexDim + 1]
	       <<std::setw(17)<<nodeArray[nN*vertexDim + 2]
	       <<std::endl;
    }
  meshFile.close();
  return false;
}

}//meshIO
