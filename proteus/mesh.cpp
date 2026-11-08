#include "mesh.h"
#include "PyEmbeddedFunctions.h"
#define DEBUG_REFINE
/**
   \ingroup mesh mesh
   @{
*/
//****************************************************
#pragma mark -
#pragma mark * local ( static ) function prototypes *
//----------------------------------------------------
static double CurrentTime(void)
{
  //     static double scale = 0.0;
        
  //     if (0.0 == scale) {
  //         mach_timebase_info_data_t info;
  //         mach_timebase_info(&info);
  //         scale = info.numer / info.denom * 1e-9;
  //     }
        
  //     return mach_absolute_time() * scale;
  return 0.0;
}       // CurrentTime
//****************************************************
#pragma mark -
#pragma mark * exported function implementations *
//----------------------------------------------------

#ifndef REAL
#define REAL double
#define REAL_LOCAL
#endif
#ifdef REAL_LOCAL
#undef REAL
#endif

#include "meshio.h"
#include <set>
#include <valarray>
#include <string>
#include <sstream>
#include <iostream>
#include <fstream>
#include <cstring>

//mwftodo decide where to put mesh type tags
const int DEFAULT_ELEMENT_MATERIAL=0;
const int DEFAULT_NODE_MATERIAL=-1;
//a node the reader or generator gave no flag gets 0 ("no tag"), on the
//boundary too: a node's flag is its vertex marker or 0, never derived (the
//node-flag rule of 2026-11-06; the parun -F partitioners do the same). So
//does a marker -1, TetGen's for the vertices of segments, which equals
//DEFAULT_NODE_MATERIAL. The exterior faces still get
//EXTERIOR_ELEMENT_BOUNDARY_MATERIAL.
const int INTERIOR_NODE_MATERIAL=0;
const int EXTERIOR_NODE_MATERIAL=0;
const int INTERIOR_ELEMENT_BOUNDARY_MATERIAL=0;
const int EXTERIOR_ELEMENT_BOUNDARY_MATERIAL=1;

//todo compute geometric info, node star
extern "C"
{

  int edgeMeshElements(const int& nx, Mesh& mesh)
  {
    //todo put in check to see if memory was allocated, do something about material types
    mesh.nNodes_element = 2;
    mesh.nNodes_global = nx;
    mesh.nElements_global = nx-1;
    mesh.elementNodesArray = new int[mesh.nElements_global*mesh.nNodes_element];
    mesh.elementMaterialTypes = new int[mesh.nElements_global];
    memset(mesh.elementMaterialTypes,DEFAULT_ELEMENT_MATERIAL,mesh.nElements_global*sizeof(int));
    for(int i=0,eN=0;i<nx-1;i++)
      eN=newEdge(eN,mesh.elementNodesArray,i,i+1);
    return 0;
  }
  
  int regularEdgeMeshNodes(const int& nx, const double& Lx, Mesh& mesh)
  {
    const double hx=Lx/(nx-1.0);
    mesh.nodeArray = new double[mesh.nNodes_global*3];
    memset(mesh.nodeArray,0,nx*3*sizeof(double));
    mesh.nodeMaterialTypes = new int[mesh.nNodes_global];
    //set interior and exterior node material flags after get boundary info in
    //constructElementBoundaryElementsArray_edge 
    //if nodeMaterialTypes left as DEFAULT_NODE_MATERIAL
    memset(mesh.nodeMaterialTypes,DEFAULT_NODE_MATERIAL,mesh.nNodes_global*sizeof(int));
    for(int i=0;i<nx;i++)
      mesh.nodeArray[i*3+0]=i*hx;
    return 0;
  }
  
  int regularRectangularToTriangularMeshElements(const int& nx, const int& ny, Mesh& mesh,int triangleFlag)
  {
    mesh.nNodes_element = 3;
    mesh.nElements_global = 2*(nx-1)*(ny-1);
    mesh.nNodes_global = nx*ny;
    mesh.elementNodesArray = new int[mesh.nElements_global*mesh.nNodes_element];
    mesh.elementMaterialTypes = new int[mesh.nElements_global];
    memset(mesh.elementMaterialTypes,DEFAULT_ELEMENT_MATERIAL,mesh.nElements_global*sizeof(int));
    //for local refinement
    mesh.newestNodeBases = new int[mesh.nElements_global];

    for(int i=0,eN=0;i<ny-1;i++)
      for(int j=0;j<nx-1;j++)
	{
	  int 
	    n0 =(j+0) + (i+0)*nx,
	    n1 =(j+1) + (i+0)*nx,
	    n2 =(j+0) + (i+1)*nx,
	    n3 =(j+1) + (i+1)*nx;
	  if (triangleFlag == 2)//left leaning
	    {
	      eN=newTriangle(eN,mesh.elementNodesArray,n0,n1,n2);
	      eN=newTriangle(eN,mesh.elementNodesArray,n2,n1,n3);

	    }
	  else if (triangleFlag == 1)//union jack
	    {
	      if (i%2 + j%2 == 0 || i%2 + j%2 == 2)
		{
		  eN=newTriangle(eN,mesh.elementNodesArray,n0,n1,n3);
		  eN=newTriangle(eN,mesh.elementNodesArray,n0,n3,n2);
		}
	      else
		{
		  eN=newTriangle(eN,mesh.elementNodesArray,n0,n1,n2);
		  eN=newTriangle(eN,mesh.elementNodesArray,n2,n1,n3);
		}
	    }
	  else //right leaning
	    {
	      eN=newTriangle(eN,mesh.elementNodesArray,n0,n1,n3);
	      eN=newTriangle(eN,mesh.elementNodesArray,n0,n3,n2);
	    }
	  //NW element is (n0,n3,n1), SE element is (n0,n2,n3)
	  //eN was incremented twice just now
	  mesh.newestNodeBases[eN-2] = 2; //SE across from node n1
	  mesh.newestNodeBases[eN-1] = 1; //NW is across from node n2
	}
    return 0;
  }
  
  int regularRectangularToTriangularElementBoundaryMaterials(const double& Lx, const double& Ly, Mesh& mesh)
  {
    for (int ebNE = 0; ebNE < mesh.nExteriorElementBoundaries_global; ebNE++)
      {
	int ebN,nN_0,nN_1;
	double x_0,y_0,x_1,y_1,epsilon=1.0e-8;
	ebN = mesh.exteriorElementBoundariesArray[ebNE];
	nN_0 = mesh.elementBoundaryNodesArray[ebN*2 + 0];
	nN_1 = mesh.elementBoundaryNodesArray[ebN*2 + 1];
	x_0 = mesh.nodeArray[nN_0*3+0];
	y_0 = mesh.nodeArray[nN_0*3+1];
	x_1 = mesh.nodeArray[nN_1*3+0];
	y_1 = mesh.nodeArray[nN_1*3+1];
	if (y_0 <= epsilon && y_1 <= epsilon)
	  mesh.elementBoundaryMaterialTypes[ebN] = 1;
	else if (y_0 >= Ly - epsilon && y_1 >= Ly -  epsilon)
	  mesh.elementBoundaryMaterialTypes[ebN] = 3;
	else if (x_0 <= epsilon && x_1 <= epsilon)
	  mesh.elementBoundaryMaterialTypes[ebN] = 4;
	else if (x_0 >= Lx - epsilon && x_1 >= Lx - epsilon)
	  mesh.elementBoundaryMaterialTypes[ebN] = 2;
	else
	  assert(false);
      }
    return 0;
  }

  int regularRectangularToTriangularMeshNodes(const int& nx, const int& ny, const double& Lx, const double& Ly, Mesh& mesh)
  {
    const double hx=Lx/(nx-1.0),hy=Ly/(ny-1.0);
    mesh.nodeArray = new double[mesh.nNodes_global*3];
    memset(mesh.nodeArray,0,mesh.nNodes_global*3*sizeof(double));
    mesh.nodeMaterialTypes = new int[mesh.nNodes_global];
    //set interior and exterior node material flags after get boundary info in
    //constructElementBoundaryElementsArray_edge 
    //if nodeMaterialTypes left as DEFAULT_NODE_MATERIAL
    memset(mesh.nodeMaterialTypes,DEFAULT_NODE_MATERIAL,mesh.nNodes_global*sizeof(int));
    int nN;
    for(int i=0;i<ny;i++)
      for(int j=0;j<nx;j++)
        {
          nN = i*nx+j;
          mesh.nodeArray[3*nN+0]=j*hx;
          mesh.nodeArray[3*nN+1]=i*hy;
	  if (i==0)
	    mesh.nodeMaterialTypes[nN] = 1;
	  else if(i==ny-1)
	    mesh.nodeMaterialTypes[nN] = 3;
	  else if (j==0)
	    mesh.nodeMaterialTypes[nN] = 4;
	  else if(j==nx-1)
	    mesh.nodeMaterialTypes[nN] = 2;
	  else
	    mesh.nodeMaterialTypes[nN] = 0;
        }
    return 0;
  }
  
  inline void reorientNodes_tet(double* nodeArray, int* nodes)
  {
    int &n0(nodes[0]),&n1(nodes[1]),&n2(nodes[2]),&n3(nodes[3]);
    double t[3][3];
    t[0][0] = nodeArray[n1*3+0] - nodeArray[n0*3+0];
    t[0][1] = nodeArray[n1*3+1] - nodeArray[n0*3+1];
    t[0][2] = nodeArray[n1*3+2] - nodeArray[n0*3+2];
    
    t[1][0] = nodeArray[n2*3+0] - nodeArray[n0*3+0];
    t[1][1] = nodeArray[n2*3+1] - nodeArray[n0*3+1];
    t[1][2] = nodeArray[n2*3+2] - nodeArray[n0*3+2];
    
    t[2][0] = nodeArray[n3*3+0] - nodeArray[n0*3+0];
    t[2][1] = nodeArray[n3*3+1] - nodeArray[n0*3+1];
    t[2][2] = nodeArray[n3*3+2] - nodeArray[n0*3+2];
    
    double det = t[0][0]*(t[1][1]*t[2][2] - t[1][2]*t[2][1]) -
      t[0][1]*(t[1][0]*t[2][2] - t[1][2]*t[2][0]) +
      t[0][2]*(t[1][0]*t[2][1] - t[1][1]*t[2][0]);
    
    if (det < 0)
      {
	int tmp = n2;
	n2 = n3;
	n3 = tmp;
      }
  }
  
  int reorientTetrahedralMesh(Mesh& mesh)
  {
    for (int eN=0;eN<mesh.nElements_global;eN++)
      reorientNodes_tet(mesh.nodeArray,&mesh.elementNodesArray[eN*mesh.nNodes_element]);
    return 0;
  }
  
  int regularHexahedralToTetrahedralMeshElements(const int& nx, 
                                                 const int& ny, 
                                                 const int& nz, 
                                                 Mesh& mesh)
  {
    mesh.nNodes_element = 4;
    mesh.nNodes_global=nx*ny*nz;
    mesh.nElements_global = 6*(nx-1)*(ny-1)*(nz-1);
    int nxy=nx*ny;
    //std::cout<<"element nodes array"<<std::endl;
    mesh.elementNodesArray = new int[mesh.nElements_global*mesh.nNodes_element];
    mesh.elementMaterialTypes = new int[mesh.nElements_global];
    memset(mesh.elementMaterialTypes,DEFAULT_ELEMENT_MATERIAL,mesh.nElements_global*sizeof(int));
    for(int i=0,eN=0;i<nz-1;i++)
      for(int j=0;j<ny-1;j++)
        for(int k=0;k<nx-1;k++)
          {
            int 
              n0 = (k+0) + (j+0)*nx + (i+0)*nxy,
              n1 = (k+1) + (j+0)*nx + (i+0)*nxy,
              n2 = (k+0) + (j+1)*nx + (i+0)*nxy,
              n3 = (k+1) + (j+1)*nx + (i+0)*nxy,
              n4 = (k+0) + (j+0)*nx + (i+1)*nxy,
              n5 = (k+1) + (j+0)*nx + (i+1)*nxy,
              n6 = (k+0) + (j+1)*nx + (i+1)*nxy,
              n7 = (k+1) + (j+1)*nx + (i+1)*nxy;
            eN=newTetrahedron(eN,mesh.elementNodesArray,n0,n1,n3,n7);
            eN=newTetrahedron(eN,mesh.elementNodesArray,n0,n1,n5,n7);
            eN=newTetrahedron(eN,mesh.elementNodesArray,n0,n2,n3,n7);
            eN=newTetrahedron(eN,mesh.elementNodesArray,n0,n2,n6,n7);
            eN=newTetrahedron(eN,mesh.elementNodesArray,n0,n4,n5,n7);
            eN=newTetrahedron(eN,mesh.elementNodesArray,n0,n4,n6,n7);
          }
    return 0;
  }
  
  int regularHexahedralToTetrahedralElementBoundaryMaterials(const double& Lx, const double& Ly, const double& Lz, Mesh& mesh)
  {
    for (int ebNE = 0; ebNE < mesh.nExteriorElementBoundaries_global; ebNE++)
      {
	int ebN,nN_0,nN_1,nN_2;
	double x_0,y_0,z_0,
	  x_1,y_1,z_1,
	  x_2,y_2,z_2,
	  epsilon=1.0e-8;
	ebN = mesh.exteriorElementBoundariesArray[ebNE];
	nN_0 = mesh.elementBoundaryNodesArray[ebN*mesh.nNodes_elementBoundary + 0];
	nN_1 = mesh.elementBoundaryNodesArray[ebN*mesh.nNodes_elementBoundary + 1];
	nN_2 = mesh.elementBoundaryNodesArray[ebN*mesh.nNodes_elementBoundary + 2];

	x_0 = mesh.nodeArray[nN_0*3+0];
	y_0 = mesh.nodeArray[nN_0*3+1];
	z_0 = mesh.nodeArray[nN_0*3+2];

	x_1 = mesh.nodeArray[nN_1*3+0];
	y_1 = mesh.nodeArray[nN_1*3+1];	
	z_1 = mesh.nodeArray[nN_1*3+2];

	x_2 = mesh.nodeArray[nN_2*3+0];
	y_2 = mesh.nodeArray[nN_2*3+1];	
	z_2 = mesh.nodeArray[nN_2*3+2];

	if (z_0 <= epsilon && z_1 <= epsilon && z_2 <= epsilon)
	  mesh.elementBoundaryMaterialTypes[ebN] = 1;
	else if (z_0 >= Lz - epsilon && z_1 >= Lz -  epsilon && z_2 >= Lz -  epsilon)
	  mesh.elementBoundaryMaterialTypes[ebN] = 4;
	else if (y_0 <= epsilon && y_1 <= epsilon && y_2 <= epsilon)
	  mesh.elementBoundaryMaterialTypes[ebN] = 2;
	else if (y_0 >= Ly - epsilon && y_1 >= Ly -  epsilon && y_2 >= Ly -  epsilon)
	  mesh.elementBoundaryMaterialTypes[ebN] = 6;
	else if (x_0 <= epsilon && x_1 <= epsilon && x_2 <= epsilon)
	  mesh.elementBoundaryMaterialTypes[ebN] = 3;
	else if (x_0 >= Lx - epsilon && x_1 >= Lx - epsilon && x_2 >= Lx - epsilon)
	  mesh.elementBoundaryMaterialTypes[ebN] = 5;
	else
	  assert(false);
      }
    return 0;
  }
  
  int regularHexahedralMeshElementBoundaryMaterials(const double& Lx, const double& Ly, const double& Lz, Mesh& mesh)
  {
    for (int ebNE = 0; ebNE < mesh.nExteriorElementBoundaries_global; ebNE++)
      {
	int ebN,nN_0,nN_1,nN_2, nN_3;
	double x_0,y_0,z_0,
	  x_1,y_1,z_1,
	  x_2,y_2,z_2,
	  x_3,y_3,z_3,
	  epsilon=1.0e-8;
	ebN = mesh.exteriorElementBoundariesArray[ebNE];
	nN_0 = mesh.elementBoundaryNodesArray[ebN*mesh.nNodes_elementBoundary + 0];
	nN_1 = mesh.elementBoundaryNodesArray[ebN*mesh.nNodes_elementBoundary + 1];
	nN_2 = mesh.elementBoundaryNodesArray[ebN*mesh.nNodes_elementBoundary + 2];
	nN_3 = mesh.elementBoundaryNodesArray[ebN*mesh.nNodes_elementBoundary + 3];

	x_0 = mesh.nodeArray[nN_0*3+0];
	y_0 = mesh.nodeArray[nN_0*3+1];
	z_0 = mesh.nodeArray[nN_0*3+2];

	x_1 = mesh.nodeArray[nN_1*3+0];
	y_1 = mesh.nodeArray[nN_1*3+1];	
	z_1 = mesh.nodeArray[nN_1*3+2];

	x_2 = mesh.nodeArray[nN_2*3+0];
	y_2 = mesh.nodeArray[nN_2*3+1];	
	z_2 = mesh.nodeArray[nN_2*3+2];
        
	x_3 = mesh.nodeArray[nN_3*3+0];
	y_3 = mesh.nodeArray[nN_3*3+1];	
	z_3 = mesh.nodeArray[nN_3*3+2];

	if (z_0 <= epsilon && z_1 <= epsilon && z_2 <= epsilon && z_3 <= epsilon)
	  mesh.elementBoundaryMaterialTypes[ebN] = 1;
	else if (z_0 >= Lz - epsilon && z_1 >= Lz -  epsilon && z_2 >= Lz - epsilon && z_3 >= Lz -  epsilon)
	  mesh.elementBoundaryMaterialTypes[ebN] = 4;
	else if (y_0 <= epsilon && y_1 <= epsilon && y_2 <= epsilon && y_3 <= epsilon)
	  mesh.elementBoundaryMaterialTypes[ebN] = 2;
	else if (y_0 >= Ly - epsilon && y_1 >= Ly -  epsilon && y_2 >= Ly -  epsilon && y_3 >= Ly -  epsilon)
	  mesh.elementBoundaryMaterialTypes[ebN] = 6;
	else if (x_0 <= epsilon && x_1 <= epsilon && x_2 <= epsilon && x_3 <= epsilon)
	  mesh.elementBoundaryMaterialTypes[ebN] = 3;
	else if (x_0 >= Lx - epsilon && x_1 >= Lx - epsilon && x_2 >= Lx - epsilon && x_3 >= Lx - epsilon)
	  mesh.elementBoundaryMaterialTypes[ebN] = 5;
	else
	  assert(false);
      }
    return 0;
  }

  int regularQuadrilateralMeshElementBoundaryMaterials(const double& Lx, const double& Ly, Mesh& mesh)
  {
    int result = regularRectangularToTriangularElementBoundaryMaterials(Lx, Ly, mesh);
    return result;
  }
  
  int regularMeshNodes(const int& nx, 
                       const int& ny, 
                       const int& nz,
                       const double& Lx, 
                       const double& Ly, 
                       const double& Lz,
                       Mesh& mesh)
  {
    const int nxy=nx*ny;
    const double hx=Lx/(nx-1.0),
      hy=Ly/(ny-1.0),
      hz=Lz/(nz-1.0);
    mesh.nNodes_global=nx*ny*nz;
    //std::cout<<"node array length "<<mesh.nNodes_global*3<<std::endl;
    mesh.nodeArray = new double[mesh.nNodes_global*3];
    //std::cout<<"memset "<<std::endl;
    memset(mesh.nodeArray,0,mesh.nNodes_global*3*sizeof(double));
    //std::cout<<"mat types"<<std::endl;
    mesh.nodeMaterialTypes = new int[mesh.nNodes_global];
    //set interior and exterior node material flags after get boundary info in
    //constructElementBoundaryElementsArray_edge 
    //if nodeMaterialTypes left as DEFAULT_NODE_MATERIAL
    //std::cout<<"memset"<<std::endl;
    memset(mesh.nodeMaterialTypes,DEFAULT_NODE_MATERIAL,mesh.nNodes_global*sizeof(int));
    int nN;
    for(int i=0;i<nz;i++)
      for(int j=0;j<ny;j++)
        for(int k=0;k<nx;k++)
	  {
	    nN = k + j*nx + i*nxy;
	    mesh.nodeArray[3*nN+0]=k*hx;
	    mesh.nodeArray[3*nN+1]=j*hy;
	    mesh.nodeArray[3*nN+2]=i*hz;
	    if (i==0)
	      mesh.nodeMaterialTypes[nN] = 1;
	    else if(i==nz-1)
	      mesh.nodeMaterialTypes[nN] = 4;
	    else if (j==0)
	      mesh.nodeMaterialTypes[nN] = 2;
	    else if(j==ny-1)
	      mesh.nodeMaterialTypes[nN] = 6;
	    else if (k==0)
	      mesh.nodeMaterialTypes[nN] = 3;
	    else if(k==nx-1)
	      mesh.nodeMaterialTypes[nN] = 5;
	    else
	      mesh.nodeMaterialTypes[nN] = 0;
	  }
    return 0;
  }

  int regularMeshNodes2D(const int& nx, 
                         const int& ny,
                         const double& Lx, 
                         const double& Ly, 
                         Mesh& mesh)
  {
    const double hx=Lx/(nx-1.0),
      hy=Ly/(ny-1.0);
    mesh.nNodes_global=nx*ny;   
    mesh.nodeArray = new double[mesh.nNodes_global*3];
    memset(mesh.nodeArray,0,mesh.nNodes_global*3*sizeof(double));
    mesh.nodeMaterialTypes = new int[mesh.nNodes_global];
    //set interior and exterior node material flags after get boundary info in
    //constructElementBoundaryElementsArray_edge 
    //if nodeMaterialTypes left as DEFAULT_NODE_MATERIAL
    memset(mesh.nodeMaterialTypes,DEFAULT_NODE_MATERIAL,mesh.nNodes_global*sizeof(int));
    int nN;
    for(int j=0;j<ny;j++)
      for(int k=0;k<nx;k++)
        {
          nN = k*ny + j;
          mesh.nodeArray[3*nN+0]=k*hx;
          mesh.nodeArray[3*nN+1]=j*hy;
          mesh.nodeArray[3*nN+2]=0.0;
          if (j==0)
            mesh.nodeMaterialTypes[nN] = 1;
          else if(j==ny-1)
            mesh.nodeMaterialTypes[nN] = 3;
          else if (k==0)
            mesh.nodeMaterialTypes[nN] = 4;
          else if(k==nx-1)
            mesh.nodeMaterialTypes[nN] = 2;
          else
            mesh.nodeMaterialTypes[nN] = 0;
        }
    return 0;
  }

  int regularHexahedralToTetrahedralMeshNodes(const int& nx, 
                                              const int& ny, 
                                              const int& nz,
                                              const double& Lx, 
                                              const double& Ly, 
                                              const double& Lz,
                                              Mesh& mesh)
  {
    regularMeshNodes(nx,ny,nz,Lx,Ly,Lz,mesh);  	   
    //std::cout<<"regularHexahedralToTetrahedralMeshNodes is Deprecated\n";    
    return 0;                                     
  }                                            
    
  int regularHexahedralMeshElements(const int& nx, 
                                    const int& ny, 
                                    const int& nz, 
                                    const int& px, 
                                    const int& py, 
                                    const int& pz, 
                                    Mesh& mesh)
  {
    std::cout<<"mesh elements"<<std::endl;
    mesh.nNodes_element = 8;
    mesh.nNodes_global=nx*ny*nz;
    mesh.nElements_global = (nx-1)*(ny-1)*(nz-1);
    int nxy=nx*ny;

    mesh.nx=nx;
    mesh.ny=ny;
    mesh.nz=nz;

    //mesh.px=px;
    //mesh.py=py;
    //mesh.pz=pz;

    mesh.elementNodesArray = new int[mesh.nElements_global*mesh.nNodes_element];
    mesh.elementMaterialTypes = new int[mesh.nElements_global];
    memset(mesh.elementMaterialTypes,DEFAULT_ELEMENT_MATERIAL,mesh.nElements_global*sizeof(int));
    int eN=0;
    for(int i=0;i<nz-1;i++)
      for(int j=0;j<ny-1;j++)
        for(int k=0;k<nx-1;k++)
          {
            eN=newHexahedron(eN,mesh.elementNodesArray,
			     (k+0) + (j+0)*nx + (i+0)*nxy,
			     (k+1) + (j+0)*nx + (i+0)*nxy,
			     (k+1) + (j+1)*nx + (i+0)*nxy,
			     (k+0) + (j+1)*nx + (i+0)*nxy,
			     (k+0) + (j+0)*nx + (i+1)*nxy,
			     (k+1) + (j+0)*nx + (i+1)*nxy,
			     (k+1) + (j+1)*nx + (i+1)*nxy,
			     (k+0) + (j+1)*nx + (i+1)*nxy);
          }
    return 0;
  }
  
  int regularQuadrilateralMeshElements(const int& nx, 
                                       const int& ny,
                                       Mesh& mesh)
  {
    mesh.nNodes_element = 4;
    mesh.nNodes_global=nx*ny;
    mesh.nElements_global = (nx-1)*(ny-1);

    mesh.nx=nx;
    mesh.ny=ny;
    mesh.nz=1;

    mesh.elementNodesArray = new int[mesh.nElements_global*mesh.nNodes_element];
    mesh.elementMaterialTypes = new int[mesh.nElements_global];
    memset(mesh.elementMaterialTypes,DEFAULT_ELEMENT_MATERIAL,mesh.nElements_global*sizeof(int));
    int eN=0;
    for(int j=0;j<ny-1;j++)
      for(int k=0;k<nx-1;k++)
        {
          eN=newQuadrilateral(eN,mesh.elementNodesArray,
                              (k+0)*ny + (j+0),
                              (k+0)*ny + (j+1),
                              (k+1)*ny + (j+1),
                              (k+1)*ny + (j+0));
        }
    return 0;
  }
  
  int regularNURBSMeshElements(const int& nx, 
			       const int& ny, 
			       const int& nz, 
			       const int& px, 
			       const int& py, 
			       const int& pz, 
			       Mesh& mesh)
  {

    mesh.nNodes_element = (px+1)*(py+1)*(pz+1);
    mesh.nNodes_global=nx*ny*nz;
    mesh.nElements_global = (nx-px)*(ny-py)*(nz-pz);
    
    mesh.nx=nx;
    mesh.ny=ny;
    mesh.nz=nz;
    mesh.px=px;
    mesh.py=py;
    mesh.pz=pz;
    
    int nxy=nx*ny;
    mesh.elementNodesArray = new int[mesh.nElements_global*mesh.nNodes_element];
    mesh.elementMaterialTypes = new int[mesh.nElements_global];
    //memset(mesh.elementMaterialTypes,DEFAULT_ELEMENT_MATERIAL,mesh.nElements_global*sizeof(int));
    mesh.elementIJK = new int[mesh.nElements_global*3];

    int eN=0;
    for(int i=0;i<nz-px;i++)
      for(int j=0;j<ny-py;j++)
        for(int k=0;k<nx-pz;k++)
          {
	    mesh.elementIJK[eN*3+0] = i;
	    mesh.elementIJK[eN*3+1] = j;
	    mesh.elementIJK[eN*3+2] = k;

	    int sN = 0;
	    for(int ii=0;ii<px+1;ii++)
	      for(int jj=0;jj<py+1;jj++)
		for(int kk=0;kk<pz+1;kk++)
		  {          	
		    mesh.elementNodesArray[eN*mesh.nNodes_element+sN] = (k+kk) + (j+jj)*nx + (i+ii)*nxy;                    
                    sN++;
		  }    
	    eN++;          	

          } 
                      
    mesh.U_KNOT = new double[nx+px+1];  
    mesh.V_KNOT = new double[ny+py+1];    
    mesh.W_KNOT = new double[nz+pz+1];

    for(int i=0;i<px+1;i++)
      mesh.U_KNOT[i] = 0.0;
    for(int i=px+1;i<nx;i++)
      mesh.U_KNOT[i] = double(i-px-1);       
    for(int i=nx;i<nx+px+1;i++)
      mesh.U_KNOT[i] = double(nx);

    for(int i=0;i<py+1;i++)
      mesh.V_KNOT[i] = 0.0;
    for(int i=py+1;i<ny;i++)
      mesh.V_KNOT[i] = double(i-py-1);       
    for(int i=ny;i<ny+py+1;i++)
      mesh.V_KNOT[i] = double(ny);
       
    for(int i=0;i<pz+1;i++)
      mesh.W_KNOT[i] = 0.0;
    for(int i=pz+1;i<pz;i++)
      mesh.W_KNOT[i] = double(i-pz-1);       
    for(int i=nz;i<nz+pz+1;i++)
      mesh.W_KNOT[i] = double(nz);           


    mesh.weights = new double[mesh.nNodes_global];

    for(int i=0;i<mesh.nNodes_global;i++)
      mesh.weights[i] = 1.0;
    //std::cout<<"NURBS MESH BUILD"<<std::endl;             
    return 0;
  }

  int constructElementBoundaryElementsArray_edge(Mesh& mesh)
  {
    mesh.nNodes_elementBoundary = 1;
    mesh.nElementBoundaries_element = 2;
    using namespace std;
    double start,stop;
    logEvent("Constructing element boundary map",6);
    map<NodeTuple<1>,
      ElementNeighbors> elementBoundaryElements;
    start=CurrentTime();
    for(int eN=0;eN<mesh.nElements_global;eN++)
      for(int ebN=0;ebN<2;ebN++)
        {
          int nodes[1];
          nodes[0] = mesh.elementNodesArray[eN*2+(ebN+1)%2];
          NodeTuple<1> ebt(nodes);
          if(elementBoundaryElements.find(ebt) != elementBoundaryElements.end())
            {
              elementBoundaryElements[ebt].right=eN;
              elementBoundaryElements[ebt].right_ebN_element=ebN;
            }
          else
            {
              elementBoundaryElements.insert(elementBoundaryElements.end(),make_pair(ebt,ElementNeighbors(eN,ebN)));
            }
        }
    stop = CurrentTime();
    //cout<<"Elapsed time for building element boundary elements map= "<<(stop-start)<<"s"<<endl;
    mesh.nElementBoundaries_global = elementBoundaryElements.size();
    //cout<<"nElementBoundaries_global = "<<mesh.nElementBoundaries_global<<endl;

    //cout<<"Allocating Arrays"<<endl;
    start = CurrentTime();
    set<int> interiorElementBoundaries,exteriorElementBoundaries;
    mesh.elementBoundaryNodesArray =  new int[mesh.nElementBoundaries_global*mesh.nNodes_elementBoundary];
    mesh.elementBoundaryElementsArray = new int[mesh.nElementBoundaries_global*2];
    mesh.elementBoundaryLocalElementBoundariesArray = new int[mesh.nElementBoundaries_global*2];
    mesh.elementNeighborsArray = new int[mesh.nElements_global*mesh.nElementBoundaries_element];
    //mwf added
    mesh.elementBoundariesArray= new int[mesh.nElements_global*mesh.nElementBoundaries_element];
    stop = CurrentTime();
    //cout<<"Elapsed time for allocating arrays = "<<(stop-start)<<"s"<<endl;

    //cout<<"Populating Arrays"<<endl;
    start = CurrentTime();
    int ebN=0;
    for(map<NodeTuple<1>,ElementNeighbors>::iterator eb=elementBoundaryElements.begin();
        eb != elementBoundaryElements.end();
        eb++,ebN++)
      {
        mesh.elementBoundaryNodesArray[ebN] = eb->first.nodes[0];
        mesh.elementBoundaryElementsArray[ebN*2 + 0] = eb->second.left;
        mesh.elementBoundaryLocalElementBoundariesArray[ebN*2 + 0] = eb->second.left_ebN_element;
        mesh.elementBoundaryElementsArray[ebN*2 + 1] = eb->second.right;
        mesh.elementBoundaryLocalElementBoundariesArray[ebN*2 + 1] = eb->second.right_ebN_element;
        mesh.elementNeighborsArray[eb->second.left*mesh.nElementBoundaries_element + eb->second.left_ebN_element] = eb->second.right; 
        if(eb->second.right != -1)
          {
            interiorElementBoundaries.insert(ebN);
            mesh.elementNeighborsArray[eb->second.right*mesh.nElementBoundaries_element + eb->second.right_ebN_element] = eb->second.left;
          }
        else
          exteriorElementBoundaries.insert(ebN);
	//mwf added
	mesh.elementBoundariesArray[eb->second.left*mesh.nElementBoundaries_element + eb->second.left_ebN_element] = ebN;
	if (eb->second.right != -1)
	  {
	    mesh.elementBoundariesArray[eb->second.right*mesh.nElementBoundaries_element + eb->second.right_ebN_element] = ebN;
 	  }
      }
    mesh.nInteriorElementBoundaries_global = interiorElementBoundaries.size();
    mesh.interiorElementBoundariesArray = new int[mesh.nInteriorElementBoundaries_global];
    mesh.nExteriorElementBoundaries_global = exteriorElementBoundaries.size();
    mesh.exteriorElementBoundariesArray = new int[mesh.nExteriorElementBoundaries_global];
    int ebNI=0,ebNE=0;
    for (set<int>::iterator ebN=interiorElementBoundaries.begin();ebN != interiorElementBoundaries.end(); ebN++,ebNI++)
      mesh.interiorElementBoundariesArray[ebNI] = *ebN;
    for (set<int>::iterator ebN=exteriorElementBoundaries.begin();ebN != exteriorElementBoundaries.end(); ebN++,ebNE++)
      mesh.exteriorElementBoundariesArray[ebNE] = *ebN;
    //edges are elements in 1D so we just copy over but could be slicker
    mesh.nEdges_global = mesh.nElements_global;
    mesh.edgeNodesArray = new int[mesh.nElements_global*2];
    for (int eN=0;eN<mesh.nElements_global;eN++)
      {
        mesh.edgeNodesArray[eN*2+0] = mesh.elementNodesArray[eN*2+0];
        mesh.edgeNodesArray[eN*2+1] = mesh.elementNodesArray[eN*2+1];
      }
    vector<set<int> > nodeStar(mesh.nNodes_global);
    for (int eN=0;eN<mesh.nElements_global;eN++)
      {
        nodeStar[mesh.edgeNodesArray[eN*2+0]].insert(mesh.edgeNodesArray[eN*2+1]);
        nodeStar[mesh.edgeNodesArray[eN*2+1]].insert(mesh.edgeNodesArray[eN*2+0]);
      }
    mesh.nodeStarOffsets = new int[mesh.nNodes_global+1];
    mesh.nodeStarOffsets[0] = 0;
    for (int nN=1;nN<mesh.nNodes_global+1;nN++)
      mesh.nodeStarOffsets[nN] = mesh.nodeStarOffsets[nN-1]+nodeStar[nN-1].size();
    mesh.nodeStarArray = new int[mesh.nodeStarOffsets[mesh.nNodes_global]];
    for (int nN=0,offset=0;nN<mesh.nNodes_global;nN++)
      for (set<int>::iterator nN_star=nodeStar[nN].begin();nN_star!=nodeStar[nN].end();nN_star++,offset++)
        mesh.nodeStarArray[offset] = *nN_star;
    mesh.max_nNodeNeighbors_node=0;
    for (int nN=0;nN<mesh.nNodes_global;nN++)
      mesh.max_nNodeNeighbors_node=max(mesh.max_nNodeNeighbors_node,mesh.nodeStarOffsets[nN+1]-mesh.nodeStarOffsets[nN]);
    stop = CurrentTime();
    //repeat for node-->elements arrays
    vector<set<int> > nodeElementsStar(mesh.nNodes_global);
    for (int eN = 0; eN < mesh.nElements_global; eN++)
      {
	for (int nN = 0; nN < mesh.nNodes_element; nN++)
	  nodeElementsStar[mesh.elementNodesArray[eN*mesh.nNodes_element+nN]].insert(eN);
      }
    mesh.nodeElementOffsets = new int[mesh.nNodes_global+1];
    mesh.nodeElementOffsets[0] = 0;
    for (int nN = 0; nN < mesh.nNodes_global; nN++)
      mesh.nodeElementOffsets[nN+1] = mesh.nodeElementOffsets[nN]+nodeElementsStar[nN].size();
    mesh.nodeElementsArray  = new int[mesh.nodeElementOffsets[mesh.nNodes_global]];
    for (int nN=0,offset=0; nN < mesh.nNodes_global; nN++)
      {
	for (set<int>::iterator eN_star = nodeElementsStar[nN].begin(); eN_star != nodeElementsStar[nN].end();
	     eN_star++,offset++)
	  {
	    mesh.nodeElementsArray[offset] = *eN_star;
	  }
      }
    //end node-->elements construction
    mesh.elementBoundaryMaterialTypes = new int[mesh.nElementBoundaries_global];
    //if nodeMaterial is DEFAULT, go ahead and set to interior or exterior
    //depending on which boundary node belongs to. 
    //If node on at least one exterior boundary then it's exterior
    for (int ebNE = 0; ebNE < mesh.nExteriorElementBoundaries_global; ebNE++)
      {
	int ebN = mesh.exteriorElementBoundariesArray[ebNE];
	mesh.elementBoundaryMaterialTypes[ebN] = EXTERIOR_ELEMENT_BOUNDARY_MATERIAL;
	for (int nN_local = 0; nN_local < mesh.nNodes_elementBoundary; nN_local++)
	  {
	    int nN = mesh.elementBoundaryNodesArray[ebN*mesh.nNodes_elementBoundary+nN_local];
	    if (mesh.nodeMaterialTypes[nN] == DEFAULT_NODE_MATERIAL)
	      mesh.nodeMaterialTypes[nN] = EXTERIOR_NODE_MATERIAL;
	  }
      }
    for (int ebNI = 0; ebNI < mesh.nInteriorElementBoundaries_global; ebNI++)
      {
	int ebN = mesh.interiorElementBoundariesArray[ebNI];
	mesh.elementBoundaryMaterialTypes[ebN] = INTERIOR_ELEMENT_BOUNDARY_MATERIAL;
	for (int nN_local = 0; nN_local < mesh.nNodes_elementBoundary; nN_local++)
	  {
	    int nN = mesh.elementBoundaryNodesArray[ebN*mesh.nNodes_elementBoundary+nN_local];
	    if (mesh.nodeMaterialTypes[nN] == DEFAULT_NODE_MATERIAL)
	      mesh.nodeMaterialTypes[nN] = INTERIOR_NODE_MATERIAL;
	  }
      }


    //cout<<"Elapsed time for populating arrays = "<<(stop-start)<<"s"<<endl;
    return 0;
  }

  int constructElementBoundaryElementsArray_triangle(Mesh& mesh)
  {
    using namespace std;
    mesh.nNodes_elementBoundary = 2;
    mesh.nElementBoundaries_element = 3;
    double start,stop;
    //cout<<"Constructing element boundary map"<<endl;
    map<NodeTuple<2>,
      ElementNeighbors> elementBoundaryElements;
    start=CurrentTime();
    for(int eN=0;eN<mesh.nElements_global;eN++)
      for(int ebN=0;ebN<3;ebN++)
        {
          int nodes[2];
          nodes[0] = mesh.elementNodesArray[eN*3+(ebN+1)%3];
          nodes[1] = mesh.elementNodesArray[eN*3+(ebN+2)%3];
          NodeTuple<2> ebt(nodes);
          if(elementBoundaryElements.find(ebt) != elementBoundaryElements.end())
            {
              elementBoundaryElements[ebt].right=eN;
              elementBoundaryElements[ebt].right_ebN_element=ebN;
            }
          else
            {
              elementBoundaryElements.insert(elementBoundaryElements.end(),make_pair(ebt,ElementNeighbors(eN,ebN)));
            }
        }
    stop = CurrentTime();
    //cout<<"Elapsed time for building element boundary elements map= "<<(stop-start)<<"s"<<endl;
    mesh.nElementBoundaries_global = elementBoundaryElements.size();
    //cout<<"nElementBoundaries_global = "<<mesh.nElementBoundaries_global<<endl;
    
    //cout<<"Allocating Arrays"<<endl;
    start = CurrentTime();
    set<int> interiorElementBoundaries,exteriorElementBoundaries;
    mesh.elementBoundaryNodesArray =  new int[mesh.nElementBoundaries_global*mesh.nNodes_elementBoundary];
    mesh.elementBoundaryElementsArray = new int[mesh.nElementBoundaries_global*2];
    mesh.elementBoundaryLocalElementBoundariesArray = new int[mesh.nElementBoundaries_global*2];
    mesh.elementNeighborsArray = new int[mesh.nElements_global*mesh.nElementBoundaries_element];
    mesh.elementBoundariesArray= new int[mesh.nElements_global*mesh.nElementBoundaries_element];
    stop = CurrentTime();
    //cout<<"Elapsed time for allocating arrays = "<<(stop-start)<<"s"<<endl;

    //cout<<"Populating arrays"<<endl;
    start = CurrentTime();
    int ebN=0;
    for(map<NodeTuple<2>,ElementNeighbors>::iterator eb=elementBoundaryElements.begin();
        eb != elementBoundaryElements.end();
        eb++,ebN++)
      {
        mesh.elementBoundaryNodesArray[ebN*2 + 0] = eb->first.nodes[0];
        mesh.elementBoundaryNodesArray[ebN*2 + 1] = eb->first.nodes[1];
        mesh.elementBoundaryElementsArray[ebN*2 + 0] = eb->second.left;
        mesh.elementBoundaryLocalElementBoundariesArray[ebN*2 + 0] = eb->second.left_ebN_element;
        mesh.elementBoundaryElementsArray[ebN*2 + 1] = eb->second.right;
        mesh.elementBoundaryLocalElementBoundariesArray[ebN*2 + 1] = eb->second.right_ebN_element;
        mesh.elementNeighborsArray[eb->second.left*mesh.nElementBoundaries_element + eb->second.left_ebN_element] = eb->second.right; 
        if(eb->second.right != -1)
          {
            interiorElementBoundaries.insert(ebN);
            mesh.elementNeighborsArray[eb->second.right*mesh.nElementBoundaries_element + eb->second.right_ebN_element] = eb->second.left;
          }
        else
          exteriorElementBoundaries.insert(ebN);          
	mesh.elementBoundariesArray[eb->second.left*mesh.nElementBoundaries_element + eb->second.left_ebN_element] = ebN;
	if (eb->second.right != -1)
	  {
	    mesh.elementBoundariesArray[eb->second.right*mesh.nElementBoundaries_element + eb->second.right_ebN_element] = ebN;
	  }
      }
    mesh.nInteriorElementBoundaries_global = interiorElementBoundaries.size();
    mesh.interiorElementBoundariesArray = new int[mesh.nInteriorElementBoundaries_global];
    mesh.nExteriorElementBoundaries_global = exteriorElementBoundaries.size();
    mesh.exteriorElementBoundariesArray = new int[mesh.nExteriorElementBoundaries_global];
    int ebNI=0,ebNE=0;
    for (set<int>::iterator ebN=interiorElementBoundaries.begin();ebN != interiorElementBoundaries.end(); ebN++,ebNI++)
      mesh.interiorElementBoundariesArray[ebNI] = *ebN;
    for (set<int>::iterator ebN=exteriorElementBoundaries.begin();ebN != exteriorElementBoundaries.end(); ebN++,ebNE++)
      mesh.exteriorElementBoundariesArray[ebNE] = *ebN;
    set<NodeTuple<2> > edges;
    for (int eN=0;eN<mesh.nElements_global;eN++)
      {
        int nodes[2];
        for (int nN_L=0;nN_L<mesh.nNodes_element;nN_L++)
          for (int nN_R=nN_L+1;nN_R<mesh.nNodes_element;nN_R++)
            {
              nodes[0] = mesh.elementNodesArray[eN*3+nN_L];
              nodes[1] = mesh.elementNodesArray[eN*3+nN_R];
              edges.insert(NodeTuple<2>(nodes));
            }
      }
    mesh.nEdges_global = edges.size();
    mesh.edgeNodesArray = new int[mesh.nEdges_global*2];
    int edgeN=0;
    for (set<NodeTuple<2> >::iterator edgeTuple_p=edges.begin();edgeTuple_p != edges.end();edgeTuple_p++,edgeN++)
      {
        mesh.edgeNodesArray[edgeN*2+0] = edgeTuple_p->nodes[0];
        mesh.edgeNodesArray[edgeN*2+1] = edgeTuple_p->nodes[1];
      }
    vector<set<int> > nodeStar(mesh.nNodes_global);
    for (int edgeN=0;edgeN<mesh.nEdges_global;edgeN++)
      {
        nodeStar[mesh.edgeNodesArray[edgeN*2+0]].insert(mesh.edgeNodesArray[edgeN*2+1]);
        nodeStar[mesh.edgeNodesArray[edgeN*2+1]].insert(mesh.edgeNodesArray[edgeN*2+0]);
      }
    mesh.nodeStarOffsets = new int[mesh.nNodes_global+1];
    mesh.nodeStarOffsets[0] = 0;
    for (int nN=1;nN<mesh.nNodes_global+1;nN++)
      mesh.nodeStarOffsets[nN] = mesh.nodeStarOffsets[nN-1]+nodeStar[nN-1].size();
    mesh.nodeStarArray = new int[mesh.nodeStarOffsets[mesh.nNodes_global]];
    for (int nN=0,offset=0;nN<mesh.nNodes_global;nN++)
      for (set<int>::iterator nN_star=nodeStar[nN].begin();nN_star!=nodeStar[nN].end();nN_star++,offset++)
        mesh.nodeStarArray[offset] = *nN_star;
    stop = CurrentTime();
    mesh.max_nNodeNeighbors_node=0;
    for (int nN=0;nN<mesh.nNodes_global;nN++)
      mesh.max_nNodeNeighbors_node=max(mesh.max_nNodeNeighbors_node,mesh.nodeStarOffsets[nN+1]-mesh.nodeStarOffsets[nN]);
    //repeat for node-->elements arrays
    vector<set<int> > nodeElementsStar(mesh.nNodes_global);
    for (int eN = 0; eN < mesh.nElements_global; eN++)
      {
	for (int nN = 0; nN < mesh.nNodes_element; nN++)
	  nodeElementsStar[mesh.elementNodesArray[eN*mesh.nNodes_element+nN]].insert(eN);
      }
    mesh.nodeElementOffsets = new int[mesh.nNodes_global+1];
    mesh.nodeElementOffsets[0] = 0;
    for (int nN = 0; nN < mesh.nNodes_global; nN++)
      mesh.nodeElementOffsets[nN+1] = mesh.nodeElementOffsets[nN]+nodeElementsStar[nN].size();
    mesh.nodeElementsArray  = new int[mesh.nodeElementOffsets[mesh.nNodes_global]];
    for (int nN=0,offset=0; nN < mesh.nNodes_global; nN++)
      {
	for (set<int>::iterator eN_star = nodeElementsStar[nN].begin(); eN_star != nodeElementsStar[nN].end();
	     eN_star++,offset++)
	  {
	    mesh.nodeElementsArray[offset] = *eN_star;
	  }
      }
    //end node-->elements construction
    mesh.elementBoundaryMaterialTypes = new int[mesh.nElementBoundaries_global];
    //if nodeMaterial is DEFAULT, go ahead and set to interior or exterior
    //depending on which boundary node belongs to. 
    //If node on at least one exterior boundary then it's exterior
    //set actual values elsewhere
    for (int ebNE = 0; ebNE < mesh.nExteriorElementBoundaries_global; ebNE++)
      {
	int ebN = mesh.exteriorElementBoundariesArray[ebNE];
	mesh.elementBoundaryMaterialTypes[ebN] = EXTERIOR_ELEMENT_BOUNDARY_MATERIAL;
	for (int nN_local = 0; nN_local < mesh.nNodes_elementBoundary; nN_local++)
	  {
	    int nN = mesh.elementBoundaryNodesArray[ebN*mesh.nNodes_elementBoundary+nN_local];
	    if (mesh.nodeMaterialTypes[nN] == DEFAULT_NODE_MATERIAL)
	      mesh.nodeMaterialTypes[nN] = EXTERIOR_NODE_MATERIAL;
	  }
      }
    for (int ebNI = 0; ebNI < mesh.nInteriorElementBoundaries_global; ebNI++)
      {
	int ebN = mesh.interiorElementBoundariesArray[ebNI];
	mesh.elementBoundaryMaterialTypes[ebN] = INTERIOR_ELEMENT_BOUNDARY_MATERIAL;
	for (int nN_local = 0; nN_local < mesh.nNodes_elementBoundary; nN_local++)
	  {
	    int nN = mesh.elementBoundaryNodesArray[ebN*mesh.nNodes_elementBoundary+nN_local];
	    if (mesh.nodeMaterialTypes[nN] == DEFAULT_NODE_MATERIAL)
	      mesh.nodeMaterialTypes[nN] = INTERIOR_NODE_MATERIAL;
	  }
      }
    //cout<<"Elapsed time for populating arrays = "<<(stop-start)<<"s"<<endl;
    return 0;
  }

  int constructElementBoundaryElementsArray_quadrilateral(Mesh& mesh)
  {
    using namespace std;
    mesh.nNodes_elementBoundary = 2;
    mesh.nElementBoundaries_element = 4;
    double start,stop;
    //cout<<"Constructing element boundary map"<<endl;
    map<NodeTuple<2>,
      ElementNeighbors> elementBoundaryElements;
    start=CurrentTime();
    for(int eN=0;eN<mesh.nElements_global;eN++)
      for(int ebN=0;ebN<4;ebN++)
        {
          int nodes[2];
          nodes[0] = mesh.elementNodesArray[eN*4+ebN];
          nodes[1] = mesh.elementNodesArray[eN*4+(ebN+1)%4];
          NodeTuple<2> ebt(nodes);
          if(elementBoundaryElements.find(ebt) != elementBoundaryElements.end())
            {
              elementBoundaryElements[ebt].right=eN;
              elementBoundaryElements[ebt].right_ebN_element=ebN;
            }
          else
            {
              elementBoundaryElements.insert(elementBoundaryElements.end(),make_pair(ebt,ElementNeighbors(eN,ebN)));
            }
        }
    stop = CurrentTime();
    //cout<<"Elapsed time for building element boundary elements map= "<<(stop-start)<<"s"<<endl;
    mesh.nElementBoundaries_global = elementBoundaryElements.size();
    cout<<"nElementBoundaries_global = "<<mesh.nElementBoundaries_global<<endl;
    
    //cout<<"Allocating Arrays"<<endl;
    start = CurrentTime();
    set<int> interiorElementBoundaries,exteriorElementBoundaries;
    mesh.elementBoundaryNodesArray =  new int[mesh.nElementBoundaries_global*mesh.nNodes_elementBoundary];
    mesh.elementBoundaryElementsArray = new int[mesh.nElementBoundaries_global*2];
    mesh.elementBoundaryLocalElementBoundariesArray = new int[mesh.nElementBoundaries_global*2];
    mesh.elementNeighborsArray = new int[mesh.nElements_global*mesh.nElementBoundaries_element];
    mesh.elementBoundariesArray= new int[mesh.nElements_global*mesh.nElementBoundaries_element];
    stop = CurrentTime();
    //cout<<"Elapsed time for allocating arrays = "<<(stop-start)<<"s"<<endl;

    //cout<<"Populating arrays"<<endl;
    start = CurrentTime();
    int ebN=0;
    for(map<NodeTuple<2>,ElementNeighbors>::iterator eb=elementBoundaryElements.begin();
        eb != elementBoundaryElements.end();
        eb++,ebN++)
      {
        mesh.elementBoundaryNodesArray[ebN*2 + 0] = eb->first.nodes[0];
        mesh.elementBoundaryNodesArray[ebN*2 + 1] = eb->first.nodes[1];
        mesh.elementBoundaryElementsArray[ebN*2 + 0] = eb->second.left;
        mesh.elementBoundaryLocalElementBoundariesArray[ebN*2 + 0] = eb->second.left_ebN_element;
        mesh.elementBoundaryElementsArray[ebN*2 + 1] = eb->second.right;
        mesh.elementBoundaryLocalElementBoundariesArray[ebN*2 + 1] = eb->second.right_ebN_element;
        mesh.elementNeighborsArray[eb->second.left*mesh.nElementBoundaries_element + eb->second.left_ebN_element] = eb->second.right; 
        if(eb->second.right != -1)
          {
            interiorElementBoundaries.insert(ebN);
            mesh.elementNeighborsArray[eb->second.right*mesh.nElementBoundaries_element + eb->second.right_ebN_element] = eb->second.left;
          }
        else
          exteriorElementBoundaries.insert(ebN);          
	mesh.elementBoundariesArray[eb->second.left*mesh.nElementBoundaries_element + eb->second.left_ebN_element] = ebN;
	if (eb->second.right != -1)
	  {
	    mesh.elementBoundariesArray[eb->second.right*mesh.nElementBoundaries_element + eb->second.right_ebN_element] = ebN;
	  }
      }
    mesh.nInteriorElementBoundaries_global = interiorElementBoundaries.size();
    mesh.interiorElementBoundariesArray = new int[mesh.nInteriorElementBoundaries_global];
    mesh.nExteriorElementBoundaries_global = exteriorElementBoundaries.size();
    mesh.exteriorElementBoundariesArray = new int[mesh.nExteriorElementBoundaries_global];
    int ebNI=0,ebNE=0;
    for (set<int>::iterator ebN=interiorElementBoundaries.begin();ebN != interiorElementBoundaries.end(); ebN++,ebNI++)
      mesh.interiorElementBoundariesArray[ebNI] = *ebN;
    for (set<int>::iterator ebN=exteriorElementBoundaries.begin();ebN != exteriorElementBoundaries.end(); ebN++,ebNE++)
      mesh.exteriorElementBoundariesArray[ebNE] = *ebN;
    set<NodeTuple<2> > edges;
    for (int eN=0;eN<mesh.nElements_global;eN++)
      {
        int nodes[2];
        for (int nN_L=0;nN_L<mesh.nNodes_element;nN_L++)
          for (int nN_R=nN_L+1;nN_R<mesh.nNodes_element;nN_R++)
            {
              nodes[0] = mesh.elementNodesArray[eN*4+nN_L];
              nodes[1] = mesh.elementNodesArray[eN*4+nN_R];
              edges.insert(NodeTuple<2>(nodes));
            }
      }
    mesh.nEdges_global = edges.size();
    mesh.edgeNodesArray = new int[mesh.nEdges_global*2];
    int edgeN=0;
    for (set<NodeTuple<2> >::iterator edgeTuple_p=edges.begin();edgeTuple_p != edges.end();edgeTuple_p++,edgeN++)
      {
        mesh.edgeNodesArray[edgeN*2+0] = edgeTuple_p->nodes[0];
        mesh.edgeNodesArray[edgeN*2+1] = edgeTuple_p->nodes[1];
      }
    vector<set<int> > nodeStar(mesh.nNodes_global);
    for (int edgeN=0;edgeN<mesh.nEdges_global;edgeN++)
      {
        nodeStar[mesh.edgeNodesArray[edgeN*2+0]].insert(mesh.edgeNodesArray[edgeN*2+1]);
        nodeStar[mesh.edgeNodesArray[edgeN*2+1]].insert(mesh.edgeNodesArray[edgeN*2+0]);
      }
    mesh.nodeStarOffsets = new int[mesh.nNodes_global+1];
    mesh.nodeStarOffsets[0] = 0;
    for (int nN=1;nN<mesh.nNodes_global+1;nN++)
      mesh.nodeStarOffsets[nN] = mesh.nodeStarOffsets[nN-1]+nodeStar[nN-1].size();
    mesh.nodeStarArray = new int[mesh.nodeStarOffsets[mesh.nNodes_global]];
    for (int nN=0,offset=0;nN<mesh.nNodes_global;nN++)
      for (set<int>::iterator nN_star=nodeStar[nN].begin();nN_star!=nodeStar[nN].end();nN_star++,offset++)
        mesh.nodeStarArray[offset] = *nN_star;
    stop = CurrentTime();
    mesh.max_nNodeNeighbors_node=0;
    for (int nN=0;nN<mesh.nNodes_global;nN++)
      mesh.max_nNodeNeighbors_node=max(mesh.max_nNodeNeighbors_node,mesh.nodeStarOffsets[nN+1]-mesh.nodeStarOffsets[nN]);
    //repeat for node-->elements arrays
    vector<set<int> > nodeElementsStar(mesh.nNodes_global);
    for (int eN = 0; eN < mesh.nElements_global; eN++)
      {
	for (int nN = 0; nN < mesh.nNodes_element; nN++)
	  nodeElementsStar[mesh.elementNodesArray[eN*mesh.nNodes_element+nN]].insert(eN);
      }
    mesh.nodeElementOffsets = new int[mesh.nNodes_global+1];
    mesh.nodeElementOffsets[0] = 0;
    for (int nN = 0; nN < mesh.nNodes_global; nN++)
      mesh.nodeElementOffsets[nN+1] = mesh.nodeElementOffsets[nN]+nodeElementsStar[nN].size();
    mesh.nodeElementsArray  = new int[mesh.nodeElementOffsets[mesh.nNodes_global]];
    for (int nN=0,offset=0; nN < mesh.nNodes_global; nN++)
      {
	for (set<int>::iterator eN_star = nodeElementsStar[nN].begin(); eN_star != nodeElementsStar[nN].end();
	     eN_star++,offset++)
	  {
	    mesh.nodeElementsArray[offset] = *eN_star;
	  }
      }
    //end node-->elements construction
    mesh.elementBoundaryMaterialTypes = new int[mesh.nElementBoundaries_global];
    //if nodeMaterial is DEFAULT, go ahead and set to interior or exterior
    //depending on which boundary node belongs to. 
    //If node on at least one exterior boundary then it's exterior
    //set actual values elsewhere
    for (int ebNE = 0; ebNE < mesh.nExteriorElementBoundaries_global; ebNE++)
      {
	int ebN = mesh.exteriorElementBoundariesArray[ebNE];
	mesh.elementBoundaryMaterialTypes[ebN] = EXTERIOR_ELEMENT_BOUNDARY_MATERIAL;
	for (int nN_local = 0; nN_local < mesh.nNodes_elementBoundary; nN_local++)
	  {
	    int nN = mesh.elementBoundaryNodesArray[ebN*mesh.nNodes_elementBoundary+nN_local];
	    if (mesh.nodeMaterialTypes[nN] == DEFAULT_NODE_MATERIAL)
	      mesh.nodeMaterialTypes[nN] = EXTERIOR_NODE_MATERIAL;
	  }
      }
    for (int ebNI = 0; ebNI < mesh.nInteriorElementBoundaries_global; ebNI++)
      {
	int ebN = mesh.interiorElementBoundariesArray[ebNI];
	mesh.elementBoundaryMaterialTypes[ebN] = INTERIOR_ELEMENT_BOUNDARY_MATERIAL;
	for (int nN_local = 0; nN_local < mesh.nNodes_elementBoundary; nN_local++)
	  {
	    int nN = mesh.elementBoundaryNodesArray[ebN*mesh.nNodes_elementBoundary+nN_local];
	    if (mesh.nodeMaterialTypes[nN] == DEFAULT_NODE_MATERIAL)
	      mesh.nodeMaterialTypes[nN] = INTERIOR_NODE_MATERIAL;
	  }
      }
    //cout<<"Elapsed time for populating arrays = "<<(stop-start)<<"s"<<endl;
    return 0;
  }

  int constructElementBoundaryElementsArray_tetrahedron(Mesh& mesh)
  {
    mesh.nNodes_elementBoundary = 3;
    mesh.nElementBoundaries_element = 4;
    
    using namespace std;
    double start,stop;
    typedef map<NodeTuple<3>, ElementNeighbors> ElementBoundaryElementsType;
    {
    ElementBoundaryElementsType elementBoundaryElements;
    start=CurrentTime();
    //std::cout<<"Extracting boundary elements"<<endl;
    for(int eN=0;eN<mesh.nElements_global;eN++)
      for(int ebN=0;ebN<mesh.nElementBoundaries_element;ebN++)
        {
          int nodes[3];
          nodes[0] = mesh.elementNodesArray[eN*4+((ebN+1)%4)];
          nodes[1] = mesh.elementNodesArray[eN*4+((ebN+2)%4)];
          nodes[2] = mesh.elementNodesArray[eN*4+((ebN+3)%4)];
          NodeTuple<3> ebt(nodes);
          if(elementBoundaryElements.find(ebt) != elementBoundaryElements.end())
            {
              elementBoundaryElements[ebt].right=eN;
              elementBoundaryElements[ebt].right_ebN_element=ebN;
            }
          else
            {
              elementBoundaryElements.insert(elementBoundaryElements.end(),make_pair(ebt,ElementNeighbors(eN,ebN)));
            }
        }
    stop = CurrentTime();
    //std::cout<<"Elapsed time for building element boundary elements map= "<<(stop-start)<<"s"<<endl;
    mesh.nElementBoundaries_global = elementBoundaryElements.size();
    //std::cout<<"nElementBoundaries_global = "<<mesh.nElementBoundaries_global<<endl;

    //std::cout<<"Allocating Arrays"<<endl;
    start = CurrentTime();
    typedef set<int> ElementBoundariesType;
    ElementBoundariesType interiorElementBoundaries,exteriorElementBoundaries;
    mesh.elementBoundaryNodesArray =  new int[mesh.nElementBoundaries_global*mesh.nNodes_elementBoundary];
    mesh.elementBoundaryElementsArray = new int[mesh.nElementBoundaries_global*2];
    mesh.elementBoundaryLocalElementBoundariesArray = new int[mesh.nElementBoundaries_global*2];
    mesh.elementNeighborsArray = new int[mesh.nElements_global*mesh.nElementBoundaries_element];
    mesh.elementBoundariesArray= new int[mesh.nElements_global*mesh.nElementBoundaries_element];
    stop = CurrentTime();
    
    //std::cout<<"Elapsed time for allocating arrays = "<<(stop-start)<<"s"<<endl;
    //std::cout<<"Generating elementBoundaryElementsArray and elementBoundaryNodesArray"<<endl;
    start = CurrentTime();
    int ebN=0;
    for(map<NodeTuple<3>,ElementNeighbors>::iterator eb=elementBoundaryElements.begin();
        eb != elementBoundaryElements.end();
        eb++,ebN++)
      {
        mesh.elementBoundaryNodesArray[ebN*3 + 0] = eb->first.nodes[0];
        mesh.elementBoundaryNodesArray[ebN*3 + 1] = eb->first.nodes[1];
        mesh.elementBoundaryNodesArray[ebN*3 + 2] = eb->first.nodes[2];

        mesh.elementBoundaryElementsArray[ebN*2 + 0] = eb->second.left;
        mesh.elementBoundaryLocalElementBoundariesArray[ebN*2 + 0] = eb->second.left_ebN_element;
        mesh.elementBoundaryElementsArray[ebN*2 + 1] = eb->second.right;
        mesh.elementBoundaryLocalElementBoundariesArray[ebN*2 + 1] = eb->second.right_ebN_element;
        mesh.elementNeighborsArray[eb->second.left*mesh.nElementBoundaries_element + eb->second.left_ebN_element] = eb->second.right; 
        if(eb->second.right != -1)
	  {
	    interiorElementBoundaries.insert(ebN);
	    mesh.elementNeighborsArray[eb->second.right*mesh.nElementBoundaries_element + eb->second.right_ebN_element] = eb->second.left; 
	  }
        else
          exteriorElementBoundaries.insert(ebN);
	mesh.elementBoundariesArray[eb->second.left*mesh.nElementBoundaries_element + eb->second.left_ebN_element] = ebN;
	if (eb->second.right != -1)
	  {
	    mesh.elementBoundariesArray[eb->second.right*mesh.nElementBoundaries_element + eb->second.right_ebN_element] = ebN;
 	  }
      }
    mesh.nInteriorElementBoundaries_global = interiorElementBoundaries.size();
    mesh.interiorElementBoundariesArray = new int[mesh.nInteriorElementBoundaries_global];
    mesh.nExteriorElementBoundaries_global = exteriorElementBoundaries.size();
    mesh.exteriorElementBoundariesArray = new int[mesh.nExteriorElementBoundaries_global];
    int ebNI=0,ebNE=0;
    for (set<int>::iterator ebN=interiorElementBoundaries.begin();ebN != interiorElementBoundaries.end(); ebN++,ebNI++)
      mesh.interiorElementBoundariesArray[ebNI] = *ebN;
    for (set<int>::iterator ebN=exteriorElementBoundaries.begin();ebN != exteriorElementBoundaries.end(); ebN++,ebNE++)
      mesh.exteriorElementBoundariesArray[ebNE] = *ebN;
    }
    //elementBoundaryElements.~ElementBoundaryElementsType();
    //interiorElementBoundaries.~ElementBoundariesType();
    //exteriorElementBoundaries.~ElementBoundariesType();
    {
      typedef set<NodeTuple<2> > EdgesType;
    EdgesType edges;
    //std::cout<<"extracting edges"<<std::endl;
    for (int eN=0;eN<mesh.nElements_global;eN++)
      {
        int nodes[2];
        for (int nN_L=0;nN_L<mesh.nNodes_element;nN_L++)
          for (int nN_R=nN_L+1;nN_R<mesh.nNodes_element;nN_R++)
            {
              nodes[0] = mesh.elementNodesArray[eN*4+nN_L];
              nodes[1] = mesh.elementNodesArray[eN*4+nN_R];
              edges.insert(NodeTuple<2>(nodes));
            }
      }
    mesh.nEdges_global = edges.size();
    mesh.edgeNodesArray = new int[mesh.nEdges_global*2];
    EdgesType::iterator edge_p=edges.begin();
    for (int edgeN=0;edgeN<int(edges.size());edgeN++)
      {
        mesh.edgeNodesArray[edgeN*2+0] = edge_p->nodes[0];
        mesh.edgeNodesArray[edgeN*2+1] = edge_p->nodes[1];
        edge_p++;
      }
    }//edges.~EdgesType();
    {
      //std::cout<<"node star"<<std::endl;
    typedef vector<set<int> > NodeStarType;
    NodeStarType nodeStar(mesh.nNodes_global);
    for (int edgeN=0;edgeN<mesh.nEdges_global;edgeN++)
      {
        nodeStar[mesh.edgeNodesArray[edgeN*2+0]].insert(mesh.edgeNodesArray[edgeN*2+1]);
        nodeStar[mesh.edgeNodesArray[edgeN*2+1]].insert(mesh.edgeNodesArray[edgeN*2+0]);
      }
    mesh.nodeStarOffsets = new int[mesh.nNodes_global+1];
    mesh.nodeStarOffsets[0] = 0;
    for (int nN=1;nN<mesh.nNodes_global+1;nN++)
      mesh.nodeStarOffsets[nN] = mesh.nodeStarOffsets[nN-1] + nodeStar[nN-1].size();
    mesh.nodeStarArray = new int[mesh.nodeStarOffsets[mesh.nNodes_global]];
    for (int nN=0,offset=0;nN<mesh.nNodes_global;nN++)
      for (set<int>::iterator nN_star=nodeStar[nN].begin();nN_star!=nodeStar[nN].end();nN_star++,offset++)
        mesh.nodeStarArray[offset] = *nN_star;
    }//nodeStar.~NodeStarType();
    {
      //std::cout<<"node elements star"<<std::endl;
    stop = CurrentTime();
    mesh.max_nNodeNeighbors_node=0;
    for (int nN=0;nN<mesh.nNodes_global;nN++)
      mesh.max_nNodeNeighbors_node=max(mesh.max_nNodeNeighbors_node,mesh.nodeStarOffsets[nN+1]-mesh.nodeStarOffsets[nN]);
    typedef vector<set<int> > NodeElementsStarType;
    NodeElementsStarType nodeElementsStar(mesh.nNodes_global);
    for (int eN = 0; eN < mesh.nElements_global; eN++)
      {
	for (int nN = 0; nN < mesh.nNodes_element; nN++)
	  nodeElementsStar[mesh.elementNodesArray[eN*mesh.nNodes_element+nN]].insert(eN);
      }
    mesh.nodeElementOffsets = new int[mesh.nNodes_global+1];
    mesh.nodeElementOffsets[0] = 0;
    for (int nN = 0; nN < mesh.nNodes_global; nN++)
      mesh.nodeElementOffsets[nN+1] = mesh.nodeElementOffsets[nN]+nodeElementsStar[nN].size();
    mesh.nodeElementsArray  = new int[mesh.nodeElementOffsets[mesh.nNodes_global]];
    for (int nN=0,offset=0; nN < mesh.nNodes_global; nN++)
      {
	for (set<int>::iterator eN_star = nodeElementsStar[nN].begin(); eN_star != nodeElementsStar[nN].end();
	     eN_star++,offset++)
	  {
	    mesh.nodeElementsArray[offset] = *eN_star;
	  }
      }
    }//nodeElementsStar.~NodeElementsStarType();
    mesh.elementBoundaryMaterialTypes = new int[mesh.nElementBoundaries_global];
    //std::cout<<"nodematerials"<<std::endl;
    //if nodeMaterial is DEFAULT, go ahead and set to interior or exterior
    //depending on which boundary node belongs to. 
    //If node on at least one exterior boundary then it's exterior
    for (int ebNE = 0; ebNE < mesh.nExteriorElementBoundaries_global; ebNE++)
      {
	int ebN = mesh.exteriorElementBoundariesArray[ebNE];
	mesh.elementBoundaryMaterialTypes[ebN] = EXTERIOR_ELEMENT_BOUNDARY_MATERIAL;
	for (int nN_local = 0; nN_local < mesh.nNodes_elementBoundary; nN_local++)
	  {
	    int nN = mesh.elementBoundaryNodesArray[ebN*mesh.nNodes_elementBoundary+nN_local];
	    if (mesh.nodeMaterialTypes[nN] == DEFAULT_NODE_MATERIAL)
	      mesh.nodeMaterialTypes[nN] = EXTERIOR_NODE_MATERIAL;
	  }
      }
    for (int ebNI = 0; ebNI < mesh.nInteriorElementBoundaries_global; ebNI++)
      {
	int ebN = mesh.interiorElementBoundariesArray[ebNI];
	mesh.elementBoundaryMaterialTypes[ebN] = INTERIOR_ELEMENT_BOUNDARY_MATERIAL;
	for (int nN_local = 0; nN_local < mesh.nNodes_elementBoundary; nN_local++)
	  {
	    int nN = mesh.elementBoundaryNodesArray[ebN*mesh.nNodes_elementBoundary+nN_local];
	    if (mesh.nodeMaterialTypes[nN] == DEFAULT_NODE_MATERIAL)
	      mesh.nodeMaterialTypes[nN] = INTERIOR_NODE_MATERIAL;
	  }
      }
    //std::cout<<"Elapsed time for populating arrays = "<<(stop-start)<<"s"<<endl;
    return 0;
  }

  int constructElementBoundaryElementsArray_hexahedron(Mesh& mesh)
  {
    mesh.nNodes_elementBoundary = 4;
    mesh.nElementBoundaries_element = 6;
    using namespace std;
    double start,stop;
    map<NodeTuple<4>,
      ElementNeighbors> elementBoundaryElements;
    start=CurrentTime();
    
    int lface[6][4] = {{0,1,2,3},
                       {0,1,5,4},
                       {1,2,6,5},
                       {2,3,7,6},
                       {3,0,4,7},
                       {4,5,6,7}};
    
    //cout<<"Extracting boundary elements"<<endl;
    for(int eN=0;eN<mesh.nElements_global;eN++)
      for(int ebN=0;ebN<mesh.nElementBoundaries_element;ebN++)
        {
          int nodes[4];
          nodes[0] = mesh.elementNodesArray[eN*8+lface[ebN][0]];
          nodes[1] = mesh.elementNodesArray[eN*8+lface[ebN][1]];
          nodes[2] = mesh.elementNodesArray[eN*8+lface[ebN][2]];      
          nodes[3] = mesh.elementNodesArray[eN*8+lface[ebN][3]];
                    
          NodeTuple<4> ebt(nodes);
          if(elementBoundaryElements.find(ebt) != elementBoundaryElements.end())
            {
              elementBoundaryElements[ebt].right=eN;
              elementBoundaryElements[ebt].right_ebN_element=ebN;
            }
          else
            {
              elementBoundaryElements.insert(elementBoundaryElements.end(),make_pair(ebt,ElementNeighbors(eN,ebN)));
            }
        }        
    stop = CurrentTime();
    //cout<<"Elapsed time for building element boundary elements map= "<<(stop-start)<<"s"<<endl;
    mesh.nElementBoundaries_global = elementBoundaryElements.size();
    //cout<<"nElementBoundaries_global = "<<mesh.nElementBoundaries_global<<endl;

    //cout<<"Allocating Arrays"<<endl;
    start = CurrentTime();
    set<int> interiorElementBoundaries,exteriorElementBoundaries;
    mesh.elementBoundaryNodesArray =  new int[mesh.nElementBoundaries_global*mesh.nNodes_elementBoundary];
    mesh.elementBoundaryElementsArray = new int[mesh.nElementBoundaries_global*2];
    mesh.elementBoundaryLocalElementBoundariesArray = new int[mesh.nElementBoundaries_global*2];
    mesh.elementNeighborsArray = new int[mesh.nElements_global*mesh.nElementBoundaries_element];
    //mwf added
    mesh.elementBoundariesArray= new int[mesh.nElements_global*mesh.nElementBoundaries_element];
    stop = CurrentTime();
    //cout<<"Elapsed time for allocating arrays = "<<(stop-start)<<"s"<<endl;

    //cout<<"Generating elementBoundaryElementsArray and elementBoundaryNodesArray"<<endl;
    start = CurrentTime();
    int ebN=0;
    for(map<NodeTuple<4>,ElementNeighbors>::iterator eb=elementBoundaryElements.begin();
        eb != elementBoundaryElements.end();
        eb++,ebN++)
      {
        mesh.elementBoundaryNodesArray[ebN*4 + 0] = eb->first.nodes_unsorted[0];
        mesh.elementBoundaryNodesArray[ebN*4 + 1] = eb->first.nodes_unsorted[1];
        mesh.elementBoundaryNodesArray[ebN*4 + 2] = eb->first.nodes_unsorted[2];
        mesh.elementBoundaryNodesArray[ebN*4 + 3] = eb->first.nodes_unsorted[3];

        mesh.elementBoundaryElementsArray[ebN*2 + 0] = eb->second.left;
        mesh.elementBoundaryLocalElementBoundariesArray[ebN*2 + 0] = eb->second.left_ebN_element;
        mesh.elementBoundaryElementsArray[ebN*2 + 1] = eb->second.right;
        mesh.elementBoundaryLocalElementBoundariesArray[ebN*2 + 1] = eb->second.right_ebN_element;
        mesh.elementNeighborsArray[eb->second.left*mesh.nElementBoundaries_element + eb->second.left_ebN_element] = eb->second.right; 
        if(eb->second.right != -1)
	  {
	    interiorElementBoundaries.insert(ebN);
	    mesh.elementNeighborsArray[eb->second.right*mesh.nElementBoundaries_element + eb->second.right_ebN_element] = eb->second.left; 
	  }
        else
          exteriorElementBoundaries.insert(ebN);    
          //mwf added
	    mesh.elementBoundariesArray[eb->second.left*mesh.nElementBoundaries_element + eb->second.left_ebN_element] = ebN;
	    if (eb->second.right != -1)
	    {
	       mesh.elementBoundariesArray[eb->second.right*mesh.nElementBoundaries_element + eb->second.right_ebN_element] = ebN;
 	    }
      }    
    mesh.nInteriorElementBoundaries_global = interiorElementBoundaries.size();
    mesh.interiorElementBoundariesArray  =  new int[mesh.nInteriorElementBoundaries_global];
    mesh.nExteriorElementBoundaries_global = exteriorElementBoundaries.size();
    mesh.exteriorElementBoundariesArray = new int[mesh.nExteriorElementBoundaries_global];
    int ebNI=0,ebNE=0;
    for (set<int>::iterator ebN=interiorElementBoundaries.begin();ebN != interiorElementBoundaries.end(); ebN++,ebNI++)
      mesh.interiorElementBoundariesArray[ebNI] = *ebN;
    for (set<int>::iterator ebN=exteriorElementBoundaries.begin();ebN != exteriorElementBoundaries.end(); ebN++,ebNE++)
      mesh.exteriorElementBoundariesArray[ebNE] = *ebN;
    set<NodeTuple<2> > edges;
     
    int ledge[12][2] = {{0,1},{1,2},{2,3},{3,0},
                        {0,4},{1,5},{2,6},{3,7},
                        {4,5},{5,6},{6,7},{7,4}};
    
    
    
    
    //std::cout<<"Extracting edges"<<std::endl;
    for (int eN=0;eN<mesh.nElements_global;eN++)
      {
	int nodes[2];
	for (int e=0;e<12;e++)
	  {
    	
	    nodes[0] = mesh.elementNodesArray[eN*8+ledge[e][0]];
	    nodes[1] = mesh.elementNodesArray[eN*8+ledge[e][1]];
                  
            edges.insert(NodeTuple<2>(nodes));
	  }   
      }
      
      
    mesh.nEdges_global = edges.size();
    mesh.edgeNodesArray = new int[mesh.nEdges_global*2];
    set<NodeTuple<2> >::iterator edge_p=edges.begin();
    for (int edgeN=0;edgeN<int(edges.size());edgeN++)
      {
        mesh.edgeNodesArray[edgeN*2+0] = edge_p->nodes[0];
        mesh.edgeNodesArray[edgeN*2+1] = edge_p->nodes[1];
        edge_p++;
      }

    //std::cout<<"Extracting nodeStar"<<std::endl;      
    vector<set<int> > nodeStar(mesh.nNodes_global);
    for (int edgeN=0;edgeN<mesh.nEdges_global;edgeN++)
      {
        nodeStar[mesh.edgeNodesArray[edgeN*2+0]].insert(mesh.edgeNodesArray[edgeN*2+1]);
        nodeStar[mesh.edgeNodesArray[edgeN*2+1]].insert(mesh.edgeNodesArray[edgeN*2+0]);
      }

    mesh.nodeStarOffsets = new int[mesh.nNodes_global+1];
    mesh.nodeStarOffsets[0] = 0;
    for (int nN=1;nN<mesh.nNodes_global+1;nN++)
      {
        mesh.nodeStarOffsets[nN] = mesh.nodeStarOffsets[nN-1] + nodeStar[nN-1].size();
      }  
    mesh.nodeStarArray = new int[mesh.nodeStarOffsets[mesh.nNodes_global]];
    for (int nN=0,offset=0;nN<mesh.nNodes_global;nN++)
      for (set<int>::iterator nN_star=nodeStar[nN].begin();nN_star!=nodeStar[nN].end();nN_star++,offset++)
        mesh.nodeStarArray[offset] = *nN_star;
    stop = CurrentTime();
    mesh.max_nNodeNeighbors_node=0;
    for (int nN=0;nN<mesh.nNodes_global;nN++)
      mesh.max_nNodeNeighbors_node=max(mesh.max_nNodeNeighbors_node,mesh.nodeStarOffsets[nN+1]-mesh.nodeStarOffsets[nN]);
    //mwf repeat for node-->elements arrays

    //std::cout<<"Extracting nodeElementsStar"<<std::endl;   
    vector<set<int> > nodeElementsStar(mesh.nNodes_global);
    for (int eN = 0; eN < mesh.nElements_global; eN++)
      {
	for (int nN = 0; nN < mesh.nNodes_element; nN++)
	  nodeElementsStar[mesh.elementNodesArray[eN*mesh.nNodes_element+nN]].insert(eN);
      }
    mesh.nodeElementOffsets = new int[mesh.nNodes_global+1];
    mesh.nodeElementOffsets[0] = 0;
    for (int nN = 0; nN < mesh.nNodes_global; nN++)
      mesh.nodeElementOffsets[nN+1] = mesh.nodeElementOffsets[nN]+nodeElementsStar[nN].size();
    mesh.nodeElementsArray  = new int[mesh.nodeElementOffsets[mesh.nNodes_global]];
    for (int nN=0,offset=0; nN < mesh.nNodes_global; nN++)
      {      	
	for (set<int>::iterator eN_star = nodeElementsStar[nN].begin(); eN_star != nodeElementsStar[nN].end();
	     eN_star++,offset++)
	  {
	    mesh.nodeElementsArray[offset] = *eN_star;
	  }
      }

    //std::cout<<"Set material types"<<std::endl;     
    //mwf end node-->elements construction
    mesh.elementBoundaryMaterialTypes = new int[mesh.nElementBoundaries_global];
    //if nodeMaterial is DEFAULT, go ahead and set to interior or exterior
    //depending on which boundary node belongs to. 
    //If node on at least one exterior boundary then it's exterior
    for (int ebNE = 0; ebNE < mesh.nExteriorElementBoundaries_global; ebNE++)
      {
	int ebN = mesh.exteriorElementBoundariesArray[ebNE];
	mesh.elementBoundaryMaterialTypes[ebN] = EXTERIOR_ELEMENT_BOUNDARY_MATERIAL;
	for (int nN_local = 0; nN_local < mesh.nNodes_elementBoundary; nN_local++)
	  {
	    int nN = mesh.elementBoundaryNodesArray[ebN*mesh.nNodes_elementBoundary+nN_local];
	    if (mesh.nodeMaterialTypes[nN] == DEFAULT_NODE_MATERIAL)
	      mesh.nodeMaterialTypes[nN] = EXTERIOR_NODE_MATERIAL;
	  }
      }
        
    for (int ebNI = 0; ebNI < mesh.nInteriorElementBoundaries_global; ebNI++)
      {
	int ebN = mesh.interiorElementBoundariesArray[ebNI];
	mesh.elementBoundaryMaterialTypes[ebN] = INTERIOR_ELEMENT_BOUNDARY_MATERIAL;
	for (int nN_local = 0; nN_local < mesh.nNodes_elementBoundary; nN_local++)
	  {
	    int nN = mesh.elementBoundaryNodesArray[ebN*mesh.nNodes_elementBoundary+nN_local];
	    if (mesh.nodeMaterialTypes[nN] == DEFAULT_NODE_MATERIAL)
	      mesh.nodeMaterialTypes[nN] = INTERIOR_NODE_MATERIAL;
	  }
      }
    //cout<<"Elapsed time for populating arrays = "<<(stop-start)<<"s"<<endl;
    return 0;
  }


  int constructElementBoundaryElementsArray_NURBS(Mesh& mesh)
  {
    using namespace std;

    int n0 = 0;
    int n1 = mesh.px ;
    int n2 = (mesh.px+1)*(mesh.py+1)-1 ;
    int n3 = (mesh.px+1)*mesh.py;


    int n4 = (mesh.px+1)*(mesh.py+1)*mesh.pz + n0;
    int n5 = (mesh.px+1)*(mesh.py+1)*mesh.pz + n1;
    int n6 = (mesh.px+1)*(mesh.py+1)*mesh.pz + n2;
    int n7 = (mesh.px+1)*(mesh.py+1)*mesh.pz + n3;
    
    
    
    mesh.nNodes_elementBoundary = 4;
    mesh.nElementBoundaries_element = 6;
    using namespace std;
    double start,stop;
    map<NodeTuple<4>,
      ElementNeighbors> elementBoundaryElements;
    start=CurrentTime();
    
    int lface[6][4] = {{n0,n1,n2,n3},
                       {n0,n1,n5,n4},
                       {n1,n2,n6,n5},
                       {n2,n3,n7,n6},
                       {n3,n0,n4,n7},
                       {n4,n5,n6,n7}};
    
    
    
    //cout<<"Extracting boundary elements"<<endl;
    for(int eN=0;eN<mesh.nElements_global;eN++)
      for(int ebN=0;ebN<mesh.nElementBoundaries_element;ebN++)
        {
          int nodes[4];
          nodes[0] = mesh.elementNodesArray[eN*8+lface[ebN][0]];
          nodes[1] = mesh.elementNodesArray[eN*8+lface[ebN][1]];
          nodes[2] = mesh.elementNodesArray[eN*8+lface[ebN][2]];      
          nodes[3] = mesh.elementNodesArray[eN*8+lface[ebN][3]];
                    
          NodeTuple<4> ebt(nodes);
          if(elementBoundaryElements.find(ebt) != elementBoundaryElements.end())
            {
              elementBoundaryElements[ebt].right=eN;
              elementBoundaryElements[ebt].right_ebN_element=ebN;
            }
          else
            {
              elementBoundaryElements.insert(elementBoundaryElements.end(),make_pair(ebt,ElementNeighbors(eN,ebN)));
            }
        }
    stop = CurrentTime();
    //cout<<"Elapsed time for building element boundary elements map= "<<(stop-start)<<"s"<<endl;
    mesh.nElementBoundaries_global = elementBoundaryElements.size();
    //cout<<"nElementBoundaries_global = "<<mesh.nElementBoundaries_global<<endl;

    //cout<<"Allocating Arrays"<<endl;
    start = CurrentTime();
    set<int> interiorElementBoundaries,exteriorElementBoundaries;
    mesh.elementBoundaryNodesArray =  new int[mesh.nElementBoundaries_global*mesh.nNodes_elementBoundary];
    mesh.elementBoundaryElementsArray = new int[mesh.nElementBoundaries_global*2];
    mesh.elementBoundaryLocalElementBoundariesArray = new int[mesh.nElementBoundaries_global*2];
    mesh.elementNeighborsArray = new int[mesh.nElements_global*mesh.nElementBoundaries_element];
    //mwf added
    mesh.elementBoundariesArray= new int[mesh.nElements_global*mesh.nElementBoundaries_element];
    stop = CurrentTime();
    //cout<<"Elapsed time for allocating arrays = "<<(stop-start)<<"s"<<endl;

    //cout<<"Generating elementBoundaryElementsArray and elementBoundaryNodesArray"<<endl;
    start = CurrentTime();
    int ebN=0;
    for(map<NodeTuple<4>,ElementNeighbors>::iterator eb=elementBoundaryElements.begin();
        eb != elementBoundaryElements.end();
        eb++,ebN++)
      {
        mesh.elementBoundaryNodesArray[ebN*4 + 0] = eb->first.nodes[0];
        mesh.elementBoundaryNodesArray[ebN*4 + 1] = eb->first.nodes[1];
        mesh.elementBoundaryNodesArray[ebN*4 + 2] = eb->first.nodes[2];
        mesh.elementBoundaryNodesArray[ebN*4 + 3] = eb->first.nodes[3];

        mesh.elementBoundaryElementsArray[ebN*2 + 0] = eb->second.left;
        mesh.elementBoundaryLocalElementBoundariesArray[ebN*2 + 0] = eb->second.left_ebN_element;
        mesh.elementBoundaryElementsArray[ebN*2 + 1] = eb->second.right;
        mesh.elementBoundaryLocalElementBoundariesArray[ebN*2 + 1] = eb->second.right_ebN_element;
        mesh.elementNeighborsArray[eb->second.left*mesh.nElementBoundaries_element + eb->second.left_ebN_element] = eb->second.right; 
        if(eb->second.right != -1)
	  {
	    interiorElementBoundaries.insert(ebN);
	    mesh.elementNeighborsArray[eb->second.right*mesh.nElementBoundaries_element + eb->second.right_ebN_element] = eb->second.left; 
	  }
        else
          exteriorElementBoundaries.insert(ebN);          
	//mwf added
	mesh.elementBoundariesArray[eb->second.left*mesh.nElementBoundaries_element + eb->second.left_ebN_element] = ebN;
	if (eb->second.right != -1)
	  {
	    mesh.elementBoundariesArray[eb->second.right*mesh.nElementBoundaries_element + eb->second.right_ebN_element] = ebN;
 	  }
      }
    mesh.nInteriorElementBoundaries_global = interiorElementBoundaries.size();
    mesh.interiorElementBoundariesArray = new int[mesh.nInteriorElementBoundaries_global];
    mesh.nExteriorElementBoundaries_global = exteriorElementBoundaries.size();
    mesh.exteriorElementBoundariesArray = new int[mesh.nExteriorElementBoundaries_global];
    int ebNI=0,ebNE=0;
    for (set<int>::iterator ebN=interiorElementBoundaries.begin();ebN != interiorElementBoundaries.end(); ebN++,ebNI++)
      mesh.interiorElementBoundariesArray[ebNI] = *ebN;
    for (set<int>::iterator ebN=exteriorElementBoundaries.begin();ebN != exteriorElementBoundaries.end(); ebN++,ebNE++)
      mesh.exteriorElementBoundariesArray[ebNE] = *ebN;
    set<NodeTuple<2> > edges;    
    
    

   
    int ledge[12][2] = {{n0,n1},{n1,n2},{n2,n3},{n3,n0},
                        {n0,n4},{n1,n5},{n2,n6},{n3,n7},
                        {n4,n5},{n5,n6},{n6,n7},{n7,n4}};
    
    
    
    
    //std::cout<<"extracting edges"<<std::endl;
    for (int eN=0;eN<mesh.nElements_global;eN++)
      {
	int nodes[2];
	for (int e=0;e<12;e++)
	  {
           	
	    nodes[0] = mesh.elementNodesArray[eN*8+ledge[e][0]];
	    nodes[1] = mesh.elementNodesArray[eN*8+ledge[e][1]];
                     
	    edges.insert(NodeTuple<2>(nodes));
	  }   
      }
      
      
    mesh.nEdges_global = edges.size();
    mesh.edgeNodesArray = new int[mesh.nEdges_global*2];
    set<NodeTuple<2> >::iterator edge_p=edges.begin();
    for (int edgeN=0;edgeN<int(edges.size());edgeN++)
      {
        mesh.edgeNodesArray[edgeN*2+0] = edge_p->nodes[0];
        mesh.edgeNodesArray[edgeN*2+1] = edge_p->nodes[1];
        edge_p++;
      }
      
    vector<set<int> > nodeStar(mesh.nNodes_global);
    for (int edgeN=0;edgeN<mesh.nEdges_global;edgeN++)
      {
        nodeStar[mesh.edgeNodesArray[edgeN*2+0]].insert(mesh.edgeNodesArray[edgeN*2+1]);
        nodeStar[mesh.edgeNodesArray[edgeN*2+1]].insert(mesh.edgeNodesArray[edgeN*2+0]);
      }

    mesh.nodeStarOffsets = new int[mesh.nNodes_global+1];
    mesh.nodeStarOffsets[0] = 0;
    for (int nN=1;nN<mesh.nNodes_global+1;nN++)
      mesh.nodeStarOffsets[nN] = mesh.nodeStarOffsets[nN-1] + nodeStar[nN-1].size();
    mesh.nodeStarArray = new int[mesh.nodeStarOffsets[mesh.nNodes_global]];
    for (int nN=0,offset=0;nN<mesh.nNodes_global;nN++)
      for (set<int>::iterator nN_star=nodeStar[nN].begin();nN_star!=nodeStar[nN].end();nN_star++,offset++)
        mesh.nodeStarArray[offset] = *nN_star;
    stop = CurrentTime();
    mesh.max_nNodeNeighbors_node=0;
    for (int nN=0;nN<mesh.nNodes_global;nN++)
      mesh.max_nNodeNeighbors_node=max(mesh.max_nNodeNeighbors_node,mesh.nodeStarOffsets[nN+1]-mesh.nodeStarOffsets[nN]);
    //mwf repeat for node-->elements arrays
  
    vector<set<int> > nodeElementsStar(mesh.nNodes_global);
    for (int eN = 0; eN < mesh.nElements_global; eN++)
      {
	for (int nN = 0; nN < mesh.nNodes_element; nN++)
	  nodeElementsStar[mesh.elementNodesArray[eN*mesh.nNodes_element+nN]].insert(eN);
      }
    mesh.nodeElementOffsets = new int[mesh.nNodes_global+1];
    mesh.nodeElementOffsets[0] = 0;
    for (int nN = 0; nN < mesh.nNodes_global; nN++)
      mesh.nodeElementOffsets[nN+1] = mesh.nodeElementOffsets[nN]+nodeElementsStar[nN].size();
    mesh.nodeElementsArray  = new int[mesh.nodeElementOffsets[mesh.nNodes_global]];
    for (int nN=0,offset=0; nN < mesh.nNodes_global; nN++)
      {      	
	for (set<int>::iterator eN_star = nodeElementsStar[nN].begin(); eN_star != nodeElementsStar[nN].end();
	     eN_star++,offset++)
	  {
	    mesh.nodeElementsArray[offset] = *eN_star;
	  }
      }
    
    //mwf end node-->elements construction
    mesh.elementBoundaryMaterialTypes = new int[mesh.nElementBoundaries_global];
    //if nodeMaterial is DEFAULT, go ahead and set to interior or exterior
    //depending on which boundary node belongs to. 
    //If node on at least one exterior boundary then it's exterior
    for (int ebNE = 0; ebNE < mesh.nExteriorElementBoundaries_global; ebNE++)
      {
	int ebN = mesh.exteriorElementBoundariesArray[ebNE];
	mesh.elementBoundaryMaterialTypes[ebN] = EXTERIOR_ELEMENT_BOUNDARY_MATERIAL;
	for (int nN_local = 0; nN_local < mesh.nNodes_elementBoundary; nN_local++)
	  {
	    int nN = mesh.elementBoundaryNodesArray[ebN*mesh.nNodes_elementBoundary+nN_local];
	    if (mesh.nodeMaterialTypes[nN] == DEFAULT_NODE_MATERIAL)
	      mesh.nodeMaterialTypes[nN] = EXTERIOR_NODE_MATERIAL;
	  }
      }
        
    for (int ebNI = 0; ebNI < mesh.nInteriorElementBoundaries_global; ebNI++)
      {
	int ebN = mesh.interiorElementBoundariesArray[ebNI];
	mesh.elementBoundaryMaterialTypes[ebN] = INTERIOR_ELEMENT_BOUNDARY_MATERIAL;
	for (int nN_local = 0; nN_local < mesh.nNodes_elementBoundary; nN_local++)
	  {
	    int nN = mesh.elementBoundaryNodesArray[ebN*mesh.nNodes_elementBoundary+nN_local];
	    if (mesh.nodeMaterialTypes[nN] == DEFAULT_NODE_MATERIAL)
	      mesh.nodeMaterialTypes[nN] = INTERIOR_NODE_MATERIAL;
	  }
      }
    //cout<<"Elapsed time for populating arrays = "<<(stop-start)<<"s"<<endl;
    return 0;
  }
  
  int constructElementBoundaryElementsArrayWithGivenElementBoundaryAndEdgeNumbers_NURBS(Mesh& mesh)
  {
 
    int n0 = 0;
    int n1 = mesh.px ;
    int n2 = (mesh.px+1)*(mesh.py+1)-1 ;
    int n3 = (mesh.px+1)*mesh.py;


    int n4 = (mesh.px+1)*(mesh.py+1)*mesh.pz + n0;
    int n5 = (mesh.px+1)*(mesh.py+1)*mesh.pz + n1;
    int n6 = (mesh.px+1)*(mesh.py+1)*mesh.pz + n2;
    int n7 = (mesh.px+1)*(mesh.py+1)*mesh.pz + n3;
 
    mesh.nNodes_elementBoundary = 4;
    mesh.nElementBoundaries_element = 6;
    assert(mesh.elementBoundariesArray);
    using namespace std;
    double start,stop;
    map<NodeTuple<4>,
      ElementNeighbors> elementBoundaryElements;
    map<NodeTuple<4>,
      int> elementBoundaryIds;
    start=CurrentTime();
    
    int lface[6][4] = {{n0,n1,n2,n3},
                       {n0,n1,n5,n4},
                       {n1,n2,n6,n5},
                       {n2,n3,n7,n6},
                       {n3,n0,n4,n7},
                       {n4,n5,n6,n7}};
    
    //cout<<"Extracting boundary elements"<<endl;
    for(int eN=0;eN<mesh.nElements_global;eN++)
      for(int ebN=0;ebN<mesh.nElementBoundaries_element;ebN++)
        {
	  int ebN_global = mesh.elementBoundariesArray[eN*mesh.nElementBoundaries_element+ebN];
          int nodes[4];
          nodes[0] = mesh.elementNodesArray[eN*8+lface[ebN][0]];
          nodes[1] = mesh.elementNodesArray[eN*8+lface[ebN][1]];
          nodes[2] = mesh.elementNodesArray[eN*8+lface[ebN][2]];      
          nodes[3] = mesh.elementNodesArray[eN*8+lface[ebN][3]];
          NodeTuple<4> ebt(nodes);
          if(elementBoundaryElements.find(ebt) != elementBoundaryElements.end())
            {
              elementBoundaryElements[ebt].right=eN;
              elementBoundaryElements[ebt].right_ebN_element=ebN;
 
	      // assert(elementBoundaryIds[ebt] == ebN_global);
            }
          else
            {
              elementBoundaryElements.insert(elementBoundaryElements.end(),make_pair(ebt,ElementNeighbors(eN,ebN)));
	      elementBoundaryIds.insert(elementBoundaryIds.end(),make_pair(ebt,ebN_global));
            }
        }
     stop = CurrentTime();
    //cout<<"Elapsed time for building element boundary elements map= "<<(stop-start)<<"s"<<endl;
    mesh.nElementBoundaries_global = elementBoundaryElements.size();
    //cout<<"nElementBoundaries_global = "<<mesh.nElementBoundaries_global<<endl;

    //cout<<"Allocating Arrays"<<endl;
    start = CurrentTime();
    set<int> interiorElementBoundaries,exteriorElementBoundaries;
    mesh.elementBoundaryNodesArray =  new int[mesh.nElementBoundaries_global*mesh.nNodes_elementBoundary];
    mesh.elementBoundaryElementsArray = new int[mesh.nElementBoundaries_global*2];
    mesh.elementBoundaryLocalElementBoundariesArray = new int[mesh.nElementBoundaries_global*2];
    mesh.elementNeighborsArray = new int[mesh.nElements_global*mesh.nElementBoundaries_element];
    stop = CurrentTime();
    //cout<<"Elapsed time for allocating arrays = "<<(stop-start)<<"s"<<endl;

    //cout<<"Generating elementBoundaryElementsArray and elementBoundaryNodesArray"<<endl;
    start = CurrentTime();
    for(map<NodeTuple<4>,ElementNeighbors>::iterator eb=elementBoundaryElements.begin();
        eb != elementBoundaryElements.end();
        eb++)
      {
	int ebN = elementBoundaryIds[eb->first];
        mesh.elementBoundaryNodesArray[ebN*4 + 0] = eb->first.nodes[0];
        mesh.elementBoundaryNodesArray[ebN*4 + 1] = eb->first.nodes[1];
        mesh.elementBoundaryNodesArray[ebN*4 + 2] = eb->first.nodes[2];
        mesh.elementBoundaryNodesArray[ebN*4 + 3] = eb->first.nodes[3];

        mesh.elementBoundaryElementsArray[ebN*2 + 0] = eb->second.left;
        mesh.elementBoundaryLocalElementBoundariesArray[ebN*2 + 0] = eb->second.left_ebN_element;
        mesh.elementBoundaryElementsArray[ebN*2 + 1] = eb->second.right;
        mesh.elementBoundaryLocalElementBoundariesArray[ebN*2 + 1] = eb->second.right_ebN_element;
        mesh.elementNeighborsArray[eb->second.left*mesh.nElementBoundaries_element + eb->second.left_ebN_element] = eb->second.right; 
        if(eb->second.right != -1)
	  {
	    interiorElementBoundaries.insert(ebN);
	    mesh.elementNeighborsArray[eb->second.right*mesh.nElementBoundaries_element + eb->second.right_ebN_element] = eb->second.left; 
	  }
        else
          exteriorElementBoundaries.insert(ebN);          
	//assert(mesh.elementBoundariesArray[eb->second.left*mesh.nElementBoundaries_element + eb->second.left_ebN_element] == ebN);
	if (eb->second.right != -1)
	  {
	    //  assert(mesh.elementBoundariesArray[eb->second.right*mesh.nElementBoundaries_element + eb->second.right_ebN_element] == ebN);
 	  }
      }
    mesh.nInteriorElementBoundaries_global = interiorElementBoundaries.size();
    mesh.interiorElementBoundariesArray = new int[mesh.nInteriorElementBoundaries_global];
    mesh.nExteriorElementBoundaries_global = exteriorElementBoundaries.size();
    mesh.exteriorElementBoundariesArray = new int[mesh.nExteriorElementBoundaries_global];
    int ebNI=0,ebNE=0;
    for (set<int>::iterator ebN=interiorElementBoundaries.begin();ebN != interiorElementBoundaries.end(); ebN++,ebNI++)
      mesh.interiorElementBoundariesArray[ebNI] = *ebN;
    for (set<int>::iterator ebN=exteriorElementBoundaries.begin();ebN != exteriorElementBoundaries.end(); ebN++,ebNE++)
      mesh.exteriorElementBoundariesArray[ebNE] = *ebN;
    //cek/ido todo figure out how to do this or if it's necessary
    //    assert(mesh.edgeNodesArray);

    //     set<NodeTuple<2> > edges;
    //     //std::cout<<"extracting edges"<<std::endl;
    //     for (int eN=0;eN<mesh.nElements_global;eN++)
    //       {
    //         int nodes[2];
    //         for (int nN_L=0;nN_L<mesh.nNodes_element;nN_L++)
    //           for (int nN_R=nN_L+1;nN_R<mesh.nNodes_element;nN_R++)
    //             {
    //               nodes[0] = mesh.elementNodesArray[eN*4+nN_L];
    //               nodes[1] = mesh.elementNodesArray[eN*4+nN_R];
    //               edges.insert(NodeTuple<2>(nodes));
    //             }
    //       }
    //     mesh.nEdges_global = edges.size();
    //     mesh.edgeNodesArray = new int[mesh.nEdges_global*2];
    //     set<NodeTuple<2> >::iterator edge_p=edges.begin();
    //     for (int edgeN=0;edgeN<int(edges.size());edgeN++)
    //       {
    //         mesh.edgeNodesArray[edgeN*2+0] = edge_p->nodes[0];
    //         mesh.edgeNodesArray[edgeN*2+1] = edge_p->nodes[1];
    //         edge_p++;
    //       }

    vector<set<int> > nodeStar(mesh.nNodes_global);
    for (int edgeN=0;edgeN<mesh.nEdges_global;edgeN++)
      {
        nodeStar[mesh.edgeNodesArray[edgeN*2+0]].insert(mesh.edgeNodesArray[edgeN*2+1]);
        nodeStar[mesh.edgeNodesArray[edgeN*2+1]].insert(mesh.edgeNodesArray[edgeN*2+0]);
      }
    mesh.nodeStarOffsets = new int[mesh.nNodes_global+1];
    mesh.nodeStarOffsets[0] = 0;
    for (int nN=1;nN<mesh.nNodes_global+1;nN++)
      mesh.nodeStarOffsets[nN] = mesh.nodeStarOffsets[nN-1] + nodeStar[nN-1].size();
    mesh.nodeStarArray = new int[mesh.nodeStarOffsets[mesh.nNodes_global]];
    for (int nN=0,offset=0;nN<mesh.nNodes_global;nN++)
      for (set<int>::iterator nN_star=nodeStar[nN].begin();nN_star!=nodeStar[nN].end();nN_star++,offset++)
        mesh.nodeStarArray[offset] = *nN_star;
    stop = CurrentTime();
    mesh.max_nNodeNeighbors_node=0;
    for (int nN=0;nN<mesh.nNodes_global;nN++)
      mesh.max_nNodeNeighbors_node=max(mesh.max_nNodeNeighbors_node,mesh.nodeStarOffsets[nN+1]-mesh.nodeStarOffsets[nN]);
    //mwf repeat for node-->elements arrays
    vector<set<int> > nodeElementsStar(mesh.nNodes_global);
    for (int eN = 0; eN < mesh.nElements_global; eN++)
      {
	for (int nN = 0; nN < mesh.nNodes_element; nN++)
	  nodeElementsStar[mesh.elementNodesArray[eN*mesh.nNodes_element+nN]].insert(eN);
      }
    mesh.nodeElementOffsets = new int[mesh.nNodes_global+1];
    mesh.nodeElementOffsets[0] = 0;
    for (int nN = 0; nN < mesh.nNodes_global; nN++)
      mesh.nodeElementOffsets[nN+1] = mesh.nodeElementOffsets[nN]+nodeElementsStar[nN].size();
    mesh.nodeElementsArray  = new int[mesh.nodeElementOffsets[mesh.nNodes_global]];
    for (int nN=0,offset=0; nN < mesh.nNodes_global; nN++)
      {
	for (set<int>::iterator eN_star = nodeElementsStar[nN].begin(); eN_star != nodeElementsStar[nN].end();
	     eN_star++,offset++)
	  {
	    mesh.nodeElementsArray[offset] = *eN_star;
	  }
      }
    //mwf end node-->elements construction
    mesh.elementBoundaryMaterialTypes = new int[mesh.nElementBoundaries_global];

    //if nodeMaterial is DEFAULT, go ahead and set to interior or exterior
    //depending on which boundary node belongs to. 
    //If node on at least one exterior boundary then it's exterior
    for (int ebNE = 0; ebNE < mesh.nExteriorElementBoundaries_global; ebNE++)
      {
	int ebN = mesh.exteriorElementBoundariesArray[ebNE];
	mesh.elementBoundaryMaterialTypes[ebN] = EXTERIOR_ELEMENT_BOUNDARY_MATERIAL;
	for (int nN_local = 0; nN_local < mesh.nNodes_elementBoundary; nN_local++)
	  {
	    int nN = mesh.elementBoundaryNodesArray[ebN*mesh.nNodes_elementBoundary+nN_local];
	    if (mesh.nodeMaterialTypes[nN] == DEFAULT_NODE_MATERIAL)
	      mesh.nodeMaterialTypes[nN] = EXTERIOR_NODE_MATERIAL;
	  }
      }
    for (int ebNI = 0; ebNI < mesh.nInteriorElementBoundaries_global; ebNI++)
      {
	int ebN = mesh.interiorElementBoundariesArray[ebNI];
	mesh.elementBoundaryMaterialTypes[ebN] = INTERIOR_ELEMENT_BOUNDARY_MATERIAL;
	for (int nN_local = 0; nN_local < mesh.nNodes_elementBoundary; nN_local++)
	  {
	    int nN = mesh.elementBoundaryNodesArray[ebN*mesh.nNodes_elementBoundary+nN_local];
	    if (mesh.nodeMaterialTypes[nN] == DEFAULT_NODE_MATERIAL)
	      mesh.nodeMaterialTypes[nN] = INTERIOR_NODE_MATERIAL;
	  }
      }
    //cout<<"Elapsed time for populating arrays = "<<(stop-start)<<"s"<<endl;
    return 0;
  }

  int constructElementBoundaryElementsArrayWithGivenElementBoundaryNumbers_edge(Mesh& mesh)
  {
    mesh.nNodes_elementBoundary = 1;
    mesh.nElementBoundaries_element = 2;
    assert(mesh.elementBoundariesArray);
    using namespace std;
    double start,stop;
    //cout<<"Constructing element boundary map"<<endl;
    map<NodeTuple<1>,
      ElementNeighbors> elementBoundaryElements;
    map<NodeTuple<1>,
      int> elementBoundaryIds;
    start=CurrentTime();
    for(int eN=0;eN<mesh.nElements_global;eN++)
      for(int ebN=0;ebN<2;ebN++)
        {
	  int ebN_global = mesh.elementBoundariesArray[eN*2+ebN];
          int nodes[1];
          nodes[0] = mesh.elementNodesArray[eN*2+(ebN+1)%2];
          NodeTuple<1> ebt(nodes);
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
    stop = CurrentTime();
    //cout<<"Elapsed time for building element boundary elements map= "<<(stop-start)<<"s"<<endl;
    mesh.nElementBoundaries_global = elementBoundaryElements.size();
    //cout<<"nElementBoundaries_global = "<<mesh.nElementBoundaries_global<<endl;

    //cout<<"Allocating Arrays"<<endl;
    start = CurrentTime();
    set<int> interiorElementBoundaries,exteriorElementBoundaries;
    mesh.elementBoundaryNodesArray =  new int[mesh.nElementBoundaries_global*mesh.nNodes_elementBoundary];
    mesh.elementBoundaryElementsArray = new int[mesh.nElementBoundaries_global*2];
    mesh.elementBoundaryLocalElementBoundariesArray = new int[mesh.nElementBoundaries_global*2];
    mesh.elementNeighborsArray = new int[mesh.nElements_global*mesh.nElementBoundaries_element];
    stop = CurrentTime();
    //cout<<"Elapsed time for allocating arrays = "<<(stop-start)<<"s"<<endl;

    //cout<<"Populating Arrays"<<endl;
    start = CurrentTime();
    for(map<NodeTuple<1>,ElementNeighbors>::iterator eb=elementBoundaryElements.begin();
        eb != elementBoundaryElements.end();
        eb++)
      {
	int ebN = elementBoundaryIds[eb->first];
        mesh.elementBoundaryNodesArray[ebN] = eb->first.nodes[0];
        mesh.elementBoundaryElementsArray[ebN*2 + 0] = eb->second.left;
        mesh.elementBoundaryLocalElementBoundariesArray[ebN*2 + 0] = eb->second.left_ebN_element;
        mesh.elementBoundaryElementsArray[ebN*2 + 1] = eb->second.right;
        mesh.elementBoundaryLocalElementBoundariesArray[ebN*2 + 1] = eb->second.right_ebN_element;
        mesh.elementNeighborsArray[eb->second.left*mesh.nElementBoundaries_element + eb->second.left_ebN_element] = eb->second.right; 
        if(eb->second.right != -1)
          {
            interiorElementBoundaries.insert(ebN);
            mesh.elementNeighborsArray[eb->second.right*mesh.nElementBoundaries_element + eb->second.right_ebN_element] = eb->second.left;
          }
        else
          exteriorElementBoundaries.insert(ebN);
	assert(mesh.elementBoundariesArray[eb->second.left*mesh.nElementBoundaries_element + eb->second.left_ebN_element] == ebN);
	if (eb->second.right != -1)
	  {
	    assert(mesh.elementBoundariesArray[eb->second.right*mesh.nElementBoundaries_element + eb->second.right_ebN_element] == ebN);
 	  }
      }
    mesh.nInteriorElementBoundaries_global = interiorElementBoundaries.size();
    mesh.interiorElementBoundariesArray = new int[mesh.nInteriorElementBoundaries_global];
    mesh.nExteriorElementBoundaries_global = exteriorElementBoundaries.size();
    mesh.exteriorElementBoundariesArray = new int[mesh.nExteriorElementBoundaries_global];
    int ebNI=0,ebNE=0;
    for (set<int>::iterator ebN=interiorElementBoundaries.begin();ebN != interiorElementBoundaries.end(); ebN++,ebNI++)
      mesh.interiorElementBoundariesArray[ebNI] = *ebN;
    for (set<int>::iterator ebN=exteriorElementBoundaries.begin();ebN != exteriorElementBoundaries.end(); ebN++,ebNE++)
      mesh.exteriorElementBoundariesArray[ebNE] = *ebN;
    //edges are elements in 1D so we just copy over but could be slicker
    mesh.nEdges_global = mesh.nElements_global;
    mesh.edgeNodesArray = new int[mesh.nElements_global*2];
    for (int eN=0;eN<mesh.nElements_global;eN++)
      {
        mesh.edgeNodesArray[eN*2+0] = mesh.elementNodesArray[eN*2+0];
        mesh.edgeNodesArray[eN*2+1] = mesh.elementNodesArray[eN*2+1];
      }
    vector<set<int> > nodeStar(mesh.nNodes_global);
    for (int eN=0;eN<mesh.nElements_global;eN++)
      {
        nodeStar[mesh.edgeNodesArray[eN*2+0]].insert(mesh.edgeNodesArray[eN*2+1]);
        nodeStar[mesh.edgeNodesArray[eN*2+1]].insert(mesh.edgeNodesArray[eN*2+0]);
      }
    mesh.nodeStarOffsets = new int[mesh.nNodes_global+1];
    mesh.nodeStarOffsets[0] = 0;
    for (int nN=1;nN<mesh.nNodes_global+1;nN++)
      mesh.nodeStarOffsets[nN] = mesh.nodeStarOffsets[nN-1]+nodeStar[nN-1].size();
    mesh.nodeStarArray = new int[mesh.nodeStarOffsets[mesh.nNodes_global]];
    for (int nN=0,offset=0;nN<mesh.nNodes_global;nN++)
      for (set<int>::iterator nN_star=nodeStar[nN].begin();nN_star!=nodeStar[nN].end();nN_star++,offset++)
        mesh.nodeStarArray[offset] = *nN_star;
    mesh.max_nNodeNeighbors_node=0;
    for (int nN=0;nN<mesh.nNodes_global;nN++)
      mesh.max_nNodeNeighbors_node=max(mesh.max_nNodeNeighbors_node,mesh.nodeStarOffsets[nN+1]-mesh.nodeStarOffsets[nN]);
    stop = CurrentTime();
    //repeat for node-->elements arrays
    vector<set<int> > nodeElementsStar(mesh.nNodes_global);
    for (int eN = 0; eN < mesh.nElements_global; eN++)
      {
	for (int nN = 0; nN < mesh.nNodes_element; nN++)
	  nodeElementsStar[mesh.elementNodesArray[eN*mesh.nNodes_element+nN]].insert(eN);
      }
    mesh.nodeElementOffsets = new int[mesh.nNodes_global+1];
    mesh.nodeElementOffsets[0] = 0;
    for (int nN = 0; nN < mesh.nNodes_global; nN++)
      mesh.nodeElementOffsets[nN+1] = mesh.nodeElementOffsets[nN]+nodeElementsStar[nN].size();
    mesh.nodeElementsArray  = new int[mesh.nodeElementOffsets[mesh.nNodes_global]];
    for (int nN=0,offset=0; nN < mesh.nNodes_global; nN++)
      {
	for (set<int>::iterator eN_star = nodeElementsStar[nN].begin(); eN_star != nodeElementsStar[nN].end();
	     eN_star++,offset++)
	  {
	    mesh.nodeElementsArray[offset] = *eN_star;
	  }
      }
    //end node-->elements construction
    mesh.elementBoundaryMaterialTypes = new int[mesh.nElementBoundaries_global];
    //if nodeMaterial is DEFAULT, go ahead and set to interior or exterior
    //depending on which boundary node belongs to. 
    //If node on at least one exterior boundary then it's exterior
    for (int ebNE = 0; ebNE < mesh.nExteriorElementBoundaries_global; ebNE++)
      {
	int ebN = mesh.exteriorElementBoundariesArray[ebNE];
	mesh.elementBoundaryMaterialTypes[ebN] = EXTERIOR_ELEMENT_BOUNDARY_MATERIAL;
	for (int nN_local = 0; nN_local < mesh.nNodes_elementBoundary; nN_local++)
	  {
	    int nN = mesh.elementBoundaryNodesArray[ebN*mesh.nNodes_elementBoundary+nN_local];
	    if (mesh.nodeMaterialTypes[nN] == DEFAULT_NODE_MATERIAL)
	      mesh.nodeMaterialTypes[nN] = EXTERIOR_NODE_MATERIAL;
	  }
      }
    for (int ebNI = 0; ebNI < mesh.nInteriorElementBoundaries_global; ebNI++)
      {
	int ebN = mesh.interiorElementBoundariesArray[ebNI];
	mesh.elementBoundaryMaterialTypes[ebN] = INTERIOR_ELEMENT_BOUNDARY_MATERIAL;
	for (int nN_local = 0; nN_local < mesh.nNodes_elementBoundary; nN_local++)
	  {
	    int nN = mesh.elementBoundaryNodesArray[ebN*mesh.nNodes_elementBoundary+nN_local];
	    if (mesh.nodeMaterialTypes[nN] == DEFAULT_NODE_MATERIAL)
	      mesh.nodeMaterialTypes[nN] = INTERIOR_NODE_MATERIAL;
	  }
      }


    //cout<<"Elapsed time for populating arrays = "<<(stop-start)<<"s"<<endl;
    return 0;
  }

  int constructElementBoundaryElementsArrayWithGivenElementBoundaryNumbers_triangle(Mesh& mesh)
  {
    using namespace std;
    mesh.nNodes_elementBoundary = 2;
    mesh.nElementBoundaries_element = 3;
    assert(mesh.elementBoundariesArray);
    double start,stop;
    //cout<<"Constructing element boundary map"<<endl;
    map<NodeTuple<2>,
      ElementNeighbors> elementBoundaryElements;
    map<NodeTuple<2>,
      int> elementBoundaryIds;
    start=CurrentTime();
    for(int eN=0;eN<mesh.nElements_global;eN++)
      for(int ebN=0;ebN<3;ebN++)
        {
	  int ebN_global = mesh.elementBoundariesArray[eN*3+ebN];
          int nodes[2];
          nodes[0] = mesh.elementNodesArray[eN*3+(ebN+1)%3];
          nodes[1] = mesh.elementNodesArray[eN*3+(ebN+2)%3];
          NodeTuple<2> ebt(nodes);
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
    stop = CurrentTime();
    //cout<<"Elapsed time for building element boundary elements map= "<<(stop-start)<<"s"<<endl;
    mesh.nElementBoundaries_global = elementBoundaryElements.size();
    //cout<<"nElementBoundaries_global = "<<mesh.nElementBoundaries_global<<endl;
    
    //cout<<"Allocating Arrays"<<endl;
    start = CurrentTime();
    set<int> interiorElementBoundaries,exteriorElementBoundaries;
    mesh.elementBoundaryNodesArray =  new int[mesh.nElementBoundaries_global*mesh.nNodes_elementBoundary];
    mesh.elementBoundaryElementsArray = new int[mesh.nElementBoundaries_global*2];
    mesh.elementBoundaryLocalElementBoundariesArray = new int[mesh.nElementBoundaries_global*2];
    mesh.elementNeighborsArray = new int[mesh.nElements_global*mesh.nElementBoundaries_element];
    stop = CurrentTime();
    //cout<<"Elapsed time for allocating arrays = "<<(stop-start)<<"s"<<endl;

    //cout<<"Populating arrays"<<endl;
    start = CurrentTime();
    for(map<NodeTuple<2>,ElementNeighbors>::iterator eb=elementBoundaryElements.begin();
        eb != elementBoundaryElements.end();
        eb++)
      {
	int ebN = elementBoundaryIds[eb->first];
        mesh.elementBoundaryNodesArray[ebN*2 + 0] = eb->first.nodes[0];
        mesh.elementBoundaryNodesArray[ebN*2 + 1] = eb->first.nodes[1];
        mesh.elementBoundaryElementsArray[ebN*2 + 0] = eb->second.left;
        mesh.elementBoundaryLocalElementBoundariesArray[ebN*2 + 0] = eb->second.left_ebN_element;
        mesh.elementBoundaryElementsArray[ebN*2 + 1] = eb->second.right;
        mesh.elementBoundaryLocalElementBoundariesArray[ebN*2 + 1] = eb->second.right_ebN_element;
        mesh.elementNeighborsArray[eb->second.left*mesh.nElementBoundaries_element + eb->second.left_ebN_element] = eb->second.right; 
        if(eb->second.right != -1)
          {
            interiorElementBoundaries.insert(ebN);
            mesh.elementNeighborsArray[eb->second.right*mesh.nElementBoundaries_element + eb->second.right_ebN_element] = eb->second.left;
          }
        else
          exteriorElementBoundaries.insert(ebN);          
	assert(mesh.elementBoundariesArray[eb->second.left*mesh.nElementBoundaries_element + eb->second.left_ebN_element] == ebN);
	if (eb->second.right != -1)
	  {
	    assert(mesh.elementBoundariesArray[eb->second.right*mesh.nElementBoundaries_element + eb->second.right_ebN_element] == ebN);
 	  }
      }
    mesh.nInteriorElementBoundaries_global = interiorElementBoundaries.size();
    mesh.interiorElementBoundariesArray = new int[mesh.nInteriorElementBoundaries_global];
    mesh.nExteriorElementBoundaries_global = exteriorElementBoundaries.size();
    mesh.exteriorElementBoundariesArray = new int[mesh.nExteriorElementBoundaries_global];
    int ebNI=0,ebNE=0;
    for (set<int>::iterator ebN=interiorElementBoundaries.begin();ebN != interiorElementBoundaries.end(); ebN++,ebNI++)
      mesh.interiorElementBoundariesArray[ebNI] = *ebN;
    for (set<int>::iterator ebN=exteriorElementBoundaries.begin();ebN != exteriorElementBoundaries.end(); ebN++,ebNE++)
      mesh.exteriorElementBoundariesArray[ebNE] = *ebN;
    set<NodeTuple<2> > edges;
    for (int eN=0;eN<mesh.nElements_global;eN++)
      {
        int nodes[2];
        for (int nN_L=0;nN_L<mesh.nNodes_element;nN_L++)
          for (int nN_R=nN_L+1;nN_R<mesh.nNodes_element;nN_R++)
            {
              nodes[0] = mesh.elementNodesArray[eN*3+nN_L];
              nodes[1] = mesh.elementNodesArray[eN*3+nN_R];
              edges.insert(NodeTuple<2>(nodes));
            }
      }
    mesh.nEdges_global = edges.size();
    mesh.edgeNodesArray = new int[mesh.nEdges_global*2];
    int edgeN=0;
    for (set<NodeTuple<2> >::iterator edgeTuple_p=edges.begin();edgeTuple_p != edges.end();edgeTuple_p++,edgeN++)
      {
        mesh.edgeNodesArray[edgeN*2+0] = edgeTuple_p->nodes[0];
        mesh.edgeNodesArray[edgeN*2+1] = edgeTuple_p->nodes[1];
      }
    vector<set<int> > nodeStar(mesh.nNodes_global);
    for (int edgeN=0;edgeN<mesh.nEdges_global;edgeN++)
      {
        nodeStar[mesh.edgeNodesArray[edgeN*2+0]].insert(mesh.edgeNodesArray[edgeN*2+1]);
        nodeStar[mesh.edgeNodesArray[edgeN*2+1]].insert(mesh.edgeNodesArray[edgeN*2+0]);
      }
    mesh.nodeStarOffsets = new int[mesh.nNodes_global+1];
    mesh.nodeStarOffsets[0] = 0;
    for (int nN=1;nN<mesh.nNodes_global+1;nN++)
      mesh.nodeStarOffsets[nN] = mesh.nodeStarOffsets[nN-1]+nodeStar[nN-1].size();
    mesh.nodeStarArray = new int[mesh.nodeStarOffsets[mesh.nNodes_global]];
    for (int nN=0,offset=0;nN<mesh.nNodes_global;nN++)
      for (set<int>::iterator nN_star=nodeStar[nN].begin();nN_star!=nodeStar[nN].end();nN_star++,offset++)
        mesh.nodeStarArray[offset] = *nN_star;
    stop = CurrentTime();
    mesh.max_nNodeNeighbors_node=0;
    for (int nN=0;nN<mesh.nNodes_global;nN++)
      mesh.max_nNodeNeighbors_node=max(mesh.max_nNodeNeighbors_node,mesh.nodeStarOffsets[nN+1]-mesh.nodeStarOffsets[nN]);
    //repeat for node-->elements arrays
    vector<set<int> > nodeElementsStar(mesh.nNodes_global);
    for (int eN = 0; eN < mesh.nElements_global; eN++)
      {
	for (int nN = 0; nN < mesh.nNodes_element; nN++)
	  nodeElementsStar[mesh.elementNodesArray[eN*mesh.nNodes_element+nN]].insert(eN);
      }
    mesh.nodeElementOffsets = new int[mesh.nNodes_global+1];
    mesh.nodeElementOffsets[0] = 0;
    for (int nN = 0; nN < mesh.nNodes_global; nN++)
      mesh.nodeElementOffsets[nN+1] = mesh.nodeElementOffsets[nN]+nodeElementsStar[nN].size();
    mesh.nodeElementsArray  = new int[mesh.nodeElementOffsets[mesh.nNodes_global]];
    for (int nN=0,offset=0; nN < mesh.nNodes_global; nN++)
      {
	for (set<int>::iterator eN_star = nodeElementsStar[nN].begin(); eN_star != nodeElementsStar[nN].end();
	     eN_star++,offset++)
	  {
	    mesh.nodeElementsArray[offset] = *eN_star;
	  }
      }
    //end node-->elements construction
    mesh.elementBoundaryMaterialTypes = new int[mesh.nElementBoundaries_global];
    //if nodeMaterial is DEFAULT, go ahead and set to interior or exterior
    //depending on which boundary node belongs to. 
    //If node on at least one exterior boundary then it's exterior
    //set actual values elsewhere
    for (int ebNE = 0; ebNE < mesh.nExteriorElementBoundaries_global; ebNE++)
      {
	int ebN = mesh.exteriorElementBoundariesArray[ebNE];
	mesh.elementBoundaryMaterialTypes[ebN] = EXTERIOR_ELEMENT_BOUNDARY_MATERIAL;
	for (int nN_local = 0; nN_local < mesh.nNodes_elementBoundary; nN_local++)
	  {
	    int nN = mesh.elementBoundaryNodesArray[ebN*mesh.nNodes_elementBoundary+nN_local];
	    if (mesh.nodeMaterialTypes[nN] == DEFAULT_NODE_MATERIAL)
	      mesh.nodeMaterialTypes[nN] = EXTERIOR_NODE_MATERIAL;
	  }
      }
    for (int ebNI = 0; ebNI < mesh.nInteriorElementBoundaries_global; ebNI++)
      {
	int ebN = mesh.interiorElementBoundariesArray[ebNI];
	mesh.elementBoundaryMaterialTypes[ebN] = INTERIOR_ELEMENT_BOUNDARY_MATERIAL;
	for (int nN_local = 0; nN_local < mesh.nNodes_elementBoundary; nN_local++)
	  {
	    int nN = mesh.elementBoundaryNodesArray[ebN*mesh.nNodes_elementBoundary+nN_local];
	    if (mesh.nodeMaterialTypes[nN] == DEFAULT_NODE_MATERIAL)
	      mesh.nodeMaterialTypes[nN] = INTERIOR_NODE_MATERIAL;
	  }
      }
    //cout<<"Elapsed time for populating arrays = "<<(stop-start)<<"s"<<endl;
    return 0;
  }
  
  int constructElementBoundaryElementsArrayWithGivenElementBoundaryNumbers_quadrilateral(Mesh& mesh)
  {
    using namespace std;
    mesh.nNodes_elementBoundary = 2;
    mesh.nElementBoundaries_element = 4;
    assert(mesh.elementBoundariesArray);
    double start,stop;
    //cout<<"Constructing element boundary map"<<endl;
    map<NodeTuple<2>,
      ElementNeighbors> elementBoundaryElements;
    map<NodeTuple<2>,
      int> elementBoundaryIds;
    start=CurrentTime();
    for(int eN=0;eN<mesh.nElements_global;eN++)
      for(int ebN=0;ebN<4;ebN++)
        {
	  int ebN_global = mesh.elementBoundariesArray[eN*4+ebN];
          int nodes[2];
          nodes[0] = mesh.elementNodesArray[eN*4+ebN];
          nodes[1] = mesh.elementNodesArray[eN*4+(ebN+1)%4];
          NodeTuple<2> ebt(nodes);
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
    stop = CurrentTime();
    //cout<<"Elapsed time for building element boundary elements map= "<<(stop-start)<<"s"<<endl;
    mesh.nElementBoundaries_global = elementBoundaryElements.size();
    //cout<<"nElementBoundaries_global = "<<mesh.nElementBoundaries_global<<endl;
    
    //cout<<"Allocating Arrays"<<endl;
    start = CurrentTime();
    set<int> interiorElementBoundaries,exteriorElementBoundaries;
    mesh.elementBoundaryNodesArray =  new int[mesh.nElementBoundaries_global*mesh.nNodes_elementBoundary];
    mesh.elementBoundaryElementsArray = new int[mesh.nElementBoundaries_global*2];
    mesh.elementBoundaryLocalElementBoundariesArray = new int[mesh.nElementBoundaries_global*2];
    mesh.elementNeighborsArray = new int[mesh.nElements_global*mesh.nElementBoundaries_element];
    stop = CurrentTime();
    //cout<<"Elapsed time for allocating arrays = "<<(stop-start)<<"s"<<endl;

    //cout<<"Populating arrays"<<endl;
    start = CurrentTime();
    for(map<NodeTuple<2>,ElementNeighbors>::iterator eb=elementBoundaryElements.begin();
        eb != elementBoundaryElements.end();
        eb++)
      {
	int ebN = elementBoundaryIds[eb->first];
        mesh.elementBoundaryNodesArray[ebN*2 + 0] = eb->first.nodes[0];
        mesh.elementBoundaryNodesArray[ebN*2 + 1] = eb->first.nodes[1];
        mesh.elementBoundaryElementsArray[ebN*2 + 0] = eb->second.left;
        mesh.elementBoundaryLocalElementBoundariesArray[ebN*2 + 0] = eb->second.left_ebN_element;
        mesh.elementBoundaryElementsArray[ebN*2 + 1] = eb->second.right;
        mesh.elementBoundaryLocalElementBoundariesArray[ebN*2 + 1] = eb->second.right_ebN_element;
        mesh.elementNeighborsArray[eb->second.left*mesh.nElementBoundaries_element + eb->second.left_ebN_element] = eb->second.right; 
        if(eb->second.right != -1)
          {
            interiorElementBoundaries.insert(ebN);
            mesh.elementNeighborsArray[eb->second.right*mesh.nElementBoundaries_element + eb->second.right_ebN_element] = eb->second.left;
          }
        else
          exteriorElementBoundaries.insert(ebN);          
	assert(mesh.elementBoundariesArray[eb->second.left*mesh.nElementBoundaries_element + eb->second.left_ebN_element] == ebN);
	if (eb->second.right != -1)
	  {
	    assert(mesh.elementBoundariesArray[eb->second.right*mesh.nElementBoundaries_element + eb->second.right_ebN_element] == ebN);
 	  }
      }
    mesh.nInteriorElementBoundaries_global = interiorElementBoundaries.size();
    mesh.interiorElementBoundariesArray = new int[mesh.nInteriorElementBoundaries_global];
    mesh.nExteriorElementBoundaries_global = exteriorElementBoundaries.size();
    mesh.exteriorElementBoundariesArray = new int[mesh.nExteriorElementBoundaries_global];
    int ebNI=0,ebNE=0;
    for (set<int>::iterator ebN=interiorElementBoundaries.begin();ebN != interiorElementBoundaries.end(); ebN++,ebNI++)
      mesh.interiorElementBoundariesArray[ebNI] = *ebN;
    for (set<int>::iterator ebN=exteriorElementBoundaries.begin();ebN != exteriorElementBoundaries.end(); ebN++,ebNE++)
      mesh.exteriorElementBoundariesArray[ebNE] = *ebN;
    set<NodeTuple<2> > edges;
    for (int eN=0;eN<mesh.nElements_global;eN++)
      {
        int nodes[2];
        for (int nN_L=0;nN_L<mesh.nNodes_element;nN_L++)
          {
            nodes[0] = mesh.elementNodesArray[eN*4+nN_L];
            nodes[1] = mesh.elementNodesArray[eN*4+(nN_L+1)%4];
            edges.insert(NodeTuple<2>(nodes));
          }
      }
    mesh.nEdges_global = edges.size();
    mesh.edgeNodesArray = new int[mesh.nEdges_global*2];
    int edgeN=0;
    for (set<NodeTuple<2> >::iterator edgeTuple_p=edges.begin();edgeTuple_p != edges.end();edgeTuple_p++,edgeN++)
      {
        mesh.edgeNodesArray[edgeN*2+0] = edgeTuple_p->nodes[0];
        mesh.edgeNodesArray[edgeN*2+1] = edgeTuple_p->nodes[1];
      }
    vector<set<int> > nodeStar(mesh.nNodes_global);
    for (int edgeN=0;edgeN<mesh.nEdges_global;edgeN++)
      {
        nodeStar[mesh.edgeNodesArray[edgeN*2+0]].insert(mesh.edgeNodesArray[edgeN*2+1]);
        nodeStar[mesh.edgeNodesArray[edgeN*2+1]].insert(mesh.edgeNodesArray[edgeN*2+0]);
      }
    mesh.nodeStarOffsets = new int[mesh.nNodes_global+1];
    mesh.nodeStarOffsets[0] = 0;
    for (int nN=1;nN<mesh.nNodes_global+1;nN++)
      mesh.nodeStarOffsets[nN] = mesh.nodeStarOffsets[nN-1]+nodeStar[nN-1].size();
    mesh.nodeStarArray = new int[mesh.nodeStarOffsets[mesh.nNodes_global]];
    for (int nN=0,offset=0;nN<mesh.nNodes_global;nN++)
      for (set<int>::iterator nN_star=nodeStar[nN].begin();nN_star!=nodeStar[nN].end();nN_star++,offset++)
        mesh.nodeStarArray[offset] = *nN_star;
    stop = CurrentTime();
    mesh.max_nNodeNeighbors_node=0;
    for (int nN=0;nN<mesh.nNodes_global;nN++)
      mesh.max_nNodeNeighbors_node=max(mesh.max_nNodeNeighbors_node,mesh.nodeStarOffsets[nN+1]-mesh.nodeStarOffsets[nN]);
    //repeat for node-->elements arrays
    vector<set<int> > nodeElementsStar(mesh.nNodes_global);
    for (int eN = 0; eN < mesh.nElements_global; eN++)
      {
	for (int nN = 0; nN < mesh.nNodes_element; nN++)
	  nodeElementsStar[mesh.elementNodesArray[eN*mesh.nNodes_element+nN]].insert(eN);
      }
    mesh.nodeElementOffsets = new int[mesh.nNodes_global+1];
    mesh.nodeElementOffsets[0] = 0;
    for (int nN = 0; nN < mesh.nNodes_global; nN++)
      mesh.nodeElementOffsets[nN+1] = mesh.nodeElementOffsets[nN]+nodeElementsStar[nN].size();
    mesh.nodeElementsArray  = new int[mesh.nodeElementOffsets[mesh.nNodes_global]];
    for (int nN=0,offset=0; nN < mesh.nNodes_global; nN++)
      {
	for (set<int>::iterator eN_star = nodeElementsStar[nN].begin(); eN_star != nodeElementsStar[nN].end();
	     eN_star++,offset++)
	  {
	    mesh.nodeElementsArray[offset] = *eN_star;
	  }
      }
    //end node-->elements construction
    mesh.elementBoundaryMaterialTypes = new int[mesh.nElementBoundaries_global];
    //if nodeMaterial is DEFAULT, go ahead and set to interior or exterior
    //depending on which boundary node belongs to. 
    //If node on at least one exterior boundary then it's exterior
    //set actual values elsewhere
    for (int ebNE = 0; ebNE < mesh.nExteriorElementBoundaries_global; ebNE++)
      {
	int ebN = mesh.exteriorElementBoundariesArray[ebNE];
	mesh.elementBoundaryMaterialTypes[ebN] = EXTERIOR_ELEMENT_BOUNDARY_MATERIAL;
	for (int nN_local = 0; nN_local < mesh.nNodes_elementBoundary; nN_local++)
	  {
	    int nN = mesh.elementBoundaryNodesArray[ebN*mesh.nNodes_elementBoundary+nN_local];
	    if (mesh.nodeMaterialTypes[nN] == DEFAULT_NODE_MATERIAL)
	      mesh.nodeMaterialTypes[nN] = EXTERIOR_NODE_MATERIAL;
	  }
      }
    for (int ebNI = 0; ebNI < mesh.nInteriorElementBoundaries_global; ebNI++)
      {
	int ebN = mesh.interiorElementBoundariesArray[ebNI];
	mesh.elementBoundaryMaterialTypes[ebN] = INTERIOR_ELEMENT_BOUNDARY_MATERIAL;
	for (int nN_local = 0; nN_local < mesh.nNodes_elementBoundary; nN_local++)
	  {
	    int nN = mesh.elementBoundaryNodesArray[ebN*mesh.nNodes_elementBoundary+nN_local];
	    if (mesh.nodeMaterialTypes[nN] == DEFAULT_NODE_MATERIAL)
	      mesh.nodeMaterialTypes[nN] = INTERIOR_NODE_MATERIAL;
	  }
      }
    //cout<<"Elapsed time for populating arrays = "<<(stop-start)<<"s"<<endl;
    return 0;
  }

  int constructElementBoundaryElementsArrayWithGivenElementBoundaryNumbers_tetrahedron(Mesh& mesh)
  {
    mesh.nNodes_elementBoundary = 3;
    mesh.nElementBoundaries_element = 4;
    assert(mesh.elementBoundariesArray);
    using namespace std;
    double start,stop;
    map<NodeTuple<3>,
      ElementNeighbors> elementBoundaryElements;
    map<NodeTuple<3>,
      int> elementBoundaryIds;
    start=CurrentTime();
    //cout<<"Extracting boundary elements"<<endl;
    for(int eN=0;eN<mesh.nElements_global;eN++)
      for(int ebN=0;ebN<mesh.nElementBoundaries_element;ebN++)
        {
	  int ebN_global = mesh.elementBoundariesArray[eN*mesh.nElementBoundaries_element+ebN];
          int nodes[3];
          nodes[0] = mesh.elementNodesArray[eN*4+((ebN+1)%4)];
          nodes[1] = mesh.elementNodesArray[eN*4+((ebN+2)%4)];
          nodes[2] = mesh.elementNodesArray[eN*4+((ebN+3)%4)];
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
    stop = CurrentTime();
    //cout<<"Elapsed time for building element boundary elements map= "<<(stop-start)<<"s"<<endl;
    mesh.nElementBoundaries_global = elementBoundaryElements.size();
    //cout<<"nElementBoundaries_global = "<<mesh.nElementBoundaries_global<<endl;

    //cout<<"Allocating Arrays"<<endl;
    start = CurrentTime();
    set<int> interiorElementBoundaries,exteriorElementBoundaries;
    mesh.elementBoundaryNodesArray =  new int[mesh.nElementBoundaries_global*mesh.nNodes_elementBoundary];
    mesh.elementBoundaryElementsArray = new int[mesh.nElementBoundaries_global*2];
    mesh.elementBoundaryLocalElementBoundariesArray = new int[mesh.nElementBoundaries_global*2];
    mesh.elementNeighborsArray = new int[mesh.nElements_global*mesh.nElementBoundaries_element];
    stop = CurrentTime();
    //cout<<"Elapsed time for allocating arrays = "<<(stop-start)<<"s"<<endl;

    //cout<<"Generating elementBoundaryElementsArray and elementBoundaryNodesArray"<<endl;
    start = CurrentTime();
    for(map<NodeTuple<3>,ElementNeighbors>::iterator eb=elementBoundaryElements.begin();
        eb != elementBoundaryElements.end();
        eb++)
      {
	int ebN = elementBoundaryIds[eb->first];
        mesh.elementBoundaryNodesArray[ebN*3 + 0] = eb->first.nodes[0];
        mesh.elementBoundaryNodesArray[ebN*3 + 1] = eb->first.nodes[1];
        mesh.elementBoundaryNodesArray[ebN*3 + 2] = eb->first.nodes[2];

        mesh.elementBoundaryElementsArray[ebN*2 + 0] = eb->second.left;
        mesh.elementBoundaryLocalElementBoundariesArray[ebN*2 + 0] = eb->second.left_ebN_element;
        mesh.elementBoundaryElementsArray[ebN*2 + 1] = eb->second.right;
        mesh.elementBoundaryLocalElementBoundariesArray[ebN*2 + 1] = eb->second.right_ebN_element;
        mesh.elementNeighborsArray[eb->second.left*mesh.nElementBoundaries_element + eb->second.left_ebN_element] = eb->second.right; 
        if(eb->second.right != -1)
	  {
	    interiorElementBoundaries.insert(ebN);
	    mesh.elementNeighborsArray[eb->second.right*mesh.nElementBoundaries_element + eb->second.right_ebN_element] = eb->second.left; 
	  }
        else
          exteriorElementBoundaries.insert(ebN);          
	assert(mesh.elementBoundariesArray[eb->second.left*mesh.nElementBoundaries_element + eb->second.left_ebN_element] == ebN);
	if (eb->second.right != -1)
	  {
	    assert(mesh.elementBoundariesArray[eb->second.right*mesh.nElementBoundaries_element + eb->second.right_ebN_element] == ebN);
 	  }
      }
    mesh.nInteriorElementBoundaries_global = interiorElementBoundaries.size();
    mesh.interiorElementBoundariesArray = new int[mesh.nInteriorElementBoundaries_global];
    mesh.nExteriorElementBoundaries_global = exteriorElementBoundaries.size();
    mesh.exteriorElementBoundariesArray = new int[mesh.nExteriorElementBoundaries_global];
    int ebNI=0,ebNE=0;
    for (set<int>::iterator ebN=interiorElementBoundaries.begin();ebN != interiorElementBoundaries.end(); ebN++,ebNI++)
      mesh.interiorElementBoundariesArray[ebNI] = *ebN;
    for (set<int>::iterator ebN=exteriorElementBoundaries.begin();ebN != exteriorElementBoundaries.end(); ebN++,ebNE++)
      mesh.exteriorElementBoundariesArray[ebNE] = *ebN;
    set<NodeTuple<2> > edges;
    //std::cout<<"extracting edges"<<std::endl;
    for (int eN=0;eN<mesh.nElements_global;eN++)
      {
        int nodes[2];
        for (int nN_L=0;nN_L<mesh.nNodes_element;nN_L++)
          for (int nN_R=nN_L+1;nN_R<mesh.nNodes_element;nN_R++)
            {
              nodes[0] = mesh.elementNodesArray[eN*4+nN_L];
              nodes[1] = mesh.elementNodesArray[eN*4+nN_R];
              edges.insert(NodeTuple<2>(nodes));
            }
      }
    mesh.nEdges_global = edges.size();
    mesh.edgeNodesArray = new int[mesh.nEdges_global*2];
    set<NodeTuple<2> >::iterator edge_p=edges.begin();
    for (int edgeN=0;edgeN<int(edges.size());edgeN++)
      {
        mesh.edgeNodesArray[edgeN*2+0] = edge_p->nodes[0];
        mesh.edgeNodesArray[edgeN*2+1] = edge_p->nodes[1];
        edge_p++;
      }
    vector<set<int> > nodeStar(mesh.nNodes_global);
    for (int edgeN=0;edgeN<mesh.nEdges_global;edgeN++)
      {
        nodeStar[mesh.edgeNodesArray[edgeN*2+0]].insert(mesh.edgeNodesArray[edgeN*2+1]);
        nodeStar[mesh.edgeNodesArray[edgeN*2+1]].insert(mesh.edgeNodesArray[edgeN*2+0]);
      }
    mesh.nodeStarOffsets = new int[mesh.nNodes_global+1];
    mesh.nodeStarOffsets[0] = 0;
    for (int nN=1;nN<mesh.nNodes_global+1;nN++)
      mesh.nodeStarOffsets[nN] = mesh.nodeStarOffsets[nN-1] + nodeStar[nN-1].size();
    mesh.nodeStarArray = new int[mesh.nodeStarOffsets[mesh.nNodes_global]];
    for (int nN=0,offset=0;nN<mesh.nNodes_global;nN++)
      for (set<int>::iterator nN_star=nodeStar[nN].begin();nN_star!=nodeStar[nN].end();nN_star++,offset++)
        mesh.nodeStarArray[offset] = *nN_star;
    stop = CurrentTime();
    mesh.max_nNodeNeighbors_node=0;
    for (int nN=0;nN<mesh.nNodes_global;nN++)
      mesh.max_nNodeNeighbors_node=max(mesh.max_nNodeNeighbors_node,mesh.nodeStarOffsets[nN+1]-mesh.nodeStarOffsets[nN]);
    //mwf repeat for node-->elements arrays
    vector<set<int> > nodeElementsStar(mesh.nNodes_global);
    for (int eN = 0; eN < mesh.nElements_global; eN++)
      {
	for (int nN = 0; nN < mesh.nNodes_element; nN++)
	  nodeElementsStar[mesh.elementNodesArray[eN*mesh.nNodes_element+nN]].insert(eN);
      }
    mesh.nodeElementOffsets = new int[mesh.nNodes_global+1];
    mesh.nodeElementOffsets[0] = 0;
    for (int nN = 0; nN < mesh.nNodes_global; nN++)
      mesh.nodeElementOffsets[nN+1] = mesh.nodeElementOffsets[nN]+nodeElementsStar[nN].size();
    mesh.nodeElementsArray  = new int[mesh.nodeElementOffsets[mesh.nNodes_global]];
    for (int nN=0,offset=0; nN < mesh.nNodes_global; nN++)
      {
	for (set<int>::iterator eN_star = nodeElementsStar[nN].begin(); eN_star != nodeElementsStar[nN].end();
	     eN_star++,offset++)
	  {
	    mesh.nodeElementsArray[offset] = *eN_star;
	  }
      }
    //mwf end node-->elements construction
    mesh.elementBoundaryMaterialTypes = new int[mesh.nElementBoundaries_global];
    //if nodeMaterial is DEFAULT, go ahead and set to interior or exterior
    //depending on which boundary node belongs to. 
    //If node on at least one exterior boundary then it's exterior
    for (int ebNE = 0; ebNE < mesh.nExteriorElementBoundaries_global; ebNE++)
      {
	int ebN = mesh.exteriorElementBoundariesArray[ebNE];
	mesh.elementBoundaryMaterialTypes[ebN] = EXTERIOR_ELEMENT_BOUNDARY_MATERIAL;
	for (int nN_local = 0; nN_local < mesh.nNodes_elementBoundary; nN_local++)
	  {
	    int nN = mesh.elementBoundaryNodesArray[ebN*mesh.nNodes_elementBoundary+nN_local];
	    if (mesh.nodeMaterialTypes[nN] == DEFAULT_NODE_MATERIAL)
	      mesh.nodeMaterialTypes[nN] = EXTERIOR_NODE_MATERIAL;
	  }
      }
    for (int ebNI = 0; ebNI < mesh.nInteriorElementBoundaries_global; ebNI++)
      {
	int ebN = mesh.interiorElementBoundariesArray[ebNI];
	mesh.elementBoundaryMaterialTypes[ebN] = INTERIOR_ELEMENT_BOUNDARY_MATERIAL;
	for (int nN_local = 0; nN_local < mesh.nNodes_elementBoundary; nN_local++)
	  {
	    int nN = mesh.elementBoundaryNodesArray[ebN*mesh.nNodes_elementBoundary+nN_local];
	    if (mesh.nodeMaterialTypes[nN] == DEFAULT_NODE_MATERIAL)
	      mesh.nodeMaterialTypes[nN] = INTERIOR_NODE_MATERIAL;
	  }
      }
    //cout<<"Elapsed time for populating arrays = "<<(stop-start)<<"s"<<endl;
    return 0;
  }



  int constructElementBoundaryElementsArrayWithGivenElementBoundaryAndEdgeNumbers_edge(Mesh& mesh)
  {
    mesh.nNodes_elementBoundary = 1;
    mesh.nElementBoundaries_element = 2;
    assert(mesh.elementBoundariesArray);
    using namespace std;
    double start,stop;
    //cout<<"Constructing element boundary map"<<endl;
    map<NodeTuple<1>,
      ElementNeighbors> elementBoundaryElements;
    map<NodeTuple<1>,
      int> elementBoundaryIds;
    start=CurrentTime();
    for(int eN=0;eN<mesh.nElements_global;eN++)
      for(int ebN=0;ebN<2;ebN++)
        {
	  int ebN_global = mesh.elementBoundariesArray[eN*2+ebN];
          int nodes[1];
          nodes[0] = mesh.elementNodesArray[eN*2+(ebN+1)%2];
          NodeTuple<1> ebt(nodes);
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
    stop = CurrentTime();
    //cout<<"Elapsed time for building element boundary elements map= "<<(stop-start)<<"s"<<endl;
    mesh.nElementBoundaries_global = elementBoundaryElements.size();
    //cout<<"nElementBoundaries_global = "<<mesh.nElementBoundaries_global<<endl;

    //cout<<"Allocating Arrays"<<endl;
    start = CurrentTime();
    set<int> interiorElementBoundaries,exteriorElementBoundaries;
    mesh.elementBoundaryNodesArray =  new int[mesh.nElementBoundaries_global*mesh.nNodes_elementBoundary];
    mesh.elementBoundaryElementsArray = new int[mesh.nElementBoundaries_global*2];
    mesh.elementBoundaryLocalElementBoundariesArray = new int[mesh.nElementBoundaries_global*2];
    mesh.elementNeighborsArray = new int[mesh.nElements_global*mesh.nElementBoundaries_element];
    stop = CurrentTime();
    //cout<<"Elapsed time for allocating arrays = "<<(stop-start)<<"s"<<endl;

    //cout<<"Populating Arrays"<<endl;
    start = CurrentTime();
    for(map<NodeTuple<1>,ElementNeighbors>::iterator eb=elementBoundaryElements.begin();
        eb != elementBoundaryElements.end();
        eb++)
      {
	int ebN = elementBoundaryIds[eb->first];
        mesh.elementBoundaryNodesArray[ebN] = eb->first.nodes[0];
        mesh.elementBoundaryElementsArray[ebN*2 + 0] = eb->second.left;
        mesh.elementBoundaryLocalElementBoundariesArray[ebN*2 + 0] = eb->second.left_ebN_element;
        mesh.elementBoundaryElementsArray[ebN*2 + 1] = eb->second.right;
        mesh.elementBoundaryLocalElementBoundariesArray[ebN*2 + 1] = eb->second.right_ebN_element;
        mesh.elementNeighborsArray[eb->second.left*mesh.nElementBoundaries_element + eb->second.left_ebN_element] = eb->second.right; 
        if(eb->second.right != -1)
          {
            interiorElementBoundaries.insert(ebN);
            mesh.elementNeighborsArray[eb->second.right*mesh.nElementBoundaries_element + eb->second.right_ebN_element] = eb->second.left;
          }
        else
          exteriorElementBoundaries.insert(ebN);
	assert(mesh.elementBoundariesArray[eb->second.left*mesh.nElementBoundaries_element + eb->second.left_ebN_element] == ebN);
	if (eb->second.right != -1)
	  {
	    assert(mesh.elementBoundariesArray[eb->second.right*mesh.nElementBoundaries_element + eb->second.right_ebN_element] == ebN);
 	  }
      }
    mesh.nInteriorElementBoundaries_global = interiorElementBoundaries.size();
    mesh.interiorElementBoundariesArray = new int[mesh.nInteriorElementBoundaries_global];
    mesh.nExteriorElementBoundaries_global = exteriorElementBoundaries.size();
    mesh.exteriorElementBoundariesArray = new int[mesh.nExteriorElementBoundaries_global];
    int ebNI=0,ebNE=0;
    for (set<int>::iterator ebN=interiorElementBoundaries.begin();ebN != interiorElementBoundaries.end(); ebN++,ebNI++)
      mesh.interiorElementBoundariesArray[ebNI] = *ebN;
    for (set<int>::iterator ebN=exteriorElementBoundaries.begin();ebN != exteriorElementBoundaries.end(); ebN++,ebNE++)
      mesh.exteriorElementBoundariesArray[ebNE] = *ebN;

    assert(mesh.edgeNodesArray);
    vector<set<int> > nodeStar(mesh.nNodes_global);
    for (int eN=0;eN<mesh.nElements_global;eN++)
      {
        nodeStar[mesh.edgeNodesArray[eN*2+0]].insert(mesh.edgeNodesArray[eN*2+1]);
        nodeStar[mesh.edgeNodesArray[eN*2+1]].insert(mesh.edgeNodesArray[eN*2+0]);
      }
    mesh.nodeStarOffsets = new int[mesh.nNodes_global+1];
    mesh.nodeStarOffsets[0] = 0;
    for (int nN=1;nN<mesh.nNodes_global+1;nN++)
      mesh.nodeStarOffsets[nN] = mesh.nodeStarOffsets[nN-1]+nodeStar[nN-1].size();
    mesh.nodeStarArray = new int[mesh.nodeStarOffsets[mesh.nNodes_global]];
    for (int nN=0,offset=0;nN<mesh.nNodes_global;nN++)
      for (set<int>::iterator nN_star=nodeStar[nN].begin();nN_star!=nodeStar[nN].end();nN_star++,offset++)
        mesh.nodeStarArray[offset] = *nN_star;
    mesh.max_nNodeNeighbors_node=0;
    for (int nN=0;nN<mesh.nNodes_global;nN++)
      mesh.max_nNodeNeighbors_node=max(mesh.max_nNodeNeighbors_node,mesh.nodeStarOffsets[nN+1]-mesh.nodeStarOffsets[nN]);
    stop = CurrentTime();
    //repeat for node-->elements arrays
    vector<set<int> > nodeElementsStar(mesh.nNodes_global);
    for (int eN = 0; eN < mesh.nElements_global; eN++)
      {
	for (int nN = 0; nN < mesh.nNodes_element; nN++)
	  nodeElementsStar[mesh.elementNodesArray[eN*mesh.nNodes_element+nN]].insert(eN);
      }
    mesh.nodeElementOffsets = new int[mesh.nNodes_global+1];
    mesh.nodeElementOffsets[0] = 0;
    for (int nN = 0; nN < mesh.nNodes_global; nN++)
      mesh.nodeElementOffsets[nN+1] = mesh.nodeElementOffsets[nN]+nodeElementsStar[nN].size();
    mesh.nodeElementsArray  = new int[mesh.nodeElementOffsets[mesh.nNodes_global]];
    for (int nN=0,offset=0; nN < mesh.nNodes_global; nN++)
      {
	for (set<int>::iterator eN_star = nodeElementsStar[nN].begin(); eN_star != nodeElementsStar[nN].end();
	     eN_star++,offset++)
	  {
	    mesh.nodeElementsArray[offset] = *eN_star;
	  }
      }
    //end node-->elements construction
    mesh.elementBoundaryMaterialTypes = new int[mesh.nElementBoundaries_global];
    //if nodeMaterial is DEFAULT, go ahead and set to interior or exterior
    //depending on which boundary node belongs to. 
    //If node on at least one exterior boundary then it's exterior
    for (int ebNE = 0; ebNE < mesh.nExteriorElementBoundaries_global; ebNE++)
      {
	int ebN = mesh.exteriorElementBoundariesArray[ebNE];
	mesh.elementBoundaryMaterialTypes[ebN] = EXTERIOR_ELEMENT_BOUNDARY_MATERIAL;
	for (int nN_local = 0; nN_local < mesh.nNodes_elementBoundary; nN_local++)
	  {
	    int nN = mesh.elementBoundaryNodesArray[ebN*mesh.nNodes_elementBoundary+nN_local];
	    if (mesh.nodeMaterialTypes[nN] == DEFAULT_NODE_MATERIAL)
	      mesh.nodeMaterialTypes[nN] = EXTERIOR_NODE_MATERIAL;
	  }
      }
    for (int ebNI = 0; ebNI < mesh.nInteriorElementBoundaries_global; ebNI++)
      {
	int ebN = mesh.interiorElementBoundariesArray[ebNI];
	mesh.elementBoundaryMaterialTypes[ebN] = INTERIOR_ELEMENT_BOUNDARY_MATERIAL;
	for (int nN_local = 0; nN_local < mesh.nNodes_elementBoundary; nN_local++)
	  {
	    int nN = mesh.elementBoundaryNodesArray[ebN*mesh.nNodes_elementBoundary+nN_local];
	    if (mesh.nodeMaterialTypes[nN] == DEFAULT_NODE_MATERIAL)
	      mesh.nodeMaterialTypes[nN] = INTERIOR_NODE_MATERIAL;
	  }
      }


    //cout<<"Elapsed time for populating arrays = "<<(stop-start)<<"s"<<endl;
    return 0;
  }

  int constructElementBoundaryElementsArrayWithGivenElementBoundaryAndEdgeNumbers_triangle(Mesh& mesh)
  {
    using namespace std;
    mesh.nNodes_elementBoundary = 2;
    mesh.nElementBoundaries_element = 3;
    assert(mesh.elementBoundariesArray);
    double start,stop;
    //cout<<"Constructing element boundary map"<<endl;
    map<NodeTuple<2>,
      ElementNeighbors> elementBoundaryElements;
    map<NodeTuple<2>,
      int> elementBoundaryIds;
    start=CurrentTime();
    for(int eN=0;eN<mesh.nElements_global;eN++)
      for(int ebN=0;ebN<3;ebN++)
        {
	  int ebN_global = mesh.elementBoundariesArray[eN*3+ebN];
          int nodes[2];
          nodes[0] = mesh.elementNodesArray[eN*3+(ebN+1)%3];
          nodes[1] = mesh.elementNodesArray[eN*3+(ebN+2)%3];
          NodeTuple<2> ebt(nodes);
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
    stop = CurrentTime();
    //cout<<"Elapsed time for building element boundary elements map= "<<(stop-start)<<"s"<<endl;
    mesh.nElementBoundaries_global = elementBoundaryElements.size();
    //cout<<"nElementBoundaries_global = "<<mesh.nElementBoundaries_global<<endl;
    
    //cout<<"Allocating Arrays"<<endl;
    start = CurrentTime();
    set<int> interiorElementBoundaries,exteriorElementBoundaries;
    mesh.elementBoundaryNodesArray =  new int[mesh.nElementBoundaries_global*mesh.nNodes_elementBoundary];
    mesh.elementBoundaryElementsArray = new int[mesh.nElementBoundaries_global*2];
    mesh.elementBoundaryLocalElementBoundariesArray = new int[mesh.nElementBoundaries_global*2];
    mesh.elementNeighborsArray = new int[mesh.nElements_global*mesh.nElementBoundaries_element];
    stop = CurrentTime();
    //cout<<"Elapsed time for allocating arrays = "<<(stop-start)<<"s"<<endl;

    //cout<<"Populating arrays"<<endl;
    start = CurrentTime();
    for(map<NodeTuple<2>,ElementNeighbors>::iterator eb=elementBoundaryElements.begin();
        eb != elementBoundaryElements.end();
        eb++)
      {
	int ebN = elementBoundaryIds[eb->first];
        mesh.elementBoundaryNodesArray[ebN*2 + 0] = eb->first.nodes[0];
        mesh.elementBoundaryNodesArray[ebN*2 + 1] = eb->first.nodes[1];
        mesh.elementBoundaryElementsArray[ebN*2 + 0] = eb->second.left;
        mesh.elementBoundaryLocalElementBoundariesArray[ebN*2 + 0] = eb->second.left_ebN_element;
        mesh.elementBoundaryElementsArray[ebN*2 + 1] = eb->second.right;
        mesh.elementBoundaryLocalElementBoundariesArray[ebN*2 + 1] = eb->second.right_ebN_element;
        mesh.elementNeighborsArray[eb->second.left*mesh.nElementBoundaries_element + eb->second.left_ebN_element] = eb->second.right; 
        if(eb->second.right != -1)
          {
            interiorElementBoundaries.insert(ebN);
            mesh.elementNeighborsArray[eb->second.right*mesh.nElementBoundaries_element + eb->second.right_ebN_element] = eb->second.left;
          }
        else
          exteriorElementBoundaries.insert(ebN);          
	assert(mesh.elementBoundariesArray[eb->second.left*mesh.nElementBoundaries_element + eb->second.left_ebN_element] == ebN);
	if (eb->second.right != -1)
	  {
	    assert(mesh.elementBoundariesArray[eb->second.right*mesh.nElementBoundaries_element + eb->second.right_ebN_element] == ebN);
 	  }
      }
    mesh.nInteriorElementBoundaries_global = interiorElementBoundaries.size();
    mesh.interiorElementBoundariesArray = new int[mesh.nInteriorElementBoundaries_global];
    mesh.nExteriorElementBoundaries_global = exteriorElementBoundaries.size();
    mesh.exteriorElementBoundariesArray = new int[mesh.nExteriorElementBoundaries_global];
    int ebNI=0,ebNE=0;
    for (set<int>::iterator ebN=interiorElementBoundaries.begin();ebN != interiorElementBoundaries.end(); ebN++,ebNI++)
      mesh.interiorElementBoundariesArray[ebNI] = *ebN;
    for (set<int>::iterator ebN=exteriorElementBoundaries.begin();ebN != exteriorElementBoundaries.end(); ebN++,ebNE++)
      mesh.exteriorElementBoundariesArray[ebNE] = *ebN;
    assert(mesh.edgeNodesArray);
    //    set<NodeTuple<2> > edges;
    //     for (int eN=0;eN<mesh.nElements_global;eN++)
    //       {
    //         int nodes[2];
    //         for (int nN_L=0;nN_L<mesh.nNodes_element;nN_L++)
    //           for (int nN_R=nN_L+1;nN_R<mesh.nNodes_element;nN_R++)
    //             {
    //               nodes[0] = mesh.elementNodesArray[eN*3+nN_L];
    //               nodes[1] = mesh.elementNodesArray[eN*3+nN_R];
    //               edges.insert(NodeTuple<2>(nodes));
    //             }
    //       }
    //     mesh.nEdges_global = edges.size();
    //     mesh.edgeNodesArray = new int[mesh.nEdges_global*2];
    //     int edgeN=0;
    //     for (set<NodeTuple<2> >::iterator edgeTuple_p=edges.begin();edgeTuple_p != edges.end();edgeTuple_p++,edgeN++)
    //       {
    //         mesh.edgeNodesArray[edgeN*2+0] = edgeTuple_p->nodes[0];
    //         mesh.edgeNodesArray[edgeN*2+1] = edgeTuple_p->nodes[1];
    //       }
    vector<set<int> > nodeStar(mesh.nNodes_global);
    for (int edgeN=0;edgeN<mesh.nEdges_global;edgeN++)
      {
        nodeStar[mesh.edgeNodesArray[edgeN*2+0]].insert(mesh.edgeNodesArray[edgeN*2+1]);
        nodeStar[mesh.edgeNodesArray[edgeN*2+1]].insert(mesh.edgeNodesArray[edgeN*2+0]);
      }
    mesh.nodeStarOffsets = new int[mesh.nNodes_global+1];
    mesh.nodeStarOffsets[0] = 0;
    for (int nN=1;nN<mesh.nNodes_global+1;nN++)
      mesh.nodeStarOffsets[nN] = mesh.nodeStarOffsets[nN-1]+nodeStar[nN-1].size();
    mesh.nodeStarArray = new int[mesh.nodeStarOffsets[mesh.nNodes_global]];
    for (int nN=0,offset=0;nN<mesh.nNodes_global;nN++)
      for (set<int>::iterator nN_star=nodeStar[nN].begin();nN_star!=nodeStar[nN].end();nN_star++,offset++)
        mesh.nodeStarArray[offset] = *nN_star;
    stop = CurrentTime();
    mesh.max_nNodeNeighbors_node=0;
    for (int nN=0;nN<mesh.nNodes_global;nN++)
      mesh.max_nNodeNeighbors_node=max(mesh.max_nNodeNeighbors_node,mesh.nodeStarOffsets[nN+1]-mesh.nodeStarOffsets[nN]);
    //repeat for node-->elements arrays
    vector<set<int> > nodeElementsStar(mesh.nNodes_global);
    for (int eN = 0; eN < mesh.nElements_global; eN++)
      {
	for (int nN = 0; nN < mesh.nNodes_element; nN++)
	  nodeElementsStar[mesh.elementNodesArray[eN*mesh.nNodes_element+nN]].insert(eN);
      }
    mesh.nodeElementOffsets = new int[mesh.nNodes_global+1];
    mesh.nodeElementOffsets[0] = 0;
    for (int nN = 0; nN < mesh.nNodes_global; nN++)
      mesh.nodeElementOffsets[nN+1] = mesh.nodeElementOffsets[nN]+nodeElementsStar[nN].size();
    mesh.nodeElementsArray  = new int[mesh.nodeElementOffsets[mesh.nNodes_global]];
    for (int nN=0,offset=0; nN < mesh.nNodes_global; nN++)
      {
	for (set<int>::iterator eN_star = nodeElementsStar[nN].begin(); eN_star != nodeElementsStar[nN].end();
	     eN_star++,offset++)
	  {
	    mesh.nodeElementsArray[offset] = *eN_star;
	  }
      }
    //end node-->elements construction
    mesh.elementBoundaryMaterialTypes = new int[mesh.nElementBoundaries_global];
    //if nodeMaterial is DEFAULT, go ahead and set to interior or exterior
    //depending on which boundary node belongs to. 
    //If node on at least one exterior boundary then it's exterior
    //set actual values elsewhere
    for (int ebNE = 0; ebNE < mesh.nExteriorElementBoundaries_global; ebNE++)
      {
	int ebN = mesh.exteriorElementBoundariesArray[ebNE];
	mesh.elementBoundaryMaterialTypes[ebN] = EXTERIOR_ELEMENT_BOUNDARY_MATERIAL;
	for (int nN_local = 0; nN_local < mesh.nNodes_elementBoundary; nN_local++)
	  {
	    int nN = mesh.elementBoundaryNodesArray[ebN*mesh.nNodes_elementBoundary+nN_local];
	    if (mesh.nodeMaterialTypes[nN] == DEFAULT_NODE_MATERIAL)
	      mesh.nodeMaterialTypes[nN] = EXTERIOR_NODE_MATERIAL;
	  }
      }
    for (int ebNI = 0; ebNI < mesh.nInteriorElementBoundaries_global; ebNI++)
      {
	int ebN = mesh.interiorElementBoundariesArray[ebNI];
	mesh.elementBoundaryMaterialTypes[ebN] = INTERIOR_ELEMENT_BOUNDARY_MATERIAL;
	for (int nN_local = 0; nN_local < mesh.nNodes_elementBoundary; nN_local++)
	  {
	    int nN = mesh.elementBoundaryNodesArray[ebN*mesh.nNodes_elementBoundary+nN_local];
	    if (mesh.nodeMaterialTypes[nN] == DEFAULT_NODE_MATERIAL)
	      mesh.nodeMaterialTypes[nN] = INTERIOR_NODE_MATERIAL;
	  }
      }
    //cout<<"Elapsed time for populating arrays = "<<(stop-start)<<"s"<<endl;
    return 0;
  }

  int constructElementBoundaryElementsArrayWithGivenElementBoundaryAndEdgeNumbers_quadrilateral(Mesh& mesh)
  {
    using namespace std;
    mesh.nNodes_elementBoundary = 2;
    mesh.nElementBoundaries_element = 4;
    assert(mesh.elementBoundariesArray);
    double start,stop;
    //cout<<"Constructing element boundary map"<<endl;
    map<NodeTuple<2>,
      ElementNeighbors> elementBoundaryElements;
    map<NodeTuple<2>,
      int> elementBoundaryIds;
    start=CurrentTime();
    for(int eN=0;eN<mesh.nElements_global;eN++)
      for(int ebN=0;ebN<4;ebN++)
        {
	  int ebN_global = mesh.elementBoundariesArray[eN*4+ebN];
	  int nodes[2];
          nodes[0] = mesh.elementNodesArray[eN*4+ebN];
          nodes[1] = mesh.elementNodesArray[eN*4+(ebN+1)%4];
          NodeTuple<2> ebt(nodes);
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
    stop = CurrentTime();
    //cout<<"Elapsed time for building element boundary elements map= "<<(stop-start)<<"s"<<endl;
    mesh.nElementBoundaries_global = elementBoundaryElements.size();
    //cout<<"nElementBoundaries_global = "<<mesh.nElementBoundaries_global<<endl;
    
    //cout<<"Allocating Arrays"<<endl;
    start = CurrentTime();
    set<int> interiorElementBoundaries,exteriorElementBoundaries;
    mesh.elementBoundaryNodesArray =  new int[mesh.nElementBoundaries_global*mesh.nNodes_elementBoundary];
    mesh.elementBoundaryElementsArray = new int[mesh.nElementBoundaries_global*2];
    mesh.elementBoundaryLocalElementBoundariesArray = new int[mesh.nElementBoundaries_global*2];
    mesh.elementNeighborsArray = new int[mesh.nElements_global*mesh.nElementBoundaries_element];
    stop = CurrentTime();
    //cout<<"Elapsed time for allocating arrays = "<<(stop-start)<<"s"<<endl;

    //cout<<"Populating arrays"<<endl;
    start = CurrentTime();
    for(map<NodeTuple<2>,ElementNeighbors>::iterator eb=elementBoundaryElements.begin();
        eb != elementBoundaryElements.end();
        eb++)
      {
	int ebN = elementBoundaryIds[eb->first];
        mesh.elementBoundaryNodesArray[ebN*2 + 0] = eb->first.nodes[0];
        mesh.elementBoundaryNodesArray[ebN*2 + 1] = eb->first.nodes[1];
        mesh.elementBoundaryElementsArray[ebN*2 + 0] = eb->second.left;
        mesh.elementBoundaryLocalElementBoundariesArray[ebN*2 + 0] = eb->second.left_ebN_element;
        mesh.elementBoundaryElementsArray[ebN*2 + 1] = eb->second.right;
        mesh.elementBoundaryLocalElementBoundariesArray[ebN*2 + 1] = eb->second.right_ebN_element;
        mesh.elementNeighborsArray[eb->second.left*mesh.nElementBoundaries_element + eb->second.left_ebN_element] = eb->second.right; 
        if(eb->second.right != -1)
          {
            interiorElementBoundaries.insert(ebN);
            mesh.elementNeighborsArray[eb->second.right*mesh.nElementBoundaries_element + eb->second.right_ebN_element] = eb->second.left;
          }
        else
          exteriorElementBoundaries.insert(ebN);          
	assert(mesh.elementBoundariesArray[eb->second.left*mesh.nElementBoundaries_element + eb->second.left_ebN_element] == ebN);
	if (eb->second.right != -1)
	  {
	    assert(mesh.elementBoundariesArray[eb->second.right*mesh.nElementBoundaries_element + eb->second.right_ebN_element] == ebN);
 	  }
      }
    mesh.nInteriorElementBoundaries_global = interiorElementBoundaries.size();
    mesh.interiorElementBoundariesArray = new int[mesh.nInteriorElementBoundaries_global];
    mesh.nExteriorElementBoundaries_global = exteriorElementBoundaries.size();
    mesh.exteriorElementBoundariesArray = new int[mesh.nExteriorElementBoundaries_global];
    int ebNI=0,ebNE=0;
    for (set<int>::iterator ebN=interiorElementBoundaries.begin();ebN != interiorElementBoundaries.end(); ebN++,ebNI++)
      mesh.interiorElementBoundariesArray[ebNI] = *ebN;
    for (set<int>::iterator ebN=exteriorElementBoundaries.begin();ebN != exteriorElementBoundaries.end(); ebN++,ebNE++)
      mesh.exteriorElementBoundariesArray[ebNE] = *ebN;
    assert(mesh.edgeNodesArray);
    vector<set<int> > nodeStar(mesh.nNodes_global);
    for (int edgeN=0;edgeN<mesh.nEdges_global;edgeN++)
      {
        nodeStar[mesh.edgeNodesArray[edgeN*2+0]].insert(mesh.edgeNodesArray[edgeN*2+1]);
        nodeStar[mesh.edgeNodesArray[edgeN*2+1]].insert(mesh.edgeNodesArray[edgeN*2+0]);
      }
    mesh.nodeStarOffsets = new int[mesh.nNodes_global+1];
    mesh.nodeStarOffsets[0] = 0;
    for (int nN=1;nN<mesh.nNodes_global+1;nN++)
      mesh.nodeStarOffsets[nN] = mesh.nodeStarOffsets[nN-1]+nodeStar[nN-1].size();
    mesh.nodeStarArray = new int[mesh.nodeStarOffsets[mesh.nNodes_global]];
    for (int nN=0,offset=0;nN<mesh.nNodes_global;nN++)
      for (set<int>::iterator nN_star=nodeStar[nN].begin();nN_star!=nodeStar[nN].end();nN_star++,offset++)
        mesh.nodeStarArray[offset] = *nN_star;
    stop = CurrentTime();
    mesh.max_nNodeNeighbors_node=0;
    for (int nN=0;nN<mesh.nNodes_global;nN++)
      mesh.max_nNodeNeighbors_node=max(mesh.max_nNodeNeighbors_node,mesh.nodeStarOffsets[nN+1]-mesh.nodeStarOffsets[nN]);
    //repeat for node-->elements arrays
    vector<set<int> > nodeElementsStar(mesh.nNodes_global);
    for (int eN = 0; eN < mesh.nElements_global; eN++)
      {
	for (int nN = 0; nN < mesh.nNodes_element; nN++)
	  nodeElementsStar[mesh.elementNodesArray[eN*mesh.nNodes_element+nN]].insert(eN);
      }
    mesh.nodeElementOffsets = new int[mesh.nNodes_global+1];
    mesh.nodeElementOffsets[0] = 0;
    for (int nN = 0; nN < mesh.nNodes_global; nN++)
      mesh.nodeElementOffsets[nN+1] = mesh.nodeElementOffsets[nN]+nodeElementsStar[nN].size();
    mesh.nodeElementsArray  = new int[mesh.nodeElementOffsets[mesh.nNodes_global]];
    for (int nN=0,offset=0; nN < mesh.nNodes_global; nN++)
      {
	for (set<int>::iterator eN_star = nodeElementsStar[nN].begin(); eN_star != nodeElementsStar[nN].end();
	     eN_star++,offset++)
	  {
	    mesh.nodeElementsArray[offset] = *eN_star;
	  }
      }
    //end node-->elements construction
    mesh.elementBoundaryMaterialTypes = new int[mesh.nElementBoundaries_global];
    //if nodeMaterial is DEFAULT, go ahead and set to interior or exterior
    //depending on which boundary node belongs to. 
    //If node on at least one exterior boundary then it's exterior
    //set actual values elsewhere
    for (int ebNE = 0; ebNE < mesh.nExteriorElementBoundaries_global; ebNE++)
      {
	int ebN = mesh.exteriorElementBoundariesArray[ebNE];
	mesh.elementBoundaryMaterialTypes[ebN] = EXTERIOR_ELEMENT_BOUNDARY_MATERIAL;
	for (int nN_local = 0; nN_local < mesh.nNodes_elementBoundary; nN_local++)
	  {
	    int nN = mesh.elementBoundaryNodesArray[ebN*mesh.nNodes_elementBoundary+nN_local];
	    if (mesh.nodeMaterialTypes[nN] == DEFAULT_NODE_MATERIAL)
	      mesh.nodeMaterialTypes[nN] = EXTERIOR_NODE_MATERIAL;
	  }
      }
    for (int ebNI = 0; ebNI < mesh.nInteriorElementBoundaries_global; ebNI++)
      {
	int ebN = mesh.interiorElementBoundariesArray[ebNI];
	mesh.elementBoundaryMaterialTypes[ebN] = INTERIOR_ELEMENT_BOUNDARY_MATERIAL;
	for (int nN_local = 0; nN_local < mesh.nNodes_elementBoundary; nN_local++)
	  {
	    int nN = mesh.elementBoundaryNodesArray[ebN*mesh.nNodes_elementBoundary+nN_local];
	    if (mesh.nodeMaterialTypes[nN] == DEFAULT_NODE_MATERIAL)
	      mesh.nodeMaterialTypes[nN] = INTERIOR_NODE_MATERIAL;
	  }
      }
    //cout<<"Elapsed time for populating arrays = "<<(stop-start)<<"s"<<endl;
    return 0;
  }

  int constructElementBoundaryElementsArrayWithGivenElementBoundaryAndEdgeNumbers_tetrahedron(Mesh& mesh)
  {
    // printf("nNodes_global = %d\n", mesh.nNodes_global);
    // printf("nElements_global = %d\n", mesh.nElements_global);
    // printf("nNodes_element = %d\n", mesh.nNodes_element);
    // printf("nElementBoundaries_element = %d\n", mesh.nElementBoundaries_element);
    // printf("nNodes_elementBoundary = %d\n", mesh.nNodes_elementBoundary);
    // printf("nElementBoundaries_global = %d\n", mesh.nElementBoundaries_global);
    
    mesh.nNodes_elementBoundary = 3;
    mesh.nElementBoundaries_element = 4;
    assert(mesh.elementBoundariesArray);
    using namespace std;
    double start,stop;
    map<NodeTuple<3>,
      ElementNeighbors> elementBoundaryElements;
    map<NodeTuple<3>,
      int> elementBoundaryIds;
    start=CurrentTime();
    //cout<<"Extracting boundary elements"<<endl;
    for(int eN=0;eN<mesh.nElements_global;eN++)
      for(int ebN=0;ebN<mesh.nElementBoundaries_element;ebN++)
        {
	  int ebN_global = mesh.elementBoundariesArray[eN*mesh.nElementBoundaries_element+ebN];
          int nodes[3];
          nodes[0] = mesh.elementNodesArray[eN*4+((ebN+1)%4)];
          nodes[1] = mesh.elementNodesArray[eN*4+((ebN+2)%4)];
          nodes[2] = mesh.elementNodesArray[eN*4+((ebN+3)%4)];
          // printf("eN = %d, ebN = %d, ebN_global = %d, nodes = %d %d %d\n", eN, ebN, ebN_global, nodes[0], nodes[1], nodes[2]);

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
    stop = CurrentTime();
    //cout<<"Elapsed time for building element boundary elements map= "<<(stop-start)<<"s"<<endl;
    mesh.nElementBoundaries_global = elementBoundaryElements.size();
    //cout<<"nElementBoundaries_global = "<<mesh.nElementBoundaries_global<<endl;

    //cout<<"Allocating Arrays"<<endl;
    start = CurrentTime();
    set<int> interiorElementBoundaries,exteriorElementBoundaries;
    mesh.elementBoundaryNodesArray =  new int[mesh.nElementBoundaries_global*mesh.nNodes_elementBoundary];
    mesh.elementBoundaryElementsArray = new int[mesh.nElementBoundaries_global*2];
    mesh.elementBoundaryLocalElementBoundariesArray = new int[mesh.nElementBoundaries_global*2];
    mesh.elementNeighborsArray = new int[mesh.nElements_global*mesh.nElementBoundaries_element];
    stop = CurrentTime();
    //cout<<"Elapsed time for allocating arrays = "<<(stop-start)<<"s"<<endl;

    //cout<<"Generating elementBoundaryElementsArray and elementBoundaryNodesArray"<<endl;
    start = CurrentTime();
    for(map<NodeTuple<3>,ElementNeighbors>::iterator eb=elementBoundaryElements.begin();
        eb != elementBoundaryElements.end();
        eb++)
      {
        //  printf("eb->first.nodes = %d %d %d\n", eb->first.nodes[0], eb->first.nodes[1], eb->first.nodes[2]);   
	int ebN = elementBoundaryIds[eb->first];
        mesh.elementBoundaryNodesArray[ebN*3 + 0] = eb->first.nodes[0];
        mesh.elementBoundaryNodesArray[ebN*3 + 1] = eb->first.nodes[1];
        mesh.elementBoundaryNodesArray[ebN*3 + 2] = eb->first.nodes[2];

        mesh.elementBoundaryElementsArray[ebN*2 + 0] = eb->second.left;
        mesh.elementBoundaryLocalElementBoundariesArray[ebN*2 + 0] = eb->second.left_ebN_element;
        mesh.elementBoundaryElementsArray[ebN*2 + 1] = eb->second.right;
        mesh.elementBoundaryLocalElementBoundariesArray[ebN*2 + 1] = eb->second.right_ebN_element;
        mesh.elementNeighborsArray[eb->second.left*mesh.nElementBoundaries_element + eb->second.left_ebN_element] = eb->second.right; 
        if(eb->second.right != -1)
	  {
	    interiorElementBoundaries.insert(ebN);
	    mesh.elementNeighborsArray[eb->second.right*mesh.nElementBoundaries_element + eb->second.right_ebN_element] = eb->second.left; 
	  }
        else
          exteriorElementBoundaries.insert(ebN);          
	assert(mesh.elementBoundariesArray[eb->second.left*mesh.nElementBoundaries_element + eb->second.left_ebN_element] == ebN);
	if (eb->second.right != -1)
	  {
	    assert(mesh.elementBoundariesArray[eb->second.right*mesh.nElementBoundaries_element + eb->second.right_ebN_element] == ebN);
 	  }
      }
    mesh.nInteriorElementBoundaries_global = interiorElementBoundaries.size();
    mesh.interiorElementBoundariesArray = new int[mesh.nInteriorElementBoundaries_global];
    mesh.nExteriorElementBoundaries_global = exteriorElementBoundaries.size();
    mesh.exteriorElementBoundariesArray = new int[mesh.nExteriorElementBoundaries_global];
    int ebNI=0,ebNE=0;
    for (set<int>::iterator ebN=interiorElementBoundaries.begin();ebN != interiorElementBoundaries.end(); ebN++,ebNI++)
      mesh.interiorElementBoundariesArray[ebNI] = *ebN;
    for (set<int>::iterator ebN=exteriorElementBoundaries.begin();ebN != exteriorElementBoundaries.end(); ebN++,ebNE++)
      mesh.exteriorElementBoundariesArray[ebNE] = *ebN;
    assert(mesh.edgeNodesArray);
    //     set<NodeTuple<2> > edges;
    //     //std::cout<<"extracting edges"<<std::endl;
    //     for (int eN=0;eN<mesh.nElements_global;eN++)
    //       {
    //         int nodes[2];
    //         for (int nN_L=0;nN_L<mesh.nNodes_element;nN_L++)
    //           for (int nN_R=nN_L+1;nN_R<mesh.nNodes_element;nN_R++)
    //             {
    //               nodes[0] = mesh.elementNodesArray[eN*4+nN_L];
    //               nodes[1] = mesh.elementNodesArray[eN*4+nN_R];
    //               edges.insert(NodeTuple<2>(nodes));
    //             }
    //       }
    //     mesh.nEdges_global = edges.size();
    //     mesh.edgeNodesArray = new int[mesh.nEdges_global*2];
    //     set<NodeTuple<2> >::iterator edge_p=edges.begin();
    //     for (int edgeN=0;edgeN<int(edges.size());edgeN++)
    //       {
    //         mesh.edgeNodesArray[edgeN*2+0] = edge_p->nodes[0];
    //         mesh.edgeNodesArray[edgeN*2+1] = edge_p->nodes[1];
    //         edge_p++;
    //       }
    vector<set<int> > nodeStar(mesh.nNodes_global);
    for (int edgeN=0;edgeN<mesh.nEdges_global;edgeN++)
      {
        nodeStar[mesh.edgeNodesArray[edgeN*2+0]].insert(mesh.edgeNodesArray[edgeN*2+1]);
        nodeStar[mesh.edgeNodesArray[edgeN*2+1]].insert(mesh.edgeNodesArray[edgeN*2+0]);
      }
    mesh.nodeStarOffsets = new int[mesh.nNodes_global+1];
    mesh.nodeStarOffsets[0] = 0;
    for (int nN=1;nN<mesh.nNodes_global+1;nN++)
      mesh.nodeStarOffsets[nN] = mesh.nodeStarOffsets[nN-1] + nodeStar[nN-1].size();
    mesh.nodeStarArray = new int[mesh.nodeStarOffsets[mesh.nNodes_global]];
    for (int nN=0,offset=0;nN<mesh.nNodes_global;nN++)
      for (set<int>::iterator nN_star=nodeStar[nN].begin();nN_star!=nodeStar[nN].end();nN_star++,offset++)
        mesh.nodeStarArray[offset] = *nN_star;
    mesh.max_nNodeNeighbors_node=0;
    for (int nN=0;nN<mesh.nNodes_global;nN++)
      mesh.max_nNodeNeighbors_node=max(mesh.max_nNodeNeighbors_node,mesh.nodeStarOffsets[nN+1]-mesh.nodeStarOffsets[nN]);
    //mwf repeat for node-->elements arrays
    vector<set<int> > nodeElementsStar(mesh.nNodes_global);
    for (int eN = 0; eN < mesh.nElements_global; eN++)
      {
	for (int nN = 0; nN < mesh.nNodes_element; nN++)
	  nodeElementsStar[mesh.elementNodesArray[eN*mesh.nNodes_element+nN]].insert(eN);
      }
    mesh.nodeElementOffsets = new int[mesh.nNodes_global+1];
    mesh.nodeElementOffsets[0] = 0;
    for (int nN = 0; nN < mesh.nNodes_global; nN++)
      mesh.nodeElementOffsets[nN+1] = mesh.nodeElementOffsets[nN]+nodeElementsStar[nN].size();
    mesh.nodeElementsArray  = new int[mesh.nodeElementOffsets[mesh.nNodes_global]];
    for (int nN=0,offset=0; nN < mesh.nNodes_global; nN++)
      {
	for (set<int>::iterator eN_star = nodeElementsStar[nN].begin(); eN_star != nodeElementsStar[nN].end();
	     eN_star++,offset++)
	  {
	    mesh.nodeElementsArray[offset] = *eN_star;
	  }
      }
    //mwf end node-->elements construction
    mesh.elementBoundaryMaterialTypes = new int[mesh.nElementBoundaries_global];
    //if nodeMaterial is DEFAULT, go ahead and set to interior or exterior
    //depending on which boundary node belongs to. 
    //If node on at least one exterior boundary then it's exterior
    for (int ebNE = 0; ebNE < mesh.nExteriorElementBoundaries_global; ebNE++)
      {
	int ebN = mesh.exteriorElementBoundariesArray[ebNE];
	mesh.elementBoundaryMaterialTypes[ebN] = EXTERIOR_ELEMENT_BOUNDARY_MATERIAL;
	for (int nN_local = 0; nN_local < mesh.nNodes_elementBoundary; nN_local++)
	  {
	    int nN = mesh.elementBoundaryNodesArray[ebN*mesh.nNodes_elementBoundary+nN_local];
	    if (mesh.nodeMaterialTypes[nN] == DEFAULT_NODE_MATERIAL)
	      mesh.nodeMaterialTypes[nN] = EXTERIOR_NODE_MATERIAL;
	  }
      }
    for (int ebNI = 0; ebNI < mesh.nInteriorElementBoundaries_global; ebNI++)
      {
	int ebN = mesh.interiorElementBoundariesArray[ebNI];
	mesh.elementBoundaryMaterialTypes[ebN] = INTERIOR_ELEMENT_BOUNDARY_MATERIAL;
	for (int nN_local = 0; nN_local < mesh.nNodes_elementBoundary; nN_local++)
	  {
	    int nN = mesh.elementBoundaryNodesArray[ebN*mesh.nNodes_elementBoundary+nN_local];
	    if (mesh.nodeMaterialTypes[nN] == DEFAULT_NODE_MATERIAL)
	      mesh.nodeMaterialTypes[nN] = INTERIOR_NODE_MATERIAL;
	  }
      }
    //cout<<"Elapsed time for populating arrays = "<<(stop-start)<<"s"<<endl;
    return 0;
  }

  int constructElementBoundaryElementsArrayWithGivenElementBoundaryAndEdgeNumbers_hexahedron(Mesh& mesh)
  {
    mesh.nNodes_elementBoundary = 4;
    mesh.nElementBoundaries_element = 6;
    assert(mesh.elementBoundariesArray);
    using namespace std;
    double start,stop;
    map<NodeTuple<4>,
      ElementNeighbors> elementBoundaryElements;
    map<NodeTuple<4>,
      int> elementBoundaryIds;
    start=CurrentTime();
    
    int lface[6][4] = {{0,1,2,3},
                       {0,1,5,4},
                       {1,2,6,5},
                       {2,3,7,6},
                       {3,0,4,7},
                       {4,5,6,7}};    
    
    //cout<<"Extracting boundary elements"<<endl;
    for(int eN=0;eN<mesh.nElements_global;eN++)
      for(int ebN=0;ebN<mesh.nElementBoundaries_element;ebN++)
        {
	  int ebN_global = mesh.elementBoundariesArray[eN*mesh.nElementBoundaries_element+ebN];
          int nodes[4];
          nodes[0] = mesh.elementNodesArray[eN*8+lface[ebN][0]];
          nodes[1] = mesh.elementNodesArray[eN*8+lface[ebN][1]];
          nodes[2] = mesh.elementNodesArray[eN*8+lface[ebN][2]];      
          nodes[3] = mesh.elementNodesArray[eN*8+lface[ebN][3]];
          NodeTuple<4> ebt(nodes);
          if(elementBoundaryElements.find(ebt) != elementBoundaryElements.end())
            {
              elementBoundaryElements[ebt].right=eN;
              elementBoundaryElements[ebt].right_ebN_element=ebN;
            }
          else
            {
              elementBoundaryElements.insert(elementBoundaryElements.end(),make_pair(ebt,ElementNeighbors(eN,ebN)));
	      elementBoundaryIds.insert(elementBoundaryIds.end(),make_pair(ebt,ebN_global));
            }
        }
    stop = CurrentTime();
    //cout<<"Elapsed time for building element boundary elements map= "<<(stop-start)<<"s"<<endl;
    mesh.nElementBoundaries_global = elementBoundaryElements.size();
    //cout<<"nElementBoundaries_global = "<<mesh.nElementBoundaries_global<<endl;

    //cout<<"Allocating Arrays"<<endl;
    start = CurrentTime();
    set<int> interiorElementBoundaries,exteriorElementBoundaries;
    mesh.elementBoundaryNodesArray =  new int[mesh.nElementBoundaries_global*mesh.nNodes_elementBoundary];
    mesh.elementBoundaryElementsArray = new int[mesh.nElementBoundaries_global*2];
    mesh.elementBoundaryLocalElementBoundariesArray = new int[mesh.nElementBoundaries_global*2];
    mesh.elementNeighborsArray = new int[mesh.nElements_global*mesh.nElementBoundaries_element];
    stop = CurrentTime();
    //cout<<"Elapsed time for allocating arrays = "<<(stop-start)<<"s"<<endl;

    //cout<<"Generating elementBoundaryElementsArray and elementBoundaryNodesArray"<<endl;
    start = CurrentTime();
    for(map<NodeTuple<4>,ElementNeighbors>::iterator eb=elementBoundaryElements.begin();
        eb != elementBoundaryElements.end();
        eb++)
      {
	int ebN = elementBoundaryIds[eb->first];
        mesh.elementBoundaryNodesArray[ebN*4 + 0] = eb->first.nodes[0];
        mesh.elementBoundaryNodesArray[ebN*4 + 1] = eb->first.nodes[1];
        mesh.elementBoundaryNodesArray[ebN*4 + 2] = eb->first.nodes[2];
        mesh.elementBoundaryNodesArray[ebN*4 + 3] = eb->first.nodes[3];

        mesh.elementBoundaryElementsArray[ebN*2 + 0] = eb->second.left;
        mesh.elementBoundaryLocalElementBoundariesArray[ebN*2 + 0] = eb->second.left_ebN_element;
        mesh.elementBoundaryElementsArray[ebN*2 + 1] = eb->second.right;
        mesh.elementBoundaryLocalElementBoundariesArray[ebN*2 + 1] = eb->second.right_ebN_element;
        mesh.elementNeighborsArray[eb->second.left*mesh.nElementBoundaries_element + eb->second.left_ebN_element] = eb->second.right; 
        if(eb->second.right != -1)
	  {
	    interiorElementBoundaries.insert(ebN);
	    mesh.elementNeighborsArray[eb->second.right*mesh.nElementBoundaries_element + eb->second.right_ebN_element] = eb->second.left; 
	  }
        else
          exteriorElementBoundaries.insert(ebN);          
	assert(mesh.elementBoundariesArray[eb->second.left*mesh.nElementBoundaries_element + eb->second.left_ebN_element] == ebN);
	if (eb->second.right != -1)
	  {
	    assert(mesh.elementBoundariesArray[eb->second.right*mesh.nElementBoundaries_element + eb->second.right_ebN_element] == ebN);
 	  }
      }
    mesh.nInteriorElementBoundaries_global = interiorElementBoundaries.size();
    mesh.interiorElementBoundariesArray = new int[mesh.nInteriorElementBoundaries_global];
    mesh.nExteriorElementBoundaries_global = exteriorElementBoundaries.size();
    mesh.exteriorElementBoundariesArray = new int[mesh.nExteriorElementBoundaries_global];
    int ebNI=0,ebNE=0;
    for (set<int>::iterator ebN=interiorElementBoundaries.begin();ebN != interiorElementBoundaries.end(); ebN++,ebNI++)
      mesh.interiorElementBoundariesArray[ebNI] = *ebN;
    for (set<int>::iterator ebN=exteriorElementBoundaries.begin();ebN != exteriorElementBoundaries.end(); ebN++,ebNE++)
      mesh.exteriorElementBoundariesArray[ebNE] = *ebN;
    assert(mesh.edgeNodesArray);
    //     set<NodeTuple<2> > edges;
    //     //std::cout<<"extracting edges"<<std::endl;
    //     for (int eN=0;eN<mesh.nElements_global;eN++)
    //       {
    //         int nodes[2];
    //         for (int nN_L=0;nN_L<mesh.nNodes_element;nN_L++)
    //           for (int nN_R=nN_L+1;nN_R<mesh.nNodes_element;nN_R++)
    //             {
    //               nodes[0] = mesh.elementNodesArray[eN*4+nN_L];
    //               nodes[1] = mesh.elementNodesArray[eN*4+nN_R];
    //               edges.insert(NodeTuple<2>(nodes));
    //             }
    //       }
    //     mesh.nEdges_global = edges.size();
    //     mesh.edgeNodesArray = new int[mesh.nEdges_global*2];
    //     set<NodeTuple<2> >::iterator edge_p=edges.begin();
    //     for (int edgeN=0;edgeN<int(edges.size());edgeN++)
    //       {
    //         mesh.edgeNodesArray[edgeN*2+0] = edge_p->nodes[0];
    //         mesh.edgeNodesArray[edgeN*2+1] = edge_p->nodes[1];
    //         edge_p++;
    //       }

    vector<set<int> > nodeStar(mesh.nNodes_global);
    for (int edgeN=0;edgeN<mesh.nEdges_global;edgeN++)
      {
        nodeStar[mesh.edgeNodesArray[edgeN*2+0]].insert(mesh.edgeNodesArray[edgeN*2+1]);
        nodeStar[mesh.edgeNodesArray[edgeN*2+1]].insert(mesh.edgeNodesArray[edgeN*2+0]);
      }
    mesh.nodeStarOffsets = new int[mesh.nNodes_global+1];
    mesh.nodeStarOffsets[0] = 0;
    for (int nN=1;nN<mesh.nNodes_global+1;nN++)
      mesh.nodeStarOffsets[nN] = mesh.nodeStarOffsets[nN-1] + nodeStar[nN-1].size();
    mesh.nodeStarArray = new int[mesh.nodeStarOffsets[mesh.nNodes_global]];
    for (int nN=0,offset=0;nN<mesh.nNodes_global;nN++)
      for (set<int>::iterator nN_star=nodeStar[nN].begin();nN_star!=nodeStar[nN].end();nN_star++,offset++)
        mesh.nodeStarArray[offset] = *nN_star;
    stop = CurrentTime();
    mesh.max_nNodeNeighbors_node=0;
    for (int nN=0;nN<mesh.nNodes_global;nN++)
      mesh.max_nNodeNeighbors_node=max(mesh.max_nNodeNeighbors_node,mesh.nodeStarOffsets[nN+1]-mesh.nodeStarOffsets[nN]);
    //mwf repeat for node-->elements arrays
    vector<set<int> > nodeElementsStar(mesh.nNodes_global);
    for (int eN = 0; eN < mesh.nElements_global; eN++)
      {
	for (int nN = 0; nN < mesh.nNodes_element; nN++)
	  nodeElementsStar[mesh.elementNodesArray[eN*mesh.nNodes_element+nN]].insert(eN);
      }
    mesh.nodeElementOffsets = new int[mesh.nNodes_global+1];
    mesh.nodeElementOffsets[0] = 0;
    for (int nN = 0; nN < mesh.nNodes_global; nN++)
      mesh.nodeElementOffsets[nN+1] = mesh.nodeElementOffsets[nN]+nodeElementsStar[nN].size();
    mesh.nodeElementsArray  = new int[mesh.nodeElementOffsets[mesh.nNodes_global]];
    for (int nN=0,offset=0; nN < mesh.nNodes_global; nN++)
      {
	for (set<int>::iterator eN_star = nodeElementsStar[nN].begin(); eN_star != nodeElementsStar[nN].end();
	     eN_star++,offset++)
	  {
	    mesh.nodeElementsArray[offset] = *eN_star;
	  }
      }
    //mwf end node-->elements construction
    mesh.elementBoundaryMaterialTypes = new int[mesh.nElementBoundaries_global];
    //if nodeMaterial is DEFAULT, go ahead and set to interior or exterior
    //depending on which boundary node belongs to. 
    //If node on at least one exterior boundary then it's exterior
    for (int ebNE = 0; ebNE < mesh.nExteriorElementBoundaries_global; ebNE++)
      {
	int ebN = mesh.exteriorElementBoundariesArray[ebNE];
	mesh.elementBoundaryMaterialTypes[ebN] = EXTERIOR_ELEMENT_BOUNDARY_MATERIAL;
	for (int nN_local = 0; nN_local < mesh.nNodes_elementBoundary; nN_local++)
	  {
	    int nN = mesh.elementBoundaryNodesArray[ebN*mesh.nNodes_elementBoundary+nN_local];
	    if (mesh.nodeMaterialTypes[nN] == DEFAULT_NODE_MATERIAL)
	      mesh.nodeMaterialTypes[nN] = EXTERIOR_NODE_MATERIAL;
	  }
      }
    for (int ebNI = 0; ebNI < mesh.nInteriorElementBoundaries_global; ebNI++)
      {
	int ebN = mesh.interiorElementBoundariesArray[ebNI];
	mesh.elementBoundaryMaterialTypes[ebN] = INTERIOR_ELEMENT_BOUNDARY_MATERIAL;
	for (int nN_local = 0; nN_local < mesh.nNodes_elementBoundary; nN_local++)
	  {
	    int nN = mesh.elementBoundaryNodesArray[ebN*mesh.nNodes_elementBoundary+nN_local];
	    if (mesh.nodeMaterialTypes[nN] == DEFAULT_NODE_MATERIAL)
	      mesh.nodeMaterialTypes[nN] = INTERIOR_NODE_MATERIAL;
	  }
      }
    //cout<<"Elapsed time for populating arrays = "<<(stop-start)<<"s"<<endl;
    return 0;
  }


  inline double edgeLength(int nL,int nR, const double* nodeArray)
  {
    double dx,dy,dz;
    dx = nodeArray[nL*3+0] - nodeArray[nR*3+0];
    dy = nodeArray[nL*3+1] - nodeArray[nR*3+1];
    dz = nodeArray[nL*3+2] - nodeArray[nR*3+2];
    return sqrt(dx*dx+dy*dy+dz*dz);
  }

  inline double triangleArea(int n0, int n1, int n2, const double* nodeArray)
  {
    double va[3],vb[3];
    va[0] = nodeArray[n1*3+0] - nodeArray[n0*3+0];
    va[1] = nodeArray[n1*3+1] - nodeArray[n0*3+1];
    va[2] = nodeArray[n1*3+2] - nodeArray[n0*3+2];
    vb[0] = nodeArray[n2*3+0] - nodeArray[n0*3+0];
    vb[1] = nodeArray[n2*3+1] - nodeArray[n0*3+1];
    vb[2] = nodeArray[n2*3+2] - nodeArray[n0*3+2];
    return 0.5*fabs(va[1]*vb[2] - vb[1]*va[2] - (va[0]*vb[2] - vb[0]*va[2]) + (va[0]*vb[1] - va[1]*vb[0]));
  }

  inline double tetrahedronVolume(int n0, int n1, int n2, int n3, const double* nodeArray)
  {
    double t[3][3];
    t[0][0] = nodeArray[n1*3+0] - nodeArray[n0*3+0];
    t[0][1] = nodeArray[n1*3+1] - nodeArray[n0*3+1];
    t[0][2] = nodeArray[n1*3+2] - nodeArray[n0*3+2];

    t[1][0] = nodeArray[n2*3+0] - nodeArray[n0*3+0];
    t[1][1] = nodeArray[n2*3+1] - nodeArray[n0*3+1];
    t[1][2] = nodeArray[n2*3+2] - nodeArray[n0*3+2];

    t[2][0] = nodeArray[n3*3+0] - nodeArray[n0*3+0];
    t[2][1] = nodeArray[n3*3+1] - nodeArray[n0*3+1];
    t[2][2] = nodeArray[n3*3+2] - nodeArray[n0*3+2];
    return fabs(t[0][0]*(t[1][1]*t[2][2] - t[1][2]*t[2][1]) - \
                t[0][1]*(t[1][0]*t[2][2] - t[1][2]*t[2][0]) + \
                t[0][2]*(t[1][0]*t[2][1] - t[1][1]*t[2][0]))/6.0;
  }

  int allocateGeometricInfo_tetrahedron(Mesh& mesh)
  {
    mesh.elementDiametersArray = new double[mesh.nElements_global];
    mesh.elementInnerDiametersArray = new double[mesh.nElements_global];
    mesh.elementBoundaryDiametersArray = new double[mesh.nElementBoundaries_global];
    mesh.elementBarycentersArray  = new double[mesh.nElements_global*3];
    mesh.elementBoundaryBarycentersArray = new double[mesh.nElementBoundaries_global*3];
    mesh.nodeDiametersArray = new double[mesh.nNodes_global];
    mesh.nodeSupportArray = new double[mesh.nNodes_global];
    return 0;
  }

  int computeGeometricInfo_tetrahedron(Mesh& mesh)
  {
    memset(mesh.elementDiametersArray,0,mesh.nElements_global*sizeof(double));
    memset(mesh.elementInnerDiametersArray,0,mesh.nElements_global*sizeof(double));
    memset(mesh.elementBoundaryDiametersArray,0,mesh.nElementBoundaries_global*sizeof(double));
    memset(mesh.elementBarycentersArray,0,mesh.nElements_global*3*sizeof(double));
    memset(mesh.elementBoundaryBarycentersArray,0,mesh.nElementBoundaries_global*3*sizeof(double));
    memset(mesh.nodeDiametersArray,0,mesh.nNodes_global*sizeof(double));
    memset(mesh.nodeSupportArray,0,mesh.nNodes_global*sizeof(double));
    //a subdomain mesh is empty on a rank that owns nothing
    mesh.hMin = mesh.nElements_global > 0 ? edgeLength(mesh.elementNodesArray[0],
                                                       mesh.elementNodesArray[1],
                                                       mesh.nodeArray) : 0.0;
    const double nNperElemInv = 1.0/double(mesh.nNodes_element);
    const double nNperElemBInv= 1.0/double(mesh.nNodes_elementBoundary);
    for (int ebN=0;ebN<mesh.nElementBoundaries_global;ebN++)
      {
	mesh.elementBoundaryBarycentersArray[ebN*3 + 0] = 0.0;
	mesh.elementBoundaryBarycentersArray[ebN*3 + 1] = 0.0;
	mesh.elementBoundaryBarycentersArray[ebN*3 + 2] = 0.0;
        // printf("ebN = %d, nElementBoundaries_global = %d\n",ebN,mesh.nElementBoundaries_global);
        for (int nN_L=0;nN_L<mesh.nNodes_elementBoundary;nN_L++)
          {
            // printf("nN_L = %d, nNodes_elementBoundary = %d \t elementBoundaryNodesArray[%d] %d \n",nN_L,mesh.nNodes_elementBoundary, ebN*mesh.nNodes_elementBoundary+nN_L, mesh.elementBoundaryNodesArray[ebN*mesh.nNodes_elementBoundary+nN_L]);
            // printf("nodeArray = %f \n", mesh.nodeArray[mesh.elementBoundaryNodesArray[ebN*mesh.nNodes_elementBoundary+nN_L]*3 + 0]);
	    mesh.elementBoundaryBarycentersArray[ebN*3 + 0] += 
	      nNperElemBInv*mesh.nodeArray[mesh.elementBoundaryNodesArray[ebN*mesh.nNodes_elementBoundary+nN_L]*3 + 0];
	    mesh.elementBoundaryBarycentersArray[ebN*3 + 1] += 
	      nNperElemBInv*mesh.nodeArray[mesh.elementBoundaryNodesArray[ebN*mesh.nNodes_elementBoundary+nN_L]*3 + 1];
	    mesh.elementBoundaryBarycentersArray[ebN*3 + 2] += 
	      nNperElemBInv*mesh.nodeArray[mesh.elementBoundaryNodesArray[ebN*mesh.nNodes_elementBoundary+nN_L]*3 + 2];

	    for (int nN_R=nN_L+1;nN_R<mesh.nNodes_elementBoundary;nN_R++) {
	      mesh.elementBoundaryDiametersArray[ebN] = fmax(mesh.elementBoundaryDiametersArray[ebN],
							     edgeLength(mesh.elementBoundaryNodesArray[ebN*3+nN_L],
									mesh.elementBoundaryNodesArray[ebN*3+nN_R],
									mesh.nodeArray));
            }
	  }
      } 
    for (int eN=0;eN<mesh.nElements_global;eN++)
      {
        double volume, surfaceArea=0.0, hMin=0.0,hMax=0.0;
        volume = tetrahedronVolume(mesh.elementNodesArray[eN*4+0],
                                   mesh.elementNodesArray[eN*4+1],
                                   mesh.elementNodesArray[eN*4+2],
                                   mesh.elementNodesArray[eN*4+3],
                                   mesh.nodeArray);
        //loop over faces to get surface error
        for (int ebN=0;ebN<4;ebN++)
          {
            surfaceArea += triangleArea(mesh.elementNodesArray[eN*4+(ebN+1)%4],
                                        mesh.elementNodesArray[eN*4+(ebN+2)%4],
                                        mesh.elementNodesArray[eN*4+(ebN+3)%4],
                                        mesh.nodeArray);
          }
        hMin = 6.0*volume/surfaceArea;
        //hMax
        for (int nN_L=0;nN_L<mesh.nNodes_element;nN_L++)
          for (int nN_R=nN_L+1;nN_R<mesh.nNodes_element;nN_R++)
            hMax = fmax(hMax,
                        edgeLength(mesh.elementNodesArray[eN*4+nN_L],
                                   mesh.elementNodesArray[eN*4+nN_R],
                                   mesh.nodeArray));
        mesh.elementInnerDiametersArray[eN] = hMin;
        mesh.elementDiametersArray[eN] = hMax;
        mesh.volume += volume;
        mesh.sigmaMax = fmax(hMax/hMin,mesh.sigmaMax);
        mesh.h = fmax(hMax,mesh.h);
        mesh.hMin = fmin(hMin,mesh.hMin);
	
	mesh.elementBarycentersArray[eN*3 + 0] = 0.0; 
	mesh.elementBarycentersArray[eN*3 + 1] = 0.0; 
	mesh.elementBarycentersArray[eN*3 + 2] = 0.0; 
	for (int nN=0;nN<mesh.nNodes_element;nN++)
	  {
	    mesh.elementBarycentersArray[eN*3 + 0] += 
	      nNperElemInv*mesh.nodeArray[mesh.elementNodesArray[eN*mesh.nNodes_element + nN]*3 + 0];
	    mesh.elementBarycentersArray[eN*3 + 1] += 
	      nNperElemInv*mesh.nodeArray[mesh.elementNodesArray[eN*mesh.nNodes_element + nN]*3 + 1];
	    mesh.elementBarycentersArray[eN*3 + 2] += 
	      nNperElemInv*mesh.nodeArray[mesh.elementNodesArray[eN*mesh.nNodes_element + nN]*3 + 2];
	    mesh.nodeDiametersArray[mesh.elementNodesArray[eN*mesh.nNodes_element + nN]] += hMax*volume;
	    mesh.nodeSupportArray[mesh.elementNodesArray[eN*mesh.nNodes_element + nN]] += volume;
	  }
      }
    for (int nN=0;nN<mesh.nNodes_global;nN++)
      {
	mesh.nodeDiametersArray[nN] /= mesh.nodeSupportArray[nN];
      }
//    printf("volume = %12.5e \n",mesh.volume);
//    printf("h = %12.5e \n",mesh.h);
//    printf("hMin = %12.5e \n",mesh.hMin);
//    printf("sigmaMax = %12.5e \n",mesh.sigmaMax);
    return 0;
  }
  inline double hexahedronVolume(int n0, int n1, int n2, int n3, int n4, int n5, int n6, int n7, const double* nodeArray)
  {
    double t[3];
    t[0] = nodeArray[n0*3+0] - nodeArray[n1*3+0];
    t[1] = nodeArray[n0*3+1] - nodeArray[n3*3+1];
    t[2] = nodeArray[n0*3+2] - nodeArray[n4*3+2];
    
    return fabs(t[0]*t[1]*t[2]);
  }
  int allocateGeometricInfo_hexahedron(Mesh& mesh)
  {
    mesh.elementDiametersArray = new double[mesh.nElements_global];
    mesh.elementInnerDiametersArray = new double[mesh.nElements_global];
    mesh.elementBoundaryDiametersArray = new double[mesh.nElementBoundaries_global];
    mesh.elementBarycentersArray  = new double[mesh.nElements_global*3];
    mesh.elementBoundaryBarycentersArray = new double[mesh.nElementBoundaries_global*3];
    mesh.nodeDiametersArray = new double[mesh.nNodes_global];
    mesh.nodeSupportArray = new double[mesh.nNodes_global];
    return 0;
  }

  int computeGeometricInfo_hexahedron(Mesh& mesh)
  {
    memset(mesh.elementDiametersArray,0,mesh.nElements_global*sizeof(double));
    memset(mesh.elementInnerDiametersArray,0,mesh.nElements_global*sizeof(double));
    memset(mesh.elementBoundaryDiametersArray,0,mesh.nElementBoundaries_global*sizeof(double));
    memset(mesh.nodeDiametersArray,0,mesh.nNodes_global*sizeof(double));
    memset(mesh.nodeSupportArray,0,mesh.nNodes_global*sizeof(double));

    //a subdomain mesh is empty on a rank that owns nothing
    mesh.hMin = mesh.nElements_global > 0 ? edgeLength(mesh.elementNodesArray[0],
                                                       mesh.elementNodesArray[1],
                                                       mesh.nodeArray) : 0.0;
    //const double nNperElemInv = 1.0/double(mesh.nNodes_element);
    //const double nNperElemBInv= 1.0/double(mesh.nNodes_elementBoundary);
    for (int ebN=0;ebN<mesh.nElementBoundaries_global;ebN++)
      {


        for (int nN_L=0;nN_L<mesh.nNodes_elementBoundary;nN_L++)
          {
	    for (int nN_R=nN_L+1;nN_R<mesh.nNodes_elementBoundary;nN_R++)
	      mesh.elementBoundaryDiametersArray[ebN] = fmax(mesh.elementBoundaryDiametersArray[ebN],
							     edgeLength(mesh.elementBoundaryNodesArray[ebN*4+nN_L],
									        mesh.elementBoundaryNodesArray[ebN*4+nN_R],
									        mesh.nodeArray));
	    }
	  	  
      } 
    for (int eN=0;eN<mesh.nElements_global;eN++)
      {
        double volume, hMin=0.0,hMax=0.0;
        volume = hexahedronVolume(mesh.elementNodesArray[eN*8+0],
	                          mesh.elementNodesArray[eN*8+1],
     	                          mesh.elementNodesArray[eN*8+2],
	                          mesh.elementNodesArray[eN*8+3],
				  mesh.elementNodesArray[eN*8+4],
	                          mesh.elementNodesArray[eN*8+5],
     	                          mesh.elementNodesArray[eN*8+6],
	                          mesh.elementNodesArray[eN*8+7],
	                          mesh.nodeArray);
        
        //hMax
        hMax = 0.0;
        hMin = 9e99;
        for (int nN_L=0;nN_L<mesh.nNodes_element;nN_L++)
          for (int nN_R=nN_L+1;nN_R<mesh.nNodes_element;nN_R++)
          {
            hMax = fmax(hMax,
                        edgeLength(mesh.elementNodesArray[eN*8+nN_L],
                                   mesh.elementNodesArray[eN*8+nN_R],
                                   mesh.nodeArray));
            hMin = fmin(hMin,
                        edgeLength(mesh.elementNodesArray[eN*8+nN_L],
                                   mesh.elementNodesArray[eN*8+nN_R],
                                   mesh.nodeArray));
        }
        mesh.elementInnerDiametersArray[eN] = hMin;
        mesh.elementDiametersArray[eN] = hMax;
        mesh.volume += volume;
        mesh.sigmaMax = fmax(hMax/hMin,mesh.sigmaMax);
        mesh.h = fmax(hMax,mesh.h);
        mesh.hMin = fmin(hMin,mesh.hMin);

	for (int nN=0;nN<mesh.nNodes_element;nN++)
	  {
	    mesh.nodeDiametersArray[mesh.elementNodesArray[eN*mesh.nNodes_element + nN]] += hMax*volume;
	    mesh.nodeSupportArray[mesh.elementNodesArray[eN*mesh.nNodes_element + nN]] += volume;
	  }
      }
    for (int nN=0;nN<mesh.nNodes_global;nN++)
      {
	mesh.nodeDiametersArray[nN] /= mesh.nodeSupportArray[nN];
      }

    //printf("volume = %12.5e \n",mesh.volume);
    //printf("h = %12.5e \n",mesh.h);
    //printf("hMin = %12.5e \n",mesh.hMin);
    //printf("sigmaMax = %12.5e \n",mesh.sigmaMax);
    return 0;
  }

  int allocateGeometricInfo_NURBS(Mesh& mesh)
  {
    allocateGeometricInfo_hexahedron(mesh);
    return 0;
  }
   
  int computeGeometricInfo_NURBS(Mesh& mesh)
  {
    computeGeometricInfo_hexahedron(mesh);
    return 0;
  }


  int allocateGeometricInfo_triangle(Mesh& mesh)
  {
    mesh.elementDiametersArray = new double[mesh.nElements_global];
    mesh.elementInnerDiametersArray = new double[mesh.nElements_global];
    mesh.elementBoundaryDiametersArray = new double[mesh.nElementBoundaries_global];
    mesh.elementBarycentersArray  = new double[mesh.nElements_global*3];
    mesh.elementBoundaryBarycentersArray = new double[mesh.nElementBoundaries_global*3];
    mesh.nodeDiametersArray = new double[mesh.nNodes_global];
    mesh.nodeSupportArray = new double[mesh.nNodes_global];
    return 0;
  }

  int computeGeometricInfo_triangle(Mesh& mesh)
  {
    memset(mesh.elementDiametersArray,0,mesh.nElements_global*sizeof(double));
    memset(mesh.elementInnerDiametersArray,0,mesh.nElements_global*sizeof(double));
    memset(mesh.elementBoundaryDiametersArray,0,mesh.nElementBoundaries_global*sizeof(double));
    memset(mesh.elementBarycentersArray,0,mesh.nElements_global*3*sizeof(double));
    memset(mesh.elementBoundaryBarycentersArray,0,mesh.nElementBoundaries_global*3*sizeof(double));
    memset(mesh.nodeDiametersArray,0,mesh.nNodes_global*sizeof(double));
    memset(mesh.nodeSupportArray,0,mesh.nNodes_global*sizeof(double));
    //a subdomain mesh is empty on a rank that owns nothing
    mesh.hMin = mesh.nElements_global > 0 ? edgeLength(mesh.elementNodesArray[0],
                                                       mesh.elementNodesArray[1],
                                                       mesh.nodeArray) : 0.0;
    for (int ebN=0;ebN<mesh.nElementBoundaries_global;ebN++)
      mesh.elementBoundaryDiametersArray[ebN] = edgeLength(mesh.elementBoundaryNodesArray[ebN*2+0],
                                                           mesh.elementBoundaryNodesArray[ebN*2+1],
                                                           mesh.nodeArray);
    const double nNperElemInv = 1.0/double(mesh.nNodes_element);
    const double nNperElemBInv= 1.0/double(mesh.nNodes_elementBoundary);

    for (int eN=0;eN<mesh.nElements_global;eN++)
      {
        double area=0.0, perimeter=0.0,h=0.0, hMin=0.0,hMax=0.0;
        area = triangleArea(mesh.elementNodesArray[eN*3 + 0],
                            mesh.elementNodesArray[eN*3 + 1],
                            mesh.elementNodesArray[eN*3 + 2],
                            mesh.nodeArray);
        //hMax
        for (int nN_L=0;nN_L<mesh.nNodes_element;nN_L++)
          for (int nN_R=nN_L+1;nN_R<mesh.nNodes_element;nN_R++)
            {
              h = edgeLength(mesh.elementNodesArray[eN*3+nN_L],
                             mesh.elementNodesArray[eN*3+nN_R],
                             mesh.nodeArray);
              perimeter += h;
              hMax = fmax(hMax,h);
            } 
        hMin = 4.0*area/perimeter; 
        mesh.elementInnerDiametersArray[eN] = hMin;
        mesh.elementDiametersArray[eN] = hMax;
        mesh.volume += area;
        mesh.sigmaMax = fmax(hMax/hMin,mesh.sigmaMax);
        mesh.h = fmax(hMax,mesh.h);
        mesh.hMin = fmin(hMin,mesh.hMin);

	mesh.elementBarycentersArray[eN*3 + 0] = 0.0; 
	mesh.elementBarycentersArray[eN*3 + 1] = 0.0; 
	mesh.elementBarycentersArray[eN*3 + 2] = 0.0; 
	for (int nN=0;nN<mesh.nNodes_element;nN++)
	  {
	    mesh.elementBarycentersArray[eN*3 + 0] += 
	      nNperElemInv*mesh.nodeArray[mesh.elementNodesArray[eN*mesh.nNodes_element + nN]*3 + 0];
	    mesh.elementBarycentersArray[eN*3 + 1] += 
	      nNperElemInv*mesh.nodeArray[mesh.elementNodesArray[eN*mesh.nNodes_element + nN]*3 + 1];
	    mesh.elementBarycentersArray[eN*3 + 2] += 
	      nNperElemInv*mesh.nodeArray[mesh.elementNodesArray[eN*mesh.nNodes_element + nN]*3 + 2];
	    mesh.nodeDiametersArray[mesh.elementNodesArray[eN*mesh.nNodes_element + nN]] += hMax*area;
	    mesh.nodeSupportArray[mesh.elementNodesArray[eN*mesh.nNodes_element + nN]] += area;
	  }

      }

    for (int ebN=0;ebN<mesh.nElementBoundaries_global;ebN++)
      {
	mesh.elementBoundaryBarycentersArray[ebN*3 + 0] = 0.0;
	mesh.elementBoundaryBarycentersArray[ebN*3 + 1] = 0.0;
	mesh.elementBoundaryBarycentersArray[ebN*3 + 2] = 0.0;

        for (int nN=0;nN<mesh.nNodes_elementBoundary;nN++)
          {
	    mesh.elementBoundaryBarycentersArray[ebN*3 + 0] += 
	      nNperElemBInv*mesh.nodeArray[mesh.elementBoundaryNodesArray[ebN*mesh.nNodes_elementBoundary+nN]*3 + 0];
	    mesh.elementBoundaryBarycentersArray[ebN*3 + 1] += 
	      nNperElemBInv*mesh.nodeArray[mesh.elementBoundaryNodesArray[ebN*mesh.nNodes_elementBoundary+nN]*3 + 1];
	    mesh.elementBoundaryBarycentersArray[ebN*3 + 2] += 
	      nNperElemBInv*mesh.nodeArray[mesh.elementBoundaryNodesArray[ebN*mesh.nNodes_elementBoundary+nN]*3 + 2];
	  }
      }
    for (int nN=0;nN<mesh.nNodes_global;nN++)
      {
	mesh.nodeDiametersArray[nN] /= mesh.nodeSupportArray[nN];
      }
    //printf("volume = %12.5e \n",mesh.volume);
    //printf("h = %12.5e \n",mesh.h);
    //printf("hMin = %12.5e \n",mesh.hMin);
    //printf("sigmaMax = %12.5e \n",mesh.sigmaMax);
    return 0;
  }
  
  int allocateGeometricInfo_quadrilateral(Mesh& mesh)
  {
    mesh.elementDiametersArray = new double[mesh.nElements_global];
    mesh.elementInnerDiametersArray = new double[mesh.nElements_global];
    mesh.elementBoundaryDiametersArray = new double[mesh.nElementBoundaries_global];
    mesh.nodeDiametersArray = new double[mesh.nNodes_global];
    mesh.nodeSupportArray = new double[mesh.nNodes_global];
    return 0;
  }

  int computeGeometricInfo_quadrilateral(Mesh& mesh)
  {
    memset(mesh.elementDiametersArray,0,mesh.nElements_global*sizeof(double));
    memset(mesh.elementInnerDiametersArray,0,mesh.nElements_global*sizeof(double));
    memset(mesh.elementBoundaryDiametersArray,0,mesh.nElementBoundaries_global*sizeof(double));
    memset(mesh.nodeDiametersArray,0,mesh.nNodes_global*sizeof(double));
    memset(mesh.nodeSupportArray,0,mesh.nNodes_global*sizeof(double));
    //a subdomain mesh is empty on a rank that owns nothing
    mesh.hMin = mesh.nElements_global > 0 ? edgeLength(mesh.elementNodesArray[0],
                                                       mesh.elementNodesArray[1],
                                                       mesh.nodeArray) : 0.0;
    for (int ebN=0;ebN<mesh.nElementBoundaries_global;ebN++)
      mesh.elementBoundaryDiametersArray[ebN] = edgeLength(mesh.elementBoundaryNodesArray[ebN*2+0],
                                                           mesh.elementBoundaryNodesArray[ebN*2+1],
                                                           mesh.nodeArray);
    const double nNperElemInv = 1.0/double(mesh.nNodes_element);
    const double nNperElemBInv= 1.0/double(mesh.nNodes_elementBoundary);

    for (int eN=0;eN<mesh.nElements_global;eN++)
      {
        double area=0.0, perimeter=0.0,h=0.0, hMin=1.0e6,hMax=0.0;
        area = triangleArea(mesh.elementNodesArray[eN*4 + 0],
                            mesh.elementNodesArray[eN*4 + 1],
                            mesh.elementNodesArray[eN*4 + 2],
                            mesh.nodeArray) +
          triangleArea(mesh.elementNodesArray[eN*4 + 0],
                       mesh.elementNodesArray[eN*4 + 2],
                       mesh.elementNodesArray[eN*4 + 3],
                       mesh.nodeArray);
          
        //hMax &&  hMin
        for (int nN_L=0;nN_L<mesh.nNodes_element;nN_L++)
          for (int nN_R=nN_L+1;nN_R<mesh.nNodes_element;nN_R++)
            {
              h = edgeLength(mesh.elementNodesArray[eN*4+nN_L],
                             mesh.elementNodesArray[eN*4+nN_R],
                             mesh.nodeArray);
              perimeter += h;
              hMax = fmax(hMax,h);
              hMin = fmin(hMin,h);
            } 
        mesh.elementInnerDiametersArray[eN] = hMin;
        mesh.elementDiametersArray[eN] = hMax;
        mesh.volume += area;
        mesh.sigmaMax = fmax(hMax/hMin,mesh.sigmaMax);
        mesh.h = fmax(hMax,mesh.h);
        mesh.hMin = fmin(hMin,mesh.hMin);
	for (int nN=0;nN<mesh.nNodes_element;nN++)
	  {
	    mesh.nodeDiametersArray[mesh.elementNodesArray[eN*mesh.nNodes_element + nN]] += hMax*area;
	    mesh.nodeSupportArray[mesh.elementNodesArray[eN*mesh.nNodes_element + nN]] += area;
	  }
      }

    for (int nN=0;nN<mesh.nNodes_global;nN++)
      {
	mesh.nodeDiametersArray[nN] /= mesh.nodeSupportArray[nN];
      }
    return 0;
  }
  
  int allocateGeometricInfo_edge(Mesh& mesh)
  {
    mesh.elementDiametersArray = new double[mesh.nElements_global];
    mesh.elementInnerDiametersArray = new double[mesh.nElements_global];
    mesh.elementBoundaryDiametersArray = new double[mesh.nElementBoundaries_global];
    mesh.elementBarycentersArray  = new double[mesh.nElements_global*3];
    mesh.elementBoundaryBarycentersArray = new double[mesh.nElementBoundaries_global*3];
    mesh.nodeDiametersArray = new double[mesh.nNodes_global];
    mesh.nodeSupportArray = new double[mesh.nNodes_global];
    return 0;
  }

  int computeGeometricInfo_edge(Mesh& mesh)
  {
    memset(mesh.elementDiametersArray,0,mesh.nElements_global*sizeof(double));
    memset(mesh.elementInnerDiametersArray,0,mesh.nElements_global*sizeof(double));
    memset(mesh.elementBoundaryDiametersArray,0,mesh.nElementBoundaries_global*sizeof(double));
    memset(mesh.elementBarycentersArray,0,mesh.nElements_global*3*sizeof(double));
    memset(mesh.elementBoundaryBarycentersArray,0,mesh.nElementBoundaries_global*3*sizeof(double));
    memset(mesh.nodeDiametersArray,0,mesh.nNodes_global*sizeof(double));
    memset(mesh.nodeSupportArray,0,mesh.nNodes_global*sizeof(double));
    //a subdomain mesh is empty on a rank that owns nothing
    mesh.hMin = mesh.nElements_global > 0 ? edgeLength(mesh.elementNodesArray[0],
                                                       mesh.elementNodesArray[1],
                                                       mesh.nodeArray) : 0.0;
    const double nNperElemInv = 1.0/double(mesh.nNodes_element);
    const double nNperElemBInv= 1.0/double(mesh.nNodes_elementBoundary);
    for (int eN=0;eN<mesh.nElements_global;eN++)
      {
        mesh.elementDiametersArray[eN] = edgeLength(mesh.elementNodesArray[eN*2+0],
                                                    mesh.elementNodesArray[eN*2+1],
                                                    mesh.nodeArray);
        mesh.elementInnerDiametersArray[eN] = mesh.elementDiametersArray[eN];
        mesh.h = fmax(mesh.h,mesh.elementDiametersArray[eN]);
        mesh.hMin = fmin(mesh.hMin,mesh.elementDiametersArray[eN]);
        mesh.volume += mesh.elementDiametersArray[eN];

	mesh.elementBarycentersArray[eN*3 + 0] = 0.0; 
	mesh.elementBarycentersArray[eN*3 + 1] = 0.0; 
	mesh.elementBarycentersArray[eN*3 + 2] = 0.0; 
	for (int nN=0;nN<mesh.nNodes_element;nN++)
	  {
	    mesh.elementBarycentersArray[eN*3 + 0] += 
	      nNperElemInv*mesh.nodeArray[mesh.elementNodesArray[eN*mesh.nNodes_element + nN]*3 + 0];
	    mesh.elementBarycentersArray[eN*3 + 1] += 
	      nNperElemInv*mesh.nodeArray[mesh.elementNodesArray[eN*mesh.nNodes_element + nN]*3 + 1];
	    mesh.elementBarycentersArray[eN*3 + 2] += 
	      nNperElemInv*mesh.nodeArray[mesh.elementNodesArray[eN*mesh.nNodes_element + nN]*3 + 2];
	    mesh.nodeDiametersArray[mesh.elementNodesArray[eN*mesh.nNodes_element + nN]] += mesh.elementDiametersArray[eN]*mesh.elementDiametersArray[eN];
	    mesh.nodeSupportArray[mesh.elementNodesArray[eN*mesh.nNodes_element + nN]] += mesh.elementDiametersArray[eN];
	  }
      }
    for (int ebN=0;ebN<mesh.nElementBoundaries_global;ebN++)
      {
	mesh.elementBoundaryDiametersArray[ebN] = 1.0;

 	mesh.elementBoundaryBarycentersArray[ebN*3 + 0] = 0.0;
	mesh.elementBoundaryBarycentersArray[ebN*3 + 1] = 0.0;
	mesh.elementBoundaryBarycentersArray[ebN*3 + 2] = 0.0;

        for (int nN=0;nN<mesh.nNodes_elementBoundary;nN++)
          {
	    mesh.elementBoundaryBarycentersArray[ebN*3 + 0] += 
	      nNperElemBInv*mesh.nodeArray[mesh.elementBoundaryNodesArray[ebN*mesh.nNodes_elementBoundary+nN]*3 + 0];
	    mesh.elementBoundaryBarycentersArray[ebN*3 + 1] += 
	      nNperElemBInv*mesh.nodeArray[mesh.elementBoundaryNodesArray[ebN*mesh.nNodes_elementBoundary+nN]*3 + 1];
	    mesh.elementBoundaryBarycentersArray[ebN*3 + 2] += 
	      nNperElemBInv*mesh.nodeArray[mesh.elementBoundaryNodesArray[ebN*mesh.nNodes_elementBoundary+nN]*3 + 2];
	  }
      }    
    for (int nN=0;nN<mesh.nNodes_global;nN++)
      {
	mesh.nodeDiametersArray[nN] /= mesh.nodeSupportArray[nN];
      }
    mesh.sigmaMax = 1.0;
    //printf("volume = %12.5e \n",mesh.volume);
    //printf("h = %12.5e \n",mesh.h);
    //printf("hMin = %12.5e \n",mesh.hMin);
    //printf("sigmaMax = %12.5e \n",mesh.sigmaMax);
    return 0;
  }
  //allocate node and element-node connectivity tables so they can be filled in externally
  int allocateNodeAndElementNodeDataStructures(Mesh& mesh, int nElements_global, int nNodes_global, int nNodes_element)
  {
    assert(!mesh.nodeArray);
    assert(!mesh.elementNodesArray);
    assert(!mesh.elementMaterialTypes);
    assert(!mesh.nodeMaterialTypes);

    mesh.nElements_global = nElements_global;
    mesh.nNodes_global    = nNodes_global;
    mesh.nNodes_element   = nNodes_element;

    mesh.nodeArray         = new double[mesh.nNodes_global*3];
    mesh.nodeMaterialTypes = new int[mesh.nNodes_global];
    mesh.elementNodesArray = new int[mesh.nElements_global*mesh.nNodes_element];
    mesh.elementMaterialTypes = new int[mesh.nElements_global];

    return 0;
  }

  //mwftodo get global refinement to preserve element boundary type   
  int globallyRefineEdgeMesh(const int& nLevels, Mesh& mesh, MultilevelMesh& multilevelMesh, bool averageNewNodeFlags)
  {
    using namespace  std;
    multilevelMesh.nLevels = nLevels;
    multilevelMesh.meshArray = new Mesh[nLevels];
    for(int i=0;i<nLevels;i++)
      initializeMesh(multilevelMesh.meshArray[i]);
    multilevelMesh.elementChildrenArray = new int*[nLevels];
    multilevelMesh.elementChildrenOffsets = new int*[nLevels];
    multilevelMesh.elementParentsArray = new int*[nLevels];
    multilevelMesh.meshArray[0] = mesh; //shallow copy
    for(int i=1;i<nLevels;i++)
      {
        //cout<<"Refinement Level "<<i<<endl;
        set<Node> newNodeSet;
        set<Node>::iterator nodeItr;
        pair<set<Node>::iterator,bool> ret;
        multilevelMesh.meshArray[i].nNodes_element=2;
        //2 children per parent
        multilevelMesh.meshArray[i].nElements_global  = 2*multilevelMesh.meshArray[i-1].nElements_global;
        multilevelMesh.meshArray[i].elementNodesArray = new int[multilevelMesh.meshArray[i].nElements_global*2];
        multilevelMesh.elementChildrenArray[i-1]      = new int[multilevelMesh.meshArray[i-1].nElements_global*2];
        multilevelMesh.elementChildrenOffsets[i-1]  = new int[multilevelMesh.meshArray[i-1].nElements_global+1];
        multilevelMesh.elementParentsArray[i]         = new int[multilevelMesh.meshArray[i].nElements_global];
        multilevelMesh.meshArray[i].elementMaterialTypes = new int[multilevelMesh.meshArray[i].nElements_global];
        int nN_new = multilevelMesh.meshArray[i-1].nNodes_global;
        multilevelMesh.elementChildrenOffsets[i-1][0] = 0;
        for(int eN_parent=0,eN=0;eN_parent<multilevelMesh.meshArray[i-1].nElements_global;eN_parent++)
          {
            multilevelMesh.elementChildrenOffsets[i-1][eN_parent+1] = multilevelMesh.elementChildrenOffsets[i-1][eN_parent]+2;
            multilevelMesh.elementChildrenArray[i-1][2*eN_parent + 0 ] = eN + 0;
            multilevelMesh.elementChildrenArray[i-1][2*eN_parent + 1 ] = eN + 1;
            multilevelMesh.elementParentsArray[i][eN + 0] = eN_parent;
            multilevelMesh.elementParentsArray[i][eN + 1] = eN_parent;
            multilevelMesh.meshArray[i].elementMaterialTypes[eN + 0] = multilevelMesh.meshArray[i-1].elementMaterialTypes[eN_parent];
            multilevelMesh.meshArray[i].elementMaterialTypes[eN + 1] = multilevelMesh.meshArray[i-1].elementMaterialTypes[eN_parent];
            Node midpoints[1];
            midpoint(multilevelMesh.meshArray[i-1].nodeArray + multilevelMesh.meshArray[i-1].elementNodesArray[eN_parent*2 + 0]*3,
                     multilevelMesh.meshArray[i-1].nodeArray + multilevelMesh.meshArray[i-1].elementNodesArray[eN_parent*2 + 1]*3,
                     midpoints[0]);
            midpoints[0].nN = nN_new;
            if (multilevelMesh.meshArray[i-1].nodeMaterialTypes[multilevelMesh.meshArray[i-1].elementNodesArray[eN_parent*2 + 0]] ==
                multilevelMesh.meshArray[i-1].nodeMaterialTypes[multilevelMesh.meshArray[i-1].elementNodesArray[eN_parent*2 + 1]])
              midpoints[0].flag = multilevelMesh.meshArray[i-1].nodeMaterialTypes[multilevelMesh.meshArray[i-1].elementNodesArray[eN_parent*2 + 0]];
            else if (averageNewNodeFlags)
              midpoints[0].flag = 0.5*(multilevelMesh.meshArray[i-1].nodeMaterialTypes[multilevelMesh.meshArray[i-1].elementNodesArray[eN_parent*2 + 0]] +
                                       multilevelMesh.meshArray[i-1].nodeMaterialTypes[multilevelMesh.meshArray[i-1].elementNodesArray[eN_parent*2 + 1]]);
            else
              midpoints[0].flag = DEFAULT_NODE_MATERIAL;

            newNodeSet.insert(midpoints[0]);
            nN_new++;
            //the two new edges
            eN = newEdge(eN,multilevelMesh.meshArray[i].elementNodesArray,
                         multilevelMesh.meshArray[i-1].elementNodesArray[2*eN_parent+0],
                         midpoints[0].nN);
            eN = newEdge(eN,multilevelMesh.meshArray[i].elementNodesArray,
                         midpoints[0].nN,
                         multilevelMesh.meshArray[i-1].elementNodesArray[2*eN_parent+1]);
          }
        multilevelMesh.elementChildrenOffsets[i-1][multilevelMesh.meshArray[i-1].nElements_global] = multilevelMesh.elementChildrenOffsets[i-1][multilevelMesh.meshArray[i-1].nElements_global-1]+2;
        assert(unsigned(nN_new) == (newNodeSet.size()+multilevelMesh.meshArray[i-1].nNodes_global));
        multilevelMesh.meshArray[i].nNodes_global = nN_new;
        multilevelMesh.meshArray[i].nodeArray = new double[multilevelMesh.meshArray[i].nNodes_global*3];
        for(int nN=0;nN<multilevelMesh.meshArray[i-1].nNodes_global;nN++)
          {
            multilevelMesh.meshArray[i].nodeArray[nN*3+0] = multilevelMesh.meshArray[i-1].nodeArray[nN*3+0];
            multilevelMesh.meshArray[i].nodeArray[nN*3+1] = multilevelMesh.meshArray[i-1].nodeArray[nN*3+1];
            multilevelMesh.meshArray[i].nodeArray[nN*3+2] = multilevelMesh.meshArray[i-1].nodeArray[nN*3+2];
          }
        for(nodeItr=newNodeSet.begin();nodeItr!=newNodeSet.end();nodeItr++)
          {
            multilevelMesh.meshArray[i].nodeArray[nodeItr->nN*3+0] = nodeItr->x;
            multilevelMesh.meshArray[i].nodeArray[nodeItr->nN*3+1] = nodeItr->y;
            multilevelMesh.meshArray[i].nodeArray[nodeItr->nN*3+2] = nodeItr->z;
          }
	multilevelMesh.meshArray[i].nodeMaterialTypes = new int[multilevelMesh.meshArray[i].nNodes_global];
	std::copy(multilevelMesh.meshArray[i-1].nodeMaterialTypes,
		  multilevelMesh.meshArray[i-1].nodeMaterialTypes+multilevelMesh.meshArray[i-1].nNodes_global,
		  multilevelMesh.meshArray[i].nodeMaterialTypes);
	//new nodes get default material type, should be set on interior and 
	//boundary in constructElementBoundaryElementsArray_*
	if (multilevelMesh.meshArray[i].nNodes_global > multilevelMesh.meshArray[i-1].nNodes_global)
	  memset(multilevelMesh.meshArray[i].nodeMaterialTypes+multilevelMesh.meshArray[i-1].nNodes_global,
		 DEFAULT_NODE_MATERIAL,
		 (multilevelMesh.meshArray[i].nNodes_global-multilevelMesh.meshArray[i-1].nNodes_global)*sizeof(int));

        for(nodeItr=newNodeSet.begin();nodeItr!=newNodeSet.end();nodeItr++)
          {
            multilevelMesh.meshArray[i].nodeMaterialTypes[nodeItr->nN] = nodeItr->flag;
          }
      }
    return 0;
  }

  int globallyRefineTriangularMesh(const int& nLevels, Mesh& mesh, MultilevelMesh& multilevelMesh, bool averageNewNodeFlags)
  {
    using namespace  std;
    multilevelMesh.nLevels = nLevels;
    multilevelMesh.meshArray = new Mesh[nLevels];
    for(int i=0;i<nLevels;i++)
      initializeMesh(multilevelMesh.meshArray[i]);
    multilevelMesh.elementChildrenArray = new int*[nLevels];
    multilevelMesh.elementChildrenOffsets = new int*[nLevels];
    multilevelMesh.elementParentsArray = new int*[nLevels];
    multilevelMesh.meshArray[0] = mesh; //shallow copy
    for(int i=1;i<nLevels;i++)
      {
        //cout<<"Refinement Level "<<i<<endl;
        set<Node> newNodeSet;
        set<Node>::iterator nodeItr;
        pair<set<Node>::iterator,bool> ret;
        multilevelMesh.meshArray[i].nNodes_element=3;
        //4 children per parent
        multilevelMesh.meshArray[i].nElements_global  = 4*multilevelMesh.meshArray[i-1].nElements_global;
        multilevelMesh.meshArray[i].elementNodesArray = new int[multilevelMesh.meshArray[i].nElements_global*3];
        multilevelMesh.elementChildrenArray[i-1]      = new int[multilevelMesh.meshArray[i-1].nElements_global*4];
        multilevelMesh.elementChildrenOffsets[i-1]  = new int[multilevelMesh.meshArray[i-1].nElements_global+1];
        multilevelMesh.elementParentsArray[i]         = new int[multilevelMesh.meshArray[i].nElements_global];
        multilevelMesh.meshArray[i].elementMaterialTypes = new int[multilevelMesh.meshArray[i].nElements_global];
        multilevelMesh.elementChildrenOffsets[i-1][0] = 0;
        int nN_new = multilevelMesh.meshArray[i-1].nNodes_global;
        for(int eN_parent=0,eN=0;eN_parent<multilevelMesh.meshArray[i-1].nElements_global;eN_parent++)
          {
            multilevelMesh.elementChildrenOffsets[i-1][eN_parent+1] = multilevelMesh.elementChildrenOffsets[i-1][eN_parent] + 4;
            multilevelMesh.elementChildrenArray[i-1][4*eN_parent + 0 ] = eN + 0;
            multilevelMesh.elementChildrenArray[i-1][4*eN_parent + 1 ] = eN + 1;
            multilevelMesh.elementChildrenArray[i-1][4*eN_parent + 2 ] = eN + 2;
            multilevelMesh.elementChildrenArray[i-1][4*eN_parent + 3 ] = eN + 3;
            multilevelMesh.elementParentsArray[i][eN + 0] = eN_parent;
            multilevelMesh.elementParentsArray[i][eN + 1] = eN_parent;
            multilevelMesh.elementParentsArray[i][eN + 2] = eN_parent;
            multilevelMesh.elementParentsArray[i][eN + 3] = eN_parent;
            multilevelMesh.meshArray[i].elementMaterialTypes[eN + 0] = multilevelMesh.meshArray[i-1].elementMaterialTypes[eN_parent];
            multilevelMesh.meshArray[i].elementMaterialTypes[eN + 1] = multilevelMesh.meshArray[i-1].elementMaterialTypes[eN_parent];
            multilevelMesh.meshArray[i].elementMaterialTypes[eN + 2] = multilevelMesh.meshArray[i-1].elementMaterialTypes[eN_parent];
            multilevelMesh.meshArray[i].elementMaterialTypes[eN + 3] = multilevelMesh.meshArray[i-1].elementMaterialTypes[eN_parent];
            Node midpoints[3];
            for(int nN_element_0=0,nN_midpoint=0;nN_element_0<mesh.nNodes_element;nN_element_0++)
              for(int nN_element_1=nN_element_0+1;nN_element_1<mesh.nNodes_element;nN_element_1++,nN_midpoint++)
                {
                  midpoint(multilevelMesh.meshArray[i-1].nodeArray + multilevelMesh.meshArray[i-1].elementNodesArray[eN_parent*3 + nN_element_0]*3,
                           multilevelMesh.meshArray[i-1].nodeArray + multilevelMesh.meshArray[i-1].elementNodesArray[eN_parent*3 + nN_element_1]*3,
                           midpoints[nN_midpoint]);
                  nodeItr = newNodeSet.find(midpoints[nN_midpoint]);
                  if(nodeItr == newNodeSet.end())
                    {
                      midpoints[nN_midpoint].nN = nN_new;
                      if (multilevelMesh.meshArray[i-1].nodeMaterialTypes[multilevelMesh.meshArray[i-1].elementNodesArray[eN_parent*3 + nN_element_0]] ==
                          multilevelMesh.meshArray[i-1].nodeMaterialTypes[multilevelMesh.meshArray[i-1].elementNodesArray[eN_parent*3 + nN_element_1]])
                        midpoints[nN_midpoint].flag = multilevelMesh.meshArray[i-1].nodeMaterialTypes[multilevelMesh.meshArray[i-1].elementNodesArray[eN_parent*3 + nN_element_0]];
                      else if (averageNewNodeFlags)
                        midpoints[nN_midpoint].flag = 0.5*(multilevelMesh.meshArray[i-1].nodeMaterialTypes[multilevelMesh.meshArray[i-1].elementNodesArray[eN_parent*3 + nN_element_0]] +
                                                           multilevelMesh.meshArray[i-1].nodeMaterialTypes[multilevelMesh.meshArray[i-1].elementNodesArray[eN_parent*3 + nN_element_1]]);
                      else
                        midpoints[nN_midpoint].flag = DEFAULT_NODE_MATERIAL;

                      newNodeSet.insert(midpoints[nN_midpoint]);
                      nN_new++;
                    }
                  else
                    midpoints[nN_midpoint].nN = nodeItr->nN;
                }
            //the triangles formed by chopping the points off the parent
            eN = newTriangle(eN,multilevelMesh.meshArray[i].elementNodesArray,
			     multilevelMesh.meshArray[i-1].elementNodesArray[3*eN_parent+0],
			     midpoints[0].nN,
			     midpoints[1].nN);
            eN = newTriangle(eN,multilevelMesh.meshArray[i].elementNodesArray,
			     multilevelMesh.meshArray[i-1].elementNodesArray[3*eN_parent+1],
			     midpoints[0].nN,
			     midpoints[2].nN);
            eN = newTriangle(eN,multilevelMesh.meshArray[i].elementNodesArray,
			     multilevelMesh.meshArray[i-1].elementNodesArray[3*eN_parent+2],
			     midpoints[1].nN,
			     midpoints[2].nN);
            eN = newTriangle(eN,multilevelMesh.meshArray[i].elementNodesArray,
			     midpoints[0].nN,
			     midpoints[1].nN,
			     midpoints[2].nN);
          }
        multilevelMesh.elementChildrenOffsets[i-1][multilevelMesh.meshArray[i-1].nElements_global] = multilevelMesh.elementChildrenOffsets[i-1][multilevelMesh.meshArray[i-1].nElements_global-1]+4;
        assert(unsigned(nN_new) == (newNodeSet.size()+multilevelMesh.meshArray[i-1].nNodes_global));
        multilevelMesh.meshArray[i].nNodes_global = nN_new;
        multilevelMesh.meshArray[i].nodeArray = new double[multilevelMesh.meshArray[i].nNodes_global*3];
        for(int nN=0;nN<multilevelMesh.meshArray[i-1].nNodes_global;nN++)
          {
            multilevelMesh.meshArray[i].nodeArray[nN*3+0] = multilevelMesh.meshArray[i-1].nodeArray[nN*3+0];
            multilevelMesh.meshArray[i].nodeArray[nN*3+1] = multilevelMesh.meshArray[i-1].nodeArray[nN*3+1];
            multilevelMesh.meshArray[i].nodeArray[nN*3+2] = multilevelMesh.meshArray[i-1].nodeArray[nN*3+2];
          }
        for(nodeItr=newNodeSet.begin();nodeItr!=newNodeSet.end();nodeItr++)
          {
            multilevelMesh.meshArray[i].nodeArray[nodeItr->nN*3+0] = nodeItr->x;
            multilevelMesh.meshArray[i].nodeArray[nodeItr->nN*3+1] = nodeItr->y;
            multilevelMesh.meshArray[i].nodeArray[nodeItr->nN*3+2] = nodeItr->z;
          }
	multilevelMesh.meshArray[i].nodeMaterialTypes = new int[multilevelMesh.meshArray[i].nNodes_global];
	std::copy(multilevelMesh.meshArray[i-1].nodeMaterialTypes,
		  multilevelMesh.meshArray[i-1].nodeMaterialTypes+multilevelMesh.meshArray[i-1].nNodes_global,
		  multilevelMesh.meshArray[i].nodeMaterialTypes);
	//new nodes get default material type, should be set on interior and 
	//boundary in constructElementBoundaryElementsArray_*
	if (multilevelMesh.meshArray[i].nNodes_global > multilevelMesh.meshArray[i-1].nNodes_global)
	  memset(multilevelMesh.meshArray[i].nodeMaterialTypes+multilevelMesh.meshArray[i-1].nNodes_global,
		 DEFAULT_NODE_MATERIAL,
		 (multilevelMesh.meshArray[i].nNodes_global-multilevelMesh.meshArray[i-1].nNodes_global)*sizeof(int));
	
        for(nodeItr=newNodeSet.begin();nodeItr!=newNodeSet.end();nodeItr++)
          {
            //mwf hack 
            //std::cout<<"refined triangular mesh: nN parent= "<<multilevelMesh.meshArray[i-1].nNodes_global<<" nN child= "<<multilevelMesh.meshArray[i].nNodes_global
            //         <<" nodeItr->nN,flag,x= "<<nodeItr->nN<<" , "<<nodeItr->flag <<" , " << nodeItr->x <<" , "<< nodeItr->y << " , "<< nodeItr->z << std::endl;
            multilevelMesh.meshArray[i].nodeMaterialTypes[nodeItr->nN] = nodeItr->flag;
          }
	
      }
    return 0;
  }
  
  int globallyRefineQuadrilateralMesh(const int& nLevels, Mesh& mesh, MultilevelMesh& multilevelMesh, bool averageNewNodeFlags)
  {
    using namespace  std;
    multilevelMesh.nLevels = nLevels;
    multilevelMesh.meshArray = new Mesh[nLevels];
    for(int i=0;i<nLevels;i++)
      initializeMesh(multilevelMesh.meshArray[i]);
    multilevelMesh.elementChildrenArray = new int*[nLevels];
    multilevelMesh.elementChildrenOffsets = new int*[nLevels];
    multilevelMesh.elementParentsArray = new int*[nLevels];
    multilevelMesh.meshArray[0] = mesh; //shallow copy
    for(int i=1;i<nLevels;i++)
      {
        std::cout<<"Quad refinement not imlemented "<<i<<endl;
        assert(false);
      }
    return 0;
  }

  int globallyRefineTetrahedralMesh(const int& nLevels, Mesh& mesh, MultilevelMesh& multilevelMesh, bool averageNewNodeFlags)
  {
    using namespace std;
    multilevelMesh.nLevels = nLevels;
    multilevelMesh.meshArray = new Mesh[nLevels];
    for(int i=0;i<nLevels;i++)
      initializeMesh(multilevelMesh.meshArray[i]);
    multilevelMesh.elementChildrenArray = new int*[nLevels];
    multilevelMesh.elementChildrenOffsets = new int*[nLevels];
    multilevelMesh.elementParentsArray = new int*[nLevels];
    multilevelMesh.meshArray[0] = mesh; //shallow copy
    for(int i=1;i<nLevels;i++)
      {
        //cout<<"Refinement Level "<<i<<endl;
        set<Node> newNodeSet;
        set<Node>::iterator nodeItr;
        pair<set<Node>::iterator,bool> ret;
        multilevelMesh.meshArray[i].nNodes_element=4;
        //8 children per parent
        multilevelMesh.meshArray[i].nElements_global  = 8*multilevelMesh.meshArray[i-1].nElements_global;
        multilevelMesh.meshArray[i].elementNodesArray = new int[multilevelMesh.meshArray[i].nElements_global*4];
        multilevelMesh.elementChildrenArray[i-1]      = new int[multilevelMesh.meshArray[i-1].nElements_global*8];
        multilevelMesh.elementChildrenOffsets[i-1]  = new int[multilevelMesh.meshArray[i-1].nElements_global+1];
        multilevelMesh.elementParentsArray[i]         = new int[multilevelMesh.meshArray[i].nElements_global];
        multilevelMesh.meshArray[i].elementMaterialTypes = new int[multilevelMesh.meshArray[i].nElements_global];
        multilevelMesh.elementChildrenOffsets[i-1][0] = 0;
        int nN_new = multilevelMesh.meshArray[i-1].nNodes_global;
        for(int eN_parent=0,eN=0;eN_parent<multilevelMesh.meshArray[i-1].nElements_global;eN_parent++)
          {
            multilevelMesh.elementChildrenOffsets[i-1][eN_parent+1] = multilevelMesh.elementChildrenOffsets[i-1][eN_parent]+8;
            multilevelMesh.elementChildrenArray[i-1][8*eN_parent + 0 ] = eN + 0;
            multilevelMesh.elementChildrenArray[i-1][8*eN_parent + 1 ] = eN + 1;
            multilevelMesh.elementChildrenArray[i-1][8*eN_parent + 2 ] = eN + 2;
            multilevelMesh.elementChildrenArray[i-1][8*eN_parent + 3 ] = eN + 3;
            multilevelMesh.elementChildrenArray[i-1][8*eN_parent + 4 ] = eN + 4;
            multilevelMesh.elementChildrenArray[i-1][8*eN_parent + 5 ] = eN + 5;
            multilevelMesh.elementChildrenArray[i-1][8*eN_parent + 6 ] = eN + 6;
            multilevelMesh.elementChildrenArray[i-1][8*eN_parent + 7 ] = eN + 7; 
            multilevelMesh.elementParentsArray[i][eN + 0] = eN_parent;
            multilevelMesh.elementParentsArray[i][eN + 1] = eN_parent;
            multilevelMesh.elementParentsArray[i][eN + 2] = eN_parent;
            multilevelMesh.elementParentsArray[i][eN + 3] = eN_parent;
            multilevelMesh.elementParentsArray[i][eN + 4] = eN_parent;
            multilevelMesh.elementParentsArray[i][eN + 5] = eN_parent;
            multilevelMesh.elementParentsArray[i][eN + 6] = eN_parent;
            multilevelMesh.elementParentsArray[i][eN + 7] = eN_parent; 
            multilevelMesh.meshArray[i].elementMaterialTypes[eN + 0] = multilevelMesh.meshArray[i-1].elementMaterialTypes[eN_parent];
            multilevelMesh.meshArray[i].elementMaterialTypes[eN + 1] = multilevelMesh.meshArray[i-1].elementMaterialTypes[eN_parent];
            multilevelMesh.meshArray[i].elementMaterialTypes[eN + 2] = multilevelMesh.meshArray[i-1].elementMaterialTypes[eN_parent];
            multilevelMesh.meshArray[i].elementMaterialTypes[eN + 3] = multilevelMesh.meshArray[i-1].elementMaterialTypes[eN_parent];
            multilevelMesh.meshArray[i].elementMaterialTypes[eN + 4] = multilevelMesh.meshArray[i-1].elementMaterialTypes[eN_parent];
            multilevelMesh.meshArray[i].elementMaterialTypes[eN + 5] = multilevelMesh.meshArray[i-1].elementMaterialTypes[eN_parent];
            multilevelMesh.meshArray[i].elementMaterialTypes[eN + 6] = multilevelMesh.meshArray[i-1].elementMaterialTypes[eN_parent];
            multilevelMesh.meshArray[i].elementMaterialTypes[eN + 7] = multilevelMesh.meshArray[i-1].elementMaterialTypes[eN_parent]; 
            
            Node midpoints[6];
            double mind;
            int mindN;
            for(int nN_element_0=0,nN_midpoint=0;nN_element_0<mesh.nNodes_element;nN_element_0++)
              for(int nN_element_1=nN_element_0+1;nN_element_1<mesh.nNodes_element;nN_element_1++,nN_midpoint++)
                {
                  midpoint(multilevelMesh.meshArray[i-1].nodeArray + multilevelMesh.meshArray[i-1].elementNodesArray[eN_parent*4 + nN_element_0]*3,
                           multilevelMesh.meshArray[i-1].nodeArray + multilevelMesh.meshArray[i-1].elementNodesArray[eN_parent*4 + nN_element_1]*3,
                           midpoints[nN_midpoint]);
                  nodeItr = newNodeSet.find(midpoints[nN_midpoint]);
                  if(nodeItr == newNodeSet.end())
                    {
                      midpoints[nN_midpoint].nN = nN_new;
                      if (multilevelMesh.meshArray[i-1].nodeMaterialTypes[multilevelMesh.meshArray[i-1].elementNodesArray[eN_parent*4 + nN_element_0]] ==
                          multilevelMesh.meshArray[i-1].nodeMaterialTypes[multilevelMesh.meshArray[i-1].elementNodesArray[eN_parent*4 + nN_element_1]])
                        midpoints[nN_midpoint].flag = multilevelMesh.meshArray[i-1].nodeMaterialTypes[multilevelMesh.meshArray[i-1].elementNodesArray[eN_parent*4 + nN_element_0]];
                      else if (averageNewNodeFlags)
                        midpoints[nN_midpoint].flag = 0.5*(multilevelMesh.meshArray[i-1].nodeMaterialTypes[multilevelMesh.meshArray[i-1].elementNodesArray[eN_parent*4 + nN_element_0]] +
                                                           multilevelMesh.meshArray[i-1].nodeMaterialTypes[multilevelMesh.meshArray[i-1].elementNodesArray[eN_parent*4 + nN_element_1]]);
                      else
                        midpoints[nN_midpoint].flag = DEFAULT_NODE_MATERIAL;

                      newNodeSet.insert(midpoints[nN_midpoint]);
                      nN_new++;
                    }
                  else
                    midpoints[nN_midpoint].nN = nodeItr->nN;
                }
            //the tets formed by chopping the points of the parent
            eN = newTetrahedron(eN,multilevelMesh.meshArray[i].elementNodesArray,
                                multilevelMesh.meshArray[i-1].elementNodesArray[4*eN_parent+0],
                                midpoints[0].nN,
                                midpoints[1].nN,
                                midpoints[2].nN);
            eN = newTetrahedron(eN,multilevelMesh.meshArray[i].elementNodesArray,
                                multilevelMesh.meshArray[i-1].elementNodesArray[4*eN_parent+1],
                                midpoints[0].nN,
                                midpoints[3].nN,
                                midpoints[4].nN);
            eN = newTetrahedron(eN,multilevelMesh.meshArray[i].elementNodesArray,
                                multilevelMesh.meshArray[i-1].elementNodesArray[4*eN_parent+2],
                                midpoints[1].nN,
                                midpoints[3].nN,
                                midpoints[5].nN);
            eN = newTetrahedron(eN,multilevelMesh.meshArray[i].elementNodesArray,
                                multilevelMesh.meshArray[i-1].elementNodesArray[4*eN_parent+3],
                                midpoints[2].nN,
                                midpoints[4].nN,
                                midpoints[5].nN);
            mind=edgeLength(midpoints[0],midpoints[5]);
            mindN=0;
            double dd;
            for(int dN=1;dN<3;dN++)
              {
                dd = edgeLength(midpoints[dN],midpoints[5-dN]);
                if (dd < mind)
                  {
                    mind = dd;
                    mindN = dN;
                  }
                else if (dd == mind)
                  {
                    if(midpoints[dN] < midpoints[mindN])
                      mindN = dN;
                    else if (!(midpoints[mindN] < midpoints[dN]) && (midpoints[5-dN] < midpoints[5-mindN]))
                      mindN = dN;
                  }
              }
            if(mindN == 0)
              {
                eN = newTetrahedron(eN,multilevelMesh.meshArray[i].elementNodesArray,
                                    midpoints[0].nN, 
                                    midpoints[5].nN, 
                                    midpoints[2].nN, 
                                    midpoints[4].nN);
                eN = newTetrahedron(eN,multilevelMesh.meshArray[i].elementNodesArray,
                                    midpoints[0].nN, 
                                    midpoints[5].nN, 
                                    midpoints[2].nN, 
                                    midpoints[1].nN);
                eN = newTetrahedron(eN,multilevelMesh.meshArray[i].elementNodesArray,
                                    midpoints[0].nN, 
                                    midpoints[5].nN, 
                                    midpoints[1].nN, 
                                    midpoints[3].nN);
                eN = newTetrahedron(eN,multilevelMesh.meshArray[i].elementNodesArray,
                                    midpoints[0].nN, 
                                    midpoints[5].nN, 
                                    midpoints[3].nN, 
                                    midpoints[4].nN);
              }
            else if (mindN == 1)
              {
                eN = newTetrahedron(eN,multilevelMesh.meshArray[i].elementNodesArray,
                                    midpoints[1].nN, 
                                    midpoints[4].nN, 
                                    midpoints[2].nN, 
                                    midpoints[5].nN);
                eN = newTetrahedron(eN,multilevelMesh.meshArray[i].elementNodesArray,
                                    midpoints[1].nN, 
                                    midpoints[4].nN, 
                                    midpoints[5].nN, 
                                    midpoints[3].nN);
                eN = newTetrahedron(eN,multilevelMesh.meshArray[i].elementNodesArray,
                                    midpoints[1].nN, 
                                    midpoints[4].nN, 
                                    midpoints[3].nN, 
                                    midpoints[0].nN);
                eN = newTetrahedron(eN,multilevelMesh.meshArray[i].elementNodesArray,
                                    midpoints[1].nN, 
                                    midpoints[4].nN, 
                                    midpoints[0].nN, 
                                    midpoints[2].nN);
              }
            else
              {
                eN = newTetrahedron(eN,multilevelMesh.meshArray[i].elementNodesArray,
                                    midpoints[2].nN, 
                                    midpoints[3].nN, 
                                    midpoints[0].nN, 
                                    midpoints[4].nN);
                eN = newTetrahedron(eN,multilevelMesh.meshArray[i].elementNodesArray,
                                    midpoints[2].nN, 
                                    midpoints[3].nN, 
                                    midpoints[4].nN, 
                                    midpoints[5].nN);
                eN = newTetrahedron(eN,multilevelMesh.meshArray[i].elementNodesArray,
                                    midpoints[2].nN, 
                                    midpoints[3].nN, 
                                    midpoints[5].nN, 
                                    midpoints[1].nN);
                eN = newTetrahedron(eN,multilevelMesh.meshArray[i].elementNodesArray,
                                    midpoints[2].nN, 
                                    midpoints[3].nN, 
                                    midpoints[1].nN, 
                                    midpoints[0].nN);
              }
          }
        multilevelMesh.elementChildrenOffsets[i-1][multilevelMesh.meshArray[i-1].nElements_global] = multilevelMesh.elementChildrenOffsets[i-1][multilevelMesh.meshArray[i-1].nElements_global-1]+8;
        assert(unsigned(nN_new) == (newNodeSet.size()+multilevelMesh.meshArray[i-1].nNodes_global));
        multilevelMesh.meshArray[i].nNodes_global = nN_new;
        multilevelMesh.meshArray[i].nodeArray = new double[multilevelMesh.meshArray[i].nNodes_global*3];
        for(int nN=0;nN<multilevelMesh.meshArray[i-1].nNodes_global;nN++)
          {
            multilevelMesh.meshArray[i].nodeArray[nN*3+0] = multilevelMesh.meshArray[i-1].nodeArray[nN*3+0];
            multilevelMesh.meshArray[i].nodeArray[nN*3+1] = multilevelMesh.meshArray[i-1].nodeArray[nN*3+1];
            multilevelMesh.meshArray[i].nodeArray[nN*3+2] = multilevelMesh.meshArray[i-1].nodeArray[nN*3+2];
          }
        for(nodeItr=newNodeSet.begin();nodeItr!=newNodeSet.end();nodeItr++)
          {
            multilevelMesh.meshArray[i].nodeArray[nodeItr->nN*3+0] = nodeItr->x;
            multilevelMesh.meshArray[i].nodeArray[nodeItr->nN*3+1] = nodeItr->y;
            multilevelMesh.meshArray[i].nodeArray[nodeItr->nN*3+2] = nodeItr->z;
          }
        /** \todo Add option to re-order mesh nodes on elements to make determinant positive? */
	//         //cout<<"re-ordeing nodes llllllllllllllllllll"<<endl;
	//         for (int eN=0;eN<multilevelMesh.meshArray[i].nElements_global;eN++)
	//           {
	//             int n0,n1,n2,n3;
	//             double t[3][3],det;
            
	//             n0 = multilevelMesh.meshArray[i].elementNodesArray[eN*4+0];
	//             n1 = multilevelMesh.meshArray[i].elementNodesArray[eN*4+1];
	//             n2 = multilevelMesh.meshArray[i].elementNodesArray[eN*4+2];
	//             n3 = multilevelMesh.meshArray[i].elementNodesArray[eN*4+3];
            
	//             t[0][0] = multilevelMesh.meshArray[i].nodeArray[n1*3+0] - multilevelMesh.meshArray[i].nodeArray[n0*3+0];
	//             t[0][1] = multilevelMesh.meshArray[i].nodeArray[n1*3+1] - multilevelMesh.meshArray[i].nodeArray[n0*3+1];
	//             t[0][2] = multilevelMesh.meshArray[i].nodeArray[n1*3+2] - multilevelMesh.meshArray[i].nodeArray[n0*3+2];
            
	//             t[1][0] = multilevelMesh.meshArray[i].nodeArray[n2*3+0] - multilevelMesh.meshArray[i].nodeArray[n0*3+0];
	//             t[1][1] = multilevelMesh.meshArray[i].nodeArray[n2*3+1] - multilevelMesh.meshArray[i].nodeArray[n0*3+1];
	//             t[1][2] = multilevelMesh.meshArray[i].nodeArray[n2*3+2] - multilevelMesh.meshArray[i].nodeArray[n0*3+2];
            
	//             t[2][0] = multilevelMesh.meshArray[i].nodeArray[n3*3+0] - multilevelMesh.meshArray[i].nodeArray[n0*3+0];
	//             t[2][1] = multilevelMesh.meshArray[i].nodeArray[n3*3+1] - multilevelMesh.meshArray[i].nodeArray[n0*3+1];
	//             t[2][2] = multilevelMesh.meshArray[i].nodeArray[n3*3+2] - multilevelMesh.meshArray[i].nodeArray[n0*3+2];
            
	//             det = t[0][0]*(t[1][1]*t[2][2] - t[1][2]*t[2][1]) -   
	//               t[0][1]*(t[1][0]*t[2][2] - t[1][2]*t[2][0]) +       
	//               t[0][2]*(t[1][0]*t[2][1] - t[1][1]*t[2][0]);
            
	//             if(det < 0.0)
	//               {
	//                 multilevelMesh.meshArray[i].elementNodesArray[eN*4+2] = n3;
	//                 multilevelMesh.meshArray[i].elementNodesArray[eN*4+3] = n2;
	//               }
	//             n0 = multilevelMesh.meshArray[i].elementNodesArray[eN*4+0];
	//             n1 = multilevelMesh.meshArray[i].elementNodesArray[eN*4+1];
	//             n2 = multilevelMesh.meshArray[i].elementNodesArray[eN*4+2];
	//             n3 = multilevelMesh.meshArray[i].elementNodesArray[eN*4+3];
            
	//             t[0][0] = multilevelMesh.meshArray[i].nodeArray[n1*3+0] - multilevelMesh.meshArray[i].nodeArray[n0*3+0];
	//             t[0][1] = multilevelMesh.meshArray[i].nodeArray[n1*3+1] - multilevelMesh.meshArray[i].nodeArray[n0*3+1];
	//             t[0][2] = multilevelMesh.meshArray[i].nodeArray[n1*3+2] - multilevelMesh.meshArray[i].nodeArray[n0*3+2];
            
	//             t[1][0] = multilevelMesh.meshArray[i].nodeArray[n2*3+0] - multilevelMesh.meshArray[i].nodeArray[n0*3+0];
	//             t[1][1] = multilevelMesh.meshArray[i].nodeArray[n2*3+1] - multilevelMesh.meshArray[i].nodeArray[n0*3+1];
	//             t[1][2] = multilevelMesh.meshArray[i].nodeArray[n2*3+2] - multilevelMesh.meshArray[i].nodeArray[n0*3+2];
            
	//             t[2][0] = multilevelMesh.meshArray[i].nodeArray[n3*3+0] - multilevelMesh.meshArray[i].nodeArray[n0*3+0];
	//             t[2][1] = multilevelMesh.meshArray[i].nodeArray[n3*3+1] - multilevelMesh.meshArray[i].nodeArray[n0*3+1];
	//             t[2][2] = multilevelMesh.meshArray[i].nodeArray[n3*3+2] - multilevelMesh.meshArray[i].nodeArray[n0*3+2];
            
	//             det = fabs(t[0][0]*(t[1][1]*t[2][2] - t[1][2]*t[2][1]) -   
	//                        t[0][1]*(t[1][0]*t[2][2] - t[1][2]*t[2][0]) +   
	//                        t[0][2]*(t[1][0]*t[2][1] - t[1][1]*t[2][0]));
	//             //cout<<"det "<<det<<endl;
            
	//           }
	//mwftodo need to come up with convention for assigning new node ids
	multilevelMesh.meshArray[i].nodeMaterialTypes = new int[multilevelMesh.meshArray[i].nNodes_global];
	std::copy(multilevelMesh.meshArray[i-1].nodeMaterialTypes,
		  multilevelMesh.meshArray[i-1].nodeMaterialTypes+multilevelMesh.meshArray[i-1].nNodes_global,
		  multilevelMesh.meshArray[i].nodeMaterialTypes);
	//new nodes get default material type, should be set on interior and 
	//boundary in constructElementBoundaryElementsArray_*
	if (multilevelMesh.meshArray[i].nNodes_global > multilevelMesh.meshArray[i-1].nNodes_global)
	  memset(multilevelMesh.meshArray[i].nodeMaterialTypes+multilevelMesh.meshArray[i-1].nNodes_global,
		 DEFAULT_NODE_MATERIAL,
		 (multilevelMesh.meshArray[i].nNodes_global-multilevelMesh.meshArray[i-1].nNodes_global)*sizeof(int));

        for(nodeItr=newNodeSet.begin();nodeItr!=newNodeSet.end();nodeItr++)
          {
            multilevelMesh.meshArray[i].nodeMaterialTypes[nodeItr->nN] = nodeItr->flag;
          }
      }
    return 0;
  }
  
  int globallyRefineHexahedralMesh(const int& nLevels, Mesh& mesh, MultilevelMesh& multilevelMesh, bool averageNewNodeFlags)
  {
    using namespace std;
    multilevelMesh.nLevels = nLevels;
    multilevelMesh.meshArray = new Mesh[nLevels];
    for(int i=0;i<nLevels;i++)
      initializeMesh(multilevelMesh.meshArray[i]);
    multilevelMesh.elementChildrenArray = new int*[nLevels];
    multilevelMesh.elementChildrenOffsets = new int*[nLevels];
    multilevelMesh.elementParentsArray = new int*[nLevels];
    multilevelMesh.meshArray[0] = mesh; //shallow copy
    for(int i=1;i<nLevels;i++)
      {
        std::cout<<"Hexahedron refinement not implemented"<<std::endl;
        assert(false);
      }
    return 0;
  }

  int assignElementBoundaryMaterialTypesFromParent(Mesh& parentMesh, Mesh& childMesh, const int* levelElementParentsArray,
						   const int& nSpace_global)
  {
    int failed = 0;
    for (int ebNI = 0; ebNI < childMesh.nInteriorElementBoundaries_global; ebNI++)
      {
	int ebN = childMesh.interiorElementBoundariesArray[ebNI];
	int eN_left  = childMesh.elementBoundaryElementsArray[ebN*2+0];
	int eN_right = childMesh.elementBoundaryElementsArray[ebN*2+1];
	int eN_left_parent = levelElementParentsArray[eN_left];
	int eN_right_parent= levelElementParentsArray[eN_right];
	if (eN_left_parent == eN_right_parent) //interior to element on coarser level
	  childMesh.elementBoundaryMaterialTypes[ebN] = INTERIOR_ELEMENT_BOUNDARY_MATERIAL;
	else
	  {
	    //find local element boundary id on left parent
	    int left_parent_ebN = -1;
	    for (int ebN_left_parent_element = 0; ebN_left_parent_element < parentMesh.nElementBoundaries_element; 
		 ebN_left_parent_element++)
	      {
		if (parentMesh.elementNeighborsArray[eN_left_parent*parentMesh.nElementBoundaries_element +
						     ebN_left_parent_element] 
		    == eN_right_parent)
		  {
		    left_parent_ebN = ebN_left_parent_element;
		    break;
		  }
	      }
	    assert(0 <= left_parent_ebN && left_parent_ebN < parentMesh.nElementBoundaries_element);
	    int ebN_parent = parentMesh.elementBoundariesArray[eN_left_parent*parentMesh.nElementBoundaries_element + 
							       left_parent_ebN];       
	    childMesh.elementBoundaryMaterialTypes[ebN] = parentMesh.elementBoundaryMaterialTypes[ebN_parent];
	  }
      }//interior element boundaries
    for (int ebNE = 0; ebNE < childMesh.nExteriorElementBoundaries_global; ebNE++)
      {
	int ebN = childMesh.exteriorElementBoundariesArray[ebNE];
	int eN  = childMesh.elementBoundaryElementsArray[ebN*2+0];
	int eN_parent = levelElementParentsArray[eN];
	int lastExteriorElementBoundaryOnParent=-1;
	int nExteriorElementBoundariesOnParent =0;
	for (int ebN_element = 0; ebN_element < parentMesh.nElementBoundaries_element; ebN_element++)
	  {
	    if (parentMesh.elementNeighborsArray[eN_parent*parentMesh.nElementBoundaries_element + ebN_element] < 0)
	      {
		lastExteriorElementBoundaryOnParent = ebN_element;
		nExteriorElementBoundariesOnParent++;
	      }
	  }
	assert(nExteriorElementBoundariesOnParent > 0); assert(lastExteriorElementBoundaryOnParent >= 0);
	if (nExteriorElementBoundariesOnParent > 1)
	  {
	    //more than 1 face of parent is on exterior boundary, have to figure out which one
	    //holds the current face on the child
	    if (nSpace_global == 1)
	      {
		const int ebN_element_child = childMesh.elementBoundaryLocalElementBoundariesArray[ebN*2+0];
		//across from ebN
		const int nN_child0         = childMesh.elementNodesArray[eN*childMesh.nNodes_element+ebN_element_child];
		//on ebN
		const int nN_child1         = childMesh.elementNodesArray[eN*childMesh.nNodes_element+((ebN_element_child+1)%2)];     
	        const double n_child  = (childMesh.nodeArray[nN_child1*3+0]-childMesh.nodeArray[nN_child0*3+0])/
		  fabs(childMesh.nodeArray[nN_child1*3+0]-childMesh.nodeArray[nN_child0*3+0]);

		const int nN_parent0 = parentMesh.elementNodesArray[eN_parent*childMesh.nNodes_element + 0];
		const int nN_parent1 = parentMesh.elementNodesArray[eN_parent*childMesh.nNodes_element + 1];
	        const double n_parent0  = (parentMesh.nodeArray[nN_parent1*3+0]-parentMesh.nodeArray[nN_parent0*3+0])/
		  fabs(parentMesh.nodeArray[nN_parent1*3+0]-parentMesh.nodeArray[nN_parent0*3+0]);
	        const double n_parent1  = (parentMesh.nodeArray[nN_parent0*3+0]-parentMesh.nodeArray[nN_parent1*3+0])/
		  fabs(parentMesh.nodeArray[nN_parent0*3+0]-parentMesh.nodeArray[nN_parent1*3+0]);
		
		int ebN_parent = -1;
		if (fabs(n_child-n_parent1) < 1.0e-8)
		  ebN_parent = parentMesh.elementBoundariesArray[eN_parent*parentMesh.nElementBoundaries_element + 1];
		else if (fabs(n_child-n_parent0) < 1.0e-8)
		  ebN_parent = parentMesh.elementBoundariesArray[eN_parent*parentMesh.nElementBoundaries_element + 0];
		assert(ebN_parent >= 0);
		childMesh.elementBoundaryMaterialTypes[ebN] = parentMesh.elementBoundaryMaterialTypes[ebN_parent];
	      }//1d
	    else if (nSpace_global == 2)
	      {
		//mwftodo assumes relavant coordinates are first two
		const int nN_ebN_child[2] = {childMesh.elementBoundaryNodesArray[2*ebN+0],childMesh.elementBoundaryNodesArray[2*ebN+1]};
		double n_child[2]   = {childMesh.nodeArray[nN_ebN_child[1]*3+1]-childMesh.nodeArray[nN_ebN_child[0]*3+1],
				       childMesh.nodeArray[nN_ebN_child[0]*3+0]-childMesh.nodeArray[nN_ebN_child[1]*3+0]};
		double tmp = sqrt(n_child[0]*n_child[0] + n_child[1]*n_child[1]);
		assert(tmp > 0.0);
		n_child[0] /= tmp; n_child[1] /= tmp;

		int ebN_parent = -1;
		for (int ebN_element_parent = 0; ebN_element_parent < parentMesh.nElementBoundaries_element; ebN_element_parent++)
		  {
		    int ebN_cur = parentMesh.elementBoundariesArray[eN_parent*parentMesh.nElementBoundaries_element + ebN_element_parent];
		    const int nN_ebN_parent[2] = {parentMesh.elementBoundaryNodesArray[2*ebN_cur+0],
						  parentMesh.elementBoundaryNodesArray[2*ebN_cur+1]};
		    
		    double n_parent[2]   = {parentMesh.nodeArray[nN_ebN_parent[1]*3+1]-parentMesh.nodeArray[nN_ebN_parent[0]*3+1],
					    parentMesh.nodeArray[nN_ebN_parent[0]*3+0]-parentMesh.nodeArray[nN_ebN_parent[1]*3+0]};
		    tmp = sqrt(n_parent[0]*n_parent[0] + n_parent[1]*n_parent[1]);
		    assert(tmp > 0.0);
		    n_parent[0] /= tmp; n_parent[1] /= tmp;
		    //looking for face on parent with unit normal in same direction as child face
		    //but don't enforce same outside inside convention on mesh orderings
		    tmp = n_parent[0]*n_child[0] + n_parent[1]*n_child[1];
			       
		    if (fabs(sqrt(tmp*tmp) - 1.0) < 1.0e-8)
		      {
			ebN_parent = ebN_cur;
			break;
		      }
		  }
		assert(ebN_parent >= 0);
		childMesh.elementBoundaryMaterialTypes[ebN] = parentMesh.elementBoundaryMaterialTypes[ebN_parent];
	      }//2d
	    else
	      {
		const double * nN_ebN_child_0_x = childMesh.nodeArray + 3*childMesh.elementBoundaryNodesArray[3*ebN+0];
		const double * nN_ebN_child_1_x = childMesh.nodeArray + 3*childMesh.elementBoundaryNodesArray[3*ebN+1];
		const double * nN_ebN_child_2_x = childMesh.nodeArray + 3*childMesh.elementBoundaryNodesArray[3*ebN+2];
		double n_child[3] = {0.,0.,0.};
		//(nN_1-nN_0) x (nN_2-nN_0)
		n_child[0] = (nN_ebN_child_1_x[1]-nN_ebN_child_0_x[1])*(nN_ebN_child_2_x[2]-nN_ebN_child_0_x[2])
		  -(nN_ebN_child_1_x[2]-nN_ebN_child_0_x[2])*(nN_ebN_child_2_x[1]-nN_ebN_child_0_x[1]);

		n_child[1] = (nN_ebN_child_1_x[2]-nN_ebN_child_0_x[2])*(nN_ebN_child_2_x[0]-nN_ebN_child_0_x[0])
		  -(nN_ebN_child_1_x[0]-nN_ebN_child_0_x[0])*(nN_ebN_child_2_x[2]-nN_ebN_child_0_x[2]);

		n_child[2] = (nN_ebN_child_1_x[0]-nN_ebN_child_0_x[0])*(nN_ebN_child_2_x[1]-nN_ebN_child_0_x[1])
		  -(nN_ebN_child_1_x[1]-nN_ebN_child_0_x[1])*(nN_ebN_child_2_x[0]-nN_ebN_child_0_x[0]);

		double tmp = sqrt(n_child[0]*n_child[0] + n_child[1]*n_child[1] + n_child[2]*n_child[2]);
		assert(tmp > 0.0);
		n_child[0] /= tmp; n_child[1] /= tmp; n_child[2] /= tmp;

		int ebN_parent = -1;
		double n_parent[3] = {0.,0.,0.};
		for (int ebN_element_parent = 0; ebN_element_parent < parentMesh.nElementBoundaries_element; ebN_element_parent++)
		  {
		    int ebN_cur = parentMesh.elementBoundariesArray[eN_parent*parentMesh.nElementBoundaries_element + ebN_element_parent];

		    const double * nN_ebN_parent_0_x = parentMesh.nodeArray + 3*parentMesh.elementBoundaryNodesArray[3*ebN_cur+0];
		    const double * nN_ebN_parent_1_x = parentMesh.nodeArray + 3*parentMesh.elementBoundaryNodesArray[3*ebN_cur+1];
		    const double * nN_ebN_parent_2_x = parentMesh.nodeArray + 3*parentMesh.elementBoundaryNodesArray[3*ebN_cur+2];
		    //(nN_1-nN_0) x (nN_2-nN_0)
		    n_parent[0] = (nN_ebN_parent_1_x[1]-nN_ebN_parent_0_x[1])*(nN_ebN_parent_2_x[2]-nN_ebN_parent_0_x[2])
		      -(nN_ebN_parent_1_x[2]-nN_ebN_parent_0_x[2])*(nN_ebN_parent_2_x[1]-nN_ebN_parent_0_x[1]);
		    
		    n_parent[1] = (nN_ebN_parent_1_x[2]-nN_ebN_parent_0_x[2])*(nN_ebN_parent_2_x[0]-nN_ebN_parent_0_x[0])
		      -(nN_ebN_parent_1_x[0]-nN_ebN_parent_0_x[0])*(nN_ebN_parent_2_x[2]-nN_ebN_parent_0_x[2]);
		    
		    n_parent[2] = (nN_ebN_parent_1_x[0]-nN_ebN_parent_0_x[0])*(nN_ebN_parent_2_x[1]-nN_ebN_parent_0_x[1])
		      -(nN_ebN_parent_1_x[1]-nN_ebN_parent_0_x[1])*(nN_ebN_parent_2_x[0]-nN_ebN_parent_0_x[0]);

		    tmp = sqrt(n_parent[0]*n_parent[0] + n_parent[1]*n_parent[1] + n_parent[2]*n_parent[2]);
		    assert(tmp > 0.0);
		    n_parent[0] /= tmp; n_parent[1] /= tmp; n_parent[2] /= tmp;
		
		    //looking for face on parent with unit normal in same direction as child face
		    //but don't enforce same outside inside convention on mesh orderings
		    tmp = n_parent[0]*n_child[0] + n_parent[1]*n_child[1] + n_parent[2]*n_child[2];
			       
		    if (fabs(sqrt(tmp*tmp) - 1.0) < 1.0e-8)
		      {
			ebN_parent = ebN_cur;
			break;
		      }
		    
		  }
		assert(ebN_parent >= 0);
		childMesh.elementBoundaryMaterialTypes[ebN] = parentMesh.elementBoundaryMaterialTypes[ebN_parent];
	      }//3d
	  }//nExteriorBoundaries > 1
	else 
	  {
	    //only 1 exterior boundary on parent so must be same as child's face
	    int ebN_parent = parentMesh.elementBoundariesArray[eN_parent*parentMesh.nElementBoundaries_element + 
							       lastExteriorElementBoundaryOnParent];
	    childMesh.elementBoundaryMaterialTypes[ebN] = parentMesh.elementBoundaryMaterialTypes[ebN_parent];
	    
	  }

      }//exterior element boundaries
    return failed;
  }
}

int readElements(std::istream& meshFile, Mesh& mesh)
{
  using namespace std;
  assert(meshFile);

  string word,elementType;
  //read in the mesh file. I just read in each token in the order it
  //appeards in the .3dm files. This will break if there is
  //non-whitespace trash in the file.
  meshFile>>word;
  if(word == "MESH1D")
    {
      elementType = "E2E";
      mesh.nNodes_element = 2;
      //cout<<"Reading 1D edge mesh"<<endl;
    }
  else if(word == "MESH2D")
    {
      elementType = "E3T";
      mesh.nNodes_element = 3;
      //cout<<"Reading 2D triangular mesh"<<endl;
    }
  else if (word == "MESH3D")
    {
      elementType = "E4T";
      mesh.nNodes_element = 4;
      //cout<<"Reading 3D tetrahedral mesh"<<endl;
    }
  else
    {
      cerr<<"Unrecognized mesh type"<<endl;
      return 1;
    }
  vector<vector<int> > elementNodesVector;
  vector<int> nodes(mesh.nNodes_element);
  vector<int> materialTypes;
  int type;
  meshFile>>word;
  while(word == elementType)
    {
      meshFile>>word; //discard element numbering
      for(int nN=0;nN<mesh.nNodes_element;nN++)
        {
          meshFile>>nodes[nN];
          nodes[nN]-=1;
        }
      elementNodesVector.push_back(nodes);
      meshFile>>type;
      materialTypes.push_back(type);
      meshFile>>word;
    }
  //cout<<"Number of  elements = "<<elementNodesVector.size()<<endl;
  mesh.nElements_global = elementNodesVector.size();
  mesh.elementNodesArray = new int[mesh.nElements_global*mesh.nNodes_element];
  mesh.elementMaterialTypes = new int[mesh.nElements_global];
  for (int eN=0;eN<mesh.nElements_global;eN++)
    {
      mesh.elementMaterialTypes[eN] = materialTypes[eN];
      for (int nN=0;nN<mesh.nNodes_element;nN++)
        mesh.elementNodesArray[eN*mesh.nNodes_element+nN]=elementNodesVector[eN][nN];
    }
  return 1;
}

int writeNodes(std::ostream& meshFile, const Mesh& mesh)
{
  using namespace std;
  for(int nN=0;nN<mesh.nNodes_global;nN++)
    meshFile<<"ND"<<setw(7)<<nN+1
            <<scientific<<setprecision(8)<<setw(16)<<mesh.nodeArray[3*nN+0]
            <<scientific<<setprecision(8)<<setw(16)<<mesh.nodeArray[3*nN+1]
            <<scientific<<setprecision(8)<<setw(16)<<mesh.nodeArray[3*nN+2]
            <<endl;
  return 0;
}

int writeElements(std::ostream& meshFile, const Mesh& mesh)
{
  //std::cout<<"printing mesh "<<mesh.nElements_global<<"\t"<<mesh.nNodes_element<<std::endl;
  meshFile<<"Try to write something"<<std::endl;
  using namespace std;
  string elementType;
  int width;
  if(mesh.nNodes_element == 2)
    {
      width=7;
      elementType = "E2E";
      meshFile<<"MESH1D"<<endl;
    }
  else if(mesh.nNodes_element == 3)
    {
      width=7;
      elementType = "E3T";
      meshFile<<"MESH2D"<<endl;
    }
  else if (mesh.nNodes_element == 4)
    {
      width=8;
      elementType = "E4T";
      meshFile<<"MESH3D"<<endl;
    }
  else
    {
      cerr<<"Unknown element type"<<endl;
      return 1;
    }
  for (int eN=0;eN<mesh.nElements_global;eN++)
    {
      meshFile<<elementType;
      meshFile<<setw(width)<<eN+1;
      for (int nN=0;nN<mesh.nNodes_element;nN++)
	meshFile<<setw(width)<<(mesh.elementNodesArray[eN*mesh.nNodes_element+nN]+1);
      //mwftodo decide if have convention about material types and base 0
      meshFile<<setw(width)<<mesh.elementMaterialTypes[eN]+1;
      meshFile<<endl;
    }
  return 0;
}

int readTriangleMesh(Mesh& mesh, const char* filebase, int triangleIndexBase)
{
  /***************************************************
    read nodes and element information from triangle
     formatted mesh assuming base name in filebase
    triangle vertex numbering base as input

  **************************************************/
  using namespace std;
  using namespace meshIO;
  assert(filebase);

  int nElements_file, nNodes_file;
  vector<double> nodeArray_file;
  vector<int> elementNodesArray_file;
  vector<int> nodeMaterialTypes_file;
  vector<int> elementMaterialTypes_file;
  bool failed = readTriangleMeshNodesAndElements(filebase,
						 triangleIndexBase,
						 nElements_file,
						 nNodes_file,
						 nodeArray_file,
						 elementNodesArray_file,
						 nodeMaterialTypes_file,
						 elementMaterialTypes_file,
						 DEFAULT_ELEMENT_MATERIAL,
						 DEFAULT_NODE_MATERIAL);
  if (failed)
    {
      //cout<<"readTriangleMesh call failed"<<endl;
      return failed;
    }
  
  mesh.nNodes_element =  3; //2d

  //nodes
  mesh.nNodes_global = nNodes_file;
  assert(!mesh.nodeArray);
  mesh.nodeArray     = new double[mesh.nNodes_global*3];

  copy(nodeArray_file.begin(),nodeArray_file.end(),mesh.nodeArray);
  assert(!mesh.nodeMaterialTypes);
  mesh.nodeMaterialTypes = new int[mesh.nNodes_global];
  copy(nodeMaterialTypes_file.begin(),nodeMaterialTypes_file.end(),
       mesh.nodeMaterialTypes);

  mesh.nElements_global = nElements_file;
  assert(!mesh.elementNodesArray);
  mesh.elementNodesArray = new int[mesh.nElements_global*mesh.nNodes_element];
  assert(!mesh.elementMaterialTypes);
  mesh.elementMaterialTypes = new int[mesh.nElements_global];

  copy(elementNodesArray_file.begin(),elementNodesArray_file.end(),
       mesh.elementNodesArray);
  copy(elementMaterialTypes_file.begin(),elementMaterialTypes_file.end(),
       mesh.elementMaterialTypes);

  return 0;
}

int writeTriangleMesh(Mesh& mesh, const char* filebase, int triangleIndexBase)
{
  /***************************************************
    write nodes and element information in triangle format

  **************************************************/
  using namespace std;
  using namespace meshIO;
  assert(filebase);

  bool failed = writeTriangleMeshNodesAndElements(filebase,
						  triangleIndexBase,
						  mesh.nElements_global,
						  mesh.nNodes_global,
						  mesh.nodeArray,
						  mesh.elementNodesArray,
						  mesh.nodeMaterialTypes,
						  mesh.elementMaterialTypes);
  if (failed)
    return failed;
  //beware, element boundary numbering scheme between triangle and
  //mesh tools may not be the same
  failed = writeTriangleElementBoundaryNodes(filebase,
					     triangleIndexBase,
					     mesh.nElementBoundaries_global,
					     mesh.elementBoundaryNodesArray,
					     mesh.elementBoundaryMaterialTypes);
  return failed;
}

int readTriangleElementBoundaryMaterialTypes(Mesh& mesh, const char* filebase, int triangleIndexBase)
{
  /***************************************************
   read in triangle edge file, in case we need element boundary
    identifiers
   we're not enforcing the same numbering scheme on element boundaries
   in mesh and triangle so have to translate outside 
 
   if material types not found, revert to convention on interior and exterior label
  **************************************************/
  using namespace std;
  using namespace meshIO;
  assert(filebase);

  int nElementBoundaries_file;
  vector<int> elementBoundaryNodesArray_file;
  vector<int> elementBoundaryMaterialTypes_file;
  bool elementBoundaryMaterialTypesInFile = false;
  bool failed = readTriangleElementBoundaries(filebase,
					      triangleIndexBase,
					      elementBoundaryMaterialTypesInFile,
					      nElementBoundaries_file,
					      elementBoundaryNodesArray_file,
					      elementBoundaryMaterialTypes_file,
					      INTERIOR_ELEMENT_BOUNDARY_MATERIAL);
  if (failed)
    {
      //cout<<"readTriangleElementBoundary call failed"<<endl;
      return failed;
    }
  if (!elementBoundaryMaterialTypesInFile)
    return failed;
  assert(mesh.nElementBoundaries_global == nElementBoundaries_file);
  assert(mesh.elementBoundaryMaterialTypes);
  //assume node numberings in triangle and mesh same but not necessarily 
  //element boundaries
  map<NodeTuple<2>,int>  triangleElementBoundaryMaterialTypes;
  //brute force
  for (int ebN = 0; ebN < nElementBoundaries_file; ebN++)
    {
      int nodes[2];
      nodes[0] = elementBoundaryNodesArray_file[ebN*2+0];
      nodes[1] = elementBoundaryNodesArray_file[ebN*2+1];
      NodeTuple<2> ttuple(nodes);
      triangleElementBoundaryMaterialTypes[ttuple] = elementBoundaryMaterialTypes_file[ebN];
    }
  for (int ebN = 0; ebN < mesh.nElementBoundaries_global; ebN++)
    {
      int nodes[2];
      nodes[0] = mesh.elementBoundaryNodesArray[ebN*2+0];
      nodes[1] = mesh.elementBoundaryNodesArray[ebN*2+1];
      NodeTuple<2> ttuple(nodes);
      mesh.elementBoundaryMaterialTypes[ebN] = triangleElementBoundaryMaterialTypes[ttuple];
    }

  return 0;
}

int readTetgenMesh(Mesh& mesh, const char* filebase, int tetgenIndexBase)
{
  /***************************************************
    read nodes and element information from tetgen
     formatted mesh assuming base name in filebase
    tetgen vertex numbering base as input

  **************************************************/
  using namespace std;
  using namespace meshIO;
  assert(filebase);

  int nElements_file, nNodes_file;
  vector<double> nodeArray_file;
  vector<int> elementNodesArray_file;
  vector<int> nodeMaterialTypes_file;
  vector<int> elementMaterialTypes_file;
  bool failed = readTetgenMeshNodesAndElements(filebase,
					       tetgenIndexBase,
					       nElements_file,
					       nNodes_file,
					       nodeArray_file,
					       elementNodesArray_file,
					       nodeMaterialTypes_file,
					       elementMaterialTypes_file,
					       DEFAULT_ELEMENT_MATERIAL,
					       DEFAULT_NODE_MATERIAL);
  if (failed)
    {
      //cout<<"readTetgenMesh call failed"<<endl;
      return failed;
    }
  
  mesh.nNodes_element =  4; //3d

  //nodes
  mesh.nNodes_global = nNodes_file;
  assert(!mesh.nodeArray);
  mesh.nodeArray     = new double[mesh.nNodes_global*3];
  copy(nodeArray_file.begin(),nodeArray_file.end(),mesh.nodeArray);

  assert(!mesh.nodeMaterialTypes);
  mesh.nodeMaterialTypes = new int[mesh.nNodes_global];
  copy(nodeMaterialTypes_file.begin(),nodeMaterialTypes_file.end(),
       mesh.nodeMaterialTypes);
  
  mesh.nElements_global = nElements_file;
  assert(!mesh.elementNodesArray);
  mesh.elementNodesArray = new int[mesh.nElements_global*mesh.nNodes_element];
  assert(!mesh.elementMaterialTypes);
  mesh.elementMaterialTypes = new int[mesh.nElements_global];

  copy(elementNodesArray_file.begin(),elementNodesArray_file.end(),
       mesh.elementNodesArray);
  copy(elementMaterialTypes_file.begin(),elementMaterialTypes_file.end(),
       mesh.elementMaterialTypes);

  return 0;
}
int readTetgenElementBoundaryMaterialTypes(Mesh& mesh, const char* filebase, int tetgenIndexBase)
{
  /***************************************************
   read in tetgen element boundary file, in case we need element boundary
    identifiers
   we're not enforcing the same numbering scheme on element boundaries
   in mesh and triangle so have to translate outside 
 
   if material types not found, revert to convention on interior and exterior label
  **************************************************/
  using namespace std;
  using namespace meshIO;
  assert(filebase);

  int nElementBoundaries_file;
  vector<int> elementBoundaryNodesArray_file;
  vector<int> elementBoundaryMaterialTypes_file;
  bool elementBoundaryMaterialTypesInFile = false;
  bool failed = readTetgenElementBoundaries(filebase,
					    tetgenIndexBase,
					    elementBoundaryMaterialTypesInFile,
					    nElementBoundaries_file,
					    elementBoundaryNodesArray_file,
					    elementBoundaryMaterialTypes_file,
					    INTERIOR_ELEMENT_BOUNDARY_MATERIAL);
  if (failed)
    {
      //cout<<"readTetgenElementBoundary call failed"<<endl;
      return failed;
    }
  if (!elementBoundaryMaterialTypesInFile)
    return failed;
  if (mesh.nElementBoundaries_global == 0)
    {
      mesh.nNodes_elementBoundary = 3;
      mesh.nElementBoundaries_element = 4;
      using namespace std;
      double start,stop;
      {
	map<NodeTuple<3>,
	    ElementNeighbors> elementBoundaryElements;
	start=CurrentTime();
	//cout<<"Extracting boundary elements"<<endl;
	for(int eN=0;eN<mesh.nElements_global;eN++)
	  for(int ebN=0;ebN<mesh.nElementBoundaries_element;ebN++)
	    {
	      int nodes[3];
	      nodes[0] = mesh.elementNodesArray[eN*4+((ebN+1)%4)];
	      nodes[1] = mesh.elementNodesArray[eN*4+((ebN+2)%4)];
	      nodes[2] = mesh.elementNodesArray[eN*4+((ebN+3)%4)];
	      NodeTuple<3> ebt(nodes);
	      if(elementBoundaryElements.find(ebt) != elementBoundaryElements.end())
		{
		  elementBoundaryElements[ebt].right=eN;
		  elementBoundaryElements[ebt].right_ebN_element=ebN;
		}
	      else
		{
		  elementBoundaryElements.insert(elementBoundaryElements.end(),make_pair(ebt,ElementNeighbors(eN,ebN)));
		}
	    }
	stop = CurrentTime();
	//cout<<"Elapsed time for building element boundary elements map= "<<(stop-start)<<"s"<<endl;
	mesh.nElementBoundaries_global = elementBoundaryElements.size();
	//cout<<"nElementBoundaries_global = "<<mesh.nElementBoundaries_global<<endl;
	
	//cout<<"Allocating Arrays"<<endl;
	start = CurrentTime();
	//set<int> interiorElementBoundaries,exteriorElementBoundaries;
	//mesh.elementBoundaryNodesArray =  new int[mesh.nElementBoundaries_global*mesh.nNodes_elementBoundary];
	//mesh.elementBoundaryElementsArray = new int[mesh.nElementBoundaries_global*2];
	//mesh.elementBoundaryLocalElementBoundariesArray = new int[mesh.nElementBoundaries_global*2];
      //mesh.elementNeighborsArray = new int[mesh.nElements_global*mesh.nElementBoundaries_element];
      //mwf added
	mesh.elementBoundariesArray= new int[mesh.nElements_global*mesh.nElementBoundaries_element];
	stop = CurrentTime();
	//cout<<"Elapsed time for allocating arrays = "<<(stop-start)<<"s"<<endl;
	
	//cout<<"Generating elementBoundaryElementsArray and elementBoundaryNodesArray"<<endl;
	start = CurrentTime();
	int ebN=0;
	for(map<NodeTuple<3>,ElementNeighbors>::iterator eb=elementBoundaryElements.begin();
	    eb != elementBoundaryElements.end();
	    eb++,ebN++)
	  {
	    // mesh.elementBoundaryNodesArray[ebN*3 + 0] = eb->first.nodes[0];
	    // mesh.elementBoundaryNodesArray[ebN*3 + 1] = eb->first.nodes[1];
	    // mesh.elementBoundaryNodesArray[ebN*3 + 2] = eb->first.nodes[2];
	    
	    // mesh.elementBoundaryElementsArray[ebN*2 + 0] = eb->second.left;
	    // mesh.elementBoundaryLocalElementBoundariesArray[ebN*2 + 0] = eb->second.left_ebN_element;
	    // mesh.elementBoundaryElementsArray[ebN*2 + 1] = eb->second.right;
	    // mesh.elementBoundaryLocalElementBoundariesArray[ebN*2 + 1] = eb->second.right_ebN_element;
	    // mesh.elementNeighborsArray[eb->second.left*mesh.nElementBoundaries_element + eb->second.left_ebN_element] = eb->second.right; 
	    // if(eb->second.right != -1)
	    //   {
	    //     //interiorElementBoundaries.insert(ebN);
	    //     mesh.elementNeighborsArray[eb->second.right*mesh.nElementBoundaries_element + eb->second.right_ebN_element] = eb->second.left; 
	    //   }
	    //else
	    //exteriorElementBoundaries.insert(ebN);          
	    //mwf added
	    mesh.elementBoundariesArray[eb->second.left*mesh.nElementBoundaries_element + eb->second.left_ebN_element] = ebN;
	    if (eb->second.right != -1)
	      {
		mesh.elementBoundariesArray[eb->second.right*mesh.nElementBoundaries_element + eb->second.right_ebN_element] = ebN;
	      }
	  }
       
    }
      //mesh.nInteriorElementBoundaries_global = interiorElementBoundaries.size();
      //mesh.interiorElementBoundariesArray = new int[mesh.nInteriorElementBoundaries_global];
      //mesh.nExteriorElementBoundaries_global = exteriorElementBoundaries.size();
      //mesh.exteriorElementBoundariesArray = new int[mesh.nExteriorElementBoundaries_global];
      //int ebNI=0,ebNE=0;
      //for (set<int>::iterator ebN=interiorElementBoundaries.begin();ebN != interiorElementBoundaries.end(); ebN++,ebNI++)
      //  mesh.interiorElementBoundariesArray[ebNI] = *ebN;
      //for (set<int>::iterator ebN=exteriorElementBoundaries.begin();ebN != exteriorElementBoundaries.end(); ebN++,ebNE++)
      //  mesh.exteriorElementBoundariesArray[ebNE] = *ebN;
      {
	set<NodeTuple<2> > edges;
	//std::cout<<"extracting edges"<<std::endl;
	for (int eN=0;eN<mesh.nElements_global;eN++)
	  {
	    int nodes[2];
	    for (int nN_L=0;nN_L<mesh.nNodes_element;nN_L++)
	      for (int nN_R=nN_L+1;nN_R<mesh.nNodes_element;nN_R++)
		{
		  nodes[0] = mesh.elementNodesArray[eN*4+nN_L];
		  nodes[1] = mesh.elementNodesArray[eN*4+nN_R];
		  edges.insert(NodeTuple<2>(nodes));
		}
	  }
	mesh.nEdges_global = edges.size();
	mesh.edgeNodesArray = new int[mesh.nEdges_global*2];
	set<NodeTuple<2> >::iterator edge_p=edges.begin();
	for (int edgeN=0;edgeN<int(edges.size());edgeN++)
	  {
	    mesh.edgeNodesArray[edgeN*2+0] = edge_p->nodes[0];
	    mesh.edgeNodesArray[edgeN*2+1] = edge_p->nodes[1];
	    edge_p++;
	  }
      }
      vector<set<int> > nodeStar(mesh.nNodes_global);
      for (int edgeN=0;edgeN<mesh.nEdges_global;edgeN++)
	{
	  nodeStar[mesh.edgeNodesArray[edgeN*2+0]].insert(mesh.edgeNodesArray[edgeN*2+1]);
	  nodeStar[mesh.edgeNodesArray[edgeN*2+1]].insert(mesh.edgeNodesArray[edgeN*2+0]);
	}
      mesh.nodeStarOffsets = new int[mesh.nNodes_global+1];
      mesh.nodeStarOffsets[0] = 0;
      for (int nN=1;nN<mesh.nNodes_global+1;nN++)
	mesh.nodeStarOffsets[nN] = mesh.nodeStarOffsets[nN-1] + nodeStar[nN-1].size();
      mesh.nodeStarArray = new int[mesh.nodeStarOffsets[mesh.nNodes_global]];
      for (int nN=0,offset=0;nN<mesh.nNodes_global;nN++)
	for (set<int>::iterator nN_star=nodeStar[nN].begin();nN_star!=nodeStar[nN].end();nN_star++,offset++)
	  mesh.nodeStarArray[offset] = *nN_star;
      stop = CurrentTime();
      mesh.max_nNodeNeighbors_node=0;
      for (int nN=0;nN<mesh.nNodes_global;nN++)
	mesh.max_nNodeNeighbors_node=max(mesh.max_nNodeNeighbors_node,mesh.nodeStarOffsets[nN+1]-mesh.nodeStarOffsets[nN]);
      //mwf repeat for node-->elements arrays
      {
	vector<set<int> > nodeElementsStar(mesh.nNodes_global);
	for (int eN = 0; eN < mesh.nElements_global; eN++)
	  {
	    for (int nN = 0; nN < mesh.nNodes_element; nN++)
	      nodeElementsStar[mesh.elementNodesArray[eN*mesh.nNodes_element+nN]].insert(eN);
	  }
	mesh.nodeElementOffsets = new int[mesh.nNodes_global+1];
	mesh.nodeElementOffsets[0] = 0;
	for (int nN = 0; nN < mesh.nNodes_global; nN++)
	  mesh.nodeElementOffsets[nN+1] = mesh.nodeElementOffsets[nN]+nodeElementsStar[nN].size();
	mesh.nodeElementsArray  = new int[mesh.nodeElementOffsets[mesh.nNodes_global]];
	for (int nN=0,offset=0; nN < mesh.nNodes_global; nN++)
	  {
	    for (set<int>::iterator eN_star = nodeElementsStar[nN].begin(); eN_star != nodeElementsStar[nN].end();
		 eN_star++,offset++)
	      {
		mesh.nodeElementsArray[offset] = *eN_star;
	      }
	  }
      }
      //mwf end node-->elements construction
      mesh.elementBoundaryMaterialTypes = new int[mesh.nElementBoundaries_global];
      //if nodeMaterial is DEFAULT, go ahead and set to interior or exterior
      //depending on which boundary node belongs to. 
      //If node on at least one exterior boundary then it's exterior
      for (int ebNE = 0; ebNE < mesh.nExteriorElementBoundaries_global; ebNE++)
      {
	int ebN = mesh.exteriorElementBoundariesArray[ebNE];
	mesh.elementBoundaryMaterialTypes[ebN] = EXTERIOR_ELEMENT_BOUNDARY_MATERIAL;
	for (int nN_local = 0; nN_local < mesh.nNodes_elementBoundary; nN_local++)
	  {
	    int nN = mesh.elementBoundaryNodesArray[ebN*mesh.nNodes_elementBoundary+nN_local];
	    if (mesh.nodeMaterialTypes[nN] == DEFAULT_NODE_MATERIAL)
	      mesh.nodeMaterialTypes[nN] = EXTERIOR_NODE_MATERIAL;
	  }
      }
      for (int ebNI = 0; ebNI < mesh.nInteriorElementBoundaries_global; ebNI++)
      {
	int ebN = mesh.interiorElementBoundariesArray[ebNI];
	mesh.elementBoundaryMaterialTypes[ebN] = INTERIOR_ELEMENT_BOUNDARY_MATERIAL;
	for (int nN_local = 0; nN_local < mesh.nNodes_elementBoundary; nN_local++)
	  {
	    int nN = mesh.elementBoundaryNodesArray[ebN*mesh.nNodes_elementBoundary+nN_local];
	    if (mesh.nodeMaterialTypes[nN] == DEFAULT_NODE_MATERIAL)
	      mesh.nodeMaterialTypes[nN] = INTERIOR_NODE_MATERIAL;
	  }
      }
      //cout<<"Elapsed time for populating arrays = "<<(stop-start)<<"s"<<endl;
    }
  //mwf debug
  // std::cout<<"readTetgenElementBoundaryMaterialTypes filebase= "<<filebase<<" after read failed= "<<failed
  // 	       <<" nElementBoundaries_file= "<<nElementBoundaries_file<<" mesh.nElementBoundaries_global= "<<mesh.nElementBoundaries_global 
  // 	       <<" mesh.nExteriorElementBoundaries_global= "<<mesh.nExteriorElementBoundaries_global<<std::endl;
  assert(mesh.nElementBoundaries_global == nElementBoundaries_file ||
	 mesh.nExteriorElementBoundaries_global <= nElementBoundaries_file);
  
  assert(mesh.elementBoundaryMaterialTypes);
  //assume node numberings in triangle and mesh same but not necessarily 
  //	element boundaries
  if (mesh.nElementBoundaries_global == nElementBoundaries_file)
    {
      map<NodeTuple<3>,int>  tetgenElementBoundaryMaterialTypes;
      //brute force
      for (int ebN = 0; ebN < nElementBoundaries_file; ebN++)
	{
	  int nodes[3];
	  nodes[0] = elementBoundaryNodesArray_file[ebN*3+0];
	  nodes[1] = elementBoundaryNodesArray_file[ebN*3+1];
	  nodes[2] = elementBoundaryNodesArray_file[ebN*3+2];
	  NodeTuple<3> ttuple(nodes);
	  tetgenElementBoundaryMaterialTypes[ttuple] = elementBoundaryMaterialTypes_file[ebN];
	}
      for (int ebN = 0; ebN < mesh.nElementBoundaries_global; ebN++)
	{
	  int nodes[3];
	  nodes[0] = mesh.elementBoundaryNodesArray[ebN*3+0];
	  nodes[1] = mesh.elementBoundaryNodesArray[ebN*3+1];
	  nodes[2] = mesh.elementBoundaryNodesArray[ebN*3+2];
	  NodeTuple<3> ttuple(nodes);
	  mesh.elementBoundaryMaterialTypes[ebN] = tetgenElementBoundaryMaterialTypes[ttuple];
	}
    }
  else if (mesh.nExteriorElementBoundaries_global <= nElementBoundaries_file) //just read exterior boundaries
    {
      memset(mesh.elementBoundaryMaterialTypes,INTERIOR_ELEMENT_BOUNDARY_MATERIAL,
	     mesh.nElementBoundaries_global*sizeof(int));

      map<NodeTuple<3>,int>  tetgenElementBoundaryMaterialTypes;
      //brute force
      for (int ebNE = 0; ebNE < nElementBoundaries_file; ebNE++)
	{
	  int nodes[3];
	  nodes[0] = elementBoundaryNodesArray_file[ebNE*3+0];
	  nodes[1] = elementBoundaryNodesArray_file[ebNE*3+1];
	  nodes[2] = elementBoundaryNodesArray_file[ebNE*3+2];
	  NodeTuple<3> ttuple(nodes);
	  tetgenElementBoundaryMaterialTypes[ttuple] = elementBoundaryMaterialTypes_file[ebNE];
	}
      for (int ebNE = 0; ebNE < mesh.nExteriorElementBoundaries_global; ebNE++)
	{
	  int ebN = mesh.exteriorElementBoundariesArray[ebNE];
	  int nodes[3];
	  nodes[0] = mesh.elementBoundaryNodesArray[ebN*3+0];
	  nodes[1] = mesh.elementBoundaryNodesArray[ebN*3+1];
	  nodes[2] = mesh.elementBoundaryNodesArray[ebN*3+2];
	  NodeTuple<3> ttuple(nodes);
	  mesh.elementBoundaryMaterialTypes[ebN] = tetgenElementBoundaryMaterialTypes[ttuple];
	}
    }
  else
    {
      printf("Assertion failed in readTetgenElementBoundaryMaterialTypes\n");
      assert(0);//shouldnt be here
    }
  // std::cout << "elementBoundaryNodesArray:" << std::endl;
  // for (int ebN = 0; ebN < mesh.nElementBoundaries_global; ++ebN) {
  //     std::cout << "Boundary " << ebN << ":";
  //     for (int nN = 0; nN < mesh.nNodes_elementBoundary; ++nN) {
  //         std::cout << " " << mesh.elementBoundaryNodesArray[ebN * mesh.nNodes_elementBoundary + nN];
  //     }
  //     std::cout << std::endl;
  // }
  return 0;
}

int writeTetgenMesh(Mesh& mesh, const char* filebase, int tetgenIndexBase)
{
  /***************************************************
    write nodes and element information in triangle format

  **************************************************/
  using namespace std;
  using namespace meshIO;
  assert(filebase);

  bool failed = writeTetgenMeshNodesAndElements(filebase,
						tetgenIndexBase,
						mesh.nElements_global,
						mesh.nNodes_global,
						mesh.nodeArray,
						mesh.elementNodesArray,
						mesh.nodeMaterialTypes,
						mesh.elementMaterialTypes);
  if (failed)
    return failed;
  //beware, element boundary numbering scheme between tetgen and
  //mesh tools may not be the same
  //by default write just exterior element boundaries ...
  bool writeExteriorElementBoundariesOnly = true;
  int nElementBoundariesToWrite = mesh.nExteriorElementBoundaries_global;
  failed = writeTetgenElementBoundaryNodes(filebase,
					   tetgenIndexBase,
					   nElementBoundariesToWrite,
					   mesh.elementBoundaryNodesArray,
					   mesh.elementBoundaryMaterialTypes,
					   writeExteriorElementBoundariesOnly,
					   mesh.exteriorElementBoundariesArray);
  return failed;
}

int readHex(Mesh& mesh, const char* filebase, int indexBase)
{
  /***************************************************
    read nodes and element information from 
     formatted mesh assuming base name in filebase
     vertex numbering base as input

  **************************************************/
  using namespace std;
  assert(filebase);
  bool failed=true;
  std::string meshFilename= std::string(filebase)+".mesh";
  std::ifstream meshFile(meshFilename.c_str());

  //std::cout<<"Reading hex mesh: "<<meshFilename<<std::endl;

  if (!meshFile.good())
    {
      std::cerr<<"readHex cannot open file "
	       <<meshFilename<<std::endl;
      failed = true;
      return failed;
    }
  //read element type
  std::string fileType;
  meshFile>>fileType;
  if (fileType != "HEX")
    {
      std::cerr<<"readHex does not recognize filetype "
	       <<fileType<<std::endl;
      failed = true;
      return failed;
    }
  
  //read mesh size 
  meshFile>>mesh.nNodes_global>>mesh.nElements_global; 

  //read nodes
  mesh.nodeArray         = new double[mesh.nNodes_global*3];
  mesh.nodeMaterialTypes = new int   [mesh.nNodes_global];
  for (int nN=0;nN<mesh.nNodes_global;nN++) 
    {
        meshFile>>mesh.nodeArray[nN*3+0]>>
                  mesh.nodeArray[nN*3+1]>>
                  mesh.nodeArray[nN*3+2];
        
	mesh.nodeMaterialTypes[nN] = 0;    
    }

  //read elements 
  mesh.nNodes_element = 8;
  mesh.elementNodesArray    = new int[mesh.nElements_global*mesh.nNodes_element];
  mesh.elementMaterialTypes = new int[mesh.nElements_global];

  int n0,n1,n2,n3,n4,n5,n6,n7,emt;
  for (int eN=0;eN<mesh.nElements_global;eN++) 
    {
       int eNne = eN*mesh.nNodes_element;
       meshFile>>n0>>n1>>n2>>n3>>n4>>n5>>n6>>n7>>emt;

       mesh.elementNodesArray[eNne+0] = n0-indexBase;
       mesh.elementNodesArray[eNne+1] = n1-indexBase;
       mesh.elementNodesArray[eNne+2] = n2-indexBase;
       mesh.elementNodesArray[eNne+3] = n3-indexBase;
       mesh.elementNodesArray[eNne+4] = n4-indexBase;
       mesh.elementNodesArray[eNne+5] = n5-indexBase;
       mesh.elementNodesArray[eNne+6] = n6-indexBase;
       mesh.elementNodesArray[eNne+7] = n7-indexBase;

       mesh.elementMaterialTypes[eN] = emt-indexBase;
    }
 
  return 0;
}

int write3dmMesh(Mesh& mesh, const char* filebase, int adhIndexBase)
{
  /***************************************************
    write nodes and element information in 3dm format

  **************************************************/
  using namespace std;
  using namespace meshIO;
  assert(filebase);

  bool failed = write3dmMeshNodesAndElements(filebase,
					     adhIndexBase,
					     mesh.nElements_global,
					     mesh.nNodes_global,
					     mesh.nodeArray,
					     mesh.elementNodesArray,
					     mesh.elementMaterialTypes);
					     
  return failed;
}
int write2dmMesh(Mesh& mesh, const char* filebase, int adhIndexBase)
{
  /***************************************************
    write nodes and element information in 2dm format

  **************************************************/
  using namespace std;
  using namespace meshIO;
  assert(filebase);

  bool failed = write2dmMeshNodesAndElements(filebase,
					     adhIndexBase,
					     mesh.nElements_global,
					     mesh.nNodes_global,
					     mesh.nodeArray,
					     mesh.elementNodesArray,
					     mesh.elementMaterialTypes);
					     
  return failed;
}

extern "C"
{
  int growMultilevelMesh(int nLevels2add, MultilevelMesh& multilevelMesh)
  {
    using namespace  std;
    //first create new meshArray, elementChildren, elementChildrenOffset, elementParentArray's
    //and make shallow copies
    int nLevelsNew = multilevelMesh.nLevels+nLevels2add;
    Mesh * meshArrayTmp                 = new Mesh[nLevelsNew];
    int** elementChildrenArrayTmp   = new int*[nLevelsNew];
    int** elementChildrenOffsetsTmp = new int*[nLevelsNew];
    int** elementParentsArrayTmp    = new int*[nLevelsNew];
    
    //hope shallow copies are allright
    for (int i=0; i < multilevelMesh.nLevels; i++)
      {
	meshArrayTmp[i]               = multilevelMesh.meshArray[i];
	elementChildrenArrayTmp[i]    = multilevelMesh.elementChildrenArray[i];
	elementChildrenOffsetsTmp[i]  = multilevelMesh.elementChildrenOffsets[i];
	elementParentsArrayTmp[i]     = multilevelMesh.elementParentsArray[i];
      }
    delete [] multilevelMesh.meshArray;    
    delete [] multilevelMesh.elementChildrenArray;
    delete [] multilevelMesh.elementChildrenOffsets;
    delete [] multilevelMesh.elementParentsArray;

    multilevelMesh.meshArray              = meshArrayTmp;
    multilevelMesh.elementChildrenArray   = elementChildrenArrayTmp;
    multilevelMesh.elementChildrenOffsets = elementChildrenOffsetsTmp;
    multilevelMesh.elementParentsArray    = elementParentsArrayTmp;
    
    //allocate new levels
    for (int i=multilevelMesh.nLevels; i < nLevelsNew; i++)
      {
	initializeMesh(multilevelMesh.meshArray[i]);
	multilevelMesh.elementChildrenArray[i]     = NULL;
	multilevelMesh.elementChildrenOffsets[i]   = NULL;
	multilevelMesh.elementParentsArray[i]      = NULL;
      }
    multilevelMesh.nLevels = nLevelsNew;
    return 0;
  }

  int locallyRefineEdgeMesh(MultilevelMesh& multilevelMesh, 
			    int * elementTagArray)
  {
    using namespace  std;
    //elementTagArray for now at least 
    //1 --> refine
    //0 --> don't

    //first extend arrays and data structures in multilevelMesh
    int nLevelsPrev = multilevelMesh.nLevels;
    growMultilevelMesh(1,multilevelMesh);
    assert(multilevelMesh.nLevels == nLevelsPrev+1);
    //start by initializing children offsets on previous finest mesh
    multilevelMesh.elementChildrenOffsets[nLevelsPrev-1] = 
      new int [multilevelMesh.meshArray[nLevelsPrev-1].nElements_global+1];
    //figure out how many elments and nodes to add
    int nElements_tagged = 0;
    for (int eN = 0; eN < multilevelMesh.meshArray[nLevelsPrev-1].nElements_global; eN++)
      {
	if (elementTagArray[eN] > 0)
	  nElements_tagged++;
      }
    //2 children per parent, lose the parent 
    int nElements_new = multilevelMesh.meshArray[nLevelsPrev-1].nElements_global+nElements_tagged;
    //1 new node per parent
    int nNodes_new    = multilevelMesh.meshArray[nLevelsPrev-1].nNodes_global+nElements_tagged;
    //allocate multilevel arrays for new level
    //childrenArray <-- old level
    //  choice 1 size nElementsNew only keep track of actual children
    //  choice 2 size nElements_global (on new mesh) an element considers itself to be
    //      a child on the new mesh
    multilevelMesh.elementChildrenArray[nLevelsPrev-1] = new int[nElements_new];
    //parent array : nElementsNew --> new level
    multilevelMesh.elementParentsArray[nLevelsPrev]    = new int[nElements_new];
    

    //allocate mesh data structures on new level
    multilevelMesh.meshArray[nLevelsPrev].nNodes_element   = 2;
    multilevelMesh.meshArray[nLevelsPrev].nElements_global = nElements_new;
    multilevelMesh.meshArray[nLevelsPrev].nNodes_global    = nNodes_new;
    //elementNodesArray
    multilevelMesh.meshArray[nLevelsPrev].elementNodesArray= 
      new int[multilevelMesh.meshArray[nLevelsPrev].nElements_global*2];
    //elementMaterialTypes
    multilevelMesh.meshArray[nLevelsPrev].elementMaterialTypes = 
      new int[multilevelMesh.meshArray[nLevelsPrev].nElements_global];
    //node array
    multilevelMesh.meshArray[nLevelsPrev].nodeArray = 
      new double[3*multilevelMesh.meshArray[nLevelsPrev].nNodes_global];
    //node material types
    multilevelMesh.meshArray[nLevelsPrev].nodeMaterialTypes = 
      new int[multilevelMesh.meshArray[nLevelsPrev].nNodes_global];

    //first copy over nodes from old mesh since nested
    for (int nN = 0; nN < multilevelMesh.meshArray[nLevelsPrev-1].nNodes_global; nN++)
      {
	multilevelMesh.meshArray[nLevelsPrev].nodeArray[3*nN+0] = 
	  multilevelMesh.meshArray[nLevelsPrev-1].nodeArray[3*nN+0];
	multilevelMesh.meshArray[nLevelsPrev].nodeArray[3*nN+1] = 
	  multilevelMesh.meshArray[nLevelsPrev-1].nodeArray[3*nN+1];
	multilevelMesh.meshArray[nLevelsPrev].nodeArray[3*nN+2] = 
	  multilevelMesh.meshArray[nLevelsPrev-1].nodeArray[3*nN+2];
	//material types too
	multilevelMesh.meshArray[nLevelsPrev].nodeMaterialTypes[nN] = 
	  multilevelMesh.meshArray[nLevelsPrev-1].nodeMaterialTypes[nN];
      }
    //where to start with new node numbers        
    int nN_new = multilevelMesh.meshArray[nLevelsPrev-1].nNodes_global;
    int eN_new = 0; //where to start with new element numbers
    memset(multilevelMesh.elementChildrenOffsets[nLevelsPrev-1],0,
	   (multilevelMesh.meshArray[nLevelsPrev-1].nElements_global+1)*sizeof(int));

    //go ahead and  use node set even though should be no chance of duplicating nodes
    set<Node> newNodeSet;
    set<Node>::iterator nodeItr;

    for (int eN_parent = 0; eN_parent < multilevelMesh.meshArray[nLevelsPrev-1].nElements_global; 
	 eN_parent++)
      {
	if (elementTagArray[eN_parent] == 0)
	  {
	    //not refining so just add to new mesh
	    multilevelMesh.meshArray[nLevelsPrev].elementNodesArray[eN_new*2+0] = 
	      multilevelMesh.meshArray[nLevelsPrev-1].elementNodesArray[eN_parent*2+0];
	    multilevelMesh.meshArray[nLevelsPrev].elementNodesArray[eN_new*2+1] = 
	      multilevelMesh.meshArray[nLevelsPrev-1].elementNodesArray[eN_parent*2+1];
	    //should element be its own child?
	    int offset = multilevelMesh.elementChildrenOffsets[nLevelsPrev-1][eN_parent];
	    multilevelMesh.elementChildrenOffsets[nLevelsPrev-1][eN_parent+1] = offset+1;
	    multilevelMesh.elementChildrenArray[nLevelsPrev-1][offset+0] = eN_new;
	    multilevelMesh.elementParentsArray[nLevelsPrev][eN_new]       = eN_parent;
	    eN_new += 1;
	  }
	else if (elementTagArray[eN_parent] > 0)
	  {
	    //2 children per parent
	    //add middle node
	    int offset = multilevelMesh.elementChildrenOffsets[nLevelsPrev-1][eN_parent];
	    multilevelMesh.elementChildrenOffsets[nLevelsPrev-1][eN_parent+1] = offset+2;
	    multilevelMesh.elementChildrenArray[nLevelsPrev-1][offset+0] = eN_new;
	    multilevelMesh.elementChildrenArray[nLevelsPrev-1][offset+1] = eN_new+1;
	    multilevelMesh.elementParentsArray[nLevelsPrev][eN_new]   = eN_parent;
	    multilevelMesh.elementParentsArray[nLevelsPrev][eN_new+1] = eN_parent;

            Node midpoints[1];
            midpoint(multilevelMesh.meshArray[nLevelsPrev-1].nodeArray + 
		     multilevelMesh.meshArray[nLevelsPrev-1].elementNodesArray[eN_parent*2 + 0]*3,
                     multilevelMesh.meshArray[nLevelsPrev-1].nodeArray + 
		     multilevelMesh.meshArray[nLevelsPrev-1].elementNodesArray[eN_parent*2 + 1]*3,
                     midpoints[0]);
	    midpoints[0].nN = nN_new;
            newNodeSet.insert(midpoints[0]);
	    nN_new++;
	    eN_new = newEdge(eN_new,multilevelMesh.meshArray[nLevelsPrev].elementNodesArray,
			     multilevelMesh.meshArray[nLevelsPrev-1].elementNodesArray[2*eN_parent+0],
			     midpoints[0].nN);
	    eN_new = newEdge(eN_new,multilevelMesh.meshArray[nLevelsPrev].elementNodesArray,
			     midpoints[0].nN,
			     multilevelMesh.meshArray[nLevelsPrev-1].elementNodesArray[2*eN_parent+1]);

	  }//actually refine
	
      }//parent loop
    assert(unsigned(nN_new) == (newNodeSet.size()+multilevelMesh.meshArray[nLevelsPrev-1].nNodes_global));
    assert(nN_new == multilevelMesh.meshArray[nLevelsPrev].nNodes_global);
    
    //could copy over nodes here like in globallyRefine      
    //now insert new nodes
    for(nodeItr=newNodeSet.begin();nodeItr!=newNodeSet.end();nodeItr++)
      {
	multilevelMesh.meshArray[nLevelsPrev].nodeArray[nodeItr->nN*3+0] = nodeItr->x;
	multilevelMesh.meshArray[nLevelsPrev].nodeArray[nodeItr->nN*3+1] = nodeItr->y;
	multilevelMesh.meshArray[nLevelsPrev].nodeArray[nodeItr->nN*3+2] = nodeItr->z;
      }
    //copy element material types from parents
    for (int eN = 0; eN < multilevelMesh.meshArray[nLevelsPrev].nElements_global; eN++)
      {
	int eN_parent = multilevelMesh.elementParentsArray[nLevelsPrev][eN];
	multilevelMesh.meshArray[nLevelsPrev].elementMaterialTypes[eN] = 
	  multilevelMesh.meshArray[nLevelsPrev-1].elementMaterialTypes[eN_parent];
      }
    //new nodes get default material type
    if (multilevelMesh.meshArray[nLevelsPrev].nNodes_global > multilevelMesh.meshArray[nLevelsPrev-1].nNodes_global)
      memset(multilevelMesh.meshArray[nLevelsPrev].nodeMaterialTypes+multilevelMesh.meshArray[nLevelsPrev-1].nNodes_global,
	     DEFAULT_NODE_MATERIAL,
	     (multilevelMesh.meshArray[nLevelsPrev].nNodes_global-multilevelMesh.meshArray[nLevelsPrev-1].nNodes_global)*sizeof(int));
    
    //build element boundary info etc here or in calling routine?
    return 0;
  }//locallyRefineEdgeMesh
  int locallyRefineTriangleMesh(MultilevelMesh& multilevelMesh, 
				int * elementTagArray)
  {
    using namespace  std;
    //elementTagArray for now at least 
    //1 --> refine
    //0 --> don't
    int failed = 0;
    //first extend arrays and data structures in multilevelMesh
    int nLevelsPrev = multilevelMesh.nLevels;
    if (multilevelMesh.meshArray[nLevelsPrev-1].newestNodeBases == NULL)
      return 1;
    growMultilevelMesh(1,multilevelMesh);
    assert(multilevelMesh.nLevels == nLevelsPrev+1);
    int nElements_tagged = 0;
    for (int eN = 0; eN < multilevelMesh.meshArray[nLevelsPrev-1].nElements_global; eN++)
      {
	if (elementTagArray[eN] > 0)
	  nElements_tagged++;
      }

    //to get things working, use temporaries for mesh that can be resized "easily"
    // and copy back to c interface
    vector<int> elementNodesArray_tmp,elementNeighborsArray_tmp,bases_tmp,
      elementParentsArray_tmp;
    vector<double>  nodeArray_tmp;
    //live on previous mesh
    vector<list<int> > elementChildrenList(multilevelMesh.meshArray[nLevelsPrev-1].nElements_global);
    vector<bool>  refined(multilevelMesh.meshArray[nLevelsPrev-1].nElements_global,
			  false);
    //try to use upper bound on new elements and grab memory up front?
    elementNodesArray_tmp.reserve(3*(multilevelMesh.meshArray[nLevelsPrev-1].nElements_global+
				     4*nElements_tagged));
    elementNeighborsArray_tmp.reserve(3*(multilevelMesh.meshArray[nLevelsPrev-1].nElements_global+
					 4*nElements_tagged));
    bases_tmp.reserve(multilevelMesh.meshArray[nLevelsPrev-1].nElements_global+
		      4*nElements_tagged);

    nodeArray_tmp.reserve(3*(multilevelMesh.meshArray[nLevelsPrev-1].nNodes_global+
			     3*nElements_tagged));

    
    elementParentsArray_tmp.reserve(multilevelMesh.meshArray[nLevelsPrev-1].nElements_global+
				    4*nElements_tagged);
    //copy over nodeArray, elementNodesArray, and elementNeighborsArray, bases
    //since these are used in the refinement process 

    elementNodesArray_tmp.insert(elementNodesArray_tmp.begin(),
				 multilevelMesh.meshArray[nLevelsPrev-1].elementNodesArray,
				 multilevelMesh.meshArray[nLevelsPrev-1].elementNodesArray+
				 multilevelMesh.meshArray[nLevelsPrev-1].nElements_global*3);

    elementNeighborsArray_tmp.insert(elementNeighborsArray_tmp.begin(),
				     multilevelMesh.meshArray[nLevelsPrev-1].elementNeighborsArray,
				     multilevelMesh.meshArray[nLevelsPrev-1].elementNeighborsArray+
				     multilevelMesh.meshArray[nLevelsPrev-1].nElements_global*3);

    bases_tmp.insert(bases_tmp.begin(),
		     multilevelMesh.meshArray[nLevelsPrev-1].newestNodeBases,
		     multilevelMesh.meshArray[nLevelsPrev-1].newestNodeBases+
		     multilevelMesh.meshArray[nLevelsPrev-1].nElements_global);

 
    nodeArray_tmp.insert(nodeArray_tmp.begin(),
			 multilevelMesh.meshArray[nLevelsPrev-1].nodeArray,
			 multilevelMesh.meshArray[nLevelsPrev-1].nodeArray+
			 3*multilevelMesh.meshArray[nLevelsPrev-1].nNodes_global);

    //new mesh starts off as parent mesh
    int nElements_new = multilevelMesh.meshArray[nLevelsPrev-1].nElements_global;
    int nNodes_new    = multilevelMesh.meshArray[nLevelsPrev-1].nNodes_global;
    for (int eN = 0; eN < multilevelMesh.meshArray[nLevelsPrev-1].nElements_global; eN++)
      {
	//every element is its own parent to start off with
	elementParentsArray_tmp.push_back(eN); 
      }
    for (int eN_parent = 0; 
	 eN_parent < multilevelMesh.meshArray[nLevelsPrev-1].nElements_global; eN_parent++)
      {
	//use loop through elements on parent mesh but these numbers will get
	//reassigned in refinement process
	if (elementTagArray[eN_parent] == 1 && !refined[eN_parent])
	  {
	    failed = failed || newestNodeBisect(eN_parent,
						nElements_new,
						nNodes_new,
						nodeArray_tmp,
						elementNodesArray_tmp,
						elementNeighborsArray_tmp,
						elementChildrenList,
						elementParentsArray_tmp,
						bases_tmp,
						refined);
	  }//eN_parent is tagged and not already refined
      }//elements on original mesh
    //std::cout<<"Done with refinement"<<std::endl;
    //now have to allocate c interface data and copy over
    assert(elementParentsArray_tmp.size() == unsigned(nElements_new));
    //parent array : nElementsNew --> new level
    multilevelMesh.elementParentsArray[nLevelsPrev]    = new int[nElements_new];
    copy(elementParentsArray_tmp.begin(),elementParentsArray_tmp.end(),
	 multilevelMesh.elementParentsArray[nLevelsPrev]);

    //initialize children offsets on previous finest mesh
    multilevelMesh.elementChildrenOffsets[nLevelsPrev-1] = 
      new int [multilevelMesh.meshArray[nLevelsPrev-1].nElements_global+1];
    multilevelMesh.elementChildrenArray[nLevelsPrev-1] = new int[nElements_new];
    multilevelMesh.elementChildrenOffsets[nLevelsPrev-1][0] = 0;
    for (int eN = 0; eN < multilevelMesh.meshArray[nLevelsPrev-1].nElements_global; eN++)
      {
	int offset = multilevelMesh.elementChildrenOffsets[nLevelsPrev-1][eN];
	if (elementChildrenList[eN].size() > 0) //eN was refined
	  {
	    multilevelMesh.elementChildrenOffsets[nLevelsPrev-1][eN+1] = 
	      offset + elementChildrenList[eN].size();
	    int i = 0;
	    for (std::list<int>::iterator it = elementChildrenList[eN].begin(); 
		 it != elementChildrenList[eN].end(); it++)
	      {
		multilevelMesh.elementChildrenArray[nLevelsPrev-1][offset+i] = *it;
		i++;
	      }
	  }
	else //element is its own child
	  {
	    multilevelMesh.elementChildrenOffsets[nLevelsPrev-1][eN+1] = offset+1;
	    multilevelMesh.elementChildrenArray[nLevelsPrev-1][offset+0] = eN;
	  }
      }
    
    //now take care of mesh data structures
    multilevelMesh.meshArray[nLevelsPrev].nNodes_element   = 3;
    multilevelMesh.meshArray[nLevelsPrev].nElements_global = nElements_new;
    multilevelMesh.meshArray[nLevelsPrev].nNodes_global    = nNodes_new;
    
    //elementNodesArray
    multilevelMesh.meshArray[nLevelsPrev].elementNodesArray= 
      new int[multilevelMesh.meshArray[nLevelsPrev].nElements_global*3];
    //elementMaterialTypes
    multilevelMesh.meshArray[nLevelsPrev].elementMaterialTypes = 
      new int[multilevelMesh.meshArray[nLevelsPrev].nElements_global];
    //nodal material types
    multilevelMesh.meshArray[nLevelsPrev].nodeMaterialTypes = 
      new int[multilevelMesh.meshArray[nLevelsPrev].nNodes_global];

    //node array
    multilevelMesh.meshArray[nLevelsPrev].nodeArray = 
      new double[3*multilevelMesh.meshArray[nLevelsPrev].nNodes_global];
    multilevelMesh.meshArray[nLevelsPrev].newestNodeBases= 
      new int[multilevelMesh.meshArray[nLevelsPrev].nElements_global];

    //copy over from tmp's
    copy(elementNodesArray_tmp.begin(),elementNodesArray_tmp.end(),
	 multilevelMesh.meshArray[nLevelsPrev].elementNodesArray);
    copy(nodeArray_tmp.begin(),nodeArray_tmp.end(),
	 multilevelMesh.meshArray[nLevelsPrev].nodeArray);
    copy(bases_tmp.begin(),bases_tmp.end(),
	 multilevelMesh.meshArray[nLevelsPrev].newestNodeBases);

    //copy element material types from parents
    for (int eN = 0; eN < multilevelMesh.meshArray[nLevelsPrev].nElements_global; eN++)
      {
	int eN_parent = multilevelMesh.elementParentsArray[nLevelsPrev][eN];
	multilevelMesh.meshArray[nLevelsPrev].elementMaterialTypes[eN] = 
	  multilevelMesh.meshArray[nLevelsPrev-1].elementMaterialTypes[eN_parent];
      }
    
    //nodes on parent mesh retain ids
    copy(multilevelMesh.meshArray[nLevelsPrev-1].nodeMaterialTypes,
	 multilevelMesh.meshArray[nLevelsPrev-1].nodeMaterialTypes + multilevelMesh.meshArray[nLevelsPrev-1].nNodes_global,
	 multilevelMesh.meshArray[nLevelsPrev].nodeMaterialTypes);
    if (multilevelMesh.meshArray[nLevelsPrev].nNodes_global > multilevelMesh.meshArray[nLevelsPrev-1].nNodes_global)
      memset(multilevelMesh.meshArray[nLevelsPrev].nodeMaterialTypes+multilevelMesh.meshArray[nLevelsPrev-1].nNodes_global,
	     DEFAULT_NODE_MATERIAL,
	     (multilevelMesh.meshArray[nLevelsPrev].nNodes_global-multilevelMesh.meshArray[nLevelsPrev-1].nNodes_global)*sizeof(int));

    return failed;
  }//locallyRefineTriangularMesh
  //may be able to get newest node alg to work even when bases not defined?
  int setNewestNodeBasesToLongestEdge(MultilevelMesh& multilevelMesh)//or just mesh? Mesh& mesh
  {
    using namespace std;
    int nLevels = multilevelMesh.nLevels;
    if (multilevelMesh.meshArray[nLevels-1].newestNodeBases != NULL)
      {
	logEvent("WARNING setNewestNodeBasesToLongestEdge newestNodeBases !=NULL exiting",5);
	return 1;
      }
    //2d 
    if (multilevelMesh.meshArray[nLevels-1].nElementBoundaries_element != 3)
      {
	logEvent("WARNING setNewestNodeBasesToLongestEdge 2d only exiting",5);
	return 1;
      }
    
    multilevelMesh.meshArray[nLevels-1].newestNodeBases = 
      new int[multilevelMesh.meshArray[nLevels-1].nElements_global];
    for (int eN = 0; eN < multilevelMesh.meshArray[nLevels-1].nElements_global; eN++)
      {
	int ebN_longest_local = findLocalLongestEdge2d(eN,
						       multilevelMesh.meshArray[nLevels-1].elementNodesArray,
						       multilevelMesh.meshArray[nLevels-1].nodeArray);
	multilevelMesh.meshArray[nLevels-1].newestNodeBases[eN] = ebN_longest_local;
      }
    return 0;
  }
  
  int locallyRefineTriangleMesh_redGreen(MultilevelMesh& multilevelMesh, 
                                         int * elementTagArray)
  {
    using namespace  std;
    
    //first extend arrays and data structures in multilevelMesh to make room for new mesh
    int nLevelsPrev = multilevelMesh.nLevels;
    growMultilevelMesh(1,multilevelMesh);
    assert(multilevelMesh.nLevels == nLevelsPrev+1);

    //build a set of the nodes added in refinement
    set<Node> newNodeSet;
    set<Node>::iterator nodeItr;
    //build the elementChildren and elementParent maps as maps so we can accumulate on the fly
    map<int, vector<int> > elementChildren;
    map<int, int> elementParents;
    //collect the set of parent elements slated for bisection or uniform refinement
    map<int,int> elementsForBisection;
    map<int,vector<int> > elementsForTrisection,newElementsForTrisection,nextElementsForTrisection;
    set<int> elementsForUniform,newElementsForUniform,nextElementsForUniform;
    //keep track of the new elements as a vector of 4 ints (eN,n0,n1,n2)
    vector<valarray<int> > newElements;
    //keep track of what old elements  survive on the new mesh
    set<int> oldElements;
    int i = nLevelsPrev;
    //initialize old elements to all elements and tag some for uniform refinement
    for(int eN_parent=0;eN_parent<multilevelMesh.meshArray[i-1].nElements_global;eN_parent++)
      {
        oldElements.insert(eN_parent);
        if (elementTagArray[eN_parent] > 0)
          {
            newElementsForUniform.insert(eN_parent);
            elementsForUniform.insert(eN_parent);
          }
      }
    int nN_new = multilevelMesh.meshArray[i-1].nNodes_global;
    int eN_new = multilevelMesh.meshArray[i-1].nElements_global;
    while (!newElementsForUniform.empty() || !newElementsForTrisection.empty())
      {
        //std::cout<<"new uniform"<<std::endl;
        for(set<int>::iterator eN_uniform_itr = newElementsForUniform.begin(); eN_uniform_itr != newElementsForUniform.end(); eN_uniform_itr++)
          {
            int eN_parent = *eN_uniform_itr;
            oldElements.erase(eN_parent);
            elementChildren[eN_parent].push_back(eN_new+0);
            elementChildren[eN_parent].push_back(eN_new+1);
            elementChildren[eN_parent].push_back(eN_new+2);
            elementChildren[eN_parent].push_back(eN_parent);
            elementParents[eN_new + 0] = eN_parent;
            elementParents[eN_new + 1] = eN_parent;
            elementParents[eN_new + 2] = eN_parent;
            elementParents[eN_parent]  = eN_parent;
            Node midpoints[3];
            for(int nN_element_0=0,nN_midpoint=0;nN_element_0<multilevelMesh.meshArray[i-1].nNodes_element;nN_element_0++)
              for(int nN_element_1=nN_element_0+1;nN_element_1<multilevelMesh.meshArray[i-1].nNodes_element;nN_element_1++,nN_midpoint++)
                {
                  midpoint(multilevelMesh.meshArray[i-1].nodeArray + multilevelMesh.meshArray[i-1].elementNodesArray[eN_parent*3 + nN_element_0]*3,
                           multilevelMesh.meshArray[i-1].nodeArray + multilevelMesh.meshArray[i-1].elementNodesArray[eN_parent*3 + nN_element_1]*3,
                           midpoints[nN_midpoint]);
                  nodeItr = newNodeSet.find(midpoints[nN_midpoint]);
                  if(nodeItr == newNodeSet.end())
                    {
                      midpoints[nN_midpoint].nN = nN_new;
                      newNodeSet.insert(midpoints[nN_midpoint]);
                      nN_new++;
                    }
                  else
                    midpoints[nN_midpoint].nN = nodeItr->nN;
                }
            //the triangles formed by chopping the points off the parent
            valarray<int> newElement(4);
            newElement[0] = eN_new;
            newElement[1] = multilevelMesh.meshArray[i-1].elementNodesArray[3*eN_parent+0];
            newElement[2] = midpoints[0].nN;
            newElement[3] = midpoints[1].nN;
            newElements.push_back(newElement);
            eN_new++;
            newElement[0] = eN_new;
            newElement[1] = multilevelMesh.meshArray[i-1].elementNodesArray[3*eN_parent+1];
            newElement[2] = midpoints[0].nN;
            newElement[3] = midpoints[2].nN;
            newElements.push_back(newElement);
            eN_new++;
            newElement[0] = eN_new;
            newElement[1] = multilevelMesh.meshArray[i-1].elementNodesArray[3*eN_parent+2];
            newElement[2] = midpoints[1].nN;
            newElement[3] = midpoints[2].nN;
            newElements.push_back(newElement);
            eN_new++;
            newElement[0] = eN_parent;
            newElement[1] = midpoints[0].nN;
            newElement[2] = midpoints[1].nN;
            newElement[3] = midpoints[2].nN;
            newElements.push_back(newElement);
            //put neighbors in correct refinement class to make conforming
            for (int ebN = 0; ebN < multilevelMesh.meshArray[i-1].nElementBoundaries_element; ebN++)
              {
                int eN_neighbor = multilevelMesh.meshArray[i-1].elementNeighborsArray[eN_parent*multilevelMesh.meshArray[i-1].nElementBoundaries_element +
                                                                                      ebN],
                  ebN_global =  multilevelMesh.meshArray[i-1].elementBoundariesArray[eN_parent*multilevelMesh.meshArray[i-1].nElementBoundaries_element+
                                                                                     ebN],
                  ebN_neighbor_element=0;
                if(eN_neighbor == multilevelMesh.meshArray[i-1].elementBoundaryElementsArray[ebN_global*2+0])
                  ebN_neighbor_element = multilevelMesh.meshArray[i-1].elementBoundaryLocalElementBoundariesArray[ebN_global*2 + 0];
                else
                  ebN_neighbor_element = multilevelMesh.meshArray[i-1].elementBoundaryLocalElementBoundariesArray[ebN_global*2 + 1];
                if (eN_neighbor != -1 && elementsForUniform.find(eN_neighbor) == elementsForUniform.end()) //if not in uniform
                  {
                    if (elementsForTrisection.find(eN_neighbor) == elementsForTrisection.end()) //if not in trisection
                      {
                        if(elementsForBisection.find(eN_neighbor) == elementsForBisection.end()) // if not in bisection
                          {
                            //find longest edge to see whether to bisect or trisect 
                            int leftNode,rightNode,
                              leftNode1,rightNode1,
                              ebN_neighbor_element1,ebN_global1,
                              longestEdge,longestEdge_element;
                            double edgeLength,edgeLength1;
                            leftNode = multilevelMesh.meshArray[i-1].edgeNodesArray[ebN_global*2 + 0];
                            rightNode = multilevelMesh.meshArray[i-1].edgeNodesArray[ebN_global*2 + 1];
                            edgeLength = edgeLengthFromNodeNumbers(multilevelMesh.meshArray[i-1].nodeArray,leftNode,rightNode);
                            longestEdge=ebN_global;
                            longestEdge_element=ebN_neighbor_element;
                            for (int offset=1;offset < multilevelMesh.meshArray[i-1].nElementBoundaries_element; offset++)
                              {
                                ebN_neighbor_element1 = (ebN_neighbor_element + offset)%multilevelMesh.meshArray[i-1].nElementBoundaries_element;
                                ebN_global1 = multilevelMesh.meshArray[i-1].elementBoundariesArray[eN_neighbor*multilevelMesh.meshArray[i-1].nElementBoundaries_element+
                                                                                                   ebN_neighbor_element1];
                                leftNode1 = multilevelMesh.meshArray[i-1].edgeNodesArray[ebN_global1*2 + 0];
                                rightNode1 = multilevelMesh.meshArray[i-1].edgeNodesArray[ebN_global1*2 + 1];
                                edgeLength1 = edgeLengthFromNodeNumbers(multilevelMesh.meshArray[i-1].nodeArray,leftNode1,rightNode1);
                                if (edgeLength1 > edgeLength)
                                  {
                                    longestEdge = ebN_global1;
                                    longestEdge_element=ebN_neighbor_element1;
                                  }
                              }
                            if (longestEdge == ebN_global) //if longest edge then bisect
                              {
                                elementsForBisection[eN_neighbor] = longestEdge_element;
                              }
                            else //if not longest edge then trisect
                              {
                                newElementsForTrisection[eN_neighbor].push_back(longestEdge_element);
                                newElementsForTrisection[eN_neighbor].push_back(ebN_neighbor_element);
                                elementsForTrisection[eN_neighbor].push_back(longestEdge_element);
                                elementsForTrisection[eN_neighbor].push_back(ebN_neighbor_element);
                              }
                          }
                        else //if already bisected (on another edge) then trisect
                          {
                            if (elementsForBisection[eN_neighbor] != ebN_neighbor_element)
                              {
                                newElementsForTrisection[eN_neighbor].push_back(elementsForBisection[eN_neighbor]);
                                newElementsForTrisection[eN_neighbor].push_back(ebN_neighbor_element);
                                elementsForTrisection[eN_neighbor].push_back(elementsForBisection[eN_neighbor]);
                                elementsForTrisection[eN_neighbor].push_back(ebN_neighbor_element);
                                elementsForBisection.erase(eN_neighbor);
                              }
                          }
                      }
                    else //if already trisected
                      {
                        if (elementsForTrisection[eN_neighbor][0] != ebN and elementsForTrisection[eN_neighbor][0] != ebN) //if this edge is non-conforming
                          {
                            nextElementsForUniform.insert(eN_neighbor);
                            elementsForUniform.insert(eN_neighbor);
                            elementsForTrisection.erase(eN_neighbor);
                            newElementsForTrisection.erase(eN_neighbor);
                          }
                      }
                  }
              }
          }
        newElementsForUniform.clear();
        newElementsForUniform = nextElementsForUniform;
        nextElementsForUniform.clear();
        //std::cout<<"new trisection"<<std::endl;
        for(map<int,vector<int> >::iterator eN_trisection_itr = newElementsForTrisection.begin(); eN_trisection_itr != newElementsForTrisection.end(); eN_trisection_itr++)
          {
            int eN_parent = eN_trisection_itr->first,
              longestEdge_element =  eN_trisection_itr->second[0],
              otherEdge_element = eN_trisection_itr->second[1];
            int longestEdge = multilevelMesh.meshArray[i-1].elementBoundariesArray[eN_parent*3 + longestEdge_element],
              otherEdge = multilevelMesh.meshArray[i-1].elementBoundariesArray[eN_parent*3 + otherEdge_element];
            //add the new node if  necessary, it can only be the longest  edge
            Node midpoint_new;
            midpoint(multilevelMesh.meshArray[i-1].nodeArray + multilevelMesh.meshArray[i-1].edgeNodesArray[longestEdge*2 + 0]*3,
                     multilevelMesh.meshArray[i-1].nodeArray + multilevelMesh.meshArray[i-1].edgeNodesArray[longestEdge*2 + 1]*3,
                     midpoint_new);
            nodeItr = newNodeSet.find(midpoint_new);
            if(nodeItr == newNodeSet.end()) //if this node is new then add it and decide what to do with the neighbor
              {
                midpoint_new.nN = nN_new;
                newNodeSet.insert(midpoint_new);
                nN_new++;
                //since the node is new the neighbor will be non-conforming
                //put neighbor in right refinement class
                int eN_neighbor = multilevelMesh.meshArray[i-1].elementNeighborsArray[eN_parent*multilevelMesh.meshArray[i-1].nElementBoundaries_element + longestEdge_element],
                  ebN_global =  multilevelMesh.meshArray[i-1].elementBoundariesArray[eN_parent*multilevelMesh.meshArray[i-1].nElementBoundaries_element+
                                                                                     longestEdge_element],
                  ebN_neighbor_element=0;
                if(eN_neighbor == multilevelMesh.meshArray[i-1].elementBoundaryElementsArray[ebN_global*2+0])
                  ebN_neighbor_element = multilevelMesh.meshArray[i-1].elementBoundaryLocalElementBoundariesArray[ebN_global*2 + 0];
                else
                  ebN_neighbor_element = multilevelMesh.meshArray[i-1].elementBoundaryLocalElementBoundariesArray[ebN_global*2 + 1];
                if (eN_neighbor != -1 && elementsForUniform.find(eN_neighbor) == elementsForUniform.end()) //if not in uniform
                  {
                    if (elementsForTrisection.find(eN_neighbor) == elementsForTrisection.end()) //if not in trisection
                      {
                        if(elementsForBisection.find(eN_neighbor) == elementsForBisection.end()) // if not in bisection
                          {
                            //find longest edge to see whether to bisect or trisect 
                            int leftNode,rightNode,
                              leftNode1,rightNode1,
                              ebN_neighbor_element1,ebN_global1,
                              longestEdge,longestEdge_element;
                            double edgeLength,edgeLength1;
                            leftNode = multilevelMesh.meshArray[i-1].edgeNodesArray[ebN_global*2 + 0];
                            rightNode = multilevelMesh.meshArray[i-1].edgeNodesArray[ebN_global*2 + 1];
                            edgeLength = edgeLengthFromNodeNumbers(multilevelMesh.meshArray[i-1].nodeArray,leftNode,rightNode);
                            longestEdge=ebN_global;
                            longestEdge_element=ebN_neighbor_element;
                            for (int offset=1;offset < multilevelMesh.meshArray[i-1].nElementBoundaries_element; offset++)
                              {
                                ebN_neighbor_element1 = (ebN_neighbor_element + offset)%multilevelMesh.meshArray[i-1].nElementBoundaries_element;
                                ebN_global1 = multilevelMesh.meshArray[i-1].elementBoundariesArray[eN_neighbor*multilevelMesh.meshArray[i-1].nElementBoundaries_element+
                                                                                                   ebN_neighbor_element1];
                                
                                leftNode1 = multilevelMesh.meshArray[i-1].edgeNodesArray[ebN_global1*2 + 0];
                                rightNode1 = multilevelMesh.meshArray[i-1].edgeNodesArray[ebN_global1*2 + 1];
                                edgeLength1 = edgeLengthFromNodeNumbers(multilevelMesh.meshArray[i-1].nodeArray,leftNode1,rightNode1);
                                if (edgeLength1 > edgeLength)
                                  {
                                    longestEdge = ebN_global1;
                                    longestEdge_element=ebN_neighbor_element1;
                                  }
                              }
                            if (longestEdge == ebN_global) //if longest edge then bisect
                              {
                                elementsForBisection[eN_neighbor] = longestEdge_element;
                              }
                            else //if not longest edge then trisect
                              {
                                nextElementsForTrisection[eN_neighbor].push_back(longestEdge_element);
                                nextElementsForTrisection[eN_neighbor].push_back(ebN_neighbor_element);
                                elementsForTrisection[eN_neighbor].push_back(longestEdge_element);
                                elementsForTrisection[eN_neighbor].push_back(ebN_neighbor_element);
                              }
                          }
                        else //if already bisected (on another edge) then trisect
                          {
                            assert(elementsForBisection[eN_neighbor] != ebN_neighbor_element);
                            nextElementsForTrisection[eN_neighbor].push_back(elementsForBisection[eN_neighbor]);
                            nextElementsForTrisection[eN_neighbor].push_back(ebN_neighbor_element);
                            elementsForTrisection[eN_neighbor].push_back(elementsForBisection[eN_neighbor]);
                            elementsForTrisection[eN_neighbor].push_back(ebN_neighbor_element);
                            elementsForBisection.erase(eN_neighbor);
                          }
                      }
                    else //if already trisected
                      {
                        if (elementsForTrisection[eN_neighbor][0] != longestEdge and elementsForTrisection[eN_neighbor][0] != longestEdge) //if this edge is non-conforming
                          {
                            newElementsForUniform.insert(eN_neighbor);
                            elementsForUniform.insert(eN_neighbor);
                            elementsForTrisection.erase(eN_neighbor);
                            nextElementsForTrisection.erase(eN_neighbor);
                          }
                      }
                  }
              }
            Node midpoint_other;
            midpoint(multilevelMesh.meshArray[i-1].nodeArray + multilevelMesh.meshArray[i-1].edgeNodesArray[otherEdge*2 + 0]*3,
                     multilevelMesh.meshArray[i-1].nodeArray + multilevelMesh.meshArray[i-1].edgeNodesArray[otherEdge*2 + 1]*3,
                     midpoint_other);
            nodeItr = newNodeSet.find(midpoint_other);
            assert(nodeItr != newNodeSet.end());
          }
        newElementsForTrisection.clear();
        newElementsForTrisection = nextElementsForTrisection;
        nextElementsForTrisection.clear();
      }
    //std::cout<<"building trisected elements"<<std::endl;
    for(map<int,vector<int> >::iterator eN_trisection_itr = elementsForTrisection.begin(); eN_trisection_itr != elementsForTrisection.end(); eN_trisection_itr++)
      {
        int eN_parent = eN_trisection_itr->first,
          longestEdge_element =  eN_trisection_itr->second[0],
          otherEdge_element = eN_trisection_itr->second[1];
        int longestEdge = multilevelMesh.meshArray[i-1].elementBoundariesArray[eN_parent*3 + longestEdge_element],
          otherEdge = multilevelMesh.meshArray[i-1].elementBoundariesArray[eN_parent*3 + otherEdge_element];
        int longestEdge_leftNode = multilevelMesh.meshArray[i-1].edgeNodesArray[longestEdge*2 + 0],
          longestEdge_rightNode = multilevelMesh.meshArray[i-1].edgeNodesArray[longestEdge*2 + 1],
          otherEdge_leftNode = multilevelMesh.meshArray[i-1].edgeNodesArray[otherEdge*2 + 0],
          otherEdge_rightNode = multilevelMesh.meshArray[i-1].edgeNodesArray[otherEdge*2 + 1],
          n0,n1,n2;
        if (otherEdge_rightNode == longestEdge_leftNode)
          {
            n0 = otherEdge_rightNode;
            n1 = otherEdge_leftNode;
            n2 = longestEdge_rightNode;
          }
        else if (otherEdge_rightNode == longestEdge_rightNode)
          {
            n0 = otherEdge_rightNode;
            n1 = otherEdge_leftNode;
            n2 = longestEdge_leftNode;
          }
        else if (otherEdge_leftNode == longestEdge_leftNode)
          {
            n0 = otherEdge_leftNode;
            n1 = otherEdge_rightNode;
            n2 = longestEdge_rightNode;
          }
        else
          {
            n0 = otherEdge_leftNode;
            n1 = otherEdge_rightNode;
            n2 = longestEdge_leftNode;
          }
        oldElements.erase(eN_parent);
        elementChildren[eN_parent].push_back(eN_new+0);
        elementChildren[eN_parent].push_back(eN_new+1);
        elementChildren[eN_parent].push_back(eN_parent);
        elementParents[eN_new + 0] = eN_parent;
        elementParents[eN_new + 1] = eN_parent;
        elementParents[eN_parent]  = eN_parent;
        //no new  nodes should need to be added
        Node midpoint_new;
        midpoint(multilevelMesh.meshArray[i-1].nodeArray + multilevelMesh.meshArray[i-1].edgeNodesArray[longestEdge*2 + 0]*3,
                 multilevelMesh.meshArray[i-1].nodeArray + multilevelMesh.meshArray[i-1].edgeNodesArray[longestEdge*2 + 1]*3,
                 midpoint_new);
        nodeItr = newNodeSet.find(midpoint_new);
        assert(nodeItr != newNodeSet.end());
        midpoint_new.nN = nodeItr->nN;
        Node midpoint_other;
        midpoint(multilevelMesh.meshArray[i-1].nodeArray + multilevelMesh.meshArray[i-1].edgeNodesArray[otherEdge*2 + 0]*3,
                 multilevelMesh.meshArray[i-1].nodeArray + multilevelMesh.meshArray[i-1].edgeNodesArray[otherEdge*2 + 1]*3,
                 midpoint_other);
        nodeItr = newNodeSet.find(midpoint_other);
        assert(nodeItr != newNodeSet.end());
        midpoint_other.nN = nodeItr->nN;
        //not worrying about orientation for now
        valarray<int> newElement(4);
        newElement[0] = eN_new;
        newElement[1] = n0;
        newElement[2] = midpoint_new.nN;
        newElement[3] = midpoint_other.nN;
        newElements.push_back(newElement);
        eN_new++;
        newElement[0] = eN_new;
        newElement[1] = n1;
        newElement[2] = midpoint_other.nN;
        newElement[3] = midpoint_new.nN;
        newElements.push_back(newElement);
        eN_new++;
        newElement[0] = eN_parent;
        newElement[1] = n1;
        newElement[2] = midpoint_new.nN;
        newElement[3] = n2;
        newElements.push_back(newElement);
      }

    //std::cout<<"building bisected elements"<<std::endl;
    //now we just have to bisect remaining elements
    for(map<int,int>::iterator eN_bisect_itr = elementsForBisection.begin(); eN_bisect_itr != elementsForBisection.end(); eN_bisect_itr++)
      {
        int eN_parent = eN_bisect_itr->first;
        //longestEdge_element = eN_bisect_itr->second;
        //fix  later to use the longest edge info instead of searching again
	//         int longestEdge = multilevelMesh.meshArray[i-1].elementBoundaryElementsArray[eN_parent*3 + longestEdge_element],
	//           longestEdge_leftNode = multilevelMesh.meshArray[i-1].edgeNodesArray[longestEdge*2 + 0],
	//           longestEdge_rightNode = multilevelMesh.meshArray[i-1].edgeNodesArray[longestEdge*2 + 1],
	//           n0,n1,n2;

	//         if (otherEdge_rightNode == longestEdge_leftNode)
	//           {
	//             n0 = otherEdge_rightNode;
	//             n1 = otherEdge_leftNode;
	//             n2 = longestEdge_rightNode;
	//           }
	//         else if (otherEdge_rightNode == longestEdge_rightNode)
	//           {
	//             n0 = otherEdge_rightNode;
	//             n1 = otherEdge_leftNode;
	//             n2 = longestEdge_leftNode;
	//           }
	//         else if (otherEdge_leftNode == longestEdge_leftNode)
	//           {
	//             n0 = otherEdge_leftNode;
	//             n1 = otherEdge_rightNode;
	//             n2 = longestEdge_rightNode;
	//           }
	//         else
	//           {
	//             n0 = otherEdge_leftNode;
	//             n1 = otherEdge_rightNode;
	//             n2 = longestEdge_leftNode;
	//           }
        oldElements.erase(eN_parent);
        //elementsForBisection.erase(eN_parent);
        elementChildren[eN_parent].push_back(eN_new);
        elementChildren[eN_parent].push_back(eN_parent);
        elementParents[eN_new] = eN_parent;
        elementParents[eN_parent] = eN_parent;
        Node m;
        //go around the nodes to figure out which one connects to the new midpoint
        int nN_element_1,nN_element_2;
        bool foundMidpoint=false;
        for(int nN_element_0=0;nN_element_0<multilevelMesh.meshArray[i-1].nNodes_element;nN_element_0++)
          {
            foundMidpoint=false;
            nN_element_1 = (nN_element_0+1)%3;
            nN_element_2 = (nN_element_0+2)%3;
            midpoint(multilevelMesh.meshArray[i-1].nodeArray + multilevelMesh.meshArray[i-1].elementNodesArray[eN_parent*3 + nN_element_1]*3,
                     multilevelMesh.meshArray[i-1].nodeArray + multilevelMesh.meshArray[i-1].elementNodesArray[eN_parent*3 + nN_element_2]*3,
                     m);
            nodeItr = newNodeSet.find(m);
            if(nodeItr != newNodeSet.end())
              {
                //the triangles formed by bisecting
                valarray<int> newElement(4);
                newElement[0] = eN_new;
                newElement[1] = multilevelMesh.meshArray[i-1].elementNodesArray[3*eN_parent+nN_element_0];
                newElement[2] = multilevelMesh.meshArray[i-1].elementNodesArray[3*eN_parent+nN_element_1];
                newElement[3] = nodeItr->nN;
                newElements.push_back(newElement);
                eN_new++;
                newElement[0] = eN_parent;
                newElement[1] = multilevelMesh.meshArray[i-1].elementNodesArray[3*eN_parent+nN_element_0];
                newElement[2] = nodeItr->nN;
                newElement[3] = multilevelMesh.meshArray[i-1].elementNodesArray[3*eN_parent+nN_element_2];
                newElements.push_back(newElement);
                foundMidpoint=true;
                break;
              }
          }
      }
    //elementsForBisection.
    //assert(elementsForBisection.empty());
    //std::cout<<"finishing mesh"<<std::endl;
    //now the mesh should be conforming so build the mesh arrays
    multilevelMesh.meshArray[i].nNodes_element=3;
    multilevelMesh.meshArray[i].nElements_global  = eN_new;
    multilevelMesh.meshArray[i].elementNodesArray = new int[multilevelMesh.meshArray[i].nElements_global*3];
    multilevelMesh.elementChildrenArray[i-1]      = new int[multilevelMesh.meshArray[i].nElements_global];
    multilevelMesh.elementChildrenOffsets[i-1]  = new int[multilevelMesh.meshArray[i-1].nElements_global+1];
    multilevelMesh.elementParentsArray[i]         = new int[multilevelMesh.meshArray[i].nElements_global];
    multilevelMesh.meshArray[i].elementMaterialTypes = new int[multilevelMesh.meshArray[i].nElements_global];
    //write the old elements
    for(set<int>::iterator eN_itr=oldElements.begin();eN_itr != oldElements.end();eN_itr++)
      {
        int eN = *eN_itr;
        multilevelMesh.meshArray[i].elementNodesArray[eN*3+0] = multilevelMesh.meshArray[i-1].elementNodesArray[eN*3+0];
        multilevelMesh.meshArray[i].elementNodesArray[eN*3+1] = multilevelMesh.meshArray[i-1].elementNodesArray[eN*3+1];
        multilevelMesh.meshArray[i].elementNodesArray[eN*3+2] = multilevelMesh.meshArray[i-1].elementNodesArray[eN*3+2];
      }
    //write the new elements
    for(vector<valarray<int> >::iterator element_itr=newElements.begin();element_itr != newElements.end();element_itr++)
      {
        int eN = (*element_itr)[0];
        multilevelMesh.meshArray[i].elementNodesArray[eN*3+0] = (*element_itr)[1];
        multilevelMesh.meshArray[i].elementNodesArray[eN*3+1] = (*element_itr)[2];
        multilevelMesh.meshArray[i].elementNodesArray[eN*3+2] = (*element_itr)[3];
      }
    //write the nodes
    assert(unsigned(nN_new) == (newNodeSet.size()+multilevelMesh.meshArray[i-1].nNodes_global));
    multilevelMesh.meshArray[i].nNodes_global = nN_new;
    multilevelMesh.meshArray[i].nodeArray = new double[multilevelMesh.meshArray[i].nNodes_global*3];
    for(int nN=0;nN<multilevelMesh.meshArray[i-1].nNodes_global;nN++)
      {
        multilevelMesh.meshArray[i].nodeArray[nN*3+0] = multilevelMesh.meshArray[i-1].nodeArray[nN*3+0];
        multilevelMesh.meshArray[i].nodeArray[nN*3+1] = multilevelMesh.meshArray[i-1].nodeArray[nN*3+1];
        multilevelMesh.meshArray[i].nodeArray[nN*3+2] = multilevelMesh.meshArray[i-1].nodeArray[nN*3+2];
      }
    for(nodeItr=newNodeSet.begin();nodeItr!=newNodeSet.end();nodeItr++)
      {
        multilevelMesh.meshArray[i].nodeArray[nodeItr->nN*3+0] = nodeItr->x;
        multilevelMesh.meshArray[i].nodeArray[nodeItr->nN*3+1] = nodeItr->y;
        multilevelMesh.meshArray[i].nodeArray[nodeItr->nN*3+2] = nodeItr->z;
      }
    //element parents and material types
    for(int eN = 0; eN < multilevelMesh.meshArray[i].nElements_global;eN++)
      {
        multilevelMesh.elementParentsArray[i][eN] = elementParents[eN];
        multilevelMesh.meshArray[i].elementMaterialTypes[eN] = multilevelMesh.meshArray[i-1].elementMaterialTypes[elementParents[eN]];
      }
    //element children
    multilevelMesh.elementChildrenOffsets[i-1][0] = 0;
    int offset=0;
    for(int eN_parent = 0; eN_parent < multilevelMesh.meshArray[i-1].nElements_global;eN_parent++)
      {
        for(unsigned int childE=0;childE<elementChildren[eN_parent].size();childE++)
          {
            multilevelMesh.elementChildrenArray[i-1][offset] = elementChildren[eN_parent][childE]; 
            offset++;
          }
        multilevelMesh.elementChildrenOffsets[i-1][eN_parent+1] = offset;
      }

    //nodes on parent mesh retain ids
    copy(multilevelMesh.meshArray[i-1].nodeMaterialTypes,
	 multilevelMesh.meshArray[i-1].nodeMaterialTypes + multilevelMesh.meshArray[i-1].nNodes_global,
	 multilevelMesh.meshArray[i].nodeMaterialTypes);
    if (multilevelMesh.meshArray[i].nNodes_global > multilevelMesh.meshArray[i-1].nNodes_global)
      memset(multilevelMesh.meshArray[i].nodeMaterialTypes+multilevelMesh.meshArray[i-1].nNodes_global,
	     DEFAULT_NODE_MATERIAL,
	     (multilevelMesh.meshArray[i].nNodes_global-multilevelMesh.meshArray[i-1].nNodes_global)*sizeof(int));

    return 0;
  }
  int locallyRefineTriangleMesh_4T(MultilevelMesh& multilevelMesh, 
				   int * elementTagArray)
  {
    using namespace  std;
    //elementTagArray for now at least 
    //1 --> refine
    //0 --> don't
    int failed = 0;
    //first extend arrays and data structures in multilevelMesh
    int nLevelsPrev = multilevelMesh.nLevels;
    growMultilevelMesh(1,multilevelMesh);
    assert(multilevelMesh.nLevels == nLevelsPrev+1);
    int nElements_tagged = 0;
    for (int eN = 0; eN < multilevelMesh.meshArray[nLevelsPrev-1].nElements_global; eN++)
      {
	if (elementTagArray[eN] > 0)
	  nElements_tagged++;
      }

    vector<bool> refined(multilevelMesh.meshArray[nLevelsPrev-1].nElements_global,false);
    vector<int> edgeMidNodesArray(multilevelMesh.meshArray[nLevelsPrev-1].nElementBoundaries_global,-1);
    //new mesh starts off as parent mesh
    int nElements_new = multilevelMesh.meshArray[nLevelsPrev-1].nElements_global;
    int nNodes_new    = multilevelMesh.meshArray[nLevelsPrev-1].nNodes_global;

    //loop through tagged elements, subdivied all their edges and fix up
    //neighbors for conformity by subdividing their (necessary) edges
    for (int eN_parent = 0; 
	 eN_parent < multilevelMesh.meshArray[nLevelsPrev-1].nElements_global; eN_parent++)
      {
	//algorithm knows for now if it's refined an element or not
	if (elementTagArray[eN_parent] == 1)
	  {
	    failed = add4TnodesForRefinement2d(eN_parent,
					       nNodes_new,
					       refined,
					       edgeMidNodesArray,
					       multilevelMesh.meshArray[nLevelsPrev-1].elementNodesArray,
					       multilevelMesh.meshArray[nLevelsPrev-1].elementBoundariesArray,
					       multilevelMesh.meshArray[nLevelsPrev-1].elementNeighborsArray,
					       multilevelMesh.meshArray[nLevelsPrev-1].nodeArray);
	  }
      }
    /***********************************************************************
      Now have a set of tagged elements for refinement and new nodes to add at their
      edge midpoints

      4 cases to consider for each eN in parent
         0). element not tagged (0T)
      Otherwise if tagged
         1). all 3 element boundaries have been bisected (4T)
        
         2). 2 element boundaries have been bisected (3T)
 
         3). 1 element boundary has been bisected(2T)
           
    ***********************************************************************/
    for (int eN_parent = 0; 
	 eN_parent < multilevelMesh.meshArray[nLevelsPrev-1].nElements_global; eN_parent++)
      {
	int nBisectedEdges = 0;
	if (refined[eN_parent])
	  {
	    for (int ebN_local = 0; 
		 ebN_local < multilevelMesh.meshArray[nLevelsPrev-1].nElementBoundaries_element; ebN_local++)
	      {
		const int ebN = 
		  multilevelMesh.meshArray[nLevelsPrev-1].elementBoundariesArray[eN_parent*
										 multilevelMesh.meshArray[nLevelsPrev-1].nElementBoundaries_element+
										 ebN_local];
		if (edgeMidNodesArray[ebN] >= 0)
		  nBisectedEdges++;
	      }
	    //count based on cases
	    if (nBisectedEdges == 3)
	      nElements_new += 3; //4T lose parent
	    else if (nBisectedEdges == 2)
	      nElements_new += 2; //3T lose parent
	    else if (nBisectedEdges == 1)
	      nElements_new += 1; //2T lose parent
	  }
      }


    //go ahead and allocate memory now

    //multilevel genealogy
    //parent array : nElementsNew --> new level
    multilevelMesh.elementParentsArray[nLevelsPrev]    = new int[nElements_new];
    //initialize children offsets on previous finest mesh
    multilevelMesh.elementChildrenOffsets[nLevelsPrev-1] = 
      new int [multilevelMesh.meshArray[nLevelsPrev-1].nElements_global+1];
    multilevelMesh.elementChildrenArray[nLevelsPrev-1] = new int[nElements_new];
    multilevelMesh.elementChildrenOffsets[nLevelsPrev-1][0] = 0;

    //child mesh
    //now take care of mesh data structures
    multilevelMesh.meshArray[nLevelsPrev].nNodes_element   = 3;
    multilevelMesh.meshArray[nLevelsPrev].nElements_global = nElements_new;
    multilevelMesh.meshArray[nLevelsPrev].nNodes_global    = nNodes_new;
    
    //elementNodesArray
    multilevelMesh.meshArray[nLevelsPrev].elementNodesArray= 
      new int[multilevelMesh.meshArray[nLevelsPrev].nElements_global*3];
    //elementMaterialTypes
    multilevelMesh.meshArray[nLevelsPrev].elementMaterialTypes = 
      new int[multilevelMesh.meshArray[nLevelsPrev].nElements_global];
    //node array
    multilevelMesh.meshArray[nLevelsPrev].nodeArray = 
      new double[3*multilevelMesh.meshArray[nLevelsPrev].nNodes_global];

    //first copy over node information so that we can maintain (for now)
    //property that nodes and elements inherit their numbers if they arent
    //refined
        
    copy(multilevelMesh.meshArray[nLevelsPrev-1].nodeArray,
	 multilevelMesh.meshArray[nLevelsPrev-1].nodeArray+
	 3*multilevelMesh.meshArray[nLevelsPrev-1].nNodes_global,
	 multilevelMesh.meshArray[nLevelsPrev].nodeArray);

    //now go through and create nodes
    for (int ebN_parent = 0; ebN_parent < multilevelMesh.meshArray[nLevelsPrev-1].nElementBoundaries_global; ebN_parent++)
      {
	if (edgeMidNodesArray[ebN_parent] >= 0)
	  {
	    assert(edgeMidNodesArray[ebN_parent] >= multilevelMesh.meshArray[nLevelsPrev-1].nNodes_global);
	    Node midpoints[1];
	    const int nN0 = multilevelMesh.meshArray[nLevelsPrev-1].elementBoundaryNodesArray[ebN_parent*2+0];
	    const int nN1 = multilevelMesh.meshArray[nLevelsPrev-1].elementBoundaryNodesArray[ebN_parent*2+1];
	    midpoints[0].nN = edgeMidNodesArray[ebN_parent];
	    midpoint(multilevelMesh.meshArray[nLevelsPrev-1].nodeArray + nN0*3,
		     multilevelMesh.meshArray[nLevelsPrev-1].nodeArray + nN1*3,
		     midpoints[0]);
	    multilevelMesh.meshArray[nLevelsPrev].nodeArray[3*midpoints[0].nN+0]= midpoints[0].x;
	    multilevelMesh.meshArray[nLevelsPrev].nodeArray[3*midpoints[0].nN+1]= midpoints[0].y;
	    multilevelMesh.meshArray[nLevelsPrev].nodeArray[3*midpoints[0].nN+2]= midpoints[0].z;
	  }
      }
    //now take care of refining (or not) elements
    //put this in subdivide element routine
    int eN_new = multilevelMesh.meshArray[nLevelsPrev-1].nElements_global;
    bool subdivideFailed = false;
    for (int eN_parent = 0; 
	 eN_parent < multilevelMesh.meshArray[nLevelsPrev-1].nElements_global; eN_parent++)
      {
	subdivideFailed = subdivideTriangle4T(eN_parent,
					      eN_new,
					      multilevelMesh.elementParentsArray[nLevelsPrev],
					      multilevelMesh.elementChildrenOffsets[nLevelsPrev-1],
					      multilevelMesh.elementChildrenArray[nLevelsPrev-1],
					      multilevelMesh.meshArray[nLevelsPrev].elementNodesArray,
					      edgeMidNodesArray,
					      refined,
					      multilevelMesh.meshArray[nLevelsPrev-1].elementNodesArray,
					      multilevelMesh.meshArray[nLevelsPrev-1].elementBoundariesArray,
					      multilevelMesh.meshArray[nLevelsPrev-1].nodeArray);
	if (subdivideFailed)
	  return 1;
      }//parents
    //copy element material types from parents
    for (int eN = 0; eN < multilevelMesh.meshArray[nLevelsPrev].nElements_global; eN++)
      {
	int eN_parent = multilevelMesh.elementParentsArray[nLevelsPrev][eN];
	multilevelMesh.meshArray[nLevelsPrev].elementMaterialTypes[eN] = 
	  multilevelMesh.meshArray[nLevelsPrev-1].elementMaterialTypes[eN_parent];
      }
    //nodes on parent mesh retain ids
    copy(multilevelMesh.meshArray[nLevelsPrev-1].nodeMaterialTypes,
	 multilevelMesh.meshArray[nLevelsPrev-1].nodeMaterialTypes + multilevelMesh.meshArray[nLevelsPrev-1].nNodes_global,
	 multilevelMesh.meshArray[nLevelsPrev].nodeMaterialTypes);
    if (multilevelMesh.meshArray[nLevelsPrev].nNodes_global > multilevelMesh.meshArray[nLevelsPrev-1].nNodes_global)
      memset(multilevelMesh.meshArray[nLevelsPrev].nodeMaterialTypes+multilevelMesh.meshArray[nLevelsPrev-1].nNodes_global,
	     DEFAULT_NODE_MATERIAL,
	     (multilevelMesh.meshArray[nLevelsPrev].nNodes_global-multilevelMesh.meshArray[nLevelsPrev-1].nNodes_global)*sizeof(int));
    return 0;
  }
}//extern "C"

bool newestNodeBisect(int eN,
		      int& nElements_global,
		      int& nNodes_global,
		      std::vector<double>& nodeArray,
		      std::vector<int>& elementNodesArray,
		      std::vector<int>& elementNeighborsArray,
		      std::vector<std::list<int> >& childrenList,
		      std::vector<int>& elementParentsArray,
		      std::vector<int>& bases,
		      std::vector<bool>& refined)

{
  bool failed = false;
  const int nElementBoundaries_element = 3;
  const int nSpace = 2;
  /*********************
    nodeArray --> 3d not 2d
    ie --> eN, ibase --> ebN_base, simplexDim->nElementBoundaries_element
    nNodes--> nNodes_global, nElements--> nElements_global
    spaceDim --> nSpace
  ********************/  
  double x[3] = {0.0,0.0,0.0}; //node coordinates
  int ib[3];   //local node numbers starting with base
  int ibn[3];  //local node numbers starting with base for neigbhor
  int IB[3];   //global node numbers starting with base
  int E1[3],E2[3]; //global nodes of new triangles
  int N1[3],N2[3]; //global element neighbors of new triangles
  int E1n[3],E2n[3]; //global nodes of new triangles for neig
  int N1n[3],N2n[3]; //global element neighbors of new triangles for neig
  int ebN_base  = bases[eN]; //local base (edge number and node across from it)
  ib[0] = ebN_base; ib[1] = (ebN_base+1)%nElementBoundaries_element; ib[2]=(ebN_base+2)%nElementBoundaries_element;
  for (int i=0; i < nElementBoundaries_element; i++)
    IB[i] = elementNodesArray[nElementBoundaries_element*eN + ib[i]];

  //neighbor of base
  int eN_neig = elementNeighborsArray[nElementBoundaries_element*eN + ib[0]];
  assert(eN_neig < nElements_global);
  if (eN_neig < 0)
    {
      /**************************************************
       base is a boundary edge
       so just refine it by bisecting edge
      **************************************************/
      //create new node at center of base
      int newNodeNumber = nNodes_global;
      x[0] = 0.0; x[1] = 0.0; x[2] = 0.0;
      for (int nN = 0; nN < nSpace; nN++)//loop through nodes on base
	for (int I = 0; I < 3; I++)
	  x[I] += 0.5*nodeArray[3*IB[nN+1]+I];
#ifdef DEBUG_REFINE
      //std::cout<<"eN= "<<eN<<" base= "<<ebN_base<<" eN_neig= "<<eN_neig
      //<<" newNode nN= "<<newNodeNumber<<" x= ["<<x[0]
      //       <<","<<x[1]<<"]"<<std::endl;
#endif		    
      //insert new node, bm, at end of array, x coord then y coord
      for (int I = 0; I < 3; I++)
	nodeArray.push_back(x[I]);

      /***************************************************
         two new triangles to add:
           E1 = (base,base+1,bm)
           E2 = (bm,base+2,base)
         with neighbor arrays
           N1 = (-1,E2,N(base+2))
           N2 = (N(base+1),E1,-1)
         replace eN with E1
         Have to update interior neighbors' own neighbor info
      **************************************************/
      int newElementNumber = nElements_global;
      E1[0] = elementNodesArray[nElementBoundaries_element*eN + ib[0]]; 
      E1[1] = elementNodesArray[nElementBoundaries_element*eN + ib[1]];
      E1[2] = newNodeNumber;
      E2[0] = newNodeNumber; 
      E2[1] = elementNodesArray[nElementBoundaries_element*eN + ib[2]];
      E2[2] = elementNodesArray[nElementBoundaries_element*eN + ib[0]];

      N1[0] = -1;
      N1[1] = newElementNumber;//E2
      N1[2] = elementNeighborsArray[nElementBoundaries_element*eN + ib[2]];
      //don't have to change E1 neighbor because it inherits
      //old element number
      N2[0] = elementNeighborsArray[nElementBoundaries_element*eN + ib[1]];
      //find the neighbor across from 0, get its local id for
      //this element
      if (N2[0] >= 0)
	{
	  assert(N2[0] < nElements_global);
	  for (int ebN = 0; ebN < nElementBoundaries_element; ebN++)
	    {
	      if (elementNeighborsArray[nElementBoundaries_element*N2[0] + ebN] == eN)
		{
		  elementNeighborsArray[nElementBoundaries_element*N2[0] + ebN] = 
		    newElementNumber;//E2
		}
	    }
	}
      N2[1] = eN;//E1
      N2[2] = -1;

#ifdef DEBUG_REFINE
      //std::cout<<"eN= "<<eN<<" adding ";
      //std::cout<<"\n\t E1= ["<<E1[0]<<","<<E1[1]
      //       <<","<<E1[2]<<"]"<<std::endl;
      //std::cout<<"\n\t N1= ["<<N1[0]<<","<<N1[1]
      //       <<","<<N1[2]<<"]"<<std::endl;
      //std::cout<<"\n\t E2= ["<<E2[0]<<","<<E2[1]
      //      <<","<<E2[2]<<"]"<<std::endl;
      //std::cout<<"\n\t N2= ["<<N2[0]<<","<<N2[1]
      //       <<","<<N2[2]<<"]"<<std::endl;
#endif		    

      for (int ebN = 0; ebN < nElementBoundaries_element; ebN++)
	elementNodesArray[nElementBoundaries_element*eN + ebN]=E1[ebN]; //replace eN with E1
      //now append new element
      for (int ebN = 0; ebN < nElementBoundaries_element; ebN++)
	elementNodesArray.push_back(E2[ebN]);
      //same thing with neighbor array
      for (int ebN = 0; ebN < nElementBoundaries_element; ebN++)
	elementNeighborsArray[nElementBoundaries_element*eN + ebN]=N1[ebN]; //replace eN with E1
      for (int ebN = 0; ebN < nElementBoundaries_element; ebN++)
	elementNeighborsArray.push_back(N2[ebN]);
      
      //now update bases with local indeces of new node that was inserted
      bases[eN] = 2;   //E1
      bases.push_back(0); //E2

      //set parent and children
      if (unsigned(eN) < refined.size() && 
	  !refined[eN]) //refined is just for parent level
	{
	  elementParentsArray[eN] = eN; //E1 replaced eN
	  elementParentsArray.push_back(eN);//E2 is global element nElements_global (0 indexing)
	  childrenList[eN].push_back(eN);
	  childrenList[eN].push_back(newElementNumber);
	}
      else //eN has to be a "refined" element
	{
	  //eN has been refined already on this level, so get its parents
	  assert(unsigned(eN) < elementParentsArray.size());
	  //E1 remains unchanged, just get's element eN's parent
	  elementParentsArray.push_back(elementParentsArray[eN]);
	  //climb back up until find an element on the original mesh
	  int eN_tmp = eN;
	  while (unsigned(eN_tmp) >= refined.size())
	    eN_tmp = elementParentsArray[eN_tmp];
	  assert(unsigned(elementParentsArray[eN_tmp]) < childrenList.size());
	  //no need to insert or remove eN because it's already listed as a child
	  childrenList[eN_tmp].push_back(newElementNumber);
	}

      
      //update number of elements and nodes now
      nElements_global += 1;
      nNodes_global += 1;
      //refined is just for the original parent elements
      if (unsigned(eN) < refined.size())
	refined[eN] = true;
    }//end base on boundary
  else if (elementNeighborsArray[nElementBoundaries_element*eN_neig + bases[eN_neig]] == eN) 
    {
      /***************************************************
        neighboring element shares the same base
        so refine them both
      **************************************************/

      //create new node at center of base
      int newNodeNumber = nNodes_global;
      x[0] = 0.0; x[1] = 0.0; x[2] = 0.0;
      for (int nN = 0; nN < nSpace; nN++)//loop through nodes on base
	for (int I = 0; I < 3;  I++)
	  {
            x[I] += 0.5*nodeArray[3*IB[nN+1]+I];
          }
      //insert new node, bm, at end of array, x,y,z
      for (int I = 0; I < 3; I++)
	nodeArray.push_back(x[I]);

      /***************************************************
         two new triangles to add on each element.
         On eN: E1,E2 and on eN_neig: E1' and E2'
           E1 = (base,base+1,bm)
           E2 = (bm,base+2,base)
         with neighbor arrays
           N1 = (E2',E2,N(base+2))
           N2 = (N(base+1),E1,E1')
         replace eN with E1
         On eN_neig have:
           E1 = (base',base'+1,bm)
           E2 = (bm,base'+2,base')
         with neighbor arrays
           N1 = (E2,E2',N(base'+2))
           N2 = (N(base'+1),E1',E1)
 
      **************************************************/
      int ebN_base_neig = bases[eN_neig];
      ibn[0]=ebN_base_neig; ibn[1]=(ebN_base_neig+1)%nElementBoundaries_element; 
      ibn[2]=(ebN_base_neig+2)%nElementBoundaries_element;

      int newElementNumber = nElements_global;
      int newElementNumberNeig = nElements_global+1;
      E1[0] = elementNodesArray[nElementBoundaries_element*eN + ib[0]]; 
      E1[1] = elementNodesArray[nElementBoundaries_element*eN + ib[1]];
      E1[2] = newNodeNumber;//bm
      E2[0] = newNodeNumber;//bm 
      E2[1] = elementNodesArray[nElementBoundaries_element*eN + ib[2]];
      E2[2] = elementNodesArray[nElementBoundaries_element*eN + ib[0]];

      N1[0] = newElementNumberNeig; //E2'
      N1[1] = newElementNumber;     //E2
      N1[2] = elementNeighborsArray[nElementBoundaries_element*eN + ib[2]];
      //don't have to change E1 neighbor because it inherits
      //old element number

      N2[0] = elementNeighborsArray[nElementBoundaries_element*eN + ib[1]];
      //find the neighbor across from 0, get its local id for
      //this element
      if (N2[0] >= 0)
	{
	  assert(N2[0] < nElements_global);
	  for (int ebN = 0; ebN < nElementBoundaries_element; ebN++)
	    {
	      if (elementNeighborsArray[nElementBoundaries_element*N2[0] + ebN] == eN)
		{
		  elementNeighborsArray[nElementBoundaries_element*N2[0] + ebN] = 
		    newElementNumber;//E2
		}
	    }
	}
      N2[1] = eN;       //E1
      N2[2] = eN_neig;  //E1'

      //now neighbor
      E1n[0] = elementNodesArray[nElementBoundaries_element*eN_neig + ibn[0]]; 
      E1n[1] = elementNodesArray[nElementBoundaries_element*eN_neig + ibn[1]];
      E1n[2] = newNodeNumber;//bm
      E2n[0] = newNodeNumber;//bm 
      E2n[1] = elementNodesArray[nElementBoundaries_element*eN_neig + ibn[2]];
      E2n[2] = elementNodesArray[nElementBoundaries_element*eN_neig + ibn[0]];
      
      N1n[0] = newElementNumber;         //E2
      N1n[1] = newElementNumberNeig;     //E2'
      N1n[2] = elementNeighborsArray[nElementBoundaries_element*eN_neig + ibn[2]];
      //again don't have to update this neighbor because its going
      //to point to E1' <--- ieNeig

      N2n[0] = elementNeighborsArray[nElementBoundaries_element*eN_neig + ibn[1]];
      //find the neighbor across from 0, get its local id for
      //this element
      if (N2n[0] >= 0)
	{
	  assert(N2n[0] < nElements_global);
	  for (int ebN = 0; ebN < nElementBoundaries_element; ebN++)
	    {
	      if (elementNeighborsArray[nElementBoundaries_element*N2n[0] + ebN] == eN_neig)
		{
		  elementNeighborsArray[nElementBoundaries_element*N2n[0] + ebN] = 
		    newElementNumberNeig;//E2'
		}
	    }
	}

      N2n[1] = eN_neig; //E1'
      N2n[2] = eN;     //E1

      for (int ebN = 0; ebN < nElementBoundaries_element; ebN++)
	{
	  elementNodesArray[nElementBoundaries_element*eN + ebN]     =E1[ebN];  //replace eN with E1
	  elementNodesArray[nElementBoundaries_element*eN_neig + ebN]=E1n[ebN]; //replace eN_neig with E1'
	}
      //now append new element for eN
      for (int ebN = 0; ebN < nElementBoundaries_element; ebN++)
	elementNodesArray.push_back(E2[ebN]);
      //now append new element for neighbor
      for (int ebN = 0; ebN < nElementBoundaries_element; ebN++)
	elementNodesArray.push_back(E2n[ebN]);

      //same thing with neighbor array
      for (int ebN = 0; ebN < nElementBoundaries_element; ebN++)
	{
	  elementNeighborsArray[nElementBoundaries_element*eN + ebN]     =N1[ebN]; //replace ie with E1
	  elementNeighborsArray[nElementBoundaries_element*eN_neig + ebN]=N1n[ebN]; //replace eN_neig with E1'
	}
      for (int ebN = 0; ebN < nElementBoundaries_element; ebN++)
	elementNeighborsArray.push_back(N2[ebN]);
      for (int ebN = 0; ebN < nElementBoundaries_element; ebN++)
	elementNeighborsArray.push_back(N2n[ebN]);
      
      //now update bases with local indeces of new node that was inserted
      bases[eN]     = 2; //E1
      bases[eN_neig] = 2; //E1' 
      bases.push_back(0); //E2
      bases.push_back(0); //E2'

      //set parent and children for eN
      if (unsigned(eN) < refined.size() && 
	  !refined[eN]) //refined is just for parent level
	{
	  elementParentsArray[eN] = eN; //E1 replaced eN
	  elementParentsArray.push_back(eN);//E2 is global element nElements_global (0 indexing)
	  childrenList[eN].push_back(eN);
	  childrenList[eN].push_back(newElementNumber);
	}
      else //eN has to be a "refined" element
	{
	  //eN has been refined already on this level, so get its parents
	  assert(unsigned(eN) < elementParentsArray.size());
	  //E1 remains unchanged, just get's element eN's parent
	  elementParentsArray.push_back(elementParentsArray[eN]);
	  //climb back up until find an element on the original mesh
	  int eN_tmp = eN;
	  while (unsigned(eN_tmp) >= refined.size())
	    eN_tmp = elementParentsArray[eN_tmp];
	  assert(unsigned(elementParentsArray[eN_tmp]) < childrenList.size());
	  //no need to insert or remove eN because it's already listed as a child
	  childrenList[eN_tmp].push_back(newElementNumber);
	}
      //repeat for neighbor
      if (unsigned(eN_neig) < refined.size() && 
	  !refined[eN_neig]) //refined is just for parent level
	{
	  elementParentsArray[eN_neig] = eN_neig; //E1' replaced eN_neig
	  elementParentsArray.push_back(eN_neig);//E2' is global element nElements_global+1 (0 indexing)
	  childrenList[eN_neig].push_back(eN_neig); //E1'
	  childrenList[eN_neig].push_back(newElementNumberNeig); //E2'
	}
      else //eN_neig has to be a "refined" element
	{
	  //eN has been refined already on this level, so get its parents
	  assert(unsigned(eN_neig) < elementParentsArray.size());
	  //E1' remains unchanged, just get's element eN_neig's parent for E2'
	  elementParentsArray.push_back(elementParentsArray[eN_neig]);
	  //climb back up until find an element on the original mesh
	  int eN_tmp = eN_neig;
	  while (unsigned(eN_tmp) >= refined.size())
	    eN_tmp = elementParentsArray[eN_tmp];
	  assert(unsigned(elementParentsArray[eN_tmp]) < childrenList.size());
	  //no need to insert or remove eN because it's already listed as a child
	  childrenList[eN_tmp].push_back(newElementNumberNeig);
	}

      //update number of elements and nodes now
      nElements_global += 2;
      nNodes_global += 1;
      //refined is just for the original parent elements
      if (unsigned(eN) < refined.size())
	refined[eN]     = true;
      //refined is just for the original parent elements
      if (unsigned(eN_neig) < refined.size())
	refined[eN_neig] = true;
     
    }
  else
    {
      //recursive call for neighbor first
      failed = newestNodeBisect(eN_neig,
				nElements_global,
				nNodes_global,
				nodeArray,
				elementNodesArray,
				elementNeighborsArray,
				childrenList,
				elementParentsArray,
				bases,
				refined);
      failed = newestNodeBisect(eN,
				nElements_global,
				nNodes_global,
				nodeArray,
				elementNodesArray,
				elementNeighborsArray,
				childrenList,
				elementParentsArray,
				bases,
				refined);
    }
  return failed;
}

bool add4TnodesForRefinement2d(int eN,//element to be refined
			       int& nNodes_global,//number of nodes in mesh, will grow as refine
			       std::vector<bool>& refined,  //is an element to be refined or not?
			       std::vector<int>& edgeMidNodesArray,//edge--> new node from bisection (-1 = none)
			       const int* elementNodesArray,    //parent mesh representation
			       const int* elementBoundariesArray,
			       const int* elementNeighborsArray,
			       const double * nodeArray)
{
  bool failed = false;
  const int nElementBoundaries_element = 3; //2d only
  bool refinedAlready = refined[eN];
#ifdef DEBUG_REFINE
  //std::cout<<"Entering add4TnodesRef eN= "<<eN<<" refined = "<<refined[eN]<<std::endl;
#endif
  refined[eN] = true;
  int eN_longest_local = findLocalLongestEdge2d(eN,
						elementNodesArray,
						nodeArray);
  int eN_longest;
  eN_longest = elementBoundariesArray[eN*nElementBoundaries_element+eN_longest_local];
  for (int ebN_local = 0; ebN_local < nElementBoundaries_element; ebN_local++)
    {
      const int ebN = elementBoundariesArray[eN*nElementBoundaries_element+ebN_local];
#ifdef DEBUG_REFINE
      if (edgeMidNodesArray[ebN] >= 0 && !refinedAlready)
	{
	  //std::cout<<"WARNING add4Tnodes for refinement eN="<<eN<<" ebN_loc="<<ebN_local
	  //   <<" ebN= "<<ebN<<" already bisected edgeMidNodesArray[ebN] = "
	  //   <<edgeMidNodesArray[ebN]<<" not tagged as already refined"<<std::endl;
	}
#endif
      assert(edgeMidNodesArray[ebN] < 0 || refinedAlready);//otherwise this element should be tagged already
      if (edgeMidNodesArray[ebN] < 0)
	{
	  edgeMidNodesArray[ebN] = nNodes_global;
	  nNodes_global++;

	  int eN_neig = elementNeighborsArray[eN*nElementBoundaries_element+ebN_local];
	  if (eN_neig >= 0) //not physical boundary
	    {
	      //orig, used eN_longest for original call, but what if start with ebN
	      failed = add4TnodesForConformity2d(eN,ebN,//eN_longest,
						 ebN,eN_neig,
						 nNodes_global,
						 refined,
						 edgeMidNodesArray,
						 elementNodesArray,    //parent mesh representation
						 elementBoundariesArray,
						 elementNeighborsArray,
						 nodeArray);
	    }//has a neighbor across ebN
	}//edge not bisected already
    }//local element boundaries
  
  return failed;
}

bool add4TnodesForConformity2d(int eN, int ebN_longest, 
			       int ebN_neig,int eN_neig,
			       int& nNodes_global,
			       std::vector<bool>& refined,
			       std::vector<int>& edgeMidNodesArray,
			       const int* elementNodesArray,    //parent mesh representation
			       const int* elementBoundariesArray,
			       const int* elementNeighborsArray,
			       const double * nodeArray)
{
  /***********************************************************************
     here eN has been looked at, eN_longest is its longest edge and
     eN_neig is the neighbor across from ebN_neig
  ***********************************************************************/
  bool failed = false;
  //hardwire for 2d
  const int nElementBoundaries_element = 3;
  assert(eN >= 0);
#ifdef DEBUG_REFINE
  //mwf debug
  //std::cout<<"Entering add4Tnodes4Conf eN= "<<eN<<" refined = "<<refined[eN]
  //<<"eN_neig= "<<eN_neig<<" refined= ";
  //if (eN_neig >= 0)
    //std::cout<<refined[eN_neig]<<std::endl;
  //else
    //std::cout<<" on boundary "<<std::endl;
#endif
  //going to have to refine this one regardless
  refined[eN] = true;
  if (eN_neig < 0) //hit boundary edge
    {
      //ok to bisect eN using ebN because at boundary
      //assume eN already tagged for refinement?
      if (edgeMidNodesArray[ebN_neig] < 0)
	{
	  edgeMidNodesArray[ebN_neig] = nNodes_global;
	  nNodes_global++;
	}
      return failed;
    }
  assert(eN_neig >=0);
  int eN_neig_longest_local = findLocalLongestEdge2d(eN_neig,
						     elementNodesArray,
						     nodeArray);
  int ebN_neig_longest       = elementBoundariesArray[eN_neig*nElementBoundaries_element+eN_neig_longest_local];
  //going to have to refine neighbor regardless
  refined[eN_neig] = true;
  if (edgeMidNodesArray[ebN_neig_longest] < 0)
    {
      edgeMidNodesArray[ebN_neig_longest] = nNodes_global;
      nNodes_global++;
    }
  if (ebN_longest == ebN_neig_longest)
    {
      //means ebN_longest == ebN_neig
      assert(ebN_longest == ebN_neig);
      return failed; //done
    }

  int eN_neig_neig = elementNeighborsArray[eN_neig*nElementBoundaries_element+eN_neig_longest_local];
  failed = add4TnodesForConformity2d(eN_neig,ebN_neig_longest,
				     ebN_neig_longest,eN_neig_neig,
				     nNodes_global,
				     refined,
				     edgeMidNodesArray,
				     elementNodesArray,
				     elementBoundariesArray,
				     elementNeighborsArray,
				     nodeArray);
  return failed;
}
int findLocalLongestEdge2d(int eN,
			   const int* elementNodesArray,
			   const double * nodeArray)
{
  const int nElementBoundaries_element = 3;
  int longest = 0; double h_longest=0.0;
  int ebN = 0;
  int nN0 = elementNodesArray[eN*nElementBoundaries_element+((ebN+1)%nElementBoundaries_element)];
  int nN1 = elementNodesArray[eN*nElementBoundaries_element+((ebN+2)%nElementBoundaries_element)];
  double len = fabs((nodeArray[3*nN1+0]-nodeArray[3*nN0+0])*(nodeArray[3*nN1+0]-nodeArray[3*nN0+0])+
		    (nodeArray[3*nN1+1]-nodeArray[3*nN0+1])*(nodeArray[3*nN1+1]-nodeArray[3*nN0+1])+
		    (nodeArray[3*nN1+2]-nodeArray[3*nN0+2])*(nodeArray[3*nN1+2]-nodeArray[3*nN0+2]));
  
  longest = ebN; h_longest = len;
  for (ebN = 1; ebN < nElementBoundaries_element; ebN++)
    {
      nN0 = elementNodesArray[eN*nElementBoundaries_element+((ebN+1)%nElementBoundaries_element)];
      nN1 = elementNodesArray[eN*nElementBoundaries_element+((ebN+2)%nElementBoundaries_element)];
      len = fabs((nodeArray[3*nN1+0]-nodeArray[3*nN0+0])*(nodeArray[3*nN1+0]-nodeArray[3*nN0+0])+
		 (nodeArray[3*nN1+1]-nodeArray[3*nN0+1])*(nodeArray[3*nN1+1]-nodeArray[3*nN0+1])+
		 (nodeArray[3*nN1+2]-nodeArray[3*nN0+2])*(nodeArray[3*nN1+2]-nodeArray[3*nN0+2]));
 
      if (len > h_longest)
	{ len = h_longest; longest = ebN;}
    }
  return longest;
}

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
			 const double* nodeArray_parent)
{
  bool failed = false,u4T=false;//cek added u4T to test as alternative to rivara 4T
  //hardwire for 2d
  const int simplexDim = 3;
  const int childOffset = elementChildrenOffsets[eN_parent];
  if (!refined[eN_parent])
    {
      //copy over parent info
      for (int nN = 0; nN < simplexDim; nN++)
	{
	  elementNodesArray_child[eN_parent*simplexDim+nN] = 
	    elementNodesArray_parent[eN_parent*simplexDim+nN];
	}
      
      elementParentsArray[eN_parent] = eN_parent; //own parent
      elementChildrenOffsets[eN_parent+1]= childOffset+1;
      elementChildrenArray[childOffset] = eN_parent;//own child
    }
  else
    {
      //count number of refined edges
      int nBisectedEdges = 0;
      int midnodes[3],vnodes[3];
      for (int ebN_local = 0; ebN_local < simplexDim; ebN_local++)
	{
	  const int ebN = elementBoundariesArray_parent[eN_parent*simplexDim+ebN_local];
	  vnodes[ebN_local]   = elementNodesArray_parent[eN_parent*simplexDim+ebN_local];
	  midnodes[ebN_local] = edgeMidNodesArray[ebN];
	  if (edgeMidNodesArray[ebN] >= 0)
	    nBisectedEdges++;
	}
      assert(nBisectedEdges > 0);//shouldnt be refined if no bisected edges
      int refCase = 0;
      if (nBisectedEdges == 3)
	{
          //cek added uniform 4T to test difference with rivara 4T
          if (u4T)
            {
              refCase = 3;
              //4 children 
              elementChildrenOffsets[eN_parent+1]= childOffset+4;
	  
              //corners
              //center
              elementNodesArray_child[eN_parent*simplexDim+0]=
                midnodes[0];
              elementNodesArray_child[eN_parent*simplexDim+1]=
                midnodes[1];
              elementNodesArray_child[eN_parent*simplexDim+2]=
                midnodes[2];
              //parent/child info
              elementParentsArray[eN_parent] = eN_parent; 
              elementChildrenArray[childOffset+0] = eN_parent;
              for (int c=0;c<simplexDim;c++)
                {
                  elementNodesArray_child[eN_new*simplexDim+0]=
                    vnodes[c];
                  elementNodesArray_child[eN_new*simplexDim+1]=
                    midnodes[(c+1)%simplexDim];
                  elementNodesArray_child[eN_new*simplexDim+2]=
                    midnodes[(c+2)%simplexDim];
                  //parent/child info
                  elementParentsArray[eN_new] = eN_parent; //own parent
                  elementChildrenArray[childOffset+c+1] = eN_new;
                  eN_new++;
                } 
            }
          else
            {
              refCase = 3;
              /***************************************************
                   base = longest edge, 
                   nB   = also node across from it
                   note mid(nB,nB+1) = midnodes[nB+2] (across from other node)
                   new triangles:
                     (nB,mid(nB,nB+1),mid(nB+1,nB+2)   eN_parent
                     (mid(nB,nB+1),nB+1,mid(nB+1,nB+2) eN_new
                     (mid(nB+1,nB+2,nB+2,mid(nB+2,nB)) eN_new+1
                     (mid(nB+2,nB),nB,mid(nB+1,nB+2))  eN_new+2

                     
              **************************************************/
              int base = findLocalLongestEdge2d(eN_parent,
                                                elementNodesArray_parent,
                                                nodeArray_parent);
              assert(midnodes[base] >= 0);
              //4 children 
              elementChildrenOffsets[eN_parent+1]= childOffset+4;
	  
              //E0
              elementNodesArray_child[eN_parent*simplexDim+0]=
                vnodes[base];
              elementNodesArray_child[eN_parent*simplexDim+1]=
                midnodes[(base+2)%simplexDim];
              elementNodesArray_child[eN_parent*simplexDim+2]=
                midnodes[base];
              //parent/child info
              elementParentsArray[eN_parent] = eN_parent; //own parent
              elementChildrenArray[childOffset+0] = eN_parent; 
              //E1
              elementNodesArray_child[eN_new*simplexDim+0]=
                midnodes[(base+2)%simplexDim];
              elementNodesArray_child[eN_new*simplexDim+1]=
                vnodes[(base+1)%simplexDim];
              elementNodesArray_child[eN_new*simplexDim+2]=
                midnodes[base];
              //parent/child info
              elementParentsArray[eN_new] = eN_parent; 
              elementChildrenArray[childOffset+1] = eN_new;
              eN_new++;
              //E2
              elementNodesArray_child[eN_new*simplexDim+0]=
                midnodes[base];
              elementNodesArray_child[eN_new*simplexDim+1]=
                vnodes[(base+2)%simplexDim];
              elementNodesArray_child[eN_new*simplexDim+2]=
                midnodes[(base+1)%simplexDim];
              //parent/child info
              elementParentsArray[eN_new] = eN_parent; 
              elementChildrenArray[childOffset+2] = eN_new;
              eN_new++;
              //E3
              elementNodesArray_child[eN_new*simplexDim+0]=
                midnodes[(base+1)%simplexDim];
              elementNodesArray_child[eN_new*simplexDim+1]=
                vnodes[base];
              elementNodesArray_child[eN_new*simplexDim+2]=
                midnodes[base];
              //parent/child info
              elementParentsArray[eN_new] = eN_parent; 
              elementChildrenArray[childOffset+3] = eN_new;
              eN_new++;
            }
	}//4T
      else if (nBisectedEdges == 2)
	{
	  refCase = 2;
	  /***************************************************
                   base = longest edge, 
                   nB   = also node across from it
                   nn   = node across from other bisected edge
                   note mid(nB,nB+1) = midnodes[nB+2] (across from other node)
                   new triangles:
                     if nn=nB+1
                       (nB,nB+1,mid(nB+1,nB+2)             eN_parent
                       (nB,mid(nB+1,nB+2),mid(nB+2,nB))    eN_new
                       (mid(nB+1,nB+2),nB+2,mid(nB+2,nB))  eN_new+2
                     else
                       (nB+2,nB,mid(nB+1,nB+2)             eN_parent
                       (nB,mid(nB,nB+1),mid(nB+1,nB+2))    eN_new
                       (mid(nB,nB+1),nB+1,mid(nB+1,nB+2))  eN_new+2

                   
	  **************************************************/
	  int base = findLocalLongestEdge2d(eN_parent,
					    elementNodesArray_parent,
					    nodeArray_parent);
	  assert(midnodes[base] >= 0);
	  //3 children 
	  elementChildrenOffsets[eN_parent+1]= childOffset+3;

	  if (midnodes[(base+1)%simplexDim] < 0)//edge across from node nB+1 is not bisected
	    {
	      assert(midnodes[(base+2)%simplexDim] >= 0);
	      //nn = nB+2
	      //E0
	      elementNodesArray_child[eN_parent*simplexDim+0]=
		vnodes[(base+2)%simplexDim];
	      elementNodesArray_child[eN_parent*simplexDim+1]=
		vnodes[base];
	      elementNodesArray_child[eN_parent*simplexDim+2]=
		midnodes[base];
	      //parent/child info
	      elementParentsArray[eN_parent] = eN_parent; //own parent
	      elementChildrenArray[childOffset+0] = eN_parent; 
	      //E1
	      elementNodesArray_child[eN_new*simplexDim+0]=
		vnodes[base];
	      elementNodesArray_child[eN_new*simplexDim+1]=
		midnodes[(base+2)%simplexDim];
	      elementNodesArray_child[eN_new*simplexDim+2]=
		midnodes[base];
	      //parent/child info
	      elementParentsArray[eN_new] = eN_parent; 
	      elementChildrenArray[childOffset+1] = eN_new;
	      eN_new++;
	      //E2
	      elementNodesArray_child[eN_new*simplexDim+0]=
		midnodes[(base+2)%simplexDim];
	      elementNodesArray_child[eN_new*simplexDim+1]=
		vnodes[(base+1)%simplexDim];
	      elementNodesArray_child[eN_new*simplexDim+2]=
		midnodes[base];
	      //parent/child info
	      elementParentsArray[eN_new] = eN_parent; 
	      elementChildrenArray[childOffset+2] = eN_new;
	      eN_new++;
	    }
	  else//edge across from node nB+1 is bisected
	    {
	      assert(midnodes[(base+2)%simplexDim] < 0);
	      //nn = nB+1
	      //E0
	      elementNodesArray_child[eN_parent*simplexDim+0]=
		vnodes[base];
	      elementNodesArray_child[eN_parent*simplexDim+1]=
		vnodes[(base+1)%simplexDim];
	      elementNodesArray_child[eN_parent*simplexDim+2]=
		midnodes[base];
	      //parent/child info
	      elementParentsArray[eN_parent] = eN_parent; //own parent
	      elementChildrenArray[childOffset+0] = eN_parent; 
	      //E1
	      elementNodesArray_child[eN_new*simplexDim+0]=
		vnodes[base];
	      elementNodesArray_child[eN_new*simplexDim+1]=
		midnodes[base];
	      elementNodesArray_child[eN_new*simplexDim+2]=
		midnodes[(base+1)%simplexDim];
	      //parent/child info
	      elementParentsArray[eN_new] = eN_parent; 
	      elementChildrenArray[childOffset+1] = eN_new;
	      eN_new++;
	      //E2
	      elementNodesArray_child[eN_new*simplexDim+0]=
		midnodes[base];
	      elementNodesArray_child[eN_new*simplexDim+1]=
		vnodes[(base+2)%simplexDim];
	      elementNodesArray_child[eN_new*simplexDim+2]=
		midnodes[(base+1)%simplexDim];
	      //parent/child info
	      elementParentsArray[eN_new] = eN_parent; 
	      elementChildrenArray[childOffset+2] = eN_new;
	      eN_new++;
	    }//orientation switch
	}//3T
      else if (nBisectedEdges == 1)
	{
	  refCase = 1;
	  /***************************************************
                   base = longest edge, 
                   nB   = also node across from it
                   new triangles:
                     (nB,nB+1,mid(nB+1,nB+2))   eN_parent
                     (mid(nB+1,nB+2),nB+2,nB)   eN_new
                     
	  **************************************************/
	  int base = findLocalLongestEdge2d(eN_parent,
					    elementNodesArray_parent,
					    nodeArray_parent);
	  //2 children 
	  elementChildrenOffsets[eN_parent+1]= childOffset+2;
	  
	  assert(midnodes[base] >= 0);
	  //E0
	  elementNodesArray_child[eN_parent*simplexDim+0]=
	    vnodes[base];
	  elementNodesArray_child[eN_parent*simplexDim+1]=
	    vnodes[(base+1)%simplexDim];
	  elementNodesArray_child[eN_parent*simplexDim+2]=
	    midnodes[base];
	  //parent/child info
	  elementParentsArray[eN_parent] = eN_parent; //own parent
	  elementChildrenArray[childOffset+0] = eN_parent; 
	  //E1
	  elementNodesArray_child[eN_new*simplexDim+0]=
	    midnodes[base];
	  elementNodesArray_child[eN_new*simplexDim+1]=
	    vnodes[(base+2)%simplexDim];
	  elementNodesArray_child[eN_new*simplexDim+2]=
	    vnodes[base];
	  //parent/child info
	  elementParentsArray[eN_new] = eN_parent; 
	  elementChildrenArray[childOffset+1] = eN_new;
	  eN_new++;
	}//2T
#ifdef DEBUG_REFINE
      //std::cout<<"subdivide4T eN_parent = "<<eN_parent<<" case= "<<refCase<<std::endl;
      int child_start = elementChildrenOffsets[eN_parent]; 
      int child_end   = elementChildrenOffsets[eN_parent+1];
      //std::cout<<"child elements --- "<<std::endl;
      // for (int i = child_start; i < child_end; i++)
      // 	{
      // 	  int eN_child = elementChildrenArray[i];
      // 	  //std::cout<<"eN_child= "<<eN_child<<" nodes= ";
      // 	  //for (int nN = 0; nN < 3; nN++)
      // 	    //std::cout<<elementNodesArray_child[eN_child*3+nN]<<" ";
      // 	  //std::cout<<std::endl;
      // 	}
#endif
    }//element refined
  return failed;
}

// DMPlex and Pybind11 stuff
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
namespace py = pybind11;
// Function to read DMPlex mesh from MeshTools
int readDMPlexMesh(PyObject* dmplexMesh, Mesh& mesh)
{
  /***************************************************
   Access mesh info from DMPlex mesh set in MeshTools. 
   
   **************************************************/
  py::module_ meshTools = py::module_::import("proteus.MeshTools");
  py::object plexMesh = py::reinterpret_borrow<py::object>(dmplexMesh);

  if (py::isinstance(plexMesh, meshTools.attr("Mesh"))) {
    logEvent("DMPlex mesh is available in MeshTools.", 1);
    logEvent("Reading DMPlex mesh from MeshTools", 1);
    mesh.usePlex = true;
    mesh.nNodes_global = plexMesh.attr("nNodes_global").cast<int>();
    mesh.nElements_global = plexMesh.attr("nElements_global").cast<int>();    
    mesh.nElementBoundaries_global = plexMesh.attr("nElementBoundaries_global").cast<int>();
    mesh.nEdges_global = plexMesh.attr("nEdges_global").cast<int>();
    mesh.nNodes_element = plexMesh.attr("nNodes_element").cast<int>();
    mesh.nNodes_elementBoundary = plexMesh.attr("nNodes_elementBoundary").cast<int>();
    mesh.nElementBoundaries_element = plexMesh.attr("nElementBoundaries_element").cast<int>();
    mesh.nInteriorElementBoundaries_global = plexMesh.attr("nInteriorElementBoundaries_global").cast<int>();
    mesh.nExteriorElementBoundaries_global = plexMesh.attr("nExteriorElementBoundaries_global").cast<int>();
    
    mesh.max_nNodeNeighbors_node = plexMesh.attr("max_nNodeNeighbors_node").cast<int>();
    
    py::array_t<int> nodeElementsArray = plexMesh.attr("nodeElementsArray").cast<py::array_t<int>>();
    auto nodeElementsArrayBuf = nodeElementsArray.request();
    mesh.nodeElementsArray = static_cast<int*>(nodeElementsArrayBuf.ptr);
    
    py::array_t<int> elementNodesArray = plexMesh.attr("elementNodesArray").cast<py::array_t<int>>();
    auto elementNodesArrayBuf = elementNodesArray.request();
    mesh.elementNodesArray = static_cast<int*>(elementNodesArrayBuf.ptr);    
    
    py::array_t<int> nodeElementOffsets = plexMesh.attr("nodeElementOffsets").cast<py::array_t<int>>();
    auto nodeElementOffsetsBuf = nodeElementOffsets.request();
    mesh.nodeElementOffsets = static_cast<int*>(nodeElementOffsetsBuf.ptr);
    
    // py::array_t<int> elementNeighborsArray = plexMesh.attr("elementNeighborsArray").cast<py::array_t<int>>();
    // auto elementNeighborsArrayBuf = elementNeighborsArray.request();
    // mesh.elementNeighborsArray = static_cast<int*>(elementNeighborsArrayBuf.ptr);
    
    py::array_t<int> elementBoundariesArray = plexMesh.attr("elementBoundariesArray").cast<py::array_t<int>>();
    auto elementBoundariesArrayBuf = elementBoundariesArray.request();
    mesh.elementBoundariesArray = static_cast<int*>(elementBoundariesArrayBuf.ptr);
    
    // py::array_t<int> elementBoundaryNodesArray = plexMesh.attr("elementBoundaryNodesArray").cast<py::array_t<int>>();
    // auto elementBoundaryNodesArrayBuf = elementBoundaryNodesArray.request();
    // mesh.elementBoundaryNodesArray = static_cast<int*>(elementBoundaryNodesArrayBuf.ptr);
  //   std::cout << "elementBoundaryNodesArray:" << std::endl;
  // for (int ebN = 0; ebN < mesh.nElementBoundaries_global; ++ebN) {
  //     std::cout << "Boundary " << ebN << ":";
  //     for (int nN = 0; nN < mesh.nNodes_elementBoundary; ++nN) {
  //         std::cout << " " << mesh.elementBoundaryNodesArray[ebN * mesh.nNodes_elementBoundary + nN];
  //     }
  //     std::cout << std::endl;
  // }
  py::array_t<int> elementBoundaryElementsArray = plexMesh.attr("elementBoundaryElementsArray").cast<py::array_t<int>>();
    auto elementBoundaryElementsArrayBuf = elementBoundaryElementsArray.request();
    mesh.elementBoundaryElementsArray = static_cast<int*>(elementBoundaryElementsArrayBuf.ptr);
    
    // py::array_t<int> elementBoundaryLocalElementBoundariesArray = plexMesh.attr("elementBoundaryLocalElementBoundariesArray").cast<py::array_t<int>>();
    // auto elementBoundaryLocalElementBoundariesArrayBuf = elementBoundaryLocalElementBoundariesArray.request();
    // mesh.elementBoundaryLocalElementBoundariesArray = static_cast<int*>(elementBoundaryLocalElementBoundariesArrayBuf.ptr);
    
    py::array_t<int> interiorElementBoundariesArray = plexMesh.attr("interiorElementBoundariesArray").cast<py::array_t<int>>();
    auto interiorElementBoundariesArrayBuf = interiorElementBoundariesArray.request();
    mesh.interiorElementBoundariesArray = static_cast<int*>(interiorElementBoundariesArrayBuf.ptr);
    
    py::array_t<int> exteriorElementBoundariesArray = plexMesh.attr("exteriorElementBoundariesArray").cast<py::array_t<int>>();
    auto exteriorElementBoundariesArrayBuf = exteriorElementBoundariesArray.request();
    mesh.exteriorElementBoundariesArray = static_cast<int*>(exteriorElementBoundariesArrayBuf.ptr);
    
    py::array_t<int> edgeNodesArray = plexMesh.attr("edgeNodesArray").cast<py::array_t<int>>();
    auto edgeNodesArrayBuf = edgeNodesArray.request();
    mesh.edgeNodesArray = static_cast<int*>(edgeNodesArrayBuf.ptr);
    
    py::array_t<int> nodeStarArray = plexMesh.attr("nodeStarArray").cast<py::array_t<int>>();
    auto nodeStarArrayBuf = nodeStarArray.request();
    mesh.nodeStarArray = static_cast<int*>(nodeStarArrayBuf.ptr);
    
    py::array_t<int> nodeStarOffsets = plexMesh.attr("nodeStarOffsets").cast<py::array_t<int>>();
    auto nodeStarOffsetsBuf = nodeStarOffsets.request();
    mesh.nodeStarOffsets = static_cast<int*>(nodeStarOffsetsBuf.ptr);
    
    py::array_t<int> elementMaterialTypes = plexMesh.attr("elementMaterialTypes").cast<py::array_t<int>>();
    auto elementMaterialTypesBuf = elementMaterialTypes.request();
    mesh.elementMaterialTypes = static_cast<int*>(elementMaterialTypesBuf.ptr);
    
    py::array_t<int> elementBoundaryMaterialTypes = plexMesh.attr("elementBoundaryMaterialTypes").cast<py::array_t<int>>();
    auto elementBoundaryMaterialTypesBuf = elementBoundaryMaterialTypes.request();
    mesh.elementBoundaryMaterialTypes = static_cast<int*>(elementBoundaryMaterialTypesBuf.ptr);
    
    py::array_t<int> nodeMaterialTypes = plexMesh.attr("nodeMaterialTypes").cast<py::array_t<int>>();
    auto nodeMaterialTypesBuf = nodeMaterialTypes.request();
    mesh.nodeMaterialTypes = static_cast<int*>(nodeMaterialTypesBuf.ptr);
    
    py::array_t<double> nodeArray = plexMesh.attr("nodeArray").cast<py::array_t<double>>();
    auto nodeArrayBuf = nodeArray.request();
    mesh.nodeArray = static_cast<double*>(nodeArrayBuf.ptr);

    // py::array_t<double> elementDiametersArray = plexMesh.attr("elementDiametersArray").cast<py::array_t<double>>();
    // auto elementDiametersArrayBuf = elementDiametersArray.request();
    // mesh.elementDiametersArray = static_cast<double*>(elementDiametersArrayBuf.ptr);

    // py::array_t<double> elementInnerDiametersArray = plexMesh.attr("elementInnerDiametersArray").cast<py::array_t<double>>();
    // auto elementInnerDiametersArrayBuf = elementInnerDiametersArray.request();
    // mesh.elementInnerDiametersArray = static_cast<double*>(elementInnerDiametersArrayBuf.ptr);

    // py::array_t<double> elementBoundaryDiametersArray = plexMesh.attr("elementBoundaryDiametersArray").cast<py::array_t<double>>();
    // auto elementBoundaryDiametersArrayBuf = elementBoundaryDiametersArray.request();
    // mesh.elementBoundaryDiametersArray = static_cast<double*>(elementBoundaryDiametersArrayBuf.ptr);

    // py::array_t<double> elementBarycentersArray = plexMesh.attr("elementBarycentersArray").cast<py::array_t<double>>();
    // auto elementBarycentersArrayBuf = elementBarycentersArray.request();
    // mesh.elementBarycentersArray = static_cast<double*>(elementBarycentersArrayBuf.ptr);

    // py::array_t<double> elementBoundaryBarycentersArray = plexMesh.attr("elementBoundaryBarycentersArray").cast<py::array_t<double>>();
    // auto elementBoundaryBarycentersArrayBuf = elementBoundaryBarycentersArray.request();
    // mesh.elementBoundaryBarycentersArray = static_cast<double*>(elementBoundaryBarycentersArrayBuf.ptr);

    // py::array_t<double> nodeDiametersArray = plexMesh.attr("nodeDiametersArray").cast<py::array_t<double>>();
    // auto nodeDiametersArrayBuf = nodeDiametersArray.request();
    // mesh.nodeDiametersArray = static_cast<double*>(nodeDiametersArrayBuf.ptr);

    // py::array_t<double> nodeSupportArray = plexMesh.attr("nodeSupportArray").cast<py::array_t<double>>();
    // auto nodeSupportArrayBuf = nodeSupportArray.request();
    // mesh.nodeSupportArray = static_cast<double*>(nodeSupportArrayBuf.ptr);

    // mesh.h = plexMesh.attr("h").cast<double>();
    // mesh.hMin = plexMesh.attr("hMin").cast<double>();
    // mesh.sigmaMax = plexMesh.attr("sigmaMax").cast<double>();
    // mesh.volume = plexMesh.attr("volume").cast<double>();

  } else {
    std::cerr << "DMPlex mesh is not available in MeshTools." << std::endl;
    return -1;
  }
  return 0;
}

#ifdef MWF_HACK_2D_COARSE
int findGlueNeighbor2d(int eN,
		       std::vector<int>& elementNeighborsArray,
		       std::vector<int>& elementParentsArray)
{
  /*****************************************
     glue neighbor is neighbor that has same parent
  ****************************************/
  const int nElementBoundaries_element = 3;
  int eN_glue = -1;
  for (int ebN = 0; ebN < nElementBoundaries_element; ebN++)
    {
      const int eN_neig = elementNeighborsArray[eN*nElementBoundaries_element+ebN];
      if (eN_neig >= 0 && elementParentsArray[eN_neig] == elementParentsArray[eN])
	{
	  //neighbor has same parent
	  eN_glue = eN_neig;
	}
    }
  return eN_glue;
}
int findT2Neighbor(int eN, int eN_base,
		   std::vector<int>& elementNeighborsArray,
		   std::vector<int>& elementParentsArray)
{
  /*****************************************
     "other neighbor" t2, is the neighbor of eN that contains
       the base node but is not the glue neighbor
       That is, t2 is not across the from the base and is not the glue 
       neighbor 
  ****************************************/
  int eN_t2 = -1;
  for (int ebN = 0; ebN < nElementBoundaries_element; ebN++)
    {
      const int eN_neig = elementNeighborsArray[eN*nElementBoundaries_element+ebN];
      if (ebN != eN_base && eN_neig >= 0 && 
	  elementParentsArray[eN_neig] != elementParentsArray[eN])
	{
	  eN_t2 = eN_neig;
	}
    }
  return eN_t2;
}
//try to implement glue unrefinement (Algorithm 3) from Kossaczky 94
bool newestNodeGlue(int eN,
		    std::vector<bool>& mayCoarsen, //tag for current level
		    int& nElements_global,
		    int& nNodes_global,
		    std::vector<double>& nodeArray,
		    std::vector<int>& elementNodesArray,
		    std::vector<int>& elementNeighborsArray,
		    std::vector<int>& elementParentsArray,
		    std::vector<int>& bases,
		    std::vector<bool>& coarsened)
{
  using namespace std;
  bool failed = false;
  if (!mayCoarsen[eN])
    {failed = true; return failed;}
  //2d only
  const int nElementBoundaries_element = 3;
  bool connectable = false;
  int eN_base = bases[eN];
  int nN_global_base = elementNodesArray[eN*nElementBoundaries_element+eN_base];
  int eN_glue = -1, eN_t2 = -1;
  eN_glue = findGlueNeighbor(eN,
			     elementNeighborsArray,
			     elementParentsArray);
  eN_t2   = findT2Neighbor(eN,eN_base,
			   elementNeighborsArray,
			   elementParentsArray);
  if (eN_glue < 0)//no sibling found
    {failed = true; return failed;}
  if (eN_t2 >= 0 && elementNodesArray[eN_t2*nElementBoundaries_element+bases[eN_t2]] != nN_global_base)
    failed = newestNodeGlue(eN_t2,
			    mayCoarsen,
			    nElements_global,
			    nNodes_global,
			    nodeArray,
			    elementNodesArray,
			    elementNeighborsArray,
			    elementParentsArray,
			    bases,
			    coarsened);
  if (failed)
    return failed;
  //connectable if glue neighbor (ie sibling shares same base)
  connectable = (nN_global_base == elementNodesArray[eN_glue*nElementBoundaries_element+bases[eN_glue]]);
  if (!connectable)
    failed = newestNodeGlue(eN_glue,
			    mayCoarsen,
			    nElements_global,
			    nNodes_global,
			    nodeArray,
			    elementNodesArray,
			    elementNeighborsArray,
			    elementParentsArray,
			    bases,
			    coarsened);
  if (failed)
    return failed;
  //neighbor information may have changed, just t2?
  eN_glue = findGlueNeighbor(eN,
			     elementNeighborsArray,
			     elementParentsArray);
  if (eN_glue < 0)
    { failed =true; return failed;}
  eN_t2   = findT2Neighbor(eN,eN_base,
			   elementNeighborsArray,
			   elementParentsArray);
  bool t2connectable = false;
  //find glue neighbor of t2

  int eN_t2_glue = -1;
  if (eN_t2 >= 0)
    eN_t2_glue = findGlueNeighbor(eN_t2,
				  elementNeighborsArray,
				  elementParentsArray);

  if (eN_t2_glue < 0 && eN_t2 >= 0)
    {failed = true; return failed;}
  t2connectable = (eN_t2 >= 0 && 
		   (elementNodesArray[eN_t2*nElementBoundaries_element+bases[eN_t2]] ==
		    elementNodesArray[eN_t2_glue*nElementBoundaries_element+bases[eN_t2_glue]]));
  if (eN_t2 >= 0 && !t2connectable)
    failed = newestNodeGlue(eN_t2_glue,
			    mayCoarsen,
			    nElements_global,
			    nNodes_global,
			    nodeArray,
			    elementNodesArray,
			    elementNeighborsArray,
			    elementParentsArray,
			    bases,
			    coarsened);
  if (failed)
    return failed;
  t2connectable = false;
  if (eN_t2 >= 0)
    eN_t2_glue = findGlueNeighbor(eN_t2,
				  elementNeighborsArray,
				  elementParentsArray);
  if (eN_t2 >= 0 && eN_t2_glue >= 0)
    t2connectable = (elementNodesArray[eN_t2*nElementBoundaries_element+bases[eN_t2]] ==
		     elementNodesArray[eN_t2_glue*nElementBoundaries_element+bases[eN_t2_glue]]);
  if (t2connectable)
    failed = glueElements(eN_t2,eN_t2_glue,
			  mayCoarsen,
			  nElements_global,
			  nNodes_global,
			  nodeArray,
			  elementNodesArray,
			  elementNeighborsArray,
			  elementParentsArray,
			  bases,
			  coarsened);
  if (failed)
    return failed;
  failed = glueElements(eN,eN_glue,
			mayCoarsen,
			nElements_global,
			nNodes_global,
			nodeArray,
			elementNodesArray,
			elementNeighborsArray,
			elementParentsArray,
			bases,
			coarsened);
  
  return failed;
}

bool glueElements(int eN, int eN_glue,
		  std::vector<bool>& mayCoarsen, //tag for current level
		  int& nElements_global,
		  int& nNodes_global,
		  std::vector<double>& nodeArray,
		  std::vector<int>& elementNodesArray,
		  std::vector<int>& elementNeighborsArray,
		  std::vector<int>& elementParentsArray,
		  std::vector<int>& bases,
		  std::vector<bool>& coarsened)
{
  bool failed = true;
  //2d only
  const int nElementBoundaries_element = 3;
  assert(eN >= 0 && eN_glue >= 0);
  if (!mayCoarsen[eN] || !mayCoarsen[eN_glue])
    return failed;
  bool connectable = (elementNodesArray[eN*nElementBoundaries_element+bases[eN]] ==
		      elementNodesArray[eN_glue*nElementBoundaries_element+bases[eN_glue]]);
  if (!connectable)
    return failed;

  /*****************************************
     eN and eN_glue share the 
     same local base so have same newest node

    How to maintain counter clockwise order
      create new element
       (eN_base+1,eN_base+2,eN_glue_base+1)
    assuming elements were originally in 
    clockwise order
  ****************************************/
  int base_parent; 
  base_parent = (bases[eN]+1) % nElementBoundaries_element;
  int nN_global_gone = elementNodesArray[eN*nElementBoundaries_element+bases[eN]];
  //overwrite eN with parent
  for (int nN = 0; nN < nElementBoundaries_element; nN++)
    { 
      if (nN == bases[eN])
	elementNodesArray[eN*nElementBoundaries_element+ebN] = (bases[eN_glue]+1) % nElementBoundaries_element;
    }
  coarsened[eN] = true;
  //how to get rid of nodes in node array and keep track of global node numbers???
}		  
#endif //2d HACK COARSEN
/** @} */
