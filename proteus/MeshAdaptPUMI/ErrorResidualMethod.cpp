#include "MeshAdaptPUMI.h"
#include <PCU.h>
#include <petscksp.h>

#include <apf.h>
#include <apfMesh.h>
#include <apfShape.h>
#include <apfDynamicMatrix.h>
#include <apfNumbering.h>

#include <iostream>
#include <fstream>

/** 
 * \file ErrorResidualMethod.cpp
 * \ingroup MeshAdaptPUMI
 * @{
 */
//Global variables used to make it easier to pass these variables from MeshAdaptPUMIDrvr
int approx_order; //shape function order
int int_order; //integration order
double nu_0,nu_1,rho_0,rho_1;
double a_kl = 0.5; //flux term weight

inline void getProps(double*rho,double*nu)
//Function used to transfer MeshAdaptPUMIDrvr variables into global variables
{
  rho_0 = rho[0];
  nu_0 = nu[0];      
  rho_1 = rho[1];
  nu_1 = nu[1];
  return;
}

double MeshAdaptPUMIDrvr::getMPvalue(double field_val,double val_0, double val_1)
/**
 * @brief Function primarily used to get the VOF-weighted average of physical properties at a given point
 *
 * Due to the potential multi-phase nature of the problem, material properties at quadrature points
 * are weighted averages of the different phases.
 *
 * \param field_val VOF value
 * \param val_0 value of property for one phase
 * \param val_1 value of property for second phase
 */
{
  return val_0*(1-field_val)+val_1*field_val;
}

apf::Vector3 getFaceNormal(apf::Mesh* mesh, apf::MeshEntity* face)
//Function used to get the unit vector normal to an element face
{ 
  apf::Vector3 normal;
  apf::Adjacent verts;
  mesh->getAdjacent(face,0,verts);
  //apf::Vector3 vtxs[verts.getSize()];
  apf::Vector3 vtxs[3];
  for(int i=0;i<verts.getSize();i++){
    mesh->getPoint(verts[i],0,vtxs[i]); 
  } 
  apf::Vector3 a,b;
  if(mesh->getDimension()==2){
    vtxs[2] = vtxs[0];
    vtxs[2][2] = 1.0; 
  }
  a = vtxs[1]-vtxs[0];
  b = vtxs[2]-vtxs[0];
  normal = apf::cross(a,b);

  return normal.normalize();
}

double getDotProduct(apf::Vector3 a, apf::Vector3 b)
//Function used to get the dot product between two vectors
{
  return (a[0]*b[0] + a[1]*b[1] + a[2]*b[2]);
}

double getDotProduct(apf::Matrix3x3 a, apf::Matrix3x3 b)
//Overloaded function used to get the dot product analog between two matrices
{
  double temp =0;
  for(int i=0;i<3;i++){
    for(int j=0;j<3;j++){
      temp = temp + a[i][j]*b[i][j];
    }
  }
  return temp;
}


bool isInTet(apf::Mesh* mesh, apf::MeshEntity* ent, apf::Vector3 pt)
//Function used to test if a point pt is inside the tetrahedron ent
//Returns a boolean: 1 if is in tet, 0 if not
{
  bool isin=0;

  apf::Adjacent verts;
  mesh->getAdjacent(ent,0,verts);
  apf::Vector3 vtxs[4]; //4 points
  for(int i=0;i<4;i++){
    mesh->getPoint(verts[i],0,vtxs[i]); 
  } 
  apf::Vector3 c[4];
  c[0] = vtxs[1]-vtxs[0];
  c[1] = vtxs[2]-vtxs[0];
  c[2] = vtxs[3]-vtxs[0];
  c[3] = pt-vtxs[0];

  apf::Matrix3x3 K,Kinv;
  apf::Vector3 F;
  for(int i=0;i<3;i++){
    for(int j=0;j<3;j++){
      K[i][j] = getDotProduct(c[i],c[j]);
    }
    F[i] = getDotProduct(c[3],c[i]);
  }
  Kinv = invert(K);
  apf::DynamicMatrix Kinv_dyn = apf::fromMatrix(Kinv);
  apf::DynamicVector F_dyn = apf::fromVector(F);
  apf::DynamicVector uvw; //result
  apf::multiply(Kinv_dyn,F_dyn,uvw);
  if(uvw[0] >= 0 && uvw[1] >=0 && uvw[2] >=0 && (uvw[0]+uvw[1]+uvw[2])<=1) isin = 1;
  return isin;
}

bool isInSimplex(apf::Mesh* mesh, apf::MeshEntity* ent, apf::Vector3 pt, int dim)
//Function used to test if a point pt is inside the tetrahedron ent
//Returns a boolean: 1 if is in tet, 0 if not
{
  bool isin=0;
  int numverts = dim+1;
  apf::Adjacent verts;
  mesh->getAdjacent(ent,0,verts);
  apf::Vector3 vtxs[4];
  for(int i=0;i<numverts;i++){
    mesh->getPoint(verts[i],0,vtxs[i]); 
  } 
  apf::Vector3 c[4];
  if(dim==2){
    c[0] = vtxs[1]-vtxs[0];
    c[1] = vtxs[2]-vtxs[0];
    c[2] = pt-vtxs[0];
  }
  else if(dim==3){
    c[0] = vtxs[1]-vtxs[0];
    c[1] = vtxs[2]-vtxs[0];
    c[2] = vtxs[3]-vtxs[0];
    c[3] = pt-vtxs[0];
  }

  apf::Matrix3x3 K(0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0);
  apf::Matrix3x3 Kinv;
  apf::Vector3 F(0.0,0.0,0.0);
  for(int i=0;i<dim;i++){
    for(int j=0;j<dim;j++){
      K[i][j] = getDotProduct(c[i],c[j]);
    }
    F[i] = getDotProduct(c[dim],c[i]);
  }
  if(dim==2)
    K[2][2] = 1.0;
  Kinv = invert(K);
  apf::DynamicMatrix Kinv_dyn = apf::fromMatrix(Kinv);
  apf::DynamicVector F_dyn = apf::fromVector(F);
  apf::DynamicVector uvw; //result
  apf::multiply(Kinv_dyn,F_dyn,uvw);
  if(dim==2){
    if(uvw[0] >= 0 && uvw[1] >=0 && (uvw[0]+uvw[1])<=1) isin = 1;
  }
  else{
    if(uvw[0] >= 0 && uvw[1] >=0 && uvw[2] >=0 && (uvw[0]+uvw[1]+uvw[2])<=1) isin = 1;
  }
  return isin;
}


/*
double a_k(apf::Matrix3x3 u, apf::Matrix3x3 v,double nu){
  //u and v are gradients of a vector
  apf::Matrix3x3 temp_u = u+apf::transpose(u);
  apf::Matrix3x3 temp_v = v+apf::transpose(v);
  return nu*getDotProduct(temp_u,temp_v);
} 

double b_k(double a, apf::Matrix3x3 b){
  //b is a gradient of a vector
  return a*(b[0][0]+b[1][1]+b[1][1]);
}

double c_k(apf::Vector3 a, apf::Matrix3x3 b, apf::Vector3 c){
  //b is a gradient of a vector
  return getDotProduct(b*a,c);
}
*/

void getLHS(Mat &K,apf::NewArray <apf::DynamicVector> &shdrv,int nsd,double weight, double visc_val,int nshl)
//Function used to get the LHS of the local error problem. 
//The LHS is entirely A(\phi,\phi) which can be decomposed into a diagonal contributions and off-diagonal contributions
//Inputs:
//  shdrv is the set of shape function derivatives for an element evaluated at a quadrature point
//  nsd is the number of spatial dimensions
//  weight is the corresponding weight for a given quadrature point
//  visc_val is the viscosity at that quadrature point
//  nshl is the number of local shape functions in an element
//Outputs:
//  K is the matrix representing the LHS
{
      PetscScalar term1[nshl][nshl], term2[nshl][nshl];
      //Calculate LHS Diagonal Block Term
      for(int s=0; s<nshl;s++){
        for(int t=0; t<nshl;t++){
          double temp=0;
          for(int j=0;j<nsd;j++){
            temp+=shdrv[s][j]*shdrv[t][j];
          }
          term1[s][t] = temp*weight*visc_val;
        }
      } 
      PetscInt idx[nshl]; //indices for PETSc Mat insertion
      for(int i=0; i<nsd;i++){
        for(int j=0;j<nshl;j++){
          idx[j] = i*nshl+j;
        }
        MatSetValues(K,nshl,idx,nshl,idx,term1[0],ADD_VALUES);
      }
      PetscInt idxr[nshl],idxc[nshl]; //indices for PETSc rows and columns
      for(int i = 0; i< nsd;i++){
        for(int j=0; j< nsd;j++){
          for(int s=0;s<nshl;s++){
            for(int t=0;t<nshl;t++){
              term2[s][t] = shdrv[s][j]*shdrv[t][i]*weight*visc_val;
            }
          }
          for(int count=0;count<nshl;count++){
            idxr[count] = i*nshl+count;
            idxc[count] = j*nshl+count;
          }
          MatSetValues(K,nshl,idxr,nshl,idxc,term2[0],ADD_VALUES);
        }
      } //end 2nd term loop
}

void getRHS(Vec &F,apf::NewArray <double> &shpval,apf::NewArray <apf::DynamicVector> &shdrv,apf::Vector3 vel_vect,apf::Matrix3x3 grad_vel,
            int nsd,double weight,int nshl,
            double visc_val,double density,apf::Vector3 grad_density,double pressure,
            double g[3])
//Function used to get the RHS of the local error problem. 
//The RHS is the weak residual of the N-S equations
//Inputs:
//  shpval are the local shape functions evaluated at a quadrature point
//  shdrv are the local shape function derivatives evaluated at a quadrature point
//  vel_vect is the velocity vector at a quadrature point
//  grad_vel is the velocity gradient at a quadrature point
//  nsd is the number of spatial dimensions
//  weight is the corresponding weight for a given quadrature point
//  nshl is the number of local shape functions in an element
//  visc_val is the viscosity at a quadrature point
//  density is the density at a quadrature point
//  grad_density is the density gradient at a quadrature point
//  pressure is the pressure at a quadrature point
//  g is the gravity vector
//Outputs:
//  F is the vector representing the RHS

{
      PetscInt idx[nshl];
      for( int i = 0; i<nsd; i++){
        double temp_vect[nshl];
        for( int s=0;s<nshl;s++){
          idx[s] = i*nshl+s;

          //forcing term
          //temp_vect[s] = (g[i]+0.0)*shpval[s];
          //temp_vect[s] += pressure/density*shdrv[s][i]; //pressure term
          double force = (g[i]+0.0)*shpval[s];
          double pressure_force = pressure/density*shdrv[s][i];
          double a_rho_term = 0;
          double b_rho_term = -pressure/(density*density)*grad_density[i]*shpval[s];
          double a_term = 0; 
          double c_term = 0;
          //a(u,v) and c(u,u,v) term
          for(int j=0;j<nsd;j++){
            a_term += -visc_val*shdrv[s][j]*(grad_vel[i][j]+grad_vel[j][i]);
            a_rho_term += visc_val*(grad_vel[i][j]+grad_vel[j][i])*shpval[s]*grad_density[j]/(density);
            c_term += -shpval[s]*grad_vel[i][j]*vel_vect[j];///density;
          }
          temp_vect[s] = force+pressure_force+a_term+c_term; 
          //temp_vect[s] = force+pressure_force+a_rho_term+b_rho_term+a_term+c_term;
          temp_vect[s] = temp_vect[s]*weight;
        } //end loop over number of shape functions
        VecSetValues(F,nshl,idx,temp_vect,ADD_VALUES);
      } //end loop over spatial dimensions
}


void MeshAdaptPUMIDrvr::computeDiffusiveFlux(apf::Mesh*m,apf::Field* voff, apf::Field* visc,apf::Field* pref, apf::Field* velf)
/** 
 * @brief Function used to compute the diffusive flux at interelement boundaries and stores the values as tags at the boundaries
 *
 * Based on the mesh database, each face has a default orientation that is outward normal to a given element.
 * The flux has a different value depending on the element.
 * The value from the default element is stored in the first slot of the tag and the other value is stored in the second slot
 * Special considerations are made for global domain boundaries if boundary conditions exist and for parallel communications
 * Inputs:
 * \param m is the mesh
 * \param voff is the volume of fluid field
 * \param visc is the field of viscosity
 * \param pref is the pressure field
 * \param velf is the velocity field
 */
{
  if(comm_rank==0)
    std::cerr<<"Begin computeDiffusiveFlux()"<<std::endl;
  int numbqpt, nshl;
  int hier_off;
  if(nsd==2)
    hier_off=3;
  else if(nsd==3)
    hier_off=4;
  apf::MeshEntity* bent,*ent;
  apf::MeshIterator* iter = m->begin(nsd-1); //loop over faces

  //Need to get number of bqpt
  while(bent = m->iterate(iter)){
    apf::MeshElement* b_elem;
    b_elem = apf::createMeshElement(m,bent);
    numbqpt = apf::countIntPoints(b_elem,int_order);
    apf::destroyMeshElement(b_elem);
    break;
  }
  m->end(iter);
    
  diffFlux = m->createDoubleTag("diffFlux",numbqpt*nsd*2);
  apf::MeshElement* tempelem; apf::Element * tempvelo,*temppres,*tempvoff;
  apf::MeshElement* b_elem;
  apf::Adjacent adjFaces;
  apf::Vector3 normal,centerdir;
  int orientation;
  double tempflux[numbqpt*nsd];
  double *flux; flux = (double*) calloc(numbqpt*nsd*2,sizeof(double));
  apf::NewArray <double> shpval;
  apf::NewArray <double> shpval_temp;

  apf::FieldShape* err_shape = apf::getHierarchic(2);
  apf::EntityShape* elem_shape;
  apf::Vector3 bqpt,bqptl,bqptshp;
  double weight, Jdet;
  apf::Matrix3x3 J;
  apf::Matrix3x3 tempgrad_velo;
  apf::Matrix3x3 identity(1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0);
  
  iter=m->begin(nsd-1);
  while(ent=m->iterate(iter))
  {
    m->setDoubleTag(ent,diffFlux,flux);
  }
  m->end(iter);
  free(flux);

  if(comm_rank==0)
   std::cerr<<"Initialized flux"<<std::endl;
  //loop over regions
  PCU_Comm_Begin(PCUObj);
  iter = m->begin(nsd);
  int ent_count=0;
  while(ent = m->iterate(iter))
  { 
    //Shape functions of the region and not the boundaries
    if(ent_count==0){
      nshl=apf::countElementNodes(err_shape,m->getType(ent));
      shpval_temp.allocate(nshl);
      nshl= nshl-hier_off;
      shpval.allocate(nshl);
      elem_shape = err_shape->getEntityShape(m->getType(ent));
    }

    m->getAdjacent(ent,nsd-1,adjFaces);
    for(int adjcount =0;adjcount<adjFaces.getSize();adjcount++){
      bent = adjFaces[adjcount];
      normal=getFaceNormal(m,bent);
      centerdir=apf::getLinearCentroid(m,ent)-apf::getLinearCentroid(m,bent);
      //if(isInTet(m,ent,apf::project(normal,centerdir)*centerdir.getLength()+apf::getLinearCentroid(m,bent)))
      if(isInSimplex(m,ent,apf::project(normal,centerdir).normalize()*centerdir.getLength()+apf::getLinearCentroid(m,bent),nsd)){
        orientation = 1;
      }
      else{
        orientation = 0;
      }

      //begin calculation of flux
      b_elem = apf::createMeshElement(m,bent);
      tempelem = apf::createMeshElement(m,ent);
      temppres = apf::createElement(pref,tempelem);
      tempvelo = apf::createElement(velf,tempelem);
      tempvoff = apf::createElement(voff,tempelem);

      for(int l = 0;l<numbqpt;l++)
      {
        apf::Vector3 bflux(0.0,0.0,0.0); 
        apf::Matrix3x3 tempbflux(0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0);
        apf::getIntPoint(b_elem,int_order,l,bqpt);
        weight = apf::getIntWeight(b_elem,int_order,l);
        apf::getJacobian(b_elem,bqpt,J); //evaluate the Jacobian at the quadrature point
        Jdet=fabs(apf::getJacobianDeterminant(J,nsd-1));
        bqptl=apf::boundaryToElementXi(m,bent,ent,bqpt); 
        apf::getVectorGrad(tempvelo,bqptl,tempgrad_velo);
        tempgrad_velo = apf::transpose(tempgrad_velo);

        apf::ModelEntity* me=m->toModel(bent);
        int tag = m->getModelTag(me);
        apf::ModelEntity* boundary_face = m->findModelEntity(nsd-1,tag);

        if(me==boundary_face && has_gBC){
          int BCtype[4];
          double fluxdata[4][numbqpt];
          //for(int i=1;i<nsd+1;i++){ //ignores 0th index because that's pressure
          //  m->getIntTag(bent,BCtag[i],&(BCtype[i]));                 
          //}
          m->getIntTag(bent,BCtag,&(BCtype[0]));                 
          if((BCtype[1]+BCtype[2]+BCtype[3] != 3) && BCtype[1] == 1 ){
            std::cerr << "diffusive flux not fully specified on face " << localNumber(bent) << '\n';
            std::cerr << "BCtype "<<BCtype[1]<<" "<<BCtype[2]<<" "<<BCtype[3]<<std::endl;
            abort();
          }        
          if(BCtype[1]+BCtype[2]+BCtype[3] == 3){
            for(int i=1;i<nsd+1;i++)
              m->getDoubleTag(bent,fluxtag[i],&(fluxdata[i][0]));
            bflux = apf::Vector3(fluxdata[1][l],fluxdata[2][l],fluxdata[3][l]);
            bflux = bflux-identity*apf::getScalar(temppres,bqptl)/getMPvalue(apf::getScalar(tempvoff,bqptl),rho_0,rho_1)*normal;
          }
          else{
            tempbflux = (tempgrad_velo+apf::transpose(tempgrad_velo))*getMPvalue(apf::getScalar(tempvoff,bqptl),nu_0,nu_1)
                   -identity*apf::getScalar(temppres,bqptl)/getMPvalue(apf::getScalar(tempvoff,bqptl),rho_0,rho_1);
            bflux = tempbflux*normal;
          }
        }
        else{
          tempbflux = (tempgrad_velo+apf::transpose(tempgrad_velo))*getMPvalue(apf::getScalar(tempvoff,bqptl),nu_0,nu_1)
              -identity*apf::getScalar(temppres,bqptl)/getMPvalue(apf::getScalar(tempvoff,bqptl),rho_0,rho_1);
          bflux = tempbflux*normal;
        } //end if boundary
        bflux = bflux*weight*Jdet;
        // bflux.toArray() always writes all 3 components of an apf::Vector3,
        // regardless of nsd -- tempflux is only sized/strided for nsd components
        // per quadrature point, so using toArray() here overflows it by one
        // component per point when nsd==2 (harmless no-op when nsd==3, since the
        // stride then matches the 3 components written).
        for(int d = 0; d < nsd; d++) tempflux[l*nsd+d] = bflux[d];
      }
      flux = (double*) calloc(numbqpt*nsd*2,sizeof(double));
      m->getDoubleTag(bent,diffFlux,flux);
      for (int i=0;i<numbqpt*nsd;i++){
        flux[orientation*numbqpt*nsd+i] = tempflux[i];
      }
      m->setDoubleTag(bent,diffFlux,flux);
      free(flux);
      apf::destroyMeshElement(tempelem);apf::destroyElement(tempvelo);apf::destroyElement(temppres); apf::destroyElement(tempvoff);
      
      //Parallel Communications
      apf::ModelEntity* me=m->toModel(bent);
      apf::ModelEntity* boundary_face = m->findModelEntity(nsd-1,m->getModelTag(me));
      apf::Copies remotes;
      if(m->isShared(bent))
      {
        m->getRemotes(bent,remotes);
        for(apf::Copies::iterator it=remotes.begin(); it!=remotes.end();++it)
        {
          PCU_COMM_PACK(PCUObj, it->first, it->second);
          PCU_COMM_PACK(PCUObj, it->first, orientation);
          PCU_COMM_PACK(PCUObj, it->first, tempflux);
        }
      } //end if
      ent_count++;
    } //end loop over faces
  } //end loop over regions
  m->end(iter);
  if(comm_rank==0)
    std::cerr<<"Sending flux"<<std::endl;
  PCU_Comm_Send(PCUObj); 
  flux = (double*) calloc(numbqpt*nsd*2,sizeof(double));
  while(PCU_Comm_Receive(PCUObj))
  {
    PCU_COMM_UNPACK(PCUObj, bent);
    PCU_COMM_UNPACK(PCUObj, orientation);
    PCU_COMM_UNPACK(PCUObj, tempflux);
    m->getDoubleTag(bent,diffFlux,flux);
    for (int i=0;i<numbqpt*nsd;i++){
      flux[orientation*numbqpt*nsd+i] = flux[orientation*numbqpt*nsd+i]+tempflux[i];
    }
    m->setDoubleTag(bent,diffFlux,flux);
  }
  PCU_Barrier(PCUObj);
  free(flux);
  if(comm_rank==0)
    std::cerr<<"End computeDiffusiveFlux()"<<std::endl;
}

void MeshAdaptPUMIDrvr::getBoundaryFlux(apf::Mesh* m, apf::MeshEntity* ent, double * endflux)
/** 
 * @brief This function reads in the stored tags and computes the boundary flux according to the RHS formulation
 * 
 * \param m is the input mesh
 * \param ent is an input element (tetrahedron)
 * \param endflux is the output boundary flux used in the RHS 
 */
{
    int nshl;
    apf::NewArray <double> shpval;
    apf::NewArray <double> shpval_temp;
    double* flux;

    apf::FieldShape* err_shape = apf::getHierarchic(2);
    apf::EntityShape* elem_shape;

    //loop over element faces
    apf::Adjacent boundaries;
    apf::MeshEntity* bent;
    apf::MeshElement* b_elem;
    apf::Vector3 bqpt,bqptl,bqptshp;

    apf::Vector3 normal;
    apf::Vector3 centerdir;

    //Shape functions of the region and not the boundaries
    nshl=apf::countElementNodes(err_shape,m->getType(ent));
    shpval_temp.allocate(nshl);
    int hier_off;
    if(nsd==2)
      hier_off=3;
    else if(nsd==3)
      hier_off=4;
    nshl= nshl-hier_off;
    shpval.allocate(nshl);
    elem_shape = err_shape->getEntityShape(m->getType(ent));

    m->getAdjacent(ent,nsd-1,boundaries);
    for(int adjcount =0;adjcount<boundaries.getSize();adjcount++){

      apf::Vector3 bflux(0.0,0.0,0.0); 
      bent = boundaries[adjcount];
          
      b_elem = apf::createMeshElement(m,bent);
      normal=getFaceNormal(m,bent);
      centerdir=apf::getLinearCentroid(m,ent)-apf::getLinearCentroid(m,bent);
      int orientation = 0;
      
      if(isInSimplex(m,ent,apf::project(normal,centerdir).normalize()*centerdir.getLength()+apf::getLinearCentroid(m,bent),nsd)){
            normal = normal*-1.0; //normal needs to face the other direction
            orientation=1;
      }
      apf::ModelEntity* me=m->toModel(bent);
      int tag = m->getModelTag(me);
      apf::ModelEntity* boundary_face = m->findModelEntity(nsd-1,tag);

      double flux_weight[2];         
      if(me==boundary_face){
        if(orientation==0){flux_weight[0]=1; flux_weight[1]=0;}
        else{flux_weight[0]=0; flux_weight[1]=-1;}
      }
      else{
        if(orientation==0){flux_weight[0]=(1-a_kl); flux_weight[1]=a_kl;}
        else{ flux_weight[0]=-a_kl; flux_weight[1]=-1*(1-a_kl);}
      }

      int numbqpt = apf::countIntPoints(b_elem,int_order); 
      flux = (double*) calloc(numbqpt*nsd*2,sizeof(double));
      m->getDoubleTag(bent,diffFlux,flux);
      for(int l=0; l<numbqpt;l++){
        apf::getIntPoint(b_elem,int_order,l,bqpt);
        bqptshp=apf::boundaryToElementXi(m,bent,ent,bqpt); 
        elem_shape->getValues(NULL,NULL,bqptshp,shpval_temp);
        for(int j=0;j<nshl;j++){shpval[j] = shpval_temp[hier_off+j];}
        for(int i=0;i<nsd;i++){
          for(int s=0;s<nshl;s++){
            endflux[i*nshl+s] = endflux[i*nshl+s]+(flux_weight[0]*flux[l*nsd+i]+flux_weight[1]*flux[numbqpt*nsd+l*nsd+i])*shpval[s];
          }
        }
      }//end of boundary integration loop
      free(flux);
   } //end for adjacent faces
}

apf::Field* MeshAdaptPUMIDrvr::getViscosityField(apf::Field* voff)
/** 
 * @brief Function used to derive a viscosity field from a VOF field
 *
 * For convenience, derive a viscosity field using the getMPvalue() function. 
 * \param voff is the VOF field. 
 */
{
  apf::Field* visc = apf::createLagrangeField(m,"viscosity",apf::SCALAR,1);
  apf::MeshEntity* ent;
  apf::MeshIterator* iter = m->begin(0);
  double vof_val, visc_val;
  while(ent = m->iterate(iter)){ //loop through all vertices
    vof_val=apf::getScalar(voff,ent,0);
    visc_val = getMPvalue(vof_val,nu_0, nu_1);
    apf::setScalar(visc, ent, 0,visc_val);
  }
  m->end(iter);
  return visc;
}


void setErrorField(apf::Field* estimate,Vec coef,apf::MeshEntity* ent,int nsd,int nshl)
//Function used to store the computed coefficients from the local error problem onto a field
{
    apf::Mesh* m = apf::getMesh(estimate);
    double coef_ez[nshl*nsd];
    PetscInt ez_idx[nshl*nsd];
    for(int ez=0;ez<nshl*nsd;ez++){ez_idx[ez]=ez;}
    VecGetValues(coef,nshl*nsd,ez_idx,coef_ez);

    //Copy coefficients onto field
    apf::Adjacent adjvert;
    m->getAdjacent(ent,0,adjvert);
    for(int idx=0;idx<adjvert.getSize();idx++){
      double coef_sub[3]={0,0,0};
      apf::setVector(estimate,adjvert[idx],0,&coef_sub[0]);
    }
    
    apf::Adjacent adjedg;
    m->getAdjacent(ent,1,adjedg);
    for(int idx=0;idx<nshl;idx++){
      //coef_ez holds nshl*nsd values. The third component only exists in 3D:
      //in 2D, coef_ez[nshl*2+idx] indexes past the end of the array and reads
      //whatever the stack frame below happened to leave there. That is a real
      //out-of-bounds read on every 2D element, and it is why
      //test2DgmshLoadAndAdapt returned 8.567e-18 on some runs and garbage
      //(1.2e+122, 2.4e+33, 2.1e+147) on others, host- and build-dependent,
      //while the 3D case was always stable. A 2D velocity has no third
      //component, so it contributes zero.
      double coef_sub[3] ={coef_ez[idx],
                           coef_ez[nshl+idx],
                           (nsd > 2) ? coef_ez[nshl*2+idx] : 0.0};
      apf::setVector(estimate,adjedg[idx],0,&coef_sub[0]);
    }
}

void MeshAdaptPUMIDrvr::removeBCData()
/** 
 * @brief Function used to remove the BC tags that were created during the computeDiffusiveFlux() function
 *
 * This is the simple way of avoiding errors from creating the same tags the next time the error estimator is called
 * The function just cycles through all of the faces and removes the apf tags.
 */
{
  if(comm_rank==0) std::cout<<"Start removing BC tags/data"<<std::endl;
  apf::MeshEntity* ent;   
  apf::MeshIterator* fIter = m->begin(nsd-1);
  while(ent=m->iterate(fIter))
  {
    if(has_gBC && m->hasTag(ent,BCtag)){
      m->removeTag(ent,BCtag);
      for(int i=0;i<4;i++)
      {
        if(i>0 && m->hasTag(ent,fluxtag[i]))
          m->removeTag(ent,fluxtag[i]);
      }
    }
    if(m->hasTag(ent,diffFlux))
      m->removeTag(ent,diffFlux);

  }
  m->end(fIter);
  if(has_gBC){
    m->destroyTag(BCtag);
    for(int i=0;i<4;i++)
    {
      if(i>0)
        m->destroyTag(fluxtag[i]);
    }
  }
  m->destroyTag(diffFlux);
  if(comm_rank==0) std::cerr<<"Destroyed BC and flux tags"<<std::endl;
}

void MeshAdaptPUMIDrvr::get_local_error(double &total_error) 
/** 
 * @brief This function aims to compute error at each element via an Element Residual Method.
 *
 * See Oden, J. Tinsley, Weihan Wu, and Mark Ainsworth. "An a posteriori error estimate for finite element approximations of the Navier-Stokes equations." Computer Methods in Applied Mechanics and Engineering 111.1 (1994): 185-202.
 * Effectively, the method involves projecting the weak residual onto a higher order space.
 * Boundary condition considerations are discussed in Ainsworth, Mark, and J. Tinsley Oden. A posteriori error estimation in finite element analysis. Vol. 37. John Wiley & Sons, 2011.
 * The function loops through each element and solves the local problem.
 */
{
  getProps(rho,nu);
  approx_order = approximation_order; 
  int_order = integration_order;
  nsd = m->getDimension();

  //***** Get Solution Fields First *****//
  apf::Field* voff = m->findField("vof");
  assert(voff);
  apf::Field* velf = m->findField("velocity");
  assert(velf);
  apf::Field* pref = m->findField("p");
  assert(pref);
  //*****               *****//
  
  //***** Compute the viscosity field *****//
  apf::Field* visc = getViscosityField(voff);

  //***** Compute diffusive flux *****//
  computeDiffusiveFlux(m,voff,visc,pref,velf);

  //Initialize the Error Fields
  freeField(err_reg);
  freeField(errRho_reg);
  freeField(errRel_reg);
  //err_reg = apf::createField(m,"ErrorRegion",apf::VECTOR,apf::getConstant(nsd));
  err_reg = apf::createField(m,"ErrorRegion",apf::SCALAR,apf::getVoronoiShape(nsd,1));
  errRho_reg = apf::createField(m,"ErrorDensity",apf::SCALAR,apf::getVoronoiShape(nsd,1));
  errRel_reg = apf::createField(m,"RelativeError",apf::SCALAR,apf::getVoronoiShape(nsd,1));

  //Start computing element quantities
  int numqpt; //number of quadrature points
  int nshl; //number of local shape functions
  int elem_type; //what type of topology
  double weight; //value container for the weight at each qpt
  double Jdet;
  apf::FieldShape* err_shape = apf::getHierarchic(approx_order);
  apf::Field * estimate = apf::createField(m, "err_est", apf::VECTOR, err_shape);
  apf::EntityShape* elem_shape;
  apf::Vector3 qpt; //container for quadrature points
  apf::MeshElement* element;
  apf::Element* visc_elem, *pres_elem,*velo_elem,*vof_elem;
  apf::Element* est_elem;
  apf::Matrix3x3 J; //actual Jacobian matrix
  apf::Matrix3x3 invJ; //inverse of Jacobian
  apf::NewArray <double> shpval; //array to store shape function values at quadrature points
  apf::NewArray <double> shpval_temp; //array to store shape function values at quadrature points temporarily
  apf::NewArray <apf::Vector3> shgval; //array to store shape function values at quadrature points

  apf::DynamicMatrix invJ_copy;
  apf::NewArray <apf::DynamicVector> shdrv;
  apf::NewArray <apf::DynamicVector> shgval_copy;
  
  apf::MeshIterator* iter = m->begin(nsd); //loop over elements
  apf::MeshEntity* ent;
  

  double err_est = 0;
  double err_est_total=0;
  long nLocalSolveFailures=0; //element-local solves that failed outright
  double u_norm_total=0;
  errRho_max = 0;
  while(ent = m->iterate(iter)){ //loop through all elements
    
    elem_type = m->getType(ent);
    if(!(elem_type != 4 || elem_type != 2)){ //2|TRI, 4|TET
      std::cout<<"Not a Tri or Tet present"<<std::endl;
      exit(0); 
    }
    element = apf::createMeshElement(m,ent);
    pres_elem = apf::createElement(pref,element);
    velo_elem = apf::createElement(velf,element);
    visc_elem = apf::createElement(visc,element); //at vof currently
    vof_elem = apf::createElement(voff,element);
  
    numqpt=apf::countIntPoints(element,int_order); //generally p*p maximum for shape functions
    nshl=apf::countElementNodes(err_shape,elem_type);
    shgval.allocate(nshl);
    shpval_temp.allocate(nshl);

    int hier_off;//there is an offset that needs to be made to isolate the hierarchic edge modes
    if(nsd ==2)
      hier_off=3;
    else if(nsd==3)
      hier_off= 4; 
    nshl = nshl - hier_off;

    shpval.allocate(nshl);   shgval_copy.allocate(nshl); shdrv.allocate(nshl);

    //LHS Matrix Initialization
    int ndofs = nshl*nsd;
    Mat K; //matrix size depends on nshl, which may vary from element to element
    MatCreate(PETSC_COMM_SELF,&K);
    MatSetSizes(K,ndofs,ndofs,ndofs,ndofs);
    MatSetFromOptions(K);
    MatSetUp(K); //is this inefficient? check later

    //RHS Vector Initialization
    Vec F;
    VecCreate(PETSC_COMM_SELF,&F);
    VecSetSizes(F,ndofs,ndofs);
    VecSetUp(F);

    //loop through all qpts
    for(int k=0;k<numqpt;k++){
      apf::getIntPoint(element,int_order,k,qpt); //get a quadrature point and store in qpt
      apf::getJacobian(element,qpt,J); //evaluate the Jacobian at the quadrature point
      J = apf::transpose(J); //Is PUMI still defined in this way?
      if(nsd==2)
        J[2][2] = 1.0; //this is necessary to avoid singular matrix
      invJ = invert(J);
      Jdet=fabs(apf::getJacobianDeterminant(J,nsd)); 
      weight = apf::getIntWeight(element,int_order,k);
      invJ_copy = apf::fromMatrix(invJ);

      //first get the shape function values for error shape functions
      elem_shape = err_shape->getEntityShape(elem_type);
      elem_shape->getValues(NULL,NULL,qpt,shpval_temp);
      elem_shape->getLocalGradients(NULL,NULL,qpt,shgval); 

      for(int i =0;i<nshl;i++){ //get the true derivative and copy only the edge modes for use
        shgval_copy[i] = apf::fromVector(shgval[i+hier_off]);
        shpval[i] = shpval_temp[i+hier_off];
        apf::multiply(shgval_copy[i],invJ_copy,shdrv[i]); 
      }

      //obtain needed values

      apf::Vector3 vel_vect;
      apf::Matrix3x3 grad_vel;
      apf::getVector(velo_elem,qpt,vel_vect);
      apf::getVectorGrad(velo_elem,qpt,grad_vel);
      grad_vel = apf::transpose(grad_vel);
      apf::Vector3 grad_vof;
      apf::getGrad(vof_elem,qpt,grad_vof);
    
      double density = getMPvalue(apf::getScalar(vof_elem,qpt),rho_0,rho_1);
      double pressure = apf::getScalar(pres_elem,qpt);
      double visc_val = apf::getScalar(visc_elem,qpt);
      apf::Vector3 grad_rho = grad_vof*(rho_1-rho_0);

      //Left-Hand Side
      getLHS(K,shdrv,nsd,weight,visc_val,nshl);

      //Get RHS
      getRHS(F,shpval,shdrv,vel_vect,grad_vel,nsd,weight,nshl,visc_val,density,grad_rho,pressure,g);

    } // end quadrature loop

    //to complete integration, scale by the determinant of the Jacobian

    MatAssemblyBegin(K,MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(K,MAT_FINAL_ASSEMBLY);
    MatScale(K,Jdet); //must be done after assembly
    VecAssemblyBegin(F);
    VecAssemblyEnd(F); 
    VecScale(F,Jdet); //must be done after assembly
    double* bflux;
    PetscInt F_idx[ndofs];
    bflux = (double*) calloc(ndofs,sizeof(double));

    getBoundaryFlux(m, ent,bflux);
    for(int s=0;s<ndofs;s++){
      F_idx[s]=s;
    }
    VecSetValues(F,ndofs,F_idx,bflux,ADD_VALUES);
    VecAssemblyBegin(F); VecAssemblyEnd(F);
    free(bflux);
    Vec coef;
    VecCreate(PETSC_COMM_SELF,&coef);
    VecSetSizes(coef,ndofs,ndofs);
    VecSetUp(coef);
    // VecSetUp does not define the vector's values. If the local solve below
    // fails, KSPSolve leaves coef untouched and setErrorField() then reads
    // uninitialized memory straight into the error field -- observed as
    // get_local_error() returning 2.98e+21 on one run and 2.24e+98 on the next
    // (assert errorTotal<1e-14 in test2DgmshLoadAndAdapt). Zero it so a failed
    // solve contributes nothing rather than noise.
    VecZeroEntries(coef);

    KSP ksp; //initialize solver context
    KSPCreate(PETSC_COMM_SELF,&ksp);
    KSPSetOperators(ksp,K,K);
    KSPSetType(ksp,KSPPREONLY);
    PC pc;
    KSPGetPC(ksp,&pc);
    PCSetType(pc,PCLU);
    KSPSetFromOptions(ksp);

    KSPSolve(ksp,F,coef);

    // The local problem is solved with PREONLY+LU, so a singular element matrix
    // fails outright rather than converging poorly. That was silent: the return
    // code was discarded and no reason was queried, so a failed factorization
    // was indistinguishable from a good solve and propagated into the error
    // estimate. Count the failures and report once after the loop -- an error
    // estimate computed from a mesh with singular local problems is not
    // trustworthy, and the user needs to know rather than receive a number.
    KSPConvergedReason localReason;
    KSPGetConvergedReason(ksp,&localReason);
    if(localReason < 0)
      nLocalSolveFailures++;

    KSPDestroy(&ksp); //destroy ksp

    setErrorField(estimate,coef,ent,nsd,nshl);

    //compute the local error  
    double Acomp=0;
    double Bcomp=0;
    double visc_avg=0;
    double u_norm = 0;
    apf::Matrix3x3 phi_ij;
    apf::Matrix3x3 vel_ij;
    apf::Vector3 vel_vect;
    apf::Vector3 grad_vof;

    est_elem= apf::createElement(estimate,element);   
    for(int k=0; k<numqpt;k++){ 
      apf::getIntPoint(element,int_order,k,qpt); //get a quadrature point and store in qpt
      apf::getJacobian(element,qpt,J); //evaluate the Jacobian at the quadrature point

      invJ = invert(J);
      invJ = apf::transpose(invJ);
      Jdet=fabs(apf::getJacobianDeterminant(J,nsd)); 
      weight = apf::getIntWeight(element,int_order,k);
      invJ_copy = apf::fromMatrix(invJ);

      //first get the shape function values for error shape functions
      elem_shape = err_shape->getEntityShape(elem_type);
      elem_shape->getValues(NULL,NULL,qpt,shpval_temp);
      elem_shape->getLocalGradients(NULL,NULL,qpt,shgval); 

      for(int i =0;i<nshl;i++){ //get the true derivative and copy only the edge modes for use
        shgval_copy[i] = apf::fromVector(shgval[i+hier_off]);
        shpval[i] = shpval_temp[i+hier_off];
        apf::multiply(shgval_copy[i],invJ_copy,shdrv[i]); 
      }
      double visc_val = apf::getScalar(visc_elem,qpt);
      double pres_val = apf::getScalar(pres_elem,qpt);
      double density = getMPvalue(apf::getScalar(vof_elem,qpt),rho_0,rho_1);
      apf::getVectorGrad(est_elem,qpt,phi_ij);
      apf::getGrad(vof_elem,qpt,grad_vof);
      apf::getVector(velo_elem,qpt,vel_vect);
      apf::getVectorGrad(velo_elem,qpt,vel_ij);
      vel_ij = apf::transpose(vel_ij);
      phi_ij = apf::transpose(phi_ij);

      Acomp = Acomp + visc_val*getDotProduct(phi_ij,phi_ij+apf::transpose(phi_ij))*weight;
      Bcomp = Bcomp + apf::getDiv(velo_elem,qpt)*apf::getDiv(velo_elem,qpt)*weight;
      visc_avg = visc_avg + visc_val*weight;
      u_norm = u_norm + visc_val*getDotProduct(vel_ij,vel_ij+apf::transpose(vel_ij))*weight;

    } //end compute local error
    visc_avg = visc_avg*Jdet/apf::measure(element);
    Acomp = Acomp*Jdet/visc_avg; //nondimensionalize with average viscosity, Jacobians can cancel out, but this is done for clarity
    Bcomp = Bcomp*Jdet;
    u_norm = u_norm/visc_avg*Jdet;
    err_est = sqrt(Acomp); 

    apf::Vector3 err_in(err_est,Acomp,Bcomp);
    //apf::setVector(err_reg,ent,0,err_in);
    apf::setScalar(err_reg,ent,0,err_est);
    double errRho = err_est/sqrt(apf::measure(element));
    apf::setScalar(errRho_reg,ent,0,errRho);
    if(errRho>errRho_max)
      errRho_max = errRho;

    double err_rel = err_est/sqrt(u_norm);
    apf::setScalar(errRel_reg,ent,0,err_rel);

    err_est_total = err_est_total+(Acomp); //for tracking the upper bound
    u_norm_total = u_norm_total + u_norm;
   
    MatDestroy(&K); //destroy the matrix
    VecDestroy(&F); //destroy vector
    VecDestroy(&coef); //destroy vector

    apf::destroyElement(visc_elem);apf::destroyElement(pres_elem);apf::destroyElement(velo_elem);apf::destroyElement(est_elem);apf::destroyElement(vof_elem);
  } //end element loop

  PCU_Add_Doubles(PCUObj, &err_est_total,1);
  PCU_Add_Doubles(PCUObj, &u_norm_total,1);

  if(nLocalSolveFailures > 0 && comm_rank==0)
    std::cerr<<"WARNING: "<<nLocalSolveFailures<<" element-local error problems "
             <<"failed to solve (singular matrix under PREONLY+LU); their "
             <<"contribution to the error estimate is zero and the estimate "
             <<"below should not be trusted."<<std::endl;

  total_error = sqrt(err_est_total);
  u_norm_total = sqrt(u_norm_total);
  rel_err_total = total_error/u_norm_total;

  if(comm_rank==0){
    std::cerr<<std::setprecision(10)<<std::endl;
    std::cerr<<"Error estimate "<<total_error<<std::endl; 
    std::cerr<<"Error density maximum "<<errRho_max<<std::endl;
    std::cerr<<"U_norm_total "<<u_norm_total<<std::endl;
  }

  if(logging_config=="errorOnly"){ //feature to just look at the error fields without adapting the mesh
    if(comm_rank==0)
      std::cout<<"outputting error field\n";
    char namebuffer[20];
    sprintf(namebuffer,"err_reg_%i",nEstimate);
    apf::writeVtkFiles(namebuffer, m);
    //target_error = total_error*2; //this is a hack to prevent adapting
    THRESHOLD = total_error*2;
    nEstimate++;
    removeBCData();
  }

  m->end(iter);
  apf::destroyField(visc);
  apf::destroyField(estimate);

  if(comm_rank==0)
    std::cerr<<"It cleared the ERM function.\n";
}

/** @} */
