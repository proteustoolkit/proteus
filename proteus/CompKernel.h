#ifndef COMPKERNEL_H
#define COMPKERNEL_H
#include <cmath>
#include "index_type.h"
/**
 *   A class to provide indexing into Euclidean vectors and tensors.
 */
template<int NSPACE>
class EIndex
{
};

template<>
class EIndex<3>
{
public:
  const int X,Y,Z,
    XX,XY,XZ,
    YX,YY,YZ,
    ZX,ZY,ZZ,
    sXX,sXY,sXZ,
    sYX,sYY,sYZ,
    sZX,sZY,sZZ,
    nSymTen,
    XHX,XHY,
    YHX,YHY,
    ZHX,ZHY,
    HXHX,HXHY,
    HYHX,HYHY;
  EIndex():
    X(0),
    Y(1),
    Z(2),
    XX(0),XY(1),XZ(2),
    YX(3),YY(4),YZ(5),
    ZX(6),ZY(7),ZZ(8),
    sXX(0),sXY(5),sXZ(4),
    sYX(5),sYY(1),sYZ(3),
    sZX(4),sZY(3),sZZ(2),
    nSymTen(6),
    XHX(0),XHY(1),
    YHX(2),YHY(3),
    ZHX(4),ZHY(5),
    HXHX(0),HXHY(1),
    HYHX(2),HYHY(3)
  {}
};

template<>
class EIndex<2>
{
public:
  const int X,Y,
    XX,XY,
    YX,YY,
    sXX,sXY,
    sYX,sYY,
    nSymTen,
    XHX,XHY,
    YHX,YHY,
    HXHX;
  EIndex():
    X(0),
    Y(1),
    XX(0),XY(1),
    YX(2),YY(3),
    sXX(0),sXY(2),
    sYX(2),sYY(1),
    nSymTen(3),
    XHX(0),XHY(1),
    YHX(2),YHY(3),
    HXHX(0)
  {}
};

template<>
class EIndex<1>
{
public:
  const int X,
    XX,
    sXX,
    nSymTen,
    XHX,
    HXHX;
  EIndex():
    X(0),
    XX(0),
    sXX(0),
    nSymTen(1),
    XHX(0),
    HXHX(0)
  {}
};

//I separated the space mapping part of the kernel so I could partially specialize the template on NSPACE
template<int NSPACE, int NDOF_MESH_TRIAL_ELEMENT>
class CompKernelSpaceMapping
{
public:
  inline void calculateMapping_element(const proteus_index_t eN,
				       const int k,
				       double* mesh_dof,
				       proteus_index_t* mesh_l2g,
				       //proteus::pyarray<double>& mesh_trial_ref,
				       double* mesh_trial_ref,
				       double* mesh_grad_trial_ref,
				       double* jac,
				       double& jacDet,
				       double* jacInv,
				       double& x,
				       double& y,
				       double& z);
  inline void calculateH_element(const proteus_index_t eN,
				 const int k,
				 double* h_dof,
				 proteus_index_t* mesh_l2g,
				 //proteus::pyarray<double>& mesh_trial_ref,
				 double* mesh_trial_ref,
				 double& h);
  inline void calculateMappingVelocity_element(const proteus_index_t eN,
					       const int k,
					       double* mesh_velocity_dof,
					       proteus_index_t* mesh_l2g,
					       //proteus::pyarray<double>& mesh_trial_ref,
					       double* mesh_trial_ref,
					       double& xt,
					       double& yt,
					       double& zt);
  inline void calculateMapping_elementBoundary(const proteus_index_t eN,
					       const int ebN_local,
					       const int kb,
					       const int ebN_local_kb,
					       double* mesh_dof,
					       proteus_index_t* mesh_l2g,
					       double* mesh_trial_trace_ref,
					       double* mesh_grad_trial_trace_ref,
					       double* boundaryJac_ref,
					       double* jac,
					       double& jacDet,
					       double* jacInv,
					       double* boundaryJac,
					       double* metricTensor,
					       double& metricTensorDetSqrt,
					       double* normal_ref,
					       double* normal,
					       double& x,
					       double& y,
					       double& z);
  inline void calculateMappingVelocity_elementBoundary(const proteus_index_t eN,
						       const int ebN_local,
						       const int kb,
						       const int ebN_local_kb,
						       double* mesh_velocity_dof,
						       proteus_index_t* mesh_l2g,
						       double* mesh_trial_trace_ref,
						       double& xt,
						       double& yt,
						       double& zt,
						       double* normal,
						       double* boundaryJac,						       
						       double* metricTensor,
						       double& metricTensorDetSqrt);
  inline void valFromDOF(const double* dof,const proteus_index_t* l2g_element,const double* trial_ref,double& val)
  {
    val=0.0;
    for (int j=0;j<NDOF_MESH_TRIAL_ELEMENT;j++)
      val+=dof[l2g_element[j]]*trial_ref[j];
  }
  
  inline void gradFromDOF(const double* dof,const proteus_index_t* l2g_element,const double* grad_trial,double* grad)
  {
    for(int I=0;I<NSPACE;I++)
      grad[I] = 0.0;
    for (int j=0;j<NDOF_MESH_TRIAL_ELEMENT;j++)
      for(int I=0;I<NSPACE;I++)
	grad[I] += dof[l2g_element[j]]*grad_trial[j*NSPACE+I];
  }
  
  inline void hessFromDOF(const double* dof,const proteus_index_t* l2g_element,const double* hess_trial,double* hess)
  {
    const int NSPACE2=NSPACE*NSPACE;
    for(int I=0;I<NSPACE;I++)
      for(int J=0;J<NSPACE;J++)
	hess[I*NSPACE+J] = 0.0;
    for (int j=0;j<NDOF_MESH_TRIAL_ELEMENT;j++)
      for(int I=0;I<NSPACE;I++)
	for(int J=0;J<NSPACE;J++)
	  hess[I*NSPACE+J] += dof[l2g_element[j]]*hess_trial[j*NSPACE2+I*NSPACE+J];
  }
};

//specialization for 3D
template<int NDOF_MESH_TRIAL_ELEMENT>
class CompKernelSpaceMapping<3,NDOF_MESH_TRIAL_ELEMENT>
{
public:
  const int X,Y,Z,
    XX,XY,XZ,
    YX,YY,YZ,
    ZX,ZY,ZZ,
    sXX,sXY,sXZ,
    sYX,sYY,sYZ,
    sZX,sZY,sZZ,
    nSymTen,
    XHX,XHY,
    YHX,YHY,
    ZHX,ZHY,
    HXHX,HXHY,
    HYHX,HYHY;
  ;
  CompKernelSpaceMapping():
    X(0),
    Y(1),
    Z(2),
    XX(0),XY(1),XZ(2),
    YX(3),YY(4),YZ(5),
    ZX(6),ZY(7),ZZ(8),
    sXX(0),sXY(5),sXZ(4),
    sYX(5),sYY(1),sYZ(3),
    sZX(4),sZY(3),sZZ(2),
    nSymTen(6),
    XHX(0),XHY(1),
    YHX(2),YHY(3),
    ZHX(4),ZHY(5),
    HXHX(0),HXHY(1),
    HYHX(2),HYHY(3)
  {}
  inline void calculateMapping_element(const proteus_index_t eN,
				       const int k,
				       double* mesh_dof,
				       proteus_index_t* mesh_l2g,
				       //proteus::pyarray<double>& mesh_trial_ref,
				       double* mesh_trial_ref,
				       double* mesh_grad_trial_ref,
				       double* jac,
				       double& jacDet,
				       double* jacInv,
				       double& x,
				       double& y,
				       double& z)
  {
    double Grad_x[3],Grad_y[3],Grad_z[3],oneOverJacDet;
    
    //
    //mapping of reference element to physical element
    //
    x=0.0;y=0.0;z=0.0;
    for (int I=0;I<3;I++)
      {
	Grad_x[I]=0.0;Grad_y[I]=0.0;Grad_z[I]=0.0;
      }
    for (int j=0;j<NDOF_MESH_TRIAL_ELEMENT;j++)
      {
	proteus_index_t eN_j=eN*NDOF_MESH_TRIAL_ELEMENT+j;
	//x += mesh_dof[mesh_l2g[eN_j]*3+0]*mesh_trial_ref(k, j);
	//y += mesh_dof[mesh_l2g[eN_j]*3+1]*mesh_trial_ref(k, j);
	//z += mesh_dof[mesh_l2g[eN_j]*3+2]*mesh_trial_ref(k, j);	      
        x += mesh_dof[mesh_l2g[eN_j]*3+0]*mesh_trial_ref[k*NDOF_MESH_TRIAL_ELEMENT+j];
	y += mesh_dof[mesh_l2g[eN_j]*3+1]*mesh_trial_ref[k*NDOF_MESH_TRIAL_ELEMENT+j];
	z += mesh_dof[mesh_l2g[eN_j]*3+2]*mesh_trial_ref[k*NDOF_MESH_TRIAL_ELEMENT+j];
	for (int I=0;I<3;I++)
	  {
	    Grad_x[I] += mesh_dof[mesh_l2g[eN_j]*3+0]*mesh_grad_trial_ref[k*NDOF_MESH_TRIAL_ELEMENT*3+j*3+I];
	    Grad_y[I] += mesh_dof[mesh_l2g[eN_j]*3+1]*mesh_grad_trial_ref[k*NDOF_MESH_TRIAL_ELEMENT*3+j*3+I];
	    Grad_z[I] += mesh_dof[mesh_l2g[eN_j]*3+2]*mesh_grad_trial_ref[k*NDOF_MESH_TRIAL_ELEMENT*3+j*3+I];
	  }
      }
    jac[XX] = Grad_x[X];
    jac[XY] = Grad_x[Y];
    jac[XZ] = Grad_x[Z];
    jac[YX] = Grad_y[X];
    jac[YY] = Grad_y[Y];
    jac[YZ] = Grad_y[Z];
    jac[ZX] = Grad_z[X];
    jac[ZY] = Grad_z[Y];
    jac[ZZ] = Grad_z[Z];
    jacDet = 
      jac[XX]*(jac[YY]*jac[ZZ] - jac[YZ]*jac[ZY]) -
      jac[XY]*(jac[YX]*jac[ZZ] - jac[YZ]*jac[ZX]) +
      jac[XZ]*(jac[YX]*jac[ZY] - jac[YY]*jac[ZX]);
    oneOverJacDet = 1.0/jacDet;
    jacInv[XX] = oneOverJacDet*(jac[YY]*jac[ZZ] - jac[YZ]*jac[ZY]);
    jacInv[YX] = oneOverJacDet*(jac[YZ]*jac[ZX] - jac[YX]*jac[ZZ]);
    jacInv[ZX] = oneOverJacDet*(jac[YX]*jac[ZY] - jac[YY]*jac[ZX]);
    jacInv[XY] = oneOverJacDet*(jac[ZY]*jac[XZ] - jac[ZZ]*jac[XY]);
    jacInv[YY] = oneOverJacDet*(jac[ZZ]*jac[XX] - jac[ZX]*jac[XZ]);
    jacInv[ZY] = oneOverJacDet*(jac[ZX]*jac[XY] - jac[ZY]*jac[XX]);
    jacInv[XZ] = oneOverJacDet*(jac[XY]*jac[YZ] - jac[XZ]*jac[YY]);
    jacInv[YZ] = oneOverJacDet*(jac[XZ]*jac[YX] - jac[XX]*jac[YZ]);
    jacInv[ZZ] = oneOverJacDet*(jac[XX]*jac[YY] - jac[XY]*jac[YX]);
  }
  
  inline void calculateH_element(const proteus_index_t eN,
				 const int k,
				 double* h_dof,
				 proteus_index_t* mesh_l2g,
				 //proteus::pyarray<double>& mesh_trial_ref,
				 double* mesh_trial_ref,
				 double& h)
  {
    h=0.0;
    for (int j=0;j<NDOF_MESH_TRIAL_ELEMENT;j++)
      {
	proteus_index_t eN_j=eN*NDOF_MESH_TRIAL_ELEMENT+j;
	//h += h_dof[mesh_l2g[eN_j]]*mesh_trial_ref(k, j);
        h += h_dof[mesh_l2g[eN_j]]*mesh_trial_ref[k*NDOF_MESH_TRIAL_ELEMENT+j];
      }
  }

  inline void calculateMappingVelocity_element(const proteus_index_t eN,
					       const int k,
					       double* mesh_velocity_dof,
					       proteus_index_t* mesh_l2g,
					       //proteus::pyarray<double>& mesh_trial_ref,
					       double* mesh_trial_ref,
					       double& xt,
					       double& yt,
					       double& zt)
  {
    //
    //time derivative of mapping of reference element to physical element
    //
    xt=0.0;yt=0.0;zt=0.0;
    for (int j=0;j<NDOF_MESH_TRIAL_ELEMENT;j++)
      {
	proteus_index_t eN_j=eN*NDOF_MESH_TRIAL_ELEMENT+j;
	//xt += mesh_velocity_dof[mesh_l2g[eN_j]*3+0]*mesh_trial_ref(k, j);
	//yt += mesh_velocity_dof[mesh_l2g[eN_j]*3+1]*mesh_trial_ref(k, j);
	//zt += mesh_velocity_dof[mesh_l2g[eN_j]*3+2]*mesh_trial_ref(k, j);	      
        xt += mesh_velocity_dof[mesh_l2g[eN_j]*3+0]*mesh_trial_ref[k*NDOF_MESH_TRIAL_ELEMENT+j];
	yt += mesh_velocity_dof[mesh_l2g[eN_j]*3+1]*mesh_trial_ref[k*NDOF_MESH_TRIAL_ELEMENT+j];
	zt += mesh_velocity_dof[mesh_l2g[eN_j]*3+2]*mesh_trial_ref[k*NDOF_MESH_TRIAL_ELEMENT+j];
      }
  }

  inline void calculateMapping_elementBoundary(const proteus_index_t eN,
					       const int ebN_local,
					       const int kb,
					       const int ebN_local_kb,
					       double* mesh_dof,
					       proteus_index_t* mesh_l2g,
					       double* mesh_trial_trace_ref,
					       double* mesh_grad_trial_trace_ref,
					       double* boundaryJac_ref,
					       double* jac,
					       double& jacDet,
					       double* jacInv,
					       double* boundaryJac,
					       double* metricTensor,
					       double& metricTensorDetSqrt,
					       double* normal_ref,
					       double* normal,
					       double& x,
					       double& y,
					       double& z)
  {
    const int ebN_local_kb_nSpace = ebN_local_kb*3,
      ebN_local_kb_nSpace_nSpacem1 = ebN_local_kb*3*2;
  
    double Grad_x_ext[3],Grad_y_ext[3],Grad_z_ext[3],oneOverJacDet,norm_normal=0.0;
    // 
    //calculate mapping from the reference element to the physical element
    // 
    x=0.0;y=0.0;z=0.0;
    for (int I=0;I<3;I++)
      {
	Grad_x_ext[I] = 0.0;
	Grad_y_ext[I] = 0.0;
	Grad_z_ext[I] = 0.0;
      }
    for (int j=0;j<NDOF_MESH_TRIAL_ELEMENT;j++) 
      { 
	proteus_index_t eN_j = eN*NDOF_MESH_TRIAL_ELEMENT+j;
	int ebN_local_kb_j = ebN_local_kb*NDOF_MESH_TRIAL_ELEMENT+j;
	int ebN_local_kb_j_nSpace = ebN_local_kb_j*3;
	x += mesh_dof[mesh_l2g[eN_j]*3+0]*mesh_trial_trace_ref[ebN_local_kb_j]; 
	y += mesh_dof[mesh_l2g[eN_j]*3+1]*mesh_trial_trace_ref[ebN_local_kb_j]; 
	z += mesh_dof[mesh_l2g[eN_j]*3+2]*mesh_trial_trace_ref[ebN_local_kb_j]; 
	for (int I=0;I<3;I++)
	  {
	    Grad_x_ext[I] += mesh_dof[mesh_l2g[eN_j]*3+0]*mesh_grad_trial_trace_ref[ebN_local_kb_j_nSpace+I];
	    Grad_y_ext[I] += mesh_dof[mesh_l2g[eN_j]*3+1]*mesh_grad_trial_trace_ref[ebN_local_kb_j_nSpace+I]; 
	    Grad_z_ext[I] += mesh_dof[mesh_l2g[eN_j]*3+2]*mesh_grad_trial_trace_ref[ebN_local_kb_j_nSpace+I];
	  } 
      }
    //Space Mapping Jacobian
    jac[XX] = Grad_x_ext[X];
    jac[XY] = Grad_x_ext[Y];
    jac[XZ] = Grad_x_ext[Z];
    jac[YX] = Grad_y_ext[X];
    jac[YY] = Grad_y_ext[Y];
    jac[YZ] = Grad_y_ext[Z];
    jac[ZX] = Grad_z_ext[X];
    jac[ZY] = Grad_z_ext[Y];
    jac[ZZ] = Grad_z_ext[Z];
    jacDet = 
      jac[XX]*(jac[YY]*jac[ZZ] - jac[YZ]*jac[ZY]) -
      jac[XY]*(jac[YX]*jac[ZZ] - jac[YZ]*jac[ZX]) +
      jac[XZ]*(jac[YX]*jac[ZY] - jac[YY]*jac[ZX]);
    oneOverJacDet = 1.0/jacDet;
    jacInv[XX] = oneOverJacDet*(jac[YY]*jac[ZZ] - jac[YZ]*jac[ZY]);
    jacInv[YX] = oneOverJacDet*(jac[YZ]*jac[ZX] - jac[YX]*jac[ZZ]);
    jacInv[ZX] = oneOverJacDet*(jac[YX]*jac[ZY] - jac[YY]*jac[ZX]);
    jacInv[XY] = oneOverJacDet*(jac[ZY]*jac[XZ] - jac[ZZ]*jac[XY]);
    jacInv[YY] = oneOverJacDet*(jac[ZZ]*jac[XX] - jac[ZX]*jac[XZ]);
    jacInv[ZY] = oneOverJacDet*(jac[ZX]*jac[XY] - jac[ZY]*jac[XX]);
    jacInv[XZ] = oneOverJacDet*(jac[XY]*jac[YZ] - jac[XZ]*jac[YY]);
    jacInv[YZ] = oneOverJacDet*(jac[XZ]*jac[YX] - jac[XX]*jac[YZ]);
    jacInv[ZZ] = oneOverJacDet*(jac[XX]*jac[YY] - jac[XY]*jac[YX]);
    //normal
    norm_normal=0.0;
    for (int I=0;I<3;I++)
      normal[I] = 0.0;
    for (int I=0;I<3;I++)
      {
	for (int J=0;J<3;J++)
	  {
	    normal[I] += jacInv[J*3+I]*normal_ref[ebN_local_kb_nSpace+J];
	  }
	norm_normal+=normal[I]*normal[I];
      }
    norm_normal = sqrt(norm_normal);
    for (int I=0;I<3;I++)
      {
	normal[I] /= norm_normal;
      }
    //metric tensor and determinant
    boundaryJac[XHX] = jac[XX]*boundaryJac_ref[ebN_local_kb_nSpace_nSpacem1+XHX]+jac[XY]*boundaryJac_ref[ebN_local_kb_nSpace_nSpacem1+YHX]+jac[XZ]*boundaryJac_ref[ebN_local_kb_nSpace_nSpacem1+ZHX];
    boundaryJac[XHY] = jac[XX]*boundaryJac_ref[ebN_local_kb_nSpace_nSpacem1+XHY]+jac[XY]*boundaryJac_ref[ebN_local_kb_nSpace_nSpacem1+YHY]+jac[XZ]*boundaryJac_ref[ebN_local_kb_nSpace_nSpacem1+ZHY];
    boundaryJac[YHX] = jac[YX]*boundaryJac_ref[ebN_local_kb_nSpace_nSpacem1+XHX]+jac[YY]*boundaryJac_ref[ebN_local_kb_nSpace_nSpacem1+YHX]+jac[YZ]*boundaryJac_ref[ebN_local_kb_nSpace_nSpacem1+ZHX];
    boundaryJac[YHY] = jac[YX]*boundaryJac_ref[ebN_local_kb_nSpace_nSpacem1+XHY]+jac[YY]*boundaryJac_ref[ebN_local_kb_nSpace_nSpacem1+YHY]+jac[YZ]*boundaryJac_ref[ebN_local_kb_nSpace_nSpacem1+ZHY];
    boundaryJac[ZHX] = jac[ZX]*boundaryJac_ref[ebN_local_kb_nSpace_nSpacem1+XHX]+jac[ZY]*boundaryJac_ref[ebN_local_kb_nSpace_nSpacem1+YHX]+jac[ZZ]*boundaryJac_ref[ebN_local_kb_nSpace_nSpacem1+ZHX];
    boundaryJac[ZHY] = jac[ZX]*boundaryJac_ref[ebN_local_kb_nSpace_nSpacem1+XHY]+jac[ZY]*boundaryJac_ref[ebN_local_kb_nSpace_nSpacem1+YHY]+jac[ZZ]*boundaryJac_ref[ebN_local_kb_nSpace_nSpacem1+ZHY];
  
    metricTensor[HXHX] = boundaryJac[XHX]*boundaryJac[XHX]+boundaryJac[YHX]*boundaryJac[YHX]+boundaryJac[ZHX]*boundaryJac[ZHX];
    metricTensor[HXHY] = boundaryJac[XHX]*boundaryJac[XHY]+boundaryJac[YHX]*boundaryJac[YHY]+boundaryJac[ZHX]*boundaryJac[ZHY];
    metricTensor[HYHX] = boundaryJac[XHY]*boundaryJac[XHX]+boundaryJac[YHY]*boundaryJac[YHX]+boundaryJac[ZHY]*boundaryJac[ZHX];
    metricTensor[HYHY] = boundaryJac[XHY]*boundaryJac[XHY]+boundaryJac[YHY]*boundaryJac[YHY]+boundaryJac[ZHY]*boundaryJac[ZHY];
  
    metricTensorDetSqrt=sqrt(metricTensor[HXHX]*metricTensor[HYHY]- metricTensor[HXHY]*metricTensor[HYHX]);
  }

  inline void calculateMappingVelocity_elementBoundary(const proteus_index_t eN,
						       const int ebN_local,
						       const int kb,
						       const int ebN_local_kb,
						       double* mesh_velocity_dof,
						       proteus_index_t* mesh_l2g,
						       double* mesh_trial_trace_ref,
						       double& xt,
						       double& yt,
						       double& zt,
						       double* normal,
						       double* boundaryJac,						       
						       double* metricTensor,
						       double& metricTensorDetSqrt)
  {
    //const int ebN_local_kb_nSpace = ebN_local_kb*3,
    //  ebN_local_kb_nSpace_nSpacem1 = ebN_local_kb*3*2;
    // 
    //calculate velocity of mapping from the reference element to the physical element
    // 
    xt=0.0;yt=0.0;zt=0.0;
    for (int j=0;j<NDOF_MESH_TRIAL_ELEMENT;j++) 
      { 
	proteus_index_t eN_j = eN*NDOF_MESH_TRIAL_ELEMENT+j;
	int ebN_local_kb_j = ebN_local_kb*NDOF_MESH_TRIAL_ELEMENT+j;
	xt += mesh_velocity_dof[mesh_l2g[eN_j]*3+0]*mesh_trial_trace_ref[ebN_local_kb_j]; 
	yt += mesh_velocity_dof[mesh_l2g[eN_j]*3+1]*mesh_trial_trace_ref[ebN_local_kb_j]; 
	zt += mesh_velocity_dof[mesh_l2g[eN_j]*3+2]*mesh_trial_trace_ref[ebN_local_kb_j]; 
      }
    //modify the metricTensorDetSqrt to include the effect of the moving domain
    //it's not exactly the Sqrt(Det(G_tr_G)) now, see notes
    //just do it brute force
    double 
      Gy_tr_Gy_00 = 1.0 + xt*xt + yt*yt + zt*zt,
      Gy_tr_Gy_01 = boundaryJac[XHX]*xt+boundaryJac[YHX]*yt+boundaryJac[ZHX]*zt,
      Gy_tr_Gy_02 = boundaryJac[XHY]*xt+boundaryJac[YHY]*yt+boundaryJac[ZHY]*zt,
      Gy_tr_Gy_10 = Gy_tr_Gy_01,
      Gy_tr_Gy_20 = Gy_tr_Gy_02,
      Gy_tr_Gy_11 = metricTensor[HXHX],
      Gy_tr_Gy_12 = metricTensor[HXHY],
      Gy_tr_Gy_21 = metricTensor[HYHX],
      Gy_tr_Gy_22 = metricTensor[HYHY],
      xt_dot_n = xt*normal[X]+yt*normal[Y]+zt*normal[Z];
    metricTensorDetSqrt=sqrt((Gy_tr_Gy_00*Gy_tr_Gy_11*Gy_tr_Gy_22 +
			      Gy_tr_Gy_01*Gy_tr_Gy_12*Gy_tr_Gy_20 +
			      Gy_tr_Gy_02*Gy_tr_Gy_10*Gy_tr_Gy_21 -
			      Gy_tr_Gy_20*Gy_tr_Gy_11*Gy_tr_Gy_02 -
			      Gy_tr_Gy_21*Gy_tr_Gy_12*Gy_tr_Gy_00 -
			      Gy_tr_Gy_22*Gy_tr_Gy_10*Gy_tr_Gy_01) / (1.0+xt_dot_n*xt_dot_n));
  }
  inline void valFromDOF(const double* dof,const proteus_index_t* l2g_element,const double* trial_ref,double& val)
  {
    val=0.0;
    for (int j=0;j<NDOF_MESH_TRIAL_ELEMENT;j++)
      val+=dof[l2g_element[j]]*trial_ref[j];
  }
  
  inline void gradFromDOF(const double* dof,const proteus_index_t* l2g_element,const double* grad_trial,double* grad)
  {
    for(int I=0;I<3;I++)
      grad[I] = 0.0;
    for (int j=0;j<NDOF_MESH_TRIAL_ELEMENT;j++)
      for(int I=0;I<3;I++)
	grad[I] += dof[l2g_element[j]]*grad_trial[j*3+I];
  }

  inline void hessFromDOF(const double* dof,const proteus_index_t* l2g_element,const double* hess_trial,double* hess)
  {
    for(int I=0;I<3;I++)
      for(int J=0;J<3;J++)
	hess[I*3+J] = 0.0;
    for (int j=0;j<NDOF_MESH_TRIAL_ELEMENT;j++)
      for(int I=0;I<3;I++)
	for(int J=0;J<3;J++)
	  hess[I*3+J] += dof[l2g_element[j]]*hess_trial[j*9+I*3+J];
  }
  
  inline void calculateMapping_element(const proteus_index_t eN,
				       const int k,
				       double* mesh_dof,
				       proteus_index_t* mesh_l2g,
				       //proteus::pyarray<double>& mesh_trial_ref,
				       double* mesh_trial_ref,
				       double* mesh_grad_trial_ref,
				       double* jac,
				       double& jacDet,
				       double* jacInv,
				       double& x,
				       double& y)
  {
    calculateMapping_element(eN,
			     k,
			     mesh_dof,
			     mesh_l2g,
			     mesh_trial_ref,
			     mesh_grad_trial_ref,
			     jac,
			     jacDet,
			     jacInv,
			     x,
			     y,
			     0.0);
  }

  inline void calculateMappingVelocity_element(const proteus_index_t eN,
					       const int k,
					       double* mesh_velocity_dof,
					       proteus_index_t* mesh_l2g,
					       //proteus::pyarray<double>& mesh_trial_ref,
					       double* mesh_trial_ref,
					       double& xt,
					       double& yt)
  {
    calculateMappingVelocity_element(eN,
				     k,
				     mesh_velocity_dof,
				     mesh_l2g,
				     mesh_trial_ref,
				     xt,
				     yt,
				     0.0);
  }
  inline void calculateMapping_elementBoundary(const proteus_index_t eN,
					       const int ebN_local,
					       const int kb,
					       const int ebN_local_kb,
					       double* mesh_dof,
					       proteus_index_t* mesh_l2g,
					       double* mesh_trial_trace_ref,
					       double* mesh_grad_trial_trace_ref,
					       double* boundaryJac_ref,
					       double* jac,
					       double& jacDet,
					       double* jacInv,
					       double* boundaryJac,
					       double* metricTensor,
					       double& metricTensorDetSqrt,
					       double* normal_ref,
					       double* normal,
					       double& x,
					       double& y)
  {
    calculateMapping_elementBoundary(eN,
				     ebN_local,
				     kb,
				     ebN_local_kb,
				     mesh_dof,
				     mesh_l2g,
				     mesh_trial_trace_ref,
				     mesh_grad_trial_trace_ref,
				     boundaryJac_ref,
				     jac,
				     jacDet,
				     jacInv,
				     boundaryJac,
				     metricTensor,
				     metricTensorDetSqrt,
				     normal_ref,
				     normal,
				     x,
				     y,
				     0.0);  
  }
  inline void calculateMappingVelocity_elementBoundary(const proteus_index_t eN,
						       const int ebN_local,
						       const int kb,
						       const int ebN_local_kb,
						       double* mesh_velocity_dof,
						       proteus_index_t* mesh_l2g,
						       double* mesh_trial_trace_ref,
						       double& xt,
						       double& yt,
						       double* normal,
						       double* boundaryJac,						       
						       double* metricTensor,
						       double& metricTensorDetSqrt)
  {
    calculateMappingVelocity_elementBoundary(eN,
					     ebN_local,
					     kb,
					     ebN_local_kb,
					     mesh_velocity_dof,
					     mesh_l2g,
					     mesh_trial_trace_ref,
					     xt,
					     yt,
					     0.0,
					     normal,
					     boundaryJac,						       
					     metricTensor,
					     metricTensorDetSqrt);
  }

};

//specialization for 2D
template<int NDOF_MESH_TRIAL_ELEMENT>
class CompKernelSpaceMapping<2,NDOF_MESH_TRIAL_ELEMENT>
{
public:
  const int X,Y,
    XX,XY,
    YX,YY,
    sXX,sXY,
    sYX,sYY,
    nSymTen,
    XHX,
    YHX,
    HXHX;
  CompKernelSpaceMapping():
    X(0),
    Y(1),
    XX(0),XY(1),
    YX(2),YY(3),
    sXX(0),sXY(2),
    sYX(2),sYY(1),
    nSymTen(3),
    XHX(0),
    YHX(1),
    HXHX(0)
  {}
  inline void calculateMapping_element(const proteus_index_t eN,
				       const int k,
				       double* mesh_dof,
				       proteus_index_t* mesh_l2g,
				       //proteus::pyarray<double>& mesh_trial_ref,
				       double* mesh_trial_ref,
				       double* mesh_grad_trial_ref,
				       double* jac,
				       double& jacDet,
				       double* jacInv,
				       double& x,
				       double& y)
  {
    double Grad_x[2],Grad_y[2],oneOverJacDet;
    
    //
    //mapping of reference element to physical element
    //
    x=0.0;y=0.0;
    for (int I=0;I<2;I++)
      {
	Grad_x[I]=0.0;Grad_y[I]=0.0;
      }
    for (int j=0;j<NDOF_MESH_TRIAL_ELEMENT;j++)
      {
	proteus_index_t eN_j=eN*NDOF_MESH_TRIAL_ELEMENT+j;
	/* x += mesh_dof[mesh_l2g[eN_j]*2+0]*mesh_trial_ref[k*NDOF_MESH_TRIAL_ELEMENT+j]; */
	/* y += mesh_dof[mesh_l2g[eN_j]*2+1]*mesh_trial_ref[k*NDOF_MESH_TRIAL_ELEMENT+j]; */
	/* for (int I=0;I<2;I++) */
	/*   { */
	/*     Grad_x[I] += mesh_dof[mesh_l2g[eN_j]*2+0]*mesh_grad_trial_ref[k*NDOF_MESH_TRIAL_ELEMENT*2+j*2+I]; */
	/*     Grad_y[I] += mesh_dof[mesh_l2g[eN_j]*2+1]*mesh_grad_trial_ref[k*NDOF_MESH_TRIAL_ELEMENT*2+j*2+I]; */
	/*   } */
	//x += mesh_dof[mesh_l2g[eN_j]*3+0]*mesh_trial_ref(k, j);
	//y += mesh_dof[mesh_l2g[eN_j]*3+1]*mesh_trial_ref(k, j);
        x += mesh_dof[mesh_l2g[eN_j]*3+0]*mesh_trial_ref[k*NDOF_MESH_TRIAL_ELEMENT+j];
	y += mesh_dof[mesh_l2g[eN_j]*3+1]*mesh_trial_ref[k*NDOF_MESH_TRIAL_ELEMENT+j];
	for (int I=0;I<2;I++)
	  {
	    Grad_x[I] += mesh_dof[mesh_l2g[eN_j]*3+0]*mesh_grad_trial_ref[k*NDOF_MESH_TRIAL_ELEMENT*2+j*2+I];
	    Grad_y[I] += mesh_dof[mesh_l2g[eN_j]*3+1]*mesh_grad_trial_ref[k*NDOF_MESH_TRIAL_ELEMENT*2+j*2+I];
	  }
      }
    jac[XX] = Grad_x[X];
    jac[XY] = Grad_x[Y];
    jac[YX] = Grad_y[X];
    jac[YY] = Grad_y[Y];
    jacDet = jac[XX]*jac[YY] - jac[XY]*jac[YX];
    oneOverJacDet = 1.0/jacDet;
    jacInv[XX] = oneOverJacDet*jac[YY];
    jacInv[XY] = -oneOverJacDet*jac[XY];
    jacInv[YX] = -oneOverJacDet*jac[YX];
    jacInv[YY] = oneOverJacDet*jac[XX];
  }
  
  inline void calculateH_element(const proteus_index_t eN,
				 const int k,
				 double* h_dof,
				 proteus_index_t* mesh_l2g,
				 //proteus::pyarray<double>& mesh_trial_ref,
				 double* mesh_trial_ref,
				 double& h)
  {
    h=0.0;
    for (int j=0;j<NDOF_MESH_TRIAL_ELEMENT;j++)
      {
	proteus_index_t eN_j=eN*NDOF_MESH_TRIAL_ELEMENT+j;
	//h += h_dof[mesh_l2g[eN_j]]*mesh_trial_ref(k, j);
        h += h_dof[mesh_l2g[eN_j]]*mesh_trial_ref[k*NDOF_MESH_TRIAL_ELEMENT+j];
      }
  }
  
  inline void calculateMappingVelocity_element(const proteus_index_t eN,
					       const int k,
					       double* mesh_velocity_dof,
					       proteus_index_t* mesh_l2g,
					       //proteus::pyarray<double>& mesh_trial_ref,
					       double* mesh_trial_ref,
					       double& xt,
					       double& yt)
  {
    //
    //time derivative of mapping of reference element to physical element
    //
    xt=0.0;yt=0.0;
    for (int j=0;j<NDOF_MESH_TRIAL_ELEMENT;j++)
      {
	proteus_index_t eN_j=eN*NDOF_MESH_TRIAL_ELEMENT+j;
	/* xt += mesh_velocity_dof[mesh_l2g[eN_j]*2+0]*mesh_trial_ref[k*NDOF_MESH_TRIAL_ELEMENT+j]; */
	/* yt += mesh_velocity_dof[mesh_l2g[eN_j]*2+1]*mesh_trial_ref[k*NDOF_MESH_TRIAL_ELEMENT+j]; */
	//xt += mesh_velocity_dof[mesh_l2g[eN_j]*3+0]*mesh_trial_ref(k, j);
	//yt += mesh_velocity_dof[mesh_l2g[eN_j]*3+1]*mesh_trial_ref(k, j);
        xt += mesh_velocity_dof[mesh_l2g[eN_j]*3+0]*mesh_trial_ref[k*NDOF_MESH_TRIAL_ELEMENT+j];
	yt += mesh_velocity_dof[mesh_l2g[eN_j]*3+1]*mesh_trial_ref[k*NDOF_MESH_TRIAL_ELEMENT+j];
      }
  }

  inline void calculateMapping_elementBoundary(const proteus_index_t eN,
					       const int ebN_local,
					       const int kb,
					       const int ebN_local_kb,
					       double* mesh_dof,
					       proteus_index_t* mesh_l2g,
					       double* mesh_trial_trace_ref,
					       double* mesh_grad_trial_trace_ref,
					       double* boundaryJac_ref,
					       double* jac,
					       double& jacDet,
					       double* jacInv,
					       double* boundaryJac,
					       double* metricTensor,
					       double& metricTensorDetSqrt,
					       double* normal_ref,
					       double* normal,
					       double& x,
					       double& y)
  {
    const int ebN_local_kb_nSpace = ebN_local_kb*2,
      ebN_local_kb_nSpace_nSpacem1 = ebN_local_kb*2*1;
  
    double Grad_x_ext[2],Grad_y_ext[2],oneOverJacDet,norm_normal=0.0;
    // 
    //calculate mapping from the reference element to the physical element
    // 
    x=0.0;y=0.0;
    for (int I=0;I<2;I++)
      {
	Grad_x_ext[I] = 0.0;
	Grad_y_ext[I] = 0.0;
      }
    for (int j=0;j<NDOF_MESH_TRIAL_ELEMENT;j++) 
      { 
	proteus_index_t eN_j = eN*NDOF_MESH_TRIAL_ELEMENT+j;
	int ebN_local_kb_j = ebN_local_kb*NDOF_MESH_TRIAL_ELEMENT+j;
	int ebN_local_kb_j_nSpace = ebN_local_kb_j*2;
	/* x += mesh_dof[mesh_l2g[eN_j]*2+0]*mesh_trial_trace_ref[ebN_local_kb_j];  */
	/* y += mesh_dof[mesh_l2g[eN_j]*2+1]*mesh_trial_trace_ref[ebN_local_kb_j];  */
	/* for (int I=0;I<2;I++) */
	/*   { */
	/*     Grad_x_ext[I] += mesh_dof[mesh_l2g[eN_j]*2+0]*mesh_grad_trial_trace_ref[ebN_local_kb_j_nSpace+I]; */
	/*     Grad_y_ext[I] += mesh_dof[mesh_l2g[eN_j]*2+1]*mesh_grad_trial_trace_ref[ebN_local_kb_j_nSpace+I];  */
	/*   }  */
	x += mesh_dof[mesh_l2g[eN_j]*3+0]*mesh_trial_trace_ref[ebN_local_kb_j]; 
	y += mesh_dof[mesh_l2g[eN_j]*3+1]*mesh_trial_trace_ref[ebN_local_kb_j]; 
	for (int I=0;I<2;I++)
	  {
	    Grad_x_ext[I] += mesh_dof[mesh_l2g[eN_j]*3+0]*mesh_grad_trial_trace_ref[ebN_local_kb_j_nSpace+I];
	    Grad_y_ext[I] += mesh_dof[mesh_l2g[eN_j]*3+1]*mesh_grad_trial_trace_ref[ebN_local_kb_j_nSpace+I]; 
	  } 
      }
    //Space Mapping Jacobian
    jac[XX] = Grad_x_ext[X];
    jac[XY] = Grad_x_ext[Y];
    jac[YX] = Grad_y_ext[X];
    jac[YY] = Grad_y_ext[Y];
    jacDet =  jac[XX]*jac[YY] - jac[XY]*jac[YX]; 
    oneOverJacDet = 1.0/jacDet;
    jacInv[XX] = oneOverJacDet*jac[YY];
    jacInv[XY] = -oneOverJacDet*jac[XY];
    jacInv[YX] = -oneOverJacDet*jac[YX];
    jacInv[YY] = oneOverJacDet*jac[XX];
    //normal
    norm_normal=0.0;
    for (int I=0;I<2;I++)
      normal[I] = 0.0;
    for (int I=0;I<2;I++)
      {
	for (int J=0;J<2;J++)
	  {
	    normal[I] += jacInv[J*2+I]*normal_ref[ebN_local_kb_nSpace+J];
	  }
	norm_normal+=normal[I]*normal[I];
      }
    norm_normal = sqrt(norm_normal);
    for (int I=0;I<2;I++)
      {
	normal[I] /= norm_normal;
      }
    //metric tensor and determinant
    boundaryJac[XHX] = jac[XX]*boundaryJac_ref[ebN_local_kb_nSpace_nSpacem1+XHX]+jac[XY]*boundaryJac_ref[ebN_local_kb_nSpace_nSpacem1+YHX];
    boundaryJac[YHX] = jac[YX]*boundaryJac_ref[ebN_local_kb_nSpace_nSpacem1+XHX]+jac[YY]*boundaryJac_ref[ebN_local_kb_nSpace_nSpacem1+YHX];
  
    metricTensor[HXHX] = boundaryJac[XHX]*boundaryJac[XHX]+boundaryJac[YHX]*boundaryJac[YHX];
  
    metricTensorDetSqrt=sqrt(metricTensor[HXHX]);
  }

  inline void calculateMappingVelocity_elementBoundary(const proteus_index_t eN,
						       const int ebN_local,
						       const int kb,
						       const int ebN_local_kb,
						       double* mesh_velocity_dof,
						       proteus_index_t* mesh_l2g,
						       double* mesh_trial_trace_ref,
						       double& xt,
						       double& yt,
						       double* normal,
						       double* boundaryJac,						       
						       double* metricTensor,
						       double& metricTensorDetSqrt)
  {
    //const int ebN_local_kb_nSpace = ebN_local_kb*2,
    //  ebN_local_kb_nSpace_nSpacem1 = ebN_local_kb*2*2;
    // 
    //calculate velocity of mapping from the reference element to the physical element
    // 
    xt=0.0;yt=0.0;
    for (int j=0;j<NDOF_MESH_TRIAL_ELEMENT;j++) 
      { 
	proteus_index_t eN_j = eN*NDOF_MESH_TRIAL_ELEMENT+j;
	int ebN_local_kb_j = ebN_local_kb*NDOF_MESH_TRIAL_ELEMENT+j;
	/* xt += mesh_velocity_dof[mesh_l2g[eN_j]*2+0]*mesh_trial_trace_ref[ebN_local_kb_j];  */
	/* yt += mesh_velocity_dof[mesh_l2g[eN_j]*2+1]*mesh_trial_trace_ref[ebN_local_kb_j];  */
	xt += mesh_velocity_dof[mesh_l2g[eN_j]*3+0]*mesh_trial_trace_ref[ebN_local_kb_j]; 
	yt += mesh_velocity_dof[mesh_l2g[eN_j]*3+1]*mesh_trial_trace_ref[ebN_local_kb_j]; 
      }
    //modify the metricTensorDetSqrt to include the effect of the moving domain
    //it's not exactly the Sqrt(Det(G_tr_G)) now, see notes
    //just do it brute force
    double 
      Gy_tr_Gy_00 = 1.0 + xt*xt + yt*yt,
      Gy_tr_Gy_01 = boundaryJac[XHX]*xt+boundaryJac[YHX]*yt,
      Gy_tr_Gy_10 = Gy_tr_Gy_01,
      Gy_tr_Gy_11 = metricTensor[HXHX],
      xt_dot_n = xt*normal[X]+yt*normal[Y];
    metricTensorDetSqrt=sqrt((Gy_tr_Gy_00*Gy_tr_Gy_11 - Gy_tr_Gy_01*Gy_tr_Gy_10) / (1.0+xt_dot_n*xt_dot_n));
  }
  inline void valFromDOF(const double* dof,const proteus_index_t* l2g_element,const double* trial_ref,double& val)
  {
    val=0.0;
    for (int j=0;j<NDOF_MESH_TRIAL_ELEMENT;j++)
      val+=dof[l2g_element[j]]*trial_ref[j];
  }
  
  inline void gradFromDOF(const double* dof,const proteus_index_t* l2g_element,const double* grad_trial,double* grad)
  {
    for(int I=0;I<2;I++)
      grad[I] = 0.0;
    for (int j=0;j<NDOF_MESH_TRIAL_ELEMENT;j++)
      for(int I=0;I<2;I++)
	grad[I] += dof[l2g_element[j]]*grad_trial[j*2+I];
  }

  inline void hessFromDOF(const double* dof,const proteus_index_t* l2g_element,const double* hess_trial,double* hess)
  {
    for(int I=0;I<2;I++)
      for(int J=0;J<2;J++)
	hess[I*2+J] = 0.0;
    for (int j=0;j<NDOF_MESH_TRIAL_ELEMENT;j++)
      for(int I=0;I<2;I++)
	for(int J=0;J<2;J++)
	  hess[I*2+J] += dof[l2g_element[j]]*hess_trial[j*4+I*2+J];
  }
  
  inline void calculateMapping_element(const proteus_index_t eN,
				       const int k,
				       double* mesh_dof,
				       proteus_index_t* mesh_l2g,
				       //proteus::pyarray<double>& mesh_trial_ref,
				       double* mesh_trial_ref,
				       double* mesh_grad_trial_ref,
				       double* jac,
				       double& jacDet,
				       double* jacInv,
				       double& x,
				       double& y,
				       double& z)
  {
    calculateMapping_element(eN,
			     k,
			     mesh_dof,
			     mesh_l2g,
			     mesh_trial_ref,
			     mesh_grad_trial_ref,
			     jac,
			     jacDet,
			     jacInv,
			     x,
			     y);
  }
  inline void calculateMappingVelocity_element(const proteus_index_t eN,
					       const int k,
					       double* mesh_velocity_dof,
					       proteus_index_t* mesh_l2g,
					       //proteus::pyarray<double>& mesh_trial_ref,
					       double* mesh_trial_ref,
					       double& xt,
					       double& yt,
					       double& zt)
  {
    calculateMappingVelocity_element(eN,
				     k,
				     mesh_velocity_dof,
				     mesh_l2g,
				     mesh_trial_ref,
				     xt,
				     yt);
  }
  inline void calculateMapping_elementBoundary(const proteus_index_t eN,
					       const int ebN_local,
					       const int kb,
					       const int ebN_local_kb,
					       double* mesh_dof,
					       proteus_index_t* mesh_l2g,
					       double* mesh_trial_trace_ref,
					       double* mesh_grad_trial_trace_ref,
					       double* boundaryJac_ref,
					       double* jac,
					       double& jacDet,
					       double* jacInv,
					       double* boundaryJac,
					       double* metricTensor,
					       double& metricTensorDetSqrt,
					       double* normal_ref,
					       double* normal,
					       double& x,
					       double& y,
					       double& z)
  {
    calculateMapping_elementBoundary(eN,
				     ebN_local,
				     kb,
				     ebN_local_kb,
				     mesh_dof,
				     mesh_l2g,
				     mesh_trial_trace_ref,
				     mesh_grad_trial_trace_ref,
				     boundaryJac_ref,
				     jac,
				     jacDet,
				     jacInv,
				     boundaryJac,
				     metricTensor,
				     metricTensorDetSqrt,
				     normal_ref,
				     normal,
				     x,
				     y);
    //A 2D mapping has no z component, but z is an OUTPUT parameter the caller will
    //read, so it must be assigned: leaving it alone hands back whatever the caller's
    //stack slot happened to hold.
    z=0.0;
  }
  inline void calculateMappingVelocity_elementBoundary(const proteus_index_t eN,
						       const int ebN_local,
						       const int kb,
						       const int ebN_local_kb,
						       double* mesh_velocity_dof,
						       proteus_index_t* mesh_l2g,
						       double* mesh_trial_trace_ref,
						       double& xt,
						       double& yt,
						       double& zt,
						       double* normal,
						       double* boundaryJac,						       
						       double* metricTensor,
						       double& metricTensorDetSqrt)
  {
    calculateMappingVelocity_elementBoundary(eN,
					     ebN_local,
					     kb,
					     ebN_local_kb,
					     mesh_velocity_dof,
					     mesh_l2g,
					     mesh_trial_trace_ref,
					     xt,
					     yt,
					     normal,
					     boundaryJac,						       
					     metricTensor,
					     metricTensorDetSqrt);
  }
};

//specialization for 1D
template<int NDOF_MESH_TRIAL_ELEMENT>
class CompKernelSpaceMapping<1,NDOF_MESH_TRIAL_ELEMENT>
{
public:
  const int X,
    XX,
    sXX,
    nSymTen,
    XHX,
    HXHX;
  CompKernelSpaceMapping():
    X(0),
    XX(0),
    sXX(0),
    nSymTen(1),
    XHX(0),
    HXHX(0)
  {}
  inline void calculateMapping_element(const proteus_index_t eN,
				       const int k,
				       double* mesh_dof,
				       proteus_index_t* mesh_l2g,
				       //proteus::pyarray<double>& mesh_trial_ref,
				       double* mesh_trial_ref,
				       double* mesh_grad_trial_ref,
				       double* jac,
				       double& jacDet,
				       double* jacInv,
				       double& x)
  {
    double Grad_x[1],Grad_y[1],oneOverJacDet;
    
    //
    //mapping of reference element to physical element
    //
    x=0.0;
    for (int I=0;I<1;I++)
      {
	Grad_x[I]=0.0;
      }
    for (int j=0;j<NDOF_MESH_TRIAL_ELEMENT;j++)
      {
	proteus_index_t eN_j=eN*NDOF_MESH_TRIAL_ELEMENT+j;
        x += mesh_dof[mesh_l2g[eN_j]*3+0]*mesh_trial_ref[k*NDOF_MESH_TRIAL_ELEMENT+j];
	for (int I=0;I<1;I++)
	  {
	    Grad_x[I] += mesh_dof[mesh_l2g[eN_j]*3+0]*mesh_grad_trial_ref[k*NDOF_MESH_TRIAL_ELEMENT+j+I];
	  }
      }
    jac[XX] = Grad_x[X];
    jacDet = jac[XX];
    oneOverJacDet = 1.0/jacDet;
    jacInv[XX] = oneOverJacDet;
  }
  
  inline void calculateH_element(const proteus_index_t eN,
				 const int k,
				 double* h_dof,
				 proteus_index_t* mesh_l2g,
				 //proteus::pyarray<double>& mesh_trial_ref,
				 double* mesh_trial_ref,
				 double& h)
  {
    h=0.0;
    for (int j=0;j<NDOF_MESH_TRIAL_ELEMENT;j++)
      {
	proteus_index_t eN_j=eN*NDOF_MESH_TRIAL_ELEMENT+j;
        h += h_dof[mesh_l2g[eN_j]]*mesh_trial_ref[k*NDOF_MESH_TRIAL_ELEMENT+j];
      }
  }
  
  inline void calculateMappingVelocity_element(const proteus_index_t eN,
					       const int k,
					       double* mesh_velocity_dof,
					       proteus_index_t* mesh_l2g,
					       //proteus::pyarray<double>& mesh_trial_ref,
					       double* mesh_trial_ref,
					       double& xt)
  {
    //
    //time derivative of mapping of reference element to physical element
    //
    xt=0.0;
    for (int j=0;j<NDOF_MESH_TRIAL_ELEMENT;j++)
      {
	proteus_index_t eN_j=eN*NDOF_MESH_TRIAL_ELEMENT+j;
        xt += mesh_velocity_dof[mesh_l2g[eN_j]*3+0]*mesh_trial_ref[k*NDOF_MESH_TRIAL_ELEMENT+j];
      }
  }

  inline void calculateMapping_elementBoundary(const proteus_index_t eN,
					       const int ebN_local,
					       const int kb,
					       const int ebN_local_kb,
					       double* mesh_dof,
					       proteus_index_t* mesh_l2g,
					       double* mesh_trial_trace_ref,
					       double* mesh_grad_trial_trace_ref,
					       double* boundaryJac_ref,
					       double* jac,
					       double& jacDet,
					       double* jacInv,
					       double* boundaryJac,
					       double* metricTensor,
					       double& metricTensorDetSqrt,
					       double* normal_ref,
					       double* normal,
					       double& x)
  {
    const int ebN_local_kb_nSpace = ebN_local_kb*1;
  
    double Grad_x_ext[1],oneOverJacDet,norm_normal=0.0;
    // 
    //calculate mapping from the reference element to the physical element
    // 
    x=0.0;
    for (int I=0;I<1;I++)
      {
	Grad_x_ext[I] = 0.0;
      }
    for (int j=0;j<NDOF_MESH_TRIAL_ELEMENT;j++) 
      { 
	proteus_index_t eN_j = eN*NDOF_MESH_TRIAL_ELEMENT+j;
	int ebN_local_kb_j = ebN_local_kb*NDOF_MESH_TRIAL_ELEMENT+j;
	int ebN_local_kb_j_nSpace = ebN_local_kb_j*1;
	x += mesh_dof[mesh_l2g[eN_j]*3+0]*mesh_trial_trace_ref[ebN_local_kb_j]; 
	for (int I=0;I<1;I++)
	  {
	    Grad_x_ext[I] += mesh_dof[mesh_l2g[eN_j]*3+0]*mesh_grad_trial_trace_ref[ebN_local_kb_j_nSpace+I];
	  } 
      }
    //Space Mapping Jacobian
    jac[XX] = Grad_x_ext[X];
    jacDet =  jac[XX];
    oneOverJacDet = 1.0/jacDet;
    jacInv[XX] = oneOverJacDet;
    //normal
    norm_normal=0.0;
    for (int I=0;I<1;I++)
      normal[I] = 0.0;
    for (int I=0;I<1;I++)
      {
	for (int J=0;J<1;J++)
	  {
	    normal[I] += jacInv[J+I]*normal_ref[ebN_local_kb_nSpace+J];
	  }
	norm_normal+=normal[I]*normal[I];
      }
    norm_normal = sqrt(norm_normal);
    for (int I=0;I<1;I++)
      {
	normal[I] /= norm_normal;
      }
    //metric tensor and determinant
    boundaryJac[XHX] = 1.0;
  
    metricTensor[HXHX] = 1.0;
  
    metricTensorDetSqrt=1.0;
  }

  inline void calculateMappingVelocity_elementBoundary(const proteus_index_t eN,
						       const int ebN_local,
						       const int kb,
						       const int ebN_local_kb,
						       double* mesh_velocity_dof,
						       proteus_index_t* mesh_l2g,
						       double* mesh_trial_trace_ref,
						       double& xt,
						       double* normal,
						       double* boundaryJac,						       
						       double* metricTensor,
						       double& metricTensorDetSqrt)
  {
    // 
    //calculate velocity of mapping from the reference element to the physical element
    // 
    xt=0.0;
    for (int j=0;j<NDOF_MESH_TRIAL_ELEMENT;j++) 
      { 
	proteus_index_t eN_j = eN*NDOF_MESH_TRIAL_ELEMENT+j;
	int ebN_local_kb_j = ebN_local_kb*NDOF_MESH_TRIAL_ELEMENT+j;
	xt += mesh_velocity_dof[mesh_l2g[eN_j]*3+0]*mesh_trial_trace_ref[ebN_local_kb_j]; 
      }
    //modify the metricTensorDetSqrt to include the effect of the moving domain
    //it's not exactly the Sqrt(Det(G_tr_G)) now, see notes
    //just do it brute force
    double 
      Gy_tr_Gy_00 = 1.0 + xt*xt,
      Gy_tr_Gy_01 = boundaryJac[XHX]*xt,
      Gy_tr_Gy_10 = Gy_tr_Gy_01,
      Gy_tr_Gy_11 = metricTensor[HXHX],
      xt_dot_n = xt*normal[X];
    metricTensorDetSqrt=sqrt((Gy_tr_Gy_00*Gy_tr_Gy_11 - Gy_tr_Gy_01*Gy_tr_Gy_10) / (1.0+xt_dot_n*xt_dot_n));
  }
  inline void valFromDOF(const double* dof,const proteus_index_t* l2g_element,const double* trial_ref,double& val)
  {
    val=0.0;
    for (int j=0;j<NDOF_MESH_TRIAL_ELEMENT;j++)
      val+=dof[l2g_element[j]]*trial_ref[j];
  }
  
  inline void gradFromDOF(const double* dof,const proteus_index_t* l2g_element,const double* grad_trial,double* grad)
  {
    for(int I=0;I<1;I++)
      grad[I] = 0.0;
    for (int j=0;j<NDOF_MESH_TRIAL_ELEMENT;j++)
      for(int I=0;I<1;I++)
	grad[I] += dof[l2g_element[j]]*grad_trial[j+I];
  }

  inline void hessFromDOF(const double* dof,const proteus_index_t* l2g_element,const double* hess_trial,double* hess)
  {
    for(int I=0;I<1;I++)
      for(int J=0;J<1;J++)
	hess[I+J] = 0.0;
    for (int j=0;j<NDOF_MESH_TRIAL_ELEMENT;j++)
      for(int I=0;I<1;I++)
	for(int J=0;J<1;J++)
	  hess[I+J] += dof[l2g_element[j]]*hess_trial[j+I+J];
  }
  
  inline void calculateMapping_element(const proteus_index_t eN,
				       const int k,
				       double* mesh_dof,
				       proteus_index_t* mesh_l2g,
				       //proteus::pyarray<double>& mesh_trial_ref,
				       double* mesh_trial_ref,
				       double* mesh_grad_trial_ref,
				       double* jac,
				       double& jacDet,
				       double* jacInv,
				       double& x,
				       double& y,
				       double& z)
  {
    calculateMapping_element(eN,
			     k,
			     mesh_dof,
			     mesh_l2g,
			     mesh_trial_ref,
			     mesh_grad_trial_ref,
			     jac,
			     jacDet,
			     jacInv,
			     x);
  }
  inline void calculateMappingVelocity_element(const proteus_index_t eN,
					       const int k,
					       double* mesh_velocity_dof,
					       proteus_index_t* mesh_l2g,
					       //proteus::pyarray<double>& mesh_trial_ref,
					       double* mesh_trial_ref,
					       double& xt,
					       double& yt,
					       double& zt)
  {
    calculateMappingVelocity_element(eN,
				     k,
				     mesh_velocity_dof,
				     mesh_l2g,
				     mesh_trial_ref,
				     xt);
  }
  inline void calculateMapping_elementBoundary(const proteus_index_t eN,
					       const int ebN_local,
					       const int kb,
					       const int ebN_local_kb,
					       double* mesh_dof,
					       proteus_index_t* mesh_l2g,
					       double* mesh_trial_trace_ref,
					       double* mesh_grad_trial_trace_ref,
					       double* boundaryJac_ref,
					       double* jac,
					       double& jacDet,
					       double* jacInv,
					       double* boundaryJac,
					       double* metricTensor,
					       double& metricTensorDetSqrt,
					       double* normal_ref,
					       double* normal,
					       double& x,
					       double& y,
					       double& z)
  {
    calculateMapping_elementBoundary(eN,
				     ebN_local,
				     kb,
				     ebN_local_kb,
				     mesh_dof,
				     mesh_l2g,
				     mesh_trial_trace_ref,
				     mesh_grad_trial_trace_ref,
				     boundaryJac_ref,
				     jac,
				     jacDet,
				     jacInv,
				     boundaryJac,
				     metricTensor,
				     metricTensorDetSqrt,
				     normal_ref,
				     normal,
				     x);
    //likewise for a 1D mapping: y and z have no counterpart in the mapping but are
    //outputs the caller will read.
    y=0.0;
    z=0.0;
  }
  inline void calculateMappingVelocity_elementBoundary(const proteus_index_t eN,
						       const int ebN_local,
						       const int kb,
						       const int ebN_local_kb,
						       double* mesh_velocity_dof,
						       proteus_index_t* mesh_l2g,
						       double* mesh_trial_trace_ref,
						       double& xt,
						       double& yt,
						       double& zt,
						       double* normal,
						       double* boundaryJac,						       
						       double* metricTensor,
						       double& metricTensorDetSqrt)
  {
    calculateMappingVelocity_elementBoundary(eN,
					     ebN_local,
					     kb,
					     ebN_local_kb,
					     mesh_velocity_dof,
					     mesh_l2g,
					     mesh_trial_trace_ref,
					     xt,
					     normal,
					     boundaryJac,						       
					     metricTensor,
					     metricTensorDetSqrt);
  }
};


template<int NSPACE, int NDOF_MESH_TRIAL_ELEMENT, int NDOF_TRIAL_ELEMENT, int NDOF_TEST_ELEMENT>
class CompKernel
{
public:
  CompKernelSpaceMapping<NSPACE,NDOF_MESH_TRIAL_ELEMENT> mapping;
  const int X,
    Y,
    Z,
    XX,XY,XZ,
    YX,YY,YZ,
    ZX,ZY,ZZ,
    sXX,sXY,sXZ,
    sYX,sYY,sYZ,
    sZX,sZY,sZZ,
    nSymTen,
    XHX,XHY,
    YHX,YHY,
    ZHX,ZHY,
    HXHX,HXHY,
    HYHX,HYHY;
  CompKernel():
    X(mapping.X),
    Y(mapping.Y),
    Z(mapping.Z),
    XX(mapping.XX),XY(mapping.XY),XZ(mapping.XZ),
    YX(mapping.YX),YY(mapping.YY),YZ(mapping.YZ),
    ZX(mapping.ZX),ZY(mapping.ZY),ZZ(mapping.ZZ),
    sXX(mapping.sXX),sXY(mapping.sXY),sXZ(mapping.sXZ),
    sYX(mapping.sYX),sYY(mapping.sYY),sYZ(mapping.sYZ),
    sZX(mapping.sZX),sZY(mapping.sZY),sZZ(mapping.sZZ),
    nSymTen(mapping.nSymTen),
    XHX(mapping.XHX),XHY(mapping.XHY),
    YHX(mapping.YHX),YHY(mapping.YHY),
    ZHX(mapping.ZHX),ZHY(mapping.ZHY),
    HXHX(mapping.HXHX),HXHY(mapping.HXHY),
    HYHX(mapping.HYHX),HYHY(mapping.HYHY)
  {}
  inline void calculateG(double* jacInv,double* G,double& G_dd_G, double& tr_G)
    {
      //get the metric tensor
      //cek todo use symmetry
      for (int I=0;I<NSPACE;I++)
	for (int J=0;J<NSPACE;J++)
	  {
	    G[I*NSPACE+J] = 0.0;
	    for (int K=0;K<NSPACE;K++)
	      G[I*NSPACE+J] += jacInv[K*NSPACE+I]*jacInv[K*NSPACE+J];
	  }
      G_dd_G = 0.0;
      tr_G = 0.0;
      for (int I=0;I<NSPACE;I++)
	{
	  tr_G += G[I*NSPACE+I];
	  for (int J=0;J<NSPACE;J++)
	    {
	      G_dd_G += G[I*NSPACE+J]*G[I*NSPACE+J];
	    }
	}
    }
  inline void calculateGScale(double* G,double* v,double& h)
  {
    h = 0.0;
    for (int I=0;I<NSPACE;I++)
      for (int J=0;J<NSPACE;J++)
	h += v[I]*G[I*NSPACE+J]*v[J];
    h = 1.0/sqrt(h+1.0e-16);//cek hack
  }
  inline void valFromDOF(const double* dof,const proteus_index_t* l2g_element,const double* trial_ref,double& val)
  {
    val=0.0;
    for (int j=0;j<NDOF_TRIAL_ELEMENT;j++)
      val+=dof[l2g_element[j]]*trial_ref[j];
  }

  inline void gradFromDOF(const double* dof,const proteus_index_t* l2g_element,const double* grad_trial,double* grad)
  {
    for(int I=0;I<NSPACE;I++)
      grad[I] = 0.0;
    for (int j=0;j<NDOF_TRIAL_ELEMENT;j++)
      for(int I=0;I<NSPACE;I++)
	grad[I] += dof[l2g_element[j]]*grad_trial[j*NSPACE+I];
  }

  inline void hessFromDOF(const double* dof,const proteus_index_t* l2g_element,const double* hess_trial,double* hess)
  {
    const int NSPACE2=NSPACE*NSPACE;
    for(int I=0;I<NSPACE;I++)
      for(int J=0;J<NSPACE;J++)
	hess[I*NSPACE+J] = 0.0;
    for (int j=0;j<NDOF_TRIAL_ELEMENT;j++)
      for(int I=0;I<NSPACE;I++)
	for(int J=0;J<NSPACE;J++)
	  hess[I*NSPACE+J] += dof[l2g_element[j]]*hess_trial[j*NSPACE2+I*NSPACE+J];
  }

  inline void valFromElementDOF(const double* dof,const double* trial_ref,double& val)
  {
    val=0.0;
    for (int j=0;j<NDOF_TRIAL_ELEMENT;j++)
      val+=dof[j]*trial_ref[j];
  }

  inline void gradFromElementDOF(const double* dof,const double* grad_trial,double* grad)
  {
    for(int I=0;I<NSPACE;I++)
      grad[I] = 0.0;
    for (int j=0;j<NDOF_TRIAL_ELEMENT;j++)
      for(int I=0;I<NSPACE;I++)
	grad[I] += dof[j]*grad_trial[j*NSPACE+I];
  }

  inline void gradTrialFromRef(const double* grad_trial_ref, const double* jacInv, double* grad_trial)
  {
    for (int j=0;j<NDOF_TRIAL_ELEMENT;j++)
      for(int I=0;I<NSPACE;I++)
	grad_trial[j*NSPACE+I] = 0.0;
    for (int j=0;j<NDOF_TRIAL_ELEMENT;j++)
      for(int I=0;I<NSPACE;I++)
	for(int J=0;J<NSPACE;J++)
	  grad_trial[j*NSPACE+I] += jacInv[J*NSPACE+I]*grad_trial_ref[j*NSPACE+J];
  }
  
  /*
   *  DOFaverage
   *  ----------
   *
   *  Calculate the average DOF value for at a given mesh element.
   *
   *  @param dof array of finite element DOF values
   *  @param l2g_element local 2 global mapping for the current mesh element
   *  @param val return value with the average DOF values
   */

  inline void DOFaverage (const double* dof, const proteus_index_t* l2g_element, double& val)
  {
    val = 0.0;

    for (int j=0; j<NDOF_MESH_TRIAL_ELEMENT; j++)
      val+=dof[l2g_element[j]];

    val /= NDOF_MESH_TRIAL_ELEMENT;
  }
  
  inline void hessTrialFromRef(const double* hess_trial_ref, const double* jacInv, double* hess_trial)
  {
    const int NSPACE2=NSPACE*NSPACE;
    for (int j=0;j<NDOF_TRIAL_ELEMENT;j++)
      for(int I=0;I<NSPACE;I++)
	for(int J=0;J<NSPACE;J++)
	  hess_trial[j*NSPACE2+I*NSPACE+J] = 0.0;
    for (int j=0;j<NDOF_TRIAL_ELEMENT;j++)
      for(int I=0;I<NSPACE;I++)
	for(int J=0;J<NSPACE;J++)
	  for(int K=0;K<NSPACE;K++)
	    for(int L=0;L<NSPACE;L++)
	      hess_trial[j*NSPACE2+I*NSPACE+J] += hess_trial_ref[j*NSPACE2+K*NSPACE+L]*jacInv[L*NSPACE+J]*jacInv[K*NSPACE+I];
  }
  
  inline void gradTestFromRef(const double* grad_test_ref, const double* jacInv, double* grad_test)
  {
    for (int i=0;i<NDOF_TEST_ELEMENT;i++)
      for(int I=0;I<NSPACE;I++)
	grad_test[i*NSPACE+I] = 0.0;
    for (int i=0;i<NDOF_TEST_ELEMENT;i++)
      for(int I=0;I<NSPACE;I++)
	for(int J=0;J<NSPACE;J++)
	  grad_test[i*NSPACE+I] += jacInv[J*NSPACE+I]*grad_test_ref[i*NSPACE+J];
  }
  
  inline void backwardEuler(const double& dt, const double& m_old, const double& m, const double& dm, double& mt, double& dmt)
  {  
    mt =(m-m_old)/dt;
    dmt = dm/dt;
  }

  inline void bdf(const double& alpha, const double& beta, const double& m, const double& dm, double& mt, double& dmt)
  {  
    mt =alpha*m + beta;
    dmt = alpha*dm;
  }
  inline void bdfC2(const double& alpha, const double& beta, const double& m, const double& dm, const double& dm2, double& mt, double& dmt, double& dm2t)
  {  
    mt =alpha*m + beta;
    dmt = alpha*dm;
    dm2t = alpha*dm2;
  }

  inline double Mass_weak(const double& mt, const double& w_dV)
  {
    return mt*w_dV;
  }

  inline double MassJacobian_weak(const double& dmt,
				  const double& v,
				  const double& w_dV)
  {
    return dmt*v*w_dV;
  }

  inline double Mass_strong(const double& mt)
  {
    return mt; 
  }

  inline double MassJacobian_strong(const double& dmt,
				    const double& v)
  {
    return dmt*v;
  }

  inline double Mass_adjoint(const double& dmt,
			     const double& w_dV)
  {
    return dmt*w_dV;
  }

  /*
   *  pressureProjection_weak
   *  -----------------------
   *
   *  Inner product calculation of the pressure projection
   *  stablization method of Bochev, Dohrmann and
   *  Gunzburger (2006).
   *
   *  @param viscosity viscosity at point
   *  @param p the pressure value (either trial function
   *           or actual value)
   *  @param p_avg the pressure projection value (either
   *           1./3. for test functions of the average
   *	       value of the pressure on the element)
   *  @param dV this is the integral weight
   *
   */

  inline double pressureProjection_weak(const double& viscosity,
					const double& p,
					const double& p_avg,
					const double& q,
					const double& dV)
  {
    if (viscosity==0.){ return 0.;}
    return (1./viscosity)*(p-p_avg)*(q-1./3.)*dV;
  }
  
  inline double Advection_weak(const double  f[NSPACE],
			       const double grad_w_dV[NSPACE])
  {
    double tmp=0.0;
    for(int I=0;I<NSPACE;I++)
      tmp -= f[I]*grad_w_dV[I];
    return tmp;
  }

  inline double AdvectionJacobian_weak(const double df[NSPACE],
				       const double& v,
				       const double grad_w_dV[NSPACE])
  {
    double tmp=0.0;
    for(int I=0;I<NSPACE;I++)
      tmp -= df[I]*v*grad_w_dV[I];
    return tmp;
  }

  inline double Advection_strong(const double df[NSPACE],
				 const double grad_u[NSPACE])
  {
    double tmp=0.0;
    for(int I=0;I<NSPACE;I++)
      tmp += df[I]*grad_u[I];
    return tmp;
  }

  inline double AdvectionJacobian_strong(const double df[NSPACE],
					 const double grad_v[NSPACE])
  {
    double tmp=0.0;
    for(int I=0;I<NSPACE;I++)
      tmp += df[I]*grad_v[I];
    return tmp;
  }

  inline double Advection_adjoint(const double df[NSPACE],
				  const double grad_w_dV[NSPACE])
  {
    double tmp=0.0;
    for(int I=0;I<NSPACE;I++)
      tmp -= df[I]*grad_w_dV[I];
    return tmp;
  }

  inline double Hamiltonian_weak(const double& H,
				 const double& w_dV)
  {
    return H*w_dV;
  }

  inline double HamiltonianJacobian_weak(const double dH[NSPACE],
					 const double grad_v[NSPACE],
					 const double& w_dV)
  {
    double tmp=0.0;
    for(int I=0;I<NSPACE;I++)
      tmp += dH[I]*grad_v[I]*w_dV;
    return tmp;
  }

  inline double Hamiltonian_strong(const double dH[NSPACE],
				   const double grad_u[NSPACE])
  {
    double tmp=0.0;
    for(int I=0;I<NSPACE;I++)
      tmp += dH[I]*grad_u[I];
    return tmp;
  }

  inline double HamiltonianJacobian_strong(const double dH[NSPACE],
					   const double grad_v[NSPACE])
  {
    double tmp=0.0;
    for(int I=0;I<NSPACE;I++)
      tmp += dH[I]*grad_v[I];
    return tmp;
  }

  inline double Hamiltonian_adjoint(const double dH[NSPACE],
				    const double grad_w_dV[NSPACE])
  {
    double tmp=0.0;
    for(int I=0;I<NSPACE;I++)
      tmp -= dH[I]*grad_w_dV[I];
    return tmp;
  }

  inline double Diffusion_weak(proteus_csr_index_t* rowptr,
			       proteus_csr_index_t* colind,
			       double* a,
			       const double grad_phi[NSPACE],
			       const double grad_w_dV[NSPACE])
  {
    double tmp=0.0;
    for(int I=0;I<NSPACE;I++)
      for (proteus_csr_index_t m=rowptr[I];m<rowptr[I+1];m++)
	tmp += a[m]*grad_phi[colind[m]]*grad_w_dV[I];
    return tmp;
  }

  inline double DiffusionJacobian_weak(proteus_csr_index_t* rowptr,
				       proteus_csr_index_t* colind,
				       double* a,
				       double* da,
				       const double grad_phi[NSPACE],
				       const double grad_w_dV[NSPACE],
				       const double& dphi,
				       const double& v,
				       const double grad_v[NSPACE])
  {
    double daProduct=0.0,dphiProduct=0.0;
    for (int I=0;I<NSPACE;I++)
      for (proteus_csr_index_t m=rowptr[I];m<rowptr[I+1];m++)
	{
	  daProduct += da[m]*grad_phi[colind[m]]*grad_w_dV[I];
	  dphiProduct += a[m]*grad_v[colind[m]]*grad_w_dV[I];
	}
    return daProduct*v+dphiProduct*dphi;
  }

  inline double SimpleDiffusionJacobian_weak(proteus_csr_index_t* rowptr,
					     proteus_csr_index_t* colind,
					     double* a,
					     const double grad_v[NSPACE],
					     const double grad_w_dV[NSPACE])
  {
    double dphiProduct=0.0;
    for (int I=0;I<NSPACE;I++)
      for (proteus_csr_index_t m=rowptr[I];m<rowptr[I+1];m++)
	{
	  dphiProduct += a[m]*grad_v[colind[m]]*grad_w_dV[I];
	}
    return dphiProduct;
  }

  inline double Reaction_weak(const double& r,
			      const double& w_dV)
  {
    return r*w_dV;
  }
  
  inline double ReactionJacobian_weak(const double& dr,
				      const double& v,
				      const double& w_dV)
  {
    return dr*v*w_dV;
  }

  inline double Reaction_strong(const double& r)
  {
    return r; 
  }

  inline double ReactionJacobian_strong(const double& dr,
					const double& v)
  {
    return dr*v;
  }

  inline double Reaction_adjoint(const double& dr,
				 const double& w_dV)
  {
    return dr*w_dV;
  }
  
  inline void calculateNumericalDiffusion(const double& shockCapturingDiffusion,
					  const double& elementDiameter,
					  const double& strong_residual,
					  const double grad_u[NSPACE],
					  double& numDiff)
  {
    double h,
      num,
      den,
      n_grad_u;
    h = elementDiameter;
    n_grad_u = 0.0;
    for (int I=0;I<NSPACE;I++)
      n_grad_u += grad_u[I]*grad_u[I];
    num = shockCapturingDiffusion*0.5*h*fabs(strong_residual);
    den = sqrt(n_grad_u+1.0e-12);
    numDiff = num/den;
  }


  inline void calculateNumericalDiffusion(const double& shockCapturingDiffusion,
					  const double  G[NSPACE*NSPACE],
					  const double& strong_residual,
					  const double grad_u[NSPACE],
					  double& numDiff)
  {
    double  den = 0.0;
    for (int I=0;I<NSPACE;I++)
      for (int J=0;J<NSPACE;J++)
        den += grad_u[I]*G[I*NSPACE+J]*grad_u[J];
    numDiff = shockCapturingDiffusion*fabs(strong_residual)/(sqrt(den+1.0e-12));
  }

  inline void calculateNumericalDiffusion(const double& shockCapturingDiffusion,
                                          const double& uref, const double& beta,
					  const double  G[NSPACE*NSPACE],
					  const double& G_dd_G,
					  const double& strong_residual,
					  const double grad_u[NSPACE],
					  double& numDiff)
  {
    double  den = 0.0;    
    for (int I=0;I<NSPACE;I++)
      for (int J=0;J<NSPACE;J++)
        den += grad_u[I]*G[I*NSPACE+J]*grad_u[J];

    double h2_uref_1 = 1.0/(sqrt(den+1.0e-12));
    double h2_uref_2 = 1.0/(uref*sqrt(G_dd_G+1.0e-12));
    numDiff = shockCapturingDiffusion*fabs(strong_residual)*pow(h2_uref_1, 2.0-beta)*pow(h2_uref_2,beta-1.0);
  }



  inline void calculateNumericalDiffusion(const double& shockCapturingDiffusion,
					  const double  G[NSPACE*NSPACE],
					  const double& strong_residual,
					  const double  vel[NSPACE],
					  const double  grad_u[NSPACE],
					  double& numDiff)
  {
    double  den1 = 0.0,den2=0.0, nom=0.0;
    for (int I=0;I<NSPACE;I++)
      {
      nom += vel[I]*vel[I];
      den2+= grad_u[I]*grad_u[I]; 
      for (int J=0;J<NSPACE;J++)
        den1 += vel[I]*G[I*NSPACE+J]*vel[J];
      }
    numDiff = shockCapturingDiffusion*fabs(strong_residual)*(sqrt(nom/(den1*den2 + 1.0e-12)));
  }


  inline void calculateNumericalDiffusion(const double& shockCapturingDiffusion,
					  const double& elementDiameter,
					  const double& strong_residual,
					  const double grad_u[NSPACE],
					  double& gradNorm,
					  double& gradNorm_last,
					  double& numDiff)
  {
    double h,
      num,
      n_grad_u;
    h = elementDiameter;
    n_grad_u = 0.0;
    for (int I=0;I<NSPACE;I++)
      n_grad_u += grad_u[I]*grad_u[I];
    num = shockCapturingDiffusion*0.5*h*fabs(strong_residual);
    gradNorm = sqrt(n_grad_u+1.0e-12);
    //cek hack shockCapturingDiffusion*fabs(strong_residual)*grad_phi_G_grad_phi    
    numDiff = num/gradNorm_last;
  }

  inline double SubgridError(const double& error,
			     const double& Lstar_w_dV)
  {
    return error*Lstar_w_dV;
  }

  inline double SubgridErrorJacobian(const double& derror,
				     const double& Lstar_w_dV)
  {
    return derror*Lstar_w_dV;
  }

  inline double NumericalDiffusion(const double& numDiff,
				   const double grad_u[NSPACE],
				   const double grad_w_dV[NSPACE])
  {
    double tmp=0.0;
    for (int I=0;I<NSPACE;I++)
      tmp +=  numDiff*grad_u[I]*grad_w_dV[I];
    return tmp;
  }

  inline double NumericalDiffusionJacobian(const double& numDiff,
					   const double grad_v[NSPACE],
					   const double grad_w_dV[NSPACE])
  {
    double tmp=0.0;
    for (int I=0;I<NSPACE;I++)
      tmp += numDiff*grad_v[I]*grad_w_dV[I];
    return tmp;
  }

 

  inline double ExteriorElementBoundaryFlux(const double& flux,
					    const double& w_dS)
  {
    return flux*w_dS;
  }

  inline double InteriorElementBoundaryFlux(const double& flux,
					    const double& w_dS)
  {
    return flux*w_dS;
  }

  inline double ExteriorNumericalAdvectiveFluxJacobian(const double& dflux_left,
						       const double& v)
  {
    return dflux_left*v;
  }

  inline double InteriorNumericalAdvectiveFluxJacobian(const double& dflux_left,
						       const double& v)
  {
    return dflux_left*v;
  }

  inline double ExteriorElementBoundaryScalarDiffusionAdjoint(const int& isDOFBoundary,
							      const int& isFluxBoundary,
							      const double& sigma,
							      const double& u,
							      const double& bc_u,
							      const double normal[NSPACE],
							      const double& a,
							      const double grad_w_dS[NSPACE])
  {
    double tmp=0.0;
    for(int I=0;I<NSPACE;I++)
      {
	tmp += normal[I]*grad_w_dS[I];
      }
    tmp *= (1.0-isFluxBoundary)*isDOFBoundary*sigma*(u-bc_u)*a;
    return tmp;
  }
  
  inline double ExteriorElementBoundaryScalarDiffusionAdjointJacobian(const int& isDOFBoundary,
								      const int& isFluxBoundary,
								      const double& sigma,
								      const double& v,
								      const double normal[NSPACE],
								      const double& a,
								      const double grad_w_dS[NSPACE])
  {
    double tmp=0.0;
    for(int I=0;I<NSPACE;I++)
      {
	tmp += normal[I]*grad_w_dS[I];
      }
    tmp *= (1.0-isFluxBoundary)*isDOFBoundary*sigma*v*a;
    return tmp;
  }

  inline double ExteriorElementBoundaryDiffusionAdjoint(const int& isDOFBoundary,
							const int& isFluxBoundary,
							const double& sigma,
							const double& u,
							const double& bc_u,
							const double normal[NSPACE],
							proteus_csr_index_t* rowptr,
							proteus_csr_index_t* colind,
							double* a,
							const double grad_w_dS[NSPACE])
  {
    double tmp=0.0;
    for(int I=0;I<NSPACE;I++)
      for (proteus_csr_index_t m=rowptr[I];m<rowptr[I+1];m++)
	tmp += (1.0-isFluxBoundary)*isDOFBoundary*sigma*(u-bc_u)*a[m]*normal[colind[m]]*grad_w_dS[I];
    return tmp;
  }

  inline double ExteriorElementBoundaryDiffusionAdjointJacobian(const int& isDOFBoundary,
								const int& isFluxBoundary,
								const double& sigma,
								const double& v,
								const double normal[NSPACE],
								proteus_csr_index_t* rowptr,
								proteus_csr_index_t* colind,
								double* a,
								const double grad_w_dS[NSPACE])
  {
    double tmp=0.0;
    for(int I=0;I<NSPACE;I++)
      for (proteus_csr_index_t m=rowptr[I];m<rowptr[I+1];m++)
	tmp += (1.0-isFluxBoundary)*isDOFBoundary*sigma*v*a[m]*normal[colind[m]]*grad_w_dS[I];
    return tmp;
  }

  inline void calculateMapping_element(const proteus_index_t eN,
				       const int k,
				       double* mesh_dof,
				       proteus_index_t* mesh_l2g,
				       //proteus::pyarray<double>& mesh_trial_ref,
				       double* mesh_trial_ref,
				       double* mesh_grad_trial_ref,
				       double* jac,
				       double& jacDet,
				       double* jacInv,
				       double& x,
				       double& y,
				       double& z)
  {
    mapping.calculateMapping_element(eN,k,mesh_dof,mesh_l2g,mesh_trial_ref,mesh_grad_trial_ref,jac,jacDet,jacInv,x,y,z);
  }

  inline void calculateH_element(const proteus_index_t eN,
				 const int k,
				 double* h_dof,
				 proteus_index_t* mesh_l2g,
				 //proteus::pyarray<double>& mesh_trial_ref,
				 double* mesh_trial_ref,
				 double& h)
  {
    mapping.calculateH_element(eN,
			       k,
			       h_dof,
			       mesh_l2g,
			       mesh_trial_ref,
			       h);
  }

  inline void calculateMappingVelocity_element(const proteus_index_t eN,
					       const int k,
					       double* meshVelocity_dof,
					       proteus_index_t* mesh_l2g,
					       //proteus::pyarray<double>& mesh_trial_ref,
					       double* mesh_trial_ref,
					       double& xt,
					       double& yt,
					       double& zt)
  {
    mapping.calculateMappingVelocity_element(eN,k,meshVelocity_dof,mesh_l2g,mesh_trial_ref,xt,yt,zt);
  }

  inline 
  void calculateMapping_elementBoundary(const proteus_index_t eN,
					const int ebN_local,
					const int kb,
					const int ebN_local_kb,
					double* mesh_dof,
					proteus_index_t* mesh_l2g,
					double* mesh_trial_trace_ref,
					double* mesh_grad_trial_trace_ref,
					double* boundaryJac_ref,
					double* jac,
					double& jacDet,
					double* jacInv,
					double* boundaryJac,
					double* metricTensor,
					double& metricTensorDetSqrt,
					double* normal_ref,
					double* normal,
					double& x,
					double& y,
					double& z)
  {
    mapping.calculateMapping_elementBoundary(eN,
					     ebN_local,
					     kb,
					     ebN_local_kb,
					     mesh_dof,
					     mesh_l2g,
					     mesh_trial_trace_ref,
					     mesh_grad_trial_trace_ref,
					     boundaryJac_ref,
					     jac,
					     jacDet,
					     jacInv,
					     boundaryJac,
					     metricTensor,
					     metricTensorDetSqrt,
					     normal_ref,
					     normal,
					     x,
					     y,
					     z);
  }

  inline 
    void calculateMappingVelocity_elementBoundary(const proteus_index_t eN,
						  const int ebN_local,
						  const int kb,
						  const int ebN_local_kb,
						  double* mesh_velocity_dof,
						  proteus_index_t* mesh_l2g,
						  double* mesh_trial_trace_ref,
						  double& xt,
						  double& yt,
						  double& zt,
						  double* normal,
						  double* boundaryJac,						       
						  double* metricTensor,
						  double& metricTensorDetSqrt)
  {
    mapping.calculateMappingVelocity_elementBoundary(eN,
						     ebN_local,
						     kb,
						     ebN_local_kb,
						     mesh_velocity_dof,
						     mesh_l2g,
						     mesh_trial_trace_ref,
						     xt,
						     yt,
						     zt,
						     normal,
						     boundaryJac,
						     metricTensor,
						     metricTensorDetSqrt);
  }
  double Stress_u_weak(double* stress, double* grad_test_dV)
  {
    return stress[sXX]*grad_test_dV[X] + stress[sXY]*grad_test_dV[Y] + stress[sXZ]*grad_test_dV[Z];
  }
  double StressJacobian_u_u_weak(double* dstress, double* grad_trial, double* grad_test_dV)
  {
    return 
      (dstress[sXX*nSymTen+sXX]*grad_trial[X]+dstress[sXX*nSymTen+sXY]*grad_trial[Y]+dstress[sXX*nSymTen+sXZ]*grad_trial[Z])*grad_test_dV[X] +
      (dstress[sXY*nSymTen+sXX]*grad_trial[X]+dstress[sXY*nSymTen+sXY]*grad_trial[Y]+dstress[sXY*nSymTen+sXZ]*grad_trial[Z])*grad_test_dV[Y] +
      (dstress[sXZ*nSymTen+sXX]*grad_trial[X]+dstress[sXZ*nSymTen+sXY]*grad_trial[Y]+dstress[sXZ*nSymTen+sXZ]*grad_trial[Z])*grad_test_dV[Z];
  }
  double StressJacobian_u_v_weak(double* dstress, double* grad_trial, double* grad_test_dV)
  {
    return 
      (dstress[sXX*nSymTen+sYX]*grad_trial[X]+dstress[sXX*nSymTen+sYY]*grad_trial[Y]+dstress[sXX*nSymTen+sYZ]*grad_trial[Z])*grad_test_dV[X] +
      (dstress[sXY*nSymTen+sYX]*grad_trial[X]+dstress[sXY*nSymTen+sYY]*grad_trial[Y]+dstress[sXY*nSymTen+sYZ]*grad_trial[Z])*grad_test_dV[Y] +
      (dstress[sXZ*nSymTen+sYX]*grad_trial[X]+dstress[sXZ*nSymTen+sYY]*grad_trial[Y]+dstress[sXZ*nSymTen+sYZ]*grad_trial[Z])*grad_test_dV[Z];
  }
  double StressJacobian_u_w_weak(double* dstress, double* grad_trial, double* grad_test_dV)
  {
    return 
      (dstress[sXX*nSymTen+sZX]*grad_trial[X]+dstress[sXX*nSymTen+sZY]*grad_trial[Y]+dstress[sXX*nSymTen+sZZ]*grad_trial[Z])*grad_test_dV[X] +
      (dstress[sXY*nSymTen+sZX]*grad_trial[X]+dstress[sXY*nSymTen+sZY]*grad_trial[Y]+dstress[sXY*nSymTen+sZZ]*grad_trial[Z])*grad_test_dV[Y] +
      (dstress[sXZ*nSymTen+sZX]*grad_trial[X]+dstress[sXZ*nSymTen+sZY]*grad_trial[Y]+dstress[sXZ*nSymTen+sZZ]*grad_trial[Z])*grad_test_dV[Z];
  }
  double Stress_v_weak(double* stress, double* grad_test_dV)
  {
    return stress[sYX]*grad_test_dV[X] + stress[sYY]*grad_test_dV[Y] + stress[sYZ]*grad_test_dV[Z];
  }
  double StressJacobian_v_u_weak(double* dstress, double* grad_trial,double* grad_test_dV)
  {
    return 
      (dstress[sYX*nSymTen+sXX]*grad_trial[X]+dstress[sYX*nSymTen+sXY]*grad_trial[Y]+dstress[sYX*nSymTen+sXZ]*grad_trial[Z])*grad_test_dV[X] + 
      (dstress[sYY*nSymTen+sXX]*grad_trial[X]+dstress[sYY*nSymTen+sXY]*grad_trial[Y]+dstress[sYY*nSymTen+sXZ]*grad_trial[Z])*grad_test_dV[Y] + 
      (dstress[sYZ*nSymTen+sXX]*grad_trial[X]+dstress[sYZ*nSymTen+sXY]*grad_trial[Y]+dstress[sYZ*nSymTen+sXZ]*grad_trial[Z])*grad_test_dV[Z];
  }
  double StressJacobian_v_v_weak(double* dstress, double* grad_trial,double* grad_test_dV)
  {
    return 
      (dstress[sYX*nSymTen+sYX]*grad_trial[X]+dstress[sYX*nSymTen+sYY]*grad_trial[Y]+dstress[sYX*nSymTen+sYZ]*grad_trial[Z])*grad_test_dV[X] + 
      (dstress[sYY*nSymTen+sYX]*grad_trial[X]+dstress[sYY*nSymTen+sYY]*grad_trial[Y]+dstress[sYY*nSymTen+sYZ]*grad_trial[Z])*grad_test_dV[Y] + 
      (dstress[sYZ*nSymTen+sYX]*grad_trial[X]+dstress[sYZ*nSymTen+sYY]*grad_trial[Y]+dstress[sYZ*nSymTen+sYZ]*grad_trial[Z])*grad_test_dV[Z];
  }
  double StressJacobian_v_w_weak(double* dstress, double* grad_trial,double* grad_test_dV)
  {
    return 
      (dstress[sYX*nSymTen+sZX]*grad_trial[X]+dstress[sYX*nSymTen+sZY]*grad_trial[Y]+dstress[sYX*nSymTen+sZZ]*grad_trial[Z])*grad_test_dV[X] + 
      (dstress[sYY*nSymTen+sZX]*grad_trial[X]+dstress[sYY*nSymTen+sZY]*grad_trial[Y]+dstress[sYY*nSymTen+sZZ]*grad_trial[Z])*grad_test_dV[Y] + 
      (dstress[sYZ*nSymTen+sZX]*grad_trial[X]+dstress[sYZ*nSymTen+sZY]*grad_trial[Y]+dstress[sYZ*nSymTen+sZZ]*grad_trial[Z])*grad_test_dV[Z];
  }
  double Stress_w_weak(double* stress, double* grad_test_dV)
  {
    return stress[sZX]*grad_test_dV[X] + stress[sZY]*grad_test_dV[Y] + stress[sZZ]*grad_test_dV[Z];
  }
  double StressJacobian_w_u_weak(double* dstress, double* grad_trial, double* grad_test_dV)
  {
    return 
      (dstress[sZX*nSymTen+sXX]*grad_trial[X]+dstress[sZX*nSymTen+sXY]*grad_trial[Y]+dstress[sZX*nSymTen+sXZ]*grad_trial[Z])*grad_test_dV[X] + 
      (dstress[sZY*nSymTen+sXX]*grad_trial[X]+dstress[sZY*nSymTen+sXY]*grad_trial[Y]+dstress[sZY*nSymTen+sXZ]*grad_trial[Z])*grad_test_dV[Y] + 
      (dstress[sZZ*nSymTen+sXX]*grad_trial[X]+dstress[sZZ*nSymTen+sXY]*grad_trial[Y]+dstress[sZZ*nSymTen+sXZ]*grad_trial[Z])*grad_test_dV[Z];
  }
  double StressJacobian_w_v_weak(double* dstress, double* grad_trial, double* grad_test_dV)
  {
    return 
      (dstress[sZX*nSymTen+sYX]*grad_trial[X]+dstress[sZX*nSymTen+sYY]*grad_trial[Y]+dstress[sZX*nSymTen+sYZ]*grad_trial[Z])*grad_test_dV[X] + 
      (dstress[sZY*nSymTen+sYX]*grad_trial[X]+dstress[sZY*nSymTen+sYY]*grad_trial[Y]+dstress[sZY*nSymTen+sYZ]*grad_trial[Z])*grad_test_dV[Y] + 
      (dstress[sZZ*nSymTen+sYX]*grad_trial[X]+dstress[sZZ*nSymTen+sYY]*grad_trial[Y]+dstress[sZZ*nSymTen+sYZ]*grad_trial[Z])*grad_test_dV[Z];
  }
  double StressJacobian_w_w_weak(double* dstress, double* grad_trial, double* grad_test_dV)
  {
    return 
      (dstress[sZX*nSymTen+sZX]*grad_trial[X]+dstress[sZX*nSymTen+sZY]*grad_trial[Y]+dstress[sZX*nSymTen+sZZ]*grad_trial[Z])*grad_test_dV[X] + 
      (dstress[sZY*nSymTen+sZX]*grad_trial[X]+dstress[sZY*nSymTen+sZY]*grad_trial[Y]+dstress[sZY*nSymTen+sZZ]*grad_trial[Z])*grad_test_dV[Y] + 
      (dstress[sZZ*nSymTen+sZX]*grad_trial[X]+dstress[sZZ*nSymTen+sZY]*grad_trial[Y]+dstress[sZZ*nSymTen+sZZ]*grad_trial[Z])*grad_test_dV[Z];
  }
  double ExteriorElementBoundaryStressFlux(const double& stressFlux,const double& disp_test_dS)
  {
    return stressFlux*disp_test_dS;
  }
  double ExteriorElementBoundaryStressFluxJacobian(const double& dstressFlux,const double& disp_test_dS)
  {
    return dstressFlux*disp_test_dS;
  }
};

//specialization for 2D
template<int NDOF_MESH_TRIAL_ELEMENT, int NDOF_TRIAL_ELEMENT, int NDOF_TEST_ELEMENT>
class CompKernel<2,NDOF_MESH_TRIAL_ELEMENT,NDOF_TRIAL_ELEMENT,NDOF_TEST_ELEMENT>
{
public:
  CompKernelSpaceMapping<2,NDOF_MESH_TRIAL_ELEMENT> mapping;
  const int X,
    Y,
    XX,XY,
    YX,YY,
    sXX,sXY,
    sYX,sYY,
    nSymTen,
    XHX,
    YHX,
    HXHX;
  CompKernel():
    X(mapping.X),
    Y(mapping.Y),
    XX(mapping.XX),XY(mapping.XY),
    YX(mapping.YX),YY(mapping.YY),
    sXX(mapping.sXX),sXY(mapping.sXY),
    sYX(mapping.sYX),sYY(mapping.sYY),
    nSymTen(mapping.nSymTen),
    XHX(mapping.XHX),
    YHX(mapping.YHX),
    HXHX(mapping.HXHX)
  {}
  inline void calculateG(double* jacInv,double* G,double& G_dd_G, double& tr_G)
    {
      //get the metric tensor
      //cek todo use symmetry
      for (int I=0;I<2;I++)
	for (int J=0;J<2;J++)
	  {
	    G[I*2+J] = 0.0;
	    for (int K=0;K<2;K++)
	      G[I*2+J] += jacInv[K*2+I]*jacInv[K*2+J];
	  }
      G_dd_G = 0.0;
      tr_G = 0.0;
      for (int I=0;I<2;I++)
	{
	  tr_G += G[I*2+I];
	  for (int J=0;J<2;J++)
	    {
	      G_dd_G += G[I*2+J]*G[I*2+J];
	    }
	}
    }
  inline void calculateGScale(double* G,double* v,double& h)
  {
    h = 0.0;
    for (int I=0;I<2;I++)
      for (int J=0;J<2;J++)
	h += v[I]*G[I*2+J]*v[J];
    h = 1.0/sqrt(h+1.0e-12);
  }
  inline void valFromDOF(const double* dof,const proteus_index_t* l2g_element,const double* trial_ref,double& val)
  {
    val=0.0;
    for (int j=0;j<NDOF_TRIAL_ELEMENT;j++)
      val+=dof[l2g_element[j]]*trial_ref[j];
  }

  inline void gradFromDOF(const double* dof,const proteus_index_t* l2g_element,const double* grad_trial,double* grad)
  {
    for(int I=0;I<2;I++)
      grad[I] = 0.0;
    for (int j=0;j<NDOF_TRIAL_ELEMENT;j++)
      for(int I=0;I<2;I++)
	grad[I] += dof[l2g_element[j]]*grad_trial[j*2+I];
  }

  inline void hessFromDOF(const double* dof,const proteus_index_t* l2g_element,const double* hess_trial,double* hess)
  {
    for(int I=0;I<2;I++)
      for(int J=0;J<2;J++)
	hess[I*2+J] = 0.0;
    for (int j=0;j<NDOF_TRIAL_ELEMENT;j++)
      for(int I=0;I<2;I++)
	for(int J=0;J<2;J++)
	  hess[I*2+J] += dof[l2g_element[j]]*hess_trial[j*4+I*2+J];
  }

  inline void valFromElementDOF(const double* dof,const double* trial_ref,double& val)
  {
    val=0.0;
    for (int j=0;j<NDOF_TRIAL_ELEMENT;j++)
      val+=dof[j]*trial_ref[j];
  }

  inline void gradFromElementDOF(const double* dof,const double* grad_trial,double* grad)
  {
    for(int I=0;I<2;I++)
      grad[I] = 0.0;
    for (int j=0;j<NDOF_TRIAL_ELEMENT;j++)
      for(int I=0;I<2;I++)
	grad[I] += dof[j]*grad_trial[j*2+I];
  }

  inline void gradTrialFromRef(const double* grad_trial_ref, const double* jacInv, double* grad_trial)
  {
    for (int j=0;j<NDOF_TRIAL_ELEMENT;j++)
      for(int I=0;I<2;I++)
	grad_trial[j*2+I] = 0.0;
    for (int j=0;j<NDOF_TRIAL_ELEMENT;j++)
      for(int I=0;I<2;I++)
	for(int J=0;J<2;J++)
	  grad_trial[j*2+I] += jacInv[J*2+I]*grad_trial_ref[j*2+J];
  }

  /*
   *  DOFaverage
   *  ----------
   *
   *  Calculate the average DOF value for at a given mesh element.
   *
   *  @param dof array of finite element DOF values
   *  @param l2g_element local 2 global mapping for the current mesh element
   *  @param val return value with the average DOF values
   */

  inline void DOFaverage (const double* dof, const proteus_index_t* l2g_element, double& val)
  {
    val = 0.0;

    for (int j=0; j<NDOF_MESH_TRIAL_ELEMENT; j++)
      val+=dof[l2g_element[j]];

    val /= NDOF_MESH_TRIAL_ELEMENT;
  }

  
  inline void hessTrialFromRef(const double* hess_trial_ref, const double* jacInv, double* hess_trial)
  {
    for (int j=0;j<NDOF_TRIAL_ELEMENT;j++)
      for(int I=0;I<2;I++)
	for(int J=0;J<2;J++)
	  hess_trial[j*4+I*2+J] = 0.0;
    for (int j=0;j<NDOF_TRIAL_ELEMENT;j++)
      for(int I=0;I<2;I++)
	for(int J=0;J<2;J++)
	  for(int K=0;K<2;K++)
	    for(int L=0;L<2;L++)
	      hess_trial[j*4+I*2+J] += hess_trial_ref[j*4+K*2+L]*jacInv[L*2+J]*jacInv[K*2+I];
  }
  
  inline void gradTestFromRef(const double* grad_test_ref, const double* jacInv, double* grad_test)
  {
    for (int i=0;i<NDOF_TEST_ELEMENT;i++)
      for(int I=0;I<2;I++)
	grad_test[i*2+I] = 0.0;
    for (int i=0;i<NDOF_TEST_ELEMENT;i++)
      for(int I=0;I<2;I++)
	for(int J=0;J<2;J++)
	  grad_test[i*2+I] += jacInv[J*2+I]*grad_test_ref[i*2+J];
  }
  
  inline void backwardEuler(const double& dt, const double& m_old, const double& m, const double& dm, double& mt, double& dmt)
  {  
    mt =(m-m_old)/dt;
    dmt = dm/dt;
  }

  inline void bdf(const double& alpha, const double& beta, const double& m, const double& dm, double& mt, double& dmt)
  {  
    mt =alpha*m + beta;
    dmt = alpha*dm;
  }

  inline void bdfC2(const double& alpha, const double& beta, const double& m, const double& dm, const double& dm2, double& mt, double& dmt, double& dm2t)
  {  
    mt =alpha*m + beta;
    dmt = alpha*dm;
    dm2t = alpha*dm2;
  }

  inline double Mass_weak(const double& mt, const double& w_dV)
  {
    return mt*w_dV;
  }

  inline double MassJacobian_weak(const double& dmt,
				  const double& v,
				  const double& w_dV)
  {
    return dmt*v*w_dV;
  }

  inline double Mass_strong(const double& mt)
  {
    return mt; 
  }

  inline double MassJacobian_strong(const double& dmt,
				    const double& v)
  {
    return dmt*v;
  }

  inline double Mass_adjoint(const double& dmt,
			     const double& w_dV)
  {
    return dmt*w_dV;
  }

  /*
   *  pressureProjection_weak
   *  -----------------------
   *
   *  Inner product calculation of the pressure projection
   *  stablization method of Bochev, Dohrmann and
   *  Gunzburger (2006).
   *
   *  @param viscosity viscosity at point
   *  @param p the pressure value (either trial function
   *           or actual value)
   *  @param p_avg the pressure projection value (either
   *           1./3. for test functions of the average
   *	       value of the pressure on the element)
   *  @param dV this is the integral weight
   *
   */

  inline double pressureProjection_weak(const double& viscosity,
					const double& p,
					const double& p_avg,
					const double& q,
					const double& dV)
  {
    if (viscosity==0.){ return 0.;}
    return (1./viscosity)*(p-p_avg)*(q-1./3.)*dV;
  }

  inline double Advection_weak(const double  f[2],
			       const double grad_w_dV[2])
  {
    double tmp=0.0;
    for(int I=0;I<2;I++)
      tmp -= f[I]*grad_w_dV[I];
    return tmp;
  }

  inline double AdvectionJacobian_weak(const double df[2],
				       const double& v,
				       const double grad_w_dV[2])
  {
    double tmp=0.0;
    for(int I=0;I<2;I++)
      tmp -= df[I]*v*grad_w_dV[I];
    return tmp;
  }

  inline double Advection_strong(const double df[2],
				 const double grad_u[2])
  {
    double tmp=0.0;
    for(int I=0;I<2;I++)
      tmp += df[I]*grad_u[I];
    return tmp;
  }

  inline double AdvectionJacobian_strong(const double df[2],
					 const double grad_v[2])
  {
    double tmp=0.0;
    for(int I=0;I<2;I++)
      tmp += df[I]*grad_v[I];
    return tmp;
  }

  inline double Advection_adjoint(const double df[2],
				  const double grad_w_dV[2])
  {
    double tmp=0.0;
    for(int I=0;I<2;I++)
      tmp -= df[I]*grad_w_dV[I];
    return tmp;
  }

  inline double Hamiltonian_weak(const double& H,
				 const double& w_dV)
  {
    return H*w_dV;
  }

  inline double HamiltonianJacobian_weak(const double dH[2],
					 const double grad_v[2],
					 const double& w_dV)
  {
    double tmp=0.0;
    for(int I=0;I<2;I++)
      tmp += dH[I]*grad_v[I]*w_dV;
    return tmp;
  }

  inline double Hamiltonian_strong(const double dH[2],
				   const double grad_u[2])
  {
    double tmp=0.0;
    for(int I=0;I<2;I++)
      tmp += dH[I]*grad_u[I];
    return tmp;
  }

  inline double HamiltonianJacobian_strong(const double dH[2],
					   const double grad_v[2])
  {
    double tmp=0.0;
    for(int I=0;I<2;I++)
      tmp += dH[I]*grad_v[I];
    return tmp;
  }

  inline double Hamiltonian_adjoint(const double dH[2],
				    const double grad_w_dV[2])
  {
    double tmp=0.0;
    for(int I=0;I<2;I++)
      tmp -= dH[I]*grad_w_dV[I];
    return tmp;
  }

  inline double Diffusion_weak(proteus_csr_index_t* rowptr,
			       proteus_csr_index_t* colind,
			       double* a,
			       const double grad_phi[2],
			       const double grad_w_dV[2])
  {
    double tmp=0.0;
    for(int I=0;I<2;I++)
      for (proteus_csr_index_t m=rowptr[I];m<rowptr[I+1];m++)
	tmp += a[m]*grad_phi[colind[m]]*grad_w_dV[I];
    return tmp;
  }

  inline double DiffusionJacobian_weak(proteus_csr_index_t* rowptr,
				       proteus_csr_index_t* colind,
				       double* a,
				       double* da,
				       const double grad_phi[2],
				       const double grad_w_dV[2],
				       const double& dphi,
				       const double& v,
				       const double grad_v[2])
  {
    double daProduct=0.0,dphiProduct=0.0;
    for (int I=0;I<2;I++)
      for (proteus_csr_index_t m=rowptr[I];m<rowptr[I+1];m++)
	{
	  daProduct += da[m]*grad_phi[colind[m]]*grad_w_dV[I];
	  dphiProduct += a[m]*grad_v[colind[m]]*grad_w_dV[I];
	}
    return daProduct*v+dphiProduct*dphi;
  }

  inline double SimpleDiffusionJacobian_weak(proteus_csr_index_t* rowptr,
					     proteus_csr_index_t* colind,
					     double* a,
					     const double grad_v[2],
					     const double grad_w_dV[2])
  {
    double dphiProduct=0.0;
    for (int I=0;I<2;I++)
      for (proteus_csr_index_t m=rowptr[I];m<rowptr[I+1];m++)
	{
	  dphiProduct += a[m]*grad_v[colind[m]]*grad_w_dV[I];
	}
    return dphiProduct;
  }

  inline double Reaction_weak(const double& r,
			      const double& w_dV)
  {
    return r*w_dV;
  }
  
  inline double ReactionJacobian_weak(const double& dr,
				      const double& v,
				      const double& w_dV)
  {
    return dr*v*w_dV;
  }

  inline double Reaction_strong(const double& r)
  {
    return r; 
  }

  inline double ReactionJacobian_strong(const double& dr,
					const double& v)
  {
    return dr*v;
  }

  inline double Reaction_adjoint(const double& dr,
				 const double& w_dV)
  {
    return dr*w_dV;
  }
  
  inline void calculateNumericalDiffusion(const double& shockCapturingDiffusion,
					  const double& elementDiameter,
					  const double& strong_residual,
					  const double grad_u[2],
					  double& numDiff)
  {
    double h,
      num,
      den,
      n_grad_u;
    h = elementDiameter;
    n_grad_u = 0.0;
    for (int I=0;I<2;I++)
      n_grad_u += grad_u[I]*grad_u[I];
    num = shockCapturingDiffusion*0.5*h*fabs(strong_residual);
    den = sqrt(n_grad_u+1.0e-12);
    //cek hack shockCapturingDiffusion*fabs(strong_residual)*grad_phi_G_grad_phi
    numDiff = num/den;
  }


  inline void calculateNumericalDiffusion(const double& shockCapturingDiffusion,
					  const double  G[2*2],
					  const double& strong_residual,
					  const double grad_u[2],
					  double& numDiff)
  {
    double  den = 0.0;
    for (int I=0;I<2;I++)
      for (int J=0;J<2;J++)
        den += grad_u[I]*G[I*2+J]*grad_u[J];

    numDiff = shockCapturingDiffusion*fabs(strong_residual)/(sqrt(den+1.0e-12));
  }

  inline void calculateNumericalDiffusion(const double& shockCapturingDiffusion,
                                          const double& uref, const double& beta,
					  const double  G[2*2],
					  const double& G_dd_G,
					  const double& strong_residual,
					  const double grad_u[2],
					  double& numDiff)
  {
    double  den = 0.0;    
    for (int I=0;I<2;I++)
      for (int J=0;J<2;J++)
        den += grad_u[I]*G[I*2+J]*grad_u[J];

    double h2_uref_1 = 1.0/(sqrt(den+1.0e-12));
    double h2_uref_2 = 1.0/(uref*sqrt(G_dd_G+1.0e-12));
    numDiff = shockCapturingDiffusion*fabs(strong_residual)*pow(h2_uref_1, 2.0-beta)*pow(h2_uref_2,beta-1.0);
  }



  inline void calculateNumericalDiffusion(const double& shockCapturingDiffusion,
					  const double  G[2*2],
					  const double& strong_residual,
					  const double  vel[2],
					  const double  grad_u[2],
					  double& numDiff)
  {
    double  den1 = 0.0,den2=0.0, nom=0.0;
    for (int I=0;I<2;I++)
      {
      nom += vel[I]*vel[I];
      den2+= grad_u[I]*grad_u[I]; 
      for (int J=0;J<2;J++)
        den1 += vel[I]*G[I*2+J]*vel[J];
      }
    numDiff = shockCapturingDiffusion*fabs(strong_residual)*(sqrt(nom/(den1*den2 + 1.0e-12)));
  }


  inline void calculateNumericalDiffusion(const double& shockCapturingDiffusion,
					  const double& elementDiameter,
					  const double& strong_residual,
					  const double grad_u[2],
					  double& gradNorm,
					  double& gradNorm_last,
					  double& numDiff)
  {
    double h,
      num,
      n_grad_u;
    h = elementDiameter;
    n_grad_u = 0.0;
    for (int I=0;I<2;I++)
      n_grad_u += grad_u[I]*grad_u[I];
    num = shockCapturingDiffusion*0.5*h*fabs(strong_residual);
    gradNorm = sqrt(n_grad_u+1.0e-12);
    //cek hack shockCapturingDiffusion*fabs(strong_residual)*grad_phi_G_grad_phi    
    numDiff = num/gradNorm_last;
  }

  inline double SubgridError(const double& error,
			     const double& Lstar_w_dV)
  {
    return error*Lstar_w_dV;
  }

  inline double SubgridErrorJacobian(const double& derror,
				     const double& Lstar_w_dV)
  {
    return derror*Lstar_w_dV;
  }

  inline double NumericalDiffusion(const double& numDiff,
				   const double grad_u[2],
				   const double grad_w_dV[2])
  {
    double tmp=0.0;
    for (int I=0;I<2;I++)
      tmp +=  numDiff*grad_u[I]*grad_w_dV[I];
    return tmp;
  }

  inline double NumericalDiffusionJacobian(const double& numDiff,
					   const double grad_v[2],
					   const double grad_w_dV[2])
  {
    double tmp=0.0;
    for (int I=0;I<2;I++)
      tmp += numDiff*grad_v[I]*grad_w_dV[I];
    return tmp;
  }

 

  inline double ExteriorElementBoundaryFlux(const double& flux,
					    const double& w_dS)
  {
    return flux*w_dS;
  }

  inline double InteriorElementBoundaryFlux(const double& flux,
					    const double& w_dS)
  {
    return flux*w_dS;
  }

  inline double ExteriorNumericalAdvectiveFluxJacobian(const double& dflux_left,
						       const double& v)
  {
    return dflux_left*v;
  }

  inline double InteriorNumericalAdvectiveFluxJacobian(const double& dflux_left,
						       const double& v)
  {
    return dflux_left*v;
  }

  inline double ExteriorElementBoundaryScalarDiffusionAdjoint(const int& isDOFBoundary,
							      const int& isFluxBoundary,
							      const double& sigma,
							      const double& u,
							      const double& bc_u,
							      const double normal[2],
							      const double& a,
							      const double grad_w_dS[2])
  {
    double tmp=0.0;
    for(int I=0;I<2;I++)
      {
	tmp += normal[I]*grad_w_dS[I];
      }
    tmp *= (1.0-isFluxBoundary)*isDOFBoundary*sigma*(u-bc_u)*a;
    return tmp;
  }

  inline double ExteriorElementBoundaryScalarDiffusionAdjointJacobian(const int& isDOFBoundary,
								      const int& isFluxBoundary,
								      const double& sigma,
								      const double& v,
								      const double normal[2],
								      const double& a,
								      const double grad_w_dS[2])
  {
    double tmp=0.0;
    for(int I=0;I<2;I++)
      {
	tmp += normal[I]*grad_w_dS[I];
      }
    tmp *= (1.0-isFluxBoundary)*isDOFBoundary*sigma*v*a;
    return tmp;
  }
						
  inline double ExteriorElementBoundaryDiffusionAdjoint(const int& isDOFBoundary,
							const int& isFluxBoundary,
							const double& sigma,
							const double& u,
							const double& bc_u,
							const double normal[2],
							proteus_csr_index_t* rowptr,
							proteus_csr_index_t* colind,
							double* a,
							const double grad_w_dS[2])
  {
    double tmp=0.0;
    for(int I=0;I<2;I++)
      for (proteus_csr_index_t m=rowptr[I];m<rowptr[I+1];m++)
	tmp += (1.0-isFluxBoundary)*isDOFBoundary*sigma*(u-bc_u)*a[m]*normal[colind[m]]*grad_w_dS[I];
    return tmp;
  }

  inline double ExteriorElementBoundaryDiffusionAdjointJacobian(const int& isDOFBoundary,
								const int& isFluxBoundary,
								const double& sigma,
								const double& v,
								const double normal[2],
								proteus_csr_index_t* rowptr,
								proteus_csr_index_t* colind,
								double* a,
								const double grad_w_dS[2])
  {
    double tmp=0.0;
    for(int I=0;I<2;I++)
      for (proteus_csr_index_t m=rowptr[I];m<rowptr[I+1];m++)
	tmp += (1.0-isFluxBoundary)*isDOFBoundary*sigma*v*a[m]*normal[colind[m]]*grad_w_dS[I];
    return tmp;
  }

  inline void calculateMapping_element(const proteus_index_t eN,
				       const int k,
				       double* mesh_dof,
				       proteus_index_t* mesh_l2g,
				       //proteus::pyarray<double>& mesh_trial_ref,
				       double* mesh_trial_ref,
				       double* mesh_grad_trial_ref,
				       double* jac,
				       double& jacDet,
				       double* jacInv,
				       double& x,
				       double& y)
  {
    mapping.calculateMapping_element(eN,k,mesh_dof,mesh_l2g,mesh_trial_ref,mesh_grad_trial_ref,jac,jacDet,jacInv,x,y);
  }

  inline void calculateH_element(const proteus_index_t eN,
				 const int k,
				 double* h_dof,
				 proteus_index_t* mesh_l2g,
				 //proteus::pyarray<double>& mesh_trial_ref,
				 double* mesh_trial_ref,
				 double& h)
  {
    mapping.calculateH_element(eN,
			       k,
			       h_dof,
			       mesh_l2g,
			       mesh_trial_ref,
			       h);
  }

  inline void calculateMapping_element(const proteus_index_t eN,
				       const int k,
				       double* mesh_dof,
				       proteus_index_t* mesh_l2g,
				       //proteus::pyarray<double>& mesh_trial_ref,
				       double* mesh_trial_ref,
				       double* mesh_grad_trial_ref,
				       double* jac,
				       double& jacDet,
				       double* jacInv,
				       double& x,
				       double& y,
				       double& z)
  {
    mapping.calculateMapping_element(eN,k,mesh_dof,mesh_l2g,mesh_trial_ref,mesh_grad_trial_ref,jac,jacDet,jacInv,x,y,z);
  }

  inline void calculateMappingVelocity_element(const proteus_index_t eN,
					       const int k,
					       double* meshVelocity_dof,
					       proteus_index_t* mesh_l2g,
					       //proteus::pyarray<double>& mesh_trial_ref,
					       double* mesh_trial_ref,
					       double& xt,
					       double& yt)
  {
    mapping.calculateMappingVelocity_element(eN,k,meshVelocity_dof,mesh_l2g,mesh_trial_ref,xt,yt);
  }

  inline void calculateMappingVelocity_element(const proteus_index_t eN,
					       const int k,
					       double* meshVelocity_dof,
					       proteus_index_t* mesh_l2g,
					       //proteus::pyarray<double>& mesh_trial_ref,
					       double* mesh_trial_ref,
					       double& xt,
					       double& yt,
					       double& zt)
  {
    mapping.calculateMappingVelocity_element(eN,k,meshVelocity_dof,mesh_l2g,mesh_trial_ref,xt,yt,zt);
  }

  inline
  void calculateMapping_elementBoundary(const proteus_index_t eN,
					const int ebN_local,
					const int kb,
					const int ebN_local_kb,
					double* mesh_dof,
					proteus_index_t* mesh_l2g,
					double* mesh_trial_trace_ref,
					double* mesh_grad_trial_trace_ref,
					double* boundaryJac_ref,
					double* jac,
					double& jacDet,
					double* jacInv,
					double* boundaryJac,
					double* metricTensor,
					double& metricTensorDetSqrt,
					double* normal_ref,
					double* normal,
					double& x,
					double& y)
  {
    mapping.calculateMapping_elementBoundary(eN,
					     ebN_local,
					     kb,
					     ebN_local_kb,
					     mesh_dof,
					     mesh_l2g,
					     mesh_trial_trace_ref,
					     mesh_grad_trial_trace_ref,
					     boundaryJac_ref,
					     jac,
					     jacDet,
					     jacInv,
					     boundaryJac,
					     metricTensor,
					     metricTensorDetSqrt,
					     normal_ref,
					     normal,
					     x,
					     y);
  }

  inline
  void calculateMapping_elementBoundary(const proteus_index_t eN,
					const int ebN_local,
					const int kb,
					const int ebN_local_kb,
					double* mesh_dof,
					proteus_index_t* mesh_l2g,
					double* mesh_trial_trace_ref,
					double* mesh_grad_trial_trace_ref,
					double* boundaryJac_ref,
					double* jac,
					double& jacDet,
					double* jacInv,
					double* boundaryJac,
					double* metricTensor,
					double& metricTensorDetSqrt,
					double* normal_ref,
					double* normal,
					double& x,
					double& y,
					double& z)
  {
    mapping.calculateMapping_elementBoundary(eN,
					     ebN_local,
					     kb,
					     ebN_local_kb,
					     mesh_dof,
					     mesh_l2g,
					     mesh_trial_trace_ref,
					     mesh_grad_trial_trace_ref,
					     boundaryJac_ref,
					     jac,
					     jacDet,
					     jacInv,
					     boundaryJac,
					     metricTensor,
					     metricTensorDetSqrt,
					     normal_ref,
					     normal,
					     x,
					     y,
					     z);
  }

  inline
    void calculateMappingVelocity_elementBoundary(const proteus_index_t eN,
						  const int ebN_local,
						  const int kb,
						  const int ebN_local_kb,
						  double* mesh_velocity_dof,
						  proteus_index_t* mesh_l2g,
						  double* mesh_trial_trace_ref,
						  double& xt,
						  double& yt,
						  double* normal,
						  double* boundaryJac,
						  double* metricTensor,
						  double& metricTensorDetSqrt)
  {
    mapping.calculateMappingVelocity_elementBoundary(eN,
						     ebN_local,
						     kb,
						     ebN_local_kb,
						     mesh_velocity_dof,
						     mesh_l2g,
						     mesh_trial_trace_ref,
						     xt,
						     yt,
						     normal,
						     boundaryJac,
						     metricTensor,
						     metricTensorDetSqrt);
  }
  inline
    void calculateMappingVelocity_elementBoundary(const proteus_index_t eN,
						  const int ebN_local,
						  const int kb,
						  const int ebN_local_kb,
						  double* mesh_velocity_dof,
						  proteus_index_t* mesh_l2g,
						  double* mesh_trial_trace_ref,
						  double& xt,
						  double& yt,
						  double& zt,
						  double* normal,
						  double* boundaryJac,
						  double* metricTensor,
						  double& metricTensorDetSqrt)
  {
    mapping.calculateMappingVelocity_elementBoundary(eN,
						     ebN_local,
						     kb,
						     ebN_local_kb,
						     mesh_velocity_dof,
						     mesh_l2g,
						     mesh_trial_trace_ref,
						     xt,
						     yt,
						     zt,
						     normal,
						     boundaryJac,
						     metricTensor,
						     metricTensorDetSqrt);
  }
  double Stress_u_weak(double* stress, double* grad_test_dV)
  {
    return stress[sXX]*grad_test_dV[X] + stress[sXY]*grad_test_dV[Y];
  }
  double StressJacobian_u_u_weak(double* dstress, double* grad_trial, double* grad_test_dV)
  {
    return
      (dstress[sXX*nSymTen+sXX]*grad_trial[X]+dstress[sXX*nSymTen+sXY]*grad_trial[Y])*grad_test_dV[X] +
      (dstress[sXY*nSymTen+sXX]*grad_trial[X]+dstress[sXY*nSymTen+sXY]*grad_trial[Y])*grad_test_dV[Y];
  }
  double StressJacobian_u_v_weak(double* dstress, double* grad_trial, double* grad_test_dV)
  {
    return
      (dstress[sXX*nSymTen+sYX]*grad_trial[X]+dstress[sXX*nSymTen+sYY]*grad_trial[Y])*grad_test_dV[X] +
      (dstress[sXY*nSymTen+sYX]*grad_trial[X]+dstress[sXY*nSymTen+sYY]*grad_trial[Y])*grad_test_dV[Y];
  }
  double Stress_v_weak(double* stress, double* grad_test_dV)
  {
    return stress[sYX]*grad_test_dV[X] + stress[sYY]*grad_test_dV[Y];
  }
  double StressJacobian_v_u_weak(double* dstress, double* grad_trial,double* grad_test_dV)
  {
    return
      (dstress[sYX*nSymTen+sXX]*grad_trial[X]+dstress[sYX*nSymTen+sXY]*grad_trial[Y])*grad_test_dV[X] +
      (dstress[sYY*nSymTen+sXX]*grad_trial[X]+dstress[sYY*nSymTen+sXY]*grad_trial[Y])*grad_test_dV[Y];
  }
  double StressJacobian_v_v_weak(double* dstress, double* grad_trial,double* grad_test_dV)
  {
    return
      (dstress[sYX*nSymTen+sYX]*grad_trial[X]+dstress[sYX*nSymTen+sYY]*grad_trial[Y])*grad_test_dV[X] +
      (dstress[sYY*nSymTen+sYX]*grad_trial[X]+dstress[sYY*nSymTen+sYY]*grad_trial[Y])*grad_test_dV[Y];
  }
  double ExteriorElementBoundaryStressFlux(const double& stressFlux,const double& disp_test_dS)
  {
    return stressFlux*disp_test_dS;
  }
  double ExteriorElementBoundaryStressFluxJacobian(const double& dstressFlux,const double& disp_test_dS)
  {
    return dstressFlux*disp_test_dS;
  }
};

//specialization for 1D
template<int NDOF_MESH_TRIAL_ELEMENT, int NDOF_TRIAL_ELEMENT, int NDOF_TEST_ELEMENT>
class CompKernel<1,NDOF_MESH_TRIAL_ELEMENT,NDOF_TRIAL_ELEMENT,NDOF_TEST_ELEMENT>
{
public:
  CompKernelSpaceMapping<1,NDOF_MESH_TRIAL_ELEMENT> mapping;
  const int X,
    XX,
    sXX,
    nSymTen,
    XHX,
    HXHX;
  CompKernel():
    X(mapping.X),
    XX(mapping.XX),
    sXX(mapping.sXX),
    nSymTen(mapping.nSymTen),
    XHX(mapping.XHX),
    HXHX(mapping.HXHX)
  {}
  inline void calculateG(double* jacInv,double* G,double& G_dd_G, double& tr_G)
    {
      for (int I=0;I<1;I++)
	for (int J=0;J<1;J++)
	  {
	    G[I+J] = 0.0;
	    for (int K=0;K<1;K++)
	      G[I+J] += jacInv[K+I]*jacInv[K+J];
	  }
      G_dd_G = 0.0;
      tr_G = 0.0;
      for (int I=0;I<1;I++)
	{
	  tr_G += G[I+I];
	  for (int J=0;J<1;J++)
	    {
	      G_dd_G += G[I+J]*G[I+J];
	    }
	}
    }
  inline void calculateGScale(double* G,double* v,double& h)
  {
    h = 0.0;
    for (int I=0;I<1;I++)
      for (int J=0;J<1;J++)
	h += v[I]*G[I+J]*v[J];
    h = 1.0/sqrt(h+1.0e-12);
  }
  inline void valFromDOF(const double* dof,const proteus_index_t* l2g_element,const double* trial_ref,double& val)
  {
    val=0.0;
    for (int j=0;j<NDOF_TRIAL_ELEMENT;j++)
      val+=dof[l2g_element[j]]*trial_ref[j];
  }

  inline void gradFromDOF(const double* dof,const proteus_index_t* l2g_element,const double* grad_trial,double* grad)
  {
    for(int I=0;I<1;I++)
      grad[I] = 0.0;
    for (int j=0;j<NDOF_TRIAL_ELEMENT;j++)
      for(int I=0;I<1;I++)
	grad[I] += dof[l2g_element[j]]*grad_trial[j+I];
  }

  inline void hessFromDOF(const double* dof,const proteus_index_t* l2g_element,const double* hess_trial,double* hess)
  {
    for(int I=0;I<1;I++)
      for(int J=0;J<1;J++)
	hess[I+J] = 0.0;
    for (int j=0;j<NDOF_TRIAL_ELEMENT;j++)
      for(int I=0;I<1;I++)
	for(int J=0;J<1;J++)
	  hess[I*2+J] += dof[l2g_element[j]]*hess_trial[j+I+J];
  }

  inline void valFromElementDOF(const double* dof,const double* trial_ref,double& val)
  {
    val=0.0;
    for (int j=0;j<NDOF_TRIAL_ELEMENT;j++)
      val+=dof[j]*trial_ref[j];
  }

  inline void gradFromElementDOF(const double* dof,const double* grad_trial,double* grad)
  {
    for(int I=0;I<1;I++)
      grad[I] = 0.0;
    for (int j=0;j<NDOF_TRIAL_ELEMENT;j++)
      for(int I=0;I<1;I++)
	grad[I] += dof[j]*grad_trial[j+I];
  }

  inline void gradTrialFromRef(const double* grad_trial_ref, const double* jacInv, double* grad_trial)
  {
    for (int j=0;j<NDOF_TRIAL_ELEMENT;j++)
      for(int I=0;I<1;I++)
	grad_trial[j+I] = 0.0;
    for (int j=0;j<NDOF_TRIAL_ELEMENT;j++)
      for(int I=0;I<1;I++)
	for(int J=0;J<1;J++)
	  grad_trial[j+I] += jacInv[J+I]*grad_trial_ref[j+J];
  }

  /*
   *  DOFaverage
   *  ----------
   *
   *  Calculate the average DOF value for at a given mesh element.
   *
   *  @param dof array of finite element DOF values
   *  @param l2g_element local 2 global mapping for the current mesh element
   *  @param val return value with the average DOF values
   */

  inline void DOFaverage (const double* dof, const proteus_index_t* l2g_element, double& val)
  {
    val = 0.0;

    for (int j=0; j<NDOF_MESH_TRIAL_ELEMENT; j++)
      val+=dof[l2g_element[j]];

    val /= NDOF_MESH_TRIAL_ELEMENT;
  }

  
  inline void hessTrialFromRef(const double* hess_trial_ref, const double* jacInv, double* hess_trial)
  {
    for (int j=0;j<NDOF_TRIAL_ELEMENT;j++)
      for(int I=0;I<1;I++)
	for(int J=0;J<1;J++)
	  hess_trial[j+I+J] = 0.0;
    for (int j=0;j<NDOF_TRIAL_ELEMENT;j++)
      for(int I=0;I<1;I++)
	for(int J=0;J<1;J++)
	  for(int K=0;K<1;K++)
	    for(int L=0;L<1;L++)
	      hess_trial[j+I+J] += hess_trial_ref[j+K+L]*jacInv[L+J]*jacInv[K+I];
  }
  
  inline void gradTestFromRef(const double* grad_test_ref, const double* jacInv, double* grad_test)
  {
    for (int i=0;i<NDOF_TEST_ELEMENT;i++)
      for(int I=0;I<1;I++)
	grad_test[i+I] = 0.0;
    for (int i=0;i<NDOF_TEST_ELEMENT;i++)
      for(int I=0;I<1;I++)
	for(int J=0;J<1;J++)
	  grad_test[i+I] += jacInv[J+I]*grad_test_ref[i+J];
  }
  
  inline void backwardEuler(const double& dt, const double& m_old, const double& m, const double& dm, double& mt, double& dmt)
  {  
    mt =(m-m_old)/dt;
    dmt = dm/dt;
  }

  inline void bdf(const double& alpha, const double& beta, const double& m, const double& dm, double& mt, double& dmt)
  {  
    mt =alpha*m + beta;
    dmt = alpha*dm;
  }

  inline void bdfC2(const double& alpha, const double& beta, const double& m, const double& dm, const double& dm2, double& mt, double& dmt, double& dm2t)
  {  
    mt =alpha*m + beta;
    dmt = alpha*dm;
    dm2t = alpha*dm2;
  }

  inline double Mass_weak(const double& mt, const double& w_dV)
  {
    return mt*w_dV;
  }

  inline double MassJacobian_weak(const double& dmt,
				  const double& v,
				  const double& w_dV)
  {
    return dmt*v*w_dV;
  }

  inline double Mass_strong(const double& mt)
  {
    return mt; 
  }

  inline double MassJacobian_strong(const double& dmt,
				    const double& v)
  {
    return dmt*v;
  }

  inline double Mass_adjoint(const double& dmt,
			     const double& w_dV)
  {
    return dmt*w_dV;
  }

  /*
   *  pressureProjection_weak
   *  -----------------------
   *
   *  Inner product calculation of the pressure projection
   *  stablization method of Bochev, Dohrmann and
   *  Gunzburger (2006).
   *
   *  @param viscosity viscosity at point
   *  @param p the pressure value (either trial function
   *           or actual value)
   *  @param p_avg the pressure projection value (either
   *           1./3. for test functions of the average
   *	       value of the pressure on the element)
   *  @param dV this is the integral weight
   *
   */

  inline double pressureProjection_weak(const double& viscosity,
					const double& p,
					const double& p_avg,
					const double& q,
					const double& dV)
  {
    if (viscosity==0.){ return 0.;}
    return (1./viscosity)*(p-p_avg)*(q-1./3.)*dV;
  }

  inline double Advection_weak(const double  f[1],
			       const double grad_w_dV[1])
  {
    double tmp=0.0;
    for(int I=0;I<1;I++)
      tmp -= f[I]*grad_w_dV[I];
    return tmp;
  }

  inline double AdvectionJacobian_weak(const double df[1],
				       const double& v,
				       const double grad_w_dV[1])
  {
    double tmp=0.0;
    for(int I=0;I<1;I++)
      tmp -= df[I]*v*grad_w_dV[I];
    return tmp;
  }

  inline double Advection_strong(const double df[1],
				 const double grad_u[1])
  {
    double tmp=0.0;
    for(int I=0;I<1;I++)
      tmp += df[I]*grad_u[I];
    return tmp;
  }

  inline double AdvectionJacobian_strong(const double df[1],
					 const double grad_v[1])
  {
    double tmp=0.0;
    for(int I=0;I<1;I++)
      tmp += df[I]*grad_v[I];
    return tmp;
  }

  inline double Advection_adjoint(const double df[1],
				  const double grad_w_dV[1])
  {
    double tmp=0.0;
    for(int I=0;I<1;I++)
      tmp -= df[I]*grad_w_dV[I];
    return tmp;
  }

  inline double Hamiltonian_weak(const double& H,
				 const double& w_dV)
  {
    return H*w_dV;
  }

  inline double HamiltonianJacobian_weak(const double dH[1],
					 const double grad_v[1],
					 const double& w_dV)
  {
    double tmp=0.0;
    for(int I=0;I<1;I++)
      tmp += dH[I]*grad_v[I]*w_dV;
    return tmp;
  }

  inline double Hamiltonian_strong(const double dH[1],
				   const double grad_u[1])
  {
    double tmp=0.0;
    for(int I=0;I<1;I++)
      tmp += dH[I]*grad_u[I];
    return tmp;
  }

  inline double HamiltonianJacobian_strong(const double dH[1],
					   const double grad_v[1])
  {
    double tmp=0.0;
    for(int I=0;I<1;I++)
      tmp += dH[I]*grad_v[I];
    return tmp;
  }

  inline double Hamiltonian_adjoint(const double dH[1],
				    const double grad_w_dV[1])
  {
    double tmp=0.0;
    for(int I=0;I<1;I++)
      tmp -= dH[I]*grad_w_dV[I];
    return tmp;
  }

  inline double Diffusion_weak(proteus_csr_index_t* rowptr,
			       proteus_csr_index_t* colind,
			       double* a,
			       const double grad_phi[1],
			       const double grad_w_dV[1])
  {
    double tmp=0.0;
    for(int I=0;I<1;I++)
      for (proteus_csr_index_t m=rowptr[I];m<rowptr[I+1];m++)
	tmp += a[m]*grad_phi[colind[m]]*grad_w_dV[I];
    return tmp;
  }

  inline double DiffusionJacobian_weak(proteus_csr_index_t* rowptr,
				       proteus_csr_index_t* colind,
				       double* a,
				       double* da,
				       const double grad_phi[1],
				       const double grad_w_dV[1],
				       const double& dphi,
				       const double& v,
				       const double grad_v[1])
  {
    double daProduct=0.0,dphiProduct=0.0;
    for (int I=0;I<1;I++)
      for (proteus_csr_index_t m=rowptr[I];m<rowptr[I+1];m++)
	{
	  daProduct += da[m]*grad_phi[colind[m]]*grad_w_dV[I];
	  dphiProduct += a[m]*grad_v[colind[m]]*grad_w_dV[I];
	}
    return daProduct*v+dphiProduct*dphi;
  }

  inline double SimpleDiffusionJacobian_weak(proteus_csr_index_t* rowptr,
					     proteus_csr_index_t* colind,
					     double* a,
					     const double grad_v[1],
					     const double grad_w_dV[1])
  {
    double dphiProduct=0.0;
    for (int I=0;I<1;I++)
      for (proteus_csr_index_t m=rowptr[I];m<rowptr[I+1];m++)
	{
	  dphiProduct += a[m]*grad_v[colind[m]]*grad_w_dV[I];
	}
    return dphiProduct;
  }

  inline double Reaction_weak(const double& r,
			      const double& w_dV)
  {
    return r*w_dV;
  }
  
  inline double ReactionJacobian_weak(const double& dr,
				      const double& v,
				      const double& w_dV)
  {
    return dr*v*w_dV;
  }

  inline double Reaction_strong(const double& r)
  {
    return r; 
  }

  inline double ReactionJacobian_strong(const double& dr,
					const double& v)
  {
    return dr*v;
  }

  inline double Reaction_adjoint(const double& dr,
				 const double& w_dV)
  {
    return dr*w_dV;
  }
  
  inline void calculateNumericalDiffusion(const double& shockCapturingDiffusion,
					  const double& elementDiameter,
					  const double& strong_residual,
					  const double grad_u[1],
					  double& numDiff)
  {
    double h,
      num,
      den,
      n_grad_u;
    h = elementDiameter;
    n_grad_u = 0.0;
    for (int I=0;I<1;I++)
      n_grad_u += grad_u[I]*grad_u[I];
    num = shockCapturingDiffusion*0.5*h*fabs(strong_residual);
    den = sqrt(n_grad_u+1.0e-12);
    //cek hack shockCapturingDiffusion*fabs(strong_residual)*grad_phi_G_grad_phi
    numDiff = num/den;
  }


  inline void calculateNumericalDiffusion(const double& shockCapturingDiffusion,
					  const double  G[1*1],
					  const double& strong_residual,
					  const double grad_u[1],
					  double& numDiff)
  {
    double  den = 0.0;
    for (int I=0;I<1;I++)
      for (int J=0;J<1;J++)
        den += grad_u[I]*G[I+J]*grad_u[J];

    numDiff = shockCapturingDiffusion*fabs(strong_residual)/(sqrt(den+1.0e-12));
  }

  inline void calculateNumericalDiffusion(const double& shockCapturingDiffusion,
                                          const double& uref, const double& beta,
					  const double  G[1*1],
					  const double& G_dd_G,
					  const double& strong_residual,
					  const double grad_u[1],
					  double& numDiff)
  {
    double  den = 0.0;    
    for (int I=0;I<1;I++)
      for (int J=0;J<1;J++)
        den += grad_u[I]*G[I+J]*grad_u[J];

    double h2_uref_1 = 1.0/(sqrt(den+1.0e-12));
    double h2_uref_2 = 1.0/(uref*sqrt(G_dd_G+1.0e-12));
    numDiff = shockCapturingDiffusion*fabs(strong_residual)*pow(h2_uref_1, 2.0-beta)*pow(h2_uref_2,beta-1.0);
  }



  inline void calculateNumericalDiffusion(const double& shockCapturingDiffusion,
					  const double  G[1*1],
					  const double& strong_residual,
					  const double  vel[1],
					  const double  grad_u[1],
					  double& numDiff)
  {
    double  den1 = 0.0,den2=0.0, nom=0.0;
    for (int I=0;I<1;I++)
      {
      nom += vel[I]*vel[I];
      den2+= grad_u[I]*grad_u[I]; 
      for (int J=0;J<1;J++)
        den1 += vel[I]*G[I+J]*vel[J];
      }
    numDiff = shockCapturingDiffusion*fabs(strong_residual)*(sqrt(nom/(den1*den2 + 1.0e-12)));
  }


  inline void calculateNumericalDiffusion(const double& shockCapturingDiffusion,
					  const double& elementDiameter,
					  const double& strong_residual,
					  const double grad_u[1],
					  double& gradNorm,
					  double& gradNorm_last,
					  double& numDiff)
  {
    double h,
      num,
      n_grad_u;
    h = elementDiameter;
    n_grad_u = 0.0;
    for (int I=0;I<1;I++)
      n_grad_u += grad_u[I]*grad_u[I];
    num = shockCapturingDiffusion*0.5*h*fabs(strong_residual);
    gradNorm = sqrt(n_grad_u+1.0e-12);
    //cek hack shockCapturingDiffusion*fabs(strong_residual)*grad_phi_G_grad_phi    
    numDiff = num/gradNorm_last;
  }

  inline double SubgridError(const double& error,
			     const double& Lstar_w_dV)
  {
    return error*Lstar_w_dV;
  }

  inline double SubgridErrorJacobian(const double& derror,
				     const double& Lstar_w_dV)
  {
    return derror*Lstar_w_dV;
  }

  inline double NumericalDiffusion(const double& numDiff,
				   const double grad_u[1],
				   const double grad_w_dV[1])
  {
    double tmp=0.0;
    for (int I=0;I<1;I++)
      tmp +=  numDiff*grad_u[I]*grad_w_dV[I];
    return tmp;
  }

  inline double NumericalDiffusionJacobian(const double& numDiff,
					   const double grad_v[1],
					   const double grad_w_dV[1])
  {
    double tmp=0.0;
    for (int I=0;I<1;I++)
      tmp += numDiff*grad_v[I]*grad_w_dV[I];
    return tmp;
  }

 

  inline double ExteriorElementBoundaryFlux(const double& flux,
					    const double& w_dS)
  {
    return flux*w_dS;
  }

  inline double InteriorElementBoundaryFlux(const double& flux,
					    const double& w_dS)
  {
    return flux*w_dS;
  }

  inline double ExteriorNumericalAdvectiveFluxJacobian(const double& dflux_left,
						       const double& v)
  {
    return dflux_left*v;
  }

  inline double InteriorNumericalAdvectiveFluxJacobian(const double& dflux_left,
						       const double& v)
  {
    return dflux_left*v;
  }

  inline double ExteriorElementBoundaryScalarDiffusionAdjoint(const int& isDOFBoundary,
							      const int& isFluxBoundary,
							      const double& sigma,
							      const double& u,
							      const double& bc_u,
							      const double normal[1],
							      const double& a,
							      const double grad_w_dS[1])
  {
    double tmp=0.0;
    for(int I=0;I<1;I++)
      {
	tmp += normal[I]*grad_w_dS[I];
      }
    tmp *= (1.0-isFluxBoundary)*isDOFBoundary*sigma*(u-bc_u)*a;
    return tmp;
  }

  inline double ExteriorElementBoundaryScalarDiffusionAdjointJacobian(const int& isDOFBoundary,
								      const int& isFluxBoundary,
								      const double& sigma,
								      const double& v,
								      const double normal[1],
								      const double& a,
								      const double grad_w_dS[1])
  {
    double tmp=0.0;
    for(int I=0;I<1;I++)
      {
	tmp += normal[I]*grad_w_dS[I];
      }
    tmp *= (1.0-isFluxBoundary)*isDOFBoundary*sigma*v*a;
    return tmp;
  }
						
  inline double ExteriorElementBoundaryDiffusionAdjoint(const int& isDOFBoundary,
							const int& isFluxBoundary,
							const double& sigma,
							const double& u,
							const double& bc_u,
							const double normal[1],
							proteus_csr_index_t* rowptr,
							proteus_csr_index_t* colind,
							double* a,
							const double grad_w_dS[1])
  {
    double tmp=0.0;
    for(int I=0;I<1;I++)
      for (proteus_csr_index_t m=rowptr[I];m<rowptr[I+1];m++)
	tmp += (1.0-isFluxBoundary)*isDOFBoundary*sigma*(u-bc_u)*a[m]*normal[colind[m]]*grad_w_dS[I];
    return tmp;
  }

  inline double ExteriorElementBoundaryDiffusionAdjointJacobian(const int& isDOFBoundary,
								const int& isFluxBoundary,
								const double& sigma,
								const double& v,
								const double normal[1],
								proteus_csr_index_t* rowptr,
								proteus_csr_index_t* colind,
								double* a,
								const double grad_w_dS[1])
  {
    double tmp=0.0;
    for(int I=0;I<1;I++)
      for (proteus_csr_index_t m=rowptr[I];m<rowptr[I+1];m++)
	tmp += (1.0-isFluxBoundary)*isDOFBoundary*sigma*v*a[m]*normal[colind[m]]*grad_w_dS[I];
    return tmp;
  }

  inline void calculateMapping_element(const proteus_index_t eN,
				       const int k,
				       double* mesh_dof,
				       proteus_index_t* mesh_l2g,
				       //proteus::pyarray<double>& mesh_trial_ref,
				       double* mesh_trial_ref,
				       double* mesh_grad_trial_ref,
				       double* jac,
				       double& jacDet,
				       double* jacInv,
				       double& x,
				       double& y)
  {
    mapping.calculateMapping_element(eN,k,mesh_dof,mesh_l2g,mesh_trial_ref,mesh_grad_trial_ref,jac,jacDet,jacInv,x,y);
  }

  inline void calculateH_element(const proteus_index_t eN,
				 const int k,
				 double* h_dof,
				 proteus_index_t* mesh_l2g,
				 //proteus::pyarray<double>& mesh_trial_ref,
				 double* mesh_trial_ref,
				 double& h)
  {
    mapping.calculateH_element(eN,
			       k,
			       h_dof,
			       mesh_l2g,
			       mesh_trial_ref,
			       h);
  }

  inline void calculateMapping_element(const proteus_index_t eN,
				       const int k,
				       double* mesh_dof,
				       proteus_index_t* mesh_l2g,
				       //proteus::pyarray<double>& mesh_trial_ref,
				       double* mesh_trial_ref,
				       double* mesh_grad_trial_ref,
				       double* jac,
				       double& jacDet,
				       double* jacInv,
				       double& x,
				       double& y,
				       double& z)
  {
    mapping.calculateMapping_element(eN,k,mesh_dof,mesh_l2g,mesh_trial_ref,mesh_grad_trial_ref,jac,jacDet,jacInv,x,y,z);
  }

  inline void calculateMappingVelocity_element(const proteus_index_t eN,
					       const int k,
					       double* meshVelocity_dof,
					       proteus_index_t* mesh_l2g,
					       //proteus::pyarray<double>& mesh_trial_ref,
					       double* mesh_trial_ref,
					       double& xt,
					       double& yt)
  {
    mapping.calculateMappingVelocity_element(eN,k,meshVelocity_dof,mesh_l2g,mesh_trial_ref,xt,yt);
  }

  inline void calculateMappingVelocity_element(const proteus_index_t eN,
					       const int k,
					       double* meshVelocity_dof,
					       proteus_index_t* mesh_l2g,
					       //proteus::pyarray<double>& mesh_trial_ref,
					       double* mesh_trial_ref,
					       double& xt,
					       double& yt,
					       double& zt)
  {
    mapping.calculateMappingVelocity_element(eN,k,meshVelocity_dof,mesh_l2g,mesh_trial_ref,xt,yt,zt);
  }

  inline
  void calculateMapping_elementBoundary(const proteus_index_t eN,
					const int ebN_local,
					const int kb,
					const int ebN_local_kb,
					double* mesh_dof,
					proteus_index_t* mesh_l2g,
					double* mesh_trial_trace_ref,
					double* mesh_grad_trial_trace_ref,
					double* boundaryJac_ref,
					double* jac,
					double& jacDet,
					double* jacInv,
					double* boundaryJac,
					double* metricTensor,
					double& metricTensorDetSqrt,
					double* normal_ref,
					double* normal,
					double& x,
					double& y)
  {
    mapping.calculateMapping_elementBoundary(eN,
					     ebN_local,
					     kb,
					     ebN_local_kb,
					     mesh_dof,
					     mesh_l2g,
					     mesh_trial_trace_ref,
					     mesh_grad_trial_trace_ref,
					     boundaryJac_ref,
					     jac,
					     jacDet,
					     jacInv,
					     boundaryJac,
					     metricTensor,
					     metricTensorDetSqrt,
					     normal_ref,
					     normal,
					     x,
					     y);
  }

  inline
  void calculateMapping_elementBoundary(const proteus_index_t eN,
					const int ebN_local,
					const int kb,
					const int ebN_local_kb,
					double* mesh_dof,
					proteus_index_t* mesh_l2g,
					double* mesh_trial_trace_ref,
					double* mesh_grad_trial_trace_ref,
					double* boundaryJac_ref,
					double* jac,
					double& jacDet,
					double* jacInv,
					double* boundaryJac,
					double* metricTensor,
					double& metricTensorDetSqrt,
					double* normal_ref,
					double* normal,
					double& x,
					double& y,
					double& z)
  {
    mapping.calculateMapping_elementBoundary(eN,
					     ebN_local,
					     kb,
					     ebN_local_kb,
					     mesh_dof,
					     mesh_l2g,
					     mesh_trial_trace_ref,
					     mesh_grad_trial_trace_ref,
					     boundaryJac_ref,
					     jac,
					     jacDet,
					     jacInv,
					     boundaryJac,
					     metricTensor,
					     metricTensorDetSqrt,
					     normal_ref,
					     normal,
					     x,
					     y,
					     z);
  }

  inline
    void calculateMappingVelocity_elementBoundary(const proteus_index_t eN,
						  const int ebN_local,
						  const int kb,
						  const int ebN_local_kb,
						  double* mesh_velocity_dof,
						  proteus_index_t* mesh_l2g,
						  double* mesh_trial_trace_ref,
						  double& xt,
						  double& yt,
						  double* normal,
						  double* boundaryJac,
						  double* metricTensor,
						  double& metricTensorDetSqrt)
  {
    mapping.calculateMappingVelocity_elementBoundary(eN,
						     ebN_local,
						     kb,
						     ebN_local_kb,
						     mesh_velocity_dof,
						     mesh_l2g,
						     mesh_trial_trace_ref,
						     xt,
						     yt,
						     normal,
						     boundaryJac,
						     metricTensor,
						     metricTensorDetSqrt);
  }
  inline
    void calculateMappingVelocity_elementBoundary(const proteus_index_t eN,
						  const int ebN_local,
						  const int kb,
						  const int ebN_local_kb,
						  double* mesh_velocity_dof,
						  proteus_index_t* mesh_l2g,
						  double* mesh_trial_trace_ref,
						  double& xt,
						  double& yt,
						  double& zt,
						  double* normal,
						  double* boundaryJac,
						  double* metricTensor,
						  double& metricTensorDetSqrt)
  {
    mapping.calculateMappingVelocity_elementBoundary(eN,
						     ebN_local,
						     kb,
						     ebN_local_kb,
						     mesh_velocity_dof,
						     mesh_l2g,
						     mesh_trial_trace_ref,
						     xt,
						     yt,
						     zt,
						     normal,
						     boundaryJac,
						     metricTensor,
						     metricTensorDetSqrt);
  }
  double Stress_u_weak(double* stress, double* grad_test_dV)
  {
    return stress[sXX]*grad_test_dV[X];
  }
  double StressJacobian_u_u_weak(double* dstress, double* grad_trial, double* grad_test_dV)
  {
    return
      dstress[sXX*nSymTen]*grad_trial[X];
  }

  double ExteriorElementBoundaryStressFlux(const double& stressFlux,const double& disp_test_dS)
  {
    return stressFlux*disp_test_dS;
  }
  double ExteriorElementBoundaryStressFluxJacobian(const double& dstressFlux,const double& disp_test_dS)
  {
    return dstressFlux*disp_test_dS;
  }
};
#endif
