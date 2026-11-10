#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <strings.h>
#include <assert.h>
#include "timeIntegration.h"
/** \file timeIntegration.c
    \ingroup timeIntegration
    @{
*/
/*Pseudo-Transient Continuation TTE formula choice for dt*/
void psiTCtteDT(int nPoints,
		double tau,
		double dtn,
		double dtnm1,
		double * yn,
		double * ypn,
		double * ypnm1,
		double * dtnp1)
{
  int k;
  double dy2,dtavgInv,maxdy2,dtOut,safety,eps;

  dtavgInv = 2.0/(dtn+dtnm1);
  maxdy2 = 0.0;
  safety = 0.9;
  eps    = 1.0e-8;
  /*mwf debug
  printf("psiTCtte nPoints=%d tau= %g dtn=%g dtnm1=%g \n",nPoints,tau,dtn,dtnm1);
  */
  for (k=0; k < nPoints; k++)
    {
      dy2 = ypn[k]-ypnm1[k];
      maxdy2 = fmax(maxdy2,fabs(dy2)*0.5/(1.0+fabs(yn[k])));
      /*mwf debug
      printf("k=%d ypn= %g ypnm1= %g maxdy2= %g\n",k,ypn[k],ypnm1[k],maxdy2);
      */
    }
  /*mwf debug
  printf("psiTCtte maxdy2= %g\n",maxdy2);
  */
  dtOut = sqrt(tau/(maxdy2+eps))*safety;
  *dtnp1 = dtOut;
}

/***********************************************************************
  RKDG limiting procedures, probably need a better spot
 ***********************************************************************/
double msign(double a)
{
  return (a < 0.0 ? -1.0 : 1.0);
}
double minmod2(double a, double b)
{
  double aa = fabs(a), bb = fabs(b);
  if (a*b < 0.0) return 0.0;
  return aa <= bb ? a : b;  
}
double musclmod(double dU, double dU1, double dU2)
{
  double tmp = minmod2(dU1,dU2);
  return minmod2(dU,2.0*tmp);
}
double minmod3(double dU, double dU1, double dU2)
{
  double tmp = minmod2(dU1,dU2);
  return minmod2(dU,tmp);
}
double mminmod2(double dU, double dU1, double M, int* tag)
{
  if (fabs(dU) <= M)
    {
      *tag = 1;
      return dU;
    }
  *tag = 0;
  return minmod2(dU,dU1);
}
/*make sure doesn't conflict with postprocessing's version*/
int invertLocal3d(double A[3][3], double AI[3][3])
{
  int i,j;
  double detA,detAinv;
  detA = 
    A[0][0]*(A[1][1]*A[2][2]-A[2][1]*A[1][2])-
    A[0][1]*(A[1][0]*A[2][2]-A[2][0]*A[1][2])+
    A[0][2]*(A[1][0]*A[2][1]-A[2][0]*A[1][1]);
  /*mwf debug*/
  if (fabs(detA) <= 0.0)
    {
      printf("WARNING invertLocal3d detA= %g A= \n",detA);
      for (i=0; i < 3; i++)
	{
	  for (j=0; j < 3; j++)
	    printf("%g  ",A[i][j]);
	  printf("\n");
	}
      return -1;
    }
  assert(fabs(detA) > 0.0);
  detAinv = 1.0/detA;
  AI[0][0] = detAinv*(A[1][1]*A[2][2]-A[1][2]*A[2][1]);
  AI[0][1] = detAinv*(A[0][2]*A[2][1]-A[0][1]*A[2][2]);
  AI[0][2] = detAinv*(A[0][1]*A[1][2]-A[0][2]*A[1][1]);
  
  AI[1][0] = detAinv*(A[1][2]*A[2][0]-A[1][0]*A[2][2]);
  AI[1][1] = detAinv*(A[0][0]*A[2][2]-A[0][2]*A[2][0]);
  AI[1][2] = detAinv*(A[0][2]*A[1][0]-A[0][0]*A[1][2]);
  
  AI[2][0] = detAinv*(A[1][0]*A[2][1]-A[1][1]*A[2][0]);
  AI[2][1] = detAinv*(A[0][1]*A[2][0]-A[0][0]*A[2][1]);
  AI[2][2] = detAinv*(A[0][0]*A[1][1]-A[0][1]*A[1][0]);
  return 0;
}


/* local computation ignores orientation */
void computeLocalGradient1d(double * x0, double * x1, double * dN0, double * dN1, 
			    double u0, double u1, double * dU)
{
  register double volume;
  volume = fabs(x1[0]-x0[0]); 
  dN0[0]= -(x1[0]-x0[0]);/* -n0*area/volume/nSpace */
  dN1[0]= -(x0[0]-x1[0]);/* -n1*area/volume/nSpace */
  dU[0] = dN0[0]*u0 + dN1[0]*u1;
} 
/* local computation ignores orientation */
void computeLocalGradient2d(double * x0, double * x1, double * x2, double * dN0, 
			    double * dN1, double * dN2,
			    double u0, double u1, double u2, double * dU)
{
  double detJ,dx10[2],dx20[2],JinvT[2][2];
  /***************************************************
     \hat{N}_0 = 1-\hat{x}-\hat{y}, \hat{N}_1 = \hat{x}
     \hat{N}_2 = \hat{y}
   ***************************************************/
  const double dN0h[2] = {-1.,-1.}; 
  const double dN1h[2] = { 1., 0.};
  const double dN2h[2] = { 0., 1.};
  dx10[0] = x1[0]-x0[0]; dx10[1] = x1[1]-x0[1]; 
  dx20[0] = x2[0]-x0[0]; dx20[1] = x2[1]-x0[1];
  detJ = dx10[0]*dx20[1]-dx10[1]*dx20[0];

  /***************************************************
    J = |x1-x0  x2-x0|  Jinv = |y2-y0  x0-x2|*1/det JinvT = |y2-y0 y0-y1|*1/det
        |y1-y0  y2-y0|         |y0-y1  x1-x0|               |x0-x2 x1-x0|
   *******************************/
  JinvT[0][0] = dx20[1]/detJ; JinvT[0][1] =-dx10[1]/detJ; 
  JinvT[1][0] =-dx20[0]/detJ; JinvT[1][1] = dx10[0]/detJ;

  dN0[0] = JinvT[0][0]*dN0h[0]+JinvT[0][1]*dN0h[1];
  dN0[1] = JinvT[1][0]*dN0h[0]+JinvT[1][1]*dN0h[1];

  dN1[0] = JinvT[0][0]*dN1h[0]+JinvT[0][1]*dN1h[1];
  dN1[1] = JinvT[1][0]*dN1h[0]+JinvT[1][1]*dN1h[1];

  dN2[0] = JinvT[0][0]*dN2h[0]+JinvT[0][1]*dN2h[1];
  dN2[1] = JinvT[1][0]*dN2h[0]+JinvT[1][1]*dN2h[1];

  dU[0]  = u0*dN0[0] + u1*dN1[0] + u2*dN2[0];
  dU[1]  = u0*dN0[1] + u1*dN1[1] + u2*dN2[1];
}

void computeLocalGradient3d(double * x0, double * x1, double * x2, double * x3,
			    double * dN0, double * dN1, double * dN2, double * dN3,
			    double u0, double u1, double u2, double u3, double * dU)
{
  double dx[3][3],Jinv[3][3],JinvT[3][3];
  int gradientFailed = 0;
  /***************************************************
     \hat{N}_0 = 1-\hat{x}-\hat{y}-\hat{z}, \hat{N}_1 = \hat{x}
     \hat{N}_2 = \hat{y}, \hat{N}_3 = \hat{z}
   ***************************************************/
  const double dN0h[3] = {-1.,-1.,-1.}; 
  const double dN1h[3] = { 1., 0.,0.};
  const double dN2h[3] = { 0., 1.,0.};
  const double dN3h[3] = { 0., 0.,1.};
  dx[0][0] = x1[0]-x0[0]; dx[1][0] = x1[1]-x0[1]; dx[2][0] = x1[2]-x0[2];
  dx[0][1] = x2[0]-x0[0]; dx[1][1] = x2[1]-x0[1]; dx[2][1] = x2[2]-x0[2];
  dx[0][2] = x3[0]-x0[0]; dx[1][2] = x3[1]-x0[1]; dx[2][2] = x3[2]-x0[2];
 
  /*find routine for 3x3 inverse and jacobian*/
  gradientFailed = invertLocal3d(dx,Jinv);
  if (gradientFailed != 0)
    {
      dU[0]=0.0; dU[1]=0.0; dU[2]=0.0;
    }
  JinvT[0][0] = Jinv[0][0]; JinvT[0][1] = Jinv[1][0]; JinvT[0][2] = Jinv[2][0];
  JinvT[1][0] = Jinv[0][1]; JinvT[1][1] = Jinv[1][1]; JinvT[1][2] = Jinv[2][1];
  JinvT[2][0] = Jinv[0][2]; JinvT[2][1] = Jinv[1][2]; JinvT[2][2] = Jinv[2][2];


  
  dN0[0] = JinvT[0][0]*dN0h[0]+JinvT[0][1]*dN0h[1]+JinvT[0][2]*dN0h[2];
  dN0[1] = JinvT[1][0]*dN0h[0]+JinvT[1][1]*dN0h[1]+JinvT[1][2]*dN0h[2];
  dN0[2] = JinvT[2][0]*dN0h[0]+JinvT[2][1]*dN0h[1]+JinvT[2][2]*dN0h[2];

  dN1[0] = JinvT[0][0]*dN1h[0]+JinvT[0][1]*dN1h[1]+JinvT[0][2]*dN1h[2];
  dN1[1] = JinvT[1][0]*dN1h[0]+JinvT[1][1]*dN1h[1]+JinvT[1][2]*dN1h[2];
  dN1[2] = JinvT[2][0]*dN1h[0]+JinvT[2][1]*dN1h[1]+JinvT[2][2]*dN1h[2];

  dN2[0] = JinvT[0][0]*dN2h[0]+JinvT[0][1]*dN2h[1]+JinvT[0][2]*dN2h[2];
  dN2[1] = JinvT[1][0]*dN2h[0]+JinvT[1][1]*dN2h[1]+JinvT[1][2]*dN2h[2];
  dN2[2] = JinvT[2][0]*dN2h[0]+JinvT[2][1]*dN2h[1]+JinvT[2][2]*dN2h[2];

  dN3[0] = JinvT[0][0]*dN3h[0]+JinvT[0][1]*dN3h[1]+JinvT[0][2]*dN3h[2];
  dN3[1] = JinvT[1][0]*dN3h[0]+JinvT[1][1]*dN3h[1]+JinvT[1][2]*dN3h[2];
  dN3[2] = JinvT[2][0]*dN3h[0]+JinvT[2][1]*dN3h[1]+JinvT[2][2]*dN3h[2];

  dU[0]  = u0*dN0[0] + u1*dN1[0] + u2*dN2[0] + u3*dN3[0];
  dU[1]  = u0*dN0[1] + u1*dN1[1] + u2*dN2[1] + u3*dN3[1];
  dU[2]  = u0*dN0[2] + u1*dN1[2] + u2*dN2[2] + u3*dN3[2];
}
void applyDGlimitingP1Lagrange1d(int limiterFlag,
				 proteus_index_t nElements_global,
				 int nNodes_element,
				 int nElementBoundaries_element,
				 int nDOF_element,
				 proteus_index_t * elementNodesArray,  
				 proteus_index_t * elementNeighborsArray,
				 double * nodeArray,
				 double * elementBarycentersArray,
				 proteus_index_t * l2g,
				 int * tag,
				 double * Uin,
				 double * Uout)
{
  /***********************************************************************
     Apply basic DG limiting procedure in 1d assuming input/output fem functions 
      have a P1 Lagrange basis representation over each element

     Assumes that local node numbering agrees with local dof numbering
     Assumes that local elementBoundary i is across from node i

     Input:
       limitingFlag                 --- which limiter to use on slopes
                                        0 minmod3(\pd{u}{x}_i,\Delta_x u_i+1/2,\Delta_x u_i-1/2)
                                        1 muscl(\pd{u}{x}_i,\Delta_x u_i+1/2,\Delta_x u_i-1/2) (doubles slopes)
                                        2 minmod2(\Delta_x u_i+1/2,\Delta_x u_i-1/2)
       elementNodesArray[eN,:]      --- global node numbers on element eN
       elementBoundariesArray[eN,:] --- global element boundaries on element eN
       elementNeighborsArray[eN,i]  --- global element number across local elementBoundary i
                                        -1 ---> exterior boundary
       nodeArray[nN,:]              --- physical coordinates of node nN (always 3)
       elementBarycentersArray[eN,:]--- physical coordiantes of barycenter for element eN (always 3)
       l2g[eN,:]                    --- global degree of freedom numbers for local unknowns
       Uin                          --- input P1 degrees of freedom values


     Output:
       Uout                         --- output P1 degrees of freedom values 
       tag[eN]                      --- 1 if original P1 representation on element eN ok (not limited)

   ***********************************************************************/
  
  int eN;
  proteus_index_t eN0, eN1, nN0, nN1, ndof0, ndof1;

  double dUlim,dU,dU0,dU1,Ubar,Ubar0,Ubar1;
  double xbar,xbar0,xbar1,dx;

  for (eN=0; eN < nElements_global; eN++)
    {
      ndof0 = l2g[eN*nDOF_element + 0]; ndof1 = l2g[eN*nDOF_element + 1];
      nN0   = elementNodesArray[eN*nNodes_element + 0]; 
      nN1   = elementNodesArray[eN*nNodes_element + 1];
      dx    = nodeArray[nN1*3 + 0] - nodeArray[nN0*3 + 0];
      xbar  = elementBarycentersArray[eN*3 + 0];

      dU    = (Uin[ndof1]-Uin[ndof0])/dx;
      Ubar  = 0.5*(Uin[ndof1]+Uin[ndof0]);

      /*** neighbor across from node 0 and adjacent to elementBoundary 0***/
      /* in case on physical boundary*/
      xbar0 = nodeArray[nN1*3 + 0]; Ubar0 = Uin[ndof1];
      eN0   = elementNeighborsArray[eN*nElementBoundaries_element + 0];
      if (eN0 >= 0)
	{
	  xbar0 = elementBarycentersArray[eN0*3 + 0];
	  Ubar0 = 0.5*(Uin[l2g[eN0*nDOF_element + 0]] + Uin[l2g[eN0*nDOF_element + 1]]); 
	}
      dU0 = (Ubar0-Ubar)/(xbar0-xbar);

      /*** neighbor across from node 1 and adjacent to elementBoundary 1***/
      /* in case on physical boundary*/
      xbar1 = nodeArray[nN0*3 + 0]; Ubar1 = Uin[ndof0];
      eN1   = elementNeighborsArray[eN*nElementBoundaries_element + 1];
      if (eN1 >= 0)
	{
	  xbar1 = elementBarycentersArray[eN1*3 + 0];
	  Ubar1 = 0.5*(Uin[l2g[eN1*nDOF_element + 0]] + Uin[l2g[eN1*nDOF_element + 1]]); 
	}
      dU1 = (Ubar1-Ubar)/(xbar1-xbar);
      /*dU's have dx's */ 
      if (limiterFlag == 2)
	dUlim = minmod2(dU0,dU1); 
      else if (limiterFlag == 1)
	dUlim = musclmod(dU,dU0,dU1);
      else
	dUlim = minmod3(dU,dU0,dU1); 
      tag[eN] = fabs(dUlim-dU) < 1.0e-6; /*need better tolerance*/

      /*back out nodal dofs from average and slope*/
      Uout[ndof0] = Ubar + (nodeArray[nN0*3 + 0]-xbar)*dUlim;
      Uout[ndof1] = Ubar + (nodeArray[nN1*3 + 0]-xbar)*dUlim;
    }/*eN*/

}

void applyDGlimitingP1Lagrange1d_withVacuumTol(int enforcePositivity,
					       double vacuumTol,
					       proteus_index_t nElements_global,
					       int nNodes_element,
					       int nElementBoundaries_element,
					       int nDOF_element,
					       proteus_index_t * elementNodesArray,  
					       proteus_index_t * elementNeighborsArray,
					       double * nodeArray,
					       double * elementBarycentersArray,
					       proteus_index_t * l2g,
					       int * tag,
					       double * Uin,
					       double * Uout)
{
  /***********************************************************************
     Apply basic DG limiting procedure in 1d assuming input/output fem functions 
      have a P1 Lagrange basis representation over each element

     Assumes that local node numbering agrees with local dof numbering
     Assumes that local elementBoundary i is across from node i

     Input:
       vacuumTol                    --- if |\bar{u}| < tol use minmod2 else muscl
                                          minmod3(\pd{u}{x}_i,\Delta_x u_i+1/2,\Delta_x u_i-1/2)
                                          muscl(\pd{u}{x}_i,\Delta_x u_i+1/2,\Delta_x u_i-1/2) (doubles slopes)
                                          minmod2(\Delta_x u_i+1/2,\Delta_x u_i-1/2)
       elementNodesArray[eN,:]      --- global node numbers on element eN
       elementBoundariesArray[eN,:] --- global element boundaries on element eN
       elementNeighborsArray[eN,i]  --- global element number across local elementBoundary i
                                        -1 ---> exterior boundary
       nodeArray[nN,:]              --- physical coordinates of node nN (always 3)
       elementBarycentersArray[eN,:]--- physical coordiantes of barycenter for element eN (always 3)
       l2g[eN,:]                    --- global degree of freedom numbers for local unknowns
       Uin                          --- input P1 degrees of freedom values


     Output:
       Uout                         --- output P1 degrees of freedom values 
       tag[eN]                      --- 1 if original P1 representation on element eN ok (not limited)

   ***********************************************************************/
  
  int eN;
  proteus_index_t eN0, eN1, nN0, nN1, ndof0, ndof1;

  double dUlim,dU,dU0,dU1,Ubar,Ubar0,Ubar1,Umin,UminA;
  double xbar,xbar0,xbar1,dx;

  for (eN=0; eN < nElements_global; eN++)
    {
      ndof0 = l2g[eN*nDOF_element + 0]; ndof1 = l2g[eN*nDOF_element + 1];
      nN0   = elementNodesArray[eN*nNodes_element + 0]; 
      nN1   = elementNodesArray[eN*nNodes_element + 1];
      dx    = nodeArray[nN1*3 + 0] - nodeArray[nN0*3 + 0];
      xbar  = elementBarycentersArray[eN*3 + 0];

      dU    = (Uin[ndof1]-Uin[ndof0])/dx;
      Ubar  = 0.5*(Uin[ndof1]+Uin[ndof0]);
      Umin  = fmin(Uin[ndof1],Uin[ndof0]);/*or Ubar?*/
      UminA = fmin(fabs(Uin[ndof1]),fabs(Uin[ndof0])); /*or |Ubar| ?*/

      /*** neighbor across from node 0 and adjacent to elementBoundary 0***/
      /* in case on physical boundary*/
      xbar0 = nodeArray[nN1*3 + 0]; Ubar0 = Uin[ndof1];
      eN0   = elementNeighborsArray[eN*nElementBoundaries_element + 0];
      if (eN0 >= 0)
	{
	  xbar0 = elementBarycentersArray[eN0*3 + 0];
	  Ubar0 = 0.5*(Uin[l2g[eN0*nDOF_element + 0]] + Uin[l2g[eN0*nDOF_element + 1]]); 
	}
      dU0 = (Ubar0-Ubar)/(xbar0-xbar);

      /*** neighbor across from node 1 and adjacent to elementBoundary 1***/
      /* in case on physical boundary*/
      xbar1 = nodeArray[nN0*3 + 0]; Ubar1 = Uin[ndof0];
      eN1   = elementNeighborsArray[eN*nElementBoundaries_element + 1];
      if (eN1 >= 0)
	{
	  xbar1 = elementBarycentersArray[eN1*3 + 0];
	  Ubar1 = 0.5*(Uin[l2g[eN1*nDOF_element + 0]] + Uin[l2g[eN1*nDOF_element + 1]]); 
	}
      dU1 = (Ubar1-Ubar)/(xbar1-xbar);
      /*dU's have dx's */ 
      if (enforcePositivity && Umin < vacuumTol)
	dUlim = minmod3(dU,dU0,dU1); 
      else if (UminA < vacuumTol)
	dUlim = minmod3(dU,dU0,dU1); 
      else
	dUlim = musclmod(dU,dU0,dU1);
      tag[eN] = fabs(dUlim-dU) < 1.0e-6; /*need better tolerance*/

      /*back out nodal dofs from average and slope*/
      Uout[ndof0] = Ubar + (nodeArray[nN0*3 + 0]-xbar)*dUlim;
      Uout[ndof1] = Ubar + (nodeArray[nN1*3 + 0]-xbar)*dUlim;
    }/*eN*/

}

void computeElementNeighborShapeGradients(proteus_index_t nElements_global,
					  int nElementBoundaries_element,
					  int nSpace,
					  const proteus_index_t * elementBoundariesArray,
					  const proteus_index_t * elementNeighborsArray,
					  double * elementBarycentersArray,
					  double * elementBoundaryBarycentersArray,
					  double * elementNeighborShapeGradients)
{
  /***********************************************************************
     compute gradient of linear shape functions (aka barycentric coordinates)
     for the simplices formed by connecting element neighbor barycenters

     store these in 
       elementNeighborShapeGradients[eN,i,j,k] ---> 
          kth coordinate of jth barycentric coordinate for ith simplex 
          formed from element eN and its nd+1 neighbors

          ith neighbor consists of element eN, neighbor i and neighbor i+1 % (nSpace+1)

       
   ***********************************************************************/
  proteus_index_t eN_opposite, ebN_global, I; int eN, ebN, ebNplus1, ebNplus2;
  double u[4] = {1.0,1.0,1.0,1.0};
  double dU[3]= {0.0,0.0,0.0};
  /*local shape gradients (nSpace+1)*/
  double dN[4][3] = {{0.0,0.0,0.0},
		     {0.0,0.0,0.0},
		     {0.0,0.0,0.0},
		     {0.0,0.0,0.0}};
  /*neighboring barycenters (nSpace+2)*/
  double xn[5][3] = {{0.0,0.0,0.0},
		     {0.0,0.0,0.0},
		     {0.0,0.0,0.0},
		     {0.0,0.0,0.0},
		     {0.0,0.0,0.0}};
  if (nSpace == 1)
    {
      for (eN = 0; eN < nElements_global; eN++)
	{
	  xn[0][0] = elementBarycentersArray[eN*3 + 0];
	  xn[0][1] = elementBarycentersArray[eN*3 + 1];
	  xn[0][2] = elementBarycentersArray[eN*3 + 2];

	  /*go through boundaries on this element and form simpleces (x0,x1), (x0,x2) etc*/
	  for (ebN=0; ebN < nElementBoundaries_element; ebN++)
	    {
	      eN_opposite = elementNeighborsArray[eN*nElementBoundaries_element + ebN];
	      if (eN_opposite < 0)/*outside domain*/
		{
		  ebN_global          = elementBoundariesArray[eN*nElementBoundaries_element + ebN];
		  xn[1][0]        = elementBoundaryBarycentersArray[ebN_global*3 +0];
		  xn[1][1]        = elementBoundaryBarycentersArray[ebN_global*3 +1];
		  xn[1][2]        = elementBoundaryBarycentersArray[ebN_global*3 +2];
		}
	      else
		{
		  xn[1][0]        = elementBarycentersArray[eN_opposite*3 +0];
		  xn[1][1]        = elementBarycentersArray[eN_opposite*3 +1];
		  xn[1][2]        = elementBarycentersArray[eN_opposite*3 +2];
		}

	      computeLocalGradient1d(xn[0],xn[1],dN[0],dN[1],u[0],u[1],dU);
	      elementNeighborShapeGradients[eN*nElementBoundaries_element*nElementBoundaries_element*nSpace + 
					    ebN*nElementBoundaries_element*nSpace + 
					    0*nSpace + 
					    0] = dN[0][0];
	      elementNeighborShapeGradients[eN*nElementBoundaries_element*nElementBoundaries_element*nSpace + 
					    ebN*nElementBoundaries_element*nSpace + 
					    1*nSpace + 
					    0] = dN[1][0];
	    }/*ebN*/
	}/*eN*/
    }/*1d*/
  else if (nSpace == 2)
    {
      for (eN = 0; eN < nElements_global; eN++)
	{
	  xn[0][0] = elementBarycentersArray[eN*3 + 0];
	  xn[0][1] = elementBarycentersArray[eN*3 + 1];
	  xn[0][2] = elementBarycentersArray[eN*3 + 2];

	  /*first collect all the relevant barycenters*/
	  for (ebN=0; ebN < nElementBoundaries_element; ebN++)
	    {
	      eN_opposite = elementNeighborsArray[eN*nElementBoundaries_element + ebN];

	      if (eN_opposite < 0)/*outside domain*/
		{
		  ebN_global          = elementBoundariesArray[eN*nElementBoundaries_element + ebN];
		  xn[ebN+1][0]        = elementBoundaryBarycentersArray[ebN_global*3 +0];
		  xn[ebN+1][1]        = elementBoundaryBarycentersArray[ebN_global*3 +1];
		  xn[ebN+1][2]        = elementBoundaryBarycentersArray[ebN_global*3 +2];
		}
	      else
		{
		  xn[ebN+1][0]        = elementBarycentersArray[eN_opposite*3 +0];
		  xn[ebN+1][1]        = elementBarycentersArray[eN_opposite*3 +1];
		  xn[ebN+1][2]        = elementBarycentersArray[eN_opposite*3 +2];
		}
	    }

	  /*
	    now form local approximations, local neighbor simplex i consists of
             (\bar{x}_{eN},\bar{x}^i_{eN},\bar{x}^{i+1}_{eN}
	  */
	  for (ebN=0; ebN < nElementBoundaries_element; ebN++)
	    {
	      ebNplus1 = (ebN+1) % nElementBoundaries_element;
	      /*offset of 1 is because xn[0] is this element*/
	      computeLocalGradient2d(xn[0],xn[ebN+1],xn[ebNplus1+1],
				     dN[0],dN[1],dN[2],
				     u[0],u[1],u[2],dU);
	      /*mwf debug
		printf("eN=%d ebN=%d ebNplus1=%d xn[0]=[%g,%g], xn[ebN+1]=[%g,%g],xn[ebNplus1+1]=[%g,%g]\n",
		eN,ebN,ebNplus1,xn[0][0],xn[0][1],xn[ebN+1][0],xn[ebN+1][1],
		xn[ebNplus1+1][0],xn[ebNplus1+1][1]);
		printf("\t dN[0]=[%g,%g],dN[1]=[%g,%g],dN[2]=[%g,%g]\n",
	        dN[0][0],dN[0][1],dN[1][0],dN[1][1],dN[2][0],dN[2][1]);
	      */
	      for (I = 0; I < nSpace; I++)
		{
		  /*this element*/
		  elementNeighborShapeGradients[eN*nElementBoundaries_element*nElementBoundaries_element*nSpace +
						ebN*nElementBoundaries_element*nSpace + 
						0*nSpace + 
						I] = dN[0][I];
		  /*neighbor ebN*/
		  elementNeighborShapeGradients[eN*nElementBoundaries_element*nElementBoundaries_element*nSpace +
						ebN*nElementBoundaries_element*nSpace + 
						1*nSpace + 
						I] = dN[1][I];
		  /*neighbor ebN+1*/
		  elementNeighborShapeGradients[eN*nElementBoundaries_element*nElementBoundaries_element*nSpace +
						ebN*nElementBoundaries_element*nSpace + 
						2*nSpace + 
						I] = dN[2][I];
		  
		}/*I*/
	    }/*ebN*/
	}/*eN*/
    }/*2d*/
  else /*3d*/
    {
      for (eN = 0; eN < nElements_global; eN++)
	{
	  xn[0][0] = elementBarycentersArray[eN*3 + 0];
	  xn[0][1] = elementBarycentersArray[eN*3 + 1];
	  xn[0][2] = elementBarycentersArray[eN*3 + 2];

	  /*first collect all the relevant barycenters*/
	  for (ebN=0; ebN < nElementBoundaries_element; ebN++)
	    {
	      eN_opposite = elementNeighborsArray[eN*nElementBoundaries_element + ebN];

	      if (eN_opposite < 0)/*outside domain*/
		{
		  ebN_global          = elementBoundariesArray[eN*nElementBoundaries_element + ebN];
		  xn[ebN+1][0]        = elementBoundaryBarycentersArray[ebN_global*3 +0];
		  xn[ebN+1][1]        = elementBoundaryBarycentersArray[ebN_global*3 +1];
		  xn[ebN+1][2]        = elementBoundaryBarycentersArray[ebN_global*3 +2];
		  /*mwf debug*/
/* 		  printf("Durlofsky3d calling computeLocalGradient3d eN=%d ebN=%d eN_opposite= %d\n", */
/* 			 eN,ebN,eN_opposite); */

/* 		  for (I=0; I < nSpace; I++) */
/* 		    { */
/* 		      printf("xn[%d][%d]= %g \n",ebN+1,I,xn[ebN+1][I]); */
/* 		    } */

		}		
	      else
		{
		  xn[ebN+1][0]        = elementBarycentersArray[eN_opposite*3 +0];
		  xn[ebN+1][1]        = elementBarycentersArray[eN_opposite*3 +1];
		  xn[ebN+1][2]        = elementBarycentersArray[eN_opposite*3 +2];
		  /*mwf debug*/
/* 		  printf("Durlofsky3d calling computeLocalGradient3d eN=%d ebN=%d eN_opposite= %d\n", */
/* 			 eN,ebN,eN_opposite); */

/* 		  for (I=0; I < nSpace; I++) */
/* 		    { */
/* 		      printf("xn[%d][%d]= %g \n",ebN+1,I,xn[ebN+1][I]); */
/* 		    } */

		}
	    }

	  /*
	    now form local approximations, local neighbor simplex i consists of
             (\bar{x}_{eN},\bar{x}^i_{eN},\bar{x}^{i+1}_{eN},\bar{x}^{i+2}
	  */
	  for (ebN=0; ebN < nElementBoundaries_element; ebN++)
	    {
	      ebNplus1 = (ebN+1) % nElementBoundaries_element;
	      ebNplus2 = (ebN+2) % nElementBoundaries_element;
	      /*offset of 1 is because xn[0] is this element*/
	      /*mwf debug*/
/* 	      printf("Durlofsky3d calling computeLocalGradient3d eN=%d ebN=%d nElementBoundaries_element=%d \n", */
/* 		     eN,ebN,nElementBoundaries_element); */
/* 	      for (I=0; I < nSpace; I++) */
/* 		{ */
/* 		  printf("xn[%d][%d]= %g xn[%d][%d]= %g xn[%d][%d]= %g xn[%d][%d]= %g \n", */
/* 			 0,I,xn[0][I], */
/* 			 ebN+1,I,xn[ebN+1][I], */
/* 			 ebNplus1+1,I,xn[ebNplus1+1][I], */
/* 			 ebNplus2+1,I,xn[ebNplus2+1][I]); */
/* 		} */
	      computeLocalGradient3d(xn[0],xn[ebN+1],xn[ebNplus1+1],xn[ebNplus2+1],
				     dN[0],dN[1],dN[2],dN[3],
				     u[0],u[1],u[2],u[3],dU);
	      for (I = 0; I < nSpace; I++)
		{
		  /*this element*/
		  elementNeighborShapeGradients[eN*nElementBoundaries_element*nElementBoundaries_element*nSpace +
						ebN*nElementBoundaries_element*nSpace + 
						0*nSpace + 
						I] = dN[0][I];
		  /*neighbor ebN*/
		  elementNeighborShapeGradients[eN*nElementBoundaries_element*nElementBoundaries_element*nSpace +
						ebN*nElementBoundaries_element*nSpace + 
						1*nSpace + 
						I] = dN[1][I];
		  /*neighbor ebN+1*/
		  elementNeighborShapeGradients[eN*nElementBoundaries_element*nElementBoundaries_element*nSpace +
						ebN*nElementBoundaries_element*nSpace + 
						2*nSpace + 
						I] = dN[2][I];
		  /*neighbor ebN+2*/
		  elementNeighborShapeGradients[eN*nElementBoundaries_element*nElementBoundaries_element*nSpace +
						ebN*nElementBoundaries_element*nSpace + 
						3*nSpace + 
						I] = dN[3][I];
		  
		}/*I*/
	    }/*ebN*/
	}/*eN*/
    }/*3d*/
}

void computeCockburnDGlimiterArrays2d(proteus_index_t nElements_global,
				      int nElementBoundaries_element,
				      int nSpace,
				      const proteus_index_t * elementBoundariesArray,
				      const proteus_index_t * elementNeighborsArray,
				      const double * elementBarycentersArray,
				      const double * elementBoundaryBarycentersArray,
				      const double * elementNeighborShapeGradients,
				      double * alphas,
				      proteus_index_t * alphaNeighbors)
{
  proteus_index_t eN,ebN,ebN_global,eN_ebN_I,eN_ebN_Iplus1,lamn[2];
  int ebN_I,ebN_Iplus1,
    onBoundary,positiveFound;
  double lam[2];

  for (eN=0; eN < nElements_global; eN++)
    {
      onBoundary = 
	elementNeighborsArray[eN*nElementBoundaries_element + 0] < 0 ||
	elementNeighborsArray[eN*nElementBoundaries_element + 1] < 0 ||
	elementNeighborsArray[eN*nElementBoundaries_element + 2] < 0;

      for (ebN=0; ebN < nElementBoundaries_element; ebN++)
	{
	  ebN_global = elementBoundariesArray[eN*nElementBoundaries_element+ebN];
	  if (onBoundary > 0)
	    {
	      lam[0] = 0.0; lam[1] = 0.0;
	      lamn[0]=-1;   lamn[1]=-1;
	      alphas[eN*nElementBoundaries_element*nSpace + ebN*nSpace + 0] = lam[0];
	      alphas[eN*nElementBoundaries_element*nSpace + ebN*nSpace + 1] = lam[1];
	      alphaNeighbors[eN*nElementBoundaries_element*nSpace + ebN*nSpace + 0] = lamn[0];
	      alphaNeighbors[eN*nElementBoundaries_element*nSpace + ebN*nSpace + 1] = lamn[1];
	    }
	  else
	    {
	      positiveFound = 0; ebN_I=0;
	      while (ebN_I < nElementBoundaries_element && positiveFound == 0)
		{
		  lam[0] = 0.0; lam[1] = 0.0; lamn[0] = -1; lamn[1] = -1;
		  /***********
                    in local neighbor simplex ordering
                      0 is always this element
                      1 is element across from current face (say i)
                      2 is i+1
		   ***********/
		  ebN_Iplus1    = (ebN_I + 1) % nElementBoundaries_element;
		  eN_ebN_I      = elementNeighborsArray[eN*nElementBoundaries_element+ebN_I];
		  eN_ebN_Iplus1 = elementNeighborsArray[eN*nElementBoundaries_element+ebN_Iplus1];
		  lamn[0] = eN_ebN_I; lamn[1] = eN_ebN_Iplus1;

		  lam[0]  = 1.0 + 
		    elementNeighborShapeGradients[eN*nElementBoundaries_element*nElementBoundaries_element*nSpace +
						  ebN_I*nElementBoundaries_element*nSpace + 
						  1*nSpace + 0] 
		    *(elementBoundaryBarycentersArray[ebN_global*3 + 0] - elementBarycentersArray[eN_ebN_I*3 + 0]) 
		    + 
		    elementNeighborShapeGradients[eN*nElementBoundaries_element*nElementBoundaries_element*nSpace +
						  ebN_I*nElementBoundaries_element*nSpace + 
						  1*nSpace + 1] 
		    *(elementBoundaryBarycentersArray[ebN_global*3 + 1] - elementBarycentersArray[eN_ebN_I*3 + 1]); 
		  lam[1]  = 1.0 + 
		    elementNeighborShapeGradients[eN*nElementBoundaries_element*nElementBoundaries_element*nSpace +
						  ebN_I*nElementBoundaries_element*nSpace + 
						  2*nSpace + 0] 
		    *(elementBoundaryBarycentersArray[ebN_global*3 + 0] - elementBarycentersArray[eN_ebN_Iplus1*3 + 0]) 
		    + 
		    elementNeighborShapeGradients[eN*nElementBoundaries_element*nElementBoundaries_element*nSpace +
						  ebN_I*nElementBoundaries_element*nSpace + 
						  2*nSpace + 1] 
		    *(elementBoundaryBarycentersArray[ebN_global*3 + 1] - elementBarycentersArray[eN_ebN_Iplus1*3 + 1]); 
		  if (lam[0] >= -1.0e-10 && lam[1] >= -1.0e-10)
		    {
		      positiveFound = 1;
		      alphas[eN*nElementBoundaries_element*nSpace + ebN*nSpace + 0] = lam[0];
		      alphas[eN*nElementBoundaries_element*nSpace + ebN*nSpace + 1] = lam[1];
		      alphaNeighbors[eN*nElementBoundaries_element*nSpace + ebN*nSpace + 0] = lamn[0];
		      alphaNeighbors[eN*nElementBoundaries_element*nSpace + ebN*nSpace + 1] = lamn[1];
		    }
		  ebN_I += 1;
		}/*while*/		  
	      assert(positiveFound); /*asserts not turned on right now stupid gcc python compile options*/
	    }/*not on boundary*/
	}/*ebn*/
    }/*eN*/
}

void applyCockburnDGlimiterP1Lagrange2d(double nu,
					double Mh2,
					proteus_index_t nElements_global,
					int nElementBoundaries_element,
					int nSpace,
					int nDOF_element,
					proteus_index_t * elementNeighborsArray,
					proteus_index_t * l2g,
					int * tag,
					double * alphas,
					proteus_index_t * alphaNeighbors,
					double * Uin,
					double * Uout)
{
  /***********************************************************************
    try to implement DG limiter for triangles according to Cockburn notes
    assumes P1 lagrange representation coming in
   ***********************************************************************/
  double dU,deltaU[3],tdeltaU[3],pos[3],neg[3],pos_eN,neg_eN,thp,thm,sumDU,
    uM[3],uMlim[3],uBar,uBar_0,uBar_1;
  int eN,ebN,ebNplus1,ebNplus2,eN_0,eN_1,onBoundary,tag_eN[3];
  /*for debugging*/
  double uavgIn,uavgOut;
  double oneThird=1.0/3.0;
  /*coulud check dimensions all agree but asserts turned off*/
  for (eN = 0; eN < nElements_global; eN++)
    {
      uBar = (Uin[l2g[eN*nDOF_element+0]]+Uin[l2g[eN*nDOF_element+1]]+Uin[l2g[eN*nDOF_element+2]])*oneThird;
      uavgIn = uBar;
      onBoundary = 
	elementNeighborsArray[eN*nElementBoundaries_element + 0] < 0 ||
	elementNeighborsArray[eN*nElementBoundaries_element + 1] < 0 ||
	elementNeighborsArray[eN*nElementBoundaries_element + 2] < 0;
      if (onBoundary > 0)
	{
	  tdeltaU[0] = 0.0; tdeltaU[1] = 0.0; tdeltaU[2] = 0.0;
	}
      else
	{
	  for (ebN=0; ebN < nElementBoundaries_element; ebN++)
	    {
	      ebNplus1 = (ebN+1) % nElementBoundaries_element; 
	      ebNplus2 = (ebN+2) % nElementBoundaries_element;
	      /*
		midpoint value for this face assuming correspondence 
		between local node numbers and faces (across)
                and nodal dofs (same point)
	      */
	      uM[ebN] = 0.5*(Uin[l2g[eN*nDOF_element+ebNplus1]]+Uin[l2g[eN*nDOF_element+ebNplus2]]);
	      eN_0    = alphaNeighbors[eN*nElementBoundaries_element*nSpace + ebN*nSpace + 0];
	      eN_1    = alphaNeighbors[eN*nElementBoundaries_element*nSpace + ebN*nSpace + 1];
	      uBar_0  = (Uin[l2g[eN_0*nDOF_element+0]]+Uin[l2g[eN_0*nDOF_element+1]]+Uin[l2g[eN_0*nDOF_element+2]])*oneThird;
	      uBar_1  = (Uin[l2g[eN_1*nDOF_element+0]]+Uin[l2g[eN_1*nDOF_element+1]]+Uin[l2g[eN_1*nDOF_element+2]])*oneThird;
	      dU = alphas[eN*nElementBoundaries_element*nSpace + ebN*nSpace + 0]*(uBar_0-uBar)+
		alphas[eN*nElementBoundaries_element*nSpace + ebN*nSpace + 1]*(uBar_1-uBar);
	      deltaU[ebN] = mminmod2(uM[ebN]-uBar,nu*dU,Mh2,&tag_eN[ebN]);
	      tdeltaU[ebN]= deltaU[ebN];
	    }
	  /*check that slopes sum to zero for mass conservation*/
	  sumDU = tdeltaU[0] + tdeltaU[1] + tdeltaU[2];
	  if (fabs(sumDU) > 1.0e-6) /*need tolerance*/
	    {
	      /*should unroll this stuff too*/
	      pos_eN = 0.0; neg_eN = 0.0;
	      for (ebN=0; ebN < nElementBoundaries_element; ebN++)
		{
		  pos[ebN] = ( deltaU[ebN] > 0.0) ?  deltaU[ebN] : 0.0;
		  neg[ebN] = (-deltaU[ebN] > 0.0) ? -deltaU[ebN] : 0.0; 
		  pos_eN += pos[ebN];
		  neg_eN += neg[ebN];
		}
	      thp = (1.0 < neg_eN/pos_eN) ? 1.0 : neg_eN/pos_eN;
	      thm = (1.0 < pos_eN/neg_eN) ? 1.0 : pos_eN/neg_eN;
	      for (ebN=0; ebN  < nElementBoundaries_element; ebN++)
		{
		  tdeltaU[ebN] = thp*pos[ebN] - thm*neg[ebN];
		  tag_eN[ebN] = 0;
		}
	    }/*extra limit for mass bal*/
	}/*not at boundary*/
      /*get limited values at midpoints of element boundaries*/
      /*based on Crouzeix-Raviart type representation with limited slopes
	so phi_{i}(\bar{x^{j}) = \delta_ij */
      uMlim[0] = uBar + tdeltaU[0];
      uMlim[1] = uBar + tdeltaU[1];
      uMlim[2] = uBar + tdeltaU[2];
	  
      /*convert back to Lagrange representation for output*/
      Uout[l2g[eN*nDOF_element + 0]] = uMlim[1]+uMlim[2]-uMlim[0];
      Uout[l2g[eN*nDOF_element + 1]] = uMlim[2]+uMlim[0]-uMlim[1];
      Uout[l2g[eN*nDOF_element + 2]] = uMlim[0]+uMlim[1]-uMlim[2];
      tag[eN] = tag_eN[0] + tag_eN[1] + tag_eN[2] == 3;
      /*mwf debug*/
      uavgOut = (Uout[l2g[eN*nDOF_element + 0]]+Uout[l2g[eN*nDOF_element + 1]]+Uout[l2g[eN*nDOF_element + 2]])*oneThird;
      /*
      if (fabs(uavgOut-uavgIn) > 1.0e-7)
	printf("problem CockburnLimit eN=%d uavgIn=%g uavgOut=%g Uin=[%g,%g,%g] Uout=[%g,%g,%g]\n",
	       eN,uavgIn,uavgOut,
	       Uin[l2g[eN*nDOF_element + 0]],Uin[l2g[eN*nDOF_element + 1]],Uin[l2g[eN*nDOF_element + 2]],
	       Uout[l2g[eN*nDOF_element + 0]],Uout[l2g[eN*nDOF_element + 1]],Uout[l2g[eN*nDOF_element + 2]]);
      */
    }/*eN*/
}
void computeElementAveragesP1Lagrange(proteus_index_t nElements_global,
				      int nDOF_element,
				      const proteus_index_t * l2g,
				      const double * Uin,
				      double * elementAverages)
{
  proteus_index_t I; int eN, i;
  const double denom = 1.0/nDOF_element;
  for (eN = 0; eN < nElements_global; eN++)
    {
      elementAverages[eN] = 0.0;
      for (i=0; i < nDOF_element; i++)
	{
	  I = l2g[eN*nDOF_element+i];
	  elementAverages[eN] += Uin[I];
	}
      elementAverages[eN] *= denom;
    }
}
/*
  try simple sorting of A, but also keep track of sorted order
  for other quantity based on A
*/
void shortSortDescending(double *A, int * order, int len)
{
  register int i,j,itmp;
  register double tmp;
  for (i=0; i < len; i++)
    {
      /*order[i] = i;*/
      for (j = i+1; j < len; j++)
	{
	  if (A[j] > A[i])
	    {
	      tmp = A[i]; A[i]= A[j];  A[j]= tmp;
	      itmp= order[i]; order[i] = order[j]; order[j] = itmp;
	    }
	}
    }
}
void applyDurlofskyDGlimiterP1Lagrange2d(int killExtrema,
					 int allowMinWithUndershoot,
					 proteus_index_t nElements_global,
					 int nElementBoundaries_element,
					 int nNodes_element,
					 int nSpace,
					 int nDOF_element,
					 const proteus_index_t * elementNeighborsArray,
					 const proteus_index_t * elementBoundariesArray,
					 const proteus_index_t * elementNodesArray,
					 const double * nodeArray,
					 const double * elementBarycentersArray,
					 const double * elementBoundaryBarycentersArray,
					 const double * elementNeighborShapeGradients,
					 const proteus_index_t * l2g,
					 const double * grad_v0,
					 double * elementAverages,
					 int * tag,
					 double * Uin,
					 double * Uout)
{
  /*this should be Durlofsky's original version with Phi limiter (not the min one)*/
  proteus_index_t nN_global, ebN_global, eN_ebN, eN_ebN_1; int eN, ebN, ebN_1, onBoundary, isExtremum;
  int dUdescendingOrder[4];
  double normDU[4],dU[4][2],max_ebN,min_ebN,uM;
  /*mwf debug*/
  double normDUsave[4];
  int maxFound,minFound,islot,i,itmp,okGradient;
  int nElementBoundaries_element2 = nElementBoundaries_element*nElementBoundaries_element;
  register proteus_index_t DOF0, DOF1, DOF2;
  register double dUlim0,dUlim1;
  register double uBar,uBar_ebN,uBar_ebN_1;

  const double otol =  1.0e-5;

  computeElementAveragesP1Lagrange(nElements_global,nDOF_element,l2g,Uin,elementAverages);

  for (eN = 0; eN < nElements_global; eN++)
    {
      /*current element*/
      uBar = elementAverages[eN];
      DOF0 = l2g[eN*nDOF_element+0]; DOF1 = l2g[eN*nDOF_element+1]; DOF2 = l2g[eN*nDOF_element+2]; 
      onBoundary = 
	elementNeighborsArray[eN*nElementBoundaries_element + 0] < 0 ||
	elementNeighborsArray[eN*nElementBoundaries_element + 1] < 0 ||
	elementNeighborsArray[eN*nElementBoundaries_element + 2] < 0;
      if (onBoundary > 0)
	{
	  dUlim0 = 0.0; dUlim1 = 0.0;
	  tag[eN]  = 0;
	}
      else
	{
	  /*check to see if this is a local extremum*/
	  maxFound = 0; minFound = 0;
	  for (ebN = 0; ebN < nElementBoundaries_element; ebN++)
	    {
	      eN_ebN        = elementNeighborsArray[eN*nElementBoundaries_element + ebN];
	      uBar_ebN      = elementAverages[eN_ebN];
	      maxFound = uBar_ebN > uBar ? 1 : maxFound;
	      minFound = uBar_ebN < uBar ? 1 : minFound;
	    }
	  isExtremum = (maxFound*minFound == 0);
	  if (isExtremum && killExtrema > 0)
	    {
	      /*mwf debug
		printf("Durlofsky extremum found eN=%d uBar= %g uBar_ebN[%g,%g,%g]\n",
		eN,uBar,uBar_ebN[0],uBar_ebN[1],uBar_ebN[2]);
	      */
	      dUlim0 = 0.0; dUlim1 = 0.0;
	      tag[eN]  = 0;
	    }
	  else
	    {
	      /*by default take zero slope*/
	      dUlim0 = 0.0; dUlim1 = 0.0;
	      dU[0][0] = 
		grad_v0[eN*1*nDOF_element*nSpace + 0*nDOF_element*nSpace + 0*nSpace + 0]*Uin[DOF0]+
		grad_v0[eN*1*nDOF_element*nSpace + 0*nDOF_element*nSpace + 1*nSpace + 0]*Uin[DOF1]+
		grad_v0[eN*1*nDOF_element*nSpace + 0*nDOF_element*nSpace + 2*nSpace + 0]*Uin[DOF2];
	      dU[0][1] = 
		grad_v0[eN*1*nDOF_element*nSpace + 0*nDOF_element*nSpace + 0*nSpace + 1]*Uin[DOF0]+
		grad_v0[eN*1*nDOF_element*nSpace + 0*nDOF_element*nSpace + 1*nSpace + 1]*Uin[DOF1]+
		grad_v0[eN*1*nDOF_element*nSpace + 0*nDOF_element*nSpace + 2*nSpace + 1]*Uin[DOF2];
	      normDU[0] = sqrt(dU[0][0]*dU[0][0] + dU[0][1]*dU[0][1]);
	      normDU[1] = 0.0; normDU[2] = 0.0; normDU[3] = 0.0;
	      /*loop through neighbor simplexes and compute local interpolant gradients*/
	      for (ebN = 0; ebN < nElementBoundaries_element; ebN++)
		{
		  ebN_1   = (ebN+1) % nElementBoundaries_element;
		  /*two neighbors for this simplex
		    local numbering is 
		    0 <--> this element
                    1 <--> ebN neighbor
		    2 <--> ebN+1 neighbor
		   */
		  eN_ebN    = elementNeighborsArray[eN*nElementBoundaries_element + ebN];
		  eN_ebN_1  = elementNeighborsArray[eN*nElementBoundaries_element + ebN_1];
		  uBar_ebN  = elementAverages[eN_ebN];
		  uBar_ebN_1= elementAverages[eN_ebN_1];

		  /*local number zero is always this element*/
		  dU[ebN+1][0] = 0.0; dU[ebN+1][1] = 0.0;
		  dU[ebN+1][0]= 
		    uBar*elementNeighborShapeGradients[eN*nElementBoundaries_element2*nSpace + 
						       ebN*nElementBoundaries_element*nSpace +
						       0*nSpace + 0]
		    +
		    uBar_ebN*elementNeighborShapeGradients[eN*nElementBoundaries_element2*nSpace + 
							   ebN*nElementBoundaries_element*nSpace +
							   1*nSpace + 0]
		    +
		    uBar_ebN_1*elementNeighborShapeGradients[eN*nElementBoundaries_element2*nSpace + 
							     ebN*nElementBoundaries_element*nSpace +
							     2*nSpace + 0];
		  
		  dU[ebN+1][1]= 
		    uBar*elementNeighborShapeGradients[eN*nElementBoundaries_element2*nSpace + 
						       ebN*nElementBoundaries_element*nSpace +
						       0*nSpace + 1]
		    +
		    uBar_ebN*elementNeighborShapeGradients[eN*nElementBoundaries_element2*nSpace + 
							   ebN*nElementBoundaries_element*nSpace +
							   1*nSpace + 1]
		    +
		    uBar_ebN_1*elementNeighborShapeGradients[eN*nElementBoundaries_element2*nSpace + 
							     ebN*nElementBoundaries_element*nSpace +
							     2*nSpace + 1];
		  normDU[ebN+1] = sqrt(dU[ebN+1][0]*dU[ebN+1][0] + dU[ebN+1][1]*dU[ebN+1][1]);
		}/*ebN*/
	      /*now sort the gradients from largest to smallest*/
	      dUdescendingOrder[0] = 0; dUdescendingOrder[1] = 1; dUdescendingOrder[2]=2; dUdescendingOrder[3]=3;
	      normDUsave[0] = normDU[0]; normDUsave[1]=normDU[1]; normDUsave[2]=normDU[2]; normDUsave[3]=normDU[3];
	      shortSortDescending(normDU,dUdescendingOrder,4);
	      /*mwf debug check ordering
	      for (i = 0; i < nElementBoundaries_element; i++)
		{
		  if (normDU[i+1] > normDU[i])
		    {
		      printf("\nPROBLEM Durlofsky out of order eN=%d normDU[%d]=%g > normDU[%d] =%g \n",
			     eN,i+1,normDU[i+1],
			     i,normDU[i]);
		      printf("normDU=[%g,%g,%g,%g] dUdescendingOrder=[%d,%d,%d,%d]\n",
			     normDU[0],normDU[1],normDU[2],normDU[3],
			     dUdescendingOrder[0],dUdescendingOrder[1],dUdescendingOrder[2],dUdescendingOrder[3]);
		      exit(1);
		    }
		  if (fabs(normDU[i]-normDUsave[dUdescendingOrder[i]]) > 1.0e-8)
		    {
		      printf("\nPROBLEM Durlofsky order wrong eN=%d normDU[%d]=%g normDUsave[%d] =%g \n",
			     eN,i,normDU[i],dUdescendingOrder[i],normDUsave[dUdescendingOrder[i]]);
		      printf("normDU=[%g,%g,%g,%g] normDUsave=[%g,%g,%g,%g] dUdescendingOrder=[%d,%d,%d,%d]\n",
			     normDU[0],normDU[1],normDU[2],normDU[3],
			     normDUsave[0],normDUsave[1],normDUsave[2],normDUsave[3],
			     dUdescendingOrder[0],dUdescendingOrder[1],dUdescendingOrder[2],dUdescendingOrder[3]);
		      exit(1);

		    }
		  
		}
	      mwf debug end */
	      /*now start checking for overshoot, starting with largest du*/
	      okGradient = 0; islot = 0;
	      /*use minimum slope if undershoot/overshoot detected for others*/
	      while (okGradient == 0 && islot < nElementBoundaries_element)
		{
		  itmp = dUdescendingOrder[islot];
		  /*start with ok*/
		  okGradient = 1;
		  for (ebN = 0; ebN < nElementBoundaries_element; ebN++)
		    {
		      eN_ebN     = elementNeighborsArray[eN*nElementBoundaries_element + ebN];
		      uBar_ebN   = elementAverages[eN_ebN];
		      ebN_global = elementBoundariesArray[eN*nElementBoundaries_element + ebN];
		      uM = uBar 
			+ 
			dU[itmp][0]*(elementBoundaryBarycentersArray[ebN_global*3 + 0]-
				     elementBarycentersArray[eN*3 + 0])
			+
			dU[itmp][1]*(elementBoundaryBarycentersArray[ebN_global*3 + 1]-
				     elementBarycentersArray[eN*3 + 1]);
		      max_ebN = uBar > uBar_ebN ? uBar : uBar_ebN;
		      min_ebN = uBar < uBar_ebN ? uBar : uBar_ebN;
		      
		      okGradient = (okGradient > 0) && min_ebN <= uM -otol && uM <= max_ebN + otol;
		      
		    }
		  islot++;/*undo this later if okGradient==1*/
		}
	      if (okGradient == 1)
		islot--;
	      if (okGradient || allowMinWithUndershoot > 0)
		{
		  itmp = dUdescendingOrder[islot];
		  dUlim0 = dU[itmp][0];
		  dUlim1 = dU[itmp][1];
		  tag[eN] = islot == 0 ? 1 : 0;
		  /*mwf debug 
		    printf("Durlofsky eN=%d okGradient islot=%d dUlim[0]=%g dUlim[1]=%g \n",
		    eN,islot,dUlim[0],dUlim[1]);
		  */
		}
	      
	    }/*not an extremum*/
	}/*in interior*/
      /*interpolate values to nodes assuming correspondence with local dofs*/
      nN_global = elementNodesArray[eN*nNodes_element+0];
      Uout[DOF0] = uBar 
	+ 
	dUlim0*(nodeArray[nN_global*3 + 0] - elementBarycentersArray[eN*3 + 0])
	+
	dUlim1*(nodeArray[nN_global*3 + 1] - elementBarycentersArray[eN*3 + 1]);
      nN_global = elementNodesArray[eN*nNodes_element+1];
      Uout[DOF1] = uBar 
	+ 
	dUlim0*(nodeArray[nN_global*3 + 0] - elementBarycentersArray[eN*3 + 0])
	+
	dUlim1*(nodeArray[nN_global*3 + 1] - elementBarycentersArray[eN*3 + 1]);
      nN_global = elementNodesArray[eN*nNodes_element+2];
      Uout[DOF2] = uBar 
	+ 
	dUlim0*(nodeArray[nN_global*3 + 0] - elementBarycentersArray[eN*3 + 0])
	+
	dUlim1*(nodeArray[nN_global*3 + 1] - elementBarycentersArray[eN*3 + 1]);
      
    }/*eN*/
}

void applyDurlofskyDGlimiterP1Lagrange3d(int killExtrema,
					 int allowMinWithUndershoot,
					 proteus_index_t nElements_global,
					 int nElementBoundaries_element,
					 int nNodes_element,
					 int nSpace,
					 int nDOF_element,
					 const proteus_index_t * elementNeighborsArray,
					 const proteus_index_t * elementBoundariesArray,
					 const proteus_index_t * elementNodesArray,
					 const double * nodeArray,
					 const double * elementBarycentersArray,
					 const double * elementBoundaryBarycentersArray,
					 const double * elementNeighborShapeGradients,
					 const proteus_index_t * l2g,
					 const double * grad_v0,
					 double * elementAverages,
					 int * tag,
					 double * Uin,
					 double * Uout)
{
  /*try 3d version of  Durlofsky's original version with Phi limiter (not the min one)*/
  proteus_index_t nN_global, ebN_global, eN_ebN, eN_ebN_1, eN_ebN_2; int eN, ebN, ebN_1, ebN_2, onBoundary, isExtremum;
  int dUdescendingOrder[5];
  double normDU[5],dU[5][3],max_ebN,min_ebN,uM;
  /*mwf for debugging*/
  double normDUsave[5];
  int maxFound,minFound,islot,i,itmp,okGradient;
  int nElementBoundaries_element2 = nElementBoundaries_element*nElementBoundaries_element;
  register proteus_index_t DOF0, DOF1, DOF2, DOF3;
  register double dUlim0,dUlim1,dUlim2;
  register double uBar,uBar_ebN,uBar_ebN_1,uBar_ebN_2;

  const double otol =  1.0e-5;
  computeElementAveragesP1Lagrange(nElements_global,nDOF_element,l2g,Uin,elementAverages);

  for (eN = 0; eN < nElements_global; eN++)
    {
      /*current element*/
      uBar = elementAverages[eN];
      DOF0 = l2g[eN*nDOF_element+0]; DOF1 = l2g[eN*nDOF_element+1]; 
      DOF2 = l2g[eN*nDOF_element+2]; DOF3 = l2g[eN*nDOF_element+3]; 
      onBoundary = 
	elementNeighborsArray[eN*nElementBoundaries_element + 0] < 0 ||
	elementNeighborsArray[eN*nElementBoundaries_element + 1] < 0 ||
	elementNeighborsArray[eN*nElementBoundaries_element + 2] < 0 ||
	elementNeighborsArray[eN*nElementBoundaries_element + 3] < 0;
      if (onBoundary > 0)
	{
	  dUlim0 = 0.0; dUlim1 = 0.0; dUlim2 = 0.0;
	  tag[eN]  = 0;
	}
      else
	{
	  /*check to see if this is a local extremum*/
	  maxFound = 0; minFound = 0;
	  for (ebN = 0; ebN < nElementBoundaries_element; ebN++)
	    {
	      eN_ebN        = elementNeighborsArray[eN*nElementBoundaries_element + ebN];
	      uBar_ebN      = elementAverages[eN_ebN];
	      maxFound = uBar_ebN > uBar ? 1 : maxFound;
	      minFound = uBar_ebN < uBar ? 1 : minFound;
	    }
	  isExtremum = (maxFound*minFound == 0);
	  if (isExtremum && killExtrema > 0)
	    {
	      /*mwf debug
		printf("Durlofsky extremum found eN=%d uBar= %g uBar_ebN[%g,%g,%g]\n",
		eN,uBar,uBar_ebN[0],uBar_ebN[1],uBar_ebN[2]);
	      */
	      dUlim0 = 0.0; dUlim1 = 0.0; dUlim2 = 0.0;
	      tag[eN]  = 0;
	    }
	  else
	    {
	      /*by default take zero slope*/
	      dUlim0 = 0.0; dUlim1 = 0.0; dUlim2 = 0.0;
	      dU[0][0] = 
		grad_v0[eN*1*nDOF_element*nSpace + 0*nDOF_element*nSpace + 0*nSpace + 0]*Uin[DOF0]+
		grad_v0[eN*1*nDOF_element*nSpace + 0*nDOF_element*nSpace + 1*nSpace + 0]*Uin[DOF1]+
		grad_v0[eN*1*nDOF_element*nSpace + 0*nDOF_element*nSpace + 2*nSpace + 0]*Uin[DOF2]+
		grad_v0[eN*1*nDOF_element*nSpace + 0*nDOF_element*nSpace + 3*nSpace + 0]*Uin[DOF3];
	      dU[0][1] = 
		grad_v0[eN*1*nDOF_element*nSpace + 0*nDOF_element*nSpace + 0*nSpace + 1]*Uin[DOF0]+
		grad_v0[eN*1*nDOF_element*nSpace + 0*nDOF_element*nSpace + 1*nSpace + 1]*Uin[DOF1]+
		grad_v0[eN*1*nDOF_element*nSpace + 0*nDOF_element*nSpace + 2*nSpace + 1]*Uin[DOF2]+
		grad_v0[eN*1*nDOF_element*nSpace + 0*nDOF_element*nSpace + 3*nSpace + 1]*Uin[DOF3];
	      dU[0][2] = 
		grad_v0[eN*1*nDOF_element*nSpace + 0*nDOF_element*nSpace + 0*nSpace + 2]*Uin[DOF0]+
		grad_v0[eN*1*nDOF_element*nSpace + 0*nDOF_element*nSpace + 1*nSpace + 2]*Uin[DOF1]+
		grad_v0[eN*1*nDOF_element*nSpace + 0*nDOF_element*nSpace + 2*nSpace + 2]*Uin[DOF2]+
		grad_v0[eN*1*nDOF_element*nSpace + 0*nDOF_element*nSpace + 3*nSpace + 2]*Uin[DOF3];
	      normDU[0] = sqrt(dU[0][0]*dU[0][0] + dU[0][1]*dU[0][1] + dU[0][2]*dU[0][2]);
	      normDU[1] = 0.0; normDU[2] = 0.0; normDU[3] = 0.0; normDU[4] = 0.0;
	      /*loop through neighbor simplexes and compute local interpolant gradients*/
	      for (ebN = 0; ebN < nElementBoundaries_element; ebN++)
		{
		  ebN_1   = (ebN+1) % nElementBoundaries_element;
		  ebN_2   = (ebN+2) % nElementBoundaries_element;
		  /*two neighbors for this simplex
		    local numbering is 
		    0 <--> this element
                    1 <--> ebN neighbor
		    2 <--> ebN+1 neighbor
		    3 <--> ebN+2 neighbor
		   */
		  eN_ebN    = elementNeighborsArray[eN*nElementBoundaries_element + ebN];
		  eN_ebN_1  = elementNeighborsArray[eN*nElementBoundaries_element + ebN_1];
		  eN_ebN_2  = elementNeighborsArray[eN*nElementBoundaries_element + ebN_2];
		  uBar_ebN  = elementAverages[eN_ebN];
		  uBar_ebN_1= elementAverages[eN_ebN_1];
		  uBar_ebN_2= elementAverages[eN_ebN_2];

		  /*local number zero is always this element*/
		  dU[ebN+1][0] = 0.0; dU[ebN+1][1] = 0.0; dU[ebN+1][2] = 0.0;
		  dU[ebN+1][0]= 
		    uBar*elementNeighborShapeGradients[eN*nElementBoundaries_element2*nSpace + 
						       ebN*nElementBoundaries_element*nSpace +
						       0*nSpace + 0]
		    +
		    uBar_ebN*elementNeighborShapeGradients[eN*nElementBoundaries_element2*nSpace + 
							   ebN*nElementBoundaries_element*nSpace +
							   1*nSpace + 0]
		    +
		    uBar_ebN_1*elementNeighborShapeGradients[eN*nElementBoundaries_element2*nSpace + 
							     ebN*nElementBoundaries_element*nSpace +
							     2*nSpace + 0]
		    +
		    uBar_ebN_2*elementNeighborShapeGradients[eN*nElementBoundaries_element2*nSpace + 
							     ebN*nElementBoundaries_element*nSpace +
							     3*nSpace + 0];
		  
		  dU[ebN+1][1]= 
		    uBar*elementNeighborShapeGradients[eN*nElementBoundaries_element2*nSpace + 
						       ebN*nElementBoundaries_element*nSpace +
						       0*nSpace + 1]
		    +
		    uBar_ebN*elementNeighborShapeGradients[eN*nElementBoundaries_element2*nSpace + 
							   ebN*nElementBoundaries_element*nSpace +
							   1*nSpace + 1]
		    +
		    uBar_ebN_1*elementNeighborShapeGradients[eN*nElementBoundaries_element2*nSpace + 
							     ebN*nElementBoundaries_element*nSpace +
							     2*nSpace + 1]
		    +
		    uBar_ebN_1*elementNeighborShapeGradients[eN*nElementBoundaries_element2*nSpace + 
							     ebN*nElementBoundaries_element*nSpace +
							     3*nSpace + 1];

		  dU[ebN+1][2]= 
		    uBar*elementNeighborShapeGradients[eN*nElementBoundaries_element2*nSpace + 
						       ebN*nElementBoundaries_element*nSpace +
						       0*nSpace + 2]
		    +
		    uBar_ebN*elementNeighborShapeGradients[eN*nElementBoundaries_element2*nSpace + 
							   ebN*nElementBoundaries_element*nSpace +
							   1*nSpace + 2]
		    +
		    uBar_ebN_1*elementNeighborShapeGradients[eN*nElementBoundaries_element2*nSpace + 
							     ebN*nElementBoundaries_element*nSpace +
							     2*nSpace + 2]
		    +
		    uBar_ebN_1*elementNeighborShapeGradients[eN*nElementBoundaries_element2*nSpace + 
							     ebN*nElementBoundaries_element*nSpace +
							     3*nSpace + 2];

		  normDU[ebN+1] = sqrt(dU[ebN+1][0]*dU[ebN+1][0] + dU[ebN+1][1]*dU[ebN+1][1] + dU[ebN+1][2]*dU[ebN+1][2]);
		}/*ebN*/
	      /*now sort the gradients from largest to smallest*/
	      dUdescendingOrder[0] = 0; dUdescendingOrder[1] = 1; dUdescendingOrder[2]=2; 
	      dUdescendingOrder[3]=3;  dUdescendingOrder[4] = 4;
	      /*save for debugging*/
	      normDUsave[0] = normDU[0]; normDUsave[1]=normDU[1]; normDUsave[2]=normDU[2]; normDUsave[3]=normDU[3];
	      normDUsave[4] = normDU[4];
	      shortSortDescending(normDU,dUdescendingOrder,5);
	      /*mwf debug check ordering
	      for (i = 0; i < nElementBoundaries_element; i++)
		{
		  if (normDU[i+1] > normDU[i])
		    {
		      printf("\nPROBLEM Durlofsky 3d out of order eN=%d normDU[%d]=%g > normDU[%d] =%g \n",
			     eN,i+1,normDU[i+1],
			     i,normDU[i]);
		      printf("normDU=[%g,%g,%g,%g,%g] dUdescendingOrder=[%d,%d,%d,%d,%d]\n",
			     normDU[0],normDU[1],normDU[2],normDU[3],normDU[4],
			     dUdescendingOrder[0],dUdescendingOrder[1],dUdescendingOrder[2],
			     dUdescendingOrder[3],dUdescendingOrder[4]);
		      exit(1);
		    }
		  if (fabs(normDU[i]-normDUsave[dUdescendingOrder[i]]) > 1.0e-8)
		    {
		      printf("\nPROBLEM Durlofsky order wrong eN=%d normDU[%d]=%g normDUsave[%d] =%g \n",
			     eN,i,normDU[i],dUdescendingOrder[i],normDUsave[dUdescendingOrder[i]]);
		      printf("normDU=[%g,%g,%g,%g,%g] normDUsave=[%g,%g,%g,%g,%g] dUdescendingOrder=[%d,%d,%d,%d,%d]\n",
			     normDU[0],normDU[1],normDU[2],normDU[3],normDU[4],
			     normDUsave[0],normDUsave[1],normDUsave[2],normDUsave[3],normDUsave[4],
			     dUdescendingOrder[0],dUdescendingOrder[1],dUdescendingOrder[2],dUdescendingOrder[3],
			     dUdescendingOrder[4]);
		      exit(1);

		    }
		}
	       mwf debug end */
	      /*now start checking for overshoot, starting with largest du*/
	      okGradient = 0; islot = 0;
	      /*use minimum slope if undershoot/overshoot detected for others*/
	      while (okGradient == 0 && islot < nElementBoundaries_element)
		{
		  itmp = dUdescendingOrder[islot];
		  /*start with ok*/
		  okGradient = 1;
		  for (ebN = 0; ebN < nElementBoundaries_element; ebN++)
		    {
		      eN_ebN     = elementNeighborsArray[eN*nElementBoundaries_element + ebN];
		      uBar_ebN   = elementAverages[eN_ebN];
		      ebN_global = elementBoundariesArray[eN*nElementBoundaries_element + ebN];
		      uM = uBar 
			+ 
			dU[itmp][0]*(elementBoundaryBarycentersArray[ebN_global*3 + 0]-
				     elementBarycentersArray[eN*3 + 0])
			+
			dU[itmp][1]*(elementBoundaryBarycentersArray[ebN_global*3 + 1]-
				     elementBarycentersArray[eN*3 + 1])
			+
			dU[itmp][2]*(elementBoundaryBarycentersArray[ebN_global*3 + 2]-
				     elementBarycentersArray[eN*3 + 2]);

		      max_ebN = uBar > uBar_ebN ? uBar : uBar_ebN;
		      min_ebN = uBar < uBar_ebN ? uBar : uBar_ebN;
		      
		      okGradient = (okGradient > 0) && min_ebN <= uM -otol && uM <= max_ebN + otol;
		      
		    }
		  islot++;/*undo this later if okGradient==1*/
		}
	      if (okGradient == 1)
		islot--;
	      if (okGradient || allowMinWithUndershoot > 0)
		{
		  itmp = dUdescendingOrder[islot];
		  dUlim0 = dU[itmp][0];
		  dUlim1 = dU[itmp][1];
		  dUlim2 = dU[itmp][2];
		  tag[eN] = islot == 0 ? 1 : 0;
		  /*mwf debug 
		    printf("Durlofsky eN=%d okGradient islot=%d dUlim[0]=%g dUlim[1]=%g dUlim[2]=%g\n",
		    eN,islot,dUlim0,dUlim1,dUlim2);
		  */
		}
	    
	      
	    }/*not an extremum*/
	}/*in interior*/
      /*interpolate values to nodes assuming correspondence with local dofs*/
      nN_global = elementNodesArray[eN*nNodes_element+0];
      Uout[DOF0] = uBar 
	+ 
	dUlim0*(nodeArray[nN_global*3 + 0] - elementBarycentersArray[eN*3 + 0])
	+
	dUlim1*(nodeArray[nN_global*3 + 1] - elementBarycentersArray[eN*3 + 1])
	+
	dUlim2*(nodeArray[nN_global*3 + 2] - elementBarycentersArray[eN*3 + 2]);
      nN_global = elementNodesArray[eN*nNodes_element+1];
      Uout[DOF1] = uBar 
	+ 
	dUlim0*(nodeArray[nN_global*3 + 0] - elementBarycentersArray[eN*3 + 0])
	+
	dUlim1*(nodeArray[nN_global*3 + 1] - elementBarycentersArray[eN*3 + 1])
	+
	dUlim2*(nodeArray[nN_global*3 + 2] - elementBarycentersArray[eN*3 + 2]);
      nN_global = elementNodesArray[eN*nNodes_element+2];
      Uout[DOF2] = uBar 
	+ 
	dUlim0*(nodeArray[nN_global*3 + 0] - elementBarycentersArray[eN*3 + 0])
	+
	dUlim1*(nodeArray[nN_global*3 + 1] - elementBarycentersArray[eN*3 + 1])
	+
	dUlim2*(nodeArray[nN_global*3 + 2] - elementBarycentersArray[eN*3 + 2]);
      nN_global = elementNodesArray[eN*nNodes_element+3];
      Uout[DOF3] = uBar 
	+ 
	dUlim0*(nodeArray[nN_global*3 + 0] - elementBarycentersArray[eN*3 + 0])
	+
	dUlim1*(nodeArray[nN_global*3 + 1] - elementBarycentersArray[eN*3 + 1])
	+
	dUlim2*(nodeArray[nN_global*3 + 2] - elementBarycentersArray[eN*3 + 2]);
      
    }/*eN*/
}

void applyDurlofskyDGlimiterP1Lagrange2d_withVacuumTol(int killExtrema,
						       int allowMinWithUndershoot,
						       int enforcePositivity,
						       double vacuumTol,
						       proteus_index_t nElements_global,
						       int nElementBoundaries_element,
						       int nNodes_element,
						       int nSpace,
						       int nDOF_element,
						       const proteus_index_t * elementNeighborsArray,
						       const proteus_index_t * elementBoundariesArray,
						       const proteus_index_t * elementNodesArray,
						       const double * nodeArray,
						       const double * elementBarycentersArray,
						       const double * elementBoundaryBarycentersArray,
						       const double * elementNeighborShapeGradients,
						       const proteus_index_t * l2g,
						       const double * grad_v0,
						       double * elementAverages,
						       int * tag,
						       double * Uin,
						       double * Uout)
{
  /*this should be Durlofsky's original version with Phi limiter (not the min one)*/
  proteus_index_t nN_global, ebN_global, eN_ebN, eN_ebN_1; int eN, ebN, ebN_1, onBoundary, isExtremum;
  int dUdescendingOrder[4];
  double normDU[4],dU[4][2],max_ebN,min_ebN,uM;
  /*mwf debug*/
  double normDUsave[4];
  int maxFound,minFound,islot,i,itmp,okGradient;
  int nElementBoundaries_element2 = nElementBoundaries_element*nElementBoundaries_element;
  register proteus_index_t DOF0, DOF1, DOF2;
  register double dUlim0,dUlim1;
  register double uBar,uBar_ebN,uBar_ebN_1;

  const double otol =  1.0e-5;

  computeElementAveragesP1Lagrange(nElements_global,nDOF_element,l2g,Uin,elementAverages);

  for (eN = 0; eN < nElements_global; eN++)
    {
      /*current element*/
      uBar = elementAverages[eN];
      DOF0 = l2g[eN*nDOF_element+0]; DOF1 = l2g[eN*nDOF_element+1]; DOF2 = l2g[eN*nDOF_element+2]; 
      onBoundary = 
	elementNeighborsArray[eN*nElementBoundaries_element + 0] < 0 ||
	elementNeighborsArray[eN*nElementBoundaries_element + 1] < 0 ||
	elementNeighborsArray[eN*nElementBoundaries_element + 2] < 0;
      if (onBoundary > 0)
	{
	  dUlim0 = 0.0; dUlim1 = 0.0;
	  tag[eN]  = 0;
	}
      else
	{
	  /*check to see if this is a local extremum*/
	  maxFound = 0; minFound = 0;
	  for (ebN = 0; ebN < nElementBoundaries_element; ebN++)
	    {
	      eN_ebN        = elementNeighborsArray[eN*nElementBoundaries_element + ebN];
	      uBar_ebN      = elementAverages[eN_ebN];
	      maxFound = uBar_ebN > uBar ? 1 : maxFound;
	      minFound = uBar_ebN < uBar ? 1 : minFound;
	    }
	  isExtremum = (maxFound*minFound == 0);
	  if (isExtremum && killExtrema > 0)
	    {
	      /*mwf debug
		printf("Durlofsky extremum found eN=%d uBar= %g uBar_ebN[%g,%g,%g]\n",
		eN,uBar,uBar_ebN[0],uBar_ebN[1],uBar_ebN[2]);
	      */
	      dUlim0 = 0.0; dUlim1 = 0.0;
	      tag[eN]  = 0;
	    }
	  else if (fabs(uBar) < vacuumTol)
	    {
	      dUlim0 = 0.0; dUlim1 = 0.0;
	      tag[eN]  = 0;
	    }
	  else if (enforcePositivity && uBar < vacuumTol)
	    {
	      dUlim0 = 0.0; dUlim1 = 0.0;
	      tag[eN]  = 0;
	    }
	  else
	    {
	      /*by default take zero slope*/
	      dUlim0 = 0.0; dUlim1 = 0.0;
	      dU[0][0] = 
		grad_v0[eN*1*nDOF_element*nSpace + 0*nDOF_element*nSpace + 0*nSpace + 0]*Uin[DOF0]+
		grad_v0[eN*1*nDOF_element*nSpace + 0*nDOF_element*nSpace + 1*nSpace + 0]*Uin[DOF1]+
		grad_v0[eN*1*nDOF_element*nSpace + 0*nDOF_element*nSpace + 2*nSpace + 0]*Uin[DOF2];
	      dU[0][1] = 
		grad_v0[eN*1*nDOF_element*nSpace + 0*nDOF_element*nSpace + 0*nSpace + 1]*Uin[DOF0]+
		grad_v0[eN*1*nDOF_element*nSpace + 0*nDOF_element*nSpace + 1*nSpace + 1]*Uin[DOF1]+
		grad_v0[eN*1*nDOF_element*nSpace + 0*nDOF_element*nSpace + 2*nSpace + 1]*Uin[DOF2];
	      normDU[0] = sqrt(dU[0][0]*dU[0][0] + dU[0][1]*dU[0][1]);
	      normDU[1] = 0.0; normDU[2] = 0.0; normDU[3] = 0.0;
	      /*loop through neighbor simplexes and compute local interpolant gradients*/
	      for (ebN = 0; ebN < nElementBoundaries_element; ebN++)
		{
		  ebN_1   = (ebN+1) % nElementBoundaries_element;
		  /*two neighbors for this simplex
		    local numbering is 
		    0 <--> this element
                    1 <--> ebN neighbor
		    2 <--> ebN+1 neighbor
		   */
		  eN_ebN    = elementNeighborsArray[eN*nElementBoundaries_element + ebN];
		  eN_ebN_1  = elementNeighborsArray[eN*nElementBoundaries_element + ebN_1];
		  uBar_ebN  = elementAverages[eN_ebN];
		  uBar_ebN_1= elementAverages[eN_ebN_1];

		  /*local number zero is always this element*/
		  dU[ebN+1][0] = 0.0; dU[ebN+1][1] = 0.0;
		  dU[ebN+1][0]= 
		    uBar*elementNeighborShapeGradients[eN*nElementBoundaries_element2*nSpace + 
						       ebN*nElementBoundaries_element*nSpace +
						       0*nSpace + 0]
		    +
		    uBar_ebN*elementNeighborShapeGradients[eN*nElementBoundaries_element2*nSpace + 
							   ebN*nElementBoundaries_element*nSpace +
							   1*nSpace + 0]
		    +
		    uBar_ebN_1*elementNeighborShapeGradients[eN*nElementBoundaries_element2*nSpace + 
							     ebN*nElementBoundaries_element*nSpace +
							     2*nSpace + 0];
		  
		  dU[ebN+1][1]= 
		    uBar*elementNeighborShapeGradients[eN*nElementBoundaries_element2*nSpace + 
						       ebN*nElementBoundaries_element*nSpace +
						       0*nSpace + 1]
		    +
		    uBar_ebN*elementNeighborShapeGradients[eN*nElementBoundaries_element2*nSpace + 
							   ebN*nElementBoundaries_element*nSpace +
							   1*nSpace + 1]
		    +
		    uBar_ebN_1*elementNeighborShapeGradients[eN*nElementBoundaries_element2*nSpace + 
							     ebN*nElementBoundaries_element*nSpace +
							     2*nSpace + 1];
		  normDU[ebN+1] = sqrt(dU[ebN+1][0]*dU[ebN+1][0] + dU[ebN+1][1]*dU[ebN+1][1]);
		}/*ebN*/
	      /*now sort the gradients from largest to smallest*/
	      dUdescendingOrder[0] = 0; dUdescendingOrder[1] = 1; dUdescendingOrder[2]=2; dUdescendingOrder[3]=3;
	      normDUsave[0] = normDU[0]; normDUsave[1]=normDU[1]; normDUsave[2]=normDU[2]; normDUsave[3]=normDU[3];
	      shortSortDescending(normDU,dUdescendingOrder,4);
	      /*mwf debug check ordering
	      for (i = 0; i < nElementBoundaries_element; i++)
		{
		  if (normDU[i+1] > normDU[i])
		    {
		      printf("\nPROBLEM Durlofsky out of order eN=%d normDU[%d]=%g > normDU[%d] =%g \n",
			     eN,i+1,normDU[i+1],
			     i,normDU[i]);
		      printf("normDU=[%g,%g,%g,%g] dUdescendingOrder=[%d,%d,%d,%d]\n",
			     normDU[0],normDU[1],normDU[2],normDU[3],
			     dUdescendingOrder[0],dUdescendingOrder[1],dUdescendingOrder[2],dUdescendingOrder[3]);
		      exit(1);
		    }
		  if (fabs(normDU[i]-normDUsave[dUdescendingOrder[i]]) > 1.0e-8)
		    {
		      printf("\nPROBLEM Durlofsky order wrong eN=%d normDU[%d]=%g normDUsave[%d] =%g \n",
			     eN,i,normDU[i],dUdescendingOrder[i],normDUsave[dUdescendingOrder[i]]);
		      printf("normDU=[%g,%g,%g,%g] normDUsave=[%g,%g,%g,%g] dUdescendingOrder=[%d,%d,%d,%d]\n",
			     normDU[0],normDU[1],normDU[2],normDU[3],
			     normDUsave[0],normDUsave[1],normDUsave[2],normDUsave[3],
			     dUdescendingOrder[0],dUdescendingOrder[1],dUdescendingOrder[2],dUdescendingOrder[3]);
		      exit(1);

		    }
		  
		}
	      mwf debug end */
	      /*now start checking for overshoot, starting with largest du*/
	      okGradient = 0; islot = 0;
	      /*use minimum slope if undershoot/overshoot detected for others*/
	      while (okGradient == 0 && islot < nElementBoundaries_element)
		{
		  itmp = dUdescendingOrder[islot];
		  /*start with ok*/
		  okGradient = 1;
		  for (ebN = 0; ebN < nElementBoundaries_element; ebN++)
		    {
		      eN_ebN     = elementNeighborsArray[eN*nElementBoundaries_element + ebN];
		      uBar_ebN   = elementAverages[eN_ebN];
		      ebN_global = elementBoundariesArray[eN*nElementBoundaries_element + ebN];
		      uM = uBar 
			+ 
			dU[itmp][0]*(elementBoundaryBarycentersArray[ebN_global*3 + 0]-
				     elementBarycentersArray[eN*3 + 0])
			+
			dU[itmp][1]*(elementBoundaryBarycentersArray[ebN_global*3 + 1]-
				     elementBarycentersArray[eN*3 + 1]);
		      max_ebN = uBar > uBar_ebN ? uBar : uBar_ebN;
		      min_ebN = uBar < uBar_ebN ? uBar : uBar_ebN;
		      
		      okGradient = (okGradient > 0) && min_ebN <= uM -otol && uM <= max_ebN + otol &&
			fabs(min_ebN) >= vacuumTol && fabs(max_ebN) >= vacuumTol;
		      if (enforcePositivity)
			{
			  okGradient = (okGradient > 0) && min_ebN >= vacuumTol && max_ebN >= vacuumTol;
			}
		      
		    }
		  islot++;/*undo this later if okGradient==1*/
		}
	      if (okGradient == 1)
		islot--;
	      if ((okGradient || allowMinWithUndershoot > 0) && uBar > vacuumTol)
		{
		  itmp = dUdescendingOrder[islot];
		  dUlim0 = dU[itmp][0];
		  dUlim1 = dU[itmp][1];
		  tag[eN] = islot == 0 ? 1 : 0;
		  /*mwf debug 
		    printf("Durlofsky eN=%d okGradient islot=%d dUlim[0]=%g dUlim[1]=%g \n",
		    eN,islot,dUlim[0],dUlim[1]);
		  */
		}
	      
	    }/*not an extremum*/
	}/*in interior*/
      /*interpolate values to nodes assuming correspondence with local dofs*/
      nN_global = elementNodesArray[eN*nNodes_element+0];
      Uout[DOF0] = uBar 
	+ 
	dUlim0*(nodeArray[nN_global*3 + 0] - elementBarycentersArray[eN*3 + 0])
	+
	dUlim1*(nodeArray[nN_global*3 + 1] - elementBarycentersArray[eN*3 + 1]);
      nN_global = elementNodesArray[eN*nNodes_element+1];
      Uout[DOF1] = uBar 
	+ 
	dUlim0*(nodeArray[nN_global*3 + 0] - elementBarycentersArray[eN*3 + 0])
	+
	dUlim1*(nodeArray[nN_global*3 + 1] - elementBarycentersArray[eN*3 + 1]);
      nN_global = elementNodesArray[eN*nNodes_element+2];
      Uout[DOF2] = uBar 
	+ 
	dUlim0*(nodeArray[nN_global*3 + 0] - elementBarycentersArray[eN*3 + 0])
	+
	dUlim1*(nodeArray[nN_global*3 + 1] - elementBarycentersArray[eN*3 + 1]);
      
    }/*eN*/
}

/** @} */

