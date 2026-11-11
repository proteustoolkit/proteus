#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <strings.h>
#include <assert.h>
#include "postprocessing.h"
#include PROTEUS_LAPACK_H
/** \file postprocessing.c
    \ingroup postprocessing
    @{
*/
void invertLocal(int nSpace,double A[3][3], double AI[3][3])
{
  double detA,detAinv;
  if (nSpace == 1)
    {
      assert(fabs(A[0][0]) > 0.0);
      AI[0][0] = 1.0/A[0][0];
    }
  else if (nSpace == 2)
    {
      detA = A[0][0]*A[1][1]-A[0][1]*A[1][0];
      assert(fabs(detA) > 0.0);
      detAinv = 1.0/detA;
      AI[0][0] = detAinv*A[1][1]; AI[0][1] =-detAinv*A[0][1];
      AI[1][0] =-detAinv*A[1][0]; AI[1][1] = detAinv*A[0][0];
    }
  else
    {
      detA = 
	A[0][0]*(A[1][1]*A[2][2]-A[2][1]*A[1][2])-
	A[0][1]*(A[1][0]*A[2][2]-A[2][0]*A[1][2])+
	A[0][2]*(A[1][0]*A[2][1]-A[2][0]*A[1][1]);
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
    }
}


void postProcessRT0velocityFromP1nc(proteus_index_t nElements_global,
				    int nQuadraturePoints_element,
				    int nDOF_test_element,
				    int nElementBoundaries_element,
				    int nQuadraturePoints_elementBoundary,
				    int nSpace,
				    int * nFreeDOF_element,
				    int * freeLocal_element,
				    double * detJ,
				    double * sqrt_det_g,
				    double * n,
				    double * elementBarycenters,
				    double * quad_a,
				    double * quad_f,
				    double * w_dV_r,
				    double * w_dV_m,
				    double * u,
				    double * gradu,
				    double * a, 
				    double * f, 
				    double * r, 
				    double * mt,
				    double * rt0vdofs)
{
  /***********************************************************************
    combine example in Chen '94 paper and Chou and Tang '00 paper to
    take scalar finite element function, assuming it's from NC_Affine
    ...  space and compute an RT0 velocity. 

    This version includes a piecewise constant 'gravity'/advection
    term, f as well as mass and source terms that may not be computed
    using L_2 projection. It also likely only holds for diagonal tensor A.

    This means that we lose the correspondence with RT_0 mixed finite
    element approximation, but should still be able to get a velocity
    in RT_0 that's locally conservative (and has continuous normal
    fluxes).  Chou and Tang show that their approach should still
    converge to the correct solution O(h) I believe.

    Velocity (Darcy flux) definition is
  
     q =  -A\grad u + \vec f 

    where \vec f wouldy be \rho A\vec g, A is the (diagonal) 
    conductivity \vec g is gravity and \rho is a density.
    
    A and f need to be evaluated as the 'averages' aka L_2 projections
    on element-wise constants. Note that in the P^1_nc weak formulation, we get

     (A_h\grad u_h,\grad w_h)_E = (\bar{A}\grad u_h,\grad w_h)_E 
     (f_h,\grad w_h)_E          = (\bar{f},grad w_h)_E

    for a diagonal A, since \grad u_h and grad w_h are constant on E.

    Weak formulation for P^1_nc should be

    (m_{h,t},w_h) + \sum_{E}(A_h \grad u_h,\grad w_h) - \sum_{E}(f_h,\grad w_h)_E + 
       (r_h,w_h) = (s_h,w_h) - <q^b,w_h>_{\Gamma_N}


    Where, q^b = total flux on Neumann boundary

    To represent the RT0 velocity, we use
    
      \vec q_h = \vec a_E + b_E\vec x 

    which turns out to be on triangle E

      \vec q_h = -\bar{A}_h\grad u_h + \bar{\vec f}_h + 
             (\bar{s}_E-\bar{r}_{E}-\bar{m}_{t})/d(\vec x - \vec x_{E}) + \vec c_{E}

    The scalar b_E is determined exclusively by the mass-conservation constraint:

      b_E = (\bar{s}_E-\bar{r}_{E}-\bar{m}_{t})/d

    and the constant \vec a_{E} comes about through normal flux continuity condition 

        \vec a_T = -\bar{A}_{h,E}\grad u_h + \bar{\vec f} - b_E \vec x_{E} + \vec c_E
    
    To get the correspondence with the RT_0 solution, we have to use L^2 projections for
    the mass, reaction, and source terms. Otherwise, we can still choose \vec c_E to enforce
    continuity of the normal fluxes.

    To solve for \vec c_E, we solve a d x d system on each element

      \mat{G}_{E}\vec c_E = \vec j_E

       G_{E,i:} = |e_i|\vec n_i, the unit outer normal for face i times its area/length
  
       j_{E,i}  = (s_h - r_h - m_{h,t},N_i)_{E} - |E|(\bar{s}-\bar{r}-bar{m}_t)/(d+1)

    where the integral (,)_{E} and averages should be computed with
    the same quadrature used for the P^1_nc solution, N_i is the test
    function that's 1 on e_i. For Dirichlet boundaries, we set j_{E,i}
    = 0

    
    In the notation
        \bar{s}_{E} is the average source over element E
        \bar{r}_{E} is the average reaction term
        \bar{m}_{t,E} is the average accumulation term
        \bar{\vec f}_{E} is the constant advection (gravity) term
        \bar{A}_{h} is the average diagonal conductivity over E
        \vec x_{E}  is the element barycenter
         d is the spatial dimension

    assumes that solution holds all degrees of freedom and boundary
    values.
 
    Note, r array holds -s + r terms.

    returns array rt0vdofs that's nelements by nd+1, that stores
       rt0vdofs[eN,:] = [\vec a_E,b_E]

   ***********************************************************************/
  proteus_index_t eN, ebN; int I, J, i, k, ebN_free, ebN_dir;
  int nSpace2 = nSpace*nSpace;
  int nDOF_RT0V_element = nSpace+1; /*number of dofs for vector part of RT0*/
  double volume,area,volFact,areaFact;
  double ah[3][3] = {{0.0,0.0,0.0},
		     {0.0,0.0,0.0},
		     {0.0,0.0,0.0}};
  double vecf[3]    = {0.0,0.0,0.0};
  double rh[4]    = {0.0,0.0,0.0,0.0}; /*int_{E}r N_i\dx */
  double mth[4]   = {0.0,0.0,0.0,0.0}; /*int_{E}m_t N_i\dx */
  double rbar,mtbar;
  double G_E[3][3] = {{0.0,0.0,0.0},
		      {0.0,0.0,0.0},
		      {0.0,0.0,0.0}};
  double G_Ei[3][3]={{0.0,0.0,0.0},
		     {0.0,0.0,0.0},
		     {0.0,0.0,0.0}}; 
  double rhs_c[3]  ={0.0,0.0,0.0};
  double c_E[3]   = {0.0,0.0,0.0};

  double dDim = (double) nSpace;
  double b_E = 0.;
  
  assert(nDOF_test_element == nSpace+1);
  volFact = 1.0; areaFact = 1.0;
  if (nSpace == 2)
    volFact = 0.5;
  if (nSpace == 3)
    {
      volFact = 1.0/6.0; areaFact = 0.5;
    }
  /*mwf debug
  printf("P1ncV2 MASS CALLED\n");
  */
  /*
    compute average for r,mt, f  and A from integration point values
  */
  for (eN = 0; eN < nElements_global; eN++)
    {
      b_E = 0.0; 
      rh[0] = 0.0; rh[1] = 0.0; rh[2]  = 0.0; rh[3]  = 0.0; rbar = 0.0;
      mth[0]= 0.0; mth[1]= 0.0; mth[2] = 0.0; mth[3] = 0.0; mtbar= 0.0;
      vecf[0]=0.0; vecf[1] = 0.0; vecf[2] = 0.0;
      ah[0][0] = 0.0; ah[0][1]=0.0; ah[0][2] = 0.0; 
      ah[1][0] = 0.0; ah[1][1]=0.0; ah[1][2] = 0.0; 
      ah[2][0] = 0.0; ah[2][1]=0.0; ah[2][2] = 0.0;
      
      /*assume affine*/
      volume = fabs(detJ[eN*nQuadraturePoints_element + 0])*volFact; 
      
      for (k = 0; k < nQuadraturePoints_element; k++)
	{

	  for (I = 0; I < nSpace; I++)
	    {
	      vecf[I] += 
		f[eN*nQuadraturePoints_element*nSpace + 
		  k*nSpace + 
		  I]
		*
		quad_f[k]
		*
		fabs(detJ[eN*nQuadraturePoints_element + k])/volume;
		
	      for (J = 0; J < nSpace; J++)
		{
		  ah[I][J] += 
		    a[eN*nQuadraturePoints_element*nSpace2 + 
		      k*nSpace2 +
		      I*nSpace  + 
		      J]
		    *
		    quad_a[k]
		    *
		    fabs(detJ[eN*nQuadraturePoints_element + k])/volume;
		}/*J*/
	    }/*I*/

	  for (i = 0; i < nDOF_test_element; i++) 
	    {
	      rh[i] += 
		r[eN*nQuadraturePoints_element + k] 
		*
		w_dV_r[eN*nQuadraturePoints_element*nDOF_test_element + 
		       k*nDOF_test_element + 
		       i];
	      
	      mth[i] += 
		mt[eN*nQuadraturePoints_element + k] 
		*
		w_dV_m[eN*nQuadraturePoints_element*nDOF_test_element + 
		       k*nDOF_test_element + 
		       i];
	    }/*i*/
	}/*k*/
      /*sum over test functions to get avgs, can't do this on the
       fly because of more points than just thoughs used in r*/
      for (i = 0; i < nDOF_test_element; i++) 
	{
	  rbar += rh[i]/volume; mtbar += mth[i]/volume;
	}
      b_E = (-rbar - mtbar)/dDim; /*r holds -s + r in notes*/
      /*b_E*/
      rt0vdofs[eN*nDOF_RT0V_element + nSpace] = b_E;
      /*compute base part of constant term as before*/
      for (I=0; I < nSpace; I++)
	{
	  rt0vdofs[eN*nDOF_RT0V_element + I] = 0.0;
	  for (J=0; J < nSpace; J++)
	    {
	      /*know that gradu is constant over element*/
	      rt0vdofs[eN*nDOF_RT0V_element + I] -= 
		ah[I][J] 
		*
		gradu[eN*nQuadraturePoints_element*nSpace +
		      0*nSpace +
		      I];
	    }/*J*/
	  /*advection term*/
	  /*mwf debug
	    printf("rt0 pp eN=%d vecc[%d]=%g \n",eN,I,vecc[I]);
	  */
	  rt0vdofs[eN*nDOF_RT0V_element + I] += vecf[I];
	  
	  rt0vdofs[eN*nDOF_RT0V_element + I] -= 
	    b_E
	    *
	    elementBarycenters[eN*3 + I]; /*points always 3d*/
	}/*I*/
      /*mwf debug
      printf("v2pp eN=%d b_E=%g b4 corr, a_E= ",eN,b_E);
      for (I=0; I < nSpace; I++)
	printf("%g ",rt0vdofs[eN*nDOF_RT0V_element + I]);
      printf("\n");
      */
      /*now compute correction due to not using L_2 projection for
	mass, reaction, and source terms*/
      rhs_c[0]  = 0.0; rhs_c[1]  = 0.0; rhs_c[2]  = 0.0;
      G_E[0][0] = 1.0; G_E[0][1] = 0.0; G_E[0][2] = 0.0;
      G_E[1][0] = 0.0; G_E[1][1] = 1.0; G_E[1][2] = 0.0; 
      G_E[2][0] = 0.0; G_E[2][1] = 0.0; G_E[2][2] = 1.0;
      
      ebN_free = nSpace; 
      if (nFreeDOF_element[eN] < ebN_free)
	ebN_free = nFreeDOF_element[eN];
      for (i = 0; i < ebN_free; i++)
	{
	  ebN = freeLocal_element[eN*nDOF_test_element + i];/*should be _trial_element*/
	  /*assumed affine*/ 
	  area = sqrt_det_g[eN*nElementBoundaries_element*nQuadraturePoints_elementBoundary + 
			    ebN*nQuadraturePoints_elementBoundary + 0]
	    *
	    areaFact;
	  for (I = 0; I < nSpace; I++)
	    G_E[i][I] = 
	      area
	      *
	      n[eN*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace + 
		ebN*nQuadraturePoints_elementBoundary*nSpace + 
		0*nSpace + 
		I];
	  /*recall r holds -s + r from notes*/
	  rhs_c[i] = -rh[ebN] + volume*rbar/(dDim+1.) -mth[ebN] + mtbar*volume/(dDim+1.);
	  /*mwf debug
	  if (fabs(rhs_c[i]) > 1.0e-16)
	    {
	      printf("v2pp eN=%d rhs_c[%d]= %g ebN=%d rh=%g rbar=%g vol*rbar/(d+1)= %g mth=%g mtbar=%g \n",
		     eN,i,rhs_c[i],ebN,rh[ebN],rbar,volume*rbar/(dDim+1.),mth[ebN],mtbar);
	      
	    }
	  */
	}

      /*mwf debug
      printf("v2pp eN=%d after nonDir, G_E= \n",eN);
      for (i=0; i < ebN_free; i++)
	{
	  for (I = 0; I < nSpace; I++)
	    printf("%g ",G_E[i][I]);
	  printf("\n");
	}
      */
      ebN = 0;
      while (ebN_free < nSpace && ebN < nElementBoundaries_element)
	{
	  /*
	    at most d-1 non-Dirichlet boundaries, so have to include 
	    dirichlet boundaries with rhs = 0
	  */
	  ebN_dir = 1;
	  for (k = 0; k < ebN_free; k++)
	    {
	      if (ebN == freeLocal_element[eN*nDOF_test_element + k])
		ebN_dir = 0;
	    }
	  if (ebN_dir > 0)
	    {
	      /*assumed affine*/ 
	      area = sqrt_det_g[eN*nElementBoundaries_element*nQuadraturePoints_elementBoundary + 
				ebN*nQuadraturePoints_elementBoundary + 0] 
		*
		areaFact;
	      for (I = 0; I < nSpace; I++)
		G_E[ebN_free][I] = 
		  area
		  *
		  n[eN*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace + 
		    ebN*nQuadraturePoints_elementBoundary*nSpace + 
		    0*nSpace + 
		    I];
	      rhs_c[ebN_free] = 0.0;

	      ebN_free++;
	    }
	  ebN++;
	}/*ebN_free*/
      assert (ebN_free >= nSpace);
      /*mwf debug
      printf("v2pp eN=%d after dir, G_E= \n",eN);
      for (i=0; i < nSpace; i++)
	{
	  for (I = 0; I < nSpace; I++)
	    printf("%g ",G_E[i][I]);
	  printf("\n");
	}
      */	    
      invertLocal(nSpace,G_E,G_Ei);
      for (I = 0; I < nSpace; I++)
	{
	  c_E[I] = 0.0;
	  for (J = 0; J < nSpace; J++)
	    c_E[I] += G_Ei[I][J]*rhs_c[J];
	}
      /*mwf debug
      printf("v2pp eN=%d after solve, \n\t rhs_c= ",eN);
      for (I=0; I < nSpace; I++)
	printf("%g ",rhs_c[I]);
      printf("\n\t c_E= ",eN);
      for (I=0; I < nSpace; I++)
	printf("%g ",c_E[I]);
      printf("\n");
      */
      /*now correct RT_0 approximation*/
      for (I = 0; I < nSpace; I++)
	rt0vdofs[eN*nDOF_RT0V_element + I] += c_E[I];
    }/*eN*/
}

void postProcessRT0velocityFromP1nc_sd(proteus_index_t nElements_global,
				       int nQuadraturePoints_element,
				       int nDOF_test_element,
				       int nElementBoundaries_element,
				       int nQuadraturePoints_elementBoundary,
				       int nSpace,
				       proteus_csr_index_t* rowptr,
				       proteus_csr_index_t* colind,
				       int * nFreeDOF_element,
				       int * freeLocal_element,
				       double * detJ,
				       double * sqrt_det_g,
				       double * n,
				       double * elementBarycenters,
				       double * quad_a,
				       double * quad_f,
				       double * w_dV_r,
				       double * w_dV_m,
				       double * u,
				       double * gradu,
				       double * a, 
				       double * f, 
				       double * r, 
				       double * mt,
				       double * rt0vdofs)
{
  /***********************************************************************
    combine example in Chen '94 paper and Chou and Tang '00 paper to
    take scalar finite element function, assuming it's from NC_Affine
    ...  space and compute an RT0 velocity. 

    This version includes a piecewise constant 'gravity'/advection
    term, f as well as mass and source terms that may not be computed
    using L_2 projection. It also likely only holds for diagonal tensor A.

    This means that we lose the correspondence with RT_0 mixed finite
    element approximation, but should still be able to get a velocity
    in RT_0 that's locally conservative (and has continuous normal
    fluxes).  Chou and Tang show that their approach should still
    converge to the correct solution O(h) I believe.

    Velocity (Darcy flux) definition is
  
     q =  -A\grad u + \vec f 

    where \vec f wouldy be \rho A\vec g, A is the (diagonal) 
    conductivity \vec g is gravity and \rho is a density.
    
    A and f need to be evaluated as the 'averages' aka L_2 projections
    on element-wise constants. Note that in the P^1_nc weak formulation, we get

     (A_h\grad u_h,\grad w_h)_E = (\bar{A}\grad u_h,\grad w_h)_E 
     (f_h,\grad w_h)_E          = (\bar{f},grad w_h)_E

    for a diagonal A, since \grad u_h and grad w_h are constant on E.

    Weak formulation for P^1_nc should be

    (m_{h,t},w_h) + \sum_{E}(A_h \grad u_h,\grad w_h) - \sum_{E}(f_h,\grad w_h)_E + 
       (r_h,w_h) = (s_h,w_h) - <q^b,w_h>_{\Gamma_N}


    Where, q^b = total flux on Neumann boundary

    To represent the RT0 velocity, we use
    
      \vec q_h = \vec a_E + b_E\vec x 

    which turns out to be on triangle E

      \vec q_h = -\bar{A}_h\grad u_h + \bar{\vec f}_h + 
             (\bar{s}_E-\bar{r}_{E}-\bar{m}_{t})/d(\vec x - \vec x_{E}) + \vec c_{E}

    The scalar b_E is determined exclusively by the mass-conservation constraint:

      b_E = (\bar{s}_E-\bar{r}_{E}-\bar{m}_{t})/d

    and the constant \vec a_{E} comes about through normal flux continuity condition 

        \vec a_T = -\bar{A}_{h,E}\grad u_h + \bar{\vec f} - b_E \vec x_{E} + \vec c_E
    
    To get the correspondence with the RT_0 solution, we have to use L^2 projections for
    the mass, reaction, and source terms. Otherwise, we can still choose \vec c_E to enforce
    continuity of the normal fluxes.

    To solve for \vec c_E, we solve a d x d system on each element

      \mat{G}_{E}\vec c_E = \vec j_E

       G_{E,i:} = |e_i|\vec n_i, the unit outer normal for face i times its area/length
  
       j_{E,i}  = (s_h - r_h - m_{h,t},N_i)_{E} - |E|(\bar{s}-\bar{r}-bar{m}_t)/(d+1)

    where the integral (,)_{E} and averages should be computed with
    the same quadrature used for the P^1_nc solution, N_i is the test
    function that's 1 on e_i. For Dirichlet boundaries, we set j_{E,i}
    = 0

    
    In the notation
        \bar{s}_{E} is the average source over element E
        \bar{r}_{E} is the average reaction term
        \bar{m}_{t,E} is the average accumulation term
        \bar{\vec f}_{E} is the constant advection (gravity) term
        \bar{A}_{h} is the average diagonal conductivity over E
        \vec x_{E}  is the element barycenter
         d is the spatial dimension

    assumes that solution holds all degrees of freedom and boundary
    values.
 
    Note, r array holds -s + r terms.

    returns array rt0vdofs that's nelements by nd+1, that stores
       rt0vdofs[eN,:] = [\vec a_E,b_E]

   ***********************************************************************/
  proteus_index_t eN, ebN; int I, J, i, k, ebN_free, ebN_dir;
  proteus_csr_index_t m; proteus_csr_index_t nnz=rowptr[nSpace];
  int nDOF_RT0V_element = nSpace+1; /*number of dofs for vector part of RT0*/
  double volume,area,volFact,areaFact;
  double ah[3][3] = {{0.0,0.0,0.0},
		     {0.0,0.0,0.0},
		     {0.0,0.0,0.0}};
  double vecf[3]    = {0.0,0.0,0.0};
  double rh[4]    = {0.0,0.0,0.0,0.0}; /*int_{E}r N_i\dx */
  double mth[4]   = {0.0,0.0,0.0,0.0}; /*int_{E}m_t N_i\dx */
  double rbar,mtbar;
  double G_E[3][3] = {{0.0,0.0,0.0},
		      {0.0,0.0,0.0},
		      {0.0,0.0,0.0}};
  double G_Ei[3][3]={{0.0,0.0,0.0},
		     {0.0,0.0,0.0},
		     {0.0,0.0,0.0}}; 
  double rhs_c[3]  ={0.0,0.0,0.0};
  double c_E[3]   = {0.0,0.0,0.0};

  double dDim = (double) nSpace;
  double b_E = 0.;
  
  assert(nDOF_test_element == nSpace+1);
  volFact = 1.0; areaFact = 1.0;
  if (nSpace == 2)
    volFact = 0.5;
  if (nSpace == 3)
    {
      volFact = 1.0/6.0; areaFact = 0.5;
    }
  /*mwf debug
  printf("P1ncV2 MASS CALLED\n");
  */
  /*
    compute average for r,mt, f  and A from integration point values
  */
  for (eN = 0; eN < nElements_global; eN++)
    {
      b_E = 0.0; 
      rh[0] = 0.0; rh[1] = 0.0; rh[2]  = 0.0; rh[3]  = 0.0; rbar = 0.0;
      mth[0]= 0.0; mth[1]= 0.0; mth[2] = 0.0; mth[3] = 0.0; mtbar= 0.0;
      vecf[0]=0.0; vecf[1] = 0.0; vecf[2] = 0.0;
      ah[0][0] = 0.0; ah[0][1]=0.0; ah[0][2] = 0.0; 
      ah[1][0] = 0.0; ah[1][1]=0.0; ah[1][2] = 0.0; 
      ah[2][0] = 0.0; ah[2][1]=0.0; ah[2][2] = 0.0;
      
      /*assume affine*/
      volume = fabs(detJ[eN*nQuadraturePoints_element + 0])*volFact; 
      
      for (k = 0; k < nQuadraturePoints_element; k++)
	{

	  for (I = 0; I < nSpace; I++)
	    {
	      vecf[I] += 
		f[eN*nQuadraturePoints_element*nSpace + 
		  k*nSpace + 
		  I]
		*
		quad_f[k]
		*
		fabs(detJ[eN*nQuadraturePoints_element + k])/volume;
		
	      for (m=rowptr[I];m<rowptr[I+1];m++)
		{
		  ah[I][colind[m]] += 
		    a[eN*nQuadraturePoints_element*nnz+
		      k*nnz+
		      m]
		    *
		    quad_a[k]
		    *
		    fabs(detJ[eN*nQuadraturePoints_element + k])/volume;
		}/*J*/
	    }/*I*/

	  for (i = 0; i < nDOF_test_element; i++) 
	    {
	      rh[i] += 
		r[eN*nQuadraturePoints_element + k] 
		*
		w_dV_r[eN*nQuadraturePoints_element*nDOF_test_element + 
		       k*nDOF_test_element + 
		       i];
	      
	      mth[i] += 
		mt[eN*nQuadraturePoints_element + k] 
		*
		w_dV_m[eN*nQuadraturePoints_element*nDOF_test_element + 
		       k*nDOF_test_element + 
		       i];
	    }/*i*/
	}/*k*/
      /*sum over test functions to get avgs, can't do this on the
       fly because of more points than just thoughs used in r*/
      for (i = 0; i < nDOF_test_element; i++) 
	{
	  rbar += rh[i]/volume; mtbar += mth[i]/volume;
	}
      b_E = (-rbar - mtbar)/dDim; /*r holds -s + r in notes*/
      /*b_E*/
      rt0vdofs[eN*nDOF_RT0V_element + nSpace] = b_E;
      /*compute base part of constant term as before*/
      for (I=0; I < nSpace; I++)
	{
	  rt0vdofs[eN*nDOF_RT0V_element + I] = 0.0;
	  for (J=0; J < nSpace; J++)
	    {
	      /*know that gradu is constant over element*/
	      rt0vdofs[eN*nDOF_RT0V_element + I] -= 
		ah[I][J] 
		*
		gradu[eN*nQuadraturePoints_element*nSpace +
		      0*nSpace +
		      I];
	    }/*J*/
	  /*advection term*/
	  /*mwf debug
	    printf("rt0 pp eN=%d vecc[%d]=%g \n",eN,I,vecc[I]);
	  */
	  rt0vdofs[eN*nDOF_RT0V_element + I] += vecf[I];
	  
	  rt0vdofs[eN*nDOF_RT0V_element + I] -= 
	    b_E
	    *
	    elementBarycenters[eN*3 + I]; /*points always 3d*/
	}/*I*/
      /*mwf debug
      printf("v2pp eN=%d b_E=%g b4 corr, a_E= ",eN,b_E);
      for (I=0; I < nSpace; I++)
	printf("%g ",rt0vdofs[eN*nDOF_RT0V_element + I]);
      printf("\n");
      */
      /*now compute correction due to not using L_2 projection for
	mass, reaction, and source terms*/
      rhs_c[0]  = 0.0; rhs_c[1]  = 0.0; rhs_c[2]  = 0.0;
      G_E[0][0] = 1.0; G_E[0][1] = 0.0; G_E[0][2] = 0.0;
      G_E[1][0] = 0.0; G_E[1][1] = 1.0; G_E[1][2] = 0.0; 
      G_E[2][0] = 0.0; G_E[2][1] = 0.0; G_E[2][2] = 1.0;
      
      ebN_free = nSpace; 
      if (nFreeDOF_element[eN] < ebN_free)
	ebN_free = nFreeDOF_element[eN];
      for (i = 0; i < ebN_free; i++)
	{
	  ebN = freeLocal_element[eN*nDOF_test_element + i];/*should be _trial_element*/
	  /*assumed affine*/ 
	  area = sqrt_det_g[eN*nElementBoundaries_element*nQuadraturePoints_elementBoundary + 
			    ebN*nQuadraturePoints_elementBoundary + 0]
	    *
	    areaFact;
	  for (I = 0; I < nSpace; I++)
	    G_E[i][I] = 
	      area
	      *
	      n[eN*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace + 
		ebN*nQuadraturePoints_elementBoundary*nSpace + 
		0*nSpace + 
		I];
	  /*recall r holds -s + r from notes*/
	  rhs_c[i] = -rh[ebN] + volume*rbar/(dDim+1.) -mth[ebN] + mtbar*volume/(dDim+1.);
	  /*mwf debug
	  if (fabs(rhs_c[i]) > 1.0e-16)
	    {
	      printf("v2pp eN=%d rhs_c[%d]= %g ebN=%d rh=%g rbar=%g vol*rbar/(d+1)= %g mth=%g mtbar=%g \n",
		     eN,i,rhs_c[i],ebN,rh[ebN],rbar,volume*rbar/(dDim+1.),mth[ebN],mtbar);
	      
	    }
	  */
	}

      /*mwf debug
      printf("v2pp eN=%d after nonDir, G_E= \n",eN);
      for (i=0; i < ebN_free; i++)
	{
	  for (I = 0; I < nSpace; I++)
	    printf("%g ",G_E[i][I]);
	  printf("\n");
	}
      */
      ebN = 0;
      while (ebN_free < nSpace && ebN < nElementBoundaries_element)
	{
	  /*
	    at most d-1 non-Dirichlet boundaries, so have to include 
	    dirichlet boundaries with rhs = 0
	  */
	  ebN_dir = 1;
	  for (k = 0; k < ebN_free; k++)
	    {
	      if (ebN == freeLocal_element[eN*nDOF_test_element + k])
		ebN_dir = 0;
	    }
	  if (ebN_dir > 0)
	    {
	      /*assumed affine*/ 
	      area = sqrt_det_g[eN*nElementBoundaries_element*nQuadraturePoints_elementBoundary + 
				ebN*nQuadraturePoints_elementBoundary + 0] 
		*
		areaFact;
	      for (I = 0; I < nSpace; I++)
		G_E[ebN_free][I] = 
		  area
		  *
		  n[eN*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace + 
		    ebN*nQuadraturePoints_elementBoundary*nSpace + 
		    0*nSpace + 
		    I];
	      rhs_c[ebN_free] = 0.0;

	      ebN_free++;
	    }
	  ebN++;
	}/*ebN_free*/
      assert (ebN_free >= nSpace);
      /*mwf debug
      printf("v2pp eN=%d after dir, G_E= \n",eN);
      for (i=0; i < nSpace; i++)
	{
	  for (I = 0; I < nSpace; I++)
	    printf("%g ",G_E[i][I]);
	  printf("\n");
	}
      */	    
      invertLocal(nSpace,G_E,G_Ei);
      for (I = 0; I < nSpace; I++)
	{
	  c_E[I] = 0.0;
	  for (J = 0; J < nSpace; J++)
	    c_E[I] += G_Ei[I][J]*rhs_c[J];
	}
      /*mwf debug
      printf("v2pp eN=%d after solve, \n\t rhs_c= ",eN);
      for (I=0; I < nSpace; I++)
	printf("%g ",rhs_c[I]);
      printf("\n\t c_E= ",eN);
      for (I=0; I < nSpace; I++)
	printf("%g ",c_E[I]);
      printf("\n");
      */
      /*now correct RT_0 approximation*/
      for (I = 0; I < nSpace; I++)
	rt0vdofs[eN*nDOF_RT0V_element + I] += c_E[I];
    }/*eN*/
}

void postProcessRT0velocityFromP1ncNoMass(proteus_index_t nElements_global,
					    int nQuadraturePoints_element,
					    int nDOF_test_element,
					    int nElementBoundaries_element,
					    int nQuadraturePoints_elementBoundary,
					    int nSpace,
					    int * nFreeDOF_element,
					    int * freeLocal_element,
					    double * detJ,
					    double * sqrt_det_g,
					    double * n,
					    double * elementBarycenters,
					    double * quad_a,
					    double * quad_f,
					    double * w_dV_r,
					    double * u,
					    double * gradu,
					    double * a, 
					    double * f, 
					    double * r, 
					    double * rt0vdofs)
{
  /***********************************************************************
    version of postProcessRT0velocityFromP1nc without a mass variable
   ***********************************************************************/
  proteus_index_t eN, ebN; int I, J, i, k, ebN_free, ebN_dir;
  int nSpace2 = nSpace*nSpace;
  int nDOF_RT0V_element = nSpace+1; /*number of dofs for vector part of RT0*/
  double volume,area,volFact,areaFact;
  double ah[3][3] = {{0.0,0.0,0.0},
		     {0.0,0.0,0.0},
		     {0.0,0.0,0.0}};
  double vecf[3]    = {0.0,0.0,0.0};
  double rh[4]    = {0.0,0.0,0.0,0.0}; /*int_{E}r N_i\dx */
  double rbar;
  double G_E[3][3] = {{0.0,0.0,0.0},
		      {0.0,0.0,0.0},
		      {0.0,0.0,0.0}};
  double G_Ei[3][3]={{0.0,0.0,0.0},
		     {0.0,0.0,0.0},
		     {0.0,0.0,0.0}}; 
  double rhs_c[3]  ={0.0,0.0,0.0};
  double c_E[3]   = {0.0,0.0,0.0};

  double dDim = (double) nSpace;
  double b_E = 0.;
  
  assert(nDOF_test_element == nSpace+1);
  volFact = 1.0; areaFact = 1.0;
  if (nSpace == 2)
    volFact = 0.5;
  if (nSpace == 3)
    {
      volFact = 1.0/6.0; areaFact = 0.5;
    }

  /*mwf debug
  printf("P1ncV2 NO MASS CALLED\n");
  */
  /*
    compute average for r, f  and A from integration point values
  */


  for (eN = 0; eN < nElements_global; eN++)
    {
      b_E = 0.0; 
      rh[0] = 0.0; rh[1] = 0.0; rh[2]  = 0.0; rh[3]  = 0.0; rbar = 0.0;
      vecf[0]=0.0; vecf[1] = 0.0; vecf[2] = 0.0;
      ah[0][0] = 0.0; ah[0][1]=0.0; ah[0][2] = 0.0; 
      ah[1][0] = 0.0; ah[1][1]=0.0; ah[1][2] = 0.0; 
      ah[2][0] = 0.0; ah[2][1]=0.0; ah[2][2] = 0.0;
      
      /*assume affine*/
      volume = fabs(detJ[eN*nQuadraturePoints_element + 0])*volFact; 
      
      for (k = 0; k < nQuadraturePoints_element; k++)
	{

	  for (I = 0; I < nSpace; I++)
	    {
	      vecf[I] += 
		f[eN*nQuadraturePoints_element*nSpace + 
		  k*nSpace + 
		  I]
		*
		quad_f[k]
		*
		fabs(detJ[eN*nQuadraturePoints_element + k])/volume;
		
	      for (J = 0; J < nSpace; J++)
		{
		  ah[I][J] += 
		    a[eN*nQuadraturePoints_element*nSpace2 + 
		      k*nSpace2 +
		      I*nSpace  + 
		      J]
		    *
		    quad_a[k]
		    *
		    fabs(detJ[eN*nQuadraturePoints_element + k])/volume;
		}/*J*/
	    }/*I*/

	  for (i = 0; i < nDOF_test_element; i++) 
	    {
	      rh[i] += 
		r[eN*nQuadraturePoints_element + k] 
		*
		w_dV_r[eN*nQuadraturePoints_element*nDOF_test_element + 
		       k*nDOF_test_element + 
		       i];
	      
	    }/*i*/
	}/*k*/
      /*sum over test functions to get avgs, can't do this on the
       fly because of more points than just thoughs used in r*/
      for (i = 0; i < nDOF_test_element; i++) 
	{
	  rbar += rh[i]/volume; 
	}
      b_E = -rbar/dDim; /*r holds -s + r in notes*/
      /*b_E*/
      rt0vdofs[eN*nDOF_RT0V_element + nSpace] = b_E;
      /*compute base part of constant term as before*/
      for (I=0; I < nSpace; I++)
	{
	  rt0vdofs[eN*nDOF_RT0V_element + I] = 0.0;
	  for (J=0; J < nSpace; J++)
	    {
	      /*know that gradu is constant over element*/
	      rt0vdofs[eN*nDOF_RT0V_element + I] -= 
		ah[I][J] 
		*
		gradu[eN*nQuadraturePoints_element*nSpace +
		      0*nSpace +
		      I];
	    }/*J*/
	  /*advection term*/
	  /*mwf debug
	    printf("rt0 pp eN=%d vecc[%d]=%g \n",eN,I,vecc[I]);
	  */
	  rt0vdofs[eN*nDOF_RT0V_element + I] += vecf[I];
	  
	  rt0vdofs[eN*nDOF_RT0V_element + I] -= 
	    b_E
	    *
	    elementBarycenters[eN*3 + I]; /*points always 3d*/
	}/*I*/
      /*mwf debug
      printf("v2pp eN=%d b_E=%g b4 corr, a_E= ",eN,b_E);
      for (I=0; I < nSpace; I++)
	printf("%g ",rt0vdofs[eN*nDOF_RT0V_element + I]);
      printf("\n");
      */
      /*now compute correction due to not using L_2 projection for
	mass, reaction, and source terms*/
      rhs_c[0]  = 0.0; rhs_c[1]  = 0.0; rhs_c[2]  = 0.0;
      G_E[0][0] = 1.0; G_E[0][1] = 0.0; G_E[0][2] = 0.0;
      G_E[1][0] = 0.0; G_E[1][1] = 1.0; G_E[1][2] = 0.0; 
      G_E[2][0] = 0.0; G_E[2][1] = 0.0; G_E[2][2] = 1.0;
      
      ebN_free = nSpace; 
      if (nFreeDOF_element[eN] < ebN_free)
	ebN_free = nFreeDOF_element[eN];
      for (i = 0; i < ebN_free; i++)
	{
	  ebN = freeLocal_element[eN*nDOF_test_element + i];/*should be _trial_element*/
	  /*assumed affine*/ 
	  area = sqrt_det_g[eN*nElementBoundaries_element*nQuadraturePoints_elementBoundary + 
			    ebN*nQuadraturePoints_elementBoundary + 0]
	    *
	    areaFact;
	  for (I = 0; I < nSpace; I++)
	    G_E[i][I] = 
	      area
	      *
	      n[eN*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace + 
		ebN*nQuadraturePoints_elementBoundary*nSpace + 
		0*nSpace + 
		I];
	  /*recall r holds -s + r from notes*/
	  rhs_c[i] = -rh[ebN] + volume*rbar/(dDim+1.);
	  /*mwf debug
	  if (fabs(rhs_c[i]) > 1.0e-16)
	    {
	      printf("v2pp eN=%d rhs_c[%d]= %g ebN=%d rh=%g rbar=%g vol*rbar/(d+1)= %g mth=%g mtbar=%g \n",
		     eN,i,rhs_c[i],ebN,rh[ebN],rbar,volume*rbar/(dDim+1.),mth[ebN],mtbar);
	      
	    }
	  */
	}

      /*mwf debug
      printf("v2pp eN=%d after nonDir, G_E= \n",eN);
      for (i=0; i < ebN_free; i++)
	{
	  for (I = 0; I < nSpace; I++)
	    printf("%g ",G_E[i][I]);
	  printf("\n");
	}
      */
      ebN = 0;
      while (ebN_free < nSpace && ebN < nElementBoundaries_element)
	{
	  /*
	    at most d-1 non-Dirichlet boundaries, so have to include 
	    dirichlet boundaries with rhs = 0
	  */
	  ebN_dir = 1;
	  for (k = 0; k < ebN_free; k++)
	    {
	      if (ebN == freeLocal_element[eN*nDOF_test_element + k])
		ebN_dir = 0;
	    }
	  if (ebN_dir > 0)
	    {
	      /*assumed affine*/ 
	      area = sqrt_det_g[eN*nElementBoundaries_element*nQuadraturePoints_elementBoundary + 
				ebN*nQuadraturePoints_elementBoundary + 0] 
		*
		areaFact;
	      for (I = 0; I < nSpace; I++)
		G_E[ebN_free][I] = 
		  area
		  *
		  n[eN*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace + 
		    ebN*nQuadraturePoints_elementBoundary*nSpace + 
		    0*nSpace + 
		    I];
	      rhs_c[ebN_free] = 0.0;

	      ebN_free++;
	    }
	  ebN++;
	}/*ebN_free*/
      assert (ebN_free >= nSpace);
      /*mwf debug
      printf("v2pp eN=%d after dir, G_E= \n",eN);
      for (i=0; i < nSpace; i++)
	{
	  for (I = 0; I < nSpace; I++)
	    printf("%g ",G_E[i][I]);
	  printf("\n");
	}
      */	    
      invertLocal(nSpace,G_E,G_Ei);
      for (I = 0; I < nSpace; I++)
	{
	  c_E[I] = 0.0;
	  for (J = 0; J < nSpace; J++)
	    c_E[I] += G_Ei[I][J]*rhs_c[J];
	}
      /*mwf debug
      printf("v2pp eN=%d after solve, \n\t rhs_c= ",eN);
      for (I=0; I < nSpace; I++)
	printf("%g ",rhs_c[I]);
      printf("\n\t c_E= ",eN);
      for (I=0; I < nSpace; I++)
	printf("%g ",c_E[I]);
      printf("\n");
      */
      /*now correct RT_0 approximation*/
      for (I = 0; I < nSpace; I++)
	rt0vdofs[eN*nDOF_RT0V_element + I] += c_E[I];
    }/*eN*/
}

void postProcessRT0velocityFromP1ncNoMass_sd(proteus_index_t nElements_global,
					     int nQuadraturePoints_element,
					     int nDOF_test_element,
					     int nElementBoundaries_element,
					     int nQuadraturePoints_elementBoundary,
					     int nSpace,
					     proteus_csr_index_t* rowptr,
					     proteus_csr_index_t* colind,
					     int * nFreeDOF_element,
					     int * freeLocal_element,
					     double * detJ,
					     double * sqrt_det_g,
					     double * n,
					     double * elementBarycenters,
					     double * quad_a,
					     double * quad_f,
					     double * w_dV_r,
					     double * u,
					     double * gradu,
					     double * a, 
					     double * f, 
					     double * r, 
					     double * rt0vdofs)
{
  /***********************************************************************
    version of postProcessRT0velocityFromP1nc without a mass variable
   ***********************************************************************/
  proteus_index_t eN, ebN; int I, J, i, k, ebN_free, ebN_dir;
  proteus_csr_index_t m; proteus_csr_index_t nnz=rowptr[nSpace];
  int nDOF_RT0V_element = nSpace+1; /*number of dofs for vector part of RT0*/
  double volume,area,volFact,areaFact;
  double ah[3][3] = {{0.0,0.0,0.0},
		     {0.0,0.0,0.0},
		     {0.0,0.0,0.0}};
  double vecf[3]    = {0.0,0.0,0.0};
  double rh[4]    = {0.0,0.0,0.0,0.0}; /*int_{E}r N_i\dx */
  double rbar;
  double G_E[3][3] = {{0.0,0.0,0.0},
		      {0.0,0.0,0.0},
		      {0.0,0.0,0.0}};
  double G_Ei[3][3]={{0.0,0.0,0.0},
		     {0.0,0.0,0.0},
		     {0.0,0.0,0.0}}; 
  double rhs_c[3]  ={0.0,0.0,0.0};
  double c_E[3]   = {0.0,0.0,0.0};

  double dDim = (double) nSpace;
  double b_E = 0.;
  
  assert(nDOF_test_element == nSpace+1);
  volFact = 1.0; areaFact = 1.0;
  if (nSpace == 2)
    volFact = 0.5;
  if (nSpace == 3)
    {
      volFact = 1.0/6.0; areaFact = 0.5;
    }

  /*mwf debug
  printf("P1ncV2 NO MASS CALLED\n");
  */
  /*
    compute average for r, f  and A from integration point values
  */


  for (eN = 0; eN < nElements_global; eN++)
    {
      b_E = 0.0; 
      rh[0] = 0.0; rh[1] = 0.0; rh[2]  = 0.0; rh[3]  = 0.0; rbar = 0.0;
      vecf[0]=0.0; vecf[1] = 0.0; vecf[2] = 0.0;
      ah[0][0] = 0.0; ah[0][1]=0.0; ah[0][2] = 0.0; 
      ah[1][0] = 0.0; ah[1][1]=0.0; ah[1][2] = 0.0; 
      ah[2][0] = 0.0; ah[2][1]=0.0; ah[2][2] = 0.0;
      
      /*assume affine*/
      volume = fabs(detJ[eN*nQuadraturePoints_element + 0])*volFact; 
      
      for (k = 0; k < nQuadraturePoints_element; k++)
	{

	  for (I = 0; I < nSpace; I++)
	    {
	      vecf[I] += 
		f[eN*nQuadraturePoints_element*nSpace + 
		  k*nSpace + 
		  I]
		*
		quad_f[k]
		*
		fabs(detJ[eN*nQuadraturePoints_element + k])/volume;
		
	      for (m=rowptr[I];m<rowptr[I+1];m++)
		{
		  ah[I][colind[m]] += 
		    a[eN*nQuadraturePoints_element*nnz+
		      k*nnz+
		      m]
		    *
		    quad_a[k]
		    *
		    fabs(detJ[eN*nQuadraturePoints_element + k])/volume;
		}/*J*/
	    }/*I*/

	  for (i = 0; i < nDOF_test_element; i++) 
	    {
	      rh[i] += 
		r[eN*nQuadraturePoints_element + k] 
		*
		w_dV_r[eN*nQuadraturePoints_element*nDOF_test_element + 
		       k*nDOF_test_element + 
		       i];
	      
	    }/*i*/
	}/*k*/
      /*sum over test functions to get avgs, can't do this on the
       fly because of more points than just thoughs used in r*/
      for (i = 0; i < nDOF_test_element; i++) 
	{
	  rbar += rh[i]/volume; 
	}
      b_E = -rbar/dDim; /*r holds -s + r in notes*/
      /*b_E*/
      rt0vdofs[eN*nDOF_RT0V_element + nSpace] = b_E;
      /*compute base part of constant term as before*/
      for (I=0; I < nSpace; I++)
	{
	  rt0vdofs[eN*nDOF_RT0V_element + I] = 0.0;
	  for (J=0; J < nSpace; J++)
	    {
	      /*know that gradu is constant over element*/
	      rt0vdofs[eN*nDOF_RT0V_element + I] -= 
		ah[I][J] 
		*
		gradu[eN*nQuadraturePoints_element*nSpace +
		      0*nSpace +
		      I];
	    }/*J*/
	  /*advection term*/
	  /*mwf debug
	    printf("rt0 pp eN=%d vecc[%d]=%g \n",eN,I,vecc[I]);
	  */
	  rt0vdofs[eN*nDOF_RT0V_element + I] += vecf[I];
	  
	  rt0vdofs[eN*nDOF_RT0V_element + I] -= 
	    b_E
	    *
	    elementBarycenters[eN*3 + I]; /*points always 3d*/
	}/*I*/
      /*mwf debug
      printf("v2pp eN=%d b_E=%g b4 corr, a_E= ",eN,b_E);
      for (I=0; I < nSpace; I++)
	printf("%g ",rt0vdofs[eN*nDOF_RT0V_element + I]);
      printf("\n");
      */
      /*now compute correction due to not using L_2 projection for
	mass, reaction, and source terms*/
      rhs_c[0]  = 0.0; rhs_c[1]  = 0.0; rhs_c[2]  = 0.0;
      G_E[0][0] = 1.0; G_E[0][1] = 0.0; G_E[0][2] = 0.0;
      G_E[1][0] = 0.0; G_E[1][1] = 1.0; G_E[1][2] = 0.0; 
      G_E[2][0] = 0.0; G_E[2][1] = 0.0; G_E[2][2] = 1.0;
      
      ebN_free = nSpace; 
      if (nFreeDOF_element[eN] < ebN_free)
	ebN_free = nFreeDOF_element[eN];
      for (i = 0; i < ebN_free; i++)
	{
	  ebN = freeLocal_element[eN*nDOF_test_element + i];/*should be _trial_element*/
	  /*assumed affine*/ 
	  area = sqrt_det_g[eN*nElementBoundaries_element*nQuadraturePoints_elementBoundary + 
			    ebN*nQuadraturePoints_elementBoundary + 0]
	    *
	    areaFact;
	  for (I = 0; I < nSpace; I++)
	    G_E[i][I] = 
	      area
	      *
	      n[eN*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace + 
		ebN*nQuadraturePoints_elementBoundary*nSpace + 
		0*nSpace + 
		I];
	  /*recall r holds -s + r from notes*/
	  rhs_c[i] = -rh[ebN] + volume*rbar/(dDim+1.);
	  /*mwf debug
	  if (fabs(rhs_c[i]) > 1.0e-16)
	    {
	      printf("v2pp eN=%d rhs_c[%d]= %g ebN=%d rh=%g rbar=%g vol*rbar/(d+1)= %g mth=%g mtbar=%g \n",
		     eN,i,rhs_c[i],ebN,rh[ebN],rbar,volume*rbar/(dDim+1.),mth[ebN],mtbar);
	      
	    }
	  */
	}

      /*mwf debug
      printf("v2pp eN=%d after nonDir, G_E= \n",eN);
      for (i=0; i < ebN_free; i++)
	{
	  for (I = 0; I < nSpace; I++)
	    printf("%g ",G_E[i][I]);
	  printf("\n");
	}
      */
      ebN = 0;
      while (ebN_free < nSpace && ebN < nElementBoundaries_element)
	{
	  /*
	    at most d-1 non-Dirichlet boundaries, so have to include 
	    dirichlet boundaries with rhs = 0
	  */
	  ebN_dir = 1;
	  for (k = 0; k < ebN_free; k++)
	    {
	      if (ebN == freeLocal_element[eN*nDOF_test_element + k])
		ebN_dir = 0;
	    }
	  if (ebN_dir > 0)
	    {
	      /*assumed affine*/ 
	      area = sqrt_det_g[eN*nElementBoundaries_element*nQuadraturePoints_elementBoundary + 
				ebN*nQuadraturePoints_elementBoundary + 0] 
		*
		areaFact;
	      for (I = 0; I < nSpace; I++)
		G_E[ebN_free][I] = 
		  area
		  *
		  n[eN*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace + 
		    ebN*nQuadraturePoints_elementBoundary*nSpace + 
		    0*nSpace + 
		    I];
	      rhs_c[ebN_free] = 0.0;

	      ebN_free++;
	    }
	  ebN++;
	}/*ebN_free*/
      assert (ebN_free >= nSpace);
      /*mwf debug
      printf("v2pp eN=%d after dir, G_E= \n",eN);
      for (i=0; i < nSpace; i++)
	{
	  for (I = 0; I < nSpace; I++)
	    printf("%g ",G_E[i][I]);
	  printf("\n");
	}
      */	    
      invertLocal(nSpace,G_E,G_Ei);
      for (I = 0; I < nSpace; I++)
	{
	  c_E[I] = 0.0;
	  for (J = 0; J < nSpace; J++)
	    c_E[I] += G_Ei[I][J]*rhs_c[J];
	}
      /*mwf debug
      printf("v2pp eN=%d after solve, \n\t rhs_c= ",eN);
      for (I=0; I < nSpace; I++)
	printf("%g ",rhs_c[I]);
      printf("\n\t c_E= ",eN);
      for (I=0; I < nSpace; I++)
	printf("%g ",c_E[I]);
      printf("\n");
      */
      /*now correct RT_0 approximation*/
      for (I = 0; I < nSpace; I++)
	rt0vdofs[eN*nDOF_RT0V_element + I] += c_E[I];
    }/*eN*/
}

void updateRT0velocityWithAveragedPotentialP1nc(proteus_index_t nElements_global,
						int nQuadraturePoints_element,
						int nSpace,
						double * detJ,
						double * quad_a,
						double * phi,
						double * gradphi,
						double * a, 
						double * rt0vdofs)
{
  /***********************************************************************
    In case have multiple potentials for conservation equation:
      -\ten{a}_{j}\grad \phi_j
    
    correct constant part of p1nc RT0 flux with the corresponding
    product. This routine computes average for \ten_{a}_j and product
    that then goes into constant term for velocity (\vec a_E)

    returns array rt0vdofs that's nelements by nd+1, that stores
       rt0vdofs[eN,:] = [\vec a_E,b_E]

   ***********************************************************************/
  proteus_index_t eN; int I, J, k;
  int nSpace2 = nSpace*nSpace;
  int nDOF_RT0V_element = nSpace+1; /*number of dofs for vector part of RT0*/
  double volume,volFact;
  double ah[3][3] = {{0.0,0.0,0.0},
		     {0.0,0.0,0.0},
		     {0.0,0.0,0.0}};

  volFact = 1.0;
  if (nSpace == 2)
    volFact = 0.5;
  if (nSpace == 3)
    volFact = 1.0/6.0;

  /*mwf debug
  printf("P1ncV2 potential correction  CALLED\n");
  */
  /*
    compute average for A from integration point values
  */
  for (eN = 0; eN < nElements_global; eN++)
    {
      ah[0][0] = 0.0; ah[0][1]=0.0; ah[0][2] = 0.0; 
      ah[1][0] = 0.0; ah[1][1]=0.0; ah[1][2] = 0.0; 
      ah[2][0] = 0.0; ah[2][1]=0.0; ah[2][2] = 0.0;
      
      /*assume affine*/
      volume = fabs(detJ[eN*nQuadraturePoints_element + 0])*volFact; 
      
      for (k = 0; k < nQuadraturePoints_element; k++)
	{

	  for (I = 0; I < nSpace; I++)
	    {
	      for (J = 0; J < nSpace; J++)
		{
		  ah[I][J] += 
		    a[eN*nQuadraturePoints_element*nSpace2 + 
		      k*nSpace2 +
		      I*nSpace  + 
		      J]
		    *
		    quad_a[k]
		    *
		    fabs(detJ[eN*nQuadraturePoints_element + k])/volume;
		}/*J*/
	    }/*I*/

	}/*k*/
      /*compute base part of constant term as before*/
      for (I=0; I < nSpace; I++)
	{
	  for (J=0; J < nSpace; J++)
	    {
	      /*know that gradu is constant over element*/
	      rt0vdofs[eN*nDOF_RT0V_element + I] -= 
		ah[I][J] 
		*
		gradphi[eN*nQuadraturePoints_element*nSpace +
			0*nSpace +
			I];
	    }/*J*/
	}/*I*/
      /*mwf debug
      printf("v2pp eN=%d b_E=%g b4 corr, a_E= ",eN,b_E);
      for (I=0; I < nSpace; I++)
	printf("%g ",rt0vdofs[eN*nDOF_RT0V_element + I]);
      printf("\n");
      */
    }/*eN*/
}

void updateRT0velocityWithAveragedPotentialP1nc_sd(proteus_index_t nElements_global,
						   int nQuadraturePoints_element,
						   int nSpace,
						   proteus_csr_index_t* rowptr,
						   proteus_csr_index_t* colind,
						   double * detJ,
						   double * quad_a,
						   double * phi,
						   double * gradphi,
						   double * a, 
						   double * rt0vdofs)
{
  /***********************************************************************
    In case have multiple potentials for conservation equation:
      -\ten{a}_{j}\grad \phi_j
    
    correct constant part of p1nc RT0 flux with the corresponding
    product. This routine computes average for \ten_{a}_j and product
    that then goes into constant term for velocity (\vec a_E)

    returns array rt0vdofs that's nelements by nd+1, that stores
       rt0vdofs[eN,:] = [\vec a_E,b_E]

   ***********************************************************************/
  proteus_index_t eN; int I, J, k;
  proteus_csr_index_t m; proteus_csr_index_t nnz=rowptr[nSpace];
  int nDOF_RT0V_element = nSpace+1; /*number of dofs for vector part of RT0*/
  double volume,volFact;
  double ah[3][3] = {{0.0,0.0,0.0},
		     {0.0,0.0,0.0},
		     {0.0,0.0,0.0}};

  volFact = 1.0;
  if (nSpace == 2)
    volFact = 0.5;
  if (nSpace == 3)
    volFact = 1.0/6.0;

  /*mwf debug
  printf("P1ncV2 potential correction  CALLED\n");
  */
  /*
    compute average for A from integration point values
  */
  for (eN = 0; eN < nElements_global; eN++)
    {
      ah[0][0] = 0.0; ah[0][1]=0.0; ah[0][2] = 0.0; 
      ah[1][0] = 0.0; ah[1][1]=0.0; ah[1][2] = 0.0; 
      ah[2][0] = 0.0; ah[2][1]=0.0; ah[2][2] = 0.0;
      
      /*assume affine*/
      volume = fabs(detJ[eN*nQuadraturePoints_element + 0])*volFact; 
      
      for (k = 0; k < nQuadraturePoints_element; k++)
	{

	  for (I = 0; I < nSpace; I++)
	    {
	      for(m=rowptr[I];m<rowptr[I+1];m++)
		{
		  ah[I][colind[m]] += 
		    a[eN*nQuadraturePoints_element*nnz+
		      k*nnz+
		      m]
		    *
		    quad_a[k]
		    *
		    fabs(detJ[eN*nQuadraturePoints_element + k])/volume;
		}/*J*/
	    }/*I*/

	}/*k*/
      /*compute base part of constant term as before*/
      for (I=0; I < nSpace; I++)
	{
	  for (J=0; J < nSpace; J++)
	    {
	      /*know that gradu is constant over element*/
	      rt0vdofs[eN*nDOF_RT0V_element + I] -= 
		ah[I][J] 
		*
		gradphi[eN*nQuadraturePoints_element*nSpace +
			0*nSpace +
			I];
	    }/*J*/
	}/*I*/
      /*mwf debug
      printf("v2pp eN=%d b_E=%g b4 corr, a_E= ",eN,b_E);
      for (I=0; I < nSpace; I++)
	printf("%g ",rt0vdofs[eN*nDOF_RT0V_element + I]);
      printf("\n");
      */
    }/*eN*/
}

void getElementRT0velocityValues(proteus_index_t nElements_global,
				 int nPoints_element,
				 int nSpace,
				 double * x_element,
				 double * rt0vdofs_element,
				 double * v_element)
{
  /***********************************************************************
    Compute \vec q_h at physical points stored on each element

    On element T:

      \vec q_h = \vec a_T + b_t\vec x
 
    where rt0vdofs_element is logically nElements_global x nSpace+1
    assumes 
       x         stored as nElements_global \times nPoints \times 3
       v_element stored as nElements_global \times nPoints \times nSpace

   ***********************************************************************/
  proteus_index_t eN; int I, k;
  int nDOF_RT0V_element = nSpace+1; /*number of dofs for vector part of RT0*/
  double b_T = 0.;
  for (eN = 0; eN < nElements_global; eN++)
    {

      b_T = rt0vdofs_element[eN*nDOF_RT0V_element + nSpace];
      for (k = 0; k < nPoints_element; k++)
	{
	  for (I = 0; I < nSpace; I++)
	    {
	      v_element[eN*nPoints_element*nSpace + k*nSpace + I] = 
		rt0vdofs_element[eN*nDOF_RT0V_element + I] + 
		b_T * x_element[eN*nPoints_element*3 + k*3 + I];
	    }
	}/*k*/
    }/*eN*/

}
void getElementBoundaryRT0velocityValues(proteus_index_t nElements_global,
					 int nElementBoundaries_element,
					 int nPoints_elementBoundary,
					 int nSpace,
					 double * x_elementBoundary,
					 double * rt0vdofs_element,
					 double * v_elementBoundary)
{
  /***********************************************************************
    Compute \vec q_h at physical points stored on each element boundary

    On element T:

      \vec q_h = \vec a_T + b_t\vec x
 
    where rt0vdofs_element is logically nElements_global x nSpace+1
    assumes 
       x         stored as nElements_global \times nElementBoundaries_element 
                    \times nPoints \times 3
       v_element stored as nElements_global \times nElementBoundaries_element 
                   \times nPoints \times nSpace

   ***********************************************************************/
  proteus_index_t eN, ebN; int I, k;
  int nDOF_RT0V_element = nSpace+1; /*number of dofs for vector part of RT0*/
  double b_T = 0.;
  for (eN = 0; eN < nElements_global; eN++)
    {

      b_T = rt0vdofs_element[eN*nDOF_RT0V_element + nSpace];
      for (ebN = 0; ebN < nElementBoundaries_element; ebN++)
	{
	  for (k = 0; k < nPoints_elementBoundary; k++)
	    {
	      for (I = 0; I < nSpace; I++)
		{
		  v_elementBoundary[eN*nElementBoundaries_element*nPoints_elementBoundary*nSpace +
				    ebN*nPoints_elementBoundary*nSpace + 
				    k*nSpace + I] 
		    = 
		    rt0vdofs_element[eN*nDOF_RT0V_element + I] + 
		    b_T * x_elementBoundary[eN*nElementBoundaries_element*nPoints_elementBoundary*3 + 
					    ebN*nPoints_elementBoundary*3 + 
					    k*3 + I];
		}/*I*/
	    }/*k*/
	}/*ebN*/
    }/*eN*/

}
void getGlobalElementBoundaryRT0velocityValues(proteus_index_t nElementBoundaries_global,
					       int nPoints_elementBoundary,
					       int nSpace,
					       proteus_index_t * elementBoundaryElementsArray,
					       double * x_elementBoundary_global,
					       double * rt0vdofs_element,
					       double * v_elementBoundary_global)
{
  /***********************************************************************
    Compute \vec q_h at physical points stored on each global
    elementBoundary Just use the left neighbor because it always
    exists, and the flux is "known" to be continuous. A good check
    would be to compute the values from the right too.
 
    Recall on element T:

      \vec q_h = \vec a_T + b_t\vec x
 
    rt0vdofs_element is logically nElements_global x nSpace+1
    assumes 
       x                         stored as nElementBoundaries_global \times nPoints \times 3
       v_elementBoundary_global  stored as nElementBoundaries_global \times nPoints \times nSpace

   ***********************************************************************/
  proteus_index_t ebN, eN; int I, k;
  int nDOF_RT0V_element = nSpace+1; /*number of dofs for vector part of RT0*/
  double b_T = 0.;
  for (ebN = 0; ebN < nElementBoundaries_global; ebN++)
    {
      eN = elementBoundaryElementsArray[ebN*2 + 0]; /*left neighbor*/
      b_T = rt0vdofs_element[eN*nDOF_RT0V_element + nSpace];
      for (k = 0; k < nPoints_elementBoundary; k++)
	{
	  for (I = 0; I < nSpace; I++)
	    {
	      v_elementBoundary_global[ebN*nPoints_elementBoundary*nSpace + k*nSpace + I] = 
		rt0vdofs_element[eN*nDOF_RT0V_element + I] + 
		b_T * x_elementBoundary_global[ebN*nPoints_elementBoundary*3 + k*3 + I];
	    }/*I*/
	}/*k*/
    }/*ebN*/

}
void getGlobalExteriorElementBoundaryRT0velocityValues(proteus_index_t nExteriorElementBoundaries_global,
						       int nPoints_elementBoundary,
						       int nSpace,
						       proteus_index_t * elementBoundaryElementsArray,
						       proteus_index_t * exteriorElementBoundariesArray,
						       double * x_elementBoundary_global,
						       double * rt0vdofs_element,
						       double * v_elementBoundary_global)
{
  /***********************************************************************
    Compute \vec q_h at physical points stored on each global exterior
    elementBoundary Just use the left neighbor because it always
    exists, and the flux is "known" to be continuous. A good check
    would be to compute the values from the right too.
 
    Recall on element T:

      \vec q_h = \vec a_T + b_t\vec x
 
    rt0vdofs_element is logically nElements_global x nSpace+1
    assumes 
       x                         stored as nElementBoundaries_global \times nPoints \times 3
       v_elementBoundary_global  stored as nElementBoundaries_global \times nPoints \times nSpace

   ***********************************************************************/
  proteus_index_t ebN, eN; int ebNE, I, k;
  int nDOF_RT0V_element = nSpace+1; /*number of dofs for vector part of RT0*/
  double b_T = 0.;
  for (ebNE = 0; ebNE < nExteriorElementBoundaries_global; ebNE++)
    {
      ebN = exteriorElementBoundariesArray[ebNE];
      eN = elementBoundaryElementsArray[ebN*2 + 0]; /*left neighbor*/
      b_T = rt0vdofs_element[eN*nDOF_RT0V_element + nSpace];
      for (k = 0; k < nPoints_elementBoundary; k++)
	{
	  for (I = 0; I < nSpace; I++)
	    {
	      v_elementBoundary_global[ebNE*nPoints_elementBoundary*nSpace + k*nSpace + I] = 
		rt0vdofs_element[eN*nDOF_RT0V_element + I] + 
		b_T * x_elementBoundary_global[ebNE*nPoints_elementBoundary*3 + k*3 + I];
	    }/*I*/
	}/*k*/
    }/*ebN*/

}

void postProcessRT0potentialFromP1nc(proteus_index_t nElements_global,
				     int nQuadraturePoints_element,
				     int nElementBoundaries_element,
				     int nQuadraturePoints_elementBoundary,
				     int nSpace,
				     double * uQuadratureWeights_element,
				     double * elementBarycenters,
				     double * aElementQuadratureWeights,
				     double * detJ,
				     double * uQuadratureWeights_elementBoundary,
				     double * x,
				     double * u,
				     double * gradu,
				     double * x_elementBoundary,
				     double * u_elementBoundary,
				     double * n,
				     double * a,
				     double * f,
				     double * r,
				     double * rt0vdofs,
				     double * rt0potential)
{
  /***********************************************************************
     follow example in Chen '94 paper to get RTO potential value.
     Assumes RT0 flux has already been calculated!
     
     here, we basically have to plug in the P^1_{nc} solution,
     and the postprocessed flux into the local Darcy's law expression
     from the mixed hybrid formulation using the test function
       \vec v_h= \vec x.
     Then we solve for the pressure unknown.
     
     (\mat{A}_h^{-1}\vec q_h,\vec v_h)_T  - (p_h,\div \vec v_h)_T
        + (\lambda_h,\vec v_h\cdot n_{T})_{\partial T} - (\bar{\vec c},\vec v_h)_T = 0
     
     Recall, that \lambda^{j}_h = p^j_h, the P^1_{nc} solution for edge j
    
     To evaluate the weak integrals, I'll use the quadrature formula

      \int_{T} f \dx \approx |T|/3 \sum_{j}f(\vec \bar{x}^j)
    
     for the mass matrix term, and basically exact integration (midpoint rule)
     for the boundary integrals of the Lagrange multipler term since it's
     linear on each edge, and midpoint rule for the "advection" term since it's linear
    
   ***********************************************************************/
  proteus_index_t eN, ebN; int I, J, k;
  int nSpace2 = nSpace*nSpace;
  int nDOF_RT0V_element = nSpace+1; /*number of dofs for vector part of RT0*/
  double volume,vmass,bndsum,adot,ahqInvI,xdotn,gravsum;
  double ah[3][3] = {{0.0,0.0,0.0},
		     {0.0,0.0,0.0},
		     {0.0,0.0,0.0}};
  double ahInv[3][3] = {{0.0,0.0,0.0},
			{0.0,0.0,0.0},
			{0.0,0.0,0.0}};
  double vecc[3]  = {0.0,0.0,0.0};

  double dDim = nSpace;
  for (eN = 0; eN < nElements_global; eN++)
    {
      volume = 0.0;
      for (k = 0; k < nQuadraturePoints_element; k++)
	{
	  volume += uQuadratureWeights_element[eN*nQuadraturePoints_element + k];
	}
      /*mwf debug 
      printf("RT0pot. volume[%d]= %g \n",eN,volume);
      */
      for (I = 0; I < nSpace; I++)
	{
	  for (J = 0; J < nSpace; J++)
	    {
	      /*need to stick to a quad points and weights to be consistent with
		original calculation*/
	      ah[I][J] = 0.0;
	      for (k = 0; k < nQuadraturePoints_element; k++)
		{
		  ah[I][J] +=
		    a[eN*nQuadraturePoints_element*nSpace2 +
		      k*nSpace2 +
		      I*nSpace +
		      J]
		    *
		    aElementQuadratureWeights[k]*fabs(detJ[eN*nQuadraturePoints_element+k]);
		}/*k*/
	      ah[I][J] = ah[I][J]/volume;
	    }/*J*/
	  vecc[I] = 0.0;
	  for (k = 0; k < nQuadraturePoints_element; k++)
	    {
	      vecc[I] +=
		f[eN*nQuadraturePoints_element*nSpace +
		  k*nSpace + I]
		*
		aElementQuadratureWeights[k]*fabs(detJ[eN*nQuadraturePoints_element+k]);
	    }/*k*/
	  vecc[I] = vecc[I]/volume;
	}/*I*/
      /*mwf debug 
	printf("RT0pot. before invertLocal ah= \n %g %g \n %g %g \n",ah[0][0],ah[0][1],
	ah[1][0],ah[1][1]);
      */
      invertLocal(nSpace,ah,ahInv);
      /*mwf debug
	printf("RT0pot. after invertLocal AhInv= \n %g %g \n %g %g \n",ahInv[0][0],ahInv[0][1],
	ahInv[1][0],ahInv[1][1]);
      */
      vmass = 0.0;
      for (k = 0; k < nQuadraturePoints_element; k++)
	{
	  adot = 0.0;
	  for (I = 0; I < nSpace; I++)
	    {
	      ahqInvI = 0.0;
	      for (J = 0; J < nSpace; J++)
		ahqInvI+= ahInv[I][J]*(rt0vdofs[eN*nDOF_RT0V_element + J]
				       +
				       rt0vdofs[eN*nDOF_RT0V_element + nSpace]
				       *
				       x[eN*nQuadraturePoints_element*3 + 
					 k*3 + J]);
	      
	      adot += ahqInvI*x[eN*nQuadraturePoints_element*3 + 
				k*3 + I];
	    }/*I*/
	  vmass += adot*uQuadratureWeights_element[eN*nQuadraturePoints_element + k];
	}

      bndsum = 0.0;
      for (ebN = 0; ebN < nElementBoundaries_element; ebN++)
	{
	  for (k = 0; k < nQuadraturePoints_elementBoundary; k++)
	    {
	      xdotn = 0.0;
	      for (I = 0; I < nSpace; I++)
		{
		  xdotn +=
		    x_elementBoundary[eN*nElementBoundaries_element*nQuadraturePoints_elementBoundary*3 +
				      ebN*nQuadraturePoints_elementBoundary*3 +
				      k*3 + I]
		    *
		    n[eN*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace +
		      ebN*nQuadraturePoints_elementBoundary*nSpace +
		      k*nSpace + I];
		}
	      bndsum += u_elementBoundary[eN*nElementBoundaries_element*nQuadraturePoints_elementBoundary +
					  ebN*nQuadraturePoints_elementBoundary + k]
		* xdotn 
		* uQuadratureWeights_elementBoundary[eN*nElementBoundaries_element*nQuadraturePoints_elementBoundary +
						     ebN*nQuadraturePoints_elementBoundary + k];
	    }
	}/*ebN*/
      gravsum = 0.0;
      for (I=0; I < nSpace; I++)
	gravsum -= vecc[I]*elementBarycenters[eN*3 + I]*volume;
      rt0potential[eN] = (bndsum + vmass + gravsum)/(dDim*volume);
    }/*eN*/
}

void postProcessRT0potentialFromP1nc_sd(proteus_index_t nElements_global,
					int nQuadraturePoints_element,
					int nElementBoundaries_element,
					int nQuadraturePoints_elementBoundary,
					int nSpace,
					proteus_csr_index_t* rowptr,
					proteus_csr_index_t* colind,
					double * uQuadratureWeights_element,
					double * elementBarycenters,
					double * aElementQuadratureWeights,
					double * detJ,
					double * uQuadratureWeights_elementBoundary,
					double * x,
					double * u,
					double * gradu,
					double * x_elementBoundary,
					double * u_elementBoundary,
					double * n,
					double * a,
					double * f,
					double * r,
					double * rt0vdofs,
					double * rt0potential)
{
  /***********************************************************************
     follow example in Chen '94 paper to get RTO potential value.
     Assumes RT0 flux has already been calculated!
     
     here, we basically have to plug in the P^1_{nc} solution,
     and the postprocessed flux into the local Darcy's law expression
     from the mixed hybrid formulation using the test function
       \vec v_h= \vec x.
     Then we solve for the pressure unknown.
     
     (\mat{A}_h^{-1}\vec q_h,\vec v_h)_T  - (p_h,\div \vec v_h)_T
        + (\lambda_h,\vec v_h\cdot n_{T})_{\partial T} - (\bar{\vec c},\vec v_h)_T = 0
     
     Recall, that \lambda^{j}_h = p^j_h, the P^1_{nc} solution for edge j
    
     To evaluate the weak integrals, I'll use the quadrature formula

      \int_{T} f \dx \approx |T|/3 \sum_{j}f(\vec \bar{x}^j)
    
     for the mass matrix term, and basically exact integration (midpoint rule)
     for the boundary integrals of the Lagrange multipler term since it's
     linear on each edge, and midpoint rule for the "advection" term since it's linear
    
   ***********************************************************************/
  proteus_index_t eN, ebN; int I, J, k;
  proteus_csr_index_t m; proteus_csr_index_t nnz=rowptr[nSpace];
  int nDOF_RT0V_element = nSpace+1; /*number of dofs for vector part of RT0*/
  double volume,vmass,bndsum,adot,ahqInvI,xdotn,gravsum;
  double ah[3][3] = {{0.0,0.0,0.0},
		     {0.0,0.0,0.0},
		     {0.0,0.0,0.0}};
  double ahInv[3][3] = {{0.0,0.0,0.0},
			{0.0,0.0,0.0},
			{0.0,0.0,0.0}};
  double vecc[3]  = {0.0,0.0,0.0};

  double dDim = nSpace;
  for (eN = 0; eN < nElements_global; eN++)
    {
      volume = 0.0;
      for (k = 0; k < nQuadraturePoints_element; k++)
	{
	  volume += uQuadratureWeights_element[eN*nQuadraturePoints_element + k];
	}
      /*mwf debug 
      printf("RT0pot. volume[%d]= %g \n",eN,volume);
      */
      for (I = 0; I < nSpace; I++)
	{
	  for(m=rowptr[I];m<rowptr[I+1];m++)
	    {
	      /*need to stick to a quad points and weights to be consistent with
		original calculation*/
	      ah[I][colind[m]] = 0.0;
	      for (k = 0; k < nQuadraturePoints_element; k++)
		{
		  ah[I][colind[m]] +=
		    a[eN*nQuadraturePoints_element*nnz+
		      k*nnz+
		      m]
		    *
		    aElementQuadratureWeights[k]*fabs(detJ[eN*nQuadraturePoints_element+k]);
		}/*k*/
	      ah[I][colind[m]] = ah[I][colind[m]]/volume;
	    }/*J*/
	  vecc[I] = 0.0;
	  for (k = 0; k < nQuadraturePoints_element; k++)
	    {
	      vecc[I] +=
		f[eN*nQuadraturePoints_element*nSpace +
		  k*nSpace + I]
		*
		aElementQuadratureWeights[k]*fabs(detJ[eN*nQuadraturePoints_element+k]);
	    }/*k*/
	  vecc[I] = vecc[I]/volume;
	}/*I*/
      /*mwf debug 
	printf("RT0pot. before invertLocal ah= \n %g %g \n %g %g \n",ah[0][0],ah[0][1],
	ah[1][0],ah[1][1]);
      */
      invertLocal(nSpace,ah,ahInv);
      /*mwf debug
	printf("RT0pot. after invertLocal AhInv= \n %g %g \n %g %g \n",ahInv[0][0],ahInv[0][1],
	ahInv[1][0],ahInv[1][1]);
      */
      vmass = 0.0;
      for (k = 0; k < nQuadraturePoints_element; k++)
	{
	  adot = 0.0;
	  for (I = 0; I < nSpace; I++)
	    {
	      ahqInvI = 0.0;
	      for (J = 0; J < nSpace; J++)
		ahqInvI+= ahInv[I][J]*(rt0vdofs[eN*nDOF_RT0V_element + J]
				       +
				       rt0vdofs[eN*nDOF_RT0V_element + nSpace]
				       *
				       x[eN*nQuadraturePoints_element*3 + 
					 k*3 + J]);
	      
	      adot += ahqInvI*x[eN*nQuadraturePoints_element*3 + 
				k*3 + I];
	    }/*I*/
	  vmass += adot*uQuadratureWeights_element[eN*nQuadraturePoints_element + k];
	}

      bndsum = 0.0;
      for (ebN = 0; ebN < nElementBoundaries_element; ebN++)
	{
	  for (k = 0; k < nQuadraturePoints_elementBoundary; k++)
	    {
	      xdotn = 0.0;
	      for (I = 0; I < nSpace; I++)
		{
		  xdotn +=
		    x_elementBoundary[eN*nElementBoundaries_element*nQuadraturePoints_elementBoundary*3 +
				      ebN*nQuadraturePoints_elementBoundary*3 +
				      k*3 + I]
		    *
		    n[eN*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace +
		      ebN*nQuadraturePoints_elementBoundary*nSpace +
		      k*nSpace + I];
		}
	      bndsum += u_elementBoundary[eN*nElementBoundaries_element*nQuadraturePoints_elementBoundary +
					  ebN*nQuadraturePoints_elementBoundary + k]
		* xdotn 
		* uQuadratureWeights_elementBoundary[eN*nElementBoundaries_element*nQuadraturePoints_elementBoundary +
						     ebN*nQuadraturePoints_elementBoundary + k];
	    }
	}/*ebN*/
      gravsum = 0.0;
      for (I=0; I < nSpace; I++)
	gravsum -= vecc[I]*elementBarycenters[eN*3 + I]*volume;
      rt0potential[eN] = (bndsum + vmass + gravsum)/(dDim*volume);
    }/*eN*/
}

void projectElementBoundaryVelocityToRT0fluxRep(proteus_index_t nElements_global,
						int nElementBoundaries_element,
						int nQuadraturePoints_elementBoundary,
						int nSpace,
						double * elementBoundaryQuadratureWeights,
						double * n,
						double * v_elementBoundary,
						double * rt0vdofs_element)
{
  /***********************************************************************
    Compute local projection of velocity to RT_0, where the local basis
    representation is

      \vec N_i = \frac{1}{d|E|}(\vec x - p_i), i=0,...,d

    where p_i is the vertex across from face i, |E| is the volume of the element,
     and d is the space dimension.

    The degrees of freedom are 
      V^i = \int_{e_i}\vec v\dot n_{i}\ds

    Assumes velocity is already consistent so that the normal fluxes are
     the same for neighboring elements and that the velocity is stored
     in an ebq array of size 
       nElements x nElementBoundaries_element 
        x nQuadraturePoints_elementBoundary x nSpace
     
   Uses physical quadrature points on element boundary to calculate flux integral 
   ***********************************************************************/
  
  proteus_index_t eN, ebN; int I, k;
  double fluxsum,dotk;
  int nDOF_RT0V_element = nSpace+1;
  /*mwf debug*/
  double area;
  for (eN = 0; eN < nElements_global; eN++)
    {
      for (ebN = 0; ebN < nElementBoundaries_element; ebN++)
	{
	  fluxsum = 0.0;
	  /*mwf debug*/ 
	  area = 0.0;
	  for (k = 0; k < nQuadraturePoints_elementBoundary; k++)
	    {
	      dotk = 0.0;
	      for (I=0; I < nSpace; I++)
		{
		  dotk += 
		    v_elementBoundary[eN*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
				      ebN*nQuadraturePoints_elementBoundary*nSpace+
				      k*nSpace + I]
		    *
		    n[eN*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
		      ebN*nQuadraturePoints_elementBoundary*nSpace+
		      k*nSpace + I];
		  /*mwf debug
		  printf("getRT0 flux rep dofs v_eb[%d,%d,%d,%d]=%g ; n_eb=%g \n",eN,ebN,k,I,
			 v_elementBoundary[eN*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
					   ebN*nQuadraturePoints_elementBoundary*nSpace+
					   k*nSpace + I],
			 n[eN*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
			   ebN*nQuadraturePoints_elementBoundary*nSpace+
			   k*nSpace + I]);
		  */

		}/*I*/
	      fluxsum += dotk*elementBoundaryQuadratureWeights[eN*nElementBoundaries_element*nQuadraturePoints_elementBoundary+
							       ebN*nQuadraturePoints_elementBoundary+
							       k];
	      /*mwf debug*/
	      area += elementBoundaryQuadratureWeights[eN*nElementBoundaries_element*nQuadraturePoints_elementBoundary+
						       ebN*nQuadraturePoints_elementBoundary+
						       k];
	    }/*k*/
	  rt0vdofs_element[eN*nDOF_RT0V_element + ebN] = fluxsum;
	  /*mwf debug
	    printf("rt0vdofs[%d,%d]=%g ; area= %g \n",eN,ebN,fluxsum,area);
	  */
	}/*ebN*/
    }/*eN*/
}
void projectElementBoundaryFluxToRT0fluxRep(proteus_index_t nElements_global,
                                            int nElementBoundaries_element,
                                            int nQuadraturePoints_elementBoundary,
                                            int nDOF_RT0V_element,
                                            proteus_index_t* elementBoundaryElementsArray,
                                            proteus_index_t* elementBoundariesArray,
                                            double * elementBoundaryQuadratureWeights,
                                            double * flux_elementBoundary,
                                            double * rt0vdofs_element)
{
  /***********************************************************************
    Compute local projection of normal flux to RT_0, where the local basis
    representation is

      \vec N_i = \frac{1}{d|E|}(\vec x - p_i), i=0,...,d

    where p_i is the vertex across from face i, |E| is the volume of the element,
     and d is the space dimension.

    The degrees of freedom are 
      V^i = \int_{e_i}\vec v\dot n_{i}\ds

    Assumes velocity is already consistent so that the normal fluxes are
     the same for neighboring elements and that the velocity is stored
     in an ebq array of size 
       nElements x nElementBoundaries_element 
        x nQuadraturePoints_elementBoundary x nSpace
     
   Uses physical quadrature points on element boundary to calculate flux integral 
   ***********************************************************************/
  
  proteus_index_t eN, ebN, ebN_global; int k;
  double fluxsum,sign;
  for (eN = 0; eN < nElements_global; eN++)
    {
      for (ebN = 0; ebN < nElementBoundaries_element; ebN++)
	{
          ebN_global = elementBoundariesArray[eN*nElementBoundaries_element+ebN];
	  fluxsum = 0.0;
          sign=1.0;
          if(elementBoundaryElementsArray[2*ebN_global+1] == eN)
            sign=-1.0;
	  for (k = 0; k < nQuadraturePoints_elementBoundary; k++)
            {
              fluxsum += sign*flux_elementBoundary[ebN_global*nQuadraturePoints_elementBoundary+
                                                   k]
                *
                elementBoundaryQuadratureWeights[eN*nElementBoundaries_element*nQuadraturePoints_elementBoundary+
                                                 ebN*nQuadraturePoints_elementBoundary+
                                               k];
            }
	  rt0vdofs_element[eN*nDOF_RT0V_element + ebN] = fluxsum;
	  /*mwf debug
	    printf("rt0vdofs[%d,%d]=%g \n",eN,ebN,fluxsum);
	  */
	}/*ebN*/
    }/*eN*/
}

void getElementRT0velocityValuesFluxRep(proteus_index_t nElements_global,
					int nElementBoundaries_element,
					int nPoints_element,
					int nSpace,
					int nDetVals_element,
					double * nodeArray,
					proteus_index_t * elementNodesArray,
					double * abs_det_J,
					double * x_element,
					double * rt0vdofs_element,
					double * v_element)
{
  /***********************************************************************
    Compute \vec q_h at physical points stored on each element using the
     standard local basis representation

    On element T:
      \vec q_h = \sum^d_{i=0}V^i\vec N_{T,i}
    for 
      \vec N_{T,i} = \frac{1}{d|E|}(\vec x - p_i), i=0,...,d
      
 
    where rt0vdofs_element is logically nElements_global x nSpace+1
    assumes 
       x         stored as nElements_global \times nPoints \times 3
       v_element stored as nElements_global \times nPoints \times nSpace

   ***********************************************************************/
  proteus_index_t eN, jg; int I, k, j;
  int nDOF_RT0V_element = nSpace+1; /*number of dofs for vector part of RT0*/
  double volFact,dvolInv;
  double ddim = nSpace;
  double volume = 0.0;
  volFact = 1.0;
  if (nSpace > 1) volFact = 0.5;
  if (nSpace > 2) volFact = 1.0/6.0;
  
  for (eN = 0; eN < nElements_global; eN++)
    {
      volume = volFact*abs_det_J[eN*nDetVals_element + 0];/*assumed affine*/
      assert(volume > 0.0);
      dvolInv = 1.0/(ddim*volume);

      for (k = 0; k < nPoints_element; k++)
	{
	  for (I = 0; I < nSpace; I++)
	    {
	      v_element[eN*nPoints_element*nSpace + k*nSpace + I] = 0.0;
	      for (j = 0; j < nElementBoundaries_element; j++)
		{
		  jg = elementNodesArray[eN*nElementBoundaries_element + j];
		  v_element[eN*nPoints_element*nSpace + k*nSpace + I] += 
		    rt0vdofs_element[eN*nDOF_RT0V_element + j]
		    *
		    dvolInv
		    *(x_element[eN*nPoints_element*3 + k*3 + I]- nodeArray[jg*3 + I]);
		  /*mwf debug
		  printf("getRT0 flux rep v[%d,%d,%d]=%g \n",eN,k,I,
		  v_element[eN*nPoints_element*nSpace + k*nSpace + I]);
		  */
		}/*j*/
	    }/*I*/
	}/*k*/
    }/*eN*/
}

void getElementBoundaryRT0velocityValuesFluxRep(proteus_index_t nElements_global,
                                                int nElementBoundaries_element,
                                                int nPoints_elementBoundary,
                                                int nSpace,
                                                int nDetVals_element,
                                                double * nodeArray,
                                                proteus_index_t * elementNodesArray,
                                                double * abs_det_J,
                                                double * x_elementBoundary,
                                                double * rt0vdofs_element,
                                                double * v_elementBoundary)
{
  /***********************************************************************
    Compute \vec q_h at physical points stored on each local element boundary using the
     standard local basis representation

    On element T:
      \vec q_h = \sum^d_{i=0}V^i\vec N_{T,i}
    for 
      \vec N_{T,i} = \frac{1}{d|E|}(\vec x - p_i), i=0,...,d
      
 
    where rt0vdofs_element is logically nElements_global x nSpace+1
    assumes 
       x         stored as nElements_global \times nPoints \times 3
       v_element stored as nElements_global \times nPoints \times nSpace

   ***********************************************************************/
  proteus_index_t eN, ebN, jg; int I, k, j;
  int nDOF_RT0V_element = nSpace+1; /*number of dofs for vector part of RT0*/
  double volFact,dvolInv;
  double ddim = nSpace;
  double volume = 0.0;
  volFact = 1.0;
  if (nSpace > 1) volFact = 0.5;
  if (nSpace > 2) volFact = 1.0/6.0;
  
  for (eN = 0; eN < nElements_global; eN++)
    {
      volume = volFact*abs_det_J[eN*nDetVals_element + 0];/*assumed affine*/
      assert(volume > 0.0);
      dvolInv = 1.0/(ddim*volume);
      for (ebN = 0; ebN < nElementBoundaries_element; ebN++)
        {
          for (k = 0; k < nPoints_elementBoundary; k++)
            {
              for (I = 0; I < nSpace; I++)
                {
                  v_elementBoundary[eN*nElementBoundaries_element*nPoints_elementBoundary*nSpace + 
                                    ebN*nPoints_elementBoundary*nSpace+
                                    k*nSpace + I] = 0.0;
                  for (j = 0; j < nElementBoundaries_element; j++)
                    {
                      jg = elementNodesArray[eN*nElementBoundaries_element + j];
                      v_elementBoundary[eN*nElementBoundaries_element*nPoints_elementBoundary*nSpace + 
                                        ebN*nPoints_elementBoundary*nSpace+
                                        k*nSpace + I] += 
                        rt0vdofs_element[eN*nDOF_RT0V_element + j]
                        *
                        dvolInv
                        *(x_elementBoundary[eN*nElementBoundaries_element*nPoints_elementBoundary*3 + ebN*nPoints_elementBoundary*3+k*3 + I]- nodeArray[jg*3 + I]);
                    }/*j*/
                }/*I*/
            }/*k*/
        }/*ebN*/
    }/*eN*/
}

void getGlobalElementBoundaryRT0velocityValuesFluxRep(proteus_index_t nElementBoundaries_global,
                                                      int nPoints_elementBoundary_global,
                                                      int nSpace,
                                                      int nDetVals_element,
                                                      double * nodeArray,
                                                      proteus_index_t *elementNodesArray,
                                                      proteus_index_t *elementBoundaryElementsArray,
                                                      double * abs_det_J,
                                                      double * x_elementBoundary_global,
                                                      double * rt0vdofs_element,
                                                      double * v_elementBoundary_global)
{
  /***********************************************************************
    Compute \vec q_h at physical points stored on each local element boundary using the
     standard local basis representation

    On element T:
      \vec q_h = \sum^d_{i=0}V^i\vec N_{T,i}
    for 
      \vec N_{T,i} = \frac{1}{d|E|}(\vec x - p_i), i=0,...,d
      
 
    where rt0vdofs_element is logically nElements_global x nSpace+1
    assumes 
       x         stored as nElements_global \times nPoints \times 3
       v_element stored as nElements_global \times nPoints \times nSpace

   ***********************************************************************/
  proteus_index_t eN, ebN, jg; int I, k, j;
  int nDOF_RT0V_element = nSpace+1; /*number of dofs for vector part of RT0*/
  double volFact,dvolInv;
  double ddim = nSpace;
  double volume = 0.0;
  volFact = 1.0;
  if (nSpace > 1) volFact = 0.5;
  if (nSpace > 2) volFact = 1.0/6.0;
  
  for (ebN = 0; ebN < nElementBoundaries_global; ebN++)
    {
      eN = elementBoundaryElementsArray[ebN*2 + 0]; /*left neighbor*/
      volume = volFact*abs_det_J[eN*nDetVals_element + 0];/*assumed affine*/
      assert(volume > 0.0);
      dvolInv = 1.0/(ddim*volume);
      for (k = 0; k < nPoints_elementBoundary_global; k++)
        {
          for (I = 0; I < nSpace; I++)
            {
              v_elementBoundary_global[ebN*nPoints_elementBoundary_global*nSpace + 
                                       k*nSpace + 
                                       I] = 0.0;
              for (j = 0; j < nDOF_RT0V_element; j++)
                {
                  jg = elementNodesArray[eN*nDOF_RT0V_element + j];
                  v_elementBoundary_global[ebN*nPoints_elementBoundary_global*nSpace+k*nSpace + I] += 
                    rt0vdofs_element[eN*nDOF_RT0V_element + j]
                    *
                    dvolInv
                    *(x_elementBoundary_global[ebN*nPoints_elementBoundary_global*3+k*3 + I]- nodeArray[jg*3 + I]);
                }/*j*/
            }/*I*/
        }/*k*/
    }/*ebN*/
}
void getGlobalExteriorElementBoundaryRT0velocityValuesFluxRep(proteus_index_t nExteriorElementBoundaries_global,
							      int nPoints_elementBoundary_global,
							      int nSpace,
							      int nDetVals_element,
							      double * nodeArray,
							      proteus_index_t *elementNodesArray,
							      proteus_index_t *elementBoundaryElementsArray,
							      proteus_index_t* exteriorElementBoundariesArray,
							      double * abs_det_J,
							      double * x_ebqe,
							      double * rt0vdofs_element,
							      double * v_ebqe)
{
  /***********************************************************************
    Compute \vec q_h at physical points stored on each local element boundary using the
     standard local basis representation

    On element T:
      \vec q_h = \sum^d_{i=0}V^i\vec N_{T,i}
    for 
      \vec N_{T,i} = \frac{1}{d|E|}(\vec x - p_i), i=0,...,d
      
 
    where rt0vdofs_element is logically nElements_global x nSpace+1
    assumes 
       x         stored as nElements_global \times nPoints \times 3
       v_element stored as nElements_global \times nPoints \times nSpace

   ***********************************************************************/
  proteus_index_t eN, ebN, jg; int ebNE, I, k, j;
  int nDOF_RT0V_element = nSpace+1; /*number of dofs for vector part of RT0*/
  double volFact,dvolInv;
  double ddim = nSpace;
  double volume = 0.0;
  volFact = 1.0;
  if (nSpace > 1) volFact = 0.5;
  if (nSpace > 2) volFact = 1.0/6.0;
  for (ebNE = 0; ebNE < nExteriorElementBoundaries_global; ebNE++)
    {
      ebN = exteriorElementBoundariesArray[ebNE];
      eN = elementBoundaryElementsArray[ebN*2 + 0]; /*left neighbor*/
      volume = volFact*abs_det_J[eN*nDetVals_element + 0];/*assumed affine*/
      assert(volume > 0.0);
      dvolInv = 1.0/(ddim*volume);
      for (k = 0; k < nPoints_elementBoundary_global; k++)
        {
          for (I = 0; I < nSpace; I++)
            {
              v_ebqe[ebNE*nPoints_elementBoundary_global*nSpace + 
		     k*nSpace + 
		     I] = 0.0;
              for (j = 0; j < nDOF_RT0V_element; j++)
                {
                  jg = elementNodesArray[eN*nDOF_RT0V_element + j];
                  v_ebqe[ebNE*nPoints_elementBoundary_global*nSpace+k*nSpace + I] += 
                    rt0vdofs_element[eN*nDOF_RT0V_element + j]
                    *
                    dvolInv
                    *(x_ebqe[ebNE*nPoints_elementBoundary_global*3+k*3 + I]- nodeArray[jg*3 + I]);
                }/*j*/
            }/*I*/
        }/*k*/
    }/*ebN*/
}

void getRT0velocityValuesFluxRep_arbitraryElementMembership(proteus_index_t nElements_global,
							    int nElementBoundaries_element,
							    int nPoints,
							    int nSpace,
							    int nDetVals_element,
							    const double * nodeArray,
							    const proteus_index_t * elementNodesArray,
							    const double * abs_det_J,
							    const double * x,
							    const int * element_locations,
							    const double * rt0vdofs_element,
							    double * v_element)
{
  /***********************************************************************
    Compute \vec q_h at physical points stored on in x belonging to  element_location[x] using the
     standard local basis representation

    On element T:
      \vec q_h = \sum^d_{i=0}V^i\vec N_{T,i}
    for 
      \vec N_{T,i} = \frac{1}{d|E|}(\vec x - p_i), i=0,...,d
      
 
    where rt0vdofs_element is logically nElements_global x nSpace+1
    assumes 
       x         stored as nElements_global \times nPoints \times 3
       v_element stored as nElements_global \times nPoints \times nSpace

   ***********************************************************************/
  proteus_index_t eN, jg; int I, k, j;
  int nDOF_RT0V_element = nSpace+1; /*number of dofs for vector part of RT0*/
  double volFact,dvolInv;
  double ddim = nSpace;
  double volume = 0.0;
  volFact = 1.0;
  if (nSpace > 1) volFact = 0.5;
  if (nSpace > 2) volFact = 1.0/6.0;
  for (k = 0; k < nPoints; k++)
    {
      eN = element_locations[k];
      assert(0 <= eN && eN < nElements_global);
      volume = volFact*abs_det_J[eN*nDetVals_element + 0];/*assumed affine*/
      assert(volume > 0.0);
      dvolInv = 1.0/(ddim*volume);
      for (I = 0; I < nSpace; I++)
	{
	  v_element[k*nSpace + I] = 0.0;
	  for (j = 0; j < nElementBoundaries_element; j++)
	    {
	      jg = elementNodesArray[eN*nElementBoundaries_element + j];
	      v_element[k*nSpace + I] += 
		rt0vdofs_element[eN*nDOF_RT0V_element + j]
		*
		dvolInv
		*(x[k*3 + I]- nodeArray[jg*3 + I]);
	      /*mwf debug
		printf("getRT0 flux rep v[%d,%d,%d]=%g \n",eN,k,I,
		v_element[eN*nPoints_element*nSpace + k*nSpace + I]);
	      */
	    }/*j*/
	}/*I*/
    }/*k*/
}

void buildLocalBDM1projectionMatrices_orig(proteus_index_t nElements_global,
					   int nElementBoundaries_element,
					   int nQuadraturePoints_elementBoundary,
					   int nSpace,
					   int nDOFs_test_element,
					   int nDOFs_trial_element,
					   int nVDOFs_element,
					   double * w_dS_f,
					   double * ebq_n,
					   double * ebq_v,
					   double * BDMprojectionMat_element)
{
  /***********************************************************************
    loop through and build local \f$BDM_1\f$ projection representation for
    each element for a simplicial mesh in 2d or 3d. Involves
    integration over each face, e, of normal flux weighted by basis
    for \f$P^1(e)\f$. Local velocity space is \f$P^1(E)\f$
   \f{eqnarray}
      P_{ij} &=& \int_{ebN} w_{s} \vec N_j \cdot \vec n_{ebN}\ds \\

       i &=& ebN*nd + s (\mbox{local test function index}) \\
     ebN &=& 0,\ldots,nd    (\mbox{local element boundaries}) \\
       s &=& 0,\ldots,nd   (\mbox{index into local basis for} P^1(e)) \\
       j &=& 0,\ldots,nd*(nd+1)-1 \mbox{index for local basis functions} \\
\f{eqnarray}

    Assumes local basis functions are defined as
\f]
      \vec N_j = \lambda_k \vec e_l  where k = j % nd+1, l = j/(nd+1)
\f]
    \f$\lambda_k\f$ is barycentric coordinate associated with node \f$k\f$ 
    \f$\vec e_l\f$ is coordinate vector for axis \f$l\f$.

     Assumes nodes \f$k\f$ is corresponds to face \f$k \f$across from it.
   ***********************************************************************/

  proteus_index_t eN, ebN; proteus_index_t irow; int s, j, k, l, kp, ibq, nVDOFs_element2, nSimplex;
  int TRANSPOSE_FOR_LAPACK=1;
  double pval;
  nSimplex = nSpace+1;
  assert(nVDOFs_element == nSpace*(nSpace+1));
  assert(nSimplex == nDOFs_trial_element);
  assert(nSimplex == nDOFs_test_element);
  nVDOFs_element2 = nVDOFs_element*nVDOFs_element;
  /*mwf debug
    printf("build local BDM nE= %d nEb=%d nBq=%d nvd=%d nd=%d\n",
    nElements_global,nElementBoundaries_element,nQuadraturePoints_elementBoundary,
    nVDOFs_element,nSpace);
  */

  for (eN=0; eN < nElements_global; eN++)
    {
      for (ebN = 0; ebN < nElementBoundaries_element; ebN++)
	{
	  for (s = 0; s < nSpace; s++)
	    {
	      irow = ebN*nSpace + s;
	      for (j = 0; j < nVDOFs_element; j++)
		{
		  k = j % nSimplex;
		  l = j/nSimplex;
		  kp= (ebN+s+1) % nSimplex; /*neighbor (s) of node k*/
		  /*mwf debug
		    printf("eN=%d ebN=%d s=%d irow=%d j=%d k=%d l=%d kp=%d\n",
		    eN,ebN,s,irow,j,k,l,kp);
		  */
		  if (TRANSPOSE_FOR_LAPACK > 0)
		    BDMprojectionMat_element[eN*nVDOFs_element2 + irow + j*nVDOFs_element] = 0.0;
		  else
		    BDMprojectionMat_element[eN*nVDOFs_element2 + irow*nVDOFs_element + j] = 0.0;
		  for (ibq = 0; ibq < nQuadraturePoints_elementBoundary; ibq++)
		    {
		      
		      pval = 
			ebq_n[eN*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace +
			      ebN*nQuadraturePoints_elementBoundary*nSpace+
			      ibq*nSpace+
			      l]
			*
			w_dS_f[eN*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOFs_test_element+
			       ebN*nQuadraturePoints_elementBoundary*nDOFs_test_element+
			       ibq*nDOFs_test_element+
			       kp]
			*
			ebq_v[eN*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOFs_trial_element+
			      ebN*nQuadraturePoints_elementBoundary*nDOFs_trial_element+
			      ibq*nDOFs_trial_element+
			      k];
		      if (TRANSPOSE_FOR_LAPACK > 0)
			BDMprojectionMat_element[eN*nVDOFs_element2 + irow + j*nVDOFs_element] += pval;
		      else
			BDMprojectionMat_element[eN*nVDOFs_element2 + irow*nVDOFs_element + j] += pval;
		      
		    }/*ibq*/
		}/*j*/
	    }/*s*/
	}/*ebN*/

    }/*eN*/

}

void buildLocalBDM1projectionMatrices(proteus_index_t nElements_global,
				      int nElementBoundaries_element,
				      int nQuadraturePoints_elementBoundary,
				      int nSpace,
				      int nDOFs_test_element,
				      int nDOFs_trial_element,
				      int nVDOFs_element,
				      double * w_dS_f,
				      double * ebq_n,
				      double * ebq_v,
				      double * BDMprojectionMat_element)
{
  /***********************************************************************

Input Variables

     nElements_global - Number of elements in triangulation

     nElementBoundaries_element - Number of boundaries per element for
     2D triangles this is 3, quarilateral - 4 etc.

     nQuadraturePoints_elementBoundary - This is the number of quadrature
     points taken along the boundary.  This value is typically set in 
     the numerics file with a flag like quad_order.

     nSpace - dimension of the problem (typically 2 or 3)

     nDOFs_test_element - 
 
     nDOFs_trial_element - 

     nVDOFs_element - number of velocity DoF per element

  loop through and build local \f$BDM_1\f$ projection representation for
    each element for a simplicial mesh in 2d or 3d. Involves
    integration over each face, e, of normal flux weighted by basis
    for \f$P^1(e)\f$. Local velocity space is \f$P^1(E)\f$
   \f{eqnarray}
      P_{ij} &=& \int_{ebN} w_{s} \vec N_j \cdot \vec n_{ebN}\ds \\

       i &=& ebN*nd + s (\mbox{local test function index}) \\
     ebN &=& 0,\ldots,nd    (\mbox{local element boundaries}) \\
       s &=& 0,\ldots,nd-1   (\mbox{index into local basis for} P^1(e)) \\
       j &=& 0,\ldots,nd*(nd+1)-1 \mbox{index for local basis functions} \\
\f{eqnarray}

    Assumes local basis functions are defined as
\f]
      \vec N_j = \lambda_k \vec e_l  where k = j / nd, l = j % nd
\f]
    \f$\lambda_k\f$ is barycentric coordinate associated with node \f$k\f$ 
    \f$\vec e_l\f$ is coordinate vector for axis \f$l\f$.

     Assumes nodes \f$k\f$ is corresponds to face \f$k \f$across from it.
   ***********************************************************************/

  proteus_index_t eN, ebN; proteus_index_t irow; int s, j, k, l, kp, ibq, nVDOFs_element2, nSimplex;
  int TRANSPOSE_FOR_LAPACK=1;
  double pval;
  nSimplex = nSpace+1;
  assert(nVDOFs_element == nSpace*(nSpace+1));
  assert(nSimplex == nDOFs_trial_element);
  assert(nSimplex == nDOFs_test_element);
  nVDOFs_element2 = nVDOFs_element*nVDOFs_element;
  /*mwf debug
    printf("build local BDM nE= %d nEb=%d nBq=%d nvd=%d nd=%d\n",
    nElements_global,nElementBoundaries_element,nQuadraturePoints_elementBoundary,
    nVDOFs_element,nSpace);
  */

  /* printf("BDM1 test information: \n"); */
  /* printf("nVDOFs_element: %d\n",nVDOFs_element); */

  for (eN=0; eN < nElements_global; eN++)
    {
      for (ebN = 0; ebN < nElementBoundaries_element; ebN++)
	{
	  for (s = 0; s < nSpace; s++)
	    {
	      irow = ebN*nSpace + s;
	      for (j = 0; j < nVDOFs_element; j++)
		{
		  k = j / nSpace;
		  l = j % nSpace;
		  kp= (ebN+s+1) % nSimplex; /*neighbor (s) of node k*/
		  		   /*mwf debug
		  printf("BDM1 new eN=%d ebN=%d s=%d irow=%d j=%d k=%d l=%d kp=%d\n",
			 eN,ebN,s,irow,j,k,l,kp);
		  printf("BDMprojectionMat_element: %d\n", eN*nVDOFs_element2 + irow + j*nVDOFs_element);
		  printf("nQuadraturePoints_elementBoundary: %d \n", nQuadraturePoints_elementBoundary);
		   */

		  if (TRANSPOSE_FOR_LAPACK > 0)
		    BDMprojectionMat_element[eN*nVDOFs_element2 + irow + j*nVDOFs_element] = 0.0;
		  else
		    BDMprojectionMat_element[eN*nVDOFs_element2 + irow*nVDOFs_element + j] = 0.0;
		  for (ibq = 0; ibq < nQuadraturePoints_elementBoundary; ibq++)
		    {

		      pval = 
			ebq_n[eN*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace +
			      ebN*nQuadraturePoints_elementBoundary*nSpace+
			      ibq*nSpace+
			      l]
			*
			w_dS_f[eN*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOFs_test_element+
			       ebN*nQuadraturePoints_elementBoundary*nDOFs_test_element+
			       ibq*nDOFs_test_element+
			       kp]
			*
			ebq_v[eN*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOFs_trial_element+
			      ebN*nQuadraturePoints_elementBoundary*nDOFs_trial_element+
			      ibq*nDOFs_trial_element+
			      k];
		      if (TRANSPOSE_FOR_LAPACK > 0){
			BDMprojectionMat_element[eN*nVDOFs_element2 + irow + j*nVDOFs_element] += pval;
		      }
		      else
			BDMprojectionMat_element[eN*nVDOFs_element2 + irow*nVDOFs_element + j] += pval;
		      
		    }/*ibq*/
		  /*
		  if (TRANSPOSE_FOR_LAPACK > 0)
		    printf("BDM1 new  eN=%d B(%d,%d)=%g\n",
			   eN,irow,j,BDMprojectionMat_element[eN*nVDOFs_element2 + irow + j*nVDOFs_element]);
		  else
		    printf("BDM1 new eN=%d B(%d,%d)=%g\n",
			   eN,irow,j,BDMprojectionMat_element[eN*nVDOFs_element2 + irow*nVDOFs_element + j]);
		  */
		}/*j*/
	    }/*s*/
	}/*ebN*/

    }/*eN*/

}


void buildLocalBDM2projectionMatrices(int degree,
				      proteus_index_t nElements_global,
				      int nElementBoundaries_element,
				      int nQuadraturePoints_elementBoundary,
				      int nQuadraturePoints_elementInterior,
				      int nSpace,
				      int nDOFs_test_element,
				      int nDOFs_trial_boundary_element,
				      int nDOFs_trial_interior_element,
				      int nVDOFs_element,
				      int *edgeFlags,
				      double * w_dS_f,
				      double * ebq_n,
				      double * ebq_v,
				      double * BDMprojectionMat_element,
				      double * q_basis_vals,
				      double * w_int_test_grads,
				      double * w_int_div_free,
				      double * piola_trial_fun)
{
  /***********************************************************************
   This function builds the LocalBDM2projectionMatrices.  This includes
   three boundary integrals per edge and three interior degrees of freedom.
   NOTE - this function has only been tested for TRANSPOSE_FOR_LAPACK=1.
   ***********************************************************************/

  proteus_index_t eN, ebN; proteus_index_t irow; int s, i, j, k, l, kp, ibq, nSimplex;
  int dof, dof_edge, boundary_dof;
  int num_div_free;
  int TRANSPOSE_FOR_LAPACK=1;
  double pval, pvalx, pvaly;
  nSimplex = nSpace+1;
  assert(degree == 2);

  int interiorPspace = nDOFs_trial_interior_element;
  
  if (nSpace == 2){
    dof = (degree+1)*(degree+2);
    dof_edge = degree + 1;
    boundary_dof = nElementBoundaries_element*dof_edge;
    num_div_free = 1;
  }
  else if (nSpace == 3){
    dof = (degree+1)*(degree+2)*(degree+3) / 2;
    dof_edge = degree*(degree+1);
    boundary_dof = nElementBoundaries_element*dof_edge;
    num_div_free = 3;
  }
  else {
    assert(1 == 0);
  }
  
  int interior_dof = dof - boundary_dof;
  // Begin populating the projection matrix

  // Loop over elements of the triangulation
  for (eN=0; eN < nElements_global; eN++)
    {
      // Boundary Integrals

      for (ebN = 0; ebN < nElementBoundaries_element; ebN++)
	{
	  for (s = 0; s < dof_edge; s++)
	    {
	      irow = ebN*dof_edge + s;

	      for (j = 0; j < dof; j++)
		{

		  k = j / nSpace;
		  l = j % nSpace; 
		  kp = edgeFlags[ebN*dof_edge+s];

		  if (TRANSPOSE_FOR_LAPACK > 0)
		    BDMprojectionMat_element[eN*dof*dof + irow + j*nVDOFs_element] = 0.;
		  else
		    BDMprojectionMat_element[eN*dof*dof + irow*nVDOFs_element + j] = 0.;

		  for (ibq = 0; ibq < nQuadraturePoints_elementBoundary; ibq++)
		    {

		      pval =
		      	ebq_n[eN*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace +
		      	      ebN*nQuadraturePoints_elementBoundary*nSpace+
		      	      ibq*nSpace+
		      	      l]
		      	*
		      	w_dS_f[eN*nElementBoundaries_element*nQuadraturePoints_elementBoundary* nDOFs_test_element+
		      	       ebN*nQuadraturePoints_elementBoundary*nDOFs_test_element+
		      	       ibq*nDOFs_test_element+
		      	       kp]
		      	*
		      	ebq_v[eN*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOFs_trial_boundary_element+
		      	      ebN*nQuadraturePoints_elementBoundary*nDOFs_trial_boundary_element+
		      	      ibq*nDOFs_trial_boundary_element+
		      	      k];


		      if (TRANSPOSE_FOR_LAPACK > 0)
		      	BDMprojectionMat_element[eN*dof*dof + irow + j*nVDOFs_element] += pval;
		      else
		      	BDMprojectionMat_element[eN*dof*dof + irow*nVDOFs_element + j] += pval;
		      
		    }/*ibq*/
		}/*j*/
	    }/*s*/
	}/*ebN*/

      // **** Interior Integrals (two interior integrals come from gradients, one comes from div-free element) ****

      // **** Gradient Integrals ****      
      for (s = 0; s < interiorPspace-1; s++){
      	// Iterate over interior polynomial test space
      	irow = boundary_dof + s;
      	for (j=0; j < (dof/nSpace); j++){

      	// Iterate over trial functions
      	  if (TRANSPOSE_FOR_LAPACK > 0){
	    for (i = 0; i < nSpace; i++){
      	    BDMprojectionMat_element[eN*dof*dof + irow + j*dof*nSpace + i*dof] = 0.0;
	    }
	    
      	  }
      	  else {
	    for (i = 0; i < nSpace; i++){
      	    BDMprojectionMat_element[eN*dof*dof + irow*nVDOFs_element + j + i] = 0.0;
	    }
	  }

      	  for (ibq=0; ibq < nQuadraturePoints_elementInterior; ibq++){
      	  // Iterate over quadrature points

	    for (i = 0; i < nSpace; i++){

                if (TRANSPOSE_FOR_LAPACK > 0){

		  BDMprojectionMat_element[eN*dof*dof + irow + j*dof*nSpace + i*dof] +=

		    q_basis_vals[eN* (dof/nSpace) *nQuadraturePoints_elementInterior +
				 ibq*nDOFs_test_element +
				 j]
      		  * w_int_test_grads[eN*nSpace*interiorPspace*nQuadraturePoints_elementInterior+
				     s*nSpace +
				     ibq* nSpace *interiorPspace + i];  
		}
		
		else {
		  BDMprojectionMat_element[eN*dof*dof + irow*nVDOFs_element + j] += pval;
		  BDMprojectionMat_element[eN*dof*dof + irow*nVDOFs_element + j + 1] += pval;
		}

	    }
	
      	  }  /* end ibq */
        }   /* end j*/
      }    /* end s */

      /* // **** DIV-FREE Integrals **** */
      irow = boundary_dof + interiorPspace - 1;
      for (j=0; j < dof; j++){

      	if (TRANSPOSE_FOR_LAPACK > 0){
	  for (s = 0; s < num_div_free; s++){
	    BDMprojectionMat_element[eN*dof*dof + (irow + s) + j*nVDOFs_element] = 0.0;
	  }
      	}
      	else
          BDMprojectionMat_element[eN*dof*dof + irow*nVDOFs_element + j] = 0.0;

      	for (ibq=0; ibq<nQuadraturePoints_elementInterior; ibq++)
      	{

	  if (TRANSPOSE_FOR_LAPACK > 0){
	    for(s = 0; s < num_div_free; s++){
	       for (i=0 ; i < nSpace; i++){
		 BDMprojectionMat_element[eN*dof*dof + (irow + s) + j*nVDOFs_element] +=

		   w_int_div_free[eN * nQuadraturePoints_elementInterior * nSpace * num_div_free  +
				  ibq * nSpace * num_div_free  +
				  i * num_div_free +
				  s]
		   
		   * piola_trial_fun[eN * nQuadraturePoints_elementInterior * dof * nSpace  +
				     ibq * dof * nSpace +
				     j * nSpace +
				     i] ;
	       }
	    }
	  }
	    else
	      BDMprojectionMat_element[eN*dof*dof + irow*nVDOFs_element + j] = 0.0;
      	}
      }
    }/*eN*/
}


void factorLocalBDM1projectionMatrices(proteus_index_t nElements_global,
				       int nVDOFs_element,
				       double *BDMprojectionMat_element,
				       int *BDMprojectionMatPivots_element)
{
  PROTEUS_LAPACK_INTEGER INFO=0;
  proteus_index_t eN; int i, nVDOFs_element2;
  PROTEUS_LAPACK_INTEGER pivots_element[12]; /*maximum size for local space is 3*(3+1)*/
  PROTEUS_LAPACK_INTEGER nE_n = ((PROTEUS_LAPACK_INTEGER) nVDOFs_element);
  nVDOFs_element2 = nVDOFs_element*nVDOFs_element;
  for (eN = 0; eN < nElements_global; eN++)
    {
      dgetrf_(&nE_n,
              &nE_n,
              &BDMprojectionMat_element[eN*nVDOFs_element2],
              &nE_n,
              &pivots_element[0],
              &INFO);
      //     /*mwf debug
      //	printf("factor local BDM eN=%d INFO=%d",eN,INFO);
	// */
      for (i = 0; i < nVDOFs_element; i++)
	  BDMprojectionMatPivots_element[eN*nVDOFs_element+i] = (int) pivots_element[i];
	  
    }

}


void factorLocalBDM2projectionMatrices(proteus_index_t nElements_global,
				       int nVDOFs_element,
				       double *BDMprojectionMat_element,
				       int *BDMprojectionMatPivots_element)
{
  PROTEUS_LAPACK_INTEGER INFO=0;
  proteus_index_t eN; int i, nVDOFs_element2;
  PROTEUS_LAPACK_INTEGER pivots_element[30]; /*maximum size for local space is 12*/
  PROTEUS_LAPACK_INTEGER nE_n = ((PROTEUS_LAPACK_INTEGER) nVDOFs_element);
  nVDOFs_element2 = nVDOFs_element*nVDOFs_element;
  for (eN = 0; eN < nElements_global; eN++)
    {
      dgetrf_(&nE_n,
              &nE_n,
              &BDMprojectionMat_element[eN*nVDOFs_element2],
              &nE_n,
              &pivots_element[0],
              &INFO);
     
      //	printf("factor local BDM eN=%d INFO=%d\n",eN,INFO);
     
      for (i = 0; i < nVDOFs_element; i++)
	  BDMprojectionMatPivots_element[eN*nVDOFs_element+i] = (int) pivots_element[i];
	  
    }

}


void solveLocalBDM1projection(proteus_index_t nElements_global,
			      int nElementBoundaries_element,
			      int nQuadraturePoints_elementBoundary,
			      int nSpace,
			      int nDOFs_test_element,
			      int nVDOFs_element,
			      double * BDMprojectionMatFact_element,
			      int* BDMprojectionMatPivots_element,
			      double * w_dS_f,
			      double * ebq_n,
			      double * ebq_velocity,
			      double * p1_velocity_dofs)
{
  /***********************************************************************
     build right hand side for projection to BDM1 space and then solve
     the local projection system. 

     Assumes ebq_velocity holds the velocity values at element boundaries
     that will be used in the projection. \f$w_dS_f\f$ holds test function values
     times surface quadrature weights. The test functions are technically
     defined for \f$P^1(E)\f$ but take advantage of the fact that \f$ k+1,k+2 (\mod d+1)\f$
     for basis for \f$P^1(e_k)\f$ where \f$e_k\f$ is the face across from node \f$k\f$. 

  
     Also assumes local projection has been factored and 
      BDMprojectionMatPivots holds the pivots.


   ***********************************************************************/

  PROTEUS_LAPACK_INTEGER INFO=0,NRHS=1;
  char TRANS='N';
  proteus_index_t eN, ebN; proteus_index_t irow; int s, kp, ibq, J, nSimplex, nVDOFs_element2;
  double btmp;
  PROTEUS_LAPACK_INTEGER pivots_element[12]; /*maximum size for local space is 3*(3+1)*/
  PROTEUS_LAPACK_INTEGER nE_n = ((PROTEUS_LAPACK_INTEGER) nVDOFs_element);

  nSimplex = nSpace+1;
  assert(nVDOFs_element == nSpace*(nSpace+1));
  assert(nSimplex == nDOFs_test_element);
  assert(BDMprojectionMatPivots_element);
  nVDOFs_element2 = nVDOFs_element*nVDOFs_element;

  for (eN = 0; eN < nElements_global; eN++)
    {
      for (ebN = 0; ebN < nElementBoundaries_element; ebN++)
	{
	  for (s = 0; s < nSpace; s++)
	    {
	      irow = ebN*nSpace + s;
	      kp = (ebN+s+1) % nSimplex;
	      btmp = 0.0;
	      for (ibq = 0; ibq < nQuadraturePoints_elementBoundary; ibq++)
		{
		  for (J = 0; J < nSpace; J++)
		    {
		      btmp += 
			ebq_n[eN*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
			      ebN*nQuadraturePoints_elementBoundary*nSpace+
			      ibq*nSpace+
			      J]
			*
			ebq_velocity[eN*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
				     ebN*nQuadraturePoints_elementBoundary*nSpace+
				     ibq*nSpace+
				     J]
			* w_dS_f[eN*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOFs_test_element+
				 ebN*nQuadraturePoints_elementBoundary*nDOFs_test_element+
				 ibq*nDOFs_test_element+
				 kp];
		    }/*J*/
		}/*ibq*/
	      p1_velocity_dofs[eN*nVDOFs_element+irow] = btmp;
	    }
	}/*ebN*/
      for (irow = 0; irow < nVDOFs_element; irow++)
	pivots_element[irow] = BDMprojectionMatPivots_element[eN*nVDOFs_element+irow];
      dgetrs_(&TRANS,
	      &nE_n,
	      &NRHS,
	      &BDMprojectionMatFact_element[eN*nVDOFs_element2],
	      &nE_n,
	      &pivots_element[0],
	      &p1_velocity_dofs[eN*nVDOFs_element],
	      &nE_n,
	      &INFO);

    }/*eN*/
}

void buildBDM2rhs(proteus_index_t nElements_global,
                  int nElementBoundaries_element,
	          int nQuadraturePoints_elementBoundary,
		  int nQuadraturePoints_elementInterior,
	          int nSpace,
	          int nDOFs_test_element,
	          int nVDOFs_element,
		  int nDOFs_trial_interior_element,
	          double * BDMprojectionMatFact_element,
	          int* BDMprojectionMatPivots_element,
		  int *edgeFlags,
	          double * w_dS_f,
	          double * ebq_n,
		  double * w_interior_grads,
		  double * w_interior_divfree,
	          double * ebq_velocity,
		  double * q_velocity,
	          double * p1_velocity_dofs)
{
  /***********************************************************************
     NOTE - *b represents the constructed righthand side

     build right hand side for projection to BDM2 space and then solve
     the local projection system. 

     Assumes ebq_velocity holds the velocity values at element boundaries
     that will be used in the projection. \f$w_dS_f\f$ holds test function values
     times surface quadrature weights. The test functions are technically
     defined for \f$P^1(E)\f$ but take advantage of the fact that \f$ k+1,k+2 (\mod d+1)\f$
     for basis for \f$P^1(e_k)\f$ where \f$e_k\f$ is the face across from node \f$k\f$. 

  
     Also assumes local projection has been factored and 
      BDMprojectionMatPivots holds the pivots.


   ***********************************************************************/

  PROTEUS_LAPACK_INTEGER INFO=0,NRHS=1;
  char TRANS='N';
  proteus_index_t eN, ebN; proteus_index_t irow; int s, kp, ibq, j, dof_edge, num_div_free, boundary_dof, dof;
  double btmp,pvalx,pvaly;
  PROTEUS_LAPACK_INTEGER pivots_element[30]; /*maximum size for local space is ???*/
  PROTEUS_LAPACK_INTEGER nE_n = ((PROTEUS_LAPACK_INTEGER) nVDOFs_element);

  // temporary variables...these will be added as inputs
  int degree = 2;
  int interiorPspace = nDOFs_trial_interior_element;

  if (nSpace == 2){
      dof = (degree+1)*(degree+2);
      dof_edge = degree + 1;
      boundary_dof = nElementBoundaries_element*dof_edge;
      num_div_free = 1;
    }
    else if (nSpace == 3){
      dof = (degree+1)*(degree+2)*(degree+3) / 2;
      dof_edge = degree*(degree+1);
      boundary_dof = nElementBoundaries_element*dof_edge;
      num_div_free = 3;
    }
    else {
      assert(1 == 0);
    }

  assert(nVDOFs_element == dof);
  assert(BDMprojectionMatPivots_element);

  for (eN = 0; eN < nElements_global; eN++)
    {
      // Boundary Integrals
      for (ebN = 0; ebN < nElementBoundaries_element; ebN++)
  	{
  	  for (s = 0; s < dof_edge; s++)
  	    {
  	      irow = ebN*dof_edge + s;
	      kp = edgeFlags[ebN*dof_edge + s];
  	      btmp = 0.0;

  	      for (ibq = 0; ibq < nQuadraturePoints_elementBoundary; ibq++)
  		{
  		  for (j = 0; j < nSpace; j++) {

  		      btmp +=
  			ebq_n[eN*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
  			      ebN*nQuadraturePoints_elementBoundary*nSpace+
  			      ibq*nSpace+
  			      j]
  			*
  			ebq_velocity[eN*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
  				     ebN*nQuadraturePoints_elementBoundary*nSpace+
  				     ibq*nSpace+
  				     j]
  			* w_dS_f[eN*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOFs_test_element+
  				 ebN*nQuadraturePoints_elementBoundary*nDOFs_test_element+
  				 ibq*nDOFs_test_element+
  				 kp];

		  }/*J*/
		}/*ibq*/
  	      p1_velocity_dofs[eN*nVDOFs_element+irow] = btmp;
	    }
	}/*ebN*/
      
      // interior gradient integrals

      for (s = 0; s < interiorPspace-1; s++){
	// Iterate over interior polynomial test space      
	  irow = boundary_dof + s;
	  p1_velocity_dofs[eN*nVDOFs_element+irow] = 0. ;
	
	  for (ibq=0; ibq < nQuadraturePoints_elementInterior; ibq++){
	  // Iterate over quadrature points

	    for (j = 0 ; j < nSpace; j++){

	    p1_velocity_dofs[eN*nVDOFs_element+irow] +=
	      
	      q_velocity[eN*nQuadraturePoints_elementInterior*nSpace +
			 ibq*nSpace +
			 j] *
	      w_interior_grads[eN*nSpace*interiorPspace*nQuadraturePoints_elementInterior +
			       s* nSpace +
			       ibq * nSpace * interiorPspace + 
			       j];
	      }
	    	
	  }  /* end ibq */
      }    /* end s */

      // div free elements

      irow = boundary_dof + interiorPspace - 1;
      for (s = 0 ; s < num_div_free ; s++){
	p1_velocity_dofs[eN*nVDOFs_element + (irow + s) ] = 0.0 ;
      }
      
      for (ibq = 0; ibq < nQuadraturePoints_elementInterior; ibq++){
	for (s = 0 ; s < num_div_free ; s++){
	  for (j = 0 ; j < nSpace ; j++){

	    p1_velocity_dofs[eN*nVDOFs_element + (irow + s) ] += 
	
	      q_velocity[eN * nQuadraturePoints_elementInterior * nSpace +
			 ibq * nSpace + 
			 j ] *
	      
	      w_interior_divfree[eN * nQuadraturePoints_elementInterior * num_div_free * nSpace+
				 ibq * num_div_free * nSpace +
				 j * num_div_free +
				 s ];
	 }
	}
      }
          
    }/*eN*/

}

void solveLocalBDM2projection(proteus_index_t nElements_global,
			      int nElementBoundaries_element,
			      int nQuadraturePoints_elementBoundary,
			      int nSpace,
			      int nDOFs_test_element,
			      int nVDOFs_element,
			      double * BDMprojectionMatFact_element,
			      int* BDMprojectionMatPivots_element,
			      double * w_dS_f,
			      double * ebq_n,
			      double * w_interior_gradients,
			      double * q_velocity,
			      double * ebq_velocity,
			      double * p1_velocity_dofs)
{
  /***********************************************************************
   Solve the BDM2 projection and save answer in p1_velocity_dofs vector.

   ***********************************************************************/

  PROTEUS_LAPACK_INTEGER INFO=0,NRHS=1;
  char TRANS='N';
  proteus_index_t eN, ebN; proteus_index_t irow; int s, kp, ibq, J, nSimplex, nVDOFs_element2;
  double btmp;
  PROTEUS_LAPACK_INTEGER pivots_element[30];
  PROTEUS_LAPACK_INTEGER nE_n = ((PROTEUS_LAPACK_INTEGER) nVDOFs_element);

  for (eN = 0; eN < nElements_global; eN++)
    {
  
      for (irow = 0; irow < nVDOFs_element; irow++)
  	pivots_element[irow] = BDMprojectionMatPivots_element[eN*nVDOFs_element+irow];
      
      dgetrs_(&TRANS,
  	      &nE_n,
  	      &NRHS,
  	      &BDMprojectionMatFact_element[eN*nVDOFs_element*nVDOFs_element],
  	      &nE_n,
  	      &pivots_element[0],
  	      &p1_velocity_dofs[eN*nVDOFs_element],
  	      &nE_n,
  	      &INFO);

    }/*eN*/

}


void solveLocalBDM1projectionFromFlux(proteus_index_t nElements_global,
                                      int nElementBoundaries_element,
                                      int nQuadraturePoints_elementBoundary,
                                      int nDOFs_test_element,
                                      int nVDOFs_element,
                                      double * BDMprojectionMatFact_element,
                                      int* BDMprojectionMatPivots_element,
                                      proteus_index_t* elementBoundaryElementsArray,
                                      proteus_index_t* elementBoundariesArray,
                                      double * w_dS_f,
                                      double * ebq_global_flux,
                                      double * p1_velocity_dofs)
{
  /***********************************************************************
     build right hand side for projection to BDM1 space and then solve
     the local projection system. 

     Assumes ebq_velocity holds the velocity values at element boundaries
     that will be used in the projection. \f$w_dS_f\f$ holds test function values
     times surface quadrature weights. The test functions are technically
     defined for \f$P^1(E)\f$ but take advantage of the fact that \f$ k+1,k+2 (\mod d+1)\f$
     for basis for \f$P^1(e_k)\f$ where \f$e_k\f$ is the face across from node \f$k\f$. 

  
     Also assumes local projection has been factored and 
      BDMprojectionMatPivots holds the pivots.


   ***********************************************************************/
  
  PROTEUS_LAPACK_INTEGER INFO=0,NRHS=1;
  char TRANS='N';
  proteus_index_t eN, ebN, ebN_global; proteus_index_t irow; int nSpace, s, kp, ibq, nSimplex, nVDOFs_element2;
  double btmp,sign;
  PROTEUS_LAPACK_INTEGER pivots_element[12]; /*maximum size for local space is 3*(3+1)*/
  PROTEUS_LAPACK_INTEGER nE_n = ((PROTEUS_LAPACK_INTEGER) nVDOFs_element);
  
  nSimplex = nDOFs_test_element;
  nSpace = nSimplex - 1;
  assert(nVDOFs_element == nSpace*(nSpace+1));
  assert(nSimplex == nDOFs_test_element);
  assert(BDMprojectionMatPivots_element);
  nVDOFs_element2 = nVDOFs_element*nVDOFs_element;

  for (eN = 0; eN < nElements_global; eN++)
    {
      for (ebN = 0; ebN < nElementBoundaries_element; ebN++)
	{
          ebN_global = elementBoundariesArray[eN*nElementBoundaries_element+ebN];
          sign=1.0;
          if(elementBoundaryElementsArray[2*ebN_global+1] == eN)
            sign=-1.0;
	  for (s = 0; s < nSpace; s++) /* nSpace test functions per face; s = nSpace wrote one row past the element */
	    {
	      irow = ebN*nSpace + s;
	      kp = (ebN+s+1) % nSimplex;
	      btmp = 0.0;
	      for (ibq = 0; ibq < nQuadraturePoints_elementBoundary; ibq++)
		{
                  btmp += sign*ebq_global_flux[ebN_global*nQuadraturePoints_elementBoundary+ibq]
                    *
                    w_dS_f[eN*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOFs_test_element+
                           ebN*nQuadraturePoints_elementBoundary*nDOFs_test_element+
                           ibq*nDOFs_test_element+
                           kp];
		}/*ibq*/
	      p1_velocity_dofs[eN*nVDOFs_element+irow] = btmp;
	    }
	}/*ebN*/
      for (irow = 0; irow < nVDOFs_element; irow++)
	pivots_element[irow] = BDMprojectionMatPivots_element[eN*nVDOFs_element+irow];
      dgetrs_(&TRANS,
	      &nE_n,
	      &NRHS,
	      &BDMprojectionMatFact_element[eN*nVDOFs_element2],
	      &nE_n,
	      &pivots_element[0],
	      &p1_velocity_dofs[eN*nVDOFs_element],
	      &nE_n,
	      &INFO);

    }/*eN*/
}

void getElementBDM1velocityValuesLagrangeRep_orig(proteus_index_t nElements_global,
						  int nQuadraturePoints_element,
						  int nSpace,
						  int nDOF_trial_element,
						  int nVDOF_element,
						  double * q_v, /*scalar P^1 shape fncts*/
						  double * p1_velocity_dofs,
						  double * q_velocity)
{
  /***********************************************************************
    Assumes local representation for \f$[P^1(E)]^d\f$ is
\f[
    \vec N_j= \lambda_k \vec e_{id}
\f]
\f[
       k = j % (nd+1), id = j/(nd+1)
\f]
   **********************************************************************/
  proteus_index_t eN; int iq, id, k, j;

  for (eN = 0; eN < nElements_global; eN++)
    {
      for (iq = 0; iq < nQuadraturePoints_element; iq++)
	{
	  for (id = 0; id < nSpace; id++)
	    {
	      q_velocity[eN*nQuadraturePoints_element*nSpace + iq*nSpace + id] = 0.0;
	      for (k = 0; k < nSpace+1; k++)
		{
		  j = id*(nSpace+1) + k;
		  q_velocity[eN*nQuadraturePoints_element*nSpace + iq*nSpace + id] +=
		    q_v[eN*nQuadraturePoints_element*nDOF_trial_element + iq*nDOF_trial_element + k]
		    *
		    p1_velocity_dofs[eN*nVDOF_element + j];
		}/*k*/ 
	    }/*id*/
	}/*iq*/
    }/*eN*/

}

void getElementBDM1velocityValuesLagrangeRep(proteus_index_t nElements_global,
					     int nQuadraturePoints_element,
					     int nSpace,
					     int nDOF_trial_element,
					     int nVDOF_element,
					     double * q_v, /*scalar P^1 shape fncts*/
					     double * p1_velocity_dofs,
					     double * q_velocity)
{
  /***********************************************************************
    Assumes local representation for \f$[P^1(E)]^d\f$ is
\f[
    \vec N_j= \lambda_k \vec e_{id}
\f]
\f[
       k = j / nd, id = j % nd
\f]
   **********************************************************************/
  proteus_index_t eN; int iq, id, k, j;

  for (eN = 0; eN < nElements_global; eN++)
    {
      for (iq = 0; iq < nQuadraturePoints_element; iq++)
	{
	  for (id = 0; id < nSpace; id++)
	    {
	      q_velocity[eN*nQuadraturePoints_element*nSpace + iq*nSpace + id] = 0.0;
	      for (k = 0; k < nSpace+1; k++)
		{
		  j = k*nSpace+ id; /*id*(nSpace+1) + k;*/
		  q_velocity[eN*nQuadraturePoints_element*nSpace + iq*nSpace + id] +=
		    q_v[eN*nQuadraturePoints_element*nDOF_trial_element + iq*nDOF_trial_element + k]
		    *
		    p1_velocity_dofs[eN*nVDOF_element + j];
		}/*k*/ 
	    }/*id*/
	}/*iq*/
    }/*eN*/

}

void getElementBDM2velocityValuesLagrangeRep(proteus_index_t nElements_global,
					     int nQuadraturePoints_element,
					     int nSpace,
					     int nDOF_trial_element,
					     int nVDOF_element,
					     double * q_v, /*scalar P^1 shape fncts*/
					     double * p1_velocity_dofs,
					     double * q_velocity)
{
  /***********************************************************************
    Assumes local representation for \f$[P^1(E)]^d\f$ is
\f[
    \vec N_j= \lambda_k \vec e_{id}
\f]
\f[
       k = j / nd, id = j % nd
\f]
   **********************************************************************/
  proteus_index_t eN; int iq, id, k, j;
  for (eN = 0; eN < nElements_global; eN++)
    {
      for (iq = 0; iq < nQuadraturePoints_element; iq++)
	{
	  for (id = 0; id < nSpace; id++)
	    {
	      q_velocity[eN*nQuadraturePoints_element*nSpace + iq*nSpace + id] = 0.0;
	      for (k = 0; k < nDOF_trial_element; k++)
		{
		  j = k*nSpace+ id; /*id*(nSpace+1) + k;*/		  
		  q_velocity[eN*nQuadraturePoints_element*nSpace + iq*nSpace + id] +=
		    q_v[eN*nQuadraturePoints_element*nDOF_trial_element + iq*nDOF_trial_element + k]
		    *
		    p1_velocity_dofs[eN*nVDOF_element + j];

		}/*k*/ 
	    }/*id*/
	}/*iq*/
    }/*eN*/

}

void getGlobalExteriorElementBoundaryBDM1velocityValuesLagrangeRep(proteus_index_t nExteriorElementBoundaries_global,
								   int nQuadraturePoints_elementBoundary,
								   int nSpace,
								   int nDOF_trial_element,
								   int nVDOF_element,
								   proteus_index_t *elementBoundaryElementsArray,
								   proteus_index_t *exteriorElementBoundariesArray,
								   double * ebqe_v, /*scalar P^1 shape fncts*/
								   double * p1_velocity_dofs,
								   double * ebqe_velocity)
{
  /***********************************************************************
    Assumes local representation for \f$[P^1(E)]^d\f$ is
\f[
    \vec N_j= \lambda_k \vec e_{id}
\f]
\f[
       k = j / nd, id = j % nd
\f]
   **********************************************************************/
  proteus_index_t ebN, eN; int ebNE, iq, id, k, j;

  for (ebNE = 0; ebNE < nExteriorElementBoundaries_global; ebNE++)
    {
      ebN = exteriorElementBoundariesArray[ebNE];
      eN  = elementBoundaryElementsArray[ebN*2 + 0];
      
      for (iq = 0; iq < nQuadraturePoints_elementBoundary; iq++)
	{
	  for (id = 0; id < nSpace; id++)
	    {
	      ebqe_velocity[ebNE*nQuadraturePoints_elementBoundary*nSpace + iq*nSpace + id] = 0.0;
	      for (k = 0; k < nSpace+1; k++)
		{
		  j = k*nSpace+ id; /*id*(nSpace+1) + k;*/
		  ebqe_velocity[ebNE*nQuadraturePoints_elementBoundary*nSpace + iq*nSpace + id] +=
		    ebqe_v[ebNE*nQuadraturePoints_elementBoundary*nDOF_trial_element + iq*nDOF_trial_element + k]
		    *
		    p1_velocity_dofs[eN*nVDOF_element + j];
		}/*k*/ 
	    }/*id*/
	}/*iq*/
    }/*eN*/

}
void getGlobalElementBoundaryBDM1velocityValuesLagrangeRep(proteus_index_t nExteriorElementBoundaries_global,
								   int nQuadraturePoints_elementBoundary,
								   int nSpace,
								   int nDOF_trial_element,
								   int nVDOF_element,
								   proteus_index_t *elementBoundaryElementsArray,
								   proteus_index_t *exteriorElementBoundariesArray,
								   double * ebqe_v, /*scalar P^1 shape fncts*/
								   double * p1_velocity_dofs,
								   double * ebq_global_velocity)
{
  /***********************************************************************
    Assumes local representation for \f$[P^1(E)]^d\f$ is
\f[
    \vec N_j= \lambda_k \vec e_{id}
\f]
\f[
       k = j / nd, id = j % nd
\f]
   **********************************************************************/
  proteus_index_t ebN, eN; int ebNE, iq, id, k, j;
 
  for (ebNE = 0; ebNE < nExteriorElementBoundaries_global; ebNE++)
    {
      ebN = exteriorElementBoundariesArray[ebNE]; /* global boundary number */
      eN  = elementBoundaryElementsArray[ebN*2 + 0];
      
      for (iq = 0; iq < nQuadraturePoints_elementBoundary; iq++)
	{
	  for (id = 0; id < nSpace; id++)
	    {
	      ebq_global_velocity[ebN*nQuadraturePoints_elementBoundary*nSpace + iq*nSpace + id] = 0.0;
	      for (k = 0; k < nSpace+1; k++) /* looping over P1 degrees of freedom */
		{
		  j = k*nSpace+ id; /*id*(nSpace+1) + k;*/
		  ebq_global_velocity[ebN*nQuadraturePoints_elementBoundary*nSpace + iq*nSpace + id] +=
		    ebqe_v[ebNE*nQuadraturePoints_elementBoundary*nDOF_trial_element + iq*nDOF_trial_element + k]
		    *
		    p1_velocity_dofs[eN*nVDOF_element + j];
		}/*k*/ 
	    }/*id*/
	}/*iq*/
    }/*eN*/

}

void getElementBoundaryBDM1velocityValuesLagrangeRep(proteus_index_t nElements_global,
								   int nBoundaries_Element,
								   int nQuadraturePoints_elementBoundary,
								   int nSpace,
								   int nDOF_trial_element,
								   int nVDOF_element,
						                   proteus_index_t *elementBoundaryElementsArray, /* not used */
								   proteus_index_t *exteriorElementBoundariesArray, /*not used */
								   double * ebq_v, /*scalar P^1 shape fncts*/
								   double * p1_velocity_dofs,
								   double * ebq_velocity)
{
  /***********************************************************************
    Assumes local representation for \f$[P^1(E)]^d\f$ is
\f[
    \vec N_j= \lambda_k \vec e_{id}
\f]
\f[
       k = j / nd, id = j % nd
\f]
   **********************************************************************/
  proteus_index_t ebN, eN; int iq, id, k, j;

  for (eN = 0; eN < nElements_global; eN++)
    {
     
     for (ebN=0; ebN < nBoundaries_Element; ebN ++)
       {
      for (iq = 0; iq < nQuadraturePoints_elementBoundary; iq++)
	{
	  for (id = 0; id < nSpace; id++)
	    {
	      ebq_velocity[eN*nBoundaries_Element*nQuadraturePoints_elementBoundary*nSpace 
			+ ebN*nQuadraturePoints_elementBoundary*nSpace + iq*nSpace + id] = 0.0;
	      for (k = 0; k < nSpace+1; k++)
		{
		  j = k*nSpace+ id; /*id*(nSpace+1) + k;*/
		  ebq_velocity[eN*nBoundaries_Element*nQuadraturePoints_elementBoundary*nSpace
			+ ebN*nQuadraturePoints_elementBoundary*nSpace + iq*nSpace + id] +=
		    ebq_v[eN*nBoundaries_Element*nQuadraturePoints_elementBoundary*nDOF_trial_element
			+ ebN*nQuadraturePoints_elementBoundary*nDOF_trial_element + iq*nDOF_trial_element + k]
		    *
		    p1_velocity_dofs[eN*nVDOF_element + j];
		}/*k*/ 
	    }/*id*/
	}/*iq*/
       }/*ebN*/
    }/*eN*/

}

/***********************************************************************
   try implementing Sun-Wheeler Gauss-Seidel velocity postprocessing
 ***********************************************************************/
void calculateConservationResidualGlobalBoundaries(proteus_index_t nElements_global,
						   proteus_index_t nInteriorElementBoundaries_global,
						   proteus_index_t nExteriorElementBoundaries_global,
						   int nElementBoundaries_element,
						   int nQuadraturePoints_elementBoundary,
						   int nNodes_element,
						   int nSpace,
						   proteus_index_t* interiorElementBoundaries,
						   proteus_index_t* exteriorElementBoundaries,
						   proteus_index_t* elementBoundaryElements,
						   int* elementBoundaryLocalElementBoundaries,
						   int* exteriorElementBoundariesToSkip,
						   double* dS,
						   double* normal,
						   double* elementResidual,
						   double* velocity,
						   double* conservationResidual)
{
  proteus_index_t ebN, eN, nN, left_eN, right_eN; int ebNI, ebNE, ebN_element, left_ebN_element, right_ebN_element, k, I;
  register double flux,ds;
  /*mwf debug*/
/*   register double signDebug = -1.0; */
  /*first loop through and get element residual sums*/
  for (eN = 0; eN < nElements_global; eN++)
    {
      for (nN = 0; nN < nNodes_element; nN++)
	{
	  conservationResidual[eN] += elementResidual[eN*nNodes_element + nN];
	}
    }
  /*now loop through element boundaries and update element sums*/
  /*interior*/
  for (ebNI = 0; ebNI < nInteriorElementBoundaries_global; ebNI++)
    {
      ebN = interiorElementBoundaries[ebNI];
      left_eN = elementBoundaryElements[ebN*2+0];
      right_eN = elementBoundaryElements[ebN*2+1];
      left_ebN_element = elementBoundaryLocalElementBoundaries[ebN*2+0];
      right_ebN_element = elementBoundaryLocalElementBoundaries[ebN*2+1];

      flux = 0.0;
      for(k=0;k<nQuadraturePoints_elementBoundary;k++)
	{
          ds = dS[left_eN*nElementBoundaries_element*nQuadraturePoints_elementBoundary+
		  left_ebN_element*nQuadraturePoints_elementBoundary+
		  k];
	  for (I = 0; I < nSpace; I++)
	    {
	      flux+= velocity[ebN*nQuadraturePoints_elementBoundary*nSpace+
			      k*nSpace+
			      I]
		*
		normal[ebN*nQuadraturePoints_elementBoundary*nSpace+
		       k*nSpace+
		       I] 
		* 
		ds;
	    }
	}/*k*/
      conservationResidual[left_eN] += flux;
      conservationResidual[right_eN]-= flux;

    }/*ebNI*/
  /*exterior*/
  for (ebNE = 0; ebNE < nExteriorElementBoundaries_global; ebNE++)
    {
      if (!exteriorElementBoundariesToSkip[ebNE])
	{
	  ebN = exteriorElementBoundaries[ebNE];
	  eN = elementBoundaryElements[ebN*2+0];
	  ebN_element = elementBoundaryLocalElementBoundaries[ebN*2+0];
	  flux = 0.0;
	  for(k=0;k<nQuadraturePoints_elementBoundary;k++)
	    {
	      ds = dS[eN*nElementBoundaries_element*nQuadraturePoints_elementBoundary+
		      ebN_element*nQuadraturePoints_elementBoundary+
		      k];
	      for (I = 0; I < nSpace; I++)
		{
		  flux+= velocity[ebN*nQuadraturePoints_elementBoundary*nSpace+
				  k*nSpace+
				  I]
		    *
		    normal[ebN*nQuadraturePoints_elementBoundary*nSpace+
			   k*nSpace+
			   I]
		    *
		    ds;
		}
	    }/*k*/
	  conservationResidual[eN] += flux;
	}/*not skipping*/
    }/*ebNE*/
}

void sunWheelerGSsweep(proteus_index_t nElements_global,
		       proteus_index_t nElementBoundaries_global,
		       proteus_index_t nInteriorElementBoundaries_global,
		       proteus_index_t nExteriorElementBoundaries_global,
		       int nElementBoundaries_element,
		       int nQuadraturePoints_elementBoundary,
		       int nSpace,
		       proteus_index_t* interiorElementBoundaries,
		       proteus_index_t* exteriorElementBoundaries,
		       proteus_index_t* elementBoundaryElements,
		       int* elementBoundaryLocalElementBoundaries,
		       double* dS,
		       double* normal,
		       double* sqrt_det_g,
		       double* alpha,
		       double* fluxCorrection,
		       double* conservationResidual)
{
  proteus_index_t ebN, left_eN, right_eN; int ebNI, left_ebN_element, right_ebN_element;
  register double area,areaFact,F_ebN;
  areaFact = 1.0;
  if (nSpace == 3)
    areaFact = 0.5;
  /*interior faces*/
  for (ebNI = 0; ebNI < nInteriorElementBoundaries_global; ebNI++)
    {
      ebN = interiorElementBoundaries[ebNI];
      left_eN = elementBoundaryElements[ebN*2+0];
      right_eN = elementBoundaryElements[ebN*2+1];
      left_ebN_element = elementBoundaryLocalElementBoundaries[ebN*2+0];
      right_ebN_element = elementBoundaryLocalElementBoundaries[ebN*2+1];
      /*assumed affine*/
      area = areaFact*sqrt_det_g[left_eN*nElementBoundaries_element*nQuadraturePoints_elementBoundary +
				 left_ebN_element*nQuadraturePoints_elementBoundary + 0];
      F_ebN = alpha[ebN*2+0]*conservationResidual[left_eN] 
	+ alpha[ebN*2+1]*conservationResidual[right_eN];

      fluxCorrection[ebN] += F_ebN;
      
      /*Our residual is (R,1)_e-<Fn_f,n_e>_e which is opposite of our paper formulation*/
      conservationResidual[left_eN] -= area*F_ebN;
      conservationResidual[right_eN]+= area*F_ebN;
    }
  /*need to figure out what to do about exterior faces*/
  /*what about reversing?*/


}

void fluxCorrectionVelocityUpdate(proteus_index_t nElements_global,
				  proteus_index_t nElementBoundaries_global,
				  proteus_index_t nInteriorElementBoundaries_global,
				  proteus_index_t nExteriorElementBoundaries_global,
				  int nElementBoundaries_element,
				  int nQuadraturePoints_elementBoundary,
				  int nSpace,
				  proteus_index_t* interiorElementBoundaries,
				  proteus_index_t* exteriorElementBoundaries,
				  proteus_index_t* elementBoundaryElements,
				  int* elementBoundaryLocalElementBoundaries,
				  double* dS,
				  double* normal,
				  double* fluxCorrection,
				  double* vConservative,
				  double* vConservative_element)
{
  proteus_index_t eN, ebN, left_eN, right_eN; int ebN_element, ebNI, ebNE, left_ebN_element, right_ebN_element, k, I;

  for (ebN = 0; ebN < nElementBoundaries_global; ebN++)
    {
      for (k=0; k < nQuadraturePoints_elementBoundary; k++)
	{
	  for (I=0; I < nSpace; I++)
	    {
	      vConservative[ebN*nQuadraturePoints_elementBoundary*nSpace +
			    k*nSpace + I]
		-= fluxCorrection[ebN]
		* normal[ebN*nQuadraturePoints_elementBoundary*nSpace + 
			 k*nSpace + I];
	    }
	}/*k*/
    }/*ebN*/
  /*copy the global velocity onto the elements*/
  for (ebNI=0;ebNI<nInteriorElementBoundaries_global;ebNI++)
    {
      ebN = interiorElementBoundaries[ebNI];
      left_eN = elementBoundaryElements[ebN*2+0];
      right_eN = elementBoundaryElements[ebN*2+1];
      left_ebN_element = elementBoundaryLocalElementBoundaries[ebN*2+0];
      right_ebN_element = elementBoundaryLocalElementBoundaries[ebN*2+1];
      for(k=0;k<nQuadraturePoints_elementBoundary;k++)
        for(I=0;I<nSpace;I++)
          {
            vConservative_element[left_eN*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                                  left_ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                                  k*nSpace+
                                  I]
              =
              vConservative[ebN*nQuadraturePoints_elementBoundary*nSpace+
                            k*nSpace+
                            I];
            vConservative_element[right_eN*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                                  right_ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                                  k*nSpace+
                                  I]
              =
	      vConservative[ebN*nQuadraturePoints_elementBoundary*nSpace+
			    k*nSpace+
			    I];
	  }/*I,k*/
    }/*ebNI*/
  for (ebNE=0;ebNE<nExteriorElementBoundaries_global;ebNE++)
    {
      ebN = exteriorElementBoundaries[ebNE];
      eN = elementBoundaryElements[ebN*2+0];
      ebN_element = elementBoundaryLocalElementBoundaries[ebN*2+0];
      for(k=0;k<nQuadraturePoints_elementBoundary;k++)
        for(I=0;I<nSpace;I++)
          {
	    vConservative_element[eN*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
				  ebN_element*nQuadraturePoints_elementBoundary*nSpace+
				  k*nSpace+
				  I]
	      =
	      vConservative[ebN*nQuadraturePoints_elementBoundary*nSpace+
			    k*nSpace+
			    I];
	  }
    }/*ebNE*/
}

void computeFluxCorrectionPWC(proteus_index_t nElementBoundaries_global,
			      proteus_index_t nInteriorElementBoundaries_global,
			      proteus_index_t nExteriorElementBoundaries_global,
			      proteus_index_t* interiorElementBoundaries,
			      proteus_index_t* exteriorElementBoundaries,
			      proteus_index_t* elementBoundaryElements,
			      double* pwcW,
			      double* pwcV,
			      double* fluxCorrection)
{
  /*
    generate flux correction from element V's
    correction on face f = \partial \Omega_l \cap \partial \Omega_{r}
    \Delta_f = (V_l - V_r)/|\gamma_f|
  */

  proteus_index_t ebN, eN_left, eN_right; int ebNI, ebNE;
  double V_left,V_right,w_left,w_right;
  for (ebNI = 0; ebNI < nInteriorElementBoundaries_global; ebNI++)
    {
      ebN = interiorElementBoundaries[ebNI];
      eN_left = elementBoundaryElements[ebN*2 + 0];
      eN_right= elementBoundaryElements[ebN*2 + 1];
      V_left  = pwcV[eN_left];
      V_right = pwcV[eN_right];
      w_left  = pwcW[ebN*2 + 0];
      w_right = pwcW[ebN*2 + 1];
      fluxCorrection[ebN] = V_left*w_left + V_right*w_right;
    }
  for (ebNE = 0; ebNE < nExteriorElementBoundaries_global; ebNE++)
    {
      ebN = exteriorElementBoundaries[ebNE];
      eN_left = elementBoundaryElements[ebN*2 + 0];
      V_left  = pwcV[eN_left];
      w_left  = pwcW[ebN*2 + 0];
      fluxCorrection[ebN] = V_left*w_left;
    }
}

/***********************************************************************
  node star solver data type stuff
 ***********************************************************************/


int nodeStar_init(proteus_index_t nElements_global,
		  int nNodes_element,
		  proteus_index_t nNodes_global,
		  int* nElements_node,
		  proteus_index_t* nodeStarElementsArray,
		  proteus_index_t* nodeStarElementNeighborsArray,
		  int* N_p,
		  int** subdomain_dim_p, 
		  double *** subdomain_L_p,
		  double *** subdomain_R_p,
		  double *** subdomain_U_p,
		  PROTEUS_LAPACK_INTEGER*** subdomain_pivots_p,
		  PROTEUS_LAPACK_INTEGER*** subdomain_column_pivots_p)
{
  int I;
  
  proteus_index_t N;
  int* subdomain_dim; 
  double** subdomain_R; 
  double** subdomain_U;
  double** subdomain_L; 
  PROTEUS_LAPACK_INTEGER** subdomain_pivots;
  PROTEUS_LAPACK_INTEGER** subdomain_column_pivots;

  N = nNodes_global;

  *N_p = N;
  *subdomain_dim_p    = (int*) malloc(N*sizeof(int));
  *subdomain_pivots_p = (PROTEUS_LAPACK_INTEGER**)malloc(N*sizeof(PROTEUS_LAPACK_INTEGER*));
  *subdomain_column_pivots_p = (PROTEUS_LAPACK_INTEGER**)malloc(N*sizeof(PROTEUS_LAPACK_INTEGER*));
  *subdomain_R_p      = (double**)malloc(N*sizeof(double*));
  *subdomain_U_p      = (double**)malloc(N*sizeof(double*));
  *subdomain_L_p      = (double**)malloc(N*sizeof(double*));

  if ( (*subdomain_dim_p == NULL) ||
       (*subdomain_R_p == NULL) ||
       (*subdomain_U_p == NULL) ||
       (*subdomain_L_p == NULL) ||
       (*subdomain_pivots_p == NULL)  ||
       (*subdomain_column_pivots_p == NULL))
    {
      return 1;
    }

  subdomain_dim = *subdomain_dim_p;
  subdomain_pivots = *subdomain_pivots_p;
  subdomain_column_pivots = *subdomain_column_pivots_p;
  subdomain_R = *subdomain_R_p;
  subdomain_U = *subdomain_U_p;
  subdomain_L = *subdomain_L_p;

  /*setup local node star system sizes*/
  for (I = 0; I < N; I++)
    {
      subdomain_dim[I]    = nElements_node[I];
      subdomain_pivots[I] = (PROTEUS_LAPACK_INTEGER*) malloc(subdomain_dim[I]*sizeof(PROTEUS_LAPACK_INTEGER));
      subdomain_column_pivots[I] = (PROTEUS_LAPACK_INTEGER*) malloc(subdomain_dim[I]*sizeof(PROTEUS_LAPACK_INTEGER));
      subdomain_R[I]      = (double*) malloc(subdomain_dim[I]*sizeof(double));
      subdomain_U[I]      = (double*) malloc(subdomain_dim[I]*sizeof(double));
      subdomain_L[I]      = (double*) malloc(subdomain_dim[I]*subdomain_dim[I]*sizeof(double));

      if ((subdomain_pivots[I] == NULL) ||
	  (subdomain_column_pivots[I] == NULL) ||
          (subdomain_R[I] == NULL) ||
          (subdomain_U[I] == NULL) ||
          (subdomain_L[I] == NULL))
        {
          return 1;
        }

    }/*I*/

  return 0;
}

int nodeStar_free(int N,
		  int * subdomain_dim,
		  double ** subdomain_L,
		  double ** subdomain_R,
		  double ** subdomain_U,
		  PROTEUS_LAPACK_INTEGER** subdomain_pivots,
		  PROTEUS_LAPACK_INTEGER** subdomain_column_pivots)
{
  int I;
  free(subdomain_dim);
  for (I = 0; I < N; I++)
    {
      free(subdomain_pivots[I]);
      free(subdomain_column_pivots[I]);
      free(subdomain_R[I]);
      free(subdomain_U[I]);
      free(subdomain_L[I]);
    }
  free(subdomain_pivots);
  free(subdomain_column_pivots);
  free(subdomain_R);
  free(subdomain_U);
  free(subdomain_L);
  
  subdomain_pivots = 0;
  subdomain_column_pivots = 0;
  subdomain_R      = 0;
  subdomain_U      = 0;
  subdomain_L      = 0;
  return 0;
}

int nodeStar_setU(NodeStarFactorStruct* nodeStarFactor, double val)
{
  int I,i;
  assert(nodeStarFactor);
  for (I = 0; I < nodeStarFactor->N; I++)
    for(i=0; i < nodeStarFactor->subdomain_dim[I]; i++)
      nodeStarFactor->subdomain_U[I][i] = val;
  return 0;
}

int nodeStar_copy(int other_N,
		  int * other_subdomain_dim,
		  double ** other_subdomain_L,
		  double ** other_subdomain_R,
		  double ** other_subdomain_U,
		  PROTEUS_LAPACK_INTEGER** other_subdomain_pivots,
		  PROTEUS_LAPACK_INTEGER** other_subdomain_column_pivots,
		  int* N_p,
		  int** subdomain_dim_p, 
		  double *** subdomain_L_p,
		  double *** subdomain_R_p,
		  double *** subdomain_U_p,
		  PROTEUS_LAPACK_INTEGER*** subdomain_pivots_p,
		  PROTEUS_LAPACK_INTEGER*** subdomain_column_pivots_p)
{
  /*assumes both sets of structure have been allocated*/
  proteus_index_t N; int I, i;
  int* subdomain_dim; 
  double** subdomain_R; 
  double** subdomain_U;
  double** subdomain_L; 
  PROTEUS_LAPACK_INTEGER** subdomain_pivots;
  PROTEUS_LAPACK_INTEGER** subdomain_column_pivots;

  int failed = 0;
  int realloc = 0;
  realloc = other_N != *N_p;
  if (!realloc)
    {
      for (I=0; I < other_N; I++)
	{
	  realloc = realloc || (other_subdomain_dim[I] != (*subdomain_dim_p)[I]);
	}
    }
  if (realloc)
    {
      /*mwf debug*/
      printf("nodeStar_copy needs to reaalloc self_N=%d other_N=%d \n",*N_p,other_N);
      failed = nodeStar_free(*N_p,
			     *subdomain_dim_p, 
			     *subdomain_L_p,
			     *subdomain_R_p,
			     *subdomain_U_p,
			     *subdomain_pivots_p,
			     *subdomain_column_pivots_p);

      if (failed)
	return 1;
      N = other_N;
      *N_p = N;
      *subdomain_dim_p    = (int*)malloc(N*sizeof(int));
      *subdomain_pivots_p = (PROTEUS_LAPACK_INTEGER**)malloc(N*sizeof(PROTEUS_LAPACK_INTEGER*));
      *subdomain_column_pivots_p = (PROTEUS_LAPACK_INTEGER**)malloc(N*sizeof(PROTEUS_LAPACK_INTEGER*));
      *subdomain_R_p      = (double**)malloc(N*sizeof(double*));
      *subdomain_U_p      = (double**)malloc(N*sizeof(double*));
      *subdomain_L_p      = (double**)malloc(N*sizeof(double*));

      if ( (*subdomain_dim_p == NULL) ||
	   (*subdomain_R_p == NULL) ||
	   (*subdomain_U_p == NULL) ||
	   (*subdomain_L_p == NULL) ||
	   (*subdomain_pivots_p == NULL) ||
	   (*subdomain_column_pivots_p == NULL) )
	{
	  return 1;
	}
      subdomain_dim = *subdomain_dim_p;
      subdomain_pivots = *subdomain_pivots_p;
      subdomain_column_pivots = *subdomain_column_pivots_p;
      subdomain_R = *subdomain_R_p;
      subdomain_U = *subdomain_U_p;
      subdomain_L = *subdomain_L_p;
      
      /*setup local node star system sizes*/
      for (I = 0; I < N; I++)
	{
	  subdomain_dim[I]    = other_subdomain_dim[I];
	  subdomain_pivots[I] = (PROTEUS_LAPACK_INTEGER*)malloc(subdomain_dim[I]*sizeof(PROTEUS_LAPACK_INTEGER));
	  subdomain_column_pivots[I] = (PROTEUS_LAPACK_INTEGER*)malloc(subdomain_dim[I]*sizeof(PROTEUS_LAPACK_INTEGER));
	  subdomain_R[I]      = (double*)malloc(subdomain_dim[I]*sizeof(double));
	  subdomain_U[I]      = (double*)malloc(subdomain_dim[I]*sizeof(double));
	  subdomain_L[I]      = (double*)malloc(subdomain_dim[I]*subdomain_dim[I]*sizeof(double));
	  
	  if ((subdomain_pivots[I] == NULL) ||
	      (subdomain_column_pivots[I] == NULL) ||
	      (subdomain_R[I] == NULL) ||
	      (subdomain_U[I] == NULL) ||
	      (subdomain_L[I] == NULL))
	    {
	      return 1;
	    }
	  
	}/*I*/
    }/*had to realloc*/
  N = *N_p;
  subdomain_dim = *subdomain_dim_p;
  subdomain_pivots = *subdomain_pivots_p;
  subdomain_column_pivots = *subdomain_column_pivots_p;
  subdomain_R = *subdomain_R_p;
  subdomain_U = *subdomain_U_p;
  subdomain_L = *subdomain_L_p;

  /*now copy*/
  /*mwf debug
  printf("nodeStar_copy data now N=%d \n",N);
  */
  for (I = 0; I < N; I++)
    {
      assert(subdomain_dim[I] == other_subdomain_dim[I]);
      for (i = 0; i < subdomain_dim[I]; i++)
	{
	  subdomain_pivots[I][i] = other_subdomain_pivots[I][i];
	  subdomain_column_pivots[I][i] = other_subdomain_column_pivots[I][i];
	  subdomain_R[I][i]      = other_subdomain_R[I][i];
	  subdomain_U[I][i]      = other_subdomain_U[I][i];
	}
      for (i = 0; i < subdomain_dim[I]*subdomain_dim[I]; i++)
	subdomain_L[I][i] = other_subdomain_L[I][i];

    }/*I*/

  return 0;
}
/***********************************************************************
  end node star solver data type stuff
 ***********************************************************************/

/***********************************************************************
 try to put in version of Swedish Postprocessing with Neumann boundaries
  enforced explicitly and NodeStarFactorStruct data type

  This version, 
    skips flux boundaries in element integrals: calculateConservationResidualPWL,
                                                calculateConservationJacobianPWL

    sets row 0 of flux boundary nodes jacobian to Id and rhs to 0:
           calculateConservationJacobianPWL, calculateConservationFluxPWL

  When using this version, 
     do not remove boundary flux terms from element residual
     must load in boundary fluxes into ebq_global velocity and ebq velocity
      after calling

  Note,
    fluxElementBoundaries holds global exterior element boundary ids for flux bcs
    fluxElementBoundaryNodes holds global node numbers for flux bcs
 ***********************************************************************/
void calculateConservationResidualPWL(proteus_index_t nElements_global,
				      proteus_index_t nInteriorElementBoundaries_global,
				      proteus_index_t nExteriorElementBoundaries_global,
				      int nElementBoundaries_element,
				      int nQuadraturePoints_elementBoundary,
				      int nNodes_element,
				      int nDOF_element,
				      int nSpace,
				      proteus_index_t* interiorElementBoundaries,
				      proteus_index_t* exteriorElementBoundaries,
				      proteus_index_t* elementBoundaryElements,
				      int* elementBoundaryLocalElementBoundaries,
				      proteus_index_t* elementNodes,
				      proteus_index_t* dofMapl2g,
				      proteus_index_t* nodeStarElements,
				      proteus_index_t* nodeStarElementNeighbors,
				      int* nElements_node,
				      proteus_index_t* fluxElementBoundaries,
				      double* elementResidual,
				      double* vAverage,
				      double* dX,
				      double* w,
				      double* normal,
				      NodeStarFactorStruct* nodeStarFactor,
				      double* conservationResidual,
				      double* vConservative,
				      double* vConservative_element)
{
  proteus_index_t ebN, eN, eN_star, left_eN, right_eN, left_eN_star, right_eN_star, nN, nN_global; int ebNI, ebNE, ebN_element, left_ebN_element, right_ebN_element, k, I;
  register double flux,fluxAverage,fluxCorrection,dx=0.0;
  /*mwf now access node star system using NodeStarFactorStruct*/
  double ** starR, ** starU;
  /*mwf add for debugging
    double * fluxSumDebug;
    double resSum;
    fluxSumDebug = calloc(nElements_global,sizeof(double));
    memset(fluxSumDebug,0,sizeof(double)*nElements_global);
  */
  /*mwf end for debugging*/
  memset(conservationResidual,0,sizeof(double)*nElements_global);
  /*mwf debug
    printf("calcConsResidPWL nQuadraturePoints_elementBoundary=%d \n",
    nQuadraturePoints_elementBoundary);
  */
  assert(nodeStarFactor);
  starR = nodeStarFactor->subdomain_R;
  starU = nodeStarFactor->subdomain_U;
  /*initial residual with element residual*/
  for (eN=0;eN<nElements_global;eN++)
   {
     for (nN=0;nN<nNodes_element;nN++)
       {
	 nN_global = dofMapl2g[eN*nDOF_element+
			       nN];
	 eN_star  = nodeStarElements[eN*nNodes_element+
				     nN];
	 starR[nN_global][eN_star] 
	   = 
	   elementResidual[eN*nNodes_element+
			   nN];
	 conservationResidual[eN]
	   += 
	   elementResidual[eN*nNodes_element+
			   nN];
	 /*mwf debug
	 printf("calcConsResPWL eN=%d nN=%d starR=%g\n",eN,nN,
		starR[nN_global][eN_star]);
	 //	 */
       }
     //    /*mwf debug
     //       printf("calcConsResPWL eN=%d consRes=%g\n",eN,
     //  conservationResidual[eN]);
     //  */
   }
  /*calculate interior element boundary fluxes and update residual*/
  for (ebNI=0;ebNI<nInteriorElementBoundaries_global;ebNI++)
    {
      ebN = interiorElementBoundaries[ebNI];
      left_eN = elementBoundaryElements[ebN*2+0];
      right_eN = elementBoundaryElements[ebN*2+1];
      left_ebN_element = elementBoundaryLocalElementBoundaries[ebN*2+0];
      right_ebN_element = elementBoundaryLocalElementBoundaries[ebN*2+1];
      /* printf("ebN = %d, left_eN = %d, right_eN = %d, left_ebN_element = %d, right_ebN_element = %d\n", */
      /* 	     ebN, */
      /* 	     left_eN, */
      /* 	     right_eN, */
      /* 	     left_ebN_element, */
      /* 	     right_ebN_element); */
      for(k=0;k<nQuadraturePoints_elementBoundary;k++)
	{
          fluxAverage = 0.0;
          /* get the integration weight */
          dx = dX[left_eN*nElementBoundaries_element*nQuadraturePoints_elementBoundary+
		  left_ebN_element*nQuadraturePoints_elementBoundary+
		  k];
	  /*mwf debug
	    printf("calcConsResPWL ebNI=%d k=%d dx=%g \n",ebNI,k,dx);
	  */
	  for (I=0;I<nSpace;I++)
	    {
              vConservative[ebN*nQuadraturePoints_elementBoundary*nSpace+
                            k*nSpace+
                            I] 
                =
                vAverage[ebN*nQuadraturePoints_elementBoundary*nSpace+
                         k*nSpace+
                         I];
	      /*mwf debug
		printf("pwl get flux vAverage int I=%d, val=%g \n",I,
		vAverage[ebN*nQuadraturePoints_elementBoundary*nSpace+
		k*nSpace+
		I]);
	      */
              fluxAverage
                +=
                vAverage[ebN*nQuadraturePoints_elementBoundary*nSpace+
                         k*nSpace+
                         I]
                *
                normal[ebN*nQuadraturePoints_elementBoundary*nSpace+
                       k*nSpace+
                       I];
            }
          for (nN=0;nN<nNodes_element;nN++)
	    {
	      // ARB CHANGE
	      nN_global = dofMapl2g[left_eN*nDOF_element+
				      nN];
	      left_eN_star = nodeStarElements[left_eN*nNodes_element+
					      nN];
	      /* check if node is opposite element boundary we're computing and ignore 0 contribution */
	      /* this shouldn't be necessary */
	      if (nN != left_ebN_element)
		{
		  right_eN_star = nodeStarElementNeighbors[left_eN*nNodes_element*nElementBoundaries_element+
							   nN*nElementBoundaries_element+
							   left_ebN_element];
                  fluxCorrection = (starU[nN_global][left_eN_star]
                                    -
                                    starU[nN_global][right_eN_star])
                    *
                    w[left_eN*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nNodes_element+
                      left_ebN_element*nQuadraturePoints_elementBoundary*nNodes_element+
                      k*nNodes_element+
                      nN];
		  for (I=0;I<nSpace;I++)
		    {
		      vConservative[ebN*nQuadraturePoints_elementBoundary*nSpace+
				    k*nSpace+
				    I] 
                        +=
			fluxCorrection
			*
			normal[ebN*nQuadraturePoints_elementBoundary*nSpace+
                               k*nSpace+
                               I];
		    }
		  flux = (fluxAverage*w[left_eN*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nNodes_element+
                                        left_ebN_element*nQuadraturePoints_elementBoundary*nNodes_element+
                                        k*nNodes_element+
                                        nN]
                          + fluxCorrection)*dx;
		  starR[nN_global][left_eN_star]
		    += flux;
		  starR[nN_global][right_eN_star]
		    -= flux;
		  conservationResidual[left_eN] += flux;
		  conservationResidual[right_eN] -= flux;
		  /*mwf debug
		  printf("ppwl nN_global=%d left_eN=%d right_eN=%d fluxAvg=%g fluxCorr=%g flux=%g \n",
			 nN_global,left_eN,right_eN,fluxAverage,fluxCorrection,flux);
		  */
		  /*mwf debug
		  fluxSumDebug[left_eN] += flux;
		  fluxSumDebug[right_eN]-= flux;
		  */
		}
	    }
	}
    }
  /*calculate fluxes on exterior element boundaries and update residual*/
  for (ebNE=0;ebNE<nExteriorElementBoundaries_global;ebNE++)
    {
      ebN = exteriorElementBoundaries[ebNE];
      eN = elementBoundaryElements[ebN*2+0];
      ebN_element = elementBoundaryLocalElementBoundaries[ebN*2+0];
      
      for(k=0;k<nQuadraturePoints_elementBoundary;k++)
	{
          /* get the integration weight, I don't think this was here before */
	  dx = dX[eN*nElementBoundaries_element*nQuadraturePoints_elementBoundary+
		  ebN_element*nQuadraturePoints_elementBoundary+
		  k];
	  
          fluxAverage=0.0;
	  for (I=0;I<nSpace;I++)
            {
              vConservative[ebN*nQuadraturePoints_elementBoundary*nSpace+
                            k*nSpace+
                            I] 
                =
                vAverage[ebN*nQuadraturePoints_elementBoundary*nSpace+
                         k*nSpace+
                         I];
	      /*mwf debug
		printf("pwl get flux vAverage ext ebN=%d free=%d I=%d, val=%g \n",ebN,fluxElementBoundaries[ebNE],
		I,vAverage[ebN*nQuadraturePoints_elementBoundary*nSpace+
		k*nSpace+
		I]);
	      */
              fluxAverage += 
                vAverage[ebN*nQuadraturePoints_elementBoundary*nSpace+
                         k*nSpace+
                         I]
                *
                normal[ebN*nQuadraturePoints_elementBoundary*nSpace+
                       k*nSpace+
                       I];
            }
	  for (nN=0;nN<nNodes_element;nN++)
	    {
	      nN_global = dofMapl2g[eN*nDOF_element+
				       nN];
	      eN_star = nodeStarElements[eN*nNodes_element+
					      nN];
	      /* check if this node lies opposite the element boundary whose contribution we're computing */
	      /* in that case there is no flux contribution because the test function is zero*/
	      /* however, we may be able to remove this conditional for speed */
	      if (nN != ebN_element && !fluxElementBoundaries[ebNE])/*mwf add skip for flux*/ 
		{
                  fluxCorrection = starU[nN_global][eN_star]
                    *
                    w[eN*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nNodes_element+
                      ebN_element*nQuadraturePoints_elementBoundary*nNodes_element+
                      k*nNodes_element+
                      nN]; 
		  for (I=0;I<nSpace;I++)
                    vConservative[ebN*nQuadraturePoints_elementBoundary*nSpace+
                                  k*nSpace+
                                  I] +=
                      fluxCorrection
                      *
                      normal[ebN*nQuadraturePoints_elementBoundary*nSpace+
                             k*nSpace+
                             I];
		  flux = (fluxAverage
                          *
                          w[eN*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nNodes_element+
                            ebN_element*nQuadraturePoints_elementBoundary*nNodes_element+
                            k*nNodes_element+
                            nN]
                          +
                          fluxCorrection)*dx;
		  starR[nN_global][eN_star]
		    += flux;
		  conservationResidual[eN] += flux;
		  /*mwf debug
		    fluxSumDebug[eN] += flux;
		  */
		  /*mwf debug 
		    printf("pwl get flux corr ext ebN=%d eN=%d fluxBC=%d fluxCorr=%g flux=%g \n",ebN,eN,
		    fluxElementBoundaries[ebNE],
		    fluxCorrection,flux);
		  */

	      
		}
	    }
	}
    }
  /*copy the global velocity onto the elements*/
  for (ebNI=0;ebNI<nInteriorElementBoundaries_global;ebNI++)
    {
      ebN = interiorElementBoundaries[ebNI];
      left_eN = elementBoundaryElements[ebN*2+0];
      right_eN = elementBoundaryElements[ebN*2+1];
      left_ebN_element = elementBoundaryLocalElementBoundaries[ebN*2+0];
      right_ebN_element = elementBoundaryLocalElementBoundaries[ebN*2+1];
      for(k=0;k<nQuadraturePoints_elementBoundary;k++)
        for(I=0;I<nSpace;I++)
          {
            vConservative_element[left_eN*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                                  left_ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                                  k*nSpace+
                                  I]
              =
              vConservative[ebN*nQuadraturePoints_elementBoundary*nSpace+
                            k*nSpace+
                            I];
            vConservative_element[right_eN*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                                  right_ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                                  k*nSpace+
                                  I]
              =
	      vConservative[ebN*nQuadraturePoints_elementBoundary*nSpace+
			    k*nSpace+
			    I];
	    /*mwf debug
	    printf("pwl int copy ebN=%d eN=[%d,%d] ebN_element=[%d,%d] k=%d vl[%d]=%g, vr[%d]=%g \n",
		   ebN,left_eN,right_eN,left_ebN_element,right_ebN_element,k,I,
		   vConservative_element[left_eN*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
					 left_ebN_element*nQuadraturePoints_elementBoundary*nSpace+
					 k*nSpace+
					 I],I,
		   vConservative_element[right_eN*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
					 right_ebN_element*nQuadraturePoints_elementBoundary*nSpace+
					 k*nSpace+
					 I]);
	    */

          }
    }

  /*copy the global velocity onto the elements*/
  for (ebNE=0;ebNE<nExteriorElementBoundaries_global;ebNE++)
    {
      ebN = exteriorElementBoundaries[ebNE];
      eN = elementBoundaryElements[ebN*2+0];
      ebN_element = elementBoundaryLocalElementBoundaries[ebN*2+0];
      for(k=0;k<nQuadraturePoints_elementBoundary;k++)
        for(I=0;I<nSpace;I++)
          {
	    vConservative_element[eN*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
				  ebN_element*nQuadraturePoints_elementBoundary*nSpace+
				  k*nSpace+
				  I]
	      =
	      vConservative[ebN*nQuadraturePoints_elementBoundary*nSpace+
			    k*nSpace+
			    I];
	    /*mwf debug
	    printf("pwl ext copy ebN=%d eN=%d ebN_element=%d k=%d v[%d]=%g \n",
		   ebN,eN,ebN_element,k,I,
		   vConservative_element[eN*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
					 ebN_element*nQuadraturePoints_elementBoundary*nSpace+
					 k*nSpace+
					 I]);
	    */

	  }
    }
  /*mwf debug
  for (eN=0; eN < nElements_global; eN++)
    {
      printf("leaving calcConsResPWL eN=%d consResid=%g \n",eN,conservationResidual[eN]);
    }
  */
  /*mwf debug
  for (eN=0; eN < nElements_global; eN++)
    {
      resSum = 0;
      for (ebN = 0; ebN < nDOF_element; ebN++)
	resSum += elementResidual[eN*nDOF_element+ebN];
      printf("eN=%d fluxSum=%g elemResid=%g consResid=%g \n",eN,fluxSumDebug[eN],resSum,
	     conservationResidual[eN]);
    }
  */
  /*mwf debug  
    free(fluxSumDebug);
  */
}

void calculateConservationJacobianPWL(proteus_index_t nDOF_global,
				      proteus_index_t nNodes_internal,
				      proteus_index_t nElements_global,
				      proteus_index_t nInteriorElementBoundaries_global,
				      proteus_index_t nExteriorElementBoundaries_global,
				      int nElementBoundaries_element,
				      int nQuadraturePoints_elementBoundary,
				      int nNodes_element,
				      int nDOF_element,
				      int nSpace,
				      proteus_index_t* interiorElementBoundaries,
				      proteus_index_t* exteriorElementBoundaries,
				      proteus_index_t* elementBoundaryElements,
				      int* elementBoundaryLocalElementBoundaries,
				      proteus_index_t* elementNodes,
				      proteus_index_t* dofMapl2g,
				      proteus_index_t* dofStarElements,
				      proteus_index_t* dofStarElementNeighbors,
				      int* nElements_node,
				      proteus_index_t* internalNodes,
				      proteus_index_t* fluxElementBoundaries,
				      int* fluxBoundaryNodes,
				      double* w,
				      double* normal,
				      NodeStarFactorStruct* nodeStarFactor)

{
  proteus_index_t eN, ebN, eN_star, left_eN, right_eN, left_eN_star, right_eN_star, nN, nN_global; int ebNI, ebNE, ebN_element, left_ebN_element, right_ebN_element, nNI, k;
  PROTEUS_LAPACK_INTEGER INFO=0;
  register double wflux;
  /*mwf add for boundaries*/
  int ii,jj;
  double ** starJacobian = nodeStarFactor->subdomain_L;
  int * subdomain_dim    = nodeStarFactor->subdomain_dim;
  PROTEUS_LAPACK_INTEGER ** starPivots = nodeStarFactor->subdomain_pivots;
  /*for full pivoting, to avoid pathological bow-tie nodes*/
  PROTEUS_LAPACK_INTEGER ** starColPivots = nodeStarFactor->subdomain_column_pivots;
 
  /*zero everything for safety*/
  assert(nodeStarFactor);
  for (nN = 0; nN < nDOF_global; nN++)
    {
      for(ii=0; ii < subdomain_dim[nN]; ii++)
	{
	  starPivots[nN][ii]=0;
	  starColPivots[nN][ii]=0;
	  for (jj=0; jj < subdomain_dim[nN]; jj++)
	    starJacobian[nN][ii + jj*subdomain_dim[nN]] = 0.0;
	}
    }
  //  printf("break1\n");
  /*Load Jacobian entries arising from iterior element boundaries*/
  for (ebNI=0;ebNI<nInteriorElementBoundaries_global;ebNI++)
    {
      ebN = interiorElementBoundaries[ebNI];
      left_eN = elementBoundaryElements[ebN*2+0];
      right_eN = elementBoundaryElements[ebN*2+1];
      left_ebN_element = elementBoundaryLocalElementBoundaries[ebN*2+0];
      right_ebN_element = elementBoundaryLocalElementBoundaries[ebN*2+1];
      for(k=0;k<nQuadraturePoints_elementBoundary;k++)
	{
	  for (nN=0;nN<nNodes_element;nN++)
	    {
	      nN_global = dofMapl2g[left_eN*nDOF_element+
				    nN];
	      left_eN_star = dofStarElements[left_eN*nNodes_element+
					      nN];
	      /* check if node is opposite element boundary we're computing and ignore 0 contribution */
	      /* this shouldn't be necessary */
	      if (nN != left_ebN_element)
		{
		  right_eN_star = dofStarElementNeighbors[left_eN*nNodes_element*nElementBoundaries_element+
							   nN*nElementBoundaries_element+
							   left_ebN_element];
		  wflux = w[left_eN*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nNodes_element+
                            left_ebN_element*nQuadraturePoints_elementBoundary*nNodes_element+
                            k*nNodes_element+
                            nN];
                  
		  starJacobian[nN_global][left_eN_star+
					  left_eN_star*subdomain_dim[nN_global]]
		    += wflux;
		  starJacobian[nN_global][left_eN_star+
					  right_eN_star*subdomain_dim[nN_global]]
		    -= wflux;
		  starJacobian[nN_global][right_eN_star+
					  left_eN_star*subdomain_dim[nN_global]]
		    -= wflux;
		  starJacobian[nN_global][right_eN_star+
					  right_eN_star*subdomain_dim[nN_global]]
		    += wflux;
		}
	    }
	}
    }  
  //  printf("break2\n");
  /*Load Jacobian entries arising from exterior element boundaries*/
  for (ebNE=0;ebNE<nExteriorElementBoundaries_global;ebNE++)
    {
      ebN = exteriorElementBoundaries[ebNE];
      eN = elementBoundaryElements[ebN*2+0];
      ebN_element = elementBoundaryLocalElementBoundaries[ebN*2+0];
      for(k=0;k<nQuadraturePoints_elementBoundary;k++)
	{
	  for (nN=0;nN<nNodes_element;nN++)
	    {
	      nN_global = dofMapl2g[eN*nDOF_element+
				    nN];
	      eN_star = dofStarElements[eN*nNodes_element+
					 nN];
	      /* check if this node lies opposite the element boundary whose contribution we're computing */
	      /* in that case there is no flux contribution because the test function is zero*/
	      /* however, we may be able to remove this conditional for speed */
              /* cek modified to be: also exclude if the flux is specified on this boundary AND nN is a free node */
	      /* mwf try to change definition of fluxElementBoundaries so that they are excluded if they contain a node
                 that is Dirichlet but has no "Dirichlet" faces associated with it 
		 if (nN != ebN_element && !(fluxElementBoundaries[ebNE] && fluxBoundaryNodes[nN_global]) )*/
	      if (nN != ebN_element && !fluxElementBoundaries[ebNE])
		{
	          starJacobian[nN_global][eN_star+
					  eN_star*nElements_node[nN_global]]
		    += 
                    w[eN*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nNodes_element+
                      ebN_element*nQuadraturePoints_elementBoundary*nNodes_element+
                      k*nNodes_element+
                      nN];
		}
	    }
	}
    }
  /*set Dirichlet boundary conditions at one element on interior node stars*/
  for (nNI=0;nNI<nNodes_internal;nNI++)
    {
      nN = internalNodes[nNI];
      starJacobian[nN][0]=1.0;
      for (eN=1;eN<nElements_node[nN];eN++)
        starJacobian[nN][eN*subdomain_dim[nN]]
	  =0.0;
    }
  /*repeat for Neumann boundary node stars*/
  for (nN=0; nN < nDOF_global; nN++)
    {
      if (fluxBoundaryNodes[nN]==1)
	{
	  starJacobian[nN][0]=1.0;
	  for (eN=1;eN<nElements_node[nN];eN++)
	    starJacobian[nN][eN*subdomain_dim[nN]]
	      =0.0;

	  /*mwf debug
	    printf("jacobian fluxNode=%d nElements_node=%d \n",nN,nElements_node[nN]);
	  */
	}
    }

  /*factor with lapack*/
  for (nN=0;nN<nDOF_global;nN++)
    {
      PROTEUS_LAPACK_INTEGER nE_n = ((PROTEUS_LAPACK_INTEGER)subdomain_dim[nN]);
/*       dgetrf_(&nE_n, */
/*               &nE_n, */
/*               starJacobian[nN], */
/*               &nE_n, */
/*               starPivots[nN], */
/*               &INFO); */
      /*could try using dgetc2 and dgesc2 to solve when have a zero row because of
	isolated nodes*/
      dgetc2_(&nE_n,
              starJacobian[nN],
              &nE_n,
              starPivots[nN],
              starColPivots[nN],
              &INFO);
      
      /*mwf debug*/
	/* if (INFO > 0) */
	/*   { */
        /*     printf("velPP jac dgetrf INFO=%d nN=%d \n",(int)(INFO),nN); */
        /*     for (ii=0;ii<nE_n;ii++) */
        /*       { */
        /*         for(jj=0;jj<nE_n;jj++) */
        /*           { */
                    
        /*             printf("%12.5e \t",starJacobian[nN][ii*nE_n + jj]); */
        /*           } */
        /*         printf("\n"); */
        /*       } */
        /*   } */
	/*assert(INFO == 0);*//*need to turn off if use dgetc2*/
    }
}
void calculateConservationFluxPWL(proteus_index_t nNodes_global,
				  proteus_index_t nNodes_internal,
				  int* nElements_node,
				  proteus_index_t* internalNodes,
				  int* fluxBoundaryNodes,
				  NodeStarFactorStruct* nodeStarFactor)
{
  proteus_index_t nN, eN; int nNI;
  PROTEUS_LAPACK_INTEGER NRHS=1,INFO=0;
  double ** starJ = nodeStarFactor->subdomain_L;
  double ** starR = nodeStarFactor->subdomain_R;
  double ** starU = nodeStarFactor->subdomain_U;
  PROTEUS_LAPACK_INTEGER ** starPivots = nodeStarFactor->subdomain_pivots;
  /*for full pivoting, to avoid pathological bow-tie nodes*/
  PROTEUS_LAPACK_INTEGER ** starColPivots = nodeStarFactor->subdomain_column_pivots;
  double scale;
  int * subdomain_dim = nodeStarFactor->subdomain_dim;
  char TRANS='N';
  /*load -R into U*/
  for (nN=0;nN<nNodes_global;nN++)
    for(eN=0;eN<subdomain_dim[nN];eN++)
      starU[nN][eN] = -starR[nN][eN];
  /*set Dirichlet boundary conditions on interior node stars*/
  for (nNI=0;nNI<nNodes_internal;nNI++)
    {
      nN = internalNodes[nNI];
      starU[nN][0]=0.0;
    }
  /*repeat for Neumann boundary node stars*/
  for (nN=0; nN < nNodes_global; nN++)
    {
      if (fluxBoundaryNodes[nN] == 1)
	{
	  starU[nN][0]=0.0;
	}
    }
  /*solve with lapack*/
  for  (nN=0;nN<nNodes_global;nN++)
    {
      PROTEUS_LAPACK_INTEGER nE_n = ((PROTEUS_LAPACK_INTEGER)subdomain_dim[nN]);
      /*mwf orig, partial pivoting*/
/*       dgetrs_(&TRANS, */
/*               &nE_n, */
/*               &NRHS, */
/*               starJ[nN], */
/*               &nE_n, */
/*               starPivots[nN], */
/*               starU[nN], */
/*               &nE_n, */
/*               &INFO); */
      /*could try using dgetc2 and dgesc2 to solve when have a zero row because of
	isolated nodes*/
	
	dgesc2_(&nE_n,
		starJ[nN],
		&nE_n,
		starU[nN],
		starPivots[nN],
		starColPivots[nN],
		&scale);
    }
}

void calculateConservationFluxPWL_noNeumannFix(proteus_index_t nNodes_global,
					       int* nElements_node,
					       NodeStarFactorStruct* nodeStarFactor)
{
  proteus_index_t nN, eN; int nNI;
  PROTEUS_LAPACK_INTEGER NRHS=1,INFO=0;
  double ** starJ = nodeStarFactor->subdomain_L;
  double ** starR = nodeStarFactor->subdomain_R;
  double ** starU = nodeStarFactor->subdomain_U;
  PROTEUS_LAPACK_INTEGER ** starPivots = nodeStarFactor->subdomain_pivots;
  /*for full pivoting, to avoid pathological bow-tie nodes*/
  PROTEUS_LAPACK_INTEGER ** starColPivots = nodeStarFactor->subdomain_column_pivots;
  double scale;
  int * subdomain_dim = nodeStarFactor->subdomain_dim;
  char TRANS='N';
  /*load -R into U*/
  for (nN=0;nN<nNodes_global;nN++)
    for(eN=0;eN<subdomain_dim[nN];eN++)
      starU[nN][eN] = -starR[nN][eN];
  /*solve with lapack*/
  for  (nN=0;nN<nNodes_global;nN++)
    {
      PROTEUS_LAPACK_INTEGER nE_n = ((PROTEUS_LAPACK_INTEGER)subdomain_dim[nN]);
      dgesc2_(&nE_n,
	      starJ[nN],
	      &nE_n,
	      starU[nN],
	      starPivots[nN],
	      starColPivots[nN],
	      &scale);
    }
}

void calculateConservationResidualPWL_opt(proteus_index_t nNodes_owned,
				      proteus_index_t nElements_global,
				      proteus_index_t nInteriorElementBoundaries_global,
				      proteus_index_t nExteriorElementBoundaries_global,
				      int nElementBoundaries_element,
				      int nQuadraturePoints_elementBoundary,
				      int nNodes_element,
				      int nSpace,
				      proteus_index_t* interiorElementBoundaries,
				      proteus_index_t* exteriorElementBoundaries,
				      proteus_index_t* elementBoundaryElements,
				      int* elementBoundaryLocalElementBoundaries,
				      proteus_index_t* elementNodes,
				      proteus_index_t* nodeStarElements,
				      proteus_index_t* nodeStarElementNeighbors,
				      int* nElements_node,
				      proteus_index_t* fluxElementBoundaries,
				      double* elementResidual,
				      double* vAverage,
				      double* dX,
				      double* w,
				      double* normal,
				      NodeStarFactorStruct* nodeStarFactor,
				      double* conservationResidual,
				      double* vConservative,
				      double* vConservative_element)
{
  int foundNonzeroR;
  proteus_index_t ebN, eN, eN_star, left_eN, right_eN, left_eN_star, right_eN_star, nN, nN_global; int ebNI, ebNE, ebN_element, left_ebN_element, right_ebN_element, k, I;
  register double flux,fluxAverage,fluxCorrection,dx=0.0;
  /*mwf now access node star system using NodeStarFactorStruct*/
  double ** starR, ** starU;
  /*mwf add for debugging
    double * fluxSumDebug;
    double resSum;
    fluxSumDebug = calloc(nElements_global,sizeof(double));
    memset(fluxSumDebug,0,sizeof(double)*nElements_global);
  */
  /*mwf end for debugging*/
  memset(conservationResidual,0,sizeof(double)*nElements_global);
  memset(vConservative,0,sizeof(double)*(nInteriorElementBoundaries_global+nExteriorElementBoundaries_global)*nQuadraturePoints_elementBoundary*nSpace);
  /*mwf debug
    printf("calcConsResidPWL nQuadraturePoints_elementBoundary=%d \n",
    nQuadraturePoints_elementBoundary);
  */
  assert(nodeStarFactor);
  starR = nodeStarFactor->subdomain_R;
  starU = nodeStarFactor->subdomain_U;

  for (nN=nNodes_owned;nN<nodeStarFactor->N;nN++)
    for(eN=0; eN < nodeStarFactor->subdomain_dim[nN]; eN++)
      {
	starU[nN][eN]=0.0;
      }
  /*initial residual with element residual*/
  for (eN=0;eN<nElements_global;eN++)
   {
     for (nN=0;nN<nNodes_element;nN++)
       {
	 nN_global = elementNodes[eN*nNodes_element+
				  nN];
	 eN_star  = nodeStarElements[eN*nNodes_element+
				     nN];
	 starR[nN_global][eN_star] 
	   = 
	   elementResidual[eN*nNodes_element+
			   nN];
	 conservationResidual[eN]
	   += 
	   elementResidual[eN*nNodes_element+
			   nN];
	 /*mwf debug
	 printf("calcConsResPWL eN=%d nN=%d starR=%g\n",eN,nN,
		starR[nN_global][eN_star]);
	 */
       }
     /*mwf debug
       printf("calcConsResPWL eN=%d consRes=%g\n",eN,
       conservationResidual[eN]);
     */
   }
  /*calculate interior element boundary fluxes and update residual*/
  for (ebNI=0;ebNI<nInteriorElementBoundaries_global;ebNI++)
    {
      ebN = interiorElementBoundaries[ebNI];
      left_eN = elementBoundaryElements[ebN*2+0];
      right_eN = elementBoundaryElements[ebN*2+1];
      left_ebN_element = elementBoundaryLocalElementBoundaries[ebN*2+0];
      right_ebN_element = elementBoundaryLocalElementBoundaries[ebN*2+1];
      for(k=0;k<nQuadraturePoints_elementBoundary;k++)
	{
          fluxAverage = 0.0;
          /* get the integration weight */
          dx = dX[left_eN*nElementBoundaries_element*nQuadraturePoints_elementBoundary+
		  left_ebN_element*nQuadraturePoints_elementBoundary+
		  k];
	  /*mwf debug
	    printf("calcConsResPWL ebNI=%d k=%d dx=%g \n",ebNI,k,dx);
	  */
	  for (I=0;I<nSpace;I++)
	    {

              fluxAverage
                +=
                vAverage[ebN*nQuadraturePoints_elementBoundary*nSpace+
                         k*nSpace+
                         I]
                *
                normal[ebN*nQuadraturePoints_elementBoundary*nSpace+
                       k*nSpace+
                       I];
            }
          for (nN=0;nN<nNodes_element;nN++)
	    {
	      nN_global = elementNodes[left_eN*nNodes_element+
				      nN];
	      left_eN_star = nodeStarElements[left_eN*nNodes_element+
					      nN];
	      /* check if node is opposite element boundary we're computing and ignore 0 contribution */
	      /* this shouldn't be necessary */
	      if (nN != left_ebN_element)
		{
		  right_eN_star = nodeStarElementNeighbors[left_eN*nNodes_element*nElementBoundaries_element+
							   nN*nElementBoundaries_element+
							   left_ebN_element];
                  fluxCorrection = (starU[nN_global][left_eN_star]
                                    -
                                    starU[nN_global][right_eN_star])
                    *
                    w[left_eN*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nNodes_element+
                      left_ebN_element*nQuadraturePoints_elementBoundary*nNodes_element+
                      k*nNodes_element+
                      nN];
		  for (I=0;I<nSpace;I++)
		    {
		      if (nN_global < nNodes_owned)
			{
			  vConservative[ebN*nQuadraturePoints_elementBoundary*nSpace+
				k*nSpace+
				I] 
			    +=
			    fluxCorrection
			    *
			    normal[ebN*nQuadraturePoints_elementBoundary*nSpace+
				   k*nSpace+
				   I];
			}
		    }
		  flux = (fluxAverage*w[left_eN*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nNodes_element+
                                        left_ebN_element*nQuadraturePoints_elementBoundary*nNodes_element+
                                        k*nNodes_element+
                                        nN]
                          + fluxCorrection)*dx;
		  starR[nN_global][left_eN_star]
		    += flux;
		  starR[nN_global][right_eN_star]
		    -= flux;
		  conservationResidual[left_eN] += flux;
		  conservationResidual[right_eN] -= flux;
		  /*mwf debug
		  printf("ppwl nN_global=%d left_eN=%d right_eN=%d fluxAvg=%g fluxCorr=%g flux=%g \n",
			 nN_global,left_eN,right_eN,fluxAverage,fluxCorrection,flux);
		  */
		  /*mwf debug
		  fluxSumDebug[left_eN] += flux;
		  fluxSumDebug[right_eN]-= flux;
		  */
		}
	    }
	}
    }
  /*calculate fluxes on exterior element boundaries and update residual*/
  for (ebNE=0;ebNE<nExteriorElementBoundaries_global;ebNE++)
    {
      ebN = exteriorElementBoundaries[ebNE];
      eN = elementBoundaryElements[ebN*2+0];
      ebN_element = elementBoundaryLocalElementBoundaries[ebN*2+0];
      
      for(k=0;k<nQuadraturePoints_elementBoundary;k++)
	{
          /* get the integration weight, I don't think this was here before */
	  dx = dX[eN*nElementBoundaries_element*nQuadraturePoints_elementBoundary+
		  ebN_element*nQuadraturePoints_elementBoundary+
		  k];
	  
          fluxAverage=0.0;
	  for (I=0;I<nSpace;I++)
            {
              fluxAverage += 
                vAverage[ebN*nQuadraturePoints_elementBoundary*nSpace+
                         k*nSpace+
                         I]
                *
                normal[ebN*nQuadraturePoints_elementBoundary*nSpace+
                       k*nSpace+
                       I];
            }
	  for (nN=0;nN<nNodes_element;nN++)
	    {
	      nN_global = elementNodes[eN*nNodes_element+
				       nN];
	      eN_star = nodeStarElements[eN*nNodes_element+
					      nN];
	      /* check if this node lies opposite the element boundary whose contribution we're computing */
	      /* in that case there is no flux contribution because the test function is zero*/
	      /* however, we may be able to remove this conditional for speed */
	      if (nN != ebN_element && !fluxElementBoundaries[ebNE])/*mwf add skip for flux*/ 
		{
                  fluxCorrection = starU[nN_global][eN_star]
                    *
                    w[eN*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nNodes_element+
                      ebN_element*nQuadraturePoints_elementBoundary*nNodes_element+
                      k*nNodes_element+
                      nN]; 
		  if (nN_global < nNodes_owned)
		    {
		      for (I=0;I<nSpace;I++)
			{
			  vConservative[ebN*nQuadraturePoints_elementBoundary*nSpace+
				k*nSpace+
				I] +=
			    fluxCorrection
			    *
			    normal[ebN*nQuadraturePoints_elementBoundary*nSpace+
				   k*nSpace+
				   I];
			}
		    }
		  flux = (fluxAverage
                          *
                          w[eN*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nNodes_element+
                            ebN_element*nQuadraturePoints_elementBoundary*nNodes_element+
                            k*nNodes_element+
                            nN]
                          +
                          fluxCorrection)*dx;
		  starR[nN_global][eN_star]
		    += flux;
		  conservationResidual[eN] += flux;
		}
	    }
	}
    }
}

void calculateConservationResidualPWL_primative(proteus_index_t nElements_global,
						proteus_index_t nInteriorElementBoundaries_global,
						proteus_index_t nExteriorElementBoundaries_global,
						int nElementBoundaries_element,
						int nQuadraturePoints_elementBoundary,
						int nNodes_element,
						int nSpace,
						proteus_index_t* interiorElementBoundaries,
						proteus_index_t* exteriorElementBoundaries,
						proteus_index_t* elementBoundaryElements,
						int* elementBoundaryLocalElementBoundaries,
						int* skipflag_elementBoundaries,
						double* elementResidual,
						double* dX,
						double* normal,
						double* conservationResidual,
						double* vConservative)
{
  proteus_index_t ebN, eN, left_eN, right_eN, nN; int ebNI, ebNE, ebN_element, left_ebN_element, right_ebN_element, k, I;
  register double flux,divergence=0.0;
  memset(conservationResidual,0,sizeof(double)*nElements_global);
  /* /\*initialize residual with element residual and assume external flux terms are in the element residual*\/ */
  /* for (eN=0;eN<nElements_global;eN++) */
  /*  { */
  /*    for (nN=0;nN<nNodes_element;nN++) */
  /*      { */
  /* 	 conservationResidual[eN] */
  /* 	   += */
  /* 	   elementResidual[eN*nNodes_element+ */
  /* 			   nN]; */
  /*      } */
  /*  } */
  /*calculate interior element boundary fluxes and update residual*/
  for (ebNI=0;ebNI<nInteriorElementBoundaries_global;ebNI++)
    {
      ebN = interiorElementBoundaries[ebNI];
      left_eN = elementBoundaryElements[ebN*2+0];
      right_eN = elementBoundaryElements[ebN*2+1];
      left_ebN_element = elementBoundaryLocalElementBoundaries[ebN*2+0];
      right_ebN_element = elementBoundaryLocalElementBoundaries[ebN*2+1];
      for(k=0;k<nQuadraturePoints_elementBoundary;k++)
	{
          flux = 0.0;
	  for (I=0;I<nSpace;I++)
	    {
              flux
                +=
                vConservative[ebN*nQuadraturePoints_elementBoundary*nSpace+
			      k*nSpace+
			      I]
                *
                normal[ebN*nQuadraturePoints_elementBoundary*nSpace+
                       k*nSpace+
                       I];
            }
	  flux*=dX[left_eN*nElementBoundaries_element*nQuadraturePoints_elementBoundary+
		   left_ebN_element*nQuadraturePoints_elementBoundary+
		   k];
	  conservationResidual[left_eN]  += flux;
	  conservationResidual[right_eN] -= flux;
	}
    }
  /*calculate fluxes on exterior element boundaries and update residual*/
  for (ebNE=0;ebNE<nExteriorElementBoundaries_global;ebNE++)
    {
      ebN = exteriorElementBoundaries[ebNE];
      eN = elementBoundaryElements[ebN*2+0];
      ebN_element = elementBoundaryLocalElementBoundaries[ebN*2+0];
      for(k=0;k<nQuadraturePoints_elementBoundary;k++)
	{
	  flux=0.0;
	  for (I=0;I<nSpace;I++)
            {
              flux += 
                vConservative[ebN*nQuadraturePoints_elementBoundary*nSpace+
			      k*nSpace+
			      I]
                *
                normal[ebN*nQuadraturePoints_elementBoundary*nSpace+
                       k*nSpace+
                       I];
            }
	  flux*=dX[eN*nElementBoundaries_element*nQuadraturePoints_elementBoundary+
		   ebN_element*nQuadraturePoints_elementBoundary+
		   k];
	  conservationResidual[eN] += flux;
	  divergence += flux;
	}
    }
}

void calculateConservationJacobianPWL_opt(proteus_index_t nNodes_owned,
				      proteus_index_t nNodes_global,
				      proteus_index_t nNodes_internal,
				      proteus_index_t nElements_global,
				      proteus_index_t nInteriorElementBoundaries_global,
				      proteus_index_t nExteriorElementBoundaries_global,
				      int nElementBoundaries_element,
				      int nQuadraturePoints_elementBoundary,
				      int nNodes_element,
				      int nSpace,
				      proteus_index_t* interiorElementBoundaries,
				      proteus_index_t* exteriorElementBoundaries,
				      proteus_index_t* elementBoundaryElements,
				      int* elementBoundaryLocalElementBoundaries,
				      proteus_index_t* elementNodes,
				      proteus_index_t* nodeStarElements,
				      proteus_index_t* nodeStarElementNeighbors,
				      int* nElements_node,
				      proteus_index_t* internalNodes,
				      proteus_index_t* fluxElementBoundaries,
				      int* fluxBoundaryNodes,
				      double* w,
				      double* normal,
				      NodeStarFactorStruct* nodeStarFactor)

{
  proteus_index_t eN, ebN, eN_star, left_eN, right_eN, left_eN_star, right_eN_star, nN, nN_global; int ebNI, ebNE, ebN_element, left_ebN_element, right_ebN_element, nNI, k;
  PROTEUS_LAPACK_INTEGER INFO=0;
  register double wflux;
  /*mwf add for boundaries*/
  int ii,jj;
  double ** starJacobian = nodeStarFactor->subdomain_L;
  int * subdomain_dim    = nodeStarFactor->subdomain_dim;
  PROTEUS_LAPACK_INTEGER ** starPivots = nodeStarFactor->subdomain_pivots;
  /*for full pivoting, to avoid pathological bow-tie nodes*/
  PROTEUS_LAPACK_INTEGER ** starColPivots = nodeStarFactor->subdomain_column_pivots;
 
  /*zero everything for safety*/
  assert(nodeStarFactor);
  for (nN = 0; nN < nNodes_global; nN++)
    {
      for(ii=0; ii < subdomain_dim[nN]; ii++)
	{
	  starPivots[nN][ii]=0;
	  starColPivots[nN][ii]=0;
	  for (jj=0; jj < subdomain_dim[nN]; jj++)
	    starJacobian[nN][ii + jj*subdomain_dim[nN]] = 0.0;
	}
    }
  /*Load Jacobian entries arising from iterior element boundaries*/
  for (ebNI=0;ebNI<nInteriorElementBoundaries_global;ebNI++)
    {
      ebN = interiorElementBoundaries[ebNI];
      left_eN = elementBoundaryElements[ebN*2+0];
      right_eN = elementBoundaryElements[ebN*2+1];
      left_ebN_element = elementBoundaryLocalElementBoundaries[ebN*2+0];
      right_ebN_element = elementBoundaryLocalElementBoundaries[ebN*2+1];
      for(k=0;k<nQuadraturePoints_elementBoundary;k++)
	{
	  for (nN=0;nN<nNodes_element;nN++)
	    {
	      nN_global = elementNodes[left_eN*nNodes_element+
				      nN];
	      left_eN_star = nodeStarElements[left_eN*nNodes_element+
					      nN];
	      /* check if node is opposite element boundary we're computing and ignore 0 contribution */
	      /* this shouldn't be necessary */
	      if (nN != left_ebN_element  && nN_global < nNodes_owned)
		{
		  right_eN_star = nodeStarElementNeighbors[left_eN*nNodes_element*nElementBoundaries_element+
							   nN*nElementBoundaries_element+
							   left_ebN_element];
		  wflux = w[left_eN*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nNodes_element+
                            left_ebN_element*nQuadraturePoints_elementBoundary*nNodes_element+
                            k*nNodes_element+
                            nN];
                  
		  starJacobian[nN_global][left_eN_star+
					  left_eN_star*subdomain_dim[nN_global]]
		    += wflux;
		  starJacobian[nN_global][left_eN_star+
					  right_eN_star*subdomain_dim[nN_global]]
		    -= wflux;
		  starJacobian[nN_global][right_eN_star+
					  left_eN_star*subdomain_dim[nN_global]]
		    -= wflux;
		  starJacobian[nN_global][right_eN_star+
					  right_eN_star*subdomain_dim[nN_global]]
		    += wflux;
		}
	    }
	}
    }  
  /*Load Jacobian entries arising from exterior element boundaries*/
  for (ebNE=0;ebNE<nExteriorElementBoundaries_global;ebNE++)
    {
      ebN = exteriorElementBoundaries[ebNE];
      eN = elementBoundaryElements[ebN*2+0];
      ebN_element = elementBoundaryLocalElementBoundaries[ebN*2+0];
      for(k=0;k<nQuadraturePoints_elementBoundary;k++)
	{
	  for (nN=0;nN<nNodes_element;nN++)
	    {
	      nN_global = elementNodes[eN*nNodes_element+
				       nN];
	      eN_star = nodeStarElements[eN*nNodes_element+
					 nN];
	      /* check if this node lies opposite the element boundary whose contribution we're computing */
	      /* in that case there is no flux contribution because the test function is zero*/
	      /* however, we may be able to remove this conditional for speed */
              /* cek modified to be: also exclude if the flux is specified on this boundary AND nN is a free node */
	      /* mwf try to change definition of fluxElementBoundaries so that they are excluded if they contain a node
                 that is Dirichlet but has no "Dirichlet" faces associated with it 
		 if (nN != ebN_element && !(fluxElementBoundaries[ebNE] && fluxBoundaryNodes[nN_global]) )*/
	      if (nN != ebN_element && !fluxElementBoundaries[ebNE] && nN_global < nNodes_owned)
		{
		  starJacobian[nN_global][eN_star+
					  eN_star*nElements_node[nN_global]]
		    += 
                    w[eN*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nNodes_element+
                      ebN_element*nQuadraturePoints_elementBoundary*nNodes_element+
                      k*nNodes_element+
                      nN];
		}
	    }
	}
    }
  /*set Dirichlet boundary conditions at one element on interior node stars*/
  for (nNI=0;nNI<nNodes_internal;nNI++)
    {
      nN = internalNodes[nNI];
      if (nN < nNodes_owned)
	{
	  starJacobian[nN][0]=1.0;
	  for (eN=1;eN<nElements_node[nN];eN++)
	    starJacobian[nN][eN*subdomain_dim[nN]]
	      =0.0;
	}
    }
  /*repeat for Neumann boundary node stars*/
  for (nN=0; nN < nNodes_owned; nN++)
    {
      if (fluxBoundaryNodes[nN]==1)
	{
	  starJacobian[nN][0]=1.0;
	  for (eN=1;eN<nElements_node[nN];eN++)
	    starJacobian[nN][eN*subdomain_dim[nN]]
	      =0.0;

	  /*mwf debug
	    printf("jacobian fluxNode=%d nElements_node=%d \n",nN,nElements_node[nN]);
	  */
	}
    }

  /*factor with lapack*/
  for (nN=0;nN<nNodes_owned;nN++)
    {
      PROTEUS_LAPACK_INTEGER nE_n = ((PROTEUS_LAPACK_INTEGER)subdomain_dim[nN]);
/*       dgetrf_(&nE_n, */
/*               &nE_n, */
/*               starJacobian[nN], */
/*               &nE_n, */
/*               starPivots[nN], */
/*               &INFO); */
      /*could try using dgetc2 and dgesc2 to solve when have a zero row because of
	isolated nodes*/
      dgetc2_(&nE_n,
              starJacobian[nN],
              &nE_n,
              starPivots[nN],
              starColPivots[nN],
              &INFO);
      
      /*mwf debug*/
	if (INFO > 0)
	  {
            printf("velPP jac dgetrf INFO=%d nN=%d \n",(int)(INFO),nN);
            for (ii=0;ii<nE_n;ii++)
              {
                for(jj=0;jj<nE_n;jj++)
                  {
                    
                    printf("%12.5e \t",starJacobian[nN][ii*nE_n + jj]);
                  }
                printf("\n");
              }
          }
	/*assert(INFO == 0);*//*need to turn off if use dgetc2*/
    }
}
void calculateConservationFluxPWL_opt(proteus_index_t nNodes_owned,
				  proteus_index_t nNodes_global,
				  proteus_index_t nNodes_internal,
				  int* nElements_node,
				  proteus_index_t* internalNodes,
				  int* fluxBoundaryNodes,
				  NodeStarFactorStruct* nodeStarFactor)
{
  proteus_index_t nN, eN; int nNI;
  PROTEUS_LAPACK_INTEGER NRHS=1,INFO=0;
  double ** starJ = nodeStarFactor->subdomain_L;
  double ** starR = nodeStarFactor->subdomain_R;
  double ** starU = nodeStarFactor->subdomain_U;
  PROTEUS_LAPACK_INTEGER ** starPivots = nodeStarFactor->subdomain_pivots;
  /*for full pivoting, to avoid pathological bow-tie nodes*/
  PROTEUS_LAPACK_INTEGER ** starColPivots = nodeStarFactor->subdomain_column_pivots;
  double scale;
  int * subdomain_dim = nodeStarFactor->subdomain_dim;
  char TRANS='N';
  /*load -R into U*/
  for (nN=0;nN<nNodes_owned;nN++)
    for(eN=0;eN<subdomain_dim[nN];eN++)
      starU[nN][eN] = -starR[nN][eN];
  /*set Dirichlet boundary conditions on interior node stars*/
  for (nNI=0;nNI<nNodes_internal;nNI++)
    {
      nN = internalNodes[nNI];
      if (nN < nNodes_owned)
	starU[nN][0]=0.0;
    }
  /*repeat for Neumann boundary node stars*/
  for (nN=0; nN < nNodes_owned; nN++)
    {
      if (fluxBoundaryNodes[nN] == 1)
	{
	  starU[nN][0]=0.0;
	}
    }
  /*solve with lapack*/
  for  (nN=0;nN<nNodes_owned;nN++)
    {
      PROTEUS_LAPACK_INTEGER nE_n = ((PROTEUS_LAPACK_INTEGER)subdomain_dim[nN]);
      /*mwf orig, partial pivoting*/
/*       dgetrs_(&TRANS, */
/*               &nE_n, */
/*               &NRHS, */
/*               starJ[nN], */
/*               &nE_n, */
/*               starPivots[nN], */
/*               starU[nN], */
/*               &nE_n, */
/*               &INFO); */
      /*could try using dgetc2 and dgesc2 to solve when have a zero row because of
	isolated nodes*/
	
	dgesc2_(&nE_n,
		starJ[nN],
		&nE_n,
		starU[nN],
		starPivots[nN],
		starColPivots[nN],
		&scale);
    }
}
void subdomain_U_copy_global2local(int max_nN_owned,
                                   proteus_index_t nElements_global,
                                   int nNodes_element,
                                   proteus_index_t* elementNodes,
                                   proteus_index_t* nodeStarElements,
                                   NodeStarFactorStruct* nodeStarFactor,
				   double* subdomain_U)
{
  proteus_index_t eN, nN, eN_star, nN_global;
  double ** starU = nodeStarFactor->subdomain_U;
  for (eN=0;eN<nElements_global;eN++)
    for (nN=0;nN<nNodes_element;nN++)
      {
        nN_global = elementNodes[eN*nNodes_element+
                                 nN];
        eN_star  = nodeStarElements[eN*nNodes_element+
                                    nN];
        starU[nN_global][eN_star] 
          = 
          subdomain_U[eN*nNodes_element+
                     nN];
      }
}

void subdomain_U_copy_local2global(int max_nN_owned,
                                   proteus_index_t nElements_global,
                                   int nNodes_element,
                                   proteus_index_t* elementNodes,
                                   proteus_index_t* nodeStarElements,
                                   NodeStarFactorStruct* nodeStarFactor,
				   double* subdomain_U)
{
  proteus_index_t eN, nN, eN_star, nN_global;
  double ** starU = nodeStarFactor->subdomain_U;
  for (eN=0;eN<nElements_global;eN++)
    for (nN=0;nN<nNodes_element;nN++)
      {

        nN_global = elementNodes[eN*nNodes_element+
                                 nN];
        eN_star  = nodeStarElements[eN*nNodes_element+
                                    nN];
        if (nN_global < max_nN_owned)
          subdomain_U[eN*nNodes_element+
                      nN] =
            starU[nN_global][eN_star];
        else//throw out correction if this node star isn't owned
          {
            subdomain_U[eN*nNodes_element+
                        nN] = 0.0;
          }
      }
}

/**
   \brief Update the element boundary flux on exterior element boundaries
*/
void updateSelectedExteriorElementBoundaryFlux(proteus_index_t nExteriorElementBoundaries_global,
					       int nElementBoundaries_element,
					       int nQuadraturePoints_elementBoundary,
					       int nDOF_test_element,
					       proteus_index_t* exteriorElementBoundaries,
					       proteus_index_t* elementBoundaryElements,
					       int* elementBoundaryLocalElementBoundaries,
					       int* skipflag_elementBoundaries,
					       double* flux,
					       double* w_dS,
					       double* residual)
{
  proteus_index_t ebN, eN_global; int ebNE, ebN_element, i, k;
  for(ebNE=0;ebNE<nExteriorElementBoundaries_global;ebNE++)
    {
      ebN = exteriorElementBoundaries[ebNE];
      eN_global = elementBoundaryElements[ebN*2+0];
      ebN_element = elementBoundaryLocalElementBoundaries[ebN*2+0];
      if (skipflag_elementBoundaries[ebNE] == 0)
	{
	  for(i=0;i<nDOF_test_element;i++)
	    for(k=0;k<nQuadraturePoints_elementBoundary;k++)
	      {
		/*mwf debug
		if (fabs(flux[ebN*nQuadraturePoints_elementBoundary+
		k]) > 0.0)
		{
		printf("postproc updateSelectedExtBnd ebN=%d k=%d eN = %d flux=%g\n",ebN,k,eN_global,
		flux[ebN*nQuadraturePoints_elementBoundary+
		k]);
		}
		mwf end debug*/
		residual[eN_global*nDOF_test_element+
			 i]
		  +=
		  flux[ebN*nQuadraturePoints_elementBoundary+
		       k]
		  *
		  w_dS[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_test_element+
		       ebN_element*nQuadraturePoints_elementBoundary*nDOF_test_element+
		       k*nDOF_test_element+
		       i];
	      }
	}
    }/*ebNE*/
}

void postprocessAdvectiveVelocityPointEval(int nPoints,
					   int nSpace,
					   double updateCoef,
					   const double* f,
					   double * velocity)
{
  /*advective portion of pointwise evaluation of velocity 
    v = -\ten{a}_h\grad phi_h + \vec f_h

    q_a and or q_f may be null if not defined for system
    (only reason why calculateFlowVelocity wouldnt work)
  */
  int k,I;
  for (k = 0; k < nPoints; k++)
    {
      for (I = 0; I < nSpace; I++)
	{
	  velocity[k*nSpace + I] *= updateCoef;
	  velocity[k*nSpace + I] += f[k*nSpace + I];
	}/*I*/
    }/*k*/

}
void postprocessDiffusiveVelocityPointEval(int nPoints,
					   int nSpace,
					   double updateCoef,
					   const double* a,
					   const double* grad_phi,
					   double * velocity)
{
  /*diffusive velocity part of pointwise evaluation of velocity 
    v = -\ten{a}_h\grad phi_h + \vec f_h

    q_a and or q_f may be null if not defined for system
    (only reason why calculateFlowVelocity wouldnt work)
  */
  proteus_index_t eN; int k, I, J;
  const int nSpace2 = nSpace*nSpace;
  for (k = 0; k < nPoints; k++)
    {
      for (I = 0; I < nSpace; I++)
	{
	  velocity[k*nSpace + I] *= updateCoef;
	  for (J=0; J < nSpace; J++)
	    {
	      velocity[k*nSpace + I] -=
		a[k*nSpace2 + I*nSpace + J]
		*
		grad_phi[k*nSpace + J];
	    }/*J*/
	}/*I*/
    }/*k*/
}

void postprocessDiffusiveVelocityPointEval_sd(int nPoints,
					      int nSpace,
					      double updateCoef,
					      proteus_csr_index_t* rowptr,
					      proteus_csr_index_t* colind,
					      const double* a,
					      const double* grad_phi,
					      double * velocity)
{
  /*diffusive velocity part of pointwise evaluation of velocity 
    v = -\ten{a}_h\grad phi_h + \vec f_h

    q_a and or q_f may be null if not defined for system
    (only reason why calculateFlowVelocity wouldnt work)
  */
  proteus_index_t eN; proteus_csr_index_t m; int k, I;
  const proteus_csr_index_t nnz=rowptr[nSpace];
  for (k = 0; k < nPoints; k++)
    {
      for (I = 0; I < nSpace; I++)
	{
	  velocity[k*nSpace + I] *= updateCoef;
	  for(m=rowptr[I];m<rowptr[I+1];m++)
	    {
	      velocity[k*nSpace + I] -=
		a[k*nnz+m]
		*
		grad_phi[k*nSpace + colind[m]];
	    }/*J*/
	}/*I*/
    }/*k*/
}

void calculateElementResidualPWL(proteus_index_t nElements, int nDOF_element_res, int nDOF_element_resPWL,double* alpha, double* elementResidual, double* elementResidualPWL)
{
  proteus_index_t eN; int i, j;
  memset(elementResidualPWL,0,sizeof(double)*nElements*nDOF_element_resPWL);
  for(eN=0;eN<nElements;eN++)
    for(i=0;i<nDOF_element_resPWL;i++)
      for(j=0;j<nDOF_element_res;j++)
	elementResidualPWL[eN*nDOF_element_resPWL+i] += alpha[i*nDOF_element_res+j]*elementResidual[eN*nDOF_element_res+j];
}


/***********************************************************************
  version of Swedish Postprocessing with internal Neumann boundaries
  enforced explicitly 

  This version, 
    skips flux boundaries in element integrals: calculateConservationResidualPWL,
                                                calculateConservationJacobianPWL


    also will use this with routines that do NOT modify jacobians for
      node-stars since it uses dgetc2 Lapack routines that have column
      pivoting and can handle the simple singularity

  When using this version, 
     do not remove boundary flux terms from element residual
     must load in boundary fluxes into ebq_global velocity and ebq velocity
      after calling

  Note,
    fluxElementBoundaries holds global element boundary ids for flux bcs

 ***********************************************************************/
void calculateConservationResidualPWL_interiorBoundaries(proteus_index_t nElements_global,
							 proteus_index_t nInteriorElementBoundaries_global,
							 proteus_index_t nExteriorElementBoundaries_global,
							 int nElementBoundaries_element,
							 int nQuadraturePoints_elementBoundary,
							 int nNodes_element,
							 int nSpace,
							 proteus_index_t* interiorElementBoundaries,
							 proteus_index_t* exteriorElementBoundaries,
							 proteus_index_t* elementBoundaryElements,
							 int* elementBoundaryLocalElementBoundaries,
							 proteus_index_t* elementNodes,
							 proteus_index_t* nodeStarElements,
							 proteus_index_t* nodeStarElementNeighbors,
							 int* nElements_node,
							 proteus_index_t* fluxElementBoundaries,
							 double* elementResidual,
							 double* vAverage,
							 double* dX,
							 double* w,
							 double* normal,
							 NodeStarFactorStruct* nodeStarFactor,
							 double* conservationResidual,
							 double* vConservative,
							 double* vConservative_element)
{
  proteus_index_t ebN, eN, eN_star, left_eN, right_eN, left_eN_star, right_eN_star, nN, nN_global; int ebNI, ebNE, ebN_element, left_ebN_element, right_ebN_element, k, I;
  register double flux,fluxAverage,fluxCorrection,dx=0.0;
  /*mwf now access node star system using NodeStarFactorStruct*/
  double ** starR, ** starU;
  /*mwf add for debugging
    double * fluxSumDebug;
    double resSum;
    fluxSumDebug = calloc(nElements_global,sizeof(double));
    memset(fluxSumDebug,0,sizeof(double)*nElements_global);
  */
  /*mwf end for debugging*/
  memset(conservationResidual,0,sizeof(double)*nElements_global);
  /*mwf debug
    printf("calcConsResidPWL nQuadraturePoints_elementBoundary=%d \n",
    nQuadraturePoints_elementBoundary);
  */
  assert(nodeStarFactor);
  starR = nodeStarFactor->subdomain_R;
  starU = nodeStarFactor->subdomain_U;

  /*initial residual with element residual*/
  for (eN=0;eN<nElements_global;eN++)
   {
     for (nN=0;nN<nNodes_element;nN++)
       {
	 nN_global = elementNodes[eN*nNodes_element+
				  nN];
	 eN_star  = nodeStarElements[eN*nNodes_element+
				     nN];
	 starR[nN_global][eN_star] 
	   = 
	   elementResidual[eN*nNodes_element+
			   nN];
	 conservationResidual[eN]
	   += 
	   elementResidual[eN*nNodes_element+
			   nN];
	 /*mwf debug
	 printf("calcConsResPWL eN=%d nN=%d starR=%g\n",eN,nN,
		starR[nN_global][eN_star]);
	 */
       }
     /*mwf debug
       printf("calcConsResPWL eN=%d consRes=%g\n",eN,
       conservationResidual[eN]);
     */
   }
  /*calculate interior element boundary fluxes and update residual*/
  for (ebNI=0;ebNI<nInteriorElementBoundaries_global;ebNI++)
    {
      ebN = interiorElementBoundaries[ebNI];
      left_eN = elementBoundaryElements[ebN*2+0];
      right_eN = elementBoundaryElements[ebN*2+1];
      left_ebN_element = elementBoundaryLocalElementBoundaries[ebN*2+0];
      right_ebN_element = elementBoundaryLocalElementBoundaries[ebN*2+1];
      for(k=0;k<nQuadraturePoints_elementBoundary;k++)
	{
          fluxAverage = 0.0;
          /* get the integration weight */
          dx = dX[left_eN*nElementBoundaries_element*nQuadraturePoints_elementBoundary+
		  left_ebN_element*nQuadraturePoints_elementBoundary+
		  k];
	  /*mwf debug
	    printf("calcConsResPWL ebNI=%d k=%d dx=%g \n",ebNI,k,dx);
	  */
	  for (I=0;I<nSpace;I++)
	    {
              vConservative[ebN*nQuadraturePoints_elementBoundary*nSpace+
                            k*nSpace+
                            I] 
                =
                vAverage[ebN*nQuadraturePoints_elementBoundary*nSpace+
                         k*nSpace+
                         I];
	      /*mwf debug
		printf("pwl get flux vAverage int I=%d, val=%g \n",I,
		vAverage[ebN*nQuadraturePoints_elementBoundary*nSpace+
		k*nSpace+
		I]);
	      */
              fluxAverage
                +=
                vAverage[ebN*nQuadraturePoints_elementBoundary*nSpace+
                         k*nSpace+
                         I]
                *
                normal[ebN*nQuadraturePoints_elementBoundary*nSpace+
                       k*nSpace+
                       I];
            }
          for (nN=0;nN<nNodes_element;nN++)
	    {
	      nN_global = elementNodes[left_eN*nNodes_element+
				      nN];
	      left_eN_star = nodeStarElements[left_eN*nNodes_element+
					      nN];
	      /* check if node is opposite element boundary we're computing and ignore 0 contribution */
	      /* also skip if the face is an internal boundary*/
	      if (nN != left_ebN_element && !fluxElementBoundaries[ebN])
		{
		  right_eN_star = nodeStarElementNeighbors[left_eN*nNodes_element*nElementBoundaries_element+
							   nN*nElementBoundaries_element+
							   left_ebN_element];
                  fluxCorrection = (starU[nN_global][left_eN_star]
                                    -
                                    starU[nN_global][right_eN_star])
                    *
                    w[left_eN*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nNodes_element+
                      left_ebN_element*nQuadraturePoints_elementBoundary*nNodes_element+
                      k*nNodes_element+
                      nN];
		  for (I=0;I<nSpace;I++)
		    {
		      vConservative[ebN*nQuadraturePoints_elementBoundary*nSpace+
				    k*nSpace+
				    I] 
                        +=
			fluxCorrection
			*
			normal[ebN*nQuadraturePoints_elementBoundary*nSpace+
                               k*nSpace+
                               I];
		    }
		  flux = (fluxAverage*w[left_eN*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nNodes_element+
                                        left_ebN_element*nQuadraturePoints_elementBoundary*nNodes_element+
                                        k*nNodes_element+
                                        nN]
                          + fluxCorrection)*dx;
		  starR[nN_global][left_eN_star]
		    += flux;
		  starR[nN_global][right_eN_star]
		    -= flux;
		  conservationResidual[left_eN] += flux;
		  conservationResidual[right_eN] -= flux;
		  /*mwf debug
		  printf("ppwl nN_global=%d left_eN=%d right_eN=%d fluxAvg=%g fluxCorr=%g flux=%g \n",
			 nN_global,left_eN,right_eN,fluxAverage,fluxCorrection,flux);
		  */
		  /*mwf debug
		  fluxSumDebug[left_eN] += flux;
		  fluxSumDebug[right_eN]-= flux;
		  */
		}
	    }
	}
    }
  /*calculate fluxes on exterior element boundaries and update residual*/
  for (ebNE=0;ebNE<nExteriorElementBoundaries_global;ebNE++)
    {
      ebN = exteriorElementBoundaries[ebNE];
      eN = elementBoundaryElements[ebN*2+0];
      ebN_element = elementBoundaryLocalElementBoundaries[ebN*2+0];
      
      for(k=0;k<nQuadraturePoints_elementBoundary;k++)
	{
          /* get the integration weight, I don't think this was here before */
	  dx = dX[eN*nElementBoundaries_element*nQuadraturePoints_elementBoundary+
		  ebN_element*nQuadraturePoints_elementBoundary+
		  k];
	  
          fluxAverage=0.0;
	  for (I=0;I<nSpace;I++)
            {
              vConservative[ebN*nQuadraturePoints_elementBoundary*nSpace+
                            k*nSpace+
                            I] 
                =
                vAverage[ebN*nQuadraturePoints_elementBoundary*nSpace+
                         k*nSpace+
                         I];
	      /*mwf debug
		printf("pwl get flux vAverage ext ebN=%d free=%d I=%d, val=%g \n",ebN,fluxElementBoundaries[ebNE],
		I,vAverage[ebN*nQuadraturePoints_elementBoundary*nSpace+
		k*nSpace+
		I]);
	      */
              fluxAverage += 
                vAverage[ebN*nQuadraturePoints_elementBoundary*nSpace+
                         k*nSpace+
                         I]
                *
                normal[ebN*nQuadraturePoints_elementBoundary*nSpace+
                       k*nSpace+
                       I];
            }
	  for (nN=0;nN<nNodes_element;nN++)
	    {
	      nN_global = elementNodes[eN*nNodes_element+
				       nN];
	      eN_star = nodeStarElements[eN*nNodes_element+
					      nN];
	      /* check if this node lies opposite the element boundary whose contribution we're computing */
	      /* in that case there is no flux contribution because the test function is zero*/
	      /* however, we may be able to remove this conditional for speed */
	      if (nN != ebN_element && !fluxElementBoundaries[ebN])/*skip for flux note ebN not ebNE*/ 
		{
                  fluxCorrection = starU[nN_global][eN_star]
                    *
                    w[eN*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nNodes_element+
                      ebN_element*nQuadraturePoints_elementBoundary*nNodes_element+
                      k*nNodes_element+
                      nN]; 
		  for (I=0;I<nSpace;I++)
                    vConservative[ebN*nQuadraturePoints_elementBoundary*nSpace+
                                  k*nSpace+
                                  I] +=
                      fluxCorrection
                      *
                      normal[ebN*nQuadraturePoints_elementBoundary*nSpace+
                             k*nSpace+
                             I];
		  flux = (fluxAverage
                          *
                          w[eN*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nNodes_element+
                            ebN_element*nQuadraturePoints_elementBoundary*nNodes_element+
                            k*nNodes_element+
                            nN]
                          +
                          fluxCorrection)*dx;
		  starR[nN_global][eN_star]
		    += flux;
		  conservationResidual[eN] += flux;
		  /*mwf debug
		    fluxSumDebug[eN] += flux;
		  */
		  /*mwf debug 
		    printf("pwl get flux corr ext ebN=%d eN=%d fluxBC=%d fluxCorr=%g flux=%g \n",ebN,eN,
		    fluxElementBoundaries[ebNE],
		    fluxCorrection,flux);
		  */

	      
		}
	    }
	}
    }
  /*copy the global velocity onto the elements*/
  for (ebNI=0;ebNI<nInteriorElementBoundaries_global;ebNI++)
    {
      ebN = interiorElementBoundaries[ebNI];
      left_eN = elementBoundaryElements[ebN*2+0];
      right_eN = elementBoundaryElements[ebN*2+1];
      left_ebN_element = elementBoundaryLocalElementBoundaries[ebN*2+0];
      right_ebN_element = elementBoundaryLocalElementBoundaries[ebN*2+1];
      for(k=0;k<nQuadraturePoints_elementBoundary;k++)
        for(I=0;I<nSpace;I++)
          {
            vConservative_element[left_eN*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                                  left_ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                                  k*nSpace+
                                  I]
              =
              vConservative[ebN*nQuadraturePoints_elementBoundary*nSpace+
                            k*nSpace+
                            I];
            vConservative_element[right_eN*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                                  right_ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                                  k*nSpace+
                                  I]
              =
	      vConservative[ebN*nQuadraturePoints_elementBoundary*nSpace+
			    k*nSpace+
			    I];
	    /*mwf debug
	    printf("pwl int copy ebN=%d eN=[%d,%d] ebN_element=[%d,%d] k=%d vl[%d]=%g, vr[%d]=%g \n",
		   ebN,left_eN,right_eN,left_ebN_element,right_ebN_element,k,I,
		   vConservative_element[left_eN*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
					 left_ebN_element*nQuadraturePoints_elementBoundary*nSpace+
					 k*nSpace+
					 I],I,
		   vConservative_element[right_eN*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
					 right_ebN_element*nQuadraturePoints_elementBoundary*nSpace+
					 k*nSpace+
					 I]);
	    */

          }
    }
  /*copy the global velocity onto the elements*/
  for (ebNE=0;ebNE<nExteriorElementBoundaries_global;ebNE++)
    {
      ebN = exteriorElementBoundaries[ebNE];
      eN = elementBoundaryElements[ebN*2+0];
      ebN_element = elementBoundaryLocalElementBoundaries[ebN*2+0];
      for(k=0;k<nQuadraturePoints_elementBoundary;k++)
        for(I=0;I<nSpace;I++)
          {
	    vConservative_element[eN*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
				  ebN_element*nQuadraturePoints_elementBoundary*nSpace+
				  k*nSpace+
				  I]
	      =
	      vConservative[ebN*nQuadraturePoints_elementBoundary*nSpace+
			    k*nSpace+
			    I];
	    /*mwf debug
	    printf("pwl ext copy ebN=%d eN=%d ebN_element=%d k=%d v[%d]=%g \n",
		   ebN,eN,ebN_element,k,I,
		   vConservative_element[eN*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
					 ebN_element*nQuadraturePoints_elementBoundary*nSpace+
					 k*nSpace+
					 I]);
	    */

	  }
    }
  /*mwf debug
  for (eN=0; eN < nElements_global; eN++)
    {
      printf("leaving calcConsResPWL eN=%d consResid=%g \n",eN,conservationResidual[eN]);
    }
  */
  /*mwf debug
  for (eN=0; eN < nElements_global; eN++)
    {
      resSum = 0;
      for (ebN = 0; ebN < nNodes_element; ebN++)
	resSum += elementResidual[eN*nNodes_element+ebN];
      printf("eN=%d fluxSum=%g elemResid=%g consResid=%g \n",eN,fluxSumDebug[eN],resSum,
	     conservationResidual[eN]);
    }
  */
  /*mwf debug  
    free(fluxSumDebug);
  */
}
/***************************************************
  has internal boundaries and does not "fix" 
  node star systems for pure-neumann degeneracy
 **************************************************/
void calculateConservationJacobianPWL_interiorBoundaries(proteus_index_t nNodes_global,
							 proteus_index_t nElements_global,
							 proteus_index_t nInteriorElementBoundaries_global,
							 proteus_index_t nExteriorElementBoundaries_global,
							 int nElementBoundaries_element,
							 int nQuadraturePoints_elementBoundary,
							 int nNodes_element,
							 int nSpace,
							 proteus_index_t* interiorElementBoundaries,
							 proteus_index_t* exteriorElementBoundaries,
							 proteus_index_t* elementBoundaryElements,
							 int* elementBoundaryLocalElementBoundaries,
							 proteus_index_t* elementNodes,
							 proteus_index_t* nodeStarElements,
							 proteus_index_t* nodeStarElementNeighbors,
							 int* nElements_node,
							 proteus_index_t* fluxElementBoundaries,
							 double* w,
							 double* normal,
							 NodeStarFactorStruct* nodeStarFactor)

{
  proteus_index_t eN, ebN, eN_star, left_eN, right_eN, left_eN_star, right_eN_star, nN, nN_global; int ebNI, ebNE, ebN_element, left_ebN_element, right_ebN_element, nNI, k;
  PROTEUS_LAPACK_INTEGER INFO=0;
  register double wflux;
  /*mwf add for boundaries*/
  int ii,jj;
  double ** starJacobian = nodeStarFactor->subdomain_L;
  int * subdomain_dim    = nodeStarFactor->subdomain_dim;
  PROTEUS_LAPACK_INTEGER ** starPivots = nodeStarFactor->subdomain_pivots;
  /*for full pivoting, to avoid pathological bow-tie nodes*/
  PROTEUS_LAPACK_INTEGER ** starColPivots = nodeStarFactor->subdomain_column_pivots;
 
  /*zero everything for safety*/
  assert(nodeStarFactor);
  for (nN = 0; nN < nNodes_global; nN++)
    {
      for(ii=0; ii < subdomain_dim[nN]; ii++)
	{
	  starPivots[nN][ii]=0;
	  starColPivots[nN][ii]=0;
	  for (jj=0; jj < subdomain_dim[nN]; jj++)
	    starJacobian[nN][ii + jj*subdomain_dim[nN]] = 0.0;
	}
    }
  /*Load Jacobian entries arising from iterior element boundaries*/
  for (ebNI=0;ebNI<nInteriorElementBoundaries_global;ebNI++)
    {
      ebN = interiorElementBoundaries[ebNI];
      left_eN = elementBoundaryElements[ebN*2+0];
      right_eN = elementBoundaryElements[ebN*2+1];
      left_ebN_element = elementBoundaryLocalElementBoundaries[ebN*2+0];
      right_ebN_element = elementBoundaryLocalElementBoundaries[ebN*2+1];
      for(k=0;k<nQuadraturePoints_elementBoundary;k++)
	{
	  for (nN=0;nN<nNodes_element;nN++)
	    {
	      nN_global = elementNodes[left_eN*nNodes_element+
				      nN];
	      left_eN_star = nodeStarElements[left_eN*nNodes_element+
					      nN];
	      /* check if node is opposite element boundary we're computing and ignore 0 contribution */
	      /* this shouldn't be necessary */
	      /* skip if flux boundary (now interior or exterior */
	      if (nN != left_ebN_element && !fluxElementBoundaries[ebN])
		{
		  right_eN_star = nodeStarElementNeighbors[left_eN*nNodes_element*nElementBoundaries_element+
							   nN*nElementBoundaries_element+
							   left_ebN_element];
		  wflux = w[left_eN*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nNodes_element+
                            left_ebN_element*nQuadraturePoints_elementBoundary*nNodes_element+
                            k*nNodes_element+
                            nN];
                  
		  starJacobian[nN_global][left_eN_star+
					  left_eN_star*subdomain_dim[nN_global]]
		    += wflux;
		  starJacobian[nN_global][left_eN_star+
					  right_eN_star*subdomain_dim[nN_global]]
		    -= wflux;
		  starJacobian[nN_global][right_eN_star+
					  left_eN_star*subdomain_dim[nN_global]]
		    -= wflux;
		  starJacobian[nN_global][right_eN_star+
					  right_eN_star*subdomain_dim[nN_global]]
		    += wflux;
		}
	    }
	}
    }  
  /*Load Jacobian entries arising from exterior element boundaries*/
  for (ebNE=0;ebNE<nExteriorElementBoundaries_global;ebNE++)
    {
      ebN = exteriorElementBoundaries[ebNE];
      eN = elementBoundaryElements[ebN*2+0];
      ebN_element = elementBoundaryLocalElementBoundaries[ebN*2+0];
      for(k=0;k<nQuadraturePoints_elementBoundary;k++)
	{
	  for (nN=0;nN<nNodes_element;nN++)
	    {
	      nN_global = elementNodes[eN*nNodes_element+
				       nN];
	      eN_star = nodeStarElements[eN*nNodes_element+
					 nN];
	      /* check if this node lies opposite the element boundary whose contribution we're computing */
	      /* in that case there is no flux contribution because the test function is zero*/
	      /* however, we may be able to remove this conditional for speed */
	      /* skip if flux boundary (now interior or exterior */
	      if (nN != ebN_element && !fluxElementBoundaries[ebN])
		{
		  starJacobian[nN_global][eN_star+
					  eN_star*nElements_node[nN_global]]
		    += 
                    w[eN*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nNodes_element+
                      ebN_element*nQuadraturePoints_elementBoundary*nNodes_element+
                      k*nNodes_element+
                      nN];
		}
	    }
	}
    }

  /*factor with lapack*/
  for (nN=0;nN<nNodes_global;nN++)
    {
      PROTEUS_LAPACK_INTEGER nE_n = ((PROTEUS_LAPACK_INTEGER)subdomain_dim[nN]);
      dgetc2_(&nE_n,
              starJacobian[nN],
              &nE_n,
              starPivots[nN],
              starColPivots[nN],
              &INFO);
      
      /*mwf debug
	if (INFO > 0)
	  {
            printf("velPP jac dgetrf INFO=%d nN=%d \n",(int)(INFO),nN);
            for (ii=0;ii<nE_n;ii++)
              {
                for(jj=0;jj<nE_n;jj++)
                  {
                    
                    printf("%12.5e \t",starJacobian[nN][ii*nE_n + jj]);
                  }
                printf("\n");
              }
          }
      */

    }
}

/** @} */
