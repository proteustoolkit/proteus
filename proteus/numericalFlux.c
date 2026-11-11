#include "numericalFlux.h"

#define TR_ALPHA 0.5
#define TR_ALPHA_EXT 1.0


/***********************************************************************
 try some different numerical fluxes
 ***********************************************************************/

/**
   \brief Calculate the advective flux at at interior element boundaries for simple scalar nonlinear hyperbolic pdes
*/
void calculateInteriorNumericalAdvectiveFluxConvexOneSonicPoint(double sonicPoint,
								double sonicFlux,
								proteus_index_t nInteriorElementBoundaries_global,
								int nElementBoundaries_element,
								int nQuadraturePoints_elementBoundary,
								int nSpace,
								proteus_index_t* interiorElementBoundaries,
								proteus_index_t* elementBoundaryElements,
								int* elementBoundaryLocalElementBoundaries,
								double* n,
								double* u,
								double* f,
								double* df,
								double* flux,
								double* dflux_left,
								double* dflux_right)
{
  proteus_index_t ebN, left_eN_global, right_eN_global; int ebNI, left_ebN_element, right_ebN_element, k, J;
  double left_flux,right_flux,u_left,u_right,left_speed,right_speed,sonicSpeed;
  sonicSpeed = 0.0;
  for(ebNI=0;ebNI<nInteriorElementBoundaries_global;ebNI++)
    {
      ebN = interiorElementBoundaries[ebNI];
      left_eN_global = elementBoundaryElements[ebN*2+0];
      right_eN_global = elementBoundaryElements[ebN*2+1];
      left_ebN_element = elementBoundaryLocalElementBoundaries[ebN*2+0];
      right_ebN_element = elementBoundaryLocalElementBoundaries[ebN*2+1];
      for(k=0;k<nQuadraturePoints_elementBoundary;k++)
        {
	  left_speed =0.0;
	  right_speed=0.0;
          left_flux=0.0;
          right_flux=0.0;
          for(J=0;J<nSpace;J++)
            {
              left_speed 
                += 
                n[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                  left_ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                  k*nSpace+
		  J]
                *
                df[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                   left_ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                   k*nSpace+
		   J];
              right_speed 
                += 
                n[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                  left_ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                  k*nSpace+
		  J]
                *
                df[right_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                   right_ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                   k*nSpace+
		   J];
             left_flux 
                += 
                n[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                  left_ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                  k*nSpace+
		  J]
                *
                f[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                   left_ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                   k*nSpace+
		   J];
             right_flux 
               += 
               n[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                 left_ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                 k*nSpace+
                 J]
               *
               f[right_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                 right_ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                 k*nSpace+
                 J];
            }

	  u_left = u[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary+
		     left_ebN_element*nQuadraturePoints_elementBoundary+
		     k];
	  u_right= u[right_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary+
		     right_ebN_element*nQuadraturePoints_elementBoundary+
		     k];
	  /***************************************************
             Simple convex flux with one sonic point potentially
             Generic scalar riemann solution (1d too)
               f(u_riem) = max_{u_R <= u <= u_L} if u_L >= u_R
                         = min_{u_L <= u <= u_R} if u_L <  u_R
           **************************************************/ 
	  /*cases*/
	  if (u_left >= u_right)
	    {
	      if (left_flux >= right_flux)
		{
		  flux[ebN*nQuadraturePoints_elementBoundary+
		       k] = left_flux;
		  dflux_left[ebN*nQuadraturePoints_elementBoundary+
			     k] = left_speed;
		  dflux_right[ebN*nQuadraturePoints_elementBoundary+
			      k] = 0.0;
		}
	      else
		{
		  flux[ebN*nQuadraturePoints_elementBoundary+
		       k] = right_flux;
		  dflux_left[ebN*nQuadraturePoints_elementBoundary+
			     k] = 0.0;
		  dflux_right[ebN*nQuadraturePoints_elementBoundary+
			      k] = right_speed;
		}
	    }/*max*/
	  else
	    {
	      /*min*/
	      flux[ebN*nQuadraturePoints_elementBoundary+
		   k] = left_flux;
	      dflux_left[ebN*nQuadraturePoints_elementBoundary+
			 k] = left_speed;
	      dflux_right[ebN*nQuadraturePoints_elementBoundary+
			  k] = 0.0;

	      if (right_flux <= flux[ebN*nQuadraturePoints_elementBoundary+k])
		{
		  flux[ebN*nQuadraturePoints_elementBoundary+
		       k] = right_flux;
		  dflux_left[ebN*nQuadraturePoints_elementBoundary+
			     k] = 0.0;
		  dflux_right[ebN*nQuadraturePoints_elementBoundary+
			      k] = right_speed;
		}
	      if (u_left <= sonicPoint && sonicPoint <= u_right &&
		  sonicFlux < flux[ebN*nQuadraturePoints_elementBoundary+k])/*only consider if sonicPoint in interval*/
		{
		  flux[ebN*nQuadraturePoints_elementBoundary+
		       k] = sonicFlux;

		  dflux_left[ebN*nQuadraturePoints_elementBoundary+
			     k] = sonicSpeed;
		  dflux_right[ebN*nQuadraturePoints_elementBoundary+
			      k]= sonicSpeed;
		}
	      
	    }/*min not containing sonic point*/
	  /*mwf debug
	  if (fabs(u_left-u_right) > 1.0e-2)
	    {
	      printf("conv flux ebN=%d eN_left=%d eN_right=%d u_left=%g u_right=%g left_flux=%g right_flux=%g flux=%g\n\t",
		     ebN,left_eN_global,right_eN_global,u_left,u_right,left_flux,right_flux,flux[ebN*nQuadraturePoints_elementBoundary+k]);
	      for (J=0; J < nSpace; J++)
		{
		  printf("n[%d] = %g ",J,n[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
					   left_ebN_element*nQuadraturePoints_elementBoundary*nSpace+
					   k*nSpace+J]);
		}
	      printf("\n");
	      
	    }
	  */
        }/*k*/
    }/*ebnI*/
}
void calculateInteriorNumericalAdvectiveFluxRusanov(double safetyFactor,
						    proteus_index_t nInteriorElementBoundaries_global,
						    int nElementBoundaries_element,
						    int nQuadraturePoints_elementBoundary,
						    int nQuadraturePoints_element,
						    int nSpace,
						    proteus_index_t* interiorElementBoundaries,
						    proteus_index_t* elementBoundaryElements,
						    int* elementBoundaryLocalElementBoundaries,
						    double* n,
						    double* u,
						    double* f,
						    double* df,
						    double* df_element,
						    double* flux,
						    double* dflux_left,
						    double* dflux_right)
{
  proteus_index_t ebN, left_eN_global, right_eN_global; int ebNI, left_ebN_element, right_ebN_element, k, J;
  double left_flux,right_flux,u_left,u_right,left_speed,right_speed,
    maxSpeed_left,maxSpeed_right,maxSpeed_element,maxSpeed,tmp_left,tmp_right;
  /*for now use outer normal at first quadrature point for element speed calculations*/
  
  for(ebNI=0;ebNI<nInteriorElementBoundaries_global;ebNI++)
    {
      ebN = interiorElementBoundaries[ebNI];
      left_eN_global = elementBoundaryElements[ebN*2+0];
      right_eN_global = elementBoundaryElements[ebN*2+1];
      left_ebN_element = elementBoundaryLocalElementBoundaries[ebN*2+0];
      right_ebN_element = elementBoundaryLocalElementBoundaries[ebN*2+1];
      /*first calculate maximum value for df/du on neighboring elements and use this 
	in Rusanov (local lax-friedrichs) flux
	h(a,b) = 0.5*(f(a)+f(b)) - 0.5*alpha*(b-a)
      */
      maxSpeed_left =0.0; maxSpeed_right=0.0; maxSpeed=0.0;
      for (k=0; k < nQuadraturePoints_element; k++)
	{
	  tmp_left = 0.0; tmp_right=0.0;
	  /*evaluate df at interior elemnent point but compute its value dotted with normal for speed*/ 
	  for (J=0; J < nSpace; J++)
	    {
	      tmp_left += 
		df_element[left_eN_global*nQuadraturePoints_element*nSpace+ k*nSpace + J]
		*
		n[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                  left_ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                  0*nSpace+
		  J];
			 
	      tmp_right += 
		df_element[right_eN_global*nQuadraturePoints_element*nSpace+ k*nSpace + J]
		*
		n[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                  left_ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                  0*nSpace+
		  J];
	    }
	  maxSpeed_left = fabs(tmp_left)  > maxSpeed_left ? fabs(tmp_left) : maxSpeed_left;
	  maxSpeed_right= fabs(tmp_right) > maxSpeed_right ? fabs(tmp_right) : maxSpeed_right;	  
	}
      maxSpeed_element = maxSpeed_right > maxSpeed_left ? maxSpeed_right : maxSpeed_left;
      for(k=0;k<nQuadraturePoints_elementBoundary;k++)
        {
	  left_speed =0.0;
	  right_speed=0.0;
          left_flux=0.0;
          right_flux=0.0;
          for(J=0;J<nSpace;J++)
            {
              left_speed 
                += 
                n[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                  left_ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                  k*nSpace+
		  J]
                *
                df[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                   left_ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                   k*nSpace+
		   J];
              right_speed 
                += 
                n[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                  left_ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                  k*nSpace+
		  J]
                *
                df[right_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                   right_ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                   k*nSpace+
		   J];
             left_flux 
                += 
                n[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                  left_ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                  k*nSpace+
		  J]
                *
                f[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                   left_ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                   k*nSpace+
		   J];
             right_flux 
               += 
               n[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                 left_ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                 k*nSpace+
                 J]
               *
               f[right_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                 right_ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                 k*nSpace+
                 J];
            }

	  u_left = u[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary+
		     left_ebN_element*nQuadraturePoints_elementBoundary+
		     k];
	  u_right= u[right_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary+
		     right_ebN_element*nQuadraturePoints_elementBoundary+
		     k];
	  maxSpeed = fabs(left_speed)  > maxSpeed_element ? fabs(left_speed) : maxSpeed_element;
	  maxSpeed = fabs(right_speed) > maxSpeed ? fabs(right_speed) : maxSpeed;
	  maxSpeed*= safetyFactor;
	  flux[ebN*nQuadraturePoints_elementBoundary+k] = 0.5*(left_flux + right_flux) - 0.5*maxSpeed*(u_right-u_left);
	  dflux_left[ebN*nQuadraturePoints_elementBoundary+k] = 0.5*left_speed  + 0.5*maxSpeed;
	  dflux_right[ebN*nQuadraturePoints_elementBoundary+k]= 0.5*right_speed - 0.5*maxSpeed;
	  /*mwf debug
	  if (fabs(u_left-u_right) > 1.0e-2)
	    {
	      printf("Rusanov ebN=%d eN_left=%d eN_right=%d u_left=%g u_right=%g left_flux=%g right_flux=%g flux=%g\n\t",
		     ebN,left_eN_global,right_eN_global,u_left,u_right,left_flux,right_flux,flux[ebN*nQuadraturePoints_elementBoundary+k]);
	      for (J=0; J < nSpace; J++)
		{
		  printf("n[%d] = %g ",J,n[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
					   left_ebN_element*nQuadraturePoints_elementBoundary*nSpace+
					   k*nSpace+J]);
		}
	      printf("\n");
	      
	      printf("maxSpeed_element= %g maxSpeed_left=%g maxSpeed_right=%g left_speed=%g right_speed=%g maxSpeed=%g\n",
		     maxSpeed_element,maxSpeed_left,maxSpeed_right,left_speed,right_speed,maxSpeed);
	    }
 	  mwf end debug*/
        }/*k*/
    }/*ebnI*/
}
void calculateExteriorNumericalAdvectiveFluxRusanov(double safetyFactor,
						    proteus_index_t nExteriorElementBoundaries_global,
						    int nElementBoundaries_element,
						    int nQuadraturePoints_elementBoundary,
						    int nQuadraturePoints_element,
						    int nSpace,
						    proteus_index_t* exteriorElementBoundaries,
						    proteus_index_t* elementBoundaryElements,
						    int* elementBoundaryLocalElementBoundaries,
						    int* isDOFBoundary,
						    int* inflowFlag,
						    double* n,
						    double* bc_u,
						    double* bc_f,
						    double* bc_df,
						    double* u,
						    double* f,
						    double* df,
						    double* df_element,
						    double* flux,
						    double* dflux)
{
  proteus_index_t ebN, eN_global; int ebNE, ebN_element, k, J;
  double left_flux,right_flux,u_left,u_right,left_speed,right_speed,
    maxSpeed_element,maxSpeed,tmp_left;
  /*for now use outer normal at first quadrature point for element speed calculations*/
  
  for(ebNE=0;ebNE<nExteriorElementBoundaries_global;ebNE++)
    {
      ebN = exteriorElementBoundaries[ebNE];
      eN_global = elementBoundaryElements[ebN*2+0];
      ebN_element = elementBoundaryLocalElementBoundaries[ebN*2+0];
      /*first calculate maximum value for df/du on neighboring element and use this 
	in Rusanov (local lax-friedrichs) flux
	h(a,b) = 0.5*(f(a)+f(b)) - 0.5*alpha*(b-a)
      */
      maxSpeed_element =0.0; maxSpeed=0.0;
      for (k=0; k < nQuadraturePoints_element; k++)
	{
	  tmp_left = 0.0;
	  /*evaluate df at interior elemnent point but compute its value dotted with normal for speed*/ 
	  for (J=0; J < nSpace; J++)
	    {
	      tmp_left += 
		df_element[eN_global*nQuadraturePoints_element*nSpace+ k*nSpace + J]
		*
		n[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                  ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                  0*nSpace+
		  J];
	    }
	  maxSpeed_element = fabs(tmp_left)  > maxSpeed_element ? fabs(tmp_left) : maxSpeed_element;
	}
      for(k=0;k<nQuadraturePoints_elementBoundary;k++)
        {
	  left_speed =0.0;
	  right_speed=0.0;
          left_flux=0.0;
          right_flux=0.0;
          for(J=0;J<nSpace;J++)
            {
              left_speed 
                += 
                n[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                  ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                  k*nSpace+
		  J]
                *
                df[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                   ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                   k*nSpace+
		   J];
              right_speed 
                += 
                n[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                  ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                  k*nSpace+
		  J]
                *
                bc_df[ebNE*nQuadraturePoints_elementBoundary*nSpace+
		      k*nSpace+
		      J];
	      left_flux 
                += 
                n[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                  ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                  k*nSpace+
		  J]
                *
                f[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                  ebN_element*nQuadraturePoints_elementBoundary*nSpace+
		  k*nSpace+
		  J];
             right_flux 
               += 
               n[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                 ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                 k*nSpace+
                 J]
               *
               bc_f[ebNE*nQuadraturePoints_elementBoundary*nSpace+
		    k*nSpace+
		    J];
            }

	  u_left = u[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary+
		     ebN_element*nQuadraturePoints_elementBoundary+
		     k];
	  u_right= bc_u[ebNE*nQuadraturePoints_elementBoundary+
			k];
	  maxSpeed = fabs(left_speed)  > maxSpeed_element ? fabs(left_speed) : maxSpeed_element;
	  maxSpeed = fabs(right_speed) > maxSpeed ? fabs(right_speed) : maxSpeed;
	  maxSpeed*= safetyFactor;
	  flux[ebN*nQuadraturePoints_elementBoundary+k] = 0.5*(left_flux + right_flux) - 0.5*maxSpeed*(u_right-u_left);
	  dflux[ebN*nQuadraturePoints_elementBoundary+k] = 0.5*left_speed  + 0.5*maxSpeed;

	  /*mwf debug
	  if (fabs(u_left-u_right) > 1.0e-2)
	    {
	      printf("Rusanov exterior ebN=%d eN=%d u_left=%g u_right=%g left_flux=%g right_flux=%g flux=%g\n\t",
		     ebN,eN_global,u_left,u_right,left_flux,right_flux,flux[ebN*nQuadraturePoints_elementBoundary+k]);
	      for (J=0; J < nSpace; J++)
		{
		  printf("n[%d] = %g ",J,n[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
					   ebN_element*nQuadraturePoints_elementBoundary*nSpace+
					   k*nSpace+J]);
		}
	      printf("\n");
	      
	      printf("maxSpeed_element= %g left_speed=%g right_speed=%g maxSpeed=%g\n",
		     maxSpeed_element,left_speed,right_speed,maxSpeed);
		     }
 	  mwf end debug*/
        }/*k*/
    }/*ebnI*/
}

void calculateGlobalExteriorNumericalAdvectiveFluxRusanov(double safetyFactor,
							  proteus_index_t nExteriorElementBoundaries_global,
							  int nQuadraturePoints_elementBoundary,
							  int nQuadraturePoints_element,
							  int nSpace,
							  proteus_index_t* exteriorElementBoundaries,
							  proteus_index_t* elementBoundaryElements,
							  int* elementBoundaryLocalElementBoundaries,
							  int* isDOFBoundary,
							  int* inflowFlag,
							  double* n,
							  double* bc_u,
							  double* bc_f,
							  double* bc_df,
							  double* u,
							  double* f,
							  double* df,
							  double* df_element,
							  double* flux,
							  double* dflux)
{
  proteus_index_t ebN, eN_global; int ebNE, k, J;
  double left_flux,right_flux,u_left,u_right,left_speed,right_speed,
    maxSpeed_element,maxSpeed,tmp_left;
  /*for now use outer normal at first quadrature point for element speed calculations*/
  
  for(ebNE=0;ebNE<nExteriorElementBoundaries_global;ebNE++)
    {
      ebN = exteriorElementBoundaries[ebNE];
      eN_global = elementBoundaryElements[ebN*2+0];
      /*first calculate maximum value for df/du on neighboring element and use this 
	in Rusanov (local lax-friedrichs) flux
	h(a,b) = 0.5*(f(a)+f(b)) - 0.5*alpha*(b-a)
      */
      maxSpeed_element =0.0; maxSpeed=0.0;
      for (k=0; k < nQuadraturePoints_element; k++)
	{
	  tmp_left = 0.0;
	  /*evaluate df at interior elemnent point but compute its value dotted with normal for speed*/ 
	  for (J=0; J < nSpace; J++)
	    {
	      tmp_left += 
		df_element[eN_global*nQuadraturePoints_element*nSpace+ k*nSpace + J]
		*
		n[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                  0*nSpace+
		  J];
	    }
	  maxSpeed_element = fabs(tmp_left)  > maxSpeed_element ? fabs(tmp_left) : maxSpeed_element;
	}
      for(k=0;k<nQuadraturePoints_elementBoundary;k++)
        {
	  left_speed =0.0;
	  right_speed=0.0;
          left_flux=0.0;
          right_flux=0.0;
          for(J=0;J<nSpace;J++)
            {
              left_speed 
                += 
                n[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                  k*nSpace+
		  J]
                *
                df[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                   k*nSpace+
		   J];
              right_speed 
                += 
                n[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                  k*nSpace+
		  J]
                *
                bc_df[ebNE*nQuadraturePoints_elementBoundary*nSpace+
		      k*nSpace+
		      J];
	      left_flux 
                += 
                n[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                  k*nSpace+
		  J]
                *
                f[ebNE*nQuadraturePoints_elementBoundary*nSpace+
		  k*nSpace+
		  J];
             right_flux 
               += 
               n[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                 k*nSpace+
                 J]
               *
               bc_f[ebNE*nQuadraturePoints_elementBoundary*nSpace+
		    k*nSpace+
		    J];
            }

	  u_left = u[ebNE*nQuadraturePoints_elementBoundary+
		     k];
	  u_right= bc_u[ebNE*nQuadraturePoints_elementBoundary+
			k];
	  maxSpeed = fabs(left_speed)  > maxSpeed_element ? fabs(left_speed) : maxSpeed_element;
	  maxSpeed = fabs(right_speed) > maxSpeed ? fabs(right_speed) : maxSpeed;
	  maxSpeed*= safetyFactor;
	  flux[ebNE*nQuadraturePoints_elementBoundary+k] = 0.5*(left_flux + right_flux) - 0.5*maxSpeed*(u_right-u_left);
	  dflux[ebNE*nQuadraturePoints_elementBoundary+k] = 0.5*left_speed  + 0.5*maxSpeed;

	  /*mwf debug
	  if (fabs(u_left-u_right) > 1.0e-2)
	    {
	      printf("Rusanov exterior ebN=%d u_left=%g u_right=%g left_flux=%g right_flux=%g flux=%g\n\t",
		     ebNE,u_left,u_right,left_flux,right_flux,flux[ebNE*nQuadraturePoints_elementBoundary+k]);
	      for (J=0; J < nSpace; J++)
		{
		  printf("n[%d] = %g ",J,n[ebNE*nQuadraturePoints_elementBoundary*nSpace+
					   k*nSpace+J]);
		}
	      printf("\n");
	      
	      printf("maxSpeed_element= %g left_speed=%g right_speed=%g maxSpeed=%g\n",
		     maxSpeed_element,left_speed,right_speed,maxSpeed);
		     }
 	  mwf end debug*/
        }/*k*/
    }/*ebnE*/
}
/**
   need to add non-diagonal flux derivatives for Jacobian
 */
void calculateInteriorNumericalAdvectiveFluxRusanovWithEigenvalueBound(double safetyFactor,
								       proteus_index_t nInteriorElementBoundaries_global,
								       int nElementBoundaries_element,
								       int nQuadraturePoints_elementBoundary,
								       int nQuadraturePoints_element,
								       int nSpace,
								       proteus_index_t* interiorElementBoundaries,
								       proteus_index_t* elementBoundaryElements,
								       int* elementBoundaryLocalElementBoundaries,
								       double* n,
								       double* u,
								       double* f,
								       double* lambda_bar_element,
								       double* flux)
{
  proteus_index_t ebN, left_eN_global, right_eN_global; int ebNI, left_ebN_element, right_ebN_element, k, J;
  double left_flux,right_flux,u_left,u_right,
    maxSpeed_left,maxSpeed_right,maxSpeed_element,maxSpeed,tmp_left,tmp_right;
  /*for now use outer normal at first quadrature point for element speed calculations*/
  
  for(ebNI=0;ebNI<nInteriorElementBoundaries_global;ebNI++)
    {
      ebN = interiorElementBoundaries[ebNI];
      left_eN_global = elementBoundaryElements[ebN*2+0];
      right_eN_global = elementBoundaryElements[ebN*2+1];
      left_ebN_element = elementBoundaryLocalElementBoundaries[ebN*2+0];
      right_ebN_element = elementBoundaryLocalElementBoundaries[ebN*2+1];
      /*first calculate maximum value for df/du on neighboring elements and use this 
	in Rusanov (local lax-friedrichs) flux
	h(a,b) = 0.5*(f(a)+f(b)) - 0.5*alpha*(b-a)
      */
      maxSpeed_left =0.0; maxSpeed_right=0.0; maxSpeed=0.0;
      for (k=0; k < nQuadraturePoints_element; k++)
	{
	  tmp_left = lambda_bar_element[left_eN_global*nQuadraturePoints_element + k]; 
	  tmp_right= lambda_bar_element[right_eN_global*nQuadraturePoints_element + k]; 
	  maxSpeed_left = fabs(tmp_left)  > maxSpeed_left ? fabs(tmp_left) : maxSpeed_left;
	  maxSpeed_right= fabs(tmp_right) > maxSpeed_right ? fabs(tmp_right) : maxSpeed_right;	  
	}
      maxSpeed_element = maxSpeed_right > maxSpeed_left ? maxSpeed_right : maxSpeed_left;
      for(k=0;k<nQuadraturePoints_elementBoundary;k++)
        {
          left_flux=0.0;
          right_flux=0.0;
          for(J=0;J<nSpace;J++)
            {
             left_flux 
                += 
                n[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                  left_ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                  k*nSpace+
		  J]
                *
                f[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                   left_ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                   k*nSpace+
		   J];
             right_flux 
               += 
               n[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                 left_ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                 k*nSpace+
                 J]
               *
               f[right_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                 right_ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                 k*nSpace+
                 J];
            }

	  u_left = u[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary+
		     left_ebN_element*nQuadraturePoints_elementBoundary+
		     k];
	  u_right= u[right_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary+
		     right_ebN_element*nQuadraturePoints_elementBoundary+
		     k];
/* 	  maxSpeed = fabs(left_speed)  > maxSpeed_element ? fabs(left_speed) : maxSpeed_element; */
/* 	  maxSpeed = fabs(right_speed) > maxSpeed ? fabs(right_speed) : maxSpeed; */
	  maxSpeed = maxSpeed_element;
	  maxSpeed*= safetyFactor;
	  flux[ebN*nQuadraturePoints_elementBoundary+k] = 0.5*(left_flux + right_flux) - 0.5*maxSpeed*(u_right-u_left);
	  /*mwf debug
	  if (fabs(u_left-u_right) > 1.0e-2)
	    {
	      printf("Rusanov ebN=%d eN_left=%d eN_right=%d u_left=%g u_right=%g left_flux=%g right_flux=%g flux=%g\n\t",
		     ebN,left_eN_global,right_eN_global,u_left,u_right,left_flux,right_flux,flux[ebN*nQuadraturePoints_elementBoundary+k]);
	      for (J=0; J < nSpace; J++)
		{
		  printf("n[%d] = %g ",J,n[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
					   left_ebN_element*nQuadraturePoints_elementBoundary*nSpace+
					   k*nSpace+J]);
		}
	      printf("\n");
	      
	      printf("maxSpeed_element= %g maxSpeed_left=%g maxSpeed_right=%g left_speed=%g right_speed=%g maxSpeed=%g\n",
		     maxSpeed_element,maxSpeed_left,maxSpeed_right,left_speed,right_speed,maxSpeed);
	    }
 	  mwf end debug*/
        }/*k*/
    }/*ebnI*/
}
void calculateGlobalExteriorNumericalAdvectiveFluxRusanovWithEigenvalueBound(double safetyFactor,
									     proteus_index_t nExteriorElementBoundaries_global,
									     int nQuadraturePoints_elementBoundary,
									     int nQuadraturePoints_element,
									     int nSpace,
									     proteus_index_t* exteriorElementBoundaries,
									     proteus_index_t* elementBoundaryElements,
									     int* elementBoundaryLocalElementBoundaries,
									     int* isDOFBoundary,
									     int* inflowFlag,
									     double* n,
									     double* bc_u,
									     double* bc_f,
									     double* u,
									     double* f,
									     double* lambda_bar,
									     double* flux)
{
  proteus_index_t ebN, eN_global; int ebNE, k, J;
  double left_flux,right_flux,u_left,u_right,
    maxSpeed_element,maxSpeed,tmp_left;
  /*for now use outer normal at first quadrature point for element speed calculations*/
  
  for(ebNE=0;ebNE<nExteriorElementBoundaries_global;ebNE++)
    {
      ebN = exteriorElementBoundaries[ebNE];
      eN_global = elementBoundaryElements[ebN*2+0];
      /*first calculate maximum value for df/du on neighboring element and use this 
	in Rusanov (local lax-friedrichs) flux
	h(a,b) = 0.5*(f(a)+f(b)) - 0.5*alpha*(b-a)
      */
      maxSpeed_element =0.0; maxSpeed=0.0;
      for (k=0; k < nQuadraturePoints_element; k++)
	{
	  tmp_left = lambda_bar[eN_global*nQuadraturePoints_element + k];
	  maxSpeed_element = fabs(tmp_left)  > maxSpeed_element ? fabs(tmp_left) : maxSpeed_element;
	}
      for(k=0;k<nQuadraturePoints_elementBoundary;k++)
        {
          left_flux=0.0;
          right_flux=0.0;
          for(J=0;J<nSpace;J++)
            {
	      left_flux 
                += 
                n[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                  k*nSpace+
		  J]
                *
                f[ebNE*nQuadraturePoints_elementBoundary*nSpace+
		  k*nSpace+
		  J];
             right_flux 
               += 
               n[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                 k*nSpace+
                 J]
               *
               bc_f[ebNE*nQuadraturePoints_elementBoundary*nSpace+
		    k*nSpace+
		    J];
            }

	  u_left = u[ebNE*nQuadraturePoints_elementBoundary+
		     k];
	  u_right= bc_u[ebNE*nQuadraturePoints_elementBoundary+
			k];
/* 	  maxSpeed = fabs(left_speed)  > maxSpeed_element ? fabs(left_speed) : maxSpeed_element; */
/* 	  maxSpeed = fabs(right_speed) > maxSpeed ? fabs(right_speed) : maxSpeed; */
	  maxSpeed =maxSpeed_element;
	  maxSpeed*= safetyFactor;
	  flux[ebNE*nQuadraturePoints_elementBoundary+k] = 0.5*(left_flux + right_flux) - 0.5*maxSpeed*(u_right-u_left);

	  /*mwf debug
	  if (fabs(u_left-u_right) > 1.0e-2)
	    {
	      printf("Rusanov exterior ebN=%d u_left=%g u_right=%g left_flux=%g right_flux=%g flux=%g\n\t",
		     ebNE,u_left,u_right,left_flux,right_flux,flux[ebNE*nQuadraturePoints_elementBoundary+k]);
	      for (J=0; J < nSpace; J++)
		{
		  printf("n[%d] = %g ",J,n[ebNE*nQuadraturePoints_elementBoundary*nSpace+
					   k*nSpace+J]);
		}
	      printf("\n");
	      
	      printf("maxSpeed_element= %g left_speed=%g right_speed=%g maxSpeed=%g\n",
		     maxSpeed_element,left_speed,right_speed,maxSpeed);
		     }
 	  mwf end debug*/
        }/*k*/
    }/*ebnE*/
}
void calculateExteriorNumericalAdvectiveFluxRusanovWithEigenvalueBound(double safetyFactor,
								       proteus_index_t nExteriorElementBoundaries_global,
								       int nElementBoundaries_element,
								       int nQuadraturePoints_elementBoundary,
								       int nQuadraturePoints_element,
								       int nSpace,
								       proteus_index_t* exteriorElementBoundaries,
								       proteus_index_t* elementBoundaryElements,
								       int* elementBoundaryLocalElementBoundaries,
								       int* isDOFBoundary,
								       int* inflowFlag,
								       double* n,
								       double* bc_u,
								       double* bc_f,
								       double* u,
								       double* f,
								       double* lambda_bar,
								       double* flux)
{
  proteus_index_t ebN, eN_global; int ebNE, ebN_element, k, J;
  double left_flux,right_flux,u_left,u_right,
    maxSpeed_element,maxSpeed,tmp_left;
  /*for now use outer normal at first quadrature point for element speed calculations*/
  
  for(ebNE=0;ebNE<nExteriorElementBoundaries_global;ebNE++)
    {
      ebN = exteriorElementBoundaries[ebNE];
      eN_global = elementBoundaryElements[ebN*2+0];
      ebN_element = elementBoundaryLocalElementBoundaries[ebN*2+0];
      /*first calculate maximum value for df/du on neighboring element and use this 
	in Rusanov (local lax-friedrichs) flux
	h(a,b) = 0.5*(f(a)+f(b)) - 0.5*alpha*(b-a)
      */
      maxSpeed_element =0.0; maxSpeed=0.0;
      for (k=0; k < nQuadraturePoints_element; k++)
	{
	  tmp_left = lambda_bar[eN_global*nQuadraturePoints_element+k];
	  maxSpeed_element = fabs(tmp_left)  > maxSpeed_element ? fabs(tmp_left) : maxSpeed_element;
	}
      for(k=0;k<nQuadraturePoints_elementBoundary;k++)
        {
          left_flux=0.0;
          right_flux=0.0;
          for(J=0;J<nSpace;J++)
            {
	      left_flux 
                += 
                n[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                  ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                  k*nSpace+
		  J]
                *
                f[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                  ebN_element*nQuadraturePoints_elementBoundary*nSpace+
		  k*nSpace+
		  J];
             right_flux 
               += 
               n[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                 ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                 k*nSpace+
                 J]
               *
               bc_f[ebNE*nQuadraturePoints_elementBoundary*nSpace+
		    k*nSpace+
		    J];
            }

	  u_left = u[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary+
		     ebN_element*nQuadraturePoints_elementBoundary+
		     k];
	  u_right= bc_u[ebNE*nQuadraturePoints_elementBoundary+
			k];
/* 	  maxSpeed = fabs(left_speed)  > maxSpeed_element ? fabs(left_speed) : maxSpeed_element; */
/* 	  maxSpeed = fabs(right_speed) > maxSpeed ? fabs(right_speed) : maxSpeed; */
	  maxSpeed = maxSpeed_element;
	  maxSpeed*= safetyFactor;
	  flux[ebN*nQuadraturePoints_elementBoundary+k] = 0.5*(left_flux + right_flux) - 0.5*maxSpeed*(u_right-u_left);

	  /*mwf debug
	  if (fabs(u_left-u_right) > 1.0e-2)
	    {
	      printf("Rusanov exterior ebN=%d eN=%d u_left=%g u_right=%g left_flux=%g right_flux=%g flux=%g\n\t",
		     ebN,eN_global,u_left,u_right,left_flux,right_flux,flux[ebN*nQuadraturePoints_elementBoundary+k]);
	      for (J=0; J < nSpace; J++)
		{
		  printf("n[%d] = %g ",J,n[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
					   ebN_element*nQuadraturePoints_elementBoundary*nSpace+
					   k*nSpace+J]);
		}
	      printf("\n");
	      
	      printf("maxSpeed_element= %g left_speed=%g right_speed=%g maxSpeed=%g\n",
		     maxSpeed_element,left_speed,right_speed,maxSpeed);
		     }
 	  mwf end debug*/
        }/*k*/
    }/*ebnI*/
}

/************************************************************************
  begin moving over exterior numerical flux routines and changing to index
  boundary quadrature arrays as nExternalElementBoundaries * .
 ************************************************************************/
/**
   \brief Calculate the diffusive flux at interior element boundary quadrature points
*/ 
void calculateInteriorNumericalDiffusiveFlux(int scale_penalty,
                                             double penalty_floor,
                                             proteus_index_t nInteriorElementBoundaries_global,
                                             int nElementBoundaries_element,
                                             int nQuadraturePoints_elementBoundary,
                                             int nSpace,
                                             proteus_index_t* interiorElementBoundaries,
                                             proteus_index_t* elementBoundaryElements,
                                             int* elementBoundaryLocalElementBoundaries,
                                             double* n,
                                             double* a,
                                             double* grad_phi,
                                             double* u,
                                             double* penalty,
                                             double* flux)
{
  proteus_index_t ebN, left_eN_global, right_eN_global; int ebNI, left_ebN_element, right_ebN_element, k, J, I, nSpace2=nSpace*nSpace;
  double diffusiveVelocityComponent_I,max_a=0.0;
  for(ebNI=0;ebNI<nInteriorElementBoundaries_global;ebNI++)
    {
      ebN = interiorElementBoundaries[ebNI];
      left_eN_global = elementBoundaryElements[ebN*2+0];
      right_eN_global = elementBoundaryElements[ebN*2+1];
      left_ebN_element = elementBoundaryLocalElementBoundaries[ebN*2+0];
      right_ebN_element = elementBoundaryLocalElementBoundaries[ebN*2+1];
      for(k=0;k<nQuadraturePoints_elementBoundary;k++)
        {
          max_a = 0.0;
          flux[ebN*nQuadraturePoints_elementBoundary+k] = 0.0;
          for(I=0;I<nSpace;I++)
            {
              diffusiveVelocityComponent_I=0.0;
              for(J=0;J<nSpace;J++)
                {
                  diffusiveVelocityComponent_I 
                    -= 
                    (a[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace2+
                       left_ebN_element*nQuadraturePoints_elementBoundary*nSpace2+
                       k*nSpace2+
                       I*nSpace+
                       J]
                     *
                     grad_phi[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                              left_ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                              k*nSpace+J]
                     +
                     a[right_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace2+
                       right_ebN_element*nQuadraturePoints_elementBoundary*nSpace2+
                       k*nSpace2+
                       I*nSpace+
                       J]
                     *
                     grad_phi[right_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                              right_ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                              k*nSpace+J]);
                  max_a = fmax(max_a,a[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace2+
                                       left_ebN_element*nQuadraturePoints_elementBoundary*nSpace2+
                                       k*nSpace2+
                                       I*nSpace+
                                       J]);
                  max_a = fmax(max_a,a[right_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace2+
                                       right_ebN_element*nQuadraturePoints_elementBoundary*nSpace2+
                                       k*nSpace2+
                                       I*nSpace+
                                       J]);
                }
              flux[ebN*nQuadraturePoints_elementBoundary+
                   k] 
                += 
                diffusiveVelocityComponent_I
                *
                n[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                  left_ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                  k*nSpace+
                  I];
            }
          flux[ebN*nQuadraturePoints_elementBoundary+
               k] *= 0.5;
          /** \todo make penalty in DG on phi instead of u */
          max_a = fmax(max_a,penalty_floor);
          double penalty_flux = penalty[ebN*nQuadraturePoints_elementBoundary+
                                        k]
            *
            (u[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary+
               left_ebN_element*nQuadraturePoints_elementBoundary+
               k]-
             u[right_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary+
               right_ebN_element*nQuadraturePoints_elementBoundary+
               k]);
          if (scale_penalty) penalty_flux *= max_a;
          flux[ebN*nQuadraturePoints_elementBoundary+
               k] 
            += penalty_flux;
        }
    }
}
void calculateInteriorNumericalDiffusiveFlux_sd(int scale_penalty,
                                                double penalty_floor,
                                                proteus_index_t nInteriorElementBoundaries_global,
						int nElementBoundaries_element,
						int nQuadraturePoints_elementBoundary,
						int nSpace,
						proteus_csr_index_t* rowptr,
						proteus_csr_index_t* colind,
						proteus_index_t* interiorElementBoundaries,
						proteus_index_t* elementBoundaryElements,
						int* elementBoundaryLocalElementBoundaries,
						double* n,
						double* a,
						double* grad_phi,
						double* u,
						double* penalty,
						double* flux)
{
  proteus_index_t ebN, left_eN_global, right_eN_global; proteus_csr_index_t m; proteus_csr_index_t nnz=rowptr[nSpace]; int ebNI, left_ebN_element, right_ebN_element, k, I;
  double diffusiveVelocityComponent_I,max_a=0.0;
  for(ebNI=0;ebNI<nInteriorElementBoundaries_global;ebNI++)
    {
      ebN = interiorElementBoundaries[ebNI];
      left_eN_global = elementBoundaryElements[ebN*2+0];
      right_eN_global = elementBoundaryElements[ebN*2+1];
      left_ebN_element = elementBoundaryLocalElementBoundaries[ebN*2+0];
      right_ebN_element = elementBoundaryLocalElementBoundaries[ebN*2+1];
      for(k=0;k<nQuadraturePoints_elementBoundary;k++)
        {
          max_a = 0.0;
          flux[ebN*nQuadraturePoints_elementBoundary+k] = 0.0;
          for(I=0;I<nSpace;I++)
            {
              diffusiveVelocityComponent_I=0.0;
              for(m=rowptr[I];m<rowptr[I+1];m++)
                {
                  diffusiveVelocityComponent_I 
                    -= 
                    (a[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nnz+
                       left_ebN_element*nQuadraturePoints_elementBoundary*nnz+
                       k*nnz+
                       m]
                     *
                     grad_phi[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                              left_ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                              k*nSpace+colind[m]]
                     +
                     a[right_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nnz+
                       right_ebN_element*nQuadraturePoints_elementBoundary*nnz+
                       k*nnz+
                       m]
                     *
                     grad_phi[right_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                              right_ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                              k*nSpace+colind[m]]);
                  max_a = fmax(max_a,a[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nnz+
                                       left_ebN_element*nQuadraturePoints_elementBoundary*nnz+
                                       k*nnz+
                                       m]);
                  max_a = fmax(max_a,a[right_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nnz+
                                       right_ebN_element*nQuadraturePoints_elementBoundary*nnz+
                                       k*nnz+
                                       m]);
                }
              flux[ebN*nQuadraturePoints_elementBoundary+
                   k] 
                += 
                diffusiveVelocityComponent_I
                *
                n[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                  left_ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                  k*nSpace+
                  I];
            }
          flux[ebN*nQuadraturePoints_elementBoundary+
               k] *= 0.5;
          /** \todo make penalty in DG on phi instead of u */
          max_a = fmax(max_a,penalty_floor);
          double penalty_flux = penalty[ebN*nQuadraturePoints_elementBoundary+
                                        k]
            *
            (u[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary+
               left_ebN_element*nQuadraturePoints_elementBoundary+
               k]-
             u[right_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary+
               right_ebN_element*nQuadraturePoints_elementBoundary+
               k]);
          if (scale_penalty) penalty_flux *= max_a;
          /*mwf debug
          printf("calcIntNumDiffFlux_sd scale= %d k= %d max_a= %g penalty= %g \n",scale_penalty,k,max_a,penalty_flux);
          */
          flux[ebN*nQuadraturePoints_elementBoundary+
               k] 
            += penalty_flux;
        }
    }
}

/*
   \brief Calculate the diffusive flux at interior element boundary quadrature points
   
void calculateInteriorNumericalDiffusiveFlux(proteus_index_t nInteriorElementBoundaries_global,
                                             int nElementBoundaries_element,
                                             int nQuadraturePoints_elementBoundary,
                                             int nSpace,
                                             proteus_index_t* interiorElementBoundaries,
                                             proteus_index_t* elementBoundaryElements,
                                             int* elementBoundaryLocalElementBoundaries,
                                             double* n,
                                             double* a,
                                             double* grad_phi,
                                             double* u,
                                             double* penalty,
                                             double* flux)
{
  proteus_index_t ebN, left_eN_global, right_eN_global; int ebNI, left_ebN_element, right_ebN_element, k, J, I, nSpace2=nSpace*nSpace;
  double diffusiveVelocityComponent_I;
  for(ebNI=0;ebNI<nInteriorElementBoundaries_global;ebNI++)
    {
      ebN = interiorElementBoundaries[ebNI];
      left_eN_global = elementBoundaryElements[ebN*2+0];
      right_eN_global = elementBoundaryElements[ebN*2+1];
      left_ebN_element = elementBoundaryLocalElementBoundaries[ebN*2+0];
      right_ebN_element = elementBoundaryLocalElementBoundaries[ebN*2+1];
      for(k=0;k<nQuadraturePoints_elementBoundary;k++)
        {
          flux[ebN*nQuadraturePoints_elementBoundary+k] = 0.0;
          for(I=0;I<nSpace;I++)
            {
              diffusiveVelocityComponent_I=0.0;
              for(J=0;J<nSpace;J++)
                {
                  diffusiveVelocityComponent_I 
                    -= 
                    (a[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace2+
                       left_ebN_element*nQuadraturePoints_elementBoundary*nSpace2+
                       k*nSpace2+
                       I*nSpace+
                       J]
                     *
                     grad_phi[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                              left_ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                              k*nSpace+J]
                     +
                     a[right_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace2+
                       right_ebN_element*nQuadraturePoints_elementBoundary*nSpace2+
                       k*nSpace2+
                       I*nSpace+
                       J]
                     *
                     grad_phi[right_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                              right_ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                              k*nSpace+J]);
                }
              flux[ebN*nQuadraturePoints_elementBoundary+
                   k] 
                += 
                diffusiveVelocityComponent_I
                *
                n[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                  left_ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                  k*nSpace+
                  I];
            }
          flux[ebN*nQuadraturePoints_elementBoundary+
               k] *= 0.5;
	       //  \todo make penalty in DG on phi instead of u 
          flux[ebN*nQuadraturePoints_elementBoundary+
               k] 
            += 
            penalty[ebN*nQuadraturePoints_elementBoundary+
                    k]
            *
            (u[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary+
               left_ebN_element*nQuadraturePoints_elementBoundary+
               k]-
             u[right_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary+
               right_ebN_element*nQuadraturePoints_elementBoundary+
               k]);
        }
    }
}
void calculateInteriorNumericalDiffusiveFlux_sd(proteus_index_t nInteriorElementBoundaries_global,
						int nElementBoundaries_element,
						int nQuadraturePoints_elementBoundary,
						int nSpace,
						proteus_csr_index_t* rowptr,
						proteus_csr_index_t* colind,
						proteus_index_t* interiorElementBoundaries,
						proteus_index_t* elementBoundaryElements,
						int* elementBoundaryLocalElementBoundaries,
						double* n,
						double* a,
						double* grad_phi,
						double* u,
						double* penalty,
						double* flux)
{
  proteus_index_t ebN, left_eN_global, right_eN_global, m; proteus_csr_index_t nnz=rowptr[nSpace]; int ebNI, left_ebN_element, right_ebN_element, k, I;
  double diffusiveVelocityComponent_I;
  for(ebNI=0;ebNI<nInteriorElementBoundaries_global;ebNI++)
    {
      ebN = interiorElementBoundaries[ebNI];
      left_eN_global = elementBoundaryElements[ebN*2+0];
      right_eN_global = elementBoundaryElements[ebN*2+1];
      left_ebN_element = elementBoundaryLocalElementBoundaries[ebN*2+0];
      right_ebN_element = elementBoundaryLocalElementBoundaries[ebN*2+1];
      for(k=0;k<nQuadraturePoints_elementBoundary;k++)
        {
          flux[ebN*nQuadraturePoints_elementBoundary+k] = 0.0;
          for(I=0;I<nSpace;I++)
            {
              diffusiveVelocityComponent_I=0.0;
              for(m=rowptr[I];m<rowptr[I+1];m++)
                {
                  diffusiveVelocityComponent_I 
                    -= 
                    (a[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nnz+
                       left_ebN_element*nQuadraturePoints_elementBoundary*nnz+
                       k*nnz+
                       m]
                     *
                     grad_phi[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                              left_ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                              k*nSpace+colind[m]]
                     +
                     a[right_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nnz+
                       right_ebN_element*nQuadraturePoints_elementBoundary*nnz+
                       k*nnz+
                       m]
                     *
                     grad_phi[right_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                              right_ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                              k*nSpace+colind[m]]);
                }
              flux[ebN*nQuadraturePoints_elementBoundary+
                   k] 
                += 
                diffusiveVelocityComponent_I
                *
                n[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                  left_ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                  k*nSpace+
                  I];
            }
          flux[ebN*nQuadraturePoints_elementBoundary+
               k] *= 0.5;
          //  \todo make penalty in DG on phi instead of u 
          flux[ebN*nQuadraturePoints_elementBoundary+
               k] 
            += 
            penalty[ebN*nQuadraturePoints_elementBoundary+
                    k]
            *
            (u[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary+
               left_ebN_element*nQuadraturePoints_elementBoundary+
               k]-
             u[right_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary+
               right_ebN_element*nQuadraturePoints_elementBoundary+
               k]);
        }
    }
}
*/

/**
   \brief Calculate the diffusive flux Jacobian at interior element boundary quadrature points
*/ 
void updateInteriorNumericalDiffusiveFluxJacobian(int scale_penalty,
                                                  double penalty_floor,
                                                  proteus_index_t nInteriorElementBoundaries_global,
                                                  int nElementBoundaries_element,
                                                  int nQuadraturePoints_elementBoundary,
                                                  int nDOF_trial_element,
                                                  int nSpace,
                                                  proteus_index_t* l2g,
                                                  proteus_index_t* interiorElementBoundaries,
                                                  proteus_index_t* elementBoundaryElements,
                                                  int* elementBoundaryLocalElementBoundaries,
                                                  double* n,
                                                  double* a,
                                                  double* da,
                                                  double* grad_phi,
                                                  double* dphi,
                                                  double* v,
                                                  double* grad_v,
                                                  double* penalty,
                                                  double* fluxJacobian)
{
  proteus_index_t ebN, left_eN_global, right_eN_global, left_j_global, right_j_global; int ebNI, left_ebN_element, right_ebN_element, k, j, I, J, nSpace2=nSpace*nSpace;
  double leftJacobian,rightJacobian,diffusiveVelocityComponent_I_leftJacobian,diffusiveVelocityComponent_I_rightJacobian,diffusiveVelocityComponent_I_leftJacobian2,diffusiveVelocityComponent_I_rightJacobian2,max_a=0.0;
  for(ebNI=0;ebNI<nInteriorElementBoundaries_global;ebNI++)
    {
      ebN = interiorElementBoundaries[ebNI];
      left_eN_global = elementBoundaryElements[ebN*2+0];
      right_eN_global = elementBoundaryElements[ebN*2+1];
      left_ebN_element = elementBoundaryLocalElementBoundaries[ebN*2+0];
      right_ebN_element = elementBoundaryLocalElementBoundaries[ebN*2+1];
      for(k=0;k<nQuadraturePoints_elementBoundary;k++)
        {
          max_a = 0.0;
          for(j=0;j<nDOF_trial_element;j++)
            {
              leftJacobian=0.0;
              rightJacobian=0.0;
              left_j_global = l2g[left_eN_global*nDOF_trial_element+j];
              right_j_global= l2g[right_eN_global*nDOF_trial_element+j];
              for(I=0;I<nSpace;I++)
                {
                  diffusiveVelocityComponent_I_leftJacobian=0.0;
                  diffusiveVelocityComponent_I_leftJacobian2=0.0;
                  diffusiveVelocityComponent_I_rightJacobian=0.0;
                  diffusiveVelocityComponent_I_rightJacobian2=0.0;
                  for(J=0;J<nSpace;J++)
                    {
                      diffusiveVelocityComponent_I_leftJacobian 
                        -= 
                        da[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace2+
                           left_ebN_element*nQuadraturePoints_elementBoundary*nSpace2+
                           k*nSpace2+
                           I*nSpace+
                           J]
                        *
                        grad_phi[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                                 left_ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                                 k*nSpace+
                                 J];
                      diffusiveVelocityComponent_I_rightJacobian 
                        -= 
                        da[right_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace2+
                           right_ebN_element*nQuadraturePoints_elementBoundary*nSpace2+
                           k*nSpace2+
                           I*nSpace+
                           J]
                        *
                        grad_phi[right_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                                 right_ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                                 k*nSpace+
                                 J];
                      diffusiveVelocityComponent_I_leftJacobian2 
                        -= 
                        a[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace2+
                          left_ebN_element*nQuadraturePoints_elementBoundary*nSpace2+
                          k*nSpace2+
                          I*nSpace+
                          J]
                        *
                        grad_v[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_trial_element*nSpace+
                               left_ebN_element*nQuadraturePoints_elementBoundary*nDOF_trial_element*nSpace+
                               k*nDOF_trial_element*nSpace+
                               j*nSpace+
                               J];
                      diffusiveVelocityComponent_I_rightJacobian2 
                        -= 
                        a[right_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace2+
                          right_ebN_element*nQuadraturePoints_elementBoundary*nSpace2+
                          k*nSpace2+
                          I*nSpace+
                          J]
                        *
                        grad_v[right_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_trial_element*nSpace+
                               right_ebN_element*nQuadraturePoints_elementBoundary*nDOF_trial_element*nSpace+
                               k*nDOF_trial_element*nSpace+
                               j*nSpace+
                               J];
                      max_a = fmax(max_a,a[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace2+
                                           left_ebN_element*nQuadraturePoints_elementBoundary*nSpace2+
                                           k*nSpace2+
                                           I*nSpace+
                                           J]);
                      max_a = fmax(max_a,a[right_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace2+
                                           right_ebN_element*nQuadraturePoints_elementBoundary*nSpace2+
                                           k*nSpace2+
                                           I*nSpace+
                                           J]);
                      
                    }
                  leftJacobian 
                    += 
                    (diffusiveVelocityComponent_I_leftJacobian
                     *
                     v[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                       left_ebN_element*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                       k*nDOF_trial_element+
                       j] 
                     +
                     diffusiveVelocityComponent_I_leftJacobian2*
                     dphi[left_j_global])
                    *
                    n[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                      left_ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                      k*nSpace+
                      I];
                  rightJacobian 
                    += 
                    (diffusiveVelocityComponent_I_rightJacobian
                     *
                     v[right_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                       right_ebN_element*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                       k*nDOF_trial_element+
                       j] 
                     +
                     diffusiveVelocityComponent_I_rightJacobian2*dphi[right_j_global])
                    *
                    n[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                      left_ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                      k*nSpace+
                      I];
                }
              leftJacobian *= 0.5;
              rightJacobian *= 0.5;
              max_a = fmax(max_a,penalty_floor);
              double penaltyJacobian_term = penalty[ebN*nQuadraturePoints_elementBoundary+
                                                    k];
              if (scale_penalty) penaltyJacobian_term *= max_a;
              leftJacobian 
                += 
                penaltyJacobian_term
                *
                v[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                  left_ebN_element*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                  k*nDOF_trial_element+
                  j];
              rightJacobian 
                -= 
                penaltyJacobian_term
                *
                v[right_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                  right_ebN_element*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                  k*nDOF_trial_element+
                  j];
              fluxJacobian[ebN*2*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                           0*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                           k*nDOF_trial_element+
                           j] += leftJacobian;
              fluxJacobian[ebN*2*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                           1*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                           k*nDOF_trial_element+
                           j] += rightJacobian;
            }
        }
    }
}

void updateInteriorNumericalDiffusiveFluxJacobian_sd(int scale_penalty,
                                                     double penalty_floor,
                                                     proteus_index_t nInteriorElementBoundaries_global,
						     int nElementBoundaries_element,
						     int nQuadraturePoints_elementBoundary,
						     int nDOF_trial_element,
						     int nSpace,
						     proteus_csr_index_t* rowptr,
						     proteus_csr_index_t* colind,
						     proteus_index_t* l2g,
						     proteus_index_t* interiorElementBoundaries,
						     proteus_index_t* elementBoundaryElements,
						     int* elementBoundaryLocalElementBoundaries,
						     double* n,
						     double* a,
						     double* da,
						     double* grad_phi,
						     double* dphi,
						     double* v,
						     double* grad_v,
						     double* penalty,
						     double* fluxJacobian)
{
  proteus_index_t ebN, left_eN_global, right_eN_global, left_j_global, right_j_global; proteus_csr_index_t m; proteus_csr_index_t nnz=rowptr[nSpace]; int ebNI, left_ebN_element, right_ebN_element, k, j, I;
  double leftJacobian,rightJacobian,diffusiveVelocityComponent_I_leftJacobian,diffusiveVelocityComponent_I_rightJacobian,diffusiveVelocityComponent_I_leftJacobian2,diffusiveVelocityComponent_I_rightJacobian2,max_a;
  for(ebNI=0;ebNI<nInteriorElementBoundaries_global;ebNI++)
    {
      ebN = interiorElementBoundaries[ebNI];
      left_eN_global = elementBoundaryElements[ebN*2+0];
      right_eN_global = elementBoundaryElements[ebN*2+1];
      left_ebN_element = elementBoundaryLocalElementBoundaries[ebN*2+0];
      right_ebN_element = elementBoundaryLocalElementBoundaries[ebN*2+1];
      for(k=0;k<nQuadraturePoints_elementBoundary;k++)
        {
          max_a = 0.0;
          for(j=0;j<nDOF_trial_element;j++)
            {
              leftJacobian=0.0;
              rightJacobian=0.0;
              left_j_global = l2g[left_eN_global*nDOF_trial_element+j];
              right_j_global= l2g[right_eN_global*nDOF_trial_element+j];
              for(I=0;I<nSpace;I++)
                {
                  diffusiveVelocityComponent_I_leftJacobian=0.0;
                  diffusiveVelocityComponent_I_leftJacobian2=0.0;
                  diffusiveVelocityComponent_I_rightJacobian=0.0;
                  diffusiveVelocityComponent_I_rightJacobian2=0.0;
                  for(m=rowptr[I];m<rowptr[I+1];m++)
                    {
                      diffusiveVelocityComponent_I_leftJacobian 
                        -= 
                        da[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nnz+
                           left_ebN_element*nQuadraturePoints_elementBoundary*nnz+
                           k*nnz+
                           m]
                        *
                        grad_phi[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                                 left_ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                                 k*nSpace+
                                 colind[m]];
                      diffusiveVelocityComponent_I_rightJacobian 
                        -= 
                        da[right_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nnz+
                           right_ebN_element*nQuadraturePoints_elementBoundary*nnz+
                           k*nnz+
                           m]
                        *
                        grad_phi[right_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                                 right_ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                                 k*nSpace+
                                 colind[m]];
                      diffusiveVelocityComponent_I_leftJacobian2 
                        -= 
                        a[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nnz+
                          left_ebN_element*nQuadraturePoints_elementBoundary*nnz+
                          k*nnz+
                          m]
                        *
                        grad_v[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_trial_element*nSpace+
                               left_ebN_element*nQuadraturePoints_elementBoundary*nDOF_trial_element*nSpace+
                               k*nDOF_trial_element*nSpace+
                               j*nSpace+
                               colind[m]];
                      diffusiveVelocityComponent_I_rightJacobian2 
                        -= 
                        a[right_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nnz+
                          right_ebN_element*nQuadraturePoints_elementBoundary*nnz+
                          k*nnz+
                          m]
                        *
                        grad_v[right_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_trial_element*nSpace+
                               right_ebN_element*nQuadraturePoints_elementBoundary*nDOF_trial_element*nSpace+
                               k*nDOF_trial_element*nSpace+
                               j*nSpace+
                               colind[m]];
                      max_a = fmax(max_a,a[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nnz+
                                           left_ebN_element*nQuadraturePoints_elementBoundary*nnz+
                                           k*nnz+
                                           m]);
                      max_a = fmax(max_a,a[right_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nnz+
                                           right_ebN_element*nQuadraturePoints_elementBoundary*nnz+
                                           k*nnz+
                                           m]);
                      
                    }
                  leftJacobian 
                    += 
                    (diffusiveVelocityComponent_I_leftJacobian
                     *
                     v[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                       left_ebN_element*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                       k*nDOF_trial_element+
                       j] 
                     +
                     diffusiveVelocityComponent_I_leftJacobian2*
                     dphi[left_j_global])
                    *
                    n[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                      left_ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                      k*nSpace+
                      I];
                  rightJacobian 
                    += 
                    (diffusiveVelocityComponent_I_rightJacobian
                     *
                     v[right_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                       right_ebN_element*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                       k*nDOF_trial_element+
                       j] 
                     +
                     diffusiveVelocityComponent_I_rightJacobian2*dphi[right_j_global])
                    *
                    n[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                      left_ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                      k*nSpace+
                      I];
                }
              leftJacobian *= 0.5;
              rightJacobian *= 0.5;
              max_a = fmax(max_a,penalty_floor);
              double penaltyJacobian_term = penalty[ebN*nQuadraturePoints_elementBoundary+
                                                    k];
              if (scale_penalty) penaltyJacobian_term *= max_a;
              leftJacobian 
                += 
                penaltyJacobian_term
                *
                v[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                  left_ebN_element*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                  k*nDOF_trial_element+
                  j];
              rightJacobian 
                -= 
                penaltyJacobian_term
                *
                v[right_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                  right_ebN_element*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                  k*nDOF_trial_element+
                  j];
              fluxJacobian[ebN*2*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                           0*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                           k*nDOF_trial_element+
                           j] += leftJacobian;
              fluxJacobian[ebN*2*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                           1*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                           k*nDOF_trial_element+
                           j] += rightJacobian;
            }
        }
    }
}

/*
   \brief Calculate the diffusive flux Jacobian at interior element boundary quadrature points
 
void updateInteriorNumericalDiffusiveFluxJacobian(proteus_index_t nInteriorElementBoundaries_global,
                                                  int nElementBoundaries_element,
                                                  int nQuadraturePoints_elementBoundary,
                                                  int nDOF_trial_element,
                                                  int nSpace,
                                                  proteus_index_t* l2g,
                                                  proteus_index_t* interiorElementBoundaries,
                                                  proteus_index_t* elementBoundaryElements,
                                                  int* elementBoundaryLocalElementBoundaries,
                                                  double* n,
                                                  double* a,
                                                  double* da,
                                                  double* grad_phi,
                                                  double* dphi,
                                                  double* v,
                                                  double* grad_v,
                                                  double* penalty,
                                                  double* fluxJacobian)
{
  proteus_index_t ebN, left_eN_global, right_eN_global, left_j_global, right_j_global; int ebNI, left_ebN_element, right_ebN_element, k, j, I, J, nSpace2=nSpace*nSpace;
  double leftJacobian,rightJacobian,diffusiveVelocityComponent_I_leftJacobian,diffusiveVelocityComponent_I_rightJacobian,diffusiveVelocityComponent_I_leftJacobian2,diffusiveVelocityComponent_I_rightJacobian2;
  for(ebNI=0;ebNI<nInteriorElementBoundaries_global;ebNI++)
    {
      ebN = interiorElementBoundaries[ebNI];
      left_eN_global = elementBoundaryElements[ebN*2+0];
      right_eN_global = elementBoundaryElements[ebN*2+1];
      left_ebN_element = elementBoundaryLocalElementBoundaries[ebN*2+0];
      right_ebN_element = elementBoundaryLocalElementBoundaries[ebN*2+1];
      for(k=0;k<nQuadraturePoints_elementBoundary;k++)
        {
          for(j=0;j<nDOF_trial_element;j++)
            {
              leftJacobian=0.0;
              rightJacobian=0.0;
              left_j_global = l2g[left_eN_global*nDOF_trial_element+j];
              right_j_global= l2g[right_eN_global*nDOF_trial_element+j];
              for(I=0;I<nSpace;I++)
                {
                  diffusiveVelocityComponent_I_leftJacobian=0.0;
                  diffusiveVelocityComponent_I_leftJacobian2=0.0;
                  diffusiveVelocityComponent_I_rightJacobian=0.0;
                  diffusiveVelocityComponent_I_rightJacobian2=0.0;
                  for(J=0;J<nSpace;J++)
                    {
                      diffusiveVelocityComponent_I_leftJacobian 
                        -= 
                        da[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace2+
                           left_ebN_element*nQuadraturePoints_elementBoundary*nSpace2+
                           k*nSpace2+
                           I*nSpace+
                           J]
                        *
                        grad_phi[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                                 left_ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                                 k*nSpace+
                                 J];
                      diffusiveVelocityComponent_I_rightJacobian 
                        -= 
                        da[right_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace2+
                           right_ebN_element*nQuadraturePoints_elementBoundary*nSpace2+
                           k*nSpace2+
                           I*nSpace+
                           J]
                        *
                        grad_phi[right_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                                 right_ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                                 k*nSpace+
                                 J];
                      diffusiveVelocityComponent_I_leftJacobian2 
                        -= 
                        a[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace2+
                          left_ebN_element*nQuadraturePoints_elementBoundary*nSpace2+
                          k*nSpace2+
                          I*nSpace+
                          J]
                        *
                        grad_v[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_trial_element*nSpace+
                               left_ebN_element*nQuadraturePoints_elementBoundary*nDOF_trial_element*nSpace+
                               k*nDOF_trial_element*nSpace+
                               j*nSpace+
                               J];
                      diffusiveVelocityComponent_I_rightJacobian2 
                        -= 
                        a[right_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace2+
                          right_ebN_element*nQuadraturePoints_elementBoundary*nSpace2+
                          k*nSpace2+
                          I*nSpace+
                          J]
                        *
                        grad_v[right_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_trial_element*nSpace+
                               right_ebN_element*nQuadraturePoints_elementBoundary*nDOF_trial_element*nSpace+
                               k*nDOF_trial_element*nSpace+
                               j*nSpace+
                               J];
                      
                    }
                  leftJacobian 
                    += 
                    (diffusiveVelocityComponent_I_leftJacobian
                     *
                     v[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                       left_ebN_element*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                       k*nDOF_trial_element+
                       j] 
                     +
                     diffusiveVelocityComponent_I_leftJacobian2*
                     dphi[left_j_global])
                    *
                    n[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                      left_ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                      k*nSpace+
                      I];
                  rightJacobian 
                    += 
                    (diffusiveVelocityComponent_I_rightJacobian
                     *
                     v[right_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                       right_ebN_element*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                       k*nDOF_trial_element+
                       j] 
                     +
                     diffusiveVelocityComponent_I_rightJacobian2*dphi[right_j_global])
                    *
                    n[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                      left_ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                      k*nSpace+
                      I];
                }
              leftJacobian *= 0.5;
              rightJacobian *= 0.5;
              leftJacobian 
                += 
                penalty[ebN*nQuadraturePoints_elementBoundary+
                        k]
                *
                v[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                  left_ebN_element*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                  k*nDOF_trial_element+
                  j];
              rightJacobian 
                -= 
                penalty[ebN*nQuadraturePoints_elementBoundary+
                        k]
                *
                v[right_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                  right_ebN_element*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                  k*nDOF_trial_element+
                  j];
              fluxJacobian[ebN*2*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                           0*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                           k*nDOF_trial_element+
                           j] += leftJacobian;
              fluxJacobian[ebN*2*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                           1*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                           k*nDOF_trial_element+
                           j] += rightJacobian;
            }
        }
    }
}

void updateInteriorNumericalDiffusiveFluxJacobian_sd(proteus_index_t nInteriorElementBoundaries_global,
						     int nElementBoundaries_element,
						     int nQuadraturePoints_elementBoundary,
						     int nDOF_trial_element,
						     int nSpace,
						     proteus_csr_index_t* rowptr,
						     proteus_csr_index_t* colind,
						     proteus_index_t* l2g,
						     proteus_index_t* interiorElementBoundaries,
						     proteus_index_t* elementBoundaryElements,
						     int* elementBoundaryLocalElementBoundaries,
						     double* n,
						     double* a,
						     double* da,
						     double* grad_phi,
						     double* dphi,
						     double* v,
						     double* grad_v,
						     double* penalty,
						     double* fluxJacobian)
{
  proteus_index_t ebN, left_eN_global, right_eN_global, left_j_global, right_j_global, m; proteus_csr_index_t nnz=rowptr[nSpace]; int ebNI, left_ebN_element, right_ebN_element, k, j, I;
  double leftJacobian,rightJacobian,diffusiveVelocityComponent_I_leftJacobian,diffusiveVelocityComponent_I_rightJacobian,diffusiveVelocityComponent_I_leftJacobian2,diffusiveVelocityComponent_I_rightJacobian2;
  for(ebNI=0;ebNI<nInteriorElementBoundaries_global;ebNI++)
    {
      ebN = interiorElementBoundaries[ebNI];
      left_eN_global = elementBoundaryElements[ebN*2+0];
      right_eN_global = elementBoundaryElements[ebN*2+1];
      left_ebN_element = elementBoundaryLocalElementBoundaries[ebN*2+0];
      right_ebN_element = elementBoundaryLocalElementBoundaries[ebN*2+1];
      for(k=0;k<nQuadraturePoints_elementBoundary;k++)
        {
          for(j=0;j<nDOF_trial_element;j++)
            {
              leftJacobian=0.0;
              rightJacobian=0.0;
              left_j_global = l2g[left_eN_global*nDOF_trial_element+j];
              right_j_global= l2g[right_eN_global*nDOF_trial_element+j];
              for(I=0;I<nSpace;I++)
                {
                  diffusiveVelocityComponent_I_leftJacobian=0.0;
                  diffusiveVelocityComponent_I_leftJacobian2=0.0;
                  diffusiveVelocityComponent_I_rightJacobian=0.0;
                  diffusiveVelocityComponent_I_rightJacobian2=0.0;
                  for(m=rowptr[I];m<rowptr[I+1];m++)
                    {
                      diffusiveVelocityComponent_I_leftJacobian 
                        -= 
                        da[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nnz+
                           left_ebN_element*nQuadraturePoints_elementBoundary*nnz+
                           k*nnz+
                           m]
                        *
                        grad_phi[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                                 left_ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                                 k*nSpace+
                                 colind[m]];
                      diffusiveVelocityComponent_I_rightJacobian 
                        -= 
                        da[right_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nnz+
                           right_ebN_element*nQuadraturePoints_elementBoundary*nnz+
                           k*nnz+
                           m]
                        *
                        grad_phi[right_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                                 right_ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                                 k*nSpace+
                                 colind[m]];
                      diffusiveVelocityComponent_I_leftJacobian2 
                        -= 
                        a[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nnz+
                          left_ebN_element*nQuadraturePoints_elementBoundary*nnz+
                          k*nnz+
                          m]
                        *
                        grad_v[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_trial_element*nSpace+
                               left_ebN_element*nQuadraturePoints_elementBoundary*nDOF_trial_element*nSpace+
                               k*nDOF_trial_element*nSpace+
                               j*nSpace+
                               colind[m]];
                      diffusiveVelocityComponent_I_rightJacobian2 
                        -= 
                        a[right_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nnz+
                          right_ebN_element*nQuadraturePoints_elementBoundary*nnz+
                          k*nnz+
                          m]
                        *
                        grad_v[right_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_trial_element*nSpace+
                               right_ebN_element*nQuadraturePoints_elementBoundary*nDOF_trial_element*nSpace+
                               k*nDOF_trial_element*nSpace+
                               j*nSpace+
                               colind[m]];
                      
                    }
                  leftJacobian 
                    += 
                    (diffusiveVelocityComponent_I_leftJacobian
                     *
                     v[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                       left_ebN_element*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                       k*nDOF_trial_element+
                       j] 
                     +
                     diffusiveVelocityComponent_I_leftJacobian2*
                     dphi[left_j_global])
                    *
                    n[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                      left_ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                      k*nSpace+
                      I];
                  rightJacobian 
                    += 
                    (diffusiveVelocityComponent_I_rightJacobian
                     *
                     v[right_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                       right_ebN_element*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                       k*nDOF_trial_element+
                       j] 
                     +
                     diffusiveVelocityComponent_I_rightJacobian2*dphi[right_j_global])
                    *
                    n[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                      left_ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                      k*nSpace+
                      I];
                }
              leftJacobian *= 0.5;
              rightJacobian *= 0.5;
              leftJacobian 
                += 
                penalty[ebN*nQuadraturePoints_elementBoundary+
                        k]
                *
                v[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                  left_ebN_element*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                  k*nDOF_trial_element+
                  j];
              rightJacobian 
                -= 
                penalty[ebN*nQuadraturePoints_elementBoundary+
                        k]
                *
                v[right_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                  right_ebN_element*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                  k*nDOF_trial_element+
                  j];
              fluxJacobian[ebN*2*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                           0*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                           k*nDOF_trial_element+
                           j] += leftJacobian;
              fluxJacobian[ebN*2*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                           1*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                           k*nDOF_trial_element+
                           j] += rightJacobian;
            }
        }
    }
}
*/

/**
   \brief Calculate the diffusive flux at exterior element boundary quadrature points
*/ 
void calculateExteriorNumericalDiffusiveFlux(int scale_penalty,
                                             double penalty_floor,
                                             proteus_index_t nExteriorElementBoundaries_global,
                                             int nElementBoundaries_element,
                                             int nQuadraturePoints_elementBoundary,
                                             int nSpace,
                                             proteus_index_t* exteriorElementBoundaries,
                                             proteus_index_t* elementBoundaryElements,
                                             int* elementBoundaryLocalElementBoundaries,
                                             int* isDOFBoundary,
                                             double* n,
                                             double* bc_a,
                                             double* bc_grad_phi,
                                             double* bc_u,
                                             double* a,
                                             double* grad_phi,
                                             double* u,
                                             double* penalty,
                                             double* flux)
{
  proteus_index_t ebN, eN_global; int ebNE, ebN_element, k, J, I, nSpace2=nSpace*nSpace;
  double diffusiveVelocityComponent_I,penaltyFlux,max_a=0.0;
  for(ebNE=0;ebNE<nExteriorElementBoundaries_global;ebNE++)
    {
      ebN = exteriorElementBoundaries[ebNE];
      eN_global = elementBoundaryElements[ebN*2+0];
      ebN_element = elementBoundaryLocalElementBoundaries[ebN*2+0];
      for(k=0;k<nQuadraturePoints_elementBoundary;k++)
        {
          if(isDOFBoundary[ebNE*nQuadraturePoints_elementBoundary+k] == 1)
            {
              max_a = 0.0;
              flux[ebN*nQuadraturePoints_elementBoundary+k] = 0.0;
              for(I=0;I<nSpace;I++)
                {
                  diffusiveVelocityComponent_I=0.0;
                  for(J=0;J<nSpace;J++)
                    {
                      diffusiveVelocityComponent_I 
                        -= 
                        a[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace2+
                          ebN_element*nQuadraturePoints_elementBoundary*nSpace2+
                          k*nSpace2+
                          I*nSpace+
                          J]
                        *
                        grad_phi[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                                 ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                                 k*nSpace+J];
                      max_a = fmax(max_a,a[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace2+
                                           ebN_element*nQuadraturePoints_elementBoundary*nSpace2+
                                           k*nSpace2+
                                           I*nSpace+
                                           J]);
                    }
                  flux[ebN*nQuadraturePoints_elementBoundary+k] 
                    += 
                    diffusiveVelocityComponent_I
                    *
                    n[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                      ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                      k*nSpace+
                      I];
                }
              max_a = fmax(penalty_floor,max_a);
              penaltyFlux = penalty[ebN*nQuadraturePoints_elementBoundary+
                                          k]
                *
                (u[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary+
                   ebN_element*nQuadraturePoints_elementBoundary+
                   k]
                 -
                 bc_u[ebNE*nQuadraturePoints_elementBoundary+
                      k]);
              if (scale_penalty) penaltyFlux *= max_a;
              flux[ebN*nQuadraturePoints_elementBoundary+k]  += penaltyFlux;
            }
        }
    }
}
void calculateExteriorNumericalDiffusiveFlux_sd(int scale_penalty,
                                                double penalty_floor,
                                                proteus_index_t nExteriorElementBoundaries_global,
						int nElementBoundaries_element,
						int nQuadraturePoints_elementBoundary,
						int nSpace,
						proteus_csr_index_t* rowptr,
						proteus_csr_index_t* colind,
						proteus_index_t* exteriorElementBoundaries,
						proteus_index_t* elementBoundaryElements,
						int* elementBoundaryLocalElementBoundaries,
						int* isDOFBoundary,
						double* n,
						double* bc_a,
						double* bc_grad_phi,
						double* bc_u,
						double* a,
						double* grad_phi,
						double* u,
						double* penalty,
						double* flux)
{
  proteus_index_t ebN, eN_global; proteus_csr_index_t m; proteus_csr_index_t nnz=rowptr[nSpace]; int ebNE, ebN_element, k, I;
  double diffusiveVelocityComponent_I,penaltyFlux,max_a=0.0;
  for(ebNE=0;ebNE<nExteriorElementBoundaries_global;ebNE++)
    {
      ebN = exteriorElementBoundaries[ebNE];
      eN_global = elementBoundaryElements[ebN*2+0];
      ebN_element = elementBoundaryLocalElementBoundaries[ebN*2+0];
      for(k=0;k<nQuadraturePoints_elementBoundary;k++)
        {
	  flux[ebN*nQuadraturePoints_elementBoundary+k] = 0.0;//cek initializing to zero, hack, need to warn user if diffusion with no Dirichlet or Neumann condition 
          if(isDOFBoundary[ebNE*nQuadraturePoints_elementBoundary+k] == 1)
            {
              max_a = 0.0;
              for(I=0;I<nSpace;I++)
                {
                  diffusiveVelocityComponent_I=0.0;
                  for(m=rowptr[I];m<rowptr[I+1];m++)
                    {
                      diffusiveVelocityComponent_I 
                        -= 
                        a[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nnz+
                          ebN_element*nQuadraturePoints_elementBoundary*nnz+
                          k*nnz+
                          m]
                        *
                        grad_phi[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                                 ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                                 k*nSpace+colind[m]];
                      max_a = fmax(max_a,a[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nnz+
                                          ebN_element*nQuadraturePoints_elementBoundary*nnz+
                                          k*nnz+
                                          m]);
                    }
                  flux[ebN*nQuadraturePoints_elementBoundary+k] 
                    += 
                    diffusiveVelocityComponent_I
                    *
                    n[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                      ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                      k*nSpace+
                      I];
                }
              max_a = fmax(penalty_floor,max_a);
              penaltyFlux = penalty[ebN*nQuadraturePoints_elementBoundary+
                                    k]
                *
                (u[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary+
                   ebN_element*nQuadraturePoints_elementBoundary+
                   k]
                 -
                 bc_u[ebNE*nQuadraturePoints_elementBoundary+
                      k]);
              if (scale_penalty) penaltyFlux *= max_a;
              flux[ebN*nQuadraturePoints_elementBoundary+k]  += penaltyFlux;
            }
        }
    }
}
/**
   \brief Calculate the diffusive flux at exterior element boundary quadrature points
*/ 
void calculateGlobalExteriorNumericalDiffusiveFlux(int scale_penalty,
                                                   double penalty_floor,
                                                   proteus_index_t nExteriorElementBoundaries_global,
						   int nQuadraturePoints_elementBoundary,
						   int nSpace,
						   proteus_index_t* exteriorElementBoundaries,
						   proteus_index_t* elementBoundaryElements,
						   int* elementBoundaryLocalElementBoundaries,
						   int* isDOFBoundary,
						   double* n,
						   double* bc_a,
						   double* bc_grad_phi,
						   double* bc_u,
						   double* a,
						   double* grad_phi,
						   double* u,
						   double* penalty,
						   double* flux)
{
  int ebNE,k,J,I,nSpace2=nSpace*nSpace;
  double diffusiveVelocityComponent_I,penaltyFlux,max_a=0.0;
  for(ebNE=0;ebNE<nExteriorElementBoundaries_global;ebNE++)
    {
      for(k=0;k<nQuadraturePoints_elementBoundary;k++)
        {
          if(isDOFBoundary[ebNE*nQuadraturePoints_elementBoundary+k] == 1)
            {
              flux[ebNE*nQuadraturePoints_elementBoundary+k] = 0.0;
              max_a=0.0;
              for(I=0;I<nSpace;I++)
                {
                  diffusiveVelocityComponent_I=0.0;
                  for(J=0;J<nSpace;J++)
                    {
                      diffusiveVelocityComponent_I 
                        -= 
                        a[ebNE*nQuadraturePoints_elementBoundary*nSpace2+
                          k*nSpace2+
                          I*nSpace+
                          J]
                        *
                        grad_phi[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                                 k*nSpace+J];
                      max_a = fmax(max_a,a[ebNE*nQuadraturePoints_elementBoundary*nSpace2+
                                          k*nSpace2+
                                          I*nSpace+
                                          J]);
                    }
                  flux[ebNE*nQuadraturePoints_elementBoundary+k] 
                    += 
                    diffusiveVelocityComponent_I
                    *
                    n[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                      k*nSpace+
                      I];
                }
              max_a = fmax(max_a,penalty_floor);
              penaltyFlux = penalty[ebNE*nQuadraturePoints_elementBoundary+
                                          k]
                *
                (u[ebNE*nQuadraturePoints_elementBoundary+
                   k]
                 -
                 bc_u[ebNE*nQuadraturePoints_elementBoundary+
                      k]);
              if (scale_penalty) penaltyFlux *= max_a;
              flux[ebNE*nQuadraturePoints_elementBoundary+k]  += penaltyFlux;
            }
        }
    }
}

void calculateGlobalExteriorNumericalDiffusiveFlux_sd(int scale_penalty,
                                                      double penalty_floor,
                                                      proteus_index_t nExteriorElementBoundaries_global,
						      int nQuadraturePoints_elementBoundary,
						      int nSpace,
						      proteus_csr_index_t* rowptr,
						      proteus_csr_index_t* colind,
						      proteus_index_t* exteriorElementBoundaries,
						      proteus_index_t* elementBoundaryElements,
						      int* elementBoundaryLocalElementBoundaries,
						      int* isDOFBoundary,
						      double* n,
						      double* bc_a,
						      double* bc_grad_phi,
						      double* bc_u,
						      double* a,
						      double* grad_phi,
						      double* u,
						      double* penalty,
						      double* flux)
{
  proteus_csr_index_t m; proteus_csr_index_t nnz=rowptr[nSpace]; int ebNE, k, I;
  double diffusiveVelocityComponent_I,penaltyFlux,max_a;
  for(ebNE=0;ebNE<nExteriorElementBoundaries_global;ebNE++)
    {
      for(k=0;k<nQuadraturePoints_elementBoundary;k++)
        {
          if(isDOFBoundary[ebNE*nQuadraturePoints_elementBoundary+k] == 1)
            {
              flux[ebNE*nQuadraturePoints_elementBoundary+k] = 0.0;
              max_a=0.0;
              for(I=0;I<nSpace;I++)
                {
                  diffusiveVelocityComponent_I=0.0;
                  for(m=rowptr[I];m<rowptr[I+1];m++)
                    {
                      diffusiveVelocityComponent_I 
                        -= 
                        a[ebNE*nQuadraturePoints_elementBoundary*nnz+
                          k*nnz+
			  m]
                        *
                        grad_phi[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                                 k*nSpace+colind[m]];
                      max_a = fmax(max_a,a[ebNE*nQuadraturePoints_elementBoundary*nnz+
                                          k*nnz+
					   m]);
                    }
                  flux[ebNE*nQuadraturePoints_elementBoundary+k] 
                    += 
                    diffusiveVelocityComponent_I
                    *
                    n[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                      k*nSpace+
                      I];
                }
              max_a = fmax(penalty_floor,max_a);
              penaltyFlux = penalty[ebNE*nQuadraturePoints_elementBoundary+
                                    k]
                *
                (u[ebNE*nQuadraturePoints_elementBoundary+
                   k]
                 -
                 bc_u[ebNE*nQuadraturePoints_elementBoundary+
                      k]);

              if (scale_penalty) penaltyFlux *= max_a;
              flux[ebNE*nQuadraturePoints_elementBoundary+k]  += penaltyFlux;
            }
        }
    }
}


/*
   \brief Calculate the diffusive flux at exterior element boundary quadrature points

void calculateExteriorNumericalDiffusiveFlux(proteus_index_t nExteriorElementBoundaries_global,
                                             int nElementBoundaries_element,
                                             int nQuadraturePoints_elementBoundary,
                                             int nSpace,
                                             proteus_index_t* exteriorElementBoundaries,
                                             proteus_index_t* elementBoundaryElements,
                                             int* elementBoundaryLocalElementBoundaries,
                                             int* isDOFBoundary,
                                             double* n,
                                             double* bc_a,
                                             double* bc_grad_phi,
                                             double* bc_u,
                                             double* a,
                                             double* grad_phi,
                                             double* u,
                                             double* penalty,
                                             double* flux)
{
  proteus_index_t ebN, eN_global; int ebNE, ebN_element, k, J, I, nSpace2=nSpace*nSpace;
  double diffusiveVelocityComponent_I,penaltyFlux;
  for(ebNE=0;ebNE<nExteriorElementBoundaries_global;ebNE++)
    {
      ebN = exteriorElementBoundaries[ebNE];
      eN_global = elementBoundaryElements[ebN*2+0];
      ebN_element = elementBoundaryLocalElementBoundaries[ebN*2+0];
      for(k=0;k<nQuadraturePoints_elementBoundary;k++)
        {
          if(isDOFBoundary[ebNE*nQuadraturePoints_elementBoundary+k] == 1)
            {
              flux[ebN*nQuadraturePoints_elementBoundary+k] = 0.0;
              for(I=0;I<nSpace;I++)
                {
                  diffusiveVelocityComponent_I=0.0;
                  for(J=0;J<nSpace;J++)
                    {
                      diffusiveVelocityComponent_I 
                        -= 
                        a[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace2+
                          ebN_element*nQuadraturePoints_elementBoundary*nSpace2+
                          k*nSpace2+
                          I*nSpace+
                          J]
                        *
                        grad_phi[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                                 ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                                 k*nSpace+J];
                    }
                  flux[ebN*nQuadraturePoints_elementBoundary+k] 
                    += 
                    diffusiveVelocityComponent_I
                    *
                    n[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                      ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                      k*nSpace+
                      I];
                }
              penaltyFlux = penalty[ebN*nQuadraturePoints_elementBoundary+
                                    k]
                *
                (u[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary+
                   ebN_element*nQuadraturePoints_elementBoundary+
                   k]
                 -
                 bc_u[ebNE*nQuadraturePoints_elementBoundary+
                      k]);
              flux[ebN*nQuadraturePoints_elementBoundary+k]  += penaltyFlux;
            }
        }
    }
}
void calculateExteriorNumericalDiffusiveFlux_sd(proteus_index_t nExteriorElementBoundaries_global,
						int nElementBoundaries_element,
						int nQuadraturePoints_elementBoundary,
						int nSpace,
						proteus_csr_index_t* rowptr,
						proteus_csr_index_t* colind,
						proteus_index_t* exteriorElementBoundaries,
						proteus_index_t* elementBoundaryElements,
						int* elementBoundaryLocalElementBoundaries,
						int* isDOFBoundary,
						double* n,
						double* bc_a,
						double* bc_grad_phi,
						double* bc_u,
						double* a,
						double* grad_phi,
						double* u,
						double* penalty,
						double* flux)
{
  proteus_index_t ebN, eN_global, m; proteus_csr_index_t nnz=rowptr[nSpace]; int ebNE, ebN_element, k, I;
  double diffusiveVelocityComponent_I,penaltyFlux;
  for(ebNE=0;ebNE<nExteriorElementBoundaries_global;ebNE++)
    {
      ebN = exteriorElementBoundaries[ebNE];
      eN_global = elementBoundaryElements[ebN*2+0];
      ebN_element = elementBoundaryLocalElementBoundaries[ebN*2+0];
      for(k=0;k<nQuadraturePoints_elementBoundary;k++)
        {
	  flux[ebN*nQuadraturePoints_elementBoundary+k] = 0.0;//cek initializing to zero, hack, need to warn user if diffusion with no Dirichlet or Neumann condition 
          if(isDOFBoundary[ebNE*nQuadraturePoints_elementBoundary+k] == 1)
            {
              for(I=0;I<nSpace;I++)
                {
                  diffusiveVelocityComponent_I=0.0;
                  for(m=rowptr[I];m<rowptr[I+1];m++)
                    {
                      diffusiveVelocityComponent_I 
                        -= 
                        a[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nnz+
                          ebN_element*nQuadraturePoints_elementBoundary*nnz+
                          k*nnz+
                          m]
                        *
                        grad_phi[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                                 ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                                 k*nSpace+colind[m]];
                    }
                  flux[ebN*nQuadraturePoints_elementBoundary+k] 
                    += 
                    diffusiveVelocityComponent_I
                    *
                    n[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                      ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                      k*nSpace+
                      I];
                }
              penaltyFlux = penalty[ebN*nQuadraturePoints_elementBoundary+
                                    k]
                *
                (u[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary+
                   ebN_element*nQuadraturePoints_elementBoundary+
                   k]
                 -
                 bc_u[ebNE*nQuadraturePoints_elementBoundary+
                      k]);
              flux[ebN*nQuadraturePoints_elementBoundary+k]  += penaltyFlux;
            }
        }
    }
} */
/*
   \brief Calculate the diffusive flux at exterior element boundary quadrature points
   
void calculateGlobalExteriorNumericalDiffusiveFlux(proteus_index_t nExteriorElementBoundaries_global,
						   int nQuadraturePoints_elementBoundary,
						   int nSpace,
						   proteus_index_t* exteriorElementBoundaries,
						   proteus_index_t* elementBoundaryElements,
						   int* elementBoundaryLocalElementBoundaries,
						   int* isDOFBoundary,
						   double* n,
						   double* bc_a,
						   double* bc_grad_phi,
						   double* bc_u,
						   double* a,
						   double* grad_phi,
						   double* u,
						   double* penalty,
						   double* flux)
{
  int ebNE,k,J,I,nSpace2=nSpace*nSpace;
  double diffusiveVelocityComponent_I,penaltyFlux,max_a;
  for(ebNE=0;ebNE<nExteriorElementBoundaries_global;ebNE++)
    {
      for(k=0;k<nQuadraturePoints_elementBoundary;k++)
        {
          if(isDOFBoundary[ebNE*nQuadraturePoints_elementBoundary+k] == 1)
            {
              flux[ebNE*nQuadraturePoints_elementBoundary+k] = 0.0;
              max_a=0.0;
              for(I=0;I<nSpace;I++)
                {
                  diffusiveVelocityComponent_I=0.0;
                  for(J=0;J<nSpace;J++)
                    {
                      diffusiveVelocityComponent_I 
                        -= 
                        a[ebNE*nQuadraturePoints_elementBoundary*nSpace2+
                          k*nSpace2+
                          I*nSpace+
                          J]
                        *
                        grad_phi[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                                 k*nSpace+J];
                      max_a = fmax(max_a,a[ebNE*nQuadraturePoints_elementBoundary*nSpace2+
                                          k*nSpace2+
                                          I*nSpace+
                                          J]);
                    }
                  flux[ebNE*nQuadraturePoints_elementBoundary+k] 
                    += 
                    diffusiveVelocityComponent_I
                    *
                    n[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                      k*nSpace+
                      I];
                }
              penaltyFlux = penalty[ebNE*nQuadraturePoints_elementBoundary+
                                    k]
                *
                (u[ebNE*nQuadraturePoints_elementBoundary+
                   k]
                 -
                 bc_u[ebNE*nQuadraturePoints_elementBoundary+
                      k]);
              flux[ebNE*nQuadraturePoints_elementBoundary+k]  += penaltyFlux;
            }
        }
    }
}

void calculateGlobalExteriorNumericalDiffusiveFlux_sd(proteus_index_t nExteriorElementBoundaries_global,
						      int nQuadraturePoints_elementBoundary,
						      int nSpace,
						      proteus_csr_index_t* rowptr,
						      proteus_csr_index_t* colind,
						      proteus_index_t* exteriorElementBoundaries,
						      proteus_index_t* elementBoundaryElements,
						      int* elementBoundaryLocalElementBoundaries,
						      int* isDOFBoundary,
						      double* n,
						      double* bc_a,
						      double* bc_grad_phi,
						      double* bc_u,
						      double* a,
						      double* grad_phi,
						      double* u,
						      double* penalty,
						      double* flux)
{
  proteus_index_t m; proteus_csr_index_t nnz=rowptr[nSpace]; int ebNE, k, I;
  double diffusiveVelocityComponent_I,penaltyFlux,max_a;
  for(ebNE=0;ebNE<nExteriorElementBoundaries_global;ebNE++)
    {
      for(k=0;k<nQuadraturePoints_elementBoundary;k++)
        {
          if(isDOFBoundary[ebNE*nQuadraturePoints_elementBoundary+k] == 1)
            {
              flux[ebNE*nQuadraturePoints_elementBoundary+k] = 0.0;
              max_a=0.0;
              for(I=0;I<nSpace;I++)
                {
                  diffusiveVelocityComponent_I=0.0;
                  for(m=rowptr[I];m<rowptr[I+1];m++)
                    {
                      diffusiveVelocityComponent_I 
                        -= 
                        a[ebNE*nQuadraturePoints_elementBoundary*nnz+
                          k*nnz+
			  m]
                        *
                        grad_phi[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                                 k*nSpace+colind[m]];
                      max_a = fmax(max_a,a[ebNE*nQuadraturePoints_elementBoundary*nnz+
                                          k*nnz+
					   m]);
                    }
                  flux[ebNE*nQuadraturePoints_elementBoundary+k] 
                    += 
                    diffusiveVelocityComponent_I
                    *
                    n[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                      k*nSpace+
                      I];
                }
              penaltyFlux = penalty[ebNE*nQuadraturePoints_elementBoundary+
                                    k]
                *
                (u[ebNE*nQuadraturePoints_elementBoundary+
                   k]
                 -
                 bc_u[ebNE*nQuadraturePoints_elementBoundary+
                      k]);
              flux[ebNE*nQuadraturePoints_elementBoundary+k]  += penaltyFlux;
            }
        }
    }
}
*/

void calculateExteriorNumericalDiffusiveFlux_free(proteus_index_t nExteriorElementBoundaries_global,
                                             int nElementBoundaries_element,
                                             int nQuadraturePoints_elementBoundary,
                                             int nSpace,
                                             proteus_index_t* exteriorElementBoundaries,
                                             proteus_index_t* elementBoundaryElements,
                                             int* elementBoundaryLocalElementBoundaries,
                                             int* isDOFBoundary,
                                             double* n,
                                             double* bc_a,
                                             double* bc_grad_phi,
                                             double* bc_u,
                                             double* a,
                                             double* grad_phi,
                                             double* u,
                                             double* penalty,
                                             double* flux)
{
  proteus_index_t ebN, eN_global; int ebNE, ebN_element, k, J, I, nSpace2=nSpace*nSpace;
  double diffusiveVelocityComponent_I;
  for(ebNE=0;ebNE<nExteriorElementBoundaries_global;ebNE++)
    {
      ebN = exteriorElementBoundaries[ebNE];
      eN_global = elementBoundaryElements[ebN*2+0];
      ebN_element = elementBoundaryLocalElementBoundaries[ebN*2+0];
      for(k=0;k<nQuadraturePoints_elementBoundary;k++)
        {
          flux[ebN*nQuadraturePoints_elementBoundary+k] = 0.0;
          for(I=0;I<nSpace;I++)
            {
              diffusiveVelocityComponent_I=0.0;
              for(J=0;J<nSpace;J++)
                {
                  diffusiveVelocityComponent_I 
                    -= 
                    a[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace2+
                      ebN_element*nQuadraturePoints_elementBoundary*nSpace2+
                      k*nSpace2+
                      I*nSpace+
                      J]
                    *
                    grad_phi[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                             ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                             k*nSpace+J];
                }
              flux[ebN*nQuadraturePoints_elementBoundary+k] 
                += 
                diffusiveVelocityComponent_I
                *
                n[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                  ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                  k*nSpace+
                  I];
            }
        }
    }
}
void calculateExteriorNumericalDiffusiveFlux_free_sd(proteus_index_t nExteriorElementBoundaries_global,
						     int nElementBoundaries_element,
						     int nQuadraturePoints_elementBoundary,
						     int nSpace,
						     proteus_csr_index_t* rowptr,
						     proteus_csr_index_t* colind,
						     proteus_index_t* exteriorElementBoundaries,
						     proteus_index_t* elementBoundaryElements,
						     int* elementBoundaryLocalElementBoundaries,
						     int* isDOFBoundary,
						     double* n,
						     double* bc_a,
						     double* bc_grad_phi,
						     double* bc_u,
						     double* a,
						     double* grad_phi,
						     double* u,
						     double* penalty,
						     double* flux)
{
  proteus_index_t ebN, eN_global; proteus_csr_index_t m; proteus_csr_index_t nnz=rowptr[nSpace]; int ebNE, ebN_element, k, I;
  double diffusiveVelocityComponent_I;
  for(ebNE=0;ebNE<nExteriorElementBoundaries_global;ebNE++)
    {
      ebN = exteriorElementBoundaries[ebNE];
      eN_global = elementBoundaryElements[ebN*2+0];
      ebN_element = elementBoundaryLocalElementBoundaries[ebN*2+0];
      for(k=0;k<nQuadraturePoints_elementBoundary;k++)
        {
          flux[ebN*nQuadraturePoints_elementBoundary+k] = 0.0;
          for(I=0;I<nSpace;I++)
            {
              diffusiveVelocityComponent_I=0.0;
              for(m=rowptr[I];m<rowptr[I+1];m++)
                {
                  diffusiveVelocityComponent_I 
                    -= 
                    a[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nnz+
                      ebN_element*nQuadraturePoints_elementBoundary*nnz+
                      k*nnz+
                      m]
                    *
                    grad_phi[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                             ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                             k*nSpace+colind[m]];
                }
              flux[ebN*nQuadraturePoints_elementBoundary+k] 
                += 
                diffusiveVelocityComponent_I
                *
                n[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                  ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                  k*nSpace+
                  I];
            }
        }
    }
}
void calculateGlobalExteriorNumericalDiffusiveFlux_free(proteus_index_t nExteriorElementBoundaries_global,
							int nQuadraturePoints_elementBoundary,
							int nSpace,
							proteus_index_t* exteriorElementBoundaries,
							proteus_index_t* elementBoundaryElements,
							int* elementBoundaryLocalElementBoundaries,
							int* isDOFBoundary,
							double* n,
							double* bc_a,
							double* bc_grad_phi,
							double* bc_u,
							double* a,
							double* grad_phi,
							double* u,
							double* penalty,
							double* flux)
{
  int ebNE,k,J,I,nSpace2=nSpace*nSpace;
  double diffusiveVelocityComponent_I;
  for(ebNE=0;ebNE<nExteriorElementBoundaries_global;ebNE++)
    {
      for(k=0;k<nQuadraturePoints_elementBoundary;k++)
        {
          flux[ebNE*nQuadraturePoints_elementBoundary+k] = 0.0;
          for(I=0;I<nSpace;I++)
            {
              diffusiveVelocityComponent_I=0.0;
              for(J=0;J<nSpace;J++)
                {
                  diffusiveVelocityComponent_I 
                    -= 
                    a[ebNE*nQuadraturePoints_elementBoundary*nSpace2+
                      k*nSpace2+
                      I*nSpace+
                      J]
                    *
                    grad_phi[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                             k*nSpace+J];
                }
              flux[ebNE*nQuadraturePoints_elementBoundary+k] 
                += 
                diffusiveVelocityComponent_I
                *
                n[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                  k*nSpace+
                  I];
            }
        }
    }
}
void calculateGlobalExteriorNumericalDiffusiveFlux_free_sd(proteus_index_t nExteriorElementBoundaries_global,
							   int nQuadraturePoints_elementBoundary,
							   int nSpace,
							   proteus_csr_index_t* rowptr,
							   proteus_csr_index_t* colind,
							   proteus_index_t* exteriorElementBoundaries,
							   proteus_index_t* elementBoundaryElements,
							   int* elementBoundaryLocalElementBoundaries,
							   int* isDOFBoundary,
							   double* n,
							   double* bc_a,
							   double* bc_grad_phi,
							   double* bc_u,
							   double* a,
							   double* grad_phi,
							   double* u,
							   double* penalty,
							   double* flux)
{
  proteus_csr_index_t m; proteus_csr_index_t nnz=rowptr[nSpace]; int ebNE, k, I;
  double diffusiveVelocityComponent_I;
  for(ebNE=0;ebNE<nExteriorElementBoundaries_global;ebNE++)
    {
      for(k=0;k<nQuadraturePoints_elementBoundary;k++)
        {
          flux[ebNE*nQuadraturePoints_elementBoundary+k] = 0.0;
          for(I=0;I<nSpace;I++)
            {
              diffusiveVelocityComponent_I=0.0;
              for(m=rowptr[I];m<rowptr[I+1];m++)
                {
                  diffusiveVelocityComponent_I 
                    -= 
                    a[ebNE*nQuadraturePoints_elementBoundary*nnz+
                      k*nnz+
                      m]
                    *
                    grad_phi[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                             k*nSpace+colind[m]];
                }
              flux[ebNE*nQuadraturePoints_elementBoundary+k] 
                += 
                diffusiveVelocityComponent_I
                *
                n[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                  k*nSpace+
                  I];
            }
        }
    }
}

/**
   \brief Update the diffusive flux Jacobian at exterior element boundary quadrature points
*/ 
void updateExteriorNumericalDiffusiveFluxJacobian(int scale_penalty,
                                                  double penalty_floor,
                                                  proteus_index_t nExteriorElementBoundaries_global,
                                                  int nElementBoundaries_element,
                                                  int nQuadraturePoints_elementBoundary,
                                                  int nDOF_trial_element,
                                                  int nSpace,
                                                  proteus_index_t* l2g,
                                                  proteus_index_t* exteriorElementBoundaries,
                                                  proteus_index_t* elementBoundaryElements,
                                                  int* elementBoundaryLocalElementBoundaries,
                                                  int* isDOFBoundary,
                                                  double* n,
                                                  double* a,
                                                  double* da,
                                                  double* grad_phi,
                                                  double* dphi,
                                                  double* v,
                                                  double* grad_v,
                                                  double* penalty,
                                                  double* fluxJacobian)
{
  proteus_index_t ebN, eN_global, j_global; int ebNE, ebN_element, k, j, I, J, nSpace2=nSpace*nSpace;
  double Jacobian,diffusiveVelocityComponent_I_Jacobian,diffusiveVelocityComponent_I_Jacobian2,max_a=0.0;
  for(ebNE=0;ebNE<nExteriorElementBoundaries_global;ebNE++)
    {
      ebN = exteriorElementBoundaries[ebNE];
      eN_global = elementBoundaryElements[ebN*2+0];
      ebN_element = elementBoundaryLocalElementBoundaries[ebN*2+0];
      for(k=0;k<nQuadraturePoints_elementBoundary;k++)
        {
          if(isDOFBoundary[ebNE*nQuadraturePoints_elementBoundary+k] == 1)
            {
              max_a = 0.0;
              for(j=0;j<nDOF_trial_element;j++)
                {
                  Jacobian=0.0;
                  j_global = l2g[eN_global*nDOF_trial_element+j];
                  for(I=0;I<nSpace;I++)
                    {
                      diffusiveVelocityComponent_I_Jacobian=0.0;
                      diffusiveVelocityComponent_I_Jacobian2=0.0;
                      for(J=0;J<nSpace;J++)
                        {
                          diffusiveVelocityComponent_I_Jacobian 
                            -= 
                            da[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace2+
                               ebN_element*nQuadraturePoints_elementBoundary*nSpace2+
                               k*nSpace2+
                               I*nSpace+
                               J]
                            *
                            grad_phi[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                                     ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                                     k*nSpace+
                                     J];
                          diffusiveVelocityComponent_I_Jacobian2 
                            -= 
                            a[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace2+
                              ebN_element*nQuadraturePoints_elementBoundary*nSpace2+
                              k*nSpace2+
                              I*nSpace+
                              J]
                            *
                            grad_v[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_trial_element*nSpace+
                                   ebN_element*nQuadraturePoints_elementBoundary*nDOF_trial_element*nSpace+
                                   k*nDOF_trial_element*nSpace+
                                   j*nSpace+
                                   J];
                          max_a = fmax(max_a,a[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace2+
                                               ebN_element*nQuadraturePoints_elementBoundary*nSpace2+
                                               k*nSpace2+
                                               I*nSpace+
                                               J]);
                        }
                      Jacobian 
                        += 
                        (diffusiveVelocityComponent_I_Jacobian
                         *
                         v[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                           ebN_element*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                           k*nDOF_trial_element+
                           j] 
                         +
                         diffusiveVelocityComponent_I_Jacobian2*
                         dphi[j_global])
                        *
                        n[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                          ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                          k*nSpace+
                          I];
                    }
                  max_a = fmax(penalty_floor,max_a);
                  double penaltyJacobian = penalty[ebN*nQuadraturePoints_elementBoundary+
                                                   k]
                    *
                    v[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                      ebN_element*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                      k*nDOF_trial_element+
                      j];
                  if (scale_penalty) penaltyJacobian *= max_a;
                  Jacobian += penaltyJacobian;
                  fluxJacobian[ebN*2*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                               0*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                               k*nDOF_trial_element+
                               j]
                    += Jacobian;
                }
            }
        }
    }
}

void updateExteriorNumericalDiffusiveFluxJacobian_sd(int scale_penalty,
                                                     double penalty_floor,
                                                     proteus_index_t nExteriorElementBoundaries_global,
						     int nElementBoundaries_element,
						     int nQuadraturePoints_elementBoundary,
						     int nDOF_trial_element,
						     int nSpace,
						     proteus_csr_index_t* rowptr,
						     proteus_csr_index_t* colind,
						     proteus_index_t* l2g,
						     proteus_index_t* exteriorElementBoundaries,
						     proteus_index_t* elementBoundaryElements,
						     int* elementBoundaryLocalElementBoundaries,
						     int* isDOFBoundary,
						     double* n,
						     double* a,
						     double* da,
						     double* grad_phi,
						     double* dphi,
						     double* v,
						     double* grad_v,
						     double* penalty,
						     double* fluxJacobian)
{
  proteus_index_t ebN, eN_global, j_global; proteus_csr_index_t m; proteus_csr_index_t nnz=rowptr[nSpace]; int ebNE, ebN_element, k, j, I;
  double Jacobian,diffusiveVelocityComponent_I_Jacobian,diffusiveVelocityComponent_I_Jacobian2,max_a=0.0;
  for(ebNE=0;ebNE<nExteriorElementBoundaries_global;ebNE++)
    {
      ebN = exteriorElementBoundaries[ebNE];
      eN_global = elementBoundaryElements[ebN*2+0];
      ebN_element = elementBoundaryLocalElementBoundaries[ebN*2+0];
      for(k=0;k<nQuadraturePoints_elementBoundary;k++)
        {
          if(isDOFBoundary[ebNE*nQuadraturePoints_elementBoundary+k] == 1)
            {
              max_a=0.0;
              for(j=0;j<nDOF_trial_element;j++)
                {
                  Jacobian=0.0;
                  j_global = l2g[eN_global*nDOF_trial_element+j];
                  for(I=0;I<nSpace;I++)
                    {
                      diffusiveVelocityComponent_I_Jacobian=0.0;
                      diffusiveVelocityComponent_I_Jacobian2=0.0;
                      for(m=rowptr[I];m<rowptr[I+1];m++)
                        {
                          diffusiveVelocityComponent_I_Jacobian 
                            -= 
                            da[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nnz+
                               ebN_element*nQuadraturePoints_elementBoundary*nnz+
                               k*nnz+
                               m]
                            *
                            grad_phi[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                                     ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                                     k*nSpace+
                                     colind[m]];
                          diffusiveVelocityComponent_I_Jacobian2 
                            -= 
                            a[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nnz+
                              ebN_element*nQuadraturePoints_elementBoundary*nnz+
                              k*nnz+
                              m]
                            *
                            grad_v[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_trial_element*nSpace+
                                   ebN_element*nQuadraturePoints_elementBoundary*nDOF_trial_element*nSpace+
                                   k*nDOF_trial_element*nSpace+
                                   j*nSpace+
                                   colind[m]];
                          max_a = fmax(max_a,a[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nnz+
                                               ebN_element*nQuadraturePoints_elementBoundary*nnz+
                                               k*nnz+
                                               m]);
                        }
                      Jacobian 
                        += 
                        (diffusiveVelocityComponent_I_Jacobian
                         *
                         v[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                           ebN_element*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                           k*nDOF_trial_element+
                           j] 
                         +
                         diffusiveVelocityComponent_I_Jacobian2*
                         dphi[j_global])
                        *
                        n[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                          ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                          k*nSpace+
                          I];
                    }
                  max_a = fmax(penalty_floor,max_a);
                  double penaltyJacobian = penalty[ebN*nQuadraturePoints_elementBoundary+
                                                   k]
                    *
                    v[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                      ebN_element*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                      k*nDOF_trial_element+
                      j];

                  if (scale_penalty) penaltyJacobian *= max_a;

                  Jacobian += penaltyJacobian;
                    
                  fluxJacobian[ebN*2*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                               0*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                               k*nDOF_trial_element+
                               j]
                    += Jacobian;
                }
            }
        }
    }
}
/**
   \brief Update the diffusive flux Jacobian at exterior element boundary quadrature points
*/ 
void updateGlobalExteriorNumericalDiffusiveFluxJacobian(int scale_penalty,
                                                        double penalty_floor,
                                                        proteus_index_t nExteriorElementBoundaries_global,
							int nQuadraturePoints_elementBoundary,
							int nDOF_trial_element,
							int nSpace,
							proteus_index_t* l2g,
							proteus_index_t* exteriorElementBoundaries,
							proteus_index_t* elementBoundaryElements,
							int* elementBoundaryLocalElementBoundaries,
							int* isDOFBoundary,
							double* n,
							double* a,
							double* da,
							double* grad_phi,
							double* dphi,
							double* v,
							double* grad_v,
							double* penalty,
							double* fluxJacobian)
{
  proteus_index_t ebN, eN_global, j_global; int ebNE, k, j, I, J, nSpace2=nSpace*nSpace;
  double Jacobian,diffusiveVelocityComponent_I_Jacobian,diffusiveVelocityComponent_I_Jacobian2,max_a;
  for(ebNE=0;ebNE<nExteriorElementBoundaries_global;ebNE++)
    {
      ebN = exteriorElementBoundaries[ebNE];
      eN_global = elementBoundaryElements[ebN*2+0];
      for(k=0;k<nQuadraturePoints_elementBoundary;k++)
        {
          if(isDOFBoundary[ebNE*nQuadraturePoints_elementBoundary+k] >= 1)
            {
              for(j=0;j<nDOF_trial_element;j++)
                {
                  Jacobian=0.0;
                  j_global = l2g[eN_global*nDOF_trial_element+j];
                  max_a=0.0;
                  for(I=0;I<nSpace;I++)
                    {
                      diffusiveVelocityComponent_I_Jacobian=0.0;
                      diffusiveVelocityComponent_I_Jacobian2=0.0;
                      for(J=0;J<nSpace;J++)
                        {
                          diffusiveVelocityComponent_I_Jacobian 
                            -= 
                            da[ebNE*nQuadraturePoints_elementBoundary*nSpace2+
                               k*nSpace2+
                               I*nSpace+
                               J]
                            *
                            grad_phi[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                                     k*nSpace+
                                     J];
                          diffusiveVelocityComponent_I_Jacobian2 
                            -= 
                            a[ebNE*nQuadraturePoints_elementBoundary*nSpace2+
                              k*nSpace2+
                              I*nSpace+
                              J]
                            *
                            grad_v[ebNE*nQuadraturePoints_elementBoundary*nDOF_trial_element*nSpace+
                                   k*nDOF_trial_element*nSpace+
                                   j*nSpace+
                                   J];
                          max_a = fmax(max_a,a[ebNE*nQuadraturePoints_elementBoundary*nSpace2+
                                               k*nSpace2+
                                               I*nSpace+
                                               J]);
                                       
                        }
                      Jacobian 
                        += 
                        (diffusiveVelocityComponent_I_Jacobian
                         *
                         v[ebNE*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                           k*nDOF_trial_element+
                           j] 
                         +
                         diffusiveVelocityComponent_I_Jacobian2*
                         dphi[j_global])
                        *
                        n[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                          k*nSpace+
                          I];
                    }
                  max_a = fmax(penalty_floor,max_a);
                  double penaltyJacobian = penalty[ebNE*nQuadraturePoints_elementBoundary+
                                                   k]
                    *
                    v[ebNE*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                      k*nDOF_trial_element+
                      j];
                  if (scale_penalty) penaltyJacobian *= max_a;
                  
                  Jacobian += penaltyJacobian;
                    
                  fluxJacobian[ebNE*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                               k*nDOF_trial_element+
                               j]
                    += Jacobian;
                }
            }
        }
    }
}
void updateGlobalExteriorNumericalDiffusiveFluxJacobian_sd(int scale_penalty,
                                                           double penalty_floor,
                                                           proteus_index_t nExteriorElementBoundaries_global,
							   int nQuadraturePoints_elementBoundary,
							   int nDOF_trial_element,
							   int nSpace,
							   proteus_csr_index_t* rowptr,
							   proteus_csr_index_t* colind,
							   proteus_index_t* l2g,
							   proteus_index_t* exteriorElementBoundaries,
							   proteus_index_t* elementBoundaryElements,
							   int* elementBoundaryLocalElementBoundaries,
							   int* isDOFBoundary,
							   double* n,
							   double* a,
							   double* da,
							   double* grad_phi,
							   double* dphi,
							   double* v,
							   double* grad_v,
							   double* penalty,
							   double* fluxJacobian)
{
  proteus_index_t ebN, eN_global, j_global; proteus_csr_index_t m; proteus_csr_index_t nnz=rowptr[nSpace]; int ebNE, k, j, I;
  double Jacobian,diffusiveVelocityComponent_I_Jacobian,diffusiveVelocityComponent_I_Jacobian2,max_a;
  for(ebNE=0;ebNE<nExteriorElementBoundaries_global;ebNE++)
    {
      ebN = exteriorElementBoundaries[ebNE];
      eN_global = elementBoundaryElements[ebN*2+0];
      for(k=0;k<nQuadraturePoints_elementBoundary;k++)
        {
          if(isDOFBoundary[ebNE*nQuadraturePoints_elementBoundary+k] >= 1)
            {
              for(j=0;j<nDOF_trial_element;j++)
                {
                  Jacobian=0.0;
                  j_global = l2g[eN_global*nDOF_trial_element+j];
                  max_a=0.0;
                  for(I=0;I<nSpace;I++)
                    {
                      diffusiveVelocityComponent_I_Jacobian=0.0;
                      diffusiveVelocityComponent_I_Jacobian2=0.0;
                      for(m=rowptr[I];m<rowptr[I+1];m++)
                        {
                          diffusiveVelocityComponent_I_Jacobian 
                            -= 
                            da[ebNE*nQuadraturePoints_elementBoundary*nnz+
                               k*nnz+
                               m]
                            *
                            grad_phi[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                                     k*nSpace+
                                     colind[m]];
                          diffusiveVelocityComponent_I_Jacobian2 
                            -= 
                            a[ebNE*nQuadraturePoints_elementBoundary*nnz+
                              k*nnz+
                              m]
                            *
                            grad_v[ebNE*nQuadraturePoints_elementBoundary*nDOF_trial_element*nSpace+
                                   k*nDOF_trial_element*nSpace+
                                   j*nSpace+
                                   colind[m]];
                          max_a = fmax(max_a,a[ebNE*nQuadraturePoints_elementBoundary*nnz+
                                               k*nnz+
                                               m]);
                                       
                        }
                      Jacobian 
                        += 
                        (diffusiveVelocityComponent_I_Jacobian
                         *
                         v[ebNE*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                           k*nDOF_trial_element+
                           j] 
                         +
                         diffusiveVelocityComponent_I_Jacobian2*
                         dphi[j_global])
                        *
                        n[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                          k*nSpace+
                          I];
                    }
                  max_a = fmax(penalty_floor,max_a);

                  double penaltyJacobian = penalty[ebNE*nQuadraturePoints_elementBoundary+
                                                   k]
                    *
                    v[ebNE*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                      k*nDOF_trial_element+
                      j];
                  if (scale_penalty) penaltyJacobian *= max_a;

                  Jacobian += penaltyJacobian;

                  fluxJacobian[ebNE*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                               k*nDOF_trial_element+
                               j]
                    += Jacobian;
                }
            }
        }
    }
}

/*
   \brief Update the diffusive flux Jacobian at exterior element boundary quadrature points
   
void updateExteriorNumericalDiffusiveFluxJacobian(proteus_index_t nExteriorElementBoundaries_global,
                                                  int nElementBoundaries_element,
                                                  int nQuadraturePoints_elementBoundary,
                                                  int nDOF_trial_element,
                                                  int nSpace,
                                                  proteus_index_t* l2g,
                                                  proteus_index_t* exteriorElementBoundaries,
                                                  proteus_index_t* elementBoundaryElements,
                                                  int* elementBoundaryLocalElementBoundaries,
                                                  int* isDOFBoundary,
                                                  double* n,
                                                  double* a,
                                                  double* da,
                                                  double* grad_phi,
                                                  double* dphi,
                                                  double* v,
                                                  double* grad_v,
                                                  double* penalty,
                                                  double* fluxJacobian)
{
  proteus_index_t ebN, eN_global, j_global; int ebNE, ebN_element, k, j, I, J, nSpace2=nSpace*nSpace;
  double Jacobian,diffusiveVelocityComponent_I_Jacobian,diffusiveVelocityComponent_I_Jacobian2;
  for(ebNE=0;ebNE<nExteriorElementBoundaries_global;ebNE++)
    {
      ebN = exteriorElementBoundaries[ebNE];
      eN_global = elementBoundaryElements[ebN*2+0];
      ebN_element = elementBoundaryLocalElementBoundaries[ebN*2+0];
      for(k=0;k<nQuadraturePoints_elementBoundary;k++)
        {
          if(isDOFBoundary[ebNE*nQuadraturePoints_elementBoundary+k] == 1)
            {
              for(j=0;j<nDOF_trial_element;j++)
                {
                  Jacobian=0.0;
                  j_global = l2g[eN_global*nDOF_trial_element+j];
                  for(I=0;I<nSpace;I++)
                    {
                      diffusiveVelocityComponent_I_Jacobian=0.0;
                      diffusiveVelocityComponent_I_Jacobian2=0.0;
                      for(J=0;J<nSpace;J++)
                        {
                          diffusiveVelocityComponent_I_Jacobian 
                            -= 
                            da[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace2+
                               ebN_element*nQuadraturePoints_elementBoundary*nSpace2+
                               k*nSpace2+
                               I*nSpace+
                               J]
                            *
                            grad_phi[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                                     ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                                     k*nSpace+
                                     J];
                          diffusiveVelocityComponent_I_Jacobian2 
                            -= 
                            a[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace2+
                              ebN_element*nQuadraturePoints_elementBoundary*nSpace2+
                              k*nSpace2+
                              I*nSpace+
                              J]
                            *
                            grad_v[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_trial_element*nSpace+
                                   ebN_element*nQuadraturePoints_elementBoundary*nDOF_trial_element*nSpace+
                                   k*nDOF_trial_element*nSpace+
                                   j*nSpace+
                                   J];
                        }
                      Jacobian 
                        += 
                        (diffusiveVelocityComponent_I_Jacobian
                         *
                         v[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                           ebN_element*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                           k*nDOF_trial_element+
                           j] 
                         +
                         diffusiveVelocityComponent_I_Jacobian2*
                         dphi[j_global])
                        *
                        n[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                          ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                          k*nSpace+
                          I];
                    }
                  Jacobian
                    +=
                    penalty[ebN*nQuadraturePoints_elementBoundary+
                            k]
                    *
                    v[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                      ebN_element*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                      k*nDOF_trial_element+
                      j];
                  fluxJacobian[ebN*2*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                               0*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                               k*nDOF_trial_element+
                               j]
                    += Jacobian;
                }
            }
        }
    }
}
void updateExteriorNumericalDiffusiveFluxJacobian_sd(proteus_index_t nExteriorElementBoundaries_global,
						     int nElementBoundaries_element,
						     int nQuadraturePoints_elementBoundary,
						     int nDOF_trial_element,
						     int nSpace,
						     proteus_csr_index_t* rowptr,
						     proteus_csr_index_t* colind,
						     proteus_index_t* l2g,
						     proteus_index_t* exteriorElementBoundaries,
						     proteus_index_t* elementBoundaryElements,
						     int* elementBoundaryLocalElementBoundaries,
						     int* isDOFBoundary,
						     double* n,
						     double* a,
						     double* da,
						     double* grad_phi,
						     double* dphi,
						     double* v,
						     double* grad_v,
						     double* penalty,
						     double* fluxJacobian)
{
  proteus_index_t ebN, eN_global, j_global, m; proteus_csr_index_t nnz=rowptr[nSpace]; int ebNE, ebN_element, k, j, I;
  double Jacobian,diffusiveVelocityComponent_I_Jacobian,diffusiveVelocityComponent_I_Jacobian2;
  for(ebNE=0;ebNE<nExteriorElementBoundaries_global;ebNE++)
    {
      ebN = exteriorElementBoundaries[ebNE];
      eN_global = elementBoundaryElements[ebN*2+0];
      ebN_element = elementBoundaryLocalElementBoundaries[ebN*2+0];
      for(k=0;k<nQuadraturePoints_elementBoundary;k++)
        {
          if(isDOFBoundary[ebNE*nQuadraturePoints_elementBoundary+k] == 1)
            {
              for(j=0;j<nDOF_trial_element;j++)
                {
                  Jacobian=0.0;
                  j_global = l2g[eN_global*nDOF_trial_element+j];
                  for(I=0;I<nSpace;I++)
                    {
                      diffusiveVelocityComponent_I_Jacobian=0.0;
                      diffusiveVelocityComponent_I_Jacobian2=0.0;
                      for(m=rowptr[I];m<rowptr[I+1];m++)
                        {
                          diffusiveVelocityComponent_I_Jacobian 
                            -= 
                            da[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nnz+
                               ebN_element*nQuadraturePoints_elementBoundary*nnz+
                               k*nnz+
                               m]
                            *
                            grad_phi[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                                     ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                                     k*nSpace+
                                     colind[m]];
                          diffusiveVelocityComponent_I_Jacobian2 
                            -= 
                            a[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nnz+
                              ebN_element*nQuadraturePoints_elementBoundary*nnz+
                              k*nnz+
                              m]
                            *
                            grad_v[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_trial_element*nSpace+
                                   ebN_element*nQuadraturePoints_elementBoundary*nDOF_trial_element*nSpace+
                                   k*nDOF_trial_element*nSpace+
                                   j*nSpace+
                                   colind[m]];
                        }
                      Jacobian 
                        += 
                        (diffusiveVelocityComponent_I_Jacobian
                         *
                         v[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                           ebN_element*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                           k*nDOF_trial_element+
                           j] 
                         +
                         diffusiveVelocityComponent_I_Jacobian2*
                         dphi[j_global])
                        *
                        n[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                          ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                          k*nSpace+
                          I];
                    }
                  Jacobian
                    +=
                    penalty[ebN*nQuadraturePoints_elementBoundary+
                            k]
                    *
                    v[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                      ebN_element*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                      k*nDOF_trial_element+
                      j];
                  fluxJacobian[ebN*2*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                               0*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                               k*nDOF_trial_element+
                               j]
                    += Jacobian;
                }
            }
        }
    }
} */
/*
   \brief Update the diffusive flux Jacobian at exterior element boundary quadrature points
   
void updateGlobalExteriorNumericalDiffusiveFluxJacobian(proteus_index_t nExteriorElementBoundaries_global,
							int nQuadraturePoints_elementBoundary,
							int nDOF_trial_element,
							int nSpace,
							proteus_index_t* l2g,
							proteus_index_t* exteriorElementBoundaries,
							proteus_index_t* elementBoundaryElements,
							int* elementBoundaryLocalElementBoundaries,
							int* isDOFBoundary,
							double* n,
							double* a,
							double* da,
							double* grad_phi,
							double* dphi,
							double* v,
							double* grad_v,
							double* penalty,
							double* fluxJacobian)
{
  proteus_index_t ebN, eN_global, j_global; int ebNE, k, j, I, J, nSpace2=nSpace*nSpace;
  double Jacobian,diffusiveVelocityComponent_I_Jacobian,diffusiveVelocityComponent_I_Jacobian2,max_a;
  for(ebNE=0;ebNE<nExteriorElementBoundaries_global;ebNE++)
    {
      ebN = exteriorElementBoundaries[ebNE];
      eN_global = elementBoundaryElements[ebN*2+0];
      for(k=0;k<nQuadraturePoints_elementBoundary;k++)
        {
          if(isDOFBoundary[ebNE*nQuadraturePoints_elementBoundary+k] >= 1)
            {
              for(j=0;j<nDOF_trial_element;j++)
                {
                  Jacobian=0.0;
                  j_global = l2g[eN_global*nDOF_trial_element+j];
                  max_a=0.0;
                  for(I=0;I<nSpace;I++)
                    {
                      diffusiveVelocityComponent_I_Jacobian=0.0;
                      diffusiveVelocityComponent_I_Jacobian2=0.0;
                      for(J=0;J<nSpace;J++)
                        {
                          diffusiveVelocityComponent_I_Jacobian 
                            -= 
                            da[ebNE*nQuadraturePoints_elementBoundary*nSpace2+
                               k*nSpace2+
                               I*nSpace+
                               J]
                            *
                            grad_phi[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                                     k*nSpace+
                                     J];
                          diffusiveVelocityComponent_I_Jacobian2 
                            -= 
                            a[ebNE*nQuadraturePoints_elementBoundary*nSpace2+
                              k*nSpace2+
                              I*nSpace+
                              J]
                            *
                            grad_v[ebNE*nQuadraturePoints_elementBoundary*nDOF_trial_element*nSpace+
                                   k*nDOF_trial_element*nSpace+
                                   j*nSpace+
                                   J];
                          max_a = fmax(max_a,a[ebNE*nQuadraturePoints_elementBoundary*nSpace2+
                                               k*nSpace2+
                                               I*nSpace+
                                               J]);
                                       
                        }
                      Jacobian 
                        += 
                        (diffusiveVelocityComponent_I_Jacobian
                         *
                         v[ebNE*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                           k*nDOF_trial_element+
                           j] 
                         +
                         diffusiveVelocityComponent_I_Jacobian2*
                         dphi[j_global])
                        *
                        n[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                          k*nSpace+
                          I];
                    }
                  Jacobian
                    +=
                    penalty[ebNE*nQuadraturePoints_elementBoundary+
                                  k]
                    *
                    v[ebNE*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                      k*nDOF_trial_element+
                      j];
                  fluxJacobian[ebNE*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                               k*nDOF_trial_element+
                               j]
                    += Jacobian;
                }
            }
        }
    }
}
void updateGlobalExteriorNumericalDiffusiveFluxJacobian_sd(proteus_index_t nExteriorElementBoundaries_global,
							   int nQuadraturePoints_elementBoundary,
							   int nDOF_trial_element,
							   int nSpace,
							   proteus_csr_index_t* rowptr,
							   proteus_csr_index_t* colind,
							   proteus_index_t* l2g,
							   proteus_index_t* exteriorElementBoundaries,
							   proteus_index_t* elementBoundaryElements,
							   int* elementBoundaryLocalElementBoundaries,
							   int* isDOFBoundary,
							   double* n,
							   double* a,
							   double* da,
							   double* grad_phi,
							   double* dphi,
							   double* v,
							   double* grad_v,
							   double* penalty,
							   double* fluxJacobian)
{
  proteus_index_t ebN, eN_global, j_global, m; proteus_csr_index_t nnz=rowptr[nSpace]; int ebNE, k, j, I;
  double Jacobian,diffusiveVelocityComponent_I_Jacobian,diffusiveVelocityComponent_I_Jacobian2,max_a;
  for(ebNE=0;ebNE<nExteriorElementBoundaries_global;ebNE++)
    {
      ebN = exteriorElementBoundaries[ebNE];
      eN_global = elementBoundaryElements[ebN*2+0];
      for(k=0;k<nQuadraturePoints_elementBoundary;k++)
        {
          if(isDOFBoundary[ebNE*nQuadraturePoints_elementBoundary+k] >= 1)
            {
              for(j=0;j<nDOF_trial_element;j++)
                {
                  Jacobian=0.0;
                  j_global = l2g[eN_global*nDOF_trial_element+j];
                  max_a=0.0;
                  for(I=0;I<nSpace;I++)
                    {
                      diffusiveVelocityComponent_I_Jacobian=0.0;
                      diffusiveVelocityComponent_I_Jacobian2=0.0;
                      for(m=rowptr[I];m<rowptr[I+1];m++)
                        {
                          diffusiveVelocityComponent_I_Jacobian 
                            -= 
                            da[ebNE*nQuadraturePoints_elementBoundary*nnz+
                               k*nnz+
                               m]
                            *
                            grad_phi[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                                     k*nSpace+
                                     colind[m]];
                          diffusiveVelocityComponent_I_Jacobian2 
                            -= 
                            a[ebNE*nQuadraturePoints_elementBoundary*nnz+
                              k*nnz+
                              m]
                            *
                            grad_v[ebNE*nQuadraturePoints_elementBoundary*nDOF_trial_element*nSpace+
                                   k*nDOF_trial_element*nSpace+
                                   j*nSpace+
                                   colind[m]];
                          max_a = fmax(max_a,a[ebNE*nQuadraturePoints_elementBoundary*nnz+
                                               k*nnz+
                                               I*nSpace+
                                               m]);
                                       
                        }
                      Jacobian 
                        += 
                        (diffusiveVelocityComponent_I_Jacobian
                         *
                         v[ebNE*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                           k*nDOF_trial_element+
                           j] 
                         +
                         diffusiveVelocityComponent_I_Jacobian2*
                         dphi[j_global])
                        *
                        n[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                          k*nSpace+
                          I];
                    }
                  Jacobian
                    +=
                    penalty[ebNE*nQuadraturePoints_elementBoundary+
                                  k]
                    *
                    v[ebNE*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                      k*nDOF_trial_element+
                      j];
                  fluxJacobian[ebNE*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                               k*nDOF_trial_element+
                               j]
                    += Jacobian;
                }
            }
        }
    }
}*/

void updateExteriorNumericalDiffusiveFluxJacobian_free(proteus_index_t nExteriorElementBoundaries_global,
                                                  int nElementBoundaries_element,
                                                  int nQuadraturePoints_elementBoundary,
                                                  int nDOF_trial_element,
                                                  int nSpace,
                                                  proteus_index_t* l2g,
                                                  proteus_index_t* exteriorElementBoundaries,
                                                  proteus_index_t* elementBoundaryElements,
                                                  int* elementBoundaryLocalElementBoundaries,
                                                  int* isDOFBoundary,
                                                  double* n,
                                                  double* a,
                                                  double* da,
                                                  double* grad_phi,
                                                  double* dphi,
                                                  double* v,
                                                  double* grad_v,
                                                  double* penalty,
                                                  double* fluxJacobian)
{
  proteus_index_t ebN, eN_global, j_global; int ebNE, ebN_element, k, j, I, J, nSpace2=nSpace*nSpace;
  double diffusiveVelocityComponent_I_Jacobian,diffusiveVelocityComponent_I_Jacobian2;
  for(ebNE=0;ebNE<nExteriorElementBoundaries_global;ebNE++)
    {
      ebN = exteriorElementBoundaries[ebNE];
      eN_global = elementBoundaryElements[ebN*2+0];
      ebN_element = elementBoundaryLocalElementBoundaries[ebN*2+0];
      for(k=0;k<nQuadraturePoints_elementBoundary;k++)
        {
          for(j=0;j<nDOF_trial_element;j++)
            {
              j_global = l2g[eN_global*nDOF_trial_element+j];
              for(I=0;I<nSpace;I++)
                {
                  diffusiveVelocityComponent_I_Jacobian=0.0;
                  diffusiveVelocityComponent_I_Jacobian2=0.0;
                  for(J=0;J<nSpace;J++)
                    {
                      diffusiveVelocityComponent_I_Jacobian 
                        -= 
                        da[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace2+
                           ebN_element*nQuadraturePoints_elementBoundary*nSpace2+
                           k*nSpace2+
                           I*nSpace+
                           J]
                        *
                        grad_phi[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                                 ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                                 k*nSpace+
                                 J];
                      diffusiveVelocityComponent_I_Jacobian2 
                        -= 
                        a[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace2+
                          ebN_element*nQuadraturePoints_elementBoundary*nSpace2+
                          k*nSpace2+
                          I*nSpace+
                          J]
                        *
                        grad_v[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_trial_element*nSpace+
                               ebN_element*nQuadraturePoints_elementBoundary*nDOF_trial_element*nSpace+
                               k*nDOF_trial_element*nSpace+
                               j*nSpace+
                               J];
                    }
                  fluxJacobian[ebN*2*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                               0*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                               k*nDOF_trial_element+
                               j]+= 
                    (diffusiveVelocityComponent_I_Jacobian
                     *
                     v[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                       ebN_element*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                       k*nDOF_trial_element+
                       j] 
                     +
                     diffusiveVelocityComponent_I_Jacobian2*
                     dphi[j_global])
                    *
                    n[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                      ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                      k*nSpace+
                      I];
                }
            }
        }
    }
}
void updateExteriorNumericalDiffusiveFluxJacobian_free_sd(proteus_index_t nExteriorElementBoundaries_global,
							  int nElementBoundaries_element,
							  int nQuadraturePoints_elementBoundary,
							  int nDOF_trial_element,
							  int nSpace,
							  proteus_csr_index_t* rowptr,
							  proteus_csr_index_t* colind,
							  proteus_index_t* l2g,
							  proteus_index_t* exteriorElementBoundaries,
							  proteus_index_t* elementBoundaryElements,
							  int* elementBoundaryLocalElementBoundaries,
							  int* isDOFBoundary,
							  double* n,
							  double* a,
							  double* da,
							  double* grad_phi,
							  double* dphi,
							  double* v,
							  double* grad_v,
							  double* penalty,
							  double* fluxJacobian)
{
  proteus_index_t ebN, eN_global, j_global; proteus_csr_index_t m; proteus_csr_index_t nnz=rowptr[nSpace]; int ebNE, ebN_element, k, j, I;
  double diffusiveVelocityComponent_I_Jacobian,diffusiveVelocityComponent_I_Jacobian2;
  for(ebNE=0;ebNE<nExteriorElementBoundaries_global;ebNE++)
    {
      ebN = exteriorElementBoundaries[ebNE];
      eN_global = elementBoundaryElements[ebN*2+0];
      ebN_element = elementBoundaryLocalElementBoundaries[ebN*2+0];
      for(k=0;k<nQuadraturePoints_elementBoundary;k++)
        {
          for(j=0;j<nDOF_trial_element;j++)
            {
              j_global = l2g[eN_global*nDOF_trial_element+j];
              for(I=0;I<nSpace;I++)
                {
                  diffusiveVelocityComponent_I_Jacobian=0.0;
                  diffusiveVelocityComponent_I_Jacobian2=0.0;
                  for(m=rowptr[I];m<rowptr[I+1];m++)
                    {
                      diffusiveVelocityComponent_I_Jacobian 
                        -= 
                        da[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nnz+
                           ebN_element*nQuadraturePoints_elementBoundary*nnz+
                           k*nnz+
                           m]
                        *
                        grad_phi[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                                 ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                                 k*nSpace+
                                 colind[m]];
                      diffusiveVelocityComponent_I_Jacobian2 
                        -= 
                        a[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nnz+
                          ebN_element*nQuadraturePoints_elementBoundary*nnz+
                          k*nnz+
                          m]
                        *
                        grad_v[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_trial_element*nSpace+
                               ebN_element*nQuadraturePoints_elementBoundary*nDOF_trial_element*nSpace+
                               k*nDOF_trial_element*nSpace+
                               j*nSpace+
                               colind[m]];
                    }
                  fluxJacobian[ebN*2*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                               0*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                               k*nDOF_trial_element+
                               j]+= 
                    (diffusiveVelocityComponent_I_Jacobian
                     *
                     v[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                       ebN_element*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                       k*nDOF_trial_element+
                       j] 
                     +
                     diffusiveVelocityComponent_I_Jacobian2*
                     dphi[j_global])
                    *
                    n[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                      ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                      k*nSpace+
                      I];
                }
            }
        }
    }
}

void updateGlobalExteriorNumericalDiffusiveFluxJacobian_free(proteus_index_t nExteriorElementBoundaries_global,
							     int nQuadraturePoints_elementBoundary,
							     int nDOF_trial_element,
							     int nSpace,
							     proteus_index_t* l2g,
							     proteus_index_t* exteriorElementBoundaries,
							     proteus_index_t* elementBoundaryElements,
							     int* elementBoundaryLocalElementBoundaries,
							     int* isDOFBoundary,
							     double* n,
							     double* a,
							     double* da,
							     double* grad_phi,
							     double* dphi,
							     double* v,
							     double* grad_v,
							     double* penalty,
							     double* fluxJacobian)
{
  proteus_index_t ebN, eN_global, j_global; int ebNE, k, j, I, J, nSpace2=nSpace*nSpace;
  double diffusiveVelocityComponent_I_Jacobian,diffusiveVelocityComponent_I_Jacobian2;
  for(ebNE=0;ebNE<nExteriorElementBoundaries_global;ebNE++)
    {
      ebN = exteriorElementBoundaries[ebNE];
      eN_global = elementBoundaryElements[ebN*2+0];
      for(k=0;k<nQuadraturePoints_elementBoundary;k++)
        {
          for(j=0;j<nDOF_trial_element;j++)
            {
              j_global = l2g[eN_global*nDOF_trial_element+j];
              for(I=0;I<nSpace;I++)
                {
                  diffusiveVelocityComponent_I_Jacobian=0.0;
                  diffusiveVelocityComponent_I_Jacobian2=0.0;
                  for(J=0;J<nSpace;J++)
                    {
                      diffusiveVelocityComponent_I_Jacobian 
                        -= 
                        da[ebNE*nQuadraturePoints_elementBoundary*nSpace2+
                           k*nSpace2+
                           I*nSpace+
                           J]
                        *
                        grad_phi[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                                 k*nSpace+
                                 J];
                      diffusiveVelocityComponent_I_Jacobian2 
                        -= 
                        a[ebNE*nQuadraturePoints_elementBoundary*nSpace2+
                          k*nSpace2+
                          I*nSpace+
                          J]
                        *
                        grad_v[ebNE*nQuadraturePoints_elementBoundary*nDOF_trial_element*nSpace+
                               k*nDOF_trial_element*nSpace+
                               j*nSpace+
                               J];
                    }
                  fluxJacobian[ebNE*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                               k*nDOF_trial_element+
                               j]+= 
                    (diffusiveVelocityComponent_I_Jacobian
                     *
                     v[ebNE*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                       k*nDOF_trial_element+
                       j] 
                     +
                     diffusiveVelocityComponent_I_Jacobian2*
                     dphi[j_global])
                    *
                    n[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                      k*nSpace+
                      I];
                }
            }
        }
    }
}
void updateGlobalExteriorNumericalDiffusiveFluxJacobian_free_sd(proteus_index_t nExteriorElementBoundaries_global,
								int nQuadraturePoints_elementBoundary,
								int nDOF_trial_element,
								int nSpace,
								proteus_csr_index_t* rowptr,
								proteus_csr_index_t* colind,
								proteus_index_t* l2g,
								proteus_index_t* exteriorElementBoundaries,
								proteus_index_t* elementBoundaryElements,
								int* elementBoundaryLocalElementBoundaries,
								int* isDOFBoundary,
								double* n,
								double* a,
								double* da,
								double* grad_phi,
								double* dphi,
								double* v,
								double* grad_v,
								double* penalty,
								double* fluxJacobian)
{
  proteus_index_t ebN, eN_global, j_global; proteus_csr_index_t m; proteus_csr_index_t nnz=rowptr[nSpace]; int ebNE, k, j, I;
  double diffusiveVelocityComponent_I_Jacobian,diffusiveVelocityComponent_I_Jacobian2;
  for(ebNE=0;ebNE<nExteriorElementBoundaries_global;ebNE++)
    {
      ebN = exteriorElementBoundaries[ebNE];
      eN_global = elementBoundaryElements[ebN*2+0];
      for(k=0;k<nQuadraturePoints_elementBoundary;k++)
        {
          for(j=0;j<nDOF_trial_element;j++)
            {
              j_global = l2g[eN_global*nDOF_trial_element+j];
              for(I=0;I<nSpace;I++)
                {
                  diffusiveVelocityComponent_I_Jacobian=0.0;
                  diffusiveVelocityComponent_I_Jacobian2=0.0;
                  for(m=rowptr[I];m<rowptr[I+1];m++)
                    {
                      diffusiveVelocityComponent_I_Jacobian 
                        -= 
                        da[ebNE*nQuadraturePoints_elementBoundary*nnz+
                           k*nnz+
                           m]
                        *
                        grad_phi[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                                 k*nSpace+
                                 colind[m]];
                      diffusiveVelocityComponent_I_Jacobian2 
                        -= 
                        a[ebNE*nQuadraturePoints_elementBoundary*nnz+
                          k*nnz+
                          m]
                        *
                        grad_v[ebNE*nQuadraturePoints_elementBoundary*nDOF_trial_element*nSpace+
                               k*nDOF_trial_element*nSpace+
                               j*nSpace+
                               colind[m]];
                    }
                  fluxJacobian[ebNE*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                               k*nDOF_trial_element+
                               j]+= 
                    (diffusiveVelocityComponent_I_Jacobian
                     *
                     v[ebNE*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                       k*nDOF_trial_element+
                       j] 
                     +
                     diffusiveVelocityComponent_I_Jacobian2*
                     dphi[j_global])
                    *
                    n[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                      k*nSpace+
                      I];
                }
            }
        }
    }
}
/**
   \brief Calculate the advective flux at at interior element boundaries
*/
void calculateInteriorNumericalAdvectiveFlux(proteus_index_t nInteriorElementBoundaries_global,
                                             int nElementBoundaries_element,
                                             int nQuadraturePoints_elementBoundary,
                                             int nSpace,
                                             proteus_index_t* interiorElementBoundaries,
                                             proteus_index_t* elementBoundaryElements,
                                             int* elementBoundaryLocalElementBoundaries,
                                             double* n,
                                             double* u,
                                             double* f,
                                             double* df,
                                             double* flux,
                                             double* dflux_left,
                                             double* dflux_right)
{
  proteus_index_t ebN, left_eN_global, right_eN_global; int ebNI, left_ebN_element, right_ebN_element, k, J;
  double left_speed,right_speed,left_flux,right_flux,shock_speed,flux_jump,u_jump;
  for(ebNI=0;ebNI<nInteriorElementBoundaries_global;ebNI++)
    {
      ebN = interiorElementBoundaries[ebNI];
      left_eN_global = elementBoundaryElements[ebN*2+0];
      right_eN_global = elementBoundaryElements[ebN*2+1];
      left_ebN_element = elementBoundaryLocalElementBoundaries[ebN*2+0];
      right_ebN_element = elementBoundaryLocalElementBoundaries[ebN*2+1];
      for(k=0;k<nQuadraturePoints_elementBoundary;k++)
        {
          left_speed=0.0;
          right_speed=0.0;
          left_flux=0.0;
          right_flux=0.0;
	  /*mwf add default shock speed is zero*/
	  shock_speed=0.0;
          for(J=0;J<nSpace;J++)
            {
              left_speed 
                += 
                n[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                  left_ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                  k*nSpace+
		  J]
                *
                df[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                   left_ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                   k*nSpace+
		   J];
              right_speed 
                += 
                n[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                  left_ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                  k*nSpace+
		  J]
                *
                df[right_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                   right_ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                   k*nSpace+
		   J];
             left_flux 
                += 
                n[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                  left_ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                  k*nSpace+
		  J]
                *
                f[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                   left_ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                   k*nSpace+
		   J];
             right_flux 
               += 
               n[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                 left_ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                 k*nSpace+
                 J]
               *
               f[right_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                 right_ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                 k*nSpace+
                 J];
            }
          flux_jump = (right_flux - left_flux);
          u_jump = (u[right_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary+
                 right_ebN_element*nQuadraturePoints_elementBoundary+
                 k]
               -
               u[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary+
                 left_ebN_element*nQuadraturePoints_elementBoundary+
                 k]);
          if (fabs(u_jump) > fabs(flux_jump*1.0e-16))
            shock_speed = flux_jump/u_jump;
          if (left_speed >= 0.0 && right_speed >= 0.0)
            {
              flux[ebN*nQuadraturePoints_elementBoundary+
                   k] = left_flux;
              dflux_left[ebN*nQuadraturePoints_elementBoundary+
                         k] = left_speed;
              dflux_right[ebN*nQuadraturePoints_elementBoundary+
                          k] = 0.0;
            }
          else if (left_speed <= 0.0 && right_speed <= 0.0)
            {
              flux[ebN*nQuadraturePoints_elementBoundary+
                   k] = right_flux;
              dflux_left[ebN*nQuadraturePoints_elementBoundary+
                         k] = 0.0;
              dflux_right[ebN*nQuadraturePoints_elementBoundary+
                          k] = right_speed;
            }
          else if (left_speed >= 0.0 && right_speed <= 0.0)
            {
              if (shock_speed >= 0.0)
                {
                  flux[ebN*nQuadraturePoints_elementBoundary+
                       k] = left_flux;
                  dflux_left[ebN*nQuadraturePoints_elementBoundary+
                             k] = left_speed;
                  dflux_right[ebN*nQuadraturePoints_elementBoundary+
                              k] = 0.0;
                }
              else
                {
                  flux[ebN*nQuadraturePoints_elementBoundary+
                       k] = right_flux;
                  dflux_left[ebN*nQuadraturePoints_elementBoundary+
                             k] = 0.0;
                  dflux_right[ebN*nQuadraturePoints_elementBoundary+
                              k] = right_speed;
                }
            }
          else
            {
              /*transonic rarefaction*/
              printf("Transonic rarefaction detected in interior. This numerical flux treats transonic rarefactions incorrectly. left_speed= %12.5e right_speed= %12.5e \n",left_speed,right_speed);
	      for (J=0; J < nSpace; J++)
		{
		  printf("n_l[%d]=%g df_l[%d]=%g df_r[%d]=%g \n",J,
			 n[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
			   left_ebN_element*nQuadraturePoints_elementBoundary*nSpace+
			   k*nSpace+
			   J],
			 J,
			 df[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
			    left_ebN_element*nQuadraturePoints_elementBoundary*nSpace+
			    k*nSpace+
			    J],
			 J,
			 df[right_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
			    right_ebN_element*nQuadraturePoints_elementBoundary*nSpace+
			    k*nSpace+
			    J]);
			 
		}
              flux[ebN*nQuadraturePoints_elementBoundary+
                   k] = 0.5*(left_flux + right_flux);
              dflux_left[ebN*nQuadraturePoints_elementBoundary+
                         k] = 0.5*left_speed;
              dflux_right[ebN*nQuadraturePoints_elementBoundary+
                         k] = 0.5*right_speed;
            }
        }
    }
}

/**
   \brief Calculate the advective flux at at interior element boundaries
*/
void updateInteriorNumericalAdvectiveFluxJacobian(proteus_index_t nInteriorElementBoundaries_global,
                                                  int nElementBoundaries_element,
                                                  int nQuadraturePoints_elementBoundary,
                                                  int nDOF_trial_element,
                                                  proteus_index_t* interiorElementBoundaries,
                                                  proteus_index_t* elementBoundaryElements,
                                                  int* elementBoundaryLocalElementBoundaries,
                                                  double* dflux_left,
                                                  double* dflux_right,
                                                  double* v,
                                                  double* fluxJacobian)
{
  proteus_index_t ebN, left_eN_global, right_eN_global; int ebNI, left_ebN_element, right_ebN_element, k, j;
  for(ebNI=0;ebNI<nInteriorElementBoundaries_global;ebNI++)
    {
      ebN = interiorElementBoundaries[ebNI];
      left_eN_global = elementBoundaryElements[ebN*2+0];
      right_eN_global = elementBoundaryElements[ebN*2+1];
      left_ebN_element = elementBoundaryLocalElementBoundaries[ebN*2+0];
      right_ebN_element = elementBoundaryLocalElementBoundaries[ebN*2+1];
      for(k=0;k<nQuadraturePoints_elementBoundary;k++)
        for(j=0;j<nDOF_trial_element;j++)
          {
            fluxJacobian[ebN*2*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                         0*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                         k*nDOF_trial_element+
                         j] 
              += 
              dflux_left[ebN*nQuadraturePoints_elementBoundary+
                         k]
              *
              v[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                left_ebN_element*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                k*nDOF_trial_element+
                j];
            fluxJacobian[ebN*2*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                         1*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                         k*nDOF_trial_element+
                         j] 
              += 
              dflux_right[ebN*nQuadraturePoints_elementBoundary+
                          k]
              *
              v[right_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                right_ebN_element*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                k*nDOF_trial_element+
                j];
          }
    }
}
/**
   \brief Calculate the two-sided flux jacobian at at interior element boundaries
*/
void updateInteriorTwoSidedNumericalFluxJacobian(proteus_index_t nInteriorElementBoundaries_global,
						 int nElementBoundaries_element,
						 int nQuadraturePoints_elementBoundary,
						 int nDOF_trial_element,
						 proteus_index_t* interiorElementBoundaries,
						 proteus_index_t* elementBoundaryElements,
						 int* elementBoundaryLocalElementBoundaries,
						 double* dflux_left,
						 double* dflux_right,
						 double* v,
						 double* fluxJacobian_2sided)
{
  proteus_index_t ebN, left_eN_global, right_eN_global; int ebNI, left_ebN_element, right_ebN_element, k, j;
  for(ebNI=0;ebNI<nInteriorElementBoundaries_global;ebNI++)
    {
      ebN = interiorElementBoundaries[ebNI];
      left_eN_global = elementBoundaryElements[ebN*2+0];
      right_eN_global = elementBoundaryElements[ebN*2+1];
      left_ebN_element = elementBoundaryLocalElementBoundaries[ebN*2+0];
      right_ebN_element = elementBoundaryLocalElementBoundaries[ebN*2+1];
      for(k=0;k<nQuadraturePoints_elementBoundary;k++)
        for(j=0;j<nDOF_trial_element;j++)
          {
	    /*left neighbour flux first*/
            fluxJacobian_2sided[ebN*2*2*nQuadraturePoints_elementBoundary*nDOF_trial_element+
				0*2*nQuadraturePoints_elementBoundary*nDOF_trial_element+/*left flux*/
				0*nQuadraturePoints_elementBoundary*nDOF_trial_element+/*left neig. dep*/
				k*nDOF_trial_element+
				j] 
              += 
              dflux_left[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary+
			 left_ebN_element*nQuadraturePoints_elementBoundary+
                         k]
              *
              v[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                left_ebN_element*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                k*nDOF_trial_element+
                j];
            fluxJacobian_2sided[ebN*2*2*nQuadraturePoints_elementBoundary*nDOF_trial_element+
				0*2*nQuadraturePoints_elementBoundary*nDOF_trial_element+/*left flux*/
				1*nQuadraturePoints_elementBoundary*nDOF_trial_element+/*right neig. dep*/
				k*nDOF_trial_element+
				j] 
              += 
              dflux_right[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary+
			  left_ebN_element*nQuadraturePoints_elementBoundary+
                          k]
              *
              v[right_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                right_ebN_element*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                k*nDOF_trial_element+
                j];
	    /*right neighbour flux*/
            fluxJacobian_2sided[ebN*2*2*nQuadraturePoints_elementBoundary*nDOF_trial_element+
				1*2*nQuadraturePoints_elementBoundary*nDOF_trial_element+/*right flux*/
				0*nQuadraturePoints_elementBoundary*nDOF_trial_element+/*left neig. dep*/
				k*nDOF_trial_element+
				j] 
              += 
              dflux_left[right_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary+
			 right_ebN_element*nQuadraturePoints_elementBoundary+
                         k]
              *
              v[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                left_ebN_element*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                k*nDOF_trial_element+
                j];
            fluxJacobian_2sided[ebN*2*2*nQuadraturePoints_elementBoundary*nDOF_trial_element+
				1*2*nQuadraturePoints_elementBoundary*nDOF_trial_element+/*right flux*/
				1*nQuadraturePoints_elementBoundary*nDOF_trial_element+/*right neig. dep*/
				k*nDOF_trial_element+
				j] 
              += 
              dflux_right[right_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary+
			  right_ebN_element*nQuadraturePoints_elementBoundary+
                          k]
              *
              v[right_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                right_ebN_element*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                k*nDOF_trial_element+
                j];
          }
    }
}

/**
   \brief Calculate the advective flux at at interior element boundaries
*/
void calculateInteriorNumericalAdvectiveFlux_average(proteus_index_t nInteriorElementBoundaries_global,
                                             int nElementBoundaries_element,
                                             int nQuadraturePoints_elementBoundary,
                                             int nSpace,
                                             proteus_index_t* interiorElementBoundaries,
                                             proteus_index_t* elementBoundaryElements,
                                             int* elementBoundaryLocalElementBoundaries,
                                             double* n,
                                             double* u,
                                             double* f,
                                             double* df,
                                             double* flux,
                                             double* dflux_left,
                                             double* dflux_right)
{
  proteus_index_t ebN, left_eN_global, right_eN_global; int ebNI, left_ebN_element, right_ebN_element, k, J;
  for(ebNI=0;ebNI<nInteriorElementBoundaries_global;ebNI++)
    {
      ebN = interiorElementBoundaries[ebNI];
      left_eN_global = elementBoundaryElements[ebN*2+0];
      right_eN_global = elementBoundaryElements[ebN*2+1];
      left_ebN_element = elementBoundaryLocalElementBoundaries[ebN*2+0];
      right_ebN_element = elementBoundaryLocalElementBoundaries[ebN*2+1];
      for(k=0;k<nQuadraturePoints_elementBoundary;k++)
        {
          flux[ebN*nQuadraturePoints_elementBoundary+
               k] =0.0;
          dflux_left[ebN*nQuadraturePoints_elementBoundary+
                     k] = 0.0;
          dflux_right[ebN*nQuadraturePoints_elementBoundary+
                      k] = 0.0;
          for(J=0;J<nSpace;J++)
            {
              flux[ebN*nQuadraturePoints_elementBoundary+
                   k] += n[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                           left_ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                           k*nSpace+
                           J]
                *
                (f[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                   left_ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                   k*nSpace+
                   J]
                 +
                 f[right_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                   right_ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                   k*nSpace+
                   J]);
              dflux_left[ebN*nQuadraturePoints_elementBoundary+
                         k] += n[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                                 left_ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                                 k*nSpace+
                                 J]
                *
                df[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                   left_ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                   k*nSpace+
                   J];
              dflux_right[ebN*nQuadraturePoints_elementBoundary+
                          k] += n[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                                  left_ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                                  k*nSpace+
                                  J]
                *
                df[right_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                   right_ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                   k*nSpace+
                   J];
            }
          flux[ebN*nQuadraturePoints_elementBoundary+
               k] *= 0.5;
          dflux_left[ebN*nQuadraturePoints_elementBoundary+
                     k] *= 0.5;
          dflux_right[ebN*nQuadraturePoints_elementBoundary+
                      k] *= 0.5;
        }
    }
}

/**
   \brief Update the advective flux at exterior element boundaries.
*/
void calculateExteriorNumericalAdvectiveFlux_NoBC(proteus_index_t nExteriorElementBoundaries_global,
                                                  int nElementBoundaries_element,
                                                  int nQuadraturePoints_elementBoundary,
                                                  int nSpace,
                                                  proteus_index_t* exteriorElementBoundaries,
                                                  proteus_index_t* elementBoundaryElements,
                                                  int* elementBoundaryLocalElementBoundaries,
                                                  int* inflowFlag,
                                                  double* n,
                                                  double* f,
                                                  double* df,
                                                  double* flux,
                                                  double* dflux_left)
{
  proteus_index_t ebN, eN_global; int ebNE, ebN_element, k, J;
  memset(inflowFlag,0,sizeof(int)*nExteriorElementBoundaries_global*nQuadraturePoints_elementBoundary);
  for(ebNE=0;ebNE<nExteriorElementBoundaries_global;ebNE++)
    {
      ebN = exteriorElementBoundaries[ebNE];
      eN_global = elementBoundaryElements[ebN*2+0];
      ebN_element = elementBoundaryLocalElementBoundaries[ebN*2+0];
      for(k=0;k<nQuadraturePoints_elementBoundary;k++)
        {
          flux[ebN*nQuadraturePoints_elementBoundary+k] = 0.0;
          dflux_left[ebN*nQuadraturePoints_elementBoundary+k] = 0.0;
          inflowFlag[ebNE*nQuadraturePoints_elementBoundary+k]=0;
          for(J=0;J<nSpace;J++)
            {
              flux[ebN*nQuadraturePoints_elementBoundary+k]
                +=
                n[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                  ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                  k*nSpace+
                  J]
                *
                f[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                  ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                  k*nSpace+
                  J];
              dflux_left[ebN*nQuadraturePoints_elementBoundary+k]
                +=
                n[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                  ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                  k*nSpace+
                  J]
                *
                df[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                  ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                  k*nSpace+
                  J];
            }
          if(dflux_left[ebN*nQuadraturePoints_elementBoundary+k] < 0.0)
            {
              /* cek debug, setting inflow flow to zero */
              inflowFlag[ebNE*nQuadraturePoints_elementBoundary+k] = 1;
              flux[ebN*nQuadraturePoints_elementBoundary+k] = 0.0;
              dflux_left[ebN*nQuadraturePoints_elementBoundary+k] = 0.0;
            }
        }
    }
}
/**
   \brief Update the advective flux at exterior element boundaries.
*/
void calculateGlobalExteriorNumericalAdvectiveFlux_NoBC(proteus_index_t nExteriorElementBoundaries_global,
							int nQuadraturePoints_elementBoundary,
							int nSpace,
							proteus_index_t* exteriorElementBoundaries,
							proteus_index_t* elementBoundaryElements,
							int* elementBoundaryLocalElementBoundaries,
							int* inflowFlag,
							double* n,
							double* f,
							double* df,
							double* flux,
							double* dflux_left)
{
  proteus_index_t ebN, eN_global; int ebNE, ebN_element, k, J;
  memset(inflowFlag,0,sizeof(int)*nExteriorElementBoundaries_global*nQuadraturePoints_elementBoundary);
  for(ebNE=0;ebNE<nExteriorElementBoundaries_global;ebNE++)
    {
      ebN = exteriorElementBoundaries[ebNE];
      eN_global = elementBoundaryElements[ebN*2+0];
      ebN_element = elementBoundaryLocalElementBoundaries[ebN*2+0];
      for(k=0;k<nQuadraturePoints_elementBoundary;k++)
        {
          flux[ebNE*nQuadraturePoints_elementBoundary+k] = 0.0;
          dflux_left[ebNE*nQuadraturePoints_elementBoundary+k] = 0.0;
          inflowFlag[ebNE*nQuadraturePoints_elementBoundary+k]=0;
          for(J=0;J<nSpace;J++)
            {
              flux[ebNE*nQuadraturePoints_elementBoundary+k]
                +=
                n[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                  k*nSpace+
                  J]
                *
                f[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                  k*nSpace+
                  J];
              dflux_left[ebNE*nQuadraturePoints_elementBoundary+k]
                +=
                n[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                  k*nSpace+
                  J]
                *
                df[ebNE*nQuadraturePoints_elementBoundary*nSpace+
		   k*nSpace+
		   J];
            }
          if(dflux_left[ebNE*nQuadraturePoints_elementBoundary+k] < 0.0)
            {
              /* cek debug, setting inflow flow to zero */
              inflowFlag[ebNE*nQuadraturePoints_elementBoundary+k] = 1;
              flux[ebNE*nQuadraturePoints_elementBoundary+k] = 0.0;
              dflux_left[ebNE*nQuadraturePoints_elementBoundary+k] = 0.0;
            }
        }
    }
}

/**
   \brief Calculate the advective flux at at exterior element boundaries
*/
void calculateExteriorNumericalAdvectiveFlux(proteus_index_t nExteriorElementBoundaries_global,
                                             int nElementBoundaries_element,
                                             int nQuadraturePoints_elementBoundary,
                                             int nSpace,
                                             proteus_index_t* exteriorElementBoundaries,
                                             proteus_index_t* elementBoundaryElements,
                                             int* elementBoundaryLocalElementBoundaries,
                                             int *isDOFBoundary,
                                             int *inflowFlag,
                                             double* n,
                                             double* bc_u,
                                             double* bc_f,
                                             double* bc_df,
                                             double* u,
                                             double* f,
                                             double* df,
                                             double* flux,
                                             double* dflux)
{
  proteus_index_t ebN, eN_global; int ebNE, ebN_element, k, J;
  double left_speed,right_speed,left_flux,right_flux,shock_speed,flux_jump,u_jump;
  for(ebNE=0;ebNE<nExteriorElementBoundaries_global;ebNE++)
    {
      ebN = exteriorElementBoundaries[ebNE];
      eN_global = elementBoundaryElements[ebN*2+0];
      ebN_element = elementBoundaryLocalElementBoundaries[ebN*2+0];
      for(k=0;k<nQuadraturePoints_elementBoundary;k++)
        {
          left_speed=0.0;
          right_speed=0.0;
          left_flux=0.0;
          right_flux=0.0;
	  /*mwf add default shock speed is zero*/
	  shock_speed=0.0;
          for(J=0;J<nSpace;J++)
            {
              left_speed 
                += 
                n[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                  ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                  k*nSpace+
		  J]
                *
                df[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                   ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                   k*nSpace+
		   J];
              right_speed 
                += 
                n[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                  ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                  k*nSpace+
		  J]
                *
                bc_df[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                      k*nSpace+
                      J];
             left_flux 
                += 
                n[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                  ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                  k*nSpace+
		  J]
                *
                f[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                  ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                  k*nSpace+
                  J];
             right_flux 
               += 
               n[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                 ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                 k*nSpace+
                 J]
               *
               bc_f[ebNE*nQuadraturePoints_elementBoundary*nSpace+
		    k*nSpace+
                    J];
            }
          flux_jump = (right_flux - left_flux);
          u_jump = (bc_u[ebNE*nQuadraturePoints_elementBoundary+
			 k]
                    -
                    u[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary+
                      ebN_element*nQuadraturePoints_elementBoundary+
                      k]);
          if (fabs(u_jump) > fabs(flux_jump*1.0e-16))
            shock_speed = flux_jump/u_jump;
          if (left_speed >= 0.0 && right_speed >= 0.0)
            {
              flux[ebN*nQuadraturePoints_elementBoundary+
                   k] = left_flux;
              dflux[ebN*nQuadraturePoints_elementBoundary+
                    k] = left_speed;
            }
          else if (left_speed <= 0.0 && right_speed <= 0.0)
            {
              flux[ebN*nQuadraturePoints_elementBoundary+
                   k] = right_flux;
              if (isDOFBoundary[ebNE*nQuadraturePoints_elementBoundary+k] == 1)
                dflux[ebN*nQuadraturePoints_elementBoundary+
                      k] = 0.0;
              else
                dflux[ebN*nQuadraturePoints_elementBoundary+
                      k] = right_speed;
            }
          else if (left_speed >= 0.0 && right_speed <= 0.0)
            {
              if (shock_speed >= 0.0)
                {
                  flux[ebN*nQuadraturePoints_elementBoundary+
                       k] = left_flux;
                  dflux[ebN*nQuadraturePoints_elementBoundary+
                        k] = left_speed;
                }
              else
                {
                  flux[ebN*nQuadraturePoints_elementBoundary+
                       k] = right_flux;
                  if (isDOFBoundary[ebNE*nQuadraturePoints_elementBoundary+k] == 1)
                    dflux[ebN*nQuadraturePoints_elementBoundary+
                          k] = 0.0;
                  else
                    dflux[ebN*nQuadraturePoints_elementBoundary+
                          k] = right_speed;
                }
            }
          else
            {
              /*transonic rarefaction*/
              printf("Transonic rarefaction detected on exterior. This numerical flux treats transonic rarefactions incorrectly. left_speed= %12.5e right_speed= %12.5e \n",left_speed,right_speed);
	      for (J=0; J < nSpace; J++)
		{
		  printf("n_l[%d]=%g df_l[%d]=%g df_r[%d]=%g \n",J,
			 n[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
			   ebN_element*nQuadraturePoints_elementBoundary*nSpace+
			   k*nSpace+
			   J],
			 J,
			 df[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
			    ebN_element*nQuadraturePoints_elementBoundary*nSpace+
			    k*nSpace+
			    J],
			 J,
			 bc_df[ebNE*nQuadraturePoints_elementBoundary*nSpace+
			       k*nSpace+
			       J]);
			 
		}
              flux[ebN*nQuadraturePoints_elementBoundary+
                   k] = 0.5*(left_flux + right_flux);
              if (isDOFBoundary[ebNE*nQuadraturePoints_elementBoundary+k] == 1)
                dflux[ebN*nQuadraturePoints_elementBoundary+
                      k] = 0.5*left_speed;
              else
                dflux[ebN*nQuadraturePoints_elementBoundary+
                      k] = 0.5*(left_speed+right_speed);
            }
/*           printf("exterior flux %d %d %12.5e \n",ebN,k,flux[ebN*nQuadraturePoints_elementBoundary+k]);  */
	  /*mwf debug
	  printf("in exterior diagonal eN=%d u_left=%g left_flux=%g left_speed=%g bc_u=%g right_flux=%g  right_speed=%g shock_speed=%g\n",
		 eN_global,u[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary+
                      ebN_element*nQuadraturePoints_elementBoundary+
                      k],
		 left_flux,
		 left_speed,
		 bc_u[ebNE*nQuadraturePoints_elementBoundary+
		      k],
		 right_flux,
		 right_speed,
		 shock_speed);
	  printf("\tflux = %g n=[",flux[ebN*nQuadraturePoints_elementBoundary+k]);
	  for (J=0; J < nSpace; J++)
	    printf(" %g  ",n[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
			     ebN_element*nQuadraturePoints_elementBoundary*nSpace+
			     k*nSpace+
			     J]);
	  printf("]\n");
	  mwf end debug */
/*           if(dflux[ebN*nQuadraturePoints_elementBoundary+k] < 0.0) */
/*             { */
/*               /\* cek debug, setting inflow flow to zero *\/ */
/*               inflowFlag[ebNE*nQuadraturePoints_elementBoundary+k] = 1; */
/*               flux[ebN*nQuadraturePoints_elementBoundary+k] = 0.0; */
/*               dflux[ebN*nQuadraturePoints_elementBoundary+k] = 0.0; */
/*             } */
        }
    }
}

/**
   \brief Calculate the advective flux at at exterior element boundaries
*/
void calculateGlobalExteriorNumericalAdvectiveFlux(proteus_index_t nExteriorElementBoundaries_global,
						   int nQuadraturePoints_elementBoundary,
						   int nSpace,
						   proteus_index_t* exteriorElementBoundaries,
						   proteus_index_t* elementBoundaryElements,
						   int* elementBoundaryLocalElementBoundaries,
						   int *isDOFBoundary,
						   int *inflowFlag,
						   double* n,
						   double* bc_u,
						   double* bc_f,
						   double* bc_df,
						   double* u,
						   double* f,
						   double* df,
						   double* flux,
						   double* dflux)
{
  proteus_index_t ebN, eN_global; int ebNE, k, J;
  double left_speed,right_speed,left_flux,right_flux,shock_speed,flux_jump,u_jump;
  for(ebNE=0;ebNE<nExteriorElementBoundaries_global;ebNE++)
    {
      ebN = exteriorElementBoundaries[ebNE];
      eN_global = elementBoundaryElements[ebN*2+0];
      for(k=0;k<nQuadraturePoints_elementBoundary;k++)
        {
          left_speed=0.0;
          right_speed=0.0;
          left_flux=0.0;
          right_flux=0.0;
	  shock_speed=0.0;
          for(J=0;J<nSpace;J++)
            {
              left_speed 
                += 
                n[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                  k*nSpace+
		  J]
                *
                df[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                   k*nSpace+
		   J];
              right_speed 
                += 
                n[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                  k*nSpace+
		  J]
                *
                bc_df[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                      k*nSpace+
                      J];
             left_flux 
                += 
                n[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                  k*nSpace+
		  J]
                *
                f[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                  k*nSpace+
                  J];
             right_flux 
               += 
               n[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                 k*nSpace+
                 J]
               *
               bc_f[ebNE*nQuadraturePoints_elementBoundary*nSpace+
		    k*nSpace+
                    J];
            }
          flux_jump = (right_flux - left_flux);
          u_jump = (bc_u[ebNE*nQuadraturePoints_elementBoundary+
			 k]
                    -
                    u[ebNE*nQuadraturePoints_elementBoundary+
                      k]);
          if (fabs(u_jump) > fabs(flux_jump*1.0e-16))
            shock_speed = flux_jump/u_jump;
          if (left_speed >= 0.0 && right_speed >= 0.0)
            {
              flux[ebNE*nQuadraturePoints_elementBoundary+
                   k] = left_flux;
              dflux[ebNE*nQuadraturePoints_elementBoundary+
                    k] = left_speed;
            }
          else if (left_speed <= 0.0 && right_speed <= 0.0)
            {
              flux[ebNE*nQuadraturePoints_elementBoundary+
                   k] = right_flux;
              if (isDOFBoundary[ebNE*nQuadraturePoints_elementBoundary+k] == 1)
                dflux[ebNE*nQuadraturePoints_elementBoundary+
                      k] = 0.0;
              else
                dflux[ebNE*nQuadraturePoints_elementBoundary+
                      k] = right_speed;
            }
          else if (left_speed >= 0.0 && right_speed <= 0.0)
            {
              if (shock_speed >= 0.0)
                {
                  flux[ebNE*nQuadraturePoints_elementBoundary+
                       k] = left_flux;
                  dflux[ebNE*nQuadraturePoints_elementBoundary+
                        k] = left_speed;
                }
              else
                {
                  flux[ebNE*nQuadraturePoints_elementBoundary+
                       k] = right_flux;
                  if (isDOFBoundary[ebNE*nQuadraturePoints_elementBoundary+k] == 1)
                    dflux[ebNE*nQuadraturePoints_elementBoundary+
                          k] = 0.0;
                  else
                    dflux[ebNE*nQuadraturePoints_elementBoundary+
                          k] = right_speed;
                }
            }
          else
            {
              /*transonic rarefaction*/
              printf("Transonic rarefaction detected on exterior. This numerical flux treats transonic rarefactions incorrectly. left_speed= %12.5e right_speed= %12.5e \n",left_speed,right_speed);
	      for (J=0; J < nSpace; J++)
		{
		  printf("n_l[%d]=%g df_l[%d]=%g df_r[%d]=%g \n",J,
			 n[ebNE*nQuadraturePoints_elementBoundary*nSpace+
			   k*nSpace+
			   J],
			 J,
			 df[ebNE*nQuadraturePoints_elementBoundary*nSpace+
			    k*nSpace+
			    J],
			 J,
			 bc_df[ebNE*nQuadraturePoints_elementBoundary*nSpace+
			       k*nSpace+
			       J]);
			 
		}
              flux[ebNE*nQuadraturePoints_elementBoundary+
                   k] = 0.5*(left_flux + right_flux);
              if (isDOFBoundary[ebNE*nQuadraturePoints_elementBoundary+k] == 1)
                dflux[ebNE*nQuadraturePoints_elementBoundary+
                      k] = 0.5*left_speed;
              else
                dflux[ebNE*nQuadraturePoints_elementBoundary+
                      k] = 0.5*(left_speed+right_speed);
            }
/*           printf("exterior flux %d %d %12.5e \n",ebNE,k,flux[ebNE*nQuadraturePoints_elementBoundary+k]);  */
/*           if(dflux[ebN*nQuadraturePoints_elementBoundary+k] < 0.0) */
/*             { */
/*               /\* cek debug, setting inflow flow to zero *\/ */
/*               inflowFlag[ebNE*nQuadraturePoints_elementBoundary+k] = 1; */
/*               flux[ebNE*nQuadraturePoints_elementBoundary+k] = 0.0; */
/*               dflux[ebNE*nQuadraturePoints_elementBoundary+k] = 0.0; */
/*             } */
        }
    }
}
void calculateExteriorNumericalAdvectiveFlux_free(proteus_index_t nExteriorElementBoundaries_global,
                                                  int nElementBoundaries_element,
                                                  int nQuadraturePoints_elementBoundary,
                                                  int nSpace,
                                                  proteus_index_t* exteriorElementBoundaries,
                                                  proteus_index_t* elementBoundaryElements,
                                                  int* elementBoundaryLocalElementBoundaries,
                                                  int *isDOFBoundary,
                                                  int *inflowFlag,
                                                  double* n,
                                                  double* bc_u,
                                                  double* bc_f,
                                                  double* bc_df,
                                                  double* u,
                                                  double* f,
                                                  double* df,
                                                  double* flux,
                                                  double* dflux)
{
  proteus_index_t ebN, eN_global; int ebNE, ebN_element, k, J;
  double left_speed,right_speed,left_flux,right_flux,shock_speed;
  for(ebNE=0;ebNE<nExteriorElementBoundaries_global;ebNE++)
    {
      ebN = exteriorElementBoundaries[ebNE];
      eN_global = elementBoundaryElements[ebN*2+0];
      ebN_element = elementBoundaryLocalElementBoundaries[ebN*2+0];
      for(k=0;k<nQuadraturePoints_elementBoundary;k++)
        {
          left_speed=0.0;
          right_speed=0.0;
          left_flux=0.0;
          right_flux=0.0;
	  /*mwf add default shock speed is zero*/
	  shock_speed=0.0;
          for(J=0;J<nSpace;J++)
            {
              left_speed 
                += 
                n[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                  ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                  k*nSpace+
		  J]
                *
                df[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                   ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                   k*nSpace+
		   J];
             left_flux 
                += 
                n[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                  ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                  k*nSpace+
		  J]
                *
                f[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                  ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                  k*nSpace+
                  J];
            }
          flux[ebN*nQuadraturePoints_elementBoundary+
               k] = left_flux;
          dflux[ebN*nQuadraturePoints_elementBoundary+
                k] = left_speed;
        }
    }
}

void calculateGlobalExteriorNumericalAdvectiveFlux_free(proteus_index_t nExteriorElementBoundaries_global,
							int nQuadraturePoints_elementBoundary,
							int nSpace,
							proteus_index_t* exteriorElementBoundaries,
							proteus_index_t* elementBoundaryElements,
							int* elementBoundaryLocalElementBoundaries,
							int *isDOFBoundary,
							int *inflowFlag,
							double* n,
							double* bc_u,
							double* bc_f,
							double* bc_df,
							double* u,
							double* f,
							double* df,
							double* flux,
							double* dflux)
{
  proteus_index_t ebN, eN_global; int ebNE, k, J;
  double left_speed,right_speed,left_flux,right_flux,shock_speed;
  for(ebNE=0;ebNE<nExteriorElementBoundaries_global;ebNE++)
    {
      ebN = exteriorElementBoundaries[ebNE];
      eN_global = elementBoundaryElements[ebN*2+0];
      for(k=0;k<nQuadraturePoints_elementBoundary;k++)
        {
          left_speed=0.0;
          right_speed=0.0;
          left_flux=0.0;
          right_flux=0.0;
	  shock_speed=0.0;
          for(J=0;J<nSpace;J++)
            {
              left_speed 
                += 
                n[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                  k*nSpace+
		  J]
                *
                df[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                   k*nSpace+
		   J];
             left_flux 
                += 
                n[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                  k*nSpace+
		  J]
                *
                f[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                  k*nSpace+
                  J];
            }
          flux[ebNE*nQuadraturePoints_elementBoundary+
               k] = left_flux;
          dflux[ebNE*nQuadraturePoints_elementBoundary+
                k] = left_speed;
        }
    }
}
/**
   \brief Calculate the advective flux at at exterior element boundaries
*/
void calculateExteriorNumericalAdvectiveFluxStokesP2D(proteus_index_t nExteriorElementBoundaries_global,
                                                     int nElementBoundaries_element,
                                                     int nQuadraturePoints_elementBoundary,
                                                     int nSpace,
                                                     proteus_index_t* exteriorElementBoundaries,
                                                     proteus_index_t* elementBoundaryElements,
                                                     int* elementBoundaryLocalElementBoundaries,
                                                     int *isDOFBoundary_p,
                                                     int *isDOFBoundary_u,
                                                     int *isDOFBoundary_v,
                                                     double* n,
                                                     double* bc_f,
                                                     double* bc_fpu,
                                                     double* bc_fpv,
                                                     double* f,
                                                     double* fpu,
                                                     double* fpv,
                                                     double* df_du,
                                                     double* df_dv,
                                                     double* dfpu_dp,
                                                     double* dfpv_dp,
                                                     double* flux,
                                                     double* fluxpu,
                                                     double* fluxpv,
                                                     double* dflux_du,
                                                     double* dflux_dv,
                                                     double* dfluxpu_dp,
                                                     double* dfluxpv_dp)
{
  proteus_index_t ebN, eN_global; int ebNE, ebN_element, k;
  for(ebNE=0;ebNE<nExteriorElementBoundaries_global;ebNE++)
    {
      ebN = exteriorElementBoundaries[ebNE];
      eN_global = elementBoundaryElements[ebN*2+0];
      ebN_element = elementBoundaryLocalElementBoundaries[ebN*2+0];
      for(k=0;k<nQuadraturePoints_elementBoundary;k++)
        {
	  fluxpu[ebN*nQuadraturePoints_elementBoundary+
		 k] = 0.0;
	  fluxpv[ebN*nQuadraturePoints_elementBoundary+
		 k] = 0.0;
	  flux[ebN*nQuadraturePoints_elementBoundary+
	       k]  = 0.0;

          //u and v momentum fluxes due to pressure
          if (isDOFBoundary_p[ebNE*nQuadraturePoints_elementBoundary+k] != 1)
            {
              dfluxpu_dp[ebN*nQuadraturePoints_elementBoundary+
                         k]
                = 
                n[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                  ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                  k*nSpace+
                  0]
                *
                dfpu_dp[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                        ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                        k*nSpace+
                        0];
              fluxpu[ebN*nQuadraturePoints_elementBoundary+
                     k]+= 
                n[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                  ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                  k*nSpace+
		  0]
                *
                fpu[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                    ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                    k*nSpace+
                    0];
              dfluxpv_dp[ebN*nQuadraturePoints_elementBoundary+
                         k]
                = 
                n[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                  ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                  k*nSpace+
                  1]
                *
                dfpv_dp[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                        ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                        k*nSpace+
                        1];
              fluxpv[ebN*nQuadraturePoints_elementBoundary+
                     k]+= 
                n[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                  ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                  k*nSpace+
		  1]
                *
                fpv[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                    ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                    k*nSpace+
                    1];
            }
          else
            {
              fluxpu[ebN*nQuadraturePoints_elementBoundary+
                     k]+= 
                n[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                  ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                  k*nSpace+
		  0]
                *
                bc_fpu[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                       k*nSpace+
                       0];
              fluxpv[ebN*nQuadraturePoints_elementBoundary+
                     k]+= 
                n[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                  ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                  k*nSpace+
		  1]
                *
                bc_fpv[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                       k*nSpace+
                       1];
            }
          //mass flux
          if (isDOFBoundary_u[ebNE*nQuadraturePoints_elementBoundary+k] != 1)
            {
              dflux_du[ebN*nQuadraturePoints_elementBoundary+
                       k]
                = 
                n[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                  ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                  k*nSpace+
                  0]
                *
                df_du[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                      ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                      k*nSpace+
                      0];
              flux[ebN*nQuadraturePoints_elementBoundary+
                   k]+= 
                n[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                  ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                  k*nSpace+
		  0]
                *
                f[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                  ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                  k*nSpace+
                  0];
            }
          else
            {
              flux[ebN*nQuadraturePoints_elementBoundary+
                   k]+= 
                n[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                  ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                  k*nSpace+
		  0]
                *
                bc_f[ebNE*nQuadraturePoints_elementBoundary*nSpace+
		     k*nSpace+
                     0];
            }
          if (isDOFBoundary_v[ebNE*nQuadraturePoints_elementBoundary+k] != 1)
            {
              dflux_dv[ebN*nQuadraturePoints_elementBoundary+
                       k]
                = 
                n[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                  ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                  k*nSpace+
                  1]
                *
                df_dv[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                      ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                      k*nSpace+
                      1];
              flux[ebN*nQuadraturePoints_elementBoundary+
                   k]+= 
                n[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                  ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                  k*nSpace+
		  1]
                *
                f[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                  ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                  k*nSpace+
                  1];
            }
          else
            {
              flux[ebN*nQuadraturePoints_elementBoundary+
                   k]+= 
                n[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                  ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                  k*nSpace+
		  1]
                *
                bc_f[ebNE*nQuadraturePoints_elementBoundary*nSpace+
		     k*nSpace+
                     1];
            }
        }
    }
}
void calculateExteriorNumericalAdvectiveFluxNavierStokes2D(proteus_index_t nExteriorElementBoundaries_global,
                                                           int nElementBoundaries_element,
                                                           int nQuadraturePoints_elementBoundary,
                                                           int nSpace,
                                                           proteus_index_t* exteriorElementBoundaries,
                                                           proteus_index_t* elementBoundaryElements,
                                                           int* elementBoundaryLocalElementBoundaries,
                                                           int *isDOFBoundary_p,
                                                           int *isDOFBoundary_u,
                                                           int *isDOFBoundary_v,
                                                           double* n,
                                                           double* bc_p,
                                                           double* bc_f_mass,
                                                           double* bc_f_umom,
                                                           double* bc_f_vmom,
                                                           double* p,
                                                           double* f_mass,
                                                           double* f_umom,
                                                           double* f_vmom,
                                                           double* df_mass_du,
                                                           double* df_mass_dv,
                                                           double* df_umom_du,
                                                           double* df_umom_dv,
                                                           double* df_vmom_du,
                                                           double* df_vmom_dv,
                                                           double* flux_mass,
                                                           double* flux_umom,
                                                           double* flux_vmom,
                                                           double* dflux_mass_du,
                                                           double* dflux_mass_dv,
                                                           double* dflux_umom_dp,
                                                           double* dflux_umom_du,
                                                           double* dflux_umom_dv,
                                                           double* dflux_vmom_dp,
                                                           double* dflux_vmom_du,
                                                           double* dflux_vmom_dv)
{
  proteus_index_t ebN, eN_global; int ebNE, ebN_element, k;
  for(ebNE=0;ebNE<nExteriorElementBoundaries_global;ebNE++)
    {
      ebN = exteriorElementBoundaries[ebNE];
      eN_global = elementBoundaryElements[ebN*2+0];
      ebN_element = elementBoundaryLocalElementBoundaries[ebN*2+0];
      for(k=0;k<nQuadraturePoints_elementBoundary;k++)
        {
	  flux_umom[ebN*nQuadraturePoints_elementBoundary+
                    k] = 0.0;
	  flux_vmom[ebN*nQuadraturePoints_elementBoundary+
                    k] = 0.0;
	  flux_mass[ebN*nQuadraturePoints_elementBoundary+
                    k]  = 0.0;
          if (isDOFBoundary_u[ebNE*nQuadraturePoints_elementBoundary+k] != 1)
            {
              dflux_mass_du[ebN*nQuadraturePoints_elementBoundary+
                            k]
                = 
                n[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                  ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                  k*nSpace+
                  0]
                *
                df_mass_du[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                           ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                           k*nSpace+
                           0];
              flux_mass[ebN*nQuadraturePoints_elementBoundary+
                        k]+= 
                n[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                  ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                  k*nSpace+
		  0]
                *
                f_mass[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                       ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                       k*nSpace+
                       0];
              dflux_umom_du[ebN*nQuadraturePoints_elementBoundary+
                            k]
                = 
                n[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                  ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                  k*nSpace+
                  0]
                *
                df_umom_du[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                           ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                           k*nSpace+
                           0];
              flux_umom[ebN*nQuadraturePoints_elementBoundary+
                        k]
		+= 
                n[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                  ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                  k*nSpace+
		  0]
                *
                f_umom[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                       ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                       k*nSpace+
                       0];
              dflux_vmom_du[ebN*nQuadraturePoints_elementBoundary+
                            k]
                = 
                n[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                  ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                  k*nSpace+
                  0]
                *
                df_vmom_du[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                           ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                           k*nSpace+
                           0];
              flux_vmom[ebN*nQuadraturePoints_elementBoundary+
                        k]
		+= 
                n[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                  ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                  k*nSpace+
		  0]
                *
                f_vmom[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                       ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                       k*nSpace+
                       0];
            }
          else
            {
              flux_mass[ebN*nQuadraturePoints_elementBoundary+
                        k]
		+= 
                n[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                  ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                  k*nSpace+
		  0]
                *
                bc_f_mass[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                          k*nSpace+
                          0];
              flux_umom[ebN*nQuadraturePoints_elementBoundary+
                        k]
		+= 
                n[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                  ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                  k*nSpace+
		  0]
                *
                bc_f_umom[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                          k*nSpace+
                          0];
              flux_vmom[ebN*nQuadraturePoints_elementBoundary+
                        k]
		+= 
                n[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                  ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                  k*nSpace+
		  0]
                *
                bc_f_vmom[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                          k*nSpace+
                          0];
            }
          if (isDOFBoundary_p[ebNE*nQuadraturePoints_elementBoundary+k] == 1)
            {
              dflux_umom_dp[ebN*nQuadraturePoints_elementBoundary+
                            k]
                = 
                n[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                  ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                  k*nSpace+
                  0];
              flux_umom[ebN*nQuadraturePoints_elementBoundary+
                        k]
		+= 
                n[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                  ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                  k*nSpace+
		  0]
                *
                (p[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary+
                   ebN_element*nQuadraturePoints_elementBoundary+
                   k]
                 -
                 bc_p[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary+
                      ebN_element*nQuadraturePoints_elementBoundary+
                      k]);
              dflux_vmom_dp[ebN*nQuadraturePoints_elementBoundary+
                            k]
                = 
                n[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                  ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                  k*nSpace+
                  1];
              flux_vmom[ebN*nQuadraturePoints_elementBoundary+
                        k]+= 
                n[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                  ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                  k*nSpace+
		  1]
                *
                (p[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary+
                   ebN_element*nQuadraturePoints_elementBoundary+
                   k]
                 -
                 bc_p[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary+
                      ebN_element*nQuadraturePoints_elementBoundary+
                      k]);
            }
        }
    }
}

void calculateGlobalExteriorNumericalAdvectiveFluxNavierStokes2D(proteus_index_t nExteriorElementBoundaries_global,
                                                                 int nQuadraturePoints_elementBoundary,
                                                                 int nSpace,
                                                                 proteus_index_t* exteriorElementBoundaries,
                                                                 proteus_index_t* elementBoundaryElements,
                                                                 int* elementBoundaryLocalElementBoundaries,
                                                                 int *isDOFBoundary_p,
                                                                 int *isDOFBoundary_u,
                                                                 int *isDOFBoundary_v,
                                                                 double* n,
                                                                 double* bc_p,
                                                                 double* bc_f_mass,
                                                                 double* bc_f_umom,
                                                                 double* bc_f_vmom,
                                                                 double* p,
                                                                 double* oneByRho,
                                                                 double* f_mass,
                                                                 double* f_umom,
                                                                 double* f_vmom,
                                                                 double* df_mass_du,
                                                                 double* df_mass_dv,
                                                                 double* df_umom_dp,
                                                                 double* df_umom_du,
                                                                 double* df_umom_dv,
                                                                 double* df_vmom_dp,
                                                                 double* df_vmom_du,
                                                                 double* df_vmom_dv,
                                                                 double* flux_mass,
                                                                 double* flux_umom,
                                                                 double* flux_vmom,
                                                                 double* dflux_mass_dp,
                                                                 double* dflux_mass_du,
                                                                 double* dflux_mass_dv,
                                                                 double* dflux_umom_dp,
                                                                 double* dflux_umom_du,
                                                                 double* dflux_umom_dv,
                                                                 double* dflux_vmom_dp,
                                                                 double* dflux_vmom_du,
                                                                 double* dflux_vmom_dv,
								 double* velocity)
{
  int ebNE,k;
  double flowDirection;
  for(ebNE=0;ebNE<nExteriorElementBoundaries_global;ebNE++)
    {
      for(k=0;k<nQuadraturePoints_elementBoundary;k++)
        {
	  flux_mass[ebNE*nQuadraturePoints_elementBoundary+
                    k]  = 0.0;
	  dflux_mass_du[ebNE*nQuadraturePoints_elementBoundary+
                    k]  = 0.0;
	  dflux_mass_dv[ebNE*nQuadraturePoints_elementBoundary+
                    k]  = 0.0;

	  flux_umom[ebNE*nQuadraturePoints_elementBoundary+
                    k] = 0.0;
	  dflux_umom_dp[ebNE*nQuadraturePoints_elementBoundary+
                        k] = 0.0;
	  dflux_umom_du[ebNE*nQuadraturePoints_elementBoundary+
                    k] = 0.0;
	  dflux_umom_dv[ebNE*nQuadraturePoints_elementBoundary+
                    k] = 0.0;

	  flux_vmom[ebNE*nQuadraturePoints_elementBoundary+
                    k] = 0.0;
	  dflux_vmom_dp[ebNE*nQuadraturePoints_elementBoundary+
                    k] = 0.0;
	  dflux_vmom_du[ebNE*nQuadraturePoints_elementBoundary+
                    k] = 0.0;
	  dflux_vmom_dv[ebNE*nQuadraturePoints_elementBoundary+
                    k] = 0.0;

          flowDirection=n[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                          k*nSpace+
                          0]
            *
            f_mass[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                   k*nSpace+
                   0]
            +
	    n[ebNE*nQuadraturePoints_elementBoundary*nSpace+
	      k*nSpace+
	      1]
            *
            f_mass[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                   k*nSpace+
                   1];
          if (isDOFBoundary_u[ebNE*nQuadraturePoints_elementBoundary+k] != 1)
            {
              flux_mass[ebNE*nQuadraturePoints_elementBoundary+
                        k]
                +=
                n[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                  k*nSpace+
                  0]
                *
                f_mass[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                       k*nSpace+
                       0];
	      velocity[ebNE*nQuadraturePoints_elementBoundary*nSpace+
		       k*nSpace+
		       0]
		=
		f_mass[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                       k*nSpace+
                       0];
              dflux_mass_du[ebNE*nQuadraturePoints_elementBoundary+
                            k]
                +=
                n[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                  k*nSpace+
                  0]
                *
                df_mass_du[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                           k*nSpace+
                           0];
              if (flowDirection >= 0.0)
                {
                  flux_umom[ebNE*nQuadraturePoints_elementBoundary+
                            k]
		    +=
                    n[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                      k*nSpace+
                      0]
                    *
                    f_umom[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                           k*nSpace+
                           0];
                  dflux_umom_du[ebNE*nQuadraturePoints_elementBoundary+
                                k]
                    +=
                    n[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                      k*nSpace+
                      0]
                    *
                    df_umom_du[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                               k*nSpace+
                               0];
                  flux_vmom[ebNE*nQuadraturePoints_elementBoundary+
                            k] 
		    +=
                    n[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                      k*nSpace+
                      0]
                    *
                    f_vmom[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                           k*nSpace+
                           0];
                  dflux_vmom_du[ebNE*nQuadraturePoints_elementBoundary+
                                k]
                    +=
                    n[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                      k*nSpace+
                      0]
                    *
                    df_vmom_du[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                               k*nSpace+
                               0];
                  dflux_vmom_dv[ebNE*nQuadraturePoints_elementBoundary+
                                k]
                    +=
                    n[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                      k*nSpace+
                      0]
                    *
                    df_vmom_dv[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                               k*nSpace+
                               0];
               }
            }
          else
            {
              flux_mass[ebNE*nQuadraturePoints_elementBoundary+
                        k]
		+=
                n[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                  k*nSpace+
		  0]
                *
                bc_f_mass[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                          k*nSpace+
                          0];
	      velocity[ebNE*nQuadraturePoints_elementBoundary*nSpace+
		       k*nSpace+
		       0]
		=
		bc_f_mass[ebNE*nQuadraturePoints_elementBoundary*nSpace+
			  k*nSpace+
			  0];
              flux_umom[ebNE*nQuadraturePoints_elementBoundary+
                        k]
		+=
                n[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                  k*nSpace+
		  0]
                *
                bc_f_umom[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                          k*nSpace+
                          0];
              flux_vmom[ebNE*nQuadraturePoints_elementBoundary+
                        k]
		+=
                n[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                  k*nSpace+
		  0]
                *
                bc_f_vmom[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                          k*nSpace+
                          0];
              if (isDOFBoundary_v[ebNE*nQuadraturePoints_elementBoundary+k] != 1)
                dflux_vmom_dv[ebNE*nQuadraturePoints_elementBoundary+
                              k]
                  +=
                  n[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                    k*nSpace+
                    0]
                  *
                  df_vmom_dv[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                             k*nSpace+
                             0];
            }
          if (isDOFBoundary_v[ebNE*nQuadraturePoints_elementBoundary+k] != 1)
            {
              flux_mass[ebNE*nQuadraturePoints_elementBoundary+
                        k]
		+=
                n[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                  k*nSpace+
		  1]
                *
                f_mass[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                       k*nSpace+
                       1];
	      velocity[ebNE*nQuadraturePoints_elementBoundary*nSpace+
		       k*nSpace+
		       1]
		=
		f_mass[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                       k*nSpace+
                       1];
              dflux_mass_dv[ebNE*nQuadraturePoints_elementBoundary+
                            k]
                +=
                n[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                  k*nSpace+
                  1]
                *
                df_mass_dv[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                           k*nSpace+
                           1];
              if (flowDirection >= 0.0)
                {
                  flux_umom[ebNE*nQuadraturePoints_elementBoundary+
                            k]
		    +=
                    n[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                      k*nSpace+
                      1]
                    *
                    f_umom[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                           k*nSpace+
                           1];
                  dflux_umom_du[ebNE*nQuadraturePoints_elementBoundary+
                                k]
                    +=
                    n[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                      k*nSpace+
                      1]
                    *
                    df_umom_du[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                               k*nSpace+
                               1];
                  dflux_umom_dv[ebNE*nQuadraturePoints_elementBoundary+
                                k]
                    +=
                    n[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                      k*nSpace+
                      1]
                    *
                    df_umom_dv[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                               k*nSpace+
                               1];
                  flux_vmom[ebNE*nQuadraturePoints_elementBoundary+
                            k]
		    +=
                    n[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                      k*nSpace+
                      1]
                    *
                    f_vmom[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                           k*nSpace+
                           1];
                  dflux_vmom_dv[ebNE*nQuadraturePoints_elementBoundary+
                                k]
                    += n[ebNE*nQuadraturePoints_elementBoundary*nSpace+
			 k*nSpace+
			 1]
                    *
                    df_vmom_dv[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                               k*nSpace+
                               1];
                }
            }
          else
            {
              flux_mass[ebNE*nQuadraturePoints_elementBoundary+
                        k]
		+=
                n[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                  k*nSpace+
		  1]
                *
                bc_f_mass[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                          k*nSpace+
                          1];
	      velocity[ebNE*nQuadraturePoints_elementBoundary*nSpace+
		       k*nSpace+
		       1]
		=
		bc_f_mass[ebNE*nQuadraturePoints_elementBoundary*nSpace+
			  k*nSpace+
			  1];

              flux_umom[ebNE*nQuadraturePoints_elementBoundary+
                        k]
		+=
                n[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                  k*nSpace+
		  1]
                *
                bc_f_umom[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                          k*nSpace+
                          1];
              if (isDOFBoundary_u[ebNE*nQuadraturePoints_elementBoundary+k] != 1)
                dflux_umom_du[ebNE*nQuadraturePoints_elementBoundary+
                              k]
                  +=
                  n[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                    k*nSpace+
                    1]
                  *
                  df_umom_du[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                             k*nSpace+
                             1];
              flux_vmom[ebNE*nQuadraturePoints_elementBoundary+
                        k]
		+=
                n[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                  k*nSpace+
		  1]
                *
                bc_f_vmom[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                          k*nSpace+
                          1];
            }
          if (isDOFBoundary_p[ebNE*nQuadraturePoints_elementBoundary+k] == 1)
            {
              /* flux_mass[ebNE*nQuadraturePoints_elementBoundary+ */
              /*           k] */
              /*   += */
              /*   oneByRho[ebNE*nQuadraturePoints_elementBoundary+ */
	      /* 		    k]*(bc_p[ebNE*nQuadraturePoints_elementBoundary+ */
	      /* 		       k] */
	      /* 		  - */
	      /* 		  p[ebNE*nQuadraturePoints_elementBoundary+ */
	      /* 		    k]); */
              /* dflux_mass_dp[ebNE*nQuadraturePoints_elementBoundary+ */
              /*               k] */
              /*   = -oneByRho[ebNE*nQuadraturePoints_elementBoundary+ */
	      /* 		    k]; */
              flux_umom[ebNE*nQuadraturePoints_elementBoundary+
                        k]
                +=
                n[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                  k*nSpace+
		  0]
                *
                (bc_p[ebNE*nQuadraturePoints_elementBoundary+
                      k]
                 -
                 p[ebNE*nQuadraturePoints_elementBoundary+
                   k]);
              dflux_umom_dp[ebNE*nQuadraturePoints_elementBoundary+
                            k]
                = -n[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                     k*nSpace+
                     0];
              flux_vmom[ebNE*nQuadraturePoints_elementBoundary+
                        k]
		+= n[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                     k*nSpace+
                     1]
                *
                (bc_p[ebNE*nQuadraturePoints_elementBoundary+
                      k]
                 -
                 p[ebNE*nQuadraturePoints_elementBoundary+
                   k]);
              dflux_vmom_dp[ebNE*nQuadraturePoints_elementBoundary+
                            k]
                = -n[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                     k*nSpace+
                     1];
            }
        }
    }
}
void calculateGlobalExteriorNumericalAdvectiveFluxNavierStokes3D(proteus_index_t nExteriorElementBoundaries_global,
                                                                 int nQuadraturePoints_elementBoundary,
                                                                 int nSpace,
                                                                 proteus_index_t* exteriorElementBoundaries,
                                                                 proteus_index_t* elementBoundaryElements,
                                                                 int* elementBoundaryLocalElementBoundaries,
                                                                 int *isDOFBoundary_p,
                                                                 int *isDOFBoundary_u,
                                                                 int *isDOFBoundary_v,
                                                                 int *isDOFBoundary_w,
                                                                 double* n,
                                                                 double* bc_p,
                                                                 double* bc_f_mass,
                                                                 double* bc_f_umom,
                                                                 double* bc_f_vmom,
                                                                 double* bc_f_wmom,
                                                                 double* p,
                                                                 double* f_mass,
                                                                 double* f_umom,
                                                                 double* f_vmom,
                                                                 double* f_wmom,
                                                                 double* df_mass_du,
                                                                 double* df_mass_dv,
                                                                 double* df_mass_dw,
                                                                 double* df_umom_dp,
                                                                 double* df_umom_du,
                                                                 double* df_umom_dv,
                                                                 double* df_umom_dw,
                                                                 double* df_vmom_dp,
                                                                 double* df_vmom_du,
                                                                 double* df_vmom_dv,
                                                                 double* df_vmom_dw,
                                                                 double* df_wmom_dp,
                                                                 double* df_wmom_du,
                                                                 double* df_wmom_dv,
                                                                 double* df_wmom_dw,
                                                                 double* flux_mass,
                                                                 double* flux_umom,
                                                                 double* flux_vmom,
                                                                 double* flux_wmom,
                                                                 double* dflux_mass_du,
                                                                 double* dflux_mass_dv,
                                                                 double* dflux_mass_dw,
                                                                 double* dflux_umom_dp,
                                                                 double* dflux_umom_du,
                                                                 double* dflux_umom_dv,
                                                                 double* dflux_umom_dw,
                                                                 double* dflux_vmom_dp,
                                                                 double* dflux_vmom_du,
                                                                 double* dflux_vmom_dv,
                                                                 double* dflux_vmom_dw,
                                                                 double* dflux_wmom_dp,
                                                                 double* dflux_wmom_du,
                                                                 double* dflux_wmom_dv,
                                                                 double* dflux_wmom_dw,
								 double* velocity)
{
  int ebNE,k;
  double flowDirection;
  for(ebNE=0;ebNE<nExteriorElementBoundaries_global;ebNE++)
    {
      for(k=0;k<nQuadraturePoints_elementBoundary;k++)
        {
	  flux_mass[ebNE*nQuadraturePoints_elementBoundary+
                    k]  = 0.0;
	  dflux_mass_du[ebNE*nQuadraturePoints_elementBoundary+
                    k]  = 0.0;
	  dflux_mass_dv[ebNE*nQuadraturePoints_elementBoundary+
                    k]  = 0.0;
	  dflux_mass_dw[ebNE*nQuadraturePoints_elementBoundary+
                    k]  = 0.0;

	  flux_umom[ebNE*nQuadraturePoints_elementBoundary+
                    k] = 0.0;
	  dflux_umom_dp[ebNE*nQuadraturePoints_elementBoundary+
                        k] = 0.0;
	  dflux_umom_du[ebNE*nQuadraturePoints_elementBoundary+
                    k] = 0.0;
	  dflux_umom_dv[ebNE*nQuadraturePoints_elementBoundary+
                    k] = 0.0;
	  dflux_umom_dw[ebNE*nQuadraturePoints_elementBoundary+
                    k] = 0.0;

	  flux_vmom[ebNE*nQuadraturePoints_elementBoundary+
                    k] = 0.0;
	  dflux_vmom_dp[ebNE*nQuadraturePoints_elementBoundary+
                    k] = 0.0;
	  dflux_vmom_du[ebNE*nQuadraturePoints_elementBoundary+
                    k] = 0.0;
	  dflux_vmom_dv[ebNE*nQuadraturePoints_elementBoundary+
                    k] = 0.0;
	  dflux_vmom_dw[ebNE*nQuadraturePoints_elementBoundary+
                    k] = 0.0;

	  flux_wmom[ebNE*nQuadraturePoints_elementBoundary+
                    k] = 0.0;
	  dflux_wmom_dp[ebNE*nQuadraturePoints_elementBoundary+
                    k] = 0.0;
	  dflux_wmom_du[ebNE*nQuadraturePoints_elementBoundary+
                    k] = 0.0;
	  dflux_wmom_dv[ebNE*nQuadraturePoints_elementBoundary+
                    k] = 0.0;
	  dflux_wmom_dw[ebNE*nQuadraturePoints_elementBoundary+
                    k] = 0.0;

          flowDirection=n[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                          k*nSpace+
                          0]
            *
            f_mass[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                   k*nSpace+
                   0]
            +
	    n[ebNE*nQuadraturePoints_elementBoundary*nSpace+
	      k*nSpace+
	      1]
            *
            f_mass[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                   k*nSpace+
                   1]+
	    n[ebNE*nQuadraturePoints_elementBoundary*nSpace+
	      k*nSpace+
	      2]
            *
            f_mass[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                   k*nSpace+
                   2];
          if (isDOFBoundary_u[ebNE*nQuadraturePoints_elementBoundary+k] != 1)
            {
              flux_mass[ebNE*nQuadraturePoints_elementBoundary+
                        k]
                +=
                n[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                  k*nSpace+
                  0]
                *
                f_mass[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                       k*nSpace+
                       0];
	      velocity[ebNE*nQuadraturePoints_elementBoundary*nSpace+
		       k*nSpace+
		       0]
		=
		f_mass[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                       k*nSpace+
                       0];
              dflux_mass_du[ebNE*nQuadraturePoints_elementBoundary+
                            k]
                +=
                n[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                  k*nSpace+
                  0]
                *
                df_mass_du[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                           k*nSpace+
                           0];
              if (flowDirection >= 0.0)
                {
                  flux_umom[ebNE*nQuadraturePoints_elementBoundary+
                            k]
		    +=
                    n[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                      k*nSpace+
                      0]
                    *
                    f_umom[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                           k*nSpace+
                           0];
                  dflux_umom_du[ebNE*nQuadraturePoints_elementBoundary+
                                k]
                    +=
                    n[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                      k*nSpace+
                      0]
                    *
                    df_umom_du[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                               k*nSpace+
                               0];
                  flux_vmom[ebNE*nQuadraturePoints_elementBoundary+
                            k] 
		    +=
                    n[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                      k*nSpace+
                      0]
                    *
                    f_vmom[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                           k*nSpace+
                           0];
                  dflux_vmom_du[ebNE*nQuadraturePoints_elementBoundary+
                                k]
                    +=
                    n[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                      k*nSpace+
                      0]
                    *
                    df_vmom_du[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                               k*nSpace+
                               0];
                  dflux_vmom_dv[ebNE*nQuadraturePoints_elementBoundary+
                                k]
                    +=
                    n[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                      k*nSpace+
                      0]
                    *
                    df_vmom_dv[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                               k*nSpace+
                               0];
                  flux_wmom[ebNE*nQuadraturePoints_elementBoundary+
                            k] 
		    +=
                    n[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                      k*nSpace+
                      0]
                    *
                    f_wmom[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                           k*nSpace+
                           0];
                  dflux_wmom_du[ebNE*nQuadraturePoints_elementBoundary+
                                k]
                    +=
                    n[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                      k*nSpace+
                      0]
                    *
                    df_wmom_du[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                               k*nSpace+
                               0];
                  dflux_wmom_dw[ebNE*nQuadraturePoints_elementBoundary+
                                k]
                    +=
                    n[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                      k*nSpace+
                      0]
                    *
                    df_wmom_dw[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                               k*nSpace+
                               0];
               }
            }
          else
            {
              flux_mass[ebNE*nQuadraturePoints_elementBoundary+
                        k]
		+=
                n[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                  k*nSpace+
		  0]
                *
                bc_f_mass[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                          k*nSpace+
                          0];
	      velocity[ebNE*nQuadraturePoints_elementBoundary*nSpace+
		       k*nSpace+
		       0]
		=
		bc_f_mass[ebNE*nQuadraturePoints_elementBoundary*nSpace+
			  k*nSpace+
			  0];
              flux_umom[ebNE*nQuadraturePoints_elementBoundary+
                        k]
		+=
                n[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                  k*nSpace+
		  0]
                *
                bc_f_umom[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                          k*nSpace+
                          0];
              flux_vmom[ebNE*nQuadraturePoints_elementBoundary+
                        k]
		+=
                n[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                  k*nSpace+
		  0]
                *
                bc_f_vmom[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                          k*nSpace+
                          0];
              if (isDOFBoundary_v[ebNE*nQuadraturePoints_elementBoundary+k] != 1)
                dflux_vmom_dv[ebNE*nQuadraturePoints_elementBoundary+
                              k]
                  +=
                  n[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                    k*nSpace+
                    0]
                  *
                  df_vmom_dv[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                             k*nSpace+
                             0];
              flux_wmom[ebNE*nQuadraturePoints_elementBoundary+
                        k]
		+=
                n[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                  k*nSpace+
		  0]
                *
                bc_f_wmom[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                          k*nSpace+
                          0];
              if (isDOFBoundary_w[ebNE*nQuadraturePoints_elementBoundary+k] != 1)
                dflux_wmom_dw[ebNE*nQuadraturePoints_elementBoundary+
                              k]
                  +=
                  n[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                    k*nSpace+
                    0]
                  *
                  df_wmom_dw[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                             k*nSpace+
                             0];
            }
          if (isDOFBoundary_v[ebNE*nQuadraturePoints_elementBoundary+k] != 1)
            {
              flux_mass[ebNE*nQuadraturePoints_elementBoundary+
                        k]
		+=
                n[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                  k*nSpace+
		  1]
                *
                f_mass[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                       k*nSpace+
                       1];
	      velocity[ebNE*nQuadraturePoints_elementBoundary*nSpace+
		       k*nSpace+
		       1]
		=
		f_mass[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                       k*nSpace+
                       1];
              dflux_mass_dv[ebNE*nQuadraturePoints_elementBoundary+
                            k]
                +=
                n[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                  k*nSpace+
                  1]
                *
                df_mass_dv[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                           k*nSpace+
                           1];
              if (flowDirection >= 0.0)
                {
                  flux_umom[ebNE*nQuadraturePoints_elementBoundary+
                            k]
		    +=
                    n[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                      k*nSpace+
                      1]
                    *
                    f_umom[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                           k*nSpace+
                           1];
                  dflux_umom_du[ebNE*nQuadraturePoints_elementBoundary+
                                k]
                    +=
                    n[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                      k*nSpace+
                      1]
                    *
                    df_umom_du[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                               k*nSpace+
                               1];
                  dflux_umom_dv[ebNE*nQuadraturePoints_elementBoundary+
                                k]
                    +=
                    n[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                      k*nSpace+
                      1]
                    *
                    df_umom_dv[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                               k*nSpace+
                               1];
                  flux_vmom[ebNE*nQuadraturePoints_elementBoundary+
                            k]
		    +=
                    n[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                      k*nSpace+
                      1]
                    *
                    f_vmom[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                           k*nSpace+
                           1];
                  dflux_vmom_dv[ebNE*nQuadraturePoints_elementBoundary+
                                k]
                    += n[ebNE*nQuadraturePoints_elementBoundary*nSpace+
			 k*nSpace+
			 1]
                    *
                    df_vmom_dv[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                               k*nSpace+
                               1];
                  flux_wmom[ebNE*nQuadraturePoints_elementBoundary+
                            k]
		    +=
                    n[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                      k*nSpace+
                      1]
                    *
                    f_wmom[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                           k*nSpace+
                           1];
                  dflux_wmom_dw[ebNE*nQuadraturePoints_elementBoundary+
                                k]
                    +=
                    n[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                      k*nSpace+
                      1]
                    *
                    df_wmom_dw[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                               k*nSpace+
                               1];
                  dflux_wmom_dv[ebNE*nQuadraturePoints_elementBoundary+
                                k]
                    +=
                    n[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                      k*nSpace+
                      1]
                    *
                    df_wmom_dv[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                               k*nSpace+
                               1];
                }
            }
          else
            {
              flux_mass[ebNE*nQuadraturePoints_elementBoundary+
                        k]
		+=
                n[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                  k*nSpace+
		  1]
                *
                bc_f_mass[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                          k*nSpace+
                          1];
              velocity[ebNE*nQuadraturePoints_elementBoundary*nSpace+
		       k*nSpace+
		       1]
		=
                bc_f_mass[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                          k*nSpace+
                          1];
              flux_umom[ebNE*nQuadraturePoints_elementBoundary+
                        k]
		+=
                n[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                  k*nSpace+
		  1]
                *
                bc_f_umom[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                          k*nSpace+
                          1];
              if (isDOFBoundary_u[ebNE*nQuadraturePoints_elementBoundary+k] != 1)
                dflux_umom_du[ebNE*nQuadraturePoints_elementBoundary+
                              k]
                  +=
                  n[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                    k*nSpace+
                    1]
                  *
                  df_umom_du[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                             k*nSpace+
                             1];
              flux_vmom[ebNE*nQuadraturePoints_elementBoundary+
                        k]
		+=
                n[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                  k*nSpace+
		  1]
                *
                bc_f_vmom[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                          k*nSpace+
                          1];
              flux_wmom[ebNE*nQuadraturePoints_elementBoundary+
                        k]
		+=
                n[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                  k*nSpace+
		  1]
                *
                bc_f_wmom[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                          k*nSpace+
                          1];
              if (isDOFBoundary_w[ebNE*nQuadraturePoints_elementBoundary+k] != 1)
                dflux_wmom_dw[ebNE*nQuadraturePoints_elementBoundary+
                              k]
                  +=
                  n[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                    k*nSpace+
                    1]
                  *
                  df_wmom_dw[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                             k*nSpace+
                             1];
            }
          if (isDOFBoundary_w[ebNE*nQuadraturePoints_elementBoundary+k] != 1)
            {
              flux_mass[ebNE*nQuadraturePoints_elementBoundary+
                        k]
		+=
                n[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                  k*nSpace+
		  2]
                *
                f_mass[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                       k*nSpace+
                       2];
              velocity[ebNE*nQuadraturePoints_elementBoundary*nSpace+
		       k*nSpace+
		       2]
		=
                f_mass[ebNE*nQuadraturePoints_elementBoundary*nSpace+
		       k*nSpace+
		       2];
              dflux_mass_dw[ebNE*nQuadraturePoints_elementBoundary+
                            k]
                +=
                n[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                  k*nSpace+
                  2]
                *
                df_mass_dw[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                           k*nSpace+
                           2];
              if (flowDirection >= 0.0)
                {
                  flux_umom[ebNE*nQuadraturePoints_elementBoundary+
                            k]
		    +=
                    n[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                      k*nSpace+
                      2]
                    *
                    f_umom[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                           k*nSpace+
                           2];
                  dflux_umom_du[ebNE*nQuadraturePoints_elementBoundary+
                                k]
                    +=
                    n[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                      k*nSpace+
                      2]
                    *
                    df_umom_du[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                               k*nSpace+
                               2];
                  dflux_umom_dw[ebNE*nQuadraturePoints_elementBoundary+
                                k]
                    +=
                    n[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                      k*nSpace+
                      2]
                    *
                    df_umom_dw[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                               k*nSpace+
                               2];
                  flux_vmom[ebNE*nQuadraturePoints_elementBoundary+
                            k]
		    +=
                    n[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                      k*nSpace+
                      2]
                    *
                    f_vmom[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                           k*nSpace+
                           2];
                  dflux_vmom_dv[ebNE*nQuadraturePoints_elementBoundary+
                                k]
                    += n[ebNE*nQuadraturePoints_elementBoundary*nSpace+
			 k*nSpace+
			 2]
                    *
                    df_vmom_dv[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                               k*nSpace+
                               2];
                  dflux_vmom_dw[ebNE*nQuadraturePoints_elementBoundary+
                                k]
                    += n[ebNE*nQuadraturePoints_elementBoundary*nSpace+
			 k*nSpace+
			 2]
                    *
                    df_vmom_dw[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                               k*nSpace+
                               2];
                  flux_wmom[ebNE*nQuadraturePoints_elementBoundary+
                            k]
		    +=
                    n[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                      k*nSpace+
                      2]
                    *
                    f_wmom[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                           k*nSpace+
                           2];
                  dflux_wmom_dw[ebNE*nQuadraturePoints_elementBoundary+
                                k]
                    +=
                    n[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                      k*nSpace+
                      2]
                    *
                    df_wmom_dw[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                               k*nSpace+
                               2];
                }
            }
          else
            {
              flux_mass[ebNE*nQuadraturePoints_elementBoundary+
                        k]
		+=
                n[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                  k*nSpace+
		  2]
                *
                bc_f_mass[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                          k*nSpace+
                          2];
              velocity[ebNE*nQuadraturePoints_elementBoundary*nSpace+
		       k*nSpace+
		       2]
		=
                bc_f_mass[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                          k*nSpace+
                          2];
              flux_umom[ebNE*nQuadraturePoints_elementBoundary+
                        k]
		+=
                n[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                  k*nSpace+
		  2]
                *
                bc_f_umom[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                          k*nSpace+
                          2];
              if (isDOFBoundary_u[ebNE*nQuadraturePoints_elementBoundary+k] != 1)
                dflux_umom_du[ebNE*nQuadraturePoints_elementBoundary+
                              k]
                  +=
                  n[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                    k*nSpace+
                    2]
                  *
                  df_umom_du[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                             k*nSpace+
                             2];
              flux_vmom[ebNE*nQuadraturePoints_elementBoundary+
                        k]
		+=
                n[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                  k*nSpace+
		  2]
                *
                bc_f_vmom[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                          k*nSpace+
                          2];
              if (isDOFBoundary_v[ebNE*nQuadraturePoints_elementBoundary+k] != 1)
                dflux_vmom_dv[ebNE*nQuadraturePoints_elementBoundary+
                              k]
                  +=
                  n[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                    k*nSpace+
                    2]
                  *
                  df_vmom_dv[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                             k*nSpace+
                             2];
              flux_wmom[ebNE*nQuadraturePoints_elementBoundary+
                        k]
		+=
                n[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                  k*nSpace+
		  2]
                *
                bc_f_wmom[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                          k*nSpace+
                          2];
            }
          if (isDOFBoundary_p[ebNE*nQuadraturePoints_elementBoundary+k] == 1)
            {
              flux_umom[ebNE*nQuadraturePoints_elementBoundary+
                        k]
                +=
                n[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                  k*nSpace+
		  0]
                *
                (bc_p[ebNE*nQuadraturePoints_elementBoundary+
                      k]
                 -
                 p[ebNE*nQuadraturePoints_elementBoundary+
                   k]);
              dflux_umom_dp[ebNE*nQuadraturePoints_elementBoundary+
                            k]
                = -n[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                     k*nSpace+
                     0];
              flux_vmom[ebNE*nQuadraturePoints_elementBoundary+
                        k]
		+= n[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                     k*nSpace+
                     1]
                *
                (bc_p[ebNE*nQuadraturePoints_elementBoundary+
                      k]
                 -
                 p[ebNE*nQuadraturePoints_elementBoundary+
                   k]);
              dflux_vmom_dp[ebNE*nQuadraturePoints_elementBoundary+
                            k]
                = -n[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                     k*nSpace+
                     1];
              flux_wmom[ebNE*nQuadraturePoints_elementBoundary+
                        k]
		+= n[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                     k*nSpace+
                     2]
                *
                (bc_p[ebNE*nQuadraturePoints_elementBoundary+
                      k]
                 -
                 p[ebNE*nQuadraturePoints_elementBoundary+
                   k]);
              dflux_wmom_dp[ebNE*nQuadraturePoints_elementBoundary+
                            k]
                = -n[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                     k*nSpace+
                     2];
            }
        }
    }
}
/**
   \brief Calculate the advective flux at at exterior element boundaries
*/
void calculateGlobalExteriorNumericalAdvectiveFluxStokesP2D(proteus_index_t nExteriorElementBoundaries_global,
							   int nQuadraturePoints_elementBoundary,
							   int nSpace,
							   proteus_index_t* exteriorElementBoundaries,
							   proteus_index_t* elementBoundaryElements,
							   int* elementBoundaryLocalElementBoundaries,
							   int *isDOFBoundary_p,
							   int *isDOFBoundary_u,
							   int *isDOFBoundary_v,
							   double* n,
							   double* bc_f,
							   double* bc_fpu,
							   double* bc_fpv,
							   double* f,
							   double* fpu,
							   double* fpv,
							   double* df_du,
							   double* df_dv,
							   double* dfpu_dp,
							   double* dfpv_dp,
							   double* flux,
							   double* fluxpu,
							   double* fluxpv,
							   double* dflux_du,
							   double* dflux_dv,
							   double* dfluxpu_dp,
							   double* dfluxpv_dp)
{
  int ebNE,k;
  for(ebNE=0;ebNE<nExteriorElementBoundaries_global;ebNE++)
    {
      for(k=0;k<nQuadraturePoints_elementBoundary;k++)
        {
	  fluxpu[ebNE*nQuadraturePoints_elementBoundary+
		 k] = 0.0;
	  fluxpv[ebNE*nQuadraturePoints_elementBoundary+
		 k] = 0.0;
	  flux[ebNE*nQuadraturePoints_elementBoundary+
	       k]  = 0.0;

          //u and v momentum fluxes due to pressure
          if (isDOFBoundary_p[ebNE*nQuadraturePoints_elementBoundary+k] != 1)
            {
              dfluxpu_dp[ebNE*nQuadraturePoints_elementBoundary+
                         k]
                = 
                n[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                  k*nSpace+
                  0]
                *
                dfpu_dp[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                        k*nSpace+
                        0];
              fluxpu[ebNE*nQuadraturePoints_elementBoundary+
                     k]+= 
                n[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                  k*nSpace+
		  0]
                *
                fpu[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                    k*nSpace+
                    0];
              dfluxpv_dp[ebNE*nQuadraturePoints_elementBoundary+
                         k]
                = 
                n[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                  k*nSpace+
                  1]
                *
                dfpv_dp[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                        k*nSpace+
                        1];
              fluxpv[ebNE*nQuadraturePoints_elementBoundary+
                     k]+= 
                n[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                  k*nSpace+
		  1]
                *
                fpv[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                    k*nSpace+
                    1];
            }
          else
            {
              fluxpu[ebNE*nQuadraturePoints_elementBoundary+
                     k]+= 
                n[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                  k*nSpace+
		  0]
                *
                bc_fpu[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                       k*nSpace+
                       0];
              fluxpv[ebNE*nQuadraturePoints_elementBoundary+
                     k]+= 
                n[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                  k*nSpace+
		  1]
                *
                bc_fpv[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                       k*nSpace+
                       1];
            }
          //mass flux
          if (isDOFBoundary_u[ebNE*nQuadraturePoints_elementBoundary+k] != 1)
            {
              dflux_du[ebNE*nQuadraturePoints_elementBoundary+
                       k]
                = 
                n[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                  k*nSpace+
                  0]
                *
                df_du[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                      k*nSpace+
                      0];
              flux[ebNE*nQuadraturePoints_elementBoundary+
                   k]+= 
                n[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                  k*nSpace+
		  0]
                *
                f[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                  k*nSpace+
                  0];
            }
          else
            {
              flux[ebNE*nQuadraturePoints_elementBoundary+
                   k]+= 
                n[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                  k*nSpace+
		  0]
                *
                bc_f[ebNE*nQuadraturePoints_elementBoundary*nSpace+
		     k*nSpace+
                     0];
            }
          if (isDOFBoundary_v[ebNE*nQuadraturePoints_elementBoundary+k] != 1)
            {
              dflux_dv[ebNE*nQuadraturePoints_elementBoundary+
                       k]
                = 
                n[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                  k*nSpace+
                  1]
                *
                df_dv[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                      k*nSpace+
                      1];
              flux[ebNE*nQuadraturePoints_elementBoundary+
                   k]+= 
                n[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                  k*nSpace+
		  1]
                *
                f[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                  k*nSpace+
                  1];
            }
          else
            {
              flux[ebNE*nQuadraturePoints_elementBoundary+
                   k]+= 
                n[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                  k*nSpace+
		  1]
                *
                bc_f[ebNE*nQuadraturePoints_elementBoundary*nSpace+
		     k*nSpace+
                     1];
            }
        }
    }
}
/**
   \brief Calculate the advective flux at at exterior element boundaries
*/
void calculateExteriorNumericalAdvectiveFluxStokesP3D(proteus_index_t nExteriorElementBoundaries_global,
						     int nElementBoundaries_element,
                                                     int nQuadraturePoints_elementBoundary,
                                                     int nSpace,
                                                     proteus_index_t* exteriorElementBoundaries,
                                                     proteus_index_t* elementBoundaryElements,
                                                     int* elementBoundaryLocalElementBoundaries,
                                                     int *isDOFBoundary_p,
                                                     int *isDOFBoundary_u,
                                                     int *isDOFBoundary_v,
                                                     int *isDOFBoundary_w,
                                                     double* n,
                                                     double* bc_f,
                                                     double* bc_fpu,
                                                     double* bc_fpv,
                                                     double* bc_fpw,
                                                     double* f,
                                                     double* fpu,
                                                     double* fpv,
                                                     double* fpw,
                                                     double* df_du,
                                                     double* df_dv,
                                                     double* df_dw,
                                                     double* dfpu_dp,
                                                     double* dfpv_dp,
                                                     double* dfpw_dp,
                                                     double* flux,
                                                     double* fluxpu,
                                                     double* fluxpv,
                                                     double* fluxpw,
                                                     double* dflux_du,
                                                     double* dflux_dv,
                                                     double* dflux_dw,
                                                     double* dfluxpu_dp,
                                                     double* dfluxpv_dp,
						     double* dfluxpw_dp)
{
  proteus_index_t ebN, eN_global; int ebNE, ebN_element, k;
  for(ebNE=0;ebNE<nExteriorElementBoundaries_global;ebNE++)
    {
      ebN = exteriorElementBoundaries[ebNE];
      eN_global = elementBoundaryElements[ebN*2+0];
      ebN_element = elementBoundaryLocalElementBoundaries[ebN*2+0];
      for(k=0;k<nQuadraturePoints_elementBoundary;k++)
        {
	  fluxpu[ebN*nQuadraturePoints_elementBoundary+
		 k] = 0.0;
	  fluxpv[ebN*nQuadraturePoints_elementBoundary+
		 k] = 0.0;
	  fluxpw[ebN*nQuadraturePoints_elementBoundary+
		 k] = 0.0;
	  flux[ebN*nQuadraturePoints_elementBoundary+
	       k]  = 0.0;

          //u, v and w momentum fluxes due to pressure
          if (isDOFBoundary_p[ebNE*nQuadraturePoints_elementBoundary+k] != 1)
            {
              dfluxpu_dp[ebN*nQuadraturePoints_elementBoundary+
                         k]
                = 
                n[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                  ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                  k*nSpace+
                  0]
                *
                dfpu_dp[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                        ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                        k*nSpace+
                        0];
              fluxpu[ebN*nQuadraturePoints_elementBoundary+
                     k]+= 
                n[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                  ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                  k*nSpace+
		  0]
                *
                fpu[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                    ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                    k*nSpace+
                    0];
              dfluxpv_dp[ebN*nQuadraturePoints_elementBoundary+
                         k]
                = 
                n[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                  ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                  k*nSpace+
                  1]
                *
                dfpv_dp[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                        ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                        k*nSpace+
                        1];
              fluxpv[ebN*nQuadraturePoints_elementBoundary+
                     k]+= 
                n[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                  ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                  k*nSpace+
		  1]
                *
                fpv[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                    ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                    k*nSpace+
                    1];
              dfluxpw_dp[ebN*nQuadraturePoints_elementBoundary+
                         k]
                = 
                n[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                  ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                  k*nSpace+
                  2]
                *
                dfpw_dp[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                        ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                        k*nSpace+
                        2];
              fluxpw[ebN*nQuadraturePoints_elementBoundary+
                     k]+= 
                n[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                  ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                  k*nSpace+
		  2]
                *
                fpw[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                    ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                    k*nSpace+
                    2];
            }
          else
            {
              fluxpu[ebN*nQuadraturePoints_elementBoundary+
                     k]+= 
                n[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                  ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                  k*nSpace+
		  0]
                *
                bc_fpu[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                       k*nSpace+
                       0];
              fluxpv[ebN*nQuadraturePoints_elementBoundary+
                     k]+= 
                n[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                  ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                  k*nSpace+
		  1]
                *
                bc_fpv[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                       k*nSpace+
                       1];
              fluxpw[ebN*nQuadraturePoints_elementBoundary+
                     k]+= 
                n[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                  ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                  k*nSpace+
		  2]
                *
                bc_fpw[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                       k*nSpace+
                       2];
             }
          //mass flux
          if (isDOFBoundary_u[ebNE*nQuadraturePoints_elementBoundary+k] != 1)
            {
              dflux_du[ebN*nQuadraturePoints_elementBoundary+
                       k]
                = 
                n[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                  ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                  k*nSpace+
                  0]
                *
                df_du[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                      ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                      k*nSpace+
                      0];
              flux[ebN*nQuadraturePoints_elementBoundary+
                   k]+= 
                n[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                  ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                  k*nSpace+
		  0]
                *
                f[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                  ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                  k*nSpace+
                  0];
            }
          else
            {
              flux[ebN*nQuadraturePoints_elementBoundary+
                   k]+= 
                n[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                  ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                  k*nSpace+
		  0]
                *
                bc_f[ebNE*nQuadraturePoints_elementBoundary*nSpace+
		     k*nSpace+
                     0];
            }
          if (isDOFBoundary_v[ebNE*nQuadraturePoints_elementBoundary+k] != 1)
            {
              dflux_dv[ebN*nQuadraturePoints_elementBoundary+
                       k]
                = 
                n[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                  ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                  k*nSpace+
                  1]
                *
                df_dv[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                      ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                      k*nSpace+
                      1];
              flux[ebN*nQuadraturePoints_elementBoundary+
                   k]+= 
                n[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                  ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                  k*nSpace+
		  1]
                *
                f[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                  ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                  k*nSpace+
                  1];
            }
          else
            {
              flux[ebN*nQuadraturePoints_elementBoundary+
                   k]+= 
                n[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                  ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                  k*nSpace+
		  1]
                *
                bc_f[ebNE*nQuadraturePoints_elementBoundary*nSpace+
		     k*nSpace+
                     1];
            }
          if (isDOFBoundary_w[ebNE*nQuadraturePoints_elementBoundary+k] != 1)
            {
              dflux_dw[ebN*nQuadraturePoints_elementBoundary+
                       k]
                = 
                n[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                  ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                  k*nSpace+
                  2]
                *
                df_dw[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                      ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                      k*nSpace+
                      2];
              flux[ebN*nQuadraturePoints_elementBoundary+
                   k]+= 
                n[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                  ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                  k*nSpace+
		  2]
                *
                f[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                  ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                  k*nSpace+
                  2];
            }
          else
            {
              flux[ebN*nQuadraturePoints_elementBoundary+
                   k]+= 
                n[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                  ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                  k*nSpace+
		  2]
                *
                bc_f[ebNE*nQuadraturePoints_elementBoundary*nSpace+
		     k*nSpace+
                     2];
            }
        }
    }
}
/**
   \brief Calculate the advective flux at at exterior element boundaries
*/
void calculateGlobalExteriorNumericalAdvectiveFluxStokesP3D(proteus_index_t nExteriorElementBoundaries_global,
							   int nQuadraturePoints_elementBoundary,
							   int nSpace,
							   proteus_index_t* exteriorElementBoundaries,
							   proteus_index_t* elementBoundaryElements,
							   int* elementBoundaryLocalElementBoundaries,
							   int *isDOFBoundary_p,
							   int *isDOFBoundary_u,
							   int *isDOFBoundary_v,
							   int *isDOFBoundary_w,
							   double* n,
							   double* bc_f,
							   double* bc_fpu,
							   double* bc_fpv,
							   double* bc_fpw,
							   double* f,
							   double* fpu,
							   double* fpv,
							   double* fpw,
							   double* df_du,
							   double* df_dv,
							   double* df_dw,
							   double* dfpu_dp,
							   double* dfpv_dp,
							   double* dfpw_dp,
							   double* flux,
							   double* fluxpu,
							   double* fluxpv,
							   double* fluxpw,
							   double* dflux_du,
							   double* dflux_dv,
							   double* dflux_dw,
							   double* dfluxpu_dp,
							   double* dfluxpv_dp,
							   double* dfluxpw_dp)
{
  int ebNE,k;
  for(ebNE=0;ebNE<nExteriorElementBoundaries_global;ebNE++)
    {
      for(k=0;k<nQuadraturePoints_elementBoundary;k++)
        {
	  fluxpu[ebNE*nQuadraturePoints_elementBoundary+
		 k] = 0.0;
	  fluxpv[ebNE*nQuadraturePoints_elementBoundary+
		 k] = 0.0;
	  fluxpw[ebNE*nQuadraturePoints_elementBoundary+
		 k] = 0.0;
	  flux[ebNE*nQuadraturePoints_elementBoundary+
	       k]  = 0.0;

          //u, v and w momentum fluxes due to pressure
          if (isDOFBoundary_p[ebNE*nQuadraturePoints_elementBoundary+k] != 1)
            {
              dfluxpu_dp[ebNE*nQuadraturePoints_elementBoundary+
                         k]
                = 
                n[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                  k*nSpace+
                  0]
                *
                dfpu_dp[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                        k*nSpace+
                        0];
              fluxpu[ebNE*nQuadraturePoints_elementBoundary+
                     k]+= 
                n[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                  k*nSpace+
		  0]
                *
                fpu[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                    k*nSpace+
                    0];
              dfluxpv_dp[ebNE*nQuadraturePoints_elementBoundary+
                         k]
                = 
                n[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                  k*nSpace+
                  1]
                *
                dfpv_dp[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                        k*nSpace+
                        1];
              fluxpv[ebNE*nQuadraturePoints_elementBoundary+
                     k]+= 
                n[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                  k*nSpace+
		  1]
                *
                fpv[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                    k*nSpace+
                    1];
              dfluxpw_dp[ebNE*nQuadraturePoints_elementBoundary+
                         k]
                = 
                n[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                  k*nSpace+
                  2]
                *
                dfpw_dp[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                        k*nSpace+
                        2];
              fluxpw[ebNE*nQuadraturePoints_elementBoundary+
                     k]+= 
                n[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                  k*nSpace+
		  2]
                *
                fpw[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                    k*nSpace+
                    2];
            }
          else
            {
              fluxpu[ebNE*nQuadraturePoints_elementBoundary+
                     k]+= 
                n[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                  k*nSpace+
		  0]
                *
                bc_fpu[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                       k*nSpace+
                       0];
              fluxpv[ebNE*nQuadraturePoints_elementBoundary+
                     k]+= 
                n[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                  k*nSpace+
		  1]
                *
                bc_fpv[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                       k*nSpace+
                       1];
              fluxpw[ebNE*nQuadraturePoints_elementBoundary+
                     k]+= 
                n[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                  k*nSpace+
		  2]
                *
                bc_fpw[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                       k*nSpace+
                       2];
             }
          //mass flux
          if (isDOFBoundary_u[ebNE*nQuadraturePoints_elementBoundary+k] != 1)
            {
              dflux_du[ebNE*nQuadraturePoints_elementBoundary+
                       k]
                = 
                n[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                  k*nSpace+
                  0]
                *
                df_du[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                      k*nSpace+
                      0];
              flux[ebNE*nQuadraturePoints_elementBoundary+
                   k]+= 
                n[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                  k*nSpace+
		  0]
                *
                f[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                  k*nSpace+
                  0];
            }
          else
            {
              flux[ebNE*nQuadraturePoints_elementBoundary+
                   k]+= 
                n[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                  k*nSpace+
		  0]
                *
                bc_f[ebNE*nQuadraturePoints_elementBoundary*nSpace+
		     k*nSpace+
                     0];
            }
          if (isDOFBoundary_v[ebNE*nQuadraturePoints_elementBoundary+k] != 1)
            {
              dflux_dv[ebNE*nQuadraturePoints_elementBoundary+
                       k]
                = 
                n[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                  k*nSpace+
                  1]
                *
                df_dv[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                      k*nSpace+
                      1];
              flux[ebNE*nQuadraturePoints_elementBoundary+
                   k]+= 
                n[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                  k*nSpace+
		  1]
                *
                f[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                  k*nSpace+
                  1];
            }
          else
            {
              flux[ebNE*nQuadraturePoints_elementBoundary+
                   k]+= 
                n[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                  k*nSpace+
		  1]
                *
                bc_f[ebNE*nQuadraturePoints_elementBoundary*nSpace+
		     k*nSpace+
                     1];
            }
          if (isDOFBoundary_w[ebNE*nQuadraturePoints_elementBoundary+k] != 1)
            {
              dflux_dw[ebNE*nQuadraturePoints_elementBoundary+
                       k]
                = 
                n[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                  k*nSpace+
                  2]
                *
                df_dw[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                      k*nSpace+
                      2];
              flux[ebNE*nQuadraturePoints_elementBoundary+
                   k]+= 
                n[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                  k*nSpace+
		  2]
                *
                f[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                  k*nSpace+
                  2];
            }
          else
            {
              flux[ebNE*nQuadraturePoints_elementBoundary+
                   k]+= 
                n[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                  k*nSpace+
		  2]
                *
                bc_f[ebNE*nQuadraturePoints_elementBoundary*nSpace+
		     k*nSpace+
                     2];
            }
        }
    }
}

/**
   \brief Apply basic pressure boundary penalty term for Stokes
   \todo add penalty coefficient?
 */
void calculateGlobalExteriorNumericalAdvectiveFluxStokes2D(proteus_index_t nExteriorElementBoundaries_global,
							   int nQuadraturePoints_elementBoundary,
							   int nSpace,
							   proteus_index_t* exteriorElementBoundaries,
							   proteus_index_t* elementBoundaryElements,
							   int* elementBoundaryLocalElementBoundaries,
							   int *isDOFBoundary_p,
							   int *isDOFBoundary_u,
							   int *isDOFBoundary_v,
							   double* n,
							   double* bc_p,
							   double* bc_f_mass,
							   double* p,
							   double* f_mass,
							   double* df_mass_du,
							   double* df_mass_dv,
							   double* flux_mass,
							   double* flux_umom,
							   double* flux_vmom,
							   double* dflux_mass_du,
							   double* dflux_mass_dv,
							   double* dflux_umom_dp,
							   double* dflux_vmom_dp,
                                                           double* velocity)
                                                           
{
  int ebNE,k;
  for(ebNE=0;ebNE<nExteriorElementBoundaries_global;ebNE++)
    {
      for(k=0;k<nQuadraturePoints_elementBoundary;k++)
        {
	  flux_mass[ebNE*nQuadraturePoints_elementBoundary+
                    k]  = 0.0;
	  dflux_mass_du[ebNE*nQuadraturePoints_elementBoundary+
			k]  = 0.0;
	  dflux_mass_dv[ebNE*nQuadraturePoints_elementBoundary+
			k]  = 0.0;

	  flux_umom[ebNE*nQuadraturePoints_elementBoundary+
                    k] = 0.0;
	  dflux_umom_dp[ebNE*nQuadraturePoints_elementBoundary+
                        k] = 0.0;
	  flux_vmom[ebNE*nQuadraturePoints_elementBoundary+
                    k] = 0.0;
	  dflux_vmom_dp[ebNE*nQuadraturePoints_elementBoundary+
                    k] = 0.0;

          if (isDOFBoundary_u[ebNE*nQuadraturePoints_elementBoundary+k] != 1)
            {
              velocity[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                       k*nSpace+
                       0] =
                f_mass[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                       k*nSpace+
                       0];
              flux_mass[ebNE*nQuadraturePoints_elementBoundary+
                        k]
                +=
                n[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                  k*nSpace+
                  0]
                *
                f_mass[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                       k*nSpace+
                       0];
              dflux_mass_du[ebNE*nQuadraturePoints_elementBoundary+
                            k]
                +=
                n[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                  k*nSpace+
                  0]
                *
                df_mass_du[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                           k*nSpace+
                           0];
            }
          else
            {
              velocity[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                       k*nSpace+
                       0] =
                bc_f_mass[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                          k*nSpace+
                          0];
              flux_mass[ebNE*nQuadraturePoints_elementBoundary+
                        k]
		+=
                n[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                  k*nSpace+
		  0]
                *
                bc_f_mass[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                          k*nSpace+
                          0];
            }
          if (isDOFBoundary_v[ebNE*nQuadraturePoints_elementBoundary+k] != 1)
            {
              velocity[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                       k*nSpace+
                       1] =
                f_mass[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                       k*nSpace+
                       1];
              flux_mass[ebNE*nQuadraturePoints_elementBoundary+
                        k]
		+=
                n[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                  k*nSpace+
		  1]
                *
                f_mass[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                       k*nSpace+
                       1];
              dflux_mass_dv[ebNE*nQuadraturePoints_elementBoundary+
                            k]
                +=
                n[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                  k*nSpace+
                  1]
                *
                df_mass_dv[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                           k*nSpace+
                           1];
            }
          else
            {
              velocity[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                       k*nSpace+
                       1] =
                bc_f_mass[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                          k*nSpace+
                          1];
              flux_mass[ebNE*nQuadraturePoints_elementBoundary+
                        k]
		+=
                n[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                  k*nSpace+
		  1]
                *
                bc_f_mass[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                          k*nSpace+
                          1];
            }
          if (isDOFBoundary_p[ebNE*nQuadraturePoints_elementBoundary+k] == 1)
            {
              flux_umom[ebNE*nQuadraturePoints_elementBoundary+
                        k]
                +=
                n[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                  k*nSpace+
		  0]
                *
                (bc_p[ebNE*nQuadraturePoints_elementBoundary+
                      k]
                 -
                 p[ebNE*nQuadraturePoints_elementBoundary+
                   k]);
              dflux_umom_dp[ebNE*nQuadraturePoints_elementBoundary+
                            k]
                = -n[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                     k*nSpace+
                     0];
              flux_vmom[ebNE*nQuadraturePoints_elementBoundary+
                        k]
		+= n[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                     k*nSpace+
                     1]
                *
                (bc_p[ebNE*nQuadraturePoints_elementBoundary+
                      k]
                 -
                 p[ebNE*nQuadraturePoints_elementBoundary+
                   k]);
              dflux_vmom_dp[ebNE*nQuadraturePoints_elementBoundary+
                            k]
                = -n[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                     k*nSpace+
                     1];
            }
        }
    }
}

void calculateGlobalExteriorNumericalAdvectiveFluxStokes3D(proteus_index_t nExteriorElementBoundaries_global,
							   int nQuadraturePoints_elementBoundary,
							   int nSpace,
							   proteus_index_t* exteriorElementBoundaries,
							   proteus_index_t* elementBoundaryElements,
							   int* elementBoundaryLocalElementBoundaries,
							   int *isDOFBoundary_p,
							   int *isDOFBoundary_u,
							   int *isDOFBoundary_v,
							   int *isDOFBoundary_w,
							   double* n,
							   double* bc_p,
							   double* bc_f_mass,
							   double* p,
							   double* f_mass,
							   double* df_mass_du,
							   double* df_mass_dv,
							   double* df_mass_dw,
							   double* flux_mass,
							   double* flux_umom,
							   double* flux_vmom,
							   double* flux_wmom,
							   double* dflux_mass_du,
							   double* dflux_mass_dv,
							   double* dflux_mass_dw,
							   double* dflux_umom_dp,
							   double* dflux_vmom_dp,
							   double* dflux_wmom_dp,
                                                           double* velocity)
                                                           
{
  int ebNE,k;
  for(ebNE=0;ebNE<nExteriorElementBoundaries_global;ebNE++)
    {
      for(k=0;k<nQuadraturePoints_elementBoundary;k++)
        {
	  flux_mass[ebNE*nQuadraturePoints_elementBoundary+
                    k]  = 0.0;
	  dflux_mass_du[ebNE*nQuadraturePoints_elementBoundary+
			k]  = 0.0;
	  dflux_mass_dv[ebNE*nQuadraturePoints_elementBoundary+
			k]  = 0.0;
	  dflux_mass_dw[ebNE*nQuadraturePoints_elementBoundary+
			k]  = 0.0;
          
	  flux_umom[ebNE*nQuadraturePoints_elementBoundary+
                    k] = 0.0;
	  dflux_umom_dp[ebNE*nQuadraturePoints_elementBoundary+
                        k] = 0.0;
          
	  flux_vmom[ebNE*nQuadraturePoints_elementBoundary+
                    k] = 0.0;
	  dflux_vmom_dp[ebNE*nQuadraturePoints_elementBoundary+
			k] = 0.0;
          
	  flux_wmom[ebNE*nQuadraturePoints_elementBoundary+
                    k] = 0.0;
	  dflux_wmom_dp[ebNE*nQuadraturePoints_elementBoundary+
			k] = 0.0;
          
          if (isDOFBoundary_u[ebNE*nQuadraturePoints_elementBoundary+k] != 1)
            {
              velocity[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                       k*nSpace+
                       0] =
                f_mass[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                       k*nSpace+
                       0];
              flux_mass[ebNE*nQuadraturePoints_elementBoundary+
                        k]
                +=
                n[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                  k*nSpace+
                  0]
                *
                f_mass[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                       k*nSpace+
                       0];
              dflux_mass_du[ebNE*nQuadraturePoints_elementBoundary+
                            k]
                +=
                n[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                  k*nSpace+
                  0]
                *
                df_mass_du[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                           k*nSpace+
                           0];
            }
          else
            {
              velocity[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                       k*nSpace+
                       0] =
                bc_f_mass[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                          k*nSpace+
                          0];
              flux_mass[ebNE*nQuadraturePoints_elementBoundary+
                        k]
		+=
                n[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                  k*nSpace+
		  0]
                *
                bc_f_mass[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                          k*nSpace+
                          0];
            }
          if (isDOFBoundary_v[ebNE*nQuadraturePoints_elementBoundary+k] != 1)
            {
              velocity[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                       k*nSpace+
                       1] =
                f_mass[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                       k*nSpace+
                       1];
              flux_mass[ebNE*nQuadraturePoints_elementBoundary+
                        k]
		+=
                n[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                  k*nSpace+
		  1]
                *
                f_mass[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                       k*nSpace+
                       1];
              dflux_mass_dv[ebNE*nQuadraturePoints_elementBoundary+
                            k]
                +=
                n[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                  k*nSpace+
                  1]
                *
                df_mass_dv[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                           k*nSpace+
                           1];
            }
          else
            {
              velocity[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                       k*nSpace+
                       1] =
                bc_f_mass[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                          k*nSpace+
                          1];
              flux_mass[ebNE*nQuadraturePoints_elementBoundary+
                        k]
		+=
                n[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                  k*nSpace+
		  1]
                *
                bc_f_mass[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                          k*nSpace+
                          1];
            }
          if (isDOFBoundary_w[ebNE*nQuadraturePoints_elementBoundary+k] != 1)
            {
              velocity[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                       k*nSpace+
                       2] =
                f_mass[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                       k*nSpace+
                       2];
              flux_mass[ebNE*nQuadraturePoints_elementBoundary+
                        k]
		+=
                n[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                  k*nSpace+
		  2]
                *
                f_mass[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                       k*nSpace+
                       2];
              dflux_mass_dw[ebNE*nQuadraturePoints_elementBoundary+
                            k]
                +=
                n[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                  k*nSpace+
                  2]
                *
                df_mass_dw[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                           k*nSpace+
                           2];
            }
          else
            {
              velocity[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                       k*nSpace+
                       2] =
                f_mass[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                       k*nSpace+
                       2];
              flux_mass[ebNE*nQuadraturePoints_elementBoundary+
                        k]
		+=
                n[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                  k*nSpace+
		  2]
                *
                bc_f_mass[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                          k*nSpace+
                          2];
            }
          if (isDOFBoundary_p[ebNE*nQuadraturePoints_elementBoundary+k] == 1)
            {
              flux_umom[ebNE*nQuadraturePoints_elementBoundary+
                        k]
                +=
                n[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                  k*nSpace+
		  0]
                *
                (bc_p[ebNE*nQuadraturePoints_elementBoundary+
                      k]
                 -
                 p[ebNE*nQuadraturePoints_elementBoundary+
                   k]);
              dflux_umom_dp[ebNE*nQuadraturePoints_elementBoundary+
                            k]
                = -n[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                     k*nSpace+
                     0];
              flux_vmom[ebNE*nQuadraturePoints_elementBoundary+
                        k]
		+= n[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                     k*nSpace+
                     1]
                *
                (bc_p[ebNE*nQuadraturePoints_elementBoundary+
                      k]
                 -
                 p[ebNE*nQuadraturePoints_elementBoundary+
                   k]);
              dflux_vmom_dp[ebNE*nQuadraturePoints_elementBoundary+
                            k]
                = -n[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                     k*nSpace+
                     1];
              flux_wmom[ebNE*nQuadraturePoints_elementBoundary+
                        k]
		+= n[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                     k*nSpace+
                     2]
                *
                (bc_p[ebNE*nQuadraturePoints_elementBoundary+
                      k]
                 -
                 p[ebNE*nQuadraturePoints_elementBoundary+
                   k]);
              dflux_wmom_dp[ebNE*nQuadraturePoints_elementBoundary+
                            k]
                = -n[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                     k*nSpace+
                     2];
            }
        }
    }
}


/**
   \brief Calculate the advective flux at at exterior element boundaries
*/
void calculateExteriorNumericalAdvectiveFlux_average(proteus_index_t nExteriorElementBoundaries_global,
                                                     int nElementBoundaries_element,
                                                     int nQuadraturePoints_elementBoundary,
                                                     int nSpace,
                                                     proteus_index_t* exteriorElementBoundaries,
                                                     proteus_index_t* elementBoundaryElements,
                                                     int* elementBoundaryLocalElementBoundaries,
                                                     int *isDOFBoundary,
                                                     int *inflowFlag,
                                                     double* n,
                                                     double* bc_u,
                                                     double* bc_f,
                                                     double* bc_df,
                                                     double* u,
                                                     double* f,
                                                     double* df,
                                                     double* flux,
                                                     double* dflux)
{
  proteus_index_t ebN, eN_global; int ebNE, ebN_element, k, J;
  for(ebNE=0;ebNE<nExteriorElementBoundaries_global;ebNE++)
    {
      ebN = exteriorElementBoundaries[ebNE];
      eN_global = elementBoundaryElements[ebN*2+0];
      ebN_element = elementBoundaryLocalElementBoundaries[ebN*2+0];
      for(k=0;k<nQuadraturePoints_elementBoundary;k++)
        {
          flux[ebN*nQuadraturePoints_elementBoundary+k] = 0.0;
          dflux[ebN*nQuadraturePoints_elementBoundary+k] = 0.0;
          for(J=0;J<nSpace;J++)
            {
              flux[ebN*nQuadraturePoints_elementBoundary+k] +=
                n[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                  ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                  k*nSpace+
                  J]
                *
                (f[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                   ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                   k*nSpace+
                   J]
                 +
                 bc_f[ebNE*nQuadraturePoints_elementBoundary*nSpace+
		      k*nSpace+
                      J]);
              dflux[ebN*nQuadraturePoints_elementBoundary+k] +=
                n[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                  ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                  k*nSpace+
		  J]
                *
                df[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                   ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                   k*nSpace+
		   J];
            }
/*           flux[ebN*nQuadraturePoints_elementBoundary+k] *= 0.5; */
/*           if (isDOFBoundary[ebNE*nQuadraturePoints_elementBoundary+k] == 1) */
/*             dflux[ebN*nQuadraturePoints_elementBoundary+k] *= 0.5; */
/*          for(J=0;J<nSpace;J++) */
/*             { */
/*               flux[ebN*nQuadraturePoints_elementBoundary+k] +=  */
/*                 n[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+ */
/*                   ebN_element*nQuadraturePoints_elementBoundary*nSpace+ */
/*                   k*nSpace+ */
/*                   J] */
/*                 * */
/*                 f[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+ */
/*                    ebN_element*nQuadraturePoints_elementBoundary*nSpace+ */
/*                    k*nSpace+ */
/*                   J]; */
/*             } */
        }
    }
}
/**
   \brief Calculate the advective flux at at exterior element boundaries
*/
void calculateGlobalExteriorNumericalAdvectiveFlux_average(proteus_index_t nExteriorElementBoundaries_global,
							   int nQuadraturePoints_elementBoundary,
							   int nSpace,
							   proteus_index_t* exteriorElementBoundaries,
							   proteus_index_t* elementBoundaryElements,
							   int* elementBoundaryLocalElementBoundaries,
							   int *isDOFBoundary,
							   int *inflowFlag,
							   double* n,
							   double* bc_u,
							   double* bc_f,
							   double* bc_df,
							   double* u,
							   double* f,
							   double* df,
							   double* flux,
							   double* dflux)
{
  int ebNE,k,J;
  for(ebNE=0;ebNE<nExteriorElementBoundaries_global;ebNE++)
    {
      for(k=0;k<nQuadraturePoints_elementBoundary;k++)
        {
          flux[ebNE*nQuadraturePoints_elementBoundary+k] = 0.0;
          dflux[ebNE*nQuadraturePoints_elementBoundary+k] = 0.0;
          for(J=0;J<nSpace;J++)
            {
              flux[ebNE*nQuadraturePoints_elementBoundary+k] +=
                n[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                  k*nSpace+
                  J]
                *
                (f[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                   k*nSpace+
                   J]
                 +
                 bc_f[ebNE*nQuadraturePoints_elementBoundary*nSpace+
		      k*nSpace+
                      J]);
              dflux[ebNE*nQuadraturePoints_elementBoundary+k] +=
                n[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                  k*nSpace+
		  J]
                *
                df[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                   k*nSpace+
		   J];
            }
        }
    }
}
/**
   \brief Update the advective flux at exterior inflow element boundaries.
*/
void calculateGlobalExteriorInflowNumericalAdvectiveFlux(proteus_index_t nExteriorElementBoundaries_global,
							 int nQuadraturePoints_elementBoundary,
							 int nSpace,
							 proteus_index_t* exteriorElementBoundaries,
							 proteus_index_t* elementBoundaryElements,
							 int* elementBoundaryLocalElementBoundaries,
							 int* inflowFlag,
							 double* inflowFlux,
							 double* n,
							 double* f,
							 double* df,
							 double* flux,
							 double* dflux_left)
{
  int ebNE,k;
  for(ebNE=0;ebNE<nExteriorElementBoundaries_global;ebNE++)
    {
      for(k=0;k<nQuadraturePoints_elementBoundary;k++)
        {
	  if (inflowFlag[ebNE*nQuadraturePoints_elementBoundary+k])
	    {
	      flux[ebNE*nQuadraturePoints_elementBoundary+k] = 
		inflowFlux[ebNE*nQuadraturePoints_elementBoundary+
			   k];
	      dflux_left[ebNE*nQuadraturePoints_elementBoundary+k] = 0.0;
	    }
        }
    }
}
/**
   \brief Calculate the advective flux at exterior element boundaries
*/
void updateExteriorNumericalAdvectiveFluxJacobian(proteus_index_t nExteriorElementBoundaries_global,
                                                  int nElementBoundaries_element,
                                                  int nQuadraturePoints_elementBoundary,
                                                  int nDOF_trial_element,
                                                  proteus_index_t* exteriorElementBoundaries,
                                                  proteus_index_t* elementBoundaryElements,
                                                  int* elementBoundaryLocalElementBoundaries,
                                                  int* inflowFlag,
                                                  double* dflux_left,
                                                  double* v,
                                                  double* fluxJacobian)
{
  proteus_index_t ebN, left_eN_global; int ebNE, left_ebN_element, k, j;
  for(ebNE=0;ebNE<nExteriorElementBoundaries_global;ebNE++)
    {
      ebN = exteriorElementBoundaries[ebNE];
      left_eN_global = elementBoundaryElements[ebN*2+0];
      left_ebN_element = elementBoundaryLocalElementBoundaries[ebN*2+0];
      /*mwf assume inflow boundary points have had their dflux_left zeroed?*/
      for(k=0;k<nQuadraturePoints_elementBoundary;k++)
        {
	  /*mwf only set jacobian if not on inflow? causes problems I think*/
	  /*if (!inflowFlag[ebNE*nQuadraturePoints_elementBoundary+k])*/
	  if (1)
	    {
	      for(j=0;j<nDOF_trial_element;j++)
		{
		  fluxJacobian[ebN*2*nQuadraturePoints_elementBoundary*nDOF_trial_element+
			       0*nQuadraturePoints_elementBoundary*nDOF_trial_element+
			       k*nDOF_trial_element+
			       j]
		    +=
		    dflux_left[ebN*nQuadraturePoints_elementBoundary+
			       k]
		    *
		    v[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_trial_element+
		      left_ebN_element*nQuadraturePoints_elementBoundary*nDOF_trial_element+
		      k*nDOF_trial_element+
		      j];
		}/*j*/
	    }/*if on outflow*/
	}/*k*/
    }/*ebNE*/
}
/**
   \brief Calculate the advective flux at global exterior element boundaries
*/
void updateGlobalExteriorNumericalAdvectiveFluxJacobian(proteus_index_t nExteriorElementBoundaries_global,
							int nQuadraturePoints_elementBoundary,
							int nDOF_trial_element,
							proteus_index_t* exteriorElementBoundaries,
							proteus_index_t* elementBoundaryElements,
							int* elementBoundaryLocalElementBoundaries,
							int* inflowFlag,
							double* dflux_left,
							double* v,
							double* fluxJacobian)
{
  proteus_index_t ebN, left_eN_global; int ebNE, left_ebN_element, k, j;
  for(ebNE=0;ebNE<nExteriorElementBoundaries_global;ebNE++)
    {
      ebN = exteriorElementBoundaries[ebNE];
      left_eN_global = elementBoundaryElements[ebN*2+0];
      left_ebN_element = elementBoundaryLocalElementBoundaries[ebN*2+0];
      /*mwf assume inflow boundary points have had their dflux_left zeroed?*/
      for(k=0;k<nQuadraturePoints_elementBoundary;k++)
        {
	  /*mwf only set jacobian if not on inflow? causes problems I think*/
	  /*if (!inflowFlag[ebNE*nQuadraturePoints_elementBoundary+k])*/
	  if (1)
	    {
	      for(j=0;j<nDOF_trial_element;j++)
		{
		  fluxJacobian[ebNE*nQuadraturePoints_elementBoundary*nDOF_trial_element+
			       k*nDOF_trial_element+
			       j]
		    +=
		    dflux_left[ebNE*nQuadraturePoints_elementBoundary+
			       k]
		    *
		    v[ebNE*nQuadraturePoints_elementBoundary*nDOF_trial_element+
		      k*nDOF_trial_element+
		      j];
		}/*j*/
	    }/*if on outflow*/
	}/*k*/
    }/*ebNE*/
}
void updateExteriorNumericalAdvectiveFluxJacobian_free(proteus_index_t nExteriorElementBoundaries_global,
                                                       int nElementBoundaries_element,
                                                       int nQuadraturePoints_elementBoundary,
                                                       int nDOF_trial_element,
                                                       proteus_index_t* exteriorElementBoundaries,
                                                       proteus_index_t* elementBoundaryElements,
                                                       int* elementBoundaryLocalElementBoundaries,
                                                       int* inflowFlag,
                                                       double* dflux_left,
                                                       double* v,
                                                       double* fluxJacobian)
{
  proteus_index_t ebN, left_eN_global; int ebNE, left_ebN_element, k, j;
  for(ebNE=0;ebNE<nExteriorElementBoundaries_global;ebNE++)
    {
      ebN = exteriorElementBoundaries[ebNE];
      left_eN_global = elementBoundaryElements[ebN*2+0];
      left_ebN_element = elementBoundaryLocalElementBoundaries[ebN*2+0];
      for(k=0;k<nQuadraturePoints_elementBoundary;k++)
        {
          for(j=0;j<nDOF_trial_element;j++)
            {
              fluxJacobian[ebN*2*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                           0*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                           k*nDOF_trial_element+
                           j]
                +=
                dflux_left[ebN*nQuadraturePoints_elementBoundary+
                           k]
                *
                v[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                  left_ebN_element*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                  k*nDOF_trial_element+
                  j];
            }/*j*/
	}/*k*/
    }/*ebNE*/
}
void updateGlobalExteriorNumericalAdvectiveFluxJacobian_free(proteus_index_t nExteriorElementBoundaries_global,
                                                             int nQuadraturePoints_elementBoundary,
                                                             int nDOF_trial_element,
                                                             proteus_index_t* exteriorElementBoundaries,
                                                             proteus_index_t* elementBoundaryElements,
                                                             int* elementBoundaryLocalElementBoundaries,
                                                             int* inflowFlag,
                                                             double* dflux_left,
                                                             double* v,
                                                             double* fluxJacobian)
{
  proteus_index_t ebN, left_eN_global; int ebNE, left_ebN_element, k, j;
  for(ebNE=0;ebNE<nExteriorElementBoundaries_global;ebNE++)
    {
      ebN = exteriorElementBoundaries[ebNE];
      left_eN_global = elementBoundaryElements[ebN*2+0];
      left_ebN_element = elementBoundaryLocalElementBoundaries[ebN*2+0];
      for(k=0;k<nQuadraturePoints_elementBoundary;k++)
        {
          for(j=0;j<nDOF_trial_element;j++)
            {
              fluxJacobian[ebNE*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                           k*nDOF_trial_element+
                           j]
                +=
                dflux_left[ebNE*nQuadraturePoints_elementBoundary+
                           k]
                *
                v[ebNE*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                  k*nDOF_trial_element+
                  j];
/*               fluxJacobian[ebN*2*nQuadraturePoints_elementBoundary*nDOF_trial_element+ */
/*                            0*nQuadraturePoints_elementBoundary*nDOF_trial_element+ */
/*                            k*nDOF_trial_element+ */
/*                            j] */
/*                 += */
/*                 dflux_left[ebN*nQuadraturePoints_elementBoundary+ */
/*                            k] */
/*                 * */
/*                 v[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_trial_element+ */
/*                   left_ebN_element*nQuadraturePoints_elementBoundary*nDOF_trial_element+ */
/*                   k*nDOF_trial_element+ */
/*                   j]; */
            }/*j*/
	}/*k*/
    }/*ebNE*/
}
/**
   \brief Set the advective flux boundary condition at exterior element boundaries from the current exterior flux.
*/
void setInflowFlux(proteus_index_t nExteriorElementBoundaries_global,
                   int nQuadraturePoints_elementBoundary,
                   proteus_index_t* exteriorElementBoundaries,
                   double* inflowFlux,
                   double* flux)
{
  proteus_index_t ebN; int ebNE, k;
  for(ebNE=0;ebNE<nExteriorElementBoundaries_global;ebNE++)
    {
      ebN = exteriorElementBoundaries[ebNE];
      for(k=0;k<nQuadraturePoints_elementBoundary;k++)
        inflowFlux[ebNE*nQuadraturePoints_elementBoundary+
                   k]
          =
          flux[ebN*nQuadraturePoints_elementBoundary+
               k]; 
    }
}

/*********** LDG ***********/
/**
   \brief Calculate the advective flux at at interior element boundaries
*/
void calculateInteriorNumericalDiffusiveFlux_LDG_upwind(proteus_index_t nInteriorElementBoundaries_global,
                                                        int nElementBoundaries_element,
                                                        int nQuadraturePoints_elementBoundary,
                                                        int nSpace,
                                                        proteus_index_t* interiorElementBoundaries,
                                                        proteus_index_t* elementBoundaryElements,
                                                        int* elementBoundaryLocalElementBoundaries,
                                                        double* n,
                                                        double* u,
                                                        double* a,
                                                        double* phi,
                                                        double* V,
                                                        double* penalty,
                                                        double* flux)
{
  proteus_index_t ebN, left_eN_global, right_eN_global; int ebNI, left_ebN_element, right_ebN_element, k, I, J, nSpace2=nSpace*nSpace;
  for(ebNI=0;ebNI<nInteriorElementBoundaries_global;ebNI++)
    {
      ebN = interiorElementBoundaries[ebNI];
      left_eN_global = elementBoundaryElements[ebN*2+0];
      right_eN_global = elementBoundaryElements[ebN*2+1];
      left_ebN_element = elementBoundaryLocalElementBoundaries[ebN*2+0];
      right_ebN_element = elementBoundaryLocalElementBoundaries[ebN*2+1];
      for(k=0;k<nQuadraturePoints_elementBoundary;k++)
        {
          flux[ebN*nQuadraturePoints_elementBoundary+
               k] = 0.0;
          for(I=0;I<nSpace;I++)
            for(J=0;J<nSpace;J++)
              {
                flux[ebN*nQuadraturePoints_elementBoundary+
                     k]
                  +=
                  TR_ALPHA*
                  a[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace2+
                    left_ebN_element*nQuadraturePoints_elementBoundary*nSpace2+
                    k*nSpace2+
                    I*nSpace+
                    J]
                  *
                  V[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                    left_ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                    k*nSpace+
                    J]
                  *
                  n[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                    left_ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                    k*nSpace+
                    I];
                flux[ebN*nQuadraturePoints_elementBoundary+
                     k]
                  +=
                  (1.0-TR_ALPHA)*
                  a[right_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace2+
                    right_ebN_element*nQuadraturePoints_elementBoundary*nSpace2+
                    k*nSpace2+
                    I*nSpace+
                    J]
                  *
                  V[right_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                    right_ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                    k*nSpace+
                    J]
                  *
                  n[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                    left_ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                    k*nSpace+
                    I];
              }
          flux[ebN*nQuadraturePoints_elementBoundary+
               k]
            +=
            penalty[ebN*nQuadraturePoints_elementBoundary+k]
            *
            (phi[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary+
                 left_ebN_element*nQuadraturePoints_elementBoundary+
                 k]
             -
             phi[right_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary+
                 right_ebN_element*nQuadraturePoints_elementBoundary+
                 k]);
        }
    }
}

void calculateInteriorNumericalDiffusiveFlux_LDG_upwind_sd(proteus_index_t nInteriorElementBoundaries_global,
							   int nElementBoundaries_element,
							   int nQuadraturePoints_elementBoundary,
							   int nSpace,
							   proteus_csr_index_t* rowptr,
							   proteus_csr_index_t* colind,
							   proteus_index_t* interiorElementBoundaries,
							   proteus_index_t* elementBoundaryElements,
							   int* elementBoundaryLocalElementBoundaries,
							   double* n,
							   double* u,
							   double* a,
							   double* phi,
							   double* V,
							   double* penalty,
							   double* flux)
{
  proteus_index_t ebN, left_eN_global, right_eN_global; proteus_csr_index_t m; proteus_csr_index_t nnz=rowptr[nSpace]; int ebNI, left_ebN_element, right_ebN_element, k, I;
  for(ebNI=0;ebNI<nInteriorElementBoundaries_global;ebNI++)
    {
      ebN = interiorElementBoundaries[ebNI];
      left_eN_global = elementBoundaryElements[ebN*2+0];
      right_eN_global = elementBoundaryElements[ebN*2+1];
      left_ebN_element = elementBoundaryLocalElementBoundaries[ebN*2+0];
      right_ebN_element = elementBoundaryLocalElementBoundaries[ebN*2+1];
      for(k=0;k<nQuadraturePoints_elementBoundary;k++)
        {
          flux[ebN*nQuadraturePoints_elementBoundary+
               k] = 0.0;
          for(I=0;I<nSpace;I++)
            for(m=rowptr[I];m<rowptr[I+1];m++)
              {
                flux[ebN*nQuadraturePoints_elementBoundary+
                     k]
                  +=
                  TR_ALPHA*
                  a[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nnz+
                    left_ebN_element*nQuadraturePoints_elementBoundary*nnz+
                    k*nnz+
                    m]
                  *
                  V[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                    left_ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                    k*nSpace+
                    colind[m]]
                  *
                  n[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                    left_ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                    k*nSpace+
                    I];
                flux[ebN*nQuadraturePoints_elementBoundary+
                     k]
                  +=
                  (1.0-TR_ALPHA)*
                  a[right_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nnz+
                    right_ebN_element*nQuadraturePoints_elementBoundary*nnz+
                    k*nnz+
                    m]
                  *
                  V[right_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                    right_ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                    k*nSpace+
                    colind[m]]
                  *
                  n[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                    left_ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                    k*nSpace+
                    I];
              }
          flux[ebN*nQuadraturePoints_elementBoundary+
               k]
            +=
            penalty[ebN*nQuadraturePoints_elementBoundary+k]
            *
            (phi[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary+
                 left_ebN_element*nQuadraturePoints_elementBoundary+
                 k]
             -
             phi[right_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary+
                 right_ebN_element*nQuadraturePoints_elementBoundary+
                 k]);
        }
    }
}

/**
   \brief Update the advective flux at at interior element boundaries
*/
void updateInteriorNumericalDiffusiveFluxJacobian_LDG_upwind(proteus_index_t nInteriorElementBoundaries_global,
                                                             int nElementBoundaries_element,
                                                             int nQuadraturePoints_elementBoundary,
                                                             int nDOF_trial_element,
                                                             int nSpace,
                                                             proteus_index_t* interiorElementBoundaries,
                                                             proteus_index_t* elementBoundaryElements,
                                                             int* elementBoundaryLocalElementBoundaries,
                                                             double* n,
                                                             double* a,
                                                             double* da,
                                                             double* dphi,
                                                             double* V,
                                                             double* DV,
                                                             double* DV_eb,
                                                             double* v,
                                                             double* penalty,
                                                             double* fluxJacobian,
                                                             double* fluxJacobian_eb)
{
  proteus_index_t ebN, left_eN_global, right_eN_global; int ebNI, eN_ebN_element, left_ebN_element, right_ebN_element, k, j, I, J, nSpace2=nSpace*nSpace;
  for(ebNI=0;ebNI<nInteriorElementBoundaries_global;ebNI++)
    {
      ebN = interiorElementBoundaries[ebNI];
      left_eN_global = elementBoundaryElements[ebN*2+0];
      right_eN_global = elementBoundaryElements[ebN*2+1];
      left_ebN_element = elementBoundaryLocalElementBoundaries[ebN*2+0];
      right_ebN_element = elementBoundaryLocalElementBoundaries[ebN*2+1];
      for(k=0;k<nQuadraturePoints_elementBoundary;k++)
        {
          for(j=0;j<nDOF_trial_element;j++)
            {
              for(I=0;I<nSpace;I++)
                for(J=0;J<nSpace;J++)
                  {
                    fluxJacobian[ebN*2*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                                 0*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                                 k*nDOF_trial_element+
                                 j]
                      +=
                      TR_ALPHA*(a[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace2+
                         left_ebN_element*nQuadraturePoints_elementBoundary*nSpace2+
                         k*nSpace2+
                         I*nSpace+
                         J]
                       *
                       DV[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_trial_element*nSpace+
                          left_ebN_element*nQuadraturePoints_elementBoundary*nDOF_trial_element*nSpace+
                          k*nDOF_trial_element*nSpace+
                          j*nSpace+
                          J]
                       +
                       da[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace2+
                          left_ebN_element*nQuadraturePoints_elementBoundary*nSpace2+
                          k*nSpace2+
                          I*nSpace+
                          J]
                       *
                       v[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                         left_ebN_element*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                         k*nDOF_trial_element+
                         j]
                       *
                       V[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                         left_ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                         k*nSpace+
                         J]
                       )
                      *
                      n[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                        left_ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                        k*nSpace+
                        I];
                    for (eN_ebN_element=0;eN_ebN_element<nElementBoundaries_element;eN_ebN_element++)
                      fluxJacobian_eb[ebN*2*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                                      0*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                                      eN_ebN_element*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                                      k*nDOF_trial_element+
                                      j]
                        +=
                        TR_ALPHA*a[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace2+
                                   left_ebN_element*nQuadraturePoints_elementBoundary*nSpace2+
                                   k*nSpace2+
                                   I*nSpace+
                                   J]
                        *
                        DV_eb[left_eN_global*nElementBoundaries_element*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_trial_element*nSpace+
                              left_ebN_element*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_trial_element*nSpace+
                              eN_ebN_element*nQuadraturePoints_elementBoundary*nDOF_trial_element*nSpace+
                              k*nDOF_trial_element*nSpace+
                              j*nSpace+
                              J]
                        *
                        n[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                          left_ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                          k*nSpace+
                          I];
                    fluxJacobian[ebN*2*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                                 1*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                                 k*nDOF_trial_element+
                                 j]
                      +=
                      (1.0-TR_ALPHA)*a[right_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace2+
                                       right_ebN_element*nQuadraturePoints_elementBoundary*nSpace2+
                                       k*nSpace2+
                                       I*nSpace+
                                       J]
                      *
                      DV[right_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_trial_element*nSpace+
                         right_ebN_element*nQuadraturePoints_elementBoundary*nDOF_trial_element*nSpace+
                         k*nDOF_trial_element*nSpace+
                         j*nSpace+
                         J]
                      *
                      n[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                        left_ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                        k*nSpace+
                        I];
                    for (eN_ebN_element=0;eN_ebN_element<nElementBoundaries_element;eN_ebN_element++)
                      fluxJacobian_eb[ebN*2*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                                      1*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                                      eN_ebN_element*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                                      k*nDOF_trial_element+
                                      j]
                        +=
                        (1.0-TR_ALPHA)*a[right_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace2+
                                         right_ebN_element*nQuadraturePoints_elementBoundary*nSpace2+
                                         k*nSpace2+
                                         I*nSpace+
                                         J]
                        *
                        DV_eb[right_eN_global*nElementBoundaries_element*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_trial_element*nSpace+
                              right_ebN_element*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_trial_element*nSpace+
                              eN_ebN_element*nQuadraturePoints_elementBoundary*nDOF_trial_element*nSpace+
                              k*nDOF_trial_element*nSpace+
                              j*nSpace+
                              J]
                        *
                        n[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                          left_ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                          k*nSpace+
                          I];
                  }
              fluxJacobian[ebN*2*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                           0*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                           k*nDOF_trial_element+
                           j]
                +=
                penalty[ebN*nQuadraturePoints_elementBoundary +k]
                *
                dphi[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary+
                     left_ebN_element*nQuadraturePoints_elementBoundary+
                     k]
                *
                v[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                  left_ebN_element*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                  k*nDOF_trial_element+
                  j];
              fluxJacobian[ebN*2*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                           1*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                           k*nDOF_trial_element+
                           j]
                -=
                penalty[ebN*nQuadraturePoints_elementBoundary + k]
                *
                dphi[right_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary+
                     right_ebN_element*nQuadraturePoints_elementBoundary+
                     k]
                *
                v[right_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                  right_ebN_element*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                  k*nDOF_trial_element+
                  j];
            }
        }
    }
}

void updateInteriorNumericalDiffusiveFluxJacobian_LDG_upwind_sd(proteus_index_t nInteriorElementBoundaries_global,
								int nElementBoundaries_element,
								int nQuadraturePoints_elementBoundary,
								int nDOF_trial_element,
								int nSpace,
								proteus_csr_index_t* rowptr,
								proteus_csr_index_t* colind,
								proteus_index_t* interiorElementBoundaries,
								proteus_index_t* elementBoundaryElements,
								int* elementBoundaryLocalElementBoundaries,
								double* n,
								double* a,
								double* da,
								double* dphi,
								double* V,
								double* DV,
								double* DV_eb,
								double* v,
								double* penalty,
								double* fluxJacobian,
								double* fluxJacobian_eb)
{
  proteus_index_t ebN, left_eN_global, right_eN_global; proteus_csr_index_t m; proteus_csr_index_t nnz=rowptr[nSpace]; int ebNI, eN_ebN_element, left_ebN_element, right_ebN_element, k, j, I;
  for(ebNI=0;ebNI<nInteriorElementBoundaries_global;ebNI++)
    {
      ebN = interiorElementBoundaries[ebNI];
      left_eN_global = elementBoundaryElements[ebN*2+0];
      right_eN_global = elementBoundaryElements[ebN*2+1];
      left_ebN_element = elementBoundaryLocalElementBoundaries[ebN*2+0];
      right_ebN_element = elementBoundaryLocalElementBoundaries[ebN*2+1];
      for(k=0;k<nQuadraturePoints_elementBoundary;k++)
        {
          for(j=0;j<nDOF_trial_element;j++)
            {
              for(I=0;I<nSpace;I++)
                for(m=rowptr[I];m<rowptr[I+1];m++)
                  {
                    fluxJacobian[ebN*2*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                                 0*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                                 k*nDOF_trial_element+
                                 j]
                      +=
                      TR_ALPHA*(a[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nnz+
				  left_ebN_element*nQuadraturePoints_elementBoundary*nnz+
				  k*nnz+
				  m]
                       *
                       DV[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_trial_element*nSpace+
                          left_ebN_element*nQuadraturePoints_elementBoundary*nDOF_trial_element*nSpace+
                          k*nDOF_trial_element*nSpace+
                          j*nSpace+
                          colind[m]]
                       +
                       da[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nnz+
                          left_ebN_element*nQuadraturePoints_elementBoundary*nnz+
                          k*nnz+
                          m]
                       *
                       v[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                         left_ebN_element*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                         k*nDOF_trial_element+
                         j]
                       *
                       V[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                         left_ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                         k*nSpace+
                         colind[m]]
                       )
                      *
                      n[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                        left_ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                        k*nSpace+
                        I];
                    for (eN_ebN_element=0;eN_ebN_element<nElementBoundaries_element;eN_ebN_element++)
                      fluxJacobian_eb[ebN*2*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                                      0*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                                      eN_ebN_element*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                                      k*nDOF_trial_element+
                                      j]
                        +=
                        TR_ALPHA*a[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nnz+
                                   left_ebN_element*nQuadraturePoints_elementBoundary*nnz+
                                   k*nnz+
                                   m]
                        *
                        DV_eb[left_eN_global*nElementBoundaries_element*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_trial_element*nSpace+
                              left_ebN_element*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_trial_element*nSpace+
                              eN_ebN_element*nQuadraturePoints_elementBoundary*nDOF_trial_element*nSpace+
                              k*nDOF_trial_element*nSpace+
                              j*nSpace+
                              colind[m]]
                        *
                        n[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                          left_ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                          k*nSpace+
                          I];
                    fluxJacobian[ebN*2*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                                 1*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                                 k*nDOF_trial_element+
                                 j]
                      +=
                      (1.0-TR_ALPHA)*a[right_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nnz+
                                       right_ebN_element*nQuadraturePoints_elementBoundary*nnz+
                                       k*nnz+
                                       m]
                      *
                      DV[right_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_trial_element*nSpace+
                         right_ebN_element*nQuadraturePoints_elementBoundary*nDOF_trial_element*nSpace+
                         k*nDOF_trial_element*nSpace+
                         j*nSpace+
                         colind[m]]
                      *
                      n[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                        left_ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                        k*nSpace+
                        I];
                    for (eN_ebN_element=0;eN_ebN_element<nElementBoundaries_element;eN_ebN_element++)
                      fluxJacobian_eb[ebN*2*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                                      1*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                                      eN_ebN_element*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                                      k*nDOF_trial_element+
                                      j]
                        +=
                        (1.0-TR_ALPHA)*a[right_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nnz+
                                         right_ebN_element*nQuadraturePoints_elementBoundary*nnz+
                                         k*nnz+
                                         m]
                        *
                        DV_eb[right_eN_global*nElementBoundaries_element*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_trial_element*nSpace+
                              right_ebN_element*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_trial_element*nSpace+
                              eN_ebN_element*nQuadraturePoints_elementBoundary*nDOF_trial_element*nSpace+
                              k*nDOF_trial_element*nSpace+
                              j*nSpace+
                              colind[m]]
                        *
                        n[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                          left_ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                          k*nSpace+
                          I];
                  }
              fluxJacobian[ebN*2*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                           0*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                           k*nDOF_trial_element+
                           j]
                +=
                penalty[ebN*nQuadraturePoints_elementBoundary +k]
                *
                dphi[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary+
                     left_ebN_element*nQuadraturePoints_elementBoundary+
                     k]
                *
                v[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                  left_ebN_element*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                  k*nDOF_trial_element+
                  j];
              fluxJacobian[ebN*2*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                           1*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                           k*nDOF_trial_element+
                           j]
                -=
                penalty[ebN*nQuadraturePoints_elementBoundary + k]
                *
                dphi[right_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary+
                     right_ebN_element*nQuadraturePoints_elementBoundary+
                     k]
                *
                v[right_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                  right_ebN_element*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                  k*nDOF_trial_element+
                  j];
            }
        }
    }
}

/**
   \brief Calculate the advective flux at at interior element boundaries
*/
void calculateExteriorNumericalDiffusiveFlux_LDG_upwind(proteus_index_t nExteriorElementBoundaries_global,
                                                        int nElementBoundaries_element,
                                                        int nQuadraturePoints_elementBoundary,
                                                        int nSpace,
                                                        proteus_index_t* exteriorElementBoundaries,
                                                        proteus_index_t* elementBoundaryElements,
                                                        int* elementBoundaryLocalElementBoundaries,
                                                        double* n,
                                                        double* u,
                                                        double* a,
                                                        double* phi_bc,
                                                        double* phi,
                                                        double* V,
                                                        double* penalty,
                                                        double* flux)
{
  proteus_index_t ebN, eN_global; int ebNE, ebN_element, k, I, J, nSpace2=nSpace*nSpace;
  for(ebNE=0;ebNE<nExteriorElementBoundaries_global;ebNE++)
    {
      ebN = exteriorElementBoundaries[ebNE];
      eN_global = elementBoundaryElements[ebN*2+0];
      ebN_element = elementBoundaryLocalElementBoundaries[ebN*2+0];
      for(k=0;k<nQuadraturePoints_elementBoundary;k++)
        {
          flux[ebN*nQuadraturePoints_elementBoundary+
               k] =0.0;
          for(I=0;I<nSpace;I++)
            for(J=0;J<nSpace;J++)
              {
                flux[ebN*nQuadraturePoints_elementBoundary+
                     k]
                  +=
                  TR_ALPHA_EXT*
                  a[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace2+
                    ebN_element*nQuadraturePoints_elementBoundary*nSpace2+
                    k*nSpace2+
                    I*nSpace+
                    J]
                  *
                  V[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                    ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                    k*nSpace+
                    J]
                  *
                  n[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                    ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                    k*nSpace+
                    I];
              }
          flux[ebN*nQuadraturePoints_elementBoundary+
               k] +=
            penalty[ebN*nQuadraturePoints_elementBoundary+
                    k]*
            (phi[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary+
                 ebN_element*nQuadraturePoints_elementBoundary+
                 k]
             -
             phi_bc[ebNE*nQuadraturePoints_elementBoundary+
		    k]);
          
        }
    }
}
void calculateExteriorNumericalDiffusiveFlux_LDG_upwind_sd(proteus_index_t nExteriorElementBoundaries_global,
							   int nElementBoundaries_element,
							   int nQuadraturePoints_elementBoundary,
							   int nSpace,
							   proteus_csr_index_t* rowptr,
							   proteus_csr_index_t* colind,
							   proteus_index_t* exteriorElementBoundaries,
							   proteus_index_t* elementBoundaryElements,
							   int* elementBoundaryLocalElementBoundaries,
							   double* n,
							   double* u,
							   double* a,
							   double* phi_bc,
							   double* phi,
							   double* V,
							   double* penalty,
							   double* flux)
{
  proteus_index_t ebN, eN_global; proteus_csr_index_t m; proteus_csr_index_t nnz=rowptr[nSpace]; int ebNE, ebN_element, k, I;
  for(ebNE=0;ebNE<nExteriorElementBoundaries_global;ebNE++)
    {
      ebN = exteriorElementBoundaries[ebNE];
      eN_global = elementBoundaryElements[ebN*2+0];
      ebN_element = elementBoundaryLocalElementBoundaries[ebN*2+0];
      for(k=0;k<nQuadraturePoints_elementBoundary;k++)
        {
          flux[ebN*nQuadraturePoints_elementBoundary+
               k] =0.0;
          for(I=0;I<nSpace;I++)
            for(m=rowptr[I];m<rowptr[I+1];m++)
              {
                flux[ebN*nQuadraturePoints_elementBoundary+
                     k]
                  +=
                  TR_ALPHA_EXT*
                  a[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nnz+
                    ebN_element*nQuadraturePoints_elementBoundary*nnz+
                    k*nnz+
                    m]
                  *
                  V[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                    ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                    k*nSpace+
                    colind[m]]
                  *
                  n[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                    ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                    k*nSpace+
                    I];
              }
          flux[ebN*nQuadraturePoints_elementBoundary+
               k] +=
            penalty[ebN*nQuadraturePoints_elementBoundary+
                    k]*
            (phi[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary+
                 ebN_element*nQuadraturePoints_elementBoundary+
                 k]
             -
             phi_bc[ebNE*nQuadraturePoints_elementBoundary+
		    k]);
          
        }
    }
}
/**
   \brief Calculate the advective flux at at global exterior element boundaries
*/
void calculateGlobalExteriorNumericalDiffusiveFlux_LDG_upwind(proteus_index_t nExteriorElementBoundaries_global,
							      int nElementBoundaries_element,
							      int nQuadraturePoints_elementBoundary,
							      int nSpace,
							      proteus_index_t* exteriorElementBoundaries,
							      proteus_index_t* elementBoundaryElements,
							      int* elementBoundaryLocalElementBoundaries,
							      double* n,
							      double* u,
							      double* a,
							      double* phi_bc,
							      double* phi,
							      double* V,
							      double* penalty,
							      double* flux)
{
  proteus_index_t ebN, eN_global; int ebNE, ebN_element, k, I, J, nSpace2=nSpace*nSpace;
  for(ebNE=0;ebNE<nExteriorElementBoundaries_global;ebNE++)
    {
      ebN = exteriorElementBoundaries[ebNE];
      eN_global = elementBoundaryElements[ebN*2+0];
      ebN_element = elementBoundaryLocalElementBoundaries[ebN*2+0];
      for(k=0;k<nQuadraturePoints_elementBoundary;k++)
        {
          flux[ebNE*nQuadraturePoints_elementBoundary+
               k] =0.0;
          for(I=0;I<nSpace;I++)
            for(J=0;J<nSpace;J++)
              {
                flux[ebNE*nQuadraturePoints_elementBoundary+
                     k]
                  +=
                  TR_ALPHA_EXT*
                  a[ebNE*nQuadraturePoints_elementBoundary*nSpace2+
                    k*nSpace2+
                    I*nSpace+
                    J]
                  *
                  V[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                    ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                    k*nSpace+
                    J]
                  *
                  n[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                    k*nSpace+
                    I];
              }
          flux[ebNE*nQuadraturePoints_elementBoundary+
               k] +=
            penalty[ebNE*nQuadraturePoints_elementBoundary+
                    k]*
            (phi[ebNE*nQuadraturePoints_elementBoundary+
                 k]
             -
             phi_bc[ebNE*nQuadraturePoints_elementBoundary+
		    k]);
          
        }
    }
}
void calculateGlobalExteriorNumericalDiffusiveFlux_LDG_upwind_sd(proteus_index_t nExteriorElementBoundaries_global,
								 int nElementBoundaries_element,
								 int nQuadraturePoints_elementBoundary,
								 int nSpace,
								 proteus_csr_index_t* rowptr,
								 proteus_csr_index_t* colind,
								 proteus_index_t* exteriorElementBoundaries,
								 proteus_index_t* elementBoundaryElements,
								 int* elementBoundaryLocalElementBoundaries,
								 double* n,
								 double* u,
								 double* a,
								 double* phi_bc,
								 double* phi,
								 double* V,
								 double* penalty,
								 double* flux)
{
  proteus_index_t ebN, eN_global; proteus_csr_index_t m; proteus_csr_index_t nnz=rowptr[nSpace]; int ebNE, ebN_element, k, I;
  for(ebNE=0;ebNE<nExteriorElementBoundaries_global;ebNE++)
    {
      ebN = exteriorElementBoundaries[ebNE];
      eN_global = elementBoundaryElements[ebN*2+0];
      ebN_element = elementBoundaryLocalElementBoundaries[ebN*2+0];
      for(k=0;k<nQuadraturePoints_elementBoundary;k++)
        {
          flux[ebNE*nQuadraturePoints_elementBoundary+
               k] =0.0;
          for(I=0;I<nSpace;I++)
            for(m=rowptr[I];m<rowptr[I+1];m++)
              {
                flux[ebNE*nQuadraturePoints_elementBoundary+
                     k]
                  +=
                  TR_ALPHA_EXT*
                  a[ebNE*nQuadraturePoints_elementBoundary*nnz+
                    k*nnz+
                    m]
                  *
                  V[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                    ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                    k*nSpace+
                    colind[m]]
                  *
                  n[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                    k*nSpace+
                    I];
              }
          flux[ebNE*nQuadraturePoints_elementBoundary+
               k] +=
            penalty[ebNE*nQuadraturePoints_elementBoundary+
                    k]*
            (phi[ebNE*nQuadraturePoints_elementBoundary+
                 k]
             -
             phi_bc[ebNE*nQuadraturePoints_elementBoundary+
		    k]);
          
        }
    }
}
/**
   \brief update the flux Jacobian with the advective flux contribution at at exterior element boundaries
*/
void updateExteriorNumericalDiffusiveFluxJacobian_LDG_upwind(int* isDiffusiveFluxBoundary,
                                                             proteus_index_t nExteriorElementBoundaries_global,
                                                             int nElementBoundaries_element,
                                                             int nQuadraturePoints_elementBoundary,
                                                             int nDOF_trial_element,
                                                             int nSpace,
                                                             proteus_index_t* exteriorElementBoundaries,
                                                             proteus_index_t* elementBoundaryElements,
                                                             int* elementBoundaryLocalElementBoundaries,
                                                             double* n,
                                                             double* a,
                                                             double* da,
                                                             double* dphi,
                                                             double* V,
                                                             double* DV,
                                                             double* DV_eb,
                                                             double* v,
                                                             double* penalty,
                                                             double* fluxJacobian,
                                                             double* fluxJacobian_eb)
{
  proteus_index_t ebN, eN_global; int ebNE, ebN_element, eN_ebN_element, k, j, I, J, nSpace2=nSpace*nSpace;
  for(ebNE=0;ebNE<nExteriorElementBoundaries_global;ebNE++)
    {
      ebN = exteriorElementBoundaries[ebNE];
      eN_global = elementBoundaryElements[ebN*2+0];
      ebN_element = elementBoundaryLocalElementBoundaries[ebN*2+0];
      for(k=0;k<nQuadraturePoints_elementBoundary;k++)
        {
          if(isDiffusiveFluxBoundary[ebNE*nQuadraturePoints_elementBoundary+k] != 1)
            {
              for(j=0;j<nDOF_trial_element;j++)
                {
                  for(I=0;I<nSpace;I++)
                    for(J=0;J<nSpace;J++)
                      {
                        fluxJacobian[ebN*2*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                                     0*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                                     k*nDOF_trial_element+
                                     j]
                          +=
                          TR_ALPHA_EXT*
                          (a[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace2+
                             ebN_element*nQuadraturePoints_elementBoundary*nSpace2+
                             k*nSpace2+
                             I*nSpace+
                             J]
                           *
                           DV[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_trial_element*nSpace+
                              ebN_element*nQuadraturePoints_elementBoundary*nDOF_trial_element*nSpace+
                              k*nDOF_trial_element*nSpace+
                              j*nSpace+
                              J]
                           + 
                           da[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace2+
                              ebN_element*nQuadraturePoints_elementBoundary*nSpace2+
                              k*nSpace2+
                              I*nSpace+
                              J]
                           *
                           v[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                             ebN_element*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                             k*nDOF_trial_element+
                             j]
                           *
                           V[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                             ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                             k*nSpace+
                             J]
                           )
                          *
                          n[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                            ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                            k*nSpace+
                            I];
                        for (eN_ebN_element=0;eN_ebN_element<nElementBoundaries_element;eN_ebN_element++)
                          {
                            fluxJacobian_eb[ebN*2*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                                            0*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                                            eN_ebN_element*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                                            k*nDOF_trial_element+
                                            j]
                              +=
                              TR_ALPHA_EXT*a[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace2+
                                             ebN_element*nQuadraturePoints_elementBoundary*nSpace2+
                                             k*nSpace2+
                                             I*nSpace+
                                             J]
                              *
                              DV_eb[eN_global*nElementBoundaries_element*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_trial_element*nSpace+
                                    ebN_element*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_trial_element*nSpace+
                                    eN_ebN_element*nQuadraturePoints_elementBoundary*nDOF_trial_element*nSpace+
                                    k*nDOF_trial_element*nSpace+
                                    j*nSpace+
                                    J]
                              *
                              n[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                                ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                                k*nSpace+
                                I];
                          }
                      }
                  fluxJacobian[ebN*2*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                               0*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                               k*nDOF_trial_element+
                               j]
                    +=   
                    penalty[ebN*nQuadraturePoints_elementBoundary +k]
                    *
                    dphi[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary+
                         ebN_element*nQuadraturePoints_elementBoundary+
                         k]
                    *
                    v[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                      ebN_element*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                      k*nDOF_trial_element+
                      j];
                }
            }
        }
    }
}
void updateExteriorNumericalDiffusiveFluxJacobian_LDG_upwind_sd(int* isDiffusiveFluxBoundary,
								proteus_index_t nExteriorElementBoundaries_global,
								int nElementBoundaries_element,
								int nQuadraturePoints_elementBoundary,
								int nDOF_trial_element,
								int nSpace,
								proteus_csr_index_t* rowptr,
								proteus_csr_index_t* colind,
								proteus_index_t* exteriorElementBoundaries,
								proteus_index_t* elementBoundaryElements,
								int* elementBoundaryLocalElementBoundaries,
								double* n,
								double* a,
								double* da,
								double* dphi,
								double* V,
								double* DV,
								double* DV_eb,
								double* v,
								double* penalty,
								double* fluxJacobian,
								double* fluxJacobian_eb)
{
  proteus_index_t ebN, eN_global; proteus_csr_index_t m; proteus_csr_index_t nnz=rowptr[nSpace]; int ebNE, ebN_element, eN_ebN_element, k, j, I;
  for(ebNE=0;ebNE<nExteriorElementBoundaries_global;ebNE++)
    {
      ebN = exteriorElementBoundaries[ebNE];
      eN_global = elementBoundaryElements[ebN*2+0];
      ebN_element = elementBoundaryLocalElementBoundaries[ebN*2+0];
      for(k=0;k<nQuadraturePoints_elementBoundary;k++)
        {
          if(isDiffusiveFluxBoundary[ebNE*nQuadraturePoints_elementBoundary+k] != 1)
            {
              for(j=0;j<nDOF_trial_element;j++)
                {
                  for(I=0;I<nSpace;I++)
                    for(m=rowptr[I];m<rowptr[I+1];m++)
                      {
                        fluxJacobian[ebN*2*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                                     0*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                                     k*nDOF_trial_element+
                                     j]
                          +=
                          TR_ALPHA_EXT*
                          (a[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nnz+
                             ebN_element*nQuadraturePoints_elementBoundary*nnz+
                             k*nnz+
                             m]
                           *
                           DV[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_trial_element*nSpace+
                              ebN_element*nQuadraturePoints_elementBoundary*nDOF_trial_element*nSpace+
                              k*nDOF_trial_element*nSpace+
                              j*nSpace+
                              colind[m]]
                           + 
                           da[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nnz+
                              ebN_element*nQuadraturePoints_elementBoundary*nnz+
                              k*nnz+
                              m]
                           *
                           v[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                             ebN_element*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                             k*nDOF_trial_element+
                             j]
                           *
                           V[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                             ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                             k*nSpace+
                             colind[m]]
                           )
                          *
                          n[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                            ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                            k*nSpace+
                            I];
                        for (eN_ebN_element=0;eN_ebN_element<nElementBoundaries_element;eN_ebN_element++)
                          {
                            fluxJacobian_eb[ebN*2*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                                            0*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                                            eN_ebN_element*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                                            k*nDOF_trial_element+
                                            j]
                              +=
                              TR_ALPHA_EXT*a[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nnz+
                                             ebN_element*nQuadraturePoints_elementBoundary*nnz+
                                             k*nnz+
                                             m]
                              *
                              DV_eb[eN_global*nElementBoundaries_element*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_trial_element*nSpace+
                                    ebN_element*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_trial_element*nSpace+
                                    eN_ebN_element*nQuadraturePoints_elementBoundary*nDOF_trial_element*nSpace+
                                    k*nDOF_trial_element*nSpace+
                                    j*nSpace+
                                    colind[m]]
                              *
                              n[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                                ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                                k*nSpace+
                                I];
                          }
                      }
                  fluxJacobian[ebN*2*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                               0*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                               k*nDOF_trial_element+
                               j]
                    +=   
                    penalty[ebN*nQuadraturePoints_elementBoundary +k]
                    *
                    dphi[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary+
                         ebN_element*nQuadraturePoints_elementBoundary+
                         k]
                    *
                    v[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                      ebN_element*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                      k*nDOF_trial_element+
                      j];
                }
            }
        }
    }
}
/**
   \brief update the flux Jacobian with the advective flux contribution at at exterior element boundaries
*/
void updateGlobalExteriorNumericalDiffusiveFluxJacobian_LDG_upwind(int* isDiffusiveFluxBoundary,
								   proteus_index_t nExteriorElementBoundaries_global,
								   int nElementBoundaries_element,
								   int nQuadraturePoints_elementBoundary,
								   int nDOF_trial_element,
								   int nSpace,
								   proteus_index_t* exteriorElementBoundaries,
								   proteus_index_t* elementBoundaryElements,
								   int* elementBoundaryLocalElementBoundaries,
								   double* n,
								   double* a,
								   double* da,
								   double* dphi,
								   double* V,
								   double* DV,
								   double* DV_eb,
								   double* v,
								   double* penalty,
								   double* fluxJacobian_exterior,
								   double* fluxJacobian_eb)
{
  proteus_index_t ebN, eN_global; int ebNE, ebN_element, eN_ebN_element, k, j, I, J, nSpace2=nSpace*nSpace;
  for(ebNE=0;ebNE<nExteriorElementBoundaries_global;ebNE++)
    {
      ebN = exteriorElementBoundaries[ebNE];
      eN_global = elementBoundaryElements[ebN*2+0];
      ebN_element = elementBoundaryLocalElementBoundaries[ebN*2+0];
      for(k=0;k<nQuadraturePoints_elementBoundary;k++)
        {
          if(isDiffusiveFluxBoundary[ebNE*nQuadraturePoints_elementBoundary+k] != 1)
            {
              for(j=0;j<nDOF_trial_element;j++)
                {
                  for(I=0;I<nSpace;I++)
                    for(J=0;J<nSpace;J++)
                      {
                        fluxJacobian_exterior[ebNE*nQuadraturePoints_elementBoundary*nDOF_trial_element+
					      k*nDOF_trial_element+
					      j]
                          +=
                          TR_ALPHA_EXT*
                          (a[ebNE*nQuadraturePoints_elementBoundary*nSpace2+
                             k*nSpace2+
                             I*nSpace+
                             J]
                           *
                           DV[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_trial_element*nSpace+
                              ebN_element*nQuadraturePoints_elementBoundary*nDOF_trial_element*nSpace+
                              k*nDOF_trial_element*nSpace+
                              j*nSpace+
                              J]
                           + 
                           da[ebNE*nQuadraturePoints_elementBoundary*nSpace2+
                              k*nSpace2+
                              I*nSpace+
                              J]
                           *
                           v[ebNE*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                             k*nDOF_trial_element+
                             j]
                           *
                           V[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                             ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                             k*nSpace+
                             J]
                           )
                          *
                          n[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                            k*nSpace+
                            I];
                        for (eN_ebN_element=0;eN_ebN_element<nElementBoundaries_element;eN_ebN_element++)
                          {
                            fluxJacobian_eb[ebN*2*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                                            0*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                                            eN_ebN_element*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                                            k*nDOF_trial_element+
                                            j]
                              +=
                              TR_ALPHA_EXT*a[ebNE*nQuadraturePoints_elementBoundary*nSpace2+
                                             k*nSpace2+
                                             I*nSpace+
                                             J]
                              *
                              DV_eb[eN_global*nElementBoundaries_element*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_trial_element*nSpace+
                                    ebN_element*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_trial_element*nSpace+
                                    eN_ebN_element*nQuadraturePoints_elementBoundary*nDOF_trial_element*nSpace+
                                    k*nDOF_trial_element*nSpace+
                                    j*nSpace+
                                    J]
                              *
                              n[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                                k*nSpace+
                                I];
                          }
                      }
                  fluxJacobian_exterior[ebNE*nQuadraturePoints_elementBoundary*nDOF_trial_element+
					k*nDOF_trial_element+
					j]
                    +=   
                    penalty[ebNE*nQuadraturePoints_elementBoundary +k]
                    *
                    dphi[ebNE*nQuadraturePoints_elementBoundary+
                         k]
                    *
                    v[ebNE*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                      k*nDOF_trial_element+
                      j];
                }
            }
        }
    }
}

void updateGlobalExteriorNumericalDiffusiveFluxJacobian_LDG_upwind_sd(int* isDiffusiveFluxBoundary,
								      proteus_index_t nExteriorElementBoundaries_global,
								      int nElementBoundaries_element,
								      int nQuadraturePoints_elementBoundary,
								      int nDOF_trial_element,
								      int nSpace,
								      proteus_csr_index_t* rowptr,
								      proteus_csr_index_t* colind,
								      proteus_index_t* exteriorElementBoundaries,
								      proteus_index_t* elementBoundaryElements,
								      int* elementBoundaryLocalElementBoundaries,
								      double* n,
								      double* a,
								      double* da,
								      double* dphi,
								      double* V,
								      double* DV,
								      double* DV_eb,
								      double* v,
								      double* penalty,
								      double* fluxJacobian_exterior,
								      double* fluxJacobian_eb)
{
  proteus_index_t ebN, eN_global; proteus_csr_index_t m; proteus_csr_index_t nnz=rowptr[nSpace]; int ebNE, ebN_element, eN_ebN_element, k, j, I;
  for(ebNE=0;ebNE<nExteriorElementBoundaries_global;ebNE++)
    {
      ebN = exteriorElementBoundaries[ebNE];
      eN_global = elementBoundaryElements[ebN*2+0];
      ebN_element = elementBoundaryLocalElementBoundaries[ebN*2+0];
      for(k=0;k<nQuadraturePoints_elementBoundary;k++)
        {
          if(isDiffusiveFluxBoundary[ebNE*nQuadraturePoints_elementBoundary+k] != 1)
            {
              for(j=0;j<nDOF_trial_element;j++)
                {
                  for(I=0;I<nSpace;I++)
                    for(m=rowptr[I];m<rowptr[I+1];m++)
                      {
                        fluxJacobian_exterior[ebNE*nQuadraturePoints_elementBoundary*nDOF_trial_element+
					      k*nDOF_trial_element+
					      j]
                          +=
                          TR_ALPHA_EXT*
                          (a[ebNE*nQuadraturePoints_elementBoundary*nnz+
                             k*nnz+
                             m]
                           *
                           DV[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_trial_element*nSpace+
                              ebN_element*nQuadraturePoints_elementBoundary*nDOF_trial_element*nSpace+
                              k*nDOF_trial_element*nSpace+
                              j*nSpace+
                              colind[m]]
                           + 
                           da[ebNE*nQuadraturePoints_elementBoundary*nnz+
                              k*nnz+
                              m]
                           *
                           v[ebNE*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                             k*nDOF_trial_element+
                             j]
                           *
                           V[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                             ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                             k*nSpace+
                             colind[m]]
                           )
                          *
                          n[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                            k*nSpace+
                            I];
                        for (eN_ebN_element=0;eN_ebN_element<nElementBoundaries_element;eN_ebN_element++)
                          {
                            fluxJacobian_eb[ebN*2*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                                            0*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                                            eN_ebN_element*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                                            k*nDOF_trial_element+
                                            j]
                              +=
                              TR_ALPHA_EXT*a[ebNE*nQuadraturePoints_elementBoundary*nnz+
                                             k*nnz+
                                             m]
                              *
                              DV_eb[eN_global*nElementBoundaries_element*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_trial_element*nSpace+
                                    ebN_element*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_trial_element*nSpace+
                                    eN_ebN_element*nQuadraturePoints_elementBoundary*nDOF_trial_element*nSpace+
                                    k*nDOF_trial_element*nSpace+
                                    j*nSpace+
                                    colind[m]]
                              *
                              n[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                                k*nSpace+
                                I];
                          }
                      }
                  fluxJacobian_exterior[ebNE*nQuadraturePoints_elementBoundary*nDOF_trial_element+
					k*nDOF_trial_element+
					j]
                    +=   
                    penalty[ebNE*nQuadraturePoints_elementBoundary +k]
                    *
                    dphi[ebNE*nQuadraturePoints_elementBoundary+
                         k]
                    *
                    v[ebNE*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                      k*nDOF_trial_element+
                      j];
                }
            }
        }
    }
}
/*
  pick matrices in "extended mixed formulation"
  
  \hat{a}^{-1} \vec v = \grad p (check sign) 
  \pm \deld (\tilde{a} \vec v ) in mass conservation 
 */
void calculateDiffusionMatrixSplittings_LDG_sd(int aSplit,
					       proteus_index_t nElements_global,
					       int nElementBoundaries_element,
					       int nQuadraturePoints_element,
					       int nQuadraturePoints_elementBoundary,
					       int nSpace,
					       const proteus_csr_index_t * rowptr,
					       const proteus_csr_index_t * colind,
					       const double * ebq_a,
					       const double * q_a,
					       double *eb_aHat,
					       double *eb_aTilde,
					       double *aHat,
					       double *aTilde)
{
  proteus_index_t ebN; proteus_csr_index_t m; proteus_csr_index_t nnz=rowptr[nSpace]; int eN, k, I;
  double factor=0.0;
  if (aSplit == 0)
    { /*inverted matrix is identity*/
      for (eN = 0; eN < nElements_global; eN++)
	for (ebN = 0; ebN < nElementBoundaries_element; ebN++)
	  for (k = 0; k < nQuadraturePoints_elementBoundary; k++)
	    {
	      for (I = 0; I < nSpace; I++)
		for (m=rowptr[I]; m < rowptr[I+1]; m++)
		  {
		    /*for starters only treats as diagonal*/
		    factor = colind[m] == I ? 1.0 : 0.0;
		    /*mwf debug
		      printf("calc LDG split eN=%d ebN=%d k=%d I=%d m=%d J=%d factor=%g \n",
		      eN,ebN,k,I,m,colind[m],factor);
		    */
		    eb_aHat[eN*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nnz +
			    ebN*nQuadraturePoints_elementBoundary*nnz + 
			    k*nnz +
			    m] = factor;
		  }
	      /* printf("aHat*aTilde \n"); */
	      /* for (I = 0; I < nSpace; I++) */
	      /* 	{ */
	      /* 	  assert(nnz == nSpace*nSpace); */
	      /* 	  int J,K; */
	      /* 	  for (J = 0; J < nSpace; J++) */
	      /* 	    { */
	      /* 	      double res = 0.0; */
	      /* 	      for (K=0;K<nSpace;K++) */
	      /* 		res += eb_aHat[eN*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nnz + */
	      /* 			       ebN*nQuadraturePoints_elementBoundary*nnz + */
	      /* 			       k*nnz + I*nSpace + K] */
	      /* 		  *eb_aTilde[eN*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nnz + */
	      /* 			     ebN*nQuadraturePoints_elementBoundary*nnz + */
	      /* 			     k*nnz + K*nSpace + J]; */
	      /* 	      printf("%i %i %12.5e \n",I,J,res); */
	      /* 	    } */
	      /* 	} */
	    }
      for (eN = 0; eN < nElements_global; eN++)
	for (k = 0; k < nQuadraturePoints_element; k++)
	  for (I = 0; I < nSpace; I++)
	    for (m=rowptr[I]; m < rowptr[I+1]; m++)
	      {
		/*for starters only treats as diagonal*/
		factor = colind[m] == I ? 1.0 : 0.0;
		aHat[eN*nQuadraturePoints_element*nnz +
		     k*nnz +
		     m] = factor;
	      }
    }
  else if (aSplit == 1)
    { /*inverted matrix is a*/
      for (eN = 0; eN < nElements_global; eN++)
	for (ebN = 0; ebN < nElementBoundaries_element; ebN++)
	  for (k = 0; k < nQuadraturePoints_elementBoundary; k++)
	    {
	      for (I = 0; I < nSpace; I++)
		for (m=rowptr[I]; m < rowptr[I+1]; m++)
		  {
		    /*for starters only treats as diagonal*/
		    factor = colind[m] == I ? 1.0 : 0.0;
		    eb_aTilde[eN*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nnz +
			      ebN*nQuadraturePoints_elementBoundary*nnz + 
			      k*nnz +
			      m] = factor;
		  }
	      /* printf("aHat*aTilde \n"); */
	      /* for (I = 0; I < nSpace; I++) */
	      /* 	{ */
	      /* 	  assert(nnz == nSpace*nSpace); */
	      /* 	  int J,K; */
	      /* 	  for (J = 0; J < nSpace; J++) */
	      /* 	    { */
	      /* 	      double res = 0.0; */
	      /* 	      for (K=0;K<nSpace;K++) */
	      /* 		res += eb_aHat[eN*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nnz + */
	      /* 			       ebN*nQuadraturePoints_elementBoundary*nnz + */
	      /* 			       k*nnz + I*nSpace + K] */
	      /* 		  *eb_aTilde[eN*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nnz + */
	      /* 			     ebN*nQuadraturePoints_elementBoundary*nnz + */
	      /* 			     k*nnz + K*nSpace + J]; */
	      /* 	      printf("%i %i %12.5e \n",I,J,res); */
	      /* 	    } */
	      /* 	} */
	    }
      for (eN = 0; eN < nElements_global; eN++)
	for (k = 0; k < nQuadraturePoints_element; k++)
	  {
	    for (I = 0; I < nSpace; I++)
	      for (m=rowptr[I]; m < rowptr[I+1]; m++)
		{
		  /*for starters only treats as diagonal*/
		  factor = colind[m] == I ? 1.0 : 0.0;
		  aTilde[eN*nQuadraturePoints_element*nnz +
			 k*nnz +
			 m] = factor;
		}
	    /* for (I = 0; I < nSpace; I++) */
	    /*   { */
	    /* 	assert(nnz == nSpace*nSpace); */
	    /* 	int J,K; */
	    /* 	for (J = 0; J < nSpace; J++) */
	    /* 	  { */
	    /* 	    double res = 0.0; */
	    /* 	    for (K=0;K<nSpace;K++) */
	    /* 	      res += aHat[eN*nQuadraturePoints_element*nnz + */
	    /* 			  k*nnz + I*nSpace + K] */
	    /* 		*aTilde[eN*nQuadraturePoints_element*nnz + */
	    /* 			k*nnz + K*nSpace + J]; */
	    /* 	    printf("%i %i %12.5e \n",I,J,res); */
	    /* 	  } */
	    /*   } */
	  }
    }
  else
    {
      assert(aSplit == 2);
      /*inverted matrix is sqrt(a)*/
      for (eN = 0; eN < nElements_global; eN++)
	for (ebN = 0; ebN < nElementBoundaries_element; ebN++)
	  for (k = 0; k < nQuadraturePoints_elementBoundary; k++)
	    for (I = 0; I < nSpace; I++)
	      for (m=rowptr[I]; m < rowptr[I+1]; m++)
		{
		  /*for starters only treats as diagonal*/
		  factor = colind[m] == I ? sqrt(ebq_a[eN*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nnz +
						       ebN*nQuadraturePoints_elementBoundary*nnz + 
						       k*nnz +
						       m]) : 0.0;
		  eb_aTilde[eN*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nnz +
			    ebN*nQuadraturePoints_elementBoundary*nnz + 
			    k*nnz +
			    m] = factor;
		  eb_aHat[eN*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nnz +
			  ebN*nQuadraturePoints_elementBoundary*nnz + 
			  k*nnz +
			  m] = factor;
		}
      for (eN = 0; eN < nElements_global; eN++)
	for (k = 0; k < nQuadraturePoints_element; k++)
	  for (I = 0; I < nSpace; I++)
	    for (m=rowptr[I]; m < rowptr[I+1]; m++)
	      {
		/*for starters only treats as diagonal*/
		factor = colind[m] == I ? sqrt(q_a[eN*nQuadraturePoints_element*nnz +
						   k*nnz +
						   m]) : 0.0;
		aTilde[eN*nQuadraturePoints_element*nnz +
		       k*nnz +
		       m] = factor;
		aHat[eN*nQuadraturePoints_element*nnz +
		     k*nnz +
		     m] = factor;
	      }
      
    }
}

					       
/***********************************************************************
  end LDG
 **********************************************************************/
void calculateInteriorLesaintRaviartNumericalFlux(proteus_index_t nInteriorElementBoundaries_global,
						  int nElementBoundaries_element,
						  int nQuadraturePoints_elementBoundary,
						  int nSpace,
						  int speedEvalFlag,
						  proteus_index_t* interiorElementBoundaries,
						  proteus_index_t* elementBoundaryElements,
						  int* elementBoundaryLocalElementBoundaries,
						  double* n,
						  double* u,
						  double* H,
						  double* dH,
						  double* flux,
						  double* dflux_left,
						  double* dflux_right)
{
  proteus_index_t ebN, left_eN_global, right_eN_global; int ebNI, left_ebN_element, right_ebN_element, k, J;
  double left_flux,right_flux,u_left,u_right,left_speed,right_speed,
    tmp_left,tmp_right;
  /*for now use outer normal at first quadrature point for element speed calculations*/
  
  for(ebNI=0;ebNI<nInteriorElementBoundaries_global;ebNI++)
    {
      ebN = interiorElementBoundaries[ebNI];
      left_eN_global = elementBoundaryElements[ebN*2+0];
      right_eN_global = elementBoundaryElements[ebN*2+1];
      left_ebN_element = elementBoundaryLocalElementBoundaries[ebN*2+0];
      right_ebN_element = elementBoundaryLocalElementBoundaries[ebN*2+1];
      /*
	dH_{L/R} = dH_{eN_left/eN_right} . n_{eN_left}

	dH_min = min(dH_L,dH_R), dH_max = max(dH_L,dH_R)

	if dH_min  < 0
           flux_eN_left = |dH_min| (u^{L}- u^{R})
        else
           flux_eN_left = 0
        if dH_max  > 0 
           flux_eN_right = |dH_max| (u^{R}-u^{L})
	else
           flux_eN_right = 0
      */
      for(k=0;k<nQuadraturePoints_elementBoundary;k++)
        {
	  left_speed =0.0;
	  right_speed=0.0;
          left_flux=0.0;
          right_flux=0.0;
	  tmp_left = 0.0; 
	  tmp_right = 0.0;
	  /*compute speed relative to left normal*/
          for(J=0;J<nSpace;J++)
            {
              tmp_left 
                += 
                n[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                  left_ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                  k*nSpace+
		  J]
                *
                dH[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                   left_ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                   k*nSpace+
		   J];
              tmp_right 
                += 
                n[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                  left_ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                  k*nSpace+
		  J]
                *
                dH[right_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                   right_ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                   k*nSpace+
		   J];
	    }
	  /*left_speed = min(dH_L,dH_R); right_speed = max(dH_L,dH_R) */
	  left_speed = tmp_left; right_speed = tmp_right; 
	  if (tmp_right < tmp_left)
	    {
	      left_speed = tmp_right; right_speed = tmp_left; 
	    }
	  u_left = u[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary+
		     left_ebN_element*nQuadraturePoints_elementBoundary+
		     k];
	  u_right= u[right_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary+
		     right_ebN_element*nQuadraturePoints_elementBoundary+
		     k];
	  if (left_speed < 0.0)/*inflow for left*/
	    {
	      left_flux = fabs(left_speed)*(u_left-u_right);
	      flux[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary +
		   left_ebN_element*nQuadraturePoints_elementBoundary+ k ] = left_flux;
	      dflux_left[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary +
			 left_ebN_element*nQuadraturePoints_elementBoundary+ k ] = fabs(left_speed);
	      dflux_right[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary +
			  left_ebN_element*nQuadraturePoints_elementBoundary+ k ] =-fabs(left_speed);
	  
            }
	  else
	    {
	      left_flux = 0.0;
	      flux[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary +
		   left_ebN_element*nQuadraturePoints_elementBoundary+ k ] = left_flux;
	      dflux_left[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary +
			 left_ebN_element*nQuadraturePoints_elementBoundary+ k ] = 0.0;
	      dflux_right[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary +
			  left_ebN_element*nQuadraturePoints_elementBoundary+ k ] = 0.0;

	    }
	  if (right_speed > 0.0)/*inflow for right*/
	    {
	      right_flux = fabs(right_speed)*(u_right-u_left);
	      flux[right_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary +
		   right_ebN_element*nQuadraturePoints_elementBoundary+ k ] = right_flux;
	      dflux_left[right_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary +
			 right_ebN_element*nQuadraturePoints_elementBoundary+ k ] = -fabs(right_speed);
	      dflux_right[right_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary +
			  right_ebN_element*nQuadraturePoints_elementBoundary+ k ] = fabs(right_speed);
	    }
	  else
	    {
	      right_flux = 0.0;
	      flux[right_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary +
		   right_ebN_element*nQuadraturePoints_elementBoundary+ k ] = right_flux;
	      dflux_left[right_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary +
			 right_ebN_element*nQuadraturePoints_elementBoundary+ k ] = 0.0;
	      dflux_right[right_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary +
			  right_ebN_element*nQuadraturePoints_elementBoundary+ k ] = 0.0;
	    }
	  
        }/*k*/
    }/*ebnI*/
}
void calculateExteriorLesaintRaviartNumericalFlux(proteus_index_t nExteriorElementBoundaries_global,
						  int nElementBoundaries_element,
						  int nQuadraturePoints_elementBoundary,
						  int nSpace,
						  int speedEvalFlag,						  
						  proteus_index_t* exteriorElementBoundaries,
						  proteus_index_t* elementBoundaryElements,
						  int* elementBoundaryLocalElementBoundaries,
						  int* isDOFBoundary,
						  int* inflowFlag,
						  double* n,
						  double* bc_u,
						  double* bc_H,
						  double* bc_dH,
						  double* u,
						  double* H,
						  double* dH,
						  double* flux,
						  double* dflux)
{
  proteus_index_t ebN, eN_global; int ebNE, ebN_element, k, J;
  double left_flux,right_flux,u_left,u_right,left_speed,right_speed,tmp_left,tmp_right;
  /*for now use outer normal at first quadrature point for element speed calculations*/
  
  for(ebNE=0;ebNE<nExteriorElementBoundaries_global;ebNE++)
    {
      ebN = exteriorElementBoundaries[ebNE];
      eN_global = elementBoundaryElements[ebN*2+0];
      ebN_element = elementBoundaryLocalElementBoundaries[ebN*2+0];
      /*
	dH_{L/R} = dH_{eN_left/eN_right} . n_{eN_left}

	dH_min = min(dH_L,dH_R), dH_max = max(dH_L,dH_R)

	if dH_min  < 0
           flux_eN_left = |dH_min| (u^{L}- u^{R})
        else
           flux_eN_left = 0
        if dH_max  > 0 
           flux_eN_right = |dH_max| (u^{R}-u^{L})
	else
           flux_eN_right = 0
      */

      for(k=0;k<nQuadraturePoints_elementBoundary;k++)
        {
	  left_speed =0.0;
	  right_speed=0.0;
          left_flux=0.0;
          right_flux=0.0;
	  tmp_left = 0.0; 
	  tmp_right = 0.0;
          for(J=0;J<nSpace;J++)
            {
              tmp_left 
                += 
                n[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                  ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                  k*nSpace+
		  J]
                *
                dH[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                   ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                   k*nSpace+
		   J];
              tmp_right 
                += 
                n[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                  ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                  k*nSpace+
		  J]
                *
                bc_dH[ebNE*nQuadraturePoints_elementBoundary*nSpace+
		      k*nSpace+
		      J];
	    }
	  /*left_speed = min(dH_L,dH_R); right_speed = max(dH_L,dH_R) */
	  left_speed = tmp_left; right_speed = tmp_right; 
	  if (tmp_right < tmp_left)
	    {
	      left_speed = tmp_right; right_speed = tmp_left; 
	    }
	  u_left = u[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary+
		     ebN_element*nQuadraturePoints_elementBoundary+
		     k];
	  u_right= bc_u[ebNE*nQuadraturePoints_elementBoundary+
			k];
	  if (left_speed < 0.0)/*inflow for left*/
	    {
	      left_flux = fabs(left_speed)*(u_left-u_right);
	      flux[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary +
		   ebN_element*nQuadraturePoints_elementBoundary+ k ] = left_flux;
	      dflux[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary +
		    ebN_element*nQuadraturePoints_elementBoundary+ k ] = fabs(left_speed);
            }
	  else
	    {
	      left_flux = 0.0;
	      flux[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary +
		   ebN_element*nQuadraturePoints_elementBoundary+ k ] = left_flux;
	      dflux[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary +
		    ebN_element*nQuadraturePoints_elementBoundary+ k ] = 0.0;
	    }
        }/*k*/
    }/*ebnE*/
}
void calculateGlobalExteriorLesaintRaviartNumericalFlux(proteus_index_t nExteriorElementBoundaries_global,
							int nQuadraturePoints_elementBoundary,
							int nSpace,
							int speedEvalFlag,
							proteus_index_t* exteriorElementBoundaries,
							proteus_index_t* elementBoundaryElements,
							int* elementBoundaryLocalElementBoundaries,
							int* isDOFBoundary,
							int* inflowFlag,
							double* n,
							double* bc_u,
							double* bc_H,
							double* bc_dH,
							double* u,
							double* H,
							double* dH,
							double* flux,
							double* dflux)
{
  proteus_index_t ebN, eN_global; int ebNE, k, J;
  double left_flux,right_flux,u_left,u_right,left_speed,right_speed,tmp_left,tmp_right;

  /*for now use outer normal at first quadrature point for element speed calculations*/
  
  for(ebNE=0;ebNE<nExteriorElementBoundaries_global;ebNE++)
    {
      ebN = exteriorElementBoundaries[ebNE];
      eN_global = elementBoundaryElements[ebN*2+0];
      /*
	dH_{L/R} = dH_{eN_left/eN_right} . n_{eN_left}

	dH_min = min(dH_L,dH_R), dH_max = max(dH_L,dH_R)

	if dH_min  < 0
           flux_eN_left = |dH_min| (u^{L}- u^{R})
        else
           flux_eN_left = 0
        if dH_max  > 0 
           flux_eN_right = |dH_max| (u^{R}-u^{L})
	else
           flux_eN_right = 0
      */
      for(k=0;k<nQuadraturePoints_elementBoundary;k++)
        {
	  left_speed =0.0;
	  right_speed=0.0;
          left_flux=0.0;
          right_flux=0.0;
	  tmp_left = 0.0; 
	  tmp_right = 0.0;
          for(J=0;J<nSpace;J++)
            {
              tmp_left 
                += 
                n[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                  k*nSpace+
		  J]
                *
                dH[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                   k*nSpace+
		   J];
              tmp_right
                += 
                n[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                  k*nSpace+
		  J]
                *
                bc_dH[ebNE*nQuadraturePoints_elementBoundary*nSpace+
		      k*nSpace+
		      J];
            }
	  /*left_speed = min(dH_L,dH_R); right_speed = max(dH_L,dH_R) */
	  left_speed = tmp_left; right_speed = tmp_right; 
	  if (tmp_right < tmp_left)
	    {
	      left_speed = tmp_right; right_speed = tmp_left; 
	    }
	  u_left = u[ebNE*nQuadraturePoints_elementBoundary+
		     k];
	  u_right= bc_u[ebNE*nQuadraturePoints_elementBoundary+
			k];

	  if (left_speed < 0.0)/*inflow for left*/
	    {
	      left_flux = fabs(left_speed)*(u_left-u_right);
	      flux[ebNE*nQuadraturePoints_elementBoundary + k ] = left_flux;
	      dflux[ebNE*nQuadraturePoints_elementBoundary + k ] = fabs(left_speed);
            }
	  else
	    {
	      left_flux = 0.0;
	      flux[ebNE*nQuadraturePoints_elementBoundary + k ] = left_flux;
	      dflux[ebNE*nQuadraturePoints_elementBoundary + k ] = 0.0;
	    }

        }/*k*/
    }/*ebnE*/
}
void calculateGlobalExteriorNumericalFluxDarcyFCFF(proteus_index_t nExteriorElementBoundaries_global,
						   int nQuadraturePoints_elementBoundary,
						   int nSpace,
						   const proteus_index_t* exteriorElementBoundaries,
						   const proteus_index_t* elementBoundaryElements,
						   const int* elementBoundaryLocalElementBoundaries,
						   const int* isDOFBoundary_uw,
						   const int* isDOFBoundary_um,
						   const double* n,
						   const double* bc_f_m,  
						   const double* bc_a_wm,      
						   const double* bc_a_mw,      
						   const double* bc_a_mm,      
						   const double* bc_grad_phi_w,
						   const double* bc_grad_phi_m,
						   const double* bc_u_w,        
						   const double* bc_u_m,        
						   const double* f_m,          /*lambda_n K_s g(b rho_n-rho_w)*/
						   const double* df_m_dw,          /*dlambda_n K_s g(b rho_n-rho_w)*/
						   const double* a_wm,         /*lambda_w K_s*/
						   const double* a_mw,         /*lambda_n K_s*/
						   const double* a_mm,         /*lambda_t K_s*/
						   const double* grad_phi_w,   /*psi_c*/
						   const double* grad_phi_m,   /*psi_w - rho g . x*/
						   const double* u_w,           /*S_w*/
						   const double* u_m,           /*psi_w*/
						   const double* penalty_w,    
						   const double* penalty_m,
						   double * advectiveFlux_m,   
						   double * dadvectiveFlux_m_dw,   
						   double * diffusiveFlux_wm,
						   double * diffusiveFlux_mw,
						   double * diffusiveFlux_mm)
{
  proteus_index_t ebN; int ebNE, I, J, k, nSpace2=nSpace*nSpace;
  double diffusiveFlux_I=0.0,penaltyFlux = 0.0;

  for (ebNE = 0; ebNE < nExteriorElementBoundaries_global; ebNE++)
    {
      ebN = exteriorElementBoundaries[ebNE];
      for (k = 0; k < nQuadraturePoints_elementBoundary; k++)
	{
	  /*compute diffusive flux for first (w) equation (aq. mass
	    balance for part of boundary where u_0 (i.e., S_w) is
	    specified*/
	  diffusiveFlux_wm[ebNE*nQuadraturePoints_elementBoundary+k] = 0.0;
	  advectiveFlux_m[ebNE*nQuadraturePoints_elementBoundary+k] = 0.0;
	  diffusiveFlux_mw[ebNE*nQuadraturePoints_elementBoundary+k] = 0.0;
	  diffusiveFlux_mm[ebNE*nQuadraturePoints_elementBoundary+k] = 0.0;
	  
	  dadvectiveFlux_m_dw[ebNE*nQuadraturePoints_elementBoundary+k] = 0.0;
	  if (isDOFBoundary_uw[ebNE*nQuadraturePoints_elementBoundary+k] == 1)
	    {
	      /*integration by parts term for diffusive flux*/
	      for (I = 0; I < nSpace; I++)
		{
		  diffusiveFlux_I = 0.0;
		  for (J = 0; J < nSpace; J++)
		    {
		      diffusiveFlux_I -= 
			a_wm[ebNE*nQuadraturePoints_elementBoundary*nSpace2 + 
			     k*nSpace2 + 
			     I*nSpace + 
			     J]
			*
			grad_phi_m[ebNE*nQuadraturePoints_elementBoundary*nSpace + 
				   k*nSpace + 
				   J];
		    }
		  diffusiveFlux_wm[ebNE*nQuadraturePoints_elementBoundary+k] += 
		    diffusiveFlux_I
		    *
		    n[ebNE*nQuadraturePoints_elementBoundary*nSpace +
		      k*nSpace+
		      I];
		}/*I, a_wm grad phi_m term */
	      /*boundary penalty term*/
	      penaltyFlux = 
		penalty_w[ebNE*nQuadraturePoints_elementBoundary + k]
		*
		(u_w[ebNE*nQuadraturePoints_elementBoundary + k]
		 -
		 bc_u_w[ebNE*nQuadraturePoints_elementBoundary + k]);
	      diffusiveFlux_wm[ebNE*nQuadraturePoints_elementBoundary +k] += 
		penaltyFlux;
	    }/*u_w boundary*/
	  if(isDOFBoundary_um[ebNE*nQuadraturePoints_elementBoundary+k] == 1)
	    {
	      /*just evaluate the 'advective flux term for the mixture equation 
		using the internal value since gravity term actually depends on 
		u_w = S_w and not u_m = psi_w (unless it's compressible)*/
	      for (I = 0; I < nSpace; I++)
		{
		  advectiveFlux_m[ebNE*nQuadraturePoints_elementBoundary + k] += 
		    f_m[ebNE*nQuadraturePoints_elementBoundary*nSpace + 
			k*nSpace + 
			I]
		    *
		    n[ebNE*nQuadraturePoints_elementBoundary*nSpace + 
		      k*nSpace + 
		      I];
		  /*basic fc_ff class doesn't have compressibility*/
/* 		  dadvectiveFlux_m_dm[ebNE*nQuadraturePoints_elementBoundary + k] +=  */
/* 		    df_m_dm[ebNE*nQuadraturePoints_elementBoundary*nSpace +  */
/* 			    k*nSpace +  */
/* 			    I] */
/* 		    * */
/* 		    n[ebNE*nQuadraturePoints_elementBoundary*nSpace +  */
/* 		      k*nSpace +  */
/* 		      I]; */
		  dadvectiveFlux_m_dw[ebNE*nQuadraturePoints_elementBoundary + k] += 
		    df_m_dw[ebNE*nQuadraturePoints_elementBoundary*nSpace + 
			    k*nSpace + 
			    I]
		    *
		    n[ebNE*nQuadraturePoints_elementBoundary*nSpace + 
		      k*nSpace + 
		      I];
		  
		}/*calculation of advective flux*/
	      /*integration by parts term for diffusive flux wrt phi_w = psi_c*/
	      for (I = 0; I < nSpace; I++)
		{
		  diffusiveFlux_I = 0.0;
		  for (J = 0; J < nSpace; J++)
		    {
		      diffusiveFlux_I -= 
			a_mw[ebNE*nQuadraturePoints_elementBoundary*nSpace2 + 
			     k*nSpace2 +
			     I*nSpace + 
			     J]
			*
			grad_phi_w[ebNE*nQuadraturePoints_elementBoundary*nSpace + 
				   k*nSpace +
				   J];
		    }/*J*/
		  diffusiveFlux_mw[ebNE*nQuadraturePoints_elementBoundary + k] += 
		    diffusiveFlux_I
		    *
		    n[ebNE*nQuadraturePoints_elementBoundary*nSpace + 
		      k*nSpace + 
		      I];
		}/*I, a_mw grad phi_w term */
	      /*integration by parts term for diffusive flux wrt phi_m = psi_w*/
	      for (I = 0; I < nSpace; I++)
		{
		  diffusiveFlux_I = 0.0;
		  for (J = 0; J < nSpace; J++)
		    {
		      diffusiveFlux_I -= 
			a_mm[ebNE*nQuadraturePoints_elementBoundary*nSpace2 + 
			     k*nSpace2 +
			     I*nSpace + 
			     J]
			*
			grad_phi_m[ebNE*nQuadraturePoints_elementBoundary*nSpace + 
				   k*nSpace +
				   J];
		    }/*J*/
		  diffusiveFlux_mm[ebNE*nQuadraturePoints_elementBoundary + k] += 
		    diffusiveFlux_I
		    *
		    n[ebNE*nQuadraturePoints_elementBoundary*nSpace + 
		      k*nSpace + 
		      I];
		}/*I, a_mw grad phi_w term */
	      /*boundary penalty term*/
	      penaltyFlux = 
		penalty_m[ebNE*nQuadraturePoints_elementBoundary + k]
		*
		(u_m[ebNE*nQuadraturePoints_elementBoundary + k]
		 -
		 bc_u_m[ebNE*nQuadraturePoints_elementBoundary + k]);
	      diffusiveFlux_mm[ebNE*nQuadraturePoints_elementBoundary +k] += 
		penaltyFlux;
	    }/*um boundary*/
	}/*k*/
    }/*ebNE*/
}
void calculateGlobalExteriorNumericalFluxDarcyFCFF_sd(proteus_index_t nExteriorElementBoundaries_global,
						      int nQuadraturePoints_elementBoundary,
						      int nSpace,
						      proteus_csr_index_t* rowptr_wm,
						      proteus_csr_index_t* colind_wm,
						      proteus_csr_index_t* rowptr_mw,
						      proteus_csr_index_t* colind_mw,
						      proteus_csr_index_t* rowptr_mm,
						      proteus_csr_index_t* colind_mm,
						      const proteus_index_t* exteriorElementBoundaries,
						      const proteus_index_t* elementBoundaryElements,
						      const int* elementBoundaryLocalElementBoundaries,
						      const int* isDOFBoundary_uw,
						      const int* isDOFBoundary_um,
						      const double* n,
						      const double* bc_f_m,  
						      const double* bc_a_wm,      
						      const double* bc_a_mw,      
						      const double* bc_a_mm,      
						      const double* bc_grad_phi_w,
						      const double* bc_grad_phi_m,
						      const double* bc_u_w,        
						      const double* bc_u_m,        
						      const double* f_m,          /*lambda_n K_s g(b rho_n-rho_w)*/
						      const double* df_m_dw,          /*dlambda_n K_s g(b rho_n-rho_w)*/
						      const double* a_wm,         /*lambda_w K_s*/
						      const double* a_mw,         /*lambda_n K_s*/
						      const double* a_mm,         /*lambda_t K_s*/
						      const double* grad_phi_w,   /*psi_c*/
						      const double* grad_phi_m,   /*psi_w - rho g . x*/
						      const double* u_w,           /*S_w*/
						      const double* u_m,           /*psi_w*/
						      const double* penalty_w,    
						      const double* penalty_m,
						      double * advectiveFlux_m,   
						      double * dadvectiveFlux_m_dw,   
						      double * diffusiveFlux_wm,
						      double * diffusiveFlux_mw,
						      double * diffusiveFlux_mm)
{
  proteus_index_t ebN; proteus_csr_index_t m; proteus_csr_index_t nnz_mm=rowptr_mm[nSpace]; proteus_csr_index_t nnz_wm=rowptr_wm[nSpace]; proteus_csr_index_t nnz_mw=rowptr_mw[nSpace]; int ebNE, I, k;
  double diffusiveFlux_I=0.0,penaltyFlux = 0.0;

  for (ebNE = 0; ebNE < nExteriorElementBoundaries_global; ebNE++)
    {
      ebN = exteriorElementBoundaries[ebNE];
      for (k = 0; k < nQuadraturePoints_elementBoundary; k++)
	{
	  /*compute diffusive flux for first (w) equation (aq. mass
	    balance for part of boundary where u_0 (i.e., S_w) is
	    specified*/
	  diffusiveFlux_wm[ebNE*nQuadraturePoints_elementBoundary+k] = 0.0;
	  advectiveFlux_m[ebNE*nQuadraturePoints_elementBoundary+k] = 0.0;
	  diffusiveFlux_mw[ebNE*nQuadraturePoints_elementBoundary+k] = 0.0;
	  diffusiveFlux_mm[ebNE*nQuadraturePoints_elementBoundary+k] = 0.0;
	  
	  dadvectiveFlux_m_dw[ebNE*nQuadraturePoints_elementBoundary+k] = 0.0;
	  if (isDOFBoundary_uw[ebNE*nQuadraturePoints_elementBoundary+k] == 1)
	    {
	      /*integration by parts term for diffusive flux*/
	      for (I = 0; I < nSpace; I++)
		{
		  diffusiveFlux_I = 0.0;
		  for(m=rowptr_wm[I];m<rowptr_wm[I+1];m++)
		    {
		      diffusiveFlux_I -= 
			a_wm[ebNE*nQuadraturePoints_elementBoundary*nnz_wm+
			     k*nnz_wm+
			     m]
			*
			grad_phi_m[ebNE*nQuadraturePoints_elementBoundary*nSpace + 
				   k*nSpace + 
				   colind_wm[m]];
		    }
		  diffusiveFlux_wm[ebNE*nQuadraturePoints_elementBoundary+k] += 
		    diffusiveFlux_I
		    *
		    n[ebNE*nQuadraturePoints_elementBoundary*nSpace +
		      k*nSpace+
		      I];
		}/*I, a_wm grad phi_m term */
	      /*boundary penalty term*/
	      penaltyFlux = 
		penalty_w[ebNE*nQuadraturePoints_elementBoundary + k]
		*
		(u_w[ebNE*nQuadraturePoints_elementBoundary + k]
		 -
		 bc_u_w[ebNE*nQuadraturePoints_elementBoundary + k]);
	      diffusiveFlux_wm[ebNE*nQuadraturePoints_elementBoundary +k] += 
		penaltyFlux;
	    }/*u_w boundary*/
	  if(isDOFBoundary_um[ebNE*nQuadraturePoints_elementBoundary+k] == 1)
	    {
	      /*just evaluate the 'advective flux term for the mixture equation 
		using the internal value since gravity term actually depends on 
		u_w = S_w and not u_m = psi_w (unless it's compressible)*/
	      for (I = 0; I < nSpace; I++)
		{
		  advectiveFlux_m[ebNE*nQuadraturePoints_elementBoundary + k] += 
		    f_m[ebNE*nQuadraturePoints_elementBoundary*nSpace + 
			k*nSpace + 
			I]
		    *
		    n[ebNE*nQuadraturePoints_elementBoundary*nSpace + 
		      k*nSpace + 
		      I];
		  /*basic fc_ff class doesn't have compressibility*/
/* 		  dadvectiveFlux_m_dm[ebNE*nQuadraturePoints_elementBoundary + k] +=  */
/* 		    df_m_dm[ebNE*nQuadraturePoints_elementBoundary*nSpace +  */
/* 			    k*nSpace +  */
/* 			    I] */
/* 		    * */
/* 		    n[ebNE*nQuadraturePoints_elementBoundary*nSpace +  */
/* 		      k*nSpace +  */
/* 		      I]; */
		  dadvectiveFlux_m_dw[ebNE*nQuadraturePoints_elementBoundary + k] += 
		    df_m_dw[ebNE*nQuadraturePoints_elementBoundary*nSpace + 
			    k*nSpace + 
			    I]
		    *
		    n[ebNE*nQuadraturePoints_elementBoundary*nSpace + 
		      k*nSpace + 
		      I];
		  
		}/*calculation of advective flux*/
	      /*integration by parts term for diffusive flux wrt phi_w = psi_c*/
	      for (I = 0; I < nSpace; I++)
		{
		  diffusiveFlux_I = 0.0;
		  for(m=rowptr_mw[I];m<rowptr_mw[I+1];m++)
		    {
		      diffusiveFlux_I -= 
			a_mw[ebNE*nQuadraturePoints_elementBoundary*nnz_mw+
			     k*nnz_mw+
			     m]
			*
			grad_phi_w[ebNE*nQuadraturePoints_elementBoundary*nSpace + 
				   k*nSpace +
				   colind_mw[m]];
		    }/*J*/
		  diffusiveFlux_mw[ebNE*nQuadraturePoints_elementBoundary + k] += 
		    diffusiveFlux_I
		    *
		    n[ebNE*nQuadraturePoints_elementBoundary*nSpace + 
		      k*nSpace + 
		      I];
		}/*I, a_mw grad phi_w term */
	      /*integration by parts term for diffusive flux wrt phi_m = psi_w*/
	      for (I = 0; I < nSpace; I++)
		{
		  diffusiveFlux_I = 0.0;
		  for(m=rowptr_mm[I];m<rowptr_mm[I+1];m++)
		    {
		      diffusiveFlux_I -= 
			a_mm[ebNE*nQuadraturePoints_elementBoundary*nnz_mm+
			     k*nnz_mm+
			     m]
			*
			grad_phi_m[ebNE*nQuadraturePoints_elementBoundary*nSpace + 
				   k*nSpace +
				   colind_mm[m]];
		    }/*J*/
		  diffusiveFlux_mm[ebNE*nQuadraturePoints_elementBoundary + k] += 
		    diffusiveFlux_I
		    *
		    n[ebNE*nQuadraturePoints_elementBoundary*nSpace + 
		      k*nSpace + 
		      I];
		}/*I, a_mw grad phi_w term */
	      /*boundary penalty term*/
	      penaltyFlux = 
		penalty_m[ebNE*nQuadraturePoints_elementBoundary + k]
		*
		(u_m[ebNE*nQuadraturePoints_elementBoundary + k]
		 -
		 bc_u_m[ebNE*nQuadraturePoints_elementBoundary + k]);
	      diffusiveFlux_mm[ebNE*nQuadraturePoints_elementBoundary +k] += 
		penaltyFlux;
	    }/*um boundary*/
	}/*k*/
    }/*ebNE*/
}

void calculateGlobalExteriorNumericalFluxDarcyFCFF_diffusiveFluxJacobian(proteus_index_t nExteriorElementBoundaries_global,
									 int nQuadraturePoints_elementBoundary,
									 int nSpace,
									 int nDOF_trial_element,
									 const proteus_index_t* l2g, /*for now assumes both solution spaces are the same!*/
									 const proteus_index_t* exteriorElementBoundaries,
									 const proteus_index_t* elementBoundaryElements,
									 const int* elementBoundaryLocalElementBoundaries,
									 const int* isDOFBoundary_uw,
									 const int* isDOFBoundary_um,
									 const double* n,
									 const double* f_m,          /*lambda_n K_s g(b rho_n-rho_w)*/
									 const double* df_m_dw,          /*dlambda_n K_s g(b rho_n-rho_w)*/
									 const double* a_wm,         /*lambda_w K_s*/
									 const double* da_wm_dw,         /* a' wrt S_w*/
									 const double* da_wm_dm,         /* a' wrt psi_w*/
									 const double* a_mw,         /*lambda_n K_s*/
									 const double* da_mw_dw,         /* a' wrt S_w*/
									 const double* da_mw_dm,         /* a' wrt psi_w*/
									 const double* a_mm,         /*lambda_t K_s*/
									 const double* da_mm_dw,         /* a' wrt S_w*/
									 const double* da_mm_dm,         /* a' wrt psi_w*/
									 const double* grad_phi_w,   /*psi_c*/
									 const double* grad_phi_m,   /*psi_w - rho g . x*/
									 const double* dphi_w_w,     /*\pd{psi_c}{S_w} */     
									 const double* dphi_w_m,     /*\pd{psi_c}{psi_w}=  0 */
									 const double* dphi_m_w,     /*\pd{phi_w}{S_w} = 0 */
									 const double* dphi_m_m,     /*\pd{phi_w}{psi_w} = 1 - drho/dpsi_w g . x */
									 const double* u_w,           /*S_w*/
									 const double* u_m,           /*psi_w*/
									 const double* v,            /*trial functions, assumed in same space*/
									 const double* grad_v,       /*trial function gradients, assumed in same space*/
									 const double* penalty_w,    
									 const double* penalty_m,
									 double * fluxJacobian_ww,
									 double * fluxJacobian_wm,
									 double * fluxJacobian_mw,
									 double * fluxJacobian_mm)
{
  proteus_index_t ebN, eN_global, j_global; int ebNE, j, I, J, k, nSpace2=nSpace*nSpace;
  double Jacobian_w,Jacobian_m,
    diffusiveVelocityComponent_I_Jacobian_w,
    diffusiveVelocityComponent_I_Jacobian_m,  
    diffusiveVelocityComponent_I_Jacobian2_wm,
    diffusiveVelocityComponent_I_Jacobian2_ww,
    diffusiveVelocityComponent_I_Jacobian2_mw,
    diffusiveVelocityComponent_I_Jacobian2_mm;

  for (ebNE = 0; ebNE < nExteriorElementBoundaries_global; ebNE++)
    {
      ebN = exteriorElementBoundaries[ebNE];
      eN_global = elementBoundaryElements[ebN*2 + 0];
      for (k = 0; k < nQuadraturePoints_elementBoundary; k++)
	{
	  /*compute derivative of diffusive flux for first (w) equation (aq. mass
	    balance for part of boundary where u_0 (i.e., S_w) is
	    specified
	  */
	  
	  if (isDOFBoundary_uw[ebNE*nQuadraturePoints_elementBoundary + k] == 1)
	    {
	      /*NOTE! assuming u_w, u_m in the same space and nodal interpolant for potential*/
	      
	      for (j = 0; j < nDOF_trial_element; j++)
		{
		  Jacobian_w = 0.; /*derivative wrt u_w = S_w */
		  Jacobian_m = 0.; /*derivative wrt u_m = psi_w */
		  j_global = l2g[eN_global*nDOF_trial_element + j];/*assuming same space for both*/ 
		  for (I = 0; I < nSpace; I++)
		    {
		      diffusiveVelocityComponent_I_Jacobian_w = 0.0; 
		      diffusiveVelocityComponent_I_Jacobian_m = 0.0; 
		      diffusiveVelocityComponent_I_Jacobian2_mw = 0.0;
		      diffusiveVelocityComponent_I_Jacobian2_mm = 0.0;
		      for (J = 0; J < nSpace; J++)
			{
			  /*only a_wm potential here*/
			  diffusiveVelocityComponent_I_Jacobian_w -= 
			    da_wm_dw[ebNE*nQuadraturePoints_elementBoundary*nSpace2 + 
				     k*nSpace2 + 
				     I*nSpace  + 
				     J]
			    *
			    grad_phi_m[ebNE*nQuadraturePoints_elementBoundary*nSpace + 
				       k*nSpace + 
				       J];
			  diffusiveVelocityComponent_I_Jacobian_m -= 
			    da_wm_dm[ebNE*nQuadraturePoints_elementBoundary*nSpace2 + 
				     k*nSpace2 + 
				     I*nSpace  + 
				     J]
			    *
			    grad_phi_m[ebNE*nQuadraturePoints_elementBoundary*nSpace + 
				       k*nSpace + 
				       J];

			  diffusiveVelocityComponent_I_Jacobian2_mw -=
			    a_wm[ebNE*nQuadraturePoints_elementBoundary*nSpace2 + 
				 k*nSpace2 + 
				 I*nSpace  + 
				 J]
			    * /*should be grad_v_m in general I believe*/
			    grad_v[ebNE*nQuadraturePoints_elementBoundary*nDOF_trial_element*nSpace + 
				   k*nDOF_trial_element*nSpace+
                                   j*nSpace+
                                   J];
			  /*identical for now*/
			  diffusiveVelocityComponent_I_Jacobian2_mm -=
			    a_wm[ebNE*nQuadraturePoints_elementBoundary*nSpace2 + 
				 k*nSpace2 + 
				 I*nSpace  + 
				 J]
			    * /*should be grad_v_m in general I believe*/
			    grad_v[ebNE*nQuadraturePoints_elementBoundary*nDOF_trial_element*nSpace + 
				   k*nDOF_trial_element*nSpace+
                                   j*nSpace+
                                   J];
			    
			}/*J loop*/
		      Jacobian_w += 
			diffusiveVelocityComponent_I_Jacobian_w 
			*/*should be v_w*/
			v[ebNE*nQuadraturePoints_elementBoundary*nDOF_trial_element +
			  k*nDOF_trial_element +
			  j]
			*
			n[ebNE*nQuadraturePoints_elementBoundary*nSpace + 
			  k*nSpace + 
			  I];
		      Jacobian_w += 
			diffusiveVelocityComponent_I_Jacobian2_mw 
			*
			dphi_m_w[j_global]
			*
			n[ebNE*nQuadraturePoints_elementBoundary*nSpace + 
			  k*nSpace + 
			  I];
		      Jacobian_m += 
			diffusiveVelocityComponent_I_Jacobian_m 
			*/*should be v_m*/
			v[ebNE*nQuadraturePoints_elementBoundary*nDOF_trial_element +
			  k*nDOF_trial_element +
			  j]
			*
			n[ebNE*nQuadraturePoints_elementBoundary*nSpace + 
			  k*nSpace + 
			  I];
		      Jacobian_m += 
			diffusiveVelocityComponent_I_Jacobian2_mm 
			*
			dphi_m_m[j_global]
			*
			n[ebNE*nQuadraturePoints_elementBoundary*nSpace + 
			  k*nSpace + 
			  I];
		    }/*I loop */
		  /*only diagonal gets penalty term*/
		  Jacobian_w += 
		    penalty_w[ebNE*nQuadraturePoints_elementBoundary+k]
		    * /*should be v_w*/
		    v[ebNE*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                      k*nDOF_trial_element+
                      j];
		  fluxJacobian_ww[ebNE*nQuadraturePoints_elementBoundary*nDOF_trial_element + 
				  k*nDOF_trial_element +
				  j] += 
		    Jacobian_w;
		  fluxJacobian_wm[ebNE*nQuadraturePoints_elementBoundary*nDOF_trial_element + 
				  k*nDOF_trial_element +
				  j] += 
		    Jacobian_m;
		}/* j local dof loop*/
	    }/*u_w dof boundary loop*/
	  if (isDOFBoundary_um[ebNE*nQuadraturePoints_elementBoundary + k] == 1)
	    {
	      /*NOTE! assuming u_w, u_m in the same space and nodal interpolant for potential*/
	      
	      for (j = 0; j < nDOF_trial_element; j++)
		{
		  Jacobian_w = 0.; /*derivative wrt u_w = S_w */
		  Jacobian_m = 0.; /*derivative wrt u_m = psi_w */
		  j_global = l2g[eN_global*nDOF_trial_element + j];/*assuming same space for both*/ 
		  for (I = 0; I < nSpace; I++)
		    {
		      diffusiveVelocityComponent_I_Jacobian_w = 0.0;
		      diffusiveVelocityComponent_I_Jacobian_m = 0.0;  
		      diffusiveVelocityComponent_I_Jacobian2_wm = 0.0;
		      diffusiveVelocityComponent_I_Jacobian2_ww = 0.0;
		      diffusiveVelocityComponent_I_Jacobian2_mw = 0.0;
		      diffusiveVelocityComponent_I_Jacobian2_mm = 0.0;
		      for (J = 0; J < nSpace; J++)
			{
			  /*mw potential*/
			  diffusiveVelocityComponent_I_Jacobian_w -= 
			    da_mw_dw[ebNE*nQuadraturePoints_elementBoundary*nSpace2 + 
				     k*nSpace2 + 
				     I*nSpace  + 
				     J]
			    *
			    grad_phi_w[ebNE*nQuadraturePoints_elementBoundary*nSpace + 
				       k*nSpace + 
				       J];
			  diffusiveVelocityComponent_I_Jacobian_m -= 
			    da_mw_dm[ebNE*nQuadraturePoints_elementBoundary*nSpace2 + 
				     k*nSpace2 + 
				     I*nSpace  + 
				     J]
			    *
			    grad_phi_w[ebNE*nQuadraturePoints_elementBoundary*nSpace + 
				       k*nSpace + 
				       J];
			  /*mm potential*/
			  diffusiveVelocityComponent_I_Jacobian_w -= 
			    da_mm_dw[ebNE*nQuadraturePoints_elementBoundary*nSpace2 + 
				     k*nSpace2 + 
				     I*nSpace  + 
				     J]
			    *
			    grad_phi_m[ebNE*nQuadraturePoints_elementBoundary*nSpace + 
				       k*nSpace + 
				       J];
			  diffusiveVelocityComponent_I_Jacobian_m -= 
			    da_mm_dm[ebNE*nQuadraturePoints_elementBoundary*nSpace2 + 
				     k*nSpace2 + 
				     I*nSpace  + 
				     J]
			    *
			    grad_phi_m[ebNE*nQuadraturePoints_elementBoundary*nSpace + 
				       k*nSpace + 
				       J];

			  /*mw potential*/
			  diffusiveVelocityComponent_I_Jacobian2_ww -=
			    a_mw[ebNE*nQuadraturePoints_elementBoundary*nSpace2 + 
				 k*nSpace2 + 
				 I*nSpace  + 
				 J]
			    * /*should be grad_v_w in general I believe*/
			    grad_v[ebNE*nQuadraturePoints_elementBoundary*nDOF_trial_element*nSpace + 
				   k*nDOF_trial_element*nSpace+
                                   j*nSpace+
                                   J];
			  /*identical for now*/
			  diffusiveVelocityComponent_I_Jacobian2_wm -=
			    a_mw[ebNE*nQuadraturePoints_elementBoundary*nSpace2 + 
				 k*nSpace2 + 
				 I*nSpace  + 
				 J]
			    * /*should be grad_v_m in general I believe*/
			    grad_v[ebNE*nQuadraturePoints_elementBoundary*nDOF_trial_element*nSpace + 
				   k*nDOF_trial_element*nSpace+
                                   j*nSpace+
                                   J];

			  /*mm potential*/
			  diffusiveVelocityComponent_I_Jacobian2_mw -=
			    a_mm[ebNE*nQuadraturePoints_elementBoundary*nSpace2 + 
				 k*nSpace2 + 
				 I*nSpace  + 
				 J]
			    * /*should be grad_v_w in general I believe*/
			    grad_v[ebNE*nQuadraturePoints_elementBoundary*nDOF_trial_element*nSpace + 
				   k*nDOF_trial_element*nSpace+
                                   j*nSpace+
                                   J];
			  /*identical for now*/
			  diffusiveVelocityComponent_I_Jacobian2_mm -=
			    a_mm[ebNE*nQuadraturePoints_elementBoundary*nSpace2 + 
				 k*nSpace2 + 
				 I*nSpace  + 
				 J]
			    * /*should be grad_v_m in general I believe*/
			    grad_v[ebNE*nQuadraturePoints_elementBoundary*nDOF_trial_element*nSpace + 
				   k*nDOF_trial_element*nSpace+
                                   j*nSpace+
                                   J];
			    
			}/*J loop*/
		      Jacobian_w += 
			diffusiveVelocityComponent_I_Jacobian_w 
			*/*should be v_w*/
			v[ebNE*nQuadraturePoints_elementBoundary*nDOF_trial_element +
			  k*nDOF_trial_element +
			  j]
			*
			n[ebNE*nQuadraturePoints_elementBoundary*nSpace + 
			  k*nSpace + 
			  I];
		      Jacobian_w += 
			diffusiveVelocityComponent_I_Jacobian2_ww 
			*
			dphi_w_w[j_global]
			*
			n[ebNE*nQuadraturePoints_elementBoundary*nSpace + 
			  k*nSpace + 
			  I];
		      Jacobian_w += 
			diffusiveVelocityComponent_I_Jacobian2_mw 
			*
			dphi_m_w[j_global]
			*
			n[ebNE*nQuadraturePoints_elementBoundary*nSpace + 
			  k*nSpace + 
			  I];

		      Jacobian_m += 
			diffusiveVelocityComponent_I_Jacobian_m 
			*/*should be v_m*/
			v[ebNE*nQuadraturePoints_elementBoundary*nDOF_trial_element +
			  k*nDOF_trial_element +
			  j]
			*
			n[ebNE*nQuadraturePoints_elementBoundary*nSpace + 
			  k*nSpace + 
			  I];
		      Jacobian_m += 
			diffusiveVelocityComponent_I_Jacobian2_wm 
			*
			dphi_w_m[j_global]
			*
			n[ebNE*nQuadraturePoints_elementBoundary*nSpace + 
			  k*nSpace + 
			  I];
		      Jacobian_m += 
			diffusiveVelocityComponent_I_Jacobian2_mm 
			*
			dphi_m_m[j_global]
			*
			n[ebNE*nQuadraturePoints_elementBoundary*nSpace + 
			  k*nSpace + 
			  I];
		    }/*I loop */
		  /*only diagonal gets penalty term*/
		  Jacobian_m += 
		    penalty_m[ebNE*nQuadraturePoints_elementBoundary+k]
		    * /*should be v_w*/
		    v[ebNE*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                      k*nDOF_trial_element+
                      j];
		  fluxJacobian_mw[ebNE*nQuadraturePoints_elementBoundary*nDOF_trial_element + 
				  k*nDOF_trial_element +
				  j] += 
		    Jacobian_w;
		  fluxJacobian_mm[ebNE*nQuadraturePoints_elementBoundary*nDOF_trial_element + 
				  k*nDOF_trial_element +
				  j] += 
		    Jacobian_m;
		}/* j local dof loop*/
	    }/*u_w dof boundary loop*/
	  
	}/*k*/
    }/*ebNE*/

}

void calculateGlobalExteriorNumericalFluxDarcyFCFF_diffusiveFluxJacobian_sd(proteus_index_t nExteriorElementBoundaries_global,
									    int nQuadraturePoints_elementBoundary,
									    int nSpace,
									    int nDOF_trial_element,
									    proteus_csr_index_t* rowptr_wm,
									    proteus_csr_index_t* colind_wm,
									    proteus_csr_index_t* rowptr_mw,
									    proteus_csr_index_t* colind_mw,
									    proteus_csr_index_t* rowptr_mm,
									    proteus_csr_index_t* colind_mm,
									    const proteus_index_t* l2g, /*for now assumes both solution spaces are the same!*/
									    const proteus_index_t* exteriorElementBoundaries,
									    const proteus_index_t* elementBoundaryElements,
									    const int* elementBoundaryLocalElementBoundaries,
									    const int* isDOFBoundary_uw,
									    const int* isDOFBoundary_um,
									    const double* n,
									    const double* f_m,          /*lambda_n K_s g(b rho_n-rho_w)*/
									    const double* df_m_dw,          /*dlambda_n K_s g(b rho_n-rho_w)*/
									    const double* a_wm,         /*lambda_w K_s*/
									    const double* da_wm_dw,         /* a' wrt S_w*/
									    const double* da_wm_dm,         /* a' wrt psi_w*/
									    const double* a_mw,         /*lambda_n K_s*/
									    const double* da_mw_dw,         /* a' wrt S_w*/
									    const double* da_mw_dm,         /* a' wrt psi_w*/
									    const double* a_mm,         /*lambda_t K_s*/
									    const double* da_mm_dw,         /* a' wrt S_w*/
									    const double* da_mm_dm,         /* a' wrt psi_w*/
									    const double* grad_phi_w,   /*psi_c*/
									    const double* grad_phi_m,   /*psi_w - rho g . x*/
									    const double* dphi_w_w,     /*\pd{psi_c}{S_w} */     
									    const double* dphi_w_m,     /*\pd{psi_c}{psi_w}=  0 */
									    const double* dphi_m_w,     /*\pd{phi_w}{S_w} = 0 */
									    const double* dphi_m_m,     /*\pd{phi_w}{psi_w} = 1 - drho/dpsi_w g . x */
									    const double* u_w,           /*S_w*/
									    const double* u_m,           /*psi_w*/
									    const double* v,            /*trial functions, assumed in same space*/
									    const double* grad_v,       /*trial function gradients, assumed in same space*/
									    const double* penalty_w,    
									    const double* penalty_m,
									    double * fluxJacobian_ww,
									    double * fluxJacobian_wm,
									    double * fluxJacobian_mw,
									    double * fluxJacobian_mm)
{
  proteus_index_t ebN, eN_global, j_global; proteus_csr_index_t m; proteus_csr_index_t nnz_mm=rowptr_mm[nSpace]; proteus_csr_index_t nnz_wm=rowptr_wm[nSpace]; proteus_csr_index_t nnz_mw=rowptr_mw[nSpace]; int ebNE, j, I, k;
  double Jacobian_w,Jacobian_m,
    diffusiveVelocityComponent_I_Jacobian_w,
    diffusiveVelocityComponent_I_Jacobian_m,  
    diffusiveVelocityComponent_I_Jacobian2_wm,
    diffusiveVelocityComponent_I_Jacobian2_ww,
    diffusiveVelocityComponent_I_Jacobian2_mw,
    diffusiveVelocityComponent_I_Jacobian2_mm;

  for (ebNE = 0; ebNE < nExteriorElementBoundaries_global; ebNE++)
    {
      ebN = exteriorElementBoundaries[ebNE];
      eN_global = elementBoundaryElements[ebN*2 + 0];
      for (k = 0; k < nQuadraturePoints_elementBoundary; k++)
	{
	  /*compute derivative of diffusive flux for first (w) equation (aq. mass
	    balance for part of boundary where u_0 (i.e., S_w) is
	    specified
	  */
	  
	  if (isDOFBoundary_uw[ebNE*nQuadraturePoints_elementBoundary + k] == 1)
	    {
	      /*NOTE! assuming u_w, u_m in the same space and nodal interpolant for potential*/
	      
	      for (j = 0; j < nDOF_trial_element; j++)
		{
		  Jacobian_w = 0.; /*derivative wrt u_w = S_w */
		  Jacobian_m = 0.; /*derivative wrt u_m = psi_w */
		  j_global = l2g[eN_global*nDOF_trial_element + j];/*assuming same space for both*/ 
		  for (I = 0; I < nSpace; I++)
		    {
		      diffusiveVelocityComponent_I_Jacobian_w = 0.0; 
		      diffusiveVelocityComponent_I_Jacobian_m = 0.0; 
		      diffusiveVelocityComponent_I_Jacobian2_mw = 0.0;
		      diffusiveVelocityComponent_I_Jacobian2_mm = 0.0;
		      for(m=rowptr_wm[I];m<rowptr_wm[I+1];m++)
			{
			  /*only a_wm potential here*/
			  diffusiveVelocityComponent_I_Jacobian_w -= 
			    da_wm_dw[ebNE*nQuadraturePoints_elementBoundary*nnz_wm+
				     k*nnz_wm+
				     m]
			    *
			    grad_phi_m[ebNE*nQuadraturePoints_elementBoundary*nSpace + 
				       k*nSpace + 
				       colind_wm[m]];
			  diffusiveVelocityComponent_I_Jacobian_m -= 
			    da_wm_dm[ebNE*nQuadraturePoints_elementBoundary*nnz_wm+
				     k*nnz_wm+
				     m]
			    *
			    grad_phi_m[ebNE*nQuadraturePoints_elementBoundary*nSpace + 
				       k*nSpace + 
				       colind_wm[m]];
			  
			  diffusiveVelocityComponent_I_Jacobian2_mw -=
			    a_wm[ebNE*nQuadraturePoints_elementBoundary*nnz_wm+
				 k*nnz_wm+
				 m]
			    * /*should be grad_v_m in general I believe*/
			    grad_v[ebNE*nQuadraturePoints_elementBoundary*nDOF_trial_element*nSpace + 
				   k*nDOF_trial_element*nSpace+
                                   j*nSpace+
                                   colind_wm[m]];
			  /*identical for now*/
			  diffusiveVelocityComponent_I_Jacobian2_mm -=
			    a_wm[ebNE*nQuadraturePoints_elementBoundary*nnz_wm+
				 k*nnz_wm+
				 m]
			    * /*should be grad_v_m in general I believe*/
			    grad_v[ebNE*nQuadraturePoints_elementBoundary*nDOF_trial_element*nSpace + 
				   k*nDOF_trial_element*nSpace+
                                   j*nSpace+
                                   colind_wm[m]];
			  
			}/*J loop*/
		      Jacobian_w += 
			diffusiveVelocityComponent_I_Jacobian_w 
			*/*should be v_w*/
			v[ebNE*nQuadraturePoints_elementBoundary*nDOF_trial_element +
			  k*nDOF_trial_element +
			  j]
			*
			n[ebNE*nQuadraturePoints_elementBoundary*nSpace + 
			  k*nSpace + 
			  I];
		      Jacobian_w += 
			diffusiveVelocityComponent_I_Jacobian2_mw 
			*
			dphi_m_w[j_global]
			*
			n[ebNE*nQuadraturePoints_elementBoundary*nSpace + 
			  k*nSpace + 
			  I];
		      Jacobian_m += 
			diffusiveVelocityComponent_I_Jacobian_m 
			*/*should be v_m*/
			v[ebNE*nQuadraturePoints_elementBoundary*nDOF_trial_element +
			  k*nDOF_trial_element +
			  j]
			*
			n[ebNE*nQuadraturePoints_elementBoundary*nSpace + 
			  k*nSpace + 
			  I];
		      Jacobian_m += 
			diffusiveVelocityComponent_I_Jacobian2_mm 
			*
			dphi_m_m[j_global]
			*
			n[ebNE*nQuadraturePoints_elementBoundary*nSpace + 
			  k*nSpace + 
			  I];
		    }/*I loop */
		  /*only diagonal gets penalty term*/
		  Jacobian_w += 
		    penalty_w[ebNE*nQuadraturePoints_elementBoundary+k]
		    * /*should be v_w*/
		    v[ebNE*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                      k*nDOF_trial_element+
                      j];
		  fluxJacobian_ww[ebNE*nQuadraturePoints_elementBoundary*nDOF_trial_element + 
				  k*nDOF_trial_element +
				  j] += 
		    Jacobian_w;
		  fluxJacobian_wm[ebNE*nQuadraturePoints_elementBoundary*nDOF_trial_element + 
				  k*nDOF_trial_element +
				  j] += 
		    Jacobian_m;
		}/* j local dof loop*/
	    }/*u_w dof boundary loop*/
	  if (isDOFBoundary_um[ebNE*nQuadraturePoints_elementBoundary + k] == 1)
	    {
	      /*NOTE! assuming u_w, u_m in the same space and nodal interpolant for potential*/
	      
	      for (j = 0; j < nDOF_trial_element; j++)
		{
		  Jacobian_w = 0.; /*derivative wrt u_w = S_w */
		  Jacobian_m = 0.; /*derivative wrt u_m = psi_w */
		  j_global = l2g[eN_global*nDOF_trial_element + j];/*assuming same space for both*/ 
		  for (I = 0; I < nSpace; I++)
		    {
		      diffusiveVelocityComponent_I_Jacobian_w = 0.0;
		      diffusiveVelocityComponent_I_Jacobian_m = 0.0;  
		      diffusiveVelocityComponent_I_Jacobian2_wm = 0.0;
		      diffusiveVelocityComponent_I_Jacobian2_ww = 0.0;
		      diffusiveVelocityComponent_I_Jacobian2_mw = 0.0;
		      diffusiveVelocityComponent_I_Jacobian2_mm = 0.0;
		      for (m=rowptr_mw[I];m<rowptr_mw[I+1];m++)
			{
			  /*mw potential*/
			  diffusiveVelocityComponent_I_Jacobian_w -= 
			    da_mw_dw[ebNE*nQuadraturePoints_elementBoundary*nnz_mw + 
				     k*nnz_mw + 
				     m]
			    *
			    grad_phi_w[ebNE*nQuadraturePoints_elementBoundary*nSpace + 
				       k*nSpace + 
				       colind_mw[m]];
			  diffusiveVelocityComponent_I_Jacobian_m -= 
			    da_mw_dm[ebNE*nQuadraturePoints_elementBoundary*nnz_mw + 
				     k*nnz_mw + 
				     m]
			    *
			    grad_phi_w[ebNE*nQuadraturePoints_elementBoundary*nSpace + 
				       k*nSpace + 
				       colind_mw[m]];
			  /*mm potential*/
			  diffusiveVelocityComponent_I_Jacobian_w -= 
			    da_mm_dw[ebNE*nQuadraturePoints_elementBoundary*nnz_mw + 
				     k*nnz_mw + 
				     m]
			    *
			    grad_phi_m[ebNE*nQuadraturePoints_elementBoundary*nSpace + 
				       k*nSpace + 
				       colind_mw[m]];
			  diffusiveVelocityComponent_I_Jacobian_m -= 
			    da_mm_dm[ebNE*nQuadraturePoints_elementBoundary*nnz_mw + 
				     k*nnz_mw + 
				     m]
			    *
			    grad_phi_m[ebNE*nQuadraturePoints_elementBoundary*nSpace + 
				       k*nSpace + 
				       colind_mw[m]];

			  /*mw potential*/
			  diffusiveVelocityComponent_I_Jacobian2_ww -=
			    a_mw[ebNE*nQuadraturePoints_elementBoundary*nnz_mw + 
				 k*nnz_mw + 
				 m]
			    * /*should be grad_v_w in general I believe*/
			    grad_v[ebNE*nQuadraturePoints_elementBoundary*nDOF_trial_element*nSpace + 
				   k*nDOF_trial_element*nSpace+
                                   j*nSpace+
                                   colind_mw[m]];
			  /*identical for now*/
			  diffusiveVelocityComponent_I_Jacobian2_wm -=
			    a_mw[ebNE*nQuadraturePoints_elementBoundary*nnz_mw + 
				 k*nnz_mw + 
				 m]
			    * /*should be grad_v_m in general I believe*/
			    grad_v[ebNE*nQuadraturePoints_elementBoundary*nDOF_trial_element*nSpace + 
				   k*nDOF_trial_element*nSpace+
                                   j*nSpace+
                                   colind_mw[m]];

			  /*mm potential*/
			  diffusiveVelocityComponent_I_Jacobian2_mw -=
			    a_mm[ebNE*nQuadraturePoints_elementBoundary*nnz_mw + 
				 k*nnz_mw + 
				 m]
			    * /*should be grad_v_w in general I believe*/
			    grad_v[ebNE*nQuadraturePoints_elementBoundary*nDOF_trial_element*nSpace + 
				   k*nDOF_trial_element*nSpace+
                                   j*nSpace+
                                   colind_mw[m]];
			  /*identical for now*/
			  diffusiveVelocityComponent_I_Jacobian2_mm -=
			    a_mm[ebNE*nQuadraturePoints_elementBoundary*nnz_mw + 
				 k*nnz_mw + 
				 m]
			    * /*should be grad_v_m in general I believe*/
			    grad_v[ebNE*nQuadraturePoints_elementBoundary*nDOF_trial_element*nSpace + 
				   k*nDOF_trial_element*nSpace+
                                   j*nSpace+
                                   colind_mw[m]];
			    
			}/*J loop*/
		      Jacobian_w += 
			diffusiveVelocityComponent_I_Jacobian_w 
			*/*should be v_w*/
			v[ebNE*nQuadraturePoints_elementBoundary*nDOF_trial_element +
			  k*nDOF_trial_element +
			  j]
			*
			n[ebNE*nQuadraturePoints_elementBoundary*nSpace + 
			  k*nSpace + 
			  I];
		      Jacobian_w += 
			diffusiveVelocityComponent_I_Jacobian2_ww 
			*
			dphi_w_w[j_global]
			*
			n[ebNE*nQuadraturePoints_elementBoundary*nSpace + 
			  k*nSpace + 
			  I];
		      Jacobian_w += 
			diffusiveVelocityComponent_I_Jacobian2_mw 
			*
			dphi_m_w[j_global]
			*
			n[ebNE*nQuadraturePoints_elementBoundary*nSpace + 
			  k*nSpace + 
			  I];

		      Jacobian_m += 
			diffusiveVelocityComponent_I_Jacobian_m 
			*/*should be v_m*/
			v[ebNE*nQuadraturePoints_elementBoundary*nDOF_trial_element +
			  k*nDOF_trial_element +
			  j]
			*
			n[ebNE*nQuadraturePoints_elementBoundary*nSpace + 
			  k*nSpace + 
			  I];
		      Jacobian_m += 
			diffusiveVelocityComponent_I_Jacobian2_wm 
			*
			dphi_w_m[j_global]
			*
			n[ebNE*nQuadraturePoints_elementBoundary*nSpace + 
			  k*nSpace + 
			  I];
		      Jacobian_m += 
			diffusiveVelocityComponent_I_Jacobian2_mm 
			*
			dphi_m_m[j_global]
			*
			n[ebNE*nQuadraturePoints_elementBoundary*nSpace + 
			  k*nSpace + 
			  I];
		    }/*I loop */
		  /*only diagonal gets penalty term*/
		  Jacobian_m += 
		    penalty_m[ebNE*nQuadraturePoints_elementBoundary+k]
		    * /*should be v_w*/
		    v[ebNE*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                      k*nDOF_trial_element+
                      j];
		  fluxJacobian_mw[ebNE*nQuadraturePoints_elementBoundary*nDOF_trial_element + 
				  k*nDOF_trial_element +
				  j] += 
		    Jacobian_w;
		  fluxJacobian_mm[ebNE*nQuadraturePoints_elementBoundary*nDOF_trial_element + 
				  k*nDOF_trial_element +
				  j] += 
		    Jacobian_m;
		}/* j local dof loop*/
	    }/*u_w dof boundary loop*/
	  
	}/*k*/
    }/*ebNE*/

}

void calculateGlobalExteriorNumericalFluxDarcyFC(proteus_index_t nExteriorElementBoundaries_global,
						 int nQuadraturePoints_elementBoundary,
						 int nSpace,
						 const proteus_index_t* exteriorElementBoundaries,
						 const proteus_index_t* elementBoundaryElements,
						 const int* elementBoundaryLocalElementBoundaries,
						 const int* isDOFBoundary_uw,/*1 set bc for s_w*/
						 const int* isDOFBoundary_un,/*1 set bc for psi_w, 
									       2 set bc for psi_n*/
						 int fluxBoundaryFlag_uw, /*0 no flow, 1 outflow*/
						 int fluxBoundaryFlag_un,
						 const double* n,
						 const double* bc_a_ww,      
						 const double* bc_a_nn,      
						 const double* bc_grad_phi_w,
						 const double* bc_grad_phi_n,
						 const double* bc_s_w,        
						 const double* bc_psi_w,       
						 const double* bc_psi_n,
						 const double* a_ww,         /*lambda_w K_s*/
						 const double* a_nn,         /*lambda_n K_s*/
						 const double* grad_phi_w,   /*psi_w - rho_w g . x*/
						 const double* grad_phi_n,   /*psi_c + psi_w - rho_n g . x*/
						 const double* s_w,           /*s_w*/
						 const double* psi_w,           /*psi_w*/
						 const double* psi_n,
						 const double* penalty_w,    
						 const double* penalty_n,
						 double * diffusiveFlux_ww,
						 double * diffusiveFlux_nn)
{
  proteus_index_t ebN; int ebNE, I, J, k, nSpace2=nSpace*nSpace;
  double diffusiveFlux_I=0.0,penaltyFlux = 0.0,potential_gradient_w=0.0,potential_gradient_n=0.0;

  for (ebNE = 0; ebNE < nExteriorElementBoundaries_global; ebNE++)
    {
      ebN = exteriorElementBoundaries[ebNE];
      for (k = 0; k < nQuadraturePoints_elementBoundary; k++)
	{
	  /*compute diffusive flux for first (w) equation (aq. mass
	    balance for part of boundary where u_0 (i.e., S_w) is
	    specified*/
	  diffusiveFlux_ww[ebNE*nQuadraturePoints_elementBoundary+k] = 0.0;
	  diffusiveFlux_nn[ebNE*nQuadraturePoints_elementBoundary+k] = 0.0;
	  /*allow outflow where potential gradient is out even if not Dirichlet, 
	   do not include K_s in calculation for now*/
	  potential_gradient_w = 0.0; potential_gradient_n = 0.0; 
	  for (I=0; I < nSpace; I++)
	    {
	      potential_gradient_w += grad_phi_w[ebNE*nQuadraturePoints_elementBoundary*nSpace +
						 k*nSpace + I]
		*
		n[ebNE*nQuadraturePoints_elementBoundary*nSpace +
		  k*nSpace+
		  I];
	      potential_gradient_n += grad_phi_n[ebNE*nQuadraturePoints_elementBoundary*nSpace +
						 k*nSpace + I]
		*
		n[ebNE*nQuadraturePoints_elementBoundary*nSpace +
		  k*nSpace+
		  I];
	    }
	  /*only allow s_w setting here?*/
	  if (isDOFBoundary_uw[ebNE*nQuadraturePoints_elementBoundary+k] == 1 || (potential_gradient_w > 0.0 && fluxBoundaryFlag_uw == 1))
	    {
	      /*integration by parts term for diffusive flux wrt phi_w = psi_w - rho_w g . x*/
	      for (I = 0; I < nSpace; I++)
		{
		  diffusiveFlux_I = 0.0;
		  for (J = 0; J < nSpace; J++)
		    {
		      diffusiveFlux_I -= 
			a_ww[ebNE*nQuadraturePoints_elementBoundary*nSpace2 + 
			     k*nSpace2 + 
			     I*nSpace + 
			     J]
			*
			grad_phi_w[ebNE*nQuadraturePoints_elementBoundary*nSpace + 
				   k*nSpace + 
				   J];
		    }
		  diffusiveFlux_ww[ebNE*nQuadraturePoints_elementBoundary+k] += 
		    diffusiveFlux_I
		    *
		    n[ebNE*nQuadraturePoints_elementBoundary*nSpace +
		      k*nSpace+
		      I];
		}/*I, a_wm grad phi_m term */
	      if (isDOFBoundary_uw[ebNE*nQuadraturePoints_elementBoundary+k] == 1)
		{
		  /*boundary penalty term*/
		  penaltyFlux = 
		    penalty_w[ebNE*nQuadraturePoints_elementBoundary + k]
		    *
		    (s_w[ebNE*nQuadraturePoints_elementBoundary + k]
		     -
		     bc_s_w[ebNE*nQuadraturePoints_elementBoundary + k]);
		  diffusiveFlux_ww[ebNE*nQuadraturePoints_elementBoundary +k] += 
		    penaltyFlux;
		}
	    }/*s_w boundary*/
	  /*1 set psi_w, 2 set psi_n*/
	  if(isDOFBoundary_un[ebNE*nQuadraturePoints_elementBoundary+k] >= 1 || (potential_gradient_n > 0.0 && fluxBoundaryFlag_un == 1))
	    {
	      /*integration by parts term for diffusive flux wrt phi_n = psi_c + psi_w - rho_n g . x*/
	      for (I = 0; I < nSpace; I++)
		{
		  diffusiveFlux_I = 0.0;
		  for (J = 0; J < nSpace; J++)
		    {
		      diffusiveFlux_I -= 
			a_nn[ebNE*nQuadraturePoints_elementBoundary*nSpace2 + 
			     k*nSpace2 +
			     I*nSpace + 
			     J]
			*
			grad_phi_n[ebNE*nQuadraturePoints_elementBoundary*nSpace + 
				   k*nSpace +
				   J];
		    }/*J*/
		  diffusiveFlux_nn[ebNE*nQuadraturePoints_elementBoundary + k] += 
		    diffusiveFlux_I
		    *
		    n[ebNE*nQuadraturePoints_elementBoundary*nSpace + 
		      k*nSpace + 
		      I];
		}/*I, a_mw grad phi_n term */
	      /*boundary penalty term*/
	      penaltyFlux = 0.0;
	      if (isDOFBoundary_un[ebNE*nQuadraturePoints_elementBoundary+k] == 2)
		{
		  penaltyFlux = 
		    penalty_n[ebNE*nQuadraturePoints_elementBoundary + k]
		    *
		    (psi_n[ebNE*nQuadraturePoints_elementBoundary + k]
		     -
		     bc_psi_n[ebNE*nQuadraturePoints_elementBoundary + k]);
		}
	      else if (isDOFBoundary_un[ebNE*nQuadraturePoints_elementBoundary+k] == 1)
		{
		  penaltyFlux = 
		    penalty_n[ebNE*nQuadraturePoints_elementBoundary + k]
		    *
		    (psi_w[ebNE*nQuadraturePoints_elementBoundary + k]
		     -
		     bc_psi_w[ebNE*nQuadraturePoints_elementBoundary + k]);

		}
	      diffusiveFlux_nn[ebNE*nQuadraturePoints_elementBoundary +k] += 
		penaltyFlux;
	    }/*um boundary*/
	}/*k*/
    }/*ebNE*/
}

void calculateGlobalExteriorNumericalFluxDarcyFC_sd(proteus_index_t nExteriorElementBoundaries_global,
						    int nQuadraturePoints_elementBoundary,
						    int nSpace,
						    proteus_csr_index_t* rowptr_ww,
						    proteus_csr_index_t* colind_ww,
						    proteus_csr_index_t* rowptr_nn,
						    proteus_csr_index_t* colind_nn,
						    const proteus_index_t* exteriorElementBoundaries,
						    const proteus_index_t* elementBoundaryElements,
						    const int* elementBoundaryLocalElementBoundaries,
						    const int* isDOFBoundary_uw,/*1 set bc for s_w*/
						    const int* isDOFBoundary_un,/*1 set bc for psi_w, 
										  2 set bc for psi_n*/
						    int fluxBoundaryFlag_uw, /*0 no flow, 1 outflow*/
						    int fluxBoundaryFlag_un,
						    const double* n,
						    const double* bc_a_ww,      
						    const double* bc_a_nn,      
						    const double* bc_grad_phi_w,
						    const double* bc_grad_phi_n,
						    const double* bc_s_w,        
						    const double* bc_psi_w,       
						    const double* bc_psi_n,
						    const double* a_ww,         /*lambda_w K_s*/
						    const double* a_nn,         /*lambda_n K_s*/
						    const double* grad_phi_w,   /*psi_w */
						    const double* grad_phi_n,   /*psi_c + psi_w */
						    const double* s_w,           /*s_w*/
						    const double* psi_w,           /*psi_w*/
						    const double* psi_n,
						    const double* penalty_w,    
						    const double* penalty_n,
						    double * diffusiveFlux_ww,
						    double * diffusiveFlux_nn)
{
  proteus_index_t ebN; proteus_csr_index_t m; proteus_csr_index_t nnz_nn=rowptr_nn[nSpace]; proteus_csr_index_t nnz_ww=rowptr_ww[nSpace]; int ebNE, I, k;
  double diffusiveFlux_I=0.0,penaltyFlux = 0.0,potential_gradient_w=0.0,potential_gradient_n=0.0;
  for (ebNE = 0; ebNE < nExteriorElementBoundaries_global; ebNE++)
    {
      ebN = exteriorElementBoundaries[ebNE];
      for (k = 0; k < nQuadraturePoints_elementBoundary; k++)
	{
	  /*compute diffusive flux for first (w) equation (aq. mass
	    balance for part of boundary where u_0 (i.e., S_w) is
	    specified*/
	  diffusiveFlux_ww[ebNE*nQuadraturePoints_elementBoundary+k] = 0.0;
	  diffusiveFlux_nn[ebNE*nQuadraturePoints_elementBoundary+k] = 0.0;
	  /*allow outflow where potential gradient is out even if not Dirichlet, 
	   do not include K_s in calculation for now*/
	  potential_gradient_w = 0.0; potential_gradient_n = 0.0; 
	  for (I=0; I < nSpace; I++)
	    {
	      potential_gradient_w += grad_phi_w[ebNE*nQuadraturePoints_elementBoundary*nSpace +
						 k*nSpace + I]
		*
		n[ebNE*nQuadraturePoints_elementBoundary*nSpace +
		  k*nSpace+
		  I];
	      potential_gradient_n += grad_phi_n[ebNE*nQuadraturePoints_elementBoundary*nSpace +
						 k*nSpace + I]
		*
		n[ebNE*nQuadraturePoints_elementBoundary*nSpace +
		  k*nSpace+
		  I];
	    }

	  /*only allow s_w setting here?*/
	  if (isDOFBoundary_uw[ebNE*nQuadraturePoints_elementBoundary+k] == 1 || (potential_gradient_w > 0.0 && fluxBoundaryFlag_uw==1))
	    {
	      /*integration by parts term for diffusive flux wrt phi_w = psi_w */
	      for (I = 0; I < nSpace; I++)
		{
		  diffusiveFlux_I = 0.0;
		  for(m=rowptr_ww[I];m<rowptr_ww[I+1];m++)
		    {
		      diffusiveFlux_I -= 
			a_ww[ebNE*nQuadraturePoints_elementBoundary*nnz_ww + 
			     k*nnz_ww + 
			     m]
			*
			grad_phi_w[ebNE*nQuadraturePoints_elementBoundary*nSpace + 
				   k*nSpace + 
				   colind_ww[m]];
		    }
		  diffusiveFlux_ww[ebNE*nQuadraturePoints_elementBoundary+k] += 
		    diffusiveFlux_I
		    *
		    n[ebNE*nQuadraturePoints_elementBoundary*nSpace +
		      k*nSpace+
		      I];
		}/*I, a_wm grad phi_m term */
	      /*boundary penalty term*/
	      if (isDOFBoundary_uw[ebNE*nQuadraturePoints_elementBoundary+k] == 1)
		{
		  penaltyFlux = 
		    penalty_w[ebNE*nQuadraturePoints_elementBoundary + k]
		    *
		    (s_w[ebNE*nQuadraturePoints_elementBoundary + k]
		     -
		     bc_s_w[ebNE*nQuadraturePoints_elementBoundary + k]);
		  diffusiveFlux_ww[ebNE*nQuadraturePoints_elementBoundary +k] += 
		    penaltyFlux;
		}
	    }/*s_w boundary*/
	  /*1 set psi_w, 2 set psi_n*/
	  if(isDOFBoundary_un[ebNE*nQuadraturePoints_elementBoundary+k] >= 1 || (potential_gradient_n > 0.0 && fluxBoundaryFlag_un==1))
	    {
	      /*integration by parts term for diffusive flux wrt phi_n = psi_c + psi_w */
	      for (I = 0; I < nSpace; I++)
		{
		  diffusiveFlux_I = 0.0;
		  for(m=rowptr_nn[I];m<rowptr_nn[I+1];m++)
		    {
		      diffusiveFlux_I -= 
			a_nn[ebNE*nQuadraturePoints_elementBoundary*nnz_nn + 
			     k*nnz_nn +
			     m]
			*
			grad_phi_n[ebNE*nQuadraturePoints_elementBoundary*nSpace + 
				   k*nSpace +
				   colind_nn[m]];
		    }/*J*/
		  diffusiveFlux_nn[ebNE*nQuadraturePoints_elementBoundary + k] += 
		    diffusiveFlux_I
		    *
		    n[ebNE*nQuadraturePoints_elementBoundary*nSpace + 
		      k*nSpace + 
		      I];
		}/*I, a_mw grad phi_n term */
	      /*boundary penalty term*/
	      penaltyFlux = 0.0;
	      if (isDOFBoundary_un[ebNE*nQuadraturePoints_elementBoundary+k] == 2)
		{
		  penaltyFlux = 
		    penalty_n[ebNE*nQuadraturePoints_elementBoundary + k]
		    *
		    (psi_n[ebNE*nQuadraturePoints_elementBoundary + k]
		     -
		     bc_psi_n[ebNE*nQuadraturePoints_elementBoundary + k]);
		}
	      else if (isDOFBoundary_un[ebNE*nQuadraturePoints_elementBoundary+k] == 1)
		{
		  penaltyFlux = 
		    penalty_n[ebNE*nQuadraturePoints_elementBoundary + k]
		    *
		    (psi_w[ebNE*nQuadraturePoints_elementBoundary + k]
		     -
		     bc_psi_w[ebNE*nQuadraturePoints_elementBoundary + k]);

		}
	      diffusiveFlux_nn[ebNE*nQuadraturePoints_elementBoundary +k] += 
		penaltyFlux;
	    }/*um boundary*/
	}/*k*/
    }/*ebNE*/
}

void calculateGlobalExteriorNumericalFluxDarcyFC_diffusiveFluxJacobian(proteus_index_t nExteriorElementBoundaries_global,
								       int nQuadraturePoints_elementBoundary,
								       int nSpace,
								       int nDOF_trial_element,
								       const proteus_index_t* l2g, /*for now assumes both solution spaces are the same!*/
								       const proteus_index_t* exteriorElementBoundaries,
								       const proteus_index_t* elementBoundaryElements,
								       const int* elementBoundaryLocalElementBoundaries,
								       const int* isDOFBoundary_uw,/*1 set bc for s_w*/
								       const int* isDOFBoundary_un,/*1 set bc for psi_w, 
												     2 set bc for psi_n*/
								       int fluxBoundaryFlag_uw, /*0 no flow, 1 outflow*/
								       int fluxBoundaryFlag_un,
								       const double* n,
								       const double* a_ww,         /*lambda_w K_s*/
								       const double* da_ww_dw,         /* a' wrt S_w*/
								       const double* da_ww_dn,         /* a' wrt psi_w*/
								       const double* a_nn,         /*lambda_t K_s*/
								       const double* da_nn_dw,         /* a' wrt S_w*/
								       const double* da_nn_dn,         /* a' wrt psi_w*/
								       const double* grad_phi_w,   /*psi_w - rho_w g . x*/
								       const double* grad_phi_n,   /*psi_n + psi_w - rho_n g . x*/
								       const double* dphi_w_w,     /*\pd{phi_w}{S_w} = 0 */     
								       const double* dphi_w_n,     /*\pd{phi_w}{psi_w}= 1 - drho_w g.x */
								       const double* dphi_n_w,     /*\pd{phi_n}{S_w} = \od{psi_c}{S_w}  */
								       const double* dphi_n_n,     /*\pd{phi_n}{psi_w} = 1 - drho_n/dpsi_w g . x */
								       const double* s_w,           /*S_w*/
								       const double* psi_w,           /*psi_w*/
								       const double* psi_n,           
								       const double* dpsi_n_dsw, 
								       const double* dpsi_n_dpsiw,
								       const double* v,            /*trial functions, assumed in same space*/
								       const double* grad_v,       /*trial function gradients, assumed in same space*/
								       const double* penalty_w,    
								       const double* penalty_n,
								       double * fluxJacobian_ww,
								       double * fluxJacobian_wn,
								       double * fluxJacobian_nw,
								       double * fluxJacobian_nn)
{
  proteus_index_t ebN, eN_global, j_global; int ebNE, j, I, J, k, nSpace2=nSpace*nSpace;
  double Jacobian_w,Jacobian_n,
    diffusiveVelocityComponent_I_Jacobian_w,
    diffusiveVelocityComponent_I_Jacobian_n,  
    diffusiveVelocityComponent_I_Jacobian2_wn,
    diffusiveVelocityComponent_I_Jacobian2_ww,
    diffusiveVelocityComponent_I_Jacobian2_nw,
    diffusiveVelocityComponent_I_Jacobian2_nn;
  double potential_gradient_w=0.0,potential_gradient_n=0.0;

  for (ebNE = 0; ebNE < nExteriorElementBoundaries_global; ebNE++)
    {
      ebN = exteriorElementBoundaries[ebNE];
      eN_global = elementBoundaryElements[ebN*2 + 0];
      for (k = 0; k < nQuadraturePoints_elementBoundary; k++)
	{
	  /*compute derivative of diffusive flux for first (w) equation (aq. mass
	    balance for part of boundary where u_0 (i.e., S_w) is
	    specified
	  */
	  /*allow outflow where potential gradient is out even if not Dirichlet, 
	   do not include K_s in calculation for now*/
	  potential_gradient_w = 0.0; potential_gradient_n = 0.0; 
	  for (I=0; I < nSpace; I++)
	    {
	      potential_gradient_w += grad_phi_w[ebNE*nQuadraturePoints_elementBoundary*nSpace +
						 k*nSpace + I]
		*
		n[ebNE*nQuadraturePoints_elementBoundary*nSpace +
		  k*nSpace+
		  I];
	      potential_gradient_n += grad_phi_n[ebNE*nQuadraturePoints_elementBoundary*nSpace +
						 k*nSpace + I]
		*
		n[ebNE*nQuadraturePoints_elementBoundary*nSpace +
		  k*nSpace+
		  I];
	    }
	  /*only allow setting s_w for this equation*/	  
	  if (isDOFBoundary_uw[ebNE*nQuadraturePoints_elementBoundary + k] == 1 || (potential_gradient_w > 0.0 && fluxBoundaryFlag_uw==1))
	    {
	      /*NOTE! assuming u_w, u_n in the same space and nodal interpolant for potential*/
	      
	      for (j = 0; j < nDOF_trial_element; j++)
		{
		  Jacobian_w = 0.; /*derivative wrt u_w = S_w */
		  Jacobian_n = 0.; /*derivative wrt u_n = psi_w */
		  j_global = l2g[eN_global*nDOF_trial_element + j];/*assuming same space for both*/ 
		  for (I = 0; I < nSpace; I++)
		    {
		      diffusiveVelocityComponent_I_Jacobian_w = 0.0; 
		      diffusiveVelocityComponent_I_Jacobian_n = 0.0; 
		      diffusiveVelocityComponent_I_Jacobian2_wn = 0.0;
		      diffusiveVelocityComponent_I_Jacobian2_ww = 0.0;
		      for (J = 0; J < nSpace; J++)
			{
			  /*only a_ww potential here*/
			  diffusiveVelocityComponent_I_Jacobian_w -= 
			    da_ww_dw[ebNE*nQuadraturePoints_elementBoundary*nSpace2 + 
				     k*nSpace2 + 
				     I*nSpace  + 
				     J]
			    *
			    grad_phi_w[ebNE*nQuadraturePoints_elementBoundary*nSpace + 
				       k*nSpace + 
				       J];
			  diffusiveVelocityComponent_I_Jacobian_n -= 
			    da_ww_dn[ebNE*nQuadraturePoints_elementBoundary*nSpace2 + 
				     k*nSpace2 + 
				     I*nSpace  + 
				     J]
			    *
			    grad_phi_w[ebNE*nQuadraturePoints_elementBoundary*nSpace + 
				       k*nSpace + 
				       J];

			  diffusiveVelocityComponent_I_Jacobian2_ww -=
			    a_ww[ebNE*nQuadraturePoints_elementBoundary*nSpace2 + 
				 k*nSpace2 + 
				 I*nSpace  + 
				 J]
			    * /*should be grad_v_w in general I believe*/
			    grad_v[ebNE*nQuadraturePoints_elementBoundary*nDOF_trial_element*nSpace + 
				   k*nDOF_trial_element*nSpace+
                                   j*nSpace+
                                   J];
			  /*identical for now*/
			  diffusiveVelocityComponent_I_Jacobian2_wn -=
			    a_ww[ebNE*nQuadraturePoints_elementBoundary*nSpace2 + 
				 k*nSpace2 + 
				 I*nSpace  + 
				 J]
			    * /*should be grad_v_w in general I believe*/
			    grad_v[ebNE*nQuadraturePoints_elementBoundary*nDOF_trial_element*nSpace + 
				   k*nDOF_trial_element*nSpace+
                                   j*nSpace+
                                   J];
			    
			}/*J loop*/
		      Jacobian_w += 
			diffusiveVelocityComponent_I_Jacobian_w 
			*/*should be v_w*/
			v[ebNE*nQuadraturePoints_elementBoundary*nDOF_trial_element +
			  k*nDOF_trial_element +
			  j]
			*
			n[ebNE*nQuadraturePoints_elementBoundary*nSpace + 
			  k*nSpace + 
			  I];
		      Jacobian_w += 
			diffusiveVelocityComponent_I_Jacobian2_ww 
			*
			dphi_w_w[j_global]
			*
			n[ebNE*nQuadraturePoints_elementBoundary*nSpace + 
			  k*nSpace + 
			  I];
		      Jacobian_n += 
			diffusiveVelocityComponent_I_Jacobian_n 
			*/*should be v_n*/
			v[ebNE*nQuadraturePoints_elementBoundary*nDOF_trial_element +
			  k*nDOF_trial_element +
			  j]
			*
			n[ebNE*nQuadraturePoints_elementBoundary*nSpace + 
			  k*nSpace + 
			  I];
		      Jacobian_n += 
			diffusiveVelocityComponent_I_Jacobian2_wn 
			*
			dphi_w_n[j_global]
			*
			n[ebNE*nQuadraturePoints_elementBoundary*nSpace + 
			  k*nSpace + 
			  I];
		    }/*I loop */
		  /*only diagonal gets penalty term*/
		  if (isDOFBoundary_uw[ebNE*nQuadraturePoints_elementBoundary+k] == 1)
		    {
		      Jacobian_w += 
			penalty_w[ebNE*nQuadraturePoints_elementBoundary+k]
			* /*should be v_w*/
			v[ebNE*nQuadraturePoints_elementBoundary*nDOF_trial_element+
			  k*nDOF_trial_element+
			  j];
		    }
		  fluxJacobian_ww[ebNE*nQuadraturePoints_elementBoundary*nDOF_trial_element + 
				  k*nDOF_trial_element +
				  j] += 
		    Jacobian_w;
		  fluxJacobian_wn[ebNE*nQuadraturePoints_elementBoundary*nDOF_trial_element + 
				  k*nDOF_trial_element +
				  j] += 
		    Jacobian_n;
		}/* j local dof loop*/
	    }/*u_w dof boundary loop*/
	  /*setting psi_w = 1, or psi_n = 2*/
	  if (isDOFBoundary_un[ebNE*nQuadraturePoints_elementBoundary + k] >= 1 || (potential_gradient_n > 0.0 && fluxBoundaryFlag_un==1))
	    {
	      /*NOTE! assuming u_w, u_m in the same space and nodal interpolant for potential*/
	      
	      for (j = 0; j < nDOF_trial_element; j++)
		{
		  Jacobian_w = 0.; /*derivative wrt u_w = S_w */
		  Jacobian_n = 0.; /*derivative wrt u_n = psi_w */
		  j_global = l2g[eN_global*nDOF_trial_element + j];/*assuming same space for both*/ 
		  for (I = 0; I < nSpace; I++)
		    {
		      diffusiveVelocityComponent_I_Jacobian_w = 0.0;
		      diffusiveVelocityComponent_I_Jacobian_n = 0.0;  
		      diffusiveVelocityComponent_I_Jacobian2_nw = 0.0;
		      diffusiveVelocityComponent_I_Jacobian2_nn = 0.0;
		      for (J = 0; J < nSpace; J++)
			{
			  /*nn potential*/
			  diffusiveVelocityComponent_I_Jacobian_w -= 
			    da_nn_dw[ebNE*nQuadraturePoints_elementBoundary*nSpace2 + 
				     k*nSpace2 + 
				     I*nSpace  + 
				     J]
			    *
			    grad_phi_n[ebNE*nQuadraturePoints_elementBoundary*nSpace + 
				       k*nSpace + 
				       J];
			  diffusiveVelocityComponent_I_Jacobian_n -= 
			    da_nn_dn[ebNE*nQuadraturePoints_elementBoundary*nSpace2 + 
				     k*nSpace2 + 
				     I*nSpace  + 
				     J]
			    *
			    grad_phi_n[ebNE*nQuadraturePoints_elementBoundary*nSpace + 
				       k*nSpace + 
				       J];

			  /*nn potential*/
			  diffusiveVelocityComponent_I_Jacobian2_nw -=
			    a_nn[ebNE*nQuadraturePoints_elementBoundary*nSpace2 + 
				 k*nSpace2 + 
				 I*nSpace  + 
				 J]
			    * /*should be grad_v_w in general I believe*/
			    grad_v[ebNE*nQuadraturePoints_elementBoundary*nDOF_trial_element*nSpace + 
				   k*nDOF_trial_element*nSpace+
                                   j*nSpace+
                                   J];
			  /*identical for now*/
			  diffusiveVelocityComponent_I_Jacobian2_nn -=
			    a_nn[ebNE*nQuadraturePoints_elementBoundary*nSpace2 + 
				 k*nSpace2 + 
				 I*nSpace  + 
				 J]
			    * /*should be grad_v_m in general I believe*/
			    grad_v[ebNE*nQuadraturePoints_elementBoundary*nDOF_trial_element*nSpace + 
				   k*nDOF_trial_element*nSpace+
                                   j*nSpace+
                                   J];
			    
			}/*J loop*/
		      Jacobian_w += 
			diffusiveVelocityComponent_I_Jacobian_w 
			*/*should be v_w*/
			v[ebNE*nQuadraturePoints_elementBoundary*nDOF_trial_element +
			  k*nDOF_trial_element +
			  j]
			*
			n[ebNE*nQuadraturePoints_elementBoundary*nSpace + 
			  k*nSpace + 
			  I];
		      Jacobian_w += 
			diffusiveVelocityComponent_I_Jacobian2_nw 
			*
			dphi_n_w[j_global]
			*
			n[ebNE*nQuadraturePoints_elementBoundary*nSpace + 
			  k*nSpace + 
			  I];

		      Jacobian_n += 
			diffusiveVelocityComponent_I_Jacobian_n 
			*/*should be v_n*/
			v[ebNE*nQuadraturePoints_elementBoundary*nDOF_trial_element +
			  k*nDOF_trial_element +
			  j]
			*
			n[ebNE*nQuadraturePoints_elementBoundary*nSpace + 
			  k*nSpace + 
			  I];
		      Jacobian_n += 
			diffusiveVelocityComponent_I_Jacobian2_nn 
			*
			dphi_n_n[j_global]
			*
			n[ebNE*nQuadraturePoints_elementBoundary*nSpace + 
			  k*nSpace + 
			  I];
		    }/*I loop */
		 
		  if (isDOFBoundary_un[ebNE*nQuadraturePoints_elementBoundary + k] == 2)
		    {
		      /*dependency of psi_n on psi_w*/
		      Jacobian_n += 
			penalty_n[ebNE*nQuadraturePoints_elementBoundary+k]
			* /*should be v_n*/
			v[ebNE*nQuadraturePoints_elementBoundary*nDOF_trial_element+
			  k*nDOF_trial_element+
			  j]
			*
			dpsi_n_dpsiw[ebNE*nQuadraturePoints_elementBoundary + k];
		      /*dependency of psi_n on s_w */
		      Jacobian_w += 
			penalty_n[ebNE*nQuadraturePoints_elementBoundary+k]
			* /*should be v_w*/
			v[ebNE*nQuadraturePoints_elementBoundary*nDOF_trial_element+
			  k*nDOF_trial_element+
			  j]
			*
			dpsi_n_dsw[ebNE*nQuadraturePoints_elementBoundary + k];
		    }
		  else if (isDOFBoundary_un[ebNE*nQuadraturePoints_elementBoundary + k] == 1)
		    {
		      /*only diagonal gets penalty term*/
		      Jacobian_n += 
			penalty_n[ebNE*nQuadraturePoints_elementBoundary+k]
			* /*should be v_w*/
			v[ebNE*nQuadraturePoints_elementBoundary*nDOF_trial_element+
			  k*nDOF_trial_element+
			  j];

		    }
		  fluxJacobian_nw[ebNE*nQuadraturePoints_elementBoundary*nDOF_trial_element + 
				  k*nDOF_trial_element +
				  j] += 
		    Jacobian_w;
		  fluxJacobian_nn[ebNE*nQuadraturePoints_elementBoundary*nDOF_trial_element + 
				  k*nDOF_trial_element +
				  j] += 
		    Jacobian_n;
		}/* j local dof loop*/
	    }/*u_w dof boundary loop*/
	  
	}/*k*/
    }/*ebNE*/

}

void calculateGlobalExteriorNumericalFluxDarcyFC_diffusiveFluxJacobian_sd(proteus_index_t nExteriorElementBoundaries_global,
									  int nQuadraturePoints_elementBoundary,
									  int nSpace,
									  int nDOF_trial_element,
									  proteus_csr_index_t* rowptr_ww,
									  proteus_csr_index_t* colind_ww,
									  proteus_csr_index_t* rowptr_nn,
									  proteus_csr_index_t* colind_nn,
									  const proteus_index_t* l2g, /*for now assumes both solution spaces are the same!*/
									  const proteus_index_t* exteriorElementBoundaries,
									  const proteus_index_t* elementBoundaryElements,
									  const int* elementBoundaryLocalElementBoundaries,
									  const int* isDOFBoundary_uw,/*1 set bc for s_w*/
									  const int* isDOFBoundary_un,/*1 set bc for psi_w, 
													2 set bc for psi_n*/
									  int fluxBoundaryFlag_uw, /*0 no flow, 1 outflow*/
									  int fluxBoundaryFlag_un,
									  const double* n,
									  const double* a_ww,         /*lambda_w K_s*/
									  const double* da_ww_dw,         /* a' wrt S_w*/
									  const double* da_ww_dn,         /* a' wrt psi_w*/
									  const double* a_nn,         /*lambda_t K_s*/
									  const double* da_nn_dw,         /* a' wrt S_w*/
									  const double* da_nn_dn,         /* a' wrt psi_w*/
									  const double* grad_phi_w,   /*psi_w - rho_w g . x*/
									  const double* grad_phi_n,   /*psi_n + psi_w - rho_n g . x*/
									  const double* dphi_w_w,     /*\pd{phi_w}{S_w} = 0 */     
									  const double* dphi_w_n,     /*\pd{phi_w}{psi_w}= 1 - drho_w g.x */
									  const double* dphi_n_w,     /*\pd{phi_n}{S_w} = \od{psi_c}{S_w}  */
									  const double* dphi_n_n,     /*\pd{phi_n}{psi_w} = 1 - drho_n/dpsi_w g . x */
									  const double* s_w,           /*S_w*/
									  const double* psi_w,           /*psi_w*/
									  const double* psi_n,           
									  const double* dpsi_n_dsw, 
									  const double* dpsi_n_dpsiw,
									  const double* v,            /*trial functions, assumed in same space*/
									  const double* grad_v,       /*trial function gradients, assumed in same space*/
									  const double* penalty_w,    
									  const double* penalty_n,
									  double * fluxJacobian_ww,
									  double * fluxJacobian_wn,
									  double * fluxJacobian_nw,
									  double * fluxJacobian_nn)
{
  proteus_index_t ebN, eN_global, j_global; proteus_csr_index_t m; proteus_csr_index_t nnz_ww=rowptr_ww[nSpace]; proteus_csr_index_t nnz_nn=rowptr_nn[nSpace]; int ebNE, j, I, k;
  double Jacobian_w,Jacobian_n,
    diffusiveVelocityComponent_I_Jacobian_w,
    diffusiveVelocityComponent_I_Jacobian_n,  
    diffusiveVelocityComponent_I_Jacobian2_wn,
    diffusiveVelocityComponent_I_Jacobian2_ww,
    diffusiveVelocityComponent_I_Jacobian2_nw,
    diffusiveVelocityComponent_I_Jacobian2_nn;
  double potential_gradient_w=0.0,potential_gradient_n=0.0;
  for (ebNE = 0; ebNE < nExteriorElementBoundaries_global; ebNE++)
    {
      ebN = exteriorElementBoundaries[ebNE];
      eN_global = elementBoundaryElements[ebN*2 + 0];
      
      for (k = 0; k < nQuadraturePoints_elementBoundary; k++)
	{
	  /*compute derivative of diffusive flux for first (w) equation (aq. mass
	    balance for part of boundary where u_0 (i.e., S_w) is
	    specified
	  */
	  /*allow outflow where potential gradient is out even if not Dirichlet, 
	   do not include K_s in calculation for now*/
	  potential_gradient_w = 0.0; potential_gradient_n = 0.0; 
	  for (I=0; I < nSpace; I++)
	    {
	      potential_gradient_w += grad_phi_w[ebNE*nQuadraturePoints_elementBoundary*nSpace +
						 k*nSpace + I]
		*
		n[ebNE*nQuadraturePoints_elementBoundary*nSpace +
		  k*nSpace+
		  I];
	      potential_gradient_n += grad_phi_n[ebNE*nQuadraturePoints_elementBoundary*nSpace +
						 k*nSpace + I]
		*
		n[ebNE*nQuadraturePoints_elementBoundary*nSpace +
		  k*nSpace+
		  I];
	    }

	  /*only allow setting s_w for this equation*/	  
	  if (isDOFBoundary_uw[ebNE*nQuadraturePoints_elementBoundary + k] == 1 || (potential_gradient_w > 0.0 && fluxBoundaryFlag_uw==1))
	    {
	      /*NOTE! assuming u_w, u_n in the same space and nodal interpolant for potential*/
	      
	      for (j = 0; j < nDOF_trial_element; j++)
		{
		  Jacobian_w = 0.; /*derivative wrt u_w = S_w */
		  Jacobian_n = 0.; /*derivative wrt u_n = psi_w */
		  j_global = l2g[eN_global*nDOF_trial_element + j];/*assuming same space for both*/ 
		  for (I = 0; I < nSpace; I++)
		    {
		      diffusiveVelocityComponent_I_Jacobian_w = 0.0; 
		      diffusiveVelocityComponent_I_Jacobian_n = 0.0; 
		      diffusiveVelocityComponent_I_Jacobian2_wn = 0.0;
		      diffusiveVelocityComponent_I_Jacobian2_ww = 0.0;
		      for (m=rowptr_ww[I];m<rowptr_ww[I+1];m++)
			{
			  /*only a_ww potential here*/
			  diffusiveVelocityComponent_I_Jacobian_w -= 
			    da_ww_dw[ebNE*nQuadraturePoints_elementBoundary*nnz_ww + 
				     k*nnz_ww + 
				     m]
			    *
			    grad_phi_w[ebNE*nQuadraturePoints_elementBoundary*nSpace + 
				       k*nSpace + 
				       colind_ww[m]];
			  diffusiveVelocityComponent_I_Jacobian_n -= 
			    da_ww_dn[ebNE*nQuadraturePoints_elementBoundary*nnz_ww + 
				     k*nnz_ww + 
				     m]
			    *
			    grad_phi_w[ebNE*nQuadraturePoints_elementBoundary*nSpace + 
				       k*nSpace + 
				       colind_ww[m]];

			  diffusiveVelocityComponent_I_Jacobian2_ww -=
			    a_ww[ebNE*nQuadraturePoints_elementBoundary*nnz_ww + 
				 k*nnz_ww + 
				 m]
			    * /*should be grad_v_w in general I believe*/
			    grad_v[ebNE*nQuadraturePoints_elementBoundary*nDOF_trial_element*nSpace + 
				   k*nDOF_trial_element*nSpace+
                                   j*nSpace+
                                   colind_ww[m]];
			  /*identical for now*/
			  diffusiveVelocityComponent_I_Jacobian2_wn -=
			    a_ww[ebNE*nQuadraturePoints_elementBoundary*nnz_ww + 
				 k*nnz_ww + 
				 m]
			    * /*should be grad_v_w in general I believe*/
			    grad_v[ebNE*nQuadraturePoints_elementBoundary*nDOF_trial_element*nSpace + 
				   k*nDOF_trial_element*nSpace+
                                   j*nSpace+
                                   colind_ww[m]];
			    
			}/*J loop*/
		      Jacobian_w += 
			diffusiveVelocityComponent_I_Jacobian_w 
			*/*should be v_w*/
			v[ebNE*nQuadraturePoints_elementBoundary*nDOF_trial_element +
			  k*nDOF_trial_element +
			  j]
			*
			n[ebNE*nQuadraturePoints_elementBoundary*nSpace + 
			  k*nSpace + 
			  I];
		      Jacobian_w += 
			diffusiveVelocityComponent_I_Jacobian2_ww 
			*
			dphi_w_w[j_global]
			*
			n[ebNE*nQuadraturePoints_elementBoundary*nSpace + 
			  k*nSpace + 
			  I];
		      Jacobian_n += 
			diffusiveVelocityComponent_I_Jacobian_n 
			*/*should be v_n*/
			v[ebNE*nQuadraturePoints_elementBoundary*nDOF_trial_element +
			  k*nDOF_trial_element +
			  j]
			*
			n[ebNE*nQuadraturePoints_elementBoundary*nSpace + 
			  k*nSpace + 
			  I];
		      Jacobian_n += 
			diffusiveVelocityComponent_I_Jacobian2_wn 
			*
			dphi_w_n[j_global]
			*
			n[ebNE*nQuadraturePoints_elementBoundary*nSpace + 
			  k*nSpace + 
			  I];
		    }/*I loop */
		  /*only diagonal gets penalty term*/
		  if (isDOFBoundary_uw[ebNE*nQuadraturePoints_elementBoundary+k] == 1)
		    {
		      Jacobian_w += 
			penalty_w[ebNE*nQuadraturePoints_elementBoundary+k]
			* /*should be v_w*/
			v[ebNE*nQuadraturePoints_elementBoundary*nDOF_trial_element+
			  k*nDOF_trial_element+
			  j];
		    }
		  fluxJacobian_ww[ebNE*nQuadraturePoints_elementBoundary*nDOF_trial_element + 
				  k*nDOF_trial_element +
				  j] += 
		    Jacobian_w;
		  fluxJacobian_wn[ebNE*nQuadraturePoints_elementBoundary*nDOF_trial_element + 
				  k*nDOF_trial_element +
				  j] += 
		    Jacobian_n;
		}/* j local dof loop*/
	    }/*u_w dof boundary loop*/
	  /*setting psi_w = 1, or psi_n = 2*/
	  if (isDOFBoundary_un[ebNE*nQuadraturePoints_elementBoundary + k] >= 1 || (potential_gradient_n > 0.0 && fluxBoundaryFlag_un==1))
	    {
	      /*NOTE! assuming u_w, u_m in the same space and nodal interpolant for potential*/
	      
	      for (j = 0; j < nDOF_trial_element; j++)
		{
		  Jacobian_w = 0.; /*derivative wrt u_w = S_w */
		  Jacobian_n = 0.; /*derivative wrt u_n = psi_w */
		  j_global = l2g[eN_global*nDOF_trial_element + j];/*assuming same space for both*/ 
		  for (I = 0; I < nSpace; I++)
		    {
		      diffusiveVelocityComponent_I_Jacobian_w = 0.0;
		      diffusiveVelocityComponent_I_Jacobian_n = 0.0;  
		      diffusiveVelocityComponent_I_Jacobian2_nw = 0.0;
		      diffusiveVelocityComponent_I_Jacobian2_nn = 0.0;
		      for (m=rowptr_nn[I];m<rowptr_nn[I+1];m++)
			{
			  /*nn potential*/
			  diffusiveVelocityComponent_I_Jacobian_w -= 
			    da_nn_dw[ebNE*nQuadraturePoints_elementBoundary*nnz_nn + 
				     k*nnz_nn + 
				     m]
			    *
			    grad_phi_n[ebNE*nQuadraturePoints_elementBoundary*nSpace + 
				       k*nSpace + 
				       colind_nn[m]];
			  diffusiveVelocityComponent_I_Jacobian_n -= 
			    da_nn_dn[ebNE*nQuadraturePoints_elementBoundary*nnz_nn + 
				     k*nnz_nn + 
				     m]
			    *
			    grad_phi_n[ebNE*nQuadraturePoints_elementBoundary*nSpace + 
				       k*nSpace + 
				       colind_nn[m]];

			  /*nn potential*/
			  diffusiveVelocityComponent_I_Jacobian2_nw -=
			    a_nn[ebNE*nQuadraturePoints_elementBoundary*nnz_nn + 
				 k*nnz_nn + 
				 m]
			    * /*should be grad_v_w in general I believe*/
			    grad_v[ebNE*nQuadraturePoints_elementBoundary*nDOF_trial_element*nSpace + 
				   k*nDOF_trial_element*nSpace+
                                   j*nSpace+
                                   colind_nn[m]];
			  /*identical for now*/
			  diffusiveVelocityComponent_I_Jacobian2_nn -=
			    a_nn[ebNE*nQuadraturePoints_elementBoundary*nnz_nn + 
				 k*nnz_nn + 
				 m]
			    * /*should be grad_v_m in general I believe*/
			    grad_v[ebNE*nQuadraturePoints_elementBoundary*nDOF_trial_element*nSpace + 
				   k*nDOF_trial_element*nSpace+
                                   j*nSpace+
                                   colind_nn[m]];
			    
			}/*J loop*/
		      Jacobian_w += 
			diffusiveVelocityComponent_I_Jacobian_w 
			*/*should be v_w*/
			v[ebNE*nQuadraturePoints_elementBoundary*nDOF_trial_element +
			  k*nDOF_trial_element +
			  j]
			*
			n[ebNE*nQuadraturePoints_elementBoundary*nSpace + 
			  k*nSpace + 
			  I];
		      Jacobian_w += 
			diffusiveVelocityComponent_I_Jacobian2_nw 
			*
			dphi_n_w[j_global]
			*
			n[ebNE*nQuadraturePoints_elementBoundary*nSpace + 
			  k*nSpace + 
			  I];

		      Jacobian_n += 
			diffusiveVelocityComponent_I_Jacobian_n 
			*/*should be v_n*/
			v[ebNE*nQuadraturePoints_elementBoundary*nDOF_trial_element +
			  k*nDOF_trial_element +
			  j]
			*
			n[ebNE*nQuadraturePoints_elementBoundary*nSpace + 
			  k*nSpace + 
			  I];
		      Jacobian_n += 
			diffusiveVelocityComponent_I_Jacobian2_nn 
			*
			dphi_n_n[j_global]
			*
			n[ebNE*nQuadraturePoints_elementBoundary*nSpace + 
			  k*nSpace + 
			  I];
		    }/*I loop */
		 
		  if (isDOFBoundary_un[ebNE*nQuadraturePoints_elementBoundary + k] == 2)
		    {
		      /*dependency of psi_n on psi_w*/
		      Jacobian_n += 
			penalty_n[ebNE*nQuadraturePoints_elementBoundary+k]
			* /*should be v_n*/
			v[ebNE*nQuadraturePoints_elementBoundary*nDOF_trial_element+
			  k*nDOF_trial_element+
			  j]
			*
			dpsi_n_dpsiw[ebNE*nQuadraturePoints_elementBoundary + k];
		      /*dependency of psi_n on s_w */
		      Jacobian_w += 
			penalty_n[ebNE*nQuadraturePoints_elementBoundary+k]
			* /*should be v_w*/
			v[ebNE*nQuadraturePoints_elementBoundary*nDOF_trial_element+
			  k*nDOF_trial_element+
			  j]
			*
			dpsi_n_dsw[ebNE*nQuadraturePoints_elementBoundary + k];
		    }
		  else if (isDOFBoundary_un[ebNE*nQuadraturePoints_elementBoundary + k] == 1)
		    {
		      /*only diagonal gets penalty term*/
		      Jacobian_n += 
			penalty_n[ebNE*nQuadraturePoints_elementBoundary+k]
			* /*should be v_w*/
			v[ebNE*nQuadraturePoints_elementBoundary*nDOF_trial_element+
			  k*nDOF_trial_element+
			  j];

		    }
		  fluxJacobian_nw[ebNE*nQuadraturePoints_elementBoundary*nDOF_trial_element + 
				  k*nDOF_trial_element +
				  j] += 
		    Jacobian_w;
		  fluxJacobian_nn[ebNE*nQuadraturePoints_elementBoundary*nDOF_trial_element + 
				  k*nDOF_trial_element +
				  j] += 
		    Jacobian_n;
		}/* j local dof loop*/
	    }/*u_w dof boundary loop*/
	  
	}/*k*/
    }/*ebNE*/

}

/**
   \brief Calculate the advective (gravity) flux at at exterior element boundaries for Darcy FC
   does not upwind right now
*/
void calculateGlobalExteriorNumericalAdvectiveFlux_DarcyFC(proteus_index_t nExteriorElementBoundaries_global,
							   int nQuadraturePoints_elementBoundary,
							   int nSpace,
							   proteus_index_t* exteriorElementBoundaries,
							   proteus_index_t* elementBoundaryElements,
							   int* elementBoundaryLocalElementBoundaries,
							   int *isDOFBoundary_sw,
							   int *isDOFBoundary_psiw,
							   double* n,
							   double* bc_sw,
							   double* bc_psiw,
							   double* bc_fw,
							   double* bc_dfw_dsw,
							   double* bc_dfw_dpsiw,
							   double* bc_fn,
							   double* bc_dfn_dsw,
							   double* bc_dfn_dpsiw,
							   double* sw,
							   double* psiw,
							   double* fw,
							   double* dfw_dsw,
							   double* dfw_dpsiw,
							   double* fn,
							   double* dfn_dsw,
							   double* dfn_dpsiw,
							   double* fluxw,
							   double* dfluxw_dsw,
							   double* dfluxw_dpsiw,
							   double* fluxn,
							   double* dfluxn_dsw,
							   double* dfluxn_dpsiw)
{
  proteus_index_t ebN, eN_global; int ebNE, k, J;
  double left_flux;
  double dflux_dsw_left,dflux_dpsiw_left;
  int enforceOutflow = 1;
  for(ebNE=0;ebNE<nExteriorElementBoundaries_global;ebNE++)
    {
      ebN = exteriorElementBoundaries[ebNE];
      eN_global = elementBoundaryElements[ebN*2+0];
      for(k=0;k<nQuadraturePoints_elementBoundary;k++)
        {
          left_flux=0.0;
	  dflux_dsw_left=0.0; 
	  dflux_dpsiw_left=0.0;
          for(J=0;J<nSpace;J++)
            {
	      dflux_dsw_left +=
		n[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                  k*nSpace+
		  J]
                *
                dfw_dsw[ebNE*nQuadraturePoints_elementBoundary*nSpace+
			k*nSpace+
			J];
	      dflux_dpsiw_left +=
		n[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                  k*nSpace+
		  J]
                *
                dfw_dpsiw[ebNE*nQuadraturePoints_elementBoundary*nSpace+
			  k*nSpace+
			  J];

             left_flux 
                += 
                n[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                  k*nSpace+
		  J]
                *
                fw[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                  k*nSpace+
                  J];
            }
	  if (!enforceOutflow || isDOFBoundary_sw[ebNE*nQuadraturePoints_elementBoundary+k] || left_flux >= 0.0)
	    {
	      fluxw[ebNE*nQuadraturePoints_elementBoundary+
		    k] = left_flux;
	      dfluxw_dsw[ebNE*nQuadraturePoints_elementBoundary+
			 k] = dflux_dsw_left;
	      dfluxw_dpsiw[ebNE*nQuadraturePoints_elementBoundary+
			   k] = dflux_dpsiw_left;
	    }
	  else
	    {
	      fluxw[ebNE*nQuadraturePoints_elementBoundary+
		    k] = 0.0;
	      dfluxw_dsw[ebNE*nQuadraturePoints_elementBoundary+
			 k] = 0.0;
	      dfluxw_dpsiw[ebNE*nQuadraturePoints_elementBoundary+
			   k] = 0.0;

	    }
	  /*now repeat for non-wetting phase*/
          left_flux=0.0;
	  dflux_dsw_left=0.0; 
	  dflux_dpsiw_left=0.0;
          for(J=0;J<nSpace;J++)
            {
	      dflux_dsw_left +=
		n[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                  k*nSpace+
		  J]
                *
                dfn_dsw[ebNE*nQuadraturePoints_elementBoundary*nSpace+
			k*nSpace+
			J];
	      dflux_dpsiw_left +=
		n[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                  k*nSpace+
		  J]
                *
                dfn_dpsiw[ebNE*nQuadraturePoints_elementBoundary*nSpace+
			  k*nSpace+
			  J];
             left_flux 
                += 
                n[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                  k*nSpace+
		  J]
                *
                fn[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                  k*nSpace+
                  J];
            }
	  if (!enforceOutflow || isDOFBoundary_psiw[ebNE*nQuadraturePoints_elementBoundary+k] || left_flux >= 0.0)
	    {
	      fluxn[ebNE*nQuadraturePoints_elementBoundary+
		    k] = left_flux;
	      dfluxn_dsw[ebNE*nQuadraturePoints_elementBoundary+
			 k] = dflux_dsw_left;
	      dfluxn_dpsiw[ebNE*nQuadraturePoints_elementBoundary+
			   k] = dflux_dpsiw_left;
	    }
	  else
	    {
	      fluxn[ebNE*nQuadraturePoints_elementBoundary+
		    k] = 0.0;
	      dfluxn_dsw[ebNE*nQuadraturePoints_elementBoundary+
			 k] = 0.0;
	      dfluxn_dpsiw[ebNE*nQuadraturePoints_elementBoundary+
			   k] = 0.0;

	    }
        }/*k*/
    }
}


/*begin FCPP exterior flux terms*/
void calculateGlobalExteriorNumericalFluxDarcyFCPP(proteus_index_t nExteriorElementBoundaries_global,
						   int nQuadraturePoints_elementBoundary,
						   int nSpace,
						   const proteus_index_t* exteriorElementBoundaries,
						   const proteus_index_t* elementBoundaryElements,
						   const int* elementBoundaryLocalElementBoundaries,
						   const int* isDOFBoundary_uw,/*1 set bc for psi_w*/
						   const int* isDOFBoundary_un,/*1 set bc for psi_c, 
										 2 set bc for psi_n*/
						   int fluxBoundaryFlag_uw, /*0 no flow, 1 outflow*/
						   int fluxBoundaryFlag_un,
						   const double* n,
						   const double* bc_a_ww,      
						   const double* bc_a_nn,      
						   const double* bc_grad_phi_w,
						   const double* bc_grad_phi_n,
						   const double* bc_psi_w,        
						   const double* bc_psi_c,       
						   const double* bc_psi_n,
						   const double* a_ww,         /*lambda_w K_s*/
						   const double* a_nn,         /*lambda_n K_s*/
						   const double* grad_phi_w,   /*psi_w - rho_w g . x*/
						   const double* grad_phi_n,   /*psi_c + psi_w - rho_n g . x*/
						   const double* psi_w,           /*psi_w*/
						   const double* psi_c,           /*psi_c*/
						   const double* psi_n,
						   const double* penalty_w,    
						   const double* penalty_n,
						   double * diffusiveFlux_ww,
						   double * diffusiveFlux_nn)
{
  proteus_index_t ebN; int ebNE, I, J, k, nSpace2=nSpace*nSpace;
  double diffusiveFlux_I=0.0,penaltyFlux = 0.0;
  double potential_gradient_w=0.0,potential_gradient_n=0.0;

  for (ebNE = 0; ebNE < nExteriorElementBoundaries_global; ebNE++)
    {
      ebN = exteriorElementBoundaries[ebNE];
      for (k = 0; k < nQuadraturePoints_elementBoundary; k++)
	{
	  /*compute diffusive flux for first (w) equation (aq. mass
	    balance for part of boundary where u_0 (i.e., S_w) is
	    specified*/
	  diffusiveFlux_ww[ebNE*nQuadraturePoints_elementBoundary+k] = 0.0;
	  diffusiveFlux_nn[ebNE*nQuadraturePoints_elementBoundary+k] = 0.0;
	  /*allow outflow where potential gradient is out even if not Dirichlet, 
	   do not include K_s in calculation for now*/
	  potential_gradient_w = 0.0; potential_gradient_n = 0.0; 
	  for (I=0; I < nSpace; I++)
	    {
	      potential_gradient_w += grad_phi_w[ebNE*nQuadraturePoints_elementBoundary*nSpace +
						 k*nSpace + I]
		*
		n[ebNE*nQuadraturePoints_elementBoundary*nSpace +
		  k*nSpace+
		  I];
	      potential_gradient_n += grad_phi_n[ebNE*nQuadraturePoints_elementBoundary*nSpace +
						 k*nSpace + I]
		*
		n[ebNE*nQuadraturePoints_elementBoundary*nSpace +
		  k*nSpace+
		  I];
	    }
	  
	  /*only allow psi_w setting here?*/
	  if (isDOFBoundary_uw[ebNE*nQuadraturePoints_elementBoundary+k] == 1 || (potential_gradient_w > 0.0 && fluxBoundaryFlag_uw == 1))
	    {
	      /*integration by parts term for diffusive flux wrt phi_w = psi_w - rho_w g . x*/
	      for (I = 0; I < nSpace; I++)
		{
		  diffusiveFlux_I = 0.0;
		  for (J = 0; J < nSpace; J++)
		    {
		      diffusiveFlux_I -= 
			a_ww[ebNE*nQuadraturePoints_elementBoundary*nSpace2 + 
			     k*nSpace2 + 
			     I*nSpace + 
			     J]
			*
			grad_phi_w[ebNE*nQuadraturePoints_elementBoundary*nSpace + 
				   k*nSpace + 
				   J];
		    }
		  diffusiveFlux_ww[ebNE*nQuadraturePoints_elementBoundary+k] += 
		    diffusiveFlux_I
		    *
		    n[ebNE*nQuadraturePoints_elementBoundary*nSpace +
		      k*nSpace+
		      I];
		}/*I, a_wm grad phi_m term */
	      /*boundary penalty term*/
	      if (isDOFBoundary_uw[ebNE*nQuadraturePoints_elementBoundary+k] == 1)
		{
		  penaltyFlux = 
		    penalty_w[ebNE*nQuadraturePoints_elementBoundary + k]
		    *
		    (psi_w[ebNE*nQuadraturePoints_elementBoundary + k]
		     -
		     bc_psi_w[ebNE*nQuadraturePoints_elementBoundary + k]);
		  diffusiveFlux_ww[ebNE*nQuadraturePoints_elementBoundary +k] += 
		    penaltyFlux;
		}
	    }/*psi_w boundary*/
	  /*1 set psi_c, 2 set psi_n*/
	  if(isDOFBoundary_un[ebNE*nQuadraturePoints_elementBoundary+k] >= 1 || (potential_gradient_n > 0.0 && fluxBoundaryFlag_un == 1))
	    {
	      /*integration by parts term for diffusive flux wrt phi_n = psi_c + psi_w - rho_n g . x*/
	      for (I = 0; I < nSpace; I++)
		{
		  diffusiveFlux_I = 0.0;
		  for (J = 0; J < nSpace; J++)
		    {
		      diffusiveFlux_I -= 
			a_nn[ebNE*nQuadraturePoints_elementBoundary*nSpace2 + 
			     k*nSpace2 +
			     I*nSpace + 
			     J]
			*
			grad_phi_n[ebNE*nQuadraturePoints_elementBoundary*nSpace + 
				   k*nSpace +
				   J];
		    }/*J*/
		  diffusiveFlux_nn[ebNE*nQuadraturePoints_elementBoundary + k] += 
		    diffusiveFlux_I
		    *
		    n[ebNE*nQuadraturePoints_elementBoundary*nSpace + 
		      k*nSpace + 
		      I];
		}/*I, a_mw grad phi_n term */
	      /*boundary penalty term*/
	      penaltyFlux = 0.0;
	      //need to enforce psi_c >= 0
	      if (isDOFBoundary_un[ebNE*nQuadraturePoints_elementBoundary+k] == 2)
		{
		  penaltyFlux = 
		    penalty_n[ebNE*nQuadraturePoints_elementBoundary + k]
		    *
		    (psi_n[ebNE*nQuadraturePoints_elementBoundary + k]
		     -
		     bc_psi_n[ebNE*nQuadraturePoints_elementBoundary + k]);
		}
	      else if (isDOFBoundary_un[ebNE*nQuadraturePoints_elementBoundary+k] == 1)
		{
		  penaltyFlux = 
		    penalty_n[ebNE*nQuadraturePoints_elementBoundary + k]
		    *
		    (psi_c[ebNE*nQuadraturePoints_elementBoundary + k]
		     -
		     bc_psi_c[ebNE*nQuadraturePoints_elementBoundary + k]);

		}
	      diffusiveFlux_nn[ebNE*nQuadraturePoints_elementBoundary +k] += 
		penaltyFlux;
	    }/*um boundary*/
	}/*k*/
    }/*ebNE*/
}

void calculateGlobalExteriorNumericalFluxDarcyFCPP_sd(proteus_index_t nExteriorElementBoundaries_global,
						      int nQuadraturePoints_elementBoundary,
						      int nSpace,
						      proteus_csr_index_t* rowptr_ww,
						      proteus_csr_index_t* colind_ww,
						      proteus_csr_index_t* rowptr_nn,
						      proteus_csr_index_t* colind_nn,
						      const proteus_index_t* exteriorElementBoundaries,
						      const proteus_index_t* elementBoundaryElements,
						      const int* elementBoundaryLocalElementBoundaries,
						      const int* isDOFBoundary_uw,/*1 set bc for psi_w*/
						      const int* isDOFBoundary_un,/*1 set bc for psi_c, 
										    2 set bc for psi_n*/
						      int fluxBoundaryFlag_uw, /*0 no flow, 1 outflow*/
						      int fluxBoundaryFlag_un,
						      const double* n,
						      const double* bc_a_ww,      
						      const double* bc_a_nn,      
						      const double* bc_grad_phi_w,
						      const double* bc_grad_phi_n,
						      const double* bc_psi_w,        
						      const double* bc_psi_c,       
						      const double* bc_psi_n,
						      const double* a_ww,         /*lambda_w K_s*/
						      const double* a_nn,         /*lambda_n K_s*/
						      const double* grad_phi_w,   /*psi_w - rho_w g . x*/
						      const double* grad_phi_n,   /*psi_c + psi_w - rho_n g . x*/
						      const double* psi_w,           /*s_w*/
						      const double* psi_c,           /*psi_w*/
						      const double* psi_n,
						      const double* penalty_w,    
						      const double* penalty_n,
						      double * diffusiveFlux_ww,
						      double * diffusiveFlux_nn)
{
  proteus_index_t ebN; proteus_csr_index_t m; proteus_csr_index_t nnz_nn=rowptr_nn[nSpace]; proteus_csr_index_t nnz_ww=rowptr_ww[nSpace]; int ebNE, I, k;
  double diffusiveFlux_I=0.0,penaltyFlux = 0.0,potential_gradient_w=0.0,potential_gradient_n=0.0;
  for (ebNE = 0; ebNE < nExteriorElementBoundaries_global; ebNE++)
    {
      ebN = exteriorElementBoundaries[ebNE];
      for (k = 0; k < nQuadraturePoints_elementBoundary; k++)
	{
	  /*compute diffusive flux for first (w) equation (aq. mass
	    balance for part of boundary where u_0 (i.e., S_w) is
	    specified*/
	  diffusiveFlux_ww[ebNE*nQuadraturePoints_elementBoundary+k] = 0.0;
	  diffusiveFlux_nn[ebNE*nQuadraturePoints_elementBoundary+k] = 0.0;
	  potential_gradient_w = 0.0; potential_gradient_n = 0.0; 
	  for (I=0; I < nSpace; I++)
	    {
	      potential_gradient_w += grad_phi_w[ebNE*nQuadraturePoints_elementBoundary*nSpace +
						 k*nSpace + I]
		*
		n[ebNE*nQuadraturePoints_elementBoundary*nSpace +
		  k*nSpace+
		  I];
	      potential_gradient_n += grad_phi_n[ebNE*nQuadraturePoints_elementBoundary*nSpace +
						 k*nSpace + I]
		*
		n[ebNE*nQuadraturePoints_elementBoundary*nSpace +
		  k*nSpace+
		  I];
	    }
	  /*mwf hack
	    potential_gradient_w = 0.0; potential_gradient_n = 0.0;
	  */
	  /*only allow psi_w setting here?*/
	  if (isDOFBoundary_uw[ebNE*nQuadraturePoints_elementBoundary+k] == 1 || (potential_gradient_w > 0.0 && fluxBoundaryFlag_uw == 1))
	    {
	      /*integration by parts term for diffusive flux wrt phi_w = psi_w - rho_w g . x*/
	      for (I = 0; I < nSpace; I++)
		{
		  diffusiveFlux_I = 0.0;
		  for(m=rowptr_ww[I];m<rowptr_ww[I+1];m++)
		    {
		      diffusiveFlux_I -= 
			a_ww[ebNE*nQuadraturePoints_elementBoundary*nnz_ww + 
			     k*nnz_ww + 
			     m]
			*
			grad_phi_w[ebNE*nQuadraturePoints_elementBoundary*nSpace + 
				   k*nSpace + 
				   colind_ww[m]];
		    }
		  diffusiveFlux_ww[ebNE*nQuadraturePoints_elementBoundary+k] += 
		    diffusiveFlux_I
		    *
		    n[ebNE*nQuadraturePoints_elementBoundary*nSpace +
		      k*nSpace+
		      I];
		}/*I, a_wm grad phi_m term */
	      /*boundary penalty term*/
	      if (isDOFBoundary_uw[ebNE*nQuadraturePoints_elementBoundary+k] == 1)
		{
		  penaltyFlux = 
		    penalty_w[ebNE*nQuadraturePoints_elementBoundary + k]
		    *
		    (psi_w[ebNE*nQuadraturePoints_elementBoundary + k]
		     -
		     bc_psi_w[ebNE*nQuadraturePoints_elementBoundary + k]);
		  diffusiveFlux_ww[ebNE*nQuadraturePoints_elementBoundary +k] += 
		    penaltyFlux;
		}
	    }/*psi_w boundary*/
	  /*1 set psi_c, 2 set psi_n*/
	  if(isDOFBoundary_un[ebNE*nQuadraturePoints_elementBoundary+k] >= 1 || (potential_gradient_n > 0.0 && fluxBoundaryFlag_un == 1))
	    {
	      /*integration by parts term for diffusive flux wrt phi_n = psi_c + psi_w - rho_n g . x*/
	      for (I = 0; I < nSpace; I++)
		{
		  diffusiveFlux_I = 0.0;
		  for(m=rowptr_nn[I];m<rowptr_nn[I+1];m++)
		    {
		      diffusiveFlux_I -= 
			a_nn[ebNE*nQuadraturePoints_elementBoundary*nnz_nn + 
			     k*nnz_nn +
			     m]
			*
			grad_phi_n[ebNE*nQuadraturePoints_elementBoundary*nSpace + 
				   k*nSpace +
				   colind_nn[m]];
		    }/*J*/
		  diffusiveFlux_nn[ebNE*nQuadraturePoints_elementBoundary + k] += 
		    diffusiveFlux_I
		    *
		    n[ebNE*nQuadraturePoints_elementBoundary*nSpace + 
		      k*nSpace + 
		      I];
		}/*I, a_mw grad phi_n term */
	      /*boundary penalty term*/
	      penaltyFlux = 0.0;
	      if (isDOFBoundary_un[ebNE*nQuadraturePoints_elementBoundary+k] == 2)
		{
		  penaltyFlux = 
		    penalty_n[ebNE*nQuadraturePoints_elementBoundary + k]
		    *
		    (psi_n[ebNE*nQuadraturePoints_elementBoundary + k]
		     -
		     bc_psi_n[ebNE*nQuadraturePoints_elementBoundary + k]);
		}
	      else if (isDOFBoundary_un[ebNE*nQuadraturePoints_elementBoundary+k] == 1)
		{
		  penaltyFlux = 
		    penalty_n[ebNE*nQuadraturePoints_elementBoundary + k]
		    *
		    (psi_c[ebNE*nQuadraturePoints_elementBoundary + k]
		     -
		     bc_psi_c[ebNE*nQuadraturePoints_elementBoundary + k]);

		}
	      diffusiveFlux_nn[ebNE*nQuadraturePoints_elementBoundary +k] += 
		penaltyFlux;
	    }/*um boundary*/
	}/*k*/
    }/*ebNE*/
}

void calculateGlobalExteriorNumericalFluxDarcyFCPP_diffusiveFluxJacobian(proteus_index_t nExteriorElementBoundaries_global,
									 int nQuadraturePoints_elementBoundary,
									 int nSpace,
									 int nDOF_trial_element,
									 const proteus_index_t* l2g, /*for now assumes both solution spaces are the same!*/
									 const proteus_index_t* exteriorElementBoundaries,
									 const proteus_index_t* elementBoundaryElements,
									 const int* elementBoundaryLocalElementBoundaries,
									 const int* isDOFBoundary_uw,/*1 set bc for psi_w*/
									 const int* isDOFBoundary_un,/*1 set bc for psi_c, 
												       2 set bc for psi_n*/
									 int fluxBoundaryFlag_uw, /*0 no flow, 1 outflow*/
									 int fluxBoundaryFlag_un,
									 const double* n,
									 const double* a_ww,         /*lambda_w K_s*/
									 const double* da_ww_dw,         /* a' wrt S_w*/
									 const double* da_ww_dn,         /* a' wrt psi_w*/
									 const double* a_nn,         /*lambda_t K_s*/
									 const double* da_nn_dw,         /* a' wrt S_w*/
									 const double* da_nn_dn,         /* a' wrt psi_w*/
									 const double* grad_phi_w,   /*psi_w - rho_w g . x*/
									 const double* grad_phi_n,   /*psi_n + psi_w - rho_n g . x*/
									 const double* dphi_w_w,     /*\pd{phi_w}{psi_w} = 1 */     
									 const double* dphi_w_n,     /*\pd{phi_w}{psi_c}= 0 */
									 const double* dphi_n_w,     /*\pd{phi_n}{psi_w} = 1  */
									 const double* dphi_n_n,     /*\pd{phi_n}{psi_c} = 1  */
									 const double* psi_w,           /*psi_w*/
									 const double* psi_c,           /*psi_c*/
									 const double* psi_n,           
									 const double* dpsi_n_dpsiw, 
									 const double* dpsi_n_dpsic,
									 const double* v,            /*trial functions, assumed in same space*/
									 const double* grad_v,       /*trial function gradients, assumed in same space*/
									 const double* penalty_w,    
									 const double* penalty_n,
									 double * fluxJacobian_ww,
									 double * fluxJacobian_wn,
									 double * fluxJacobian_nw,
									 double * fluxJacobian_nn)
{
  proteus_index_t ebN, eN_global, j_global; int ebNE, j, I, J, k, nSpace2=nSpace*nSpace;
  double Jacobian_w,Jacobian_n,
    diffusiveVelocityComponent_I_Jacobian_w,
    diffusiveVelocityComponent_I_Jacobian_n,  
    diffusiveVelocityComponent_I_Jacobian2_wn,
    diffusiveVelocityComponent_I_Jacobian2_ww,
    diffusiveVelocityComponent_I_Jacobian2_nw,
    diffusiveVelocityComponent_I_Jacobian2_nn;
  double potential_gradient_w=0.0,potential_gradient_n=0.0;
  for (ebNE = 0; ebNE < nExteriorElementBoundaries_global; ebNE++)
    {
      ebN = exteriorElementBoundaries[ebNE];
      eN_global = elementBoundaryElements[ebN*2 + 0];
      for (k = 0; k < nQuadraturePoints_elementBoundary; k++)
	{
	  /*compute derivative of diffusive flux for first (w) equation (aq. mass
	    balance for part of boundary where u_0 (i.e., psi_w) is
	    specified
	  */
	  /*allow outflow where potential gradient is out even if not Dirichlet, 
	   do not include K_s in calculation for now*/
	  potential_gradient_w = 0.0; potential_gradient_n = 0.0; 
	  for (I=0; I < nSpace; I++)
	    {
	      potential_gradient_w += grad_phi_w[ebNE*nQuadraturePoints_elementBoundary*nSpace +
						 k*nSpace + I]
		*
		n[ebNE*nQuadraturePoints_elementBoundary*nSpace +
		  k*nSpace+
		  I];
	      potential_gradient_n += grad_phi_n[ebNE*nQuadraturePoints_elementBoundary*nSpace +
						 k*nSpace + I]
		*
		n[ebNE*nQuadraturePoints_elementBoundary*nSpace +
		  k*nSpace+
		  I];
	    }
	  /*only allow setting psi_w for this equation*/	  
	  if (isDOFBoundary_uw[ebNE*nQuadraturePoints_elementBoundary + k] == 1 || (potential_gradient_w > 0.0 && fluxBoundaryFlag_uw == 1))
	    {
	      /*NOTE! assuming u_w, u_n in the same space and nodal interpolant for potential*/
	      
	      for (j = 0; j < nDOF_trial_element; j++)
		{
		  Jacobian_w = 0.; /*derivative wrt u_w = S_w */
		  Jacobian_n = 0.; /*derivative wrt u_n = psi_w */
		  j_global = l2g[eN_global*nDOF_trial_element + j];/*assuming same space for both*/ 
		  for (I = 0; I < nSpace; I++)
		    {
		      diffusiveVelocityComponent_I_Jacobian_w = 0.0; 
		      diffusiveVelocityComponent_I_Jacobian_n = 0.0; 
		      diffusiveVelocityComponent_I_Jacobian2_wn = 0.0;
		      diffusiveVelocityComponent_I_Jacobian2_ww = 0.0;
		      for (J = 0; J < nSpace; J++)
			{
			  /*only a_ww potential here*/
			  diffusiveVelocityComponent_I_Jacobian_w -= 
			    da_ww_dw[ebNE*nQuadraturePoints_elementBoundary*nSpace2 + 
				     k*nSpace2 + 
				     I*nSpace  + 
				     J]
			    *
			    grad_phi_w[ebNE*nQuadraturePoints_elementBoundary*nSpace + 
				       k*nSpace + 
				       J];
			  diffusiveVelocityComponent_I_Jacobian_n -= 
			    da_ww_dn[ebNE*nQuadraturePoints_elementBoundary*nSpace2 + 
				     k*nSpace2 + 
				     I*nSpace  + 
				     J]
			    *
			    grad_phi_w[ebNE*nQuadraturePoints_elementBoundary*nSpace + 
				       k*nSpace + 
				       J];

			  diffusiveVelocityComponent_I_Jacobian2_ww -=
			    a_ww[ebNE*nQuadraturePoints_elementBoundary*nSpace2 + 
				 k*nSpace2 + 
				 I*nSpace  + 
				 J]
			    * /*should be grad_v_w in general I believe*/
			    grad_v[ebNE*nQuadraturePoints_elementBoundary*nDOF_trial_element*nSpace + 
				   k*nDOF_trial_element*nSpace+
                                   j*nSpace+
                                   J];
			  /*identical for now*/
			  diffusiveVelocityComponent_I_Jacobian2_wn -=
			    a_ww[ebNE*nQuadraturePoints_elementBoundary*nSpace2 + 
				 k*nSpace2 + 
				 I*nSpace  + 
				 J]
			    * /*should be grad_v_w in general I believe*/
			    grad_v[ebNE*nQuadraturePoints_elementBoundary*nDOF_trial_element*nSpace + 
				   k*nDOF_trial_element*nSpace+
                                   j*nSpace+
                                   J];
			    
			}/*J loop*/
		      Jacobian_w += 
			diffusiveVelocityComponent_I_Jacobian_w 
			*/*should be v_w*/
			v[ebNE*nQuadraturePoints_elementBoundary*nDOF_trial_element +
			  k*nDOF_trial_element +
			  j]
			*
			n[ebNE*nQuadraturePoints_elementBoundary*nSpace + 
			  k*nSpace + 
			  I];
		      Jacobian_w += 
			diffusiveVelocityComponent_I_Jacobian2_ww 
			*
			dphi_w_w[j_global]
			*
			n[ebNE*nQuadraturePoints_elementBoundary*nSpace + 
			  k*nSpace + 
			  I];
		      Jacobian_n += 
			diffusiveVelocityComponent_I_Jacobian_n 
			*/*should be v_n*/
			v[ebNE*nQuadraturePoints_elementBoundary*nDOF_trial_element +
			  k*nDOF_trial_element +
			  j]
			*
			n[ebNE*nQuadraturePoints_elementBoundary*nSpace + 
			  k*nSpace + 
			  I];
		      Jacobian_n += 
			diffusiveVelocityComponent_I_Jacobian2_wn 
			*
			dphi_w_n[j_global]
			*
			n[ebNE*nQuadraturePoints_elementBoundary*nSpace + 
			  k*nSpace + 
			  I];
		    }/*I loop */
		  /*only diagonal gets penalty term*/
		  if (isDOFBoundary_uw[ebNE*nQuadraturePoints_elementBoundary+k] == 1)
		    {
		      Jacobian_w += 
			penalty_w[ebNE*nQuadraturePoints_elementBoundary+k]
			* /*should be v_w*/
			v[ebNE*nQuadraturePoints_elementBoundary*nDOF_trial_element+
			  k*nDOF_trial_element+
			  j];
		    }
		  fluxJacobian_ww[ebNE*nQuadraturePoints_elementBoundary*nDOF_trial_element + 
				  k*nDOF_trial_element +
				  j] += 
		    Jacobian_w;
		  fluxJacobian_wn[ebNE*nQuadraturePoints_elementBoundary*nDOF_trial_element + 
				  k*nDOF_trial_element +
				  j] += 
		    Jacobian_n;
		}/* j local dof loop*/
	    }/*u_w dof boundary loop*/
	  /*setting psi_w = 1, or psi_n = 2*/
	  if (isDOFBoundary_un[ebNE*nQuadraturePoints_elementBoundary + k] >= 1 || (potential_gradient_n > 0.0 && fluxBoundaryFlag_un == 1))
	    {
	      /*NOTE! assuming u_w, u_m in the same space and nodal interpolant for potential*/
	      
	      for (j = 0; j < nDOF_trial_element; j++)
		{
		  Jacobian_w = 0.; /*derivative wrt u_w = psi_w */
		  Jacobian_n = 0.; /*derivative wrt u_n = psi_c */
		  j_global = l2g[eN_global*nDOF_trial_element + j];/*assuming same space for both*/ 
		  for (I = 0; I < nSpace; I++)
		    {
		      diffusiveVelocityComponent_I_Jacobian_w = 0.0;
		      diffusiveVelocityComponent_I_Jacobian_n = 0.0;  
		      diffusiveVelocityComponent_I_Jacobian2_nw = 0.0;
		      diffusiveVelocityComponent_I_Jacobian2_nn = 0.0;
		      for (J = 0; J < nSpace; J++)
			{
			  /*nn potential*/
			  diffusiveVelocityComponent_I_Jacobian_w -= 
			    da_nn_dw[ebNE*nQuadraturePoints_elementBoundary*nSpace2 + 
				     k*nSpace2 + 
				     I*nSpace  + 
				     J]
			    *
			    grad_phi_n[ebNE*nQuadraturePoints_elementBoundary*nSpace + 
				       k*nSpace + 
				       J];
			  diffusiveVelocityComponent_I_Jacobian_n -= 
			    da_nn_dn[ebNE*nQuadraturePoints_elementBoundary*nSpace2 + 
				     k*nSpace2 + 
				     I*nSpace  + 
				     J]
			    *
			    grad_phi_n[ebNE*nQuadraturePoints_elementBoundary*nSpace + 
				       k*nSpace + 
				       J];

			  /*nn potential*/
			  diffusiveVelocityComponent_I_Jacobian2_nw -=
			    a_nn[ebNE*nQuadraturePoints_elementBoundary*nSpace2 + 
				 k*nSpace2 + 
				 I*nSpace  + 
				 J]
			    * /*should be grad_v_w in general I believe*/
			    grad_v[ebNE*nQuadraturePoints_elementBoundary*nDOF_trial_element*nSpace + 
				   k*nDOF_trial_element*nSpace+
                                   j*nSpace+
                                   J];
			  /*identical for now*/
			  diffusiveVelocityComponent_I_Jacobian2_nn -=
			    a_nn[ebNE*nQuadraturePoints_elementBoundary*nSpace2 + 
				 k*nSpace2 + 
				 I*nSpace  + 
				 J]
			    * /*should be grad_v_m in general I believe*/
			    grad_v[ebNE*nQuadraturePoints_elementBoundary*nDOF_trial_element*nSpace + 
				   k*nDOF_trial_element*nSpace+
                                   j*nSpace+
                                   J];
			    
			}/*J loop*/
		      Jacobian_w += 
			diffusiveVelocityComponent_I_Jacobian_w 
			*/*should be v_w*/
			v[ebNE*nQuadraturePoints_elementBoundary*nDOF_trial_element +
			  k*nDOF_trial_element +
			  j]
			*
			n[ebNE*nQuadraturePoints_elementBoundary*nSpace + 
			  k*nSpace + 
			  I];
		      Jacobian_w += 
			diffusiveVelocityComponent_I_Jacobian2_nw 
			*
			dphi_n_w[j_global]
			*
			n[ebNE*nQuadraturePoints_elementBoundary*nSpace + 
			  k*nSpace + 
			  I];

		      Jacobian_n += 
			diffusiveVelocityComponent_I_Jacobian_n 
			*/*should be v_n*/
			v[ebNE*nQuadraturePoints_elementBoundary*nDOF_trial_element +
			  k*nDOF_trial_element +
			  j]
			*
			n[ebNE*nQuadraturePoints_elementBoundary*nSpace + 
			  k*nSpace + 
			  I];
		      Jacobian_n += 
			diffusiveVelocityComponent_I_Jacobian2_nn 
			*
			dphi_n_n[j_global]
			*
			n[ebNE*nQuadraturePoints_elementBoundary*nSpace + 
			  k*nSpace + 
			  I];
		    }/*I loop */
		 
		  if (isDOFBoundary_un[ebNE*nQuadraturePoints_elementBoundary + k] == 2)
		    {
		      /*dependency of psi_n on psi_w*/
		      Jacobian_n += 
			penalty_n[ebNE*nQuadraturePoints_elementBoundary+k]
			* /*should be v_n*/
			v[ebNE*nQuadraturePoints_elementBoundary*nDOF_trial_element+
			  k*nDOF_trial_element+
			  j]
			*
			dpsi_n_dpsiw[ebNE*nQuadraturePoints_elementBoundary + k];
		      /*dependency of psi_n on s_w */
		      Jacobian_w += 
			penalty_n[ebNE*nQuadraturePoints_elementBoundary+k]
			* /*should be v_w*/
			v[ebNE*nQuadraturePoints_elementBoundary*nDOF_trial_element+
			  k*nDOF_trial_element+
			  j]
			*
			dpsi_n_dpsic[ebNE*nQuadraturePoints_elementBoundary + k];
		    }
		  else if (isDOFBoundary_un[ebNE*nQuadraturePoints_elementBoundary + k] == 1)
		    {
		      /*only diagonal gets penalty term*/
		      
		      Jacobian_n += 
			penalty_n[ebNE*nQuadraturePoints_elementBoundary+k]
			* /*should be v_w*/
			v[ebNE*nQuadraturePoints_elementBoundary*nDOF_trial_element+
			  k*nDOF_trial_element+
			  j];

		    }
		  fluxJacobian_nw[ebNE*nQuadraturePoints_elementBoundary*nDOF_trial_element + 
				  k*nDOF_trial_element +
				  j] += 
		    Jacobian_w;
		  fluxJacobian_nn[ebNE*nQuadraturePoints_elementBoundary*nDOF_trial_element + 
				  k*nDOF_trial_element +
				  j] += 
		    Jacobian_n;
		}/* j local dof loop*/
	    }/*u_w dof boundary loop*/
	  
	}/*k*/
    }/*ebNE*/

}

void calculateGlobalExteriorNumericalFluxDarcyFCPP_diffusiveFluxJacobian_sd(proteus_index_t nExteriorElementBoundaries_global,
									    int nQuadraturePoints_elementBoundary,
									    int nSpace,
									    int nDOF_trial_element,
									    proteus_csr_index_t* rowptr_ww,
									    proteus_csr_index_t* colind_ww,
									    proteus_csr_index_t* rowptr_nn,
									    proteus_csr_index_t* colind_nn,
									    const proteus_index_t* l2g, /*for now assumes both solution spaces are the same!*/
									    const proteus_index_t* exteriorElementBoundaries,
									    const proteus_index_t* elementBoundaryElements,
									    const int* elementBoundaryLocalElementBoundaries,
									    const int* isDOFBoundary_uw,/*1 set bc for psi_w*/
									    const int* isDOFBoundary_un,/*1 set bc for psi_c, 
													  2 set bc for psi_n*/
									    int fluxBoundaryFlag_uw, /*0 no flow, 1 outflow*/
									    int fluxBoundaryFlag_un,
									    const double* n,
									    const double* a_ww,         /*lambda_w K_s*/
									    const double* da_ww_dw,         /* a' wrt S_w*/
									    const double* da_ww_dn,         /* a' wrt psi_w*/
									    const double* a_nn,         /*lambda_t K_s*/
									    const double* da_nn_dw,         /* a' wrt S_w*/
									    const double* da_nn_dn,         /* a' wrt psi_w*/
									    const double* grad_phi_w,   /*psi_w - rho_w g . x*/
									    const double* grad_phi_n,   /*psi_n + psi_w - rho_n g . x*/
									    const double* dphi_w_w,     /*\pd{phi_w}{psi_w} = 1 */     
									    const double* dphi_w_n,     /*\pd{phi_w}{psi_c}= 0 */
									    const double* dphi_n_w,     /*\pd{phi_n}{psi_w} = 1  */
									    const double* dphi_n_n,     /*\pd{phi_n}{psi_c} = 1 */
									    const double* psi_w,           /*psi_w*/
									    const double* psi_c,           /*psi_c*/
									    const double* psi_n,           
									    const double* dpsi_n_dpsiw, 
									    const double* dpsi_n_dpsic,
									    const double* v,            /*trial functions, assumed in same space*/
									    const double* grad_v,       /*trial function gradients, assumed in same space*/
									    const double* penalty_w,    
									    const double* penalty_n,
									    double * fluxJacobian_ww,
									    double * fluxJacobian_wn,
									    double * fluxJacobian_nw,
									    double * fluxJacobian_nn)
{
  proteus_index_t ebN, eN_global, j_global; proteus_csr_index_t m; proteus_csr_index_t nnz_ww=rowptr_ww[nSpace]; proteus_csr_index_t nnz_nn=rowptr_nn[nSpace]; int ebNE, j, I, k;
  double Jacobian_w,Jacobian_n,
    diffusiveVelocityComponent_I_Jacobian_w,
    diffusiveVelocityComponent_I_Jacobian_n,  
    diffusiveVelocityComponent_I_Jacobian2_wn,
    diffusiveVelocityComponent_I_Jacobian2_ww,
    diffusiveVelocityComponent_I_Jacobian2_nw,
    diffusiveVelocityComponent_I_Jacobian2_nn;
  double potential_gradient_w=0.0,potential_gradient_n=0.0;
  for (ebNE = 0; ebNE < nExteriorElementBoundaries_global; ebNE++)
    {
      ebN = exteriorElementBoundaries[ebNE];
      eN_global = elementBoundaryElements[ebN*2 + 0];
      
      for (k = 0; k < nQuadraturePoints_elementBoundary; k++)
	{
	  /*compute derivative of diffusive flux for first (w) equation (aq. mass
	    balance for part of boundary where u_0 (i.e., psi_w) is
	    specified
	  */
	  /*allow outflow where potential gradient is out even if not Dirichlet, 
	   do not include K_s in calculation for now*/
	  potential_gradient_w = 0.0; potential_gradient_n = 0.0; 
	  for (I=0; I < nSpace; I++)
	    {
	      potential_gradient_w += grad_phi_w[ebNE*nQuadraturePoints_elementBoundary*nSpace +
						 k*nSpace + I]
		*
		n[ebNE*nQuadraturePoints_elementBoundary*nSpace +
		  k*nSpace+
		  I];
	      potential_gradient_n += grad_phi_n[ebNE*nQuadraturePoints_elementBoundary*nSpace +
						 k*nSpace + I]
		*
		n[ebNE*nQuadraturePoints_elementBoundary*nSpace +
		  k*nSpace+
		  I];
	    }
	  /*mwf hack
	    potential_gradient_w = 0.0; potential_gradient_n = 0.0;
	  */
	  /*only allow setting psi_w for this equation*/	  
	  if (isDOFBoundary_uw[ebNE*nQuadraturePoints_elementBoundary + k] == 1 || (potential_gradient_w > 0.0 && fluxBoundaryFlag_uw == 1))
	    {
	      /*NOTE! assuming u_w, u_n in the same space and nodal interpolant for potential*/
	      
	      for (j = 0; j < nDOF_trial_element; j++)
		{
		  Jacobian_w = 0.; /*derivative wrt u_w = S_w */
		  Jacobian_n = 0.; /*derivative wrt u_n = psi_w */
		  j_global = l2g[eN_global*nDOF_trial_element + j];/*assuming same space for both*/ 
		  for (I = 0; I < nSpace; I++)
		    {
		      diffusiveVelocityComponent_I_Jacobian_w = 0.0; 
		      diffusiveVelocityComponent_I_Jacobian_n = 0.0; 
		      diffusiveVelocityComponent_I_Jacobian2_wn = 0.0;
		      diffusiveVelocityComponent_I_Jacobian2_ww = 0.0;
		      for (m=rowptr_ww[I];m<rowptr_ww[I+1];m++)
			{
			  /*only a_ww potential here*/
			  diffusiveVelocityComponent_I_Jacobian_w -= 
			    da_ww_dw[ebNE*nQuadraturePoints_elementBoundary*nnz_ww + 
				     k*nnz_ww + 
				     m]
			    *
			    grad_phi_w[ebNE*nQuadraturePoints_elementBoundary*nSpace + 
				       k*nSpace + 
				       colind_ww[m]];
			  diffusiveVelocityComponent_I_Jacobian_n -= 
			    da_ww_dn[ebNE*nQuadraturePoints_elementBoundary*nnz_ww + 
				     k*nnz_ww + 
				     m]
			    *
			    grad_phi_w[ebNE*nQuadraturePoints_elementBoundary*nSpace + 
				       k*nSpace + 
				       colind_ww[m]];

			  diffusiveVelocityComponent_I_Jacobian2_ww -=
			    a_ww[ebNE*nQuadraturePoints_elementBoundary*nnz_ww + 
				 k*nnz_ww + 
				 m]
			    * /*should be grad_v_w in general I believe*/
			    grad_v[ebNE*nQuadraturePoints_elementBoundary*nDOF_trial_element*nSpace + 
				   k*nDOF_trial_element*nSpace+
                                   j*nSpace+
                                   colind_ww[m]];
			  /*identical for now*/
			  diffusiveVelocityComponent_I_Jacobian2_wn -=
			    a_ww[ebNE*nQuadraturePoints_elementBoundary*nnz_ww + 
				 k*nnz_ww + 
				 m]
			    * /*should be grad_v_w in general I believe*/
			    grad_v[ebNE*nQuadraturePoints_elementBoundary*nDOF_trial_element*nSpace + 
				   k*nDOF_trial_element*nSpace+
                                   j*nSpace+
                                   colind_ww[m]];
			    
			}/*J loop*/
		      Jacobian_w += 
			diffusiveVelocityComponent_I_Jacobian_w 
			*/*should be v_w*/
			v[ebNE*nQuadraturePoints_elementBoundary*nDOF_trial_element +
			  k*nDOF_trial_element +
			  j]
			*
			n[ebNE*nQuadraturePoints_elementBoundary*nSpace + 
			  k*nSpace + 
			  I];
		      Jacobian_w += 
			diffusiveVelocityComponent_I_Jacobian2_ww 
			*
			dphi_w_w[j_global]
			*
			n[ebNE*nQuadraturePoints_elementBoundary*nSpace + 
			  k*nSpace + 
			  I];
		      Jacobian_n += 
			diffusiveVelocityComponent_I_Jacobian_n 
			*/*should be v_n*/
			v[ebNE*nQuadraturePoints_elementBoundary*nDOF_trial_element +
			  k*nDOF_trial_element +
			  j]
			*
			n[ebNE*nQuadraturePoints_elementBoundary*nSpace + 
			  k*nSpace + 
			  I];
		      Jacobian_n += 
			diffusiveVelocityComponent_I_Jacobian2_wn 
			*
			dphi_w_n[j_global]
			*
			n[ebNE*nQuadraturePoints_elementBoundary*nSpace + 
			  k*nSpace + 
			  I];
		    }/*I loop */
		  /*only diagonal gets penalty term*/
		  if (isDOFBoundary_uw[ebNE*nQuadraturePoints_elementBoundary + k] == 1)
		    {
		      Jacobian_w += 
			penalty_w[ebNE*nQuadraturePoints_elementBoundary+k]
			* /*should be v_w*/
			v[ebNE*nQuadraturePoints_elementBoundary*nDOF_trial_element+
			  k*nDOF_trial_element+
			  j];
		    }
		  fluxJacobian_ww[ebNE*nQuadraturePoints_elementBoundary*nDOF_trial_element + 
				  k*nDOF_trial_element +
				  j] += 
		    Jacobian_w;
		  fluxJacobian_wn[ebNE*nQuadraturePoints_elementBoundary*nDOF_trial_element + 
				  k*nDOF_trial_element +
				  j] += 
		    Jacobian_n;
		}/* j local dof loop*/
	    }/*u_w dof boundary loop*/
	  /*setting psi_w = 1, or psi_n = 2*/
	  if (isDOFBoundary_un[ebNE*nQuadraturePoints_elementBoundary + k] >= 1 || (potential_gradient_n > 0.0 && fluxBoundaryFlag_un == 1))
	    {
	      /*NOTE! assuming u_w, u_m in the same space and nodal interpolant for potential*/
	      
	      for (j = 0; j < nDOF_trial_element; j++)
		{
		  Jacobian_w = 0.; /*derivative wrt u_w = psi_w */
		  Jacobian_n = 0.; /*derivative wrt u_n = psi_c */
		  j_global = l2g[eN_global*nDOF_trial_element + j];/*assuming same space for both*/ 
		  for (I = 0; I < nSpace; I++)
		    {
		      diffusiveVelocityComponent_I_Jacobian_w = 0.0;
		      diffusiveVelocityComponent_I_Jacobian_n = 0.0;  
		      diffusiveVelocityComponent_I_Jacobian2_nw = 0.0;
		      diffusiveVelocityComponent_I_Jacobian2_nn = 0.0;
		      for (m=rowptr_nn[I];m<rowptr_nn[I+1];m++)
			{
			  /*nn potential*/
			  diffusiveVelocityComponent_I_Jacobian_w -= 
			    da_nn_dw[ebNE*nQuadraturePoints_elementBoundary*nnz_nn + 
				     k*nnz_nn + 
				     m]
			    *
			    grad_phi_n[ebNE*nQuadraturePoints_elementBoundary*nSpace + 
				       k*nSpace + 
				       colind_nn[m]];
			  diffusiveVelocityComponent_I_Jacobian_n -= 
			    da_nn_dn[ebNE*nQuadraturePoints_elementBoundary*nnz_nn + 
				     k*nnz_nn + 
				     m]
			    *
			    grad_phi_n[ebNE*nQuadraturePoints_elementBoundary*nSpace + 
				       k*nSpace + 
				       colind_nn[m]];

			  /*nn potential*/
			  diffusiveVelocityComponent_I_Jacobian2_nw -=
			    a_nn[ebNE*nQuadraturePoints_elementBoundary*nnz_nn + 
				 k*nnz_nn + 
				 m]
			    * /*should be grad_v_w in general I believe*/
			    grad_v[ebNE*nQuadraturePoints_elementBoundary*nDOF_trial_element*nSpace + 
				   k*nDOF_trial_element*nSpace+
                                   j*nSpace+
                                   colind_nn[m]];
			  /*identical for now*/
			  diffusiveVelocityComponent_I_Jacobian2_nn -=
			    a_nn[ebNE*nQuadraturePoints_elementBoundary*nnz_nn + 
				 k*nnz_nn + 
				 m]
			    * /*should be grad_v_m in general I believe*/
			    grad_v[ebNE*nQuadraturePoints_elementBoundary*nDOF_trial_element*nSpace + 
				   k*nDOF_trial_element*nSpace+
                                   j*nSpace+
                                   colind_nn[m]];
			    
			}/*J loop*/
		      Jacobian_w += 
			diffusiveVelocityComponent_I_Jacobian_w 
			*/*should be v_w*/
			v[ebNE*nQuadraturePoints_elementBoundary*nDOF_trial_element +
			  k*nDOF_trial_element +
			  j]
			*
			n[ebNE*nQuadraturePoints_elementBoundary*nSpace + 
			  k*nSpace + 
			  I];
		      Jacobian_w += 
			diffusiveVelocityComponent_I_Jacobian2_nw 
			*
			dphi_n_w[j_global]
			*
			n[ebNE*nQuadraturePoints_elementBoundary*nSpace + 
			  k*nSpace + 
			  I];

		      Jacobian_n += 
			diffusiveVelocityComponent_I_Jacobian_n 
			*/*should be v_n*/
			v[ebNE*nQuadraturePoints_elementBoundary*nDOF_trial_element +
			  k*nDOF_trial_element +
			  j]
			*
			n[ebNE*nQuadraturePoints_elementBoundary*nSpace + 
			  k*nSpace + 
			  I];
		      Jacobian_n += 
			diffusiveVelocityComponent_I_Jacobian2_nn 
			*
			dphi_n_n[j_global]
			*
			n[ebNE*nQuadraturePoints_elementBoundary*nSpace + 
			  k*nSpace + 
			  I];
		    }/*I loop */
		 
		  if (isDOFBoundary_un[ebNE*nQuadraturePoints_elementBoundary + k] == 2)
		    {
		      /*dependency of psi_n on psi_w*/
		      Jacobian_n += 
			penalty_n[ebNE*nQuadraturePoints_elementBoundary+k]
			* /*should be v_n*/
			v[ebNE*nQuadraturePoints_elementBoundary*nDOF_trial_element+
			  k*nDOF_trial_element+
			  j]
			*
			dpsi_n_dpsiw[ebNE*nQuadraturePoints_elementBoundary + k];
		      /*dependency of psi_n on psi_c */
		      Jacobian_w += 
			penalty_n[ebNE*nQuadraturePoints_elementBoundary+k]
			* /*should be v_w*/
			v[ebNE*nQuadraturePoints_elementBoundary*nDOF_trial_element+
			  k*nDOF_trial_element+
			  j]
			*
			dpsi_n_dpsic[ebNE*nQuadraturePoints_elementBoundary + k];
		    }
		  else if (isDOFBoundary_un[ebNE*nQuadraturePoints_elementBoundary + k] == 1)
		    {
		      /*only diagonal gets penalty term*/
		      Jacobian_n += 
			penalty_n[ebNE*nQuadraturePoints_elementBoundary+k]
			* /*should be v_w*/
			v[ebNE*nQuadraturePoints_elementBoundary*nDOF_trial_element+
			  k*nDOF_trial_element+
			  j];

		    }
		  fluxJacobian_nw[ebNE*nQuadraturePoints_elementBoundary*nDOF_trial_element + 
				  k*nDOF_trial_element +
				  j] += 
		    Jacobian_w;
		  fluxJacobian_nn[ebNE*nQuadraturePoints_elementBoundary*nDOF_trial_element + 
				  k*nDOF_trial_element +
				  j] += 
		    Jacobian_n;
		}/* j local dof loop*/
	    }/*u_w dof boundary loop*/
	  
	}/*k*/
    }/*ebNE*/

}

/*end FCPP exterior fluxes*/
void calculateGlobalExteriorNumericalFluxDarcySplitPressure(proteus_index_t nExteriorElementBoundaries_global,
							    int nQuadraturePoints_elementBoundary,
							    int nSpace,
							    const proteus_index_t* exteriorElementBoundaries,
							    const proteus_index_t* elementBoundaryElements,
							    const int* elementBoundaryLocalElementBoundaries,
							    const int* isDOFBoundary_u,/*1 set bc for psi_w, 
											 2 set bc for psi_n*/
							    const double* n,
							    const double* bc_a,      
							    const double* bc_grad_phi,
							    const double* bc_psi_w,       
							    const double* bc_psi_n,
							    const double* a,         
							    const double* grad_phi,   
							    const double* psi_w,           /*psi_w*/
							    const double* psi_n,
							    const double* penalty,
							    double * diffusiveFlux)
{
  proteus_index_t ebN; int ebNE, I, J, k, nSpace2=nSpace*nSpace;
  double diffusiveFlux_I=0.0,penaltyFlux = 0.0;

  for (ebNE = 0; ebNE < nExteriorElementBoundaries_global; ebNE++)
    {
      ebN = exteriorElementBoundaries[ebNE];
      for (k = 0; k < nQuadraturePoints_elementBoundary; k++)
	{
	  diffusiveFlux[ebNE*nQuadraturePoints_elementBoundary+k] = 0.0;
	  /*1 set psi_w, 2 set psi_n*/
	  if(isDOFBoundary_u[ebNE*nQuadraturePoints_elementBoundary+k] >= 1)
	    {
	      /*integration by parts term for diffusive flux wrt phi = psi_w*/
	      for (I = 0; I < nSpace; I++)
		{
		  diffusiveFlux_I = 0.0;
		  for (J = 0; J < nSpace; J++)
		    {
		      diffusiveFlux_I -= 
			a[ebNE*nQuadraturePoints_elementBoundary*nSpace2 + 
			  k*nSpace2 +
			  I*nSpace + 
			  J]
			*
			grad_phi[ebNE*nQuadraturePoints_elementBoundary*nSpace + 
				 k*nSpace +
				 J];
		    }/*J*/
		  diffusiveFlux[ebNE*nQuadraturePoints_elementBoundary + k] += 
		    diffusiveFlux_I
		    *
		    n[ebNE*nQuadraturePoints_elementBoundary*nSpace + 
		      k*nSpace + 
		      I];
		}/*I, a_mw grad phi_n term */
	      /*boundary penalty term*/
	      if (isDOFBoundary_u[ebNE*nQuadraturePoints_elementBoundary+k] == 2)
		{
		  penaltyFlux = 
		    penalty[ebNE*nQuadraturePoints_elementBoundary + k]
		    *
		    (psi_n[ebNE*nQuadraturePoints_elementBoundary + k]
		     -
		     bc_psi_n[ebNE*nQuadraturePoints_elementBoundary + k]);
		}
	      else
		{
		  assert(isDOFBoundary_u[ebNE*nQuadraturePoints_elementBoundary+k] == 1);
		  penaltyFlux = 
		    penalty[ebNE*nQuadraturePoints_elementBoundary + k]
		    *
		    (psi_w[ebNE*nQuadraturePoints_elementBoundary + k]
		     -
		     bc_psi_w[ebNE*nQuadraturePoints_elementBoundary + k]);

		}
	      diffusiveFlux[ebNE*nQuadraturePoints_elementBoundary +k] += 
		penaltyFlux;
	    }/*um boundary*/
	}/*k*/
    }/*ebNE*/
}
void calculateGlobalExteriorNumericalFluxDarcySplitPressure_sd(proteus_index_t nExteriorElementBoundaries_global,
							       int nQuadraturePoints_elementBoundary,
							       int nSpace,
							       const proteus_csr_index_t* rowptr,
							       const proteus_csr_index_t* colind,
							       const proteus_index_t* exteriorElementBoundaries,
							       const proteus_index_t* elementBoundaryElements,
							       const int* elementBoundaryLocalElementBoundaries,
							       const int* isDOFBoundary_u,/*1 set bc for psi_w, 
											    2 set bc for psi_n*/
							       const double* n,
							       const double* bc_a,      
							       const double* bc_grad_phi,
							       const double* bc_psi_w,       
							       const double* bc_psi_n,
							       const double* a,         
							       const double* grad_phi,   
							       const double* psi_w,           /*psi_w*/
							       const double* psi_n,
							       const double* penalty,
							       double * diffusiveFlux)
{
  proteus_index_t ebN; proteus_csr_index_t m; proteus_csr_index_t nnz = rowptr[nSpace]; int ebNE, I, k;
  double diffusiveFlux_I=0.0,penaltyFlux = 0.0;

  for (ebNE = 0; ebNE < nExteriorElementBoundaries_global; ebNE++)
    {
      ebN = exteriorElementBoundaries[ebNE];
      for (k = 0; k < nQuadraturePoints_elementBoundary; k++)
	{
	  diffusiveFlux[ebNE*nQuadraturePoints_elementBoundary+k] = 0.0;
	  /*1 set psi_w, 2 set psi_n*/
	  if(isDOFBoundary_u[ebNE*nQuadraturePoints_elementBoundary+k] >= 1)
	    {
	      /*integration by parts term for diffusive flux wrt phi = psi_w*/
	      for (I = 0; I < nSpace; I++)
		{
		  diffusiveFlux_I = 0.0;
		  for (m = rowptr[I]; m < rowptr[I+1]; m++)
		    {
		      diffusiveFlux_I -= 
			a[ebNE*nQuadraturePoints_elementBoundary*nnz + 
			  k*nnz +
			  m]
			*
			grad_phi[ebNE*nQuadraturePoints_elementBoundary*nSpace + 
				 k*nSpace +
				 colind[m]];
		    }/*J*/
		  diffusiveFlux[ebNE*nQuadraturePoints_elementBoundary + k] += 
		    diffusiveFlux_I
		    *
		    n[ebNE*nQuadraturePoints_elementBoundary*nSpace + 
		      k*nSpace + 
		      I];
		}/*I, a_mw grad phi_n term */
	      /*boundary penalty term*/
	      if (isDOFBoundary_u[ebNE*nQuadraturePoints_elementBoundary+k] == 2)
		{
		  penaltyFlux = 
		    penalty[ebNE*nQuadraturePoints_elementBoundary + k]
		    *
		    (psi_n[ebNE*nQuadraturePoints_elementBoundary + k]
		     -
		     bc_psi_n[ebNE*nQuadraturePoints_elementBoundary + k]);
		}
	      else
		{
		  assert(isDOFBoundary_u[ebNE*nQuadraturePoints_elementBoundary+k] == 1);
		  penaltyFlux = 
		    penalty[ebNE*nQuadraturePoints_elementBoundary + k]
		    *
		    (psi_w[ebNE*nQuadraturePoints_elementBoundary + k]
		     -
		     bc_psi_w[ebNE*nQuadraturePoints_elementBoundary + k]);

		}
	      diffusiveFlux[ebNE*nQuadraturePoints_elementBoundary +k] += 
		penaltyFlux;
	    }/*um boundary*/
	}/*k*/
    }/*ebNE*/
}

void shallowWater_phi(double g, 
		      double h_l, double h_r, 
		      double u_l, double u_r,
                      double c_l, double c_r,
		      double h, 
		      double* phi, double* dphi, double* u_f, 
                      int* w_1, int* w_2)
{
  double phi_l,dphi_l,phi_r,dphi_r,c;
  c = sqrt(g*h);
  if(h <= h_l)
    {
      phi_l = u_l + 2.0*(c_l - c);
      dphi_l = -g/c;
      *w_1 = 2;
    }
  else
    {
      phi_l = u_l - (h - h_l)*sqrt(0.5*g*(1.0/h+1.0/h_l));
      dphi_l = -sqrt(0.5*g*(1.0/h+1.0/h_l)) - (h - h_l)*(0.5/sqrt(0.5*g*(1.0/h+1.0/h_l)))*(-0.5*g/(h*h));
      *w_1 = 1;
    }
  if(h <= h_r)
    {
      phi_r = u_r - 2.0*(c_r - c);
      dphi_r = g/c;
      *w_2 = 2;
    }
  else
    {
      phi_r = u_r + (h - h_r)*sqrt(g*(1.0/h+1.0/h_r)/2.0);
      dphi_r = sqrt(0.5*g*(1.0/h+1.0/h_r)) + (h - h_r)*(0.5/sqrt(0.5*g*(1.0/h+1.0/h_r)))*(-0.5*g/(h*h));
      *w_2 = 1;
    }
  *phi = phi_l - phi_r;
  *dphi = dphi_l - dphi_r;
  *u_f =phi_l;
}

void shallowWater_Riemann(int verbose,
                          double h_eps,double tol_u,
                          double g, 
                          double h_l, double h_r,
                          double hu_l, double hu_r,
                          double* h_G, double* u_G)
{
  int w_1,w_2;//wave types: 1=shock, 2=rarefaction, 3=intermediate dry region with wet left and right states
  double u_l,u_r,//left and right velocities
    c_l,c_r,//left and right characteristic speeds
    h_m,u_m,c_m,u_ml,u_mr,//intermediate states and characteristic speed
    phi_m,dphi_m,phi_m0,dh_m,//nonlinear function value,derivative, Newton correction
    s_1,s_2,//shock speeds
    sl_1,sl_2,sr_1,sr_2;//wave speeds at left and right edges of waves
  //compute u_l and u_r from momentum, make sure h_l and h_r non-negative, and  catch small h  relative  to hu
  if (h_l < h_eps)
    {
      h_l = 0.0;
      u_l = 0.0;
    }
  else
    u_l = hu_l/h_l;
  if (h_r < h_eps)
    {
      h_r = 0.0;
      u_r = 0.0;
    }
  else
    u_r = hu_r/h_r;
  //
  //find intermediate state
  //
  c_l = sqrt(g*h_l);
  c_r = sqrt(g*h_r);
  h_m = (1.0/(16.0*g))*pow(u_l - u_r + 2.0*(c_l + c_r),2);   //use h_m from two rarefaction solution as initial guess
  u_ml = u_l + 2.0*c_l;//at right edge of 0-rarefaction connecting left state and h=0
  u_mr = u_r - 2.0*c_r;//at left edge of 1-rarefaction connecting right state and h=0
  if (verbose)
    printf("%12.5e %12.5e %12.5e\n",h_m,h_l,h_r);
  if (h_l < h_eps || h_r < h_eps)//left or right state is dry
    {
      //initialize to rarefactions
      w_1 = 2;
      w_2 = 2;
      if (h_l < h_eps)
        w_1 = 1;//1-wave is shock
      if (h_r < h_eps)
        w_2 = 1;//2-wave is shock
      h_m = 0.0;//intermediate state is dry
      u_m = u_r + u_l - 2.0*(c_r - c_l);//connect to wet state with rarefaction or 0 if both states dry
      c_m = 0.0;
    }
  else if (u_ml <= u_mr)//intermediate state is dry
    {
      h_m = 0.0;
      u_m = 0.0;
      c_m = 0.0;
      w_1 = 3;
      w_2 = 3;
    }
  else if (h_m <= h_l && h_m <= h_r) //the solution is two rarefactions
    {
      c_m = sqrt(g*h_m);
      u_m = u_l + 2.0*(c_l - c_m);
      w_1 = 2;
      w_2 = 2;
      if (verbose)
        printf("picking 2 rarefactions %i %i %12.5e %12.5e \n",w_1,w_2,h_m,u_m);
    }
  else//calculate h_m,u_m using Newton's method
    {
      tol_u=fmin(fabs(u_l),fabs(u_r))*1.0e-8+1.0e-8;
      shallowWater_phi(g,h_l,h_r,u_l,u_r,c_l,c_r,h_m,&phi_m,&dphi_m,&u_m,&w_1,&w_2);
      phi_m0=phi_m;
      while (fabs(phi_m) > tol_u)
        {
          dh_m = -phi_m/dphi_m;
          h_m += dh_m;
          shallowWater_phi(g,h_l,h_r,u_l,u_r,c_l,c_r,h_m,&phi_m,&dphi_m,&u_m,&w_1,&w_2);
        } 
      if (verbose)
        printf("wave types from newton %i %i \n",w_1,w_2);
      c_m = sqrt(g*h_m);
    }
  //compute characteristic speeds (used to find Godunov values)
  if (w_1 == 3)//two rarefactions connecting wet states to intermediate dry state
    {
      sl_1 = u_l - c_l;
      sr_1 = u_ml - c_m;
      sl_2 = u_mr + c_m;
      sr_2 = u_r + c_r;
    }
  else
    {
      if (w_1 == 1)//1-shock
        {
          if (fabs(h_l - h_m) <= 1.0e-10)//tiny shock, use  left characteristic
            s_1 = u_l - c_l;
          else
            s_1 = (h_l * u_l - h_m * u_m)/(h_l - h_m);
          sl_1 = s_1;
          sr_1 = s_1;
        }
      else//1-rarefaction
        {
          sl_1 = u_l - c_l;
          sr_1 = u_m - c_m;
        }
      if (w_2 == 1)//2-shock
        {
          if (fabs(h_r - h_m) <= 1.0e-10)//tiny shock, use right characteristic
            s_2 = u_r + c_r;
          else
            s_2 = (h_r*u_r - h_m*u_m)/(h_r - h_m);
          sl_2 = s_2;
          sr_2 = s_2;
        }
      else//2-rarefaction
        {
          sl_2 = u_m + c_m;
          sr_2 = u_r + c_r;
        }
    }
  //compute Godunov values of h and u along x/t=0
  if (sl_1 < 0.0 && sr_1 > 0.0)//1-wave with left edge left going and right edge right going  (transonic rarefaction)
    {
      *h_G = (1.0/(9.0*g))*pow(u_l + 2.0*c_l,2);
      *u_G = u_l - 2.0*(sqrt(g*(*h_G)) - c_l);
      if (verbose)
        printf("1-wave transonic rarefaction");
    }
  else if (sl_2 < 0.0 && sr_2 > 0.0) //2-wave is transonic rarefaction
    {
      *h_G = (1.0/(9.0*g))*pow(u_r - 2.0*c_r,2);
      *u_G = u_r + 2.0*(sqrt(g*(*h_G)) - c_r);
      if (verbose)
        printf("2-wave transonic rarefaction");
    }
  else if (sl_1 > 0.0)//left edge of 1-wave is right going (shock or supersonic rarefaction)
    {
      if (verbose)
        printf("1-wave supersonic");
      *h_G = h_l;
      *u_G = u_l;
    }
  else if (sr_2 < 0.0)//right edge of 2-wave is left  going (shock or supersonic rarefaction)
    {
      if (verbose)
        printf("2-wave supersonic");
      *h_G = h_r;
      *u_G = u_r;
    }
  else//right edge of 1-wave is left going and left edge of 2-wave  is right going so use the intermediate state
    {
      if (verbose)
        printf("intermediate state");
      *h_G = h_m;
      *u_G = u_m;
    }
  if (verbose)
    printf("%12.5e %12.5e %12.5e %12.5e\n",sl_1,sr_1,sl_2,sr_2);
}

void calculateInteriorNumericalFluxShallowWater_1D(proteus_index_t nInteriorElementBoundaries_global,
						   int nElementBoundaries_element,
						   int nQuadraturePoints_elementBoundary,
						   double h_eps,
                                                   double tol_u,
                                                   double g,
						   proteus_index_t* interiorElementBoundaries,
						   proteus_index_t* elementBoundaryElements,
						   int* elementBoundaryLocalElementBoundaries,
						   double* n,
						   double* h,
						   double* hu,
						   double* flux_h,
						   double* flux_hu)
{
  proteus_index_t ebN, left_eN_global, right_eN_global; int ebNI, left_ebN_element, right_ebN_element, k, verbose=0;
  double h_l,h_r,hu_l,hu_r,h_G,u_G,n_lr;
  for(ebNI=0;ebNI<nInteriorElementBoundaries_global;ebNI++)
    {
      ebN = interiorElementBoundaries[ebNI];
      left_eN_global = elementBoundaryElements[ebN*2+0];
      right_eN_global = elementBoundaryElements[ebN*2+1];
      left_ebN_element = elementBoundaryLocalElementBoundaries[ebN*2+0];
      right_ebN_element = elementBoundaryLocalElementBoundaries[ebN*2+1];
      for(k=0;k<nQuadraturePoints_elementBoundary;k++)
        {
	  n_lr = n[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary+
		   left_ebN_element*nQuadraturePoints_elementBoundary+
		   k+
		   0];
	  h_l = fmax(0.0,h[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary+
			   left_ebN_element*nQuadraturePoints_elementBoundary+
			   k]);
	  h_r = fmax(0.0,h[right_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary+
			   right_ebN_element*nQuadraturePoints_elementBoundary+
			   k]);
          hu_l = hu[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary+
                   left_ebN_element*nQuadraturePoints_elementBoundary+
                    k]*n_lr;
          hu_r = hu[right_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary+
                   right_ebN_element*nQuadraturePoints_elementBoundary+
                   k]*n_lr;
          shallowWater_Riemann(verbose,h_eps,tol_u,g,h_l,h_r,hu_l,hu_r,&h_G,&u_G);
          u_G *= n_lr;
	  flux_h[ebN*nQuadraturePoints_elementBoundary+
                 k] = h_G*u_G*n_lr;
	  flux_hu[ebN*nQuadraturePoints_elementBoundary+
		  k] = (h_G*u_G*u_G + 0.5*g*h_G*h_G)*n_lr;
        }/*k*/
    }/*ebnI*/
}
void calculateExteriorNumericalFluxShallowWater_1D(proteus_index_t nExteriorElementBoundaries_global,
						   int nQuadraturePoints_elementBoundary,
                                                   double h_eps,
                                                   double tol_u,
						   double g,
						   double* n,
						   double* h_lv,
						   double* hu_lv,
						   double* h_rv,
						   double* hu_rv,
						   double* flux_h,
						   double* flux_hu)
{
  int ebNE,k,verbose=0;
  double h_l,h_r,hu_l,hu_r,h_G,u_G,n_lr;
  for(ebNE=0;ebNE<nExteriorElementBoundaries_global;ebNE++)
    {
      for(k=0;k<nQuadraturePoints_elementBoundary;k++)
        {
	  n_lr = n[ebNE*nQuadraturePoints_elementBoundary+
		   k+
		   0];
	  h_l = h_lv[ebNE*nQuadraturePoints_elementBoundary+
		     k];
	  h_r = h_rv[ebNE*nQuadraturePoints_elementBoundary+
		     k];
          hu_l = hu_lv[ebNE*nQuadraturePoints_elementBoundary+
                       k]*n_lr;
          hu_r = hu_rv[ebNE*nQuadraturePoints_elementBoundary+
                       k]*n_lr;
          shallowWater_Riemann(verbose,h_eps,tol_u,g,h_l,h_r,hu_l, hu_r,&h_G,&u_G);
          u_G *= n_lr;
	  flux_h[ebNE*nQuadraturePoints_elementBoundary+
		 k] = h_G*u_G*n_lr;
	  flux_hu[ebNE*nQuadraturePoints_elementBoundary+
		  k] = (h_G*u_G*u_G + 0.5*g*h_G*h_G)*n_lr;
        }/*k*/
    }/*ebnE*/
}

void calculateInteriorNumericalFluxShallowWater_2D(proteus_index_t nInteriorElementBoundaries_global,
						   int nElementBoundaries_element,
						   int nQuadraturePoints_elementBoundary,
						   double h_eps,
                                                   double tol_u,
                                                   double g,
						   proteus_index_t* interiorElementBoundaries,
						   proteus_index_t* elementBoundaryElements,
						   int* elementBoundaryLocalElementBoundaries,
						   double* n,
						   double* h,
						   double* hu,
						   double* hv,
						   double* flux_h,
						   double* flux_hu,
						   double* flux_hv)
{
  proteus_index_t ebN, left_eN_global, right_eN_global; int ebNI, left_ebN_element, right_ebN_element, k, verbose=0;
  double h_l,h_r,hu_l,hu_r,hv_l,hv_r,h_G,u_G,v_G,nx_lr,ny_lr,hVn_l,hVn_r,hVt_l,hVt_r,Vn_G,Vt_G,hVt_G;
  for(ebNI=0;ebNI<nInteriorElementBoundaries_global;ebNI++)
    {
      ebN = interiorElementBoundaries[ebNI];
      left_eN_global = elementBoundaryElements[ebN*2+0];
      right_eN_global = elementBoundaryElements[ebN*2+1];
      left_ebN_element = elementBoundaryLocalElementBoundaries[ebN*2+0];
      right_ebN_element = elementBoundaryLocalElementBoundaries[ebN*2+1];
      for(k=0;k<nQuadraturePoints_elementBoundary;k++)
        {
	  nx_lr = n[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*2+
		    left_ebN_element*nQuadraturePoints_elementBoundary*2+
		    k*2+
		    0];
	  ny_lr = n[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*2+
		    left_ebN_element*nQuadraturePoints_elementBoundary*2+
		    k*2+
		    1];
	  h_l = h[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary+
		  left_ebN_element*nQuadraturePoints_elementBoundary+
		  k];
	  h_r = h[right_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary+
		  right_ebN_element*nQuadraturePoints_elementBoundary+
		  k];
	  hu_l = hu[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary+
                    left_ebN_element*nQuadraturePoints_elementBoundary+
                    k];
	  hu_r = hu[right_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary+
                    right_ebN_element*nQuadraturePoints_elementBoundary+
                    k];
	  hv_l = hv[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary+
                    left_ebN_element*nQuadraturePoints_elementBoundary+
                    k];
	  hv_r = hv[right_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary+
                    right_ebN_element*nQuadraturePoints_elementBoundary+
                    k];
	  //project u and v onto interface normal and tangential components
	  hVn_l = nx_lr*hu_l + ny_lr*hv_l;
	  hVt_l = ny_lr*hu_l - nx_lr*hv_l;
	  hVn_r = nx_lr*hu_r + ny_lr*hv_r;
	  hVt_r = ny_lr*hu_r - nx_lr*hv_r;
          shallowWater_Riemann(verbose,h_eps,tol_u,g,h_l,h_r,hVn_l, hVn_r,&h_G,&Vn_G);
          if (Vn_G > 0.0)
	    hVt_G = hVt_l;
	  else
	    hVt_G = hVt_r;
          if (h_G < h_eps)
            {
              Vt_G = 0.0;
            }
          else
            {
              Vt_G = hVt_G/h_G;
            }
	  u_G = Vn_G*nx_lr + Vt_G*ny_lr;
	  v_G = Vn_G*ny_lr - Vt_G*nx_lr;
	  //calculate fluxes
	  flux_h[ebN*nQuadraturePoints_elementBoundary+
		 k] = h_G*u_G*nx_lr + h_G*v_G*ny_lr;
	  flux_hu[ebN*nQuadraturePoints_elementBoundary+
		 k] = (h_G*u_G*u_G + 0.5*g*h_G*h_G)*nx_lr + h_G*u_G*v_G*ny_lr;
	  flux_hv[ebN*nQuadraturePoints_elementBoundary+
                  k] = h_G*u_G*v_G*nx_lr + (h_G*v_G*v_G + 0.5*g*h_G*h_G)*ny_lr;
        }/*k*/
    }/*ebnI*/
}

void calculateExteriorNumericalFluxShallowWater_2D(proteus_index_t nExteriorElementBoundaries_global,
						   int nQuadraturePoints_elementBoundary,
                                                   double h_eps,
                                                   double tol_u,
						   double g,
						   double* n,
						   double* h_lq,
						   double* hu_lq,
						   double* hv_lq,
						   double* h_rq,
						   double* hu_rq,
						   double* hv_rq,
						   double* flux_h,
						   double* flux_hu,
						   double* flux_hv)
{
  int ebNE,k,verbose=0;
  double h_l,h_r,hu_l,hu_r,hv_l,hv_r,h_G,u_G,v_G,nx_lr,ny_lr,hVn_l,hVn_r,hVt_l,hVt_r,Vn_G,Vt_G,hVt_G;
  for(ebNE=0;ebNE<nExteriorElementBoundaries_global;ebNE++)
    {
      for(k=0;k<nQuadraturePoints_elementBoundary;k++)
        {
	  nx_lr = n[ebNE*nQuadraturePoints_elementBoundary*2+
		    k*2+
		    0];
	  ny_lr = n[ebNE*nQuadraturePoints_elementBoundary*2+
		    k*2+
		    1];
	  h_l = h_lq[ebNE*nQuadraturePoints_elementBoundary+
		     k];
	  h_r = h_rq[ebNE*nQuadraturePoints_elementBoundary+
		     k];
	  hu_l = hu_lq[ebNE*nQuadraturePoints_elementBoundary+
                       k];
	  hu_r = hu_rq[ebNE*nQuadraturePoints_elementBoundary+
                       k];
	  hv_l = hv_lq[ebNE*nQuadraturePoints_elementBoundary+
                       k];
	  hv_r = hv_rq[ebNE*nQuadraturePoints_elementBoundary+
                       k];
	  //project hu and hv onto interface normal and tangential components
	  hVn_l = nx_lr*hu_l + ny_lr*hv_l;
	  hVt_l = ny_lr*hu_l - nx_lr*hv_l;
	  hVn_r = nx_lr*hu_r + ny_lr*hv_r;
	  hVt_r = ny_lr*hu_r - nx_lr*hv_r;
          shallowWater_Riemann(verbose,h_eps,tol_u,g,h_l,h_r,hVn_l,hVn_r,&h_G,&Vn_G);
          if (Vn_G > 0.0)
	    hVt_G = hVt_l;
	  else
	    hVt_G = hVt_r;
          if (h_G < h_eps)
            {
              Vt_G = 0.0;
            }
          else
            {
              Vt_G = hVt_G/h_G;
            }
	  u_G = Vn_G*nx_lr + Vt_G*ny_lr;
	  v_G = Vn_G*ny_lr - Vt_G*nx_lr;
	  flux_h[ebNE*nQuadraturePoints_elementBoundary+
		 k] = h_G*u_G*nx_lr + h_G*v_G*ny_lr;
	  flux_hu[ebNE*nQuadraturePoints_elementBoundary+
                  k] = (h_G*u_G*u_G + 0.5*g*h_G*h_G)*nx_lr + h_G*u_G*v_G*ny_lr;
	  flux_hv[ebNE*nQuadraturePoints_elementBoundary+
                  k] = h_G*u_G*v_G*nx_lr + (h_G*v_G*v_G + 0.5*g*h_G*h_G)*ny_lr;
        }/*k*/
    }/*ebnE*/
}
void calculateInteriorNumericalFluxShallowWaterHLL_1D(proteus_index_t nInteriorElementBoundaries_global,
						      int nElementBoundaries_element,
						      int nQuadraturePoints_elementBoundary,
						      double h_eps,
						      double tol_u,
						      double g,
						      proteus_index_t* interiorElementBoundaries,
						      proteus_index_t* elementBoundaryElements,
						      int* elementBoundaryLocalElementBoundaries,
						      double* n,
						      double* h,
						      double* hu,
						      double* flux_h,
						      double* flux_hu)
{
  proteus_index_t ebN, left_eN_global, right_eN_global; int ebNI, left_ebN_element, right_ebN_element, k, verbose=0;
  double h_l,h_r,hu_l,hu_r,u_l,u_r,n_lr,h_G,u_G;
  
  double h_Roe,u_Roe,/*avg h and u for calculating eigenvalues for Roe matrix,*/
    lambda_1L,lambda_2R,/*1 family wave for left state, 2 family wave for right state*/
    lambda_1Roe,lambda_2Roe,/*1 and 2 wave eigen values for Roe matrix*/
    lambda_min,lambda_max,/*upper and lower bound for eigenvalues*/
    h_LR,hu_LR;/*intermediate state 
	   u_LR = \lambda_max U_R -lambda_min U_L / (lambda_max - lambda_min)
                  - (f(U_r)-f(U_L)/(lambda_max-lambda_min)
	 */
  /*
    flux is
    f_HLL(U_L,U_R) = \frac{\lambda_max^- - \lambda_min^-}{\lambda_max-\lambda_min} f(U_R)
                    +\frac{\lambda_max^+ - \lambda_min^+}{\lambda_max-\lambda_min} f(U_L)
		    -\frac{1}{2}\frac{\lambda_max|\lambda_min| - \lambda_min|\lambda_max|}{\lambda_max-\lambda_min}(U_R-U_L)
   */
  double lambda_diff;
  for(ebNI=0;ebNI<nInteriorElementBoundaries_global;ebNI++)
    {
      ebN = interiorElementBoundaries[ebNI];
      left_eN_global = elementBoundaryElements[ebN*2+0];
      right_eN_global = elementBoundaryElements[ebN*2+1];
      left_ebN_element = elementBoundaryLocalElementBoundaries[ebN*2+0];
      right_ebN_element = elementBoundaryLocalElementBoundaries[ebN*2+1];
      for(k=0;k<nQuadraturePoints_elementBoundary;k++)
        {
	  n_lr = n[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary+
		   left_ebN_element*nQuadraturePoints_elementBoundary+
		   k+
		   0];
	  h_l = fmax(0.0,h[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary+
			   left_ebN_element*nQuadraturePoints_elementBoundary+
			   k]);
	  h_r = fmax(0.0,h[right_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary+
			   right_ebN_element*nQuadraturePoints_elementBoundary+
			   k]);
          hu_l = hu[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary+
                   left_ebN_element*nQuadraturePoints_elementBoundary+
                    k];
          hu_r = hu[right_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary+
                   right_ebN_element*nQuadraturePoints_elementBoundary+
                   k];
	  u_l = 0.0; u_r =0.0; u_Roe = 0.0; 
	  if (h_l > h_eps) 
	    u_l = n_lr*hu_l/h_l; 
	  else
	    h_l = 0.0;
	  if (h_r > h_eps) 
	    u_r = n_lr*hu_r/h_r;
	  else
	    h_r = 0.0;
	  /*Roe averages*/
	  h_Roe = 0.5*(h_l+h_r);
	  if (h_l + h_r > h_eps)
	    u_Roe = (sqrt(h_l)*u_l + sqrt(h_r)*u_r)/(sqrt(h_l)+sqrt(h_r));
	  /*Roe matrix speeds*/
	  lambda_1Roe = u_Roe - sqrt(h_Roe*g); lambda_2Roe = u_Roe + sqrt(h_Roe*g);
	  /*slowest/fastest eigenvalue at left/right state*/
	  lambda_1L = u_l - sqrt(g*h_l); lambda_2R = u_r + sqrt(g*h_r);
	  /*lower/upper bound on speeds*/
	  lambda_min = fmin(lambda_1L,lambda_1Roe);
	  lambda_max = fmax(lambda_2R,lambda_2Roe);

	  /*mwf debug
	    shallowWater_Riemann(verbose,h_eps,tol_u,g,h_l,h_r,hu_l*n_lr,hu_r*n_lr,&h_G,&u_G);
	    u_G *= n_lr;
	  */
	  
	  if (lambda_min >= 0.0)
	    {
	      /*left state travels at zero speed*/
	      flux_h[ebN*nQuadraturePoints_elementBoundary+
		     k] = h_l*u_l*n_lr;/*hu_l*n_lr;*/
	      flux_hu[ebN*nQuadraturePoints_elementBoundary+
		      k] = (h_l*u_l*u_l + 0.5*g*h_l*h_l)*n_lr;
	      
	      /*mwf debug
	      if (fabs(flux_h[ebN*nQuadraturePoints_elementBoundary+k]-h_G*u_G*n_lr) > 1.0e-4 ||
		  fabs(flux_hu[ebN*nQuadraturePoints_elementBoundary+k]-(h_G*u_G*u_G + 0.5*g*h_G*h_G)*n_lr) > 1.0e-4)
		{
		  printf("SW HLL ebN=%d k=%d h_l=%g h_r=%g hu_l=%g hu_r=%g n_lr=%g u_l= %g u_r=%g  h_Roe=%g u_Roe=%g\n",ebN,k,h_l,h_r,hu_l,hu_r,n_lr,u_l,u_r,h_Roe,u_Roe);
		  printf("\t lambda_1Roe=%g lambda_2Roe=%g lambda_1L=%g lambda_2R=%g lambda_min=%g lambda_max=%g\n",
			 lambda_1Roe,lambda_2Roe,lambda_1L,lambda_2R,lambda_min,lambda_max);
		  printf("\t left state chosen flux_h= %g flux_hu= %g \n",
			 flux_h[ebN*nQuadraturePoints_elementBoundary+k],
			 flux_hu[ebN*nQuadraturePoints_elementBoundary+k]);
		  printf("\t result from exact solve h_G= %g u_G=%g flux_h= %g flux_hu= %g \n",
			 h_G,u_G,h_G*u_G*n_lr,(h_G*u_G*u_G + 0.5*g*h_G*h_G)*n_lr);
		}
	      */
	    }
	  else if (lambda_max <= 0.0)
	    {
	      /*right state travels at zero speed*/
	      flux_h[ebN*nQuadraturePoints_elementBoundary+
		     k] = h_r*u_r*n_lr;/*hu_r*n_lr;*/
	      flux_hu[ebN*nQuadraturePoints_elementBoundary+
		      k] = (h_r*u_r*u_r + 0.5*g*h_r*h_r)*n_lr;

	      /*mwf debug
	      if (fabs(flux_h[ebN*nQuadraturePoints_elementBoundary+k]-h_G*u_G*n_lr) > 1.0e-4 ||
		  fabs(flux_hu[ebN*nQuadraturePoints_elementBoundary+k]-(h_G*u_G*u_G + 0.5*g*h_G*h_G)*n_lr) > 1.0e-4)
		{
		  printf("SW HLL ebN=%d k=%d h_l=%g h_r=%g hu_l=%g hu_r=%g n_lr=%g u_l= %g u_r=%g  h_Roe=%g u_Roe=%g\n",ebN,k,h_l,h_r,hu_l,hu_r,n_lr,u_l,u_r,h_Roe,u_Roe);
		  printf("\t lambda_1Roe=%g lambda_2Roe=%g lambda_1L=%g lambda_2R=%g lambda_min=%g lambda_max=%g\n",
			 lambda_1Roe,lambda_2Roe,lambda_1L,lambda_2R,lambda_min,lambda_max);
		  printf("\t right state chosen flux_h= %g flux_hu= %g \n",
			 flux_h[ebN*nQuadraturePoints_elementBoundary+k],
			 flux_hu[ebN*nQuadraturePoints_elementBoundary+k]);
		  printf("\t result from exact solve h_G= %g u_G=%g flux_h= %g flux_hu= %g \n",
			 h_G,u_G,h_G*u_G*n_lr,(h_G*u_G*u_G + 0.5*g*h_G*h_G)*n_lr);
		}
	      */
	    }
	  else /*intermediate state travels with zero speed*/
	    {
	      lambda_diff = lambda_max-lambda_min;
	      assert(fabs(lambda_diff) > 0.0);
	      /*intermediate state chosen to enforce conservation of Riemann solution
		equivalent to decomposing flux jump as piecewise linear function
		f(U_R)-f(U_L) = \lambda_max(U_R-U_LR) + \lambda_min(U_LR-U_L)
	      */
	      h_LR = (lambda_max*h_r - lambda_min*h_l
		      -n_lr*(h_r*u_r-h_l*u_l))/lambda_diff;
	      assert(h_LR >= 0.0);
	      hu_LR= (lambda_max*h_r*u_r - lambda_min*h_l*u_l
		      -n_lr*(h_r*u_r*u_r + 0.5*g*h_r*h_r - h_l*u_l*u_l - 0.5*g*h_l*h_l))/lambda_diff;

	      flux_h[ebN*nQuadraturePoints_elementBoundary+k] =
		(n_lr*(lambda_max*h_l*u_l-lambda_min*h_r*u_r) + (h_r-h_l)*lambda_max*lambda_min)/lambda_diff;
	      flux_hu[ebN*nQuadraturePoints_elementBoundary+k] =
		(n_lr*(lambda_max*(h_l*u_l*u_l + 0.5*g*h_l*h_l)-lambda_min*(h_r*u_r*u_r + 0.5*g*h_r*h_r)) + (h_r*u_r-h_l*u_l)*lambda_max*lambda_min)/lambda_diff;

/* 	      flux_h[ebN*nQuadraturePoints_elementBoundary+k] =  */
/* 		(n_lr*(lambda_max*hu_l-lambda_min*hu_r) + (h_r-h_l)*lambda_max*lambda_min)/lambda_diff; */
/* 	      flux_hu[ebN*nQuadraturePoints_elementBoundary+k] =  */
/* 		(n_lr*(lambda_max*(h_l*u_l*u_l + 0.5*g*h_l*h_l)-lambda_min*(h_r*u_r*u_r + 0.5*g*h_r*h_r)) + (hu_r-hu_l)*lambda_max*lambda_min)/lambda_diff; */

/* 	      flux_h[ebN*nQuadraturePoints_elementBoundary+ */
/* 		     k] = */
/* 		h_l*u_l*n_lr + lambda_min*(h_LR-h_l); */
	      
/* 	      flux_hu[ebN*nQuadraturePoints_elementBoundary+ */
/* 		      k] = */
/* 		(h_l*u_l*u_l + 0.5*g*h_l*h_l)*n_lr + lambda_min*(hu_LR-h_l*u_l); */


/* 	      if (h_LR < 1.0e-3) */
/* 		{ */
/* 		  /\*mwf hack*\/ */
/* 		  flux_h[ebN*nQuadraturePoints_elementBoundary+ */
/* 			 k] = */
/* 		    h_G*u_G*n_lr; */
	      
/* 		  flux_hu[ebN*nQuadraturePoints_elementBoundary+ */
/* 			  k] = */
/* 		    (h_G*u_G*u_G + 0.5*g*h_G*h_G)*n_lr; */
	    
/* 		} */
	      /*mwf debug
	      if (fabs(flux_h[ebN*nQuadraturePoints_elementBoundary+k]-h_G*u_G*n_lr) > 1.0e-4 ||
		  fabs(flux_hu[ebN*nQuadraturePoints_elementBoundary+k]-(h_G*u_G*u_G + 0.5*g*h_G*h_G)*n_lr) > 1.0e-4)
		{
		  printf("SW HLL ebN=%d k=%d h_l=%g h_r=%g hu_l=%g hu_r=%g n_lr=%g u_l= %g u_r=%g  h_Roe=%g u_Roe=%g\n",ebN,k,h_l,h_r,hu_l,hu_r,n_lr,u_l,u_r,h_Roe,u_Roe);
		  printf("\t lambda_1Roe=%g lambda_2Roe=%g lambda_1L=%g lambda_2R=%g lambda_min=%g lambda_max=%g\n",
			 lambda_1Roe,lambda_2Roe,lambda_1L,lambda_2R,lambda_min,lambda_max);
		  printf("\t intermediate state chosen h_LR= %g hu_LR=%g flux_h= %g flux_hu= %g \n",
			 h_LR,hu_LR,
			 flux_h[ebN*nQuadraturePoints_elementBoundary+k],
			 flux_hu[ebN*nQuadraturePoints_elementBoundary+k]);
		  printf("\t result from exact solve h_G= %g u_G=%g flux_h= %g flux_hu= %g \n",
			 h_G,u_G,h_G*u_G*n_lr,(h_G*u_G*u_G + 0.5*g*h_G*h_G)*n_lr);
		}
	      */
	    }
        }/*k*/
    }/*ebnI*/
}
void calculateExteriorNumericalFluxShallowWaterHLL_1D(proteus_index_t nExteriorElementBoundaries_global,
						      int nQuadraturePoints_elementBoundary,
						      double h_eps,
						      double tol_u,
						      double g,
						      double* n,
						      double* h_lv,
						      double* hu_lv,
						      double* h_rv,
						      double* hu_rv,
						      double* flux_h,
						      double* flux_hu)
{
  int ebNE,k,verbose=0;
  double h_l,h_r,hu_l,hu_r,u_l,u_r,n_lr;
  double h_Roe,u_Roe,/*avg h and u for calculating eigenvalues for Roe matrix,*/
    lambda_1L,lambda_2R,/*1 family wave for left state, 2 family wave for right state*/
    lambda_1Roe,lambda_2Roe,/*1 and 2 wave eigen values for Roe matrix*/
    lambda_min,lambda_max,/*upper and lower bound for eigenvalues*/
    h_LR,hu_LR;/*intermediate state 
	   u_LR = \lambda_max U_R -lambda_min U_L / (lambda_max - lambda_min)
                  - (f(U_r)-f(U_L)/(lambda_max-lambda_min)
	 */
  /*
    flux is
    f_HLL(U_L,U_R) = \frac{\lambda_max^- - \lambda_min^-}{\lambda_max-\lambda_min} f(U_R)
                    +\frac{\lambda_max^+ - \lambda_min^+}{\lambda_max-\lambda_min} f(U_L)
		    -\frac{1}{2}\frac{\lambda_max|\lambda_min| - \lambda_min|\lambda_max|}{\lambda_max-\lambda_min}(U_R-U_L)
   */
  double lambda_min_p,lambda_max_p,lambda_min_m,lambda_max_m,lambda_diff,pos,neg,mid;
  for(ebNE=0;ebNE<nExteriorElementBoundaries_global;ebNE++)
    {
      for(k=0;k<nQuadraturePoints_elementBoundary;k++)
        {
	  n_lr = n[ebNE*nQuadraturePoints_elementBoundary+
		   k+
		   0];
	  h_l = fmax(0.0,h_lv[ebNE*nQuadraturePoints_elementBoundary+
			      k]);
	  h_r = fmax(0.0,h_rv[ebNE*nQuadraturePoints_elementBoundary+
			      k]);
          hu_l = hu_lv[ebNE*nQuadraturePoints_elementBoundary+
                       k];
          hu_r = hu_rv[ebNE*nQuadraturePoints_elementBoundary+
                       k];

	  u_l = 0.0; u_r =0.0; u_Roe = 0.0; 
	  if (h_l > h_eps) 
	    u_l = n_lr*hu_l/h_l;
	  else
	    h_l = 0.0;
	  if (h_r > h_eps) 
	    u_r = n_lr*hu_r/h_r;
	  else
	    h_r = 0.0;
	  /*Roe averages*/
	  h_Roe = 0.5*(h_l+h_r);
	  if (h_l + h_r > h_eps)
	    u_Roe = (sqrt(h_l)*u_l + sqrt(h_r)*u_r)/(sqrt(h_l)+sqrt(h_r));
	  /*Roe matrix speeds*/
	  lambda_1Roe = u_Roe - sqrt(h_Roe*g); lambda_2Roe = u_Roe + sqrt(h_Roe*g);
	  /*slowest/fastest eigenvalue at left/right state*/
	  lambda_1L = u_l - sqrt(g*h_l); lambda_2R = u_r + sqrt(g*h_r);
	  /*lower/upper bound on speeds*/
	  lambda_min = fmin(lambda_1L,lambda_1Roe);
	  lambda_max = fmax(lambda_2R,lambda_2Roe);


	  if (lambda_min >= 0.0)
	    {
	      /*left state travels at zero speed*/
	      flux_h[ebNE*nQuadraturePoints_elementBoundary+
		     k] = h_l*u_l*n_lr;/*hu_l*n_lr;*/
	      flux_hu[ebNE*nQuadraturePoints_elementBoundary+
		      k] = (h_l*u_l*u_l + 0.5*g*h_l*h_l)*n_lr;


	    }
	  else if (lambda_max <= 0.0)
	    {
	      /*right state travels at zero speed*/
	      flux_h[ebNE*nQuadraturePoints_elementBoundary+
		     k] = h_r*u_r*n_lr;/*hu_r*n_lr*/;
	      flux_hu[ebNE*nQuadraturePoints_elementBoundary+
		      k] = (h_r*u_r*u_r + 0.5*g*h_r*h_r)*n_lr;

	    }

	  else /*intermediate state travels with zero speed*/
	    {
	      lambda_diff = lambda_max-lambda_min;
	      assert(fabs(lambda_diff) > 0.0);
	      /*intermediate state chosen to enforce conservation of Riemann solution
		equivalent to decomposing flux jump as piecewise linear function
		f(U_R)-f(U_L) = \lambda_max(U_R-U_LR) + \lambda_min(U_LR-U_L)
	      */
	      h_LR = (lambda_max*h_r - lambda_min*h_l
		      -n_lr*(h_r*u_r-h_l*u_l))/lambda_diff;
	      hu_LR= (lambda_max*hu_r - lambda_min*hu_l
		      -n_lr*(h_r*u_r*u_r + 0.5*g*h_r*h_r - h_l*u_l*u_l - 0.5*g*h_l*h_l))/lambda_diff;
	      
	      if (h_LR < h_eps)
		{
		  h_LR = 0.0; hu_LR = 0.0;
		}
/* 	      flux_h[ebN*nQuadraturePoints_elementBoundary+k] =  */
/* 		(n_lr*(lambda_max*h_l*u_l-lambda_min*h_r*u_r) + (h_r-h_l)*lambda_max*lambda_min)/lambda_diff; */
/* 	      flux_hu[ebN*nQuadraturePoints_elementBoundary+k] =  */
/* 		(n_lr*(lambda_max*(h_l*u_l*u_l + 0.5*g*h_l*h_l)-lambda_min*(h_r*u_r*u_r + 0.5*g*h_r*h_r)) + (h_r*u_r-h_l*u_l)*lambda_max*lambda_min)/lambda_diff; */

	      flux_h[ebNE*nQuadraturePoints_elementBoundary+k] = 
		(n_lr*(lambda_max*hu_l-lambda_min*hu_r) + (h_r-h_l)*lambda_max*lambda_min)/lambda_diff;
	      flux_hu[ebNE*nQuadraturePoints_elementBoundary+k] = 
		(n_lr*(lambda_max*(h_l*u_l*u_l + 0.5*g*h_l*h_l)-lambda_min*(h_r*u_r*u_r + 0.5*g*h_r*h_r)) + (hu_r-hu_l)*lambda_max*lambda_min)/lambda_diff;
/* 	      if (h_LR < h_eps) */
/* 		{ */
/* 		  h_LR = 0.0; hu_LR = 0.0;  */
/* 		} */
/* 	      flux_h[ebNE*nQuadraturePoints_elementBoundary+ */
/* 		     k] = */
/* 		h_l*u_l*n_lr + lambda_min*(h_LR-h_l); */

/* 	      flux_hu[ebNE*nQuadraturePoints_elementBoundary+ */
/* 		      k] = */
/* 		(h_l*u_l*u_l + 0.5*g*h_l*h_l)*n_lr + lambda_min*(hu_LR-h_l*u_l); */
	      
	    }

        }/*k*/
    }/*ebnE*/
}


void calculateInteriorChengShuNumericalFlux(proteus_index_t nInteriorElementBoundaries_global,
					    int nElementBoundaries_element,
					    int nQuadraturePoints_elementBoundary,
					    int nQuadraturePoints_element,
					    int nSpace,
					    int speedEvalFlag,
					    proteus_index_t* interiorElementBoundaries,
					    proteus_index_t* elementBoundaryElements,
					    int* elementBoundaryLocalElementBoundaries,
					    double* n,
					    double* u,
					    double* H,
					    double* dH,
					    double* H_element,
					    double* dH_element,
					    double* flux,
					    double* dflux_left,
					    double* dflux_right)
{
  proteus_index_t ebN, left_eN_global, right_eN_global; int ebNI, left_ebN_element, right_ebN_element, k, J;
  double left_flux,right_flux,u_left,u_right,left_speed,right_speed,
    tmp_left,tmp_right,minSpeed_element,maxSpeed_element,minSpeed,
    maxSpeed;
  /*for now use outer normal at first quadrature point for element speed calculations*/
  
  for(ebNI=0;ebNI<nInteriorElementBoundaries_global;ebNI++)
    {
      ebN = interiorElementBoundaries[ebNI];
      left_eN_global = elementBoundaryElements[ebN*2+0];
      right_eN_global = elementBoundaryElements[ebN*2+1];
      left_ebN_element = elementBoundaryLocalElementBoundaries[ebN*2+0];
      right_ebN_element = elementBoundaryLocalElementBoundaries[ebN*2+1];
      /*
	
	dH_{L/R} = dH_{eN_left/eN_right} . n_{eN_left}
	
	first cut, take min over interior points as well as traces
	dH_min = min_{\Omega_{L} \cup \Omega_{R}}(dH_L,dH_R), 
        dH_max = max_{\Omega_{L} \cup \Omega_{R}}(dH_L,dH_R)

	if dH_min  < 0
           flux_eN_left = |dH_min| (u^{L}- u^{R})
        else
           flux_eN_left = 0
        if dH_max  > 0 
           flux_eN_right = |dH_max| (u^{R}-u^{L})
	else
           flux_eN_right = 0
      */
      minSpeed_element=0.0; maxSpeed_element=0.0;
      for(k=0; k < nQuadraturePoints_element; k++)
	{
	  tmp_left = 0.0; tmp_right = 0.0;
	  /*evaluate df at interior elemnent point but compute its value 
	    dotted with normal for speed
	    assume normal constant over face*/
	  /*compute speed relative to left normal*/
          for(J=0;J<nSpace;J++)
            {
              tmp_left 
                += 
                n[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                  left_ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                  0*nSpace+
		  J]
                *
                dH_element[left_eN_global*nQuadraturePoints_element*nSpace+
			   k*nSpace+
			   J];
              tmp_right 
                += 
                n[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                  left_ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                  0*nSpace+
		  J]
                *
                dH_element[right_eN_global*nQuadraturePoints_element*nSpace+
			   k*nSpace+
			   J];
	    }
	  if (tmp_left < minSpeed_element || k == 0)
	    minSpeed_element = tmp_left;
	  if (tmp_right < minSpeed_element)
	    minSpeed_element = tmp_right;
	  if (tmp_right > maxSpeed_element || k == 0)
	    maxSpeed_element = tmp_right;
	  if (tmp_left > maxSpeed_element)
	    maxSpeed_element = tmp_left;
	}/*element loop*/
      /*now repeat min/max over quadrature points on boundary*/
      minSpeed = minSpeed_element; maxSpeed = maxSpeed_element;
      for(k=0;k<nQuadraturePoints_elementBoundary;k++)
        {
	  tmp_left = 0.0; 
	  tmp_right = 0.0;
	  /*compute speed relative to left normal*/
          for(J=0;J<nSpace;J++)
            {
              tmp_left 
                += 
                n[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                  left_ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                  k*nSpace+
		  J]
                *
                dH[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                   left_ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                   k*nSpace+
		   J];
              tmp_right 
                += 
                n[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                  left_ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                  k*nSpace+
		  J]
                *
                dH[right_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                   right_ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                   k*nSpace+
		   J];
	    }
	  if (tmp_left < minSpeed)
	    minSpeed = tmp_left;
	  if (tmp_right < minSpeed)
	    minSpeed = tmp_right;
	  if (tmp_right > maxSpeed)
	    maxSpeed = tmp_right;
	  if (tmp_left > maxSpeed)
	    maxSpeed = tmp_left;
	}/*first k loop*/
      /*now set penalty/flux terms*/
      for(k=0;k<nQuadraturePoints_elementBoundary;k++)
        {
	  /*now recompute and try rusanov style as hack*/
	  tmp_left = 0.0; 
	  tmp_right = 0.0;
	  /*compute speed relative to left normal*/
          for(J=0;J<nSpace;J++)
            {
              tmp_left 
                += 
                n[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                  left_ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                  k*nSpace+
		  J]
                *
                dH[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                   left_ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                   k*nSpace+
		   J];
              tmp_right 
                += 
                n[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                  left_ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                  k*nSpace+
		  J]
                *
                dH[right_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                   right_ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                   k*nSpace+
		   J];
	    }
	  /*original*/
	  left_speed = minSpeed; right_speed = maxSpeed; 
	  if (speedEvalFlag == 1)
	    {
	      left_speed = 0.5*(tmp_left + minSpeed); right_speed = 0.5*(tmp_right + maxSpeed); 
	    }
	  else if (speedEvalFlag == 2)
	    {
	      left_speed = minSpeed_element; right_speed = maxSpeed_element; 
	    }
	  left_flux = 0.0; right_flux = 0.0;
	  u_left = u[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary+
		     left_ebN_element*nQuadraturePoints_elementBoundary+
		     k];
	  u_right= u[right_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary+
		     right_ebN_element*nQuadraturePoints_elementBoundary+
		     k];
	  if (left_speed < 0.0)/*inflow for left*/
	    {
	      left_flux = fabs(left_speed)*(u_left-u_right);
	      flux[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary +
		   left_ebN_element*nQuadraturePoints_elementBoundary+ k ] = left_flux;
	      dflux_left[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary +
			 left_ebN_element*nQuadraturePoints_elementBoundary+ k ] = fabs(left_speed);
	      dflux_right[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary +
			  left_ebN_element*nQuadraturePoints_elementBoundary+ k ] =-fabs(left_speed);
	  
            }
	  else
	    {
	      left_flux = 0.0;
	      flux[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary +
		   left_ebN_element*nQuadraturePoints_elementBoundary+ k ] = left_flux;
	      dflux_left[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary +
			 left_ebN_element*nQuadraturePoints_elementBoundary+ k ] = 0.0;
	      dflux_right[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary +
			  left_ebN_element*nQuadraturePoints_elementBoundary+ k ] = 0.0;

	    }
	  if (right_speed > 0.0)/*inflow for right*/
	    {
	      right_flux = fabs(right_speed)*(u_right-u_left);
	      flux[right_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary +
		   right_ebN_element*nQuadraturePoints_elementBoundary+ k ] = right_flux;
	      dflux_left[right_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary +
			 right_ebN_element*nQuadraturePoints_elementBoundary+ k ] = -fabs(right_speed);
	      dflux_right[right_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary +
			  right_ebN_element*nQuadraturePoints_elementBoundary+ k ] = fabs(right_speed);
	    }
	  else
	    {
	      right_flux = 0.0;
	      flux[right_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary +
		   right_ebN_element*nQuadraturePoints_elementBoundary+ k ] = right_flux;
	      dflux_left[right_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary +
			 right_ebN_element*nQuadraturePoints_elementBoundary+ k ] = 0.0;
	      dflux_right[right_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary +
			  right_ebN_element*nQuadraturePoints_elementBoundary+ k ] = 0.0;
	    }
	  
	  /*mwf debug
	  printf("HJ Cheng Shu ebN=%d eN_left=%d eN_right=%d k=%d \n",ebN,left_eN_global,right_eN_global,k);
	  printf("HJ Cheng Shu left_speed=%g right_speed=%g u_left=%g u_right=%g maxSpeed_element=%g minSpeed_element=%g\n",
		 left_speed,right_speed,u_left,u_right,maxSpeed_element,minSpeed_element);
	  */
        }/*k*/
    }/*ebnI*/
}
double smoothedHeaviside(double eps, double phi)
{
  double H;
  if (phi > eps)
    H=1.0;
  else if (phi < -eps)
    H=0.0;
  else if (phi==0.0)
    H=0.5;
  else
    H = 0.5*(1.0 + phi/eps + sin(M_PI*phi/eps)/M_PI);
  return H;
}

double smoothedHeaviside_integral(double eps, double phi)
{
  double HI;
  if (phi > eps)
    {
      HI= phi - eps + 	0.5*(eps + 0.5*eps*eps/eps - eps*cos(M_PI*eps/eps)/(M_PI*M_PI)) - 0.5*((-eps) + 0.5*(-eps)*(-eps)/eps - eps*cos(M_PI*(-eps)/eps)/(M_PI*M_PI));
    }
  else if (phi < -eps)
    {
      HI=0.0;
    }
  else
    {
      HI = 0.5*(phi + 0.5*phi*phi/eps - eps*cos(M_PI*phi/eps)/(M_PI*M_PI)) - 0.5*((-eps) + 0.5*(-eps)*(-eps)/eps - eps*cos(M_PI*(-eps)/eps)/(M_PI*M_PI));
    }
  return HI;
}

double smoothedDirac(double eps, double phi)
{
  double d;
  if (phi > eps)
    d=0.0;
  else if (phi < -eps)
    d=0.0;
  else
    d = 0.5*(1.0 + cos(M_PI*phi/eps))/eps;
  return d;
}

void applySeepageFace(proteus_index_t nExteriorElementBoundaries_global,
		      int nQuadraturePoints_elementBoundary,
		      int nSpace,
		      proteus_index_t* exteriorElementBoundaries,
		      proteus_index_t* elementBoundaryElements,
		      int* elementBoundaryLocalElementBoundaries,
		      int* isSeepageFace,
		      int* isDOFBoundary,
		      double epsFact,
		      double* elementDiameters,
		      double* g,
		      double* n,
		      double* grad_u,
		      double* u,
		      double* advectiveFlux,
		      double* diffusiveFlux)
{
  proteus_index_t ebN, eN_global; int ebNE, k, J;
  double flow_direction,eps;
  for(ebNE=0;ebNE<nExteriorElementBoundaries_global;ebNE++)
    {
      ebN = exteriorElementBoundaries[ebNE];
      eN_global = elementBoundaryElements[ebN*2+0];
      eps=epsFact*elementDiameters[ebN];
      for(k=0;k<nQuadraturePoints_elementBoundary;k++)
        {
	  if (isSeepageFace[ebNE])
	    {
	      flow_direction=0.0;
	      for(J=0;J<nSpace;J++)
		{
		  flow_direction
		    += 
		    n[ebNE*nQuadraturePoints_elementBoundary*nSpace+
		      k*nSpace+
		      J]
		    *
		    (g[J]-grad_u[ebNE*nQuadraturePoints_elementBoundary*nSpace+
				 k*nSpace+
				 J]);
		}
	      isDOFBoundary[ebNE*nQuadraturePoints_elementBoundary+k] = 1;
	      if (flow_direction < 0.0 || u[ebNE*nQuadraturePoints_elementBoundary+k] < 0.0) //flow is coming back in and/or unsaturated
		{
		  isDOFBoundary[ebNE*nQuadraturePoints_elementBoundary+k] = 0;
		  advectiveFlux[ebNE*nQuadraturePoints_elementBoundary+
		  		k] = 0.0;
		  diffusiveFlux[ebNE*nQuadraturePoints_elementBoundary+
		  		k] = 0.0;
		}
	      else if (u[ebNE*nQuadraturePoints_elementBoundary+k] < eps) //unsaturated
	      	{
	      	  isDOFBoundary[ebNE*nQuadraturePoints_elementBoundary+k] = 1;
	      	  advectiveFlux[ebNE*nQuadraturePoints_elementBoundary+
	      			k]*=smoothedHeaviside(eps,u[ebNE*nQuadraturePoints_elementBoundary+k]);
	      	  diffusiveFlux[ebNE*nQuadraturePoints_elementBoundary+
	      			k]*=smoothedHeaviside(eps,u[ebNE*nQuadraturePoints_elementBoundary+k]);
	      	}
	    }
	}
    }
}

void calculateExteriorNumericalFluxRichards_sd(proteus_csr_index_t* rowptr,
					       proteus_csr_index_t* colind,
					       proteus_index_t nExteriorElementBoundaries_global,
					       int nQuadraturePoints_elementBoundary,
					       int nSpace,
					       int* isSeepageFace,
					       int* isDOFBoundary,
					       double* n,
					       double* bc_u,
					       double* K,
					       double* grad_psi,
					       double* u,
					       double* K_rho_g,
					       double* penalty,
					       double* diffusiveFlux)
{
  proteus_csr_index_t m; proteus_csr_index_t nnz=rowptr[nSpace]; int ebNE, k, I;
  double flux,v_I;
  for(ebNE=0;ebNE<nExteriorElementBoundaries_global;ebNE++)
    {
      for(k=0;k<nQuadraturePoints_elementBoundary;k++)
        {
	  if (isSeepageFace[ebNE] || isDOFBoundary[ebNE*nQuadraturePoints_elementBoundary+k])
	    {
	      flux = 0.0;
	      for(I=0;I<nSpace;I++)
		{
		  //initialize to gravity term
		  v_I = K_rho_g[ebNE*nQuadraturePoints_elementBoundary*nSpace+
				k*nSpace+
				I];
		  //add pressure head term
		  for(m=rowptr[I];m<rowptr[I+1];m++)
		    {
		      v_I 
			-= 
			K[ebNE*nQuadraturePoints_elementBoundary*nnz+
			  k*nnz+
			  m]
			*
			grad_psi[ebNE*nQuadraturePoints_elementBoundary*nSpace+
				 k*nSpace+
                                 colind[m]];
		    }
		  flux += 
		    v_I
		    *
		    n[ebNE*nQuadraturePoints_elementBoundary*nSpace+
		      k*nSpace+
		      I];
		}
	      //add Dirichlet penalty
              if (isSeepageFace[ebNE])
                bc_u[ebNE*nQuadraturePoints_elementBoundary+k] = 0.0;
	      diffusiveFlux[ebNE*nQuadraturePoints_elementBoundary+k] = 
                flux
		+ 
		penalty[ebNE*nQuadraturePoints_elementBoundary+k]
		*
		(u[ebNE*nQuadraturePoints_elementBoundary+k]
		 -
		 bc_u[ebNE*nQuadraturePoints_elementBoundary+k]);
              /* printf("Seepage Face %d %d psi = %12.5e psi_bc = %12.5e penalty = %12.5e flux = %12.5e \n", */
              /*        ebNE, */
              /*        isSeepageFace[ebNE], */
              /*        u[ebNE*nQuadraturePoints_elementBoundary+k], */
              /*        bc_u[ebNE*nQuadraturePoints_elementBoundary+k], */
              /*        penalty[ebNE*nQuadraturePoints_elementBoundary+k], */
              /*        flux); */
	      if (isSeepageFace[ebNE])
		{
		  //if (u[ebNE*nQuadraturePoints_elementBoundary+k] >= -0.01 || diffusiveFlux[ebNE*nQuadraturePoints_elementBoundary+k] > 0.0)
		  if (diffusiveFlux[ebNE*nQuadraturePoints_elementBoundary+k] > 0.0)
		  /* if ( */
                  /*     //                      (u[ebNE*nQuadraturePoints_elementBoundary+k] > 0.0) &&  */
                  /*     (flux > 0.0) */
                  /*     ) */
		    {
		      isDOFBoundary[ebNE*nQuadraturePoints_elementBoundary+k] = 1;
		    }
		  else
		    {
		      isDOFBoundary[ebNE*nQuadraturePoints_elementBoundary+k] = 0;
		      diffusiveFlux[ebNE*nQuadraturePoints_elementBoundary+k] = 0.0;
		    }
		}
	    }
	}
    }
}

void calculateExteriorNumericalFluxJacobianRichards_sd(proteus_csr_index_t* rowptr,
						       proteus_csr_index_t* colind,
						       proteus_index_t nExteriorElementBoundaries_global,
						       int nQuadraturePoints_elementBoundary,
						       int nDOF_trial_element,
						       int nSpace,
						       int* isDOFBoundary,
						       double* n,
						       double* bc_u,
						       double* K,
						       double* dK,
						       double* grad_psi,
						       double* grad_v,
						       double* u,
						       double* dK_rho_g,
						       double* v,
						       double* penalty,
						       double* fluxJacobian)
{
  proteus_csr_index_t m; proteus_csr_index_t nnz=rowptr[nSpace]; int ebNE, k, j, I;
  double dFlux_j,dv_I_j;
  for(ebNE=0;ebNE<nExteriorElementBoundaries_global;ebNE++)
    {
      for(k=0;k<nQuadraturePoints_elementBoundary;k++)
        {
	  if (isDOFBoundary[ebNE*nQuadraturePoints_elementBoundary+k])
	    {
	      for (j=0;j<nDOF_trial_element;j++)
		{
		  dFlux_j = 0.0;
		  for(I=0;I<nSpace;I++)
		    {
		      //initialize to gravity term
		      dv_I_j = dK_rho_g[ebNE*nQuadraturePoints_elementBoundary*nSpace+
					k*nSpace+
					I]
			*
			v[ebNE*nQuadraturePoints_elementBoundary*nDOF_trial_element+
			  k*nDOF_trial_element+
			  j];
		      //add pressure head term
		      for(m=rowptr[I];m<rowptr[I+1];m++)
			{
			  dv_I_j 
			    -= 
			    K[ebNE*nQuadraturePoints_elementBoundary*nnz+
			      k*nnz+
			      m]
			    *
			    grad_v[ebNE*nQuadraturePoints_elementBoundary*nDOF_trial_element*nSpace+
				   k*nDOF_trial_element*nSpace+
				   j*nSpace+
				   colind[m]]
			    +
			    dK[ebNE*nQuadraturePoints_elementBoundary*nnz+
			       k*nnz+
			       m]
			    *
			    grad_psi[ebNE*nQuadraturePoints_elementBoundary*nSpace+
				     k*nSpace+
				     colind[m]]
			    *v[ebNE*nQuadraturePoints_elementBoundary*nDOF_trial_element+
			       k*nDOF_trial_element+
			       j];
			}
		      dFlux_j += 
			dv_I_j
			*
			n[ebNE*nQuadraturePoints_elementBoundary*nSpace+
			  k*nSpace+
			  I];
		    }
		  //add Dirichlet penalty
		  fluxJacobian[ebNE*nQuadraturePoints_elementBoundary*nDOF_trial_element+
			       k*nDOF_trial_element+
			       j]
		    =
		    dFlux_j
		    + 
		    penalty[ebNE*nQuadraturePoints_elementBoundary+k]
		    *v[ebNE*nQuadraturePoints_elementBoundary*nDOF_trial_element+
		       k*nDOF_trial_element+
		       j];
		}
	    }
	}
    }
}

void applySeepageFaceJacobian(proteus_index_t nExteriorElementBoundaries_global,
			      int nQuadraturePoints_elementBoundary,
			      int nDOF_trial_element,
			      int nSpace,
			      proteus_index_t* exteriorElementBoundaries,
			      proteus_index_t* elementBoundaryElements,
			      int* elementBoundaryLocalElementBoundaries,
			      int* isSeepageFace,
			      double epsFact,
			      double* elementDiameters,
			      double* g,
			      double* n,
			      double* grad_u,
			      double* u,
			      double* advectiveFlux,
			      double* diffusiveFlux,
			      double* v,
			      double* fluxJacobian)
{
  proteus_index_t ebN, eN_global; int ebNE, k, J, j;
  double flow_direction,eps;
  for(ebNE=0;ebNE<nExteriorElementBoundaries_global;ebNE++)
    {
      ebN = exteriorElementBoundaries[ebNE];
      eN_global = elementBoundaryElements[ebN*2+0];
      eps=epsFact*elementDiameters[ebN];
      for(k=0;k<nQuadraturePoints_elementBoundary;k++)
        {
	  if (isSeepageFace[ebNE])
	    {
	      flow_direction=0.0;
	      for(J=0;J<nSpace;J++)
		{
		  flow_direction
		    += 
		    n[ebNE*nQuadraturePoints_elementBoundary*nSpace+
		      k*nSpace+
		      J]
		    *
		    (g[J]-grad_u[ebNE*nQuadraturePoints_elementBoundary*nSpace+
				 k*nSpace+
				 J]);
		}
	      for(j=0;j<nDOF_trial_element;j++)
		{
		  if (flow_direction < 0.0 || u[ebNE*nQuadraturePoints_elementBoundary+k] < 0.0) //flow is coming back in and/or unsaturated
		    {
		      fluxJacobian[ebNE*nQuadraturePoints_elementBoundary*nDOF_trial_element+
				   k*nDOF_trial_element+
		      		   j] = 0.0;
		    }
		  else if (u[ebNE*nQuadraturePoints_elementBoundary+k] < eps)//unsaturated
		    {
		      fluxJacobian[ebNE*nQuadraturePoints_elementBoundary*nDOF_trial_element+
		  		   k*nDOF_trial_element+
		  		   j]*=smoothedHeaviside(eps,u[ebNE*nQuadraturePoints_elementBoundary+k]);
		      fluxJacobian[ebNE*nQuadraturePoints_elementBoundary*nDOF_trial_element+
		  		   k*nDOF_trial_element+
		  		   j] += (advectiveFlux[ebNE*nQuadraturePoints_elementBoundary+
		  					k] +
		  			  diffusiveFlux[ebNE*nQuadraturePoints_elementBoundary+
		  					k])
		  	*
		  	smoothedDirac(eps,u[ebNE*nQuadraturePoints_elementBoundary+k])
		  	*
		  	v[ebNE*nQuadraturePoints_elementBoundary*nDOF_trial_element+
		  	  k*nDOF_trial_element+
		  	  j];
		      
		    }
		}
	    }
	}
    }
}

void calculateGlobalExteriorNumericalStressFlux(proteus_index_t nExteriorElementBoundaries_global,
						int nQuadraturePoints_elementBoundary,
						int nSpace,
						proteus_index_t* exteriorElementBoundaries,
						proteus_index_t* elementBoundaryElements,
						int* elementBoundaryLocalElementBoundaries,
						int *isDOFBoundary_u,
						int *isDOFBoundary_v,
						int *isDOFBoundary_w,
						double* n,
						double* bc_u,
						double* bc_v,
						double* bc_w,
						double* sigma,
						double* u,
						double* v,
						double* w,
						double* penalty,
						double* stressFlux_u,
						double* stressFlux_v,
						double* stressFlux_w)
{
  int ebNE,k,nSpace2=nSpace*nSpace;
  for(ebNE=0;ebNE<nExteriorElementBoundaries_global;ebNE++)
    {
      for(k=0;k<nQuadraturePoints_elementBoundary;k++)
        {
	  //cek hack debug
	  penalty[ebNE*nQuadraturePoints_elementBoundary+k] = 1.0e5;
	  //
	  double *normal = n + ebNE*nQuadraturePoints_elementBoundary*nSpace+k*nSpace;
	  double *stress = sigma + ebNE*nQuadraturePoints_elementBoundary*nSpace2+k*nSpace2;
	  if (isDOFBoundary_u[ebNE*nQuadraturePoints_elementBoundary+k] == 1)
	    {
	      double u_jump = -penalty[ebNE*nQuadraturePoints_elementBoundary+k]
		*
		(u[ebNE*nQuadraturePoints_elementBoundary+k]
		 - bc_u[ebNE*nQuadraturePoints_elementBoundary+k]);
	      stressFlux_u[ebNE*nQuadraturePoints_elementBoundary+k] = -(stress[0]*normal[0] + stress[1]*normal[1] + stress[2]*normal[2] + u_jump);
	      //stressFlux_u[ebNE*nQuadraturePoints_elementBoundary+k] = -(u_jump);
	    }
	  if (isDOFBoundary_v[ebNE*nQuadraturePoints_elementBoundary+k] == 1)
	    {
	      double v_jump = -penalty[ebNE*nQuadraturePoints_elementBoundary+k]*
		(v[ebNE*nQuadraturePoints_elementBoundary+k]
		 - bc_v[ebNE*nQuadraturePoints_elementBoundary+k]);
	      stressFlux_v[ebNE*nQuadraturePoints_elementBoundary+k] = -(stress[3]*normal[0] + stress[4]*normal[1] + stress[5]*normal[2] + v_jump);
	      //stressFlux_v[ebNE*nQuadraturePoints_elementBoundary+k] = -(v_jump);
	    }
	  if (isDOFBoundary_w[ebNE*nQuadraturePoints_elementBoundary+k] == 1)
	    {
	      double w_jump = -penalty[ebNE*nQuadraturePoints_elementBoundary+k]
		*
		(w[ebNE*nQuadraturePoints_elementBoundary+k]
		 - bc_w[ebNE*nQuadraturePoints_elementBoundary+k]);
	      stressFlux_w[ebNE*nQuadraturePoints_elementBoundary+k] = -(stress[6]*normal[0] + stress[7]*normal[1] + stress[8]*normal[2] + w_jump);
	      //stressFlux_w[ebNE*nQuadraturePoints_elementBoundary+k] = -(w_jump);
	    }
	}
    }
}

void updateExteriorNumericalStressFluxJacobian(proteus_index_t nExteriorElementBoundaries_global,
					       int nQuadraturePoints_elementBoundary,
					       int nDOF_trial_element,
					       int nSpace,
					       proteus_index_t* exteriorElementBoundaries,
					       proteus_index_t* elementBoundaryElements,
					       int* elementBoundaryLocalElementBoundaries,
					       int* isDOFBoundary_u,
					       int* isDOFBoundary_v,
					       int* isDOFBoundary_w,
					       int* isStressBoundary_u,
					       int* isStressBoundary_v,
					       int* isStressBoundary_w,
					       double* n,
					       double* dstress_u_u,
					       double* dstress_u_v,
					       double* dstress_u_w,
					       double* dstress_v_u,
					       double* dstress_v_v,
					       double* dstress_v_w,
					       double* dstress_w_u,
					       double* dstress_w_v,
					       double* dstress_w_w,
					       double* v,
					       double* grad_v,
					       double* penalty,
					       double* fluxJacobian_u_u,
					       double* fluxJacobian_u_v,
					       double* fluxJacobian_u_w,
					       double* fluxJacobian_v_u,
					       double* fluxJacobian_v_v,
					       double* fluxJacobian_v_w,
					       double* fluxJacobian_w_u,
					       double* fluxJacobian_w_v,
					       double* fluxJacobian_w_w)
{
  //int ebNE,ebN,eN_global,k,j,j_global,I,J,nSpace2=nSpace*nSpace;
  int ebNE,k,j,I,J,nSpace2=nSpace*nSpace;
  //double Jacobian,diffusiveVelocityComponent_I_Jacobian,diffusiveVelocityComponent_I_Jacobian2,max_a;
  for(ebNE=0;ebNE<nExteriorElementBoundaries_global;ebNE++)
    {
      //ebN = exteriorElementBoundaries[ebNE];
      //eN_global = elementBoundaryElements[ebN*2+0];
      for(k=0;k<nQuadraturePoints_elementBoundary;k++)
        {
          if(isDOFBoundary_u[ebNE*nQuadraturePoints_elementBoundary+k] == 1)
            {
              for(j=0;j<nDOF_trial_element;j++)
                {
		  fluxJacobian_u_u[ebNE*nQuadraturePoints_elementBoundary*nDOF_trial_element+
				   k*nDOF_trial_element+
				   j]
		    =penalty[ebNE*nQuadraturePoints_elementBoundary+k]
		    *
		    v[ebNE*nQuadraturePoints_elementBoundary*nDOF_trial_element+
		      k*nDOF_trial_element+
		      j];
		  for (I=0;I<nSpace;I++)
		    for (J=0;J<nSpace;J++)
		      {
		  	fluxJacobian_u_u[ebNE*nQuadraturePoints_elementBoundary*nDOF_trial_element+
		  			 k*nDOF_trial_element+
		  			 j]
		  	  -=
		  	  dstress_u_u[ebNE*nQuadraturePoints_elementBoundary*nSpace2+
				      k*nSpace2+
				      I*nSpace+
				      J]
		  	  *
		  	  grad_v[ebNE*nQuadraturePoints_elementBoundary*nDOF_trial_element*nSpace+
		  		 k*nDOF_trial_element*nSpace+
		  		 j*nSpace+
		  		 J]
		  	  *
		  	  n[ebNE*nQuadraturePoints_elementBoundary*nSpace+
			    k*nSpace+
			    I];
		  	fluxJacobian_u_v[ebNE*nQuadraturePoints_elementBoundary*nDOF_trial_element+
		  			 k*nDOF_trial_element+
		  			 j]
		  	  -=
		  	  dstress_u_v[ebNE*nQuadraturePoints_elementBoundary*nSpace2+
				      k*nSpace2+
				      I*nSpace+
				      J]
		  	  *
		  	  grad_v[ebNE*nQuadraturePoints_elementBoundary*nDOF_trial_element*nSpace+
		  		 k*nDOF_trial_element*nSpace+
		  		 j*nSpace+
		  		 J]
		  	  *
		  	  n[ebNE*nQuadraturePoints_elementBoundary*nSpace+
			    k*nSpace+
			    I];
		  	fluxJacobian_u_w[ebNE*nQuadraturePoints_elementBoundary*nDOF_trial_element+
		  			 k*nDOF_trial_element+
		  			 j]
		  	  -=
		  	  dstress_u_w[ebNE*nQuadraturePoints_elementBoundary*nSpace2+
				      k*nSpace2+
				      I*nSpace+
				      J]
		  	  *
		  	  grad_v[ebNE*nQuadraturePoints_elementBoundary*nDOF_trial_element*nSpace+
		  		 k*nDOF_trial_element*nSpace+
		  		 j*nSpace+
		  		 J]
		  	  *
		  	  n[ebNE*nQuadraturePoints_elementBoundary*nSpace+
			    k*nSpace+
			    I];
		      }
		}
	    }
          if(isDOFBoundary_v[ebNE*nQuadraturePoints_elementBoundary+k] == 1)
            {
              for(j=0;j<nDOF_trial_element;j++)
                {
		  fluxJacobian_v_v[ebNE*nQuadraturePoints_elementBoundary*nDOF_trial_element+
				   k*nDOF_trial_element+
				   j]
		    =penalty[ebNE*nQuadraturePoints_elementBoundary+
			     k]
		    *
		    v[ebNE*nQuadraturePoints_elementBoundary*nDOF_trial_element+
		      k*nDOF_trial_element+
		      j];
		  for (I=0;I<nSpace;I++)
		    for (J=0;J<nSpace;J++)
		      {
		  	fluxJacobian_v_u[ebNE*nQuadraturePoints_elementBoundary*nDOF_trial_element+
		  			 k*nDOF_trial_element+
		  			 j]
		  	  -=
		  	  dstress_v_u[ebNE*nQuadraturePoints_elementBoundary*nSpace2+
				      k*nSpace2+
				      I*nSpace+
				      J]
		  	  *
		  	  grad_v[ebNE*nQuadraturePoints_elementBoundary*nDOF_trial_element*nSpace+
		  		 k*nDOF_trial_element*nSpace+
		  		 j*nSpace+
		  		 J]
		  	  *
		  	  n[ebNE*nQuadraturePoints_elementBoundary*nSpace+
			    k*nSpace+
			    I];
		  	fluxJacobian_v_v[ebNE*nQuadraturePoints_elementBoundary*nDOF_trial_element+
		  			 k*nDOF_trial_element+
		  			 j]
		  	  -=
		  	  dstress_v_v[ebNE*nQuadraturePoints_elementBoundary*nSpace2+
				      k*nSpace2+
				      I*nSpace+
				      J]
		  	  *
		  	  grad_v[ebNE*nQuadraturePoints_elementBoundary*nDOF_trial_element*nSpace+
		  		 k*nDOF_trial_element*nSpace+
		  		 j*nSpace+
		  		 J]
		  	  *
		  	  n[ebNE*nQuadraturePoints_elementBoundary*nSpace+
			    k*nSpace+
			    I];
		  	fluxJacobian_v_w[ebNE*nQuadraturePoints_elementBoundary*nDOF_trial_element+
		  			 k*nDOF_trial_element+
		  			 j]
		  	  -=
		  	  dstress_v_w[ebNE*nQuadraturePoints_elementBoundary*nSpace2+
				      k*nSpace2+
				      I*nSpace+
				      J]
		  	  *
		  	  grad_v[ebNE*nQuadraturePoints_elementBoundary*nDOF_trial_element*nSpace+
		  		 k*nDOF_trial_element*nSpace+
		  		 j*nSpace+
		  		 J]
		  	  *
		  	  n[ebNE*nQuadraturePoints_elementBoundary*nSpace+
			    k*nSpace+
			    I];
		      }
		}
	    }
          if(isDOFBoundary_w[ebNE*nQuadraturePoints_elementBoundary+k] == 1)
            {
              for(j=0;j<nDOF_trial_element;j++)
                {
		  fluxJacobian_w_w[ebNE*nQuadraturePoints_elementBoundary*nDOF_trial_element+
				   k*nDOF_trial_element+
				   j]
		    =
		    penalty[ebNE*nQuadraturePoints_elementBoundary+
			    k]
		    *
		    v[ebNE*nQuadraturePoints_elementBoundary*nDOF_trial_element+
		      k*nDOF_trial_element+
		      j];
		  for (I=0;I<nSpace;I++)
		    for (J=0;J<nSpace;J++)
		      {
		  	fluxJacobian_w_u[ebNE*nQuadraturePoints_elementBoundary*nDOF_trial_element+
		  			 k*nDOF_trial_element+
		  			 j]
		  	  -=
		  	  dstress_w_u[ebNE*nQuadraturePoints_elementBoundary*nSpace2+
				      k*nSpace2+
				      I*nSpace+
				      J]
		  	  *
		  	  grad_v[ebNE*nQuadraturePoints_elementBoundary*nDOF_trial_element*nSpace+
		  		 k*nDOF_trial_element*nSpace+
		  		 j*nSpace+
		  		 J]
		  	  *
		  	  n[ebNE*nQuadraturePoints_elementBoundary*nSpace+
			    k*nSpace+
			    I];
		  	fluxJacobian_w_v[ebNE*nQuadraturePoints_elementBoundary*nDOF_trial_element+
		  			 k*nDOF_trial_element+
		  			 j]
		  	  -=
		  	  dstress_w_v[ebNE*nQuadraturePoints_elementBoundary*nSpace2+
				      k*nSpace2+
				      I*nSpace+
				      J]
		  	  *
		  	  grad_v[ebNE*nQuadraturePoints_elementBoundary*nDOF_trial_element*nSpace+
		  		 k*nDOF_trial_element*nSpace+
		  		 j*nSpace+
		  		 J]
		  	  *
		  	  n[ebNE*nQuadraturePoints_elementBoundary*nSpace+
			    k*nSpace+
			    I];
		  	fluxJacobian_w_w[ebNE*nQuadraturePoints_elementBoundary*nDOF_trial_element+
		  			 k*nDOF_trial_element+
		  			 j]
		  	  -=
		  	  dstress_w_w[ebNE*nQuadraturePoints_elementBoundary*nSpace2+
				      k*nSpace2+
				      I*nSpace+
				      J]
		  	  *
		  	  grad_v[ebNE*nQuadraturePoints_elementBoundary*nDOF_trial_element*nSpace+
		  		 k*nDOF_trial_element*nSpace+
		  		 j*nSpace+
		  		 J]
		  	  *
		  	  n[ebNE*nQuadraturePoints_elementBoundary*nSpace+
			    k*nSpace+
			    I];
		      }
		}
	    }
	}
    }
}

/**
   Allow upwinding on the diffusion term if this actually is part of an advective process.
   Comes up in fully coupled Darcy flow and Transport formulations
 */
void calculateGlobalExteriorNumericalDiffusiveFluxWithUpwinding_sd(int scale_penalty,
								   double penalty_floor,
								   proteus_index_t nExteriorElementBoundaries_global,
								   int nQuadraturePoints_elementBoundary,
								   int nSpace,
								   proteus_csr_index_t* rowptr,
								   proteus_csr_index_t* colind,
								   proteus_index_t* exteriorElementBoundaries,
								   proteus_index_t* elementBoundaryElements,
								   int* elementBoundaryLocalElementBoundaries,
								   int* isDOFBoundary,
								   int* fluxBoundaryFlag, /*0 no flow, 1 outflow */
								   double* n,
								   double* bc_a,
								   double* bc_grad_phi,
								   double* bc_u,
								   double* a,
								   double* grad_phi,
								   double* u,
								   double* penalty,
								   double* flux)
{
  proteus_csr_index_t m; proteus_csr_index_t nnz=rowptr[nSpace]; int ebNE, k, I;
  double diffusiveVelocityComponent_I,penaltyFlux,max_a;
  double potential_gradient=0.0;
  for(ebNE=0;ebNE<nExteriorElementBoundaries_global;ebNE++)
    {
      for(k=0;k<nQuadraturePoints_elementBoundary;k++)
        {
	  /*figure out if the potential gradient is pointing out or not to upwind*/
	  potential_gradient=0.;
	  for (I=0; I < nSpace; I++)
	    {
	      potential_gradient += grad_phi[ebNE*nQuadraturePoints_elementBoundary*nSpace +
					     k*nSpace + I]
		*
		n[ebNE*nQuadraturePoints_elementBoundary*nSpace +
		  k*nSpace+
		  I];
	    }
	  /* apply diffusive flux term if it's a Dirichlet boundary or outflow boundary and the potential gradient is out*/
          if(isDOFBoundary[ebNE*nQuadraturePoints_elementBoundary+k] == 1 || (potential_gradient > 0.0 && fluxBoundaryFlag == 1))
            {
              flux[ebNE*nQuadraturePoints_elementBoundary+k] = 0.0;
              max_a=0.0;
              for(I=0;I<nSpace;I++)
                {
                  diffusiveVelocityComponent_I=0.0;
                  for(m=rowptr[I];m<rowptr[I+1];m++)
                    {
                      diffusiveVelocityComponent_I 
                        -= 
                        a[ebNE*nQuadraturePoints_elementBoundary*nnz+
                          k*nnz+
			  m]
                        *
                        grad_phi[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                                 k*nSpace+colind[m]];
                      max_a = fmax(max_a,a[ebNE*nQuadraturePoints_elementBoundary*nnz+
                                          k*nnz+
					   m]);
                    }
                  flux[ebNE*nQuadraturePoints_elementBoundary+k] 
                    += 
                    diffusiveVelocityComponent_I
                    *
                    n[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                      k*nSpace+
                      I];
                }
              max_a = fmax(penalty_floor,max_a);
	      if (isDOFBoundary[ebNE*nQuadraturePoints_elementBoundary+k]==1)
		{
		  penaltyFlux = penalty[ebNE*nQuadraturePoints_elementBoundary+
                                    k]
		    *
		    (u[ebNE*nQuadraturePoints_elementBoundary+
		       k]
		     -
		     bc_u[ebNE*nQuadraturePoints_elementBoundary+
			  k]);

		  if (scale_penalty) penaltyFlux *= max_a;
		  flux[ebNE*nQuadraturePoints_elementBoundary+k]  += penaltyFlux;
		}
            }
        }
    }
}

void updateGlobalExteriorNumericalDiffusiveFluxWithUpwindingJacobian_sd(int scale_penalty,
									double penalty_floor,
									proteus_index_t nExteriorElementBoundaries_global,
									int nQuadraturePoints_elementBoundary,
									int nDOF_trial_element,
									int nSpace,
									proteus_csr_index_t* rowptr,
									proteus_csr_index_t* colind,
									proteus_index_t* l2g,
									proteus_index_t* exteriorElementBoundaries,
									proteus_index_t* elementBoundaryElements,
									int* elementBoundaryLocalElementBoundaries,
									int* isDOFBoundary,
									int* fluxBoundaryFlag, /*0 no flow, 1 outflow */
									double* n,
									double* a,
									double* da,
									double* grad_phi,
									double* dphi,
									double* v,
									double* grad_v,
									double* penalty,
									double* fluxJacobian)
{
  proteus_index_t ebN, eN_global, j_global; proteus_csr_index_t m; proteus_csr_index_t nnz=rowptr[nSpace]; int ebNE, k, j, I;
  double Jacobian,diffusiveVelocityComponent_I_Jacobian,diffusiveVelocityComponent_I_Jacobian2,max_a;
  double potential_gradient=0.;
  for(ebNE=0;ebNE<nExteriorElementBoundaries_global;ebNE++)
    {
      ebN = exteriorElementBoundaries[ebNE];
      eN_global = elementBoundaryElements[ebN*2+0];
      for(k=0;k<nQuadraturePoints_elementBoundary;k++)
        {
	  /*figure out if the potential gradient is pointing out or not to upwind*/
	  potential_gradient=0.;
	  for (I=0; I < nSpace; I++)
	    {
	      potential_gradient += grad_phi[ebNE*nQuadraturePoints_elementBoundary*nSpace +
					     k*nSpace + I]
		*
		n[ebNE*nQuadraturePoints_elementBoundary*nSpace +
		  k*nSpace+
		  I];
	    }
	  /* apply diffusive flux term if it's a Dirichlet boundary or outflow boundary and the potential gradient is out*/
          if(isDOFBoundary[ebNE*nQuadraturePoints_elementBoundary+k] >= 1 || (potential_gradient > 0.0 && fluxBoundaryFlag == 1))
            {
              for(j=0;j<nDOF_trial_element;j++)
                {
                  Jacobian=0.0;
                  j_global = l2g[eN_global*nDOF_trial_element+j];
                  max_a=0.0;
                  for(I=0;I<nSpace;I++)
                    {
                      diffusiveVelocityComponent_I_Jacobian=0.0;
                      diffusiveVelocityComponent_I_Jacobian2=0.0;
                      for(m=rowptr[I];m<rowptr[I+1];m++)
                        {
                          diffusiveVelocityComponent_I_Jacobian 
                            -= 
                            da[ebNE*nQuadraturePoints_elementBoundary*nnz+
                               k*nnz+
                               m]
                            *
                            grad_phi[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                                     k*nSpace+
                                     colind[m]];
                          diffusiveVelocityComponent_I_Jacobian2 
                            -= 
                            a[ebNE*nQuadraturePoints_elementBoundary*nnz+
                              k*nnz+
                              m]
                            *
                            grad_v[ebNE*nQuadraturePoints_elementBoundary*nDOF_trial_element*nSpace+
                                   k*nDOF_trial_element*nSpace+
                                   j*nSpace+
                                   colind[m]];
                          max_a = fmax(max_a,a[ebNE*nQuadraturePoints_elementBoundary*nnz+
                                               k*nnz+
                                               m]);
                                       
                        }
                      Jacobian 
                        += 
                        (diffusiveVelocityComponent_I_Jacobian
                         *
                         v[ebNE*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                           k*nDOF_trial_element+
                           j] 
                         +
                         diffusiveVelocityComponent_I_Jacobian2*
                         dphi[j_global])
                        *
                        n[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                          k*nSpace+
                          I];
                    }
                  max_a = fmax(penalty_floor,max_a);

		  if(isDOFBoundary[ebNE*nQuadraturePoints_elementBoundary+k] == 1)
		    {
		      double penaltyJacobian = penalty[ebNE*nQuadraturePoints_elementBoundary+
                                                   k]
			*
			v[ebNE*nQuadraturePoints_elementBoundary*nDOF_trial_element+
			  k*nDOF_trial_element+
			  j];
		      if (scale_penalty) penaltyJacobian *= max_a;

		      Jacobian += penaltyJacobian;
		    }
                  fluxJacobian[ebNE*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                               k*nDOF_trial_element+
                               j]
                    += Jacobian;
                }
            }
        }
    }
}
