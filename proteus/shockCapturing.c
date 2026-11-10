#include <math.h>
#include "shockCapturing.h"

/** \file shockCapturing.c
    \ingroup shockCapturing
    @{
*/
void calculateNumericalDiffusionResGrad(proteus_index_t nElements_global,
                                        int nQuadraturePoints_element,
                                        int nSpace,
                                        double shockCapturingDiffusion,
                                        double* elementDiameter,
                                        double* strong_residual,
                                        double* grad_u,
                                        double* numDiff)
{
  int eN,k,I;
  double h,
    num,
    den,
    n_grad_u[nQuadraturePoints_element],
    n_R_eN,
    maxNumDiff;
  for(eN=0;eN<nElements_global;eN++)
    {
      h = elementDiameter[eN];
      n_R_eN=0.0;
      for (k=0;k<nQuadraturePoints_element;k++)
        {
          n_grad_u[k] = 0.0;
          for (I=0;I<nSpace;I++)
            {
              n_grad_u[k] += grad_u[eN*nQuadraturePoints_element*nSpace+k*nSpace+I]*grad_u[eN*nQuadraturePoints_element*nSpace+k*nSpace+I];
            }
          n_grad_u[k] = sqrt(n_grad_u[k]);
          num = shockCapturingDiffusion*0.5*h*fabs(strong_residual[eN*nQuadraturePoints_element+k]);
          den = n_grad_u[k];
          if (den > num*1.0e-8) 
            numDiff[eN*nQuadraturePoints_element+k] = num/den;
          else
            numDiff[eN*nQuadraturePoints_element+k]=0.0;
          n_R_eN=fmax(n_R_eN,fabs(strong_residual[eN*nQuadraturePoints_element+k]));
        }
      maxNumDiff=0.0;
      for (k=0;k<nQuadraturePoints_element;k++)
        {
          num = shockCapturingDiffusion*0.5*h*n_R_eN;
          den = n_grad_u[k];
          if (den > num*1.0e-8)
            numDiff[eN*nQuadraturePoints_element+k] = num/den;
          else
            numDiff[eN*nQuadraturePoints_element+k]=0.0;
          maxNumDiff = fmax(maxNumDiff,numDiff[eN*nQuadraturePoints_element+k]);
        }
      for (k=0;k<nQuadraturePoints_element;k++)
         numDiff[eN*nQuadraturePoints_element+k] = maxNumDiff;
    }
}

void calculateNumericalDiffusionResGradQuad(proteus_index_t nElements_global,
                                            int nQuadraturePoints_element,
                                            int nSpace,
                                            double shockCapturingDiffusion,
                                            double* elementDiameter,
                                            double* strong_residual,
                                            double* grad_u,
                                            double* numDiff)
{
  int eN,k,I;
  double h,
    num,
    den,
    n_grad_u;
  for(eN=0;eN<nElements_global;eN++)
    {
      h = elementDiameter[eN];
      for (k=0;k<nQuadraturePoints_element;k++)
        {
          n_grad_u = 0.0;
          for (I=0;I<nSpace;I++)
            {
              n_grad_u += grad_u[eN*nQuadraturePoints_element*nSpace+k*nSpace+I]*grad_u[eN*nQuadraturePoints_element*nSpace+k*nSpace+I];
            }
          num = shockCapturingDiffusion*0.5*h*fabs(strong_residual[eN*nQuadraturePoints_element+k]);
          den = sqrt(n_grad_u) + 1.0e-8;
          numDiff[eN*nQuadraturePoints_element+k] = num/den;
        }
    }
}

void calculateNumericalDiffusionEikonal(proteus_index_t nElements_global,
					int nQuadraturePoints_element,
					double shockCapturingDiffusion,
					double* elementDiameter,
					double* strong_residual,
					double* numDiff)
{
  int eN,k;
  double h;
  for(eN=0;eN<nElements_global;eN++)
    {
      h = elementDiameter[eN];
      for (k=0;k<nQuadraturePoints_element;k++)
        {
          numDiff[eN*nQuadraturePoints_element+k] = shockCapturingDiffusion*0.5*h*fabs(strong_residual[eN*nQuadraturePoints_element+k]);
	  //printf("%12.5e\n",numDiff[eN*nQuadraturePoints_element+k]);
        }
    }
}

/**
   \brief Calculate the shock capturing diffusion for a Hamilton-Jacobi equation at the quadrature  points
*/
void calculateNumericalDiffusionHJ(proteus_index_t nElements_global,
				   int nQuadraturePoints_element,
				   char shockCapturing,
				   double shockCapturingDiffusion,
				   double* elementDiameter,
				   double* strong_residual,
				   double* mt,
				   double* H,
				   double* numDiff)
{
  int eN,k;
  double h,
    num,
    den,
    n_R_eN;
  //maxNumDiff;
  for(eN=0;eN<nElements_global;eN++)
    {
      h = elementDiameter[eN];
      n_R_eN=0.0;
      for (k=0;k<nQuadraturePoints_element;k++)
        {
          num = shockCapturingDiffusion*0.5*h*fabs(strong_residual[eN*nQuadraturePoints_element+k]);
          den = fabs(H[eN*nQuadraturePoints_element+k]) + 
            fabs(mt[eN*nQuadraturePoints_element+k]);
          if (den > num*1.0e-8) 
            numDiff[eN*nQuadraturePoints_element+k] = num/den;
          else
            numDiff[eN*nQuadraturePoints_element+k]=0.0;
          n_R_eN=fmax(n_R_eN,fabs(strong_residual[eN*nQuadraturePoints_element+k]));
        }
/*       maxNumDiff=0.0; */
/*       for (k=0;k<nQuadraturePoints_element;k++) */
/*         { */
/*           num = shockCapturingDiffusion*0.5*h*n_R_eN; */
/*           den = fabs(H[eN*nQuadraturePoints_element+k])+ */
/*             fabs(mt[eN*nQuadraturePoints_element+k]); */
/*           if (den > num*1.0e-8) */
/*             numDiff[eN*nQuadraturePoints_element+k] = num/den; */
/*           else */
/*             numDiff[eN*nQuadraturePoints_element+k]=0.0; */
/*           maxNumDiff = fmax(maxNumDiff,numDiff[eN*nQuadraturePoints_element+k]); */
/*         } */
/*       for (k=0;k<nQuadraturePoints_element;k++) */
/*          numDiff[eN*nQuadraturePoints_element+k] = maxNumDiff; */
    }
}

/**
   \brief Calculate the shock capturing diffusion for a Hamilton-Jacobi equation at the quadrature  points
 mwf try this with \f$H(\nabla u)\f$ in denominator
*/
void calculateNumericalDiffusionHJV2(proteus_index_t nElements_global,
				     int nQuadraturePoints_element,
				     char shockCapturing,
				     double shockCapturingDiffusion,
				     double* elementDiameter,
				     double* strong_residual,
				     double* mt,
				     double* H,
				     double* numDiff)
{
  int eN,k;
  double h,num,den=0.0,enorm_h;
  for(eN=0;eN<nElements_global;eN++)
    {
      h = elementDiameter[eN];
      for (k=0;k<nQuadraturePoints_element;k++)
        {
          num = shockCapturingDiffusion*h*fabs(strong_residual[eN*nQuadraturePoints_element+
                                                           k]); 
          enorm_h = fabs(H[eN*nQuadraturePoints_element + k]);
          if(shockCapturing == '1')
            {
              den = (fabs(mt[eN*nQuadraturePoints_element+
                             k]) +
                     enorm_h)+num*1.0e-8;
            }
          else if(shockCapturing == '2')
            {
              den = sqrt(mt[eN*nQuadraturePoints_element+
                            k]
                         *
                         mt[eN*nQuadraturePoints_element+
                            k] 
                         +
                         enorm_h*enorm_h) + num*1.0e-8;
            }
          numDiff[eN*nQuadraturePoints_element+k] = num/den;
        }
    }
}
 
void calculateNumericalDiffusion_A_1(proteus_index_t nElements_global,
                                     int nQuadraturePoints_element,
                                     int nSpace,
                                     double shockCapturingFactor,
                                     double* elementDiameter,
                                     double* strong_residual,
                                     double* mt,
                                     double* df,
                                     double* numDiff)
{
  int eN,k,I;
  double h,num,den,enorm_df;
  for(eN=0;eN<nElements_global;eN++)
    {
      h = elementDiameter[eN];
      for (k=0;k<nQuadraturePoints_element;k++)
        {
          num = shockCapturingFactor*h*fabs(strong_residual[eN*nQuadraturePoints_element+
                                                            k]); 
          enorm_df = 0.0;
          for(I=0;I<nSpace;I++)
            {
              enorm_df
                +=
                df[eN*nQuadraturePoints_element*nSpace+
                   k*nSpace+
                   I]
                *
                df[eN*nQuadraturePoints_element*nSpace+
                   k*nSpace+
                   I];
            }
          enorm_df = sqrt(enorm_df);
          den = (fabs(mt[eN*nQuadraturePoints_element+
                         k]) +
                 enorm_df);
          numDiff[eN*nQuadraturePoints_element+k] = num/den;
        }
    }
}
/*mwf try Juanes'  formula ...
  D_{sc,g} = C_{sc}h|R|/|u/h| */
void calculateNumericalDiffusionResGradJuanes(proteus_index_t nElements_global,
					      int nQuadraturePoints_element,
					      int nSpace,
					      double shockCapturingDiffusion,
					      double uSC,
					      double* elementDiameter,
					      double* strong_residual,
					      double* grad_u,
					      double* numDiff)
{
  int eN,k;
  double h,
    n_R_eN,
    num,
    den,
    maxNumDiff;
  for(eN=0;eN<nElements_global;eN++)
    {
      h = elementDiameter[eN];
      n_R_eN=0.0;
      for (k=0;k<nQuadraturePoints_element;k++)
	{
          num = shockCapturingDiffusion*0.5*h*fabs(strong_residual[eN*nQuadraturePoints_element+k]);
          den = fabs(uSC)/h;
          if (den > num*1.0e-8) 
            numDiff[eN*nQuadraturePoints_element+k] = num/den;
          else
            numDiff[eN*nQuadraturePoints_element+k]=0.0;
          n_R_eN=fmax(n_R_eN,fabs(strong_residual[eN*nQuadraturePoints_element+k]));
        }
      maxNumDiff=0.0;
      for (k=0;k<nQuadraturePoints_element;k++)
        {
          num = shockCapturingDiffusion*0.5*h*n_R_eN;
          den = fabs(uSC)/h;
          if (den > num*1.0e-8)
            numDiff[eN*nQuadraturePoints_element+k] = num/den;
          else
            numDiff[eN*nQuadraturePoints_element+k]=0.0;
          maxNumDiff = fmax(maxNumDiff,numDiff[eN*nQuadraturePoints_element+k]);
        }
      for (k=0;k<nQuadraturePoints_element;k++)
         numDiff[eN*nQuadraturePoints_element+k] = maxNumDiff;
    }
}

/****
     supposed to be nu_i = nu_c h^{2-beta}|R_i|
 ***/

void calculateNumericalDiffusionJaffre(proteus_index_t nElements_global,
				       int nQuadraturePoints_element,
				       int nSpace,
				       double shockCapturingDiffusion,
				       double beta,
				       double* elementDiameter,
				       double* strong_residual,
				       double* grad_u,
				       double* numDiff)
{
  int eN,k,I;
  double h,
    num,
    hfactor,
    n_grad_u[nQuadraturePoints_element],
    n_R_eN,
    maxNumDiff;
  
  for(eN=0;eN<nElements_global;eN++)
    {
      h = elementDiameter[eN];
      hfactor = pow(h,2.0-beta);
      for (k=0;k<nQuadraturePoints_element;k++)
        {
          num = shockCapturingDiffusion*hfactor*fabs(strong_residual[eN*nQuadraturePoints_element+k]);
	  numDiff[eN*nQuadraturePoints_element+k] = num;
        }
      
      maxNumDiff=0.0;
      for (k=0;k<nQuadraturePoints_element;k++)
        {
          maxNumDiff = fmax(maxNumDiff,numDiff[eN*nQuadraturePoints_element+k]);
        }
      for (k=0;k<nQuadraturePoints_element;k++)
         numDiff[eN*nQuadraturePoints_element+k] = maxNumDiff;
    }
}


/** @} */
