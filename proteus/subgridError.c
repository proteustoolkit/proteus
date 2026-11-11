#include "subgridError.h"

/**
   \file subgridError.c
   \ingroup subgridError
   @{
*/

/**
   \brief Calculate the ASGS subgrid error given tau and the strong residual
*/
void calculateSubgridError_tauRes(proteus_index_t nElements_global,
                                  int nQuadraturePoints_element,
                                  int nDOF_trial_element,
                                  double* tau,
                                  double* pdeResidual,
                                  double* dpdeResidual,
                                  double* subgridError,
                                  double* dsubgridError)
{
  int eN,k,j;
   for(eN=0;eN<nElements_global;eN++)
     for (k=0;k<nQuadraturePoints_element;k++)
       {
         subgridError[eN*nQuadraturePoints_element+
                      k] = 
           -tau[eN*nQuadraturePoints_element+
               k]
           *
           pdeResidual[eN*nQuadraturePoints_element+
                       k];
         for (j=0;j<nDOF_trial_element;j++)
           dsubgridError[eN*nQuadraturePoints_element*nDOF_trial_element+
                         k*nDOF_trial_element+
                         j]
             =
             -tau[eN*nQuadraturePoints_element+
                 k]
             *
             dpdeResidual[eN*nQuadraturePoints_element*nDOF_trial_element+
                          k*nDOF_trial_element+
                          j];
       }
}

/**
   \brief Calculate the stabilization parameter for the scalar advection-diffusion-reaction equation using the peclet  number formula
*/
void calculateSubgridError_ADR_tau_p(proteus_index_t nElements_global,
                                     int nQuadraturePoints_element,
                                     int nSpace,
                                     double* elementDiameter,
                                     double* dmt,
                                     double* df,
                                     double* a,
                                     double* da,
                                     double* grad_phi,
                                     double* dphi,
                                     double* dr,
                                     double* pe,
                                     double* cfl,
                                     double* tau)
{
  proteus_index_t nSpace2=nSpace*nSpace; int eN, k, I, J;
  double h,Vlin,Alin,A_II,num,den,cfl1,cfl2,vlin[3],alpha,beta;
  for(eN=0;eN<nElements_global;eN++)
    {
      h = elementDiameter[eN];
      for (k=0;k<nQuadraturePoints_element;k++)
        {
          Vlin = 0.0;
          Alin = 0.0;
          for(I=0;I<nSpace;I++)
            {
              vlin[I] = df[eN*nQuadraturePoints_element*nSpace + 
                           k*nSpace + 
                           I];
              for(J=0;J<nSpace;J++)
                vlin[I] 
                  -= 
                  da[eN*nQuadraturePoints_element*nSpace2 + 
                     k*nSpace2 + 
                     I*nSpace + 
                     J]
                  *
                  grad_phi[eN*nQuadraturePoints_element*nSpace + 
                           k*nSpace + 
                           J];
            }
          for(I=0;I<nSpace;I++)
            Vlin += vlin[I]*vlin[I];
          Vlin = sqrt(Vlin);
          /*max diagonal entry for A. This choice needs to be looked into*/
          for(I=0;I<nSpace;I++)
            {
              A_II = a[eN*nQuadraturePoints_element*nSpace2 + 
                       k*nSpace2 + 
                       I*nSpace + 
                       I];
              Alin = (A_II > Alin) ? A_II : Alin;
            }
          Alin*=dphi[eN*nQuadraturePoints_element + 
                     k];
          cfl1 = Vlin/h;
          cfl2 = Alin/(h*h);
          if (cfl1 > cfl2)
            cfl[eN*nQuadraturePoints_element + 
                k] = cfl1;
          else
            cfl[eN*nQuadraturePoints_element + 
                k] = cfl2;
          num = 0.5*Vlin*h + 1.0e-8;
          den = Alin + num*1.0e-8;
          pe[eN*nQuadraturePoints_element + 
             k] = num/den;
          tau[eN*nQuadraturePoints_element + k]=0.5*h*(
                                                       1.0/tanh(pe[eN*nQuadraturePoints_element+
                                                                   k]) -
                                                       1.0/pe[eN*nQuadraturePoints_element+
                                                              k])/(Vlin+1.0e-8);
/*           alpha = pe[eN*nQuadraturePoints_element + */
/*                      k]; */
/*           beta = Vlin; */
/*           num = h*h*h*(-3.0*exp(2.0*alpha)/alpha+exp(2.0*alpha)+3.0*exp(2.0*alpha)/(alpha*alpha) - 3.0/(alpha*alpha) - 1.0 - 1.0/alpha); */
/*           den = 72.0*beta*(exp(2.0*alpha) - exp(2.0*alpha)/alpha + 1.0 + 1.0/alpha); */
/*           tau[eN*nQuadraturePoints_element + k]=num/(den+1.0e-8); */
/*           printf("tau %12.5e \n",tau[eN*nQuadraturePoints_element+k]); */
        }
    }
}

void calculateSubgridError_ADR_tau_p_sd(proteus_index_t nElements_global,
					int nQuadraturePoints_element,
					int nSpace,
					proteus_csr_index_t* rowptr,
					proteus_csr_index_t* colind,
					double* elementDiameter,
					double* dmt,
					double* df,
					double* a,
					double* da,
					double* grad_phi,
					double* dphi,
					double* dr,
					double* pe,
					double* cfl,
					double* tau)
{
  proteus_csr_index_t m; proteus_csr_index_t nnz=rowptr[nSpace]; int eN, k, I, J;
  double h,Vlin,Alin,A_II,num,den,cfl1,cfl2,vlin[3],alpha,beta;
  for(eN=0;eN<nElements_global;eN++)
    {
      h = elementDiameter[eN];
      for (k=0;k<nQuadraturePoints_element;k++)
        {
          Vlin = 0.0;
          Alin = 0.0;
          for(I=0;I<nSpace;I++)
            {
              vlin[I] = df[eN*nQuadraturePoints_element*nSpace + 
                           k*nSpace + 
                           I];
              for(J=0;J<nSpace;J++)
                vlin[I] 
                  -= 
                  da[eN*nQuadraturePoints_element*nnz+
                     k*nnz+
		     m]
                  *
                  grad_phi[eN*nQuadraturePoints_element*nSpace + 
                           k*nSpace + 
                           colind[m]];
            }
          for(I=0;I<nSpace;I++)
            Vlin += vlin[I]*vlin[I];
          Vlin = sqrt(Vlin);
          /*max diagonal entry for A. This choice needs to be looked into*/
          for(I=0;I<nSpace;I++)
            {
	      for(m=rowptr[I];m<rowptr[I+1];m++)
		{
		  if (I==colind[m])
		    {
		      A_II = a[eN*nQuadraturePoints_element*nnz+
			       k*nnz+
			       m];
		      Alin = (A_II > Alin) ? A_II : Alin;
		    }
		}
	    }
          Alin*=dphi[eN*nQuadraturePoints_element + 
                     k];
          cfl1 = Vlin/h;
          cfl2 = Alin/(h*h);
          if (cfl1 > cfl2)
            cfl[eN*nQuadraturePoints_element + 
                k] = cfl1;
          else
            cfl[eN*nQuadraturePoints_element + 
                k] = cfl2;
          num = 0.5*Vlin*h + 1.0e-8;
          den = Alin + num*1.0e-8;
          pe[eN*nQuadraturePoints_element + 
             k] = num/den;
          tau[eN*nQuadraturePoints_element + k]=0.5*h*(
                                                       1.0/tanh(pe[eN*nQuadraturePoints_element+
                                                                   k]) -
                                                       1.0/pe[eN*nQuadraturePoints_element+
                                                              k])/(Vlin+1.0e-8);
/*           alpha = pe[eN*nQuadraturePoints_element + */
/*                      k]; */
/*           beta = Vlin; */
/*           num = h*h*h*(-3.0*exp(2.0*alpha)/alpha+exp(2.0*alpha)+3.0*exp(2.0*alpha)/(alpha*alpha) - 3.0/(alpha*alpha) - 1.0 - 1.0/alpha); */
/*           den = 72.0*beta*(exp(2.0*alpha) - exp(2.0*alpha)/alpha + 1.0 + 1.0/alpha); */
/*           tau[eN*nQuadraturePoints_element + k]=num/(den+1.0e-8); */
/*           printf("tau %12.5e \n",tau[eN*nQuadraturePoints_element+k]); */
        }
    }
}

/**
   \brief Calculate the stabilization parameter for the scalar advection-diffusion-reaction equation using the "l_1 norm formula"
*/
void calculateSubgridError_ADR_tau_1(proteus_index_t nElements_global,
				     int nQuadraturePoints_element,
				     int nSpace,
                                      double* elementDiameter,
                                      double* dmt,
                                      double* df,
                                      double* a,
                                      double* da,
                                      double* grad_phi,
                                      double* dphi,
                                      double* dr,
                                      double* pe,
                                      double* cfl,
                                      double* tau)
{
  proteus_index_t nSpace2=nSpace*nSpace; int eN, k, I, J;
  double h,Vlin,Alin,A_II,num,den,cfl1,cfl2,vlin[3],tauMax;
  for(eN=0;eN<nElements_global;eN++)
    {
      h = elementDiameter[eN];
      tauMax=0.0;
      for (k=0;k<nQuadraturePoints_element;k++)
        {
          Vlin = 0.0;
          Alin = 0.0;
          for(I=0;I<nSpace;I++)
            {
              vlin[I] = df[eN*nQuadraturePoints_element*nSpace + 
                           k*nSpace + 
                           I];
              for(J=0;J<nSpace;J++)
                vlin[I] 
                  += 
                  da[eN*nQuadraturePoints_element*nSpace2 + 
                     k*nSpace2 + 
                     I*nSpace + 
                     J]
                  *
                  grad_phi[eN*nQuadraturePoints_element*nSpace + 
                           k*nSpace + 
                           J];
            }
          for(I=0;I<nSpace;I++)
            Vlin += vlin[I]*vlin[I];
          Vlin = sqrt(Vlin);
          /*max diagonal entry for A. This choice needs to be looked into*/
          for(I=0;I<nSpace;I++)
            {
              A_II = a[eN*nQuadraturePoints_element*nSpace2 + 
                       k*nSpace2 + 
                       I*nSpace + 
                       I];
              Alin = (A_II > Alin) ? A_II : Alin;
            }
          Alin*=dphi[eN*nQuadraturePoints_element + 
                     k];
          cfl1 = Vlin/h;
          cfl2 = Alin/(h*h);
          if (cfl1 > cfl2)
            cfl[eN*nQuadraturePoints_element + 
                k] = cfl1;
          else
            cfl[eN*nQuadraturePoints_element + 
                k] = cfl2;
          num = 0.5*Vlin*h + 1.0e-8;
          den = Alin + num*1.0e-8;
          pe[eN*nQuadraturePoints_element + 
             k] = num/den;
          tau[eN*nQuadraturePoints_element + k]=1.0/((2.0*Vlin/h)+ 
                                                     (4.0*Alin/(h*h)) +
                                                     fabs(dr[eN*nQuadraturePoints_element + 
                                                             k])+
                                                     fabs(dmt[eN*nQuadraturePoints_element + 
                                                              k])+1.0e-8);
/*           tauMax=fmax(tauMax,tau[eN*nQuadraturePoints_element + k]); */
        }
/*       for (k=0;k<nQuadraturePoints_element;k++) */
/*         tau[eN*nQuadraturePoints_element + k]=tauMax; */
    }
}
void calculateSubgridError_ADR_tau_1_sd(proteus_index_t nElements_global,
					int nQuadraturePoints_element,
					int nSpace,
					proteus_csr_index_t* rowptr,
					proteus_csr_index_t* colind,
					double* elementDiameter,
					double* dmt,
					double* df,
					double* a,
					double* da,
					double* grad_phi,
					double* dphi,
					double* dr,
					double* pe,
					double* cfl,
					double* tau)
{
  proteus_csr_index_t m; proteus_csr_index_t nnz=rowptr[nSpace]; int eN, k, I;
  double h,Vlin,Alin,A_II,num,den,cfl1,cfl2,vlin[3],tauMax;
  for(eN=0;eN<nElements_global;eN++)
    {
      h = elementDiameter[eN];
      tauMax=0.0;
      for (k=0;k<nQuadraturePoints_element;k++)
        {
          Vlin = 0.0;
          Alin = 0.0;
          for(I=0;I<nSpace;I++)
            {
              vlin[I] = df[eN*nQuadraturePoints_element*nSpace + 
                           k*nSpace + 
                           I];
              for(m=rowptr[I];m<rowptr[I+1];m++)
                vlin[I] 
                  += 
                  da[eN*nQuadraturePoints_element*nnz+
                     k*nnz+
                     m]
                  *
                  grad_phi[eN*nQuadraturePoints_element*nSpace + 
                           k*nSpace + 
                           colind[m]];
            }
          for(I=0;I<nSpace;I++)
            Vlin += vlin[I]*vlin[I];
          Vlin = sqrt(Vlin);
          /*max diagonal entry for A. This choice needs to be looked into*/
          for(I=0;I<nSpace;I++)
            {
	      for(m=rowptr[I];m<rowptr[I+1];m++)
		{
		  if(I==colind[m])
		    {
		      A_II = a[eN*nQuadraturePoints_element*nnz+
			       k*nnz+
			       m];
		      Alin = (A_II > Alin) ? A_II : Alin;
		    }
		}
            }
          Alin*=dphi[eN*nQuadraturePoints_element + 
                     k];
          cfl1 = Vlin/h;
          cfl2 = Alin/(h*h);
          if (cfl1 > cfl2)
            cfl[eN*nQuadraturePoints_element + 
                k] = cfl1;
          else
            cfl[eN*nQuadraturePoints_element + 
                k] = cfl2;
          num = 0.5*Vlin*h + 1.0e-8;
          den = Alin + num*1.0e-8;
          pe[eN*nQuadraturePoints_element + 
             k] = num/den;
          tau[eN*nQuadraturePoints_element + k]=1.0/((2.0*Vlin/h)+ 
                                                     (4.0*Alin/(h*h)) +
                                                     fabs(dr[eN*nQuadraturePoints_element + 
                                                             k])+
                                                     fabs(dmt[eN*nQuadraturePoints_element + 
                                                              k])+1.0e-8);
/*           tauMax=fmax(tauMax,tau[eN*nQuadraturePoints_element + k]); */
        }
/*       for (k=0;k<nQuadraturePoints_element;k++) */
/*         tau[eN*nQuadraturePoints_element + k]=tauMax; */
    }
}

/**
   \brief Calculate the stabilization parameter for the scalar advection-diffusion-reaction equation using the "l_2 norm" formula
*/
void calculateSubgridError_ADR_tau_2(proteus_index_t nElements_global,
                                     int nQuadraturePoints_element,
                                     int nSpace,
                                     double* elementDiameter,
                                     double* dmt,
                                     double* df,
                                     double* a,
                                     double* da,
                                     double* grad_phi,
                                     double* dphi,
                                     double* dr,
                                     double* pe,
                                     double* cfl,
                                     double* tau)
{
  proteus_index_t nSpace2=nSpace*nSpace; int eN, k, I, J;
  double h,Vlin,Alin,A_II,num,den,cfl1,cfl2,vlin[3];
  for(eN=0;eN<nElements_global;eN++)
    {
      h = elementDiameter[eN];
      for (k=0;k<nQuadraturePoints_element;k++)
        {
          Vlin = 0.0;
          Alin = 0.0;
          for(I=0;I<nSpace;I++)
            {
              vlin[I] = df[eN*nQuadraturePoints_element*nSpace + 
                           k*nSpace + 
                           I];
              for(J=0;J<nSpace;J++)
                {
                  vlin[I] 
                    -= 
                    da[eN*nQuadraturePoints_element*nSpace2 + 
                       k*nSpace2 + 
                       I*nSpace + 
                       J]
                    *
                    grad_phi[eN*nQuadraturePoints_element*nSpace + 
                             k*nSpace + 
                             J];
                }
            }
          for(I=0;I<nSpace;I++)
            Vlin += vlin[I]*vlin[I];
          Vlin = sqrt(Vlin);
          /*max diagonal entry for A. This choice needs to be looked into*/
          for(I=0;I<nSpace;I++)
            {
              A_II = a[eN*nQuadraturePoints_element*nSpace2 + 
                       k*nSpace2 + 
                       I*nSpace + 
                       I];
              Alin = (A_II > Alin) ? A_II : Alin;
            }
          Alin*=dphi[eN*nQuadraturePoints_element + 
                     k];
          cfl1 = Vlin/h;
          cfl2 = Alin/(h*h);
          if (cfl1 > cfl2)
            cfl[eN*nQuadraturePoints_element + 
                k] = cfl1;
          else
            cfl[eN*nQuadraturePoints_element + 
                k] = cfl2;
          num = 0.5*Vlin*h + 1.0e-8;
          den = Alin + num*1.0e-8;
/*           num = 0.5*Vlin*h; */
/*           den = Alin; */
          tau[eN*nQuadraturePoints_element + 
              k]
            =1.0/sqrt((2.0*Vlin/h)*(2.0*Vlin/h)+ 
                      9.0*(4.0*Alin/(h*h))*(4.0*Alin/(h*h)) +
                      dr[eN*nQuadraturePoints_element + 
                         k]
                      *
                      dr[eN*nQuadraturePoints_element + 
                         k]
                      +
                      dmt[eN*nQuadraturePoints_element + 
                          k]
                      *
                      dmt[eN*nQuadraturePoints_element + 
                          k]+1.0e-8);
          pe[eN*nQuadraturePoints_element + 
             k] = num/den;
        }
    }
}

void calculateSubgridError_ADR_tau_2_sd(proteus_index_t nElements_global,
					int nQuadraturePoints_element,
					int nSpace,
					proteus_csr_index_t* rowptr,
					proteus_csr_index_t* colind,
					double* elementDiameter,
					double* dmt,
					double* df,
					double* a,
					double* da,
					double* grad_phi,
					double* dphi,
					double* dr,
					double* pe,
					double* cfl,
					double* tau)
{
  proteus_csr_index_t m; proteus_csr_index_t nnz=rowptr[nSpace]; int eN, k, I;
  double h,Vlin,Alin,A_II,num,den,cfl1,cfl2,vlin[3];
  for(eN=0;eN<nElements_global;eN++)
    {
      h = elementDiameter[eN];
      for (k=0;k<nQuadraturePoints_element;k++)
        {
          Vlin = 0.0;
          Alin = 0.0;
          for(I=0;I<nSpace;I++)
            {
              vlin[I] = df[eN*nQuadraturePoints_element*nSpace + 
                           k*nSpace + 
                           I];
              for(m=rowptr[I];m<rowptr[I+1];m++)
                {
                  vlin[I] 
                    -= 
                    da[eN*nQuadraturePoints_element*nnz+
                       k*nnz+
		       m]
                    *
                    grad_phi[eN*nQuadraturePoints_element*nSpace + 
                             k*nSpace + 
                             colind[m]];
                }
            }
          for(I=0;I<nSpace;I++)
            Vlin += vlin[I]*vlin[I];
          Vlin = sqrt(Vlin);
          /*max diagonal entry for A. This choice needs to be looked into*/
          for(I=0;I<nSpace;I++)
            {
	      for(m=rowptr[I];m<rowptr[I+1];m++)
		{
		  if(I==colind[m])
		    {
		      A_II = a[eN*nQuadraturePoints_element*nnz+
			       k*nnz+
			       m];
		      Alin = (A_II > Alin) ? A_II : Alin;
		    }
		}
            }
          Alin*=dphi[eN*nQuadraturePoints_element + 
                     k];
          cfl1 = Vlin/h;
          cfl2 = Alin/(h*h);
          if (cfl1 > cfl2)
            cfl[eN*nQuadraturePoints_element + 
                k] = cfl1;
          else
            cfl[eN*nQuadraturePoints_element + 
                k] = cfl2;
          num = 0.5*Vlin*h + 1.0e-8;
          den = Alin + num*1.0e-8;
/*           num = 0.5*Vlin*h; */
/*           den = Alin; */
          tau[eN*nQuadraturePoints_element + 
              k]
            =1.0/sqrt((2.0*Vlin/h)*(2.0*Vlin/h)+ 
                      9.0*(4.0*Alin/(h*h))*(4.0*Alin/(h*h)) +
                      dr[eN*nQuadraturePoints_element + 
                         k]
                      *
                      dr[eN*nQuadraturePoints_element + 
                         k]
                      +
                      dmt[eN*nQuadraturePoints_element + 
                          k]
                      *
                      dmt[eN*nQuadraturePoints_element + 
                          k]+1.0e-8);
          pe[eN*nQuadraturePoints_element + 
             k] = num/den;
        }
    }
}

void calculateSubgridError_ADR_generic_tau(proteus_index_t nElements_global,
                                           int nQuadraturePoints_element,
                                           int nSpace,
                                           double* inverseJ,
                                           double* dmt,
                                           double* df,
                                           double* a,
                                           double* da,
                                           double* grad_phi,
                                           double* dphi,
                                           double* dr,
                                           double* pe,
                                           double* cfl,
                                           double* tau)
{
  proteus_index_t nSpace2=nSpace*nSpace; int eN, k, I, J, K;
  double Vlin,Alin,A_II,cfl1,cfl2,vlin[3],G_IJ,v_dot_Gv,CI_nu2_G_ddot_G,Gv_I,CI=36.0*36.0,g_I,g_dot_g;
  for(eN=0;eN<nElements_global;eN++)
    {
      for (k=0;k<nQuadraturePoints_element;k++)
        {
          Vlin = 0.0;
          Alin = 0.0;
          for(I=0;I<nSpace;I++)
            {
              vlin[I] = df[eN*nQuadraturePoints_element*nSpace +
                           k*nSpace +
                           I];
              for(J=0;J<nSpace;J++)
                {
                  vlin[I]
                    -=
                    da[eN*nQuadraturePoints_element*nSpace2 +
                       k*nSpace2 +
                       I*nSpace +
                       J]
                    *
                    grad_phi[eN*nQuadraturePoints_element*nSpace +
                             k*nSpace +
                             J];
                }
            }
          v_dot_Gv=0.0;
          CI_nu2_G_ddot_G=0.0;
          g_dot_g=0.0;
          for (I=0;I<nSpace;I++)
            {
              Gv_I=0.0;
              g_I=0.0;
              for (J=0;J<nSpace;J++)
                {
                  G_IJ=0.0;
                  for (K=0;K<nSpace;K++)
                    G_IJ +=inverseJ[eN*nQuadraturePoints_element*nSpace2+
                                    k*nSpace2+
                                    K*nSpace+
                                    I]
                      *
                      inverseJ[eN*nQuadraturePoints_element*nSpace2+
                               k*nSpace2+
                               K*nSpace+
                               J];
                  Gv_I += G_IJ
                    *
                    vlin[J];
                  CI_nu2_G_ddot_G+=G_IJ*G_IJ;
                  g_I+=inverseJ[eN*nQuadraturePoints_element*nSpace2+
                                k*nSpace2+
                                J*nSpace+
                                I];
                }
              g_dot_g += g_I*g_I;
              v_dot_Gv += vlin[I]
                *
                Gv_I;
              
            }
          /*max diagonal entry for A. This choice needs to be looked into*/
          for(I=0;I<nSpace;I++)
            {
              A_II = a[eN*nQuadraturePoints_element*nSpace2 +
                       k*nSpace2 +
                       I*nSpace +
                       I];
              Alin = (A_II > Alin) ? A_II : Alin;
            }
          Alin*=dphi[eN*nQuadraturePoints_element +
                     k];
          CI_nu2_G_ddot_G*=CI*Alin*Alin;
          cfl1 = 0.5*sqrt(v_dot_Gv);
          cfl2 = 0.25*sqrt(CI_nu2_G_ddot_G);
          if (cfl1 > cfl2)
            cfl[eN*nQuadraturePoints_element +
                k] = cfl1;
          else
            cfl[eN*nQuadraturePoints_element +
                k] = cfl2;
          tau[eN*nQuadraturePoints_element +
              k]
            =1.0/sqrt(v_dot_Gv +
                      CI_nu2_G_ddot_G +
                      4.0*dr[eN*nQuadraturePoints_element +
                             k]
                      *
                      dr[eN*nQuadraturePoints_element +
                         k]
                      +
                      4.0*dmt[eN*nQuadraturePoints_element +
                              k]
                      *
                      dmt[eN*nQuadraturePoints_element +
                          k]+1.0e-8);
/*           printf("tau %12.5e \n",tau[eN*nQuadraturePoints_element+k]); */
          pe[eN*nQuadraturePoints_element +
             k] = sqrt(v_dot_Gv)/sqrt(CI_nu2_G_ddot_G);
        }
    }
}

void calculateSubgridError_ADR_generic_tau_sd(proteus_index_t nElements_global,
					      int nQuadraturePoints_element,
					      int nSpace,
					      proteus_csr_index_t* rowptr,
					      proteus_csr_index_t* colind,
					      double* inverseJ,
					      double* dmt,
					      double* df,
					      double* a,
					      double* da,
					      double* grad_phi,
					      double* dphi,
					      double* dr,
					      double* pe,
					      double* cfl,
					      double* tau)
{
  proteus_index_t  nSpace2=nSpace*nSpace; proteus_csr_index_t m; proteus_csr_index_t nnz=rowptr[nSpace]; int eN, k, I, J, K;
  double Vlin,Alin,A_II,cfl1,cfl2,vlin[3],G_IJ,v_dot_Gv,CI_nu2_G_ddot_G,Gv_I,CI=36.0*36.0,g_I,g_dot_g;
  for(eN=0;eN<nElements_global;eN++)
    {
      for (k=0;k<nQuadraturePoints_element;k++)
        {
          Vlin = 0.0;
          Alin = 0.0;
          for(I=0;I<nSpace;I++)
            {
              vlin[I] = df[eN*nQuadraturePoints_element*nSpace +
                           k*nSpace +
                           I];
              for(m=rowptr[I];m<rowptr[I+1];m++)
                {
                  vlin[I]
                    -=
                    da[eN*nQuadraturePoints_element*nnz+
                       k*nnz+
		       m]
                    *
                    grad_phi[eN*nQuadraturePoints_element*nSpace +
                             k*nSpace +
                             colind[m]];
                }
            }
          v_dot_Gv=0.0;
          CI_nu2_G_ddot_G=0.0;
          g_dot_g=0.0;
          for (I=0;I<nSpace;I++)
            {
              Gv_I=0.0;
              g_I=0.0;
              for (J=0;J<nSpace;J++)
                {
                  G_IJ=0.0;
                  for (K=0;K<nSpace;K++)
                    G_IJ +=inverseJ[eN*nQuadraturePoints_element*nSpace2+
                                    k*nSpace2+
                                    K*nSpace+
                                    I]
                      *
                      inverseJ[eN*nQuadraturePoints_element*nSpace2+
                               k*nSpace2+
                               K*nSpace+
                               J];
                  Gv_I += G_IJ
                    *
                    vlin[J];
                  CI_nu2_G_ddot_G+=G_IJ*G_IJ;
                  g_I+=inverseJ[eN*nQuadraturePoints_element*nSpace2+
                                k*nSpace2+
                                J*nSpace+
                                I];
                }
              g_dot_g += g_I*g_I;
              v_dot_Gv += vlin[I]
                *
                Gv_I;
              
            }
          /*max diagonal entry for A. This choice needs to be looked into*/
          for(I=0;I<nSpace;I++)
            {
	      for(m=rowptr[I];m<rowptr[I+1];m++)
		{
		  if (I==colind[m])
		    {
		      A_II = a[eN*nQuadraturePoints_element*nnz +
			       k*nnz + m];
		      Alin = (A_II > Alin) ? A_II : Alin;
		    }
		}
            }
          Alin*=dphi[eN*nQuadraturePoints_element +
                     k];
          CI_nu2_G_ddot_G*=CI*Alin*Alin;
          cfl1 = 0.5*sqrt(v_dot_Gv);
          cfl2 = 0.25*sqrt(CI_nu2_G_ddot_G);
          if (cfl1 > cfl2)
            cfl[eN*nQuadraturePoints_element +
                k] = cfl1;
          else
            cfl[eN*nQuadraturePoints_element +
                k] = cfl2;
          tau[eN*nQuadraturePoints_element +
              k]
            =1.0/sqrt(v_dot_Gv +
                      CI_nu2_G_ddot_G +
                      4.0*dr[eN*nQuadraturePoints_element +
                             k]
                      *
                      dr[eN*nQuadraturePoints_element +
                         k]
                      +
                      4.0*dmt[eN*nQuadraturePoints_element +
                              k]
                      *
                      dmt[eN*nQuadraturePoints_element +
                          k]+1.0e-8);
/*           printf("tau %12.5e \n",tau[eN*nQuadraturePoints_element+k]); */
          pe[eN*nQuadraturePoints_element +
             k] = sqrt(v_dot_Gv)/sqrt(CI_nu2_G_ddot_G);
        }
    }
}

/**
   \brief Calculate the stabilization parameter for the scalar advection-diffusion-reaction equation
*/
void calculateSubgridError_ADR_tau(proteus_index_t nElements_global,
                                   int nQuadraturePoints_element,
                                   int nSpace,
                                   char stabilization,
                                   double* elementDiameter,
                                   double* dmt,
                                   double* df,
                                   double* a,
                                   double* da,
                                   double* grad_phi,
                                   double* dphi,
                                   double* dr,
                                   double* pe,
                                   double* cfl,
                                   double* tau)
{
  if(stabilization == '2')
    calculateSubgridError_ADR_tau_2(nElements_global, nQuadraturePoints_element, nSpace, elementDiameter, dmt, df, a, da, grad_phi, dphi, dr, pe, cfl, tau);
  else if(stabilization == '1')
    calculateSubgridError_ADR_tau_1(nElements_global, nQuadraturePoints_element, nSpace, elementDiameter, dmt, df, a, da, grad_phi, dphi, dr, pe, cfl, tau);
  else if(stabilization == 'p')
    calculateSubgridError_ADR_tau_p(nElements_global, nQuadraturePoints_element, nSpace, elementDiameter, dmt, df, a, da, grad_phi, dphi, dr, pe, cfl, tau);
	
}

void calculateSubgridError_ADR_tau_sd(proteus_index_t nElements_global,
				      int nQuadraturePoints_element,
				      int nSpace,
				      proteus_csr_index_t* rowptr,
				      proteus_csr_index_t* colind,
				      char stabilization,
				      double* elementDiameter,
				      double* dmt,
				      double* df,
				      double* a,
				      double* da,
				      double* grad_phi,
				      double* dphi,
				      double* dr,
				      double* pe,
				      double* cfl,
				      double* tau)
{
  if(stabilization == '2')
    calculateSubgridError_ADR_tau_2_sd(nElements_global, nQuadraturePoints_element, nSpace, rowptr,colind,elementDiameter, dmt, df, a, da, grad_phi, dphi, dr, pe, cfl, tau);
  else if(stabilization == '1')
    calculateSubgridError_ADR_tau_1_sd(nElements_global, nQuadraturePoints_element, nSpace, rowptr,colind,elementDiameter, dmt, df, a, da, grad_phi, dphi, dr, pe, cfl, tau);
  else if(stabilization == 'p')
    calculateSubgridError_ADR_tau_p_sd(nElements_global, nQuadraturePoints_element, nSpace, rowptr,colind,elementDiameter, dmt, df, a, da, grad_phi, dphi, dr, pe, cfl, tau);
}

/**
   \brief Calculate the stabilization parameter for the scalar advection-diffusion-reaction equation using the "l_2 norm" formula
*/
void calculateSubgridError_A_tau_1(proteus_index_t nElements_global,
                                   int nQuadraturePoints_element,
                                   int nSpace,
                                   double* elementDiameter,
                                   double* dmt,
                                   double* df,
                                   double* cfl,
                                   double* tau)
{
  int eN,k,I;
  double h,Vlin;
  for(eN=0;eN<nElements_global;eN++)
    {
      h = elementDiameter[eN];
      for (k=0;k<nQuadraturePoints_element;k++)
        {
          Vlin = 0.0;
          for(I=0;I<nSpace;I++)
            {
              Vlin += 
                df[eN*nQuadraturePoints_element*nSpace + 
                   k*nSpace + 
                   I]
                *
                df[eN*nQuadraturePoints_element*nSpace + 
                   k*nSpace + 
                   I];
            }
          Vlin = sqrt(Vlin);
          cfl[eN*nQuadraturePoints_element + 
              k] = Vlin/h;
          tau[eN*nQuadraturePoints_element + 
              k]
            =1.0/((2.0*Vlin/h)+ 
                  fabs(dmt[eN*nQuadraturePoints_element + 
                           k])+1.0e-8);
        }
    }
}

/**
   \brief Calculate the stabilization parameter for the scalar advection-diffusion-reaction equation using the "l_2 norm" formula
*/
void calculateSubgridError_A_tau_2(proteus_index_t nElements_global,
                                     int nQuadraturePoints_element,
                                     int nSpace,
                                     double* elementDiameter,
                                     double* dmt,
                                     double* df,
                                     double* cfl,
                                     double* tau)
{
  int eN,k,I;
  double h,Vlin;
  for(eN=0;eN<nElements_global;eN++)
    {
      h = elementDiameter[eN];
      for (k=0;k<nQuadraturePoints_element;k++)
        {
          Vlin = 0.0;
          for(I=0;I<nSpace;I++)
            {
              Vlin += 
                df[eN*nQuadraturePoints_element*nSpace + 
                   k*nSpace + 
                   I]
                *
                df[eN*nQuadraturePoints_element*nSpace + 
                   k*nSpace + 
                   I];
            }
          Vlin = sqrt(Vlin);
          cfl[eN*nQuadraturePoints_element + 
              k] = Vlin/h;
          tau[eN*nQuadraturePoints_element + 
              k]
            =1.0/sqrt((2.0*Vlin/h)*(2.0*Vlin/h)+ 
                      dmt[eN*nQuadraturePoints_element + 
                          k]
                      *
                      dmt[eN*nQuadraturePoints_element + 
                          k]+1.0e-8);
/*           printf("tau[%i,%i] = %12.5e \n",eN,k,tau[eN*nQuadraturePoints_element +  */
/*                                                 k]); */
        }
    }
}

/**
   \brief Calculate the stabilization parameter for the scalar advection-diffusion-reaction equation
*/
void calculateSubgridError_A_tau(proteus_index_t nElements_global,
                                  int nQuadraturePoints_element,
                                  int nSpace,
                                  char stabilization,
                                  double* elementDiameter,
                                  double* dmt,
                                  double* df,
                                  double* cfl,
                                  double* tau)
{
  if(stabilization == '2')
    calculateSubgridError_A_tau_2(nElements_global, nQuadraturePoints_element, nSpace, elementDiameter, dmt, df, cfl, tau);
  else if(stabilization == '1')
    calculateSubgridError_A_tau_1(nElements_global, nQuadraturePoints_element, nSpace, elementDiameter, dmt, df, cfl, tau);
}

/**
   \brief Calculate the subgrid error for velocity in 2D Stokes equation with a GLS-like formula
*/
void calculateSubgridErrorStokes2D_GLS_velocity(proteus_index_t nElements_global,
                                                int nQuadraturePoints_element,
                                                int nDOF_trial_element,
                                                int nSpace,
                                                double* elementDiameter,
                                                double* a,
                                                double* pdeResidualU,
                                                double* dpdeResidualU_dp,
                                                double* dpdeResidualU_du,
                                                double* pdeResidualV,
                                                double* dpdeResidualV_dp,
                                                double* dpdeResidualV_dv,
                                                double* subgridErrorU,
                                                double* dsubgridErrorU_dp,
                                                double* dsubgridErrorU_du,
                                                double* subgridErrorV,
                                                double* dsubgridErrorV_dp,
                                                double* dsubgridErrorV_dv)
{
  proteus_index_t nSpace2=nSpace*nSpace; int eN, k, j;
  double h,viscosity,tau1;
  for(eN=0;eN<nElements_global;eN++)
    {
      h = elementDiameter[eN];
      for (k=0;k<nQuadraturePoints_element;k++)
        {
          viscosity =  a[eN*nQuadraturePoints_element*nSpace2 + 
                         k*nSpace2];
          tau1 = h*h/(12.0*viscosity);
          subgridErrorU[eN*nQuadraturePoints_element+
                       k] =
            tau1
            *
            pdeResidualU[eN*nQuadraturePoints_element+
                         k];
          subgridErrorV[eN*nQuadraturePoints_element+
                       k] = 
            tau1
            *
            pdeResidualV[eN*nQuadraturePoints_element+
                         k];
          for (j=0;j<nDOF_trial_element;j++)
            {
              /* u */
              dsubgridErrorU_dp[eN*nQuadraturePoints_element*nDOF_trial_element+
                                k*nDOF_trial_element+
                                j] =
                tau1
                *
                dpdeResidualU_dp[eN*nQuadraturePoints_element*nDOF_trial_element+
                                 k*nDOF_trial_element+
                                 j];
              dsubgridErrorU_du[eN*nQuadraturePoints_element*nDOF_trial_element+
                                k*nDOF_trial_element+
                                j] = 
                tau1
                *
                dpdeResidualU_du[eN*nQuadraturePoints_element*nDOF_trial_element+
                                 k*nDOF_trial_element+
                                 j];
              /* v */
              dsubgridErrorV_dp[eN*nQuadraturePoints_element*nDOF_trial_element+
                                k*nDOF_trial_element+
                                j] = 
                tau1
                *
                dpdeResidualV_dp[eN*nQuadraturePoints_element*nDOF_trial_element+
                                 k*nDOF_trial_element+
                                 j];
              dsubgridErrorV_dv[eN*nQuadraturePoints_element*nDOF_trial_element+
                                k*nDOF_trial_element+
                                j] = 
                tau1
                *
                dpdeResidualV_dv[eN*nQuadraturePoints_element*nDOF_trial_element+
                                 k*nDOF_trial_element+
                                 j];
             }
        }
    }
}

void calculateSubgridErrorStokes2D_GLS_velocity_sd(proteus_index_t nElements_global,
						   int nQuadraturePoints_element,
						   int nDOF_trial_element,
						   int nSpace,
						   double* elementDiameter,
						   double* a,
						   double* pdeResidualU,
						   double* dpdeResidualU_dp,
						   double* dpdeResidualU_du,
						   double* pdeResidualV,
						   double* dpdeResidualV_dp,
						   double* dpdeResidualV_dv,
						   double* subgridErrorU,
						   double* dsubgridErrorU_dp,
						   double* dsubgridErrorU_du,
						   double* subgridErrorV,
						   double* dsubgridErrorV_dp,
						   double* dsubgridErrorV_dv)
{
  proteus_index_t nSpace2=nSpace*nSpace; int eN, k, j;
  double h,viscosity,tau1;
  for(eN=0;eN<nElements_global;eN++)
    {
      h = elementDiameter[eN];
      for (k=0;k<nQuadraturePoints_element;k++)
        {
          viscosity =  a[eN*nQuadraturePoints_element*nSpace + 
                         k*nSpace];
          tau1 = h*h/(12.0*viscosity);
          subgridErrorU[eN*nQuadraturePoints_element+
                       k] =
            tau1
            *
            pdeResidualU[eN*nQuadraturePoints_element+
                         k];
          subgridErrorV[eN*nQuadraturePoints_element+
                       k] = 
            tau1
            *
            pdeResidualV[eN*nQuadraturePoints_element+
                         k];
          for (j=0;j<nDOF_trial_element;j++)
            {
              /* u */
              dsubgridErrorU_dp[eN*nQuadraturePoints_element*nDOF_trial_element+
                                k*nDOF_trial_element+
                                j] =
                tau1
                *
                dpdeResidualU_dp[eN*nQuadraturePoints_element*nDOF_trial_element+
                                 k*nDOF_trial_element+
                                 j];
              dsubgridErrorU_du[eN*nQuadraturePoints_element*nDOF_trial_element+
                                k*nDOF_trial_element+
                                j] = 
                tau1
                *
                dpdeResidualU_du[eN*nQuadraturePoints_element*nDOF_trial_element+
                                 k*nDOF_trial_element+
                                 j];
              /* v */
              dsubgridErrorV_dp[eN*nQuadraturePoints_element*nDOF_trial_element+
                                k*nDOF_trial_element+
                                j] = 
                tau1
                *
                dpdeResidualV_dp[eN*nQuadraturePoints_element*nDOF_trial_element+
                                 k*nDOF_trial_element+
                                 j];
              dsubgridErrorV_dv[eN*nQuadraturePoints_element*nDOF_trial_element+
                                k*nDOF_trial_element+
                                j] = 
                tau1
                *
                dpdeResidualV_dv[eN*nQuadraturePoints_element*nDOF_trial_element+
                                 k*nDOF_trial_element+
                                 j];
             }
        }
    }
}

/**
   \brief Calculate the subgrid error for velocity in 3D Stokes equation with a GLS-like formula
*/
void calculateSubgridErrorStokes3D_GLS_velocity(proteus_index_t nElements_global,
                                                int nQuadraturePoints_element,
                                                int nDOF_trial_element,
                                                int nSpace,
                                                double* elementDiameter,
                                                double* a,
                                                double* pdeResidualU,
                                                double* dpdeResidualU_dp,
                                                double* dpdeResidualU_du,
                                                double* pdeResidualV,
                                                double* dpdeResidualV_dp,
                                                double* dpdeResidualV_dv,
                                                double* pdeResidualW,
                                                double* dpdeResidualW_dp,
                                                double* dpdeResidualW_dw,
                                                double* subgridErrorU,
                                                double* dsubgridErrorU_dp,
                                                double* dsubgridErrorU_du,
                                                double* subgridErrorV,
                                                double* dsubgridErrorV_dp,
                                                double* dsubgridErrorV_dv,
                                                double* subgridErrorW,
                                                double* dsubgridErrorW_dp,
                                                double* dsubgridErrorW_dw)
{
  proteus_index_t nSpace2=nSpace*nSpace; int eN, k, j;
  double h,viscosity,tau1;
  for(eN=0;eN<nElements_global;eN++)
    {
      h = elementDiameter[eN];
      for (k=0;k<nQuadraturePoints_element;k++)
        {
          viscosity =  a[eN*nQuadraturePoints_element*nSpace2 + 
                         k*nSpace2]; 
          tau1 = h*h/(12.0*viscosity);
          subgridErrorU[eN*nQuadraturePoints_element+
                       k] =
            tau1
            *
            pdeResidualU[eN*nQuadraturePoints_element+
                         k];
          subgridErrorV[eN*nQuadraturePoints_element+
                       k] = 
            tau1
            *
            pdeResidualV[eN*nQuadraturePoints_element+
                         k];
          subgridErrorW[eN*nQuadraturePoints_element+
                       k] = 
            tau1
            *
            pdeResidualW[eN*nQuadraturePoints_element+
                         k];
          for (j=0;j<nDOF_trial_element;j++)
            {
              /* u */
              dsubgridErrorU_dp[eN*nQuadraturePoints_element*nDOF_trial_element+
                                k*nDOF_trial_element+
                                j] =
                tau1
                *
                dpdeResidualU_dp[eN*nQuadraturePoints_element*nDOF_trial_element+
                                 k*nDOF_trial_element+
                                 j];
              dsubgridErrorU_du[eN*nQuadraturePoints_element*nDOF_trial_element+
                                k*nDOF_trial_element+
                                j] = 
                tau1
                *
                dpdeResidualU_du[eN*nQuadraturePoints_element*nDOF_trial_element+
                                 k*nDOF_trial_element+
                                 j];
              /* v */
              dsubgridErrorV_dp[eN*nQuadraturePoints_element*nDOF_trial_element+
                                k*nDOF_trial_element+
                                j] = 
                tau1
                *
                dpdeResidualV_dp[eN*nQuadraturePoints_element*nDOF_trial_element+
                                 k*nDOF_trial_element+
                                 j];
              dsubgridErrorV_dv[eN*nQuadraturePoints_element*nDOF_trial_element+
                                k*nDOF_trial_element+
                                j] = 
                tau1
                *
                dpdeResidualV_dv[eN*nQuadraturePoints_element*nDOF_trial_element+
                                 k*nDOF_trial_element+
                                 j];
              /* w */
              dsubgridErrorW_dp[eN*nQuadraturePoints_element*nDOF_trial_element+
                                k*nDOF_trial_element+
                                j] = 
                tau1
                *
                dpdeResidualW_dp[eN*nQuadraturePoints_element*nDOF_trial_element+
                                 k*nDOF_trial_element+
                                 j];
              dsubgridErrorW_dw[eN*nQuadraturePoints_element*nDOF_trial_element+
                                k*nDOF_trial_element+
                                j] = 
                tau1
                *
                dpdeResidualW_dw[eN*nQuadraturePoints_element*nDOF_trial_element+
                                 k*nDOF_trial_element+
                                 j];
             }
        }
    }
}

void calculateSubgridErrorStokes3D_GLS_velocity_sd(proteus_index_t nElements_global,
						   int nQuadraturePoints_element,
						   int nDOF_trial_element,
						   int nSpace,
						   double* elementDiameter,
						   double* a,
						   double* pdeResidualU,
						   double* dpdeResidualU_dp,
						   double* dpdeResidualU_du,
						   double* pdeResidualV,
						   double* dpdeResidualV_dp,
						   double* dpdeResidualV_dv,
						   double* pdeResidualW,
						   double* dpdeResidualW_dp,
						   double* dpdeResidualW_dw,
						   double* subgridErrorU,
						   double* dsubgridErrorU_dp,
						   double* dsubgridErrorU_du,
						   double* subgridErrorV,
						   double* dsubgridErrorV_dp,
						   double* dsubgridErrorV_dv,
						   double* subgridErrorW,
						   double* dsubgridErrorW_dp,
						   double* dsubgridErrorW_dw)
{
  proteus_index_t nSpace2=nSpace*nSpace; int eN, k, j;
  double h,viscosity,tau1;
  for(eN=0;eN<nElements_global;eN++)
    {
      h = elementDiameter[eN];
      for (k=0;k<nQuadraturePoints_element;k++)
        {
          viscosity =  a[eN*nQuadraturePoints_element*nSpace + 
                         k*nSpace]; 
          tau1 = h*h/(12.0*viscosity);
          subgridErrorU[eN*nQuadraturePoints_element+
                       k] =
            tau1
            *
            pdeResidualU[eN*nQuadraturePoints_element+
                         k];
          subgridErrorV[eN*nQuadraturePoints_element+
                       k] = 
            tau1
            *
            pdeResidualV[eN*nQuadraturePoints_element+
                         k];
          subgridErrorW[eN*nQuadraturePoints_element+
                       k] = 
            tau1
            *
            pdeResidualW[eN*nQuadraturePoints_element+
                         k];
          for (j=0;j<nDOF_trial_element;j++)
            {
              /* u */
              dsubgridErrorU_dp[eN*nQuadraturePoints_element*nDOF_trial_element+
                                k*nDOF_trial_element+
                                j] =
                tau1
                *
                dpdeResidualU_dp[eN*nQuadraturePoints_element*nDOF_trial_element+
                                 k*nDOF_trial_element+
                                 j];
              dsubgridErrorU_du[eN*nQuadraturePoints_element*nDOF_trial_element+
                                k*nDOF_trial_element+
                                j] = 
                tau1
                *
                dpdeResidualU_du[eN*nQuadraturePoints_element*nDOF_trial_element+
                                 k*nDOF_trial_element+
                                 j];
              /* v */
              dsubgridErrorV_dp[eN*nQuadraturePoints_element*nDOF_trial_element+
                                k*nDOF_trial_element+
                                j] = 
                tau1
                *
                dpdeResidualV_dp[eN*nQuadraturePoints_element*nDOF_trial_element+
                                 k*nDOF_trial_element+
                                 j];
              dsubgridErrorV_dv[eN*nQuadraturePoints_element*nDOF_trial_element+
                                k*nDOF_trial_element+
                                j] = 
                tau1
                *
                dpdeResidualV_dv[eN*nQuadraturePoints_element*nDOF_trial_element+
                                 k*nDOF_trial_element+
                                 j];
              /* w */
              dsubgridErrorW_dp[eN*nQuadraturePoints_element*nDOF_trial_element+
                                k*nDOF_trial_element+
                                j] = 
                tau1
                *
                dpdeResidualW_dp[eN*nQuadraturePoints_element*nDOF_trial_element+
                                 k*nDOF_trial_element+
                                 j];
              dsubgridErrorW_dw[eN*nQuadraturePoints_element*nDOF_trial_element+
                                k*nDOF_trial_element+
                                j] = 
                tau1
                *
                dpdeResidualW_dw[eN*nQuadraturePoints_element*nDOF_trial_element+
                                 k*nDOF_trial_element+
                                 j];
             }
        }
    }
}

/**
   \brief Calculate the stabilization parameter for the scalar advection-diffusion-reaction equation using the "l_1 norm formula"
*/
void calculateSubgridErrorStokes2D_GLS_velocity_pressure(proteus_index_t nElements_global,
                                                         int nQuadraturePoints_element,
                                                         int nDOF_trial_element,
                                                         int nSpace,
                                                         double* elementDiameter,
                                                         double* a,
                                                         double* pdeResidualP,
                                                         double* dpdeResidualP_du,
                                                         double* dpdeResidualP_dv,
                                                         double* pdeResidualU,
                                                         double* dpdeResidualU_dp,
                                                         double* dpdeResidualU_du,
                                                         double* pdeResidualV,
                                                         double* dpdeResidualV_dp,
                                                         double* dpdeResidualV_dv,
                                                         double* subgridErrorP,
                                                         double* dsubgridErrorP_du,
                                                         double* dsubgridErrorP_dv,
                                                         double* subgridErrorU,
                                                         double* dsubgridErrorU_dp,
                                                         double* dsubgridErrorU_du,
                                                         double* subgridErrorV,
                                                         double* dsubgridErrorV_dp,
                                                         double* dsubgridErrorV_dv)
{
  proteus_index_t nSpace2=nSpace*nSpace; int eN, k, j;
  double h,viscosity,tau1,tau2;
  for(eN=0;eN<nElements_global;eN++)
    {
      h = elementDiameter[eN];
      for (k=0;k<nQuadraturePoints_element;k++)
        {
          viscosity =  a[eN*nQuadraturePoints_element*nSpace2 + 
                         k*nSpace2]; 
          /* GLS momentum */
          tau1 = h*h/(12.0*viscosity);
          subgridErrorU[eN*nQuadraturePoints_element+
                       k] =
            tau1
            *
            pdeResidualU[eN*nQuadraturePoints_element+
                         k];
          subgridErrorV[eN*nQuadraturePoints_element+
                       k] = 
            tau1
            *
            pdeResidualV[eN*nQuadraturePoints_element+
                         k];
          /* GLS pressure */
          tau2 = 6.0*viscosity;
          subgridErrorP[eN*nQuadraturePoints_element+
                       k] =
            tau2
            *pdeResidualP[eN*nQuadraturePoints_element+
                          k];
          for (j=0;j<nDOF_trial_element;j++)
            {
              /* GLS  momentum*/
              /* u */
              dsubgridErrorU_dp[eN*nQuadraturePoints_element*nDOF_trial_element+
                                k*nDOF_trial_element+
                                j] =
                tau1
                *
                dpdeResidualU_dp[eN*nQuadraturePoints_element*nDOF_trial_element+
                                 k*nDOF_trial_element+
                                 j];
              dsubgridErrorU_du[eN*nQuadraturePoints_element*nDOF_trial_element+
                                k*nDOF_trial_element+
                                j] = 
                tau1
                *
                dpdeResidualU_du[eN*nQuadraturePoints_element*nDOF_trial_element+
                                 k*nDOF_trial_element+
                                 j];
              /* v */
              dsubgridErrorV_dp[eN*nQuadraturePoints_element*nDOF_trial_element+
                                k*nDOF_trial_element+
                                j] = 
                tau1
                *
                dpdeResidualV_dp[eN*nQuadraturePoints_element*nDOF_trial_element+
                                 k*nDOF_trial_element+
                                 j];
              dsubgridErrorV_dv[eN*nQuadraturePoints_element*nDOF_trial_element+
                                k*nDOF_trial_element+
                                j] = 
                tau1
                *
                dpdeResidualV_dv[eN*nQuadraturePoints_element*nDOF_trial_element+
                                 k*nDOF_trial_element+
                                 j];
              /* GLS pressure */
              dsubgridErrorP_du[eN*nQuadraturePoints_element*nDOF_trial_element+
                                k*nDOF_trial_element+
                                j]
                =
                tau2
                *
                dpdeResidualP_du[eN*nQuadraturePoints_element*nDOF_trial_element+
                                 k*nDOF_trial_element+
                                 j];
              dsubgridErrorP_dv[eN*nQuadraturePoints_element*nDOF_trial_element+
                                k*nDOF_trial_element+
                                j]
                =
                tau2
                *
                dpdeResidualP_dv[eN*nQuadraturePoints_element*nDOF_trial_element+
                                 k*nDOF_trial_element+
                                 j];
            }
        }
    }
}

void calculateSubgridErrorStokes2D_GLS_velocity_pressure_sd(proteus_index_t nElements_global,
                                                         int nQuadraturePoints_element,
                                                         int nDOF_trial_element,
                                                         int nSpace,
                                                         double* elementDiameter,
                                                         double* a,
                                                         double* pdeResidualP,
                                                         double* dpdeResidualP_du,
                                                         double* dpdeResidualP_dv,
                                                         double* pdeResidualU,
                                                         double* dpdeResidualU_dp,
                                                         double* dpdeResidualU_du,
                                                         double* pdeResidualV,
                                                         double* dpdeResidualV_dp,
                                                         double* dpdeResidualV_dv,
                                                         double* subgridErrorP,
                                                         double* dsubgridErrorP_du,
                                                         double* dsubgridErrorP_dv,
                                                         double* subgridErrorU,
                                                         double* dsubgridErrorU_dp,
                                                         double* dsubgridErrorU_du,
                                                         double* subgridErrorV,
                                                         double* dsubgridErrorV_dp,
                                                         double* dsubgridErrorV_dv)
{
  proteus_index_t nSpace2=nSpace*nSpace; int eN, k, j;
  double h,viscosity,tau1,tau2;
  for(eN=0;eN<nElements_global;eN++)
    {
      h = elementDiameter[eN];
      for (k=0;k<nQuadraturePoints_element;k++)
        {
          viscosity =  a[eN*nQuadraturePoints_element*nSpace + 
                         k*nSpace]; 
          /* GLS momentum */
          tau1 = h*h/(12.0*viscosity);
          subgridErrorU[eN*nQuadraturePoints_element+
                       k] =
            tau1
            *
            pdeResidualU[eN*nQuadraturePoints_element+
                         k];
          subgridErrorV[eN*nQuadraturePoints_element+
                       k] = 
            tau1
            *
            pdeResidualV[eN*nQuadraturePoints_element+
                         k];
          /* GLS pressure */
          tau2 = 6.0*viscosity;
          subgridErrorP[eN*nQuadraturePoints_element+
                       k] =
            tau2
            *pdeResidualP[eN*nQuadraturePoints_element+
                          k];
          for (j=0;j<nDOF_trial_element;j++)
            {
              /* GLS  momentum*/
              /* u */
              dsubgridErrorU_dp[eN*nQuadraturePoints_element*nDOF_trial_element+
                                k*nDOF_trial_element+
                                j] =
                tau1
                *
                dpdeResidualU_dp[eN*nQuadraturePoints_element*nDOF_trial_element+
                                 k*nDOF_trial_element+
                                 j];
              dsubgridErrorU_du[eN*nQuadraturePoints_element*nDOF_trial_element+
                                k*nDOF_trial_element+
                                j] = 
                tau1
                *
                dpdeResidualU_du[eN*nQuadraturePoints_element*nDOF_trial_element+
                                 k*nDOF_trial_element+
                                 j];
              /* v */
              dsubgridErrorV_dp[eN*nQuadraturePoints_element*nDOF_trial_element+
                                k*nDOF_trial_element+
                                j] = 
                tau1
                *
                dpdeResidualV_dp[eN*nQuadraturePoints_element*nDOF_trial_element+
                                 k*nDOF_trial_element+
                                 j];
              dsubgridErrorV_dv[eN*nQuadraturePoints_element*nDOF_trial_element+
                                k*nDOF_trial_element+
                                j] = 
                tau1
                *
                dpdeResidualV_dv[eN*nQuadraturePoints_element*nDOF_trial_element+
                                 k*nDOF_trial_element+
                                 j];
              /* GLS pressure */
              dsubgridErrorP_du[eN*nQuadraturePoints_element*nDOF_trial_element+
                                k*nDOF_trial_element+
                                j]
                =
                tau2
                *
                dpdeResidualP_du[eN*nQuadraturePoints_element*nDOF_trial_element+
                                 k*nDOF_trial_element+
                                 j];
              dsubgridErrorP_dv[eN*nQuadraturePoints_element*nDOF_trial_element+
                                k*nDOF_trial_element+
                                j]
                =
                tau2
                *
                dpdeResidualP_dv[eN*nQuadraturePoints_element*nDOF_trial_element+
                                 k*nDOF_trial_element+
                                 j];
            }
        }
    }
}

/**
   \brief Calculate the stabilization parameter for the scalar advection-diffusion-reaction equation using the "l_1 norm formula"
*/
void calculateSubgridErrorStokes3D_GLS_velocity_pressure(proteus_index_t nElements_global,
                                                         int nQuadraturePoints_element,
                                                         int nDOF_trial_element,
                                                         int nSpace,
                                                         double* elementDiameter,
                                                         double* a,
                                                         double* pdeResidualP,
                                                         double* dpdeResidualP_du,
                                                         double* dpdeResidualP_dv,
                                                         double* dpdeResidualP_dw,
                                                         double* pdeResidualU,
                                                         double* dpdeResidualU_dp,
                                                         double* dpdeResidualU_du,
                                                         double* pdeResidualV,
                                                         double* dpdeResidualV_dp,
                                                         double* dpdeResidualV_dv,
                                                         double* pdeResidualW,
                                                         double* dpdeResidualW_dp,
                                                         double* dpdeResidualW_dw,
                                                         double* subgridErrorP,
                                                         double* dsubgridErrorP_du,
                                                         double* dsubgridErrorP_dv,
                                                         double* dsubgridErrorP_dw,
                                                         double* subgridErrorU,
                                                         double* dsubgridErrorU_dp,
                                                         double* dsubgridErrorU_du,
                                                         double* subgridErrorV,
                                                         double* dsubgridErrorV_dp,
                                                         double* dsubgridErrorV_dv,
                                                         double* subgridErrorW,
                                                         double* dsubgridErrorW_dp,
                                                         double* dsubgridErrorW_dw)
{
  proteus_index_t nSpace2=nSpace*nSpace; int eN, k, j;
  double h,viscosity,tau1,tau2;
  for(eN=0;eN<nElements_global;eN++)
    {
      h = elementDiameter[eN];
      for (k=0;k<nQuadraturePoints_element;k++)
        {
          viscosity =  a[eN*nQuadraturePoints_element*nSpace2 + 
                         k*nSpace2]; 
          /* GLS momentum */
          tau1 = h*h/(12.0*viscosity);
          subgridErrorU[eN*nQuadraturePoints_element+
                       k] =
            tau1
            *
            pdeResidualU[eN*nQuadraturePoints_element+
                         k];
          subgridErrorV[eN*nQuadraturePoints_element+
                       k] = 
            tau1
            *
            pdeResidualV[eN*nQuadraturePoints_element+
                         k];
          subgridErrorW[eN*nQuadraturePoints_element+
                       k] = 
            tau1
            *
            pdeResidualW[eN*nQuadraturePoints_element+
                         k];
          /* GLS pressure */
          tau2 = 6.0*viscosity;
          subgridErrorP[eN*nQuadraturePoints_element+
                       k] =
            tau2
            *pdeResidualP[eN*nQuadraturePoints_element+
                          k];
          for (j=0;j<nDOF_trial_element;j++)
            {
              /* GLS  momentum*/
              /* u */
              dsubgridErrorU_dp[eN*nQuadraturePoints_element*nDOF_trial_element+
                                k*nDOF_trial_element+
                                j] =
                tau1
                *
                dpdeResidualU_dp[eN*nQuadraturePoints_element*nDOF_trial_element+
                                 k*nDOF_trial_element+
                                 j];
              dsubgridErrorU_du[eN*nQuadraturePoints_element*nDOF_trial_element+
                                k*nDOF_trial_element+
                                j] = 
                tau1
                *
                dpdeResidualU_du[eN*nQuadraturePoints_element*nDOF_trial_element+
                                 k*nDOF_trial_element+
                                 j];
              /* v */
              dsubgridErrorV_dp[eN*nQuadraturePoints_element*nDOF_trial_element+
                                k*nDOF_trial_element+
                                j] = 
                tau1
                *
                dpdeResidualV_dp[eN*nQuadraturePoints_element*nDOF_trial_element+
                                 k*nDOF_trial_element+
                                 j];
              dsubgridErrorV_dv[eN*nQuadraturePoints_element*nDOF_trial_element+
                                k*nDOF_trial_element+
                                j] = 
                tau1
                *
                dpdeResidualV_dv[eN*nQuadraturePoints_element*nDOF_trial_element+
                                 k*nDOF_trial_element+
                                 j];
              /* w */
              dsubgridErrorW_dp[eN*nQuadraturePoints_element*nDOF_trial_element+
                                k*nDOF_trial_element+
                                j] = 
                tau1
                *
                dpdeResidualW_dp[eN*nQuadraturePoints_element*nDOF_trial_element+
                                 k*nDOF_trial_element+
                                 j];
              dsubgridErrorW_dw[eN*nQuadraturePoints_element*nDOF_trial_element+
                                k*nDOF_trial_element+
                                j] = 
                tau1
                *
                dpdeResidualW_dw[eN*nQuadraturePoints_element*nDOF_trial_element+
                                 k*nDOF_trial_element+
                                 j];
              /* GLS pressure */
              dsubgridErrorP_du[eN*nQuadraturePoints_element*nDOF_trial_element+
                                k*nDOF_trial_element+
                                j]
                =
                tau2
                *
                dpdeResidualP_du[eN*nQuadraturePoints_element*nDOF_trial_element+
                                 k*nDOF_trial_element+
                                 j];
              dsubgridErrorP_dv[eN*nQuadraturePoints_element*nDOF_trial_element+
                                k*nDOF_trial_element+
                                j]
                =
                tau2
                *
                dpdeResidualP_dv[eN*nQuadraturePoints_element*nDOF_trial_element+
                                 k*nDOF_trial_element+
                                 j];
            }
        }
    }
}
void calculateSubgridErrorStokes3D_GLS_velocity_pressure_sd(proteus_index_t nElements_global,
                                                         int nQuadraturePoints_element,
                                                         int nDOF_trial_element,
                                                         int nSpace,
                                                         double* elementDiameter,
                                                         double* a,
                                                         double* pdeResidualP,
                                                         double* dpdeResidualP_du,
                                                         double* dpdeResidualP_dv,
                                                         double* dpdeResidualP_dw,
                                                         double* pdeResidualU,
                                                         double* dpdeResidualU_dp,
                                                         double* dpdeResidualU_du,
                                                         double* pdeResidualV,
                                                         double* dpdeResidualV_dp,
                                                         double* dpdeResidualV_dv,
                                                         double* pdeResidualW,
                                                         double* dpdeResidualW_dp,
                                                         double* dpdeResidualW_dw,
                                                         double* subgridErrorP,
                                                         double* dsubgridErrorP_du,
                                                         double* dsubgridErrorP_dv,
                                                         double* dsubgridErrorP_dw,
                                                         double* subgridErrorU,
                                                         double* dsubgridErrorU_dp,
                                                         double* dsubgridErrorU_du,
                                                         double* subgridErrorV,
                                                         double* dsubgridErrorV_dp,
                                                         double* dsubgridErrorV_dv,
                                                         double* subgridErrorW,
                                                         double* dsubgridErrorW_dp,
                                                         double* dsubgridErrorW_dw)
{
  proteus_index_t nSpace2=nSpace*nSpace; int eN, k, j;
  double h,viscosity,tau1,tau2;
  for(eN=0;eN<nElements_global;eN++)
    {
      h = elementDiameter[eN];
      for (k=0;k<nQuadraturePoints_element;k++)
        {
          viscosity =  a[eN*nQuadraturePoints_element*nSpace + 
                         k*nSpace]; 
          /* GLS momentum */
          tau1 = h*h/(12.0*viscosity);
          subgridErrorU[eN*nQuadraturePoints_element+
                       k] =
            tau1
            *
            pdeResidualU[eN*nQuadraturePoints_element+
                         k];
          subgridErrorV[eN*nQuadraturePoints_element+
                       k] = 
            tau1
            *
            pdeResidualV[eN*nQuadraturePoints_element+
                         k];
          subgridErrorW[eN*nQuadraturePoints_element+
                       k] = 
            tau1
            *
            pdeResidualW[eN*nQuadraturePoints_element+
                         k];
          /* GLS pressure */
          tau2 = 6.0*viscosity;
          subgridErrorP[eN*nQuadraturePoints_element+
                       k] =
            tau2
            *pdeResidualP[eN*nQuadraturePoints_element+
                          k];
          for (j=0;j<nDOF_trial_element;j++)
            {
              /* GLS  momentum*/
              /* u */
              dsubgridErrorU_dp[eN*nQuadraturePoints_element*nDOF_trial_element+
                                k*nDOF_trial_element+
                                j] =
                tau1
                *
                dpdeResidualU_dp[eN*nQuadraturePoints_element*nDOF_trial_element+
                                 k*nDOF_trial_element+
                                 j];
              dsubgridErrorU_du[eN*nQuadraturePoints_element*nDOF_trial_element+
                                k*nDOF_trial_element+
                                j] = 
                tau1
                *
                dpdeResidualU_du[eN*nQuadraturePoints_element*nDOF_trial_element+
                                 k*nDOF_trial_element+
                                 j];
              /* v */
              dsubgridErrorV_dp[eN*nQuadraturePoints_element*nDOF_trial_element+
                                k*nDOF_trial_element+
                                j] = 
                tau1
                *
                dpdeResidualV_dp[eN*nQuadraturePoints_element*nDOF_trial_element+
                                 k*nDOF_trial_element+
                                 j];
              dsubgridErrorV_dv[eN*nQuadraturePoints_element*nDOF_trial_element+
                                k*nDOF_trial_element+
                                j] = 
                tau1
                *
                dpdeResidualV_dv[eN*nQuadraturePoints_element*nDOF_trial_element+
                                 k*nDOF_trial_element+
                                 j];
              /* w */
              dsubgridErrorW_dp[eN*nQuadraturePoints_element*nDOF_trial_element+
                                k*nDOF_trial_element+
                                j] = 
                tau1
                *
                dpdeResidualW_dp[eN*nQuadraturePoints_element*nDOF_trial_element+
                                 k*nDOF_trial_element+
                                 j];
              dsubgridErrorW_dw[eN*nQuadraturePoints_element*nDOF_trial_element+
                                k*nDOF_trial_element+
                                j] = 
                tau1
                *
                dpdeResidualW_dw[eN*nQuadraturePoints_element*nDOF_trial_element+
                                 k*nDOF_trial_element+
                                 j];
              /* GLS pressure */
              dsubgridErrorP_du[eN*nQuadraturePoints_element*nDOF_trial_element+
                                k*nDOF_trial_element+
                                j]
                =
                tau2
                *
                dpdeResidualP_du[eN*nQuadraturePoints_element*nDOF_trial_element+
                                 k*nDOF_trial_element+
                                 j];
              dsubgridErrorP_dv[eN*nQuadraturePoints_element*nDOF_trial_element+
                                k*nDOF_trial_element+
                                j]
                =
                tau2
                *
                dpdeResidualP_dv[eN*nQuadraturePoints_element*nDOF_trial_element+
                                 k*nDOF_trial_element+
                                 j];
            }
        }
    }
}


/**
   \brief Calculate the stabilization parameter for the scalar advection-diffusion-reaction equation using the "l_1 norm formula"
*/
void calculateSubgridErrorNavierStokes2D_GLS_tau(proteus_index_t nElements_global,
                                                 int nQuadraturePoints_element,
                                                 int nSpace,
                                                 double  hFactor,
                                                 double* elementDiameter,
                                                 double* dmt,
                                                 double* dm,
                                                 double* f,
                                                 double* a,
                                                 double* tau0,
                                                 double* tau1,
                                                 double* cfl)
{
  proteus_index_t nSpace2=nSpace*nSpace; int eN, k, I;
  double h,oneByAbsdt,density,viscosity,Re_max=0.0,CFL_max=0.0,nrm_v;
  for(eN=0;eN<nElements_global;eN++)
    {
      h = hFactor*elementDiameter[eN];
      //printf("h %12.5e \n",h);
      for (k=0;k<nQuadraturePoints_element;k++)
        {
          density = dm[eN*nQuadraturePoints_element+
                       k];
          viscosity =  a[eN*nQuadraturePoints_element*nSpace2 + 
                         k*nSpace2+
                         nSpace+1]; 
          nrm_v=0.0;
          for(I=0;I<nSpace;I++)
            nrm_v+=f[eN*nQuadraturePoints_element*nSpace+
                     k*nSpace+
                     I]*
              f[eN*nQuadraturePoints_element*nSpace+
                k*nSpace+
                I];
          nrm_v = sqrt(nrm_v);
          Re_max = fmax(nrm_v*h/(viscosity/density),Re_max);
          CFL_max = fmax(nrm_v/h,CFL_max);
          cfl[eN*nQuadraturePoints_element+k] = nrm_v/h;
          if (0)//abs(dmt[eN*nQuadraturePoints_element+k]) > 10.0*(density*nrm_v/h))
            {
              oneByAbsdt = 10.0*(density*nrm_v/h);
            }
          else
            {
              oneByAbsdt =  fabs(dmt[eN*nQuadraturePoints_element+k]);
            }
          tau0[eN*nQuadraturePoints_element+k] = 1.0/(4.0*viscosity/(h*h) +
                                                      2.0*density*nrm_v/h +
                                                      oneByAbsdt);
          tau1[eN*nQuadraturePoints_element+k] = 4.0*viscosity +
            2.0*density*nrm_v*h+
            oneByAbsdt*h*h;
	  printf("nrm_v %12.5e tau_v %12.5e tau_p %12.5e \n",nrm_v,tau0[eN*nQuadraturePoints_element+k],tau1[eN*nQuadraturePoints_element+k]);
/*           tau0[eN*nQuadraturePoints_element+k] = density/(4.0*viscosity/(h*h) + */
/*                                                       2.0*density*nrm_v/h + */
/*                                                       oneByAbsdt); */
/*           tau1[eN*nQuadraturePoints_element+k] = (4.0*viscosity + */
/*             2.0*density*nrm_v*h+ */
/*                                                   oneByAbsdt*h*h)/density; */
          //printf("tau0 %12.5e \n",tau0[eN*nQuadraturePoints_element+k]);
          //printf("tau1 %12.5e \n",tau1[eN*nQuadraturePoints_element+k]);
/*           tau0[eN*nQuadraturePoints_element+k] = 1.0/(4.0*viscosity/(h*h) + */
/*                                                       2.0*density*nrm_v/h); */
/*           tau1[eN*nQuadraturePoints_element+k] = 4.0*viscosity + 2.0*density*nrm_v*h; */
/*           tau0[eN*nQuadraturePoints_element+k] = 1.0/(4.0*viscosity/(h*h) + */
/*                                                       2.0*density*nrm_v/h); */
/*           tau1[eN*nQuadraturePoints_element+k] = 4.0*viscosity + */
/*             2.0*density*nrm_v*h; */
/*           if (fabs(dmt[eN*nQuadraturePoints_element+k]) < nrm_v/h) */
/*             { */
/*               tau0[eN*nQuadraturePoints_element+k] = 1.0/(4.0*viscosity/(h*h) + */
/*                                                           2.0*density*nrm_v/h + */
/*                                                           fabs(dmt[eN*nQuadraturePoints_element+k])); */
/*               tau1[eN*nQuadraturePoints_element+k] = 4.0*viscosity + */
/*                 2.0*density*nrm_v*h+ */
/*                 fabs(dmt[eN*nQuadraturePoints_element+k])*h*h; */
/*             } */
/*           else */
/*             { */
/*               tau0[eN*nQuadraturePoints_element+k] = 1.0/(4.0*viscosity/(h*h) +2.0*density*nrm_v/h); */
/*               tau1[eN*nQuadraturePoints_element+k] = 4.0*viscosity + 2.0*density*nrm_v*h; */
/*             } */
/*           tau0[eN*nQuadraturePoints_element+k] = 1.0/(4.0*viscosity/(h*h) + */
/*                                                       2.0*density*nrm_v/h); */
/*           if (fabs(dmt[eN*nQuadraturePoints_element+k]) > nrm_v/h) */
/*             { */
/*               tau1[eN*nQuadraturePoints_element+k] = (4.0*viscosity + 2.0*density*nrm_v*h); */
/*             } */
/*           else */
/*             tau1[eN*nQuadraturePoints_element+k] = h*h/tau0[eN*nQuadraturePoints_element+k]; */
/*           tau0[eN*nQuadraturePoints_element+k] = 1.0/(4.0*viscosity/(h*h) + */
/*                                                       2.0*density*nrm_v/h + */
/*                                                       fabs(dmt[eN*nQuadraturePoints_element+k])); */
/*           tau1[eN*nQuadraturePoints_element+k] = 4.0*viscosity + */
/*             2.0*density*nrm_v*h; */
/*           tau0[eN*nQuadraturePoints_element+k] = 1.0/(4.0*viscosity/(h*h) + */
/*                                                       2.0*density*nrm_v/h); */
/*           tau1[eN*nQuadraturePoints_element+k] = 4.0*viscosity + */
/*             2.0*density*nrm_v*h+ */
/*             fabs(dmt[eN*nQuadraturePoints_element+k])*h*h; */
         }
    }
  /*  printf("Element Re = %12.5e \n",Re_max);
      printf("Element CFL = %12.5e \n",CFL_max);*/
}

void calculateSubgridErrorNavierStokes2D_GLS_tau_sd(proteus_index_t nElements_global,
                                                 int nQuadraturePoints_element,
                                                 int nSpace,
                                                 double  hFactor,
                                                 double* elementDiameter,
                                                 double* dmt,
                                                 double* dm,
                                                 double* f,
                                                 double* a,
                                                 double* tau0,
                                                 double* tau1,
                                                 double* cfl)
{
  proteus_index_t nSpace2=nSpace*nSpace; int eN, k, I;
  double h,oneByAbsdt,density,viscosity,Re_max=0.0,CFL_max=0.0,nrm_v;
  for(eN=0;eN<nElements_global;eN++)
    {
      h = hFactor*elementDiameter[eN];
      //printf("h %12.5e \n",h);
      for (k=0;k<nQuadraturePoints_element;k++)
        {
          density = dm[eN*nQuadraturePoints_element+
                       k];
          viscosity =  a[eN*nQuadraturePoints_element*nSpace + 
                         k*nSpace+1]; 
          nrm_v=0.0;
          for(I=0;I<nSpace;I++)
            nrm_v+=f[eN*nQuadraturePoints_element*nSpace+
                     k*nSpace+
                     I]*
              f[eN*nQuadraturePoints_element*nSpace+
                k*nSpace+
                I];
          nrm_v = sqrt(nrm_v);
          Re_max = fmax(nrm_v*h/(viscosity/density),Re_max);
          CFL_max = fmax(nrm_v/h,CFL_max);
          cfl[eN*nQuadraturePoints_element+k] = nrm_v/h;
          if (0)//abs(dmt[eN*nQuadraturePoints_element+k]) > 10.0*(density*nrm_v/h))
            {
              oneByAbsdt = 10.0*(density*nrm_v/h);
            }
          else
            {
              oneByAbsdt =  fabs(dmt[eN*nQuadraturePoints_element+k]);
            }
          tau0[eN*nQuadraturePoints_element+k] = 1.0/(4.0*viscosity/(h*h) +
                                                      2.0*density*nrm_v/h +
                                                      oneByAbsdt);
          tau1[eN*nQuadraturePoints_element+k] = 4.0*viscosity +
            2.0*density*nrm_v*h+
            oneByAbsdt*h*h;
	  /* printf("nrm_v %12.5e tau_v %12.5e tau_p %12.5e \n",nrm_v,tau0[eN*nQuadraturePoints_element+k],tau1[eN*nQuadraturePoints_element+k]); */
/*           tau0[eN*nQuadraturePoints_element+k] = density/(4.0*viscosity/(h*h) + */
/*                                                       2.0*density*nrm_v/h + */
/*                                                       oneByAbsdt); */
/*           tau1[eN*nQuadraturePoints_element+k] = (4.0*viscosity + */
/*             2.0*density*nrm_v*h+ */
/*                                                   oneByAbsdt*h*h)/density; */
          //printf("tau0 %12.5e \n",tau0[eN*nQuadraturePoints_element+k]);
          //printf("tau1 %12.5e \n",tau1[eN*nQuadraturePoints_element+k]);
/*           tau0[eN*nQuadraturePoints_element+k] = 1.0/(4.0*viscosity/(h*h) + */
/*                                                       2.0*density*nrm_v/h); */
/*           tau1[eN*nQuadraturePoints_element+k] = 4.0*viscosity + 2.0*density*nrm_v*h; */
/*           tau0[eN*nQuadraturePoints_element+k] = 1.0/(4.0*viscosity/(h*h) + */
/*                                                       2.0*density*nrm_v/h); */
/*           tau1[eN*nQuadraturePoints_element+k] = 4.0*viscosity + */
/*             2.0*density*nrm_v*h; */
/*           if (fabs(dmt[eN*nQuadraturePoints_element+k]) < nrm_v/h) */
/*             { */
/*               tau0[eN*nQuadraturePoints_element+k] = 1.0/(4.0*viscosity/(h*h) + */
/*                                                           2.0*density*nrm_v/h + */
/*                                                           fabs(dmt[eN*nQuadraturePoints_element+k])); */
/*               tau1[eN*nQuadraturePoints_element+k] = 4.0*viscosity + */
/*                 2.0*density*nrm_v*h+ */
/*                 fabs(dmt[eN*nQuadraturePoints_element+k])*h*h; */
/*             } */
/*           else */
/*             { */
/*               tau0[eN*nQuadraturePoints_element+k] = 1.0/(4.0*viscosity/(h*h) +2.0*density*nrm_v/h); */
/*               tau1[eN*nQuadraturePoints_element+k] = 4.0*viscosity + 2.0*density*nrm_v*h; */
/*             } */
/*           tau0[eN*nQuadraturePoints_element+k] = 1.0/(4.0*viscosity/(h*h) + */
/*                                                       2.0*density*nrm_v/h); */
/*           if (fabs(dmt[eN*nQuadraturePoints_element+k]) > nrm_v/h) */
/*             { */
/*               tau1[eN*nQuadraturePoints_element+k] = (4.0*viscosity + 2.0*density*nrm_v*h); */
/*             } */
/*           else */
/*             tau1[eN*nQuadraturePoints_element+k] = h*h/tau0[eN*nQuadraturePoints_element+k]; */
/*           tau0[eN*nQuadraturePoints_element+k] = 1.0/(4.0*viscosity/(h*h) + */
/*                                                       2.0*density*nrm_v/h + */
/*                                                       fabs(dmt[eN*nQuadraturePoints_element+k])); */
/*           tau1[eN*nQuadraturePoints_element+k] = 4.0*viscosity + */
/*             2.0*density*nrm_v*h; */
/*           tau0[eN*nQuadraturePoints_element+k] = 1.0/(4.0*viscosity/(h*h) + */
/*                                                       2.0*density*nrm_v/h); */
/*           tau1[eN*nQuadraturePoints_element+k] = 4.0*viscosity + */
/*             2.0*density*nrm_v*h+ */
/*             fabs(dmt[eN*nQuadraturePoints_element+k])*h*h; */
         }
    }
  /*  printf("Element Re = %12.5e \n",Re_max);
      printf("Element CFL = %12.5e \n",CFL_max);*/
}

void calculateSubgridErrorNavierStokes2D_generic_tau(proteus_index_t nElements_global,
                                                     int nQuadraturePoints_element,
                                                     int nSpace,
                                                     double* inverseJ,
                                                     double* dmt,
                                                     double* dm,
                                                     double* f,
                                                     double* a,
                                                     double* tau0,
                                                     double* tau1,
                                                     double* cfl)
{
  proteus_index_t nSpace2=nSpace*nSpace; int eN, k, I, J, K;
  double density,viscosity,Re_max=0.0,CFL_max=0.0,nrm_v,G_IJ,v_dot_Gv,CI_nu2_G_ddot_G,Gv_I,CI=36.0*36.0,g_I,g_dot_g,den;
  for(eN=0;eN<nElements_global;eN++)
    {
      for (k=0;k<nQuadraturePoints_element;k++)
        {
          v_dot_Gv=0.0;
          CI_nu2_G_ddot_G=0.0;
          g_dot_g=0.0;
          for (I=0;I<nSpace;I++)
            {
              Gv_I=0.0;
              g_I=0.0;
              for (J=0;J<nSpace;J++)
                {
                  G_IJ=0.0;
                  for (K=0;K<nSpace;K++)
                    G_IJ +=inverseJ[eN*nQuadraturePoints_element*nSpace2+
                                    k*nSpace2+
                                    K*nSpace+
                                    I]
                      *
                      inverseJ[eN*nQuadraturePoints_element*nSpace2+
                               k*nSpace2+
                               K*nSpace+
                               J];
                  Gv_I += G_IJ
                    *
                    f[eN*nQuadraturePoints_element*nSpace+
                      k*nSpace+
                      J];
                  CI_nu2_G_ddot_G+=G_IJ*G_IJ;
                  g_I+=inverseJ[eN*nQuadraturePoints_element*nSpace2+
                                k*nSpace2+
                                J*nSpace+
                                I];
                }
              g_dot_g += g_I*g_I;
              v_dot_Gv += f[eN*nQuadraturePoints_element*nSpace+
                            k*nSpace+
                            I]
                *
                Gv_I;
              
            }
          density = dm[eN*nQuadraturePoints_element+
                       k];
	  /*mwf add another version for Laplace form?*/
          /*cek this should now work for either form*/
          viscosity =  a[eN*nQuadraturePoints_element*nSpace2 + 
                         k*nSpace2+
                         nSpace+1]; 
          CI_nu2_G_ddot_G*=CI*viscosity*viscosity;
          cfl[eN*nQuadraturePoints_element+k] = sqrt(v_dot_Gv);
	  den = sqrt(4.0*dmt[eN*nQuadraturePoints_element+k]*dmt[eN*nQuadraturePoints_element+k]+
		     v_dot_Gv+
		     CI_nu2_G_ddot_G+
		     1.0e-8);
/* 	  den = sqrt(v_dot_Gv+ */
/* 		     CI_nu2_G_ddot_G+ */
/* 		     1.0e-8); */
	  if(den > 1.0e-8)
	    {
	      tau0[eN*nQuadraturePoints_element+k] = 1.0/den;
              tau1[eN*nQuadraturePoints_element+k] = sqrt(v_dot_Gv+CI_nu2_G_ddot_G)/g_dot_g;
              //tau1[eN*nQuadraturePoints_element+k] = 1.0/(tau0[eN*nQuadraturePoints_element+k]*g_dot_g);
	    }
	  else
	    {
	      tau0[eN*nQuadraturePoints_element+k] = 0.0;
	      tau1[eN*nQuadraturePoints_element+k] = 0.0;
	    }
/*           printf("%12.5e %12.5e \n",tau0[eN*nQuadraturePoints_element+k],tau1[eN*nQuadraturePoints_element+k]); */
        }
    }
/*   printf("Element Re = %12.5e \n",Re_max); */
/*   printf("Element CFL = %12.5e \n",CFL_max); */
}
void calculateSubgridErrorNavierStokes2D_generic_tau_sd(proteus_index_t nElements_global,
                                                     int nQuadraturePoints_element,
                                                     int nSpace,
                                                     double* inverseJ,
                                                     double* dmt,
                                                     double* dm,
                                                     double* f,
                                                     double* a,
                                                     double* tau0,
                                                     double* tau1,
                                                     double* cfl)
{
  proteus_index_t nSpace2=nSpace*nSpace; int eN, k, I, J, K;
  double density,viscosity,Re_max=0.0,CFL_max=0.0,nrm_v,G_IJ,v_dot_Gv,CI_nu2_G_ddot_G,Gv_I,CI=36.0*36.0,g_I,g_dot_g,den;
  for(eN=0;eN<nElements_global;eN++)
    {
      for (k=0;k<nQuadraturePoints_element;k++)
        {
          v_dot_Gv=0.0;
          CI_nu2_G_ddot_G=0.0;
          g_dot_g=0.0;
          for (I=0;I<nSpace;I++)
            {
              Gv_I=0.0;
              g_I=0.0;
              for (J=0;J<nSpace;J++)
                {
                  G_IJ=0.0;
                  for (K=0;K<nSpace;K++)
                    G_IJ +=inverseJ[eN*nQuadraturePoints_element*nSpace2+
                                    k*nSpace2+
                                    K*nSpace+
                                    I]
                      *
                      inverseJ[eN*nQuadraturePoints_element*nSpace2+
                               k*nSpace2+
                               K*nSpace+
                               J];
                  Gv_I += G_IJ
                    *
                    f[eN*nQuadraturePoints_element*nSpace+
                      k*nSpace+
                      J];
                  CI_nu2_G_ddot_G+=G_IJ*G_IJ;
                  g_I+=inverseJ[eN*nQuadraturePoints_element*nSpace2+
                                k*nSpace2+
                                J*nSpace+
                                I];
                }
              g_dot_g += g_I*g_I;
              v_dot_Gv += f[eN*nQuadraturePoints_element*nSpace+
                            k*nSpace+
                            I]
                *
                Gv_I;
              
            }
          density = dm[eN*nQuadraturePoints_element+
                       k];
	  /*mwf add another version for Laplace form?*/
          /*cek this should now work for either form*/
          viscosity =  a[eN*nQuadraturePoints_element*nSpace + 
                         k*nSpace+1]; 
          CI_nu2_G_ddot_G*=CI*viscosity*viscosity;
          cfl[eN*nQuadraturePoints_element+k] = sqrt(v_dot_Gv);
	  den = sqrt(4.0*dmt[eN*nQuadraturePoints_element+k]*dmt[eN*nQuadraturePoints_element+k]+
		     v_dot_Gv+
		     CI_nu2_G_ddot_G+
		     1.0e-8);
/* 	  den = sqrt(v_dot_Gv+ */
/* 		     CI_nu2_G_ddot_G+ */
/* 		     1.0e-8); */
	  if(den > 1.0e-8)
	    {
	      tau0[eN*nQuadraturePoints_element+k] = 1.0/den;
              tau1[eN*nQuadraturePoints_element+k] = sqrt(v_dot_Gv+CI_nu2_G_ddot_G)/g_dot_g;
              //tau1[eN*nQuadraturePoints_element+k] = 1.0/(tau0[eN*nQuadraturePoints_element+k]*g_dot_g);
	    }
	  else
	    {
	      tau0[eN*nQuadraturePoints_element+k] = 0.0;
	      tau1[eN*nQuadraturePoints_element+k] = 0.0;
	    }
/*           printf("%12.5e %12.5e \n",tau0[eN*nQuadraturePoints_element+k],tau1[eN*nQuadraturePoints_element+k]); */
        }
    }
/*   printf("Element Re = %12.5e \n",Re_max); */
/*   printf("Element CFL = %12.5e \n",CFL_max); */
}
void calculateSubgridErrorNavierStokes2D_generic_withBodyForce_tau(proteus_index_t nElements_global,
								   int nQuadraturePoints_element,
								   int nSpace,
								   double* inverseJ,
								   double* dmt,
								   double* dm,
								   double* f,
								   double* a,
								   double* dr,
								   double* tau0,
								   double* tau1,
								   double* cfl)
{
  proteus_index_t nSpace2=nSpace*nSpace; int eN, k, I, J, K;
  double h,density,viscosity,Re_max=0.0,CFL_max=0.0,nrm_v,G_IJ,v_dot_Gv,CI_nu2_G_ddot_G,Gv_I,CI=36.0,g_I,g_dot_g,den;
  for(eN=0;eN<nElements_global;eN++)
    {
      for (k=0;k<nQuadraturePoints_element;k++)
        {
          v_dot_Gv=0.0;
          CI_nu2_G_ddot_G=0.0;
          g_dot_g=0.0;
          for (I=0;I<nSpace;I++)
            {
              Gv_I=0.0;
              g_I=0.0;
              for (J=0;J<nSpace;J++)
                {
                  G_IJ=0.0;
                  for (K=0;K<nSpace;K++)
                    G_IJ +=inverseJ[eN*nQuadraturePoints_element*nSpace2+
                                    k*nSpace2+
                                    K*nSpace+
                                    I]
                      *
                      inverseJ[eN*nQuadraturePoints_element*nSpace2+
                               k*nSpace2+
                               K*nSpace+
                               J];
                  Gv_I += G_IJ
                    *
                    f[eN*nQuadraturePoints_element*nSpace+
                      k*nSpace+
                      J];
                  CI_nu2_G_ddot_G+=G_IJ*G_IJ;
                  g_I+=inverseJ[eN*nQuadraturePoints_element*nSpace2+
                                k*nSpace2+
                                J*nSpace+
                                I];
                }
              g_dot_g += g_I*g_I;
              v_dot_Gv += f[eN*nQuadraturePoints_element*nSpace+
                            k*nSpace+
                            I]
                *
                Gv_I;
              
            }
          density = dm[eN*nQuadraturePoints_element+
                       k];
          viscosity =  0.5*a[eN*nQuadraturePoints_element*nSpace2 + 
                             k*nSpace2]; 
          CI_nu2_G_ddot_G*=CI*viscosity*viscosity;
          nrm_v = sqrt(v_dot_Gv);
          Re_max = fmax(nrm_v/(viscosity/density),Re_max);
          CFL_max = fmax(nrm_v,CFL_max);
          cfl[eN*nQuadraturePoints_element+k] = nrm_v/h;
	  den = sqrt(4.0*dmt[eN*nQuadraturePoints_element+k]*dmt[eN*nQuadraturePoints_element+k]+
		     4.0*dr[eN*nQuadraturePoints_element+k]*dr[eN*nQuadraturePoints_element+k]+
		     v_dot_Gv+
		     CI_nu2_G_ddot_G+
		     1.0e-8);
	  if(den > 1.0e-8)
	    {
	      tau0[eN*nQuadraturePoints_element+k] = 1.0/den;
	      tau1[eN*nQuadraturePoints_element+k] = 1.0/(tau0[eN*nQuadraturePoints_element+k]*g_dot_g);
	    }
	  else
	    {
	      tau0[eN*nQuadraturePoints_element+k] = 0.0;
	      tau1[eN*nQuadraturePoints_element+k] = 0.0;
	    }
	  /*mwf debug
	  if (den < 1.0e-4 || 1)
	    {
	      printf("NSbodyForce tau eN=%d den = %12.5e dmt=%12.5e dr=%12.5e density=%12.5e viscosity= %12.5e\n",eN,den,
		     dmt[eN*nQuadraturePoints_element+k],dr[eN*nQuadraturePoints_element+k],density,viscosity);
	    }
	  */
/*           printf("%12.5e %12.5e \n",tau0[eN*nQuadraturePoints_element+k],tau1[eN*nQuadraturePoints_element+k]); */
        }
    }
/*   printf("Element Re = %12.5e \n",Re_max); */
/*   printf("Element CFL = %12.5e \n",CFL_max); */
}

void calculateSubgridErrorNavierStokes2D_generic_withBodyForce_tau_sd(proteus_index_t nElements_global,
								   int nQuadraturePoints_element,
								   int nSpace,
								   double* inverseJ,
								   double* dmt,
								   double* dm,
								   double* f,
								   double* a,
								   double* dr,
								   double* tau0,
								   double* tau1,
								   double* cfl)
{
  proteus_index_t nSpace2=nSpace*nSpace; int eN, k, I, J, K;
  double h,density,viscosity,Re_max=0.0,CFL_max=0.0,nrm_v,G_IJ,v_dot_Gv,CI_nu2_G_ddot_G,Gv_I,CI=36.0,g_I,g_dot_g,den;
  for(eN=0;eN<nElements_global;eN++)
    {
      for (k=0;k<nQuadraturePoints_element;k++)
        {
          v_dot_Gv=0.0;
          CI_nu2_G_ddot_G=0.0;
          g_dot_g=0.0;
          for (I=0;I<nSpace;I++)
            {
              Gv_I=0.0;
              g_I=0.0;
              for (J=0;J<nSpace;J++)
                {
                  G_IJ=0.0;
                  for (K=0;K<nSpace;K++)
                    G_IJ +=inverseJ[eN*nQuadraturePoints_element*nSpace2+
                                    k*nSpace2+
                                    K*nSpace+
                                    I]
                      *
                      inverseJ[eN*nQuadraturePoints_element*nSpace2+
                               k*nSpace2+
                               K*nSpace+
                               J];
                  Gv_I += G_IJ
                    *
                    f[eN*nQuadraturePoints_element*nSpace+
                      k*nSpace+
                      J];
                  CI_nu2_G_ddot_G+=G_IJ*G_IJ;
                  g_I+=inverseJ[eN*nQuadraturePoints_element*nSpace2+
                                k*nSpace2+
                                J*nSpace+
                                I];
                }
              g_dot_g += g_I*g_I;
              v_dot_Gv += f[eN*nQuadraturePoints_element*nSpace+
                            k*nSpace+
                            I]
                *
                Gv_I;
              
            }
          density = dm[eN*nQuadraturePoints_element+
                       k];
          viscosity =  0.5*a[eN*nQuadraturePoints_element*nSpace + 
                             k*nSpace]; 
          CI_nu2_G_ddot_G*=CI*viscosity*viscosity;
          nrm_v = sqrt(v_dot_Gv);
          Re_max = fmax(nrm_v/(viscosity/density),Re_max);
          CFL_max = fmax(nrm_v,CFL_max);
          cfl[eN*nQuadraturePoints_element+k] = nrm_v/h;
	  den = sqrt(4.0*dmt[eN*nQuadraturePoints_element+k]*dmt[eN*nQuadraturePoints_element+k]+
		     4.0*dr[eN*nQuadraturePoints_element+k]*dr[eN*nQuadraturePoints_element+k]+
		     v_dot_Gv+
		     CI_nu2_G_ddot_G+
		     1.0e-8);
	  if(den > 1.0e-8)
	    {
	      tau0[eN*nQuadraturePoints_element+k] = 1.0/den;
	      tau1[eN*nQuadraturePoints_element+k] = 1.0/(tau0[eN*nQuadraturePoints_element+k]*g_dot_g);
	    }
	  else
	    {
	      tau0[eN*nQuadraturePoints_element+k] = 0.0;
	      tau1[eN*nQuadraturePoints_element+k] = 0.0;
	    }
	  /*mwf debug
	  if (den < 1.0e-4 || 1)
	    {
	      printf("NSbodyForce tau eN=%d den = %12.5e dmt=%12.5e dr=%12.5e density=%12.5e viscosity= %12.5e\n",eN,den,
		     dmt[eN*nQuadraturePoints_element+k],dr[eN*nQuadraturePoints_element+k],density,viscosity);
	    }
	  */
/*           printf("%12.5e %12.5e \n",tau0[eN*nQuadraturePoints_element+k],tau1[eN*nQuadraturePoints_element+k]); */
        }
    }
/*   printf("Element Re = %12.5e \n",Re_max); */
/*   printf("Element CFL = %12.5e \n",CFL_max); */
}

/**
   \brief Calculate the stabilization parameter for the scalar advection-diffusion-reaction equation using the "l_1 norm formula"
*/
void calculateSubgridErrorNavierStokes2D_GLS_tauRes(proteus_index_t nElements_global,
                                                    int nQuadraturePoints_element,
                                                    int nDOF_trial_element,
                                                    int nSpace,
                                                    double* tau0,
                                                    double* tau1,
                                                    double* pdeResidualP,
                                                    double* dpdeResidualP_du,
                                                    double* dpdeResidualP_dv,
                                                    double* pdeResidualU,
                                                    double* dpdeResidualU_dp,
                                                    double* dpdeResidualU_du,
                                                    double* dpdeResidualU_dv,
                                                    double* pdeResidualV,
                                                    double* dpdeResidualV_dp,
                                                    double* dpdeResidualV_du,
                                                    double* dpdeResidualV_dv,
                                                    double* subgridErrorP,
                                                    double* dsubgridErrorP_du,
                                                    double* dsubgridErrorP_dv,
                                                    double* subgridErrorU,
                                                    double* dsubgridErrorU_dp,
                                                    double* dsubgridErrorU_du,
                                                    double* dsubgridErrorU_dv,
                                                    double* subgridErrorV,
                                                    double* dsubgridErrorV_dp,
                                                    double* dsubgridErrorV_du,
                                                    double* dsubgridErrorV_dv)
{
  int eN,k,j;
  for(eN=0;eN<nElements_global;eN++)
    {
      for (k=0;k<nQuadraturePoints_element;k++)
        {
           /* GLS momentum */
          subgridErrorU[eN*nQuadraturePoints_element+
                       k] =
            -tau0[eN*nQuadraturePoints_element+k]
            *
            pdeResidualU[eN*nQuadraturePoints_element+
                         k];
          subgridErrorV[eN*nQuadraturePoints_element+
                       k] = 
            -tau0[eN*nQuadraturePoints_element+k]
            *
            pdeResidualV[eN*nQuadraturePoints_element+
                         k];
          /* GLS pressure */
          subgridErrorP[eN*nQuadraturePoints_element+
                       k] =
            -tau1[eN*nQuadraturePoints_element+k]
            *pdeResidualP[eN*nQuadraturePoints_element+
                          k];
          for (j=0;j<nDOF_trial_element;j++)
            {
              /* GLS  momentum*/
              /* u */
              dsubgridErrorU_dp[eN*nQuadraturePoints_element*nDOF_trial_element+
                                k*nDOF_trial_element+
                                j] =
                -tau0[eN*nQuadraturePoints_element+k]
                *
                dpdeResidualU_dp[eN*nQuadraturePoints_element*nDOF_trial_element+
                                 k*nDOF_trial_element+
                                 j];
              dsubgridErrorU_du[eN*nQuadraturePoints_element*nDOF_trial_element+
                                k*nDOF_trial_element+
                                j] = 
                -tau0[eN*nQuadraturePoints_element+k]
                *
                dpdeResidualU_du[eN*nQuadraturePoints_element*nDOF_trial_element+
                                 k*nDOF_trial_element+
                                 j];
              dsubgridErrorU_dv[eN*nQuadraturePoints_element*nDOF_trial_element+
                                k*nDOF_trial_element+
                                j] = 
                -tau0[eN*nQuadraturePoints_element+k]
                *
                dpdeResidualU_dv[eN*nQuadraturePoints_element*nDOF_trial_element+
                                 k*nDOF_trial_element+
                                 j];
              /* v */
              dsubgridErrorV_dp[eN*nQuadraturePoints_element*nDOF_trial_element+
                                k*nDOF_trial_element+
                                j] = 
                -tau0[eN*nQuadraturePoints_element+k]
                *
                dpdeResidualV_dp[eN*nQuadraturePoints_element*nDOF_trial_element+
                                 k*nDOF_trial_element+
                                 j];
              dsubgridErrorV_du[eN*nQuadraturePoints_element*nDOF_trial_element+
                                k*nDOF_trial_element+
                                j] = 
                -tau0[eN*nQuadraturePoints_element+k]
                *
                dpdeResidualV_du[eN*nQuadraturePoints_element*nDOF_trial_element+
                                 k*nDOF_trial_element+
                                 j];
              dsubgridErrorV_dv[eN*nQuadraturePoints_element*nDOF_trial_element+
                                k*nDOF_trial_element+
                                j] = 
                -tau0[eN*nQuadraturePoints_element+k]
                *
                dpdeResidualV_dv[eN*nQuadraturePoints_element*nDOF_trial_element+
                                 k*nDOF_trial_element+
                                 j];
               /* GLS pressure */
              dsubgridErrorP_du[eN*nQuadraturePoints_element*nDOF_trial_element+
                                k*nDOF_trial_element+
                                j]
                =
                -tau1[eN*nQuadraturePoints_element+k]
                *
                dpdeResidualP_du[eN*nQuadraturePoints_element*nDOF_trial_element+
                                 k*nDOF_trial_element+
                                 j];
              dsubgridErrorP_dv[eN*nQuadraturePoints_element*nDOF_trial_element+
                                k*nDOF_trial_element+
                                j]
                =
                -tau1[eN*nQuadraturePoints_element+k]
                *
                dpdeResidualP_dv[eN*nQuadraturePoints_element*nDOF_trial_element+
                                 k*nDOF_trial_element+
                                 j];
            }
        }
    }
}

/**
   \brief Calculate the stabilization parameter for the scalar advection-diffusion-reaction equation using the "l_1 norm formula"
*/
void calculateSubgridErrorStokes_GLS_tau(proteus_index_t nElements_global,
                                         int nQuadraturePoints_element,
                                         int nSpace,
                                         double* elementDiameter,
                                         double* pfac,
                                         double* a,
                                         double* tau0,
                                         double* tau1)
{
  proteus_index_t nSpace2=nSpace*nSpace; int eN, k;
  double h,viscosity,density;//caller should scale if kinematic viscosity
  for(eN=0;eN<nElements_global;eN++)
    {
      h = elementDiameter[eN];
      for (k=0;k<nQuadraturePoints_element;k++)
        {
          viscosity =  a[eN*nQuadraturePoints_element*nSpace2 + 
                         k*nSpace2];
          density = pfac[eN*nQuadraturePoints_element*nSpace+k*nSpace+0];
          tau0[eN*nQuadraturePoints_element+k] = 1.0/(4.0*viscosity/(h*h));
          tau1[eN*nQuadraturePoints_element+k] = 4.0*viscosity/density;
         }
    }
}

void calculateSubgridErrorStokes_GLS_tau_sd(proteus_index_t nElements_global,
                                            int nQuadraturePoints_element,
                                            int nSpace,
                                            double* elementDiameter,
                                            double* pfac,
                                            double* a,
                                            double* tau0,
                                            double* tau1)
{
  int eN,k;
  double h,viscosity, density;
  for(eN=0;eN<nElements_global;eN++)
    {
      h = elementDiameter[eN];
      for (k=0;k<nQuadraturePoints_element;k++)
        {
          viscosity =  a[eN*nQuadraturePoints_element*nSpace*nSpace + 
                         k*nSpace*nSpace];
          density = pfac[eN*nQuadraturePoints_element*nSpace+k*nSpace+0];
          tau0[eN*nQuadraturePoints_element+k] = 1.0/(4.0*viscosity/(h*h));
          tau1[eN*nQuadraturePoints_element+k] = 4.0*viscosity/density;
         }
    }
}

/**
   \brief Calculate the stabilization parameter for the scalar advection-diffusion-reaction equation using the "l_1 norm formula"
*/
void calculateSubgridErrorStokes2D_GLS_tauRes(proteus_index_t nElements_global,
                                              int nQuadraturePoints_element,
                                              int nDOF_trial_element,
                                              int nSpace,
                                              double* tau0,
                                              double* tau1,
                                              double* pdeResidualP,
                                              double* dpdeResidualP_du,
                                              double* dpdeResidualP_dv,
                                              double* pdeResidualU,
                                              double* dpdeResidualU_dp,
                                              double* dpdeResidualU_du,
                                              double* pdeResidualV,
                                              double* dpdeResidualV_dp,
                                              double* dpdeResidualV_dv,
                                              double* subgridErrorP,
                                              double* dsubgridErrorP_du,
                                              double* dsubgridErrorP_dv,
                                              double* subgridErrorU,
                                              double* dsubgridErrorU_dp,
                                              double* dsubgridErrorU_du,
                                              double* subgridErrorV,
                                              double* dsubgridErrorV_dp,
                                              double* dsubgridErrorV_dv)
{
  int eN,k,j;
  for(eN=0;eN<nElements_global;eN++)
    {
      for (k=0;k<nQuadraturePoints_element;k++)
        {
           /* GLS momentum */
          subgridErrorU[eN*nQuadraturePoints_element+
                       k] =
            -tau0[eN*nQuadraturePoints_element+k]
            *
            pdeResidualU[eN*nQuadraturePoints_element+
                         k];
          subgridErrorV[eN*nQuadraturePoints_element+
                       k] = 
            -tau0[eN*nQuadraturePoints_element+k]
            *
            pdeResidualV[eN*nQuadraturePoints_element+
                         k];
          /* GLS pressure */
          subgridErrorP[eN*nQuadraturePoints_element+
                       k] =
            -tau1[eN*nQuadraturePoints_element+k]
            *pdeResidualP[eN*nQuadraturePoints_element+
                          k];
          for (j=0;j<nDOF_trial_element;j++)
            {
              /* GLS  momentum*/
              /* u */
              dsubgridErrorU_dp[eN*nQuadraturePoints_element*nDOF_trial_element+
                                k*nDOF_trial_element+
                                j] =
                -tau0[eN*nQuadraturePoints_element+k]
                *
                dpdeResidualU_dp[eN*nQuadraturePoints_element*nDOF_trial_element+
                                 k*nDOF_trial_element+
                                 j];
              dsubgridErrorU_du[eN*nQuadraturePoints_element*nDOF_trial_element+
                                k*nDOF_trial_element+
                                j] = 
                -tau0[eN*nQuadraturePoints_element+k]
                *
                dpdeResidualU_du[eN*nQuadraturePoints_element*nDOF_trial_element+
                                 k*nDOF_trial_element+
                                 j];
              /* v */
              dsubgridErrorV_dp[eN*nQuadraturePoints_element*nDOF_trial_element+
                                k*nDOF_trial_element+
                                j] = 
                -tau0[eN*nQuadraturePoints_element+k]
                *
                dpdeResidualV_dp[eN*nQuadraturePoints_element*nDOF_trial_element+
                                 k*nDOF_trial_element+
                                 j];
              dsubgridErrorV_dv[eN*nQuadraturePoints_element*nDOF_trial_element+
                                k*nDOF_trial_element+
                                j] = 
                -tau0[eN*nQuadraturePoints_element+k]
                *
                dpdeResidualV_dv[eN*nQuadraturePoints_element*nDOF_trial_element+
                                 k*nDOF_trial_element+
                                 j];
              /* GLS pressure */
              dsubgridErrorP_du[eN*nQuadraturePoints_element*nDOF_trial_element+
                                k*nDOF_trial_element+
                                j]
                =
                -tau1[eN*nQuadraturePoints_element+k]
                *
                dpdeResidualP_du[eN*nQuadraturePoints_element*nDOF_trial_element+
                                 k*nDOF_trial_element+
                                 j];
              dsubgridErrorP_dv[eN*nQuadraturePoints_element*nDOF_trial_element+
                                k*nDOF_trial_element+
                                j]
                =
                -tau1[eN*nQuadraturePoints_element+k]
                *
                dpdeResidualP_dv[eN*nQuadraturePoints_element*nDOF_trial_element+
                                 k*nDOF_trial_element+
                                 j];
            }
        }
    }
}
/**
   \brief 3D version of Stokes GLS tau 
*/
void calculateSubgridErrorStokes3D_GLS_tauRes(proteus_index_t nElements_global,
					      int nQuadraturePoints_element,
					      int nDOF_trial_element,
					      int nSpace,
					      double* tau0,
					      double* tau1,
					      double* pdeResidualP,
					      double* dpdeResidualP_du,
					      double* dpdeResidualP_dv,
					      double* dpdeResidualP_dw,
					      double* pdeResidualU,
					      double* dpdeResidualU_dp,
					      double* dpdeResidualU_du,
					      double* pdeResidualV,
					      double* dpdeResidualV_dp,
					      double* dpdeResidualV_dv,
					      double* pdeResidualW,
					      double* dpdeResidualW_dp,
					      double* dpdeResidualW_dw,
					      double* subgridErrorP,
					      double* dsubgridErrorP_du,
					      double* dsubgridErrorP_dv,
					      double* dsubgridErrorP_dw,
					      double* subgridErrorU,
					      double* dsubgridErrorU_dp,
					      double* dsubgridErrorU_du,
					      double* subgridErrorV,
					      double* dsubgridErrorV_dp,
					      double* dsubgridErrorV_dv,
					      double* subgridErrorW,
					      double* dsubgridErrorW_dp,
					      double* dsubgridErrorW_dw)
{
  int eN,k,j;
  for(eN=0;eN<nElements_global;eN++)
    {
      for (k=0;k<nQuadraturePoints_element;k++)
        {
           /* GLS momentum */
          subgridErrorU[eN*nQuadraturePoints_element+
                       k] =
            -tau0[eN*nQuadraturePoints_element+k]
            *
            pdeResidualU[eN*nQuadraturePoints_element+
                         k];
          subgridErrorV[eN*nQuadraturePoints_element+
                       k] = 
            -tau0[eN*nQuadraturePoints_element+k]
            *
            pdeResidualV[eN*nQuadraturePoints_element+
                         k];
          subgridErrorW[eN*nQuadraturePoints_element+
                       k] = 
            -tau0[eN*nQuadraturePoints_element+k]
            *
            pdeResidualW[eN*nQuadraturePoints_element+
                         k];
          /* GLS pressure */
          subgridErrorP[eN*nQuadraturePoints_element+
                       k] =
            -tau1[eN*nQuadraturePoints_element+k]
            *pdeResidualP[eN*nQuadraturePoints_element+
                          k];
          for (j=0;j<nDOF_trial_element;j++)
            {
              /* GLS  momentum*/
              /* u */
              dsubgridErrorU_dp[eN*nQuadraturePoints_element*nDOF_trial_element+
                                k*nDOF_trial_element+
                                j] =
                -tau0[eN*nQuadraturePoints_element+k]
                *
                dpdeResidualU_dp[eN*nQuadraturePoints_element*nDOF_trial_element+
                                 k*nDOF_trial_element+
                                 j];
              dsubgridErrorU_du[eN*nQuadraturePoints_element*nDOF_trial_element+
                                k*nDOF_trial_element+
                                j] = 
                -tau0[eN*nQuadraturePoints_element+k]
                *
                dpdeResidualU_du[eN*nQuadraturePoints_element*nDOF_trial_element+
                                 k*nDOF_trial_element+
                                 j];
              /* v */
              dsubgridErrorV_dp[eN*nQuadraturePoints_element*nDOF_trial_element+
                                k*nDOF_trial_element+
                                j] = 
                -tau0[eN*nQuadraturePoints_element+k]
                *
                dpdeResidualV_dp[eN*nQuadraturePoints_element*nDOF_trial_element+
                                 k*nDOF_trial_element+
                                 j];
              dsubgridErrorV_dv[eN*nQuadraturePoints_element*nDOF_trial_element+
                                k*nDOF_trial_element+
                                j] = 
                -tau0[eN*nQuadraturePoints_element+k]
                *
                dpdeResidualV_dv[eN*nQuadraturePoints_element*nDOF_trial_element+
                                 k*nDOF_trial_element+
                                 j];
              /* w */
              dsubgridErrorW_dp[eN*nQuadraturePoints_element*nDOF_trial_element+
                                k*nDOF_trial_element+
                                j] = 
                -tau0[eN*nQuadraturePoints_element+k]
                *
                dpdeResidualW_dp[eN*nQuadraturePoints_element*nDOF_trial_element+
                                 k*nDOF_trial_element+
                                 j];
              dsubgridErrorW_dw[eN*nQuadraturePoints_element*nDOF_trial_element+
                                k*nDOF_trial_element+
                                j] = 
                -tau0[eN*nQuadraturePoints_element+k]
                *
                dpdeResidualW_dw[eN*nQuadraturePoints_element*nDOF_trial_element+
                                 k*nDOF_trial_element+
                                 j];
                /* GLS pressure */
              dsubgridErrorP_du[eN*nQuadraturePoints_element*nDOF_trial_element+
                                k*nDOF_trial_element+
                                j]
                =
                -tau1[eN*nQuadraturePoints_element+k]
                *
                dpdeResidualP_du[eN*nQuadraturePoints_element*nDOF_trial_element+
                                 k*nDOF_trial_element+
                                 j];
              dsubgridErrorP_dv[eN*nQuadraturePoints_element*nDOF_trial_element+
                                k*nDOF_trial_element+
                                j]
                =
                -tau1[eN*nQuadraturePoints_element+k]
                *
                dpdeResidualP_dv[eN*nQuadraturePoints_element*nDOF_trial_element+
                                 k*nDOF_trial_element+
                                 j];
              dsubgridErrorP_dw[eN*nQuadraturePoints_element*nDOF_trial_element+
                                k*nDOF_trial_element+
                                j]
                =
                -tau1[eN*nQuadraturePoints_element+k]
                *
                dpdeResidualP_dw[eN*nQuadraturePoints_element*nDOF_trial_element+
                                 k*nDOF_trial_element+
                                 j];
	    }
        }
    }
}

/**
   \brief Calculate the stabilization parameter for the scalar advection-diffusion-reaction equation using the "l_1 norm formula"
*/
void calculateSubgridErrorNavierStokes3D_GLS_tauRes(proteus_index_t nElements_global,
                                                    int nQuadraturePoints_element,
                                                    int nDOF_trial_element,
                                                    int nSpace,
                                                    double* tau0,
                                                    double* tau1,
                                                    double* pdeResidualP,
                                                    double* dpdeResidualP_du,
                                                    double* dpdeResidualP_dv,
                                                    double* dpdeResidualP_dw,
                                                    double* pdeResidualU,
                                                    double* dpdeResidualU_dp,
                                                    double* dpdeResidualU_du,
                                                    double* dpdeResidualU_dv,
                                                    double* dpdeResidualU_dw,
                                                    double* pdeResidualV,
                                                    double* dpdeResidualV_dp,
                                                    double* dpdeResidualV_du,
                                                    double* dpdeResidualV_dv,
                                                    double* dpdeResidualV_dw,
                                                    double* pdeResidualW,
                                                    double* dpdeResidualW_dp,
                                                    double* dpdeResidualW_du,
                                                    double* dpdeResidualW_dv,
                                                    double* dpdeResidualW_dw,
                                                    double* subgridErrorP,
                                                    double* dsubgridErrorP_du,
                                                    double* dsubgridErrorP_dv,
                                                    double* dsubgridErrorP_dw,
                                                    double* subgridErrorU,
                                                    double* dsubgridErrorU_dp,
                                                    double* dsubgridErrorU_du,
                                                    double* dsubgridErrorU_dv,
                                                    double* dsubgridErrorU_dw,
                                                    double* subgridErrorV,
                                                    double* dsubgridErrorV_dp,
                                                    double* dsubgridErrorV_du,
                                                    double* dsubgridErrorV_dv,
                                                    double* dsubgridErrorV_dw,
                                                    double* subgridErrorW,
                                                    double* dsubgridErrorW_dp,
                                                    double* dsubgridErrorW_du,
                                                    double* dsubgridErrorW_dv,
                                                    double* dsubgridErrorW_dw)
{
  proteus_index_t nSpace2=nSpace*nSpace; int eN, k, j;
  for(eN=0;eN<nElements_global;eN++)
    {
      for (k=0;k<nQuadraturePoints_element;k++)
        {
          /* GLS momentum */
          subgridErrorU[eN*nQuadraturePoints_element+
                        k] =
            -tau0[eN*nQuadraturePoints_element+k]
            *
            pdeResidualU[eN*nQuadraturePoints_element+
                         k];
          subgridErrorV[eN*nQuadraturePoints_element+
                        k] = 
            -tau0[eN*nQuadraturePoints_element+k]
            *
            pdeResidualV[eN*nQuadraturePoints_element+
                         k];
          subgridErrorW[eN*nQuadraturePoints_element+
                        k] = 
            -tau0[eN*nQuadraturePoints_element+k]
            *
            pdeResidualW[eN*nQuadraturePoints_element+
                         k];
          /* GLS pressure */
          subgridErrorP[eN*nQuadraturePoints_element+
                        k] =
            -tau1[eN*nQuadraturePoints_element+k]
            *pdeResidualP[eN*nQuadraturePoints_element+
                          k];
          for (j=0;j<nDOF_trial_element;j++)
            {
              /* GLS  momentum*/
              /* u */
              dsubgridErrorU_dp[eN*nQuadraturePoints_element*nDOF_trial_element+
                                k*nDOF_trial_element+
                                j] =
                -tau0[eN*nQuadraturePoints_element+k]
                *
                dpdeResidualU_dp[eN*nQuadraturePoints_element*nDOF_trial_element+
                                 k*nDOF_trial_element+
                                 j];
              dsubgridErrorU_du[eN*nQuadraturePoints_element*nDOF_trial_element+
                                k*nDOF_trial_element+
                                j] = 
                -tau0[eN*nQuadraturePoints_element+k]
                *
                dpdeResidualU_du[eN*nQuadraturePoints_element*nDOF_trial_element+
                                 k*nDOF_trial_element+
                                 j];
              dsubgridErrorU_dv[eN*nQuadraturePoints_element*nDOF_trial_element+
                                k*nDOF_trial_element+
                                j] = 
                -tau0[eN*nQuadraturePoints_element+k]
                *
                dpdeResidualU_dv[eN*nQuadraturePoints_element*nDOF_trial_element+
                                 k*nDOF_trial_element+
                                 j];
              dsubgridErrorU_dw[eN*nQuadraturePoints_element*nDOF_trial_element+
                                k*nDOF_trial_element+
                                j] = 
                -tau0[eN*nQuadraturePoints_element+k]
                *
                dpdeResidualU_dw[eN*nQuadraturePoints_element*nDOF_trial_element+
                                 k*nDOF_trial_element+
                                 j];
              /* v */
              dsubgridErrorV_dp[eN*nQuadraturePoints_element*nDOF_trial_element+
                                k*nDOF_trial_element+
                                j] = 
                -tau0[eN*nQuadraturePoints_element+k]
                *
                dpdeResidualV_dp[eN*nQuadraturePoints_element*nDOF_trial_element+
                                 k*nDOF_trial_element+
                                 j];
              dsubgridErrorV_du[eN*nQuadraturePoints_element*nDOF_trial_element+
                                k*nDOF_trial_element+
                                j] = 
                -tau0[eN*nQuadraturePoints_element+k]
                *
                dpdeResidualV_du[eN*nQuadraturePoints_element*nDOF_trial_element+
                                 k*nDOF_trial_element+
                                 j];
              dsubgridErrorV_dv[eN*nQuadraturePoints_element*nDOF_trial_element+
                                k*nDOF_trial_element+
                                j] = 
                -tau0[eN*nQuadraturePoints_element+k]
                *
                dpdeResidualV_dv[eN*nQuadraturePoints_element*nDOF_trial_element+
                                 k*nDOF_trial_element+
                                 j];
              dsubgridErrorV_dw[eN*nQuadraturePoints_element*nDOF_trial_element+
                                k*nDOF_trial_element+
                                j] = 
                -tau0[eN*nQuadraturePoints_element+k]
                *
                dpdeResidualV_dw[eN*nQuadraturePoints_element*nDOF_trial_element+
                                 k*nDOF_trial_element+
                                 j];
              /* w */
              dsubgridErrorW_dp[eN*nQuadraturePoints_element*nDOF_trial_element+
                                k*nDOF_trial_element+
                                j] = 
                -tau0[eN*nQuadraturePoints_element+k]
                *
                dpdeResidualW_dp[eN*nQuadraturePoints_element*nDOF_trial_element+
                                 k*nDOF_trial_element+
                                 j];
              dsubgridErrorW_du[eN*nQuadraturePoints_element*nDOF_trial_element+
                                k*nDOF_trial_element+
                                j] = 
                -tau0[eN*nQuadraturePoints_element+k]
                *
                dpdeResidualW_du[eN*nQuadraturePoints_element*nDOF_trial_element+
                                 k*nDOF_trial_element+
                                 j];
              dsubgridErrorW_dv[eN*nQuadraturePoints_element*nDOF_trial_element+
                                k*nDOF_trial_element+
                                j] = 
                -tau0[eN*nQuadraturePoints_element+k]
                *
                dpdeResidualW_dv[eN*nQuadraturePoints_element*nDOF_trial_element+
                                 k*nDOF_trial_element+
                                 j];
              dsubgridErrorW_dw[eN*nQuadraturePoints_element*nDOF_trial_element+
                                k*nDOF_trial_element+
                                j] = 
                -tau0[eN*nQuadraturePoints_element+k]
                *
                dpdeResidualW_dw[eN*nQuadraturePoints_element*nDOF_trial_element+
                                 k*nDOF_trial_element+
                                 j];
              /* GLS pressure */
              dsubgridErrorP_du[eN*nQuadraturePoints_element*nDOF_trial_element+
                                k*nDOF_trial_element+
                                j]
                =
                -tau1[eN*nQuadraturePoints_element+k]
                *
                dpdeResidualP_du[eN*nQuadraturePoints_element*nDOF_trial_element+
                                 k*nDOF_trial_element+
                                 j];
              dsubgridErrorP_dv[eN*nQuadraturePoints_element*nDOF_trial_element+
                                k*nDOF_trial_element+
                                j]
                =
                -tau1[eN*nQuadraturePoints_element+k]
                *
                dpdeResidualP_dv[eN*nQuadraturePoints_element*nDOF_trial_element+
                                 k*nDOF_trial_element+
                                 j];
              dsubgridErrorP_dw[eN*nQuadraturePoints_element*nDOF_trial_element+
                                k*nDOF_trial_element+
                                j]
                =
                -tau1[eN*nQuadraturePoints_element+k]
                *
                dpdeResidualP_dw[eN*nQuadraturePoints_element*nDOF_trial_element+
                                 k*nDOF_trial_element+
                                 j];
            }
        }
    }
}

/**
   \brief Calculate the stabilization parameter for the scalar advection-diffusion-reaction equation using the "l_1 norm formula"
*/
void calculateSubgridErrorNavierStokes3D_GLS_velocity_pressure(proteus_index_t nElements_global,
                                                               int nQuadraturePoints_element,
                                                               int nDOF_trial_element,
                                                               int nSpace,
                                                               double* elementDiameter,
                                                               double* dm,
                                                               double* f,
                                                               double* a,
                                                               double* pdeResidualP,
                                                               double* dpdeResidualP_du,
                                                               double* dpdeResidualP_dv,
                                                               double* dpdeResidualP_dw,
                                                               double* pdeResidualU,
                                                               double* dpdeResidualU_dp,
                                                               double* dpdeResidualU_du,
                                                               double* pdeResidualV,
                                                               double* dpdeResidualV_dp,
                                                               double* dpdeResidualV_dv,
                                                               double* pdeResidualW,
                                                               double* dpdeResidualW_dp,
                                                               double* dpdeResidualW_dw,
                                                               double* subgridErrorP,
                                                               double* dsubgridErrorP_du,
                                                               double* dsubgridErrorP_dv,
                                                               double* dsubgridErrorP_dw,
                                                               double* subgridErrorU,
                                                               double* dsubgridErrorU_dp,
                                                               double* dsubgridErrorU_du,
                                                               double* subgridErrorV,
                                                               double* dsubgridErrorV_dp,
                                                               double* dsubgridErrorV_dv,
                                                               double* subgridErrorW,
                                                               double* dsubgridErrorW_dp,
                                                               double* dsubgridErrorW_dw)
{
  proteus_index_t nSpace2=nSpace*nSpace; int eN, k, j;
  double h,viscosity,tau1,tau2;
  for(eN=0;eN<nElements_global;eN++)
    {
      h = elementDiameter[eN];
      for (k=0;k<nQuadraturePoints_element;k++)
        {
          viscosity =  a[eN*nQuadraturePoints_element*nSpace2 + 
                         k*nSpace2]; 
          /* GLS momentum */
          tau1 = h*h/(12.0*viscosity);
          subgridErrorU[eN*nQuadraturePoints_element+
                       k] =
            tau1
            *
            pdeResidualU[eN*nQuadraturePoints_element+
                         k];
          subgridErrorV[eN*nQuadraturePoints_element+
                       k] = 
            tau1
            *
            pdeResidualV[eN*nQuadraturePoints_element+
                         k];
          subgridErrorW[eN*nQuadraturePoints_element+
                       k] = 
            tau1
            *
            pdeResidualW[eN*nQuadraturePoints_element+
                         k];
          /* GLS pressure */
          tau2 = 6.0*viscosity;
          subgridErrorP[eN*nQuadraturePoints_element+
                       k] =
            tau2
            *pdeResidualP[eN*nQuadraturePoints_element+
                          k];
          for (j=0;j<nDOF_trial_element;j++)
            {
              /* GLS  momentum*/
              /* u */
              dsubgridErrorU_dp[eN*nQuadraturePoints_element*nDOF_trial_element+
                                k*nDOF_trial_element+
                                j] =
                tau1
                *
                dpdeResidualU_dp[eN*nQuadraturePoints_element*nDOF_trial_element+
                                 k*nDOF_trial_element+
                                 j];
              dsubgridErrorU_du[eN*nQuadraturePoints_element*nDOF_trial_element+
                                k*nDOF_trial_element+
                                j] = 
                tau1
                *
                dpdeResidualU_du[eN*nQuadraturePoints_element*nDOF_trial_element+
                                 k*nDOF_trial_element+
                                 j];
              /* v */
              dsubgridErrorV_dp[eN*nQuadraturePoints_element*nDOF_trial_element+
                                k*nDOF_trial_element+
                                j] = 
                tau1
                *
                dpdeResidualV_dp[eN*nQuadraturePoints_element*nDOF_trial_element+
                                 k*nDOF_trial_element+
                                 j];
              dsubgridErrorV_dv[eN*nQuadraturePoints_element*nDOF_trial_element+
                                k*nDOF_trial_element+
                                j] = 
                tau1
                *
                dpdeResidualV_dv[eN*nQuadraturePoints_element*nDOF_trial_element+
                                 k*nDOF_trial_element+
                                 j];
              /* w */
              dsubgridErrorW_dp[eN*nQuadraturePoints_element*nDOF_trial_element+
                                k*nDOF_trial_element+
                                j] = 
                tau1
                *
                dpdeResidualW_dp[eN*nQuadraturePoints_element*nDOF_trial_element+
                                 k*nDOF_trial_element+
                                 j];
              dsubgridErrorW_dw[eN*nQuadraturePoints_element*nDOF_trial_element+
                                k*nDOF_trial_element+
                                j] = 
                tau1
                *
                dpdeResidualW_dw[eN*nQuadraturePoints_element*nDOF_trial_element+
                                 k*nDOF_trial_element+
                                 j];
              /* GLS pressure */
              dsubgridErrorP_du[eN*nQuadraturePoints_element*nDOF_trial_element+
                                k*nDOF_trial_element+
                                j]
                =
                tau2
                *
                dpdeResidualP_du[eN*nQuadraturePoints_element*nDOF_trial_element+
                                 k*nDOF_trial_element+
                                 j];
              dsubgridErrorP_dv[eN*nQuadraturePoints_element*nDOF_trial_element+
                                k*nDOF_trial_element+
                                j]
                =
                tau2
                *
                dpdeResidualP_dv[eN*nQuadraturePoints_element*nDOF_trial_element+
                                 k*nDOF_trial_element+
                                 j];
            }
        }
    }
}

void calculateSubgridErrorNavierStokes3D_GLS_velocity_pressure_sd(proteus_index_t nElements_global,
                                                               int nQuadraturePoints_element,
                                                               int nDOF_trial_element,
                                                               int nSpace,
                                                               double* elementDiameter,
                                                               double* dm,
                                                               double* f,
                                                               double* a,
                                                               double* pdeResidualP,
                                                               double* dpdeResidualP_du,
                                                               double* dpdeResidualP_dv,
                                                               double* dpdeResidualP_dw,
                                                               double* pdeResidualU,
                                                               double* dpdeResidualU_dp,
                                                               double* dpdeResidualU_du,
                                                               double* pdeResidualV,
                                                               double* dpdeResidualV_dp,
                                                               double* dpdeResidualV_dv,
                                                               double* pdeResidualW,
                                                               double* dpdeResidualW_dp,
                                                               double* dpdeResidualW_dw,
                                                               double* subgridErrorP,
                                                               double* dsubgridErrorP_du,
                                                               double* dsubgridErrorP_dv,
                                                               double* dsubgridErrorP_dw,
                                                               double* subgridErrorU,
                                                               double* dsubgridErrorU_dp,
                                                               double* dsubgridErrorU_du,
                                                               double* subgridErrorV,
                                                               double* dsubgridErrorV_dp,
                                                               double* dsubgridErrorV_dv,
                                                               double* subgridErrorW,
                                                               double* dsubgridErrorW_dp,
                                                               double* dsubgridErrorW_dw)
{
  int eN,k,j;
  double h,viscosity,tau1,tau2;
  for(eN=0;eN<nElements_global;eN++)
    {
      h = elementDiameter[eN];
      for (k=0;k<nQuadraturePoints_element;k++)
        {
          viscosity =  a[eN*nQuadraturePoints_element*nSpace + 
                         k*nSpace]; 
          /* GLS momentum */
          tau1 = h*h/(12.0*viscosity);
          subgridErrorU[eN*nQuadraturePoints_element+
                       k] =
            tau1
            *
            pdeResidualU[eN*nQuadraturePoints_element+
                         k];
          subgridErrorV[eN*nQuadraturePoints_element+
                       k] = 
            tau1
            *
            pdeResidualV[eN*nQuadraturePoints_element+
                         k];
          subgridErrorW[eN*nQuadraturePoints_element+
                       k] = 
            tau1
            *
            pdeResidualW[eN*nQuadraturePoints_element+
                         k];
          /* GLS pressure */
          tau2 = 6.0*viscosity;
          subgridErrorP[eN*nQuadraturePoints_element+
                       k] =
            tau2
            *pdeResidualP[eN*nQuadraturePoints_element+
                          k];
          for (j=0;j<nDOF_trial_element;j++)
            {
              /* GLS  momentum*/
              /* u */
              dsubgridErrorU_dp[eN*nQuadraturePoints_element*nDOF_trial_element+
                                k*nDOF_trial_element+
                                j] =
                tau1
                *
                dpdeResidualU_dp[eN*nQuadraturePoints_element*nDOF_trial_element+
                                 k*nDOF_trial_element+
                                 j];
              dsubgridErrorU_du[eN*nQuadraturePoints_element*nDOF_trial_element+
                                k*nDOF_trial_element+
                                j] = 
                tau1
                *
                dpdeResidualU_du[eN*nQuadraturePoints_element*nDOF_trial_element+
                                 k*nDOF_trial_element+
                                 j];
              /* v */
              dsubgridErrorV_dp[eN*nQuadraturePoints_element*nDOF_trial_element+
                                k*nDOF_trial_element+
                                j] = 
                tau1
                *
                dpdeResidualV_dp[eN*nQuadraturePoints_element*nDOF_trial_element+
                                 k*nDOF_trial_element+
                                 j];
              dsubgridErrorV_dv[eN*nQuadraturePoints_element*nDOF_trial_element+
                                k*nDOF_trial_element+
                                j] = 
                tau1
                *
                dpdeResidualV_dv[eN*nQuadraturePoints_element*nDOF_trial_element+
                                 k*nDOF_trial_element+
                                 j];
              /* w */
              dsubgridErrorW_dp[eN*nQuadraturePoints_element*nDOF_trial_element+
                                k*nDOF_trial_element+
                                j] = 
                tau1
                *
                dpdeResidualW_dp[eN*nQuadraturePoints_element*nDOF_trial_element+
                                 k*nDOF_trial_element+
                                 j];
              dsubgridErrorW_dw[eN*nQuadraturePoints_element*nDOF_trial_element+
                                k*nDOF_trial_element+
                                j] = 
                tau1
                *
                dpdeResidualW_dw[eN*nQuadraturePoints_element*nDOF_trial_element+
                                 k*nDOF_trial_element+
                                 j];
              /* GLS pressure */
              dsubgridErrorP_du[eN*nQuadraturePoints_element*nDOF_trial_element+
                                k*nDOF_trial_element+
                                j]
                =
                tau2
                *
                dpdeResidualP_du[eN*nQuadraturePoints_element*nDOF_trial_element+
                                 k*nDOF_trial_element+
                                 j];
              dsubgridErrorP_dv[eN*nQuadraturePoints_element*nDOF_trial_element+
                                k*nDOF_trial_element+
                                j]
                =
                tau2
                *
                dpdeResidualP_dv[eN*nQuadraturePoints_element*nDOF_trial_element+
                                 k*nDOF_trial_element+
                                 j];
            }
        }
    }
}

/**
   \brief Calculate the stabilization parameter for the scalar advection-diffusion-reaction equation using the "l_1 norm formula"
*/
void calculateSubgridErrorStokes2D_1(proteus_index_t nElements_global,
                                     int nQuadraturePoints_element,
                                     int nDOF_trial_element,
                                     int nSpace,
                                     double* elementDiameter,
                                     double* u,
                                     double* v,
                                     double* a,
                                     double* pdeResidualP,
                                     double* dpdeResidualP_du,
                                     double* dpdeResidualP_dv,
                                     double* pdeResidualU,
                                     double* dpdeResidualU_dp,
                                     double* dpdeResidualU_du,
                                     double* pdeResidualV,
                                     double* dpdeResidualV_dp,
                                     double* dpdeResidualV_dv,
                                     double* subgridErrorP,
                                     double* dsubgridErrorP_dp,
                                     double* dsubgridErrorP_du,
                                     double* dsubgridErrorP_dv,
                                     double* subgridErrorU,
                                     double* dsubgridErrorU_dp,
                                     double* dsubgridErrorU_du,
                                     double* dsubgridErrorU_dv,
                                     double* subgridErrorV,
                                     double* dsubgridErrorV_dp,
                                     double* dsubgridErrorV_du,
                                     double* dsubgridErrorV_dv)
{
  proteus_index_t nSpace2=nSpace*nSpace; int eN, k, j;
  double h,velocity_norm,viscosity,tau1,tau2,U,V;
  for(eN=0;eN<nElements_global;eN++)
    {
      h = elementDiameter[eN];
      for (k=0;k<nQuadraturePoints_element;k++)
        {
          U = u[eN*nQuadraturePoints_element+ 
                k];
          V = v[eN*nQuadraturePoints_element+ 
                k];
          velocity_norm = sqrt(U*U + V*V);
          viscosity =  a[eN*nQuadraturePoints_element*nSpace2 + 
                         k*nSpace2]; 
          tau1 = 1.0/(4.0*viscosity/(h*h)+2.0*velocity_norm/h+1.0e-8);
          tau2 = 4.0*viscosity+2.0*velocity_norm*h;
/*           subgridErrorP[eN*nQuadraturePoints_element+ */
/*                        k] = */
/*             tau1 */
/*             * */
/*             (pdeResidualU[eN*nQuadraturePoints_element+ */
/*                           k] + */
/*              pdeResidualV[eN*nQuadraturePoints_element+ */
/*                           k]); */
/*           subgridErrorU[eN*nQuadraturePoints_element+ */
/*                        k] = */
/*             tau1 */
/*             * */
/*             pdeResidualU[eN*nQuadraturePoints_element+ */
/*                         k] */
/*             + */
/*             tau2 */
/*             * */
/*             pdeResidualP[eN*nQuadraturePoints_element+ */
/*                          k]; */
/*           subgridErrorV[eN*nQuadraturePoints_element+ */
/*                        k] =  */
/*             tau1 */
/*             * */
/*             pdeResidualV[eN*nQuadraturePoints_element+ */
/*                          k] */
/*             + */
/*             tau2 */
/*             * */
/*             pdeResidualP[eN*nQuadraturePoints_element+ */
/*                          k]; */
          /* GLS momentum */
          tau1 = h*h/(12.0*viscosity);
          subgridErrorU[eN*nQuadraturePoints_element+
                       k] =
            tau1
            *
            pdeResidualU[eN*nQuadraturePoints_element+
                         k];
          subgridErrorV[eN*nQuadraturePoints_element+
                       k] = 
            tau1
            *
            pdeResidualV[eN*nQuadraturePoints_element+
                         k];
          /* GLS pressure */
          tau2 = 6.0*viscosity;
          subgridErrorP[eN*nQuadraturePoints_element+
                       k] =
            tau2
            *pdeResidualP[eN*nQuadraturePoints_element+
                          k];
          for (j=0;j<nDOF_trial_element;j++)
            {
 /*              /\* p *\/ */
/*               dsubgridErrorP_dp[eN*nQuadraturePoints_element*nDOF_trial_element+ */
/*                                 k*nDOF_trial_element+ */
/*                                 j] */
/*                 = */
/*                 tau1 */
/*                 * */
/*                 (dpdeResidualU_dp[eN*nQuadraturePoints_element*nDOF_trial_element+ */
/*                                   k*nDOF_trial_element+ */
/*                                   j] + */
/*                  dpdeResidualV_dp[eN*nQuadraturePoints_element*nDOF_trial_element+ */
/*                                   k*nDOF_trial_element+ */
/*                                   j]); */
/*               dsubgridErrorP_du[eN*nQuadraturePoints_element*nDOF_trial_element+ */
/*                                 k*nDOF_trial_element+ */
/*                                 j] */
/*                 = */
/*                 tau1 */
/*                 * */
/*                 dpdeResidualU_du[eN*nQuadraturePoints_element*nDOF_trial_element+ */
/*                                   k*nDOF_trial_element+ */
/*                                  j]; */
/*               dsubgridErrorP_dv[eN*nQuadraturePoints_element*nDOF_trial_element+ */
/*                                 k*nDOF_trial_element+ */
/*                                 j] */
/*                 = */
/*                 tau1 */
/*                 * */
/*                 dpdeResidualV_dv[eN*nQuadraturePoints_element*nDOF_trial_element+ */
/*                                  k*nDOF_trial_element+ */
/*                                  j]; */
/*               /\* u *\/ */
/*               dsubgridErrorU_dp[eN*nQuadraturePoints_element*nDOF_trial_element+ */
/*                                 k*nDOF_trial_element+ */
/*                                 j] = */
/*                 tau1 */
/*                 * */
/*                 dpdeResidualU_dp[eN*nQuadraturePoints_element*nDOF_trial_element+ */
/*                                  k*nDOF_trial_element+ */
/*                                  j]; */
/*               dsubgridErrorU_du[eN*nQuadraturePoints_element*nDOF_trial_element+ */
/*                                 k*nDOF_trial_element+ */
/*                                 j] =  */
/*                 tau1 */
/*                 * */
/*                 dpdeResidualU_du[eN*nQuadraturePoints_element*nDOF_trial_element+ */
/*                                  k*nDOF_trial_element+ */
/*                                  j] */
/*                 + */
/*                 tau2 */
/*                 * */
/*                 dpdeResidualP_du[eN*nQuadraturePoints_element*nDOF_trial_element+ */
/*                                  k*nDOF_trial_element+ */
/*                                  j]; */
/*               dsubgridErrorU_dv[eN*nQuadraturePoints_element*nDOF_trial_element+ */
/*                                 k*nDOF_trial_element+ */
/*                                 j] =  */
/*                 tau2 */
/*                 * */
/*                 dpdeResidualP_dv[eN*nQuadraturePoints_element*nDOF_trial_element+ */
/*                              k*nDOF_trial_element+ */
/*                                 j]; */
/*               /\* v *\/ */
/*               dsubgridErrorV_dp[eN*nQuadraturePoints_element*nDOF_trial_element+ */
/*                                 k*nDOF_trial_element+ */
/*                                 j] =  */
/*                 tau1 */
/*                 * */
/*                 dpdeResidualV_dp[eN*nQuadraturePoints_element*nDOF_trial_element+ */
/*                                  k*nDOF_trial_element+ */
/*                                  j]; */
/*               dsubgridErrorV_du[eN*nQuadraturePoints_element*nDOF_trial_element+ */
/*                                 k*nDOF_trial_element+ */
/*                                 j] =  */
/*                 tau2 */
/*                 * */
/*                 dpdeResidualP_du[eN*nQuadraturePoints_element*nDOF_trial_element+ */
/*                                  k*nDOF_trial_element+ */
/*                                  j]; */
/*               dsubgridErrorV_dv[eN*nQuadraturePoints_element*nDOF_trial_element+ */
/*                                 k*nDOF_trial_element+ */
/*                                 j] =  */
/*                 tau1 */
/*                 * */
/*                 dpdeResidualV_dv[eN*nQuadraturePoints_element*nDOF_trial_element+ */
/*                                  k*nDOF_trial_element+ */
/*                                  j] */
/*                 + */
/*                 tau2 */
/*                 * */
/*                 dpdeResidualP_dv[eN*nQuadraturePoints_element*nDOF_trial_element+ */
/*                                  k*nDOF_trial_element+ */
/*                                  j]; */
              /* GLS  momentum*/
              /* u */
              dsubgridErrorU_dp[eN*nQuadraturePoints_element*nDOF_trial_element+
                                k*nDOF_trial_element+
                                j] =
                tau1
                *
                dpdeResidualU_dp[eN*nQuadraturePoints_element*nDOF_trial_element+
                                 k*nDOF_trial_element+
                                 j];
              dsubgridErrorU_du[eN*nQuadraturePoints_element*nDOF_trial_element+
                                k*nDOF_trial_element+
                                j] = 
                tau1
                *
                dpdeResidualU_du[eN*nQuadraturePoints_element*nDOF_trial_element+
                                 k*nDOF_trial_element+
                                 j];
              /* v */
              dsubgridErrorV_dp[eN*nQuadraturePoints_element*nDOF_trial_element+
                                k*nDOF_trial_element+
                                j] = 
                tau1
                *
                dpdeResidualV_dp[eN*nQuadraturePoints_element*nDOF_trial_element+
                                 k*nDOF_trial_element+
                                 j];
              dsubgridErrorV_dv[eN*nQuadraturePoints_element*nDOF_trial_element+
                                k*nDOF_trial_element+
                                j] = 
                tau1
                *
                dpdeResidualV_dv[eN*nQuadraturePoints_element*nDOF_trial_element+
                                 k*nDOF_trial_element+
                                 j];
              /* GLS pressure */
              dsubgridErrorP_du[eN*nQuadraturePoints_element*nDOF_trial_element+
                                k*nDOF_trial_element+
                                j]
                =
                tau2
                *
                dpdeResidualP_du[eN*nQuadraturePoints_element*nDOF_trial_element+
                                 k*nDOF_trial_element+
                                 j];
              dsubgridErrorP_dv[eN*nQuadraturePoints_element*nDOF_trial_element+
                                k*nDOF_trial_element+
                                j]
                =
                tau2
                *
                dpdeResidualP_dv[eN*nQuadraturePoints_element*nDOF_trial_element+
                                 k*nDOF_trial_element+
                                 j];
            }
        }
    }
}

void calculateSubgridErrorStokes2D_1_sd(proteus_index_t nElements_global,
                                     int nQuadraturePoints_element,
                                     int nDOF_trial_element,
                                     int nSpace,
                                     double* elementDiameter,
                                     double* u,
                                     double* v,
                                     double* a,
                                     double* pdeResidualP,
                                     double* dpdeResidualP_du,
                                     double* dpdeResidualP_dv,
                                     double* pdeResidualU,
                                     double* dpdeResidualU_dp,
                                     double* dpdeResidualU_du,
                                     double* pdeResidualV,
                                     double* dpdeResidualV_dp,
                                     double* dpdeResidualV_dv,
                                     double* subgridErrorP,
                                     double* dsubgridErrorP_dp,
                                     double* dsubgridErrorP_du,
                                     double* dsubgridErrorP_dv,
                                     double* subgridErrorU,
                                     double* dsubgridErrorU_dp,
                                     double* dsubgridErrorU_du,
                                     double* dsubgridErrorU_dv,
                                     double* subgridErrorV,
                                     double* dsubgridErrorV_dp,
                                     double* dsubgridErrorV_du,
                                     double* dsubgridErrorV_dv)
{
  int eN,k,j;
  double h,velocity_norm,viscosity,tau1,tau2,U,V;
  for(eN=0;eN<nElements_global;eN++)
    {
      h = elementDiameter[eN];
      for (k=0;k<nQuadraturePoints_element;k++)
        {
          U = u[eN*nQuadraturePoints_element+ 
                k];
          V = v[eN*nQuadraturePoints_element+ 
                k];
          velocity_norm = sqrt(U*U + V*V);
          viscosity =  a[eN*nQuadraturePoints_element*nSpace + 
                         k*nSpace]; 
          tau1 = 1.0/(4.0*viscosity/(h*h)+2.0*velocity_norm/h+1.0e-8);
          tau2 = 4.0*viscosity+2.0*velocity_norm*h;
/*           subgridErrorP[eN*nQuadraturePoints_element+ */
/*                        k] = */
/*             tau1 */
/*             * */
/*             (pdeResidualU[eN*nQuadraturePoints_element+ */
/*                           k] + */
/*              pdeResidualV[eN*nQuadraturePoints_element+ */
/*                           k]); */
/*           subgridErrorU[eN*nQuadraturePoints_element+ */
/*                        k] = */
/*             tau1 */
/*             * */
/*             pdeResidualU[eN*nQuadraturePoints_element+ */
/*                         k] */
/*             + */
/*             tau2 */
/*             * */
/*             pdeResidualP[eN*nQuadraturePoints_element+ */
/*                          k]; */
/*           subgridErrorV[eN*nQuadraturePoints_element+ */
/*                        k] =  */
/*             tau1 */
/*             * */
/*             pdeResidualV[eN*nQuadraturePoints_element+ */
/*                          k] */
/*             + */
/*             tau2 */
/*             * */
/*             pdeResidualP[eN*nQuadraturePoints_element+ */
/*                          k]; */
          /* GLS momentum */
          tau1 = h*h/(12.0*viscosity);
          subgridErrorU[eN*nQuadraturePoints_element+
                       k] =
            tau1
            *
            pdeResidualU[eN*nQuadraturePoints_element+
                         k];
          subgridErrorV[eN*nQuadraturePoints_element+
                       k] = 
            tau1
            *
            pdeResidualV[eN*nQuadraturePoints_element+
                         k];
          /* GLS pressure */
          tau2 = 6.0*viscosity;
          subgridErrorP[eN*nQuadraturePoints_element+
                       k] =
            tau2
            *pdeResidualP[eN*nQuadraturePoints_element+
                          k];
          for (j=0;j<nDOF_trial_element;j++)
            {
 /*              /\* p *\/ */
/*               dsubgridErrorP_dp[eN*nQuadraturePoints_element*nDOF_trial_element+ */
/*                                 k*nDOF_trial_element+ */
/*                                 j] */
/*                 = */
/*                 tau1 */
/*                 * */
/*                 (dpdeResidualU_dp[eN*nQuadraturePoints_element*nDOF_trial_element+ */
/*                                   k*nDOF_trial_element+ */
/*                                   j] + */
/*                  dpdeResidualV_dp[eN*nQuadraturePoints_element*nDOF_trial_element+ */
/*                                   k*nDOF_trial_element+ */
/*                                   j]); */
/*               dsubgridErrorP_du[eN*nQuadraturePoints_element*nDOF_trial_element+ */
/*                                 k*nDOF_trial_element+ */
/*                                 j] */
/*                 = */
/*                 tau1 */
/*                 * */
/*                 dpdeResidualU_du[eN*nQuadraturePoints_element*nDOF_trial_element+ */
/*                                   k*nDOF_trial_element+ */
/*                                  j]; */
/*               dsubgridErrorP_dv[eN*nQuadraturePoints_element*nDOF_trial_element+ */
/*                                 k*nDOF_trial_element+ */
/*                                 j] */
/*                 = */
/*                 tau1 */
/*                 * */
/*                 dpdeResidualV_dv[eN*nQuadraturePoints_element*nDOF_trial_element+ */
/*                                  k*nDOF_trial_element+ */
/*                                  j]; */
/*               /\* u *\/ */
/*               dsubgridErrorU_dp[eN*nQuadraturePoints_element*nDOF_trial_element+ */
/*                                 k*nDOF_trial_element+ */
/*                                 j] = */
/*                 tau1 */
/*                 * */
/*                 dpdeResidualU_dp[eN*nQuadraturePoints_element*nDOF_trial_element+ */
/*                                  k*nDOF_trial_element+ */
/*                                  j]; */
/*               dsubgridErrorU_du[eN*nQuadraturePoints_element*nDOF_trial_element+ */
/*                                 k*nDOF_trial_element+ */
/*                                 j] =  */
/*                 tau1 */
/*                 * */
/*                 dpdeResidualU_du[eN*nQuadraturePoints_element*nDOF_trial_element+ */
/*                                  k*nDOF_trial_element+ */
/*                                  j] */
/*                 + */
/*                 tau2 */
/*                 * */
/*                 dpdeResidualP_du[eN*nQuadraturePoints_element*nDOF_trial_element+ */
/*                                  k*nDOF_trial_element+ */
/*                                  j]; */
/*               dsubgridErrorU_dv[eN*nQuadraturePoints_element*nDOF_trial_element+ */
/*                                 k*nDOF_trial_element+ */
/*                                 j] =  */
/*                 tau2 */
/*                 * */
/*                 dpdeResidualP_dv[eN*nQuadraturePoints_element*nDOF_trial_element+ */
/*                              k*nDOF_trial_element+ */
/*                                 j]; */
/*               /\* v *\/ */
/*               dsubgridErrorV_dp[eN*nQuadraturePoints_element*nDOF_trial_element+ */
/*                                 k*nDOF_trial_element+ */
/*                                 j] =  */
/*                 tau1 */
/*                 * */
/*                 dpdeResidualV_dp[eN*nQuadraturePoints_element*nDOF_trial_element+ */
/*                                  k*nDOF_trial_element+ */
/*                                  j]; */
/*               dsubgridErrorV_du[eN*nQuadraturePoints_element*nDOF_trial_element+ */
/*                                 k*nDOF_trial_element+ */
/*                                 j] =  */
/*                 tau2 */
/*                 * */
/*                 dpdeResidualP_du[eN*nQuadraturePoints_element*nDOF_trial_element+ */
/*                                  k*nDOF_trial_element+ */
/*                                  j]; */
/*               dsubgridErrorV_dv[eN*nQuadraturePoints_element*nDOF_trial_element+ */
/*                                 k*nDOF_trial_element+ */
/*                                 j] =  */
/*                 tau1 */
/*                 * */
/*                 dpdeResidualV_dv[eN*nQuadraturePoints_element*nDOF_trial_element+ */
/*                                  k*nDOF_trial_element+ */
/*                                  j] */
/*                 + */
/*                 tau2 */
/*                 * */
/*                 dpdeResidualP_dv[eN*nQuadraturePoints_element*nDOF_trial_element+ */
/*                                  k*nDOF_trial_element+ */
/*                                  j]; */
              /* GLS  momentum*/
              /* u */
              dsubgridErrorU_dp[eN*nQuadraturePoints_element*nDOF_trial_element+
                                k*nDOF_trial_element+
                                j] =
                tau1
                *
                dpdeResidualU_dp[eN*nQuadraturePoints_element*nDOF_trial_element+
                                 k*nDOF_trial_element+
                                 j];
              dsubgridErrorU_du[eN*nQuadraturePoints_element*nDOF_trial_element+
                                k*nDOF_trial_element+
                                j] = 
                tau1
                *
                dpdeResidualU_du[eN*nQuadraturePoints_element*nDOF_trial_element+
                                 k*nDOF_trial_element+
                                 j];
              /* v */
              dsubgridErrorV_dp[eN*nQuadraturePoints_element*nDOF_trial_element+
                                k*nDOF_trial_element+
                                j] = 
                tau1
                *
                dpdeResidualV_dp[eN*nQuadraturePoints_element*nDOF_trial_element+
                                 k*nDOF_trial_element+
                                 j];
              dsubgridErrorV_dv[eN*nQuadraturePoints_element*nDOF_trial_element+
                                k*nDOF_trial_element+
                                j] = 
                tau1
                *
                dpdeResidualV_dv[eN*nQuadraturePoints_element*nDOF_trial_element+
                                 k*nDOF_trial_element+
                                 j];
              /* GLS pressure */
              dsubgridErrorP_du[eN*nQuadraturePoints_element*nDOF_trial_element+
                                k*nDOF_trial_element+
                                j]
                =
                tau2
                *
                dpdeResidualP_du[eN*nQuadraturePoints_element*nDOF_trial_element+
                                 k*nDOF_trial_element+
                                 j];
              dsubgridErrorP_dv[eN*nQuadraturePoints_element*nDOF_trial_element+
                                k*nDOF_trial_element+
                                j]
                =
                tau2
                *
                dpdeResidualP_dv[eN*nQuadraturePoints_element*nDOF_trial_element+
                                 k*nDOF_trial_element+
                                 j];
            }
        }
    }
}

void calculateSubgridErrorShallowWater1D(proteus_index_t nElements_global,
                                         int nQuadraturePoints_element,
                                         double g,
                                         double* elementDiameter,
                                         double* h,
                                         double* hu,
                                         double* cfl_1,
                                         double* cfl_2)
{
  int eN,k;
  double u,c;
  for(eN=0;eN<nElements_global;eN++)
    {
      for (k=0;k<nQuadraturePoints_element;k++)
        {
          u = hu[eN*nQuadraturePoints_element+k]/fmax(h[eN*nQuadraturePoints_element+k],1.0e-8);
          c = sqrt(fabs(g*h[eN*nQuadraturePoints_element+k]));
          cfl_1[eN*nQuadraturePoints_element+k] = (u-c)/elementDiameter[eN];
          cfl_2[eN*nQuadraturePoints_element+k] = (u+c)/elementDiameter[eN];
        }
    }
}
void calculateSubgridErrorShallowWater2D(proteus_index_t nElements_global,
                                         int nQuadraturePoints_element,
                                         double g,
                                         double* elementDiameter,
                                         double* h,
                                         double* hu,
                                         double* hv,
                                         double* cfl_1,
                                         double* cfl_2,
                                         double* cfl_3)
{
  int eN,k;
  double u,v,speed,c;
  for(eN=0;eN<nElements_global;eN++)
    {
      for (k=0;k<nQuadraturePoints_element;k++)
        {
          u = hu[eN*nQuadraturePoints_element+k]/fmax(h[eN*nQuadraturePoints_element+k],1.0e-8);
          v = hv[eN*nQuadraturePoints_element+k]/fmax(h[eN*nQuadraturePoints_element+k],1.0e-8);
          speed = sqrt(u*u + v*v);
          c = sqrt(fabs(g*h[eN*nQuadraturePoints_element+k]));
          cfl_1[eN*nQuadraturePoints_element+k] = (speed-c)/elementDiameter[eN];
          cfl_2[eN*nQuadraturePoints_element+k] = speed/elementDiameter[eN];
          cfl_3[eN*nQuadraturePoints_element+k] = (speed+c)/elementDiameter[eN];
        }
    }
}

/*this should reduce to harari's tau * dt in the linear case*/
void calculateSubgridError_Harari_tau_sd(proteus_index_t nElements_global,
					 int nQuadraturePoints_element,
					 int nSpace,
					 double dt,
					 proteus_csr_index_t* rowptr,
					 proteus_csr_index_t* colind,
					 double* elementDiameter,
					 double* a,
					 double* tau)
{
  proteus_csr_index_t m; proteus_csr_index_t nnz=rowptr[nSpace]; int eN, k, I;
  double h,A_max,A_II,tauMax,heps;
  for(eN=0;eN<nElements_global;eN++)
    {
      h = elementDiameter[eN];
      tauMax=0.0;
      for (k=0;k<nQuadraturePoints_element;k++)
        {
	  A_max = 0.0;
          /*max diagonal entry for A. This choice needs to be looked into*/
          for(I=0;I<nSpace;I++)
            {
	      for(m=rowptr[I];m<rowptr[I+1];m++)
		{
		  if(I==colind[m])
		    {
		      A_II = a[eN*nQuadraturePoints_element*nnz+
			       k*nnz+
			       m];
		      A_max = (A_II > A_max) ? A_II : A_max;
		    }
		}
            }
	  if (A_max > 0.0)
	    {
	      heps = h/(A_max*dt);
	      heps = (heps > 20.0) ? 20.0 : heps;
	    }
          tau[eN*nQuadraturePoints_element + k]=dt + 6.0*A_max*dt*dt/(h*h)*(1.0-cosh(heps))/(2.0+cosh(heps));
        }
    }
}


/**
   \brief Calculate the ASGS subgrid error given tau and the strong residual
*/
void calculateSubgridErrorGradient_tauRes(proteus_index_t nElements_global,
					  int nQuadraturePoints_element,
					  int nSpace,
					  double* tau_gradient,
					  double* grad_pdeResidual,
					  double* grad_subgridError)
{
  int eN,k,j,I;
   for(eN=0;eN<nElements_global;eN++)
     for (k=0;k<nQuadraturePoints_element;k++)
       {
	 for (I=0; I < nSpace; I++)
	   {
	     grad_subgridError[eN*nQuadraturePoints_element*nSpace+
			       k*nSpace + I] = 
	       -tau_gradient[eN*nQuadraturePoints_element+
			     k]
	       *
	       grad_pdeResidual[eN*nQuadraturePoints_element*nSpace+
				k*nSpace + I];
	   }	     
	     
       }
}



void calculateSubgridError_ADR_Sangalli_tau(proteus_index_t nElements_global,
					    int nQuadraturePoints_element,
					    int nSpace,
					    double* inverseJ,
					    double* dmt,
					    double* df,
					    double* a,
					    double* da,
					    double* grad_phi,
					    double* dphi,
					    double* dr,
					    double* pe,
					    double* cfl,
					    double* tau,
					    double* tau_gradient)
{
  proteus_index_t nSpace2=nSpace*nSpace; int eN, k, I, J, K;
  double Vlin,Alin,A_II,cfl1,cfl2,Pe,Da,vlin[3],G_IJ,v_dot_Gv,CI_nu2_G_ddot_G,Gv_I,CI=36.0*36.0,g_I,g_dot_g;
  double h_e,tau_00,tau_11,t_00,t_11,coshGamma;
  for(eN=0;eN<nElements_global;eN++)
    {
      for (k=0;k<nQuadraturePoints_element;k++)
        {
          Vlin = 0.0;
          Alin = 0.0;
          for(I=0;I<nSpace;I++)
            {
              vlin[I] = df[eN*nQuadraturePoints_element*nSpace +
                           k*nSpace +
                           I];
              for(J=0;J<nSpace;J++)
                {
                  vlin[I]
                    -=
                    da[eN*nQuadraturePoints_element*nSpace2 +
                       k*nSpace2 +
                       I*nSpace +
                       J]
                    *
                    grad_phi[eN*nQuadraturePoints_element*nSpace +
                             k*nSpace +
                             J];
                }
            }
          v_dot_Gv=0.0;
          CI_nu2_G_ddot_G=0.0;
          g_dot_g=0.0;
          for (I=0;I<nSpace;I++)
            {
              Gv_I=0.0;
              g_I=0.0;
              for (J=0;J<nSpace;J++)
                {
                  G_IJ=0.0;
                  for (K=0;K<nSpace;K++)
                    G_IJ +=inverseJ[eN*nQuadraturePoints_element*nSpace2+
                                    k*nSpace2+
                                    K*nSpace+
                                    I]
                      *
                      inverseJ[eN*nQuadraturePoints_element*nSpace2+
                               k*nSpace2+
                               K*nSpace+
                               J];
                  Gv_I += G_IJ
                    *
                    vlin[J];
                  CI_nu2_G_ddot_G+=G_IJ*G_IJ;
                  g_I+=inverseJ[eN*nQuadraturePoints_element*nSpace2+
                                k*nSpace2+
                                J*nSpace+
                                I];
                }
              g_dot_g += g_I*g_I;
              v_dot_Gv += vlin[I]
                *
                Gv_I;
              
            }
          /*max diagonal entry for A. This choice needs to be looked into*/
          for(I=0;I<nSpace;I++)
            {
              A_II = a[eN*nQuadraturePoints_element*nSpace2 +
                       k*nSpace2 +
                       I*nSpace +
                       I];
              Alin = (A_II > Alin) ? A_II : Alin;
            }
          Alin*=dphi[eN*nQuadraturePoints_element +
                     k];
          CI_nu2_G_ddot_G*=CI*Alin*Alin;
	  /*need to check this*/
	  h_e = sqrt(g_dot_g);
          cfl1 = 0.5*sqrt(v_dot_Gv);
          cfl2 = 0.25*sqrt(CI_nu2_G_ddot_G);
	  Pe   = sqrt(v_dot_Gv)/sqrt(CI_nu2_G_ddot_G);
	  Da   = (dmt[eN*nQuadraturePoints_element + k] + dr[eN*nQuadraturePoints_element + k])/(cfl1 + 1.0e-12);


	  /*handle different cases here?*/
	  if (fabs(Da) < 1.0e-8)
	    {
	      tau_11 = 0.0;
	      tau_00 = 1.0/sqrt(v_dot_Gv +
				CI_nu2_G_ddot_G +
				+1.0e-8);
	    }
	  else if (Da > Pe*10.0)
	    {
	      tau_00 = 0.5/dmt[eN*nQuadraturePoints_element + k];
	      tau_11 = h_e*h_e/(12.0*dmt[eN*nQuadraturePoints_element + k]);
	    }
	  else if (cfl1 < 1.0e-8)
	    {
	      if (Pe*(2.0*Da + Pe) < 0.0)
		coshGamma = cos(sqrt(Pe*(-2.0*Da - Pe)));
	      else if (Pe*(2.0*Da + Pe) > 1000.0)
		coshGamma = cosh(100.0);
	      else 
		coshGamma = cosh(sqrt(Pe*(2.0*Da + Pe)));

	      tau_00 = -1.0/dmt[eN*nQuadraturePoints_element + k]*1.0/(-2.0 - Pe*Da/(coshGamma - 1.0 - 2.0*Pe*Da) + 1.0e-8);
	      tau_11 = -h_e*h_e/(dmt[eN*nQuadraturePoints_element + k]*6.0)*(-1.0 - 6.0/(2.0*Pe*Da) + 3.0*coshGamma -Pe*Da/(-2.0 + 2.0*coshGamma - Pe*Da));
	    }
	  else
	    {
	      t_00 = fmin(Pe/6.0,fmin(0.5/Da,0.5));
	      t_11 = fmin(Pe/60.0,fmin(-1.0/(12.0*Da),1.0/24.0));
	      t_11*= -1.0;
	      tau_00 = t_00/cfl1;
	      tau_11 = t_11/cfl1/(h_e*h_e);
	    }
	  assert(tau_11 <= 0.0);
	  assert(tau_00 >= 0.0);

	  tau[eN*nQuadraturePoints_element + k] = tau_00;
	  tau_gradient[eN*nQuadraturePoints_element + k] = tau_11;

          if (cfl1 > cfl2)
            cfl[eN*nQuadraturePoints_element +
                k] = cfl1;
          else
            cfl[eN*nQuadraturePoints_element +
                k] = cfl2;

          pe[eN*nQuadraturePoints_element +
             k] = sqrt(v_dot_Gv)/sqrt(CI_nu2_G_ddot_G);


        }
    }
}
/*note computes cfl as always advection cfl*/
void calculateSubgridError_ADR_Sangalli_tau_sd(proteus_index_t nElements_global,
					       int nQuadraturePoints_element,
					       int nSpace,
					       proteus_csr_index_t* rowptr,
					       proteus_csr_index_t* colind,
					       double* inverseJ,
					       double* dmt,
					       double* df,
					       double* a,
					       double* da,
					       double* grad_phi,
					       double* dphi,
					       double* dr,
					       double* pe,
					       double* cfl,
					       double* tau,
					       double* tau_gradient)
{
  proteus_index_t  nSpace2=nSpace*nSpace; proteus_csr_index_t m; proteus_csr_index_t nnz=rowptr[nSpace]; int eN, k, I, J, K;
  double Vlin,Alin,A_II,cfl1,cfl2,vlin[3],G_IJ,v_dot_Gv,CI_nu2_G_ddot_G,Gv_I,CI=36.0*36.0,g_I,g_dot_g;
  double Da,h_e,tau_00,tau_11,t_00,t_11,coshGamma,Pe,sinhAlpha,coshAlpha,PeEval,gamma2,gam2Eval,DaEval;
  for(eN=0;eN<nElements_global;eN++)
    {
      for (k=0;k<nQuadraturePoints_element;k++)
        {
          Vlin = 0.0;
          Alin = 0.0;
          for(I=0;I<nSpace;I++)
            {
              vlin[I] = df[eN*nQuadraturePoints_element*nSpace +
                           k*nSpace +
                           I];
              for(m=rowptr[I];m<rowptr[I+1];m++)
                {
                  vlin[I]
                    -=
                    da[eN*nQuadraturePoints_element*nnz+
                       k*nnz+
		       m]
                    *
                    grad_phi[eN*nQuadraturePoints_element*nSpace +
                             k*nSpace +
                             colind[m]];
                }
            }
          v_dot_Gv=0.0;
          CI_nu2_G_ddot_G=0.0;
          g_dot_g=0.0;
          for (I=0;I<nSpace;I++)
            {
              Gv_I=0.0;
              g_I=0.0;
              for (J=0;J<nSpace;J++)
                {
                  G_IJ=0.0;
                  for (K=0;K<nSpace;K++)
                    G_IJ +=inverseJ[eN*nQuadraturePoints_element*nSpace2+
                                    k*nSpace2+
                                    K*nSpace+
                                    I]
                      *
                      inverseJ[eN*nQuadraturePoints_element*nSpace2+
                               k*nSpace2+
                               K*nSpace+
                               J];
                  Gv_I += G_IJ
                    *
                    vlin[J];
                  CI_nu2_G_ddot_G+=G_IJ*G_IJ;
                  g_I+=inverseJ[eN*nQuadraturePoints_element*nSpace2+
                                k*nSpace2+
                                J*nSpace+
                                I];
                }
              g_dot_g += g_I*g_I;
              v_dot_Gv += vlin[I]
                *
                Gv_I;
              
            }
          /*max diagonal entry for A. This choice needs to be looked into*/
          for(I=0;I<nSpace;I++)
            {
	      for(m=rowptr[I];m<rowptr[I+1];m++)
		{
		  if (I==colind[m])
		    {
		      A_II = a[eN*nQuadraturePoints_element*nnz +
			       k*nnz + m];
		      Alin = (A_II > Alin) ? A_II : Alin;
		    }
		}
            }
          Alin*=dphi[eN*nQuadraturePoints_element +
                     k];
          CI_nu2_G_ddot_G*=CI*Alin*Alin;
	  /*need to check this*/
	  h_e = 1.0/sqrt(g_dot_g);
          cfl1 = sqrt(v_dot_Gv);/*mwf not sure about 0.5 here 0.5*sqrt(v_dot_Gv);*/
          cfl2 = 0.25*sqrt(CI_nu2_G_ddot_G);
	  Pe   = sqrt(v_dot_Gv)/(sqrt(CI_nu2_G_ddot_G)+1.0e-12);
	  Da   = (dmt[eN*nQuadraturePoints_element + k] + dr[eN*nQuadraturePoints_element + k])/(cfl1 + 1.0e-12);


	  /*handle different cases here?*/
	  if (fabs(Da) < 1.0e-8)
	    {
	      tau_11 = 0.0;
	      tau_00 = 1.0/sqrt(v_dot_Gv +
				CI_nu2_G_ddot_G +
				+1.0e-8);
	      /*mwf debug
		printf("Sangalli Da ~ 0 case Da= %12.5e Pe=%12.5e cfl=%12.5e tau_11=%12.5e tau_00=%12.5e \n",Da,Pe,cfl1,tau_11,tau_00);
	      */
	    }
	  else if (Da > Pe*10.0)
	    {
	      tau_00 = 0.5/(dmt[eN*nQuadraturePoints_element + k]+dr[eN*nQuadraturePoints_element + k]);
	      tau_11 = -h_e*h_e/(12.0*(dmt[eN*nQuadraturePoints_element + k] + dr[eN*nQuadraturePoints_element + k]));
	      /*mwf debug
	      printf("Sangalli Da >> 1 case h_e= %12.5e dmt=%12.5e dr=%12.5e Da= %12.5e Pe=%12.5e cfl=%12.5e tau_11=%12.5e tau_00=%12.5e \n",h_e,
		     dmt[eN*nQuadraturePoints_element + k],dr[eN*nQuadraturePoints_element + k],
		     Da,Pe,cfl1,tau_11,tau_00);
	      */
	    }
	  else if (cfl1 < 1.0e-8)
	    {
	      if (Pe*(2.0*Da + Pe) < 0.0)
		coshGamma = cos(sqrt(Pe*(-2.0*Da - Pe)));
	      else if (Pe*(2.0*Da + Pe) > 1000.0)
		coshGamma = cosh(100.0);
	      else 
		coshGamma = cosh(sqrt(Pe*(2.0*Da + Pe)));

	      tau_00 = -1.0/dmt[eN*nQuadraturePoints_element + k]*1.0/(-2.0 - Pe*Da/(coshGamma - 1.0 - 2.0*Pe*Da) + 1.0e-8);
	      tau_11 = -h_e*h_e/(dmt[eN*nQuadraturePoints_element + k]*6.0)*(-1.0 - 6.0/(2.0*Pe*Da) + 3.0*coshGamma -Pe*Da/(-2.0 + 2.0*coshGamma - Pe*Da));
	      /*mwf debug
		printf("Sangalli cfl ~ 0 case Da= %12.5e Pe=%12.5e cfl=%12.5e tau_11=%12.5e tau_00=%12.5e \n",Da,Pe,cfl1,tau_11,tau_00);
	      */
	    }
	  else if (Da >= 0.0 && 0)
	    {
	      t_00 = fmin(Pe/6.0,fmin(0.5/Da,0.5));
	      t_11 = fmin(Pe/60.0,fmin(1.0/(12.0*Da),1.0/24.0));
	      t_11*= -1.0;
	      tau_00 = t_00/cfl1;
	      tau_11 = h_e*h_e*t_11/cfl1;
	      /*mwf debug */
	      printf("Sangalli general reduced formula Da= %12.5e Pe=%12.5e cfl=%12.5e t_11=%12.5e tau_11=%12.5e t_00=%12.5e tau_00=%12.5e \n",Da,Pe,cfl1,t_11,tau_11,t_00,tau_00);
	      
	    }
	  else
	    {
	      PeEval = fmin(Pe,10.0);
	      DaEval = -Da;
	      if (DaEval < -10.0)
		DaEval = -10.0;
	      if (DaEval > 10.0)
		DaEval = 10.0;
	      gamma2 = PeEval*(-2.0*DaEval + PeEval);
	      
	      if (gamma2 < 0.0)
		coshGamma = cos(sqrt(PeEval*(2.0*DaEval - PeEval)));
	      else
		coshGamma = cosh(sqrt(gamma2));
	      sinhAlpha = sinh(PeEval);
	      coshAlpha = cosh(PeEval);

	      t_00 = 1.0/(-2.0*DaEval + DaEval*DaEval*sinhAlpha/(-coshAlpha + coshGamma + DaEval*sinhAlpha));
	      t_11 = (-3.0 -DaEval*DaEval + 3.0*DaEval/(PeEval+1.0e-12) + DaEval*(3.0*DaEval * coshGamma + sinhAlpha*(-3.0 + DaEval*DaEval))/
		      (-2.0*coshAlpha + 2.0*coshGamma + DaEval*sinhAlpha))/(6.0*DaEval*DaEval*DaEval + 1.0e-12);
	      /*go ahead and switch sign to match Sangalli convention?*/
/* 	      if (Da > 0.0) */
/* 		DaEval = fmin(Da,10.0); */
/* 	      if (Da < 0.0) */
/* 		DaEval = fmax(Da,-10.0); */
/* 	      gamma2 = PeEval*(2.0*DaEval + PeEval); */
/* 	      gam2Eval= gamma2; */
/* 	      if (gamma2 < 0.0) */
/* 		coshGamma = cos(sqrt(PeEval*(-2.0*DaEval - PeEval))); */
/* 	      else */
/* 		coshGamma = cosh(sqrt(gam2Eval)); */
/* 	      sinhAlpha = sinh(PeEval); */
/* 	      coshAlpha = cosh(PeEval); */
		
/* 	      t_00 = 1.0/(2.0*DaEval + DaEval*DaEval*sinhAlpha/(-coshAlpha + coshGamma - DaEval*sinhAlpha + 1.0e-12) + 1.0e-12); */
/* 	      t_11 = -1.0/(6.0*DaEval*DaEval*DaEval+1.0e-12)* */
/* 		(-3.0 - DaEval*DaEval -3.0*DaEval/(PeEval+1.0e-12)  */
/* 		 -DaEval*(-3.0*DaEval*coshGamma +sinhAlpha*(-3.0+DaEval*DaEval))/(-2.0*coshAlpha + 2.0*coshGamma -DaEval*sinhAlpha)); */

/* 	      t_11 = (DaEval*(sinhAlpha*(DaEval*DaEval - 3.) - 3.*DaEval*coshGamma)/(2.*coshGamma - sinhAlpha*DaEval - 2*coshAlpha) + DaEval*DaEval + 3.*DaEval/PeEval + 3.)/(6.*DaEval*DaEval*DaEval); */




	      tau_00 = t_00/cfl1;
	      tau_11 = h_e*h_e*t_11/cfl1;
	      /*mwf debug*/
	      printf("\t Sangalli general h_e= %12.5e Da= %12.5e DaEval= %12.5e Pe=%12.5e PeEval= %12.5e cfl=%12.5e \n\t gamma2= %12.5e  coshGamma=%12.5e sinhAlpha=%12.5e coshAlpha=%12.5e \n\t t_11=%12.5e tau_11=%12.5e t_00= %12.5e tau_00=%12.5e \n",h_e,Da,DaEval,Pe,PeEval,cfl1,gamma2,coshGamma,sinhAlpha,coshAlpha,t_11,tau_11,t_00,tau_00);
	      
	    }
	  
	  /*mwf try to catch eval errors*/
	  if (tau_11 > 0.0)
	    {
	      /*mwf debug*/
	      printf("Sangalli tau_11 = %12.5e > 0 switching ",tau_11);
	      t_11 = fmin(Pe/60.0,fmin(1.0/(12.0*Da),1.0/24.0));
	      t_11*= -1.0;
	      tau_11 = h_e*h_e*t_11/cfl1;
	      /*mwf debug*/
	      printf("tau_11 = %12.5e \n",tau_11);
	      

	    }
	  assert(tau_11 <= 0.0);
	  assert(tau_00 >= 0.0);

	  tau[eN*nQuadraturePoints_element + k] = tau_00;
	  tau_gradient[eN*nQuadraturePoints_element + k] = tau_11;

          if (cfl1 > cfl2)
            cfl[eN*nQuadraturePoints_element +
                k] = cfl1;
          else
            cfl[eN*nQuadraturePoints_element +
                k] = cfl2;

          pe[eN*nQuadraturePoints_element +
             k] = sqrt(v_dot_Gv)/sqrt(CI_nu2_G_ddot_G);

        }
    }
}

/** @}*/

