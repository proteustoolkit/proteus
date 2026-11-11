#include "transportCoefficients.h"

/**
   \file transportCoefficients.c
   \ingroup transportCoefficients
   @{
*/
/*define relaxation function according to Jacobsen et al 2012, INJNMF*/
double relaxationFunction(double phi, double phiStart, double phiEnd)
{
  double H;
  double x;
  double Length;
    
    if(phiStart < phiEnd)
      { 
	Length = phiEnd - phiStart;
	x = (phi - phiStart)/Length;
	H = 1 - (exp(pow(x,3.5)) - 1.)/ (exp(1) - 1.);
      }
    else
      { 
	Length = -(phiEnd - phiStart);
	x = 1 - (phi - phiStart)/Length;
	H = 1 - (exp(pow(x,3.5)) - 1.)/ (exp(1) - 1.);
      }      
    return H;
	
	  
  
}
/*#define SCALAR_DIFFUSION*/
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

double linearHeaviside(double eps, double phi)
{
  double H;
  if (phi > eps)
    H=1.0;
  else if (phi < -eps)
    H=0.0;
  else
    H = 0.5*((phi+eps)/eps);
  return H;
}

double linearDirac(double eps, double phi)
{
  double d;
  if (phi > eps)
    d=0.0;
  else if (phi < -eps)
    d=0.0;
  else
    d = 0.5/eps;
  return d;
}

void linearADR_ConstantCoefficientsEvaluate(const int nPoints,
                                            const int nSpace,
                                            const double M,
                                            const double *A,
                                            const double *B,
                                            const double C,
                                            const double t,
                                            const double *x,
                                            const double *u,
                                            double *m,
                                            double *dm,
                                            double *f,
                                            double *df,
                                            double *a,
                                            double *r,
                                            double *dr)
{
  proteus_index_t J; int k, I;
  const int nSpace2=nSpace*nSpace;
  for (k=0;k<nPoints;k++)
    {
      m[k]=M*u[k];
      dm[k]=M;
      
      for (I=0;I<nSpace;I++)
        {
          f[k*nSpace+I]=B[I]*u[k];
          df[k*nSpace+I]=B[I];
          for (J=0;J<nSpace;J++)
            {
              a[k*nSpace2+I*nSpace+J]=A[I*nSpace+J];
            }
        }

      r[k]=C*u[k];
      dr[k]=C;
    }
}

void groundwaterTransportCoefficientsEvaluate(const int nPoints,
                                              const int nSpace,
                                              const double omega,
                                              const double d,
                                              const double alpha_L,
                                              const double alpha_T,
                                              const double *v,
                                              const double *u,
                                              double *m,
                                              double *dm,
                                              double *f,
                                              double *df,
                                              double *a)
{
  proteus_index_t J; int k, I;
  const int nSpace2=nSpace*nSpace;
  double norm_v;

  for (k=0;k<nPoints;k++)
    {
      m[k]=omega*u[k];
      dm[k]=omega;
      norm_v = 0.0;
      for (I=0;I<nSpace;I++)
        {
          f[k*nSpace+I]=v[k*nSpace+I]*u[k];
          df[k*nSpace+I]=v[k*nSpace+I];
          norm_v += v[k*nSpace+I]*v[k*nSpace+I];
        }
      norm_v = sqrt(norm_v);
      if (norm_v > 0.0)
        {
          for (I=0;I<nSpace;I++)
            {
              a[k*nSpace2+I*nSpace+I]=omega*d + alpha_T*norm_v + (alpha_L - alpha_T)*v[k*nSpace+I]*v[k*nSpace+I]/norm_v;
              for (J=I+1;J<nSpace;J++)
                {
                  a[k*nSpace2+I*nSpace+J]=(alpha_L - alpha_T)*v[k*nSpace+I]*v[k*nSpace+J]/norm_v;
                  a[k*nSpace2+J*nSpace+I]=a[k*nSpace2+I*nSpace+J];
                }
            }
        }
      else
        for (I=0;I<nSpace;I++)
          a[k*nSpace2+I*nSpace+I]=omega*d;
    }
}
void groundwaterBiodegradation01EvaluateFC(const int nPoints,
					   const int nSpace,
					   const double omega,
					   const double d_c,
					   const double d_e,
					   const double alpha_L,
					   const double alpha_T,
					   const double Kox_max,
					   const double Kox_C,
					   const double Kox_E,
					   const double Kox_X,
					   const double Yield,
					   const double k_d,
					   const double *v,
					   const double *c_c,
					   const double *c_e,
					   const double *c_x,
					   double *m_c,
					   double *dm_c,
					   double *m_e,
					   double *dm_e,
					   double *m_x,
					   double *dm_x,
					   double *f_c,
					   double *df_c,
					   double *f_e,
					   double *df_e,
					   double *a_c,
					   double *a_e,
					   double *r_c,
					   double *dr_c_dc,
					   double *dr_c_de,
					   double *dr_c_dx,
					   double *r_e,
					   double *dr_e_dc,
					   double *dr_e_de,
					   double *dr_e_dx,
					   double *r_x,
					   double *dr_x_dc,
					   double *dr_x_de,
					   double *dr_x_dx)
{
  proteus_index_t J; int k, I;
  const int nSpace2=nSpace*nSpace;
  double norm_v;
  double C,E,X,denomC,denomE,denomX,rox,drox_dC,drox_dE,drox_dX;
  for (k=0;k<nPoints;k++)
    {
      C = c_c[k]; E = c_e[k]; X = c_x[k];
      m_c[k]=omega*C;
      dm_c[k]=omega;
      m_e[k]=omega*E;
      dm_e[k]=omega;
      m_x[k]=omega*X;
      dm_x[k]=omega;

      norm_v = 0.0;
      for (I=0;I<nSpace;I++)
        {
          f_c[k*nSpace+I]=v[k*nSpace+I]*C;
          df_c[k*nSpace+I]=v[k*nSpace+I];
          f_e[k*nSpace+I]=v[k*nSpace+I]*E;
          df_e[k*nSpace+I]=v[k*nSpace+I];

          norm_v += v[k*nSpace+I]*v[k*nSpace+I];
        }
      norm_v = sqrt(norm_v);
      if (norm_v > 0.0)
        {
          for (I=0;I<nSpace;I++)
            {
              a_c[k*nSpace2+I*nSpace+I]=omega*d_c + alpha_T*norm_v + (alpha_L - alpha_T)*v[k*nSpace+I]*v[k*nSpace+I]/norm_v;
              for (J=I+1;J<nSpace;J++)
                {
                  a_c[k*nSpace2+I*nSpace+J]= (alpha_L - alpha_T)*v[k*nSpace+I]*v[k*nSpace+J]/norm_v;
                  a_c[k*nSpace2+J*nSpace+I]=a_c[k*nSpace2+I*nSpace+J];
                }
              a_e[k*nSpace2+I*nSpace+I]=omega*d_e + alpha_T*norm_v + (alpha_L - alpha_T)*v[k*nSpace+I]*v[k*nSpace+I]/norm_v;
              for (J=I+1;J<nSpace;J++)
                {
                  a_e[k*nSpace2+I*nSpace+J]= (alpha_L - alpha_T)*v[k*nSpace+I]*v[k*nSpace+J]/norm_v;
                  a_e[k*nSpace2+J*nSpace+I]=a_e[k*nSpace2+I*nSpace+J];
                }
            }
        }
      else
        {
	  for (I=0;I<nSpace;I++)
	    {
	      a_c[k*nSpace2+I*nSpace+I]=omega*d_c;
	      a_e[k*nSpace2+I*nSpace+I]=omega*d_e;
	    }
	}/*dispersion calc*/
      /*reactions*/
      denomC = C + Kox_C; denomE = E + Kox_E; denomX = X + Kox_X;
      rox = Kox_max*X*(C/denomC)*(E/denomE)*(Kox_X/denomX);
      drox_dC = Kox_max*X*(E/denomE)*(Kox_X/denomX)*(1.0/denomC -C/(denomC*denomC));
      drox_dE = Kox_max*X*(C/denomC)*(Kox_X/denomX)*(1.0/denomE -E/(denomE*denomE));
      drox_dX = Kox_max*(C/denomC)*(E/denomE)*(Kox_X/denomX -Kox_X*X/(denomX*denomX));

      r_c[k] = omega*rox;
      r_e[k] = 3.0*omega*rox;
      r_x[k] = -Yield*omega*rox + X*omega*k_d;

      dr_c_dc[k] = omega*drox_dC;     dr_c_de[k] =  omega*drox_dE;     dr_c_dx[k] =  omega*drox_dX;

      dr_e_dc[k] = 3.0*omega*drox_dC; dr_e_de[k] =  3.0*omega*drox_dE; dr_e_dx[k] =  3.0*omega*drox_dX;

      dr_x_dc[k] = -Yield*omega*drox_dC;
      dr_x_de[k] = -Yield*omega*drox_dE;
      dr_x_dx[k] = -Yield*omega*drox_dX + omega*k_d;
/*      /\*mwf debug*\/ */
/*       printf("bio01eval k=%d C=%g E=%g X=%g rox=%g \n",k,C,E,X,rox); */
     }
}

void groundwaterBryantDawsonIonExEvaluateFC(const int nPoints,
					    const int nSpace,
					    const double omega,
					    const double d_m,
					    const double d_h,
					    const double alpha_L,
					    const double alpha_T,
					    const double K_m,
					    const double K_h,
					    const double K_w,
					    const double Z_tot,
					    const double *v,
					    const double *c_m,
					    const double *c_h,
					    double *m_m,
					    double *dm_m_m,
					    double *dm_m_h,
					    double *m_h,
					    double *dm_h_m,
					    double *dm_h_h,
					    double *f_m,
					    double *df_m,
					    double *f_h,
					    double *df_h,
					    double *a_m,
					    double *a_h,
					    double *phi_h,
					    double *dphi_h,
					    double *r_m,
					    double *dr_m_dm,
					    double *dr_m_dh,
					    double *r_h,
					    double *dr_h_dm,
					    double *dr_h_dh)

{
  proteus_index_t J; int k, I;
  const int nSpace2=nSpace*nSpace;
  double norm_v;
  double C_m,C_h,C_oh,C_a,Z_m,Z_h,dC_a_dC_h,denomZ;
  const double eps = 1.0e-12;
  for (k=0;k<nPoints;k++)
    {
      /*metal, proton, hydroxyl*/
      C_m = c_m[k];  C_h= c_h[k]; C_oh = K_w/(C_h+eps); 
      /*acidity*/
      C_a = C_h - C_oh; dC_a_dC_h = 1.0 + K_w/(C_h*C_h+eps);
      /*sorbed concentrations*/
      denomZ  = 1.0 + K_m*C_m + K_h*C_h;
      Z_m    = K_m*C_m*Z_tot/denomZ;
      Z_h    = K_h*C_h*Z_tot/denomZ;
      
      m_m[k]   =omega*(C_m + Z_m);
      dm_m_m[k]=omega*(1.0 + K_m*Z_tot/denomZ - K_m*C_m*Z_tot/(denomZ*denomZ)*K_m*C_m);
      dm_m_h[k]=omega*(                       - K_m*C_m*Z_tot/(denomZ*denomZ)*K_h*C_h);
      m_h[k]   =omega*(C_a + Z_h);
      dm_h_m[k]=omega*(                       - K_h*C_h*Z_tot/(denomZ*denomZ)*K_m*C_m);
      dm_h_h[k]=omega*(dC_a_dC_h + K_h*Z_tot/denomZ - K_h*C_h*Z_tot/(denomZ*denomZ)*K_h*C_h);

      /*nonlinear potential for C_h*/
      phi_h[k]  = C_a;
      dphi_h[k] = dC_a_dC_h;
      norm_v = 0.0;
      for (I=0;I<nSpace;I++)
        {
          f_m[k*nSpace+I]=v[k*nSpace+I]*C_m;
          df_m[k*nSpace+I]=v[k*nSpace+I];
          f_h[k*nSpace+I]=v[k*nSpace+I]*C_a;
          df_h[k*nSpace+I]=v[k*nSpace+I]*dC_a_dC_h;

          norm_v += v[k*nSpace+I]*v[k*nSpace+I];
        }
      norm_v = sqrt(norm_v);
      if (norm_v > 0.0)
        {
          for (I=0;I<nSpace;I++)
            {
              a_m[k*nSpace2+I*nSpace+I]=omega*d_m + alpha_T*norm_v + (alpha_L - alpha_T)*v[k*nSpace+I]*v[k*nSpace+I]/norm_v;
              for (J=I+1;J<nSpace;J++)
                {
                  a_m[k*nSpace2+I*nSpace+J]= (alpha_L - alpha_T)*v[k*nSpace+I]*v[k*nSpace+J]/norm_v;
                  a_m[k*nSpace2+J*nSpace+I]=a_m[k*nSpace2+I*nSpace+J];
                }
              a_h[k*nSpace2+I*nSpace+I]=omega*d_h + alpha_T*norm_v + (alpha_L - alpha_T)*v[k*nSpace+I]*v[k*nSpace+I]/norm_v;
              for (J=I+1;J<nSpace;J++)
                {
                  a_h[k*nSpace2+I*nSpace+J]= (alpha_L - alpha_T)*v[k*nSpace+I]*v[k*nSpace+J]/norm_v;
                  a_h[k*nSpace2+J*nSpace+I]=a_h[k*nSpace2+I*nSpace+J];
                }
            }
        }
      else
        {
	  for (I=0;I<nSpace;I++)
	    {
	      a_m[k*nSpace2+I*nSpace+I]=omega*d_m;
	      a_h[k*nSpace2+I*nSpace+I]=omega*d_h;
	    }
	}/*dispersion calc*/
      r_m[k] = 0.0;
      r_h[k] = 0.0;
      dr_m_dm[k] = 0.0;
      dr_m_dh[k] = 0.0;
      dr_h_dm[k] = 0.0;
      dr_h_dh[k] = 0.0;

/*      /\*mwf debug*\/ */
/*       printf("ionExeval k=%d C_m=%g C_h=%g  \n",k,C_m,C_h); */
     }
}
void groundwaterTransportCoefficientsEvaluate_hetMat(const int nSimplex,
						     const int nPointsPerSimplex,
						     const int nSpace,
						     const double d,
						     const int* materialTypes,
						     const double *omega_types,
						     const double *alpha_L_types,
						     const double *alpha_T_types,
						     const double *v,
						     const double *u,
						     double *m,
						     double *dm,
						     double *f,
						     double *df,
						     double *a)
{
  proteus_index_t J; int i, j, k, I, matID;
  const int nSpace2=nSpace*nSpace;
  double norm_v;

  for (i=0;i<nSimplex;i++)
    {
      matID = materialTypes[i];
      for (j=0; j < nPointsPerSimplex; j++)
	{
	  k = i*nPointsPerSimplex+j;

	  m[k]=omega_types[matID]*u[k];
	  dm[k]=omega_types[matID];
	  norm_v = 0.0;
	  for (I=0;I<nSpace;I++)
	    {
	      f[k*nSpace+I]=v[k*nSpace+I]*u[k];
	      df[k*nSpace+I]=v[k*nSpace+I];
	      norm_v += v[k*nSpace+I]*v[k*nSpace+I];
	    }
	  norm_v = sqrt(norm_v);
	  if (norm_v > 0.0)
	    {
	      for (I=0;I<nSpace;I++)
		{
		  a[k*nSpace2+I*nSpace+I]=omega_types[matID]*d + alpha_T_types[matID]*norm_v + (alpha_L_types[matID] - alpha_T_types[matID])*v[k*nSpace+I]*v[k*nSpace+I]/norm_v;
		  for (J=I+1;J<nSpace;J++)
		    {
		      a[k*nSpace2+I*nSpace+J]=(alpha_L_types[matID] - alpha_T_types[matID])*v[k*nSpace+I]*v[k*nSpace+J]/norm_v;
		      a[k*nSpace2+J*nSpace+I]=a[k*nSpace2+I*nSpace+J];
		    }
		}
	    }
	  else
	    for (I=0;I<nSpace;I++)
	      a[k*nSpace2+I*nSpace+I]=omega_types[matID]*d;
	}
    }
}
void variablySaturatedGroundwaterTransportCoefficientsEvaluate_hetMat(const int nSimplex,
								      const int nPointsPerSimplex,
								      const int nSpace,
								      const double d,
								      const int* materialTypes,
								      const double *theta, /*phase volume fraction*/
								      const double *alpha_L_types,
								      const double *alpha_T_types,
								      const double *v,/*phase darcy velocity*/
								      const double *u,
								      double *m,
								      double *dm,
								      double *f,
								      double *df,
								      double *a)
{
  proteus_index_t J; int i, j, k, I, matID;
  const int nSpace2=nSpace*nSpace;
  double norm_v;

  for (i=0;i<nSimplex;i++)
    {
      matID = materialTypes[i];
      for (j=0; j < nPointsPerSimplex; j++)
	{
	  k = i*nPointsPerSimplex+j;

	  m[k]=theta[k]*u[k];
	  dm[k]=theta[k];
	  norm_v = 0.0;
	  for (I=0;I<nSpace;I++)
	    {
	      f[k*nSpace+I]=v[k*nSpace+I]*u[k];
	      df[k*nSpace+I]=v[k*nSpace+I];
	      norm_v += v[k*nSpace+I]*v[k*nSpace+I];
	    }
	  norm_v = sqrt(norm_v);
	  if (norm_v > 0.0)
	    {
	      for (I=0;I<nSpace;I++)
		{
		  a[k*nSpace2+I*nSpace+I]=theta[k]*d + alpha_T_types[matID]*norm_v + (alpha_L_types[matID] - alpha_T_types[matID])*v[k*nSpace+I]*v[k*nSpace+I]/norm_v;
		  for (J=I+1;J<nSpace;J++)
		    {
		      a[k*nSpace2+I*nSpace+J]=(alpha_L_types[matID] - alpha_T_types[matID])*v[k*nSpace+I]*v[k*nSpace+J]/norm_v;
		      a[k*nSpace2+J*nSpace+I]=a[k*nSpace2+I*nSpace+J];
		    }
		}
	    }
	  else
	    for (I=0;I<nSpace;I++)
	      a[k*nSpace2+I*nSpace+I]=theta[k]*d;
	}
    }
}
void variablySaturatedGroundwaterEnergyTransportCoefficientsEvaluate_hetMat(const int nSimplex,
									    const int nPointsPerSimplex,
									    const int nSpace,
									    const double rho_w,
									    const double rho_n,
									    const double specificHeat_w,
									    const double specificHeat_n,
									    const int* materialTypes,
									    const double *theta, /*phase volume fraction*/
									    const double *thetaS_types,
									    const double *alpha_L_types,
									    const double *alpha_T_types,
									    const double *rho_s_types,
									    const double *specificHeat_s_types,
									    const double *lambda_sat_types,
									    const double *lambda_dry_types,
									    const double *lambda_aniso_types,
									    const double *v,/*phase darcy velocity*/
									    const double *u,
									    double *m,
									    double *dm,
									    double *f,
									    double *df,
									    double *a)
{
  proteus_index_t J; int i, j, k, I, matID;
  const int nSpace2=nSpace*nSpace;
  double norm_v,tmp,sw,Ke,lambda;

  for (i=0;i<nSimplex;i++)
    {
      matID = materialTypes[i];
      for (j=0; j < nPointsPerSimplex; j++)
	{
	  k = i*nPointsPerSimplex+j;
	  tmp = theta[k]*rho_w*specificHeat_w + (thetaS_types[matID]-theta[k])*rho_n*specificHeat_n + 
	    (1.0-thetaS_types[matID])*rho_s_types[matID]*specificHeat_s_types[matID];
	  m[k]=tmp*u[k];
	  dm[k]=tmp;
	  norm_v = 0.0;
	  for (I=0;I<nSpace;I++)
	    {
	      f[k*nSpace+I]=v[k*nSpace+I]*rho_w*specificHeat_w*u[k];
	      df[k*nSpace+I]=v[k*nSpace+I]*rho_w*specificHeat_w;
	      norm_v += v[k*nSpace+I]*v[k*nSpace+I];
	    }
	  norm_v = sqrt(norm_v);
	  sw = theta[k]/thetaS_types[matID];
	  if (norm_v > 0.0)
	    {
	      for (I=0;I<nSpace;I++)
		{
		  a[k*nSpace2+I*nSpace+I]= rho_w*specificHeat_w*alpha_T_types[matID]*norm_v + rho_w*specificHeat_w*(alpha_L_types[matID] - alpha_T_types[matID])*v[k*nSpace+I]*v[k*nSpace+I]/norm_v;
		  for (J=I+1;J<nSpace;J++)
		    {
		      a[k*nSpace2+I*nSpace+J]=rho_w*specificHeat_w*(alpha_L_types[matID] - alpha_T_types[matID])*v[k*nSpace+I]*v[k*nSpace+J]/norm_v;
		      a[k*nSpace2+J*nSpace+I]=a[k*nSpace2+I*nSpace+J];
		    }
		}
	    }

	  /*todo check with Stacy for right form*/
	  if (sw > 0.05)
	    Ke = 0.7*log(sw) + 1.0;
	  lambda = (lambda_sat_types[matID]-lambda_dry_types[matID])*Ke + lambda_dry_types[matID];
	  for (I=0;I<nSpace;I++)
	    {
	      a[k*nSpace2+I*nSpace+I] += lambda*lambda_aniso_types[matID*nSpace+I];
	    }

	}
    }
}

void nonlinearADR_pqrstEvaluate(const int nPoints,
                                const int nSpace,
                                const double M,
                                const double* A,
                                const double* B,
                                const double C,
                                const double p_pow,
                                const double q_pow,
                                const double r_pow,
                                const double s_pow,
                                const double t_pow,
                                const double t,
                                const double *x,
                                const double *u,
                                double *m,
                                double *dm,
                                double *f,
                                double *df,
                                double *a,
                                double *da,
                                double *phi,
                                double *dphi,
                                double *r,
                                double *dr)
{
  proteus_index_t J; int k, I;
  const int nSpace2=nSpace*nSpace;
  double uPlus,
    tmp_f, 
    tmp_df,
    tmp_a,
    tmp_da;
  const double pM1_pow=p_pow-1.0,
    qM1_pow=q_pow-1.0,
    rM1_pow=r_pow-1.0,
    sM1_pow=s_pow-1.0,
    tM1_pow=t_pow-1.0;
  for (k=0;k<nPoints;k++)
    {
      uPlus = u[k] > 0.0 ? u[k] : 0.0;
      m[k]  = p_pow > 1.0 ?       M*pow(uPlus,p_pow)   : M*u[k];
      dm[k] = p_pow > 1.0 ? p_pow*M*pow(uPlus,pM1_pow) : M;

      tmp_f  = q_pow > 1.0 ?       pow(uPlus,q_pow)   : u[k];
      tmp_df = q_pow > 1.0 ? q_pow*pow(uPlus,qM1_pow) : 1.0;

      tmp_a  = t_pow > 0.0 ?       pow(uPlus,t_pow)   : 1.0;
      tmp_da = t_pow > 0.0 ? t_pow*pow(uPlus,tM1_pow) : 0.0;

      for (I=0;I<nSpace;I++)
        {
          f[k*nSpace+I]  = B[I]*tmp_f;
          df[k*nSpace+I] = B[I]*tmp_df;
          for (J=0;J<nSpace;J++)
            {
              a[k*nSpace2+I*nSpace+J]  = A[I*nSpace + J]*tmp_a;
              da[k*nSpace2+I*nSpace+J] = A[I*nSpace + J]*tmp_da;
            }
        }
      phi[k]  = r_pow > 1.0 ?       pow(uPlus,r_pow)   : u[k];
      dphi[k] = r_pow > 1.0 ? r_pow*pow(uPlus,rM1_pow) : 1.0;
      
      r[k]  =  s_pow > 1.0 ?       C*pow(uPlus,s_pow)   : C*u[k];
      dr[k] =  s_pow > 1.0 ? s_pow*C*pow(uPlus,sM1_pow) : C;
    }
}

void nonlinearADR_pqrstDualEvaluate(const int nPoints,
                                    const int nSpace,
                                    const double M,
                                    const double* A,
                                    const double* B,
                                    const double C,
                                    const double p1,
                                    const double q1,
                                    const double r1,
                                    const double s1,
                                    const double t1,
                                    const double p2,
                                    const double q2,
                                    const double r2,
                                    const double s2,
                                    const double t2,
                                    const double t,
                                    const double *x,
                                    const double *u,
                                    double *m,
                                    double *dm,
                                    double *f,
                                    double *df,
                                    double *a,
                                    double *da,
                                    double *phi,
                                    double *dphi,
                                    double *r,
                                    double *dr)
{
  proteus_index_t J; int k, I;
  const int nSpace2=nSpace*nSpace;
  double max_1mu_0,atmp,datmp;
  const double p2M1=p2-1.0,
    q2M1=q2-1.0,
    r2M1=r2-1.0,
    s2M1=s2-1.0,
    t2M1=t2-1.0;

  nonlinearADR_pqrstEvaluate(nPoints,
                             nSpace,
                             M,
                             A,
                             B,
                             C,
                             p1,q1,r1,s1,t1,
                             t,
                             x,
                             u,
                             m,dm,
                             f,df,
                             a,da,
                             phi,dphi,
                             r,
                             dr);

  for (k=0; k < nPoints; k++)
    {
      max_1mu_0 = 1.0-u[k] > 0.0 ? 1.0-u[k] : 0.0;

      if (p2 > 1.0)
        {
          m[k] *= pow(max_1mu_0,p2);
          dm[k] *= pow(max_1mu_0,p2M1)*p2;
        }

      if (q2 > 1.0)
        {
          for (I=0; I < nSpace; I++)
            {
              f[k*nSpace+I] *= pow(max_1mu_0,q2);
              df[k*nSpace+I] *= pow(max_1mu_0,q2M1)*q2;
            }
        }

      if (t2 > 0.0)
        {
          atmp = pow(max_1mu_0,t2);
          datmp = pow(max_1mu_0,t2M1);
          for (I=0; I < nSpace; I++)
            {
              for (J=0; J < nSpace; J++)
                {
                  a[k*nSpace2+I*nSpace+J] *= atmp;
                  da[k*nSpace2+I*nSpace+J] *= datmp*t2;
                }
            }
        }

      if (r2 > 1.0)
        {
          phi[k] *= pow(max_1mu_0,r2);
          dphi[k] *= pow(max_1mu_0,r2M1)*r2;
        }

      if (s2 > 1.0)
        {
          r[k] *= pow(max_1mu_0,s2*r[k]);
          dr[k] *= pow(max_1mu_0,s2M1)*s2;
        }
    }
}

void unitSquareRotationEvaluate(const int nPoints,
                                const int nSpace,
                                const double *x,
                                const double *u,
                                double *m,
                                double *dm,
                                double *f,
                                double *df)
{
  double vx, vy;
  int k;
  for (k=0; k < nPoints; k++)
    {
      m[k] = u[k];
      dm[k] = 1.0;
      vx = 2.0*M_PI*(x[k*3+1] - 0.5);
      vy = 2.0*M_PI*(0.5     - x[k*3]);
      f[k*nSpace] =   vx*u[k];
      f[k*nSpace+1] = vy*u[k];
      df[k*nSpace] = vx;
      df[k*nSpace+1] = vy;
    }
}

void unitCubeRotationEvaluate(const int nPoints,
			      const int nSpace,
			      const double *x,
			      const double *u,
			      double *m,
			      double *dm,
			      double *f,
			      double *df)
{
  double vx, vy, vz;
  int k;
  for (k=0; k < nPoints; k++)
    {
      m[k] = u[k];
      dm[k] = 1.0;
      vx = 2.0*M_PI*(x[k*3+1] - 0.5);
      vy = 2.0*M_PI*(0.5     - x[k*3]);
      vz = 0.0;
      f[k*nSpace] =   vx*u[k];
      f[k*nSpace+1] = vy*u[k];
      f[k*nSpace+2] = vz*u[k];
      df[k*nSpace] = vx;
      df[k*nSpace+1] = vy;
      df[k*nSpace+2] = vz;
    }
}

void rotatingPulseVelEvaluate(const int nPoints,
                              const int nSpace,
                              const double self_a,
                              const double *x,
                              const double *u,
                              double *m,
                              double *dm,
                              double *f,
                              double *df,
                              double *a,
                              double *da,
                              double *phi,
                              double *dphi)
{
  /*mwf add variable declarations*/
  double vx,vy;
  int k,I;
  const int nSpace2 = nSpace*nSpace;
  memset(da, 0, nPoints * nSpace2 * sizeof(double));
  memcpy(m, u, nPoints * sizeof(double)); 
  memcpy(phi, u, nPoints * sizeof(double)); 


  for (k=0; k < nPoints; k++)
    {
      dm[k] = dphi[k] = 1.0;

      vx = -4.0*(x[k*3+1] - 0.5);
      vy = 4.0*(x[k*3] - 0.5);
      f[k*nSpace] = vx*u[k];
      f[k*nSpace+1] = vy*u[k];
      df[k*nSpace] = vx;
      df[k*nSpace+1] = vy;

      for (I=0; I < nSpace; I++)
        {
          a[k*nSpace2 + I*nSpace + I] = self_a;
        }
    }
}

void disRotatingPulseVelEvaluate(const int nPoints,
                                 const int nSpace,
                                 const double self_a,
                                 const double *x,
                                 const double *u,
                                 double *m,
                                 double *dm,
                                 double *f,
                                 double *df,
                                 double *a,
                                 double *da,
                                 double *phi,
                                 double *dphi)
{
  double X,Y,vx,vy;
  int k,I;
  const int nSpace2 = nSpace*nSpace;

  memcpy(m, u, nPoints * sizeof(double));
  memcpy(phi, u, nPoints * sizeof(double)); 
  memset(da, 0, nPoints * nSpace2 * sizeof(double)); 

  for (k=0; k < nPoints; k++)
    {
      dm[k] = dphi[k] = 1.0;

      X = x[k*3+1] - 0.5;
      Y = x[k*3] - 0.5;
      vx = -4.0*X;
      vy = 4.0*Y;
      f[k*nSpace] = vx*u[k];
      f[k*nSpace+1] = vy*u[k];
      df[k*nSpace] = vx;
      df[k*nSpace+1] = vy;
      if (sqrt(X*X+Y*Y) < 0.25)
        {
          f[k*nSpace] *= 0.001;
          f[k*nSpace+1] *= 0.001;
          df[k*nSpace] *= 0.001;
          df[k*nSpace+1] *= 0.001;
        }

      for (I=0; I < nSpace; I++)
        {
          a[k*nSpace2 + I*nSpace + I] = self_a;
        }
    }
}

void disVelEvaluate(const int nPoints,
                    const int nSpace,
                    const double self_a,
                    const double *x,
                    const double *u,
                    double *m,
                    double *dm,
                    double *f,
                    double *df,
                    double *a,
                    double *da,
                    double *phi,
                    double *dphi)
{
  proteus_index_t J; int k, I;
  const int nSpace2 = nSpace*nSpace;

  for (k=0; k < nPoints; k++)
    {
      m[k] = u[k];
      phi[k] = 0.0;
      dm[k] = dphi[k] = 1.0;

      f[k*nSpace] = u[k];
      f[k*nSpace+1] = 0.0;
      df[k*nSpace] = 1.0;
      df[k*nSpace+1] = 0.0;
      if (x[k*3+1] > 0.5)
        {
          f[k*nSpace] *= 0.25;
          df[k*nSpace] *= 0.25;
        }

      for (I=0; I < nSpace; I++)
        {
          a[k*nSpace2 + I*nSpace + I] = self_a;

          for (J=0; J < nSpace; J++)
            {
              da[k*nSpace2 + I*nSpace + J] = 0.0;
            }
        }
    }
}

void burgersDiagonalVelEvaluate(const int nPoints,
                                const int nSpace,
                                const double self_a,
				const double *self_v,
                                const double *u,
                                double *m,
                                double *dm,
                                double *f,
                                double *df,
                                double *a,
                                double *phi,
                                double *dphi)
{
  double u2;
  int k,I;
  const int nSpace2 = nSpace*nSpace;
  /*mwf changed to remove maxu0*/
  for (k=0; k < nPoints; k++)
    {
      m[k] = phi[k] = u[k];
      dm[k] = dphi[k] = 1.0;
      u2    = u[k]*u[k];
      for (I=0; I < nSpace; I++)
        {
          a[k*nSpace2 + I*nSpace + I] = self_a;
          f[k*nSpace+I] = self_v[I] * u[k] * u[k] * 0.5;
          df[k*nSpace+I]= self_v[I] * u[k];

/*           for (J=0; J < nSpace; J++) */
/*             { */
/*               da[k*nSpace2 + I*nSpace + J] = 0.0; */
/*             } */
        }
    }
}

void burgersDiagonalVelHJEvaluate(const int nPoints,
				  const int nSpace,
				  const double self_a,
				  const double *self_v,
				  const double *u,
				  const double *grad_u,
				  double *m,
				  double *dm,
				  double *H,
				  double *dH,
				  double *a,
				  double *phi,
				  double *dphi)
{
  double u2;
  int k,I;
  const int nSpace2 = nSpace*nSpace;
  /*mwf changed to remove maxu0*/
  for (k=0; k < nPoints; k++)
    {
      m[k] = phi[k] = u[k];
      dm[k] = dphi[k] = 1.0;
      u2    = u[k]*u[k];
      H[k] = 0.0;
      for (I=0; I < nSpace; I++)
        {
          a[k*nSpace2 + I*nSpace + I] = self_a;
          H[k] += self_v[I] * grad_u[k*nSpace+I] * u[k];
          dH[k*nSpace+I]= self_v[I] * u[k];

/*           for (J=0; J < nSpace; J++) */
/*             { */
/*               da[k*nSpace2 + I*nSpace + J] = 0.0; */
/*             } */
        }
    }
}

void twophasePotentialFlowEvaluate(int nPoints,
                                   int nSpace,
                                   double *M,
                                   double *A,
                                   double *B,
                                   double *Bcon,
                                   double *C,
                                   double t,
                                   double *x,
                                   double *u,
                                   double *m,
                                   double *dm,
                                   double *f,
                                   double *df,
                                   double *a,
                                   double *da,
                                   double *phi,
                                   double *dphi,
                                   double *r,
                                   double *dr)
{
  proteus_index_t J; int k, I;
  const int nSpace2=nSpace*nSpace;
  for (k=0;k<nPoints;k++)
    {
      m[k]=M[k]*u[k];
      dm[k]=M[k];
      for (I=0;I<nSpace;I++)
        {
          f[k*nSpace+I]=B[k*nSpace+I]*u[k] + Bcon[k*nSpace+I];
          df[k*nSpace+I]=B[k*nSpace+I];
          for (J=0;J<nSpace;J++)
            {
              a[k*nSpace2+I*nSpace+J]=A[k*nSpace2 + I*nSpace+J];
              da[k*nSpace2+I*nSpace+J]=0.0;
            }
        }
      
      phi[k]=u[k];
      dphi[k]=1.0;
      
      r[k]=C[k]*u[k];
      dr[k]=C[k];
    }
}

void twophasePotentialFlowUpdateFreeSurface(int nPoints,
                                            int nSpace,
                                            double eps,
                                            double* u_levelSet,
                                            double M1, double M2, double *M,
                                            double* A1, double* A2, double *A,
                                            double* B1, double* B2, double *B,
                                            double* Bcon1, double* Bcon2, double *Bcon,
                                            double C1, double C2, double *C)
{
  proteus_index_t J; int k, I;
  const int nSpace2=nSpace*nSpace;
  double H,oneMinusH;
  for (k=0;k<nPoints;k++)
    {
      if (u_levelSet[k] > eps)
        {
          M[k] = M1;
          for (I=0;I<nSpace;I++)
            {
              B[k*nSpace+I]=B1[I];
              Bcon[k*nSpace+I]=Bcon1[I];
              for (J=0;J<nSpace;J++)
                {
                  A[k*nSpace2+I*nSpace+J]=A1[I*nSpace+J];
                }
            }
          C[k] = C1;
        }
      else if (u_levelSet[k] < -eps)
        {
          M[k] = M2;
          for (I=0;I<nSpace;I++)
            {
              B[k*nSpace+I]=B2[I];
              Bcon[k*nSpace+I]=Bcon2[I];
              for (J=0;J<nSpace;J++)
                {
                  A[k*nSpace2+I*nSpace+J]=A2[I*nSpace+J];
                }
            }
          C[k] = C2;
        }
      else
        {
          H = 0.5*(1.0 + u_levelSet[k]/eps + sin((M_PI*u_levelSet[k])/eps)/M_PI);
          oneMinusH=1.0-H;
          M[k] = oneMinusH*M2 + H*M1;
          for (I=0;I<nSpace;I++)
            {
              B[k*nSpace+I]=oneMinusH*B2[I] + H*B1[I];
              Bcon[k*nSpace+I]=oneMinusH*Bcon2[I] + H*Bcon1[I];
              for (J=0;J<nSpace;J++)
                {
                  A[k*nSpace2+I*nSpace+J]=oneMinusH*A2[I*nSpace+J] + H*A1[I*nSpace+J];
                }
            }
          C[k]=oneMinusH*C2+H*C1;
        }
    }
}

void twophaseLevelSetCoefficientsUpdateVelocity(int nPoints,
                                                int nSpace,
                                                double v_scale,
                                                double* vIn,
                                                double* vOut)
{
  int i,I;
  for  (i=0;i<nPoints;i++)
    for (I=0;I<nSpace;I++)
      vOut[i*nSpace+I]=v_scale*vIn[i*nSpace+I];
}

void twophaseLevelSetCoefficientsEvaluate(int nPoints,
                                          int nSpace,
                                          double* B,
                                          double  t,
                                          double* x,
                                          double* u,
                                          double* grad_u,
                                          double* m, double* dm,
                                          double* h, double* dh,
                                          double* rh)
{
  int i,I;
  for (i=0;i<nPoints;i++)
    {
      rh[i]=0.0;
      h[i]=0.0;
      m[i]=u[i];
      dm[i]=1.0;
      for (I=0;I<nSpace;I++)
        {
          h[i] += B[i*nSpace+I]*grad_u[i*nSpace+I];
          dh[i*nSpace+I]=B[i*nSpace+I];
        }
    }
}

void twophaseLevelSetCoefficientsEvaluateCI(int nPoints,
                                            int nSpace,
                                            double* B,
                                            double  t,
                                            double* x,
                                            double* u,
                                            double* m, double* dm,
                                            double* f, double* df,
                                            double* a, double* da,
                                            double* phi, double* dphi,
                                            double* r, double* dr)
{
  int i,I;
  for (i=0;i<nPoints;i++)
    {
      m[i]=u[i];
      dm[i]=1.0;
      for (I=0;I<nSpace;I++)
        {
          f[i*nSpace+I] = B[i*nSpace+I]*u[i];
          df[i*nSpace+I]=B[i*nSpace+I];
        }
    }
}

void ncLevelSetCoefficientsEvaluate(int nPoints,
                                    int nSpace,
                                    double* v,
                                    double* u,
                                    double* grad_u,
                                    double* m,
                                    double* dm,
                                    double* H,
                                    double* dH)
{
  int i,I;
  for (i=0;i<nPoints;i++)
    {
      m[i]=u[i];
      dm[i]=1.0;
      H[i] = 0.0;
      for (I=0;I<nSpace;I++)
        {
          H[i] += v[i*nSpace+I]*grad_u[i*nSpace+I];
          dH[i*nSpace+I] = v[i*nSpace+I];
        }
    }
}

void cLevelSetCoefficientsEvaluate(int nPoints,
                                   int nSpace,
                                   double* v,
                                   double* u,
                                   double* m,
                                   double* dm,
                                   double* f,
                                   double* df)
{
  int i,I;
  for (i=0;i<nPoints;i++)
    {
      m[i]=u[i];
      dm[i]=1.0;
      for (I=0;I<nSpace;I++)
        {
          f[i*nSpace+I] = v[i*nSpace+I]*u[i];
          df[i*nSpace+I] = v[i*nSpace+I];
        }
    }
}

void VOFCoefficientsEvaluate(int nPoints,
                             int nSpace,
                             double eps,
                             double* v,
                             double* phi,
                             double* u,
                             double* m,
                             double* dm,
                             double* f,
                             double* df)
{
  int i,I;
  for (i=0;i<nPoints;i++)
    {
      m[i]=u[i];
      dm[i]=1.0;
      for (I=0;I<nSpace;I++)
        {
/*           f[i*nSpace+I] = v[i*nSpace+I]*smoothedHeaviside(eps,phi[i]); */
/*           df[i*nSpace+I] = 0.0; */
          f[i*nSpace+I] = v[i*nSpace+I]*u[i];
          df[i*nSpace+I] = v[i*nSpace+I];
        }
    }
}

void levelSetCurvatureCoefficientsEvaluate(int nPoints,
                                           int nSpace,
                                           double *grad_phi,
                                           double *u,
                                           double *f,
                                           double *r,
                                           double *dr)
{
  int i,I;
  double norm_grad_phi;
  for (i=0;i<nPoints;i++)
    {
      r[i] = u[i];
      dr[i] = 1.0;
      norm_grad_phi = 0.0;
      for (I=0;I<nSpace;I++)
        norm_grad_phi += grad_phi[i*nSpace+I]*grad_phi[i*nSpace+I];
      norm_grad_phi = sqrt(norm_grad_phi);
      if (norm_grad_phi > 0.0)
        {
          for (I=0;I<nSpace;I++)
            f[i*nSpace+I] = grad_phi[i*nSpace+I]/norm_grad_phi;
        }
      else
        f[i*nSpace+I] = 0.0;
    }
}

void twophaseSignedDistanceCoefficientsUpdateSignFunction(int nPoints,
                                                          double eps,
                                                          double* u_levelSet,
                                                          double* S)
{
  int k;
  double H;
  for (k=0;k<nPoints;k++)
    {
      if (u_levelSet[k] > eps)
	S[k]= 1.0;
      else if (u_levelSet[k] < -eps)
        S[k] = -1.0;
      else
        {
          H = 0.5*(1.0 + u_levelSet[k]/eps + sin((M_PI*u_levelSet[k])/eps)/M_PI);
          S[k]=2.0*H - 1.0;
        }
    }
/*   for  (k=0;k<nPoints;k++) */
/*     S[k] = u_levelSet[k]/sqrt(u_levelSet[k]*u_levelSet[k] +  eps*eps); */
/*     if(u_levelSet[k] > 0.0) */
/*       S[k] = 1.0; */
/*     else */
/*       S[k] = -1.0; */
}

void twophaseSignedDistanceCoefficientsEvaluate(int nPoints,
                                                int nSpace,
                                                double* S,
                                                double* u,
                                                double* grad_u,
                                                double* m, double* dm,
                                                double* h, double* dh,
                                                double* rh)
{
  int i,I;
  for (i=0;i<nPoints;i++)
    {
      m[i]=u[i];
      dm[i]=1.0;
      rh[i]=-1.0;
      h[i]=0.0;
      for (I=0;I<nSpace;I++)
        h[i] += grad_u[i*nSpace+I]*grad_u[i*nSpace+I];
      h[i] = sqrt(h[i]);
      for (I=0;I<nSpace;I++)
	if(h[i]>= fabs(S[i]*grad_u[i*nSpace+I])*1.0e-8)
	  dh[i*nSpace+I] = (S[i]*grad_u[i*nSpace+I])/h[i];
	else
          dh[i*nSpace+I] = (S[i]*grad_u[i*nSpace+I])/1.0e-8;
      h[i]+=rh[i];
      h[i]*=S[i];
      rh[i]*=S[i];
    }
}
void eikonalEquationEvaluate(int nPoints,
			     int nSpace,
			     double rhs,
			     double* u,
			     double* grad_u,
			     double* m,
			     double* dm,
			     double* H,
			     double* dH,
			     double* r)
{
  int i,I;
  double normGradU;
  for (i=0;i<nPoints;i++)
    {
      m[i]=u[i];
      dm[i]=1.0;
      H[i] = 0.0;
      r[i]=-rhs;
      normGradU=0.0;
      for (I=0;I<nSpace;I++)
	normGradU+= grad_u[i*nSpace+I]*grad_u[i*nSpace+I];
      normGradU = sqrt(normGradU);
      H[i] = normGradU;
      for (I=0;I<nSpace;I++)
        {
          dH[i*nSpace+I] = grad_u[i*nSpace+I]/(normGradU+1.0e-12);
        }/*I*/
    }/*i*/
}
void redistanceLevelSetCoefficientsEvaluate(int nPoints,
					    int nSpace,
					    double eps,
					    double* u_levelSet,
					    double* u,
					    double* grad_u,
					    double* m,
					    double* dm,
					    double* H,
					    double* dH,
					    double* r)
{
  int i,I;
  double Si,normGradU;/* He */
  /*mwf debug
  printf("redistanceLS nPoints= %d nSpace= %d eps= %g \n",nPoints,nSpace,eps); 
  */
  for (i=0;i<nPoints;i++)
    {
      m[i]=u[i];
      dm[i]=1.0;
      H[i] = 0.0;
      Si = -1.0+2.0*smoothedHeaviside(eps,u_levelSet[i]);
/*       if (u_levelSet[i] > eps) */
/* 	Si=1.0; */
/*       else if (u_levelSet[i] < -eps) */
/* 	Si=-1.0; */
/*       else */
/* 	{ */
/* 	  He=0.5*(1.0 + u_levelSet[i]/eps + sin(M_PI*u_levelSet[i]/eps)/M_PI); */
/* 	  Si= 2.0*He-1.0; */
/* 	} */
      /*mwf now try just straight sign with small eps*/
      /*Si = u_levelSet[i]/sqrt(u_levelSet[i]*u_levelSet[i]+1.0e-12)*/;
      /*
	r =-S
	H =S*|\grad d|
	dH=S*\grad d/|\grad d|
       */
      r[i]=-Si;
      normGradU=0.0;
      for (I=0;I<nSpace;I++)
	normGradU+= grad_u[i*nSpace+I]*grad_u[i*nSpace+I];
      normGradU = sqrt(normGradU);
      H[i] = Si*normGradU;
      /*
	mwf debug what about solving with r= 0 and H=S*(|\grad d|-1)?
	no longer homogeneous of order 1, gets Hamiltonian wrong in stabilization
      */
      /*
	r[i] = 0.0;
	H[i] = Si*(normGradU-1.0);
      */
      for (I=0;I<nSpace;I++)
        {
          dH[i*nSpace+I] = Si*grad_u[i*nSpace+I]/(normGradU+1.0e-12);
        }/*I*/
    }/*i*/
}
void redistanceLevelSetCoefficientsWithWeakPenaltyEvaluate(int nPoints,
							   int nSpace,
							   double eps,
							   double lambda_penalty,
							   double* u_levelSet,
							   double* u,
							   double* grad_u,
							   double* m,
							   double* dm,
							   double* H,
							   double* dH,
							   double* r,
							   double* dr)
{
  int i,I;
  double Si,normGradU;/* He */
  /*mwf debug
  printf("redistanceLS nPoints= %d nSpace= %d eps= %g \n",nPoints,nSpace,eps); 
  */
  for (i=0;i<nPoints;i++)
    {
      m[i]=u[i];
      dm[i]=1.0;
      H[i] = 0.0;
      Si = -1.0+2.0*smoothedHeaviside(eps,u_levelSet[i]);
/*       if (u_levelSet[i] > eps) */
/* 	Si=1.0; */
/*       else if (u_levelSet[i] < -eps) */
/* 	Si=-1.0; */
/*       else */
/* 	{ */
/* 	  He=0.5*(1.0 + u_levelSet[i]/eps + sin(M_PI*u_levelSet[i]/eps)/M_PI); */
/* 	  Si= 2.0*He-1.0; */
/* 	} */
      /*mwf now try just straight sign with small eps*/
      /*Si = u_levelSet[i]/sqrt(u_levelSet[i]*u_levelSet[i]+1.0e-12)*/;
      /*
	r =-S
	H =S*|\grad d|
	dH=S*\grad d/|\grad d|
       */
      r[i]=-Si;
      normGradU=0.0;
      for (I=0;I<nSpace;I++)
	normGradU+= grad_u[i*nSpace+I]*grad_u[i*nSpace+I];
      normGradU = sqrt(normGradU);
      H[i] = Si*normGradU;
      /*
	mwf debug what about solving with r= 0 and H=S*(|\grad d|-1)?
	no longer homogeneous of order 1, gets Hamiltonian wrong in stabilization
      */
      /*
	r[i] = 0.0;
	H[i] = Si*(normGradU-1.0);
      */
      for (I=0;I<nSpace;I++)
        {
          dH[i*nSpace+I] = Si*grad_u[i*nSpace+I]/(normGradU+1.0e-12);
        }/*I*/
      /*add in weak penalty*/
      r[i] += (u[i]-u_levelSet[i])*lambda_penalty*smoothedDirac(eps,u_levelSet[i]);
      dr[i] = lambda_penalty*smoothedDirac(eps,u_levelSet[i]);
    }/*i*/
}

/*
   redistance as before, but include lagrange multiplier source term to improve volume
   conservation following Sussman and Fatemi 99
*/
void redistanceLevelSetSandFCoefficientsEvaluate(int nSimplex, 
						 int nPointsPerSimplex,
						 int nSpace,
						 double eps,
						 double* u_levelSet,
						 double* dV,
						 double* u,
						 double* grad_u,
						 double* m,
						 double* dm,
						 double* H,
						 double* dH,
						 double* r)
{
  int ie,iq,i,I;
  double He,Si,normGradU,
    dHe,Li,lambda;
  /*mwf debug
  printf("redistanceLSSandF nSimplex=%d nPointsPerSimplex= %d nSpace= %d eps= %g \n",nSimplex,
	 nPointsPerSimplex,nSpace,eps); 
  */
  for (ie=0; ie < nSimplex; ie++)
    {
      /*first loop through simplex and compute normal coefficients,
	then loop back through and include lagrange multiplier term
      */
      for (iq=0;iq<nPointsPerSimplex;iq++)
	{
	  i = ie*nPointsPerSimplex + iq;
	  m[i]=u[i];
	  dm[i]=1.0;
	  H[i] = 0.0;
	  dHe = 0.0; 
	  if (u_levelSet[i] > eps)
	    {
	      Si=1.0; He = 1.0;
	    }
	  else if (u_levelSet[i] < -eps)
	    {
	      Si=-1.0; He = 0.0;
	    }
	  else
	    {
	      He =0.5*(1.0 + u_levelSet[i]/eps + sin(M_PI*u_levelSet[i]/eps)/M_PI);
	      dHe=0.5*(0.0 + 1.0/eps + cos(M_PI*u_levelSet[i]/eps)/eps);
	      Si= 2.0*He-1.0;
	    }
	  normGradU=0.0;
	  for (I=0;I<nSpace;I++)
	    normGradU+= grad_u[i*nSpace+I]*grad_u[i*nSpace+I];
	  normGradU = sqrt(normGradU);
	  /*now loop through simplex and compute averages*/
	  r[i]=-Si;
	  H[i] = Si*normGradU;
	  for (I=0;I<nSpace;I++)
	    {
	      dH[i*nSpace+I] = Si*grad_u[i*nSpace+I]/(normGradU+1.0e-12);
	    }/*I*/
	}/*iq loop 1*/
      /*the following is wasteful*/
      lambda = 0.0;
      for (iq = 0; iq < nPointsPerSimplex; iq++)
	{
	  i = ie*nPointsPerSimplex + iq;
	  Li= -r[i] - H[i];/* Si(1-|gradU|) */
	  dHe = 0.0;
	  if (u_levelSet[i] > eps)
	    {
	      Si=1.0; He = 1.0;
	    }
	  else if (u_levelSet[i] < -eps)
	    {
	      Si=-1.0; He = 0.0;
	    }
	  else
	    {
	      He =0.5*(1.0 + u_levelSet[i]/eps + sin(M_PI*u_levelSet[i]/eps)/M_PI);
	      dHe=0.5*(0.0 + 1.0/eps + cos(M_PI*u_levelSet[i]/eps)/eps);
	      Si= 2.0*He-1.0;
	    }
	  normGradU = H[i]/Si;
	  lambda -= dHe*Li*dV[i]/(dHe*dHe*normGradU+1.0e-8);
	}/*iq*/
      /*go back through and update source term*/
      for (iq = 0; iq < nPointsPerSimplex; iq++)
	{
	  i = ie*nPointsPerSimplex + iq;
	  dHe = 0.0; He=1.0;
	  if (u_levelSet[i] > eps)
	    {Si=1.0; He = 1.0;}
	  else if (u_levelSet[i] < -eps)
	    {Si=-1.0; He = 0.0;}
	  else
	    {
	      He =0.5*(1.0 + u_levelSet[i]/eps + sin(M_PI*u_levelSet[i]/eps)/M_PI);
	      dHe=0.5*(0.0 + 1.0/eps + cos(M_PI*u_levelSet[i]/eps)/eps);
	      Si= 2.0*He-1.0;
	    }
	  normGradU = H[i]/Si;
	  r[i] -= lambda*dHe*normGradU;/*recall r is minus right hand side*/
	  /*mwf debug
	  printf("redistSandF ie=%d iq=%d u=%g He=%g dHe=%g Si=%g |gradu|=%g lam=%g r=%g\n",
	  ie,iq,u_levelSet[i],He,dHe,Si,normGradU,lambda,r[i]);
	  */
	}/*iq*/
    }/*ie*/
}/*func*/

void setWeakDirichletConditionsForLevelSet(proteus_index_t nElements_global,
					   int nDOF_trial_element,
					   double epsilon_freeze_factor,
					   const double *elementDiameter,
					   const int * u_l2g,
					   const double *u_dof,
					   int * freeze_nodes_tmp,
					   int * weakDirichletConditionFlags)
{
  proteus_index_t J; int eN, j, jj, signU, j0, J0;
  double eps;
  for (eN = 0; eN < nElements_global; eN++)
    {
      for (j = 0; j < nDOF_trial_element; j++)
	freeze_nodes_tmp[j] = 0;
      eps = epsilon_freeze_factor*elementDiameter[eN];
      signU = 0; j0 = 0;
      while (signU == 0 && j0 < nDOF_trial_element)
	{
	  J0 = u_l2g[eN*nDOF_trial_element + j0];
	  if (u_dof[J0] < -eps) 
	    signU = -1;
	  else if (u_dof[J0]  > eps) 
	    signU = 1;
	  else 
	    freeze_nodes_tmp[j0] = 1;
	  j0++;
	}
      for (j = j0; j < nDOF_trial_element; j++)
	{
	  J = u_l2g[eN*nDOF_trial_element + j];
	  if ((u_dof[J] < -eps && signU == 1) ||
	      (u_dof[J] > eps && signU == -1))
	    {
	      for (jj = 0; jj < nDOF_trial_element; jj++)
		freeze_nodes_tmp[jj] = 1;
	      break;
	    }
	  else if (fabs(u_dof[J]) < eps)
	    freeze_nodes_tmp[j] = 1;
	}
      for (j = 0; j < nDOF_trial_element; j++)
	{
	  if (freeze_nodes_tmp[j] == 1)
	    {
	      J = u_l2g[eN*nDOF_trial_element + j];
	      weakDirichletConditionFlags[J] = 1;
	    }
	}
    }//eN
}

void setSimpleWeakDirichletConditionsForLevelSet(proteus_index_t nElements_global,
                                                 int nDOF_trial_element,
                                                 double epsilon_freeze_factor,
                                                 const double *elementDiameter,
                                                 const int * u_l2g,
                                                 const double *u_dof,
                                                 int * freeze_nodes_tmp,
                                                 int * weakDirichletConditionFlags)
{
  proteus_index_t J; int eN, j;
  double eps;
  for (eN = 0; eN < nElements_global; eN++)
    {
      eps = epsilon_freeze_factor*elementDiameter[eN];
      for (j = 0; j < nDOF_trial_element; j++)
        {
          J = u_l2g[eN*nDOF_trial_element + j];
          if (fabs(u_dof[J]) < eps)
            weakDirichletConditionFlags[J] = 1;
        }
    }
}

/***********************************************************************
 begin two phase potential flow duplication efforts by mwf
 ***********************************************************************/
void darcySharpInterfaceFlowEvaluate(int nPoints,
				     int nSpace,
				     double Km, double rhoM,
				     double Kp, double rhoP,
				     double eps,
				     double * gravity_u,
				     double * u,
				     double * gradu,
				     double * u_levelSet,
				     double * phi_pot,
				     double * a,
				     double * f,
				     double * r,
				     double * m,
				     double * dphi_pot,
				     double * da,
				     double * df,
				     double * dr,
				     double * dm)
{
  proteus_index_t J; int k, I, nSpace2;
  double He;

  nSpace2 = nSpace*nSpace;
  for (k = 0; k < nPoints; k++)
    {
      m[k] = 0.0; dm[k] = 1.0;
      r[k] = 0.0; dr[k] = 0.0;
      phi_pot[k] =u[k];
      dphi_pot[k]=1.0;

      He = smoothedHeaviside(eps,u_levelSet[k]);
      /*mwf debug
	printf("u_ls[%d]= %g He=%g \n",k,u_levelSet[k],He);
      */
      for (I = 0; I < nSpace; I++)
	{
	  for (J = 0; J < nSpace; J++)
	    {
	      if (I==J)
		a[k*nSpace2 + I*nSpace+J] = Km + He*(Kp-Km);
	      else
		a[k*nSpace2 + I*nSpace+J] = 0.0;
	      da[k*nSpace2  + I*nSpace+J] = 0.0;
	    }
	  f[k*nSpace + I] = (Km*rhoM + He*(Kp*rhoP-Km*rhoM))*gravity_u[I];
	  df[k*nSpace + I]= 0.0;
	}/*I*/
    }/*k*/

}

void darcySharpInterfaceFlowImEvaluate(int nPoints,
				       int nSpace,
				       double Km, double rhoM,
				       double Kp, double rhoP,
				       double eps,
				       double * gravity_u,
				       double * u,
				       double * gradu,
				       double * u_levelSet,
				       double * phi_pot,
				       double * a,
				       double * f,
				       double * r,
				       double * m,
				       double * dphi_pot,
				       double * da,
				       double * df,
				       double * dr,
				       double * dm)
{
  proteus_index_t J; int k, I, nSpace2;
  double He,rhoHe;
  double rhoEps = 0.0; /*do not smear density*/
  nSpace2 = nSpace*nSpace;
  for (k = 0; k < nPoints; k++)
    {
      m[k] = 0.0; dm[k] = 1.0;
      r[k] = 0.0; dr[k] = 0.0;
      phi_pot[k] =u[k];
      dphi_pot[k]=1.0;

      He = smoothedHeaviside(eps,u_levelSet[k]);
      rhoHe = smoothedHeaviside(rhoEps,u_levelSet[k]);
      /*mwf debug
	printf("u_ls[%d]= %g He=%g \n",k,u_levelSet[k],He);
      */
      for (I = 0; I < nSpace; I++)
	{
	  for (J = 0; J < nSpace; J++)
	    {
	      if (I==J)
		a[k*nSpace2 + I*nSpace+J] = Km + He*(Kp-Km);
	      else
		a[k*nSpace2 + I*nSpace+J] = 0.0;
	      da[k*nSpace2  + I*nSpace+J] = 0.0;
	    }
	  f[k*nSpace + I] = (Km*rhoM + He*(Kp*rhoP-Km*rhoM))*gravity_u[I];
	  df[k*nSpace + I]= 0.0;
	}/*I*/
    }/*k*/

}

void Laplace_Evaluate2D(const int nPoints,
		      double *mom_p_diff_ten,
		      double *mom_u_diff_ten,
		      double *mom_v_diff_ten)
{
  int k;
  for (k=0; k<nPoints; k++)
    {
      mom_p_diff_ten[k*2+0] = 1.0;
      mom_p_diff_ten[k*2+1] = 1.0;

      mom_u_diff_ten[k*2+0] = 1.0;
      mom_u_diff_ten[k*2+1] = 1.0;

      mom_v_diff_ten[k*2+0] = 1.0;
      mom_v_diff_ten[k*2+1] = 1.0;
    } 
}

void Laplace_Evaluate3D(const int nPoints,
		      double *mom_p_diff_ten,
		      double *mom_u_diff_ten,
		      double *mom_v_diff_ten,
		      double *mom_w_diff_ten)
{
  int k;
  for (k=0; k<nPoints; k++)
    {
      mom_p_diff_ten[k*3+0] = 1.0;
      mom_p_diff_ten[k*3+1] = 1.0;
      mom_p_diff_ten[k*3+2] = 1.0;

      mom_u_diff_ten[k*3+0] = 1.0;
      mom_u_diff_ten[k*3+1] = 1.0;
      mom_u_diff_ten[k*3+2] = 1.0;

      mom_v_diff_ten[k*3+0] = 1.0;
      mom_v_diff_ten[k*3+1] = 1.0;
      mom_v_diff_ten[k*3+2] = 1.0;

      mom_w_diff_ten[k*3+0] = 1.0;
      mom_w_diff_ten[k*3+1] = 1.0;
      mom_w_diff_ten[k*3+2] = 1.0;
    } 
}


void NavierStokes_2D_Evaluate(const int nPoints,
                              const double rho,
                              const double nu,
                              const double *g,
                              const double *p,
                              const double *grad_p,
                              const double *u,
                              const double *v,
                              double *mom_u_acc,
                              double *dmom_u_acc_u,
                              double *mom_v_acc,
                              double *dmom_v_acc_v,
                              double *mass_adv,
                              double *dmass_adv_u,
                              double *dmass_adv_v,
                              double *mom_u_adv,
                              double *dmom_u_adv_u,
                              double *dmom_u_adv_v,
                              double *mom_v_adv,
                              double *dmom_v_adv_u,
                              double *dmom_v_adv_v,
                              double *mom_u_diff_ten,
                              double *mom_v_diff_ten,
                              double *mom_u_source,
                              double *mom_v_source,
                              double *mom_u_ham,
                              double *dmom_u_ham_grad_p,
                              double *mom_v_ham,
                              double *dmom_v_ham_grad_p)
{
  int k;
  for (k=0;k<nPoints;k++)
    {
      /** \todo optimize coefficients */
      //momentum accumulation
      mom_u_acc[k]=u[k];
      dmom_u_acc_u[k]=1.0;
      
      mom_v_acc[k]=v[k];
      dmom_v_acc_v[k]=1.0;
      
      //mass advective flux
      mass_adv[k*2+0]=u[k];
      mass_adv[k*2+1]=v[k];
      
      dmass_adv_u[k*2+0]=1.0;
      dmass_adv_v[k*2+1]=1.0;

      //u momentum advective flux
      mom_u_adv[k*2+0]=u[k]*u[k];
      mom_u_adv[k*2+1]=u[k]*v[k];

      dmom_u_adv_u[k*2+0]=2.0*u[k];
      dmom_u_adv_u[k*2+1]=v[k];

      dmom_u_adv_v[k*2+1]=u[k];

      //v momentum advective_flux
      mom_v_adv[k*2+0]=v[k]*u[k];
      mom_v_adv[k*2+1]=v[k]*v[k];
      
      dmom_v_adv_u[k*2+0]=v[k];
      
      dmom_v_adv_v[k*2+0]=u[k];
      dmom_v_adv_v[k*2+1]=2.0*v[k];

      //u momentum diffusion tensor
      mom_u_diff_ten[k*4+0] = nu;
      mom_u_diff_ten[k*4+3] = nu;
      
      //v momentum diffusion tensor
      mom_v_diff_ten[k*4+0] = nu;
      mom_v_diff_ten[k*4+3] = nu;
      
      //momentum sources
      mom_u_source[k] = -g[0];
      mom_v_source[k] = -g[1];
      
      //u momentum Hamiltonian (pressure)
      mom_u_ham[k] = grad_p[k*2+0]/rho;
      dmom_u_ham_grad_p[k*2+0]=1.0/rho;

      //v momentum Hamiltonian (pressure)
      mom_v_ham[k] = grad_p[k*2+1]/rho;
      dmom_v_ham_grad_p[k*2+1]=1.0/rho;
    }
}

void NavierStokes_3D_Evaluate(const int nPoints,
                              const double rho,
                              const double nu,
                              const double *g,
                              const double *p,
                              const double *grad_p,
                              const double *u,
                              const double *v,
                              const double *w,
                              double *mom_u_acc,
                              double *dmom_u_acc_u,
                              double *mom_v_acc,
                              double *dmom_v_acc_v,
                              double *mom_w_acc,
                              double *dmom_w_acc_w,
                              double *mass_adv,
                              double *dmass_adv_u,
                              double *dmass_adv_v,
                              double *dmass_adv_w,
                              double *mom_u_adv,
                              double *dmom_u_adv_u,
                              double *dmom_u_adv_v,
                              double *dmom_u_adv_w,
                              double *mom_v_adv,
                              double *dmom_v_adv_u,
                              double *dmom_v_adv_v,
                              double *dmom_v_adv_w,
                              double *mom_w_adv,
                              double *dmom_w_adv_u,
                              double *dmom_w_adv_v,
                              double *dmom_w_adv_w,
                              double *mom_u_diff_ten,
                              double *mom_v_diff_ten,
                              double *mom_w_diff_ten,
                              double *mom_u_source,
                              double *mom_v_source,
                              double *mom_w_source,
                              double *mom_u_ham,
                              double *dmom_u_ham_grad_p,
                              double *mom_v_ham,
                              double *dmom_v_ham_grad_p,
                              double *mom_w_ham,
                              double *dmom_w_ham_grad_p)
{
  int k;
  for (k=0;k<nPoints;k++)
    {
      /** \todo optimize coefficients */
      //momentum accumulation

      mom_u_acc[k]=u[k];
      dmom_u_acc_u[k]=1.0;
      
      mom_v_acc[k]=v[k];
      dmom_v_acc_v[k]=1.0;
      
      mom_w_acc[k]=w[k];
      dmom_w_acc_w[k]=1.0;
      
      //mass advective flux
      mass_adv[k*3+0]=u[k];
      mass_adv[k*3+1]=v[k];
      mass_adv[k*3+2]=w[k];
      
      dmass_adv_u[k*3+0]=1.0;
      dmass_adv_v[k*3+1]=1.0;
      dmass_adv_w[k*3+2]=1.0;

      //u momentum advective flux
      mom_u_adv[k*3+0]=u[k]*u[k];
      mom_u_adv[k*3+1]=u[k]*v[k];
      mom_u_adv[k*3+2]=u[k]*w[k];

      dmom_u_adv_u[k*3+0]=2.0*u[k];
      dmom_u_adv_u[k*3+1]=v[k];
      dmom_u_adv_u[k*3+2]=w[k];

      dmom_u_adv_v[k*3+1]=u[k];

      dmom_u_adv_w[k*3+2]=u[k];

      //v momentum advective_flux
      mom_v_adv[k*3+0]=v[k]*u[k];
      mom_v_adv[k*3+1]=v[k]*v[k];
      mom_v_adv[k*3+2]=v[k]*w[k];
      
      dmom_v_adv_u[k*3+0]=v[k];
      
      dmom_v_adv_v[k*3+0]=u[k];
      dmom_v_adv_v[k*3+1]=2.0*v[k];
      dmom_v_adv_v[k*3+2]=w[k];

      dmom_v_adv_w[k*3+2]=v[k];
      
      //w momentum advective_flux
      mom_w_adv[k*3+0]=w[k]*u[k];
      mom_w_adv[k*3+1]=w[k]*v[k];
      mom_w_adv[k*3+2]=w[k]*w[k];
      
      dmom_w_adv_u[k*3+0]=w[k];
      
      dmom_w_adv_v[k*3+0]=w[k];
      
      dmom_w_adv_w[k*3+0]=u[k];
      dmom_w_adv_w[k*3+1]=v[k];
      dmom_w_adv_w[k*3+2]=2.0*w[k];
      
      //u momentum diffusion tensor
      mom_u_diff_ten[k*9+0] = nu;
      mom_u_diff_ten[k*9+4] = nu;
      mom_u_diff_ten[k*9+8] = nu;
      
      //v momentum diffusion tensor
      mom_v_diff_ten[k*9+0] = nu;
      mom_v_diff_ten[k*9+4] = nu;
      mom_v_diff_ten[k*9+8] = nu;
      
      //w momentum diffusion tensor
      mom_w_diff_ten[k*9+0] = nu;
      mom_w_diff_ten[k*9+4] = nu;
      mom_w_diff_ten[k*9+8] = nu;
      
      //momentum sources
      mom_u_source[k] = -g[0];
      mom_v_source[k] = -g[1];
      mom_w_source[k] = -g[2];

      //u momentum Hamiltonian (pressure)
      mom_u_ham[k] = grad_p[k*3+0]/rho;
      dmom_u_ham_grad_p[k*3+0]=1.0/rho;

      //v momentum Hamiltonian (pressure)
      mom_v_ham[k] = grad_p[k*3+1]/rho;
      dmom_v_ham_grad_p[k*3+1]=1.0/rho;

      //w momentum Hamiltonian (pressure)
      mom_w_ham[k] = grad_p[k*3+2]/rho;
      dmom_w_ham_grad_p[k*3+2]=1.0/rho;
    }
}

void Stokes_2D_Evaluate(const int nPoints,
                        const double rho,
                        const double nu,
                        const double *g,
                        const double *p,
                        const double *grad_p,
                        const double *u,
                        const double *v,
                        double *mom_u_acc,
                        double *dmom_u_acc_u,
                        double *mom_v_acc,
                        double *dmom_v_acc_v,
                        double *mass_adv,
                        double *dmass_adv_u,
                        double *dmass_adv_v,
                        double *mom_u_diff_ten,
                        double *mom_v_diff_ten,
                        double *mom_u_source,
                        double *mom_v_source,
                        double *mom_u_ham,
                        double *dmom_u_ham_grad_p,
                        double *mom_v_ham,
                        double *dmom_v_ham_grad_p)
{
  int k;
  for (k=0;k<nPoints;k++)
    {
      /** \todo optimize coefficients */
      //momentum accumulation
      mom_u_acc[k]=u[k];
      dmom_u_acc_u[k]=1.0;
      
      mom_v_acc[k]=v[k];
      dmom_v_acc_v[k]=1.0;
      
      //mass advective flux
      mass_adv[k*2+0]=u[k];
      mass_adv[k*2+1]=v[k];
      
      dmass_adv_u[k*2+0]=1.0;
      dmass_adv_v[k*2+1]=1.0;

      //u momentum diffusion tensor
      mom_u_diff_ten[k*4+0] = nu;
      mom_u_diff_ten[k*4+3] = nu;
      
      //v momentum diffusion tensor
      mom_v_diff_ten[k*4+0] = nu;
      mom_v_diff_ten[k*4+3] = nu;
      
      //momentum sources
      mom_u_source[k] = -g[0];
      mom_v_source[k] = -g[1];
      
      //u momentum Hamiltonian (pressure)
      mom_u_ham[k] = grad_p[k*2+0]/rho;
      dmom_u_ham_grad_p[k*2+0]=1.0/rho;

      //v momentum Hamiltonian (pressure)
      mom_v_ham[k] = grad_p[k*2+1]/rho;
      dmom_v_ham_grad_p[k*2+1]=1.0/rho;
    }
}

void StokesP_2D_Evaluate(const int nPoints,
                         const double rho,
                         const double nu,
                         const double *g,
                         const double *p,
                         const double *u,
                         const double *v,
                         double *mom_u_acc,
                         double *dmom_u_acc_u,
                         double *mom_v_acc,
                         double *dmom_v_acc_v,
                         double *mass_adv,
                         double *dmass_adv_u,
                         double *dmass_adv_v,
                         double *mom_u_adv,
                         double *dmom_u_adv_p,
                         double *mom_v_adv,
                         double *dmom_v_adv_p,
                         double *mom_u_diff_ten,
                         double *mom_v_diff_ten,
                         double *mom_u_source,
                         double *mom_v_source)
{
  int k;
  for (k=0;k<nPoints;k++)
    {
      /** \todo optimize coefficients */
      //momentum accumulation
      mom_u_acc[k]=u[k];
      dmom_u_acc_u[k]=1.0;
      
      mom_v_acc[k]=v[k];
      dmom_v_acc_v[k]=1.0;
      
      //mass advective flux
      mass_adv[k*2+0]=u[k];
      mass_adv[k*2+1]=v[k];
      
      dmass_adv_u[k*2+0]=1.0;
      dmass_adv_v[k*2+1]=1.0;

      //u momentum advective flux
      mom_u_adv[k*2+0]=p[k]/rho;
      dmom_u_adv_p[k*2+0]=1.0/rho;

      //v momentum advective flux
      mom_v_adv[k*2+1]=p[k]/rho;
      dmom_v_adv_p[k*2+1]=1.0/rho;

      //u momentum diffusion tensor
      mom_u_diff_ten[k*4+0] = nu;
      mom_u_diff_ten[k*4+3] = nu;
      
      //v momentum diffusion tensor
      mom_v_diff_ten[k*4+0] = nu;
      mom_v_diff_ten[k*4+3] = nu;
      
      //momentum sources
      mom_u_source[k] = -g[0];
      mom_v_source[k] = -g[1];
    }
}

void Stokes_3D_Evaluate(const int nPoints,
                        const double rho,
                        const double nu,
                        const double *g,
                        const double *p,
                        const double *grad_p,
                        const double *u,
                        const double *v,
                        const double *w,
                        double *mom_u_acc,
                        double *dmom_u_acc_u,
                        double *mom_v_acc,
                        double *dmom_v_acc_v,
                        double *mom_w_acc,
                        double *dmom_w_acc_w,
                        double *mass_adv,
                        double *dmass_adv_u,
                        double *dmass_adv_v,
                        double *dmass_adv_w,
                        double *mom_u_diff_ten,
                        double *mom_v_diff_ten,
                        double *mom_w_diff_ten,
                        double *mom_u_source,
                        double *mom_v_source,
                        double *mom_w_source,
                        double *mom_u_ham,
                        double *dmom_u_ham_grad_p,
                        double *mom_v_ham,
                        double *dmom_v_ham_grad_p,
                        double *mom_w_ham,
                        double *dmom_w_ham_grad_p)
{
  int k;
  for (k=0;k<nPoints;k++)
    {
      /** \todo optimize coefficients */
      //momentum accumulation
      mom_u_acc[k]=u[k];
      dmom_u_acc_u[k]=1.0;
      
      mom_v_acc[k]=v[k];
      dmom_v_acc_v[k]=1.0;
      
      mom_w_acc[k]=w[k];
      dmom_w_acc_w[k]=1.0;
      
      //mass advective flux
      mass_adv[k*3+0]=u[k];
      mass_adv[k*3+1]=v[k];
      mass_adv[k*3+2]=w[k];

      dmass_adv_u[k*3+0]=1.0;
      dmass_adv_v[k*3+1]=1.0;
      dmass_adv_w[k*3+2]=1.0;
      
      //u momentum diffusion tensor
      mom_u_diff_ten[k*9+0] = nu;
      mom_u_diff_ten[k*9+4] = nu;
      mom_u_diff_ten[k*9+8] = nu;
      
      //v momentum diffusion tensor
      mom_v_diff_ten[k*9+0] = nu;
      mom_v_diff_ten[k*9+4] = nu;
      mom_v_diff_ten[k*9+8] = nu;
      
      //w momentum diffusion tensor
      mom_w_diff_ten[k*9+0] = nu;
      mom_w_diff_ten[k*9+4] = nu;
      mom_w_diff_ten[k*9+8] = nu;
      
      //momentum sources
      mom_u_source[k] = -g[0];
      mom_v_source[k] = -g[1];
      mom_w_source[k] = -g[2];

      //u momentum Hamiltonian (pressure)
      mom_u_ham[k] = grad_p[k*3+0]/rho;
      dmom_u_ham_grad_p[k*3+0]=1.0/rho;

      //v momentum Hamiltonian (pressure)
      mom_v_ham[k] = grad_p[k*3+1]/rho;
      dmom_v_ham_grad_p[k*3+1]=1.0/rho;

      //w momentum Hamiltonian (pressure)
      mom_w_ham[k] = grad_p[k*3+2]/rho;
      dmom_w_ham_grad_p[k*3+2]=1.0/rho;
    }
}

void StokesP_3D_Evaluate(const int nPoints,
                         const double rho,
                         const double nu,
                         const double *g,
                         const double *p,
                         const double *u,
                         const double *v,
                         const double *w,
                         double *mom_u_acc,
                         double *dmom_u_acc_u,
                         double *mom_v_acc,
                         double *dmom_v_acc_v,
			 double *mom_w_acc,
			 double *dmom_w_acc_w,
                         double *mass_adv,
                         double *dmass_adv_u,
                         double *dmass_adv_v,
			 double *dmass_adv_w,
                         double *mom_u_adv,
                         double *dmom_u_adv_p,
                         double *mom_v_adv,
                         double *dmom_v_adv_p,
			 double *mom_w_adv,
			 double *dmom_w_adv_p,
                         double *mom_u_diff_ten,
                         double *mom_v_diff_ten,
			 double *mom_w_diff_ten,
                         double *mom_u_source,
                         double *mom_v_source,
			 double *mom_w_source)
{
  int k;
  for (k=0;k<nPoints;k++)
    {
      /** \todo optimize coefficients */
      //momentum accumulation
      mom_u_acc[k]=u[k];
      dmom_u_acc_u[k]=1.0;
      
      mom_v_acc[k]=v[k];
      dmom_v_acc_v[k]=1.0;
      
      mom_w_acc[k]=w[k];
      dmom_w_acc_w[k]=1.0;
      
      //mass advective flux
      mass_adv[k*3+0]=u[k];
      mass_adv[k*3+1]=v[k];
      mass_adv[k*3+2]=w[k];
      
      dmass_adv_u[k*3+0]=1.0;
      dmass_adv_v[k*3+1]=1.0;
      dmass_adv_w[k*3+2]=1.0;

      //u momentum advective flux
      mom_u_adv[k*3+0]=p[k]/rho;
      dmom_u_adv_p[k*3+0]=1.0/rho;

      //v momentum advective flux
      mom_v_adv[k*3+1]=p[k]/rho;
      dmom_v_adv_p[k*3+1]=1.0/rho;

      //w momentum advective flux
      mom_w_adv[k*3+2]=p[k]/rho;
      dmom_w_adv_p[k*3+2]=1.0/rho;

      //u momentum diffusion tensor
      mom_u_diff_ten[k*9+0] = nu;
      mom_u_diff_ten[k*9+4] = nu;
      mom_u_diff_ten[k*9+8] = nu;
      
      //v momentum diffusion tensor
      mom_v_diff_ten[k*9+0] = nu;
      mom_v_diff_ten[k*9+4] = nu;
      mom_v_diff_ten[k*9+8] = nu;
      
      //w momentum diffusion tensor
      mom_w_diff_ten[k*9+0] = nu;
      mom_w_diff_ten[k*9+4] = nu;
      mom_w_diff_ten[k*9+8] = nu;
      
      //momentum sources
      mom_u_source[k] = -g[0];
      mom_v_source[k] = -g[1];
      mom_w_source[k] = -g[2];
    }
}

void TwophaseNavierStokes_LS_SO_2D_Evaluate(const int nPoints,
                                            const double eps,
                                            const double rho_0,
                                            const double nu_0,
                                            const double rho_1,
                                            const double nu_1,
                                            const double* g,
                                            const double* phi,
                                            const double *p,
                                            const double *grad_p,
                                            const double *u,
                                            const double *v,
                                            double *mom_u_acc,
                                            double *dmom_u_acc_u,
                                            double *mom_v_acc,
                                            double *dmom_v_acc_v,
                                            double *mass_adv,
                                            double *dmass_adv_u,
                                            double *dmass_adv_v,
                                            double *mom_u_adv,
                                            double *dmom_u_adv_u,
                                            double *dmom_u_adv_v,
                                            double *mom_v_adv,
                                            double *dmom_v_adv_u,
                                            double *dmom_v_adv_v,
                                            double *mom_u_diff_ten,
                                            double *mom_v_diff_ten,
                                            double *mom_u_source,
                                            double *mom_v_source,
                                            double *mom_u_ham,
                                            double *dmom_u_ham_grad_p,
                                            double *mom_v_ham,
                                            double *dmom_v_ham_grad_p)
{
  int k;
  double rho,nu,H;
  for (k=0;k<nPoints;k++)
    {
      /** \todo optimize coefficients */
      H = smoothedHeaviside(eps,phi[k]);
      rho = rho_0*(1.0-H)+rho_1*H;
      nu  = nu_0*(1.0-H)+nu_1*H;

      //u momentum accumulation
      mom_u_acc[k]=u[k];
      dmom_u_acc_u[k]=1.0;
      
      //v momentum accumulation
      mom_v_acc[k]=v[k];
      dmom_v_acc_v[k]=1.0;

      //mass advective flux
      mass_adv[k*2+0]=u[k];
      mass_adv[k*2+1]=v[k];
      
      dmass_adv_u[k*2+0]=1.0;
      dmass_adv_v[k*2+1]=1.0;

      //u momentum advective flux
      mom_u_adv[k*2+0]=u[k]*u[k];
      mom_u_adv[k*2+1]=u[k]*v[k];

      dmom_u_adv_u[k*2+0]=2.0*u[k];
      dmom_u_adv_u[k*2+1]=v[k];

      dmom_u_adv_v[k*2+1]=u[k];

      //v momentum advective_flux
      mom_v_adv[k*2+0]=v[k]*u[k];
      mom_v_adv[k*2+1]=v[k]*v[k];
            
      dmom_v_adv_u[k*2+0]=v[k];
      
      dmom_v_adv_v[k*2+0]=u[k];
      dmom_v_adv_v[k*2+1]=2.0*v[k];

      //u momentum diffusion tensor
      mom_u_diff_ten[k*4+0] = nu;
      mom_u_diff_ten[k*4+3] = nu;
      
      //v momentum diffusion tensor
      mom_v_diff_ten[k*4+0] = nu;
      mom_v_diff_ten[k*4+3] = nu;
      
      //momentum sources
      mom_u_source[k] = -g[0];
      mom_v_source[k] = -g[1];

      //u momentum Hamiltonian (pressure)
      mom_u_ham[k] = grad_p[k*2+0]/rho;
      dmom_u_ham_grad_p[k*2+0]=1.0/rho;
      
      //v momentum Hamiltonian (pressure)
      mom_v_ham[k] = grad_p[k*2+1]/rho;
      dmom_v_ham_grad_p[k*2+1]=1.0/rho;
    }
}

void TwophaseNavierStokes_ST_LS_SO_2D_Evaluate(const int nPoints,
                                               const double eps_rho,
                                               const double eps_mu,
                                               const double sigma,
                                               const double rho_0,
                                               const double nu_0,
                                               const double rho_1,
                                               const double nu_1,
                                               const double* g,
                                               const double* phi,
                                               const double* n,
                                               const double* kappa,
                                               const double *p,
                                               const double *grad_p,
                                               const double *u,
                                               const double *v,
                                               double *mom_u_acc,
                                               double *dmom_u_acc_u,
                                               double *mom_v_acc,
                                               double *dmom_v_acc_v,
                                               double *mass_adv,
                                               double *dmass_adv_u,
                                               double *dmass_adv_v,
                                               double *mom_u_adv,
                                               double *dmom_u_adv_u,
                                               double *dmom_u_adv_v,
                                               double *mom_v_adv,
                                               double *dmom_v_adv_u,
                                               double *dmom_v_adv_v,
                                               double *mom_u_diff_ten,
                                               double *mom_v_diff_ten,
                                               double *mom_uv_diff_ten,
                                               double *mom_vu_diff_ten,
                                               double *mom_u_source,
                                               double *mom_v_source,
                                               double *mom_u_ham,
                                               double *dmom_u_ham_grad_p,
                                               double *mom_v_ham,
                                               double *dmom_v_ham_grad_p)
{
  int k;
  double rho,nu,mu,H_rho,d_rho,H_mu,d_mu,norm_n;
  for (k=0;k<nPoints;k++)
    {
      /** \todo optimize coefficients */
      H_rho = smoothedHeaviside(eps_rho,phi[k]);
      d_rho = smoothedDirac(eps_rho,phi[k]);
      H_mu = smoothedHeaviside(eps_mu,phi[k]);
      d_mu = smoothedDirac(eps_mu,phi[k]);

      rho = rho_0*(1.0-H_rho)+rho_1*H_rho;
      nu  = nu_0*(1.0-H_mu)+nu_1*H_mu;
      mu  = rho_0*nu_0*(1.0-H_mu)+rho_1*nu_1*H_mu;

/*       //u momentum accumulation */
/*       mom_u_acc[k]=rho*u[k]; */
/*       dmom_u_acc_u[k]=rho; */
      
/*       //v momentum accumulation */
/*       mom_v_acc[k]=rho*v[k]; */
/*       dmom_v_acc_v[k]=rho; */

/*       //mass advective flux */
/*       mass_adv[k*2+0]=u[k]; */
/*       mass_adv[k*2+1]=v[k]; */
      
/*       dmass_adv_u[k*2+0]=1.0; */
/*       dmass_adv_v[k*2+1]=1.0; */

/*       //u momentum advective flux */
/*       mom_u_adv[k*2+0]=rho*u[k]*u[k]; */
/*       mom_u_adv[k*2+1]=rho*u[k]*v[k]; */

/*       dmom_u_adv_u[k*2+0]=2.0*rho*u[k]; */
/*       dmom_u_adv_u[k*2+1]=rho*v[k]; */

/*       dmom_u_adv_v[k*2+1]=rho*u[k]; */

/*       //v momentum advective_flux */
/*       mom_v_adv[k*2+0]=rho*v[k]*u[k]; */
/*       mom_v_adv[k*2+1]=rho*v[k]*v[k]; */
            
/*       dmom_v_adv_u[k*2+0]=rho*v[k]; */
      
/*       dmom_v_adv_v[k*2+0]=rho*u[k]; */
/*       dmom_v_adv_v[k*2+1]=2.0*rho*v[k]; */

/* #ifdef SCALAR_DIFFUSION */
/*      //u momentum diffusion tensor */
/*       mom_u_diff_ten[k*4+0] = mu; */
/*       mom_u_diff_ten[k*4+3] = mu; */

/*       //v momentum diffusion tensor */
/*       mom_v_diff_ten[k*4+0] = mu; */
/*       mom_v_diff_ten[k*4+3] = mu; */
/* #else */
/*       //u momentum diffusion tensor */
/*       mom_u_diff_ten[k*4+0] = 2.0*mu; */
/*       mom_u_diff_ten[k*4+3] = mu; */
/*       mom_uv_diff_ten[k*4+2]=mu; */

/*       //v momentum diffusion tensor */
/*       mom_v_diff_ten[k*4+0] = mu; */
/*       mom_v_diff_ten[k*4+3] = 2.0*mu; */
/*       mom_vu_diff_ten[k*4+1] = mu; */
/* #endif */

/*       //momentum sources */
/*       norm_n = sqrt(n[k*2+0]*n[k*2+0]+n[k*2+1]*n[k*2+1]); */
/*       if (norm_n < 1.0e-8) */
/*         norm_n = 1.0e-8; */
/*       mom_u_source[k] = -rho*g[0] - d_mu*sigma*kappa[k]*n[k*2+0]/(norm_n); */
/*       mom_v_source[k] = -rho*g[1] - d_mu*sigma*kappa[k]*n[k*2+1]/(norm_n); */
      
/*       //u momentum Hamiltonian (pressure) */

/*       mom_u_ham[k] = grad_p[k*2+0]; */
/*       dmom_u_ham_grad_p[k*2+0]=1.0; */
      
/*       //v momentum Hamiltonian (pressure) */
/*       mom_v_ham[k] = grad_p[k*2+1]; */
/*       dmom_v_ham_grad_p[k*21]=1.0; */

      //cek incomp form
      //u momentum accumulation
      mom_u_acc[k]=u[k];
      dmom_u_acc_u[k]=1.0;
      
      //v momentum accumulation
      mom_v_acc[k]=v[k];
      dmom_v_acc_v[k]=1.0;

      //mass advective flux
      mass_adv[k*2+0]=u[k];
      mass_adv[k*2+1]=v[k];
      
      dmass_adv_u[k*2+0]=1.0;
      dmass_adv_v[k*2+1]=1.0;

      //u momentum advective flux
      mom_u_adv[k*2+0]=u[k]*u[k];
      mom_u_adv[k*2+1]=u[k]*v[k];

      dmom_u_adv_u[k*2+0]=2.0*u[k];
      dmom_u_adv_u[k*2+1]=v[k];

      dmom_u_adv_v[k*2+1]=u[k];

      //v momentum advective_flux
      mom_v_adv[k*2+0]=v[k]*u[k];
      mom_v_adv[k*2+1]=v[k]*v[k];
            
      dmom_v_adv_u[k*2+0]=v[k];
      
      dmom_v_adv_v[k*2+0]=u[k];
      dmom_v_adv_v[k*2+1]=2.0*v[k];

#ifdef SCALAR_DIFFUSION
     //u momentum diffusion tensor
      mom_u_diff_ten[k*4+0] = nu;
      mom_u_diff_ten[k*4+3] = nu;

      //v momentum diffusion tensor
      mom_v_diff_ten[k*4+0] = nu;
      mom_v_diff_ten[k*4+3] = nu;
#else
      //u momentum diffusion tensor
      mom_u_diff_ten[k*4+0] = 2.0*nu;
      mom_u_diff_ten[k*4+3] = nu;
      mom_uv_diff_ten[k*4+2]=nu;

      //v momentum diffusion tensor
      mom_v_diff_ten[k*4+0] = nu;
      mom_v_diff_ten[k*4+3] = 2.0*nu;
      mom_vu_diff_ten[k*4+1] = nu;
#endif

      //momentum sources
/*       mom_u_source[k] = -g[0]; */
/*       mom_v_source[k] = -g[1]; */
      norm_n = sqrt(n[k*2+0]*n[k*2+0]+n[k*2+1]*n[k*2+1]);
      mom_u_source[k] = -g[0] - d_mu*sigma*kappa[k]*n[k*2+0]/(rho*(norm_n+1.0e-8));
      mom_v_source[k] = -g[1] - d_mu*sigma*kappa[k]*n[k*2+1]/(rho*(norm_n+1.0e-8));
      
      //u momentum Hamiltonian (pressure)

      mom_u_ham[k] = grad_p[k*2+0]/rho;
      dmom_u_ham_grad_p[k*2+0]=1.0/rho;
      
      //v momentum Hamiltonian (pressure)
      mom_v_ham[k] = grad_p[k*2+1]/rho;
      dmom_v_ham_grad_p[k*2+1]=1.0/rho;
    }
}


void TwophaseNavierStokes_ST_LS_SO_2D_Evaluate_sd(const int nPoints,
                                                  const double eps_rho,
                                                  const double eps_mu,
                                                  const double sigma,
                                                  const double rho_0,
                                                  const double nu_0,
                                                  const double rho_1,
                                                  const double nu_1,
                                                  const double* g,
                                                  const double* phi,
                                                  const double* n,
                                                  const double* kappa,
                                                  const double *p,
                                                  const double *grad_p,
                                                  const double *u,
                                                  const double *v,
                                                  double *mom_u_acc,
                                                  double *dmom_u_acc_u,
                                                  double *mom_v_acc,
                                                  double *dmom_v_acc_v,
                                                  double *mass_adv,
                                                  double *dmass_adv_u,
                                                  double *dmass_adv_v,
                                                  double *mom_u_adv,
                                                  double *dmom_u_adv_u,
                                                  double *dmom_u_adv_v,
                                                  double *mom_v_adv,
                                                  double *dmom_v_adv_u,
                                                  double *dmom_v_adv_v,
                                                  double *mom_u_diff_ten,
                                                  double *mom_v_diff_ten,
                                                  double *mom_uv_diff_ten,
                                                  double *mom_vu_diff_ten,
                                                  double *mom_u_source,
                                                  double *mom_v_source,
                                                  double *mom_u_ham,
                                                  double *dmom_u_ham_grad_p,
                                                  double *mom_v_ham,
                                                  double *dmom_v_ham_grad_p)
{
  int k;
  double rho,nu,mu,H_rho,d_rho,H_mu,d_mu,norm_n;
  for (k=0;k<nPoints;k++)
    {
      /** \todo optimize coefficients */
      H_rho = smoothedHeaviside(eps_rho,phi[k]);
      d_rho = smoothedDirac(eps_rho,phi[k]);
      H_mu = smoothedHeaviside(eps_mu,phi[k]);
      d_mu = smoothedDirac(eps_mu,phi[k]);

      rho = rho_0*(1.0-H_rho)+rho_1*H_rho;
      nu  = nu_0*(1.0-H_mu)+nu_1*H_mu;
      mu  = rho_0*nu_0*(1.0-H_mu)+rho_1*nu_1*H_mu;

      //u momentum accumulation
      mom_u_acc[k]=u[k];
      dmom_u_acc_u[k]=1.0;
      
      //v momentum accumulation
      mom_v_acc[k]=v[k];
      dmom_v_acc_v[k]=1.0;

      //mass advective flux
      mass_adv[k*2+0]=u[k];
      mass_adv[k*2+1]=v[k];
      
      dmass_adv_u[k*2+0]=1.0;
      dmass_adv_v[k*2+1]=1.0;

      //u momentum advective flux
      mom_u_adv[k*2+0]=u[k]*u[k];
      mom_u_adv[k*2+1]=u[k]*v[k];

      dmom_u_adv_u[k*2+0]=2.0*u[k];
      dmom_u_adv_u[k*2+1]=v[k];

      dmom_u_adv_v[k*2+1]=u[k];

      //v momentum advective_flux
      mom_v_adv[k*2+0]=v[k]*u[k];
      mom_v_adv[k*2+1]=v[k]*v[k];
            
      dmom_v_adv_u[k*2+0]=v[k];
      
      dmom_v_adv_v[k*2+0]=u[k];
      dmom_v_adv_v[k*2+1]=2.0*v[k];

      //u momentum diffusion tensor
      mom_u_diff_ten[k*2+0] = 2.0*nu;
      mom_u_diff_ten[k*2+1] = nu;
      mom_uv_diff_ten[k]=nu;

      //v momentum diffusion tensor
      mom_v_diff_ten[k*2+0] = nu;
      mom_v_diff_ten[k*2+1] = 2.0*nu;
      mom_vu_diff_ten[k] = nu;

      //momentum sources
      /* printf("rho = %.2f\n",rho); */
      /* printf("mom_u_source = %f\n",mom_u_source[k]); */
      /* printf("mom_v_source = %f\n",mom_u_source[k]); */
      /* printf("d_mu = %f\n" , d_mu); */
      /* printf("sigma = %f\n " , sigma); */
      
      norm_n = sqrt(n[k*2+0]*n[k*2+0]+n[k*2+1]*n[k*2+1]);
      mom_u_source[k] = -g[0] - d_mu*sigma*kappa[k]*n[k*2+0]/(rho*(norm_n+1.0e-8));
      mom_v_source[k] = -g[1] - d_mu*sigma*kappa[k]*n[k*2+1]/(rho*(norm_n+1.0e-8));
      //u momentum Hamiltonian (pressure)

      mom_u_ham[k] = grad_p[k*2+0]/rho;
      dmom_u_ham_grad_p[k*2+0]=1.0/rho;
      
      //v momentum Hamiltonian (pressure)
      mom_v_ham[k] = grad_p[k*2+1]/rho;
      dmom_v_ham_grad_p[k*2+1]=1.0/rho;

      /* //compressible form */
      /* //u momentum accumulation */
      /* mom_u_acc[k]=rho*u[k]; */
      /* dmom_u_acc_u[k]=rho; */
      
      /* //v momentum accumulation */
      /* mom_v_acc[k]=rho*v[k]; */
      /* dmom_v_acc_v[k]=rho; */

      /* //mass advective flux */
      /* mass_adv[k*2+0]=u[k]; */
      /* mass_adv[k*2+1]=v[k]; */
      
      /* dmass_adv_u[k*2+0]=1.0; */
      /* dmass_adv_v[k*2+1]=1.0; */

      /* //u momentum advective flux */
      /* mom_u_adv[k*2+0]=rho*u[k]*u[k]; */
      /* mom_u_adv[k*2+1]=rho*u[k]*v[k]; */

      /* dmom_u_adv_u[k*2+0]=rho*2.0*u[k]; */
      /* dmom_u_adv_u[k*2+1]=rho*v[k]; */

      /* dmom_u_adv_v[k*2+1]=rho*u[k]; */

      /* //v momentum advective_flux */
      /* mom_v_adv[k*2+0]=rho*v[k]*u[k]; */
      /* mom_v_adv[k*2+1]=rho*v[k]*v[k]; */
            
      /* dmom_v_adv_u[k*2+0]=rho*v[k]; */
      
      /* dmom_v_adv_v[k*2+0]=rho*u[k]; */
      /* dmom_v_adv_v[k*2+1]=rho*2.0*v[k]; */

      /* //u momentum diffusion tensor */
      /* mom_u_diff_ten[k*2+0] = 2.0*mu; */
      /* mom_u_diff_ten[k*2+1] = mu; */
      /* mom_uv_diff_ten[k]=mu; */

      /* //v momentum diffusion tensor */
      /* mom_v_diff_ten[k*2+0] = mu; */
      /* mom_v_diff_ten[k*2+1] = 2.0*mu; */
      /* mom_vu_diff_ten[k] = mu; */

      /* //momentum sources */
      /* norm_n = sqrt(n[k*2+0]*n[k*2+0]+n[k*2+1]*n[k*2+1]); */
      /* mom_u_source[k] = -rho*g[0] - d_mu*sigma*kappa[k]*n[k*2+0]/(norm_n+1.0e-8); */
      /* mom_v_source[k] = -rho*g[1] - d_mu*sigma*kappa[k]*n[k*2+1]/(norm_n+1.0e-8); */
      
      /* //u momentum Hamiltonian (pressure) */

      /* mom_u_ham[k] = grad_p[k*2+0]; */
      /* dmom_u_ham_grad_p[k*2+0]=1.0; */
      
      /* //v momentum Hamiltonian (pressure) */
      /* mom_v_ham[k] = grad_p[k*2+1]; */
      /* dmom_v_ham_grad_p[k*2+1]=1.0; */
    }
}

void ThreephaseNavierStokes_ST_LS_SO_2D_EvaluateOrig(const int nPoints,
						 const double eps_rho,
						 const double eps_mu,
						 const double sigma,
						 const double rho_0,
						 const double nu_0,
						 const double rho_1,
						 const double nu_1,
						 const double rho_s,
						 const double nu_s,
						 const double* g,
						 const double* phi,
						 const double* n,
						 const double* kappa,
						 const double* phi_s,
						 const double* n_s,
						 const double *p,
						 const double *grad_p,
						 const double *u,
						 const double *v,
						 double *mom_u_acc,
						 double *dmom_u_acc_u,
						 double *mom_v_acc,
						 double *dmom_v_acc_v,
						 double *mass_adv,
						 double *dmass_adv_u,
						 double *dmass_adv_v,
						 double *mom_u_adv,
						 double *dmom_u_adv_u,
						 double *dmom_u_adv_v,
						 double *mom_v_adv,
						 double *dmom_v_adv_u,
						 double *dmom_v_adv_v,
						 double *mom_u_diff_ten,
						 double *mom_v_diff_ten,
						 double *mom_uv_diff_ten,
						 double *mom_vu_diff_ten,
						 double *mom_u_source,
						 double *dmom_u_source_u,
						 double *dmom_u_source_v,
						 double *mom_v_source,
						 double *dmom_v_source_u,
						 double *dmom_v_source_v,
						 double *mom_u_ham,
						 double *dmom_u_ham_grad_p,
						 double *mom_v_ham,
						 double *dmom_v_ham_grad_p)
{
  int k;
  double rho,nu,mu,H_rho,d_rho,H_mu,d_mu,
    H_rho_s,d_rho_s,H_mu_s,d_mu_s,norm_n,norm_n_s;
  for (k=0;k<nPoints;k++)
    {
      /** \todo optimize coefficients */
      H_rho = smoothedHeaviside(eps_rho,phi[k]);
      d_rho = smoothedDirac(eps_rho,phi[k]);
      H_mu = smoothedHeaviside(eps_mu,phi[k]);
      d_mu = smoothedDirac(eps_mu,phi[k]);
      
      rho = rho_0*(1.0-H_rho)+rho_1*H_rho;
      nu  = nu_0*(1.0-H_mu)+nu_1*H_mu;
      mu  = rho_0*nu_0*(1.0-H_mu)+rho_1*nu_1*H_mu;
      rho = rho_0;
      nu  = nu_0;
      mu  = rho_0*nu_0;
      
/*       H_rho_s = smoothedHeaviside(eps_rho,phi_s[k]); */
/*       d_rho_s = smoothedDirac(eps_rho,phi_s[k]); */
/*       H_mu_s = smoothedHeaviside(eps_mu,phi_s[k]); */
/*       d_mu_s = smoothedDirac(eps_mu,phi_s[k]); */
      
/*       rho = rho_s*(1.0-H_rho_s)+rho*H_rho_s; */
/*       nu  = nu_s*(1.0-H_mu_s)+nu*H_mu_s; */
/*       mu  = rho_s*nu_s*(1.0-H_mu_s)+rho*nu*H_mu_s; */

      //u momentum accumulation
      mom_u_acc[k]=rho*u[k];
      dmom_u_acc_u[k]=rho;
      
      //v momentum accumulation
      mom_v_acc[k]=rho*v[k];
      dmom_v_acc_v[k]=rho;
      
      //mass advective flux
      mass_adv[k*2+0]=u[k];
      mass_adv[k*2+1]=v[k];
      
      dmass_adv_u[k*2+0]=1.0;
      dmass_adv_v[k*2+1]=1.0;
      
      //u momentum advective flux
      mom_u_adv[k*2+0]=rho*u[k]*u[k];
      mom_u_adv[k*2+1]=rho*u[k]*v[k];

      dmom_u_adv_u[k*2+0]=2.0*rho*u[k];
      dmom_u_adv_u[k*2+1]=rho*v[k];

      dmom_u_adv_v[k*2+1]=rho*u[k];
      
      //v momentum advective_flux
      mom_v_adv[k*2+0]=rho*v[k]*u[k];
      mom_v_adv[k*2+1]=rho*v[k]*v[k];
            
      dmom_v_adv_u[k*2+0]=rho*v[k];
      
      dmom_v_adv_v[k*2+0]=rho*u[k];
      dmom_v_adv_v[k*2+1]=2.0*rho*v[k];
      
      //u momentum diffusion tensor
      mom_u_diff_ten[k*4+0] = 2.0*mu;
      mom_u_diff_ten[k*4+3] = mu;
      mom_uv_diff_ten[k*4+2]=mu;

      //v momentum diffusion tensor
      mom_v_diff_ten[k*4+0] = mu;
      mom_v_diff_ten[k*4+3] = 2.0*mu;
      mom_vu_diff_ten[k*4+1] = mu;

      //momentum sources
      norm_n = sqrt(n[k*2+0]*n[k*2+0]+n[k*2+1]*n[k*2+1]);
      norm_n_s = sqrt(n_s[k*2+0]*n_s[k*2+0]+n_s[k*2+1]*n_s[k*2+1]);
      
/*       mom_u_source[k] = -rho*g[0] */
/*       	- d_mu*sigma*kappa[k]*n[k*2+0]/norm_n */
/*       	+rho*d_mu_s*(u[k]*u[k]*n_s[k*2+0] + u[k]*v[k]*n_s[k*2+1])/norm_n_s */
/*       	+rho*(1.0-H_mu_s)*u[k]; */
/*       dmom_u_source_u[k] = rho*d_mu_s*(2.0*u[k]*n_s[k*2+0] */
/* 				       + v[k]*n_s[k*2+1])/norm_n_s */
/*       	+rho*(1.0-H_mu_s); */
/*       dmom_u_source_v[k] = rho*d_mu_s*u[k]*n_s[k*2+1]/norm_n_s; */
      
/*       mom_v_source[k] = -rho*g[1] */
/*       	- d_mu*sigma*kappa[k]*n[k*2+1]/norm_n */
/*       	+rho*d_mu_s*(v[k]*u[k]*n_s[k*2+0] */
/* 		     + v[k]*v[k]*n_s[k*2+1])/norm_n_s */
/*       	+rho*(1.0-H_mu_s)*v[k]; */
/*       dmom_v_source_u[k] = rho*d_mu_s*v[k]*n_s[k*2+0]/norm_n_s; */
/*       dmom_v_source_v[k] = rho*d_mu_s*(u[k]*n_s[k*2+0] */
/* 				       + 2.0*v[k]*n_s[k*2+1])/norm_n_s */
/*       	+rho*(1.0-H_mu_s); */

      /* mom_u_source[k] = -rho*g[0] */
      /* 	- d_mu*sigma*kappa[k]*n[k*2+0]/norm_n */
      /* 	+rho*2.0*(1.0-H_mu_s)*u[k]; */
      /* dmom_u_source_u[k] = rho*2.0*(1.0-H_mu_s); */
      /* dmom_u_source_v[k] = 0.0; */
      
      /* mom_v_source[k] = -rho*g[1] */
      /* 	- d_mu*sigma*kappa[k]*n[k*2+1]/norm_n */
      /* 	+rho*2.0*(1.0-H_mu_s)*v[k]; */
      /* dmom_v_source_u[k] = 0.0; */
      /* dmom_v_source_v[k] = rho*2.0*(1.0-H_mu_s); */
      
      /* mom_u_source[k] = -rho*g[0] */
      /* 	- d_mu*sigma*kappa[k]*n[k*2+0]/norm_n */
      /* 	- rho*d_mu_s*(u[k]*u[k]*n_s[k*2+0] + u[k]*v[k]*n_s[k*2+1])/norm_n_s; */

      /* dmom_u_source_u[k] = -rho*d_mu_s*(2.0*u[k]*n_s[k*2+0] */
      /* 					+ v[k]*n_s[k*2+1])/norm_n_s; */
      
      /* dmom_u_source_v[k] = -rho*d_mu_s*u[k]*n_s[k*2+1]/norm_n_s; */
      
      /* mom_v_source[k] = -rho*g[1] */
      /* 	- d_mu*sigma*kappa[k]*n[k*2+1]/norm_n */
      /* 	- rho*d_mu_s*(v[k]*u[k]*n_s[k*2+0] */
      /* 		      + v[k]*v[k]*n_s[k*2+1])/norm_n_s; */
      
      /* dmom_v_source_u[k] = -rho*d_mu_s*v[k]*n_s[k*2+0]/norm_n_s; */
      /* dmom_v_source_v[k] = -rho*d_mu_s*(u[k]*n_s[k*2+0] */
      /* 					+ 2.0*v[k]*n_s[k*2+1])/norm_n_s; */
      
      /* mom_u_source[k] = -rho*g[0] +(1.0-H_mu_s)*u[k]; */
      /* dmom_u_source_u[k] = (1.0-H_mu_s); */
      /* dmom_u_source_v[k] = 0.0; */
      
      /* mom_v_source[k] = -rho*g[1] +(1.0-H_mu_s)*v[k]; */
      /* dmom_v_source_u[k] = 0.0; */
      /* dmom_v_source_v[k] = (1.0-H_mu_s); */
      mom_u_source[k] = -rho*g[0];
      dmom_u_source_u[k] = 0.0;
      dmom_u_source_v[k] = 0.0;
      
      mom_v_source[k] = -rho*g[1];
      dmom_v_source_u[k] = 0.0;
      dmom_v_source_v[k] = 0.0;
      
      //u momentum Hamiltonian (pressure)

      mom_u_ham[k] = grad_p[k*2+0];
      dmom_u_ham_grad_p[k*2+0]=1.0;
      
      //v momentum Hamiltonian (pressure)
      mom_v_ham[k] = grad_p[k*2+1];
      dmom_v_ham_grad_p[k*2+1]=1.0;
    }
}

void ThreephaseNavierStokes_ST_LS_SO_2D_Evaluate(const int nPoints,
						 const double boundaryPenaltyCoef,
						 const double volumePenaltyCoef,
						 const double eps_rho,
						 const double eps_mu,
						 const double sigma,
						 const double rho_0,
						 const double nu_0,
						 const double rho_1,
						 const double nu_1,
						 const double rho_s,
						 const double nu_s,
						 const double* g,
						 const double* phi,
						 const double* n,
						 const double* kappa,
						 const double* phi_s,
						 const double* n_s,
						 const double *p,
						 const double *grad_p,
						 const double *u,
						 const double *v,
						 double *mom_u_acc,
						 double *dmom_u_acc_u,
						 double *mom_v_acc,
						 double *dmom_v_acc_v,
						 double *mass_adv,
						 double *dmass_adv_u,
						 double *dmass_adv_v,
						 double *mom_u_adv,
						 double *dmom_u_adv_u,
						 double *dmom_u_adv_v,
						 double *mom_v_adv,
						 double *dmom_v_adv_u,
						 double *dmom_v_adv_v,
						 double *mom_u_diff_ten,
						 double *mom_v_diff_ten,
						 double *mom_uv_diff_ten,
						 double *mom_vu_diff_ten,
						 double *mom_u_source,
						 double *dmom_u_source_u,
						 double *dmom_u_source_v,
						 double *mom_v_source,
						 double *dmom_v_source_u,
						 double *dmom_v_source_v,
						 double *mom_u_ham,
						 double *dmom_u_ham_grad_p,
						 double *mom_v_ham,
						 double *dmom_v_ham_grad_p)
{
  int k;
  double rho,nu,mu,H_rho,d_rho,H_mu,d_mu,
    H_rho_s,d_rho_s,H_mu_s,d_mu_s,norm_n,norm_n_s,volumeFlux,sp;
  int quadraticPenalty = 0,sipgPenalty=1.0;
  sp=(double)(sipgPenalty);
  //sp=0.0;
  for (k=0;k<nPoints;k++)
    {
      /** \todo optimize coefficients */
      H_rho = smoothedHeaviside(eps_rho,phi[k]);
      d_rho = smoothedDirac(eps_rho,phi[k]);
      H_mu = smoothedHeaviside(eps_mu,phi[k]);
      d_mu = smoothedDirac(eps_mu,phi[k]);
      
      rho = rho_0*(1.0-H_rho)+rho_1*H_rho;
      nu  = nu_0*(1.0-H_mu)+nu_1*H_mu;
      mu  = rho_0*nu_0*(1.0-H_mu)+rho_1*nu_1*H_mu;
      
      H_rho_s = smoothedHeaviside(eps_rho,phi_s[k]);
      d_rho_s = smoothedDirac(eps_rho,phi_s[k]);
      H_mu_s = smoothedHeaviside(eps_mu,phi_s[k]);
      d_mu_s = smoothedDirac(eps_mu,phi_s[k]);

      norm_n = sqrt(n[k*2+0]*n[k*2+0]+n[k*2+1]*n[k*2+1]);
      norm_n_s = sqrt(n_s[k*2+0]*n_s[k*2+0]+n_s[k*2+1]*n_s[k*2+1]);

      //mwf start hacking here ... 
      //make coefficients just be fluid values first?      
      /* rho = rho_s*(1.0-H_rho_s)+rho*H_rho_s; */
      /* nu  = nu_s*(1.0-H_mu_s)+nu*H_mu_s; */
      /* mu  = rho_s*nu_s*(1.0-H_mu_s)+rho*nu*H_mu_s; */

      //u momentum accumulation
      mom_u_acc[k]=H_mu_s*rho*u[k];
      dmom_u_acc_u[k]=H_mu_s*rho;
      
      //v momentum accumulation
      mom_v_acc[k]=H_mu_s*rho*v[k];
      dmom_v_acc_v[k]=H_mu_s*rho;
      
      //mwf volume conservation holds in both solid and fluid 
      //mass advective flux
      mass_adv[k*2+0]=u[k];
      mass_adv[k*2+1]=v[k];
      
      dmass_adv_u[k*2+0]=1.0;
      dmass_adv_v[k*2+1]=1.0;
      
      //mwf continue killing acceleration in solid phase
      //u momentum advective flux
      mom_u_adv[k*2+0]=H_mu_s*rho*u[k]*u[k] - sp*(u[k]-0.0)*d_mu_s*n_s[k*2+0]/norm_n_s;
      mom_u_adv[k*2+1]=H_mu_s*rho*u[k]*v[k] - sp*(u[k]-0.0)*d_mu_s*n_s[k*2+1]/norm_n_s;

      dmom_u_adv_u[k*2+0]=2.0*H_mu_s*rho*u[k] - sp*d_mu_s*n_s[k*2+0]/norm_n_s;
      dmom_u_adv_u[k*2+1]=H_mu_s*rho*v[k] - sp*d_mu_s*n_s[k*2+1]/norm_n_s;

      dmom_u_adv_v[k*2+1]=H_mu_s*rho*u[k];
      
      //v momentum advective_flux
      mom_v_adv[k*2+0]=H_mu_s*rho*v[k]*u[k] - sp*(v[k]-0.0)*d_mu_s*n_s[k*2+0]/norm_n_s;
      mom_v_adv[k*2+1]=H_mu_s*rho*v[k]*v[k] - sp*(v[k]-0.0)*d_mu_s*n_s[k*2+1]/norm_n_s;
            
      dmom_v_adv_u[k*2+0]=H_mu_s*rho*v[k];
      
      dmom_v_adv_v[k*2+0]=H_mu_s*rho*u[k] - sp*d_mu_s*n_s[k*2+0]/norm_n_s;
      dmom_v_adv_v[k*2+1]=2.0*H_mu_s*rho*v[k] - sp*d_mu_s*n_s[k*2+1]/norm_n_s;
      
      //mwf no diffusion of momentum in solid phase either,
      //mwf also need to switch to Laplace form temporarily because of bc's?
      //u momentum diffusion tensor
      mom_u_diff_ten[k*4+0] = 2.0*H_mu_s*mu;
      mom_u_diff_ten[k*4+3] = H_mu_s*mu;
      mom_uv_diff_ten[k*4+2]=H_mu_s*mu;

      //v momentum diffusion tensor
      mom_v_diff_ten[k*4+0] = H_mu_s*mu;
      mom_v_diff_ten[k*4+3] = 2.0*H_mu_s*mu;
      mom_vu_diff_ten[k*4+1] = H_mu_s*mu;
/*       //u momentum diffusion tensor */
/*       mom_u_diff_ten[k*4+0] = H_mu_s*mu; */
/*       mom_u_diff_ten[k*4+3] = H_mu_s*mu; */
/*       mom_uv_diff_ten[k*4+2]= 0.0; */

/*       //v momentum diffusion tensor */
/*       mom_v_diff_ten[k*4+0] = H_mu_s*mu; */
/*       mom_v_diff_ten[k*4+3] = H_mu_s*mu; */
/*       mom_vu_diff_ten[k*4+1] = 0.0; */

      //momentum sources
      
      //mwf momentum in solid phase is just \grad p = 0 (i.e. pressure is constant) with penalties to enforce
      //mwf velocity is equal to input (v^s = 0 for now) on boundary and in solid region
      if (quadraticPenalty)
	{
	  mom_u_source[k] = -H_mu_s*rho*g[0]
	    - H_mu_s*d_mu*sigma*kappa[k]*n[k*2+0]/norm_n
	    +boundaryPenaltyCoef*rho*d_mu_s*(u[k] - 0.0)
	    +volumePenaltyCoef*rho*(1.0-H_mu_s)*(u[k] - 0.0)*(u[k] - 0.0);

	  dmom_u_source_u[k] = boundaryPenaltyCoef*rho*d_mu_s
	    +volumePenaltyCoef*rho*(1.0-H_mu_s)*2.0*(u[k]-0.0);
	  dmom_u_source_v[k] = 0.0;
	  
	  mom_v_source[k] = -H_mu_s*rho*g[1]
	    - H_mu_s*d_mu*sigma*kappa[k]*n[k*2+1]/norm_n
	    +boundaryPenaltyCoef*rho*d_mu_s*(v[k] - 0.0)
	    +volumePenaltyCoef*rho*(1.0-H_mu_s)*(v[k] - 0.0)*(v[k] - 0.0);

	  dmom_v_source_u[k] = 0.0;
	  dmom_v_source_v[k] = boundaryPenaltyCoef*rho*d_mu_s
	    +volumePenaltyCoef*rho*(1.0-H_mu_s)*2.0*(v[k] - 0.0);
	}
      else if (sipgPenalty)
	{
	  mom_u_source[k] = -H_mu_s*rho*g[0]
	    - H_mu_s*d_mu*sigma*kappa[k]*n[k*2+0]/norm_n
	    +volumePenaltyCoef*rho*(1.0-H_mu_s)*(u[k] - 0.0);
	  
	  dmom_u_source_u[k] = volumePenaltyCoef*rho*(1.0-H_mu_s);
	  dmom_u_source_v[k] = 0.0;
	  
	  mom_v_source[k] = -H_mu_s*rho*g[1]
	    - H_mu_s*d_mu*sigma*kappa[k]*n[k*2+1]/norm_n
	    +volumePenaltyCoef*rho*(1.0-H_mu_s)*(v[k] - 0.0);
	  
	  dmom_v_source_u[k] = 0.0;
	  dmom_v_source_v[k] = volumePenaltyCoef*rho*(1.0-H_mu_s);

	  volumeFlux = u[k]*n_s[k*2+0] + v[k]*n_s[k*2+1];
	  if (volumeFlux < 0.0)
	    {
	      mom_u_source[k] += rho*d_mu_s*u[k]*(u[k]*n_s[k*2+0] + v[k]*n_s[k*2+1])/norm_n_s;
	      
	      dmom_u_source_u[k] += rho*d_mu_s*(2.0*u[k]*n_s[k*2+0]
						+ v[k]*n_s[k*2+1])/norm_n_s;
	      
	      dmom_u_source_v[k] = rho*d_mu_s*u[k]*n_s[k*2+1]/norm_n_s;
	      
	      mom_v_source[k] += rho*d_mu_s*(v[k]*u[k]*n_s[k*2+0]
					     + v[k]*v[k]*n_s[k*2+1])/norm_n_s;
	      
	      dmom_v_source_u[k] += rho*d_mu_s*v[k]*n_s[k*2+0]/norm_n_s;
	      
	      dmom_v_source_v[k] += rho*d_mu_s*(u[k]*n_s[k*2+0]
						+ 2.0*v[k]*n_s[k*2+1])/norm_n_s;
	    }
	}
      else
	{
	  mom_u_source[k] = -H_mu_s*rho*g[0]
	    - H_mu_s*d_mu*sigma*kappa[k]*n[k*2+0]/norm_n
	    +boundaryPenaltyCoef*rho*d_mu_s*(u[k] - 0.0)
	    +volumePenaltyCoef*rho*(1.0-H_mu_s)*(u[k] - 0.0);
	  
	  dmom_u_source_u[k] = boundaryPenaltyCoef*rho*d_mu_s
	    +volumePenaltyCoef*rho*(1.0-H_mu_s);
	  dmom_u_source_v[k] = 0.0;
	  
	  mom_v_source[k] = -H_mu_s*rho*g[1]
	    - H_mu_s*d_mu*sigma*kappa[k]*n[k*2+1]/norm_n
	    +boundaryPenaltyCoef*rho*d_mu_s*(v[k] - 0.0)
	    +volumePenaltyCoef*rho*(1.0-H_mu_s)*(v[k] - 0.0);
	  
	  dmom_v_source_u[k] = 0.0;
	  dmom_v_source_v[k] = boundaryPenaltyCoef*rho*d_mu_s
	    +volumePenaltyCoef*rho*(1.0-H_mu_s);

	}
      //mwf include grad_p term in solid phase to enforce p = constant (or integral around boundary is zero?)      
      //u momentum Hamiltonian (pressure)
      mom_u_ham[k] = grad_p[k*2+0];
      dmom_u_ham_grad_p[k*2+0]=1.0;
      
      //v momentum Hamiltonian (pressure)
      mom_v_ham[k] = grad_p[k*2+1];
      dmom_v_ham_grad_p[k*2+1]=1.0;
    }
}

void TwophaseNavierStokes_ST_LS_SO_3D_Evaluate(const int nPoints,
                                               const double eps_rho,
                                               const double eps_mu,
                                               const double sigma,
                                               const double rho_0,
                                               const double nu_0,
                                               const double rho_1,
                                               const double nu_1,
                                               const double* g,
                                               const double* phi,
                                               const double* n,
                                               const double* kappa,
                                               const double *p,
                                               const double *grad_p,
                                               const double *u,
                                               const double *v,
                                               const double *w,
                                               double *mom_u_acc,
                                               double *dmom_u_acc_u,
                                               double *mom_v_acc,
                                               double *dmom_v_acc_v,
                                               double *mom_w_acc,
                                               double *dmom_w_acc_w,
                                               double *mass_adv,
                                               double *dmass_adv_u,
                                               double *dmass_adv_v,
                                               double *dmass_adv_w,
                                               double *mom_u_adv,
                                               double *dmom_u_adv_u,
                                               double *dmom_u_adv_v,
                                               double *dmom_u_adv_w,
                                               double *mom_v_adv,
                                               double *dmom_v_adv_u,
                                               double *dmom_v_adv_v,
                                               double *dmom_v_adv_w,
                                               double *mom_w_adv,
                                               double *dmom_w_adv_u,
                                               double *dmom_w_adv_v,
                                               double *dmom_w_adv_w,
                                               double *mom_u_diff_ten,
                                               double *mom_v_diff_ten,
                                               double *mom_w_diff_ten,
                                               double *mom_uv_diff_ten,
                                               double *mom_uw_diff_ten,
                                               double *mom_vu_diff_ten,
                                               double *mom_vw_diff_ten,
                                               double *mom_wu_diff_ten,
                                               double *mom_wv_diff_ten,
                                               double *mom_u_source,
                                               double *mom_v_source,
                                               double *mom_w_source,
                                               double *mom_u_ham,
                                               double *dmom_u_ham_grad_p,
                                               double *mom_v_ham,
                                               double *dmom_v_ham_grad_p,
                                               double *mom_w_ham,
                                               double *dmom_w_ham_grad_p)
{
  int k;
  double rho,nu,mu,H_rho,d_rho,H_mu,d_mu,norm_n;
  for (k=0;k<nPoints;k++)
    {
      /** \todo optimize coefficients */
      /*H = smoothedHeaviside(eps,phi[k]);*/
      H_rho = smoothedHeaviside(eps_rho,phi[k]);
      d_rho = smoothedDirac(eps_rho,phi[k]);
      H_mu = smoothedHeaviside(eps_mu,phi[k]);
      d_mu = smoothedDirac(eps_mu,phi[k]);

      rho = rho_0*(1.0-H_rho)+rho_1*H_rho;
      nu  = nu_0*(1.0-H_mu)+nu_1*H_mu;
      mu  = rho_0*nu_0*(1.0-H_mu)+rho_1*nu_1*H_mu;

/*       //u momentum accumulation */
/*       mom_u_acc[k]=rho*u[k]; */
/*       dmom_u_acc_u[k]=rho; */
      
/*       //v momentum accumulation */
/*       mom_v_acc[k]=rho*v[k]; */
/*       dmom_v_acc_v[k]=rho; */

/*       //w momentum accumulation */
/*       mom_w_acc[k]=rho*w[k]; */
/*       dmom_w_acc_w[k]=rho; */


/*      //mass advective flux */
/*       mass_adv[k*3+0]=u[k]; */
/*       mass_adv[k*3+1]=v[k]; */
/*       mass_adv[k*3+2]=w[k]; */
      
/*       dmass_adv_u[k*3+0]=1.0; */
/*       dmass_adv_v[k*3+1]=1.0; */
/*       dmass_adv_w[k*3+2]=1.0; */

/*       //u momentum advective flux */
/*       mom_u_adv[k*3+0]=rho*u[k]*u[k]; */
/*       mom_u_adv[k*3+1]=rho*u[k]*v[k]; */
/*       mom_u_adv[k*3+2]=rho*u[k]*w[k]; */

/*       dmom_u_adv_u[k*3+0]=2.0*rho*u[k]; */
/*       dmom_u_adv_u[k*3+1]=rho*v[k]; */
/*       dmom_u_adv_u[k*3+2]=rho*w[k]; */

/*       dmom_u_adv_v[k*3+1]=rho*u[k]; */
      
/*       dmom_u_adv_w[k*3+2]=rho*u[k]; */

/*       //v momentum advective_flux */
/*       mom_v_adv[k*3+0]=rho*v[k]*u[k]; */
/*       mom_v_adv[k*3+1]=rho*v[k]*v[k]; */
/*       mom_v_adv[k*3+2]=rho*v[k]*w[k]; */
            
/*       dmom_v_adv_u[k*3+0]=rho*v[k]; */
      
/*       dmom_v_adv_w[k*3+2]=rho*v[k]; */
      
/*       dmom_v_adv_v[k*3+0]=rho*u[k]; */
/*       dmom_v_adv_v[k*3+1]=2.0*rho*v[k]; */
/*       dmom_v_adv_v[k*3+2]=rho*w[k]; */

/*       //w momentum advective_flux */
/*       mom_w_adv[k*3+0]=rho*w[k]*u[k]; */
/*       mom_w_adv[k*3+1]=rho*w[k]*v[k]; */
/*       mom_w_adv[k*3+2]=rho*w[k]*w[k]; */
            
/*       dmom_w_adv_u[k*3+0]=rho*w[k]; */
      
/*       dmom_w_adv_v[k*3+1]=rho*w[k]; */
      
/*       dmom_w_adv_w[k*3+0]=rho*u[k]; */
/*       dmom_w_adv_w[k*3+1]=rho*v[k]; */
/*       dmom_w_adv_w[k*3+2]=2.0*rho*w[k]; */

/*       //u momentum diffusion tensor */
/*       mom_u_diff_ten[k*9+0] = 2.0*mu; */
/*       mom_u_diff_ten[k*9+4] = mu; */
/*       mom_u_diff_ten[k*9+8] = mu; */

/*       mom_uv_diff_ten[k*9+3]=mu; */

/*       mom_uw_diff_ten[k*9+6]=mu; */

/*       //v momentum diffusion tensor */
/*       mom_v_diff_ten[k*9+0] = mu; */
/*       mom_v_diff_ten[k*9+4] = 2.0*mu; */
/*       mom_v_diff_ten[k*9+8] = mu; */

/*       mom_vu_diff_ten[k*9+1]=mu; */

/*       mom_vw_diff_ten[k*9+7]=mu; */

/*       //w momentum diffusion tensor */
/*       mom_w_diff_ten[k*9+0] = mu; */
/*       mom_w_diff_ten[k*9+4] = mu; */
/*       mom_w_diff_ten[k*9+8] = 2.0*mu; */

/*       mom_wu_diff_ten[k*9+2]=mu; */

/*       mom_wv_diff_ten[k*9+5]=mu; */

/*       //momentum sources */
/*       norm_n = sqrt(n[k*3+0]*n[k*3+0]+n[k*3+1]*n[k*3+1]+n[k*3+2]*n[k*3+2]); */
/*       mom_u_source[k] = -rho*g[0] - d_mu*sigma*kappa[k]*n[k*3+0]/(norm_n); */
/*       mom_v_source[k] = -rho*g[1] - d_mu*sigma*kappa[k]*n[k*3+1]/(norm_n); */
/*       mom_w_source[k] = -rho*g[2] - d_mu*sigma*kappa[k]*n[k*3+2]/(norm_n); */


/*       //u momentum Hamiltonian (pressure) */
/*       mom_u_ham[k] = grad_p[k*3+0]; */
/*       dmom_u_ham_grad_p[k*3+0]=1.0; */
      
/*       //v momentum Hamiltonian (pressure) */
/*       mom_v_ham[k] = grad_p[k*3+1]; */
/*       dmom_v_ham_grad_p[k*3+1]=1.0; */

/*       //w momentum Hamiltonian (pressure) */
/*       mom_w_ham[k] = grad_p[k*3+2]; */
/*       dmom_w_ham_grad_p[k*3+2]=1.0; */

      //cek "incompressible" form
      //u momentum accumulation
      mom_u_acc[k]=u[k];
      dmom_u_acc_u[k]=1.0;
      
      //v momentum accumulation
      mom_v_acc[k]=v[k];
      dmom_v_acc_v[k]=1.0;

      //w momentum accumulation
      mom_w_acc[k]=w[k];
      dmom_w_acc_w[k]=1.0;


     //mass advective flux
      mass_adv[k*3+0]=u[k];
      mass_adv[k*3+1]=v[k];
      mass_adv[k*3+2]=w[k];
      
      dmass_adv_u[k*3+0]=1.0;
      dmass_adv_v[k*3+1]=1.0;
      dmass_adv_w[k*3+2]=1.0;

      //u momentum advective flux
      mom_u_adv[k*3+0]=u[k]*u[k];
      mom_u_adv[k*3+1]=u[k]*v[k];
      mom_u_adv[k*3+2]=u[k]*w[k];

      dmom_u_adv_u[k*3+0]=2.0*u[k];
      dmom_u_adv_u[k*3+1]=v[k];
      dmom_u_adv_u[k*3+2]=w[k];

      dmom_u_adv_v[k*3+1]=u[k];
      
      dmom_u_adv_w[k*3+2]=u[k];

      //v momentum advective_flux
      mom_v_adv[k*3+0]=v[k]*u[k];
      mom_v_adv[k*3+1]=v[k]*v[k];
      mom_v_adv[k*3+2]=v[k]*w[k];
            
      dmom_v_adv_u[k*3+0]=v[k];
      
      dmom_v_adv_w[k*3+2]=v[k];
      
      dmom_v_adv_v[k*3+0]=u[k];
      dmom_v_adv_v[k*3+1]=2.0*v[k];
      dmom_v_adv_v[k*3+2]=w[k];

      //w momentum advective_flux
      mom_w_adv[k*3+0]=w[k]*u[k];
      mom_w_adv[k*3+1]=w[k]*v[k];
      mom_w_adv[k*3+2]=w[k]*w[k];
            
      dmom_w_adv_u[k*3+0]=w[k];
      
      dmom_w_adv_v[k*3+1]=w[k];
      
      dmom_w_adv_w[k*3+0]=u[k];
      dmom_w_adv_w[k*3+1]=v[k];
      dmom_w_adv_w[k*3+2]=2.0*w[k];

      //u momentum diffusion tensor
      mom_u_diff_ten[k*9+0] = 2.0*nu;
      mom_u_diff_ten[k*9+4] = nu;
      mom_u_diff_ten[k*9+8] = nu;

      mom_uv_diff_ten[k*9+3]=nu;

      mom_uw_diff_ten[k*9+6]=nu;

      //v momentum diffusion tensor
      mom_v_diff_ten[k*9+0] = nu;
      mom_v_diff_ten[k*9+4] = 2.0*nu;
      mom_v_diff_ten[k*9+8] = nu;

      mom_vu_diff_ten[k*9+1]=nu;

      mom_vw_diff_ten[k*9+7]=nu;

      //w momentum diffusion tensor
      mom_w_diff_ten[k*9+0] = nu;
      mom_w_diff_ten[k*9+4] = nu;
      mom_w_diff_ten[k*9+8] = 2.0*nu;

      mom_wu_diff_ten[k*9+2]=nu;

      mom_wv_diff_ten[k*9+5]=nu;

      //momentum sources
      norm_n = sqrt(n[k*3+0]*n[k*3+0]+n[k*3+1]*n[k*3+1]+n[k*3+2]*n[k*3+2]);
      mom_u_source[k] = -g[0] - d_mu*sigma*kappa[k]*n[k*3+0]/(rho*(norm_n+1.0e-8));
      mom_v_source[k] = -g[1] - d_mu*sigma*kappa[k]*n[k*3+1]/(rho*(norm_n+1.0e-8));
      mom_w_source[k] = -g[2] - d_mu*sigma*kappa[k]*n[k*3+2]/(rho*(norm_n+1.0e-8));


      //u momentum Hamiltonian (pressure)
      mom_u_ham[k] = grad_p[k*3+0]/rho;
      dmom_u_ham_grad_p[k*3+0]=1.0/rho;
      
      //v momentum Hamiltonian (pressure)
      mom_v_ham[k] = grad_p[k*3+1]/rho;
      dmom_v_ham_grad_p[k*3+1]=1.0/rho;

      //w momentum Hamiltonian (pressure)
      mom_w_ham[k] = grad_p[k*3+2]/rho;
      dmom_w_ham_grad_p[k*3+2]=1.0/rho;
    }
}
void TwophaseNavierStokes_ST_LS_SO_3D_Evaluate_sd(const int nPoints,
                                                  const double eps_rho,
                                                  const double eps_mu,
                                                  const double sigma,
                                                  const double rho_0,
                                                  const double nu_0,
                                                  const double rho_1,
                                                  const double nu_1,
                                                  const double* g,
                                                  const double* phi,
                                                  const double* n,
                                                  const double* kappa,
                                                  const double *p,
                                                  const double *grad_p,
                                                  const double *u,
                                                  const double *v,
                                                  const double *w,
                                                  double *mom_u_acc,
                                                  double *dmom_u_acc_u,
                                                  double *mom_v_acc,
                                                  double *dmom_v_acc_v,
                                                  double *mom_w_acc,
                                                  double *dmom_w_acc_w,
                                                  double *mass_adv,
                                                  double *dmass_adv_u,
                                                  double *dmass_adv_v,
                                                  double *dmass_adv_w,
                                                  double *mom_u_adv,
                                                  double *dmom_u_adv_u,
                                                  double *dmom_u_adv_v,
                                                  double *dmom_u_adv_w,
                                                  double *mom_v_adv,
                                                  double *dmom_v_adv_u,
                                                  double *dmom_v_adv_v,
                                                  double *dmom_v_adv_w,
                                                  double *mom_w_adv,
                                                  double *dmom_w_adv_u,
                                                  double *dmom_w_adv_v,
                                                  double *dmom_w_adv_w,
                                                  double *mom_u_diff_ten,
                                                  double *mom_v_diff_ten,
                                                  double *mom_w_diff_ten,
                                                  double *mom_uv_diff_ten,
                                                  double *mom_uw_diff_ten,
                                                  double *mom_vu_diff_ten,
                                                  double *mom_vw_diff_ten,
                                                  double *mom_wu_diff_ten,
                                                  double *mom_wv_diff_ten,
                                                  double *mom_u_source,
                                                  double *mom_v_source,
                                                  double *mom_w_source,
                                                  double *mom_u_ham,
                                                  double *dmom_u_ham_grad_p,
                                                  double *mom_v_ham,
                                                  double *dmom_v_ham_grad_p,
                                                  double *mom_w_ham,
                                                  double *dmom_w_ham_grad_p)
{
  int k;
  double rho,nu,mu,H_rho,d_rho,H_mu,d_mu,norm_n;
  for (k=0;k<nPoints;k++)
    {
      /** \todo optimize coefficients */
      /*H = smoothedHeaviside(eps,phi[k]);*/
      H_rho = smoothedHeaviside(eps_rho,phi[k]);
      d_rho = smoothedDirac(eps_rho,phi[k]);
      H_mu = smoothedHeaviside(eps_mu,phi[k]);
      d_mu = smoothedDirac(eps_mu,phi[k]);

      rho = rho_0*(1.0-H_rho)+rho_1*H_rho;
      nu  = nu_0*(1.0-H_mu)+nu_1*H_mu;
      mu  = rho_0*nu_0*(1.0-H_mu)+rho_1*nu_1*H_mu;

      //u momentum accumulation
      mom_u_acc[k]=u[k];
      dmom_u_acc_u[k]=1.0;
      
      //v momentum accumulation
      mom_v_acc[k]=v[k];
      dmom_v_acc_v[k]=1.0;

      //w momentum accumulation
      mom_w_acc[k]=w[k];
      dmom_w_acc_w[k]=1.0;


     //mass advective flux
      mass_adv[k*3+0]=u[k];
      mass_adv[k*3+1]=v[k];
      mass_adv[k*3+2]=w[k];
      
      dmass_adv_u[k*3+0]=1.0;
      dmass_adv_v[k*3+1]=1.0;
      dmass_adv_w[k*3+2]=1.0;

      //u momentum advective flux
      mom_u_adv[k*3+0]=u[k]*u[k];
      mom_u_adv[k*3+1]=u[k]*v[k];
      mom_u_adv[k*3+2]=u[k]*w[k];

      dmom_u_adv_u[k*3+0]=2.0*u[k];
      dmom_u_adv_u[k*3+1]=v[k];
      dmom_u_adv_u[k*3+2]=w[k];

      dmom_u_adv_v[k*3+1]=u[k];
      
      dmom_u_adv_w[k*3+2]=u[k];

      //v momentum advective_flux
      mom_v_adv[k*3+0]=v[k]*u[k];
      mom_v_adv[k*3+1]=v[k]*v[k];
      mom_v_adv[k*3+2]=v[k]*w[k];
            
      dmom_v_adv_u[k*3+0]=v[k];
      
      dmom_v_adv_w[k*3+2]=v[k];
      
      dmom_v_adv_v[k*3+0]=u[k];
      dmom_v_adv_v[k*3+1]=2.0*v[k];
      dmom_v_adv_v[k*3+2]=w[k];

      //w momentum advective_flux
      mom_w_adv[k*3+0]=w[k]*u[k];
      mom_w_adv[k*3+1]=w[k]*v[k];
      mom_w_adv[k*3+2]=w[k]*w[k];
            
      dmom_w_adv_u[k*3+0]=w[k];
      
      dmom_w_adv_v[k*3+1]=w[k];
      
      dmom_w_adv_w[k*3+0]=u[k];
      dmom_w_adv_w[k*3+1]=v[k];
      dmom_w_adv_w[k*3+2]=2.0*w[k];

      //u momentum diffusion tensor
      mom_u_diff_ten[k*3+0] = 2.0*nu;
      mom_u_diff_ten[k*3+1] = nu;
      mom_u_diff_ten[k*3+2] = nu;

      mom_uv_diff_ten[k]=nu;

      mom_uw_diff_ten[k]=nu;

      //v momentum diffusion tensor
      mom_v_diff_ten[k*3+0] = nu;
      mom_v_diff_ten[k*3+1] = 2.0*nu;
      mom_v_diff_ten[k*3+2] = nu;

      mom_vu_diff_ten[k]=nu;

      mom_vw_diff_ten[k]=nu;

      //w momentum diffusion tensor
      mom_w_diff_ten[k*3+0] = nu;
      mom_w_diff_ten[k*3+1] = nu;
      mom_w_diff_ten[k*3+2] = 2.0*nu;

      mom_wu_diff_ten[k]=nu;

      mom_wv_diff_ten[k]=nu;

      //momentum sources
      norm_n = sqrt(n[k*3+0]*n[k*3+0]+n[k*3+1]*n[k*3+1]+n[k*3+2]*n[k*3+2]);
      mom_u_source[k] = -g[0] - d_mu*sigma*kappa[k]*n[k*3+0]/(rho*(norm_n+1.0e-8));
      mom_v_source[k] = -g[1] - d_mu*sigma*kappa[k]*n[k*3+1]/(rho*(norm_n+1.0e-8));
      mom_w_source[k] = -g[2] - d_mu*sigma*kappa[k]*n[k*3+2]/(rho*(norm_n+1.0e-8));


      //u momentum Hamiltonian (pressure)
      mom_u_ham[k] = grad_p[k*3+0]/rho;
      dmom_u_ham_grad_p[k*3+0]=1.0/rho;
      
      //v momentum Hamiltonian (pressure)
      mom_v_ham[k] = grad_p[k*3+1]/rho;
      dmom_v_ham_grad_p[k*3+1]=1.0/rho;

      //w momentum Hamiltonian (pressure)
      mom_w_ham[k] = grad_p[k*3+2]/rho;
      dmom_w_ham_grad_p[k*3+2]=1.0/rho;
    }
}
void ThreephaseNavierStokes_ST_LS_SO_3D_Evaluate(const int nPoints,
						 const double boundaryPenaltyCoef,
						 const double volumePenaltyCoef,
						 const double eps_rho,
						 const double eps_mu,
						 const double sigma,
						 const double rho_0,
						 const double nu_0,
						 const double rho_1,
						 const double nu_1,
						 const double rho_s,
						 const double nu_s,
						 const double* g,
						 const double* phi,
						 const double* n,
						 const double* kappa,
						 const double* phi_s,
						 const double* n_s,
						 const double *p,
						 const double *grad_p,
						 const double *u,
						 const double *v,
						 const double *w,
						 double *mom_u_acc,
						 double *dmom_u_acc_u,
						 double *mom_v_acc,
						 double *dmom_v_acc_v,
						 double *mom_w_acc,
						 double *dmom_w_acc_w,
						 double *mass_adv,
						 double *dmass_adv_u,
						 double *dmass_adv_v,
						 double *dmass_adv_w,
						 double *mom_u_adv,
						 double *dmom_u_adv_u,
						 double *dmom_u_adv_v,
						 double *dmom_u_adv_w,
						 double *mom_v_adv,
						 double *dmom_v_adv_u,
						 double *dmom_v_adv_v,
						 double *dmom_v_adv_w,
						 double *mom_w_adv,
						 double *dmom_w_adv_u,
						 double *dmom_w_adv_v,
						 double *dmom_w_adv_w,
						 double *mom_u_diff_ten,
						 double *mom_v_diff_ten,
						 double *mom_w_diff_ten,
						 double *mom_uv_diff_ten,
						 double *mom_uw_diff_ten,
						 double *mom_vu_diff_ten,
						 double *mom_vw_diff_ten,
						 double *mom_wu_diff_ten,
						 double *mom_wv_diff_ten,
						 double *mom_u_source,
						 double *dmom_u_source_u,
						 double *dmom_u_source_v,
						 double *dmom_u_source_w,
						 double *mom_v_source,
						 double *dmom_v_source_u,
						 double *dmom_v_source_v,
						 double *dmom_v_source_w,
						 double *mom_w_source,
						 double *dmom_w_source_u,
						 double *dmom_w_source_v,
						 double *dmom_w_source_w,
						 double *mom_u_ham,
						 double *dmom_u_ham_grad_p,
						 double *mom_v_ham,
						 double *dmom_v_ham_grad_p,
						 double *mom_w_ham,
						 double *dmom_w_ham_grad_p)
{
  int k;
  double rho,nu,mu,H_rho,d_rho,H_mu,d_mu,
    H_rho_s,d_rho_s,H_mu_s,d_mu_s,norm_n,norm_n_s;

  for (k=0;k<nPoints;k++)
    {
      /** \todo optimize coefficients */
      /*H = smoothedHeaviside(eps,phi[k]);*/
      H_rho = smoothedHeaviside(eps_rho,phi[k]);
      d_rho = smoothedDirac(eps_rho,phi[k]);
      H_mu = smoothedHeaviside(eps_mu,phi[k]);
      d_mu = smoothedDirac(eps_mu,phi[k]);

      rho = rho_0*(1.0-H_rho)+rho_1*H_rho;
      nu  = nu_0*(1.0-H_mu)+nu_1*H_mu;
      mu  = rho_0*nu_0*(1.0-H_mu)+rho_1*nu_1*H_mu;

      H_rho_s = smoothedHeaviside(eps_rho,phi_s[k]);
      d_rho_s = smoothedDirac(eps_rho,phi_s[k]);
      H_mu_s = smoothedHeaviside(eps_mu,phi_s[k]);
      d_mu_s = smoothedDirac(eps_mu,phi_s[k]);

      rho = rho_s*(1.0-H_rho_s)+rho*H_rho_s;
      nu  = nu_s*(1.0-H_mu_s)+nu*H_mu_s;
      mu  = rho_s*nu_s*(1.0-H_mu_s)+rho*nu*H_mu_s;

      //u momentum accumulation
      mom_u_acc[k]=H_mu_s*rho*u[k];
      dmom_u_acc_u[k]=H_mu_s*rho;
      
      //v momentum accumulation
      mom_v_acc[k]=H_mu_s*rho*v[k];
      dmom_v_acc_v[k]=H_mu_s*rho;

      //w momentum accumulation
      mom_w_acc[k]=H_mu_s*rho*w[k];
      dmom_w_acc_w[k]=H_mu_s*rho;


     //mass advective flux
      mass_adv[k*3+0]=u[k];
      mass_adv[k*3+1]=v[k];
      mass_adv[k*3+2]=w[k];
      
      dmass_adv_u[k*3+0]=1.0;
      dmass_adv_v[k*3+1]=1.0;
      dmass_adv_w[k*3+2]=1.0;

      //u momentum advective flux
      mom_u_adv[k*3+0]=H_mu_s*rho*u[k]*u[k];
      mom_u_adv[k*3+1]=H_mu_s*rho*u[k]*v[k];
      mom_u_adv[k*3+2]=H_mu_s*rho*u[k]*w[k];

      dmom_u_adv_u[k*3+0]=2.0*H_mu_s*rho*u[k];
      dmom_u_adv_u[k*3+1]=H_mu_s*rho*v[k];
      dmom_u_adv_u[k*3+2]=H_mu_s*rho*w[k];

      dmom_u_adv_v[k*3+1]=H_mu_s*rho*u[k];
      
      dmom_u_adv_w[k*3+2]=H_mu_s*rho*u[k];

      //v momentum advective_flux
      mom_v_adv[k*3+0]=H_mu_s*rho*v[k]*u[k];
      mom_v_adv[k*3+1]=H_mu_s*rho*v[k]*v[k];
      mom_v_adv[k*3+2]=H_mu_s*rho*v[k]*w[k];
            
      dmom_v_adv_u[k*3+0]=H_mu_s*rho*v[k];
      
      dmom_v_adv_w[k*3+2]=H_mu_s*rho*v[k];
      
      dmom_v_adv_v[k*3+0]=H_mu_s*rho*u[k];
      dmom_v_adv_v[k*3+1]=2.0*H_mu_s*rho*v[k];
      dmom_v_adv_v[k*3+2]=H_mu_s*rho*w[k];

      //w momentum advective_flux
      mom_w_adv[k*3+0]=H_mu_s*rho*w[k]*u[k];
      mom_w_adv[k*3+1]=H_mu_s*rho*w[k]*v[k];
      mom_w_adv[k*3+2]=H_mu_s*rho*w[k]*w[k];
            
      dmom_w_adv_u[k*3+0]=H_mu_s*rho*w[k];
      
      dmom_w_adv_v[k*3+1]=H_mu_s*rho*w[k];
      
      dmom_w_adv_w[k*3+0]=H_mu_s*rho*u[k];
      dmom_w_adv_w[k*3+1]=H_mu_s*rho*v[k];
      dmom_w_adv_w[k*3+2]=2.0*H_mu_s*rho*w[k];

      //u momentum diffusion tensor
      mom_u_diff_ten[k*9+0] = 2.0*H_mu_s*mu;
      mom_u_diff_ten[k*9+4] = H_mu_s*mu;
      mom_u_diff_ten[k*9+8] = H_mu_s*mu;

      mom_uv_diff_ten[k*9+3]=H_mu_s*mu;

      mom_uw_diff_ten[k*9+6]=H_mu_s*mu;

      //v momentum diffusion tensor
      mom_v_diff_ten[k*9+0] = H_mu_s*mu;
      mom_v_diff_ten[k*9+4] = 2.0*H_mu_s*mu;
      mom_v_diff_ten[k*9+8] = H_mu_s*mu;

      mom_vu_diff_ten[k*9+1]=H_mu_s*mu;

      mom_vw_diff_ten[k*9+7]=H_mu_s*mu;

      //w momentum diffusion tensor
      mom_w_diff_ten[k*9+0] = H_mu_s*mu;
      mom_w_diff_ten[k*9+4] = H_mu_s*mu;
      mom_w_diff_ten[k*9+8] = 2.0*H_mu_s*mu;

      mom_wu_diff_ten[k*9+2]=H_mu_s*mu;

      mom_wv_diff_ten[k*9+5]=H_mu_s*mu;

      //momentum sources
      norm_n = sqrt(n[k*3+0]*n[k*3+0]+n[k*3+1]*n[k*3+1]+n[k*3+2]*n[k*3+2]);
      mom_u_source[k] = -H_mu_s*rho*g[0] - H_mu_s*d_mu*sigma*kappa[k]*n[k*3+0]/(norm_n)
	+boundaryPenaltyCoef*rho*d_mu_s*(u[k] - 0.0)
	+volumePenaltyCoef*rho*(1.0-H_mu_s)*(u[k] - 0.0);
	
      dmom_u_source_u[k] = boundaryPenaltyCoef*rho*d_mu_s
	+volumePenaltyCoef*rho*(1.0-H_mu_s);
      dmom_u_source_v[k] = 0.0;
      dmom_u_source_w[k] = 0.0;
      
      mom_v_source[k] = -H_mu_s*rho*g[1] - H_mu_s*d_mu*sigma*kappa[k]*n[k*3+1]/(norm_n)
	+boundaryPenaltyCoef*rho*d_mu_s*(v[k] - 0.0)
	+volumePenaltyCoef*rho*(1.0-H_mu_s)*(v[k] - 0.0);

      dmom_v_source_u[k] = 0.0;
      dmom_v_source_v[k] = boundaryPenaltyCoef*rho*d_mu_s
	+volumePenaltyCoef*rho*(1.0-H_mu_s);
      dmom_v_source_w[k] = 0.0;
      
      
      mom_w_source[k] = -H_mu_s*rho*g[2] - H_mu_s*d_mu*sigma*kappa[k]*n[k*3+2]/(norm_n)
	+boundaryPenaltyCoef*rho*d_mu_s*(w[k] - 0.0)
	+volumePenaltyCoef*rho*(1.0-H_mu_s)*(w[k] - 0.0);
      dmom_w_source_u[k] = 0.0;
      dmom_w_source_v[k] = 0.0;
      dmom_w_source_w[k] = boundaryPenaltyCoef*rho*d_mu_s
	+volumePenaltyCoef*rho*(1.0-H_mu_s);



      //u momentum Hamiltonian (pressure)
      mom_u_ham[k] = grad_p[k*3+0];
      dmom_u_ham_grad_p[k*3+0]=1.0;
      
      //v momentum Hamiltonian (pressure)
      mom_v_ham[k] = grad_p[k*3+1];
      dmom_v_ham_grad_p[k*3+1]=1.0;

      //w momentum Hamiltonian (pressure)
      mom_w_ham[k] = grad_p[k*3+2];
      dmom_w_ham_grad_p[k*3+2]=1.0;
    }
}

void TwophaseNavierStokes_LS_SO_3D_Evaluate(const int nPoints,
                                            const double eps,
                                            const double rho_0,
                                            const double nu_0,
                                            const double rho_1,
                                            const double nu_1,
                                            const double* g,
                                            const double* phi,
                                            const double *p,
                                            const double *grad_p,
                                            const double *u,
                                            const double *v,
                                            const double *w,
                                            double *mom_u_acc,
                                            double *dmom_u_acc_u,
                                            double *mom_v_acc,
                                            double *dmom_v_acc_v,
                                            double *mom_w_acc,
                                            double *dmom_w_acc_w,
                                            double *mass_adv,
                                            double *dmass_adv_u,
                                            double *dmass_adv_v,
                                            double *dmass_adv_w,
                                            double *mom_u_adv,
                                            double *dmom_u_adv_u,
                                            double *dmom_u_adv_v,
                                            double *dmom_u_adv_w,
                                            double *mom_v_adv,
                                            double *dmom_v_adv_u,
                                            double *dmom_v_adv_v,
                                            double *dmom_v_adv_w,
                                            double *mom_w_adv,
                                            double *dmom_w_adv_u,
                                            double *dmom_w_adv_v,
                                            double *dmom_w_adv_w,
                                            double *mom_u_diff_ten,
                                            double *mom_v_diff_ten,
                                            double *mom_w_diff_ten,
                                            double *mom_u_source,
                                            double *mom_v_source,
                                            double *mom_w_source,
                                            double *mom_u_ham,
                                            double *dmom_u_ham_grad_p,
                                            double *mom_v_ham,
                                            double *dmom_v_ham_grad_p,
                                            double *mom_w_ham,
                                            double *dmom_w_ham_grad_p)
{
  int k;
  double rho,nu,H;
  for (k=0;k<nPoints;k++)
    {
      H = smoothedHeaviside(eps,phi[k]);
      rho = rho_0*(1.0-H)+rho_1*H;
      nu  = nu_0*(1.0-H)+nu_1*H;

      //momentum accumulation
      mom_u_acc[k]=u[k];
      dmom_u_acc_u[k]=1.0;
      
      mom_v_acc[k]=v[k];
      dmom_v_acc_v[k]=1.0;
      
      mom_w_acc[k]=w[k];
      dmom_w_acc_w[k]=1.0;
      
      //mass advective flux
      mass_adv[k*3+0]=u[k];
      mass_adv[k*3+1]=v[k];
      mass_adv[k*3+2]=w[k];

      dmass_adv_u[k*3+0]=1.0;
      dmass_adv_v[k*3+1]=1.0;
      dmass_adv_w[k*3+2]=1.0;

      //u momentum advective flux
      mom_u_adv[k*3+0]=u[k]*u[k];
      mom_u_adv[k*3+1]=u[k]*v[k];
      mom_u_adv[k*3+2]=u[k]*w[k];

      dmom_u_adv_u[k*3+0]=2.0*u[k];
      dmom_u_adv_u[k*3+1]=v[k];
      dmom_u_adv_u[k*3+2]=w[k];

      dmom_u_adv_v[k*3+1]=u[k];

      dmom_u_adv_w[k*3+2]=u[k];

      //v momentum advective_flux
      mom_v_adv[k*3+0]=v[k]*u[k];
      mom_v_adv[k*3+1]=v[k]*v[k];
      mom_v_adv[k*3+2]=v[k]*w[k];
      
      dmom_v_adv_u[k*3+0]=v[k];
      
      dmom_v_adv_v[k*3+0]=u[k];
      dmom_v_adv_v[k*3+1]=2.0*v[k];
      dmom_v_adv_v[k*3+2]=w[k];

      dmom_v_adv_w[k*3+2]=v[k];
      
      //w momentum advective_flux
      mom_w_adv[k*3+0]=w[k]*u[k];
      mom_w_adv[k*3+1]=w[k]*v[k];
      mom_w_adv[k*3+2]=w[k]*w[k];
      
      dmom_w_adv_u[k*3+0]=w[k];
      
      dmom_w_adv_v[k*3+0]=w[k];
      
      dmom_w_adv_w[k*3+0]=u[k];
      dmom_w_adv_w[k*3+1]=v[k];
      dmom_w_adv_w[k*3+2]=2.0*w[k];
      
      //u momentum diffusion tensor
      mom_u_diff_ten[k*9+0] = nu;
      mom_u_diff_ten[k*9+4] = nu;
      mom_u_diff_ten[k*9+8] = nu;
      
      //v momentum diffusion tensor
      mom_v_diff_ten[k*9+0] = nu;
      mom_v_diff_ten[k*9+4] = nu;
      mom_v_diff_ten[k*9+8] = nu;
      
      //w momentum diffusion tensor
      mom_w_diff_ten[k*9+0] = nu;
      mom_w_diff_ten[k*9+4] = nu;
      mom_w_diff_ten[k*9+8] = nu;
      
      //momentum sources
      mom_u_source[k] = -g[0];
      mom_v_source[k] = -g[1];
      mom_w_source[k] = -g[2];
      
      //u momentum Hamiltonian (pressure)
      mom_u_ham[k] = grad_p[k*3+0]/rho;
      dmom_u_ham_grad_p[k*3+0]=1.0/rho;

      //v momentum Hamiltonian (pressure)
      mom_v_ham[k] = grad_p[k*3+1]/rho;
      dmom_v_ham_grad_p[k*3+1]=1.0/rho;

      //w momentum Hamiltonian (pressure)
      mom_w_ham[k] = grad_p[k*3+2]/rho;
      dmom_w_ham_grad_p[k*3+2]=1.0/rho;
    }
}

void TwophaseStokes_LS_SO_2D_Evaluate(const int nPoints,
                                      const double eps,
                                      const double rho_0,
                                      const double nu_0,
                                      const double rho_1,
                                      const double nu_1,
                                      const double* g,
                                      const double* phi,
                                      const double *p,
                                      const double *grad_p,
                                      const double *u,
                                      const double *v,
                                      double *mom_u_acc,
                                      double *dmom_u_acc_u,
                                      double *mom_v_acc,
                                      double *dmom_v_acc_v,
                                      double *mass_adv,
                                      double *dmass_adv_u,
                                      double *dmass_adv_v,
                                      double *mom_u_diff_ten,
                                      double *mom_v_diff_ten,
                                      double *mom_u_source,
                                      double *mom_v_source,
                                      double *mom_u_ham,
                                      double *dmom_u_ham_grad_p,
                                      double *mom_v_ham,
                                      double *dmom_v_ham_grad_p)
{
  int k;
  double rho,nu,H;
  for (k=0;k<nPoints;k++)
    {
      /** \todo optimize coefficients */
      H = smoothedHeaviside(eps,phi[k]);
      rho = rho_0*(1.0-H)+rho_1*H;
      nu  = nu_0*(1.0-H)+nu_1*H;

      //u momentum accumulation
      mom_u_acc[k]=u[k];
      dmom_u_acc_u[k]=1.0;
      
      //v momentum accumulation
      mom_v_acc[k]=v[k];
      dmom_v_acc_v[k]=1.0;
      
      //mass advective flux
      mass_adv[k*2+0]=u[k];
      mass_adv[k*2+1]=v[k];
      
      dmass_adv_u[k*2+0]=1.0;
      dmass_adv_v[k*2+1]=1.0;

      //u momentum diffusion tensor
      mom_u_diff_ten[k*4+0] = nu;
      mom_u_diff_ten[k*4+3] = nu;
      
      //v momentum diffusion tensor
      mom_v_diff_ten[k*4+0] = nu;
      mom_v_diff_ten[k*4+3] = nu;
      
      //momentum sources
      mom_u_source[k] = -g[0];
      mom_v_source[k] = -g[1];

      //u momentum Hamiltonian (pressure)
      mom_u_ham[k] = grad_p[k*2+0]/rho;
      dmom_u_ham_grad_p[k*2+0]=1.0/rho;
      
      //v momentum Hamiltonian (pressure)
      mom_v_ham[k] = grad_p[k*2+1]/rho;
      dmom_v_ham_grad_p[k*2+1]=1.0/rho;
    }
}

void TwophaseStokes_LS_SO_3D_Evaluate(const int nPoints,
                                      const double eps,
                                      const double rho_0,
                                      const double nu_0,
                                      const double rho_1,
                                      const double nu_1,
                                      const double* g,
                                      const double* phi,
                                      const double *p,
                                      const double *grad_p,
                                      const double *u,
                                      const double *v,
                                      const double *w,
                                      double *mom_u_acc,
                                      double *dmom_u_acc_u,
                                      double *mom_v_acc,
                                      double *dmom_v_acc_v,
                                      double *mom_w_acc,
                                      double *dmom_w_acc_w,
                                      double *mass_adv,
                                      double *dmass_adv_u,
                                      double *dmass_adv_v,
                                      double *dmass_adv_w,
                                      double *mom_u_diff_ten,
                                      double *mom_v_diff_ten,
                                      double *mom_w_diff_ten,
                                      double *mom_u_source,
                                      double *mom_v_source,
                                      double *mom_w_source,
                                      double *mom_u_ham,
                                      double *dmom_u_ham_grad_p,
                                      double *mom_v_ham,
                                      double *dmom_v_ham_grad_p,
                                      double *mom_w_ham,
                                      double *dmom_w_ham_grad_p)
{
  int k;
  double rho,nu,H;
  for (k=0;k<nPoints;k++)
    {
      H = smoothedHeaviside(eps,phi[k]);
      rho = rho_0*(1.0-H)+rho_1*H;
      nu  = nu_0*(1.0-H)+nu_1*H;

      //momentum accumulation
      mom_u_acc[k]=u[k];
      dmom_u_acc_u[k]=1.0;
      
      mom_v_acc[k]=v[k];
      dmom_v_acc_v[k]=1.0;
      
      mom_w_acc[k]=w[k];
      dmom_w_acc_w[k]=1.0;
      
      //mass advective flux
      mass_adv[k*3+0]=u[k];
      mass_adv[k*3+1]=v[k];
      mass_adv[k*3+2]=w[k];

      dmass_adv_u[k*3+0]=1.0;
      dmass_adv_v[k*3+1]=1.0;
      dmass_adv_w[k*3+2]=1.0;
      
      //u momentum diffusion tensor
      mom_u_diff_ten[k*9+0] = nu;
      mom_u_diff_ten[k*9+4] = nu;
      mom_u_diff_ten[k*9+8] = nu;
      
      //v momentum diffusion tensor
      mom_v_diff_ten[k*9+0] = nu;
      mom_v_diff_ten[k*9+4] = nu;
      mom_v_diff_ten[k*9+8] = nu;
      
      //w momentum diffusion tensor
      mom_w_diff_ten[k*9+0] = nu;
      mom_w_diff_ten[k*9+4] = nu;
      mom_w_diff_ten[k*9+8] = nu;
      
      //momentum sources
      mom_u_source[k] = -g[0];
      mom_v_source[k] = -g[1];
      mom_w_source[k] = -g[2];
      
      //u momentum Hamiltonian (pressure)
      mom_u_ham[k] = grad_p[k*3+0]/rho;
      dmom_u_ham_grad_p[k*3+0]=1.0/rho;

      //v momentum Hamiltonian (pressure)
      mom_v_ham[k] = grad_p[k*3+1]/rho;
      dmom_v_ham_grad_p[k*3+1]=1.0/rho;

      //w momentum Hamiltonian (pressure)
      mom_w_ham[k] = grad_p[k*3+2]/rho;
      dmom_w_ham_grad_p[k*3+2]=1.0/rho;
    }
}

void TwophaseNavierStokes_VOF_SO_2D_Evaluate(const int nPoints,
                                             const double eps,
                                             const double rho_0,
                                             const double nu_0,
                                             const double rho_1,
                                             const double nu_1,
                                             const double* g,
                                             const double* vof,
                                             const double *p,
                                             const double *grad_p,
                                             const double *u,
                                             const double *v,
                                             double *mom_u_acc,
                                             double *dmom_u_acc_u,
                                             double *mom_v_acc,
                                             double *dmom_v_acc_v,
                                             double *mass_adv,
                                             double *dmass_adv_u,
                                             double *dmass_adv_v,
                                             double *mom_u_adv,
                                             double *dmom_u_adv_u,
                                             double *dmom_u_adv_v,
                                             double *mom_v_adv,
                                             double *dmom_v_adv_u,
                                             double *dmom_v_adv_v,
                                             double *mom_u_diff_ten,
                                             double *mom_v_diff_ten,
                                             double *mom_u_source,
                                             double *mom_v_source,
                                             double *mom_u_ham,
                                             double *dmom_u_ham_grad_p,
                                             double *mom_v_ham,
                                             double *dmom_v_ham_grad_p)
{
  int k;
  double rho,nu,H;
  for (k=0;k<nPoints;k++)
    {
      /** \todo optimize coefficients */
      H = fmax(0.0,fmin(1.0,vof[k]));
      rho = rho_0*(1.0-H)+rho_1*H;
      nu  = nu_0*(1.0-H)+nu_1*H;

      //u momentum accumulation
      mom_u_acc[k]=u[k];
      dmom_u_acc_u[k]=1.0;
      
      //v momentum accumulation
      mom_v_acc[k]=v[k];
      dmom_v_acc_v[k]=1.0;
      
      //mass advective flux
      mass_adv[k*2+0]=u[k];
      mass_adv[k*2+1]=v[k];
      
      dmass_adv_u[k*2+0]=1.0;
      dmass_adv_v[k*2+1]=1.0;

      //u momentum advective flux
      mom_u_adv[k*2+0]=u[k]*u[k];
      mom_u_adv[k*2+1]=u[k]*v[k];

      dmom_u_adv_u[k*2+0]=2.0*u[k];
      dmom_u_adv_u[k*2+1]=v[k];

      dmom_u_adv_v[k*2+1]=u[k];

      //v momentum advective_flux
      mom_v_adv[k*2+0]=v[k]*u[k];
      mom_v_adv[k*2+1]=v[k]*v[k];
            
      dmom_v_adv_u[k*2+0]=v[k];
      
      dmom_v_adv_v[k*2+0]=u[k];
      dmom_v_adv_v[k*2+1]=2.0*v[k];

      //u momentum diffusion tensor
      mom_u_diff_ten[k*4+0] = nu;
      mom_u_diff_ten[k*4+3] = nu;
      
      //v momentum diffusion tensor
      mom_v_diff_ten[k*4+0] = nu;
      mom_v_diff_ten[k*4+3] = nu;
      
      //momentum sources
      mom_u_source[k] = -g[0];
      mom_v_source[k] = -g[1];

      //u momentum Hamiltonian (pressure)
      mom_u_ham[k] = grad_p[k*2+0]/rho;
      dmom_u_ham_grad_p[k*2+0]=1.0/rho;
      
      //v momentum Hamiltonian (pressure)
      mom_v_ham[k] = grad_p[k*2+1]/rho;
      dmom_v_ham_grad_p[k*2+1]=1.0/rho;
    }
}

void TwophaseNavierStokes_VOF_SO_3D_Evaluate(const int nPoints,
                                             const double eps,
                                             const double rho_0,
                                             const double nu_0,
                                             const double rho_1,
                                             const double nu_1,
                                             const double* g,
                                             const double* vof,
                                             const double *p,
                                             const double *grad_p,
                                             const double *u,
                                             const double *v,
                                             const double *w,
                                             double *mom_u_acc,
                                             double *dmom_u_acc_u,
                                             double *mom_v_acc,
                                             double *dmom_v_acc_v,
                                             double *mom_w_acc,
                                             double *dmom_w_acc_w,
                                             double *mass_adv,
                                             double *dmass_adv_u,
                                             double *dmass_adv_v,
                                             double *dmass_adv_w,
                                             double *mom_u_adv,
                                             double *dmom_u_adv_u,
                                             double *dmom_u_adv_v,
                                             double *dmom_u_adv_w,
                                             double *mom_v_adv,
                                             double *dmom_v_adv_u,
                                             double *dmom_v_adv_v,
                                             double *dmom_v_adv_w,
                                             double *mom_w_adv,
                                             double *dmom_w_adv_u,
                                             double *dmom_w_adv_v,
                                             double *dmom_w_adv_w,
                                             double *mom_u_diff_ten,
                                             double *mom_v_diff_ten,
                                             double *mom_w_diff_ten,
                                             double *mom_u_source,
                                             double *mom_v_source,
                                             double *mom_w_source,
                                             double *mom_u_ham,
                                             double *dmom_u_ham_grad_p,
                                             double *mom_v_ham,
                                             double *dmom_v_ham_grad_p,
                                             double *mom_w_ham,
                                             double *dmom_w_ham_grad_p)
{
  int k;
  double rho,nu,H;
  for (k=0;k<nPoints;k++)
    {
      H = fmax(0.0,fmin(1.0,vof[k]));
      rho = rho_0*(1.0-H)+rho_1*H;
      nu  = nu_0*(1.0-H)+nu_1*H;

      //momentum accumulation
      mom_u_acc[k]=u[k];
      dmom_u_acc_u[k]=1.0;
      
      mom_v_acc[k]=v[k];
      dmom_v_acc_v[k]=1.0;
      
      mom_w_acc[k]=w[k];
      dmom_w_acc_w[k]=1.0;
      
      //mass advective flux
      mass_adv[k*3+0]=u[k];
      mass_adv[k*3+1]=v[k];
      mass_adv[k*3+2]=w[k];

      dmass_adv_u[k*3+0]=1.0;
      dmass_adv_v[k*3+1]=1.0;
      dmass_adv_w[k*3+2]=1.0;

      //u momentum advective flux
      mom_u_adv[k*3+0]=u[k]*u[k];
      mom_u_adv[k*3+1]=u[k]*v[k];
      mom_u_adv[k*3+2]=u[k]*w[k];

      dmom_u_adv_u[k*3+0]=2.0*u[k];
      dmom_u_adv_u[k*3+1]=v[k];
      dmom_u_adv_u[k*3+2]=w[k];

      dmom_u_adv_v[k*3+1]=u[k];

      dmom_u_adv_w[k*3+2]=u[k];

      //v momentum advective_flux
      mom_v_adv[k*3+0]=v[k]*u[k];
      mom_v_adv[k*3+1]=v[k]*v[k];
      mom_v_adv[k*3+2]=v[k]*w[k];
      
      dmom_v_adv_u[k*3+0]=v[k];
      
      dmom_v_adv_v[k*3+0]=u[k];
      dmom_v_adv_v[k*3+1]=2.0*v[k];
      dmom_v_adv_v[k*3+2]=w[k];

      dmom_v_adv_w[k*3+2]=v[k];
      
      //w momentum advective_flux
      mom_w_adv[k*3+0]=w[k]*u[k];
      mom_w_adv[k*3+1]=w[k]*v[k];
      mom_w_adv[k*3+2]=w[k]*w[k];
      
      dmom_w_adv_u[k*3+0]=w[k];
      
      dmom_w_adv_v[k*3+0]=w[k];
      
      dmom_w_adv_w[k*3+0]=u[k];
      dmom_w_adv_w[k*3+1]=v[k];
      dmom_w_adv_w[k*3+2]=2.0*w[k];
      
      //u momentum diffusion tensor
      mom_u_diff_ten[k*9+0] = nu;
      mom_u_diff_ten[k*9+4] = nu;
      mom_u_diff_ten[k*9+8] = nu;
      
      //v momentum diffusion tensor
      mom_v_diff_ten[k*9+0] = nu;
      mom_v_diff_ten[k*9+4] = nu;
      mom_v_diff_ten[k*9+8] = nu;
      
      //w momentum diffusion tensor
      mom_w_diff_ten[k*9+0] = nu;
      mom_w_diff_ten[k*9+4] = nu;
      mom_w_diff_ten[k*9+8] = nu;
      
      //momentum sources
      mom_u_source[k] = -g[0];
      mom_v_source[k] = -g[1];
      mom_w_source[k] = -g[2];
      
      //u momentum Hamiltonian (pressure)
      mom_u_ham[k] = grad_p[k*3+0]/rho;
      dmom_u_ham_grad_p[k*3+0]=1.0/rho;

      //v momentum Hamiltonian (pressure)
      mom_v_ham[k] = grad_p[k*3+1]/rho;
      dmom_v_ham_grad_p[k*3+1]=1.0/rho;

      //w momentum Hamiltonian (pressure)
      mom_w_ham[k] = grad_p[k*3+2]/rho;
      dmom_w_ham_grad_p[k*3+2]=1.0/rho;
    }
}

void TwophaseStokes_VOF_SO_2D_Evaluate(const int nPoints,
                                       const double eps,
                                       const double rho_0,
                                       const double nu_0,
                                       const double rho_1,
                                       const double nu_1,
                                       const double* g,
                                       const double* vof,
                                       const double *p,
                                       const double *grad_p,
                                       const double *u,
                                       const double *v,
                                       double *mom_u_acc,
                                       double *dmom_u_acc_u,
                                       double *mom_v_acc,
                                       double *dmom_v_acc_v,
                                       double *mass_adv,
                                       double *dmass_adv_u,
                                       double *dmass_adv_v,
                                       double *mom_u_diff_ten,
                                       double *mom_v_diff_ten,
                                       double *mom_u_source,
                                       double *mom_v_source,
                                       double *mom_u_ham,
                                       double *dmom_u_ham_grad_p,
                                       double *mom_v_ham,
                                       double *dmom_v_ham_grad_p)
{
  int k;
  double rho,nu,H;
  for (k=0;k<nPoints;k++)
    {
      /** \todo optimize coefficients */
      H = fmax(0.0,fmin(1.0,vof[k]));
      rho = rho_0*(1.0-H)+rho_1*H;
      nu  = nu_0*(1.0-H)+nu_1*H;

      //u momentum accumulation
      mom_u_acc[k]=u[k];
      dmom_u_acc_u[k]=1.0;
      
      //v momentum accumulation
      mom_v_acc[k]=v[k];
      dmom_v_acc_v[k]=1.0;
      
      //mass advective flux
      mass_adv[k*2+0]=u[k];
      mass_adv[k*2+1]=v[k];
      
      dmass_adv_u[k*2+0]=1.0;
      dmass_adv_v[k*2+1]=1.0;

      //u momentum diffusion tensor
      mom_u_diff_ten[k*4+0] = nu;
      mom_u_diff_ten[k*4+3] = nu;
      
      //v momentum diffusion tensor
      mom_v_diff_ten[k*4+0] = nu;
      mom_v_diff_ten[k*4+3] = nu;
      
      //momentum sources
      mom_u_source[k] = -g[0];
      mom_v_source[k] = -g[1];

      //u momentum Hamiltonian (pressure)
      mom_u_ham[k] = grad_p[k*2+0]/rho;
      dmom_u_ham_grad_p[k*2+0]=1.0/rho;
      
      //v momentum Hamiltonian (pressure)
      mom_v_ham[k] = grad_p[k*2+1]/rho;
      dmom_v_ham_grad_p[k*2+1]=1.0/rho;
    }
}

void TwophaseStokes_VOF_SO_3D_Evaluate(const int nPoints,
                                       const double eps,
                                       const double rho_0,
                                       const double nu_0,
                                       const double rho_1,
                                       const double nu_1,
                                       const double* g,
                                       const double* vof,
                                       const double *p,
                                       const double *grad_p,
                                       const double *u,
                                       const double *v,
                                       const double *w,
                                       double *mom_u_acc,
                                       double *dmom_u_acc_u,
                                       double *mom_v_acc,
                                       double *dmom_v_acc_v,
                                       double *mom_w_acc,
                                       double *dmom_w_acc_w,
                                       double *mass_adv,
                                       double *dmass_adv_u,
                                       double *dmass_adv_v,
                                       double *dmass_adv_w,
                                       double *mom_u_diff_ten,
                                       double *mom_v_diff_ten,
                                       double *mom_w_diff_ten,
                                       double *mom_u_source,
                                       double *mom_v_source,
                                       double *mom_w_source,
                                       double *mom_u_ham,
                                       double *dmom_u_ham_grad_p,
                                       double *mom_v_ham,
                                       double *dmom_v_ham_grad_p,
                                       double *mom_w_ham,
                                       double *dmom_w_ham_grad_p)
{
  int k;
  double rho,nu,H;
  for (k=0;k<nPoints;k++)
    {
      H = fmax(0.0,fmin(1.0,vof[k]));
      rho = rho_0*(1.0-H)+rho_1*H;
      nu  = nu_0*(1.0-H)+nu_1*H;

      //momentum accumulation
      mom_u_acc[k]=u[k];
      dmom_u_acc_u[k]=1.0;
      
      mom_v_acc[k]=v[k];
      dmom_v_acc_v[k]=1.0;
      
      mom_w_acc[k]=w[k];
      dmom_w_acc_w[k]=1.0;
      
      //mass advective flux
      mass_adv[k*3+0]=u[k];
      mass_adv[k*3+1]=v[k];
      mass_adv[k*3+2]=w[k];

      dmass_adv_u[k*3+0]=1.0;
      dmass_adv_v[k*3+1]=1.0;
      dmass_adv_w[k*3+2]=1.0;
      
      //u momentum diffusion tensor
      mom_u_diff_ten[k*9+0] = nu;
      mom_u_diff_ten[k*9+4] = nu;
      mom_u_diff_ten[k*9+8] = nu;
      
      //v momentum diffusion tensor
      mom_v_diff_ten[k*9+0] = nu;
      mom_v_diff_ten[k*9+4] = nu;
      mom_v_diff_ten[k*9+8] = nu;
      
      //w momentum diffusion tensor
      mom_w_diff_ten[k*9+0] = nu;
      mom_w_diff_ten[k*9+4] = nu;
      mom_w_diff_ten[k*9+8] = nu;
      
      //momentum sources
      mom_u_source[k] = -g[0];
      mom_v_source[k] = -g[1];
      mom_w_source[k] = -g[2];
      
      //u momentum Hamiltonian (pressure)
      mom_u_ham[k] = grad_p[k*3+0]/rho;
      dmom_u_ham_grad_p[k*3+0]=1.0/rho;

      //v momentum Hamiltonian (pressure)
      mom_v_ham[k] = grad_p[k*3+1]/rho;
      dmom_v_ham_grad_p[k*3+1]=1.0/rho;

      //w momentum Hamiltonian (pressure)
      mom_w_ham[k] = grad_p[k*3+2]/rho;
      dmom_w_ham_grad_p[k*3+2]=1.0/rho;
    }
}

void unitSquareVortexEvaluate(const int nPoints,
			      const int nSpace,
			      double t,
			      const double *x,
			      const double *u,
			      double *m,
			      double *dm,
			      double *f,
			      double *df)
{
  double vx, vy, xk, yk;
  int k;
  double one8 = 1.0/8.0;
  for (k=0; k < nPoints; k++)
    {
      m[k] = u[k];
      dm[k] = 1.0;
      xk = x[k*3]; yk = x[k*3+1];
      vx = cos(M_PI*one8*t)*sin(2.0*M_PI*yk)*sin(M_PI*xk)*sin(M_PI*xk);
      vy =-cos(M_PI*one8*t)*sin(2.0*M_PI*xk)*sin(M_PI*yk)*sin(M_PI*yk);
      f[k*nSpace] =   vx*u[k];
      f[k*nSpace+1] = vy*u[k];
      df[k*nSpace] = vx;
      df[k*nSpace+1] = vy;
    }
}

/*for HJ testing*/
void constantVelocityLevelSetEvaluate(const int nPoints,
				      const int nSpace,
				      const double *b,
				      const double *x,
				      const double *u,
				      const double *gradu,
				      double *m,
				      double *dm,
				      double *f,
				      double *df,
				      double *H,
				      double *dH)
{
  double bdotgrad;
  int k,id;
  for (k=0; k < nPoints; k++)
    {
      m[k] = u[k];
      dm[k] = 1.0;
      bdotgrad=0.0;
      for (id=0; id < nSpace; id++)
	{
	  f[k*nSpace+id] = 0.0; /*just for now since helps get all the right quad terms*/
	  df[k*nSpace+id]= 0.0; /*just for now since helps get all the right quad terms*/

	  bdotgrad+= gradu[k*nSpace+id]*b[id];
	  dH[k*nSpace+id]=b[id];
	}
      H[k] = bdotgrad;
    }
}
void constantNormalVelocityLevelSetEvaluate(const int nPoints,
					    const int nSpace,
					    double b,
					    const double *x,
					    const double *u,
					    const double *gradu,
					    double *m,
					    double *dm,
					    double *f,
					    double *df,
					    double *H,
					    double *dH)
{
  double normgradu;
  int k,id;
  for (k=0; k < nPoints; k++)
    {
      m[k] = u[k];
      dm[k] = 1.0;
      normgradu=0.0;
      for (id=0; id < nSpace; id++)
	{
	  f[k*nSpace+id] = 0.0; /*just for now since helps get all the right quad terms*/
	  df[k*nSpace+id]= 0.0; /*just for now since helps get all the right quad terms*/

	  normgradu+= gradu[k*nSpace+id]*gradu[k*nSpace+id];
	  
	}
      normgradu = sqrt(normgradu);
      H[k] = b*normgradu;
      for (id=0; id < nSpace; id++)
	{
	  dH[k*nSpace+id]=gradu[k*nSpace+id]/(normgradu+1.0e-8);
	}
    }
}
void unitSquareVortexLevelSetEvaluate(const int nPoints,
				      const int nSpace,
				      double t,
				      const double *x,
				      const double *u,
				      const double *gradu,
				      double *m,
				      double *dm,
				      double *f,
				      double *df,
				      double *H,
				      double *dH)
{
  double v[3], xk, yk, vdotgrad, one8;
  int k,id;
  v[2] = 0.0;
  one8 = 1.0/8.0;
  for (k=0; k < nPoints; k++)
    {
      m[k] = u[k];
      dm[k] = 1.0;
      xk = x[k*3]; yk = x[k*3+1];
      v[0] = cos(M_PI*one8*t)*sin(2.0*M_PI*yk)*sin(M_PI*xk)*sin(M_PI*xk);
      v[1] =-cos(M_PI*one8*t)*sin(2.0*M_PI*xk)*sin(M_PI*yk)*sin(M_PI*yk);
      vdotgrad=0.0;
      for (id=0; id < nSpace; id++)
	{
	  f[k*nSpace+id] = 0.0; /*just for now since helps get all the right quad terms*/
	  df[k*nSpace+id]= 0.0; /*just for now since helps get all the right quad terms*/

	  vdotgrad+= gradu[k*nSpace+id]*v[id];
	  dH[k*nSpace+id]=v[id];
	}
      H[k] = vdotgrad;
    }
}
void unitSquareRotationLevelSetEvaluate(const int nPoints,
					const int nSpace,
					double t,
					const double *x,
					const double *u,
					const double *gradu,
					double *m,
					double *dm,
					double *f,
					double *df,
					double *H,
					double *dH)
{
  double v[3], xk, yk, vdotgrad, one8;
  int k,id;
  v[2] = 0.0;
  one8 = 1.0/8.0;
  for (k=0; k < nPoints; k++)
    {
      m[k] = u[k];
      dm[k] = 1.0;
      xk = x[k*3]; yk = x[k*3+1];
      v[0] = 2.0*M_PI*(x[k*3+1] - 0.5);
      v[1] =2.0*M_PI*(0.5     - x[k*3]);
      vdotgrad=0.0;
      for (id=0; id < nSpace; id++)
	{
	  f[k*nSpace+id] = 0.0; /*just for now since helps get all the right quad terms*/
	  df[k*nSpace+id]= 0.0; /*just for now since helps get all the right quad terms*/

	  vdotgrad+= gradu[k*nSpace+id]*v[id];
	  dH[k*nSpace+id]=v[id];
	}
      H[k] = vdotgrad;
    }
}

void HJBurgersEvaluate(const int nPoints,
		       const int nSpace,
		       const double offset,
		       const double *u,
		       const double *gradu,
		       double *m,
		       double *dm,
		       double *H,
		       double *dH)
{
  int k,id,jd;
  double tmp;
  for (k=0; k < nPoints; k++)
    {
      m[k] = u[k];
      dm[k] = 1.0;
      tmp = offset;
      for (id=0; id < nSpace; id++)
	{
	  tmp += gradu[k*nSpace+id];
	  dH[k*nSpace+id]= offset;
	  for (jd=0; jd < nSpace; jd++)
	    dH[k*nSpace+id] += gradu[k*nSpace+jd];
	}
      H[k] = tmp*tmp*0.5;
    }
}

/* /\** Coefficients for the mass conservative head-based Richards' equation using Mualem-Van Genuchten. */
/*  *\/ */
/* void conservativeHeadRichardsMualemVanGenuchtenEvaluate(const int nPoints, */
/*                                                         const int nSpace, */
/*                                                         const double rho, */
/*                                                         const double* gravity, */
/*                                                         const double* alpha, */
/*                                                         const double* n, */
/*                                                         const double* m, */
/*                                                         const double* thetaS, */
/*                                                         const double* thetaR, */
/*                                                         const double* thetaSR, */
/*                                                         const double* KWs, */
/*                                                         double *u, */
/*                                                         double *mass, */
/*                                                         double *dmass, */
/*                                                         double *f, */
/*                                                         double *df, */
/*                                                         double *a, */
/*                                                         double *da, */
/*                                                         double *phi, */
/*                                                         double *dphi) */
/* { */
/*   proteus_index_t J; int k, I; */
/*   const int nSpace2=nSpace*nSpace; */
/*   register double psiC,alphaPsiC,alphaPsiC_n,alphaPsiC_nM1,onePlus_alphaPsiC_n,sBar,sBarByOnePlus_alphaPsiC_n,sqrt_sBar,sqrt_1minusSbar,thetaW,DsBar_DpC,DthetaW_DpC,vBar,uBar,krW,krN, */
/*     alphaPsiC_nM2,sBarBy_onePlus_alphaPsiC_n_2,DDsBar_DDpC,DDthetaW_DDpC,DkrW_DpC,rho2=rho*rho; */
/*   double KW[9],KN[9],DKW_DpC[9]; */
/*   for (k=0;k<nPoints;k++) */
/*     { */
/*       psiC = -u[k]; */
/*       if (psiC > 0.0) */
/*         { */
/*           alphaPsiC = alpha[k]*psiC; */
/*           alphaPsiC_n = pow(alphaPsiC,n[k]); */
/*           alphaPsiC_nM1 = alphaPsiC_n/alphaPsiC; */
/*           alphaPsiC_nM2 =   alphaPsiC_nM1/alphaPsiC;                 */
/*           onePlus_alphaPsiC_n = 1.0 + alphaPsiC_n; */
/*           sBar = pow(onePlus_alphaPsiC_n,-m[k]); */
/*           sBarByOnePlus_alphaPsiC_n = sBar/onePlus_alphaPsiC_n; */
/*           sqrt_sBar = sqrt(sBar); */
/*           sqrt_1minusSbar = sqrt(1.0 - sBar); */
/*           thetaW = thetaSR[k]*sBar + thetaR[k]; */
/*           DsBar_DpC = -alpha[k]*(n[k]-1.0)*alphaPsiC_nM1  */
/*             *sBarByOnePlus_alphaPsiC_n; */
/*           DthetaW_DpC = thetaSR[k] * DsBar_DpC;  */
/*           vBar = 1.0-alphaPsiC_nM1*sBar; */
/*           uBar = alphaPsiC_nM1*sBar; */
/*           krW = sqrt_sBar*vBar*vBar; */
/*           DkrW_DpC = (0.5/sqrt_sBar)*DsBar_DpC*vBar*vBar */
/*             - */
/*             2.0*sqrt_sBar*vBar* */
/*             (alpha[k]*(n[k]-1.0)*alphaPsiC_nM2*sBar */
/*              + alphaPsiC_nM1 * DsBar_DpC); */
/*           for (I=0;I<nSpace;I++) */
/*             for(J=0;J<nSpace;J++) */
/*               { */
/*                 KW[I*nSpace + J] = KWs[k*nSpace2 + I*nSpace + J]*krW;  */
/*                 DKW_DpC[I*nSpace + J] = KWs[k*nSpace2 + I*nSpace + J]*DkrW_DpC;  */
/*               } */
/*         } */
/*       else */
/*         { */
/*           sBar = 1.0; */
/*           thetaW = thetaS[k]; */
/*           DsBar_DpC = 0.0; */
/*           DthetaW_DpC = 0.0; */
/*           krW = 1.0; */
/*           for (I=0;I<nSpace;I++) */
/*             for(J=0;J<nSpace;J++) */
/*               { */
/*                 KW[I*nSpace + J] = KWs[k*nSpace2 + I*nSpace + J];  */
/*                 DKW_DpC[I*nSpace + J] = 0.0;  */
/*               } */
/*         } */
/*       mass[k] = rho*thetaW; */
/*       dmass[k] = -rho*DthetaW_DpC; */
/*       for (I=0;I<nSpace;I++) */
/*         { */
/*           f[k*nSpace+I] = 0.0; */
/*           df[k*nSpace+I] = 0.0; */
/*           for (J=0;J<nSpace;J++) */
/*             { */
/*               f[k*nSpace+I]  += rho2*KW[I*nSpace + J]*gravity[J]; */
/*               df[k*nSpace+I] -= rho2*DKW_DpC[I*nSpace + J]*gravity[J]; */
/*               a[k*nSpace2+I*nSpace+J]  = rho*KW[I*nSpace + J]; */
/*               da[k*nSpace2+I*nSpace+J] = -rho*DKW_DpC[I*nSpace + J]; */
/*             } */
/*         } */
/*     } */
/* } */

/** Coefficients for the mass conservative head-based Richards' equation using Mualem-Van Genuchten.
 */
void conservativeHeadRichardsMualemVanGenuchtenHomEvaluate(const int nPoints,
                                                           const int nSpace,
                                                           const double rho,
                                                           const double beta,
                                                           const double* gravity,
                                                           const double* x,
                                                           const double alpha,
                                                           const double n,
                                                           const double m,
                                                           const double thetaR,
                                                           const double thetaSR,
                                                           const double KWs,
                                                           double *u,
                                                           double *mass,
                                                           double *dmass,
                                                           double *f,
                                                           double *df,
                                                           double *a,
                                                           double *da,
                                                           double *phi,
                                                           double *dphi)
{
  int k,I;
  const int nSpace2=nSpace*nSpace;
  register double psiC,
    pcBar,pcBar_n,pcBar_nM1,pcBar_nM2,
    onePlus_pcBar_n,
    sBar,sqrt_sBar,DsBar_DpsiC,
    thetaW,DthetaW_DpsiC,
    vBar,vBar2,DvBar_DpsiC,
    KW,DKW_DpsiC,
    rho2=rho*rho,
    thetaS=thetaR+thetaSR,
    rhom,drhom;
  for (k=0;k<nPoints;k++)
    {
      psiC = -u[k];
      if (psiC > 0.0)
        {
          pcBar = alpha*psiC;
          pcBar_nM2 = pow(pcBar,n-2);
          pcBar_nM1 = pcBar_nM2*pcBar;
          pcBar_n   = pcBar_nM1*pcBar;
          onePlus_pcBar_n = 1.0 + pcBar_n;
          
          sBar = pow(onePlus_pcBar_n,-m);
          /* using -mn = 1-n */
          DsBar_DpsiC = alpha*(1.0-n)*(sBar/onePlus_pcBar_n)*pcBar_nM1;
          
          vBar = 1.0-pcBar_nM1*sBar;
          vBar2 = vBar*vBar;
          DvBar_DpsiC = -alpha*(n-1.0)*pcBar_nM2*sBar - pcBar_nM1*DsBar_DpsiC;
          
          thetaW = thetaSR*sBar + thetaR;
          DthetaW_DpsiC = thetaSR * DsBar_DpsiC; 
          
          sqrt_sBar = sqrt(sBar);
          KW= KWs*sqrt_sBar*vBar2;
          DKW_DpsiC= KWs*
            ((0.5/sqrt_sBar)*DsBar_DpsiC*vBar2
             +
             2.0*sqrt_sBar*vBar*DvBar_DpsiC);
        }
      else
        {
          thetaW        = thetaS;
          DthetaW_DpsiC = 0.0;
          KW            = KWs; 
          DKW_DpsiC     = 0.0;
        }
      //slight compressibility
      rhom = rho*exp(beta*u[k]);
      drhom = beta*rhom;
      
      mass[k] = rhom*thetaW;
      dmass[k] = -rhom*DthetaW_DpsiC+drhom*thetaW;
      for (I=0;I<nSpace;I++)
        {
          f[k*nSpace+I]  = rho2*KW*gravity[I];
          df[k*nSpace+I] = -rho2*DKW_DpsiC*gravity[I];

          a[k*nSpace2+I*nSpace+I]  = rho*KW;
          da[k*nSpace2+I*nSpace+I] = -rho*DKW_DpsiC;
        }
    }
}
/* mwf begin unnecessary RE additions */
/* TODO: figure out how to handle dmass term to get right jacobian*/
void conservativeHeadRichardsL2projMualemVanGenuchtenHomEvaluate(const int nSimplices,
								 const int nPointsPerSimplex,
								 const int nSpace,
								 const double rho,
								 const double* gravity,
								 const double alpha,
								 const double n,
								 const double m,
								 const double thetaR,
								 const double thetaSR,
								 const double KWs,
								 double *dV,
								 double *u,
								 double *mass,
								 double *dmass,
								 double *f,
								 double *df,
								 double *a,
								 double *da)
{
  proteus_index_t J; int k, I, eN;
  const int nSpace2=nSpace*nSpace;
  register double psiC,
    pcBar,pcBar_n,pcBar_nM1,pcBar_nM2,
    onePlus_pcBar_n,
    sBar,sqrt_sBar,DsBar_DpsiC,
    thetaW,DthetaW_DpsiC,
    vBar,vBar2,DvBar_DpsiC,
    KW,DKW_DpsiC,
    rho2=rho*rho,
    thetaS=thetaR+thetaSR;

  double mavg,dmavg,vol;
  double favg[3]   = {0.0,0.0,0.0};
  double dfavg[3]  = {0.0,0.0,0.0};
  double aavg[3][3]= {{0.0,0.0,0.0},
		      {0.0,0.0,0.0},
		      {0.0,0.0,0.0}};
  double daavg[3][3]= {{0.0,0.0,0.0},
		       {0.0,0.0,0.0},
		       {0.0,0.0,0.0}};
  /*mwf debug
    printf("revgL2proj nSimplices=%d nPerSimp=%d nSpace=%d\n",nSimplices,nPointsPerSimplex,
    nSpace);
  */
  for (eN = 0; eN < nSimplices; eN++)
    {
      mavg = 0.0; dmavg = 0.0; vol = 0.0;
      for (I=0; I < nSpace; I++)
	{
	  favg[I] =0.0; dfavg[I] = 0.0;
	  for (J=0; J < nSpace;J++)
	    {
	      aavg[I][J] = 0.0; daavg[I][J] = 0.0;
	    }
	}
      for (k=0;k<nPointsPerSimplex;k++)
	{
	  psiC = -u[eN*nPointsPerSimplex + k];
	  if (psiC > 0.0)
	    {
	      pcBar = alpha*psiC;
	      pcBar_nM2 = pow(pcBar,n-2);
	      pcBar_nM1 = pcBar_nM2*pcBar;
	      pcBar_n   = pcBar_nM1*pcBar;
	      onePlus_pcBar_n = 1.0 + pcBar_n;
          
	      sBar = pow(onePlus_pcBar_n,-m);
	      /* using -mn = 1-n */
	      DsBar_DpsiC = alpha*(1.0-n)*(sBar/onePlus_pcBar_n)*pcBar_nM1;
          
	      vBar = 1.0-pcBar_nM1*sBar;
	      vBar2 = vBar*vBar;
	      DvBar_DpsiC = -alpha*(n-1.0)*pcBar_nM2*sBar - pcBar_nM1*DsBar_DpsiC;
          
	      thetaW = thetaSR*sBar + thetaR;
	      DthetaW_DpsiC = thetaSR * DsBar_DpsiC; 
          
	      sqrt_sBar = sqrt(sBar);
	      KW= KWs*sqrt_sBar*vBar2;
	      DKW_DpsiC= KWs*
		((0.5/sqrt_sBar)*DsBar_DpsiC*vBar2
		 +
		 2.0*sqrt_sBar*vBar*DvBar_DpsiC);
	    }
	  else
	    {
	      thetaW        = thetaS;
	      DthetaW_DpsiC = 0.0;
	      KW            = KWs; 
	      DKW_DpsiC     = 0.0;
	    }
	  /*mwf debug
	    printf("revgmL2 eN=%d k=%d psiC=%g thetaW=%g KW=%g rho=%g DKW_DpsiC=%g\n",
	    eN,k,psiC,thetaW,KW,rho,DKW_DpsiC);
	  */ 
	  vol   += dV[eN*nPointsPerSimplex + k];
	  mavg  += rho*thetaW*dV[eN*nPointsPerSimplex + k];
	  dmavg +=-rho*DthetaW_DpsiC*dV[eN*nPointsPerSimplex + k];
	  /*go ahead and assign point values for derivs since this is what's actually
	    appropriate at least for stiffness and advection terms,
	    but it's not correct for mass term*/
	  dmass[eN*nPointsPerSimplex+ k] =-rho*DthetaW_DpsiC; 

	  for (I=0; I < nSpace; I++)
	    {
	      favg[I] += rho2*KW*gravity[I]*dV[eN*nPointsPerSimplex + k];
	      dfavg[I]+=-rho2*DKW_DpsiC*gravity[I]*dV[eN*nPointsPerSimplex + k]; 
	      J=I;/*assume isotropic for now*/
	      aavg[I][J]  += rho*KW*dV[eN*nPointsPerSimplex + k];
	      daavg[I][J] +=-rho*DKW_DpsiC*dV[eN*nPointsPerSimplex + k];

	      df[eN*nPointsPerSimplex*nSpace+ k*nSpace + I] = -rho2*DKW_DpsiC*gravity[I];
	      da[eN*nPointsPerSimplex*nSpace2 + k*nSpace2 + I*nSpace + J] = -rho*DKW_DpsiC;
	    }
	}/*end k 1*/
      assert(vol > 0.0);
      /*mwf debug
	printf("eN=%d vol=%g mavg=%g dmavg=%g favg=[%g,%g,%g] aavg=[%g,%g,%g] \n",
	eN,vol,mavg,dmavg,favg[0],favg[1],favg[2],aavg[0][0],aavg[1][1],aavg[2][2]);
      */
      for (k=0; k < nPointsPerSimplex; k++)
	{
	  mass[eN*nPointsPerSimplex + k] = mavg/vol; 
	  /*dmass[eN*nPointsPerSimplex+ k] = dmavg/vol;*/ 
	  for (I=0; I < nSpace; I++)
	    {
	      f[eN*nPointsPerSimplex*nSpace + k*nSpace + I] = favg[I]/vol;
	      /*df[eN*nPointsPerSimplex*nSpace+ k*nSpace + I] = dfavg[I]/vol;*/
	      for (J=0; J < nSpace; J++)
		{
		  if (J == I)
		    {
		      a[eN*nPointsPerSimplex*nSpace2 + k*nSpace2 + I*nSpace + J] = aavg[I][J]/vol;
		      /*da[eN*nPointsPerSimplex*nSpace2 + k*nSpace2 + I*nSpace + J]= daavg[I][J]/vol;*/
		    }
		  else
		    {
		      a[eN*nPointsPerSimplex*nSpace2 + k*nSpace2 + I*nSpace + J] = 0.0;
		      /*da[eN*nPointsPerSimplex*nSpace2 + k*nSpace2 + I*nSpace + J]= 0.0;*/
		    }
		}
		
	    }/*I*/
        }/*k*/
    }
}
/* TODO: figure out how to handle dmass term to get right jacobian*/
void conservativeHeadRichardsL2projBndMualemVanGenuchtenHomEvaluate(const proteus_index_t nElements,
								    const int nElementBoundaries_element,
								    const int nPointsPerElementBoundary,
								    const int nSpace,
								    const double rho,
								    const double* gravity,
								    const double alpha,
								    const double n,
								    const double m,
								    const double thetaR,
								    const double thetaSR,
								    const double KWs,
								    double *dV,
								    double *u,
								    double *mass,
								    double *dmass,
								    double *f,
								    double *df,
								    double *a,
								    double *da)
{
  proteus_index_t J; int k, I, eN, ebN;
  const int nSpace2=nSpace*nSpace;
  register double psiC,
    pcBar,pcBar_n,pcBar_nM1,pcBar_nM2,
    onePlus_pcBar_n,
    sBar,sqrt_sBar,DsBar_DpsiC,
    thetaW,DthetaW_DpsiC,
    vBar,vBar2,DvBar_DpsiC,
    KW,DKW_DpsiC,
    rho2=rho*rho,
    thetaS=thetaR+thetaSR;

  double mavg,dmavg,vol;
  double favg[3]   = {0.0,0.0,0.0};
  double dfavg[3]  = {0.0,0.0,0.0};
  double aavg[3][3]= {{0.0,0.0,0.0},
		      {0.0,0.0,0.0},
		      {0.0,0.0,0.0}};
  double daavg[3][3]= {{0.0,0.0,0.0},
		       {0.0,0.0,0.0},
		       {0.0,0.0,0.0}};
  /*mwf debug
    printf("revgL2projBnd nElements=%d nElemBndp=%d nPtsPerBnd=%d nSpace=%d\n",nElements,nElementBoundaries_element,
	   nPointsPerElementBoundary,
	   nSpace);
  */
  for (eN = 0; eN < nElements; eN++)
    {
      for (ebN = 0; ebN < nElementBoundaries_element; ebN++)
	{
	  mavg = 0.0; dmavg = 0.0; vol = 0.0;
	  for (I=0; I < nSpace; I++)
	    {
	      favg[I] =0.0; dfavg[I] = 0.0;
	      for (J=0; J < nSpace;J++)
		{
		  aavg[I][J] = 0.0; daavg[I][J] = 0.0;
		}
	    }
	  for (k=0;k<nPointsPerElementBoundary;k++)
	    {
	      psiC = -u[eN*nElementBoundaries_element*nPointsPerElementBoundary + ebN*nPointsPerElementBoundary  + k];
	      if (psiC > 0.0)
		{
		  pcBar = alpha*psiC;
		  pcBar_nM2 = pow(pcBar,n-2);
		  pcBar_nM1 = pcBar_nM2*pcBar;
		  pcBar_n   = pcBar_nM1*pcBar;
		  onePlus_pcBar_n = 1.0 + pcBar_n;
          
		  sBar = pow(onePlus_pcBar_n,-m);
		  /* using -mn = 1-n */
		  DsBar_DpsiC = alpha*(1.0-n)*(sBar/onePlus_pcBar_n)*pcBar_nM1;
		  
		  vBar = 1.0-pcBar_nM1*sBar;
		  vBar2 = vBar*vBar;
		  DvBar_DpsiC = -alpha*(n-1.0)*pcBar_nM2*sBar - pcBar_nM1*DsBar_DpsiC;
		  
		  thetaW = thetaSR*sBar + thetaR;
		  DthetaW_DpsiC = thetaSR * DsBar_DpsiC; 
		  
		  sqrt_sBar = sqrt(sBar);
		  KW= KWs*sqrt_sBar*vBar2;
		  DKW_DpsiC= KWs*
		    ((0.5/sqrt_sBar)*DsBar_DpsiC*vBar2
		     +
		     2.0*sqrt_sBar*vBar*DvBar_DpsiC);
		}
	      else
		{
		  thetaW        = thetaS;
		  DthetaW_DpsiC = 0.0;
		  KW            = KWs; 
		  DKW_DpsiC     = 0.0;
		}
	      /*mwf debug
	      printf("revgmL2 eN=%d ebN=%d k=%d psiC=%g thetaW=%g KW=%g rho=%g DKW_DpsiC=%g\n",
		     eN,ebN,k,psiC,thetaW,KW,rho,DKW_DpsiC);
	      */
	      vol   += dV[eN*nElementBoundaries_element*nPointsPerElementBoundary + ebN*nPointsPerElementBoundary  + k];
	      mavg  += rho*thetaW*dV[eN*nElementBoundaries_element*nPointsPerElementBoundary + ebN*nPointsPerElementBoundary  + k];
	      dmavg +=-rho*DthetaW_DpsiC*dV[eN*nElementBoundaries_element*nPointsPerElementBoundary + ebN*nPointsPerElementBoundary  + k];
	      /*mwf go ahead and assign point values for derivs since this is what's needed for stiffness terms but
		even though it's not the right thing to do for mass term*/
	      dmass[eN*nElementBoundaries_element*nPointsPerElementBoundary + ebN*nPointsPerElementBoundary + k] = 
		-rho*DthetaW_DpsiC;
	      for (I=0; I < nSpace; I++)
		{
		  favg[I] += rho2*KW*gravity[I]*dV[eN*nElementBoundaries_element*nPointsPerElementBoundary + 
						   ebN*nPointsPerElementBoundary  + k];
		  dfavg[I]+=-rho2*DKW_DpsiC*gravity[I]*dV[eN*nElementBoundaries_element*nPointsPerElementBoundary + 
							  ebN*nPointsPerElementBoundary  + k]; 
		  J = I; /*assume isotropic*/
		  aavg[I][J]  += rho*KW*dV[eN*nElementBoundaries_element*nPointsPerElementBoundary + 
					   ebN*nPointsPerElementBoundary  + k];
		  daavg[I][J] +=-rho*DKW_DpsiC*dV[eN*nElementBoundaries_element*nPointsPerElementBoundary + 
						  ebN*nPointsPerElementBoundary  + k];
		  
		  df[eN*nElementBoundaries_element*nPointsPerElementBoundary*nSpace + ebN*nPointsPerElementBoundary*nSpace  + 
		     k*nSpace + I] = -rho2*DKW_DpsiC*gravity[I];
		  da[eN*nElementBoundaries_element*nPointsPerElementBoundary*nSpace2+ ebN*nPointsPerElementBoundary*nSpace2 + 
		     k*nSpace2 + I*nSpace + J]= -rho*DKW_DpsiC;
		  
		}
	    }/*end k 1*/
	  assert(vol > 0.0);
	  /*mwf debug
	    printf("eN=%d vol=%g mavg=%g dmavg=%g favg=[%g,%g,%g] aavg=[%g,%g,%g] \n",
	    eN,vol,mavg,dmavg,favg[0],favg[1],favg[2],aavg[0][0],aavg[1][1],aavg[2][2]);
	  */
	  for (k=0; k < nPointsPerElementBoundary; k++)
	    {
	      mass[eN*nElementBoundaries_element*nPointsPerElementBoundary + ebN*nPointsPerElementBoundary  + k] = mavg/vol; 
	      /*dmass[eN*nElementBoundaries_element*nPointsPerElementBoundary + ebN*nPointsPerElementBoundary + k] = 
		dmavg/vol;*/ 
	      for (I=0; I < nSpace; I++)
		{
		  f[eN*nElementBoundaries_element*nPointsPerElementBoundary*nSpace  + ebN*nPointsPerElementBoundary*nSpace  +
		    k*nSpace + I] = favg[I]/vol;
		  /*df[eN*nElementBoundaries_element*nPointsPerElementBoundary*nSpace + ebN*nPointsPerElementBoundary*nSpace  +
		    k*nSpace + I] = dfavg[I]/vol;*/
		  for (J=0; J < nSpace; J++)
		    {
		      /*assume diagonal for now*/
		      if (J == I)
			{
			  a[eN*nElementBoundaries_element*nPointsPerElementBoundary*nSpace2 + ebN*nPointsPerElementBoundary*nSpace2 + 
			    k*nSpace2 + I*nSpace + J] = aavg[I][J]/vol;
			  /*da[eN*nElementBoundaries_element*nPointsPerElementBoundary*nSpace2+ 
			    ebN*nPointsPerElementBoundary*nSpace2 + 
			    k*nSpace2 + I*nSpace + J]= daavg[I][J]/vol;*/

			}
		      else
			{
			  a[eN*nElementBoundaries_element*nPointsPerElementBoundary*nSpace2 + ebN*nPointsPerElementBoundary*nSpace2 + 
			    k*nSpace2 + I*nSpace + J] = 0.0;
			  /*da[eN*nElementBoundaries_element*nPointsPerElementBoundary*nSpace2+ 
			    ebN*nPointsPerElementBoundary*nSpace2 + 
			    k*nSpace2 + I*nSpace + J]= 0.0;*/

			}
		    }
		}/*I*/
	    }/*k*/
	}/*ebN*/
    }/*eN*/
}
void conservativeHeadRichardsL2projMualemVanGenuchtenHetEvaluate(const int nSimplices,
								 const int nPointsPerSimplex,
								 const int nSpace,
								 const double rho,
								 const double *gravity,
								 const double *alpha,
								 const double *n,
								 const double *thetaR,
								 const double *thetaSR,
								 const double *KWs,
								 double *dV,
								 double *u,
								 double *mass,
								 double *dmass,
								 double *f,
								 double *df,
								 double *a,
								 double *da)
{
  proteus_index_t J; int k, I, eN;
  const int nSpace2=nSpace*nSpace;
  register double psiC,
    pcBar,pcBar_n,pcBar_nM1,pcBar_nM2,
    onePlus_pcBar_n,
    sBar,sqrt_sBar,DsBar_DpsiC,
    thetaW,DthetaW_DpsiC,
    vBar,vBar2,DvBar_DpsiC,
    KW,DKW_DpsiC,
    rho2=rho*rho;
  register double thetaS,m;

  double mavg,dmavg,vol;
  double favg[3]   = {0.0,0.0,0.0};
  double dfavg[3]  = {0.0,0.0,0.0};
  double aavg[3][3]= {{0.0,0.0,0.0},
		      {0.0,0.0,0.0},
		      {0.0,0.0,0.0}};
  double daavg[3][3]= {{0.0,0.0,0.0},
		       {0.0,0.0,0.0},
		       {0.0,0.0,0.0}};
  /*mwf debug
    printf("revgL2proj nSimplices=%d nPerSimp=%d nSpace=%d\n",nSimplices,nPointsPerSimplex,
    nSpace);
  */
  for (eN = 0; eN < nSimplices; eN++)
    {
      mavg = 0.0; dmavg = 0.0; vol = 0.0;
      for (I=0; I < nSpace; I++)
	{
	  favg[I] =0.0; dfavg[I] = 0.0;
	  for (J=0; J < nSpace;J++)
	    {
	      aavg[I][J] = 0.0; daavg[I][J] = 0.0;
	    }
	}
      for (k=0;k<nPointsPerSimplex;k++)
	{
	  psiC = -u[eN*nPointsPerSimplex + k];
	  m = 1.0 - 1.0/n[eN*nPointsPerSimplex + k];
	  thetaS = thetaR[eN*nPointsPerSimplex + k] + thetaSR[eN*nPointsPerSimplex + k];
	  if (psiC > 0.0)
	    {
	      pcBar = alpha[eN*nPointsPerSimplex + k]*psiC;
	      pcBar_nM2 = pow(pcBar,n[eN*nPointsPerSimplex + k]-2);
	      pcBar_nM1 = pcBar_nM2*pcBar;
	      pcBar_n   = pcBar_nM1*pcBar;
	      onePlus_pcBar_n = 1.0 + pcBar_n;
          
	      sBar = pow(onePlus_pcBar_n,-m);
	      /* using -mn = 1-n */
	      DsBar_DpsiC = alpha[eN*nPointsPerSimplex + k]*(1.0-n[eN*nPointsPerSimplex + k])*(sBar/onePlus_pcBar_n)*pcBar_nM1;
          
	      vBar = 1.0-pcBar_nM1*sBar;
	      vBar2 = vBar*vBar;
	      DvBar_DpsiC = -alpha[eN*nPointsPerSimplex + k]*(n[eN*nPointsPerSimplex + k]-1.0)*pcBar_nM2*sBar - pcBar_nM1*DsBar_DpsiC;
          
	      thetaW = thetaSR[eN*nPointsPerSimplex + k]*sBar + thetaR[eN*nPointsPerSimplex + k];
	      DthetaW_DpsiC = thetaSR[eN*nPointsPerSimplex + k] * DsBar_DpsiC; 
          
	      sqrt_sBar = sqrt(sBar);
	      KW= KWs[eN*nPointsPerSimplex + k]*sqrt_sBar*vBar2;
	      DKW_DpsiC= KWs[eN*nPointsPerSimplex + k]*
		((0.5/sqrt_sBar)*DsBar_DpsiC*vBar2
		 +
		 2.0*sqrt_sBar*vBar*DvBar_DpsiC);
	    }
	  else
	    {
	      thetaW        = thetaS;
	      DthetaW_DpsiC = 0.0;
	      KW            = KWs[eN*nPointsPerSimplex + k]; 
	      DKW_DpsiC     = 0.0;
	    }
	  /*mwf debug
	    printf("revgmL2 eN=%d k=%d psiC=%g thetaW=%g KW=%g rho=%g DKW_DpsiC=%g\n",
	    eN,k,psiC,thetaW,KW,rho,DKW_DpsiC);
	  */ 
	  vol   += dV[eN*nPointsPerSimplex + k];
	  mavg  += rho*thetaW*dV[eN*nPointsPerSimplex + k];
	  dmavg +=-rho*DthetaW_DpsiC*dV[eN*nPointsPerSimplex + k];
	  /*go ahead and assign point values for deriv terms since this is right thing to do
	    for stiffness terms even though it's not right for mass terms*/
	  dmass[eN*nPointsPerSimplex+ k] = -rho*DthetaW_DpsiC;
	  for (I=0; I < nSpace; I++)
	    {
	      favg[I] += rho2*KW*gravity[I]*dV[eN*nPointsPerSimplex + k];
	      dfavg[I]+=-rho2*DKW_DpsiC*gravity[I]*dV[eN*nPointsPerSimplex + k]; 

	      df[eN*nPointsPerSimplex*nSpace+ k*nSpace + I] = -rho2*DKW_DpsiC*gravity[I];
	      J=I;
	      aavg[I][J]  += rho*KW*dV[eN*nPointsPerSimplex + k];
	      daavg[I][J] +=-rho*DKW_DpsiC*dV[eN*nPointsPerSimplex + k];
	      da[eN*nPointsPerSimplex*nSpace2 + k*nSpace2 + I*nSpace + I] = -rho*DKW_DpsiC;
	    }
	}/*end k 1*/
      assert(vol > 0.0);
      /*mwf debug
	printf("eN=%d vol=%g mavg=%g dmavg=%g favg=[%g,%g,%g] aavg=[%g,%g,%g] \n",
	eN,vol,mavg,dmavg,favg[0],favg[1],favg[2],aavg[0][0],aavg[1][1],aavg[2][2]);
      */
      for (k=0; k < nPointsPerSimplex; k++)
	{
	  mass[eN*nPointsPerSimplex + k] = mavg/vol; 
	  /*dmass[eN*nPointsPerSimplex+ k] = dmavg/vol;*/ 
	  for (I=0; I < nSpace; I++)
	    {
	      /*assume diagonal*/
	      f[eN*nPointsPerSimplex*nSpace + k*nSpace + I] = favg[I]/vol;
	      /*df[eN*nPointsPerSimplex*nSpace+ k*nSpace + I] = dfavg[I]/vol;*/
	      a[eN*nPointsPerSimplex*nSpace2 + k*nSpace2 + I*nSpace + I] = aavg[I][I]/vol;
	      /*da[eN*nPointsPerSimplex*nSpace2 + k*nSpace2 + I*nSpace + I]= daavg[I][I]/vol;*/
	    }/*I*/
        }/*k*/
    }
}

/** Coefficients for the mass conservative total head-based (\f[ h=\psi+z \f]) Richards' equation using Mualem-Van Genuchten.
 */
void conservativeTotalHeadRichardsMualemVanGenuchtenHomEvaluate(const int nPoints,
								const int nSpace,
								const double rho,
								const double* gravity,
								const double* x,
								const double alpha,
								const double n,
								const double m,
								const double thetaR,
								const double thetaSR,
								const double KWs,
								double *u,
								double *mass,
								double *dmass,
								double *f,
								double *df,
								double *a,
								double *da,
								double *phi,
								double *dphi)
{
  int k,I;
  const int nSpace2=nSpace*nSpace;
  register double psiC,
    pcBar,pcBar_n,pcBar_nM1,pcBar_nM2,
    onePlus_pcBar_n,
    sBar,sqrt_sBar,DsBar_DpsiC,
    thetaW,DthetaW_DpsiC,
    vBar,vBar2,DvBar_DpsiC,
    KW,DKW_DpsiC,
    thetaS=thetaR+thetaSR;
  /*mwf elevation */
  register double elev;
  for (k=0;k<nPoints;k++)
    {
      elev = 0.0;
      for (I=0; I < nSpace; I++)
	elev += gravity[I]*x[k*3+I];
      /*mwf if unknown is h
      psiC = -u[k]-elev;
      */
      /*mwf if unknown is psi*/
      psiC = -u[k];
      
      if (psiC > 0.0)
        {
          pcBar = alpha*psiC;
          pcBar_nM2 = pow(pcBar,n-2);
          pcBar_nM1 = pcBar_nM2*pcBar;
          pcBar_n   = pcBar_nM1*pcBar;
          onePlus_pcBar_n = 1.0 + pcBar_n;
          
          sBar = pow(onePlus_pcBar_n,-m);
          /* using -mn = 1-n */
          DsBar_DpsiC = alpha*(1.0-n)*(sBar/onePlus_pcBar_n)*pcBar_nM1;
          
          vBar = 1.0-pcBar_nM1*sBar;
          vBar2 = vBar*vBar;
          DvBar_DpsiC = -alpha*(n-1.0)*pcBar_nM2*sBar - pcBar_nM1*DsBar_DpsiC;
          
          thetaW = thetaSR*sBar + thetaR;
          DthetaW_DpsiC = thetaSR * DsBar_DpsiC; 
          
          sqrt_sBar = sqrt(sBar);
          KW= KWs*sqrt_sBar*vBar2;
          DKW_DpsiC= KWs*
            ((0.5/sqrt_sBar)*DsBar_DpsiC*vBar2
             +
             2.0*sqrt_sBar*vBar*DvBar_DpsiC);
        }
      else
        {
          thetaW        = thetaS;
          DthetaW_DpsiC = 0.0;
          KW            = KWs; 
          DKW_DpsiC     = 0.0;
        }
      mass[k] = rho*thetaW;
      dmass[k] = -rho*DthetaW_DpsiC;
      /* remember to turn on nonlinear potential if you want to use total head */
      /*mwf unknown is psi*/
      phi[k] = -psiC; 
      dphi[k] = 1.0; 
      
      /*mwf unknown is h
	phi[k] = u[k]; 
	dphi[k] = 1.0;
      */
      for (I=0;I<nSpace;I++)
        {
          f[k*nSpace+I]  = 0.0;
          df[k*nSpace+I] = 0.0;
	  /*mwf unknown is psi*/
	  phi[k] -= rho*gravity[I]*x[k*3+I];
	  
          a[k*nSpace2+I*nSpace+I]  = rho*KW;
          da[k*nSpace2+I*nSpace+I] = -rho*DKW_DpsiC;
        }
    }
}

void l2projectScalar(const int nSimplices,
		     const int nPointsPerSimplex,
		     double * dV,
		     double * r)
{
  int eN,k;
  double ravg,vol;

  for (eN = 0; eN < nSimplices; eN++)
    {
      ravg = 0.0; vol = 0.0;
      for (k=0; k < nPointsPerSimplex; k++)
	{
	  vol  += dV[eN*nPointsPerSimplex+k];
	  ravg += r[eN*nPointsPerSimplex + k]*dV[eN*nPointsPerSimplex+k];
	}

      assert(vol > 0.0);
      ravg = ravg / vol;
      for (k=0; k < nPointsPerSimplex; k++)
	{
	  r[eN*nPointsPerSimplex + k] = ravg;
	}
    }
}
void l2projectVector(const int nSimplices,
		     const int nPointsPerSimplex,
		     const int nSpace,
		     double * dV,
		     double * r)
{
  int eN,k,I;
  double vol;
  /*need different max size, take a chance on variable length array?*/
  double ravg[nSpace];
  for (eN = 0; eN < nSimplices; eN++)
    {
      vol = 0.0;
      for (I=0; I < nSpace; I++)
	ravg[I] = 0.0;
      for (k=0; k < nPointsPerSimplex; k++)
	{
	  vol  += dV[eN*nPointsPerSimplex+k];
	  for (I=0; I < nSpace; I++)
	    ravg[I] += r[eN*nPointsPerSimplex*nSpace + k*nSpace + I]*dV[eN*nPointsPerSimplex+k];
	}
      assert(vol > 0.0);
      for (k=0; k < nPointsPerSimplex; k++)
	{
	  for (I=0; I < nSpace; I++)
	    r[eN*nPointsPerSimplex*nSpace + k*nSpace + I] = ravg[I]/vol;
	}
    }
}
void l2project2Tensor(const int nSimplices,
		      const int nPointsPerSimplex,
		      const int nSpace,
		      double * dV,
		      double * r)
{
  proteus_index_t J; int eN, k, I, nSpace2;
  double vol;
  /*need different max size, take a chance on variable length array?*/
  double ravg[nSpace*nSpace];
  nSpace2 = nSpace*nSpace;
  for (eN = 0; eN < nSimplices; eN++)
    {
      vol = 0.0;
      for (I=0; I < nSpace2; I++)
	ravg[I] = 0.0;
      for (k=0; k < nPointsPerSimplex; k++)
	{
	  vol  += dV[eN*nPointsPerSimplex+k];
	  for (I=0; I < nSpace; I++)
	    for (J=0; J < nSpace; J++)
	      ravg[I*nSpace+J] += r[eN*nPointsPerSimplex*nSpace2 + k*nSpace2 + I*nSpace + J]
		*
		dV[eN*nPointsPerSimplex+k];
	}
      assert(vol > 0.0);
      for (k=0; k < nPointsPerSimplex; k++)
	{
	  for (I=0; I < nSpace; I++)
	    for (J=0; J < nSpace; J++)
	      r[eN*nPointsPerSimplex*nSpace2 + k*nSpace2 + I*nSpace + J] = ravg[I*nSpace+J]/vol;
	}
    }
}

void conservativeHeadRichardsMualemVanGenuchten_sd_het(const int nSimplex,
						       const int nPointsPerSimplex,
						       const int nSpace,
						       double pc_eps,
						       const proteus_csr_index_t * rowptr,
						       const proteus_csr_index_t * colind,
						       const int* materialTypes,
						       const double rho,
						       const double beta,
						       const double* gravity,
						       const double* alpha,
						       const double* n,
						       const double* thetaR,
						       const double* thetaSR,
						       const double* KWs,
						       double *u,
						       double *mass,
						       double *dmass,
						       double *f,
						       double *df,
						       double *a,
						       double *da,
						       double* vol_frac)
{
  proteus_csr_index_t ii; int i, j, k, I, matID;
  const int nSpace2=nSpace*nSpace;
  register double psiC,pcBarStar,
    pcBar,pcBar_n,pcBar_nM1,pcBar_nM2,
    onePlus_pcBar_n,
    sBar,sqrt_sBar,DsBar_DpsiC,
    thetaW,DthetaW_DpsiC,
    vBar,vBar2,DvBar_DpsiC,
    KWr,DKWr_DpsiC,
    rho2=rho*rho,
    thetaS,
    rhom,drhom,m,
    betauStar;
  const proteus_csr_index_t nnz = rowptr[nSpace];
  for (i=0; i < nSimplex; i++)
    {
      matID= materialTypes[i];
      for (j=0;j<nPointsPerSimplex;j++)
	{
	  k = i*nPointsPerSimplex + j;
	  psiC = -u[k];
	  m = 1.0 - 1.0/n[matID];
	  thetaS = thetaR[matID] + thetaSR[matID];
	  if (psiC > 0.0)
	    {
	      pcBar = alpha[matID]*psiC;
	      pcBarStar = fmax(pcBar,pc_eps);

	      pcBar_nM2 = pow(pcBarStar,n[matID]-2);
	      pcBar_nM1 = pcBar_nM2*pcBar;
	      pcBar_n   = pcBar_nM1*pcBar;
	      onePlus_pcBar_n = 1.0 + pcBar_n;
	      
	      sBar = pow(onePlus_pcBar_n,-m);
	      /* using -mn = 1-n */
	      DsBar_DpsiC = alpha[matID]*(1.0-n[matID])*(sBar/onePlus_pcBar_n)*pcBar_nM1;
	      
	      vBar = 1.0-pcBar_nM1*sBar;
	      vBar2 = vBar*vBar;
	      DvBar_DpsiC = -alpha[matID]*(n[matID]-1.0)*pcBar_nM2*sBar - pcBar_nM1*DsBar_DpsiC;

	      thetaW = thetaSR[matID]*sBar + thetaR[matID];
	      DthetaW_DpsiC = thetaSR[matID] * DsBar_DpsiC; 
          
	      sqrt_sBar = sqrt(sBar);
	      KWr= sqrt_sBar*vBar2;
	      DKWr_DpsiC= ((0.5/sqrt_sBar)*DsBar_DpsiC*vBar2
			   +
			   2.0*sqrt_sBar*vBar*DvBar_DpsiC);
	    }
	  else
	    {
	      thetaW        = thetaS;
	      DthetaW_DpsiC = 0.0;
	      KWr           = 1.0;
	      DKWr_DpsiC    = 0.0;
	    }
          //slight compressibility
	  betauStar = fmin(beta*u[k],1000.0);
          rhom = rho*exp(betauStar);
          drhom = beta*rhom;
          mass[k] = rhom*thetaW;
          dmass[k] = -rhom*DthetaW_DpsiC+drhom*thetaW;
	  vol_frac[k] = thetaW;
	  //mass[k] = rho*thetaW;
	  //dmass[k] = -rho*DthetaW_DpsiC;
	  for (I=0;I<nSpace;I++)
	    {
              f[k*nSpace+I] = 0.0;
              df[k*nSpace+I] = 0.0;
	      for (ii=rowptr[I]; ii < rowptr[I+1]; ii++)
		{
                  f[k*nSpace+I]  += rho2*KWr*KWs[matID*nnz+ii]*gravity[colind[ii]];
                  df[k*nSpace+I] += -rho2*DKWr_DpsiC*KWs[matID*nnz+ii]*gravity[colind[ii]];
		  a[k*nnz+ii]  = rho*KWr*KWs[matID*nnz+ii];
		  da[k*nnz+ii] = -rho*DKWr_DpsiC*KWs[matID*nnz+ii];
		}/*m*/
	    }/*I*/
	}/*k*/
    }/*j*/

}

void conservativeHeadRichardsMualemVanGenuchten_sd_het_linearized_at_saturation(const int nSimplex,
										const int nPointsPerSimplex,
										const int nSpace,
										double linear_break,
										const proteus_csr_index_t * rowptr,
										const proteus_csr_index_t * colind,
										const int* materialTypes,
										const double rho,
										const double beta,
										const double* gravity,
										const double* alpha,
										const double* n,
										const double* thetaR,
										const double* thetaSR,
										const double* KWs,
										double *u,
										double *mass,
										double *dmass,
										double *f,
										double *df,
										double *a,
										double *da,
										double* vol_frac)
{
  proteus_csr_index_t ii; int i, j, k, I, matID;
  const int nSpace2=nSpace*nSpace;
  register double psiC,pcBarStar,sBarStar,KWrStar,
    pcBar,pcBar_n,pcBar_nM1,pcBar_nM2,
    onePlus_pcBar_n,
    sBar,sqrt_sBar,DsBar_DpsiC,
    thetaW,DthetaW_DpsiC,
    vBar,vBar2,DvBar_DpsiC,
    KWr,DKWr_DpsiC,
    rho2=rho*rho,
    thetaS,
    rhom,drhom,m,
    betauStar;
  const proteus_csr_index_t nnz = rowptr[nSpace];
  pcBarStar = linear_break;
  
  for (i=0; i < nSimplex; i++)
    {
      matID= materialTypes[i];
      for (j=0;j<nPointsPerSimplex;j++)
	{
	  k = i*nPointsPerSimplex + j;
	  psiC = -u[k];
	  m = 1.0 - 1.0/n[matID];
	  thetaS = thetaR[matID] + thetaSR[matID];
	  if (psiC > 0.0)
	    {
	      pcBar = alpha[matID]*psiC;
	      if (psiC <= pcBarStar)
		{
		  /*calculate regularization parameters again because n varies*/
		  pcBar_nM2 = pow(pcBarStar,n[matID]-2.);
		  pcBar_nM1 = pcBar_nM2*pcBar;
		  pcBar_n   = pcBar_nM1*pcBar;
		  onePlus_pcBar_n = 1.0 + pcBar_n;
	      
		  sBar = pow(onePlus_pcBar_n,-m);
		  /* using -mn = 1-n */
		  DsBar_DpsiC = alpha[matID]*(1.0-n[matID])*(sBar/onePlus_pcBar_n)*pcBar_nM1;
	      
		  vBar = 1.0-pcBar_nM1*sBar;
		  vBar2 = vBar*vBar;
		  DvBar_DpsiC = -alpha[matID]*(n[matID]-1.0)*pcBar_nM2*sBar - pcBar_nM1*DsBar_DpsiC;

		  sBarStar = sBar;
		  sqrt_sBar = sqrt(sBar);
		  KWrStar= sqrt_sBar*vBar2;
	      
		  /*now compute linearized solution*/
		  DsBar_DpsiC = (sBarStar-1.0)/(pcBarStar-0.0);
		  sBar = DsBar_DpsiC*(psiC-0.0) + 1.0;
		  thetaW = thetaSR[matID]*sBar + thetaR[matID];
		  DthetaW_DpsiC = thetaSR[matID] * DsBar_DpsiC; 
  
		  DKWr_DpsiC= (KWrStar - 1.0)/(pcBarStar-0.0);
		  KWr = DKWr_DpsiC*(psiC-0.0) + 1.0;
		}
	      else
		{
		  pcBar = alpha[matID]*psiC;
		  pcBarStar = fmax(pcBar,1.0e-8);
		  
		  pcBar_nM2 = pow(pcBarStar,n[matID]-2);
		  pcBar_nM1 = pcBar_nM2*pcBar;
		  pcBar_n   = pcBar_nM1*pcBar;
		  onePlus_pcBar_n = 1.0 + pcBar_n;
		  
		  sBar = pow(onePlus_pcBar_n,-m);
		  /* using -mn = 1-n */
		  DsBar_DpsiC = alpha[matID]*(1.0-n[matID])*(sBar/onePlus_pcBar_n)*pcBar_nM1;
		  
		  vBar = 1.0-pcBar_nM1*sBar;
		  vBar2 = vBar*vBar;
		  DvBar_DpsiC = -alpha[matID]*(n[matID]-1.0)*pcBar_nM2*sBar - pcBar_nM1*DsBar_DpsiC;
		  
		  thetaW = thetaSR[matID]*sBar + thetaR[matID];
		  DthetaW_DpsiC = thetaSR[matID] * DsBar_DpsiC; 
		  
		  sqrt_sBar = sqrt(sBar);
		  KWr= sqrt_sBar*vBar2;
		  DKWr_DpsiC= ((0.5/sqrt_sBar)*DsBar_DpsiC*vBar2
			       +
			       2.0*sqrt_sBar*vBar*DvBar_DpsiC);

		}
	    }
	  else
	    {
	      thetaW        = thetaS;
	      DthetaW_DpsiC = 0.0;
	      KWr           = 1.0;
	      DKWr_DpsiC    = 0.0;
	    }
          //slight compressibility
	  betauStar = fmin(beta*u[k],1000.0);
          rhom = rho*exp(betauStar);
          drhom = beta*rhom;
          mass[k] = rhom*thetaW;
          dmass[k] = -rhom*DthetaW_DpsiC+drhom*thetaW;
	  vol_frac[k] = thetaW;
	  //mass[k] = rho*thetaW;
	  //dmass[k] = -rho*DthetaW_DpsiC;
	  for (I=0;I<nSpace;I++)
	    {
              f[k*nSpace+I] = 0.0;
              df[k*nSpace+I] = 0.0;
	      for (ii=rowptr[I]; ii < rowptr[I+1]; ii++)
		{
                  f[k*nSpace+I]  += rho2*KWr*KWs[matID*nnz+ii]*gravity[colind[ii]];
                  df[k*nSpace+I] += -rho2*DKWr_DpsiC*KWs[matID*nnz+ii]*gravity[colind[ii]];
		  a[k*nnz+ii]  = rho*KWr*KWs[matID*nnz+ii];
		  da[k*nnz+ii] = -rho*DKWr_DpsiC*KWs[matID*nnz+ii];
		}/*m*/
	    }/*I*/
	}/*k*/
    }/*j*/

}

void conservativeHeadRichardsMualemVanGenuchtenHetEvaluateV2(const int nSimplex,
							     const int nPointsPerSimplex,
							     const int nSpace,
							     const int* materialTypes,
							     const double rho,
                                                             const double beta,
							     const double* gravity,
							     const double* alpha,
							     const double* n,
							     const double* thetaR,
							     const double* thetaSR,
							     const double* KWs,
							     double *u,
							     double *mass,
							     double *dmass,
							     double *f,
							     double *df,
							     double *a,
							     double *da)
{
  int i,j,k,I,matID;
  const int nSpace2=nSpace*nSpace;
  register double psiC,
    pcBar,pcBar_n,pcBar_nM1,pcBar_nM2,
    onePlus_pcBar_n,
    sBar,sqrt_sBar,DsBar_DpsiC,
    thetaW,DthetaW_DpsiC,
    vBar,vBar2,DvBar_DpsiC,
    KW,DKW_DpsiC,
    rho2=rho*rho,
    thetaS,
    rhom,drhom,m;
  for (i=0; i < nSimplex; i++)
    {
      matID= materialTypes[i];
      for (j=0;j<nPointsPerSimplex;j++)
	{
	  k = i*nPointsPerSimplex + j;
	  psiC = -u[k];
	  m = 1.0 - 1.0/n[matID];
	  thetaS = thetaR[matID] + thetaSR[matID];
	  if (psiC > 0.0)
	    {
	      pcBar = alpha[matID]*psiC;
	      pcBar_nM2 = pow(pcBar,n[matID]-2);
	      pcBar_nM1 = pcBar_nM2*pcBar;
	      pcBar_n   = pcBar_nM1*pcBar;
	      onePlus_pcBar_n = 1.0 + pcBar_n;
	      
	      sBar = pow(onePlus_pcBar_n,-m);
	      /* using -mn = 1-n */
	      DsBar_DpsiC = alpha[matID]*(1.0-n[matID])*(sBar/onePlus_pcBar_n)*pcBar_nM1;
	      
	      vBar = 1.0-pcBar_nM1*sBar;
	      vBar2 = vBar*vBar;
	      DvBar_DpsiC = -alpha[matID]*(n[matID]-1.0)*pcBar_nM2*sBar - pcBar_nM1*DsBar_DpsiC;

	      thetaW = thetaSR[matID]*sBar + thetaR[matID];
	      DthetaW_DpsiC = thetaSR[matID] * DsBar_DpsiC; 
          
	      sqrt_sBar = sqrt(sBar);
	      KW= KWs[matID]*sqrt_sBar*vBar2;
	      DKW_DpsiC= KWs[matID]*
		((0.5/sqrt_sBar)*DsBar_DpsiC*vBar2
		 +
		 2.0*sqrt_sBar*vBar*DvBar_DpsiC);
	    }
	  else
	    {
	      thetaW        = thetaS;
	      DthetaW_DpsiC = 0.0;
	      KW            = KWs[matID]; 
	      DKW_DpsiC     = 0.0;
	    }
          //slight compressibility
          rhom = rho*exp(beta*u[k]);
          drhom = beta*rhom;
          mass[k] = rhom*thetaW;
          dmass[k] = -rhom*DthetaW_DpsiC+drhom*thetaW;
	  //mass[k] = rho*thetaW;
	  //dmass[k] = -rho*DthetaW_DpsiC;
	  for (I=0;I<nSpace;I++)
	    {
	      f[k*nSpace+I]  = rho2*KW*gravity[I];
	      df[k*nSpace+I] = -rho2*DKW_DpsiC*gravity[I];
	      a[k*nSpace2+I*nSpace+I]  = rho*KW;
	      da[k*nSpace2+I*nSpace+I] = -rho*DKW_DpsiC;
	    }/*I*/
	}/*k*/
    }/*j*/

}

void seepageBrezis(const int nSimplex,
		   const int nPointsPerSimplex,
		   const int nSpace,
		   const int* materialTypes,
		   const double epsFact,
		   const double rho,
		   const double beta,
		   const double* elementDiameter,
		   const double* gravity,
		   const double* alpha,
		   const double* n,
		   const double* thetaR,
		   const double* thetaSR,
		   const double* KWs,
		   double *u,
		   double *mass,
		   double *dmass,
		   double *f,
		   double *df,
		   double *a,
		   double *da)
{
  int i,j,k,I,matID;
  const int nSpace2=nSpace*nSpace;
  register double psiC,
    pcBar,pcBar_n,pcBar_nM1,pcBar_nM2,
    onePlus_pcBar_n,
    sBar,sqrt_sBar,DsBar_DpsiC,
    thetaW,DthetaW_DpsiC,
    vBar,vBar2,DvBar_DpsiC,
    KW,DKW_DpsiC,
    rho2=rho*rho,
    thetaS,
    rhom,drhom,m,eps,hStar,dhStar;
  for (i=0; i < nSimplex; i++)
    {
      matID= materialTypes[i];
      eps = epsFact*elementDiameter[i];
      for (j=0;j<nPointsPerSimplex;j++)
	{
	  k = i*nPointsPerSimplex + j;
	  thetaS = thetaR[matID] + thetaSR[matID];
	  /* hStar = linearHeaviside(eps,u[k]+eps)*(1.0-1.0e-5)+1.0e-5; */
	  /* dhStar = linearDirac(eps,u[k]+eps)*(1.0-1.0e-5); */
	  hStar = smoothedHeaviside(eps,u[k]+eps)*(1.0-1.0e-5)+1.0e-5;
	  dhStar = smoothedDirac(eps,u[k]+eps)*(1.0-1.0e-5);
          mass[k] = thetaS*hStar;
          dmass[k] = thetaS*dhStar;
	  for (I=0;I<nSpace;I++)
	    {
	      f[k*nSpace+I]  = KWs[matID]*gravity[I]*hStar;
	      df[k*nSpace+I] = KWs[matID]*gravity[I]*dhStar;
	      a[k*nSpace2+I*nSpace+I]  = KWs[matID]*hStar;
	      da[k*nSpace2+I*nSpace+I] = KWs[matID]*dhStar;
	    }/*I*/
	}/*k*/
    }/*j*/

}

void conservativeHeadRichardsJLeverett(const int nSimplex,
                                       const int nPointsPerSimplex,
                                       const int nSpace,
                                       const int* materialTypes,
                                       const double rho,
                                       const double beta,
                                       const double* gravity,
                                       const double* phi,
                                       const double* psiD,
                                       const double* ns,
                                       const double* nk,
                                       const double* S_wirr,
                                       const double* S_nwr,
                                       const double* kr0,
                                       double *u,
                                       double *mass,
                                       double *dmass,
                                       double *f,
                                       double *df,
                                       double *a,
                                       double *da)
{
  int i,j,k,I,matID;
  const int nSpace2=nSpace*nSpace;
  register double psiC,
    Se,Sw,dSw,
    kr,dkr,
    rho2=rho*rho,
    rhom,drhom;
  const double reg_diff=1.0e-2;
  for (i=0; i < nSimplex; i++)
    {
      matID= materialTypes[i];
      for (j=0;j<nPointsPerSimplex;j++)
	{
	  k = i*nPointsPerSimplex + j;
	  psiC = -u[k];
	  if (psiC > 0.0 && psiC < psiD[matID])
	    {
              Se = 1.0 - pow(psiC/psiD[matID],ns[matID]);
              Sw = Se*(1.0 - S_wirr[matID] - S_nwr[matID]) + S_wirr[matID];
              dSw = (1.0 - S_wirr[matID] - S_nwr[matID])*pow(psiC/psiD[matID],ns[matID]-1.0)*ns[matID]/psiD[matID];   
              kr = kr0[matID]*pow(Se,nk[matID])+reg_diff;
              dkr = kr0[matID]*pow(Se,nk[matID]-1.0)*nk[matID]*pow(psiC/psiD[matID],ns[matID]-1.0)*ns[matID]/psiD[matID];
	    }
	  else if (psiC <= 0.0)
	    {
              Se = 1.0;
              Sw = 1.0 - S_nwr[matID];
              dSw = 0.0;
              kr = kr0[matID]+reg_diff;
              dkr = 0.0;
	    }
          else
            {
              Se = 0.0;
              Sw = S_wirr[matID];
              dSw = 0.0;
              kr = reg_diff;
              dkr = 0.0;
            }
          //slight compressibility
          rhom = rho*exp(beta*u[k]);
          drhom = beta*rhom;

          mass[k] = rhom*Sw*phi[matID];
          dmass[k] = rhom*dSw*phi[matID]+drhom*Sw*phi[matID];
	  for (I=0;I<nSpace;I++)
	    {
	      f[k*nSpace+I]  = rho2*kr*gravity[I];
	      df[k*nSpace+I] = rho2*dkr*gravity[I];
	      a[k*nSpace2+I*nSpace+I]  = rho*kr;
	      da[k*nSpace2+I*nSpace+I] = rho*dkr;
	    }/*I*/
	}/*k*/
    }/*j*/
}
void conservativeHeadRichardsJLeverettAni(const int nSimplex,
                                       const int nPointsPerSimplex,
                                       const int nSpace,
                                       const int* materialTypes,
                                       const double rho,
                                       const double beta,
                                       const double* gravity,
                                       const double* phi,
                                       const double* psiD,
                                       const double* ns,
                                       const double* nk,
                                       const double* S_wirr,
                                       const double* S_nwr,
                                       const double* kr0x,
                                       const double* kr0y,
                                       const double* kr0z,
                                       double *u,
                                       double *mass,
                                       double *dmass,
                                       double *f,
                                       double *df,
                                       double *a,
                                       double *da)
{
  int i,j,k,I,matID;
  const int nSpace2=nSpace*nSpace;
  register double psiC,pcBar,
    Se,dSe,Sw,dSw,
    kr,dkr,
    rho2=rho*rho,
    rhom,drhom;
  const double reg_diff=1.0e-2;
  for (i=0; i < nSimplex; i++)
    {
      matID= materialTypes[i];
      for (j=0;j<nPointsPerSimplex;j++)
	{
	  k = i*nPointsPerSimplex + j;
	  psiC = -u[k];
          if (psiC >= psiD[matID])
            {
              pcBar = psiC/psiD[matID];
              Se = pow(pcBar, -ns[matID]);
              Sw = Se*(1.0 - S_wirr[matID] - S_nwr[matID]) + S_wirr[matID];
              dSe = ns[matID]*Se/(pcBar*psiD[matID]);
              dSw = (1.0 - S_wirr[matID] - S_nwr[matID])*dSe;
              kr = pow(Se, (2.0+3.0*ns[matID])/ns[matID]);
              dkr = ((2.0+3.0*ns[matID])/ns[matID]*pow(Se, 2.0/ns[matID]*(1.0+ns[matID]))*dSe);
              kr = pow(Se,nk[matID]);
              dkr = nk[matID]*pow(Se,nk[matID]-1)*dSe;
            }
          else
            {
              Se = 1.0;
              Sw = 1.0 - S_nwr[matID];
              dSw = 0.0;
              kr = 1.0;
              dkr = 0.0;
            }
/* 	  if (psiC > 0.0 && psiC < psiD[matID]) */
/* 	    { */
/*               Se = 1.0 - pow(psiC/psiD[matID],ns[matID]); */
/*               Sw = Se*(1.0 - S_wirr[matID] - S_nwr[matID]) + S_wirr[matID]; */
/*               dSw = (1.0 - S_wirr[matID] - S_nwr[matID])*pow(psiC/psiD[matID],ns[matID]-1.0)*ns[matID]/psiD[matID];    */
/*               kr = pow(Se,nk[matID])+reg_diff; */
/*               dkr = pow(Se,nk[matID]-1.0)*nk[matID]*pow(psiC/psiD[matID],ns[matID]-1.0)*ns[matID]/psiD[matID]; */
/* 	    } */
/* 	  else if (psiC <= 0.0) */
/* 	    { */
/*               Se = 1.0; */
/*               Sw = 1.0 - S_nwr[matID]; */
/*               dSw = 0.0; */
/*               kr = 1.0+reg_diff; */
/*               dkr = 0.0; */
/* 	    } */
/*           else */
/*             { */
/*               Se = 0.0; */
/*               Sw = S_wirr[matID]; */
/*               dSw = 0.0; */
/*               kr = reg_diff; */
/*               dkr = 0.0; */
/*             } */
          //slight compressibility
          rhom = rho*exp(beta*u[k]);
          drhom = beta*rhom;

          mass[k] = rhom*Sw*phi[matID];
          dmass[k] = rhom*dSw*phi[matID]+drhom*Sw*phi[matID];
          I=0;
          f[k*nSpace+I]  = rho2*kr0x[matID]*kr*gravity[I];
          df[k*nSpace+I] = rho2*kr0x[matID]*dkr*gravity[I];
          a[k*nSpace2+I*nSpace+I]  = rho*kr0x[matID]*kr;
          da[k*nSpace2+I*nSpace+I] = rho*kr0x[matID]*dkr;
          if (nSpace > 1)
            {
              I=1;
              f[k*nSpace+I]  = rho2*kr0y[matID]*kr*gravity[I];
              df[k*nSpace+I] = rho2*kr0y[matID]*dkr*gravity[I];
              a[k*nSpace2+I*nSpace+I]  = rho*kr0y[matID]*kr;
              da[k*nSpace2+I*nSpace+I] = rho*kr0y[matID]*dkr;
              if (nSpace > 2)
                {
                  I=2;
                  f[k*nSpace+I]  = rho2*kr0z[matID]*kr*gravity[I];
                  df[k*nSpace+I] = rho2*kr0z[matID]*dkr*gravity[I];
                  a[k*nSpace2+I*nSpace+I]  = rho*kr0z[matID]*kr;
                  da[k*nSpace2+I*nSpace+I] = rho*kr0z[matID]*dkr;
                }
            }
	}/*k*/
    }/*j*/
}


/* mwf end unnecessary additions */

void conservativeHeadRichardsMualemVanGenuchtenHetEvaluate(const int nPoints,
                                                           const int nSpace,
                                                           const double rho,
                                                           const double* gravity,
                                                           const double* alpha,
                                                           const double* n,
                                                           const double* thetaR,
                                                           const double* thetaSR,
                                                           const double* KWs,
                                                           double *u,
                                                           double *mass,
                                                           double *dmass,
                                                           double *f,
                                                           double *df,
                                                           double *a,
                                                           double *da)
{
  int k,I;
  const int nSpace2=nSpace*nSpace;
  register double psiC,
    pcBar,pcBar_n,pcBar_nM1,pcBar_nM2,
    onePlus_pcBar_n,
    sBar,sqrt_sBar,DsBar_DpsiC,
    thetaW,DthetaW_DpsiC,
    vBar,vBar2,DvBar_DpsiC,
    KW,DKW_DpsiC,
    rho2=rho*rho,
    thetaS,
    m;
  for (k=0;k<nPoints;k++)
    {
      psiC = -u[k];
      m = 1.0 - 1.0/n[k];
      thetaS = thetaR[k] + thetaSR[k];
      if (psiC > 0.0)
        {
          pcBar = alpha[k]*psiC;
          pcBar_nM2 = pow(pcBar,n[k]-2);
          pcBar_nM1 = pcBar_nM2*pcBar;
          pcBar_n   = pcBar_nM1*pcBar;
          onePlus_pcBar_n = 1.0 + pcBar_n;
          
          sBar = pow(onePlus_pcBar_n,-m);
          /* using -mn = 1-n */
          DsBar_DpsiC = alpha[k]*(1.0-n[k])*(sBar/onePlus_pcBar_n)*pcBar_nM1;
          
          vBar = 1.0-pcBar_nM1*sBar;
          vBar2 = vBar*vBar;
          DvBar_DpsiC = -alpha[k]*(n[k]-1.0)*pcBar_nM2*sBar - pcBar_nM1*DsBar_DpsiC;
          
          thetaW = thetaSR[k]*sBar + thetaR[k];
          DthetaW_DpsiC = thetaSR[k] * DsBar_DpsiC; 
          
          sqrt_sBar = sqrt(sBar);
          KW= KWs[k]*sqrt_sBar*vBar2;
          DKW_DpsiC= KWs[k]*
            ((0.5/sqrt_sBar)*DsBar_DpsiC*vBar2
             +
             2.0*sqrt_sBar*vBar*DvBar_DpsiC);
        }
      else
        {
          thetaW        = thetaS;
          DthetaW_DpsiC = 0.0;
          KW            = KWs[k]; 
          DKW_DpsiC     = 0.0;
        }
      mass[k] = rho*thetaW;
      dmass[k] = -rho*DthetaW_DpsiC;
      for (I=0;I<nSpace;I++)
        {
          f[k*nSpace+I]  = rho2*KW*gravity[I];
          df[k*nSpace+I] = -rho2*DKW_DpsiC*gravity[I];
          a[k*nSpace2+I*nSpace+I]  = rho*KW;
          da[k*nSpace2+I*nSpace+I] = -rho*DKW_DpsiC;
        }
    }
}

/** Coefficients for the mass conservative saturation-based Richards' equation using Mualem-Van Genuchten.
 */
void conservativeSatRichardsMualemVanGenuchtenHomEvaluate(const int nPoints,
                                                          const int nSpace,
                                                          const double rho,
                                                          const double* gravity,
                                                          const double* x,
                                                          const double alpha,
                                                          const double n,
                                                          const double m,
                                                          const double thetaR,
                                                          const double thetaSR,
                                                          const double KWs,
                                                          double *u,
                                                          double *mass,
                                                          double *dmass,
                                                          double *f,
                                                          double *df,
                                                          double *a,
                                                          double *da,
                                                          double *phi,
                                                          double *dphi)
{
  int k,I;
  const int nSpace2=nSpace*nSpace;
  const double eps=1.0e-8;
  register double psiC,
    pcBar,pcBar_n,pcBar_nM1,pcBar_nM2,
    onePlus_pcBar_n,
    sBar,sqrt_sBar,DsBar_DpsiC,
    thetaW,
    vBar,vBar2,DvBar_DpsiC,
    KW,DKW_DpsiC,
    rho2=rho*rho,
    thetaS=thetaR+thetaSR;
  for (k=0;k<nPoints;k++)
    {
      sBar = u[k];
      thetaW = thetaS*u[k];
      if (u[k] < 1.0-eps &&
          u[k] > 0.0+eps)
        {
          /* piggy back  on head  based formulas */
          psiC = pow(pow(sBar,-1.0/m)-1.0,1.0/n)/alpha;
          
          pcBar = alpha*psiC;
          pcBar_nM2 = pow(pcBar,n-2);
          pcBar_nM1 = pcBar_nM2*pcBar;
          pcBar_n   = pcBar_nM1*pcBar;
          onePlus_pcBar_n = 1.0 + pcBar_n;
          
          /* using -mn = 1-n */
          DsBar_DpsiC = alpha*(1.0-n)*(sBar/onePlus_pcBar_n)*pcBar_nM1;
          
          vBar = 1.0-pcBar_nM1*sBar;
          vBar2 = vBar*vBar;
          DvBar_DpsiC = -alpha*(n-1.0)*pcBar_nM2*sBar - pcBar_nM1*DsBar_DpsiC;
          
          sqrt_sBar = sqrt(sBar);
          KW= KWs*sqrt_sBar*vBar2;
          DKW_DpsiC= KWs*
            ((0.5/sqrt_sBar)*DsBar_DpsiC*vBar2
             +
             2.0*sqrt_sBar*vBar*DvBar_DpsiC);
        }
      else if (u[k] >= 1.0-eps)
        {
          psiC = 0.0;
          DsBar_DpsiC = -1.0;
          KW            = KWs; 
          DKW_DpsiC     = 0.0;
        }
      else
        {
          psiC = pow(pow(eps,-1.0/m)-1.0,1.0/n)/alpha;
          DsBar_DpsiC = -1.0;
          KW            = 0.0;
          DKW_DpsiC     = 0.0;
        }
      mass[k] = rho*thetaW;
      dmass[k] = rho*thetaS;
      phi[k] = -psiC;
      dphi[k] = -1.0/DsBar_DpsiC;
      for (I=0;I<nSpace;I++)
        {
          f[k*nSpace+I]  = rho2*KW*gravity[I];
          df[k*nSpace+I] = rho2*DKW_DpsiC*gravity[I]/DsBar_DpsiC;
          /* calculate the total head the slow way for now */
/*           f[k*nSpace+I]  = 0.0; */
/*           df[k*nSpace+I] = 0.0; */
/*           phi[k] -= rho*gravity[I]*x[k*3+I]; */
          a[k*nSpace2+I*nSpace+I]  = rho*KW;
          da[k*nSpace2+I*nSpace+I] = rho*DKW_DpsiC/DsBar_DpsiC;
        }
/*       /\*  regularized diffusion  *\/ */
/*       phi[k] = -psiC+9.0e-1*u[k];; */
/*       dphi[k] = -1.0/DsBar_DpsiC+9.0e-1; */
/*       for (I=0;I<nSpace;I++) */
/*         { */
/*           f[k*nSpace+I]  = rho2*KW*gravity[I]; */
/*           df[k*nSpace+I] = rho2*DKW_DpsiC*gravity[I]/DsBar_DpsiC; */
/*           /\* calculate the total head the slow way for now *\/ */
/*           f[k*nSpace+I]  = 0.0; */
/*           df[k*nSpace+I] = 0.0; */
/*           phi[k] -= rho*gravity[I]*x[k*3+I]; */
/*           a[k*nSpace2+I*nSpace+I]  = rho*KW+9.0e-1; */
/*           da[k*nSpace2+I*nSpace+I] = rho*DKW_DpsiC/DsBar_DpsiC; */
/*         } */
    }
}

/* /\** Coefficients for the mass conservative Saturation equation for the fractional flow form of two-phase flow equations using Mualem-Van Genuchten. */
/*  *\/ */
/* void conservativeTwophaseSaturationMualemVanGenuchtenHomEvaluate(const int nPoints, */
/*                                                                  const int nSpace, */
/*                                                                  const double rho, */
/*                                                                  const double* gravity, */
/*                                                                  const double alpha, */
/*                                                                  const double n, */
/*                                                                  const double m, */
/*                                                                  const double thetaR, */
/*                                                                  const double thetaSR, */
/*                                                                  const double KWs, */
/*                                                                  const double viscosityRatio, */
/*                                                                  const double densityRatio, */
/*                                                                  const double* v, */
/*                                                                  double *u, */
/*                                                                  double *mass, */
/*                                                                  double *dmass, */
/*                                                                  double *f, */
/*                                                                  double *df, */
/*                                                                  double *a, */
/*                                                                  double *da) */
/* { */
/*   int k,I; */
/*   const int nSpace2=nSpace*nSpace; */
/*   register double psiC, */
/*     pcBar,pcBar_n,pcBar_nM1,pcBar_nM2, */
/*     onePlus_pcBar_n, */
/*     sBar,sqrt_sBar,DsBar_DpsiC, */
/*     thetaW,DthetaW_DpsiC, */
/*     vBar,vBar2,DvBar_DpsiC, */
/*     KW,DKW_DpsiC, */
/*     rho2=rho*rho, */
/*     thetaS=thetaR+thetaSR; */
/*   for (k=0;k<nPoints;k++) */
/*     { */
/*       sBar = fmin(fmax((u[k] - sIR)/sMir,0.0),1.0); */
/*       DsBar_Du = 1.0/sMIR; */

/*       onePlus_alphaPsiC_n = pow(sBar,1.0/-m); */
/*       alphaPsiC_n = onePlus_alphaPsiC_n - 1.0; */
/*       alphaPsiC = pow(alphaPsiC_n,1.0/n); */
      
/*       psiC = alphaPsiC/alpha; */
  
/*       alphaPsiC_nM1 = alphaPsiC_n/alphaPsiC; */
/*       sBarByOnePlus_alphaPsiC_n = sBar/onePlus_alphaPsiC_n; */
/*       DsBar_DpC = -alpha[i]*(n[i]-1.0)*alphaPsiC_nM1  */
/*         *sBarByOnePlus_alphaPsiC_n; */
      
/*       if(psiC<=0.0)  */
/*         { */
/*           DsBar_DpC = 0.0; */
/*         } */
/* } */


/* inline void VanGenuchten2p::calculateDerivatives() */
/* { */
/*   alphaPsiC_nM2 =   alphaPsiC_nM1/alphaPsiC;       */
  
/*   sBarBy_onePlus_alphaPsiC_n_2 = sBarByOnePlus_alphaPsiC_n */
/*     /onePlus_alphaPsiC_n; */
/*   DDsBar_DDpC =  alpha[i]*alpha[i]*(n[i]-1) */
/*     *((2*n[i]-1)*alphaPsiC_nM1*alphaPsiC_nM1 */
/*       *sBarBy_onePlus_alphaPsiC_n_2 */
/*       - */
/*       (n[i]-1)*alphaPsiC_nM2 */
/*       *sBarByOnePlus_alphaPsiC_n); */

/*   if (psiC <= 0.0) */
/*     { */
/*       DDsBar_DDpC = 0.0; */
/*     } */
/* } */
/*       psiC = -u[k]; */
/*       if (psiC > 0.0) */
/*         { */
/*           pcBar = alpha*psiC; */
/*           pcBar_nM2 = pow(pcBar,n-2); */
/*           pcBar_nM1 = pcBar_nM2*pcBar; */
/*           pcBar_n   = pcBar_nM1*pcBar; */
/*           onePlus_pcBar_n = 1.0 + pcBar_n; */
          
/*           sBar = pow(onePlus_pcBar_n,-m); */
/*           /\* using -mn = 1-n *\/ */
/*           DsBar_DpsiC = alpha*(1.0-n)*(sBar/onePlus_pcBar_n)*pcBar_nM1; */
          
/*           vBar = 1.0-pcBar_nM1*sBar; */
/*           vBar2 = vBar*vBar; */
/*           DvBar_DpsiC = -alpha*(n-1.0)*pcBar_nM2*sBar - pcBar_nM1*DsBar_DpsiC; */
          
/*           thetaW = thetaSR*sBar + thetaR; */
/*           DthetaW_DpsiC = thetaSR * DsBar_DpsiC;  */
          
/*           sqrt_sBar = sqrt(sBar); */
/*           krw = sqrt_sBar*vBar2; */
/*           Dkrw_DpsiC= KWs* */
/*             ((0.5/sqrt_sBar)*DsBar_DpsiC*vBar2 */
/*              + */
/*              2.0*sqrt_sBar*vBar*DvBar_DpsiC); */
/*         } */
/*       else */
/*         { */
/*           thetaW        = thetaS; */
/*           DthetaW_DpsiC = 0.0; */
/*           KW            = KWs;  */
/*           DKW_DpsiC     = 0.0; */
/*         } */
/*       mass[k] = rho*thetaW; */
/*       dmass[k] = -rho*DthetaW_DpsiC; */
/*       for (I=0;I<nSpace;I++) */
/*         { */
/*           f[k*nSpace+I]  = rho2*KW*gravity[I]; */
/*           df[k*nSpace+I] = -rho2*DKW_DpsiC*gravity[I]; */
/*           a[k*nSpace2+I*nSpace+I]  = rho*KW; */
/*           da[k*nSpace2+I*nSpace+I] = -rho*DKW_DpsiC; */
/*         } */
/*     } */
/* } */

void conservativeHeadRichardsBrooksCoreyBurdineHetEvaluate(const int nPoints,
                                                           const int nSpace,
                                                           const double rho,
                                                           const double* gravity,
                                                           const double* lambda,
                                                           const double* pd,
                                                           const double* thetaR,
                                                           const double* thetaS,
                                                           const double* KWs,
                                                           double *u,
                                                           double *mass,
                                                           double *dmass,
                                                           double *f,
                                                           double *df,
                                                           double *a,
                                                           double *da)
{
  int k,I;
  const int nSpace2=nSpace*nSpace;
  register double psiC,
    pcBar,
    sBar,DsBar_DpsiC,
    thetaW,DthetaW_DpsiC,
    KW,DKW_DpsiC,
    rho2=rho*rho,thetaSR;
  for (k=0;k<nPoints;k++)
    {
      psiC = -u[k];
      if (psiC >= pd[k])
        {
          pcBar = psiC/pd[k];
	  sBar = pow(pcBar, -lambda[k]);
	  DsBar_DpsiC = -lambda[k]*sBar/(pcBar*pd[k]);
	  KW = KWs[k]*pow(sBar, (2.0+3.0*lambda[k])/lambda[k]);
	  DKW_DpsiC = KWs[k]*((2.0+3.0*lambda[k])/lambda[k]*pow(sBar, 2.0/lambda[k]*(1.0+lambda[k]))*DsBar_DpsiC);
          thetaSR = thetaS[k]-thetaR[k];
	  thetaW = thetaSR*sBar + thetaR[k];
	  DthetaW_DpsiC = thetaSR*DsBar_DpsiC;
        }
      else
        {
          thetaW        = thetaS[k];
          DthetaW_DpsiC = 0.0;
          KW            = KWs[k]; 
          DKW_DpsiC     = 0.0;
        }
      mass[k] = rho*thetaW;
      dmass[k] = -rho*DthetaW_DpsiC;
      for (I=0;I<nSpace;I++)
        {
          f[k*nSpace+I]  = rho2*KW*gravity[I];
          df[k*nSpace+I] = -rho2*DKW_DpsiC*gravity[I];
          a[k*nSpace2+I*nSpace+I]  = rho*KW;
          da[k*nSpace2+I*nSpace+I] = -rho*DKW_DpsiC;
        }
    }
}

void conservativeHeadRichardsBrooksCoreyBurdineHomEvaluate(const int nPoints,
                                                           const int nSpace,
                                                           const double rho,
							   const double beta,
                                                           const double* gravity,
                                                           const double lambda,
                                                           const double pd,
                                                           const double thetaR,
                                                           const double thetaSR,
                                                           const double KWs,
                                                           double *u,
                                                           double *mass,
                                                           double *dmass,
                                                           double *f,
                                                           double *df,
                                                           double *a,
                                                           double *da)
{
  int k,I;
  const int nSpace2=nSpace*nSpace;
  register double psiC,
    pcBar,
    sBar,DsBar_DpsiC,
    thetaW,DthetaW_DpsiC,
    KW,DKW_DpsiC,
    rho2=rho*rho,
    thetaS=thetaR+thetaSR,
    rhom,drhom;
  
/*   /\* see what  the Lipschitz constants are like *\/ */
/*   pcBar = 1.0; */
/*   sBar = pow(pcBar, -lambda); */
/*   DsBar_DpsiC = -lambda*sBar/(pcBar*pd); */
/*   /\* DsBar_DpsiC = -lambda*sBar/psiC; *\/ */
/*   KW = KWs*pow(sBar, (2+3*lambda)/lambda); */
/*   DKW_DpsiC = KWs*((2+3*lambda)/lambda*pow(sBar, 2/lambda*(1+lambda))*DsBar_DpsiC); */
/*   thetaW = thetaSR*sBar + thetaR; */
/*   DthetaW_DpsiC = thetaSR*DsBar_DpsiC; */
/*   k=0; */
/*   mass[k] = rho*thetaW; */
/*   dmass[k] = -rho*DthetaW_DpsiC; */
/*   for (I=0;I<nSpace;I++) */
/*     { */
/*       f[k*nSpace+I]  = rho2*KW*gravity[I]; */
/*       df[k*nSpace+I] = -rho2*DKW_DpsiC*gravity[I]; */
/*       a[k*nSpace2+I*nSpace+I]  = rho*KW; */
/*       da[k*nSpace2+I*nSpace+I] = -rho*DKW_DpsiC; */
/*     } */
/*   printf("m = %12.5e \ndm=%12.5e \nf=%12.5e \ndf=%12.5e \na=%12.5e \nda=%12.5e\n",mass[k],dmass[k],f[k],df[k],a[k],da[k]); */
  for (k=0;k<nPoints;k++)
    {
      psiC = -u[k];
      if (psiC >= pd)
        {
          pcBar = psiC/pd;
	  sBar = pow(pcBar, -lambda);
	  DsBar_DpsiC = -lambda*sBar/psiC;
	  KW = KWs*pow(sBar, (2.0+3.0*lambda)/lambda);
	  DKW_DpsiC = KWs*((2.0+3.0*lambda)/lambda*pow(sBar, 2.0/lambda*(1.0+lambda))*DsBar_DpsiC);
	  thetaW = thetaSR*sBar + thetaR;
	  DthetaW_DpsiC = thetaSR*DsBar_DpsiC;
/*           /\*cek stupid version *\/ */
/*           sBar = pow(psiC/pd,-lambda); */
/*           DsBar_DpsiC = -lambda*pow(psiC/pd,-lambda-1.0)/pd; */
/*           KW = KWs*pow(sBar,(2.0+3.0*lambda)/lambda); */
/*           DKW_DpsiC = KWs*((2.0+3.0*lambda)/lambda)*pow(sBar,(2.0+3.0*lambda)/lambda - 1.0)*DsBar_DpsiC; */
/* 	  thetaW = thetaSR*sBar + thetaR; */
/* 	  DthetaW_DpsiC = thetaSR*DsBar_DpsiC; */
        }
      else
        {
          thetaW        = thetaS;
          DthetaW_DpsiC = 0.0;
          KW            = KWs; 
          DKW_DpsiC     = 0.0;
        }
      //slight compressibility
      rhom = rho*exp(beta*u[k]);
      drhom = beta*rhom;
      
      mass[k] = rhom*thetaW;
      dmass[k] = -rhom*DthetaW_DpsiC+drhom*thetaW;
      for (I=0;I<nSpace;I++)
        {
          f[k*nSpace+I]  = rho2*KW*gravity[I];
          df[k*nSpace+I] = -rho2*DKW_DpsiC*gravity[I];
          a[k*nSpace2+I*nSpace+I]  = rho*KW;
          da[k*nSpace2+I*nSpace+I] = -rho*DKW_DpsiC;
        }
    }
}

void conservativeSatRichardsBrooksCoreyBurdineHomEvaluate(const int nPoints,
							  const int nSpace,
							  const double rho,
							  const double* gravity,
							  const double lambda,
							  const double pd,
							  const double thetaR,
							  const double thetaSR,
							  const double KWs,
							  double *u,
							  double *mass,
							  double *dmass,
							  double *f,
							  double *df,
							  double *a,
							  double *da,
							  double *phi,
							  double *dphi)
{
  int k,I;
  const int nSpace2=nSpace*nSpace;
  register double psiC=0.0,
    pcBar,
    sBar,DsBar_DpsiC=0.0,
    thetaW,DthetaW_DpsiC=0.0,
    KW,DKW_DpsiC,
    rho2=rho*rho,
    thetaS=thetaR+thetaSR;
  for (k=0;k<nPoints;k++)
    {
      sBar = u[k];
      thetaW = thetaS*sBar;
      if (u[k] < 1.0)
        {
	  psiC = pow(sBar, -1/lambda)*pd;
	  /* pcBar = pow(sBar, -1/lambda); */
	  
	  /* same as before */
          pcBar = psiC/pd;
	  sBar = pow(pcBar, -lambda);
	  DsBar_DpsiC = -lambda*sBar/(pcBar*pd);
	  KW = KWs*pow(sBar, (2+3*lambda)/lambda);
	  DKW_DpsiC = KWs*((2+3*lambda)/lambda*pow(sBar, 2/lambda*(1+lambda))*DsBar_DpsiC);
	  thetaW = thetaSR*sBar + thetaR;
	  DthetaW_DpsiC = thetaSR*DsBar_DpsiC;
        }
      else
        {
          KW            = KWs; 
          DKW_DpsiC     = 0.0;
        }
      mass[k] = rho*thetaW;
      dmass[k] = -rho*DthetaW_DpsiC;
      phi[k] = -psiC;
      dphi[k] = -1.0/DsBar_DpsiC;
      for (I=0;I<nSpace;I++)
        {
          f[k*nSpace+I]  = rho2*KW*gravity[I];
          df[k*nSpace+I] = rho2*DKW_DpsiC*gravity[I]/DsBar_DpsiC;
          a[k*nSpace2+I*nSpace+I]  = rho*KW;
          da[k*nSpace2+I*nSpace+I] = rho*DKW_DpsiC/DsBar_DpsiC;
        }
    }
}


void conservativeHeadRichardsBCBfromMVGHomEvaluate(const int nPoints,
						   const int nSpace,
						   const double rho,
						   const double* gravity,
						   const double alpha,
						   const double n,
						   const double m,
						   const double thetaR,
						   const double thetaSR,
						   const double KWs,
						   double *u,
						   double *mass,
						   double *dmass,
						   double *f,
						   double *df,
						   double *a,
						   double *da)
{
  int k,I;
  const int nSpace2=nSpace*nSpace;
  register double psiC,
    pcBar,pd,lambda,
    sBar,DsBar_DpsiC,
    thetaW,DthetaW_DpsiC,
    KW,DKW_DpsiC,
    rho2=rho*rho,
    thetaS=thetaR+thetaSR;
  for (k=0;k<nPoints;k++)
    {
      psiC = -u[k];
      /* alternative representations given in Johns paper */
      pd = 1/alpha;
      lambda = n-1;
      /* printf("pd = %10.6e\t lambda = %10.6e\n", pd, lambda); */
      /* I do not know if this will work; cite Russell Johns paper */
      /* lambda = m/(1-m)*(1-pow(0.5,1/m));
	 thetaStar=0.72-0.35*exp(-pow(n,4));
	 pd = pow(thetaStar,1/lambda)/alpha*pow(pow(thetaStar,-1/m)-1, 1-m); */
      if (psiC > pd)
        {
          pcBar = psiC/pd;
	  sBar = pow(pcBar, -lambda);
	  DsBar_DpsiC = -lambda*sBar/(pcBar*pd);
	  KW = KWs*pow(sBar, (2+3*lambda)/lambda);
	  DKW_DpsiC = KWs*((2+3*lambda)/lambda*pow(sBar, 2/lambda*(1+lambda))*DsBar_DpsiC);
	  thetaW = thetaSR*sBar + thetaR;
	  DthetaW_DpsiC = thetaSR*DsBar_DpsiC;
        }
      else
        {
          thetaW        = thetaS;
          DthetaW_DpsiC = 0.0;
          KW            = KWs; 
          DKW_DpsiC     = 0.0;
        }
      mass[k] = rho*thetaW;
      dmass[k] = -rho*DthetaW_DpsiC;
      for (I=0;I<nSpace;I++)
        {
          f[k*nSpace+I]  = rho2*KW*gravity[I];
          df[k*nSpace+I] = -rho2*DKW_DpsiC*gravity[I];
          a[k*nSpace2+I*nSpace+I]  = rho*KW;
          da[k*nSpace2+I*nSpace+I] = -rho*DKW_DpsiC;
        }
    }
}

/* jcc for two phase flow in porous media, modified by cek and mwf*/
 
/* int psk_set(int pskModelFlag, */
/*             double* rwork, */
/*             double mvg_m, */
/*             double alpha, */
/*             double bc_lambda, */
/*             double bc_pd, */
/*             int (**psk_eval)(double Se, */
/*                             double *rwork, */
/*                             double *krw, */
/*                             double *dkrw, */
/*                             double *krn, */
/*                             double *dkrn, */
/*                             double *psic, */
/*                             double *dpsic)) */
/* { */
/*   if(pskModelFlag == 0) */
/*     { */
/*       *psk_eval = psk_eval_simp; */
/*     } */
/*   else if(pskModelFlag == 1) */
/*     { */
/*       *psk_eval = psk_eval_VGM; */
/*       rwork[0] = mvg_m; */
/*       rwork[1] = alpha; */
/*    } */
/*   else if(pskModelFlag == 2) */
/*     { */
/*       *psk_eval = psk_eval_VGB; */
/*       rwork[0] = mvg_m; */
/*       rwork[1] = alpha; */
/*     } */
/*   else if(pskModelFlag == 3) */
/*     { */
/*       *psk_eval = psk_eval_BCM; */
/*       rwork[0] = bc_lambda; */
/*       rwork[1] = bc_pd; */
/*     } */
/*   else if(pskModelFlag == 4) */
/*     { */
/*       *psk_eval = psk_eval_BCB; */
/*       rwork[0] = bc_lambda; */
/*       rwork[1] = bc_pd; */
/*     } */
/*   else */
/*     { */
/*       printf("Error with pskModelFlag, using simple quadratic model \n"); */
/*       *psk_eval = psk_eval_simp; */
/*       return 1; */
/*     } */
/*   return 0; */
/* } */

/* void FractionalFlowPhaseForm_saturationEvaluate( */
/*                   const int nPoints, */
/*                   const int nSpace, */
/* 		  const int nc, */
/* 		  const int pskModelFlag, */
/*                   const double Kbar, */
/* 		  const double rhon, */
/* 		  const double rhow, */
/* 		  const double *g,  */
/* 		  const double g_norm, */
/* 		  const double alpha, */
/* 		  const double bc_lambda, */
/* 		  const double bc_pd,  */
/* 		  const double mvg_n, */
/* 		  const double mvg_m, */
/* 		  const double omega,  */
/* 		  const double mun, */
/* 		  const double muw, */
/* 		  const double sw_min, */
/* 		  const double sw_max, */
/* 		  const double M, */
/* 		  const double R, */
/* 		  const double Temp, */
/* 		  const double p_o, */
/* 		  const double b,     */
/*                   double *u, */
/*                   double *m, */
/*                   double *dm, */
/* 		  double *phi, */
/* 		  double *dphi, */
/*                   double *f, */
/*                   double *df, */
/*                   double *a, */
/*                   double *da, */
/* 		  double *q_t, */
/* 		  double *psiw) */
/* { */
/*   int (*psk_eval)(double Se, */
/*                   double *rwork,  */
/*                   double *krw,  */
/*                   double *dkrw, */
/*                   double *krn, */
/*                   double *dkrn, */
/*                   double *psic, */
/*                   double *dpsic); */
/*   double rwork[2]; */
/*   psk_set(pskModelFlag,rwork,mvg_m,alpha,bc_lambda,bc_pd,&psk_eval); */
/*   int i, pi,I; */
/*   const int nSpace2=nSpace*nSpace; */
/*   double Se,dSe_dSw,krw,krn,dkrw,dkrn,psi,dpsi; */
/*   double lambdaw,dlambdaw; */
/*   double lambdan,dlambdan; */
/*   double lambdat,dlambdat; */
/* /\*   double dlambdaw_psiw,dlambdan_psiw,dlambdat_psiw; *\/ */
/*   double fw,dfw,fn,dfn; */
/* /\*   double dfw_psiw,dfn_psiw; *\/ */
/*   double RHON,dRHON,dRHON_psiw;  */
  
/*   double omega_rhow = omega*rhow;   */
 
/*   for(i=0;i<nc;i++) */
/*     { */
/*       for (pi=0;pi<nPoints;pi++) */
/*         { */
/*           effectiveSaturation(u[pi],sw_min,sw_max,&Se,&dSe_dSw); */
/*           psk_eval(Se,rwork,&krw,&dkrw,&krn,&dkrn,&psi,&dpsi); */
/*           dkrw*=dSe_dSw; */
/*           dkrn*=dSe_dSw; */
/*           dpsi*=dSe_dSw; */
/*           /\* Get the auxiliary variables for a compressable wetting phase. *\/ 	     */
/*           /\* 	    psk_auxVarCom_n_phase(krw,dkrw,krn,dkrn,psi,dpsi, *\/ */
/*           /\* 	               rhow,rhon,muw,mun,g_norm,u[pi],M,R,Temp,p_o, *\/ */
/*           /\* 		       &lambdaw,&dlambdaw,&dlambdaw_psiw,&RHON,&dRHON,&dRHON_psiw, *\/ */
/*           /\*     		       &lambdan,&dlambdan,&dlambdan_psiw,&lambdat,&dlambdat,&dlambdat_psiw, *\/ */
/*           /\*     		       &fw,&dfw,&dfw_psiw,&fn,&dfn,&dfn_psiw); *\/ */
          
/*           psk_auxVar(krw, */
/*                      dkrw, */
/*                      krn, */
/*                      dkrn, */
/*                      psi, */
/*                      dpsi, */
/*                      rhow, */
/*                      rhon, */
/*                      muw, */
/*                      mun, */
/*                      &lambdaw, */
/*                      &dlambdaw, */
/*                      &lambdan, */
/*                      &dlambdan, */
/*                      &lambdat, */
/*                      &dlambdat, */
/*                      &fw, */
/*                      &dfw, */
/*                      &fn, */
/*                      &dfn); */
/*           RHON=rhon; */
/*           dRHON_psiw=0.0; */
/*           dRHON=0.0; */
/* /\*           dlambdan_psiw=0.0; *\/ */
/* /\*           dlambdat_psiw=0.0; *\/ */
/* /\*           dfn_psiw=0.0; *\/ */
          
/*           m[pi]   = omega_rhow*(u[pi]*(sw_max-sw_min)+sw_min);  */
/*           dm[pi]  = omega_rhow*(sw_max-sw_min);  */
          
/*           phi[pi] = psi; */
/*           dphi[pi]= dpsi; */
	  
/*           for (I=0;I<nSpace;I++) */
/*             { */
/*               f[pi*nSpace+I]  = q_t[pi*nSpace+I]*fw  -  Kbar*lambdaw*fn*(b*RHON-rhow)*g[I]; */
/*               df[pi*nSpace+I] = q_t[pi*nSpace+I]*dfw */
/*                 - (Kbar*dlambdaw*fn*(b*RHON-rhow)*g[I]+ */
/*                    Kbar*lambdaw*dfn*(b*RHON-rhow)*g[I]+ */
/*                    Kbar*lambdaw*fn*(b*dRHON-rhow)*g[I]); */
              
/*               a[pi*nSpace2+I*nSpace+I]  = -Kbar*lambdaw*fn; */
/*               da[pi*nSpace2+I*nSpace+I] = -Kbar*(dlambdaw*fn + lambdaw*dfn); */
/*             } */
/*         } */
/*     } */
/* } */


/* void FractionalFlowPhaseForm_potentialEvaluate( */
/*                   const int nPoints, */
/*                   const int nSpace, */
/* 		  const int nc, */
/* 		  const int pskModelFlag, */
/*                   const double Kbar, */
/* 		  const double rhon, */
/* 		  const double rhow, */
/* 		  const double *g,  */
/* 		  const double g_norm, */
/* 		  const double alpha, */
/* 		  const double bc_lambda, */
/* 		  const double bc_pd,  */
/* 		  const double mvg_n, */
/* 		  const double mvg_m, */
/* 		  const double omega,  */
/* 		  const double mun, */
/* 		  const double muw, */
/* 		  const double sw_min, */
/* 		  const double sw_max,  */
/* 		  const double M, */
/* 		  const double R,  */
/* 		  const double Temp, */
/* 		  const double p_o, */
/* 		  const double b,    */
/*                   double *u, */
/*                   double *m, */
/*                   double *dm, */
/* 		  double *phi, */
/* 		  double *dphi, */
/*                   double *f, */
/*                   double *df, */
/*                   double *a, */
/*                   double *da, */
/* 		  double *s_w, */
/* 		  double *grad_psic) */
/* { */
/*   int (*psk_eval)(double Se, */
/*                   double *rwork,  */
/*                   double *krw,  */
/*                   double *dkrw, */
/*                   double *krn, */
/*                   double *dkrn, */
/*                   double *psic, */
/*                   double *dpsic); */
/*   double rwork[2]; */
/*   psk_set(pskModelFlag,rwork,mvg_m,alpha,bc_lambda,bc_pd,&psk_eval); */
/*   int i, pi,I; */
/*   const int nSpace2=nSpace*nSpace; */
/*   double sw;  */
/*   double dSe_dSw,Se,krw,krn,dkrw,dkrn,psi,dpsi; */
/*   double lambdaw,dlambdaw;/\* ,dlambdaw_psiw; *\/ */
/*   double lambdan,dlambdan,dlambdan_psiw; */
/*   double lambdat,dlambdat,dlambdat_psiw; */
/*   double fw,dfw,fn,dfn; */
/*   double dfn_psiw;/\* dfw_psiw *\/ */
/*   double RHON,dRHON,dRHON_psiw;  */
  
/*   for(i=0;i<nc;i++) */
/*     { */
/*       for (pi=0;pi<nPoints;pi++) */
/*         {  */
/*      if (s_w[pi] < sw_min) */
/*         { */
/*           Se = 0.0; */
/*           dSe_dSw = 0.0; */
/*         } */
/*       else if (s_w[pi] > sw_max) */
/*         { */
/*           Se = 1.0; */
/*           dSe_dSw =0.0; */
/*         } */
/*       else */
/*         { */
/*           Se = (s_w[pi] - sw_min)/(sw_max-sw_min); */
/*           dSe_dSw = 1.0/(sw_max - sw_min); */
/*         } */
/*           /\* Get the psk relation values *\/  */
/*           psk_eval(s_w[pi],rwork,&krw,&dkrw,&krn,&dkrn,&psi,&dpsi); */
/*       dkrw*=dSe_dSw; */
/*       dkrn*=dSe_dSw; */
/*       dpsi*=dSe_dSw; */
          
/*           /\* Get the auxiliary variables *\/  */
	    
/*           /\* 	    psk_auxVarCom_n_phase(krw,dkrw,krn,dkrn,psi,dpsi, *\/ */
/*           /\* 	               rhow,rhon,muw,mun,g_norm,u[pi],M,R,Temp,p_o, *\/ */
/*           /\* 		       &lambdaw,&dlambdaw,&dlambdaw_psiw,&RHON,&dRHON,&dRHON_psiw, *\/ */
/*           /\*     		       &lambdan,&dlambdan,&dlambdan_psiw,&lambdat,&dlambdat,&dlambdat_psiw, *\/ */
/*           /\*     		       &fw,&dfw,&dfw_psiw,&fn,&dfn,&dfn_psiw); *\/ */

/*           psk_auxVar(krw, */
/*                      dkrw, */
/*                      krn, */
/*                      dkrn, */
/*                      psi, */
/*                      dpsi, */
/*                      rhow, */
/*                      rhon, */
/*                      muw, */
/*                      mun, */
/*                      &lambdaw, */
/*                      &dlambdaw, */
/*                      &lambdan, */
/*                      &dlambdan, */
/*                      &lambdat, */
/*                      &dlambdat, */
/*                      &fw, */
/*                      &dfw, */
/*                      &fn, */
/*                      &dfn); */
/*           RHON=rhon; */
/*           dRHON_psiw=0.0; */
/*           dRHON=0.0; */
/*           dlambdan_psiw=0.0; */
/*           dlambdat_psiw=0.0; */
/*           dfn_psiw=0.0; */
          
/*           m[pi]   = omega*((sw*(sw_max-sw_min)+sw_min)*(rhow-RHON) + RHON); */
/*           dm[pi]  = -omega*(sw*(sw_max-sw_min)+sw_min)*dRHON_psiw + omega*dRHON_psiw; */
	  
/*           phi[pi] = u[pi]; */
/*           dphi[pi] = 1.0; */
/*           for (I=0;I<nSpace;I++) */
/*             { */
/*               f[pi*nSpace+I]  = - Kbar*lambdat*(fn*grad_psic[pi*nSpace+I]) + Kbar*lambdat*(rhow + fn*(b*RHON-rhow))*g[I]; */
/*               df[pi*nSpace+I] = ( -Kbar*grad_psic[pi*nSpace+I]*( lambdat*dfn_psiw + fn*dlambdat_psiw ) */
/*                                   +Kbar*g[I]*( dlambdat_psiw*(rhow + fn*(b*RHON-rhow) ) */
/* 				               + lambdat*( dfn_psiw*(b*RHON - rhow) */
/*                                                           + fn*(b*dRHON_psiw) ) ) ); */
              
/*               a[pi*nSpace2+I*nSpace+I]  = Kbar*lambdat; */
/*               da[pi*nSpace2+I*nSpace+I] = Kbar*dlambdat_psiw; */
/*             } */
/*         } */
/*     } */
/* } */

/* void FractionalFlowPhaseForm_saturationHetEvaluate( */
/*                   const int nPoints, */
/*                   const int nSpace, */
/* 		  const int nc, */
/* 		  const int pskModelFlag, */
/*                   const double *Kbar, */
/* 		  const double rhon, */
/* 		  const double rhow, */
/* 		  const double *g,  */
/* 		  const double *alpha, */
/* 		  const double *bc_lambda, */
/* 		  const double *bc_pd,  */
/* 		  const double *mvg_m, */
/* 		  const double *thetaS, */
/* 		  const double *thetaR,  */
/* 		  const double mun, */
/* 		  const double muw, */
/* 		  const double b,     */
/*                   double *u, */
/*                   double *m, */
/*                   double *dm, */
/* 		  double *phi, */
/* 		  double *dphi, */
/*                   double *f, */
/*                   double *df, */
/*                   double *a, */
/*                   double *da, */
/* 		  double *q_t) */
/* {		   */
/*   int (*psk_eval)(double Se, */
/*                    double *rwork,  */
/*                    double *krw,  */
/*                    double *dkrw, */
/*                    double *krn, */
/*                    double *dkrn, */
/*                    double *psic, */
/*                    double *dpsic); */
/*   double rwork[2]; */
/*   psk_set(pskModelFlag,rwork,mvg_m[0],alpha[0],bc_lambda[0],bc_pd[0],&psk_eval); */
/*   int i, pi,I; */
/*   const int nSpace2=nSpace*nSpace; */
/*   double krw,krn,dkrw,dkrn,psic,dpsic; */
/*   double lambdaw,dlambdaw,lambdan,dlambdan,lambdat,dlambdat,fw,dfw,fn,dfn;  */
  
/*   for(i=0;i<nc;i++){ */
/*     for (pi=0;pi<nPoints;pi++) */
/*       { */
/*         if((pskModelFlag==1)||(pskModelFlag==2)) */
/*           { */
/*             rwork[0] = mvg_m[pi];  */
/*             rwork[1] = alpha[pi]; */
/*           } */
/*         else if ((pskModelFlag==3)||(pskModelFlag==4)) */
/*           { */
/*             rwork[0] = bc_lambda[pi];  */
/*             rwork[1] = bc_pd[pi];		 */
/*           } */
/*         psk_eval(u[pi],rwork,&krw,&dkrw,&krn,&dkrn,&psic,&dpsic); */
        
/*         /\* Get the auxiliary variables *\/  */
/*         psk_auxVar(krw,dkrw,krn,dkrn,psic,dpsic, */
/*                    rhow,rhon,muw,mun, */
/*                    &lambdaw,&dlambdaw,&lambdan,&dlambdan,&lambdat, */
/*                    &dlambdat,&fw,&dfw,&fn,&dfn); */

/*         m[pi]   = rhow*( (thetaS[pi]-thetaR[pi])*u[pi] + thetaR[pi] );  */
/*         dm[pi]  = rhow*(thetaS[pi]-thetaR[pi]);  */
        
/*         phi[pi] = psic;  */
/*         dphi[pi]= dpsic;  */
	
/*         for (I=0;I<nSpace;I++) */
/*           { */
/*             f[pi*nSpace+I]  = (q_t[pi*nSpace+I]*fw */
/*                                - Kbar[pi]*lambdaw*fn*(b*rhon-rhow)*g[I]) ; */
/*             df[pi*nSpace+I] = (q_t[pi*nSpace+I]*dfw */
/*                                - (Kbar[pi]*g[I]*(b*rhon-rhow))*(lambdaw*dfn + fn*dlambdaw)); */
            
/*             a[pi*nSpace2+I*nSpace+I]  = -Kbar[pi]*lambdaw*fn; */
/*             da[pi*nSpace2+I*nSpace+I] = -Kbar[pi]*(dlambdaw*fn + lambdaw*dfn); */
/*           } */
/*       } */
/*   } */
/* } */

/* void FractionalFlowPhaseForm_potentialHetEvaluate( */
/*                   const int nPoints, */
/*                   const int nSpace, */
/* 		  const int nc, */
/* 		  const int pskModelFlag, */
/*                   const double *Kbar, */
/* 		  const double rhon, */
/* 		  const double rhow, */
/* 		  const double *g,  */
/* 		  const double *alpha, */
/* 		  const double *bc_lambda, */
/* 		  const double *bc_pd,  */
/* 		  const double *mvg_m, */
/* 		  const double *thetaS, */
/* 		  const double *thetaR,  */
/* 		  const double mun, */
/* 		  const double muw,  */
/* 		  const double b,    */
/*                   double *u, */
/*                   double *m, */
/*                   double *dm, */
/* 		  double *phi, */
/* 		  double *dphi, */
/*                   double *f, */
/*                   double *df, */
/*                   double *a, */
/*                   double *da, */
/* 		  double *s_w, */
/* 		  double *grad_psic) */
/* { */
/*   int (*psk_eval)(double Se, */
/*                    double *rwork,  */
/*                    double *krw,  */
/*                    double *dkrw, */
/*                    double *krn, */
/*                    double *dkrn, */
/*                    double *psic, */
/*                    double *dpsic); */
/*   double rwork[2]; */
/*   psk_set(pskModelFlag,rwork,mvg_m[0],alpha[0],bc_lambda[0],bc_pd[0],&psk_eval); */
/*   int i, pi,I; */
/*   const int nSpace2=nSpace*nSpace; */
/*   double krw,krn,dkrw,dkrn,psi,dpsi; */
/*   double lambdaw,dlambdaw,lambdan,dlambdan,lambdat,dlambdat,fw,dfw,fn,dfn;  */
  
/*   for(i=0;i<nc;i++) */
/*     { */
/*       for (pi=0;pi<nPoints;pi++) */
/*         { */
/*           if((pskModelFlag==1)||(pskModelFlag==2)) */
/*             { */
/*               rwork[0] = mvg_m[pi];  */
/*               rwork[1] = alpha[pi]; */
/*             } */
/*           else if ((pskModelFlag==3)||(pskModelFlag==4)) */
/*             { */
/*               rwork[0] = bc_lambda[pi];  */
/*               rwork[1] = bc_pd[pi]; */
/*             }             */
/*           /\* Get the psk relation values *\/  */
/*           psk_eval(s_w[pi],rwork,&krw,&dkrw,&krn,&dkrn,&psi,&dpsi); */

/*           /\* Get the auxiliary variables *\/  */
/*           psk_auxVar(krw,dkrw,krn,dkrn,psi,dpsi, */
/*                      rhow,rhon,muw,mun, */
/*                      &lambdaw,&dlambdaw,&lambdan,&dlambdan,&lambdat, */
/*                      &dlambdat,&fw,&dfw,&fn,&dfn);  */
          
/*           m[pi]   = ((thetaS[pi]-thetaR[pi])*s_w[pi] + thetaR[pi])*(rhow-rhon) + thetaS[pi]*rhon;  */
/*           dm[pi]  = 0.0;  */
          
/*           phi[pi] = u[pi];  */
/*           dphi[pi]= 1.0;  */
	  
/*           for (I=0;I<nSpace;I++) */
/*             { */
/*               f[pi*nSpace+I]  = -Kbar[pi]*lambdat*( fn*grad_psic[pi*nSpace+I] - (rhow + fn*(b*rhon-rhow))*g[I] ); */
/*               df[pi*nSpace+I] = 0.0; */
              
/*               a[pi*nSpace2+I*nSpace+I]  = Kbar[pi]*lambdat; */
/*               da[pi*nSpace2+I*nSpace+I] = 0.0; */
/*             } */
/*         } */
/*     } */
/* } */

/* /\* end jcc additions for two phase flow *\/  */

/* void TwophaseDarcyFC_Evaluate(const int nPoints, */
/*                               const int nSpace, */
/*                               const int pskModelFlag, */
/*                               const double Kbar, */
/*                               const double rhon, */
/*                               const double rhow, */
/*                               const double *g, */
/*                               const double *x, */
/*                               const double alpha, */
/*                               const double bc_lambda, */
/*                               const double bc_pd, */
/*                               const double mvg_n, */
/*                               const double mvg_m, */
/*                               const double omega, */
/*                               const double omega_r, */
/*                               const double mun, */
/*                               const double muw, */
/*                               const double b, */
/*                               double *sw, */
/*                               double *psiw, */
/*                               double *mw, */
/*                               double *dmw, */
/*                               double *mn, */
/*                               double *dmn, */
/*                               double *phi_psiw, */
/*                               double *dphi_psiw_dpsiw, */
/*                               double *phi_psin, */
/*                               double *dphi_psin_dpsiw, */
/*                               double *dphi_psin_dsw, */
/*                               double *fw, */
/*                               double *dfw, */
/*                               double *fn, */
/*                               double *dfn, */
/*                               double *aw, */
/*                               double *daw, */
/*                               double *an, */
/*                               double *dan) */
/* { */
/*   int (*psk_eval)(double Sw, */
/*                    double *rwork, */
/*                    double *krw, */
/*                    double *dkrw, */
/*                    double *krn, */
/*                    double *dkrn, */
/*                    double *psic, */
/*                    double *dpsic); */
/*   double rwork[4]; */
/*   psk_set(pskModelFlag,rwork,mvg_m,alpha,bc_lambda,bc_pd,&psk_eval); */
/*   int pi,I; */
/*   const int nSpace2=nSpace*nSpace; */
/*   double se; */
/*   double krw,krn,dkrw,dkrn,psic,dpsic,KN,DKN,KW,DKW; */
/*   double omega_rhow = omega*rhow,omega_rhon = omega*rhon; */
/*   const double omega_sr_inv = 1.0/(omega - omega_r); */
/*   for (pi=0;pi<nPoints;pi++) */
/*     { */
/*       /\*effective saturation*\/ */
/*       se = (omega*sw[pi] - omega_r)*omega_sr_inv; */
      
/*       /\* Get the psk relation values *\/ */
/*       psk_eval(se,rwork,&krw,&dkrw,&krn,&dkrn,&psic,&dpsic); */
      
/*       /\* phase mass terms *\/ */
/*       mw[pi]   = omega_rhow*sw[pi]; */
/*       dmw[pi]  = omega_rhow; */
      
/*       mn[pi]   = omega_rhon*(1.0-sw[pi]); */
/*       dmn[pi]  =-omega_rhon; */
      
/*       /\* potentials *\/ */
/*       phi_psiw[pi] = psiw[pi]; */
/*       dphi_psiw_dpsiw[pi]= 1.0; */
      
/*       phi_psin[pi] = psiw[pi] + psic; */
/*       dphi_psin_dpsiw[pi]= 1.0; */
/*       dphi_psin_dsw[pi]= dpsic; */
      
/*       KW  = rhow*Kbar*krw/muw; */
/*       DKW = rhow*Kbar*dkrw/muw; */
      
/*       KN  = rhon*Kbar*krn/mun; */
/*       DKN = rhon*Kbar*dkrn/mun; */
      
/*       for (I=0;I<nSpace;I++) */
/*         { */
/*           phi_psiw[pi] -= rhow*g[I]*x[pi*3+I]; */
/*           phi_psin[pi] -= b*rhon*g[I]*x[pi*3+I]; */
          
/*           aw[pi*nSpace2+I*nSpace+I]  = KW; */
/*           daw[pi*nSpace2+I*nSpace+I] = DKW; */
          
/*           an[pi*nSpace2+I*nSpace+I]  = KN; */
/*           dan[pi*nSpace2+I*nSpace+I] = DKN; */
/*         } */
/*     } */
/* } */

/* void TwophaseFFDarcyFC_Evaluate(const int nPoints, */
/*                                 const int nSpace, */
/*                                 const int pskModelFlag, */
/*                                 const double Kbar, */
/*                                 const double rhon, */
/*                                 const double rhow, */
/*                                 const double *g,  */
/*                                 const double *x,  */
/*                                 const double alpha, */
/*                                 const double bc_lambda, */
/*                                 const double bc_pd,  */
/*                                 const double mvg_n, */
/*                                 const double mvg_m, */
/*                                 const double omega,  */
/* 				const double omega_r, */
/*                                 const double mun, */
/*                                 const double muw, */
/*                                 const double b,     */
/*                                 double *sw, */
/*                                 double *psiw, */
/*                                 double *mw, */
/*                                 double *dmw_dsw, */
/*                                 double *mm, */
/*                                 double *dmm_dsw, */
/*                                 double *phi_psic, */
/*                                 double *dphi_psic_dsw, */
/*                                 double *phi_psiw, */
/*                                 double *dphi_psiw_dpsiw, */
/*                                 double *fm, */
/*                                 double *dfm_dsw, */
/*                                 double *aw_psiw, */
/*                                 double *daw_psiw_dsw, */
/*                                 double *am_psiw, */
/*                                 double *dam_psiw_dsw, */
/*                                 double *am_psic, */
/*                                 double *dam_psic_dsw) */
/* {		   */
/*   int (*psk_eval)(double Se, */
/*                    double *rwork,  */
/*                    double *krw,  */
/*                    double *dkrw, */
/*                    double *krn, */
/*                    double *dkrn, */
/*                    double *psic, */
/*                    double *dpsic); */
/*   double rwork[2]; */
/*   psk_set(pskModelFlag,rwork,mvg_m,alpha,bc_lambda,bc_pd,&psk_eval); */
/*   int pi,I; */
/*   const int nSpace2=nSpace*nSpace; */
/*   double se,sw_max=1.0,sw_min=0.0;  */
/*   double krw,krn,dkrw,dkrn,psic,dpsic,KW,DKW; */
/*   double lambdaw,dlambdaw,lambdan,dlambdan,lambdat,dlambdat,fw,dfw,fn,dfn;  */
/*   double omega_rhow = omega*rhow;   */
/*   /\*     double max_krw=0.0,max_krn=0.0; *\/ */
/*   double RHON; */
/*   const double omega_sr_inv = 1.0/(omega - omega_r); */
  
/*   for (pi=0;pi<nPoints;pi++) */
/*     { */
/*       se     = (omega*sw[pi] - omega_r)*omega_sr_inv; */

/*       /\* Get the psk relation values *\/  */
/*       psk_eval(se,rwork,&krw,&dkrw,&krn,&dkrn,&psic,&dpsic); */
      
/*       /\* Get the auxiliary variables *\/  */
/*       psk_auxVar(krw,dkrw,krn,dkrn,psic,dpsic, */
/*                  rhow,rhon,muw,mun, */
/*                  &lambdaw,&dlambdaw,&lambdan,&dlambdan,&lambdat, */
/*                  &dlambdat,&fw,&dfw,&fn,&dfn);  */
      
/*       /\* w-phase mass term *\/ */
/*       mw[pi]   = omega_rhow*(sw[pi]*(sw_max-sw_min)+sw_min); */
/*       dmw_dsw[pi]  = omega_rhow*(sw_max-sw_min); */
      
/*       /\* mixture mass term *\/ */
/*       mm[pi]   = omega*((sw[pi]*(sw_max-sw_min)+sw_min)*(rhow-RHON) + RHON); */
/*       dmm_dsw[pi]  = omega*(((sw_max-sw_min)+sw_min)*(rhow-RHON)); */
      
/*       /\* capillary potential*\/ */
/*       phi_psic[pi] = psic; */
/*       dphi_psic_dsw[pi]= dpsic; */
      
/*       /\* w-phase potential *\/ */
/*       phi_psiw[pi] = psiw[pi]; */
/*       dphi_psiw_dpsiw[pi]= 1.0; */
      
/*       KW  = rhow*Kbar*krw/muw; */
/*       DKW = rhow*Kbar*dkrw/muw; */
/*       for (I=0;I<nSpace;I++) */
/*         { */
/*           /\* w phase *\/ */
/*           phi_psiw[pi] -= rhow*g[I]*x[pi*3+I]; */
          
/*           aw_psiw[pi*nSpace2+I*nSpace+I]  = KW; */
/*           daw_psiw_dsw[pi*nSpace2+I*nSpace+I] = DKW; */
          
/*           /\* mixture *\/ */
/*           fm[pi*nSpace+I]  = Kbar*lambdat*fn*(b*rhon-rhow)*g[I]; */
/*           dfm_dsw[pi*nSpace+I] = Kbar*(dlambdat*fn + lambdat*dfn)*(b*rhon-rhow)*g[I]; */
          
/*           am_psiw[pi*nSpace2+I*nSpace+I]  = Kbar*lambdat; */
/*           dam_psiw_dsw[pi*nSpace2+I*nSpace+I] = Kbar*dlambdat; */
          
/*           am_psic[pi*nSpace2+I*nSpace+I]  = Kbar*lambdat*fn; */
/*           dam_psic_dsw[pi*nSpace2+I*nSpace+I] = Kbar*(dlambdat*fn+lambdat*dfn); */
/*         } */
/*     } */
/* } */

/* void TwophaseDarcyFCHet_Evaluate(const int nPoints, */
/*                               const int nSpace, */
/*                               const int pskModelFlag, */
/*                               const double *Kbar, */
/*                               const double rhon, */
/*                               const double rhow, */
/*                               const double *g,  */
/*                               const double *x,  */
/*                               const double *alpha, */
/*                               const double *bc_lambda, */
/*                               const double *bc_pd,  */
/*                               const double *mvg_m, */
/*                               const double *omega,  */
/*                               const double *omega_r,  */
/*                               const double mun, */
/*                               const double muw, */
/*                               const double b,     */
/*                               double *sw, */
/*                               double *psiw, */
/*                               double *mw, */
/*                               double *dmw, */
/*                               double *mn, */
/*                               double *dmn, */
/*                               double *phi_psiw, */
/*                               double *dphi_psiw_dpsiw, */
/*                               double *phi_psin, */
/*                               double *dphi_psin_dpsiw, */
/*                               double *dphi_psin_dsw, */
/*                               double *fw, */
/*                               double *dfw, */
/*                               double *fn, */
/*                               double *dfn, */
/*                               double *aw, */
/*                               double *daw, */
/*                               double *an, */
/*                               double *dan) */
/* {		   */
/*   int (*psk_eval)(double Se, */
/*                    double *rwork,  */
/*                    double *krw,  */
/*                    double *dkrw, */
/*                    double *krn, */
/*                    double *dkrn, */
/*                    double *psic, */
/*                    double *dpsic); */
/*   double rwork[2]; */
/*   psk_set(pskModelFlag,rwork,mvg_m[0],alpha[0],bc_lambda[0],bc_pd[0],&psk_eval); */
    
/*   int pi,I; */
/*   const int nSpace2=nSpace*nSpace; */
/*   double se;  */
/*   double krw,krn,dkrw,dkrn,psic,dpsic,KN,DKN,KW,DKW; */
  
/*   for (pi=0;pi<nPoints;pi++) */
/*     {     */
/*       if((pskModelFlag==1)||(pskModelFlag==2)) */
/*         { */
/*           rwork[0] = mvg_m[pi];  */
/*           rwork[1] = alpha[pi];	 */
/* 	} */
/*       else if ((pskModelFlag==3)||(pskModelFlag==4)) */
/*         { */
/*           rwork[0] = bc_lambda[pi];  */
/*           rwork[1] = bc_pd[pi];		 */
/* 	} */
/*       /\*effective saturation*\/ */
/*       se     = (omega[pi]*sw[pi] - omega_r[pi])/(omega[pi]-omega_r[pi]); */
      
/*       /\* Get the psk relation values *\/  */
/*       psk_eval(se,rwork,&krw,&dkrw,&krn,&dkrn,&psic,&dpsic); */
      
/*       /\* phase mass terms *\/ */
/*       mw[pi]   = omega[pi]*rhow*sw[pi]; */
/*       dmw[pi]  = omega[pi]*rhow;  */
      
/*       mn[pi]   = omega[pi]*rhon*(1.0-sw[pi]); */
/*       dmn[pi]  = -omega[pi]*rhon;  */
      
/*       /\* potentials *\/ */
/*       phi_psiw[pi] = psiw[pi]; */
/*       dphi_psiw_dpsiw[pi]= 1.0; */
      
/*       phi_psin[pi] = psiw[pi] + psic; */
/*       dphi_psin_dpsiw[pi]= 1.0; */
/*       dphi_psin_dsw[pi]= dpsic; */
      
/*       KW  = rhow*Kbar[pi]*krw/muw; */
/*       DKW = rhow*Kbar[pi]*dkrw/muw; */

/*       KN  = rhon*Kbar[pi]*krn/mun; */
/*       DKN = rhon*Kbar[pi]*dkrn/mun; */

/*       for (I=0;I<nSpace;I++) */
/*         { */
/*           phi_psiw[pi] -= rhow*g[I]*x[pi*3+I]; */
/*           phi_psin[pi] -= b*rhon*g[I]*x[pi*3+I]; */
          
/*           aw[pi*nSpace2+I*nSpace+I]  = KW; */
/*           daw[pi*nSpace2+I*nSpace+I] = DKW; */
          
/*           an[pi*nSpace2+I*nSpace+I]  = KN; */
/*           dan[pi*nSpace2+I*nSpace+I] = DKN; */
/*         } */
/*     } */
/* } */

/* void TwophaseDarcyFCHet_EvaluateV2(const int nSimplex, */
/* 				   const int nPointsPerSimplex, */
/* 				   const int nSpace, */
/* 				   const int nTypes, */
/* 				   const int pskModelFlag, */
/* 				   const int* materialTypes, */
/* 				   const double *Kbar, */
/* 				   const double rhon, */
/* 				   const double rhow, */
/* 				   const double b, */
/* 				   const double *g, */
/* 				   const double *x,  */
/* 				   const double *mvg_alpha, */
/* 				   const double *mvg_n, */
/* 				   const double *mvg_m, */
/* 				   const double *bc_pd,  */
/* 				   const double *bc_lambda, */
/* 				   const double *thetaS, */
/* 				   const double *thetaR, */
/* 				   const double mun, */
/* 				   const double muw, */
/* 				   double *sw, */
/* 				   double *psiw, */
/* 				   double *mw, */
/* 				   double *dmw, */
/* 				   double *mn, */
/* 				   double *dmn, */
/* 				   double *phi_psiw, */
/* 				   double *dphi_psiw_dpsiw, */
/* 				   double *phi_psin, */
/* 				   double *dphi_psin_dpsiw, */
/* 				   double *dphi_psin_dsw, */
/* 				   double *fw, */
/* 				   double *dfw, */
/* 				   double *fn, */
/* 				   double *dfn, */
/* 				   double *aw, */
/* 				   double *daw, */
/* 				   double *an, */
/* 				   double *dan) */
/* {		   */
/*   int (*psk_eval)(double Se, */
/*                    double *rwork,  */
/*                    double *krw,  */
/*                    double *dkrw, */
/*                    double *krn, */
/*                    double *dkrn, */
/*                    double *psic, */
/*                    double *dpsic); */
/*   double rwork[2]; */
/*   psk_set(pskModelFlag,rwork,mvg_m[0],mvg_alpha[0],bc_lambda[0],bc_pd[0],&psk_eval); */
  
/*   int i,j,k,I,matID; */
/*   const int nSpace2=nSpace*nSpace; */
/*   double se,thw,thn,ths,thr;  */
/*   double krw,krn,dkrw,dkrn,psic,dpsic,KN,DKN,KW,DKW; */
  
/*   for (i=0; i < nSimplex; i++) */
/*     { */
/*       matID=materialTypes[i]; */
/*       if (matID < 0) */
/*         matID = 0; */
/*       if (matID > nTypes-1) */
/*         matID = nTypes-1; */
/*       for (j=0; j < nPointsPerSimplex; j++) */
/*         { */
/*           k = i*nPointsPerSimplex + j; */
/*           if(pskModelFlag==1 || pskModelFlag==2) */
/*             { */
/*               rwork[0] = mvg_m[matID];  */
/*               rwork[1] = mvg_alpha[matID];	 */
/*             } */
/*           else if (pskModelFlag==3 ||pskModelFlag==4) */
/*             { */
/*               rwork[0] = bc_lambda[matID];  */
/*               rwork[1] = bc_pd[matID];		 */
/*             } */
/*           /\*volume fractions*\/ */
/*           ths = thetaS[matID]; thr = thetaR[matID]; */
/*           thw = ths*sw[k]; /\*wetting*\/ */
/*           thn = ths*(1.0-sw[k]);/\*non wetting*\/ */
          
/*           /\*effective saturation*\/ */
/*           se  = (thw-thr)/(ths-thr); */
          
/*           /\* Get the psk relation values *\/  */
/*           psk_eval(se,rwork,&krw,&dkrw,&krn,&dkrn,&psic,&dpsic); */
/* /\*           /\\* if using BC have to enforce displacement pressure here?*\\/ *\/ */
/* /\*           if((pskModelFlag == 3 || pskModelFlag == 4) && *\/ */
/* /\*              se >= 1.0-pd_eps) *\/ */
/* /\*             { *\/ */
/* /\*               psic = 0.0; dpsic = 0.0; krn = 0.0; dkrn = 0.0; krw = 1.0; *\/ */
/* /\*             } *\/ */

/*           /\* phase mass terms *\/ */
/*           mw[k]   = thw*rhow; */
/*           dmw[k]  = ths*rhow;  */
	  
/*           mn[k]   = thn*rhon; */
/*           dmn[k]  =-ths*rhon;  */
	  
/*           /\* potentials *\/ */
/*           phi_psiw[k] = psiw[k]; */
/*           dphi_psiw_dpsiw[k]= 1.0; */
          
/*           phi_psin[k] = psiw[k] + psic; */
/*           dphi_psin_dpsiw[k]= 1.0; */
/*           dphi_psin_dsw[k]= dpsic; */
          
/*           /\*conductivities*\/ */
/*           KW  = rhow*Kbar[matID]*krw/muw; /\*mu's are normalized by muw*\/ */
/*           DKW = rhow*Kbar[matID]*dkrw/muw; */

/*           KN  = rhon*Kbar[matID]*krn/mun; */
/*           DKN = rhon*Kbar[matID]*dkrn/mun; */
          
/*           for (I=0;I<nSpace;I++) */
/*             { */
/*               phi_psiw[k] -= rhow*g[I]*x[k*3+I]; */
/*               phi_psin[k] -= rhon*g[I]*x[k*3+I]; */
              
/*               aw[k*nSpace2+I*nSpace+I]  = KW;/\*have rho's in them*\/ */
/*               daw[k*nSpace2+I*nSpace+I] = DKW; */
              
/*               an[k*nSpace2+I*nSpace+I]  = KN; */
/*               dan[k*nSpace2+I*nSpace+I] = DKN; */
/*             }/\*I*\/ */
/*         }/\*j*\/ */
/*     }/\*i*\/ */
/* } */

/* void TwophaseFFDarcyFCHet_EvaluateV2(const int nSimplex, */
/* 				     const int nPointsPerSimplex, */
/* 				     const int nSpace, */
/* 				     const int nTypes, */
/* 				     const int pskModelFlag, */
/* 				     const int* materialTypes, */
/* 				     const double *Kbar, */
/* 				     const double rhon, */
/* 				     const double rhow, */
/* 				     const double b, */
/* 				     const double *g, */
/* 				     const double *x,  */
/* 				     const double *mvg_alpha, */
/* 				     const double *mvg_n, */
/* 				     const double *mvg_m, */
/* 				     const double *bc_pd,  */
/* 				     const double *bc_lambda, */
/* 				     const double *thetaS, */
/* 				     const double *thetaR, */
/* 				     const double mun, */
/* 				     const double muw, */
/* 				     double *sw, */
/* 				     double *psiw, */
/* 				     double *mw, */
/* 				     double *dmw_dsw, */
/* 				     double *mm, */
/* 				     double *dmm_dsw, */
/* 				     double *phi_psic, */
/* 				     double *dphi_psic_dsw, */
/* 				     double *phi_psiw, */
/* 				     double *dphi_psiw_dpsiw, */
/* 				     double *fm, */
/* 				     double *dfm_dsw, */
/* 				     double *fw, */
/* 				     double *dfw_dsw, */
/* 				     double *aw_psiw, */
/* 				     double *daw_psiw_dsw, */
/* 				     double *am_psiw, */
/* 				     double *dam_psiw_dsw, */
/* 				     double *am_psic, */
/* 				     double *dam_psic_dsw) */
/* {		   */
/*   int (*psk_eval)(double Se, */
/*                    double *rwork,  */
/*                    double *krw,  */
/*                    double *dkrw, */
/*                    double *krn, */
/*                    double *dkrn, */
/*                    double *psic, */
/*                    double *dpsic); */
/*   double rwork[2]; */
/*   psk_set(pskModelFlag,rwork,mvg_m[0],mvg_alpha[0],bc_lambda[0],bc_pd[0],&psk_eval); */
  
/*   int i,j,k,I,matID; */
/*   const int nSpace2=nSpace*nSpace; */
/*   const double pd_eps = 9.0e-6; */
/*   double se,thw,thn,ths,thr;  */
/*   double krw,krn,dkrw,dkrn,psic,dpsic,KN,DKN,KW,DKW; */
  
/*   for (i=0; i < nSimplex; i++) */
/*     { */
/*       matID=materialTypes[i]; */
/*       if (matID < 0) */
/*         matID = 0; */
/*       if (matID > nTypes-1) */
/*         matID = nTypes-1; */
/*       for (j=0; j < nPointsPerSimplex; j++) */
/*         { */
/*           k = i*nPointsPerSimplex + j; */
/*           if(pskModelFlag==1 || pskModelFlag==2) */
/*             { */
/*               rwork[0] = mvg_m[matID];  */
/*               rwork[1] = mvg_alpha[matID];	 */
/*             } */
/*           else if (pskModelFlag==3 ||pskModelFlag==4) */
/*             { */
/*               rwork[0] = bc_lambda[matID];  */
/*               rwork[1] = bc_pd[matID];		 */
/*             } */
/*           /\*volume fractions*\/ */
/*           ths = thetaS[matID]; thr = thetaR[matID]; */
/*           thw = ths*sw[k]; /\*wetting*\/ */
/*           thn = ths*(1.0-sw[k]);/\*non wetting*\/ */
/*           /\*effective saturation*\/ */
/*           se  = (thw-thr)/(ths-thr); */
          
/*           /\* Get the psk relation values *\/  */
/*           psk_eval(se,rwork,&krw,&dkrw,&krn,&dkrn,&psic,&dpsic); */
/*           /\* if using BC have to enforce displacement pressure here?*\/ */
/*           if((pskModelFlag == 3 || pskModelFlag == 4) && */
/*              se >= 1.0-pd_eps) */
/*             { */
/*               psic = 0.0; dpsic = 0.0; krn = 0.0; dkrn = 0.0; krw = 1.0; */
/*             } */
          
/*           /\* w-phase mass term *\/ */
/*           mw[k]     = thw*rhow; */
/*           dmw_dsw[k]= ths*rhow; */
          
/*           /\* mixture mass term *\/ */
/*           mm[k]     = thw*rhow + thn*rhon; */
/*           dmm_dsw[k]= ths*(rhow - rhon); */
          
/*           /\* capillary potential*\/ */
/*           phi_psic[k] = psic; */
/*           dphi_psic_dsw[k]= dpsic; */
          
/*           /\* w-phase potential *\/ */
/*           phi_psiw[k] = psiw[k]; */
/*           dphi_psiw_dpsiw[k]= 1.0; */
          
/*           KW  = rhow*Kbar[matID]*krw/muw; */
/*           DKW = rhow*Kbar[matID]*dkrw/muw; */

/*           KN  = rhon*Kbar[matID]*krn/mun; */
/*           DKN = rhon*Kbar[matID]*dkrn/mun; */
/*           for (I=0;I<nSpace;I++) */
/*             { */
/*               /\* w phase *\/ */
/*               /\*don't include gravity in potential?*\/ */
/*               /\*phi_psiw[k] -= rhow*g[I]*x[pi*3+I]*\/; */
/*               /\* w [hase *\/ */
/*               fw[k*nSpace+I]     = rhow*KW*g[I]; */
/*               dfw_dsw[k*nSpace+I]= rhow*DKW*g[I]; */
              
/*               aw_psiw[k*nSpace2+I*nSpace+I]  = KW; */
/*               daw_psiw_dsw[k*nSpace2+I*nSpace+I] = DKW; */
              
/*               /\* mixture *\/ */
/*               fm[k*nSpace+I]     = rhow*KW*g[I] + rhon*KN*b*g[I]; */
/*               dfm_dsw[k*nSpace+I]= rhow*DKW*g[I]+ rhon*DKN*b*g[I]; */
              
/*               am_psiw[k*nSpace2+I*nSpace+I]  = KW + KN; */
/*               dam_psiw_dsw[k*nSpace2+I*nSpace+I] = DKW + DKN; */
              
/*               am_psic[k*nSpace2+I*nSpace+I]  = KN; */
/*               dam_psic_dsw[k*nSpace2+I*nSpace+I] = DKN; */
/*             }/\*I*\/ */
/*         }/\*j*\/ */
/*     }/\*i*\/ */
/* } */

/* void TwophaseFFDarcyFCHet_Evaluate(const int nPoints, */
/*                                    const int nSpace, */
/*                                    const int pskModelFlag, */
/*                                    const double *Kbar, */
/*                                    const double rhon, */
/*                                    const double rhow, */
/*                                    const double *g,  */
/*                                    const double *x,  */
/*                                    const double *alpha, */
/*                                    const double *bc_lambda, */
/*                                    const double *bc_pd,  */
/*                                    const double *mvg_m, */
/*                                    const double *omega,  */
/*                                    const double *omega_r,  */
/*                                    const double mun, */
/*                                    const double muw, */
/*                                    const double b,     */
/*                                    double *sw, */
/*                                    double *psiw, */
/*                                    double *mw, */
/*                                    double *dmw_dsw, */
/*                                    double *mm, */
/*                                    double *dmm_dsw, */
/*                                    double *phi_psic, */
/*                                    double *dphi_psic_dsw, */
/*                                    double *phi_psiw, */
/*                                    double *dphi_psiw_dpsiw, */
/*                                    double *fm, */
/*                                    double *dfm_dsw, */
/*                                    double *aw_psiw, */
/*                                    double *daw_psiw_dsw, */
/*                                    double *am_psiw, */
/*                                    double *dam_psiw_dsw, */
/*                                    double *am_psic, */
/*                                    double *dam_psic_dsw) */
/* {		   */
/*   int (*psk_eval)(double Se, */
/*                    double *rwork,  */
/*                    double *krw,  */
/*                    double *dkrw, */
/*                    double *krn, */
/*                    double *dkrn, */
/*                    double *psic, */
/*                    double *dpsic); */
/*   double rwork[2]; */
/*   psk_set(pskModelFlag,rwork,mvg_m[0],alpha[0],bc_lambda[0],bc_pd[0],&psk_eval); */
/*   int pi,I; */
/*   const int nSpace2=nSpace*nSpace; */
/*   double se,seeval;  */
/*   double krw,krn,dkrw,dkrn,psic,dpsic,KW,DKW; */
/*   double lambdaw,dlambdaw,lambdan,dlambdan,lambdat,dlambdat,fw,dfw,fn,dfn;  */
  
/*   for (pi=0;pi<nPoints;pi++){ */
    
/*     if((pskModelFlag==1)||(pskModelFlag==2)) */
/*       { */
/*         rwork[0] = mvg_m[pi];  */
/*         rwork[1] = alpha[pi];	 */
/*       } */
/*     else if ((pskModelFlag==3)||(pskModelFlag==4)) */
/*       { */
/*         rwork[0] = bc_lambda[pi];  */
/*         rwork[1] = bc_pd[pi];		 */
/*       } */
/*     /\*effective saturation*\/ */
/*     se     = (omega[pi]*sw[pi] - omega_r[pi])/(omega[pi]-omega_r[pi]); */
    
/*     /\* Get the psk relation values *\/  */
/*     psk_eval(seeval,rwork,&krw,&dkrw,&krn,&dkrn,&psic,&dpsic); */
    
/*     /\* Get the auxiliary variables *\/  */
/*     psk_auxVar(krw,dkrw,krn,dkrn,psic,dpsic, */
/*                rhow,rhon,muw,mun, */
/*                &lambdaw,&dlambdaw,&lambdan,&dlambdan,&lambdat, */
/*                &dlambdat,&fw,&dfw,&fn,&dfn);  */
    
/*     /\* w-phase mass term *\/ */
/*     mw[pi]   = omega[pi]*rhow*sw[pi]; */
/*     dmw_dsw[pi]  = omega[pi]*rhow;  */
    
/*     /\* mixture mass term *\/ */
/*     mm[pi]   = omega[pi]*(sw[pi]*(rhow-rhon) + rhon);  */
/*     dmm_dsw[pi]  = omega[pi]*(rhow-rhon);  */
    
/*     /\* capillary potential*\/ */
/*     phi_psic[pi] = psic; */
/*     dphi_psic_dsw[pi]= dpsic; */
    
/*     /\* w-phase potential *\/ */
/*     phi_psiw[pi] = psiw[pi]; */
/*     dphi_psiw_dpsiw[pi]= 1.0; */
    
/*     KW  = rhow*Kbar[pi]*krw/muw; */
/*     DKW = rhow*Kbar[pi]*dkrw/muw; */

/*     for (I=0;I<nSpace;I++) */
/*       { */
/*         /\* w phase *\/ */
/*         phi_psiw[pi] -= rhow*g[I]*x[pi*3+I]; */
        
/*         aw_psiw[pi*nSpace2+I*nSpace+I]  = KW; */
/*         daw_psiw_dsw[pi*nSpace2+I*nSpace+I] = DKW; */
        
/*         /\* mixture *\/ */
/*         fm[pi*nSpace+I]  = Kbar[pi]*lambdat*fn*(b*rhon-rhow)*g[I]; */
/*         dfm_dsw[pi*nSpace+I] = Kbar[pi]*(dlambdat*fn + lambdat*dfn)*(b*rhon-rhow)*g[I]; */
        
/*         am_psiw[pi*nSpace2+I*nSpace+I]  = Kbar[pi]*lambdat; */
/*         dam_psiw_dsw[pi*nSpace2+I*nSpace+I] = Kbar[pi]*dlambdat; */
        
/*         am_psic[pi*nSpace2+I*nSpace+I]  = Kbar[pi]*lambdat*fn; */
/*         dam_psic_dsw[pi*nSpace2+I*nSpace+I] = Kbar[pi]*(dlambdat*fn+lambdat*dfn); */
/*       } */
/*   } */
/* } */

/* end two-phase flow in porous media coefficients */

void LinearElasticity_1D_Evaluate(const int nPoints,
                                  const double E,
                                  const double nu,
                                  const double *g,
                                  const double *u,
                                  double *uu_diff_ten,
                                  double *u_force)
{
  int k;
  for (k=0;k<nPoints;k++)
    {
      /* \sigma^x */
      
      /* a^{xx} */

      uu_diff_ten[k*1+0] = (E/(1.0+nu))*(1.0+nu/(1.0-2.0*nu));
    }
}

void LinearElasticity_2D_Evaluate(const int nPoints,
                                  const double E,
                                  const double nu,
                                  const double *g,
                                  const double *u,
                                  const double *v,
                                  double *uu_diff_ten,double *uv_diff_ten,
                                  double *vu_diff_ten,double *vv_diff_ten,
                                  double *u_force,
                                  double *v_force)
{
  int k;
  for (k=0;k<nPoints;k++)
    {
      /* \sigma^x */
      
      /* a^{xx} */

      uu_diff_ten[k*4+0] = (E/(1.0+nu))*(1.0+nu/(1.0-2.0*nu));
      uu_diff_ten[k*4+3] = (E/(1.0+nu))*0.5;
      
      /* a^{xy} */

      uv_diff_ten[k*4+1] = (E/(1.0+nu))*(nu/(1.0-2.0*nu));
      uv_diff_ten[k*4+2] = (E/(1.0+nu))*0.5;

      /* \sigma^y */
      
      /* a^{yx} */
      
      vu_diff_ten[k*4+1] = (E/(1.0+nu))*0.5;
      vu_diff_ten[k*4+2] = (E/(1.0+nu))*(nu/(1.0-2.0*nu));
      
      /* a^{yy} */
      
      vv_diff_ten[k*4+0] = (E/(1.0+nu))*0.5;
      vv_diff_ten[k*4+3] = (E/(1.0+nu))*(1.0+nu/(1.0-2.0*nu));
      
      u_force[k] = -g[0];
      v_force[k] = -g[1];
    }
}

void LinearElasticity_3D_Evaluate(const int nPoints,
                                  const double E,
                                  const double nu,
                                  const double *g,
                                  const double *u,
                                  const double *v,
                                  const double *w,
                                  double *uu_diff_ten,double *uv_diff_ten,double *uw_diff_ten,
                                  double *vu_diff_ten,double *vv_diff_ten,double *vw_diff_ten,
                                  double *wu_diff_ten,double *wv_diff_ten,double *ww_diff_ten,
                                  double *u_force,
                                  double *v_force,
                                  double *w_force)
{
  int k;
  for (k=0;k<nPoints;k++)
    {
      /* \sigma^x */

      /* a^{xx} */
      uu_diff_ten[k*9+0] = (E/(1.0+nu))*(1.0+nu/(1.0-2.0*nu));
      uu_diff_ten[k*9+4] = (E/(1.0+nu))*0.5;
      uu_diff_ten[k*9+8] = (E/(1.0+nu))*0.5;
      
      /* a^{xy} */
      uv_diff_ten[k*9+1] = (E/(1.0+nu))*(nu/(1.0-2.0*nu));
      uv_diff_ten[k*9+3] = (E/(1.0+nu))*0.5;

      /* a^{xz} */
      uw_diff_ten[k*9+2] = (E/(1.0+nu))*(nu/(1.0-2.0*nu));
      uw_diff_ten[k*9+6] = (E/(1.0+nu))*0.5;
      
      /* \sigma^y */
      
      /* a^{yx} */
      vu_diff_ten[k*9+1] = (E/(1.0+nu))*0.5;
      vu_diff_ten[k*9+3] = (E/(1.0+nu))*(nu/(1.0-2.0*nu));
      
      /* a^{yy} */
      
      vv_diff_ten[k*9+0] = (E/(1.0+nu))*0.5;
      vv_diff_ten[k*9+4] = (E/(1.0+nu))*(1.0+nu/(1.0-2.0*nu));
      vv_diff_ten[k*9+8] = (E/(1.0+nu))*0.5;
      
      /* a^{yz} */
      vw_diff_ten[k*9+5] = (E/(1.0+nu))*(nu/(1.0-2.0*nu));
      vw_diff_ten[k*9+7] = (E/(1.0+nu))*0.5;
      
      /* \sigma^z */
      
      /* a^{zx} */
      wu_diff_ten[k*9+2] = (E/(1.0+nu))*0.5;
      wu_diff_ten[k*9+6] = (E/(1.0+nu))*(nu/(1.0-2.0*nu));
      
      /* a^{zy} */
      wv_diff_ten[k*9+5] = (E/(1.0+nu))*0.5;
      wv_diff_ten[k*9+7] = (E/(1.0+nu))*(nu/(1.0-2.0*nu));
      
      /* a^{zz} */
      ww_diff_ten[k*9+0] = (E/(1.0+nu))*0.5;
      ww_diff_ten[k*9+4] = (E/(1.0+nu))*0.5;
      ww_diff_ten[k*9+8] = (E/(1.0+nu))*(1.0+nu/(1.0-2.0*nu));
      
      u_force[k] = -g[0];
      v_force[k] = -g[1];
      w_force[k] = -g[2];
    }
}

void MovingMesh_1D_Evaluate(const int nPoints,
			    const double E0,
			    const double nu,
			    const double *g,
			    const double *det_J,
			    const double *u,
			    double *uu_diff_ten,
			    double *u_force)
{
  double E;
  int k;
  for (k=0;k<nPoints;k++)
    {
      E = E0/det_J[k];
      /* \sigma^x */
      
      /* a^{xx} */
      uu_diff_ten[k*1+0] = (E/(1.0+nu))*(1.0+nu/(1.0-2.0*nu));
    }
}

void MovingMesh_2D_Evaluate(const int nPoints,
			    const double E0,
			    const double nu,
			    const double *g,
			    const double *det_J,
			    const double *u,
			    const double *v,
			    double *uu_diff_ten,double *uv_diff_ten,
			    double *vu_diff_ten,double *vv_diff_ten,
			    double *u_force,
			    double *v_force)
{
  double E;
  int k;
  for (k=0;k<nPoints;k++)
    {
      E = E0/det_J[k];
      /* \sigma^x */
      
      /* a^{xx} */
      uu_diff_ten[k*4+0] = (E/(1.0+nu))*(1.0+nu/(1.0-2.0*nu));
      uu_diff_ten[k*4+3] = (E/(1.0+nu))*0.5;
      
      /* a^{xy} */
      
      uv_diff_ten[k*4+1] = (E/(1.0+nu))*(nu/(1.0-2.0*nu));
      uv_diff_ten[k*4+2] = (E/(1.0+nu))*0.5;
      
      /* \sigma^y */
      
      /* a^{yx} */
      
      vu_diff_ten[k*4+1] = (E/(1.0+nu))*0.5;
      vu_diff_ten[k*4+2] = (E/(1.0+nu))*(nu/(1.0-2.0*nu));
      
      /* a^{yy} */
      
      vv_diff_ten[k*4+0] = (E/(1.0+nu))*0.5;
      vv_diff_ten[k*4+3] = (E/(1.0+nu))*(1.0+nu/(1.0-2.0*nu));
      
      u_force[k] = -g[0];
      v_force[k] = -g[1];
    }
}

void MovingMesh_3D_Evaluate(const int nPoints,
			    const double E0,
			    const double nu,
			    const double *g,
			    const double *det_J,
			    const double *u,
			    const double *v,
			    const double *w,
			    double *uu_diff_ten,double *uv_diff_ten,double *uw_diff_ten,
			    double *vu_diff_ten,double *vv_diff_ten,double *vw_diff_ten,
			    double *wu_diff_ten,double *wv_diff_ten,double *ww_diff_ten,
			    double *u_force,
			    double *v_force,
			    double *w_force)
{
  double E;
  int k;
  for (k=0;k<nPoints;k++)
    {
      E = E0/det_J[k];
      /* \sigma^x */

      /* a^{xx} */
      uu_diff_ten[k*9+0] = (E/(1.0+nu))*(1.0+nu/(1.0-2.0*nu));
      uu_diff_ten[k*9+4] = (E/(1.0+nu))*0.5;
      uu_diff_ten[k*9+8] = (E/(1.0+nu))*0.5;
      
      /* a^{xy} */
      uv_diff_ten[k*9+1] = (E/(1.0+nu))*(nu/(1.0-2.0*nu));
      uv_diff_ten[k*9+3] = (E/(1.0+nu))*0.5;

      /* a^{xz} */
      uw_diff_ten[k*9+2] = (E/(1.0+nu))*(nu/(1.0-2.0*nu));
      uw_diff_ten[k*9+6] = (E/(1.0+nu))*0.5;
      
      /* \sigma^y */
      
      /* a^{yx} */
      vu_diff_ten[k*9+1] = (E/(1.0+nu))*0.5;
      vu_diff_ten[k*9+3] = (E/(1.0+nu))*(nu/(1.0-2.0*nu));
      
      /* a^{yy} */
      
      vv_diff_ten[k*9+0] = (E/(1.0+nu))*0.5;
      vv_diff_ten[k*9+4] = (E/(1.0+nu))*(1.0+nu/(1.0-2.0*nu));
      vv_diff_ten[k*9+8] = (E/(1.0+nu))*0.5;
      
      /* a^{yz} */
      vw_diff_ten[k*9+5] = (E/(1.0+nu))*(nu/(1.0-2.0*nu));
      vw_diff_ten[k*9+7] = (E/(1.0+nu))*0.5;
      
      /* \sigma^z */
      
      /* a^{zx} */
      wu_diff_ten[k*9+2] = (E/(1.0+nu))*0.5;
      wu_diff_ten[k*9+6] = (E/(1.0+nu))*(nu/(1.0-2.0*nu));
      
      /* a^{zy} */
      wv_diff_ten[k*9+5] = (E/(1.0+nu))*0.5;
      wv_diff_ten[k*9+7] = (E/(1.0+nu))*(nu/(1.0-2.0*nu));
      
      /* a^{zz} */
      ww_diff_ten[k*9+0] = (E/(1.0+nu))*0.5;
      ww_diff_ten[k*9+4] = (E/(1.0+nu))*0.5;
      ww_diff_ten[k*9+8] = (E/(1.0+nu))*(1.0+nu/(1.0-2.0*nu));
      
      u_force[k] = -g[0];
      v_force[k] = -g[1];
      w_force[k] = -g[2];
    }
}

void levelSetConservationCoefficientsEvaluate(int nPoints,
                                              int nSpace,
                                              double epsHeaviside,
                                              double epsDirac,
                                              double epsDiffusion,
                                              double* u_ls,
                                              double* H_vof,
                                              double* u,
                                              double* r,
                                              double* dr,
                                              double* a)
{
  int i,I,nSpace2=nSpace*nSpace;
/*   double eps=1.0e-1; */

  for (i=0;i<nPoints;i++)
    {
      //      r[i] = (1.0-smoothedHeaviside(epsHeaviside,u[i] + u_ls[i])) - H_vof[i];
      //      dr[i] = -smoothedDirac(epsDirac,u[i] + u_ls[i]);
      r[i] = smoothedHeaviside(epsHeaviside,u[i] + u_ls[i]) - H_vof[i];
      dr[i] = smoothedDirac(epsDirac,u[i] + u_ls[i]);
      for (I=0;I<nSpace;I++)
        a[nSpace2*i+I*nSpace+I] = epsDiffusion;
    }
  
  
/*   for (i=0;i<nPoints;i++) */
/*     { */
/*       if ((H_vof[i] > eps) && (H_vof[i] < (1.0-eps)) ) */
/*         { */
/*           r[i] = smoothedHeaviside(epsHeaviside,u[i] + u_ls[i]) - H_vof[i]; */
/*           dr[i] = smoothedDirac(epsDirac,u[i] + u_ls[i]); */
/*           for (I=0;I<nSpace;I++) */
/*             a[nSpace2*i+I*nSpace+I] = epsDiffusion; */
/*         } */
/*       else */
/*         { */
/*           r[i] = u[i]; */
/*           dr[i] = 1.0; */
/*           for (I=0;I<nSpace;I++) */
/*             a[nSpace2*i+I*nSpace+I] = 0.0; */
/*         } */
/*     } */
}

void levelSetConservationCoefficientsEvaluate_sd(int nPoints,
						 double epsHeaviside,
						 double epsDirac,
						 double* u_ls,
						 double* H_vof,
						 double* u,
						 double* r,
						 double* dr)
{
  int i;
  for (i=0;i<nPoints;i++)
    {
      r[i] = smoothedHeaviside(epsHeaviside,u[i] + u_ls[i]) - H_vof[i];
      dr[i] = smoothedDirac(epsDirac,u[i] + u_ls[i]);
    }
}

void evaluateBuckleyLeverettLiuExample(int nPoints,
				       int nSpace,
				       const double * x,
				       const double * u,
				       double * m,
				       double * dm,
				       double * f,
				       double * df,
				       double * a)
{
  /***********************************************************************
     Buckley Leverett 5 spot example from Liu SIAM Num 93
     
     velocity described by potential phi = -0.01 log((x^2 + y^2)^{1/2})
     
     flux function is f = u^2/(0.2 - 0.4u + 1.2 u^2)

     u is water saturation
   ***********************************************************************/
  double vx,vy,r,drdx,drdy,frac,dfrac,denom,ddenom;
  int k;
  const int nSpace2 = nSpace*nSpace;
  const double eps = 1.0e-3;
  memset(a, 0, nPoints * nSpace2 * sizeof(double));
  /*memset(da, 0, nPoints * nSpace2 * sizeof(double));*/

  for (k = 0; k < nPoints; k++)
    {
      r     = sqrt(x[k*3+0]*x[k*3+0] + x[k*3+1]*x[k*3+1] + eps);
      drdx  = x[k*3+0]/r; drdy = x[k*3+1]/r;
      vx    = 0.01*drdx/r; vy = 0.01*drdy/r;
      denom = (0.2 - 0.4*u[k] + 1.2*u[k]*u[k]);
      ddenom= 2.4*u[k] - 0.4;
      frac= u[k]*u[k]/denom;
      dfrac = 2.0*u[k]/denom - u[k]*u[k]*ddenom/(denom*denom);
      m[k] = u[k];
      dm[k]= 1.0;
	  
      f[k*nSpace+0]=vx*frac;
      f[k*nSpace+1]=vy*frac;
      
      df[k*nSpace+0]=vx*dfrac;
      df[k*nSpace+1]=vy*dfrac;
    }
}

/*Simplified NS in a porous region with spatially variable porosity 
  but uniform mean grain size and other empirical fitting parameters for now
  taken from Breugem etal Journal of Fluid Mechanics 06
  full tensor version needs to be verified may not be correct combination
  with Darcy-Forcheimer drag terms */
void VolumeAveragedNavierStokesFullDevStress_2D_Evaluate(const int nPoints,
							 const double rho,
							 const double mu,
							 const double *meanGrainSize,
							 const double *g,
							 const double *p,
							 const double *grad_p,
							 const double *u,
							 const double *v,
							 const double *porosity,
							 double *mom_u_acc,
							 double *dmom_u_acc_u,
							 double *mom_v_acc,
							 double *dmom_v_acc_v,
							 double *mass_adv,
							 double *dmass_adv_u,
							 double *dmass_adv_v,
							 double *mom_u_adv,
							 double *dmom_u_adv_u,
							 double *dmom_u_adv_v,
							 double *mom_v_adv,
							 double *dmom_v_adv_u,
							 double *dmom_v_adv_v,
							 double *mom_u_diff_ten,
							 double *mom_v_diff_ten,
							 double *mom_uv_diff_ten,
							 double *mom_vu_diff_ten,
							 double *mom_u_source,
							 double *mom_v_source,
							 double *dmom_u_source_u,
							 double *dmom_u_source_v,
							 double *dmom_v_source_u,
							 double *dmom_v_source_v,
							 double *mom_u_ham,
							 double *dmom_u_ham_grad_p,
							 double *mom_v_ham,
							 double *dmom_v_ham_grad_p)
{
  int k;
  double Ftilde,Kinv,uc;
  for (k=0;k<nPoints;k++)
    {
      /** \todo optimize coefficients */
      //u momentum accumulation
      mom_u_acc[k]=porosity[k]*rho*u[k];
      dmom_u_acc_u[k]=porosity[k]*rho;
      
      //v momentum accumulation
      mom_v_acc[k]=porosity[k]*rho*v[k];
      dmom_v_acc_v[k]=porosity[k]*rho;

      //mass advective flux
      mass_adv[k*2+0]=porosity[k]*u[k];
      mass_adv[k*2+1]=porosity[k]*v[k];
      
      dmass_adv_u[k*2+0]=porosity[k];
      dmass_adv_v[k*2+1]=porosity[k];

      //u momentum advective flux
      mom_u_adv[k*2+0]=porosity[k]*rho*u[k]*u[k];
      mom_u_adv[k*2+1]=porosity[k]*rho*u[k]*v[k];

      dmom_u_adv_u[k*2+0]=2.0*porosity[k]*rho*u[k];
      dmom_u_adv_u[k*2+1]=porosity[k]*rho*v[k];

      dmom_u_adv_v[k*2+1]=porosity[k]*rho*u[k];

      //v momentum advective_flux
      mom_v_adv[k*2+0]=porosity[k]*rho*v[k]*u[k];
      mom_v_adv[k*2+1]=porosity[k]*rho*v[k]*v[k];
            
      dmom_v_adv_u[k*2+0]=porosity[k]*rho*v[k];
      
      dmom_v_adv_v[k*2+0]=porosity[k]*rho*u[k];
      dmom_v_adv_v[k*2+1]=2.0*porosity[k]*rho*v[k];

      //u momentum diffusion tensor
      mom_u_diff_ten[k*4+0] = 2.0*porosity[k]*mu;
      mom_u_diff_ten[k*4+3] = porosity[k]*mu;
      mom_uv_diff_ten[k*4+2]= porosity[k]*mu;

      //v momentum diffusion tensor
      mom_v_diff_ten[k*4+0] = porosity[k]*mu;
      mom_v_diff_ten[k*4+3] = 2.0*porosity[k]*mu;
      mom_vu_diff_ten[k*4+1] = porosity[k]*mu;


      //momentum sources
      //porous medium contribution
      //end up with extra porosity term in final expression because multiply whole momentum 
      //equation through by porosity
      uc     = sqrt(u[k]*u[k]+v[k]*v[k]);
      if (fabs(1.0-porosity[k]) < 1.0e-7)
	Ftilde = 0.0;
      else
	Ftilde = porosity[k]*meanGrainSize[k]*1.0e-2/(1.0-porosity[k])/mu;
      /*mwf hack
	Ftilde =0.0;
      */
      //trap divide by zero here 
      if (fabs(porosity[k]) < 1.0e-7)
	Kinv = 0.0;
      else
	Kinv   = 180.0*(1.0-porosity[k])*(1.0-porosity[k])/(meanGrainSize[k]*meanGrainSize[k]*porosity[k]*porosity[k]*porosity[k]);
      
      mom_u_source[k] = -porosity[k]*rho*g[0] + porosity[k]*porosity[k]*mu*Kinv*(1.0+Ftilde*uc)*u[k];
      mom_v_source[k] = -porosity[k]*rho*g[1] + porosity[k]*porosity[k]*mu*Kinv*(1.0+Ftilde*uc)*v[k];
      
      dmom_u_source_u[k] = porosity[k]*porosity[k]*mu*Kinv*(1.0 + Ftilde*(uc + u[k]*u[k]/(uc+1.0e-12)));
      dmom_u_source_v[k] = porosity[k]*porosity[k]*mu*Kinv*(0.0 + Ftilde*(u[k]*v[k]/(uc+1.0e-12)));

      dmom_v_source_v[k] = porosity[k]*porosity[k]*mu*Kinv*(1.0 + Ftilde*(uc + v[k]*v[k]/(uc+1.0e-12)));
      dmom_v_source_u[k] = porosity[k]*porosity[k]*mu*Kinv*(0.0 + Ftilde*(u[k]*v[k]/(uc+1.0e-12)));

      //u momentum Hamiltonian (pressure)

      mom_u_ham[k] = porosity[k]*grad_p[k*2+0];
      dmom_u_ham_grad_p[k*2+0]=porosity[k];
      
      //v momentum Hamiltonian (pressure)
      mom_v_ham[k] = porosity[k]*grad_p[k*2+1];
      dmom_v_ham_grad_p[k*2+1]=porosity[k];
    }
}
void VolumeAveragedNavierStokesFullDevStress_3D_Evaluate(const int nPoints,
							 const double rho,
							 const double mu,
							 const double* meanGrainSize,
							 const double* g,
							 const double *p,
							 const double *grad_p,
							 const double *u,
							 const double *v,
							 const double *w,
							 const double *porosity,
							 double *mom_u_acc,
							 double *dmom_u_acc_u,
							 double *mom_v_acc,
							 double *dmom_v_acc_v,
							 double *mom_w_acc,
							 double *dmom_w_acc_w,
							 double *mass_adv,
							 double *dmass_adv_u,
							 double *dmass_adv_v,
							 double *dmass_adv_w,
							 double *mom_u_adv,
							 double *dmom_u_adv_u,
							 double *dmom_u_adv_v,
							 double *dmom_u_adv_w,
							 double *mom_v_adv,
							 double *dmom_v_adv_u,
							 double *dmom_v_adv_v,
							 double *dmom_v_adv_w,
							 double *mom_w_adv,
							 double *dmom_w_adv_u,
							 double *dmom_w_adv_v,
							 double *dmom_w_adv_w,
							 double *mom_u_diff_ten,
							 double *mom_v_diff_ten,
							 double *mom_w_diff_ten,
							 double *mom_uv_diff_ten,
							 double *mom_uw_diff_ten,
							 double *mom_vu_diff_ten,
							 double *mom_vw_diff_ten,
							 double *mom_wu_diff_ten,
							 double *mom_wv_diff_ten,
							 double *mom_u_source,
							 double *mom_v_source,
							 double *mom_w_source,
							 double *dmom_u_source_u,
							 double *dmom_u_source_v,
							 double *dmom_u_source_w,
							 double *dmom_v_source_u,
							 double *dmom_v_source_v,
							 double *dmom_v_source_w,
							 double *dmom_w_source_u,
							 double *dmom_w_source_v,
							 double *dmom_w_source_w,
							 double *mom_u_ham,
							 double *dmom_u_ham_grad_p,
							 double *mom_v_ham,
							 double *dmom_v_ham_grad_p,
							 double *mom_w_ham,
							 double *dmom_w_ham_grad_p)
{
  int k;
  double Ftilde,Kinv,uc;
  for (k=0;k<nPoints;k++)
    {
      /** \todo optimize coefficients */
      //u momentum accumulation
      mom_u_acc[k]=porosity[k]*rho*u[k];
      dmom_u_acc_u[k]=porosity[k]*rho;
      
      //v momentum accumulation
      mom_v_acc[k]=porosity[k]*rho*v[k];
      dmom_v_acc_v[k]=porosity[k]*rho;

      //w momentum accumulation
      mom_w_acc[k]=porosity[k]*rho*w[k];
      dmom_w_acc_w[k]=porosity[k]*rho;


     //mass advective flux
      mass_adv[k*3+0]=porosity[k]*u[k];
      mass_adv[k*3+1]=porosity[k]*v[k];
      mass_adv[k*3+2]=porosity[k]*w[k];
      
      dmass_adv_u[k*3+0]=porosity[k];
      dmass_adv_v[k*3+1]=porosity[k];
      dmass_adv_w[k*3+2]=porosity[k];

      //u momentum advective flux
      mom_u_adv[k*3+0]=porosity[k]*rho*u[k]*u[k];
      mom_u_adv[k*3+1]=porosity[k]*rho*u[k]*v[k];
      mom_u_adv[k*3+2]=porosity[k]*rho*u[k]*w[k];

      dmom_u_adv_u[k*3+0]=2.0*porosity[k]*rho*u[k];
      dmom_u_adv_u[k*3+1]=porosity[k]*rho*v[k];
      dmom_u_adv_u[k*3+2]=porosity[k]*rho*w[k];

      dmom_u_adv_v[k*3+1]=porosity[k]*rho*u[k];
      
      dmom_u_adv_w[k*3+2]=porosity[k]*rho*u[k];

      //v momentum advective_flux
      mom_v_adv[k*3+0]=porosity[k]*rho*v[k]*u[k];
      mom_v_adv[k*3+1]=porosity[k]*rho*v[k]*v[k];
      mom_v_adv[k*3+2]=porosity[k]*rho*v[k]*w[k];
            
      dmom_v_adv_u[k*3+0]=porosity[k]*rho*v[k];
      
      dmom_v_adv_w[k*3+2]=porosity[k]*rho*v[k];
      
      dmom_v_adv_v[k*3+0]=porosity[k]*rho*u[k];
      dmom_v_adv_v[k*3+1]=2.0*porosity[k]*rho*v[k];
      dmom_v_adv_v[k*3+2]=porosity[k]*rho*w[k];

      //w momentum advective_flux
      mom_w_adv[k*3+0]=porosity[k]*rho*w[k]*u[k];
      mom_w_adv[k*3+1]=porosity[k]*rho*w[k]*v[k];
      mom_w_adv[k*3+2]=porosity[k]*rho*w[k]*w[k];
            
      dmom_w_adv_u[k*3+0]=porosity[k]*rho*w[k];
      
      dmom_w_adv_v[k*3+1]=porosity[k]*rho*w[k];
      
      dmom_w_adv_w[k*3+0]=porosity[k]*rho*u[k];
      dmom_w_adv_w[k*3+1]=porosity[k]*rho*v[k];
      dmom_w_adv_w[k*3+2]=2.0*porosity[k]*rho*w[k];

      //u momentum diffusion tensor
      mom_u_diff_ten[k*9+0] = 2.0*porosity[k]*mu;
      mom_u_diff_ten[k*9+4] = porosity[k]*mu;
      mom_u_diff_ten[k*9+8] = porosity[k]*mu;

      mom_uv_diff_ten[k*9+3]=porosity[k]*mu;

      mom_uw_diff_ten[k*9+6]=porosity[k]*mu;

      //v momentum diffusion tensor
      mom_v_diff_ten[k*9+0] = porosity[k]*mu;
      mom_v_diff_ten[k*9+4] = 2.0*porosity[k]*mu;
      mom_v_diff_ten[k*9+8] = porosity[k]*mu;

      mom_vu_diff_ten[k*9+1]=porosity[k]*mu;

      mom_vw_diff_ten[k*9+7]=porosity[k]*mu;

      //w momentum diffusion tensor
      mom_w_diff_ten[k*9+0] = porosity[k]*mu;
      mom_w_diff_ten[k*9+4] = porosity[k]*mu;
      mom_w_diff_ten[k*9+8] = 2.0*porosity[k]*mu;

      mom_wu_diff_ten[k*9+2]=porosity[k]*mu;

      mom_wv_diff_ten[k*9+5]=porosity[k]*mu;

      //momentum sources
      //porous medium contribution
      //end up with extra porosity term in final expression because multiply whole momentum 
      //equation through by porosity
      uc     = sqrt(u[k]*u[k]+v[k]*v[k]+w[k]*w[k]);
      if (fabs(1.0-porosity[k]) < 1.0e-7)
	Ftilde = 0.0;
      else
	Ftilde = porosity[k]*meanGrainSize[k]*1.0e-2/(1.0-porosity[k])/mu;
      /*mwf hack
	Ftilde =0.0;
      */
      //trap divide by zero here 
      if (fabs(porosity[k]) < 1.0e-7)
	Kinv = 0.0;
      else
	Kinv   = 180.0*(1.0-porosity[k])*(1.0-porosity[k])/(meanGrainSize[k]*meanGrainSize[k]*porosity[k]*porosity[k]*porosity[k]);
 
      mom_u_source[k] = -porosity[k]*rho*g[0] + porosity[k]*porosity[k]*mu*Kinv*(1.0+Ftilde*uc)*u[k]; 
      mom_v_source[k] = -porosity[k]*rho*g[1] + porosity[k]*porosity[k]*mu*Kinv*(1.0+Ftilde*uc)*v[k]; 
      mom_w_source[k] = -porosity[k]*rho*g[2] + porosity[k]*porosity[k]*mu*Kinv*(1.0+Ftilde*uc)*w[k]; 

      dmom_u_source_u[k] = porosity[k]*porosity[k]*mu*Kinv*(1.0 + Ftilde*(uc + u[k]*u[k]/(uc+1.0e-12)));
      dmom_u_source_v[k] = porosity[k]*porosity[k]*mu*Kinv*(0.0 + Ftilde*(v[k]*u[k]/(uc+1.0e-12)));
      dmom_u_source_w[k] = porosity[k]*porosity[k]*mu*Kinv*(0.0 + Ftilde*(w[k]*u[k]/(uc+1.0e-12)));

      dmom_v_source_v[k] = porosity[k]*porosity[k]*mu*Kinv*(1.0 + Ftilde*(uc + v[k]*v[k]/(uc+1.0e-12)));
      dmom_v_source_u[k] = porosity[k]*porosity[k]*mu*Kinv*(0.0 + Ftilde*(u[k]*v[k]/(uc+1.0e-12)));
      dmom_v_source_w[k] = porosity[k]*porosity[k]*mu*Kinv*(0.0 + Ftilde*(w[k]*v[k]/(uc+1.0e-12)));

      dmom_w_source_w[k] = porosity[k]*porosity[k]*mu*Kinv*(1.0 + Ftilde*(uc + w[k]*w[k]/(uc+1.0e-12)));
      dmom_w_source_u[k] = porosity[k]*porosity[k]*mu*Kinv*(0.0 + Ftilde*(u[k]*w[k]/(uc+1.0e-12)));
      dmom_w_source_v[k] = porosity[k]*porosity[k]*mu*Kinv*(0.0 + Ftilde*(v[k]*w[k]/(uc+1.0e-12)));

      //u momentum Hamiltonian (pressure)
      mom_u_ham[k] = porosity[k]*grad_p[k*3+0];
      dmom_u_ham_grad_p[k*3+0]=porosity[k];
      
      //v momentum Hamiltonian (pressure)
      mom_v_ham[k] = porosity[k]*grad_p[k*3+1];
      dmom_v_ham_grad_p[k*3+1]=porosity[k];

      //w momentum Hamiltonian (pressure)
      mom_w_ham[k] = porosity[k]*grad_p[k*3+2];
      dmom_w_ham_grad_p[k*3+2]=porosity[k];
    }
}

void VolumeAveragedTwophaseNavierStokes_ST_LS_SO_2D_Evaluate(const int nPoints,
							     const int killNonlinearDrag,
							     const double eps_rho,
							     const double eps_mu,
							     const double sigma,
							     const double rho_0,
							     const double nu_0,
							     const double rho_1,
							     const double nu_1,
							     const double* meanGrainSize,
							     const double* g,
							     const double* phi,
							     const double* n,
							     const double* kappa,
							     const double *p,
							     const double *grad_p,
							     const double *u,
							     const double *v,
							     const double *porosity,
							     double *mom_u_acc,
							     double *dmom_u_acc_u,
							     double *mom_v_acc,
							     double *dmom_v_acc_v,
							     double *mass_adv,
							     double *dmass_adv_u,
							     double *dmass_adv_v,
							     double *mom_u_adv,
							     double *dmom_u_adv_u,
							     double *dmom_u_adv_v,
							     double *mom_v_adv,
							     double *dmom_v_adv_u,
							     double *dmom_v_adv_v,
							     double *mom_u_diff_ten,
							     double *mom_v_diff_ten,
							     double *mom_uv_diff_ten,
							     double *mom_vu_diff_ten,
							     double *mom_u_source,
							     double *mom_v_source,
							     double *dmom_u_source_u,
							     double *dmom_u_source_v,
							     double *dmom_v_source_u,
							     double *dmom_v_source_v,
							     double *mom_u_ham,
							     double *dmom_u_ham_grad_p,
							     double *mom_v_ham,
							     double *dmom_v_ham_grad_p)
{
  int k;
  double rho,nu,mu,H_rho,d_rho,H_mu,d_mu,norm_n;
  double Ftilde,Kinv,uc;
  double nonlinearDragFactor = 1.0;
  if (killNonlinearDrag)
    nonlinearDragFactor = 0.0;
  for (k=0;k<nPoints;k++)
    {
      /** \todo optimize coefficients */
      H_rho = smoothedHeaviside(eps_rho,phi[k]);
      d_rho = smoothedDirac(eps_rho,phi[k]);
      H_mu = smoothedHeaviside(eps_mu,phi[k]);
      d_mu = smoothedDirac(eps_mu,phi[k]);

      rho = rho_0*(1.0-H_rho)+rho_1*H_rho;
      nu  = nu_0*(1.0-H_mu)+nu_1*H_mu;
      mu  = rho_0*nu_0*(1.0-H_mu)+rho_1*nu_1*H_mu;

      //u momentum accumulation
      mom_u_acc[k]=porosity[k]*u[k];
      dmom_u_acc_u[k]=porosity[k];
      
      //v momentum accumulation
      mom_v_acc[k]=porosity[k]*v[k];
      dmom_v_acc_v[k]=porosity[k];


     //mass advective flux
      mass_adv[k*2+0]=porosity[k]*u[k];
      mass_adv[k*2+1]=porosity[k]*v[k];
      
      dmass_adv_u[k*2+0]=porosity[k];
      dmass_adv_v[k*2+1]=porosity[k];

      //u momentum advective flux
      mom_u_adv[k*2+0]=porosity[k]*u[k]*u[k];
      mom_u_adv[k*2+1]=porosity[k]*u[k]*v[k];

      dmom_u_adv_u[k*2+0]=2.0*porosity[k]*u[k];
      dmom_u_adv_u[k*2+1]=porosity[k]*v[k];

      dmom_u_adv_v[k*2+1]=porosity[k]*u[k];

      //v momentum advective_flux
      mom_v_adv[k*2+0]=porosity[k]*v[k]*u[k];
      mom_v_adv[k*2+1]=porosity[k]*v[k]*v[k];
            
      dmom_v_adv_u[k*2+0]=porosity[k]*v[k];
      
      dmom_v_adv_v[k*2+0]=porosity[k]*u[k];
      dmom_v_adv_v[k*2+1]=2.0*porosity[k]*v[k];

#ifdef SCALAR_DIFFUSION
     //u momentum diffusion tensor
      mom_u_diff_ten[k*4+0] = nu*porosity[k];
      mom_u_diff_ten[k*4+3] = nu*porosity[k];

      //v momentum diffusion tensor
      mom_v_diff_ten[k*4+0] = nu*porosity[k];
      mom_v_diff_ten[k*4+3] = nu*porosity[k];
#else
      //u momentum diffusion tensor
      mom_u_diff_ten[k*4+0] = 2.0*porosity[k]*nu;
      mom_u_diff_ten[k*4+3] = porosity[k]*nu;
      mom_uv_diff_ten[k*4+2]=porosity[k]*nu;

      //v momentum diffusion tensor
      mom_v_diff_ten[k*4+0] = porosity[k]*nu;
      mom_v_diff_ten[k*4+3] = 2.0*porosity[k]*nu;
      mom_vu_diff_ten[k*4+1] = porosity[k]*nu;
#endif

      //momentum sources
      //two-phase flow contribution
      norm_n = sqrt(n[k*2+0]*n[k*2+0]+n[k*2+1]*n[k*2+1]);
      //porous medium contribution
      //end up with extra porosity term in final expression because multiply whole momentum 
      //equation through by porosity
      uc     = sqrt(u[k]*u[k]+v[k]*v[k]);
      if (fabs(1.0-porosity[k]) < 1.0e-7)
	Ftilde = 0.0;
      else
	Ftilde = porosity[k]*meanGrainSize[k]*1.0e-2/(1.0-porosity[k])/nu;
      /*mwf hack
	Ftilde =0.0;
      */
       //allow only linear resistance for sponge layers etc
      Ftilde *= nonlinearDragFactor;
      //trap divide by zero here 
      if (fabs(porosity[k]) < 1.0e-7)
	Kinv = 0.0;
      else
	Kinv   = 180.0*(1.0-porosity[k])*(1.0-porosity[k])/(meanGrainSize[k]*meanGrainSize[k]*porosity[k]*porosity[k]*porosity[k]);
      
      mom_u_source[k] = -porosity[k]*g[0] - porosity[k]*d_mu*sigma*kappa[k]*n[k*2+0]/(norm_n) 
	+ porosity[k]*porosity[k]*nu*Kinv*(1.0+Ftilde*uc)*u[k];
      mom_v_source[k] = -porosity[k]*rho*g[1] - porosity[k]*d_mu*sigma*kappa[k]*n[k*2+1]/(norm_n)
	+ porosity[k]*porosity[k]*nu*Kinv*(1.0+Ftilde*uc)*v[k];
      
      dmom_u_source_u[k] = porosity[k]*porosity[k]*nu*Kinv*(1.0 + Ftilde*(uc + u[k]*u[k]/(uc+1.0e-12)));
      dmom_u_source_v[k] = porosity[k]*porosity[k]*nu*Kinv*(0.0 + Ftilde*(u[k]*v[k]/(uc+1.0e-12)));

      dmom_v_source_v[k] = porosity[k]*porosity[k]*nu*Kinv*(1.0 + Ftilde*(uc + v[k]*v[k]/(uc+1.0e-12)));
      dmom_v_source_u[k] = porosity[k]*porosity[k]*nu*Kinv*(0.0 + Ftilde*(u[k]*v[k]/(uc+1.0e-12)));

      //u momentum Hamiltonian (pressure)

      mom_u_ham[k] = porosity[k]*grad_p[k*2+0]/rho;
      dmom_u_ham_grad_p[k*2+0]=porosity[k]/rho;
      
      //v momentum Hamiltonian (pressure)
      mom_v_ham[k] = porosity[k]*grad_p[k*2+1]/rho;
      dmom_v_ham_grad_p[k*2+1]=porosity[k]/rho;
      //compressible form
/*       //u momentum accumulation */
/*       mom_u_acc[k]=porosity[k]*rho*u[k]; */
/*       dmom_u_acc_u[k]=porosity[k]*rho; */
      
/*       //v momentum accumulation */
/*       mom_v_acc[k]=porosity[k]*rho*v[k]; */
/*       dmom_v_acc_v[k]=porosity[k]*rho; */


/*      //mass advective flux */
/*       mass_adv[k*2+0]=porosity[k]*u[k]; */
/*       mass_adv[k*2+1]=porosity[k]*v[k]; */
      
/*       dmass_adv_u[k*2+0]=porosity[k]; */
/*       dmass_adv_v[k*2+1]=porosity[k]; */

/*       //u momentum advective flux */
/*       mom_u_adv[k*2+0]=porosity[k]*rho*u[k]*u[k]; */
/*       mom_u_adv[k*2+1]=porosity[k]*rho*u[k]*v[k]; */

/*       dmom_u_adv_u[k*2+0]=2.0*porosity[k]*rho*u[k]; */
/*       dmom_u_adv_u[k*2+1]=porosity[k]*rho*v[k]; */

/*       dmom_u_adv_v[k*2+1]=porosity[k]*rho*u[k]; */

/*       //v momentum advective_flux */
/*       mom_v_adv[k*2+0]=porosity[k]*rho*v[k]*u[k]; */
/*       mom_v_adv[k*2+1]=porosity[k]*rho*v[k]*v[k]; */
            
/*       dmom_v_adv_u[k*2+0]=porosity[k]*rho*v[k]; */
      
/*       dmom_v_adv_v[k*2+0]=porosity[k]*rho*u[k]; */
/*       dmom_v_adv_v[k*2+1]=2.0*porosity[k]*rho*v[k]; */

/*       //u momentum diffusion tensor */
/*       mom_u_diff_ten[k*4+0] = 2.0*porosity[k]*mu; */
/*       mom_u_diff_ten[k*4+3] = porosity[k]*mu; */
/*       mom_uv_diff_ten[k*4+2]=porosity[k]*mu; */

/*       //v momentum diffusion tensor */
/*       mom_v_diff_ten[k*4+0] = porosity[k]*mu; */
/*       mom_v_diff_ten[k*4+3] = 2.0*porosity[k]*mu; */
/*       mom_vu_diff_ten[k*4+1] = porosity[k]*mu; */


/*       //momentum sources */
/*       //two-phase flow contribution */
/*       norm_n = sqrt(n[k*2+0]*n[k*2+0]+n[k*2+1]*n[k*2+1]); */
/*       //porous medium contribution */
/*       //end up with extra porosity term in final expression because multiply whole momentum  */
/*       //equation through by porosity */
/*       uc     = sqrt(u[k]*u[k]+v[k]*v[k]); */
/*       if (fabs(1.0-porosity[k]) < 1.0e-7) */
/* 	Ftilde = 0.0; */
/*       else */
/* 	Ftilde = porosity[k]*meanGrainSize[k]*1.0e-2/(1.0-porosity[k])/mu; */
/*       /\*mwf hack */
/* 	Ftilde =0.0; */
/*       *\/ */
/*       //trap divide by zero here  */
/*       if (fabs(porosity[k]) < 1.0e-7) */
/* 	Kinv = 0.0; */
/*       else */
/* 	Kinv   = 180.0*(1.0-porosity[k])*(1.0-porosity[k])/(meanGrainSize[k]*meanGrainSize[k]*porosity[k]*porosity[k]*porosity[k]); */
      
/*       mom_u_source[k] = -porosity[k]*rho*g[0] - d_mu*sigma*kappa[k]*n[k*2+0]/(norm_n)  */
/* 	+ porosity[k]*porosity[k]*mu*Kinv*(1.0+Ftilde*uc)*u[k]; */
/*       mom_v_source[k] = -porosity[k]*rho*g[1] - d_mu*sigma*kappa[k]*n[k*2+1]/(norm_n) */
/* 	+ porosity[k]*porosity[k]*mu*Kinv*(1.0+Ftilde*uc)*v[k]; */
      
/*       dmom_u_source_u[k] = porosity[k]*porosity[k]*mu*Kinv*(1.0 + Ftilde*(uc + u[k]*u[k]/(uc+1.0e-12))); */
/*       dmom_u_source_v[k] = porosity[k]*porosity[k]*mu*Kinv*(0.0 + Ftilde*(u[k]*v[k]/(uc+1.0e-12))); */

/*       dmom_v_source_v[k] = porosity[k]*porosity[k]*mu*Kinv*(1.0 + Ftilde*(uc + v[k]*v[k]/(uc+1.0e-12))); */
/*       dmom_v_source_u[k] = porosity[k]*porosity[k]*mu*Kinv*(0.0 + Ftilde*(u[k]*v[k]/(uc+1.0e-12))); */

/*       //u momentum Hamiltonian (pressure) */

/*       mom_u_ham[k] = porosity[k]*grad_p[k*2+0]; */
/*       dmom_u_ham_grad_p[k*2+0]=porosity[k]; */
      
/*       //v momentum Hamiltonian (pressure) */
/*       mom_v_ham[k] = porosity[k]*grad_p[k*2+1]; */
/*       dmom_v_ham_grad_p[k*2+1]=porosity[k]; */
    }
}
void VolumeAveragedTwophaseNavierStokes_ST_LS_SO_2D_Evaluate_sd(const int nPoints,
								const int killNonlinearDrag,
								const double eps_rho,
								const double eps_mu,
								const double sigma,
								const double rho_0,
								const double nu_0,
								const double rho_1,
								const double nu_1,
								const double* meanGrainSize,
								const double* g,
								const double* phi,
								const double* n,
								const double* kappa,
								const double *p,
								const double *grad_p,
								const double *u,
								const double *v,
								const double *porosity,
								double *mom_u_acc,
								double *dmom_u_acc_u,
								double *mom_v_acc,
								double *dmom_v_acc_v,
								double *mass_adv,
								double *dmass_adv_u,
								double *dmass_adv_v,
								double *mom_u_adv,
								double *dmom_u_adv_u,
								double *dmom_u_adv_v,
								double *mom_v_adv,
								double *dmom_v_adv_u,
								double *dmom_v_adv_v,
								double *mom_u_diff_ten,
								double *mom_v_diff_ten,
								double *mom_uv_diff_ten,
								double *mom_vu_diff_ten,
								double *mom_u_source,
								double *mom_v_source,
								double *dmom_u_source_u,
								double *dmom_u_source_v,
								double *dmom_v_source_u,
								double *dmom_v_source_v,
								double *mom_u_ham,
								double *dmom_u_ham_grad_p,
								double *mom_v_ham,
								double *dmom_v_ham_grad_p)
{
  int k;
  double rho,nu,mu,H_rho,d_rho,H_mu,d_mu,norm_n,uc,Ftilde,Kinv;
  const double div_eps = 1.0e-6;
  double nonlinearDragFactor = 1.0;
  if (killNonlinearDrag)
    nonlinearDragFactor = 0.0;
  for (k=0;k<nPoints;k++)
    {
      /** \todo optimize coefficients */
      H_rho = smoothedHeaviside(eps_rho,phi[k]);
      d_rho = smoothedDirac(eps_rho,phi[k]);
      H_mu = smoothedHeaviside(eps_mu,phi[k]);
      d_mu = smoothedDirac(eps_mu,phi[k]);

      rho = rho_0*(1.0-H_rho)+rho_1*H_rho;
      nu  = nu_0*(1.0-H_mu)+nu_1*H_mu;
      mu  = rho_0*nu_0*(1.0-H_mu)+rho_1*nu_1*H_mu;
      //u momentum accumulation
      mom_u_acc[k]=porosity[k]*u[k];
      dmom_u_acc_u[k]=porosity[k];
      
      //v momentum accumulation
      mom_v_acc[k]=porosity[k]*v[k];
      dmom_v_acc_v[k]=porosity[k];

      //mass advective flux
      mass_adv[k*2+0]=porosity[k]*u[k];
      mass_adv[k*2+1]=porosity[k]*v[k];
      
      dmass_adv_u[k*2+0]=porosity[k];
      dmass_adv_v[k*2+1]=porosity[k];

      //u momentum advective flux
      mom_u_adv[k*2+0]=porosity[k]*u[k]*u[k];
      mom_u_adv[k*2+1]=porosity[k]*u[k]*v[k];

      dmom_u_adv_u[k*2+0]=2.0*porosity[k]*u[k];
      dmom_u_adv_u[k*2+1]=porosity[k]*v[k];

      dmom_u_adv_v[k*2+1]=porosity[k]*u[k];

      //v momentum advective_flux
      mom_v_adv[k*2+0]=porosity[k]*v[k]*u[k];
      mom_v_adv[k*2+1]=porosity[k]*v[k]*v[k];
            
      dmom_v_adv_u[k*2+0]=porosity[k]*v[k];
      
      dmom_v_adv_v[k*2+0]=porosity[k]*u[k];
      dmom_v_adv_v[k*2+1]=2.0*porosity[k]*v[k];

      //u momentum diffusion tensor
      mom_u_diff_ten[k*2+0] = 2.0*porosity[k]*nu;
      mom_u_diff_ten[k*2+1] = porosity[k]*nu;
      mom_uv_diff_ten[k]=porosity[k]*nu;

      //v momentum diffusion tensor
      mom_v_diff_ten[k*2+0] = porosity[k]*nu;
      mom_v_diff_ten[k*2+1] = 2.0*nu*porosity[k];
      mom_vu_diff_ten[k] = nu*porosity[k];

      //momentum sources
      norm_n = sqrt(n[k*2+0]*n[k*2+0]+n[k*2+1]*n[k*2+1]);
      //porous medium contribution
      //end up with extra porosity term in final expression because multiply whole momentum 
      //equation through by porosity
      uc     = sqrt(u[k]*u[k]+v[k]*v[k]);
      if (fabs(1.0-porosity[k]) < 1.0e-7)
	Ftilde = 0.0;
      else
	Ftilde = porosity[k]*meanGrainSize[k]*1.0e-2/(1.0-porosity[k])/nu;
      /*mwf hack
      Ftilde =0.0;
      */
      //allow only linear resistance for sponge layers etc
      Ftilde *= nonlinearDragFactor;
      //trap divide by zero here 
      if (fabs(porosity[k]) < 1.0e-7)
	Kinv = 0.0;
      else
	Kinv   = 180.0*(1.0-porosity[k])*(1.0-porosity[k])/(meanGrainSize[k]*meanGrainSize[k]*porosity[k]*porosity[k]*porosity[k]);
      
      mom_u_source[k] = -porosity[k]*g[0] - porosity[k]*d_mu*sigma*kappa[k]*n[k*2+0]/(norm_n) 
	+ porosity[k]*porosity[k]*nu*Kinv*(1.0+Ftilde*uc)*u[k];
      mom_v_source[k] = -porosity[k]*g[1] - porosity[k]*d_mu*sigma*kappa[k]*n[k*2+1]/(norm_n)
	+ porosity[k]*porosity[k]*nu*Kinv*(1.0+Ftilde*uc)*v[k];

      /*mwf orig*/      
/*       dmom_u_source_u[k] = porosity[k]*porosity[k]*nu*Kinv*(1.0 + Ftilde*(uc + u[k]*u[k]/(uc+div_eps))); */
/*       dmom_u_source_v[k] = porosity[k]*porosity[k]*nu*Kinv*(0.0 + Ftilde*(u[k]*v[k]/(uc+div_eps))); */

/*       dmom_v_source_u[k] = porosity[k]*porosity[k]*nu*Kinv*(0.0 + Ftilde*(u[k]*v[k]/(uc+div_eps))); */
/*       dmom_v_source_v[k] = porosity[k]*porosity[k]*nu*Kinv*(1.0 + Ftilde*(uc + v[k]*v[k]/(uc+div_eps))); */

      dmom_u_source_u[k] = porosity[k]*porosity[k]*nu*Kinv*(1.0 + Ftilde*uc) + 
	porosity[k]*porosity[k]*nu*Kinv*Ftilde*u[k]*u[k]/(uc+div_eps); 
      dmom_u_source_v[k] = porosity[k]*porosity[k]*nu*Kinv*Ftilde*u[k]*v[k]/(uc+div_eps); 

      dmom_v_source_v[k] = porosity[k]*porosity[k]*nu*Kinv*(1.0 + Ftilde*uc) + 
	porosity[k]*porosity[k]*nu*Kinv*Ftilde*v[k]*v[k]/(uc+div_eps); 
      dmom_v_source_u[k] = porosity[k]*porosity[k]*nu*Kinv*Ftilde*u[k]*v[k]/(uc+div_eps); 

      //mwf debug
      //printf("k=%d uc=%g norm_n=%g porosity=%g meanGrain= %g Ftilde=%g Kinv=%g \n",k,uc,norm_n,porosity[k],meanGrainSize[k],Ftilde,Kinv);
      //u momentum Hamiltonian (pressure)

      mom_u_ham[k] = porosity[k]*grad_p[k*2+0]/rho;
      dmom_u_ham_grad_p[k*2+0]=porosity[k]/rho;
      
      //v momentum Hamiltonian (pressure)
      mom_v_ham[k] = porosity[k]*grad_p[k*2+1]/rho;
      dmom_v_ham_grad_p[k*2+1]=porosity[k]/rho;

      /* //compressible form */
      /* //u momentum accumulation */
      /* mom_u_acc[k]=porosity[k]*rho*u[k]; */
      /* dmom_u_acc_u[k]=porosity[k]*rho; */
      
      /* //v momentum accumulation */
      /* mom_v_acc[k]=porosity[k]*rho*v[k]; */
      /* dmom_v_acc_v[k]=porosity[k]*rho; */

      /* //mass advective flux */
      /* mass_adv[k*2+0]=porosity[k]*u[k]; */
      /* mass_adv[k*2+1]=porosity[k]*v[k]; */
      
      /* dmass_adv_u[k*2+0]=porosity[k]; */
      /* dmass_adv_v[k*2+1]=porosity[k]; */

      /* //u momentum advective flux */
      /* mom_u_adv[k*2+0]=porosity[k]*rho*u[k]*u[k]; */
      /* mom_u_adv[k*2+1]=porosity[k]*rho*u[k]*v[k]; */

      /* dmom_u_adv_u[k*2+0]=porosity[k]*rho*2.0*u[k]; */
      /* dmom_u_adv_u[k*2+1]=porosity[k]*rho*v[k]; */

      /* dmom_u_adv_v[k*2+1]=porosity[k]*rho*u[k]; */

      /* //v momentum advective_flux */
      /* mom_v_adv[k*2+0]=porosity[k]*rho*v[k]*u[k]; */
      /* mom_v_adv[k*2+1]=porosity[k]*rho*v[k]*v[k]; */
            
      /* dmom_v_adv_u[k*2+0]=porosity[k]*rho*v[k]; */
      
      /* dmom_v_adv_v[k*2+0]=porosity[k]*rho*u[k]; */
      /* dmom_v_adv_v[k*2+1]=porosity[k]*rho*2.0*v[k]; */

      /* //u momentum diffusion tensor */
      /* mom_u_diff_ten[k*2+0] = 2.0*porosity[k]*mu; */
      /* mom_u_diff_ten[k*2+1] = porosity[k]*mu; */
      /* mom_uv_diff_ten[k]=porosity[k]*mu; */

      /* //v momentum diffusion tensor */
      /* mom_v_diff_ten[k*2+0] = porosity[k]*mu; */
      /* mom_v_diff_ten[k*2+1] = 2.0*porosity[k]*mu; */
      /* mom_vu_diff_ten[k] = porosity[k]*mu; */

      /* //momentum sources */
      /* norm_n = sqrt(n[k*2+0]*n[k*2+0]+n[k*2+1]*n[k*2+1]); */
      //porous medium contribution
      //end up with extra porosity term in final expression because multiply whole momentum 
      //equation through by porosity
/*       uc     = sqrt(u[k]*u[k]+v[k]*v[k]); */
/*       if (fabs(1.0-porosity[k]) < 1.0e-7) */
/* 	Ftilde = 0.0; */
/*       else */
/* 	Ftilde = porosity[k]*meanGrainSize[k]*1.0e-2/(1.0-porosity[k])/mu; */
/*       /\*mwf hack */
/* 	Ftilde =0.0; */
/*       *\/ */
/*       //trap divide by zero here  */
/*       if (fabs(porosity[k]) < 1.0e-7) */
/* 	Kinv = 0.0; */
/*       else */
/* 	Kinv   = 180.0*(1.0-porosity[k])*(1.0-porosity[k])/(meanGrainSize[k]*meanGrainSize[k]*porosity[k]*porosity[k]*porosity[k]); */
      
/*       mom_u_source[k] = -porosity[k]*rho*g[0] - porosity[k]*d_mu*sigma*kappa[k]*n[k*2+0]/(norm_n)  */
/* 	+ porosity[k]*porosity[k]*mu*Kinv*(1.0+Ftilde*uc)*u[k]; */
/*       mom_v_source[k] = -porosity[k]*rho*g[1] - porosity[k]*d_mu*sigma*kappa[k]*n[k*2+1]/(norm_n) */
/* 	+ porosity[k]*porosity[k]*mu*Kinv*(1.0+Ftilde*uc)*v[k]; */
      
/*       dmom_u_source_u[k] = porosity[k]*porosity[k]*mu*Kinv*(1.0 + Ftilde*(uc + u[k]*u[k]/(uc+div_eps))); */
/*       dmom_u_source_v[k] = porosity[k]*porosity[k]*mu*Kinv*(0.0 + Ftilde*(u[k]*v[k]/(uc+div_eps))); */

/*       dmom_v_source_v[k] = porosity[k]*porosity[k]*mu*Kinv*(1.0 + Ftilde*(uc + v[k]*v[k]/(uc+div_eps))); */
/*       dmom_v_source_u[k] = porosity[k]*porosity[k]*mu*Kinv*(0.0 + Ftilde*(u[k]*v[k]/(uc+div_eps))); */
      
      /* //u momentum Hamiltonian (pressure) */

      /* mom_u_ham[k] = grad_p[k*2+0]*porosity[k]; */
      /* dmom_u_ham_grad_p[k*2+0]=porosity[k]; */
      
      /* //v momentum Hamiltonian (pressure) */
      /* mom_v_ham[k] = grad_p[k*2+1]*porosity[k]; */
      /* dmom_v_ham_grad_p[k*2+1]=porosity[k]; */
    }
}
void VolumeAveragedTwophaseNavierStokes_ST_LS_SO_3D_Evaluate(const int nPoints,
							     const int killNonlinearDrag,
							     const double eps_rho,
							     const double eps_mu,
							     const double sigma,
							     const double rho_0,
							     const double nu_0,
							     const double rho_1,
							     const double nu_1,
							     const double* meanGrainSize,
							     const double* g,
							     const double* phi,
							     const double* n,
							     const double* kappa,
							     const double *p,
							     const double *grad_p,
							     const double *u,
							     const double *v,
							     const double *w,
							     const double *porosity,
							     double *mom_u_acc,
							     double *dmom_u_acc_u,
							     double *mom_v_acc,
							     double *dmom_v_acc_v,
							     double *mom_w_acc,
							     double *dmom_w_acc_w,
							     double *mass_adv,
							     double *dmass_adv_u,
							     double *dmass_adv_v,
							     double *dmass_adv_w,
							     double *mom_u_adv,
							     double *dmom_u_adv_u,
							     double *dmom_u_adv_v,
							     double *dmom_u_adv_w,
							     double *mom_v_adv,
							     double *dmom_v_adv_u,
							     double *dmom_v_adv_v,
							     double *dmom_v_adv_w,
							     double *mom_w_adv,
							     double *dmom_w_adv_u,
							     double *dmom_w_adv_v,
							     double *dmom_w_adv_w,
							     double *mom_u_diff_ten,
							     double *mom_v_diff_ten,
							     double *mom_w_diff_ten,
							     double *mom_uv_diff_ten,
							     double *mom_uw_diff_ten,
							     double *mom_vu_diff_ten,
							     double *mom_vw_diff_ten,
							     double *mom_wu_diff_ten,
							     double *mom_wv_diff_ten,
							     double *mom_u_source,
							     double *mom_v_source,
							     double *mom_w_source,
							     double *dmom_u_source_u,
							     double *dmom_u_source_v,
							     double *dmom_u_source_w,
							     double *dmom_v_source_u,
							     double *dmom_v_source_v,
							     double *dmom_v_source_w,
							     double *dmom_w_source_u,
							     double *dmom_w_source_v,
							     double *dmom_w_source_w,
							     double *mom_u_ham,
							     double *dmom_u_ham_grad_p,
							     double *mom_v_ham,
							     double *dmom_v_ham_grad_p,
							     double *mom_w_ham,
							     double *dmom_w_ham_grad_p)
{
  int k;
  double rho,nu,mu,H_rho,d_rho,H_mu,d_mu,norm_n,
    uc,Ftilde,Kinv;
  double nonlinearDragFactor = 1.0;
  if (killNonlinearDrag)
    nonlinearDragFactor = 0.0;
  for (k=0;k<nPoints;k++)
    {
      /** \todo optimize coefficients */
      /*H = smoothedHeaviside(eps,phi[k]);*/
      H_rho = smoothedHeaviside(eps_rho,phi[k]);
      d_rho = smoothedDirac(eps_rho,phi[k]);
      H_mu = smoothedHeaviside(eps_mu,phi[k]);
      d_mu = smoothedDirac(eps_mu,phi[k]);

      rho = rho_0*(1.0-H_rho)+rho_1*H_rho;
      nu  = nu_0*(1.0-H_mu)+nu_1*H_mu;
      mu  = rho_0*nu_0*(1.0-H_mu)+rho_1*nu_1*H_mu;

/*       //u momentum accumulation */
/*       mom_u_acc[k]=rho*porosity[k]*u[k]; */
/*       dmom_u_acc_u[k]=rho*porosity[k]; */
      
/*       //v momentum accumulation */
/*       mom_v_acc[k]=rho*porosity[k]*v[k]; */
/*       dmom_v_acc_v[k]=rho*porosity[k]; */

/*       //w momentum accumulation */
/*       mom_w_acc[k]=rho*porosity[k]*w[k]; */
/*       dmom_w_acc_w[k]=rho*porosity[k]; */


/*      //mass advective flux */
/*       mass_adv[k*3+0]=porosity[k]*u[k]; */
/*       mass_adv[k*3+1]=porosity[k]*v[k]; */
/*       mass_adv[k*3+2]=porosity[k]*w[k]; */
      
/*       dmass_adv_u[k*3+0]=porosity[k]; */
/*       dmass_adv_v[k*3+1]=porosity[k]; */
/*       dmass_adv_w[k*3+2]=porosity[k]; */

/*       //u momentum advective flux */
/*       mom_u_adv[k*3+0]=rho*porosity[k]*u[k]*u[k]; */
/*       mom_u_adv[k*3+1]=rho*porosity[k]*u[k]*v[k]; */
/*       mom_u_adv[k*3+2]=rho*porosity[k]*u[k]*w[k]; */

/*       dmom_u_adv_u[k*3+0]=2.0*rho*porosity[k]*u[k]; */
/*       dmom_u_adv_u[k*3+1]=rho*porosity[k]*v[k]; */
/*       dmom_u_adv_u[k*3+2]=rho*porosity[k]*w[k]; */

/*       dmom_u_adv_v[k*3+1]=rho*porosity[k]*u[k]; */
      
/*       dmom_u_adv_w[k*3+2]=rho*porosity[k]*u[k]; */

/*       //v momentum advective_flux */
/*       mom_v_adv[k*3+0]=rho*porosity[k]*v[k]*u[k]; */
/*       mom_v_adv[k*3+1]=rho*porosity[k]*v[k]*v[k]; */
/*       mom_v_adv[k*3+2]=rho*porosity[k]*v[k]*w[k]; */
            
/*       dmom_v_adv_u[k*3+0]=rho*porosity[k]*v[k]; */
      
/*       dmom_v_adv_w[k*3+2]=rho*porosity[k]*v[k]; */
      
/*       dmom_v_adv_v[k*3+0]=rho*porosity[k]*u[k]; */
/*       dmom_v_adv_v[k*3+1]=2.0*rho*porosity[k]*v[k]; */
/*       dmom_v_adv_v[k*3+2]=rho*porosity[k]*w[k]; */

/*       //w momentum advective_flux */
/*       mom_w_adv[k*3+0]=rho*porosity[k]*w[k]*u[k]; */
/*       mom_w_adv[k*3+1]=rho*porosity[k]*w[k]*v[k]; */
/*       mom_w_adv[k*3+2]=rho*porosity[k]*w[k]*w[k]; */
            
/*       dmom_w_adv_u[k*3+0]=rho*porosity[k]*w[k]; */
      
/*       dmom_w_adv_v[k*3+1]=rho*porosity[k]*w[k]; */
      
/*       dmom_w_adv_w[k*3+0]=rho*porosity[k]*u[k]; */
/*       dmom_w_adv_w[k*3+1]=rho*porosity[k]*v[k]; */
/*       dmom_w_adv_w[k*3+2]=2.0*rho*porosity[k]*w[k]; */

/*       //u momentum diffusion tensor */
/*       mom_u_diff_ten[k*9+0] = 2.0*porosity[k]*mu; */
/*       mom_u_diff_ten[k*9+4] = porosity[k]*mu; */
/*       mom_u_diff_ten[k*9+8] = porosity[k]*mu; */

/*       mom_uv_diff_ten[k*9+3]=porosity[k]*mu; */

/*       mom_uw_diff_ten[k*9+6]=porosity[k]*mu; */

/*       //v momentum diffusion tensor */
/*       mom_v_diff_ten[k*9+0] = porosity[k]*mu; */
/*       mom_v_diff_ten[k*9+4] = 2.0*porosity[k]*mu; */
/*       mom_v_diff_ten[k*9+8] = porosity[k]*mu; */

/*       mom_vu_diff_ten[k*9+1]=porosity[k]*mu; */

/*       mom_vw_diff_ten[k*9+7]=porosity[k]*mu; */

/*       //w momentum diffusion tensor */
/*       mom_w_diff_ten[k*9+0] = porosity[k]*mu; */
/*       mom_w_diff_ten[k*9+4] = porosity[k]*mu; */
/*       mom_w_diff_ten[k*9+8] = 2.0*porosity[k]*mu; */

/*       mom_wu_diff_ten[k*9+2]=porosity[k]*mu; */

/*       mom_wv_diff_ten[k*9+5]=porosity[k]*mu; */

/*       //momentum sources */
/*       norm_n = sqrt(n[k*3+0]*n[k*3+0]+n[k*3+1]*n[k*3+1]+n[k*3+2]*n[k*3+2]); */
/*       mom_u_source[k] = -rho*porosity[k]*g[0] - *porosity[k]*d_mu*sigma*kappa[k]*n[k*3+0]/(norm_n); */
/*       mom_v_source[k] = -rho*porosity[k]*g[1] - *porosity[k]*d_mu*sigma*kappa[k]*n[k*3+1]/(norm_n); */
/*       mom_w_source[k] = -rho*porosity[k]*g[2] - *porosity[k]*d_mu*sigma*kappa[k]*n[k*3+2]/(norm_n); */


/*       //u momentum Hamiltonian (pressure) */
/*       mom_u_ham[k] = grad_p[k*3+0]; */
/*       dmom_u_ham_grad_p[k*3+0]=1.0; */
      
/*       //v momentum Hamiltonian (pressure) */
/*       mom_v_ham[k] = grad_p[k*3+1]; */
/*       dmom_v_ham_grad_p[k*3+1]=1.0; */

/*       //w momentum Hamiltonian (pressure) */
/*       mom_w_ham[k] = grad_p[k*3+2]; */
/*       dmom_w_ham_grad_p[k*3+2]=1.0; */

      //cek "incompressible" form
      //u momentum accumulation
      mom_u_acc[k]=porosity[k]*u[k];
      dmom_u_acc_u[k]=porosity[k];
      
      //v momentum accumulation
      mom_v_acc[k]=porosity[k]*v[k];
      dmom_v_acc_v[k]=porosity[k];

      //w momentum accumulation
      mom_w_acc[k]=porosity[k]*w[k];
      dmom_w_acc_w[k]=porosity[k];


     //mass advective flux
      mass_adv[k*3+0]=porosity[k]*u[k];
      mass_adv[k*3+1]=porosity[k]*v[k];
      mass_adv[k*3+2]=porosity[k]*w[k];
      
      dmass_adv_u[k*3+0]=porosity[k];
      dmass_adv_v[k*3+1]=porosity[k];
      dmass_adv_w[k*3+2]=porosity[k];

      //u momentum advective flux
      mom_u_adv[k*3+0]=porosity[k]*u[k]*u[k];
      mom_u_adv[k*3+1]=porosity[k]*u[k]*v[k];
      mom_u_adv[k*3+2]=porosity[k]*u[k]*w[k];

      dmom_u_adv_u[k*3+0]=2.0*porosity[k]*u[k];
      dmom_u_adv_u[k*3+1]=porosity[k]*v[k];
      dmom_u_adv_u[k*3+2]=porosity[k]*w[k];

      dmom_u_adv_v[k*3+1]=porosity[k]*u[k];
      
      dmom_u_adv_w[k*3+2]=porosity[k]*u[k];

      //v momentum advective_flux
      mom_v_adv[k*3+0]=porosity[k]*v[k]*u[k];
      mom_v_adv[k*3+1]=porosity[k]*v[k]*v[k];
      mom_v_adv[k*3+2]=porosity[k]*v[k]*w[k];
            
      dmom_v_adv_u[k*3+0]=porosity[k]*v[k];
      
      dmom_v_adv_w[k*3+2]=porosity[k]*v[k];
      
      dmom_v_adv_v[k*3+0]=porosity[k]*u[k];
      dmom_v_adv_v[k*3+1]=2.0*porosity[k]*v[k];
      dmom_v_adv_v[k*3+2]=porosity[k]*w[k];

      //w momentum advective_flux
      mom_w_adv[k*3+0]=porosity[k]*w[k]*u[k];
      mom_w_adv[k*3+1]=porosity[k]*w[k]*v[k];
      mom_w_adv[k*3+2]=porosity[k]*w[k]*w[k];
            
      dmom_w_adv_u[k*3+0]=porosity[k]*w[k];
      
      dmom_w_adv_v[k*3+1]=porosity[k]*w[k];
      
      dmom_w_adv_w[k*3+0]=porosity[k]*u[k];
      dmom_w_adv_w[k*3+1]=porosity[k]*v[k];
      dmom_w_adv_w[k*3+2]=2.0*porosity[k]*w[k];

      //u momentum diffusion tensor
      mom_u_diff_ten[k*9+0] = 2.0*porosity[k]*nu;
      mom_u_diff_ten[k*9+4] = porosity[k]*nu;
      mom_u_diff_ten[k*9+8] = porosity[k]*nu;

      mom_uv_diff_ten[k*9+3]=porosity[k]*nu;

      mom_uw_diff_ten[k*9+6]=porosity[k]*nu;

      //v momentum diffusion tensor
      mom_v_diff_ten[k*9+0] = porosity[k]*nu;
      mom_v_diff_ten[k*9+4] = 2.0*porosity[k]*nu;
      mom_v_diff_ten[k*9+8] = porosity[k]*nu;

      mom_vu_diff_ten[k*9+1]=porosity[k]*nu;

      mom_vw_diff_ten[k*9+7]=porosity[k]*nu;

      //w momentum diffusion tensor
      mom_w_diff_ten[k*9+0] = porosity[k]*nu;
      mom_w_diff_ten[k*9+4] = porosity[k]*nu;
      mom_w_diff_ten[k*9+8] = 2.0*porosity[k]*nu;

      mom_wu_diff_ten[k*9+2]=porosity[k]*nu;

      mom_wv_diff_ten[k*9+5]=porosity[k]*nu;

      //momentum sources
      norm_n = sqrt(n[k*3+0]*n[k*3+0]+n[k*3+1]*n[k*3+1]+n[k*3+2]*n[k*3+2]);
      //porous medium contribution
      //end up with extra porosity term in final expression because multiply whole momentum 
      //equation through by porosity
      uc     = sqrt(u[k]*u[k]+v[k]*v[k]+w[k]*w[k]);
      if (fabs(1.0-porosity[k]) < 1.0e-7)
	Ftilde = 0.0;
      else
	Ftilde = porosity[k]*meanGrainSize[k]*1.0e-2/(1.0-porosity[k])/nu;
      /*mwf hack
	Ftilde =0.0;
      */
       //allow only linear resistance for sponge layers etc
      Ftilde *= nonlinearDragFactor;
      //trap divide by zero here 
      if (fabs(porosity[k]) < 1.0e-7)
	Kinv = 0.0;
      else
	Kinv   = 180.0*(1.0-porosity[k])*(1.0-porosity[k])/(meanGrainSize[k]*meanGrainSize[k]*porosity[k]*porosity[k]*porosity[k]);

      mom_u_source[k] = -porosity[k]*g[0] - porosity[k]*d_mu*sigma*kappa[k]*n[k*3+0]/(rho*(norm_n+1.0e-8))
	+ porosity[k]*porosity[k]*nu*Kinv*(1.0+Ftilde*uc)*u[k];
      mom_v_source[k] = -porosity[k]*g[1] - porosity[k]*d_mu*sigma*kappa[k]*n[k*3+1]/(rho*(norm_n+1.0e-8))
	+ porosity[k]*porosity[k]*nu*Kinv*(1.0+Ftilde*uc)*v[k];
      mom_w_source[k] = -porosity[k]*g[2] - porosity[k]*d_mu*sigma*kappa[k]*n[k*3+2]/(rho*(norm_n+1.0e-8))
	+ porosity[k]*porosity[k]*nu*Kinv*(1.0+Ftilde*uc)*w[k];

      dmom_u_source_u[k] = porosity[k]*porosity[k]*nu*Kinv*(1.0 + Ftilde*(uc + u[k]*u[k]/(uc+1.0e-12)));
      dmom_u_source_v[k] = porosity[k]*porosity[k]*nu*Kinv*(0.0 + Ftilde*(u[k]*v[k]/(uc+1.0e-12)));
      dmom_u_source_w[k] = porosity[k]*porosity[k]*nu*Kinv*(0.0 + Ftilde*(u[k]*w[k]/(uc+1.0e-12)));

      dmom_v_source_u[k] = porosity[k]*porosity[k]*nu*Kinv*(0.0 + Ftilde*(u[k]*v[k]/(uc+1.0e-12)));
      dmom_v_source_v[k] = porosity[k]*porosity[k]*nu*Kinv*(1.0 + Ftilde*(uc + v[k]*v[k]/(uc+1.0e-12)));
      dmom_v_source_w[k] = porosity[k]*porosity[k]*nu*Kinv*(0.0 + Ftilde*(w[k]*v[k]/(uc+1.0e-12)));

      dmom_w_source_u[k] = porosity[k]*porosity[k]*nu*Kinv*(0.0 + Ftilde*(w[k]*u[k]/(uc+1.0e-12)));
      dmom_w_source_v[k] = porosity[k]*porosity[k]*nu*Kinv*(0.0 + Ftilde*(w[k]*v[k]/(uc+1.0e-12)));
      dmom_w_source_w[k] = porosity[k]*porosity[k]*nu*Kinv*(1.0 + Ftilde*(uc + w[k]*w[k]/(uc+1.0e-12)));

      //u momentum Hamiltonian (pressure)
      mom_u_ham[k] = porosity[k]*grad_p[k*3+0]/rho;
      dmom_u_ham_grad_p[k*3+0]=porosity[k]/rho;
      
      //v momentum Hamiltonian (pressure)
      mom_v_ham[k] = porosity[k]*grad_p[k*3+1]/rho;
      dmom_v_ham_grad_p[k*3+1]=porosity[k]/rho;

      //w momentum Hamiltonian (pressure)
      mom_w_ham[k] = porosity[k]*grad_p[k*3+2]/rho;
      dmom_w_ham_grad_p[k*3+2]=porosity[k]/rho;
    }
}
void VolumeAveragedTwophaseNavierStokes_ST_LS_SO_3D_Evaluate_sd(const int nPoints,
							     const int killNonlinearDrag,
							     const double eps_rho,
							     const double eps_mu,
							     const double sigma,
							     const double rho_0,
							     const double nu_0,
							     const double rho_1,
							     const double nu_1,
							     const double* meanGrainSize,
							     const double* g,
							     const double* phi,
							     const double* n,
							     const double* kappa,
							     const double *p,
							     const double *grad_p,
							     const double *u,
							     const double *v,
							     const double *w,
							     const double *porosity,
							     double *mom_u_acc,
							     double *dmom_u_acc_u,
							     double *mom_v_acc,
							     double *dmom_v_acc_v,
							     double *mom_w_acc,
							     double *dmom_w_acc_w,
							     double *mass_adv,
							     double *dmass_adv_u,
							     double *dmass_adv_v,
							     double *dmass_adv_w,
							     double *mom_u_adv,
							     double *dmom_u_adv_u,
							     double *dmom_u_adv_v,
							     double *dmom_u_adv_w,
							     double *mom_v_adv,
							     double *dmom_v_adv_u,
							     double *dmom_v_adv_v,
							     double *dmom_v_adv_w,
							     double *mom_w_adv,
							     double *dmom_w_adv_u,
							     double *dmom_w_adv_v,
							     double *dmom_w_adv_w,
							     double *mom_u_diff_ten,
							     double *mom_v_diff_ten,
							     double *mom_w_diff_ten,
							     double *mom_uv_diff_ten,
							     double *mom_uw_diff_ten,
							     double *mom_vu_diff_ten,
							     double *mom_vw_diff_ten,
							     double *mom_wu_diff_ten,
							     double *mom_wv_diff_ten,
							     double *mom_u_source,
							     double *mom_v_source,
							     double *mom_w_source,
							     double *dmom_u_source_u,
							     double *dmom_u_source_v,
							     double *dmom_u_source_w,
							     double *dmom_v_source_u,
							     double *dmom_v_source_v,
							     double *dmom_v_source_w,
							     double *dmom_w_source_u,
							     double *dmom_w_source_v,
							     double *dmom_w_source_w,
							     double *mom_u_ham,
							     double *dmom_u_ham_grad_p,
							     double *mom_v_ham,
							     double *dmom_v_ham_grad_p,
							     double *mom_w_ham,
							     double *dmom_w_ham_grad_p)
{
  int k;
  double rho,nu,mu,H_rho,d_rho,H_mu,d_mu,norm_n,
    uc,Ftilde,Kinv;
  double nonlinearDragFactor = 1.0;
  if (killNonlinearDrag)
    nonlinearDragFactor = 0.0;  
  for (k=0;k<nPoints;k++)
    {
      /** \todo optimize coefficients */
      /*H = smoothedHeaviside(eps,phi[k]);*/
      H_rho = smoothedHeaviside(eps_rho,phi[k]);
      d_rho = smoothedDirac(eps_rho,phi[k]);
      H_mu = smoothedHeaviside(eps_mu,phi[k]);
      d_mu = smoothedDirac(eps_mu,phi[k]);

      rho = rho_0*(1.0-H_rho)+rho_1*H_rho;
      nu  = nu_0*(1.0-H_mu)+nu_1*H_mu;
      mu  = rho_0*nu_0*(1.0-H_mu)+rho_1*nu_1*H_mu;

      //u momentum accumulation
      mom_u_acc[k]=porosity[k]*u[k];
      dmom_u_acc_u[k]=porosity[k];
      
      //v momentum accumulation
      mom_v_acc[k]=porosity[k]*v[k];
      dmom_v_acc_v[k]=porosity[k];

      //w momentum accumulation
      mom_w_acc[k]=porosity[k]*w[k];
      dmom_w_acc_w[k]=porosity[k];


     //mass advective flux
      mass_adv[k*3+0]=porosity[k]*u[k];
      mass_adv[k*3+1]=porosity[k]*v[k];
      mass_adv[k*3+2]=porosity[k]*w[k];
      
      dmass_adv_u[k*3+0]=porosity[k];
      dmass_adv_v[k*3+1]=porosity[k];
      dmass_adv_w[k*3+2]=porosity[k];

      //u momentum advective flux
      mom_u_adv[k*3+0]=porosity[k]*u[k]*u[k];
      mom_u_adv[k*3+1]=porosity[k]*u[k]*v[k];
      mom_u_adv[k*3+2]=porosity[k]*u[k]*w[k];

      dmom_u_adv_u[k*3+0]=2.0*porosity[k]*u[k];
      dmom_u_adv_u[k*3+1]=porosity[k]*v[k];
      dmom_u_adv_u[k*3+2]=porosity[k]*w[k];

      dmom_u_adv_v[k*3+1]=porosity[k]*u[k];
      
      dmom_u_adv_w[k*3+2]=porosity[k]*u[k];

      //v momentum advective_flux
      mom_v_adv[k*3+0]=porosity[k]*v[k]*u[k];
      mom_v_adv[k*3+1]=porosity[k]*v[k]*v[k];
      mom_v_adv[k*3+2]=porosity[k]*v[k]*w[k];
            
      dmom_v_adv_u[k*3+0]=porosity[k]*v[k];
      
      dmom_v_adv_w[k*3+2]=porosity[k]*v[k];
      
      dmom_v_adv_v[k*3+0]=porosity[k]*u[k];
      dmom_v_adv_v[k*3+1]=2.0*porosity[k]*v[k];
      dmom_v_adv_v[k*3+2]=porosity[k]*w[k];

      //w momentum advective_flux
      mom_w_adv[k*3+0]=porosity[k]*w[k]*u[k];
      mom_w_adv[k*3+1]=porosity[k]*w[k]*v[k];
      mom_w_adv[k*3+2]=porosity[k]*w[k]*w[k];
            
      dmom_w_adv_u[k*3+0]=porosity[k]*w[k];
      
      dmom_w_adv_v[k*3+1]=porosity[k]*w[k];
      
      dmom_w_adv_w[k*3+0]=porosity[k]*u[k];
      dmom_w_adv_w[k*3+1]=porosity[k]*v[k];
      dmom_w_adv_w[k*3+2]=2.0*porosity[k]*w[k];

      //u momentum diffusion tensor
      mom_u_diff_ten[k*3+0] = 2.0*porosity[k]*nu;
      mom_u_diff_ten[k*3+1] = porosity[k]*nu;
      mom_u_diff_ten[k*3+2] = porosity[k]*nu;

      mom_uv_diff_ten[k]=porosity[k]*nu;

      mom_uw_diff_ten[k]=porosity[k]*nu;

      //v momentum diffusion tensor
      mom_v_diff_ten[k*3+0] = porosity[k]*nu;
      mom_v_diff_ten[k*3+1] = 2.0*porosity[k]*nu;
      mom_v_diff_ten[k*3+2] = porosity[k]*nu;

      mom_vu_diff_ten[k]=porosity[k]*nu;

      mom_vw_diff_ten[k]=porosity[k]*nu;

      //w momentum diffusion tensor
      mom_w_diff_ten[k*3+0] = porosity[k]*nu;
      mom_w_diff_ten[k*3+1] = porosity[k]*nu;
      mom_w_diff_ten[k*3+2] = 2.0*porosity[k]*nu;

      mom_wu_diff_ten[k]=porosity[k]*nu;

      mom_wv_diff_ten[k]=porosity[k]*nu;

      //momentum sources
      norm_n = sqrt(n[k*3+0]*n[k*3+0]+n[k*3+1]*n[k*3+1]+n[k*3+2]*n[k*3+2]);
      //end up with extra porosity term in final expression because multiply whole momentum 
      //equation through by porosity
      uc     = sqrt(u[k]*u[k]+v[k]*v[k]+w[k]*w[k]);
      if (fabs(1.0-porosity[k]) < 1.0e-7)
	Ftilde = 0.0;
      else
	Ftilde = porosity[k]*meanGrainSize[k]*1.0e-2/(1.0-porosity[k])/nu;
      /*mwf hack
	Ftilde =0.0;
      */
       //allow only linear resistance for sponge layers etc
      Ftilde *= nonlinearDragFactor;
      //trap divide by zero here 
      if (fabs(porosity[k]) < 1.0e-7)
	Kinv = 0.0;
      else
	Kinv   = 180.0*(1.0-porosity[k])*(1.0-porosity[k])/(meanGrainSize[k]*meanGrainSize[k]*porosity[k]*porosity[k]*porosity[k]);

      mom_u_source[k] = -porosity[k]*g[0] - porosity[k]*d_mu*sigma*kappa[k]*n[k*3+0]/(rho*(norm_n+1.0e-8))
	+ porosity[k]*porosity[k]*nu*Kinv*(1.0+Ftilde*uc)*u[k];
      mom_v_source[k] = -porosity[k]*g[1] - porosity[k]*d_mu*sigma*kappa[k]*n[k*3+1]/(rho*(norm_n+1.0e-8))
	+ porosity[k]*porosity[k]*nu*Kinv*(1.0+Ftilde*uc)*v[k];
      mom_w_source[k] = -porosity[k]*g[2] - porosity[k]*d_mu*sigma*kappa[k]*n[k*3+2]/(rho*(norm_n+1.0e-8))
	+ porosity[k]*porosity[k]*nu*Kinv*(1.0+Ftilde*uc)*w[k];

      dmom_u_source_u[k] = porosity[k]*porosity[k]*nu*Kinv*(1.0 + Ftilde*(uc + u[k]*u[k]/(uc+1.0e-12)));
      dmom_u_source_v[k] = porosity[k]*porosity[k]*nu*Kinv*(0.0 + Ftilde*(u[k]*v[k]/(uc+1.0e-12)));
      dmom_u_source_w[k] = porosity[k]*porosity[k]*nu*Kinv*(0.0 + Ftilde*(u[k]*w[k]/(uc+1.0e-12)));

      dmom_v_source_u[k] = porosity[k]*porosity[k]*nu*Kinv*(0.0 + Ftilde*(u[k]*v[k]/(uc+1.0e-12)));
      dmom_v_source_v[k] = porosity[k]*porosity[k]*nu*Kinv*(1.0 + Ftilde*(uc + v[k]*v[k]/(uc+1.0e-12)));
      dmom_v_source_w[k] = porosity[k]*porosity[k]*nu*Kinv*(0.0 + Ftilde*(w[k]*v[k]/(uc+1.0e-12)));

      dmom_w_source_u[k] = porosity[k]*porosity[k]*nu*Kinv*(0.0 + Ftilde*(w[k]*u[k]/(uc+1.0e-12)));
      dmom_w_source_v[k] = porosity[k]*porosity[k]*nu*Kinv*(0.0 + Ftilde*(w[k]*v[k]/(uc+1.0e-12)));
      dmom_w_source_w[k] = porosity[k]*porosity[k]*nu*Kinv*(1.0 + Ftilde*(uc + w[k]*w[k]/(uc+1.0e-12)));


      //u momentum Hamiltonian (pressure)
      mom_u_ham[k] = porosity[k]*grad_p[k*3+0]/rho;
      dmom_u_ham_grad_p[k*3+0]=porosity[k]/rho;
      
      //v momentum Hamiltonian (pressure)
      mom_v_ham[k] = porosity[k]*grad_p[k*3+1]/rho;
      dmom_v_ham_grad_p[k*3+1]=porosity[k]/rho;

      //w momentum Hamiltonian (pressure)
      mom_w_ham[k] = porosity[k]*grad_p[k*3+2]/rho;
      dmom_w_ham_grad_p[k*3+2]=porosity[k]/rho;
    }
}
void VolumeAveragedVOFCoefficientsEvaluate(int nPoints,
					   int nSpace,
					   double eps,
					   double* v,
					   double* phi,
					   double* porosity,
					   double* u,
					   double* m,
					   double* dm,
					   double* f,
					   double* df)
{
/*   printf("eps in vof %12.5e\n",eps); */
  int i,I;
  for (i=0;i<nPoints;i++)
    {
      m[i]=u[i]*porosity[i];
      dm[i]=porosity[i];
      for (I=0;I<nSpace;I++)
        {
/*           f[i*nSpace+I] = porosity[i]*v[i*nSpace+I]*smoothedHeaviside(eps,phi[i]); */
/*           df[i*nSpace+I] = 0.0; */
          f[i*nSpace+I] = porosity[i]*v[i*nSpace+I]*u[i];
          df[i*nSpace+I] = porosity[i]*v[i*nSpace+I];
        }
    }
}

/***********************************************************************
  Basic k-epsilon model for incompressible flow from Hutter etal Chaper 11

\bar{\vec v} = <\vec v> Reynolds-averaged (mean) velocity
\vec v^{'}   = turbulent fluctuation 
assume \vec v = <\vec v> + \vec v^{'}, with <\vec v^{'}> = 0

Reynolds averaged NS equations

\deld \bar{\vec v} = 0

\pd{\bar{\vec v}}{t} + \deld \left(\bar{\vec v} \outer \bar{\vec v}\right) 
               -\nu \deld \ten \bar{D} + \frac{1}{\rho}\grad \bar p  
               - \frac{1}{rho}\deld \ten{R} = 0

Reynolds stress term

\ten R = -\rho <\vec v^{'}\outer \vec v^{'}>
\frac{1}{\rho}\ten{R} = 2 \nu_t \bar{D} - \frac{2}{3}k\ten{I}

D_{ij}(\vec v) = \frac{1}{2} \left( \pd{v_i}{x_j} + \pd{v_j}{x_i})
\ten D \bar{\ten D} = D(<\vec v>), \ten D^{'} = \ten D(\vec v^{'})



k-epsilon tranport equations

\pd{k}{t} + \deld (k\bar{\vec v}) 
          - \deld\left[\left(\frac{\nu_t}{\sigma_k} + \nu\right)\grad k \right]
          - 4\nu_t \Pi_{D} + \epsilon = 0

\pd{\varepsilon}{t} + \deld (\varepsilon \bar{\vec v}) 
          - \deld\left[\left(\frac{\nu_t}{\sigma_\varepsilon} + \nu\right)\grad \varepsilon \right]
          - 4c_1 k \Pi_{D} + c_2 \frac{\epsilon^2}{k} = 0


k              -- turbulent kinetic energy = <\vec v^{'}\dot \vec v^{'}>
\varepsilon    -- turbulent dissipation rate = 4 \nu <\Pi_{D^{'}}>

\nu            -- kinematic viscosity (\mu/\rho)
\nu_t          -- turbulent viscosity = c_mu \frac{k^2}{\varepsilon}


\Pi_{\ten A} = \frac{1}{2}tr(\ten A^2) = 1/2 \ten A\cdot \ten A
\ten D \cdot \ten D = \frac{1}{4}\left[ (4 u_x^2 + 4 v_y^2 + 
                                        1/2 (u_y + v_x)^2 \right]
   
4 \Pi_{D} = 2 \frac{1}{4}\left[ (4 u_x^2 + 4 v_y^2 + 
                                1/2 (u_y + v_x)^2 \right]
          = \left[ (2 u_x^2 + 2 v_y^2 + (u_y + v_x)^2 \right]

\sigma_k -- Prandtl number \approx 1
\sigma_e -- c_{\mu}/c_e

c_{\mu} = 0.09, c_1 = 0.126, c_2 = 1.92, c_{\varepsilon} = 0.07

 ***********************************************************************/
void kEpsilon_2D_Evaluate(int nPoints,
			  int nSpace,
			  double sigma_k,
			  double sigma_e,
			  double c_1,
			  double c_2,
			  double c_mu,
			  double c_e,
			  double nu,
			  double *velocity,
			  double *gradu,
			  double *gradv,
			  double *k,
			  double *epsilon,
			  double *m_k,
			  double *dm_k,
			  double *m_e,
			  double *dm_e,
			  double *phi_k,
			  double *dphi_k,
			  double *phi_e,
			  double *dphi_e,
			  double *f_k,
			  double *df_k,
			  double *f_e,
			  double *df_e,
			  double *a_k,
			  double *da_k_dk,
			  double *da_k_de,
			  double *a_e,
			  double *da_e_dk,
			  double *da_e_de,
			  double *r_k,
			  double *dr_k_dk,
			  double *dr_k_de,
			  double *r_e,
			  double *dr_e_dk,
			  double *dr_e_de)
		       
{
  int i,I;
  double nu_t,dnu_t_dk,dnu_t_de,PiD4,eval,kval,disp,ddisp_dk,ddisp_de;
  const double div_eps = 1.0e-6;
  for (i = 0; i < nPoints; i++)
    {
      eval = epsilon[i] >= 0.0 ? epsilon[i] : 0.0;
      kval = k[i] >= 0.0 ? k[i] : 0.0;
      /*eddy viscosity*/
      nu_t     = c_mu*k[i]*k[i]/(epsilon[i]+div_eps);
      dnu_t_dk = 2.0*c_mu*k[i]/(epsilon[i]+div_eps);
      dnu_t_de =-c_mu*k[i]*k[i]/(epsilon[i]*epsilon[i]+div_eps);
      if (nu_t < 0.0)
	{
	  nu_t = 0.0; dnu_t_dk = 0.0; dnu_t_de = 0.0;
	}
      /*mwf debug
      printf("keps eval k[%d]=%g e[%d]=%g nu_t=%g dnu_t_dk=%g dnu_t_de=%g\n",i,k[i],i,epsilon[i],
	     nu_t,dnu_t_dk,dnu_t_de);
      */
      /*linear mass terms for k-e*/
      m_k[i] = k[i];
      dm_k[i]= 1.0;

      m_e[i] = epsilon[i];
      dm_e[i]= 1.0;
      
      /*linear advection*/
      for (I=0; I < nSpace; I++)
	{
	  f_k[i*nSpace+I] = k[i]*velocity[i*nSpace+I];
	  df_k[i*nSpace+I]= velocity[i*nSpace+I];

	  f_e[i*nSpace+I] = epsilon[i]*velocity[i*nSpace+I];
	  df_e[i*nSpace+I]= velocity[i*nSpace+I];
	}
      /*linear potentials*/ 
      phi_k[i] = k[i];
      dphi_k[i]= 1.0; 
      phi_e[i] = epsilon[i];
      dphi_e[i]= 1.0;

      /*nonlinear diffusion*/
      for (I=0; I < nSpace; I++)
	{
	  a_k[i*nSpace*nSpace + I*nSpace + I] = nu_t/sigma_k + nu;
	  da_k_dk[i*nSpace*nSpace + I*nSpace + I] = dnu_t_dk/sigma_k;
	  da_k_de[i*nSpace*nSpace + I*nSpace + I] = dnu_t_de/sigma_k;

	  a_e[i*nSpace*nSpace + I*nSpace + I] = nu_t/sigma_e + nu;
	  da_e_dk[i*nSpace*nSpace + I*nSpace + I] = dnu_t_dk/sigma_e;
	  da_e_de[i*nSpace*nSpace + I*nSpace + I] = dnu_t_de/sigma_e;
	}
      
      /*production term*/
      /*4*Pi_D*/
      PiD4 = 2.0*(gradu[i*nSpace+0]*gradu[i*nSpace+0] + gradv[i*nSpace+1]*gradv[i*nSpace+1]) +
	(gradu[i*nSpace+1] + gradv[i*nSpace+0])*(gradu[i*nSpace+1] + gradv[i*nSpace+0]);

      r_k[i]     = -nu_t*PiD4 + epsilon[i];
      dr_k_dk[i] = -dnu_t_dk*PiD4;
      dr_k_de[i] = -dnu_t_de*PiD4 + 1.0;
 
      disp = c_2*epsilon[i]*epsilon[i]/(k[i]+div_eps);
      ddisp_dk = -c_2*epsilon[i]*epsilon[i]/(k[i]*k[i]+div_eps);   
      ddisp_de = 2.0*c_2*epsilon[i]/(k[i]+div_eps); 
      if (disp < 0.0)
	{
	  disp = 0.0; ddisp_dk = 0.0; ddisp_de = 0.0;
	}
      r_e[i]      = -c_1*k[i]*PiD4  + disp;/*c_2*epsilon[i]*epsilon[i]/(k[i]+div_eps);*/
      dr_e_dk[i]  = -c_1*PiD4  + ddisp_dk;/* -c_2*epsilon[i]*epsilon[i]/(k[i]*k[i]+div_eps); */   
      dr_e_de[i]  = ddisp_de; /*2.0*c_2*epsilon[i]/(k[i]+div_eps); */
 
    }
}
void kEpsilon_2D_Evaluate_sd(int nPoints,
			     int nSpace,
			     double sigma_k,
			     double sigma_e,
			     double c_1,
			     double c_2,
			     double c_mu,
			     double c_e,
			     double nu,
			     double *velocity,
			     double *gradu,
			     double *gradv,
			     double *k,
			     double *epsilon,
			     double *m_k,
			     double *dm_k,
			     double *m_e,
			     double *dm_e,
			     double *phi_k,
			     double *dphi_k,
			     double *phi_e,
			     double *dphi_e,
			     double *f_k,
			     double *df_k,
			     double *f_e,
			     double *df_e,
			     double *a_k,
			     double *da_k_dk,
			     double *da_k_de,
			     double *a_e,
			     double *da_e_dk,
			     double *da_e_de,
			     double *r_k,
			     double *dr_k_dk,
			     double *dr_k_de,
			     double *r_e,
			     double *dr_e_dk,
			     double *dr_e_de)
		       
{
  int i,I;
  double nu_t,dnu_t_dk,dnu_t_de,PiD4,eval,kval,disp,ddisp_dk,ddisp_de;
  const double div_eps = 1.0e-6;
  for (i = 0; i < nPoints; i++)
    {
      eval = epsilon[i] >= 0.0 ? epsilon[i] : 0.0;
      kval = k[i] >= 0.0 ? k[i] : 0.0;
      /*eddy viscosity*/
      nu_t     = c_mu*k[i]*k[i]/(epsilon[i]+div_eps);
      dnu_t_dk = 2.0*c_mu*k[i]/(epsilon[i]+div_eps);
      dnu_t_de =-c_mu*k[i]*k[i]/(epsilon[i]*epsilon[i]+div_eps);
      if (nu_t < 0.0)
	{
	  nu_t = 0.0; dnu_t_dk = 0.0; dnu_t_de = 0.0;
	}
      /*mwf debug
      printf("keps eval k[%d]=%g e[%d]=%g nu_t=%g dnu_t_dk=%g dnu_t_de=%g\n",i,k[i],i,epsilon[i],
	     nu_t,dnu_t_dk,dnu_t_de);
      */
      /*linear mass terms for k-e*/
      m_k[i] = k[i];
      dm_k[i]= 1.0;

      m_e[i] = epsilon[i];
      dm_e[i]= 1.0;
      
      /*linear advection*/
      for (I=0; I < nSpace; I++)
	{
	  f_k[i*nSpace+I] = k[i]*velocity[i*nSpace+I];
	  df_k[i*nSpace+I]= velocity[i*nSpace+I];

	  f_e[i*nSpace+I] = epsilon[i]*velocity[i*nSpace+I];
	  df_e[i*nSpace+I]= velocity[i*nSpace+I];
	}
      /*linear potentials*/ 
      phi_k[i] = k[i];
      dphi_k[i]= 1.0; 
      phi_e[i] = epsilon[i];
      dphi_e[i]= 1.0;

      /*nonlinear diffusion*/
      for (I=0; I < nSpace; I++)
	{
	  a_k[i*2+I] = nu_t/sigma_k + nu;
	  da_k_dk[i*2 + I] = dnu_t_dk/sigma_k;
	  da_k_de[i*2 + I] = dnu_t_de/sigma_k;

	  a_e[i*2 + I] = nu_t/sigma_e + nu;
	  da_e_dk[i*2 + I] = dnu_t_dk/sigma_e;
	  da_e_de[i*2 + I] = dnu_t_de/sigma_e;
	}
      
      /*production term*/
      /*4*Pi_D*/
      PiD4 = 2.0*(gradu[i*nSpace+0]*gradu[i*nSpace+0] + gradv[i*nSpace+1]*gradv[i*nSpace+1]) +
	(gradu[i*nSpace+1] + gradv[i*nSpace+0])*(gradu[i*nSpace+1] + gradv[i*nSpace+0]);

      r_k[i]     = -nu_t*PiD4 + epsilon[i];
      dr_k_dk[i] = -dnu_t_dk*PiD4;
      dr_k_de[i] = -dnu_t_de*PiD4 + 1.0;
 
      disp = c_2*epsilon[i]*epsilon[i]/(k[i]+div_eps);
      ddisp_dk = -c_2*epsilon[i]*epsilon[i]/(k[i]*k[i]+div_eps);   
      ddisp_de = 2.0*c_2*epsilon[i]/(k[i]+div_eps); 
      if (disp < 0.0)
	{
	  disp = 0.0; ddisp_dk = 0.0; ddisp_de = 0.0;
	}
      r_e[i]      = -c_1*k[i]*PiD4  + disp;/*c_2*epsilon[i]*epsilon[i]/(k[i]+div_eps);*/
      dr_e_dk[i]  = -c_1*PiD4  + ddisp_dk;/* -c_2*epsilon[i]*epsilon[i]/(k[i]*k[i]+div_eps); */   
      dr_e_de[i]  = ddisp_de; /*2.0*c_2*epsilon[i]/(k[i]+div_eps); */
 
    }
}
void kEpsilon_3D_Evaluate_sd(int nPoints,
			     int nSpace,
			     double sigma_k,
			     double sigma_e,
			     double c_1,
			     double c_2,
			     double c_mu,
			     double c_e,
			     double nu,
			     double *velocity,
			     double *gradu,
			     double *gradv,
			     double *gradw,
			     double *k,
			     double *epsilon,
			     double *m_k,
			     double *dm_k,
			     double *m_e,
			     double *dm_e,
			     double *phi_k,
			     double *dphi_k,
			     double *phi_e,
			     double *dphi_e,
			     double *f_k,
			     double *df_k,
			     double *f_e,
			     double *df_e,
			     double *a_k,
			     double *da_k_dk,
			     double *da_k_de,
			     double *a_e,
			     double *da_e_dk,
			     double *da_e_de,
			     double *r_k,
			     double *dr_k_dk,
			     double *dr_k_de,
			     double *r_e,
			     double *dr_e_dk,
			     double *dr_e_de)
		       
{
  int i,I;
  double nu_t,dnu_t_dk,dnu_t_de,PiD4,eval,kval,disp,ddisp_dk,ddisp_de;
  const double div_eps = 1.0e-6;
  for (i = 0; i < nPoints; i++)
    {
      eval = epsilon[i] >= 0.0 ? epsilon[i] : 0.0;
      kval = k[i] >= 0.0 ? k[i] : 0.0;
      /*eddy viscosity*/
      nu_t     = c_mu*k[i]*k[i]/(epsilon[i]+div_eps);
      dnu_t_dk = 2.0*c_mu*k[i]/(epsilon[i]+div_eps);
      dnu_t_de =-c_mu*k[i]*k[i]/(epsilon[i]*epsilon[i]+div_eps);
      if (nu_t < 0.0)
	{
	  nu_t = 0.0; dnu_t_dk = 0.0; dnu_t_de = 0.0;
	}
      /*mwf debug
      printf("keps eval k[%d]=%g e[%d]=%g nu_t=%g dnu_t_dk=%g dnu_t_de=%g\n",i,k[i],i,epsilon[i],
	     nu_t,dnu_t_dk,dnu_t_de);
      */
      /*linear mass terms for k-e*/
      m_k[i] = k[i];
      dm_k[i]= 1.0;

      m_e[i] = epsilon[i];
      dm_e[i]= 1.0;
      
      /*linear advection*/
      for (I=0; I < nSpace; I++)
	{
	  f_k[i*nSpace+I] = k[i]*velocity[i*nSpace+I];
	  df_k[i*nSpace+I]= velocity[i*nSpace+I];

	  f_e[i*nSpace+I] = epsilon[i]*velocity[i*nSpace+I];
	  df_e[i*nSpace+I]= velocity[i*nSpace+I];
	}
      /*linear potentials*/ 
      phi_k[i] = k[i];
      dphi_k[i]= 1.0; 
      phi_e[i] = epsilon[i];
      dphi_e[i]= 1.0;

      /*nonlinear diffusion*/
      for (I=0; I < nSpace; I++)
	{
	  a_k[i*nSpace+I] = nu_t/sigma_k + nu;
	  da_k_dk[i*nSpace + I] = dnu_t_dk/sigma_k;
	  da_k_de[i*nSpace + I] = dnu_t_de/sigma_k;

	  a_e[i*nSpace + I] = nu_t/sigma_e + nu;
	  da_e_dk[i*nSpace + I] = dnu_t_dk/sigma_e;
	  da_e_de[i*nSpace + I] = dnu_t_de/sigma_e;
	}
      
      /*production term*/
      /*4*Pi_D*/
      PiD4 = 2.0*(gradu[i*nSpace+0]*gradu[i*nSpace+0] + 
		  gradv[i*nSpace+1]*gradv[i*nSpace+1] + 
		  gradw[i*nSpace+2]*gradw[i*nSpace+2]) 
	+
	(gradu[i*nSpace+1] + gradv[i*nSpace+0])*(gradu[i*nSpace+1] + gradv[i*nSpace+0])
	+
	(gradu[i*nSpace+2] + gradw[i*nSpace+0])*(gradu[i*nSpace+2] + gradw[i*nSpace+0])
	+
	(gradv[i*nSpace+2] + gradw[i*nSpace+1])*(gradv[i*nSpace+2] + gradw[i*nSpace+1]);


      r_k[i]     = -nu_t*PiD4 + epsilon[i];
      dr_k_dk[i] = -dnu_t_dk*PiD4;
      dr_k_de[i] = -dnu_t_de*PiD4 + 1.0;
 
      disp = c_2*epsilon[i]*epsilon[i]/(k[i]+div_eps);
      ddisp_dk = -c_2*epsilon[i]*epsilon[i]/(k[i]*k[i]+div_eps);   
      ddisp_de = 2.0*c_2*epsilon[i]/(k[i]+div_eps); 
      if (disp < 0.0)
	{
	  disp = 0.0; ddisp_dk = 0.0; ddisp_de = 0.0;
	}
      r_e[i]      = -c_1*k[i]*PiD4  + disp;/*c_2*epsilon[i]*epsilon[i]/(k[i]+div_eps);*/
      dr_e_dk[i]  = -c_1*PiD4  + ddisp_dk;/* -c_2*epsilon[i]*epsilon[i]/(k[i]*k[i]+div_eps); */   
      dr_e_de[i]  = ddisp_de; /*2.0*c_2*epsilon[i]/(k[i]+div_eps); */
 
    }
}
void kEpsilon_3D_Evaluate(int nPoints,
			     int nSpace,
			     double sigma_k,
			     double sigma_e,
			     double c_1,
			     double c_2,
			     double c_mu,
			     double c_e,
			     double nu,
			     double *velocity,
			     double *gradu,
			     double *gradv,
			     double *gradw,
			     double *k,
			     double *epsilon,
			     double *m_k,
			     double *dm_k,
			     double *m_e,
			     double *dm_e,
			     double *phi_k,
			     double *dphi_k,
			     double *phi_e,
			     double *dphi_e,
			     double *f_k,
			     double *df_k,
			     double *f_e,
			     double *df_e,
			     double *a_k,
			     double *da_k_dk,
			     double *da_k_de,
			     double *a_e,
			     double *da_e_dk,
			     double *da_e_de,
			     double *r_k,
			     double *dr_k_dk,
			     double *dr_k_de,
			     double *r_e,
			     double *dr_e_dk,
			     double *dr_e_de)
		       
{
  int i,I;
  double nu_t,dnu_t_dk,dnu_t_de,PiD4,eval,kval,disp,ddisp_dk,ddisp_de;
  const double div_eps = 1.0e-6;
  for (i = 0; i < nPoints; i++)
    {
      eval = epsilon[i] >= 0.0 ? epsilon[i] : 0.0;
      kval = k[i] >= 0.0 ? k[i] : 0.0;
      /*eddy viscosity*/
      nu_t     = c_mu*k[i]*k[i]/(epsilon[i]+div_eps);
      dnu_t_dk = 2.0*c_mu*k[i]/(epsilon[i]+div_eps);
      dnu_t_de =-c_mu*k[i]*k[i]/(epsilon[i]*epsilon[i]+div_eps);
      if (nu_t < 0.0)
	{
	  nu_t = 0.0; dnu_t_dk = 0.0; dnu_t_de = 0.0;
	}
      /*mwf debug
      printf("keps eval k[%d]=%g e[%d]=%g nu_t=%g dnu_t_dk=%g dnu_t_de=%g\n",i,k[i],i,epsilon[i],
	     nu_t,dnu_t_dk,dnu_t_de);
      */
      /*linear mass terms for k-e*/
      m_k[i] = k[i];
      dm_k[i]= 1.0;

      m_e[i] = epsilon[i];
      dm_e[i]= 1.0;
      
      /*linear advection*/
      for (I=0; I < nSpace; I++)
	{
	  f_k[i*nSpace+I] = k[i]*velocity[i*nSpace+I];
	  df_k[i*nSpace+I]= velocity[i*nSpace+I];

	  f_e[i*nSpace+I] = epsilon[i]*velocity[i*nSpace+I];
	  df_e[i*nSpace+I]= velocity[i*nSpace+I];
	}
      /*linear potentials*/ 
      phi_k[i] = k[i];
      dphi_k[i]= 1.0; 
      phi_e[i] = epsilon[i];
      dphi_e[i]= 1.0;

      /*nonlinear diffusion*/
      for (I=0; I < nSpace; I++)
	{
	  a_k[i*nSpace*nSpace + I*nSpace + I] = nu_t/sigma_k + nu;
	  da_k_dk[i*nSpace*nSpace + I*nSpace + I] = dnu_t_dk/sigma_k;
	  da_k_de[i*nSpace*nSpace + I*nSpace + I] = dnu_t_de/sigma_k;

	  a_e[i*nSpace*nSpace + I*nSpace + I] = nu_t/sigma_e + nu;
	  da_e_dk[i*nSpace*nSpace + I*nSpace + I] = dnu_t_dk/sigma_e;
	  da_e_de[i*nSpace*nSpace + I*nSpace + I] = dnu_t_de/sigma_e;
	}
      
      /*production term*/
      /*4*Pi_D*/
      PiD4 = 2.0*(gradu[i*nSpace+0]*gradu[i*nSpace+0] + 
		  gradv[i*nSpace+1]*gradv[i*nSpace+1] + 
		  gradw[i*nSpace+2]*gradw[i*nSpace+2]) 
	+
	(gradu[i*nSpace+1] + gradv[i*nSpace+0])*(gradu[i*nSpace+1] + gradv[i*nSpace+0])
	+
	(gradu[i*nSpace+2] + gradw[i*nSpace+0])*(gradu[i*nSpace+2] + gradw[i*nSpace+0])
	+
	(gradv[i*nSpace+2] + gradw[i*nSpace+1])*(gradv[i*nSpace+2] + gradw[i*nSpace+1]);


      r_k[i]     = -nu_t*PiD4 + epsilon[i];
      dr_k_dk[i] = -dnu_t_dk*PiD4;
      dr_k_de[i] = -dnu_t_de*PiD4 + 1.0;
 
      disp = c_2*epsilon[i]*epsilon[i]/(k[i]+div_eps);
      ddisp_dk = -c_2*epsilon[i]*epsilon[i]/(k[i]*k[i]+div_eps);   
      ddisp_de = 2.0*c_2*epsilon[i]/(k[i]+div_eps); 
      if (disp < 0.0)
	{
	  disp = 0.0; ddisp_dk = 0.0; ddisp_de = 0.0;
	}
      r_e[i]      = -c_1*k[i]*PiD4  + disp;/*c_2*epsilon[i]*epsilon[i]/(k[i]+div_eps);*/
      dr_e_dk[i]  = -c_1*PiD4  + ddisp_dk;/* -c_2*epsilon[i]*epsilon[i]/(k[i]*k[i]+div_eps); */   
      dr_e_de[i]  = ddisp_de; /*2.0*c_2*epsilon[i]/(k[i]+div_eps); */
 
    }
}


/*version of kEpsilon where k and epsilon are solved independently */
void kEpsilon_k_2D_Evaluate_sd(int nPoints,
			       int nSpace,
			       double sigma_k,
			       double c_mu,
			       double nu,
			       double *velocity,
			       double *gradu,
			       double *gradv,
			       double *k,
			       double *epsilon,
			       double *m_k,
			       double *dm_k,
			       double *phi_k,
			       double *dphi_k,
			       double *f_k,
			       double *df_k,
			       double *a_k,
			       double *da_k_dk,
			       double *r_k,
			       double *dr_k_dk)
		       
{
  int i,I;
  double nu_t,dnu_t_dk,PiD4,eval,kval,disp,ddisp_dk,ddisp_de;
  const double div_eps = 1.0e-3;
  for (i = 0; i < nPoints; i++)
    {
      eval = epsilon[i] >= 0.0 ? epsilon[i] : 0.0;
      kval = k[i] >= 0.0 ? k[i] : 0.0;
      /*eddy viscosity*/
      nu_t     = c_mu*k[i]*k[i]/(epsilon[i]+div_eps);
      dnu_t_dk = 2.0*c_mu*k[i]/(epsilon[i]+div_eps);
      if (nu_t < 0.0)
	{
	  nu_t = 0.0; dnu_t_dk = 0.0;
	}
      /*mwf debug
      printf("keps eval k[%d]=%g e[%d]=%g nu_t=%g dnu_t_dk=%g\n",i,k[i],i,epsilon[i],
	     nu_t,dnu_t_dk);
      */
      /*linear mass terms for k-e*/
      m_k[i] = k[i];
      dm_k[i]= 1.0;
      /*linear advection*/
      for (I=0; I < nSpace; I++)
	{
	  f_k[i*nSpace+I] = k[i]*velocity[i*nSpace+I];
	  df_k[i*nSpace+I]= velocity[i*nSpace+I];
	}
      /*linear potentials*/ 
      phi_k[i] = k[i];
      dphi_k[i]= 1.0; 
      /*nonlinear diffusion*/
      for (I=0; I < nSpace; I++)
	{
	  a_k[i*2+I] = nu_t/sigma_k + nu;
	  da_k_dk[i*2 + I] = dnu_t_dk/sigma_k;
	
	}
      
      /*production term*/
      /*4*Pi_D*/
      PiD4 = 2.0*(gradu[i*nSpace+0]*gradu[i*nSpace+0] + gradv[i*nSpace+1]*gradv[i*nSpace+1]) +
	(gradu[i*nSpace+1] + gradv[i*nSpace+0])*(gradu[i*nSpace+1] + gradv[i*nSpace+0]);

      r_k[i]     = -nu_t*PiD4 + epsilon[i];
      dr_k_dk[i] = -dnu_t_dk*PiD4;
 
    }
}
void kEpsilon_epsilon_2D_Evaluate_sd(int nPoints,
				     int nSpace,
				     double sigma_e,
				     double c_1,
				     double c_2,
				     double c_mu,
				     double c_e,
				     double nu,
				     double *velocity,
				     double *gradu,
				     double *gradv,
				     double *k,
				     double *epsilon,
				     double *m_e,
				     double *dm_e,
				     double *phi_e,
				     double *dphi_e,
				     double *f_e,
				     double *df_e,
				     double *a_e,
				     double *da_e_de,
				     double *r_e,
				     double *dr_e_de)
		       
{
  int i,I;
  double nu_t,dnu_t_de,PiD4,eval,kval,disp,ddisp_de;
  const double div_eps = 1.0e-3;
  for (i = 0; i < nPoints; i++)
    {
      eval = epsilon[i] >= 0.0 ? epsilon[i] : 0.0;
      kval = k[i] >= 0.0 ? k[i] : 0.0;
      /*eddy viscosity*/
      nu_t     = c_mu*k[i]*k[i]/(epsilon[i]+div_eps);
      dnu_t_de =-c_mu*k[i]*k[i]/(epsilon[i]*epsilon[i]+div_eps);
      if (nu_t < 0.0)
	{
	  nu_t = 0.0; dnu_t_de = 0.0;
	}
      /*mwf debug
      printf("keps eval k[%d]=%g e[%d]=%g nu_t=%g dnu_t_de=%g\n",i,k[i],i,epsilon[i],
	     nu_t,dnu_t_de);
      */
      /*linear mass terms for e*/
      m_e[i] = epsilon[i];
      dm_e[i]= 1.0;
      
      /*linear advection*/
      for (I=0; I < nSpace; I++)
	{
	  f_e[i*nSpace+I] = epsilon[i]*velocity[i*nSpace+I];
	  df_e[i*nSpace+I]= velocity[i*nSpace+I];
	}
      /*linear potential*/ 
      phi_e[i] = epsilon[i];
      dphi_e[i]= 1.0;

      /*nonlinear diffusion*/
      for (I=0; I < nSpace; I++)
	{
	  a_e[i*2 + I] = nu_t/sigma_e + nu;
	  da_e_de[i*2 + I] = dnu_t_de/sigma_e;
	}
      
      /*production term*/
      /*4*Pi_D*/
      PiD4 = 2.0*(gradu[i*nSpace+0]*gradu[i*nSpace+0] + gradv[i*nSpace+1]*gradv[i*nSpace+1]) +
	(gradu[i*nSpace+1] + gradv[i*nSpace+0])*(gradu[i*nSpace+1] + gradv[i*nSpace+0]);

      disp = c_2*epsilon[i]*epsilon[i]/(k[i]+div_eps);
      ddisp_de = 2.0*c_2*epsilon[i]/(k[i]+div_eps); 
      if (disp < 0.0)
	{
	  disp = 0.0; ddisp_de = 0.0;
	}
      r_e[i]      = -c_1*k[i]*PiD4  + disp;/*c_2*epsilon[i]*epsilon[i]/(k[i]+div_eps);*/
      dr_e_de[i]  = ddisp_de; /*2.0*c_2*epsilon[i]/(k[i]+div_eps); */
 
    }
}
void kEpsilon_k_3D_Evaluate_sd(int nPoints,
			       int nSpace,
			       double sigma_k,
			       double c_mu,
			       double nu,
			       double *velocity,
			       double *gradu,
			       double *gradv,
			       double *gradw,
			       double *k,
			       double *epsilon,
			       double *m_k,
			       double *dm_k,
			       double *phi_k,
			       double *dphi_k,
			       double *f_k,
			       double *df_k,
			       double *a_k,
			       double *da_k_dk,
			       double *r_k,
			       double *dr_k_dk)
{
  int i,I;
  double nu_t,dnu_t_dk,PiD4,eval,kval;
  const double div_eps = 1.0e-6;
  for (i = 0; i < nPoints; i++)
    {
      eval = epsilon[i] >= 0.0 ? epsilon[i] : 0.0;
      kval = k[i] >= 0.0 ? k[i] : 0.0;
      /*eddy viscosity*/
      nu_t     = c_mu*k[i]*k[i]/(epsilon[i]+div_eps);
      dnu_t_dk = 2.0*c_mu*k[i]/(epsilon[i]+div_eps);
      if (nu_t < 0.0)
	{
	  nu_t = 0.0; dnu_t_dk = 0.0;
	}
      /*mwf debug
      printf("keps eval k[%d]=%g e[%d]=%g nu_t=%g dnu_t_dk=%g \n",i,k[i],i,epsilon[i],
	     nu_t,dnu_t_dk);
      */
      /*linear mass terms for k-e*/
      m_k[i] = k[i];
      dm_k[i]= 1.0;

      /*linear advection*/
      for (I=0; I < nSpace; I++)
	{
	  f_k[i*nSpace+I] = k[i]*velocity[i*nSpace+I];
	  df_k[i*nSpace+I]= velocity[i*nSpace+I];

	}
      /*linear potentials*/ 
      phi_k[i] = k[i];
      dphi_k[i]= 1.0; 

      /*nonlinear diffusion*/
      for (I=0; I < nSpace; I++)
	{
	  a_k[i*nSpace+I] = nu_t/sigma_k + nu;
	  da_k_dk[i*nSpace + I] = dnu_t_dk/sigma_k;
	}
      
      /*production term*/
      /*4*Pi_D*/
      PiD4 = 2.0*(gradu[i*nSpace+0]*gradu[i*nSpace+0] + 
		  gradv[i*nSpace+1]*gradv[i*nSpace+1] + 
		  gradw[i*nSpace+2]*gradw[i*nSpace+2]) 
	+
	(gradu[i*nSpace+1] + gradv[i*nSpace+0])*(gradu[i*nSpace+1] + gradv[i*nSpace+0])
	+
	(gradu[i*nSpace+2] + gradw[i*nSpace+0])*(gradu[i*nSpace+2] + gradw[i*nSpace+0])
	+
	(gradv[i*nSpace+2] + gradw[i*nSpace+1])*(gradv[i*nSpace+2] + gradw[i*nSpace+1]);


      r_k[i]     = -nu_t*PiD4 + epsilon[i];
      dr_k_dk[i] = -dnu_t_dk*PiD4;
 
    }
}
void kEpsilon_epsilon_3D_Evaluate_sd(int nPoints,
				     int nSpace,
				     double sigma_e,
				     double c_1,
				     double c_2,
				     double c_mu,
				     double c_e,
				     double nu,
				     double *velocity,
				     double *gradu,
				     double *gradv,
				     double *gradw,
				     double *k,
				     double *epsilon,
				     double *m_e,
				     double *dm_e,
				     double *phi_e,
				     double *dphi_e,
				     double *f_e,
				     double *df_e,
				     double *a_e,
				     double *da_e_de,
				     double *r_e,
				     double *dr_e_de)
		       
{
  int i,I;
  double nu_t,dnu_t_de,PiD4,eval,kval,disp,ddisp_de;
  const double div_eps = 1.0e-6;
  for (i = 0; i < nPoints; i++)
    {
      eval = epsilon[i] >= 0.0 ? epsilon[i] : 0.0;
      kval = k[i] >= 0.0 ? k[i] : 0.0;
      /*eddy viscosity*/
      nu_t     = c_mu*k[i]*k[i]/(epsilon[i]+div_eps);
      dnu_t_de =-c_mu*k[i]*k[i]/(epsilon[i]*epsilon[i]+div_eps);
      if (nu_t < 0.0)
	{
	  nu_t = 0.0; dnu_t_de = 0.0;
	}
      /*mwf debug
      printf("keps eval k[%d]=%g e[%d]=%g nu_t=%g dnu_t_de=%g\n",i,k[i],i,epsilon[i],
	     nu_t,dnu_t_de);
      */
      /*linear mass terms for e*/
      m_e[i] = epsilon[i];
      dm_e[i]= 1.0;
      
      /*linear advection*/
      for (I=0; I < nSpace; I++)
	{
	  f_e[i*nSpace+I] = epsilon[i]*velocity[i*nSpace+I];
	  df_e[i*nSpace+I]= velocity[i*nSpace+I];
	}
      /*linear potentials*/ 
      phi_e[i] = epsilon[i];
      dphi_e[i]= 1.0;

      /*nonlinear diffusion*/
      for (I=0; I < nSpace; I++)
	{
	  a_e[i*nSpace + I] = nu_t/sigma_e + nu;
	  da_e_de[i*nSpace + I] = dnu_t_de/sigma_e;
	}
      
      /*production term*/
      /*4*Pi_D*/
      PiD4 = 2.0*(gradu[i*nSpace+0]*gradu[i*nSpace+0] + 
		  gradv[i*nSpace+1]*gradv[i*nSpace+1] + 
		  gradw[i*nSpace+2]*gradw[i*nSpace+2]) 
	+
	(gradu[i*nSpace+1] + gradv[i*nSpace+0])*(gradu[i*nSpace+1] + gradv[i*nSpace+0])
	+
	(gradu[i*nSpace+2] + gradw[i*nSpace+0])*(gradu[i*nSpace+2] + gradw[i*nSpace+0])
	+
	(gradv[i*nSpace+2] + gradw[i*nSpace+1])*(gradv[i*nSpace+2] + gradw[i*nSpace+1]);


      disp = c_2*epsilon[i]*epsilon[i]/(k[i]+div_eps);
      ddisp_de = 2.0*c_2*epsilon[i]/(k[i]+div_eps); 
      if (disp < 0.0)
	{
	  disp = 0.0; ddisp_de = 0.0;
	}
      r_e[i]      = -c_1*k[i]*PiD4  + disp;/*c_2*epsilon[i]*epsilon[i]/(k[i]+div_eps);*/
      dr_e_de[i]  = ddisp_de; /*2.0*c_2*epsilon[i]/(k[i]+div_eps); */
 
    }
}

/***********************************************************************
   Evolution equations (mass and momentum conservation) 
   for Reynolds Averaged Navier Stokes formulation with k-epsilon turbulence
   model


Reynolds averaged NS equations

\deld \bar{\vec v} = 0

\pd{\bar{\vec v}}{t} + \deld \left(\bar{\vec v} \outer \bar{\vec v}\right) 
               -\nu \deld \ten \bar{D} + \frac{1}{\rho}\grad \bar p  
               - \frac{1}{rho}\deld \ten{R} = 0

Reynolds stress term

\ten R = -\rho <\vec v^{'}\outer \vec v^{'}>
\frac{1}{\rho}\ten{R} = 2 \nu_t \bar{D} - \frac{2}{3}k\ten{I}

D_{ij}(\vec v) = \frac{1}{2} \left( \pd{v_i}{x_j} + \pd{v_j}{x_i})
\ten D \bar{\ten D} = D(<\vec v>), \ten D^{'} = \ten D(\vec v^{'})


For simplicity, the 2/3 k\ten{I} part of the Reynolds stress will be incorporated into
   the momentum source term

The eddy viscosity will simply appear as an additional, spatially varying
  term in the momentum diffusion tensor 

This version assumes standard NS terms have been evaluated and only 
  updates diffusion tensor and source term
 ***********************************************************************/
void ReynoldsAveragedNavierStokes_kEpsilon_2D_Update(const int nPoints,
						     const double nu,
						     const double c_mu,
						     const double* k,
						     const double* grad_k,
						     const double* epsilon,
						     double *mom_u_diff_ten,
						     double *mom_v_diff_ten,
						     double *mom_uv_diff_ten,
						     double *mom_vu_diff_ten,
						     double *mom_u_source,
						     double *mom_v_source)
{
  int i;
  double nu_t,eval,kval;
  const double twoThirds = 2.0/3.0;
  const double div_zero = 1.0e-6;
  for (i=0;i<nPoints;i++)
    {
      eval = epsilon[i] >= 0.0 ? epsilon[i] : 0.0;
      kval = k[i] >= 0.0 ? k[i] : 0.0;

      nu_t= c_mu*kval*kval/(eval+div_zero);
#ifdef SCALAR_DIFFUSION
     //u momentum diffusion tensor
      mom_u_diff_ten[i*4+0] += nu_t;
      mom_u_diff_ten[i*4+3] += nu_t;

      //v momentum diffusion tensor
      mom_v_diff_ten[i*4+0] += nu_t;
      mom_v_diff_ten[i*4+3] += nu_t;
#else
      //u momentum diffusion tensor
      mom_u_diff_ten[i*4+0] += 2.0*nu_t;
      mom_u_diff_ten[i*4+3] += nu_t;
      mom_uv_diff_ten[i*4+2]+= nu_t;

      //v momentum diffusion tensor
      mom_v_diff_ten[i*4+0] += nu_t;
      mom_v_diff_ten[i*4+3] += 2.0*nu_t;
      mom_vu_diff_ten[i*4+1] += nu_t;
#endif


      //momentum sources
      mom_u_source[i] += twoThirds*grad_k[i*2+0]; 
      mom_v_source[i] += twoThirds*grad_k[i*2+1]; 
      
    }
}
/***********************************************************************
   Evolution equations (mass and momentum conservation) 
   for Reynolds Averaged Navier Stokes formulation with k-epsilon turbulence
   model


Reynolds averaged NS equations

\deld \bar{\vec v} = 0

\pd{\bar{\vec v}}{t} + \deld \left(\bar{\vec v} \outer \bar{\vec v}\right) 
               -\nu \deld \ten \bar{D} + \frac{1}{\rho}\grad \bar p  
               - \frac{1}{rho}\deld \ten{R} = 0

Reynolds stress term

\ten R = -\rho <\vec v^{'}\outer \vec v^{'}>
\frac{1}{\rho}\ten{R} = 2 \nu_t \bar{D} - \frac{2}{3}k\ten{I}

D_{ij}(\vec v) = \frac{1}{2} \left( \pd{v_i}{x_j} + \pd{v_j}{x_i})
\ten D \bar{\ten D} = D(<\vec v>), \ten D^{'} = \ten D(\vec v^{'})


For simplicity, the 2/3 k\ten{I} part of the Reynolds stress will be incorporated into
   the momentum source term

The eddy viscosity will simply appear as an additional, spatially varying
  term in the momentum diffusion tensor 

This version assumes standard NS terms have been evaluated and only 
  updates diffusion tensor and source term
 ***********************************************************************/
void ReynoldsAveragedNavierStokes_kEpsilon_2D_Update_sd(const int nPoints,
							const double rho,
							const double nu,
							const double c_mu,
							const double* k,
							const double* grad_k,
							const double* epsilon,
							double *mom_u_diff_ten,
							double *mom_v_diff_ten,
							double *mom_uv_diff_ten,
							double *mom_vu_diff_ten,
							double *mom_u_source,
							double *mom_v_source)
{
  int i;
  double nu_t,eval,kval;
  const double twoThirds = 2.0/3.0;
  const double div_zero = 1.0e-6;
  for (i=0;i<nPoints;i++)
    {
      eval = epsilon[i] >= 0.0 ? epsilon[i] : 0.0;
      kval = k[i] >= 0.0 ? k[i] : 0.0;

      nu_t= c_mu*kval*kval/(eval+div_zero);

      //u momentum diffusion tensor
      mom_u_diff_ten[i*2+0] += 2.0*nu_t;
      mom_u_diff_ten[i*2+1] += nu_t;
      mom_uv_diff_ten[i]+= nu_t;

      //v momentum diffusion tensor
      mom_v_diff_ten[i*2+0] += nu_t;
      mom_v_diff_ten[i*2+1] += 2.0*nu_t;
      mom_vu_diff_ten[i] += nu_t;


      //momentum sources
      mom_u_source[i] += twoThirds*grad_k[i*2+0]; 
      mom_v_source[i] += twoThirds*grad_k[i*2+1]; 
      
    }
}
void ReynoldsAveragedNavierStokes_kEpsilon_3D_Update(const int nPoints,
						     const double nu,
						     const double c_mu,
						     const double* k,
						     const double* grad_k,
						     const double* epsilon,
						     double *mom_u_diff_ten,
						     double *mom_v_diff_ten,
						     double *mom_w_diff_ten,
						     double *mom_uv_diff_ten,
						     double *mom_uw_diff_ten,
						     double *mom_vu_diff_ten,
						     double *mom_vw_diff_ten,
						     double *mom_wu_diff_ten,
						     double *mom_wv_diff_ten,
						     double *mom_u_source,
						     double *mom_v_source,
						     double *mom_w_source)
{
  int i;
  double nu_t,eval,kval;
  const double twoThirds = 2.0/3.0;
  const double div_zero = 1.0e-6;
  for (i=0;i<nPoints;i++)
    {
      eval = epsilon[i] >= 0.0 ? epsilon[i] : 0.0;
      kval = k[i] >= 0.0 ? k[i] : 0.0;

      nu_t= c_mu*kval*kval/(eval+div_zero);
      //u momentum diffusion tensor
      mom_u_diff_ten[i*9+0] += 2.0*nu_t;
      mom_u_diff_ten[i*9+4] += nu_t;
      mom_u_diff_ten[i*9+8] += nu_t;

      mom_uv_diff_ten[i*9+3]+=nu_t;

      mom_uw_diff_ten[i*9+6]+=nu_t;

      //v momentum diffusion tensor
      mom_v_diff_ten[i*9+0] += nu_t;
      mom_v_diff_ten[i*9+4] += 2.0*nu_t;
      mom_v_diff_ten[i*9+8] += nu_t;

      mom_vu_diff_ten[i*9+1]+=nu_t;

      mom_vw_diff_ten[i*9+7]+=nu_t;

      //w momentum diffusion tensor
      mom_w_diff_ten[i*9+0] += nu_t;
      mom_w_diff_ten[i*9+4] += nu_t;
      mom_w_diff_ten[i*9+8] += 2.0*nu_t;

      mom_wu_diff_ten[i*9+2]+=nu_t;

      mom_wv_diff_ten[i*9+5]+=nu_t;


      //momentum sources
      mom_u_source[i] += twoThirds*grad_k[i*3+0]; 
      mom_v_source[i] += twoThirds*grad_k[i*3+1]; 
      mom_w_source[i] += twoThirds*grad_k[i*3+2]; 
      
    }
}
void ReynoldsAveragedNavierStokes_kEpsilon_3D_Update_sd(const int nPoints,
							const double nu,
							const double c_mu,
							const double* k,
							const double* grad_k,
							const double* epsilon,
							double *mom_u_diff_ten,
							double *mom_v_diff_ten,
							double *mom_w_diff_ten,
							double *mom_uv_diff_ten,
							double *mom_uw_diff_ten,
							double *mom_vu_diff_ten,
							double *mom_vw_diff_ten,
							double *mom_wu_diff_ten,
							double *mom_wv_diff_ten,
							double *mom_u_source,
							double *mom_v_source,
							double *mom_w_source)
{
  int i;
  double nu_t,eval,kval;
  const double twoThirds = 2.0/3.0;
  const double div_zero = 1.0e-6;
  for (i=0;i<nPoints;i++)
    {
      eval = epsilon[i] >= 0.0 ? epsilon[i] : 0.0;
      kval = k[i] >= 0.0 ? k[i] : 0.0;

      nu_t= c_mu*kval*kval/(eval+div_zero);
      //u momentum diffusion tensor
      mom_u_diff_ten[i*3+0] += 2.0*nu_t;
      mom_u_diff_ten[i*3+1] += nu_t;
      mom_u_diff_ten[i*3+2] += nu_t;

      mom_uv_diff_ten[i]+=nu_t;

      mom_uw_diff_ten[i]+=nu_t;

      //v momentum diffusion tensor
      mom_v_diff_ten[i*3+0] += nu_t;
      mom_v_diff_ten[i*3+1] += 2.0*nu_t;
      mom_v_diff_ten[i*3+2] += nu_t;

      mom_vu_diff_ten[i]+=nu_t;

      mom_vw_diff_ten[i]+=nu_t;

      //w momentum diffusion tensor
      mom_w_diff_ten[i*3+0] += nu_t;
      mom_w_diff_ten[i*3+1] += nu_t;
      mom_w_diff_ten[i*3+2] += 2.0*nu_t;

      mom_wu_diff_ten[i]+=nu_t;

      mom_wv_diff_ten[i]+=nu_t;
      //momentum sources
      mom_u_source[i] += twoThirds*grad_k[i*3+0]; 
      mom_v_source[i] += twoThirds*grad_k[i*3+1]; 
      mom_w_source[i] += twoThirds*grad_k[i*3+2]; 
      
    }
}

void scriptedSphereMotionSignedDistance(const int nPoints,
					const double t,
					const int nSpace,
					const int nSpheres,
					const double * radii,
					const double * centers,
					const double * x,
					double * phi,
					double * n)
{
  /*brute force calculation of signed distance from nSphere moving spheres*/
  int i,j,k;
  double minDistance,distance,tmp;

  for (k=0; k < nPoints; k++)
    {
      for (j=0; j < nSpheres; j++)
	{
	  tmp = 0.0;
	  for (i=0; i < nSpace; i++)
	    {
	      tmp += (x[3*k+i]-centers[3*j+i])*(x[3*k+i]-centers[3*j+i]);
	    }
	  distance = sqrt(tmp) - radii[j];
	  if (j == 0) 
	    {
	      minDistance = distance;
	      for (i=0; i < nSpace; i++)
		{
		  n[k*nSpace + i]= (x[3*k+i]-centers[3*j+i])/(distance+1.0e-12);
		}
	    }
	  else if (fabs(distance) < fabs(minDistance))
	    {
	      minDistance=distance;
	      for (i=0; i < nSpace; i++)
		{
		  n[k*nSpace + i]= (x[3*k+i]-centers[3*j+i])/(distance+1.0e-12);
		}
	    }
	}/*sphere loop*/
      phi[k] = minDistance;
      
    }/*point loop*/


}
			     
void shallowWater_1D_Evaluate(const int nPoints,
                              const double h_eps,
			      const double g,
			      const double bedFrictionCoefficient,
			      const double bedFrictionPower,
			      const double eddyViscosity,
			      const double* x, /*bathymetry elevation in index 1 (y)*/
			      const double* db_dx,/*db/dx (bed slope) */
			      const double* h,
			      const double* hu,
			      double *H,
			      double *mass_acc,
			      double *dmass_acc_dh,
			      double *mom_acc,
			      double *dmom_acc_dhu,
			      double *mass_adv,
			      double *dmass_adv_dhu,
			      double *mom_adv,
			      double *dmom_adv_dh,
			      double *dmom_adv_dhu,
			      double *mom_source,
			      double *dmom_source_dh,
			      double *dmom_source_dhu,
			      double *mom_diff)
{
  int k;
  double u,du_dh,du_dhu,heval,hpow,hunorm;
  for (k=0;k<nPoints;k++)
    {
      if (h[k] < h_eps)
        {
          u = 0.0;
          du_dh = 0.0;
          du_dhu = 0.0;
        }
      else
	{
	  u = hu[k]/h[k];
	  du_dh = -hu[k]/(h[k]*h[k]);
	  du_dhu = 1.0/h[k];
	}
      /*elevation above reference point*/
      H[k] = h[k] + x[k*3+1];

      mass_acc[k] = h[k];
      dmass_acc_dh[k] = 1.0;

      mom_acc[k] = hu[k];
      dmom_acc_dhu[k] = 1.0;

      mass_adv[k] = hu[k];
      dmass_adv_dhu[k] = 1.0;
      
      mom_adv[k] = hu[k]*u+0.5*g*h[k]*h[k];
      dmom_adv_dh[k] = hu[k]*du_dh+g*h[k];
      dmom_adv_dhu[k] = u + hu[k]*du_dhu;

      /*constant eddy viscosity for now*/
      mom_diff[k] = eddyViscosity;
      /*bathymetry contributions, go on left hand side*/
      mom_source[k]    = g*h[k]*db_dx[k];
      dmom_source_dh[k]= g*db_dx[k];
      /*bed Friction contributions, go on left hand side*/
      heval=fmax(1.0e-3,h[k]);
      hpow =pow(heval,-bedFrictionPower-2.0);/*have extra powers of h because formula numerator has hu*/
      hunorm = fabs(hu[k]);
      mom_source[k]   += g*hu[k]*bedFrictionCoefficient*hunorm*hpow; /*recall g is magnitude of gravity here*/
      dmom_source_dh[k]   += -(bedFrictionPower+2.0)*g*hu[k]*bedFrictionCoefficient*hunorm*pow(heval,-bedFrictionPower-3.0);
      dmom_source_dhu[k]  = g*bedFrictionCoefficient*hunorm*hpow
	+g*bedFrictionCoefficient*hu[k]*hu[k]*hpow/(hunorm+h_eps);
    }
}

void shallowWater_2D_Evaluate(const int nPoints,
                              const double h_eps,
			      const double g,
			      const double bedFrictionCoefficient,
			      const double bedFrictionPower,
			      const double eddyViscosity,
			      const double* x, /*bathymetry elevation in index 2 (z)*/
			      const double* grad_b,/*\nabla b (bed slope) */
			      const double* h,
			      const double* hu,
			      const double* hv,
			      double *H,
			      double *mass_acc,
			      double *dmass_acc_dh,
			      double *mom_u_acc,
			      double *dmom_u_acc_dhu,
			      double *mom_v_acc,
			      double *dmom_v_acc_dhv,
			      double *mass_adv,
			      double *dmass_adv_dhu,
			      double *dmass_adv_dhv,
			      double *mom_u_adv,
			      double *dmom_u_adv_dh,
			      double *dmom_u_adv_dhu,
			      double *dmom_u_adv_dhv,
			      double *mom_v_adv,
			      double *dmom_v_adv_dh,
			      double *dmom_v_adv_dhu,
			      double *dmom_v_adv_dhv,
			      double *mom_u_diff,
			      double *mom_v_diff,
			      double *mom_u_source,
			      double *dmom_u_source_dh,
			      double *dmom_u_source_dhu,
			      double *dmom_u_source_dhv,
			      double *mom_v_source,
			      double *dmom_v_source_dh,
			      double *dmom_v_source_dhu,
			      double *dmom_v_source_dhv)
{
  int k;
  double u,du_dh,du_dhu,v,dv_dh,dv_dhv,heval,unorm,hpow;
  for (k=0;k<nPoints;k++)
    {
      if (h[k] < h_eps)
	{
	  u =0.0;
	  du_dh = 0.0;
	  du_dhu = 0.0;

	  v =0.0;
	  dv_dh = 0.0;
	  dv_dhv = 0.0;
	}
      else
	{
	  u = hu[k]/h[k];
	  du_dh = -hu[k]/(h[k]*h[k]);
	  du_dhu = 1.0/h[k];
          
	  v = hv[k]/h[k];
	  dv_dh = -hv[k]/(h[k]*h[k]);
	  dv_dhv = 1.0/h[k];
	}
      //height above reference point
      H[k] = h[k] + x[k*3+2];
      //mass
      mass_acc[k] = h[k];
      dmass_acc_dh[k] = 1.0;
      
      mass_adv[2*k+0] = hu[k];
      dmass_adv_dhu[2*k+0] = 1.0;
      
      mass_adv[2*k+1] = hv[k];
      dmass_adv_dhv[2*k+1] = 1.0;
      
      //mom u
      mom_u_acc[k] = hu[k];
      dmom_u_acc_dhu[k] = 1.0;

      mom_u_adv[2*k+0] = hu[k]*u+0.5*g*h[k]*h[k];
      dmom_u_adv_dh[2*k+0] = hu[k]*du_dh+g*h[k];
      dmom_u_adv_dhu[2*k+0] = u+hu[k]*du_dhu;

      mom_u_adv[2*k+1] = hu[k]*v;
      dmom_u_adv_dh[2*k+1] = hu[k]*dv_dh;
      dmom_u_adv_dhu[2*k+1] = v;
      dmom_u_adv_dhv[2*k+1] = hu[k]*dv_dhv;

      //constant eddy viscosity for now,
      //temporarily assume full tensor storage even though diagonal
      //mom_u_diff[4*k+0] = eddyViscosity;
      //mom_u_diff[4*k+3] = eddyViscosity;
      mom_u_diff[2*k+0] = eddyViscosity;
      mom_u_diff[2*k+1] = eddyViscosity;
      
      //mom v
      mom_v_acc[k] = hv[k];
      dmom_v_acc_dhv[k] = 1.0;

      mom_v_adv[2*k+0] = hv[k]*u;
      dmom_v_adv_dh[2*k+0] = hv[k]*du_dh;
      dmom_v_adv_dhu[2*k+0] = hv[k]*du_dhu;
      dmom_v_adv_dhv[2*k+0] = u;

      mom_v_adv[2*k+1] = hv[k]*v+0.5*g*h[k]*h[k];
      dmom_v_adv_dh[2*k+1] = hv[k]*dv_dh+g*h[k];
      dmom_v_adv_dhv[2*k+1] = v + hv[k]*dv_dhv;

      //constant eddy viscosity for now,
      //temporarily assume full tensor storage even though diagonal
      //mom_v_diff[4*k+0] = eddyViscosity;
      //mom_v_diff[4*k+3] = eddyViscosity;
      mom_v_diff[2*k+0] = eddyViscosity;
      mom_v_diff[2*k+1] = eddyViscosity;
      /*bathymetry contributions, goes on left hand side*/
      mom_u_source[k]    = g*h[k]*grad_b[k*2+0];
      dmom_u_source_dh[k]= g*grad_b[k*2+0];

      mom_v_source[k]    = g*h[k]*grad_b[k*2+1];
      dmom_v_source_dh[k]= g*grad_b[k*2+1];
      /*bed Friction contributions, go on left hand side*/
      heval=fmax(1.0e-3,h[k]);
      hpow =pow(heval,-bedFrictionPower-2.0);/*have extra powers of h because formula numerator has hu*/
      unorm = sqrt(hu[k]*hu[k] + hv[k]*hv[k]);

      mom_u_source[k]   += g*hu[k]*bedFrictionCoefficient*unorm*hpow;
      dmom_u_source_dh[k]   += -(bedFrictionPower+2.0)*g*hu[k]*bedFrictionCoefficient*unorm*pow(heval,-bedFrictionPower-3.0);
      dmom_u_source_dhu[k]   = g*bedFrictionCoefficient*unorm*hpow
	+g*bedFrictionCoefficient*hu[k]*hu[k]*hpow/(unorm+h_eps);
      dmom_u_source_dhv[k]   = g*bedFrictionCoefficient*unorm*hpow
	+g*bedFrictionCoefficient*hu[k]*hv[k]*hpow/(unorm+h_eps);
      
      mom_v_source[k]   += g*hv[k]*bedFrictionCoefficient*unorm*hpow;
      dmom_v_source_dh[k]   += -(bedFrictionPower+2.0)*g*hv[k]*bedFrictionCoefficient*unorm*pow(heval,-bedFrictionPower-3.0);
      dmom_v_source_dhu[k]   = g*bedFrictionCoefficient*unorm*hpow
	+g*bedFrictionCoefficient*hu[k]*hv[k]*hpow/(unorm+h_eps);
      dmom_v_source_dhv[k]   = g*bedFrictionCoefficient*unorm*hpow
	+g*bedFrictionCoefficient*hv[k]*hv[k]*hpow/(unorm+h_eps);
     
    }
}

void conservativeHeadRichardsMualemVanGenuchtenHetEvaluateV2withUpwind(const int upwindFlag,
								       const int computeAverages,
								       const int nSimplex,
								       const int nPointsPerSimplex,
								       const int nSpace,
								       const int nQuadraturePoints_elementBoundary,
								       const proteus_index_t* elementBoundaryElementsArray,
								       const int* quadraturePointToElementBoundary,
								       const int* materialTypes,
								       const double rho,
								       const double beta,
								       const double* gravity,
								       const double* alpha,
								       const double* n_vg,
								       const double* thetaR,
								       const double* thetaSR,
								       const double* KWs,
								       const double *u,
								       const double *gradu,
								       const double *n_global,
								       const double *dV,
								       double *mass,
								       double *dmass,
								       double *f_avg,
								       double *df_avg,
								       double *a_avg,
								       double *da_avg,
								       double *f,
								       double *df,
								       double *a,
								       double *da)
{
  int i,j,k,I,matID;
  const int nSpace2=nSpace*nSpace;
  register double psiC,
    pcBar,pcBar_n,pcBar_nM1,pcBar_nM2,
    onePlus_pcBar_n,
    sBar,sqrt_sBar,DsBar_DpsiC,
    thetaW,DthetaW_DpsiC,
    vBar,vBar2,DvBar_DpsiC,
    KW,DKW_DpsiC,
    rho2=rho*rho,
    thetaS,
    rhom,drhom,m;
  proteus_index_t eN_upwind; int ebN_global;
  register double drive,avgfI,avgdfI,avgaII,avgdaII,vol;
  /*
    loop through and compute point values as normal
       could add harmonic average for point values at faces here

    if upwinding,
       loop through and compute average for each simplex
       loop through each point,
           find face for that point
           find upwind direction for that face
           set point value and derivative to average from upwind neighbor
            
   */
  for (i=0; i < nSimplex; i++)
    {
      matID= materialTypes[i];
      for (j=0;j<nPointsPerSimplex;j++)
	{
	  k = i*nPointsPerSimplex + j;
	  psiC = -u[k];
	  m = 1.0 - 1.0/n_vg[matID];
	  thetaS = thetaR[matID] + thetaSR[matID];
	  if (psiC > 0.0)
	    {
	      pcBar = alpha[matID]*psiC;
	      pcBar_nM2 = pow(pcBar,n_vg[matID]-2);
	      pcBar_nM1 = pcBar_nM2*pcBar;
	      pcBar_n   = pcBar_nM1*pcBar;
	      onePlus_pcBar_n = 1.0 + pcBar_n;
	      
	      sBar = pow(onePlus_pcBar_n,-m);
	      /* using -mn = 1-n */
	      DsBar_DpsiC = alpha[matID]*(1.0-n_vg[matID])*(sBar/onePlus_pcBar_n)*pcBar_nM1;
	      
	      vBar = 1.0-pcBar_nM1*sBar;
	      vBar2 = vBar*vBar;
	      DvBar_DpsiC = -alpha[matID]*(n_vg[matID]-1.0)*pcBar_nM2*sBar - pcBar_nM1*DsBar_DpsiC;
          
	      thetaW = thetaSR[matID]*sBar + thetaR[matID];
	      DthetaW_DpsiC = thetaSR[matID] * DsBar_DpsiC;
          
	      sqrt_sBar = sqrt(sBar);
	      KW= KWs[matID]*sqrt_sBar*vBar2;
	      DKW_DpsiC= KWs[matID]*
		((0.5/sqrt_sBar)*DsBar_DpsiC*vBar2
		 +
		 2.0*sqrt_sBar*vBar*DvBar_DpsiC);
	    }
	  else
	    {
	      thetaW        = thetaS;
	      DthetaW_DpsiC = 0.0;
	      KW            = KWs[matID];
	      DKW_DpsiC     = 0.0;
	    }
          //slight compressibility
          rhom = rho*exp(beta*u[k]);
          drhom = beta*rhom;

          mass[k] = rhom*thetaW;
          dmass[k] = -rhom*DthetaW_DpsiC+drhom*thetaW;
	  //mass[k] = rho*thetaW;
	  //dmass[k] = -rho*DthetaW_DpsiC;
	  for (I=0;I<nSpace;I++)
	    {
	      f[k*nSpace+I]  = rho2*KW*gravity[I];
	      df[k*nSpace+I] = -rho2*DKW_DpsiC*gravity[I];
	      a[i*nPointsPerSimplex*nSpace2 + j*nSpace2+I*nSpace+I]  = rho*KW;
	      da[i*nPointsPerSimplex*nSpace2 + j*nSpace2+I*nSpace+I] = -rho*DKW_DpsiC;
	    }/*I*/
	}/*k*/
    }/*j*/
  if (upwindFlag == 1 || upwindFlag == 2)
    {
      if (computeAverages == 1) /*compute averages for element quadrature call only*/
	{
	  for (i=0; i < nSimplex; i++)
	    {
	      matID= materialTypes[i];
	      vol = 0.0;
	      for (j=0; j < nPointsPerSimplex; j++)
		vol += dV[i*nPointsPerSimplex + j];
	      /*put dimensions on outside of loop to make averaging accumulation easier*/
	      for (I=0; I < nSpace; I++)
		{
		  avgaII = 0.0; avgfI = 0.0;
		  avgdaII= 0.0; avgdfI= 0.0;
		  for (j=0;j<nPointsPerSimplex;j++)
		    {
		      k = i*nPointsPerSimplex + j;
		      avgfI   += f[k*nSpace + I]*dV[k];
		      avgdfI  += df[k*nSpace+ I]*dV[k];
		      avgaII  += a[i*nPointsPerSimplex*nSpace2 + j*nSpace2+I*nSpace+I]*dV[k];
		      avgdaII += da[i*nPointsPerSimplex*nSpace2 + j*nSpace2+I*nSpace+I]*dV[k];
		    }
		  avgfI /= vol; avgdfI /= vol; avgaII /= vol; avgdaII /= vol;
		  f_avg[i*nSpace + I] = avgfI;
		  df_avg[i*nSpace +I] = avgdfI;
		  a_avg[i*nSpace2+I*nSpace+I] = avgaII;
		  da_avg[i*nSpace2+I*nSpace+I]= avgdaII;
		  /*mwf debug
		  printf("RE V2 upwind nSimplex= %d nPerSimplex= %d i=%d j=%d I=%d k=%d \n",nSimplex,nPointsPerSimplex,i,j,I,k);
		  */
		}/*space dim loop*/
	    }/*simplex loop for averages*/
	}
      for (i=0; i < nSimplex; i++)
	{
	  matID= materialTypes[i];
	  for (j=0;j<nPointsPerSimplex;j++)
	    {
	      k = i*nPointsPerSimplex + j;
	      ebN_global = quadraturePointToElementBoundary[k];
	      drive = 0.0;
	      for (I=0; I < nSpace; I++)
		{
		  drive += (rho2*KWs[matID]*gravity[I]-rho*KWs[matID]*gradu[k*nSpace+I])*n_global[ebN_global*nQuadraturePoints_elementBoundary*nSpace + 
												  0*nSpace + I];
		}
	      if (drive >= 0.0 || elementBoundaryElementsArray[ebN_global*2 + 1] < 0)
		eN_upwind = elementBoundaryElementsArray[ebN_global*2 + 0];
	      else
		eN_upwind = elementBoundaryElementsArray[ebN_global*2 + 1];
	      /*mwf debug
	      printf("RE V2 upwind nSimplex= %d nPerSimplex= %d i=%d j=%d k=%d ebN_global= %d eN_upwind= %d\n",nSimplex,nPointsPerSimplex,i,j,k,ebN_global,
		     eN_upwind);
	      */
	      for (I = 0; I < nSpace; I++)
		{
		  f[k*nSpace + I] = f_avg[eN_upwind*nSpace+I];
		  /*mwf don't update derivative?*/
		  /*df[k*nSpace+ I] = df_avg[eN_upwind*nSpace+I];*/
		  a[i*nPointsPerSimplex*nSpace2 + j*nSpace2+ I*nSpace+I] = a_avg[eN_upwind*nSpace2+I*nSpace+I];
		  /*da[k*nSpace2+ I*nSpace+I] = da_avg[eN_upwind*nSpace2+I*nSpace+I];*/
		}
	      if (upwindFlag == 2)
		{
		  for (I = 0; I < nSpace; I++)
		    {
		      df[k*nSpace+ I] = df_avg[eN_upwind*nSpace + I];
		      da[i*nPointsPerSimplex*nSpace2 + j*nSpace2+ I*nSpace+I] = da_avg[eN_upwind*nSpace2+I*nSpace+I];
		    }
		}
	    }/*point loop for simplex*/
	}/*simplex loop for upwinding*/
    }/*if upwind*/
}
void conservativeHeadRichardsMualemVanGenuchtenHetEvaluateV2withUpwindAndHarm(const int upwindFlag,
									      const int computeAverages,
									      const int nSimplex,
									      const int nPointsPerSimplex,
									      const int nSpace,
									      const int nQuadraturePoints_elementBoundary,
									      const proteus_index_t* elementBoundaryElementsArray,
									      const int* quadraturePointToElementBoundary,
									      const int* materialTypes,
									      const double rho,
									      const double beta,
									      const double* gravity,
									      const double* alpha,
									      const double* n_vg,
									      const double* thetaR,
									      const double* thetaSR,
									      const double* KWs,
									      const double *u,
									      const double *gradu,
									      const double *n_global,
									      const double *dV,
									      double *mass,
									      double *dmass,
									      double *f_avg,
									      double *df_avg,
									      double *a_avg,
									      double *da_avg,
									      double *f,
									      double *df,
									      double *a,
									      double *da)
{
  int i,j,k,I,matID;
  const int nSpace2=nSpace*nSpace;
  register double psiC,
    pcBar,pcBar_n,pcBar_nM1,pcBar_nM2,
    onePlus_pcBar_n,
    sBar,sqrt_sBar,DsBar_DpsiC,
    thetaW,DthetaW_DpsiC,
    vBar,vBar2,DvBar_DpsiC,
    KWr,DKWr_DpsiC,
    rho2=rho*rho,
    thetaS,
    rhom,drhom,m;
  proteus_index_t eN_upwind; int ebN_global;
  register double drive,KWs_harm,KWs_left,KWs_right,krw_val,dkrw_val,vol;
  /*
    loop through and compute point values for m as usual
       store point values for kr, dkr and
       calculate average value for kr, dkr

    loop again through each point
       find face for that point
       compute harmonic average of (rho*Ks)

       if upwinding,
           find upwind direction for that face
           set point value to 
           a_{ij,f} = (rho*Ks)_{ij,h,f}k_r^{up}
           f_{i}    = (rho*rho*Ks)_{h,f}k_r^{up}\vec g
       else
           set point value to 
           a_{ij,f} = (rho*Ks)_{ij,h,f}k_r(\psi_f)
           f_{i}    = (rho*rho*Ks)_{h,f}k_r(\psi_f)\vec g
     
   */

  if (upwindFlag > 0)
    assert(computeAverages);
  for (i=0; i < nSimplex; i++)
    {
      matID= materialTypes[i];
      /*for computing averages*/
      vol = 0.0;
      for (j=0; j < nPointsPerSimplex; j++)
	vol += dV[i*nPointsPerSimplex + j];
      /*store average information with a_avg first entry for now*/
      if (computeAverages)
	{
	  a_avg[i*nSpace2+0*nSpace+0] = 0.0;
	  da_avg[i*nSpace2+0*nSpace+0] = 0.0;
	}
      for (j=0;j<nPointsPerSimplex;j++)
	{
	  k = i*nPointsPerSimplex + j;
	  psiC = -u[k];
	  m = 1.0 - 1.0/n_vg[matID];
	  thetaS = thetaR[matID] + thetaSR[matID];
	  if (psiC > 0.0)
	    {
	      pcBar = alpha[matID]*psiC;
	      pcBar_nM2 = pow(pcBar,n_vg[matID]-2);
	      pcBar_nM1 = pcBar_nM2*pcBar;
	      pcBar_n   = pcBar_nM1*pcBar;
	      onePlus_pcBar_n = 1.0 + pcBar_n;
	      
	      sBar = pow(onePlus_pcBar_n,-m);
	      /* using -mn = 1-n */
	      DsBar_DpsiC = alpha[matID]*(1.0-n_vg[matID])*(sBar/onePlus_pcBar_n)*pcBar_nM1;
	      
	      vBar = 1.0-pcBar_nM1*sBar;
	      vBar2 = vBar*vBar;
	      DvBar_DpsiC = -alpha[matID]*(n_vg[matID]-1.0)*pcBar_nM2*sBar - pcBar_nM1*DsBar_DpsiC;
          
	      thetaW = thetaSR[matID]*sBar + thetaR[matID];
	      DthetaW_DpsiC = thetaSR[matID] * DsBar_DpsiC;
          
	      sqrt_sBar = sqrt(sBar);
	      KWr= sqrt_sBar*vBar2;
	      DKWr_DpsiC= 
		((0.5/sqrt_sBar)*DsBar_DpsiC*vBar2
		 +
		 2.0*sqrt_sBar*vBar*DvBar_DpsiC);
	    }
	  else
	    {
	      thetaW        = thetaS;
	      DthetaW_DpsiC = 0.0;
	      KWr           = 1.0;
	      DKWr_DpsiC     = 0.0;
	    }
          /*slight compressibility*/
          rhom = rho*exp(beta*u[k]);
          drhom = beta*rhom;

          mass[k] = rhom*thetaW;
          dmass[k] = -rhom*DthetaW_DpsiC+drhom*thetaW;
	  /*store point values of kr in a and da*/
	  a[i*nPointsPerSimplex*nSpace2 + j*nSpace2 + 0*nSpace + 0] = KWr;
	  da[i*nPointsPerSimplex*nSpace2 + j*nSpace2+ 0*nSpace + 0]= -DKWr_DpsiC;
	  if (computeAverages)/*accumulate averages*/
	    {
	      a_avg[i*nSpace2 + 0*nSpace + 0] += KWr*dV[k]; 
	      da_avg[i*nSpace2+ 0*nSpace + 0] += -DKWr_DpsiC*dV[k];
	    }
	}/*j*/
      /*finish average calculations*/
      if (computeAverages)/*accumulate averages*/
	{
	  a_avg[i*nSpace2 + 0*nSpace + 0] /= vol; 
	  da_avg[i*nSpace2 + 0*nSpace + 0] /= vol;
	}
    }/*i*/


  for (i=0; i < nSimplex; i++)
    {
      matID= materialTypes[i];
      for (j=0;j<nPointsPerSimplex;j++)
	{
	  k = i*nPointsPerSimplex + j;
	  ebN_global = quadraturePointToElementBoundary[k];
	  /*first grab harmonic average for conductivity*/
	  KWs_left = KWs[matID]; KWs_right = KWs_left;
	  if (elementBoundaryElementsArray[ebN_global*2 + 1] >= 0)
	    KWs_right = KWs[materialTypes[elementBoundaryElementsArray[ebN_global*2 + 1]]];
	  KWs_harm = 2.0*KWs_left*KWs_right/(KWs_left + KWs_right + 1.0e-24);

	  drive = 0.0;
	  for (I=0; I < nSpace; I++)
	    {
	      drive += (rho2*KWs_harm*gravity[I]-rho*KWs_harm*gradu[k*nSpace+I])*n_global[ebN_global*nQuadraturePoints_elementBoundary*nSpace + 
											  0*nSpace + I];
	    }
	  if (drive >= 0.0 || elementBoundaryElementsArray[ebN_global*2 + 1] < 0)
	    eN_upwind = elementBoundaryElementsArray[ebN_global*2 + 0];
	  else
	    eN_upwind = elementBoundaryElementsArray[ebN_global*2 + 1];
	  /*mwf debug
	    printf("RE V2 upwind nSimplex= %d nPerSimplex= %d i=%d j=%d k=%d ebN_global= %d eN_upwind= %d\n",nSimplex,nPointsPerSimplex,i,j,k,ebN_global,
	    eN_upwind);
	      */
	  /*start with point val for kr*/
	  krw_val = a[i*nPointsPerSimplex*nSpace2 + j*nSpace2 + 0*nSpace  +0];
	  dkrw_val= da[i*nPointsPerSimplex*nSpace2 + j*nSpace2+ 0*nSpace  +0];
	  if (upwindFlag > 0) /*don't have a choice for point upwinding*/
	    {
	      krw_val = a_avg[eN_upwind*nSpace2 + 0*nSpace + 0];
	      /*use point value in jacobian regardless*/
	    }


	  for (I = 0; I < nSpace; I++)
	    {
	      f[k*nSpace + I] = rho2*krw_val*KWs_harm*gravity[I];
	      df[k*nSpace+ I] = rho2*dkrw_val*KWs_harm*gravity[I];
	      a[i*nPointsPerSimplex*nSpace2 + j*nSpace2+I*nSpace+I] = rho*krw_val*KWs_harm;
	      da[i*nPointsPerSimplex*nSpace2 + j*nSpace2+I*nSpace+I]= rho*dkrw_val*KWs_harm;
	    }
	
	}/*point loop for simplex*/
    }/*simplex loop for upwinding*/
  
}

void conservativeHeadRichardsMualemVanGenuchtenHetEvaluateV2withUpwindAndHarm_sd(const int upwindFlag,
										 const int computeAverages,
										 const int nSimplex,
										 const int nPointsPerSimplex,
										 const int nSpace,
										 const int nQuadraturePoints_elementBoundary,
										 const proteus_csr_index_t *rowptr,
										 const proteus_csr_index_t *colind,
										 const proteus_index_t* elementBoundaryElementsArray,
										 const int* quadraturePointToElementBoundary,
										 const int* materialTypes,
										 const double rho,
										 const double beta,
										 const double* gravity,
										 const double* alpha,
										 const double* n_vg,
										 const double* thetaR,
										 const double* thetaSR,
										 const double* KWs,
										 const double *u,
										 const double *gradu,
										 const double *n_global,
										 const double *dV,
										 double *mass,
										 double *dmass,
										 double *f_avg,
										 double *df_avg,
										 double *a_avg,
										 double *da_avg,
										 double *f,
										 double *df,
										 double *a,
										 double *da)
{
  proteus_csr_index_t J; proteus_csr_index_t nnz; int i, j, k, I, matID;
  const int nSpace2=nSpace*nSpace;
  register double psiC,
    pcBar,pcBar_n,pcBar_nM1,pcBar_nM2,
    onePlus_pcBar_n,
    sBar,sqrt_sBar,DsBar_DpsiC,
    thetaW,DthetaW_DpsiC,
    vBar,vBar2,DvBar_DpsiC,
    KWr,DKWr_DpsiC,
    rho2=rho*rho,
    thetaS,
    rhom,drhom,m;
  proteus_index_t eN_upwind; int ebN_global;
  register double drive,KWs_harm,KWs_left,KWs_right,krw_val,dkrw_val,vol;
  nnz = rowptr[nSpace]; /*sparse mat rep for diffusion*/ 
  /*
    loop through and compute point values for m as usual
       store point values for kr, dkr and
       calculate average value for kr, dkr

    loop again through each point
       find face for that point
       compute harmonic average of (rho*Ks)

       if upwinding,
           find upwind direction for that face
           set point value to 
           a_{ij,f} = (rho*Ks)_{ij,h,f}k_r^{up}
           f_{i}    = (rho*rho*Ks)_{h,f}k_r^{up}\vec g
       else
           set point value to 
           a_{ij,f} = (rho*Ks)_{ij,h,f}k_r(\psi_f)
           f_{i}    = (rho*rho*Ks)_{h,f}k_r(\psi_f)\vec g
     
   */

  if (upwindFlag > 0)
    assert(computeAverages);
  for (i=0; i < nSimplex; i++)
    {
      matID= materialTypes[i];
      /*for computing averages*/
      vol = 0.0;
      for (j=0; j < nPointsPerSimplex; j++)
	vol += dV[i*nPointsPerSimplex + j];
      /*store average information with a_avg first entry for now*/
      if (computeAverages)
	{
	  a_avg[i*nSpace2+0*nSpace+0] = 0.0;
	  da_avg[i*nSpace2+0*nSpace+0] = 0.0;
	}
      for (j=0;j<nPointsPerSimplex;j++)
	{
	  k = i*nPointsPerSimplex + j;
	  psiC = -u[k];
	  m = 1.0 - 1.0/n_vg[matID];
	  thetaS = thetaR[matID] + thetaSR[matID];
	  if (psiC > 0.0)
	    {
	      pcBar = alpha[matID]*psiC;
	      pcBar_nM2 = pow(pcBar,n_vg[matID]-2);
	      pcBar_nM1 = pcBar_nM2*pcBar;
	      pcBar_n   = pcBar_nM1*pcBar;
	      onePlus_pcBar_n = 1.0 + pcBar_n;
	      
	      sBar = pow(onePlus_pcBar_n,-m);
	      /* using -mn = 1-n */
	      DsBar_DpsiC = alpha[matID]*(1.0-n_vg[matID])*(sBar/onePlus_pcBar_n)*pcBar_nM1;
	      
	      vBar = 1.0-pcBar_nM1*sBar;
	      vBar2 = vBar*vBar;
	      DvBar_DpsiC = -alpha[matID]*(n_vg[matID]-1.0)*pcBar_nM2*sBar - pcBar_nM1*DsBar_DpsiC;
          
	      thetaW = thetaSR[matID]*sBar + thetaR[matID];
	      DthetaW_DpsiC = thetaSR[matID] * DsBar_DpsiC;
          
	      sqrt_sBar = sqrt(sBar);
	      KWr= sqrt_sBar*vBar2;
	      DKWr_DpsiC= 
		((0.5/sqrt_sBar)*DsBar_DpsiC*vBar2
		 +
		 2.0*sqrt_sBar*vBar*DvBar_DpsiC);
	    }
	  else
	    {
	      thetaW        = thetaS;
	      DthetaW_DpsiC = 0.0;
	      KWr           = 1.0;
	      DKWr_DpsiC     = 0.0;
	    }
          /*slight compressibility*/
          rhom = rho*exp(beta*u[k]);
          drhom = beta*rhom;

          mass[k] = rhom*thetaW;
          dmass[k] = -rhom*DthetaW_DpsiC+drhom*thetaW;
	  /*store point values of kr in a and da using first entry*/
	  J = rowptr[0];
	  a[i*nPointsPerSimplex*nnz + j*nnz + J] = KWr;
	  da[i*nPointsPerSimplex*nnz + j*nnz + J]= -DKWr_DpsiC;
	  if (computeAverages)/*accumulate averages*/
	    {
	      a_avg[i*nnz + J] += KWr*dV[k]; 
	      da_avg[i*nnz+ J] += -DKWr_DpsiC*dV[k];
	    }
	}/*j*/
      /*finish average calculations*/
      if (computeAverages)/*accumulate averages*/
	{
	  a_avg[i*nnz + J] /= vol; 
	  da_avg[i*nnz + J] /= vol;
	}
    }/*i*/


  for (i=0; i < nSimplex; i++)
    {
      matID= materialTypes[i];
      for (j=0;j<nPointsPerSimplex;j++)
	{
	  k = i*nPointsPerSimplex + j;
	  ebN_global = quadraturePointToElementBoundary[k];
	  /*first grab harmonic average for conductivity*/
	  KWs_left = KWs[matID]; KWs_right = KWs_left;
	  if (elementBoundaryElementsArray[ebN_global*2 + 1] >= 0)
	    KWs_right = KWs[materialTypes[elementBoundaryElementsArray[ebN_global*2 + 1]]];
	  KWs_harm = 2.0*KWs_left*KWs_right/(KWs_left + KWs_right + 1.0e-24);

	  drive = 0.0;
	  for (I=0; I < nSpace; I++)
	    {
	      drive += (rho2*KWs_harm*gravity[I]-rho*KWs_harm*gradu[k*nSpace+I])*n_global[ebN_global*nQuadraturePoints_elementBoundary*nSpace + 
											  0*nSpace + I];
	    }
	  if (drive >= 0.0 || elementBoundaryElementsArray[ebN_global*2 + 1] < 0)
	    eN_upwind = elementBoundaryElementsArray[ebN_global*2 + 0];
	  else
	    eN_upwind = elementBoundaryElementsArray[ebN_global*2 + 1];
	  /*mwf debug
	    printf("RE V2 upwind nSimplex= %d nPerSimplex= %d i=%d j=%d k=%d ebN_global= %d eN_upwind= %d\n",nSimplex,nPointsPerSimplex,i,j,k,ebN_global,
	    eN_upwind);
	      */
	  /*start with point val for kr*/
	  J = rowptr[0]; /*just grab first entry because assuming diagonal for now*/
	  krw_val = a[i*nPointsPerSimplex*nnz + j*nnz + J];
	  dkrw_val= da[i*nPointsPerSimplex*nnz + j*nnz + J];
	  if (upwindFlag > 0) /*don't have a choice for point upwinding*/
	    {
	      krw_val = a_avg[eN_upwind*nnz + J];
/* 	      if (i != eN_upwind) */
/* 		dkrw_val = 0.0; */
/* 	      else */
/* 		dkrw_val = da_avg[eN_upwind*nnz + J]; */
	    }


	  for (I = 0; I < nSpace; I++)
	    {
	      f[k*nSpace + I] = rho2*krw_val*KWs_harm*gravity[I];
	      df[k*nSpace+ I] = rho2*dkrw_val*KWs_harm*gravity[I];
	      for (J=rowptr[I]; J < rowptr[I+1]; J++)
		{
		  if (colind[J] == I)
		    {
		      a[i*nPointsPerSimplex*nnz + j*nnz + J] = rho*krw_val*KWs_harm;
		      da[i*nPointsPerSimplex*nnz + j*nnz + J]= rho*dkrw_val*KWs_harm;
		    }
		  else
		    {
		      a[i*nPointsPerSimplex*nnz + j*nnz + J] = 0.0;
		      da[i*nPointsPerSimplex*nnz + j*nnz + J]= 0.0;
		    }
		}
	    }
	  
	}/*point loop for simplex*/
    }/*simplex loop for upwinding*/
  
}

void applyContactLineSlip(proteus_index_t nExteriorElementBoundaries_global,
                          int nQuadraturePoints_elementBoundary,
                          double eps,
                          int* isDOFBoundary,
                          double* phi,
                          double* advectiveFlux,
                          double* diffusiveFlux)
{
  int ebNE,k;
  double weight;
  for(ebNE=0;ebNE<nExteriorElementBoundaries_global;ebNE++)
    {
      for(k=0;k<nQuadraturePoints_elementBoundary;k++)
        {
          if(isDOFBoundary[ebNE*nQuadraturePoints_elementBoundary+k] == 1)
            {
              weight = (smoothedDirac(eps,0) - smoothedDirac(eps,phi[ebNE*nQuadraturePoints_elementBoundary+k]))/smoothedDirac(eps,0); 
              advectiveFlux[ebNE*nQuadraturePoints_elementBoundary+k] *= weight;
              diffusiveFlux[ebNE*nQuadraturePoints_elementBoundary+k] *= weight;
            }
        }
    }
}

void applyContactLineSlipJacobian(proteus_index_t nExteriorElementBoundaries_global,
                                  int nQuadraturePoints_elementBoundary,
                                  int nDOF_trial_element,
                                  double eps,
                                  int* isDOFBoundary,
                                  double* phi,
                                  double* fluxJacobian)
{
  int ebNE,k,j;
  double weight;
  for(ebNE=0;ebNE<nExteriorElementBoundaries_global;ebNE++)
    {
      for(k=0;k<nQuadraturePoints_elementBoundary;k++)
        {
          if(isDOFBoundary[ebNE*nQuadraturePoints_elementBoundary+k] == 1)
            {
              weight = (smoothedDirac(eps,0) - smoothedDirac(eps,phi[ebNE*nQuadraturePoints_elementBoundary+k]))/smoothedDirac(eps,0);
              for(j=0;j<nDOF_trial_element;j++)
                {
                  fluxJacobian[ebNE*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                               k*nDOF_trial_element+
                               j] *= weight;
                }
            }
        }
    }
}
/** @} */


void diffusiveWave1DEvaluate(const int nPoints,
			     const double alpha,
			     const double gamma,
			     const double epsilon,
			     const double* x,
			     const double* u,
			     const double* grad_u,
			     double* m,
			     double* dm,
			     double* a,
			     double* da)
{
  int i;
  double depth=0.0;
  double hold=0.0;

  for (i=0; i<nPoints; i++)
    {
      m[i]=u[i];
      dm[i]=1.0;
      hold=u[i]-x[i*3+2];
      depth=fmax(hold, 0.0);
      hold=fabs(grad_u[i]);
      a[i]= (pow(depth,alpha))/(pow(hold, 1.0-gamma)+epsilon);
      da[i]=(alpha*pow(depth,alpha-1.0))/(pow(hold, 1.0-gamma)+epsilon);

    }
}

void diffusiveWave2DEvaluate(const int nd,
			     const int nPoints,
			     const double alpha,
			     const double gamma,
			     const double epsilon,
			     const double* x,
			     const double* u,
			     const double* grad_u,
			     double* m,
			     double* dm,
			     double* a,
			     double* da)
{
  int i;
  double depth=0.0;
  double hold=0.0;

  for (i=0; i<nPoints; i++)
    {
      m[i]=u[i];
      dm[i]=1.0;
      depth=fmax(u[i]-x[i*3+2],0.0);
      hold=sqrt(grad_u[2*i]*grad_u[2*i]+grad_u[2*i+1]*grad_u[2*i+1]);
      a[i*4]=a[i*4+3]= (pow(depth,alpha))/(pow(hold, 1.0-gamma)+epsilon);
      da[i*4]=da[i*4+3]=(alpha*pow(depth,alpha-1.0))/(pow(hold, 1.0-gamma)+epsilon);

    }
}

void calculateEddyViscosity_Smagorinsky_2D(const proteus_index_t nElements_global,
					   const int nQuadraturePoints_element,
					   const double smagorinskyConstant,
					   const double * h_e,
					   const double * grad_u,
					   const double * grad_v,
					   double * nu_t)
{
  int eN,k;
  const int nSpace = 2;
  double norm_S2,norm_S;
  for (eN = 0; eN < nElements_global; eN++)
    {
      for (k = 0; k < nQuadraturePoints_element; k++)
	{
	  norm_S2 = 
	    grad_u[eN*nQuadraturePoints_element*nSpace + 
		   k*nSpace + 0]
	    *
	    grad_u[eN*nQuadraturePoints_element*nSpace + 
		   k*nSpace + 0]
	    +
	    grad_v[eN*nQuadraturePoints_element*nSpace + 
		   k*nSpace + 1]
	    *
	    grad_v[eN*nQuadraturePoints_element*nSpace + 
		   k*nSpace + 1]
	    +
	    0.5*
	    (grad_u[eN*nQuadraturePoints_element*nSpace + 
		    k*nSpace + 1]
	     +
	     grad_v[eN*nQuadraturePoints_element*nSpace + 
		    k*nSpace + 0]) 
	    *
	    (grad_u[eN*nQuadraturePoints_element*nSpace + 
		    k*nSpace + 1]
	     +
	     grad_v[eN*nQuadraturePoints_element*nSpace + 
		    k*nSpace + 0]);
	  norm_S = sqrt(2.0*norm_S2);
	  nu_t[eN*nQuadraturePoints_element + k] = 
	    smagorinskyConstant*smagorinskyConstant*h_e[eN]*h_e[eN]*norm_S;

	}
    }

}
void calculateEddyViscosity_Smagorinsky_3D(const proteus_index_t nElements_global,
					   const int nQuadraturePoints_element,
					   const double smagorinskyConstant,
					   const double * h_e,
					   const double * grad_u,
					   const double * grad_v,
					   const double * grad_w,
					   double * nu_t)
{
  int eN,k;
  const int nSpace = 3;
  double norm_S2,norm_S;
  for (eN = 0; eN < nElements_global; eN++)
    {
      for (k = 0; k < nQuadraturePoints_element; k++)
	{
	  norm_S2 = 
	    grad_u[eN*nQuadraturePoints_element*nSpace + 
		   k*nSpace + 0]
	    *
	    grad_u[eN*nQuadraturePoints_element*nSpace + 
		   k*nSpace + 0]
	    +
	    grad_v[eN*nQuadraturePoints_element*nSpace + 
		   k*nSpace + 1]
	    *
	    grad_v[eN*nQuadraturePoints_element*nSpace + 
		   k*nSpace + 1]
	    +
	    grad_w[eN*nQuadraturePoints_element*nSpace + 
		   k*nSpace + 2]
	    *
	    grad_w[eN*nQuadraturePoints_element*nSpace + 
		   k*nSpace + 2]
	    +
	    0.5*
	    (grad_u[eN*nQuadraturePoints_element*nSpace + 
		    k*nSpace + 1]
	     +
	     grad_v[eN*nQuadraturePoints_element*nSpace + 
		    k*nSpace + 0]) 
	    *
	    (grad_u[eN*nQuadraturePoints_element*nSpace + 
		    k*nSpace + 1]
	     +
	     grad_v[eN*nQuadraturePoints_element*nSpace + 
		    k*nSpace + 0])
	    +
	    0.5*
	    (grad_u[eN*nQuadraturePoints_element*nSpace + 
		    k*nSpace + 2]
	     +
	     grad_w[eN*nQuadraturePoints_element*nSpace + 
		    k*nSpace + 0]) 
	    *
	    (grad_u[eN*nQuadraturePoints_element*nSpace + 
		    k*nSpace + 2]
	     +
	     grad_w[eN*nQuadraturePoints_element*nSpace + 
		    k*nSpace + 0])
	    +
	    0.5*
	    (grad_v[eN*nQuadraturePoints_element*nSpace + 
		    k*nSpace + 2]
	     +
	     grad_w[eN*nQuadraturePoints_element*nSpace + 
		    k*nSpace + 1]) 
	    *
	    (grad_v[eN*nQuadraturePoints_element*nSpace + 
		    k*nSpace + 2]
	     +
	     grad_w[eN*nQuadraturePoints_element*nSpace + 
		    k*nSpace + 1]);
	  norm_S = sqrt(2.0*norm_S2);
	  nu_t[eN*nQuadraturePoints_element + k] = 
	    smagorinskyConstant*smagorinskyConstant*h_e[eN]*h_e[eN]*norm_S;

	}
    }

}
void calculateEddyViscosity_Smagorinsky2P_2D(const proteus_index_t nElements_global,
					     const int nQuadraturePoints_element,
					     const double smagorinskyConstant_0,
					     const double smagorinskyConstant_1,
					     const double  eps,
					     const double * phi_ls,
					     const double * h_e,
					     const double * grad_u,
					     const double * grad_v,
					     double * nu_t)
{
  int eN,k;
  const int nSpace = 2;
  double norm_S2,norm_S,H_smc,smc;
  for (eN = 0; eN < nElements_global; eN++)
    {
      for (k = 0; k < nQuadraturePoints_element; k++)
	{
	  H_smc = smoothedHeaviside(eps,phi_ls[k]);
	  smc   = smagorinskyConstant_0*(1.0-H_smc) + smagorinskyConstant_1*H_smc;

	  norm_S2 = 
	    grad_u[eN*nQuadraturePoints_element*nSpace + 
		   k*nSpace + 0]
	    *
	    grad_u[eN*nQuadraturePoints_element*nSpace + 
		   k*nSpace + 0]
	    +
	    grad_v[eN*nQuadraturePoints_element*nSpace + 
		   k*nSpace + 1]
	    *
	    grad_v[eN*nQuadraturePoints_element*nSpace + 
		   k*nSpace + 1]
	    +
	    0.5*
	    (grad_u[eN*nQuadraturePoints_element*nSpace + 
		    k*nSpace + 1]
	     +
	     grad_v[eN*nQuadraturePoints_element*nSpace + 
		    k*nSpace + 0]) 
	    *
	    (grad_u[eN*nQuadraturePoints_element*nSpace + 
		    k*nSpace + 1]
	     +
	     grad_v[eN*nQuadraturePoints_element*nSpace + 
		    k*nSpace + 0]);
	  norm_S = sqrt(2.0*norm_S2);
	  nu_t[eN*nQuadraturePoints_element + k] = 
	    smc*smc*h_e[eN]*h_e[eN]*norm_S;

	}
    }

}
void calculateEddyViscosity_Smagorinsky2P_3D(const proteus_index_t nElements_global,
					     const int nQuadraturePoints_element,
					     const double smagorinskyConstant_0,
					     const double smagorinskyConstant_1,
					     const double eps,
					     const double * phi_ls,
					     const double * h_e,
					     const double * grad_u,
					     const double * grad_v,
					     const double * grad_w,
					     double * nu_t)
{
  int eN,k;
  const int nSpace = 3;
  double norm_S2,norm_S,H_smc,smc;
  for (eN = 0; eN < nElements_global; eN++)
    {
      for (k = 0; k < nQuadraturePoints_element; k++)
	{
	  H_smc = smoothedHeaviside(eps,phi_ls[k]);
	  smc   = smagorinskyConstant_0*(1.0-H_smc) + smagorinskyConstant_1*H_smc;

	  norm_S2 = 
	    grad_u[eN*nQuadraturePoints_element*nSpace + 
		   k*nSpace + 0]
	    *
	    grad_u[eN*nQuadraturePoints_element*nSpace + 
		   k*nSpace + 0]
	    +
	    grad_v[eN*nQuadraturePoints_element*nSpace + 
		   k*nSpace + 1]
	    *
	    grad_v[eN*nQuadraturePoints_element*nSpace + 
		   k*nSpace + 1]
	    +
	    grad_w[eN*nQuadraturePoints_element*nSpace + 
		   k*nSpace + 2]
	    *
	    grad_w[eN*nQuadraturePoints_element*nSpace + 
		   k*nSpace + 2]
	    +
	    0.5*
	    (grad_u[eN*nQuadraturePoints_element*nSpace + 
		    k*nSpace + 1]
	     +
	     grad_v[eN*nQuadraturePoints_element*nSpace + 
		    k*nSpace + 0]) 
	    *
	    (grad_u[eN*nQuadraturePoints_element*nSpace + 
		    k*nSpace + 1]
	     +
	     grad_v[eN*nQuadraturePoints_element*nSpace + 
		    k*nSpace + 0])
	    +
	    0.5*
	    (grad_u[eN*nQuadraturePoints_element*nSpace + 
		    k*nSpace + 2]
	     +
	     grad_w[eN*nQuadraturePoints_element*nSpace + 
		    k*nSpace + 0]) 
	    *
	    (grad_u[eN*nQuadraturePoints_element*nSpace + 
		    k*nSpace + 2]
	     +
	     grad_w[eN*nQuadraturePoints_element*nSpace + 
		    k*nSpace + 0])
	    +
	    0.5*
	    (grad_v[eN*nQuadraturePoints_element*nSpace + 
		    k*nSpace + 2]
	     +
	     grad_w[eN*nQuadraturePoints_element*nSpace + 
		    k*nSpace + 1]) 
	    *
	    (grad_v[eN*nQuadraturePoints_element*nSpace + 
		    k*nSpace + 2]
	     +
	     grad_w[eN*nQuadraturePoints_element*nSpace + 
		    k*nSpace + 1]);
	  norm_S = sqrt(2.0*norm_S2);
	  nu_t[eN*nQuadraturePoints_element + k] = 
	    smc*smc*h_e[eN]*h_e[eN]*norm_S;

	}
    }

}

void eddyViscosity_2D_Update(const int nPoints,
			     const double* nu_t,
			     double *mom_u_diff_ten,
			     double *mom_v_diff_ten,
			     double *mom_uv_diff_ten,
			     double *mom_vu_diff_ten)
{
  int i;
  for (i=0;i<nPoints;i++)
    {
#ifdef SCALAR_DIFFUSION
     //u momentum diffusion tensor
      mom_u_diff_ten[i*4+0] += nu_t[i];
      mom_u_diff_ten[i*4+3] += nu_t[i];

      //v momentum diffusion tensor
      mom_v_diff_ten[i*4+0] += nu_t[i];
      mom_v_diff_ten[i*4+3] += nu_t[i];
#else
      //u momentum diffusion tensor
      mom_u_diff_ten[i*4+0] += 2.0*nu_t[i];
      mom_u_diff_ten[i*4+3] += nu_t[i];
      mom_uv_diff_ten[i*4+2]+= nu_t[i];

      //v momentum diffusion tensor
      mom_v_diff_ten[i*4+0] += nu_t[i];
      mom_v_diff_ten[i*4+3] += 2.0*nu_t[i];
      mom_vu_diff_ten[i*4+1] += nu_t[i];
#endif


    }
}
void eddyViscosity_2D_Update_sd(const int nPoints,
				const double* nu_t,
				double *mom_u_diff_ten,
				double *mom_v_diff_ten,
				double *mom_uv_diff_ten,
				double *mom_vu_diff_ten)
{
  int i;
  for (i=0;i<nPoints;i++)
    {
      //u momentum diffusion tensor
      mom_u_diff_ten[i*2+0] += 2.0*nu_t[i];
      mom_u_diff_ten[i*2+1] += nu_t[i];
      mom_uv_diff_ten[i]+= nu_t[i];

      //v momentum diffusion tensor
      mom_v_diff_ten[i*2+0] += nu_t[i];
      mom_v_diff_ten[i*2+1] += 2.0*nu_t[i];
      mom_vu_diff_ten[i] += nu_t[i];

    }
}
void eddyViscosity_3D_Update(const int nPoints,
			     const double* nu_t,
			     double *mom_u_diff_ten,
			     double *mom_v_diff_ten,
			     double *mom_w_diff_ten,
			     double *mom_uv_diff_ten,
			     double *mom_uw_diff_ten,
			     double *mom_vu_diff_ten,
			     double *mom_vw_diff_ten,
			     double *mom_wu_diff_ten,
			     double *mom_wv_diff_ten)
{
  int k;
  for (k=0;k<nPoints;k++)
    {
      //u momentum diffusion tensor
      mom_u_diff_ten[k*9+0] = 2.0*nu_t[k];
      mom_u_diff_ten[k*9+4] = nu_t[k];
      mom_u_diff_ten[k*9+8] = nu_t[k];

      mom_uv_diff_ten[k*9+3]=nu_t[k];

      mom_uw_diff_ten[k*9+6]=nu_t[k];

      //v momentum diffusion tensor
      mom_v_diff_ten[k*9+0] = nu_t[k];
      mom_v_diff_ten[k*9+4] = 2.0*nu_t[k];
      mom_v_diff_ten[k*9+8] = nu_t[k];

      mom_vu_diff_ten[k*9+1]=nu_t[k];

      mom_vw_diff_ten[k*9+7]=nu_t[k];

      //w momentum diffusion tensor
      mom_w_diff_ten[k*9+0] = nu_t[k];
      mom_w_diff_ten[k*9+4] = nu_t[k];
      mom_w_diff_ten[k*9+8] = 2.0*nu_t[k];

      mom_wu_diff_ten[k*9+2]=nu_t[k];

      mom_wv_diff_ten[k*9+5]=nu_t[k];


    }
}
void eddyViscosity_3D_Update_sd(const int nPoints,
				const double* nu_t,
				double *mom_u_diff_ten,
				double *mom_v_diff_ten,
				double *mom_w_diff_ten,
				double *mom_uv_diff_ten,
				double *mom_uw_diff_ten,
				double *mom_vu_diff_ten,
				double *mom_vw_diff_ten,
				double *mom_wu_diff_ten,
				double *mom_wv_diff_ten)
{
  int k;
  for (k=0;k<nPoints;k++)
    {
      //u momentum diffusion tensor
      mom_u_diff_ten[k*3+0] += 2.0*nu_t[k];
      mom_u_diff_ten[k*3+1] += nu_t[k];
      mom_u_diff_ten[k*3+2] += nu_t[k];

      mom_uv_diff_ten[k]+=nu_t[k];

      mom_uw_diff_ten[k]+=nu_t[k];

      //v momentum diffusion tensor
      mom_v_diff_ten[k*3+0] += nu_t[k];
      mom_v_diff_ten[k*3+1] += 2.0*nu_t[k];
      mom_v_diff_ten[k*3+2] += nu_t[k];

      mom_vu_diff_ten[k]+=nu_t[k];

      mom_vw_diff_ten[k]+=nu_t[k];

      //w momentum diffusion tensor
      mom_w_diff_ten[k*3+0] += nu_t[k];
      mom_w_diff_ten[k*3+1] += nu_t[k];
      mom_w_diff_ten[k*3+2] += 2.0*nu_t[k];

      mom_wu_diff_ten[k]+=nu_t[k];

      mom_wv_diff_ten[k]+=nu_t[k];

    }
}

void calculateWaveFunction3d_ref(//mesh rep 
				 int nElements_global,
				 int nDOF_element_mesh,
				 int nQuadraturePoints_element,
				 const double* mesh_trial_ref,
				 const double* mesh_dof,
				 const int* mesh_l2g,
				 const double* elementDiametersArray,
				 //
				 const double* omega_s_x, //source region (rectangular)
				 const double* omega_s_y,
				 const double* omega_s_z,
				 double t,
				 int waveFlag, //1 secondOrderStokes
				               //2 solitaryWave
                                               //0 monochromaticWave
				 double epsFact,
				 double waveHeight,
				 double waveCelerity,
				 double waveFrequency,
				 double waveNumber,
				 double waterDepth,
				 double* source)
{
  int eN,k,j,eN_j;
  double x,y,z,factor,dx_source,dy_source,dz_source,source_volume,N_j,
    distance_x,distance_y,distance_z,delta,eps;
  /*stokes wave parameters*/
  double p_s,a_s,b_s,kd,term1,term2,term3,sinhkd;
  dx_source=omega_s_x[1]-omega_s_x[0]; dy_source=omega_s_y[1]-omega_s_y[0]; dz_source=omega_s_z[1]-omega_s_z[0];
  source_volume = dx_source*dy_source*dz_source;
  if (waveFlag == 1)
    {
      kd  = waveNumber*waterDepth;
      a_s = waveHeight*0.5;
      sinhkd = sinh(kd);
      b_s = waveHeight*waveHeight*waveNumber*cosh(kd)*(2.0 + cosh(2.*kd)/(16.0+sinhkd*sinhkd*sinhkd));
      term1 = -a_s + sqrt(a_s*a_s + 8.0*b_s*b_s)/(4.0*b_s);
      p_s = asin(term1);
      term1 = waveCelerity*waveHeight*cos(M_PI*0.5 - waveFrequency*t - p_s);
      term2 = waveCelerity*waveHeight*waveHeight*cosh(kd)/(8.0*sinhkd*sinhkd*sinhkd);
      term3 = 2.0 + cosh(2.0*kd)*cos(2.0*(M_PI*0.5 - waveFrequency*t - p_s));
      factor = (term1 + term2*term3)/source_volume;
      
    }
  else if (waveFlag == 2)
    {
      term1 = 4.0*waveHeight/sqrt(waveHeight/waterDepth);/*x_s*/
      term2= sqrt(3.0*waveHeight/(4.0*waterDepth*waterDepth*waterDepth))*(term1 - waveCelerity*t);
      term2= fmax(term2,-80.0);
      term2= fmin(term2, 80.0);
      term3= 1.0/(cosh(term2)+1.0e-12);
      factor = waveHeight*waveCelerity*term3*term3/source_volume;
    }
  else
    {
      factor = waveHeight/source_volume*waveCelerity*sin(waveFrequency*t);
    }
  for (eN = 0; eN < nElements_global; eN++)
    {
      eps = epsFact*elementDiametersArray[eN];
      for (k = 0; k < nQuadraturePoints_element; k++)
	{
	  x=0.; y=0.; z=0.;
	  for (j=0; j < nDOF_element_mesh; j++)
	    {
	      eN_j= eN*nDOF_element_mesh+j;
	      N_j = mesh_trial_ref[k*nDOF_element_mesh+j];

	      x += mesh_dof[mesh_l2g[eN_j]*3+0]*N_j;
	      y += mesh_dof[mesh_l2g[eN_j]*3+1]*N_j;
	      z += mesh_dof[mesh_l2g[eN_j]*3+2]*N_j;

	    }
	  distance_x = fabs(x-0.5*(omega_s_x[1]+omega_s_x[0])) - 0.5*dx_source;
	  distance_y = fabs(y-0.5*(omega_s_y[1]+omega_s_y[0])) - 0.5*dy_source;
	  distance_z = fabs(z-0.5*(omega_s_z[1]+omega_s_z[0])) - 0.5*dz_source;
	  delta = (1.0-smoothedHeaviside(eps,distance_x))*(1.0-smoothedHeaviside(eps,distance_y))*(1.0-smoothedHeaviside(eps,distance_z));
	  source[eN*nQuadraturePoints_element+k] = -factor*delta;
	  
	}
    }

}
/* /\*simple piecewise linear interpolation from a table */
/*   assumes xv are increasing */
/*  *\/ */
/* int findInterval(const double* vertices, int nv, double x, int* ival, double tol) */
/* { */
/*   int leftInt=0,rightInt=nv-2,failed=1,mid=0; */
/*   assert(rightInt >= leftInt); */
/*   /\*take care of easy cases first*\/ */
/*   if (fabs(x-vertices[leftInt]) < tol) */
/*     { */
/*       *ival=leftInt; */
/*       failed=0; */
/*       return failed; */
/*     } */
/*   if (x <= vertices[leftInt]-tol) */
/*     { */
/*       *ival=-1; */
/*       failed=1; */
/*       return failed; */
/*     } */
/*   if (fabs(x-vertices[rightInt+1]) < tol) */
/*     { */
/*       *ival=rightInt; */
/*       failed=0; */
/*       return failed; */
/*     } */
/*   if (x >= vertices[rightInt+1]+tol) */
/*     { */
/*       *ival = nv; */
/*       failed=1; */
/*       return failed; */
/*     } */
/*   /\*otherwise, should have x in (left,right)*\/ */
/*   while (leftInt <= rightInt) */
/*     { */
/*       mid = (int)(floor(0.5*(leftInt+rightInt))); */
/*       if (vertices[mid] <= x && x < vertices[mid+1])/\*x in interval mid*\/ */
/* 	{ */
/* 	  *ival = mid; */
/* 	  failed = 0; */
/* 	  return failed; */
/* 	} */
/*       else if (x < vertices[mid])/\*x to the left of mid*\/ */
/* 	rightInt = mid-1; */
/*       else if (x >= vertices[mid+1]) /\*x to the right of mid*\/ */
/* 	leftInt = mid+1; */
/*       else */
/* 	{ */
/* 	  printf("findInterval shouldn't be here leftInt=%d rightInt=%d \n",leftInt,rightInt); */
/* 	  assert(0); */
/* 	  failed = 1; */
/* 	  return failed; */
/* 	} */
/*     } */
/*   failed = 1; */
/*   return failed; */
/* }  */
/* void piecewiseLinearTableLookup(double x, */
/* 				int nv, */
/* 				int* start, */
/* 				double* y, */
/* 				double* dy, */
/* 				const double* xv, */
/* 				const double* yv) */
/* { */
/*   int index=*start,findFailed=0; */
/*   double val,tol=1.0e-8; */
/*   findFailed = findInterval(xv,nv,x,&index,tol); */
/*   if (findFailed && index == -1) */
/*     { */
/*       /\*extrapolate off left, could use endpoint instead*\/ */
/*       index=0; */
/*     } */
/*   else if (findFailed && index == nv) */
/*     { */
/*       /\*extrapolate off right, could use endpoint instead*\/ */
/*       index = nv-2; */
/*     } */
/*   else */
/*     { */
/*       assert(0 <= index && index < nv-1); */
/*       assert(xv[index]-tol <= x && x<= xv[index+1]+tol); */
/*     } */
/*   assert(0 <= index && index < nv-1); */
/*   val =  yv[index] +  (yv[index+1]-yv[index])/(xv[index+1]-xv[index])*(x-xv[index]); */
/*   *y = val;  */
/*   *dy = (yv[index+1]-yv[index])/(xv[index+1]-xv[index]);  */
/*   *start = index; */
/* } */

void Mass_2D_Evaluate(const int nPoints,
		      double rho,
		      double *p,
		      double *u,
		      double *v,
		      double *mom_p_acc,
		      double *mom_u_acc,
		      double *mom_v_acc,
		      double *dmom_p_acc_p,
		      double *dmom_u_acc_u,
		      double *dmom_v_acc_v)
{
  int k;
  for (k=0; k<nPoints; k++){
    mom_p_acc[k] = p[k];
    dmom_p_acc_p[k] = 1.0;

    mom_u_acc[k] = u[k];
    dmom_u_acc_u[k] = 1.0;

    mom_v_acc[k] = v[k];
    dmom_v_acc_v[k] = 1.0;
  }
}
		
void Mass_3D_Evaluate(const int nPoints,
		      double rho,
		      double *p,
		      double *u,
		      double *v,
		      double *w,
		      double *mom_p_acc,
		      double *mom_u_acc,
		      double *mom_v_acc,
		      double *mom_w_acc,
		      double *dmom_p_acc_p,
		      double *dmom_u_acc_u,
		      double *dmom_v_acc_v,
		      double *dmom_w_acc_w)
{
  int k;
  for (k=0; k<nPoints; k++){
    mom_p_acc[k] = p[k];
    dmom_p_acc_p[k] = 1.0;

    mom_u_acc[k] = u[k];
    dmom_u_acc_u[k] = 1.0;

    mom_v_acc[k] = v[k];
    dmom_v_acc_v[k] = 1.0;

    mom_w_acc[k] = w[k];
    dmom_w_acc_w[k] = 1.0;
  }
}

void TwoPhaseInvScaledLaplace_2D_Evaluate(const int nPoints,
                                         const double eps,
                                         const double rho_0,
                                         const double nu_0,
                                         const double rho_1,
                                         const double nu_1,
                                         const double* phi,
                                         double *mom_p_diff_ten,
                                         double *mom_u_diff_ten,
                                         double *mom_v_diff_ten)
{
  int k;
  double rho,nu,mu,H;
  
  for (k=0; k<nPoints; k++)
    {
      H = smoothedHeaviside(eps,phi[k]);
      rho = rho_0*(1.0-H) + rho_1*H;
      nu = nu_0*(1.0-H) + nu_1*H;
      mu = rho_0*nu_0*(1.0-H) + rho_1*nu_1*H;
      
      mom_p_diff_ten[k*2+0] = 1.0 / rho;
      mom_p_diff_ten[k*2+1] = 1.0 / rho;

      mom_u_diff_ten[k*2+0] = 1.0 / rho;
      mom_u_diff_ten[k*2+1] = 1.0 / rho;

      mom_v_diff_ten[k*2+0] = 1.0 / rho;
      mom_v_diff_ten[k*2+1] = 1.0 / rho;
    } 
}

void TwoPhaseAdvection_2D_Evaluate(const int nPoints,
                                  const double eps,
                                  const double rho_0,
                                  const double nu_0,
                                  const double rho_1,
                                  const double nu_1,
                                  const double *phi,
                                  const double *p,
                                  const double *u,
                                  const double *v,
                                  double *mass_adv,
                                  double *dmass_adv_p,
                                  double *dmass_adv_u,
                                  double *dmass_adv_v,
                                  double *mom_u_adv,
                                  double *dmom_u_adv_u,
                                  double *dmom_u_adv_v,
                                  double *mom_v_adv,
                                  double *dmom_v_adv_u,
                                  double *dmom_v_adv_v)
{
  int k;
  double rho, nu, mu, H;
  
  for (k=0;k<nPoints;k++)
    {
      H = smoothedHeaviside(eps,phi[k]);
      rho = rho_0*(1.0-H) + rho_1*H;
      nu = nu_0*(1.0-H) + nu_1*H;
      mu = rho_0*nu_0*(1.0-H) + rho_1*nu_1*H;

      //mass advective flux
      mass_adv[k*2+0]=rho*u[k]*p[k];
      mass_adv[k*2+1]=rho*v[k]*p[k];
      
      dmass_adv_p[k*2+0] = rho*u[k];
      dmass_adv_p[k*2+1] = rho*v[k];
      // ARB - NOTE TO SELF...Why arent these derivatives p[k]?
      // Possible error to investigate.
      dmass_adv_u[k*2+0]= 0.0;
      dmass_adv_v[k*2+1]= 0.0;
      
      mom_u_adv[k*2+0] = rho*u[k]*u[k];
      mom_u_adv[k*2+1] = rho*u[k]*v[k];
      
      dmom_u_adv_u[k*2+0] = rho*2.0*u[k];
      dmom_u_adv_u[k*2+1] = rho*v[k];
      
      dmom_u_adv_v[k*2+1] = rho*u[k];
      
      mom_v_adv[k*2+0] = rho*v[k]*u[k];
      mom_v_adv[k*2+1] = rho*v[k]*v[k];
  
      dmom_v_adv_u[k*2+0] = rho*v[k];

      dmom_v_adv_v[k*2+0] = rho*u[k];
      dmom_v_adv_v[k*2+1] = rho*2.0*v[k];
    }
}
