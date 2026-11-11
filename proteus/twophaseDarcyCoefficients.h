#ifndef TWOPHASEDARCYCOEFFICIENTS_H
#define TWOPHASEDARCYCOEFFICIENTS_H
#include "index_type.h"
#include "densityRelations.h"
#include "pskRelations.h"

/* fully coupled, phase continuity formulation, 
   slight-compressibility assumption is used for densities
   heterogeneity is represented by material types
   and diffusion tensors are sparse
*/
template<class PSK, class DENSITY_W, class DENSITY_N>
inline int twophaseDarcy_fc_sd_het_matType(int nSimplex,
					   int nPointsPerSimplex,
					   int nSpace,
					   int nParams,		       
					   const proteus_csr_index_t* rowptr,
					   const proteus_csr_index_t* colind,
					   const int* materialTypes,
					   double muw,
					   double mun,
					   const double* omega,
					   const double* Kbar, /*now has to be tensor*/
					   double b,
					   const double* rwork_psk, const int* iwork_psk,
					   const double* rwork_psk_tol,
					   const double* rwork_density_w,
					   const double* rwork_density_n,
					   const double* g,
					   const double* x,
					   const double* sw,
					   const double* psiw,
					   double* mw,
					   double* dmw_dsw,
					   double* dmw_dpsiw,
					   double* mn,
					   double* dmn_dsw,
					   double* dmn_dpsiw,
					   double* psin,
					   double* dpsin_dsw,
					   double* dpsin_dpsiw,
					   double* phi_psiw,
					   double* dphi_psiw_dpsiw,
					   double* phi_psin,
					   double* dphi_psin_dpsiw,
					   double* dphi_psin_dsw,
					   double* aw,
					   double* daw_dsw,
					   double* daw_dpsiw,
					   double* an,
					   double* dan_dsw,
					   double* dan_dpsiw)
{
  int matID;
  PSK psk(rwork_psk,iwork_psk);  psk.setTolerances(rwork_psk_tol);
  DENSITY_W density_w(rwork_density_w);
  DENSITY_N density_n(rwork_density_n);
  double drhow_dsw,drhon_dsw,drhow_dpsiw,drhon_dpsiw;

  const int nnz = rowptr[nSpace];
  for (int eN=0;eN<nSimplex;eN++)
    {
      matID = materialTypes[eN];
      psk.setParams(&rwork_psk[matID*nParams]);
      for(int pN=0,i;pN<nPointsPerSimplex;pN++)
        {
          i = eN*nPointsPerSimplex+pN;
	  psk.calc(sw[i]);
	  /*non-wetting phase pressure head */
	  psin[i] = psiw[i] + psk.psic;
	  dpsin_dsw[i]  = psk.dpsic;
	  dpsin_dpsiw[i]= 1.0;

	  /*mwf need to make sure normalized by rhow and rhon respectively!*/
	  density_w.calc(psiw[i]); 
	  density_n.calc(psin[i]);
	  /*density of wetting phase just a function of psiw*/
	  drhow_dpsiw = density_w.drho;
	  drhow_dsw   = 0.0;
	  /*density of nonwetting phase a function of sw and psiw through psin*/
	  drhon_dpsiw = density_n.drho;
	  drhon_dsw   = density_n.drho*psk.dpsic;
	  
	  /* w-phase mass */
	  mw[i]   = omega[matID]*density_w.rho*sw[i];
	  dmw_dsw[i]  = omega[matID]*(density_w.rho + sw[i]*drhow_dsw); 
	  dmw_dpsiw[i]= omega[matID]*sw[i]*drhow_dpsiw;

	  /* n-phase mass */
	  mn[i]       = omega[matID]*density_n.rho*(1.0-sw[i]);
	  dmn_dsw[i]  = omega[matID]*((1.0-sw[i])*drhon_dsw -density_n.rho); 
	  dmn_dpsiw[i]= omega[matID]*(1.0-sw[i])*drhon_dpsiw;

	  /* w-phase potential */
	  phi_psiw[i] = psiw[i];
	  dphi_psiw_dpsiw[i] = 1.0; /*include density dependence below*/
      
	  /* n-phase  potential */
	  phi_psin[i] = psiw[i] + psk.psic;
	  dphi_psin_dpsiw[i]= 1.0;
	  dphi_psin_dsw[i]= psk.dpsic;

	  for (int I=0;I<nSpace;I++)
	    {
	      /* update potentials with gravity */
	      /*slight compressibility, normalized density=1.0*/
	      phi_psiw[i] -= g[I]*x[i*3+I];
	      	      
	      phi_psin[i] -= b*g[I]*x[i*3+I];
	      
	      for (int m=rowptr[I]; m < rowptr[I+1]; m++)
		{
		  /* w-phase diffusion */
	      
		  aw[i*nnz+m]  = density_w.rho*Kbar[matID*nnz+m]*psk.krw/muw;
		  daw_dsw[i*nnz+m]  = density_w.rho*Kbar[matID*nnz+m]*psk.dkrw/muw + drhow_dsw*Kbar[matID*nnz+m]*psk.krw/muw;
		  daw_dpsiw[i*nnz+m]= drhow_dpsiw*Kbar[matID*nnz+m]*psk.krw/muw;
	      
		  /* n-phase  diffusion */
		  an[i*nnz+m]        = density_n.rho*Kbar[matID*nnz+m]*psk.krn/mun;
		  dan_dsw[i*nnz+m]   = density_n.rho*Kbar[matID*nnz+m]*psk.dkrn/mun + drhon_dsw*Kbar[matID*nnz+m]*psk.krn/mun;
		  dan_dpsiw[i*nnz+m] = drhon_dpsiw*Kbar[matID*nnz+m]*psk.krn/mun;
		}
	    }
	}
    }
  return 0;
}
template<class PSK, class DENSITY_W, class DENSITY_N>
inline int twophaseDarcy_fc_sd_het_matType_nonPotentialForm(int compressibilityFlag,
							    int nSimplex,
							    int nPointsPerSimplex,
							    int nSpace,
							    int nParams,		       
							    const proteus_csr_index_t* rowptr,
							    const proteus_csr_index_t* colind,
							    const int* materialTypes,
							    double muw,
							    double mun,
							    const double* omega,
							    const double* Kbar, /*now has to be tensor*/
							    double b,
							    const double* rwork_psk, const int* iwork_psk,
							    const double* rwork_psk_tol,
							    const double* rwork_density_w,
							    const double* rwork_density_n,
							    const double* g,
							    const double* x,
							    const double* sw,
							    const double* psiw,
							    double* mw,
							    double* dmw_dsw,
							    double* dmw_dpsiw,
							    double* mn,
							    double* dmn_dsw,
							    double* dmn_dpsiw,
							    double* psin,
							    double* dpsin_dsw,
							    double* dpsin_dpsiw,
							    double* phi_psiw,
							    double* dphi_psiw_dpsiw,
							    double* phi_psin,
							    double* dphi_psin_dpsiw,
							    double* dphi_psin_dsw,
							    double* fw,
							    double* dfw_dsw,
							    double* dfw_dpsiw,
							    double* fn,
							    double* dfn_dsw,
							    double* dfn_dpsiw,
							    double* aw,
							    double* daw_dsw,
							    double* daw_dpsiw,
							    double* an,
							    double* dan_dsw,
							    double* dan_dpsiw)
{
  int matID;
  PSK psk(rwork_psk,iwork_psk);  psk.setTolerances(rwork_psk_tol);
  DENSITY_W density_w(rwork_density_w);
  DENSITY_N density_n(rwork_density_n);
  double drhow_dsw,drhon_dsw,drhow_dpsiw,drhon_dpsiw;
  double rhow_x,drhow_x_dsw,drhow_x_dpsiw,rhon_x,drhon_x_dsw,drhon_x_dpsiw;

  const int nnz = rowptr[nSpace];
  for (int eN=0;eN<nSimplex;eN++)
    {
      matID = materialTypes[eN];
      psk.setParams(&rwork_psk[matID*nParams]);
      for(int pN=0,i;pN<nPointsPerSimplex;pN++)
        {
          i = eN*nPointsPerSimplex+pN;
	  psk.calc(sw[i]);
	  /*non-wetting phase pressure head */
	  psin[i] = psiw[i] + psk.psic;
	  dpsin_dsw[i]  = psk.dpsic;
	  dpsin_dpsiw[i]= 1.0;

	  /*mwf need to make sure normalized by rhow and rhon respectively!*/
	  density_w.calc(psiw[i]); 
	  density_n.calc(psin[i]);
	  /*density of wetting phase just a function of psiw*/
	  drhow_dpsiw = density_w.drho;
	  drhow_dsw   = 0.0;
	  /*density of nonwetting phase a function of sw and psiw through psin*/
	  drhon_dpsiw = density_n.drho;
	  drhon_dsw   = density_n.drho*psk.dpsic;
	  
	  /* w-phase mass */
	  mw[i]   = omega[matID]*density_w.rho*sw[i];
	  dmw_dsw[i]  = omega[matID]*(density_w.rho + sw[i]*drhow_dsw); 
	  dmw_dpsiw[i]= omega[matID]*sw[i]*drhow_dpsiw;

	  /* n-phase mass */
	  mn[i]       = omega[matID]*density_n.rho*(1.0-sw[i]);
	  dmn_dsw[i]  = omega[matID]*((1.0-sw[i])*drhon_dsw -density_n.rho); 
	  dmn_dpsiw[i]= omega[matID]*(1.0-sw[i])*drhon_dpsiw;

	  /* w-phase potential */
	  phi_psiw[i] = psiw[i];
	  dphi_psiw_dpsiw[i] = 1.0; 
      
	  /* n-phase  potential */
	  phi_psin[i] = psiw[i] + psk.psic;
	  dphi_psin_dpsiw[i]= 1.0;
	  dphi_psin_dsw[i]= psk.dpsic;

	  rhow_x       = density_w.rho;
	  drhow_x_dsw  = drhow_dsw;
	  drhow_x_dpsiw= drhow_dpsiw;
	  rhon_x       = density_n.rho;
	  drhon_x_dsw  = drhon_dsw;
	  drhon_x_dpsiw= drhon_dpsiw;
	  if (compressibilityFlag == 0) /*slight compressibility*/
	    {
	      rhow_x = 1.0;
	      drhow_x_dsw  = 0.0;
	      drhow_x_dpsiw= 0.0;
	      rhon_x = 1.0;
	      drhon_x_dsw  = 0.0;
	      drhon_x_dpsiw= 0.0;
	    }
	    
	  for (int I=0;I<nSpace;I++)
	    {
	      fw[i*nSpace+I]       = 0.0;
	      dfw_dsw[i*nSpace+I]  = 0.0;
	      dfw_dpsiw[i*nSpace+I]= 0.0;

	      fn[i*nSpace+I]       = 0.0;
	      dfn_dsw[i*nSpace+I]  = 0.0;
	      dfn_dpsiw[i*nSpace+I]= 0.0;
	      
	      for (int m=rowptr[I]; m < rowptr[I+1]; m++)
		{
		  /*assuming slight compressibility*/
		  /* w-phase advection (gravity) */
		  fw[i*nSpace+I] += rhow_x*rhow_x*Kbar[matID*nnz+m]*psk.krw/muw*g[colind[m]];
		  dfw_dsw[i*nSpace+I] += rhow_x*rhow_x*Kbar[matID*nnz+m]*psk.dkrw/muw*g[colind[m]] + 2.0*rhow_x*drhow_x_dsw*Kbar[matID*nnz+m]*psk.krw/muw*g[colind[m]];
		  dfw_dpsiw[i*nSpace+I] += 2.0*rhow_x*drhow_x_dpsiw*Kbar[matID*nnz+m]*psk.krw/muw*g[colind[m]];
		  /* n-phase advection (gravity) */
		  fn[i*nSpace+I] += rhon_x*rhon_x*Kbar[matID*nnz+m]*psk.krn/mun*b*g[colind[m]];
		  dfn_dsw[i*nSpace+I] += rhon_x*rhon_x*Kbar[matID*nnz+m]*psk.dkrn/mun*b*g[colind[m]] + 2.0*rhon_x*drhon_x_dsw*Kbar[matID*nnz+m]*psk.krn/mun*b*g[colind[m]];
		  dfn_dpsiw[i*nSpace+I] += 2.0*rhon_x*drhon_x_dpsiw*Kbar[matID*nnz+m]*psk.krn/mun*b*g[colind[m]];

		  /* w-phase diffusion */
		  aw[i*nnz+m]  = rhow_x*Kbar[matID*nnz+m]*psk.krw/muw;
		  daw_dsw[i*nnz+m]  = rhow_x*Kbar[matID*nnz+m]*psk.dkrw/muw + drhow_x_dsw*Kbar[matID*nnz+m]*psk.krw/muw;
		  daw_dpsiw[i*nnz+m]= drhow_x_dpsiw*Kbar[matID*nnz+m]*psk.krw/muw;
	      
		  /* n-phase  diffusion */
		  an[i*nnz+m]        = rhon_x*Kbar[matID*nnz+m]*psk.krn/mun;
		  dan_dsw[i*nnz+m]   = rhon_x*Kbar[matID*nnz+m]*psk.dkrn/mun + drhon_x_dsw*Kbar[matID*nnz+m]*psk.krn/mun;
		  dan_dpsiw[i*nnz+m] = drhon_x_dpsiw*Kbar[matID*nnz+m]*psk.krn/mun;

//mwf original
// 		  /* w-phase diffusion */
	      
// 		  aw[i*nnz+m]  = density_w.rho*Kbar[matID*nnz+m]*psk.krw/muw;
// 		  daw_dsw[i*nnz+m]  = density_w.rho*Kbar[matID*nnz+m]*psk.dkrw/muw + drhow_dsw*Kbar[matID*nnz+m]*psk.krw/muw;
// 		  daw_dpsiw[i*nnz+m]= drhow_dpsiw*Kbar[matID*nnz+m]*psk.krw/muw;
	      
// 		  /* n-phase  diffusion */
// 		  an[i*nnz+m]        = density_n.rho*Kbar[matID*nnz+m]*psk.krn/mun;
// 		  dan_dsw[i*nnz+m]   = density_n.rho*Kbar[matID*nnz+m]*psk.dkrn/mun + drhon_dsw*Kbar[matID*nnz+m]*psk.krn/mun;
// 		  dan_dpsiw[i*nnz+m] = drhon_dpsiw*Kbar[matID*nnz+m]*psk.krn/mun;
		}
	    }
	}
    }
  return 0;
}

inline int twophaseDarcy_vol_frac(int nSimplex,
				  int nPointsPerSimplex,
				  const int* materialTypes,
				  const double* omega,
				  const double* sw,
				  double * vol_frac_w,
				  double * vol_frac_n)
{
  int matID;
  for (int eN=0; eN < nSimplex; eN++)
    {
      matID = materialTypes[eN];
      for (int pN=0,i;pN<nPointsPerSimplex;pN++)
	{    
	  i = eN*nPointsPerSimplex+pN;
	  vol_frac_w[i] = omega[matID]*sw[i];
	  vol_frac_n[i] = omega[matID]*(1.0-sw[i]);

	}  
    }
  return 0;
}

/* fully coupled, phase continuity formulation, wetting phase pressure and capillary pressure are primary dependent
   variables except say u_1 = 1-S for u_1 <= 0
   slight-compressibility assumption is used for densities
   heterogeneity is represented by material types
   and diffusion tensors are sparse
*/
template<class PSK, class DENSITY_W, class DENSITY_N>
inline int twophaseDarcy_fc_pp_sd_het_matType(int nSimplex,
					      int nPointsPerSimplex,
					      int nSpace,
					      int nParams,		       
					      const proteus_csr_index_t* rowptr,
					      const proteus_csr_index_t* colind,
					      const int* materialTypes,
					      double muw,
					      double mun,
					      const double* omega,
					      const double* Kbar, /*now has to be tensor*/
					      double b,
					      const double* rwork_psk, const int* iwork_psk,
					      const double* rwork_psk_tol,
					      const double* rwork_density_w,
					      const double* rwork_density_n,
					      const double* g,
					      const double* x,
					      const double* psiw,
					      const double* psic,
					      double *sw,
					      double* mw,
					      double* dmw_dpsiw,
					      double* dmw_dpsic,
					      double* mn,
					      double* dmn_dpsiw,
					      double* dmn_dpsic,
					      double* phi_psiw,
					      double* dphi_psiw_dpsiw,
					      double* phi_psin,
					      double* dphi_psin_dpsiw,
					      double* dphi_psin_dpsic,
					      double* aw,
					      double* daw_dpsiw,
					      double* daw_dpsic,
					      double* an,
					      double* dan_dpsiw,
					      double* dan_dpsic)
{
  int matID;
  PSK psk(rwork_psk,iwork_psk);  psk.setTolerances(rwork_psk_tol);
  DENSITY_W density_w(rwork_density_w);
  DENSITY_N density_n(rwork_density_n);
  double drhow_dpsiw,drhow_dpsic,drhon_dpsiw,drhon_dpsic,
    swi,psin,dsw_dpsic,dpsin_dpsic;
  const int nnz = rowptr[nSpace];
  for (int eN=0;eN<nSimplex;eN++)
    {
      matID = materialTypes[eN];
      psk.setParams(&rwork_psk[matID*nParams]);
      for(int pN=0,i;pN<nPointsPerSimplex;pN++)
        {
          i = eN*nPointsPerSimplex+pN;
	  psk.calc_from_psic(psic[i]);
	  
	  /*non-wetting phase pressure head */
	  psin = psiw[i] + psic[i];
	  dpsin_dpsic = 1.0;
	  if (psic[i] <= 0.0)
	    {
	      psin = psiw[i];
	      dpsin_dpsic = 0.0;
	    }
	  /*aqueous phase saturation*/
	  swi = psk.Se*(psk.Sw_max-psk.Sw_min) + psk.Sw_min;
	  sw[i] = swi;
	  dsw_dpsic = psk.dSe_dpsic/psk.dSe_dSw;

	  /*mwf need to make sure normalized by rhow and rhon respectively!*/
	  density_w.calc(psiw[i]); 
	  density_n.calc(psin);
	  /*density of wetting phase just a function of psiw*/
	  drhow_dpsiw = density_w.drho;
	  drhow_dpsic = 0.0;
	  /*density of nonwetting phase a function of psic and psiw through psin*/
	  drhon_dpsiw = density_n.drho;
	  drhon_dpsic = density_n.drho;
	  if (psic[i] <= 0.0)
	    drhon_dpsic = 0.0;
	  /* w-phase mass */
	  mw[i]   = omega[matID]*density_w.rho*swi;
	  dmw_dpsic[i]= omega[matID]*(density_w.rho*dsw_dpsic + swi*drhow_dpsic); 
	  dmw_dpsiw[i]= omega[matID]*swi*drhow_dpsiw;

	  /* n-phase mass */
	  mn[i]       = omega[matID]*density_n.rho*(1.0-swi);
	  dmn_dpsic[i]= omega[matID]*((1.0-swi)*drhon_dpsic -density_n.rho*dsw_dpsic); 
	  dmn_dpsiw[i]= omega[matID]*(1.0-swi)*drhon_dpsiw;

	  /* w-phase potential */
	  phi_psiw[i] = psiw[i];
	  dphi_psiw_dpsiw[i] = 1.0; /*include density dependence below*/
      
	  /* n-phase  potential */
	  phi_psin[i] = psin;
	  dphi_psin_dpsiw[i]= 1.0;
	  dphi_psin_dpsic[i]= dpsin_dpsic;
      
	  for (int I=0;I<nSpace;I++)
	    {
	      /* update potentials with gravity */
	      /*slight compressibility, normalized density=1.0*/
	      phi_psiw[i] -= g[I]*x[i*3+I];
	      	      
	      phi_psin[i] -= b*g[I]*x[i*3+I];
	      
	      for (int m=rowptr[I]; m < rowptr[I+1]; m++)
		{
		  /* w-phase diffusion */
	      
		  aw[i*nnz+m]  = density_w.rho*Kbar[matID*nnz+m]*psk.krw/muw;
		  daw_dpsic[i*nnz+m]= density_w.rho*Kbar[matID*nnz+m]*psk.dkrw/muw + drhow_dpsic*Kbar[matID*nnz+m]*psk.krw/muw;
		  daw_dpsiw[i*nnz+m]= drhow_dpsiw*Kbar[matID*nnz+m]*psk.krw/muw;
	      
		  /* n-phase  diffusion */
		  an[i*nnz+m]        = density_n.rho*Kbar[matID*nnz+m]*psk.krn/mun;
		  dan_dpsic[i*nnz+m] = density_n.rho*Kbar[matID*nnz+m]*psk.dkrn/mun + drhon_dpsic*Kbar[matID*nnz+m]*psk.krn/mun;
		  dan_dpsiw[i*nnz+m] = drhon_dpsiw*Kbar[matID*nnz+m]*psk.krn/mun;
		}
	    }
	  /*adjust mass terms if psic <= 0, */
	  if (psic[i] <= 0.0)/*0.0)*/
	    {
	      /*mwf debug
	      printf("psic[%d]=%g krn=%g dkrn=%g krw=%g dkrw=%g sw=%g dsw_dpsic=%g drhow_dpsic=%g drhon_dpsic=%g \n",i,psic[i],psk.krn,psk.dkrn,psk.krw,psk.dkrw,swi,
		     dsw_dpsic,drhow_dpsic,drhon_dpsic); 
	      */
	      
	      dmn_dpsic[i] = omega[matID]*density_n.rho;
	      dmw_dpsic[i] =-omega[matID]*density_n.rho;//0.0;
	      //dmn_dpsiw[i] = 0.0;
	      phi_psin[i]  = 0.0;
	      dphi_psin_dpsiw[i]= 0.0;
	      dphi_psin_dpsic[i]= 0.0;
	      for (int I=0;I<nSpace;I++)
		{
		  for (int m=rowptr[I]; m < rowptr[I+1]; m++)
		    {
		      an[i*nnz+m] = 0.0;
		      dan_dpsic[i*nnz+m] = 0.0;
		      dan_dpsiw[i*nnz+m] = 0.0;
		      daw_dpsic[i*nnz+m] = 0.0;
		    }
		}

	    }
	}
    }
  return 0;
}

/* saturation equation in fractional flow formulation for incompressible flow
   heterogeneity is represented by materialTypes
 */
template<class PSK>
static inline int twophaseDarcy_incompressible_split_sd_saturation_het_matType(int nSimplex,
									       int nPointsPerSimplex,
									       int nSpace,
									       int nParams,
									       const proteus_csr_index_t* rowptr,
									       const proteus_csr_index_t* colind,
									       const int* materialTypes,
									       double muw,
									       double mun,
									       const double* omega,
									       const double* Kbar,
									       double b,
									       double capillaryDiffusionScaling,
									       double advectionScaling,
									       const double* rwork_psk, const int* iwork_psk,
									       const double* rwork_psk_tol,
									       const double* rwork_density_w,
									       const double* rwork_density_n,
									       const double* g,
									       const double* qt,
									       const double* sw,
									       double* m,
									       double* dm,
									       double* phi,
									       double* dphi,
									       double* f,
									       double* df,
									       double* a,
									       double* da)
{
  int matID;
  PSK psk(rwork_psk,iwork_psk);  psk.setTolerances(rwork_psk_tol);
  FractionalFlowVariables fracFlow(muw,mun);
  ConstantDensity density_w(rwork_density_w),density_n(rwork_density_n);
  const int nnz = rowptr[nSpace];
  //mwf debug
  //std::cout<<"entering twpffinc rhow= "<<density_w.rho<<" rhon= "<<density_n.rho<<" b= "<<b<<" ";
  //for (int I=0; I < nSpace; I++)
  //  {
  //    std::cout<<"g["<<I<<"]= "<<g[I]<<" ";
  //  }
  for (int eN=0;eN<nSimplex;eN++)
    {
      matID = materialTypes[eN];
      psk.setParams(&rwork_psk[matID*nParams]);
      for(int pN=0,i;pN<nPointsPerSimplex;pN++)
        {
          i = eN*nPointsPerSimplex+pN;
	  psk.calc(sw[i]);
	  fracFlow.calc(psk,density_w,density_n);
	  
	  /* wetting phase  mass */
	  m[i]   = omega[matID]*density_w.rho*sw[i]; 
	  dm[i]  = omega[matID]*density_w.rho; 
	  
	  /* capillary potential */
	  phi[i] = psk.psic; 
	  dphi[i]= psk.dpsic; 
	  
	  for (int I=0;I<nSpace;I++)
	    {
	      /* wetting phase advection */
	      /* todo, remove diagonal assumption on K*/
	      for (int m=rowptr[I]; m < rowptr[I+1]; m++)
		{
		  const int J = colind[m];
		  if (I==J)
		    {
		      f[i*nSpace+I]  = advectionScaling*(qt[i*nSpace+I]*fracFlow.fw
							 - Kbar[matID*nnz+m]*fracFlow.lambdaw*fracFlow.fn*(b*density_n.rho-density_w.rho)*g[I]) ;
		      df[i*nSpace+I] = advectionScaling*(qt[i*nSpace+I]*fracFlow.dfw
							 - (Kbar[matID*nnz+m]*g[I]*(b*density_n.rho-density_w.rho))*(fracFlow.lambdaw*fracFlow.dfn + fracFlow.fn*fracFlow.dlambdaw));
		    } 
		  /* wetting phase  capillary diffusion */
		  /*include scaling factor in case want to turn of capillary diffusion to test hyperbolic approximations*/
		  a[i*nnz+m]  = -capillaryDiffusionScaling*Kbar[matID*nnz+m]*fracFlow.lambdaw*fracFlow.fn;
		  da[i*nnz+m] = -capillaryDiffusionScaling*Kbar[matID*nnz+m]*(fracFlow.dlambdaw*fracFlow.fn + fracFlow.lambdaw*fracFlow.dfn);
		}
	    }
        }
    }
  return 0;
}

/* saturation equation in fractional flow formulation for slight compressible flow
   heterogeneity is represented by materialTypes
 */
template<class PSK, class DENSITY_W>
static inline int twophaseDarcy_slightCompressible_split_sd_saturation_het_matType(int nSimplex,
										   int nPointsPerSimplex,
										   int nSpace,
										   int nParams,
										   const proteus_csr_index_t* rowptr,
										   const proteus_csr_index_t* colind,
										   const int* materialTypes,
										   double muw,
										   double mun,
										   const double* omega,
										   const double* Kbar,
										   double b,
										   double capillaryDiffusionScaling,
										   double advectionScaling,
										   const double* rwork_psk, const int* iwork_psk,
										   const double* rwork_psk_tol,
										   const double* rwork_density_w,
										   const double* rwork_density_n,
										   const double* g,
										   const double* qt,
										   const double* psiw,
										   const double* sw,
										   double* m,
										   double* dm,
										   double* phi,
										   double* dphi,
										   double* f,
										   double* df,
										   double* a,
										   double* da)
{
  int matID;
  PSK psk(rwork_psk,iwork_psk);  psk.setTolerances(rwork_psk_tol);
  FractionalFlowVariables fracFlow(muw,mun);
  /*normalized densities \rho_{\alpha} = \varrho_{\alpha}/\varrho_{\alpha,0}
    for spatial term, assuming slight compressiblity so assume \rho_{\alpha} = 1
  */
  const double rwork_density_w_x[1] = {1.0}; const double rwork_density_n_x[1] = {1.0}; 
  ConstantDensity density_w_x(rwork_density_w_x),density_n_x(rwork_density_n_x);
  /*for accumulation term*/
  DENSITY_W density_w(rwork_density_w);
  const int nnz = rowptr[nSpace];
  //mwf debug
  //std::cout<<"entering twpffsl rhow= "<<density_w.rho<<" rhon= "<<density_n.rho<<" b= "<<b<<" ";
  //for (int I=0; I < nSpace; I++)
  //  {
  //    std::cout<<"g["<<I<<"]= "<<g[I]<<" ";
  //  }
  for (int eN=0;eN<nSimplex;eN++)
    {
      matID = materialTypes[eN];
      psk.setParams(&rwork_psk[matID*nParams]);
      for(int pN=0,i;pN<nPointsPerSimplex;pN++)
        {
          i = eN*nPointsPerSimplex+pN;
	  psk.calc(sw[i]);
	  density_w.calc(psiw[i]);
	  fracFlow.calc(psk,density_w_x,density_n_x);
	  
	  
	  /* wetting phase  mass */
	  m[i]   = omega[matID]*density_w.rho*sw[i]; 
	  /*density of wetting phase just a function of psiw*/
	  dm[i]  = omega[matID]*density_w.rho; 
	  
	  /* capillary potential */
	  phi[i] = psk.psic; 
	  dphi[i]= psk.dpsic; 
	  
	  for (int I=0;I<nSpace;I++)
	    {
	      /* wetting phase advection */
	      /* todo remove diagonal K assumption*/
	      for (int m=rowptr[I]; m < rowptr[I+1]; m++)
		{
		  const int J = colind[m];
		  if (I==J)
		    {
		      f[i*nSpace+I]  = advectionScaling*(qt[i*nSpace+I]*fracFlow.fw
							 - Kbar[matID*nnz+m]*fracFlow.lambdaw*fracFlow.fn*(b*density_n_x.rho-density_w_x.rho)*g[I]) ;
		      df[i*nSpace+I] = advectionScaling*(qt[i*nSpace+I]*fracFlow.dfw
							 - (Kbar[matID*nnz+m]*g[I]*(b*density_n_x.rho-density_w_x.rho))*(fracFlow.lambdaw*fracFlow.dfn + fracFlow.fn*fracFlow.dlambdaw));
		    } 
		  /* wetting phase  capillary diffusion */
		  /*include scaling factor in case want to turn of capillary diffusion to test hyperbolic approximations*/
		  a[i*nnz+m]  = -capillaryDiffusionScaling*Kbar[matID*nnz+m]*fracFlow.lambdaw*fracFlow.fn;
		  da[i*nnz+m] = -capillaryDiffusionScaling*Kbar[matID*nnz+m]*(fracFlow.dlambdaw*fracFlow.fn + fracFlow.lambdaw*fracFlow.dfn);
		}
	    }
        }
    }
  return 0;
}

/* pressure equation in fractional flow formulation for incompressible flow
   heterogeneity is represented by materialTypes
*/
template<class PSK>
static inline int twophaseDarcy_incompressible_split_sd_pressure_het_matType(int nSimplex,
									     int nPointsPerSimplex,
									     int nSpace,
									     int nParams,
									     const proteus_csr_index_t* rowptr,
									     const proteus_csr_index_t* colind,
									     const int* materialTypes,
									     double muw,
									     double mun,
									     const double* omega,
									     const double* Kbar,
									     double b,
									     double capillaryDiffusionScaling,
									     const double* rwork_psk, const int* iwork_psk,
									     const double* rwork_psk_tol,
									     const double* rwork_density_w,
									     const double* rwork_density_n,
									     const double* g,
									     const double* sw,
									     const double* grad_psic,
									     double* f,
									     double* a)
{
  int matID;
  PSK psk(rwork_psk,iwork_psk); psk.setTolerances(rwork_psk_tol);
  FractionalFlowVariables fracFlow(muw,mun);
  ConstantDensity density_w(rwork_density_w),density_n(rwork_density_n);
  const int nnz = rowptr[nSpace];
  for (int eN=0;eN<nSimplex;eN++)
    {
      matID = materialTypes[eN];
      psk.setParams(&rwork_psk[matID*nParams]);
      for(int pN=0,i;pN<nPointsPerSimplex;pN++)
        {
          i = eN*nPointsPerSimplex+pN;
	  psk.calc(sw[i]);
	  fracFlow.calc(psk,density_w,density_n);
	  /*todo remove diagonal assumption on f*/
	  for (int I=0;I<nSpace;I++)
	    {
	      /*include scaling factor in case want to turn of capillary diffusion to test hyperbolic approximations*/
	      for (int m=rowptr[I]; m < rowptr[I+1]; m++)
		{
		  const int J = colind[m];
		  if (I == J)
		    {
		      //mwf original
		      f[i*nSpace+I]  = - capillaryDiffusionScaling*Kbar[matID*nnz+m]*fracFlow.lambdat*(fracFlow.fn*grad_psic[i*nSpace+I]) + 
			Kbar[matID*nnz+m]*fracFlow.lambdat*(density_w.rho + fracFlow.fn*(b*density_n.rho-density_w.rho))*g[I];
		      //f[i*nSpace+I]  = - capillaryDiffusionScaling*Kbar[matID*nnz+m]*fracFlow.lambdat*(fracFlow.fn*psk.dpsic*grad_sw[i*nSpace+I]) + 
		      //	Kbar[matID*nnz+m]*fracFlow.lambdat*(density_w.rho + fracFlow.fn*(b*density_n.rho-density_w.rho))*g[I];
		    }
		  a[i*nnz+m]  = Kbar[matID*nnz+m]*fracFlow.lambdat;
		}
	    }
        }
    }
  return 0.0;
}

/* pressure equation in fractional flow formulation for incompressible flow
   heterogeneity is represented by materialTypes
*/
template<class PSK, class DENSITY_W, class DENSITY_N>
static inline int twophaseDarcy_slightCompressible_split_sd_pressure_het_matType(int nSimplex,
										   int nPointsPerSimplex,
										   int nSpace,
										   int nParams,
										   const proteus_csr_index_t* rowptr,
										   const proteus_csr_index_t* colind,
										   const int* materialTypes,
										   double muw,
										   double mun,
										   const double* omega,
										   const double* Kbar,
										   double b,
										   double capillaryDiffusionScaling,
										   const double* rwork_psk, const int* iwork_psk,
										   const double* rwork_psk_tol,
										   const double* rwork_density_w,
										   const double* rwork_density_n,
										   const double* g,
										   const double* sw,
										   const double* psiw,
										   const double* psin,
										   const double* grad_psic,
										   double* m,
										   double* dm,
										   double* f,
										   double* a)
{
  int matID;
  PSK psk(rwork_psk,iwork_psk); psk.setTolerances(rwork_psk_tol);
  FractionalFlowVariables fracFlow(muw,mun);
  /*normalized densities \rho_{\alpha} = \varrho_{\alpha}/\varrho_{\alpha,0}
    for spatial term, assuming slight compressiblity so assume \rho_{\alpha} = 1
  */
  const double rwork_density_w_x[1] = {1.0}; const double rwork_density_n_x[1] = {1.0}; 
  ConstantDensity density_w_x(rwork_density_w_x),density_n_x(rwork_density_n_x);
  DENSITY_W density_w(rwork_density_w);
  DENSITY_N density_n(rwork_density_n);
  const int nnz = rowptr[nSpace];
  double drhow_dpsiw,drhon_dpsiw;
  for (int eN=0;eN<nSimplex;eN++)
    {
      matID = materialTypes[eN];
      psk.setParams(&rwork_psk[matID*nParams]);
      for(int pN=0,i;pN<nPointsPerSimplex;pN++)
        {
          i = eN*nPointsPerSimplex+pN;
	  psk.calc(sw[i]);
	  density_w.calc(psiw[i]);
	  density_n.calc(psin[i]);

	  fracFlow.calc(psk,density_w_x,density_n_x);

	  /*density of wetting phase just a function of psiw*/
	  drhow_dpsiw = density_w.drho;
	  /*density of nonwetting phase a function of sw and psiw through psin but ignore sw derivative because of splitting*/
	  drhon_dpsiw = density_n.drho;
	  /*mwf debug
	  printf("twpff slc psiw=%g psin=%g rhow=%g rhon=%g drhow=%g drhon=%g \n",psiw[i],psin[i],density_w.rho,density_n.rho,drhow_dpsiw,drhon_dpsiw);
	  */
	  m[i] = omega[matID]*density_w.rho*sw[i] + omega[matID]*density_n.rho*(1.0-sw[i]);
	  dm[i]= omega[matID]*drhow_dpsiw*sw[i]   + omega[matID]*drhon_dpsiw*(1.0-sw[i]);

	  for (int I=0;I<nSpace;I++)
	    {
	      /*include scaling factor in case want to turn of capillary diffusion to test hyperbolic approximations*/
	      for (int m=rowptr[I]; m < rowptr[I+1]; m++)
		{
		  const int J = colind[m];
		  if (I == J)
		    {
		      f[i*nSpace+I]  = - capillaryDiffusionScaling*Kbar[matID*nnz+m]*fracFlow.lambdat*(fracFlow.fn*grad_psic[i*nSpace+I]) + 
			Kbar[matID*nnz+m]*fracFlow.lambdat*(density_w_x.rho + fracFlow.fn*(b*density_n_x.rho-density_w_x.rho))*g[I];
		    }
		  a[i*nnz+m]  = Kbar[matID*nnz+m]*fracFlow.lambdat;
		}
	    }
        }
    }
  return 0.0;
}

/* saturation equation in fractional flow formulation for slight compressible flow
   heterogeneity is represented by materialTypes
 */
template<class PSK, class DENSITY_N>
static inline int twophaseDarcy_compressibleN_split_sd_saturation_het_matType(int nSimplex,
									      int nPointsPerSimplex,
									      int nSpace,
									      int nParams,
									      const proteus_csr_index_t* rowptr,
									      const proteus_csr_index_t* colind,
									      const int* materialTypes,
									      double muw,
									      double mun,
									      const double* omega,
									      const double* Kbar,
									      double b,
									      double capillaryDiffusionScaling,
									      double advectionScaling,
									      const double* rwork_psk, const int* iwork_psk,
									      const double* rwork_psk_tol,
									      const double* rwork_density_w,
									      const double* rwork_density_n,
									      const double* g,
									      const double* qt,
									      const double* psiw,
									      const double* sw,
									      double* m,
									      double* dm,
									      double* phi,
									      double* dphi,
									      double* f,
									      double* df,
									      double* a,
									      double* da)
{
  int matID;
  PSK psk(rwork_psk,iwork_psk);  psk.setTolerances(rwork_psk_tol);
  CompressibleN_FractionalFlowVariables fracFlow(muw,mun);
  ConstantDensity density_w(rwork_density_w);
  DENSITY_N density_n(rwork_density_n);
  const int nnz = rowptr[nSpace];
  //mwf debug
  //std::cout<<"entering twpffc rhow= "<<density_w.rho<<" rhon= "<<density_n.rho<<" b= "<<b<<" ";
  //for (int I=0; I < nSpace; I++)
  //  {
  //    std::cout<<"g["<<I<<"]= "<<g[I]<<" ";
  //  }
  double psin;
  for (int eN=0;eN<nSimplex;eN++)
    {
      matID = materialTypes[eN];
      psk.setParams(&rwork_psk[matID*nParams]);
      for(int pN=0,i;pN<nPointsPerSimplex;pN++)
        {
          i = eN*nPointsPerSimplex+pN;
	  psk.calc(sw[i]);
	  
	  psin = psiw[i] + psk.psic;

	  density_w.calc(psiw[i]);
	  density_n.calc(psin);

	  fracFlow.calc(psk,density_w,density_n);
	  
	  
	  /* wetting phase  mass */
	  m[i]   = omega[matID]*density_w.rho*sw[i]; 
	  dm[i]  = omega[matID]*density_w.rho; 
	  
	  /* capillary potential */
	  phi[i] = psk.psic; 
	  dphi[i]= psk.dpsic; 
	  
	  for (int I=0;I<nSpace;I++)
	    {
	      /* wetting phase advection */
	      for (int m=rowptr[I]; m < rowptr[I+1]; m++)
		{
		  const int J = colind[m];
		  if (I==J)
		    {
		      f[i*nSpace+I]  = advectionScaling*(qt[i*nSpace+I]*fracFlow.fw
							 - Kbar[matID*nnz+m]*fracFlow.lambdaw*fracFlow.fn*(b*density_n.rho-density_w.rho)*g[I]) ;
		      df[i*nSpace+I] = advectionScaling*(qt[i*nSpace+I]*fracFlow.dfw
							 - (Kbar[matID*nnz+m]*g[I]*(b*density_n.rho-density_w.rho))*(fracFlow.lambdaw*fracFlow.dfn + fracFlow.fn*fracFlow.dlambdaw));
		    } 
		  /* wetting phase  capillary diffusion */
		  /*include scaling factor in case want to turn of capillary diffusion to test hyperbolic approximations*/
		  a[i*nnz+m]  = -capillaryDiffusionScaling*Kbar[matID*nnz+m]*fracFlow.lambdaw*fracFlow.fn;
		  da[i*nnz+m] = -capillaryDiffusionScaling*Kbar[matID*nnz+m]*(fracFlow.dlambdaw*fracFlow.fn + fracFlow.lambdaw*fracFlow.dfn);
		}
	    }
        }
    }
  return 0;
}

/* pressure equation in fractional flow formulation for incompressible flow
   heterogeneity is represented by materialTypes
*/
template<class PSK, class DENSITY_N>
static inline int twophaseDarcy_compressibleN_split_sd_pressure_het_matType(int nSimplex,
									    int nPointsPerSimplex,
									    int nSpace,
									    int nParams,
									    const proteus_csr_index_t* rowptr,
									    const proteus_csr_index_t* colind,
									    const int* materialTypes,
									    double muw,
									    double mun,
									    const double* omega,
									    const double* Kbar,
									    double b,
									    double capillaryDiffusionScaling,
									    const double* rwork_psk, const int* iwork_psk,
									    const double* rwork_psk_tol,
									    const double* rwork_density_w,
									    const double* rwork_density_n,
									    const double* g,
									    const double* sw,
									    const double* psiw,
									    const double* psin,
									    const double* grad_psic,
									    double* m,
									    double* dm,
									    double* f,
									    double* a)
{
  int matID;
  PSK psk(rwork_psk,iwork_psk); psk.setTolerances(rwork_psk_tol);
  CompressibleN_FractionalFlowVariables fracFlow(muw,mun);
  ConstantDensity density_w(rwork_density_w);
  DENSITY_N density_n(rwork_density_n);
  const int nnz = rowptr[nSpace];
  double drhon_dpsiw;
  for (int eN=0;eN<nSimplex;eN++)
    {
      matID = materialTypes[eN];
      psk.setParams(&rwork_psk[matID*nParams]);
      for(int pN=0,i;pN<nPointsPerSimplex;pN++)
        {
          i = eN*nPointsPerSimplex+pN;
	  psk.calc(sw[i]);
	  density_w.calc(psiw[i]);
	  density_n.calc(psin[i]);

	  fracFlow.calc(psk,density_w,density_n);

	  /*density of nonwetting phase a function of sw and psiw through psin but ignore sw derivative because of splitting*/
	  drhon_dpsiw = density_n.drho;
	  /*mwf debug
	  printf("twpffc psiw=%g psin=%g rhow=%g rhon=%g drhon=%g \n",psiw[i],psin[i],density_w.rho,density_n.rho,drhon_dpsiw);
	  */
	  m[i] = omega[matID]*density_w.rho*sw[i] + omega[matID]*density_n.rho*(1.0-sw[i]);
	  dm[i]= omega[matID]*drhon_dpsiw*(1.0-sw[i]);

	  for (int I=0;I<nSpace;I++)
	    {
	      /*include scaling factor in case want to turn of capillary diffusion to test hyperbolic approximations*/
	      for (int m=rowptr[I]; m < rowptr[I+1]; m++)
		{
		  const int J = colind[m];
		  if (I == J)
		    {
		      f[i*nSpace+I]  = - capillaryDiffusionScaling*Kbar[matID*nnz+m]*fracFlow.lambdat*(fracFlow.fn*grad_psic[i*nSpace+I]) + 
			Kbar[matID*nnz+m]*fracFlow.lambdat*(density_w.rho + fracFlow.fn*(b*density_n.rho-density_w.rho))*g[I];
		    }
		  a[i*nnz+m]  = Kbar[matID*nnz+m]*fracFlow.lambdat;
		}
	    }
        }
    }
  return 0.0;
}

template<class PSK>
static inline int twophaseDarcy_incompressible_split_pp_sd_saturation_het_matType(int nSimplex,
										  int nPointsPerSimplex,
										  int nSpace,
										  int nParams,
										  const proteus_csr_index_t* rowptr,
										  const proteus_csr_index_t* colind,
										  const int* materialTypes,
										  double muw,
										  double mun,
										  const double* omega,
										  const double* Kbar,
										  double b,
										  double capillaryDiffusionScaling,
										  double advectionScaling,
										  const double* rwork_psk, const int* iwork_psk,
										  const double* rwork_psk_tol,
										  const double* rwork_density_w,
										  const double* rwork_density_n,
										  const double* g,
										  const double* qt,
										  const double* u,//psic u >= 0, sn u < 0
										  double* sw,
										  double* m,
										  double* dm,
										  double* phi,
										  double* dphi,
										  double* f,
										  double* df,
										  double* a,
										  double* da)
{
  int matID;
  PSK psk(rwork_psk,iwork_psk);  psk.setTolerances(rwork_psk_tol);
  FractionalFlowVariables fracFlow(muw,mun);
  ConstantDensity density_w(rwork_density_w),density_n(rwork_density_n);
  const int nnz = rowptr[nSpace];
  double psic_eval(0.),dpsic_eval(0.),dsw_du(0.);
  //mwf debug
  //std::cout<<"entering twpffinc rhow= "<<density_w.rho<<" rhon= "<<density_n.rho<<" b= "<<b<<" ";
  //for (int I=0; I < nSpace; I++)
  //  {
  //    std::cout<<"g["<<I<<"]= "<<g[I]<<" ";
  //  }
  for (int eN=0;eN<nSimplex;eN++)
    {
      matID = materialTypes[eN];
      psk.setParams(&rwork_psk[matID*nParams]);
      for(int pN=0,i;pN<nPointsPerSimplex;pN++)
        {
          i = eN*nPointsPerSimplex+pN;
	  if (u[i] < 0.0)
	    {
	      sw[i] = psk.Sw_max;
	      psic_eval = 0.0;
	      dsw_du = -1.0;
	      psk.calc(sw[i]); //calculate psk as function of sw=1-u
	      dpsic_eval=psk.dpsic;
	    }
	  else
	    {
	      psic_eval = u[i]; dpsic_eval=1.0;
	      psk.calc_from_psic(psic_eval);
	      sw[i] = psk.Se*(psk.Sw_max-psk.Sw_min) + psk.Sw_min;
	      dsw_du = psk.dSe_dpsic/psk.dSe_dSw;
	    }
	  fracFlow.calc(psk,density_w,density_n);
	  
	  /* wetting phase  mass */
	  m[i]   = omega[matID]*density_w.rho*sw[i]; 
	  dm[i]  = omega[matID]*density_w.rho*dsw_du; 
	  
	  /* capillary potential */
	  phi[i] = psic_eval; 
	  dphi[i]= dpsic_eval; 
	  
	  for (int I=0;I<nSpace;I++)
	    {
	      /* wetting phase advection */
	      /* todo, remove diagonal assumption on K*/
	      for (int m=rowptr[I]; m < rowptr[I+1]; m++)
		{
		  const int J = colind[m];
		  if (I==J)
		    {
		      f[i*nSpace+I]  = advectionScaling*(qt[i*nSpace+I]*fracFlow.fw
							 - Kbar[matID*nnz+m]*fracFlow.lambdaw*fracFlow.fn*(b*density_n.rho-density_w.rho)*g[I]) ;
		      df[i*nSpace+I] = advectionScaling*(qt[i*nSpace+I]*fracFlow.dfw
							 - (Kbar[matID*nnz+m]*g[I]*(b*density_n.rho-density_w.rho))*(fracFlow.lambdaw*fracFlow.dfn + fracFlow.fn*fracFlow.dlambdaw));
		    } 
		  /* wetting phase  capillary diffusion */
		  /*include scaling factor in case want to turn of capillary diffusion to test hyperbolic approximations*/
		  a[i*nnz+m]  = -capillaryDiffusionScaling*Kbar[matID*nnz+m]*fracFlow.lambdaw*fracFlow.fn;
		  da[i*nnz+m] = -capillaryDiffusionScaling*Kbar[matID*nnz+m]*(fracFlow.dlambdaw*fracFlow.fn + fracFlow.lambdaw*fracFlow.dfn);
		}
	    }
        }
    }
  return 0;
}

//loop through and generate a spline table using nknots points given in array domain
//insert values into splineTable in the order
//u_0,..u_{nk-1},uinv_0,...,uinv_{nk-1},krw_0,...,krw_{nk-1},krn_0,...,krn_{nk-1}
//where uinv = psic if u=Sw and vice versa
//assumes all for one media type
template <class PSK>
inline void generateSplineTables(int nknots,
				 int startIndex,
				 int calcFlag, //0 --- genate tables for f(S_w), 1, generate tables for f(psi_c)
				 const double* domain,
				 const double* rwork_psk,
				 const int* iwork_psk,
				 const double* rwork_psk_tol,
				 double* splineTable)
{
  PSK psk(rwork_psk,iwork_psk);
  psk.setTolerances(rwork_psk_tol);

  if (calcFlag == 0)
    {
      for (int i=0; i < nknots; i++)
	{
	  double sw = domain[i];
	  psk.calc(sw);
	  //mwf debug
	  std::cout<<"generate splineTable calcFlag=0 startIndex= "<<startIndex<<" nknots= "<<nknots<<" sw["<<i<<"]= "<<sw<<" psic= "<<psk.psic<<" krw= "<<psk.krw<<" krn= "<<psk.krn<<std::endl;
	  splineTable[startIndex+i] =sw;
	  splineTable[startIndex + nknots  +i]=psk.psic;
	  splineTable[startIndex + nknots*2+i]=psk.krw;
	  splineTable[startIndex + nknots*3+i]=psk.krn;
	}
      //go ahead and fix enpoints manually 
      //capillary pressure at dry end
      splineTable[startIndex + nknots] = splineTable[startIndex + nknots+1]+1.0;
    }
  else
    {
      for (int i=0; i < nknots; i++)
	{
	  double psic = domain[i];
	  psk.calc_from_psic(psic);
	  splineTable[startIndex+i] =psic;
	  splineTable[startIndex + nknots  +i]=psk.Se*(psk.Sw_max-psk.Sw_min) + psk.Sw_min;
	  splineTable[startIndex + nknots*2+i]=psk.krw;
	  splineTable[startIndex + nknots*3+i]=psk.krn;
	}
      //fix capillary pressure at wet end
      splineTable[startIndex + nknots] = splineTable[startIndex + nknots+1]*(1.0+1.e-8);
      //krw at wet end
      splineTable[startIndex + nknots*2] = splineTable[startIndex + nknots*2+1];
      //krn at wet end
      splineTable[startIndex + nknots*3] = splineTable[startIndex + nknots*3+1];


    }
  
}
#endif
