#ifndef SEDCLOSURE_H
#define SEDCLOSURE_H

#include <cmath>
#include <iostream>
#include "pyarray.h"

namespace proteus
{  
  template<int nSpace>
  class cppHsuSedStress
{
public:
 cppHsuSedStress(
		 double aDarcy, // darcy parameter for drag term. Default value from Ergun (1952) is 150
		 double betaForch, // forchheimer parameter for drag term. Default value from Ergun (1952) is 1.75
		 double grain, // Grain size, default assumed as d50
		 double packFraction, //Critical volume fraction for switching the drag relation 0.2 by default, see Chen and Hsu 2014
		 double packMargin, // For packFraction +- packmargin, the drag coefficient is calculated by taking a weighted combination of the two relation (for packed and non-packed sediment
		 double maxFraction,
		 double frFraction,
		 double sigmaC,
		 double C3e,
		 double C4e,
		 double eR,
		 double fContact,
		 double mContact,
		 double nContact,
		 double angFriction,
		 double vos_limiter,
		 double mu_fr_limiter
): 
 
    aDarcy_(aDarcy), 
    betaForch_(betaForch), 
    grain_(grain),
    packFraction_(packFraction),
    packMargin_(packMargin),
    frFraction_(frFraction),
    maxFraction_(maxFraction),    
    sigmaC_(sigmaC),  
    C3e_(C3e),
    C4e_(C4e),
    eR_(eR),
    fContact_(fContact),
    mContact_(mContact),
    nContact_(nContact),
    angFriction_(angFriction),
    small_(1e-100),
    notSoLarge_(1e+6),
    large_(1e+100),
    vos_limiter_(vos_limiter),
	mu_fr_limiter_(mu_fr_limiter)


         
    {}
  
    inline double  xt_betaCoeff(     
			      double sedF, // Sediment fraction
			      double rhoFluid,
			      const proteus::pyarray<double>& uFluid, //Fluid velocity
			      const proteus::pyarray<double>& uSolid, //Sediment velocity
			      double nu //Kinematic viscosity
			      )
    {
        return betaCoeff(sedF, rhoFluid, uFluid.data(), uSolid.data(), nu);
    }


    inline double  betaCoeff(     
			      double sedF, // Sediment fraction
			      double rhoFluid,
			      const double uFluid[nSpace], //Fluid velocity
			      const double uSolid[nSpace], //Sediment velocity
			      double nu //Kinematic viscosity
			      )
  {
    double du2 = 0.;
    for (int ii=0; ii<nSpace;  ii++)
      {
	du2+= (uFluid[ii] - uSolid[ii])*(uFluid[ii] - uSolid[ii]);
      }
    double du = sqrt(du2);
    double weight = 1.;


    double gDrag1 = (aDarcy_*sedF*nu/((1. - sedF)*grain_*grain_) + betaForch_*du/grain_); // Darcy forchheimer term     
    double Cd = 0.44;
    double Re = (1-sedF)*du*grain_/nu;
    double Re_min = 1.0e-3;
    if (Re < 1000) 
      {
        Cd = 24*(1. + 0.15*pow(fmax(Re_min,Re), 0.687))/fmax(Re_min,Re);                       //Particle cloud resistance Wen and Yu 1966
      }

    double gDrag2 = 0.75 * Cd * du * pow(1. - sedF, -1.65)/grain_;
	  
    //cek  debug -- this makes the drag term nonlinear
     if(sedF < packFraction_ + packMargin_)  
       {  
         if (sedF > packFraction_ - packMargin_) 
           {	   
             weight =  0.5 + 0.5* (sedF - packFraction_) /packMargin_; 
           } 
         else 
           {
            weight =  0.; 
           } 
       } 
    
    return (weight*gDrag1 + (1.-weight)*gDrag2)*rhoFluid;
    }

    inline double gs0(double sedF)
    {
      double g0(0.);
      if(sedF< 0.635)
	{
	  if(sedF< 0.49)
	    {
	      g0 = 0.5*(2.-sedF)/((1-sedF)*(1-sedF)*(1-sedF));
	    }
	  else 
	    {
	      g0= 0.853744035/(0.64-sedF);
	    }
	}
      else g0 = 170.74880702;
      return g0;
    }

     inline double xt_deps_sed_deps(
		      double sedF, // Sediment fraction
                      double rhoFluid,
		      double rhoSolid,
		      const proteus::pyarray<double>& uFluid,
		      const proteus::pyarray<double>& uSolid,
		      const proteus::pyarray<double>& gradC, //Sediment velocity
		      double nu, //Kinematic viscosity
		      double theta_n,
		      double kappa_n,
		      double epsilon_n,
		      double nuT_n,
		      const proteus::pyarray<double>& g)
     {
         return deps_sed_deps(sedF,
                              rhoFluid,
                              rhoSolid,
                              uFluid.data(),
                              uSolid.data(),
                              gradC.data(),
                              nu,
                              theta_n,
                              kappa_n,
                              epsilon_n,
                              nuT_n,
                              g.data());

     }
    
     inline double deps_sed_deps(
		      double sedF, // Sediment fraction
              double rhoFluid,
		      double rhoSolid,
		      const double uFluid[nSpace],
		      const double uSolid[nSpace],
		      const double gradC[nSpace], //Sediment velocity
		      double nu, //Kinematic viscosity
		      double theta_n,
		      double kappa_n,
		      double epsilon_n,
		      double nuT_n,
		      const double g[nSpace])
		   
    {		   
      
      double beta = betaCoeff(sedF,rhoFluid,uFluid,uSolid,nu)+small_;
      double gs = gs0(sedF)+small_;
      double l_c = sqrt(M_PI)*grain_/(24.*(sedF+small_)*gs);
      double t_p = rhoSolid/beta;
      double t_c = l_c/(sqrt(theta_n) + small_);
      double t_l = 0.165*kappa_n/(epsilon_n + small_);
      double t_cl = std::min(t_c,t_l);
      double alpha= t_cl/(t_cl + t_p);
      double term = beta/(rhoFluid*(1.-sedF));
      double es_1 = 2.*term*(1-alpha)*sedF*kappa_n;
      double g_gradC = 0.;
      for (int ii=0; ii<nSpace;  ii++)
	{
	  g_gradC+= g[ii]*gradC[ii];
	}

      double ss = rhoSolid/rhoFluid - 1.;
      double es_2 = nuT_n*ss*g_gradC/sigmaC_/(1.-sedF) ;
      
      return C3e_ * es_1 / kappa_n +C4e_ * es_2 / kappa_n;

    }
      
		      
    inline double xt_kappa_sed1(
		      double sedF, // Sediment fraction
		      double rhoFluid,
		      double rhoSolid,
		      const proteus::pyarray<double>& uFluid,
		      const proteus::pyarray<double>& uSolid,
		      const proteus::pyarray<double>& gradC,
		      double nu, 
		      double theta_n,
		      double kappa_n,
		      double epsilon_n,
		      double nuT_n,
		      const proteus::pyarray<double>& g)
			   
    {
        return kappa_sed1(sedF,
                          rhoFluid,
                          rhoSolid,
                          uFluid.data(),
                          uSolid.data(),
                          gradC.data(),
                          nu,
                          theta_n,
                          kappa_n,
                          epsilon_n,
                          nuT_n,
                          g.data());
    }

    inline double kappa_sed1(
		      double sedF, // Sediment fraction
		      double rhoFluid,
		      double rhoSolid,
		      const double uFluid[nSpace],
		      const double uSolid[nSpace],
		      const double gradC[nSpace],
		      double nu, 
		      double theta_n,
		      double kappa_n,
		      double epsilon_n,
		      double nuT_n,
		      const double g[nSpace])
    {
      
      double beta = betaCoeff(sedF,rhoFluid,uFluid,uSolid,nu)+small_;
      double gs = gs0(sedF)+small_;
      double l_c = sqrt(M_PI)*grain_/(24.*(sedF+small_)*gs);
      double t_p = rhoSolid/beta;
      double t_c = l_c/(sqrt(theta_n) + small_);
      double t_l = 0.165*kappa_n/(epsilon_n + small_);
      double t_cl = std::min(t_c,t_l);
      double alpha= t_cl/(t_cl + t_p);
      double term = beta/(rhoFluid*(1.-sedF));
      double es_1 = 2.*term*(1-alpha)*sedF*kappa_n;
      double g_gradC = 0.;
      for (int ii=0; ii<nSpace;  ii++)
	{
	  g_gradC+= g[ii]*gradC[ii];
	}

      double ss = rhoSolid/rhoFluid - 1.;
      double es_2 = nuT_n*ss*g_gradC/sigmaC_/(1.-sedF) ;
      
      return  es_1  + es_2;

    }

    inline double xt_dkappa_sed1_dk(
		      double sedF, // Sediment fraction
		      double rhoFluid,
		      double rhoSolid,
		      const proteus::pyarray<double>& uFluid, //Fluid velocity
		      const proteus::pyarray<double>& uSolid, //Sediment velocity
		      const proteus::pyarray<double>& gradC, //Sediment velocity
		      double nu, //Kinematic viscosity
		      double theta_n,
		      double kappa_n,
		      double epsilon_n,
		      double nuT_n)
    {
        return dkappa_sed1_dk(sedF,
                              rhoFluid,
                              rhoSolid,
                              uFluid.data(),
                              uSolid.data(),
                              gradC.data(),
                              nu,
                              theta_n,
                              kappa_n,
                              epsilon_n,
                              nuT_n);

    }

    inline double dkappa_sed1_dk(
		      double sedF, // Sediment fraction
		      double rhoFluid,
		      double rhoSolid,
		      const double uFluid[nSpace], //Fluid velocity
		      const double uSolid[nSpace], //Sediment velocity
		      const double gradC[nSpace], //Sediment velocity
		      double nu, //Kinematic viscosity
		      double theta_n,
		      double kappa_n,
		      double epsilon_n,
		      double nuT_n)			   
    {		   
      double beta = betaCoeff(sedF,rhoFluid,uFluid,uSolid,nu)+small_;
      double gs = gs0(sedF)+small_;
      double l_c = sqrt(M_PI)*grain_/(24.*(sedF+small_)*gs);
      double t_p = rhoSolid/beta;
      double t_c = l_c/(sqrt(theta_n) + small_);
      double t_l = 0.165*kappa_n/(epsilon_n + small_);
      double t_cl = std::min(t_c,t_l);
      double alpha= t_cl/(t_cl + t_p);
      double term = beta/(rhoFluid*(1.-sedF));
      double es_1 = 2.*term*rhoSolid*(1-alpha)*sedF;
      return es_1;

    }



    inline double psc(		      double sedF, 
				      double rhoSolid,
				      double theta )
      
    {
      double gs_0 = gs0(sedF);
      double eRp1 = 1.+eR_;
      double psc = rhoSolid * sedF * (1.+2.*eRp1*sedF*gs_0)*theta;
      return psc;
    }

    inline double psc_term(		      double sedF, 
					      double rhoSolid,
					      double theta_np1,
					      double du_dx,
					      double dv_dy,
					      double dw_dz)
      
    {
      
      return -2.*psc(sedF,rhoSolid,theta_np1)*(du_dx + dv_dy + dw_dz)/(3.*rhoSolid * sedF);
    }

    inline double dpsc_term_dtheta(		      double sedF, 
					      double rhoSolid,
					      double du_dx,
					      double dv_dy,
					      double dw_dz)
      
    {
      
      double gs_0 = gs0(sedF);
      double eRp1 = 1.+eR_;
      double dpsc = rhoSolid * sedF * (1.+2.*eRp1*sedF*gs_0);
      return -2.*dpsc*(du_dx + dv_dy + dw_dz)/(3.*rhoSolid * sedF);
    }

    
    inline double mu_sc(		      double sedF, 
					      double rhoSolid, 
					      double theta )

    {
      double gs_0 = gs0(sedF);
      double sq_pi = (sqrt(M_PI));
      double sedF2 = sedF * sedF;
      double eRp1 = 1.+eR_;
      double sq_theta = sqrt(theta);

      double mu_sc = rhoSolid*grain_*sq_theta*( 0.8*sedF2*gs_0*eRp1/(sq_pi) + (1./15.)*sq_pi*sedF2*gs_0*eRp1 + (1./6.)*sq_pi*sedF + (5./48)*sq_pi/(gs_0*eRp1) );

      return mu_sc;
    }

    inline double l_sc(		      double sedF, 
				      double rhoSolid, 
				      double theta_n )

    {
      double gs_0 = gs0(sedF);
      double sq_pi = (sqrt(M_PI));
      double sedF2 = sedF * sedF;
      double eRp1 = 1.+eR_;
      double sq_theta = sqrt(theta_n);
      double lambda = (4./3.)*sedF2*rhoSolid*grain_*gs_0*eRp1*sq_theta/sq_pi;

      return lambda;
    }


    
    
      
    


    inline double tausc_term_theta(
		      double sedF, // IN
		      double rhoSolid, //IN
		      double theta_n, //IN
		      double du_dx,
		      double du_dy,
		      double du_dz,
		      double dv_dx,
		      double dv_dy,
		      double dv_dz,
		      double dw_dx,
		      double dw_dy,
		      double dw_dz)
		     
			   
    {		   

      double la = l_sc(sedF, rhoSolid, theta_n);
      double mu = mu_sc(sedF, rhoSolid, theta_n);

      
      double divU = du_dx + dv_dy + dw_dz;
      double t11 = 2*mu*du_dx + (la - (2./3.)*mu)*divU ;
      double t22 = 2*mu*dv_dy + (la - (2./3.)*mu)*divU ;
      double t33 = 2*mu*dw_dz + (la - (2./3.)*mu)*divU ;
      double t13 = mu*(du_dz + dw_dx);
      double t23 = mu*(dv_dz + dw_dy);
      double t12 = mu*(du_dy + dv_dx);
      // No need to define t31, t32 and t21 as tensor is symmetric

      double term = t11 * du_dx + t22 * dv_dy + t33 * dw_dz;
      term +=  t12*(du_dy +dv_dx);
      term +=  t13*(du_dz +dw_dx);
      term +=  t23*(dv_dz +dw_dy);						   

      return term* (2./3.) / (rhoSolid*sedF);

    }

    inline double  gamma_s(  double sedF,
			     double rhoSolid,
			     double theta_n,
			     double theta_np1,
			     double du_dx,
			     double dv_dy,
			     double dw_dz)

			      
    {
      double sq_pi = (sqrt(M_PI));
      double sq_theta = sqrt(theta_n);
      double divU = du_dx + dv_dy + dw_dz;
      double gamma_s =  - 2*(1-eR_*eR_)*sedF*gs0(sedF)*(4.*sq_theta/(sq_pi*grain_) - divU)*theta_np1;
      return gamma_s;
      
    }
    inline double  dgamma_s_dtheta(  double sedF,
			     double rhoSolid,
			     double theta_n,
			     double du_dx,
			     double dv_dy,
			     double dw_dz)

			      
    {
      double sq_pi = (sqrt(M_PI));
      double sq_theta = sqrt(theta_n);
      double divU = du_dx + dv_dy + dw_dz;
      double gamma_s =  - 2*(1-eR_*eR_)*sedF*gs0(sedF)*(4.*sq_theta/(sq_pi*grain_) - divU);
      return gamma_s;
      
    }
    inline double  xt_jint1(  double sedF,
			   double rhoFluid,
			   double rhoSolid,
			   const proteus::pyarray<double>& uFluid,
			   const proteus::pyarray<double>& uSolid,
			   double kappa,
			   double epsilon,
			   double theta,
			   double nu)
    {
        return jint1(sedF,
                     rhoFluid,
                     rhoSolid,
                     uFluid.data(),
                     uSolid.data(),
                     kappa,
                     epsilon,
                     theta,
                     nu);
    }
    
    inline double  jint1(  double sedF,
			   double rhoFluid,
			   double rhoSolid,
			   const double uFluid[nSpace],
			   const double uSolid[nSpace],
			   double kappa,
			   double epsilon,
			   double theta,
			   double nu)		      
    {
      
      double beta = betaCoeff(sedF,rhoFluid,uFluid,uSolid,nu)+small_;
      double gs = gs0(sedF)+small_;
      double l_c = sqrt(M_PI)*grain_/(24.*(sedF+small_)*gs);
      double t_p = rhoSolid/beta;
      double t_c = l_c/(sqrt(theta) + small_);
      double t_l = 0.165*kappa/(epsilon + small_);
      double t_cl = std::min(t_c,t_l);
      double alpha= t_cl/(t_cl + t_p);

      double Jint1 = 4. * alpha * beta * kappa /( 3.* rhoSolid)  ;
      return Jint1;

    }
    inline double  xt_jint2(  double sedF,
			   double rhoFluid,
			   double rhoSolid,
			   const proteus::pyarray<double>& uFluid,
			   const proteus::pyarray<double>& uSolid,
			   double theta,
			   double nu)

    {
        return jint2(sedF,
                     rhoFluid,
                     rhoSolid,
                     uFluid.data(),
                     uSolid.data(),
                     theta,
                     nu);
    }
    
    inline double  jint2(  double sedF,
			   double rhoFluid,
			   double rhoSolid,
			   const double uFluid[nSpace],
			   const double uSolid[nSpace],
			   double theta,
			   double nu)		      
    {
      
      double beta = betaCoeff(sedF,rhoFluid,uFluid,uSolid,nu)+small_;
      return - 2*beta*theta/rhoSolid;

    }

    inline double  xt_djint2_dtheta(  double sedF,
			   double rhoFluid,
			   double rhoSolid,
			   const proteus::pyarray<double>& uFluid,
			   const proteus::pyarray<double>& uSolid,
			   double nu)
    {
        return djint2_dtheta(sedF,
                             rhoFluid,
                             rhoSolid,
                             uFluid.data(),
                             uSolid.data(),
                             nu);
    }

    inline double  djint2_dtheta(  double sedF,
			   double rhoFluid,
			   double rhoSolid,
			   const double uFluid[nSpace],
			   const double uSolid[nSpace],
			   double nu)      
    {
      
      double beta = betaCoeff(sedF,rhoFluid,uFluid,uSolid,nu)+small_;
      return - 2*beta/rhoSolid;

    }

    inline double k_diff(		      double sedF, 
					      double rhoSolid, 
					      double theta )

    {
      double gs_0 = gs0(sedF);
      double sq_pi = (sqrt(M_PI));
      double sedF2 = sedF * sedF;
      double eRp1 = 1.+eR_;
      double sq_theta = sqrt(theta);
      double k_diff = rhoSolid*grain_*sq_theta*( 2.*sedF2*gs_0*eRp1/(sq_pi) + (0.5625)*sq_pi*sedF2*gs_0*eRp1 + (0.9375)*sq_pi*sedF + (0.390625)*sq_pi/(gs_0*eRp1) );


      return k_diff;
    }
   
    inline double p_friction(double sedF)

    {
      double pf = 0.;
      if (sedF > frFraction_)
	{
	  double sedLim =std::min(sedF,vos_limiter_);
	  pf =fContact_*pow(sedLim-frFraction_,mContact_) / ( pow(maxFraction_ - sedLim,nContact_) + small_);
	} 

      return pf;

    }

    
    inline double gradp_friction(double sedF)

    {
      double coeff = 0;
      double pf = p_friction(sedF);
      if (sedF > frFraction_ ) 
	{
	  double sedLim =std::min(sedF,vos_limiter_);

	  double den1 = (sedLim - frFraction_);
	  double den2 = (maxFraction_ - sedLim);

	  coeff = pf *( (mContact_/den1) + (nContact_/den2) );
	}
	  
      return coeff;
    }
    inline double mu_fr(double sedF,
		      double du_dx,
		      double du_dy,
		      double du_dz,
		      double dv_dx,
		      double dv_dy,
		      double dv_dz,
		      double dw_dx,
		      double dw_dy,
			  double dw_dz)
    {
      double divU = du_dx + dv_dy + dw_dz;
      double pf = p_friction(sedF);
      double s11 = du_dx - (1./nSpace)*divU;
      double s22 = dv_dy - (1./nSpace)*divU;
      double s33 = dw_dz - (1./nSpace)*divU;
      double s12 = 0.5*(du_dy + dv_dx);
      double s13 = 0.5*(du_dz + dw_dx);
      double s23 = 0.5*(dv_dz + dw_dy);
      double sumS = s11*s11 + s22*s22 + s33*s33 + 2.*s12*s12 + 2.*s13*s13 + 2.*s23*s23;
      double mu_sf = 0.0;
      double mu_fr = 0.0;
      if (sedF > frFraction_) 
	{
	  mu_sf = pf * sqrt(2.) * sin(angFriction_) / (2 * sqrt(sumS) + small_);
	  mu_fr = std::min(mu_sf,mu_fr_limiter_); 
	} 
//      if (sedF  > 0.0 )
//	{
//     printf("sedF --> %2.20f", sedF);
//	} 
//      if (mu_sf  > 0.0 )
//	{
//     printf("mu_sf --> %2.20f", mu_sf);
//	} 
//      if (mu_fr  > 0.0 )
//	{
//     printf("mu_fr --> %2.20f", mu_fr);
//	} 
//      if (pf  > 0.0 )
//	{
//     printf("pf --> %2.20f", pf);
//	} 

      return mu_fr;
    }






    /*
    inline double  diffusion_theta_rhs(  double sedF,
					 double rhoSolid,
				     double theta_n,
				     double dtheta_dx,
				     double dtheta_dy,
				     double dtheta_dz
			      )
    {
      
      double gs_0 = gs0(sedF);
      double sq_pi = (sqrt(M_PI));
      double sedF2 = sedF * sedF;
      double eRp1 = 1.+eR_;
      double sq_theta = sqrt(theta_n);

      double divTheta = dtheta_dx + dtheta_dy + dtheta_dz; 

      double k_diff = k_diff(rhoSolid);

      return kappa*divTheta;
      
    }


 
      
      }*/



    

    inline proteus::pyarray<double> xt_mIntFluid(double sedF,
			   double rhoFluid,
			      const proteus::pyarray<double>& uFluid_n, //Fluid velocity
			      const proteus::pyarray<double>& uSolid_n, //Sediment velocity
			      const proteus::pyarray<double>& uFluid_np1, //Fluid velocity
			      double nu, //Kinematic viscosity
			      double nuT, //Turbulent viscosity
			      const proteus::pyarray<double>& gradc
			      )
    {
        auto mint2 = proteus::pyarray<double>::from_shape({2});
        mIntFluid(mint2.data(),
                  sedF,
                  rhoFluid,
                  uFluid_n.data(),
                  uSolid_n.data(),
                  uFluid_np1.data(),
                  nu,
                  nuT,
                  gradc.data());
        return mint2;
    }

    inline void  mIntFluid( double * mint2, double sedF,
			   double rhoFluid,
			      const double uFluid_n[nSpace], //Fluid velocity
			      const double uSolid_n[nSpace], //Sediment velocity
			      const double uFluid_np1[nSpace], //Fluid velocity
			      double nu, //Kinematic viscosity
			      double nuT, //Turbulent viscosity
			      const double gradc[nSpace]
			      )
    {
      
      double beta = betaCoeff(sedF,rhoFluid,uFluid_n,uSolid_n,nu);
      for  (int ii=0; ii<nSpace;  ii++)
	{
	  mint2[ii] =  -sedF*beta*(uFluid_np1[ii])/(rhoFluid * (1. - sedF)) ;
	    }
      
      }

    inline proteus::pyarray<double> xt_mIntSolid(double sedF,
			   double rhoFluid,
			      const proteus::pyarray<double>& uFluid_n, //Fluid velocity
			      const proteus::pyarray<double>& uSolid_n, //Sediment velocity
			      const proteus::pyarray<double>& uSolid_np1, //Sediment velocity
			      double nu, //Kinematic viscosity
			      double nuT, //Turbulent viscosity
			      const proteus::pyarray<double>& gradc
			      )
    {
        auto mint2 = proteus::pyarray<double>::from_shape({2});
        mIntSolid(mint2.data(),
                  sedF,
                  rhoFluid,
                  uFluid_n.data(),
                  uSolid_n.data(),
                  uSolid_np1.data(),
                  nu,
                  nuT,
                  gradc.data());
        return mint2;
    }

    inline void mIntSolid(   double * mint2, double sedF,
			   double rhoFluid,
			      const double uFluid_n[nSpace], //Fluid velocity
			      const double uSolid_n[nSpace], //Sediment velocity
			      const double uSolid_np1[nSpace], //Sediment velocity
			      double nu, //Kinematic viscosity
			      double nuT, //Turbulent viscosity
			      const double gradc[nSpace]
			      )
    {

      double beta = betaCoeff(sedF,rhoFluid,uFluid_n,uSolid_n,nu);
      for  (int ii=0; ii<nSpace;  ii++)
	{
	  mint2[ii] = -sedF*beta*(-uSolid_np1[ii])/(rhoFluid * (1. - sedF)) ;
	    }

    }
    inline proteus::pyarray<double> xt_mIntgradC(double sedF,
			   double rhoFluid,
				  const proteus::pyarray<double>& uFluid_n, //Fluid velocity
				  const proteus::pyarray<double>& uSolid_n, //Sediment velocity
				  double nu, //Kinematic viscosity
				  double nuT, //Turbulent viscosity
				  const proteus::pyarray<double>& gradc
			      )
    {
        
        auto mint2 = proteus::pyarray<double>::from_shape({2});
        mIntgradC(mint2.data(),
                  sedF,
                  rhoFluid,
                  uFluid_n.data(),
                  uSolid_n.data(),
                  nu,
                  nuT,
                  gradc.data());
        return mint2;
    }

    inline void mIntgradC(double * mint2,  double sedF,
			   double rhoFluid,
				  const double uFluid_n[nSpace], //Fluid velocity
				  const double uSolid_n[nSpace], //Sediment velocity
				  double nu, //Kinematic viscosity
				  double nuT, //Turbulent viscosity
				  const double gradc[nSpace]
			      )

    {

      double beta = betaCoeff(sedF,rhoFluid,uFluid_n,uSolid_n,nu);
      for  (int ii=0; ii<nSpace;  ii++)
	{
	  mint2[ii] = - sedF*beta*nuT*gradc[ii]/sigmaC_/(rhoFluid * (1. - sedF));
	    }

     
    }


   
    inline double  xt_dmInt_duFluid
                            (  double sedF,
			   double rhoFluid,
			      const proteus::pyarray<double>& uFluid_n, //Fluid velocity
			      const proteus::pyarray<double>& uSolid_n, //Sediment velocity
			      double nu //Kinematic viscosity

			      )
    {
        return dmInt_duFluid(sedF,
                             rhoFluid,
                             uFluid_n.data(),
                             uSolid_n.data(),
                             nu);
    }

    inline double  dmInt_duFluid
                            (  double sedF,
			   double rhoFluid,
			      const double uFluid_n[nSpace], //Fluid velocity
			      const double uSolid_n[nSpace], //Sediment velocity
			      double nu //Kinematic viscosity

			      )
    {
      return -sedF*betaCoeff(sedF,rhoFluid,uFluid_n,uSolid_n,nu)/(rhoFluid * (1. - sedF));

    }


       inline double  xt_dmInt_duSolid
                            (  double sedF,
			   double rhoFluid,
			      const proteus::pyarray<double>& uFluid_n, //Fluid velocity
			      const proteus::pyarray<double>& uSolid_n, //Sediment velocity
			      double nu //Kinematic viscosity

			      )
    {
        return dmInt_duSolid(sedF,
                             rhoFluid,
                             uFluid_n.data(),
                             uSolid_n.data(),
                             nu);
    }

       inline double  dmInt_duSolid
                            (  double sedF,
			   double rhoFluid,
			      const double uFluid_n[nSpace], //Fluid velocity
			      const double uSolid_n[nSpace], //Sediment velocity
			      double nu //Kinematic viscosity

			      )
    {

      return +sedF*betaCoeff(sedF,rhoFluid,uFluid_n,uSolid_n,nu)/(rhoFluid * (1. - sedF));
    }

       inline double p_s(		      double sedF, 
					      double rhoSolid,
					      double theta,
					      double du_dx,
					      double du_dy,
					      double du_dz,
					      double dv_dx,
					      double dv_dy,
					      double dv_dz,
					      double dw_dx,
					      double dw_dy,
					      double dw_dz)

      
    {
      double divU = du_dx + dv_dy + dw_dz;
      double mf = mu_fr(sedF, du_dx, du_dy, du_dz, dv_dx, dv_dy, dv_dz, dw_dx, dw_dy, dw_dz);
      double msc = mu_sc(sedF, rhoSolid, theta );
      double lam = l_sc(sedF, rhoSolid, theta );
      double pcorr = ( (2./3.)*(msc + mf) - lam ) * divU;
	return p_friction(sedF) + psc(sedF, rhoSolid, theta) + pcorr ;
    }

 

    
    

  double aDarcy_;
  double betaForch_;
  double grain_; 
  double packFraction_;
  double packMargin_;
  double frFraction_;
  double maxFraction_;
  double sigmaC_;
  double C3e_;
  double C4e_;
  double eR_;
  double fContact_;
  double mContact_;
  double nContact_;
  double angFriction_;
  double small_;
  double notSoLarge_;
  double large_;
  double vos_limiter_;
  double mu_fr_limiter_;

};

  typedef cppHsuSedStress<2> cppHsuSedStress2D;
}

#endif

