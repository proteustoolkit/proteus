#ifndef RANS3PSed2D_H
#define RANS3PSed2D_H
#include <cmath>
#include <iostream>
#include "CompKernel.h"
#include "ModelFactory.h"
#include "SedClosure.h"
#include "ArgumentsDict.h"
#include "pyarray.h"

namespace py = pybind11;

const  double DM=0.0;//1-mesh conservation and divergence, 0 - weak div(v) only
const  double DM2=0.0;//1-point-wise mesh volume strong-residual, 0 - div(v) only
const  double DM3=1.0;//1-point-wise divergence, 0-point-wise rate of volume change
#define DRAG_FAC 1.0
#define TURB_FORCE_FAC 1.0

namespace proteus
{
  class cppRANS3PSed2D_base
  {
  public:
    virtual ~cppRANS3PSed2D_base(){}
    virtual void setSedClosure(double aDarcy,
                               double betaForch,
                               double grain,
                               double packFraction,
                               double packMargin,
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
                               double mu_fr_limiter){}

    virtual void calculateResidual(arguments_dict& args)=0;
    virtual void calculateJacobian(arguments_dict& args)=0;
    virtual void calculateVelocityAverage(arguments_dict& args) = 0;
  };

  template<class CompKernelType,
    int nSpace,
    int nQuadraturePoints_element,
    int nDOF_mesh_trial_element,
    int nDOF_trial_element,
    int nDOF_test_element,
    int nQuadraturePoints_elementBoundary>
    class cppRANS3PSed2D : public cppRANS3PSed2D_base
    {
    public:
      cppHsuSedStress<2> closure;
      const int nDOF_test_X_trial_element;
      CompKernelType ck;
    cppRANS3PSed2D():
      closure(150.0,
              0.0,
              0.0102,
              0.2,
              0.01,
              0.635,
              0.57,
              1.1,
              1.2,
              1.0,
              0.8,
              0.02,
              2.0,
              5.0,
              M_PI/6.,
              0.05,
              1.00),
        nDOF_test_X_trial_element(nDOF_test_element*nDOF_trial_element),
        ck()
          {
          }

      void setSedClosure(double aDarcy,
                         double betaForch,
                         double grain,
                         double packFraction,
                         double packMargin,
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
                         double mu_fr_limiter)
      {
        closure = cppHsuSedStress<2>(aDarcy,
                                     betaForch,
                                     grain,
                                     packFraction,
                                     packMargin,
                                     maxFraction,
                                     frFraction,
                                     sigmaC,
                                     C3e,
                                     C4e,
                                     eR,
                                     fContact,
                                     mContact,
                                     nContact,
                                     angFriction,
                                     vos_limiter,
                                     mu_fr_limiter);
      }

      inline double smoothedHeaviside(double eps, double phi)
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
    
      inline double smoothedHeaviside_integral(double eps, double phi)
      {
        double HI;
        if (phi > eps)
          {
            HI= phi - eps                                                       \
              + 0.5*(eps + 0.5*eps*eps/eps - eps*cos(M_PI*eps/eps)/(M_PI*M_PI)) \
              - 0.5*((-eps) + 0.5*(-eps)*(-eps)/eps - eps*cos(M_PI*(-eps)/eps)/(M_PI*M_PI));
          }
        else if (phi < -eps)
          {
            HI=0.0;
          }
        else
          {
            HI = 0.5*(phi + 0.5*phi*phi/eps - eps*cos(M_PI*phi/eps)/(M_PI*M_PI)) \
              - 0.5*((-eps) + 0.5*(-eps)*(-eps)/eps - eps*cos(M_PI*(-eps)/eps)/(M_PI*M_PI));
          }
        return HI;
      }
 
      inline double smoothedDirac(double eps, double phi)
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

      inline
        void evaluateCoefficients(const double eps_rho,
                                  const double eps_mu,
                                  const double sigma,
                                  const double rho_0,
                                  double nu_0,
                                  const double rho_1,
                                  double nu_1,
                                  const double rho_s,
                                  const double h_e,
                                  const double smagorinskyConstant,
                                  const int turbulenceClosureModel,
                                  const double g[nSpace],
                                  const double useVF,
                                  const double& vf,
                                  const double& phi,
                                  const double n[nSpace],
                                  const double& kappa,
                                  const double vos,//VRANS specific
                                  const double& p,
                                  const double grad_p[nSpace],
                                  const double grad_u[nSpace],
                                  const double grad_v[nSpace],
                                  const double grad_w[nSpace],
                                  const double& u,
                                  const double& v,
                                  const double& w,
                                  const double& uStar,
                                  const double& vStar,
                                  const double& wStar,
                                  double& eddy_viscosity,
                                  double& mom_u_acc,
                                  double& dmom_u_acc_u,
                                  double& mom_v_acc,
                                  double& dmom_v_acc_v,
                                  double& mom_w_acc,
                                  double& dmom_w_acc_w,
                                  double mass_adv[nSpace],
                                  double dmass_adv_u[nSpace],
                                  double dmass_adv_v[nSpace],
                                  double dmass_adv_w[nSpace],
                                  double mom_u_adv[nSpace],
                                  double dmom_u_adv_u[nSpace],
                                  double dmom_u_adv_v[nSpace],
                                  double dmom_u_adv_w[nSpace],
                                  double mom_v_adv[nSpace],
                                  double dmom_v_adv_u[nSpace],
                                  double dmom_v_adv_v[nSpace],
                                  double dmom_v_adv_w[nSpace],
                                  double mom_w_adv[nSpace],
                                  double dmom_w_adv_u[nSpace],
                                  double dmom_w_adv_v[nSpace],
                                  double dmom_w_adv_w[nSpace],
                                  double mom_uu_diff_ten[nSpace],
                                  double mom_vv_diff_ten[nSpace],
                                  double mom_ww_diff_ten[nSpace],
                                  double mom_uv_diff_ten[1],
                                  double mom_uw_diff_ten[1],
                                  double mom_vu_diff_ten[1],
                                  double mom_vw_diff_ten[1],
                                  double mom_wu_diff_ten[1],
                                  double mom_wv_diff_ten[1],
                                  double& mom_u_source,
                                  double& mom_v_source,
                                  double& mom_w_source,
                                  double& mom_u_ham,
                                  double dmom_u_ham_grad_p[nSpace],
                                  double dmom_u_ham_grad_u[nSpace],
                                  double& mom_v_ham,
                                  double dmom_v_ham_grad_p[nSpace],
                                  double dmom_v_ham_grad_v[nSpace],
                                  double& mom_w_ham,
                                  double dmom_w_ham_grad_p[nSpace],
                                  double dmom_w_ham_grad_w[nSpace])
      {
        double rho,rho_f,H_rho;
        H_rho = (1.0-useVF)*smoothedHeaviside(eps_rho,phi) + useVF*fmin(1.0,fmax(0.0,vf));
	//        H_mu = (1.0-useVF)*smoothedHeaviside(eps_mu,phi) + useVF*fmin(1.0,fmax(0.0,vf));
	//        d_mu = (1.0-useVF)*smoothedDirac(eps_mu,phi);
  
        //calculate eddy viscosity - NO NEED FOR SOLID PHASE Coefficients
	/*        switch (turbulenceClosureModel)
          {
            double norm_S;
          case 1:
            {
              norm_S = sqrt(2.0*(grad_u[0]*grad_u[0] + grad_v[1]*grad_v[1] +
                                 0.5*(grad_u[1]+grad_v[0])*(grad_u[1]+grad_v[0])));
              nu_t0 = smagorinskyConstant*smagorinskyConstant*h_e*h_e*norm_S;
              nu_t1 = smagorinskyConstant*smagorinskyConstant*h_e*h_e*norm_S;
            }
          case 2:
            {
              double re_0,cs_0=0.0,re_1,cs_1=0.0;
              norm_S = sqrt(2.0*(grad_u[0]*grad_u[0] + grad_v[1]*grad_v[1] +
                                 0.5*(grad_u[1]+grad_v[0])*(grad_u[1]+grad_v[0])));
              re_0 = h_e*h_e*norm_S/nu_0;
              if (re_0 > 1.0)
                cs_0=0.027*pow(10.0,-3.23*pow(re_0,-0.92));
              nu_t0 = cs_0*h_e*h_e*norm_S;
              re_1 = h_e*h_e*norm_S/nu_1;
              if (re_1 > 1.0)
                cs_1=0.027*pow(10.0,-3.23*pow(re_1,-0.92));
              nu_t1 = cs_1*h_e*h_e*norm_S;
            }
          }*/
      
        rho = rho_s;
        rho_f = rho_0*(1.0-H_rho)+rho_1*H_rho;
	//        nu_t= nu_t0*(1.0-H_mu)+nu_t1*H_mu;
	//        nu = nu_0*(1.0-H_mu)+nu_1*H_mu;
        //nu  = nu_0*(1.0-H_mu)+nu_1*H_mu;
	//        nu += nu_t;
        //mu  = rho_0*nu_0*(1.0-H_mu)+rho_1*nu_1*H_mu;
	//        mu = rho_0*nu_0*(1.0-H_mu)+rho_1*nu_1*H_mu;

	//        eddy_viscosity = rho_0*nu_t0*(1.0-H_mu)+rho_1*nu_t1*H_mu;

        // mass (volume accumulation)
        //..hardwired
      
        //u momentum accumulation
        mom_u_acc=u;//trick for non-conservative form
        dmom_u_acc_u=rho;
  
        //v momentum accumulation
        mom_v_acc=v;
        dmom_v_acc_v=rho;
  
        /* //w momentum accumulation */
        /* mom_w_acc=w; */
        /* dmom_w_acc_w=rho*vos; */

        //mass advective flux
        mass_adv[0]=vos*u;
        mass_adv[1]=vos*v;
        /* mass_adv[2]=qw; */
  
        dmass_adv_u[0]=vos;
        dmass_adv_u[1]=0.0;
        /* dmass_adv_u[2]=0.0; */
      
        dmass_adv_v[0]=0.0;
        dmass_adv_v[1]=vos;
        /* dmass_adv_v[2]=0.0; */
      
        /* dmass_adv_w[0]=0.0; */
        /* dmass_adv_w[1]=0.0; */
        /* dmass_adv_w[2]=1.0; */

        //advection switched to non-conservative form but could be used for mesh motion...
        //u momentum advective flux
        mom_u_adv[0]=0.0;
        mom_u_adv[1]=0.0;
        /* mom_u_adv[2]=0.0; */
      
        dmom_u_adv_u[0]=0.0;
        dmom_u_adv_u[1]=0.0;
        /* dmom_u_adv_u[2]=0.0; */
      
        dmom_u_adv_v[0]=0.0;
        dmom_u_adv_v[1]=0.0;
        /* dmom_u_adv_v[2]=0.0; */
      
        /* dmom_u_adv_w[0]=0.0; */
        /* dmom_u_adv_w[1]=0.0; */
        /* dmom_u_adv_w[2]=0.0; */
      
        //v momentum advective_flux
        mom_v_adv[0]=0.0;
        mom_v_adv[1]=0.0;
        /* mom_v_adv[2]=0.0; */
      
        dmom_v_adv_u[0]=0.0;
        dmom_v_adv_u[1]=0.0;
        /* dmom_v_adv_u[2]=0.0; */
      
        /* dmom_v_adv_w[0]=0.0; */
        /* dmom_v_adv_w[1]=0.0; */
        /* dmom_v_adv_w[2]=0.0; */
      
        dmom_v_adv_v[0]=0.0;
        dmom_v_adv_v[1]=0.0;
        /* dmom_v_adv_v[2]=0.0; */
      
        /* //w momentum advective_flux */
        /* mom_w_adv[0]=0.0; */
        /* mom_w_adv[1]=0.0; */
        /* mom_w_adv[2]=0.0; */
      
        /* dmom_w_adv_u[0]=0.0; */
        /* dmom_w_adv_u[1]=0.0; */
        /* dmom_w_adv_u[2]=0.0; */
      
        /* dmom_w_adv_v[0]=0.0; */
        /* dmom_w_adv_v[1]=0.0; */
        /* dmom_w_adv_v[2]=0.0; */
      
        /* dmom_w_adv_w[0]=0.0; */
        /* dmom_w_adv_w[1]=0.0; */
        /* dmom_w_adv_w[2]=0.0; */
      
        //u momentum diffusion tensor
        mom_uu_diff_ten[0] =0.0;
        mom_uu_diff_ten[1] =0.0;
        /* mom_uu_diff_ten[2] = vos*nu; */
  
        mom_uv_diff_ten[0]=0.0;
  
        /* mom_uw_diff_ten[0]=vos*nu; */
  
        //v momentum diffusion tensor
        mom_vv_diff_ten[0] =0.0;
        mom_vv_diff_ten[1] =0.0;
        /* mom_vv_diff_ten[2] = vos*mu; */
  
        mom_vu_diff_ten[0]=0.0;
  
        /* mom_vw_diff_ten[0]=vos*mu; */
  
        /* //w momentum diffusion tensor */
        /* mom_ww_diff_ten[0] = vos*mu; */
        /* mom_ww_diff_ten[1] = vos*mu; */
        /* mom_ww_diff_ten[2] = vos*2.0*mu; */
  
        /* mom_wu_diff_ten[0]=vos*mu; */
  
        /* mom_wv_diff_ten[0]=vos*mu; */
  
        //momentum sources
        mom_u_source = -rho*g[0];// - vos*d_mu*sigma*kappa*n[0]/(rho*(norm_n+1.0e-8));
        mom_v_source = -rho*g[1];// - vos*d_mu*sigma*kappa*n[1]/(rho*(norm_n+1.0e-8));
        /* mom_w_source = -vos*g[2];// - vos*d_mu*sigma*kappa*n[2]/(rho*(norm_n+1.0e-8)); */
	if(vos > closure.frFraction_)
	  {
	    mom_u_source += (rho-rho_f)*g[0];
	    mom_v_source += (rho-rho_f)*g[1];
 
	  }
   
        //u momentum Hamiltonian (pressure)
        mom_u_ham = grad_p[0];
        dmom_u_ham_grad_p[0]=1.0;
        dmom_u_ham_grad_p[1]=0.0;
        /* dmom_u_ham_grad_p[2]=0.0; */
          
        //v momentum Hamiltonian (pressure)
        mom_v_ham = grad_p[1];
        dmom_v_ham_grad_p[0]=0.0;
        dmom_v_ham_grad_p[1]=1.0;
        /* dmom_v_ham_grad_p[2]=0.0; */

        /* //w momentum Hamiltonian (pressure) */
        /* mom_w_ham = vos*grad_p[2]; */
        /* dmom_w_ham_grad_p[0]=0.0; */
        /* dmom_w_ham_grad_p[1]=0.0; */
        /* dmom_w_ham_grad_p[2]=vos/rho; */

        //u momentum Hamiltonian (advection)
        mom_u_ham +=  rho*(uStar*grad_u[0]+vStar*grad_u[1]);
        dmom_u_ham_grad_u[0]= rho*uStar;
        dmom_u_ham_grad_u[1]= rho*vStar;
        /* dmom_u_ham_grad_u[2]=vos*wStar; */
  
        //v momentum Hamiltonian (advection)
        mom_v_ham += rho*(uStar*grad_v[0]+vStar*grad_v[1]);
        dmom_v_ham_grad_v[0]= rho*uStar;
        dmom_v_ham_grad_v[1]= rho*vStar;
        /* dmom_v_ham_grad_v[2]=vos*wStar; */
      
        /* //w momentum Hamiltonian (advection) */
        /* mom_w_ham = vos*(uStar*grad_w[0]+vStar*grad_w[1]+wStar*grad_w[2]); */
        /* dmom_w_ham_grad_w[0]=vos*uStar; */
        /* dmom_w_ham_grad_w[1]=vos*vStar; */
        /* dmom_w_ham_grad_w[2]=vos*wStar; */
      }
      //VRANS specific
      inline
        void updateDarcyForchheimerTerms_Ergun(/* const double linearDragFactor, */
                                               /* const double nonlinearDragFactor, */
                                               /* const double vos, */
                                               /* const double meanGrainSize, */
                                               const double alpha,
                                               const double beta,
                                               const double eps_rho,
                                               const double eps_mu,
                                               const double rho_0,
                                               const double nu_0,
                                               const double rho_1,
                                               const double nu_1,
                                               double nu_t,
                                               const double useVF,
                                               const double vf,
                                               const double phi,
                                               const double u,
                                               const double v,
                                               const double w,
                                               const double uStar,
                                               const double vStar,
                                               const double wStar,
                                               const double eps_s,
                                               const double vos,
                                               const double u_f,
                                               const double v_f,
                                               const double w_f,
                                               const double uStar_f,
                                               const double vStar_f,
                                               const double wStar_f,
                                               double& mom_u_source,
                                               double& mom_v_source,
                                               double& mom_w_source,
                                               double dmom_u_source[nSpace],
                                               double dmom_v_source[nSpace],
                                               double dmom_w_source[nSpace],
                                               double gradC_x,
                                               double gradC_y,
                                               double gradC_z)
      {
        double rhoFluid, nuFluid,H_mu,viscosity;
        H_mu = (1.0-useVF)*smoothedHeaviside(eps_mu,phi)+useVF*fmin(1.0,fmax(0.0,vf));
        nuFluid  = nu_0*(1.0-H_mu)+nu_1*H_mu;
        rhoFluid  = rho_0*(1.0-H_mu)+rho_1*H_mu;
        //gco kinematic viscosity used, in sedclosure betaterm is multiplied by fluidDensity
        viscosity = nuFluid;//mu; gco check
        //vos is sediment fraction in this case - gco check

        double solid_velocity[2]={uStar,vStar}, fluid_velocity[2]={uStar_f, vStar_f};
        double new_beta =    closure.betaCoeff(vos,
                                               rhoFluid,
                                               fluid_velocity,
                                               solid_velocity,
                                               viscosity)*DRAG_FAC;
      
        //new_beta = 254800.0;//hack fall velocity of 0.1 with no pressure gradient
        double beta2 = 156976.4;//hack, fall velocity of 0.1 with hydrostatic water
        double one_by_vos = 2.0*vos/(vos*vos + fmax(1.0e-8,vos*vos));
        
        //new_beta/=rhoFluid;
        //std::cout<<"total "<<(1.0-phi_s)*new_beta<<std::endl;
        mom_u_source +=  new_beta*((u - u_f) + TURB_FORCE_FAC*DRAG_FAC*one_by_vos*nu_t*gradC_x/closure.sigmaC_) + (1.0-DRAG_FAC)*beta2*(u-u_f);

        mom_v_source +=  new_beta*((v - v_f) + TURB_FORCE_FAC*DRAG_FAC*one_by_vos*nu_t*gradC_y/closure.sigmaC_) + (1.0-DRAG_FAC)*beta2*(v-v_f);

        /* mom_w_source += new_beta*(w-w_s); */

        dmom_u_source[0] = new_beta + (1.0-DRAG_FAC)*beta2;
        dmom_u_source[1] = 0.0;
        /* dmom_u_source[2] = 0.0; */

        dmom_v_source[0] = 0.0;
        dmom_v_source[1] = new_beta + (1.0-DRAG_FAC)*beta2;
        /*dmom_v_source[2] = 0.0;*/

        /*      dmom_w_source[0] = 0.0;
                dmom_w_source[1] = 0.0;
                dmom_w_source[2] = new_beta;*/
      }


      inline void updatePenaltyForPacking(const double vos,
                                          const double u,
                                          const double v,
                                          const double w,
                                          double& mom_u_source,
                                          double& mom_v_source,
                                          double& mom_w_source,
                                          double dmom_u_source[nSpace],
                                          double dmom_v_source[nSpace],
                                          double dmom_w_source[nSpace])
				   
      {
        double meanPack = (closure.maxFraction_ + closure.vos_limiter_)/2.;
        double epsPack = (closure.maxFraction_ - closure.vos_limiter_)/2.;
        double dVos = vos - meanPack;
        double sigma = smoothedHeaviside( epsPack, dVos);
        double packPenalty = 1e6;
        mom_u_source += sigma * packPenalty*u;
        mom_v_source += sigma * packPenalty*v;
        dmom_u_source[0] += sigma * packPenalty;
        dmom_v_source[1] += sigma * packPenalty;

      }  


    
      inline
        void updateFrictionalPressure(const double vos,
                                      const double grad_vos[nSpace],
                                      double& mom_u_source,
                                      double& mom_v_source,
                                      double& mom_w_source)
      {
        double coeff = closure.gradp_friction(vos);
        double one_by_vos = 2.0*vos/(vos*vos + fmax(1.0e-8,vos*vos));
   
        mom_u_source += coeff * grad_vos[0]*one_by_vos;
        mom_v_source += coeff * grad_vos[1]*one_by_vos;
        //mom_w_source += coeff * grad_vos[2];

	}  

      inline
        void updateFrictionalStress(const double LAG_MU_FR,
                                    const double mu_fr_last,
                                    double& mu_fr_new,
                                    const double vos,
                                    const double eps_rho,
                                    const double eps_mu,
                                    const double rho_0,
                                    const double nu_0,
                                    const double rho_1,
                                    const double nu_1,
                                    const double rho_s,
                                    const double useVF,
                                    const double vf,
                                    const double phi,
                                    const double grad_u[nSpace],
                                    const double grad_v[nSpace],
                                    const double grad_w[nSpace], 
                                    double mom_uu_diff_ten[nSpace],
                                    double mom_uv_diff_ten[1],
                                    double mom_uw_diff_ten[1],
                                    double mom_vv_diff_ten[nSpace],
                                    double mom_vu_diff_ten[1],
                                    double mom_vw_diff_ten[1],
                                    double mom_ww_diff_ten[nSpace],
                                    double mom_wu_diff_ten[1],
                                    double mom_wv_diff_ten[1])
      {
        double one_by_vos = 2.0*vos/(vos*vos + fmax(1.0e-8,vos*vos));

        mu_fr_new = closure.mu_fr(vos,
                                  grad_u[0], grad_u[1], 0., 
                                  grad_v[0], grad_v[1], 0., 
                                  0., 0. , 0.)*one_by_vos; 
        double mu_fr = LAG_MU_FR*mu_fr_last + (1.0 - LAG_MU_FR)*mu_fr_new;

        mom_uu_diff_ten[0] += 2. * mu_fr * (2./3.); 
        mom_uu_diff_ten[1] += mu_fr;
        /*mom_uu_diff_ten[2] += mu_fr; */

        mom_uv_diff_ten[0] += mu_fr;
        /*mom_uw_diff_ten[0] += mu_fr; */

        mom_vv_diff_ten[0] += mu_fr; 
        mom_vv_diff_ten[1] += 2. * mu_fr * (2./3.) ;
        /*mom_vv_diff_ten[2] += mu_fr; */

        mom_vu_diff_ten[0] += mu_fr;
        /*mom_vw_diff_ten[0] += mu_fr; */

        /*mom_ww_diff_ten[0] += mu_fr;  */
        /*mom_ww_diff_ten[1] += mu_fr; */
        /*mom_ww_diff_ten[2] += 2. * mu_fr * (2./3.) ; */

        /*mom_wu_diff_ten[0] += mu_fr; */
        /*mom_wv_diff_ten[0] += mu_fr; */

      }




      inline void calculateSubgridError_tau(const double &hFactor,
                                            const double &elementDiameter,
                                            const double &dmt,
                                            const double &dm,
                                            const double df[nSpace],
                                            const double &a,
                                            const double &pfac,
                                            double &tau_v,
                                            double &tau_p,
                                            double &cfl)
      {
        double h, oneByAbsdt, density, viscosity, nrm_df;
        h = hFactor * elementDiameter;
        density = dm;
        viscosity = a;
        nrm_df = 0.0;
        for (int I = 0; I < nSpace; I++)
          {
            nrm_df += df[I] * df[I];
          }
	nrm_df = sqrt(nrm_df);
	if(density > 1e-8)
	  {
	    cfl = nrm_df / (fabs(h * density)); 
	  }
	else
	  {
	    cfl = nrm_df / fabs(h ); 
	  }
        oneByAbsdt = fabs(dmt);
        tau_v = 1.0 / (4.0 * viscosity / (h * h) + 2.0 * nrm_df / h + oneByAbsdt);
        tau_p = (4.0 * viscosity + 2.0 * nrm_df * h + oneByAbsdt * h * h) / pfac;
      }

      inline void calculateSubgridError_tau(const double &Ct_sge,
                                            const double &Cd_sge,
                                            const double G[nSpace * nSpace],
                                            const double &G_dd_G,
                                            const double &tr_G,
                                            const double &A0,
                                            const double Ai[nSpace],
                                            const double &Kij,
                                            const double &pfac,
                                            double &tau_v,
                                            double &tau_p,
                                            double &q_cfl)
      {
        double v_d_Gv = 0.0;
        for (int I = 0; I < nSpace; I++)
          for (int J = 0; J < nSpace; J++)
            v_d_Gv += Ai[I] * G[I * nSpace + J] * Ai[J];
        tau_v = 1.0 / sqrt(Ct_sge * A0 * A0 + v_d_Gv + Cd_sge * Kij * Kij * G_dd_G + 1.0e-12);
        tau_p = 1.0 / (pfac * tr_G * tau_v);
      }

      inline void calculateSubgridError_tauRes(const double &tau_p,
                                               const double &tau_v,
                                               const double &pdeResidualP,
                                               const double &pdeResidualU,
                                               const double &pdeResidualV,
                                               const double &pdeResidualW,
                                               double &subgridErrorP,
                                               double &subgridErrorU,
                                               double &subgridErrorV,
                                               double &subgridErrorW)
      {
        /* GLS pressure */
        subgridErrorP = -tau_p * pdeResidualP;
        /* GLS momentum */
        subgridErrorU = -tau_v * pdeResidualU;
        subgridErrorV = -tau_v * pdeResidualV;
        /* subgridErrorW = -tau_v*pdeResidualW; */
      }

      inline void calculateSubgridErrorDerivatives_tauRes(const double &tau_p,
                                                          const double &tau_v,
                                                          const double dpdeResidualP_du[nDOF_trial_element],
                                                          const double dpdeResidualP_dv[nDOF_trial_element],
                                                          const double dpdeResidualP_dw[nDOF_trial_element],
                                                          const double dpdeResidualU_dp[nDOF_trial_element],
                                                          const double dpdeResidualU_du[nDOF_trial_element],
                                                          const double dpdeResidualV_dp[nDOF_trial_element],
                                                          const double dpdeResidualV_dv[nDOF_trial_element],
                                                          const double dpdeResidualW_dp[nDOF_trial_element],
                                                          const double dpdeResidualW_dw[nDOF_trial_element],
                                                          double dsubgridErrorP_du[nDOF_trial_element],
                                                          double dsubgridErrorP_dv[nDOF_trial_element],
                                                          double dsubgridErrorP_dw[nDOF_trial_element],
                                                          double dsubgridErrorU_dp[nDOF_trial_element],
                                                          double dsubgridErrorU_du[nDOF_trial_element],
                                                          double dsubgridErrorV_dp[nDOF_trial_element],
                                                          double dsubgridErrorV_dv[nDOF_trial_element],
                                                          double dsubgridErrorW_dp[nDOF_trial_element],
                                                          double dsubgridErrorW_dw[nDOF_trial_element])
      {
        for (int j = 0; j < nDOF_trial_element; j++)
          {
            /* GLS pressure */
            dsubgridErrorP_du[j] = -tau_p * dpdeResidualP_du[j];
            dsubgridErrorP_dv[j] = -tau_p * dpdeResidualP_dv[j];
            /* dsubgridErrorP_dw[j] = -tau_p*dpdeResidualP_dw[j]; */
            /* GLS  momentum*/
            /* u */
            dsubgridErrorU_dp[j] = -tau_v * dpdeResidualU_dp[j];
            dsubgridErrorU_du[j] = -tau_v * dpdeResidualU_du[j];
            /* v */
            dsubgridErrorV_dp[j] = -tau_v * dpdeResidualV_dp[j];
            dsubgridErrorV_dv[j] = -tau_v * dpdeResidualV_dv[j];
            /* /\* w *\/ */
            /* dsubgridErrorW_dp[j] = -tau_v*dpdeResidualW_dp[j]; */
            /* dsubgridErrorW_dw[j] = -tau_v*dpdeResidualW_dw[j]; */
          }
      }

      inline
        void exteriorNumericalAdvectiveFlux(const int& isDOFBoundary_p,
                                            const int& isDOFBoundary_u,
                                            const int& isDOFBoundary_v,
                                            const int& isDOFBoundary_w,
                                            const int& isFluxBoundary_p,
                                            const int& isFluxBoundary_u,
                                            const int& isFluxBoundary_v,
                                            const int& isFluxBoundary_w,
                                            const double& oneByRho,
                                            const double& bc_oneByRho,
                                            const double n[nSpace],
                                            const double& vos,
                                            const double& bc_p,
                                            const double& bc_u,
                                            const double& bc_v,
                                            const double& bc_w,
                                            const double bc_f_mass[nSpace],
                                            const double bc_f_umom[nSpace],
                                            const double bc_f_vmom[nSpace],
                                            const double bc_f_wmom[nSpace],
                                            const double& bc_flux_mass,
                                            const double& bc_flux_umom,
                                            const double& bc_flux_vmom,
                                            const double& bc_flux_wmom,
                                            const double& p,
                                            const double& u,
                                            const double& v,
                                            const double& w,
                                            const double f_mass[nSpace],
                                            const double f_umom[nSpace],
                                            const double f_vmom[nSpace],
                                            const double f_wmom[nSpace],
                                            const double df_mass_du[nSpace],
                                            const double df_mass_dv[nSpace],
                                            const double df_mass_dw[nSpace],
                                            const double df_umom_dp[nSpace],
                                            const double df_umom_du[nSpace],
                                            const double df_umom_dv[nSpace],
                                            const double df_umom_dw[nSpace],
                                            const double df_vmom_dp[nSpace],
                                            const double df_vmom_du[nSpace],
                                            const double df_vmom_dv[nSpace],
                                            const double df_vmom_dw[nSpace],
                                            const double df_wmom_dp[nSpace],
                                            const double df_wmom_du[nSpace],
                                            const double df_wmom_dv[nSpace],
                                            const double df_wmom_dw[nSpace],
                                            double& flux_mass,
                                            double& flux_umom,
                                            double& flux_vmom,
                                            double& flux_wmom,
                                            double* velocity_star,
                                            double* velocity)
      {
        double flowSpeedNormal;
        flux_mass = 0.0;
        flux_umom = 0.0;
        flux_vmom = 0.0;
        /* flux_wmom = 0.0; */
        flowSpeedNormal=(n[0]*velocity_star[0] +
                         n[1]*velocity_star[1]);
        velocity[0] = u;
        velocity[1] = v;
        /* velocity[2] = w; */
        //cek todo, switch back to conservative form weak BC's
        if (isDOFBoundary_u != 1)
          {
            flux_mass += n[0]*f_mass[0];
          }
        else
          {
            flux_mass += n[0]*f_mass[0];
            if (flowSpeedNormal < 0.0)
              {
                flux_umom+=flowSpeedNormal*(bc_u - u);
                velocity[0] = bc_u;
              }
          }
        if (isDOFBoundary_v != 1)
          {
            flux_mass+=n[1]*f_mass[1];
          }
        else
          {
            flux_mass+=n[1]*f_mass[1];
            if (flowSpeedNormal < 0.0)
              {
                flux_vmom+=flowSpeedNormal*(bc_v - v);
                velocity[1] = bc_v;
              }
          }
        /* if (isDOFBoundary_w != 1) */
        /*   { */
        /*     flux_mass+=n[2]*f_mass[2]; */
        /*   } */
        /* else */
        /*   { */
        /*     flux_mass +=n[2]*f_mass[2]; */
        /*     if (flowSpeedNormal < 0.0) */
        /*       { */
        /*         flux_wmom+=flowSpeedNormal*(bc_w - w); */
        /*       } */
        /*   } */
        /* if (isDOFBoundary_w != 1) */
        /*   { */
        /*     flux_mass+=n[2]*f_mass[2]; */
        /*   } */
        /* else */
        /*   { */
        /*     flux_mass +=n[2]*f_mass[2]; */
        /*     if (flowSpeedNormal < 0.0) */
        /*       flux_wmom+=bc_speed*(bc_w - w); */
        /*   } */
	
        if (isFluxBoundary_u == 1)
          {
            flux_umom = bc_flux_umom;
            velocity[0] = bc_flux_umom;
          }
        if (isFluxBoundary_v == 1)
          {
            flux_vmom = bc_flux_vmom;
            velocity[1] = bc_flux_umom;
          }
        /* if (isFluxBoundary_w == 1) */
        /*   { */
        /*     flux_wmom = bc_flux_wmom; */
        /*   } */
      }

      inline
        void exteriorNumericalAdvectiveFluxDerivatives(const int& isDOFBoundary_p,
                                                       const int& isDOFBoundary_u,
                                                       const int& isDOFBoundary_v,
                                                       const int& isDOFBoundary_w,
                                                       const int& isFluxBoundary_p,
                                                       const int& isFluxBoundary_u,
                                                       const int& isFluxBoundary_v,
                                                       const int& isFluxBoundary_w,
                                                       const double& oneByRho,
                                                       const double n[nSpace],
                                                       const double& vos,
                                                       const double& bc_p,
                                                       const double& bc_u,
                                                       const double& bc_v,
                                                       const double& bc_w,
                                                       const double bc_f_mass[nSpace],
                                                       const double bc_f_umom[nSpace],
                                                       const double bc_f_vmom[nSpace],
                                                       const double bc_f_wmom[nSpace],
                                                       const double& bc_flux_mass,
                                                       const double& bc_flux_umom,
                                                       const double& bc_flux_vmom,
                                                       const double& bc_flux_wmom,
                                                       const double& p,
                                                       const double& u,
                                                       const double& v,
                                                       const double& w,
                                                       const double f_mass[nSpace],
                                                       const double f_umom[nSpace],
                                                       const double f_vmom[nSpace],
                                                       const double f_wmom[nSpace],
                                                       const double df_mass_du[nSpace],
                                                       const double df_mass_dv[nSpace],
                                                       const double df_mass_dw[nSpace],
                                                       const double df_umom_dp[nSpace],
                                                       const double df_umom_du[nSpace],
                                                       const double df_umom_dv[nSpace],
                                                       const double df_umom_dw[nSpace],
                                                       const double df_vmom_dp[nSpace],
                                                       const double df_vmom_du[nSpace],
                                                       const double df_vmom_dv[nSpace],
                                                       const double df_vmom_dw[nSpace],
                                                       const double df_wmom_dp[nSpace],
                                                       const double df_wmom_du[nSpace],
                                                       const double df_wmom_dv[nSpace],
                                                       const double df_wmom_dw[nSpace],
                                                       double& dflux_mass_du,
                                                       double& dflux_mass_dv,
                                                       double& dflux_mass_dw,
                                                       double& dflux_umom_dp,
                                                       double& dflux_umom_du,
                                                       double& dflux_umom_dv,
                                                       double& dflux_umom_dw,
                                                       double& dflux_vmom_dp,
                                                       double& dflux_vmom_du,
                                                       double& dflux_vmom_dv,
                                                       double& dflux_vmom_dw,
                                                       double& dflux_wmom_dp,
                                                       double& dflux_wmom_du,
                                                       double& dflux_wmom_dv,
                                                       double& dflux_wmom_dw,
                                                       double* velocity_star)
      {
        double flowSpeedNormal;
        dflux_mass_du = 0.0;
        dflux_mass_dv = 0.0;
        /* dflux_mass_dw = 0.0; */

        dflux_umom_dp = 0.0;
        dflux_umom_du = 0.0;
        dflux_umom_dv = 0.0;
        /* dflux_umom_dw = 0.0; */
        dflux_vmom_dp = 0.0;
        dflux_vmom_du = 0.0;
        dflux_vmom_dv = 0.0;
        /* dflux_vmom_dw = 0.0; */


        dflux_wmom_dp = 0.0;
        dflux_wmom_du = 0.0;
        dflux_wmom_dv = 0.0;
        /* dflux_wmom_dw = 0.0; */
        flowSpeedNormal=(n[0]*velocity_star[0] +
                         n[1]*velocity_star[1]);
        if (isDOFBoundary_u != 1)
          {
            dflux_mass_du += n[0]*df_mass_du[0];
          }
        else
          {
            dflux_mass_du += n[0]*df_mass_du[0];
            if (flowSpeedNormal < 0.0)
              dflux_umom_du -= flowSpeedNormal;
          }
        if (isDOFBoundary_v != 1)
          {
            dflux_mass_dv += n[1]*df_mass_dv[1];
          }
        else
          {
            dflux_mass_dv += n[1]*df_mass_dv[1];
            if (flowSpeedNormal < 0.0)
              dflux_vmom_dv -= flowSpeedNormal;
          }
        /* if (isDOFBoundary_w != 1) */
        /*   { */
        /*     dflux_mass_dw+=n[2]*df_mass_dw[2]; */
        /*   } */
        /* else */
        /*   { */
        /*     dflux_mass_dw += n[2]*df_mass_dw[2]; */
        /*     if (flowSpeedNormal < 0.0) */
        /*       dflux_wmom_dw -= flowSpeedNormal; */
        /*   } */
        /* if (isDOFBoundary_w != 1) */
        /*   { */
        /*     dflux_mass_dw+=n[2]*df_mass_dw[2]; */
        /*   } */
        /* else */
        /*   { */
        /*     dflux_mass_dw += n[2]*df_mass_dw[2]; */
        /*     if (flowSpeedNormal < 0.0) */
        /*       dflux_wmom_dw += bc_speed; */
        /*   } */
        /* if (isDOFBoundary_p == 1) */
        /*   { */
        /*     dflux_umom_dp= -n[0]*oneByRho; */
        /*     dflux_vmom_dp= -n[1]*oneByRho; */
        /*     /\* dflux_wmom_dp= -n[2]*oneByRho; *\/ */
        /*   } */
        /* if (isFluxBoundary_p == 1) */
        /*   { */
        /*     dflux_mass_du = 0.0; */
        /*     dflux_mass_dv = 0.0; */
        /*     /\* dflux_mass_dw = 0.0; *\/ */
        /*   } */
        if (isFluxBoundary_u == 1)
          {
            dflux_umom_dp = 0.0;
            dflux_umom_du = 0.0;
            dflux_umom_dv = 0.0;
            /* dflux_umom_dw = 0.0; */
          }
        if (isFluxBoundary_v == 1)
          {
            dflux_vmom_dp = 0.0;
            dflux_vmom_du = 0.0;
            dflux_vmom_dv = 0.0;
            /* dflux_vmom_dw = 0.0; */
          }
        /* if (isFluxBoundary_w == 1) */
        /*      { */
        /*        dflux_wmom_dp = 0.0; */
        /*        dflux_wmom_du = 0.0; */
        /*        dflux_wmom_dv = 0.0; */
        /*        dflux_wmom_dw = 0.0; */
        /*      } */
      }

      inline
        void exteriorNumericalDiffusiveFlux(const double& eps,
                                            const double& phi,
                                            proteus_csr_index_t* rowptr,
                                            proteus_csr_index_t* colind,
                                            const int& isDOFBoundary,
                                            const int& isFluxBoundary,
                                            const double n[nSpace],
                                            double* bc_a,
                                            const double& bc_u,
                                            const double& bc_flux,
                                            double* a,
                                            const double grad_potential[nSpace],
                                            const double& u,
                                            const double& penalty,
                                            double& flux)
      {
        double diffusiveVelocityComponent_I,penaltyFlux,max_a;
        if(isFluxBoundary == 1)
          {
            flux = bc_flux;
          }
        else if(isDOFBoundary == 1)
          {
            flux = 0.0;
            max_a=0.0;
            for(int I=0;I<nSpace;I++)
              {
                diffusiveVelocityComponent_I=0.0;
                for(proteus_csr_index_t m=rowptr[I];m<rowptr[I+1];m++)
                  {
                    diffusiveVelocityComponent_I -= a[m]*grad_potential[colind[m]];
                    max_a = fmax(max_a,a[m]);
                  }
                flux+= diffusiveVelocityComponent_I*n[I];
              }
            penaltyFlux = max_a*penalty*(u-bc_u);
            flux += penaltyFlux;
            //contact line slip
            //flux*=(smoothedDirac(eps,0) - smoothedDirac(eps,phi))/smoothedDirac(eps,0);
          }
        else
          {
            //std::cerr<<"warning, diffusion term with no boundary condition set, setting diffusive flux to 0.0"<<std::endl;
            flux = 0.0;
          }
      }

      inline
        double ExteriorNumericalDiffusiveFluxJacobian(const double& eps,
                                                      const double& phi,
                                                      proteus_csr_index_t* rowptr,
                                                      proteus_csr_index_t* colind,
                                                      const int& isDOFBoundary,
                                                      const int& isFluxBoundary,
                                                      const double n[nSpace],
                                                      double* a,
                                                      const double& v,
                                                      const double grad_v[nSpace],
                                                      const double& penalty)
      {
        double dvel_I,tmp=0.0,max_a=0.0;
        if(isFluxBoundary==0 && isDOFBoundary==1)
          {
            for(int I=0;I<nSpace;I++)
              {
                dvel_I=0.0;
                for(proteus_csr_index_t m=rowptr[I];m<rowptr[I+1];m++)
                  {
                    dvel_I -= a[m]*grad_v[colind[m]];
                    max_a = fmax(max_a,a[m]);
                  }
                tmp += dvel_I*n[I];
              }
            tmp +=max_a*penalty*v;
            //contact line slip
            //tmp*=(smoothedDirac(eps,0) - smoothedDirac(eps,phi))/smoothedDirac(eps,0);
          }
        return tmp;
      }

      void calculateResidual(arguments_dict& args)
      {
        proteus::pyarray<double>& mesh_trial_ref = args.array<double>("mesh_trial_ref");
        proteus::pyarray<double>& mesh_grad_trial_ref = args.array<double>("mesh_grad_trial_ref");
        proteus::pyarray<double>& mesh_dof = args.array<double>("mesh_dof");
        proteus::pyarray<double>& mesh_velocity_dof = args.array<double>("mesh_velocity_dof");
        double MOVING_DOMAIN = args.scalar<double>("MOVING_DOMAIN");
        double PSTAB = args.scalar<double>("PSTAB");
        proteus::pyarray<proteus_index_t>& mesh_l2g = args.array<proteus_index_t>("mesh_l2g");
        proteus::pyarray<double>& dV_ref = args.array<double>("dV_ref");
        proteus::pyarray<double>& p_trial_ref = args.array<double>("p_trial_ref");
        proteus::pyarray<double>& p_grad_trial_ref = args.array<double>("p_grad_trial_ref");
        proteus::pyarray<double>& p_test_ref = args.array<double>("p_test_ref");
        proteus::pyarray<double>& p_grad_test_ref = args.array<double>("p_grad_test_ref");
        proteus::pyarray<double>& q_p = args.array<double>("q_p");
        proteus::pyarray<double>& q_grad_p = args.array<double>("q_grad_p");
        proteus::pyarray<double>& ebqe_p = args.array<double>("ebqe_p");
        proteus::pyarray<double>& ebqe_grad_p = args.array<double>("ebqe_grad_p");
        proteus::pyarray<double>& vel_trial_ref = args.array<double>("vel_trial_ref");
        proteus::pyarray<double>& vel_grad_trial_ref = args.array<double>("vel_grad_trial_ref");
        proteus::pyarray<double>& vel_test_ref = args.array<double>("vel_test_ref");
        proteus::pyarray<double>& vel_grad_test_ref = args.array<double>("vel_grad_test_ref");
        proteus::pyarray<double>& mesh_trial_trace_ref = args.array<double>("mesh_trial_trace_ref");
        proteus::pyarray<double>& mesh_grad_trial_trace_ref = args.array<double>("mesh_grad_trial_trace_ref");
        proteus::pyarray<double>& dS_ref = args.array<double>("dS_ref");
        proteus::pyarray<double>& p_trial_trace_ref = args.array<double>("p_trial_trace_ref");
        proteus::pyarray<double>& p_grad_trial_trace_ref = args.array<double>("p_grad_trial_trace_ref");
        proteus::pyarray<double>& p_test_trace_ref = args.array<double>("p_test_trace_ref");
        proteus::pyarray<double>& p_grad_test_trace_ref = args.array<double>("p_grad_test_trace_ref");
        proteus::pyarray<double>& vel_trial_trace_ref = args.array<double>("vel_trial_trace_ref");
        proteus::pyarray<double>& vel_grad_trial_trace_ref = args.array<double>("vel_grad_trial_trace_ref");
        proteus::pyarray<double>& vel_test_trace_ref = args.array<double>("vel_test_trace_ref");
        proteus::pyarray<double>& vel_grad_test_trace_ref = args.array<double>("vel_grad_test_trace_ref");
        proteus::pyarray<double>& normal_ref = args.array<double>("normal_ref");
        proteus::pyarray<double>& boundaryJac_ref = args.array<double>("boundaryJac_ref");
        double eb_adjoint_sigma = args.scalar<double>("eb_adjoint_sigma");
        proteus::pyarray<double>& elementDiameter = args.array<double>("elementDiameter");
        proteus::pyarray<double>& nodeDiametersArray = args.array<double>("nodeDiametersArray");
        double hFactor = args.scalar<double>("hFactor");
        proteus_index_t nElements_global = args.scalar<proteus_index_t>("nElements_global");
        proteus_index_t nElementBoundaries_owned = args.scalar<proteus_index_t>("nElementBoundaries_owned");
        double useRBLES = args.scalar<double>("useRBLES");
        double useMetrics = args.scalar<double>("useMetrics");
        double alphaBDF = args.scalar<double>("alphaBDF");
        double epsFact_rho = args.scalar<double>("epsFact_rho");
        double epsFact_mu = args.scalar<double>("epsFact_mu");
        double sigma = args.scalar<double>("sigma");
        double rho_0 = args.scalar<double>("rho_0");
        double nu_0 = args.scalar<double>("nu_0");
        double rho_1 = args.scalar<double>("rho_1");
        double nu_1 = args.scalar<double>("nu_1");
        double rho_s = args.scalar<double>("rho_s");
        double smagorinskyConstant = args.scalar<double>("smagorinskyConstant");
        int turbulenceClosureModel = args.scalar<int>("turbulenceClosureModel");
        double Ct_sge = args.scalar<double>("Ct_sge");
        double Cd_sge = args.scalar<double>("Cd_sge");
        double C_dc = args.scalar<double>("C_dc");
        double C_b = args.scalar<double>("C_b");
        const proteus::pyarray<double>& eps_solid = args.array<double>("eps_solid");
        const proteus::pyarray<double>& q_velocity_fluid = args.array<double>("q_velocity_fluid");
        const proteus::pyarray<double>& q_velocityStar_fluid = args.array<double>("q_velocityStar_fluid");
        const proteus::pyarray<double>& q_vos = args.array<double>("q_vos");
        const proteus::pyarray<double>& q_dvos_dt = args.array<double>("q_dvos_dt");
        const proteus::pyarray<double>& q_grad_vos = args.array<double>("q_grad_vos");
        const proteus::pyarray<double>& q_dragAlpha = args.array<double>("q_dragAlpha");
        const proteus::pyarray<double>& q_dragBeta = args.array<double>("q_dragBeta");
        const proteus::pyarray<double>& q_mass_source = args.array<double>("q_mass_source");
        const proteus::pyarray<double>& q_turb_var_0 = args.array<double>("q_turb_var_0");
        const proteus::pyarray<double>& q_turb_var_1 = args.array<double>("q_turb_var_1");
        const proteus::pyarray<double>& q_turb_var_grad_0 = args.array<double>("q_turb_var_grad_0");
        proteus::pyarray<double>&  q_eddy_viscosity = args.array<double>("q_eddy_viscosity");
        proteus::pyarray<proteus_index_t>& p_l2g = args.array<proteus_index_t>("p_l2g");
        proteus::pyarray<proteus_index_t>& vel_l2g = args.array<proteus_index_t>("vel_l2g");
        proteus::pyarray<double>& p_dof = args.array<double>("p_dof");
        proteus::pyarray<double>& u_dof = args.array<double>("u_dof");
        proteus::pyarray<double>& v_dof = args.array<double>("v_dof");
        proteus::pyarray<double>& w_dof = args.array<double>("w_dof");
        proteus::pyarray<double>& g = args.array<double>("g");
        const double useVF = args.scalar<double>("useVF");
        proteus::pyarray<double>& vf = args.array<double>("vf");
        proteus::pyarray<double>& phi = args.array<double>("phi");
        proteus::pyarray<double>& normal_phi = args.array<double>("normal_phi");
        proteus::pyarray<double>& kappa_phi = args.array<double>("kappa_phi");
        proteus::pyarray<double>& q_mom_u_acc = args.array<double>("q_mom_u_acc");
        proteus::pyarray<double>& q_mom_v_acc = args.array<double>("q_mom_v_acc");
        proteus::pyarray<double>& q_mom_w_acc = args.array<double>("q_mom_w_acc");
        proteus::pyarray<double>& q_mass_adv = args.array<double>("q_mass_adv");
        proteus::pyarray<double>& q_mom_u_acc_beta_bdf = args.array<double>("q_mom_u_acc_beta_bdf");
        proteus::pyarray<double>& q_mom_v_acc_beta_bdf = args.array<double>("q_mom_v_acc_beta_bdf");
        proteus::pyarray<double>& q_mom_w_acc_beta_bdf = args.array<double>("q_mom_w_acc_beta_bdf");
        proteus::pyarray<double>& q_dV = args.array<double>("q_dV");
        proteus::pyarray<double>& q_dV_last = args.array<double>("q_dV_last");
        proteus::pyarray<double>& q_velocity_sge = args.array<double>("q_velocity_sge");
        proteus::pyarray<double>& ebqe_velocity_star = args.array<double>("ebqe_velocity_star");
        proteus::pyarray<double>& q_cfl = args.array<double>("q_cfl");
        proteus::pyarray<double>& q_numDiff_u = args.array<double>("q_numDiff_u");
        proteus::pyarray<double>& q_numDiff_v = args.array<double>("q_numDiff_v");
        proteus::pyarray<double>& q_numDiff_w = args.array<double>("q_numDiff_w");
        proteus::pyarray<double>& q_numDiff_u_last = args.array<double>("q_numDiff_u_last");
        proteus::pyarray<double>& q_numDiff_v_last = args.array<double>("q_numDiff_v_last");
        proteus::pyarray<double>& q_numDiff_w_last = args.array<double>("q_numDiff_w_last");
        proteus::pyarray<proteus_csr_index_t>& sdInfo_u_u_rowptr = args.array<proteus_csr_index_t>("sdInfo_u_u_rowptr");
        proteus::pyarray<proteus_csr_index_t>& sdInfo_u_u_colind = args.array<proteus_csr_index_t>("sdInfo_u_u_colind");
        proteus::pyarray<proteus_csr_index_t>& sdInfo_u_v_rowptr = args.array<proteus_csr_index_t>("sdInfo_u_v_rowptr");
        proteus::pyarray<proteus_csr_index_t>& sdInfo_u_v_colind = args.array<proteus_csr_index_t>("sdInfo_u_v_colind");
        proteus::pyarray<proteus_csr_index_t>& sdInfo_u_w_rowptr = args.array<proteus_csr_index_t>("sdInfo_u_w_rowptr");
        proteus::pyarray<proteus_csr_index_t>& sdInfo_u_w_colind = args.array<proteus_csr_index_t>("sdInfo_u_w_colind");
        proteus::pyarray<proteus_csr_index_t>& sdInfo_v_v_rowptr = args.array<proteus_csr_index_t>("sdInfo_v_v_rowptr");
        proteus::pyarray<proteus_csr_index_t>& sdInfo_v_v_colind = args.array<proteus_csr_index_t>("sdInfo_v_v_colind");
        proteus::pyarray<proteus_csr_index_t>& sdInfo_v_u_rowptr = args.array<proteus_csr_index_t>("sdInfo_v_u_rowptr");
        proteus::pyarray<proteus_csr_index_t>& sdInfo_v_u_colind = args.array<proteus_csr_index_t>("sdInfo_v_u_colind");
        proteus::pyarray<proteus_csr_index_t>& sdInfo_v_w_rowptr = args.array<proteus_csr_index_t>("sdInfo_v_w_rowptr");
        proteus::pyarray<proteus_csr_index_t>& sdInfo_v_w_colind = args.array<proteus_csr_index_t>("sdInfo_v_w_colind");
        proteus::pyarray<proteus_csr_index_t>& sdInfo_w_w_rowptr = args.array<proteus_csr_index_t>("sdInfo_w_w_rowptr");
        proteus::pyarray<proteus_csr_index_t>& sdInfo_w_w_colind = args.array<proteus_csr_index_t>("sdInfo_w_w_colind");
        proteus::pyarray<proteus_csr_index_t>& sdInfo_w_u_rowptr = args.array<proteus_csr_index_t>("sdInfo_w_u_rowptr");
        proteus::pyarray<proteus_csr_index_t>& sdInfo_w_u_colind = args.array<proteus_csr_index_t>("sdInfo_w_u_colind");
        proteus::pyarray<proteus_csr_index_t>& sdInfo_w_v_rowptr = args.array<proteus_csr_index_t>("sdInfo_w_v_rowptr");
        proteus::pyarray<proteus_csr_index_t>& sdInfo_w_v_colind = args.array<proteus_csr_index_t>("sdInfo_w_v_colind");
        proteus_index_t offset_p = args.scalar<proteus_index_t>("offset_p");
        proteus_index_t offset_u = args.scalar<proteus_index_t>("offset_u");
        proteus_index_t offset_v = args.scalar<proteus_index_t>("offset_v");
        proteus_index_t offset_w = args.scalar<proteus_index_t>("offset_w");
        proteus_index_t stride_p = args.scalar<proteus_index_t>("stride_p");
        proteus_index_t stride_u = args.scalar<proteus_index_t>("stride_u");
        proteus_index_t stride_v = args.scalar<proteus_index_t>("stride_v");
        proteus_index_t stride_w = args.scalar<proteus_index_t>("stride_w");
        proteus::pyarray<double>& globalResidual = args.array<double>("globalResidual");
        proteus_index_t nExteriorElementBoundaries_global = args.scalar<proteus_index_t>("nExteriorElementBoundaries_global");
        proteus::pyarray<proteus_index_t>& exteriorElementBoundariesArray = args.array<proteus_index_t>("exteriorElementBoundariesArray");
        proteus::pyarray<proteus_index_t>& elementBoundaryElementsArray = args.array<proteus_index_t>("elementBoundaryElementsArray");
        proteus::pyarray<int>& elementBoundaryLocalElementBoundariesArray = args.array<int>("elementBoundaryLocalElementBoundariesArray");
        proteus::pyarray<double>& ebqe_vf_ext = args.array<double>("ebqe_vf_ext");
        proteus::pyarray<double>& bc_ebqe_vf_ext = args.array<double>("bc_ebqe_vf_ext");
        proteus::pyarray<double>& ebqe_phi_ext = args.array<double>("ebqe_phi_ext");
        proteus::pyarray<double>& bc_ebqe_phi_ext = args.array<double>("bc_ebqe_phi_ext");
        proteus::pyarray<double>& ebqe_normal_phi_ext = args.array<double>("ebqe_normal_phi_ext");
        proteus::pyarray<double>& ebqe_kappa_phi_ext = args.array<double>("ebqe_kappa_phi_ext");
        const proteus::pyarray<double>& ebqe_vos_ext = args.array<double>("ebqe_vos_ext");
        const proteus::pyarray<double>& ebqe_turb_var_0 = args.array<double>("ebqe_turb_var_0");
        const proteus::pyarray<double>& ebqe_turb_var_1 = args.array<double>("ebqe_turb_var_1");
        proteus::pyarray<int>& isDOFBoundary_p = args.array<int>("isDOFBoundary_p");
        proteus::pyarray<int>& isDOFBoundary_u = args.array<int>("isDOFBoundary_u");
        proteus::pyarray<int>& isDOFBoundary_v = args.array<int>("isDOFBoundary_v");
        proteus::pyarray<int>& isDOFBoundary_w = args.array<int>("isDOFBoundary_w");
        proteus::pyarray<int>& isAdvectiveFluxBoundary_p = args.array<int>("isAdvectiveFluxBoundary_p");
        proteus::pyarray<int>& isAdvectiveFluxBoundary_u = args.array<int>("isAdvectiveFluxBoundary_u");
        proteus::pyarray<int>& isAdvectiveFluxBoundary_v = args.array<int>("isAdvectiveFluxBoundary_v");
        proteus::pyarray<int>& isAdvectiveFluxBoundary_w = args.array<int>("isAdvectiveFluxBoundary_w");
        proteus::pyarray<int>& isDiffusiveFluxBoundary_u = args.array<int>("isDiffusiveFluxBoundary_u");
        proteus::pyarray<int>& isDiffusiveFluxBoundary_v = args.array<int>("isDiffusiveFluxBoundary_v");
        proteus::pyarray<int>& isDiffusiveFluxBoundary_w = args.array<int>("isDiffusiveFluxBoundary_w");
        proteus::pyarray<double>& ebqe_bc_p_ext = args.array<double>("ebqe_bc_p_ext");
        proteus::pyarray<double>& ebqe_bc_flux_mass_ext = args.array<double>("ebqe_bc_flux_mass_ext");
        proteus::pyarray<double>& ebqe_bc_flux_mom_u_adv_ext = args.array<double>("ebqe_bc_flux_mom_u_adv_ext");
        proteus::pyarray<double>& ebqe_bc_flux_mom_v_adv_ext = args.array<double>("ebqe_bc_flux_mom_v_adv_ext");
        proteus::pyarray<double>& ebqe_bc_flux_mom_w_adv_ext = args.array<double>("ebqe_bc_flux_mom_w_adv_ext");
        proteus::pyarray<double>& ebqe_bc_u_ext = args.array<double>("ebqe_bc_u_ext");
        proteus::pyarray<double>& ebqe_bc_flux_u_diff_ext = args.array<double>("ebqe_bc_flux_u_diff_ext");
        proteus::pyarray<double>& ebqe_penalty_ext = args.array<double>("ebqe_penalty_ext");
        proteus::pyarray<double>& ebqe_bc_v_ext = args.array<double>("ebqe_bc_v_ext");
        proteus::pyarray<double>& ebqe_bc_flux_v_diff_ext = args.array<double>("ebqe_bc_flux_v_diff_ext");
        proteus::pyarray<double>& ebqe_bc_w_ext = args.array<double>("ebqe_bc_w_ext");
        proteus::pyarray<double>& ebqe_bc_flux_w_diff_ext = args.array<double>("ebqe_bc_flux_w_diff_ext");
        proteus::pyarray<double>& q_x = args.array<double>("q_x");
        proteus::pyarray<double>& q_velocity = args.array<double>("q_velocity");
        proteus::pyarray<double>& ebqe_velocity = args.array<double>("ebqe_velocity");
        proteus::pyarray<double>& flux = args.array<double>("flux");
        proteus::pyarray<double>& elementResidual_p_save = args.array<double>("elementResidual_p_save");
        proteus::pyarray<int>& elementFlags = args.array<int>("elementFlags");
        proteus::pyarray<int>& boundaryFlags = args.array<int>("boundaryFlags");
        proteus::pyarray<double>& barycenters = args.array<double>("barycenters");
        proteus::pyarray<double>& wettedAreas = args.array<double>("wettedAreas");
        proteus::pyarray<double>& netForces_p = args.array<double>("netForces_p");
        proteus::pyarray<double>& netForces_v = args.array<double>("netForces_v");
        proteus::pyarray<double>& netMoments = args.array<double>("netMoments");
        proteus::pyarray<double>& ncDrag = args.array<double>("ncDrag");
        double LAG_MU_FR = args.scalar<double>("LAG_MU_FR");
        proteus::pyarray<double>& q_mu_fr_last = args.array<double>("q_mu_fr_last");
        proteus::pyarray<double>& q_mu_fr = args.array<double>("q_mu_fr");
        //
        //loop over elements to compute volume integrals and load them into element and global residual
        //
        for(proteus_index_t eN=0;eN<nElements_global;eN++)
          {
            //declare local storage for element residual and initialize
            double elementResidual_u[nDOF_test_element],
              elementResidual_v[nDOF_test_element],
              //elementResidual_w[nDOF_test_element],
              mom_u_source_i[nDOF_test_element],
              mom_v_source_i[nDOF_test_element],
              eps_rho,eps_mu;
            double mesh_volume_conservation_element=0.0;
            for (int i=0;i<nDOF_test_element;i++)
              {
                proteus_index_t eN_i = eN*nDOF_test_element+i;
                elementResidual_p_save[eN_i]=0.0;
                elementResidual_u[i]=0.0;
                elementResidual_v[i]=0.0;
                /* elementResidual_w[i]=0.0; */
                mom_u_source_i[i]=0.0;
                mom_v_source_i[i]=0.0;
              }//i
            //
            //loop over quadrature points and compute integrands
            //
            for(int k=0;k<nQuadraturePoints_element;k++)
              {
                //compute indices and declare local storage
                proteus_index_t eN_k = eN*nQuadraturePoints_element+k,
                  eN_k_nSpace = eN_k*nSpace,
                  eN_k_3d     = eN_k*3,
                  eN_nDOF_trial_element = eN*nDOF_trial_element;
                double p=0.0,u=0.0,v=0.0,w=0.0,
                  grad_p[nSpace],grad_u[nSpace],grad_v[nSpace],grad_w[nSpace],grad_vos[nSpace],
                  mom_u_acc=0.0,
                  dmom_u_acc_u=0.0,
                  mom_v_acc=0.0,
                  dmom_v_acc_v=0.0,
                  mom_w_acc=0.0,
                  dmom_w_acc_w=0.0,
                  mass_adv[nSpace],
                  dmass_adv_u[nSpace],
                  dmass_adv_v[nSpace],
                  dmass_adv_w[nSpace],
                  mom_u_adv[nSpace],
                  dmom_u_adv_u[nSpace],
                  dmom_u_adv_v[nSpace],
                  dmom_u_adv_w[nSpace],
                  mom_v_adv[nSpace],
                  dmom_v_adv_u[nSpace],
                  dmom_v_adv_v[nSpace],
                  dmom_v_adv_w[nSpace],
                  mom_w_adv[nSpace],
                  dmom_w_adv_u[nSpace],
                  dmom_w_adv_v[nSpace],
                  dmom_w_adv_w[nSpace],
                  mom_uu_diff_ten[nSpace],
                  mom_vv_diff_ten[nSpace],
                  mom_ww_diff_ten[nSpace],
                  mom_uv_diff_ten[1],
                  mom_uw_diff_ten[1],
                  mom_vu_diff_ten[1],
                  mom_vw_diff_ten[1],
                  mom_wu_diff_ten[1],
                  mom_wv_diff_ten[1],
                  mom_u_source=0.0,
                  mom_v_source=0.0,
                  mom_w_source=0.0,
                  mom_u_ham=0.0,
                  dmom_u_ham_grad_p[nSpace],
                  dmom_u_ham_grad_u[nSpace],
                  mom_v_ham=0.0,
                  dmom_v_ham_grad_p[nSpace],
                  dmom_v_ham_grad_v[nSpace],
                  mom_w_ham=0.0,
                  dmom_w_ham_grad_p[nSpace],
                  dmom_w_ham_grad_w[nSpace],
                  mom_u_acc_t=0.0,
                  dmom_u_acc_u_t=0.0,
                  mom_v_acc_t=0.0,
                  dmom_v_acc_v_t=0.0,
                  pdeResidual_p=0.0,
                  pdeResidual_u=0.0,
                  pdeResidual_v=0.0,
                  pdeResidual_w=0.0,
                  Lstar_u_u[nDOF_test_element],
                  Lstar_v_v[nDOF_test_element],
                  subgridError_p=0.0,
                  subgridError_u=0.0,
                  subgridError_v=0.0,
                  subgridError_w=0.0,
                  tau_p=0.0,tau_p0=0.0,tau_p1=0.0,
                  tau_v=0.0,tau_v0=0.0,tau_v1=0.0,
                  jac[nSpace*nSpace],
                  jacDet,
                  jacInv[nSpace*nSpace],
                  vel_grad_trial[nDOF_trial_element*nSpace],
                  vel_test_dV[nDOF_trial_element],
                  vel_grad_test_dV[nDOF_test_element*nSpace],
                  dV,x,y,z,xt,yt,zt,
                  //
                  vos,
                  //meanGrainSize,
                  mass_source,
                  dmom_u_source[nSpace],
                  dmom_v_source[nSpace],
                  dmom_w_source[nSpace],
                  //
                  G[nSpace*nSpace],G_dd_G,tr_G,norm_Rv,h_phi, dmom_adv_star[nSpace],dmom_adv_sge[nSpace];
                //get jacobian, etc for mapping reference element
                ck.calculateMapping_element(eN,
                                            k,
                                            mesh_dof.data(),
                                            mesh_l2g.data(),
                                            mesh_trial_ref.data(),
                                            mesh_grad_trial_ref.data(),
                                            jac,
                                            jacDet,
                                            jacInv,
                                            x,y,z);
                ck.calculateH_element(eN,
                                      k,
                                      nodeDiametersArray.data(),
                                      mesh_l2g.data(),
                                      mesh_trial_ref.data(),
                                      h_phi);
                ck.calculateMappingVelocity_element(eN,
                                                    k,
                                                    mesh_velocity_dof.data(),
                                                    mesh_l2g.data(),
                                                    mesh_trial_ref.data(),
                                                    xt,yt,zt);
                //xt=0.0;yt=0.0;zt=0.0;
                //std::cout<<"xt "<<xt<<'\t'<<yt<<'\t'<<zt<<std::endl;
                //get the physical integration weight
                dV = fabs(jacDet)*dV_ref.data()[k];
                ck.calculateG(jacInv,G,G_dd_G,tr_G);
                //ck.calculateGScale(G,&normal_phi.data()[eN_k_nSpace],h_phi);
              
                eps_rho = epsFact_rho*(useMetrics*h_phi+(1.0-useMetrics)*elementDiameter.data()[eN]);
                eps_mu  = epsFact_mu *(useMetrics*h_phi+(1.0-useMetrics)*elementDiameter.data()[eN]);
             
                //get the trial function gradients
                /* ck.gradTrialFromRef(&p_grad_trial_ref.data()[k*nDOF_trial_element*nSpace],jacInv,p_grad_trial); */
                ck.gradTrialFromRef(&vel_grad_trial_ref.data()[k*nDOF_trial_element*nSpace],jacInv,vel_grad_trial);
                //get the solution
                /* ck.valFromDOF(p_dof.data(),&p_l2g.data()[eN_nDOF_trial_element],&p_trial_ref.data()[k*nDOF_trial_element],p); */
                p = q_p.data()[eN_k];
                ck.valFromDOF(u_dof.data(),&vel_l2g.data()[eN_nDOF_trial_element],&vel_trial_ref.data()[k*nDOF_trial_element],u);
                ck.valFromDOF(v_dof.data(),&vel_l2g.data()[eN_nDOF_trial_element],&vel_trial_ref.data()[k*nDOF_trial_element],v);
                /* ck.valFromDOF(w_dof.data(),&vel_l2g.data()[eN_nDOF_trial_element],&vel_trial_ref.data()[k*nDOF_trial_element],w); */
                //get the solution gradients
                /* ck.gradFromDOF(p_dof.data(),&p_l2g.data()[eN_nDOF_trial_element],p_grad_trial,grad_p); */
                for (int I=0;I<nSpace;I++)
                  grad_p[I] = q_grad_p.data()[eN_k_nSpace + I];
                for (int I=0;I<nSpace;I++)
                  grad_vos[I] = q_grad_vos.data()[eN_k_nSpace + I];                         
                ck.gradFromDOF(u_dof.data(),&vel_l2g.data()[eN_nDOF_trial_element],vel_grad_trial,grad_u);
                ck.gradFromDOF(v_dof.data(),&vel_l2g.data()[eN_nDOF_trial_element],vel_grad_trial,grad_v);
                /* ck.gradFromDOF(w_dof.data(),&vel_l2g.data()[eN_nDOF_trial_element],vel_grad_trial,grad_w); */
                //VRANS
                vos      = q_vos.data()[eN_k];//sed fraction - gco check
                //precalculate test function products with integration weights
                for (int j=0;j<nDOF_trial_element;j++)
                  {
                    /* p_test_dV[j] = p_test_ref.data()[k*nDOF_trial_element+j]*dV; */
                    vel_test_dV[j] = vel_test_ref.data()[k*nDOF_trial_element+j]*dV;
                    for (int I=0;I<nSpace;I++)
                      {
                        /* p_grad_test_dV[j*nSpace+I]   = p_grad_trial[j*nSpace+I]*dV;//cek warning won't work for Petrov-Galerkin */
                        vel_grad_test_dV[j*nSpace+I] = vel_grad_trial[j*nSpace+I]*dV;//cek warning won't work for Petrov-Galerkin
                      }
                  }
                //cek hack
                double div_mesh_velocity=0.0;
                int NDOF_MESH_TRIAL_ELEMENT=3;
                for (int j=0;j<NDOF_MESH_TRIAL_ELEMENT;j++)
                  {
                    proteus_index_t eN_j=eN*NDOF_MESH_TRIAL_ELEMENT+j;
                    div_mesh_velocity +=
                      mesh_velocity_dof.data()[mesh_l2g.data()[eN_j]*3+0]*vel_grad_trial[j*nSpace+0] +
                      mesh_velocity_dof.data()[mesh_l2g.data()[eN_j]*3+1]*vel_grad_trial[j*nSpace+1];
                  }
                mesh_volume_conservation_element += (alphaBDF*(dV-q_dV_last.data()[eN_k])/dV - div_mesh_velocity)*dV;
                div_mesh_velocity = DM3*div_mesh_velocity + (1.0-DM3)*alphaBDF*(dV-q_dV_last.data()[eN_k])/dV;

                //meanGrainSize = q_meanGrain[eN_k]; 
                //
                q_x.data()[eN_k_3d+0]=x;
                q_x.data()[eN_k_3d+1]=y;
                /* q_x.data()[eN_k_3d+2]=z; */
                /* // */
                //calculate pde coefficients at quadrature points
                //
                evaluateCoefficients(eps_rho,
                                     eps_mu,
                                     sigma,
                                     rho_0,
                                     nu_0,
                                     rho_1,
                                     nu_1,
                                     rho_s,
                                     elementDiameter.data()[eN],
                                     smagorinskyConstant,
                                     turbulenceClosureModel,
                                     g.data(),
                                     useVF,
                                     vf.data()[eN_k],
                                     phi.data()[eN_k],
                                     &normal_phi.data()[eN_k_nSpace],
                                     kappa_phi.data()[eN_k],
                                     //VRANS
                                     vos,
                                     //
                                     p,
                                     grad_p,
                                     grad_u,
                                     grad_v,
                                     grad_w,
                                     u,
                                     v,
                                     w,
                                     q_velocity_sge.data()[eN_k_nSpace+0],
                                     q_velocity_sge.data()[eN_k_nSpace+1],
                                     q_velocity_sge.data()[eN_k_nSpace+1],//hack, shouldn't  be used
                                     q_eddy_viscosity.data()[eN_k],
                                     mom_u_acc,
                                     dmom_u_acc_u,
                                     mom_v_acc,
                                     dmom_v_acc_v,
                                     mom_w_acc,
                                     dmom_w_acc_w,
                                     mass_adv,
                                     dmass_adv_u,
                                     dmass_adv_v,
                                     dmass_adv_w,
                                     mom_u_adv,
                                     dmom_u_adv_u,
                                     dmom_u_adv_v,
                                     dmom_u_adv_w,
                                     mom_v_adv,
                                     dmom_v_adv_u,
                                     dmom_v_adv_v,
                                     dmom_v_adv_w,
                                     mom_w_adv,
                                     dmom_w_adv_u,
                                     dmom_w_adv_v,
                                     dmom_w_adv_w,
                                     mom_uu_diff_ten,
                                     mom_vv_diff_ten,
                                     mom_ww_diff_ten,
                                     mom_uv_diff_ten,
                                     mom_uw_diff_ten,
                                     mom_vu_diff_ten,
                                     mom_vw_diff_ten,
                                     mom_wu_diff_ten,
                                     mom_wv_diff_ten,
                                     mom_u_source,
                                     mom_v_source,
                                     mom_w_source,
                                     mom_u_ham,
                                     dmom_u_ham_grad_p,
                                     dmom_u_ham_grad_u,
                                     mom_v_ham,
                                     dmom_v_ham_grad_p,
                                     dmom_v_ham_grad_v,
                                     mom_w_ham,
                                     dmom_w_ham_grad_p,          
                                     dmom_w_ham_grad_w);          
                //VRANS
                mass_source = q_mass_source.data()[eN_k];
                for (int I=0;I<nSpace;I++)
                  {
                    dmom_u_source[I] = 0.0;
                    dmom_v_source[I] = 0.0;
                    dmom_w_source[I] = 0.0;
                  }
                updateDarcyForchheimerTerms_Ergun(// linearDragFactor,
                                                  // nonlinearDragFactor,
                                                  // vos,
                                                  // meanGrainSize,
                                                  q_dragAlpha.data()[eN_k],
                                                  q_dragBeta.data()[eN_k],
                                                  eps_rho,
                                                  eps_mu,
                                                  rho_0,
                                                  nu_0,
                                                  rho_1,
                                                  nu_1,
                                                  q_eddy_viscosity.data()[eN_k],
                                                  useVF,
                                                  vf.data()[eN_k],
                                                  phi.data()[eN_k],
                                                  u,
                                                  v,
                                                  w,
                                                  q_velocity_sge.data()[eN_k_nSpace+0],
                                                  q_velocity_sge.data()[eN_k_nSpace+1],
                                                  q_velocity_sge.data()[eN_k_nSpace+1],//hack, shouldn't  be used
                                                  eps_solid.data()[elementFlags.data()[eN]],
                                                  vos,
                                                  q_velocity_fluid.data()[eN_k_nSpace+0],
                                                  q_velocity_fluid.data()[eN_k_nSpace+1],
                                                  q_velocity_fluid.data()[eN_k_nSpace+1],
                                                  q_velocityStar_fluid.data()[eN_k_nSpace+0],
                                                  q_velocityStar_fluid.data()[eN_k_nSpace+1],
                                                  q_velocityStar_fluid.data()[eN_k_nSpace+1],
                                                  mom_u_source,
                                                  mom_v_source,
                                                  mom_w_source,
                                                  dmom_u_source,
                                                  dmom_v_source,
                                                  dmom_w_source,
                                                  q_grad_vos.data()[eN_k_nSpace+0],
                                                  q_grad_vos.data()[eN_k_nSpace+1],
                                                  q_grad_vos.data()[eN_k_nSpace+1]);

		updateFrictionalPressure(vos,
                                         grad_vos,
                                         mom_u_source,
                                         mom_v_source,
                                         mom_w_source);
                updateFrictionalStress(LAG_MU_FR,
                                       q_mu_fr_last.data()[eN_k],
                                       q_mu_fr.data()[eN_k],
                                       vos,
                                       eps_rho,
                                       eps_mu,
                                       rho_0,
                                       nu_0,
                                       rho_1,
                                       nu_1,
                                       rho_s,
                                       useVF,
                                       vf.data()[eN_k],
                                       phi.data()[eN_k],
                                       grad_u,
                                       grad_v,
                                       grad_w,
                                       mom_uu_diff_ten,
                                       mom_uv_diff_ten,
                                       mom_uw_diff_ten,
                                       mom_vv_diff_ten,
                                       mom_vu_diff_ten,
                                       mom_vw_diff_ten,
                                       mom_ww_diff_ten,
                                       mom_wu_diff_ten,
                                       mom_wv_diff_ten);
                //Turbulence closure model
                //
                //save momentum for time history and velocity for subgrid error
                //
                q_mom_u_acc.data()[eN_k] = mom_u_acc;                            
                q_mom_v_acc.data()[eN_k] = mom_v_acc;                            
                /* q_mom_w_acc.data()[eN_k] = mom_w_acc; */
                //subgrid error uses grid scale velocity
                q_mass_adv.data()[eN_k_nSpace+0] = u;
                q_mass_adv.data()[eN_k_nSpace+1] = v;
                /* q_mass_adv.data()[eN_k_nSpace+2] = w; */
                //
                //moving mesh
                //
                mom_u_adv[0] -= MOVING_DOMAIN*dmom_u_acc_u*mom_u_acc*xt;
                mom_u_adv[1] -= MOVING_DOMAIN*dmom_u_acc_u*mom_u_acc*yt;
                /* mom_u_adv[2] -= MOVING_DOMAIN*dmom_u_acc_u*mom_u_acc*zt; */
                dmom_u_adv_u[0] -= MOVING_DOMAIN*dmom_u_acc_u*xt;
                dmom_u_adv_u[1] -= MOVING_DOMAIN*dmom_u_acc_u*yt;
                /* dmom_u_adv_u[2] -= MOVING_DOMAIN*dmom_u_acc_u*zt; */

                mom_v_adv[0] -= MOVING_DOMAIN*dmom_v_acc_v*mom_v_acc*xt;  
                mom_v_adv[1] -= MOVING_DOMAIN*dmom_v_acc_v*mom_v_acc*yt;
                /* mom_v_adv[2] -= MOVING_DOMAIN*dmom_v_acc_v*mom_v_acc*zt; */
                dmom_v_adv_v[0] -= MOVING_DOMAIN*dmom_v_acc_v*xt;
                dmom_v_adv_v[1] -= MOVING_DOMAIN*dmom_v_acc_v*yt;
                /* dmom_v_adv_v[2] -= MOVING_DOMAIN*dmom_v_acc_v*zt; */

                /* mom_w_adv[0] -= MOVING_DOMAIN*dmom_w_acc_w*mom_w_acc*xt; */ 
                /* mom_w_adv[1] -= MOVING_DOMAIN*dmom_w_acc_w*mom_w_acc*yt; */
                /* mom_w_adv[2] -= MOVING_DOMAIN*dmom_w_acc_w*mom_w_acc*zt; */
                /* dmom_w_adv_w[0] -= MOVING_DOMAIN*dmom_w_acc_w*xt; */
                /* dmom_w_adv_w[1] -= MOVING_DOMAIN*dmom_w_acc_w*yt; */
                /* dmom_w_adv_w[2] -= MOVING_DOMAIN*dmom_w_acc_w*zt; */
                //
                //calculate time derivative at quadrature points
                //
                if (q_dV_last.data()[eN_k] <= -100)
                  q_dV_last.data()[eN_k] = dV;
                q_dV.data()[eN_k] = dV;
                ck.bdf(alphaBDF,
                       q_mom_u_acc_beta_bdf.data()[eN_k]*q_dV_last.data()[eN_k]/dV,
                       mom_u_acc,
                       dmom_u_acc_u,
                       mom_u_acc_t,
                       dmom_u_acc_u_t);
                ck.bdf(alphaBDF,
                       q_mom_v_acc_beta_bdf.data()[eN_k]*q_dV_last.data()[eN_k]/dV,
                       mom_v_acc,
                       dmom_v_acc_v,
                       mom_v_acc_t,
                       dmom_v_acc_v_t);
        
                /* ck.bdf(alphaBDF, */
                /*           q_mom_w_acc_beta_bdf.data()[eN_k]*q_dV_last.data()[eN_k]/dV, */
                /*           mom_w_acc, */
                /*           dmom_w_acc_w, */
                /*           mom_w_acc_t, */
                /*           dmom_w_acc_w_t); */
                /* // */

                mom_u_acc_t *= dmom_u_acc_u;
                mom_v_acc_t *= dmom_v_acc_v; 

                //calculate subgrid error (strong residual and adjoint)
                //
                //calculate strong residual
                pdeResidual_p =
		  ck.Mass_strong(q_dvos_dt.data()[eN_k]) +
                  ck.Advection_strong(dmass_adv_u,grad_u) +
                  ck.Advection_strong(dmass_adv_v,grad_v) +
                  /* ck.Advection_strong(dmass_adv_w,grad_w) + */
                  DM2*MOVING_DOMAIN*ck.Reaction_strong(alphaBDF*(dV-q_dV_last.data()[eN_k])/dV - div_mesh_velocity) +
                  //VRANS
                  ck.Reaction_strong(mass_source);
                //
          
                dmom_adv_sge[0] = dmom_u_acc_u*(q_velocity_sge.data()[eN_k_nSpace+0] - MOVING_DOMAIN*xt);
                dmom_adv_sge[1] = dmom_u_acc_u*(q_velocity_sge.data()[eN_k_nSpace+1] - MOVING_DOMAIN*yt);
                /* dmom_adv_sge[2] = dmom_u_acc_u*(q_velocity_sge.data()[eN_k_nSpace+2] - MOVING_DOMAIN*zt); */

                pdeResidual_u = 
                  ck.Mass_strong(mom_u_acc_t) +
                  ck.Advection_strong(dmom_adv_sge,grad_u) + //note here and below: same in cons. and non-cons.
                  ck.Hamiltonian_strong(dmom_u_ham_grad_p,grad_p) +
                  ck.Reaction_strong(mom_u_source) -
                  ck.Reaction_strong(u*div_mesh_velocity);
          
                pdeResidual_v = 
                  ck.Mass_strong(mom_v_acc_t) +
                  ck.Advection_strong(dmom_adv_sge,grad_v) +
                  ck.Hamiltonian_strong(dmom_v_ham_grad_p,grad_p) + 
                  ck.Reaction_strong(mom_v_source) -
                  ck.Reaction_strong(v*div_mesh_velocity);
              
                /* pdeResidual_w = ck.Mass_strong(mom_w_acc_t) + */
                /*        ck.Advection_strong(dmom_adv_sge,grad_w) + */
                /*        ck.Hamiltonian_strong(dmom_w_ham_grad_p,grad_p) + */
                /*        ck.Reaction_strong(mom_w_source) - */
                /*   ck.Reaction_strong(w*div_mesh_velocity); */
        
                //calculate tau and tau*Res
                //cek debug
                double tmpR=dmom_u_acc_u_t + dmom_u_source[0];
                calculateSubgridError_tau(hFactor,
                                          elementDiameter.data()[eN],
                                          tmpR,//dmom_u_acc_u_t,
                                          dmom_u_acc_u,
                                          dmom_adv_sge,
                                          mom_uu_diff_ten[1],
                                          dmom_u_ham_grad_p[0],
                                          tau_v0,
                                          tau_p0,
                                          q_cfl.data()[eN_k]);

                calculateSubgridError_tau(Ct_sge,Cd_sge,
                                          G,G_dd_G,tr_G,
                                          tmpR,//dmom_u_acc_u_t,
                                          dmom_adv_sge,
                                          mom_uu_diff_ten[1],
                                          dmom_u_ham_grad_p[0],
                                          tau_v1,
                                          tau_p1,
                                          q_cfl.data()[eN_k]); 

                tau_v = useMetrics*tau_v1+(1.0-useMetrics)*tau_v0;
                tau_p = PSTAB*(useMetrics*tau_p1+(1.0-useMetrics)*tau_p0);

                calculateSubgridError_tauRes(tau_p,
                                             tau_v,
                                             pdeResidual_p,
                                             pdeResidual_u,
                                             pdeResidual_v,
                                             pdeResidual_w,
                                             subgridError_p,
                                             subgridError_u,
                                             subgridError_v,
                                             subgridError_w);
                // velocity used in adjoint (VMS or RBLES, with or without lagging the grid scale velocity)
                dmom_adv_star[0] = dmom_u_acc_u*(q_velocity_sge.data()[eN_k_nSpace+0] - MOVING_DOMAIN*xt + useRBLES*subgridError_u);
                dmom_adv_star[1] = dmom_u_acc_u*(q_velocity_sge.data()[eN_k_nSpace+1] - MOVING_DOMAIN*yt + useRBLES*subgridError_v);
                /* dmom_adv_star[2] = dmom_u_acc_u*(q_velocity_sge.data()[eN_k_nSpace+2] - MOVING_DOMAIN*zt + useRBLES*subgridError_w); */
         
                mom_u_adv[0] += dmom_u_acc_u*(useRBLES*subgridError_u*q_velocity_sge.data()[eN_k_nSpace+0]);        
                mom_u_adv[1] += dmom_u_acc_u*(useRBLES*subgridError_v*q_velocity_sge.data()[eN_k_nSpace+0]); 
                /* mom_u_adv[2] += dmom_u_acc_u*(useRBLES*subgridError_w*q_velocity_sge.data()[eN_k_nSpace+0]);  */
         
                // adjoint times the test functions 
                for (int i=0;i<nDOF_test_element;i++)
                  {
                    int i_nSpace = i*nSpace;
                    /* Lstar_u_p[i]=ck.Advection_adjoint(dmass_adv_u,&p_grad_test_dV[i_nSpace]); */
                    /* Lstar_v_p[i]=ck.Advection_adjoint(dmass_adv_v,&p_grad_test_dV[i_nSpace]); */
                    /* Lstar_w_p[i]=ck.Advection_adjoint(dmass_adv_w,&p_grad_test_dV[i_nSpace]); */
                    //use the same advection adjoint for all three since we're approximating the linearized adjoint
                    Lstar_u_u[i]=ck.Advection_adjoint(dmom_adv_star,&vel_grad_test_dV[i_nSpace]);
                    Lstar_v_v[i]=ck.Advection_adjoint(dmom_adv_star,&vel_grad_test_dV[i_nSpace]);
                    /* Lstar_w_w[i]=ck.Advection_adjoint(dmom_adv_star,&vel_grad_test_dV[i_nSpace]); */
                    /* Lstar_p_w[i]=ck.Hamiltonian_adjoint(dmom_w_ham_grad_p,&vel_grad_test_dV[i_nSpace]); */
                  
                    //VRANS account for drag terms, diagonal only here ... decide if need off diagonal terms too
                    Lstar_u_u[i]+=ck.Reaction_adjoint(dmom_u_source[0],vel_test_dV[i]);
                    Lstar_v_v[i]+=ck.Reaction_adjoint(dmom_v_source[1],vel_test_dV[i]);
                    /* Lstar_w_w[i]+=ck.Reaction_adjoint(dmom_w_source[2],vel_test_dV[i]); */
                    //
                  }

                norm_Rv = sqrt(pdeResidual_u*pdeResidual_u + pdeResidual_v*pdeResidual_v);// + pdeResidual_w*pdeResidual_w);
                q_numDiff_u.data()[eN_k] = C_dc*norm_Rv*(useMetrics/sqrt(G_dd_G+1.0e-12)  + 
                                                  (1.0-useMetrics)*hFactor*hFactor*elementDiameter.data()[eN]*elementDiameter.data()[eN]);
                q_numDiff_v.data()[eN_k] = q_numDiff_u.data()[eN_k];
                q_numDiff_w.data()[eN_k] = q_numDiff_u.data()[eN_k];
                // Save velocity and its gradient (to be used in other models and to compute errors)
                q_velocity.data()[eN_k_nSpace+0]=u;
                q_velocity.data()[eN_k_nSpace+1]=v;
                /* q_velocity.data()[eN_k_nSpace+2]=w;*/
                for(int i=0;i<nDOF_test_element;i++) 
                  { 
                    int i_nSpace=i*nSpace;
                    /* std::cout<<"elemRes_mesh "<<mesh_vel[0]<<'\t'<<mesh_vel[2]<<'\t'<<p_test_dV[i]<<'\t'<<(q_dV_last.data()[eN_k]/dV)<<'\t'<<dV<<std::endl; */
                    /* elementResidual_mesh[i] += ck.Reaction_weak(1.0,p_test_dV[i]) - */
                    /*   ck.Reaction_weak(1.0,p_test_dV[i]*q_dV_last.data()[eN_k]/dV) - */
                    /*   ck.Advection_weak(mesh_vel,&p_grad_test_dV[i_nSpace]); */
                  
                    /* elementResidual_p.data()[i] += ck.Mass_weak(-q_dvos_dt.data()[eN_k],p_test_dV[i]) + */
                    /*   ck.Advection_weak(mass_adv,&p_grad_test_dV[i_nSpace]) + */
                    /*   DM*MOVING_DOMAIN*(ck.Reaction_weak(alphaBDF*1.0,p_test_dV[i]) - */
                    /*                     ck.Reaction_weak(alphaBDF*1.0,p_test_dV[i]*q_dV_last.data()[eN_k]/dV) - */
                    /*                     ck.Advection_weak(mesh_vel,&p_grad_test_dV[i_nSpace])) + */
                    /*   //VRANS */
                    /*   ck.Reaction_weak(mass_source,p_test_dV[i])   + //VRANS source term for wave maker */
                    /*   // */
                    /*   ck.SubgridError(subgridError_u,Lstar_u_p[i]) +  */
                    /*   ck.SubgridError(subgridError_v,Lstar_v_p[i]);// +  */
                    /*   /\* ck.SubgridError(subgridError_w,Lstar_w_p[i]); *\/ */

                    elementResidual_u[i] += 
                      ck.Mass_weak(mom_u_acc_t,vel_test_dV[i]) + 
                      ck.Advection_weak(mom_u_adv,&vel_grad_test_dV[i_nSpace]) +
                      ck.Diffusion_weak(sdInfo_u_u_rowptr.data(),sdInfo_u_u_colind.data(),mom_uu_diff_ten,grad_u,&vel_grad_test_dV[i_nSpace]) + 
                      ck.Diffusion_weak(sdInfo_u_v_rowptr.data(),sdInfo_u_v_colind.data(),mom_uv_diff_ten,grad_v,&vel_grad_test_dV[i_nSpace]) + 
                      /* ck.Diffusion_weak(sdInfo_u_w_rowptr.data(),sdInfo_u_w_colind.data(),mom_uw_diff_ten,grad_w,&vel_grad_test_dV[i_nSpace]) +  */
                      ck.Reaction_weak(mom_u_source,vel_test_dV[i]) + 
                      ck.Hamiltonian_weak(mom_u_ham,vel_test_dV[i]) + 
                      /* ck.SubgridError(subgridError_p,Lstar_p_u[i]) + */
                      ck.SubgridError(subgridError_u,Lstar_u_u[i]) + 
                      ck.NumericalDiffusion(q_numDiff_u_last.data()[eN_k],grad_u,&vel_grad_test_dV[i_nSpace]); 
                    mom_u_source_i[i] += ck.Reaction_weak(mom_u_source,vel_test_dV[i]);

                    elementResidual_v[i] += 
                      ck.Mass_weak(mom_v_acc_t,vel_test_dV[i]) + 
                      ck.Advection_weak(mom_v_adv,&vel_grad_test_dV[i_nSpace]) +
                      ck.Diffusion_weak(sdInfo_v_u_rowptr.data(),sdInfo_v_u_colind.data(),mom_vu_diff_ten,grad_u,&vel_grad_test_dV[i_nSpace]) + 
                      ck.Diffusion_weak(sdInfo_v_v_rowptr.data(),sdInfo_v_v_colind.data(),mom_vv_diff_ten,grad_v,&vel_grad_test_dV[i_nSpace]) + 
                      /* ck.Diffusion_weak(sdInfo_v_w_rowptr.data(),sdInfo_v_w_colind.data(),mom_vw_diff_ten,grad_w,&vel_grad_test_dV[i_nSpace]) +  */
                      ck.Reaction_weak(mom_v_source,vel_test_dV[i]) + 
                      ck.Hamiltonian_weak(mom_v_ham,vel_test_dV[i]) + 
                      /* ck.SubgridError(subgridError_p,Lstar_p_v[i]) + */
                      ck.SubgridError(subgridError_v,Lstar_v_v[i]) + 
                      ck.NumericalDiffusion(q_numDiff_v_last.data()[eN_k],grad_v,&vel_grad_test_dV[i_nSpace]); 
                    mom_v_source_i[i] += ck.Reaction_weak(mom_v_source,vel_test_dV[i]);

                    /* elementResidual_w[i] += 
                       ck.Mass_weak(mom_w_acc_t,vel_test_dV[i]) + */
                    /*   ck.Advection_weak(mom_w_adv,&vel_grad_test_dV[i_nSpace]) +  */
                    /*   ck.Diffusion_weak(sdInfo_w_u_rowptr.data(),sdInfo_w_u_colind.data(),mom_wu_diff_ten,grad_u,&vel_grad_test_dV[i_nSpace]) +  */
                    /*   ck.Diffusion_weak(sdInfo_w_v_rowptr.data(),sdInfo_w_v_colind.data(),mom_wv_diff_ten,grad_v,&vel_grad_test_dV[i_nSpace]) +  */
                    /*   ck.Diffusion_weak(sdInfo_w_w_rowptr.data(),sdInfo_w_w_colind.data(),mom_ww_diff_ten,grad_w,&vel_grad_test_dV[i_nSpace]) +  */
                    /*   ck.Reaction_weak(mom_w_source,vel_test_dV[i]) +  */
                    /*   ck.Hamiltonian_weak(mom_w_ham,vel_test_dV[i]) +  */
                    /*   ck.SubgridError(subgridError_p,Lstar_p_w[i]) +  */
                    /*   ck.SubgridError(subgridError_w,Lstar_w_w[i]) +  */
                    /*   ck.NumericalDiffusion(q_numDiff_w_last.data()[eN_k],grad_w,&vel_grad_test_dV[i_nSpace]);  */
                  }//i
              }
            //
            //load element into global residual and save element residual
            //
            for(int i=0;i<nDOF_test_element;i++) 
              { 
                proteus_index_t eN_i=eN*nDOF_test_element+i;

                /* elementResidual_p_save[eN_i] +=  elementResidual_p.data()[i]; */
                /* mesh_volume_conservation_element_weak += elementResidual_mesh[i]; */
                /* globalResidual.data()[offset_p+stride_p*p_l2g.data()[eN_i]]+=elementResidual_p.data()[i]; */
                globalResidual.data()[offset_u+stride_u*vel_l2g.data()[eN_i]]+=elementResidual_u[i];
                globalResidual.data()[offset_v+stride_v*vel_l2g.data()[eN_i]]+=elementResidual_v[i];
                ncDrag.data()[offset_u+stride_u*vel_l2g.data()[eN_i]]+=mom_u_source_i[i]; 
                ncDrag.data()[offset_v+stride_v*vel_l2g.data()[eN_i]]+=mom_v_source_i[i]; 
                /* globalResidual.data()[offset_w+stride_w*vel_l2g.data()[eN_i]]+=elementResidual_w[i]; */
              }//i
            /* mesh_volume_conservation += mesh_volume_conservation_element; */
            /* mesh_volume_conservation_weak += mesh_volume_conservation_element_weak; */
            /* mesh_volume_conservation_err_max=fmax(mesh_volume_conservation_err_max,fabs(mesh_volume_conservation_element)); */
            /* mesh_volume_conservation_err_max_weak=fmax(mesh_volume_conservation_err_max_weak,fabs(mesh_volume_conservation_element_weak)); */
          }//elements
        //
        //loop over exterior element boundaries to calculate surface integrals and load into element and global residuals
        //
        //ebNE is the Exterior element boundary INdex
        //ebN is the element boundary INdex
        //eN is the element index
        for (proteus_index_t ebNE = 0; ebNE < nExteriorElementBoundaries_global; ebNE++)
          {
            proteus_index_t ebN = exteriorElementBoundariesArray.data()[ebNE],
              eN  = elementBoundaryElementsArray.data()[ebN*2+0];
            int ebN_local = elementBoundaryLocalElementBoundariesArray.data()[ebN*2+0];
            proteus_index_t eN_nDOF_trial_element = eN*nDOF_trial_element;
            double elementResidual_u[nDOF_test_element],
              elementResidual_v[nDOF_test_element],
              //elementResidual_w[nDOF_test_element],
              eps_rho,eps_mu;
            for (int i=0;i<nDOF_test_element;i++)
              {
                elementResidual_u[i]=0.0;
                elementResidual_v[i]=0.0;
                /* elementResidual_w[i]=0.0; */
              }
            for  (int kb=0;kb<nQuadraturePoints_elementBoundary;kb++)
              {
                proteus_index_t ebNE_kb = ebNE*nQuadraturePoints_elementBoundary+kb,
                  ebNE_kb_nSpace = ebNE_kb*nSpace;
                int ebN_local_kb = ebN_local*nQuadraturePoints_elementBoundary+kb,
                  ebN_local_kb_nSpace = ebN_local_kb*nSpace;
                double p_ext=0.0,
                  u_ext=0.0,
                  v_ext=0.0,
                  w_ext=0.0,
                  grad_p_ext[nSpace],
                  grad_u_ext[nSpace],
                  grad_v_ext[nSpace],
                  grad_w_ext[nSpace],
                  mom_u_acc_ext=0.0,
                  dmom_u_acc_u_ext=0.0,
                  mom_v_acc_ext=0.0,
                  dmom_v_acc_v_ext=0.0,
                  mom_w_acc_ext=0.0,
                  dmom_w_acc_w_ext=0.0,
                  mass_adv_ext[nSpace],
                  dmass_adv_u_ext[nSpace],
                  dmass_adv_v_ext[nSpace],
                  dmass_adv_w_ext[nSpace],
                  mom_u_adv_ext[nSpace],
                  dmom_u_adv_u_ext[nSpace],
                  dmom_u_adv_v_ext[nSpace],
                  dmom_u_adv_w_ext[nSpace],
                  mom_v_adv_ext[nSpace],
                  dmom_v_adv_u_ext[nSpace],
                  dmom_v_adv_v_ext[nSpace],
                  dmom_v_adv_w_ext[nSpace],
                  mom_w_adv_ext[nSpace],
                  dmom_w_adv_u_ext[nSpace],
                  dmom_w_adv_v_ext[nSpace],
                  dmom_w_adv_w_ext[nSpace],
                  mom_uu_diff_ten_ext[nSpace],
                  mom_vv_diff_ten_ext[nSpace],
                  mom_ww_diff_ten_ext[nSpace],
                  mom_uv_diff_ten_ext[1],
                  mom_uw_diff_ten_ext[1],
                  mom_vu_diff_ten_ext[1],
                  mom_vw_diff_ten_ext[1],
                  mom_wu_diff_ten_ext[1],
                  mom_wv_diff_ten_ext[1],
                  mom_u_source_ext=0.0,
                  mom_v_source_ext=0.0,
                  mom_w_source_ext=0.0,
                  mom_u_ham_ext=0.0,
                  dmom_u_ham_grad_p_ext[nSpace],
                  dmom_u_ham_grad_u_ext[nSpace],
                  mom_v_ham_ext=0.0,
                  dmom_v_ham_grad_p_ext[nSpace],
                  dmom_v_ham_grad_v_ext[nSpace],
                  mom_w_ham_ext=0.0,
                  dmom_w_ham_grad_p_ext[nSpace],
                  dmom_w_ham_grad_w_ext[nSpace],
                  dmom_u_adv_p_ext[nSpace],
                  dmom_v_adv_p_ext[nSpace],
                  dmom_w_adv_p_ext[nSpace],
                  flux_mass_ext=0.0,
                  flux_mom_u_adv_ext=0.0,
                  flux_mom_v_adv_ext=0.0,
                  flux_mom_w_adv_ext=0.0,
                  flux_mom_uu_diff_ext=0.0,
                  flux_mom_uv_diff_ext=0.0,
                  flux_mom_uw_diff_ext=0.0,
                  flux_mom_vu_diff_ext=0.0,
                  flux_mom_vv_diff_ext=0.0,
                  flux_mom_vw_diff_ext=0.0,
                  bc_p_ext=0.0,
                  bc_u_ext=0.0,
                  bc_v_ext=0.0,
                  bc_w_ext=0.0,
                  bc_mom_u_acc_ext=0.0,
                  bc_dmom_u_acc_u_ext=0.0,
                  bc_mom_v_acc_ext=0.0,
                  bc_dmom_v_acc_v_ext=0.0,
                  bc_mom_w_acc_ext=0.0,
                  bc_dmom_w_acc_w_ext=0.0,
                  bc_mass_adv_ext[nSpace],
                  bc_dmass_adv_u_ext[nSpace],
                  bc_dmass_adv_v_ext[nSpace],
                  bc_dmass_adv_w_ext[nSpace],
                  bc_mom_u_adv_ext[nSpace],
                  bc_dmom_u_adv_u_ext[nSpace],
                  bc_dmom_u_adv_v_ext[nSpace],
                  bc_dmom_u_adv_w_ext[nSpace],
                  bc_mom_v_adv_ext[nSpace],
                  bc_dmom_v_adv_u_ext[nSpace],
                  bc_dmom_v_adv_v_ext[nSpace],
                  bc_dmom_v_adv_w_ext[nSpace],
                  bc_mom_w_adv_ext[nSpace],
                  bc_dmom_w_adv_u_ext[nSpace],
                  bc_dmom_w_adv_v_ext[nSpace],
                  bc_dmom_w_adv_w_ext[nSpace],
                  bc_mom_uu_diff_ten_ext[nSpace],
                  bc_mom_vv_diff_ten_ext[nSpace],
                  bc_mom_ww_diff_ten_ext[nSpace],
                  bc_mom_uv_diff_ten_ext[1],
                  bc_mom_uw_diff_ten_ext[1],
                  bc_mom_vu_diff_ten_ext[1],
                  bc_mom_vw_diff_ten_ext[1],
                  bc_mom_wu_diff_ten_ext[1],
                  bc_mom_wv_diff_ten_ext[1],
                  bc_mom_u_source_ext=0.0,
                  bc_mom_v_source_ext=0.0,
                  bc_mom_w_source_ext=0.0,
                  bc_mom_u_ham_ext=0.0,
                  bc_dmom_u_ham_grad_p_ext[nSpace],
                  bc_dmom_u_ham_grad_u_ext[nSpace],
                  bc_mom_v_ham_ext=0.0,
                  bc_dmom_v_ham_grad_p_ext[nSpace],
                  bc_dmom_v_ham_grad_v_ext[nSpace],
                  bc_mom_w_ham_ext=0.0,
                  bc_dmom_w_ham_grad_p_ext[nSpace],
                  bc_dmom_w_ham_grad_w_ext[nSpace],
                  jac_ext[nSpace*nSpace],
                  jacDet_ext,
                  jacInv_ext[nSpace*nSpace],
                  boundaryJac[nSpace*(nSpace-1)],
                  metricTensor[(nSpace-1)*(nSpace-1)],
                  metricTensorDetSqrt,
                  dS,vel_test_dS[nDOF_test_element],
                  vel_grad_trial_trace[nDOF_trial_element*nSpace],
                  vel_grad_test_dS[nDOF_trial_element*nSpace],
                  normal[2],x_ext,y_ext,z_ext,xt_ext,yt_ext,zt_ext,integralScaling,
                  //VRANS
                  vos_ext,
                  //
                  G[nSpace*nSpace],G_dd_G,tr_G,h_phi,h_penalty,penalty,
                  force_x,force_y,force_p_x,force_p_y,force_v_x,force_v_y,r_x,r_y;
                //compute information about mapping from reference element to physical element
                ck.calculateMapping_elementBoundary(eN,
                                                    ebN_local,
                                                    kb,
                                                    ebN_local_kb,
                                                    mesh_dof.data(),
                                                    mesh_l2g.data(),
                                                    mesh_trial_trace_ref.data(),
                                                    mesh_grad_trial_trace_ref.data(),
                                                    boundaryJac_ref.data(),
                                                    jac_ext,
                                                    jacDet_ext,
                                                    jacInv_ext,
                                                    boundaryJac,
                                                    metricTensor,
                                                    metricTensorDetSqrt,
                                                    normal_ref.data(),
                                                    normal,
                                                    x_ext,y_ext,z_ext);
                ck.calculateMappingVelocity_elementBoundary(eN,
                                                            ebN_local,
                                                            kb,
                                                            ebN_local_kb,
                                                            mesh_velocity_dof.data(),
                                                            mesh_l2g.data(),
                                                            mesh_trial_trace_ref.data(),
                                                            xt_ext,yt_ext,zt_ext,
                                                            normal,
                                                            boundaryJac,
                                                            metricTensor,
                                                            integralScaling);
                //xt_ext=0.0;yt_ext=0.0;zt_ext=0.0;
                //std::cout<<"xt_ext "<<xt_ext<<'\t'<<yt_ext<<'\t'<<zt_ext<<std::endl;
                //std::cout<<"x_ext "<<x_ext<<'\t'<<y_ext<<'\t'<<z_ext<<std::endl;
                //std::cout<<"integralScaling - metricTensorDetSrt ==============================="<<integralScaling-metricTensorDetSqrt<<std::endl;
                /* std::cout<<"metricTensorDetSqrt "<<metricTensorDetSqrt */
                /*             <<"dS_ref.data()[kb]"<<dS_ref.data()[kb]<<std::endl; */
                //dS = ((1.0-MOVING_DOMAIN)*metricTensorDetSqrt + MOVING_DOMAIN*integralScaling)*dS_ref.data()[kb];//cek need to test effect on accuracy
                dS = metricTensorDetSqrt*dS_ref.data()[kb];
                //get the metric tensor
                //cek todo use symmetry
                ck.calculateG(jacInv_ext,G,G_dd_G,tr_G);
                ck.calculateGScale(G,&ebqe_normal_phi_ext.data()[ebNE_kb_nSpace],h_phi);
              
                eps_rho = epsFact_rho*(useMetrics*h_phi+(1.0-useMetrics)*elementDiameter.data()[eN]);
                eps_mu  = epsFact_mu *(useMetrics*h_phi+(1.0-useMetrics)*elementDiameter.data()[eN]);
              
                //compute shape and solution information
                //shape
                /* ck.gradTrialFromRef(&p_grad_trial_trace_ref.data()[ebN_local_kb_nSpace*nDOF_trial_element],jacInv_ext,p_grad_trial_trace); */
                ck.gradTrialFromRef(&vel_grad_trial_trace_ref.data()[ebN_local_kb_nSpace*nDOF_trial_element],jacInv_ext,vel_grad_trial_trace);
                //cek hack use trial ck.gradTrialFromRef(&vel_grad_test_trace_ref.data()[ebN_local_kb_nSpace*nDOF_trial_element],jacInv_ext,vel_grad_test_trace);
                //solution and gradients
                /* ck.valFromDOF(p_dof.data(),&p_l2g.data()[eN_nDOF_trial_element],&p_trial_trace_ref.data()[ebN_local_kb*nDOF_test_element],p_ext); */
                p_ext = ebqe_p.data()[ebNE_kb];
                ck.valFromDOF(u_dof.data(),&vel_l2g.data()[eN_nDOF_trial_element],&vel_trial_trace_ref.data()[ebN_local_kb*nDOF_test_element],u_ext);
                ck.valFromDOF(v_dof.data(),&vel_l2g.data()[eN_nDOF_trial_element],&vel_trial_trace_ref.data()[ebN_local_kb*nDOF_test_element],v_ext);
                /* ck.valFromDOF(w_dof.data(),&vel_l2g.data()[eN_nDOF_trial_element],&vel_trial_trace_ref.data()[ebN_local_kb*nDOF_test_element],w_ext); */
                /* ck.gradFromDOF(p_dof.data(),&p_l2g.data()[eN_nDOF_trial_element],p_grad_trial_trace,grad_p_ext); */
                for (int I=0;I<nSpace;I++)
                  grad_p_ext[I] = ebqe_grad_p.data()[ebNE_kb_nSpace + I];
                ck.gradFromDOF(u_dof.data(),&vel_l2g.data()[eN_nDOF_trial_element],vel_grad_trial_trace,grad_u_ext);
                ck.gradFromDOF(v_dof.data(),&vel_l2g.data()[eN_nDOF_trial_element],vel_grad_trial_trace,grad_v_ext);
                /* ck.gradFromDOF(w_dof.data(),&vel_l2g.data()[eN_nDOF_trial_element],vel_grad_trial_trace,grad_w_ext); */
                //precalculate test function products with integration weights
                for (int j=0;j<nDOF_trial_element;j++)
                  {
                    /* p_test_dS[j] = p_test_trace_ref.data()[ebN_local_kb*nDOF_test_element+j]*dS; */
                    vel_test_dS[j] = vel_test_trace_ref.data()[ebN_local_kb*nDOF_test_element+j]*dS;
                    for (int I=0;I<nSpace;I++)
                      vel_grad_test_dS[j*nSpace+I] = vel_grad_trial_trace[j*nSpace+I]*dS;//cek hack, using trial
                  }
                bc_p_ext = isDOFBoundary_p.data()[ebNE_kb]*ebqe_bc_p_ext.data()[ebNE_kb]+(1-isDOFBoundary_p.data()[ebNE_kb])*p_ext;
                //note, our convention is that bc values at moving boundaries are relative to boundary velocity so we add it here
                bc_u_ext = isDOFBoundary_u.data()[ebNE_kb]*(ebqe_bc_u_ext.data()[ebNE_kb] + MOVING_DOMAIN*xt_ext) + (1-isDOFBoundary_u.data()[ebNE_kb])*u_ext;
                bc_v_ext = isDOFBoundary_v.data()[ebNE_kb]*(ebqe_bc_v_ext.data()[ebNE_kb] + MOVING_DOMAIN*yt_ext) + (1-isDOFBoundary_v.data()[ebNE_kb])*v_ext;
                /* bc_w_ext = isDOFBoundary_w.data()[ebNE_kb]*(ebqe_bc_w_ext.data()[ebNE_kb] + MOVING_DOMAIN*zt_ext) + (1-isDOFBoundary_w.data()[ebNE_kb])*w_ext; */
                //VRANS
                vos_ext = ebqe_vos_ext.data()[ebNE_kb];//sed fraction - gco check

                //
                //calculate the pde coefficients using the solution and the boundary values for the solution 
                // 
                double eddy_viscosity_ext(0.),bc_eddy_viscosity_ext(0.); //not interested in saving boundary eddy viscosity for now
                evaluateCoefficients(eps_rho,
                                     eps_mu,
                                     sigma,
                                     rho_0,
                                     nu_0,
                                     rho_1,
                                     nu_1,
                                     rho_s,
                                     elementDiameter.data()[eN],
                                     smagorinskyConstant,
                                     turbulenceClosureModel,
                                     g.data(),
                                     useVF,
                                     ebqe_vf_ext.data()[ebNE_kb],
                                     ebqe_phi_ext.data()[ebNE_kb],
                                     &ebqe_normal_phi_ext.data()[ebNE_kb_nSpace],
                                     ebqe_kappa_phi_ext.data()[ebNE_kb],
                                     //VRANS
                                     vos_ext,
                                     //
                                     p_ext,
                                     grad_p_ext,
                                     grad_u_ext,
                                     grad_v_ext,
                                     grad_w_ext,
                                     u_ext,
                                     v_ext,
                                     w_ext,
                                     ebqe_velocity_star.data()[ebNE_kb_nSpace+0],
                                     ebqe_velocity_star.data()[ebNE_kb_nSpace+1],
                                     ebqe_velocity_star.data()[ebNE_kb_nSpace+1],//hack,not used
                                     eddy_viscosity_ext,
                                     mom_u_acc_ext,
                                     dmom_u_acc_u_ext,
                                     mom_v_acc_ext,
                                     dmom_v_acc_v_ext,
                                     mom_w_acc_ext,
                                     dmom_w_acc_w_ext,
                                     mass_adv_ext,
                                     dmass_adv_u_ext,
                                     dmass_adv_v_ext,
                                     dmass_adv_w_ext,
                                     mom_u_adv_ext,
                                     dmom_u_adv_u_ext,
                                     dmom_u_adv_v_ext,
                                     dmom_u_adv_w_ext,
                                     mom_v_adv_ext,
                                     dmom_v_adv_u_ext,
                                     dmom_v_adv_v_ext,
                                     dmom_v_adv_w_ext,
                                     mom_w_adv_ext,
                                     dmom_w_adv_u_ext,
                                     dmom_w_adv_v_ext,
                                     dmom_w_adv_w_ext,
                                     mom_uu_diff_ten_ext,
                                     mom_vv_diff_ten_ext,
                                     mom_ww_diff_ten_ext,
                                     mom_uv_diff_ten_ext,
                                     mom_uw_diff_ten_ext,
                                     mom_vu_diff_ten_ext,
                                     mom_vw_diff_ten_ext,
                                     mom_wu_diff_ten_ext,
                                     mom_wv_diff_ten_ext,
                                     mom_u_source_ext,
                                     mom_v_source_ext,
                                     mom_w_source_ext,
                                     mom_u_ham_ext,
                                     dmom_u_ham_grad_p_ext,
                                     dmom_u_ham_grad_u_ext,
                                     mom_v_ham_ext,
                                     dmom_v_ham_grad_p_ext,
                                     dmom_v_ham_grad_v_ext,
                                     mom_w_ham_ext,
                                     dmom_w_ham_grad_p_ext,          
                                     dmom_w_ham_grad_w_ext);          
                evaluateCoefficients(eps_rho,
                                     eps_mu,
                                     sigma,
                                     rho_0,
                                     nu_0,
                                     rho_1,
                                     nu_1,
                                     rho_s,
                                     elementDiameter.data()[eN],
                                     smagorinskyConstant,
                                     turbulenceClosureModel,
                                     g.data(),
                                     useVF,
                                     bc_ebqe_vf_ext.data()[ebNE_kb],
                                     bc_ebqe_phi_ext.data()[ebNE_kb],
                                     &ebqe_normal_phi_ext.data()[ebNE_kb_nSpace],
                                     ebqe_kappa_phi_ext.data()[ebNE_kb],
                                     //VRANS
                                     vos_ext,
                                     //
                                     p_ext,
                                     grad_p_ext,
                                     grad_u_ext,
                                     grad_v_ext,
                                     grad_w_ext,
                                     bc_u_ext,
                                     bc_v_ext,
                                     bc_w_ext,
                                     ebqe_velocity_star.data()[ebNE_kb_nSpace+0],
                                     ebqe_velocity_star.data()[ebNE_kb_nSpace+1],
                                     ebqe_velocity_star.data()[ebNE_kb_nSpace+1],//hack,not used
                                     bc_eddy_viscosity_ext,
                                     bc_mom_u_acc_ext, 
                                     bc_dmom_u_acc_u_ext,
                                     bc_mom_v_acc_ext,
                                     bc_dmom_v_acc_v_ext,
                                     bc_mom_w_acc_ext,
                                     bc_dmom_w_acc_w_ext,
                                     bc_mass_adv_ext,
                                     bc_dmass_adv_u_ext,
                                     bc_dmass_adv_v_ext,
                                     bc_dmass_adv_w_ext,
                                     bc_mom_u_adv_ext,
                                     bc_dmom_u_adv_u_ext,
                                     bc_dmom_u_adv_v_ext,
                                     bc_dmom_u_adv_w_ext,
                                     bc_mom_v_adv_ext,
                                     bc_dmom_v_adv_u_ext,
                                     bc_dmom_v_adv_v_ext,
                                     bc_dmom_v_adv_w_ext,
                                     bc_mom_w_adv_ext,
                                     bc_dmom_w_adv_u_ext,
                                     bc_dmom_w_adv_v_ext,
                                     bc_dmom_w_adv_w_ext,
                                     bc_mom_uu_diff_ten_ext,
                                     bc_mom_vv_diff_ten_ext,
                                     bc_mom_ww_diff_ten_ext,
                                     bc_mom_uv_diff_ten_ext,
                                     bc_mom_uw_diff_ten_ext,
                                     bc_mom_vu_diff_ten_ext,
                                     bc_mom_vw_diff_ten_ext,
                                     bc_mom_wu_diff_ten_ext,
                                     bc_mom_wv_diff_ten_ext,
                                     bc_mom_u_source_ext,
                                     bc_mom_v_source_ext,
                                     bc_mom_w_source_ext,
                                     bc_mom_u_ham_ext,
                                     bc_dmom_u_ham_grad_p_ext,
                                     bc_dmom_u_ham_grad_u_ext,
                                     bc_mom_v_ham_ext,
                                     bc_dmom_v_ham_grad_p_ext,
                                     bc_dmom_v_ham_grad_v_ext,
                                     bc_mom_w_ham_ext,
                                     bc_dmom_w_ham_grad_p_ext,          
                                     bc_dmom_w_ham_grad_w_ext);          

                //Turbulence closure model

                //
                //moving domain
                //
                mom_u_adv_ext[0] -= MOVING_DOMAIN*dmom_u_acc_u_ext*mom_u_acc_ext*xt_ext;
                mom_u_adv_ext[1] -= MOVING_DOMAIN*dmom_u_acc_u_ext*mom_u_acc_ext*yt_ext;
                /* mom_u_adv_ext[2] -= MOVING_DOMAIN*dmom_u_acc_u_ext*mom_u_acc_ext*zt_ext; */
                dmom_u_adv_u_ext[0] -= MOVING_DOMAIN*dmom_u_acc_u_ext*xt_ext;
                dmom_u_adv_u_ext[1] -= MOVING_DOMAIN*dmom_u_acc_u_ext*yt_ext;
                /* dmom_u_adv_u_ext[2] -= MOVING_DOMAIN*dmom_u_acc_u_ext*zt_ext; */

                mom_v_adv_ext[0] -= MOVING_DOMAIN*dmom_v_acc_v_ext*mom_v_acc_ext*xt_ext;
                mom_v_adv_ext[1] -= MOVING_DOMAIN*dmom_v_acc_v_ext*mom_v_acc_ext*yt_ext;
                /* mom_v_adv_ext[2] -= MOVING_DOMAIN*dmom_v_acc_v_ext*mom_v_acc_ext*zt_ext; */
                dmom_v_adv_v_ext[0] -= MOVING_DOMAIN*dmom_v_acc_v_ext*xt_ext;
                dmom_v_adv_v_ext[1] -= MOVING_DOMAIN*dmom_v_acc_v_ext*yt_ext;
                /* dmom_v_adv_v_ext[2] -= MOVING_DOMAIN*dmom_v_acc_v_ext*zt_ext; */

                /* mom_w_adv_ext[0] -= MOVING_DOMAIN*dmom_w_acc_w_ext*mom_w_acc_ext*xt_ext; */
                /* mom_w_adv_ext[1] -= MOVING_DOMAIN*dmom_w_acc_w_ext*mom_w_acc_ext*yt_ext; */
                /* mom_w_adv_ext[2] -= MOVING_DOMAIN*dmom_w_acc_w_ext*mom_w_acc_ext*zt_ext; */
                /* dmom_w_adv_w_ext[0] -= MOVING_DOMAIN*dmom_w_acc_w_ext*xt_ext; */
                /* dmom_w_adv_w_ext[1] -= MOVING_DOMAIN*dmom_w_acc_w_ext*yt_ext; */
                /* dmom_w_adv_w_ext[2] -= MOVING_DOMAIN*dmom_w_acc_w_ext*zt_ext; */

                //bc's
                // mql. CHECK.
                bc_mom_u_adv_ext[0] -= MOVING_DOMAIN*dmom_u_acc_u_ext*bc_mom_u_acc_ext*xt_ext;
                bc_mom_u_adv_ext[1] -= MOVING_DOMAIN*dmom_u_acc_u_ext*bc_mom_u_acc_ext*yt_ext;
                /* bc_mom_u_adv_ext[2] -= MOVING_DOMAIN*dmom_u_acc_u_ext*bc_mom_u_acc_ext*zt_ext; */

                bc_mom_v_adv_ext[0] -= MOVING_DOMAIN*dmom_v_acc_v_ext*bc_mom_v_acc_ext*xt_ext;
                bc_mom_v_adv_ext[1] -= MOVING_DOMAIN*dmom_v_acc_v_ext*bc_mom_v_acc_ext*yt_ext;
                /* bc_mom_v_adv_ext[2] -= MOVING_DOMAIN*dmom_v_acc_v_ext*bc_mom_v_acc_ext*zt_ext; */

                /* bc_mom_w_adv_ext[0] -= MOVING_DOMAIN*dmom_w_acc_w_ext*bc_mom_w_acc_ext*xt_ext; */
                /* bc_mom_w_adv_ext[1] -= MOVING_DOMAIN*dmom_w_acc_w_ext*bc_mom_w_acc_ext*yt_ext; */
                /* bc_mom_w_adv_ext[2] -= MOVING_DOMAIN*dmom_w_acc_w_ext*bc_mom_w_acc_ext*zt_ext; */
                //
                //calculate the numerical fluxes
                //
                ck.calculateGScale(G,normal,h_penalty);
                penalty = useMetrics*C_b/h_penalty + (1.0-useMetrics)*ebqe_penalty_ext.data()[ebNE_kb];
                exteriorNumericalAdvectiveFlux(isDOFBoundary_p.data()[ebNE_kb],
                                               isDOFBoundary_u.data()[ebNE_kb],
                                               isDOFBoundary_v.data()[ebNE_kb],
                                               isDOFBoundary_w.data()[ebNE_kb],
                                               isAdvectiveFluxBoundary_p.data()[ebNE_kb],
                                               isAdvectiveFluxBoundary_u.data()[ebNE_kb],
                                               isAdvectiveFluxBoundary_v.data()[ebNE_kb],
                                               isAdvectiveFluxBoundary_w.data()[ebNE_kb],
                                               dmom_u_ham_grad_p_ext[0],//=1/rho,
                                               bc_dmom_u_ham_grad_p_ext[0],//=1/bc_rho,
                                               normal,
                                               dmom_u_acc_u_ext,
                                               bc_p_ext,
                                               bc_u_ext,
                                               bc_v_ext,
                                               bc_w_ext,
                                               bc_mass_adv_ext,
                                               bc_mom_u_adv_ext,
                                               bc_mom_v_adv_ext,
                                               bc_mom_w_adv_ext,
                                               ebqe_bc_flux_mass_ext.data()[ebNE_kb]+MOVING_DOMAIN*(xt_ext*normal[0]+yt_ext*normal[1]),//BC is relative mass flux
                                               ebqe_bc_flux_mom_u_adv_ext.data()[ebNE_kb],
                                               ebqe_bc_flux_mom_v_adv_ext.data()[ebNE_kb],
                                               ebqe_bc_flux_mom_w_adv_ext.data()[ebNE_kb],
                                               p_ext,
                                               u_ext,
                                               v_ext,
                                               w_ext,
                                               mass_adv_ext,
                                               mom_u_adv_ext,
                                               mom_v_adv_ext,
                                               mom_w_adv_ext,
                                               dmass_adv_u_ext,
                                               dmass_adv_v_ext,
                                               dmass_adv_w_ext,
                                               dmom_u_adv_p_ext,
                                               dmom_u_adv_u_ext,
                                               dmom_u_adv_v_ext,
                                               dmom_u_adv_w_ext,
                                               dmom_v_adv_p_ext,
                                               dmom_v_adv_u_ext,
                                               dmom_v_adv_v_ext,
                                               dmom_v_adv_w_ext,
                                               dmom_w_adv_p_ext,
                                               dmom_w_adv_u_ext,
                                               dmom_w_adv_v_ext,
                                               dmom_w_adv_w_ext,
                                               flux_mass_ext,
                                               flux_mom_u_adv_ext,
                                               flux_mom_v_adv_ext,
                                               flux_mom_w_adv_ext,
                                               &ebqe_velocity_star.data()[ebNE_kb_nSpace],
                                               &ebqe_velocity.data()[ebNE_kb_nSpace]);
                exteriorNumericalDiffusiveFlux(eps_rho,
                                               ebqe_phi_ext.data()[ebNE_kb],
                                               sdInfo_u_u_rowptr.data(),
                                               sdInfo_u_u_colind.data(),
                                               isDOFBoundary_u.data()[ebNE_kb],
                                               isDiffusiveFluxBoundary_u.data()[ebNE_kb],
                                               normal,
                                               bc_mom_uu_diff_ten_ext,
                                               bc_u_ext,
                                               ebqe_bc_flux_u_diff_ext.data()[ebNE_kb],
                                               mom_uu_diff_ten_ext,
                                               grad_u_ext,
                                               u_ext,
                                               penalty,//ebqe_penalty_ext.data()[ebNE_kb],
                                               flux_mom_uu_diff_ext);
                exteriorNumericalDiffusiveFlux(eps_rho,
                                               ebqe_phi_ext.data()[ebNE_kb],
                                               sdInfo_u_v_rowptr.data(),
                                               sdInfo_u_v_colind.data(),
                                               isDOFBoundary_v.data()[ebNE_kb],
                                               isDiffusiveFluxBoundary_v.data()[ebNE_kb],
                                               normal,
                                               bc_mom_uv_diff_ten_ext,
                                               bc_v_ext,
                                               0.0,//assume all of the flux gets applied in diagonal component
                                               mom_uv_diff_ten_ext,
                                               grad_v_ext,
                                               v_ext,
                                               penalty,//ebqe_penalty_ext.data()[ebNE_kb],
                                               flux_mom_uv_diff_ext);
                /* exteriorNumericalDiffusiveFlux(eps_rho, */
                /*                                   ebqe_phi_ext.data()[ebNE_kb], */
                /*                                   sdInfo_u_w_rowptr.data(), */
                /*                                   sdInfo_u_w_colind.data(), */
                /*                                   isDOFBoundary_w.data()[ebNE_kb], */
                /*                                   isDiffusiveFluxBoundary_u.data()[ebNE_kb], */
                /*                                   normal, */
                /*                                   bc_mom_uw_diff_ten_ext, */
                /*                                   bc_w_ext, */
                /*                                   0.0,//see above */
                /*                                   mom_uw_diff_ten_ext, */
                /*                                   grad_w_ext, */
                /*                                   w_ext, */
                /*                                   penalty,//ebqe_penalty_ext.data()[ebNE_kb], */
                /*                                   flux_mom_uw_diff_ext); */
                exteriorNumericalDiffusiveFlux(eps_rho,
                                               ebqe_phi_ext.data()[ebNE_kb],
                                               sdInfo_v_u_rowptr.data(),
                                               sdInfo_v_u_colind.data(),
                                               isDOFBoundary_u.data()[ebNE_kb],
                                               isDiffusiveFluxBoundary_u.data()[ebNE_kb],
                                               normal,
                                               bc_mom_vu_diff_ten_ext,
                                               bc_u_ext,
                                               0.0,//see above
                                               mom_vu_diff_ten_ext,
                                               grad_u_ext,
                                               u_ext,
                                               penalty,//ebqe_penalty_ext.data()[ebNE_kb],
                                               flux_mom_vu_diff_ext);
                exteriorNumericalDiffusiveFlux(eps_rho,
                                               ebqe_phi_ext.data()[ebNE_kb],
                                               sdInfo_v_v_rowptr.data(),
                                               sdInfo_v_v_colind.data(),
                                               isDOFBoundary_v.data()[ebNE_kb],
                                               isDiffusiveFluxBoundary_v.data()[ebNE_kb],
                                               normal,
                                               bc_mom_vv_diff_ten_ext,
                                               bc_v_ext,
                                               ebqe_bc_flux_v_diff_ext.data()[ebNE_kb],
                                               mom_vv_diff_ten_ext,
                                               grad_v_ext,
                                               v_ext,
                                               penalty,//ebqe_penalty_ext.data()[ebNE_kb],
                                               flux_mom_vv_diff_ext);
                /* exteriorNumericalDiffusiveFlux(eps_rho, */
                /*                                   ebqe_phi_ext.data()[ebNE_kb], */
                /*                                   sdInfo_v_w_rowptr.data(), */
                /*                                   sdInfo_v_w_colind.data(), */
                /*                                   isDOFBoundary_w.data()[ebNE_kb], */
                /*                                   isDiffusiveFluxBoundary_v.data()[ebNE_kb], */
                /*                                   normal, */
                /*                                   bc_mom_vw_diff_ten_ext, */
                /*                                   bc_w_ext, */
                /*                                   0.0,//see above */
                /*                                   mom_vw_diff_ten_ext, */
                /*                                   grad_w_ext, */
                /*                                   w_ext, */
                /*                                   penalty,//ebqe_penalty_ext.data()[ebNE_kb], */
                /*                                   flux_mom_vw_diff_ext); */
                /* exteriorNumericalDiffusiveFlux(eps_rho, */
                /*                                   ebqe_phi_ext.data()[ebNE_kb], */
                /*                                   sdInfo_w_u_rowptr.data(), */
                /*                                   sdInfo_w_u_colind.data(), */
                /*                                   isDOFBoundary_u.data()[ebNE_kb], */
                /*                                   isDiffusiveFluxBoundary_w.data()[ebNE_kb], */
                /*                                   normal, */
                /*                                   bc_mom_wu_diff_ten_ext, */
                /*                                   bc_u_ext, */
                /*                                   0.0,//see above */
                /*                                   mom_wu_diff_ten_ext, */
                /*                                   grad_u_ext, */
                /*                                   u_ext, */
                /*                                   penalty,//ebqe_penalty_ext.data()[ebNE_kb], */
                /*                                   flux_mom_wu_diff_ext); */
                /* exteriorNumericalDiffusiveFlux(eps_rho, */
                /*                                   ebqe_phi_ext.data()[ebNE_kb], */
                /*                                   sdInfo_w_v_rowptr.data(), */
                /*                                   sdInfo_w_v_colind.data(), */
                /*                                   isDOFBoundary_v.data()[ebNE_kb], */
                /*                                   isDiffusiveFluxBoundary_w.data()[ebNE_kb], */
                /*                                   normal, */
                /*                                   bc_mom_wv_diff_ten_ext, */
                /*                                   bc_v_ext, */
                /*                                   0.0,//see above */
                /*                                   mom_wv_diff_ten_ext, */
                /*                                   grad_v_ext, */
                /*                                   v_ext, */
                /*                                   penalty,//ebqe_penalty_ext.data()[ebNE_kb], */
                /*                                   flux_mom_wv_diff_ext); */
                /* exteriorNumericalDiffusiveFlux(eps_rho, */
                /*                                   ebqe_phi_ext.data()[ebNE_kb], */
                /*                                   sdInfo_w_w_rowptr.data(), */
                /*                                   sdInfo_w_w_colind.data(), */
                /*                                   isDOFBoundary_w.data()[ebNE_kb], */
                /*                                   isDiffusiveFluxBoundary_w.data()[ebNE_kb], */
                /*                                   normal, */
                /*                                   bc_mom_ww_diff_ten_ext, */
                /*                                   bc_w_ext, */
                /*                                   ebqe_bc_flux_w_diff_ext.data()[ebNE_kb], */
                /*                                   mom_ww_diff_ten_ext, */
                /*                                   grad_w_ext, */
                /*                                   w_ext, */
                /*                                   penalty,//ebqe_penalty_ext.data()[ebNE_kb], */
                /*                                   flux_mom_ww_diff_ext); */
                flux.data()[ebN*nQuadraturePoints_elementBoundary+kb] = flux_mass_ext;
                /* std::cout<<"external u,v,u_n " */
                /*             <<ebqe_velocity.data()[ebNE_kb_nSpace+0]<<'\t' */
                /*             <<ebqe_velocity.data()[ebNE_kb_nSpace+1]<<'\t' */
                /*             <<flux.data()[ebN*nQuadraturePoints_elementBoundary+kb]<<std::endl; */
                //
                //integrate the net force and moment on flagged boundaries
                //
                if (ebN < nElementBoundaries_owned)
                  {
                    force_v_x = (flux_mom_u_adv_ext + flux_mom_uu_diff_ext + flux_mom_uv_diff_ext + flux_mom_uw_diff_ext)/dmom_u_ham_grad_p_ext[0];//same as *rho
                    force_v_y = (flux_mom_v_adv_ext + flux_mom_vu_diff_ext + flux_mom_vv_diff_ext + flux_mom_vw_diff_ext)/dmom_u_ham_grad_p_ext[0];
                    //force_v_z = (flux_mom_wu_diff_ext + flux_mom_wv_diff_ext + flux_mom_ww_diff_ext)/dmom_u_ham_grad_p_ext[0];
                  
                    force_p_x = p_ext*normal[0];
                    force_p_y = p_ext*normal[1];
                    //force_p_z = p_ext*normal[2];
                  
                    force_x = force_p_x + force_v_x;
                    force_y = force_p_y + force_v_y;
                    //force_z = force_p_z + force_v_z;
                  
                    r_x = x_ext - barycenters.data()[3*boundaryFlags.data()[ebN]+0];
                    r_y = y_ext - barycenters.data()[3*boundaryFlags.data()[ebN]+1];
                    //r_z = z_ext - barycenters.data()[3*boundaryFlags.data()[ebN]+2];
                  
                    wettedAreas.data()[boundaryFlags.data()[ebN]] += dS*(1.0-ebqe_vf_ext.data()[ebNE_kb]);
                  
                    netForces_p.data()[3*boundaryFlags.data()[ebN]+0] += force_p_x*dS;
                    netForces_p.data()[3*boundaryFlags.data()[ebN]+1] += force_p_y*dS;
                    //netForces_p.data()[3*boundaryFlags.data()[ebN]+2] += force_p_z*dS;
                  
                    netForces_v.data()[3*boundaryFlags.data()[ebN]+0] += force_v_x*dS;
                    netForces_v.data()[3*boundaryFlags.data()[ebN]+1] += force_v_y*dS;
                    //netForces_v.data()[3*boundaryFlags.data()[ebN]+2] += force_v_z*dS;
                  
                    //netMoments.data()[3*boundaryFlags.data()[ebN]+0] += (r_y*force_z - r_z*force_y)*dS;
                    //netMoments.data()[3*boundaryFlags.data()[ebN]+1] += (r_z*force_x - r_x*force_z)*dS;
                    netMoments.data()[3*boundaryFlags.data()[ebN]+2] += (r_x*force_y - r_y*force_x)*dS;
                  }
                //
                //update residuals
                //
                for (int i=0;i<nDOF_test_element;i++)
                  {
                    /* elementResidual_mesh[i] -= ck.ExteriorElementBoundaryFlux(MOVING_DOMAIN*(xt_ext*normal[0]+yt_ext*normal[1]),p_test_dS[i]); */
                    /* elementResidual_p.data()[i] += ck.ExteriorElementBoundaryFlux(flux_mass_ext,p_test_dS[i]); */
                    /* elementResidual_p.data()[i] -= DM*ck.ExteriorElementBoundaryFlux(MOVING_DOMAIN*(xt_ext*normal[0]+yt_ext*normal[1]),p_test_dS[i]); */
                    /* globalConservationError += ck.ExteriorElementBoundaryFlux(flux_mass_ext,p_test_dS[i]); */
                    elementResidual_u[i] +=
                      ck.ExteriorElementBoundaryFlux(flux_mom_u_adv_ext,vel_test_dS[i])+
                      ck.ExteriorElementBoundaryFlux(flux_mom_uu_diff_ext,vel_test_dS[i])+
                      ck.ExteriorElementBoundaryFlux(flux_mom_uv_diff_ext,vel_test_dS[i])+
                      ck.ExteriorElementBoundaryDiffusionAdjoint(isDOFBoundary_u.data()[ebNE_kb],
                                                                 isDiffusiveFluxBoundary_u.data()[ebNE_kb],
                                                                 eb_adjoint_sigma,
                                                                 u_ext,
                                                                 bc_u_ext,
                                                                 normal,
                                                                 sdInfo_u_u_rowptr.data(),
                                                                 sdInfo_u_u_colind.data(),
                                                                 mom_uu_diff_ten_ext,
                                                                 &vel_grad_test_dS[i*nSpace])+
                      ck.ExteriorElementBoundaryDiffusionAdjoint(isDOFBoundary_v.data()[ebNE_kb],
                                                                 isDiffusiveFluxBoundary_u.data()[ebNE_kb],
                                                                 eb_adjoint_sigma,
                                                                 v_ext,
                                                                 bc_v_ext,
                                                                 normal,
                                                                 sdInfo_u_v_rowptr.data(),
                                                                 sdInfo_u_v_colind.data(),
                                                                 mom_uv_diff_ten_ext,
                                                                 &vel_grad_test_dS[i*nSpace]);//+
                    /* ck.ExteriorElementBoundaryDiffusionAdjoint(isDOFBoundary_w.data()[ebNE_kb], */
                    /*                                         isDiffusiveFluxBoundary_u.data()[ebNE_kb], */
                    /*                                         eb_adjoint_sigma, */
                    /*                                         w_ext, */
                    /*                                         bc_w_ext, */
                    /*                                         normal, */
                    /*                                         sdInfo_u_w_rowptr.data(), */
                    /*                                         sdInfo_u_w_colind.data(), */
                    /*                                         mom_uw_diff_ten_ext, */
                    /*                                         &vel_grad_test_dS[i*nSpace]); */
                    elementResidual_v[i] += ck.ExteriorElementBoundaryFlux(flux_mom_v_adv_ext,vel_test_dS[i]) +
                      ck.ExteriorElementBoundaryFlux(flux_mom_vu_diff_ext,vel_test_dS[i])+
                      ck.ExteriorElementBoundaryFlux(flux_mom_vv_diff_ext,vel_test_dS[i])+
                      ck.ExteriorElementBoundaryDiffusionAdjoint(isDOFBoundary_u.data()[ebNE_kb],
                                                                 isDiffusiveFluxBoundary_v.data()[ebNE_kb],
                                                                 eb_adjoint_sigma,
                                                                 u_ext,
                                                                 bc_u_ext,
                                                                 normal,
                                                                 sdInfo_v_u_rowptr.data(),
                                                                 sdInfo_v_u_colind.data(),
                                                                 mom_vu_diff_ten_ext,
                                                                 &vel_grad_test_dS[i*nSpace])+
                      ck.ExteriorElementBoundaryDiffusionAdjoint(isDOFBoundary_v.data()[ebNE_kb],
                                                                 isDiffusiveFluxBoundary_v.data()[ebNE_kb],
                                                                 eb_adjoint_sigma,
                                                                 v_ext,
                                                                 bc_v_ext,
                                                                 normal,
                                                                 sdInfo_v_v_rowptr.data(),
                                                                 sdInfo_v_v_colind.data(),
                                                                 mom_vv_diff_ten_ext,
                                                                 &vel_grad_test_dS[i*nSpace]);//+
                    /* ck.ExteriorElementBoundaryDiffusionAdjoint(isDOFBoundary_w.data()[ebNE_kb], */
                    /*                                         isDiffusiveFluxBoundary_v.data()[ebNE_kb], */
                    /*                                         eb_adjoint_sigma, */
                    /*                                         w_ext, */
                    /*                                         bc_w_ext, */
                    /*                                         normal, */
                    /*                                         sdInfo_v_w_rowptr.data(), */
                    /*                                         sdInfo_v_w_colind.data(), */
                    /*                                         mom_vw_diff_ten_ext, */
                    /*                                         &vel_grad_test_dS[i*nSpace]);  */
                  
                    /* elementResidual_w[i] += ck.ExteriorElementBoundaryFlux(flux_mom_w_adv_ext,vel_test_dS[i]) + */
                    /*   ck.ExteriorElementBoundaryFlux(flux_mom_wu_diff_ext,vel_test_dS[i])+ */
                    /*   ck.ExteriorElementBoundaryFlux(flux_mom_wv_diff_ext,vel_test_dS[i])+ */
                    /*   ck.ExteriorElementBoundaryFlux(flux_mom_ww_diff_ext,vel_test_dS[i])+ */
                    /*   ck.ExteriorElementBoundaryDiffusionAdjoint(isDOFBoundary_u.data()[ebNE_kb], */
                    /*                                         isDiffusiveFluxBoundary_w.data()[ebNE_kb], */
                    /*                                         eb_adjoint_sigma, */
                    /*                                         u_ext, */
                    /*                                         bc_u_ext, */
                    /*                                         normal, */
                    /*                                         sdInfo_w_u_rowptr.data(), */
                    /*                                         sdInfo_w_u_colind.data(), */
                    /*                                         mom_wu_diff_ten_ext, */
                    /*                                         &vel_grad_test_dS[i*nSpace])+ */
                    /*   ck.ExteriorElementBoundaryDiffusionAdjoint(isDOFBoundary_v.data()[ebNE_kb], */
                    /*                                         isDiffusiveFluxBoundary_w.data()[ebNE_kb], */
                    /*                                         eb_adjoint_sigma, */
                    /*                                         v_ext, */
                    /*                                         bc_v_ext, */
                    /*                                         normal, */
                    /*                                         sdInfo_w_v_rowptr.data(), */
                    /*                                         sdInfo_w_v_colind.data(), */
                    /*                                         mom_wv_diff_ten_ext, */
                    /*                                         &vel_grad_test_dS[i*nSpace])+ */
                    /*   ck.ExteriorElementBoundaryDiffusionAdjoint(isDOFBoundary_w.data()[ebNE_kb], */
                    /*                                         isDiffusiveFluxBoundary_w.data()[ebNE_kb], */
                    /*                                         eb_adjoint_sigma, */
                    /*                                         w_ext, */
                    /*                                         bc_w_ext, */
                    /*                                         normal, */
                    /*                                         sdInfo_w_w_rowptr.data(), */
                    /*                                         sdInfo_w_w_colind.data(), */
                    /*                                         mom_ww_diff_ten_ext, */
                    /*                                         &vel_grad_test_dS[i*nSpace]);  */
                  }//i
              }//kb
            //
            //update the element and global residual storage
            //
            for (int i=0;i<nDOF_test_element;i++)
              {
                proteus_index_t eN_i = eN*nDOF_test_element+i;
              
                /* elementResidual_p_save[eN_i] +=  elementResidual_p.data()[i]; */
                /* mesh_volume_conservation_weak += elementResidual_mesh[i];               */
                /* globalResidual.data()[offset_p+stride_p*p_l2g.data()[eN_i]]+=elementResidual_p.data()[i]; */
                globalResidual.data()[offset_u+stride_u*vel_l2g.data()[eN_i]]+=elementResidual_u[i];
                globalResidual.data()[offset_v+stride_v*vel_l2g.data()[eN_i]]+=elementResidual_v[i];
                /* globalResidual.data()[offset_w+stride_w*vel_l2g.data()[eN_i]]+=elementResidual_w[i]; */
              }//i
          }//ebNE
        /* std::cout<<"mesh volume conservation = "<<mesh_volume_conservation<<std::endl; */
        /* std::cout<<"mesh volume conservation weak = "<<mesh_volume_conservation_weak<<std::endl; */
        /* std::cout<<"mesh volume conservation err max= "<<mesh_volume_conservation_err_max<<std::endl; */
        /* std::cout<<"mesh volume conservation err max weak = "<<mesh_volume_conservation_err_max_weak<<std::endl; */
      }

      void calculateJacobian(arguments_dict& args)
      {
        proteus::pyarray<double>& mesh_trial_ref = args.array<double>("mesh_trial_ref");
        proteus::pyarray<double>& mesh_grad_trial_ref = args.array<double>("mesh_grad_trial_ref");
        proteus::pyarray<double>& mesh_dof = args.array<double>("mesh_dof");
        proteus::pyarray<double>& mesh_velocity_dof = args.array<double>("mesh_velocity_dof");
        double MOVING_DOMAIN = args.scalar<double>("MOVING_DOMAIN");
        double PSTAB = args.scalar<double>("PSTAB");
        proteus::pyarray<proteus_index_t>& mesh_l2g = args.array<proteus_index_t>("mesh_l2g");
        proteus::pyarray<double>& dV_ref = args.array<double>("dV_ref");
        proteus::pyarray<double>& p_trial_ref = args.array<double>("p_trial_ref");
        proteus::pyarray<double>& p_grad_trial_ref = args.array<double>("p_grad_trial_ref");
        proteus::pyarray<double>& p_test_ref = args.array<double>("p_test_ref");
        proteus::pyarray<double>& p_grad_test_ref = args.array<double>("p_grad_test_ref");
        proteus::pyarray<double>& q_p = args.array<double>("q_p");
        proteus::pyarray<double>& q_grad_p = args.array<double>("q_grad_p");
        proteus::pyarray<double>& ebqe_p = args.array<double>("ebqe_p");
        proteus::pyarray<double>& ebqe_grad_p = args.array<double>("ebqe_grad_p");
        proteus::pyarray<double>& vel_trial_ref = args.array<double>("vel_trial_ref");
        proteus::pyarray<double>& vel_grad_trial_ref = args.array<double>("vel_grad_trial_ref");
        proteus::pyarray<double>& vel_test_ref = args.array<double>("vel_test_ref");
        proteus::pyarray<double>& vel_grad_test_ref = args.array<double>("vel_grad_test_ref");
        proteus::pyarray<double>& mesh_trial_trace_ref = args.array<double>("mesh_trial_trace_ref");
        proteus::pyarray<double>& mesh_grad_trial_trace_ref = args.array<double>("mesh_grad_trial_trace_ref");
        proteus::pyarray<double>& dS_ref = args.array<double>("dS_ref");
        proteus::pyarray<double>& p_trial_trace_ref = args.array<double>("p_trial_trace_ref");
        proteus::pyarray<double>& p_grad_trial_trace_ref = args.array<double>("p_grad_trial_trace_ref");
        proteus::pyarray<double>& p_test_trace_ref = args.array<double>("p_test_trace_ref");
        proteus::pyarray<double>& p_grad_test_trace_ref = args.array<double>("p_grad_test_trace_ref");
        proteus::pyarray<double>& vel_trial_trace_ref = args.array<double>("vel_trial_trace_ref");
        proteus::pyarray<double>& vel_grad_trial_trace_ref = args.array<double>("vel_grad_trial_trace_ref");
        proteus::pyarray<double>& vel_test_trace_ref = args.array<double>("vel_test_trace_ref");
        proteus::pyarray<double>& vel_grad_test_trace_ref = args.array<double>("vel_grad_test_trace_ref");
        proteus::pyarray<double>& normal_ref = args.array<double>("normal_ref");
        proteus::pyarray<double>& boundaryJac_ref = args.array<double>("boundaryJac_ref");
        double eb_adjoint_sigma = args.scalar<double>("eb_adjoint_sigma");
        proteus::pyarray<double>& elementDiameter = args.array<double>("elementDiameter");
        proteus::pyarray<double>& nodeDiametersArray = args.array<double>("nodeDiametersArray");
        double hFactor = args.scalar<double>("hFactor");
        proteus_index_t nElements_global = args.scalar<proteus_index_t>("nElements_global");
        double useRBLES = args.scalar<double>("useRBLES");
        double useMetrics = args.scalar<double>("useMetrics");
        double alphaBDF = args.scalar<double>("alphaBDF");
        double epsFact_rho = args.scalar<double>("epsFact_rho");
        double epsFact_mu = args.scalar<double>("epsFact_mu");
        double sigma = args.scalar<double>("sigma");
        double rho_0 = args.scalar<double>("rho_0");
        double nu_0 = args.scalar<double>("nu_0");
        double rho_1 = args.scalar<double>("rho_1");
        double nu_1 = args.scalar<double>("nu_1");
        double rho_s = args.scalar<double>("rho_s");
        double smagorinskyConstant = args.scalar<double>("smagorinskyConstant");
        int turbulenceClosureModel = args.scalar<int>("turbulenceClosureModel");
        double Ct_sge = args.scalar<double>("Ct_sge");
        double Cd_sge = args.scalar<double>("Cd_sge");
        double C_dg = args.scalar<double>("C_dg");
        double C_b = args.scalar<double>("C_b");
        const proteus::pyarray<double>& eps_solid = args.array<double>("eps_solid");
        const proteus::pyarray<double>& q_velocity_fluid = args.array<double>("q_velocity_fluid");
        const proteus::pyarray<double>& q_velocityStar_fluid = args.array<double>("q_velocityStar_fluid");
        const proteus::pyarray<double>& q_vos = args.array<double>("q_vos");
        const proteus::pyarray<double>& q_dvos_dt = args.array<double>("q_dvos_dt");
        const proteus::pyarray<double>& q_grad_vos = args.array<double>("q_grad_vos");
        const proteus::pyarray<double>& q_dragAlpha = args.array<double>("q_dragAlpha");
        const proteus::pyarray<double>& q_dragBeta = args.array<double>("q_dragBeta");
        const proteus::pyarray<double>& q_mass_source = args.array<double>("q_mass_source");
        const proteus::pyarray<double>& q_turb_var_0 = args.array<double>("q_turb_var_0");
        const proteus::pyarray<double>& q_turb_var_1 = args.array<double>("q_turb_var_1");
        const proteus::pyarray<double>& q_turb_var_grad_0 = args.array<double>("q_turb_var_grad_0");
        proteus::pyarray<proteus_index_t>& p_l2g = args.array<proteus_index_t>("p_l2g");
        proteus::pyarray<proteus_index_t>& vel_l2g = args.array<proteus_index_t>("vel_l2g");
        proteus::pyarray<double>& p_dof = args.array<double>("p_dof");
        proteus::pyarray<double>& u_dof = args.array<double>("u_dof");
        proteus::pyarray<double>& v_dof = args.array<double>("v_dof");
        proteus::pyarray<double>& w_dof = args.array<double>("w_dof");
        proteus::pyarray<double>& g = args.array<double>("g");
        const double useVF = args.scalar<double>("useVF");
        proteus::pyarray<double>& vf = args.array<double>("vf");
        proteus::pyarray<double>& phi = args.array<double>("phi");
        proteus::pyarray<double>& normal_phi = args.array<double>("normal_phi");
        proteus::pyarray<double>& kappa_phi = args.array<double>("kappa_phi");
        proteus::pyarray<double>& q_mom_u_acc_beta_bdf = args.array<double>("q_mom_u_acc_beta_bdf");
        proteus::pyarray<double>& q_mom_v_acc_beta_bdf = args.array<double>("q_mom_v_acc_beta_bdf");
        proteus::pyarray<double>& q_mom_w_acc_beta_bdf = args.array<double>("q_mom_w_acc_beta_bdf");
        proteus::pyarray<double>& q_dV = args.array<double>("q_dV");
        proteus::pyarray<double>& q_dV_last = args.array<double>("q_dV_last");
        proteus::pyarray<double>& q_velocity_sge = args.array<double>("q_velocity_sge");
        proteus::pyarray<double>& ebqe_velocity_star = args.array<double>("ebqe_velocity_star");
        proteus::pyarray<double>& q_cfl = args.array<double>("q_cfl");
        proteus::pyarray<double>& q_numDiff_u_last = args.array<double>("q_numDiff_u_last");
        proteus::pyarray<double>& q_numDiff_v_last = args.array<double>("q_numDiff_v_last");
        proteus::pyarray<double>& q_numDiff_w_last = args.array<double>("q_numDiff_w_last");
        proteus::pyarray<proteus_csr_index_t>& sdInfo_u_u_rowptr = args.array<proteus_csr_index_t>("sdInfo_u_u_rowptr");
        proteus::pyarray<proteus_csr_index_t>& sdInfo_u_u_colind = args.array<proteus_csr_index_t>("sdInfo_u_u_colind");
        proteus::pyarray<proteus_csr_index_t>& sdInfo_u_v_rowptr = args.array<proteus_csr_index_t>("sdInfo_u_v_rowptr");
        proteus::pyarray<proteus_csr_index_t>& sdInfo_u_v_colind = args.array<proteus_csr_index_t>("sdInfo_u_v_colind");
        proteus::pyarray<proteus_csr_index_t>& sdInfo_u_w_rowptr = args.array<proteus_csr_index_t>("sdInfo_u_w_rowptr");
        proteus::pyarray<proteus_csr_index_t>& sdInfo_u_w_colind = args.array<proteus_csr_index_t>("sdInfo_u_w_colind");
        proteus::pyarray<proteus_csr_index_t>& sdInfo_v_v_rowptr = args.array<proteus_csr_index_t>("sdInfo_v_v_rowptr");
        proteus::pyarray<proteus_csr_index_t>& sdInfo_v_v_colind = args.array<proteus_csr_index_t>("sdInfo_v_v_colind");
        proteus::pyarray<proteus_csr_index_t>& sdInfo_v_u_rowptr = args.array<proteus_csr_index_t>("sdInfo_v_u_rowptr");
        proteus::pyarray<proteus_csr_index_t>& sdInfo_v_u_colind = args.array<proteus_csr_index_t>("sdInfo_v_u_colind");
        proteus::pyarray<proteus_csr_index_t>& sdInfo_v_w_rowptr = args.array<proteus_csr_index_t>("sdInfo_v_w_rowptr");
        proteus::pyarray<proteus_csr_index_t>& sdInfo_v_w_colind = args.array<proteus_csr_index_t>("sdInfo_v_w_colind");
        proteus::pyarray<proteus_csr_index_t>& sdInfo_w_w_rowptr = args.array<proteus_csr_index_t>("sdInfo_w_w_rowptr");
        proteus::pyarray<proteus_csr_index_t>& sdInfo_w_w_colind = args.array<proteus_csr_index_t>("sdInfo_w_w_colind");
        proteus::pyarray<proteus_csr_index_t>& sdInfo_w_u_rowptr = args.array<proteus_csr_index_t>("sdInfo_w_u_rowptr");
        proteus::pyarray<proteus_csr_index_t>& sdInfo_w_u_colind = args.array<proteus_csr_index_t>("sdInfo_w_u_colind");
        proteus::pyarray<proteus_csr_index_t>& sdInfo_w_v_rowptr = args.array<proteus_csr_index_t>("sdInfo_w_v_rowptr");
        proteus::pyarray<proteus_csr_index_t>& sdInfo_w_v_colind = args.array<proteus_csr_index_t>("sdInfo_w_v_colind");
        proteus::pyarray<proteus_csr_index_t>& csrRowIndeces_p_p = args.array<proteus_csr_index_t>("csrRowIndeces_p_p");
        proteus::pyarray<proteus_csr_index_t>& csrColumnOffsets_p_p = args.array<proteus_csr_index_t>("csrColumnOffsets_p_p");
        proteus::pyarray<proteus_csr_index_t>& csrRowIndeces_p_u = args.array<proteus_csr_index_t>("csrRowIndeces_p_u");
        proteus::pyarray<proteus_csr_index_t>& csrColumnOffsets_p_u = args.array<proteus_csr_index_t>("csrColumnOffsets_p_u");
        proteus::pyarray<proteus_csr_index_t>& csrRowIndeces_p_v = args.array<proteus_csr_index_t>("csrRowIndeces_p_v");
        proteus::pyarray<proteus_csr_index_t>& csrColumnOffsets_p_v = args.array<proteus_csr_index_t>("csrColumnOffsets_p_v");
        proteus::pyarray<proteus_csr_index_t>& csrRowIndeces_p_w = args.array<proteus_csr_index_t>("csrRowIndeces_p_w");
        proteus::pyarray<proteus_csr_index_t>& csrColumnOffsets_p_w = args.array<proteus_csr_index_t>("csrColumnOffsets_p_w");
        proteus::pyarray<proteus_csr_index_t>& csrRowIndeces_u_p = args.array<proteus_csr_index_t>("csrRowIndeces_u_p");
        proteus::pyarray<proteus_csr_index_t>& csrColumnOffsets_u_p = args.array<proteus_csr_index_t>("csrColumnOffsets_u_p");
        proteus::pyarray<proteus_csr_index_t>& csrRowIndeces_u_u = args.array<proteus_csr_index_t>("csrRowIndeces_u_u");
        proteus::pyarray<proteus_csr_index_t>& csrColumnOffsets_u_u = args.array<proteus_csr_index_t>("csrColumnOffsets_u_u");
        proteus::pyarray<proteus_csr_index_t>& csrRowIndeces_u_v = args.array<proteus_csr_index_t>("csrRowIndeces_u_v");
        proteus::pyarray<proteus_csr_index_t>& csrColumnOffsets_u_v = args.array<proteus_csr_index_t>("csrColumnOffsets_u_v");
        proteus::pyarray<proteus_csr_index_t>& csrRowIndeces_u_w = args.array<proteus_csr_index_t>("csrRowIndeces_u_w");
        proteus::pyarray<proteus_csr_index_t>& csrColumnOffsets_u_w = args.array<proteus_csr_index_t>("csrColumnOffsets_u_w");
        proteus::pyarray<proteus_csr_index_t>& csrRowIndeces_v_p = args.array<proteus_csr_index_t>("csrRowIndeces_v_p");
        proteus::pyarray<proteus_csr_index_t>& csrColumnOffsets_v_p = args.array<proteus_csr_index_t>("csrColumnOffsets_v_p");
        proteus::pyarray<proteus_csr_index_t>& csrRowIndeces_v_u = args.array<proteus_csr_index_t>("csrRowIndeces_v_u");
        proteus::pyarray<proteus_csr_index_t>& csrColumnOffsets_v_u = args.array<proteus_csr_index_t>("csrColumnOffsets_v_u");
        proteus::pyarray<proteus_csr_index_t>& csrRowIndeces_v_v = args.array<proteus_csr_index_t>("csrRowIndeces_v_v");
        proteus::pyarray<proteus_csr_index_t>& csrColumnOffsets_v_v = args.array<proteus_csr_index_t>("csrColumnOffsets_v_v");
        proteus::pyarray<proteus_csr_index_t>& csrRowIndeces_v_w = args.array<proteus_csr_index_t>("csrRowIndeces_v_w");
        proteus::pyarray<proteus_csr_index_t>& csrColumnOffsets_v_w = args.array<proteus_csr_index_t>("csrColumnOffsets_v_w");
        proteus::pyarray<proteus_csr_index_t>& csrRowIndeces_w_p = args.array<proteus_csr_index_t>("csrRowIndeces_w_p");
        proteus::pyarray<proteus_csr_index_t>& csrColumnOffsets_w_p = args.array<proteus_csr_index_t>("csrColumnOffsets_w_p");
        proteus::pyarray<proteus_csr_index_t>& csrRowIndeces_w_u = args.array<proteus_csr_index_t>("csrRowIndeces_w_u");
        proteus::pyarray<proteus_csr_index_t>& csrColumnOffsets_w_u = args.array<proteus_csr_index_t>("csrColumnOffsets_w_u");
        proteus::pyarray<proteus_csr_index_t>& csrRowIndeces_w_v = args.array<proteus_csr_index_t>("csrRowIndeces_w_v");
        proteus::pyarray<proteus_csr_index_t>& csrColumnOffsets_w_v = args.array<proteus_csr_index_t>("csrColumnOffsets_w_v");
        proteus::pyarray<proteus_csr_index_t>& csrRowIndeces_w_w = args.array<proteus_csr_index_t>("csrRowIndeces_w_w");
        proteus::pyarray<proteus_csr_index_t>& csrColumnOffsets_w_w = args.array<proteus_csr_index_t>("csrColumnOffsets_w_w");
        proteus::pyarray<double>& globalJacobian = args.array<double>("globalJacobian");
        proteus_index_t nExteriorElementBoundaries_global = args.scalar<proteus_index_t>("nExteriorElementBoundaries_global");
        proteus::pyarray<proteus_index_t>& exteriorElementBoundariesArray = args.array<proteus_index_t>("exteriorElementBoundariesArray");
        proteus::pyarray<proteus_index_t>& elementBoundaryElementsArray = args.array<proteus_index_t>("elementBoundaryElementsArray");
        proteus::pyarray<int>& elementBoundaryLocalElementBoundariesArray = args.array<int>("elementBoundaryLocalElementBoundariesArray");
        proteus::pyarray<double>& ebqe_vf_ext = args.array<double>("ebqe_vf_ext");
        proteus::pyarray<double>& bc_ebqe_vf_ext = args.array<double>("bc_ebqe_vf_ext");
        proteus::pyarray<double>& ebqe_phi_ext = args.array<double>("ebqe_phi_ext");
        proteus::pyarray<double>& bc_ebqe_phi_ext = args.array<double>("bc_ebqe_phi_ext");
        proteus::pyarray<double>& ebqe_normal_phi_ext = args.array<double>("ebqe_normal_phi_ext");
        proteus::pyarray<double>& ebqe_kappa_phi_ext = args.array<double>("ebqe_kappa_phi_ext");
        const proteus::pyarray<double>& ebqe_vos_ext = args.array<double>("ebqe_vos_ext");
        const proteus::pyarray<double>& ebqe_turb_var_0 = args.array<double>("ebqe_turb_var_0");
        const proteus::pyarray<double>& ebqe_turb_var_1 = args.array<double>("ebqe_turb_var_1");
        proteus::pyarray<int>& isDOFBoundary_p = args.array<int>("isDOFBoundary_p");
        proteus::pyarray<int>& isDOFBoundary_u = args.array<int>("isDOFBoundary_u");
        proteus::pyarray<int>& isDOFBoundary_v = args.array<int>("isDOFBoundary_v");
        proteus::pyarray<int>& isDOFBoundary_w = args.array<int>("isDOFBoundary_w");
        proteus::pyarray<int>& isAdvectiveFluxBoundary_p = args.array<int>("isAdvectiveFluxBoundary_p");
        proteus::pyarray<int>& isAdvectiveFluxBoundary_u = args.array<int>("isAdvectiveFluxBoundary_u");
        proteus::pyarray<int>& isAdvectiveFluxBoundary_v = args.array<int>("isAdvectiveFluxBoundary_v");
        proteus::pyarray<int>& isAdvectiveFluxBoundary_w = args.array<int>("isAdvectiveFluxBoundary_w");
        proteus::pyarray<int>& isDiffusiveFluxBoundary_u = args.array<int>("isDiffusiveFluxBoundary_u");
        proteus::pyarray<int>& isDiffusiveFluxBoundary_v = args.array<int>("isDiffusiveFluxBoundary_v");
        proteus::pyarray<int>& isDiffusiveFluxBoundary_w = args.array<int>("isDiffusiveFluxBoundary_w");
        proteus::pyarray<double>& ebqe_bc_p_ext = args.array<double>("ebqe_bc_p_ext");
        proteus::pyarray<double>& ebqe_bc_flux_mass_ext = args.array<double>("ebqe_bc_flux_mass_ext");
        proteus::pyarray<double>& ebqe_bc_flux_mom_u_adv_ext = args.array<double>("ebqe_bc_flux_mom_u_adv_ext");
        proteus::pyarray<double>& ebqe_bc_flux_mom_v_adv_ext = args.array<double>("ebqe_bc_flux_mom_v_adv_ext");
        proteus::pyarray<double>& ebqe_bc_flux_mom_w_adv_ext = args.array<double>("ebqe_bc_flux_mom_w_adv_ext");
        proteus::pyarray<double>& ebqe_bc_u_ext = args.array<double>("ebqe_bc_u_ext");
        proteus::pyarray<double>& ebqe_bc_flux_u_diff_ext = args.array<double>("ebqe_bc_flux_u_diff_ext");
        proteus::pyarray<double>& ebqe_penalty_ext = args.array<double>("ebqe_penalty_ext");
        proteus::pyarray<double>& ebqe_bc_v_ext = args.array<double>("ebqe_bc_v_ext");
        proteus::pyarray<double>& ebqe_bc_flux_v_diff_ext = args.array<double>("ebqe_bc_flux_v_diff_ext");
        proteus::pyarray<double>& ebqe_bc_w_ext = args.array<double>("ebqe_bc_w_ext");
        proteus::pyarray<double>& ebqe_bc_flux_w_diff_ext = args.array<double>("ebqe_bc_flux_w_diff_ext");
        proteus::pyarray<proteus_csr_index_t>& csrColumnOffsets_eb_p_p = args.array<proteus_csr_index_t>("csrColumnOffsets_eb_p_p");
        proteus::pyarray<proteus_csr_index_t>& csrColumnOffsets_eb_p_u = args.array<proteus_csr_index_t>("csrColumnOffsets_eb_p_u");
        proteus::pyarray<proteus_csr_index_t>& csrColumnOffsets_eb_p_v = args.array<proteus_csr_index_t>("csrColumnOffsets_eb_p_v");
        proteus::pyarray<proteus_csr_index_t>& csrColumnOffsets_eb_p_w = args.array<proteus_csr_index_t>("csrColumnOffsets_eb_p_w");
        proteus::pyarray<proteus_csr_index_t>& csrColumnOffsets_eb_u_p = args.array<proteus_csr_index_t>("csrColumnOffsets_eb_u_p");
        proteus::pyarray<proteus_csr_index_t>& csrColumnOffsets_eb_u_u = args.array<proteus_csr_index_t>("csrColumnOffsets_eb_u_u");
        proteus::pyarray<proteus_csr_index_t>& csrColumnOffsets_eb_u_v = args.array<proteus_csr_index_t>("csrColumnOffsets_eb_u_v");
        proteus::pyarray<proteus_csr_index_t>& csrColumnOffsets_eb_u_w = args.array<proteus_csr_index_t>("csrColumnOffsets_eb_u_w");
        proteus::pyarray<proteus_csr_index_t>& csrColumnOffsets_eb_v_p = args.array<proteus_csr_index_t>("csrColumnOffsets_eb_v_p");
        proteus::pyarray<proteus_csr_index_t>& csrColumnOffsets_eb_v_u = args.array<proteus_csr_index_t>("csrColumnOffsets_eb_v_u");
        proteus::pyarray<proteus_csr_index_t>& csrColumnOffsets_eb_v_v = args.array<proteus_csr_index_t>("csrColumnOffsets_eb_v_v");
        proteus::pyarray<proteus_csr_index_t>& csrColumnOffsets_eb_v_w = args.array<proteus_csr_index_t>("csrColumnOffsets_eb_v_w");
        proteus::pyarray<proteus_csr_index_t>& csrColumnOffsets_eb_w_p = args.array<proteus_csr_index_t>("csrColumnOffsets_eb_w_p");
        proteus::pyarray<proteus_csr_index_t>& csrColumnOffsets_eb_w_u = args.array<proteus_csr_index_t>("csrColumnOffsets_eb_w_u");
        proteus::pyarray<proteus_csr_index_t>& csrColumnOffsets_eb_w_v = args.array<proteus_csr_index_t>("csrColumnOffsets_eb_w_v");
        proteus::pyarray<proteus_csr_index_t>& csrColumnOffsets_eb_w_w = args.array<proteus_csr_index_t>("csrColumnOffsets_eb_w_w");
        proteus::pyarray<int>& elementFlags = args.array<int>("elementFlags");
        double LAG_MU_FR = args.scalar<double>("LAG_MU_FR");
        proteus::pyarray<double>& q_mu_fr_last = args.array<double>("q_mu_fr_last");
        proteus::pyarray<double>& q_mu_fr = args.array<double>("q_mu_fr");
        //
        //loop over elements to compute volume integrals and load them into the element Jacobians and global Jacobian
        //
        for(proteus_index_t eN=0;eN<nElements_global;eN++)
          {
            double eps_rho,eps_mu;

            double elementJacobian_u_u[nDOF_test_element][nDOF_trial_element],
              elementJacobian_u_v[nDOF_test_element][nDOF_trial_element],
              elementJacobian_v_u[nDOF_test_element][nDOF_trial_element],
              elementJacobian_v_v[nDOF_test_element][nDOF_trial_element];
            for (int i=0;i<nDOF_test_element;i++)
              for (int j=0;j<nDOF_trial_element;j++)

                {
                  elementJacobian_u_u[i][j]=0.0;
                  elementJacobian_u_v[i][j]=0.0;
                  elementJacobian_v_u[i][j]=0.0;
                  elementJacobian_v_v[i][j]=0.0;
                }

            for  (int k=0;k<nQuadraturePoints_element;k++)
              {
                proteus_index_t eN_k = eN*nQuadraturePoints_element+k, //index to a scalar at a quadrature point
                  eN_k_nSpace = eN_k*nSpace,
                  eN_nDOF_trial_element = eN*nDOF_trial_element; //index to a vector at a quadrature point

                //declare local storage
                double p=0.0,u=0.0,v=0.0,w=0.0,
                  grad_p[nSpace],grad_u[nSpace],grad_v[nSpace],grad_w[nSpace],grad_vos[nSpace],
                  mom_u_acc=0.0,
                  dmom_u_acc_u=0.0,
                  mom_v_acc=0.0,
                  dmom_v_acc_v=0.0,
                  mom_w_acc=0.0,
                  dmom_w_acc_w=0.0,
                  mass_adv[nSpace],
                  dmass_adv_u[nSpace],
                  dmass_adv_v[nSpace],
                  dmass_adv_w[nSpace],
                  mom_u_adv[nSpace],
                  dmom_u_adv_u[nSpace],
                  dmom_u_adv_v[nSpace],
                  dmom_u_adv_w[nSpace],
                  mom_v_adv[nSpace],
                  dmom_v_adv_u[nSpace],
                  dmom_v_adv_v[nSpace],
                  dmom_v_adv_w[nSpace],
                  mom_w_adv[nSpace],
                  dmom_w_adv_u[nSpace],
                  dmom_w_adv_v[nSpace],
                  dmom_w_adv_w[nSpace],
                  mom_uu_diff_ten[nSpace],
                  mom_vv_diff_ten[nSpace],
                  mom_ww_diff_ten[nSpace],
                  mom_uv_diff_ten[1],
                  mom_uw_diff_ten[1],
                  mom_vu_diff_ten[1],
                  mom_vw_diff_ten[1],
                  mom_wu_diff_ten[1],
                  mom_wv_diff_ten[1],
                  mom_u_source=0.0,
                  mom_v_source=0.0,
                  mom_w_source=0.0,
                  mom_u_ham=0.0,
                  dmom_u_ham_grad_p[nSpace],
                  dmom_u_ham_grad_u[nSpace],
                  mom_v_ham=0.0,
                  dmom_v_ham_grad_p[nSpace],
                  dmom_v_ham_grad_v[nSpace],
                  mom_w_ham=0.0,
                  dmom_w_ham_grad_p[nSpace],
                  dmom_w_ham_grad_w[nSpace],
                  mom_u_acc_t=0.0,
                  dmom_u_acc_u_t=0.0,
                  mom_v_acc_t=0.0,
                  dmom_v_acc_v_t=0.0,
                  pdeResidual_p=0.0,
                  pdeResidual_u=0.0,
                  pdeResidual_v=0.0,
                  pdeResidual_w=0.0,        
                  dpdeResidual_p_u[nDOF_trial_element],dpdeResidual_p_v[nDOF_trial_element],dpdeResidual_p_w[nDOF_trial_element],
                  dpdeResidual_u_p[nDOF_trial_element],dpdeResidual_u_u[nDOF_trial_element],
                  dpdeResidual_v_p[nDOF_trial_element],dpdeResidual_v_v[nDOF_trial_element],
                  dpdeResidual_w_p[nDOF_trial_element],dpdeResidual_w_w[nDOF_trial_element],
                  Lstar_u_u[nDOF_test_element],
                  Lstar_v_v[nDOF_test_element],
                  subgridError_p=0.0,
                  subgridError_u=0.0,
                  subgridError_v=0.0,
                  subgridError_w=0.0,       
                  dsubgridError_p_u[nDOF_trial_element],
                  dsubgridError_p_v[nDOF_trial_element],
                  dsubgridError_p_w[nDOF_trial_element],
                  dsubgridError_u_p[nDOF_trial_element],
                  dsubgridError_u_u[nDOF_trial_element],
                  dsubgridError_v_p[nDOF_trial_element],
                  dsubgridError_v_v[nDOF_trial_element],
                  dsubgridError_w_p[nDOF_trial_element],
                  dsubgridError_w_w[nDOF_trial_element],
                  tau_p=0.0,tau_p0=0.0,tau_p1=0.0,
                  tau_v=0.0,tau_v0=0.0,tau_v1=0.0,
                  jac[nSpace*nSpace],
                  jacDet,
                  jacInv[nSpace*nSpace],
                  p_grad_trial[nDOF_trial_element*nSpace],vel_grad_trial[nDOF_trial_element*nSpace],
                  dV,
                  vel_test_dV[nDOF_test_element],
                  vel_grad_test_dV[nDOF_test_element*nSpace],
                  x,y,z,xt,yt,zt,
                  //VRANS
                  vos,
                  //meanGrainSize,
                  dmom_u_source[nSpace],
                  dmom_v_source[nSpace],
                  dmom_w_source[nSpace],
                  mass_source,
                  //
                  G[nSpace*nSpace],G_dd_G,tr_G,h_phi, dmom_adv_star[nSpace], dmom_adv_sge[nSpace];
                //get jacobian, etc for mapping reference element
                ck.calculateMapping_element(eN,
                                            k,
                                            mesh_dof.data(),
                                            mesh_l2g.data(),
                                            mesh_trial_ref.data(),
                                            mesh_grad_trial_ref.data(),
                                            jac,
                                            jacDet,
                                            jacInv,
                                            x,y,z);
                ck.calculateH_element(eN,
                                      k,
                                      nodeDiametersArray.data(),
                                      mesh_l2g.data(),
                                      mesh_trial_ref.data(),
                                      h_phi);
                ck.calculateMappingVelocity_element(eN,
                                                    k,
                                                    mesh_velocity_dof.data(),
                                                    mesh_l2g.data(),
                                                    mesh_trial_ref.data(),
                                                    xt,yt,zt);
                //xt=0.0;yt=0.0;zt=0.0;
                //std::cout<<"xt "<<xt<<'\t'<<yt<<'\t'<<zt<<std::endl;
                //get the physical integration weight
                dV = fabs(jacDet)*dV_ref.data()[k];
                ck.calculateG(jacInv,G,G_dd_G,tr_G);
                //ck.calculateGScale(G,&normal_phi.data()[eN_k_nSpace],h_phi);
        
                eps_rho = epsFact_rho*(useMetrics*h_phi+(1.0-useMetrics)*elementDiameter.data()[eN]);
                eps_mu  = epsFact_mu *(useMetrics*h_phi+(1.0-useMetrics)*elementDiameter.data()[eN]);
              
                //get the trial function gradients
                /* ck.gradTrialFromRef(&p_grad_trial_ref.data()[k*nDOF_trial_element*nSpace],jacInv,p_grad_trial); */
                ck.gradTrialFromRef(&vel_grad_trial_ref.data()[k*nDOF_trial_element*nSpace],jacInv,vel_grad_trial);
                //get the solution        
                /* ck.valFromDOF(p_dof.data(),&p_l2g.data()[eN_nDOF_trial_element],&p_trial_ref.data()[k*nDOF_trial_element],p); */
                p = q_p.data()[eN_k];
                ck.valFromDOF(u_dof.data(),&vel_l2g.data()[eN_nDOF_trial_element],&vel_trial_ref.data()[k*nDOF_trial_element],u);
                ck.valFromDOF(v_dof.data(),&vel_l2g.data()[eN_nDOF_trial_element],&vel_trial_ref.data()[k*nDOF_trial_element],v);
                /* ck.valFromDOF(w_dof.data(),&vel_l2g.data()[eN_nDOF_trial_element],&vel_trial_ref.data()[k*nDOF_trial_element],w); */
                //get the solution gradients
                /* ck.gradFromDOF(p_dof.data(),&p_l2g.data()[eN_nDOF_trial_element],p_grad_trial,grad_p); */
                for (int I=0;I<nSpace;I++)
                  grad_p[I] = q_grad_p.data()[eN_k_nSpace+I];
                for (int I=0;I<nSpace;I++)
                  grad_vos[I] = q_grad_vos.data()[eN_k_nSpace + I];
                ck.gradFromDOF(u_dof.data(),&vel_l2g.data()[eN_nDOF_trial_element],vel_grad_trial,grad_u);
                ck.gradFromDOF(v_dof.data(),&vel_l2g.data()[eN_nDOF_trial_element],vel_grad_trial,grad_v);
                /* ck.gradFromDOF(w_dof.data(),&vel_l2g.data()[eN_nDOF_trial_element],vel_grad_trial,grad_w); */
                //VRANS
                vos = q_vos.data()[eN_k];//sed fraction - gco check
                //precalculate test function products with integration weights
                for (int j=0;j<nDOF_trial_element;j++)
                  {
                    /* p_test_dV[j] = p_test_ref.data()[k*nDOF_trial_element+j]*dV; */
                    vel_test_dV[j] = vel_test_ref.data()[k*nDOF_trial_element+j]*dV;
                    for (int I=0;I<nSpace;I++)
                      {
                        /* p_grad_test_dV[j*nSpace+I]   = p_grad_trial[j*nSpace+I]*dV;//cek warning won't work for Petrov-Galerkin */
                        vel_grad_test_dV[j*nSpace+I] = vel_grad_trial[j*nSpace+I]*dV;//cek warning won't work for Petrov-Galerkin}
                      }
                  }
                //cek hack
                double div_mesh_velocity=0.0;
                int NDOF_MESH_TRIAL_ELEMENT=3;
                for (int j=0;j<NDOF_MESH_TRIAL_ELEMENT;j++)
                  {
                    proteus_index_t eN_j=eN*NDOF_MESH_TRIAL_ELEMENT+j;
                    div_mesh_velocity +=
                      mesh_velocity_dof.data()[mesh_l2g.data()[eN_j]*3+0]*vel_grad_trial[j*2+0] +
                      mesh_velocity_dof.data()[mesh_l2g.data()[eN_j]*3+1]*vel_grad_trial[j*2+1];
                  }
                div_mesh_velocity = DM3*div_mesh_velocity + (1.0-DM3)*alphaBDF*(dV-q_dV_last.data()[eN_k])/dV;
                //

                //
                //
                //calculate pde coefficients and derivatives at quadrature points
                //
                double eddy_viscosity(0.);//not really interested in saving eddy_viscosity in jacobian
                evaluateCoefficients(eps_rho,
                                     eps_mu,
                                     sigma,
                                     rho_0,
                                     nu_0,
                                     rho_1,
                                     nu_1,
                                     rho_s,
                                     elementDiameter.data()[eN],
                                     smagorinskyConstant,
                                     turbulenceClosureModel,
                                     g.data(),
                                     useVF,
                                     vf.data()[eN_k],
                                     phi.data()[eN_k],
                                     &normal_phi.data()[eN_k_nSpace],
                                     kappa_phi.data()[eN_k],
                                     //VRANS
                                     vos,
                                     //
                                     p,
                                     grad_p,
                                     grad_u,
                                     grad_v,
                                     grad_w,
                                     u,
                                     v,
                                     w,
                                     q_velocity_sge.data()[eN_k_nSpace+0],
                                     q_velocity_sge.data()[eN_k_nSpace+1],
                                     q_velocity_sge.data()[eN_k_nSpace+1],//hack, shouldn't be used
                                     eddy_viscosity,
                                     mom_u_acc,
                                     dmom_u_acc_u,
                                     mom_v_acc,
                                     dmom_v_acc_v,
                                     mom_w_acc,
                                     dmom_w_acc_w,
                                     mass_adv,
                                     dmass_adv_u,
                                     dmass_adv_v,
                                     dmass_adv_w,
                                     mom_u_adv,
                                     dmom_u_adv_u,
                                     dmom_u_adv_v,
                                     dmom_u_adv_w,
                                     mom_v_adv,
                                     dmom_v_adv_u,
                                     dmom_v_adv_v,
                                     dmom_v_adv_w,
                                     mom_w_adv,
                                     dmom_w_adv_u,
                                     dmom_w_adv_v,
                                     dmom_w_adv_w,
                                     mom_uu_diff_ten,
                                     mom_vv_diff_ten,
                                     mom_ww_diff_ten,
                                     mom_uv_diff_ten,
                                     mom_uw_diff_ten,
                                     mom_vu_diff_ten,
                                     mom_vw_diff_ten,
                                     mom_wu_diff_ten,
                                     mom_wv_diff_ten,
                                     mom_u_source,
                                     mom_v_source,
                                     mom_w_source,
                                     mom_u_ham,
                                     dmom_u_ham_grad_p,
                                     dmom_u_ham_grad_u,
                                     mom_v_ham,
                                     dmom_v_ham_grad_p,
                                     dmom_v_ham_grad_v,
                                     mom_w_ham,
                                     dmom_w_ham_grad_p,          
                                     dmom_w_ham_grad_w);          
                //VRANS
                mass_source = q_mass_source.data()[eN_k];
                for (int I=0;I<nSpace;I++)
                  {
                    dmom_u_source[I] = 0.0;
                    dmom_v_source[I] = 0.0;
                    dmom_w_source[I] = 0.0;
                  }
                updateDarcyForchheimerTerms_Ergun(// linearDragFactor,
                                                  // nonlinearDragFactor,
                                                  // vos,
                                                  // meanGrainSize,
                                                  q_dragAlpha.data()[eN_k],
                                                  q_dragBeta.data()[eN_k],
                                                  eps_rho,
                                                  eps_mu,
                                                  rho_0,
                                                  nu_0,
                                                  rho_1,
                                                  nu_1,
                                                  eddy_viscosity,
                                                  useVF,
                                                  vf.data()[eN_k],
                                                  phi.data()[eN_k],
                                                  u,
                                                  v,
                                                  w,
                                                  q_velocity_sge.data()[eN_k_nSpace+0],
                                                  q_velocity_sge.data()[eN_k_nSpace+1],
                                                  q_velocity_sge.data()[eN_k_nSpace+1],//hack, shouldn't  be used
                                                  eps_solid.data()[elementFlags.data()[eN]],
                                                  vos,
                                                  q_velocity_fluid.data()[eN_k_nSpace+0],
                                                  q_velocity_fluid.data()[eN_k_nSpace+1],
                                                  q_velocity_fluid.data()[eN_k_nSpace+1],
                                                  q_velocityStar_fluid.data()[eN_k_nSpace+0],
                                                  q_velocityStar_fluid.data()[eN_k_nSpace+1],
                                                  q_velocityStar_fluid.data()[eN_k_nSpace+1],
                                                  mom_u_source,
                                                  mom_v_source,
                                                  mom_w_source,
                                                  dmom_u_source,
                                                  dmom_v_source,
                                                  dmom_w_source,
                                                  q_grad_vos.data()[eN_k_nSpace+0],
                                                  q_grad_vos.data()[eN_k_nSpace+1],
                                                  q_grad_vos.data()[eN_k_nSpace+1]);

                double mu_fr_tmp=0.0;
		updateFrictionalPressure(vos,
                                         grad_vos,
                                         mom_u_source,
                                         mom_v_source,
                                         mom_w_source);
                updateFrictionalStress(LAG_MU_FR,
                                       q_mu_fr_last.data()[eN_k],
                                       mu_fr_tmp,
                                       vos,
                                       eps_rho,
                                       eps_mu,
                                       rho_0,
                                       nu_0,
                                       rho_1,
                                       nu_1,
                                       rho_s,
                                       useVF,
                                       vf.data()[eN_k],
                                       phi.data()[eN_k],
                                       grad_u,
                                       grad_v,
                                       grad_w,
                                       mom_uu_diff_ten,
                                       mom_uv_diff_ten,
                                       mom_uw_diff_ten,
                                       mom_vv_diff_ten,
                                       mom_vu_diff_ten,
                                       mom_vw_diff_ten,
                                       mom_ww_diff_ten,
                                       mom_wu_diff_ten,
                                       mom_wv_diff_ten);
                //
                //moving mesh
                //
                mom_u_adv[0] -= MOVING_DOMAIN*dmom_u_acc_u*mom_u_acc*xt;
                mom_u_adv[1] -= MOVING_DOMAIN*dmom_u_acc_u*mom_u_acc*yt;
                /* mom_u_adv[2] -= MOVING_DOMAIN*dmom_u_acc_u*mom_u_acc*zt; */
                dmom_u_adv_u[0] -= MOVING_DOMAIN*dmom_u_acc_u*xt;
                dmom_u_adv_u[1] -= MOVING_DOMAIN*dmom_u_acc_u*yt;
                /* dmom_u_adv_u[2] -= MOVING_DOMAIN*dmom_u_acc_u*zt; */

                mom_v_adv[0] -= MOVING_DOMAIN*dmom_v_acc_v*mom_v_acc*xt; 
                mom_v_adv[1] -= MOVING_DOMAIN*dmom_v_acc_v*mom_v_acc*yt;
                /* mom_v_adv[2] -= MOVING_DOMAIN*dmom_v_acc_v*mom_v_acc*zt; */
                dmom_v_adv_v[0] -= MOVING_DOMAIN*dmom_v_acc_v*xt;
                dmom_v_adv_v[1] -= MOVING_DOMAIN*dmom_v_acc_v*yt;
                /* dmom_v_adv_v[2] -= MOVING_DOMAIN*dmom_v_acc_v*zt; */

                /* mom_w_adv[0] -= MOVING_DOMAIN*dmom_w_acc_w*mom_w_acc*xt; */
                /* mom_w_adv[1] -= MOVING_DOMAIN*dmom_w_acc_w*mom_w_acc*yt; */
                /* mom_w_adv[2] -= MOVING_DOMAIN*dmom_w_acc_w*mom_w_acc*zt; */
                /* dmom_w_adv_w[0] -= MOVING_DOMAIN*dmom_w_acc_w*xt; */
                /* dmom_w_adv_w[1] -= MOVING_DOMAIN*dmom_w_acc_w*yt; */
                /* dmom_w_adv_w[2] -= MOVING_DOMAIN*dmom_w_acc_w*zt; */
                //
                //calculate time derivatives
                //
                ck.bdf(alphaBDF,
                       q_mom_u_acc_beta_bdf.data()[eN_k]*q_dV_last.data()[eN_k]/dV,
                       mom_u_acc,
                       dmom_u_acc_u,
                       mom_u_acc_t,
                       dmom_u_acc_u_t);
                ck.bdf(alphaBDF,
                       q_mom_v_acc_beta_bdf.data()[eN_k]*q_dV_last.data()[eN_k]/dV,
                       mom_v_acc,
                       dmom_v_acc_v,
                       mom_v_acc_t,
                       dmom_v_acc_v_t);
                /* ck.bdf(alphaBDF, */
                /*           q_mom_w_acc_beta_bdf.data()[eN_k]*q_dV_last.data()[eN_k]/dV, */
                /*           mom_w_acc, */
                /*           dmom_w_acc_w, */
                /*           mom_w_acc_t, */
                /*           dmom_w_acc_w_t); */
                //
                //calculate subgrid error contribution to the Jacobian (strong residual, adjoint, jacobian of strong residual)
		//
		
                mom_u_acc_t *= dmom_u_acc_u;
                mom_v_acc_t *= dmom_v_acc_v; 
		
                //
                dmom_adv_sge[0] = dmom_u_acc_u*(q_velocity_sge.data()[eN_k_nSpace+0] - MOVING_DOMAIN*xt);
                dmom_adv_sge[1] = dmom_u_acc_u*(q_velocity_sge.data()[eN_k_nSpace+1] - MOVING_DOMAIN*yt);
                /* dmom_adv_sge[2] = dmom_u_acc_u*(q_velocity_sge.data()[eN_k_nSpace+2] - MOVING_DOMAIN*zt); */
                //
                //calculate strong residual
                //
                pdeResidual_p =
		  ck.Mass_strong(q_dvos_dt.data()[eN_k]) +
                  ck.Advection_strong(dmass_adv_u,grad_u) +
                  ck.Advection_strong(dmass_adv_v,grad_v) +
                  /* ck.Advection_strong(dmass_adv_w,grad_w) + */
                  DM2*MOVING_DOMAIN*ck.Reaction_strong(alphaBDF*(dV-q_dV_last.data()[eN_k])/dV - div_mesh_velocity) +
                  //VRANS
                  ck.Reaction_strong(mass_source);
                //
              
                pdeResidual_u = 
                  ck.Mass_strong(mom_u_acc_t) +
                  ck.Advection_strong(dmom_adv_sge,grad_u) +
                  ck.Hamiltonian_strong(dmom_u_ham_grad_p,grad_p) +
                  ck.Reaction_strong(mom_u_source) -
                  ck.Reaction_strong(u*div_mesh_velocity);
          
                pdeResidual_v = 
                  ck.Mass_strong(mom_v_acc_t) +
                  ck.Advection_strong(dmom_adv_sge,grad_v) +
                  ck.Hamiltonian_strong(dmom_v_ham_grad_p,grad_p) + 
                  ck.Reaction_strong(mom_v_source)  - 
                  ck.Reaction_strong(v*div_mesh_velocity);
          
                /* pdeResidual_w = 
                   ck.Mass_strong(mom_w_acc_t) +  */
                /*   ck.Advection_strong(dmom_adv_sge,grad_w) + */
                /*   ck.Hamiltonian_strong(dmom_w_ham_grad_p,grad_p) + */
                /*   ck.Reaction_strong(mom_w_source) -  */
                /*   ck.Reaction_strong(w*div_mesh_velocity); */

                //calculate the Jacobian of strong residual
                for (int j=0;j<nDOF_trial_element;j++)
                  {
                    int j_nSpace = j*nSpace;
                    dpdeResidual_p_u[j]=ck.AdvectionJacobian_strong(dmass_adv_u,&vel_grad_trial[j_nSpace]);
                    dpdeResidual_p_v[j]=ck.AdvectionJacobian_strong(dmass_adv_v,&vel_grad_trial[j_nSpace]);
                    /* dpdeResidual_p_w[j]=ck.AdvectionJacobian_strong(dmass_adv_w,&vel_grad_trial[j_nSpace]); */

                    dpdeResidual_u_p[j]=ck.HamiltonianJacobian_strong(dmom_u_ham_grad_p,&p_grad_trial[j_nSpace]);
                    dpdeResidual_u_u[j]=ck.MassJacobian_strong(dmom_u_acc_u_t,vel_trial_ref.data()[k*nDOF_trial_element+j]) +
                      ck.AdvectionJacobian_strong(dmom_adv_sge,&vel_grad_trial[j_nSpace]) -
                      ck.ReactionJacobian_strong(div_mesh_velocity,vel_trial_ref.data()[k*nDOF_trial_element+j]);
              
                    dpdeResidual_v_p[j]=ck.HamiltonianJacobian_strong(dmom_v_ham_grad_p,&p_grad_trial[j_nSpace]);
                    dpdeResidual_v_v[j]=ck.MassJacobian_strong(dmom_v_acc_v_t,vel_trial_ref.data()[k*nDOF_trial_element+j]) +
                      ck.AdvectionJacobian_strong(dmom_adv_sge,&vel_grad_trial[j_nSpace]) -
                      ck.ReactionJacobian_strong(div_mesh_velocity,vel_trial_ref.data()[k*nDOF_trial_element+j]);
              
                    /* dpdeResidual_w_p[j]=ck.HamiltonianJacobian_strong(dmom_w_ham_grad_p,&p_grad_trial[j_nSpace]); */
                    /* dpdeResidual_w_w[j]=ck.MassJacobian_strong(dmom_w_acc_w_t,vel_trial_ref.data()[k*nDOF_trial_element+j]) +  */
                    /*   ck.AdvectionJacobian_strong(dmom_adv_sge,&vel_grad_trial[j_nSpace]) -
                         ck.ReactionJacobian_strong(div_mesh_velocity,vel_trial_ref.data()[k*nDOF_trial_element+j]); */

                    //VRANS account for drag terms, diagonal only here ... decide if need off diagonal terms too
                    dpdeResidual_u_u[j]+= ck.ReactionJacobian_strong(dmom_u_source[0],vel_trial_ref.data()[k*nDOF_trial_element+j]);
                    dpdeResidual_v_v[j]+= ck.ReactionJacobian_strong(dmom_v_source[1],vel_trial_ref.data()[k*nDOF_trial_element+j]);
                    /* dpdeResidual_w_w[j]+= ck.ReactionJacobian_strong(dmom_w_source[2],vel_trial_ref.data()[k*nDOF_trial_element+j]); */
                    //
                  }
                //calculate tau and tau*Res
                //cek debug
                double tmpR=dmom_u_acc_u_t + dmom_u_source[0];
                calculateSubgridError_tau(hFactor,
                                          elementDiameter.data()[eN],
                                          tmpR,//dmom_u_acc_u_t,
                                          dmom_u_acc_u,
                                          dmom_adv_sge,
                                          mom_uu_diff_ten[1],
                                          dmom_u_ham_grad_p[0],
                                          tau_v0,
                                          tau_p0,
                                          q_cfl.data()[eN_k]);
                                        
                calculateSubgridError_tau(Ct_sge,Cd_sge,
                                          G,G_dd_G,tr_G,
                                          tmpR,//dmom_u_acc_u_t,
                                          dmom_adv_sge,
                                          mom_uu_diff_ten[1],
                                          dmom_u_ham_grad_p[0],                                 
                                          tau_v1,
                                          tau_p1,
                                          q_cfl.data()[eN_k]);                                 
                                        
                                        
                tau_v = useMetrics*tau_v1+(1.0-useMetrics)*tau_v0;
                tau_p = PSTAB*(useMetrics*tau_p1+(1.0-useMetrics)*tau_p0);
                calculateSubgridError_tauRes(tau_p,
                                             tau_v,
                                             pdeResidual_p,
                                             pdeResidual_u,
                                             pdeResidual_v,
                                             pdeResidual_w,
                                             subgridError_p,
                                             subgridError_u,
                                             subgridError_v,
                                             subgridError_w);         
              
                calculateSubgridErrorDerivatives_tauRes(tau_p,
                                                        tau_v,
                                                        dpdeResidual_p_u,
                                                        dpdeResidual_p_v,
                                                        dpdeResidual_p_w,
                                                        dpdeResidual_u_p,
                                                        dpdeResidual_u_u,
                                                        dpdeResidual_v_p,
                                                        dpdeResidual_v_v,
                                                        dpdeResidual_w_p,
                                                        dpdeResidual_w_w,
                                                        dsubgridError_p_u,
                                                        dsubgridError_p_v,
                                                        dsubgridError_p_w,
                                                        dsubgridError_u_p,
                                                        dsubgridError_u_u,
                                                        dsubgridError_v_p,
                                                        dsubgridError_v_v,
                                                        dsubgridError_w_p,
                                                        dsubgridError_w_w);
                // velocity used in adjoint (VMS or RBLES, with or without lagging the grid scale velocity)
                dmom_adv_star[0] = dmom_u_acc_u*(q_velocity_sge.data()[eN_k_nSpace+0] - MOVING_DOMAIN*xt + useRBLES*subgridError_u);
                dmom_adv_star[1] = dmom_u_acc_u*(q_velocity_sge.data()[eN_k_nSpace+1] - MOVING_DOMAIN*yt + useRBLES*subgridError_v);
                /* dmom_adv_star[2] = dmom_u_acc_u*(q_velocity_sge.data()[eN_k_nSpace+2] - MOVING_DOMAIN*zt + useRBLES*subgridError_w); */
          
                //calculate the adjoint times the test functions
                for (int i=0;i<nDOF_test_element;i++)
                  {
                    int i_nSpace = i*nSpace;
                    Lstar_u_u[i]=ck.Advection_adjoint(dmom_adv_star,&vel_grad_test_dV[i_nSpace]);
                    Lstar_v_v[i]=ck.Advection_adjoint(dmom_adv_star,&vel_grad_test_dV[i_nSpace]);
                    //VRANS account for drag terms, diagonal only here ... decide if need off diagonal terms too
                    Lstar_u_u[i]+=ck.Reaction_adjoint(dmom_u_source[0],vel_test_dV[i]);
                    Lstar_v_v[i]+=ck.Reaction_adjoint(dmom_v_source[1],vel_test_dV[i]);
                  }

                // Assumes non-lagged subgrid velocity
                dmom_u_adv_u[0] += dmom_u_acc_u*(useRBLES*subgridError_u);         
                dmom_u_adv_u[1] += dmom_u_acc_u*(useRBLES*subgridError_v); 
                /* dmom_u_adv_u[2] += dmom_u_acc_u*(useRBLES*subgridError_w);  */
         
                dmom_v_adv_v[0] += dmom_u_acc_u*(useRBLES*subgridError_u);         
                dmom_v_adv_v[1] += dmom_u_acc_u*(useRBLES*subgridError_v); 
                /* dmom_v_adv_v[2] += dmom_u_acc_u*(useRBLES*subgridError_w);  */
         
                /* dmom_w_adv_w[0] += dmom_u_acc_u*(useRBLES*subgridError_u);               */
                /* dmom_w_adv_w[1] += dmom_u_acc_u*(useRBLES*subgridError_v);  */
                /* dmom_w_adv_w[2] += dmom_u_acc_u*(useRBLES*subgridError_w);  */


                //cek todo add RBLES terms consistent to residual modifications or ignore the partials w.r.t the additional RBLES terms
                for(int i=0;i<nDOF_test_element;i++)
                  {
                    int i_nSpace = i*nSpace;
                    for(int j=0;j<nDOF_trial_element;j++) 
                      {
                        int j_nSpace = j*nSpace;
                        /* elementJacobian_p_p[i][j] += ck.SubgridErrorJacobian(dsubgridError_u_p[j],Lstar_u_p[i]) +  */
                        /*   ck.SubgridErrorJacobian(dsubgridError_v_p[j],Lstar_v_p[i]);// +  */
                        /*   /\* ck.SubgridErrorJacobian(dsubgridError_w_p[j],Lstar_w_p[i]);  *\/ */

                        /* elementJacobian_p_u[i][j] += ck.AdvectionJacobian_weak(dmass_adv_u,vel_trial_ref.data()[k*nDOF_trial_element+j],&p_grad_test_dV[i_nSpace]) +  */
                        /*   ck.SubgridErrorJacobian(dsubgridError_u_u[j],Lstar_u_p[i]);  */
                        /* elementJacobian_p_v[i][j] += ck.AdvectionJacobian_weak(dmass_adv_v,vel_trial_ref.data()[k*nDOF_trial_element+j],&p_grad_test_dV[i_nSpace]) +  */
                        /*   ck.SubgridErrorJacobian(dsubgridError_v_v[j],Lstar_v_p[i]);  */
                        /* elementJacobian_p_w[i][j] += ck.AdvectionJacobian_weak(dmass_adv_w,vel_trial_ref.data()[k*nDOF_trial_element+j],&p_grad_test_dV[i_nSpace]) +  */
                        /*      ck.SubgridErrorJacobian(dsubgridError_w_w[j],Lstar_w_p[i]);  */

                        /* elementJacobian_u_p[i][j] += ck.HamiltonianJacobian_weak(dmom_u_ham_grad_p,&p_grad_trial[j_nSpace],vel_test_dV[i]) +  */
                        /*   ck.SubgridErrorJacobian(dsubgridError_u_p[j],Lstar_u_u[i]);  */
                        elementJacobian_u_u[i][j] += 
                          ck.MassJacobian_weak(dmom_u_acc_u_t,vel_trial_ref.data()[k*nDOF_trial_element+j],vel_test_dV[i]) +
                          ck.HamiltonianJacobian_weak(dmom_u_ham_grad_u,&vel_grad_trial[j_nSpace],vel_test_dV[i]) + 
                          ck.AdvectionJacobian_weak(dmom_u_adv_u,vel_trial_ref.data()[k*nDOF_trial_element+j],&vel_grad_test_dV[i_nSpace]) +
                          ck.SimpleDiffusionJacobian_weak(sdInfo_u_u_rowptr.data(),sdInfo_u_u_colind.data(),mom_uu_diff_ten,&vel_grad_trial[j_nSpace],&vel_grad_test_dV[i_nSpace]) + 
                          //VRANS
                          ck.ReactionJacobian_weak(dmom_u_source[0],vel_trial_ref.data()[k*nDOF_trial_element+j],vel_test_dV[i]) +
                          //
                          //ck.SubgridErrorJacobian(dsubgridError_p_u[j],Lstar_p_u[i]) +
                          ck.SubgridErrorJacobian(dsubgridError_u_u[j],Lstar_u_u[i]) + 
                          ck.NumericalDiffusionJacobian(q_numDiff_u_last.data()[eN_k],&vel_grad_trial[j_nSpace],&vel_grad_test_dV[i_nSpace]); 
                        elementJacobian_u_v[i][j] += 
                          ck.AdvectionJacobian_weak(dmom_u_adv_v,vel_trial_ref.data()[k*nDOF_trial_element+j],&vel_grad_test_dV[i_nSpace]) + 
                          ck.SimpleDiffusionJacobian_weak(sdInfo_u_v_rowptr.data(),sdInfo_u_v_colind.data(),mom_uv_diff_ten,&vel_grad_trial[j_nSpace],&vel_grad_test_dV[i_nSpace]) + 
                          //VRANS
                          ck.ReactionJacobian_weak(dmom_u_source[1],vel_trial_ref.data()[k*nDOF_trial_element+j],vel_test_dV[i]);// +
                        //
                        //ck.SubgridErrorJacobian(dsubgridError_p_v[j],Lstar_p_u[i]);
                        /* elementJacobian_u_w[i][j] += ck.AdvectionJacobian_weak(dmom_u_adv_w,vel_trial_ref.data()[k*nDOF_trial_element+j],&vel_grad_test_dV[i_nSpace]) +  */
                        /*      ck.SimpleDiffusionJacobian_weak(sdInfo_u_w_rowptr.data(),sdInfo_u_w_colind.data(),mom_uw_diff_ten,&vel_grad_trial[j_nSpace],&vel_grad_test_dV[i_nSpace]) +  */
                        /*      //VRANS */
                        /*      ck.ReactionJacobian_weak(dmom_u_source[2],vel_trial_ref.data()[k*nDOF_trial_element+j],vel_test_dV[i]) + */
                        /*      // */
                        /*      ck.SubgridErrorJacobian(dsubgridError_p_w[j],Lstar_p_u[i]);  */

                        /* elementJacobian_v_p[i][j] += ck.HamiltonianJacobian_weak(dmom_v_ham_grad_p,&p_grad_trial[j_nSpace],vel_test_dV[i]) +  */
                        /*   ck.SubgridErrorJacobian(dsubgridError_v_p[j],Lstar_v_v[i]);  */
                        elementJacobian_v_u[i][j] += 
                          ck.AdvectionJacobian_weak(dmom_v_adv_u,vel_trial_ref.data()[k*nDOF_trial_element+j],&vel_grad_test_dV[i_nSpace]) + 
                          ck.SimpleDiffusionJacobian_weak(sdInfo_v_u_rowptr.data(),sdInfo_v_u_colind.data(),mom_vu_diff_ten,&vel_grad_trial[j_nSpace],&vel_grad_test_dV[i_nSpace]) + 
                          //VRANS
                          ck.ReactionJacobian_weak(dmom_v_source[0],vel_trial_ref.data()[k*nDOF_trial_element+j],vel_test_dV[i]);// + 
                        //
                        //ck.SubgridErrorJacobian(dsubgridError_p_u[j],Lstar_p_v[i]);
                        elementJacobian_v_v[i][j] += 
                          ck.MassJacobian_weak(dmom_v_acc_v_t,vel_trial_ref.data()[k*nDOF_trial_element+j],vel_test_dV[i]) + 
                          ck.HamiltonianJacobian_weak(dmom_v_ham_grad_v,&vel_grad_trial[j_nSpace],vel_test_dV[i]) + 
                          ck.AdvectionJacobian_weak(dmom_v_adv_v,vel_trial_ref.data()[k*nDOF_trial_element+j],&vel_grad_test_dV[i_nSpace]) +
                          ck.SimpleDiffusionJacobian_weak(sdInfo_v_v_rowptr.data(),sdInfo_v_v_colind.data(),mom_vv_diff_ten,&vel_grad_trial[j_nSpace],&vel_grad_test_dV[i_nSpace]) + 
                          //VRANS
                          ck.ReactionJacobian_weak(dmom_v_source[1],vel_trial_ref.data()[k*nDOF_trial_element+j],vel_test_dV[i]) +
                          //
                          //ck.SubgridErrorJacobian(dsubgridError_p_v[j],Lstar_p_v[i]) +
                          ck.SubgridErrorJacobian(dsubgridError_v_v[j],Lstar_v_v[i]) + 
                          ck.NumericalDiffusionJacobian(q_numDiff_v_last.data()[eN_k],&vel_grad_trial[j_nSpace],&vel_grad_test_dV[i_nSpace]); 
                        /* elementJacobian_v_w[i][j] += ck.AdvectionJacobian_weak(dmom_v_adv_w,vel_trial_ref.data()[k*nDOF_trial_element+j],&vel_grad_test_dV[i_nSpace]) +   */
                        /*   ck.SimpleDiffusionJacobian_weak(sdInfo_v_w_rowptr.data(),sdInfo_v_w_colind.data(),mom_vw_diff_ten,&vel_grad_trial[j_nSpace],&vel_grad_test_dV[i_nSpace]) +  */
                        /*   //VRANS */
                        /*   ck.ReactionJacobian_weak(dmom_v_source[2],vel_trial_ref.data()[k*nDOF_trial_element+j],vel_test_dV[i]) + */
                        /*   // */
                        /*   ck.SubgridErrorJacobian(dsubgridError_p_w[j],Lstar_p_v[i]); */

                        /* elementJacobian_w_p[i][j] += ck.HamiltonianJacobian_weak(dmom_w_ham_grad_p,&p_grad_trial[j_nSpace],vel_test_dV[i]) +  */
                        /*   ck.SubgridErrorJacobian(dsubgridError_w_p[j],Lstar_w_w[i]);  */
                        /* elementJacobian_w_u[i][j] += ck.AdvectionJacobian_weak(dmom_w_adv_u,vel_trial_ref.data()[k*nDOF_trial_element+j],&vel_grad_test_dV[i_nSpace]) +   */
                        /*   ck.SimpleDiffusionJacobian_weak(sdInfo_w_u_rowptr.data(),sdInfo_w_u_colind.data(),mom_wu_diff_ten,&vel_grad_trial[j_nSpace],&vel_grad_test_dV[i_nSpace]) +  */
                        /*   //VRANS */
                        /*   ck.ReactionJacobian_weak(dmom_w_source[0],vel_trial_ref.data()[k*nDOF_trial_element+j],vel_test_dV[i]) + */
                        /*   // */
                        /*   ck.SubgridErrorJacobian(dsubgridError_p_u[j],Lstar_p_w[i]);  */
                        /* elementJacobian_w_v[i][j] += ck.AdvectionJacobian_weak(dmom_w_adv_v,vel_trial_ref.data()[k*nDOF_trial_element+j],&vel_grad_test_dV[i_nSpace]) +  */
                        /*   ck.SimpleDiffusionJacobian_weak(sdInfo_w_v_rowptr.data(),sdInfo_w_v_colind.data(),mom_wv_diff_ten,&vel_grad_trial[j_nSpace],&vel_grad_test_dV[i_nSpace]) +  */
                        /*   //VRANS */
                        /*   ck.ReactionJacobian_weak(dmom_w_source[1],vel_trial_ref.data()[k*nDOF_trial_element+j],vel_test_dV[i]) + */
                        /*   // */
                        /*   ck.SubgridErrorJacobian(dsubgridError_p_v[j],Lstar_p_w[i]);  */
                        /* elementJacobian_w_w[i][j] += ck.MassJacobian_weak(dmom_w_acc_w_t,vel_trial_ref.data()[k*nDOF_trial_element+j],vel_test_dV[i]) +  */
                        /*   ck.HamiltonianJacobian_weak(dmom_w_ham_grad_w,&vel_grad_trial[j_nSpace],vel_test_dV[i]) +  */
                        /*   ck.AdvectionJacobian_weak(dmom_w_adv_w,vel_trial_ref.data()[k*nDOF_trial_element+j],&vel_grad_test_dV[i_nSpace]) +   */
                        /*   ck.SimpleDiffusionJacobian_weak(sdInfo_w_w_rowptr.data(),sdInfo_w_w_colind.data(),mom_ww_diff_ten,&vel_grad_trial[j_nSpace],&vel_grad_test_dV[i_nSpace]) +  */
                        /*   //VRANS */
                        /*   ck.ReactionJacobian_weak(dmom_w_source[2],vel_trial_ref.data()[k*nDOF_trial_element+j],vel_test_dV[i]) + */
                        /*   // */
                        /*   ck.SubgridErrorJacobian(dsubgridError_p_w[j],Lstar_p_w[i]) +  */
                        /*   ck.SubgridErrorJacobian(dsubgridError_w_w[j],Lstar_w_w[i]) +  */
                        /*   ck.NumericalDiffusionJacobian(q_numDiff_w_last.data()[eN_k],&vel_grad_trial[j_nSpace],&vel_grad_test_dV[i_nSpace]);  */
                      }//j
                  }//i
              }//k
            //
            //load into element Jacobian into global Jacobian
            //
            for (int i=0;i<nDOF_test_element;i++)
              {
                proteus_index_t eN_i = eN*nDOF_test_element+i;
                for (int j=0;j<nDOF_trial_element;j++)
                  {
                    proteus_index_t eN_i_j = eN_i*nDOF_trial_element+j;
                    /* globalJacobian.data()[csrRowIndeces_p_p[eN_i] + csrColumnOffsets_p_p[eN_i_j]] += elementJacobian_p_p[i][j]; */
                    /* globalJacobian.data()[csrRowIndeces_p_u[eN_i] + csrColumnOffsets_p_u[eN_i_j]] += elementJacobian_p_u[i][j]; */
                    /* globalJacobian.data()[csrRowIndeces_p_v[eN_i] + csrColumnOffsets_p_v[eN_i_j]] += elementJacobian_p_v[i][j]; */
                    /* globalJacobian.data()[csrRowIndeces_p_w[eN_i] + csrColumnOffsets_p_w[eN_i_j]] += elementJacobian_p_w[i][j]; */

                    /* globalJacobian.data()[csrRowIndeces_u_p[eN_i] + csrColumnOffsets_u_p[eN_i_j]] += elementJacobian_u_p[i][j]; */
                    globalJacobian.data()[csrRowIndeces_u_u[eN_i] + csrColumnOffsets_u_u[eN_i_j]] += elementJacobian_u_u[i][j];
                    globalJacobian.data()[csrRowIndeces_u_v[eN_i] + csrColumnOffsets_u_v[eN_i_j]] += elementJacobian_u_v[i][j];
                    /* globalJacobian.data()[csrRowIndeces_u_w[eN_i] + csrColumnOffsets_u_w[eN_i_j]] += elementJacobian_u_w[i][j]; */

                    /* globalJacobian.data()[csrRowIndeces_v_p[eN_i] + csrColumnOffsets_v_p[eN_i_j]] += elementJacobian_v_p[i][j]; */
                    globalJacobian.data()[csrRowIndeces_v_u[eN_i] + csrColumnOffsets_v_u[eN_i_j]] += elementJacobian_v_u[i][j];
                    globalJacobian.data()[csrRowIndeces_v_v[eN_i] + csrColumnOffsets_v_v[eN_i_j]] += elementJacobian_v_v[i][j];
                    /* globalJacobian.data()[csrRowIndeces_v_w[eN_i] + csrColumnOffsets_v_w[eN_i_j]] += elementJacobian_v_w[i][j]; */

                    /* globalJacobian.data()[csrRowIndeces_w_p[eN_i] + csrColumnOffsets_w_p[eN_i_j]] += elementJacobian_w_p[i][j]; */
                    /* globalJacobian.data()[csrRowIndeces_w_u[eN_i] + csrColumnOffsets_w_u[eN_i_j]] += elementJacobian_w_u[i][j]; */
                    /* globalJacobian.data()[csrRowIndeces_w_v[eN_i] + csrColumnOffsets_w_v[eN_i_j]] += elementJacobian_w_v[i][j]; */
                    /* globalJacobian.data()[csrRowIndeces_w_w[eN_i] + csrColumnOffsets_w_w[eN_i_j]] += elementJacobian_w_w[i][j]; */
                  }//j
              }//i
          }//elements
        //
        //loop over exterior element boundaries to compute the surface integrals and load them into the global Jacobian
        //
        for (proteus_index_t ebNE = 0; ebNE < nExteriorElementBoundaries_global; ebNE++)
          {
            proteus_index_t ebN = exteriorElementBoundariesArray.data()[ebNE],
              eN  = elementBoundaryElementsArray.data()[ebN*2+0],
              eN_nDOF_trial_element = eN*nDOF_trial_element;
            int ebN_local = elementBoundaryLocalElementBoundariesArray.data()[ebN*2+0];
            double eps_rho,eps_mu;
            for  (int kb=0;kb<nQuadraturePoints_elementBoundary;kb++)
              {
                proteus_index_t ebNE_kb = ebNE*nQuadraturePoints_elementBoundary+kb,
                  ebNE_kb_nSpace = ebNE_kb*nSpace;
                int ebN_local_kb = ebN_local*nQuadraturePoints_elementBoundary+kb,
                  ebN_local_kb_nSpace = ebN_local_kb*nSpace;

                double p_ext=0.0,
                  u_ext=0.0,
                  v_ext=0.0,
                  w_ext=0.0,
                  grad_p_ext[nSpace],
                  grad_u_ext[nSpace],
                  grad_v_ext[nSpace],
                  grad_w_ext[nSpace],
                  mom_u_acc_ext=0.0,
                  dmom_u_acc_u_ext=0.0,
                  mom_v_acc_ext=0.0,
                  dmom_v_acc_v_ext=0.0,
                  mom_w_acc_ext=0.0,
                  dmom_w_acc_w_ext=0.0,
                  mass_adv_ext[nSpace],
                  dmass_adv_u_ext[nSpace],
                  dmass_adv_v_ext[nSpace],
                  dmass_adv_w_ext[nSpace],
                  mom_u_adv_ext[nSpace],
                  dmom_u_adv_u_ext[nSpace],
                  dmom_u_adv_v_ext[nSpace],
                  dmom_u_adv_w_ext[nSpace],
                  mom_v_adv_ext[nSpace],
                  dmom_v_adv_u_ext[nSpace],
                  dmom_v_adv_v_ext[nSpace],
                  dmom_v_adv_w_ext[nSpace],
                  mom_w_adv_ext[nSpace],
                  dmom_w_adv_u_ext[nSpace],
                  dmom_w_adv_v_ext[nSpace],
                  dmom_w_adv_w_ext[nSpace],
                  mom_uu_diff_ten_ext[nSpace],
                  mom_vv_diff_ten_ext[nSpace],
                  mom_ww_diff_ten_ext[nSpace],
                  mom_uv_diff_ten_ext[1],
                  mom_uw_diff_ten_ext[1],
                  mom_vu_diff_ten_ext[1],
                  mom_vw_diff_ten_ext[1],
                  mom_wu_diff_ten_ext[1],
                  mom_wv_diff_ten_ext[1],
                  mom_u_source_ext=0.0,
                  mom_v_source_ext=0.0,
                  mom_w_source_ext=0.0,
                  mom_u_ham_ext=0.0,
                  dmom_u_ham_grad_p_ext[nSpace],
                  dmom_u_ham_grad_u_ext[nSpace],
                  mom_v_ham_ext=0.0,
                  dmom_v_ham_grad_p_ext[nSpace],
                  dmom_v_ham_grad_v_ext[nSpace],
                  mom_w_ham_ext=0.0,
                  dmom_w_ham_grad_p_ext[nSpace],
                  dmom_w_ham_grad_w_ext[nSpace],
                  dmom_u_adv_p_ext[nSpace],
                  dmom_v_adv_p_ext[nSpace],
                  dmom_w_adv_p_ext[nSpace],
                  dflux_mass_u_ext=0.0,
                  dflux_mass_v_ext=0.0,
                  dflux_mass_w_ext=0.0,
                  dflux_mom_u_adv_p_ext=0.0,
                  dflux_mom_u_adv_u_ext=0.0,
                  dflux_mom_u_adv_v_ext=0.0,
                  dflux_mom_u_adv_w_ext=0.0,
                  dflux_mom_v_adv_p_ext=0.0,
                  dflux_mom_v_adv_u_ext=0.0,
                  dflux_mom_v_adv_v_ext=0.0,
                  dflux_mom_v_adv_w_ext=0.0,
                  dflux_mom_w_adv_p_ext=0.0,
                  dflux_mom_w_adv_u_ext=0.0,
                  dflux_mom_w_adv_v_ext=0.0,
                  dflux_mom_w_adv_w_ext=0.0,
                  bc_p_ext=0.0,
                  bc_u_ext=0.0,
                  bc_v_ext=0.0,
                  bc_w_ext=0.0,
                  bc_mom_u_acc_ext=0.0,
                  bc_dmom_u_acc_u_ext=0.0,
                  bc_mom_v_acc_ext=0.0,
                  bc_dmom_v_acc_v_ext=0.0,
                  bc_mom_w_acc_ext=0.0,
                  bc_dmom_w_acc_w_ext=0.0,
                  bc_mass_adv_ext[nSpace],
                  bc_dmass_adv_u_ext[nSpace],
                  bc_dmass_adv_v_ext[nSpace],
                  bc_dmass_adv_w_ext[nSpace],
                  bc_mom_u_adv_ext[nSpace],
                  bc_dmom_u_adv_u_ext[nSpace],
                  bc_dmom_u_adv_v_ext[nSpace],
                  bc_dmom_u_adv_w_ext[nSpace],
                  bc_mom_v_adv_ext[nSpace],
                  bc_dmom_v_adv_u_ext[nSpace],
                  bc_dmom_v_adv_v_ext[nSpace],
                  bc_dmom_v_adv_w_ext[nSpace],
                  bc_mom_w_adv_ext[nSpace],
                  bc_dmom_w_adv_u_ext[nSpace],
                  bc_dmom_w_adv_v_ext[nSpace],
                  bc_dmom_w_adv_w_ext[nSpace],
                  bc_mom_uu_diff_ten_ext[nSpace],
                  bc_mom_vv_diff_ten_ext[nSpace],
                  bc_mom_ww_diff_ten_ext[nSpace],
                  bc_mom_uv_diff_ten_ext[1],
                  bc_mom_uw_diff_ten_ext[1],
                  bc_mom_vu_diff_ten_ext[1],
                  bc_mom_vw_diff_ten_ext[1],
                  bc_mom_wu_diff_ten_ext[1],
                  bc_mom_wv_diff_ten_ext[1],
                  bc_mom_u_source_ext=0.0,
                  bc_mom_v_source_ext=0.0,
                  bc_mom_w_source_ext=0.0,
                  bc_mom_u_ham_ext=0.0,
                  bc_dmom_u_ham_grad_p_ext[nSpace],
                  bc_dmom_u_ham_grad_u_ext[nSpace],
                  bc_mom_v_ham_ext=0.0,
                  bc_dmom_v_ham_grad_p_ext[nSpace],
                  bc_dmom_v_ham_grad_v_ext[nSpace],
                  bc_mom_w_ham_ext=0.0,
                  bc_dmom_w_ham_grad_p_ext[nSpace],
                  bc_dmom_w_ham_grad_w_ext[nSpace],
                  fluxJacobian_u_u[nDOF_trial_element],
                  fluxJacobian_u_v[nDOF_trial_element],
                  fluxJacobian_v_u[nDOF_trial_element],
                  fluxJacobian_v_v[nDOF_trial_element],
                  jac_ext[nSpace*nSpace],
                  jacDet_ext,
                  jacInv_ext[nSpace*nSpace],
                  boundaryJac[nSpace*(nSpace-1)],
                  metricTensor[(nSpace-1)*(nSpace-1)],
                  metricTensorDetSqrt,
                  vel_grad_trial_trace[nDOF_trial_element*nSpace],
                  dS,
                  vel_test_dS[nDOF_test_element],
                  normal[2],
                  x_ext,y_ext,z_ext,xt_ext,yt_ext,zt_ext,integralScaling,
                  vel_grad_test_dS[nDOF_trial_element*nSpace],
                  //VRANS
                  vos_ext,
                  //
                  G[nSpace*nSpace],G_dd_G,tr_G,h_phi,h_penalty,penalty;
                ck.calculateMapping_elementBoundary(eN,
                                                    ebN_local,
                                                    kb,
                                                    ebN_local_kb,
                                                    mesh_dof.data(),
                                                    mesh_l2g.data(),
                                                    mesh_trial_trace_ref.data(),
                                                    mesh_grad_trial_trace_ref.data(),
                                                    boundaryJac_ref.data(),
                                                    jac_ext,
                                                    jacDet_ext,
                                                    jacInv_ext,
                                                    boundaryJac,
                                                    metricTensor,
                                                    metricTensorDetSqrt,
                                                    normal_ref.data(),
                                                    normal,
                                                    x_ext,y_ext,z_ext);
                ck.calculateMappingVelocity_elementBoundary(eN,
                                                            ebN_local,
                                                            kb,
                                                            ebN_local_kb,
                                                            mesh_velocity_dof.data(),
                                                            mesh_l2g.data(),
                                                            mesh_trial_trace_ref.data(),
                                                            xt_ext,yt_ext,zt_ext,
                                                            normal,
                                                            boundaryJac,
                                                            metricTensor,
                                                            integralScaling);
                //dS = ((1.0-MOVING_DOMAIN)*metricTensorDetSqrt + MOVING_DOMAIN*integralScaling)*dS_ref.data()[kb];
                dS = metricTensorDetSqrt*dS_ref.data()[kb];
                ck.calculateG(jacInv_ext,G,G_dd_G,tr_G);
                ck.calculateGScale(G,&ebqe_normal_phi_ext.data()[ebNE_kb_nSpace],h_phi);

                eps_rho = epsFact_rho*(useMetrics*h_phi+(1.0-useMetrics)*elementDiameter.data()[eN]);
                eps_mu  = epsFact_mu *(useMetrics*h_phi+(1.0-useMetrics)*elementDiameter.data()[eN]);

                //compute shape and solution information
                //shape
                /* ck.gradTrialFromRef(&p_grad_trial_trace_ref.data()[ebN_local_kb_nSpace*nDOF_trial_element],jacInv_ext,p_grad_trial_trace); */
                ck.gradTrialFromRef(&vel_grad_trial_trace_ref.data()[ebN_local_kb_nSpace*nDOF_trial_element],jacInv_ext,vel_grad_trial_trace);
                //solution and gradients
                /* ck.valFromDOF(p_dof.data(),&p_l2g.data()[eN_nDOF_trial_element],&p_trial_trace_ref.data()[ebN_local_kb*nDOF_test_element],p_ext); */
                p_ext = ebqe_p.data()[ebNE_kb];
                ck.valFromDOF(u_dof.data(),&vel_l2g.data()[eN_nDOF_trial_element],&vel_trial_trace_ref.data()[ebN_local_kb*nDOF_test_element],u_ext);
                ck.valFromDOF(v_dof.data(),&vel_l2g.data()[eN_nDOF_trial_element],&vel_trial_trace_ref.data()[ebN_local_kb*nDOF_test_element],v_ext);
                /* ck.valFromDOF(w_dof.data(),&vel_l2g.data()[eN_nDOF_trial_element],&vel_trial_trace_ref.data()[ebN_local_kb*nDOF_test_element],w_ext); */
                /* ck.gradFromDOF(p_dof.data(),&p_l2g.data()[eN_nDOF_trial_element],p_grad_trial_trace,grad_p_ext); */
                for (int I=0;I<nSpace;I++)
                  grad_p_ext[I] = ebqe_grad_p.data()[ebNE_kb_nSpace+I];
                ck.gradFromDOF(u_dof.data(),&vel_l2g.data()[eN_nDOF_trial_element],vel_grad_trial_trace,grad_u_ext);
                ck.gradFromDOF(v_dof.data(),&vel_l2g.data()[eN_nDOF_trial_element],vel_grad_trial_trace,grad_v_ext);
                /* ck.gradFromDOF(w_dof.data(),&vel_l2g.data()[eN_nDOF_trial_element],vel_grad_trial_trace,grad_w_ext); */
                //precalculate test function products with integration weights
                for (int j=0;j<nDOF_trial_element;j++)
                  {
                    /* p_test_dS[j] = p_test_trace_ref.data()[ebN_local_kb*nDOF_test_element+j]*dS; */
                    vel_test_dS[j] = vel_test_trace_ref.data()[ebN_local_kb*nDOF_test_element+j]*dS;
                    for (int I=0;I<nSpace;I++)
                      vel_grad_test_dS[j*nSpace+I] = vel_grad_trial_trace[j*nSpace+I]*dS;//cek hack, using trial
                  }
                //
                //load the boundary values
                //
                bc_p_ext = isDOFBoundary_p.data()[ebNE_kb]*ebqe_bc_p_ext.data()[ebNE_kb]+(1-isDOFBoundary_p.data()[ebNE_kb])*p_ext;
                //bc values at moving boundaries are specified relative to boundary motion so we need to add it here
                bc_u_ext = isDOFBoundary_u.data()[ebNE_kb]*(ebqe_bc_u_ext.data()[ebNE_kb] + MOVING_DOMAIN*xt_ext) + (1-isDOFBoundary_u.data()[ebNE_kb])*u_ext;
                bc_v_ext = isDOFBoundary_v.data()[ebNE_kb]*(ebqe_bc_v_ext.data()[ebNE_kb] + MOVING_DOMAIN*yt_ext) + (1-isDOFBoundary_v.data()[ebNE_kb])*v_ext;
                /* bc_w_ext = isDOFBoundary_w.data()[ebNE_kb]*(ebqe_bc_w_ext.data()[ebNE_kb] + MOVING_DOMAIN*zt_ext) + (1-isDOFBoundary_w.data()[ebNE_kb])*w_ext; */
                //VRANS
                vos_ext = ebqe_vos_ext.data()[ebNE_kb];//sed fraction - gco check

                // 
                //calculate the internal and external trace of the pde coefficients 
                // 
                double eddy_viscosity_ext(0.),bc_eddy_viscosity_ext(0.);//not interested in saving boundary eddy viscosity for now
                evaluateCoefficients(eps_rho,
                                     eps_mu,
                                     sigma,
                                     rho_0,
                                     nu_0,
                                     rho_1,
                                     nu_1,
                                     rho_s,
                                     elementDiameter.data()[eN],
                                     smagorinskyConstant,
                                     turbulenceClosureModel,
                                     g.data(),
                                     useVF,
                                     ebqe_vf_ext.data()[ebNE_kb],
                                     ebqe_phi_ext.data()[ebNE_kb],
                                     &ebqe_normal_phi_ext.data()[ebNE_kb_nSpace],
                                     ebqe_kappa_phi_ext.data()[ebNE_kb],
                                     //VRANS
                                     vos_ext,
                                     //
                                     p_ext,
                                     grad_p_ext,
                                     grad_u_ext,
                                     grad_v_ext,
                                     grad_w_ext,
                                     u_ext,
                                     v_ext,
                                     w_ext,
                                     ebqe_velocity_star.data()[ebNE_kb_nSpace+0],
                                     ebqe_velocity_star.data()[ebNE_kb_nSpace+1],
                                     ebqe_velocity_star.data()[ebNE_kb_nSpace+1],//hack,not used
                                     eddy_viscosity_ext,
                                     mom_u_acc_ext,
                                     dmom_u_acc_u_ext,
                                     mom_v_acc_ext,
                                     dmom_v_acc_v_ext,
                                     mom_w_acc_ext,
                                     dmom_w_acc_w_ext,
                                     mass_adv_ext,
                                     dmass_adv_u_ext,
                                     dmass_adv_v_ext,
                                     dmass_adv_w_ext,
                                     mom_u_adv_ext,
                                     dmom_u_adv_u_ext,
                                     dmom_u_adv_v_ext,
                                     dmom_u_adv_w_ext,
                                     mom_v_adv_ext,
                                     dmom_v_adv_u_ext,
                                     dmom_v_adv_v_ext,
                                     dmom_v_adv_w_ext,
                                     mom_w_adv_ext,
                                     dmom_w_adv_u_ext,
                                     dmom_w_adv_v_ext,
                                     dmom_w_adv_w_ext,
                                     mom_uu_diff_ten_ext,
                                     mom_vv_diff_ten_ext,
                                     mom_ww_diff_ten_ext,
                                     mom_uv_diff_ten_ext,
                                     mom_uw_diff_ten_ext,
                                     mom_vu_diff_ten_ext,
                                     mom_vw_diff_ten_ext,
                                     mom_wu_diff_ten_ext,
                                     mom_wv_diff_ten_ext,
                                     mom_u_source_ext,
                                     mom_v_source_ext,
                                     mom_w_source_ext,
                                     mom_u_ham_ext,
                                     dmom_u_ham_grad_p_ext,
                                     dmom_u_ham_grad_u_ext,
                                     mom_v_ham_ext,
                                     dmom_v_ham_grad_p_ext,
                                     dmom_v_ham_grad_v_ext,
                                     mom_w_ham_ext,
                                     dmom_w_ham_grad_p_ext,          
                                     dmom_w_ham_grad_w_ext);          
                evaluateCoefficients(eps_rho,
                                     eps_mu,
                                     sigma,
                                     rho_0,
                                     nu_0,
                                     rho_1,
                                     nu_1,
                                     rho_s,
                                     elementDiameter.data()[eN],
                                     smagorinskyConstant,
                                     turbulenceClosureModel,
                                     g.data(),
                                     useVF,
                                     bc_ebqe_vf_ext.data()[ebNE_kb],
                                     bc_ebqe_phi_ext.data()[ebNE_kb],
                                     &ebqe_normal_phi_ext.data()[ebNE_kb_nSpace],
                                     ebqe_kappa_phi_ext.data()[ebNE_kb],
                                     //VRANS
                                     vos_ext,
                                     //
                                     bc_p_ext,
                                     grad_p_ext,
                                     grad_u_ext,
                                     grad_v_ext,
                                     grad_w_ext,
                                     bc_u_ext,
                                     bc_v_ext,
                                     bc_w_ext,
                                     ebqe_velocity_star.data()[ebNE_kb_nSpace+0],
                                     ebqe_velocity_star.data()[ebNE_kb_nSpace+1],
                                     ebqe_velocity_star.data()[ebNE_kb_nSpace+1],//hack,not used
                                     bc_eddy_viscosity_ext,
                                     bc_mom_u_acc_ext,
                                     bc_dmom_u_acc_u_ext,
                                     bc_mom_v_acc_ext,
                                     bc_dmom_v_acc_v_ext,
                                     bc_mom_w_acc_ext,
                                     bc_dmom_w_acc_w_ext,
                                     bc_mass_adv_ext,
                                     bc_dmass_adv_u_ext,
                                     bc_dmass_adv_v_ext,
                                     bc_dmass_adv_w_ext,
                                     bc_mom_u_adv_ext,
                                     bc_dmom_u_adv_u_ext,
                                     bc_dmom_u_adv_v_ext,
                                     bc_dmom_u_adv_w_ext,
                                     bc_mom_v_adv_ext,
                                     bc_dmom_v_adv_u_ext,
                                     bc_dmom_v_adv_v_ext,
                                     bc_dmom_v_adv_w_ext,
                                     bc_mom_w_adv_ext,
                                     bc_dmom_w_adv_u_ext,
                                     bc_dmom_w_adv_v_ext,
                                     bc_dmom_w_adv_w_ext,
                                     bc_mom_uu_diff_ten_ext,
                                     bc_mom_vv_diff_ten_ext,
                                     bc_mom_ww_diff_ten_ext,
                                     bc_mom_uv_diff_ten_ext,
                                     bc_mom_uw_diff_ten_ext,
                                     bc_mom_vu_diff_ten_ext,
                                     bc_mom_vw_diff_ten_ext,
                                     bc_mom_wu_diff_ten_ext,
                                     bc_mom_wv_diff_ten_ext,
                                     bc_mom_u_source_ext,
                                     bc_mom_v_source_ext,
                                     bc_mom_w_source_ext,
                                     bc_mom_u_ham_ext,
                                     bc_dmom_u_ham_grad_p_ext,
                                     bc_dmom_u_ham_grad_u_ext,
                                     bc_mom_v_ham_ext,
                                     bc_dmom_v_ham_grad_p_ext,
                                     bc_dmom_v_ham_grad_v_ext,
                                     bc_mom_w_ham_ext,
                                     bc_dmom_w_ham_grad_p_ext,          
                                     bc_dmom_w_ham_grad_w_ext);          

                //
                //moving domain
                //
                mom_u_adv_ext[0] -= MOVING_DOMAIN*dmom_u_acc_u_ext*mom_u_acc_ext*xt_ext;
                mom_u_adv_ext[1] -= MOVING_DOMAIN*dmom_u_acc_u_ext*mom_u_acc_ext*yt_ext;
                /* mom_u_adv_ext[2] -= MOVING_DOMAIN*dmom_u_acc_u_ext*mom_u_acc_ext*zt_ext; */
                dmom_u_adv_u_ext[0] -= MOVING_DOMAIN*dmom_u_acc_u_ext*xt_ext;
                dmom_u_adv_u_ext[1] -= MOVING_DOMAIN*dmom_u_acc_u_ext*yt_ext;
                /* dmom_u_adv_u_ext[2] -= MOVING_DOMAIN*dmom_u_acc_u_ext*zt_ext; */
              
                mom_v_adv_ext[0] -= MOVING_DOMAIN*dmom_v_acc_v_ext*mom_v_acc_ext*xt_ext;
                mom_v_adv_ext[1] -= MOVING_DOMAIN*dmom_v_acc_v_ext*mom_v_acc_ext*yt_ext;
                /* mom_v_adv_ext[2] -= MOVING_DOMAIN*dmom_v_acc_v_ext*mom_v_acc_ext*zt_ext; */
                dmom_v_adv_v_ext[0] -= MOVING_DOMAIN*dmom_v_acc_v_ext*xt_ext;
                dmom_v_adv_v_ext[1] -= MOVING_DOMAIN*dmom_v_acc_v_ext*yt_ext;
                /* dmom_v_adv_v_ext[2] -= MOVING_DOMAIN*dmom_v_acc_v_ext*zt_ext; */
              
                /* mom_w_adv_ext[0] -= MOVING_DOMAIN*dmom_w_acc_w_ext*mom_w_acc_ext*xt_ext; */
                /* mom_w_adv_ext[1] -= MOVING_DOMAIN*dmom_w_acc_w_ext*mom_w_acc_ext*yt_ext; */
                /* mom_w_adv_ext[2] -= MOVING_DOMAIN*dmom_w_acc_w_ext*mom_w_acc_ext*zt_ext; */
                /* dmom_w_adv_w_ext[0] -= MOVING_DOMAIN*dmom_w_acc_w_ext*xt_ext; */
                /* dmom_w_adv_w_ext[1] -= MOVING_DOMAIN*dmom_w_acc_w_ext*yt_ext; */
                /* dmom_w_adv_w_ext[2] -= MOVING_DOMAIN*dmom_w_acc_w_ext*zt_ext; */
              
                //moving domain bc's
                // mql. CHECK.
                bc_mom_u_adv_ext[0] -= MOVING_DOMAIN*dmom_u_acc_u_ext*bc_mom_u_acc_ext*xt_ext;
                bc_mom_u_adv_ext[1] -= MOVING_DOMAIN*dmom_u_acc_u_ext*bc_mom_u_acc_ext*yt_ext;
                /* bc_mom_u_adv_ext[2] -= MOVING_DOMAIN*dmom_u_acc_u_ext*bc_mom_u_acc_ext*zt_ext; */
              
                bc_mom_v_adv_ext[0] -= MOVING_DOMAIN*dmom_v_acc_v_ext*bc_mom_v_acc_ext*xt_ext;
                bc_mom_v_adv_ext[1] -= MOVING_DOMAIN*dmom_v_acc_v_ext*bc_mom_v_acc_ext*yt_ext;
                /* bc_mom_v_adv_ext[2] -= MOVING_DOMAIN*dmom_v_acc_v_ext*bc_mom_v_acc_ext*zt_ext; */

                /* bc_mom_w_adv_ext[0] -= MOVING_DOMAIN*dmom_w_acc_w_ext*bc_mom_w_acc_ext*xt_ext; */
                /* bc_mom_w_adv_ext[1] -= MOVING_DOMAIN*dmom_w_acc_w_ext*bc_mom_w_acc_ext*yt_ext; */
                /* bc_mom_w_adv_ext[2] -= MOVING_DOMAIN*dmom_w_acc_w_ext*bc_mom_w_acc_ext*zt_ext; */
                //
                //calculate the numerical fluxes
                //
                exteriorNumericalAdvectiveFluxDerivatives(isDOFBoundary_p.data()[ebNE_kb],
                                                          isDOFBoundary_u.data()[ebNE_kb],
                                                          isDOFBoundary_v.data()[ebNE_kb],
                                                          isDOFBoundary_w.data()[ebNE_kb],
                                                          isAdvectiveFluxBoundary_p.data()[ebNE_kb],
                                                          isAdvectiveFluxBoundary_u.data()[ebNE_kb],
                                                          isAdvectiveFluxBoundary_v.data()[ebNE_kb],
                                                          isAdvectiveFluxBoundary_w.data()[ebNE_kb],
                                                          dmom_u_ham_grad_p_ext[0],//=1/rho
                                                          normal,
                                                          dmom_u_acc_u_ext,
                                                          bc_p_ext,
                                                          bc_u_ext,
                                                          bc_v_ext,
                                                          bc_w_ext,
                                                          bc_mass_adv_ext,
                                                          bc_mom_u_adv_ext,
                                                          bc_mom_v_adv_ext,
                                                          bc_mom_w_adv_ext,
                                                          ebqe_bc_flux_mass_ext.data()[ebNE_kb]+MOVING_DOMAIN*(xt_ext*normal[0]+yt_ext*normal[1]),//bc is relative mass  flux
                                                          ebqe_bc_flux_mom_u_adv_ext.data()[ebNE_kb],
                                                          ebqe_bc_flux_mom_v_adv_ext.data()[ebNE_kb],
                                                          ebqe_bc_flux_mom_w_adv_ext.data()[ebNE_kb],
                                                          p_ext,
                                                          u_ext,
                                                          v_ext,
                                                          w_ext,
                                                          mass_adv_ext,
                                                          mom_u_adv_ext,
                                                          mom_v_adv_ext,
                                                          mom_w_adv_ext,
                                                          dmass_adv_u_ext,
                                                          dmass_adv_v_ext,
                                                          dmass_adv_w_ext,
                                                          dmom_u_adv_p_ext,
                                                          dmom_u_adv_u_ext,
                                                          dmom_u_adv_v_ext,
                                                          dmom_u_adv_w_ext,
                                                          dmom_v_adv_p_ext,
                                                          dmom_v_adv_u_ext,
                                                          dmom_v_adv_v_ext,
                                                          dmom_v_adv_w_ext,
                                                          dmom_w_adv_p_ext,
                                                          dmom_w_adv_u_ext,
                                                          dmom_w_adv_v_ext,
                                                          dmom_w_adv_w_ext,
                                                          dflux_mass_u_ext,
                                                          dflux_mass_v_ext,
                                                          dflux_mass_w_ext,
                                                          dflux_mom_u_adv_p_ext,
                                                          dflux_mom_u_adv_u_ext,
                                                          dflux_mom_u_adv_v_ext,
                                                          dflux_mom_u_adv_w_ext,
                                                          dflux_mom_v_adv_p_ext,
                                                          dflux_mom_v_adv_u_ext,
                                                          dflux_mom_v_adv_v_ext,
                                                          dflux_mom_v_adv_w_ext,
                                                          dflux_mom_w_adv_p_ext,
                                                          dflux_mom_w_adv_u_ext,
                                                          dflux_mom_w_adv_v_ext,
                                                          dflux_mom_w_adv_w_ext,
                                                          &ebqe_velocity_star.data()[ebNE_kb_nSpace]);
                //
                //calculate the flux jacobian
                //
                ck.calculateGScale(G,normal,h_penalty);
                penalty = useMetrics*C_b/h_penalty + (1.0-useMetrics)*ebqe_penalty_ext.data()[ebNE_kb];
                for (int j=0;j<nDOF_trial_element;j++)
                  {
                    int j_nSpace = j*nSpace,ebN_local_kb_j=ebN_local_kb*nDOF_trial_element+j;
                    /* fluxJacobian_p_p[j]=0.0; */
                    /* fluxJacobian_p_u[j]=ck.ExteriorNumericalAdvectiveFluxJacobian(dflux_mass_u_ext,vel_trial_trace_ref.data()[ebN_local_kb_j]); */
                    /* fluxJacobian_p_v[j]=ck.ExteriorNumericalAdvectiveFluxJacobian(dflux_mass_v_ext,vel_trial_trace_ref.data()[ebN_local_kb_j]); */
                    /* fluxJacobian_p_w[j]=ck.ExteriorNumericalAdvectiveFluxJacobian(dflux_mass_w_ext,vel_trial_trace_ref.data()[ebN_local_kb_j]); */

                    /* fluxJacobian_u_p[j]=ck.ExteriorNumericalAdvectiveFluxJacobian(dflux_mom_u_adv_p_ext,p_trial_trace_ref.data()[ebN_local_kb_j]); */
                    fluxJacobian_u_u[j]=ck.ExteriorNumericalAdvectiveFluxJacobian(dflux_mom_u_adv_u_ext,vel_trial_trace_ref.data()[ebN_local_kb_j]) +
                      ExteriorNumericalDiffusiveFluxJacobian(eps_rho,
                                                             ebqe_phi_ext.data()[ebNE_kb],
                                                             sdInfo_u_u_rowptr.data(),
                                                             sdInfo_u_u_colind.data(),
                                                             isDOFBoundary_u.data()[ebNE_kb],
                                                             isDiffusiveFluxBoundary_u.data()[ebNE_kb],
                                                             normal,
                                                             mom_uu_diff_ten_ext,
                                                             vel_trial_trace_ref.data()[ebN_local_kb_j],
                                                             &vel_grad_trial_trace[j_nSpace],
                                                             penalty);//ebqe_penalty_ext.data()[ebNE_kb]);
                    fluxJacobian_u_v[j]=ck.ExteriorNumericalAdvectiveFluxJacobian(dflux_mom_u_adv_v_ext,vel_trial_trace_ref.data()[ebN_local_kb_j]) +
                      ExteriorNumericalDiffusiveFluxJacobian(eps_rho,
                                                             ebqe_phi_ext.data()[ebNE_kb],
                                                             sdInfo_u_v_rowptr.data(),
                                                             sdInfo_u_v_colind.data(),
                                                             isDOFBoundary_v.data()[ebNE_kb],
                                                             isDiffusiveFluxBoundary_v.data()[ebNE_kb],
                                                             normal,
                                                             mom_uv_diff_ten_ext,
                                                             vel_trial_trace_ref.data()[ebN_local_kb_j],
                                                             &vel_grad_trial_trace[j_nSpace],
                                                             penalty);//ebqe_penalty_ext.data()[ebNE_kb]);
                    /* fluxJacobian_u_w[j]=ck.ExteriorNumericalAdvectiveFluxJacobian(dflux_mom_u_adv_w_ext,vel_trial_trace_ref.data()[ebN_local_kb_j]) + */
                    /*   ExteriorNumericalDiffusiveFluxJacobian(eps_rho, */
                    /*                                     ebqe_phi_ext.data()[ebNE_kb], */
                    /*                                     sdInfo_u_w_rowptr.data(), */
                    /*                                     sdInfo_u_w_colind.data(), */
                    /*                                     isDOFBoundary_w.data()[ebNE_kb], */
                    /*                                     isDiffusiveFluxBoundary_u.data()[ebNE_kb], */
                    /*                                     normal, */
                    /*                                     mom_uw_diff_ten_ext, */
                    /*                                     vel_trial_trace_ref.data()[ebN_local_kb_j], */
                    /*                                     &vel_grad_trial_trace[j_nSpace], */
                    /*                                     penalty);//ebqe_penalty_ext.data()[ebNE_kb]); */

                    /* fluxJacobian_v_p[j]=ck.ExteriorNumericalAdvectiveFluxJacobian(dflux_mom_v_adv_p_ext,p_trial_trace_ref.data()[ebN_local_kb_j]); */
                    fluxJacobian_v_u[j]=ck.ExteriorNumericalAdvectiveFluxJacobian(dflux_mom_v_adv_u_ext,vel_trial_trace_ref.data()[ebN_local_kb_j]) +
                      ExteriorNumericalDiffusiveFluxJacobian(eps_rho,
                                                             ebqe_phi_ext.data()[ebNE_kb],
                                                             sdInfo_v_u_rowptr.data(),
                                                             sdInfo_v_u_colind.data(),
                                                             isDOFBoundary_u.data()[ebNE_kb],
                                                             isDiffusiveFluxBoundary_u.data()[ebNE_kb],
                                                             normal,
                                                             mom_vu_diff_ten_ext,
                                                             vel_trial_trace_ref.data()[ebN_local_kb_j],
                                                             &vel_grad_trial_trace[j_nSpace],
                                                             penalty);//ebqe_penalty_ext.data()[ebNE_kb]);
                    fluxJacobian_v_v[j]=ck.ExteriorNumericalAdvectiveFluxJacobian(dflux_mom_v_adv_v_ext,vel_trial_trace_ref.data()[ebN_local_kb_j]) +
                      ExteriorNumericalDiffusiveFluxJacobian(eps_rho,
                                                             ebqe_phi_ext.data()[ebNE_kb],
                                                             sdInfo_v_v_rowptr.data(),
                                                             sdInfo_v_v_colind.data(),
                                                             isDOFBoundary_v.data()[ebNE_kb],
                                                             isDiffusiveFluxBoundary_v.data()[ebNE_kb],
                                                             normal,
                                                             mom_vv_diff_ten_ext,
                                                             vel_trial_trace_ref.data()[ebN_local_kb_j],
                                                             &vel_grad_trial_trace[j_nSpace],
                                                             penalty);//ebqe_penalty_ext.data()[ebNE_kb]);
                    /* fluxJacobian_v_w[j]=ck.ExteriorNumericalAdvectiveFluxJacobian(dflux_mom_v_adv_w_ext,vel_trial_trace_ref.data()[ebN_local_kb_j]) + */
                    /*   ExteriorNumericalDiffusiveFluxJacobian(eps_rho, */
                    /*                                     ebqe_phi_ext.data()[ebNE_kb], */
                    /*                                     sdInfo_v_w_rowptr.data(), */
                    /*                                     sdInfo_v_w_colind.data(), */
                    /*                                     isDOFBoundary_w.data()[ebNE_kb], */
                    /*                                     isDiffusiveFluxBoundary_v.data()[ebNE_kb], */
                    /*                                     normal, */
                    /*                                     mom_vw_diff_ten_ext, */
                    /*                                     vel_trial_trace_ref.data()[ebN_local_kb_j], */
                    /*                                     &vel_grad_trial_trace[j_nSpace], */
                    /*                                     penalty);//ebqe_penalty_ext.data()[ebNE_kb]); */

                    /* fluxJacobian_w_p[j]=ck.ExteriorNumericalAdvectiveFluxJacobian(dflux_mom_w_adv_p_ext,p_trial_trace_ref.data()[ebN_local_kb_j]); */
                    /* fluxJacobian_w_u[j]=ck.ExteriorNumericalAdvectiveFluxJacobian(dflux_mom_w_adv_u_ext,vel_trial_trace_ref.data()[ebN_local_kb_j]) + */
                    /*   ExteriorNumericalDiffusiveFluxJacobian(eps_rho, */
                    /*                                     ebqe_phi_ext.data()[ebNE_kb], */
                    /*                                     sdInfo_w_u_rowptr.data(), */
                    /*                                     sdInfo_w_u_colind.data(), */
                    /*                                     isDOFBoundary_u.data()[ebNE_kb], */
                    /*                                     isDiffusiveFluxBoundary_w.data()[ebNE_kb], */
                    /*                                     normal, */
                    /*                                     mom_wu_diff_ten_ext, */
                    /*                                     vel_trial_trace_ref.data()[ebN_local_kb_j], */
                    /*                                     &vel_grad_trial_trace[j_nSpace], */
                    /*                                     penalty);//ebqe_penalty_ext.data()[ebNE_kb]); */
                    /* fluxJacobian_w_v[j]=ck.ExteriorNumericalAdvectiveFluxJacobian(dflux_mom_w_adv_v_ext,vel_trial_trace_ref.data()[ebN_local_kb_j]) + */
                    /*   ExteriorNumericalDiffusiveFluxJacobian(eps_rho, */
                    /*                                     ebqe_phi_ext.data()[ebNE_kb], */
                    /*                                     sdInfo_w_v_rowptr.data(), */
                    /*                                     sdInfo_w_v_colind.data(), */
                    /*                                     isDOFBoundary_v.data()[ebNE_kb], */
                    /*                                     isDiffusiveFluxBoundary_w.data()[ebNE_kb], */
                    /*                                     normal, */
                    /*                                     mom_wv_diff_ten_ext, */
                    /*                                     vel_trial_trace_ref.data()[ebN_local_kb_j], */
                    /*                                     &vel_grad_trial_trace[j_nSpace], */
                    /*                                     penalty);//ebqe_penalty_ext.data()[ebNE_kb]); */
                    /* fluxJacobian_w_w[j]=ck.ExteriorNumericalAdvectiveFluxJacobian(dflux_mom_w_adv_w_ext,vel_trial_trace_ref.data()[ebN_local_kb_j]) + */
                    /*   ExteriorNumericalDiffusiveFluxJacobian(eps_rho, */
                    /*                                     ebqe_phi_ext.data()[ebNE_kb], */
                    /*                                     sdInfo_w_w_rowptr.data(), */
                    /*                                     sdInfo_w_w_colind.data(), */
                    /*                                     isDOFBoundary_w.data()[ebNE_kb], */
                    /*                                     isDiffusiveFluxBoundary_w.data()[ebNE_kb], */
                    /*                                     normal, */
                    /*                                     mom_ww_diff_ten_ext, */
                    /*                                     vel_trial_trace_ref.data()[ebN_local_kb_j], */
                    /*                                     &vel_grad_trial_trace[j_nSpace], */
                    /*                                     penalty);//ebqe_penalty_ext.data()[ebNE_kb]); */
                  }//j
                //
                //update the global Jacobian from the flux Jacobian
                //
                for (int i=0;i<nDOF_test_element;i++)
                  {
                    proteus_index_t eN_i = eN*nDOF_test_element+i;
                    for (int j=0;j<nDOF_trial_element;j++)
                      {
                        proteus_index_t ebN_i_j = ebN*4*nDOF_test_X_trial_element + i*nDOF_trial_element + j;
                        int ebN_local_kb_j=ebN_local_kb*nDOF_trial_element+j;
                  
                        /* globalJacobian.data()[csrRowIndeces_p_p[eN_i] + csrColumnOffsets_eb_p_p[ebN_i_j]] += fluxJacobian_p_p[j]*p_test_dS[i]; */
                        /* globalJacobian.data()[csrRowIndeces_p_u[eN_i] + csrColumnOffsets_eb_p_u[ebN_i_j]] += fluxJacobian_p_u[j]*p_test_dS[i]; */
                        /* globalJacobian.data()[csrRowIndeces_p_v[eN_i] + csrColumnOffsets_eb_p_v[ebN_i_j]] += fluxJacobian_p_v[j]*p_test_dS[i]; */
                        /* globalJacobian.data()[csrRowIndeces_p_w[eN_i] + csrColumnOffsets_eb_p_w[ebN_i_j]] += fluxJacobian_p_w[j]*p_test_dS[i]; */
                   
                        /* globalJacobian.data()[csrRowIndeces_u_p[eN_i] + csrColumnOffsets_eb_u_p[ebN_i_j]] += fluxJacobian_u_p[j]*vel_test_dS[i]; */
                        globalJacobian.data()[csrRowIndeces_u_u[eN_i] + csrColumnOffsets_eb_u_u[ebN_i_j]] +=
                          fluxJacobian_u_u[j]*vel_test_dS[i]+
                          ck.ExteriorElementBoundaryDiffusionAdjointJacobian(isDOFBoundary_u.data()[ebNE_kb],
                                                                             isDiffusiveFluxBoundary_u.data()[ebNE_kb],
                                                                             eb_adjoint_sigma,
                                                                             vel_trial_trace_ref.data()[ebN_local_kb_j],
                                                                             normal,
                                                                             sdInfo_u_u_rowptr.data(),
                                                                             sdInfo_u_u_colind.data(),
                                                                             mom_uu_diff_ten_ext,
                                                                             &vel_grad_test_dS[i*nSpace]);
                        globalJacobian.data()[csrRowIndeces_u_v[eN_i] + csrColumnOffsets_eb_u_v[ebN_i_j]] +=
                          fluxJacobian_u_v[j]*vel_test_dS[i]+
                          ck.ExteriorElementBoundaryDiffusionAdjointJacobian(isDOFBoundary_v.data()[ebNE_kb],
                                                                             isDiffusiveFluxBoundary_u.data()[ebNE_kb],
                                                                             eb_adjoint_sigma,
                                                                             vel_trial_trace_ref.data()[ebN_local_kb_j],
                                                                             normal,
                                                                             sdInfo_u_v_rowptr.data(),
                                                                             sdInfo_u_v_colind.data(),
                                                                             mom_uv_diff_ten_ext,
                                                                             &vel_grad_test_dS[i*nSpace]);
                        /* globalJacobian.data()[csrRowIndeces_u_w[eN_i] + csrColumnOffsets_eb_u_w[ebN_i_j]] += fluxJacobian_u_w[j]*vel_test_dS[i]+ */
                        /*      ck.ExteriorElementBoundaryDiffusionAdjointJacobian(isDOFBoundary_w.data()[ebNE_kb], */
                        /*                                                         isDiffusiveFluxBoundary_u.data()[ebNE_kb], */
                        /*                                                         eb_adjoint_sigma, */
                        /*                                                         vel_trial_trace_ref.data()[ebN_local_kb_j], */
                        /*                                                         normal, */
                        /*                                                         sdInfo_u_w_rowptr.data(), */
                        /*                                                         sdInfo_u_w_colind.data(), */
                        /*                                                         mom_uw_diff_ten_ext, */
                        /*                                                         &vel_grad_test_dS[i*nSpace]); */
                   
                        /* globalJacobian.data()[csrRowIndeces_v_p[eN_i] + csrColumnOffsets_eb_v_p[ebN_i_j]] += fluxJacobian_v_p[j]*vel_test_dS[i]; */
                        globalJacobian.data()[csrRowIndeces_v_u[eN_i] + csrColumnOffsets_eb_v_u[ebN_i_j]] +=
                          fluxJacobian_v_u[j]*vel_test_dS[i]+
                          ck.ExteriorElementBoundaryDiffusionAdjointJacobian(isDOFBoundary_u.data()[ebNE_kb],
                                                                             isDiffusiveFluxBoundary_v.data()[ebNE_kb],
                                                                             eb_adjoint_sigma,
                                                                             vel_trial_trace_ref.data()[ebN_local_kb_j],
                                                                             normal,
                                                                             sdInfo_v_u_rowptr.data(),
                                                                             sdInfo_v_u_colind.data(),
                                                                             mom_vu_diff_ten_ext,
                                                                             &vel_grad_test_dS[i*nSpace]);
                        globalJacobian.data()[csrRowIndeces_v_v[eN_i] + csrColumnOffsets_eb_v_v[ebN_i_j]] +=
                          fluxJacobian_v_v[j]*vel_test_dS[i]+
                          ck.ExteriorElementBoundaryDiffusionAdjointJacobian(isDOFBoundary_v.data()[ebNE_kb],
                                                                             isDiffusiveFluxBoundary_v.data()[ebNE_kb],
                                                                             eb_adjoint_sigma,
                                                                             vel_trial_trace_ref.data()[ebN_local_kb_j],
                                                                             normal,
                                                                             sdInfo_v_v_rowptr.data(),
                                                                             sdInfo_v_v_colind.data(),
                                                                             mom_vv_diff_ten_ext,
                                                                             &vel_grad_test_dS[i*nSpace]);
                        /* globalJacobian.data()[csrRowIndeces_v_w[eN_i] + csrColumnOffsets_eb_v_w[ebN_i_j]] += fluxJacobian_v_w[j]*vel_test_dS[i]+ */
                        /*      ck.ExteriorElementBoundaryDiffusionAdjointJacobian(isDOFBoundary_w.data()[ebNE_kb], */
                        /*                                                         isDiffusiveFluxBoundary_v.data()[ebNE_kb], */
                        /*                                                         eb_adjoint_sigma, */
                        /*                                                         vel_trial_trace_ref.data()[ebN_local_kb_j], */
                        /*                                                         normal, */
                        /*                                                         sdInfo_v_w_rowptr.data(), */
                        /*                                                         sdInfo_v_w_colind.data(), */
                        /*                                                         mom_vw_diff_ten_ext, */
                        /*                                                         &vel_grad_test_dS[i*nSpace]); */
                        /* globalJacobian.data()[csrRowIndeces_w_p[eN_i] + csrColumnOffsets_eb_w_p[ebN_i_j]] += fluxJacobian_w_p[j]*vel_test_dS[i]; */
                        /* globalJacobian.data()[csrRowIndeces_w_u[eN_i] + csrColumnOffsets_eb_w_u[ebN_i_j]] += fluxJacobian_w_u[j]*vel_test_dS[i]+ */
                        /*      ck.ExteriorElementBoundaryDiffusionAdjointJacobian(isDOFBoundary_u.data()[ebNE_kb], */
                        /*                                                         isDiffusiveFluxBoundary_w.data()[ebNE_kb], */
                        /*                                                         eb_adjoint_sigma, */
                        /*                                                         vel_trial_trace_ref.data()[ebN_local_kb_j], */
                        /*                                                         normal, */
                        /*                                                         sdInfo_w_u_rowptr.data(), */
                        /*                                                         sdInfo_w_u_colind.data(), */
                        /*                                                         mom_wu_diff_ten_ext, */
                        /*                                                         &vel_grad_test_dS[i*nSpace]); */
                        /* globalJacobian.data()[csrRowIndeces_w_v[eN_i] + csrColumnOffsets_eb_w_v[ebN_i_j]] += fluxJacobian_w_v[j]*vel_test_dS[i]+ */
                        /*      ck.ExteriorElementBoundaryDiffusionAdjointJacobian(isDOFBoundary_v.data()[ebNE_kb], */
                        /*                                                         isDiffusiveFluxBoundary_w.data()[ebNE_kb], */
                        /*                                                         eb_adjoint_sigma, */
                        /*                                                         vel_trial_trace_ref.data()[ebN_local_kb_j], */
                        /*                                                         normal, */
                        /*                                                         sdInfo_w_v_rowptr.data(), */
                        /*                                                         sdInfo_w_v_colind.data(), */
                        /*                                                         mom_wv_diff_ten_ext, */
                        /*                                                         &vel_grad_test_dS[i*nSpace]); */
                        /* globalJacobian.data()[csrRowIndeces_w_w[eN_i] + csrColumnOffsets_eb_w_w[ebN_i_j]] += fluxJacobian_w_w[j]*vel_test_dS[i]+ */
                        /*      ck.ExteriorElementBoundaryDiffusionAdjointJacobian(isDOFBoundary_w.data()[ebNE_kb], */
                        /*                                                         isDiffusiveFluxBoundary_w.data()[ebNE_kb], */
                        /*                                                         eb_adjoint_sigma, */
                        /*                                                         vel_trial_trace_ref.data()[ebN_local_kb_j], */
                        /*                                                         normal, */
                        /*                                                         sdInfo_w_w_rowptr.data(), */
                        /*                                                         sdInfo_w_w_colind.data(), */
                        /*                                                         mom_ww_diff_ten_ext, */
                        /*                                                         &vel_grad_test_dS[i*nSpace]); */
                      }//j
                  }//i
              }//kb
          }//ebNE
      }//computeJacobian

      void calculateVelocityAverage(arguments_dict& args)
      {
        proteus_index_t nExteriorElementBoundaries_global = args.scalar<proteus_index_t>("nExteriorElementBoundaries_global");
        proteus::pyarray<proteus_index_t>& exteriorElementBoundariesArray = args.array<proteus_index_t>("exteriorElementBoundariesArray");
        proteus_index_t nInteriorElementBoundaries_global = args.scalar<proteus_index_t>("nInteriorElementBoundaries_global");
        proteus::pyarray<proteus_index_t>& interiorElementBoundariesArray = args.array<proteus_index_t>("interiorElementBoundariesArray");
        proteus::pyarray<proteus_index_t>& elementBoundaryElementsArray = args.array<proteus_index_t>("elementBoundaryElementsArray");
        proteus::pyarray<int>& elementBoundaryLocalElementBoundariesArray = args.array<int>("elementBoundaryLocalElementBoundariesArray");
        proteus::pyarray<double>& mesh_dof = args.array<double>("mesh_dof");
        proteus::pyarray<double>& mesh_velocity_dof = args.array<double>("mesh_velocity_dof");
        double MOVING_DOMAIN = args.scalar<double>("MOVING_DOMAIN");
        proteus::pyarray<proteus_index_t>& mesh_l2g = args.array<proteus_index_t>("mesh_l2g");
        proteus::pyarray<double>& mesh_trial_trace_ref = args.array<double>("mesh_trial_trace_ref");
        proteus::pyarray<double>& mesh_grad_trial_trace_ref = args.array<double>("mesh_grad_trial_trace_ref");
        proteus::pyarray<double>& normal_ref = args.array<double>("normal_ref");
        proteus::pyarray<double>& boundaryJac_ref = args.array<double>("boundaryJac_ref");
        proteus::pyarray<proteus_index_t>& vel_l2g = args.array<proteus_index_t>("vel_l2g");
        proteus::pyarray<double>& u_dof = args.array<double>("u_dof");
        proteus::pyarray<double>& v_dof = args.array<double>("v_dof");
        proteus::pyarray<double>& w_dof = args.array<double>("w_dof");
        proteus::pyarray<double>& vos_dof = args.array<double>("vos_dof");
        proteus::pyarray<double>& vel_trial_trace_ref = args.array<double>("vel_trial_trace_ref");
        proteus::pyarray<double>& ebqe_velocity = args.array<double>("ebqe_velocity");
        proteus::pyarray<double>& velocityAverage = args.array<double>("velocityAverage");
        int permutations[nQuadraturePoints_elementBoundary];
        double xArray_left[nQuadraturePoints_elementBoundary*2],
          xArray_right[nQuadraturePoints_elementBoundary*2];
        for (int i=0;i<nQuadraturePoints_elementBoundary;i++)
          permutations[i]=i;//just to initialize
        for (proteus_index_t ebNE = 0; ebNE < nExteriorElementBoundaries_global; ebNE++)
          {
            proteus_index_t ebN = exteriorElementBoundariesArray.data()[ebNE];
            for  (int kb=0;kb<nQuadraturePoints_elementBoundary;kb++)
              {
                proteus_index_t ebN_kb_nSpace = ebN*nQuadraturePoints_elementBoundary*nSpace+kb*nSpace;
                proteus_index_t ebNE_kb_nSpace = ebNE*nQuadraturePoints_elementBoundary*nSpace+kb*nSpace;
                velocityAverage.data()[ebN_kb_nSpace+0]=ebqe_velocity.data()[ebNE_kb_nSpace+0];
                velocityAverage.data()[ebN_kb_nSpace+1]=ebqe_velocity.data()[ebNE_kb_nSpace+1];
              }//ebNE
          }
        for (proteus_index_t ebNI = 0; ebNI < nInteriorElementBoundaries_global; ebNI++)
          {
            proteus_index_t ebN = interiorElementBoundariesArray.data()[ebNI];
            proteus_index_t left_eN_global   = elementBoundaryElementsArray.data()[ebN*2+0],
  right_eN_global  = elementBoundaryElementsArray.data()[ebN*2+1],
  left_eN_nDOF_trial_element = left_eN_global*nDOF_trial_element,
  right_eN_nDOF_trial_element = right_eN_global*nDOF_trial_element;
  int left_ebN_element  = elementBoundaryLocalElementBoundariesArray.data()[ebN*2+0],
  right_ebN_element = elementBoundaryLocalElementBoundariesArray.data()[ebN*2+1];
            double jac[nSpace*nSpace],
              jacDet,
              jacInv[nSpace*nSpace],
              boundaryJac[nSpace*(nSpace-1)],
              metricTensor[(nSpace-1)*(nSpace-1)],
              metricTensorDetSqrt,
              normal[2],
              x,y,z,
              xt,yt,zt,integralScaling;
          
            for  (int kb=0;kb<nQuadraturePoints_elementBoundary;kb++)
              {
                ck.calculateMapping_elementBoundary(left_eN_global,
                                                    left_ebN_element,
                                                    kb,
                                                    left_ebN_element*nQuadraturePoints_elementBoundary+kb,
                                                    mesh_dof.data(),
                                                    mesh_l2g.data(),
                                                    mesh_trial_trace_ref.data(),
                                                    mesh_grad_trial_trace_ref.data(),
                                                    boundaryJac_ref.data(),
                                                    jac,
                                                    jacDet,
                                                    jacInv,
                                                    boundaryJac,
                                                    metricTensor,
                                                    metricTensorDetSqrt,
                                                    normal_ref.data(),
                                                    normal,
                                                    x,y,z);
                xArray_left[kb*2+0] = x;
                xArray_left[kb*2+1] = y;
                /* xArray_left[kb*3+2] = z; */
                ck.calculateMapping_elementBoundary(right_eN_global,
                                                    right_ebN_element,
                                                    kb,
                                                    right_ebN_element*nQuadraturePoints_elementBoundary+kb,
                                                    mesh_dof.data(),
                                                    mesh_l2g.data(),
                                                    mesh_trial_trace_ref.data(),
                                                    mesh_grad_trial_trace_ref.data(),
                                                    boundaryJac_ref.data(),
                                                    jac,
                                                    jacDet,
                                                    jacInv,
                                                    boundaryJac,
                                                    metricTensor,
                                                    metricTensorDetSqrt,
                                                    normal_ref.data(),
                                                    normal,
                                                    x,y,z);
                ck.calculateMappingVelocity_elementBoundary(left_eN_global,
                                                            left_ebN_element,
                                                            kb,
                                                            left_ebN_element*nQuadraturePoints_elementBoundary+kb,
                                                            mesh_velocity_dof.data(),
                                                            mesh_l2g.data(),
                                                            mesh_trial_trace_ref.data(),
                                                            xt,yt,zt,
                                                            normal,
                                                            boundaryJac,
                                                            metricTensor,
                                                            integralScaling);
                xArray_right[kb*2+0] = x;
                xArray_right[kb*2+1] = y;
                /* xArray_right[kb*3+2] = z; */
              }
            for  (int kb_left=0;kb_left<nQuadraturePoints_elementBoundary;kb_left++)
              {
                double errorNormMin = 1.0;
                for  (int kb_right=0;kb_right<nQuadraturePoints_elementBoundary;kb_right++)
                  {
                    double errorNorm=0.0;
                    for (int I=0;I<nSpace;I++)
                      {
                        errorNorm += fabs(xArray_left[kb_left*2+I]
                                          -
                                          xArray_right[kb_right*2+I]);
                      }
                    if (errorNorm < errorNormMin)
                      {
                        permutations[kb_right] = kb_left;
                        errorNormMin = errorNorm;
                      }
                  }
              }
            for  (int kb=0;kb<nQuadraturePoints_elementBoundary;kb++)
              {
                proteus_index_t ebN_kb_nSpace = ebN*nQuadraturePoints_elementBoundary*nSpace+kb*nSpace;
                double u_left=0.0,
                  v_left=0.0,
                  u_right=0.0,
                  v_right=0.0,
                  vos_left=0.0,
                  vos_right=0.0;
                int left_kb = kb,
                  right_kb = permutations[kb],
                  left_ebN_element_kb_nDOF_test_element=(left_ebN_element*nQuadraturePoints_elementBoundary+left_kb)*nDOF_test_element,
                  right_ebN_element_kb_nDOF_test_element=(right_ebN_element*nQuadraturePoints_elementBoundary+right_kb)*nDOF_test_element;
                //
                //calculate the velocity solution at quadrature points on left and right
                //
                ck.valFromDOF(vos_dof.data(),&vel_l2g.data()[left_eN_nDOF_trial_element],&vel_trial_trace_ref.data()[left_ebN_element_kb_nDOF_test_element],vos_left);
                ck.valFromDOF(u_dof.data(),&vel_l2g.data()[left_eN_nDOF_trial_element],&vel_trial_trace_ref.data()[left_ebN_element_kb_nDOF_test_element],u_left);
                ck.valFromDOF(v_dof.data(),&vel_l2g.data()[left_eN_nDOF_trial_element],&vel_trial_trace_ref.data()[left_ebN_element_kb_nDOF_test_element],v_left);
                /* ck.valFromDOF(w_dof.data(),&vel_l2g.data()[left_eN_nDOF_trial_element],&vel_trial_trace_ref.data()[left_ebN_element_kb_nDOF_test_element],w_left); */
                //
                ck.valFromDOF(vos_dof.data(),&vel_l2g.data()[right_eN_nDOF_trial_element],&vel_trial_trace_ref.data()[right_ebN_element_kb_nDOF_test_element],vos_right);
                ck.valFromDOF(u_dof.data(),&vel_l2g.data()[right_eN_nDOF_trial_element],&vel_trial_trace_ref.data()[right_ebN_element_kb_nDOF_test_element],u_right);
                ck.valFromDOF(v_dof.data(),&vel_l2g.data()[right_eN_nDOF_trial_element],&vel_trial_trace_ref.data()[right_ebN_element_kb_nDOF_test_element],v_right);
                /* ck.valFromDOF(w_dof.data(),&vel_l2g.data()[right_eN_nDOF_trial_element],&vel_trial_trace_ref.data()[right_ebN_element_kb_nDOF_test_element],w_right); */
                //
                velocityAverage.data()[ebN_kb_nSpace+0]=0.5*(u_left + u_right);
                velocityAverage.data()[ebN_kb_nSpace+1]=0.5*(v_left + v_right);
                /* velocityAverage.data()[ebN_kb_nSpace+2]=0.5*(w_left + w_right); */
              }//ebNI
          }
      }
    };//RANS3PSed2D
  

  inline cppRANS3PSed2D_base* newRANS3PSed2D(int nSpaceIn,
                                             int nQuadraturePoints_elementIn,
                                             int nDOF_mesh_trial_elementIn,
                                             int nDOF_trial_elementIn,
                                             int nDOF_test_elementIn,
                                             int nQuadraturePoints_elementBoundaryIn,
                                             int CompKernelFlag,
                                             double aDarcy,
                                             double betaForch,
                                             double grain,
                                             double packFraction,
                                             double packMargin,
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
                                             double mu_fr_limiter)
  {
    cppRANS3PSed2D_base* rvalue = proteus::chooseAndAllocateDiscretization2D
      <cppRANS3PSed2D_base,cppRANS3PSed2D,CompKernel>
      (nSpaceIn,
       nQuadraturePoints_elementIn,
       nDOF_mesh_trial_elementIn,
       nDOF_trial_elementIn,
       nDOF_test_elementIn,
       nQuadraturePoints_elementBoundaryIn,
       CompKernelFlag);
    rvalue->setSedClosure(aDarcy,
                          betaForch,
                          grain,
                          packFraction,
                          packMargin,
                          maxFraction,
                          frFraction,
                          sigmaC,
                          C3e,
                          C4e,
                          eR,
                          fContact,
                          mContact,
                          nContact,
                          angFriction,
                          vos_limiter,
                          mu_fr_limiter);
    return rvalue;
  }
} //proteus

#endif
