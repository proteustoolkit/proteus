#ifndef RANS3PF_H
#define RANS3PF_H
#include <cmath>
#include <valarray>
#include <iostream>
#include <vector>
#include <set>
#include <cstring>
#include "CompKernel.h"
#include "ModelFactory.h"
#include "SedClosure.h"
#include "equivalent_polynomials.h"
#include "ArgumentsDict.h"
#include "pyarray.h"

double sgn3p(double val) {
  return double((0.0 < val) - (val < 0.0));
}

//////////////////////
// ***** TODO ***** //
//////////////////////
// *fix the following w.r.t. not dividing momentum eqn by rho
//      * updateSolidParticleTerms
// *Double check the following w.r.t. not dividing momentum eqn by rho
//      * updateDarcyForchheimerTerms_Ergun
//      * updateTurbulenceClosure
//      * check pdeResidual_p. In particular check the term with q_dvos_dt
//      * double check exteriorNumericalAdvectiveFlux. I multiply from outside porosity*rho
//      * MOVING MESH. Double check.
//      * Turbulence: double check eddy_viscosity within evaluateCoefficients
// ***** END OF TODO *****

#define CELL_BASED_EV_COEFF 1
#define POWER_SMOOTHNESS_INDICATOR 2
#define EPS_FOR_GAMMA_INDICATOR 1E-10
#define C_FOR_GAMMA_INDICATOR 0.25 // increase gamma to make the indicator more agressive (less dissipative)
#define USE_GAMMA_INDICATOR 0
#define ANISOTROPIC_DIFFUSION 0

static const  double DM=0.0;//1-mesh conservation and divergence, 0 - weak div(v) only
static const  double DM2=0.0;//1-point-wise mesh volume strong-residual, 0 - div(v) only
static const  double DM3=1.0;//1-point-wise divergence, 0-point-wise rate of volume change

#define USE_CYLINDER_AS_PARTICLE//just for debug

namespace proteus
{
  template<int nSpace, int nP, int nQ, int nEBQ>
  // The trailing flag is the IFEM gate. It has always been false here -- by
  // default rather than by statement -- and must stay false: with it set, an
  // element whose interface passes through an edge or corner node takes the
  // IFEM branch in Simplex::set_quad, which forces D to 0 and H/ImH to a hard
  // 0/1 instead of the moment fit. That deletes the interface measure this
  // model integrates over. Written out so the choice is visible at the call
  // site and cannot change underneath us if the template default changes.
  using GeneralizedFunctions = equivalent_polynomials::GeneralizedFunctions_mix<nSpace, nP, nP, nQ, nEBQ, false>;

  class cppRANS3PF_base
  {
  public:
    std::valarray<double> TransportMatrix, TransposeTransportMatrix;
    std::valarray<double> uStar_psi, vStar_psi, wStar_psi;
    std::valarray<double> uStar_hi, vStar_hi, wStar_hi, den_hi;
    std::valarray<double> uStar_min_hiHe, vStar_min_hiHe, wStar_min_hiHe;
    std::valarray<double> uStar_gamma, vStar_gamma, wStar_gamma;
    virtual ~cppRANS3PF_base() {}
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
                               double mu_fr_limiter
                                ) {}
    virtual void calculateResidual(arguments_dict& args,
                                   bool useExact
				   )=0;

    virtual void calculateJacobian(arguments_dict& args,
                                   bool useExact)=0;
    virtual void calculateVelocityAverage(arguments_dict& args) = 0;
    virtual void getBoundaryDOFs(arguments_dict& args)=0;
  };

  template<class CompKernelType,
    int nSpace,
    int nQuadraturePoints_element,
    int nDOF_mesh_trial_element,
    int nDOF_trial_element,
    int nDOF_test_element,
    int nQuadraturePoints_elementBoundary>
    class cppRANS3PF : public cppRANS3PF_base
    {
    public:
      std::vector<int> surrogate_boundaries, surrogate_boundary_elements, surrogate_boundary_particle;
      double C_sbm, beta_sbm;
      cppHsuSedStress<3> closure;
      const int nDOF_test_X_trial_element,
        nSpace2=9;
      CompKernelType ck;
      GeneralizedFunctions<nSpace,1,nQuadraturePoints_element,nQuadraturePoints_elementBoundary> gf;
      GeneralizedFunctions<nSpace,1,nQuadraturePoints_element,nQuadraturePoints_elementBoundary> gf_s;
    cppRANS3PF():
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
        ck(),
        C_sbm(2.0),
        beta_sbm(0.0)
          {/*        std::cout<<"Constructing cppRANS3PF<CompKernelTemplate<"
                     <<0<<","
                     <<0<<","
                     <<0<<","
                     <<0<<">,"*/
            /*  <<nSpaceIn<<","
                <<nQuadraturePoints_elementIn<<","
                <<nDOF_mesh_trial_elementIn<<","
                <<nDOF_trial_elementIn<<","
                <<nDOF_test_elementIn<<","
                <<nQuadraturePoints_elementBoundaryIn<<">());"*/
            /*  <<std::endl<<std::flush; */
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
        closure = cppHsuSedStress<3>(aDarcy,
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

      inline double Dot(const double vec1[nSpace],
                        const double vec2[nSpace])
      {
        double dot = 0;
        for (int I=0; I<nSpace; I++)
          dot += vec1[I]*vec2[I];
        return dot;
      }

      inline void calculateTangentialGradient(const double normal[nSpace],
                                              const double vel_grad[nSpace],
                                              double vel_tgrad[nSpace])
      {
        double normal_dot_vel_grad = Dot(normal,vel_grad);
        for (int I=0; I<nSpace; I++)
          vel_tgrad[I] = vel_grad[I] - normal_dot_vel_grad*normal[I];
      }

      inline
        void evaluateCoefficients(const double eps_rho,
                                  const double eps_mu,
                                  const double eps_s,
                                  const double sigma,
                                  const double rho_0,
                                  double nu_0,
                                  const double rho_1,
                                  double nu_1,
                                  const double h_e,
                                  const double smagorinskyConstant,
                                  const int turbulenceClosureModel,
                                  const double g[nSpace],
                                  const double useVF,
                                  const double& vf,
                                  const double& phi,
                                  const double n[nSpace],
                                  const double distance_to_omega_solid,
                                  const double& kappa,
                                  const double porosity,//VRANS specific
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
                                  double dmom_w_ham_grad_w[nSpace],
                                  double& rhoSave,
                                  double& nuSave,
                                  int KILL_PRESSURE_TERM,
                                  int MULTIPLY_EXTERNAL_FORCE_BY_DENSITY,
                                  double forcex,
                                  double forcey,
                                  double forcez,
                                  int MATERIAL_PARAMETERS_AS_FUNCTION,
                                  double density_as_function,
                                  double dynamic_viscosity_as_function,
                                  int USE_SBM,
                                  double x, double y, double z,
                                  int use_ball_as_particle,
                                  double* ball_center,
                                  double* ball_radius,
                                  double* ball_velocity,
                                  double* ball_angular_velocity,
				  // int by parts pressure
				  int INT_BY_PARTS_PRESSURE)
      {
        double rho,nu,mu,H_rho,ImH_rho,d_rho,H_mu,ImH_mu,d_mu,norm_n,nu_t0=0.0,nu_t1=0.0,nu_t;
        H_rho = (1.0-useVF)*gf.H(eps_rho,phi) + useVF*fmin(1.0,fmax(0.0,vf));
        ImH_rho = (1.0-useVF)*gf.ImH(eps_rho,phi) + useVF*(1.0-fmin(1.0,fmax(0.0,vf)));
        d_rho = (1.0-useVF)*gf.D(eps_rho,phi);
        H_mu = (1.0-useVF)*gf.H(eps_mu,phi) + useVF*fmin(1.0,fmax(0.0,vf));
        ImH_mu = (1.0-useVF)*gf.ImH(eps_mu,phi) + useVF*(1.0-fmin(1.0,fmax(0.0,vf)));
        d_mu = (1.0-useVF)*gf.D(eps_mu,phi);

        //calculate eddy viscosity
        switch (turbulenceClosureModel)
          {
            double norm_S;
          case 1:
            {
              norm_S = sqrt(2.0*(grad_u[0]*grad_u[0] + grad_v[1]*grad_v[1] + grad_w[2]*grad_w[2] +
                                 0.5*(grad_u[1]+grad_v[0])*(grad_u[1]+grad_v[0]) +
                                 0.5*(grad_u[2]+grad_w[0])*(grad_u[2]+grad_w[0]) +
                                 0.5*(grad_v[2]+grad_w[1])*(grad_v[2]+grad_w[1])));
              nu_t0 = smagorinskyConstant*smagorinskyConstant*h_e*h_e*norm_S;
              nu_t1 = smagorinskyConstant*smagorinskyConstant*h_e*h_e*norm_S;
            }
          case 2:
            {
              double re_0,cs_0=0.0,re_1,cs_1=0.0;
              norm_S = sqrt(2.0*(grad_u[0]*grad_u[0] + grad_v[1]*grad_v[1] + grad_w[2]*grad_w[2] +
                                 0.5*(grad_u[1]+grad_v[0])*(grad_u[1]+grad_v[0]) +
                                 0.5*(grad_u[2]+grad_w[0])*(grad_u[2]+grad_w[0]) +
                                 0.5*(grad_v[2]+grad_w[1])*(grad_v[2]+grad_w[1])));
              re_0 = h_e*h_e*norm_S/nu_0;
              if (re_0 > 1.0)
                cs_0=0.027*pow(10.0,-3.23*pow(re_0,-0.92));
              nu_t0 = cs_0*h_e*h_e*norm_S;
              re_1 = h_e*h_e*norm_S/nu_1;
              if (re_1 > 1.0)
                cs_1=0.027*pow(10.0,-3.23*pow(re_1,-0.92));
              nu_t1 = cs_1*h_e*h_e*norm_S;
            }
          }

        if (MATERIAL_PARAMETERS_AS_FUNCTION==0)
          {
            rho = rho_0*ImH_rho+rho_1*H_rho;
            nu_t= nu_t0*ImH_mu+nu_t1*H_mu;
            nu  = nu_0*ImH_mu+nu_1*H_mu;
            nu += nu_t;
            mu  = rho_0*nu_0*ImH_mu+rho_1*nu_1*H_mu;
          }
        else // set the material parameters by a function. To check convergence
          {
            rho = density_as_function;
            nu_t= 0;
            mu  = dynamic_viscosity_as_function;
            nu  = mu/rho;
          }

        rhoSave = rho;
        nuSave = nu;

        eddy_viscosity = nu_t*rho; // mql. CHECK. Most changes about not divide by rho are here
        // mass (volume accumulation)
        //..hardwired

        double phi_s_effect = (distance_to_omega_solid > 0.0) ? 1.0 : 1e-10;

        if(USE_SBM>0)
          phi_s_effect = 1.0;
        //u momentum accumulation
        mom_u_acc=u;//trick for non-conservative form
        dmom_u_acc_u=phi_s_effect * rho*porosity;

        //v momentum accumulation
        mom_v_acc=v;
        dmom_v_acc_v=phi_s_effect * rho*porosity;

        //w momentum accumulation
        mom_w_acc=phi_s_effect*w;
        dmom_w_acc_w=phi_s_effect*rho*porosity;

        //mass advective flux
        mass_adv[0]=phi_s_effect * porosity*u;
        mass_adv[1]=phi_s_effect * porosity*v;
        mass_adv[2]=phi_s_effect * porosity*w;

        dmass_adv_u[0]=phi_s_effect * porosity;
        dmass_adv_u[1]=0.0;
        dmass_adv_u[2]=0.0;

        dmass_adv_v[0]=0.0;
        dmass_adv_v[1]=phi_s_effect * porosity;
        dmass_adv_v[2]=0.0;

        dmass_adv_w[0]=0.0;
        dmass_adv_w[1]=0.0;
        dmass_adv_w[2]=phi_s_effect * porosity;

        //advection switched to non-conservative form but could be used for mesh motion...
        //u momentum advective flux
        mom_u_adv[0]=0.0;
        mom_u_adv[1]=0.0;
        mom_u_adv[2]=0.0;

        dmom_u_adv_u[0]=0.0;
        dmom_u_adv_u[1]=0.0;
        dmom_u_adv_u[2]=0.0;

        dmom_u_adv_v[0]=0.0;
        dmom_u_adv_v[1]=0.0;
        dmom_u_adv_v[2]=0.0;

        dmom_u_adv_w[0]=0.0;
        dmom_u_adv_w[1]=0.0;
        dmom_u_adv_w[2]=0.0;

        //v momentum advective_flux
        mom_v_adv[0]=0.0;
        mom_v_adv[1]=0.0;
        mom_v_adv[2]=0.0;

        dmom_v_adv_u[0]=0.0;
        dmom_v_adv_u[1]=0.0;
        dmom_v_adv_u[2]=0.0;

        dmom_v_adv_w[0]=0.0;
        dmom_v_adv_w[1]=0.0;
        dmom_v_adv_w[2]=0.0;

        dmom_v_adv_v[0]=0.0;
        dmom_v_adv_v[1]=0.0;
        dmom_v_adv_v[2]=0.0;

        //w momentum advective_flux
        mom_w_adv[0]=0.0;
        mom_w_adv[1]=0.0;
        mom_w_adv[2]=0.0;

        dmom_w_adv_u[0]=0.0;
        dmom_w_adv_u[1]=0.0;
        dmom_w_adv_u[2]=0.0;

        dmom_w_adv_v[0]=0.0;
        dmom_w_adv_v[1]=0.0;
        dmom_w_adv_v[2]=0.0;

        dmom_w_adv_w[0]=0.0;
        dmom_w_adv_w[1]=0.0;
        dmom_w_adv_w[2]=0.0;

        //u momentum diffusion tensor
        mom_uu_diff_ten[0] = phi_s_effect*porosity*2.0*mu;
        mom_uu_diff_ten[1] = phi_s_effect*porosity*mu;
        mom_uu_diff_ten[2] = phi_s_effect*porosity*mu;

        mom_uv_diff_ten[0]=phi_s_effect * porosity*mu;

        mom_uw_diff_ten[0]=phi_s_effect*porosity*mu;

        //v momentum diffusion tensor
        mom_vv_diff_ten[0] = phi_s_effect*porosity*mu;
        mom_vv_diff_ten[1] = phi_s_effect*porosity*2.0*mu;
        mom_vv_diff_ten[2] = phi_s_effect*porosity*mu;

        mom_vu_diff_ten[0]=phi_s_effect * porosity*mu;

        mom_vw_diff_ten[0]=phi_s_effect*porosity*mu;

        //w momentum diffusion tensor
        mom_ww_diff_ten[0] = phi_s_effect*porosity*mu;
        mom_ww_diff_ten[1] = phi_s_effect*porosity*mu;
        mom_ww_diff_ten[2] = phi_s_effect*porosity*2.0*mu;

        mom_wu_diff_ten[0]=phi_s_effect*porosity*mu;

        mom_wv_diff_ten[0]=phi_s_effect*porosity*mu;

        //momentum sources
        norm_n = sqrt(n[0]*n[0]+n[1]*n[1]+n[2]*n[2]);

        mom_u_source = -phi_s_effect*porosity*rho*g[0];// - d_mu*sigma*kappa*n[0]/(rho*(norm_n+1.0e-8));
        mom_v_source = -phi_s_effect*porosity*rho*g[1];// - d_mu*sigma*kappa*n[1]/(rho*(norm_n+1.0e-8));
        mom_w_source = -phi_s_effect*porosity*rho*g[2];// - d_mu*sigma*kappa*n[2]/(rho*(norm_n+1.0e-8));

        // mql: add general force term
        mom_u_source -= (MULTIPLY_EXTERNAL_FORCE_BY_DENSITY == 1 ? porosity*rho : 1.0)*forcex;
        mom_v_source -= (MULTIPLY_EXTERNAL_FORCE_BY_DENSITY == 1 ? porosity*rho : 1.0)*forcey;
        mom_w_source -= (MULTIPLY_EXTERNAL_FORCE_BY_DENSITY == 1 ? porosity*rho : 1.0)*forcez;

        //u momentum Hamiltonian (pressure)
	double aux_pressure = (KILL_PRESSURE_TERM==1 ? 0. : 1.)*(INT_BY_PARTS_PRESSURE==1 ? 0. : 1.);
        mom_u_ham = phi_s_effect * porosity*grad_p[0]*aux_pressure;
        dmom_u_ham_grad_p[0]=phi_s_effect * porosity*aux_pressure;
        dmom_u_ham_grad_p[1]=0.0;
        dmom_u_ham_grad_p[2]=0.0;

        //v momentum Hamiltonian (pressure)
        mom_v_ham = phi_s_effect * porosity*grad_p[1]*aux_pressure;
        dmom_v_ham_grad_p[0]=0.0;
        dmom_v_ham_grad_p[1]=phi_s_effect * porosity*aux_pressure;
        dmom_v_ham_grad_p[2]=0.0;

        //w momentum Hamiltonian (pressure)
        mom_w_ham = phi_s_effect*porosity*grad_p[2]*aux_pressure;
        dmom_w_ham_grad_p[0]=0.0;
        dmom_w_ham_grad_p[1]=0.0;
        dmom_w_ham_grad_p[2]=phi_s_effect*porosity*aux_pressure;

        //u momentum Hamiltonian (advection)
        mom_u_ham += phi_s_effect*rho*porosity*(uStar*grad_u[0]+vStar*grad_u[1]+wStar*grad_u[2]);
        dmom_u_ham_grad_u[0]=phi_s_effect*rho*porosity*uStar;
        dmom_u_ham_grad_u[1]=phi_s_effect*rho*porosity*vStar;
        dmom_u_ham_grad_u[2]=phi_s_effect*rho*porosity*wStar;

        //v momentum Hamiltonian (advection)
        mom_v_ham += phi_s_effect*rho*porosity*(uStar*grad_v[0]+vStar*grad_v[1]+wStar*grad_v[2]);
        dmom_v_ham_grad_v[0]=phi_s_effect*rho*porosity*uStar;
        dmom_v_ham_grad_v[1]=phi_s_effect*rho*porosity*vStar;
        dmom_v_ham_grad_v[2]=phi_s_effect*rho*porosity*wStar;

        //w momentum Hamiltonian (advection)
        mom_w_ham += phi_s_effect*rho*porosity*(uStar*grad_w[0]+vStar*grad_w[1]+wStar*grad_w[2]);
        dmom_w_ham_grad_w[0]=phi_s_effect*rho*porosity*uStar;
        dmom_w_ham_grad_w[1]=phi_s_effect*rho*porosity*vStar;
        dmom_w_ham_grad_w[2]=phi_s_effect*rho*porosity*wStar;
      }

      //VRANS specific
      inline
        void updateDarcyForchheimerTerms_Ergun(/* const double linearDragFactor, */
                                               /* const double nonlinearDragFactor, */
                                               /* const double porosity, */
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
                                               const double phi_s,
                                               const double u_s,
                                               const double v_s,
                                               const double w_s,
                                               const double uStar_s,
                                               const double vStar_s,
                                               const double wStar_s,
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
        double rho, mu,nu,H_mu,ImH_mu,uc,duc_du,duc_dv,duc_dw,viscosity,H_s;
        H_mu = (1.0-useVF)*gf.H(eps_mu,phi)+useVF*fmin(1.0,fmax(0.0,vf));
        ImH_mu = (1.0-useVF)*gf.ImH(eps_mu,phi) + useVF*(1.0-fmin(1.0,fmax(0.0,vf)));
        nu  = nu_0*ImH_mu+nu_1*H_mu;
        rho  = rho_0*ImH_mu+rho_1*H_mu;
        mu  = rho_0*nu_0*ImH_mu+rho_1*nu_1*H_mu;
        viscosity = nu;
        uc = sqrt(u*u+v*v*+w*w);
        duc_du = u/(uc+1.0e-12);
        duc_dv = v/(uc+1.0e-12);
        duc_dw = w/(uc+1.0e-12);
        double fluid_velocity[3]={uStar,vStar,wStar}, solid_velocity[3]={uStar_s,vStar_s,wStar_s};
        double new_beta =   closure.betaCoeff(1.0-phi_s,
                                          rho,
                                         fluid_velocity,
                                           solid_velocity,
                                            viscosity);
        //new_beta/=rho;
        mom_u_source +=  (1.0 - phi_s)*new_beta*( (u - uStar_s) - nu_t*gradC_x/closure.sigmaC_ );
        mom_v_source +=  (1.0 - phi_s)*new_beta*( (v - vStar_s)- nu_t*gradC_y/closure.sigmaC_ );
        mom_w_source +=  (1.0 - phi_s)*new_beta*( (w - wStar_s)- nu_t*gradC_z/closure.sigmaC_ );;

        dmom_u_source[0] = (1.0 - phi_s)*new_beta;
        dmom_u_source[1] = 0.0;
        dmom_u_source[2] = 0.0;

        dmom_v_source[0] = 0.0;
        dmom_v_source[1] = (1.0 - phi_s)*new_beta;
        dmom_v_source[2] = 0.0;

        dmom_w_source[0] = 0.0;
        dmom_w_source[1] = 0.0;
        dmom_w_source[2] = (1.0 - phi_s)*new_beta;
      }

      inline void updateSolidParticleTerms(bool element_owned,
                                           const double particle_nitsche,
                                           const double dV,
                                           const int nParticles,
                                           const int sd_offset,
                                           double *particle_signed_distances,
                                           double *particle_signed_distance_normals,
                                           double *particle_velocities,
                                           double *particle_centroids,
                                           int use_ball_as_particle,
                                           double* ball_center,
                                           double* ball_radius,
                                           double* ball_velocity,
                                           double* ball_angular_velocity,
                                           const double porosity, //VRANS specific
                                           const double penalty,
                                           const double alpha,
                                           const double beta,
                                           const double eps_rho,
                                           const double eps_mu,
                                           const double rho_0,
                                           const double nu_0,
                                           const double rho_1,
                                           const double nu_1,
                                           const double useVF,
                                           const double vf,
                                           const double phi,
                                           const double x,
                                           const double y,
                                           const double z,
                                           const double p,
                                           const double u,
                                           const double v,
                                           const double w,
                                           const double uStar,
                                           const double vStar,
                                           const double wStar,
                                           const double eps_s,
                                           const double grad_u[nSpace],
                                           const double grad_v[nSpace],
                                           const double grad_w[nSpace],
                                           double &mom_u_source,
                                           double &mom_v_source,
                                           double &mom_w_source,
                                           double dmom_u_source[nSpace],
                                           double dmom_v_source[nSpace],
                                           double dmom_w_source[nSpace],
                                           double mom_u_adv[nSpace],
                                           double mom_v_adv[nSpace],
                                           double mom_w_adv[nSpace],
                                           double dmom_u_adv_u[nSpace],
                                           double dmom_v_adv_v[nSpace],
                                           double dmom_w_adv_w[nSpace],
                                           double &mom_u_ham,
                                           double dmom_u_ham_grad_u[nSpace],
                                           double &mom_v_ham,
                                           double dmom_v_ham_grad_v[nSpace],
                                           double &mom_w_ham,
                                           double dmom_w_ham_grad_w[nSpace],
                                           double *particle_netForces,
                                           double *particle_netMoments,
                                           double *particle_surfaceArea)
      {
        double C,rho, mu,nu,H_mu,ImH_mu,uc,duc_du,duc_dv,duc_dw,H_s,D_s,phi_s,u_s,v_s,w_s,force_x,force_y,force_z,r_x,r_y,r_z;
        double phi_s_normal[3];
        double fluid_outward_normal[3];
        double vel[3];
        double center[3];
        H_mu = (1.0-useVF)*gf.H(eps_mu,phi)+useVF*fmin(1.0,fmax(0.0,vf));
        ImH_mu = (1.0-useVF)*gf.ImH(eps_mu,phi) + useVF*(1.0-fmin(1.0,fmax(0.0,vf)));
        nu  = nu_0*ImH_mu+nu_1*H_mu;
        rho  = rho_0*ImH_mu+rho_1*H_mu;
        mu  = rho_0*nu_0*ImH_mu+rho_1*nu_1*H_mu;
        C=0.0;
        for (int i=0;i<nParticles;i++)
          {
            double* vel_pointer= &particle_velocities[i*sd_offset*nSpace];
            if(use_ball_as_particle==1)
            {
                get_distance_to_ith_ball(nParticles,ball_center,ball_radius,i,x,y,z,phi_s);
                get_normal_to_ith_ball(nParticles,ball_center,ball_radius,i,x,y,z,phi_s_normal[0],phi_s_normal[1],phi_s_normal[2]);
                get_velocity_to_ith_ball(nParticles,ball_center,ball_radius,
                                         ball_velocity,ball_angular_velocity,
                                         i,x,y,z,
                                         vel[0],vel[1],vel[2]);
                center[0] = ball_center[3*i + 0];
                center[1] = ball_center[3*i + 1];
                center[2] = ball_center[3*i + 2];
                u_s = vel[0];
                v_s = vel[1];
                w_s = vel[2];
            }
            else
            {
                phi_s = particle_signed_distances[i * sd_offset];
                phi_s_normal[0] = particle_signed_distance_normals[i * sd_offset * nSpace + 0];
                phi_s_normal[1] = particle_signed_distance_normals[i * sd_offset * nSpace + 1];
                phi_s_normal[2] = particle_signed_distance_normals[i * sd_offset * nSpace + 2];
                vel[0] = particle_velocities[i * sd_offset * nSpace + 0];
                vel[1] = particle_velocities[i * sd_offset * nSpace + 1];
                vel[2] = particle_velocities[i * sd_offset * nSpace + 2];
            	  u_s = vel_pointer[0];
            	  v_s = vel_pointer[1];
            	  w_s = vel_pointer[2];
                center[0] = particle_centroids[3*i + 0];
                center[1] = particle_centroids[3*i + 1];
                center[2] = particle_centroids[3*i + 2];
            }
            fluid_outward_normal[0] = -phi_s_normal[0];
            fluid_outward_normal[1] = -phi_s_normal[1];
            fluid_outward_normal[2] = -phi_s_normal[2];

            H_s = gf_s.H(eps_s, phi_s);
            D_s = gf_s.D(eps_s, phi_s);
            double rel_vel_norm=sqrt((uStar - u_s)*(uStar - u_s)+
                                     (vStar - v_s)*(vStar - v_s)+
                                     (wStar - w_s)*(wStar - w_s));
            double C_surf = (phi_s > 0.0) ? 0.0 : nu*penalty;
            double C_vol = (phi_s > 0.0) ? 0.0 : (alpha + beta*rel_vel_norm);
            C = (D_s*C_surf + gf_s.ImH(eps_s, phi_s) * C_vol);
            force_x = dV*D_s*(p*fluid_outward_normal[0] - porosity*mu*(fluid_outward_normal[0]*grad_u[0] +
                                                                       fluid_outward_normal[1]*grad_u[1] +
                                                                       fluid_outward_normal[2]*grad_u[2]) +
                              C_surf*rel_vel_norm*(u-u_s)*rho) + dV*gf_s.ImH(eps_s, phi_s)*C_vol*(u-u_s)*rho;
            force_y = dV*D_s*(p*fluid_outward_normal[1] - porosity*mu*(fluid_outward_normal[0]*grad_v[0] +
                                                                       fluid_outward_normal[1]*grad_v[1] +
                                                                       fluid_outward_normal[2]*grad_v[2]) +
                              C_surf*rel_vel_norm*(v-v_s)*rho) + dV*gf_s.ImH(eps_s, phi_s)*C_vol*(v-v_s)*rho;
            force_z = dV*D_s*(p*fluid_outward_normal[2] - porosity*mu*(fluid_outward_normal[0]*grad_w[0] +
                                                                       fluid_outward_normal[1]*grad_w[1] +
                                                                       fluid_outward_normal[2]*grad_w[2]) +
                              C_surf*rel_vel_norm*(v-v_s)*rho) + dV*gf_s.ImH(eps_s, phi_s)*C_vol*(v-v_s)*rho;
            //always 3D for particle centroids
            r_x = x - center[0];
            r_y = y - center[1];
            r_z = z - center[2];

            if (element_owned)
              {
                //always 3D for particle forces
                particle_netForces[i*3+0] += force_x;
                particle_netForces[i*3+1] += force_y;
                particle_netForces[i*3+2] += force_z;
                particle_netMoments[i*3+0] += (r_y*force_z - r_z*force_y);
                particle_netMoments[i*3+1] += (r_z*force_x - r_x*force_z);
                particle_netMoments[i*3+2] += (r_x*force_y - r_y*force_x);
              }

            // These should be done inside to make sure the correct velocity of different particles are used
            mom_u_source += C*(u - u_s);
            mom_v_source += C*(v - v_s);
            mom_w_source += C*(w - w_s);

            dmom_u_source[0] += C;
            dmom_v_source[1] += C;
            dmom_w_source[2] += C;

            //Nitsche terms
            mom_u_ham    -= D_s*porosity*nu*(fluid_outward_normal[0]*grad_u[0] + fluid_outward_normal[1]*grad_u[1] + fluid_outward_normal[2]*grad_u[2]);
            dmom_u_ham_grad_u[0] -= D_s*porosity*nu*fluid_outward_normal[0];
            dmom_u_ham_grad_u[1] -= D_s*porosity*nu*fluid_outward_normal[1];
            dmom_u_ham_grad_u[2] -= D_s*porosity*nu*fluid_outward_normal[2];

            mom_v_ham    -= D_s*porosity*nu*(fluid_outward_normal[0]*grad_v[0] + fluid_outward_normal[1]*grad_v[1]+ fluid_outward_normal[2]*grad_v[2]);
            dmom_v_ham_grad_v[0] -= D_s*porosity*nu*fluid_outward_normal[0];
            dmom_v_ham_grad_v[1] -= D_s*porosity*nu*fluid_outward_normal[1];
            dmom_v_ham_grad_v[2] -= D_s*porosity*nu*fluid_outward_normal[2];

            mom_u_adv[0] += D_s*porosity*nu*fluid_outward_normal[0]*(u - u_s);
            mom_u_adv[1] += D_s*porosity*nu*fluid_outward_normal[1]*(u - u_s);
            mom_u_adv[2] += D_s*porosity*nu*fluid_outward_normal[2]*(u - u_s);
            dmom_u_adv_u[0] += D_s*porosity*nu*fluid_outward_normal[0];
            dmom_u_adv_u[1] += D_s*porosity*nu*fluid_outward_normal[1];
            dmom_u_adv_u[2] += D_s*porosity*nu*fluid_outward_normal[2];

            mom_v_adv[0] += D_s*porosity*nu*fluid_outward_normal[0]*(v - v_s);
            mom_v_adv[1] += D_s*porosity*nu*fluid_outward_normal[1]*(v - v_s);
            mom_v_adv[2] += D_s*porosity*nu*fluid_outward_normal[2]*(v - v_s);
            dmom_v_adv_v[0] += D_s*porosity*nu*fluid_outward_normal[0];
            dmom_v_adv_v[1] += D_s*porosity*nu*fluid_outward_normal[1];
            dmom_v_adv_v[2] += D_s*porosity*nu*fluid_outward_normal[2];

            mom_w_adv[0] += D_s*porosity*nu*fluid_outward_normal[0]*(w - w_s);
            mom_w_adv[1] += D_s*porosity*nu*fluid_outward_normal[1]*(w - w_s);
            mom_w_adv[2] += D_s*porosity*nu*fluid_outward_normal[2]*(w - w_s);
            dmom_w_adv_w[0] += D_s*porosity*nu*fluid_outward_normal[0];
            dmom_w_adv_w[1] += D_s*porosity*nu*fluid_outward_normal[1];
            dmom_w_adv_w[2] += D_s*porosity*nu*fluid_outward_normal[2];
          }
      }

      inline
        void calculateCFL(const double& hFactor,
                          const double& elementDiameter,
                          const double& dm,
                          const double df[nSpace],
                          double& cfl)
      {
        double h,density,nrm_df=0.0;
        h = hFactor*elementDiameter;
        density = dm;
        for(int I=0;I<nSpace;I++)
          nrm_df+=df[I]*df[I];
        nrm_df = sqrt(nrm_df);
        if (density > 1.0e-8)
          cfl = nrm_df/(h*density);//this is really cfl/dt, but that's what we want to know, the step controller expect this
        else
          cfl = nrm_df/h;
        //cfl = nrm_df/(h*density);//this is really cfl/dt, but that's what we want to know, the step controller expect this
      }

      inline void updateTurbulenceClosure(const int turbulenceClosureModel,
                                          const double eps_rho,
                                          const double eps_mu,
                                          const double rho_0,
                                          const double nu_0,
                                          const double rho_1,
                                          const double nu_1,
                                          const double useVF,
                                          const double vf,
                                          const double phi,
                                          const double porosity,
                                          const double eddy_visc_coef_0,
                                          const double turb_var_0,                //k for k-eps or k-omega
                                          const double turb_var_1,                //epsilon for k-epsilon, omega for k-omega
                                          const double turb_grad_0[nSpace], //grad k for k-eps,k-omega
                                          double &eddy_viscosity,
                                          double mom_uu_diff_ten[nSpace],
                                          double mom_vv_diff_ten[nSpace],
                                          double mom_ww_diff_ten[nSpace],
                                          double mom_uv_diff_ten[1],
                                          double mom_uw_diff_ten[1],
                                          double mom_vu_diff_ten[1],
                                          double mom_vw_diff_ten[1],
                                          double mom_wu_diff_ten[1],
                                          double mom_wv_diff_ten[1],
                                          double &mom_u_source,
                                          double &mom_v_source,
                                          double &mom_w_source)
      {
        /****
             eddy_visc_coef
             <= 2  LES (do nothing)
             == 3  k-epsilon

        */
        assert (turbulenceClosureModel >=3);
        double rho,nu,H_mu,ImH_mu,nu_t=0.0,nu_t_keps =0.0, nu_t_komega=0.0;
        double isKEpsilon = 1.0;
        if (turbulenceClosureModel == 4)
          isKEpsilon = 0.0;
        H_mu = (1.0-useVF)*gf.H(eps_mu,phi)+useVF*fmin(1.0,fmax(0.0,vf));
        ImH_mu = (1.0-useVF)*gf.ImH(eps_mu,phi) + useVF*(1.0-fmin(1.0,fmax(0.0,vf)));
        nu  = nu_0*ImH_mu+nu_1*H_mu;
        rho  = rho_0*ImH_mu+rho_1*H_mu;

        const double twoThirds = 2.0/3.0; const double div_zero = 1.0e-2*fmin(nu_0,nu_1);
        mom_u_source += twoThirds*turb_grad_0[0];
        mom_v_source += twoThirds*turb_grad_0[1];
        mom_w_source += twoThirds*turb_grad_0[2];

        //--- closure model specific ---
        //k-epsilon
        nu_t_keps = eddy_visc_coef_0*turb_var_0*turb_var_0/(fabs(turb_var_1) + div_zero);
        //k-omega
        nu_t_komega = turb_var_0/(fabs(turb_var_1) + div_zero);
        //
        nu_t = isKEpsilon*nu_t_keps + (1.0-isKEpsilon)*nu_t_komega;
        //mwf debug
        //if (nu_t > 1.e6*nu)
        //{
        //  std::cout<<"cppRANS3PF WARNING isKEpsilon = "<<isKEpsilon<<" nu_t = " <<nu_t<<" nu= "<<nu<<" k= "<<turb_var_0<<" turb_var_1= "<<turb_var_1<<std::endl;
        //}

        nu_t = fmax(nu_t,1.0e-4*nu); //limit according to Lew, Buscaglia etal 01
        //mwf hack
        nu_t     = fmin(nu_t,1.0e6*nu);

        eddy_viscosity = nu_t*rho; // mql. CHECK.
        //u momentum diffusion tensor
        mom_uu_diff_ten[0] += porosity*2.0*eddy_viscosity;
        mom_uu_diff_ten[1] += porosity*eddy_viscosity;
        mom_uu_diff_ten[2] += porosity*eddy_viscosity;

        mom_uv_diff_ten[0]+=porosity*eddy_viscosity;

        mom_uw_diff_ten[0]+=porosity*eddy_viscosity;

        //v momentum diffusion tensor
        mom_vv_diff_ten[0] += porosity*eddy_viscosity;
        mom_vv_diff_ten[1] += porosity*2.0*eddy_viscosity;
        mom_vv_diff_ten[2] += porosity*eddy_viscosity;

        mom_vu_diff_ten[0]+=porosity*eddy_viscosity;

        mom_vw_diff_ten[0]+=porosity*eddy_viscosity;

        //w momentum diffusion tensor
        mom_ww_diff_ten[0] += porosity*eddy_viscosity;
        mom_ww_diff_ten[1] += porosity*eddy_viscosity;
        mom_ww_diff_ten[2] += porosity*2.0*eddy_viscosity;

        mom_wu_diff_ten[0]+=porosity*eddy_viscosity;

        mom_wv_diff_ten[0]+=eddy_viscosity;
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
          nrm_df += df[I] * df[I];
        nrm_df = sqrt(nrm_df);
        if (density > 1.0e-8)
          cfl = nrm_df/(h*density);//this is really cfl/dt, but that's what we want to know, the step controller expect this
        else
          cfl = nrm_df/h;
        oneByAbsdt =  fabs(dmt);
        tau_v = 1.0/(4.0*viscosity/(h*h) + 2.0*nrm_df/h + oneByAbsdt);
        tau_p = (4.0*viscosity + 2.0*nrm_df*h + oneByAbsdt*h*h)/pfac;
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
        subgridErrorU = -tau_v*pdeResidualU;
        subgridErrorV = -tau_v*pdeResidualV;
        subgridErrorW = -tau_v*pdeResidualW;
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
            dsubgridErrorP_du[j] = -tau_p*dpdeResidualP_du[j];
            dsubgridErrorP_dv[j] = -tau_p*dpdeResidualP_dv[j];
            dsubgridErrorP_dw[j] = -tau_p*dpdeResidualP_dw[j];
            /* GLS  momentum*/
            /* u */
            dsubgridErrorU_dp[j] = -tau_v * dpdeResidualU_dp[j];
            dsubgridErrorU_du[j] = -tau_v * dpdeResidualU_du[j];
            /* v */
            dsubgridErrorV_dp[j] = -tau_v*dpdeResidualV_dp[j];
            dsubgridErrorV_dv[j] = -tau_v*dpdeResidualV_dv[j];
            /* w */
            dsubgridErrorW_dp[j] = -tau_v*dpdeResidualW_dp[j];
            dsubgridErrorW_dw[j] = -tau_v*dpdeResidualW_dw[j];
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
                                            const double& porosity, //mql. CHECK. Multiply by rho outside
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
        flux_wmom = 0.0;
        flowSpeedNormal=porosity*(n[0]*velocity_star[0] +
                                  n[1]*velocity_star[1] +
                                  n[2]*velocity_star[2]);
        velocity[0] = u;
        velocity[1] = v;
        velocity[2] = w;
        if (isDOFBoundary_u != 1)
          {
            flux_mass += n[0]*f_mass[0];
            if (flowSpeedNormal < 0.0)
              {
                flux_umom+=flowSpeedNormal*(0.0 - u);
              }
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
            if (flowSpeedNormal < 0.0)
              {
                flux_vmom+=flowSpeedNormal*(0.0 - v);
              }
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
        if (isDOFBoundary_w != 1)
          {
            flux_mass+=n[2]*f_mass[2];
            if (flowSpeedNormal < 0.0)
              {
                flux_wmom+=flowSpeedNormal*(0.0 - w);
              }
          }
        else
          {
            flux_mass +=n[2]*f_mass[2];
            if (flowSpeedNormal < 0.0)
              {
                flux_wmom+=flowSpeedNormal*(bc_w - w);
                velocity[2] = bc_w;
              }
          }
        /* if (isDOFBoundary_p == 1) */
        /*   { */
        /*     flux_umom+= n[0]*(bc_p*bc_oneByRho-p*oneByRho); */
        /*     flux_vmom+= n[1]*(bc_p*bc_oneByRho-p*oneByRho); */
        /*     flux_wmom+= n[2]*(bc_p*bc_oneByRho-p*oneByRho); */
        /*   } */
        /* if (isFluxBoundary_p == 1) */
        /*   { */
        /*     /\* velocity[0] += (bc_flux_mass - flux_mass)*n[0]; *\/ */
        /*     /\* velocity[1] += (bc_flux_mass - flux_mass)*n[1]; *\/ */
        /*     /\* velocity[2] += (bc_flux_mass - flux_mass)*n[2]; *\/ */
        /*     flux_mass = bc_flux_mass; */
        /*   } */
        if (isFluxBoundary_u == 1)
          {
            flux_umom = bc_flux_umom;
	    //velocity[0] = bc_flux_umom/porosity;
          }
        if (isFluxBoundary_v == 1)
          {
            flux_vmom = bc_flux_vmom;
	    //velocity[1] = bc_flux_umom/porosity;
          }
        if (isFluxBoundary_w == 1)
          {
            flux_wmom = bc_flux_wmom;
	    //velocity[2] = bc_flux_umom/porosity;
          }
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
                                                       const double& porosity, //mql. CHECK. Multiply by rho outside
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
        dflux_mass_dw = 0.0;

        dflux_umom_dp = 0.0;
        dflux_umom_du = 0.0;
        dflux_umom_dv = 0.0;
        dflux_umom_dw = 0.0;

        dflux_vmom_dp = 0.0;
        dflux_vmom_du = 0.0;
        dflux_vmom_dv = 0.0;
        dflux_vmom_dw = 0.0;

        dflux_wmom_dp = 0.0;
        dflux_wmom_du = 0.0;
        dflux_wmom_dv = 0.0;
        dflux_wmom_dw = 0.0;
        flowSpeedNormal=porosity*(n[0]*velocity_star[0] +
                                  n[1]*velocity_star[1] +
                                  n[2]*velocity_star[2]);
        if (isDOFBoundary_u != 1)
          {
            dflux_mass_du += n[0]*df_mass_du[0];
            if (flowSpeedNormal < 0.0)
              dflux_umom_du -= flowSpeedNormal;
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
            if (flowSpeedNormal < 0.0)
              dflux_vmom_dv -= flowSpeedNormal;
          }
        else
          {
            dflux_mass_dv += n[1]*df_mass_dv[1];
            if (flowSpeedNormal < 0.0)
              dflux_vmom_dv -= flowSpeedNormal;
          }
        if (isDOFBoundary_w != 1)
          {
            dflux_mass_dw+=n[2]*df_mass_dw[2];
            if (flowSpeedNormal < 0.0)
              dflux_wmom_dw -= flowSpeedNormal;
          }
        else
          {
            dflux_mass_dw += n[2]*df_mass_dw[2];
            if (flowSpeedNormal < 0.0)
              dflux_wmom_dw -= flowSpeedNormal;
          }
        /* if (isDOFBoundary_p == 1) */
        /*   { */
        /*     dflux_umom_dp= -n[0]*oneByRho; */
        /*     dflux_vmom_dp= -n[1]*oneByRho; */
        /*     dflux_wmom_dp= -n[2]*oneByRho; */
        /*   } */
        /* if (isFluxBoundary_p == 1) */
        /*   { */
        /*     dflux_mass_du = 0.0; */
        /*     dflux_mass_dv = 0.0; */
        /*     dflux_mass_dw = 0.0; */
        /*   } */
        if (isFluxBoundary_u == 1)
          {
            dflux_umom_dp = 0.0;
            dflux_umom_du = 0.0;
            dflux_umom_dv = 0.0;
            dflux_umom_dw = 0.0;
          }
        if (isFluxBoundary_v == 1)
          {
            dflux_vmom_dp = 0.0;
            dflux_vmom_du = 0.0;
            dflux_vmom_dv = 0.0;
            dflux_vmom_dw = 0.0;
          }
        if (isFluxBoundary_w == 1)
          {
            dflux_wmom_dp = 0.0;
            dflux_wmom_du = 0.0;
            dflux_wmom_dv = 0.0;
            dflux_wmom_dw = 0.0;
          }
      }

      inline
        void exteriorNumericalDiffusiveFlux(const double& eps,
                                            const double& phi,
                                            int* rowptr,
                                            int* colind,
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
                for(int m=rowptr[I];m<rowptr[I+1];m++)
                  {
                    diffusiveVelocityComponent_I -= a[m]*grad_potential[colind[m]];
                    max_a = fmax(max_a,a[m]);
                  }
                flux+= diffusiveVelocityComponent_I*n[I];
              }
            penaltyFlux = max_a*penalty*(u-bc_u);
            flux += penaltyFlux;
            //contact line slip
            //flux*=(gf.D(eps,0) - gf.D(eps,phi))/gf.D(eps,0);
          }
        else
          {
            std::cerr<<"RANS3PF: warning, diffusion term with no boundary condition set, setting diffusive flux to 0.0"<<std::endl;
            flux = 0.0;
          }
      }

      inline
        double ExteriorNumericalDiffusiveFluxJacobian(const double& eps,
                                                      const double& phi,
                                                      int* rowptr,
                                                      int* colind,
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
                for(int m=rowptr[I];m<rowptr[I+1];m++)
                  {
                    dvel_I -= a[m]*grad_v[colind[m]];
                    max_a = fmax(max_a,a[m]);
                  }
                tmp += dvel_I*n[I];
              }
            tmp +=max_a*penalty*v;
            //contact line slip
            //tmp*=(gf.D(eps,0) - gf.D(eps,phi))/gf.D(eps,0);
          }
        return tmp;
      }

      void get_symmetric_gradient_dot_vec(const double *grad_u, const double *grad_v, const double *grad_w, const double *n,double res[3])
      {
//          res[0] =         2.0*grad_u[0]*n[0]+(grad_u[1]+grad_v[0])*n[1]+(grad_u[2]+grad_w[0])*n[2];
//          res[1] = (grad_v[0]+grad_u[1])*n[0]+          2*grad_v[1]*n[1]+(grad_v[2]+grad_w[1])*n[2];
//          res[2] = (grad_w[0]+grad_u[2])*n[0]+(grad_w[1]+grad_v[2])*n[1]+          2*grad_w[2]*n[2];
          res[0] = grad_u[0]*n[0]+grad_u[1]*n[1]+grad_u[2]*n[2];
          res[1] = grad_v[0]*n[0]+grad_v[1]*n[1]+grad_v[2]*n[2];
          res[2] = grad_w[0]*n[0]+grad_w[1]*n[1]+grad_w[2]*n[2];;
      }
      // n is the outward unit normal direction
      void get_stress_in_n(const double *grad_u, const double *grad_v, const double *grad_w, const double *n, double p, double mu, double f[3])
      {
          f[0] =         2.0*grad_u[0]*n[0]+(grad_u[1]+grad_v[0])*n[1]+(grad_u[2]+grad_w[0])*n[2];
          f[1] = (grad_v[0]+grad_u[1])*n[0]+          2*grad_v[1]*n[1]+(grad_v[2]+grad_w[1])*n[2];
          f[2] = (grad_w[0]+grad_u[2])*n[0]+(grad_w[1]+grad_v[2])*n[1]+          2*grad_w[2]*n[2];

          f[0] *= mu;
          f[1] *= mu;
          f[2] *= mu;

          f[0] -= p*n[0];
          f[1] -= p*n[1];
          f[2] -= p*n[2];
      }
      void get_cross_product(const double *u, const double *v,double res[3])
      {
          res[0] = u[1]*v[2]-u[2]*v[1];
          res[1] = u[2]*v[0]-u[0]*v[2];
          res[2] = u[0]*v[1]-u[1]*v[0];
      }
      double get_dot_product(const double *u, const double *v)
      {
          return u[0]*v[0]+u[1]*v[1]+u[2]*v[2];
      }
      int get_distance_to_ball(int n_balls,double* ball_center, double* ball_radius, double x, double y, double z, double& distance)
      {
          distance = 1e10;
          int index = -1;
          double d_ball_i;
          for (int i=0; i<n_balls; ++i)
          {
              d_ball_i = std::sqrt((ball_center[i*3+0]-x)*(ball_center[i*3+0]-x)
                                  +(ball_center[i*3+1]-y)*(ball_center[i*3+1]-y)
#ifndef USE_CYLINDER_AS_PARTICLE
                                  +(ball_center[i*3+2]-z)*(ball_center[i*3+2]-z)
#endif
                                  ) - ball_radius[i];
              if(d_ball_i<distance)
              {
                  distance = d_ball_i;
                  index = i;
              }
          }
          return index;
      }
      void get_distance_to_ith_ball(int n_balls,double* ball_center, double* ball_radius,
                                  int I,
                                  double x, double y, double z,
                                  double& distance)
      {
          distance = std::sqrt((ball_center[I*3+0]-x)*(ball_center[I*3+0]-x)
                               + (ball_center[I*3+1]-y)*(ball_center[I*3+1]-y)
#ifndef USE_CYLINDER_AS_PARTICLE
                               + (ball_center[I*3+2]-z)*(ball_center[I*3+2]-z)
#endif
                            ) - ball_radius[I];
      }
      void get_normal_to_ith_ball(int n_balls,double* ball_center, double* ball_radius,
                                  int I,
                                  double x, double y, double z,
                                  double& nx, double& ny, double& nz)
      {
          double distance = std::sqrt((ball_center[I*3+0]-x)*(ball_center[I*3+0]-x)
                                    + (ball_center[I*3+1]-y)*(ball_center[I*3+1]-y)
#ifndef USE_CYLINDER_AS_PARTICLE
                                    + (ball_center[I*3+2]-z)*(ball_center[I*3+2]-z)
#endif
                            );
          nx = (x - ball_center[I*3+0])/(distance+1e-10);
          ny = (y - ball_center[I*3+1])/(distance+1e-10);
#ifdef USE_CYLINDER_AS_PARTICLE
          nz = 0.0;
#else
          nz = (z - ball_center[I*3+2])/(distance+1e-10);
#endif
      }
      void get_velocity_to_ith_ball(int n_balls,double* ball_center, double* ball_radius,
                                    double* ball_velocity, double* ball_angular_velocity,
                                    int I,
                                    double x, double y, double z,
                                    double& vx, double& vy, double& vz)
      {
#ifdef USE_CYLINDER_AS_PARTICLE
          double position[3]={x-ball_center[3*I + 0],y-ball_center[3*I + 1],0.0};
#else
          double position[3]={x-ball_center[3*I + 0],y-ball_center[3*I + 1],z-ball_center[3*I + 2]};
#endif
          double angular_cross_position[3];
          get_cross_product(&ball_angular_velocity[3*I + 0],position,angular_cross_position);
          vx = ball_velocity[3*I + 0] + angular_cross_position[0];
          vy = ball_velocity[3*I + 1] + angular_cross_position[1];
          vz = ball_velocity[3*I + 2] + angular_cross_position[2];

      }
      void calculateResidual(arguments_dict& args,
                             bool useExact)
     {
        proteus::pyarray<double>& mesh_trial_ref = args.array<double>("mesh_trial_ref");
        proteus::pyarray<double>& mesh_grad_trial_ref = args.array<double>("mesh_grad_trial_ref");
        proteus::pyarray<double>& mesh_dof = args.array<double>("mesh_dof");
        proteus::pyarray<double>& mesh_velocity_dof = args.array<double>("mesh_velocity_dof");
        double MOVING_DOMAIN = args.scalar<double>("MOVING_DOMAIN");
        double PSTAB = args.scalar<double>("PSTAB");
        proteus::pyarray<int>& mesh_l2g = args.array<int>("mesh_l2g");
        proteus::pyarray<double>& x_ref = args.array<double>("x_ref");
        proteus::pyarray<double>& dV_ref = args.array<double>("dV_ref");
        int nDOF_per_element_pressure = args.scalar<int>("nDOF_per_element_pressure");
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
        proteus::pyarray<double>& vel_hess_trial_ref = args.array<double>("vel_hess_trial_ref");
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
        int nElements_global = args.scalar<int>("nElements_global");
        int nElements_owned = args.scalar<int>("nElements_owned");
        int nElementBoundaries_global = args.scalar<int>("nElementBoundaries_global");
        int nElementBoundaries_owned = args.scalar<int>("nElementBoundaries_owned");
        int nNodes_owned = args.scalar<int>("nNodes_owned");
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
        double smagorinskyConstant = args.scalar<double>("smagorinskyConstant");
        int turbulenceClosureModel = args.scalar<int>("turbulenceClosureModel");
        double Ct_sge = args.scalar<double>("Ct_sge");
        double Cd_sge = args.scalar<double>("Cd_sge");
        double C_dc = args.scalar<double>("C_dc");
        double C_b = args.scalar<double>("C_b");
        const proteus::pyarray<double>& eps_solid = args.array<double>("eps_solid");
        const proteus::pyarray<double>& ebq_global_phi_solid = args.array<double>("ebq_global_phi_solid");
        const proteus::pyarray<double>& ebq_global_grad_phi_solid = args.array<double>("ebq_global_grad_phi_solid");
        const proteus::pyarray<double>& ebq_particle_velocity_solid = args.array<double>("ebq_particle_velocity_solid");
        proteus::pyarray<double>& phi_solid_nodes = args.array<double>("phi_solid_nodes");
        proteus::pyarray<double>& phi_solid = args.array<double>("phi_solid");
        const proteus::pyarray<double>& q_velocity_solid = args.array<double>("q_velocity_solid");
        const proteus::pyarray<double>& q_velocityStar_solid = args.array<double>("q_velocityStar_solid");
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
        proteus::pyarray<int>& p_l2g = args.array<int>("p_l2g");
        proteus::pyarray<int>& vel_l2g = args.array<int>("vel_l2g");
        proteus::pyarray<double>& p_dof = args.array<double>("p_dof");
        proteus::pyarray<double>& u_dof = args.array<double>("u_dof");
        proteus::pyarray<double>& v_dof = args.array<double>("v_dof");
        proteus::pyarray<double>& w_dof = args.array<double>("w_dof");
        proteus::pyarray<double>& u_dof_old = args.array<double>("u_dof_old");
        proteus::pyarray<double>& v_dof_old = args.array<double>("v_dof_old");
        proteus::pyarray<double>& w_dof_old = args.array<double>("w_dof_old");
        proteus::pyarray<double>& u_dof_old_old = args.array<double>("u_dof_old_old");
        proteus::pyarray<double>& v_dof_old_old = args.array<double>("v_dof_old_old");
        proteus::pyarray<double>& w_dof_old_old = args.array<double>("w_dof_old_old");
        proteus::pyarray<double>& uStar_dof = args.array<double>("uStar_dof");
        proteus::pyarray<double>& vStar_dof = args.array<double>("vStar_dof");
        proteus::pyarray<double>& wStar_dof = args.array<double>("wStar_dof");
        proteus::pyarray<double>& g = args.array<double>("g");
        const double useVF = args.scalar<double>("useVF");
        proteus::pyarray<double>& vf = args.array<double>("vf");
        proteus::pyarray<double>& phi = args.array<double>("phi");
        proteus::pyarray<double>& phi_dof = args.array<double>("phi_dof");
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
        proteus::pyarray<int>& sdInfo_u_u_rowptr = args.array<int>("sdInfo_u_u_rowptr");
        proteus::pyarray<int>& sdInfo_u_u_colind = args.array<int>("sdInfo_u_u_colind");
        proteus::pyarray<int>& sdInfo_u_v_rowptr = args.array<int>("sdInfo_u_v_rowptr");
        proteus::pyarray<int>& sdInfo_u_v_colind = args.array<int>("sdInfo_u_v_colind");
        proteus::pyarray<int>& sdInfo_u_w_rowptr = args.array<int>("sdInfo_u_w_rowptr");
        proteus::pyarray<int>& sdInfo_u_w_colind = args.array<int>("sdInfo_u_w_colind");
        proteus::pyarray<int>& sdInfo_v_v_rowptr = args.array<int>("sdInfo_v_v_rowptr");
        proteus::pyarray<int>& sdInfo_v_v_colind = args.array<int>("sdInfo_v_v_colind");
        proteus::pyarray<int>& sdInfo_v_u_rowptr = args.array<int>("sdInfo_v_u_rowptr");
        proteus::pyarray<int>& sdInfo_v_u_colind = args.array<int>("sdInfo_v_u_colind");
        proteus::pyarray<int>& sdInfo_v_w_rowptr = args.array<int>("sdInfo_v_w_rowptr");
        proteus::pyarray<int>& sdInfo_v_w_colind = args.array<int>("sdInfo_v_w_colind");
        proteus::pyarray<int>& sdInfo_w_w_rowptr = args.array<int>("sdInfo_w_w_rowptr");
        proteus::pyarray<int>& sdInfo_w_w_colind = args.array<int>("sdInfo_w_w_colind");
        proteus::pyarray<int>& sdInfo_w_u_rowptr = args.array<int>("sdInfo_w_u_rowptr");
        proteus::pyarray<int>& sdInfo_w_u_colind = args.array<int>("sdInfo_w_u_colind");
        proteus::pyarray<int>& sdInfo_w_v_rowptr = args.array<int>("sdInfo_w_v_rowptr");
        proteus::pyarray<int>& sdInfo_w_v_colind = args.array<int>("sdInfo_w_v_colind");
        int offset_p = args.scalar<int>("offset_p");
        int offset_u = args.scalar<int>("offset_u");
        int offset_v = args.scalar<int>("offset_v");
        int offset_w = args.scalar<int>("offset_w");
        int stride_p = args.scalar<int>("stride_p");
        int stride_u = args.scalar<int>("stride_u");
        int stride_v = args.scalar<int>("stride_v");
        int stride_w = args.scalar<int>("stride_w");
        proteus::pyarray<double>& globalResidual = args.array<double>("globalResidual");
        int nExteriorElementBoundaries_global = args.scalar<int>("nExteriorElementBoundaries_global");
        proteus::pyarray<int>& exteriorElementBoundariesArray = args.array<int>("exteriorElementBoundariesArray");
        proteus::pyarray<int>& elementBoundariesArray = args.array<int>("elementBoundariesArray");
        proteus::pyarray<int>& elementBoundaryElementsArray = args.array<int>("elementBoundaryElementsArray");
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
        proteus::pyarray<double>& q_grad_u = args.array<double>("q_grad_u");
        proteus::pyarray<double>& q_grad_v = args.array<double>("q_grad_v");
        proteus::pyarray<double>& q_grad_w = args.array<double>("q_grad_w");
        proteus::pyarray<double>& q_divU = args.array<double>("q_divU");
        proteus::pyarray<double>& ebqe_grad_u = args.array<double>("ebqe_grad_u");
        proteus::pyarray<double>& ebqe_grad_v = args.array<double>("ebqe_grad_v");
        proteus::pyarray<double>& ebqe_grad_w = args.array<double>("ebqe_grad_w");
        proteus::pyarray<double>& flux = args.array<double>("flux");
        proteus::pyarray<double>& elementResidual_p_save = args.array<double>("elementResidual_p_save");
        proteus::pyarray<int>& elementFlags = args.array<int>("elementFlags");
        proteus::pyarray<int>& boundaryFlags = args.array<int>("boundaryFlags");
        proteus::pyarray<double>& barycenters = args.array<double>("barycenters");
        proteus::pyarray<double>& wettedAreas = args.array<double>("wettedAreas");
        proteus::pyarray<double>& netForces_p = args.array<double>("netForces_p");
        proteus::pyarray<double>& netForces_v = args.array<double>("netForces_v");
        proteus::pyarray<double>& netMoments = args.array<double>("netMoments");
        proteus::pyarray<double>& q_rho = args.array<double>("q_rho");
        proteus::pyarray<double>& ebqe_rho = args.array<double>("ebqe_rho");
        proteus::pyarray<double>& q_nu = args.array<double>("q_nu");
        proteus::pyarray<double>& ebqe_nu = args.array<double>("ebqe_nu");
        int nParticles = args.scalar<int>("nParticles");
        double particle_epsFact = args.scalar<double>("particle_epsFact");
        double particle_alpha = args.scalar<double>("particle_alpha");
        double particle_beta = args.scalar<double>("particle_beta");
        double particle_penalty_constant = args.scalar<double>("particle_penalty_constant");
        proteus::pyarray<double>& particle_signed_distances = args.array<double>("particle_signed_distances");
        proteus::pyarray<double>& particle_signed_distance_normals = args.array<double>("particle_signed_distance_normals");
        proteus::pyarray<double>& particle_velocities = args.array<double>("particle_velocities");
        proteus::pyarray<double>& particle_centroids = args.array<double>("particle_centroids");
        proteus::pyarray<double>& particle_netForces = args.array<double>("particle_netForces");
        proteus::pyarray<double>& particle_netMoments = args.array<double>("particle_netMoments");
        proteus::pyarray<double>& particle_surfaceArea = args.array<double>("particle_surfaceArea");
        double particle_nitsche = args.scalar<double>("particle_nitsche");
        int use_ball_as_particle = args.scalar<int>("use_ball_as_particle");
        proteus::pyarray<double>& ball_center = args.array<double>("ball_center");
        proteus::pyarray<double>& ball_radius = args.array<double>("ball_radius");
        proteus::pyarray<double>& ball_velocity = args.array<double>("ball_velocity");
        proteus::pyarray<double>& ball_angular_velocity = args.array<double>("ball_angular_velocity");
        proteus::pyarray<double>& phisError = args.array<double>("phisError");
        proteus::pyarray<double>& phisErrorNodal = args.array<double>("phisErrorNodal");
        int USE_SUPG = args.scalar<int>("USE_SUPG");
        int ARTIFICIAL_VISCOSITY = args.scalar<int>("ARTIFICIAL_VISCOSITY");
        double cMax = args.scalar<double>("cMax");
        double cE = args.scalar<double>("cE");
        int MULTIPLY_EXTERNAL_FORCE_BY_DENSITY = args.scalar<int>("MULTIPLY_EXTERNAL_FORCE_BY_DENSITY");
        proteus::pyarray<double>& forcex = args.array<double>("forcex");
        proteus::pyarray<double>& forcey = args.array<double>("forcey");
        proteus::pyarray<double>& forcez = args.array<double>("forcez");
        int KILL_PRESSURE_TERM = args.scalar<int>("KILL_PRESSURE_TERM");
        double dt = args.scalar<double>("dt");
        proteus::pyarray<double>& quantDOFs = args.array<double>("quantDOFs");
        int MATERIAL_PARAMETERS_AS_FUNCTION = args.scalar<int>("MATERIAL_PARAMETERS_AS_FUNCTION");
        proteus::pyarray<double>& density_as_function = args.array<double>("density_as_function");
        proteus::pyarray<double>& dynamic_viscosity_as_function = args.array<double>("dynamic_viscosity_as_function");
        proteus::pyarray<double>& ebqe_density_as_function = args.array<double>("ebqe_density_as_function");
        proteus::pyarray<double>& ebqe_dynamic_viscosity_as_function = args.array<double>("ebqe_dynamic_viscosity_as_function");
        double order_polynomial = args.scalar<double>("order_polynomial");
        proteus::pyarray<double>& isActiveDOF = args.array<double>("isActiveDOF");
        int USE_SBM = args.scalar<int>("USE_SBM");
        proteus::pyarray<double>& ncDrag = args.array<double>("ncDrag");
        proteus::pyarray<double>& betaDrag = args.array<double>("betaDrag");
        proteus::pyarray<double>& vos_vel_nodes = args.array<double>("vos_vel_nodes");
        proteus::pyarray<double>&  entropyResidualPerNode = args.array<double>("entropyResidualPerNode");
        proteus::pyarray<double>&  laggedEntropyResidualPerNode = args.array<double>("laggedEntropyResidualPerNode");
        proteus::pyarray<double>&  uStar_dMatrix = args.array<double>("uStar_dMatrix");
        proteus::pyarray<double>&  vStar_dMatrix = args.array<double>("vStar_dMatrix");
        proteus::pyarray<double>&  wStar_dMatrix = args.array<double>("wStar_dMatrix");
        int numDOFs_1D = args.scalar<int>("numDOFs_1D");
        int NNZ_1D = args.scalar<int>("NNZ_1D");
        proteus::pyarray<int>& csrRowIndeces_1D = args.array<int>("csrRowIndeces_1D");
        proteus::pyarray<int>& csrColumnOffsets_1D = args.array<int>("csrColumnOffsets_1D");
        proteus::pyarray<int>& rowptr_1D = args.array<int>("rowptr_1D");
        proteus::pyarray<int>& colind_1D = args.array<int>("colind_1D");
        proteus::pyarray<double>& uStar_psi_dof = args.array<double>("uStar_psi_dof");
        proteus::pyarray<double>& vStar_psi_dof = args.array<double>("vStar_psi_dof");
        proteus::pyarray<double>& wStar_psi_dof = args.array<double>("wStar_psi_dof");
        proteus::pyarray<double>& isBoundary_1D = args.array<double>("isBoundary_1D");
        int INT_BY_PARTS_PRESSURE = args.scalar<int>("INT_BY_PARTS_PRESSURE");
        gf.useExact=useExact;
        gf_s.useExact=useExact;
	double element_uStar_He[nElements_global], element_vStar_He[nElements_global], element_wStar_He[nElements_global];
	uStar_hi.resize(numDOFs_1D,0.0);
	vStar_hi.resize(numDOFs_1D,0.0);
	wStar_hi.resize(numDOFs_1D,0.0);
	den_hi.resize(numDOFs_1D,0.0);
	uStar_min_hiHe.resize(numDOFs_1D,0.0);
	vStar_min_hiHe.resize(numDOFs_1D,0.0);
	wStar_min_hiHe.resize(numDOFs_1D,0.0);
	uStar_gamma.resize(numDOFs_1D,0.0);
	vStar_gamma.resize(numDOFs_1D,0.0);
	wStar_gamma.resize(numDOFs_1D,0.0);
	TransportMatrix.resize(NNZ_1D,0.0);
	TransposeTransportMatrix.resize(NNZ_1D,0.0);
	uStar_psi.resize(numDOFs_1D,0.0);
	vStar_psi.resize(numDOFs_1D,0.0);
	wStar_psi.resize(numDOFs_1D,0.0);

	if (ARTIFICIAL_VISCOSITY==3 || ARTIFICIAL_VISCOSITY==4)
	  {
	    for (int i=0; i<NNZ_1D; i++)
	      {
		uStar_dMatrix[i]=0.;
		vStar_dMatrix[i]=0.;
		wStar_dMatrix[i]=0.;
		TransportMatrix[i] = 0.;
		TransposeTransportMatrix[i] = 0.;
	      }
	    for (int i=0; i<numDOFs_1D; i++)
	      {
		uStar_min_hiHe[i] = 1E100;
		vStar_min_hiHe[i] = 1E100;
		wStar_min_hiHe[i] = 1E100;
		entropyResidualPerNode[i]=0.;
		uStar_hi[i] = 0.;
		vStar_hi[i] = 0;
		wStar_hi[i] = 0.;
		den_hi[i] = 0.;
	      }
	  }

        //
        //loop over elements to compute volume integrals and load them into element and global residual
        //
        double mesh_volume_conservation=0.0,
          mesh_volume_conservation_weak=0.0,
          mesh_volume_conservation_err_max=0.0,
          mesh_volume_conservation_err_max_weak=0.0;
        double globalConservationError=0.0;
        const int nQuadraturePoints_global(nElements_global*nQuadraturePoints_element);
        for(int eN=0;eN<nElements_global;eN++)
          {
	    double  elementTransport[nDOF_test_element][nDOF_trial_element];
	    double  elementTransposeTransport[nDOF_test_element][nDOF_trial_element];
            //declare local storage for element residual and initialize
            double elementResidual_p[nDOF_test_element],elementResidual_mesh[nDOF_test_element],
              elementResidual_u[nDOF_test_element],
              elementResidual_v[nDOF_test_element],
              mom_u_source_i[nDOF_test_element],
              mom_v_source_i[nDOF_test_element],
              mom_w_source_i[nDOF_test_element],
              betaDrag_i[nDOF_test_element],
              vos_i[nDOF_test_element],
              phisErrorElement[nDOF_test_element],
              elementResidual_w[nDOF_test_element],
	      elementEntropyResidual[nDOF_test_element],
              eps_rho,eps_mu;
            double element_active=1.0;//use 1 since by default it is ibm
            double mesh_volume_conservation_element=0.0,
              mesh_volume_conservation_element_weak=0.0;
	    // for entropy viscosity
	    double linVisc_eN = 0, nlinVisc_eN_num = 0, nlinVisc_eN_den = 0;
	    // for hessians of uStar
	    double det_hess_uStar_Ke=0.0, det_hess_vStar_Ke=0.0, det_hess_wStar_Ke=0.0, area_Ke=0.0;
            for (int i=0;i<nDOF_test_element;i++)
              {
                int eN_i = eN*nDOF_test_element+i;
                elementResidual_p_save[eN_i]=0.0;
                elementResidual_mesh[i]=0.0;
                elementResidual_p[i]=0.0;
                elementResidual_u[i]=0.0;
                elementResidual_v[i]=0.0;
                mom_u_source_i[i]=0.0;
                mom_v_source_i[i]=0.0;
                mom_w_source_i[i]=0.0;
                betaDrag_i[i]=0.0;
                vos_i[i]=0.0;
                phisErrorElement[i]=0.0;
                elementResidual_w[i]=0.0;
		elementEntropyResidual[i]=0.0;
		if (ARTIFICIAL_VISCOSITY==3 || ARTIFICIAL_VISCOSITY==4)
		  {
		    for (int j=0;j<nDOF_trial_element;j++)
		      {
			elementTransport[i][j]=0.0;
			elementTransposeTransport[i][j]=0.0;
		      }
		  }
              }//i
            //Use for plotting result
            if(use_ball_as_particle==1)
            {
                for (int I=0;I<nDOF_mesh_trial_element;I++)
                    get_distance_to_ball(nParticles, ball_center.data(), ball_radius.data(),
                                                mesh_dof[3*mesh_l2g[eN*nDOF_mesh_trial_element+I]+0],
                                                mesh_dof[3*mesh_l2g[eN*nDOF_mesh_trial_element+I]+1],
                                                mesh_dof[3*mesh_l2g[eN*nDOF_mesh_trial_element+I]+2],
                                                phi_solid_nodes[mesh_l2g[eN*nDOF_mesh_trial_element+I]]);
            }
            if(USE_SBM>0)
            {
                //isActiveDOF has value 1 for ibm.
                //isActiveDOF is initialized in python side.
                //
                //detect cut cells
                //
                double _distance[nDOF_mesh_trial_element]={0.0};
                int pos_counter=0;
                for (int I=0;I<nDOF_mesh_trial_element;I++)
                  {
                    if(use_ball_as_particle==1)
                      {
                        get_distance_to_ball(nParticles, ball_center.data(), ball_radius.data(),
                                             mesh_dof[3*mesh_l2g[eN*nDOF_mesh_trial_element+I]+0],
                                             mesh_dof[3*mesh_l2g[eN*nDOF_mesh_trial_element+I]+1],
                                             mesh_dof[3*mesh_l2g[eN*nDOF_mesh_trial_element+I]+2],
                                             _distance[I]);
                      }
                    else
                      {
                        _distance[I] = phi_solid_nodes[mesh_l2g[eN*nDOF_mesh_trial_element+I]];
                      }
                    if ( _distance[I] >= 0)//fully in fluid
                      pos_counter++;
                  }
                if (pos_counter == 3)//surrogate face
                {
                    element_active=0.0;
                    int opp_node=-1;
                    for (int I=0;I<nDOF_mesh_trial_element;I++)
                    {
                        if (_distance[I] < 0)
                            opp_node = I;
                    }
                    assert(opp_node >=0);
                    assert(opp_node <nDOF_mesh_trial_element);
                    //For parallel. Two reasons:
                    //if none of nodes of this edge is owned by this processor,
                    //1. The surrogate_boundary_elements corresponding to this edge is -1, which gives 0 JacDet and infty h_penalty.
                    //2. there is no contribution of the integral over this edge to Jacobian and residual.
                    const int ebN = elementBoundariesArray[eN*nDOF_mesh_trial_element+opp_node];//only works for simplices
                    const int eN_oppo = (eN == elementBoundaryElementsArray[ebN*2+0])?elementBoundaryElementsArray[ebN*2+1]:elementBoundaryElementsArray[ebN*2+0];
                    if((mesh_l2g[eN*nDOF_mesh_trial_element+(opp_node+1)%4]<nNodes_owned
                        || mesh_l2g[eN*nDOF_mesh_trial_element+(opp_node+2)%4]<nNodes_owned
                        || mesh_l2g[eN*nDOF_mesh_trial_element+(opp_node+3)%4]<nNodes_owned
                       )&&eN_oppo !=-1)//not a boundary face
                    {
                        surrogate_boundaries.push_back(ebN);
                        //now find which element neighbor this element is
                        //since each face has 2 neighbor elements.
                        //YY: what if this face is a boundary face?
                        if (eN == elementBoundaryElementsArray[ebN*2+0])
                            surrogate_boundary_elements.push_back(1);
                        else
                            surrogate_boundary_elements.push_back(0);

                        //check which particle this surrogate edge is related to.
                        //The method is to check one quadrature point inside of this element.
                        //It works based on the assumption that the distance between any two particles
                        //is larger than 2*h_min, otherwise it depends on the choice of the quadrature point
                        //or one edge belongs to two particles .
                        //But in any case, phi_s is well defined as the minimum.
                        int j=-1;
                        double distance=1e10, distance_to_ith_particle;
                        if(use_ball_as_particle==1)
                        {
                            double middle_point_coord[3]={0.0};
                            double middle_point_distance;
                            middle_point_coord[0] = (mesh_dof[3*mesh_l2g[eN*nDOF_mesh_trial_element+(opp_node+1)%4]+0]
                                                    +mesh_dof[3*mesh_l2g[eN*nDOF_mesh_trial_element+(opp_node+2)%4]+0]
                                                    +mesh_dof[3*mesh_l2g[eN*nDOF_mesh_trial_element+(opp_node+3)%4]+0])/3.0;
                            middle_point_coord[1] = (mesh_dof[3*mesh_l2g[eN*nDOF_mesh_trial_element+(opp_node+1)%4]+1]
                                                    +mesh_dof[3*mesh_l2g[eN*nDOF_mesh_trial_element+(opp_node+2)%4]+1]
                                                    +mesh_dof[3*mesh_l2g[eN*nDOF_mesh_trial_element+(opp_node+3)%4]+1])/3.0;
                            middle_point_coord[2] = (mesh_dof[3*mesh_l2g[eN*nDOF_mesh_trial_element+(opp_node+1)%4]+2]
                                                    +mesh_dof[3*mesh_l2g[eN*nDOF_mesh_trial_element+(opp_node+2)%4]+2]
                                                    +mesh_dof[3*mesh_l2g[eN*nDOF_mesh_trial_element+(opp_node+3)%4]+2])/3.0;
                            j = get_distance_to_ball(nParticles, ball_center.data(), ball_radius.data(),
                                    middle_point_coord[0],middle_point_coord[1],middle_point_coord[2],
                                    middle_point_distance);
                        }
                      else
                        {
                            for (int i=0;i<nParticles;++i)
                            {
                              distance_to_ith_particle=particle_signed_distances[i*nElements_global*nQuadraturePoints_element
                                                                                 +eN*nQuadraturePoints_element
                                                                                 +0];//0-th quadrature point
                              if (distance_to_ith_particle<distance)
                                {
                                  distance = distance_to_ith_particle;
                                  j = i;
                                }
                            }
                        }
                      surrogate_boundary_particle.push_back(j);
                    }else{
                    //If the integral over the surrogate boundary is needed, we have to make sure all edges are in surrogate_boundaries,
                    //which is based on the assumption that if none of its nodes is owned by the processor, then the edge is not owned
                    //by the processor. This assert is used to make sure this is the case.
                    if(ebN<nElementBoundaries_owned)//eN_oppo ==-1
                      {
                        assert(eN_oppo==-1);
                      }
                  }
                }
                else if (pos_counter == 4)// so the element is in fluid totally
                {
                  element_active=1.0;
                  for (int i=0;i<nDOF_test_element;i++)
                    {
                        isActiveDOF[offset_u+stride_u*vel_l2g[eN*nDOF_trial_element + i]]=1.0;
                        isActiveDOF[offset_v+stride_v*vel_l2g[eN*nDOF_trial_element + i]]=1.0;
                        isActiveDOF[offset_w+stride_w*vel_l2g[eN*nDOF_trial_element + i]]=1.0;
                    }
                }
              else
                {
                  element_active=0.0;
                }
            }
            double element_phi[nDOF_mesh_trial_element], element_phi_s[nDOF_mesh_trial_element];
	    for (int j=0;j<nDOF_mesh_trial_element;j++)
	      {
		int eN_j = eN*nDOF_mesh_trial_element+j;
		element_phi[j] = phi_dof[p_l2g[eN_j]];
		element_phi_s[j] = phi_solid_nodes[p_l2g[eN_j]];
	      }
            double element_nodes[nDOF_mesh_trial_element*3];
	    for (int i=0;i<nDOF_mesh_trial_element;i++)
	      {
		int eN_i=eN*nDOF_mesh_trial_element+i;
                for(int I=0;I<3;I++)
                  element_nodes[i*3 + I] = mesh_dof[mesh_l2g[eN_i]*3 + I];
	      }//i
	    gf_s.calculate(element_phi_s, element_nodes, x_ref.data(), false);
            gf.calculate(element_phi, element_nodes, x_ref.data(), false);
            //
            //loop over quadrature points and compute integrands
            //
            for(int k=0;k<nQuadraturePoints_element;k++)
              {
                gf.set_quad(k);
                gf_s.set_quad(k);
                //compute indices and declare local storage
                int eN_k = eN*nQuadraturePoints_element+k,
                  eN_k_nSpace = eN_k*nSpace,
                  eN_k_3d     = eN_k*3,
                  eN_nDOF_trial_element = eN*nDOF_trial_element;
                double p=0.0,u=0.0,v=0.0,w=0.0,un=0.0,vn=0.0,wn=0.0,
                  grad_p[nSpace],grad_u[nSpace],grad_v[nSpace],grad_w[nSpace],
                  hess_u[nSpace2],hess_v[nSpace2],hess_w[nSpace2],
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
                  mom_w_acc_t=0.0,
                  dmom_w_acc_w_t=0.0,
                  pdeResidual_p=0.0,
                  pdeResidual_u=0.0,
                  pdeResidual_v=0.0,
                  pdeResidual_w=0.0,
                  Lstar_u_p[nDOF_test_element],
                  Lstar_v_p[nDOF_test_element],
                  Lstar_w_p[nDOF_test_element],
                  Lstar_u_u[nDOF_test_element],
                  Lstar_v_v[nDOF_test_element],
                  Lstar_w_w[nDOF_test_element],
                  Lstar_p_u[nDOF_test_element],
                  Lstar_p_v[nDOF_test_element],
                  Lstar_p_w[nDOF_test_element],
                  subgridError_p=0.0,
                  subgridError_u=0.0,
                  subgridError_v=0.0,
                  subgridError_w=0.0,
                  tau_p=0.0,tau_p0=0.0,tau_p1=0.0,
                  tau_v=0.0,tau_v0=0.0,tau_v1=0.0,
                  jac[nSpace*nSpace],
                  jacDet,
                  jacInv[nSpace*nSpace],
                  p_grad_trial[nDOF_trial_element*nSpace],vel_grad_trial[nDOF_trial_element*nSpace],
                  vel_hess_trial[nDOF_trial_element*nSpace2],
                  p_test_dV[nDOF_trial_element],vel_test_dV[nDOF_trial_element],
                  p_grad_test_dV[nDOF_test_element*nSpace],vel_grad_test_dV[nDOF_test_element*nSpace],
		  u_times_vel_grad_test_dV[nDOF_test_element*nSpace], // For entropy residual
                  v_times_vel_grad_test_dV[nDOF_test_element*nSpace], // For entropy residual
		  w_times_vel_grad_test_dV[nDOF_test_element*nSpace], // For entropy residual
                  dV,x,y,z,xt,yt,zt,
                  //
                  porosity,
                  //meanGrainSize,
                  mass_source,
                  dmom_u_source[nSpace],
                  dmom_v_source[nSpace],
                  dmom_w_source[nSpace],
		  //
		  velStar[nSpace], hess_uStar[nSpace2], hess_vStar[nSpace2], hess_wStar[nSpace2],
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
                dV = fabs(jacDet)*dV_ref[k];
                ck.calculateG(jacInv,G,G_dd_G,tr_G);
                //ck.calculateGScale(G,&normal_phi[eN_k_nSpace],h_phi);

                eps_rho = epsFact_rho*(useMetrics*h_phi+(1.0-useMetrics)*elementDiameter[eN]);
                eps_mu  = epsFact_mu *(useMetrics*h_phi+(1.0-useMetrics)*elementDiameter[eN]);
                double particle_eps  = particle_epsFact*(useMetrics*h_phi+(1.0-useMetrics)*elementDiameter[eN]);

                //get the trial function gradients
                /* ck.gradTrialFromRef(&p_grad_trial_ref[k*nDOF_trial_element*nSpace],jacInv,p_grad_trial); */
                ck.gradTrialFromRef(&vel_grad_trial_ref[k*nDOF_trial_element*nSpace],jacInv,vel_grad_trial);
                ck.hessTrialFromRef(&vel_hess_trial_ref[k*nDOF_trial_element*nSpace2],jacInv,vel_hess_trial);
                //get the solution
                /* ck.valFromDOF(p_dof,&p_l2g[eN_nDOF_trial_element],&p_trial_ref[k*nDOF_trial_element],p); */
                p = q_p[eN_k];
                // get solution at quad points
                ck.valFromDOF(u_dof.data(),&vel_l2g[eN_nDOF_trial_element],&vel_trial_ref[k*nDOF_trial_element],u);
                ck.valFromDOF(v_dof.data(),&vel_l2g[eN_nDOF_trial_element],&vel_trial_ref[k*nDOF_trial_element],v);
                ck.valFromDOF(w_dof.data(),&vel_l2g[eN_nDOF_trial_element],&vel_trial_ref[k*nDOF_trial_element],w);
                // get old solution at quad points
                ck.valFromDOF(u_dof_old.data(),&vel_l2g[eN_nDOF_trial_element],&vel_trial_ref[k*nDOF_trial_element],un);
                ck.valFromDOF(v_dof_old.data(),&vel_l2g[eN_nDOF_trial_element],&vel_trial_ref[k*nDOF_trial_element],vn);
                ck.valFromDOF(w_dof_old.data(),&vel_l2g[eN_nDOF_trial_element],&vel_trial_ref[k*nDOF_trial_element],wn);
                //get the solution gradients
                /* ck.gradFromDOF(p_dof,&p_l2g[eN_nDOF_trial_element],p_grad_trial,grad_p); */
                for (int I=0;I<nSpace;I++)
                  grad_p[I] = q_grad_p[eN_k_nSpace + I];
                ck.gradFromDOF(u_dof.data(),&vel_l2g[eN_nDOF_trial_element],vel_grad_trial,grad_u);
                ck.gradFromDOF(v_dof.data(),&vel_l2g[eN_nDOF_trial_element],vel_grad_trial,grad_v);
                ck.gradFromDOF(w_dof.data(),&vel_l2g[eN_nDOF_trial_element],vel_grad_trial,grad_w);
                ck.hessFromDOF(u_dof.data(),&vel_l2g[eN_nDOF_trial_element],vel_hess_trial,hess_u);
                ck.hessFromDOF(v_dof.data(),&vel_l2g[eN_nDOF_trial_element],vel_hess_trial,hess_v);
                ck.hessFromDOF(w_dof.data(),&vel_l2g[eN_nDOF_trial_element],vel_hess_trial,hess_w);
                ck.hessFromDOF(uStar_dof.data(),&vel_l2g[eN_nDOF_trial_element],vel_hess_trial,hess_uStar);
                ck.hessFromDOF(vStar_dof.data(),&vel_l2g[eN_nDOF_trial_element],vel_hess_trial,hess_vStar);
                ck.hessFromDOF(wStar_dof.data(),&vel_l2g[eN_nDOF_trial_element],vel_hess_trial,hess_wStar);
                //precalculate test function products with integration weights
                for (int j=0;j<nDOF_trial_element;j++)
                  {
                    /* p_test_dV[j] = p_test_ref[k*nDOF_trial_element+j]*dV; */
                    vel_test_dV[j] = vel_test_ref[k*nDOF_trial_element+j]*dV;
                    for (int I=0;I<nSpace;I++)
                      {
                        /* p_grad_test_dV[j*nSpace+I]   = p_grad_trial[j*nSpace+I]*dV;//cek warning won't work for Petrov-Galerkin */
                        vel_grad_test_dV[j*nSpace+I] = vel_grad_trial[j*nSpace+I]*dV;//cek warning won't work for Petrov-Galerkin
			if (ARTIFICIAL_VISCOSITY==4)
			  {
			    // mql: for entropy residual. grad(u*phi) and grad(v*phi)
			    u_times_vel_grad_test_dV[j*nSpace+I] =
			      u*vel_grad_trial[j*nSpace+I]*dV + vel_test_dV[j]*grad_u[I];
			    v_times_vel_grad_test_dV[j*nSpace+I] =
			      v*vel_grad_trial[j*nSpace+I]*dV + vel_test_dV[j]*grad_v[I];
			    w_times_vel_grad_test_dV[j*nSpace+I] =
			      w*vel_grad_trial[j*nSpace+I]*dV + vel_test_dV[j]*grad_w[I];
			  }
                      }
                  }
		// compute determinant of Hessians
		if (ARTIFICIAL_VISCOSITY==3)
		  {
		    det_hess_uStar_Ke +=
		      (hess_uStar[0]*(hess_uStar[4]*hess_uStar[8] - hess_uStar[7]*hess_uStar[5])
		       -hess_uStar[1]*(hess_uStar[3]*hess_uStar[8] - hess_uStar[6]*hess_uStar[5])
		       +hess_uStar[2]*(hess_uStar[3]*hess_uStar[7] - hess_uStar[6]*hess_uStar[4]))*dV;
		    det_hess_vStar_Ke +=
		      (hess_vStar[0]*(hess_vStar[4]*hess_vStar[8] - hess_vStar[7]*hess_vStar[5])
		       -hess_vStar[1]*(hess_vStar[3]*hess_vStar[8] - hess_vStar[6]*hess_vStar[5])
		       +hess_vStar[2]*(hess_vStar[3]*hess_vStar[7] - hess_vStar[6]*hess_vStar[4]))*dV;
		    det_hess_wStar_Ke +=
		      (hess_wStar[0]*(hess_wStar[4]*hess_wStar[8] - hess_wStar[7]*hess_wStar[5])
		       -hess_wStar[1]*(hess_wStar[3]*hess_wStar[8] - hess_wStar[6]*hess_wStar[5])
		       +hess_wStar[2]*(hess_wStar[3]*hess_wStar[7] - hess_wStar[6]*hess_wStar[4]))*dV;
		    area_Ke += dV;
		  }
                //cek hack
                double div_mesh_velocity=0.0;
                int NDOF_MESH_TRIAL_ELEMENT=4;
                for (int j=0;j<NDOF_MESH_TRIAL_ELEMENT;j++)
                  {
                    int eN_j=eN*NDOF_MESH_TRIAL_ELEMENT+j;
                    div_mesh_velocity +=
                      mesh_velocity_dof[mesh_l2g[eN_j]*3+0]*vel_grad_trial[j*nSpace+0] +
                      mesh_velocity_dof[mesh_l2g[eN_j]*3+1]*vel_grad_trial[j*nSpace+1] +
                      mesh_velocity_dof[mesh_l2g[eN_j]*3+2]*vel_grad_trial[j*nSpace+2];
                  }
                mesh_volume_conservation_element += (alphaBDF*(dV-q_dV_last[eN_k])/dV - div_mesh_velocity)*dV;
                div_mesh_velocity = DM3*div_mesh_velocity + (1.0-DM3)*alphaBDF*(dV-q_dV_last[eN_k])/dV;
                //VRANS
                porosity      = 1.0 - q_vos[eN_k];
                //meanGrainSize = q_meanGrain[eN_k];
                //
                q_x[eN_k_3d+0]=x;
                q_x[eN_k_3d+1]=y;
                q_x[eN_k_3d+2]=z;
                //
                //calculate pde coefficients at quadrature points
                //
                double distance_to_omega_solid = 1e10;
                if(use_ball_as_particle==1)
                {
                    get_distance_to_ball(nParticles, ball_center.data(), ball_radius.data(),
                                         x,y,z,
                                         distance_to_omega_solid);
                }
                else
                {
                  for (int i = 0; i < nParticles; i++)
                  {
                    double distance_to_i_th_solid = particle_signed_distances[i * nElements_global * nQuadraturePoints_element + eN_k];
                    distance_to_omega_solid = (distance_to_i_th_solid < distance_to_omega_solid)?distance_to_i_th_solid:distance_to_omega_solid;
                  }
                }
                phi_solid[eN_k] = distance_to_omega_solid;//save it
                //
                //calculate pde coefficients at quadrature points
                //
                evaluateCoefficients(eps_rho,
                                     eps_mu,
                                     particle_eps,
                                     sigma,
                                     rho_0,
                                     nu_0,
                                     rho_1,
                                     nu_1,
                                     elementDiameter[eN],
                                     smagorinskyConstant,
                                     turbulenceClosureModel,
                                     g.data(),
                                     useVF,
                                     vf[eN_k],
                                     phi[eN_k],
                                     &normal_phi[eN_k_nSpace],
                                     distance_to_omega_solid,
                                     kappa_phi[eN_k],
                                     //VRANS
                                     porosity,
                                     //
                                     p,
                                     grad_p,
                                     grad_u,
                                     grad_v,
                                     grad_w,
                                     u,
                                     v,
                                     w,
                                     q_velocity_sge[eN_k_nSpace+0],
                                     q_velocity_sge[eN_k_nSpace+1],
                                     q_velocity_sge[eN_k_nSpace+2],
                                     q_eddy_viscosity[eN_k],
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
                                     dmom_w_ham_grad_w,
                                     q_rho[eN_k],
                                     q_nu[eN_k],
                                     KILL_PRESSURE_TERM,
                                     MULTIPLY_EXTERNAL_FORCE_BY_DENSITY,
                                     forcex[eN_k],
                                     forcey[eN_k],
                                     forcez[eN_k],
                                     MATERIAL_PARAMETERS_AS_FUNCTION,
                                     density_as_function[eN_k],
                                     dynamic_viscosity_as_function[eN_k],
                                     USE_SBM,
                                     x,y,z,
                                     use_ball_as_particle,
                                     ball_center.data(),
                                     ball_radius.data(),
                                     ball_velocity.data(),
                                     ball_angular_velocity.data(),
				     INT_BY_PARTS_PRESSURE);

                //VRANS
                mass_source = q_mass_source[eN_k];
                for (int I=0;I<nSpace;I++)
                  {
                    dmom_u_source[I] = 0.0;
                    dmom_v_source[I] = 0.0;
                    dmom_w_source[I] = 0.0;
                  }
                updateDarcyForchheimerTerms_Ergun(
                                                  q_dragAlpha[eN_k],
                                                  q_dragBeta[eN_k],
                                                  eps_rho,
                                                  eps_mu,
                                                  rho_0,
                                                  nu_0,
                                                  rho_1,
                                                  nu_1,
						  q_eddy_viscosity[eN_k],
                                                  useVF,
                                                  vf[eN_k],
                                                  phi[eN_k],
                                                  u,
                                                  v,
                                                  w,
                                                  q_velocity_sge[eN_k_nSpace+0],
                                                  q_velocity_sge[eN_k_nSpace+1],
                                                  q_velocity_sge[eN_k_nSpace+2],
                                                  eps_solid[elementFlags[eN]],
                                                  porosity,
                                                  q_velocity_solid[eN_k_nSpace+0],
                                                  q_velocity_solid[eN_k_nSpace+1],
                                                  q_velocity_solid[eN_k_nSpace+2],
                                                  q_velocityStar_solid[eN_k_nSpace+0],
                                                  q_velocityStar_solid[eN_k_nSpace+1],
                                                  q_velocityStar_solid[eN_k_nSpace+2],
                                                  mom_u_source,
                                                  mom_v_source,
                                                  mom_w_source,
                                                  dmom_u_source,
                                                  dmom_v_source,
                                                  dmom_w_source,
                                                  q_grad_vos[eN_k_nSpace+0],
                                                  q_grad_vos[eN_k_nSpace+1],
                                                  q_grad_vos[eN_k_nSpace+2]);
                double C_particles = 0.0;
                if (nParticles > 0 && USE_SBM==0)
                  updateSolidParticleTerms(eN < nElements_owned,
                                           particle_nitsche,
                                           dV,
                                           nParticles,
                                           nQuadraturePoints_global,
                                           &particle_signed_distances[eN_k],
                                           &particle_signed_distance_normals[eN_k_nSpace],
                                           &particle_velocities[eN_k_nSpace],
                                           particle_centroids.data(),
                                           use_ball_as_particle,
                                           ball_center.data(),
                                           ball_radius.data(),
                                           ball_velocity.data(),
                                           ball_angular_velocity.data(),
                                           porosity,
                                           particle_penalty_constant/h_phi,
                                           particle_alpha/h_phi,
                                           particle_beta/h_phi,
                                           eps_rho,
                                           eps_mu,
                                           rho_0,
                                           nu_0,
                                           rho_1,
                                           nu_1,
                                           useVF,
                                           vf[eN_k],
                                           phi[eN_k],
                                           x,
                                           y,
                                           z,
                                           p,
                                           u,
                                           v,
                                           w,
                                           q_velocity_sge[eN_k_nSpace + 0],
                                           q_velocity_sge[eN_k_nSpace + 1],
                                           q_velocity_sge[eN_k_nSpace + 2],
                                           particle_eps,
                                           grad_u,
                                           grad_v,
                                           grad_w,
                                           mom_u_source,
                                           mom_v_source,
                                           mom_w_source,
                                           dmom_u_source,
                                           dmom_v_source,
                                           dmom_w_source,
                                           mom_u_adv,
                                           mom_v_adv,
                                           mom_w_adv,
                                           dmom_u_adv_u,
                                           dmom_v_adv_v,
                                           dmom_w_adv_w,
                                           mom_u_ham,
                                           dmom_u_ham_grad_u,
                                           mom_v_ham,
                                           dmom_v_ham_grad_v,
                                           mom_w_ham,
                                           dmom_w_ham_grad_w,
                                           particle_netForces.data(),
                                           particle_netMoments.data(),
                                           particle_surfaceArea.data());
                //Turbulence closure model
                if (turbulenceClosureModel >= 3)
                  {
                    const double c_mu = 0.09;//mwf hack
                    updateTurbulenceClosure(turbulenceClosureModel,
                                            eps_rho,
                                            eps_mu,
                                            rho_0,
                                            nu_0,
                                            rho_1,
                                            nu_1,
                                            useVF,
                                            vf[eN_k],
                                            phi[eN_k],
                                            porosity,
                                            c_mu, //mwf hack
                                            q_turb_var_0[eN_k],
                                            q_turb_var_1[eN_k],
                                            &q_turb_var_grad_0[eN_k_nSpace],
                                            q_eddy_viscosity[eN_k],
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
                                            mom_w_source);

                  }
                //
                //save momentum for time history and velocity for subgrid error
                //
                q_mom_u_acc[eN_k] = mom_u_acc;
                q_mom_v_acc[eN_k] = mom_v_acc;
                q_mom_w_acc[eN_k] = mom_w_acc;
                //subgrid error uses grid scale velocity
                q_mass_adv[eN_k_nSpace+0] = u;
                q_mass_adv[eN_k_nSpace+1] = v;
                q_mass_adv[eN_k_nSpace+2] = w;
                //
                //moving mesh
                //
                mom_u_adv[0] -= MOVING_DOMAIN*dmom_u_acc_u*mom_u_acc*xt; // multiply by rho*porosity. mql. CHECK.
                mom_u_adv[1] -= MOVING_DOMAIN*dmom_u_acc_u*mom_u_acc*yt;
                mom_u_adv[2] -= MOVING_DOMAIN*dmom_u_acc_u*mom_u_acc*zt;
                dmom_u_adv_u[0] -= MOVING_DOMAIN*dmom_u_acc_u*xt;
                dmom_u_adv_u[1] -= MOVING_DOMAIN*dmom_u_acc_u*yt;
                dmom_u_adv_u[2] -= MOVING_DOMAIN*dmom_u_acc_u*zt;

                mom_v_adv[0] -= MOVING_DOMAIN*dmom_v_acc_v*mom_v_acc*xt;
                mom_v_adv[1] -= MOVING_DOMAIN*dmom_v_acc_v*mom_v_acc*yt;
                mom_v_adv[2] -= MOVING_DOMAIN*dmom_v_acc_v*mom_v_acc*zt;
                dmom_v_adv_v[0] -= MOVING_DOMAIN*dmom_v_acc_v*xt;
                dmom_v_adv_v[1] -= MOVING_DOMAIN*dmom_v_acc_v*yt;
                dmom_v_adv_v[2] -= MOVING_DOMAIN*dmom_v_acc_v*zt;

                mom_w_adv[0] -= MOVING_DOMAIN*dmom_w_acc_w*mom_w_acc*xt;
                mom_w_adv[1] -= MOVING_DOMAIN*dmom_w_acc_w*mom_w_acc*yt;
                mom_w_adv[2] -= MOVING_DOMAIN*dmom_w_acc_w*mom_w_acc*zt;
                dmom_w_adv_w[0] -= MOVING_DOMAIN*dmom_w_acc_w*xt;
                dmom_w_adv_w[1] -= MOVING_DOMAIN*dmom_w_acc_w*yt;
                dmom_w_adv_w[2] -= MOVING_DOMAIN*dmom_w_acc_w*zt;

                //
                //calculate time derivative at quadrature points
                //
                if (q_dV_last[eN_k] <= -100)
                  q_dV_last[eN_k] = dV;
                q_dV[eN_k] = dV;
                ck.bdf(alphaBDF,
                       q_mom_u_acc_beta_bdf[eN_k]*q_dV_last[eN_k]/dV,
                       mom_u_acc,
                       dmom_u_acc_u,
                       mom_u_acc_t,
                       dmom_u_acc_u_t);
                ck.bdf(alphaBDF,
                       q_mom_v_acc_beta_bdf[eN_k]*q_dV_last[eN_k]/dV,
                       mom_v_acc,
                       dmom_v_acc_v,
                       mom_v_acc_t,
                       dmom_v_acc_v_t);
                ck.bdf(alphaBDF,
                       q_mom_w_acc_beta_bdf[eN_k]*q_dV_last[eN_k]/dV,
                       mom_w_acc,
                       dmom_w_acc_w,
                       mom_w_acc_t,
                       dmom_w_acc_w_t);

                mom_u_acc_t *= dmom_u_acc_u; //multiply by rho*porosity. mql. CHECK.
                mom_v_acc_t *= dmom_v_acc_v;
                mom_w_acc_t *= dmom_w_acc_w;


                //
                //calculate subgrid error (strong residual and adjoint)
                //
                //calculate strong residual
                pdeResidual_p =
                  ck.Mass_strong(-q_dvos_dt[eN_k]) + // mql. CHECK.
                  ck.Advection_strong(dmass_adv_u,grad_u) +
                  ck.Advection_strong(dmass_adv_v,grad_v) +
                  ck.Advection_strong(dmass_adv_w,grad_w) +
                  DM2*MOVING_DOMAIN*ck.Reaction_strong(alphaBDF*(dV-q_dV_last[eN_k])/dV - div_mesh_velocity) +
                  //VRANS
                  ck.Reaction_strong(mass_source);
                //

                dmom_adv_sge[0] = dmom_u_acc_u*(q_velocity_sge[eN_k_nSpace+0] - MOVING_DOMAIN*xt);
                dmom_adv_sge[1] = dmom_u_acc_u*(q_velocity_sge[eN_k_nSpace+1] - MOVING_DOMAIN*yt);
                dmom_adv_sge[2] = dmom_u_acc_u*(q_velocity_sge[eN_k_nSpace+2] - MOVING_DOMAIN*zt);

                pdeResidual_u =
                  ck.Mass_strong(mom_u_acc_t) + // mql. CHECK.
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

                pdeResidual_w =
                  ck.Mass_strong(mom_w_acc_t) +
                  ck.Advection_strong(dmom_adv_sge,grad_w) +
                  ck.Hamiltonian_strong(dmom_w_ham_grad_p,grad_p) +
                  ck.Reaction_strong(mom_w_source) -
                  ck.Reaction_strong(w*div_mesh_velocity);

                //calculate tau and tau*Res
                //cek debug
                double tmpR=dmom_u_acc_u_t + dmom_u_source[0];
                calculateSubgridError_tau(hFactor,
                                          elementDiameter[eN],
                                          tmpR,//dmom_u_acc_u_t,
                                          dmom_u_acc_u,
                                          dmom_adv_sge,
                                          mom_uu_diff_ten[1],
                                          dmom_u_ham_grad_p[0],
                                          tau_v0,
                                          tau_p0,
                                          q_cfl[eN_k]);

                calculateSubgridError_tau(Ct_sge,Cd_sge,
                                          G,G_dd_G,tr_G,
                                          tmpR,//dmom_u_acc_u_t,
                                          dmom_adv_sge,
                                          mom_uu_diff_ten[1],
                                          dmom_u_ham_grad_p[0],
                                          tau_v1,
                                          tau_p1,
                                          q_cfl[eN_k]);

                tau_v = useMetrics*tau_v1+(1.0-useMetrics)*tau_v0;
                tau_p = KILL_PRESSURE_TERM == 1 ? 0. : PSTAB*(useMetrics*tau_p1+(1.0-useMetrics)*tau_p0);

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
                dmom_adv_star[0] = dmom_u_acc_u*(q_velocity_sge[eN_k_nSpace+0] - MOVING_DOMAIN*xt + useRBLES*subgridError_u);
                dmom_adv_star[1] = dmom_u_acc_u*(q_velocity_sge[eN_k_nSpace+1] - MOVING_DOMAIN*yt + useRBLES*subgridError_v);
                dmom_adv_star[2] = dmom_u_acc_u*(q_velocity_sge[eN_k_nSpace+2] - MOVING_DOMAIN*zt + useRBLES*subgridError_w);

                mom_u_adv[0] += dmom_u_acc_u*(useRBLES*subgridError_u*q_velocity_sge[eN_k_nSpace+0]);
                mom_u_adv[1] += dmom_u_acc_u*(useRBLES*subgridError_v*q_velocity_sge[eN_k_nSpace+1]);
                mom_u_adv[2] += dmom_u_acc_u*(useRBLES*subgridError_w*q_velocity_sge[eN_k_nSpace+2]);

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
                    Lstar_w_w[i]=ck.Advection_adjoint(dmom_adv_star,&vel_grad_test_dV[i_nSpace]);
                    Lstar_p_u[i]=ck.Hamiltonian_adjoint(dmom_u_ham_grad_p,&vel_grad_test_dV[i_nSpace]);
                    Lstar_p_v[i]=ck.Hamiltonian_adjoint(dmom_v_ham_grad_p,&vel_grad_test_dV[i_nSpace]);
                    Lstar_p_w[i]=ck.Hamiltonian_adjoint(dmom_w_ham_grad_p,&vel_grad_test_dV[i_nSpace]);

                    //VRANS account for drag terms, diagonal only here ... decide if need off diagonal terms too
                    Lstar_u_u[i]+=ck.Reaction_adjoint(dmom_u_source[0],vel_test_dV[i]);
                    Lstar_v_v[i]+=ck.Reaction_adjoint(dmom_v_source[1],vel_test_dV[i]);
                    Lstar_w_w[i]+=ck.Reaction_adjoint(dmom_w_source[2],vel_test_dV[i]);
                    //
                  }

		if (ARTIFICIAL_VISCOSITY==0 || ARTIFICIAL_VISCOSITY==3 || ARTIFICIAL_VISCOSITY==4)
		  {
		    q_numDiff_u[eN_k] = 0;
		    q_numDiff_v[eN_k] = 0;
		    q_numDiff_w[eN_k] = 0;
		  }
		else if (ARTIFICIAL_VISCOSITY==1) // SHOCK CAPTURING
		  {
		    norm_Rv = sqrt(pdeResidual_u*pdeResidual_u + pdeResidual_v*pdeResidual_v + pdeResidual_w*pdeResidual_w);
		    q_numDiff_u[eN_k] = C_dc*norm_Rv*(useMetrics/sqrt(G_dd_G+1.0e-12)  +
						      (1.0-useMetrics)*hFactor*hFactor*elementDiameter[eN]*elementDiameter[eN]);
		    q_numDiff_v[eN_k] = q_numDiff_u[eN_k];
		    q_numDiff_w[eN_k] = q_numDiff_u[eN_k];
		  }
		else // ARTIFICIAL_VISCOSITY==2; i.e, ENTROPY VISCOSITY
		  {
		    double rho = q_rho[eN_k];
		    double mu = q_rho[eN_k]*q_nu[eN_k];

		    double vel2 = u*u + v*v + w*w;

		    // entropy residual
		    double Res_in_x =
                      porosity*rho*((u-un)/dt + (u*grad_u[0]+v*grad_u[1]+w*grad_u[2]) - g[0])
		      + (KILL_PRESSURE_TERM == 1 ? 0 : 1.)*grad_p[0]
		      - (MULTIPLY_EXTERNAL_FORCE_BY_DENSITY == 1 ? porosity*rho : 1.0)*forcex[eN_k]
                      - mu*(hess_u[0] + hess_u[4] + hess_u[8]) //  u_xx + u_yy + u_zz
                      - mu*(hess_u[0] + hess_v[1] + hess_w[2]); // u_xx + v_xy + w_xz
                    double Res_in_y =
                      porosity*rho*((v-vn)/dt + (u*grad_v[0]+v*grad_v[1]+w*grad_v[2]) - g[1])
		      + (KILL_PRESSURE_TERM == 1 ? 0 : 1.)*grad_p[1]
		      - (MULTIPLY_EXTERNAL_FORCE_BY_DENSITY == 1 ? porosity*rho : 1.0)*forcey[eN_k]
                      - mu*(hess_v[0] + hess_v[4] + hess_v[8])  // v_xx + v_yy + v_zz
                      - mu*(hess_u[1] + hess_v[4] + hess_w[5]); // u_xy + v_yy + w_yz
                    double Res_in_z =
                      porosity*rho*((w-wn)/dt + (u*grad_w[0]+v*grad_w[1]+w*grad_w[2]) - g[2])
		      + (KILL_PRESSURE_TERM == 1 ? 0 : 1.)*grad_p[2]
		      - (MULTIPLY_EXTERNAL_FORCE_BY_DENSITY == 1 ? porosity*rho : 1.0)*forcez[eN_k]
                      - mu*(hess_w[0] + hess_w[4] + hess_w[8])  // w_xx + w_yy + w_zz
                      - mu*(hess_u[2] + hess_v[5] + hess_w[8]); // u_xz + v_yz + w_zz

		    // compute entropy residual
		    double entRes_times_u = Res_in_x*u + Res_in_y*v + Res_in_z*w;

		    double hK = elementDiameter[eN]/order_polynomial;
		    q_numDiff_u[eN_k] = fmin(cMax*porosity*rho*hK*std::sqrt(vel2),
					     cE*hK*hK*fabs(entRes_times_u)/(vel2+1E-10));
		    q_numDiff_v[eN_k] = q_numDiff_u[eN_k];
		    q_numDiff_w[eN_k] = q_numDiff_u[eN_k];

		    if (CELL_BASED_EV_COEFF)
		      {
			linVisc_eN  = fmax(porosity*rho*std::sqrt(vel2),linVisc_eN);
			nlinVisc_eN_num = fmax(fabs(entRes_times_u),nlinVisc_eN_num);
			nlinVisc_eN_den = fmax(vel2,nlinVisc_eN_den);
		      }
		  }

		//
                //update element residual
                //
                double mesh_vel[3];
                mesh_vel[0] = xt;
                mesh_vel[1] = yt;
                mesh_vel[2] = zt;
                q_velocity[eN_k_nSpace+0]=u;
                q_velocity[eN_k_nSpace+1]=v;
                q_velocity[eN_k_nSpace+2]=w;
                for (int I=0;I<nSpace;I++)
                  {
                    q_grad_u[eN_k_nSpace+I] = grad_u[I];
                    q_grad_v[eN_k_nSpace+I] = grad_v[I];
                    q_grad_w[eN_k_nSpace+I] = grad_w[I];
                  }
                // save divergence of velocity
                q_divU[eN_k] = q_grad_u[eN_k_nSpace+0] + q_grad_v[eN_k_nSpace+1] + q_grad_w[eN_k_nSpace+2];

		// SURFACE TENSION //
                double unit_normal[nSpace];
                double norm_grad_phi = 0.;
                for (int I=0;I<nSpace;I++)
                  norm_grad_phi += normal_phi[eN_k_nSpace+I]*normal_phi[eN_k_nSpace+I];
                norm_grad_phi = std::sqrt(norm_grad_phi) + 1E-10;
                for (int I=0;I<nSpace;I++)
                  unit_normal[I] = normal_phi[eN_k_nSpace+I]/norm_grad_phi;
                // compute auxiliary vectors for explicit term of 2D surf tension
                // v1 = [1-nx^2 -nx*ny -nx*ny]^T
                double v1[nSpace];
                v1[0]=1.-unit_normal[0]*unit_normal[0];
                v1[1]=-unit_normal[0]*unit_normal[1];
                v1[2]=-unit_normal[0]*unit_normal[2];
                // v2 = [-nx*ny 1-ny^2 -ny*nz]^T
                double v2[nSpace];
                v2[0]=-unit_normal[0]*unit_normal[1];
                v2[1]=1.-unit_normal[1]*unit_normal[1];
                v2[2]=-unit_normal[1]*unit_normal[2];
                // v3 = [-nx*nz -ny*nz 1-nz^2]^T
                double v3[nSpace];
                v3[0]=-unit_normal[0]*unit_normal[2];
                v3[1]=-unit_normal[1]*unit_normal[2];
                v3[2]=1.-unit_normal[2]*unit_normal[2];
                double delta = gf.D(eps_mu,phi[eN_k]); //use eps_rho instead?
                double vel_tgrad_test_i[nSpace],
                  tgrad_u[nSpace], tgrad_v[nSpace], tgrad_w[nSpace];
                calculateTangentialGradient(unit_normal,
                                            grad_u,
                                            tgrad_u);
                calculateTangentialGradient(unit_normal,
                                            grad_v,
                                            tgrad_v);
                calculateTangentialGradient(unit_normal,
                                            grad_w,
                                            tgrad_w);
                // END OF SURFACE TENSION //

		if (ARTIFICIAL_VISCOSITY==3 || ARTIFICIAL_VISCOSITY==4)
		  {
		    velStar[0] = q_velocity_sge[eN_k_nSpace+0];
		    velStar[1] = q_velocity_sge[eN_k_nSpace+1];
		    velStar[2] = q_velocity_sge[eN_k_nSpace+2];
		  }
                for(int i=0;i<nDOF_test_element;i++)
                  {
                    int i_nSpace=i*nSpace;
                    calculateTangentialGradient(unit_normal,
                                                &vel_grad_trial[i_nSpace],
                                                vel_tgrad_test_i);
                    phisErrorElement[i]+=std::abs(phisError[eN_k_nSpace+0])*p_test_dV[i];
                    /* std::cout<<"elemRes_mesh "<<mesh_vel[0]<<'\t'<<mesh_vel[2]<<'\t'<<p_test_dV[i]<<'\t'<<(q_dV_last[eN_k]/dV)<<'\t'<<dV<<std::endl; */
                    /* elementResidual_mesh[i] += ck.Reaction_weak(1.0,p_test_dV[i]) - */
                    /*   ck.Reaction_weak(1.0,p_test_dV[i]*q_dV_last[eN_k]/dV) - */
                    /*   ck.Advection_weak(mesh_vel,&p_grad_test_dV[i_nSpace]); */

                    /* elementResidual_p[i] += ck.Mass_weak(-q_dvos_dt[eN_k],p_test_dV[i]) + */
                    /*   ck.Advection_weak(mass_adv,&p_grad_test_dV[i_nSpace]) + */
                    /*   DM*MOVING_DOMAIN*(ck.Reaction_weak(alphaBDF*1.0,p_test_dV[i]) - */
                    /*                     ck.Reaction_weak(alphaBDF*1.0,p_test_dV[i]*q_dV_last[eN_k]/dV) - */
                    /*                     ck.Advection_weak(mesh_vel,&p_grad_test_dV[i_nSpace])) + */
                    /*   //VRANS */
                    /*   ck.Reaction_weak(mass_source,p_test_dV[i])   + //VRANS source term for wave maker */
                    /*   // */
                    /*   ck.SubgridError(subgridError_u,Lstar_u_p[i]) +  */
                    /*   ck.SubgridError(subgridError_v,Lstar_v_p[i]);// +  */
                    /*   /\* ck.SubgridError(subgridError_w,Lstar_w_p[i]); *\/ */

                    elementResidual_u[i] += // mql. CHECK.
                      ck.Mass_weak(mom_u_acc_t,vel_test_dV[i]) +
                      ck.Advection_weak(mom_u_adv,&vel_grad_test_dV[i_nSpace]) +
                      ck.Diffusion_weak(sdInfo_u_u_rowptr.data(),sdInfo_u_u_colind.data(),mom_uu_diff_ten,grad_u,&vel_grad_test_dV[i_nSpace]) +
                      ck.Diffusion_weak(sdInfo_u_v_rowptr.data(),sdInfo_u_v_colind.data(),mom_uv_diff_ten,grad_v,&vel_grad_test_dV[i_nSpace]) +
                      ck.Diffusion_weak(sdInfo_u_w_rowptr.data(),sdInfo_u_w_colind.data(),mom_uw_diff_ten,grad_w,&vel_grad_test_dV[i_nSpace]) +
                      ck.Reaction_weak(mom_u_source,vel_test_dV[i]) +
                      ck.Hamiltonian_weak(mom_u_ham,vel_test_dV[i]) +
		      (INT_BY_PARTS_PRESSURE==1 ? -1.0*p*vel_grad_test_dV[i_nSpace+0] : 0.) +
                      //ck.SubgridError(subgridError_p,Lstar_p_u[i]) +
                      USE_SUPG*ck.SubgridError(subgridError_u,Lstar_u_u[i]) +
                      ck.NumericalDiffusion(q_numDiff_u_last[eN_k],grad_u,&vel_grad_test_dV[i_nSpace]) +
                      //surface tension
                      ck.NumericalDiffusion(delta*sigma*dV,v1,vel_tgrad_test_i) +  //exp.
                      ck.NumericalDiffusion(dt*delta*sigma*dV,tgrad_u,vel_tgrad_test_i); //imp.
                    mom_u_source_i[i] += ck.Reaction_weak(mom_u_source,vel_test_dV[i]);
                    betaDrag_i[i] += ck.Reaction_weak(dmom_u_source[0],
                                                      vel_test_dV[i]);
                    vos_i[i] += ck.Reaction_weak(1.0-porosity,
                                                 vel_test_dV[i]);

                    elementResidual_v[i] +=
                      ck.Mass_weak(mom_v_acc_t,vel_test_dV[i]) +
                      ck.Advection_weak(mom_v_adv,&vel_grad_test_dV[i_nSpace]) +
                      ck.Diffusion_weak(sdInfo_v_u_rowptr.data(),sdInfo_v_u_colind.data(),mom_vu_diff_ten,grad_u,&vel_grad_test_dV[i_nSpace]) +
                      ck.Diffusion_weak(sdInfo_v_v_rowptr.data(),sdInfo_v_v_colind.data(),mom_vv_diff_ten,grad_v,&vel_grad_test_dV[i_nSpace]) +
                      ck.Diffusion_weak(sdInfo_v_w_rowptr.data(),sdInfo_v_w_colind.data(),mom_vw_diff_ten,grad_w,&vel_grad_test_dV[i_nSpace]) +
                      ck.Reaction_weak(mom_v_source,vel_test_dV[i]) +
                      ck.Hamiltonian_weak(mom_v_ham,vel_test_dV[i]) +
		      (INT_BY_PARTS_PRESSURE==1 ? -1.0*p*vel_grad_test_dV[i_nSpace+1] : 0.) +
                      //ck.SubgridError(subgridError_p,Lstar_p_v[i]) +
                      USE_SUPG*ck.SubgridError(subgridError_v,Lstar_v_v[i]) +
                      ck.NumericalDiffusion(q_numDiff_v_last[eN_k],grad_v,&vel_grad_test_dV[i_nSpace]) +
                      //surface tension
                      ck.NumericalDiffusion(delta*sigma*dV,v2,vel_tgrad_test_i) +  //exp.
                      ck.NumericalDiffusion(dt*delta*sigma*dV,tgrad_v,vel_tgrad_test_i); //imp.
                    mom_v_source_i[i] += ck.Reaction_weak(mom_v_source,vel_test_dV[i]);

                    elementResidual_w[i] +=
                      ck.Mass_weak(mom_w_acc_t,vel_test_dV[i]) +
                      ck.Advection_weak(mom_w_adv,&vel_grad_test_dV[i_nSpace]) +
                      ck.Diffusion_weak(sdInfo_w_u_rowptr.data(),sdInfo_w_u_colind.data(),mom_wu_diff_ten,grad_u,&vel_grad_test_dV[i_nSpace]) +
                      ck.Diffusion_weak(sdInfo_w_v_rowptr.data(),sdInfo_w_v_colind.data(),mom_wv_diff_ten,grad_v,&vel_grad_test_dV[i_nSpace]) +
                      ck.Diffusion_weak(sdInfo_w_w_rowptr.data(),sdInfo_w_w_colind.data(),mom_ww_diff_ten,grad_w,&vel_grad_test_dV[i_nSpace]) +
                      ck.Reaction_weak(mom_w_source,vel_test_dV[i]) +
                      ck.Hamiltonian_weak(mom_w_ham,vel_test_dV[i]) +
		      (INT_BY_PARTS_PRESSURE==1 ? -1.0*p*vel_grad_test_dV[i_nSpace+2] : 0.) +
                      //ck.SubgridError(subgridError_p,Lstar_p_w[i]) +
                      USE_SUPG*ck.SubgridError(subgridError_w,Lstar_w_w[i]) +
                      ck.NumericalDiffusion(q_numDiff_w_last[eN_k],grad_w,&vel_grad_test_dV[i_nSpace]) +
                      //surface tension
                      ck.NumericalDiffusion(delta*sigma*dV,v3,vel_tgrad_test_i) +  //exp.
                      ck.NumericalDiffusion(dt*delta*sigma*dV,tgrad_w,vel_tgrad_test_i); //imp.
                    mom_w_source_i[i] += ck.Reaction_weak(mom_w_source,vel_test_dV[i]);

		    if (ARTIFICIAL_VISCOSITY==4)
		      {
			// ***** COMPUTE ENTROPY RESIDUAL ***** //
			// mql. NOTE that the test functions are weighted by the velocity
			elementEntropyResidual[i] +=
			  // x-component
			  ck.Mass_weak(mom_u_acc_t,u*vel_test_dV[i]) + // time derivative
			  ck.Advection_weak(mom_u_adv,&u_times_vel_grad_test_dV[i_nSpace])+//m.mesh
			  ck.Diffusion_weak(sdInfo_u_u_rowptr.data(),
					    sdInfo_u_u_colind.data(),
					    mom_uu_diff_ten,
					    grad_u,
					    &u_times_vel_grad_test_dV[i_nSpace]) +
			  ck.Diffusion_weak(sdInfo_u_v_rowptr.data(),
					    sdInfo_u_v_colind.data(),
					    mom_uv_diff_ten,
					    grad_v,
					    &u_times_vel_grad_test_dV[i_nSpace]) +
			  ck.Diffusion_weak(sdInfo_u_w_rowptr.data(),
					    sdInfo_u_w_colind.data(),
					    mom_uw_diff_ten,
					    grad_w,
					    &u_times_vel_grad_test_dV[i_nSpace]) +
			  ck.Reaction_weak(mom_u_source,u*vel_test_dV[i]) + // Force term
			  ck.Hamiltonian_weak(mom_u_ham,u*vel_test_dV[i])  // Pres + Non-linearity
			  + // y-component
			  ck.Mass_weak(mom_v_acc_t,v*vel_test_dV[i]) + // time derivative
			  ck.Advection_weak(mom_v_adv,&v_times_vel_grad_test_dV[i_nSpace])+//m.mesh
			  ck.Diffusion_weak(sdInfo_v_u_rowptr.data(),
					    sdInfo_v_u_colind.data(),
					    mom_vu_diff_ten,
					    grad_u,
					    &v_times_vel_grad_test_dV[i_nSpace])+
			  ck.Diffusion_weak(sdInfo_v_v_rowptr.data(),
					    sdInfo_v_v_colind.data(),
					    mom_vv_diff_ten,
					    grad_v,
					    &v_times_vel_grad_test_dV[i_nSpace])+
			  ck.Diffusion_weak(sdInfo_v_w_rowptr.data(),
					    sdInfo_v_w_colind.data(),
					    mom_vw_diff_ten,
					    grad_w,
					    &v_times_vel_grad_test_dV[i_nSpace]) +
			  ck.Reaction_weak(mom_v_source,v*vel_test_dV[i]) + // force term
			  ck.Hamiltonian_weak(mom_v_ham,v*vel_test_dV[i]) // Pres + Non-linearity
			  + // z-component
			  ck.Mass_weak(mom_w_acc_t,w*vel_test_dV[i]) + // time derivative
			  ck.Advection_weak(mom_w_adv,&w_times_vel_grad_test_dV[i_nSpace])+//m.mesh
			  ck.Diffusion_weak(sdInfo_w_u_rowptr.data(),
					    sdInfo_w_u_colind.data(),
					    mom_wu_diff_ten,
					    grad_u,
					    &w_times_vel_grad_test_dV[i_nSpace]) +
			  ck.Diffusion_weak(sdInfo_w_v_rowptr.data(),
					    sdInfo_w_v_colind.data(),
					    mom_wv_diff_ten,
					    grad_v,
					    &w_times_vel_grad_test_dV[i_nSpace]) +
			  ck.Diffusion_weak(sdInfo_w_w_rowptr.data(),
					    sdInfo_w_w_colind.data(),
					    mom_ww_diff_ten,
					    grad_w,
					    &w_times_vel_grad_test_dV[i_nSpace]) +
			  ck.Reaction_weak(mom_w_source,w*vel_test_dV[i]) + // force term
			  ck.Hamiltonian_weak(mom_w_ham,w*vel_test_dV[i]); // Pres + Non-linearity
		      }
		    if (ARTIFICIAL_VISCOSITY==3 || ARTIFICIAL_VISCOSITY==4)
		      {
			for(int j=0;j<nDOF_trial_element;j++)
			  {
			    int j_nSpace = j*nSpace;
			    int i_nSpace = i*nSpace;
			    elementTransport[i][j] += // int[rho*(velStar.grad_wj)*wi*dx]
			      q_rho[eN_k]*porosity*
			      ck.AdvectionJacobian_strong(velStar,
							  &vel_grad_test_dV[j_nSpace])
			      *vel_trial_ref[k*nDOF_trial_element+i];
			    elementTransposeTransport[i][j] += // int[rho*(velStar.grad_wi)*wj*dx]
			      q_rho[eN_k]*porosity*
			      ck.AdvectionJacobian_strong(velStar,
							  &vel_grad_test_dV[i_nSpace])
			      *vel_trial_ref[k*nDOF_trial_element+j];
			  }
		      }//j
		  }//i
              }
	    element_uStar_He[eN] = det_hess_uStar_Ke/area_Ke;
	    element_vStar_He[eN] = det_hess_vStar_Ke/area_Ke;
	    element_wStar_He[eN] = det_hess_wStar_Ke/area_Ke;

	    // End computation of cell based EV coeff //
	    if (CELL_BASED_EV_COEFF && ARTIFICIAL_VISCOSITY==2)
	      {
		double hK = elementDiameter[eN];
		double artVisc = fmin(cMax*hK*linVisc_eN,
				      cE*hK*hK*nlinVisc_eN_num/(nlinVisc_eN_den+1E-10));
		for(int k=0;k<nQuadraturePoints_element;k++)
		  {
		    int eN_k = eN*nQuadraturePoints_element+k;
		    q_numDiff_u[eN_k] = artVisc;
		    q_numDiff_v[eN_k] = artVisc;
		    q_numDiff_w[eN_k] = artVisc;
		  }
	      }
            //
            //load element into global residual and save element residual
            //
            for(int i=0;i<nDOF_test_element;i++)
	      {
                int eN_i=eN*nDOF_test_element+i;
                phisErrorNodal[vel_l2g[eN_i]]+= element_active*phisErrorElement[i];
                /* elementResidual_p_save[eN_i] +=  elementResidual_p[i]; */
                /* mesh_volume_conservation_element_weak += elementResidual_mesh[i]; */
                /* globalResidual[offset_p+stride_p*p_l2g[eN_i]]+=elementResidual_p[i]; */
                globalResidual[offset_u+stride_u*vel_l2g[eN_i]]+=element_active*elementResidual_u[i];
                globalResidual[offset_v+stride_v*vel_l2g[eN_i]]+=element_active*elementResidual_v[i];
                globalResidual[offset_w+stride_w*vel_l2g[eN_i]]+=element_active*elementResidual_w[i];
                ncDrag[offset_u+stride_u*vel_l2g[eN_i]]+=mom_u_source_i[i];
                ncDrag[offset_v+stride_v*vel_l2g[eN_i]]+=mom_v_source_i[i];
                ncDrag[offset_w+stride_w*vel_l2g[eN_i]]+=mom_w_source_i[i];
                betaDrag[vel_l2g[eN_i]] += betaDrag_i[i];
                vos_vel_nodes[vel_l2g[eN_i]] += vos_i[i];

		// compute numerator and denominator of uStar_hi and vStar_hi
		if (ARTIFICIAL_VISCOSITY==3)
		  {
		    uStar_hi[vel_l2g[eN_i]] += element_uStar_He[eN]; // offset=0, stride=1 since this is per component of the equation
		    vStar_hi[vel_l2g[eN_i]] += element_vStar_He[eN];
		    wStar_hi[vel_l2g[eN_i]] += element_wStar_He[eN];
		    den_hi[vel_l2g[eN_i]] += 1;
		  }
		if (ARTIFICIAL_VISCOSITY==4)
		  {
		    // DISTRIBUTE ENTROPY RESIDUAL //
		    entropyResidualPerNode[vel_l2g[eN_i]] += elementEntropyResidual[i];
		  }
		if (ARTIFICIAL_VISCOSITY==3 || ARTIFICIAL_VISCOSITY==4)
		  {
		    for (int j=0;j<nDOF_trial_element;j++)
		      {
			int eN_i_j = eN_i*nDOF_trial_element+j;
			TransportMatrix[csrRowIndeces_1D[eN_i]
					+ csrColumnOffsets_1D[eN_i_j]]
			  += elementTransport[i][j];
			// transpose
			TransposeTransportMatrix[csrRowIndeces_1D[eN_i]
						 + csrColumnOffsets_1D[eN_i_j]]
			  += elementTransposeTransport[i][j];
		      }//j
		  }
            }//i
            /* mesh_volume_conservation += mesh_volume_conservation_element; */
            /* mesh_volume_conservation_weak += mesh_volume_conservation_element_weak; */
            /* mesh_volume_conservation_err_max=fmax(mesh_volume_conservation_err_max,fabs(mesh_volume_conservation_element)); */
            /* mesh_volume_conservation_err_max_weak=fmax(mesh_volume_conservation_err_max_weak,fabs(mesh_volume_conservation_element_weak)); */
          }//elements

	// loop in DOFs for discrete upwinding
	if (ARTIFICIAL_VISCOSITY==3 || ARTIFICIAL_VISCOSITY==4)
	  {
	    // FIRST LOOP ON DOFs //
	    for (int i=0; i<numDOFs_1D; i++)
	      {
		if (ARTIFICIAL_VISCOSITY==4) // via entropy viscosity
		  {
		    // normalize entropy residual per node
		    double max_u2i = (std::pow(u_dof[i],2.) +
				      std::pow(v_dof[i],2.) +
				      std::pow(w_dof[i],2.));
		    double min_u2i = max_u2i;
		    for (int offset=rowptr_1D[i]; offset<rowptr_1D[i+1]; offset++)
		      {
			int j = colind_1D[offset];
			double u2j = (std::pow(u_dof[j],2.) +
				      std::pow(v_dof[j],2.) +
				      std::pow(w_dof[j],2.));
			max_u2i = fmax(max_u2i,u2j);
			min_u2i = fmin(min_u2i,u2j);
		      }
		    double normi = 0.5*(max_u2i + min_u2i) + 1E-10;
		    entropyResidualPerNode[i] = fabs(entropyResidualPerNode[i])/normi;
		  }
		else // via smoothness indicator
		  {
		    // The smoothness indicator psi of dof i is computed in Python from the lagged star
		    // velocities once per time step, on the dofs whose graph rows are complete, and scattered
		    // to the ghosts. Computing it here for a ghost dof would use a truncated row.
		    uStar_psi[i] = uStar_psi_dof[i];
		    vStar_psi[i] = vStar_psi_dof[i];
		    wStar_psi[i] = wStar_psi_dof[i];
		    // for computation of gamma
		    uStar_hi[i] /= den_hi[i];
		    vStar_hi[i] /= den_hi[i];
		    wStar_hi[i] /= den_hi[i];
		  }
	      }

	    if (ARTIFICIAL_VISCOSITY==3)
	      {
		for(int eN=0;eN<nElements_global;eN++)
		  {
		    double uStar_He = element_uStar_He[eN];
		    double vStar_He = element_vStar_He[eN];
		    double wStar_He = element_wStar_He[eN];
		    for(int i=0;i<nDOF_test_element;i++)
		      {
			int eN_i=eN*nDOF_test_element+i;
			int gi = vel_l2g[eN_i]; // offset=0, stride=1
			uStar_min_hiHe[gi] = fmin(uStar_min_hiHe[gi], uStar_hi[gi]*uStar_He);
			vStar_min_hiHe[gi] = fmin(vStar_min_hiHe[gi], vStar_hi[gi]*vStar_He);
			wStar_min_hiHe[gi] = fmin(wStar_min_hiHe[gi], wStar_hi[gi]*wStar_He);
		      }
		  }
	      }

	    // EXTRA LOOP ON DOFs to COMPUTE GAMMA INDICATOR//
	    if (ARTIFICIAL_VISCOSITY==3)
	      {
		for (int i=0; i<numDOFs_1D; i++)
		  {
		    // for gamma indicator
		    double uStar_hi2 = uStar_hi[i]*uStar_hi[i];
		    double vStar_hi2 = vStar_hi[i]*vStar_hi[i];
		    double wStar_hi2 = wStar_hi[i]*wStar_hi[i];
		    if (isBoundary_1D[i] == 1)
		      {
			uStar_gamma[i] = 1; //  set gamma=1 since at boundary we don't have enough information
			vStar_gamma[i] = 1;
			wStar_gamma[i] = 1;
		      }
		    else
		      {
			if (ANISOTROPIC_DIFFUSION==1)
			  {
			    uStar_gamma[i] = 1.-fmax(0, fmin(uStar_hi2, C_FOR_GAMMA_INDICATOR*uStar_min_hiHe[i]))/(uStar_hi2+EPS_FOR_GAMMA_INDICATOR);
			    vStar_gamma[i] = 1.-fmax(0, fmin(vStar_hi2, C_FOR_GAMMA_INDICATOR*vStar_min_hiHe[i]))/(vStar_hi2+EPS_FOR_GAMMA_INDICATOR);
			    wStar_gamma[i] = 1.-fmax(0, fmin(wStar_hi2, C_FOR_GAMMA_INDICATOR*wStar_min_hiHe[i]))/(wStar_hi2+EPS_FOR_GAMMA_INDICATOR);
			  }
			else // ISOTROPIC ARTIFICIAL DIFFUSION
			  {
			    double gamma=fmax(fmax(1.-fmax(0,fmin(uStar_hi2,C_FOR_GAMMA_INDICATOR*uStar_min_hiHe[i]))/(uStar_hi2+EPS_FOR_GAMMA_INDICATOR),
						   1.-fmax(0,fmin(vStar_hi2,C_FOR_GAMMA_INDICATOR*vStar_min_hiHe[i]))/(vStar_hi2+EPS_FOR_GAMMA_INDICATOR)),
					      1.-fmax(0, fmin(wStar_hi2, C_FOR_GAMMA_INDICATOR*wStar_min_hiHe[i]))/(wStar_hi2+EPS_FOR_GAMMA_INDICATOR));
			    uStar_gamma[i] = gamma;
			    vStar_gamma[i] = gamma;
			    wStar_gamma[i] = gamma;
			  }
		      }
		  }
	      }
	    // SECOND LOOP ON DOFs //
	    int ij=0;
	    for (int i=0; i<numDOFs_1D; i++)
	      {
		int ii;
		double uStar_dii = 0;
		double vStar_dii = 0;
		double wStar_dii = 0;
		double ui = u_dof[i];
		double vi = v_dof[i];
		double wi = w_dof[i];

		double ith_u_dissipative_term = 0;
		double ith_v_dissipative_term = 0;
		double ith_w_dissipative_term = 0;

		double uStar_alphai = USE_GAMMA_INDICATOR==1 ? fmin(uStar_psi[i], uStar_gamma[i]) : uStar_psi[i];
		double vStar_alphai = USE_GAMMA_INDICATOR==1 ? fmin(vStar_psi[i], vStar_gamma[i]) : vStar_psi[i];
		double wStar_alphai = USE_GAMMA_INDICATOR==1 ? fmin(wStar_psi[i], wStar_gamma[i]) : wStar_psi[i];

		for (int offset=rowptr_1D[i]; offset<rowptr_1D[i+1]; offset++)
		  {
		    int j = colind_1D[offset];
		    if (i!=j)
		      {
			double uj = u_dof[j];
			double vj = v_dof[j];
			double wj = w_dof[j];

			double uStar_alphaj = USE_GAMMA_INDICATOR==1 ? fmin(uStar_psi[j], uStar_gamma[j]) : uStar_psi[j];
			double vStar_alphaj = USE_GAMMA_INDICATOR==1 ? fmin(vStar_psi[j], vStar_gamma[j]) : vStar_psi[j];
			double wStar_alphaj = USE_GAMMA_INDICATOR==1 ? fmin(wStar_psi[j], wStar_gamma[j]) : wStar_psi[j];

			if (ARTIFICIAL_VISCOSITY==4) // via entropy viscosity
			  {
			    double dEVij = fmax(laggedEntropyResidualPerNode[i],
						laggedEntropyResidualPerNode[j]);
			    double dLij = fmax(0.,fmax(TransportMatrix[ij],
						       TransposeTransportMatrix[ij]));
			    uStar_dMatrix[ij] = fmin(dLij,cE*dEVij);
			    vStar_dMatrix[i] = uStar_dMatrix[ij];
			    wStar_dMatrix[i] = uStar_dMatrix[ij];
			  }
			else // via smoothness indicator
			  {
			    uStar_dMatrix[ij] = fmax(0.,fmax(uStar_alphai*TransportMatrix[ij], // by S. Badia
							     uStar_alphaj*TransposeTransportMatrix[ij]));
			    vStar_dMatrix[ij] = fmax(0.,fmax(vStar_alphai*TransportMatrix[ij], // by S. Badia
							     vStar_alphaj*TransposeTransportMatrix[ij]));
			    wStar_dMatrix[ij] = fmax(0.,fmax(wStar_alphai*TransportMatrix[ij], // by S. Badia
							     wStar_alphaj*TransposeTransportMatrix[ij]));
			  }
			uStar_dii -= uStar_dMatrix[ij];
			vStar_dii -= vStar_dMatrix[ij];
			wStar_dii -= wStar_dMatrix[ij];
			//dissipative terms
			ith_u_dissipative_term += uStar_dMatrix[ij]*(uj-ui);
			ith_v_dissipative_term += vStar_dMatrix[ij]*(vj-vi);
			ith_w_dissipative_term += wStar_dMatrix[ij]*(wj-wi);
		      }
		    else
		      {
			ii = ij;
		      }
		    // update ij
		    ij++;
		  }
		uStar_dMatrix[ii] = uStar_dii;
		vStar_dMatrix[ii] = vStar_dii;
		wStar_dMatrix[ii] = wStar_dii;
		globalResidual[offset_u+stride_u*i] += -ith_u_dissipative_term;
		globalResidual[offset_v+stride_v*i] += -ith_v_dissipative_term;
		globalResidual[offset_w+stride_w*i] += -ith_w_dissipative_term;
	      }
	  }

        //
        //loop over the surrogate boundaries in SB method and assembly into residual
        //
        if(USE_SBM>0)
        {
            std::memset(particle_netForces.data(),0,nParticles*3*sizeof(double));
            std::memset(particle_netMoments.data(),0,nParticles*3*sizeof(double));
            for (int ebN_s=0;ebN_s < surrogate_boundaries.size();ebN_s++)
            {
                int ebN = surrogate_boundaries[ebN_s],
                        eN = elementBoundaryElementsArray[ebN*2+surrogate_boundary_elements[ebN_s]],
                        ebN_local = elementBoundaryLocalElementBoundariesArray[ebN*2+surrogate_boundary_elements[ebN_s]],
                        eN_nDOF_trial_element = eN*nDOF_trial_element;
                double elementResidual_mesh[nDOF_test_element],
                elementResidual_p[nDOF_test_element],
                elementResidual_u[nDOF_test_element],
                elementResidual_v[nDOF_test_element],
                elementResidual_w[nDOF_test_element],
                eps_rho,eps_mu;
                //This assumption is wrong for parallel: If one of nodes of this edge is owned by this processor,
                //then the integral over this edge has contribution to the residual and Jacobian.
                //if (ebN >= nElementBoundaries_owned) continue;
                for (int i=0;i<nDOF_test_element;i++)
                {
                    elementResidual_mesh[i]=0.0;
                    elementResidual_p[i]=0.0;
                    elementResidual_u[i]=0.0;
                    elementResidual_v[i]=0.0;
                    elementResidual_w[i]=0.0;
                }
                for  (int kb=0;kb<nQuadraturePoints_elementBoundary;kb++)
                {
                    int ebN_kb = ebN*nQuadraturePoints_elementBoundary+kb,
                            /* ebNE_kb_nSpace = ebNE_kb*nSpace, */
                            ebN_local_kb = ebN_local*nQuadraturePoints_elementBoundary+kb,
                            ebN_local_kb_nSpace = ebN_local_kb*nSpace;
                    double u_ext=0.0, v_ext=0.0, w_ext=0.0,
                    bc_u_ext=0.0, bc_v_ext=0.0, bc_w_ext=0.0,
                    grad_u_ext[nSpace], grad_v_ext[nSpace], grad_w_ext[nSpace],
                    jac_ext[nSpace*nSpace],
                    jacDet_ext,
                    jacInv_ext[nSpace*nSpace],
                    boundaryJac[nSpace*(nSpace-1)],
                    metricTensor[(nSpace-1)*(nSpace-1)],
                    metricTensorDetSqrt,
                    dS,
                    p_test_dS[nDOF_test_element],p_grad_trial_trace[nDOF_trial_element*nSpace],
                    vel_test_dS[nDOF_test_element],
                    vel_grad_trial_trace[nDOF_trial_element*nSpace],
                    vel_grad_test_dS[nDOF_trial_element*nSpace],
                    normal[3],
                    x_ext,y_ext,z_ext,xt_ext,yt_ext,zt_ext,integralScaling,
                    G[nSpace*nSpace],G_dd_G,tr_G,h_phi,h_penalty,penalty,
                    force_x,force_y,force_z,
                    force_p_x,force_p_y,force_p_z,
                    force_v_x,force_v_y,force_v_z,
                    r_x,r_y,r_z;
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
                    dS = metricTensorDetSqrt*dS_ref[kb];
                    //get the metric tensor
                    ck.calculateG(jacInv_ext,G,G_dd_G,tr_G);
                    //compute shape and solution information
                    //shape
                    ck.gradTrialFromRef(&vel_grad_trial_trace_ref[ebN_local_kb_nSpace*nDOF_trial_element],jacInv_ext,vel_grad_trial_trace);
                    //solution and gradients
                    ck.valFromDOF(u_dof.data(),&vel_l2g[eN_nDOF_trial_element],&vel_trial_trace_ref[ebN_local_kb*nDOF_test_element],u_ext);
                    ck.valFromDOF(v_dof.data(),&vel_l2g[eN_nDOF_trial_element],&vel_trial_trace_ref[ebN_local_kb*nDOF_test_element],v_ext);
                    ck.valFromDOF(w_dof.data(),&vel_l2g[eN_nDOF_trial_element],&vel_trial_trace_ref[ebN_local_kb*nDOF_test_element],w_ext);

                    ck.gradFromDOF(u_dof.data(),&vel_l2g[eN_nDOF_trial_element],vel_grad_trial_trace,grad_u_ext);
                    ck.gradFromDOF(v_dof.data(),&vel_l2g[eN_nDOF_trial_element],vel_grad_trial_trace,grad_v_ext);
                    ck.gradFromDOF(w_dof.data(),&vel_l2g[eN_nDOF_trial_element],vel_grad_trial_trace,grad_w_ext);
                    for (int j=0;j<nDOF_trial_element;j++)
                    {
                        vel_test_dS[j] = vel_test_trace_ref[ebN_local_kb*nDOF_test_element+j]*dS;
                        for (int I=0;I<nSpace;I++)
                            vel_grad_test_dS[j*nSpace+I] = vel_grad_trial_trace[j*nSpace+I]*dS;//cek hack, using trial
                    }
                    ck.calculateGScale(G,normal,h_penalty);
                    //
                    //update the element and global residual storage
                    //
                    double dist;
                    double distance[3], P_normal[3], P_tangent[3]; // distance vector, normal and tangent of the physical boundary
                    if(use_ball_as_particle==1)
                    {
                        get_distance_to_ball(nParticles,ball_center.data(),ball_radius.data(),
                                             x_ext,y_ext,z_ext,
                                             dist);
                        get_normal_to_ith_ball(nParticles,ball_center.data(),ball_radius.data(),
                                               surrogate_boundary_particle[ebN_s],
                                               x_ext,y_ext,z_ext,
                                               P_normal[0],P_normal[1],P_normal[2]);
                        get_velocity_to_ith_ball(nParticles,ball_center.data(),ball_radius.data(),
                                                 ball_velocity.data(),ball_angular_velocity.data(),
                                                 surrogate_boundary_particle[ebN_s],
                                                 x_ext-dist*P_normal[0],//corresponding point on the boundary of the particle
                                                 y_ext-dist*P_normal[1],
                                                 z_ext-dist*P_normal[2],
                                                 bc_u_ext,bc_v_ext,bc_w_ext);

                    }
                    else
                    {
                        dist = ebq_global_phi_solid[ebN_kb];
                        P_normal[0] = ebq_global_grad_phi_solid[ebN_kb*nSpace+0];
                        P_normal[1] = ebq_global_grad_phi_solid[ebN_kb*nSpace+1];
                        P_normal[2] = ebq_global_grad_phi_solid[ebN_kb*nSpace+2];
                        bc_u_ext = ebq_particle_velocity_solid [ebN_kb*nSpace+0];
                        bc_v_ext = ebq_particle_velocity_solid [ebN_kb*nSpace+1];
                        bc_w_ext = ebq_particle_velocity_solid [ebN_kb*nSpace+2];
                    }
                    distance[0] = -P_normal[0]*dist;//distance=vector from \tilde{x} to x
                    distance[1] = -P_normal[1]*dist;
                    distance[2] = -P_normal[2]*dist;
                    assert(h_penalty>0.0);
                    if (h_penalty < std::abs(dist))
                        h_penalty = std::abs(dist);

                    if(get_dot_product(P_normal,normal)>0)
                    {
                        for(int i=0;i<3;++i)
                        {
                            normal[i] *= -1.0;
                        }
                    }
                    //hack: this won't work for two-phase flow, need mixture viscosity
                    double visco = nu_0*rho_0;
                    double C_adim = C_sbm*visco/h_penalty;
                    double beta_adim = beta_sbm*h_penalty*visco;

                    const double grad_u_d[3] = {get_dot_product(distance,grad_u_ext),
                                                get_dot_product(distance,grad_v_ext),
                                                get_dot_product(distance,grad_w_ext)};
                    double res[3];
                    const double u_m_uD[3] = {u_ext - bc_u_ext,v_ext - bc_v_ext,w_ext - bc_w_ext};
                    const double zero_vec[3]={0.,0.,0.};
                    //YY: How to define tangent penalty? There are two directions.
//                    double dt1 = P_tangent[0]*grad_u_ext[0] + P_tangent[1]*grad_u_ext[1];
//                    double dt2 = P_tangent[0]*grad_v_ext[0] + P_tangent[1]*grad_v_ext[1];
                    for (int i=0;i<nDOF_test_element;i++)
                    {
                        int eN_i = eN*nDOF_test_element+i;

                        int GlobPos_u = offset_u+stride_u*vel_l2g[eN_i];
                        int GlobPos_v = offset_v+stride_v*vel_l2g[eN_i];
                        int GlobPos_w = offset_w+stride_w*vel_l2g[eN_i];
                        const double phi_i = vel_test_dS[i];
                        double *grad_phi_i = &vel_grad_test_dS[i*nSpace+0];
                        const double grad_phi_i_dot_d =  get_dot_product(distance,grad_phi_i);

                        // (1)
                        globalResidual[GlobPos_u] += C_adim*phi_i*u_m_uD[0];
                        globalResidual[GlobPos_v] += C_adim*phi_i*u_m_uD[1];
                        globalResidual[GlobPos_w] += C_adim*phi_i*u_m_uD[2];

                        // (2)
                        get_symmetric_gradient_dot_vec(grad_u_ext,grad_v_ext,grad_w_ext,normal,res);
                        globalResidual[GlobPos_u] -= visco * phi_i*res[0];
                        globalResidual[GlobPos_v] -= visco * phi_i*res[1];
                        globalResidual[GlobPos_w] -= visco * phi_i*res[2];

                        // (3)
                        get_symmetric_gradient_dot_vec(grad_phi_i,zero_vec,zero_vec,normal,res);
                        globalResidual[GlobPos_u] -= visco * get_dot_product(u_m_uD,res);
                        get_symmetric_gradient_dot_vec(zero_vec,grad_phi_i,zero_vec,normal,res);
                        globalResidual[GlobPos_v] -= visco * get_dot_product(u_m_uD,res);
                        get_symmetric_gradient_dot_vec(zero_vec,zero_vec,grad_phi_i,normal,res);
                        globalResidual[GlobPos_w] -= visco * get_dot_product(u_m_uD,res);

                        // (4)
                        globalResidual[GlobPos_u] += C_adim*grad_phi_i_dot_d*u_m_uD[0];
                        globalResidual[GlobPos_v] += C_adim*grad_phi_i_dot_d*u_m_uD[1];
                        globalResidual[GlobPos_w] += C_adim*grad_phi_i_dot_d*u_m_uD[2];

                        // (5)
                        globalResidual[GlobPos_u] += C_adim*grad_phi_i_dot_d*grad_u_d[0];
                        globalResidual[GlobPos_v] += C_adim*grad_phi_i_dot_d*grad_u_d[1];
                        globalResidual[GlobPos_w] += C_adim*grad_phi_i_dot_d*grad_u_d[2];

                        // (6)
                        globalResidual[GlobPos_u] += C_adim*phi_i*grad_u_d[0];
                        globalResidual[GlobPos_v] += C_adim*phi_i*grad_u_d[1];
                        globalResidual[GlobPos_w] += C_adim*phi_i*grad_u_d[2];

                        // (7)
                        get_symmetric_gradient_dot_vec(grad_phi_i,zero_vec,zero_vec,normal,res);
                        globalResidual[GlobPos_u] -= visco*get_dot_product(grad_u_d,res);
                        get_symmetric_gradient_dot_vec(zero_vec,grad_phi_i,zero_vec,normal,res);
                        globalResidual[GlobPos_v] -=visco*get_dot_product(grad_u_d,res);
                        get_symmetric_gradient_dot_vec(zero_vec,zero_vec,grad_phi_i,normal,res);
                        globalResidual[GlobPos_w] -= visco*get_dot_product(grad_u_d,res);

                        // the penalization on the tangential derivative (8)
//                        globalResidual[GlobPos_u] += beta_adim*dt1*(Gxphi_i*P_tangent[0] + Gyphi_i*P_tangent[1]);
//                        globalResidual[GlobPos_v] += beta_adim*dt2*(Gxphi_i*P_tangent[0] + Gyphi_i*P_tangent[1]);

                      }//i

                    //
                    // Forces
                    //
                    //compute pressure at the quadrature point of the edge from dof-value of the pressure
                    double p_ext = 0.0;
                    for (int i=0; i<nDOF_per_element_pressure;++i)
                      {
                        p_ext += p_dof[p_l2g[eN*nDOF_per_element_pressure+i]]*p_trial_trace_ref[ebN_local_kb*nDOF_per_element_pressure+i];
                    }
                    get_symmetric_gradient_dot_vec(grad_u_ext,grad_v_ext,grad_w_ext,P_normal,res);
                    double force_quad_pt[3]={0.0,0.0,0.0},torque_quad_pt[3]={0.0,0.0,0.0},position_vector_to_mass_center[3];

                    get_stress_in_n(grad_u_ext,grad_v_ext,grad_w_ext,P_normal,p_ext,visco,force_quad_pt);

                    force_quad_pt[0] *= dS;
                    force_quad_pt[1] *= dS;
                    force_quad_pt[2] *= dS;
                    if(use_ball_as_particle==1)
                    {
                        position_vector_to_mass_center[0] = x_ext - ball_center[surrogate_boundary_particle[ebN_s] * 3 + 0];
                        position_vector_to_mass_center[1] = y_ext - ball_center[surrogate_boundary_particle[ebN_s] * 3 + 1];
                        position_vector_to_mass_center[2] = z_ext - ball_center[surrogate_boundary_particle[ebN_s] * 3 + 2];
                    }
                    else
                    {
                        position_vector_to_mass_center[0] = x_ext - particle_centroids[surrogate_boundary_particle[ebN_s] * 3 + 0];
                        position_vector_to_mass_center[1] = y_ext - particle_centroids[surrogate_boundary_particle[ebN_s] * 3 + 1];
                        position_vector_to_mass_center[2] = z_ext - particle_centroids[surrogate_boundary_particle[ebN_s] * 3 + 2];
                    }
                    get_cross_product(position_vector_to_mass_center,force_quad_pt,torque_quad_pt);
                    if(ebN < nElementBoundaries_owned)//avoid double counting
                    {
                        particle_netForces[3*surrogate_boundary_particle[ebN_s]+0] += force_quad_pt[0];
                        particle_netForces[3*surrogate_boundary_particle[ebN_s]+1] += force_quad_pt[1];
                        particle_netForces[3*surrogate_boundary_particle[ebN_s]+2] += force_quad_pt[2];
                        particle_netMoments[3*surrogate_boundary_particle[ebN_s]+0] += torque_quad_pt[0];
                        particle_netMoments[3*surrogate_boundary_particle[ebN_s]+1] += torque_quad_pt[1];
                        particle_netMoments[3*surrogate_boundary_particle[ebN_s]+2] += torque_quad_pt[2];
                    }
                }//kb
            }//ebN_s
        }
        //
        //loop over exterior element boundaries to calculate surface integrals and load into element and global residuals
        //
        //ebNE is the Exterior element boundary INdex
        //ebN is the element boundary INdex
        //eN is the element index
        gf.useExact=false;
        gf_s.useExact=false;
        for (int ebNE = 0; ebNE < nExteriorElementBoundaries_global; ebNE++)
          {
            int ebN = exteriorElementBoundariesArray[ebNE],
              eN  = elementBoundaryElementsArray[ebN*2+0],
              ebN_local = elementBoundaryLocalElementBoundariesArray[ebN*2+0],
              eN_nDOF_trial_element = eN*nDOF_trial_element;
            double elementResidual_mesh[nDOF_test_element],
              elementResidual_p[nDOF_test_element],
              elementResidual_u[nDOF_test_element],
              elementResidual_v[nDOF_test_element],
              elementResidual_w[nDOF_test_element],
              eps_rho,eps_mu;
            for (int i=0;i<nDOF_test_element;i++)
              {
                elementResidual_mesh[i]=0.0;
                elementResidual_p[i]=0.0;
                elementResidual_u[i]=0.0;
                elementResidual_v[i]=0.0;
                elementResidual_w[i]=0.0;
              }
            for  (int kb=0;kb<nQuadraturePoints_elementBoundary;kb++)
              {
                int ebNE_kb = ebNE*nQuadraturePoints_elementBoundary+kb,
                  ebNE_kb_nSpace = ebNE_kb*nSpace,
                  ebN_local_kb = ebN_local*nQuadraturePoints_elementBoundary+kb,
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
                  flux_mom_wu_diff_ext=0.0,
                  flux_mom_wv_diff_ext=0.0,
                  flux_mom_ww_diff_ext=0.0,
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
                  dS,p_test_dS[nDOF_test_element],vel_test_dS[nDOF_test_element],
                  p_grad_trial_trace[nDOF_trial_element*nSpace],vel_grad_trial_trace[nDOF_trial_element*nSpace],
                  vel_grad_test_dS[nDOF_trial_element*nSpace],
                  normal[3],x_ext,y_ext,z_ext,xt_ext,yt_ext,zt_ext,integralScaling,
                  //VRANS
                  porosity_ext,
                  //
                  G[nSpace*nSpace],G_dd_G,tr_G,h_phi,h_penalty,penalty,
                  force_x,force_y,force_z,force_p_x,force_p_y,force_p_z,force_v_x,force_v_y,force_v_z,r_x,r_y,r_z;
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
                /*             <<"dS_ref[kb]"<<dS_ref[kb]<<std::endl; */
                //dS = ((1.0-MOVING_DOMAIN)*metricTensorDetSqrt + MOVING_DOMAIN*integralScaling)*dS_ref[kb];//cek need to test effect on accuracy
                dS = metricTensorDetSqrt*dS_ref[kb];
                //get the metric tensor
                //cek todo use symmetry
                ck.calculateG(jacInv_ext,G,G_dd_G,tr_G);
                ck.calculateGScale(G,&ebqe_normal_phi_ext[ebNE_kb_nSpace],h_phi);

                eps_rho = epsFact_rho*(useMetrics*h_phi+(1.0-useMetrics)*elementDiameter[eN]);
                eps_mu  = epsFact_mu *(useMetrics*h_phi+(1.0-useMetrics)*elementDiameter[eN]);
                double particle_eps  = particle_epsFact*(useMetrics*h_phi+(1.0-useMetrics)*elementDiameter[eN]);

                //compute shape and solution information
                //shape
                /* ck.gradTrialFromRef(&p_grad_trial_trace_ref[ebN_local_kb_nSpace*nDOF_trial_element],jacInv_ext,p_grad_trial_trace); */
                ck.gradTrialFromRef(&vel_grad_trial_trace_ref[ebN_local_kb_nSpace*nDOF_trial_element],jacInv_ext,vel_grad_trial_trace);
                //cek hack use trial ck.gradTrialFromRef(&vel_grad_test_trace_ref[ebN_local_kb_nSpace*nDOF_trial_element],jacInv_ext,vel_grad_test_trace);
                //solution and gradients
                /* ck.valFromDOF(p_dof,&p_l2g[eN_nDOF_trial_element],&p_trial_trace_ref[ebN_local_kb*nDOF_test_element],p_ext); */
                p_ext = ebqe_p[ebNE_kb];
                ck.valFromDOF(u_dof.data(),&vel_l2g[eN_nDOF_trial_element],&vel_trial_trace_ref[ebN_local_kb*nDOF_test_element],u_ext);
                ck.valFromDOF(v_dof.data(),&vel_l2g[eN_nDOF_trial_element],&vel_trial_trace_ref[ebN_local_kb*nDOF_test_element],v_ext);
                ck.valFromDOF(w_dof.data(),&vel_l2g[eN_nDOF_trial_element],&vel_trial_trace_ref[ebN_local_kb*nDOF_test_element],w_ext);
                /* ck.gradFromDOF(p_dof,&p_l2g[eN_nDOF_trial_element],p_grad_trial_trace,grad_p_ext); */
                for (int I=0;I<nSpace;I++)
                  grad_p_ext[I] = ebqe_grad_p[ebNE_kb_nSpace + I];
                ck.gradFromDOF(u_dof.data(),&vel_l2g[eN_nDOF_trial_element],vel_grad_trial_trace,grad_u_ext);
                ck.gradFromDOF(v_dof.data(),&vel_l2g[eN_nDOF_trial_element],vel_grad_trial_trace,grad_v_ext);
                ck.gradFromDOF(w_dof.data(),&vel_l2g[eN_nDOF_trial_element],vel_grad_trial_trace,grad_w_ext);
                //precalculate test function products with integration weights
                for (int j=0;j<nDOF_trial_element;j++)
                  {
                    /* p_test_dS[j] = p_test_trace_ref[ebN_local_kb*nDOF_test_element+j]*dS; */
                    vel_test_dS[j] = vel_test_trace_ref[ebN_local_kb*nDOF_test_element+j]*dS;
                    for (int I=0;I<nSpace;I++)
                      vel_grad_test_dS[j*nSpace+I] = vel_grad_trial_trace[j*nSpace+I]*dS;//cek hack, using trial
                  }
                bc_p_ext = isDOFBoundary_p[ebNE_kb]*ebqe_bc_p_ext[ebNE_kb]+(1-isDOFBoundary_p[ebNE_kb])*p_ext;
                //note, our convention is that bc values at moving boundaries are relative to boundary velocity so we add it here
                bc_u_ext = isDOFBoundary_u[ebNE_kb]*(ebqe_bc_u_ext[ebNE_kb] + MOVING_DOMAIN*xt_ext) + (1-isDOFBoundary_u[ebNE_kb])*u_ext;
                bc_v_ext = isDOFBoundary_v[ebNE_kb]*(ebqe_bc_v_ext[ebNE_kb] + MOVING_DOMAIN*yt_ext) + (1-isDOFBoundary_v[ebNE_kb])*v_ext;
                bc_w_ext = isDOFBoundary_w[ebNE_kb]*(ebqe_bc_w_ext[ebNE_kb] + MOVING_DOMAIN*zt_ext) + (1-isDOFBoundary_w[ebNE_kb])*w_ext;
                //VRANS
                porosity_ext = 1.0 - ebqe_vos_ext[ebNE_kb];
                //
                //calculate the pde coefficients using the solution and the boundary values for the solution
                //
                double distance_to_omega_solid = 1e10;
                if (use_ball_as_particle == 1)
                {
                  get_distance_to_ball(nParticles, ball_center.data(), ball_radius.data(), x_ext, y_ext, z_ext, distance_to_omega_solid);
                }
                else
                {
                  for (int i = 0; i < nParticles; i++)
                  {
                    double distance_to_i_th_solid = ebq_global_phi_solid[i * nElementBoundaries_global * nQuadraturePoints_elementBoundary + ebNE_kb];
                    distance_to_omega_solid = (distance_to_i_th_solid < distance_to_omega_solid)?distance_to_i_th_solid:distance_to_omega_solid;
                  }
                }
                double eddy_viscosity_ext(0.),bc_eddy_viscosity_ext(0.); //not interested in saving boundary eddy viscosity for now
                evaluateCoefficients(eps_rho,
                                     eps_mu,
                                     particle_eps,
                                     sigma,
                                     rho_0,
                                     nu_0,
                                     rho_1,
                                     nu_1,
                                     elementDiameter[eN],
                                     smagorinskyConstant,
                                     turbulenceClosureModel,
                                     g.data(),
                                     useVF,
                                     ebqe_vf_ext[ebNE_kb],
                                     ebqe_phi_ext[ebNE_kb],
                                     &ebqe_normal_phi_ext[ebNE_kb_nSpace],
                                     distance_to_omega_solid,
                                     ebqe_kappa_phi_ext[ebNE_kb],
                                     //VRANS
                                     porosity_ext,
                                     //
                                     p_ext,
                                     grad_p_ext,
                                     grad_u_ext,
                                     grad_v_ext,
                                     grad_w_ext,
                                     u_ext,
                                     v_ext,
                                     w_ext,
                                     ebqe_velocity_star[ebNE_kb_nSpace+0],
                                     ebqe_velocity_star[ebNE_kb_nSpace+1],
                                     ebqe_velocity_star[ebNE_kb_nSpace+2],
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
                                     dmom_w_ham_grad_w_ext,
                                     ebqe_rho[ebNE_kb],
                                     ebqe_nu[ebNE_kb],
                                     KILL_PRESSURE_TERM,
                                     0,
                                     0., // mql: zero force term at boundary
                                     0.,
                                     0.,
                                     MATERIAL_PARAMETERS_AS_FUNCTION,
                                     ebqe_density_as_function[ebNE_kb],
                                     ebqe_dynamic_viscosity_as_function[ebNE_kb],
                                     USE_SBM,
                                     x_ext,y_ext,z_ext,
                                     use_ball_as_particle,
                                     ball_center.data(),
                                     ball_radius.data(),
                                     ball_velocity.data(),
                                     ball_angular_velocity.data(),
				     INT_BY_PARTS_PRESSURE);
                evaluateCoefficients(eps_rho,
                                     eps_mu,
                                     particle_eps,
                                     sigma,
                                     rho_0,
                                     nu_0,
                                     rho_1,
                                     nu_1,
                                     elementDiameter[eN],
                                     smagorinskyConstant,
                                     turbulenceClosureModel,
                                     g.data(),
                                     useVF,
                                     bc_ebqe_vf_ext[ebNE_kb],
                                     bc_ebqe_phi_ext[ebNE_kb],
                                     &ebqe_normal_phi_ext[ebNE_kb_nSpace],
                                     distance_to_omega_solid,
                                     ebqe_kappa_phi_ext[ebNE_kb],
                                     //VRANS
                                     porosity_ext,
                                     //
                                     bc_p_ext,
                                     grad_p_ext,
                                     grad_u_ext,
                                     grad_v_ext,
                                     grad_w_ext,
                                     bc_u_ext,
                                     bc_v_ext,
                                     bc_w_ext,
                                     ebqe_velocity_star[ebNE_kb_nSpace+0],
                                     ebqe_velocity_star[ebNE_kb_nSpace+1],
                                     ebqe_velocity_star[ebNE_kb_nSpace+2],
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
                                     bc_dmom_w_ham_grad_w_ext,
                                     ebqe_rho[ebNE_kb],
                                     ebqe_nu[ebNE_kb],
                                     KILL_PRESSURE_TERM,
                                     0,
                                     0., // mql: zero force term at boundary
                                     0.,
                                     0.,
                                     MATERIAL_PARAMETERS_AS_FUNCTION,
                                     ebqe_density_as_function[ebNE_kb],
                                     ebqe_dynamic_viscosity_as_function[ebNE_kb],
                                     USE_SBM,
                                     x_ext,y_ext,z_ext,
                                     use_ball_as_particle,
                                     ball_center.data(),
                                     ball_radius.data(),
                                     ball_velocity.data(),
                                     ball_angular_velocity.data(),
				     INT_BY_PARTS_PRESSURE);

                //Turbulence closure model
                if (turbulenceClosureModel >= 3)
                  {
                    const double turb_var_grad_0_dummy[3] = {0.,0.,0.};
                    const double c_mu = 0.09;//mwf hack
                    updateTurbulenceClosure(turbulenceClosureModel,
                                            eps_rho,
                                            eps_mu,
                                            rho_0,
                                            nu_0,
                                            rho_1,
                                            nu_1,
                                            useVF,
                                            ebqe_vf_ext[ebNE_kb],
                                            ebqe_phi_ext[ebNE_kb],
                                            porosity_ext,
                                            c_mu, //mwf hack
                                            ebqe_turb_var_0[ebNE_kb],
                                            ebqe_turb_var_1[ebNE_kb],
                                            turb_var_grad_0_dummy, //not needed
                                            eddy_viscosity_ext,
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
                                            mom_w_source_ext);

                    updateTurbulenceClosure(turbulenceClosureModel,
                                            eps_rho,
                                            eps_mu,
                                            rho_0,
                                            nu_0,
                                            rho_1,
                                            nu_1,
                                            useVF,
                                            bc_ebqe_vf_ext[ebNE_kb],
                                            bc_ebqe_phi_ext[ebNE_kb],
                                            porosity_ext,
                                            c_mu, //mwf hack
                                            ebqe_turb_var_0[ebNE_kb],
                                            ebqe_turb_var_1[ebNE_kb],
                                            turb_var_grad_0_dummy, //not needed
                                            bc_eddy_viscosity_ext,
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
                                            bc_mom_w_source_ext);
                  }


                //
                //moving domain
                //
                mom_u_adv_ext[0] -= MOVING_DOMAIN*dmom_u_acc_u_ext*mom_u_acc_ext*xt_ext; // times rho*porosity. mql. CHECK.
                mom_u_adv_ext[1] -= MOVING_DOMAIN*dmom_u_acc_u_ext*mom_u_acc_ext*yt_ext;
                mom_u_adv_ext[2] -= MOVING_DOMAIN*dmom_u_acc_u_ext*mom_u_acc_ext*zt_ext;
                dmom_u_adv_u_ext[0] -= MOVING_DOMAIN*dmom_u_acc_u_ext*xt_ext;
                dmom_u_adv_u_ext[1] -= MOVING_DOMAIN*dmom_u_acc_u_ext*yt_ext;
                dmom_u_adv_u_ext[2] -= MOVING_DOMAIN*dmom_u_acc_u_ext*zt_ext;

                mom_v_adv_ext[0] -= MOVING_DOMAIN*dmom_v_acc_v_ext*mom_v_acc_ext*xt_ext;
                mom_v_adv_ext[1] -= MOVING_DOMAIN*dmom_v_acc_v_ext*mom_v_acc_ext*yt_ext;
                mom_v_adv_ext[2] -= MOVING_DOMAIN*dmom_v_acc_v_ext*mom_v_acc_ext*zt_ext;
                dmom_v_adv_v_ext[0] -= MOVING_DOMAIN*dmom_v_acc_v_ext*xt_ext;
                dmom_v_adv_v_ext[1] -= MOVING_DOMAIN*dmom_v_acc_v_ext*yt_ext;
                dmom_v_adv_v_ext[2] -= MOVING_DOMAIN*dmom_v_acc_v_ext*zt_ext;

                mom_w_adv_ext[0] -= MOVING_DOMAIN*dmom_w_acc_w_ext*mom_w_acc_ext*xt_ext;
                mom_w_adv_ext[1] -= MOVING_DOMAIN*dmom_w_acc_w_ext*mom_w_acc_ext*yt_ext;
                mom_w_adv_ext[2] -= MOVING_DOMAIN*dmom_w_acc_w_ext*mom_w_acc_ext*zt_ext;
                dmom_w_adv_w_ext[0] -= MOVING_DOMAIN*dmom_w_acc_w_ext*xt_ext;
                dmom_w_adv_w_ext[1] -= MOVING_DOMAIN*dmom_w_acc_w_ext*yt_ext;
                dmom_w_adv_w_ext[2] -= MOVING_DOMAIN*dmom_w_acc_w_ext*zt_ext;

                //bc's
                // mql. CHECK.
                bc_mom_u_adv_ext[0] -= MOVING_DOMAIN*bc_dmom_u_acc_u_ext*bc_mom_u_acc_ext*xt_ext; // times rho*porosity
                bc_mom_u_adv_ext[1] -= MOVING_DOMAIN*bc_dmom_u_acc_u_ext*bc_mom_u_acc_ext*yt_ext;
                bc_mom_u_adv_ext[2] -= MOVING_DOMAIN*bc_dmom_u_acc_u_ext*bc_mom_u_acc_ext*zt_ext;

                bc_mom_v_adv_ext[0] -= MOVING_DOMAIN*bc_dmom_v_acc_v_ext*bc_mom_v_acc_ext*xt_ext;
                bc_mom_v_adv_ext[1] -= MOVING_DOMAIN*bc_dmom_v_acc_v_ext*bc_mom_v_acc_ext*yt_ext;
                bc_mom_v_adv_ext[2] -= MOVING_DOMAIN*bc_dmom_v_acc_v_ext*bc_mom_v_acc_ext*zt_ext;

                bc_mom_w_adv_ext[0] -= MOVING_DOMAIN*bc_dmom_w_acc_w_ext*bc_mom_w_acc_ext*xt_ext;
                bc_mom_w_adv_ext[1] -= MOVING_DOMAIN*bc_dmom_w_acc_w_ext*bc_mom_w_acc_ext*yt_ext;
                bc_mom_w_adv_ext[2] -= MOVING_DOMAIN*bc_dmom_w_acc_w_ext*bc_mom_w_acc_ext*zt_ext;

                //
                //calculate the numerical fluxes
                //
                ck.calculateGScale(G,normal,h_penalty);
                penalty = useMetrics*C_b/h_penalty + (1.0-useMetrics)*ebqe_penalty_ext[ebNE_kb];
                exteriorNumericalAdvectiveFlux(isDOFBoundary_p[ebNE_kb],
                                               isDOFBoundary_u[ebNE_kb],
                                               isDOFBoundary_v[ebNE_kb],
                                               isDOFBoundary_w[ebNE_kb],
                                               isAdvectiveFluxBoundary_p[ebNE_kb],
                                               isAdvectiveFluxBoundary_u[ebNE_kb],
                                               isAdvectiveFluxBoundary_v[ebNE_kb],
                                               isAdvectiveFluxBoundary_w[ebNE_kb],
                                               dmom_u_ham_grad_p_ext[0],//=1/rho,
                                               bc_dmom_u_ham_grad_p_ext[0],//=1/bc_rho,
                                               normal,
                                               porosity_ext*ebqe_rho[ebNE_kb],//dmom_u_acc_u_ext,
                                               bc_p_ext,
                                               bc_u_ext,
                                               bc_v_ext,
                                               bc_w_ext,
                                               bc_mass_adv_ext,
                                               bc_mom_u_adv_ext,
                                               bc_mom_v_adv_ext,
                                               bc_mom_w_adv_ext,
                                               ebqe_bc_flux_mass_ext[ebNE_kb]+MOVING_DOMAIN*(xt_ext*normal[0]+yt_ext*normal[1]+zt_ext*normal[2]),//BC is relative mass flux
                                               ebqe_bc_flux_mom_u_adv_ext[ebNE_kb],
                                               ebqe_bc_flux_mom_v_adv_ext[ebNE_kb],
                                               ebqe_bc_flux_mom_w_adv_ext[ebNE_kb],
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
                                               &ebqe_velocity_star[ebNE_kb_nSpace],
                                               &ebqe_velocity[ebNE_kb_nSpace]);
                // mql: save gradient of solution for other models and to compute errors
                for (int I=0;I<nSpace;I++)
                  {
                    ebqe_grad_u[ebNE_kb_nSpace+I] = grad_u_ext[I];
                    ebqe_grad_v[ebNE_kb_nSpace+I] = grad_v_ext[I];
                    ebqe_grad_w[ebNE_kb_nSpace+I] = grad_w_ext[I];
                  }
                exteriorNumericalDiffusiveFlux(eps_rho,
                                               ebqe_phi_ext[ebNE_kb],
                                               sdInfo_u_u_rowptr.data(),
                                               sdInfo_u_u_colind.data(),
                                               isDOFBoundary_u[ebNE_kb],
                                               isDiffusiveFluxBoundary_u[ebNE_kb],
                                               normal,
                                               bc_mom_uu_diff_ten_ext,
                                               bc_u_ext,
                                               ebqe_bc_flux_u_diff_ext[ebNE_kb],
                                               mom_uu_diff_ten_ext,
                                               grad_u_ext,
                                               u_ext,
                                               penalty,//ebqe_penalty_ext[ebNE_kb],
                                               flux_mom_uu_diff_ext);
                exteriorNumericalDiffusiveFlux(eps_rho,
                                               ebqe_phi_ext[ebNE_kb],
                                               sdInfo_u_v_rowptr.data(),
                                               sdInfo_u_v_colind.data(),
                                               isDOFBoundary_v[ebNE_kb],
                                               isDiffusiveFluxBoundary_v[ebNE_kb],
                                               normal,
                                               bc_mom_uv_diff_ten_ext,
                                               bc_v_ext,
                                               0.0,//assume all of the flux gets applied in diagonal component
                                               mom_uv_diff_ten_ext,
                                               grad_v_ext,
                                               v_ext,
                                               penalty,//ebqe_penalty_ext[ebNE_kb],
                                               flux_mom_uv_diff_ext);
                exteriorNumericalDiffusiveFlux(eps_rho,
                                               ebqe_phi_ext[ebNE_kb],
                                               sdInfo_u_w_rowptr.data(),
                                               sdInfo_u_w_colind.data(),
                                               isDOFBoundary_w[ebNE_kb],
                                               isDiffusiveFluxBoundary_w[ebNE_kb],
                                               normal,
                                               bc_mom_uw_diff_ten_ext,
                                               bc_w_ext,
                                               0.0,//see above
                                               mom_uw_diff_ten_ext,
                                               grad_w_ext,
                                               w_ext,
                                               penalty,//ebqe_penalty_ext[ebNE_kb],
                                               flux_mom_uw_diff_ext);
                exteriorNumericalDiffusiveFlux(eps_rho,
                                               ebqe_phi_ext[ebNE_kb],
                                               sdInfo_v_u_rowptr.data(),
                                               sdInfo_v_u_colind.data(),
                                               isDOFBoundary_u[ebNE_kb],
                                               isDiffusiveFluxBoundary_u[ebNE_kb],
                                               normal,
                                               bc_mom_vu_diff_ten_ext,
                                               bc_u_ext,
                                               0.0,//see above
                                               mom_vu_diff_ten_ext,
                                               grad_u_ext,
                                               u_ext,
                                               penalty,//ebqe_penalty_ext[ebNE_kb],
                                               flux_mom_vu_diff_ext);
                exteriorNumericalDiffusiveFlux(eps_rho,
                                               ebqe_phi_ext[ebNE_kb],
                                               sdInfo_v_v_rowptr.data(),
                                               sdInfo_v_v_colind.data(),
                                               isDOFBoundary_v[ebNE_kb],
                                               isDiffusiveFluxBoundary_v[ebNE_kb],
                                               normal,
                                               bc_mom_vv_diff_ten_ext,
                                               bc_v_ext,
                                               ebqe_bc_flux_v_diff_ext[ebNE_kb],
                                               mom_vv_diff_ten_ext,
                                               grad_v_ext,
                                               v_ext,
                                               penalty,//ebqe_penalty_ext[ebNE_kb],
                                               flux_mom_vv_diff_ext);
                exteriorNumericalDiffusiveFlux(eps_rho,
                                               ebqe_phi_ext[ebNE_kb],
                                               sdInfo_v_w_rowptr.data(),
                                               sdInfo_v_w_colind.data(),
                                               isDOFBoundary_w[ebNE_kb],
                                               isDiffusiveFluxBoundary_w[ebNE_kb],
                                               normal,
                                               bc_mom_vw_diff_ten_ext,
                                               bc_w_ext,
                                               0.0,//see above
                                               mom_vw_diff_ten_ext,
                                               grad_w_ext,
                                               w_ext,
                                               penalty,//ebqe_penalty_ext[ebNE_kb],
                                               flux_mom_vw_diff_ext);
                exteriorNumericalDiffusiveFlux(eps_rho,
                                               ebqe_phi_ext[ebNE_kb],
                                               sdInfo_w_u_rowptr.data(),
                                               sdInfo_w_u_colind.data(),
                                               isDOFBoundary_u[ebNE_kb],
                                               isDiffusiveFluxBoundary_u[ebNE_kb],
                                               normal,
                                               bc_mom_wu_diff_ten_ext,
                                               bc_u_ext,
                                               0.0,//see above
                                               mom_wu_diff_ten_ext,
                                               grad_u_ext,
                                               u_ext,
                                               penalty,//ebqe_penalty_ext[ebNE_kb],
                                               flux_mom_wu_diff_ext);
                exteriorNumericalDiffusiveFlux(eps_rho,
                                               ebqe_phi_ext[ebNE_kb],
                                               sdInfo_w_v_rowptr.data(),
                                               sdInfo_w_v_colind.data(),
                                               isDOFBoundary_v[ebNE_kb],
                                               isDiffusiveFluxBoundary_v[ebNE_kb],
                                               normal,
                                               bc_mom_wv_diff_ten_ext,
                                               bc_v_ext,
                                               0.0,//see above
                                               mom_wv_diff_ten_ext,
                                               grad_v_ext,
                                               v_ext,
                                               penalty,//ebqe_penalty_ext[ebNE_kb],
                                               flux_mom_wv_diff_ext);
                exteriorNumericalDiffusiveFlux(eps_rho,
                                               ebqe_phi_ext[ebNE_kb],
                                               sdInfo_w_w_rowptr.data(),
                                               sdInfo_w_w_colind.data(),
                                               isDOFBoundary_w[ebNE_kb],
                                               isDiffusiveFluxBoundary_w[ebNE_kb],
                                               normal,
                                               bc_mom_ww_diff_ten_ext,
                                               bc_w_ext,
                                               ebqe_bc_flux_w_diff_ext[ebNE_kb],
                                               mom_ww_diff_ten_ext,
                                               grad_w_ext,
                                               w_ext,
                                               penalty,//ebqe_penalty_ext[ebNE_kb],
                                               flux_mom_ww_diff_ext);
                flux[ebN*nQuadraturePoints_elementBoundary+kb] = flux_mass_ext;
                /* std::cout<<"external u,v,u_n " */
                /*             <<ebqe_velocity[ebNE_kb_nSpace+0]<<'\t' */
                /*             <<ebqe_velocity[ebNE_kb_nSpace+1]<<'\t' */
                /*             <<flux[ebN*nQuadraturePoints_elementBoundary+kb]<<std::endl; */
                //
                //integrate the net force and moment on flagged boundaries
                //
                if (ebN < nElementBoundaries_owned)
                  {
                    force_v_x = (flux_mom_u_adv_ext + flux_mom_uu_diff_ext + flux_mom_uv_diff_ext + flux_mom_uw_diff_ext)/dmom_u_ham_grad_p_ext[0];//same as *rho
                    force_v_y = (flux_mom_v_adv_ext + flux_mom_vu_diff_ext + flux_mom_vv_diff_ext + flux_mom_vw_diff_ext)/dmom_u_ham_grad_p_ext[0];
                    force_v_z = (flux_mom_w_adv_ext + flux_mom_wu_diff_ext + flux_mom_wv_diff_ext + flux_mom_ww_diff_ext)/dmom_u_ham_grad_p_ext[0];

                    force_p_x = p_ext*normal[0];
                    force_p_y = p_ext*normal[1];
                    force_p_z = p_ext*normal[2];

                    force_x = force_p_x + force_v_x;
                    force_y = force_p_y + force_v_y;
                    force_z = force_p_z + force_v_z;

                    r_x = x_ext - barycenters[3*boundaryFlags[ebN]+0];
                    r_y = y_ext - barycenters[3*boundaryFlags[ebN]+1];
                    r_z = z_ext - barycenters[3*boundaryFlags[ebN]+2];

                    wettedAreas[boundaryFlags[ebN]] += dS*(1.0-ebqe_vf_ext[ebNE_kb]);

                    netForces_p[3*boundaryFlags[ebN]+0] += force_p_x*dS;
                    netForces_p[3*boundaryFlags[ebN]+1] += force_p_y*dS;
                    netForces_p[3*boundaryFlags[ebN]+2] += force_p_z*dS;

                    netForces_v[3*boundaryFlags[ebN]+0] += force_v_x*dS;
                    netForces_v[3*boundaryFlags[ebN]+1] += force_v_y*dS;
                    netForces_v[3*boundaryFlags[ebN]+2] += force_v_z*dS;

                    netMoments[3*boundaryFlags[ebN]+0] += (r_y*force_z - r_z*force_y)*dS;
                    netMoments[3*boundaryFlags[ebN]+1] += (r_z*force_x - r_x*force_z)*dS;
                    netMoments[3*boundaryFlags[ebN]+2] += (r_x*force_y - r_y*force_x)*dS;
                  }
                //
                //update residuals
                //
                for (int i=0;i<nDOF_test_element;i++)
                  {
                    elementResidual_mesh[i] -= ck.ExteriorElementBoundaryFlux(MOVING_DOMAIN*(xt_ext*normal[0]+yt_ext*normal[1]+zt_ext*normal[2]),p_test_dS[i]);
                    elementResidual_p[i] += ck.ExteriorElementBoundaryFlux(flux_mass_ext,p_test_dS[i]);
                    elementResidual_p[i] -= DM*ck.ExteriorElementBoundaryFlux(MOVING_DOMAIN*(xt_ext*normal[0]+yt_ext*normal[1]+zt_ext*normal[2]),p_test_dS[i]);
                    globalConservationError += ck.ExteriorElementBoundaryFlux(flux_mass_ext,p_test_dS[i]);
                    elementResidual_u[i] +=
		      (INT_BY_PARTS_PRESSURE==1 ? p_ext*vel_test_dS[i]*normal[0] : 0.) +
                      ck.ExteriorElementBoundaryFlux(flux_mom_u_adv_ext,vel_test_dS[i])+
                      ck.ExteriorElementBoundaryFlux(flux_mom_uu_diff_ext,vel_test_dS[i])+
                      ck.ExteriorElementBoundaryFlux(flux_mom_uv_diff_ext,vel_test_dS[i])+
                      ck.ExteriorElementBoundaryFlux(flux_mom_uw_diff_ext,vel_test_dS[i])+
                      ck.ExteriorElementBoundaryDiffusionAdjoint(isDOFBoundary_u[ebNE_kb],
                                                                 isDiffusiveFluxBoundary_u[ebNE_kb],
                                                                 eb_adjoint_sigma,
                                                                 u_ext,
                                                                 bc_u_ext,
                                                                 normal,
                                                                 sdInfo_u_u_rowptr.data(),
                                                                 sdInfo_u_u_colind.data(),
                                                                 mom_uu_diff_ten_ext,
                                                                 &vel_grad_test_dS[i*nSpace])+
                      ck.ExteriorElementBoundaryDiffusionAdjoint(isDOFBoundary_v[ebNE_kb],
                                                                 isDiffusiveFluxBoundary_u[ebNE_kb],
                                                                 eb_adjoint_sigma,
                                                                 v_ext,
                                                                 bc_v_ext,
                                                                 normal,
                                                                 sdInfo_u_v_rowptr.data(),
                                                                 sdInfo_u_v_colind.data(),
                                                                 mom_uv_diff_ten_ext,
                                                                 &vel_grad_test_dS[i*nSpace])+
                      ck.ExteriorElementBoundaryDiffusionAdjoint(isDOFBoundary_w[ebNE_kb],
                                                                 isDiffusiveFluxBoundary_u[ebNE_kb],
                                                                 eb_adjoint_sigma,
                                                                 w_ext,
                                                                 bc_w_ext,
                                                                 normal,
                                                                 sdInfo_u_w_rowptr.data(),
                                                                 sdInfo_u_w_colind.data(),
                                                                 mom_uw_diff_ten_ext,
                                                                 &vel_grad_test_dS[i*nSpace]);
                    elementResidual_v[i] +=
		      (INT_BY_PARTS_PRESSURE==1 ? p_ext*vel_test_dS[i]*normal[1] : 0.) +
		      ck.ExteriorElementBoundaryFlux(flux_mom_v_adv_ext,vel_test_dS[i]) +
                      ck.ExteriorElementBoundaryFlux(flux_mom_vu_diff_ext,vel_test_dS[i])+
                      ck.ExteriorElementBoundaryFlux(flux_mom_vv_diff_ext,vel_test_dS[i])+
                      ck.ExteriorElementBoundaryFlux(flux_mom_vw_diff_ext,vel_test_dS[i])+
                      ck.ExteriorElementBoundaryDiffusionAdjoint(isDOFBoundary_u[ebNE_kb],
                                                                 isDiffusiveFluxBoundary_v[ebNE_kb],
                                                                 eb_adjoint_sigma,
                                                                 u_ext,
                                                                 bc_u_ext,
                                                                 normal,
                                                                 sdInfo_v_u_rowptr.data(),
                                                                 sdInfo_v_u_colind.data(),
                                                                 mom_vu_diff_ten_ext,
                                                                 &vel_grad_test_dS[i*nSpace])+
                      ck.ExteriorElementBoundaryDiffusionAdjoint(isDOFBoundary_v[ebNE_kb],
                                                                 isDiffusiveFluxBoundary_v[ebNE_kb],
                                                                 eb_adjoint_sigma,
                                                                 v_ext,
                                                                 bc_v_ext,
                                                                 normal,
                                                                 sdInfo_v_v_rowptr.data(),
                                                                 sdInfo_v_v_colind.data(),
                                                                 mom_vv_diff_ten_ext,
                                                                 &vel_grad_test_dS[i*nSpace])+
                      ck.ExteriorElementBoundaryDiffusionAdjoint(isDOFBoundary_w[ebNE_kb],
                                                                 isDiffusiveFluxBoundary_v[ebNE_kb],
                                                                 eb_adjoint_sigma,
                                                                 w_ext,
                                                                 bc_w_ext,
                                                                 normal,
                                                                 sdInfo_v_w_rowptr.data(),
                                                                 sdInfo_v_w_colind.data(),
                                                                 mom_vw_diff_ten_ext,
                                                                 &vel_grad_test_dS[i*nSpace]);

                    elementResidual_w[i] +=
		      (INT_BY_PARTS_PRESSURE==1 ? p_ext*vel_test_dS[i]*normal[2] : 0.) +
		      ck.ExteriorElementBoundaryFlux(flux_mom_w_adv_ext,vel_test_dS[i]) +
                      ck.ExteriorElementBoundaryFlux(flux_mom_wu_diff_ext,vel_test_dS[i])+
                      ck.ExteriorElementBoundaryFlux(flux_mom_wv_diff_ext,vel_test_dS[i])+
                      ck.ExteriorElementBoundaryFlux(flux_mom_ww_diff_ext,vel_test_dS[i])+
                      ck.ExteriorElementBoundaryDiffusionAdjoint(isDOFBoundary_u[ebNE_kb],
                                                                 isDiffusiveFluxBoundary_w[ebNE_kb],
                                                                 eb_adjoint_sigma,
                                                                 u_ext,
                                                                 bc_u_ext,
                                                                 normal,
                                                                 sdInfo_w_u_rowptr.data(),
                                                                 sdInfo_w_u_colind.data(),
                                                                 mom_wu_diff_ten_ext,
                                                                 &vel_grad_test_dS[i*nSpace])+
                      ck.ExteriorElementBoundaryDiffusionAdjoint(isDOFBoundary_v[ebNE_kb],
                                                                 isDiffusiveFluxBoundary_w[ebNE_kb],
                                                                 eb_adjoint_sigma,
                                                                 v_ext,
                                                                 bc_v_ext,
                                                                 normal,
                                                                 sdInfo_w_v_rowptr.data(),
                                                                 sdInfo_w_v_colind.data(),
                                                                 mom_wv_diff_ten_ext,
                                                                 &vel_grad_test_dS[i*nSpace])+
                      ck.ExteriorElementBoundaryDiffusionAdjoint(isDOFBoundary_w[ebNE_kb],
                                                                 isDiffusiveFluxBoundary_w[ebNE_kb],
                                                                 eb_adjoint_sigma,
                                                                 w_ext,
                                                                 bc_w_ext,
                                                                 normal,
                                                                 sdInfo_w_w_rowptr.data(),
                                                                 sdInfo_w_w_colind.data(),
                                                                 mom_ww_diff_ten_ext,
                                                                 &vel_grad_test_dS[i*nSpace]);
                  }//i
              }//kb
            //
            //update the element and global residual storage
            //
            for (int i=0;i<nDOF_test_element;i++)
              {
                int eN_i = eN*nDOF_test_element+i;

                /* elementResidual_p_save[eN_i] +=  elementResidual_p[i]; */
                /* mesh_volume_conservation_weak += elementResidual_mesh[i];               */
                /* globalResidual[offset_p+stride_p*p_l2g[eN_i]]+=elementResidual_p[i]; */
                globalResidual[offset_u+stride_u*vel_l2g[eN_i]]+=elementResidual_u[i];
                globalResidual[offset_v+stride_v*vel_l2g[eN_i]]+=elementResidual_v[i];
                globalResidual[offset_w+stride_w*vel_l2g[eN_i]]+=elementResidual_w[i];
              }//i
          }//ebNE
        gf.useExact=useExact;
        gf_s.useExact=useExact;
        /* std::cout<<"mesh volume conservation = "<<mesh_volume_conservation<<std::endl; */
        /* std::cout<<"mesh volume conservation weak = "<<mesh_volume_conservation_weak<<std::endl; */
        /* std::cout<<"mesh volume conservation err max= "<<mesh_volume_conservation_err_max<<std::endl; */
        /* std::cout<<"mesh volume conservation err max weak = "<<mesh_volume_conservation_err_max_weak<<std::endl; */
      }


      void calculateJacobian(arguments_dict& args,
                             bool useExact)
     {
        proteus::pyarray<double>& mesh_trial_ref = args.array<double>("mesh_trial_ref");
        proteus::pyarray<double>& mesh_grad_trial_ref = args.array<double>("mesh_grad_trial_ref");
        proteus::pyarray<double>& mesh_dof = args.array<double>("mesh_dof");
        proteus::pyarray<double>& mesh_velocity_dof = args.array<double>("mesh_velocity_dof");
        double MOVING_DOMAIN = args.scalar<double>("MOVING_DOMAIN");
        double PSTAB = args.scalar<double>("PSTAB");
        proteus::pyarray<int>& mesh_l2g = args.array<int>("mesh_l2g");
        proteus::pyarray<double>& x_ref = args.array<double>("x_ref");
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
        proteus::pyarray<double>& vel_hess_trial_ref = args.array<double>("vel_hess_trial_ref");
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
        int nElements_global = args.scalar<int>("nElements_global");
        int nElements_owned = args.scalar<int>("nElements_owned");
        int nElementBoundaries_global = args.scalar<int>("nElementBoundaries_global");
        int nElementBoundaries_owned = args.scalar<int>("nElementBoundaries_owned");
        int nNodes_owned = args.scalar<int>("nNodes_owned");
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
        double smagorinskyConstant = args.scalar<double>("smagorinskyConstant");
        int turbulenceClosureModel = args.scalar<int>("turbulenceClosureModel");
        double Ct_sge = args.scalar<double>("Ct_sge");
        double Cd_sge = args.scalar<double>("Cd_sge");
        double C_dg = args.scalar<double>("C_dg");
        double C_b = args.scalar<double>("C_b");
        const proteus::pyarray<double>& eps_solid = args.array<double>("eps_solid");
        const proteus::pyarray<double>& ebq_global_phi_solid = args.array<double>("ebq_global_phi_solid");
        const proteus::pyarray<double>& ebq_global_grad_phi_solid = args.array<double>("ebq_global_grad_phi_solid");
        const proteus::pyarray<double>& ebq_particle_velocity_solid = args.array<double>("ebq_particle_velocity_solid");
        proteus::pyarray<double>& phi_solid_nodes = args.array<double>("phi_solid_nodes");
        const proteus::pyarray<double>& phi_solid = args.array<double>("phi_solid");
        const proteus::pyarray<double>& q_velocity_solid = args.array<double>("q_velocity_solid");
        const proteus::pyarray<double>& q_velocityStar_solid = args.array<double>("q_velocityStar_solid");
        const proteus::pyarray<double>& q_vos = args.array<double>("q_vos");
        const proteus::pyarray<double>& q_dvos_dt = args.array<double>("q_dvos_dt");
        const proteus::pyarray<double>& q_grad_vos = args.array<double>("q_grad_vos");
        const proteus::pyarray<double>& q_dragAlpha = args.array<double>("q_dragAlpha");
        const proteus::pyarray<double>& q_dragBeta = args.array<double>("q_dragBeta");
        const proteus::pyarray<double>& q_mass_source = args.array<double>("q_mass_source");
        const proteus::pyarray<double>& q_turb_var_0 = args.array<double>("q_turb_var_0");
        const proteus::pyarray<double>& q_turb_var_1 = args.array<double>("q_turb_var_1");
        const proteus::pyarray<double>& q_turb_var_grad_0 = args.array<double>("q_turb_var_grad_0");
        proteus::pyarray<int>& p_l2g = args.array<int>("p_l2g");
        proteus::pyarray<int>& vel_l2g = args.array<int>("vel_l2g");
        proteus::pyarray<double>& p_dof = args.array<double>("p_dof");
        proteus::pyarray<double>& u_dof = args.array<double>("u_dof");
        proteus::pyarray<double>& v_dof = args.array<double>("v_dof");
        proteus::pyarray<double>& w_dof = args.array<double>("w_dof");
        proteus::pyarray<double>& g = args.array<double>("g");
        const double useVF = args.scalar<double>("useVF");
        proteus::pyarray<double>& vf = args.array<double>("vf");
        proteus::pyarray<double>& phi = args.array<double>("phi");
        proteus::pyarray<double>& phi_dof = args.array<double>("phi_dof");
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
        proteus::pyarray<int>& sdInfo_u_u_rowptr = args.array<int>("sdInfo_u_u_rowptr");
        proteus::pyarray<int>& sdInfo_u_u_colind = args.array<int>("sdInfo_u_u_colind");
        proteus::pyarray<int>& sdInfo_u_v_rowptr = args.array<int>("sdInfo_u_v_rowptr");
        proteus::pyarray<int>& sdInfo_u_v_colind = args.array<int>("sdInfo_u_v_colind");
        proteus::pyarray<int>& sdInfo_u_w_rowptr = args.array<int>("sdInfo_u_w_rowptr");
        proteus::pyarray<int>& sdInfo_u_w_colind = args.array<int>("sdInfo_u_w_colind");
        proteus::pyarray<int>& sdInfo_v_v_rowptr = args.array<int>("sdInfo_v_v_rowptr");
        proteus::pyarray<int>& sdInfo_v_v_colind = args.array<int>("sdInfo_v_v_colind");
        proteus::pyarray<int>& sdInfo_v_u_rowptr = args.array<int>("sdInfo_v_u_rowptr");
        proteus::pyarray<int>& sdInfo_v_u_colind = args.array<int>("sdInfo_v_u_colind");
        proteus::pyarray<int>& sdInfo_v_w_rowptr = args.array<int>("sdInfo_v_w_rowptr");
        proteus::pyarray<int>& sdInfo_v_w_colind = args.array<int>("sdInfo_v_w_colind");
        proteus::pyarray<int>& sdInfo_w_w_rowptr = args.array<int>("sdInfo_w_w_rowptr");
        proteus::pyarray<int>& sdInfo_w_w_colind = args.array<int>("sdInfo_w_w_colind");
        proteus::pyarray<int>& sdInfo_w_u_rowptr = args.array<int>("sdInfo_w_u_rowptr");
        proteus::pyarray<int>& sdInfo_w_u_colind = args.array<int>("sdInfo_w_u_colind");
        proteus::pyarray<int>& sdInfo_w_v_rowptr = args.array<int>("sdInfo_w_v_rowptr");
        proteus::pyarray<int>& sdInfo_w_v_colind = args.array<int>("sdInfo_w_v_colind");
        proteus::pyarray<int>& csrRowIndeces_p_p = args.array<int>("csrRowIndeces_p_p");
        proteus::pyarray<int>& csrColumnOffsets_p_p = args.array<int>("csrColumnOffsets_p_p");
        proteus::pyarray<int>& csrRowIndeces_p_u = args.array<int>("csrRowIndeces_p_u");
        proteus::pyarray<int>& csrColumnOffsets_p_u = args.array<int>("csrColumnOffsets_p_u");
        proteus::pyarray<int>& csrRowIndeces_p_v = args.array<int>("csrRowIndeces_p_v");
        proteus::pyarray<int>& csrColumnOffsets_p_v = args.array<int>("csrColumnOffsets_p_v");
        proteus::pyarray<int>& csrRowIndeces_p_w = args.array<int>("csrRowIndeces_p_w");
        proteus::pyarray<int>& csrColumnOffsets_p_w = args.array<int>("csrColumnOffsets_p_w");
        proteus::pyarray<int>& csrRowIndeces_u_p = args.array<int>("csrRowIndeces_u_p");
        proteus::pyarray<int>& csrColumnOffsets_u_p = args.array<int>("csrColumnOffsets_u_p");
        proteus::pyarray<int>& csrRowIndeces_u_u = args.array<int>("csrRowIndeces_u_u");
        proteus::pyarray<int>& csrColumnOffsets_u_u = args.array<int>("csrColumnOffsets_u_u");
        proteus::pyarray<int>& csrRowIndeces_u_v = args.array<int>("csrRowIndeces_u_v");
        proteus::pyarray<int>& csrColumnOffsets_u_v = args.array<int>("csrColumnOffsets_u_v");
        proteus::pyarray<int>& csrRowIndeces_u_w = args.array<int>("csrRowIndeces_u_w");
        proteus::pyarray<int>& csrColumnOffsets_u_w = args.array<int>("csrColumnOffsets_u_w");
        proteus::pyarray<int>& csrRowIndeces_v_p = args.array<int>("csrRowIndeces_v_p");
        proteus::pyarray<int>& csrColumnOffsets_v_p = args.array<int>("csrColumnOffsets_v_p");
        proteus::pyarray<int>& csrRowIndeces_v_u = args.array<int>("csrRowIndeces_v_u");
        proteus::pyarray<int>& csrColumnOffsets_v_u = args.array<int>("csrColumnOffsets_v_u");
        proteus::pyarray<int>& csrRowIndeces_v_v = args.array<int>("csrRowIndeces_v_v");
        proteus::pyarray<int>& csrColumnOffsets_v_v = args.array<int>("csrColumnOffsets_v_v");
        proteus::pyarray<int>& csrRowIndeces_v_w = args.array<int>("csrRowIndeces_v_w");
        proteus::pyarray<int>& csrColumnOffsets_v_w = args.array<int>("csrColumnOffsets_v_w");
        proteus::pyarray<int>& csrRowIndeces_w_p = args.array<int>("csrRowIndeces_w_p");
        proteus::pyarray<int>& csrColumnOffsets_w_p = args.array<int>("csrColumnOffsets_w_p");
        proteus::pyarray<int>& csrRowIndeces_w_u = args.array<int>("csrRowIndeces_w_u");
        proteus::pyarray<int>& csrColumnOffsets_w_u = args.array<int>("csrColumnOffsets_w_u");
        proteus::pyarray<int>& csrRowIndeces_w_v = args.array<int>("csrRowIndeces_w_v");
        proteus::pyarray<int>& csrColumnOffsets_w_v = args.array<int>("csrColumnOffsets_w_v");
        proteus::pyarray<int>& csrRowIndeces_w_w = args.array<int>("csrRowIndeces_w_w");
        proteus::pyarray<int>& csrColumnOffsets_w_w = args.array<int>("csrColumnOffsets_w_w");
        proteus::pyarray<double>& globalJacobian = args.array<double>("globalJacobian");
        int nExteriorElementBoundaries_global = args.scalar<int>("nExteriorElementBoundaries_global");
        proteus::pyarray<int>& exteriorElementBoundariesArray = args.array<int>("exteriorElementBoundariesArray");
        proteus::pyarray<int>& elementBoundariesArray = args.array<int>("elementBoundariesArray");
        proteus::pyarray<int>& elementBoundaryElementsArray = args.array<int>("elementBoundaryElementsArray");
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
        proteus::pyarray<int>& csrColumnOffsets_eb_p_p = args.array<int>("csrColumnOffsets_eb_p_p");
        proteus::pyarray<int>& csrColumnOffsets_eb_p_u = args.array<int>("csrColumnOffsets_eb_p_u");
        proteus::pyarray<int>& csrColumnOffsets_eb_p_v = args.array<int>("csrColumnOffsets_eb_p_v");
        proteus::pyarray<int>& csrColumnOffsets_eb_p_w = args.array<int>("csrColumnOffsets_eb_p_w");
        proteus::pyarray<int>& csrColumnOffsets_eb_u_p = args.array<int>("csrColumnOffsets_eb_u_p");
        proteus::pyarray<int>& csrColumnOffsets_eb_u_u = args.array<int>("csrColumnOffsets_eb_u_u");
        proteus::pyarray<int>& csrColumnOffsets_eb_u_v = args.array<int>("csrColumnOffsets_eb_u_v");
        proteus::pyarray<int>& csrColumnOffsets_eb_u_w = args.array<int>("csrColumnOffsets_eb_u_w");
        proteus::pyarray<int>& csrColumnOffsets_eb_v_p = args.array<int>("csrColumnOffsets_eb_v_p");
        proteus::pyarray<int>& csrColumnOffsets_eb_v_u = args.array<int>("csrColumnOffsets_eb_v_u");
        proteus::pyarray<int>& csrColumnOffsets_eb_v_v = args.array<int>("csrColumnOffsets_eb_v_v");
        proteus::pyarray<int>& csrColumnOffsets_eb_v_w = args.array<int>("csrColumnOffsets_eb_v_w");
        proteus::pyarray<int>& csrColumnOffsets_eb_w_p = args.array<int>("csrColumnOffsets_eb_w_p");
        proteus::pyarray<int>& csrColumnOffsets_eb_w_u = args.array<int>("csrColumnOffsets_eb_w_u");
        proteus::pyarray<int>& csrColumnOffsets_eb_w_v = args.array<int>("csrColumnOffsets_eb_w_v");
        proteus::pyarray<int>& csrColumnOffsets_eb_w_w = args.array<int>("csrColumnOffsets_eb_w_w");
        proteus::pyarray<int>& elementFlags = args.array<int>("elementFlags");
        int nParticles = args.scalar<int>("nParticles");
        double particle_epsFact = args.scalar<double>("particle_epsFact");
        double particle_alpha = args.scalar<double>("particle_alpha");
        double particle_beta = args.scalar<double>("particle_beta");
        double particle_penalty_constant = args.scalar<double>("particle_penalty_constant");
        proteus::pyarray<double>& particle_signed_distances = args.array<double>("particle_signed_distances");
        proteus::pyarray<double>& particle_signed_distance_normals = args.array<double>("particle_signed_distance_normals");
        proteus::pyarray<double>& particle_velocities = args.array<double>("particle_velocities");
        proteus::pyarray<double>& particle_centroids = args.array<double>("particle_centroids");
        double particle_nitsche = args.scalar<double>("particle_nitsche");
        int use_ball_as_particle = args.scalar<int>("use_ball_as_particle");
        proteus::pyarray<double>& ball_center = args.array<double>("ball_center");
        proteus::pyarray<double>& ball_radius = args.array<double>("ball_radius");
        proteus::pyarray<double>& ball_velocity = args.array<double>("ball_velocity");
        proteus::pyarray<double>& ball_angular_velocity = args.array<double>("ball_angular_velocity");
        int USE_SUPG = args.scalar<int>("USE_SUPG");
        int KILL_PRESSURE_TERM = args.scalar<int>("KILL_PRESSURE_TERM");
        double dt = args.scalar<double>("dt");
        int MATERIAL_PARAMETERS_AS_FUNCTION = args.scalar<int>("MATERIAL_PARAMETERS_AS_FUNCTION");
        proteus::pyarray<double>& density_as_function = args.array<double>("density_as_function");
        proteus::pyarray<double>& dynamic_viscosity_as_function = args.array<double>("dynamic_viscosity_as_function");
        proteus::pyarray<double>& ebqe_density_as_function = args.array<double>("ebqe_density_as_function");
        proteus::pyarray<double>& ebqe_dynamic_viscosity_as_function = args.array<double>("ebqe_dynamic_viscosity_as_function");
        int USE_SBM = args.scalar<int>("USE_SBM");
        int ARTIFICIAL_VISCOSITY = args.scalar<int>("ARTIFICIAL_VISCOSITY");
        proteus::pyarray<double>& uStar_dMatrix = args.array<double>("uStar_dMatrix");
        proteus::pyarray<double>& vStar_dMatrix = args.array<double>("vStar_dMatrix");
        proteus::pyarray<double>& wStar_dMatrix = args.array<double>("wStar_dMatrix");
        int numDOFs_1D = args.scalar<int>("numDOFs_1D");
        int offset_u = args.scalar<int>("offset_u");
        int offset_v = args.scalar<int>("offset_v");
        int offset_w = args.scalar<int>("offset_w");
        int stride_u = args.scalar<int>("stride_u");
        int stride_v = args.scalar<int>("stride_v");
        int stride_w = args.scalar<int>("stride_w");
        proteus::pyarray<int>& rowptr_1D = args.array<int>("rowptr_1D");
        proteus::pyarray<int>& colind_1D = args.array<int>("colind_1D");
        proteus::pyarray<int>& rowptr = args.array<int>("rowptr");
        proteus::pyarray<int>& colind = args.array<int>("colind");
        int INT_BY_PARTS_PRESSURE = args.scalar<int>("INT_BY_PARTS_PRESSURE");
        //
        //loop over elements to compute volume integrals and load them into the element Jacobians and global Jacobian
        //
        std::valarray<double> particle_surfaceArea(nParticles), particle_netForces(nParticles * 3), particle_netMoments(nParticles * 3);
        const int nQuadraturePoints_global(nElements_global*nQuadraturePoints_element);
        for(int eN=0;eN<nElements_global;eN++)
          {
            double eps_rho,eps_mu;
            double element_active=1.0;//value 1 is because it is ibm by default

            double  elementJacobian_p_p[nDOF_test_element][nDOF_trial_element],
              elementJacobian_p_u[nDOF_test_element][nDOF_trial_element],
              elementJacobian_p_v[nDOF_test_element][nDOF_trial_element],
              elementJacobian_p_w[nDOF_test_element][nDOF_trial_element],
              elementJacobian_u_p[nDOF_test_element][nDOF_trial_element],
              elementJacobian_u_u[nDOF_test_element][nDOF_trial_element],
              elementJacobian_u_v[nDOF_test_element][nDOF_trial_element],
              elementJacobian_u_w[nDOF_test_element][nDOF_trial_element],
              elementJacobian_v_p[nDOF_test_element][nDOF_trial_element],
              elementJacobian_v_u[nDOF_test_element][nDOF_trial_element],
              elementJacobian_v_v[nDOF_test_element][nDOF_trial_element],
              elementJacobian_v_w[nDOF_test_element][nDOF_trial_element],
              elementJacobian_w_p[nDOF_test_element][nDOF_trial_element],
              elementJacobian_w_u[nDOF_test_element][nDOF_trial_element],
              elementJacobian_w_v[nDOF_test_element][nDOF_trial_element],
              elementJacobian_w_w[nDOF_test_element][nDOF_trial_element];
            for (int i=0;i<nDOF_test_element;i++)
              for (int j=0;j<nDOF_trial_element;j++)
                {
                  elementJacobian_p_p[i][j]=0.0;
                  elementJacobian_p_u[i][j]=0.0;
                  elementJacobian_p_v[i][j]=0.0;
                  elementJacobian_p_w[i][j]=0.0;
                  elementJacobian_u_p[i][j]=0.0;
                  elementJacobian_u_u[i][j]=0.0;
                  elementJacobian_u_v[i][j]=0.0;
                  elementJacobian_u_w[i][j]=0.0;
                  elementJacobian_v_p[i][j]=0.0;
                  elementJacobian_v_u[i][j]=0.0;
                  elementJacobian_v_v[i][j]=0.0;
                  elementJacobian_v_w[i][j]=0.0;
                  elementJacobian_w_p[i][j]=0.0;
                  elementJacobian_w_u[i][j]=0.0;
                  elementJacobian_w_v[i][j]=0.0;
                  elementJacobian_w_w[i][j]=0.0;
                }
            if(USE_SBM>0)
            {
                //
                //detect cut cells
                //
                double _distance[nDOF_mesh_trial_element]={0.0};
                int pos_counter=0;
                for (int I=0;I<nDOF_mesh_trial_element;I++)
                {
                    if(use_ball_as_particle==1)
                    {
                        get_distance_to_ball(nParticles, ball_center.data(), ball_radius.data(),
                                mesh_dof[3*mesh_l2g[eN*nDOF_mesh_trial_element+I]+0],
                                mesh_dof[3*mesh_l2g[eN*nDOF_mesh_trial_element+I]+1],
                                mesh_dof[3*mesh_l2g[eN*nDOF_mesh_trial_element+I]+2],
                                _distance[I]);
                    }
                    else
                    {
                        _distance[I] = phi_solid_nodes[mesh_l2g[eN*nDOF_mesh_trial_element+I]];
                    }
                    if ( _distance[I] >= 0)
                        pos_counter++;
                }
                if (pos_counter == 3)//surrogate face
                {
                    element_active=0.0;
                    int opp_node=-1;
                    for (int I=0;I<nDOF_mesh_trial_element;I++)
                    {
                        if (_distance[I] < 0)
                            opp_node = I;
                    }
                    assert(opp_node >=0);
                    assert(opp_node <nDOF_mesh_trial_element);
                }
                else if (pos_counter == 4)// element is in fluid totally
                {
                    element_active=1.0;
                }
                else
                {
                    element_active=0.0;
                }
            }
            double element_phi[nDOF_mesh_trial_element], element_phi_s[nDOF_mesh_trial_element];
	    for (int j=0;j<nDOF_mesh_trial_element;j++)
	      {
		int eN_j = eN*nDOF_mesh_trial_element+j;
		element_phi[j] = phi_dof[p_l2g[eN_j]];
		element_phi_s[j] = phi_solid_nodes[p_l2g[eN_j]];
	      }
            double element_nodes[nDOF_mesh_trial_element*3];
	    for (int i=0;i<nDOF_mesh_trial_element;i++)
	      {
		int eN_i=eN*nDOF_mesh_trial_element+i;
                for(int I=0;I<3;I++)
                  element_nodes[i*3 + I] = mesh_dof[mesh_l2g[eN_i]*3 + I];
	      }//i
	    gf_s.calculate(element_phi_s, element_nodes, x_ref.data(), false);
            gf.calculate(element_phi, element_nodes, x_ref.data(), false);
            for  (int k=0;k<nQuadraturePoints_element;k++)
              {
                gf.set_quad(k);
                gf_s.set_quad(k);
                int eN_k = eN*nQuadraturePoints_element+k, //index to a scalar at a quadrature point
                  eN_k_nSpace = eN_k*nSpace,
                  eN_nDOF_trial_element = eN*nDOF_trial_element; //index to a vector at a quadrature point

                //declare local storage
                double p=0.0,u=0.0,v=0.0,w=0.0,
                  grad_p[nSpace],grad_u[nSpace],grad_v[nSpace],grad_w[nSpace],
                  hess_u[nSpace2],hess_v[nSpace2],hess_w[nSpace2],
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
                  mom_w_acc_t=0.0,
                  dmom_w_acc_w_t=0.0,
                  pdeResidual_p=0.0,
                  pdeResidual_u=0.0,
                  pdeResidual_v=0.0,
                  pdeResidual_w=0.0,
                  dpdeResidual_p_u[nDOF_trial_element],dpdeResidual_p_v[nDOF_trial_element],dpdeResidual_p_w[nDOF_trial_element],
                  dpdeResidual_u_p[nDOF_trial_element],dpdeResidual_u_u[nDOF_trial_element],
                  dpdeResidual_v_p[nDOF_trial_element],dpdeResidual_v_v[nDOF_trial_element],
                  dpdeResidual_w_p[nDOF_trial_element],dpdeResidual_w_w[nDOF_trial_element],
                  Lstar_u_p[nDOF_test_element],
                  Lstar_v_p[nDOF_test_element],
                  Lstar_w_p[nDOF_test_element],
                  Lstar_u_u[nDOF_test_element],
                  Lstar_v_v[nDOF_test_element],
                  Lstar_w_w[nDOF_test_element],
                  Lstar_p_u[nDOF_test_element],
                  Lstar_p_v[nDOF_test_element],
                  Lstar_p_w[nDOF_test_element],
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
                  vel_hess_trial[nDOF_trial_element*nSpace2],
                  dV,
                  p_test_dV[nDOF_test_element],vel_test_dV[nDOF_test_element],
                  p_grad_test_dV[nDOF_test_element*nSpace],vel_grad_test_dV[nDOF_test_element*nSpace],
                  x,y,z,xt,yt,zt,
                  //VRANS
                  porosity,
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
                dV = fabs(jacDet)*dV_ref[k];
                ck.calculateG(jacInv,G,G_dd_G,tr_G);
                //ck.calculateGScale(G,&normal_phi[eN_k_nSpace],h_phi);

                eps_rho = epsFact_rho*(useMetrics*h_phi+(1.0-useMetrics)*elementDiameter[eN]);
                eps_mu  = epsFact_mu *(useMetrics*h_phi+(1.0-useMetrics)*elementDiameter[eN]);
                const double particle_eps  = particle_epsFact*(useMetrics*h_phi+(1.0-useMetrics)*elementDiameter[eN]);

                //get the trial function gradients
                /* ck.gradTrialFromRef(&p_grad_trial_ref[k*nDOF_trial_element*nSpace],jacInv,p_grad_trial); */
                ck.gradTrialFromRef(&vel_grad_trial_ref[k*nDOF_trial_element*nSpace],jacInv,vel_grad_trial);
                ck.hessTrialFromRef(&vel_hess_trial_ref[k*nDOF_trial_element*nSpace2],jacInv,vel_hess_trial);
                //get the solution
                /* ck.valFromDOF(p_dof,&p_l2g[eN_nDOF_trial_element],&p_trial_ref[k*nDOF_trial_element],p); */
                p = q_p[eN_k];
                ck.valFromDOF(u_dof.data(),&vel_l2g[eN_nDOF_trial_element],&vel_trial_ref[k*nDOF_trial_element],u);
                ck.valFromDOF(v_dof.data(),&vel_l2g[eN_nDOF_trial_element],&vel_trial_ref[k*nDOF_trial_element],v);
                ck.valFromDOF(w_dof.data(),&vel_l2g[eN_nDOF_trial_element],&vel_trial_ref[k*nDOF_trial_element],w);
                //get the solution gradients
                /* ck.gradFromDOF(p_dof,&p_l2g[eN_nDOF_trial_element],p_grad_trial,grad_p); */
                for (int I=0;I<nSpace;I++)
                  grad_p[I] = q_grad_p[eN_k_nSpace+I];
                ck.gradFromDOF(u_dof.data(),&vel_l2g[eN_nDOF_trial_element],vel_grad_trial,grad_u);
                ck.gradFromDOF(v_dof.data(),&vel_l2g[eN_nDOF_trial_element],vel_grad_trial,grad_v);
                ck.gradFromDOF(w_dof.data(),&vel_l2g[eN_nDOF_trial_element],vel_grad_trial,grad_w);
                ck.hessFromDOF(u_dof.data(),&vel_l2g[eN_nDOF_trial_element],vel_hess_trial,hess_u);
                ck.hessFromDOF(v_dof.data(),&vel_l2g[eN_nDOF_trial_element],vel_hess_trial,hess_v);
                ck.hessFromDOF(w_dof.data(),&vel_l2g[eN_nDOF_trial_element],vel_hess_trial,hess_w);
                //precalculate test function products with integration weights
                for (int j=0;j<nDOF_trial_element;j++)
                  {
                    /* p_test_dV[j] = p_test_ref[k*nDOF_trial_element+j]*dV; */
                    vel_test_dV[j] = vel_test_ref[k*nDOF_trial_element+j]*dV;
                    for (int I=0;I<nSpace;I++)
                      {
                        /* p_grad_test_dV[j*nSpace+I]   = p_grad_trial[j*nSpace+I]*dV;//cek warning won't work for Petrov-Galerkin */
                        vel_grad_test_dV[j*nSpace+I] = vel_grad_trial[j*nSpace+I]*dV;//cek warning won't work for Petrov-Galerkin}
                      }
                  }
                //cek hack
                double div_mesh_velocity=0.0;
                int NDOF_MESH_TRIAL_ELEMENT=4;
                for (int j=0;j<NDOF_MESH_TRIAL_ELEMENT;j++)
                  {
                    int eN_j=eN*NDOF_MESH_TRIAL_ELEMENT+j;
                    div_mesh_velocity +=
                      mesh_velocity_dof[mesh_l2g[eN_j]*3+0]*vel_grad_trial[j*3+0] +
                      mesh_velocity_dof[mesh_l2g[eN_j]*3+1]*vel_grad_trial[j*3+1] +
                      mesh_velocity_dof[mesh_l2g[eN_j]*3+2]*vel_grad_trial[j*3+2];
                  }
                div_mesh_velocity = DM3*div_mesh_velocity + (1.0-DM3)*alphaBDF*(dV-q_dV_last[eN_k])/dV;
                //
                //VRANS
                porosity = 1.0 - q_vos[eN_k];
                //
                //
                //calculate pde coefficients and derivatives at quadrature points
                //
                double distance_to_omega_solid = phi_solid[eN_k];//computed in getResidual
                double eddy_viscosity(0.),rhoSave,nuSave;//not really interested in saving eddy_viscosity in jacobian
                evaluateCoefficients(eps_rho,
                                     eps_mu,
                                     particle_eps,
                                     sigma,
                                     rho_0,
                                     nu_0,
                                     rho_1,
                                     nu_1,
                                     elementDiameter[eN],
                                     smagorinskyConstant,
                                     turbulenceClosureModel,
                                     g.data(),
                                     useVF,
                                     vf[eN_k],
                                     phi[eN_k],
                                     &normal_phi[eN_k_nSpace],
                                     distance_to_omega_solid,
                                     kappa_phi[eN_k],
                                     //VRANS
                                     porosity,
                                     //
                                     p,
                                     grad_p,
                                     grad_u,
                                     grad_v,
                                     grad_w,
                                     u,
                                     v,
                                     w,
                                     q_velocity_sge[eN_k_nSpace+0],
                                     q_velocity_sge[eN_k_nSpace+1],
                                     q_velocity_sge[eN_k_nSpace+2],
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
                                     dmom_w_ham_grad_w,
                                     rhoSave,
                                     nuSave,
                                     KILL_PRESSURE_TERM,
                                     0,
                                     0., // mql: the force term doesn't play a role in the Jacobian
                                     0.,
                                     0.,
                                     MATERIAL_PARAMETERS_AS_FUNCTION,
                                     density_as_function[eN_k],
                                     dynamic_viscosity_as_function[eN_k],
                                     USE_SBM,
                                     x,y,z,
                                     use_ball_as_particle,
                                     ball_center.data(),
                                     ball_radius.data(),
                                     ball_velocity.data(),
                                     ball_angular_velocity.data(),
				     INT_BY_PARTS_PRESSURE);
                //VRANS
                mass_source = q_mass_source[eN_k];
                for (int I=0;I<nSpace;I++)
                  {
                    dmom_u_source[I] = 0.0;
                    dmom_v_source[I] = 0.0;
                    dmom_w_source[I] = 0.0;
                  }
                updateDarcyForchheimerTerms_Ergun(/* linearDragFactor, */
                                                  /* nonlinearDragFactor, */
                                                  /* porosity, */
                                                  /* meanGrainSize, */
                                                  q_dragAlpha[eN_k],
                                                  q_dragBeta[eN_k],
                                                  eps_rho,
                                                  eps_mu,
                                                  rho_0,
                                                  nu_0,
                                                  rho_1,
                                                  nu_1,
						  eddy_viscosity,
                                                  useVF,
                                                  vf[eN_k],
                                                  phi[eN_k],
                                                  u,
                                                  v,
                                                  w,
                                                  q_velocity_sge[eN_k_nSpace+0],
                                                  q_velocity_sge[eN_k_nSpace+1],
                                                  q_velocity_sge[eN_k_nSpace+2],
                                                  eps_solid[elementFlags[eN]],
                                                  porosity,
                                                  q_velocity_solid[eN_k_nSpace+0],
                                                  q_velocity_solid[eN_k_nSpace+1],
                                                  q_velocity_solid[eN_k_nSpace+2],
                                                  q_velocityStar_solid[eN_k_nSpace+0],
                                                  q_velocityStar_solid[eN_k_nSpace+1],
                                                  q_velocityStar_solid[eN_k_nSpace+2],
                                                  mom_u_source,
                                                  mom_v_source,
                                                  mom_w_source,
                                                  dmom_u_source,
                                                  dmom_v_source,
                                                  dmom_w_source,
                                                  q_grad_vos[eN_k_nSpace+0],
                                                  q_grad_vos[eN_k_nSpace+1],
                                                  q_grad_vos[eN_k_nSpace+2]);
                double C_particles=0.0;
                if(nParticles > 0 && USE_SBM==0)
                  updateSolidParticleTerms(eN < nElements_owned,
                                           particle_nitsche,
                                           dV,
                                           nParticles,
                                           nQuadraturePoints_global,
                                           &particle_signed_distances[eN_k],
                                           &particle_signed_distance_normals[eN_k_nSpace],
                                           &particle_velocities[eN_k_nSpace],
                                           particle_centroids.data(),
                                           use_ball_as_particle,
                                           ball_center.data(),
                                           ball_radius.data(),
                                           ball_velocity.data(),
                                           ball_angular_velocity.data(),
                                           porosity,
                                           particle_penalty_constant/h_phi,
                                           particle_alpha/h_phi,
                                           particle_beta/h_phi,
                                           eps_rho,
                                           eps_mu,
                                           rho_0,
                                           nu_0,
                                           rho_1,
                                           nu_1,
                                           useVF,
                                           vf[eN_k],
                                           phi[eN_k],
                                           x,
                                           y,
                                           z,
                                           p,
                                           u,
                                           v,
                                           w,
                                           q_velocity_sge[eN_k_nSpace+0],
                                           q_velocity_sge[eN_k_nSpace+1],
                                           q_velocity_sge[eN_k_nSpace+2],
                                           particle_eps,
                                           grad_u,
                                           grad_v,
                                           grad_w,
                                           mom_u_source,
                                           mom_v_source,
                                           mom_w_source,
                                           dmom_u_source,
                                           dmom_v_source,
                                           dmom_w_source,
                                           mom_u_adv,
                                           mom_v_adv,
                                           mom_w_adv,
                                           dmom_u_adv_u,
                                           dmom_v_adv_v,
                                           dmom_w_adv_w,
                                           mom_u_ham,
                                           dmom_u_ham_grad_u,
                                           mom_v_ham,
                                           dmom_v_ham_grad_v,
                                           mom_w_ham,
                                           dmom_w_ham_grad_w,
                                           &particle_netForces[0],
                                           &particle_netMoments[0],
                                           &particle_surfaceArea[0]);
                //Turbulence closure model
                if (turbulenceClosureModel >= 3)
                  {
                    const double c_mu = 0.09;//mwf hack
                    updateTurbulenceClosure(turbulenceClosureModel,
                                            eps_rho,
                                            eps_mu,
                                            rho_0,
                                            nu_0,
                                            rho_1,
                                            nu_1,
                                            useVF,
                                            vf[eN_k],
                                            phi[eN_k],
                                            porosity,
                                            c_mu, //mwf hack
                                            q_turb_var_0[eN_k],
                                            q_turb_var_1[eN_k],
                                            &q_turb_var_grad_0[eN_k_nSpace],
                                            eddy_viscosity,
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
                                            mom_w_source);

                  }
                //
                //
                //moving mesh
                //
                mom_u_adv[0] -= MOVING_DOMAIN*dmom_u_acc_u*mom_u_acc*xt; // multiply by rho*porosity. mql. CHECK.
                mom_u_adv[1] -= MOVING_DOMAIN*dmom_u_acc_u*mom_u_acc*yt;
                mom_u_adv[2] -= MOVING_DOMAIN*dmom_u_acc_u*mom_u_acc*zt;
                dmom_u_adv_u[0] -= MOVING_DOMAIN*dmom_u_acc_u*xt;
                dmom_u_adv_u[1] -= MOVING_DOMAIN*dmom_u_acc_u*yt;
                dmom_u_adv_u[2] -= MOVING_DOMAIN*dmom_u_acc_u*zt;

                mom_v_adv[0] -= MOVING_DOMAIN*dmom_v_acc_v*mom_v_acc*xt;
                mom_v_adv[1] -= MOVING_DOMAIN*dmom_v_acc_v*mom_v_acc*yt;
                mom_v_adv[2] -= MOVING_DOMAIN*dmom_v_acc_v*mom_v_acc*zt;
                dmom_v_adv_v[0] -= MOVING_DOMAIN*dmom_v_acc_v*xt;
                dmom_v_adv_v[1] -= MOVING_DOMAIN*dmom_v_acc_v*yt;
                dmom_v_adv_v[2] -= MOVING_DOMAIN*dmom_v_acc_v*zt;

                mom_w_adv[0] -= MOVING_DOMAIN*dmom_w_acc_w*mom_w_acc*xt;
                mom_w_adv[1] -= MOVING_DOMAIN*dmom_w_acc_w*mom_w_acc*yt;
                mom_w_adv[2] -= MOVING_DOMAIN*dmom_w_acc_w*mom_w_acc*zt;
                dmom_w_adv_w[0] -= MOVING_DOMAIN*dmom_w_acc_w*xt;
                dmom_w_adv_w[1] -= MOVING_DOMAIN*dmom_w_acc_w*yt;
                dmom_w_adv_w[2] -= MOVING_DOMAIN*dmom_w_acc_w*zt;
                //
                //calculate time derivatives
                //
                ck.bdf(alphaBDF,
                       q_mom_u_acc_beta_bdf[eN_k]*q_dV_last[eN_k]/dV,
                       mom_u_acc,
                       dmom_u_acc_u,
                       mom_u_acc_t,
                       dmom_u_acc_u_t);
                ck.bdf(alphaBDF,
                       q_mom_v_acc_beta_bdf[eN_k]*q_dV_last[eN_k]/dV,
                       mom_v_acc,
                       dmom_v_acc_v,
                       mom_v_acc_t,
                       dmom_v_acc_v_t);
                ck.bdf(alphaBDF,
                       q_mom_w_acc_beta_bdf[eN_k]*q_dV_last[eN_k]/dV,
                       mom_w_acc,
                       dmom_w_acc_w,
                       mom_w_acc_t,
                       dmom_w_acc_w_t);

                mom_u_acc_t *= dmom_u_acc_u; //multiply by porosity*rho. mql. CHECK.
                mom_v_acc_t *= dmom_v_acc_v;
                mom_w_acc_t *= dmom_w_acc_w;

                //
                //calculate subgrid error contribution to the Jacobian (strong residual, adjoint, jacobian of strong residual)
                //
                dmom_adv_sge[0] = dmom_u_acc_u*(q_velocity_sge[eN_k_nSpace+0] - MOVING_DOMAIN*xt);
                dmom_adv_sge[1] = dmom_u_acc_u*(q_velocity_sge[eN_k_nSpace+1] - MOVING_DOMAIN*yt);
                dmom_adv_sge[2] = dmom_u_acc_u*(q_velocity_sge[eN_k_nSpace+2] - MOVING_DOMAIN*zt);
                //
                //calculate strong residual
                //
                pdeResidual_p =
                  ck.Mass_strong(-q_dvos_dt[eN_k]) + // mql. CHECK.
                  ck.Advection_strong(dmass_adv_u,grad_u) +
                  ck.Advection_strong(dmass_adv_v,grad_v) +
                  ck.Advection_strong(dmass_adv_w,grad_w) +
                  DM2*MOVING_DOMAIN*ck.Reaction_strong(alphaBDF*(dV-q_dV_last[eN_k])/dV - div_mesh_velocity) +
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

                pdeResidual_w = ck.Mass_strong(mom_w_acc_t) +
                  ck.Advection_strong(dmom_adv_sge,grad_w) +
                  ck.Hamiltonian_strong(dmom_w_ham_grad_p,grad_p) +
                  ck.Reaction_strong(mom_w_source) -
                  ck.Reaction_strong(w*div_mesh_velocity);

                //calculate the Jacobian of strong residual
                for (int j=0;j<nDOF_trial_element;j++)
                  {
                    int j_nSpace = j*nSpace;
                    dpdeResidual_p_u[j]=ck.AdvectionJacobian_strong(dmass_adv_u,&vel_grad_trial[j_nSpace]);
                    dpdeResidual_p_v[j]=ck.AdvectionJacobian_strong(dmass_adv_v,&vel_grad_trial[j_nSpace]);
                    dpdeResidual_p_w[j]=ck.AdvectionJacobian_strong(dmass_adv_w,&vel_grad_trial[j_nSpace]);

                    dpdeResidual_u_p[j]=ck.HamiltonianJacobian_strong(dmom_u_ham_grad_p,&p_grad_trial[j_nSpace]);
                    dpdeResidual_u_u[j]=ck.MassJacobian_strong(dmom_u_acc_u_t,vel_trial_ref[k*nDOF_trial_element+j]) +
                      ck.AdvectionJacobian_strong(dmom_adv_sge,&vel_grad_trial[j_nSpace]) -
                      ck.ReactionJacobian_strong(div_mesh_velocity,vel_trial_ref[k*nDOF_trial_element+j]);

                    dpdeResidual_v_p[j]=ck.HamiltonianJacobian_strong(dmom_v_ham_grad_p,&p_grad_trial[j_nSpace]);
                    dpdeResidual_v_v[j]=ck.MassJacobian_strong(dmom_v_acc_v_t,vel_trial_ref[k*nDOF_trial_element+j]) +
                      ck.AdvectionJacobian_strong(dmom_adv_sge,&vel_grad_trial[j_nSpace]) -
                      ck.ReactionJacobian_strong(div_mesh_velocity,vel_trial_ref[k*nDOF_trial_element+j]);

                    dpdeResidual_w_p[j]=ck.HamiltonianJacobian_strong(dmom_w_ham_grad_p,&p_grad_trial[j_nSpace]);
                    dpdeResidual_w_w[j]=ck.MassJacobian_strong(dmom_w_acc_w_t,vel_trial_ref[k*nDOF_trial_element+j]) +
                      ck.AdvectionJacobian_strong(dmom_adv_sge,&vel_grad_trial[j_nSpace]) -
                      ck.ReactionJacobian_strong(div_mesh_velocity,vel_trial_ref[k*nDOF_trial_element+j]);

                    //VRANS account for drag terms, diagonal only here ... decide if need off diagonal terms too
                    dpdeResidual_u_u[j]+= ck.ReactionJacobian_strong(dmom_u_source[0],vel_trial_ref[k*nDOF_trial_element+j]);
                    dpdeResidual_v_v[j]+= ck.ReactionJacobian_strong(dmom_v_source[1],vel_trial_ref[k*nDOF_trial_element+j]);
                    dpdeResidual_w_w[j]+= ck.ReactionJacobian_strong(dmom_w_source[2],vel_trial_ref[k*nDOF_trial_element+j]);
                    //
                  }
                //calculate tau and tau*Res
                //cek debug
                double tmpR=dmom_u_acc_u_t + dmom_u_source[0];
                calculateSubgridError_tau(hFactor,
                                          elementDiameter[eN],
                                          tmpR,//dmom_u_acc_u_t,
                                          dmom_u_acc_u,
                                          dmom_adv_sge,
                                          mom_uu_diff_ten[1],
                                          dmom_u_ham_grad_p[0],
                                          tau_v0,
                                          tau_p0,
                                          q_cfl[eN_k]);

                calculateSubgridError_tau(Ct_sge,Cd_sge,
                                          G,G_dd_G,tr_G,
                                          tmpR,//dmom_u_acc_u_t,
                                          dmom_adv_sge,
                                          mom_uu_diff_ten[1],
                                          dmom_u_ham_grad_p[0],
                                          tau_v1,
                                          tau_p1,
                                          q_cfl[eN_k]);


                tau_v = useMetrics*tau_v1+(1.0-useMetrics)*tau_v0;
                tau_p = KILL_PRESSURE_TERM == 1 ? 0. : PSTAB*(useMetrics*tau_p1+(1.0-useMetrics)*tau_p0);
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
                dmom_adv_star[0] =  dmom_u_acc_u*(q_velocity_sge[eN_k_nSpace+0] - MOVING_DOMAIN*xt + useRBLES*subgridError_u);
                dmom_adv_star[1] =  dmom_u_acc_u*(q_velocity_sge[eN_k_nSpace+1] - MOVING_DOMAIN*yt + useRBLES*subgridError_v);
                dmom_adv_star[2] =  dmom_u_acc_u*(q_velocity_sge[eN_k_nSpace+2] - MOVING_DOMAIN*zt + useRBLES*subgridError_w);

                //calculate the adjoint times the test functions
                for (int i=0;i<nDOF_test_element;i++)
                  {
                    int i_nSpace = i*nSpace;
                    Lstar_u_p[i]=ck.Advection_adjoint(dmass_adv_u,&p_grad_test_dV[i_nSpace]);
                    Lstar_v_p[i]=ck.Advection_adjoint(dmass_adv_v,&p_grad_test_dV[i_nSpace]);
                    Lstar_w_p[i]=ck.Advection_adjoint(dmass_adv_w,&p_grad_test_dV[i_nSpace]);
                    Lstar_u_u[i]=ck.Advection_adjoint(dmom_adv_star,&vel_grad_test_dV[i_nSpace]);
                    Lstar_v_v[i]=ck.Advection_adjoint(dmom_adv_star,&vel_grad_test_dV[i_nSpace]);
                    Lstar_w_w[i]=ck.Advection_adjoint(dmom_adv_star,&vel_grad_test_dV[i_nSpace]);
                    Lstar_p_u[i]=ck.Hamiltonian_adjoint(dmom_u_ham_grad_p,&vel_grad_test_dV[i_nSpace]);
                    Lstar_p_v[i]=ck.Hamiltonian_adjoint(dmom_v_ham_grad_p,&vel_grad_test_dV[i_nSpace]);
                    Lstar_p_w[i]=ck.Hamiltonian_adjoint(dmom_w_ham_grad_p,&vel_grad_test_dV[i_nSpace]);
                    //VRANS account for drag terms, diagonal only here ... decide if need off diagonal terms too
                    Lstar_u_u[i]+=ck.Reaction_adjoint(dmom_u_source[0],vel_test_dV[i]);
                    Lstar_v_v[i]+=ck.Reaction_adjoint(dmom_v_source[1],vel_test_dV[i]);
                    Lstar_w_w[i]+=ck.Reaction_adjoint(dmom_w_source[2],vel_test_dV[i]);
                  }

                // Assumes non-lagged subgrid velocity
                dmom_u_adv_u[0] += dmom_u_acc_u*(useRBLES*subgridError_u);
                dmom_u_adv_u[1] += dmom_u_acc_u*(useRBLES*subgridError_v);
                dmom_u_adv_u[2] += dmom_u_acc_u*(useRBLES*subgridError_w);

                dmom_v_adv_v[0] += dmom_u_acc_u*(useRBLES*subgridError_u);
                dmom_v_adv_v[1] += dmom_u_acc_u*(useRBLES*subgridError_v);
                dmom_v_adv_v[2] += dmom_u_acc_u*(useRBLES*subgridError_w);

                dmom_w_adv_w[0] += dmom_u_acc_u*(useRBLES*subgridError_u);
                dmom_w_adv_w[1] += dmom_u_acc_u*(useRBLES*subgridError_v);
                dmom_w_adv_w[2] += dmom_u_acc_u*(useRBLES*subgridError_w);

                // SURFACE TENSION //
                double unit_normal[nSpace];
                double norm_grad_phi = 0.;
                for (int I=0;I<nSpace;I++)
                  norm_grad_phi += normal_phi[eN_k_nSpace+I]*normal_phi[eN_k_nSpace+I];
                norm_grad_phi = std::sqrt(norm_grad_phi) + 1E-10;
                for (int I=0;I<nSpace;I++)
                  unit_normal[I] = normal_phi[eN_k_nSpace+I]/norm_grad_phi;
                double delta = gf.D(eps_mu,phi[eN_k]); //use eps_rho instead?
                double vel_tgrad_test_i[nSpace], vel_tgrad_test_j[nSpace];
                // END OF SURFACE TENSION //

                //cek todo add RBLES terms consistent to residual modifications or ignore the partials w.r.t the additional RBLES terms
                for(int i=0;i<nDOF_test_element;i++)
                  {
                    int i_nSpace = i*nSpace;
                    calculateTangentialGradient(unit_normal,
                                                &vel_grad_trial[i_nSpace],
                                                vel_tgrad_test_i);
                    for(int j=0;j<nDOF_trial_element;j++)
                      {
                        int j_nSpace = j*nSpace;
                        calculateTangentialGradient(unit_normal,
                                                    &vel_grad_trial[j_nSpace],
                                                    vel_tgrad_test_j);

                        /* elementJacobian_p_p[i][j] += ck.SubgridErrorJacobian(dsubgridError_u_p[j],Lstar_u_p[i]) +  */
                        /*   ck.SubgridErrorJacobian(dsubgridError_v_p[j],Lstar_v_p[i]);// +  */
                        /*   /\* ck.SubgridErrorJacobian(dsubgridError_w_p[j],Lstar_w_p[i]);  *\/ */

                        /* elementJacobian_p_u[i][j] += ck.AdvectionJacobian_weak(dmass_adv_u,vel_trial_ref[k*nDOF_trial_element+j],&p_grad_test_dV[i_nSpace]) +  */
                        /*   ck.SubgridErrorJacobian(dsubgridError_u_u[j],Lstar_u_p[i]);  */
                        /* elementJacobian_p_v[i][j] += ck.AdvectionJacobian_weak(dmass_adv_v,vel_trial_ref[k*nDOF_trial_element+j],&p_grad_test_dV[i_nSpace]) +  */
                        /*   ck.SubgridErrorJacobian(dsubgridError_v_v[j],Lstar_v_p[i]);  */
                        /* elementJacobian_p_w[i][j] += ck.AdvectionJacobian_weak(dmass_adv_w,vel_trial_ref[k*nDOF_trial_element+j],&p_grad_test_dV[i_nSpace]) +  */
                        /*      ck.SubgridErrorJacobian(dsubgridError_w_w[j],Lstar_w_p[i]);  */

                        /* elementJacobian_u_p[i][j] += ck.HamiltonianJacobian_weak(dmom_u_ham_grad_p,&p_grad_trial[j_nSpace],vel_test_dV[i]) +  */
                        /*   ck.SubgridErrorJacobian(dsubgridError_u_p[j],Lstar_u_u[i]);  */
                        elementJacobian_u_u[i][j] +=
                          ck.MassJacobian_weak(dmom_u_acc_u_t,vel_trial_ref[k*nDOF_trial_element+j],vel_test_dV[i]) +
                          ck.HamiltonianJacobian_weak(dmom_u_ham_grad_u,&vel_grad_trial[j_nSpace],vel_test_dV[i]) +
                          ck.AdvectionJacobian_weak(dmom_u_adv_u,vel_trial_ref[k*nDOF_trial_element+j],&vel_grad_test_dV[i_nSpace]) +
                          ck.SimpleDiffusionJacobian_weak(sdInfo_u_u_rowptr.data(),sdInfo_u_u_colind.data(),mom_uu_diff_ten,&vel_grad_trial[j_nSpace],&vel_grad_test_dV[i_nSpace]) +
                          //VRANS
                          ck.ReactionJacobian_weak(dmom_u_source[0],vel_trial_ref[k*nDOF_trial_element+j],vel_test_dV[i]) +
                          //
                          //ck.SubgridErrorJacobian(dsubgridError_p_u[j],Lstar_p_u[i]) +
                          USE_SUPG*ck.SubgridErrorJacobian(dsubgridError_u_u[j],Lstar_u_u[i]) +
                          ck.NumericalDiffusionJacobian(q_numDiff_u_last[eN_k],&vel_grad_trial[j_nSpace],&vel_grad_test_dV[i_nSpace]) +
                          // surface tension
                          ck.NumericalDiffusion(dt*delta*sigma*dV,
                                                vel_tgrad_test_i,
                                                vel_tgrad_test_j);

                        elementJacobian_u_v[i][j] +=
                          ck.AdvectionJacobian_weak(dmom_u_adv_v,vel_trial_ref[k*nDOF_trial_element+j],&vel_grad_test_dV[i_nSpace]) +
                          ck.SimpleDiffusionJacobian_weak(sdInfo_u_v_rowptr.data(),sdInfo_u_v_colind.data(),mom_uv_diff_ten,&vel_grad_trial[j_nSpace],&vel_grad_test_dV[i_nSpace]) +
                          //VRANS
                          ck.ReactionJacobian_weak(dmom_u_source[1],vel_trial_ref[k*nDOF_trial_element+j],vel_test_dV[i])
                          //+ck.SubgridErrorJacobian(dsubgridError_p_v[j],Lstar_p_u[i])
                          ;
                        elementJacobian_u_w[i][j] +=
                          ck.AdvectionJacobian_weak(dmom_u_adv_w,vel_trial_ref[k*nDOF_trial_element+j],&vel_grad_test_dV[i_nSpace]) +
                          ck.SimpleDiffusionJacobian_weak(sdInfo_u_w_rowptr.data(),sdInfo_u_w_colind.data(),mom_uw_diff_ten,&vel_grad_trial[j_nSpace],&vel_grad_test_dV[i_nSpace]) +
                          //VRANS
                          ck.ReactionJacobian_weak(dmom_u_source[2],vel_trial_ref[k*nDOF_trial_element+j],vel_test_dV[i])
                          //+ck.SubgridErrorJacobian(dsubgridError_p_w[j],Lstar_p_u[i])
                          ;

                        /* elementJacobian_v_p[i][j] +=
                           ck.HamiltonianJacobian_weak(dmom_v_ham_grad_p,&p_grad_trial[j_nSpace],vel_test_dV[i]) +  */
                        /*   ck.SubgridErrorJacobian(dsubgridError_v_p[j],Lstar_v_v[i]);  */
                        elementJacobian_v_u[i][j] +=
                          ck.AdvectionJacobian_weak(dmom_v_adv_u,vel_trial_ref[k*nDOF_trial_element+j],&vel_grad_test_dV[i_nSpace]) +
                          ck.SimpleDiffusionJacobian_weak(sdInfo_v_u_rowptr.data(),sdInfo_v_u_colind.data(),mom_vu_diff_ten,&vel_grad_trial[j_nSpace],&vel_grad_test_dV[i_nSpace]) +
                          //VRANS
                          ck.ReactionJacobian_weak(dmom_v_source[0],vel_trial_ref[k*nDOF_trial_element+j],vel_test_dV[i])
                          //+ck.SubgridErrorJacobian(dsubgridError_p_u[j],Lstar_p_v[i])
                          ;
                        elementJacobian_v_v[i][j] +=
                          ck.MassJacobian_weak(dmom_v_acc_v_t,vel_trial_ref[k*nDOF_trial_element+j],vel_test_dV[i]) +
                          ck.HamiltonianJacobian_weak(dmom_v_ham_grad_v,&vel_grad_trial[j_nSpace],vel_test_dV[i]) +
                          ck.AdvectionJacobian_weak(dmom_v_adv_v,vel_trial_ref[k*nDOF_trial_element+j],&vel_grad_test_dV[i_nSpace]) +
                          ck.SimpleDiffusionJacobian_weak(sdInfo_v_v_rowptr.data(),sdInfo_v_v_colind.data(),mom_vv_diff_ten,&vel_grad_trial[j_nSpace],&vel_grad_test_dV[i_nSpace]) +
                          //VRANS
                          ck.ReactionJacobian_weak(dmom_v_source[1],vel_trial_ref[k*nDOF_trial_element+j],vel_test_dV[i]) +
                          //
                          //ck.SubgridErrorJacobian(dsubgridError_p_v[j],Lstar_p_v[i]) +
                          USE_SUPG*ck.SubgridErrorJacobian(dsubgridError_v_v[j],Lstar_v_v[i]) +
                          ck.NumericalDiffusionJacobian(q_numDiff_v_last[eN_k],&vel_grad_trial[j_nSpace],&vel_grad_test_dV[i_nSpace]) +
                          // surface tension
                          ck.NumericalDiffusion(dt*delta*sigma*dV,
                                                vel_tgrad_test_i,
                                                vel_tgrad_test_j);

                        elementJacobian_v_w[i][j] +=
                          ck.AdvectionJacobian_weak(dmom_v_adv_w,vel_trial_ref[k*nDOF_trial_element+j],&vel_grad_test_dV[i_nSpace]) +
                          ck.SimpleDiffusionJacobian_weak(sdInfo_v_w_rowptr.data(),sdInfo_v_w_colind.data(),mom_vw_diff_ten,&vel_grad_trial[j_nSpace],&vel_grad_test_dV[i_nSpace]) +
                          //VRANS
                          ck.ReactionJacobian_weak(dmom_v_source[2],vel_trial_ref[k*nDOF_trial_element+j],vel_test_dV[i])
                          //+ck.SubgridErrorJacobian(dsubgridError_p_w[j],Lstar_p_v[i])
                          ;

                        /* elementJacobian_w_p[i][j] += ck.HamiltonianJacobian_weak(dmom_w_ham_grad_p,&p_grad_trial[j_nSpace],vel_test_dV[i]) +  */
                        /*   ck.SubgridErrorJacobian(dsubgridError_w_p[j],Lstar_w_w[i]);  */
                        elementJacobian_w_u[i][j] +=
                          ck.AdvectionJacobian_weak(dmom_w_adv_u,vel_trial_ref[k*nDOF_trial_element+j],&vel_grad_test_dV[i_nSpace]) +
                          ck.SimpleDiffusionJacobian_weak(sdInfo_w_u_rowptr.data(),sdInfo_w_u_colind.data(),mom_wu_diff_ten,&vel_grad_trial[j_nSpace],&vel_grad_test_dV[i_nSpace]) +
                          //VRANS
                          ck.ReactionJacobian_weak(dmom_w_source[0],vel_trial_ref[k*nDOF_trial_element+j],vel_test_dV[i])
                          //+ck.SubgridErrorJacobian(dsubgridError_p_u[j],Lstar_p_w[i])
                          ;
                        elementJacobian_w_v[i][j] +=
                          ck.AdvectionJacobian_weak(dmom_w_adv_v,vel_trial_ref[k*nDOF_trial_element+j],&vel_grad_test_dV[i_nSpace]) +
                          ck.SimpleDiffusionJacobian_weak(sdInfo_w_v_rowptr.data(),sdInfo_w_v_colind.data(),mom_wv_diff_ten,&vel_grad_trial[j_nSpace],&vel_grad_test_dV[i_nSpace]) +
                          //VRANS
                          ck.ReactionJacobian_weak(dmom_w_source[1],vel_trial_ref[k*nDOF_trial_element+j],vel_test_dV[i])
                          //+ck.SubgridErrorJacobian(dsubgridError_p_v[j],Lstar_p_w[i])
                          ;
                        elementJacobian_w_w[i][j] +=
                          ck.MassJacobian_weak(dmom_w_acc_w_t,vel_trial_ref[k*nDOF_trial_element+j],vel_test_dV[i]) +
                          ck.HamiltonianJacobian_weak(dmom_w_ham_grad_w,&vel_grad_trial[j_nSpace],vel_test_dV[i]) +
                          ck.AdvectionJacobian_weak(dmom_w_adv_w,vel_trial_ref[k*nDOF_trial_element+j],&vel_grad_test_dV[i_nSpace]) +
                          ck.SimpleDiffusionJacobian_weak(sdInfo_w_w_rowptr.data(),sdInfo_w_w_colind.data(),mom_ww_diff_ten,&vel_grad_trial[j_nSpace],&vel_grad_test_dV[i_nSpace]) +
                          //VRANS
                          ck.ReactionJacobian_weak(dmom_w_source[2],vel_trial_ref[k*nDOF_trial_element+j],vel_test_dV[i]) +
                          //
                          //ck.SubgridErrorJacobian(dsubgridError_p_w[j],Lstar_p_w[i]) +
                          USE_SUPG*ck.SubgridErrorJacobian(dsubgridError_w_w[j],Lstar_w_w[i]) +
                          ck.NumericalDiffusionJacobian(q_numDiff_w_last[eN_k],&vel_grad_trial[j_nSpace],&vel_grad_test_dV[i_nSpace]) +
                          // surface tension
                          ck.NumericalDiffusion(dt*delta*sigma*dV,
                                                vel_tgrad_test_i,
                                                vel_tgrad_test_j);
                      }//j
                  }//i
              }//k
            //
            //load into element Jacobian into global Jacobian
            //
            for (int i=0;i<nDOF_test_element;i++)
              {
                int eN_i = eN*nDOF_test_element+i;
                for (int j=0;j<nDOF_trial_element;j++)
                  {
                    int eN_i_j = eN_i*nDOF_trial_element+j;
                    /* globalJacobian[csrRowIndeces_p_p[eN_i] + csrColumnOffsets_p_p[eN_i_j]] += elementJacobian_p_p[i][j]; */
                    /* globalJacobian[csrRowIndeces_p_u[eN_i] + csrColumnOffsets_p_u[eN_i_j]] += elementJacobian_p_u[i][j]; */
                    /* globalJacobian[csrRowIndeces_p_v[eN_i] + csrColumnOffsets_p_v[eN_i_j]] += elementJacobian_p_v[i][j]; */
                    /* globalJacobian[csrRowIndeces_p_w[eN_i] + csrColumnOffsets_p_w[eN_i_j]] += elementJacobian_p_w[i][j]; */

                    /* globalJacobian[csrRowIndeces_u_p[eN_i] + csrColumnOffsets_u_p[eN_i_j]] += elementJacobian_u_p[i][j]; */
                    globalJacobian[csrRowIndeces_u_u[eN_i] + csrColumnOffsets_u_u[eN_i_j]] += element_active*elementJacobian_u_u[i][j];
                    globalJacobian[csrRowIndeces_u_v[eN_i] + csrColumnOffsets_u_v[eN_i_j]] += element_active*elementJacobian_u_v[i][j];
                    globalJacobian[csrRowIndeces_u_w[eN_i] + csrColumnOffsets_u_w[eN_i_j]] += element_active*elementJacobian_u_w[i][j];

                    /* globalJacobian[csrRowIndeces_v_p[eN_i] + csrColumnOffsets_v_p[eN_i_j]] += elementJacobian_v_p[i][j]; */
                    globalJacobian[csrRowIndeces_v_u[eN_i] + csrColumnOffsets_v_u[eN_i_j]] += element_active*elementJacobian_v_u[i][j];
                    globalJacobian[csrRowIndeces_v_v[eN_i] + csrColumnOffsets_v_v[eN_i_j]] += element_active*elementJacobian_v_v[i][j];
                    globalJacobian[csrRowIndeces_v_w[eN_i] + csrColumnOffsets_v_w[eN_i_j]] += element_active*elementJacobian_v_w[i][j];

                    /* globalJacobian[csrRowIndeces_w_p[eN_i] + csrColumnOffsets_w_p[eN_i_j]] += elementJacobian_w_p[i][j]; */
                    globalJacobian[csrRowIndeces_w_u[eN_i] + csrColumnOffsets_w_u[eN_i_j]] += element_active*elementJacobian_w_u[i][j];
                    globalJacobian[csrRowIndeces_w_v[eN_i] + csrColumnOffsets_w_v[eN_i_j]] += element_active*elementJacobian_w_v[i][j];
                    globalJacobian[csrRowIndeces_w_w[eN_i] + csrColumnOffsets_w_w[eN_i_j]] += element_active*elementJacobian_w_w[i][j];
                  }//j
              }//i
          }//elements

	// loop in DOFs for discrete upwinding
	if (ARTIFICIAL_VISCOSITY==3 || ARTIFICIAL_VISCOSITY==4)
	  {
	    int ij=0;
	    for (int i=0; i<numDOFs_1D; i++)
	      {
		// global index for each component
		int u_gi = offset_u+stride_u*i;
		int v_gi = offset_v+stride_v*i;
		int w_gi = offset_w+stride_w*i;

		// pointer to first entry in the ith row for each component
		int u_ith_row_ptr = rowptr[u_gi];
		int v_ith_row_ptr = rowptr[v_gi];
		int w_ith_row_ptr = rowptr[w_gi];

		// number of DOFs in the ith row (of the small matrix dMatrix)
		int numDOFs_ith_row = rowptr_1D[i+1]-rowptr_1D[i];
		for (int counter = 0; counter < numDOFs_ith_row; counter++)
		  {
		    // ij pointer for each component
		    int uu_ij = u_ith_row_ptr + (offset_u + counter*stride_u);
		    int vv_ij = v_ith_row_ptr + (offset_v + counter*stride_v);
		    int ww_ij = w_ith_row_ptr + (offset_w + counter*stride_w);

		    // read ij component of dissipative matrix
		    double uStar_dij = uStar_dMatrix[ij];
		    double vStar_dij = vStar_dMatrix[ij];
		    double wStar_dij = wStar_dMatrix[ij];

		    // update global Jacobian
		    globalJacobian[uu_ij] -= uStar_dij;
		    globalJacobian[vv_ij] -= vStar_dij;
		    globalJacobian[ww_ij] -= wStar_dij;

		    // update ij
		    ij++;
		  }
	      }
	  }

        //
        //loop over surrogate boundary
        //
        if(USE_SBM>0)
        {
            for (int ebN_s=0;ebN_s < surrogate_boundaries.size();ebN_s++)
            {
                int ebN = surrogate_boundaries[ebN_s],
                        eN = elementBoundaryElementsArray[ebN*2+surrogate_boundary_elements[ebN_s]],
                        ebN_local = elementBoundaryLocalElementBoundariesArray[ebN*2+surrogate_boundary_elements[ebN_s]],
                        eN_nDOF_trial_element = eN*nDOF_trial_element;
                double eps_rho,eps_mu;
                for  (int kb=0;kb<nQuadraturePoints_elementBoundary;kb++)
                  {
                    int ebN_kb = ebN*nQuadraturePoints_elementBoundary+kb,
                      ebN_kb_nSpace = ebN_kb*nSpace,
                      ebN_local_kb = ebN_local*nQuadraturePoints_elementBoundary+kb,
                      ebN_local_kb_nSpace = ebN_local_kb*nSpace;

                    double u_ext=0.0,
                            v_ext=0.0,
                            w_ext=0.0,
                            bc_u_ext=0.0,
                            bc_v_ext=0.0,
                            bc_w_ext=0.0,
                            grad_u_ext[nSpace],
                            grad_v_ext[nSpace],
                            grad_w_ext[nSpace],
                            jac_ext[nSpace*nSpace],
                            jacDet_ext,
                            jacInv_ext[nSpace*nSpace],
                            boundaryJac[nSpace*(nSpace-1)],
                            metricTensor[(nSpace-1)*(nSpace-1)],
                            metricTensorDetSqrt,
                            vel_grad_trial_trace[nDOF_trial_element*nSpace],
                            dS,
                            vel_test_dS[nDOF_test_element],
                            normal[3],
                            x_ext,y_ext,z_ext,xt_ext,yt_ext,zt_ext,integralScaling,
                            vel_grad_test_dS[nDOF_trial_element*nSpace],
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
                    dS = metricTensorDetSqrt*dS_ref[kb];
                    ck.calculateG(jacInv_ext,G,G_dd_G,tr_G);
                    //compute shape and solution information
                    //shape
                    ck.gradTrialFromRef(&vel_grad_trial_trace_ref[ebN_local_kb_nSpace*nDOF_trial_element],jacInv_ext,vel_grad_trial_trace);
                    //solution and gradients
                    ck.valFromDOF(u_dof.data(),&vel_l2g[eN_nDOF_trial_element],&vel_trial_trace_ref[ebN_local_kb*nDOF_test_element],u_ext);
                    ck.valFromDOF(v_dof.data(),&vel_l2g[eN_nDOF_trial_element],&vel_trial_trace_ref[ebN_local_kb*nDOF_test_element],v_ext);
                    ck.valFromDOF(w_dof.data(),&vel_l2g[eN_nDOF_trial_element],&vel_trial_trace_ref[ebN_local_kb*nDOF_test_element],w_ext);

                    ck.gradFromDOF(u_dof.data(),&vel_l2g[eN_nDOF_trial_element],vel_grad_trial_trace,grad_u_ext);
                    ck.gradFromDOF(v_dof.data(),&vel_l2g[eN_nDOF_trial_element],vel_grad_trial_trace,grad_v_ext);
                    ck.gradFromDOF(w_dof.data(),&vel_l2g[eN_nDOF_trial_element],vel_grad_trial_trace,grad_w_ext);
                    //precalculate test function products with integration weights
                    for (int j=0;j<nDOF_trial_element;j++)
                      {
                        vel_test_dS[j] = vel_test_trace_ref[ebN_local_kb*nDOF_test_element+j]*dS;
                        for (int I=0;I<nSpace;I++)
                            vel_grad_test_dS[j*nSpace+I] = vel_grad_trial_trace[j*nSpace+I]*dS;
                    }
                    //
                    //load the boundary values
                    //
                    ck.calculateGScale(G,normal,h_penalty);
                    penalty = h_penalty;
                    //
                    //update the global Jacobian from the flux Jacobian
                    //
                    double dist;
                    double distance[3], P_normal[3], P_tangent[3]={0.0}; // distance vector, normal and tangent of the physical boundary
                    if(use_ball_as_particle==1)
                    {
                        get_distance_to_ball(nParticles,ball_center.data(),ball_radius.data(),
                                             x_ext,y_ext,z_ext,
                                             dist);
                        get_normal_to_ith_ball(nParticles,ball_center.data(),ball_radius.data(),
                                               surrogate_boundary_particle[ebN_s],
                                               x_ext,y_ext,z_ext,
                                               P_normal[0],P_normal[1],P_normal[2]);
                        get_velocity_to_ith_ball(nParticles,ball_center.data(),ball_radius.data(),
                                                 ball_velocity.data(),ball_angular_velocity.data(),
                                                 surrogate_boundary_particle[ebN_s],
                                                 x_ext-dist*P_normal[0],//corresponding point on the boundary of the particle
                                                 y_ext-dist*P_normal[1],
                                                 z_ext-dist*P_normal[2],
                                                 bc_u_ext,bc_v_ext,bc_w_ext);

                    }
                    else
                    {
                        dist = ebq_global_phi_solid[ebN_kb];
                        P_normal[0] = ebq_global_grad_phi_solid[ebN_kb*nSpace+0];
                        P_normal[1] = ebq_global_grad_phi_solid[ebN_kb*nSpace+1];
                        P_normal[2] = ebq_global_grad_phi_solid[ebN_kb*nSpace+2];
                        bc_u_ext = ebq_particle_velocity_solid [ebN_kb*nSpace+0];
                        bc_v_ext = ebq_particle_velocity_solid [ebN_kb*nSpace+1];
                        bc_w_ext = ebq_particle_velocity_solid [ebN_kb*nSpace+2];
                    }
                    distance[0] = -P_normal[0]*dist;//distance=vector from \tilde{x} to x. It holds also when dist<0.0
                    distance[1] = -P_normal[1]*dist;
                    distance[2] = -P_normal[2]*dist;
//                    P_tangent[0] = -P_normal[1];
//                    P_tangent[1] = P_normal[0];
//                    double tx = P_tangent[0] ; double ty = P_tangent[1];
                    if(get_dot_product(P_normal,normal)>0)
                    {
                        //                        std::cout<<"YYPDB: some normal direction has wrong sign\n";
                        for(int i=0;i<3;++i)
                        {
                            normal[i] *= -1.0;
                        }
                    }
                    assert(h_penalty>0.0);
                    if (h_penalty < std::abs(dist))
                        h_penalty = std::abs(dist);
                    //hack: this won't work for two-phase flow, need mixture viscosity
                    double visco = nu_0*rho_0;
                    double C_adim = C_sbm*visco/h_penalty;
                    double beta_adim = beta_sbm*h_penalty*visco;

                    double res[3];
                    const double zero_vec[3]={0.,0.,0.};
                    for (int i=0;i<nDOF_test_element;i++)
                      {
                        int eN_i = eN*nDOF_test_element+i;
                        double phi_i = vel_test_dS[i];
                        double* grad_phi_i = &vel_grad_test_dS[i*nSpace+0];
                        const double grad_phi_i_dot_d = get_dot_product(grad_phi_i,distance);
                        for (int j=0;j<nDOF_trial_element;j++)
                          {
                            int ebN_i_j = ebN*4*nDOF_test_X_trial_element
                                                    + surrogate_boundary_elements[ebN_s]*2*nDOF_test_X_trial_element
                                                    + surrogate_boundary_elements[ebN_s]*nDOF_test_X_trial_element
                                                    + i*nDOF_trial_element
                                                    + j;

                            double phi_j = vel_test_dS[j]/dS;//since phi_i has dS
                            const double grad_phi_j[3]={vel_grad_test_dS[j*nSpace+0]/dS,
                                                    vel_grad_test_dS[j*nSpace+1]/dS,
                                                    vel_grad_test_dS[j*nSpace+2]/dS};
                            const double grad_phi_j_dot_d = get_dot_product(grad_phi_j,distance);

                            // Classical Nitsche
                            // (1)
                            globalJacobian[csrRowIndeces_u_u[eN_i] + csrColumnOffsets_eb_u_u[ebN_i_j]] +=
                              phi_i*phi_j*C_adim;
                            globalJacobian[csrRowIndeces_v_v[eN_i] + csrColumnOffsets_eb_v_v[ebN_i_j]] +=
                              phi_i*phi_j*C_adim;
                            globalJacobian[csrRowIndeces_w_w[eN_i] + csrColumnOffsets_eb_w_w[ebN_i_j]] +=
                              phi_i*phi_j*C_adim;
                            // (2)
                            get_symmetric_gradient_dot_vec(grad_phi_j,zero_vec,zero_vec,normal,res);
                            globalJacobian[csrRowIndeces_u_u[eN_i] + csrColumnOffsets_eb_u_u[ebN_i_j]] -=
                                    visco * phi_i * res[0];
                            globalJacobian[csrRowIndeces_u_v[eN_i] + csrColumnOffsets_eb_u_v[ebN_i_j]] -=
                                    visco * phi_i * res[1];
                            globalJacobian[csrRowIndeces_u_w[eN_i] + csrColumnOffsets_eb_u_w[ebN_i_j]] -=
                                    visco * phi_i * res[2];

                            get_symmetric_gradient_dot_vec(zero_vec,grad_phi_j,zero_vec,normal,res);
                            globalJacobian[csrRowIndeces_v_u[eN_i] + csrColumnOffsets_eb_v_u[ebN_i_j]] -=
                                    visco * phi_i * res[0] ;
                            globalJacobian[csrRowIndeces_v_v[eN_i] + csrColumnOffsets_eb_v_v[ebN_i_j]] -=
                                    visco * phi_i * res[1];
                            globalJacobian[csrRowIndeces_v_w[eN_i] + csrColumnOffsets_eb_v_w[ebN_i_j]] -=
                                    visco * phi_i * res[2];

                            get_symmetric_gradient_dot_vec(zero_vec,zero_vec,grad_phi_j,normal,res);
                            globalJacobian[csrRowIndeces_w_u[eN_i] + csrColumnOffsets_eb_w_u[ebN_i_j]] -=
                                    visco * phi_i * res[0] ;
                            globalJacobian[csrRowIndeces_w_v[eN_i] + csrColumnOffsets_eb_w_v[ebN_i_j]] -=
                                    visco * phi_i * res[1];
                            globalJacobian[csrRowIndeces_w_w[eN_i] + csrColumnOffsets_eb_w_w[ebN_i_j]] -=
                                    visco * phi_i * res[2];

                            // (3)
                            get_symmetric_gradient_dot_vec(grad_phi_i,zero_vec,zero_vec,normal,res);
                            globalJacobian[csrRowIndeces_u_u[eN_i] + csrColumnOffsets_eb_u_u[ebN_i_j]] -=
                                    visco * phi_j * res[0];
                            globalJacobian[csrRowIndeces_u_v[eN_i] + csrColumnOffsets_eb_u_v[ebN_i_j]] -=
                                    visco * phi_j * res[1];
                            globalJacobian[csrRowIndeces_u_w[eN_i] + csrColumnOffsets_eb_u_w[ebN_i_j]] -=
                                    visco * phi_j * res[2];

                            get_symmetric_gradient_dot_vec(zero_vec,grad_phi_i,zero_vec,normal,res);
                            globalJacobian[csrRowIndeces_v_u[eN_i] + csrColumnOffsets_eb_v_u[ebN_i_j]] -=
                                    visco * phi_j * res[0] ;
                            globalJacobian[csrRowIndeces_v_v[eN_i] + csrColumnOffsets_eb_v_v[ebN_i_j]] -=
                                    visco * phi_j * res[1];
                            globalJacobian[csrRowIndeces_v_w[eN_i] + csrColumnOffsets_eb_v_w[ebN_i_j]] -=
                                    visco * phi_j * res[2];

                            get_symmetric_gradient_dot_vec(zero_vec,zero_vec,grad_phi_i,normal,res);
                            globalJacobian[csrRowIndeces_w_u[eN_i] + csrColumnOffsets_eb_w_u[ebN_i_j]] -=
                                    visco * phi_j * res[0] ;
                            globalJacobian[csrRowIndeces_w_v[eN_i] + csrColumnOffsets_eb_w_v[ebN_i_j]] -=
                                    visco * phi_j * res[1];
                            globalJacobian[csrRowIndeces_w_w[eN_i] + csrColumnOffsets_eb_w_w[ebN_i_j]] -=
                                    visco * phi_j * res[2];

                            // (4)
                            globalJacobian[csrRowIndeces_u_u[eN_i] + csrColumnOffsets_eb_u_u[ebN_i_j]] +=
                                    C_adim*grad_phi_i_dot_d*phi_j;
                            globalJacobian[csrRowIndeces_v_v[eN_i] + csrColumnOffsets_eb_v_v[ebN_i_j]] +=
                                    C_adim*grad_phi_i_dot_d*phi_j;
                            globalJacobian[csrRowIndeces_w_w[eN_i] + csrColumnOffsets_eb_w_w[ebN_i_j]] +=
                                    C_adim*grad_phi_i_dot_d*phi_j;
                            // (5)
                            globalJacobian[csrRowIndeces_u_u[eN_i] + csrColumnOffsets_eb_u_u[ebN_i_j]] +=
                                    C_adim*grad_phi_i_dot_d*grad_phi_j_dot_d;
                            globalJacobian[csrRowIndeces_v_v[eN_i] + csrColumnOffsets_eb_v_v[ebN_i_j]] +=
                                    C_adim*grad_phi_i_dot_d*grad_phi_j_dot_d;
                            globalJacobian[csrRowIndeces_w_w[eN_i] + csrColumnOffsets_eb_w_w[ebN_i_j]] +=
                                    C_adim*grad_phi_i_dot_d*grad_phi_j_dot_d;
                            // (6)
                            globalJacobian[csrRowIndeces_u_u[eN_i] + csrColumnOffsets_eb_u_u[ebN_i_j]] +=
                                    C_adim*grad_phi_j_dot_d*phi_i;
                            globalJacobian[csrRowIndeces_v_v[eN_i] + csrColumnOffsets_eb_v_v[ebN_i_j]] +=
                                    C_adim*grad_phi_j_dot_d*phi_i;
                            globalJacobian[csrRowIndeces_w_w[eN_i] + csrColumnOffsets_eb_w_w[ebN_i_j]] +=
                                    C_adim*grad_phi_j_dot_d*phi_i;
                            // (7)
                            get_symmetric_gradient_dot_vec(grad_phi_i,zero_vec,zero_vec,normal,res);
                            globalJacobian[csrRowIndeces_u_u[eN_i] + csrColumnOffsets_eb_u_u[ebN_i_j]] -=
                                    visco * grad_phi_j_dot_d * res[0];
                            globalJacobian[csrRowIndeces_u_v[eN_i] + csrColumnOffsets_eb_u_v[ebN_i_j]] -=
                                    visco * grad_phi_j_dot_d * res[1];
                            globalJacobian[csrRowIndeces_u_w[eN_i] + csrColumnOffsets_eb_u_w[ebN_i_j]] -=
                                    visco * grad_phi_j_dot_d * res[2];

                            get_symmetric_gradient_dot_vec(zero_vec,grad_phi_i,zero_vec,normal,res);
                            globalJacobian[csrRowIndeces_v_u[eN_i] + csrColumnOffsets_eb_v_u[ebN_i_j]] -=
                                    visco * grad_phi_j_dot_d * res[0] ;
                            globalJacobian[csrRowIndeces_v_v[eN_i] + csrColumnOffsets_eb_v_v[ebN_i_j]] -=
                                    visco * grad_phi_j_dot_d * res[1];
                            globalJacobian[csrRowIndeces_v_w[eN_i] + csrColumnOffsets_eb_v_w[ebN_i_j]] -=
                                    visco * grad_phi_j_dot_d * res[2];

                            get_symmetric_gradient_dot_vec(zero_vec,zero_vec,grad_phi_i,normal,res);
                            globalJacobian[csrRowIndeces_w_u[eN_i] + csrColumnOffsets_eb_w_u[ebN_i_j]] -=
                                    visco * grad_phi_j_dot_d * res[0] ;
                            globalJacobian[csrRowIndeces_w_v[eN_i] + csrColumnOffsets_eb_w_v[ebN_i_j]] -=
                                    visco * grad_phi_j_dot_d * res[1];
                            globalJacobian[csrRowIndeces_w_w[eN_i] + csrColumnOffsets_eb_w_w[ebN_i_j]] -=
                                    visco * grad_phi_j_dot_d * res[2];

                            // the penalization on the tangential derivative
                            // B < Gw t , (Gu - GuD) t >
//                            // diag
//                            globalJacobian[csrRowIndeces_u_u[eN_i] + csrColumnOffsets_eb_u_u[ebN_i_j]] +=
//                                    beta_adim*(tx*Gxphi_i+ ty*Gyphi_i)*(tx*Gxphi_j+ ty*Gyphi_j);
//                            globalJacobian[csrRowIndeces_v_v[eN_i] + csrColumnOffsets_eb_v_v[ebN_i_j]] +=
//                                    beta_adim*(tx*Gxphi_i+ ty*Gyphi_i)*(tx*Gxphi_j+ ty*Gyphi_j);
                            //                        // extra diag
                            //                        globalJacobian[csrRowIndeces_u_v[eN_i] + csrColumnOffsets_eb_u_v[ebN_i_j]] +=
                            //                          beta_adim*tx*(Gxphi_i*ty*Gxphi_j + Gyphi_i*ty*Gyphi_j);
                            //                        globalJacobian[csrRowIndeces_v_u[eN_i] + csrColumnOffsets_eb_v_u[ebN_i_j]] +=
                            //                          beta_adim*ty*(Gxphi_i*tx*Gxphi_j + Gyphi_i*tx*Gyphi_j);

                        }//j
                    }//i

                }//kb
            }//ebN_s
        }//if USE_SBM
        //
        //loop over exterior element boundaries to compute the surface integrals and load them into the global Jacobian
        //
        //exact generalized function integration not implemented for boundaries yet
        gf.useExact=false;
        gf_s.useExact=false;
        for (int ebNE = 0; ebNE < nExteriorElementBoundaries_global; ebNE++)
          {
            int ebN = exteriorElementBoundariesArray[ebNE],
              eN  = elementBoundaryElementsArray[ebN*2+0],
              eN_nDOF_trial_element = eN*nDOF_trial_element,
              ebN_local = elementBoundaryLocalElementBoundariesArray[ebN*2+0];
            double eps_rho,eps_mu;
            for  (int kb=0;kb<nQuadraturePoints_elementBoundary;kb++)
              {
                int ebNE_kb = ebNE*nQuadraturePoints_elementBoundary+kb,
                  ebNE_kb_nSpace = ebNE_kb*nSpace,
                  ebN_local_kb = ebN_local*nQuadraturePoints_elementBoundary+kb,
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
                  fluxJacobian_p_p[nDOF_trial_element],
                  fluxJacobian_p_u[nDOF_trial_element],
                  fluxJacobian_p_v[nDOF_trial_element],
                  fluxJacobian_p_w[nDOF_trial_element],
                  fluxJacobian_u_p[nDOF_trial_element],
                  fluxJacobian_u_u[nDOF_trial_element],
                  fluxJacobian_u_v[nDOF_trial_element],
                  fluxJacobian_u_w[nDOF_trial_element],
                  fluxJacobian_v_p[nDOF_trial_element],
                  fluxJacobian_v_u[nDOF_trial_element],
                  fluxJacobian_v_v[nDOF_trial_element],
                  fluxJacobian_v_w[nDOF_trial_element],
                  fluxJacobian_w_p[nDOF_trial_element],
                  fluxJacobian_w_u[nDOF_trial_element],
                  fluxJacobian_w_v[nDOF_trial_element],
                  fluxJacobian_w_w[nDOF_trial_element],
                  jac_ext[nSpace*nSpace],
                  jacDet_ext,
                  jacInv_ext[nSpace*nSpace],
                  boundaryJac[nSpace*(nSpace-1)],
                  metricTensor[(nSpace-1)*(nSpace-1)],
                  metricTensorDetSqrt,
                  p_grad_trial_trace[nDOF_trial_element*nSpace],
                  vel_grad_trial_trace[nDOF_trial_element*nSpace],
                  dS,
                  p_test_dS[nDOF_test_element],
                  vel_test_dS[nDOF_test_element],
                  normal[3],
                  x_ext,y_ext,z_ext,xt_ext,yt_ext,zt_ext,integralScaling,
                  vel_grad_test_dS[nDOF_trial_element*nSpace],
                  //VRANS
                  porosity_ext,
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
                //dS = ((1.0-MOVING_DOMAIN)*metricTensorDetSqrt + MOVING_DOMAIN*integralScaling)*dS_ref[kb];
                dS = metricTensorDetSqrt*dS_ref[kb];
                ck.calculateG(jacInv_ext,G,G_dd_G,tr_G);
                ck.calculateGScale(G,&ebqe_normal_phi_ext[ebNE_kb_nSpace],h_phi);

                eps_rho = epsFact_rho*(useMetrics*h_phi+(1.0-useMetrics)*elementDiameter[eN]);
                eps_mu  = epsFact_mu *(useMetrics*h_phi+(1.0-useMetrics)*elementDiameter[eN]);
                const double particle_eps = particle_epsFact * (useMetrics * h_phi + (1.0 - useMetrics) * elementDiameter[eN]);

                //compute shape and solution information
                //shape
                /* ck.gradTrialFromRef(&p_grad_trial_trace_ref[ebN_local_kb_nSpace*nDOF_trial_element],jacInv_ext,p_grad_trial_trace); */
                ck.gradTrialFromRef(&vel_grad_trial_trace_ref[ebN_local_kb_nSpace*nDOF_trial_element],jacInv_ext,vel_grad_trial_trace);
                //solution and gradients
                /* ck.valFromDOF(p_dof,&p_l2g[eN_nDOF_trial_element],&p_trial_trace_ref[ebN_local_kb*nDOF_test_element],p_ext); */
                p_ext = ebqe_p[ebNE_kb];
                ck.valFromDOF(u_dof.data(),&vel_l2g[eN_nDOF_trial_element],&vel_trial_trace_ref[ebN_local_kb*nDOF_test_element],u_ext);
                ck.valFromDOF(v_dof.data(),&vel_l2g[eN_nDOF_trial_element],&vel_trial_trace_ref[ebN_local_kb*nDOF_test_element],v_ext);
                ck.valFromDOF(w_dof.data(),&vel_l2g[eN_nDOF_trial_element],&vel_trial_trace_ref[ebN_local_kb*nDOF_test_element],w_ext);
                /* ck.gradFromDOF(p_dof,&p_l2g[eN_nDOF_trial_element],p_grad_trial_trace,grad_p_ext); */
                for (int I=0;I<nSpace;I++)
                  grad_p_ext[I] = ebqe_grad_p[ebNE_kb_nSpace+I];
                ck.gradFromDOF(u_dof.data(),&vel_l2g[eN_nDOF_trial_element],vel_grad_trial_trace,grad_u_ext);
                ck.gradFromDOF(v_dof.data(),&vel_l2g[eN_nDOF_trial_element],vel_grad_trial_trace,grad_v_ext);
                ck.gradFromDOF(w_dof.data(),&vel_l2g[eN_nDOF_trial_element],vel_grad_trial_trace,grad_w_ext);
                //precalculate test function products with integration weights
                for (int j=0;j<nDOF_trial_element;j++)
                  {
                    /* p_test_dS[j] = p_test_trace_ref[ebN_local_kb*nDOF_test_element+j]*dS; */
                    vel_test_dS[j] = vel_test_trace_ref[ebN_local_kb*nDOF_test_element+j]*dS;
                    for (int I=0;I<nSpace;I++)
                      vel_grad_test_dS[j*nSpace+I] = vel_grad_trial_trace[j*nSpace+I]*dS;//cek hack, using trial
                  }
                //
                //load the boundary values
                //
                bc_p_ext = isDOFBoundary_p[ebNE_kb]*ebqe_bc_p_ext[ebNE_kb]+(1-isDOFBoundary_p[ebNE_kb])*p_ext;
                //bc values at moving boundaries are specified relative to boundary motion so we need to add it here
                bc_u_ext = isDOFBoundary_u[ebNE_kb]*(ebqe_bc_u_ext[ebNE_kb] + MOVING_DOMAIN*xt_ext) + (1-isDOFBoundary_u[ebNE_kb])*u_ext;
                bc_v_ext = isDOFBoundary_v[ebNE_kb]*(ebqe_bc_v_ext[ebNE_kb] + MOVING_DOMAIN*yt_ext) + (1-isDOFBoundary_v[ebNE_kb])*v_ext;
                bc_w_ext = isDOFBoundary_w[ebNE_kb]*(ebqe_bc_w_ext[ebNE_kb] + MOVING_DOMAIN*zt_ext) + (1-isDOFBoundary_w[ebNE_kb])*w_ext;
                //VRANS
                porosity_ext = 1.0 - ebqe_vos_ext[ebNE_kb];
                //
                //calculate the internal and external trace of the pde coefficients
                //
                double distance_to_omega_solid = 1e10;
                if (use_ball_as_particle == 1)
                {
                  get_distance_to_ball(nParticles, ball_center.data(), ball_radius.data(), x_ext, y_ext, z_ext, distance_to_omega_solid);
                }
                else
                {
                  for (int i = 0; i < nParticles; i++)
                  {
                    double distance_to_i_th_solid = ebq_global_phi_solid[i * nElementBoundaries_global * nQuadraturePoints_elementBoundary + ebNE_kb];
                    distance_to_omega_solid = (distance_to_i_th_solid < distance_to_omega_solid)?distance_to_i_th_solid:distance_to_omega_solid;
                  }
                }
                double eddy_viscosity_ext(0.),bc_eddy_viscosity_ext(0.),rhoSave, nuSave;//not interested in saving boundary eddy viscosity for now
                evaluateCoefficients(eps_rho,
                                     eps_mu,
                                     particle_eps,
                                     sigma,
                                     rho_0,
                                     nu_0,
                                     rho_1,
                                     nu_1,
                                     elementDiameter[eN],
                                     smagorinskyConstant,
                                     turbulenceClosureModel,
                                     g.data(),
                                     useVF,
                                     ebqe_vf_ext[ebNE_kb],
                                     ebqe_phi_ext[ebNE_kb],
                                     &ebqe_normal_phi_ext[ebNE_kb_nSpace],
                                     distance_to_omega_solid,
                                     ebqe_kappa_phi_ext[ebNE_kb],
                                     //VRANS
                                     porosity_ext,
                                     //
                                     p_ext,
                                     grad_p_ext,
                                     grad_u_ext,
                                     grad_v_ext,
                                     grad_w_ext,
                                     u_ext,
                                     v_ext,
                                     w_ext,
                                     ebqe_velocity_star[ebNE_kb_nSpace+0],
                                     ebqe_velocity_star[ebNE_kb_nSpace+1],
                                     ebqe_velocity_star[ebNE_kb_nSpace+2],
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
                                     dmom_w_ham_grad_w_ext,
                                     rhoSave,
                                     nuSave,
                                     KILL_PRESSURE_TERM,
                                     0,
                                     0., // mql: zero force term at boundary
                                     0.,
                                     0.,
                                     MATERIAL_PARAMETERS_AS_FUNCTION,
                                     ebqe_density_as_function[ebNE_kb],
                                     ebqe_dynamic_viscosity_as_function[ebNE_kb],
                                     USE_SBM,
                                     x_ext,y_ext,z_ext,
                                     use_ball_as_particle,
                                     ball_center.data(),
                                     ball_radius.data(),
                                     ball_velocity.data(),
                                     ball_angular_velocity.data(),
				     INT_BY_PARTS_PRESSURE);
                evaluateCoefficients(eps_rho,
                                     eps_mu,
                                     particle_eps,
                                     sigma,
                                     rho_0,
                                     nu_0,
                                     rho_1,
                                     nu_1,
                                     elementDiameter[eN],
                                     smagorinskyConstant,
                                     turbulenceClosureModel,
                                     g.data(),
                                     useVF,
                                     bc_ebqe_vf_ext[ebNE_kb],
                                     bc_ebqe_phi_ext[ebNE_kb],
                                     &ebqe_normal_phi_ext[ebNE_kb_nSpace],
                                     distance_to_omega_solid,
                                     ebqe_kappa_phi_ext[ebNE_kb],
                                     //VRANS
                                     porosity_ext,
                                     //
                                     bc_p_ext,
                                     grad_p_ext,
                                     grad_u_ext,
                                     grad_v_ext,
                                     grad_w_ext,
                                     bc_u_ext,
                                     bc_v_ext,
                                     bc_w_ext,
                                     ebqe_velocity_star[ebNE_kb_nSpace+0],
                                     ebqe_velocity_star[ebNE_kb_nSpace+1],
                                     ebqe_velocity_star[ebNE_kb_nSpace+2],
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
                                     bc_dmom_w_ham_grad_w_ext,
                                     rhoSave,
                                     nuSave,
                                     KILL_PRESSURE_TERM,
                                     0,
                                     0., // mql: zero force term at boundary
                                     0.,
                                     0.,
                                     MATERIAL_PARAMETERS_AS_FUNCTION,
                                     ebqe_density_as_function[ebNE_kb],
                                     ebqe_dynamic_viscosity_as_function[ebNE_kb],
                                     USE_SBM,
                                     x_ext,y_ext,z_ext,
                                     use_ball_as_particle,
                                     ball_center.data(),
                                     ball_radius.data(),
                                     ball_velocity.data(),
                                     ball_angular_velocity.data(),
				     INT_BY_PARTS_PRESSURE);
                //Turbulence closure model
                if (turbulenceClosureModel >= 3)
                  {
                    const double turb_var_grad_0_dummy[3] = {0.,0.,0.};
                    const double c_mu = 0.09;//mwf hack
                    updateTurbulenceClosure(turbulenceClosureModel,
                                            eps_rho,
                                            eps_mu,
                                            rho_0,
                                            nu_0,
                                            rho_1,
                                            nu_1,
                                            useVF,
                                            ebqe_vf_ext[ebNE_kb],
                                            ebqe_phi_ext[ebNE_kb],
                                            porosity_ext,
                                            c_mu, //mwf hack
                                            ebqe_turb_var_0[ebNE_kb],
                                            ebqe_turb_var_1[ebNE_kb],
                                            turb_var_grad_0_dummy, //not needed
                                            eddy_viscosity_ext,
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
                                            mom_w_source_ext);

                    updateTurbulenceClosure(turbulenceClosureModel,
                                            eps_rho,
                                            eps_mu,
                                            rho_0,
                                            nu_0,
                                            rho_1,
                                            nu_1,
                                            useVF,
                                            ebqe_vf_ext[ebNE_kb],
                                            ebqe_phi_ext[ebNE_kb],
                                            porosity_ext,
                                            c_mu, //mwf hack
                                            ebqe_turb_var_0[ebNE_kb],
                                            ebqe_turb_var_1[ebNE_kb],
                                            turb_var_grad_0_dummy, //not needed
                                            bc_eddy_viscosity_ext,
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
                                            bc_mom_w_source_ext);
                  }
                //
                //moving domain
                //
                mom_u_adv_ext[0] -= MOVING_DOMAIN*dmom_u_acc_u_ext*mom_u_acc_ext*xt_ext; //times rho*porosity. mql. CHECK.
                mom_u_adv_ext[1] -= MOVING_DOMAIN*dmom_u_acc_u_ext*mom_u_acc_ext*yt_ext;
                mom_u_adv_ext[2] -= MOVING_DOMAIN*dmom_u_acc_u_ext*mom_u_acc_ext*zt_ext;
                dmom_u_adv_u_ext[0] -= MOVING_DOMAIN*dmom_u_acc_u_ext*xt_ext;
                dmom_u_adv_u_ext[1] -= MOVING_DOMAIN*dmom_u_acc_u_ext*yt_ext;
                dmom_u_adv_u_ext[2] -= MOVING_DOMAIN*dmom_u_acc_u_ext*zt_ext;

                mom_v_adv_ext[0] -= MOVING_DOMAIN*dmom_v_acc_v_ext*mom_v_acc_ext*xt_ext;
                mom_v_adv_ext[1] -= MOVING_DOMAIN*dmom_v_acc_v_ext*mom_v_acc_ext*yt_ext;
                mom_v_adv_ext[2] -= MOVING_DOMAIN*dmom_v_acc_v_ext*mom_v_acc_ext*zt_ext;
                dmom_v_adv_v_ext[0] -= MOVING_DOMAIN*dmom_v_acc_v_ext*xt_ext;
                dmom_v_adv_v_ext[1] -= MOVING_DOMAIN*dmom_v_acc_v_ext*yt_ext;
                dmom_v_adv_v_ext[2] -= MOVING_DOMAIN*dmom_v_acc_v_ext*zt_ext;

                mom_w_adv_ext[0] -= MOVING_DOMAIN*dmom_w_acc_w_ext*mom_w_acc_ext*xt_ext;
                mom_w_adv_ext[1] -= MOVING_DOMAIN*dmom_w_acc_w_ext*mom_w_acc_ext*yt_ext;
                mom_w_adv_ext[2] -= MOVING_DOMAIN*dmom_w_acc_w_ext*mom_w_acc_ext*zt_ext;
                dmom_w_adv_w_ext[0] -= MOVING_DOMAIN*dmom_w_acc_w_ext*xt_ext;
                dmom_w_adv_w_ext[1] -= MOVING_DOMAIN*dmom_w_acc_w_ext*yt_ext;
                dmom_w_adv_w_ext[2] -= MOVING_DOMAIN*dmom_w_acc_w_ext*zt_ext;

                //moving domain bc's
                // mql. CHECK.
                bc_mom_u_adv_ext[0] -= MOVING_DOMAIN*bc_dmom_u_acc_u_ext*bc_mom_u_acc_ext*xt_ext; //times rho*porosity
                bc_mom_u_adv_ext[1] -= MOVING_DOMAIN*bc_dmom_u_acc_u_ext*bc_mom_u_acc_ext*yt_ext;
                bc_mom_u_adv_ext[2] -= MOVING_DOMAIN*bc_dmom_u_acc_u_ext*bc_mom_u_acc_ext*zt_ext;

                bc_mom_v_adv_ext[0] -= MOVING_DOMAIN*bc_dmom_v_acc_v_ext*bc_mom_v_acc_ext*xt_ext;
                bc_mom_v_adv_ext[1] -= MOVING_DOMAIN*bc_dmom_v_acc_v_ext*bc_mom_v_acc_ext*yt_ext;
                bc_mom_v_adv_ext[2] -= MOVING_DOMAIN*bc_dmom_v_acc_v_ext*bc_mom_v_acc_ext*zt_ext;

                bc_mom_w_adv_ext[0] -= MOVING_DOMAIN*bc_dmom_w_acc_w_ext*bc_mom_w_acc_ext*xt_ext;
                bc_mom_w_adv_ext[1] -= MOVING_DOMAIN*bc_dmom_w_acc_w_ext*bc_mom_w_acc_ext*yt_ext;
                bc_mom_w_adv_ext[2] -= MOVING_DOMAIN*bc_dmom_w_acc_w_ext*bc_mom_w_acc_ext*zt_ext;
                //
                //calculate the numerical fluxes
                //
                exteriorNumericalAdvectiveFluxDerivatives(isDOFBoundary_p[ebNE_kb],
                                                          isDOFBoundary_u[ebNE_kb],
                                                          isDOFBoundary_v[ebNE_kb],
                                                          isDOFBoundary_w[ebNE_kb],
                                                          isAdvectiveFluxBoundary_p[ebNE_kb],
                                                          isAdvectiveFluxBoundary_u[ebNE_kb],
                                                          isAdvectiveFluxBoundary_v[ebNE_kb],
                                                          isAdvectiveFluxBoundary_w[ebNE_kb],
                                                          dmom_u_ham_grad_p_ext[0],//=1/rho
                                                          normal,
                                                          porosity_ext*rhoSave,//dmom_u_acc_u_ext,
                                                          bc_p_ext,
                                                          bc_u_ext,
                                                          bc_v_ext,
                                                          bc_w_ext,
                                                          bc_mass_adv_ext,
                                                          bc_mom_u_adv_ext,
                                                          bc_mom_v_adv_ext,
                                                          bc_mom_w_adv_ext,
                                                          ebqe_bc_flux_mass_ext[ebNE_kb]+MOVING_DOMAIN*(xt_ext*normal[0]+yt_ext*normal[1]+zt_ext*normal[2]),//bc is relative mass  flux
                                                          ebqe_bc_flux_mom_u_adv_ext[ebNE_kb],
                                                          ebqe_bc_flux_mom_v_adv_ext[ebNE_kb],
                                                          ebqe_bc_flux_mom_w_adv_ext[ebNE_kb],
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
                                                          &ebqe_velocity_star[ebNE_kb_nSpace]);
                //
                //calculate the flux jacobian
                //
                ck.calculateGScale(G,normal,h_penalty);
                penalty = useMetrics*C_b/h_penalty + (1.0-useMetrics)*ebqe_penalty_ext[ebNE_kb];
                for (int j=0;j<nDOF_trial_element;j++)
                  {
                    int j_nSpace = j*nSpace,ebN_local_kb_j=ebN_local_kb*nDOF_trial_element+j;
                    /* fluxJacobian_p_p[j]=0.0; */
                    /* fluxJacobian_p_u[j]=ck.ExteriorNumericalAdvectiveFluxJacobian(dflux_mass_u_ext,vel_trial_trace_ref[ebN_local_kb_j]); */
                    /* fluxJacobian_p_v[j]=ck.ExteriorNumericalAdvectiveFluxJacobian(dflux_mass_v_ext,vel_trial_trace_ref[ebN_local_kb_j]); */
                    /* fluxJacobian_p_w[j]=ck.ExteriorNumericalAdvectiveFluxJacobian(dflux_mass_w_ext,vel_trial_trace_ref[ebN_local_kb_j]); */

                    /* fluxJacobian_u_p[j]=ck.ExteriorNumericalAdvectiveFluxJacobian(dflux_mom_u_adv_p_ext,p_trial_trace_ref[ebN_local_kb_j]); */
                    fluxJacobian_u_u[j]=ck.ExteriorNumericalAdvectiveFluxJacobian(dflux_mom_u_adv_u_ext,vel_trial_trace_ref[ebN_local_kb_j]) +
                      ExteriorNumericalDiffusiveFluxJacobian(eps_rho,
                                                             ebqe_phi_ext[ebNE_kb],
                                                             sdInfo_u_u_rowptr.data(),
                                                             sdInfo_u_u_colind.data(),
                                                             isDOFBoundary_u[ebNE_kb],
                                                             isDiffusiveFluxBoundary_u[ebNE_kb],
                                                             normal,
                                                             mom_uu_diff_ten_ext,
                                                             vel_trial_trace_ref[ebN_local_kb_j],
                                                             &vel_grad_trial_trace[j_nSpace],
                                                             penalty);//ebqe_penalty_ext[ebNE_kb]);
                    fluxJacobian_u_v[j]=ck.ExteriorNumericalAdvectiveFluxJacobian(dflux_mom_u_adv_v_ext,vel_trial_trace_ref[ebN_local_kb_j]) +
                      ExteriorNumericalDiffusiveFluxJacobian(eps_rho,
                                                             ebqe_phi_ext[ebNE_kb],
                                                             sdInfo_u_v_rowptr.data(),
                                                             sdInfo_u_v_colind.data(),
                                                             isDOFBoundary_v[ebNE_kb],
                                                             isDiffusiveFluxBoundary_v[ebNE_kb],
                                                             normal,
                                                             mom_uv_diff_ten_ext,
                                                             vel_trial_trace_ref[ebN_local_kb_j],
                                                             &vel_grad_trial_trace[j_nSpace],
                                                             penalty);//ebqe_penalty_ext[ebNE_kb]);
                    fluxJacobian_u_w[j]=ck.ExteriorNumericalAdvectiveFluxJacobian(dflux_mom_u_adv_w_ext,vel_trial_trace_ref[ebN_local_kb_j]) +
                      ExteriorNumericalDiffusiveFluxJacobian(eps_rho,
                                                             ebqe_phi_ext[ebNE_kb],
                                                             sdInfo_u_w_rowptr.data(),
                                                             sdInfo_u_w_colind.data(),
                                                             isDOFBoundary_w[ebNE_kb],
                                                             isDiffusiveFluxBoundary_u[ebNE_kb],
                                                             normal,
                                                             mom_uw_diff_ten_ext,
                                                             vel_trial_trace_ref[ebN_local_kb_j],
                                                             &vel_grad_trial_trace[j_nSpace],
                                                             penalty);//ebqe_penalty_ext[ebNE_kb]);

                    /* fluxJacobian_v_p[j]=ck.ExteriorNumericalAdvectiveFluxJacobian(dflux_mom_v_adv_p_ext,p_trial_trace_ref[ebN_local_kb_j]); */
                    fluxJacobian_v_u[j]=ck.ExteriorNumericalAdvectiveFluxJacobian(dflux_mom_v_adv_u_ext,vel_trial_trace_ref[ebN_local_kb_j]) +
                      ExteriorNumericalDiffusiveFluxJacobian(eps_rho,
                                                             ebqe_phi_ext[ebNE_kb],
                                                             sdInfo_v_u_rowptr.data(),
                                                             sdInfo_v_u_colind.data(),
                                                             isDOFBoundary_u[ebNE_kb],
                                                             isDiffusiveFluxBoundary_u[ebNE_kb],
                                                             normal,
                                                             mom_vu_diff_ten_ext,
                                                             vel_trial_trace_ref[ebN_local_kb_j],
                                                             &vel_grad_trial_trace[j_nSpace],
                                                             penalty);//ebqe_penalty_ext[ebNE_kb]);
                    fluxJacobian_v_v[j]=ck.ExteriorNumericalAdvectiveFluxJacobian(dflux_mom_v_adv_v_ext,vel_trial_trace_ref[ebN_local_kb_j]) +
                      ExteriorNumericalDiffusiveFluxJacobian(eps_rho,
                                                             ebqe_phi_ext[ebNE_kb],
                                                             sdInfo_v_v_rowptr.data(),
                                                             sdInfo_v_v_colind.data(),
                                                             isDOFBoundary_v[ebNE_kb],
                                                             isDiffusiveFluxBoundary_v[ebNE_kb],
                                                             normal,
                                                             mom_vv_diff_ten_ext,
                                                             vel_trial_trace_ref[ebN_local_kb_j],
                                                             &vel_grad_trial_trace[j_nSpace],
                                                             penalty);//ebqe_penalty_ext[ebNE_kb]);
                    fluxJacobian_v_w[j]=ck.ExteriorNumericalAdvectiveFluxJacobian(dflux_mom_v_adv_w_ext,vel_trial_trace_ref[ebN_local_kb_j]) +
                      ExteriorNumericalDiffusiveFluxJacobian(eps_rho,
                                                             ebqe_phi_ext[ebNE_kb],
                                                             sdInfo_v_w_rowptr.data(),
                                                             sdInfo_v_w_colind.data(),
                                                             isDOFBoundary_w[ebNE_kb],
                                                             isDiffusiveFluxBoundary_v[ebNE_kb],
                                                             normal,
                                                             mom_vw_diff_ten_ext,
                                                             vel_trial_trace_ref[ebN_local_kb_j],
                                                             &vel_grad_trial_trace[j_nSpace],
                                                             penalty);//ebqe_penalty_ext[ebNE_kb]);

                    /* fluxJacobian_w_p[j]=ck.ExteriorNumericalAdvectiveFluxJacobian(dflux_mom_w_adv_p_ext,p_trial_trace_ref[ebN_local_kb_j]); */
                    fluxJacobian_w_u[j]=ck.ExteriorNumericalAdvectiveFluxJacobian(dflux_mom_w_adv_u_ext,vel_trial_trace_ref[ebN_local_kb_j]) +
                      ExteriorNumericalDiffusiveFluxJacobian(eps_rho,
                                                             ebqe_phi_ext[ebNE_kb],
                                                             sdInfo_w_u_rowptr.data(),
                                                             sdInfo_w_u_colind.data(),
                                                             isDOFBoundary_u[ebNE_kb],
                                                             isDiffusiveFluxBoundary_w[ebNE_kb],
                                                             normal,
                                                             mom_wu_diff_ten_ext,
                                                             vel_trial_trace_ref[ebN_local_kb_j],
                                                             &vel_grad_trial_trace[j_nSpace],
                                                             penalty);//ebqe_penalty_ext[ebNE_kb]);
                    fluxJacobian_w_v[j]=ck.ExteriorNumericalAdvectiveFluxJacobian(dflux_mom_w_adv_v_ext,vel_trial_trace_ref[ebN_local_kb_j]) +
                      ExteriorNumericalDiffusiveFluxJacobian(eps_rho,
                                                             ebqe_phi_ext[ebNE_kb],
                                                             sdInfo_w_v_rowptr.data(),
                                                             sdInfo_w_v_colind.data(),
                                                             isDOFBoundary_v[ebNE_kb],
                                                             isDiffusiveFluxBoundary_w[ebNE_kb],
                                                             normal,
                                                             mom_wv_diff_ten_ext,
                                                             vel_trial_trace_ref[ebN_local_kb_j],
                                                             &vel_grad_trial_trace[j_nSpace],
                                                             penalty);//ebqe_penalty_ext[ebNE_kb]);
                    fluxJacobian_w_w[j]=ck.ExteriorNumericalAdvectiveFluxJacobian(dflux_mom_w_adv_w_ext,vel_trial_trace_ref[ebN_local_kb_j]) +
                      ExteriorNumericalDiffusiveFluxJacobian(eps_rho,
                                                             ebqe_phi_ext[ebNE_kb],
                                                             sdInfo_w_w_rowptr.data(),
                                                             sdInfo_w_w_colind.data(),
                                                             isDOFBoundary_w[ebNE_kb],
                                                             isDiffusiveFluxBoundary_w[ebNE_kb],
                                                             normal,
                                                             mom_ww_diff_ten_ext,
                                                             vel_trial_trace_ref[ebN_local_kb_j],
                                                             &vel_grad_trial_trace[j_nSpace],
                                                             penalty);//ebqe_penalty_ext[ebNE_kb]);
                  }//j
                //
                //update the global Jacobian from the flux Jacobian
                //
                for (int i=0;i<nDOF_test_element;i++)
                  {
                    int eN_i = eN*nDOF_test_element+i;
                    for (int j=0;j<nDOF_trial_element;j++)
                      {
                        int ebN_i_j = ebN*4*nDOF_test_X_trial_element + i*nDOF_trial_element + j,ebN_local_kb_j=ebN_local_kb*nDOF_trial_element+j;

                        /* globalJacobian[csrRowIndeces_p_p[eN_i] + csrColumnOffsets_eb_p_p[ebN_i_j]] += fluxJacobian_p_p[j]*p_test_dS[i]; */
                        /* globalJacobian[csrRowIndeces_p_u[eN_i] + csrColumnOffsets_eb_p_u[ebN_i_j]] += fluxJacobian_p_u[j]*p_test_dS[i]; */
                        /* globalJacobian[csrRowIndeces_p_v[eN_i] + csrColumnOffsets_eb_p_v[ebN_i_j]] += fluxJacobian_p_v[j]*p_test_dS[i]; */
                        /* globalJacobian[csrRowIndeces_p_w[eN_i] + csrColumnOffsets_eb_p_w[ebN_i_j]] += fluxJacobian_p_w[j]*p_test_dS[i]; */

                        /* globalJacobian[csrRowIndeces_u_p[eN_i] + csrColumnOffsets_eb_u_p[ebN_i_j]] += fluxJacobian_u_p[j]*vel_test_dS[i]; */
                        globalJacobian[csrRowIndeces_u_u[eN_i] + csrColumnOffsets_eb_u_u[ebN_i_j]] += fluxJacobian_u_u[j]*vel_test_dS[i]+
                          ck.ExteriorElementBoundaryDiffusionAdjointJacobian(isDOFBoundary_u[ebNE_kb],
                                                                             isDiffusiveFluxBoundary_u[ebNE_kb],
                                                                             eb_adjoint_sigma,
                                                                             vel_trial_trace_ref[ebN_local_kb_j],
                                                                             normal,
                                                                             sdInfo_u_u_rowptr.data(),
                                                                             sdInfo_u_u_colind.data(),
                                                                             mom_uu_diff_ten_ext,
                                                                             &vel_grad_test_dS[i*nSpace]);
                        globalJacobian[csrRowIndeces_u_v[eN_i] + csrColumnOffsets_eb_u_v[ebN_i_j]] += fluxJacobian_u_v[j]*vel_test_dS[i]+
                          ck.ExteriorElementBoundaryDiffusionAdjointJacobian(isDOFBoundary_v[ebNE_kb],
                                                                             isDiffusiveFluxBoundary_u[ebNE_kb],
                                                                             eb_adjoint_sigma,
                                                                             vel_trial_trace_ref[ebN_local_kb_j],
                                                                             normal,
                                                                             sdInfo_u_v_rowptr.data(),
                                                                             sdInfo_u_v_colind.data(),
                                                                             mom_uv_diff_ten_ext,
                                                                             &vel_grad_test_dS[i*nSpace]);
                        globalJacobian[csrRowIndeces_u_w[eN_i] + csrColumnOffsets_eb_u_w[ebN_i_j]] += fluxJacobian_u_w[j]*vel_test_dS[i]+
                          ck.ExteriorElementBoundaryDiffusionAdjointJacobian(isDOFBoundary_w[ebNE_kb],
                                                                             isDiffusiveFluxBoundary_u[ebNE_kb],
                                                                             eb_adjoint_sigma,
                                                                             vel_trial_trace_ref[ebN_local_kb_j],
                                                                             normal,
                                                                             sdInfo_u_w_rowptr.data(),
                                                                             sdInfo_u_w_colind.data(),
                                                                             mom_uw_diff_ten_ext,
                                                                             &vel_grad_test_dS[i*nSpace]);

                        /* globalJacobian[csrRowIndeces_v_p[eN_i] + csrColumnOffsets_eb_v_p[ebN_i_j]] += fluxJacobian_v_p[j]*vel_test_dS[i]; */
                        globalJacobian[csrRowIndeces_v_u[eN_i] + csrColumnOffsets_eb_v_u[ebN_i_j]] += fluxJacobian_v_u[j]*vel_test_dS[i]+
                          ck.ExteriorElementBoundaryDiffusionAdjointJacobian(isDOFBoundary_u[ebNE_kb],
                                                                             isDiffusiveFluxBoundary_v[ebNE_kb],
                                                                             eb_adjoint_sigma,
                                                                             vel_trial_trace_ref[ebN_local_kb_j],
                                                                             normal,
                                                                             sdInfo_v_u_rowptr.data(),
                                                                             sdInfo_v_u_colind.data(),
                                                                             mom_vu_diff_ten_ext,
                                                                             &vel_grad_test_dS[i*nSpace]);
                        globalJacobian[csrRowIndeces_v_v[eN_i] + csrColumnOffsets_eb_v_v[ebN_i_j]] += fluxJacobian_v_v[j]*vel_test_dS[i]+
                          ck.ExteriorElementBoundaryDiffusionAdjointJacobian(isDOFBoundary_v[ebNE_kb],
                                                                             isDiffusiveFluxBoundary_v[ebNE_kb],
                                                                             eb_adjoint_sigma,
                                                                             vel_trial_trace_ref[ebN_local_kb_j],
                                                                             normal,
                                                                             sdInfo_v_v_rowptr.data(),
                                                                             sdInfo_v_v_colind.data(),
                                                                             mom_vv_diff_ten_ext,
                                                                             &vel_grad_test_dS[i*nSpace]);
                        globalJacobian[csrRowIndeces_v_w[eN_i] + csrColumnOffsets_eb_v_w[ebN_i_j]] += fluxJacobian_v_w[j]*vel_test_dS[i]+
                          ck.ExteriorElementBoundaryDiffusionAdjointJacobian(isDOFBoundary_w[ebNE_kb],
                                                                             isDiffusiveFluxBoundary_v[ebNE_kb],
                                                                             eb_adjoint_sigma,
                                                                             vel_trial_trace_ref[ebN_local_kb_j],
                                                                             normal,
                                                                             sdInfo_v_w_rowptr.data(),
                                                                             sdInfo_v_w_colind.data(),
                                                                             mom_vw_diff_ten_ext,
                                                                             &vel_grad_test_dS[i*nSpace]);

                        /* globalJacobian[csrRowIndeces_w_p[eN_i] + csrColumnOffsets_eb_w_p[ebN_i_j]] += fluxJacobian_w_p[j]*vel_test_dS[i]; */
                        globalJacobian[csrRowIndeces_w_u[eN_i] + csrColumnOffsets_eb_w_u[ebN_i_j]] += fluxJacobian_w_u[j]*vel_test_dS[i]+
                          ck.ExteriorElementBoundaryDiffusionAdjointJacobian(isDOFBoundary_u[ebNE_kb],
                                                                             isDiffusiveFluxBoundary_w[ebNE_kb],
                                                                             eb_adjoint_sigma,
                                                                             vel_trial_trace_ref[ebN_local_kb_j],
                                                                             normal,
                                                                             sdInfo_w_u_rowptr.data(),
                                                                             sdInfo_w_u_colind.data(),
                                                                             mom_wu_diff_ten_ext,
                                                                             &vel_grad_test_dS[i*nSpace]);
                        globalJacobian[csrRowIndeces_w_v[eN_i] + csrColumnOffsets_eb_w_v[ebN_i_j]] += fluxJacobian_w_v[j]*vel_test_dS[i]+
                          ck.ExteriorElementBoundaryDiffusionAdjointJacobian(isDOFBoundary_v[ebNE_kb],
                                                                             isDiffusiveFluxBoundary_w[ebNE_kb],
                                                                             eb_adjoint_sigma,
                                                                             vel_trial_trace_ref[ebN_local_kb_j],
                                                                             normal,
                                                                             sdInfo_w_v_rowptr.data(),
                                                                             sdInfo_w_v_colind.data(),
                                                                             mom_wv_diff_ten_ext,
                                                                             &vel_grad_test_dS[i*nSpace]);
                        globalJacobian[csrRowIndeces_w_w[eN_i] + csrColumnOffsets_eb_w_w[ebN_i_j]] += fluxJacobian_w_w[j]*vel_test_dS[i]+
                          ck.ExteriorElementBoundaryDiffusionAdjointJacobian(isDOFBoundary_w[ebNE_kb],
                                                                             isDiffusiveFluxBoundary_w[ebNE_kb],
                                                                             eb_adjoint_sigma,
                                                                             vel_trial_trace_ref[ebN_local_kb_j],
                                                                             normal,
                                                                             sdInfo_w_w_rowptr.data(),
                                                                             sdInfo_w_w_colind.data(),
                                                                             mom_ww_diff_ten_ext,
                                                                             &vel_grad_test_dS[i*nSpace]);
                      }//j
                  }//i
              }//kb
          }//ebNE
        gf.useExact=useExact;
        gf_s.useExact=useExact;
      }//computeJacobian

      void calculateVelocityAverage(arguments_dict& args)
      {
        int nExteriorElementBoundaries_global = args.scalar<int>("nExteriorElementBoundaries_global");
        proteus::pyarray<int>& exteriorElementBoundariesArray = args.array<int>("exteriorElementBoundariesArray");
        int nInteriorElementBoundaries_global = args.scalar<int>("nInteriorElementBoundaries_global");
        proteus::pyarray<int>& interiorElementBoundariesArray = args.array<int>("interiorElementBoundariesArray");
        proteus::pyarray<int>& elementBoundaryElementsArray = args.array<int>("elementBoundaryElementsArray");
        proteus::pyarray<int>& elementBoundaryLocalElementBoundariesArray = args.array<int>("elementBoundaryLocalElementBoundariesArray");
        proteus::pyarray<double>& mesh_dof = args.array<double>("mesh_dof");
        proteus::pyarray<double>& mesh_velocity_dof = args.array<double>("mesh_velocity_dof");
        double MOVING_DOMAIN = args.scalar<double>("MOVING_DOMAIN");
        proteus::pyarray<int>& mesh_l2g = args.array<int>("mesh_l2g");
        proteus::pyarray<double>& mesh_trial_trace_ref = args.array<double>("mesh_trial_trace_ref");
        proteus::pyarray<double>& mesh_grad_trial_trace_ref = args.array<double>("mesh_grad_trial_trace_ref");
        proteus::pyarray<double>& normal_ref = args.array<double>("normal_ref");
        proteus::pyarray<double>& boundaryJac_ref = args.array<double>("boundaryJac_ref");
        proteus::pyarray<int>& vel_l2g = args.array<int>("vel_l2g");
        proteus::pyarray<double>& u_dof = args.array<double>("u_dof");
        proteus::pyarray<double>& v_dof = args.array<double>("v_dof");
        proteus::pyarray<double>& w_dof = args.array<double>("w_dof");
        proteus::pyarray<double>& vos_dof = args.array<double>("vos_dof");
        proteus::pyarray<double>& vel_trial_trace_ref = args.array<double>("vel_trial_trace_ref");
        proteus::pyarray<double>& ebqe_velocity = args.array<double>("ebqe_velocity");
        proteus::pyarray<double>& velocityAverage = args.array<double>("velocityAverage");
        int permutations[nQuadraturePoints_elementBoundary];
        double xArray_left[nQuadraturePoints_elementBoundary*3],
          xArray_right[nQuadraturePoints_elementBoundary*3];
        for (int i=0;i<nQuadraturePoints_elementBoundary;i++)
          permutations[i]=i;//just to initialize
        for (int ebNE = 0; ebNE < nExteriorElementBoundaries_global; ebNE++)
          {
            int ebN = exteriorElementBoundariesArray[ebNE];
            for  (int kb=0;kb<nQuadraturePoints_elementBoundary;kb++)
              {
                int ebN_kb_nSpace = ebN*nQuadraturePoints_elementBoundary*nSpace+kb*nSpace,
                  ebNE_kb_nSpace = ebNE*nQuadraturePoints_elementBoundary*nSpace+kb*nSpace;
                velocityAverage[ebN_kb_nSpace+0]=ebqe_velocity[ebNE_kb_nSpace+0];
                velocityAverage[ebN_kb_nSpace+1]=ebqe_velocity[ebNE_kb_nSpace+1];
                velocityAverage[ebN_kb_nSpace+2]=ebqe_velocity[ebNE_kb_nSpace+2];
              }//ebNE
          }
        for (int ebNI = 0; ebNI < nInteriorElementBoundaries_global; ebNI++)
          {
            int ebN = interiorElementBoundariesArray[ebNI],
              left_eN_global   = elementBoundaryElementsArray[ebN*2+0],
              left_ebN_element  = elementBoundaryLocalElementBoundariesArray[ebN*2+0],
              right_eN_global  = elementBoundaryElementsArray[ebN*2+1],
              right_ebN_element = elementBoundaryLocalElementBoundariesArray[ebN*2+1],
              left_eN_nDOF_trial_element = left_eN_global*nDOF_trial_element,
              right_eN_nDOF_trial_element = right_eN_global*nDOF_trial_element;
            double jac[nSpace*nSpace],
              jacDet,
              jacInv[nSpace*nSpace],
              boundaryJac[nSpace*(nSpace-1)],
              metricTensor[(nSpace-1)*(nSpace-1)],
              metricTensorDetSqrt,
              normal[3],
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
                xArray_left[kb*3+0] = x;
                xArray_left[kb*3+1] = y;
                xArray_left[kb*3+2] = z;
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
                xArray_right[kb*3+0] = x;
                xArray_right[kb*3+1] = y;
                xArray_right[kb*3+2] = z;
              }
            for  (int kb_left=0;kb_left<nQuadraturePoints_elementBoundary;kb_left++)
              {
                double errorNormMin = 1.0;
                for  (int kb_right=0;kb_right<nQuadraturePoints_elementBoundary;kb_right++)
                  {
                    double errorNorm=0.0;
                    for (int I=0;I<nSpace;I++)
                      {
                        errorNorm += fabs(xArray_left[kb_left*3+I]
                                          -
                                          xArray_right[kb_right*3+I]);
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
                int ebN_kb_nSpace = ebN*nQuadraturePoints_elementBoundary*nSpace+kb*nSpace;
                double u_left=0.0,
                  v_left=0.0,
                  w_left=0.0,
                  u_right=0.0,
                  v_right=0.0,
                  w_right=0.0,
                  vos_left=0.0,
                  vos_right=0.0,
                  porosity_left=0.0,
                  porosity_right=0.0;
                int left_kb = kb,
                  right_kb = permutations[kb],
                  left_ebN_element_kb_nDOF_test_element=(left_ebN_element*nQuadraturePoints_elementBoundary+left_kb)*nDOF_test_element,
                  right_ebN_element_kb_nDOF_test_element=(right_ebN_element*nQuadraturePoints_elementBoundary+right_kb)*nDOF_test_element;
                //
                //calculate the velocity solution at quadrature points on left and right
                //
                ck.valFromDOF(vos_dof.data(),&vel_l2g[left_eN_nDOF_trial_element],&vel_trial_trace_ref[left_ebN_element_kb_nDOF_test_element],vos_left);
                ck.valFromDOF(u_dof.data(),&vel_l2g[left_eN_nDOF_trial_element],&vel_trial_trace_ref[left_ebN_element_kb_nDOF_test_element],u_left);
                ck.valFromDOF(v_dof.data(),&vel_l2g[left_eN_nDOF_trial_element],&vel_trial_trace_ref[left_ebN_element_kb_nDOF_test_element],v_left);
                ck.valFromDOF(w_dof.data(),&vel_l2g[left_eN_nDOF_trial_element],&vel_trial_trace_ref[left_ebN_element_kb_nDOF_test_element],w_left);
                //
                ck.valFromDOF(vos_dof.data(),&vel_l2g[right_eN_nDOF_trial_element],&vel_trial_trace_ref[right_ebN_element_kb_nDOF_test_element],vos_right);
                ck.valFromDOF(u_dof.data(),&vel_l2g[right_eN_nDOF_trial_element],&vel_trial_trace_ref[right_ebN_element_kb_nDOF_test_element],u_right);
                ck.valFromDOF(v_dof.data(),&vel_l2g[right_eN_nDOF_trial_element],&vel_trial_trace_ref[right_ebN_element_kb_nDOF_test_element],v_right);
                ck.valFromDOF(w_dof.data(),&vel_l2g[right_eN_nDOF_trial_element],&vel_trial_trace_ref[right_ebN_element_kb_nDOF_test_element],w_right);
                //
                /* porosity_left = 1.0 - vos_left; */
                /* porosity_right = 1.0 - vos_right; */
                velocityAverage[ebN_kb_nSpace+0]=0.5*(u_left + u_right);
                velocityAverage[ebN_kb_nSpace+1]=0.5*(v_left + v_right);
                velocityAverage[ebN_kb_nSpace+2]=0.5*(w_left + w_right);
              }//ebNI
          }
      }

      void getBoundaryDOFs(arguments_dict& args)
      {
        proteus::pyarray<double>& mesh_dof = args.array<double>("mesh_dof");
        proteus::pyarray<int>& mesh_l2g = args.array<int>("mesh_l2g");
        proteus::pyarray<double>& mesh_trial_trace_ref = args.array<double>("mesh_trial_trace_ref");
        proteus::pyarray<double>& mesh_grad_trial_trace_ref = args.array<double>("mesh_grad_trial_trace_ref");
        proteus::pyarray<double>& dS_ref = args.array<double>("dS_ref");
        proteus::pyarray<double>& vel_test_trace_ref = args.array<double>("vel_test_trace_ref");
        proteus::pyarray<double>& normal_ref = args.array<double>("normal_ref");
        proteus::pyarray<double>& boundaryJac_ref = args.array<double>("boundaryJac_ref");
        proteus::pyarray<int>& vel_l2g = args.array<int>("vel_l2g");
        int nExteriorElementBoundaries_global = args.scalar<int>("nExteriorElementBoundaries_global");
        proteus::pyarray<int>& exteriorElementBoundariesArray = args.array<int>("exteriorElementBoundariesArray");
        proteus::pyarray<int>& elementBoundaryElementsArray = args.array<int>("elementBoundaryElementsArray");
        proteus::pyarray<int>& elementBoundaryLocalElementBoundariesArray = args.array<int>("elementBoundaryLocalElementBoundariesArray");
        proteus::pyarray<double>& isBoundary_1D = args.array<double>("isBoundary_1D");
	//
        //loop over exterior element boundaries to calculate surface integrals and load into element and global residuals
        //
        //ebNE is the Exterior element boundary INdex
        //ebN is the element boundary INdex
        //eN is the element index
        for (int ebNE = 0; ebNE < nExteriorElementBoundaries_global; ebNE++)
          {
            int ebN = exteriorElementBoundariesArray[ebNE],
              eN  = elementBoundaryElementsArray[ebN*2+0],
              ebN_local = elementBoundaryLocalElementBoundariesArray[ebN*2+0],
              eN_nDOF_trial_element = eN*nDOF_trial_element;
            double
	      elementIsBoundary[nDOF_test_element];
            for (int i=0;i<nDOF_test_element;i++)
	      elementIsBoundary[i]=0.0;
            for  (int kb=0;kb<nQuadraturePoints_elementBoundary;kb++)
              {
                int ebNE_kb = ebNE*nQuadraturePoints_elementBoundary+kb,
                  ebNE_kb_nSpace = ebNE_kb*nSpace,
                  ebN_local_kb = ebN_local*nQuadraturePoints_elementBoundary+kb,
                  ebN_local_kb_nSpace = ebN_local_kb*nSpace;
                double
                  jac_ext[nSpace*nSpace],
                  jacDet_ext,
                  jacInv_ext[nSpace*nSpace],
                  boundaryJac[nSpace*(nSpace-1)],
                  metricTensor[(nSpace-1)*(nSpace-1)],
                  metricTensorDetSqrt,
                  dS, vel_test_dS[nDOF_test_element],
                  normal[3],x_ext,y_ext,z_ext;
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
                dS = metricTensorDetSqrt*dS_ref[kb];
                //precalculate test function products with integration weights
                for (int j=0;j<nDOF_trial_element;j++)
		  vel_test_dS[j] = fabs(vel_test_trace_ref[ebN_local_kb*nDOF_test_element+j])*dS;
                //
                //update residuals
                //
                for (int i=0;i<nDOF_test_element;i++)
		  elementIsBoundary[i] += vel_test_dS[i];
              }//kb
            //
            //update the element and global residual storage
            //
            for (int i=0;i<nDOF_test_element;i++)
              {
                int eN_i = eN*nDOF_test_element+i;
                isBoundary_1D[vel_l2g[eN_i]] += elementIsBoundary[i];
              }//i
          }//ebNE
      }
    };//RANS3PF

  inline cppRANS3PF_base* newRANS3PF(int nSpaceIn,
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
                                     double mu_fr_limiter
                                      )
  {
    cppRANS3PF_base *rvalue = proteus::chooseAndAllocateDiscretization<cppRANS3PF_base, cppRANS3PF, CompKernel>(nSpaceIn,
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
                          mu_fr_limiter
                           );
    return rvalue;
  }
} //proteus

#endif
