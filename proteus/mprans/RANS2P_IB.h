#ifndef RANS2P_IB_H
#define RANS2P_IB_H
#include <cmath>
#include <iostream>
#include "CompKernel.h"
#include "ModelFactory.h"
#include "ArgumentsDict.h"
#include "pyarray.h"

namespace py = pybind11;

//#define COMPRESSIBLE_FORM
namespace proteus
{
  class RANS2P_IB_base
  {
  public:
    virtual ~RANS2P_IB_base(){}
    virtual void calculateResidual(arguments_dict& args)=0;
    virtual void calculateBeams(arguments_dict& args)=0;
    virtual void calculateJacobian(arguments_dict& args)=0;
    virtual void calculateForce(arguments_dict& args)=0;
    virtual void calculateVelocityAverage(arguments_dict& args)=0;
  };

  template<class CompKernelType,
    int nSpace,
    int nQuadraturePoints_element,
    int nDOF_mesh_trial_element,
    int nDOF_trial_element,
    int nDOF_test_element,
    int nQuadraturePoints_elementBoundary>
    class RANS2P_IB : public RANS2P_IB_base
    {
    public:
      const int nDOF_test_X_trial_element;
      CompKernelType ck;
    RANS2P_IB():
      nDOF_test_X_trial_element(nDOF_test_element*nDOF_trial_element),
        ck()
          {/*          std::cout<<"Constructing RANS2P<CompKernelTemplate<"
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
            HI= phi - eps                                                 \
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
        double delta_h(const double r)
      {
        double delta;
        if (r <= -2.0)
          delta= 0.0;
        else if (r <= -1.0)
          delta= 1.0/8.0*(5.0+2.0*r-sqrt(-7.0-12.0*r-4*r*r));
        else if  (r <= 0.0)
          delta= 1.0/8.0*(3.0+2.0*r+sqrt(1.0-4.0*r-4.0*r*r));
        else if (r <= 1.0)
          delta= 1.0/8.0*(3.0-2.0*r+sqrt(1.0+4.0*r-4.0*r*r));
        else if (r <= 2.0)
          delta= 1.0/8.0*(5.0-2.0*r-sqrt(-7.0+12.0*r-4.0*r*r));
        else
          delta= 0.0;
        return delta;
      }

      inline
        void calculateBeamSinks(const int nBeams,
                                const int nBeamElements,
                                const int beam_quadOrder,
                                const double beam_Cd,
                                const double* beamRadius,
                                const double x,
                                const double y,
                                const double z,
                                const double *xq,
                                const double *yq,
                                const double *zq,
                                const double *Beam_h,
                                const double *dV_beam,
                                const double u,
                                const double v,
                                const double w,
                                const double eps_rho,
                                const double& phi,
                                const double rho_0,
                                const double rho_1,
                                double& mom_u_source,
                                double& mom_v_source,
                                double& mom_w_source,
                                const double dV,
                                double* netBeamDrag)
      {
        double rho, H_rho, delt, vel;
        H_rho = smoothedHeaviside(eps_rho,phi);
        rho = rho_0*(1.0-H_rho)+rho_1*H_rho;
        vel = sqrt(u*u+v*v+w*w);
        mom_u_source = 0.0;
        mom_v_source = 0.0;
        mom_w_source = 0.0;
        for(int I=0; I<nBeams; I++)
          {
            for(int k=0;k<nBeamElements; k++)
              {
                for(int l=0;l<beam_quadOrder; l++)
                  {
                    delt= delta_h((x-xq[I*nBeamElements*beam_quadOrder+beam_quadOrder*k+l])/Beam_h[I*nBeamElements+k])*delta_h((y-yq[I*nBeamElements*beam_quadOrder+beam_quadOrder*k+l])/Beam_h[I*nBeamElements+k])*delta_h((z-zq[I*nBeamElements*beam_quadOrder+beam_quadOrder*k+l])/Beam_h[I*nBeamElements+k])/(Beam_h[I*nBeamElements+k]*Beam_h[I*nBeamElements+k]*Beam_h[I*nBeamElements+k]);
                    mom_u_source += beam_Cd*rho*beamRadius[I]*u*vel*delt*dV_beam[I*nBeamElements*beam_quadOrder+beam_quadOrder*k+l];
                    mom_v_source += beam_Cd*rho*beamRadius[I]*v*vel*delt*dV_beam[I*nBeamElements*beam_quadOrder+beam_quadOrder*k+l];
                    mom_w_source += beam_Cd*rho*beamRadius[I]*w*vel*delt*dV_beam[I*nBeamElements*beam_quadOrder+beam_quadOrder*k+l];
                    /* q1[I*nBeamElements*beam_quadOrder+beam_quadOrder*k+l] += beam_Cd*rho*beamRadius[I]*u*vel*delt*dV; */
                    /* q2[I*nBeamElements*beam_quadOrder+beam_quadOrder*k+l] += beam_Cd*rho*beamRadius[I]*v*vel*delt*dV; */
                    /* q3[I*nBeamElements*beam_quadOrder+beam_quadOrder*k+l] += beam_Cd*rho*beamRadius[I]*w*vel*delt*dV; */
                  }
              }
          }
        //if ((y>.25*.4) && ( y< .75*.4))
        netBeamDrag[0]+= dV*sqrt(mom_u_source*mom_u_source+mom_v_source*mom_v_source+mom_w_source*mom_w_source);
      }
      inline
        void calculateBeamLoads(const int nBeams,
                                const int nBeamElements,
                                const int beam_quadOrder,
                                const double beam_Cd,
                                const double* beamRadius,
                                const double x,
                                const double y,
                                const double z,
                                const double *xq,
                                const double *yq,
                                const double *zq,
                                const double *Beam_h,
                                const double u,
                                const double v,
                                const double w,
                                const double eps_rho,
                                const double& phi,
                                const double rho_0,
                                const double rho_1,
                                double *q1,
                                double *q2,
                                double *q3,
                                const double dV,
                                double *vel_avg)
      {
        double rho, H_rho, delt, vel,h_save, buoy;
        H_rho = smoothedHeaviside(eps_rho,phi);
        rho = rho_0*(1.0-H_rho)+rho_1*H_rho;
        vel = sqrt(u*u+v*v+w*w);
        //if (x > 0.25)
        //{
        vel_avg[0] += u*dV;
        vel_avg[1] += v*dV;
        vel_avg[2] += w*dV;
        //}
        for(int I=0; I<nBeams; I++)
          {
            buoy=0.0;
            //(rho - 368.0)*3.14159*beamRadius[I]*beamRadius[I];
            for(int k=0;k<nBeamElements; k++)
              {
                for(int l=0;l<beam_quadOrder; l++)
                  {
                    if (k==0)
                      h_save=0.25*Beam_h[I*nBeamElements+k];
                    else if (k==1)
                      h_save = 0.5*Beam_h[I*nBeamElements+k];
                    else
                      h_save = Beam_h[I*nBeamElements+k];
                    delt= delta_h((x-xq[I*nBeamElements*beam_quadOrder+beam_quadOrder*k+l])/h_save)*delta_h((y-yq[I*nBeamElements*beam_quadOrder+beam_quadOrder*k+l])/h_save)*delta_h((z-zq[I*nBeamElements*beam_quadOrder+beam_quadOrder*k+l])/h_save)/(h_save*h_save*h_save);
                    /* mom_u_source += beam_Cd*rho*beamRadius[I]*u*vel*delt*dV_beam[I*nBeamElements*beam_quadOrder+beam_quadOrder*k+l]; */
                    /* mom_v_source += beam_Cd*rho*beamRadius[I]*v*vel*delt*dV_beam[I*nBeamElements*beam_quadOrder+beam_quadOrder*k+l]; */
                    /* mom_w_source += beam_Cd*rho*beamRadius[I]*w*vel*delt*dV_beam[I*nBeamElements*beam_quadOrder+beam_quadOrder*k+l]; */
                    q1[I*nBeamElements*beam_quadOrder+beam_quadOrder*k+l] += beam_Cd*rho*beamRadius[I]*u*vel*delt*dV;
                    q2[I*nBeamElements*beam_quadOrder+beam_quadOrder*k+l] += beam_Cd*rho*beamRadius[I]*v*vel*delt*dV;
                    q3[I*nBeamElements*beam_quadOrder+beam_quadOrder*k+l] += (beam_Cd*rho*beamRadius[I]*w*vel+buoy)*delt*dV;
                  }
              }
          }
      }

      inline
        void evaluateCoefficients(const double eps_rho,
                                  const double eps_mu,
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
                                  double& mom_v_ham,
                                  double dmom_v_ham_grad_p[nSpace],
                                  double& mom_w_ham,
                                  double dmom_w_ham_grad_p[nSpace],
                                  const double& dragBeam1,
                                  const double& dragBeam2,
                                  const double& dragBeam3)
      {
        double rho,nu,mu,H_rho,d_rho,H_mu,d_mu,norm_n;
        H_rho = (1.0-useVF)*smoothedHeaviside(eps_rho,phi) + useVF*fmin(1.0,fmax(0.0,vf));
        d_rho = (1.0-useVF)*smoothedDirac(eps_rho,phi);
        H_mu = (1.0-useVF)*smoothedHeaviside(eps_mu,phi) + useVF*fmin(1.0,fmax(0.0,vf));
        d_mu = (1.0-useVF)*smoothedDirac(eps_mu,phi);

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
              nu_0 += smagorinskyConstant*smagorinskyConstant*h_e*h_e*norm_S;
              nu_1 += smagorinskyConstant*smagorinskyConstant*h_e*h_e*norm_S;
            }
          case 2:
            {
              double re_0,cs_0,re_1,cs_1;
              norm_S = sqrt(2.0*(grad_u[0]*grad_u[0] + grad_v[1]*grad_v[1] + grad_w[2]*grad_w[2] +
                                 0.5*(grad_u[1]+grad_v[0])*(grad_u[1]+grad_v[0]) +
                                 0.5*(grad_u[2]+grad_w[0])*(grad_u[2]+grad_w[0]) +
                                 0.5*(grad_v[2]+grad_w[1])*(grad_v[2]+grad_w[1])));
              re_0 = h_e*h_e*norm_S/nu_0;
              cs_0=0.027*pow(10.0,-3.23*pow(re_0,-0.92));
              nu_0 += cs_0*h_e*h_e*norm_S;
              re_1 = h_e*h_e*norm_S/nu_1;
              cs_1=0.027*pow(10.0,-3.23*pow(re_1,-0.92));
              nu_1 += cs_1*h_e*h_e*norm_S;
            }
          }

        rho = rho_0*(1.0-H_rho)+rho_1*H_rho;
        nu  = nu_0*(1.0-H_mu)+nu_1*H_mu;
        mu  = rho_0*nu_0*(1.0-H_mu)+rho_1*nu_1*H_mu;

#ifdef COMPRESSIBLE_FORM
        //u momentum accumulation
        mom_u_acc=porosity*rho*u;
        dmom_u_acc_u=porosity*rho;

        //v momentum accumulation
        mom_v_acc=porosity*rho*v;
        dmom_v_acc_v=porosity*rho;

        //w momentum accumulation
        mom_w_acc=porosity*rho*w;
        dmom_w_acc_w=porosity*rho;


        //mass advective flux
        mass_adv[0]=porosity*u;
        mass_adv[1]=porosity*v;
        mass_adv[2]=porosity*w;

        dmass_adv_u[0]=porosity;
        dmass_adv_u[1]=0.0;
        dmass_adv_u[2]=0.0;

        dmass_adv_v[0]=0.0;
        dmass_adv_v[1]=porosity;
        dmass_adv_v[2]=0.0;

        dmass_adv_w[0]=0.0;
        dmass_adv_w[1]=0.0;
        dmass_adv_w[2]=porosity;

        //u momentum advective flux
        mom_u_adv[0]=porosity*rho*u*u;
        mom_u_adv[1]=porosity*rho*u*v;
        mom_u_adv[2]=porosity*rho*u*w;

        dmom_u_adv_u[0]=porosity*rho*2.0*u;
        dmom_u_adv_u[1]=porosity*rho*v;
        dmom_u_adv_u[2]=porosity*rho*w;

        dmom_u_adv_v[0]=0.0;
        dmom_u_adv_v[1]=porosity*rho*u;
        dmom_u_adv_v[2]=0.0;

        dmom_u_adv_w[0]=0.0;
        dmom_u_adv_w[1]=0.0;
        dmom_u_adv_w[2]=porosity*rho*u;

        //v momentum advective_flux
        mom_v_adv[0]=porosity*rho*v*u;
        mom_v_adv[1]=porosity*rho*v*v;
        mom_v_adv[2]=porosity*rho*v*w;

        dmom_v_adv_u[0]=porosity*rho*v;
        dmom_v_adv_u[1]=0.0;
        dmom_v_adv_u[2]=0.0;

        dmom_v_adv_w[0]=0.0;
        dmom_v_adv_w[1]=0.0;
        dmom_v_adv_w[2]=porosity*rho*v;

        dmom_v_adv_v[0]=porosity*rho*u;
        dmom_v_adv_v[1]=porosity*rho*2.0*v;
        dmom_v_adv_v[2]=porosity*rho*w;

        //w momentum advective_flux
        mom_w_adv[0]=porosity*rho*w*u;
        mom_w_adv[1]=porosity*rho*w*v;
        mom_w_adv[2]=porosity*rho*w*w;

        dmom_w_adv_u[0]=porosity*rho*w;
        dmom_w_adv_u[1]=0.0;
        dmom_w_adv_u[2]=0.0;

        dmom_w_adv_v[0]=0.0;
        dmom_w_adv_v[1]=porosity*rho*w;
        dmom_w_adv_v[2]=0.0;

        dmom_w_adv_w[0]=porosity*rho*u;
        dmom_w_adv_w[1]=porosity*rho*v;
        dmom_w_adv_w[2]=porosity*rho*2.0*w;

        //u momentum diffusion tensor
        mom_uu_diff_ten[0] = 2.0*porosity*mu;
        mom_uu_diff_ten[1] = porosity*mu;
        mom_uu_diff_ten[2] = porosity*mu;

        mom_uv_diff_ten[0]=porosity*mu;

        mom_uw_diff_ten[0]=porosity*mu;

        //v momentum diffusion tensor
        mom_vv_diff_ten[0] = porosity*mu;
        mom_vv_diff_ten[1] = 2.0*porosity*mu;
        mom_vv_diff_ten[2] = porosity*mu;

        mom_vu_diff_ten[0]=porosity*mu;

        mom_vw_diff_ten[0]=porosity*mu;

        //w momentum diffusion tensor
        mom_ww_diff_ten[0] = porosity*mu;
        mom_ww_diff_ten[1] = porosity*mu;
        mom_ww_diff_ten[2] = 2.0*porosity*mu;

        mom_wu_diff_ten[0]=porosity*mu;

        mom_wv_diff_ten[0]=porosity*mu;

        //momentum sources
        norm_n = sqrt(n[0]*n[0]+n[1]*n[1]+n[2]*n[2]);
        mom_u_source = dragBeam1-porosity*rho*g[0];// - porosity*d_mu*sigma*kappa*n[0];
        mom_v_source = dragBeam2-porosity*rho*g[1];// - porosity*d_mu*sigma*kappa*n[1];
        mom_w_source = dragBeam3-porosity*rho*g[2];// - porosity*d_mu*sigma*kappa*n[2];

        //u momentum Hamiltonian (pressure)
        mom_u_ham = porosity*grad_p[0];
        dmom_u_ham_grad_p[0]=porosity;
        dmom_u_ham_grad_p[1]=0.0;
        dmom_u_ham_grad_p[2]=0.0;

        //v momentum Hamiltonian (pressure)
        mom_v_ham = porosity*grad_p[1];
        dmom_v_ham_grad_p[0]=0.0;
        dmom_v_ham_grad_p[1]=porosity;
        dmom_v_ham_grad_p[2]=0.0;

        //w momentum Hamiltonian (pressure)
        mom_w_ham = porosity*grad_p[2];
        dmom_w_ham_grad_p[0]=0.0;
        dmom_w_ham_grad_p[1]=0.0;
        dmom_w_ham_grad_p[2]=porosity;
#else
        //u momentum accumulation
        mom_u_acc=porosity*u;
        dmom_u_acc_u=porosity;

        //v momentum accumulation
        mom_v_acc=porosity*v;
        dmom_v_acc_v=porosity;

        //w momentum accumulation
        mom_w_acc=porosity*w;
        dmom_w_acc_w=porosity;


        //mass advective flux
        mass_adv[0]=porosity*u;
        mass_adv[1]=porosity*v;
        mass_adv[2]=porosity*w;

        dmass_adv_u[0]=porosity;
        dmass_adv_u[1]=0.0;
        dmass_adv_u[2]=0.0;

        dmass_adv_v[0]=0.0;
        dmass_adv_v[1]=porosity;
        dmass_adv_v[2]=0.0;

        dmass_adv_w[0]=0.0;
        dmass_adv_w[1]=0.0;
        dmass_adv_w[2]=porosity;

        //u momentum advective flux
        mom_u_adv[0]=porosity*u*u;
        mom_u_adv[1]=porosity*u*v;
        mom_u_adv[2]=porosity*u*w;

        dmom_u_adv_u[0]=2.0*porosity*u;
        dmom_u_adv_u[1]=porosity*v;
        dmom_u_adv_u[2]=porosity*w;

        dmom_u_adv_v[0]=0.0;
        dmom_u_adv_v[1]=porosity*u;
        dmom_u_adv_v[2]=0.0;

        dmom_u_adv_w[0]=0.0;
        dmom_u_adv_w[1]=0.0;
        dmom_u_adv_w[2]=porosity*u;

        //v momentum advective_flux
        mom_v_adv[0]=porosity*v*u;
        mom_v_adv[1]=porosity*v*v;
        mom_v_adv[2]=porosity*v*w;

        dmom_v_adv_u[0]=porosity*v;
        dmom_v_adv_u[1]=0.0;
        dmom_v_adv_u[2]=0.0;

        dmom_v_adv_w[0]=0.0;
        dmom_v_adv_w[1]=0.0;
        dmom_v_adv_w[2]=porosity*v;

        dmom_v_adv_v[0]=porosity*u;
        dmom_v_adv_v[1]=2.0*porosity*v;
        dmom_v_adv_v[2]=porosity*w;

        //w momentum advective_flux
        mom_w_adv[0]=porosity*w*u;
        mom_w_adv[1]=porosity*w*v;
        mom_w_adv[2]=porosity*w*w;

        dmom_w_adv_u[0]=porosity*w;
        dmom_w_adv_u[1]=0.0;
        dmom_w_adv_u[2]=0.0;

        dmom_w_adv_v[0]=0.0;
        dmom_w_adv_v[1]=porosity*w;
        dmom_w_adv_v[2]=0.0;

        dmom_w_adv_w[0]=porosity*u;
        dmom_w_adv_w[1]=porosity*v;
        dmom_w_adv_w[2]=2.0*porosity*w;

        //u momentum diffusion tensor
        mom_uu_diff_ten[0] = 2.0*porosity*nu;
        mom_uu_diff_ten[1] = porosity*nu;
        mom_uu_diff_ten[2] = porosity*nu;

        mom_uv_diff_ten[0]=porosity*nu;

        mom_uw_diff_ten[0]=porosity*nu;

        //v momentum diffusion tensor
        mom_vv_diff_ten[0] = porosity*nu;
        mom_vv_diff_ten[1] = 2.0*porosity*nu;
        mom_vv_diff_ten[2] = porosity*nu;

        mom_vu_diff_ten[0]=porosity*nu;

        mom_vw_diff_ten[0]=porosity*nu;

        //w momentum diffusion tensor
        mom_ww_diff_ten[0] = porosity*nu;
        mom_ww_diff_ten[1] = porosity*nu;
        mom_ww_diff_ten[2] = 2.0*porosity*nu;

        mom_wu_diff_ten[0]=porosity*nu;

        mom_wv_diff_ten[0]=porosity*nu;

        //momentum sources
        norm_n = sqrt(n[0]*n[0]+n[1]*n[1]+n[2]*n[2]);
        mom_u_source = dragBeam1/rho-porosity*g[0];// - porosity*d_mu*sigma*kappa*n[0]/(rho*(norm_n+1.0e-8));
        mom_v_source = dragBeam2/rho-porosity*g[1];// - porosity*d_mu*sigma*kappa*n[1]/(rho*(norm_n+1.0e-8));
        mom_w_source = dragBeam3/rho-porosity*g[2];// - porosity*d_mu*sigma*kappa*n[2]/(rho*(norm_n+1.0e-8));

        //u momentum Hamiltonian (pressure)
        mom_u_ham = porosity*grad_p[0]/rho;
        dmom_u_ham_grad_p[0]=porosity/rho;
        dmom_u_ham_grad_p[1]=0.0;
        dmom_u_ham_grad_p[2]=0.0;

        //v momentum Hamiltonian (pressure)
        mom_v_ham = porosity*grad_p[1]/rho;
        dmom_v_ham_grad_p[0]=0.0;
        dmom_v_ham_grad_p[1]=porosity/rho;
        dmom_v_ham_grad_p[2]=0.0;

        //w momentum Hamiltonian (pressure)
        mom_w_ham = porosity*grad_p[2]/rho;
        dmom_w_ham_grad_p[0]=0.0;
        dmom_w_ham_grad_p[1]=0.0;
        dmom_w_ham_grad_p[2]=porosity/rho;
#endif
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
                                               const double useVF,
                                               const double vf,
                                               const double phi,
                                               const double u,
                                               const double v,
                                               const double w,
                                               const double eps_s,
                                               const double phi_s,
                                               const double u_s,
                                               const double v_s,
                                               const double w_s,
                                               double& mom_u_source,
                                               double& mom_v_source,
                                               double& mom_w_source,
                                               double dmom_u_source[nSpace],
                                               double dmom_v_source[nSpace],
                                               double dmom_w_source[nSpace])
      {
        double mu,nu,H_mu,uc,duc_du,duc_dv,duc_dw,viscosity,H_s;
        H_mu = (1.0-useVF)*smoothedHeaviside(eps_mu,phi)+useVF*fmin(1.0,fmax(0.0,vf));
        nu  = nu_0*(1.0-H_mu)+nu_1*H_mu;
        mu  = rho_0*nu_0*(1.0-H_mu)+rho_1*nu_1*H_mu;
#ifdef COMPRESSIBLE_FORM
        viscosity = mu;
#else
        viscosity = nu;
#endif
        H_s = (1.0-useVF)*smoothedHeaviside(eps_s,phi_s)+useVF*fmin(1.0,fmax(0.0,vf));

        uc = sqrt(u*u+v*v*+w*w);
        duc_du = u/(uc+1.0e-12);
        duc_dv = v/(uc+1.0e-12);
        duc_dw = w/(uc+1.0e-12);

        mom_u_source += H_s*(viscosity*alpha + beta*uc)*(u-u_s);
        mom_v_source += H_s*(viscosity*alpha + beta*uc)*(v-v_s);
        mom_w_source += H_s*(viscosity*alpha + beta*uc)*(w-w_s);

        dmom_u_source[0] = H_s*(viscosity*alpha + beta*(uc + u*duc_du));
        dmom_u_source[1] = H_s*beta*u*duc_dv;
        dmom_u_source[2] = H_s*beta*u*duc_dw;

        dmom_v_source[0] = H_s*beta*v*duc_du;
        dmom_v_source[1] = H_s*(viscosity*alpha + beta*(uc + v*duc_dv));
        dmom_v_source[2] = H_s*beta*w*duc_dw;

        dmom_w_source[0] = H_s*beta*w*duc_du;
        dmom_w_source[1] = H_s*beta*w*duc_dv;
        dmom_w_source[2] = H_s*(viscosity*alpha + beta*(uc + w*duc_dw));
      }

      inline
        void updateTurbulenceClosure(const int turbulenceClosureModel,
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
                                     const double turb_var_0, //k for k-eps or k-omega
                                     const double turb_var_1, //epsilon for k-epsilon, omega for k-omega
                                     const double turb_grad_0[nSpace],//grad k for k-eps,k-omega
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
                                     double& mom_w_source)
      {
        /****
             eddy_visc_coef
             <= 2  LES (do nothing)
             == 3  k-epsilon

        */
        assert (turbulenceClosureModel >=3);
        double rho,nu,H_mu,eddy_viscosity,nu_t=0.0,nu_t_keps =0.0, nu_t_komega=0.0;
        double isKEpsilon = 1.0;
        if (turbulenceClosureModel == 4)
          isKEpsilon = 0.0;
        H_mu = (1.0-useVF)*smoothedHeaviside(eps_mu,phi)+useVF*fmin(1.0,fmax(0.0,vf));
        nu  = nu_0*(1.0-H_mu)+nu_1*H_mu;
        rho  = rho_0*(1.0-H_mu)+rho_1*H_mu;

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
        //  std::cout<<"RANS2P WARNING isKEpsilon = "<<isKEpsilon<<" nu_t = " <<nu_t<<" nu= "<<nu<<" k= "<<turb_var_0<<" turb_var_1= "<<turb_var_1<<std::endl;
        //}

        nu_t = fmax(nu_t,1.0e-4*nu); //limit according to Lew, Buscaglia etal 01
        //mwf hack
        nu_t     = fmin(nu_t,1.0e6*nu);

#ifdef COMPRESSIBLE_FORM
        eddy_viscosity = nu_t*rho;
        //u momentum diffusion tensor
        mom_uu_diff_ten[0] += 2.0*porosity*eddy_viscosity;
        mom_uu_diff_ten[1] += porosity*eddy_viscosity;
        mom_uu_diff_ten[2] += porosity*eddy_viscosity;

        mom_uv_diff_ten[0] +=porosity*eddy_viscosity;

        mom_uw_diff_ten[0] +=porosity*eddy_viscosity;

        //v momentum diffusion tensor
        mom_vv_diff_ten[0] += porosity*eddy_viscosity;
        mom_vv_diff_ten[1] += 2.0*porosity*eddy_viscosity;
        mom_vv_diff_ten[2] += porosity*eddy_viscosity;

        mom_vu_diff_ten[0] += porosity*eddy_viscosity;

        mom_vw_diff_ten[0] += porosity*eddy_viscosity;

        //w momentum diffusion tensor
        mom_ww_diff_ten[0] += porosity*eddy_viscosity;
        mom_ww_diff_ten[1] += porosity*eddy_viscosity;
        mom_ww_diff_ten[2] += 2.0*porosity*eddy_viscosity;

        mom_wu_diff_ten[0] += porosity*eddy_viscosity;

        mom_wv_diff_ten[0] += porosity*eddy_viscosity;

#else
        eddy_viscosity = nu_t;
        //u momentum diffusion tensor
        mom_uu_diff_ten[0] += 2.0*porosity*eddy_viscosity;
        mom_uu_diff_ten[1] += porosity*eddy_viscosity;
        mom_uu_diff_ten[2] += porosity*eddy_viscosity;

        mom_uv_diff_ten[0]+=porosity*eddy_viscosity;

        mom_uw_diff_ten[0]+=porosity*eddy_viscosity;

        //v momentum diffusion tensor
        mom_vv_diff_ten[0] += porosity*eddy_viscosity;
        mom_vv_diff_ten[1] += 2.0*porosity*eddy_viscosity;
        mom_vv_diff_ten[2] += porosity*eddy_viscosity;

        mom_vu_diff_ten[0]+=porosity*eddy_viscosity;

        mom_vw_diff_ten[0]+=porosity*eddy_viscosity;

        //w momentum diffusion tensor
        mom_ww_diff_ten[0] += porosity*eddy_viscosity;
        mom_ww_diff_ten[1] += porosity*eddy_viscosity;
        mom_ww_diff_ten[2] += 2.0*porosity*eddy_viscosity;

        mom_wu_diff_ten[0]+=porosity*eddy_viscosity;

        mom_wv_diff_ten[0]+=porosity*eddy_viscosity;

#endif

      }

      inline
        void calculateSubgridError_tau(const double&  hFactor,
                                       const double& elementDiameter,
                                       const double& dmt,
                                       const double& dm,
                                       const double df[nSpace],
                                       const double& a,
                                       const double&  pfac,
                                       double& tau_v,
                                       double& tau_p,
                                       double& cfl)
      {
        double h,oneByAbsdt,density,viscosity,nrm_df;
        h = hFactor*elementDiameter;
        density = dm;
        viscosity =  a;
        nrm_df=0.0;
        for(int I=0;I<nSpace;I++)
          nrm_df+=df[I]*df[I];
        nrm_df = sqrt(nrm_df);
        cfl = nrm_df/(h*density);//this is really cfl/dt, but that's what we want to know, the step controller expect this
        oneByAbsdt =  fabs(dmt);
        tau_v = 1.0/(4.0*viscosity/(h*h) + 2.0*nrm_df/h + oneByAbsdt);
        tau_p = (4.0*viscosity + 2.0*nrm_df*h + oneByAbsdt*h*h)/pfac;
        /* std::cout<<"tau_v "<<tau_v<<" tau_p "<<tau_p<<std::endl; */
      }

      inline
        void calculateSubgridError_tau(     const double&  Ct_sge,
                                            const double&  Cd_sge,
                                            const double   G[nSpace*nSpace],
                                            const double&  G_dd_G,
                                            const double&  tr_G,
                                            const double&  A0,
                                            const double   Ai[nSpace],
                                            const double&  Kij,
                                            const double&  pfac,
                                            double& tau_v,
                                            double& tau_p,
                                            double& q_cfl)
      {
        double v_d_Gv=0.0;
        for(int I=0;I<nSpace;I++)
          for (int J=0;J<nSpace;J++)
            v_d_Gv += Ai[I]*G[I*nSpace+J]*Ai[J];

        tau_v = 1.0/sqrt(Ct_sge*A0*A0 + v_d_Gv + Cd_sge*Kij*Kij*G_dd_G + 1.0e-12);
        tau_p = 1.0/(pfac*tr_G*tau_v);
      }

      inline
        void calculateSubgridError_tauRes(const double& tau_p,
                                          const double& tau_v,
                                          const double& pdeResidualP,
                                          const double& pdeResidualU,
                                          const double& pdeResidualV,
                                          const double& pdeResidualW,
                                          double& subgridErrorP,
                                          double& subgridErrorU,
                                          double& subgridErrorV,
                                          double& subgridErrorW)
      {
        /* GLS pressure */
        subgridErrorP = -tau_p*pdeResidualP;
        /* GLS momentum */
        subgridErrorU = -tau_v*pdeResidualU;
        subgridErrorV = -tau_v*pdeResidualV;
        subgridErrorW = -tau_v*pdeResidualW;
      }

      inline
        void calculateSubgridErrorDerivatives_tauRes(const double& tau_p,
                                                     const double& tau_v,
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
        for (int j=0;j<nDOF_trial_element;j++)
          {
            /* GLS pressure */
            dsubgridErrorP_du[j] = -tau_p*dpdeResidualP_du[j];
            dsubgridErrorP_dv[j] = -tau_p*dpdeResidualP_dv[j];
            dsubgridErrorP_dw[j] = -tau_p*dpdeResidualP_dw[j];
            /* GLS  momentum*/
            /* u */
            dsubgridErrorU_dp[j] = -tau_v*dpdeResidualU_dp[j];
            dsubgridErrorU_du[j] = -tau_v*dpdeResidualU_du[j];
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
                                            const double& bc_p,
                                            const double bc_f_mass[nSpace],
                                            const double bc_f_umom[nSpace],
                                            const double bc_f_vmom[nSpace],
                                            const double bc_f_wmom[nSpace],
                                            const double& bc_flux_mass,
                                            const double& bc_flux_umom,
                                            const double& bc_flux_vmom,
                                            const double& bc_flux_wmom,
                                            const double& p,
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
                                            double* velocity)
      {
        double flowDirection;
        flux_mass = 0.0;
        flux_umom = 0.0;
        flux_vmom = 0.0;
        flux_wmom = 0.0;
        flowDirection=n[0]*f_mass[0]+n[1]*f_mass[1]+n[2]*f_mass[2];
        if (isDOFBoundary_u != 1)
          {
            flux_mass += n[0]*f_mass[0];
            velocity[0] = f_mass[0];
            if (flowDirection >= 0.0)
              {
                flux_umom += n[0]*f_umom[0];
                flux_vmom += n[0]*f_vmom[0];
                flux_wmom += n[0]*f_wmom[0];
              }
          }
        else
          {
            flux_mass += n[0]*f_mass[0];
            velocity[0] = f_mass[0];
            if (flowDirection >= 0.0)
              {
                flux_umom += n[0]*f_umom[0];
                flux_vmom += n[0]*f_vmom[0];
                flux_wmom += n[0]*f_wmom[0];
              }
            else
              {
                flux_umom+=n[0]*bc_f_umom[0];
                flux_vmom+=n[0]*bc_f_vmom[0];
                flux_wmom+=n[0]*bc_f_wmom[0];
              }
          }
        if (isDOFBoundary_v != 1)
          {
            flux_mass+=n[1]*f_mass[1];
            velocity[1] = f_mass[1];
            if (flowDirection >= 0.0)
              {
                flux_umom+=n[1]*f_umom[1];
                flux_vmom+=n[1]*f_vmom[1];
                flux_wmom+=n[1]*f_wmom[1];
              }
          }
        else
          {
            flux_mass+=n[1]*f_mass[1];
            velocity[1] = f_mass[1];
            if (flowDirection >= 0.0)
              {
                flux_umom+=n[1]*f_umom[1];
                flux_vmom+=n[1]*f_vmom[1];
                flux_wmom+=n[1]*f_wmom[1];
              }
            else
              {
                flux_umom+=n[1]*bc_f_umom[1];
                flux_vmom+=n[1]*bc_f_vmom[1];
                flux_wmom+=n[1]*bc_f_wmom[1];
              }
          }
        if (isDOFBoundary_w != 1)
          {
            flux_mass+=n[2]*f_mass[2];
            velocity[2] = f_mass[2];
            if (flowDirection >= 0.0)
              {
                flux_umom+=n[2]*f_umom[2];
                flux_vmom+=n[2]*f_vmom[2];
                flux_wmom+=n[2]*f_wmom[2];
              }
          }
        else
          {
            flux_mass +=n[2]*f_mass[2];
            velocity[2] = f_mass[2];
            if (flowDirection >= 0.0)
              {
                flux_umom+=n[2]*f_umom[2];
                flux_vmom+=n[2]*f_vmom[2];
                flux_wmom+=n[2]*f_wmom[2];
              }
            else
              {
                flux_umom+=n[2]*bc_f_umom[2];
                flux_vmom+=n[2]*bc_f_vmom[2];
                flux_wmom+=n[2]*bc_f_wmom[2];
              }
          }
        if (isDOFBoundary_p == 1)
          {
            flux_umom+= n[0]*(bc_p*bc_oneByRho-p*oneByRho);
            flux_vmom+= n[1]*(bc_p*bc_oneByRho-p*oneByRho);
            flux_wmom+= n[2]*(bc_p*bc_oneByRho-p*oneByRho);
          }
        if (isFluxBoundary_p == 1)
          {
            velocity[0] += (bc_flux_mass - flux_mass)*n[0];
            velocity[1] += (bc_flux_mass - flux_mass)*n[1];
            velocity[2] += (bc_flux_mass - flux_mass)*n[2];
            flux_mass = bc_flux_mass;
          }
        if (isFluxBoundary_u == 1)
          {
            flux_umom = bc_flux_umom;
          }
        if (isFluxBoundary_v == 1)
          {
            flux_vmom = bc_flux_vmom;
          }
        if (isFluxBoundary_w == 1)
          {
            flux_wmom = bc_flux_wmom;
          }
        /* if (fabs(flux_mass) > 1.0e-4) */
        /*        std::cout<<"flux_mass "<<flux_mass<<std::endl */
        /*                 <<"n_0       "<<n[0]<<std::endl */
        /*                 <<"n_1       "<<n[1]<<std::endl */
        /*                 <<"n_2       "<<n[2]<<std::endl */
        /*                 <<"norm(n)   "<<sqrt(n[0]*n[0]+n[1]*n[1] + n[2]*n[2])<<std::endl */
        /*                 <<"v_0       "<<velocity[0]<<std::endl */
        /*                 <<"v_1       "<<velocity[1]<<std::endl */
        /*                 <<"v_2       "<<velocity[2]<<std::endl */
        /*                 <<std::endl; */
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
                                                       const double& bc_p,
                                                       const double bc_f_mass[nSpace],
                                                       const double bc_f_umom[nSpace],
                                                       const double bc_f_vmom[nSpace],
                                                       const double bc_f_wmom[nSpace],
                                                       const double& bc_flux_mass,
                                                       const double& bc_flux_umom,
                                                       const double& bc_flux_vmom,
                                                       const double& bc_flux_wmom,
                                                       const double& p,
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
                                                       double& dflux_wmom_dw)
      {
        double flowDirection;
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

        flowDirection=n[0]*f_mass[0]+n[1]*f_mass[1]+n[2]*f_mass[2];
        if (isDOFBoundary_u != 1)
          {
            dflux_mass_du += n[0]*df_mass_du[0];
            if (flowDirection >= 0.0)
              {
                dflux_umom_du += n[0]*df_umom_du[0];
                dflux_vmom_du += n[0]*df_vmom_du[0];
                dflux_vmom_dv += n[0]*df_vmom_dv[0];
                dflux_wmom_du += n[0]*df_wmom_du[0];
                dflux_wmom_dw += n[0]*df_wmom_dw[0];
              }
          }
        else
          {
            //cek still upwind the advection for Dirichlet?
            dflux_mass_du += n[0]*df_mass_du[0];
            if (flowDirection >= 0.0)
              {
                dflux_umom_du += n[0]*df_umom_du[0];
                dflux_vmom_du += n[0]*df_vmom_du[0];
                dflux_vmom_dv += n[0]*df_vmom_dv[0];
                dflux_wmom_du += n[0]*df_wmom_du[0];
                dflux_wmom_dw += n[0]*df_wmom_dw[0];
              }
            else
              {
                if (isDOFBoundary_v != 1)
                  dflux_vmom_dv += n[0]*df_vmom_dv[0];
                if (isDOFBoundary_w != 1)
                  dflux_wmom_dw += n[0]*df_wmom_dw[0];
              }
          }
        if (isDOFBoundary_v != 1)
          {
            dflux_mass_dv += n[1]*df_mass_dv[1];
            if (flowDirection >= 0.0)
              {
                dflux_umom_du += n[1]*df_umom_du[1];
                dflux_umom_dv += n[1]*df_umom_dv[1];
                dflux_vmom_dv += n[1]*df_vmom_dv[1];
                dflux_wmom_dw += n[1]*df_wmom_dw[1];
                dflux_wmom_dv += n[1]*df_wmom_dv[1];
              }
          }
        else
          {
            //cek still upwind the advection for Dirichlet?
            dflux_mass_dv += n[1]*df_mass_dv[1];
            if (flowDirection >= 0.0)
              {
                dflux_umom_du += n[1]*df_umom_du[1];
                dflux_umom_dv += n[1]*df_umom_dv[1];
                dflux_vmom_dv += n[1]*df_vmom_dv[1];
                dflux_wmom_dw += n[1]*df_wmom_dw[1];
                dflux_wmom_dv += n[1]*df_wmom_dv[1];
              }
            else
              {
                if (isDOFBoundary_u != 1)
                  dflux_umom_du += n[1]*df_umom_du[1];
                if (isDOFBoundary_w != 1)
                  dflux_wmom_dw += n[1]*df_wmom_dw[1];
              }
          }
        if (isDOFBoundary_w != 1)
          {
            dflux_mass_dw+=n[2]*df_mass_dw[2];
            if (flowDirection >= 0.0)
              {
                dflux_umom_du += n[2]*df_umom_du[2];
                dflux_umom_dw += n[2]*df_umom_dw[2];
                dflux_vmom_dv += n[2]*df_vmom_dv[2];
                dflux_vmom_dw += n[2]*df_vmom_dw[2];
                dflux_wmom_dw += n[2]*df_wmom_dw[2];
              }
          }
        else
          {
            //cek still upwind the advection for Dirichlet?
            dflux_mass_dw += n[2]*df_mass_dw[2];
            if (flowDirection >= 0.0)
              {
                dflux_umom_du += n[2]*df_umom_du[2];
                dflux_umom_dw += n[2]*df_umom_dw[2];
                dflux_vmom_dv += n[2]*df_vmom_dv[2];
                dflux_vmom_dw += n[2]*df_vmom_dw[2];
                dflux_wmom_dw += n[2]*df_wmom_dw[2];
              }
            else
              {
                if (isDOFBoundary_u != 1)
                  dflux_umom_du += n[2]*df_umom_du[2];
                if (isDOFBoundary_v != 1)
                  dflux_vmom_dv += n[2]*df_vmom_dv[2];
              }
          }
        if (isDOFBoundary_p == 1)
          {
            dflux_umom_dp= -n[0]*oneByRho;
            dflux_vmom_dp= -n[1]*oneByRho;
            dflux_wmom_dp= -n[2]*oneByRho;
          }
        if (isFluxBoundary_p == 1)
          {
            dflux_mass_du = 0.0;
            dflux_mass_dv = 0.0;
            dflux_mass_dw = 0.0;
          }
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
            //flux*=(smoothedDirac(eps,0) - smoothedDirac(eps,phi))/smoothedDirac(eps,0);
          }
        else
          {
            // std::cerr<<"warning, diffusion term with no boundary condition set, setting diffusive flux to 0.0"<<std::endl;
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
        proteus::pyarray<int>& mesh_l2g = args.array<int>("mesh_l2g");
        proteus::pyarray<double>& dV_ref = args.array<double>("dV_ref");
        proteus::pyarray<double>& p_trial_ref = args.array<double>("p_trial_ref");
        proteus::pyarray<double>& p_grad_trial_ref = args.array<double>("p_grad_trial_ref");
        proteus::pyarray<double>& p_test_ref = args.array<double>("p_test_ref");
        proteus::pyarray<double>& p_grad_test_ref = args.array<double>("p_grad_test_ref");
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
        int nElements_global = args.scalar<int>("nElements_global");
        int nElementBoundaries_owned = args.scalar<int>("nElementBoundaries_owned");
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
        const proteus::pyarray<double>& phi_solid = args.array<double>("phi_solid");
        const proteus::pyarray<double>& q_velocity_solid = args.array<double>("q_velocity_solid");
        const proteus::pyarray<double>& q_porosity = args.array<double>("q_porosity");
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
        proteus::pyarray<double>& normal_phi = args.array<double>("normal_phi");
        proteus::pyarray<double>& kappa_phi = args.array<double>("kappa_phi");
        proteus::pyarray<double>& q_mom_u_acc = args.array<double>("q_mom_u_acc");
        proteus::pyarray<double>& q_mom_v_acc = args.array<double>("q_mom_v_acc");
        proteus::pyarray<double>& q_mom_w_acc = args.array<double>("q_mom_w_acc");
        proteus::pyarray<double>& q_mass_adv = args.array<double>("q_mass_adv");
        proteus::pyarray<double>& q_mom_u_acc_beta_bdf = args.array<double>("q_mom_u_acc_beta_bdf");
        proteus::pyarray<double>& q_mom_v_acc_beta_bdf = args.array<double>("q_mom_v_acc_beta_bdf");
        proteus::pyarray<double>& q_mom_w_acc_beta_bdf = args.array<double>("q_mom_w_acc_beta_bdf");
        proteus::pyarray<double>& q_velocity_sge = args.array<double>("q_velocity_sge");
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
        proteus::pyarray<int>& elementBoundaryElementsArray = args.array<int>("elementBoundaryElementsArray");
        proteus::pyarray<int>& elementBoundaryLocalElementBoundariesArray = args.array<int>("elementBoundaryLocalElementBoundariesArray");
        proteus::pyarray<double>& ebqe_vf_ext = args.array<double>("ebqe_vf_ext");
        proteus::pyarray<double>& bc_ebqe_vf_ext = args.array<double>("bc_ebqe_vf_ext");
        proteus::pyarray<double>& ebqe_phi_ext = args.array<double>("ebqe_phi_ext");
        proteus::pyarray<double>& bc_ebqe_phi_ext = args.array<double>("bc_ebqe_phi_ext");
        proteus::pyarray<double>& ebqe_normal_phi_ext = args.array<double>("ebqe_normal_phi_ext");
        proteus::pyarray<double>& ebqe_kappa_phi_ext = args.array<double>("ebqe_kappa_phi_ext");
        const proteus::pyarray<double>& ebqe_porosity_ext = args.array<double>("ebqe_porosity_ext");
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
        proteus::pyarray<int>& boundaryFlags = args.array<int>("boundaryFlags");
        proteus::pyarray<double>& barycenters = args.array<double>("barycenters");
        proteus::pyarray<double>& wettedAreas = args.array<double>("wettedAreas");
        proteus::pyarray<double>& netForces_p = args.array<double>("netForces_p");
        proteus::pyarray<double>& netForces_v = args.array<double>("netForces_v");
        proteus::pyarray<double>& netMoments = args.array<double>("netMoments");
        proteus::pyarray<double>& q_dragBeam1 = args.array<double>("q_dragBeam1");
        proteus::pyarray<double>& q_dragBeam2 = args.array<double>("q_dragBeam2");
        proteus::pyarray<double>& q_dragBeam3 = args.array<double>("q_dragBeam3");
        proteus::pyarray<double>& ebqe_dragBeam1 = args.array<double>("ebqe_dragBeam1");
        proteus::pyarray<double>& ebqe_dragBeam2 = args.array<double>("ebqe_dragBeam2");
        proteus::pyarray<double>& ebqe_dragBeam3 = args.array<double>("ebqe_dragBeam3");
        //
        //loop over elements to compute volume integrals and load them into element and global residual
        //
        double globalConservationError=0.0;
        for(int eN=0;eN<nElements_global;eN++)
          {
            //declare local storage for element residual and initialize
            double elementResidual_p[nDOF_test_element],
              elementResidual_u[nDOF_test_element],
              elementResidual_v[nDOF_test_element],
              elementResidual_w[nDOF_test_element],
              eps_rho,eps_mu;
            for (int i=0;i<nDOF_test_element;i++)
              {
                int eN_i = eN*nDOF_test_element+i;
                elementResidual_p_save.data()[eN_i]=0.0;
                elementResidual_p[i]=0.0;
                elementResidual_u[i]=0.0;
                elementResidual_v[i]=0.0;
                elementResidual_w[i]=0.0;
              }//i
            //
            //loop over quadrature points and compute integrands
            //
            for(int k=0;k<nQuadraturePoints_element;k++)
              {
                //compute indices and declare local storage
                int eN_k = eN*nQuadraturePoints_element+k,
                  eN_k_nSpace = eN_k*nSpace,
                  eN_nDOF_trial_element = eN*nDOF_trial_element;
                double p=0.0,u=0.0,v=0.0,w=0.0,
                  grad_p[nSpace],grad_u[nSpace],grad_v[nSpace],grad_w[nSpace],
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
                  mom_v_ham=0.0,
                  dmom_v_ham_grad_p[nSpace],
                  mom_w_ham=0.0,
                  dmom_w_ham_grad_p[nSpace],
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
                  p_test_dV[nDOF_trial_element],vel_test_dV[nDOF_trial_element],
                  p_grad_test_dV[nDOF_test_element*nSpace],vel_grad_test_dV[nDOF_test_element*nSpace],
                  dV,x,y,z,xt,yt,zt,
                  //
                  porosity,
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
                ck.gradTrialFromRef(&p_grad_trial_ref.data()[k*nDOF_trial_element*nSpace],jacInv,p_grad_trial);
                ck.gradTrialFromRef(&vel_grad_trial_ref.data()[k*nDOF_trial_element*nSpace],jacInv,vel_grad_trial);
                //get the solution
                ck.valFromDOF(p_dof.data(),&p_l2g.data()[eN_nDOF_trial_element],&p_trial_ref.data()[k*nDOF_trial_element],p);
                ck.valFromDOF(u_dof.data(),&vel_l2g.data()[eN_nDOF_trial_element],&vel_trial_ref.data()[k*nDOF_trial_element],u);
                ck.valFromDOF(v_dof.data(),&vel_l2g.data()[eN_nDOF_trial_element],&vel_trial_ref.data()[k*nDOF_trial_element],v);
                ck.valFromDOF(w_dof.data(),&vel_l2g.data()[eN_nDOF_trial_element],&vel_trial_ref.data()[k*nDOF_trial_element],w);
                //get the solution gradients
                ck.gradFromDOF(p_dof.data(),&p_l2g.data()[eN_nDOF_trial_element],p_grad_trial,grad_p);
                ck.gradFromDOF(u_dof.data(),&vel_l2g.data()[eN_nDOF_trial_element],vel_grad_trial,grad_u);
                ck.gradFromDOF(v_dof.data(),&vel_l2g.data()[eN_nDOF_trial_element],vel_grad_trial,grad_v);
                ck.gradFromDOF(w_dof.data(),&vel_l2g.data()[eN_nDOF_trial_element],vel_grad_trial,grad_w);
                //precalculate test function products with integration weights
                for (int j=0;j<nDOF_trial_element;j++)
                  {
                    p_test_dV[j] = p_test_ref.data()[k*nDOF_trial_element+j]*dV;
                    vel_test_dV[j] = vel_test_ref.data()[k*nDOF_trial_element+j]*dV;
                    for (int I=0;I<nSpace;I++)
                      {
                        p_grad_test_dV[j*nSpace+I]   = p_grad_trial[j*nSpace+I]*dV;//cek warning won't work for Petrov-Galerkin
                        vel_grad_test_dV[j*nSpace+I] = vel_grad_trial[j*nSpace+I]*dV;//cek warning won't work for Petrov-Galerkin
                      }
                  }
                //VRANS
                porosity      = q_porosity.data()[eN_k];
                //meanGrainSize = q_meanGrain[eN_k];
                //
                //save velocity at quadrature points for other models to use
                q_velocity.data()[eN_k_nSpace+0]=u;
                q_velocity.data()[eN_k_nSpace+1]=v;
                q_velocity.data()[eN_k_nSpace+2]=w;
                q_x.data()[eN_k_nSpace+0]=x;
                q_x.data()[eN_k_nSpace+1]=y;
                q_x.data()[eN_k_nSpace+2]=z;
                //
                //calculate pde coefficients at quadrature points
                //
                evaluateCoefficients(eps_rho,
                                     eps_mu,
                                     sigma,
                                     rho_0,
                                     nu_0,
                                     rho_1,
                                     nu_1,
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
                                     mom_v_ham,
                                     dmom_v_ham_grad_p,
                                     mom_w_ham,
                                     dmom_w_ham_grad_p,
                                     q_dragBeam1.data()[eN_k],
                                     q_dragBeam2.data()[eN_k],
                                     q_dragBeam3.data()[eN_k]);
                //VRANS
                mass_source = q_mass_source.data()[eN_k];
                //todo: decide if these should be lagged or not?
                updateDarcyForchheimerTerms_Ergun(/* linearDragFactor, */
                                                  /* nonlinearDragFactor, */
                                                  /* porosity, */
                                                  /* meanGrainSize, */
                                                  q_dragAlpha.data()[eN_k],
                                                  q_dragBeta.data()[eN_k],
                                                  eps_rho,
                                                  eps_mu,
                                                  rho_0,
                                                  nu_0,
                                                  rho_1,
                                                  nu_1,
                                                  useVF,
                                                  vf.data()[eN_k],
                                                  phi.data()[eN_k],
                                                  u,//q_velocity_sge.data()[eN_k_nSpace+0],//u
                                                  v,//q_velocity_sge.data()[eN_k_nSpace+1],//v
                                                  w,//q_velocity_sge.data()[eN_k_nSpace+2],//w
                                                  eps_solid.data()[0],
                                                  phi_solid.data()[eN_k],
                                                  q_velocity_solid.data()[eN_k_nSpace+0],
                                                  q_velocity_solid.data()[eN_k_nSpace+1],
                                                  q_velocity_solid.data()[eN_k_nSpace+2],
                                                  mom_u_source,
                                                  mom_v_source,
                                                  mom_w_source,
                                                  dmom_u_source,
                                                  dmom_v_source,
                                                  dmom_w_source);

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
                                            vf.data()[eN_k],
                                            phi.data()[eN_k],
                                            porosity,
                                            c_mu, //mwf hack
                                            q_turb_var_0.data()[eN_k],
                                            q_turb_var_1.data()[eN_k],
                                            &q_turb_var_grad_0.data()[eN_k_nSpace],
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
                q_mom_u_acc.data()[eN_k] = mom_u_acc;
                q_mom_v_acc.data()[eN_k] = mom_v_acc;
                q_mom_w_acc.data()[eN_k] = mom_w_acc;
                //subgrid error uses grid scale velocity
                q_mass_adv.data()[eN_k_nSpace+0] = u;
                q_mass_adv.data()[eN_k_nSpace+1] = v;
                q_mass_adv.data()[eN_k_nSpace+2] = w;
                //
                //moving mesh
                //
                //transform the continuity equation as if the accumulation term was  d(1)/dt
                mass_adv[0] -= MOVING_DOMAIN*xt;
                mass_adv[1] -= MOVING_DOMAIN*yt;
                mass_adv[2] -= MOVING_DOMAIN*zt;

                mom_u_adv[0] -= MOVING_DOMAIN*mom_u_acc*xt;
                mom_u_adv[1] -= MOVING_DOMAIN*mom_u_acc*yt;
                mom_u_adv[2] -= MOVING_DOMAIN*mom_u_acc*zt;
                dmom_u_adv_u[0] -= MOVING_DOMAIN*dmom_u_acc_u*xt;
                dmom_u_adv_u[1] -= MOVING_DOMAIN*dmom_u_acc_u*yt;
                dmom_u_adv_u[2] -= MOVING_DOMAIN*dmom_u_acc_u*zt;

                mom_v_adv[0] -= MOVING_DOMAIN*mom_v_acc*xt;
                mom_v_adv[1] -= MOVING_DOMAIN*mom_v_acc*yt;
                mom_v_adv[2] -= MOVING_DOMAIN*mom_v_acc*zt;
                dmom_v_adv_v[0] -= MOVING_DOMAIN*dmom_v_acc_v*xt;
                dmom_v_adv_v[1] -= MOVING_DOMAIN*dmom_v_acc_v*yt;
                dmom_v_adv_v[2] -= MOVING_DOMAIN*dmom_v_acc_v*zt;

                mom_w_adv[0] -= MOVING_DOMAIN*mom_w_acc*xt;
                mom_w_adv[1] -= MOVING_DOMAIN*mom_w_acc*yt;
                mom_w_adv[2] -= MOVING_DOMAIN*mom_w_acc*zt;
                dmom_w_adv_w[0] -= MOVING_DOMAIN*dmom_w_acc_w*xt;
                dmom_w_adv_w[1] -= MOVING_DOMAIN*dmom_w_acc_w*yt;
                dmom_w_adv_w[2] -= MOVING_DOMAIN*dmom_w_acc_w*zt;
                //
                //calculate time derivative at quadrature points
                //
                ck.bdf(alphaBDF,
                       q_mom_u_acc_beta_bdf.data()[eN_k],
                       mom_u_acc,
                       dmom_u_acc_u,
                       mom_u_acc_t,
                       dmom_u_acc_u_t);
                ck.bdf(alphaBDF,
                       q_mom_v_acc_beta_bdf.data()[eN_k],
                       mom_v_acc,
                       dmom_v_acc_v,
                       mom_v_acc_t,
                       dmom_v_acc_v_t);
                ck.bdf(alphaBDF,
                       q_mom_w_acc_beta_bdf.data()[eN_k],
                       mom_w_acc,
                       dmom_w_acc_w,
                       mom_w_acc_t,
                       dmom_w_acc_w_t);
                //
                //calculate subgrid error (strong residual and adjoint)
                //
                //calculate strong residual
                pdeResidual_p = ck.Advection_strong(dmass_adv_u,grad_u) +
                  //VRANS
                  ck.Reaction_strong(mass_source) +
                  //
                  ck.Advection_strong(dmass_adv_v,grad_v) +
                  ck.Advection_strong(dmass_adv_w,grad_w);

                dmom_adv_sge[0] = dmom_u_acc_u*(q_velocity_sge.data()[eN_k_nSpace+0] - MOVING_DOMAIN*xt);
                dmom_adv_sge[1] = dmom_u_acc_u*(q_velocity_sge.data()[eN_k_nSpace+1] - MOVING_DOMAIN*yt);
                dmom_adv_sge[2] = dmom_u_acc_u*(q_velocity_sge.data()[eN_k_nSpace+2] - MOVING_DOMAIN*zt);

                pdeResidual_u = ck.Mass_strong(mom_u_acc_t) +
                  ck.Advection_strong(dmom_adv_sge,grad_u) +
                  ck.Hamiltonian_strong(dmom_u_ham_grad_p,grad_p) +
                  ck.Reaction_strong(mom_u_source);

                pdeResidual_v = ck.Mass_strong(mom_v_acc_t) +
                  ck.Advection_strong(dmom_adv_sge,grad_v) +
                  ck.Hamiltonian_strong(dmom_v_ham_grad_p,grad_p) +
                  ck.Reaction_strong(mom_v_source);

                pdeResidual_w = ck.Mass_strong(mom_w_acc_t) +
                  ck.Advection_strong(dmom_adv_sge,grad_w) +
                  ck.Hamiltonian_strong(dmom_w_ham_grad_p,grad_p) +
                  ck.Reaction_strong(mom_w_source);

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
                                          dmom_u_acc_u_t,
                                          dmom_adv_sge,
                                          mom_uu_diff_ten[1],
                                          dmom_u_ham_grad_p[0],
                                          tau_v1,
                                          tau_p1,
                                          q_cfl.data()[eN_k]);

                tau_v = useMetrics*tau_v1+(1.0-useMetrics)*tau_v0;
                tau_p = useMetrics*tau_p1+(1.0-useMetrics)*tau_p0;

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
                dmom_adv_star[2] = dmom_u_acc_u*(q_velocity_sge.data()[eN_k_nSpace+2] - MOVING_DOMAIN*zt + useRBLES*subgridError_w);

                mom_u_adv[0] += dmom_u_acc_u*(useRBLES*subgridError_u*q_velocity_sge.data()[eN_k_nSpace+0]);
                mom_u_adv[1] += dmom_u_acc_u*(useRBLES*subgridError_v*q_velocity_sge.data()[eN_k_nSpace+0]);
                mom_u_adv[2] += dmom_u_acc_u*(useRBLES*subgridError_w*q_velocity_sge.data()[eN_k_nSpace+0]);

                mom_v_adv[0] += dmom_u_acc_u*(useRBLES*subgridError_u*q_velocity_sge.data()[eN_k_nSpace+1]);
                mom_v_adv[1] += dmom_u_acc_u*(useRBLES*subgridError_v*q_velocity_sge.data()[eN_k_nSpace+1]);
                mom_v_adv[2] += dmom_u_acc_u*(useRBLES*subgridError_w*q_velocity_sge.data()[eN_k_nSpace+1]);

                mom_w_adv[0] += dmom_u_acc_u*(useRBLES*subgridError_u*q_velocity_sge.data()[eN_k_nSpace+2]);
                mom_w_adv[1] += dmom_u_acc_u*(useRBLES*subgridError_v*q_velocity_sge.data()[eN_k_nSpace+2]);
                mom_w_adv[2] += dmom_u_acc_u*(useRBLES*subgridError_w*q_velocity_sge.data()[eN_k_nSpace+2]);

                // adjoint times the test functions
                for (int i=0;i<nDOF_test_element;i++)
                  {
                    int i_nSpace = i*nSpace;
                    Lstar_u_p[i]=ck.Advection_adjoint(dmass_adv_u,&p_grad_test_dV[i_nSpace]);
                    Lstar_v_p[i]=ck.Advection_adjoint(dmass_adv_v,&p_grad_test_dV[i_nSpace]);
                    Lstar_w_p[i]=ck.Advection_adjoint(dmass_adv_w,&p_grad_test_dV[i_nSpace]);
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

                norm_Rv = sqrt(pdeResidual_u*pdeResidual_u + pdeResidual_v*pdeResidual_v + pdeResidual_w*pdeResidual_w);
                q_numDiff_u.data()[eN_k] = C_dc*norm_Rv*(useMetrics/sqrt(G_dd_G+1.0e-12)  +
                                                  (1.0-useMetrics)*hFactor*hFactor*elementDiameter.data()[eN]*elementDiameter.data()[eN]);
                q_numDiff_v.data()[eN_k] = q_numDiff_u.data()[eN_k];
                q_numDiff_w.data()[eN_k] = q_numDiff_u.data()[eN_k];
                //
                //update element residual
                //
                for(int i=0;i<nDOF_test_element;i++)
                  {
                    int i_nSpace=i*nSpace;

                    elementResidual_p[i] += ck.Advection_weak(mass_adv,&p_grad_test_dV[i_nSpace]) +
                      //VRANS
                      ck.Reaction_weak(mass_source,p_test_dV[i])   + //VRANS source term for wave maker
                      //
                      ck.SubgridError(subgridError_u,Lstar_u_p[i]) +
                      ck.SubgridError(subgridError_v,Lstar_v_p[i]) +
                      ck.SubgridError(subgridError_w,Lstar_w_p[i]);

                    elementResidual_u[i] += ck.Mass_weak(mom_u_acc_t,vel_test_dV[i]) +
                      ck.Advection_weak(mom_u_adv,&vel_grad_test_dV[i_nSpace]) +
                      ck.Diffusion_weak(sdInfo_u_u_rowptr.data(),sdInfo_u_u_colind.data(),mom_uu_diff_ten,grad_u,&vel_grad_test_dV[i_nSpace]) +
                      ck.Diffusion_weak(sdInfo_u_v_rowptr.data(),sdInfo_u_v_colind.data(),mom_uv_diff_ten,grad_v,&vel_grad_test_dV[i_nSpace]) +
                      ck.Diffusion_weak(sdInfo_u_w_rowptr.data(),sdInfo_u_w_colind.data(),mom_uw_diff_ten,grad_w,&vel_grad_test_dV[i_nSpace]) +
                      ck.Reaction_weak(mom_u_source,vel_test_dV[i]) +
                      ck.Hamiltonian_weak(mom_u_ham,vel_test_dV[i]) +
                      ck.SubgridError(subgridError_p,Lstar_p_u[i]) +
                      ck.SubgridError(subgridError_u,Lstar_u_u[i]) +
                      ck.NumericalDiffusion(q_numDiff_u_last.data()[eN_k],grad_u,&vel_grad_test_dV[i_nSpace]);

                    elementResidual_v[i] += ck.Mass_weak(mom_v_acc_t,vel_test_dV[i]) +
                      ck.Advection_weak(mom_v_adv,&vel_grad_test_dV[i_nSpace]) +
                      ck.Diffusion_weak(sdInfo_v_u_rowptr.data(),sdInfo_v_u_colind.data(),mom_vu_diff_ten,grad_u,&vel_grad_test_dV[i_nSpace]) +
                      ck.Diffusion_weak(sdInfo_v_v_rowptr.data(),sdInfo_v_v_colind.data(),mom_vv_diff_ten,grad_v,&vel_grad_test_dV[i_nSpace]) +
                      ck.Diffusion_weak(sdInfo_v_w_rowptr.data(),sdInfo_v_w_colind.data(),mom_vw_diff_ten,grad_w,&vel_grad_test_dV[i_nSpace]) +
                      ck.Reaction_weak(mom_v_source,vel_test_dV[i]) +
                      ck.Hamiltonian_weak(mom_v_ham,vel_test_dV[i]) +
                      ck.SubgridError(subgridError_p,Lstar_p_v[i]) +
                      ck.SubgridError(subgridError_v,Lstar_v_v[i]) +
                      ck.NumericalDiffusion(q_numDiff_v_last.data()[eN_k],grad_v,&vel_grad_test_dV[i_nSpace]);

                    elementResidual_w[i] +=  ck.Mass_weak(mom_w_acc_t,vel_test_dV[i]) +
                      ck.Advection_weak(mom_w_adv,&vel_grad_test_dV[i_nSpace]) +
                      ck.Diffusion_weak(sdInfo_w_u_rowptr.data(),sdInfo_w_u_colind.data(),mom_wu_diff_ten,grad_u,&vel_grad_test_dV[i_nSpace]) +
                      ck.Diffusion_weak(sdInfo_w_v_rowptr.data(),sdInfo_w_v_colind.data(),mom_wv_diff_ten,grad_v,&vel_grad_test_dV[i_nSpace]) +
                      ck.Diffusion_weak(sdInfo_w_w_rowptr.data(),sdInfo_w_w_colind.data(),mom_ww_diff_ten,grad_w,&vel_grad_test_dV[i_nSpace]) +
                      ck.Reaction_weak(mom_w_source,vel_test_dV[i]) +
                      ck.Hamiltonian_weak(mom_w_ham,vel_test_dV[i]) +
                      ck.SubgridError(subgridError_p,Lstar_p_w[i]) +
                      ck.SubgridError(subgridError_w,Lstar_w_w[i]) +
                      ck.NumericalDiffusion(q_numDiff_w_last.data()[eN_k],grad_w,&vel_grad_test_dV[i_nSpace]);
                  }//i
              }
            //
            //load element into global residual and save element residual
            //
            for(int i=0;i<nDOF_test_element;i++)
              {
                int eN_i=eN*nDOF_test_element+i;

                elementResidual_p_save.data()[eN_i] +=  elementResidual_p[i];

                globalResidual.data()[offset_p+stride_p*p_l2g.data()[eN_i]]+=elementResidual_p[i];
                globalResidual.data()[offset_u+stride_u*vel_l2g.data()[eN_i]]+=elementResidual_u[i];
                globalResidual.data()[offset_v+stride_v*vel_l2g.data()[eN_i]]+=elementResidual_v[i];
                globalResidual.data()[offset_w+stride_w*vel_l2g.data()[eN_i]]+=elementResidual_w[i];
              }//i
          }//elements
        //
        //loop over exterior element boundaries to calculate surface integrals and load into element and global residuals
        //
        //ebNE is the Exterior element boundary INdex
        //ebN is the element boundary INdex
        //eN is the element index
        for (int ebNE = 0; ebNE < nExteriorElementBoundaries_global; ebNE++)
          {
            int ebN = exteriorElementBoundariesArray.data()[ebNE],
              eN  = elementBoundaryElementsArray.data()[ebN*2+0],
              ebN_local = elementBoundaryLocalElementBoundariesArray.data()[ebN*2+0],
              eN_nDOF_trial_element = eN*nDOF_trial_element;
            double elementResidual_p[nDOF_test_element],
              elementResidual_u[nDOF_test_element],
              elementResidual_v[nDOF_test_element],
              elementResidual_w[nDOF_test_element],
              eps_rho,eps_mu;
            for (int i=0;i<nDOF_test_element;i++)
              {
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
                  mom_v_ham_ext=0.0,
                  dmom_v_ham_grad_p_ext[nSpace],
                  mom_w_ham_ext=0.0,
                  dmom_w_ham_grad_p_ext[nSpace],
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
                  bc_mom_v_ham_ext=0.0,
                  bc_dmom_v_ham_grad_p_ext[nSpace],
                  bc_mom_w_ham_ext=0.0,
                  bc_dmom_w_ham_grad_p_ext[nSpace],
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
                //std::cout<<"integralScaling - metricTensorDetSrt ==============================="<<integralScaling-metricTensorDetSqrt<<std::endl;
                /* std::cout<<"metricTensorDetSqrt "<<metricTensorDetSqrt */
                /*               <<"dS_ref.data()[kb]"<<dS_ref.data()[kb]<<std::endl; */
                dS = ((1.0-MOVING_DOMAIN)*metricTensorDetSqrt + MOVING_DOMAIN*integralScaling)*dS_ref.data()[kb];
                //get the metric tensor
                //cek todo use symmetry
                ck.calculateG(jacInv_ext,G,G_dd_G,tr_G);
                ck.calculateGScale(G,&ebqe_normal_phi_ext.data()[ebNE_kb_nSpace],h_phi);

                eps_rho = epsFact_rho*(useMetrics*h_phi+(1.0-useMetrics)*elementDiameter.data()[eN]);
                eps_mu  = epsFact_mu *(useMetrics*h_phi+(1.0-useMetrics)*elementDiameter.data()[eN]);

                //compute shape and solution information
                //shape
                ck.gradTrialFromRef(&p_grad_trial_trace_ref.data()[ebN_local_kb_nSpace*nDOF_trial_element],jacInv_ext,p_grad_trial_trace);
                ck.gradTrialFromRef(&vel_grad_trial_trace_ref.data()[ebN_local_kb_nSpace*nDOF_trial_element],jacInv_ext,vel_grad_trial_trace);
                //cek hack use trial ck.gradTrialFromRef(&vel_grad_test_trace_ref.data()[ebN_local_kb_nSpace*nDOF_trial_element],jacInv_ext,vel_grad_test_trace);
                //solution and gradients
                ck.valFromDOF(p_dof.data(),&p_l2g.data()[eN_nDOF_trial_element],&p_trial_trace_ref.data()[ebN_local_kb*nDOF_test_element],p_ext);
                ck.valFromDOF(u_dof.data(),&vel_l2g.data()[eN_nDOF_trial_element],&vel_trial_trace_ref.data()[ebN_local_kb*nDOF_test_element],u_ext);
                ck.valFromDOF(v_dof.data(),&vel_l2g.data()[eN_nDOF_trial_element],&vel_trial_trace_ref.data()[ebN_local_kb*nDOF_test_element],v_ext);
                ck.valFromDOF(w_dof.data(),&vel_l2g.data()[eN_nDOF_trial_element],&vel_trial_trace_ref.data()[ebN_local_kb*nDOF_test_element],w_ext);
                ck.gradFromDOF(p_dof.data(),&p_l2g.data()[eN_nDOF_trial_element],p_grad_trial_trace,grad_p_ext);
                ck.gradFromDOF(u_dof.data(),&vel_l2g.data()[eN_nDOF_trial_element],vel_grad_trial_trace,grad_u_ext);
                ck.gradFromDOF(v_dof.data(),&vel_l2g.data()[eN_nDOF_trial_element],vel_grad_trial_trace,grad_v_ext);
                ck.gradFromDOF(w_dof.data(),&vel_l2g.data()[eN_nDOF_trial_element],vel_grad_trial_trace,grad_w_ext);
                //precalculate test function products with integration weights
                for (int j=0;j<nDOF_trial_element;j++)
                  {
                    p_test_dS[j] = p_test_trace_ref.data()[ebN_local_kb*nDOF_test_element+j]*dS;
                    vel_test_dS[j] = vel_test_trace_ref.data()[ebN_local_kb*nDOF_test_element+j]*dS;
                    for (int I=0;I<nSpace;I++)
                      vel_grad_test_dS[j*nSpace+I] = vel_grad_trial_trace[j*nSpace+I]*dS;//cek hack, using trial
                  }
                bc_p_ext = isDOFBoundary_p.data()[ebNE_kb]*ebqe_bc_p_ext.data()[ebNE_kb]+(1-isDOFBoundary_p.data()[ebNE_kb])*p_ext;
                bc_u_ext = isDOFBoundary_u.data()[ebNE_kb]*ebqe_bc_u_ext.data()[ebNE_kb]+(1-isDOFBoundary_u.data()[ebNE_kb])*u_ext;
                bc_v_ext = isDOFBoundary_v.data()[ebNE_kb]*ebqe_bc_v_ext.data()[ebNE_kb]+(1-isDOFBoundary_v.data()[ebNE_kb])*v_ext;
                bc_w_ext = isDOFBoundary_w.data()[ebNE_kb]*ebqe_bc_w_ext.data()[ebNE_kb]+(1-isDOFBoundary_w.data()[ebNE_kb])*w_ext;
                //VRANS
                porosity_ext = ebqe_porosity_ext.data()[ebNE_kb];
                //
                //calculate the pde coefficients using the solution and the boundary values for the solution
                //
                evaluateCoefficients(eps_rho,
                                     eps_mu,
                                     sigma,
                                     rho_0,
                                     nu_0,
                                     rho_1,
                                     nu_1,
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
                                     mom_v_ham_ext,
                                     dmom_v_ham_grad_p_ext,
                                     mom_w_ham_ext,
                                     dmom_w_ham_grad_p_ext,
                                     ebqe_dragBeam1.data()[ebNE_kb],
                                     ebqe_dragBeam2.data()[ebNE_kb],
                                     ebqe_dragBeam3.data()[ebNE_kb]);
                evaluateCoefficients(eps_rho,
                                     eps_mu,
                                     sigma,
                                     rho_0,
                                     nu_0,
                                     rho_1,
                                     nu_1,
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
                                     bc_mom_v_ham_ext,
                                     bc_dmom_v_ham_grad_p_ext,
                                     bc_mom_w_ham_ext,
                                     bc_dmom_w_ham_grad_p_ext,
                                     ebqe_dragBeam1.data()[ebNE_kb],
                                     ebqe_dragBeam2.data()[ebNE_kb],
                                     ebqe_dragBeam3.data()[ebNE_kb]);

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
                                            ebqe_vf_ext.data()[ebNE_kb],
                                            ebqe_phi_ext.data()[ebNE_kb],
                                            porosity_ext,
                                            c_mu, //mwf hack
                                            ebqe_turb_var_0.data()[ebNE_kb],
                                            ebqe_turb_var_1.data()[ebNE_kb],
                                            turb_var_grad_0_dummy, //not needed
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
                                            bc_ebqe_vf_ext.data()[ebNE_kb],
                                            bc_ebqe_phi_ext.data()[ebNE_kb],
                                            porosity_ext,
                                            c_mu, //mwf hack
                                            ebqe_turb_var_0.data()[ebNE_kb],
                                            ebqe_turb_var_1.data()[ebNE_kb],
                                            turb_var_grad_0_dummy, //not needed
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
                mass_adv_ext[0] -= MOVING_DOMAIN*xt_ext;
                mass_adv_ext[1] -= MOVING_DOMAIN*yt_ext;
                mass_adv_ext[2] -= MOVING_DOMAIN*zt_ext;

                mom_u_adv_ext[0] -= MOVING_DOMAIN*mom_u_acc_ext*xt_ext;
                mom_u_adv_ext[1] -= MOVING_DOMAIN*mom_u_acc_ext*yt_ext;
                mom_u_adv_ext[2] -= MOVING_DOMAIN*mom_u_acc_ext*zt_ext;
                dmom_u_adv_u_ext[0] -= MOVING_DOMAIN*dmom_u_acc_u_ext*xt_ext;
                dmom_u_adv_u_ext[1] -= MOVING_DOMAIN*dmom_u_acc_u_ext*yt_ext;
                dmom_u_adv_u_ext[2] -= MOVING_DOMAIN*dmom_u_acc_u_ext*zt_ext;

                mom_v_adv_ext[0] -= MOVING_DOMAIN*mom_v_acc_ext*xt_ext;
                mom_v_adv_ext[1] -= MOVING_DOMAIN*mom_v_acc_ext*yt_ext;
                mom_v_adv_ext[2] -= MOVING_DOMAIN*mom_v_acc_ext*zt_ext;
                dmom_v_adv_v_ext[0] -= MOVING_DOMAIN*dmom_v_acc_v_ext*xt_ext;
                dmom_v_adv_v_ext[1] -= MOVING_DOMAIN*dmom_v_acc_v_ext*yt_ext;
                dmom_v_adv_v_ext[2] -= MOVING_DOMAIN*dmom_v_acc_v_ext*zt_ext;

                mom_w_adv_ext[0] -= MOVING_DOMAIN*mom_w_acc_ext*xt_ext;
                mom_w_adv_ext[1] -= MOVING_DOMAIN*mom_w_acc_ext*yt_ext;
                mom_w_adv_ext[2] -= MOVING_DOMAIN*mom_w_acc_ext*zt_ext;
                dmom_w_adv_w_ext[0] -= MOVING_DOMAIN*dmom_w_acc_w_ext*xt_ext;
                dmom_w_adv_w_ext[1] -= MOVING_DOMAIN*dmom_w_acc_w_ext*yt_ext;
                dmom_w_adv_w_ext[2] -= MOVING_DOMAIN*dmom_w_acc_w_ext*zt_ext;

                //bc's
                bc_mom_u_adv_ext[0] -= MOVING_DOMAIN*bc_mom_u_acc_ext*xt_ext;
                bc_mom_u_adv_ext[1] -= MOVING_DOMAIN*bc_mom_u_acc_ext*yt_ext;
                bc_mom_u_adv_ext[2] -= MOVING_DOMAIN*bc_mom_u_acc_ext*zt_ext;

                bc_mom_v_adv_ext[0] -= MOVING_DOMAIN*bc_mom_v_acc_ext*xt_ext;
                bc_mom_v_adv_ext[1] -= MOVING_DOMAIN*bc_mom_v_acc_ext*yt_ext;
                bc_mom_v_adv_ext[2] -= MOVING_DOMAIN*bc_mom_v_acc_ext*zt_ext;

                bc_mom_w_adv_ext[0] -= MOVING_DOMAIN*bc_mom_w_acc_ext*xt_ext;
                bc_mom_w_adv_ext[1] -= MOVING_DOMAIN*bc_mom_w_acc_ext*yt_ext;
                bc_mom_w_adv_ext[2] -= MOVING_DOMAIN*bc_mom_w_acc_ext*zt_ext;
                //
                //calculate the numerical fluxes
                //
                ck.calculateGScale(G,normal,h_penalty);
                penalty = useMetrics*C_b*h_penalty + (1.0-useMetrics)*ebqe_penalty_ext.data()[ebNE_kb];
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
                                               bc_p_ext,
                                               bc_mass_adv_ext,
                                               bc_mom_u_adv_ext,
                                               bc_mom_v_adv_ext,
                                               bc_mom_w_adv_ext,
                                               ebqe_bc_flux_mass_ext.data()[ebNE_kb],
                                               ebqe_bc_flux_mom_u_adv_ext.data()[ebNE_kb],
                                               ebqe_bc_flux_mom_v_adv_ext.data()[ebNE_kb],
                                               ebqe_bc_flux_mom_w_adv_ext.data()[ebNE_kb],
                                               p_ext,
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
                                               isDiffusiveFluxBoundary_u.data()[ebNE_kb],
                                               normal,
                                               bc_mom_uv_diff_ten_ext,
                                               bc_v_ext,
                                               0.0,//assume all of the flux gets applied in diagonal component
                                               mom_uv_diff_ten_ext,
                                               grad_v_ext,
                                               v_ext,
                                               penalty,//ebqe_penalty_ext.data()[ebNE_kb],
                                               flux_mom_uv_diff_ext);
                exteriorNumericalDiffusiveFlux(eps_rho,
                                               ebqe_phi_ext.data()[ebNE_kb],
                                               sdInfo_u_w_rowptr.data(),
                                               sdInfo_u_w_colind.data(),
                                               isDOFBoundary_w.data()[ebNE_kb],
                                               isDiffusiveFluxBoundary_u.data()[ebNE_kb],
                                               normal,
                                               bc_mom_uw_diff_ten_ext,
                                               bc_w_ext,
                                               0.0,//see above
                                               mom_uw_diff_ten_ext,
                                               grad_w_ext,
                                               w_ext,
                                               penalty,//ebqe_penalty_ext.data()[ebNE_kb],
                                               flux_mom_uw_diff_ext);
                exteriorNumericalDiffusiveFlux(eps_rho,
                                               ebqe_phi_ext.data()[ebNE_kb],
                                               sdInfo_v_u_rowptr.data(),
                                               sdInfo_v_u_colind.data(),
                                               isDOFBoundary_u.data()[ebNE_kb],
                                               isDiffusiveFluxBoundary_v.data()[ebNE_kb],
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
                exteriorNumericalDiffusiveFlux(eps_rho,
                                               ebqe_phi_ext.data()[ebNE_kb],
                                               sdInfo_v_w_rowptr.data(),
                                               sdInfo_v_w_colind.data(),
                                               isDOFBoundary_w.data()[ebNE_kb],
                                               isDiffusiveFluxBoundary_v.data()[ebNE_kb],
                                               normal,
                                               bc_mom_vw_diff_ten_ext,
                                               bc_w_ext,
                                               0.0,//see above
                                               mom_vw_diff_ten_ext,
                                               grad_w_ext,
                                               w_ext,
                                               penalty,//ebqe_penalty_ext.data()[ebNE_kb],
                                               flux_mom_vw_diff_ext);
                exteriorNumericalDiffusiveFlux(eps_rho,
                                               ebqe_phi_ext.data()[ebNE_kb],
                                               sdInfo_w_u_rowptr.data(),
                                               sdInfo_w_u_colind.data(),
                                               isDOFBoundary_u.data()[ebNE_kb],
                                               isDiffusiveFluxBoundary_w.data()[ebNE_kb],
                                               normal,
                                               bc_mom_wu_diff_ten_ext,
                                               bc_u_ext,
                                               0.0,//see above
                                               mom_wu_diff_ten_ext,
                                               grad_u_ext,
                                               u_ext,
                                               penalty,//ebqe_penalty_ext.data()[ebNE_kb],
                                               flux_mom_wu_diff_ext);
                exteriorNumericalDiffusiveFlux(eps_rho,
                                               ebqe_phi_ext.data()[ebNE_kb],
                                               sdInfo_w_v_rowptr.data(),
                                               sdInfo_w_v_colind.data(),
                                               isDOFBoundary_v.data()[ebNE_kb],
                                               isDiffusiveFluxBoundary_w.data()[ebNE_kb],
                                               normal,
                                               bc_mom_wv_diff_ten_ext,
                                               bc_v_ext,
                                               0.0,//see above
                                               mom_wv_diff_ten_ext,
                                               grad_v_ext,
                                               v_ext,
                                               penalty,//ebqe_penalty_ext.data()[ebNE_kb],
                                               flux_mom_wv_diff_ext);
                exteriorNumericalDiffusiveFlux(eps_rho,
                                               ebqe_phi_ext.data()[ebNE_kb],
                                               sdInfo_w_w_rowptr.data(),
                                               sdInfo_w_w_colind.data(),
                                               isDOFBoundary_w.data()[ebNE_kb],
                                               isDiffusiveFluxBoundary_w.data()[ebNE_kb],
                                               normal,
                                               bc_mom_ww_diff_ten_ext,
                                               bc_w_ext,
                                               ebqe_bc_flux_w_diff_ext.data()[ebNE_kb],
                                               mom_ww_diff_ten_ext,
                                               grad_w_ext,
                                               w_ext,
                                               penalty,//ebqe_penalty_ext.data()[ebNE_kb],
                                               flux_mom_ww_diff_ext);
                flux.data()[ebN*nQuadraturePoints_elementBoundary+kb] = flux_mass_ext;
                //
                //integrate the net force and moment on flagged boundaries
                //
                if (ebN < nElementBoundaries_owned)
                  {
                    force_v_x = flux_mom_uu_diff_ext + flux_mom_uv_diff_ext + flux_mom_uw_diff_ext;
                    force_v_y = flux_mom_vu_diff_ext + flux_mom_vv_diff_ext + flux_mom_vw_diff_ext;
                    force_v_z = flux_mom_wu_diff_ext + flux_mom_wv_diff_ext + flux_mom_ww_diff_ext;

                    force_p_x = p_ext*normal[0];
                    force_p_y = p_ext*normal[1];
                    force_p_z = p_ext*normal[2];

                    force_x = force_p_x + force_v_x;
                    force_y = force_p_y + force_v_y;
                    force_z = force_p_z + force_v_z;

                    r_x = x_ext - barycenters.data()[3*boundaryFlags.data()[ebN]+0];
                    r_y = y_ext - barycenters.data()[3*boundaryFlags.data()[ebN]+1];
                    r_z = z_ext - barycenters.data()[3*boundaryFlags.data()[ebN]+2];

                    wettedAreas.data()[boundaryFlags.data()[ebN]] += dS*(1.0-ebqe_vf_ext.data()[ebNE_kb]);

                    netForces_p.data()[3*boundaryFlags.data()[ebN]+0] += force_p_x*dS*(1.0-ebqe_vf_ext.data()[ebNE_kb]);
                    netForces_p.data()[3*boundaryFlags.data()[ebN]+1] += force_p_y*dS*(1.0-ebqe_vf_ext.data()[ebNE_kb]);
                    netForces_p.data()[3*boundaryFlags.data()[ebN]+2] += force_p_z*dS*(1.0-ebqe_vf_ext.data()[ebNE_kb]);

                    netForces_v.data()[3*boundaryFlags.data()[ebN]+0] += force_v_x*dS*(1.0-ebqe_vf_ext.data()[ebNE_kb]);
                    netForces_v.data()[3*boundaryFlags.data()[ebN]+1] += force_v_y*dS*(1.0-ebqe_vf_ext.data()[ebNE_kb]);
                    netForces_v.data()[3*boundaryFlags.data()[ebN]+2] += force_v_z*dS*(1.0-ebqe_vf_ext.data()[ebNE_kb]);

                    netMoments.data()[3*boundaryFlags.data()[ebN]+0] += (r_y*force_z - r_z*force_y)*dS*(1.0-ebqe_vf_ext.data()[ebNE_kb]);
                    netMoments.data()[3*boundaryFlags.data()[ebN]+1] += (r_z*force_x - r_x*force_z)*dS*(1.0-ebqe_vf_ext.data()[ebNE_kb]);
                    netMoments.data()[3*boundaryFlags.data()[ebN]+2] += (r_x*force_y - r_y*force_x)*dS*(1.0-ebqe_vf_ext.data()[ebNE_kb]);
                  }
                //
                //update residuals
                //
                for (int i=0;i<nDOF_test_element;i++)
                  {
                    elementResidual_p[i] += ck.ExteriorElementBoundaryFlux(flux_mass_ext,p_test_dS[i]);
                    globalConservationError += ck.ExteriorElementBoundaryFlux(flux_mass_ext,p_test_dS[i]);

                    elementResidual_u[i] += ck.ExteriorElementBoundaryFlux(flux_mom_u_adv_ext,vel_test_dS[i])+
                      ck.ExteriorElementBoundaryFlux(flux_mom_uu_diff_ext,vel_test_dS[i])+
                      ck.ExteriorElementBoundaryFlux(flux_mom_uv_diff_ext,vel_test_dS[i])+
                      ck.ExteriorElementBoundaryFlux(flux_mom_uw_diff_ext,vel_test_dS[i])+
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
                                                                 &vel_grad_test_dS[i*nSpace])+
                      ck.ExteriorElementBoundaryDiffusionAdjoint(isDOFBoundary_w.data()[ebNE_kb],
                                                                 isDiffusiveFluxBoundary_u.data()[ebNE_kb],
                                                                 eb_adjoint_sigma,
                                                                 w_ext,
                                                                 bc_w_ext,
                                                                 normal,
                                                                 sdInfo_u_w_rowptr.data(),
                                                                 sdInfo_u_w_colind.data(),
                                                                 mom_uw_diff_ten_ext,
                                                                 &vel_grad_test_dS[i*nSpace]);
                    elementResidual_v[i] += ck.ExteriorElementBoundaryFlux(flux_mom_v_adv_ext,vel_test_dS[i]) +
                      ck.ExteriorElementBoundaryFlux(flux_mom_vu_diff_ext,vel_test_dS[i])+
                      ck.ExteriorElementBoundaryFlux(flux_mom_vv_diff_ext,vel_test_dS[i])+
                      ck.ExteriorElementBoundaryFlux(flux_mom_vw_diff_ext,vel_test_dS[i])+
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
                                                                 &vel_grad_test_dS[i*nSpace])+
                      ck.ExteriorElementBoundaryDiffusionAdjoint(isDOFBoundary_w.data()[ebNE_kb],
                                                                 isDiffusiveFluxBoundary_v.data()[ebNE_kb],
                                                                 eb_adjoint_sigma,
                                                                 w_ext,
                                                                 bc_w_ext,
                                                                 normal,
                                                                 sdInfo_v_w_rowptr.data(),
                                                                 sdInfo_v_w_colind.data(),
                                                                 mom_vw_diff_ten_ext,
                                                                 &vel_grad_test_dS[i*nSpace]);

                    elementResidual_w[i] += ck.ExteriorElementBoundaryFlux(flux_mom_w_adv_ext,vel_test_dS[i]) +
                      ck.ExteriorElementBoundaryFlux(flux_mom_wu_diff_ext,vel_test_dS[i])+
                      ck.ExteriorElementBoundaryFlux(flux_mom_wv_diff_ext,vel_test_dS[i])+
                      ck.ExteriorElementBoundaryFlux(flux_mom_ww_diff_ext,vel_test_dS[i])+
                      ck.ExteriorElementBoundaryDiffusionAdjoint(isDOFBoundary_u.data()[ebNE_kb],
                                                                 isDiffusiveFluxBoundary_w.data()[ebNE_kb],
                                                                 eb_adjoint_sigma,
                                                                 u_ext,
                                                                 bc_u_ext,
                                                                 normal,
                                                                 sdInfo_w_u_rowptr.data(),
                                                                 sdInfo_w_u_colind.data(),
                                                                 mom_wu_diff_ten_ext,
                                                                 &vel_grad_test_dS[i*nSpace])+
                      ck.ExteriorElementBoundaryDiffusionAdjoint(isDOFBoundary_v.data()[ebNE_kb],
                                                                 isDiffusiveFluxBoundary_w.data()[ebNE_kb],
                                                                 eb_adjoint_sigma,
                                                                 v_ext,
                                                                 bc_v_ext,
                                                                 normal,
                                                                 sdInfo_w_v_rowptr.data(),
                                                                 sdInfo_w_v_colind.data(),
                                                                 mom_wv_diff_ten_ext,
                                                                 &vel_grad_test_dS[i*nSpace])+
                      ck.ExteriorElementBoundaryDiffusionAdjoint(isDOFBoundary_w.data()[ebNE_kb],
                                                                 isDiffusiveFluxBoundary_w.data()[ebNE_kb],
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

                elementResidual_p_save.data()[eN_i] +=  elementResidual_p[i];

                globalResidual.data()[offset_p+stride_p*p_l2g.data()[eN_i]]+=elementResidual_p[i];
                globalResidual.data()[offset_u+stride_u*vel_l2g.data()[eN_i]]+=elementResidual_u[i];
                globalResidual.data()[offset_v+stride_v*vel_l2g.data()[eN_i]]+=elementResidual_v[i];
                globalResidual.data()[offset_w+stride_w*vel_l2g.data()[eN_i]]+=elementResidual_w[i];
              }//i
          }//ebNE
      }

      void calculateJacobian(arguments_dict& args)
      {
        proteus::pyarray<double>& mesh_trial_ref = args.array<double>("mesh_trial_ref");
        proteus::pyarray<double>& mesh_grad_trial_ref = args.array<double>("mesh_grad_trial_ref");
        proteus::pyarray<double>& mesh_dof = args.array<double>("mesh_dof");
        proteus::pyarray<double>& mesh_velocity_dof = args.array<double>("mesh_velocity_dof");
        double MOVING_DOMAIN = args.scalar<double>("MOVING_DOMAIN");
        proteus::pyarray<int>& mesh_l2g = args.array<int>("mesh_l2g");
        proteus::pyarray<double>& dV_ref = args.array<double>("dV_ref");
        proteus::pyarray<double>& p_trial_ref = args.array<double>("p_trial_ref");
        proteus::pyarray<double>& p_grad_trial_ref = args.array<double>("p_grad_trial_ref");
        proteus::pyarray<double>& p_test_ref = args.array<double>("p_test_ref");
        proteus::pyarray<double>& p_grad_test_ref = args.array<double>("p_grad_test_ref");
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
        int nElements_global = args.scalar<int>("nElements_global");
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
        const proteus::pyarray<double>& phi_solid = args.array<double>("phi_solid");
        const proteus::pyarray<double>& q_velocity_solid = args.array<double>("q_velocity_solid");
        const proteus::pyarray<double>& q_porosity = args.array<double>("q_porosity");
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
        proteus::pyarray<double>& normal_phi = args.array<double>("normal_phi");
        proteus::pyarray<double>& kappa_phi = args.array<double>("kappa_phi");
        proteus::pyarray<double>& q_mom_u_acc_beta_bdf = args.array<double>("q_mom_u_acc_beta_bdf");
        proteus::pyarray<double>& q_mom_v_acc_beta_bdf = args.array<double>("q_mom_v_acc_beta_bdf");
        proteus::pyarray<double>& q_mom_w_acc_beta_bdf = args.array<double>("q_mom_w_acc_beta_bdf");
        proteus::pyarray<double>& q_velocity_sge = args.array<double>("q_velocity_sge");
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
        proteus::pyarray<int>& elementBoundaryElementsArray = args.array<int>("elementBoundaryElementsArray");
        proteus::pyarray<int>& elementBoundaryLocalElementBoundariesArray = args.array<int>("elementBoundaryLocalElementBoundariesArray");
        proteus::pyarray<double>& ebqe_vf_ext = args.array<double>("ebqe_vf_ext");
        proteus::pyarray<double>& bc_ebqe_vf_ext = args.array<double>("bc_ebqe_vf_ext");
        proteus::pyarray<double>& ebqe_phi_ext = args.array<double>("ebqe_phi_ext");
        proteus::pyarray<double>& bc_ebqe_phi_ext = args.array<double>("bc_ebqe_phi_ext");
        proteus::pyarray<double>& ebqe_normal_phi_ext = args.array<double>("ebqe_normal_phi_ext");
        proteus::pyarray<double>& ebqe_kappa_phi_ext = args.array<double>("ebqe_kappa_phi_ext");
        const proteus::pyarray<double>& ebqe_porosity_ext = args.array<double>("ebqe_porosity_ext");
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
        proteus::pyarray<double>& q_dragBeam1 = args.array<double>("q_dragBeam1");
        proteus::pyarray<double>& q_dragBeam2 = args.array<double>("q_dragBeam2");
        proteus::pyarray<double>& q_dragBeam3 = args.array<double>("q_dragBeam3");
        proteus::pyarray<double>& ebqe_dragBeam1 = args.array<double>("ebqe_dragBeam1");
        proteus::pyarray<double>& ebqe_dragBeam2 = args.array<double>("ebqe_dragBeam2");
        proteus::pyarray<double>& ebqe_dragBeam3 = args.array<double>("ebqe_dragBeam3");
        //
        //loop over elements to compute volume integrals and load them into the element Jacobians and global Jacobian
        //
        for(int eN=0;eN<nElements_global;eN++)
          {
            double eps_rho,eps_mu;

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
            for  (int k=0;k<nQuadraturePoints_element;k++)
              {
                int eN_k = eN*nQuadraturePoints_element+k, //index to a scalar at a quadrature point
                  eN_k_nSpace = eN_k*nSpace,
                  eN_nDOF_trial_element = eN*nDOF_trial_element; //index to a vector at a quadrature point

                //declare local storage
                double p=0.0,u=0.0,v=0.0,w=0.0,
                  grad_p[nSpace],grad_u[nSpace],grad_v[nSpace],grad_w[nSpace],
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
                  mom_v_ham=0.0,
                  dmom_v_ham_grad_p[nSpace],
                  mom_w_ham=0.0,
                  dmom_w_ham_grad_p[nSpace],
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
                dV = fabs(jacDet)*dV_ref.data()[k];
                ck.calculateG(jacInv,G,G_dd_G,tr_G);
                //ck.calculateGScale(G,&normal_phi.data()[eN_k_nSpace],h_phi);

                eps_rho = epsFact_rho*(useMetrics*h_phi+(1.0-useMetrics)*elementDiameter.data()[eN]);
                eps_mu  = epsFact_mu *(useMetrics*h_phi+(1.0-useMetrics)*elementDiameter.data()[eN]);

                //get the trial function gradients
                ck.gradTrialFromRef(&p_grad_trial_ref.data()[k*nDOF_trial_element*nSpace],jacInv,p_grad_trial);
                ck.gradTrialFromRef(&vel_grad_trial_ref.data()[k*nDOF_trial_element*nSpace],jacInv,vel_grad_trial);
                //get the solution
                ck.valFromDOF(p_dof.data(),&p_l2g.data()[eN_nDOF_trial_element],&p_trial_ref.data()[k*nDOF_trial_element],p);
                ck.valFromDOF(u_dof.data(),&vel_l2g.data()[eN_nDOF_trial_element],&vel_trial_ref.data()[k*nDOF_trial_element],u);
                ck.valFromDOF(v_dof.data(),&vel_l2g.data()[eN_nDOF_trial_element],&vel_trial_ref.data()[k*nDOF_trial_element],v);
                ck.valFromDOF(w_dof.data(),&vel_l2g.data()[eN_nDOF_trial_element],&vel_trial_ref.data()[k*nDOF_trial_element],w);
                //get the solution gradients
                ck.gradFromDOF(p_dof.data(),&p_l2g.data()[eN_nDOF_trial_element],p_grad_trial,grad_p);
                ck.gradFromDOF(u_dof.data(),&vel_l2g.data()[eN_nDOF_trial_element],vel_grad_trial,grad_u);
                ck.gradFromDOF(v_dof.data(),&vel_l2g.data()[eN_nDOF_trial_element],vel_grad_trial,grad_v);
                ck.gradFromDOF(w_dof.data(),&vel_l2g.data()[eN_nDOF_trial_element],vel_grad_trial,grad_w);
                //precalculate test function products with integration weights
                for (int j=0;j<nDOF_trial_element;j++)
                  {
                    p_test_dV[j] = p_test_ref.data()[k*nDOF_trial_element+j]*dV;
                    vel_test_dV[j] = vel_test_ref.data()[k*nDOF_trial_element+j]*dV;
                    for (int I=0;I<nSpace;I++)
                      {
                        p_grad_test_dV[j*nSpace+I]   = p_grad_trial[j*nSpace+I]*dV;//cek warning won't work for Petrov-Galerkin
                        vel_grad_test_dV[j*nSpace+I] = vel_grad_trial[j*nSpace+I]*dV;//cek warning won't work for Petrov-Galerkin}
                      }
                  }
                //
                //VRANS
                porosity = q_porosity.data()[eN_k];
                //
                //
                //calculate pde coefficients and derivatives at quadrature points
                //
                evaluateCoefficients(eps_rho,
                                     eps_mu,
                                     sigma,
                                     rho_0,
                                     nu_0,
                                     rho_1,
                                     nu_1,
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
                                     mom_v_ham,
                                     dmom_v_ham_grad_p,
                                     mom_w_ham,
                                     dmom_w_ham_grad_p,
                                     q_dragBeam1.data()[eN_k],
                                     q_dragBeam2.data()[eN_k],
                                     q_dragBeam3.data()[eN_k]);
                //VRANS
                mass_source = q_mass_source.data()[eN_k];
                //todo: decide if these should be lagged or not
                updateDarcyForchheimerTerms_Ergun(/* linearDragFactor, */
                                                  /* nonlinearDragFactor, */
                                                  /* porosity, */
                                                  /* meanGrainSize, */
                                                  q_dragAlpha.data()[eN_k],
                                                  q_dragBeta.data()[eN_k],
                                                  eps_rho,
                                                  eps_mu,
                                                  rho_0,
                                                  nu_0,
                                                  rho_1,
                                                  nu_1,
                                                  useVF,
                                                  vf.data()[eN_k],
                                                  phi.data()[eN_k],
                                                  u,//q_velocity_sge.data()[eN_k_nSpace+0],//u
                                                  v,//q_velocity_sge.data()[eN_k_nSpace+1],//v
                                                  w,//q_velocity_sge.data()[eN_k_nSpace+2],//w
                                                  eps_solid.data()[0],
                                                  phi_solid.data()[eN_k],
                                                  q_velocity_solid.data()[eN_k_nSpace+0],
                                                  q_velocity_solid.data()[eN_k_nSpace+1],
                                                  q_velocity_solid.data()[eN_k_nSpace+2],
                                                  mom_u_source,
                                                  mom_v_source,
                                                  mom_w_source,
                                                  dmom_u_source,
                                                  dmom_v_source,
                                                  dmom_w_source);
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
                                            vf.data()[eN_k],
                                            phi.data()[eN_k],
                                            porosity,
                                            c_mu, //mwf hack
                                            q_turb_var_0.data()[eN_k],
                                            q_turb_var_1.data()[eN_k],
                                            &q_turb_var_grad_0.data()[eN_k_nSpace],
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
                mass_adv[0] -= MOVING_DOMAIN*xt;
                mass_adv[1] -= MOVING_DOMAIN*yt;
                mass_adv[2] -= MOVING_DOMAIN*zt;

                mom_u_adv[0] -= MOVING_DOMAIN*mom_u_acc*xt;
                mom_u_adv[1] -= MOVING_DOMAIN*mom_u_acc*yt;
                mom_u_adv[2] -= MOVING_DOMAIN*mom_u_acc*zt;
                dmom_u_adv_u[0] -= MOVING_DOMAIN*dmom_u_acc_u*xt;
                dmom_u_adv_u[1] -= MOVING_DOMAIN*dmom_u_acc_u*yt;
                dmom_u_adv_u[2] -= MOVING_DOMAIN*dmom_u_acc_u*zt;

                mom_v_adv[0] -= MOVING_DOMAIN*mom_v_acc*xt;
                mom_v_adv[1] -= MOVING_DOMAIN*mom_v_acc*yt;
                mom_v_adv[2] -= MOVING_DOMAIN*mom_v_acc*zt;
                dmom_v_adv_v[0] -= MOVING_DOMAIN*dmom_v_acc_v*xt;
                dmom_v_adv_v[1] -= MOVING_DOMAIN*dmom_v_acc_v*yt;
                dmom_v_adv_v[2] -= MOVING_DOMAIN*dmom_v_acc_v*zt;

                mom_w_adv[0] -= MOVING_DOMAIN*mom_w_acc*xt;
                mom_w_adv[1] -= MOVING_DOMAIN*mom_w_acc*yt;
                mom_w_adv[2] -= MOVING_DOMAIN*mom_w_acc*zt;
                dmom_w_adv_w[0] -= MOVING_DOMAIN*dmom_w_acc_w*xt;
                dmom_w_adv_w[1] -= MOVING_DOMAIN*dmom_w_acc_w*yt;
                dmom_w_adv_w[2] -= MOVING_DOMAIN*dmom_w_acc_w*zt;
                //
                //calculate time derivatives
                //
                ck.bdf(alphaBDF,
                       q_mom_u_acc_beta_bdf.data()[eN_k],
                       mom_u_acc,
                       dmom_u_acc_u,
                       mom_u_acc_t,
                       dmom_u_acc_u_t);
                ck.bdf(alphaBDF,
                       q_mom_v_acc_beta_bdf.data()[eN_k],
                       mom_v_acc,
                       dmom_v_acc_v,
                       mom_v_acc_t,
                       dmom_v_acc_v_t);
                ck.bdf(alphaBDF,
                       q_mom_w_acc_beta_bdf.data()[eN_k],
                       mom_w_acc,
                       dmom_w_acc_w,
                       mom_w_acc_t,
                       dmom_w_acc_w_t);
                //
                //calculate subgrid error contribution to the Jacobian (strong residual, adjoint, jacobian of strong residual)
                //
                dmom_adv_sge[0] = dmom_u_acc_u*(q_velocity_sge.data()[eN_k_nSpace+0] - MOVING_DOMAIN*xt);
                dmom_adv_sge[1] = dmom_u_acc_u*(q_velocity_sge.data()[eN_k_nSpace+1] - MOVING_DOMAIN*yt);
                dmom_adv_sge[2] = dmom_u_acc_u*(q_velocity_sge.data()[eN_k_nSpace+2] - MOVING_DOMAIN*zt);
                //
                //calculate strong residual
                //
                pdeResidual_p = ck.Advection_strong(dmass_adv_u,grad_u) +
                  ck.Advection_strong(dmass_adv_v,grad_v) +
                  ck.Advection_strong(dmass_adv_w,grad_w);

                pdeResidual_u = ck.Mass_strong(mom_u_acc_t) +
                  ck.Advection_strong(dmom_adv_sge,grad_u) +
                  ck.Hamiltonian_strong(dmom_u_ham_grad_p,grad_p) +
                  ck.Reaction_strong(mom_u_source);

                pdeResidual_v = ck.Mass_strong(mom_v_acc_t) +
                  ck.Advection_strong(dmom_adv_sge,grad_v) +
                  ck.Hamiltonian_strong(dmom_v_ham_grad_p,grad_p) +
                  ck.Reaction_strong(mom_v_source);

                pdeResidual_w = ck.Mass_strong(mom_w_acc_t) +
                  ck.Advection_strong(dmom_adv_sge,grad_w) +
                  ck.Hamiltonian_strong(dmom_w_ham_grad_p,grad_p) +
                  ck.Reaction_strong(mom_w_source);

                //calculate the Jacobian of strong residual
                for (int j=0;j<nDOF_trial_element;j++)
                  {
                    int j_nSpace = j*nSpace;
                    dpdeResidual_p_u[j]=ck.AdvectionJacobian_strong(dmass_adv_u,&vel_grad_trial[j_nSpace]);
                    dpdeResidual_p_v[j]=ck.AdvectionJacobian_strong(dmass_adv_v,&vel_grad_trial[j_nSpace]);
                    dpdeResidual_p_w[j]=ck.AdvectionJacobian_strong(dmass_adv_w,&vel_grad_trial[j_nSpace]);

                    dpdeResidual_u_p[j]=ck.HamiltonianJacobian_strong(dmom_u_ham_grad_p,&p_grad_trial[j_nSpace]);
                    dpdeResidual_u_u[j]=ck.MassJacobian_strong(dmom_u_acc_u_t,vel_trial_ref.data()[k*nDOF_trial_element+j]) +
                      ck.AdvectionJacobian_strong(dmom_adv_sge,&vel_grad_trial[j_nSpace]);

                    dpdeResidual_v_p[j]=ck.HamiltonianJacobian_strong(dmom_v_ham_grad_p,&p_grad_trial[j_nSpace]);
                    dpdeResidual_v_v[j]=ck.MassJacobian_strong(dmom_v_acc_v_t,vel_trial_ref.data()[k*nDOF_trial_element+j]) +
                      ck.AdvectionJacobian_strong(dmom_adv_sge,&vel_grad_trial[j_nSpace]);

                    dpdeResidual_w_p[j]=ck.HamiltonianJacobian_strong(dmom_w_ham_grad_p,&p_grad_trial[j_nSpace]);
                    dpdeResidual_w_w[j]=ck.MassJacobian_strong(dmom_w_acc_w_t,vel_trial_ref.data()[k*nDOF_trial_element+j]) +
                      ck.AdvectionJacobian_strong(dmom_adv_sge,&vel_grad_trial[j_nSpace]);

                    //VRANS account for drag terms, diagonal only here ... decide if need off diagonal terms too
                    dpdeResidual_u_u[j]+= ck.ReactionJacobian_strong(dmom_u_source[0],vel_trial_ref.data()[k*nDOF_trial_element+j]);
                    dpdeResidual_v_v[j]+= ck.ReactionJacobian_strong(dmom_v_source[1],vel_trial_ref.data()[k*nDOF_trial_element+j]);
                    dpdeResidual_w_w[j]+= ck.ReactionJacobian_strong(dmom_w_source[2],vel_trial_ref.data()[k*nDOF_trial_element+j]);
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
                                          dmom_u_acc_u_t,
                                          dmom_adv_sge,
                                          mom_uu_diff_ten[1],
                                          dmom_u_ham_grad_p[0],
                                          tau_v1,
                                          tau_p1,
                                          q_cfl.data()[eN_k]);


                tau_v = useMetrics*tau_v1+(1.0-useMetrics)*tau_v0;
                tau_p = useMetrics*tau_p1+(1.0-useMetrics)*tau_p0;

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
                dmom_adv_star[2] = dmom_u_acc_u*(q_velocity_sge.data()[eN_k_nSpace+2] - MOVING_DOMAIN*zt + useRBLES*subgridError_w);

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


                //cek todo add RBLES terms consistent to residual modifications or ignore the partials w.r.t the additional RBLES terms
                for(int i=0;i<nDOF_test_element;i++)
                  {
                    int i_nSpace = i*nSpace;
                    for(int j=0;j<nDOF_trial_element;j++)
                      {
                        int j_nSpace = j*nSpace;
                        elementJacobian_p_p[i][j] += ck.SubgridErrorJacobian(dsubgridError_u_p[j],Lstar_u_p[i]) +
                          ck.SubgridErrorJacobian(dsubgridError_v_p[j],Lstar_v_p[i]) +
                          ck.SubgridErrorJacobian(dsubgridError_w_p[j],Lstar_w_p[i]);

                        elementJacobian_p_u[i][j] += ck.AdvectionJacobian_weak(dmass_adv_u,vel_trial_ref.data()[k*nDOF_trial_element+j],&p_grad_test_dV[i_nSpace]) +
                          ck.SubgridErrorJacobian(dsubgridError_u_u[j],Lstar_u_p[i]);
                        elementJacobian_p_v[i][j] += ck.AdvectionJacobian_weak(dmass_adv_v,vel_trial_ref.data()[k*nDOF_trial_element+j],&p_grad_test_dV[i_nSpace]) +
                          ck.SubgridErrorJacobian(dsubgridError_v_v[j],Lstar_v_p[i]);
                        elementJacobian_p_w[i][j] += ck.AdvectionJacobian_weak(dmass_adv_w,vel_trial_ref.data()[k*nDOF_trial_element+j],&p_grad_test_dV[i_nSpace]) +
                          ck.SubgridErrorJacobian(dsubgridError_w_w[j],Lstar_w_p[i]);

                        elementJacobian_u_p[i][j] += ck.HamiltonianJacobian_weak(dmom_u_ham_grad_p,&p_grad_trial[j_nSpace],vel_test_dV[i]) +
                          ck.SubgridErrorJacobian(dsubgridError_u_p[j],Lstar_u_u[i]);
                        elementJacobian_u_u[i][j] += ck.MassJacobian_weak(dmom_u_acc_u_t,vel_trial_ref.data()[k*nDOF_trial_element+j],vel_test_dV[i]) +
                          ck.AdvectionJacobian_weak(dmom_u_adv_u,vel_trial_ref.data()[k*nDOF_trial_element+j],&vel_grad_test_dV[i_nSpace]) +
                          ck.SimpleDiffusionJacobian_weak(sdInfo_u_u_rowptr.data(),sdInfo_u_u_colind.data(),mom_uu_diff_ten,&vel_grad_trial[j_nSpace],&vel_grad_test_dV[i_nSpace]) +
                          //VRANS
                          ck.ReactionJacobian_weak(dmom_u_source[0],vel_trial_ref.data()[k*nDOF_trial_element+j],vel_test_dV[i]) +
                          //
                          ck.SubgridErrorJacobian(dsubgridError_p_u[j],Lstar_p_u[i]) +
                          ck.SubgridErrorJacobian(dsubgridError_u_u[j],Lstar_u_u[i]) +
                          ck.NumericalDiffusionJacobian(q_numDiff_u_last.data()[eN_k],&vel_grad_trial[j_nSpace],&vel_grad_test_dV[i_nSpace]);
                        elementJacobian_u_v[i][j] += ck.AdvectionJacobian_weak(dmom_u_adv_v,vel_trial_ref.data()[k*nDOF_trial_element+j],&vel_grad_test_dV[i_nSpace]) +
                          ck.SimpleDiffusionJacobian_weak(sdInfo_u_v_rowptr.data(),sdInfo_u_v_colind.data(),mom_uv_diff_ten,&vel_grad_trial[j_nSpace],&vel_grad_test_dV[i_nSpace]) +
                          //VRANS
                          ck.ReactionJacobian_weak(dmom_u_source[1],vel_trial_ref.data()[k*nDOF_trial_element+j],vel_test_dV[i]) +
                          //
                          ck.SubgridErrorJacobian(dsubgridError_p_v[j],Lstar_p_u[i]);
                        elementJacobian_u_w[i][j] += ck.AdvectionJacobian_weak(dmom_u_adv_w,vel_trial_ref.data()[k*nDOF_trial_element+j],&vel_grad_test_dV[i_nSpace]) +
                          ck.SimpleDiffusionJacobian_weak(sdInfo_u_w_rowptr.data(),sdInfo_u_w_colind.data(),mom_uw_diff_ten,&vel_grad_trial[j_nSpace],&vel_grad_test_dV[i_nSpace]) +
                          //VRANS
                          ck.ReactionJacobian_weak(dmom_u_source[2],vel_trial_ref.data()[k*nDOF_trial_element+j],vel_test_dV[i]) +
                          //
                          ck.SubgridErrorJacobian(dsubgridError_p_w[j],Lstar_p_u[i]);

                        elementJacobian_v_p[i][j] += ck.HamiltonianJacobian_weak(dmom_v_ham_grad_p,&p_grad_trial[j_nSpace],vel_test_dV[i]) +
                          ck.SubgridErrorJacobian(dsubgridError_v_p[j],Lstar_v_v[i]);
                        elementJacobian_v_u[i][j] += ck.AdvectionJacobian_weak(dmom_v_adv_u,vel_trial_ref.data()[k*nDOF_trial_element+j],&vel_grad_test_dV[i_nSpace]) +
                          ck.SimpleDiffusionJacobian_weak(sdInfo_v_u_rowptr.data(),sdInfo_v_u_colind.data(),mom_vu_diff_ten,&vel_grad_trial[j_nSpace],&vel_grad_test_dV[i_nSpace]) +
                          //VRANS
                          ck.ReactionJacobian_weak(dmom_v_source[0],vel_trial_ref.data()[k*nDOF_trial_element+j],vel_test_dV[i]) +
                          //
                          ck.SubgridErrorJacobian(dsubgridError_p_u[j],Lstar_p_v[i]);
                        elementJacobian_v_v[i][j] += ck.MassJacobian_weak(dmom_v_acc_v_t,vel_trial_ref.data()[k*nDOF_trial_element+j],vel_test_dV[i]) +
                          ck.AdvectionJacobian_weak(dmom_v_adv_v,vel_trial_ref.data()[k*nDOF_trial_element+j],&vel_grad_test_dV[i_nSpace]) +
                          ck.SimpleDiffusionJacobian_weak(sdInfo_v_v_rowptr.data(),sdInfo_v_v_colind.data(),mom_vv_diff_ten,&vel_grad_trial[j_nSpace],&vel_grad_test_dV[i_nSpace]) +
                          //VRANS
                          ck.ReactionJacobian_weak(dmom_v_source[1],vel_trial_ref.data()[k*nDOF_trial_element+j],vel_test_dV[i]) +
                          //
                          ck.SubgridErrorJacobian(dsubgridError_p_v[j],Lstar_p_v[i]) +
                          ck.SubgridErrorJacobian(dsubgridError_v_v[j],Lstar_v_v[i]) +
                          ck.NumericalDiffusionJacobian(q_numDiff_v_last.data()[eN_k],&vel_grad_trial[j_nSpace],&vel_grad_test_dV[i_nSpace]);
                        elementJacobian_v_w[i][j] += ck.AdvectionJacobian_weak(dmom_v_adv_w,vel_trial_ref.data()[k*nDOF_trial_element+j],&vel_grad_test_dV[i_nSpace]) +
                          ck.SimpleDiffusionJacobian_weak(sdInfo_v_w_rowptr.data(),sdInfo_v_w_colind.data(),mom_vw_diff_ten,&vel_grad_trial[j_nSpace],&vel_grad_test_dV[i_nSpace]) +
                          //VRANS
                          ck.ReactionJacobian_weak(dmom_v_source[2],vel_trial_ref.data()[k*nDOF_trial_element+j],vel_test_dV[i]) +
                          //
                          ck.SubgridErrorJacobian(dsubgridError_p_w[j],Lstar_p_v[i]);

                        elementJacobian_w_p[i][j] += ck.HamiltonianJacobian_weak(dmom_w_ham_grad_p,&p_grad_trial[j_nSpace],vel_test_dV[i]) +
                          ck.SubgridErrorJacobian(dsubgridError_w_p[j],Lstar_w_w[i]);
                        elementJacobian_w_u[i][j] += ck.AdvectionJacobian_weak(dmom_w_adv_u,vel_trial_ref.data()[k*nDOF_trial_element+j],&vel_grad_test_dV[i_nSpace]) +
                          ck.SimpleDiffusionJacobian_weak(sdInfo_w_u_rowptr.data(),sdInfo_w_u_colind.data(),mom_wu_diff_ten,&vel_grad_trial[j_nSpace],&vel_grad_test_dV[i_nSpace]) +
                          //VRANS
                          ck.ReactionJacobian_weak(dmom_w_source[0],vel_trial_ref.data()[k*nDOF_trial_element+j],vel_test_dV[i]) +
                          //
                          ck.SubgridErrorJacobian(dsubgridError_p_u[j],Lstar_p_w[i]);
                        elementJacobian_w_v[i][j] += ck.AdvectionJacobian_weak(dmom_w_adv_v,vel_trial_ref.data()[k*nDOF_trial_element+j],&vel_grad_test_dV[i_nSpace]) +
                          ck.SimpleDiffusionJacobian_weak(sdInfo_w_v_rowptr.data(),sdInfo_w_v_colind.data(),mom_wv_diff_ten,&vel_grad_trial[j_nSpace],&vel_grad_test_dV[i_nSpace]) +
                          //VRANS
                          ck.ReactionJacobian_weak(dmom_w_source[1],vel_trial_ref.data()[k*nDOF_trial_element+j],vel_test_dV[i]) +
                          //
                          ck.SubgridErrorJacobian(dsubgridError_p_v[j],Lstar_p_w[i]);
                        elementJacobian_w_w[i][j] += ck.MassJacobian_weak(dmom_w_acc_w_t,vel_trial_ref.data()[k*nDOF_trial_element+j],vel_test_dV[i]) +
                          ck.AdvectionJacobian_weak(dmom_w_adv_w,vel_trial_ref.data()[k*nDOF_trial_element+j],&vel_grad_test_dV[i_nSpace]) +
                          ck.SimpleDiffusionJacobian_weak(sdInfo_w_w_rowptr.data(),sdInfo_w_w_colind.data(),mom_ww_diff_ten,&vel_grad_trial[j_nSpace],&vel_grad_test_dV[i_nSpace]) +
                          //VRANS
                          ck.ReactionJacobian_weak(dmom_w_source[2],vel_trial_ref.data()[k*nDOF_trial_element+j],vel_test_dV[i]) +
                          //
                          ck.SubgridErrorJacobian(dsubgridError_p_w[j],Lstar_p_w[i]) +
                          ck.SubgridErrorJacobian(dsubgridError_w_w[j],Lstar_w_w[i]) +
                          ck.NumericalDiffusionJacobian(q_numDiff_w_last.data()[eN_k],&vel_grad_trial[j_nSpace],&vel_grad_test_dV[i_nSpace]);
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
                    globalJacobian.data()[csrRowIndeces_p_p.data()[eN_i] + csrColumnOffsets_p_p.data()[eN_i_j]] += elementJacobian_p_p[i][j];
                    globalJacobian.data()[csrRowIndeces_p_u.data()[eN_i] + csrColumnOffsets_p_u.data()[eN_i_j]] += elementJacobian_p_u[i][j];
                    globalJacobian.data()[csrRowIndeces_p_v.data()[eN_i] + csrColumnOffsets_p_v.data()[eN_i_j]] += elementJacobian_p_v[i][j];
                    globalJacobian.data()[csrRowIndeces_p_w.data()[eN_i] + csrColumnOffsets_p_w.data()[eN_i_j]] += elementJacobian_p_w[i][j];

                    globalJacobian.data()[csrRowIndeces_u_p.data()[eN_i] + csrColumnOffsets_u_p.data()[eN_i_j]] += elementJacobian_u_p[i][j];
                    globalJacobian.data()[csrRowIndeces_u_u.data()[eN_i] + csrColumnOffsets_u_u.data()[eN_i_j]] += elementJacobian_u_u[i][j];
                    globalJacobian.data()[csrRowIndeces_u_v.data()[eN_i] + csrColumnOffsets_u_v.data()[eN_i_j]] += elementJacobian_u_v[i][j];
                    globalJacobian.data()[csrRowIndeces_u_w.data()[eN_i] + csrColumnOffsets_u_w.data()[eN_i_j]] += elementJacobian_u_w[i][j];

                    globalJacobian.data()[csrRowIndeces_v_p.data()[eN_i] + csrColumnOffsets_v_p.data()[eN_i_j]] += elementJacobian_v_p[i][j];
                    globalJacobian.data()[csrRowIndeces_v_u.data()[eN_i] + csrColumnOffsets_v_u.data()[eN_i_j]] += elementJacobian_v_u[i][j];
                    globalJacobian.data()[csrRowIndeces_v_v.data()[eN_i] + csrColumnOffsets_v_v.data()[eN_i_j]] += elementJacobian_v_v[i][j];
                    globalJacobian.data()[csrRowIndeces_v_w.data()[eN_i] + csrColumnOffsets_v_w.data()[eN_i_j]] += elementJacobian_v_w[i][j];

                    globalJacobian.data()[csrRowIndeces_w_p.data()[eN_i] + csrColumnOffsets_w_p.data()[eN_i_j]] += elementJacobian_w_p[i][j];
                    globalJacobian.data()[csrRowIndeces_w_u.data()[eN_i] + csrColumnOffsets_w_u.data()[eN_i_j]] += elementJacobian_w_u[i][j];
                    globalJacobian.data()[csrRowIndeces_w_v.data()[eN_i] + csrColumnOffsets_w_v.data()[eN_i_j]] += elementJacobian_w_v[i][j];
                    globalJacobian.data()[csrRowIndeces_w_w.data()[eN_i] + csrColumnOffsets_w_w.data()[eN_i_j]] += elementJacobian_w_w[i][j];
                  }//j
              }//i
          }//elements
        //
        //loop over exterior element boundaries to compute the surface integrals and load them into the global Jacobian
        //
        for (int ebNE = 0; ebNE < nExteriorElementBoundaries_global; ebNE++)
          {
            int ebN = exteriorElementBoundariesArray.data()[ebNE],
              eN  = elementBoundaryElementsArray.data()[ebN*2+0],
              eN_nDOF_trial_element = eN*nDOF_trial_element,
              ebN_local = elementBoundaryLocalElementBoundariesArray.data()[ebN*2+0];
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
                  mom_v_ham_ext=0.0,
                  dmom_v_ham_grad_p_ext[nSpace],
                  mom_w_ham_ext=0.0,
                  dmom_w_ham_grad_p_ext[nSpace],
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
                  bc_mom_v_ham_ext=0.0,
                  bc_dmom_v_ham_grad_p_ext[nSpace],
                  bc_mom_w_ham_ext=0.0,
                  bc_dmom_w_ham_grad_p_ext[nSpace],
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
                //xt_ext=0.0;yt_ext=0.0;zt_ext=0.0;
                //std::cout<<"xt_ext "<<xt_ext<<'\t'<<yt_ext<<'\t'<<zt_ext<<std::endl;
                dS = ((1.0-MOVING_DOMAIN)*metricTensorDetSqrt + MOVING_DOMAIN*integralScaling)*dS_ref.data()[kb];
                ck.calculateG(jacInv_ext,G,G_dd_G,tr_G);
                ck.calculateGScale(G,&ebqe_normal_phi_ext.data()[ebNE_kb_nSpace],h_phi);

                eps_rho = epsFact_rho*(useMetrics*h_phi+(1.0-useMetrics)*elementDiameter.data()[eN]);
                eps_mu  = epsFact_mu *(useMetrics*h_phi+(1.0-useMetrics)*elementDiameter.data()[eN]);

                //compute shape and solution information
                //shape
                ck.gradTrialFromRef(&p_grad_trial_trace_ref.data()[ebN_local_kb_nSpace*nDOF_trial_element],jacInv_ext,p_grad_trial_trace);
                ck.gradTrialFromRef(&vel_grad_trial_trace_ref.data()[ebN_local_kb_nSpace*nDOF_trial_element],jacInv_ext,vel_grad_trial_trace);
                //solution and gradients
                ck.valFromDOF(p_dof.data(),&p_l2g.data()[eN_nDOF_trial_element],&p_trial_trace_ref.data()[ebN_local_kb*nDOF_test_element],p_ext);
                ck.valFromDOF(u_dof.data(),&vel_l2g.data()[eN_nDOF_trial_element],&vel_trial_trace_ref.data()[ebN_local_kb*nDOF_test_element],u_ext);
                ck.valFromDOF(v_dof.data(),&vel_l2g.data()[eN_nDOF_trial_element],&vel_trial_trace_ref.data()[ebN_local_kb*nDOF_test_element],v_ext);
                ck.valFromDOF(w_dof.data(),&vel_l2g.data()[eN_nDOF_trial_element],&vel_trial_trace_ref.data()[ebN_local_kb*nDOF_test_element],w_ext);
                ck.gradFromDOF(p_dof.data(),&p_l2g.data()[eN_nDOF_trial_element],p_grad_trial_trace,grad_p_ext);
                ck.gradFromDOF(u_dof.data(),&vel_l2g.data()[eN_nDOF_trial_element],vel_grad_trial_trace,grad_u_ext);
                ck.gradFromDOF(v_dof.data(),&vel_l2g.data()[eN_nDOF_trial_element],vel_grad_trial_trace,grad_v_ext);
                ck.gradFromDOF(w_dof.data(),&vel_l2g.data()[eN_nDOF_trial_element],vel_grad_trial_trace,grad_w_ext);
                //precalculate test function products with integration weights
                for (int j=0;j<nDOF_trial_element;j++)
                  {
                    p_test_dS[j] = p_test_trace_ref.data()[ebN_local_kb*nDOF_test_element+j]*dS;
                    vel_test_dS[j] = vel_test_trace_ref.data()[ebN_local_kb*nDOF_test_element+j]*dS;
                    for (int I=0;I<nSpace;I++)
                      vel_grad_test_dS[j*nSpace+I] = vel_grad_trial_trace[j*nSpace+I]*dS;//cek hack, using trial
                  }
                //
                //load the boundary values
                //
                bc_p_ext = isDOFBoundary_p.data()[ebNE_kb]*ebqe_bc_p_ext.data()[ebNE_kb]+(1-isDOFBoundary_p.data()[ebNE_kb])*p_ext;
                bc_u_ext = isDOFBoundary_u.data()[ebNE_kb]*ebqe_bc_u_ext.data()[ebNE_kb]+(1-isDOFBoundary_u.data()[ebNE_kb])*u_ext;
                bc_v_ext = isDOFBoundary_v.data()[ebNE_kb]*ebqe_bc_v_ext.data()[ebNE_kb]+(1-isDOFBoundary_v.data()[ebNE_kb])*v_ext;
                bc_w_ext = isDOFBoundary_w.data()[ebNE_kb]*ebqe_bc_w_ext.data()[ebNE_kb]+(1-isDOFBoundary_w.data()[ebNE_kb])*w_ext;
                //VRANS
                porosity_ext = ebqe_porosity_ext.data()[ebNE_kb];
                //
                //calculate the internal and external trace of the pde coefficients
                //
                evaluateCoefficients(eps_rho,
                                     eps_mu,
                                     sigma,
                                     rho_0,
                                     nu_0,
                                     rho_1,
                                     nu_1,
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
                                     mom_v_ham_ext,
                                     dmom_v_ham_grad_p_ext,
                                     mom_w_ham_ext,
                                     dmom_w_ham_grad_p_ext,
                                     ebqe_dragBeam1.data()[ebNE_kb],
                                     ebqe_dragBeam2.data()[ebNE_kb],
                                     ebqe_dragBeam3.data()[ebNE_kb]);
                evaluateCoefficients(eps_rho,
                                     eps_mu,
                                     sigma,
                                     rho_0,
                                     nu_0,
                                     rho_1,
                                     nu_1,
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
                                     bc_mom_v_ham_ext,
                                     bc_dmom_v_ham_grad_p_ext,
                                     bc_mom_w_ham_ext,
                                     bc_dmom_w_ham_grad_p_ext,
                                     ebqe_dragBeam1.data()[ebNE_kb],
                                     ebqe_dragBeam2.data()[ebNE_kb],
                                     ebqe_dragBeam3.data()[ebNE_kb]);
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
                                            ebqe_vf_ext.data()[ebNE_kb],
                                            ebqe_phi_ext.data()[ebNE_kb],
                                            porosity_ext,
                                            c_mu, //mwf hack
                                            ebqe_turb_var_0.data()[ebNE_kb],
                                            ebqe_turb_var_1.data()[ebNE_kb],
                                            turb_var_grad_0_dummy, //not needed
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
                                            ebqe_vf_ext.data()[ebNE_kb],
                                            ebqe_phi_ext.data()[ebNE_kb],
                                            porosity_ext,
                                            c_mu, //mwf hack
                                            ebqe_turb_var_0.data()[ebNE_kb],
                                            ebqe_turb_var_1.data()[ebNE_kb],
                                            turb_var_grad_0_dummy, //not needed
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
                mass_adv_ext[0] -= MOVING_DOMAIN*xt_ext;
                mass_adv_ext[1] -= MOVING_DOMAIN*yt_ext;
                mass_adv_ext[2] -= MOVING_DOMAIN*zt_ext;

                mom_u_adv_ext[0] -= MOVING_DOMAIN*mom_u_acc_ext*xt_ext;
                mom_u_adv_ext[1] -= MOVING_DOMAIN*mom_u_acc_ext*yt_ext;
                mom_u_adv_ext[2] -= MOVING_DOMAIN*mom_u_acc_ext*zt_ext;
                dmom_u_adv_u_ext[0] -= MOVING_DOMAIN*dmom_u_acc_u_ext*xt_ext;
                dmom_u_adv_u_ext[1] -= MOVING_DOMAIN*dmom_u_acc_u_ext*yt_ext;
                dmom_u_adv_u_ext[2] -= MOVING_DOMAIN*dmom_u_acc_u_ext*zt_ext;

                mom_v_adv_ext[0] -= MOVING_DOMAIN*mom_v_acc_ext*xt_ext;
                mom_v_adv_ext[1] -= MOVING_DOMAIN*mom_v_acc_ext*yt_ext;
                mom_v_adv_ext[2] -= MOVING_DOMAIN*mom_v_acc_ext*zt_ext;
                dmom_v_adv_v_ext[0] -= MOVING_DOMAIN*dmom_v_acc_v_ext*xt_ext;
                dmom_v_adv_v_ext[1] -= MOVING_DOMAIN*dmom_v_acc_v_ext*yt_ext;
                dmom_v_adv_v_ext[2] -= MOVING_DOMAIN*dmom_v_acc_v_ext*zt_ext;

                mom_w_adv_ext[0] -= MOVING_DOMAIN*mom_w_acc_ext*xt_ext;
                mom_w_adv_ext[1] -= MOVING_DOMAIN*mom_w_acc_ext*yt_ext;
                mom_w_adv_ext[2] -= MOVING_DOMAIN*mom_w_acc_ext*zt_ext;
                dmom_w_adv_w_ext[0] -= MOVING_DOMAIN*dmom_w_acc_w_ext*xt_ext;
                dmom_w_adv_w_ext[1] -= MOVING_DOMAIN*dmom_w_acc_w_ext*yt_ext;
                dmom_w_adv_w_ext[2] -= MOVING_DOMAIN*dmom_w_acc_w_ext*zt_ext;

                //moving domain bc's
                bc_mom_u_adv_ext[0] -= MOVING_DOMAIN*bc_mom_u_acc_ext*xt_ext;
                bc_mom_u_adv_ext[1] -= MOVING_DOMAIN*bc_mom_u_acc_ext*yt_ext;
                bc_mom_u_adv_ext[2] -= MOVING_DOMAIN*bc_mom_u_acc_ext*zt_ext;

                bc_mom_v_adv_ext[0] -= MOVING_DOMAIN*bc_mom_v_acc_ext*xt_ext;
                bc_mom_v_adv_ext[1] -= MOVING_DOMAIN*bc_mom_v_acc_ext*yt_ext;
                bc_mom_v_adv_ext[2] -= MOVING_DOMAIN*bc_mom_v_acc_ext*zt_ext;

                bc_mom_w_adv_ext[0] -= MOVING_DOMAIN*bc_mom_w_acc_ext*xt_ext;
                bc_mom_w_adv_ext[1] -= MOVING_DOMAIN*bc_mom_w_acc_ext*yt_ext;
                bc_mom_w_adv_ext[2] -= MOVING_DOMAIN*bc_mom_w_acc_ext*zt_ext;
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
                                                          bc_p_ext,
                                                          bc_mass_adv_ext,
                                                          bc_mom_u_adv_ext,
                                                          bc_mom_v_adv_ext,
                                                          bc_mom_w_adv_ext,
                                                          ebqe_bc_flux_mass_ext.data()[ebNE_kb],
                                                          ebqe_bc_flux_mom_u_adv_ext.data()[ebNE_kb],
                                                          ebqe_bc_flux_mom_v_adv_ext.data()[ebNE_kb],
                                                          ebqe_bc_flux_mom_w_adv_ext.data()[ebNE_kb],
                                                          p_ext,
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
                                                          dflux_mom_w_adv_w_ext);
                //
                //calculate the flux jacobian
                //
                ck.calculateGScale(G,normal,h_penalty);
                penalty = useMetrics*C_b*h_penalty + (1.0-useMetrics)*ebqe_penalty_ext.data()[ebNE_kb];
                for (int j=0;j<nDOF_trial_element;j++)
                  {
                    int j_nSpace = j*nSpace,ebN_local_kb_j=ebN_local_kb*nDOF_trial_element+j;
                    fluxJacobian_p_p[j]=0.0;
                    fluxJacobian_p_u[j]=ck.ExteriorNumericalAdvectiveFluxJacobian(dflux_mass_u_ext,vel_trial_trace_ref.data()[ebN_local_kb_j]);
                    fluxJacobian_p_v[j]=ck.ExteriorNumericalAdvectiveFluxJacobian(dflux_mass_v_ext,vel_trial_trace_ref.data()[ebN_local_kb_j]);
                    fluxJacobian_p_w[j]=ck.ExteriorNumericalAdvectiveFluxJacobian(dflux_mass_w_ext,vel_trial_trace_ref.data()[ebN_local_kb_j]);

                    fluxJacobian_u_p[j]=ck.ExteriorNumericalAdvectiveFluxJacobian(dflux_mom_u_adv_p_ext,p_trial_trace_ref.data()[ebN_local_kb_j]);
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
                                                             isDiffusiveFluxBoundary_u.data()[ebNE_kb],
                                                             normal,
                                                             mom_uv_diff_ten_ext,
                                                             vel_trial_trace_ref.data()[ebN_local_kb_j],
                                                             &vel_grad_trial_trace[j_nSpace],
                                                             penalty);//ebqe_penalty_ext.data()[ebNE_kb]);
                    fluxJacobian_u_w[j]=ck.ExteriorNumericalAdvectiveFluxJacobian(dflux_mom_u_adv_w_ext,vel_trial_trace_ref.data()[ebN_local_kb_j]) +
                      ExteriorNumericalDiffusiveFluxJacobian(eps_rho,
                                                             ebqe_phi_ext.data()[ebNE_kb],
                                                             sdInfo_u_w_rowptr.data(),
                                                             sdInfo_u_w_colind.data(),
                                                             isDOFBoundary_w.data()[ebNE_kb],
                                                             isDiffusiveFluxBoundary_u.data()[ebNE_kb],
                                                             normal,
                                                             mom_uw_diff_ten_ext,
                                                             vel_trial_trace_ref.data()[ebN_local_kb_j],
                                                             &vel_grad_trial_trace[j_nSpace],
                                                             penalty);//ebqe_penalty_ext.data()[ebNE_kb]);

                    fluxJacobian_v_p[j]=ck.ExteriorNumericalAdvectiveFluxJacobian(dflux_mom_v_adv_p_ext,p_trial_trace_ref.data()[ebN_local_kb_j]);
                    fluxJacobian_v_u[j]=ck.ExteriorNumericalAdvectiveFluxJacobian(dflux_mom_v_adv_u_ext,vel_trial_trace_ref.data()[ebN_local_kb_j]) +
                      ExteriorNumericalDiffusiveFluxJacobian(eps_rho,
                                                             ebqe_phi_ext.data()[ebNE_kb],
                                                             sdInfo_v_u_rowptr.data(),
                                                             sdInfo_v_u_colind.data(),
                                                             isDOFBoundary_u.data()[ebNE_kb],
                                                             isDiffusiveFluxBoundary_v.data()[ebNE_kb],
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
                    fluxJacobian_v_w[j]=ck.ExteriorNumericalAdvectiveFluxJacobian(dflux_mom_v_adv_w_ext,vel_trial_trace_ref.data()[ebN_local_kb_j]) +
                      ExteriorNumericalDiffusiveFluxJacobian(eps_rho,
                                                             ebqe_phi_ext.data()[ebNE_kb],
                                                             sdInfo_v_w_rowptr.data(),
                                                             sdInfo_v_w_colind.data(),
                                                             isDOFBoundary_w.data()[ebNE_kb],
                                                             isDiffusiveFluxBoundary_v.data()[ebNE_kb],
                                                             normal,
                                                             mom_vw_diff_ten_ext,
                                                             vel_trial_trace_ref.data()[ebN_local_kb_j],
                                                             &vel_grad_trial_trace[j_nSpace],
                                                             penalty);//ebqe_penalty_ext.data()[ebNE_kb]);

                    fluxJacobian_w_p[j]=ck.ExteriorNumericalAdvectiveFluxJacobian(dflux_mom_w_adv_p_ext,p_trial_trace_ref.data()[ebN_local_kb_j]);
                    fluxJacobian_w_u[j]=ck.ExteriorNumericalAdvectiveFluxJacobian(dflux_mom_w_adv_u_ext,vel_trial_trace_ref.data()[ebN_local_kb_j]) +
                      ExteriorNumericalDiffusiveFluxJacobian(eps_rho,
                                                             ebqe_phi_ext.data()[ebNE_kb],
                                                             sdInfo_w_u_rowptr.data(),
                                                             sdInfo_w_u_colind.data(),
                                                             isDOFBoundary_u.data()[ebNE_kb],
                                                             isDiffusiveFluxBoundary_w.data()[ebNE_kb],
                                                             normal,
                                                             mom_wu_diff_ten_ext,
                                                             vel_trial_trace_ref.data()[ebN_local_kb_j],
                                                             &vel_grad_trial_trace[j_nSpace],
                                                             penalty);//ebqe_penalty_ext.data()[ebNE_kb]);
                    fluxJacobian_w_v[j]=ck.ExteriorNumericalAdvectiveFluxJacobian(dflux_mom_w_adv_v_ext,vel_trial_trace_ref.data()[ebN_local_kb_j]) +
                      ExteriorNumericalDiffusiveFluxJacobian(eps_rho,
                                                             ebqe_phi_ext.data()[ebNE_kb],
                                                             sdInfo_w_v_rowptr.data(),
                                                             sdInfo_w_v_colind.data(),
                                                             isDOFBoundary_v.data()[ebNE_kb],
                                                             isDiffusiveFluxBoundary_w.data()[ebNE_kb],
                                                             normal,
                                                             mom_wv_diff_ten_ext,
                                                             vel_trial_trace_ref.data()[ebN_local_kb_j],
                                                             &vel_grad_trial_trace[j_nSpace],
                                                             penalty);//ebqe_penalty_ext.data()[ebNE_kb]);
                    fluxJacobian_w_w[j]=ck.ExteriorNumericalAdvectiveFluxJacobian(dflux_mom_w_adv_w_ext,vel_trial_trace_ref.data()[ebN_local_kb_j]) +
                      ExteriorNumericalDiffusiveFluxJacobian(eps_rho,
                                                             ebqe_phi_ext.data()[ebNE_kb],
                                                             sdInfo_w_w_rowptr.data(),
                                                             sdInfo_w_w_colind.data(),
                                                             isDOFBoundary_w.data()[ebNE_kb],
                                                             isDiffusiveFluxBoundary_w.data()[ebNE_kb],
                                                             normal,
                                                             mom_ww_diff_ten_ext,
                                                             vel_trial_trace_ref.data()[ebN_local_kb_j],
                                                             &vel_grad_trial_trace[j_nSpace],
                                                             penalty);//ebqe_penalty_ext.data()[ebNE_kb]);
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

                        globalJacobian.data()[csrRowIndeces_p_p.data()[eN_i] + csrColumnOffsets_eb_p_p.data()[ebN_i_j]] += fluxJacobian_p_p[j]*p_test_dS[i];
                        globalJacobian.data()[csrRowIndeces_p_u.data()[eN_i] + csrColumnOffsets_eb_p_u.data()[ebN_i_j]] += fluxJacobian_p_u[j]*p_test_dS[i];
                        globalJacobian.data()[csrRowIndeces_p_v.data()[eN_i] + csrColumnOffsets_eb_p_v.data()[ebN_i_j]] += fluxJacobian_p_v[j]*p_test_dS[i];
                        globalJacobian.data()[csrRowIndeces_p_w.data()[eN_i] + csrColumnOffsets_eb_p_w.data()[ebN_i_j]] += fluxJacobian_p_w[j]*p_test_dS[i];

                        globalJacobian.data()[csrRowIndeces_u_p.data()[eN_i] + csrColumnOffsets_eb_u_p.data()[ebN_i_j]] += fluxJacobian_u_p[j]*vel_test_dS[i];
                        globalJacobian.data()[csrRowIndeces_u_u.data()[eN_i] + csrColumnOffsets_eb_u_u.data()[ebN_i_j]] += fluxJacobian_u_u[j]*vel_test_dS[i]+
                          ck.ExteriorElementBoundaryDiffusionAdjointJacobian(isDOFBoundary_u.data()[ebNE_kb],
                                                                             isDiffusiveFluxBoundary_u.data()[ebNE_kb],
                                                                             eb_adjoint_sigma,
                                                                             vel_trial_trace_ref.data()[ebN_local_kb_j],
                                                                             normal,
                                                                             sdInfo_u_u_rowptr.data(),
                                                                             sdInfo_u_u_colind.data(),
                                                                             mom_uu_diff_ten_ext,
                                                                             &vel_grad_test_dS[i*nSpace]);
                        globalJacobian.data()[csrRowIndeces_u_v.data()[eN_i] + csrColumnOffsets_eb_u_v.data()[ebN_i_j]] += fluxJacobian_u_v[j]*vel_test_dS[i]+
                          ck.ExteriorElementBoundaryDiffusionAdjointJacobian(isDOFBoundary_v.data()[ebNE_kb],
                                                                             isDiffusiveFluxBoundary_u.data()[ebNE_kb],
                                                                             eb_adjoint_sigma,
                                                                             vel_trial_trace_ref.data()[ebN_local_kb_j],
                                                                             normal,
                                                                             sdInfo_u_v_rowptr.data(),
                                                                             sdInfo_u_v_colind.data(),
                                                                             mom_uv_diff_ten_ext,
                                                                             &vel_grad_test_dS[i*nSpace]);
                        globalJacobian.data()[csrRowIndeces_u_w.data()[eN_i] + csrColumnOffsets_eb_u_w.data()[ebN_i_j]] += fluxJacobian_u_w[j]*vel_test_dS[i]+
                          ck.ExteriorElementBoundaryDiffusionAdjointJacobian(isDOFBoundary_w.data()[ebNE_kb],
                                                                             isDiffusiveFluxBoundary_u.data()[ebNE_kb],
                                                                             eb_adjoint_sigma,
                                                                             vel_trial_trace_ref.data()[ebN_local_kb_j],
                                                                             normal,
                                                                             sdInfo_u_w_rowptr.data(),
                                                                             sdInfo_u_w_colind.data(),
                                                                             mom_uw_diff_ten_ext,
                                                                             &vel_grad_test_dS[i*nSpace]);

                        globalJacobian.data()[csrRowIndeces_v_p.data()[eN_i] + csrColumnOffsets_eb_v_p.data()[ebN_i_j]] += fluxJacobian_v_p[j]*vel_test_dS[i];
                        globalJacobian.data()[csrRowIndeces_v_u.data()[eN_i] + csrColumnOffsets_eb_v_u.data()[ebN_i_j]] += fluxJacobian_v_u[j]*vel_test_dS[i]+
                          ck.ExteriorElementBoundaryDiffusionAdjointJacobian(isDOFBoundary_u.data()[ebNE_kb],
                                                                             isDiffusiveFluxBoundary_v.data()[ebNE_kb],
                                                                             eb_adjoint_sigma,
                                                                             vel_trial_trace_ref.data()[ebN_local_kb_j],
                                                                             normal,
                                                                             sdInfo_v_u_rowptr.data(),
                                                                             sdInfo_v_u_colind.data(),
                                                                             mom_vu_diff_ten_ext,
                                                                             &vel_grad_test_dS[i*nSpace]);
                        globalJacobian.data()[csrRowIndeces_v_v.data()[eN_i] + csrColumnOffsets_eb_v_v.data()[ebN_i_j]] += fluxJacobian_v_v[j]*vel_test_dS[i]+
                          ck.ExteriorElementBoundaryDiffusionAdjointJacobian(isDOFBoundary_v.data()[ebNE_kb],
                                                                             isDiffusiveFluxBoundary_v.data()[ebNE_kb],
                                                                             eb_adjoint_sigma,
                                                                             vel_trial_trace_ref.data()[ebN_local_kb_j],
                                                                             normal,
                                                                             sdInfo_v_v_rowptr.data(),
                                                                             sdInfo_v_v_colind.data(),
                                                                             mom_vv_diff_ten_ext,
                                                                             &vel_grad_test_dS[i*nSpace]);
                        globalJacobian.data()[csrRowIndeces_v_w.data()[eN_i] + csrColumnOffsets_eb_v_w.data()[ebN_i_j]] += fluxJacobian_v_w[j]*vel_test_dS[i]+
                          ck.ExteriorElementBoundaryDiffusionAdjointJacobian(isDOFBoundary_w.data()[ebNE_kb],
                                                                             isDiffusiveFluxBoundary_v.data()[ebNE_kb],
                                                                             eb_adjoint_sigma,
                                                                             vel_trial_trace_ref.data()[ebN_local_kb_j],
                                                                             normal,
                                                                             sdInfo_v_w_rowptr.data(),
                                                                             sdInfo_v_w_colind.data(),
                                                                             mom_vw_diff_ten_ext,
                                                                             &vel_grad_test_dS[i*nSpace]);

                        globalJacobian.data()[csrRowIndeces_w_p.data()[eN_i] + csrColumnOffsets_eb_w_p.data()[ebN_i_j]] += fluxJacobian_w_p[j]*vel_test_dS[i];
                        globalJacobian.data()[csrRowIndeces_w_u.data()[eN_i] + csrColumnOffsets_eb_w_u.data()[ebN_i_j]] += fluxJacobian_w_u[j]*vel_test_dS[i]+
                          ck.ExteriorElementBoundaryDiffusionAdjointJacobian(isDOFBoundary_u.data()[ebNE_kb],
                                                                             isDiffusiveFluxBoundary_w.data()[ebNE_kb],
                                                                             eb_adjoint_sigma,
                                                                             vel_trial_trace_ref.data()[ebN_local_kb_j],
                                                                             normal,
                                                                             sdInfo_w_u_rowptr.data(),
                                                                             sdInfo_w_u_colind.data(),
                                                                             mom_wu_diff_ten_ext,
                                                                             &vel_grad_test_dS[i*nSpace]);
                        globalJacobian.data()[csrRowIndeces_w_v.data()[eN_i] + csrColumnOffsets_eb_w_v.data()[ebN_i_j]] += fluxJacobian_w_v[j]*vel_test_dS[i]+
                          ck.ExteriorElementBoundaryDiffusionAdjointJacobian(isDOFBoundary_v.data()[ebNE_kb],
                                                                             isDiffusiveFluxBoundary_w.data()[ebNE_kb],
                                                                             eb_adjoint_sigma,
                                                                             vel_trial_trace_ref.data()[ebN_local_kb_j],
                                                                             normal,
                                                                             sdInfo_w_v_rowptr.data(),
                                                                             sdInfo_w_v_colind.data(),
                                                                             mom_wv_diff_ten_ext,
                                                                             &vel_grad_test_dS[i*nSpace]);
                        globalJacobian.data()[csrRowIndeces_w_w.data()[eN_i] + csrColumnOffsets_eb_w_w.data()[ebN_i_j]] += fluxJacobian_w_w[j]*vel_test_dS[i]+
                          ck.ExteriorElementBoundaryDiffusionAdjointJacobian(isDOFBoundary_w.data()[ebNE_kb],
                                                                             isDiffusiveFluxBoundary_w.data()[ebNE_kb],
                                                                             eb_adjoint_sigma,
                                                                             vel_trial_trace_ref.data()[ebN_local_kb_j],
                                                                             normal,
                                                                             sdInfo_w_w_rowptr.data(),
                                                                             sdInfo_w_w_colind.data(),
                                                                             mom_ww_diff_ten_ext,
                                                                             &vel_grad_test_dS[i*nSpace]);
                      }//j
                  }//i
              }//kb
          }//ebNE
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
        proteus::pyarray<int>& mesh_l2g = args.array<int>("mesh_l2g");
        proteus::pyarray<double>& mesh_trial_trace_ref = args.array<double>("mesh_trial_trace_ref");
        proteus::pyarray<double>& mesh_grad_trial_trace_ref = args.array<double>("mesh_grad_trial_trace_ref");
        proteus::pyarray<double>& normal_ref = args.array<double>("normal_ref");
        proteus::pyarray<double>& boundaryJac_ref = args.array<double>("boundaryJac_ref");
        proteus::pyarray<int>& vel_l2g = args.array<int>("vel_l2g");
        proteus::pyarray<double>& u_dof = args.array<double>("u_dof");
        proteus::pyarray<double>& v_dof = args.array<double>("v_dof");
        proteus::pyarray<double>& w_dof = args.array<double>("w_dof");
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
            int ebN = exteriorElementBoundariesArray.data()[ebNE];
            for  (int kb=0;kb<nQuadraturePoints_elementBoundary;kb++)
              {
                int ebN_kb_nSpace = ebN*nQuadraturePoints_elementBoundary*nSpace+kb*nSpace,
                  ebNE_kb_nSpace = ebNE*nQuadraturePoints_elementBoundary*nSpace+kb*nSpace;
                velocityAverage.data()[ebN_kb_nSpace+0]=ebqe_velocity.data()[ebNE_kb_nSpace+0];
                velocityAverage.data()[ebN_kb_nSpace+1]=ebqe_velocity.data()[ebNE_kb_nSpace+1];
                velocityAverage.data()[ebN_kb_nSpace+2]=ebqe_velocity.data()[ebNE_kb_nSpace+2];
              }//ebNE
          }
        for (int ebNI = 0; ebNI < nInteriorElementBoundaries_global; ebNI++)
          {
            int ebN = interiorElementBoundariesArray.data()[ebNI],
              left_eN_global   = elementBoundaryElementsArray.data()[ebN*2+0],
              left_ebN_element  = elementBoundaryLocalElementBoundariesArray.data()[ebN*2+0],
              right_eN_global  = elementBoundaryElementsArray.data()[ebN*2+1],
              right_ebN_element = elementBoundaryLocalElementBoundariesArray.data()[ebN*2+1],
              left_eN_nDOF_trial_element = left_eN_global*nDOF_trial_element,
              right_eN_nDOF_trial_element = right_eN_global*nDOF_trial_element;
            double jac[nSpace*nSpace],
              jacDet,
              jacInv[nSpace*nSpace],
              boundaryJac[nSpace*(nSpace-1)],
              metricTensor[(nSpace-1)*(nSpace-1)],
              metricTensorDetSqrt,
              normal[3],
              x,y,z;

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
                  w_right=0.0;
                int left_kb = kb,
                  right_kb = permutations[kb],
                  left_ebN_element_kb_nDOF_test_element=(left_ebN_element*nQuadraturePoints_elementBoundary+left_kb)*nDOF_test_element,
                  right_ebN_element_kb_nDOF_test_element=(right_ebN_element*nQuadraturePoints_elementBoundary+right_kb)*nDOF_test_element;
                //
                //calculate the velocity solution at quadrature points on left and right
                //
                ck.valFromDOF(u_dof.data(),&vel_l2g.data()[left_eN_nDOF_trial_element],&vel_trial_trace_ref.data()[left_ebN_element_kb_nDOF_test_element],u_left);
                ck.valFromDOF(v_dof.data(),&vel_l2g.data()[left_eN_nDOF_trial_element],&vel_trial_trace_ref.data()[left_ebN_element_kb_nDOF_test_element],v_left);
                ck.valFromDOF(w_dof.data(),&vel_l2g.data()[left_eN_nDOF_trial_element],&vel_trial_trace_ref.data()[left_ebN_element_kb_nDOF_test_element],w_left);
                //
                ck.valFromDOF(u_dof.data(),&vel_l2g.data()[right_eN_nDOF_trial_element],&vel_trial_trace_ref.data()[right_ebN_element_kb_nDOF_test_element],u_right);
                ck.valFromDOF(v_dof.data(),&vel_l2g.data()[right_eN_nDOF_trial_element],&vel_trial_trace_ref.data()[right_ebN_element_kb_nDOF_test_element],v_right);
                ck.valFromDOF(w_dof.data(),&vel_l2g.data()[right_eN_nDOF_trial_element],&vel_trial_trace_ref.data()[right_ebN_element_kb_nDOF_test_element],w_right);
                //
                velocityAverage.data()[ebN_kb_nSpace+0]=0.5*(u_left + u_right);
                velocityAverage.data()[ebN_kb_nSpace+1]=0.5*(v_left + v_right);
                velocityAverage.data()[ebN_kb_nSpace+2]=0.5*(w_left + w_right);
              }//ebNI
          }
      }

      void calculateForce   (arguments_dict& args)
      {
        proteus::pyarray<double>& mesh_trial_ref = args.array<double>("mesh_trial_ref");
        proteus::pyarray<double>& mesh_grad_trial_ref = args.array<double>("mesh_grad_trial_ref");
        proteus::pyarray<double>& mesh_dof = args.array<double>("mesh_dof");
        proteus::pyarray<int>&    mesh_l2g = args.array<int>("mesh_l2g");
        proteus::pyarray<double>& dV_ref = args.array<double>("dV_ref");
        proteus::pyarray<double>& p_trial_ref = args.array<double>("p_trial_ref");
        proteus::pyarray<double>& p_grad_trial_ref = args.array<double>("p_grad_trial_ref");
        proteus::pyarray<double>& p_test_ref = args.array<double>("p_test_ref");
        proteus::pyarray<double>& p_grad_test_ref = args.array<double>("p_grad_test_ref");
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
        proteus::pyarray<double>& elementDiameter = args.array<double>("elementDiameter");
        proteus::pyarray<double>& nodeDiametersArray = args.array<double>("nodeDiametersArray");
        double hFactor = args.scalar<double>("hFactor");
        int nElements_global = args.scalar<int>("nElements_global");
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
        proteus::pyarray<int>& p_l2g = args.array<int>("p_l2g");
        proteus::pyarray<int>& vel_l2g = args.array<int>("vel_l2g");
        proteus::pyarray<double>& p_dof = args.array<double>("p_dof");
        proteus::pyarray<double>& u_dof = args.array<double>("u_dof");
        proteus::pyarray<double>& v_dof = args.array<double>("v_dof");
        proteus::pyarray<double>& w_dof = args.array<double>("w_dof");
        proteus::pyarray<double>& g = args.array<double>("g");
        proteus::pyarray<double>& rho_init = args.array<double>("rho_init");
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
        proteus::pyarray<double>& q_velocity_sge = args.array<double>("q_velocity_sge");
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
        proteus::pyarray<double>& cg = args.array<double>("cg");
        proteus::pyarray<double>& force = args.array<double>("force");
        proteus::pyarray<double>& moment = args.array<double>("moment");
        int nExteriorElementBoundaries_global = args.scalar<int>("nExteriorElementBoundaries_global");
        proteus::pyarray<int>& exteriorElementBoundariesArray = args.array<int>("exteriorElementBoundariesArray");
        proteus::pyarray<int>& elementBoundaryElementsArray = args.array<int>("elementBoundaryElementsArray");
        proteus::pyarray<int>& elementBoundaryLocalElementBoundariesArray = args.array<int>("elementBoundaryLocalElementBoundariesArray");
        proteus::pyarray<int>& forceExtractionFaces = args.array<int>("forceExtractionFaces");
        int nForceExtractionFaces = args.scalar<int>("nForceExtractionFaces");
        proteus::pyarray<double>& ebqe_vf_ext = args.array<double>("ebqe_vf_ext");
        proteus::pyarray<double>& ebqe_phi_ext = args.array<double>("ebqe_phi_ext");
        proteus::pyarray<double>& ebqe_normal_phi_ext = args.array<double>("ebqe_normal_phi_ext");
        proteus::pyarray<double>& ebqe_kappa_phi_ext = args.array<double>("ebqe_kappa_phi_ext");
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
        proteus::pyarray<double>& q_velocity = args.array<double>("q_velocity");
        proteus::pyarray<double>& ebqe_velocity = args.array<double>("ebqe_velocity");
        proteus::pyarray<double>& flux = args.array<double>("flux");
        proteus::pyarray<double>& elementResidual_p_save = args.array<double>("elementResidual_p_save");
        //
        //loop over exterior element boundaries to calculate surface integrals and load into element and global residuals
        //
        //ebNE is the Exterior element boundary INdex
        //ebN is the element boundary INdex
        //eN is the element index

        double tmp1[nSpace], tmp2[nSpace*nSpace];

        for (int fef = 0; fef < nForceExtractionFaces; fef++)
          {
            int ebNE = forceExtractionFaces.data()[fef];
            //std::cout<<fef<<"  "<<ebNE<<std::endl;
            {
              int ebN = exteriorElementBoundariesArray.data()[ebNE],
                eN  = elementBoundaryElementsArray.data()[ebN*2+0],
                ebN_local = elementBoundaryLocalElementBoundariesArray.data()[ebN*2+0],
                eN_nDOF_trial_element = eN*nDOF_trial_element;
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
                    jac_ext[nSpace*nSpace],
                    jacDet_ext,
                    jacInv_ext[nSpace*nSpace],
                    boundaryJac[nSpace*(nSpace-1)],
                    metricTensor[(nSpace-1)*(nSpace-1)],
                    metricTensorDetSqrt,
                    dS,p_test_dS[nDOF_test_element],vel_test_dS[nDOF_test_element],vel_grad_test_dS[nDOF_test_element*nSpace],
                    p_grad_trial_trace[nDOF_trial_element*nSpace],vel_grad_trial_trace[nDOF_trial_element*nSpace],
                    normal[3],x_ext,y_ext,z_ext,
                    G[nSpace*nSpace],G_dd_G,tr_G,h_phi,h_penalty,penalty,gi[nSpace], unormal,gnormal,uneg,
                    H_rho,d_rho, H_mu,d_mu, rho, mu,nu;
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
                  dS = metricTensorDetSqrt*dS_ref.data()[kb];
                  //get the metric tensor
                  //cek todo use symmetry
                  ck.calculateG(jacInv_ext,G,G_dd_G,tr_G);
                  ck.calculateGScale(G,&ebqe_normal_phi_ext.data()[ebNE_kb_nSpace],h_phi);

                  eps_rho = epsFact_rho*(useMetrics*h_phi+(1.0-useMetrics)*elementDiameter.data()[eN]);
                  eps_mu  = epsFact_mu *(useMetrics*h_phi+(1.0-useMetrics)*elementDiameter.data()[eN]);

                  H_rho = (1.0-useVF)*RANS2P_IB::smoothedHeaviside(eps_rho,ebqe_phi_ext.data()[ebNE_kb]) + useVF*fmin(1.0,fmax(0.0,ebqe_vf_ext.data()[ebNE_kb]));
                  d_rho = (1.0-useVF)*RANS2P_IB::smoothedDirac(eps_rho,ebqe_phi_ext.data()[ebNE_kb]);
                  H_mu  = (1.0-useVF)*RANS2P_IB::smoothedHeaviside(eps_mu,ebqe_phi_ext.data()[ebNE_kb]) + useVF*fmin(1.0,fmax(0.0,ebqe_vf_ext.data()[ebNE_kb]));
                  d_mu  = (1.0-useVF)*RANS2P_IB::smoothedDirac(eps_mu,ebqe_phi_ext.data()[ebNE_kb]);


                  //compute shape and solution information
                  //shape
                  ck.gradTrialFromRef(&  p_grad_trial_trace_ref.data()[ebN_local_kb_nSpace*nDOF_trial_element],jacInv_ext,p_grad_trial_trace);
                  ck.gradTrialFromRef(&vel_grad_trial_trace_ref.data()[ebN_local_kb_nSpace*nDOF_trial_element],jacInv_ext,vel_grad_trial_trace);
                  //solution and gradients
                  ck.valFromDOF(p_dof.data(),&p_l2g.data()[eN_nDOF_trial_element],&p_trial_trace_ref.data()[ebN_local_kb*nDOF_test_element],p_ext);
                  ck.valFromDOF(u_dof.data(),&vel_l2g.data()[eN_nDOF_trial_element],&vel_trial_trace_ref.data()[ebN_local_kb*nDOF_test_element],u_ext);
                  ck.valFromDOF(v_dof.data(),&vel_l2g.data()[eN_nDOF_trial_element],&vel_trial_trace_ref.data()[ebN_local_kb*nDOF_test_element],v_ext);
                  ck.valFromDOF(w_dof.data(),&vel_l2g.data()[eN_nDOF_trial_element],&vel_trial_trace_ref.data()[ebN_local_kb*nDOF_test_element],w_ext);
                  ck.gradFromDOF(p_dof.data(),&p_l2g.data()[eN_nDOF_trial_element],p_grad_trial_trace,grad_p_ext);
                  ck.gradFromDOF(u_dof.data(),&vel_l2g.data()[eN_nDOF_trial_element],vel_grad_trial_trace,grad_u_ext);
                  ck.gradFromDOF(v_dof.data(),&vel_l2g.data()[eN_nDOF_trial_element],vel_grad_trial_trace,grad_v_ext);
                  ck.gradFromDOF(w_dof.data(),&vel_l2g.data()[eN_nDOF_trial_element],vel_grad_trial_trace,grad_w_ext);
                  //precalculate test function products with integration weights
                  for (int j=0;j<nDOF_trial_element;j++)
                    {
                      int j_nSpace = j*nSpace;
                      p_test_dS[j] = p_test_trace_ref.data()[ebN_local_kb*nDOF_test_element+j]*dS;
                      vel_test_dS[j] = vel_test_trace_ref.data()[ebN_local_kb*nDOF_test_element+j]*dS;

                      vel_grad_test_dS[j_nSpace+0] = vel_grad_trial_trace[j_nSpace+0]*dS;
                      vel_grad_test_dS[j_nSpace+1] = vel_grad_trial_trace[j_nSpace+1]*dS;
                      vel_grad_test_dS[j_nSpace+2] = vel_grad_trial_trace[j_nSpace+2]*dS;
                    }

                  ebqe_velocity.data()[ebNE_kb_nSpace + 0 ] = u_ext;
                  ebqe_velocity.data()[ebNE_kb_nSpace + 1 ] = v_ext;
                  ebqe_velocity.data()[ebNE_kb_nSpace + 2 ] = w_ext;

                  //calculate eddy viscosity
                  double norm_S,h_e=elementDiameter.data()[eN],t_nu_0,t_nu_1;
                  t_nu_0=nu_0;
                  t_nu_1=nu_1;
                  switch (turbulenceClosureModel)
                    {
                    case 1:
                      {
                        norm_S = sqrt(2.0*(grad_u_ext[0]*grad_u_ext[0] + grad_v_ext[1]*grad_v_ext[1] + grad_w_ext[2]*grad_w_ext[2] +
                                           0.5*(grad_u_ext[1]+grad_v_ext[0])*(grad_u_ext[1]+grad_v_ext[0]) +
                                           0.5*(grad_u_ext[2]+grad_w_ext[0])*(grad_u_ext[2]+grad_w_ext[0]) +
                                           0.5*(grad_v_ext[2]+grad_w_ext[1])*(grad_v_ext[2]+grad_w_ext[1])));
                        t_nu_0 += smagorinskyConstant*smagorinskyConstant*h_e*h_e*norm_S;
                        t_nu_1 += smagorinskyConstant*smagorinskyConstant*h_e*h_e*norm_S;
                      }
                    case 2:
                      {
                        double re_0,cs_0,re_1,cs_1;
                        norm_S = sqrt(2.0*(grad_u_ext[0]*grad_u_ext[0] + grad_v_ext[1]*grad_v_ext[1] + grad_w_ext[2]*grad_w_ext[2] +
                                           0.5*(grad_u_ext[1]+grad_v_ext[0])*(grad_u_ext[1]+grad_v_ext[0]) +
                                           0.5*(grad_u_ext[2]+grad_w_ext[0])*(grad_u_ext[2]+grad_w_ext[0]) +
                                           0.5*(grad_v_ext[2]+grad_w_ext[1])*(grad_v_ext[2]+grad_w_ext[1])));
                        re_0 = h_e*h_e*norm_S/nu_0;
                        cs_0=0.027*pow(10.0,-3.23*pow(re_0,-0.92));
                        t_nu_0 += cs_0*h_e*h_e*norm_S;
                        re_1 = h_e*h_e*norm_S/nu_1;
                        cs_1=0.027*pow(10.0,-3.23*pow(re_1,-0.92));
                        t_nu_1 += cs_1*h_e*h_e*norm_S;
                      }
                    }

                  rho = rho_0*(1.0-H_rho)+rho_1*H_rho;
                  nu  = t_nu_0*(1.0-H_mu)+t_nu_1*H_mu;
                  mu  = rho_0*t_nu_0*(1.0-H_mu)+rho_1*t_nu_1*H_mu;
                  //
                  // Assume either nono or all velocity have dir BC
                  //
                  //  if (isDOFBoundary_w[ebNE_kb] == 1)
                  // {

                  //
                  //calculate the pde coefficients using the solution and the boundary values for the solution
                  //
                  gi[0] = ebqe_bc_u_ext.data()[ebNE_kb];
                  gi[1] = ebqe_bc_v_ext.data()[ebNE_kb];
                  gi[2] = ebqe_bc_w_ext.data()[ebNE_kb];

                  unormal = normal[0]*u_ext + normal[1]*v_ext + normal[2]*w_ext;
                  gnormal = normal[0]*gi[0] + normal[1]*gi[1] + normal[2]*gi[2];
                  uneg = 0.5*(unormal - fabs(unormal));

                  //
                  //calculate the  penalty
                  //
                  ck.calculateGScale(G,normal,h_penalty);
                  penalty = C_b*mu/h_penalty;

                  //
                  //calculate temporary values
                  //
                  tmp1[0] = -mu*(grad_u_ext[0]*normal[0] + grad_u_ext[0]*normal[0]+
                                 grad_u_ext[1]*normal[1] + grad_v_ext[0]*normal[1]+
                                 grad_u_ext[2]*normal[2] + grad_w_ext[0]*normal[2])
                    + penalty*(u_ext-gi[0]) + p_ext*normal[0] - uneg*rho*(u_ext-gi[0]);

                  tmp1[1] = -mu*(grad_v_ext[0]*normal[0] + grad_u_ext[1]*normal[0]+
                                 grad_v_ext[1]*normal[1] + grad_v_ext[1]*normal[1]+
                                 grad_v_ext[2]*normal[2] + grad_w_ext[1]*normal[2])
                    + penalty*(v_ext-gi[1]) + p_ext*normal[1] - uneg*rho*(v_ext-gi[1]);

                  tmp1[2] = -mu*(grad_w_ext[0]*normal[0] + grad_u_ext[2]*normal[0]+
                                 grad_w_ext[1]*normal[1] + grad_v_ext[2]*normal[1]+
                                 grad_w_ext[2]*normal[2] + grad_w_ext[2]*normal[2])
                    + penalty*(w_ext-gi[2]) + p_ext*normal[2] - uneg*rho*(w_ext-gi[2]);

                  tmp2[0*nSpace + 0] =  0.0;
                  tmp2[0*nSpace + 1] =  mu*(  (v_ext - gi[1])*normal[0] + (u_ext - gi[0])*normal[1] );
                  tmp2[0*nSpace + 2] =  mu*(  (w_ext - gi[2])*normal[0] + (u_ext - gi[0])*normal[2] );

                  tmp2[1*nSpace + 0] =  mu*(  (u_ext - gi[0])*normal[1] + (v_ext - gi[1])*normal[0] );
                  tmp2[1*nSpace + 1] =  0.0;
                  tmp2[1*nSpace + 2] =  mu*(  (w_ext - gi[2])*normal[1] + (v_ext - gi[1])*normal[2] );

                  tmp2[2*nSpace + 0] =  mu*(  (u_ext - gi[0])*normal[2] + (w_ext - gi[2])*normal[0] );
                  tmp2[2*nSpace + 1] =  mu*(  (v_ext - gi[1])*normal[2] + (w_ext - gi[2])*normal[1] );
                  tmp2[2*nSpace + 2] =  0.0;

                  //
                  //calculate the force
                  //
                  force.data()[0] += tmp1[0]*dS;
                  force.data()[1] += tmp1[1]*dS;
                  force.data()[2] += tmp1[2]*dS;

                  //
                  //calculate the moment
                  //
                  moment.data()[0] += ( (y_ext-cg.data()[1])*tmp1[2] - (z_ext-cg.data()[2])*tmp1[1] + tmp2[2*nSpace + 1] - tmp2[1*nSpace + 2] )*dS;
                  moment.data()[1] += ( (z_ext-cg.data()[2])*tmp1[0] - (x_ext-cg.data()[0])*tmp1[2] + tmp2[0*nSpace + 2] - tmp2[2*nSpace + 0] )*dS;
                  moment.data()[2] += ( (x_ext-cg.data()[0])*tmp1[1] - (y_ext-cg.data()[1])*tmp1[0] + tmp2[1*nSpace + 0] - tmp2[0*nSpace + 1] )*dS;

                  //  }
                }//kb
            }//ebNE
          }

      }
      void calculateBeams(arguments_dict& args)
      {
        proteus::pyarray<double>& mesh_trial_ref = args.array<double>("mesh_trial_ref");
        proteus::pyarray<double>& mesh_grad_trial_ref = args.array<double>("mesh_grad_trial_ref");
        proteus::pyarray<double>& mesh_dof = args.array<double>("mesh_dof");
        proteus::pyarray<double>& mesh_velocity_dof = args.array<double>("mesh_velocity_dof");
        double MOVING_DOMAIN = args.scalar<double>("MOVING_DOMAIN");
        proteus::pyarray<int>& mesh_l2g = args.array<int>("mesh_l2g");
        proteus::pyarray<double>& dV_ref = args.array<double>("dV_ref");
        proteus::pyarray<double>& p_trial_ref = args.array<double>("p_trial_ref");
        proteus::pyarray<double>& p_grad_trial_ref = args.array<double>("p_grad_trial_ref");
        proteus::pyarray<double>& p_test_ref = args.array<double>("p_test_ref");
        proteus::pyarray<double>& p_grad_test_ref = args.array<double>("p_grad_test_ref");
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
        int nElements_global = args.scalar<int>("nElements_global");
        // the beam loads (q1-q3), vel_avg and netBeamDrag are summed over ranks by the caller, so only owned
        // elements add to them; every local element still gets q_dragBeam*, the fluid's source
        int nElements_owned = args.scalar<int>("nElements_owned");
        int nElementBoundaries_owned = args.scalar<int>("nElementBoundaries_owned");
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
        const proteus::pyarray<double>& phi_solid = args.array<double>("phi_solid");
        const proteus::pyarray<double>& q_velocity_solid = args.array<double>("q_velocity_solid");
        const proteus::pyarray<double>& q_porosity = args.array<double>("q_porosity");
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
        proteus::pyarray<double>& normal_phi = args.array<double>("normal_phi");
        proteus::pyarray<double>& kappa_phi = args.array<double>("kappa_phi");
        proteus::pyarray<double>& q_mom_u_acc = args.array<double>("q_mom_u_acc");
        proteus::pyarray<double>& q_mom_v_acc = args.array<double>("q_mom_v_acc");
        proteus::pyarray<double>& q_mom_w_acc = args.array<double>("q_mom_w_acc");
        proteus::pyarray<double>& q_mass_adv = args.array<double>("q_mass_adv");
        proteus::pyarray<double>& q_mom_u_acc_beta_bdf = args.array<double>("q_mom_u_acc_beta_bdf");
        proteus::pyarray<double>& q_mom_v_acc_beta_bdf = args.array<double>("q_mom_v_acc_beta_bdf");
        proteus::pyarray<double>& q_mom_w_acc_beta_bdf = args.array<double>("q_mom_w_acc_beta_bdf");
        proteus::pyarray<double>& q_velocity_sge = args.array<double>("q_velocity_sge");
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
        proteus::pyarray<int>& elementBoundaryElementsArray = args.array<int>("elementBoundaryElementsArray");
        proteus::pyarray<int>& elementBoundaryLocalElementBoundariesArray = args.array<int>("elementBoundaryLocalElementBoundariesArray");
        proteus::pyarray<double>& ebqe_vf_ext = args.array<double>("ebqe_vf_ext");
        proteus::pyarray<double>& bc_ebqe_vf_ext = args.array<double>("bc_ebqe_vf_ext");
        proteus::pyarray<double>& ebqe_phi_ext = args.array<double>("ebqe_phi_ext");
        proteus::pyarray<double>& bc_ebqe_phi_ext = args.array<double>("bc_ebqe_phi_ext");
        proteus::pyarray<double>& ebqe_normal_phi_ext = args.array<double>("ebqe_normal_phi_ext");
        proteus::pyarray<double>& ebqe_kappa_phi_ext = args.array<double>("ebqe_kappa_phi_ext");
        const proteus::pyarray<double>& ebqe_porosity_ext = args.array<double>("ebqe_porosity_ext");
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
        proteus::pyarray<int>& boundaryFlags = args.array<int>("boundaryFlags");
        proteus::pyarray<double>& barycenters = args.array<double>("barycenters");
        proteus::pyarray<double>& wettedAreas = args.array<double>("wettedAreas");
        proteus::pyarray<double>& netForces_p = args.array<double>("netForces_p");
        proteus::pyarray<double>& netForces_v = args.array<double>("netForces_v");
        proteus::pyarray<double>& netMoments = args.array<double>("netMoments");
        proteus::pyarray<double>& q_dragBeam1 = args.array<double>("q_dragBeam1");
        proteus::pyarray<double>& q_dragBeam2 = args.array<double>("q_dragBeam2");
        proteus::pyarray<double>& q_dragBeam3 = args.array<double>("q_dragBeam3");
        proteus::pyarray<double>& ebqe_dragBeam1 = args.array<double>("ebqe_dragBeam1");
        proteus::pyarray<double>& ebqe_dragBeam2 = args.array<double>("ebqe_dragBeam2");
        proteus::pyarray<double>& ebqe_dragBeam3 = args.array<double>("ebqe_dragBeam3");
        int nBeams = args.scalar<int>("nBeams");
        int nBeamElements = args.scalar<int>("nBeamElements");
        int beam_quadOrder = args.scalar<int>("beam_quadOrder");
        double beam_Cd = args.scalar<double>("beam_Cd");
        proteus::pyarray<double>& beamRadius = args.array<double>("beamRadius");
        proteus::pyarray<double>& xq = args.array<double>("xq");
        proteus::pyarray<double>& yq = args.array<double>("yq");
        proteus::pyarray<double>& zq = args.array<double>("zq");
        proteus::pyarray<double>& Beam_h = args.array<double>("Beam_h");
        proteus::pyarray<double>& dV_beam = args.array<double>("dV_beam");
        proteus::pyarray<double>& q1 = args.array<double>("q1");
        proteus::pyarray<double>& q2 = args.array<double>("q2");
        proteus::pyarray<double>& q3 = args.array<double>("q3");
        proteus::pyarray<double>& vel_avg = args.array<double>("vel_avg");
        proteus::pyarray<double>& netBeamDrag = args.array<double>("netBeamDrag");
        //
        //loop over elements to compute volume integrals and load them into element and global residual
        //
        double globalConservationError=0.0;
        for(int eN=0;eN<nElements_global;eN++)
          {
            const bool owned = eN < nElements_owned;
            double ghostBeamDrag[1] = {0.0};
            double eps_rho,eps_mu;
            //
            //loop over quadrature points and compute integrands
            //
            for(int k=0;k<nQuadraturePoints_element;k++)
              {
                //compute indices and declare local storage
                int eN_k = eN*nQuadraturePoints_element+k,
                  eN_k_nSpace = eN_k*nSpace,
                  eN_nDOF_trial_element = eN*nDOF_trial_element;
                double p=0.0,u=0.0,v=0.0,w=0.0,
                  grad_p[nSpace],grad_u[nSpace],grad_v[nSpace],grad_w[nSpace],
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
                  mom_v_ham=0.0,
                  dmom_v_ham_grad_p[nSpace],
                  mom_w_ham=0.0,
                  dmom_w_ham_grad_p[nSpace],
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
                  p_test_dV[nDOF_trial_element],vel_test_dV[nDOF_trial_element],
                  p_grad_test_dV[nDOF_test_element*nSpace],vel_grad_test_dV[nDOF_test_element*nSpace],
                  dV,x,y,z,xt,yt,zt,
                  //
                  porosity,
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
                ck.gradTrialFromRef(&vel_grad_trial_ref.data()[k*nDOF_trial_element*nSpace],jacInv,vel_grad_trial);
                //get the solution
                ck.valFromDOF(u_dof.data(),&vel_l2g.data()[eN_nDOF_trial_element],&vel_trial_ref.data()[k*nDOF_trial_element],u);
                ck.valFromDOF(v_dof.data(),&vel_l2g.data()[eN_nDOF_trial_element],&vel_trial_ref.data()[k*nDOF_trial_element],v);
                ck.valFromDOF(w_dof.data(),&vel_l2g.data()[eN_nDOF_trial_element],&vel_trial_ref.data()[k*nDOF_trial_element],w);

                //precalculate test function products with integration weights
                for (int j=0;j<nDOF_trial_element;j++)
                  {
                    vel_test_dV[j] = vel_test_ref.data()[k*nDOF_trial_element+j]*dV;
                  }
                //VRANS
                porosity      = q_porosity.data()[eN_k];
                //meanGrainSize = q_meanGrain[eN_k];
                //
                //save velocity at quadrature points for other models to use
                q_velocity.data()[eN_k_nSpace+0]=u;
                q_velocity.data()[eN_k_nSpace+1]=v;
                q_velocity.data()[eN_k_nSpace+2]=w;
                q_x.data()[eN_k_nSpace+0]=x;
                q_x.data()[eN_k_nSpace+1]=y;
                q_x.data()[eN_k_nSpace+2]=z;



                //
                //moving mesh
                //

                //
                //calculate time derivative at quadrature points
                //
                ck.bdf(alphaBDF,
                       q_mom_u_acc_beta_bdf.data()[eN_k],
                       mom_u_acc,
                       dmom_u_acc_u,
                       mom_u_acc_t,
                       dmom_u_acc_u_t);
                ck.bdf(alphaBDF,
                       q_mom_v_acc_beta_bdf.data()[eN_k],
                       mom_v_acc,
                       dmom_v_acc_v,
                       mom_v_acc_t,
                       dmom_v_acc_v_t);
                ck.bdf(alphaBDF,
                       q_mom_w_acc_beta_bdf.data()[eN_k],
                       mom_w_acc,
                       dmom_w_acc_w,
                       mom_w_acc_t,
                       dmom_w_acc_w_t);
                //
                calculateBeamSinks(nBeams,
                                   nBeamElements,
                                   beam_quadOrder,
                                   beam_Cd,
                                   beamRadius.data(),
                                   x,
                                   y,
                                   z,
                                   xq.data(),
                                   yq.data(),
                                   zq.data(),
                                   Beam_h.data(),
                                   dV_beam.data(),
                                   u,
                                   v,
                                   w,
                                   eps_rho,
                                   phi.data()[eN_k],
                                   rho_0,
                                   rho_1,
                                   q_dragBeam1.data()[eN_k],
                                   q_dragBeam2.data()[eN_k],
                                   q_dragBeam3.data()[eN_k],
                                   dV,
                                   owned ? netBeamDrag.data() : ghostBeamDrag);
                if (owned)
                calculateBeamLoads(nBeams,
                                   nBeamElements,
                                   beam_quadOrder,
                                   beam_Cd,
                                   beamRadius.data(),
                                   x,
                                   y,
                                   z,
                                   xq.data(),
                                   yq.data(),
                                   zq.data(),
                                   Beam_h.data(),
                                   u,
                                   v,
                                   w,
                                   eps_rho,
                                   phi.data()[eN_k],
                                   rho_0,
                                   rho_1,
                                   q1.data(),
                                   q2.data(),
                                   q3.data(),
                                   dV,
                                   vel_avg.data());

                //

              }
            //
          }//elements
        //
        //loop over exterior element boundaries to calculate surface integrals and load into element and global residuals
        //
        //ebNE is the Exterior element boundary INdex
        //ebN is the element boundary INdex
        //eN is the element index
        for (int ebNE = 0; ebNE < nExteriorElementBoundaries_global; ebNE++)
          {
            int ebN = exteriorElementBoundariesArray.data()[ebNE],
              eN  = elementBoundaryElementsArray.data()[ebN*2+0],
              ebN_local = elementBoundaryLocalElementBoundariesArray.data()[ebN*2+0],
              eN_nDOF_trial_element = eN*nDOF_trial_element;
            double eps_rho, eps_mu;
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
                  mom_v_ham_ext=0.0,
                  dmom_v_ham_grad_p_ext[nSpace],
                  mom_w_ham_ext=0.0,
                  dmom_w_ham_grad_p_ext[nSpace],
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
                  bc_mom_v_ham_ext=0.0,
                  bc_dmom_v_ham_grad_p_ext[nSpace],
                  bc_mom_w_ham_ext=0.0,
                  bc_dmom_w_ham_grad_p_ext[nSpace],
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
                  force_x,force_y,force_z,force_p_x,force_p_y,force_p_z,force_v_x,force_v_y,force_v_z,r_x,r_y,r_z, junk[nSpace];
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
                //std::cout<<"integralScaling - metricTensorDetSrt ==============================="<<integralScaling-metricTensorDetSqrt<<std::endl;
                /* std::cout<<"metricTensorDetSqrt "<<metricTensorDetSqrt */
                /*               <<"dS_ref.data()[kb]"<<dS_ref.data()[kb]<<std::endl; */
                dS = ((1.0-MOVING_DOMAIN)*metricTensorDetSqrt + MOVING_DOMAIN*integralScaling)*dS_ref.data()[kb];
                //get the metric tensor
                //cek todo use symmetry
                ck.calculateG(jacInv_ext,G,G_dd_G,tr_G);
                ck.calculateGScale(G,&ebqe_normal_phi_ext.data()[ebNE_kb_nSpace],h_phi);

                eps_rho = epsFact_rho*(useMetrics*h_phi+(1.0-useMetrics)*elementDiameter.data()[eN]);
                eps_mu  = epsFact_mu *(useMetrics*h_phi+(1.0-useMetrics)*elementDiameter.data()[eN]);

                //compute shape and solution information
                //shape
                ck.gradTrialFromRef(&vel_grad_trial_trace_ref.data()[ebN_local_kb_nSpace*nDOF_trial_element],jacInv_ext,vel_grad_trial_trace);
                //cek hack use trial ck.gradTrialFromRef(&vel_grad_test_trace_ref.data()[ebN_local_kb_nSpace*nDOF_trial_element],jacInv_ext,vel_grad_test_trace);
                //solution and gradients
                ck.valFromDOF(u_dof.data(),&vel_l2g.data()[eN_nDOF_trial_element],&vel_trial_trace_ref.data()[ebN_local_kb*nDOF_test_element],u_ext);
                ck.valFromDOF(v_dof.data(),&vel_l2g.data()[eN_nDOF_trial_element],&vel_trial_trace_ref.data()[ebN_local_kb*nDOF_test_element],v_ext);
                ck.valFromDOF(w_dof.data(),&vel_l2g.data()[eN_nDOF_trial_element],&vel_trial_trace_ref.data()[ebN_local_kb*nDOF_test_element],w_ext);

                //precalculate test function products with integration weights
                for (int j=0;j<nDOF_trial_element;j++)
                  {
                    vel_test_dS[j] = vel_test_trace_ref.data()[ebN_local_kb*nDOF_test_element+j]*dS;
                  }

                bc_u_ext = isDOFBoundary_u.data()[ebNE_kb]*ebqe_bc_u_ext.data()[ebNE_kb]+(1-isDOFBoundary_u.data()[ebNE_kb])*u_ext;
                bc_v_ext = isDOFBoundary_v.data()[ebNE_kb]*ebqe_bc_v_ext.data()[ebNE_kb]+(1-isDOFBoundary_v.data()[ebNE_kb])*v_ext;
                bc_w_ext = isDOFBoundary_w.data()[ebNE_kb]*ebqe_bc_w_ext.data()[ebNE_kb]+(1-isDOFBoundary_w.data()[ebNE_kb])*w_ext;
                //VRANS
                porosity_ext = ebqe_porosity_ext.data()[ebNE_kb];
                calculateBeamSinks(nBeams,
                                   nBeamElements,
                                   beam_quadOrder,
                                   beam_Cd,
                                   beamRadius.data(),
                                   x_ext,
                                   y_ext,
                                   z_ext,
                                   xq.data(),
                                   yq.data(),
                                   zq.data(),
                                   Beam_h.data(),
                                   dV_beam.data(),
                                   u_ext,
                                   v_ext,
                                   w_ext,
                                   eps_rho,
                                   phi.data()[ebNE_kb],
                                   rho_0,
                                   rho_1,
                                   ebqe_dragBeam1.data()[ebNE_kb],
                                   ebqe_dragBeam2.data()[ebNE_kb],
                                   ebqe_dragBeam3.data()[ebNE_kb],
                                   dS,
                                   junk);
                //

              }//kb
            //
            //update the element and global residual storage
            //

          }//ebNE
      }



    };//RANS2P

  inline RANS2P_IB_base* newRANS2P_IB(int nSpaceIn,
                                      int nQuadraturePoints_elementIn,
                                      int nDOF_mesh_trial_elementIn,
                                      int nDOF_trial_elementIn,
                                      int nDOF_test_elementIn,
                                      int nQuadraturePoints_elementBoundaryIn,
                                      int CompKernelFlag)
  {
    return proteus::chooseAndAllocateDiscretization<RANS2P_IB_base,RANS2P_IB,CompKernel>(nSpaceIn,
                                                                                         nQuadraturePoints_elementIn,
                                                                                         nDOF_mesh_trial_elementIn,
                                                                                         nDOF_trial_elementIn,
                                                                                         nDOF_test_elementIn,
                                                                                         nQuadraturePoints_elementBoundaryIn,
                                                                                         CompKernelFlag);
  }
}//proteus

#endif
