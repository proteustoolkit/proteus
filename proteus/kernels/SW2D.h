#ifndef SW2D_H
#define SW2D_H
#include <cmath>
#include <iostream>
#include "CompKernel.h"
#include "ModelFactory.h"
#include "ArgumentsDict.h"
#include "pyarray.h"

namespace py = pybind11;

//cek todo
//2. Get stabilization right
//3. Add Riemann solvers for external flux
//4. Add Riemann solvers for internal flux and DG terms
//5. Try other choices of variables h,hu,hv, Bova-Carey symmetrization?
namespace proteus
{
  class SW2D_base
  {
  public:
    virtual ~SW2D_base(){}
    virtual void calculateResidual(arguments_dict& args)=0;
    virtual void calculateJacobian(arguments_dict& args)=0;
    virtual void calculateResidual_supg(arguments_dict& args)=0;
    virtual void calculateJacobian_supg(arguments_dict& args)=0;
  };

  template<class CompKernelType,
           int nSpace,
           int nQuadraturePoints_element,
           int nDOF_mesh_trial_element,
           int nDOF_trial_element,
           int nDOF_test_element,
           int nQuadraturePoints_elementBoundary>
  class SW2D : public SW2D_base
  {
  public:
    const int nDOF_test_X_trial_element;
    CompKernelType ck;
    SW2D():
      nDOF_test_X_trial_element(nDOF_test_element*nDOF_trial_element),
      ck()
    {
      std::cout<<"Constructing SW2D<CompKernelTemplate<"
               <<nSpace<<","
               <<nQuadraturePoints_element<<","
               <<nDOF_mesh_trial_element<<","
               <<nDOF_trial_element<<","
               <<nDOF_test_element<<","
               <<nQuadraturePoints_elementBoundary<<">());"
               <<std::endl<<std::flush;
    }

    inline
      void evaluateCoefficients(const double nu,
                                const double g,
                                const double grad_b[nSpace],
                                const double& h,
                                const double& u,
                                const double& v,
                                double& mass_acc,
                                double& dmass_acc_h,
                                double& mom_u_acc,
                                double& dmom_u_acc_h,
                                double& dmom_u_acc_u,
                                double& mom_v_acc,
                                double& dmom_v_acc_h,
                                double& dmom_v_acc_v,
                                double mass_adv[nSpace],
                                double dmass_adv_h[nSpace],
                                double dmass_adv_u[nSpace],
                                double dmass_adv_v[nSpace],
                                double mom_u_adv[nSpace],
                                double dmom_u_adv_h[nSpace],
                                double dmom_u_adv_u[nSpace],
                                double dmom_u_adv_v[nSpace],
                                double mom_v_adv[nSpace],
                                double dmom_v_adv_h[nSpace],
                                double dmom_v_adv_u[nSpace],
                                double dmom_v_adv_v[nSpace],
                                double mom_u_diff_ten[nSpace],
                                double mom_v_diff_ten[nSpace],
                                double mom_uv_diff_ten[1],
                                double mom_vu_diff_ten[1],
                                double& mom_u_source,
                                double& dmom_u_source_h,
                                double& mom_v_source,
                                double& dmom_v_source_h)
    {
      //mass accumulation
      mass_acc = h;
      dmass_acc_h = 1.0;

      //u momentum accumulation
      mom_u_acc=h*u;
      dmom_u_acc_h=u;
      dmom_u_acc_u=h;

      //v momentum accumulation
      mom_v_acc=h*v;
      dmom_v_acc_h=v;
      dmom_v_acc_v=h;

      //mass advective flux
      mass_adv[0]=h*u;
      mass_adv[1]=h*v;

      dmass_adv_h[0]=u;
      dmass_adv_h[1]=v;

      dmass_adv_u[0]=h;
      dmass_adv_u[1]=0.0;

      dmass_adv_v[0]=0.0;
      dmass_adv_v[1]=h;

      //u momentum advective flux
      mom_u_adv[0]=h*u*u + 0.5*g*h*h;
      mom_u_adv[1]=h*u*v;

      dmom_u_adv_h[0]=u*u + g*h;
      dmom_u_adv_h[1]=u*v;

      dmom_u_adv_u[0]=h*2.0*u;
      dmom_u_adv_u[1]=h*v;

      dmom_u_adv_v[0]=0.0;
      dmom_u_adv_v[1]=h*u;

      //v momentum advective_flux
      mom_v_adv[0]=h*v*u;
      mom_v_adv[1]=h*v*v + 0.5*g*h*h;

      dmom_v_adv_h[0]=v*u;
      dmom_v_adv_h[1]=v*v + g*h;

      dmom_v_adv_u[0]=h*v;
      dmom_v_adv_u[1]=0.0;

      dmom_v_adv_v[0]=h*u;
      dmom_v_adv_v[1]=h*2.0*v;

      //u momentum diffusion tensor
      mom_u_diff_ten[0] = 2.0*nu;
      mom_u_diff_ten[1] = nu;

      mom_uv_diff_ten[0]=nu;

      //v momentum diffusion tensor
      mom_v_diff_ten[0] = nu;
      mom_v_diff_ten[1] = 2.0*nu;

      mom_vu_diff_ten[0]=nu;

      //momentum sources
      mom_u_source = g*h*grad_b[0];
      dmom_u_source_h = g*grad_b[0];

      mom_v_source = g*h*grad_b[1];
      dmom_v_source_h = g*grad_b[1];
    }

    inline
    void calculateSubgridError_tau(const double& elementDiameter,
                                   const double& nu,
                                   const double& g,
                                   const double& h,
                                   const double& u,
                                   const double& v,
                                   double tau[9],
                                   double& cfl)
    {
      //todo go back through rx,ry, etc. and the formula for tau in physical variables once  this is known.
      double rx[9],ry[9],rxInv[9],ryInv[9],c=sqrt(fmax(1.0e-8,g*h)),lambdax[3],lambday[3],tauxHat[3],tauyHat[3],taux[9],tauy[9],tauc[9],cflx,cfly,L[9],hStar=fmax(1.0e-8,h);
      //eigenvalues and eigenvectors for conservation variables h,hu,hv
      lambdax[0] = u - c;
      lambdax[1] = u;
      lambdax[2] = u + c;
      if (u > 0.0)
        cflx = (u+c)/elementDiameter;
      else
        cflx = fabs(u-c)/elementDiameter;

      double rn=sqrt(1.0+(u-c)*(u-c) + v*v);
      rx[0*3+0] = 1.0/rn;
      rx[1*3+0] = (u - c)/rn;
      rx[2*3+0] = v/rn;

      rx[0*3+1] = 0.0;
      rx[1*3+1] = 0.0;
      rx[2*3+1] = 1.0;

      rn = sqrt(1.0 + (u+c)*(u+c) + v*v);
      rx[0*3+2] = 1.0/rn;
      rx[1*3+2] = (u + c)/rn;
      rx[2*3+2] = v/rn;

      rxInv[0*3+0] = 1.0 - (c - u)/(2.0*c);
      rxInv[1*3+0] = -v;
      rxInv[2*3+0] = (c-u)/(2.0*c);

      rxInv[0*3+1] = -1.0/(2.0*c);
      rxInv[1*3+1] =  0.0;
      rxInv[2*3+1] =  1.0/(2.0*c);

      rxInv[0*3+2] = 0.0;
      rxInv[1*3+2] = 1.0;
      rxInv[2*3+2] = 0.0;

      /* //cek hack assume no viscosity for now */
      /* num = 0.5*fabs(lambdax[0])*elementDiameter + 1.0e-8; */
      /* den = nu + num*1.0e-8; */
      /* pe = num/den; */
      /* tauxHat[0] = 0.5*h*(1.0/tanh(pe) - 1.0/pe)/(Vlin+1.0e-8); */

      tauxHat[0] = 0.5*elementDiameter/(fabs(lambdax[0])+1.0e-8);
      tauxHat[1] = 0.5*elementDiameter/(fabs(lambdax[1])+1.0e-8);
      tauxHat[2] = 0.5*elementDiameter/(fabs(lambdax[2])+1.0e-8);

      lambday[0] = v - c;
      lambday[1] = v;
      lambday[2] = v + c;

      rn=sqrt(1.0 + u*u+(v-c)*(v-c));
      ry[0*3+0] = 1.0/rn;
      ry[1*3+0] = u/rn;
      ry[2*3+0] = (v - c)/rn;

      ry[0*3+1] =  0.0;
      ry[1*3+1] = -1.0;
      ry[2*3+1] =  0.0;

      rn = sqrt(1.0 + u*u + (v+c)*(v+c));
      ry[0*3+2] = 1.0/rn;
      ry[1*3+2] = u/rn;
      ry[2*3+2] = (v + c)/rn;

      ryInv[0*3+0] = 1.0 - (c - v)/(2*c);
      ryInv[1*3+0] = u;
      ryInv[2*3+0] = (c-v)/(2*c);

      ryInv[0*3+1] =  0.0;
      ryInv[1*3+1] = -1.0;
      ryInv[2*3+1] =  0.0;

      ryInv[0*3+2] = -1.0/(2*c);
      ryInv[1*3+2] =  0.0;
      ryInv[2*3+2] =  1.0/(2*c);

      //cek hack assume no viscosity for now
      tauyHat[0] = 0.5*elementDiameter/(fabs(lambday[0])+1.0e-8);
      tauyHat[1] = 0.5*elementDiameter/(fabs(lambday[1])+1.0e-8);
      tauyHat[2] = 0.5*elementDiameter/(fabs(lambday[2])+1.0e-8);
      if (v > 0.0)
        cfly = (v+c)/elementDiameter;
      else
        cfly = fabs(v-c)/elementDiameter;
      cfl = sqrt(cflx*cflx+cfly*cfly);//hack, conservative estimate

      //project back to solution variables
      //initialize, hack do with memset
      double tmpx[9],tmpy[9];
      for (int i=0;i<3;i++)
        for (int j=0;j<3;j++)
          {
            taux[i*3+j] = 0.0;
            tauy[i*3+j] = 0.0;
            tauc[i*3+j] = 0.0;
            tau[i*3+j] = 0.0;
            tmpx[i*3+j] = 0.0;
            tmpy[i*3+j] = 0.0;
            L[i*3+j] = 0.0;
            /* double Ix=0,Iy=0.0; */
            /* for (int k=0;k<3;k++) */
            /*   { */
            /*  Ix += rx[i*3+k]*rxInv[k*3+j]; */
            /*  Iy += ry[i*3+k]*ryInv[k*3+j]; */
            /*   } */
            /* std::cout<<i<<'\t'<<j<<'\t'<<Ix<<'\t'<<Iy<<std::endl; */
          }
      //transform from characteristic variables to conservation variables
      for (int i=0;i<3;i++)
        for (int j=0;j<3;j++)
          for (int m=0;m<3;m++)
          {
            //taux[i*3+j] += rx[i*3+m]*tauxHat[m]*rxInv[m*3+j];
            //tauy[i*3+j] += ry[i*3+m]*tauyHat[m]*ryInv[m*3+j];
            if (m==j)
              {
                tmpx[i*3+m] += rx[i*3+m]*tauxHat[m];
                tmpy[i*3+m] += ry[i*3+m]*tauyHat[m];
              }
          }
      for (int i=0;i<3;i++)
        for (int j=0;j<3;j++)
          for (int m=0;m<3;m++)
          {
            //taux[i*3+j] += tmpx[i*3+m]*rxInv[m*3 + j];
            //tauy[i*3+j] += tmpy[i*3+m]*ryInv[m*3 + j];
            taux[i*3+j] += tmpx[i*3+m]*rx[j*3 + m];
            tauy[i*3+j] += tmpy[i*3+m]*ry[j*3 + m];
          }
      //matrix norm
      for (int i=0;i<3;i++)
        for (int j=0;j<3;j++)
          tauc[i*3+j] = sqrt(taux[i*3+j]*taux[i*3+j] + tauy[i*3+j]*tauy[i*3+j]);
      //transform from conservation variables to solution variables h,u,v
      L[0*3+0] = 1.0;
      L[1*3+0] = -u/hStar;
      L[1*3+1] = 1.0/hStar;
      L[2*3+0] = -v/hStar;
      L[2*3+2] = 1.0/hStar;
      for (int i=0;i<3;i++)
        for (int j=0;j<3;j++)
          {
            for (int k=0;k<3;k++)
              tau[i*3+j] += L[i*3+k]*tauc[k*3+j];
            //cek hack, turn it off until I get the ASGS stuff straigtened out
            tau[i*3+j] = 0.0;
          }
      /* std::cout<<"tau"<<std::endl; */
      /* for (int i=0;i<3;i++) */
      /*        { */
      /*          for (int j=0;j<3;j++) */
      /*            { */
      /*              std::cout<<tau[i*3+j]<<'\t'; */
      /*            } */
      /*          std::cout<<std::endl; */
      /*        } */
    }
    inline
    void calculateSubgridError_tau_supg(const double& elementDiameter,
                                       const double& nu,
                                       const double& g,
                                       const double& h,
                                       const double& u,
                                       const double& v,
                                       const double& alpha,
                                       const double& area,
                                       double tau_x[9],
                                       double tau_y[9],
                                       double& cfl)
    {
      double c=sqrt(fmax(1.0e-8,g*h)),lambdax[3],lambday[3],cflx,cfly,hStar=fmax(1.0e-8,h),dx,dy,a,ainv;
      int i=0;
      //eigenvalues for x and y, using Berger and Stockstill notation where possible
      //tau_x = \alpha \Delta x \hat{A}, tau_y = \alpha \Delta y \hat{B}
      lambdax[0] = u + c;
      lambdax[1] = u - c;
      lambdax[2] = u;
      if (u > 0.0)
        cflx = (u+c)/elementDiameter;
      else
        cflx = fabs(u-c)/elementDiameter;
      lambday[0] = v + c;
      lambday[1] = v - c;
      lambday[2] = v;
      if (v > 0.0)
        cfly = (v+c)/elementDiameter;
      else
        cfly = fabs(v-c)/elementDiameter;

      cfl = sqrt(cflx*cflx+cfly*cfly);

      dx = sqrt(fabs(area)); dy = sqrt(fabs(area));
      a = sqrt(u*u + v*v + c*c);
      ainv = 1.0/(a+1.0e-8);
      //\hat{A}\alpha\Delta x
      tau_x[0*3+0]= u;   tau_x[0*3+1]=h;   tau_x[0*3+2]=0.;
      tau_x[1*3+0]= h*g; tau_x[1*3+1]=h*u; tau_x[1*3+2]=0.;
      tau_x[2*3+0]= 0.;  tau_x[2*3+1]=0.;  tau_x[2*3+2]=u*h;
      for (i=0; i < 9; i++)
        tau_x[i] *= alpha*dx*ainv;
      //\hat{B}\alpha\Delta y
      tau_y[0*3+0]= v;   tau_y[0*3+1]=0;   tau_y[0*3+2]=h;
      tau_y[1*3+0]= 0.;  tau_y[1*3+1]=h*v; tau_y[1*3+2]=0.;
      tau_y[2*3+0]= h*g; tau_y[2*3+1]=0.;  tau_y[2*3+2]=v*h;
      for (i=0; i < 9; i++)
        tau_y[i] *= alpha*dy*ainv;

    }

    inline
      void exteriorNumericalAdvectiveFlux(const int& isDOFBoundary_h,
                                          const int& isDOFBoundary_u,
                                          const int& isDOFBoundary_v,
                                          const int& isFluxBoundary_h,
                                          const int& isFluxBoundary_u,
                                          const int& isFluxBoundary_v,
                                          const double n[nSpace],
                                          const double& bc_h,
                                          const double bc_f_mass[nSpace],
                                          const double bc_f_umom[nSpace],
                                          const double bc_f_vmom[nSpace],
                                          const double& bc_flux_mass,
                                          const double& bc_flux_umom,
                                          const double& bc_flux_vmom,
                                          const double& h,
                                          const double f_mass[nSpace],
                                          const double f_umom[nSpace],
                                          const double f_vmom[nSpace],
                                          const double df_mass_dh[nSpace],
                                          const double df_mass_du[nSpace],
                                          const double df_mass_dv[nSpace],
                                          const double df_umom_dh[nSpace],
                                          const double df_umom_du[nSpace],
                                          const double df_umom_dv[nSpace],
                                          const double df_vmom_dh[nSpace],
                                          const double df_vmom_du[nSpace],
                                          const double df_vmom_dv[nSpace],
                                          double& flux_mass,
                                          double& flux_umom,
                                          double& flux_vmom,
                                          double* velocity)
    {
      //cek todo, need to do the Riemann solve
      /* double flowDirection; */
      flux_mass = 0.0;
      flux_umom = 0.0;
      flux_vmom = 0.0;
      /* flowDirection=n[0]*f_mass[0]+n[1]*f_mass[1]; */
      /* if (isDOFBoundary_u != 1) */
      /*        { */
      /*          flux_mass += n[0]*f_mass[0]; */
      /*          velocity[0] = f_mass[0]; */
      /*          if (flowDirection >= 0.0) */
      /*            { */
      /*              flux_umom += n[0]*f_umom[0]; */
      /*              flux_vmom += n[0]*f_vmom[0]; */
      /*            } */
      /*        } */
      /* else */
      /*        { */
      /*          flux_mass += n[0]*bc_f_mass[0]; */
      /*          velocity[0] = bc_f_mass[0]; */
      /*          //cek still upwind the advection for Dirichlet? */
      /*          if (flowDirection >= 0.0) */
      /*            { */
      /*              flux_umom += n[0]*f_umom[0]; */
      /*              flux_vmom += n[0]*f_vmom[0]; */
      /*            } */
      /*          else */
      /*            { */
      /*              flux_umom+=n[0]*bc_f_umom[0]; */
      /*              flux_vmom+=n[0]*bc_f_vmom[0]; */
      /*            } */
      /*        } */
      /* if (isDOFBoundary_v != 1) */
      /*        { */
      /*          flux_mass+=n[1]*f_mass[1]; */
      /*          velocity[1] = f_mass[1]; */
      /*          if (flowDirection >= 0.0) */
      /*            { */
      /*              flux_umom+=n[1]*f_umom[1]; */
      /*              flux_vmom+=n[1]*f_vmom[1]; */
      /*            } */
      /*        } */
      /* else */
      /*        { */
      /*          flux_mass+=n[1]*bc_f_mass[1]; */
      /*          velocity[1] = bc_f_mass[1]; */
      /*          //cek still upwind the advection for Dirichlet? */
      /*          if (flowDirection >= 0.0) */
      /*            { */
      /*              flux_umom+=n[1]*f_umom[1]; */
      /*              flux_vmom+=n[1]*f_vmom[1]; */
      /*            } */
      /*          else */
      /*            { */
      /*              flux_umom+=n[1]*bc_f_umom[1]; */
      /*              flux_vmom+=n[1]*bc_f_vmom[1]; */
      /*            } */
      /*        } */
      /* else */
      /*        { */
      /*          flux_mass +=n[2]*bc_f_mass[2]; */
      /*          velocity[2] = bc_f_mass[2]; */
      /*          //cek still upwind the advection for Dirichlet? */
      /*          if (flowDirection >= 0.0) */
      /*            { */
      /*              flux_umom+=n[2]*f_umom[2]; */
      /*              flux_vmom+=n[2]*f_vmom[2]; */
      /*            } */
      /*          else */
      /*            { */
      /*              flux_umom+=n[2]*bc_f_umom[2]; */
      /*              flux_vmom+=n[2]*bc_f_vmom[2]; */
      /*            } */
      /*        } */
      /* if (isDOFBoundary_h == 1) */
      /*        { */
      /*          flux_umom+= n[0]*(bc_h-p)*oneByRho; */
      /*          flux_vmom+= n[1]*(bc_h-p)*oneByRho; */
      /*        } */
      if (isFluxBoundary_h == 1)
        {
          //cek todo, not sure if we'll need this for SW2
          //velocity[0] += (bc_flux_mass - flux_mass)*n[0];
          //velocity[1] += (bc_flux_mass - flux_mass)*n[1];
          //velocity[2] += (bc_flux_mass - flux_mass)*n[2];
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
    }

    inline
    void exteriorNumericalAdvectiveFluxDerivatives(const int& isDOFBoundary_h,
                                                   const int& isDOFBoundary_u,
                                                   const int& isDOFBoundary_v,
                                                   const int& isFluxBoundary_h,
                                                   const int& isFluxBoundary_u,
                                                   const int& isFluxBoundary_v,
                                                   const double n[nSpace],
                                                   const double& bc_h,
                                                   const double bc_f_mass[nSpace],
                                                   const double bc_f_umom[nSpace],
                                                   const double bc_f_vmom[nSpace],
                                                   const double& bc_flux_mass,
                                                   const double& bc_flux_umom,
                                                   const double& bc_flux_vmom,
                                                   const double& h,
                                                   const double f_mass[nSpace],
                                                   const double f_umom[nSpace],
                                                   const double f_vmom[nSpace],
                                                   const double df_mass_du[nSpace],
                                                   const double df_mass_dv[nSpace],
                                                   const double df_umom_dh[nSpace],
                                                   const double df_umom_du[nSpace],
                                                   const double df_umom_dv[nSpace],
                                                   const double df_vmom_dh[nSpace],
                                                   const double df_vmom_du[nSpace],
                                                   const double df_vmom_dv[nSpace],
                                                   double& dflux_mass_dh,
                                                   double& dflux_mass_du,
                                                   double& dflux_mass_dv,
                                                   double& dflux_umom_dh,
                                                   double& dflux_umom_du,
                                                   double& dflux_umom_dv,
                                                   double& dflux_vmom_dh,
                                                   double& dflux_vmom_du,
                                                   double& dflux_vmom_dv)
    {
      double flowDirection;
      dflux_mass_dh = 0.0;
      dflux_mass_du = 0.0;
      dflux_mass_dv = 0.0;

      dflux_umom_dh = 0.0;
      dflux_umom_du = 0.0;
      dflux_umom_dv = 0.0;

      dflux_vmom_dh = 0.0;
      dflux_vmom_du = 0.0;
      dflux_vmom_dv = 0.0;

      flowDirection=n[0]*f_mass[0]+n[1]*f_mass[1];
      /* if (isDOFBoundary_u != 1) */
      /*        { */
      /*          dflux_mass_du += n[0]*df_mass_du[0]; */
      /*          if (flowDirection >= 0.0) */
      /*            { */
      /*              dflux_umom_du += n[0]*df_umom_du[0]; */
      /*              dflux_vmom_du += n[0]*df_vmom_du[0]; */
      /*              dflux_vmom_dv += n[0]*df_vmom_dv[0]; */
      /*            } */
      /*        } */
      /* else */
      /*        { */
      /*          //cek still upwind the advection for Dirichlet? */
      /*          if (flowDirection >= 0.0) */
      /*            { */
      /*              dflux_umom_du += n[0]*df_umom_du[0]; */
      /*              dflux_vmom_du += n[0]*df_vmom_du[0]; */
      /*              dflux_vmom_dv += n[0]*df_vmom_dv[0]; */
      /*            } */
      /*          else */
      /*            { */
      /*              if (isDOFBoundary_v != 1) */
      /*                dflux_vmom_dv += n[0]*df_vmom_dv[0]; */
      /*            } */
      /*        } */
      /* if (isDOFBoundary_v != 1) */
      /*        { */
      /*          dflux_mass_dv += n[1]*df_mass_dv[1]; */
      /*          if (flowDirection >= 0.0) */
      /*            { */
      /*              dflux_umom_du += n[1]*df_umom_du[1]; */
      /*              dflux_umom_dv += n[1]*df_umom_dv[1]; */
      /*              dflux_vmom_dv += n[1]*df_vmom_dv[1]; */
      /*            } */
      /*        } */
      /* else */
      /*        { */
      /*          //cek still upwind the advection for Dirichlet? */
      /*          if (flowDirection >= 0.0) */
      /*            { */
      /*              dflux_umom_du += n[1]*df_umom_du[1]; */
      /*              dflux_umom_dv += n[1]*df_umom_dv[1]; */
      /*              dflux_vmom_dv += n[1]*df_vmom_dv[1]; */
      /*            } */
      /*          else */
      /*            { */
      /*              if (isDOFBoundary_u != 1) */
      /*                dflux_umom_du += n[1]*df_umom_du[1]; */
      /*            } */
      /*        } */
      /* else */
      /*        { */
      /*          //cek still upwind the advection for Dirichlet? */
      /*          if (flowDirection >= 0.0) */
      /*            { */
      /*              dflux_umom_du += n[2]*df_umom_du[2]; */
      /*              dflux_umom_dw += n[2]*df_umom_dw[2]; */
      /*              dflux_vmom_dv += n[2]*df_vmom_dv[2]; */
      /*            } */
      /*          else */
      /*            { */
      /*              if (isDOFBoundary_u != 1) */
      /*                dflux_umom_du += n[2]*df_umom_du[2]; */
      /*              if (isDOFBoundary_v != 1) */
      /*                dflux_vmom_dv += n[2]*df_vmom_dv[2]; */
      /*            } */
      /*        } */
      /* if (isDOFBoundary_h == 1) */
      /*        { */
      /*          dflux_umom_dp= -n[0]*oneByRho; */
      /*          dflux_vmom_dp= -n[1]*oneByRho; */
      /*        } */
      if (isFluxBoundary_h == 1)
        {
          dflux_mass_dh = 0.0;
          dflux_mass_du = 0.0;
          dflux_mass_dv = 0.0;
        }
      if (isFluxBoundary_u == 1)
        {
          dflux_umom_dh = 0.0;
          dflux_umom_du = 0.0;
          dflux_umom_dv = 0.0;
        }
      if (isFluxBoundary_v == 1)
        {
          dflux_vmom_dh = 0.0;
          dflux_vmom_du = 0.0;
          dflux_vmom_dv = 0.0;
        }
    }

    /* inline */
    /* void exteriorNumericalDiffusiveFlux(const double& eps, */
    /*                                  proteus_csr_index_t* rowptr, */
    /*                                  proteus_csr_index_t* colind, */
    /*                                  const int& isDOFBoundary, */
    /*                                  const int& isFluxBoundary, */
    /*                                  const double n[nSpace], */
    /*                                  double* bc_a, */
    /*                                  const double& bc_u, */
    /*                                  const double& bc_flux, */
    /*                                  double* a, */
    /*                                  const double grad_phi[nSpace], */
    /*                                  const double& u, */
    /*                                  const double& penalty, */
    /*                                  double& flux) */
    /* { */
    /*   double diffusiveVelocityComponent_I,penaltyFlux,max_a; */
    /*   if(isDOFBoundary == 1) */
    /*  { */
    /*    flux = 0.0; */
    /*    max_a=0.0; */
    /*    for(int I=0;I<nSpace;I++) */
    /*      { */
    /*        diffusiveVelocityComponent_I=0.0; */
    /*        for(proteus_csr_index_t m=rowptr[I];m<rowptr[I+1];m++) */
    /*          { */
    /*            diffusiveVelocityComponent_I -= a[m]*grad_phi[colind[m]]; */
    /*            max_a = fmax(max_a,a[m]); */
    /*          } */
    /*        flux+= diffusiveVelocityComponent_I*n[I]; */
    /*      } */
    /*    penaltyFlux = max_a*penalty*(u-bc_u); */
    /*    flux += penaltyFlux; */
    /*  } */
    /*   else if(isFluxBoundary == 1) */
    /*  { */
    /*    flux = bc_flux; */
    /*  } */
    /*   else */
    /*  { */
    /*    std::cerr<<"warning, diffusion term with no boundary condition set, setting diffusive flux to 0.0"<<std::endl; */
    /*    flux = 0.0; */
    /*  } */
    /* } */

    /* inline */
    /* double ExteriorNumericalDiffusiveFluxJacobian(const double& eps, */
    /*                                            proteus_csr_index_t* rowptr, */
    /*                                            proteus_csr_index_t* colind, */
    /*                                            const int& isDOFBoundary, */
    /*                                            const double n[nSpace], */
    /*                                            double* a, */
    /*                                            const double& v, */
    /*                                            const double grad_v[nSpace], */
    /*                                            const double& penalty) */
    /* { */
    /*   double dvel_I,tmp=0.0,max_a=0.0; */
    /*   if(isDOFBoundary >= 1) */
    /*  { */
    /*    for(int I=0;I<nSpace;I++) */
    /*      { */
    /*        dvel_I=0.0; */
    /*        for(proteus_csr_index_t m=rowptr[I];m<rowptr[I+1];m++) */
    /*          { */
    /*            dvel_I -= a[m]*grad_v[colind[m]]; */
    /*            max_a = fmax(max_a,a[m]); */
    /*          } */
    /*        tmp += dvel_I*n[I]; */
    /*      } */
    /*    tmp +=max_a*penalty*v; */
    /*  } */
    /*   return tmp; */
    /* } */

    void calculateResidual(arguments_dict& args)
    {
        proteus::pyarray<double>& mesh_trial_ref = args.array<double>("mesh_trial_ref");
        proteus::pyarray<double>& mesh_grad_trial_ref = args.array<double>("mesh_grad_trial_ref");
        proteus::pyarray<double>& mesh_dof = args.array<double>("mesh_dof");
        proteus::pyarray<double>& mesh_velocity_dof = args.array<double>("mesh_velocity_dof");
        double MOVING_DOMAIN = args.scalar<double>("MOVING_DOMAIN");
        proteus::pyarray<proteus_index_t>& mesh_l2g = args.array<proteus_index_t>("mesh_l2g");
        proteus::pyarray<double>& dV_ref = args.array<double>("dV_ref");
        proteus::pyarray<double>& h_trial_ref = args.array<double>("h_trial_ref");
        proteus::pyarray<double>& h_grad_trial_ref = args.array<double>("h_grad_trial_ref");
        proteus::pyarray<double>& h_test_ref = args.array<double>("h_test_ref");
        proteus::pyarray<double>& h_grad_test_ref = args.array<double>("h_grad_test_ref");
        proteus::pyarray<double>& vel_trial_ref = args.array<double>("vel_trial_ref");
        proteus::pyarray<double>& vel_grad_trial_ref = args.array<double>("vel_grad_trial_ref");
        proteus::pyarray<double>& vel_test_ref = args.array<double>("vel_test_ref");
        proteus::pyarray<double>& vel_grad_test_ref = args.array<double>("vel_grad_test_ref");
        proteus::pyarray<double>& mesh_trial_trace_ref = args.array<double>("mesh_trial_trace_ref");
        proteus::pyarray<double>& mesh_grad_trial_trace_ref = args.array<double>("mesh_grad_trial_trace_ref");
        proteus::pyarray<double>& dS_ref = args.array<double>("dS_ref");
        proteus::pyarray<double>& h_trial_trace_ref = args.array<double>("h_trial_trace_ref");
        proteus::pyarray<double>& h_grad_trial_trace_ref = args.array<double>("h_grad_trial_trace_ref");
        proteus::pyarray<double>& h_test_trace_ref = args.array<double>("h_test_trace_ref");
        proteus::pyarray<double>& h_grad_test_trace_ref = args.array<double>("h_grad_test_trace_ref");
        proteus::pyarray<double>& vel_trial_trace_ref = args.array<double>("vel_trial_trace_ref");
        proteus::pyarray<double>& vel_grad_trial_trace_ref = args.array<double>("vel_grad_trial_trace_ref");
        proteus::pyarray<double>& vel_test_trace_ref = args.array<double>("vel_test_trace_ref");
        proteus::pyarray<double>& vel_grad_test_trace_ref = args.array<double>("vel_grad_test_trace_ref");
        proteus::pyarray<double>& normal_ref = args.array<double>("normal_ref");
        proteus::pyarray<double>& boundaryJac_ref = args.array<double>("boundaryJac_ref");
        proteus::pyarray<double>& elementDiameter = args.array<double>("elementDiameter");
        proteus_index_t nElements_global = args.scalar<proteus_index_t>("nElements_global");
        double useRBLES = args.scalar<double>("useRBLES");
        double useMetrics = args.scalar<double>("useMetrics");
        double alphaBDF = args.scalar<double>("alphaBDF");
        double nu = args.scalar<double>("nu");
        double g = args.scalar<double>("g");
        double shockCapturingCoefficient = args.scalar<double>("shockCapturingCoefficient");
        proteus::pyarray<proteus_index_t>& h_l2g = args.array<proteus_index_t>("h_l2g");
        proteus::pyarray<proteus_index_t>& vel_l2g = args.array<proteus_index_t>("vel_l2g");
        proteus::pyarray<double>& b_dof = args.array<double>("b_dof");
        proteus::pyarray<double>& h_dof = args.array<double>("h_dof");
        proteus::pyarray<double>& u_dof = args.array<double>("u_dof");
        proteus::pyarray<double>& v_dof = args.array<double>("v_dof");
        proteus::pyarray<double>& h_dof_sge = args.array<double>("h_dof_sge");
        proteus::pyarray<double>& u_dof_sge = args.array<double>("u_dof_sge");
        proteus::pyarray<double>& v_dof_sge = args.array<double>("v_dof_sge");
        proteus::pyarray<double>& q_mass_acc = args.array<double>("q_mass_acc");
        proteus::pyarray<double>& q_mom_u_acc = args.array<double>("q_mom_u_acc");
        proteus::pyarray<double>& q_mom_v_acc = args.array<double>("q_mom_v_acc");
        proteus::pyarray<double>& q_mass_adv = args.array<double>("q_mass_adv");
        proteus::pyarray<double>& q_mass_acc_beta_bdf = args.array<double>("q_mass_acc_beta_bdf");
        proteus::pyarray<double>& q_mom_u_acc_beta_bdf = args.array<double>("q_mom_u_acc_beta_bdf");
        proteus::pyarray<double>& q_mom_v_acc_beta_bdf = args.array<double>("q_mom_v_acc_beta_bdf");
        proteus::pyarray<double>& q_velocity_sge = args.array<double>("q_velocity_sge");
        proteus::pyarray<double>& q_cfl = args.array<double>("q_cfl");
        proteus::pyarray<double>& q_numDiff_h = args.array<double>("q_numDiff_h");
        proteus::pyarray<double>& q_numDiff_u = args.array<double>("q_numDiff_u");
        proteus::pyarray<double>& q_numDiff_v = args.array<double>("q_numDiff_v");
        proteus::pyarray<double>& q_numDiff_h_last = args.array<double>("q_numDiff_h_last");
        proteus::pyarray<double>& q_numDiff_u_last = args.array<double>("q_numDiff_u_last");
        proteus::pyarray<double>& q_numDiff_v_last = args.array<double>("q_numDiff_v_last");
        proteus::pyarray<proteus_csr_index_t>& sdInfo_u_u_rowptr = args.array<proteus_csr_index_t>("sdInfo_u_u_rowptr");
        proteus::pyarray<proteus_csr_index_t>& sdInfo_u_u_colind = args.array<proteus_csr_index_t>("sdInfo_u_u_colind");
        proteus::pyarray<proteus_csr_index_t>& sdInfo_u_v_rowptr = args.array<proteus_csr_index_t>("sdInfo_u_v_rowptr");
        proteus::pyarray<proteus_csr_index_t>& sdInfo_u_v_colind = args.array<proteus_csr_index_t>("sdInfo_u_v_colind");
        proteus::pyarray<proteus_csr_index_t>& sdInfo_v_v_rowptr = args.array<proteus_csr_index_t>("sdInfo_v_v_rowptr");
        proteus::pyarray<proteus_csr_index_t>& sdInfo_v_v_colind = args.array<proteus_csr_index_t>("sdInfo_v_v_colind");
        proteus::pyarray<proteus_csr_index_t>& sdInfo_v_u_rowptr = args.array<proteus_csr_index_t>("sdInfo_v_u_rowptr");
        proteus::pyarray<proteus_csr_index_t>& sdInfo_v_u_colind = args.array<proteus_csr_index_t>("sdInfo_v_u_colind");
        proteus_index_t offset_h = args.scalar<proteus_index_t>("offset_h");
        proteus_index_t offset_u = args.scalar<proteus_index_t>("offset_u");
        proteus_index_t offset_v = args.scalar<proteus_index_t>("offset_v");
        proteus_index_t stride_h = args.scalar<proteus_index_t>("stride_h");
        proteus_index_t stride_u = args.scalar<proteus_index_t>("stride_u");
        proteus_index_t stride_v = args.scalar<proteus_index_t>("stride_v");
        proteus::pyarray<double>& globalResidual = args.array<double>("globalResidual");
        proteus_index_t nExteriorElementBoundaries_global = args.scalar<proteus_index_t>("nExteriorElementBoundaries_global");
        proteus::pyarray<proteus_index_t>& exteriorElementBoundariesArray = args.array<proteus_index_t>("exteriorElementBoundariesArray");
        proteus::pyarray<proteus_index_t>& elementBoundaryElementsArray = args.array<proteus_index_t>("elementBoundaryElementsArray");
        proteus::pyarray<int>& elementBoundaryLocalElementBoundariesArray = args.array<int>("elementBoundaryLocalElementBoundariesArray");
        proteus::pyarray<int>& isDOFBoundary_h = args.array<int>("isDOFBoundary_h");
        proteus::pyarray<int>& isDOFBoundary_u = args.array<int>("isDOFBoundary_u");
        proteus::pyarray<int>& isDOFBoundary_v = args.array<int>("isDOFBoundary_v");
        proteus::pyarray<int>& isAdvectiveFluxBoundary_h = args.array<int>("isAdvectiveFluxBoundary_h");
        proteus::pyarray<int>& isAdvectiveFluxBoundary_u = args.array<int>("isAdvectiveFluxBoundary_u");
        proteus::pyarray<int>& isAdvectiveFluxBoundary_v = args.array<int>("isAdvectiveFluxBoundary_v");
        proteus::pyarray<int>& isDiffusiveFluxBoundary_u = args.array<int>("isDiffusiveFluxBoundary_u");
        proteus::pyarray<int>& isDiffusiveFluxBoundary_v = args.array<int>("isDiffusiveFluxBoundary_v");
        proteus::pyarray<double>& ebqe_bc_h_ext = args.array<double>("ebqe_bc_h_ext");
        proteus::pyarray<double>& ebqe_bc_flux_mass_ext = args.array<double>("ebqe_bc_flux_mass_ext");
        proteus::pyarray<double>& ebqe_bc_flux_mom_u_adv_ext = args.array<double>("ebqe_bc_flux_mom_u_adv_ext");
        proteus::pyarray<double>& ebqe_bc_flux_mom_v_adv_ext = args.array<double>("ebqe_bc_flux_mom_v_adv_ext");
        proteus::pyarray<double>& ebqe_bc_u_ext = args.array<double>("ebqe_bc_u_ext");
        proteus::pyarray<double>& ebqe_bc_flux_u_diff_ext = args.array<double>("ebqe_bc_flux_u_diff_ext");
        proteus::pyarray<double>& ebqe_penalty_ext = args.array<double>("ebqe_penalty_ext");
        proteus::pyarray<double>& ebqe_bc_v_ext = args.array<double>("ebqe_bc_v_ext");
        proteus::pyarray<double>& ebqe_bc_flux_v_diff_ext = args.array<double>("ebqe_bc_flux_v_diff_ext");
        proteus::pyarray<double>& q_velocity = args.array<double>("q_velocity");
        proteus::pyarray<double>& ebqe_velocity = args.array<double>("ebqe_velocity");
        proteus::pyarray<double>& flux = args.array<double>("flux");
        proteus::pyarray<double>& elementResidual_h_save = args.array<double>("elementResidual_h");
      //
      //loop over elements to compute volume integrals and load them into element and global residual
      //
      double globalConservationError=0.0,tauSum=0.0;
      for(proteus_index_t eN=0;eN<nElements_global;eN++)
        {
          //declare local storage for element residual and initialize
          double elementResidual_h[nDOF_test_element],
            elementResidual_u[nDOF_test_element],
            elementResidual_v[nDOF_test_element];
          for (int i=0;i<nDOF_test_element;i++)
            {
              proteus_index_t eN_i = eN*nDOF_test_element+i;
              elementResidual_h_save.data()[eN_i]=0.0;
              elementResidual_h[i]=0.0;
              elementResidual_u[i]=0.0;
              elementResidual_v[i]=0.0;
            }//i
          //
          //loop over quadrature points and compute integrands
          //
          for(int k=0;k<nQuadraturePoints_element;k++)
            {
              //compute indices and declare local storage
              proteus_index_t eN_k = eN*nQuadraturePoints_element+k,
                eN_k_nSpace = eN_k*nSpace,
                eN_nDOF_trial_element = eN*nDOF_trial_element;
              double b=0.0,h=0.0,u=0.0,v=0.0,h_sge=0.0,u_sge=0.0,v_sge=0.0,
                grad_b[nSpace],grad_h[nSpace],grad_u[nSpace],grad_v[nSpace],
                mass_acc=0.0,
                dmass_acc_h=0.0,
                mom_u_acc=0.0,
                dmom_u_acc_h=0.0,
                dmom_u_acc_u=0.0,
                mom_v_acc=0.0,
                dmom_v_acc_h=0.0,
                dmom_v_acc_v=0.0,
                mass_adv[nSpace],
                dmass_adv_h[nSpace],
                dmass_adv_u[nSpace],
                dmass_adv_v[nSpace],
                dmass_adv_h_sge[nSpace],
                dmass_adv_u_sge[nSpace],
                dmass_adv_v_sge[nSpace],
                mom_u_adv[nSpace],
                dmom_u_adv_h[nSpace],
                dmom_u_adv_u[nSpace],
                dmom_u_adv_v[nSpace],
                dmom_u_adv_h_sge[nSpace],
                dmom_u_adv_u_sge[nSpace],
                dmom_u_adv_v_sge[nSpace],
                mom_v_adv[nSpace],
                dmom_v_adv_h[nSpace],
                dmom_v_adv_u[nSpace],
                dmom_v_adv_v[nSpace],
                dmom_v_adv_h_sge[nSpace],
                dmom_v_adv_u_sge[nSpace],
                dmom_v_adv_v_sge[nSpace],
                mom_u_diff_ten[nSpace],
                mom_v_diff_ten[nSpace],
                mom_uv_diff_ten[1],
                mom_vu_diff_ten[1],
                mom_u_source=0.0,
                dmom_u_source_h=0.0,
                mom_v_source=0.0,
                dmom_v_source_h=0.0,
                mass_acc_t=0.0,
                dmass_acc_h_t=0.0,
                mom_u_acc_t=0.0,
                dmom_u_acc_h_t=0.0,
                dmom_u_acc_u_t=0.0,
                mom_v_acc_t=0.0,
                dmom_v_acc_h_t=0.0,
                dmom_v_acc_v_t=0.0,
                tau[9],
                pdeResidual_h=0.0,
                pdeResidual_u=0.0,
                pdeResidual_v=0.0,
                Lstar_h_h[nDOF_test_element],
                Lstar_u_h[nDOF_test_element],
                Lstar_v_h[nDOF_test_element],
                Lstar_h_u[nDOF_test_element],
                Lstar_u_u[nDOF_test_element],
                Lstar_v_u[nDOF_test_element],
                Lstar_h_v[nDOF_test_element],
                Lstar_u_v[nDOF_test_element],
                Lstar_v_v[nDOF_test_element],
                subgridError_h=0.0,
                subgridError_u=0.0,
                subgridError_v=0.0,
                jac[nSpace*nSpace],
                jacDet,
                jacInv[nSpace*nSpace],
                h_grad_trial[nDOF_trial_element*nSpace],vel_grad_trial[nDOF_trial_element*nSpace],
                h_test_dV[nDOF_trial_element],vel_test_dV[nDOF_trial_element],
                h_grad_test_dV[nDOF_test_element*nSpace],vel_grad_test_dV[nDOF_test_element*nSpace],
                dV,x,y,xt,yt,
                G[nSpace*nSpace],G_dd_G,tr_G,norm_Rv, dmom_adv_star[nSpace],dmom_adv_sge[nSpace];
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
                                          x,y);
              ck.calculateMappingVelocity_element(eN,
                                                  k,
                                                  mesh_velocity_dof.data(),
                                                  mesh_l2g.data(),
                                                  mesh_trial_ref.data(),
                                                  xt,yt);
              //xt=0.0;yt=0.0;
              //std::cout<<"xt "<<xt<<'\t'<<yt<<'\t'<<std::endl;
              //get the physical integration weight
              dV = fabs(jacDet)*dV_ref.data()[k];
              //get the trial function gradients
              ck.gradTrialFromRef(&h_grad_trial_ref.data()[k*nDOF_trial_element*nSpace],jacInv,h_grad_trial);
              ck.gradTrialFromRef(&vel_grad_trial_ref.data()[k*nDOF_trial_element*nSpace],jacInv,vel_grad_trial);
              //get the solution
              ck.valFromDOF(b_dof.data(),&h_l2g.data()[eN_nDOF_trial_element],&h_trial_ref.data()[k*nDOF_trial_element],b);
              ck.valFromDOF(h_dof.data(),&h_l2g.data()[eN_nDOF_trial_element],&h_trial_ref.data()[k*nDOF_trial_element],h);
              ck.valFromDOF(u_dof.data(),&vel_l2g.data()[eN_nDOF_trial_element],&vel_trial_ref.data()[k*nDOF_trial_element],u);
              ck.valFromDOF(v_dof.data(),&vel_l2g.data()[eN_nDOF_trial_element],&vel_trial_ref.data()[k*nDOF_trial_element],v);
              ck.valFromDOF(h_dof_sge.data(),&h_l2g.data()[eN_nDOF_trial_element],&h_trial_ref.data()[k*nDOF_trial_element],h_sge);
              ck.valFromDOF(u_dof_sge.data(),&vel_l2g.data()[eN_nDOF_trial_element],&vel_trial_ref.data()[k*nDOF_trial_element],u_sge);
              ck.valFromDOF(v_dof_sge.data(),&vel_l2g.data()[eN_nDOF_trial_element],&vel_trial_ref.data()[k*nDOF_trial_element],v_sge);
              //get the solution gradients
              ck.gradFromDOF(b_dof.data(),&h_l2g.data()[eN_nDOF_trial_element],h_grad_trial,grad_b);
              ck.gradFromDOF(h_dof.data(),&h_l2g.data()[eN_nDOF_trial_element],h_grad_trial,grad_h);
              ck.gradFromDOF(u_dof.data(),&vel_l2g.data()[eN_nDOF_trial_element],vel_grad_trial,grad_u);
              ck.gradFromDOF(v_dof.data(),&vel_l2g.data()[eN_nDOF_trial_element],vel_grad_trial,grad_v);
              //precalculate test function products with integration weights
              for (int j=0;j<nDOF_trial_element;j++)
                {
                  h_test_dV[j] = h_test_ref.data()[k*nDOF_trial_element+j]*dV;
                  vel_test_dV[j] = vel_test_ref.data()[k*nDOF_trial_element+j]*dV;
                  for (int I=0;I<nSpace;I++)
                    {
                      h_grad_test_dV[j*nSpace+I]   = h_grad_trial[j*nSpace+I]*dV;//cek warning won't work for Petrov-Galerkin
                      vel_grad_test_dV[j*nSpace+I] = vel_grad_trial[j*nSpace+I]*dV;//cek warning won't work for Petrov-Galerkin
                    }
                }
              //save velocity at quadrature points for other models to use
              q_velocity.data()[eN_k_nSpace+0]=u;
              q_velocity.data()[eN_k_nSpace+1]=v;
              //
              //calculate pde coefficients at quadrature points
              //
              evaluateCoefficients(nu,
                                   g,
                                   grad_b,
                                   h,
                                   u,
                                   v,
                                   mass_acc,
                                   dmass_acc_h,
                                   mom_u_acc,
                                   dmom_u_acc_h,
                                   dmom_u_acc_u,
                                   mom_v_acc,
                                   dmom_v_acc_h,
                                   dmom_v_acc_v,
                                   mass_adv,
                                   dmass_adv_h,
                                   dmass_adv_u,
                                   dmass_adv_v,
                                   mom_u_adv,
                                   dmom_u_adv_h,
                                   dmom_u_adv_u,
                                   dmom_u_adv_v,
                                   mom_v_adv,
                                   dmom_v_adv_h,
                                   dmom_v_adv_u,
                                   dmom_v_adv_v,
                                   mom_u_diff_ten,
                                   mom_v_diff_ten,
                                   mom_uv_diff_ten,
                                   mom_vu_diff_ten,
                                   mom_u_source,
                                   dmom_u_source_h,
                                   mom_v_source,
                                   dmom_v_source_h);
              //
              //save momentum for time history and velocity for subgrid error
              //
              q_mass_acc.data()[eN_k] = mass_acc;
              q_mom_u_acc.data()[eN_k] = mom_u_acc;
              q_mom_v_acc.data()[eN_k] = mom_v_acc;
              //subgrid error uses grid scale discharge
              q_mass_adv.data()[eN_k_nSpace+0] = h*u;
              q_mass_adv.data()[eN_k_nSpace+1] = h*v;
              //
              //moving mesh
              //
              //transform the continuity equation as if the accumulation term was  d(1)/dt
              /* mass_adv[0] -= MOVING_DOMAIN*mass_acc*xt; */
              /* mass_adv[1] -= MOVING_DOMAIN*mass_acc*yt; */
              /* dmass_adv_h[0] -= MOVING_DOMAIN*dmass_acc_h*xt; */
              /* dmass_adv_h[1] -= MOVING_DOMAIN*dmass_acc_h*yt; */

              /* mom_u_adv[0] -= MOVING_DOMAIN*mom_u_acc*xt; */
              /* mom_u_adv[1] -= MOVING_DOMAIN*mom_u_acc*yt; */
              /* dmom_u_adv_u[0] -= MOVING_DOMAIN*dmom_u_acc_u*xt; */
              /* dmom_u_adv_u[1] -= MOVING_DOMAIN*dmom_u_acc_u*yt; */

              /* mom_v_adv[0] -= MOVING_DOMAIN*mom_v_acc*xt; */
              /* mom_v_adv[1] -= MOVING_DOMAIN*mom_v_acc*yt; */
              /* dmom_v_adv_v[0] -= MOVING_DOMAIN*dmom_v_acc_v*xt; */
              /* dmom_v_adv_v[1] -= MOVING_DOMAIN*dmom_v_acc_v*yt; */

              //
              //calculate time derivative at quadrature points
              //
              ck.bdf(alphaBDF,
                     q_mass_acc_beta_bdf.data()[eN_k],
                     mass_acc,
                     dmass_acc_h,
                     mass_acc_t,
                     dmass_acc_h_t);
              ck.bdfC2(alphaBDF,
                     q_mom_u_acc_beta_bdf.data()[eN_k],
                     mom_u_acc,
                     dmom_u_acc_h,
                     dmom_u_acc_u,
                     mom_u_acc_t,
                     dmom_u_acc_h_t,
                     dmom_u_acc_u_t);
              ck.bdfC2(alphaBDF,
                     q_mom_v_acc_beta_bdf.data()[eN_k],
                     mom_v_acc,
                     dmom_v_acc_h,
                     dmom_v_acc_v,
                     mom_v_acc_t,
                     dmom_v_acc_h_t,
                     dmom_v_acc_v_t);
              //
              //calculate subgrid error (strong residual and adjoint)
              //
              //calculate strong residual
              dmass_adv_h_sge[0]  = dmass_adv_h[0];
              dmass_adv_h_sge[1]  = dmass_adv_h[1];
              dmass_adv_u_sge[0]  = dmass_adv_u[0];
              dmass_adv_u_sge[1]  = dmass_adv_u[1];
              dmass_adv_v_sge[0]  = dmass_adv_v[0];
              dmass_adv_v_sge[1]  = dmass_adv_v[1];
              dmom_u_adv_h_sge[0] = dmom_u_adv_h[0];
              dmom_u_adv_h_sge[1] = dmom_u_adv_h[1];
              dmom_u_adv_u_sge[0] = dmom_u_adv_u[0];
              dmom_u_adv_u_sge[1] = dmom_u_adv_u[1];
              dmom_u_adv_v_sge[0] = dmom_u_adv_v[0];
              dmom_u_adv_v_sge[1] = dmom_u_adv_v[1];
              dmom_v_adv_h_sge[0] = dmom_v_adv_h[0];
              dmom_v_adv_h_sge[1] = dmom_v_adv_h[1];
              dmom_v_adv_u_sge[0] = dmom_v_adv_u[0];
              dmom_v_adv_u_sge[1] = dmom_v_adv_u[1];
              dmom_v_adv_v_sge[0] = dmom_v_adv_v[0];
              dmom_v_adv_v_sge[1] = dmom_v_adv_v[1];

              /* //mass advective flux */
              /* dmass_adv_h_sge[0]=u_sge; */
              /* dmass_adv_h_sge[1]=v_sge; */

              /* dmass_adv_u_sge[0]=0.0;//h_sge; */
              /* dmass_adv_u_sge[1]=0.0; */

              /* dmass_adv_v_sge[0]=0.0; */
              /* dmass_adv_v_sge[1]=0.0;//h_sge; */

              /* //u momentum advective flux */
              /* dmom_u_adv_h_sge[0]=0.0;//u_sge*u_sge + g*h_sge; */
              /* dmom_u_adv_h_sge[1]=0.0;//u_sge*v_sge; */

              /* dmom_u_adv_u_sge[0]=h_sge*u_sge;//h_sge*2.0*u_sge; */
              /* dmom_u_adv_u_sge[1]=h_sge*v_sge; */

              /* dmom_u_adv_v_sge[0]=0.0; */
              /* dmom_u_adv_v_sge[1]=0.0;//h_sge*u_sge; */

              /* //v momentum advective_flux */
              /* dmom_v_adv_h_sge[0]=0.0;//v_sge*u_sge; */
              /* dmom_v_adv_h_sge[1]=0.0;//v_sge*v_sge + g*h_sge; */

              /* dmom_v_adv_u_sge[0]=0.0;//h_sge*v_sge; */
              /* dmom_v_adv_u_sge[1]=0.0; */

              /* dmom_v_adv_v_sge[0]=h_sge*u_sge; */
              /* dmom_v_adv_v_sge[1]=h_sge*v_sge;//h_sge*2.0*v_sge; */


              //full linearization, lagged

              //mass advective flux
              dmass_adv_h_sge[0]=u_sge;
              dmass_adv_h_sge[1]=v_sge;

              dmass_adv_u_sge[0]=h_sge;
              dmass_adv_u_sge[1]=0.0;

              dmass_adv_v_sge[0]=0.0;
              dmass_adv_v_sge[1]=h_sge;

              //u momentum advective flux
              dmom_u_adv_h_sge[0]=u_sge*u_sge + g*h_sge;
              dmom_u_adv_h_sge[1]=u_sge*v_sge;

              dmom_u_adv_u_sge[0]=h_sge*2.0*u_sge;
              dmom_u_adv_u_sge[1]=h_sge*v_sge;

              dmom_u_adv_v_sge[0]=0.0;
              dmom_u_adv_v_sge[1]=h_sge*u_sge;

              //v momentum advective_flux
              dmom_v_adv_h_sge[0]=v_sge*u_sge;
              dmom_v_adv_h_sge[1]=v_sge*v_sge + g*h_sge;

              dmom_v_adv_u_sge[0]=h_sge*v_sge;
              dmom_v_adv_u_sge[1]=0.0;

              dmom_v_adv_v_sge[0]=h_sge*u_sge;
              dmom_v_adv_v_sge[1]=h_sge*2.0*v_sge;

              pdeResidual_h = ck.Mass_strong(mass_acc_t) +
                ck.Advection_strong(dmass_adv_h_sge,grad_h) +
                ck.Advection_strong(dmass_adv_u_sge,grad_u) +
                ck.Advection_strong(dmass_adv_v_sge,grad_v);

              pdeResidual_u = ck.Mass_strong(mom_u_acc_t) +
                ck.Advection_strong(dmom_u_adv_h_sge,grad_h) +
                ck.Advection_strong(dmom_u_adv_u_sge,grad_u) +
                ck.Advection_strong(dmom_u_adv_v_sge,grad_v) +
                ck.Reaction_strong(mom_u_source);

              pdeResidual_v = ck.Mass_strong(mom_v_acc_t) +
                ck.Advection_strong(dmom_v_adv_h_sge,grad_h) +
                ck.Advection_strong(dmom_v_adv_u_sge,grad_u) +
                ck.Advection_strong(dmom_v_adv_v_sge,grad_v) +
                ck.Reaction_strong(mom_v_source);

              calculateSubgridError_tau(elementDiameter.data()[eN],
                                        nu,
                                        g,
                                        h_sge,
                                        u_sge,
                                        v_sge,
                                        tau,
                                        q_cfl.data()[eN_k]);
              for (int i=0;i<9;i++)
                tauSum += tau[i];

              subgridError_h = - tau[0*3+0]*pdeResidual_h - tau[0*3+1]*pdeResidual_u - tau[0*3+2]*pdeResidual_v;
              subgridError_u = - tau[1*3+0]*pdeResidual_h - tau[1*3+1]*pdeResidual_u - tau[1*3+2]*pdeResidual_v;
              subgridError_v = - tau[2*3+0]*pdeResidual_h - tau[2*3+1]*pdeResidual_u - tau[2*3+2]*pdeResidual_v;

              //adjoint times the test functions
              for (int i=0;i<nDOF_test_element;i++)
                {
                  int i_nSpace = i*nSpace;
                  Lstar_h_h[i]=ck.Advection_adjoint(dmass_adv_h_sge,&h_grad_test_dV[i_nSpace]);
                  Lstar_u_h[i]=ck.Advection_adjoint(dmass_adv_u_sge,&h_grad_test_dV[i_nSpace]);
                  Lstar_v_h[i]=ck.Advection_adjoint(dmass_adv_v_sge,&h_grad_test_dV[i_nSpace]);

                  Lstar_h_u[i]=ck.Advection_adjoint(dmom_u_adv_h_sge,&vel_grad_test_dV[i_nSpace])  +
                    ck.Reaction_adjoint(dmom_u_source_h,vel_test_dV[i]);
                  Lstar_u_u[i]=ck.Advection_adjoint(dmom_u_adv_u_sge,&vel_grad_test_dV[i_nSpace]);
                  Lstar_v_u[i]=ck.Advection_adjoint(dmom_u_adv_v_sge,&vel_grad_test_dV[i_nSpace]);

                  Lstar_h_v[i]=ck.Advection_adjoint(dmom_v_adv_h_sge,&vel_grad_test_dV[i_nSpace]) +
                    ck.Reaction_adjoint(dmom_v_source_h,vel_test_dV[i]);
                  Lstar_u_v[i]=ck.Advection_adjoint(dmom_v_adv_u_sge,&vel_grad_test_dV[i_nSpace]);
                  Lstar_v_v[i]=ck.Advection_adjoint(dmom_v_adv_v_sge,&vel_grad_test_dV[i_nSpace]);

                }

              norm_Rv = sqrt(pdeResidual_u*pdeResidual_u + pdeResidual_v*pdeResidual_v);
              /* double */
              double norm_grad = 1.0;
              q_numDiff_u.data()[eN_k] = 0.5*elementDiameter.data()[eN]*norm_Rv/(norm_grad+1.0e-8);
              q_numDiff_v.data()[eN_k] = q_numDiff_u.data()[eN_k];

              /* ck.calculateNumericalDiffusion(1.0, */
              /*                                     elementDiameter.data()[eN], */
              /*                                     pdeResidual_h, */
              /*                                     grad_h, */
              /*                                     q_numDiff_h.data()[eN_k]); */

              //update element residual

              for(int i=0;i<nDOF_test_element;i++)
                {
                  int i_nSpace=i*nSpace;

                  elementResidual_h[i] += ck.Mass_weak(mass_acc_t,h_test_dV[i]) +
                    ck.Advection_weak(mass_adv,&h_grad_test_dV[i_nSpace]) +
                    ck.SubgridError(subgridError_h,Lstar_h_h[i]) +
                    ck.SubgridError(subgridError_u,Lstar_u_h[i]) +
                    ck.SubgridError(subgridError_v,Lstar_v_h[i]) +
                    ck.NumericalDiffusion(q_numDiff_h_last.data()[eN_k],grad_h,&h_grad_test_dV[i_nSpace]);

                  elementResidual_u[i] += ck.Mass_weak(mom_u_acc_t,vel_test_dV[i]) +
                    ck.Advection_weak(mom_u_adv,&vel_grad_test_dV[i_nSpace]) +
                    ck.Diffusion_weak(sdInfo_u_u_rowptr.data(),sdInfo_u_u_colind.data(),mom_u_diff_ten,grad_u,&vel_grad_test_dV[i_nSpace]) +
                    ck.Diffusion_weak(sdInfo_u_v_rowptr.data(),sdInfo_u_v_colind.data(),mom_uv_diff_ten,grad_v,&vel_grad_test_dV[i_nSpace]) +
                    ck.Reaction_weak(mom_u_source,vel_test_dV[i]) +
                    ck.SubgridError(subgridError_h,Lstar_h_u[i]) +
                    ck.SubgridError(subgridError_u,Lstar_u_u[i]) +
                    ck.SubgridError(subgridError_v,Lstar_v_u[i]) +
                    ck.NumericalDiffusion(q_numDiff_u_last.data()[eN_k],grad_u,&vel_grad_test_dV[i_nSpace]);

                  elementResidual_v[i] += ck.Mass_weak(mom_v_acc_t,vel_test_dV[i]) +
                    ck.Advection_weak(mom_v_adv,&vel_grad_test_dV[i_nSpace]) +
                    ck.Diffusion_weak(sdInfo_v_v_rowptr.data(),sdInfo_v_v_colind.data(),mom_v_diff_ten,grad_v,&vel_grad_test_dV[i_nSpace]) +
                    ck.Diffusion_weak(sdInfo_v_u_rowptr.data(),sdInfo_v_u_colind.data(),mom_vu_diff_ten,grad_u,&vel_grad_test_dV[i_nSpace]) +
                    ck.Reaction_weak(mom_v_source,vel_test_dV[i]) +
                    ck.SubgridError(subgridError_h,Lstar_h_v[i]) +
                    ck.SubgridError(subgridError_u,Lstar_u_v[i]) +
                    ck.SubgridError(subgridError_v,Lstar_v_v[i]) +
                    ck.NumericalDiffusion(q_numDiff_v_last.data()[eN_k],grad_v,&vel_grad_test_dV[i_nSpace]);
                }
            }

          //load element into global residual and save element residual

          for(int i=0;i<nDOF_test_element;i++)
            {
              proteus_index_t eN_i=eN*nDOF_test_element+i;

              elementResidual_h_save.data()[eN_i] +=  elementResidual_h[i];

              globalResidual.data()[offset_h+stride_h*h_l2g.data()[eN_i]]+=elementResidual_h[i];
              globalResidual.data()[offset_u+stride_u*vel_l2g.data()[eN_i]]+=elementResidual_u[i];
              globalResidual.data()[offset_v+stride_v*vel_l2g.data()[eN_i]]+=elementResidual_v[i];
            }
        }
      std::cout<<"tauSum = "<<tauSum<<std::endl;
      /* */
      /* loop over exterior element boundaries to calculate surface integrals and load into element and global residuals */
      /* */
      /* ebNE is the Exterior element boundary INdex */
      /* ebN is the element boundary INdex */
      /* eN is the element index */
      /* for (proteus_index_t ebNE = 0; ebNE < nExteriorElementBoundaries_global; ebNE++)  */
      /*        {  */
      /*          proteus_index_t ebN = exteriorElementBoundariesArray.data()[ebNE],  */
      /*            eN  = elementBoundaryElementsArray.data()[ebN*2+0], */
      /*            ebN_local = elementBoundaryLocalElementBoundariesArray.data()[ebN*2+0], */
      /*            eN_nDOF_trial_element = eN*nDOF_trial_element; */
      /*          double elementResidual_h[nDOF_test_element], */
      /*            elementResidual_u[nDOF_test_element], */
      /*            elementResidual_v[nDOF_test_element], */
      /*            eps_rho,eps_mu; */
      /*          for (int i=0;i<nDOF_test_element;i++) */
      /*            { */
      /*              elementResidual_h[i]=0.0; */
      /*              elementResidual_u[i]=0.0; */
      /*              elementResidual_v[i]=0.0; */
      /*            } */
      /*          for  (int kb=0;kb<nQuadraturePoints_elementBoundary;kb++)  */
      /*            {  */
      /*              int ebNE_kb = ebNE*nQuadraturePoints_elementBoundary+kb, */
      /*                ebNE_kb_nSpace = ebNE_kb*nSpace, */
      /*                ebN_local_kb = ebN_local*nQuadraturePoints_elementBoundary+kb, */
      /*                ebN_local_kb_nSpace = ebN_local_kb*nSpace; */
      /*              double h_ext=0.0, */
      /*                u_ext=0.0, */
      /*                v_ext=0.0, */
      /*                grad_h_ext[nSpace], */
      /*                grad_u_ext[nSpace], */
      /*                grad_v_ext[nSpace], */
      /*                mom_u_acc_ext=0.0, */
      /*                dmom_u_acc_u_ext=0.0, */
      /*                mom_v_acc_ext=0.0, */
      /*                dmom_v_acc_v_ext=0.0, */
      /*                mass_adv_ext[nSpace], */
      /*                dmass_adv_u_ext[nSpace], */
      /*                dmass_adv_v_ext[nSpace], */
      /*                mom_u_adv_ext[nSpace], */
      /*                dmom_u_adv_u_ext[nSpace], */
      /*                dmom_u_adv_v_ext[nSpace], */
      /*                mom_v_adv_ext[nSpace], */
      /*                dmom_v_adv_u_ext[nSpace], */
      /*                dmom_v_adv_v_ext[nSpace], */
      /*                mom_u_diff_ten_ext[nSpace], */
      /*                mom_v_diff_ten_ext[nSpace], */
      /*                mom_uv_diff_ten_ext[1], */
      /*                mom_vu_diff_ten_ext[1], */
      /*                mom_u_source_ext=0.0, */
      /*                mom_v_source_ext=0.0, */
      /*                mom_u_ham_ext=0.0, */
      /*                dmom_u_ham_grad_h_ext[nSpace], */
      /*                mom_v_ham_ext=0.0, */
      /*                dmom_v_ham_grad_h_ext[nSpace], */
      /*                dmom_u_adv_h_ext[nSpace], */
      /*                dmom_v_adv_h_ext[nSpace], */
      /*                flux_mass_ext=0.0, */
      /*                flux_mom_u_adv_ext=0.0, */
      /*                flux_mom_v_adv_ext=0.0, */
      /*                flux_mom_u_diff_ext=0.0, */
      /*                flux_mom_v_diff_ext=0.0, */
      /*                bc_h_ext=0.0, */
      /*                bc_u_ext=0.0, */
      /*                bc_v_ext=0.0, */
      /*                bc_mom_u_acc_ext=0.0, */
      /*                bc_dmom_u_acc_u_ext=0.0, */
      /*                bc_mom_v_acc_ext=0.0, */
      /*                bc_dmom_v_acc_v_ext=0.0, */
      /*                bc_mass_adv_ext[nSpace], */
      /*                bc_dmass_adv_u_ext[nSpace], */
      /*                bc_dmass_adv_v_ext[nSpace], */
      /*                bc_mom_u_adv_ext[nSpace], */
      /*                bc_dmom_u_adv_u_ext[nSpace], */
      /*                bc_dmom_u_adv_v_ext[nSpace], */
      /*                bc_mom_v_adv_ext[nSpace], */
      /*                bc_dmom_v_adv_u_ext[nSpace], */
      /*                bc_dmom_v_adv_v_ext[nSpace], */
      /*                bc_mom_u_diff_ten_ext[nSpace], */
      /*                bc_mom_v_diff_ten_ext[nSpace], */
      /*                bc_mom_uv_diff_ten_ext[1], */
      /*                bc_mom_vu_diff_ten_ext[1], */
      /*                bc_mom_u_source_ext=0.0, */
      /*                bc_mom_v_source_ext=0.0, */
      /*                bc_mom_u_ham_ext=0.0, */
      /*                bc_dmom_u_ham_grad_h_ext[nSpace], */
      /*                bc_mom_v_ham_ext=0.0, */
      /*                bc_dmom_v_ham_grad_h_ext[nSpace], */
      /*                jac_ext[nSpace*nSpace], */
      /*                jacDet_ext, */
      /*                jacInv_ext[nSpace*nSpace], */
      /*                boundaryJac[nSpace*(nSpace-1)], */
      /*                metricTensor[(nSpace-1)*(nSpace-1)], */
      /*                metricTensorDetSqrt, */
      /*                dS,h_test_dS[nDOF_test_element],vel_test_dS[nDOF_test_element], */
      /*                h_grad_trial_trace[nDOF_trial_element*nSpace],vel_grad_trial_trace[nDOF_trial_element*nSpace], */
      /*                normal[3],x_ext,y_ext,xt_ext,yt_ext,integralScaling, */
      /*                G[nSpace*nSpace],G_dd_G,tr_G,h_penalty; */
      /*              compute information about mapping from reference element to physical element */
      /*              ck.calculateMapping_elementBoundary(eN, */
      /*                                                  ebN_local, */
      /*                                                  kb, */
      /*                                                  ebN_local_kb, */
      /*                                                  mesh_dof.data(), */
      /*                                                  mesh_l2g.data(), */
      /*                                                  mesh_trial_trace_ref.data(), */
      /*                                                  mesh_grad_trial_trace_ref.data(), */
      /*                                                  boundaryJac_ref.data(), */
      /*                                                  jac_ext, */
      /*                                                  jacDet_ext, */
      /*                                                  jacInv_ext, */
      /*                                                  boundaryJac, */
      /*                                                  metricTensor, */
      /*                                                  metricTensorDetSqrt, */
      /*                                                  normal_ref.data(), */
      /*                                                  normal, */
      /*                                                  x_ext,y_ext); */
      /*              ck.calculateMappingVelocity_elementBoundary(eN, */
      /*                                                          ebN_local, */
      /*                                                          kb, */
      /*                                                          ebN_local_kb, */
      /*                                                          mesh_velocity_dof.data(), */
      /*                                                          mesh_l2g.data(), */
      /*                                                          mesh_trial_trace_ref.data(), */
      /*                                                          xt_ext,yt_ext, */
      /*                                                          normal, */
      /*                                                          boundaryJac, */
      /*                                                          metricTensor, */
      /*                                                          integralScaling); */
      /*              xt_ext=0.0;yt_ext=0.0; */
      /*              std::cout<<"xt_ext "<<xt_ext<<'\t'<<yt_ext<<'\t'<<std::endl; */
      /*              std::cout<<"integralScaling - metricTensorDetSrt ==============================="<<integralScaling-metricTensorDetSqrt<<std::endl; */
      /*              dS = ((1.0-MOVING_DOMAIN)*metricTensorDetSqrt + MOVING_DOMAIN*integralScaling)*dS_ref.data()[kb]; */
      /*              compute shape and solution information */
      /*              shape */
      /*              ck.gradTrialFromRef(&h_grad_trial_trace_ref.data()[ebN_local_kb_nSpace*nDOF_trial_element],jacInv_ext,h_grad_trial_trace); */
      /*              ck.gradTrialFromRef(&vel_grad_trial_trace_ref.data()[ebN_local_kb_nSpace*nDOF_trial_element],jacInv_ext,vel_grad_trial_trace); */
      /*              solution and gradients     */
      /*              ck.valFromDOF(h_dof.data(),&h_l2g.data()[eN_nDOF_trial_element],&h_trial_trace_ref.data()[ebN_local_kb*nDOF_test_element],h_ext); */
      /*              ck.valFromDOF(u_dof.data(),&vel_l2g.data()[eN_nDOF_trial_element],&vel_trial_trace_ref.data()[ebN_local_kb*nDOF_test_element],u_ext); */
      /*              ck.valFromDOF(v_dof.data(),&vel_l2g.data()[eN_nDOF_trial_element],&vel_trial_trace_ref.data()[ebN_local_kb*nDOF_test_element],v_ext); */
      /*              ck.gradFromDOF(h_dof.data(),&h_l2g.data()[eN_nDOF_trial_element],h_grad_trial_trace,grad_h_ext); */
      /*              ck.gradFromDOF(u_dof.data(),&vel_l2g.data()[eN_nDOF_trial_element],vel_grad_trial_trace,grad_u_ext); */
      /*              ck.gradFromDOF(v_dof.data(),&vel_l2g.data()[eN_nDOF_trial_element],vel_grad_trial_trace,grad_v_ext); */
      /*              precalculate test function products with integration weights */
      /*              for (int j=0;j<nDOF_trial_element;j++) */
      /*                { */
      /*                  h_test_dS[j] = h_test_trace_ref.data()[ebN_local_kb*nDOF_test_element+j]*dS; */
      /*                  vel_test_dS[j] = vel_test_trace_ref.data()[ebN_local_kb*nDOF_test_element+j]*dS; */
      /*                } */
      /*              // */
      /*              //debugging section for finite element calculations on exterior */
      /*              // */
      /*              std::cout<<"ebNE = "<<ebNE<<" kb = "<<kb<<std::endl; */
      /*              for (int j=0;j<nDOF_trial_element;j++) */
      /*                { */
      /*                  std::cout<<"h_trial_trace["<<j<<"] "<<h_trial_trace_ref.data()[ebN_local_kb*nDOF_trial_element+j]<<std::endl; */
      /*                  std::cout<<"vel_trial_trace["<<j<<"] "<<vel_trial_trace_ref.data()[ebN_local_kb*nDOF_trial_element+j]<<std::endl; */
      /*                  std::cout<<"h_test_dS["<<j<<"] "<<h_test_dS[j]<<std::endl; */
      /*                  std::cout<<"vel_test_dS["<<j<<"] "<<vel_test_dS[j]<<std::endl; */
      /*                  for (int I=0;I<nSpace;I++) */
      /*                { */
      /*                  std::cout<<"h_grad_trial_trace["<<j<<","<<I<<"] "<<h_grad_trial_trace[j*nSpace+I]<<std::endl; */
      /*                  std::cout<<"vel_grad_trial_trace["<<j<<","<<I<<"] "<<vel_grad_trial_trace[j*nSpace+I]<<std::endl; */
      /*                } */
      /*                } */
      /*              std::cout<<"h_ext "<<h_ext<<std::endl; */
      /*              std::cout<<"u_ext "<<u_ext<<std::endl; */
      /*              std::cout<<"v_ext "<<v_ext<<std::endl; */
      /*              for(int I=0;I<nSpace;I++) */
      /*                { */
      /*                  std::cout<<"grad_h_ext["<<I<<"] "<<grad_h_ext[I]<<std::endl; */
      /*                  std::cout<<"grad_u_ext["<<I<<"] "<<grad_u_ext[I]<<std::endl; */
      /*                  std::cout<<"grad_v_ext["<<I<<"] "<<grad_v_ext[I]<<std::endl; */
      /*                } */
      /*              */
      /*              load the boundary values */
      /*              */
      /*              bc_h_ext = isDOFBoundary_h.data()[ebNE_kb]*ebqe_bc_h_ext.data()[ebNE_kb]+(1-isDOFBoundary_h.data()[ebNE_kb])*h_ext; */
      /*              bc_u_ext = isDOFBoundary_u.data()[ebNE_kb]*ebqe_bc_u_ext.data()[ebNE_kb]+(1-isDOFBoundary_u.data()[ebNE_kb])*u_ext; */
      /*              bc_v_ext = isDOFBoundary_v.data()[ebNE_kb]*ebqe_bc_v_ext.data()[ebNE_kb]+(1-isDOFBoundary_v.data()[ebNE_kb])*v_ext; */
      /*               */
      /*              calculate the pde coefficients using the solution and the boundary values for the solution  */
      /*               */
      /*              cek debug */
      /*              eps_rho=0.1; */
      /*              eps_mu=0.1; */
      /*              evaluateCoefficients(eps_rho, */
      /*                                   eps_mu, */
      /*                                   sigma, */
      /*                                   rho_0, */
      /*                                   nu_0, */
      /*                                   rho_1, */
      /*                                   nu_1, */
      /*                                   g, */
      /*                                   h_ext, */
      /*                                   grad_h_ext, */
      /*                                   u_ext, */
      /*                                   v_ext, */
      /*                                   mom_u_acc_ext, */
      /*                                   dmom_u_acc_u_ext, */
      /*                                   mom_v_acc_ext, */
      /*                                   dmom_v_acc_v_ext, */
      /*                                   mass_adv_ext, */
      /*                                   dmass_adv_u_ext, */
      /*                                   dmass_adv_v_ext, */
      /*                                   mom_u_adv_ext, */
      /*                                   dmom_u_adv_u_ext, */
      /*                                   dmom_u_adv_v_ext, */
      /*                                   mom_v_adv_ext, */
      /*                                   dmom_v_adv_u_ext, */
      /*                                   dmom_v_adv_v_ext, */
      /*                                   mom_u_diff_ten_ext, */
      /*                                   mom_v_diff_ten_ext, */
      /*                                   mom_uv_diff_ten_ext, */
      /*                                   mom_vu_diff_ten_ext, */
      /*                                   mom_u_source_ext, */
      /*                                   mom_v_source_ext, */
      /*                                   mom_u_ham_ext, */
      /*                                   dmom_u_ham_grad_h_ext, */
      /*                                   mom_v_ham_ext, */
      /*                                   dmom_v_ham_grad_h_ext);           */
      /*              evaluateCoefficients(eps_rho, */
      /*                                   eps_mu, */
      /*                                   sigma, */
      /*                                   rho_0, */
      /*                                   nu_0, */
      /*                                   rho_1, */
      /*                                   nu_1, */
      /*                                   g, */
      /*                                   bc_h_ext, */
      /*                                   grad_h_ext,cek should't be used */
      /*                                   bc_u_ext, */
      /*                                   bc_v_ext, */
      /*                                   bc_mom_u_acc_ext, */
      /*                                   bc_dmom_u_acc_u_ext, */
      /*                                   bc_mom_v_acc_ext, */
      /*                                   bc_dmom_v_acc_v_ext, */
      /*                                   bc_mass_adv_ext, */
      /*                                   bc_dmass_adv_u_ext, */
      /*                                   bc_dmass_adv_v_ext, */
      /*                                   bc_mom_u_adv_ext, */
      /*                                   bc_dmom_u_adv_u_ext, */
      /*                                   bc_dmom_u_adv_v_ext, */
      /*                                   bc_mom_v_adv_ext, */
      /*                                   bc_dmom_v_adv_u_ext, */
      /*                                   bc_dmom_v_adv_v_ext, */
      /*                                   bc_mom_u_diff_ten_ext, */
      /*                                   bc_mom_v_diff_ten_ext, */
      /*                                   bc_mom_uv_diff_ten_ext, */
      /*                                   bc_mom_vu_diff_ten_ext, */
      /*                                   bc_mom_u_source_ext, */
      /*                                   bc_mom_v_source_ext, */
      /*                                   bc_mom_u_ham_ext, */
      /*                                   bc_dmom_u_ham_grad_h_ext, */
      /*                                   bc_mom_v_ham_ext, */
      /*                                   bc_dmom_v_ham_grad_h_ext);           */
      /*              */
      /*              moving domain */
      /*              */
      /*              mass_adv_ext[0] -= MOVING_DOMAIN*xt_ext; */
      /*              mass_adv_ext[1] -= MOVING_DOMAIN*yt_ext; */

      /*              mom_u_adv_ext[0] -= MOVING_DOMAIN*mom_u_acc_ext*xt_ext; */
      /*              mom_u_adv_ext[1] -= MOVING_DOMAIN*mom_u_acc_ext*yt_ext; */
      /*              dmom_u_adv_u_ext[0] -= MOVING_DOMAIN*dmom_u_acc_u_ext*xt_ext; */
      /*              dmom_u_adv_u_ext[1] -= MOVING_DOMAIN*dmom_u_acc_u_ext*yt_ext; */

      /*              mom_v_adv_ext[0] -= MOVING_DOMAIN*mom_v_acc_ext*xt_ext; */
      /*              mom_v_adv_ext[1] -= MOVING_DOMAIN*mom_v_acc_ext*yt_ext; */
      /*              dmom_v_adv_v_ext[0] -= MOVING_DOMAIN*dmom_v_acc_v_ext*xt_ext; */
      /*              dmom_v_adv_v_ext[1] -= MOVING_DOMAIN*dmom_v_acc_v_ext*yt_ext; */

      /*              bc's */
      /*              bc_mom_u_adv_ext[0] -= MOVING_DOMAIN*bc_mom_u_acc_ext*xt_ext; */
      /*              bc_mom_u_adv_ext[1] -= MOVING_DOMAIN*bc_mom_u_acc_ext*yt_ext; */

      /*              bc_mom_v_adv_ext[0] -= MOVING_DOMAIN*bc_mom_v_acc_ext*xt_ext; */
      /*              bc_mom_v_adv_ext[1] -= MOVING_DOMAIN*bc_mom_v_acc_ext*yt_ext; */

      /*               */
      /*              calculate the numerical fluxes  */
      /*               */
      /*              cek debug */
      /*              ebqe_penalty_ext.data()[ebNE_kb] = 10.0; */
      /*              */
      /*              ck.calculateGScale(G,normal,h_penalty); */
      /*              h_penalty = 10.0/h_penalty; */
      /*              cek debug, do it the old way */
      /*              h_penalty = 100.0/elementDiameter.data()[eN]; */
      /*              exteriorNumericalAdvectiveFlux(isDOFBoundary_h.data()[ebNE_kb], */
      /*                                             isDOFBoundary_u.data()[ebNE_kb], */
      /*                                             isDOFBoundary_v.data()[ebNE_kb], */
      /*                                             isAdvectiveFluxBoundary_h.data()[ebNE_kb], */
      /*                                             isAdvectiveFluxBoundary_u.data()[ebNE_kb], */
      /*                                             isAdvectiveFluxBoundary_v.data()[ebNE_kb], */
      /*                                             dmom_u_ham_grad_h_ext[0],=1/rho, */
      /*                                             normal, */
      /*                                             bc_h_ext, */
      /*                                             bc_mass_adv_ext, */
      /*                                             bc_mom_u_adv_ext, */
      /*                                             bc_mom_v_adv_ext, */
      /*                                             ebqe_bc_flux_mass_ext.data()[ebNE_kb], */
      /*                                             ebqe_bc_flux_mom_u_adv_ext.data()[ebNE_kb], */
      /*                                             ebqe_bc_flux_mom_v_adv_ext.data()[ebNE_kb], */
      /*                                             h_ext, */
      /*                                             mass_adv_ext, */
      /*                                             mom_u_adv_ext, */
      /*                                             mom_v_adv_ext, */
      /*                                             dmass_adv_u_ext, */
      /*                                             dmass_adv_v_ext, */
      /*                                             dmom_u_adv_h_ext, */
      /*                                             dmom_u_adv_u_ext, */
      /*                                             dmom_u_adv_v_ext, */
      /*                                             dmom_v_adv_h_ext, */
      /*                                             dmom_v_adv_u_ext, */
      /*                                             dmom_v_adv_v_ext, */
      /*                                             flux_mass_ext, */
      /*                                             flux_mom_u_adv_ext, */
      /*                                             flux_mom_v_adv_ext, */
      /*                                             &ebqe_velocity.data()[ebNE_kb_nSpace]); */
      /*              cek todo need to switch to full stress and add adjoint consistency */
      /*              exteriorNumericalDiffusiveFlux(eps_rho, */
      /*                                             sdInfo_u_u_rowptr.data(), */
      /*                                             sdInfo_u_u_colind.data(), */
      /*                                             isDOFBoundary_u.data()[ebNE_kb], */
      /*                                             isDiffusiveFluxBoundary_u.data()[ebNE_kb], */
      /*                                             normal, */
      /*                                             bc_mom_u_diff_ten_ext, */
      /*                                             bc_u_ext, */
      /*                                             ebqe_bc_flux_u_diff_ext.data()[ebNE_kb], */
      /*                                             mom_u_diff_ten_ext, */
      /*                                             grad_u_ext, */
      /*                                             u_ext, */
      /*                                             h_penalty,ebqe_penalty_ext.data()[ebNE_kb], */
      /*                                             flux_mom_u_diff_ext); */
      /*              exteriorNumericalDiffusiveFlux(eps_rho, */
      /*                                             sdInfo_v_v_rowptr.data(), */
      /*                                             sdInfo_v_v_colind.data(), */
      /*                                             isDOFBoundary_v.data()[ebNE_kb], */
      /*                                             isDiffusiveFluxBoundary_v.data()[ebNE_kb], */
      /*                                             normal, */
      /*                                             bc_mom_v_diff_ten_ext, */
      /*                                             bc_v_ext, */
      /*                                             ebqe_bc_flux_v_diff_ext.data()[ebNE_kb], */
      /*                                             mom_v_diff_ten_ext, */
      /*                                             grad_v_ext, */
      /*                                             v_ext, */
      /*                                             h_penalty,ebqe_penalty_ext.data()[ebNE_kb], */
      /*                                             flux_mom_v_diff_ext); */
      /*              flux.data()[ebN*nQuadraturePoints_elementBoundary+kb] = flux_mass_ext; */
      /*              flux.data()[ebN*nQuadraturePoints_elementBoundary+kb] = 0.0;//cek debug */
      /*              */
      /*              update residuals */
      /*              */
      /*              for (int i=0;i<nDOF_test_element;i++) */
      /*                { */
      /*                  elementResidual_h[i] += ck.ExteriorElementBoundaryFlux(flux_mass_ext,h_test_dS[i]); */
      /*                  globalConservationError += ck.ExteriorElementBoundaryFlux(flux_mass_ext,h_test_dS[i]); */

      /*                  elementResidual_u[i] += ck.ExteriorElementBoundaryFlux(flux_mom_u_adv_ext,vel_test_dS[i])+ */
      /*                    ck.ExteriorElementBoundaryFlux(flux_mom_u_diff_ext,vel_test_dS[i]);  */

      /*                  elementResidual_v[i] += ck.ExteriorElementBoundaryFlux(flux_mom_v_adv_ext,vel_test_dS[i]) + */
      /*                    ck.ExteriorElementBoundaryFlux(flux_mom_v_diff_ext,vel_test_dS[i]);  */

      /*                }i */
      /*            }kb */
      /*          */
      /*          update the element and global residual storage */
      /*          */
      /*          for (int i=0;i<nDOF_test_element;i++) */
      /*            { */
      /*              proteus_index_t eN_i = eN*nDOF_test_element+i; */

      /*              elementResidual_h_save.data()[eN_i] +=  elementResidual_h[i]; */

      /*              globalResidual.data()[offset_h+stride_h*h_l2g.data()[eN_i]]+=elementResidual_h[i]; */
      /*              globalResidual.data()[offset_u+stride_u*vel_l2g.data()[eN_i]]+=elementResidual_u[i]; */
      /*              globalResidual.data()[offset_v+stride_v*vel_l2g.data()[eN_i]]+=elementResidual_v[i]; */
      /*            }i */
      /*          // */
      /*          //debug */
      /*          // */
      /*          for(int i=0;i<nDOF_test_element;i++)  */
      /*            {  */
      /*                  std::cout<<"ebNE "<<ebNE<<" i "<<i<<std::endl; */
      /*                  std::cout<<"r_h"<<elementResidual_h[i]<<std::endl; */
      /*                  std::cout<<"r_u"<<elementResidual_u[i]<<std::endl; */
      /*                  std::cout<<"r_v"<<elementResidual_v[i]<<std::endl; */
      /*                } */

      /*        }ebNE */
    }

    void calculateResidual_supg(arguments_dict& args)
    {
        proteus::pyarray<double>& mesh_trial_ref = args.array<double>("mesh_trial_ref");
        proteus::pyarray<double>& mesh_grad_trial_ref = args.array<double>("mesh_grad_trial_ref");
        proteus::pyarray<double>& mesh_dof = args.array<double>("mesh_dof");
        proteus::pyarray<double>& mesh_velocity_dof = args.array<double>("mesh_velocity_dof");
        double MOVING_DOMAIN = args.scalar<double>("MOVING_DOMAIN");
        proteus::pyarray<proteus_index_t>& mesh_l2g = args.array<proteus_index_t>("mesh_l2g");
        proteus::pyarray<double>& dV_ref = args.array<double>("dV_ref");
        proteus::pyarray<double>& h_trial_ref = args.array<double>("h_trial_ref");
        proteus::pyarray<double>& h_grad_trial_ref = args.array<double>("h_grad_trial_ref");
        proteus::pyarray<double>& h_test_ref = args.array<double>("h_test_ref");
        proteus::pyarray<double>& h_grad_test_ref = args.array<double>("h_grad_test_ref");
        proteus::pyarray<double>& vel_trial_ref = args.array<double>("vel_trial_ref");
        proteus::pyarray<double>& vel_grad_trial_ref = args.array<double>("vel_grad_trial_ref");
        proteus::pyarray<double>& vel_test_ref = args.array<double>("vel_test_ref");
        proteus::pyarray<double>& vel_grad_test_ref = args.array<double>("vel_grad_test_ref");
        proteus::pyarray<double>& mesh_trial_trace_ref = args.array<double>("mesh_trial_trace_ref");
        proteus::pyarray<double>& mesh_grad_trial_trace_ref = args.array<double>("mesh_grad_trial_trace_ref");
        proteus::pyarray<double>& dS_ref = args.array<double>("dS_ref");
        proteus::pyarray<double>& h_trial_trace_ref = args.array<double>("h_trial_trace_ref");
        proteus::pyarray<double>& h_grad_trial_trace_ref = args.array<double>("h_grad_trial_trace_ref");
        proteus::pyarray<double>& h_test_trace_ref = args.array<double>("h_test_trace_ref");
        proteus::pyarray<double>& h_grad_test_trace_ref = args.array<double>("h_grad_test_trace_ref");
        proteus::pyarray<double>& vel_trial_trace_ref = args.array<double>("vel_trial_trace_ref");
        proteus::pyarray<double>& vel_grad_trial_trace_ref = args.array<double>("vel_grad_trial_trace_ref");
        proteus::pyarray<double>& vel_test_trace_ref = args.array<double>("vel_test_trace_ref");
        proteus::pyarray<double>& vel_grad_test_trace_ref = args.array<double>("vel_grad_test_trace_ref");
        proteus::pyarray<double>& normal_ref = args.array<double>("normal_ref");
        proteus::pyarray<double>& boundaryJac_ref = args.array<double>("boundaryJac_ref");
        proteus::pyarray<double>& elementDiameter = args.array<double>("elementDiameter");
        proteus_index_t nElements_global = args.scalar<proteus_index_t>("nElements_global");
        double useRBLES = args.scalar<double>("useRBLES");
        double useMetrics = args.scalar<double>("useMetrics");
        double alphaBDF = args.scalar<double>("alphaBDF");
        double nu = args.scalar<double>("nu");
        double g = args.scalar<double>("g");
        double shockCapturingCoefficient = args.scalar<double>("shockCapturingCoefficient");
        proteus::pyarray<proteus_index_t>& h_l2g = args.array<proteus_index_t>("h_l2g");
        proteus::pyarray<proteus_index_t>& vel_l2g = args.array<proteus_index_t>("vel_l2g");
        proteus::pyarray<double>& b_dof = args.array<double>("b_dof");
        proteus::pyarray<double>& h_dof = args.array<double>("h_dof");
        proteus::pyarray<double>& u_dof = args.array<double>("u_dof");
        proteus::pyarray<double>& v_dof = args.array<double>("v_dof");
        proteus::pyarray<double>& h_dof_sge = args.array<double>("h_dof_sge");
        proteus::pyarray<double>& u_dof_sge = args.array<double>("u_dof_sge");
        proteus::pyarray<double>& v_dof_sge = args.array<double>("v_dof_sge");
        proteus::pyarray<double>& q_mass_acc = args.array<double>("q_mass_acc");
        proteus::pyarray<double>& q_mom_u_acc = args.array<double>("q_mom_u_acc");
        proteus::pyarray<double>& q_mom_v_acc = args.array<double>("q_mom_v_acc");
        proteus::pyarray<double>& q_mass_adv = args.array<double>("q_mass_adv");
        proteus::pyarray<double>& q_mass_acc_beta_bdf = args.array<double>("q_mass_acc_beta_bdf");
        proteus::pyarray<double>& q_mom_u_acc_beta_bdf = args.array<double>("q_mom_u_acc_beta_bdf");
        proteus::pyarray<double>& q_mom_v_acc_beta_bdf = args.array<double>("q_mom_v_acc_beta_bdf");
        proteus::pyarray<double>& q_velocity_sge = args.array<double>("q_velocity_sge");
        proteus::pyarray<double>& q_cfl = args.array<double>("q_cfl");
        proteus::pyarray<double>& q_numDiff_h = args.array<double>("q_numDiff_h");
        proteus::pyarray<double>& q_numDiff_u = args.array<double>("q_numDiff_u");
        proteus::pyarray<double>& q_numDiff_v = args.array<double>("q_numDiff_v");
        proteus::pyarray<double>& q_numDiff_h_last = args.array<double>("q_numDiff_h_last");
        proteus::pyarray<double>& q_numDiff_u_last = args.array<double>("q_numDiff_u_last");
        proteus::pyarray<double>& q_numDiff_v_last = args.array<double>("q_numDiff_v_last");
        proteus::pyarray<proteus_csr_index_t>& sdInfo_u_u_rowptr = args.array<proteus_csr_index_t>("sdInfo_u_u_rowptr");
        proteus::pyarray<proteus_csr_index_t>& sdInfo_u_u_colind = args.array<proteus_csr_index_t>("sdInfo_u_u_colind");
        proteus::pyarray<proteus_csr_index_t>& sdInfo_u_v_rowptr = args.array<proteus_csr_index_t>("sdInfo_u_v_rowptr");
        proteus::pyarray<proteus_csr_index_t>& sdInfo_u_v_colind = args.array<proteus_csr_index_t>("sdInfo_u_v_colind");
        proteus::pyarray<proteus_csr_index_t>& sdInfo_v_v_rowptr = args.array<proteus_csr_index_t>("sdInfo_v_v_rowptr");
        proteus::pyarray<proteus_csr_index_t>& sdInfo_v_v_colind = args.array<proteus_csr_index_t>("sdInfo_v_v_colind");
        proteus::pyarray<proteus_csr_index_t>& sdInfo_v_u_rowptr = args.array<proteus_csr_index_t>("sdInfo_v_u_rowptr");
        proteus::pyarray<proteus_csr_index_t>& sdInfo_v_u_colind = args.array<proteus_csr_index_t>("sdInfo_v_u_colind");
        proteus_index_t offset_h = args.scalar<proteus_index_t>("offset_h");
        proteus_index_t offset_u = args.scalar<proteus_index_t>("offset_u");
        proteus_index_t offset_v = args.scalar<proteus_index_t>("offset_v");
        proteus_index_t stride_h = args.scalar<proteus_index_t>("stride_h");
        proteus_index_t stride_u = args.scalar<proteus_index_t>("stride_u");
        proteus_index_t stride_v = args.scalar<proteus_index_t>("stride_v");
        proteus::pyarray<double>& globalResidual = args.array<double>("globalResidual");
        proteus_index_t nExteriorElementBoundaries_global = args.scalar<proteus_index_t>("nExteriorElementBoundaries_global");
        proteus::pyarray<proteus_index_t>& exteriorElementBoundariesArray = args.array<proteus_index_t>("exteriorElementBoundariesArray");
        proteus::pyarray<proteus_index_t>& elementBoundaryElementsArray = args.array<proteus_index_t>("elementBoundaryElementsArray");
        proteus::pyarray<int>& elementBoundaryLocalElementBoundariesArray = args.array<int>("elementBoundaryLocalElementBoundariesArray");
        proteus::pyarray<int>& isDOFBoundary_h = args.array<int>("isDOFBoundary_h");
        proteus::pyarray<int>& isDOFBoundary_u = args.array<int>("isDOFBoundary_u");
        proteus::pyarray<int>& isDOFBoundary_v = args.array<int>("isDOFBoundary_v");
        proteus::pyarray<int>& isAdvectiveFluxBoundary_h = args.array<int>("isAdvectiveFluxBoundary_h");
        proteus::pyarray<int>& isAdvectiveFluxBoundary_u = args.array<int>("isAdvectiveFluxBoundary_u");
        proteus::pyarray<int>& isAdvectiveFluxBoundary_v = args.array<int>("isAdvectiveFluxBoundary_v");
        proteus::pyarray<int>& isDiffusiveFluxBoundary_u = args.array<int>("isDiffusiveFluxBoundary_u");
        proteus::pyarray<int>& isDiffusiveFluxBoundary_v = args.array<int>("isDiffusiveFluxBoundary_v");
        proteus::pyarray<double>& ebqe_bc_h_ext = args.array<double>("ebqe_bc_h_ext");
        proteus::pyarray<double>& ebqe_bc_flux_mass_ext = args.array<double>("ebqe_bc_flux_mass_ext");
        proteus::pyarray<double>& ebqe_bc_flux_mom_u_adv_ext = args.array<double>("ebqe_bc_flux_mom_u_adv_ext");
        proteus::pyarray<double>& ebqe_bc_flux_mom_v_adv_ext = args.array<double>("ebqe_bc_flux_mom_v_adv_ext");
        proteus::pyarray<double>& ebqe_bc_u_ext = args.array<double>("ebqe_bc_u_ext");
        proteus::pyarray<double>& ebqe_bc_flux_u_diff_ext = args.array<double>("ebqe_bc_flux_u_diff_ext");
        proteus::pyarray<double>& ebqe_penalty_ext = args.array<double>("ebqe_penalty_ext");
        proteus::pyarray<double>& ebqe_bc_v_ext = args.array<double>("ebqe_bc_v_ext");
        proteus::pyarray<double>& ebqe_bc_flux_v_diff_ext = args.array<double>("ebqe_bc_flux_v_diff_ext");
        proteus::pyarray<double>& q_velocity = args.array<double>("q_velocity");
        proteus::pyarray<double>& ebqe_velocity = args.array<double>("ebqe_velocity");
        proteus::pyarray<double>& flux = args.array<double>("flux");
        proteus::pyarray<double>& elementResidual_h_save = args.array<double>("elementResidual_h");
      //
      //loop over elements to compute volume integrals and load them into element and global residual
      //
      double globalConservationError=0.0;
      for(proteus_index_t eN=0;eN<nElements_global;eN++)
        {
          //declare local storage for element residual and initialize
          double elementResidual_h[nDOF_test_element],
            elementResidual_u[nDOF_test_element],
            elementResidual_v[nDOF_test_element];
          for (int i=0;i<nDOF_test_element;i++)
            {
              proteus_index_t eN_i = eN*nDOF_test_element+i;
              elementResidual_h_save.data()[eN_i]=0.0;
              elementResidual_h[i]=0.0;
              elementResidual_u[i]=0.0;
              elementResidual_v[i]=0.0;
            }//i
          //
          //loop over quadrature points and compute integrands
          //
          for(int k=0;k<nQuadraturePoints_element;k++)
            {
              //compute indices and declare local storage
              proteus_index_t eN_k = eN*nQuadraturePoints_element+k,
                eN_k_nSpace = eN_k*nSpace,
                eN_nDOF_trial_element = eN*nDOF_trial_element;
              double b=0.0,h=0.0,u=0.0,v=0.0,h_sge=0.0,u_sge=0.0,v_sge=0.0,
                grad_b[nSpace],grad_h[nSpace],grad_u[nSpace],grad_v[nSpace],
                mass_acc=0.0,
                dmass_acc_h=0.0,
                mom_u_acc=0.0,
                dmom_u_acc_h=0.0,
                dmom_u_acc_u=0.0,
                mom_v_acc=0.0,
                dmom_v_acc_h=0.0,
                dmom_v_acc_v=0.0,
                mass_adv[nSpace],
                dmass_adv_h[nSpace],
                dmass_adv_u[nSpace],
                dmass_adv_v[nSpace],
                dmass_adv_h_sge[nSpace],
                dmass_adv_u_sge[nSpace],
                dmass_adv_v_sge[nSpace],
                mom_u_adv[nSpace],
                dmom_u_adv_h[nSpace],
                dmom_u_adv_u[nSpace],
                dmom_u_adv_v[nSpace],
                dmom_u_adv_h_sge[nSpace],
                dmom_u_adv_u_sge[nSpace],
                dmom_u_adv_v_sge[nSpace],
                mom_v_adv[nSpace],
                dmom_v_adv_h[nSpace],
                dmom_v_adv_u[nSpace],
                dmom_v_adv_v[nSpace],
                dmom_v_adv_h_sge[nSpace],
                dmom_v_adv_u_sge[nSpace],
                dmom_v_adv_v_sge[nSpace],
                mom_u_diff_ten[nSpace],
                mom_v_diff_ten[nSpace],
                mom_uv_diff_ten[1],
                mom_vu_diff_ten[1],
                mom_u_source=0.0,
                dmom_u_source_h=0.0,
                mom_v_source=0.0,
                dmom_v_source_h=0.0,
                mass_acc_t=0.0,
                dmass_acc_h_t=0.0,
                mom_u_acc_t=0.0,
                dmom_u_acc_h_t=0.0,
                dmom_u_acc_u_t=0.0,
                mom_v_acc_t=0.0,
                dmom_v_acc_h_t=0.0,
                dmom_v_acc_v_t=0.0,
                pdeResidual_h=0.0,
                pdeResidual_u=0.0,
                pdeResidual_v=0.0,
                //mwf switched away from usual VMS
                tau_x[9],tau_y[9],
                Lhat_x[nDOF_test_element],
                Lhat_y[nDOF_test_element],
                subgridError_hx=0.0,
                subgridError_ux=0.0,
                subgridError_vx=0.0,
                subgridError_hy=0.0,
                subgridError_uy=0.0,
                subgridError_vy=0.0,
                //
                jac[nSpace*nSpace],
                jacDet,
                jacInv[nSpace*nSpace],
                h_grad_trial[nDOF_trial_element*nSpace],vel_grad_trial[nDOF_trial_element*nSpace],
                h_test_dV[nDOF_trial_element],vel_test_dV[nDOF_trial_element],
                h_grad_test_dV[nDOF_test_element*nSpace],vel_grad_test_dV[nDOF_test_element*nSpace],
                dV,x,y,xt,yt,
                G[nSpace*nSpace],G_dd_G,tr_G,norm_Rv, dmom_adv_star[nSpace],dmom_adv_sge[nSpace];
              //mwf added
              //mwf todo add as input or calculate based on 'entropy'
              double alpha_tau=0.25;
              double include_acc_in_strong_mom = 1.0;//omit accumulation terms from momentum strong residual?
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
                                          x,y);
              ck.calculateMappingVelocity_element(eN,
                                                  k,
                                                  mesh_velocity_dof.data(),
                                                  mesh_l2g.data(),
                                                  mesh_trial_ref.data(),
                                                  xt,yt);
              //xt=0.0;yt=0.0;
              //std::cout<<"xt "<<xt<<'\t'<<yt<<'\t'<<std::endl;
              //get the physical integration weight
              dV = fabs(jacDet)*dV_ref.data()[k];
              //get the trial function gradients
              ck.gradTrialFromRef(&h_grad_trial_ref.data()[k*nDOF_trial_element*nSpace],jacInv,h_grad_trial);
              ck.gradTrialFromRef(&vel_grad_trial_ref.data()[k*nDOF_trial_element*nSpace],jacInv,vel_grad_trial);
              //get the solution
              ck.valFromDOF(b_dof.data(),&h_l2g.data()[eN_nDOF_trial_element],&h_trial_ref.data()[k*nDOF_trial_element],b);
              ck.valFromDOF(h_dof.data(),&h_l2g.data()[eN_nDOF_trial_element],&h_trial_ref.data()[k*nDOF_trial_element],h);
              ck.valFromDOF(u_dof.data(),&vel_l2g.data()[eN_nDOF_trial_element],&vel_trial_ref.data()[k*nDOF_trial_element],u);
              ck.valFromDOF(v_dof.data(),&vel_l2g.data()[eN_nDOF_trial_element],&vel_trial_ref.data()[k*nDOF_trial_element],v);
              ck.valFromDOF(h_dof_sge.data(),&h_l2g.data()[eN_nDOF_trial_element],&h_trial_ref.data()[k*nDOF_trial_element],h_sge);
              ck.valFromDOF(u_dof_sge.data(),&vel_l2g.data()[eN_nDOF_trial_element],&vel_trial_ref.data()[k*nDOF_trial_element],u_sge);
              ck.valFromDOF(v_dof_sge.data(),&vel_l2g.data()[eN_nDOF_trial_element],&vel_trial_ref.data()[k*nDOF_trial_element],v_sge);
              //get the solution gradients
              ck.gradFromDOF(b_dof.data(),&h_l2g.data()[eN_nDOF_trial_element],h_grad_trial,grad_b);
              ck.gradFromDOF(h_dof.data(),&h_l2g.data()[eN_nDOF_trial_element],h_grad_trial,grad_h);
              ck.gradFromDOF(u_dof.data(),&vel_l2g.data()[eN_nDOF_trial_element],vel_grad_trial,grad_u);
              ck.gradFromDOF(v_dof.data(),&vel_l2g.data()[eN_nDOF_trial_element],vel_grad_trial,grad_v);
              //precalculate test function products with integration weights
              for (int j=0;j<nDOF_trial_element;j++)
                {
                  h_test_dV[j] = h_test_ref.data()[k*nDOF_trial_element+j]*dV;
                  vel_test_dV[j] = vel_test_ref.data()[k*nDOF_trial_element+j]*dV;
                  for (int I=0;I<nSpace;I++)
                    {
                      h_grad_test_dV[j*nSpace+I]   = h_grad_trial[j*nSpace+I]*dV;//cek warning won't work for Petrov-Galerkin
                      vel_grad_test_dV[j*nSpace+I] = vel_grad_trial[j*nSpace+I]*dV;//cek warning won't work for Petrov-Galerkin
                    }
                }
              //save velocity at quadrature points for other models to use
              q_velocity.data()[eN_k_nSpace+0]=u;
              q_velocity.data()[eN_k_nSpace+1]=v;
              //
              //calculate pde coefficients at quadrature points
              //
              evaluateCoefficients(nu,
                                   g,
                                   grad_b,
                                   h,
                                   u,
                                   v,
                                   mass_acc,
                                   dmass_acc_h,
                                   mom_u_acc,
                                   dmom_u_acc_h,
                                   dmom_u_acc_u,
                                   mom_v_acc,
                                   dmom_v_acc_h,
                                   dmom_v_acc_v,
                                   mass_adv,
                                   dmass_adv_h,
                                   dmass_adv_u,
                                   dmass_adv_v,
                                   mom_u_adv,
                                   dmom_u_adv_h,
                                   dmom_u_adv_u,
                                   dmom_u_adv_v,
                                   mom_v_adv,
                                   dmom_v_adv_h,
                                   dmom_v_adv_u,
                                   dmom_v_adv_v,
                                   mom_u_diff_ten,
                                   mom_v_diff_ten,
                                   mom_uv_diff_ten,
                                   mom_vu_diff_ten,
                                   mom_u_source,
                                   dmom_u_source_h,
                                   mom_v_source,
                                   dmom_v_source_h);
              //
              //save momentum for time history and velocity for subgrid error
              //
              q_mass_acc.data()[eN_k] = mass_acc;
              q_mom_u_acc.data()[eN_k] = mom_u_acc;
              q_mom_v_acc.data()[eN_k] = mom_v_acc;
              //subgrid error uses grid scale discharge
              q_mass_adv.data()[eN_k_nSpace+0] = h*u;
              q_mass_adv.data()[eN_k_nSpace+1] = h*v;

              //
              //calculate time derivative at quadrature points
              //
              ck.bdf(alphaBDF,
                     q_mass_acc_beta_bdf.data()[eN_k],
                     mass_acc,
                     dmass_acc_h,
                     mass_acc_t,
                     dmass_acc_h_t);
              ck.bdfC2(alphaBDF,
                     q_mom_u_acc_beta_bdf.data()[eN_k],
                     mom_u_acc,
                     dmom_u_acc_h,
                     dmom_u_acc_u,
                     mom_u_acc_t,
                     dmom_u_acc_h_t,
                     dmom_u_acc_u_t);
              ck.bdfC2(alphaBDF,
                     q_mom_v_acc_beta_bdf.data()[eN_k],
                     mom_v_acc,
                     dmom_v_acc_h,
                     dmom_v_acc_v,
                     mom_v_acc_t,
                     dmom_v_acc_h_t,
                     dmom_v_acc_v_t);
              //
              //calculate subgrid error (strong residual and adjoint)
              //
              //calculate strong residual
              dmass_adv_h_sge[0]  = dmass_adv_h[0];
              dmass_adv_h_sge[1]  = dmass_adv_h[1];
              dmass_adv_u_sge[0]  = dmass_adv_u[0];
              dmass_adv_u_sge[1]  = dmass_adv_u[1];
              dmass_adv_v_sge[0]  = dmass_adv_v[0];
              dmass_adv_v_sge[1]  = dmass_adv_v[1];
              dmom_u_adv_h_sge[0] = dmom_u_adv_h[0];
              dmom_u_adv_h_sge[1] = dmom_u_adv_h[1];
              dmom_u_adv_u_sge[0] = dmom_u_adv_u[0];
              dmom_u_adv_u_sge[1] = dmom_u_adv_u[1];
              dmom_u_adv_v_sge[0] = dmom_u_adv_v[0];
              dmom_u_adv_v_sge[1] = dmom_u_adv_v[1];
              dmom_v_adv_h_sge[0] = dmom_v_adv_h[0];
              dmom_v_adv_h_sge[1] = dmom_v_adv_h[1];
              dmom_v_adv_u_sge[0] = dmom_v_adv_u[0];
              dmom_v_adv_u_sge[1] = dmom_v_adv_u[1];
              dmom_v_adv_v_sge[0] = dmom_v_adv_v[0];
              dmom_v_adv_v_sge[1] = dmom_v_adv_v[1];

              //full linearization, lagged

              //mass advective flux
              dmass_adv_h_sge[0]=u_sge;
              dmass_adv_h_sge[1]=v_sge;

              dmass_adv_u_sge[0]=h_sge;
              dmass_adv_u_sge[1]=0.0;

              dmass_adv_v_sge[0]=0.0;
              dmass_adv_v_sge[1]=h_sge;

              //u momentum advective flux
              dmom_u_adv_h_sge[0]=u_sge*u_sge + g*h_sge;
              dmom_u_adv_h_sge[1]=u_sge*v_sge;

              dmom_u_adv_u_sge[0]=h_sge*2.0*u_sge;
              dmom_u_adv_u_sge[1]=h_sge*v_sge;

              dmom_u_adv_v_sge[0]=0.0;
              dmom_u_adv_v_sge[1]=h_sge*u_sge;

              //v momentum advective_flux
              dmom_v_adv_h_sge[0]=v_sge*u_sge;
              dmom_v_adv_h_sge[1]=v_sge*v_sge + g*h_sge;

              dmom_v_adv_u_sge[0]=h_sge*v_sge;
              dmom_v_adv_u_sge[1]=0.0;

              dmom_v_adv_v_sge[0]=h_sge*u_sge;
              dmom_v_adv_v_sge[1]=h_sge*2.0*v_sge;

              pdeResidual_h = ck.Mass_strong(mass_acc_t) +
                ck.Advection_strong(dmass_adv_h_sge,grad_h) +
                ck.Advection_strong(dmass_adv_u_sge,grad_u) +
                ck.Advection_strong(dmass_adv_v_sge,grad_v);

              pdeResidual_u = include_acc_in_strong_mom*ck.Mass_strong(mom_u_acc_t) +
                ck.Advection_strong(dmom_u_adv_h_sge,grad_h) +
                ck.Advection_strong(dmom_u_adv_u_sge,grad_u) +
                ck.Advection_strong(dmom_u_adv_v_sge,grad_v) +
                ck.Reaction_strong(mom_u_source);

              pdeResidual_v =  include_acc_in_strong_mom*ck.Mass_strong(mom_v_acc_t) +
                ck.Advection_strong(dmom_v_adv_h_sge,grad_h) +
                ck.Advection_strong(dmom_v_adv_u_sge,grad_u) +
                ck.Advection_strong(dmom_v_adv_v_sge,grad_v) +
                ck.Reaction_strong(mom_v_source);
              //mwf switch out tau calculations to match adh supg formulation
              calculateSubgridError_tau_supg(elementDiameter.data()[eN],
                                             nu,
                                             g,
                                             h_sge,
                                             u_sge,
                                             v_sge,
                                             alpha_tau,
                                             dV,
                                             tau_x,
                                             tau_y,
                                             q_cfl.data()[eN_k]);

              subgridError_hx = - tau_x[0*3+0]*pdeResidual_h - tau_x[0*3+1]*pdeResidual_u - tau_x[0*3+2]*pdeResidual_v;
              subgridError_ux = - tau_x[1*3+0]*pdeResidual_h - tau_x[1*3+1]*pdeResidual_u - tau_x[1*3+2]*pdeResidual_v;
              subgridError_vx = - tau_x[2*3+0]*pdeResidual_h - tau_x[2*3+1]*pdeResidual_u - tau_x[2*3+2]*pdeResidual_v;

              subgridError_hy = - tau_y[0*3+0]*pdeResidual_h - tau_y[0*3+1]*pdeResidual_u - tau_y[0*3+2]*pdeResidual_v;
              subgridError_uy = - tau_y[1*3+0]*pdeResidual_h - tau_y[1*3+1]*pdeResidual_u - tau_y[1*3+2]*pdeResidual_v;
              subgridError_vy = - tau_y[2*3+0]*pdeResidual_h - tau_y[2*3+1]*pdeResidual_u - tau_y[2*3+2]*pdeResidual_v;

              //Not really SUPG, but is test function contribution
              for (int i=0;i<nDOF_test_element;i++)
                {
                  int i_nSpace = i*nSpace;
                  Lhat_x[i]= -h_grad_test_dV[i_nSpace];
                  Lhat_y[i]= -h_grad_test_dV[i_nSpace+1];
                }
              //mwf end supg tau and test function operator
              norm_Rv = sqrt(pdeResidual_u*pdeResidual_u + pdeResidual_v*pdeResidual_v);
              /* double */
              double grad_norm[2] = {1.0,0.0};
              ck.calculateNumericalDiffusion(shockCapturingCoefficient,
                                             elementDiameter.data()[eN],
                                             norm_Rv,
                                             grad_norm,
                                             q_numDiff_u.data()[eN_k]);

              /*q_numDiff_u.data()[eN_k] = 0.5*elementDiameter.data()[eN]*norm_Rv/(norm_grad+1.0e-8);*/
              q_numDiff_v.data()[eN_k] = q_numDiff_u.data()[eN_k];

              /* ck.calculateNumericalDiffusion(1.0, */
              /*                                     elementDiameter.data()[eN], */
              /*                                     pdeResidual_h, */
              /*                                     grad_h, */
              /*                                     q_numDiff_h.data()[eN_k]); */

              //update element residual

              for(int i=0;i<nDOF_test_element;i++)
                {
                  int i_nSpace=i*nSpace;

                  elementResidual_h[i] += ck.Mass_weak(mass_acc_t,h_test_dV[i]) +
                    ck.Advection_weak(mass_adv,&h_grad_test_dV[i_nSpace]) +
                    //mwf changed stabilization terms
                    ck.SubgridError(subgridError_hx,Lhat_x[i]) +
                    ck.SubgridError(subgridError_hy,Lhat_y[i]) +
                    ck.NumericalDiffusion(q_numDiff_h_last.data()[eN_k],grad_h,&h_grad_test_dV[i_nSpace]);

                  elementResidual_u[i] += ck.Mass_weak(mom_u_acc_t,vel_test_dV[i]) +
                    ck.Advection_weak(mom_u_adv,&vel_grad_test_dV[i_nSpace]) +
                    ck.Diffusion_weak(sdInfo_u_u_rowptr.data(),sdInfo_u_u_colind.data(),mom_u_diff_ten,grad_u,&vel_grad_test_dV[i_nSpace]) +
                    ck.Diffusion_weak(sdInfo_u_v_rowptr.data(),sdInfo_u_v_colind.data(),mom_uv_diff_ten,grad_v,&vel_grad_test_dV[i_nSpace]) +
                    ck.Reaction_weak(mom_u_source,vel_test_dV[i]) +
                    //mwf changed stabilization terms
                    ck.SubgridError(subgridError_ux,Lhat_x[i]) +
                    ck.SubgridError(subgridError_uy,Lhat_y[i]) +
                    ck.NumericalDiffusion(q_numDiff_u_last.data()[eN_k],grad_u,&vel_grad_test_dV[i_nSpace]);

                  elementResidual_v[i] += ck.Mass_weak(mom_v_acc_t,vel_test_dV[i]) +
                    ck.Advection_weak(mom_v_adv,&vel_grad_test_dV[i_nSpace]) +
                    ck.Diffusion_weak(sdInfo_v_v_rowptr.data(),sdInfo_v_v_colind.data(),mom_v_diff_ten,grad_v,&vel_grad_test_dV[i_nSpace]) +
                    ck.Diffusion_weak(sdInfo_v_u_rowptr.data(),sdInfo_v_u_colind.data(),mom_vu_diff_ten,grad_u,&vel_grad_test_dV[i_nSpace]) +
                    ck.Reaction_weak(mom_v_source,vel_test_dV[i]) +
                    //mwf changed stabilization terms
                    ck.SubgridError(subgridError_vx,Lhat_x[i]) +
                    ck.SubgridError(subgridError_vy,Lhat_y[i]) +
                    ck.NumericalDiffusion(q_numDiff_v_last.data()[eN_k],grad_v,&vel_grad_test_dV[i_nSpace]);
                }
            }

          //load element into global residual and save element residual

          for(int i=0;i<nDOF_test_element;i++)
            {
              proteus_index_t eN_i=eN*nDOF_test_element+i;

              elementResidual_h_save.data()[eN_i] +=  elementResidual_h[i];

              globalResidual.data()[offset_h+stride_h*h_l2g.data()[eN_i]]+=elementResidual_h[i];
              globalResidual.data()[offset_u+stride_u*vel_l2g.data()[eN_i]]+=elementResidual_u[i];
              globalResidual.data()[offset_v+stride_v*vel_l2g.data()[eN_i]]+=elementResidual_v[i];
            }
        }
    }
    void calculateJacobian(arguments_dict& args)
    {
        proteus::pyarray<double>& mesh_trial_ref = args.array<double>("mesh_trial_ref");
        proteus::pyarray<double>& mesh_grad_trial_ref = args.array<double>("mesh_grad_trial_ref");
        proteus::pyarray<double>& mesh_dof = args.array<double>("mesh_dof");
        proteus::pyarray<double>& mesh_velocity_dof = args.array<double>("mesh_velocity_dof");
        double MOVING_DOMAIN = args.scalar<double>("MOVING_DOMAIN");
        proteus::pyarray<proteus_index_t>& mesh_l2g = args.array<proteus_index_t>("mesh_l2g");
        proteus::pyarray<double>& dV_ref = args.array<double>("dV_ref");
        proteus::pyarray<double>& h_trial_ref = args.array<double>("h_trial_ref");
        proteus::pyarray<double>& h_grad_trial_ref = args.array<double>("h_grad_trial_ref");
        proteus::pyarray<double>& h_test_ref = args.array<double>("h_test_ref");
        proteus::pyarray<double>& h_grad_test_ref = args.array<double>("h_grad_test_ref");
        proteus::pyarray<double>& vel_trial_ref = args.array<double>("vel_trial_ref");
        proteus::pyarray<double>& vel_grad_trial_ref = args.array<double>("vel_grad_trial_ref");
        proteus::pyarray<double>& vel_test_ref = args.array<double>("vel_test_ref");
        proteus::pyarray<double>& vel_grad_test_ref = args.array<double>("vel_grad_test_ref");
        proteus::pyarray<double>& mesh_trial_trace_ref = args.array<double>("mesh_trial_trace_ref");
        proteus::pyarray<double>& mesh_grad_trial_trace_ref = args.array<double>("mesh_grad_trial_trace_ref");
        proteus::pyarray<double>& dS_ref = args.array<double>("dS_ref");
        proteus::pyarray<double>& h_trial_trace_ref = args.array<double>("h_trial_trace_ref");
        proteus::pyarray<double>& h_grad_trial_trace_ref = args.array<double>("h_grad_trial_trace_ref");
        proteus::pyarray<double>& h_test_trace_ref = args.array<double>("h_test_trace_ref");
        proteus::pyarray<double>& h_grad_test_trace_ref = args.array<double>("h_grad_test_trace_ref");
        proteus::pyarray<double>& vel_trial_trace_ref = args.array<double>("vel_trial_trace_ref");
        proteus::pyarray<double>& vel_grad_trial_trace_ref = args.array<double>("vel_grad_trial_trace_ref");
        proteus::pyarray<double>& vel_test_trace_ref = args.array<double>("vel_test_trace_ref");
        proteus::pyarray<double>& vel_grad_test_trace_ref = args.array<double>("vel_grad_test_trace_ref");
        proteus::pyarray<double>& normal_ref = args.array<double>("normal_ref");
        proteus::pyarray<double>& boundaryJac_ref = args.array<double>("boundaryJac_ref");
        proteus::pyarray<double>& elementDiameter = args.array<double>("elementDiameter");
        proteus_index_t nElements_global = args.scalar<proteus_index_t>("nElements_global");
        double useRBLES = args.scalar<double>("useRBLES");
        double useMetrics = args.scalar<double>("useMetrics");
        double alphaBDF = args.scalar<double>("alphaBDF");
        double nu = args.scalar<double>("nu");
        double g = args.scalar<double>("g");
        proteus::pyarray<proteus_index_t>& h_l2g = args.array<proteus_index_t>("h_l2g");
        proteus::pyarray<proteus_index_t>& vel_l2g = args.array<proteus_index_t>("vel_l2g");
        proteus::pyarray<double>& b_dof = args.array<double>("b_dof");
        proteus::pyarray<double>& h_dof = args.array<double>("h_dof");
        proteus::pyarray<double>& u_dof = args.array<double>("u_dof");
        proteus::pyarray<double>& v_dof = args.array<double>("v_dof");
        proteus::pyarray<double>& h_dof_sge = args.array<double>("h_dof_sge");
        proteus::pyarray<double>& u_dof_sge = args.array<double>("u_dof_sge");
        proteus::pyarray<double>& v_dof_sge = args.array<double>("v_dof_sge");
        proteus::pyarray<double>& q_mass_acc_beta_bdf = args.array<double>("q_mass_acc_beta_bdf");
        proteus::pyarray<double>& q_mom_u_acc_beta_bdf = args.array<double>("q_mom_u_acc_beta_bdf");
        proteus::pyarray<double>& q_mom_v_acc_beta_bdf = args.array<double>("q_mom_v_acc_beta_bdf");
        proteus::pyarray<double>& q_velocity_sge = args.array<double>("q_velocity_sge");
        proteus::pyarray<double>& q_cfl = args.array<double>("q_cfl");
        proteus::pyarray<double>& q_numDiff_h_last = args.array<double>("q_numDiff_h_last");
        proteus::pyarray<double>& q_numDiff_u_last = args.array<double>("q_numDiff_u_last");
        proteus::pyarray<double>& q_numDiff_v_last = args.array<double>("q_numDiff_v_last");
        proteus::pyarray<proteus_csr_index_t>& sdInfo_u_u_rowptr = args.array<proteus_csr_index_t>("sdInfo_u_u_rowptr");
        proteus::pyarray<proteus_csr_index_t>& sdInfo_u_u_colind = args.array<proteus_csr_index_t>("sdInfo_u_u_colind");
        proteus::pyarray<proteus_csr_index_t>& sdInfo_u_v_rowptr = args.array<proteus_csr_index_t>("sdInfo_u_v_rowptr");
        proteus::pyarray<proteus_csr_index_t>& sdInfo_u_v_colind = args.array<proteus_csr_index_t>("sdInfo_u_v_colind");
        proteus::pyarray<proteus_csr_index_t>& sdInfo_v_v_rowptr = args.array<proteus_csr_index_t>("sdInfo_v_v_rowptr");
        proteus::pyarray<proteus_csr_index_t>& sdInfo_v_v_colind = args.array<proteus_csr_index_t>("sdInfo_v_v_colind");
        proteus::pyarray<proteus_csr_index_t>& sdInfo_v_u_rowptr = args.array<proteus_csr_index_t>("sdInfo_v_u_rowptr");
        proteus::pyarray<proteus_csr_index_t>& sdInfo_v_u_colind = args.array<proteus_csr_index_t>("sdInfo_v_u_colind");
        proteus::pyarray<proteus_csr_index_t>& csrRowIndeces_h_h = args.array<proteus_csr_index_t>("csrRowIndeces_h_h");
        proteus::pyarray<proteus_csr_index_t>& csrColumnOffsets_h_h = args.array<proteus_csr_index_t>("csrColumnOffsets_h_h");
        proteus::pyarray<proteus_csr_index_t>& csrRowIndeces_h_u = args.array<proteus_csr_index_t>("csrRowIndeces_h_u");
        proteus::pyarray<proteus_csr_index_t>& csrColumnOffsets_h_u = args.array<proteus_csr_index_t>("csrColumnOffsets_h_u");
        proteus::pyarray<proteus_csr_index_t>& csrRowIndeces_h_v = args.array<proteus_csr_index_t>("csrRowIndeces_h_v");
        proteus::pyarray<proteus_csr_index_t>& csrColumnOffsets_h_v = args.array<proteus_csr_index_t>("csrColumnOffsets_h_v");
        proteus::pyarray<proteus_csr_index_t>& csrRowIndeces_u_h = args.array<proteus_csr_index_t>("csrRowIndeces_u_h");
        proteus::pyarray<proteus_csr_index_t>& csrColumnOffsets_u_h = args.array<proteus_csr_index_t>("csrColumnOffsets_u_h");
        proteus::pyarray<proteus_csr_index_t>& csrRowIndeces_u_u = args.array<proteus_csr_index_t>("csrRowIndeces_u_u");
        proteus::pyarray<proteus_csr_index_t>& csrColumnOffsets_u_u = args.array<proteus_csr_index_t>("csrColumnOffsets_u_u");
        proteus::pyarray<proteus_csr_index_t>& csrRowIndeces_u_v = args.array<proteus_csr_index_t>("csrRowIndeces_u_v");
        proteus::pyarray<proteus_csr_index_t>& csrColumnOffsets_u_v = args.array<proteus_csr_index_t>("csrColumnOffsets_u_v");
        proteus::pyarray<proteus_csr_index_t>& csrRowIndeces_v_h = args.array<proteus_csr_index_t>("csrRowIndeces_v_h");
        proteus::pyarray<proteus_csr_index_t>& csrColumnOffsets_v_h = args.array<proteus_csr_index_t>("csrColumnOffsets_v_h");
        proteus::pyarray<proteus_csr_index_t>& csrRowIndeces_v_u = args.array<proteus_csr_index_t>("csrRowIndeces_v_u");
        proteus::pyarray<proteus_csr_index_t>& csrColumnOffsets_v_u = args.array<proteus_csr_index_t>("csrColumnOffsets_v_u");
        proteus::pyarray<proteus_csr_index_t>& csrRowIndeces_v_v = args.array<proteus_csr_index_t>("csrRowIndeces_v_v");
        proteus::pyarray<proteus_csr_index_t>& csrColumnOffsets_v_v = args.array<proteus_csr_index_t>("csrColumnOffsets_v_v");
        proteus::pyarray<double>& globalJacobian = args.array<double>("globalJacobian");
        proteus_index_t nExteriorElementBoundaries_global = args.scalar<proteus_index_t>("nExteriorElementBoundaries_global");
        proteus::pyarray<proteus_index_t>& exteriorElementBoundariesArray = args.array<proteus_index_t>("exteriorElementBoundariesArray");
        proteus::pyarray<proteus_index_t>& elementBoundaryElementsArray = args.array<proteus_index_t>("elementBoundaryElementsArray");
        proteus::pyarray<int>& elementBoundaryLocalElementBoundariesArray = args.array<int>("elementBoundaryLocalElementBoundariesArray");
        proteus::pyarray<int>& isDOFBoundary_h = args.array<int>("isDOFBoundary_h");
        proteus::pyarray<int>& isDOFBoundary_u = args.array<int>("isDOFBoundary_u");
        proteus::pyarray<int>& isDOFBoundary_v = args.array<int>("isDOFBoundary_v");
        proteus::pyarray<int>& isAdvectiveFluxBoundary_h = args.array<int>("isAdvectiveFluxBoundary_h");
        proteus::pyarray<int>& isAdvectiveFluxBoundary_u = args.array<int>("isAdvectiveFluxBoundary_u");
        proteus::pyarray<int>& isAdvectiveFluxBoundary_v = args.array<int>("isAdvectiveFluxBoundary_v");
        proteus::pyarray<int>& isDiffusiveFluxBoundary_u = args.array<int>("isDiffusiveFluxBoundary_u");
        proteus::pyarray<int>& isDiffusiveFluxBoundary_v = args.array<int>("isDiffusiveFluxBoundary_v");
        proteus::pyarray<double>& ebqe_bc_h_ext = args.array<double>("ebqe_bc_h_ext");
        proteus::pyarray<double>& ebqe_bc_flux_mass_ext = args.array<double>("ebqe_bc_flux_mass_ext");
        proteus::pyarray<double>& ebqe_bc_flux_mom_u_adv_ext = args.array<double>("ebqe_bc_flux_mom_u_adv_ext");
        proteus::pyarray<double>& ebqe_bc_flux_mom_v_adv_ext = args.array<double>("ebqe_bc_flux_mom_v_adv_ext");
        proteus::pyarray<double>& ebqe_bc_u_ext = args.array<double>("ebqe_bc_u_ext");
        proteus::pyarray<double>& ebqe_bc_flux_u_diff_ext = args.array<double>("ebqe_bc_flux_u_diff_ext");
        proteus::pyarray<double>& ebqe_penalty_ext = args.array<double>("ebqe_penalty_ext");
        proteus::pyarray<double>& ebqe_bc_v_ext = args.array<double>("ebqe_bc_v_ext");
        proteus::pyarray<double>& ebqe_bc_flux_v_diff_ext = args.array<double>("ebqe_bc_flux_v_diff_ext");
        proteus::pyarray<proteus_csr_index_t>& csrColumnOffsets_eb_h_h = args.array<proteus_csr_index_t>("csrColumnOffsets_eb_h_h");
        proteus::pyarray<proteus_csr_index_t>& csrColumnOffsets_eb_h_u = args.array<proteus_csr_index_t>("csrColumnOffsets_eb_h_u");
        proteus::pyarray<proteus_csr_index_t>& csrColumnOffsets_eb_h_v = args.array<proteus_csr_index_t>("csrColumnOffsets_eb_h_v");
        proteus::pyarray<proteus_csr_index_t>& csrColumnOffsets_eb_u_h = args.array<proteus_csr_index_t>("csrColumnOffsets_eb_u_h");
        proteus::pyarray<proteus_csr_index_t>& csrColumnOffsets_eb_u_u = args.array<proteus_csr_index_t>("csrColumnOffsets_eb_u_u");
        proteus::pyarray<proteus_csr_index_t>& csrColumnOffsets_eb_u_v = args.array<proteus_csr_index_t>("csrColumnOffsets_eb_u_v");
        proteus::pyarray<proteus_csr_index_t>& csrColumnOffsets_eb_v_h = args.array<proteus_csr_index_t>("csrColumnOffsets_eb_v_h");
        proteus::pyarray<proteus_csr_index_t>& csrColumnOffsets_eb_v_u = args.array<proteus_csr_index_t>("csrColumnOffsets_eb_v_u");
        proteus::pyarray<proteus_csr_index_t>& csrColumnOffsets_eb_v_v = args.array<proteus_csr_index_t>("csrColumnOffsets_eb_v_v");
      double tauSum=0.0;
      //
      //loop over elements to compute volume integrals and load them into the element Jacobians and global Jacobian
      //
      for(proteus_index_t eN=0;eN<nElements_global;eN++)
        {
          double  elementJacobian_h_h[nDOF_test_element][nDOF_trial_element],
            elementJacobian_h_u[nDOF_test_element][nDOF_trial_element],
            elementJacobian_h_v[nDOF_test_element][nDOF_trial_element],
            elementJacobian_u_h[nDOF_test_element][nDOF_trial_element],
            elementJacobian_u_u[nDOF_test_element][nDOF_trial_element],
            elementJacobian_u_v[nDOF_test_element][nDOF_trial_element],
            elementJacobian_v_h[nDOF_test_element][nDOF_trial_element],
            elementJacobian_v_u[nDOF_test_element][nDOF_trial_element],
            elementJacobian_v_v[nDOF_test_element][nDOF_trial_element];
          for (int i=0;i<nDOF_test_element;i++)
            for (int j=0;j<nDOF_trial_element;j++)
              {
                elementJacobian_h_h[i][j]=0.0;
                elementJacobian_h_u[i][j]=0.0;
                elementJacobian_h_v[i][j]=0.0;
                elementJacobian_u_h[i][j]=0.0;
                elementJacobian_u_u[i][j]=0.0;
                elementJacobian_u_v[i][j]=0.0;
                elementJacobian_v_h[i][j]=0.0;
                elementJacobian_v_u[i][j]=0.0;
                elementJacobian_v_v[i][j]=0.0;
              }
          for  (int k=0;k<nQuadraturePoints_element;k++)
            {
              proteus_index_t eN_k = eN*nQuadraturePoints_element+k, //index to a scalar at a quadrature point
                eN_k_nSpace = eN_k*nSpace,
                eN_nDOF_trial_element = eN*nDOF_trial_element; //index to a vector at a quadrature point

              //declare local storage
              double b=0.0,
                h=0.0,
                u=0.0,
                v=0.0,
                h_sge=0.0,
                u_sge=0.0,
                v_sge=0.0,
                grad_b[nSpace],
                grad_h[nSpace],
                grad_u[nSpace],
                grad_v[nSpace],
                mass_acc=0.0,
                dmass_acc_h=0.0,
                mom_u_acc=0.0,
                dmom_u_acc_h=0.0,
                dmom_u_acc_u=0.0,
                mom_v_acc=0.0,
                dmom_v_acc_h=0.0,
                dmom_v_acc_v=0.0,
                mass_adv[nSpace],
                dmass_adv_h[nSpace],
                dmass_adv_u[nSpace],
                dmass_adv_v[nSpace],
                dmass_adv_h_sge[nSpace],
                dmass_adv_u_sge[nSpace],
                dmass_adv_v_sge[nSpace],
                mom_u_adv[nSpace],
                dmom_u_adv_h[nSpace],
                dmom_u_adv_u[nSpace],
                dmom_u_adv_v[nSpace],
                dmom_u_adv_h_sge[nSpace],
                dmom_u_adv_u_sge[nSpace],
                dmom_u_adv_v_sge[nSpace],
                mom_v_adv[nSpace],
                dmom_v_adv_h[nSpace],
                dmom_v_adv_u[nSpace],
                dmom_v_adv_v[nSpace],
                dmom_v_adv_h_sge[nSpace],
                dmom_v_adv_u_sge[nSpace],
                dmom_v_adv_v_sge[nSpace],
                mom_u_diff_ten[nSpace],
                mom_v_diff_ten[nSpace],
                mom_uv_diff_ten[1],
                mom_vu_diff_ten[1],
                mom_u_source=0.0,
                dmom_u_source_h=0.0,
                mom_v_source=0.0,
                dmom_v_source_h=0.0,
                mass_acc_t=0.0,
                dmass_acc_h_t=0.0,
                mom_u_acc_t=0.0,
                dmom_u_acc_h_t=0.0,
                dmom_u_acc_u_t=0.0,
                mom_v_acc_t=0.0,
                dmom_v_acc_h_t=0.0,
                dmom_v_acc_v_t=0.0,
                tau[9],
                pdeResidual_h=0.0,
                pdeResidual_u=0.0,
                pdeResidual_v=0.0,
                dpdeResidual_h_h[nDOF_trial_element],
                dpdeResidual_h_u[nDOF_trial_element],
                dpdeResidual_h_v[nDOF_trial_element],
                dpdeResidual_u_h[nDOF_trial_element],
                dpdeResidual_u_u[nDOF_trial_element],
                dpdeResidual_u_v[nDOF_trial_element],
                dpdeResidual_v_h[nDOF_trial_element],
                dpdeResidual_v_u[nDOF_trial_element],
                dpdeResidual_v_v[nDOF_trial_element],
                Lstar_h_h[nDOF_test_element],
                Lstar_u_h[nDOF_test_element],
                Lstar_v_h[nDOF_test_element],
                Lstar_h_u[nDOF_test_element],
                Lstar_u_u[nDOF_test_element],
                Lstar_v_u[nDOF_test_element],
                Lstar_h_v[nDOF_test_element],
                Lstar_u_v[nDOF_test_element],
                Lstar_v_v[nDOF_test_element],
                subgridError_h=0.0,
                subgridError_u=0.0,
                subgridError_v=0.0,
                dsubgridError_h_h[nDOF_trial_element],
                dsubgridError_h_u[nDOF_trial_element],
                dsubgridError_h_v[nDOF_trial_element],
                dsubgridError_u_h[nDOF_trial_element],
                dsubgridError_u_u[nDOF_trial_element],
                dsubgridError_u_v[nDOF_trial_element],
                dsubgridError_v_h[nDOF_trial_element],
                dsubgridError_v_u[nDOF_trial_element],
                dsubgridError_v_v[nDOF_trial_element],
                tau_h=0.0,tau_h0=0.0,tau_h1=0.0,
                tau_v=0.0,tau_v0=0.0,tau_v1=0.0,
                jac[nSpace*nSpace],
                jacDet,
                jacInv[nSpace*nSpace],
                h_grad_trial[nDOF_trial_element*nSpace],
                vel_grad_trial[nDOF_trial_element*nSpace],
                dV,
                h_test_dV[nDOF_test_element],
                vel_test_dV[nDOF_test_element],
                h_grad_test_dV[nDOF_test_element*nSpace],
                vel_grad_test_dV[nDOF_test_element*nSpace],
                x,y,xt,yt,
                G[nSpace*nSpace],G_dd_G,tr_G, dmom_adv_star[nSpace], dmom_adv_sge[nSpace];
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
                                          x,y);
              ck.calculateMappingVelocity_element(eN,
                                                  k,
                                                  mesh_velocity_dof.data(),
                                                  mesh_l2g.data(),
                                                  mesh_trial_ref.data(),
                                                  xt,yt);
              //get the physical integration weight
              dV = fabs(jacDet)*dV_ref.data()[k];

              //get the trial function gradients
              ck.gradTrialFromRef(&h_grad_trial_ref.data()[k*nDOF_trial_element*nSpace],jacInv,h_grad_trial);
              ck.gradTrialFromRef(&vel_grad_trial_ref.data()[k*nDOF_trial_element*nSpace],jacInv,vel_grad_trial);
              //get the solution
              ck.valFromDOF(b_dof.data(),&h_l2g.data()[eN_nDOF_trial_element],&h_trial_ref.data()[k*nDOF_trial_element],b);
              ck.valFromDOF(h_dof.data(),&h_l2g.data()[eN_nDOF_trial_element],&h_trial_ref.data()[k*nDOF_trial_element],h);
              ck.valFromDOF(u_dof.data(),&vel_l2g.data()[eN_nDOF_trial_element],&vel_trial_ref.data()[k*nDOF_trial_element],u);
              ck.valFromDOF(v_dof.data(),&vel_l2g.data()[eN_nDOF_trial_element],&vel_trial_ref.data()[k*nDOF_trial_element],v);
              ck.valFromDOF(h_dof_sge.data(),&h_l2g.data()[eN_nDOF_trial_element],&h_trial_ref.data()[k*nDOF_trial_element],h_sge);
              ck.valFromDOF(u_dof_sge.data(),&vel_l2g.data()[eN_nDOF_trial_element],&vel_trial_ref.data()[k*nDOF_trial_element],u_sge);
              ck.valFromDOF(v_dof_sge.data(),&vel_l2g.data()[eN_nDOF_trial_element],&vel_trial_ref.data()[k*nDOF_trial_element],v_sge);
              //get the solution gradients
              ck.gradFromDOF(b_dof.data(),&h_l2g.data()[eN_nDOF_trial_element],h_grad_trial,grad_b);
              ck.gradFromDOF(h_dof.data(),&h_l2g.data()[eN_nDOF_trial_element],h_grad_trial,grad_h);
              ck.gradFromDOF(u_dof.data(),&vel_l2g.data()[eN_nDOF_trial_element],vel_grad_trial,grad_u);
              ck.gradFromDOF(v_dof.data(),&vel_l2g.data()[eN_nDOF_trial_element],vel_grad_trial,grad_v);
              //precalculate test function products with integration weights
              for (int j=0;j<nDOF_trial_element;j++)
                {
                  h_test_dV[j] = h_test_ref.data()[k*nDOF_trial_element+j]*dV;
                  vel_test_dV[j] = vel_test_ref.data()[k*nDOF_trial_element+j]*dV;
                  for (int I=0;I<nSpace;I++)
                    {
                      h_grad_test_dV[j*nSpace+I]   = h_grad_trial[j*nSpace+I]*dV;//cek warning won't work for Petrov-Galerkin
                      vel_grad_test_dV[j*nSpace+I] = vel_grad_trial[j*nSpace+I]*dV;//cek warning won't work for Petrov-Galerkin}
                    }
                }
              evaluateCoefficients(nu,
                                   g,
                                   grad_b,
                                   h,
                                   u,
                                   v,
                                   mass_acc,
                                   dmass_acc_h,
                                   mom_u_acc,
                                   dmom_u_acc_h,
                                   dmom_u_acc_u,
                                   mom_v_acc,
                                   dmom_v_acc_h,
                                   dmom_v_acc_v,
                                   mass_adv,
                                   dmass_adv_h,
                                   dmass_adv_u,
                                   dmass_adv_v,
                                   mom_u_adv,
                                   dmom_u_adv_h,
                                   dmom_u_adv_u,
                                   dmom_u_adv_v,
                                   mom_v_adv,
                                   dmom_v_adv_h,
                                   dmom_v_adv_u,
                                   dmom_v_adv_v,
                                   mom_u_diff_ten,
                                   mom_v_diff_ten,
                                   mom_uv_diff_ten,
                                   mom_vu_diff_ten,
                                   mom_u_source,
                                   dmom_u_source_h,
                                   mom_v_source,
                                   dmom_v_source_h);
              //
              //moving mesh
              //
              /* mass_adv[0] -= MOVING_DOMAIN*mass_acc*xt; */
              /* mass_adv[1] -= MOVING_DOMAIN*mass_acc*yt; */

              /* dmass_adv_h[0] -= MOVING_DOMAIN*dmass_acc_h*xt; */
              /* dmass_adv_h[1] -= MOVING_DOMAIN*dmass_acc_h*yt; */

              /* mom_u_adv[0] -= MOVING_DOMAIN*mom_u_acc*xt; */
              /* mom_u_adv[1] -= MOVING_DOMAIN*mom_u_acc*yt; */

              /* dmom_u_adv_h[0] -= MOVING_DOMAIN*dmom_u_acc_h*xt; */
              /* dmom_u_adv_h[1] -= MOVING_DOMAIN*dmom_u_acc_h*yt; */

              /* dmom_u_adv_u[0] -= MOVING_DOMAIN*dmom_u_acc_u*xt; */
              /* dmom_u_adv_u[1] -= MOVING_DOMAIN*dmom_u_acc_u*yt; */

              /* mom_v_adv[0] -= MOVING_DOMAIN*mom_v_acc*xt; */
              /* mom_v_adv[1] -= MOVING_DOMAIN*mom_v_acc*yt; */

              /* dmom_v_adv_v[0] -= MOVING_DOMAIN*dmom_v_acc_h*xt; */
              /* dmom_v_adv_v[1] -= MOVING_DOMAIN*dmom_v_acc_h*yt; */

              /* dmom_v_adv_v[0] -= MOVING_DOMAIN*dmom_v_acc_v*xt; */
              /* dmom_v_adv_v[1] -= MOVING_DOMAIN*dmom_v_acc_v*yt; */
              //
              //calculate time derivatives
              //
              ck.bdf(alphaBDF,
                     q_mass_acc_beta_bdf.data()[eN_k],
                     mass_acc,
                     dmass_acc_h,
                     mass_acc_t,
                     dmass_acc_h_t);
              ck.bdfC2(alphaBDF,
                       q_mom_u_acc_beta_bdf.data()[eN_k],
                       mom_u_acc,
                       dmom_u_acc_h,
                       dmom_u_acc_u,
                       mom_u_acc_t,
                       dmom_u_acc_h_t,
                       dmom_u_acc_u_t);
              ck.bdfC2(alphaBDF,
                       q_mom_v_acc_beta_bdf.data()[eN_k],
                       mom_v_acc,
                       dmom_v_acc_h,
                       dmom_v_acc_v,
                       mom_v_acc_t,
                       dmom_v_acc_h_t,
                       dmom_v_acc_v_t);
              //
              //calculate subgrid error contribution to the Jacobian (strong residual, adjoint, jacobian of strong residual)
              //
              //
              //calculate strong residual
              //
              dmass_adv_h_sge[0]  = dmass_adv_h[0];
              dmass_adv_h_sge[1]  = dmass_adv_h[1];
              dmass_adv_u_sge[0]  = dmass_adv_u[0];
              dmass_adv_u_sge[1]  = dmass_adv_u[1];
              dmass_adv_v_sge[0]  = dmass_adv_v[0];
              dmass_adv_v_sge[1]  = dmass_adv_v[1];
              dmom_u_adv_h_sge[0] = dmom_u_adv_h[0];
              dmom_u_adv_h_sge[1] = dmom_u_adv_h[1];
              dmom_u_adv_u_sge[0] = dmom_u_adv_u[0];
              dmom_u_adv_u_sge[1] = dmom_u_adv_u[1];
              dmom_u_adv_v_sge[0] = dmom_u_adv_v[0];
              dmom_u_adv_v_sge[1] = dmom_u_adv_v[1];
              dmom_v_adv_h_sge[0] = dmom_v_adv_h[0];
              dmom_v_adv_h_sge[1] = dmom_v_adv_h[1];
              dmom_v_adv_u_sge[0] = dmom_v_adv_u[0];
              dmom_v_adv_u_sge[1] = dmom_v_adv_u[1];
              dmom_v_adv_v_sge[0] = dmom_v_adv_v[0];
              dmom_v_adv_v_sge[1] = dmom_v_adv_v[1];

              /* //mass advective flux */
              /* dmass_adv_h_sge[0]=u_sge; */
              /* dmass_adv_h_sge[1]=v_sge; */

              /* dmass_adv_u_sge[0]=0.0;//h_sge; */
              /* dmass_adv_u_sge[1]=0.0; */

              /* dmass_adv_v_sge[0]=0.0; */
              /* dmass_adv_v_sge[1]=0.0;//h_sge; */

              /* //u momentum advective flux */
              /* dmom_u_adv_h_sge[0]=0.0;//u_sge*u_sge + g*h_sge; */
              /* dmom_u_adv_h_sge[1]=0.0;//u_sge*v_sge; */

              /* dmom_u_adv_u_sge[0]=h_sge*u_sge;//h_sge*2.0*u_sge; */
              /* dmom_u_adv_u_sge[1]=h_sge*v_sge; */

              /* dmom_u_adv_v_sge[0]=0.0; */
              /* dmom_u_adv_v_sge[1]=0.0;//h_sge*u_sge; */

              /* //v momentum advective_flux */
              /* dmom_v_adv_h_sge[0]=0.0;//v_sge*u_sge; */
              /* dmom_v_adv_h_sge[1]=0.0;//v_sge*v_sge + g*h_sge; */

              /* dmom_v_adv_u_sge[0]=0.0;//h_sge*v_sge; */
              /* dmom_v_adv_u_sge[1]=0.0; */

              /* dmom_v_adv_v_sge[0]=h_sge*u_sge; */
              /* dmom_v_adv_v_sge[1]=h_sge*v_sge;//h_sge*2.0*v_sge; */

              //full linearization, lagged

              //mass advective flux
              dmass_adv_h_sge[0]=u_sge;
              dmass_adv_h_sge[1]=v_sge;

              dmass_adv_u_sge[0]=h_sge;
              dmass_adv_u_sge[1]=0.0;

              dmass_adv_v_sge[0]=0.0;
              dmass_adv_v_sge[1]=h_sge;

              //u momentum advective flux
              dmom_u_adv_h_sge[0]=u_sge*u_sge + g*h_sge;
              dmom_u_adv_h_sge[1]=u_sge*v_sge;

              dmom_u_adv_u_sge[0]=h_sge*2.0*u_sge;
              dmom_u_adv_u_sge[1]=h_sge*v_sge;

              dmom_u_adv_v_sge[0]=0.0;
              dmom_u_adv_v_sge[1]=h_sge*u_sge;

              //v momentum advective_flux
              dmom_v_adv_h_sge[0]=v_sge*u_sge;
              dmom_v_adv_h_sge[1]=v_sge*v_sge + g*h_sge;

              dmom_v_adv_u_sge[0]=h_sge*v_sge;
              dmom_v_adv_u_sge[1]=0.0;

              dmom_v_adv_v_sge[0]=h_sge*u_sge;
              dmom_v_adv_v_sge[1]=h_sge*2.0*v_sge;

              //calculate the Jacobian of strong residual
              for (int j=0;j<nDOF_trial_element;j++)
                {
                  int j_nSpace = j*nSpace;

                  dpdeResidual_h_h[j]= ck.MassJacobian_strong(dmass_acc_h_t,h_trial_ref.data()[k*nDOF_trial_element+j])+
                    ck.AdvectionJacobian_strong(dmass_adv_h_sge,&h_grad_trial[j_nSpace]);

                  dpdeResidual_h_u[j]=ck.AdvectionJacobian_strong(dmass_adv_u_sge,&vel_grad_trial[j_nSpace]);

                  dpdeResidual_h_v[j]=ck.AdvectionJacobian_strong(dmass_adv_v_sge,&vel_grad_trial[j_nSpace]);

                  dpdeResidual_u_h[j]= ck.MassJacobian_strong(dmom_u_acc_h_t,h_trial_ref.data()[k*nDOF_trial_element+j])+
                    ck.AdvectionJacobian_strong(dmom_u_adv_h_sge,&h_grad_trial[j_nSpace]) +
                    ck.ReactionJacobian_strong(dmom_u_source_h,h_trial_ref.data()[k*nDOF_trial_element+j]);

                  dpdeResidual_u_u[j]= ck.MassJacobian_strong(dmom_u_acc_u_t,vel_trial_ref.data()[k*nDOF_trial_element+j])+
                    ck.AdvectionJacobian_strong(dmom_u_adv_u_sge,&vel_grad_trial[j_nSpace]);

                  dpdeResidual_u_v[j]=ck.AdvectionJacobian_strong(dmom_u_adv_v_sge,&vel_grad_trial[j_nSpace]);

                  dpdeResidual_v_h[j]= ck.MassJacobian_strong(dmom_v_acc_h_t,h_trial_ref.data()[k*nDOF_trial_element+j])+
                    ck.AdvectionJacobian_strong(dmom_v_adv_h_sge,&h_grad_trial[j_nSpace])+
                    ck.ReactionJacobian_strong(dmom_v_source_h,h_trial_ref.data()[k*nDOF_trial_element+j]);

                  dpdeResidual_v_u[j]=ck.AdvectionJacobian_strong(dmom_v_adv_u_sge,&vel_grad_trial[j_nSpace]);

                  dpdeResidual_v_v[j]= ck.MassJacobian_strong(dmom_v_acc_v_t,vel_trial_ref.data()[k*nDOF_trial_element+j])+
                    ck.AdvectionJacobian_strong(dmom_v_adv_v_sge,&vel_grad_trial[j_nSpace]);

                }
              /* //calculate tau and tau*Res */
              calculateSubgridError_tau(elementDiameter.data()[eN],
                                        nu,
                                        g,
                                        h_sge,
                                        u_sge,
                                        v_sge,
                                        tau,
                                        q_cfl.data()[eN_k]);

              for (int i=0;i<9;i++)
                tauSum += tau[i];

              for (int j=0;j<nDOF_trial_element;j++)
                {
                  dsubgridError_h_h[j] = - tau[0*3+0]*dpdeResidual_h_h[j] - tau[0*3+1]*dpdeResidual_u_h[j] - tau[0*3+2]*dpdeResidual_v_h[j];
                  dsubgridError_h_u[j] = - tau[0*3+0]*dpdeResidual_h_u[j] - tau[0*3+1]*dpdeResidual_u_u[j] - tau[0*3+2]*dpdeResidual_v_u[j];
                  dsubgridError_h_v[j] = - tau[0*3+0]*dpdeResidual_h_v[j] - tau[0*3+1]*dpdeResidual_u_v[j] - tau[0*3+2]*dpdeResidual_v_v[j];

                  dsubgridError_u_h[j] = - tau[1*3+0]*dpdeResidual_h_h[j] - tau[1*3+1]*dpdeResidual_u_h[j] - tau[1*3+2]*dpdeResidual_v_h[j];
                  dsubgridError_u_u[j] = - tau[1*3+0]*dpdeResidual_h_u[j] - tau[1*3+1]*dpdeResidual_u_u[j] - tau[1*3+2]*dpdeResidual_v_u[j];
                  dsubgridError_u_v[j] = - tau[1*3+0]*dpdeResidual_h_v[j] - tau[1*3+1]*dpdeResidual_u_v[j] - tau[1*3+2]*dpdeResidual_v_v[j];

                  dsubgridError_v_h[j] = - tau[2*3+0]*dpdeResidual_h_h[j] - tau[2*3+1]*dpdeResidual_u_h[j] - tau[2*3+2]*dpdeResidual_v_h[j];
                  dsubgridError_v_u[j] = - tau[2*3+0]*dpdeResidual_h_u[j] - tau[2*3+1]*dpdeResidual_u_u[j] - tau[2*3+2]*dpdeResidual_v_u[j];
                  dsubgridError_v_v[j] = - tau[2*3+0]*dpdeResidual_h_v[j] - tau[2*3+1]*dpdeResidual_u_v[j] - tau[2*3+2]*dpdeResidual_v_v[j];
                }
              //adjoint times the test functions
              for (int i=0;i<nDOF_test_element;i++)
                {
                  int i_nSpace = i*nSpace;
                  Lstar_h_h[i]=ck.Advection_adjoint(dmass_adv_h_sge,&h_grad_test_dV[i_nSpace]);
                  Lstar_u_h[i]=ck.Advection_adjoint(dmass_adv_u_sge,&h_grad_test_dV[i_nSpace]);
                  Lstar_v_h[i]=ck.Advection_adjoint(dmass_adv_v_sge,&h_grad_test_dV[i_nSpace]);

                  Lstar_h_u[i]=ck.Advection_adjoint(dmom_u_adv_h_sge,&vel_grad_test_dV[i_nSpace]) +
                    ck.Reaction_adjoint(dmom_u_source_h,vel_test_dV[i]);
                  Lstar_u_u[i]=ck.Advection_adjoint(dmom_u_adv_u_sge,&vel_grad_test_dV[i_nSpace]);
                  Lstar_v_u[i]=ck.Advection_adjoint(dmom_u_adv_v_sge,&vel_grad_test_dV[i_nSpace]);

                  Lstar_h_v[i]=ck.Advection_adjoint(dmom_v_adv_h_sge,&vel_grad_test_dV[i_nSpace])+
                    ck.Reaction_adjoint(dmom_v_source_h,vel_test_dV[i]);
                  Lstar_u_v[i]=ck.Advection_adjoint(dmom_v_adv_u_sge,&vel_grad_test_dV[i_nSpace]);
                  Lstar_v_v[i]=ck.Advection_adjoint(dmom_v_adv_v_sge,&vel_grad_test_dV[i_nSpace]);
                }

              for(int i=0;i<nDOF_test_element;i++)
                {
                  int i_nSpace = i*nSpace;
                  for(int j=0;j<nDOF_trial_element;j++)
                    {
                      int j_nSpace = j*nSpace;
                      //h
                      elementJacobian_h_h[i][j] += ck.MassJacobian_weak(dmass_acc_h_t,h_trial_ref.data()[k*nDOF_trial_element+j],h_test_dV[i]) +
                        ck.AdvectionJacobian_weak(dmass_adv_h,h_trial_ref.data()[k*nDOF_trial_element+j],&h_grad_test_dV[i_nSpace]) +
                        ck.SubgridErrorJacobian(dsubgridError_h_h[j],Lstar_h_h[i]) +
                        ck.SubgridErrorJacobian(dsubgridError_u_h[j],Lstar_u_h[i]) +
                        ck.SubgridErrorJacobian(dsubgridError_v_h[j],Lstar_v_h[i]) +
                        ck.NumericalDiffusionJacobian(q_numDiff_h_last.data()[eN_k],&h_grad_trial[j_nSpace],&h_grad_test_dV[i_nSpace]);

                      elementJacobian_h_u[i][j] += ck.AdvectionJacobian_weak(dmass_adv_u,vel_trial_ref.data()[k*nDOF_trial_element+j],&h_grad_test_dV[i_nSpace]) +
                        ck.SubgridErrorJacobian(dsubgridError_h_u[j],Lstar_h_h[i]) +
                        ck.SubgridErrorJacobian(dsubgridError_u_u[j],Lstar_u_h[i]) +
                        ck.SubgridErrorJacobian(dsubgridError_v_u[j],Lstar_v_h[i]);

                      elementJacobian_h_v[i][j] += ck.AdvectionJacobian_weak(dmass_adv_v,vel_trial_ref.data()[k*nDOF_trial_element+j],&h_grad_test_dV[i_nSpace]) +
                        ck.SubgridErrorJacobian(dsubgridError_h_v[j],Lstar_h_h[i]) +
                        ck.SubgridErrorJacobian(dsubgridError_u_v[j],Lstar_u_h[i]) +
                        ck.SubgridErrorJacobian(dsubgridError_v_v[j],Lstar_v_h[i]);

                      //u
                      elementJacobian_u_h[i][j] += ck.MassJacobian_weak(dmom_u_acc_h_t,h_trial_ref.data()[k*nDOF_trial_element+j],vel_test_dV[i]) +
                        ck.AdvectionJacobian_weak(dmom_u_adv_h,h_trial_ref.data()[k*nDOF_trial_element+j],&vel_grad_test_dV[i_nSpace]) +
                        ck.ReactionJacobian_weak(dmom_u_source_h,h_trial_ref.data()[k*nDOF_trial_element+j],vel_test_dV[i]) +
                        ck.SubgridErrorJacobian(dsubgridError_h_h[j],Lstar_h_u[i]) +
                        ck.SubgridErrorJacobian(dsubgridError_u_h[j],Lstar_u_u[i]) +
                        ck.SubgridErrorJacobian(dsubgridError_v_h[j],Lstar_v_u[i]);

                      elementJacobian_u_u[i][j] += ck.MassJacobian_weak(dmom_u_acc_u_t,vel_trial_ref.data()[k*nDOF_trial_element+j],vel_test_dV[i]) +
                        ck.AdvectionJacobian_weak(dmom_u_adv_u,vel_trial_ref.data()[k*nDOF_trial_element+j],&vel_grad_test_dV[i_nSpace]) +
                        ck.SimpleDiffusionJacobian_weak(sdInfo_u_u_rowptr.data(),sdInfo_u_u_colind.data(),mom_u_diff_ten,&vel_grad_trial[j_nSpace],&vel_grad_test_dV[i_nSpace]) +
                        ck.SubgridErrorJacobian(dsubgridError_h_u[j],Lstar_h_u[i]) +
                        ck.SubgridErrorJacobian(dsubgridError_u_u[j],Lstar_u_u[i]) +
                        ck.SubgridErrorJacobian(dsubgridError_v_u[j],Lstar_v_u[i]) +
                        ck.NumericalDiffusionJacobian(q_numDiff_u_last.data()[eN_k],&vel_grad_trial[j_nSpace],&vel_grad_test_dV[i_nSpace]);

                      elementJacobian_u_v[i][j] += ck.AdvectionJacobian_weak(dmom_u_adv_v,vel_trial_ref.data()[k*nDOF_trial_element+j],&vel_grad_test_dV[i_nSpace]) +
                        ck.SimpleDiffusionJacobian_weak(sdInfo_u_v_rowptr.data(),sdInfo_u_v_colind.data(),mom_uv_diff_ten,&vel_grad_trial[j_nSpace],&vel_grad_test_dV[i_nSpace]) +
                        ck.SubgridErrorJacobian(dsubgridError_h_v[j],Lstar_h_u[i]) +
                        ck.SubgridErrorJacobian(dsubgridError_u_v[j],Lstar_u_u[i]) +
                        ck.SubgridErrorJacobian(dsubgridError_v_v[j],Lstar_v_u[i]);

                      //v
                      elementJacobian_v_h[i][j] += ck.MassJacobian_weak(dmom_v_acc_h_t,vel_trial_ref.data()[k*nDOF_trial_element+j],vel_test_dV[i]) +
                        ck.AdvectionJacobian_weak(dmom_v_adv_h,h_trial_ref.data()[k*nDOF_trial_element+j],&vel_grad_test_dV[i_nSpace]) +
                        ck.ReactionJacobian_weak(dmom_v_source_h,h_trial_ref.data()[k*nDOF_trial_element+j],vel_test_dV[i]) +
                        ck.SubgridErrorJacobian(dsubgridError_h_h[j],Lstar_h_v[i]) +
                        ck.SubgridErrorJacobian(dsubgridError_u_h[j],Lstar_u_v[i]) +
                        ck.SubgridErrorJacobian(dsubgridError_v_h[j],Lstar_v_v[i]);

                      elementJacobian_v_u[i][j] += ck.AdvectionJacobian_weak(dmom_v_adv_u,vel_trial_ref.data()[k*nDOF_trial_element+j],&vel_grad_test_dV[i_nSpace]) +
                        ck.SimpleDiffusionJacobian_weak(sdInfo_v_u_rowptr.data(),sdInfo_v_u_colind.data(),mom_vu_diff_ten,&vel_grad_trial[j_nSpace],&vel_grad_test_dV[i_nSpace]) +
                        ck.SubgridErrorJacobian(dsubgridError_h_u[j],Lstar_h_v[i]) +
                        ck.SubgridErrorJacobian(dsubgridError_u_u[j],Lstar_u_v[i]) +
                        ck.SubgridErrorJacobian(dsubgridError_v_u[j],Lstar_v_v[i]);

                      elementJacobian_v_v[i][j] += ck.MassJacobian_weak(dmom_v_acc_v_t,vel_trial_ref.data()[k*nDOF_trial_element+j],vel_test_dV[i]) +
                        ck.AdvectionJacobian_weak(dmom_v_adv_v,vel_trial_ref.data()[k*nDOF_trial_element+j],&vel_grad_test_dV[i_nSpace]) +
                        ck.SimpleDiffusionJacobian_weak(sdInfo_v_v_rowptr.data(),sdInfo_v_v_colind.data(),mom_v_diff_ten,&vel_grad_trial[j_nSpace],&vel_grad_test_dV[i_nSpace]) +
                        ck.SubgridErrorJacobian(dsubgridError_h_v[j],Lstar_h_v[i]) +
                        ck.SubgridErrorJacobian(dsubgridError_u_v[j],Lstar_u_v[i]) +
                        ck.SubgridErrorJacobian(dsubgridError_v_v[j],Lstar_v_v[i]) +
                        ck.NumericalDiffusionJacobian(q_numDiff_v_last.data()[eN_k],&vel_grad_trial[j_nSpace],&vel_grad_test_dV[i_nSpace]);
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
                  globalJacobian.data()[csrRowIndeces_h_h.data()[eN_i] + csrColumnOffsets_h_h.data()[eN_i_j]] += elementJacobian_h_h[i][j];
                  globalJacobian.data()[csrRowIndeces_h_u.data()[eN_i] + csrColumnOffsets_h_u.data()[eN_i_j]] += elementJacobian_h_u[i][j];
                  globalJacobian.data()[csrRowIndeces_h_v.data()[eN_i] + csrColumnOffsets_h_v.data()[eN_i_j]] += elementJacobian_h_v[i][j];

                  globalJacobian.data()[csrRowIndeces_u_h.data()[eN_i] + csrColumnOffsets_u_h.data()[eN_i_j]] += elementJacobian_u_h[i][j];
                  globalJacobian.data()[csrRowIndeces_u_u.data()[eN_i] + csrColumnOffsets_u_u.data()[eN_i_j]] += elementJacobian_u_u[i][j];
                  globalJacobian.data()[csrRowIndeces_u_v.data()[eN_i] + csrColumnOffsets_u_v.data()[eN_i_j]] += elementJacobian_u_v[i][j];

                  globalJacobian.data()[csrRowIndeces_v_h.data()[eN_i] + csrColumnOffsets_v_h.data()[eN_i_j]] += elementJacobian_v_h[i][j];
                  globalJacobian.data()[csrRowIndeces_v_u.data()[eN_i] + csrColumnOffsets_v_u.data()[eN_i_j]] += elementJacobian_v_u[i][j];
                  globalJacobian.data()[csrRowIndeces_v_v.data()[eN_i] + csrColumnOffsets_v_v.data()[eN_i_j]] += elementJacobian_v_v[i][j];
                }//j
            }//i
        }//elements
      std::cout<<"tauSum = "<<tauSum<<std::endl;
      //
      //loop over exterior element boundaries to compute the surface integrals and load them into the global Jacobian
      //
      /* for (proteus_index_t ebNE = 0; ebNE < nExteriorElementBoundaries_global; ebNE++)  */
      /*        {  */
      /*          proteus_index_t ebN = exteriorElementBoundariesArray.data()[ebNE], */
      /*            eN  = elementBoundaryElementsArray.data()[ebN*2+0], */
      /*            eN_nDOF_trial_element = eN*nDOF_trial_element, */
      /*            ebN_local = elementBoundaryLocalElementBoundariesArray.data()[ebN*2+0]; */
      /*          double eps_rho,eps_mu; */
      /*          for  (int kb=0;kb<nQuadraturePoints_elementBoundary;kb++)  */
      /*            {  */
      /*              int ebNE_kb = ebNE*nQuadraturePoints_elementBoundary+kb, */
      /*                ebNE_kb_nSpace = ebNE_kb*nSpace, */
      /*                ebN_local_kb = ebN_local*nQuadraturePoints_elementBoundary+kb, */
      /*                ebN_local_kb_nSpace = ebN_local_kb*nSpace; */

      /*              double h_ext=0.0, */
      /*                u_ext=0.0, */
      /*                v_ext=0.0, */
      /*                grad_h_ext[nSpace], */
      /*                grad_u_ext[nSpace], */
      /*                grad_v_ext[nSpace], */
      /*                mom_u_acc_ext=0.0, */
      /*                dmom_u_acc_u_ext=0.0, */
      /*                mom_v_acc_ext=0.0, */
      /*                dmom_v_acc_v_ext=0.0, */
      /*                mass_adv_ext[nSpace], */
      /*                dmass_adv_u_ext[nSpace], */
      /*                dmass_adv_v_ext[nSpace], */
      /*                mom_u_adv_ext[nSpace], */
      /*                dmom_u_adv_u_ext[nSpace], */
      /*                dmom_u_adv_v_ext[nSpace], */
      /*                mom_v_adv_ext[nSpace], */
      /*                dmom_v_adv_u_ext[nSpace], */
      /*                dmom_v_adv_v_ext[nSpace], */
      /*                mom_u_diff_ten_ext[nSpace], */
      /*                mom_v_diff_ten_ext[nSpace], */
      /*                mom_uv_diff_ten_ext[1], */
      /*                mom_vu_diff_ten_ext[1], */
      /*                mom_u_source_ext=0.0, */
      /*                mom_v_source_ext=0.0, */
      /*                mom_u_ham_ext=0.0, */
      /*                dmom_u_ham_grad_h_ext[nSpace], */
      /*                mom_v_ham_ext=0.0, */
      /*                dmom_v_ham_grad_h_ext[nSpace], */
      /*                dmom_u_adv_h_ext[nSpace], */
      /*                dmom_v_adv_h_ext[nSpace], */
      /*                dflux_mass_u_ext=0.0, */
      /*                dflux_mass_v_ext=0.0, */
      /*                dflux_mom_u_adv_h_ext=0.0, */
      /*                dflux_mom_u_adv_u_ext=0.0, */
      /*                dflux_mom_u_adv_v_ext=0.0, */
      /*                dflux_mom_v_adv_h_ext=0.0, */
      /*                dflux_mom_v_adv_u_ext=0.0, */
      /*                dflux_mom_v_adv_v_ext=0.0, */
      /*                bc_h_ext=0.0, */
      /*                bc_u_ext=0.0, */
      /*                bc_v_ext=0.0, */
      /*                bc_mom_u_acc_ext=0.0, */
      /*                bc_dmom_u_acc_u_ext=0.0, */
      /*                bc_mom_v_acc_ext=0.0, */
      /*                bc_dmom_v_acc_v_ext=0.0, */
      /*                bc_mass_adv_ext[nSpace], */
      /*                bc_dmass_adv_u_ext[nSpace], */
      /*                bc_dmass_adv_v_ext[nSpace], */
      /*                bc_mom_u_adv_ext[nSpace], */
      /*                bc_dmom_u_adv_u_ext[nSpace], */
      /*                bc_dmom_u_adv_v_ext[nSpace], */
      /*                bc_mom_v_adv_ext[nSpace], */
      /*                bc_dmom_v_adv_u_ext[nSpace], */
      /*                bc_dmom_v_adv_v_ext[nSpace], */
      /*                bc_mom_u_diff_ten_ext[nSpace], */
      /*                bc_mom_v_diff_ten_ext[nSpace], */
      /*                bc_mom_uv_diff_ten_ext[1], */
      /*                bc_mom_vu_diff_ten_ext[1], */
      /*                bc_mom_u_source_ext=0.0, */
      /*                bc_mom_v_source_ext=0.0, */
      /*                bc_mom_u_ham_ext=0.0, */
      /*                bc_dmom_u_ham_grad_h_ext[nSpace], */
      /*                bc_mom_v_ham_ext=0.0, */
      /*                bc_dmom_v_ham_grad_h_ext[nSpace], */
      /*                fluxJacobian_h_h[nDOF_trial_element], */
      /*                fluxJacobian_h_u[nDOF_trial_element], */
      /*                fluxJacobian_h_v[nDOF_trial_element], */
      /*                fluxJacobian_u_h[nDOF_trial_element], */
      /*                fluxJacobian_u_u[nDOF_trial_element], */
      /*                fluxJacobian_u_v[nDOF_trial_element], */
      /*                fluxJacobian_v_h[nDOF_trial_element], */
      /*                fluxJacobian_v_u[nDOF_trial_element], */
      /*                fluxJacobian_v_v[nDOF_trial_element], */
      /*                jac_ext[nSpace*nSpace], */
      /*                jacDet_ext, */
      /*                jacInv_ext[nSpace*nSpace], */
      /*                boundaryJac[nSpace*(nSpace-1)], */
      /*                metricTensor[(nSpace-1)*(nSpace-1)], */
      /*                metricTensorDetSqrt, */
      /*                h_grad_trial_trace[nDOF_trial_element*nSpace], */
      /*                vel_grad_trial_trace[nDOF_trial_element*nSpace], */
      /*                dS, */
      /*                h_test_dS[nDOF_test_element], */
      /*                vel_test_dS[nDOF_test_element], */
      /*                normal[3], */
      /*                x_ext,y_ext,xt_ext,yt_ext,integralScaling, */
      /*                G[nSpace*nSpace],G_dd_G,tr_G,h_hhi,h_henalty; */
      /*              ck.calculateMapping_elementBoundary(eN, */
      /*                                                  ebN_local, */
      /*                                                  kb, */
      /*                                                  ebN_local_kb, */
      /*                                                  mesh_dof.data(), */
      /*                                                  mesh_l2g.data(), */
      /*                                                  mesh_trial_trace_ref.data(), */
      /*                                                  mesh_grad_trial_trace_ref.data(), */
      /*                                                  boundaryJac_ref.data(), */
      /*                                                  jac_ext, */
      /*                                                  jacDet_ext, */
      /*                                                  jacInv_ext, */
      /*                                                  boundaryJac, */
      /*                                                  metricTensor, */
      /*                                                  metricTensorDetSqrt, */
      /*                                                  normal_ref.data(), */
      /*                                                  normal, */
      /*                                                  x_ext,y_ext); */
      /*              ck.calculateMappingVelocity_elementBoundary(eN, */
      /*                                                          ebN_local, */
      /*                                                          kb, */
      /*                                                          ebN_local_kb, */
      /*                                                          mesh_velocity_dof.data(), */
      /*                                                          mesh_l2g.data(), */
      /*                                                          mesh_trial_trace_ref.data(), */
      /*                                                          xt_ext,yt_ext, */
      /*                                                          normal, */
      /*                                                          boundaryJac, */
      /*                                                          metricTensor, */
      /*                                                          integralScaling); */
      /*              //xt_ext=0.0;yt_ext=0.0; */
      /*              //std::cout<<"xt_ext "<<xt_ext<<'\t'<<yt_ext<<'\t'<<std::endl; */
      /*              dS = ((1.0-MOVING_DOMAIN)*metricTensorDetSqrt + MOVING_DOMAIN*integralScaling)*dS_ref.data()[kb]; */
      /*              ck.calculateG(jacInv_ext,G,G_dd_G,tr_G); */
      /*              ck.calculateGScale(G,&ebqe_normal_hhi_ext[ebNE_kb_nSpace],h_hhi); */

      /*              eps_rho = epsFact_rho*(useMetrics*h_hhi+(1.0-useMetrics)*elementDiameter.data()[eN]); */
      /*              eps_mu  = epsFact_mu *(useMetrics*h_hhi+(1.0-useMetrics)*elementDiameter.data()[eN]); */

      /*              //compute shape and solution information */
      /*              //shape */
      /*              ck.gradTrialFromRef(&h_grad_trial_trace_ref.data()[ebN_local_kb_nSpace*nDOF_trial_element],jacInv_ext,h_grad_trial_trace); */
      /*              ck.gradTrialFromRef(&vel_grad_trial_trace_ref.data()[ebN_local_kb_nSpace*nDOF_trial_element],jacInv_ext,vel_grad_trial_trace); */
      /*              //solution and gradients   */
      /*              ck.valFromDOF(h_dof.data(),&h_l2g.data()[eN_nDOF_trial_element],&h_trial_trace_ref.data()[ebN_local_kb*nDOF_test_element],h_ext); */
      /*              ck.valFromDOF(u_dof.data(),&vel_l2g.data()[eN_nDOF_trial_element],&vel_trial_trace_ref.data()[ebN_local_kb*nDOF_test_element],u_ext); */
      /*              ck.valFromDOF(v_dof.data(),&vel_l2g.data()[eN_nDOF_trial_element],&vel_trial_trace_ref.data()[ebN_local_kb*nDOF_test_element],v_ext); */
      /*              ck.gradFromDOF(h_dof.data(),&h_l2g.data()[eN_nDOF_trial_element],h_grad_trial_trace,grad_h_ext); */
      /*              ck.gradFromDOF(u_dof.data(),&vel_l2g.data()[eN_nDOF_trial_element],vel_grad_trial_trace,grad_u_ext); */
      /*              ck.gradFromDOF(v_dof.data(),&vel_l2g.data()[eN_nDOF_trial_element],vel_grad_trial_trace,grad_v_ext); */
      /*              //precalculate test function products with integration weights */
      /*              for (int j=0;j<nDOF_trial_element;j++) */
      /*                { */
      /*                  h_test_dS[j] = h_test_trace_ref.data()[ebN_local_kb*nDOF_test_element+j]*dS; */
      /*                  vel_test_dS[j] = vel_test_trace_ref.data()[ebN_local_kb*nDOF_test_element+j]*dS; */
      /*                } */
      /*              // // */
      /*              // //debugging section for finite element calculations on exterior */
      /*              // // */
      /*              // std::cout<<"ebNE = "<<ebNE<<" kb = "<<kb<<std::endl; */
      /*              // for (int j=0;j<nDOF_trial_element;j++) */
      /*              //   { */
      /*              //     std::cout<<"h_trial_trace["<<j<<"] "<<h_trial_trace_ref.data()[ebN_local_kb*nDOF_trial_element+j]<<std::endl; */
      /*              //     std::cout<<"vel_trial_trace["<<j<<"] "<<vel_trial_trace_ref.data()[ebN_local_kb*nDOF_trial_element+j]<<std::endl; */
      /*              //     std::cout<<"h_test_dS["<<j<<"] "<<h_test_dS[j]<<std::endl; */
      /*              //     std::cout<<"vel_test_dS["<<j<<"] "<<vel_test_dS[j]<<std::endl; */
      /*              //     for (int I=0;I<nSpace;I++) */
      /*              //        { */
      /*              //          std::cout<<"h_grad_trial_trace["<<j<<","<<I<<"] "<<h_grad_trial_trace[j*nSpace+I]<<std::endl; */
      /*              //          std::cout<<"vel_grad_trial_trace["<<j<<","<<I<<"] "<<vel_grad_trial_trace[j*nSpace+I]<<std::endl; */
      /*              //        } */
      /*              //   } */
      /*              // std::cout<<"h_ext "<<h_ext<<std::endl; */
      /*              // std::cout<<"u_ext "<<u_ext<<std::endl; */
      /*              // std::cout<<"v_ext "<<v_ext<<std::endl; */
      /*              // for(int I=0;I<nSpace;I++) */
      /*              //   { */
      /*              //     std::cout<<"grad_h_ext["<<I<<"] "<<grad_h_ext[I]<<std::endl; */
      /*              //     std::cout<<"grad_u_ext["<<I<<"] "<<grad_u_ext[I]<<std::endl; */
      /*              //     std::cout<<"grad_v_ext["<<I<<"] "<<grad_v_ext[I]<<std::endl; */
      /*              //   } */
      /*              // */
      /*              //load the boundary values */
      /*              // */
      /*              bc_h_ext = isDOFBoundary_h.data()[ebNE_kb]*ebqe_bc_h_ext.data()[ebNE_kb]+(1-isDOFBoundary_h.data()[ebNE_kb])*h_ext; */
      /*              bc_u_ext = isDOFBoundary_u.data()[ebNE_kb]*ebqe_bc_u_ext.data()[ebNE_kb]+(1-isDOFBoundary_u.data()[ebNE_kb])*u_ext; */
      /*              bc_v_ext = isDOFBoundary_v.data()[ebNE_kb]*ebqe_bc_v_ext.data()[ebNE_kb]+(1-isDOFBoundary_v.data()[ebNE_kb])*v_ext; */
      /*              //  */
      /*              //calculate the internal and external trace of the pde coefficients  */
      /*              //  */
      /*              //cek debug */
      /*              //eps_rho=0.1; */
      /*              //eps_mu=0.1; */
      /*              evaluateCoefficients(eps_rho, */
      /*                                   eps_mu, */
      /*                                   sigma, */
      /*                                   rho_0, */
      /*                                   nu_0, */
      /*                                   rho_1, */
      /*                                   nu_1, */
      /*                                   g, */
      /*                                   ebqe_hhi_ext[ebNE_kb], */
      /*                                   &ebqe_normal_hhi_ext[ebNE_kb_nSpace], */
      /*                                   ebqe_kappa_hhi_ext[ebNE_kb], */
      /*                                   h_ext, */
      /*                                   grad_h_ext, */
      /*                                   u_ext, */
      /*                                   v_ext, */
      /*                                   mom_u_acc_ext, */
      /*                                   dmom_u_acc_u_ext, */
      /*                                   mom_v_acc_ext, */
      /*                                   dmom_v_acc_v_ext, */
      /*                                   mass_adv_ext, */
      /*                                   dmass_adv_u_ext, */
      /*                                   dmass_adv_v_ext, */
      /*                                   mom_u_adv_ext, */
      /*                                   dmom_u_adv_u_ext, */
      /*                                   dmom_u_adv_v_ext, */
      /*                                   mom_v_adv_ext, */
      /*                                   dmom_v_adv_u_ext, */
      /*                                   dmom_v_adv_v_ext, */
      /*                                   mom_u_diff_ten_ext, */
      /*                                   mom_v_diff_ten_ext, */
      /*                                   mom_uv_diff_ten_ext, */
      /*                                   mom_vu_diff_ten_ext, */
      /*                                   mom_u_source_ext, */
      /*                                   mom_v_source_ext, */
      /*                                   mom_u_ham_ext, */
      /*                                   dmom_u_ham_grad_h_ext, */
      /*                                   mom_v_ham_ext, */
      /*                                   dmom_v_ham_grad_h_ext);           */
      /*              evaluateCoefficients(eps_rho, */
      /*                                   eps_mu, */
      /*                                   sigma, */
      /*                                   rho_0, */
      /*                                   nu_0, */
      /*                                   rho_1, */
      /*                                   nu_1, */
      /*                                   g, */
      /*                                   ebqe_hhi_ext[ebNE_kb], */
      /*                                   &ebqe_normal_hhi_ext[ebNE_kb_nSpace], */
      /*                                   ebqe_kappa_hhi_ext[ebNE_kb], */
      /*                                   bc_h_ext, */
      /*                                   grad_h_ext, //cek shouldn't be used */
      /*                                   bc_u_ext, */
      /*                                   bc_v_ext, */
      /*                                   bc_mom_u_acc_ext, */
      /*                                   bc_dmom_u_acc_u_ext, */
      /*                                   bc_mom_v_acc_ext, */
      /*                                   bc_dmom_v_acc_v_ext, */
      /*                                   bc_mass_adv_ext, */
      /*                                   bc_dmass_adv_u_ext, */
      /*                                   bc_dmass_adv_v_ext, */
      /*                                   bc_mom_u_adv_ext, */
      /*                                   bc_dmom_u_adv_u_ext, */
      /*                                   bc_dmom_u_adv_v_ext, */
      /*                                   bc_mom_v_adv_ext, */
      /*                                   bc_dmom_v_adv_u_ext, */
      /*                                   bc_dmom_v_adv_v_ext, */
      /*                                   bc_mom_u_diff_ten_ext, */
      /*                                   bc_mom_v_diff_ten_ext, */
      /*                                   bc_mom_uv_diff_ten_ext, */
      /*                                   bc_mom_vu_diff_ten_ext, */
      /*                                   bc_mom_u_source_ext, */
      /*                                   bc_mom_v_source_ext, */
      /*                                   bc_mom_u_ham_ext, */
      /*                                   bc_dmom_u_ham_grad_h_ext, */
      /*                                   bc_mom_v_ham_ext, */
      /*                                   bc_dmom_v_ham_grad_h_ext);           */
      /*              // */
      /*              //moving domain */
      /*              // */
      /*              mass_adv_ext[0] -= MOVING_DOMAIN*xt_ext; */
      /*              mass_adv_ext[1] -= MOVING_DOMAIN*yt_ext; */

      /*              mom_u_adv_ext[0] -= MOVING_DOMAIN*mom_u_acc_ext*xt_ext; */
      /*              mom_u_adv_ext[1] -= MOVING_DOMAIN*mom_u_acc_ext*yt_ext; */
      /*              dmom_u_adv_u_ext[0] -= MOVING_DOMAIN*dmom_u_acc_u_ext*xt_ext; */
      /*              dmom_u_adv_u_ext[1] -= MOVING_DOMAIN*dmom_u_acc_u_ext*yt_ext; */

      /*              mom_v_adv_ext[0] -= MOVING_DOMAIN*mom_v_acc_ext*xt_ext; */
      /*              mom_v_adv_ext[1] -= MOVING_DOMAIN*mom_v_acc_ext*yt_ext; */
      /*              dmom_v_adv_v_ext[0] -= MOVING_DOMAIN*dmom_v_acc_v_ext*xt_ext; */
      /*              dmom_v_adv_v_ext[1] -= MOVING_DOMAIN*dmom_v_acc_v_ext*yt_ext; */

      /*              //moving domain bc's */
      /*              bc_mom_u_adv_ext[0] -= MOVING_DOMAIN*bc_mom_u_acc_ext*xt_ext; */
      /*              bc_mom_u_adv_ext[1] -= MOVING_DOMAIN*bc_mom_u_acc_ext*yt_ext; */

      /*              bc_mom_v_adv_ext[0] -= MOVING_DOMAIN*bc_mom_v_acc_ext*xt_ext; */
      /*              bc_mom_v_adv_ext[1] -= MOVING_DOMAIN*bc_mom_v_acc_ext*yt_ext; */

      /*              //  */
      /*              //calculate the numerical fluxes  */
      /*              //  */
      /*              exteriorNumericalAdvectiveFluxDerivatives(isDOFBoundary_h.data()[ebNE_kb], */
      /*                                                        isDOFBoundary_u.data()[ebNE_kb], */
      /*                                                        isDOFBoundary_v.data()[ebNE_kb], */
      /*                                                        isAdvectiveFluxBoundary_h.data()[ebNE_kb], */
      /*                                                        isAdvectiveFluxBoundary_u.data()[ebNE_kb], */
      /*                                                        isAdvectiveFluxBoundary_v.data()[ebNE_kb], */
      /*                                                        dmom_u_ham_grad_h_ext[0],//=1/rho */
      /*                                                        normal, */
      /*                                                        bc_h_ext, */
      /*                                                        bc_mass_adv_ext, */
      /*                                                        bc_mom_u_adv_ext, */
      /*                                                        bc_mom_v_adv_ext, */
      /*                                                        ebqe_bc_flux_mass_ext.data()[ebNE_kb], */
      /*                                                        ebqe_bc_flux_mom_u_adv_ext.data()[ebNE_kb], */
      /*                                                        ebqe_bc_flux_mom_v_adv_ext.data()[ebNE_kb], */
      /*                                                        h_ext, */
      /*                                                        mass_adv_ext, */
      /*                                                        mom_u_adv_ext, */
      /*                                                        mom_v_adv_ext, */
      /*                                                        dmass_adv_u_ext, */
      /*                                                        dmass_adv_v_ext, */
      /*                                                        dmom_u_adv_h_ext, */
      /*                                                        dmom_u_adv_u_ext, */
      /*                                                        dmom_u_adv_v_ext, */
      /*                                                        dmom_v_adv_h_ext, */
      /*                                                        dmom_v_adv_u_ext, */
      /*                                                        dmom_v_adv_v_ext, */
      /*                                                        dflux_mass_u_ext, */
      /*                                                        dflux_mass_v_ext, */
      /*                                                        dflux_mom_u_adv_h_ext, */
      /*                                                        dflux_mom_u_adv_u_ext, */
      /*                                                        dflux_mom_u_adv_v_ext, */
      /*                                                        dflux_mom_v_adv_h_ext, */
      /*                                                        dflux_mom_v_adv_u_ext, */
      /*                                                        dflux_mom_v_adv_v_ext); */
      /*              // */
      /*              //calculate the flux jacobian */
      /*              // */
      /*              ck.calculateGScale(G,normal,h_henalty); */
      /*              h_henalty = 10.0/h_henalty; */
      /*              //cek debug, do it the old way */
      /*              h_henalty = 100.0/elementDiameter.data()[eN]; */
      /*              for (int j=0;j<nDOF_trial_element;j++) */
      /*                { */
      /*                  int j_nSpace = j*nSpace,ebN_local_kb_j=ebN_local_kb*nDOF_trial_element+j; */
      /*                  //cek debug */
      /*                  //ebqe_henalty_ext[ebNE_kb] = 10.0; */
      /*                  // */
      /*                  //cek todo add full stress on boundaries */

      /*                  fluxJacobian_h_h[j]=0.0; */
      /*                  fluxJacobian_h_u[j]=ck.ExteriorNumericalAdvectiveFluxJacobian(dflux_mass_u_ext,vel_trial_trace_ref.data()[ebN_local_kb_j]); */
      /*                  fluxJacobian_h_v[j]=ck.ExteriorNumericalAdvectiveFluxJacobian(dflux_mass_v_ext,vel_trial_trace_ref.data()[ebN_local_kb_j]); */

      /*                  fluxJacobian_u_h[j]=ck.ExteriorNumericalAdvectiveFluxJacobian(dflux_mom_u_adv_h_ext,h_trial_trace_ref.data()[ebN_local_kb_j]); */
      /*                  fluxJacobian_u_u[j]=ck.ExteriorNumericalAdvectiveFluxJacobian(dflux_mom_u_adv_u_ext,vel_trial_trace_ref.data()[ebN_local_kb_j]) + */
      /*                    ExteriorNumericalDiffusiveFluxJacobian(eps_rho, */
      /*                                                           ebqe_hhi_ext[ebNE_kb], */
      /*                                                           sdInfo_u_u_rowptr.data(), */
      /*                                                           sdInfo_u_u_colind.data(), */
      /*                                                           isDOFBoundary_u.data()[ebNE_kb], */
      /*                                                           normal, */
      /*                                                           mom_u_diff_ten_ext, */
      /*                                                           vel_trial_trace_ref.data()[ebN_local_kb_j], */
      /*                                                           &vel_grad_trial_trace[j_nSpace], */
      /*                                                           h_henalty);//ebqe_henalty_ext[ebNE_kb]); */
      /*                  fluxJacobian_u_v[j]=ck.ExteriorNumericalAdvectiveFluxJacobian(dflux_mom_u_adv_v_ext,vel_trial_trace_ref.data()[ebN_local_kb_j]); */

      /*                  fluxJacobian_v_h[j]=ck.ExteriorNumericalAdvectiveFluxJacobian(dflux_mom_v_adv_h_ext,h_trial_trace_ref.data()[ebN_local_kb_j]); */
      /*                  fluxJacobian_v_u[j]=ck.ExteriorNumericalAdvectiveFluxJacobian(dflux_mom_v_adv_u_ext,vel_trial_trace_ref.data()[ebN_local_kb_j]); */
      /*                  fluxJacobian_v_v[j]=ck.ExteriorNumericalAdvectiveFluxJacobian(dflux_mom_v_adv_v_ext,vel_trial_trace_ref.data()[ebN_local_kb_j]) + */
      /*                    ExteriorNumericalDiffusiveFluxJacobian(eps_rho, */
      /*                                                           ebqe_hhi_ext[ebNE_kb], */
      /*                                                           sdInfo_v_v_rowptr.data(), */
      /*                                                           sdInfo_v_v_colind.data(), */
      /*                                                           isDOFBoundary_v.data()[ebNE_kb], */
      /*                                                           normal, */
      /*                                                           mom_v_diff_ten_ext, */
      /*                                                           vel_trial_trace_ref.data()[ebN_local_kb_j], */
      /*                                                           &vel_grad_trial_trace[j_nSpace], */
      /*                                                           h_henalty);//ebqe_henalty_ext[ebNE_kb]); */

      /*                  fluxJacobian_h_h[j]=0.0; */
      /*                  fluxJacobian_h_u[j]=ck.ExteriorNumericalAdvectiveFluxJacobian(dflux_mass_u_ext,vel_trial_trace_ref.data()[ebN_local_kb_j]); */
      /*                  fluxJacobian_h_v[j]=ck.ExteriorNumericalAdvectiveFluxJacobian(dflux_mass_v_ext,vel_trial_trace_ref.data()[ebN_local_kb_j]); */

      /*                  fluxJacobian_u_h[j]=ck.ExteriorNumericalAdvectiveFluxJacobian(dflux_mom_u_adv_h_ext,h_trial_trace_ref.data()[ebN_local_kb_j]); */
      /*                  fluxJacobian_u_u[j]=ck.ExteriorNumericalAdvectiveFluxJacobian(dflux_mom_u_adv_u_ext,vel_trial_trace_ref.data()[ebN_local_kb_j]) + */
      /*                    ExteriorNumericalDiffusiveFluxJacobian(eps_rho, */
      /*                                                           ebqe_hhi_ext[ebNE_kb], */
      /*                                                           sdInfo_u_u_rowptr.data(), */
      /*                                                           sdInfo_u_u_colind.data(), */
      /*                                                           isDOFBoundary_u.data()[ebNE_kb], */
      /*                                                           normal, */
      /*                                                           mom_u_diff_ten_ext, */
      /*                                                           vel_trial_trace_ref.data()[ebN_local_kb_j], */
      /*                                                           &vel_grad_trial_trace[j_nSpace], */
      /*                                                           h_henalty);//ebqe_henalty_ext[ebNE_kb]); */
      /*                  fluxJacobian_u_v[j]=ck.ExteriorNumericalAdvectiveFluxJacobian(dflux_mom_u_adv_v_ext,vel_trial_trace_ref.data()[ebN_local_kb_j]); */

      /*                  fluxJacobian_v_h[j]=ck.ExteriorNumericalAdvectiveFluxJacobian(dflux_mom_v_adv_h_ext,h_trial_trace_ref.data()[ebN_local_kb_j]); */
      /*                  fluxJacobian_v_u[j]=ck.ExteriorNumericalAdvectiveFluxJacobian(dflux_mom_v_adv_u_ext,vel_trial_trace_ref.data()[ebN_local_kb_j]); */
      /*                  fluxJacobian_v_v[j]=ck.ExteriorNumericalAdvectiveFluxJacobian(dflux_mom_v_adv_v_ext,vel_trial_trace_ref.data()[ebN_local_kb_j]) + */
      /*                    ExteriorNumericalDiffusiveFluxJacobian(eps_rho, */
      /*                                                           ebqe_hhi_ext[ebNE_kb], */
      /*                                                           sdInfo_v_v_rowptr.data(), */
      /*                                                           sdInfo_v_v_colind.data(), */
      /*                                                           isDOFBoundary_v.data()[ebNE_kb], */
      /*                                                           normal, */
      /*                                                           mom_v_diff_ten_ext, */
      /*                                                           vel_trial_trace_ref.data()[ebN_local_kb_j], */
      /*                                                           &vel_grad_trial_trace[j_nSpace], */
      /*                                                           h_henalty);//ebqe_henalty_ext[ebNE_kb]); */
      /*                  // //cek debug */
      /*                  // fluxJacobian_h_h[j]=0.0; */
      /*                  // fluxJacobian_h_u[j]=0.0; */
      /*                  // fluxJacobian_h_v[j]=0.0; */

      /*                  // fluxJacobian_u_h[j]=0.0; */
      /*                  // fluxJacobian_u_u[j]=0.0; */
      /*                  // fluxJacobian_u_v[j]=0.0; */

      /*                  // fluxJacobian_v_h[j]=0.0; */
      /*                  // fluxJacobian_v_u[j]=0.0; */
      /*                  // fluxJacobian_v_v[j]=0.0; */

      /*                  // //cek debug */
      /*                }//j */
      /*              // */
      /*              //update the global Jacobian from the flux Jacobian */
      /*              // */
      /*              for (int i=0;i<nDOF_test_element;i++) */
      /*                { */
      /*                  proteus_index_t eN_i = eN*nDOF_test_element+i; */
      /*                  for (int j=0;j<nDOF_trial_element;j++) */
      /*                    { */
      /*                      proteus_index_t ebN_i_j = ebN*4*nDOF_test_X_trial_element + i*nDOF_trial_element + j; */

      /*                      globalJacobian.data()[csrRowIndeces_h_h.data()[eN_i] + csrColumnOffsets_eb_h_h.data()[ebN_i_j]] += fluxJacobian_h_h[j]*h_test_dS[i]; */
      /*                      globalJacobian.data()[csrRowIndeces_h_u.data()[eN_i] + csrColumnOffsets_eb_h_u.data()[ebN_i_j]] += fluxJacobian_h_u[j]*h_test_dS[i]; */
      /*                      globalJacobian.data()[csrRowIndeces_h_v.data()[eN_i] + csrColumnOffsets_eb_h_v.data()[ebN_i_j]] += fluxJacobian_h_v[j]*h_test_dS[i]; */

      /*                      globalJacobian.data()[csrRowIndeces_u_h.data()[eN_i] + csrColumnOffsets_eb_u_h.data()[ebN_i_j]] += fluxJacobian_u_h[j]*vel_test_dS[i]; */
      /*                      globalJacobian.data()[csrRowIndeces_u_u.data()[eN_i] + csrColumnOffsets_eb_u_u.data()[ebN_i_j]] += fluxJacobian_u_u[j]*vel_test_dS[i]; */
      /*                      globalJacobian.data()[csrRowIndeces_u_v.data()[eN_i] + csrColumnOffsets_eb_u_v.data()[ebN_i_j]] += fluxJacobian_u_v[j]*vel_test_dS[i]; */

      /*                      globalJacobian.data()[csrRowIndeces_v_h.data()[eN_i] + csrColumnOffsets_eb_v_h.data()[ebN_i_j]] += fluxJacobian_v_h[j]*vel_test_dS[i]; */
      /*                      globalJacobian.data()[csrRowIndeces_v_u.data()[eN_i] + csrColumnOffsets_eb_v_u.data()[ebN_i_j]] += fluxJacobian_v_u[j]*vel_test_dS[i]; */
      /*                      globalJacobian.data()[csrRowIndeces_v_v.data()[eN_i] + csrColumnOffsets_eb_v_v.data()[ebN_i_j]] += fluxJacobian_v_v[j]*vel_test_dS[i]; */

      /*                    }//j */
      /*                }//i */
      /*              // //debug */
      /*              // std::cout<<"flux jacobian ebNE "<<ebNE<<" kb "<<kb<<std::endl; */
      /*              // for (int i=0;i<nDOF_test_element;i++) */
      /*              //   { */
      /*              //     for (int j=0;j<nDOF_trial_element;j++) */
      /*              //        { */
      /*              //          std::cout<< fluxJacobian_h_h[j]*h_test_dS[i]<<std::endl; */
      /*              //          std::cout<< fluxJacobian_h_u[j]*h_test_dS[i]<<std::endl; */
      /*              //          std::cout<< fluxJacobian_h_v[j]*h_test_dS[i]<<std::endl; */

      /*              //          std::cout<< fluxJacobian_u_h[j]*vel_test_dS[i]<<std::endl; */
      /*              //          std::cout<< fluxJacobian_u_u[j]*vel_test_dS[i]<<std::endl; */
      /*              //          std::cout<< fluxJacobian_u_v[j]*vel_test_dS[i]<<std::endl; */

      /*              //          std::cout<< fluxJacobian_v_h[j]*vel_test_dS[i]<<std::endl; */
      /*              //          std::cout<< fluxJacobian_v_u[j]*vel_test_dS[i]<<std::endl; */
      /*              //          std::cout<< fluxJacobian_v_v[j]*vel_test_dS[i]<<std::endl; */
      /*              //        }//j */
      /*              //   }//i */
      /*            }//kb */
      /*        }//ebNE */
    }//computeJacobian
    void calculateJacobian_supg(arguments_dict& args)
    {
        proteus::pyarray<double>& mesh_trial_ref = args.array<double>("mesh_trial_ref");
        proteus::pyarray<double>& mesh_grad_trial_ref = args.array<double>("mesh_grad_trial_ref");
        proteus::pyarray<double>& mesh_dof = args.array<double>("mesh_dof");
        proteus::pyarray<double>& mesh_velocity_dof = args.array<double>("mesh_velocity_dof");
        double MOVING_DOMAIN = args.scalar<double>("MOVING_DOMAIN");
        proteus::pyarray<proteus_index_t>& mesh_l2g = args.array<proteus_index_t>("mesh_l2g");
        proteus::pyarray<double>& dV_ref = args.array<double>("dV_ref");
        proteus::pyarray<double>& h_trial_ref = args.array<double>("h_trial_ref");
        proteus::pyarray<double>& h_grad_trial_ref = args.array<double>("h_grad_trial_ref");
        proteus::pyarray<double>& h_test_ref = args.array<double>("h_test_ref");
        proteus::pyarray<double>& h_grad_test_ref = args.array<double>("h_grad_test_ref");
        proteus::pyarray<double>& vel_trial_ref = args.array<double>("vel_trial_ref");
        proteus::pyarray<double>& vel_grad_trial_ref = args.array<double>("vel_grad_trial_ref");
        proteus::pyarray<double>& vel_test_ref = args.array<double>("vel_test_ref");
        proteus::pyarray<double>& vel_grad_test_ref = args.array<double>("vel_grad_test_ref");
        proteus::pyarray<double>& mesh_trial_trace_ref = args.array<double>("mesh_trial_trace_ref");
        proteus::pyarray<double>& mesh_grad_trial_trace_ref = args.array<double>("mesh_grad_trial_trace_ref");
        proteus::pyarray<double>& dS_ref = args.array<double>("dS_ref");
        proteus::pyarray<double>& h_trial_trace_ref = args.array<double>("h_trial_trace_ref");
        proteus::pyarray<double>& h_grad_trial_trace_ref = args.array<double>("h_grad_trial_trace_ref");
        proteus::pyarray<double>& h_test_trace_ref = args.array<double>("h_test_trace_ref");
        proteus::pyarray<double>& h_grad_test_trace_ref = args.array<double>("h_grad_test_trace_ref");
        proteus::pyarray<double>& vel_trial_trace_ref = args.array<double>("vel_trial_trace_ref");
        proteus::pyarray<double>& vel_grad_trial_trace_ref = args.array<double>("vel_grad_trial_trace_ref");
        proteus::pyarray<double>& vel_test_trace_ref = args.array<double>("vel_test_trace_ref");
        proteus::pyarray<double>& vel_grad_test_trace_ref = args.array<double>("vel_grad_test_trace_ref");
        proteus::pyarray<double>& normal_ref = args.array<double>("normal_ref");
        proteus::pyarray<double>& boundaryJac_ref = args.array<double>("boundaryJac_ref");
        proteus::pyarray<double>& elementDiameter = args.array<double>("elementDiameter");
        proteus_index_t nElements_global = args.scalar<proteus_index_t>("nElements_global");
        double useRBLES = args.scalar<double>("useRBLES");
        double useMetrics = args.scalar<double>("useMetrics");
        double alphaBDF = args.scalar<double>("alphaBDF");
        double nu = args.scalar<double>("nu");
        double g = args.scalar<double>("g");
        proteus::pyarray<proteus_index_t>& h_l2g = args.array<proteus_index_t>("h_l2g");
        proteus::pyarray<proteus_index_t>& vel_l2g = args.array<proteus_index_t>("vel_l2g");
        proteus::pyarray<double>& b_dof = args.array<double>("b_dof");
        proteus::pyarray<double>& h_dof = args.array<double>("h_dof");
        proteus::pyarray<double>& u_dof = args.array<double>("u_dof");
        proteus::pyarray<double>& v_dof = args.array<double>("v_dof");
        proteus::pyarray<double>& h_dof_sge = args.array<double>("h_dof_sge");
        proteus::pyarray<double>& u_dof_sge = args.array<double>("u_dof_sge");
        proteus::pyarray<double>& v_dof_sge = args.array<double>("v_dof_sge");
        proteus::pyarray<double>& q_mass_acc_beta_bdf = args.array<double>("q_mass_acc_beta_bdf");
        proteus::pyarray<double>& q_mom_u_acc_beta_bdf = args.array<double>("q_mom_u_acc_beta_bdf");
        proteus::pyarray<double>& q_mom_v_acc_beta_bdf = args.array<double>("q_mom_v_acc_beta_bdf");
        proteus::pyarray<double>& q_velocity_sge = args.array<double>("q_velocity_sge");
        proteus::pyarray<double>& q_cfl = args.array<double>("q_cfl");
        proteus::pyarray<double>& q_numDiff_h_last = args.array<double>("q_numDiff_h_last");
        proteus::pyarray<double>& q_numDiff_u_last = args.array<double>("q_numDiff_u_last");
        proteus::pyarray<double>& q_numDiff_v_last = args.array<double>("q_numDiff_v_last");
        proteus::pyarray<proteus_csr_index_t>& sdInfo_u_u_rowptr = args.array<proteus_csr_index_t>("sdInfo_u_u_rowptr");
        proteus::pyarray<proteus_csr_index_t>& sdInfo_u_u_colind = args.array<proteus_csr_index_t>("sdInfo_u_u_colind");
        proteus::pyarray<proteus_csr_index_t>& sdInfo_u_v_rowptr = args.array<proteus_csr_index_t>("sdInfo_u_v_rowptr");
        proteus::pyarray<proteus_csr_index_t>& sdInfo_u_v_colind = args.array<proteus_csr_index_t>("sdInfo_u_v_colind");
        proteus::pyarray<proteus_csr_index_t>& sdInfo_v_v_rowptr = args.array<proteus_csr_index_t>("sdInfo_v_v_rowptr");
        proteus::pyarray<proteus_csr_index_t>& sdInfo_v_v_colind = args.array<proteus_csr_index_t>("sdInfo_v_v_colind");
        proteus::pyarray<proteus_csr_index_t>& sdInfo_v_u_rowptr = args.array<proteus_csr_index_t>("sdInfo_v_u_rowptr");
        proteus::pyarray<proteus_csr_index_t>& sdInfo_v_u_colind = args.array<proteus_csr_index_t>("sdInfo_v_u_colind");
        proteus::pyarray<proteus_csr_index_t>& csrRowIndeces_h_h = args.array<proteus_csr_index_t>("csrRowIndeces_h_h");
        proteus::pyarray<proteus_csr_index_t>& csrColumnOffsets_h_h = args.array<proteus_csr_index_t>("csrColumnOffsets_h_h");
        proteus::pyarray<proteus_csr_index_t>& csrRowIndeces_h_u = args.array<proteus_csr_index_t>("csrRowIndeces_h_u");
        proteus::pyarray<proteus_csr_index_t>& csrColumnOffsets_h_u = args.array<proteus_csr_index_t>("csrColumnOffsets_h_u");
        proteus::pyarray<proteus_csr_index_t>& csrRowIndeces_h_v = args.array<proteus_csr_index_t>("csrRowIndeces_h_v");
        proteus::pyarray<proteus_csr_index_t>& csrColumnOffsets_h_v = args.array<proteus_csr_index_t>("csrColumnOffsets_h_v");
        proteus::pyarray<proteus_csr_index_t>& csrRowIndeces_u_h = args.array<proteus_csr_index_t>("csrRowIndeces_u_h");
        proteus::pyarray<proteus_csr_index_t>& csrColumnOffsets_u_h = args.array<proteus_csr_index_t>("csrColumnOffsets_u_h");
        proteus::pyarray<proteus_csr_index_t>& csrRowIndeces_u_u = args.array<proteus_csr_index_t>("csrRowIndeces_u_u");
        proteus::pyarray<proteus_csr_index_t>& csrColumnOffsets_u_u = args.array<proteus_csr_index_t>("csrColumnOffsets_u_u");
        proteus::pyarray<proteus_csr_index_t>& csrRowIndeces_u_v = args.array<proteus_csr_index_t>("csrRowIndeces_u_v");
        proteus::pyarray<proteus_csr_index_t>& csrColumnOffsets_u_v = args.array<proteus_csr_index_t>("csrColumnOffsets_u_v");
        proteus::pyarray<proteus_csr_index_t>& csrRowIndeces_v_h = args.array<proteus_csr_index_t>("csrRowIndeces_v_h");
        proteus::pyarray<proteus_csr_index_t>& csrColumnOffsets_v_h = args.array<proteus_csr_index_t>("csrColumnOffsets_v_h");
        proteus::pyarray<proteus_csr_index_t>& csrRowIndeces_v_u = args.array<proteus_csr_index_t>("csrRowIndeces_v_u");
        proteus::pyarray<proteus_csr_index_t>& csrColumnOffsets_v_u = args.array<proteus_csr_index_t>("csrColumnOffsets_v_u");
        proteus::pyarray<proteus_csr_index_t>& csrRowIndeces_v_v = args.array<proteus_csr_index_t>("csrRowIndeces_v_v");
        proteus::pyarray<proteus_csr_index_t>& csrColumnOffsets_v_v = args.array<proteus_csr_index_t>("csrColumnOffsets_v_v");
        proteus::pyarray<double>& globalJacobian = args.array<double>("globalJacobian");
        proteus_index_t nExteriorElementBoundaries_global = args.scalar<proteus_index_t>("nExteriorElementBoundaries_global");
        proteus::pyarray<proteus_index_t>& exteriorElementBoundariesArray = args.array<proteus_index_t>("exteriorElementBoundariesArray");
        proteus::pyarray<proteus_index_t>& elementBoundaryElementsArray = args.array<proteus_index_t>("elementBoundaryElementsArray");
        proteus::pyarray<int>& elementBoundaryLocalElementBoundariesArray = args.array<int>("elementBoundaryLocalElementBoundariesArray");
        proteus::pyarray<int>& isDOFBoundary_h = args.array<int>("isDOFBoundary_h");
        proteus::pyarray<int>& isDOFBoundary_u = args.array<int>("isDOFBoundary_u");
        proteus::pyarray<int>& isDOFBoundary_v = args.array<int>("isDOFBoundary_v");
        proteus::pyarray<int>& isAdvectiveFluxBoundary_h = args.array<int>("isAdvectiveFluxBoundary_h");
        proteus::pyarray<int>& isAdvectiveFluxBoundary_u = args.array<int>("isAdvectiveFluxBoundary_u");
        proteus::pyarray<int>& isAdvectiveFluxBoundary_v = args.array<int>("isAdvectiveFluxBoundary_v");
        proteus::pyarray<int>& isDiffusiveFluxBoundary_u = args.array<int>("isDiffusiveFluxBoundary_u");
        proteus::pyarray<int>& isDiffusiveFluxBoundary_v = args.array<int>("isDiffusiveFluxBoundary_v");
        proteus::pyarray<double>& ebqe_bc_h_ext = args.array<double>("ebqe_bc_h_ext");
        proteus::pyarray<double>& ebqe_bc_flux_mass_ext = args.array<double>("ebqe_bc_flux_mass_ext");
        proteus::pyarray<double>& ebqe_bc_flux_mom_u_adv_ext = args.array<double>("ebqe_bc_flux_mom_u_adv_ext");
        proteus::pyarray<double>& ebqe_bc_flux_mom_v_adv_ext = args.array<double>("ebqe_bc_flux_mom_v_adv_ext");
        proteus::pyarray<double>& ebqe_bc_u_ext = args.array<double>("ebqe_bc_u_ext");
        proteus::pyarray<double>& ebqe_bc_flux_u_diff_ext = args.array<double>("ebqe_bc_flux_u_diff_ext");
        proteus::pyarray<double>& ebqe_penalty_ext = args.array<double>("ebqe_penalty_ext");
        proteus::pyarray<double>& ebqe_bc_v_ext = args.array<double>("ebqe_bc_v_ext");
        proteus::pyarray<double>& ebqe_bc_flux_v_diff_ext = args.array<double>("ebqe_bc_flux_v_diff_ext");
        proteus::pyarray<proteus_csr_index_t>& csrColumnOffsets_eb_h_h = args.array<proteus_csr_index_t>("csrColumnOffsets_eb_h_h");
        proteus::pyarray<proteus_csr_index_t>& csrColumnOffsets_eb_h_u = args.array<proteus_csr_index_t>("csrColumnOffsets_eb_h_u");
        proteus::pyarray<proteus_csr_index_t>& csrColumnOffsets_eb_h_v = args.array<proteus_csr_index_t>("csrColumnOffsets_eb_h_v");
        proteus::pyarray<proteus_csr_index_t>& csrColumnOffsets_eb_u_h = args.array<proteus_csr_index_t>("csrColumnOffsets_eb_u_h");
        proteus::pyarray<proteus_csr_index_t>& csrColumnOffsets_eb_u_u = args.array<proteus_csr_index_t>("csrColumnOffsets_eb_u_u");
        proteus::pyarray<proteus_csr_index_t>& csrColumnOffsets_eb_u_v = args.array<proteus_csr_index_t>("csrColumnOffsets_eb_u_v");
        proteus::pyarray<proteus_csr_index_t>& csrColumnOffsets_eb_v_h = args.array<proteus_csr_index_t>("csrColumnOffsets_eb_v_h");
        proteus::pyarray<proteus_csr_index_t>& csrColumnOffsets_eb_v_u = args.array<proteus_csr_index_t>("csrColumnOffsets_eb_v_u");
        proteus::pyarray<proteus_csr_index_t>& csrColumnOffsets_eb_v_v = args.array<proteus_csr_index_t>("csrColumnOffsets_eb_v_v");
      //
      //loop over elements to compute volume integrals and load them into the element Jacobians and global Jacobian
      //
      for(proteus_index_t eN=0;eN<nElements_global;eN++)
        {
          double  elementJacobian_h_h[nDOF_test_element][nDOF_trial_element],
            elementJacobian_h_u[nDOF_test_element][nDOF_trial_element],
            elementJacobian_h_v[nDOF_test_element][nDOF_trial_element],
            elementJacobian_u_h[nDOF_test_element][nDOF_trial_element],
            elementJacobian_u_u[nDOF_test_element][nDOF_trial_element],
            elementJacobian_u_v[nDOF_test_element][nDOF_trial_element],
            elementJacobian_v_h[nDOF_test_element][nDOF_trial_element],
            elementJacobian_v_u[nDOF_test_element][nDOF_trial_element],
            elementJacobian_v_v[nDOF_test_element][nDOF_trial_element];
          for (int i=0;i<nDOF_test_element;i++)
            for (int j=0;j<nDOF_trial_element;j++)
              {
                elementJacobian_h_h[i][j]=0.0;
                elementJacobian_h_u[i][j]=0.0;
                elementJacobian_h_v[i][j]=0.0;
                elementJacobian_u_h[i][j]=0.0;
                elementJacobian_u_u[i][j]=0.0;
                elementJacobian_u_v[i][j]=0.0;
                elementJacobian_v_h[i][j]=0.0;
                elementJacobian_v_u[i][j]=0.0;
                elementJacobian_v_v[i][j]=0.0;
              }
          for  (int k=0;k<nQuadraturePoints_element;k++)
            {
              proteus_index_t eN_k = eN*nQuadraturePoints_element+k, //index to a scalar at a quadrature point
                eN_k_nSpace = eN_k*nSpace,
                eN_nDOF_trial_element = eN*nDOF_trial_element; //index to a vector at a quadrature point

              //declare local storage
              double b=0.0,
                h=0.0,
                u=0.0,
                v=0.0,
                h_sge=0.0,
                u_sge=0.0,
                v_sge=0.0,
                grad_b[nSpace],
                grad_h[nSpace],
                grad_u[nSpace],
                grad_v[nSpace],
                mass_acc=0.0,
                dmass_acc_h=0.0,
                mom_u_acc=0.0,
                dmom_u_acc_h=0.0,
                dmom_u_acc_u=0.0,
                mom_v_acc=0.0,
                dmom_v_acc_h=0.0,
                dmom_v_acc_v=0.0,
                mass_adv[nSpace],
                dmass_adv_h[nSpace],
                dmass_adv_u[nSpace],
                dmass_adv_v[nSpace],
                dmass_adv_h_sge[nSpace],
                dmass_adv_u_sge[nSpace],
                dmass_adv_v_sge[nSpace],
                mom_u_adv[nSpace],
                dmom_u_adv_h[nSpace],
                dmom_u_adv_u[nSpace],
                dmom_u_adv_v[nSpace],
                dmom_u_adv_h_sge[nSpace],
                dmom_u_adv_u_sge[nSpace],
                dmom_u_adv_v_sge[nSpace],
                mom_v_adv[nSpace],
                dmom_v_adv_h[nSpace],
                dmom_v_adv_u[nSpace],
                dmom_v_adv_v[nSpace],
                dmom_v_adv_h_sge[nSpace],
                dmom_v_adv_u_sge[nSpace],
                dmom_v_adv_v_sge[nSpace],
                mom_u_diff_ten[nSpace],
                mom_v_diff_ten[nSpace],
                mom_uv_diff_ten[1],
                mom_vu_diff_ten[1],
                mom_u_source=0.0,
                dmom_u_source_h=0.0,
                mom_v_source=0.0,
                dmom_v_source_h=0.0,
                mass_acc_t=0.0,
                dmass_acc_h_t=0.0,
                mom_u_acc_t=0.0,
                dmom_u_acc_h_t=0.0,
                dmom_u_acc_u_t=0.0,
                mom_v_acc_t=0.0,
                dmom_v_acc_h_t=0.0,
                dmom_v_acc_v_t=0.0,
                pdeResidual_h=0.0,
                pdeResidual_u=0.0,
                pdeResidual_v=0.0,
                dpdeResidual_h_h[nDOF_trial_element],
                dpdeResidual_h_u[nDOF_trial_element],
                dpdeResidual_h_v[nDOF_trial_element],
                dpdeResidual_u_h[nDOF_trial_element],
                dpdeResidual_u_u[nDOF_trial_element],
                dpdeResidual_u_v[nDOF_trial_element],
                dpdeResidual_v_h[nDOF_trial_element],
                dpdeResidual_v_u[nDOF_trial_element],
                dpdeResidual_v_v[nDOF_trial_element],
                //mwf switched away from usual VMS
                tau_x[9],tau_y[9],
                Lhat_x[nDOF_test_element],
                Lhat_y[nDOF_test_element],
                subgridError_hx=0.0,
                subgridError_ux=0.0,
                subgridError_vx=0.0,
                subgridError_hy=0.0,
                subgridError_uy=0.0,
                subgridError_vy=0.0,
                dsubgridError_hx_h[nDOF_trial_element],
                dsubgridError_hx_u[nDOF_trial_element],
                dsubgridError_hx_v[nDOF_trial_element],
                dsubgridError_ux_h[nDOF_trial_element],
                dsubgridError_ux_u[nDOF_trial_element],
                dsubgridError_ux_v[nDOF_trial_element],
                dsubgridError_vx_h[nDOF_trial_element],
                dsubgridError_vx_u[nDOF_trial_element],
                dsubgridError_vx_v[nDOF_trial_element],
                dsubgridError_hy_h[nDOF_trial_element],
                dsubgridError_hy_u[nDOF_trial_element],
                dsubgridError_hy_v[nDOF_trial_element],
                dsubgridError_uy_h[nDOF_trial_element],
                dsubgridError_uy_u[nDOF_trial_element],
                dsubgridError_uy_v[nDOF_trial_element],
                dsubgridError_vy_h[nDOF_trial_element],
                dsubgridError_vy_u[nDOF_trial_element],
                dsubgridError_vy_v[nDOF_trial_element],
                //mwf end changes
                jac[nSpace*nSpace],
                jacDet,
                jacInv[nSpace*nSpace],
                h_grad_trial[nDOF_trial_element*nSpace],
                vel_grad_trial[nDOF_trial_element*nSpace],
                dV,
                h_test_dV[nDOF_test_element],
                vel_test_dV[nDOF_test_element],
                h_grad_test_dV[nDOF_test_element*nSpace],
                vel_grad_test_dV[nDOF_test_element*nSpace],
                x,y,xt,yt,
                G[nSpace*nSpace],G_dd_G,tr_G, dmom_adv_star[nSpace], dmom_adv_sge[nSpace];
              //mwf added
              //mwf todo add as input or calculate based on 'entropy'
              double alpha_tau=0.25;
              double include_acc_in_strong_mom = 1.0;//omit accumulation terms from momentum strong residual?
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
                                          x,y);
              ck.calculateMappingVelocity_element(eN,
                                                  k,
                                                  mesh_velocity_dof.data(),
                                                  mesh_l2g.data(),
                                                  mesh_trial_ref.data(),
                                                  xt,yt);
              //get the physical integration weight
              dV = fabs(jacDet)*dV_ref.data()[k];

              //get the trial function gradients
              ck.gradTrialFromRef(&h_grad_trial_ref.data()[k*nDOF_trial_element*nSpace],jacInv,h_grad_trial);
              ck.gradTrialFromRef(&vel_grad_trial_ref.data()[k*nDOF_trial_element*nSpace],jacInv,vel_grad_trial);
              //get the solution
              ck.valFromDOF(b_dof.data(),&h_l2g.data()[eN_nDOF_trial_element],&h_trial_ref.data()[k*nDOF_trial_element],b);
              ck.valFromDOF(h_dof.data(),&h_l2g.data()[eN_nDOF_trial_element],&h_trial_ref.data()[k*nDOF_trial_element],h);
              ck.valFromDOF(u_dof.data(),&vel_l2g.data()[eN_nDOF_trial_element],&vel_trial_ref.data()[k*nDOF_trial_element],u);
              ck.valFromDOF(v_dof.data(),&vel_l2g.data()[eN_nDOF_trial_element],&vel_trial_ref.data()[k*nDOF_trial_element],v);
              ck.valFromDOF(h_dof_sge.data(),&h_l2g.data()[eN_nDOF_trial_element],&h_trial_ref.data()[k*nDOF_trial_element],h_sge);
              ck.valFromDOF(u_dof_sge.data(),&vel_l2g.data()[eN_nDOF_trial_element],&vel_trial_ref.data()[k*nDOF_trial_element],u_sge);
              ck.valFromDOF(v_dof_sge.data(),&vel_l2g.data()[eN_nDOF_trial_element],&vel_trial_ref.data()[k*nDOF_trial_element],v_sge);
              //get the solution gradients
              ck.gradFromDOF(b_dof.data(),&h_l2g.data()[eN_nDOF_trial_element],h_grad_trial,grad_b);
              ck.gradFromDOF(h_dof.data(),&h_l2g.data()[eN_nDOF_trial_element],h_grad_trial,grad_h);
              ck.gradFromDOF(u_dof.data(),&vel_l2g.data()[eN_nDOF_trial_element],vel_grad_trial,grad_u);
              ck.gradFromDOF(v_dof.data(),&vel_l2g.data()[eN_nDOF_trial_element],vel_grad_trial,grad_v);
              //precalculate test function products with integration weights
              for (int j=0;j<nDOF_trial_element;j++)
                {
                  h_test_dV[j] = h_test_ref.data()[k*nDOF_trial_element+j]*dV;
                  vel_test_dV[j] = vel_test_ref.data()[k*nDOF_trial_element+j]*dV;
                  for (int I=0;I<nSpace;I++)
                    {
                      h_grad_test_dV[j*nSpace+I]   = h_grad_trial[j*nSpace+I]*dV;//cek warning won't work for Petrov-Galerkin
                      vel_grad_test_dV[j*nSpace+I] = vel_grad_trial[j*nSpace+I]*dV;//cek warning won't work for Petrov-Galerkin}
                    }
                }
              evaluateCoefficients(nu,
                                   g,
                                   grad_b,
                                   h,
                                   u,
                                   v,
                                   mass_acc,
                                   dmass_acc_h,
                                   mom_u_acc,
                                   dmom_u_acc_h,
                                   dmom_u_acc_u,
                                   mom_v_acc,
                                   dmom_v_acc_h,
                                   dmom_v_acc_v,
                                   mass_adv,
                                   dmass_adv_h,
                                   dmass_adv_u,
                                   dmass_adv_v,
                                   mom_u_adv,
                                   dmom_u_adv_h,
                                   dmom_u_adv_u,
                                   dmom_u_adv_v,
                                   mom_v_adv,
                                   dmom_v_adv_h,
                                   dmom_v_adv_u,
                                   dmom_v_adv_v,
                                   mom_u_diff_ten,
                                   mom_v_diff_ten,
                                   mom_uv_diff_ten,
                                   mom_vu_diff_ten,
                                   mom_u_source,
                                   dmom_u_source_h,
                                   mom_v_source,
                                   dmom_v_source_h);
              //
              //calculate time derivatives
              //
              ck.bdf(alphaBDF,
                     q_mass_acc_beta_bdf.data()[eN_k],
                     mass_acc,
                     dmass_acc_h,
                     mass_acc_t,
                     dmass_acc_h_t);
              ck.bdfC2(alphaBDF,
                       q_mom_u_acc_beta_bdf.data()[eN_k],
                       mom_u_acc,
                       dmom_u_acc_h,
                       dmom_u_acc_u,
                       mom_u_acc_t,
                       dmom_u_acc_h_t,
                       dmom_u_acc_u_t);
              ck.bdfC2(alphaBDF,
                       q_mom_v_acc_beta_bdf.data()[eN_k],
                       mom_v_acc,
                       dmom_v_acc_h,
                       dmom_v_acc_v,
                       mom_v_acc_t,
                       dmom_v_acc_h_t,
                       dmom_v_acc_v_t);
              //
              //calculate subgrid error contribution to the Jacobian (strong residual, adjoint, jacobian of strong residual)
              //
              //
              //calculate strong residual
              //
              dmass_adv_h_sge[0]  = dmass_adv_h[0];
              dmass_adv_h_sge[1]  = dmass_adv_h[1];
              dmass_adv_u_sge[0]  = dmass_adv_u[0];
              dmass_adv_u_sge[1]  = dmass_adv_u[1];
              dmass_adv_v_sge[0]  = dmass_adv_v[0];
              dmass_adv_v_sge[1]  = dmass_adv_v[1];
              dmom_u_adv_h_sge[0] = dmom_u_adv_h[0];
              dmom_u_adv_h_sge[1] = dmom_u_adv_h[1];
              dmom_u_adv_u_sge[0] = dmom_u_adv_u[0];
              dmom_u_adv_u_sge[1] = dmom_u_adv_u[1];
              dmom_u_adv_v_sge[0] = dmom_u_adv_v[0];
              dmom_u_adv_v_sge[1] = dmom_u_adv_v[1];
              dmom_v_adv_h_sge[0] = dmom_v_adv_h[0];
              dmom_v_adv_h_sge[1] = dmom_v_adv_h[1];
              dmom_v_adv_u_sge[0] = dmom_v_adv_u[0];
              dmom_v_adv_u_sge[1] = dmom_v_adv_u[1];
              dmom_v_adv_v_sge[0] = dmom_v_adv_v[0];
              dmom_v_adv_v_sge[1] = dmom_v_adv_v[1];

              //full linearization, lagged

              //mass advective flux
              dmass_adv_h_sge[0]=u_sge;
              dmass_adv_h_sge[1]=v_sge;

              dmass_adv_u_sge[0]=h_sge;
              dmass_adv_u_sge[1]=0.0;

              dmass_adv_v_sge[0]=0.0;
              dmass_adv_v_sge[1]=h_sge;

              //u momentum advective flux
              dmom_u_adv_h_sge[0]=u_sge*u_sge + g*h_sge;
              dmom_u_adv_h_sge[1]=u_sge*v_sge;

              dmom_u_adv_u_sge[0]=h_sge*2.0*u_sge;
              dmom_u_adv_u_sge[1]=h_sge*v_sge;

              dmom_u_adv_v_sge[0]=0.0;
              dmom_u_adv_v_sge[1]=h_sge*u_sge;

              //v momentum advective_flux
              dmom_v_adv_h_sge[0]=v_sge*u_sge;
              dmom_v_adv_h_sge[1]=v_sge*v_sge + g*h_sge;

              dmom_v_adv_u_sge[0]=h_sge*v_sge;
              dmom_v_adv_u_sge[1]=0.0;

              dmom_v_adv_v_sge[0]=h_sge*u_sge;
              dmom_v_adv_v_sge[1]=h_sge*2.0*v_sge;

              //calculate the Jacobian of strong residual
              for (int j=0;j<nDOF_trial_element;j++)
                {
                  int j_nSpace = j*nSpace;

                  dpdeResidual_h_h[j]= ck.MassJacobian_strong(dmass_acc_h_t,h_trial_ref.data()[k*nDOF_trial_element+j])+
                    ck.AdvectionJacobian_strong(dmass_adv_h_sge,&h_grad_trial[j_nSpace]);

                  dpdeResidual_h_u[j]=ck.AdvectionJacobian_strong(dmass_adv_u_sge,&vel_grad_trial[j_nSpace]);

                  dpdeResidual_h_v[j]=ck.AdvectionJacobian_strong(dmass_adv_v_sge,&vel_grad_trial[j_nSpace]);

                  dpdeResidual_u_h[j]= include_acc_in_strong_mom*ck.MassJacobian_strong(dmom_u_acc_h_t,h_trial_ref.data()[k*nDOF_trial_element+j])+
                    ck.AdvectionJacobian_strong(dmom_u_adv_h_sge,&h_grad_trial[j_nSpace]) +
                    ck.ReactionJacobian_strong(dmom_u_source_h,h_trial_ref.data()[k*nDOF_trial_element+j]);

                  dpdeResidual_u_u[j]= include_acc_in_strong_mom*ck.MassJacobian_strong(dmom_u_acc_u_t,vel_trial_ref.data()[k*nDOF_trial_element+j])+
                    ck.AdvectionJacobian_strong(dmom_u_adv_u_sge,&vel_grad_trial[j_nSpace]);

                  dpdeResidual_u_v[j]=ck.AdvectionJacobian_strong(dmom_u_adv_v_sge,&vel_grad_trial[j_nSpace]);

                  dpdeResidual_v_h[j]= include_acc_in_strong_mom*ck.MassJacobian_strong(dmom_v_acc_h_t,h_trial_ref.data()[k*nDOF_trial_element+j])+
                    ck.AdvectionJacobian_strong(dmom_v_adv_h_sge,&h_grad_trial[j_nSpace])+
                    ck.ReactionJacobian_strong(dmom_v_source_h,h_trial_ref.data()[k*nDOF_trial_element+j]);

                  dpdeResidual_v_u[j]=ck.AdvectionJacobian_strong(dmom_v_adv_u_sge,&vel_grad_trial[j_nSpace]);

                  dpdeResidual_v_v[j]= include_acc_in_strong_mom*ck.MassJacobian_strong(dmom_v_acc_v_t,vel_trial_ref.data()[k*nDOF_trial_element+j])+
                    ck.AdvectionJacobian_strong(dmom_v_adv_v_sge,&vel_grad_trial[j_nSpace]);

                }
              /* //calculate tau and tau*Res */
              calculateSubgridError_tau_supg(elementDiameter.data()[eN],
                                             nu,
                                             g,
                                             h_sge,
                                             u_sge,
                                             v_sge,
                                             alpha_tau,
                                             dV,
                                             tau_x,
                                             tau_y,
                                             q_cfl.data()[eN_k]);

              for (int j=0;j<nDOF_trial_element;j++)
                {
                  dsubgridError_hx_h[j] = - tau_x[0*3+0]*dpdeResidual_h_h[j] - tau_x[0*3+1]*dpdeResidual_u_h[j] - tau_x[0*3+2]*dpdeResidual_v_h[j];
                  dsubgridError_hx_u[j] = - tau_x[0*3+0]*dpdeResidual_h_u[j] - tau_x[0*3+1]*dpdeResidual_u_u[j] - tau_x[0*3+2]*dpdeResidual_v_u[j];
                  dsubgridError_hx_v[j] = - tau_x[0*3+0]*dpdeResidual_h_v[j] - tau_x[0*3+1]*dpdeResidual_u_v[j] - tau_x[0*3+2]*dpdeResidual_v_v[j];

                  dsubgridError_ux_h[j] = - tau_x[1*3+0]*dpdeResidual_h_h[j] - tau_x[1*3+1]*dpdeResidual_u_h[j] - tau_x[1*3+2]*dpdeResidual_v_h[j];
                  dsubgridError_ux_u[j] = - tau_x[1*3+0]*dpdeResidual_h_u[j] - tau_x[1*3+1]*dpdeResidual_u_u[j] - tau_x[1*3+2]*dpdeResidual_v_u[j];
                  dsubgridError_ux_v[j] = - tau_x[1*3+0]*dpdeResidual_h_v[j] - tau_x[1*3+1]*dpdeResidual_u_v[j] - tau_x[1*3+2]*dpdeResidual_v_v[j];

                  dsubgridError_vx_h[j] = - tau_x[2*3+0]*dpdeResidual_h_h[j] - tau_x[2*3+1]*dpdeResidual_u_h[j] - tau_x[2*3+2]*dpdeResidual_v_h[j];
                  dsubgridError_vx_u[j] = - tau_x[2*3+0]*dpdeResidual_h_u[j] - tau_x[2*3+1]*dpdeResidual_u_u[j] - tau_x[2*3+2]*dpdeResidual_v_u[j];
                  dsubgridError_vx_v[j] = - tau_x[2*3+0]*dpdeResidual_h_v[j] - tau_x[2*3+1]*dpdeResidual_u_v[j] - tau_x[2*3+2]*dpdeResidual_v_v[j];
                  //
                  dsubgridError_hy_h[j] = - tau_y[0*3+0]*dpdeResidual_h_h[j] - tau_y[0*3+1]*dpdeResidual_u_h[j] - tau_y[0*3+2]*dpdeResidual_v_h[j];
                  dsubgridError_hy_u[j] = - tau_y[0*3+0]*dpdeResidual_h_u[j] - tau_y[0*3+1]*dpdeResidual_u_u[j] - tau_y[0*3+2]*dpdeResidual_v_u[j];
                  dsubgridError_hy_v[j] = - tau_y[0*3+0]*dpdeResidual_h_v[j] - tau_y[0*3+1]*dpdeResidual_u_v[j] - tau_y[0*3+2]*dpdeResidual_v_v[j];

                  dsubgridError_uy_h[j] = - tau_y[1*3+0]*dpdeResidual_h_h[j] - tau_y[1*3+1]*dpdeResidual_u_h[j] - tau_y[1*3+2]*dpdeResidual_v_h[j];
                  dsubgridError_uy_u[j] = - tau_y[1*3+0]*dpdeResidual_h_u[j] - tau_y[1*3+1]*dpdeResidual_u_u[j] - tau_y[1*3+2]*dpdeResidual_v_u[j];
                  dsubgridError_uy_v[j] = - tau_y[1*3+0]*dpdeResidual_h_v[j] - tau_y[1*3+1]*dpdeResidual_u_v[j] - tau_y[1*3+2]*dpdeResidual_v_v[j];

                  dsubgridError_vy_h[j] = - tau_y[2*3+0]*dpdeResidual_h_h[j] - tau_y[2*3+1]*dpdeResidual_u_h[j] - tau_y[2*3+2]*dpdeResidual_v_h[j];
                  dsubgridError_vy_u[j] = - tau_y[2*3+0]*dpdeResidual_h_u[j] - tau_y[2*3+1]*dpdeResidual_u_u[j] - tau_y[2*3+2]*dpdeResidual_v_u[j];
                  dsubgridError_vy_v[j] = - tau_y[2*3+0]*dpdeResidual_h_v[j] - tau_y[2*3+1]*dpdeResidual_u_v[j] - tau_y[2*3+2]*dpdeResidual_v_v[j];

                }
              //Not really SUPG, but is test function contribution
              for (int i=0;i<nDOF_test_element;i++)
                {
                  int i_nSpace = i*nSpace;
                  Lhat_x[i]= -h_grad_test_dV[i_nSpace];
                  Lhat_y[i]= -h_grad_test_dV[i_nSpace+1];
                }

              for(int i=0;i<nDOF_test_element;i++)
                {
                  int i_nSpace = i*nSpace;
                  for(int j=0;j<nDOF_trial_element;j++)
                    {
                      int j_nSpace = j*nSpace;
                      //h
                      elementJacobian_h_h[i][j] += ck.MassJacobian_weak(dmass_acc_h_t,h_trial_ref.data()[k*nDOF_trial_element+j],h_test_dV[i]) +
                        ck.AdvectionJacobian_weak(dmass_adv_h,h_trial_ref.data()[k*nDOF_trial_element+j],&h_grad_test_dV[i_nSpace]) +
                        //mwf changed stabilization terms
                        ck.SubgridErrorJacobian(dsubgridError_hx_h[j],Lhat_x[i]) +
                        ck.SubgridErrorJacobian(dsubgridError_hy_h[j],Lhat_y[i]) +
                        ck.NumericalDiffusionJacobian(q_numDiff_h_last.data()[eN_k],&h_grad_trial[j_nSpace],&h_grad_test_dV[i_nSpace]);

                      elementJacobian_h_u[i][j] += ck.AdvectionJacobian_weak(dmass_adv_u,vel_trial_ref.data()[k*nDOF_trial_element+j],&h_grad_test_dV[i_nSpace]) +
                        //mwf changed stabilization terms
                        ck.SubgridErrorJacobian(dsubgridError_hx_u[j],Lhat_x[i]) +
                        ck.SubgridErrorJacobian(dsubgridError_hy_u[j],Lhat_y[i]);

                      elementJacobian_h_v[i][j] += ck.AdvectionJacobian_weak(dmass_adv_v,vel_trial_ref.data()[k*nDOF_trial_element+j],&h_grad_test_dV[i_nSpace]) +
                        //mwf changed stabilization terms
                        ck.SubgridErrorJacobian(dsubgridError_hx_v[j],Lhat_x[i]) +
                        ck.SubgridErrorJacobian(dsubgridError_hy_v[j],Lhat_y[i]);

                      //u
                      elementJacobian_u_h[i][j] += ck.MassJacobian_weak(dmom_u_acc_h_t,h_trial_ref.data()[k*nDOF_trial_element+j],vel_test_dV[i]) +
                        ck.AdvectionJacobian_weak(dmom_u_adv_h,h_trial_ref.data()[k*nDOF_trial_element+j],&vel_grad_test_dV[i_nSpace]) +
                        ck.ReactionJacobian_weak(dmom_u_source_h,h_trial_ref.data()[k*nDOF_trial_element+j],vel_test_dV[i]) +
                        //mwf changed stabilization terms
                        ck.SubgridErrorJacobian(dsubgridError_ux_h[j],Lhat_x[i]) +
                        ck.SubgridErrorJacobian(dsubgridError_uy_h[j],Lhat_y[i]);


                      elementJacobian_u_u[i][j] += ck.MassJacobian_weak(dmom_u_acc_u_t,vel_trial_ref.data()[k*nDOF_trial_element+j],vel_test_dV[i]) +
                        ck.AdvectionJacobian_weak(dmom_u_adv_u,vel_trial_ref.data()[k*nDOF_trial_element+j],&vel_grad_test_dV[i_nSpace]) +
                        ck.SimpleDiffusionJacobian_weak(sdInfo_u_u_rowptr.data(),sdInfo_u_u_colind.data(),mom_u_diff_ten,&vel_grad_trial[j_nSpace],&vel_grad_test_dV[i_nSpace]) +
                        ck.SubgridErrorJacobian(dsubgridError_ux_u[j],Lhat_x[i]) +
                        ck.SubgridErrorJacobian(dsubgridError_uy_u[j],Lhat_y[i]) +
                        ck.NumericalDiffusionJacobian(q_numDiff_u_last.data()[eN_k],&vel_grad_trial[j_nSpace],&vel_grad_test_dV[i_nSpace]);

                      elementJacobian_u_v[i][j] += ck.AdvectionJacobian_weak(dmom_u_adv_v,vel_trial_ref.data()[k*nDOF_trial_element+j],&vel_grad_test_dV[i_nSpace]) +
                        ck.SimpleDiffusionJacobian_weak(sdInfo_u_v_rowptr.data(),sdInfo_u_v_colind.data(),mom_uv_diff_ten,&vel_grad_trial[j_nSpace],&vel_grad_test_dV[i_nSpace]) +
                        ck.SubgridErrorJacobian(dsubgridError_ux_v[j],Lhat_x[i]) +
                        ck.SubgridErrorJacobian(dsubgridError_uy_v[j],Lhat_y[i]);

                      //v
                      elementJacobian_v_h[i][j] += ck.MassJacobian_weak(dmom_v_acc_h_t,vel_trial_ref.data()[k*nDOF_trial_element+j],vel_test_dV[i]) +
                        ck.AdvectionJacobian_weak(dmom_v_adv_h,h_trial_ref.data()[k*nDOF_trial_element+j],&vel_grad_test_dV[i_nSpace]) +
                        ck.ReactionJacobian_weak(dmom_v_source_h,h_trial_ref.data()[k*nDOF_trial_element+j],vel_test_dV[i]) +
                        ck.SubgridErrorJacobian(dsubgridError_vx_h[j],Lhat_x[i]) +
                        ck.SubgridErrorJacobian(dsubgridError_vy_h[j],Lhat_y[i]);


                      elementJacobian_v_u[i][j] += ck.AdvectionJacobian_weak(dmom_v_adv_u,vel_trial_ref.data()[k*nDOF_trial_element+j],&vel_grad_test_dV[i_nSpace]) +
                        ck.SimpleDiffusionJacobian_weak(sdInfo_v_u_rowptr.data(),sdInfo_v_u_colind.data(),mom_vu_diff_ten,&vel_grad_trial[j_nSpace],&vel_grad_test_dV[i_nSpace]) +
                        ck.SubgridErrorJacobian(dsubgridError_vx_u[j],Lhat_x[i]) +
                        ck.SubgridErrorJacobian(dsubgridError_vy_u[j],Lhat_y[i]);


                      elementJacobian_v_v[i][j] += ck.MassJacobian_weak(dmom_v_acc_v_t,vel_trial_ref.data()[k*nDOF_trial_element+j],vel_test_dV[i]) +
                        ck.AdvectionJacobian_weak(dmom_v_adv_v,vel_trial_ref.data()[k*nDOF_trial_element+j],&vel_grad_test_dV[i_nSpace]) +
                        ck.SimpleDiffusionJacobian_weak(sdInfo_v_v_rowptr.data(),sdInfo_v_v_colind.data(),mom_v_diff_ten,&vel_grad_trial[j_nSpace],&vel_grad_test_dV[i_nSpace]) +
                        ck.SubgridErrorJacobian(dsubgridError_vx_v[j],Lhat_x[i]) +
                        ck.SubgridErrorJacobian(dsubgridError_vy_v[j],Lhat_y[i]) +
                        ck.NumericalDiffusionJacobian(q_numDiff_v_last.data()[eN_k],&vel_grad_trial[j_nSpace],&vel_grad_test_dV[i_nSpace]);
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
                  globalJacobian.data()[csrRowIndeces_h_h.data()[eN_i] + csrColumnOffsets_h_h.data()[eN_i_j]] += elementJacobian_h_h[i][j];
                  globalJacobian.data()[csrRowIndeces_h_u.data()[eN_i] + csrColumnOffsets_h_u.data()[eN_i_j]] += elementJacobian_h_u[i][j];
                  globalJacobian.data()[csrRowIndeces_h_v.data()[eN_i] + csrColumnOffsets_h_v.data()[eN_i_j]] += elementJacobian_h_v[i][j];

                  globalJacobian.data()[csrRowIndeces_u_h.data()[eN_i] + csrColumnOffsets_u_h.data()[eN_i_j]] += elementJacobian_u_h[i][j];
                  globalJacobian.data()[csrRowIndeces_u_u.data()[eN_i] + csrColumnOffsets_u_u.data()[eN_i_j]] += elementJacobian_u_u[i][j];
                  globalJacobian.data()[csrRowIndeces_u_v.data()[eN_i] + csrColumnOffsets_u_v.data()[eN_i_j]] += elementJacobian_u_v[i][j];

                  globalJacobian.data()[csrRowIndeces_v_h.data()[eN_i] + csrColumnOffsets_v_h.data()[eN_i_j]] += elementJacobian_v_h[i][j];
                  globalJacobian.data()[csrRowIndeces_v_u.data()[eN_i] + csrColumnOffsets_v_u.data()[eN_i_j]] += elementJacobian_v_u[i][j];
                  globalJacobian.data()[csrRowIndeces_v_v.data()[eN_i] + csrColumnOffsets_v_v.data()[eN_i_j]] += elementJacobian_v_v[i][j];
                }//j
            }//i
        }//elements
      //
      //loop over exterior element boundaries to compute the surface integrals and load them into the global Jacobian
      //
    }//computeJacobian_supg

    /* void calculateVelocityAverage(proteus_index_t nExteriorElementBoundaries_global, */
    /*                            proteus_index_t* exteriorElementBoundariesArray, */
    /*                            proteus_index_t nInteriorElementBoundaries_global, */
    /*                            proteus_index_t* interiorElementBoundariesArray, */
    /*                            proteus_index_t* elementBoundaryElementsArray, */
    /*                            int* elementBoundaryLocalElementBoundariesArray, */
    /*                            double* mesh_dof, */
    /*                            proteus_index_t* mesh_l2g, */
    /*                            double* mesh_trial_trace_ref, */
    /*                            double* mesh_grad_trial_trace_ref, */
    /*                            double* normal_ref, */
    /*                            double* boundaryJac_ref, */
    /*                            proteus_index_t* vel_l2g, */
    /*                            double* u_dof, */
    /*                            double* v_dof, */
    /*                            double* vel_trial_trace_ref, */
    /*                            double* ebqe_velocity.data(), */
    /*                            double* velocityAverage) */
    /* { */
    /*   int permutations[nQuadraturePoints_elementBoundary]; */
    /*   double xArray_left[nQuadraturePoints_elementBoundary*3], */
    /*  xArray_right[nQuadraturePoints_elementBoundary*3]; */
    /*   for (proteus_index_t ebNE = 0; ebNE < nExteriorElementBoundaries_global; ebNE++) */
    /*  { */
    /*    proteus_index_t ebN = exteriorElementBoundariesArray[ebNE]; */
    /*    for  (int kb=0;kb<nQuadraturePoints_elementBoundary;kb++) */
    /*      { */
    /*        proteus_index_t ebN_kb_nSpace = ebN*nQuadraturePoints_elementBoundary*nSpace+kb*nSpace, */
    /*          ebNE_kb_nSpace = ebNE*nQuadraturePoints_elementBoundary*nSpace+kb*nSpace; */
    /*        velocityAverage[ebN_kb_nSpace+0]=ebqe_velocity.data()[ebNE_kb_nSpace+0]; */
    /*        velocityAverage[ebN_kb_nSpace+1]=ebqe_velocity.data()[ebNE_kb_nSpace+1]; */
    /*        velocityAverage[ebN_kb_nSpace+2]=ebqe_velocity.data()[ebNE_kb_nSpace+2]; */
    /*      }//ebNE */
    /*  } */
    /*   for (proteus_index_t ebNI = 0; ebNI < nInteriorElementBoundaries_global; ebNI++) */
    /*  { */
    /*    proteus_index_t ebN = interiorElementBoundariesArray[ebNI], */
    /*      left_eN_global   = elementBoundaryElementsArray[ebN*2+0], */
    /*      left_ebN_element  = elementBoundaryLocalElementBoundariesArray[ebN*2+0], */
    /*      right_eN_global  = elementBoundaryElementsArray[ebN*2+1], */
    /*      right_ebN_element = elementBoundaryLocalElementBoundariesArray[ebN*2+1], */
    /*      left_eN_nDOF_trial_element = left_eN_global*nDOF_trial_element, */
    /*      right_eN_nDOF_trial_element = right_eN_global*nDOF_trial_element; */
    /*    double jac[nSpace*nSpace], */
    /*      jacDet, */
    /*      jacInv[nSpace*nSpace], */
    /*      boundaryJac[nSpace*(nSpace-1)], */
    /*      metricTensor[(nSpace-1)*(nSpace-1)], */
    /*      metricTensorDetSqrt, */
    /*      normal[3], */
    /*      x,y; */
    /*    //double G[nSpace*nSpace],G_dd_G,tr_G,h_hhi,h_henalty; */

    /*    for  (int kb=0;kb<nQuadraturePoints_elementBoundary;kb++) */
    /*      { */
    /*        ck.calculateMapping_elementBoundary(left_eN_global, */
    /*                                            left_ebN_element, */
    /*                                            kb, */
    /*                                            left_ebN_element*kb, */
    /*                                            mesh_dof, */
    /*                                            mesh_l2g, */
    /*                                            mesh_trial_trace_ref, */
    /*                                            mesh_grad_trial_trace_ref, */
    /*                                            boundaryJac_ref, */
    /*                                            jac, */
    /*                                            jacDet, */
    /*                                            jacInv, */
    /*                                            boundaryJac, */
    /*                                            metricTensor, */
    /*                                            metricTensorDetSqrt, */
    /*                                            normal_ref, */
    /*                                            normal, */
    /*                                            x,y); */
    /*        xArray_left[kb*3+0] = x; */
    /*        xArray_left[kb*3+1] = y; */
    /*        ck.calculateMapping_elementBoundary(right_eN_global, */
    /*                                            right_ebN_element, */
    /*                                            kb, */
    /*                                            right_ebN_element*kb, */
    /*                                            mesh_dof, */
    /*                                            mesh_l2g, */
    /*                                            mesh_trial_trace_ref, */
    /*                                            mesh_grad_trial_trace_ref, */
    /*                                            boundaryJac_ref, */
    /*                                            jac, */
    /*                                            jacDet, */
    /*                                            jacInv, */
    /*                                            boundaryJac, */
    /*                                            metricTensor, */
    /*                                            metricTensorDetSqrt, */
    /*                                            normal_ref, */
    /*                                            normal, */
    /*                                            x,y); */
    /*        xArray_right[kb*3+0] = x; */
    /*        xArray_right[kb*3+1] = y; */
    /*      } */
    /*    for  (int kb_left=0;kb_left<nQuadraturePoints_elementBoundary;kb_left++) */
    /*      { */
    /*        double errorNormMin = 1.0; */
    /*        for  (int kb_right=0;kb_right<nQuadraturePoints_elementBoundary;kb_right++) */
    /*          { */
    /*            double errorNorm=0.0; */
    /*            for (int I=0;I<nSpace;I++) */
    /*              { */
    /*                errorNorm += fabs(xArray_left[kb_left*3+I] */
    /*                                  - */
    /*                                  xArray_right[kb_right*3+I]); */
    /*              } */
    /*            if (errorNorm < errorNormMin) */
    /*              { */
    /*                permutations[kb_right] = kb_left; */
    /*                errorNormMin = errorNorm; */
    /*              } */
    /*          } */
    /*      } */
    /*    for  (int kb=0;kb<nQuadraturePoints_elementBoundary;kb++) */
    /*      { */
    /*        proteus_index_t ebN_kb_nSpace = ebN*nQuadraturePoints_elementBoundary*nSpace+kb*nSpace; */
    /*        double u_left=0.0, */
    /*          v_left=0.0, */
    /*          u_right=0.0, */
    /*          v_right=0.0; */
    /*        int left_kb = kb, */
    /*          right_kb = permutations[kb], */
    /*          left_ebN_element_kb_nDOF_test_element=left_ebN_element*left_kb*nDOF_test_element, */
    /*          right_ebN_element_kb_nDOF_test_element=right_ebN_element*right_kb*nDOF_test_element; */
    /*        // */
    /*        //calculate the velocity solution at quadrature points on left and right */
    /*        // */
    /*        ck.valFromDOF(u_dof,&vel_l2g[left_eN_nDOF_trial_element],&vel_trial_trace_ref[left_ebN_element_kb_nDOF_test_element],u_left); */
    /*        ck.valFromDOF(v_dof,&vel_l2g[left_eN_nDOF_trial_element],&vel_trial_trace_ref[left_ebN_element_kb_nDOF_test_element],v_left); */
    /*        // */
    /*        ck.valFromDOF(u_dof,&vel_l2g[right_eN_nDOF_trial_element],&vel_trial_trace_ref[right_ebN_element_kb_nDOF_test_element],u_right); */
    /*        ck.valFromDOF(v_dof,&vel_l2g[right_eN_nDOF_trial_element],&vel_trial_trace_ref[right_ebN_element_kb_nDOF_test_element],v_right); */
    /*        // */
    /*        velocityAverage[ebN_kb_nSpace+0]=0.5*(u_left + u_right); */
    /*        velocityAverage[ebN_kb_nSpace+1]=0.5*(v_left + v_right); */
    /*      }//ebNI */
    /*  } */
    /* } */

  };//SW2D

  inline SW2D_base* newSW2D(int nSpaceIn,
                            int nQuadraturePoints_elementIn,
                            int nDOF_mesh_trial_elementIn,
                            int nDOF_trial_elementIn,
                            int nDOF_test_elementIn,
                            int nQuadraturePoints_elementBoundaryIn,
                            int CompKernelFlag)
  {
    return proteus::chooseAndAllocateDiscretization2D<SW2D_base,SW2D,CompKernel>(nSpaceIn,
                                                                                 nQuadraturePoints_elementIn,
                                                                                 nDOF_mesh_trial_elementIn,
                                                                                 nDOF_trial_elementIn,
                                                                                 nDOF_test_elementIn,
                                                                                 nQuadraturePoints_elementBoundaryIn,
                                                                                 CompKernelFlag);
  }
}//proteus
#endif
