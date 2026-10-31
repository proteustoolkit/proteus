#ifndef NCLS_H
#define NCLS_H
#include <cmath>
#include <iostream>
#include <valarray>
#include "CompKernel.h"
#include "ModelFactory.h"
#include "ArgumentsDict.h"
#include "pyarray.h"

namespace py = pybind11;

#define POWER_SMOOTHNESS_INDICATOR 2
#define IS_BETAij_ONE 0

namespace proteus
{
  inline double Sign(const double& z){
    return (z >= 0.0 ? 1.0 : -1.0);
  }
  // Power entropy //
  inline double ENTROPY(const double& phi, const double& dummyL, const double& dummyR){
    return 1./2.*std::pow(fabs(phi),2.);
  }
  inline double DENTROPY(const double& phi, const double& dummyL, const double& dummyR){
    return fabs(phi)*(phi >= 0. ? 1. : -1.);
  }
  // Log entropy //
  inline double ENTROPY_LOG(const double& phi, const double& phiL, const double& phiR){
    return std::log(fabs((phi-phiL)*(phiR-phi))+1E-14);
  }
  inline double DENTROPY_LOG(const double& phi, const double& phiL, const double& phiR){
    return (phiL+phiR-2*phi)*((phi-phiL)*(phiR-phi)>=0 ? 1 : -1)/(fabs((phi-phiL)*(phiR-phi))+1E-14);
  }
}

namespace proteus
{
  class NCLS_base
  {
    //The base class defining the interface
  public:
    std::valarray<double> L2_norm_per_node;
    std::valarray<double> TransportMatrix, TransposeTransportMatrix;
    std::valarray<double> psi, etaMax, etaMin;
    std::valarray<double> global_entropy_residual;
    std::valarray<double> gx, gy, gz, eta,
      alpha_numerator_pos, alpha_numerator_neg, alpha_denominator_pos, alpha_denominator_neg;
    virtual ~NCLS_base(){}
    virtual void calculateResidual(arguments_dict& args)=0;
    virtual void calculateJacobian(arguments_dict& args)=0;
    virtual void calculateWaterline(arguments_dict& args)=0;
    virtual double calculateRedistancingResidual(arguments_dict& args)=0;
    virtual double calculateRhsSmoothing(arguments_dict& args)=0;
    virtual void calculateResidual_entropy_viscosity(arguments_dict& args) = 0;
    virtual void calculateMassMatrix(arguments_dict& args)=0;
    virtual void calculateSmoothingMatrix(arguments_dict& args) = 0;
  };

  template<class CompKernelType,
    int nSpace,
    int nQuadraturePoints_element,
    int nDOF_mesh_trial_element,
    int nDOF_trial_element,
    int nDOF_test_element,
    int nQuadraturePoints_elementBoundary>
    class NCLS : public NCLS_base
    {
    public:
      const int nDOF_test_X_trial_element;
      CompKernelType ck;
    NCLS():
      nDOF_test_X_trial_element(nDOF_test_element*nDOF_trial_element),
        ck()
          {}

      inline void evaluateCoefficients(const double v[nSpace],
                                       const double& u,
                                       const double grad_u[nSpace],
                                       double& m,
                                       double& dm,
                                       double& H,
                                       double dH[nSpace])
      {
        m = u;
        dm=1.0;
        H = 0.0;
        for (int I=0; I < nSpace; I++)
          {
            H += v[I]*grad_u[I];
            dH[I] = v[I];
          }
      }
    
      inline
        void calculateCFL(const double& elementDiameter,
                          const double df[nSpace],
                          double& cfl)
      {
        double h,nrm_v;
        h = elementDiameter;
        nrm_v=0.0;
        for(int I=0;I<nSpace;I++)
          nrm_v+=df[I]*df[I];
        nrm_v = sqrt(nrm_v);
        cfl = nrm_v/h;
      }

      inline
        void calculateSubgridError_tau(const double& elementDiameter,
                                       const double& dmt,
                                       const double dH[nSpace],
                                       double& cfl,
                                       double& tau)
      {
        double h,nrm_v,oneByAbsdt;
        h = elementDiameter;
        nrm_v=0.0;
        for(int I=0;I<nSpace;I++)
          nrm_v+=dH[I]*dH[I];
        nrm_v = sqrt(nrm_v);
        cfl = nrm_v/h;
        oneByAbsdt =  fabs(dmt);
        tau = 1.0/(2.0*nrm_v/h + oneByAbsdt + 1.0e-8);
      }


      inline
        void calculateSubgridError_tau(     const double&  Ct_sge,
                                            const double   G[nSpace*nSpace],
                                            const double&  A0,
                                            const double   Ai[nSpace],
                                            double& tau_v,
                                            double& cfl)
      {
        double v_d_Gv=0.0;
        for(int I=0;I<nSpace;I++)
          for (int J=0;J<nSpace;J++)
            v_d_Gv += Ai[I]*G[I*nSpace+J]*Ai[J];

        tau_v = 1.0/sqrt(Ct_sge*A0*A0 + v_d_Gv + 1.0e-8);
      }

      void exteriorNumericalFlux(const double n[nSpace],
                                 const double& bc_u,
                                 const double& u,
                                 const double velocity[nSpace],
                                 const double velocity_movingDomain[nSpace],
                                 double& flux)
      {
        double flow_total=0.0,flow_fluid=0.0,flow_movingDomain=0.0;
        for (int I=0; I < nSpace; I++)
          {
            flow_fluid += n[I]*velocity[I];
            //flow_movingDomain -= n[I]*velocity_movingDomain[I];
          }
        flow_total = flow_fluid+flow_movingDomain;
        if (flow_total > 0.0)
          {
            flux = u*flow_movingDomain;
          }
        else
          {
            flux = bc_u*flow_movingDomain - flow_fluid*(u-bc_u);
          }
      }

      inline
        void exteriorNumericalFluxDerivative(const double n[nSpace],
                                             const double velocity[nSpace],
                                             const double velocity_movingDomain[nSpace],
                                             double& dflux)
      {
        double flow_total=0.0,flow_fluid=0.0,flow_movingDomain=0.0;
        for (int I=0; I < nSpace; I++)
          {
            flow_fluid += n[I]*velocity[I];
            //flow_movingDomain -= n[I]*velocity_movingDomain[I];
          }
        flow_total=flow_fluid+flow_movingDomain;
        if (flow_total > 0.0)
          {
            dflux = flow_movingDomain;
          }
        else
          {
            dflux = -flow_fluid;
          }
      }

      inline
        double sign(const double phi,
                    const double eps) //eps=epsFactRedistancing=0.33*he (for instance)
      {
        double H;
        if (phi > eps)
          H = 1.0;
        else if (phi < -eps)
          H = 0.0;
        else if (phi == 0.0)
          H = 0.5;
        else
          H = 0.5*(1.0 + phi/eps + std::sin(M_PI*phi/eps)/M_PI);
        return -1.0 + 2.0*H;
        //return (u > 0 ? 1. : -1.)*(u ==0 ? 0. : 1.);
        //double tol_sign 0.1;
        //return (u > tol_sign*epsFactRedistancing ? 1. : -1.)*((u > -tol_sign*epsFactRedistancing && u < tol_sign*epsFactRedistancing) ? 0. : 1.);
      }

      void calculateResidual(arguments_dict& args)
      {
        double dt = args.scalar<double>("dt");
        double sc_drop_time = args.scalar<double>("sc_drop_time");
        proteus::pyarray<double>& mesh_trial_ref = args.array<double>("mesh_trial_ref");
        proteus::pyarray<double>& mesh_grad_trial_ref = args.array<double>("mesh_grad_trial_ref");
        proteus::pyarray<double>& mesh_dof = args.array<double>("mesh_dof");
        proteus::pyarray<double>& mesh_velocity_dof = args.array<double>("mesh_velocity_dof");
        double MOVING_DOMAIN = args.scalar<double>("MOVING_DOMAIN");
        proteus::pyarray<int>& mesh_l2g = args.array<int>("mesh_l2g");
        proteus::pyarray<double>& dV_ref = args.array<double>("dV_ref");
        proteus::pyarray<double>& u_trial_ref = args.array<double>("u_trial_ref");
        proteus::pyarray<double>& u_grad_trial_ref = args.array<double>("u_grad_trial_ref");
        proteus::pyarray<double>& u_test_ref = args.array<double>("u_test_ref");
        proteus::pyarray<double>& u_grad_test_ref = args.array<double>("u_grad_test_ref");
        proteus::pyarray<double>& mesh_trial_trace_ref = args.array<double>("mesh_trial_trace_ref");
        proteus::pyarray<double>& mesh_grad_trial_trace_ref = args.array<double>("mesh_grad_trial_trace_ref");
        proteus::pyarray<double>& dS_ref = args.array<double>("dS_ref");
        proteus::pyarray<double>& u_trial_trace_ref = args.array<double>("u_trial_trace_ref");
        proteus::pyarray<double>& u_grad_trial_trace_ref = args.array<double>("u_grad_trial_trace_ref");
        proteus::pyarray<double>& u_test_trace_ref = args.array<double>("u_test_trace_ref");
        proteus::pyarray<double>& u_grad_test_trace_ref = args.array<double>("u_grad_test_trace_ref");
        proteus::pyarray<double>& normal_ref = args.array<double>("normal_ref");
        proteus::pyarray<double>& boundaryJac_ref = args.array<double>("boundaryJac_ref");
        int nElements_global = args.scalar<int>("nElements_global");
        double useMetrics = args.scalar<double>("useMetrics");
        double alphaBDF = args.scalar<double>("alphaBDF");
        int lag_shockCapturing = args.scalar<int>("lag_shockCapturing");
        double shockCapturingDiffusion = args.scalar<double>("shockCapturingDiffusion");
        double sc_uref = args.scalar<double>("sc_uref");
        double sc_alpha = args.scalar<double>("sc_alpha");
        proteus::pyarray<int>& u_l2g = args.array<int>("u_l2g");
        proteus::pyarray<int>& r_l2g = args.array<int>("r_l2g");
        proteus::pyarray<double>& elementDiameter = args.array<double>("elementDiameter");
        proteus::pyarray<double>& nodeDiametersArray = args.array<double>("nodeDiametersArray");
        int degree_polynomial = args.scalar<int>("degree_polynomial");
        proteus::pyarray<double>& u_dof = args.array<double>("u_dof");
        proteus::pyarray<double>& u_dof_old = args.array<double>("u_dof_old");
        proteus::pyarray<double>& uStar_dof = args.array<double>("uStar_dof");
        proteus::pyarray<double>& velocity = args.array<double>("velocity");
        proteus::pyarray<double>& q_m = args.array<double>("q_m");
        proteus::pyarray<double>& q_u = args.array<double>("q_u");
        proteus::pyarray<double>& q_n = args.array<double>("q_n");
        proteus::pyarray<double>& q_dH = args.array<double>("q_dH");
        proteus::pyarray<double>& q_m_betaBDF = args.array<double>("q_m_betaBDF");
        proteus::pyarray<double>& q_dV = args.array<double>("q_dV");
        proteus::pyarray<double>& q_dV_last = args.array<double>("q_dV_last");
        proteus::pyarray<double>& cfl = args.array<double>("cfl");
        proteus::pyarray<double>& edge_based_cfl = args.array<double>("edge_based_cfl");
        proteus::pyarray<double>& q_numDiff_u = args.array<double>("q_numDiff_u");
        proteus::pyarray<double>& q_numDiff_u_last = args.array<double>("q_numDiff_u_last");
        int offset_u = args.scalar<int>("offset_u");
        int stride_u = args.scalar<int>("stride_u");
        proteus::pyarray<double>& globalResidual = args.array<double>("globalResidual");
        int nExteriorElementBoundaries_global = args.scalar<int>("nExteriorElementBoundaries_global");
        proteus::pyarray<int>& exteriorElementBoundariesArray = args.array<int>("exteriorElementBoundariesArray");
        proteus::pyarray<int>& elementBoundaryElementsArray = args.array<int>("elementBoundaryElementsArray");
        proteus::pyarray<int>& elementBoundaryLocalElementBoundariesArray = args.array<int>("elementBoundaryLocalElementBoundariesArray");
        proteus::pyarray<double>& ebqe_velocity_ext = args.array<double>("ebqe_velocity_ext");
        proteus::pyarray<int>& isDOFBoundary_u = args.array<int>("isDOFBoundary_u");
        proteus::pyarray<double>& ebqe_rd_u_ext = args.array<double>("ebqe_rd_u_ext");
        proteus::pyarray<double>& ebqe_bc_u_ext = args.array<double>("ebqe_bc_u_ext");
        proteus::pyarray<double>& ebqe_u = args.array<double>("ebqe_u");
        proteus::pyarray<double>& ebqe_grad_u = args.array<double>("ebqe_grad_u");
        proteus::pyarray<double>& interface_locator = args.array<double>("interface_locator");
        int PURE_BDF = args.scalar<int>("PURE_BDF");
        int numDOFs = args.scalar<int>("numDOFs");
        int NNZ = args.scalar<int>("NNZ");
        proteus::pyarray<int>& csrRowIndeces_DofLoops = args.array<int>("csrRowIndeces_DofLoops");
        proteus::pyarray<int>& csrColumnOffsets_DofLoops = args.array<int>("csrColumnOffsets_DofLoops");
        proteus::pyarray<int>& csrRowIndeces_CellLoops = args.array<int>("csrRowIndeces_CellLoops");
        proteus::pyarray<int>& csrColumnOffsets_CellLoops = args.array<int>("csrColumnOffsets_CellLoops");
        proteus::pyarray<int>& csrColumnOffsets_eb_CellLoops = args.array<int>("csrColumnOffsets_eb_CellLoops");
        int LUMPED_MASS_MATRIX = args.scalar<int>("LUMPED_MASS_MATRIX");
        proteus::pyarray<double>& quantDOFs = args.array<double>("quantDOFs");
        double lambda_coupez = args.scalar<double>("lambda_coupez");
        double epsCoupez = args.scalar<double>("epsCoupez");
        double epsFactRedistancing = args.scalar<double>("epsFactRedistancing");
        int COUPEZ = args.scalar<int>("COUPEZ");
        int SATURATED_LEVEL_SET = args.scalar<int>("SATURATED_LEVEL_SET");
        proteus::pyarray<double>& Cx = args.array<double>("Cx");
        proteus::pyarray<double>& Cy = args.array<double>("Cy");
        proteus::pyarray<double>& Cz = args.array<double>("Cz");
        proteus::pyarray<double>& ML = args.array<double>("ML");
        int STABILIZATION_TYPE = args.scalar<int>("STABILIZATION_TYPE");
        int ENTROPY_TYPE = args.scalar<int>("ENTROPY_TYPE");
        double cE = args.scalar<double>("cE");
        //cek should this be read in?
        double Ct_sge = 4.0;

        //loop over elements to compute volume integrals and load them into element and global residual
        //
        //eN is the element index
        //eN_k is the quadrature point index for a scalar
        //eN_k_nSpace is the quadrature point index for a vector
        //eN_i is the element test function index
        //eN_j is the element trial function index
        //eN_k_j is the quadrature point index for a trial function
        //eN_k_i is the quadrature point index for a trial function
        for(int eN=0;eN<nElements_global;eN++)
          {
            //declare local storage for element residual and initialize
            double elementResidual_u[nDOF_test_element];
            for (int i=0;i<nDOF_test_element;i++)
              {
                elementResidual_u[i]=0.0;
              }//i
            //loop over quadrature points and compute integrands
            for  (int k=0;k<nQuadraturePoints_element;k++)
              {
                //compute indeces and declare local storage
                int eN_k = eN*nQuadraturePoints_element+k,
                  eN_k_nSpace = eN_k*nSpace,
                  eN_nDOF_trial_element = eN*nDOF_trial_element;
                double u=0.0, u_old=0.0, grad_u[nSpace], grad_u_old[nSpace],
                  m=0.0, dm=0.0,
                  H=0.0, dH[nSpace],
                  f[nSpace],df[nSpace],//for MOVING_DOMAIN
                  m_t=0.0,dm_t=0.0,
                  pdeResidual_u=0.0,
                  Lstar_u[nDOF_test_element],
                  subgridError_u=0.0,
                  tau=0.0,tau0=0.0,tau1=0.0,
                  numDiff0=0.0,numDiff1=0.0,
                  jac[nSpace*nSpace],
                  jacDet,
                  jacInv[nSpace*nSpace],
                  u_grad_trial[nDOF_trial_element*nSpace],
                  u_test_dV[nDOF_trial_element],
                  u_grad_test_dV[nDOF_test_element*nSpace],
                  dV,x,y,z,xt,yt,zt,
                  G[nSpace*nSpace],G_dd_G,tr_G;//,norm_Rv;
                //
                //compute solution and gradients at quadrature points
                //
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
                ck.calculateMappingVelocity_element(eN,
                                                    k,
                                                    mesh_velocity_dof.data(),
                                                    mesh_l2g.data(),
                                                    mesh_trial_ref.data(),
                                                    xt,yt,zt);
                //get the physical integration weight
                dV = fabs(jacDet)*dV_ref.data()[k];
                ck.calculateG(jacInv,G,G_dd_G,tr_G);
                //get the trial function gradients
                ck.gradTrialFromRef(&u_grad_trial_ref.data()[k*nDOF_trial_element*nSpace],jacInv,u_grad_trial);
                //get the solution
                ck.valFromDOF(u_dof.data(),&u_l2g.data()[eN_nDOF_trial_element],&u_trial_ref.data()[k*nDOF_trial_element],u);
                //get the solution gradients
                ck.gradFromDOF(u_dof.data(),&u_l2g.data()[eN_nDOF_trial_element],u_grad_trial,grad_u);
                //precalculate test function products with integration weights
                for (int j=0;j<nDOF_trial_element;j++)
                  {
                    u_test_dV[j] = u_test_ref.data()[k*nDOF_trial_element+j]*dV;
                    for (int I=0;I<nSpace;I++)
                      {
                        u_grad_test_dV[j*nSpace+I]   = u_grad_trial[j*nSpace+I]*dV;//cek warning won't work for Petrov-Galerkin
                      }
                  }
                //save solution at quadrature points for other models to use
                q_u.data()[eN_k]=u;
                for (int I=0;I<nSpace;I++)
                  q_n.data()[eN_k_nSpace+I]  = grad_u[I];
                //
                //calculate pde coefficients at quadrature points
                //
                evaluateCoefficients(&velocity.data()[eN_k_nSpace],
                                     u,
                                     grad_u,
                                     m,
                                     dm,
                                     H,
                                     dH);
                //
                //moving mesh
                //
                double mesh_velocity[3];
                mesh_velocity[0] = xt;
                mesh_velocity[1] = yt;
                mesh_velocity[2] = zt;
                for (int I=0;I<nSpace;I++)
                  {
                    f[I] = -MOVING_DOMAIN*m*mesh_velocity[I];
                    df[I] = -MOVING_DOMAIN*dm*mesh_velocity[I];
                  }
                //
                //calculate time derivative at quadrature points
                //
                if (q_dV_last.data()[eN_k] <= -100)
                  q_dV_last.data()[eN_k] = dV;
                q_dV.data()[eN_k] = dV;
                ck.bdf(alphaBDF,
                       q_m_betaBDF.data()[eN_k],
                       m,
                       dm,
                       m_t,
                       dm_t);
                //
                //calculate subgrid error (strong residual and adjoint)
                //
                //calculate strong residual
                pdeResidual_u =
                  ck.Mass_strong(m_t) +
                  ck.Hamiltonian_strong(dH,grad_u)+
                  MOVING_DOMAIN*ck.Advection_strong(df,grad_u);//cek I don't think all mesh motion will be divergence free so we may need to go add the divergence

                //calculate adjoint
                for (int i=0;i<nDOF_test_element;i++)
                  {
                    //int eN_k_i_nSpace = (eN_k*nDOF_trial_element+i)*nSpace;
                    //Lstar_u[i]  = ck.Hamiltonian_adjoint(dH,&u_grad_test_dV[eN_k_i_nSpace]);
                    int i_nSpace = i*nSpace;
                    Lstar_u[i]  = ck.Hamiltonian_adjoint(dH,&u_grad_test_dV[i_nSpace]) + MOVING_DOMAIN*ck.Advection_adjoint(df,&u_grad_test_dV[i_nSpace]);
                  }
                //calculate tau and tau*Res
                double subgridErrorVelocity[nSpace];
                for (int I=0;I<nSpace;I++)
                  subgridErrorVelocity[I] = dH[I] - MOVING_DOMAIN*df[I];

                calculateSubgridError_tau(elementDiameter.data()[eN],
                                          dm_t,
                                          subgridErrorVelocity,//dH,
                                          cfl.data()[eN_k],
                                          tau0);

                calculateSubgridError_tau(Ct_sge,
                                          G,
                                          dm_t,
                                          subgridErrorVelocity,//dH,
                                          tau1,
                                          cfl.data()[eN_k]);

                tau = useMetrics*tau1+(1.0-useMetrics)*tau0;

                subgridError_u = -tau*pdeResidual_u;
                //
                //calculate shock capturing diffusion
                //
                ck.calculateNumericalDiffusion(shockCapturingDiffusion,elementDiameter.data()[eN],(pdeResidual_u - (sc_drop_time*ck.Mass_strong(m_t))),grad_u,numDiff0);
                ck.calculateNumericalDiffusion(shockCapturingDiffusion,sc_uref, sc_alpha,G,G_dd_G,(pdeResidual_u - (sc_drop_time*ck.Mass_strong(m_t))),grad_u,numDiff1);
                q_numDiff_u.data()[eN_k] = useMetrics*numDiff1+(1.0-useMetrics)*numDiff0;

                //
                //update element residual
                //
                double sign;
                int same_sign=1;
                for(int i=0;i<nDOF_test_element;i++)
                  {
                    int gi = offset_u+stride_u*u_l2g.data()[eN*nDOF_test_element+i];
                    if (i==0)
                      sign = Sign(u_dof.data()[gi]);
                    else if (same_sign==1)
                      {
                        same_sign = sign == Sign(u_dof.data()[gi]) ? 1 : 0;
                        sign = Sign(u_dof.data()[gi]);
                      }             
                    //int eN_k_i=eN_k*nDOF_test_element+i,
                    // eN_k_i_nSpace = eN_k_i*nSpace;
                    int  i_nSpace=i*nSpace;

                    elementResidual_u[i] +=
                      ck.Mass_weak(m_t,u_test_dV[i]) +
                      ck.Hamiltonian_weak(H,u_test_dV[i]) +
                      MOVING_DOMAIN*ck.Advection_weak(f,&u_grad_test_dV[i_nSpace])+
                      (PURE_BDF == 1 ? 0. : 1.)*
                      (ck.SubgridError(subgridError_u,Lstar_u[i]) +
                       ck.NumericalDiffusion(q_numDiff_u_last.data()[eN_k],
                                             grad_u,
                                             &u_grad_test_dV[i_nSpace]));
                  }//i
                if (same_sign == 0) // This cell contains the interface
                  for(int i=0;i<nDOF_test_element;i++)
                    {
                      int gi = offset_u+stride_u*u_l2g.data()[eN*nDOF_test_element+i];
                      interface_locator.data()[gi] = 1.0;
                    }
                //
                //cek/ido todo, get rid of m, since u=m
                //save momentum for time history and velocity for subgrid error
                //save solution for other models
                //
                q_m.data()[eN_k] = m;
                q_u.data()[eN_k] = u;
                for (int I=0;I<nSpace;I++)
                  {
                    int eN_k_I = eN_k*nSpace+I;
                    q_dH.data()[eN_k_I] = dH[I];
                  }
              }
            //
            //load element into global residual and save element residual
            //
            for(int i=0;i<nDOF_test_element;i++)
              {
                int eN_i=eN*nDOF_test_element+i;
                globalResidual.data()[offset_u+stride_u*r_l2g.data()[eN_i]] += elementResidual_u[i];
              }//i
          }//elements
        //
        //loop over exterior element boundaries to calculate surface integrals and load into element and global residuals
        //
        //ebNE is the Exterior element boundary INdex
        //ebN is the element boundary INdex
        //eN is the element index
        //std::cout <<nExteriorElementBoundaries_global<<std::endl;
        for (int ebNE = 0; ebNE < nExteriorElementBoundaries_global; ebNE++)
          {
            int ebN = exteriorElementBoundariesArray.data()[ebNE],
              eN  = elementBoundaryElementsArray.data()[ebN*2+0],
              ebN_local = elementBoundaryLocalElementBoundariesArray.data()[ebN*2+0],
              eN_nDOF_trial_element = eN*nDOF_trial_element;
            double elementResidual_u[nDOF_test_element];
            for (int i=0;i<nDOF_test_element;i++)
              {
                elementResidual_u[i]=0.0;
              }
            for  (int kb=0;kb<nQuadraturePoints_elementBoundary;kb++)
              {
                int ebNE_kb = ebNE*nQuadraturePoints_elementBoundary+kb,
                  ebNE_kb_nSpace = ebNE_kb*nSpace,
                  ebN_local_kb = ebN_local*nQuadraturePoints_elementBoundary+kb,
                  ebN_local_kb_nSpace = ebN_local_kb*nSpace;
                double u_ext=0.0,
                  grad_u_ext[nSpace],
                  m_ext=0.0,
                  dm_ext=0.0,
                  H_ext=0.0,
                  dH_ext[nSpace],
                  //f_ext[nSpace],//MOVING_DOMAIN
                  //df_ext[nSpace],//MOVING_DOMAIN
                  //flux_ext=0.0,
                  bc_u_ext=0.0,
                  bc_grad_u_ext[nSpace],
                  bc_m_ext=0.0,
                  bc_dm_ext=0.0,
                  bc_H_ext=0.0,
                  bc_dH_ext[nSpace],
                  jac_ext[nSpace*nSpace],
                  jacDet_ext,
                  jacInv_ext[nSpace*nSpace],
                  boundaryJac[nSpace*(nSpace-1)],
                  metricTensor[(nSpace-1)*(nSpace-1)],
                  metricTensorDetSqrt,
                  dS,
                  u_test_dS[nDOF_test_element],
                  u_grad_trial_trace[nDOF_trial_element*nSpace],
                  normal[nSpace],x_ext,y_ext,z_ext,xt_ext,yt_ext,zt_ext,integralScaling,
                  G[nSpace*nSpace],G_dd_G,tr_G,flux_ext;
                //
                //calculate the solution and gradients at quadrature points
                //
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
                dS = ((1.0-MOVING_DOMAIN)*metricTensorDetSqrt + MOVING_DOMAIN*integralScaling)*dS_ref.data()[kb];
                //get the metric tensor
                //cek todo use symmetry
                ck.calculateG(jacInv_ext,G,G_dd_G,tr_G);
                //compute shape and solution information
                //shape
                ck.gradTrialFromRef(&u_grad_trial_trace_ref.data()[ebN_local_kb_nSpace*nDOF_trial_element],jacInv_ext,u_grad_trial_trace);
                //solution and gradients
                ck.valFromDOF(u_dof.data(),&u_l2g.data()[eN_nDOF_trial_element],&u_trial_trace_ref.data()[ebN_local_kb*nDOF_test_element],u_ext);
                ck.gradFromDOF(u_dof.data(),&u_l2g.data()[eN_nDOF_trial_element],u_grad_trial_trace,grad_u_ext);
                //precalculate test function products with integration weights
                for (int j=0;j<nDOF_trial_element;j++)
                  {
                    u_test_dS[j] = u_test_trace_ref.data()[ebN_local_kb*nDOF_test_element+j]*dS;
                  }
                //
                //load the boundary values
                //
                bc_u_ext = isDOFBoundary_u.data()[ebNE_kb]*ebqe_bc_u_ext.data()[ebNE_kb]+(1-isDOFBoundary_u.data()[ebNE_kb])*ebqe_rd_u_ext.data()[ebNE_kb];
                //
                //calculate the pde coefficients using the solution and the boundary values for the solution
                //
                evaluateCoefficients(&ebqe_velocity_ext.data()[ebNE_kb_nSpace],
                                     u_ext,
                                     grad_u_ext,
                                     m_ext,
                                     dm_ext,
                                     H_ext,
                                     dH_ext);
                evaluateCoefficients(&ebqe_velocity_ext.data()[ebNE_kb_nSpace],
                                     bc_u_ext,
                                     bc_grad_u_ext,
                                     bc_m_ext,
                                     bc_dm_ext,
                                     bc_H_ext,
                                     bc_dH_ext);
                //
                //moving mesh
                //
                double velocity_ext[nSpace];
                double mesh_velocity[3];
                mesh_velocity[0] = xt_ext;
                mesh_velocity[1] = yt_ext;
                mesh_velocity[2] = zt_ext;
                for (int I=0;I<nSpace;I++)
                  velocity_ext[I] = - MOVING_DOMAIN*mesh_velocity[I];
                //
                //calculate the numerical fluxes
                //
                exteriorNumericalFlux(normal,
                                      bc_u_ext,
                                      u_ext,
                                      dH_ext,
                                      velocity_ext,
                                      flux_ext);
                ebqe_u.data()[ebNE_kb] = u_ext;

                // gradient //

                double norm = 1.0e-8;
                for (int I=0;I<nSpace;I++)
                  norm += grad_u_ext[I]*grad_u_ext[I];
                norm = sqrt(norm);
                for (int I=0;I<nSpace;I++)
                  ebqe_grad_u.data()[ebNE_kb_nSpace+I]  = grad_u_ext[I]/norm;

                //
                //update residuals
                //
                for (int i=0;i<nDOF_test_element;i++)
                  {
                    //int ebNE_kb_i = ebNE_kb*nDOF_test_element+i;
                    elementResidual_u[i] += ck.ExteriorElementBoundaryFlux(flux_ext,u_test_dS[i]);
                  }//i
              }//kb
            //
            //update the element and global residual storage
            //
            for (int i=0;i<nDOF_test_element;i++)
              {
                int eN_i = eN*nDOF_test_element+i;
                //globalResidual.data()[offset_u+stride_u*u_l2g.data()[eN_i]] += MOVING_DOMAIN*elementResidual_u[i];
                globalResidual.data()[offset_u+stride_u*r_l2g.data()[eN_i]] += elementResidual_u[i];

              }//i
          }//ebNE
      }

      void calculateJacobian(arguments_dict& args)
      {
        double dt = args.scalar<double>("dt");
        proteus::pyarray<double>& mesh_trial_ref = args.array<double>("mesh_trial_ref");
        proteus::pyarray<double>& mesh_grad_trial_ref = args.array<double>("mesh_grad_trial_ref");
        proteus::pyarray<double>& mesh_dof = args.array<double>("mesh_dof");
        proteus::pyarray<double>& mesh_velocity_dof = args.array<double>("mesh_velocity_dof");
        double MOVING_DOMAIN = args.scalar<double>("MOVING_DOMAIN");
        proteus::pyarray<int>& mesh_l2g = args.array<int>("mesh_l2g");
        proteus::pyarray<double>& dV_ref = args.array<double>("dV_ref");
        proteus::pyarray<double>& u_trial_ref = args.array<double>("u_trial_ref");
        proteus::pyarray<double>& u_grad_trial_ref = args.array<double>("u_grad_trial_ref");
        proteus::pyarray<double>& u_test_ref = args.array<double>("u_test_ref");
        proteus::pyarray<double>& u_grad_test_ref = args.array<double>("u_grad_test_ref");
        proteus::pyarray<double>& mesh_trial_trace_ref = args.array<double>("mesh_trial_trace_ref");
        proteus::pyarray<double>& mesh_grad_trial_trace_ref = args.array<double>("mesh_grad_trial_trace_ref");
        proteus::pyarray<double>& dS_ref = args.array<double>("dS_ref");
        proteus::pyarray<double>& u_trial_trace_ref = args.array<double>("u_trial_trace_ref");
        proteus::pyarray<double>& u_grad_trial_trace_ref = args.array<double>("u_grad_trial_trace_ref");
        proteus::pyarray<double>& u_test_trace_ref = args.array<double>("u_test_trace_ref");
        proteus::pyarray<double>& u_grad_test_trace_ref = args.array<double>("u_grad_test_trace_ref");
        proteus::pyarray<double>& normal_ref = args.array<double>("normal_ref");
        proteus::pyarray<double>& boundaryJac_ref = args.array<double>("boundaryJac_ref");
        int nElements_global = args.scalar<int>("nElements_global");
        double useMetrics = args.scalar<double>("useMetrics");
        double alphaBDF = args.scalar<double>("alphaBDF");
        int lag_shockCapturing = args.scalar<int>("lag_shockCapturing");
        double shockCapturingDiffusion = args.scalar<double>("shockCapturingDiffusion");
        proteus::pyarray<int>& u_l2g = args.array<int>("u_l2g");
        proteus::pyarray<int>& r_l2g = args.array<int>("r_l2g");
        proteus::pyarray<double>& elementDiameter = args.array<double>("elementDiameter");
        int degree_polynomial = args.scalar<int>("degree_polynomial");
        proteus::pyarray<double>& u_dof = args.array<double>("u_dof");
        proteus::pyarray<double>& velocity = args.array<double>("velocity");
        proteus::pyarray<double>& q_m_betaBDF = args.array<double>("q_m_betaBDF");
        proteus::pyarray<double>& cfl = args.array<double>("cfl");
        proteus::pyarray<double>& q_numDiff_u_last = args.array<double>("q_numDiff_u_last");
        proteus::pyarray<int>& csrRowIndeces_u_u = args.array<int>("csrRowIndeces_u_u");
        proteus::pyarray<int>& csrColumnOffsets_u_u = args.array<int>("csrColumnOffsets_u_u");
        proteus::pyarray<double>& globalJacobian = args.array<double>("globalJacobian");
        int nExteriorElementBoundaries_global = args.scalar<int>("nExteriorElementBoundaries_global");
        proteus::pyarray<int>& exteriorElementBoundariesArray = args.array<int>("exteriorElementBoundariesArray");
        proteus::pyarray<int>& elementBoundaryElementsArray = args.array<int>("elementBoundaryElementsArray");
        proteus::pyarray<int>& elementBoundaryLocalElementBoundariesArray = args.array<int>("elementBoundaryLocalElementBoundariesArray");
        proteus::pyarray<double>& ebqe_velocity_ext = args.array<double>("ebqe_velocity_ext");
        proteus::pyarray<int>& isDOFBoundary_u = args.array<int>("isDOFBoundary_u");
        proteus::pyarray<double>& ebqe_rd_u_ext = args.array<double>("ebqe_rd_u_ext");
        proteus::pyarray<double>& ebqe_bc_u_ext = args.array<double>("ebqe_bc_u_ext");
        proteus::pyarray<int>& csrColumnOffsets_eb_u_u = args.array<int>("csrColumnOffsets_eb_u_u");
        int PURE_BDF = args.scalar<int>("PURE_BDF");
        int LUMPED_MASS_MATRIX = args.scalar<int>("LUMPED_MASS_MATRIX");
        double Ct_sge = 4.0;

        //
        //loop over elements to compute volume integrals and load them into the element Jacobians and global Jacobian
        //
        for(int eN=0;eN<nElements_global;eN++)
          {
            double  elementJacobian_u_u[nDOF_test_element][nDOF_trial_element];
            for (int i=0;i<nDOF_test_element;i++)
              for (int j=0;j<nDOF_trial_element;j++)
                {
                  elementJacobian_u_u[i][j]=0.0;
                }
            for  (int k=0;k<nQuadraturePoints_element;k++)
              {
                int eN_k = eN*nQuadraturePoints_element+k, //index to a scalar at a quadrature point
                  eN_k_nSpace = eN_k*nSpace,
                  eN_nDOF_trial_element = eN*nDOF_trial_element; //index to a vector at a quadrature point

                //declare local storage
                double u=0.0,
                  grad_u[nSpace],
                  m=0.0,dm=0.0,
                  H=0.0,dH[nSpace],
                  f[nSpace],df[nSpace],//MOVING_MESH
                  m_t=0.0,dm_t=0.0,
                  dpdeResidual_u_u[nDOF_trial_element],
                  Lstar_u[nDOF_test_element],
                  dsubgridError_u_u[nDOF_trial_element],
                  tau=0.0,tau0=0.0,tau1=0.0,
                  jac[nSpace*nSpace],
                  jacDet,
                  jacInv[nSpace*nSpace],
                  u_grad_trial[nDOF_trial_element*nSpace],
                  dV,
                  u_test_dV[nDOF_test_element],
                  u_grad_test_dV[nDOF_test_element*nSpace],
                  x,y,z,xt,yt,zt,
                  G[nSpace*nSpace],G_dd_G,tr_G;
                //
                //calculate solution and gradients at quadrature points
                //
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
                ck.calculateMappingVelocity_element(eN,
                                                    k,
                                                    mesh_velocity_dof.data(),
                                                    mesh_l2g.data(),
                                                    mesh_trial_ref.data(),
                                                    xt,yt,zt);
                //get the physical integration weight
                dV = fabs(jacDet)*dV_ref.data()[k];
                ck.calculateG(jacInv,G,G_dd_G,tr_G);
                //get the trial function gradients
                ck.gradTrialFromRef(&u_grad_trial_ref.data()[k*nDOF_trial_element*nSpace],jacInv,u_grad_trial);
                //get the solution
                ck.valFromDOF(u_dof.data(),&u_l2g.data()[eN_nDOF_trial_element],&u_trial_ref.data()[k*nDOF_trial_element],u);
                //get the solution gradients
                ck.gradFromDOF(u_dof.data(),&u_l2g.data()[eN_nDOF_trial_element],u_grad_trial,grad_u);
                //precalculate test function products with integration weights
                for (int j=0;j<nDOF_trial_element;j++)
                  {
                    u_test_dV[j] = u_test_ref.data()[k*nDOF_trial_element+j]*dV;
                    for (int I=0;I<nSpace;I++)
                      {
                        u_grad_test_dV[j*nSpace+I]   = u_grad_trial[j*nSpace+I]*dV;//cek warning won't work for Petrov-Galerkin
                      }
                  }
                //
                //calculate pde coefficients and derivatives at quadrature points
                //
                evaluateCoefficients(&velocity.data()[eN_k_nSpace],
                                     u,
                                     grad_u,
                                     m,
                                     dm,
                                     H,
                                     dH);
                //
                //moving mesh
                //
                double mesh_velocity[3];
                mesh_velocity[0] = xt;
                mesh_velocity[1] = yt;
                mesh_velocity[2] = zt;
                for (int I=0;I<nSpace;I++)
                  {
                    f[I] = -MOVING_DOMAIN*m*mesh_velocity[I];
                    df[I] = -MOVING_DOMAIN*dm*mesh_velocity[I];
                  }
                //
                //calculate time derivatives
                //
                ck.bdf(alphaBDF,
                       q_m_betaBDF.data()[eN_k],
                       m,
                       dm,
                       m_t,
                       dm_t);
                //
                //calculate subgrid error contribution to the Jacobian (strong residual, adjoint, jacobian of strong residual)
                //
                //calculate the adjoint times the test functions
                for (int i=0;i<nDOF_test_element;i++)
                  {
                    //int eN_k_i_nSpace = (eN_k*nDOF_trial_element+i)*nSpace;
                    //Lstar_u[i]=ck.Hamiltonian_adjoint(dH,&u_grad_test_dV[eN_k_i_nSpace]);
                    int i_nSpace = i*nSpace;
                    Lstar_u[i]=ck.Hamiltonian_adjoint(dH,&u_grad_test_dV[i_nSpace]) + MOVING_DOMAIN*ck.Advection_adjoint(df,&u_grad_test_dV[i_nSpace]);

                  }
                //calculate the Jacobian of strong residual
                for (int j=0;j<nDOF_trial_element;j++)
                  {
                    //int eN_k_j=eN_k*nDOF_trial_element+j;
                    //int eN_k_j_nSpace = eN_k_j*nSpace;
                    int j_nSpace = j*nSpace;
                    dpdeResidual_u_u[j]=ck.MassJacobian_strong(dm_t,u_trial_ref.data()[k*nDOF_trial_element+j]) +
                      ck.HamiltonianJacobian_strong(dH,&u_grad_trial[j_nSpace]) +
                      MOVING_DOMAIN*ck.AdvectionJacobian_strong(df,&u_grad_trial[j_nSpace]);

                  }
                //tau and tau*Res
                double subgridErrorVelocity[nSpace];
                for (int I=0;I<nSpace;I++)
                  subgridErrorVelocity[I] = dH[I] - MOVING_DOMAIN*df[I];

                calculateSubgridError_tau(elementDiameter.data()[eN],
                                          dm_t,
                                          subgridErrorVelocity,//dH,
                                          cfl.data()[eN_k],
                                          tau0);

                calculateSubgridError_tau(Ct_sge,
                                          G,
                                          dm_t,
                                          subgridErrorVelocity,//dH,
                                          tau1,
                                          cfl.data()[eN_k]);
                tau = useMetrics*tau1+(1.0-useMetrics)*tau0;

                for(int j=0;j<nDOF_trial_element;j++)
                  dsubgridError_u_u[j] = -tau*dpdeResidual_u_u[j];
                for(int i=0;i<nDOF_test_element;i++)
                  {
                    //int eN_k_i=eN_k*nDOF_test_element+i;
                    //int eN_k_i_nSpace=eN_k_i*nSpace;
                    for(int j=0;j<nDOF_trial_element;j++)
                      {
                        //int eN_k_j=eN_k*nDOF_trial_element+j;
                        //int eN_k_j_nSpace = eN_k_j*nSpace;
                        int j_nSpace = j*nSpace;
                        int i_nSpace = i*nSpace;
                        elementJacobian_u_u[i][j] +=
                          ck.MassJacobian_weak(dm_t,
                                               u_trial_ref.data()[k*nDOF_trial_element+j],
                                               u_test_dV[i]) +
                          ck.HamiltonianJacobian_weak(dH,&u_grad_trial[j_nSpace],u_test_dV[i]) +
                          MOVING_DOMAIN*ck.AdvectionJacobian_weak(df,
                                                                  u_trial_ref.data()[k*nDOF_trial_element+j],
                                                                  &u_grad_test_dV[i_nSpace]) +
                          (PURE_BDF == 1 ? 0. : 1.)*
                          (ck.SubgridErrorJacobian(dsubgridError_u_u[j],Lstar_u[i]) +
                           ck.NumericalDiffusionJacobian(q_numDiff_u_last.data()[eN_k],
                                                         &u_grad_trial[j_nSpace],
                                                         &u_grad_test_dV[i_nSpace]));
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
                    globalJacobian.data()[csrRowIndeces_u_u.data()[eN_i] + csrColumnOffsets_u_u.data()[eN_i_j]] += elementJacobian_u_u[i][j];
                  }//j
              }//i
          }//elements
        //
        //loop over exterior element boundaries to compute the surface integrals and load them into the global Jacobian
        //
        for (int ebNE = 0; ebNE < nExteriorElementBoundaries_global; ebNE++)
          {
            int ebN = exteriorElementBoundariesArray.data()[ebNE];
            int eN  = elementBoundaryElementsArray.data()[ebN*2+0],
              ebN_local = elementBoundaryLocalElementBoundariesArray.data()[ebN*2+0],
              eN_nDOF_trial_element = eN*nDOF_trial_element;
            for  (int kb=0;kb<nQuadraturePoints_elementBoundary;kb++)
              {
                int ebNE_kb = ebNE*nQuadraturePoints_elementBoundary+kb,
                  ebNE_kb_nSpace = ebNE_kb*nSpace,
                  ebN_local_kb = ebN_local*nQuadraturePoints_elementBoundary+kb,
                  ebN_local_kb_nSpace = ebN_local_kb*nSpace;

                double u_ext=0.0,
                  grad_u_ext[nSpace],
                  m_ext=0.0,
                  dm_ext=0.0,
                  H_ext=0.0,
                  dH_ext[nSpace],
                  bc_grad_u_ext[nSpace],
                  bc_H_ext=0.0,
                  bc_dH_ext[nSpace],
                  //f_ext[nSpace],
                  //df_ext[nSpace],
                  dflux_u_u_ext=0.0,
                  bc_u_ext=0.0,
                  //bc_grad_u_ext[nSpace],
                  bc_m_ext=0.0,
                  bc_dm_ext=0.0,
                  //bc_f_ext[nSpace],
                  //bc_df_ext[nSpace],
                  fluxJacobian_u_u[nDOF_trial_element],
                  jac_ext[nSpace*nSpace],
                  jacDet_ext,
                  jacInv_ext[nSpace*nSpace],
                  boundaryJac[nSpace*(nSpace-1)],
                  metricTensor[(nSpace-1)*(nSpace-1)],
                  metricTensorDetSqrt,
                  dS,
                  u_test_dS[nDOF_test_element],
                  u_grad_trial_trace[nDOF_trial_element*nSpace],
                  normal[nSpace],x_ext,y_ext,z_ext,xt_ext,yt_ext,zt_ext,integralScaling,
                  G[nSpace*nSpace],G_dd_G,tr_G;
                //
                //calculate the solution and gradients at quadrature points
                //
                // u_ext=0.0;
                // for (int I=0;I<nSpace;I++)
                //   {
                //     grad_u_ext[I] = 0.0;
                //     bc_grad_u_ext[I] = 0.0;
                //   }
                // for (int j=0;j<nDOF_trial_element;j++)
                //   {
                //     int eN_j = eN*nDOF_trial_element+j,
                //       ebNE_kb_j = ebNE_kb*nDOF_trial_element+j,
                //       ebNE_kb_j_nSpace= ebNE_kb_j*nSpace;
                //     u_ext += valFromDOF_c(u_dof.data()[u_l2g.data()[eN_j]],u_trial_ext[ebNE_kb_j]);

                //     for (int I=0;I<nSpace;I++)
                //       {
                //         grad_u_ext[I] += gradFromDOF_c(u_dof.data()[u_l2g.data()[eN_j]],u_grad_trial_ext[ebNE_kb_j_nSpace+I]);
                //       }
                //   }
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
                dS = ((1.0-MOVING_DOMAIN)*metricTensorDetSqrt + MOVING_DOMAIN*integralScaling)*dS_ref.data()[kb];
                //dS = metricTensorDetSqrt*dS_ref.data()[kb];
                ck.calculateG(jacInv_ext,G,G_dd_G,tr_G);
                //compute shape and solution information
                //shape
                ck.gradTrialFromRef(&u_grad_trial_trace_ref.data()[ebN_local_kb_nSpace*nDOF_trial_element],jacInv_ext,u_grad_trial_trace);
                //solution and gradients
                ck.valFromDOF(u_dof.data(),&u_l2g.data()[eN_nDOF_trial_element],&u_trial_trace_ref.data()[ebN_local_kb*nDOF_test_element],u_ext);
                ck.gradFromDOF(u_dof.data(),&u_l2g.data()[eN_nDOF_trial_element],u_grad_trial_trace,grad_u_ext);
                //precalculate test function products with integration weights
                for (int j=0;j<nDOF_trial_element;j++)
                  {
                    u_test_dS[j] = u_test_trace_ref.data()[ebN_local_kb*nDOF_test_element+j]*dS;
                  }
                //
                //load the boundary values
                //
                bc_u_ext = isDOFBoundary_u.data()[ebNE_kb]*ebqe_bc_u_ext.data()[ebNE_kb]+(1-isDOFBoundary_u.data()[ebNE_kb])*ebqe_rd_u_ext.data()[ebNE_kb];
                //
                //calculate the internal and external trace of the pde coefficients
                //
                evaluateCoefficients(&ebqe_velocity_ext.data()[ebNE_kb_nSpace],
                                     u_ext,
                                     grad_u_ext,
                                     m_ext,
                                     dm_ext,
                                     H_ext,
                                     dH_ext);
                evaluateCoefficients(&ebqe_velocity_ext.data()[ebNE_kb_nSpace],
                                     bc_u_ext,
                                     bc_grad_u_ext,
                                     bc_m_ext,
                                     bc_dm_ext,
                                     bc_H_ext,
                                     bc_dH_ext);
                //
                //moving domain
                //
                double velocity_ext[nSpace];
                double mesh_velocity[3];
                mesh_velocity[0] = xt_ext;
                mesh_velocity[1] = yt_ext;
                mesh_velocity[2] = zt_ext;
                for (int I=0;I<nSpace;I++)
                  {
                    velocity_ext[I] = - MOVING_DOMAIN*mesh_velocity[I];
                  }
                //
                //calculate the numerical fluxes
                //
                exteriorNumericalFluxDerivative(normal,
                                                dH_ext,
                                                velocity_ext,
                                                dflux_u_u_ext);
                //
                //calculate the flux jacobian
                //
                for (int j=0;j<nDOF_trial_element;j++)
                  {
                    //int ebNE_kb_j = ebNE_kb*nDOF_trial_element+j;
                    int ebN_local_kb_j=ebN_local_kb*nDOF_trial_element+j;
                    fluxJacobian_u_u[j]=ck.ExteriorNumericalAdvectiveFluxJacobian(dflux_u_u_ext,u_trial_trace_ref.data()[ebN_local_kb_j]);
                  }//j
                //
                //update the global Jacobian from the flux Jacobian
                //
                for (int i=0;i<nDOF_test_element;i++)
                  {
                    int eN_i = eN*nDOF_test_element+i;
                    //int ebNE_kb_i = ebNE_kb*nDOF_test_element+i;
                    for (int j=0;j<nDOF_trial_element;j++)
                      {
                        int ebN_i_j = ebN*4*nDOF_test_X_trial_element + i*nDOF_trial_element + j;
                        globalJacobian.data()[csrRowIndeces_u_u.data()[eN_i] + csrColumnOffsets_eb_u_u.data()[ebN_i_j]] += fluxJacobian_u_u[j]*u_test_dS[i];
                        //globalJacobian.data()[csrRowIndeces_u_u.data()[eN_i] + csrColumnOffsets_eb_u_u.data()[ebN_i_j]] += MOVING_DOMAIN*fluxJacobian_u_u[j]*u_test_dS[i];
                      }//j
                  }//i
              }//kb
          }//ebNE
      }//computeJacobian

      void calculateWaterline(arguments_dict& args)
      {
        proteus::pyarray<int>& wlc = args.array<int>("wlc");
        proteus::pyarray<double>& waterline = args.array<double>("waterline");
        proteus::pyarray<double>& mesh_trial_ref = args.array<double>("mesh_trial_ref");
        proteus::pyarray<double>& mesh_grad_trial_ref = args.array<double>("mesh_grad_trial_ref");
        proteus::pyarray<double>& mesh_dof = args.array<double>("mesh_dof");
        proteus::pyarray<double>& mesh_velocity_dof = args.array<double>("mesh_velocity_dof");
        double MOVING_DOMAIN = args.scalar<double>("MOVING_DOMAIN");
        proteus::pyarray<int>& mesh_l2g = args.array<int>("mesh_l2g");
        proteus::pyarray<double>& dV_ref = args.array<double>("dV_ref");
        proteus::pyarray<double>& u_trial_ref = args.array<double>("u_trial_ref");
        proteus::pyarray<double>& u_grad_trial_ref = args.array<double>("u_grad_trial_ref");
        proteus::pyarray<double>& u_test_ref = args.array<double>("u_test_ref");
        proteus::pyarray<double>& u_grad_test_ref = args.array<double>("u_grad_test_ref");
        proteus::pyarray<double>& mesh_trial_trace_ref = args.array<double>("mesh_trial_trace_ref");
        proteus::pyarray<double>& mesh_grad_trial_trace_ref = args.array<double>("mesh_grad_trial_trace_ref");
        proteus::pyarray<double>& dS_ref = args.array<double>("dS_ref");
        proteus::pyarray<double>& u_trial_trace_ref = args.array<double>("u_trial_trace_ref");
        proteus::pyarray<double>& u_grad_trial_trace_ref = args.array<double>("u_grad_trial_trace_ref");
        proteus::pyarray<double>& u_test_trace_ref = args.array<double>("u_test_trace_ref");
        proteus::pyarray<double>& u_grad_test_trace_ref = args.array<double>("u_grad_test_trace_ref");
        proteus::pyarray<double>& normal_ref = args.array<double>("normal_ref");
        proteus::pyarray<double>& boundaryJac_ref = args.array<double>("boundaryJac_ref");
        int nElements_global = args.scalar<int>("nElements_global");
        double useMetrics = args.scalar<double>("useMetrics");
        double alphaBDF = args.scalar<double>("alphaBDF");
        int lag_shockCapturing = args.scalar<int>("lag_shockCapturing");
        double shockCapturingDiffusion = args.scalar<double>("shockCapturingDiffusion");
        double sc_uref = args.scalar<double>("sc_uref");
        double sc_alpha = args.scalar<double>("sc_alpha");
        proteus::pyarray<int>& u_l2g = args.array<int>("u_l2g");
        proteus::pyarray<int>& r_l2g = args.array<int>("r_l2g");
        proteus::pyarray<double>& elementDiameter = args.array<double>("elementDiameter");
        proteus::pyarray<double>& u_dof = args.array<double>("u_dof");
        proteus::pyarray<double>& u_dof_old = args.array<double>("u_dof_old");
        proteus::pyarray<double>& velocity = args.array<double>("velocity");
        proteus::pyarray<double>& q_m = args.array<double>("q_m");
        proteus::pyarray<double>& q_u = args.array<double>("q_u");
        proteus::pyarray<double>& q_n = args.array<double>("q_n");
        proteus::pyarray<double>& q_dH = args.array<double>("q_dH");
        proteus::pyarray<double>& q_m_betaBDF = args.array<double>("q_m_betaBDF");
        proteus::pyarray<double>& cfl = args.array<double>("cfl");
        proteus::pyarray<double>& q_numDiff_u = args.array<double>("q_numDiff_u");
        proteus::pyarray<double>& q_numDiff_u_last = args.array<double>("q_numDiff_u_last");
        int offset_u = args.scalar<int>("offset_u");
        int stride_u = args.scalar<int>("stride_u");
        int nExteriorElementBoundaries_global = args.scalar<int>("nExteriorElementBoundaries_global");
        proteus::pyarray<int>& exteriorElementBoundariesArray = args.array<int>("exteriorElementBoundariesArray");
        proteus::pyarray<int>& elementBoundaryElementsArray = args.array<int>("elementBoundaryElementsArray");
        proteus::pyarray<int>& elementBoundaryLocalElementBoundariesArray = args.array<int>("elementBoundaryLocalElementBoundariesArray");
        proteus::pyarray<int>& elementBoundaryMaterialTypes = args.array<int>("elementBoundaryMaterialTypes");
        proteus::pyarray<double>& ebqe_velocity_ext = args.array<double>("ebqe_velocity_ext");
        proteus::pyarray<int>& isDOFBoundary_u = args.array<int>("isDOFBoundary_u");
        proteus::pyarray<double>& ebqe_bc_u_ext = args.array<double>("ebqe_bc_u_ext");
        proteus::pyarray<double>& ebqe_u = args.array<double>("ebqe_u");
        //  Tetrehedral elements specific extraction routine for waterline extraction
        //  Loops over boundaries and checks if boundary is infact a hull (hardwired check if mattype > 6)
        //  Extracts the nodal values of boundary triangle (4th point is dropped = hardwired assumption we are dealing with linear tet)
        //  Then computes an average value and position for both negative and positive values
        //  If both positive and negative values re found, and we are actually in a triangle containing the interface
        //  a linear interpolation of negative and positive average is reported as interface location (exact in case of linear tets)

        for (int ebNE = 0; ebNE < nExteriorElementBoundaries_global; ebNE++)
          {
            int ebN = exteriorElementBoundariesArray.data()[ebNE];
            int eN  = elementBoundaryElementsArray.data()[ebN*2+0];
            int bN  = elementBoundaryLocalElementBoundariesArray.data()[ebN*2+0];

            if (elementBoundaryMaterialTypes.data()[ebN] >6)
              {

                double val,x,y,z;
                int pos=0, neg=0;
                double xn=0.0, yn=0.0, zn=0.0, vn=0.0;
                double xp=0.0, yp=0.0, zp=0.0, vp=0.0;

                for (int j=0;j<nDOF_trial_element;j++)
                  {
                    if (j != bN) {
                      int eN_nDOF_trial_element_j      = eN*nDOF_trial_element + j;
                      int eN_nDOF_mesh_trial_element_j = eN*nDOF_mesh_trial_element + j;
                      val = u_dof.data()[u_l2g.data()[eN_nDOF_trial_element_j]];
                      x = mesh_dof.data()[mesh_l2g.data()[eN_nDOF_mesh_trial_element_j]*3+0];
                      y = mesh_dof.data()[mesh_l2g.data()[eN_nDOF_mesh_trial_element_j]*3+1];
                      z = mesh_dof.data()[mesh_l2g.data()[eN_nDOF_mesh_trial_element_j]*3+2];

                      if (val < 0.0)
                        {
                          neg++;
                          vn+=val;
                          xn+=x;
                          yn+=y;
                          zn+=z;
                        }
                      else
                        {
                          pos++;
                          vp+=val;
                          xp+=x;
                          yp+=y;
                          zp+=z;
                        }
                    }
                  } // trail for


                if ((pos > 0) && (neg > 0) )
                  {
                    vp /= pos;
                    vn /= neg;

                    double alpha = vp/(vp -vn);

                    waterline.data()[wlc.data()[0]*3 + 0] =  alpha*(xn/neg) + (1.0-alpha)*(xp/pos);
                    waterline.data()[wlc.data()[0]*3 + 1] =  alpha*(yn/neg) + (1.0-alpha)*(yp/pos);
                    waterline.data()[wlc.data()[0]*3 + 2] =  alpha*(zn/neg) + (1.0-alpha)*(zp/pos);
                    wlc.data()[0]++;

                  } // end value if

              } // end bnd mat check

          }//ebNE

        //std::cout<<"CPP WLC "<<wlc.data()[0]<<std::endl;
      } // calcWaterline

      double calculateRedistancingResidual(arguments_dict& args)
      {
        double dt = args.scalar<double>("dt");
        proteus::pyarray<double>& mesh_trial_ref = args.array<double>("mesh_trial_ref");
        proteus::pyarray<double>& mesh_grad_trial_ref = args.array<double>("mesh_grad_trial_ref");
        proteus::pyarray<double>& mesh_dof = args.array<double>("mesh_dof");
        proteus::pyarray<int>& mesh_l2g = args.array<int>("mesh_l2g");
        proteus::pyarray<double>& dV_ref = args.array<double>("dV_ref");
        proteus::pyarray<double>& u_trial_ref = args.array<double>("u_trial_ref");
        proteus::pyarray<double>& u_grad_trial_ref = args.array<double>("u_grad_trial_ref");
        proteus::pyarray<double>& u_test_ref = args.array<double>("u_test_ref");
        int nElements_global = args.scalar<int>("nElements_global");
        proteus::pyarray<int>& u_l2g = args.array<int>("u_l2g");
        proteus::pyarray<int>& r_l2g = args.array<int>("r_l2g");
        proteus::pyarray<double>& elementDiameter = args.array<double>("elementDiameter");
        proteus::pyarray<double>& nodeDiametersArray = args.array<double>("nodeDiametersArray");
        proteus::pyarray<double>& u_dof = args.array<double>("u_dof");
        proteus::pyarray<double>& u_dof_old = args.array<double>("u_dof_old");
        proteus::pyarray<double>& uStar_dof = args.array<double>("uStar_dof");
        int offset_u = args.scalar<int>("offset_u");
        int stride_u = args.scalar<int>("stride_u");
        proteus::pyarray<double>& globalResidual = args.array<double>("globalResidual");
        int numDOFs = args.scalar<int>("numDOFs");
        int NNZ = args.scalar<int>("NNZ");
        proteus::pyarray<int>& csrRowIndeces_DofLoops = args.array<int>("csrRowIndeces_DofLoops");
        proteus::pyarray<int>& csrColumnOffsets_DofLoops = args.array<int>("csrColumnOffsets_DofLoops");
        proteus::pyarray<int>& csrRowIndeces_CellLoops = args.array<int>("csrRowIndeces_CellLoops");
        proteus::pyarray<int>& csrColumnOffsets_CellLoops = args.array<int>("csrColumnOffsets_CellLoops");
        double lambda_coupez = args.scalar<double>("lambda_coupez");
        double epsCoupez = args.scalar<double>("epsCoupez");
        double epsFactRedistancing = args.scalar<double>("epsFactRedistancing");
        proteus::pyarray<double>& edge_based_cfl = args.array<double>("edge_based_cfl");
        int SATURATED_LEVEL_SET = args.scalar<int>("SATURATED_LEVEL_SET");
        proteus::pyarray<double>& Cx = args.array<double>("Cx");
        proteus::pyarray<double>& Cy = args.array<double>("Cy");
        proteus::pyarray<double>& Cz = args.array<double>("Cz");
        proteus::pyarray<double>& ML = args.array<double>("ML");
        L2_norm_per_node.resize(numDOFs,0.0);
        double L2_norm=0.;
        for (int i=0; i<numDOFs; i++)
          L2_norm_per_node[i] = 0.;

        // Allocate space for the transport matrices
        TransportMatrix.resize(NNZ,0.0);
        TransposeTransportMatrix.resize(NNZ,0.0);

        //////////////////////////////////////////////
        // ** LOOP IN CELLS FOR CELL BASED TERMS ** //
        //////////////////////////////////////////////
        // HERE WE COMPUTE:
        //    * Time derivative term
        //    * Transport matrices
        for(int eN=0;eN<nElements_global;eN++)
          {
            //declare local storage for local contributions and initialize
            double
              elementResidual_u[nDOF_test_element], element_L2_norm_per_node[nDOF_test_element];
            double  elementTransport[nDOF_test_element][nDOF_trial_element];
            double  elementTransposeTransport[nDOF_test_element][nDOF_trial_element];
            for (int i=0;i<nDOF_test_element;i++)
              {
                elementResidual_u[i]=0.0;
                element_L2_norm_per_node[i]=0.;
                for (int j=0;j<nDOF_trial_element;j++)
                  {
                    elementTransport[i][j]=0.0;
                    elementTransposeTransport[i][j]=0.0;
                  }
              }
            //loop over quadrature points and compute integrands
            for  (int k=0;k<nQuadraturePoints_element;k++)
              {
                //compute indeces and declare local storage
                int eN_k = eN*nQuadraturePoints_element+k,
                  eN_k_nSpace = eN_k*nSpace,
                  eN_nDOF_trial_element = eN*nDOF_trial_element;
                double
                  hquad=0.0, u=0.0, u_test_dV[nDOF_trial_element], uStar=0.0,
                  //for entropy viscosity
                  un=0.0, grad_u[nSpace], grad_un[nSpace], vn[nSpace],
                  u_grad_trial[nDOF_trial_element*nSpace],
                  //for general use
                  jac[nSpace*nSpace], jacDet, jacInv[nSpace*nSpace],
                  dV,x,y,z;
                //get the physical integration weight
                ck.calculateMapping_element(eN,k,mesh_dof.data(),mesh_l2g.data(),mesh_trial_ref.data(),mesh_grad_trial_ref.data(),jac,jacDet,jacInv,x,y,z);
                dV = fabs(jacDet)*dV_ref.data()[k];
                // get h at quad points
                ck.valFromDOF(nodeDiametersArray.data(),&u_l2g.data()[eN_nDOF_trial_element],&u_trial_ref.data()[k*nDOF_trial_element],hquad);
                // get the star solution at quad points
                ck.valFromDOF(uStar_dof.data(),&u_l2g.data()[eN_nDOF_trial_element],&u_trial_ref.data()[k*nDOF_trial_element],uStar);
                //get the solution at quad points
                ck.valFromDOF(u_dof.data(),&u_l2g.data()[eN_nDOF_trial_element],&u_trial_ref.data()[k*nDOF_trial_element],u);
                //get the solution at quad point at tn
                ck.valFromDOF(u_dof_old.data(),&u_l2g.data()[eN_nDOF_trial_element],&u_trial_ref.data()[k*nDOF_trial_element],un);
                // solution and grads at old times at quad points
                ck.gradTrialFromRef(&u_grad_trial_ref.data()[k*nDOF_trial_element*nSpace],jacInv,u_grad_trial);
                ck.gradFromDOF(u_dof.data(),&u_l2g.data()[eN_nDOF_trial_element],u_grad_trial,grad_u);
                ck.gradFromDOF(u_dof_old.data(),&u_l2g.data()[eN_nDOF_trial_element],u_grad_trial,grad_un);
                //precalculate test function products with integration weights for mass matrix terms
                for (int j=0;j<nDOF_trial_element;j++)
                  u_test_dV[j] = u_test_ref.data()[k*nDOF_trial_element+j]*dV;

                double norm_grad_un = std::sqrt(std::pow(grad_un[0],2) + std::pow(grad_un[1],2)) + 1E-10;
                double norm_grad_u = std::sqrt(std::pow(grad_u[0],2) + std::pow(grad_u[1],2));
                double dist_error = norm_grad_u - (1-SATURATED_LEVEL_SET*std::pow(u/epsCoupez,2));

                double sgn = sign(uStar,epsFactRedistancing);
                vn[0] = lambda_coupez*sgn*grad_un[0]/norm_grad_un;
                vn[1] = lambda_coupez*sgn*grad_un[1]/norm_grad_un;

                //////////////
                // ith-LOOP //
                //////////////
                double residual = (u-un)
                  - dt*lambda_coupez*sgn*(1-SATURATED_LEVEL_SET*std::pow(un/epsCoupez,2));
                for(int i=0;i<nDOF_test_element;i++)
                  {
                    // lumped mass matrix
                    elementResidual_u[i] += residual*u_test_dV[i];
                    element_L2_norm_per_node[i] += dist_error*u_test_dV[i];
                    ///////////////
                    // j-th LOOP // To construct transport matrices
                    ///////////////
                    for(int j=0;j<nDOF_trial_element;j++)
                      {
                        int j_nSpace = j*nSpace;
                        int i_nSpace = i*nSpace;
                        // COMPUTE ELEMENT TRANSPORT MATRIX (MQL)
                        elementTransport[i][j] += // int[(vel.grad_wj)*wi*dx]
                          ck.HamiltonianJacobian_weak(vn,&u_grad_trial[j_nSpace],u_test_dV[i]);
                        elementTransposeTransport[i][j] += // int[(vel.grad_wi)*wj*dx]
                          ck.HamiltonianJacobian_weak(vn,&u_grad_trial[i_nSpace],u_test_dV[j]);
                      }
                  }//i
              }

            /////////////////
            // DISTRIBUTE // load cell based element into global residual
            ////////////////
            for(int i=0;i<nDOF_test_element;i++)
              {
                int eN_i=eN*nDOF_test_element+i;
                int gi = offset_u+stride_u*r_l2g.data()[eN_i]; //global i-th index

                // distribute global residual for (lumped) mass matrix
                globalResidual.data()[gi] += elementResidual_u[i];
                // distribute L2 norm per node
                L2_norm_per_node[gi] += element_L2_norm_per_node[i];
                // distribute transport matrices
                for (int j=0;j<nDOF_trial_element;j++)
                  {
                    int eN_i_j = eN_i*nDOF_trial_element+j;
                    TransportMatrix[csrRowIndeces_CellLoops.data()[eN_i] + csrColumnOffsets_CellLoops.data()[eN_i_j]]
                      += elementTransport[i][j];
                    TransposeTransportMatrix[csrRowIndeces_CellLoops.data()[eN_i] + csrColumnOffsets_CellLoops.data()[eN_i_j]]
                      += elementTransposeTransport[i][j];
                  }//j
              }//i
          }//elements

        //////////////////////////////////
        // COMPUTE SMOOTHNESS INDICATOR //
        //////////////////////////////////
        // Smoothness indicator is based on the solution.
        // psi_i = psi_i(alpha_i); alpha_i = |sum(betaij*(uj-ui))|/sum(betaij*|uj-ui|)
        psi.resize(numDOFs,0.0);
        for (int i=0; i<numDOFs; i++)
          {
            double alphai;
            double solni = u_dof_old.data()[i]; // solution at time tn for the ith DOF
            // for smoothness indicator
            double alpha_numi=0;
            double alpha_deni=0;
            for (int offset=csrRowIndeces_DofLoops.data()[i]; offset<csrRowIndeces_DofLoops.data()[i+1]; offset++)
              { //loop in j (sparsity pattern)
                int j = csrColumnOffsets_DofLoops.data()[offset];
                double solnj = u_dof_old.data()[j]; // solution at time tn for the jth DOF
                alpha_numi += solnj-solni;
                alpha_deni += fabs(solnj-solni);
              }
            alphai = (fabs(alpha_numi) == alpha_deni ? 1. : fabs(alpha_numi)/(alpha_deni+1E-15));
            if (POWER_SMOOTHNESS_INDICATOR==0)
              psi[i] = 1.0;
            else
              psi[i] = std::pow(alphai,POWER_SMOOTHNESS_INDICATOR); //NOTE: they use alpha^2 in the paper
            // compute L2 norm
            L2_norm += std::pow(L2_norm_per_node[i],2);
          }
        L2_norm = std::sqrt(L2_norm);
        // END OF COMPUTING SMOOTHNESS INDICATOR

        /////////////////////////////////////////////
        // ** LOOP IN DOFs FOR EDGE BASED TERMS ** //
        /////////////////////////////////////////////
        int ij=0;
        for (int i=0; i<numDOFs; i++)
          {
            double solni = u_dof_old.data()[i]; // solution at time tn for the ith DOF
            double ith_dissipative_term = 0;
            double ith_flux_term = 0;
            double dLii = 0.;

            // loop over the sparsity pattern of the i-th DOF
            for (int offset=csrRowIndeces_DofLoops.data()[i]; offset<csrRowIndeces_DofLoops.data()[i+1]; offset++)
              {
                int j = csrColumnOffsets_DofLoops.data()[offset];
                double solnj = u_dof_old.data()[j]; // solution at time tn for the jth DOF
                double dLij=0.;

                ith_flux_term += TransportMatrix[ij]*solnj;
                if (i != j) //NOTE: there is really no need to check for i!=j (see formula for ith_dissipative_term)
                  {
                    // first-order dissipative operator
                    dLij = fmax(fabs(TransportMatrix[ij]),fabs(TransposeTransportMatrix[ij]));
                    dLij *= fmax(psi[i],psi[j]);
                    //dissipative terms
                    ith_dissipative_term += dLij*(solnj-solni);
                    dLii -= dLij;
                  }
                //update ij
                ij+=1;
              }
            // update residual
            double mi = ML.data()[i];
            // compute edge_based_cfl
            edge_based_cfl.data()[i] = 2*fabs(dLii)/mi;
            globalResidual.data()[i] += dt*(ith_flux_term - ith_dissipative_term);
          }
        return L2_norm;
      }

      double calculateRhsSmoothing(arguments_dict& args)
      {
        proteus::pyarray<double>& mesh_trial_ref = args.array<double>("mesh_trial_ref");
        proteus::pyarray<double>& mesh_grad_trial_ref = args.array<double>("mesh_grad_trial_ref");
        proteus::pyarray<double>& mesh_dof = args.array<double>("mesh_dof");
        proteus::pyarray<int>& mesh_l2g = args.array<int>("mesh_l2g");
        proteus::pyarray<double>& dV_ref = args.array<double>("dV_ref");
        proteus::pyarray<double>& u_trial_ref = args.array<double>("u_trial_ref");
        proteus::pyarray<double>& u_grad_trial_ref = args.array<double>("u_grad_trial_ref");
        proteus::pyarray<double>& u_test_ref = args.array<double>("u_test_ref");
        int nElements_global = args.scalar<int>("nElements_global");
        proteus::pyarray<int>& u_l2g = args.array<int>("u_l2g");
        proteus::pyarray<int>& r_l2g = args.array<int>("r_l2g");
        proteus::pyarray<double>& elementDiameter = args.array<double>("elementDiameter");
        proteus::pyarray<double>& nodeDiametersArray = args.array<double>("nodeDiametersArray");
        proteus::pyarray<double>& u_dof_old = args.array<double>("u_dof_old");
        int offset_u = args.scalar<int>("offset_u");
        int stride_u = args.scalar<int>("stride_u");
        proteus::pyarray<double>& globalResidual = args.array<double>("globalResidual");
        //////////////////////////////////////////////
        // ** LOOP IN CELLS FOR CELL BASED TERMS ** //
        //////////////////////////////////////////////
        for(int eN=0;eN<nElements_global;eN++)
          {
            //declare local storage for local contributions and initialize
            double
              elementResidual_u[nDOF_test_element];
            for (int i=0;i<nDOF_test_element;i++)
              elementResidual_u[i]=0.0;

            //loop over quadrature points and compute integrands
            for  (int k=0;k<nQuadraturePoints_element;k++)
              {
                //compute indeces and declare local storage
                int eN_k = eN*nQuadraturePoints_element+k,
                  eN_k_nSpace = eN_k*nSpace,
                  eN_nDOF_trial_element = eN*nDOF_trial_element;
                double
                  un=0.0, u_test_dV[nDOF_trial_element], u_grad_trial[nDOF_trial_element*nSpace],
                  //for general use
                  jac[nSpace*nSpace], jacDet, jacInv[nSpace*nSpace],
                  dV,x,y,z;
                //get the physical integration weight
                ck.calculateMapping_element(eN,k,mesh_dof.data(),mesh_l2g.data(),mesh_trial_ref.data(),mesh_grad_trial_ref.data(),jac,jacDet,jacInv,x,y,z);
                dV = fabs(jacDet)*dV_ref.data()[k];
                //get the solution at quad point at tn
                ck.valFromDOF(u_dof_old.data(),&u_l2g.data()[eN_nDOF_trial_element],&u_trial_ref.data()[k*nDOF_trial_element],un);
                //precalculate test function products with integration weights for mass matrix terms
                for (int j=0;j<nDOF_trial_element;j++)
                  u_test_dV[j] = u_test_ref.data()[k*nDOF_trial_element+j]*dV;

                //////////////
                // ith-LOOP //
                //////////////
                for(int i=0;i<nDOF_test_element;i++)
                  elementResidual_u[i] += un*u_test_dV[i];
              }

            /////////////////
            // DISTRIBUTE // load cell based element into global residual
            ////////////////
            for(int i=0;i<nDOF_test_element;i++)
              {
                int eN_i=eN*nDOF_test_element+i;
                int gi = offset_u+stride_u*r_l2g.data()[eN_i]; //global i-th index

                // distribute global residual
                globalResidual.data()[gi] += elementResidual_u[i];
              }//i
          }//elements
      }

      void calculateResidual_entropy_viscosity(arguments_dict& args)
      {
        double dt = args.scalar<double>("dt");
        proteus::pyarray<double>& mesh_trial_ref = args.array<double>("mesh_trial_ref");
        proteus::pyarray<double>& mesh_grad_trial_ref = args.array<double>("mesh_grad_trial_ref");
        proteus::pyarray<double>& mesh_dof = args.array<double>("mesh_dof");
        proteus::pyarray<double>& mesh_velocity_dof = args.array<double>("mesh_velocity_dof");
        double MOVING_DOMAIN = args.scalar<double>("MOVING_DOMAIN");
        proteus::pyarray<int>& mesh_l2g = args.array<int>("mesh_l2g");
        proteus::pyarray<double>& dV_ref = args.array<double>("dV_ref");
        proteus::pyarray<double>& u_trial_ref = args.array<double>("u_trial_ref");
        proteus::pyarray<double>& u_grad_trial_ref = args.array<double>("u_grad_trial_ref");
        proteus::pyarray<double>& u_test_ref = args.array<double>("u_test_ref");
        proteus::pyarray<double>& u_grad_test_ref = args.array<double>("u_grad_test_ref");
        proteus::pyarray<double>& mesh_trial_trace_ref = args.array<double>("mesh_trial_trace_ref");
        proteus::pyarray<double>& mesh_grad_trial_trace_ref = args.array<double>("mesh_grad_trial_trace_ref");
        proteus::pyarray<double>& dS_ref = args.array<double>("dS_ref");
        proteus::pyarray<double>& u_trial_trace_ref = args.array<double>("u_trial_trace_ref");
        proteus::pyarray<double>& u_grad_trial_trace_ref = args.array<double>("u_grad_trial_trace_ref");
        proteus::pyarray<double>& u_test_trace_ref = args.array<double>("u_test_trace_ref");
        proteus::pyarray<double>& u_grad_test_trace_ref = args.array<double>("u_grad_test_trace_ref");
        proteus::pyarray<double>& normal_ref = args.array<double>("normal_ref");
        proteus::pyarray<double>& boundaryJac_ref = args.array<double>("boundaryJac_ref");
        int nElements_global = args.scalar<int>("nElements_global");
        double useMetrics = args.scalar<double>("useMetrics");
        double alphaBDF = args.scalar<double>("alphaBDF");
        int lag_shockCapturing = args.scalar<int>("lag_shockCapturing");
        double shockCapturingDiffusion = args.scalar<double>("shockCapturingDiffusion");
        double sc_uref = args.scalar<double>("sc_uref");
        double sc_alpha = args.scalar<double>("sc_alpha");
        proteus::pyarray<int>& u_l2g = args.array<int>("u_l2g");
        proteus::pyarray<int>& r_l2g = args.array<int>("r_l2g");
        proteus::pyarray<double>& elementDiameter = args.array<double>("elementDiameter");
        proteus::pyarray<double>& nodeDiametersArray = args.array<double>("nodeDiametersArray");
        int degree_polynomial = args.scalar<int>("degree_polynomial");
        proteus::pyarray<double>& u_dof = args.array<double>("u_dof");
        proteus::pyarray<double>& u_dof_old = args.array<double>("u_dof_old");
        proteus::pyarray<double>& uStar_dof = args.array<double>("uStar_dof");
        proteus::pyarray<double>& velocity = args.array<double>("velocity");
        proteus::pyarray<double>& q_m = args.array<double>("q_m");
        proteus::pyarray<double>& q_u = args.array<double>("q_u");
        proteus::pyarray<double>& q_n = args.array<double>("q_n");
        proteus::pyarray<double>& q_dH = args.array<double>("q_dH");
        proteus::pyarray<double>& q_m_betaBDF = args.array<double>("q_m_betaBDF");
        proteus::pyarray<double>& q_dV = args.array<double>("q_dV");
        proteus::pyarray<double>& q_dV_last = args.array<double>("q_dV_last");
        proteus::pyarray<double>& cfl = args.array<double>("cfl");
        proteus::pyarray<double>& edge_based_cfl = args.array<double>("edge_based_cfl");
        proteus::pyarray<double>& q_numDiff_u = args.array<double>("q_numDiff_u");
        proteus::pyarray<double>& q_numDiff_u_last = args.array<double>("q_numDiff_u_last");
        int offset_u = args.scalar<int>("offset_u");
        int stride_u = args.scalar<int>("stride_u");
        proteus::pyarray<double>& globalResidual = args.array<double>("globalResidual");
        int nExteriorElementBoundaries_global = args.scalar<int>("nExteriorElementBoundaries_global");
        proteus::pyarray<int>& exteriorElementBoundariesArray = args.array<int>("exteriorElementBoundariesArray");
        proteus::pyarray<int>& elementBoundaryElementsArray = args.array<int>("elementBoundaryElementsArray");
        proteus::pyarray<int>& elementBoundaryLocalElementBoundariesArray = args.array<int>("elementBoundaryLocalElementBoundariesArray");
        proteus::pyarray<double>& ebqe_velocity_ext = args.array<double>("ebqe_velocity_ext");
        proteus::pyarray<int>& isDOFBoundary_u = args.array<int>("isDOFBoundary_u");
        proteus::pyarray<double>& ebqe_rd_u_ext = args.array<double>("ebqe_rd_u_ext");
        proteus::pyarray<double>& ebqe_bc_u_ext = args.array<double>("ebqe_bc_u_ext");
        proteus::pyarray<double>& ebqe_u = args.array<double>("ebqe_u");
        proteus::pyarray<double>& ebqe_grad_u = args.array<double>("ebqe_grad_u");
        proteus::pyarray<double>& interface_locator = args.array<double>("interface_locator");
        int PURE_BDF = args.scalar<int>("PURE_BDF");
        int numDOFs = args.scalar<int>("numDOFs");
        int NNZ = args.scalar<int>("NNZ");
        proteus::pyarray<int>& csrRowIndeces_DofLoops = args.array<int>("csrRowIndeces_DofLoops");
        proteus::pyarray<int>& csrColumnOffsets_DofLoops = args.array<int>("csrColumnOffsets_DofLoops");
        proteus::pyarray<int>& csrRowIndeces_CellLoops = args.array<int>("csrRowIndeces_CellLoops");
        proteus::pyarray<int>& csrColumnOffsets_CellLoops = args.array<int>("csrColumnOffsets_CellLoops");
        proteus::pyarray<int>& csrColumnOffsets_eb_CellLoops = args.array<int>("csrColumnOffsets_eb_CellLoops");
        int LUMPED_MASS_MATRIX = args.scalar<int>("LUMPED_MASS_MATRIX");
        proteus::pyarray<double>& quantDOFs = args.array<double>("quantDOFs");
        double lambda_coupez = args.scalar<double>("lambda_coupez");
        double epsCoupez = args.scalar<double>("epsCoupez");
        double epsFactRedistancing = args.scalar<double>("epsFactRedistancing");
        int COUPEZ = args.scalar<int>("COUPEZ");
        int SATURATED_LEVEL_SET = args.scalar<int>("SATURATED_LEVEL_SET");
        proteus::pyarray<double>& Cx = args.array<double>("Cx");
        proteus::pyarray<double>& Cy = args.array<double>("Cy");
        proteus::pyarray<double>& Cz = args.array<double>("Cz");
        proteus::pyarray<double>& ML = args.array<double>("ML");
        int STABILIZATION_TYPE = args.scalar<int>("STABILIZATION_TYPE");
        int ENTROPY_TYPE = args.scalar<int>("ENTROPY_TYPE");
        double cE = args.scalar<double>("cE");
        // NOTE: This function follows a different (but equivalent) implementation of the smoothness based indicator than VOF.h
        // inverse of mass matrix in reference element
        //double inverse_elMassMatrix_degree2 [9][9]=
        //{
        //{81/4., 27/4., 9/4., 27/4., -81/4., -27/4., -27/4., -81/4., 81/4.},
        //{27/4., 81/4., 27/4., 9/4., -81/4., -81/4., -27/4., -27/4., 81/4.},
        //{9/4., 27/4., 81/4., 27/4., -27/4., -81/4., -81/4., -27/4., 81/4.},
        //{27/4., 9/4., 27/4., 81/4., -27/4., -27/4., -81/4., -81/4., 81/4.},
        //{-81/4., -81/4., -27/4., -27/4., 189/4., 81/4., 63/4., 81/4., -189/4.},
        //{-27/4., -81/4., -81/4., -27/4., 81/4., 189/4., 81/4., 63/4., -189/4.},
        //{-27/4., -27/4., -81/4., -81/4., 63/4., 81/4., 189/4., 81/4., -189/4.},
        //{-81/4., -27/4., -27/4., -81/4., 81/4., 63/4., 81/4., 189/4., -189/4.},
        //{81/4., 81/4., 81/4., 81/4., -189/4., -189/4., -189/4., -189/4., 441/4.}
        //};
        //double inverse_elMassMatrix_degree1 [4][4] = {{4, -2, 1, -2}, {-2, 4, -2, 1}, {1, -2, 4, -2}, {-2, 1, -2, 4}};
        double JacDet = 0;

        // Allocate space for the transport matrices
        // This is used for first order KUZMIN'S METHOD
        TransportMatrix.resize(NNZ,0.0);
        TransposeTransportMatrix.resize(NNZ,0.0);
        // Allocate and init to zero the Entropy residual vector
        global_entropy_residual.resize(numDOFs,0.0);
        if (STABILIZATION_TYPE==1) //EV stab
          for (int i=0; i<numDOFs; i++)
            global_entropy_residual[i] = 0.;

        //////////////////////////////////////////////
        // ** LOOP IN CELLS FOR CELL BASED TERMS ** //
        //////////////////////////////////////////////
        // HERE WE COMPUTE:
        //    * Time derivative term
        //    * Cell based CFL (for reference)
        //    * Entropy residual
        //    * Transport matrices
        for(int eN=0;eN<nElements_global;eN++)
          {
            //declare local storage for local contributions and initialize
            double
              elementResidual_u[nDOF_test_element],
              element_entropy_residual[nDOF_test_element];
            double  elementTransport[nDOF_test_element][nDOF_trial_element];
            double  elementTransposeTransport[nDOF_test_element][nDOF_trial_element];
            //double  preconditioned_elementTransport[nDOF_test_element][nDOF_trial_element];
            for (int i=0;i<nDOF_test_element;i++)
              {
                elementResidual_u[i]=0.0;
                element_entropy_residual[i]=0.0;
                for (int j=0;j<nDOF_trial_element;j++)
                  {
                    elementTransport[i][j]=0.0;
                    elementTransposeTransport[i][j]=0.0;
                    // preconditioned elementTransport
                    //preconditioned_elementTransport[i][j]=0.0;
                  }
              }
            //loop over quadrature points and compute integrands
            for  (int k=0;k<nQuadraturePoints_element;k++)
              {
                //compute indeces and declare local storage
                int eN_k = eN*nQuadraturePoints_element+k,
                  eN_k_nSpace = eN_k*nSpace,
                  eN_nDOF_trial_element = eN*nDOF_trial_element;
                double
                  // for entropy residual
                  aux_entropy_residual=0., DENTROPY_un, DENTROPY_uni,
                  //for mass matrix contributions
                  u=0.0, uStar=0.0, hquad=0.0,
                  u_test_dV[nDOF_trial_element],
                  //for entropy viscosity
                  un=0.0, unm1=0.0, grad_u[nSpace], grad_un[nSpace], vn[nSpace],
                  u_grad_trial[nDOF_trial_element*nSpace],
                  //for general use
                  jac[nSpace*nSpace], jacDet, jacInv[nSpace*nSpace],
                  dV,x,y,z,xt,yt,zt;
                //get the physical integration weight
                ck.calculateMapping_element(eN,
                                            k,
                                            mesh_dof.data(),
                                            mesh_l2g.data(),
                                            mesh_trial_ref.data(),
                                            mesh_grad_trial_ref.data(),
                                            jac,
                                            jacDet,
                                            jacInv,x,y,z);
                ck.calculateMappingVelocity_element(eN,
                                                    k,
                                                    mesh_velocity_dof.data(),
                                                    mesh_l2g.data(),
                                                    mesh_trial_ref.data(),
                                                    xt,yt,zt);
                dV = fabs(jacDet)*dV_ref.data()[k];
                JacDet = fabs(jacDet);
                // get h at quad points
                ck.valFromDOF(nodeDiametersArray.data(),&u_l2g.data()[eN_nDOF_trial_element],&u_trial_ref.data()[k*nDOF_trial_element],hquad);
                //get the solution (of Newton's solver). To compute time derivative term
                ck.valFromDOF(u_dof.data(),&u_l2g.data()[eN_nDOF_trial_element],&u_trial_ref.data()[k*nDOF_trial_element],u);
                ck.valFromDOF(uStar_dof.data(),&u_l2g.data()[eN_nDOF_trial_element],&u_trial_ref.data()[k*nDOF_trial_element],uStar);
                //get the solution at quad point at tn and tnm1 for entropy viscosity
                // solution and grads at old times at quad points
                ck.valFromDOF(u_dof_old.data(),&u_l2g.data()[eN_nDOF_trial_element],&u_trial_ref.data()[k*nDOF_trial_element],un);
                //get the solution gradients at tn for entropy viscosity
                ck.gradTrialFromRef(&u_grad_trial_ref.data()[k*nDOF_trial_element*nSpace],jacInv,u_grad_trial);
                ck.gradFromDOF(u_dof_old.data(),&u_l2g.data()[eN_nDOF_trial_element],u_grad_trial,grad_un);
                //precalculate test function products with integration weights for mass matrix terms
                for (int j=0;j<nDOF_trial_element;j++)
                  u_test_dV[j] = u_test_ref.data()[k*nDOF_trial_element+j]*dV;

                //
                //moving mesh
                //
                double mesh_velocity[3];
                mesh_velocity[0] = xt;
                mesh_velocity[1] = yt;
                mesh_velocity[2] = zt;
                //epsCoupez = 3*hquad;
                double norm_grad_un = 0.;
                for (int I=0;I<nSpace;I++)
                  norm_grad_un += std::pow(grad_un[I],2);
                norm_grad_un = std::sqrt(norm_grad_un) + 1E-10;
                double dist_error = fabs(norm_grad_un - (1-SATURATED_LEVEL_SET*std::pow(un/epsCoupez,2)));
                double sgn = sign(uStar,epsFactRedistancing);

                // get velocity
                for (int I=0;I<nSpace;I++)
                  vn[I] = (velocity.data()[eN_k_nSpace+I] - MOVING_DOMAIN*mesh_velocity[I]
                           + dist_error*COUPEZ*lambda_coupez*sgn*grad_un[I]/norm_grad_un);

                //////////////////////////////
                // CALCULATE CELL BASED CFL //
                //////////////////////////////
                calculateCFL(elementDiameter.data()[eN]/degree_polynomial,vn,cfl.data()[eN_k]);

                //////////////////////////////////////////////
                // CALCULATE ENTROPY RESIDUAL AT QUAD POINT //
                //////////////////////////////////////////////
                if (STABILIZATION_TYPE==1) // EV stab
                  {
                    //double entropy_residual = ((un - unm1)/dt + vn[0]*grad_un[0] + vn[1]*grad_un[1]
                    //                     -dist_error*COUPEZ*lambda_coupez*sgn
                    //                     *(1-SATURATED_LEVEL_SET*std::pow(un/epsCoupez,2)))*DENTROPY(un,-epsCoupez,epsCoupez);
                    for (int I=0;I<nSpace;I++)
                      aux_entropy_residual += vn[I]*grad_un[I];
                    aux_entropy_residual -= dist_error*COUPEZ*lambda_coupez*sgn*(1-SATURATED_LEVEL_SET*std::pow(un/epsCoupez,2));
                    DENTROPY_un = ENTROPY_TYPE==1 ? DENTROPY(un,-epsCoupez,epsCoupez) : DENTROPY_LOG(un,-epsCoupez,epsCoupez);
                  }
                //////////////
                // ith-LOOP //
                //////////////
                double residual = (u-un) - dt*dist_error*COUPEZ*lambda_coupez*sgn*(1-SATURATED_LEVEL_SET*std::pow(un/epsCoupez,2));

                for(int i=0;i<nDOF_test_element;i++)
                  {
                    int eN_i=eN*nDOF_test_element+i;
                    int gi = offset_u+stride_u*u_l2g.data()[eN_i]; //global i-th index
                    // entropy residual
                    if (STABILIZATION_TYPE==1) // EV Stab
                      {
                        DENTROPY_uni = ENTROPY_TYPE == 1 ? DENTROPY(u_dof_old.data()[gi],-epsCoupez,epsCoupez) : DENTROPY_LOG(u_dof_old.data()[gi],-epsCoupez,epsCoupez);
                        element_entropy_residual[i] += (DENTROPY_un - DENTROPY_uni)*aux_entropy_residual*u_test_dV[i];
                      }
                    // element residual
                    elementResidual_u[i] += residual*u_test_dV[i];
                    ///////////////
                    // j-th LOOP // To construct transport matrices
                    ///////////////
                    for(int j=0;j<nDOF_trial_element;j++)
                      {
                        int j_nSpace = j*nSpace;
                        int i_nSpace = i*nSpace;
                        // COMPUTE ELEMENT TRANSPORT MATRIX (MQL)
                        elementTransport[i][j] += // int[(vel.grad_wj)*wi*dx]
                          ck.HamiltonianJacobian_weak(vn,&u_grad_trial[j_nSpace],u_test_dV[i]);
                        elementTransposeTransport[i][j] += // int[(vel.grad_wi)*wj*dx]
                          ck.HamiltonianJacobian_weak(vn,&u_grad_trial[i_nSpace],u_test_dV[j]);
                      }
                  }//i
                //save solution at quadrature points for other models to use
                q_u.data()[eN_k]=u;
                q_m.data()[eN_k] = u;
                for (int I=0;I<nSpace;I++)
                  q_n.data()[eN_k_nSpace+I]  = grad_u[I];
              }
            // MULTIPLY Transport Matrix times preconditioner//
            //if (false)
            //  {
            //    for (int i=0; i < nDOF_test_element; i++)
            //for (int j=0; j < nDOF_test_element; j++)
            //  for (int r=0; r < nDOF_test_element; r++)
            //    if (degree_polynomial == 1)
            //      preconditioned_elementTransport[i][j] +=
            //    element_lumped_mass_matrix[i]*1./JacDet*inverse_elMassMatrix_degree1[i][r]*elementTransport[r][j];
            //    else
            //      preconditioned_elementTransport[i][j] +=
            //    element_lumped_mass_matrix[i]*1./JacDet*inverse_elMassMatrix_degree2[i][r]*elementTransport[r][j];
            //for (int i=0; i < nDOF_test_element; i++)
            //for (int j=0; j < nDOF_test_element; j++)
            //{
            //  elementTransport[i][j] = preconditioned_elementTransport[i][j];
            //  elementTransposeTransport[i][j] = preconditioned_elementTransport[j][i];
            //}
            //}

            /////////////////
            // DISTRIBUTE // load cell based element into global residual
            ////////////////
            for(int i=0;i<nDOF_test_element;i++)
              {
                int eN_i=eN*nDOF_test_element+i;
                int gi = offset_u+stride_u*r_l2g.data()[eN_i]; //global i-th index

                // distribute global residual for (lumped) mass matrix
                globalResidual.data()[gi] += elementResidual_u[i];
                // distribute entropy_residual
                if (STABILIZATION_TYPE==1) // EV Stab
                  global_entropy_residual[gi] += element_entropy_residual[i];

                // distribute transport matrices
                for (int j=0;j<nDOF_trial_element;j++)
                  {
                    int eN_i_j = eN_i*nDOF_trial_element+j;
                    TransportMatrix[csrRowIndeces_CellLoops.data()[eN_i] + csrColumnOffsets_CellLoops.data()[eN_i_j]]
                      += elementTransport[i][j];
                    TransposeTransportMatrix[csrRowIndeces_CellLoops.data()[eN_i] + csrColumnOffsets_CellLoops.data()[eN_i_j]]
                      += elementTransposeTransport[i][j];
                  }//j
              }//i
          }//elements

        ///////////////////////////////
        // COMPUTE g, ETA and others //
        ///////////////////////////////
        // NOTE: see VOF.h for a different but equivalent implementation of this.
        gx.resize(numDOFs,0.0);
        gy.resize(numDOFs,0.0);
        gz.resize(numDOFs,0.0);
        eta.resize(numDOFs,0.0);
        alpha_numerator_pos.resize(numDOFs,0.0);
        alpha_numerator_neg.resize(numDOFs,0.0);
        alpha_denominator_pos.resize(numDOFs,0.0);
        alpha_denominator_neg.resize(numDOFs,0.0);
        int ij = 0;
        for (int i=0; i<numDOFs; i++)
          {
            double solni = u_dof_old.data()[i];
            if (STABILIZATION_TYPE==1) //EV Stab
              eta[i] = ENTROPY_TYPE == 1 ? ENTROPY(solni,-epsCoupez,epsCoupez) : ENTROPY_LOG(solni,-epsCoupez,epsCoupez);
            else // smoothness based stab
              {
                gx[i]=0.;
                gy[i]=0.;
                gz[i]=0.;
                // for smoothness indicator //
                alpha_numerator_pos[i] = 0.;
                alpha_numerator_neg[i] = 0.;
                alpha_denominator_pos[i] = 0.;
                alpha_denominator_neg[i] = 0.;

                for (int offset=csrRowIndeces_DofLoops.data()[i]; offset<csrRowIndeces_DofLoops.data()[i+1]; offset++)
                  {
                    int j = csrColumnOffsets_DofLoops.data()[offset];
                    double solnj = u_dof_old.data()[j];

                    // for gi vector
                    gx[i] += Cx.data()[ij]*solnj;
                    gy[i] += Cy.data()[ij]*solnj;
#if nSpace==3
                    gz[i] += Cz.data()[ij]*solnj;
#endif
                    double alpha_num = solni - solnj;
                    if (alpha_num >= 0.)
                      {
                        alpha_numerator_pos[i] += alpha_num;
                        alpha_denominator_pos[i] += alpha_num;
                      }
                    else
                      {
                        alpha_numerator_neg[i] += alpha_num;
                        alpha_denominator_neg[i] += fabs(alpha_num);
                      }
                    //update ij
                    ij+=1;
                  }
                gx[i] /= ML.data()[i];
                gy[i] /= ML.data()[i];
#if nSpace==3
                gz[i] /= ML.data()[i];
#endif
              }
          }

        //////////////////////////////////////////////////////////
        // COMPUTE SMOOTHNESS INDICATOR and ENTROPY MIN and MAX //
        //////////////////////////////////////////////////////////
        // Smoothness indicator is based on the solution.
        // psi_i = psi_i(alpha_i); alpha_i = |sum(betaij*(uj-ui))|/sum(betaij*|uj-ui|)
        psi.resize(numDOFs,0.0);
        etaMax.resize(numDOFs,0.0);
        etaMin.resize(numDOFs,0.0);
        for (int i=0; i<numDOFs; i++)
          {
            double xi, yi, zi, SumPos=0., SumNeg=0.;
            if (STABILIZATION_TYPE==1) //EV Stabilization
              {
                // For eta min and max
                etaMax[i] = fabs(eta[i]);
                etaMin[i] = fabs(eta[i]);
              }
            else //smoothness based stab
              {
                xi = mesh_dof.data()[i*3+0];
                yi = mesh_dof.data()[i*3+1];
#if nSpace==3
                zi = mesh_dof.data()[i*3+2];
#endif
              }
            for (int offset=csrRowIndeces_DofLoops.data()[i]; offset<csrRowIndeces_DofLoops.data()[i+1]; offset++)
              { //loop in j (sparsity pattern)
                int j = csrColumnOffsets_DofLoops.data()[offset];
                if (STABILIZATION_TYPE==1) //EV stab
                  {
                    /////////////////////////////////
                    // COMPUTE ETA MIN AND ETA MAX //
                    /////////////////////////////////
                    etaMax[i] = fmax(etaMax[i],fabs(eta[j]));
                    etaMin[i] = fmin(etaMin[i],fabs(eta[j]));
                  }
                else //smoothness based sta
                  {
                    double xj = mesh_dof.data()[j*3+0];
                    double yj = mesh_dof.data()[j*3+1];
                    double zj = 0;
#if nSpace==3
                    zj = mesh_dof.data()[j*3+2];
#endif
                    //////////////////////////////
                    // FOR SMOOTHNESS INDICATOR //
                    //////////////////////////////
                    double gi_times_x = gx[i]*(xi-xj) + gy[i]*(yi-yj);
#if nSpace==3
                    gi_times_x += gz[i]*(zi-zj);
#endif
                    SumPos += gi_times_x > 0 ? gi_times_x : 0;
                    SumNeg += gi_times_x < 0 ? gi_times_x : 0;
                  }
              }
            if (STABILIZATION_TYPE==1) //EV stab
              {
                // Normalize entropy residual
                global_entropy_residual[i] *= etaMin[i] == etaMax[i] ? 0. : 2*cE/(etaMax[i]-etaMin[i]);
                quantDOFs.data()[i] = fabs(global_entropy_residual[i]);
              }
            else // smoothness based stab
              {
                // Compute sigmaPos and sigmaNeg
                double sigmaPosi = fmin(1.,(fabs(SumNeg)+1E-15)/(SumPos+1E-15));
                double sigmaNegi = fmin(1.,(SumPos+1E-15)/(fabs(SumNeg)+1E-15));
                double alpha_numi = fabs(sigmaPosi*alpha_numerator_pos[i] + sigmaNegi*alpha_numerator_neg[i]);
                double alpha_deni = sigmaPosi*alpha_denominator_pos[i] + sigmaNegi*alpha_denominator_neg[i];
                if (IS_BETAij_ONE == 1)
                  {
                    alpha_numi = fabs(alpha_numerator_pos[i] + alpha_numerator_neg[i]);
                    alpha_deni = alpha_denominator_pos[i] + alpha_denominator_neg[i];
                  }
                double alphai = alpha_numi/(alpha_deni+1E-15);
                quantDOFs.data()[i] = alphai;

                if (POWER_SMOOTHNESS_INDICATOR==0)
                  psi[i] = 1.0;
                else
                  psi[i] = std::pow(alphai,POWER_SMOOTHNESS_INDICATOR); //NOTE: they use alpha^2 in the paper
              }
          }
        // END OF COMPUTING SMOOTHNESS INDICATOR

        /////////////////////////////////////////////
        // ** LOOP IN DOFs FOR EDGE BASED TERMS ** //
        /////////////////////////////////////////////
        ij=0;
        for (int i=0; i<numDOFs; i++)
          {
            double solni = u_dof_old.data()[i]; // solution at time tn for the ith DOF
            double ith_dissipative_term = 0;
            double ith_flux_term = 0;
            double dLowii = 0.;

            // loop over the sparsity pattern of the i-th DOF
            for (int offset=csrRowIndeces_DofLoops.data()[i]; offset<csrRowIndeces_DofLoops.data()[i+1]; offset++)
              {
                int j = csrColumnOffsets_DofLoops.data()[offset];
                double solnj = u_dof_old.data()[j]; // solution at time tn for the jth DOF
                double dLowij, dHij, dEVij;

                ith_flux_term += TransportMatrix[ij]*solnj;
                if (i != j) //NOTE: there is really no need to check for i!=j (see formula for ith_dissipative_term)
                  {
                    // first-order dissipative operator
                    dLowij = fmax(fabs(TransportMatrix[ij]),fabs(TransposeTransportMatrix[ij]));
                    if (STABILIZATION_TYPE==1) //EV Stab
                      {
                        // high-order (entropy viscosity) dissipative operator
                        dEVij = fmax(fabs(global_entropy_residual[i]),fabs(global_entropy_residual[j]));
                        dHij = fmin(dLowij,dEVij);
                      }
                    else // smoothness based stab
                      dHij = dLowij*fmax(psi[i],psi[j]); // enhance the order to 2nd order. No EV

                    // compute dissipative term
                    ith_dissipative_term += dHij*(solnj-solni);
                    dLowii -= dLowij;
                  }
                //update ij
                ij+=1;
              }
            // update residual
            double mi = ML.data()[i];
            // compute edge_based_cfl
            edge_based_cfl.data()[i] = 2.*fabs(dLowii)/mi;

            if (LUMPED_MASS_MATRIX==1)
              {
                //globalResidual.data()[i] = mi*(u_dof.data()[i] - u_dof_old.data()[i]) + dt*(ith_flux_term - ith_dissipative_term);
                globalResidual.data()[i] = u_dof_old.data()[i] - dt/mi*(ith_flux_term - ith_dissipative_term);
              }
            else
              globalResidual.data()[i] += dt*(ith_flux_term - ith_dissipative_term);
          }

        /////////////////////////
        // BOUNDARY CONDITIONS //
        /////////////////////////
        for (int ebNE = 0; ebNE < nExteriorElementBoundaries_global; ebNE++)
          {
            int ebN = exteriorElementBoundariesArray.data()[ebNE],
              eN  = elementBoundaryElementsArray.data()[ebN*2+0],
              ebN_local = elementBoundaryLocalElementBoundariesArray.data()[ebN*2+0],
              eN_nDOF_trial_element = eN*nDOF_trial_element;
            double elementResidual_u[nDOF_test_element];
            for (int i=0;i<nDOF_test_element;i++)
              elementResidual_u[i]=0.0;
            for  (int kb=0;kb<nQuadraturePoints_elementBoundary;kb++)
              {
                int ebNE_kb = ebNE*nQuadraturePoints_elementBoundary+kb,
                  ebNE_kb_nSpace = ebNE_kb*nSpace,
                  ebN_local_kb = ebN_local*nQuadraturePoints_elementBoundary+kb,
                  ebN_local_kb_nSpace = ebN_local_kb*nSpace;
                double u_ext=0.0,
                  grad_u_ext[nSpace],
                  m_ext=0.0,
                  dm_ext=0.0,
                  H_ext=0.0,
                  dH_ext[nSpace],
                  bc_u_ext=0.0,
                  bc_grad_u_ext[nSpace],
                  bc_m_ext=0.0,
                  bc_dm_ext=0.0,
                  bc_H_ext=0.0,
                  bc_dH_ext[nSpace],
                  jac_ext[nSpace*nSpace],
                  jacDet_ext,
                  jacInv_ext[nSpace*nSpace],
                  boundaryJac[nSpace*(nSpace-1)],
                  metricTensor[(nSpace-1)*(nSpace-1)],
                  metricTensorDetSqrt,
                  dS,
                  u_test_dS[nDOF_test_element],
                  u_grad_trial_trace[nDOF_trial_element*nSpace],
                  normal[nSpace],x_ext,y_ext,z_ext,xt_ext,yt_ext,zt_ext,integralScaling,
                  G[nSpace*nSpace],G_dd_G,tr_G,flux_ext;
                //
                //calculate the solution and gradients at quadrature points
                //
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
                dS = ((1.0-MOVING_DOMAIN)*metricTensorDetSqrt + MOVING_DOMAIN*integralScaling)*dS_ref.data()[kb];
                //get the metric tensor
                //cek todo use symmetry
                ck.calculateG(jacInv_ext,G,G_dd_G,tr_G);
                //compute shape and solution information
                //shape
                ck.gradTrialFromRef(&u_grad_trial_trace_ref.data()[ebN_local_kb_nSpace*nDOF_trial_element],jacInv_ext,u_grad_trial_trace);
                //solution and gradients
                ck.valFromDOF(u_dof.data(),&u_l2g.data()[eN_nDOF_trial_element],&u_trial_trace_ref.data()[ebN_local_kb*nDOF_test_element],u_ext);
                ck.gradFromDOF(u_dof.data(),&u_l2g.data()[eN_nDOF_trial_element],u_grad_trial_trace,grad_u_ext);
                //precalculate test function products with integration weights
                for (int j=0;j<nDOF_trial_element;j++)
                  u_test_dS[j] = u_test_trace_ref.data()[ebN_local_kb*nDOF_test_element+j]*dS;
                //
                //load the boundary values
                //
                bc_u_ext = isDOFBoundary_u.data()[ebNE_kb]*ebqe_bc_u_ext.data()[ebNE_kb]+(1-isDOFBoundary_u.data()[ebNE_kb])*ebqe_rd_u_ext.data()[ebNE_kb];
                //
                //calculate the pde coefficients using the solution and the boundary values for the solution
                //
                evaluateCoefficients(&ebqe_velocity_ext.data()[ebNE_kb_nSpace],
                                     u_ext,
                                     grad_u_ext,
                                     m_ext,
                                     dm_ext,
                                     H_ext,
                                     dH_ext);
                evaluateCoefficients(&ebqe_velocity_ext.data()[ebNE_kb_nSpace],
                                     bc_u_ext,
                                     bc_grad_u_ext,
                                     bc_m_ext,
                                     bc_dm_ext,
                                     bc_H_ext,
                                     bc_dH_ext);
                //
                //moving mesh
                //
                double velocity_ext[nSpace];
                double mesh_velocity[3];
                mesh_velocity[0] = xt_ext;
                mesh_velocity[1] = yt_ext;
                mesh_velocity[2] = zt_ext;
                for (int I=0;I<nSpace;I++)
                  velocity_ext[I] = - MOVING_DOMAIN*mesh_velocity[I];
                //
                //calculate the numerical fluxes
                //
                exteriorNumericalFlux(normal,
                                      bc_u_ext,
                                      u_ext,
                                      dH_ext,
                                      velocity_ext,
                                      flux_ext);
                ebqe_u.data()[ebNE_kb] = u_ext;

                // gradient //

                double norm = 1.0e-8;
                for (int I=0;I<nSpace;I++)
                  norm += grad_u_ext[I]*grad_u_ext[I];
                norm = sqrt(norm);
                for (int I=0;I<nSpace;I++)
                  ebqe_grad_u.data()[ebNE_kb_nSpace+I]  = grad_u_ext[I]/norm;


                //
                //update residuals
                //
                for (int i=0;i<nDOF_test_element;i++)
                  {
                    //int ebNE_kb_i = ebNE_kb*nDOF_test_element+i;
                    elementResidual_u[i] += ck.ExteriorElementBoundaryFlux(flux_ext,u_test_dS[i]);
                  }//i
              }//kb
            //
            //update the element and global residual storage
            //
            for (int i=0;i<nDOF_test_element;i++)
              {
                int eN_i = eN*nDOF_test_element+i;
                double mi = ML.data()[offset_u+stride_u*u_l2g.data()[eN_i]];
                if (LUMPED_MASS_MATRIX==1)
                  globalResidual.data()[offset_u+stride_u*r_l2g.data()[eN_i]] -= dt/mi*elementResidual_u[i];
                else
                  globalResidual.data()[offset_u+stride_u*r_l2g.data()[eN_i]] += dt*elementResidual_u[i];
              }//i
          }//ebNE
        /////////////////////////
      }

      void calculateMassMatrix(arguments_dict& args)
      {
        double dt = args.scalar<double>("dt");
        proteus::pyarray<double>& mesh_trial_ref = args.array<double>("mesh_trial_ref");
        proteus::pyarray<double>& mesh_grad_trial_ref = args.array<double>("mesh_grad_trial_ref");
        proteus::pyarray<double>& mesh_dof = args.array<double>("mesh_dof");
        proteus::pyarray<double>& mesh_velocity_dof = args.array<double>("mesh_velocity_dof");
        double MOVING_DOMAIN = args.scalar<double>("MOVING_DOMAIN");
        proteus::pyarray<int>& mesh_l2g = args.array<int>("mesh_l2g");
        proteus::pyarray<double>& dV_ref = args.array<double>("dV_ref");
        proteus::pyarray<double>& u_trial_ref = args.array<double>("u_trial_ref");
        proteus::pyarray<double>& u_grad_trial_ref = args.array<double>("u_grad_trial_ref");
        proteus::pyarray<double>& u_test_ref = args.array<double>("u_test_ref");
        proteus::pyarray<double>& u_grad_test_ref = args.array<double>("u_grad_test_ref");
        proteus::pyarray<double>& mesh_trial_trace_ref = args.array<double>("mesh_trial_trace_ref");
        proteus::pyarray<double>& mesh_grad_trial_trace_ref = args.array<double>("mesh_grad_trial_trace_ref");
        proteus::pyarray<double>& dS_ref = args.array<double>("dS_ref");
        proteus::pyarray<double>& u_trial_trace_ref = args.array<double>("u_trial_trace_ref");
        proteus::pyarray<double>& u_grad_trial_trace_ref = args.array<double>("u_grad_trial_trace_ref");
        proteus::pyarray<double>& u_test_trace_ref = args.array<double>("u_test_trace_ref");
        proteus::pyarray<double>& u_grad_test_trace_ref = args.array<double>("u_grad_test_trace_ref");
        proteus::pyarray<double>& normal_ref = args.array<double>("normal_ref");
        proteus::pyarray<double>& boundaryJac_ref = args.array<double>("boundaryJac_ref");
        int nElements_global = args.scalar<int>("nElements_global");
        double useMetrics = args.scalar<double>("useMetrics");
        double alphaBDF = args.scalar<double>("alphaBDF");
        int lag_shockCapturing = args.scalar<int>("lag_shockCapturing");
        double shockCapturingDiffusion = args.scalar<double>("shockCapturingDiffusion");
        proteus::pyarray<int>& u_l2g = args.array<int>("u_l2g");
        proteus::pyarray<int>& r_l2g = args.array<int>("r_l2g");
        proteus::pyarray<double>& elementDiameter = args.array<double>("elementDiameter");
        int degree_polynomial = args.scalar<int>("degree_polynomial");
        proteus::pyarray<double>& u_dof = args.array<double>("u_dof");
        proteus::pyarray<double>& velocity = args.array<double>("velocity");
        proteus::pyarray<double>& q_m_betaBDF = args.array<double>("q_m_betaBDF");
        proteus::pyarray<double>& cfl = args.array<double>("cfl");
        proteus::pyarray<double>& q_numDiff_u_last = args.array<double>("q_numDiff_u_last");
        proteus::pyarray<int>& csrRowIndeces_u_u = args.array<int>("csrRowIndeces_u_u");
        proteus::pyarray<int>& csrColumnOffsets_u_u = args.array<int>("csrColumnOffsets_u_u");
        proteus::pyarray<double>& globalJacobian = args.array<double>("globalJacobian");
        int nExteriorElementBoundaries_global = args.scalar<int>("nExteriorElementBoundaries_global");
        proteus::pyarray<int>& exteriorElementBoundariesArray = args.array<int>("exteriorElementBoundariesArray");
        proteus::pyarray<int>& elementBoundaryElementsArray = args.array<int>("elementBoundaryElementsArray");
        proteus::pyarray<int>& elementBoundaryLocalElementBoundariesArray = args.array<int>("elementBoundaryLocalElementBoundariesArray");
        proteus::pyarray<double>& ebqe_velocity_ext = args.array<double>("ebqe_velocity_ext");
        proteus::pyarray<int>& isDOFBoundary_u = args.array<int>("isDOFBoundary_u");
        proteus::pyarray<double>& ebqe_rd_u_ext = args.array<double>("ebqe_rd_u_ext");
        proteus::pyarray<double>& ebqe_bc_u_ext = args.array<double>("ebqe_bc_u_ext");
        proteus::pyarray<int>& csrColumnOffsets_eb_u_u = args.array<int>("csrColumnOffsets_eb_u_u");
        int PURE_BDF = args.scalar<int>("PURE_BDF");
        int LUMPED_MASS_MATRIX = args.scalar<int>("LUMPED_MASS_MATRIX");
        double Ct_sge = 4.0;

        //
        //loop over elements to compute volume integrals and load them into the element Jacobians and global Jacobian
        //
        for(int eN=0;eN<nElements_global;eN++)
          {
            double  elementJacobian_u_u[nDOF_test_element][nDOF_trial_element];
            for (int i=0;i<nDOF_test_element;i++)
              for (int j=0;j<nDOF_trial_element;j++)
                {
                  elementJacobian_u_u[i][j]=0.0;
                }
            for  (int k=0;k<nQuadraturePoints_element;k++)
              {
                int eN_k = eN*nQuadraturePoints_element+k, //index to a scalar at a quadrature point
                  eN_k_nSpace = eN_k*nSpace,
                  eN_nDOF_trial_element = eN*nDOF_trial_element; //index to a vector at a quadrature point

                //declare local storage
                double u=0.0,
                  grad_u[nSpace],
                  m=0.0,dm=0.0,
                  H=0.0,dH[nSpace],
                  f[nSpace],df[nSpace],//MOVING_MESH
                  m_t=0.0,dm_t=0.0,
                  dpdeResidual_u_u[nDOF_trial_element],
                  Lstar_u[nDOF_test_element],
                  dsubgridError_u_u[nDOF_trial_element],
                  tau=0.0,tau0=0.0,tau1=0.0,
                  jac[nSpace*nSpace],
                  jacDet,
                  jacInv[nSpace*nSpace],
                  u_grad_trial[nDOF_trial_element*nSpace],
                  dV,
                  u_test_dV[nDOF_test_element],
                  u_grad_test_dV[nDOF_test_element*nSpace],
                  x,y,z,xt,yt,zt,
                  G[nSpace*nSpace],G_dd_G,tr_G;
                //
                //calculate solution and gradients at quadrature points
                //
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
                ck.calculateMappingVelocity_element(eN,
                                                    k,
                                                    mesh_velocity_dof.data(),
                                                    mesh_l2g.data(),
                                                    mesh_trial_ref.data(),
                                                    xt,yt,zt);
                //get the physical integration weight
                dV = fabs(jacDet)*dV_ref.data()[k];
                ck.calculateG(jacInv,G,G_dd_G,tr_G);
                //get the trial function gradients
                ck.gradTrialFromRef(&u_grad_trial_ref.data()[k*nDOF_trial_element*nSpace],jacInv,u_grad_trial);
                //get the solution
                ck.valFromDOF(u_dof.data(),&u_l2g.data()[eN_nDOF_trial_element],&u_trial_ref.data()[k*nDOF_trial_element],u);
                //get the solution gradients
                ck.gradFromDOF(u_dof.data(),&u_l2g.data()[eN_nDOF_trial_element],u_grad_trial,grad_u);
                //precalculate test function products with integration weights
                for (int j=0;j<nDOF_trial_element;j++)
                  {
                    u_test_dV[j] = u_test_ref.data()[k*nDOF_trial_element+j]*dV;
                    for (int I=0;I<nSpace;I++)
                      {
                        u_grad_test_dV[j*nSpace+I]   = u_grad_trial[j*nSpace+I]*dV;//cek warning won't work for Petrov-Galerkin
                      }
                  }
                //
                //calculate pde coefficients and derivatives at quadrature points
                //
                evaluateCoefficients(&velocity.data()[eN_k_nSpace],
                                     u,
                                     grad_u,
                                     m,
                                     dm,
                                     H,
                                     dH);
                //
                //moving mesh
                //
                double mesh_velocity[3];
                mesh_velocity[0] = xt;
                mesh_velocity[1] = yt;
                mesh_velocity[2] = zt;
                for (int I=0;I<nSpace;I++)
                  {
                    f[I] = -MOVING_DOMAIN*m*mesh_velocity[I];
                    df[I] = -MOVING_DOMAIN*dm*mesh_velocity[I];
                  }
                //
                //calculate time derivatives
                //
                ck.bdf(alphaBDF,
                       q_m_betaBDF.data()[eN_k],
                       m,
                       dm,
                       m_t,
                       dm_t);
                //
                //calculate subgrid error contribution to the Jacobian (strong residual, adjoint, jacobian of strong residual)
                //
                //calculate the adjoint times the test functions
                for (int i=0;i<nDOF_test_element;i++)
                  {
                    //int eN_k_i_nSpace = (eN_k*nDOF_trial_element+i)*nSpace;
                    //Lstar_u[i]=ck.Hamiltonian_adjoint(dH,&u_grad_test_dV[eN_k_i_nSpace]);
                    int i_nSpace = i*nSpace;
                    Lstar_u[i]=ck.Hamiltonian_adjoint(dH,&u_grad_test_dV[i_nSpace]) + MOVING_DOMAIN*ck.Advection_adjoint(df,&u_grad_test_dV[i_nSpace]);

                  }
                //calculate the Jacobian of strong residual
                for (int j=0;j<nDOF_trial_element;j++)
                  {
                    //int eN_k_j=eN_k*nDOF_trial_element+j;
                    //int eN_k_j_nSpace = eN_k_j*nSpace;
                    int j_nSpace = j*nSpace;
                    dpdeResidual_u_u[j]=ck.MassJacobian_strong(dm_t,u_trial_ref.data()[k*nDOF_trial_element+j]) +
                      ck.HamiltonianJacobian_strong(dH,&u_grad_trial[j_nSpace]) +
                      MOVING_DOMAIN*ck.AdvectionJacobian_strong(df,&u_grad_trial[j_nSpace]);

                  }
                //tau and tau*Res
                double subgridErrorVelocity[nSpace];
                for (int I=0;I<nSpace;I++)
                  subgridErrorVelocity[I] = dH[I] - MOVING_DOMAIN*df[I];

                calculateSubgridError_tau(elementDiameter.data()[eN],
                                          dm_t,
                                          subgridErrorVelocity,//dH,
                                          cfl.data()[eN_k],
                                          tau0);

                calculateSubgridError_tau(Ct_sge,
                                          G,
                                          dm_t,
                                          subgridErrorVelocity,//dH,
                                          tau1,
                                          cfl.data()[eN_k]);
                tau = useMetrics*tau1+(1.0-useMetrics)*tau0;

                for(int j=0;j<nDOF_trial_element;j++)
                  dsubgridError_u_u[j] = -tau*dpdeResidual_u_u[j];

                for(int i=0;i<nDOF_test_element;i++)
                  {
                    //int eN_k_i=eN_k*nDOF_test_element+i;
                    //int eN_k_i_nSpace=eN_k_i*nSpace;
                    for(int j=0;j<nDOF_trial_element;j++)
                      {
                        //int eN_k_j=eN_k*nDOF_trial_element+j;
                        //int eN_k_j_nSpace = eN_k_j*nSpace;
                        int j_nSpace = j*nSpace;
                        int i_nSpace = i*nSpace;
                        if (LUMPED_MASS_MATRIX==1)
                          {
                            if (i==j)
                              elementJacobian_u_u[i][j] += u_test_dV[i];
                          }
                        else
                          {
                            elementJacobian_u_u[i][j] +=
                              dt*ck.MassJacobian_weak(dm_t,u_trial_ref.data()[k*nDOF_trial_element+j],u_test_dV[i]);
                          }
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
                    globalJacobian.data()[csrRowIndeces_u_u.data()[eN_i] + csrColumnOffsets_u_u.data()[eN_i_j]] += elementJacobian_u_u[i][j];
                  }//j
              }//i
          }//elements
      }//computeMassMatrix

      void calculateSmoothingMatrix(arguments_dict& args)
      {
        double dt = args.scalar<double>("dt");
        proteus::pyarray<double>& mesh_trial_ref = args.array<double>("mesh_trial_ref");
        proteus::pyarray<double>& mesh_grad_trial_ref = args.array<double>("mesh_grad_trial_ref");
        proteus::pyarray<double>& mesh_dof = args.array<double>("mesh_dof");
        proteus::pyarray<double>& mesh_velocity_dof = args.array<double>("mesh_velocity_dof");
        double MOVING_DOMAIN = args.scalar<double>("MOVING_DOMAIN");
        proteus::pyarray<int>& mesh_l2g = args.array<int>("mesh_l2g");
        proteus::pyarray<double>& dV_ref = args.array<double>("dV_ref");
        proteus::pyarray<double>& u_trial_ref = args.array<double>("u_trial_ref");
        proteus::pyarray<double>& u_grad_trial_ref = args.array<double>("u_grad_trial_ref");
        proteus::pyarray<double>& u_test_ref = args.array<double>("u_test_ref");
        proteus::pyarray<double>& u_grad_test_ref = args.array<double>("u_grad_test_ref");
        proteus::pyarray<double>& mesh_trial_trace_ref = args.array<double>("mesh_trial_trace_ref");
        proteus::pyarray<double>& mesh_grad_trial_trace_ref = args.array<double>("mesh_grad_trial_trace_ref");
        proteus::pyarray<double>& dS_ref = args.array<double>("dS_ref");
        proteus::pyarray<double>& u_trial_trace_ref = args.array<double>("u_trial_trace_ref");
        proteus::pyarray<double>& u_grad_trial_trace_ref = args.array<double>("u_grad_trial_trace_ref");
        proteus::pyarray<double>& u_test_trace_ref = args.array<double>("u_test_trace_ref");
        proteus::pyarray<double>& u_grad_test_trace_ref = args.array<double>("u_grad_test_trace_ref");
        proteus::pyarray<double>& normal_ref = args.array<double>("normal_ref");
        proteus::pyarray<double>& boundaryJac_ref = args.array<double>("boundaryJac_ref");
        int nElements_global = args.scalar<int>("nElements_global");
        double useMetrics = args.scalar<double>("useMetrics");
        double alphaBDF = args.scalar<double>("alphaBDF");
        int lag_shockCapturing = args.scalar<int>("lag_shockCapturing");
        double shockCapturingDiffusion = args.scalar<double>("shockCapturingDiffusion");
        proteus::pyarray<int>& u_l2g = args.array<int>("u_l2g");
        proteus::pyarray<int>& r_l2g = args.array<int>("r_l2g");
        proteus::pyarray<double>& elementDiameter = args.array<double>("elementDiameter");
        int degree_polynomial = args.scalar<int>("degree_polynomial");
        proteus::pyarray<double>& u_dof = args.array<double>("u_dof");
        proteus::pyarray<double>& velocity = args.array<double>("velocity");
        proteus::pyarray<double>& q_m_betaBDF = args.array<double>("q_m_betaBDF");
        proteus::pyarray<double>& cfl = args.array<double>("cfl");
        proteus::pyarray<double>& q_numDiff_u_last = args.array<double>("q_numDiff_u_last");
        proteus::pyarray<int>& csrRowIndeces_u_u = args.array<int>("csrRowIndeces_u_u");
        proteus::pyarray<int>& csrColumnOffsets_u_u = args.array<int>("csrColumnOffsets_u_u");
        proteus::pyarray<double>& globalJacobian = args.array<double>("globalJacobian");
        int nExteriorElementBoundaries_global = args.scalar<int>("nExteriorElementBoundaries_global");
        proteus::pyarray<int>& exteriorElementBoundariesArray = args.array<int>("exteriorElementBoundariesArray");
        proteus::pyarray<int>& elementBoundaryElementsArray = args.array<int>("elementBoundaryElementsArray");
        proteus::pyarray<int>& elementBoundaryLocalElementBoundariesArray = args.array<int>("elementBoundaryLocalElementBoundariesArray");
        proteus::pyarray<double>& ebqe_velocity_ext = args.array<double>("ebqe_velocity_ext");
        proteus::pyarray<int>& isDOFBoundary_u = args.array<int>("isDOFBoundary_u");
        proteus::pyarray<double>& ebqe_rd_u_ext = args.array<double>("ebqe_rd_u_ext");
        proteus::pyarray<double>& ebqe_bc_u_ext = args.array<double>("ebqe_bc_u_ext");
        proteus::pyarray<int>& csrColumnOffsets_eb_u_u = args.array<int>("csrColumnOffsets_eb_u_u");
        double he = args.scalar<double>("he");
        double Ct_sge = 4.0;

        //
        //loop over elements to compute volume integrals and load them into the element Jacobians and global Jacobian
        //
        for(int eN=0;eN<nElements_global;eN++)
          {
            double  elementJacobian_u_u[nDOF_test_element][nDOF_trial_element];
            for (int i=0;i<nDOF_test_element;i++)
              for (int j=0;j<nDOF_trial_element;j++)
                {
                  elementJacobian_u_u[i][j]=0.0;
                }
            for  (int k=0;k<nQuadraturePoints_element;k++)
              {
                int eN_k = eN*nQuadraturePoints_element+k, //index to a scalar at a quadrature point
                  eN_k_nSpace = eN_k*nSpace,
                  eN_nDOF_trial_element = eN*nDOF_trial_element; //index to a vector at a quadrature point

                //declare local storage
                double u=0.0,
                  grad_u[nSpace],
                  m=0.0,dm=0.0,
                  H=0.0,dH[nSpace],
                  f[nSpace],df[nSpace],//MOVING_MESH
                  m_t=0.0,dm_t=0.0,
                  dpdeResidual_u_u[nDOF_trial_element],
                  Lstar_u[nDOF_test_element],
                  dsubgridError_u_u[nDOF_trial_element],
                  tau=0.0,tau0=0.0,tau1=0.0,
                  jac[nSpace*nSpace],
                  jacDet,
                  jacInv[nSpace*nSpace],
                  u_grad_trial[nDOF_trial_element*nSpace],
                  dV,
                  u_test_dV[nDOF_test_element],
                  u_grad_test_dV[nDOF_test_element*nSpace],
                  x,y,z,xt,yt,zt,
                  G[nSpace*nSpace],G_dd_G,tr_G;
                //
                //calculate solution and gradients at quadrature points
                //
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
                ck.calculateMappingVelocity_element(eN,
                                                    k,
                                                    mesh_velocity_dof.data(),
                                                    mesh_l2g.data(),
                                                    mesh_trial_ref.data(),
                                                    xt,yt,zt);
                //get the physical integration weight
                dV = fabs(jacDet)*dV_ref.data()[k];
                ck.calculateG(jacInv,G,G_dd_G,tr_G);
                //get the trial function gradients
                ck.gradTrialFromRef(&u_grad_trial_ref.data()[k*nDOF_trial_element*nSpace],jacInv,u_grad_trial);
                //get the solution
                ck.valFromDOF(u_dof.data(),&u_l2g.data()[eN_nDOF_trial_element],&u_trial_ref.data()[k*nDOF_trial_element],u);
                //get the solution gradients
                ck.gradFromDOF(u_dof.data(),&u_l2g.data()[eN_nDOF_trial_element],u_grad_trial,grad_u);
                //precalculate test function products with integration weights
                for (int j=0;j<nDOF_trial_element;j++)
                  {
                    u_test_dV[j] = u_test_ref.data()[k*nDOF_trial_element+j]*dV;
                    for (int I=0;I<nSpace;I++)
                      {
                        u_grad_test_dV[j*nSpace+I]   = u_grad_trial[j*nSpace+I]*dV;//cek warning won't work for Petrov-Galerkin
                      }
                  }
                //
                //calculate pde coefficients and derivatives at quadrature points
                //
                evaluateCoefficients(&velocity.data()[eN_k_nSpace],
                                     u,
                                     grad_u,
                                     m,
                                     dm,
                                     H,
                                     dH);
                //
                //moving mesh
                //
                double mesh_velocity[3];
                mesh_velocity[0] = xt;
                mesh_velocity[1] = yt;
                mesh_velocity[2] = zt;
                for (int I=0;I<nSpace;I++)
                  {
                    f[I] = -MOVING_DOMAIN*m*mesh_velocity[I];
                    df[I] = -MOVING_DOMAIN*dm*mesh_velocity[I];
                  }
                //
                //calculate time derivatives
                //
                ck.bdf(alphaBDF,
                       q_m_betaBDF.data()[eN_k],
                       m,
                       dm,
                       m_t,
                       dm_t);
                //
                //calculate subgrid error contribution to the Jacobian (strong residual, adjoint, jacobian of strong residual)
                //
                //calculate the adjoint times the test functions
                for (int i=0;i<nDOF_test_element;i++)
                  {
                    //int eN_k_i_nSpace = (eN_k*nDOF_trial_element+i)*nSpace;
                    //Lstar_u[i]=ck.Hamiltonian_adjoint(dH,&u_grad_test_dV[eN_k_i_nSpace]);
                    int i_nSpace = i*nSpace;
                    Lstar_u[i]=ck.Hamiltonian_adjoint(dH,&u_grad_test_dV[i_nSpace]) + MOVING_DOMAIN*ck.Advection_adjoint(df,&u_grad_test_dV[i_nSpace]);

                  }
                //calculate the Jacobian of strong residual
                for (int j=0;j<nDOF_trial_element;j++)
                  {
                    //int eN_k_j=eN_k*nDOF_trial_element+j;
                    //int eN_k_j_nSpace = eN_k_j*nSpace;
                    int j_nSpace = j*nSpace;
                    dpdeResidual_u_u[j]=ck.MassJacobian_strong(dm_t,u_trial_ref.data()[k*nDOF_trial_element+j]) +
                      ck.HamiltonianJacobian_strong(dH,&u_grad_trial[j_nSpace]) +
                      MOVING_DOMAIN*ck.AdvectionJacobian_strong(df,&u_grad_trial[j_nSpace]);

                  }
                //tau and tau*Res
                double subgridErrorVelocity[nSpace];
                for (int I=0;I<nSpace;I++)
                  subgridErrorVelocity[I] = dH[I] - MOVING_DOMAIN*df[I];

                calculateSubgridError_tau(elementDiameter.data()[eN],
                                          dm_t,
                                          subgridErrorVelocity,//dH,
                                          cfl.data()[eN_k],
                                          tau0);

                calculateSubgridError_tau(Ct_sge,
                                          G,
                                          dm_t,
                                          subgridErrorVelocity,//dH,
                                          tau1,
                                          cfl.data()[eN_k]);
                tau = useMetrics*tau1+(1.0-useMetrics)*tau0;

                for(int j=0;j<nDOF_trial_element;j++)
                  dsubgridError_u_u[j] = -tau*dpdeResidual_u_u[j];

                for(int i=0;i<nDOF_test_element;i++)
                  {
                    //int eN_k_i=eN_k*nDOF_test_element+i;
                    //int eN_k_i_nSpace=eN_k_i*nSpace;
                    for(int j=0;j<nDOF_trial_element;j++)
                      {
                        //int eN_k_j=eN_k*nDOF_trial_element+j;
                        //int eN_k_j_nSpace = eN_k_j*nSpace;
                        int j_nSpace = j*nSpace;
                        int i_nSpace = i*nSpace;
                        elementJacobian_u_u[i][j] +=
                          dt*ck.MassJacobian_weak(dm_t,u_trial_ref.data()[k*nDOF_trial_element+j],u_test_dV[i]) +
                          ck.NumericalDiffusionJacobian(std::pow(he,2),&u_grad_trial[j_nSpace],&u_grad_test_dV[i_nSpace]);
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
                    globalJacobian.data()[csrRowIndeces_u_u.data()[eN_i] + csrColumnOffsets_u_u.data()[eN_i_j]] += elementJacobian_u_u[i][j];
                  }//j
              }//i
          }//elements
      }//computeMassMatrix
    };//NCLS

  inline NCLS_base* newNCLS(int nSpaceIn,
                            int nQuadraturePoints_elementIn,
                            int nDOF_mesh_trial_elementIn,
                            int nDOF_trial_elementIn,
                            int nDOF_test_elementIn,
                            int nQuadraturePoints_elementBoundaryIn,
                            int CompKernelFlag)
  {
    if (nSpaceIn == 2)
      return proteus::chooseAndAllocateDiscretization2D<NCLS_base,NCLS,CompKernel>(nSpaceIn,
                                                                                   nQuadraturePoints_elementIn,
                                                                                   nDOF_mesh_trial_elementIn,
                                                                                   nDOF_trial_elementIn,
                                                                                   nDOF_test_elementIn,
                                                                                   nQuadraturePoints_elementBoundaryIn,
                                                                                   CompKernelFlag);
    else
      return proteus::chooseAndAllocateDiscretization<NCLS_base,NCLS,CompKernel>(nSpaceIn,
                                                                                 nQuadraturePoints_elementIn,
                                                                                 nDOF_mesh_trial_elementIn,
                                                                                 nDOF_trial_elementIn,
                                                                                 nDOF_test_elementIn,
                                                                                 nQuadraturePoints_elementBoundaryIn,
                                                                                 CompKernelFlag);
    /* return proteus::chooseAndAllocateDiscretization<NCLS_base,NCLS>(nSpaceIn, */
    /*                                                                         nQuadraturePoints_elementIn, */
    /*                                                                         nDOF_mesh_trial_elementIn, */
    /*                                                                         nDOF_trial_elementIn, */
    /*                                                                         nDOF_test_elementIn, */
    /*                                                                         nQuadraturePoints_elementBoundaryIn, */
    /*                                                                         CompKernelFlag); */
  }
}//proteus
#endif
