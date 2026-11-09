#ifndef RDLS_H
#define RDLS_H
#include <cmath>
#include <iostream>
#include <valarray>
#include "CompKernel.h"
#include "ModelFactory.h"
#include "equivalent_polynomials.h"
#include "ArgumentsDict.h"
#include "pyarray.h"

namespace py = pybind11;

#define SINGLE_POTENTIAL 1

namespace proteus
{
  inline double heaviside(const double &z)
  {
    return (z>0 ? 1. : (z<0 ? 0. : 0.5));
  }

  template<int nSpace, int nP, int nQ, int nEBQ>
  // The trailing flag is the IFEM gate. It has always been false here -- by
  // default rather than by statement -- and must stay false: with it set, an
  // element whose interface passes through an edge or corner node takes the
  // IFEM branch in Simplex::set_quad, which forces D to 0 and H/ImH to a hard
  // 0/1 instead of the moment fit. That deletes the interface measure this
  // model integrates over. Written out so the choice is visible at the call
  // site and cannot change underneath us if the template default changes.
  using GeneralizedFunctions = equivalent_polynomials::GeneralizedFunctions_mix<nSpace, nP, nP, nQ, nEBQ, false>;

  
  class RDLS_base
  {
  public:
    std::valarray<double> weighted_lumped_mass_matrix;
    virtual ~RDLS_base(){}
    virtual void calculateResidual(arguments_dict& args, bool useExact)=0;
    virtual void calculateJacobian(arguments_dict& args, bool useExact)=0;
    virtual void calculateResidual_ellipticRedist(arguments_dict& args, bool useExact)=0;
    virtual void calculateJacobian_ellipticRedist(arguments_dict& args, bool useExact)=0;
    virtual void normalReconstruction(arguments_dict& args)=0;
    virtual std::tuple<double, double, double> calculateMetricsAtEOS(arguments_dict& args)=0;
  };

  template<class CompKernelType,
    int nSpace,
    int nQuadraturePoints_element,
    int nDOF_mesh_trial_element,
    int nDOF_trial_element,
    int nDOF_test_element,
    int nQuadraturePoints_elementBoundary>
    class RDLS : public RDLS_base
    {
    public:
      const int nDOF_test_X_trial_element;
      CompKernelType ck;
      GeneralizedFunctions<nSpace,2,nQuadraturePoints_element,nQuadraturePoints_elementBoundary> gf,gfu;
    RDLS():
      nDOF_test_X_trial_element(nDOF_test_element*nDOF_trial_element),
        ck()
          {}

      inline
        void evaluateCoefficients(const double& eps,
                                  const double& u_levelSet,
                                  const double& u,
                                  const double grad_u[nSpace],
                                  double& m,
                                  double& dm,
                                  double& H,
                                  double dH[nSpace],
                                  double& r)
      {
        int I;
        double normGradU=0.0,Si=0.0;
        m = u;
        dm=1.0;
        H = 0.0;
        Si= gf.H(eps,u_levelSet) - gf.ImH(eps,u_levelSet);
        /* if (u_levelSet > 0.0) */
        /*   Si=1.0; */
        /* else if (u_levelSet < 0.0) */
        /*   Si = -1.0; */
        /* else */
        /*   Si=0.0; */
        r = -Si;
        for (I=0; I < nSpace; I++)
          {
            normGradU += grad_u[I]*grad_u[I];
          }
        normGradU = sqrt(normGradU);
        H = Si*normGradU;
        for (I=0; I < nSpace; I++)
          {
            dH[I] = Si*grad_u[I]/(normGradU+1.0e-12);
          }
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
        oneByAbsdt =  dmt;
        //'1'
        //tau = 1.0/(2.0*nrm_v/h + oneByAbsdt + 1.0e-8);
        //'2'
        tau = 1.0/sqrt(4.0*nrm_v*nrm_v/(h*h) + oneByAbsdt*oneByAbsdt + 1.0e-8);
      }

      inline
        void calculateSubgridError_tau(     const double   G[nSpace*nSpace],
                                            const double   Ai[nSpace],
                                            double& tau_v,
                                            double& q_cfl)
      {
        double v_d_Gv=0.0;
        for(int I=0;I<nSpace;I++)
          for (int J=0;J<nSpace;J++)
            v_d_Gv += Ai[I]*G[I*nSpace+J]*Ai[J];

        tau_v = 1.0/(sqrt(v_d_Gv) + 1.0e-8);
      }

#undef CKDEBUG
      void calculateResidual(arguments_dict& args,
                             bool useExact)
      {
        proteus::pyarray<double>& mesh_trial_ref = args.array<double>("mesh_trial_ref");
        proteus::pyarray<double>& mesh_grad_trial_ref = args.array<double>("mesh_grad_trial_ref");
        proteus::pyarray<double>& mesh_dof = args.array<double>("mesh_dof");
        proteus::pyarray<int>& mesh_l2g = args.array<int>("mesh_l2g");
        proteus::pyarray<double>& x_ref = args.array<double>("x_ref");
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
        double epsFact_redist = args.scalar<double>("epsFact_redist");
        double backgroundDiffusionFactor = args.scalar<double>("backgroundDiffusionFactor");
        double weakDirichletFactor = args.scalar<double>("weakDirichletFactor");
        int freezeLevelSet = args.scalar<int>("freezeLevelSet");
        int useTimeIntegration = args.scalar<int>("useTimeIntegration");
        int lag_shockCapturing = args.scalar<int>("lag_shockCapturing");
        int lag_subgridError = args.scalar<int>("lag_subgridError");
        double shockCapturingDiffusion = args.scalar<double>("shockCapturingDiffusion");
        proteus::pyarray<int>& u_l2g = args.array<int>("u_l2g");
        proteus::pyarray<double>& elementDiameter = args.array<double>("elementDiameter");
        proteus::pyarray<double>& nodeDiametersArray = args.array<double>("nodeDiametersArray");
        proteus::pyarray<double>& u_dof = args.array<double>("u_dof");
        proteus::pyarray<double>& phi_dof = args.array<double>("phi_dof");
        proteus::pyarray<double>& phi_ls = args.array<double>("phi_ls");
        proteus::pyarray<double>& q_m = args.array<double>("q_m");
        proteus::pyarray<double>& q_u = args.array<double>("q_u");
        proteus::pyarray<double>& q_n = args.array<double>("q_n");
        proteus::pyarray<double>& q_dH = args.array<double>("q_dH");
        proteus::pyarray<double>& u_weak_internal_bc_dofs = args.array<double>("u_weak_internal_bc_dofs");
        proteus::pyarray<double>& q_m_betaBDF = args.array<double>("q_m_betaBDF");
        proteus::pyarray<double>& q_dH_last = args.array<double>("q_dH_last");
        proteus::pyarray<double>& q_cfl = args.array<double>("q_cfl");
        proteus::pyarray<double>& q_numDiff_u = args.array<double>("q_numDiff_u");
        proteus::pyarray<double>& q_numDiff_u_last = args.array<double>("q_numDiff_u_last");
        proteus::pyarray<int>& weakDirichletConditionFlags = args.array<int>("weakDirichletConditionFlags");
        int offset_u = args.scalar<int>("offset_u");
        int stride_u = args.scalar<int>("stride_u");
        proteus::pyarray<double>& globalResidual = args.array<double>("globalResidual");
        int nExteriorElementBoundaries_global = args.scalar<int>("nExteriorElementBoundaries_global");
        proteus::pyarray<int>& exteriorElementBoundariesArray = args.array<int>("exteriorElementBoundariesArray");
        proteus::pyarray<int>& elementBoundaryElementsArray = args.array<int>("elementBoundaryElementsArray");
        proteus::pyarray<int>& elementBoundaryLocalElementBoundariesArray = args.array<int>("elementBoundaryLocalElementBoundariesArray");
        proteus::pyarray<double>& ebqe_phi_ls_ext = args.array<double>("ebqe_phi_ls_ext");
        proteus::pyarray<int>& isDOFBoundary_u = args.array<int>("isDOFBoundary_u");
        proteus::pyarray<double>& ebqe_bc_u_ext = args.array<double>("ebqe_bc_u_ext");
        proteus::pyarray<double>& ebqe_u = args.array<double>("ebqe_u");
        proteus::pyarray<double>& ebqe_n = args.array<double>("ebqe_n");
        int ELLIPTIC_REDISTANCING = args.scalar<int>("ELLIPTIC_REDISTANCING");
        double backgroundDissipationEllipticRedist = args.scalar<double>("backgroundDissipationEllipticRedist");
        proteus::pyarray<double>& lumped_qx = args.array<double>("lumped_qx");
        proteus::pyarray<double>& lumped_qy = args.array<double>("lumped_qy");
        proteus::pyarray<double>& lumped_qz = args.array<double>("lumped_qz");
        double alpha = args.scalar<double>("alpha");
        double circbc=0.0,circ=0.0;
        gf.useExact=useExact;
        gfu.useExact=useExact;
#ifdef CKDEBUG
        std::cout<<"stuff"<<"\t"
                 <<alphaBDF<<"\t"
                 <<epsFact_redist<<"\t"
                 <<freezeLevelSet<<"\t"
                 <<useTimeIntegration<<"\t"
                 <<lag_shockCapturing<< "\t"
                 <<lag_subgridError<<std::endl;
#endif
        //
        //loop over elements to compute volume integrals and load them into element and global residual
        //
        //eN is the element index
        //eN_k is the quadrature point index for a scalar
        //eN_k_nSpace is the quadrature point index for a vector
        //eN_i is the element test function index
        //eN_j is the element trial function index
        //eN_k_j is the quadrature point index for a trial function
        //eN_k_i is the quadrature point index for a trial function
        double timeIntegrationScale = 1.0;
        if (useTimeIntegration == 0)
          timeIntegrationScale = 0.0;
        double lag_shockCapturingScale = 1.0;
        if (lag_shockCapturing == 0)
          lag_shockCapturingScale = 0.0;
        for(int eN=0;eN<nElements_global;eN++)
          {
            //declare local storage for element residual and initialize
            int dummy_l2g[nDOF_mesh_trial_element];
            double elementResidual_u[nDOF_test_element],element_phi[nDOF_trial_element];
            double epsilon_redist,h_phi, dir[nSpace], norm;
            for (int i=0;i<nDOF_test_element;i++)
              {
		int eN_i=eN*nDOF_trial_element+i;
                elementResidual_u[i]=0.0;
		element_phi[i] = phi_dof.data()[u_l2g.data()[eN_i]];
                dummy_l2g[i] = i;
              }//i
            double element_nodes[nDOF_mesh_trial_element*3];
	    for (int i=0;i<nDOF_mesh_trial_element;i++)
	      {
		int eN_i=eN*nDOF_mesh_trial_element+i;
                for(int I=0;I<3;I++)
                  element_nodes[i*3 + I] = mesh_dof.data()[mesh_l2g.data()[eN_i]*3 + I];
	      }//i
            gf.calculate(element_phi, element_nodes, x_ref.data(),false);
            /* for (int i=0;i<nDOF_test_element;i++) */
            /*   { */
	    /*     int eN_i=eN*nDOF_trial_element+i; */
            /*     double eps=1.0e-4; */
            /*     if((fabs(gf.exact.phi_dof_corrected[i]) < eps) || */
            /*        (fabs(phi_dof.data()[u_l2g.data()[eN_i]]) < eps)) */
            /*       std::cout<<"Warning "<<gf.exact.phi_dof_corrected[i]<<'\t' */
            /*                <<phi_dof.data()[u_l2g.data()[eN_i]]<<'\t' */
            /*                <<u_l2g.data()[eN_i]<<std::endl; */
            /*   } */
            //loop over quadrature points and compute integrands
            for  (int k=0;k<nQuadraturePoints_element;k++)
              {
                gf.set_quad(k);
                //compute indeces and declare local storage
                int eN_k = eN*nQuadraturePoints_element+k,
                  eN_k_nSpace = eN_k*nSpace,
                  eN_nDOF_trial_element = eN*nDOF_trial_element;
                double u=0.0,grad_u[nSpace],u0=0.0, grad_phi[nSpace],
                  m=0.0,dm=0.0,
                  H=0.0,dH[nSpace],
                  m_t=0.0,dm_t=0.0,
                  r=0.0,
                  dH_tau[nSpace],//dH if not lagging or q_dH_last if lagging tau
                  dH_strong[nSpace],//dH if not lagging or q_dH_last if lagging strong residual and adjoint
                  pdeResidual_u=0.0,
                  Lstar_u[nDOF_test_element],
                  subgridError_u=0.0,
                  tau=0.0,tau0=0.0,tau1=0.0,
                  numDiff0=0.0,numDiff1=0.0,
                  nu_sc=0.0,
                  jac[nSpace*nSpace],
                  jacDet,
                  jacInv[nSpace*nSpace],
                  u_grad_trial[nDOF_trial_element*nSpace],
                  u_test_dV[nDOF_trial_element],
                  u_grad_test_dV[nDOF_test_element*nSpace],
                  dV,x,y,z,
                  G[nSpace*nSpace],G_dd_G,tr_G;
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
                //get the physical integration weight
                dV = fabs(jacDet)*dV_ref.data()[k];
                ck.calculateG(jacInv,G,G_dd_G,tr_G);

                //get the trial function gradients
                ck.gradTrialFromRef(&u_grad_trial_ref.data()[k*nDOF_trial_element*nSpace],jacInv,u_grad_trial);
                //get the solution
                ck.valFromDOF(u_dof.data(),&u_l2g.data()[eN_nDOF_trial_element],&u_trial_ref.data()[k*nDOF_trial_element],u);
                ck.valFromDOF(gf.exact.phi_dof_corrected,dummy_l2g,&u_trial_ref.data()[k*nDOF_trial_element],u0);
                if (freezeLevelSet)
                  {
                    u0 = phi_ls.data()[eN_k];
                  }
                /* double DX=(x-0.5),DY=(y-0.75); */
                /* double radius = std::sqrt(DX*DX+DY*DY); */
                /* double theta = std::atan2(DY,DX); */
                /* double kp=10.0, scaling=1.0, rp=1; */
                /* u0 = scaling*std::pow((0.15+(0.015/2.)*std::cos(kp*theta) - radius),rp);     */

                //u0 = 0.15 - std::sqrt(DX*DX + DY*DY);
                //u0 = phi_ls.data()[eN_k];//cek debug--set to exact input
                /* //cek debug */
                /* double u0_test; */
                /* ck.valFromDOF(phi_dof.data(),&u_l2g.data()[eN_nDOF_trial_element],&u_trial_ref.data()[k*nDOF_trial_element],u0_test); */
                /* if (u0 != u0_test) */
                /*   std::cout<<"eN "<<eN<<" k "<<k<<" u0 "<<u0<<" u0_test "<<u0_test<<std::endl; */
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
                //calculate pde coefficients at quadrature points
                //
                // the unit normal, recorded in q_n below (as the exterior loop records ebqe_n)
                norm = 1.0e-8;
                for (int I=0;I<nSpace;I++)
                  norm += grad_u[I]*grad_u[I];
                norm = sqrt(norm);
                for (int I=0;I<nSpace;I++)
                  dir[I] = grad_u[I]/norm;

                /* ck.calculateGScale(G,dir,h_phi); */

                epsilon_redist = epsFact_redist*(useMetrics*h_phi+(1.0-useMetrics)*elementDiameter.data()[eN]);

                evaluateCoefficients(epsilon_redist,
                                     u0,
                                     u,
                                     grad_u,
                                     m,
                                     dm,
                                     H,
                                     dH,
                                     r);
                //TODO allow not lagging of subgrid error etc,
                //remove conditional?
                //default no lagging
                for (int I=0; I < nSpace; I++)
                  {
                    dH_tau[I] = dH[I];
                    dH_strong[I] = dH[I];
                  }
                if (lag_subgridError > 0)
                  {
                    for (int I=0; I < nSpace; I++)
                      {
                        dH_tau[I] = q_dH_last.data()[eN_k_nSpace+I];
                      }
                  }
                if (lag_subgridError > 1)
                  {
                    for (int I=0; I < nSpace; I++)
                      {
                        dH_strong[I] = q_dH_last.data()[eN_k_nSpace+I];
                      }
                  }
                //save mass for time history and dH for subgrid error
                //save solution for other models
                //
                q_m.data()[eN_k] = m;
                q_u.data()[eN_k] = u;
                for (int I=0;I<nSpace;I++)
                  q_n.data()[eN_k_nSpace+I] = dir[I];

                for (int I=0;I<nSpace;I++)
                  {
                    int eN_k_nSpace_I = eN_k_nSpace+I;
                    q_dH.data()[eN_k_nSpace_I] = dH[I];
                  }

                //
                //moving mesh
                //
                //omit for now
                //
                //calculate time derivative at quadrature points
                //
                ck.bdf(alphaBDF,
                       q_m_betaBDF.data()[eN_k],
                       m,
                       dm,
                       m_t,
                       dm_t);

                //TODO add option to skip if not doing time integration (Newton stead-state solve)
                m *= timeIntegrationScale; dm *= timeIntegrationScale; m_t *= timeIntegrationScale;
                dm_t *= timeIntegrationScale;
#ifdef CKDEBUG
                std::cout<<"alpha "<<alphaBDF<<"\t"<<q_m_betaBDF.data()[eN_k]<<"\t"<<m_t<<'\t'<<m<<'\t'<<alphaBDF*m<<std::endl;
#endif
                //
                //calculate subgrid error (strong residual and adjoint)
                //
                //calculate strong residual
                pdeResidual_u = ck.Mass_strong(m_t) +
                  ck.Hamiltonian_strong(dH_strong,grad_u) + //would need dH if not lagging
                  ck.Reaction_strong(r);
#ifdef CKDEBUG
                std::cout<<"dH_strong "<<dH_strong[0]<<'\t'<<dH_strong[1]<<'\t'<<dH_strong[2]<<std::endl;
#endif
                //calculate adjoint
                for (int i=0;i<nDOF_test_element;i++)
                  {
                    //int eN_k_i_nSpace = (eN_k*nDOF_trial_element+i)*nSpace;
                    int  i_nSpace=i*nSpace;
                    Lstar_u[i]  = ck.Hamiltonian_adjoint(dH_strong,&u_grad_test_dV[i_nSpace]);
                    //reaction is constant
                  }
                //calculate tau and tau*Res
                calculateSubgridError_tau(elementDiameter.data()[eN],
                                          dm_t,dH_tau,
                                          q_cfl.data()[eN_k],
                                          tau0);
                calculateSubgridError_tau(G,
                                          dH_tau,
                                          tau1,
                                          q_cfl.data()[eN_k]);

                tau = useMetrics*tau1+(1.0-useMetrics)*tau0;

                //std::cout<<tau<<std::endl;


                subgridError_u = -tau*pdeResidual_u;
                //
                //calculate shock capturing diffusion
                //
                ck.calculateNumericalDiffusion(shockCapturingDiffusion,elementDiameter.data()[eN],pdeResidual_u,grad_u,numDiff0);
                ck.calculateNumericalDiffusion(shockCapturingDiffusion,G,pdeResidual_u,grad_u,numDiff1);

                q_numDiff_u.data()[eN_k] = useMetrics*numDiff1+(1.0-useMetrics)*numDiff0;

#ifdef CKDEBUG
                std::cout<<"q_numDiff_u[eN_k] "<<q_numDiff_u.data()[eN_k]<<" q_numDiff_u_last[eN_k] "<<q_numDiff_u_last.data()[eN_k]<<" lag "<<lag_shockCapturingScale<<std::endl;
#endif
                nu_sc = q_numDiff_u.data()[eN_k]*(1.0-lag_shockCapturingScale) + q_numDiff_u_last.data()[eN_k]*lag_shockCapturingScale;
                /* double epsilon_background_diffusion = 3.0*h_phi;//2.0*epsFact_redist*(useMetrics*h_phi+(1.0-useMetrics)*elementDiameter.data()[eN]); */
                /* if (fabs(u0) >  epsilon_background_diffusion) */
                /*   nu_sc += backgroundDiffusionFactor*h_phi; */
                
                double epsilon_background_diffusion = 2.0*epsFact_redist*(useMetrics*h_phi+(1.0-useMetrics)*elementDiameter.data()[eN]);
                if (fabs(phi_ls.data()[eN_k]) >  epsilon_background_diffusion)
                  nu_sc += backgroundDiffusionFactor*elementDiameter.data()[eN];                //
                //update element residual
                //
                for(int i=0;i<nDOF_test_element;i++)
                  {
                    int i_nSpace = i*nSpace;
                    double FREEZE=double(freezeLevelSet);
                    //assert(FREEZE==0.0);
                    //int eN_k_i=eN_k*nDOF_test_element+i;
                    //int eN_k_i_nSpace = eN_k_i*nSpace;

#ifdef CKDEBUG
                    std::cout<<"shock capturing input  nu_sc "<<nu_sc<<'\t'<<grad_u[0]<<'\t'<<grad_u[1]<<'\t'<<grad_u[1]<<'\t'<<u_grad_test_dV[i_nSpace]<<std::endl;
#endif
                    //std::cout<<element_phi[i]<<'\t'<<element_nodes[i*3+0]<<'\t'<<element_nodes[i*3+1]<<std::endl;
                    //std::cout<<"eN_k "<<eN_k<<" D "<<gf.D(epsilon_redist,phi_ls.data()[eN_k])<<std::endl;
                    circbc += ck.Reaction_weak(gf.D(epsilon_redist,u0)*(u-u0), u_test_dV[i]);
                    elementResidual_u[i] += ck.Mass_weak(m_t,u_test_dV[i]) +
		      ck.Hamiltonian_weak(H,u_test_dV[i]) +
                      ck.Reaction_weak(r,u_test_dV[i]) +
                      (1.0-FREEZE)*(weakDirichletFactor/h_phi)*ck.Reaction_weak(gf.D(epsilon_redist,u0)*(u0-u),
                                                                                              u_test_dV[i]) +
                      ck.SubgridError(subgridError_u,Lstar_u[i]) +
		      ck.NumericalDiffusion(nu_sc,grad_u,&u_grad_test_dV[i_nSpace]);
#ifdef CKDEBUG
                    std::cout<<ck.Mass_weak(m_t,u_test_dV[i])<<'\t'
                             <<ck.Hamiltonian_weak(H,u_test_dV[i]) <<'\t'
                             <<ck.Reaction_weak(r,u_test_dV[i])<<'\t'
                             <<ck.SubgridError(subgridError_u,Lstar_u[i])<<'\t'
                             <<ck.NumericalDiffusion(nu_sc,grad_u,&u_grad_test_dV[i_nSpace])<<std::endl;
#endif
                  }//i
                //
              }//k
            //
            //apply weak constraints for unknowns near zero level set
            //
            //
            if (freezeLevelSet)
              {
                for (int j = 0; j < nDOF_trial_element; j++)
                  {
                    const int eN_j = eN*nDOF_trial_element+j;
                    const int J = u_l2g.data()[eN_j];
                    //if (weakDirichletConditionFlags.data()[J] == 1)
                    if (fabs(u_weak_internal_bc_dofs.data()[J]) < epsilon_redist)
                      {
                        elementResidual_u[j] = (u_dof.data()[J]-u_weak_internal_bc_dofs.data()[J])*weakDirichletFactor*elementDiameter.data()[eN];
                      }
                  }//j
              }//freeze

            //
            //load element into global residual and save element residual
            //
            for(int i=0;i<nDOF_test_element;i++)
              {
                int eN_i=eN*nDOF_test_element+i;

#ifdef CKDEBUG
                std::cout<<"element residual i = "<<i<<"\t"<<elementResidual_u[i]<<std::endl;
#endif
                globalResidual.data()[offset_u+stride_u*u_l2g.data()[eN_i]]+=elementResidual_u[i];
              }//i
          }//elements
        //
        //loop over exterior element boundaries
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
            double epsilon_redist, h_phi;
            for  (int kb=0;kb<nQuadraturePoints_elementBoundary;kb++)
              {
                int ebNE_kb = ebNE*nQuadraturePoints_elementBoundary+kb,
                  ebNE_kb_nSpace = ebNE_kb*nSpace,
                  ebN_local_kb = ebN_local*nQuadraturePoints_elementBoundary+kb,
                  ebN_local_kb_nSpace = ebN_local_kb*nSpace;
                double u_ext=0.0,
                  grad_u_ext[nSpace],
                  jac_ext[nSpace*nSpace],
                  jacDet_ext,
                  jacInv_ext[nSpace*nSpace],
                  boundaryJac[nSpace*(nSpace-1)],
                  metricTensor[(nSpace-1)*(nSpace-1)],
                  metricTensorDetSqrt,
                  u_test_dS[nDOF_test_element],
                  u_grad_trial_trace[nDOF_trial_element*nSpace],
                  normal[nSpace],x_ext,y_ext,z_ext,
                  dir[nSpace],norm;
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
                //compute shape and solution information
                //shape
		ck.gradTrialFromRef(&u_grad_trial_trace_ref.data()[ebN_local_kb_nSpace*nDOF_trial_element],jacInv_ext,u_grad_trial_trace);
                //solution and gradients
                ck.valFromDOF(u_dof.data(),&u_l2g.data()[eN_nDOF_trial_element],&u_trial_trace_ref.data()[ebN_local_kb*nDOF_test_element],u_ext);
                ck.gradFromDOF(u_dof.data(),&u_l2g.data()[eN_nDOF_trial_element],u_grad_trial_trace,grad_u_ext);
                norm = 1.0e-8;
                for (int I=0;I<nSpace;I++)
                  norm += grad_u_ext[I]*grad_u_ext[I];
                norm = sqrt(norm);
                for (int I=0;I<nSpace;I++)
                  dir[I] = grad_u_ext[I]/norm;
                //save for other models
                ebqe_u.data()[ebNE_kb] = u_ext;
                for (int I=0;I<nSpace;I++)
                  ebqe_n.data()[ebNE_kb_nSpace+I] = dir[I];
              }//kb
          }//ebNE
        //std::cout<<"Circ Res"<<circbc<<'\t'<<circ<<std::endl;
      }

      //for now assumes that using time integration
      //and so lags stabilization and subgrid error
      //extern "C" void calculateJacobian_RDLSV2(int nElements_global,
      void calculateJacobian(arguments_dict& args,
                             bool useExact)
      {
        proteus::pyarray<double>& mesh_trial_ref = args.array<double>("mesh_trial_ref");
        proteus::pyarray<double>& mesh_grad_trial_ref = args.array<double>("mesh_grad_trial_ref");
        proteus::pyarray<double>& mesh_dof = args.array<double>("mesh_dof");
        proteus::pyarray<int>& mesh_l2g = args.array<int>("mesh_l2g");
        proteus::pyarray<double>& x_ref = args.array<double>("x_ref");
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
        double epsFact_redist = args.scalar<double>("epsFact_redist");
        double backgroundDiffusionFactor = args.scalar<double>("backgroundDiffusionFactor");
        double weakDirichletFactor = args.scalar<double>("weakDirichletFactor");
        int freezeLevelSet = args.scalar<int>("freezeLevelSet");
        int useTimeIntegration = args.scalar<int>("useTimeIntegration");
        int lag_shockCapturing = args.scalar<int>("lag_shockCapturing");
        int lag_subgridError = args.scalar<int>("lag_subgridError");
        double shockCapturingDiffusion = args.scalar<double>("shockCapturingDiffusion");
        proteus::pyarray<int>& u_l2g = args.array<int>("u_l2g");
        proteus::pyarray<double>& elementDiameter = args.array<double>("elementDiameter");
        proteus::pyarray<double>& nodeDiametersArray = args.array<double>("nodeDiametersArray");
        proteus::pyarray<double>& u_dof = args.array<double>("u_dof");
        proteus::pyarray<double>& phi_dof = args.array<double>("phi_dof");
        proteus::pyarray<double>& u_weak_internal_bc_dofs = args.array<double>("u_weak_internal_bc_dofs");
        proteus::pyarray<double>& phi_ls = args.array<double>("phi_ls");
        proteus::pyarray<double>& q_m_betaBDF = args.array<double>("q_m_betaBDF");
        proteus::pyarray<double>& q_dH_last = args.array<double>("q_dH_last");
        proteus::pyarray<double>& q_cfl = args.array<double>("q_cfl");
        proteus::pyarray<double>& q_numDiff_u = args.array<double>("q_numDiff_u");
        proteus::pyarray<double>& q_numDiff_u_last = args.array<double>("q_numDiff_u_last");
        proteus::pyarray<int>& weakDirichletConditionFlags = args.array<int>("weakDirichletConditionFlags");
        proteus::pyarray<int>& csrRowIndeces_u_u = args.array<int>("csrRowIndeces_u_u");
        proteus::pyarray<int>& csrColumnOffsets_u_u = args.array<int>("csrColumnOffsets_u_u");
        proteus::pyarray<double>& globalJacobian = args.array<double>("globalJacobian");
        int nExteriorElementBoundaries_global = args.scalar<int>("nExteriorElementBoundaries_global");
        proteus::pyarray<int>& exteriorElementBoundariesArray = args.array<int>("exteriorElementBoundariesArray");
        proteus::pyarray<int>& elementBoundaryElementsArray = args.array<int>("elementBoundaryElementsArray");
        proteus::pyarray<int>& elementBoundaryLocalElementBoundariesArray = args.array<int>("elementBoundaryLocalElementBoundariesArray");
        proteus::pyarray<double>& ebqe_phi_ls_ext = args.array<double>("ebqe_phi_ls_ext");
        proteus::pyarray<int>& isDOFBoundary_u = args.array<int>("isDOFBoundary_u");
        proteus::pyarray<double>& ebqe_bc_u_ext = args.array<double>("ebqe_bc_u_ext");
        proteus::pyarray<int>& csrColumnOffsets_eb_u_u = args.array<int>("csrColumnOffsets_eb_u_u");
        int ELLIPTIC_REDISTANCING = args.scalar<int>("ELLIPTIC_REDISTANCING");
        double backgroundDissipationEllipticRedist = args.scalar<double>("backgroundDissipationEllipticRedist");
        double alpha = args.scalar<double>("alpha");
        double circ=0.0;
        gf.useExact=useExact;
        //
        //loop over elements to compute volume integrals and load them into the element Jacobians and global Jacobian
        //
        double timeIntegrationScale = 1.0;
        if (useTimeIntegration == 0)
          timeIntegrationScale = 0.0;
        double lag_shockCapturingScale = 1.0;
        if (lag_shockCapturing == 0)
          lag_shockCapturingScale = 0.0;
        for(int eN=0;eN<nElements_global;eN++)
          {
            int dummy_l2g[nDOF_mesh_trial_element];
            double  elementJacobian_u_u[nDOF_test_element][nDOF_trial_element],element_phi[nDOF_trial_element];
            double epsilon_redist,h_phi, dir[nSpace], norm;
            for (int i=0;i<nDOF_test_element;i++)
              {
		int eN_i=eN*nDOF_trial_element+i;
                element_phi[i] = phi_dof.data()[u_l2g.data()[eN_i]];
                dummy_l2g[i] = i;
                for (int j=0;j<nDOF_trial_element;j++)
                  {
                    elementJacobian_u_u[i][j]=0.0;
                  }
              }
            double element_nodes[nDOF_mesh_trial_element*3];
	    for (int i=0;i<nDOF_mesh_trial_element;i++)
	      {
		int eN_i=eN*nDOF_mesh_trial_element+i;
                for(int I=0;I<3;I++)
                  element_nodes[i*3 + I] = mesh_dof.data()[mesh_l2g.data()[eN_i]*3 + I];
	      }//i
            gf.calculate(element_phi, element_nodes, x_ref.data(),false);            
            for  (int k=0;k<nQuadraturePoints_element;k++)
              {
                gf.set_quad(k);
                int eN_k = eN*nQuadraturePoints_element+k, //index to a scalar at a quadrature point
                  eN_k_nSpace = eN_k*nSpace,
                  eN_nDOF_trial_element = eN*nDOF_trial_element; //index to a vector at a quadrature point

                //declare local storage
                double u=0.0,u0=0.0,
                  grad_u[nSpace],
                  m=0.0,dm=0.0,
                  H=0.0,dH[nSpace],
                  m_t=0.0,dm_t=0.0,r=0.0,
                  dH_tau[nSpace],//dH or dH_last if lagging for tau formula
                  dH_strong[nSpace],//dH or dH_last if lagging for strong residual and adjoint
                  dpdeResidual_u_u[nDOF_trial_element],
                  Lstar_u[nDOF_test_element],
                  dsubgridError_u_u[nDOF_trial_element],
                  tau=0.0,tau0=0.0,tau1=0.0,
                  nu_sc=0.0,
                  jac[nSpace*nSpace],
                  jacDet,
                  jacInv[nSpace*nSpace],
                  u_grad_trial[nDOF_trial_element*nSpace],
                  dV,
                  u_test_dV[nDOF_test_element],
                  u_grad_test_dV[nDOF_test_element*nSpace],
                  x,y,z,
                  G[nSpace*nSpace],G_dd_G,tr_G;
                //
                //calculate solution and gradients at quadrature points
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
                ck.calculateH_element(eN,
                                      k,
                                      nodeDiametersArray.data(),
                                      mesh_l2g.data(),
                                      mesh_trial_ref.data(),
                                      h_phi);
                //get the physical integration weight
                dV = fabs(jacDet)*dV_ref.data()[k];
                ck.calculateG(jacInv,G,G_dd_G,tr_G);
                //get the trial function gradients
                ck.gradTrialFromRef(&u_grad_trial_ref.data()[k*nDOF_trial_element*nSpace],jacInv,u_grad_trial);
                //get the solution
                ck.valFromDOF(u_dof.data(),&u_l2g.data()[eN_nDOF_trial_element],&u_trial_ref.data()[k*nDOF_trial_element],u);
                ck.valFromDOF(gf.exact.phi_dof_corrected,dummy_l2g,&u_trial_ref.data()[k*nDOF_trial_element],u0);
                if (freezeLevelSet)
                  u0 = phi_ls.data()[eN_k];
                //u0 = phi_ls.data()[eN_k];//cek debug
                /* double DX=(x-0.5),DY=(y-0.75); */
                /* double radius = std::sqrt(DX*DX+DY*DY); */
                /* double theta = std::atan2(DY,DX); */
                /* double kp=10.0, scaling=1.0, rp=1; */
                /* u0 = scaling*std::pow((0.15+(0.015/2.)*std::cos(kp*theta) - radius),rp);     */
                //u0 = 0.15 - std::sqrt(DX*DX + DY*DY);
                //u0 = phi_ls.data()[eN_k];//cek debug--set to exact input
                /* double u0_test=0; */
                /* ck.valFromDOF(phi_dof.data(),&u_l2g.data()[eN_nDOF_trial_element],&u_trial_ref.data()[k*nDOF_trial_element],u0_test); */
                /* if (u0 != u0_test) */
                /*   std::cout<<"JAC eN "<<eN<<" k "<<k<<" u0 "<<u0<<" u0_test "<<u0_test<<std::endl; */

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
                /* norm = 1.0e-8; */
                /* for (int I=0;I<nSpace;I++) */
                /*        norm += grad_u[I]*grad_u[I]; */
                /* norm = sqrt(norm); */
                /* for (int I=0;I<nSpace;I++) */
                /*        dir[I] = grad_u[I]/norm; */

                /* ck.calculateGScale(G,dir,h_phi); */

                epsilon_redist = epsFact_redist*(useMetrics*h_phi+(1.0-useMetrics)*elementDiameter.data()[eN]);

                evaluateCoefficients(epsilon_redist,
                                     u0,
                                     u,
                                     grad_u,
                                     m,
                                     dm,
                                     H,
                                     dH,
                                     r);
                //TODO allow not lagging of subgrid error etc
                //remove conditional?
                //default no lagging
                for (int I=0; I < nSpace; I++)
                  {
                    dH_tau[I] = dH[I];
                    dH_strong[I] = dH[I];
                  }
                if (lag_subgridError > 0)
                  {
                    for (int I=0; I < nSpace; I++)
                      {
                        dH_tau[I] = q_dH_last.data()[eN_k_nSpace+I];
                      }
                  }
                if (lag_subgridError > 1)
                  {
                    for (int I=0; I < nSpace; I++)
                      {
                        dH_strong[I] = q_dH_last.data()[eN_k_nSpace+I];
                      }
                  }
                //
                //moving mesh
                //
                //omit for now
                //
                //calculate time derivatives
                //
                ck.bdf(alphaBDF,
                       q_m_betaBDF.data()[eN_k],
                       m,
                       dm,
                       m_t,
                       dm_t);
                //TODO add option to skip if not doing time integration (Newton stead-state solve)
                m *= timeIntegrationScale; dm *= timeIntegrationScale; m_t *= timeIntegrationScale;
                dm_t *= timeIntegrationScale;

                //
                //calculate subgrid error contribution to the Jacobian (strong residual, adjoint, jacobian of strong residual)
                //
                //calculate the adjoint times the test functions
                for (int i=0;i<nDOF_test_element;i++)
                  {
                    //int eN_k_i_nSpace = (eN_k*nDOF_trial_element+i)*nSpace;
                    int i_nSpace=i*nSpace;

                    Lstar_u[i]=ck.Hamiltonian_adjoint(dH_strong,&u_grad_test_dV[i_nSpace]);

                  }
                //calculate the Jacobian of strong residual
                for (int j=0;j<nDOF_trial_element;j++)
                  {
                    //int eN_k_j=eN_k*nDOF_trial_element+j;
                    //int eN_k_j_nSpace = eN_k_j*nSpace;
                    int j_nSpace = j*nSpace;
                    dpdeResidual_u_u[j]=ck.MassJacobian_strong(dm_t,u_trial_ref.data()[k*nDOF_trial_element+j]) +
                      ck.HamiltonianJacobian_strong(dH_strong,&u_grad_trial[j_nSpace]);
                  }
                //tau and tau*Res
                calculateSubgridError_tau(elementDiameter.data()[eN],
                                          dm_t,
                                          dH_tau,
                                          q_cfl.data()[eN_k],
                                          tau0);
                calculateSubgridError_tau(G,
                                          dH_tau,
                                          tau1,
                                          q_cfl.data()[eN_k]);

                tau = useMetrics*tau1+(1.0-useMetrics)*tau0;

                for (int j=0;j<nDOF_trial_element;j++)
                  dsubgridError_u_u[j] =  -tau*dpdeResidual_u_u[j];

                nu_sc = q_numDiff_u.data()[eN_k]*(1.0-lag_shockCapturingScale) + q_numDiff_u_last.data()[eN_k]*lag_shockCapturingScale;
                /* double epsilon_background_diffusion = 3.0*epsFact_redist*h_phi; */
                /* if (fabs(u0) >  epsilon_background_diffusion) */
                /*   nu_sc += backgroundDiffusionFactor*h_phi; */
                double epsilon_background_diffusion = 2.0*epsFact_redist*(useMetrics*h_phi+(1.0-useMetrics)*elementDiameter.data()[eN]);
                if (fabs(phi_ls.data()[eN_k]) >  epsilon_background_diffusion)
                  nu_sc += backgroundDiffusionFactor*elementDiameter.data()[eN];                //
                for(int i=0;i<nDOF_test_element;i++)
                  {
                    //int eN_k_i=eN_k*nDOF_test_element+i;
                    //int eN_k_i_nSpace=eN_k_i*nSpace;
                    circ += ck.Reaction_weak(gf.D(epsilon_redist, phi_ls.data()[eN_k]), u_test_dV[i]);
                    for(int j=0;j<nDOF_trial_element;j++)
                      {
                        //int eN_k_j=eN_k*nDOF_trial_element+j;
                        //int eN_k_j_nSpace = eN_k_j*nSpace;
                        int j_nSpace = j*nSpace;
                        int i_nSpace = i*nSpace;
                        double FREEZE=double(freezeLevelSet);
                        //std::cout<<element_phi[i]<<'\t'<<element_nodes[i*3+0]<<'\t'<<element_nodes[i*3+1]<<std::endl;
                        //std::cout<<"eN_k "<<eN_k<<" D-J "<<gf.D(epsilon_redist,phi_ls.data()[eN_k])<<std::endl;
                        elementJacobian_u_u[i][j] += ck.MassJacobian_weak(dm_t,u_trial_ref.data()[k*nDOF_trial_element+j],u_test_dV[i]) +
                          ck.HamiltonianJacobian_weak(dH,&u_grad_trial[j_nSpace],u_test_dV[i]) +
                          (1.0-FREEZE)*(weakDirichletFactor/h_phi)*ck.ReactionJacobian_weak(-gf.D(epsilon_redist,u0),
                                                                                             u_trial_ref.data()[k*nDOF_trial_element+j],
                                                                                             u_test_dV[i]) +
                          ck.SubgridErrorJacobian(dsubgridError_u_u[j],Lstar_u[i]) +
                          ck.NumericalDiffusionJacobian(nu_sc,&u_grad_trial[j_nSpace],&u_grad_test_dV[i_nSpace]);
                      }//j
                  }//i
              }//k
            //
            //load into element Jacobian into global Jacobian
            //

            //now try to account for weak dirichlet conditions in interior (frozen level set values)
            if (freezeLevelSet)
              {
                //assume correspondence between dof and equations
                for (int j = 0; j < nDOF_trial_element; j++)
                  {
                    const int J = u_l2g.data()[eN*nDOF_trial_element+j];
                    if (fabs(u_weak_internal_bc_dofs.data()[J]) < epsilon_redist)
                    //if (weakDirichletConditionFlags[J] == 1)
                    //if (fabs(gf.exact.phi_dof_corrected[j]) < epsilon_redist)
                      {
                        for (int jj=0; jj < nDOF_trial_element; jj++)
                          elementJacobian_u_u[j][jj] = 0.0;
                        elementJacobian_u_u[j][j] = weakDirichletFactor*elementDiameter.data()[eN];
                      }
                  }
              }
            for (int i=0;i<nDOF_test_element;i++)
              {
                int eN_i = eN*nDOF_test_element+i;
                for (int j=0;j<nDOF_trial_element;j++)
                  {
                    int eN_i_j = eN_i*nDOF_trial_element+j;
#ifdef CKDEBUG
                    std::cout<<"element jacobian i = "<<i<<"\t"<<"j = "<<j<<"\t"<<elementJacobian_u_u[i][j]<<std::endl;
#endif
                    globalJacobian.data()[csrRowIndeces_u_u.data()[eN_i] + csrColumnOffsets_u_u.data()[eN_i_j]] += elementJacobian_u_u[i][j];
                  }//j
              }//i
          }//elements
        /* //cek todo should get rid of this, see res */
        /* // */
        /* //loop over exterior element boundaries to compute the surface integrals and load them into the global Jacobian */
        /* // */
        /* gf.useExact=false;//exact Heaviside integration not implemented for boundaries yet */
        /* for (int ebNE = 0; ebNE < nExteriorElementBoundaries_global; ebNE++) */
        /*   { */
        /*     int ebN = exteriorElementBoundariesArray.data()[ebNE]; */
        /*     int eN  = elementBoundaryElementsArray.data()[ebN*2+0], */
        /*       ebN_local = elementBoundaryLocalElementBoundariesArray.data()[ebN*2+0], */
        /*       eN_nDOF_trial_element = eN*nDOF_trial_element; */
        /*     double epsilon_redist,h_phi; */
        /*     for  (int kb=0;kb<nQuadraturePoints_elementBoundary;kb++) */
        /*       { */
        /*         int ebNE_kb = ebNE*nQuadraturePoints_elementBoundary+kb; */
        /*         //int ebNE_kb_nSpace = ebNE_kb*nSpace; */
        /*         int ebN_local_kb = ebN_local*nQuadraturePoints_elementBoundary+kb; */
        /*         int ebN_local_kb_nSpace = ebN_local_kb*nSpace; */

        /*         double u_ext=0.0, */
        /*           grad_u_ext[nSpace], */
        /*           m_ext=0.0, */
        /*           dm_ext=0.0, */
        /*           H_ext=0.0, */
        /*           dH_ext[nSpace], */
        /*           r_ext=0.0, */
        /*           //              flux_ext=0.0, */
        /*           dflux_u_u_ext=0.0, */
        /*           bc_u_ext=0.0, */
        /*           bc_grad_u_ext[nSpace], */
        /*           bc_m_ext=0.0, */
        /*           bc_dm_ext=0.0, */
        /*           bc_H_ext=0.0, */
        /*           bc_dH_ext[nSpace], */
        /*           bc_r_ext=0.0, */
        /*           fluxJacobian_u_u[nDOF_trial_element], */
        /*           jac_ext[nSpace*nSpace], */
        /*           jacDet_ext, */
        /*           jacInv_ext[nSpace*nSpace], */
        /*           boundaryJac[nSpace*(nSpace-1)], */
        /*           metricTensor[(nSpace-1)*(nSpace-1)], */
        /*           metricTensorDetSqrt, */
        /*           dS, */
        /*           u_test_dS[nDOF_test_element], */
        /*           u_grad_trial_trace[nDOF_trial_element*nSpace], */
        /*           normal[nSpace],x_ext,y_ext,z_ext, */
        /*           G[nSpace*nSpace],G_dd_G,tr_G, dir[nSpace],norm; */
        /*         // */
        /*         //calculate the solution and gradients at quadrature points */
        /*         // */
        /*         ck.calculateMapping_elementBoundary(eN, */
        /*                                             ebN_local, */
        /*                                             kb, */
        /*                                             ebN_local_kb, */
        /*                                             mesh_dof.data(), */
        /*                                             mesh_l2g.data(), */
        /*                                             mesh_trial_trace_ref.data(), */
        /*                                             mesh_grad_trial_trace_ref.data(), */
        /*                                             boundaryJac_ref.data(), */
        /*                                             jac_ext, */
        /*                                             jacDet_ext, */
        /*                                             jacInv_ext, */
        /*                                             boundaryJac, */
        /*                                             metricTensor, */
        /*                                             metricTensorDetSqrt, */
        /*                                             normal_ref.data(), */
        /*                                             normal, */
        /*                                             x_ext,y_ext,z_ext); */
        /*         dS = metricTensorDetSqrt*dS_ref.data()[kb]; */
        /*         ck.calculateG(jacInv_ext,G,G_dd_G,tr_G); */
        /*         //compute shape and solution information */
        /*         //shape */
        /*         ck.gradTrialFromRef(&u_grad_trial_trace_ref.data()[ebN_local_kb_nSpace*nDOF_trial_element],jacInv_ext,u_grad_trial_trace); */
        /*         //solution and gradients */
        /*         ck.valFromDOF(u_dof.data(),&u_l2g.data()[eN_nDOF_trial_element],&u_trial_trace_ref.data()[ebN_local_kb*nDOF_test_element],u_ext); */
        /*         ck.gradFromDOF(u_dof.data(),&u_l2g.data()[eN_nDOF_trial_element],u_grad_trial_trace,grad_u_ext); */
        /*         //precalculate test function products with integration weights */
        /*         for (int j=0;j<nDOF_trial_element;j++) */
        /*           { */
        /*             u_test_dS[j] = u_test_trace_ref.data()[ebN_local_kb*nDOF_test_element+j]*dS; */
        /*           } */

        /*         norm = 1.0e-8; */
        /*         for (int I=0;I<nSpace;I++) */
        /*           norm += grad_u_ext[I]*grad_u_ext[I]; */
        /*         norm = sqrt(norm); */
        /*         for(int I=0;I<nSpace;I++) */
        /*           dir[I] = grad_u_ext[I]/norm; */

        /*         ck.calculateGScale(G,dir,h_phi); */
        /*         epsilon_redist = epsFact_redist*(useMetrics*h_phi+(1.0-useMetrics)*elementDiameter.data()[eN]); */

        /*         // */
        /*         //load the boundary values */
        /*         // */
        /*         bc_u_ext = isDOFBoundary_u.data()[ebNE_kb]*ebqe_bc_u_ext.data()[ebNE_kb]+(1-isDOFBoundary_u.data()[ebNE_kb])*u_ext; */
        /*         // */
        /*         //calculate the internal and external trace of the pde coefficients */
        /*         // */
        /*         evaluateCoefficients(epsilon_redist, */
        /*                              ebqe_phi_ls_ext.data()[ebNE_kb], */
        /*                              u_ext, */
        /*                              grad_u_ext, */
        /*                              m_ext, */
        /*                              dm_ext, */
        /*                              H_ext, */
        /*                              dH_ext, */
        /*                              r_ext); */
        /*         evaluateCoefficients(epsilon_redist, */
        /*                              ebqe_phi_ls_ext.data()[ebNE_kb], */
        /*                              bc_u_ext, */
        /*                              bc_grad_u_ext, */
        /*                              bc_m_ext, */
        /*                              bc_dm_ext, */
        /*                              bc_H_ext, */
        /*                              bc_dH_ext, */
        /*                              bc_r_ext); */
        /*         // */
        /*         //calculate the numerical fluxes */
        /*         // */
        /*         //DoNothing for now */
        /*         // */
        /*         //calculate the flux jacobian */
        /*         // */
        /*         for (int j=0;j<nDOF_trial_element;j++) */
        /*           { */
        /*             //int ebNE_kb_j = ebNE_kb*nDOF_trial_element+j; */
        /*             //int ebNE_kb_j_nSpace = ebNE_kb_j*nSpace; */
        /*             //int j_nSpace = j*nSpace; */
        /*             int ebN_local_kb_j=ebN_local_kb*nDOF_trial_element+j; */

        /*             fluxJacobian_u_u[j]=ck.ExteriorNumericalAdvectiveFluxJacobian(dflux_u_u_ext,u_trial_trace_ref.data()[ebN_local_kb_j]); */
        /*           }//j */
        /*         // */
        /*         //update the global Jacobian from the flux Jacobian */
        /*         // */
        /*         for (int i=0;i<nDOF_test_element;i++) */
        /*           { */
        /*             int eN_i = eN*nDOF_test_element+i; */
        /*             //int ebNE_kb_i = ebNE_kb*nDOF_test_element+i; */
        /*             for (int j=0;j<nDOF_trial_element;j++) */
        /*               { */
        /*                 int ebN_i_j = ebN*4*nDOF_test_X_trial_element + i*nDOF_trial_element + j; */
        /*                 //mwf debug */
        /*                 assert(fluxJacobian_u_u[j] == 0.0); */
        /*                 globalJacobian.data()[csrRowIndeces_u_u.data()[eN_i] + csrColumnOffsets_eb_u_u.data()[ebN_i_j]] += fluxJacobian_u_u[j]*u_test_dS[i]; */
        /*               }//j */
        /*           }//i */
        /*       }//kb */
        /*   }//ebNE */
        /* gf.useExact = useExact;//just to be safe */
      }//computeJacobian

      void calculateResidual_ellipticRedist(arguments_dict& args,
                                            bool useExact)
      {
        proteus::pyarray<double>& mesh_trial_ref = args.array<double>("mesh_trial_ref");
        proteus::pyarray<double>& mesh_grad_trial_ref = args.array<double>("mesh_grad_trial_ref");
        proteus::pyarray<double>& mesh_dof = args.array<double>("mesh_dof");
        proteus::pyarray<int>& mesh_l2g = args.array<int>("mesh_l2g");
        proteus::pyarray<double>& x_ref = args.array<double>("x_ref");
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
        double epsFact_redist = args.scalar<double>("epsFact_redist");
        double backgroundDiffusionFactor = args.scalar<double>("backgroundDiffusionFactor");
        double weakDirichletFactor = args.scalar<double>("weakDirichletFactor");
        int freezeLevelSet = args.scalar<int>("freezeLevelSet");
        int useTimeIntegration = args.scalar<int>("useTimeIntegration");
        int lag_shockCapturing = args.scalar<int>("lag_shockCapturing");
        int lag_subgridError = args.scalar<int>("lag_subgridError");
        double shockCapturingDiffusion = args.scalar<double>("shockCapturingDiffusion");
        proteus::pyarray<int>& u_l2g = args.array<int>("u_l2g");
        proteus::pyarray<double>& elementDiameter = args.array<double>("elementDiameter");
        proteus::pyarray<double>& nodeDiametersArray = args.array<double>("nodeDiametersArray");
        proteus::pyarray<double>& u_dof = args.array<double>("u_dof");
        proteus::pyarray<double>& phi_dof = args.array<double>("phi_dof");
        proteus::pyarray<double>& phi_ls = args.array<double>("phi_ls");
        proteus::pyarray<double>& q_m = args.array<double>("q_m");
        proteus::pyarray<double>& q_u = args.array<double>("q_u");
        proteus::pyarray<double>& q_n = args.array<double>("q_n");
        proteus::pyarray<double>& q_dH = args.array<double>("q_dH");
        proteus::pyarray<double>& u_weak_internal_bc_dofs = args.array<double>("u_weak_internal_bc_dofs");
        proteus::pyarray<double>& q_m_betaBDF = args.array<double>("q_m_betaBDF");
        proteus::pyarray<double>& q_dH_last = args.array<double>("q_dH_last");
        proteus::pyarray<double>& q_cfl = args.array<double>("q_cfl");
        proteus::pyarray<double>& q_numDiff_u = args.array<double>("q_numDiff_u");
        proteus::pyarray<double>& q_numDiff_u_last = args.array<double>("q_numDiff_u_last");
        proteus::pyarray<int>& weakDirichletConditionFlags = args.array<int>("weakDirichletConditionFlags");
        int offset_u = args.scalar<int>("offset_u");
        int stride_u = args.scalar<int>("stride_u");
        proteus::pyarray<double>& globalResidual = args.array<double>("globalResidual");
        int nExteriorElementBoundaries_global = args.scalar<int>("nExteriorElementBoundaries_global");
        proteus::pyarray<int>& exteriorElementBoundariesArray = args.array<int>("exteriorElementBoundariesArray");
        proteus::pyarray<int>& elementBoundaryElementsArray = args.array<int>("elementBoundaryElementsArray");
        proteus::pyarray<int>& elementBoundaryLocalElementBoundariesArray = args.array<int>("elementBoundaryLocalElementBoundariesArray");
        proteus::pyarray<double>& ebqe_phi_ls_ext = args.array<double>("ebqe_phi_ls_ext");
        proteus::pyarray<int>& isDOFBoundary_u = args.array<int>("isDOFBoundary_u");
        proteus::pyarray<double>& ebqe_bc_u_ext = args.array<double>("ebqe_bc_u_ext");
        proteus::pyarray<double>& ebqe_u = args.array<double>("ebqe_u");
        proteus::pyarray<double>& ebqe_n = args.array<double>("ebqe_n");
        int ELLIPTIC_REDISTANCING = args.scalar<int>("ELLIPTIC_REDISTANCING");
        double backgroundDissipationEllipticRedist = args.scalar<double>("backgroundDissipationEllipticRedist");
        proteus::pyarray<double>& lumped_qx = args.array<double>("lumped_qx");
        proteus::pyarray<double>& lumped_qy = args.array<double>("lumped_qy");
        proteus::pyarray<double>& lumped_qz = args.array<double>("lumped_qz");
        double alpha = args.scalar<double>("alpha");
        gf.useExact=useExact;
        //
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
            double elementResidual_u[nDOF_test_element],element_phi[nDOF_trial_element];
            double epsilon_redist,h_phi, norm;
            for (int i=0;i<nDOF_test_element;i++)
              {
		int eN_i=eN*nDOF_trial_element+i;
                elementResidual_u[i]=0.0;
                element_phi[i] = phi_dof.data()[u_l2g.data()[eN_i]];
              }//i
            double element_nodes[nDOF_mesh_trial_element*3];
	    for (int i=0;i<nDOF_mesh_trial_element;i++)
	      {
		int eN_i=eN*nDOF_mesh_trial_element+i;
                for(int I=0;I<3;I++)
                  element_nodes[i*3 + I] = mesh_dof.data()[mesh_l2g.data()[eN_i]*3 + I];
	      }//i
            gf.calculate(element_phi, element_nodes, x_ref.data(),false);                        
            //loop over quadrature points and compute integrands
            for  (int k=0;k<nQuadraturePoints_element;k++)
              {
                gf.set_quad(k);
                //compute indeces and declare local storage
                int eN_k = eN*nQuadraturePoints_element+k,
                  eN_k_nSpace = eN_k*nSpace,
                  eN_nDOF_trial_element = eN*nDOF_trial_element;
                double
                  coeff, delta,
                  qx, qy, qz, normalReconstruction[nSpace],
                  u=0,grad_u[nSpace],
                  m=0.0,
                  jac[nSpace*nSpace], jacDet, jacInv[nSpace*nSpace],
                  u_grad_trial[nDOF_trial_element*nSpace],
                  u_test_dV[nDOF_trial_element], u_grad_test_dV[nDOF_test_element*nSpace],
                  dV,x,y,z,G[nSpace*nSpace],G_dd_G,tr_G;
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
                //get the physical integration weight
                dV = fabs(jacDet)*dV_ref.data()[k];
                ck.calculateG(jacInv,G,G_dd_G,tr_G);
                //get the trial function gradients
                ck.gradTrialFromRef(&u_grad_trial_ref.data()[k*nDOF_trial_element*nSpace],
                                    jacInv,
                                    u_grad_trial);
                //get the solution
                ck.valFromDOF(u_dof.data(),
                              &u_l2g.data()[eN_nDOF_trial_element],
                              &u_trial_ref.data()[k*nDOF_trial_element],
                              u);
                //get the solution gradients
                ck.gradFromDOF(u_dof.data(),
                               &u_l2g.data()[eN_nDOF_trial_element],
                               u_grad_trial,
                               grad_u);
                if (ELLIPTIC_REDISTANCING > 1)
                  { // use linear elliptic re-distancing via C0 normal reconstruction
                    ck.valFromDOF(lumped_qx.data(),
                                  &u_l2g.data()[eN_nDOF_trial_element],
                                  &u_trial_ref.data()[k*nDOF_trial_element],
                                  qx);
                    ck.valFromDOF(lumped_qy.data(),
                                  &u_l2g.data()[eN_nDOF_trial_element],
                                  &u_trial_ref.data()[k*nDOF_trial_element],
                                  qy);
                    ck.valFromDOF(lumped_qz.data(),
                                  &u_l2g.data()[eN_nDOF_trial_element],
                                  &u_trial_ref.data()[k*nDOF_trial_element],
                                  qz);
                    normalReconstruction[0] = qx;
                    normalReconstruction[1] = qy;
                    if (nSpace == 3)
                      normalReconstruction[2] = qz;
                  }
                //precalculate test function products with integration weights
                for (int j=0;j<nDOF_trial_element;j++)
                  {
                    u_test_dV[j] = u_test_ref.data()[k*nDOF_trial_element+j]*dV;
                    for (int I=0;I<nSpace;I++)
                      u_grad_test_dV[j*nSpace+I] = u_grad_trial[j*nSpace+I]*dV;
                  }
                // MOVING MESH. Omit for now //
                // COMPUTE NORM OF GRAD(u) //
                double norm_grad_u = 0.;
                for (int I=0;I<nSpace;I++)
                  norm_grad_u += grad_u[I]*grad_u[I];
                norm_grad_u = std::sqrt(norm_grad_u) + 1.0E-10;

                // SAVE MASS AND SOLUTION FOR OTHER MODELS //
                q_m.data()[eN_k] = u; //m=u
                q_u.data()[eN_k] = u;
                for (int I=0;I<nSpace;I++)
                  q_n.data()[eN_k_nSpace+I] = grad_u[I]/norm_grad_u;

                // COMPUTE COEFFICIENTS //
                if (SINGLE_POTENTIAL == 1)
                  coeff = 1.0-1.0/norm_grad_u; //single potential
                else // double potential
                  coeff = 1.0+2*std::pow(norm_grad_u,2)-3*norm_grad_u;

                // COMPUTE DELTA FUNCTION //
                epsilon_redist = epsFact_redist*(useMetrics*h_phi
                                                 +(1.0-useMetrics)*elementDiameter.data()[eN]);
                delta = gf.D(epsilon_redist,phi_ls.data()[eN_k]);

		// COMPUTE STRONG RESIDUAL //
		double Si = -1.0+2.0*gf.H(epsilon_redist,phi_ls.data()[eN_k]);
		double residualEikonal = Si*(norm_grad_u-1.0);
		double backgroundDissipation = backgroundDissipationEllipticRedist*elementDiameter.data()[eN];

                // UPDATE ELEMENT RESIDUAL //
		for(int i=0;i<nDOF_test_element;i++)
		  {
		    int i_nSpace = i*nSpace;
		    // global i-th index
		    int gi = offset_u+stride_u*u_l2g.data()[eN*nDOF_test_element+i];

		    if (ELLIPTIC_REDISTANCING > 1) // (NON)LINEAR VIA C0 NORMAL RECONSTRUCTION
		      {
			elementResidual_u[i] +=
			  residualEikonal*u_test_dV[i]
			  +ck.NumericalDiffusion(1.0+backgroundDissipation,
						grad_u,
						&u_grad_test_dV[i_nSpace])
			  -ck.NumericalDiffusion(1.0,
						 normalReconstruction,
						 &u_grad_test_dV[i_nSpace])
			  +alpha*(u_dof.data()[gi]-phi_dof.data()[gi])*delta*u_test_dV[i]; // BCs
		      }
		    else // =1. Nonlinear via single or double pot.
		      {
			elementResidual_u[i] +=
			  residualEikonal*u_test_dV[i]
			  +ck.NumericalDiffusion(coeff+backgroundDissipation,
						 grad_u,
						 &u_grad_test_dV[i_nSpace])
			  + alpha*(u_dof.data()[gi]-phi_dof.data()[gi])*delta*u_test_dV[i]; // BCs
		      }
		  }//i
              }//k
            //
            //load element into global residual and save element residual
            //
            for(int i=0;i<nDOF_test_element;i++)
              {
                int eN_i=eN*nDOF_test_element+i;
                globalResidual.data()[offset_u+stride_u*u_l2g.data()[eN_i]]+=elementResidual_u[i];
              }//i
          }//elements
        //
        //loop over exterior element boundaries to save soln at quad points
        //
        for (int ebNE = 0; ebNE < nExteriorElementBoundaries_global; ebNE++)
          {
            int ebN = exteriorElementBoundariesArray.data()[ebNE],
              eN  = elementBoundaryElementsArray.data()[ebN*2+0],
              ebN_local = elementBoundaryLocalElementBoundariesArray.data()[ebN*2+0],
              eN_nDOF_trial_element = eN*nDOF_trial_element;
            for  (int kb=0;kb<nQuadraturePoints_elementBoundary;kb++)
              {
                int ebNE_kb = ebNE*nQuadraturePoints_elementBoundary+kb,
                  ebNE_kb_nSpace = ebNE_kb*nSpace,
                  ebN_local_kb = ebN_local*nQuadraturePoints_elementBoundary+kb,
                  ebN_local_kb_nSpace = ebN_local_kb*nSpace;
                double
                  u_ext=0.0,
                  grad_u_ext[nSpace],
                  jac_ext[nSpace*nSpace],jacDet_ext,jacInv_ext[nSpace*nSpace],
                  boundaryJac[nSpace*(nSpace-1)],
                  metricTensor[(nSpace-1)*(nSpace-1)],metricTensorDetSqrt,
                  u_grad_trial_trace[nDOF_trial_element*nSpace],
                  normal[nSpace],x_ext,y_ext,z_ext,
                  dir[nSpace],norm;
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
                //compute shape and solution information
                //shape
                ck.gradTrialFromRef(&u_grad_trial_trace_ref.data()[ebN_local_kb_nSpace*nDOF_trial_element],
                                    jacInv_ext,
                                    u_grad_trial_trace);
                //solution and gradients
                ck.valFromDOF(u_dof.data(),
                              &u_l2g.data()[eN_nDOF_trial_element],
                              &u_trial_trace_ref.data()[ebN_local_kb*nDOF_test_element],
                              u_ext);
                ck.gradFromDOF(u_dof.data(),
                               &u_l2g.data()[eN_nDOF_trial_element],
                               u_grad_trial_trace,
                               grad_u_ext);
                norm = 0;
                for (int I=0;I<nSpace;I++)
                  norm += grad_u_ext[I]*grad_u_ext[I];
                norm = sqrt(norm) + 1.0E-10;
                for (int I=0;I<nSpace;I++)
                  dir[I] = grad_u_ext[I]/norm;

                //save for other models
                ebqe_u.data()[ebNE_kb] = u_ext;

                for (int I=0;I<nSpace;I++)
                  ebqe_n.data()[ebNE_kb_nSpace+I] = dir[I];
              }//kb
          }//ebNE
      }

      void calculateJacobian_ellipticRedist(arguments_dict& args,
                                            bool useExact)
      {
        proteus::pyarray<double>& mesh_trial_ref = args.array<double>("mesh_trial_ref");
        proteus::pyarray<double>& mesh_grad_trial_ref = args.array<double>("mesh_grad_trial_ref");
        proteus::pyarray<double>& mesh_dof = args.array<double>("mesh_dof");
        proteus::pyarray<int>& mesh_l2g = args.array<int>("mesh_l2g");
        proteus::pyarray<double>& x_ref = args.array<double>("x_ref");
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
        double epsFact_redist = args.scalar<double>("epsFact_redist");
        double backgroundDiffusionFactor = args.scalar<double>("backgroundDiffusionFactor");
        double weakDirichletFactor = args.scalar<double>("weakDirichletFactor");
        int freezeLevelSet = args.scalar<int>("freezeLevelSet");
        int useTimeIntegration = args.scalar<int>("useTimeIntegration");
        int lag_shockCapturing = args.scalar<int>("lag_shockCapturing");
        int lag_subgridError = args.scalar<int>("lag_subgridError");
        double shockCapturingDiffusion = args.scalar<double>("shockCapturingDiffusion");
        proteus::pyarray<int>& u_l2g = args.array<int>("u_l2g");
        proteus::pyarray<double>& elementDiameter = args.array<double>("elementDiameter");
        proteus::pyarray<double>& nodeDiametersArray = args.array<double>("nodeDiametersArray");
        proteus::pyarray<double>& u_dof = args.array<double>("u_dof");
        proteus::pyarray<double>& phi_dof = args.array<double>("phi_dof");
        proteus::pyarray<double>& u_weak_internal_bc_dofs = args.array<double>("u_weak_internal_bc_dofs");
        proteus::pyarray<double>& phi_ls = args.array<double>("phi_ls");
        proteus::pyarray<double>& q_m_betaBDF = args.array<double>("q_m_betaBDF");
        proteus::pyarray<double>& q_dH_last = args.array<double>("q_dH_last");
        proteus::pyarray<double>& q_cfl = args.array<double>("q_cfl");
        proteus::pyarray<double>& q_numDiff_u = args.array<double>("q_numDiff_u");
        proteus::pyarray<double>& q_numDiff_u_last = args.array<double>("q_numDiff_u_last");
        proteus::pyarray<int>& weakDirichletConditionFlags = args.array<int>("weakDirichletConditionFlags");
        proteus::pyarray<int>& csrRowIndeces_u_u = args.array<int>("csrRowIndeces_u_u");
        proteus::pyarray<int>& csrColumnOffsets_u_u = args.array<int>("csrColumnOffsets_u_u");
        proteus::pyarray<double>& globalJacobian = args.array<double>("globalJacobian");
        int nExteriorElementBoundaries_global = args.scalar<int>("nExteriorElementBoundaries_global");
        proteus::pyarray<int>& exteriorElementBoundariesArray = args.array<int>("exteriorElementBoundariesArray");
        proteus::pyarray<int>& elementBoundaryElementsArray = args.array<int>("elementBoundaryElementsArray");
        proteus::pyarray<int>& elementBoundaryLocalElementBoundariesArray = args.array<int>("elementBoundaryLocalElementBoundariesArray");
        proteus::pyarray<double>& ebqe_phi_ls_ext = args.array<double>("ebqe_phi_ls_ext");
        proteus::pyarray<int>& isDOFBoundary_u = args.array<int>("isDOFBoundary_u");
        proteus::pyarray<double>& ebqe_bc_u_ext = args.array<double>("ebqe_bc_u_ext");
        proteus::pyarray<int>& csrColumnOffsets_eb_u_u = args.array<int>("csrColumnOffsets_eb_u_u");
        int ELLIPTIC_REDISTANCING = args.scalar<int>("ELLIPTIC_REDISTANCING");
        double backgroundDissipationEllipticRedist = args.scalar<double>("backgroundDissipationEllipticRedist");
        double alpha = args.scalar<double>("alpha");
        gf.useExact=useExact;
        //
        //loop over elements
        //
        for(int eN=0;eN<nElements_global;eN++)
          {
            double  elementJacobian_u_u[nDOF_test_element][nDOF_trial_element],element_phi[nDOF_trial_element];
            double epsilon_redist,h_phi, norm;
            for (int i=0;i<nDOF_test_element;i++)
              {
		int eN_i=eN*nDOF_trial_element+i;
                element_phi[i] = phi_dof.data()[u_l2g.data()[eN_i]];
                for (int j=0;j<nDOF_trial_element;j++)
                  {
                    elementJacobian_u_u[i][j]=0.0;
                  }
              }
            double element_nodes[nDOF_mesh_trial_element*3];
	    for (int i=0;i<nDOF_mesh_trial_element;i++)
	      {
		int eN_i=eN*nDOF_mesh_trial_element+i;
                for(int I=0;I<3;I++)
                  element_nodes[i*3 + I] = mesh_dof.data()[mesh_l2g.data()[eN_i]*3 + I];
	      }//i
            gf.calculate(element_phi, element_nodes, x_ref.data(),false);
            for  (int k=0;k<nQuadraturePoints_element;k++)
              {
                gf.set_quad(k);
                int eN_k = eN*nQuadraturePoints_element+k, //index to a scalar at a quadrature point
                  eN_k_nSpace = eN_k*nSpace,
                  eN_nDOF_trial_element = eN*nDOF_trial_element; //index to a vector at a quadrature point

                //declare local storage
                double
                  coeff1, coeff2, delta,
                  grad_u[nSpace],
                  jac[nSpace*nSpace], jacDet, jacInv[nSpace*nSpace],
                  u_grad_trial[nDOF_trial_element*nSpace],
                  dV, u_test_dV[nDOF_test_element], u_grad_test_dV[nDOF_test_element*nSpace],
                  x,y,z,G[nSpace*nSpace],G_dd_G,tr_G;
                //
                //calculate solution and gradients at quadrature points
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
                ck.calculateH_element(eN,
                                      k,
                                      nodeDiametersArray.data(),
                                      mesh_l2g.data(),
                                      mesh_trial_ref.data(),
                                      h_phi);
                //get the physical integration weight
                dV = fabs(jacDet)*dV_ref.data()[k];
                ck.calculateG(jacInv,G,G_dd_G,tr_G);
                //get the trial function gradients
                ck.gradTrialFromRef(&u_grad_trial_ref.data()[k*nDOF_trial_element*nSpace],
                                    jacInv,
                                    u_grad_trial);
                //get the solution gradients
                ck.gradFromDOF(u_dof.data(),
                               &u_l2g.data()[eN_nDOF_trial_element],
                               u_grad_trial,
                               grad_u);
                //precalculate test function products with integration weights
                for (int j=0;j<nDOF_trial_element;j++)
                  {
                    u_test_dV[j] = u_test_ref.data()[k*nDOF_trial_element+j]*dV;
                    for (int I=0;I<nSpace;I++)
                      u_grad_test_dV[j*nSpace+I] = u_grad_trial[j*nSpace+I]*dV;
                  }
                // MOVING MESH. Omit for now //
                // COMPUTE NORM OF GRAD(u) //
                double norm_grad_u = 0;
		for(int I=0;I<nSpace;I++)
		  norm_grad_u += grad_u[I]*grad_u[I];
		norm_grad_u = std::sqrt(norm_grad_u) + 1.0E-10;

                // COMPUTE COEFFICIENTS //
                if (SINGLE_POTENTIAL==1)
                  {
                    coeff1 = 0.; //-1./norm_grad_u;
                    coeff2 = 1./std::pow(norm_grad_u,3);
                  }
                else
                  {
                    coeff1 = fmax(1.0E-10, 2*std::pow(norm_grad_u,2)-3*norm_grad_u);
                    coeff2 = fmax(1.0E-10, 4.-3./norm_grad_u);
                  }

                // COMPUTE DELTA FUNCTION //
                epsilon_redist = epsFact_redist*(useMetrics*h_phi
                                                 +(1.0-useMetrics)*elementDiameter.data()[eN]);
                delta = gf.D(epsilon_redist,phi_ls.data()[eN_k]);

		// COMPUTE STRONG Jacobian //
		double Si = -1.0+2.0*gf.H(epsilon_redist,phi_ls.data()[eN_k]);
		double dH[nSpace];
		for (int I=0; I<nSpace;I++)
		  dH[I] = Si*grad_u[I]/norm_grad_u;
		double backgroundDissipation = backgroundDissipationEllipticRedist*elementDiameter.data()[eN];

		// LOOP IN I-DOFs //
                for(int i=0;i<nDOF_test_element;i++)
                  {
                    int i_nSpace = i*nSpace;
                    for(int j=0;j<nDOF_trial_element;j++)
                      {
                        int j_nSpace = j*nSpace;
                        elementJacobian_u_u[i][j] +=
			  ck.HamiltonianJacobian_weak(dH,&u_grad_trial[j_nSpace],u_test_dV[i])
                          +ck.NumericalDiffusionJacobian(1.0+backgroundDissipation,
							&u_grad_trial[j_nSpace],
							&u_grad_test_dV[i_nSpace])
                          + (ELLIPTIC_REDISTANCING == 1 ? 1. : 0.)*
                          ( ck.NumericalDiffusionJacobian(coeff1,
                                                          &u_grad_trial[j_nSpace],
                                                          &u_grad_test_dV[i_nSpace])
                            + coeff2*dV*
                            ck.NumericalDiffusion(1.0,grad_u,&u_grad_trial[i_nSpace])*
                            ck.NumericalDiffusion(1.0,grad_u,&u_grad_trial[j_nSpace]) )
                          + (i == j ? alpha*delta*u_test_dV[i] : 0.); //lumped
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
                    globalJacobian.data()[csrRowIndeces_u_u.data()[eN_i]
                                   + csrColumnOffsets_u_u.data()[eN_i_j]] += elementJacobian_u_u[i][j];
                  }//j
              }//i
          }//elements
      }//computeJacobian

      void normalReconstruction(arguments_dict& args)
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
        proteus::pyarray<double>& elementDiameter = args.array<double>("elementDiameter");
        proteus::pyarray<double>& u_dof = args.array<double>("u_dof");
        int offset_u = args.scalar<int>("offset_u");
        int stride_u = args.scalar<int>("stride_u");
        int numDOFs = args.scalar<int>("numDOFs");
        proteus::pyarray<double>& lumped_qx = args.array<double>("lumped_qx");
        proteus::pyarray<double>& lumped_qy = args.array<double>("lumped_qy");
        proteus::pyarray<double>& lumped_qz = args.array<double>("lumped_qz");
        weighted_lumped_mass_matrix.resize(numDOFs,0.0);
        for (int i=0; i<numDOFs; i++)
          {
            // output vectors
            lumped_qx.data()[i]=0.;
            lumped_qy.data()[i]=0.;
            lumped_qz.data()[i]=0.;
            // auxiliary vectors
            weighted_lumped_mass_matrix[i]=0.;
          }
        for(int eN=0;eN<nElements_global;eN++)
          {
            //declare local storage for local contributions and initialize
            double
              element_weighted_lumped_mass_matrix[nDOF_test_element],
              element_rhsx_normal_reconstruction[nDOF_test_element],
              element_rhsy_normal_reconstruction[nDOF_test_element],
              element_rhsz_normal_reconstruction[nDOF_test_element];
            for (int i=0;i<nDOF_test_element;i++)
              {
                element_weighted_lumped_mass_matrix[i]=0.0;
                element_rhsx_normal_reconstruction[i]=0.0;
                element_rhsy_normal_reconstruction[i]=0.0;
                element_rhsz_normal_reconstruction[i]=0.0;
              }
            //loop over quadrature points and compute integrands
            for  (int k=0;k<nQuadraturePoints_element;k++)
              {
                //compute indeces and declare local storage
                int eN_k = eN*nQuadraturePoints_element+k,
                  eN_k_nSpace = eN_k*nSpace,
                  eN_nDOF_trial_element = eN*nDOF_trial_element;
                double
                  //for mass matrix contributions
                  grad_u[nSpace],
                  u_grad_trial[nDOF_trial_element*nSpace],
                  u_test_dV[nDOF_trial_element],
                  //for general use
                  jac[nSpace*nSpace], jacDet, jacInv[nSpace*nSpace],
                  dV,x,y,z;
                //get the physical integration weight
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
                dV = fabs(jacDet)*dV_ref.data()[k];
                ck.gradTrialFromRef(&u_grad_trial_ref.data()[k*nDOF_trial_element*nSpace],
                                    jacInv,
                                    u_grad_trial);
                ck.gradFromDOF(u_dof.data(),
                               &u_l2g.data()[eN_nDOF_trial_element],u_grad_trial,
                               grad_u);
                //precalculate test function products with integration weights for mass matrix terms
                for (int j=0;j<nDOF_trial_element;j++)
                  u_test_dV[j] = u_test_ref.data()[k*nDOF_trial_element+j]*dV;

                double rhsx = grad_u[0];
                double rhsy = grad_u[1];
                double rhsz = 0;
                if (nSpace==3)
                  rhsz = grad_u[2];

                double norm_grad_u = 0;
                for (int I=0;I<nSpace; I++)
                  norm_grad_u += grad_u[I]*grad_u[I];
                norm_grad_u = std::sqrt(norm_grad_u) + 1.0E-10;

                for(int i=0;i<nDOF_test_element;i++)
                  {
                    element_weighted_lumped_mass_matrix[i] += norm_grad_u*u_test_dV[i];
                    element_rhsx_normal_reconstruction[i]  += rhsx*u_test_dV[i];
                    element_rhsy_normal_reconstruction[i]  += rhsy*u_test_dV[i];
                    element_rhsz_normal_reconstruction[i]  += rhsz*u_test_dV[i];
                  }
              } //k
            // DISTRIBUTE //
            for(int i=0;i<nDOF_test_element;i++)
              {
                int eN_i=eN*nDOF_test_element+i;
                int gi = offset_u+stride_u*u_l2g.data()[eN_i]; //global i-th index

                weighted_lumped_mass_matrix[gi] += element_weighted_lumped_mass_matrix[i];
                lumped_qx.data()[gi] += element_rhsx_normal_reconstruction[i];
                lumped_qy.data()[gi] += element_rhsy_normal_reconstruction[i];
                lumped_qz.data()[gi] += element_rhsz_normal_reconstruction[i];
              }//i
          }//elements
        // COMPUTE LUMPED L2 PROJECTION
        for (int i=0; i<numDOFs; i++)
          {
            // normal reconstruction
            double weighted_mi = weighted_lumped_mass_matrix[i];
            lumped_qx.data()[i] /= weighted_mi;
            lumped_qy.data()[i] /= weighted_mi;
            lumped_qz.data()[i] /= weighted_mi;
          }
      }

      std::tuple<double, double, double> calculateMetricsAtEOS(arguments_dict& args)
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
        proteus::pyarray<double>& elementDiameter = args.array<double>("elementDiameter");
        double degree_polynomial = args.scalar<double>("degree_polynomial");
        double epsFact_redist = args.scalar<double>("epsFact_redist");
        proteus::pyarray<double>& u_dof = args.array<double>("u_dof");
        proteus::pyarray<double>& u_exact = args.array<double>("u_exact");
        int offset_u = args.scalar<int>("offset_u");
        int stride_u = args.scalar<int>("stride_u)");
        double global_V = 0.;
        double global_V0 = 0.;
        double global_I_err = 0.0;
        double global_V_err = 0.0;
        double global_D_err = 0.0;
        //////////////////////
        // ** LOOP IN CELLS //
        //////////////////////        
        for(int eN=0;eN<nElements_global;eN++)
          {
            //declare local storage for local contributions and initialize
            double
              elementResidual_u[nDOF_test_element];
            double
              cell_mass_error = 0., cell_mass_exact = 0.,
              cell_I_err = 0.,
              cell_V = 0., cell_V0 = 0.,
              cell_D_err = 0.;

            //loop over quadrature points and compute integrands
            for  (int k=0;k<nQuadraturePoints_element;k++)
              {
                //compute indeces and declare local storage
                int eN_k = eN*nQuadraturePoints_element+k,
                  eN_k_nSpace = eN_k*nSpace,
                  eN_nDOF_trial_element = eN*nDOF_trial_element;
                double
                  u, uh,
                  u_grad_trial[nDOF_trial_element*nSpace],
                  grad_uh[nSpace],
                  //for general use
                  jac[nSpace*nSpace], jacDet, jacInv[nSpace*nSpace],
                  dV,x,y,z;
                //get the physical integration weight
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
                dV = fabs(jacDet)*dV_ref.data()[k];
                // get functions at quad points
                ck.valFromDOF(u_dof.data(),
                              &u_l2g.data()[eN_nDOF_trial_element],&u_trial_ref.data()[k*nDOF_trial_element],
                              uh);
                u = u_exact.data()[eN_k];
                // get gradients
                ck.gradTrialFromRef(&u_grad_trial_ref.data()[k*nDOF_trial_element*nSpace],
                                    jacInv,
                                    u_grad_trial);
                ck.gradFromDOF(u_dof.data(),&u_l2g.data()[eN_nDOF_trial_element],u_grad_trial,grad_uh);

                double epsHeaviside = epsFact_redist*elementDiameter.data()[eN]/degree_polynomial;
                // compute (smoothed) heaviside functions //
                double Hu = heaviside(u);
                double Huh = heaviside(uh);
                // compute cell metrics //
                cell_I_err += fabs(Hu - Huh)*dV;
                cell_V   += Huh*dV;
                cell_V0  += Hu*dV;

                double norm2_grad_uh = 0.;
                for (int I=0; I<nSpace; I++)
                  norm2_grad_uh += grad_uh[I]*grad_uh[I];
                cell_D_err += std::pow(std::sqrt(norm2_grad_uh) - 1, 2.)*dV;
              }
            global_V += cell_V;
            global_V0 += cell_V0;
            // metrics //
            global_I_err    += cell_I_err;
            global_D_err    += cell_D_err;
          }//elements
        global_V_err = fabs(global_V0 - global_V)/global_V0;
        global_D_err *= 0.5;
        return std::tuple<double, double, double>(global_I_err, global_V_err, global_D_err);
      }

    };//RDLS
  inline RDLS_base* newRDLS(int nSpaceIn,
                            int nQuadraturePoints_elementIn,
                            int nDOF_mesh_trial_elementIn,
                            int nDOF_trial_elementIn,
                            int nDOF_test_elementIn,
                            int nQuadraturePoints_elementBoundaryIn,
                            int CompKernelFlag)
  {
    if (nSpaceIn == 2)
      return proteus::chooseAndAllocateDiscretization2D<RDLS_base,RDLS,CompKernel>(nSpaceIn,
                                                                                   nQuadraturePoints_elementIn,
                                                                                   nDOF_mesh_trial_elementIn,
                                                                                   nDOF_trial_elementIn,
                                                                                   nDOF_test_elementIn,
                                                                                   nQuadraturePoints_elementBoundaryIn,
                                                                                   CompKernelFlag);
    else
      return proteus::chooseAndAllocateDiscretization<RDLS_base,RDLS,CompKernel>(nSpaceIn,
                                                                                 nQuadraturePoints_elementIn,
                                                                                 nDOF_mesh_trial_elementIn,
                                                                                 nDOF_trial_elementIn,
                                                                                 nDOF_test_elementIn,
                                                                                 nQuadraturePoints_elementBoundaryIn,
                                                                                 CompKernelFlag);
  }

}//proteus

#endif
