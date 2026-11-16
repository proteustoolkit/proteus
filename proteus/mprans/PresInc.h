#ifndef PRESINC_H
#define PRESINC_H
#include <cmath>
#include <iostream>
#include "CompKernel.h"
#include "ModelFactory.h"
#include "ArgumentsDict.h"
#include "pyarray.h"

namespace py = pybind11;

namespace proteus
{
  class cppPresInc_base
  {
  public:
    virtual ~cppPresInc_base(){}
    virtual void calculateResidual(arguments_dict& args)=0;
    virtual void calculateJacobian(arguments_dict& args)=0;
  };

  template<class CompKernelType,
           int nSpace,
           int nQuadraturePoints_element,
           int nDOF_mesh_trial_element,
           int nDOF_trial_element,
           int nDOF_test_element,
           int nQuadraturePoints_elementBoundary>
  class cppPresInc : public cppPresInc_base
  {
  public:
    const int nDOF_test_X_trial_element;
    CompKernelType ck;
    cppPresInc():
      nDOF_test_X_trial_element(nDOF_test_element*nDOF_trial_element),
      ck()
    {}
    inline
      void evaluateCoefficients(const double& alphaBDF,
                                const double vf[nSpace],
                                const double vs[nSpace],
                                const double& vos,
                                const double& rhos_min,
                                const double& rhof_min,
                                double f[nSpace],
                                double& a)
    {
      for (int I=0;I<nSpace;I++)
        f[I] = (1.0-vos)*vf[I] + vos*vs[I];
      //a = 1.0/( vos*rhos_min*alphaBDF) + 1.0/( (1.0-vos)*rhof_min*alphaBDF );
      a = 1.0/( vos*rhos_min*alphaBDF + (1.0-vos)*rhof_min*alphaBDF );
      //a = (1.0-vos)/(rhof_min*alphaBDF) + vos*(rhos_min/alphaBDF); 
      //a = (1.0)/(rhof_min*alphaBDF) ; 
      //a = (1.0-vos)/(rhof_min*alphaBDF) + (vos)/(rhos_min*alphaBDF); 
    }

    inline
      void exteriorNumericalAdvectiveFlux(const int& isFluxBoundary,
                                          const double& bc_flux,
                                          const double n[nSpace],
                                          const double f[nSpace],
                                          double& flux)
    {
      if (isFluxBoundary == 1)
        flux = bc_flux;
      else
        {
          flux = 0.0;
          for (int I=0; I < nSpace; I++)
            flux += n[I]*f[I];
        }
    }

    inline
    void exteriorNumericalDiffusiveFlux(const int& isDOFBoundary,
                                        const int& isFluxBoundary,
                                        const double n[nSpace],
                                        const double& a,
                                        const double grad_potential[nSpace],
                                        const double& u,
                                        const double& bc_u,
                                        const double& bc_flux,
                                        const double& penalty,
                                        double& flux)
    {
      if(isFluxBoundary == 1)
        {
          flux = bc_flux;
        }
      else if(isDOFBoundary == 1)
        {
          flux = 0.0;
          for(int I=0;I<nSpace;I++)
            flux-= a*grad_potential[I]*n[I];
          flux += a*penalty*(u-bc_u);
        }
      else
        {
          std::cerr<<"PresInc.h: warning, diffusion term with no boundary condition set, setting diffusive flux to 0.0"<<std::endl;
          flux = 0.0;
        }
    }

    inline
    double ExteriorNumericalDiffusiveFluxJacobian(const int& isDOFBoundary,
                                                  const int& isFluxBoundary,
                                                  const double n[nSpace],
                                                  const double& a,
                                                  const double& v,
                                                  const double grad_v[nSpace],
                                                  const double& penalty)
    {
      double tmp=0.0;
      if(isFluxBoundary==0 && isDOFBoundary==1)
        {
          for(int I=0;I<nSpace;I++)
            tmp -= a*grad_v[I]*n[I];
          tmp +=a*penalty*v;
        }
      return tmp;
    }

    inline void calculateElementResidual(//element
                                         double* mesh_trial_ref,
                                         double* mesh_grad_trial_ref,
                                         double* mesh_dof,
                                         proteus_index_t* mesh_l2g,
                                         double* dV_ref,
                                         double* u_trial_ref,
                                         double* u_grad_trial_ref,
                                         double* u_test_ref,
                                         double* u_grad_test_ref,
                                         //element boundary
                                         double* mesh_trial_trace_ref,
                                         double* mesh_grad_trial_trace_ref,
                                         double* dS_ref,
                                         double* u_trial_trace_ref,
                                         double* u_grad_trial_trace_ref,
                                         double* u_test_trace_ref,
                                         double* u_grad_test_trace_ref,
                                         double* normal_ref,
                                         double* boundaryJac_ref,
                                         //physics
                                         proteus_index_t nElements_global,
                                         proteus_index_t* u_l2g,
                                         double* u_dof,
                                         double alphaBDF,
                                         double* q_vf,
                                         double* q_divU,
                                         double* q_vs,
                                         double* q_vos,
                                         double rho_s,
                                         double* q_rho_f,
                                         double rho_s_min,
                                         double rho_f_min,
                                         double* ebqe_vf,
                                         double* ebqe_vs,
                                         double* ebqe_vos,
                                         double* ebqe_rho_f,
                                         double* q_u,
                                         double* q_grad_u,
                                         double* ebqe_u,
                                         double* ebqe_grad_u,
					 proteus_index_t offset_u,
                                         proteus_index_t stride_u, 
					 double* elementResidual_u,			   
					 proteus_index_t nExteriorElementBoundaries_global,
					 proteus_index_t* exteriorElementBoundariesArray,
					 proteus_index_t* elementBoundaryElementsArray,
					 int* elementBoundaryLocalElementBoundariesArray,
					 double* element_u,
					 proteus_index_t eN, 
					 double compatibility_condition, 
					 int INTEGRATE_BY_PARTS_DIV_U,
					 double* q_a)
    {
      for (int i=0;i<nDOF_test_element;i++)
        {
          elementResidual_u[i]=0.0;
        }//i
      //loop over quadrature points and compute integrands
      for  (int k=0;k<nQuadraturePoints_element;k++)
        {
          //compute indeces and declare local storage
          proteus_index_t eN_k = eN*nQuadraturePoints_element+k,
            eN_k_nSpace = eN_k*nSpace;
            //eN_nDOF_trial_element = eN*nDOF_trial_element;
          double u=0.0,grad_u[nSpace],
            a=0.0,
            f[nSpace],
            jac[nSpace*nSpace],
            jacDet,
            jacInv[nSpace*nSpace],
            u_grad_trial[nDOF_trial_element*nSpace],
            u_test_dV[nDOF_trial_element],
            u_grad_test_dV[nDOF_test_element*nSpace],
            dV,x,y,z,
            G[nSpace*nSpace],G_dd_G,tr_G;
          //
          //compute solution and gradients at quadrature points
          //
          ck.calculateMapping_element(eN,
                                      k,
                                      mesh_dof,
                                      mesh_l2g,
                                      mesh_trial_ref,
                                      mesh_grad_trial_ref,
                                      jac,
                                      jacDet,
                                      jacInv,
                                      x,y,z);
          //get the physical integration weight
          dV = fabs(jacDet)*dV_ref[k];
          ck.calculateG(jacInv,G,G_dd_G,tr_G);
          //get the trial function gradients
          ck.gradTrialFromRef(&u_grad_trial_ref[k*nDOF_trial_element*nSpace],jacInv,u_grad_trial);
          //get the solution
          ck.valFromElementDOF(element_u,&u_trial_ref[k*nDOF_trial_element],u);
          //get the solution gradients
          ck.gradFromElementDOF(element_u,u_grad_trial,grad_u);
          //precalculate test function products with integration weights
          for (int j=0;j<nDOF_trial_element;j++)
            {
              u_test_dV[j] = u_test_ref[k*nDOF_trial_element+j]*dV;
              for (int I=0;I<nSpace;I++)
                {
                  u_grad_test_dV[j*nSpace+I]   = u_grad_trial[j*nSpace+I]*dV;//cek warning won't work for Petrov-Galerkin
                }
            }
          //
          //calculate pde coefficients at quadrature points
          //
          evaluateCoefficients(alphaBDF,
                               &q_vf[eN_k_nSpace],
			       &q_vs[eN_k_nSpace],
			       q_vos[eN_k],
			       rho_s_min,
			       rho_f_min,
			       f,
			       a);
	  // 
	  //update element residual 
	  // 
          for(int i=0;i<nDOF_test_element;i++) 
            { 
              //int eN_k_i=eN_k*nDOF_test_element+i;
              //int eN_k_i_nSpace = eN_k_i*nSpace;
              int  i_nSpace=i*nSpace;
              elementResidual_u[i] += 
            (INTEGRATE_BY_PARTS_DIV_U == 1 ? ck.Advection_weak(f,&u_grad_test_dV[i_nSpace]) : q_divU[eN_k]*u_test_dV[i]) 
            //+ compatibility_condition*u_test_dV[i] // mql: to make the system solvable if int(div(u))!=0
            + ck.NumericalDiffusion(a,grad_u,&u_grad_test_dV[i_nSpace]);
            }//i
          //
          //save momentum for time history and velocity for subgrid error
          //save solution for other models
          //
          q_a[eN_k] = a;
          q_u[eN_k] = u;

          for (int I=0;I<nSpace;I++)
            q_grad_u[eN_k_nSpace+I] = grad_u[I];
        }
    }

    void calculateResidual(arguments_dict& args)
    {
        proteus::pyarray<double>& mesh_trial_ref = args.array<double>("mesh_trial_ref");
        proteus::pyarray<double>& mesh_grad_trial_ref = args.array<double>("mesh_grad_trial_ref");
        proteus::pyarray<double>& mesh_dof = args.array<double>("mesh_dof");
        proteus::pyarray<proteus_index_t>& mesh_l2g = args.array<proteus_index_t>("mesh_l2g");
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
        proteus_index_t nElements_global = args.scalar<proteus_index_t>("nElements_global");
        proteus::pyarray<int>& isDOFBoundary = args.array<int>("isDOFBoundary");
        proteus::pyarray<int>& isFluxBoundary = args.array<int>("isFluxBoundary");
        proteus::pyarray<proteus_index_t>& u_l2g = args.array<proteus_index_t>("u_l2g");
        proteus::pyarray<double>& u_dof = args.array<double>("u_dof");
        double alphaBDF = args.scalar<double>("alphaBDF");
        proteus::pyarray<double>& q_vf = args.array<double>("q_vf");
        proteus::pyarray<double>& q_divU = args.array<double>("q_divU");
        proteus::pyarray<double>& q_vs = args.array<double>("q_vs");
        proteus::pyarray<double>& q_vos = args.array<double>("q_vos");
        double rho_s = args.scalar<double>("rho_s");
        proteus::pyarray<double>& q_rho_f = args.array<double>("q_rho_f");
        double rho_s_min = args.scalar<double>("rho_s_min");
        double rho_f_min = args.scalar<double>("rho_f_min");
        proteus::pyarray<double>& ebqe_vf = args.array<double>("ebqe_vf");
        proteus::pyarray<double>& ebqe_vs = args.array<double>("ebqe_vs");
        proteus::pyarray<double>& ebqe_vos = args.array<double>("ebqe_vos");
        proteus::pyarray<double>& ebqe_rho_f = args.array<double>("ebqe_rho_f");
        proteus::pyarray<double>& q_u = args.array<double>("q_u");
        proteus::pyarray<double>& q_grad_u = args.array<double>("q_grad_u");
        proteus::pyarray<double>& ebqe_u = args.array<double>("ebqe_u");
        proteus::pyarray<double>& ebqe_grad_u = args.array<double>("ebqe_grad_u");
        proteus::pyarray<double>& ebqe_bc_u_ext = args.array<double>("ebqe_bc_u_ext");
        proteus::pyarray<double>& ebqe_adv_flux = args.array<double>("ebqe_adv_flux");
        proteus::pyarray<double>& ebqe_diff_flux = args.array<double>("ebqe_diff_flux");
        proteus::pyarray<double>& bc_adv_flux = args.array<double>("bc_adv_flux");
        proteus::pyarray<double>& bc_diff_flux = args.array<double>("bc_diff_flux");
        proteus_index_t offset_u = args.scalar<proteus_index_t>("offset_u");
        proteus_index_t stride_u = args.scalar<proteus_index_t>("stride_u");
        proteus::pyarray<double>& globalResidual = args.array<double>("globalResidual");
        proteus_index_t nExteriorElementBoundaries_global = args.scalar<proteus_index_t>("nExteriorElementBoundaries_global");
        proteus::pyarray<proteus_index_t>& exteriorElementBoundariesArray = args.array<proteus_index_t>("exteriorElementBoundariesArray");
        proteus::pyarray<proteus_index_t>& elementBoundaryElementsArray = args.array<proteus_index_t>("elementBoundaryElementsArray");
        proteus::pyarray<int>& elementBoundaryLocalElementBoundariesArray = args.array<int>("elementBoundaryLocalElementBoundariesArray");
        int INTEGRATE_BY_PARTS_DIV_U = args.scalar<int>("INTEGRATE_BY_PARTS_DIV_U");
        proteus::pyarray<double>& q_a = args.array<double>("q_a");
        proteus::pyarray<double>& ebqe_a = args.array<double>("ebqe_a");
      double compatibility_condition=0.;
      /* // COMPUTE COMPATIBILITY CONSTANT  */
      /* // mql: Modify the rhs (by adding a constant) so that the Poission system is solvable (assume diffusive flux = 0). */
      /* // Note that this is equivalent to consider a (not known) diffusive flux != 0 s.t. the system is solvable. */
      /* for(proteus_index_t eN=0;eN<nElements_global;eN++) */
      /* 	{ */
      /* 	  for  (int k=0;k<nQuadraturePoints_element;k++) */
      /* 	    { */
      /* 	      proteus_index_t eN_k = eN*nQuadraturePoints_element+k; */
      /* 	      double  */
      /* 		jac[nSpace*nSpace], */
      /* 		jacDet, */
      /* 		jacInv[nSpace*nSpace], */
      /* 		dV,x,y,z; */
      /* 	      ck.calculateMapping_element(eN, */
      /* 					  k, */
      /* 					  mesh_dof.data(), */
      /* 					  mesh_l2g.data(), */
      /* 					  mesh_trial_ref.data(), */
      /* 					  mesh_grad_trial_ref.data(), */
      /* 					  jac, */
      /* 					  jacDet, */
      /* 					  jacInv, */
      /* 					  x,y,z); */
      /* 	      //get the physical integration weight */
      /* 	      dV = fabs(jacDet)*dV_ref.data()[k]; */
      /* 	      compatibility_condition -= q_divU.data()[eN_k]*dV; */
      /* 	    } */
      /* 	} */
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
      for(proteus_index_t eN=0;eN<nElements_global;eN++)
        {
          //declare local storage for element residual and initialize
          double elementResidual_u[nDOF_test_element],element_u[nDOF_trial_element];
          for (int i=0;i<nDOF_test_element;i++)
            {
              proteus_index_t eN_i=eN*nDOF_test_element+i;
              element_u[i] = u_dof.data()[u_l2g.data()[eN_i]];
            }//i
          calculateElementResidual(mesh_trial_ref.data(),
                                   mesh_grad_trial_ref.data(),
                                   mesh_dof.data(),
                                   mesh_l2g.data(),
                                   dV_ref.data(),
                                   u_trial_ref.data(),
                                   u_grad_trial_ref.data(),
                                   u_test_ref.data(),
                                   u_grad_test_ref.data(),
                                   mesh_trial_trace_ref.data(),
                                   mesh_grad_trial_trace_ref.data(),
                                   dS_ref.data(),
                                   u_trial_trace_ref.data(),
                                   u_grad_trial_trace_ref.data(),
                                   u_test_trace_ref.data(),
                                   u_grad_test_trace_ref.data(),
                                   normal_ref.data(),
                                   boundaryJac_ref.data(),
                                   nElements_global,
                                   u_l2g.data(),
                                   u_dof.data(),
                                   alphaBDF,
                                   q_vf.data(),
                                   q_divU.data(),
                                   q_vs.data(),
                                   q_vos.data(),
                                   rho_s,
                                   q_rho_f.data(),
                                   rho_s_min,
                                   rho_f_min,
                                   ebqe_vf.data(),
                                   ebqe_vs.data(),
                                   ebqe_vos.data(),
                                   ebqe_rho_f.data(),
                                   q_u.data(),
                                   q_grad_u.data(),
                                   ebqe_u.data(),
                                   ebqe_grad_u.data(),
				   offset_u,
                                   stride_u, 
				   elementResidual_u,
				   nExteriorElementBoundaries_global,
				   exteriorElementBoundariesArray.data(),
				   elementBoundaryElementsArray.data(),
				   elementBoundaryLocalElementBoundariesArray.data(),
				   element_u,
				   eN, 
				   compatibility_condition, 
				   INTEGRATE_BY_PARTS_DIV_U,
				   q_a.data());
	  //
	  //load element into global residual and save element residual
	  //
	  for(int i=0;i<nDOF_test_element;i++) 
	    { 
	      proteus_index_t eN_i=eN*nDOF_test_element+i;
	      globalResidual.data()[offset_u+stride_u*u_l2g.data()[eN_i]]+=elementResidual_u[i];
	    }//i
	}//elements
      //
      //loop over exterior element boundaries to calculate levelset gradient
      //
      //ebNE is the Exterior element boundary INdex
      //ebN is the element boundary INdex
      //eN is the element index
      for (proteus_index_t ebNE = 0; ebNE < nExteriorElementBoundaries_global; ebNE++)
     	{
      	  proteus_index_t ebN = exteriorElementBoundariesArray.data()[ebNE],
      	    eN  = elementBoundaryElementsArray.data()[ebN*2+0];
      	  int ebN_local = elementBoundaryLocalElementBoundariesArray.data()[ebN*2+0];
      	    //eN_nDOF_trial_element = eN*nDOF_trial_element;
      	  double elementResidual_u[nDOF_test_element];
      	  double element_u[nDOF_trial_element];
      	  for (int i=0;i<nDOF_test_element;i++)
      	    {
      	      proteus_index_t eN_i=eN*nDOF_test_element+i;
      	      element_u[i] = u_dof.data()[u_l2g.data()[eN_i]];
              elementResidual_u[i] = 0.0;
            }//i
          for  (int kb=0;kb<nQuadraturePoints_elementBoundary;kb++)
            {
              proteus_index_t ebNE_kb = ebNE*nQuadraturePoints_elementBoundary+kb,
                ebNE_kb_nSpace = ebNE_kb*nSpace;
              int ebN_local_kb = ebN_local*nQuadraturePoints_elementBoundary+kb,
                ebN_local_kb_nSpace = ebN_local_kb*nSpace;
              double h_b=0.0,
                penalty=0.0,
                u_ext=0.0,
                bc_u_ext=0.0,
                adv_flux_ext=0.0,
                diff_flux_ext=0.0,
                a_ext=0.0,
                f_ext[nSpace],
                grad_u_ext[nSpace],
                jac_ext[nSpace*nSpace],
                jacDet_ext,
                jacInv_ext[nSpace*nSpace],
                boundaryJac[nSpace*(nSpace-1)],
                metricTensor[(nSpace-1)*(nSpace-1)],
                metricTensorDetSqrt,
                dS,
                u_test_dS[nDOF_test_element],
                u_grad_trial_trace[nDOF_trial_element*nSpace],
                u_grad_test_dS[nDOF_test_element*nSpace],
                normal[nSpace],x_ext,y_ext,z_ext,
                G[nSpace*nSpace],G_dd_G,tr_G;
              //
              //calculate the solution and gradients at quadrature points
              //
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
              ck.calculateGScale(G,normal,h_b);
              penalty = 10.0/h_b;
              //compute shape and solution information
              //shape
              ck.gradTrialFromRef(&u_grad_trial_trace_ref.data()[ebN_local_kb_nSpace*nDOF_trial_element],jacInv_ext,u_grad_trial_trace);
              //solution and gradients
              ck.valFromElementDOF(element_u,&u_trial_trace_ref.data()[ebN_local_kb*nDOF_test_element],u_ext);
              ck.gradFromElementDOF(element_u,u_grad_trial_trace,grad_u_ext);
              //precalculate test function products with integration weights
              for (int j=0;j<nDOF_trial_element;j++)
                {
                  u_test_dS[j] = u_test_trace_ref.data()[ebN_local_kb*nDOF_test_element+j]*dS;
                  for (int I=0;I<nSpace;I++)
                    u_grad_test_dS[j*nSpace+I] = u_grad_trial_trace[j*nSpace+I]*dS;//cek hack, using trial
                }
              //
              //load the boundary values
              //
              bc_u_ext = isDOFBoundary.data()[ebNE_kb]*ebqe_bc_u_ext.data()[ebNE_kb]+(1-isDOFBoundary.data()[ebNE_kb])*u_ext;
              //
              //calculate the pde coefficients using the solution and the boundary values for the solution
              //

              evaluateCoefficients(alphaBDF,
                                   &ebqe_vf.data()[ebNE_kb_nSpace],
                                   &ebqe_vs.data()[ebNE_kb_nSpace],
                                   ebqe_vos.data()[ebNE_kb],
                                   rho_s_min,
                                   rho_f_min,
                                   f_ext,
                                   a_ext);
              ebqe_u.data()[ebNE_kb] = u_ext;
              for (int I=0;I<nSpace;I++)
                ebqe_grad_u.data()[ebNE_kb_nSpace+I] = grad_u_ext[I];
              //
              //calculate the numerical fluxes
              //
              exteriorNumericalAdvectiveFlux(isFluxBoundary.data()[ebNE_kb],
                                             bc_adv_flux.data()[ebNE_kb],
                                             normal,
                                             f_ext,
                                             adv_flux_ext); //=f.normal = [(1-vos)*vf + vos*vs].normal
              exteriorNumericalDiffusiveFlux(isDOFBoundary.data()[ebNE_kb],
                                             isFluxBoundary.data()[ebNE_kb],
                                             normal,
                                             a_ext,
                                             grad_u_ext,
                                             u_ext,
                                             bc_u_ext,
                                             bc_diff_flux.data()[ebNE_kb],
                                             penalty,
                                             diff_flux_ext);
              if (isDOFBoundary.data()[ebNE_kb] != 1)
                diff_flux_ext = 0.0; // mql: don't consider diffusive flux unless Dirichlet BC
              //if(isFluxBoundary.data()[ebNE_kb] == 1)
              //{
              //  adv_flux_ext = 0.0;
              //  diff_flux_ext = 0.0;
              //}
	          ebqe_a.data()[ebNE_kb] = a_ext;
              ebqe_adv_flux.data()[ebNE_kb] = adv_flux_ext;
              ebqe_diff_flux.data()[ebNE_kb] = diff_flux_ext;
              //
              //update residuals
              //
              for (int i=0;i<nDOF_test_element;i++)
                {
                  elementResidual_u[i] +=
                    (INTEGRATE_BY_PARTS_DIV_U == 1 ? ck.ExteriorElementBoundaryFlux(adv_flux_ext,u_test_dS[i]) : 0.)
                    + ck.ExteriorElementBoundaryFlux(diff_flux_ext,u_test_dS[i]) // mql: just != 0 if Dirichlet BC
                    + ck.ExteriorElementBoundaryScalarDiffusionAdjoint(isDOFBoundary.data()[ebNE_kb],
                                                                       0, // mql: if Dirichlet BCs, then the flux BCs don't matter
                                                                       //isFluxBoundary.data()[ebNE_kb],
                                                                       1.0,
                                                                       u_ext,
                                                                       bc_u_ext,
                                                                       normal,
                                                                       a_ext,
                                                                       &u_grad_test_dS[i*nSpace]);
                }//i
            }//kb
          //
          //update the element and global residual storage
          //
          for (int i=0;i<nDOF_test_element;i++)
            {
              proteus_index_t eN_i = eN*nDOF_test_element+i;
              globalResidual.data()[offset_u+stride_u*u_l2g.data()[eN_i]] += elementResidual_u[i];
            }//i
        }//ebNE

    }

    inline void calculateElementJacobian(//element
                                         double* mesh_trial_ref,
                                         double* mesh_grad_trial_ref,
                                         double* mesh_dof,
                                         proteus_index_t* mesh_l2g,
                                         double* dV_ref,
                                         double* u_trial_ref,
                                         double* u_grad_trial_ref,
                                         double* u_test_ref,
                                         double* u_grad_test_ref,
                                         //element boundary
                                         double* mesh_trial_trace_ref,
                                         double* mesh_grad_trial_trace_ref,
                                         double* dS_ref,
                                         double* u_trial_trace_ref,
                                         double* u_grad_trial_trace_ref,
                                         double* u_test_trace_ref,
                                         double* u_grad_test_trace_ref,
                                         double* normal_ref,
                                         double* boundaryJac_ref,
                                         //physics
                                         proteus_index_t nElements_global,
                                         proteus_index_t* u_l2g,
                                         double* u_dof,
                                         double alphaBDF,
                                         double* q_vf,
                                         double* q_vs,
                                         double* q_vos,
                                         double rho_s,
                                         double* q_rho_f,
                                         double rho_s_min,
                                         double rho_f_min,
                                         double* elementJacobian_u_u,
                                         double* element_u,
                                         proteus_index_t eN)
    {
      for (int i=0;i<nDOF_test_element;i++)
        for (int j=0;j<nDOF_trial_element;j++)
          {
            elementJacobian_u_u[i*nDOF_trial_element+j]=0.0;
          }
      for  (int k=0;k<nQuadraturePoints_element;k++)
        {
          proteus_index_t eN_k = eN*nQuadraturePoints_element+k, //index to a scalar at a quadrature point
            eN_k_nSpace = eN_k*nSpace;
            //eN_nDOF_trial_element = eN*nDOF_trial_element; //index to a vector at a quadrature point

          //declare local storage
          double u=0.0,
            grad_u[nSpace],
            f[nSpace],
            a=0.0,
            jac[nSpace*nSpace],
            jacDet,
            jacInv[nSpace*nSpace],
            u_grad_trial[nDOF_trial_element*nSpace],
            dV,
            u_grad_test_dV[nDOF_test_element*nSpace],
            x,y,z,
            G[nSpace*nSpace],G_dd_G,tr_G;
          //
          //calculate solution and gradients at quadrature points
          //
          ck.calculateMapping_element(eN,
                                      k,
                                      mesh_dof,
                                      mesh_l2g,
                                      mesh_trial_ref,
                                      mesh_grad_trial_ref,
                                      jac,
                                      jacDet,
                                      jacInv,
                                      x,y,z);
          //get the physical integration weight
          dV = fabs(jacDet)*dV_ref[k];
          ck.calculateG(jacInv,G,G_dd_G,tr_G);
          //get the trial function gradients
          ck.gradTrialFromRef(&u_grad_trial_ref[k*nDOF_trial_element*nSpace],jacInv,u_grad_trial);
          //get the solution
          ck.valFromElementDOF(element_u,&u_trial_ref[k*nDOF_trial_element],u);
          //get the solution gradients
          ck.gradFromElementDOF(element_u,u_grad_trial,grad_u);
          //precalculate test function products with integration weights
          for (int j=0;j<nDOF_trial_element;j++)
            {
              for (int I=0;I<nSpace;I++)
                {
                  u_grad_test_dV[j*nSpace+I]   = u_grad_trial[j*nSpace+I]*dV;//cek warning won't work for Petrov-Galerkin
                }
            }
          //
          //calculate pde coefficients and derivatives at quadrature points
          //
          evaluateCoefficients(alphaBDF,
                               &q_vf[eN_k_nSpace],
                               &q_vs[eN_k_nSpace],
                               q_vos[eN_k],
                               rho_s_min,
                               rho_f_min,
                               f,
                               a);
          for(int i=0;i<nDOF_test_element;i++)
            {
              //int eN_k_i=eN_k*nDOF_test_element+i;
              //int eN_k_i_nSpace=eN_k_i*nSpace;
              int i_nSpace=i*nSpace;
              for(int j=0;j<nDOF_trial_element;j++)
                {
                  //int eN_k_j=eN_k*nDOF_trial_element+j;
                  //int eN_k_j_nSpace = eN_k_j*nSpace;
                  int j_nSpace = j*nSpace;
                  elementJacobian_u_u[i*nDOF_trial_element+j] +=
                    ck.NumericalDiffusionJacobian(a,&u_grad_trial[j_nSpace],&u_grad_test_dV[i_nSpace]);
                }//j
            }//i
        }//k
    }

    void calculateJacobian(arguments_dict& args)
    {
        proteus::pyarray<double>& mesh_trial_ref = args.array<double>("mesh_trial_ref");
        proteus::pyarray<double>& mesh_grad_trial_ref = args.array<double>("mesh_grad_trial_ref");
        proteus::pyarray<double>& mesh_dof = args.array<double>("mesh_dof");
        proteus::pyarray<proteus_index_t>& mesh_l2g = args.array<proteus_index_t>("mesh_l2g");
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
        proteus_index_t nElements_global = args.scalar<proteus_index_t>("nElements_global");
        proteus::pyarray<int>& isDOFBoundary = args.array<int>("isDOFBoundary");
        proteus::pyarray<int>& isFluxBoundary = args.array<int>("isFluxBoundary");
        proteus::pyarray<proteus_index_t>& u_l2g = args.array<proteus_index_t>("u_l2g");
        proteus::pyarray<double>& u_dof = args.array<double>("u_dof");
        double alphaBDF = args.scalar<double>("alphaBDF");
        proteus::pyarray<double>& q_vf = args.array<double>("q_vf");
        proteus::pyarray<double>& q_vs = args.array<double>("q_vs");
        proteus::pyarray<double>& q_vos = args.array<double>("q_vos");
        double rho_s = args.scalar<double>("rho_s");
        proteus::pyarray<double>& q_rho_f = args.array<double>("q_rho_f");
        double rho_s_min = args.scalar<double>("rho_s_min");
        double rho_f_min = args.scalar<double>("rho_f_min");
        proteus::pyarray<double>& ebqe_vf = args.array<double>("ebqe_vf");
        proteus::pyarray<double>& ebqe_vs = args.array<double>("ebqe_vs");
        proteus::pyarray<double>& ebqe_vos = args.array<double>("ebqe_vos");
        proteus::pyarray<double>& ebqe_rho_f = args.array<double>("ebqe_rho_f");
        proteus::pyarray<proteus_csr_index_t>& csrRowIndeces_u_u = args.array<proteus_csr_index_t>("csrRowIndeces_u_u");
        proteus::pyarray<proteus_csr_index_t>& csrColumnOffsets_u_u = args.array<proteus_csr_index_t>("csrColumnOffsets_u_u");
        proteus::pyarray<double>& globalJacobian = args.array<double>("globalJacobian");
        proteus_index_t nExteriorElementBoundaries_global = args.scalar<proteus_index_t>("nExteriorElementBoundaries_global");
        proteus::pyarray<proteus_index_t>& exteriorElementBoundariesArray = args.array<proteus_index_t>("exteriorElementBoundariesArray");
        proteus::pyarray<proteus_index_t>& elementBoundaryElementsArray = args.array<proteus_index_t>("elementBoundaryElementsArray");
        proteus::pyarray<int>& elementBoundaryLocalElementBoundariesArray = args.array<int>("elementBoundaryLocalElementBoundariesArray");
        proteus::pyarray<proteus_csr_index_t>& csrColumnOffsets_eb_u_u = args.array<proteus_csr_index_t>("csrColumnOffsets_eb_u_u");
      //
      //loop over elements to compute volume integrals and load them into the element Jacobians and global Jacobian
      //
      for(proteus_index_t eN=0;eN<nElements_global;eN++)
        {
          double  elementJacobian_u_u[nDOF_test_element*nDOF_trial_element],element_u[nDOF_trial_element];
          for (int j=0;j<nDOF_trial_element;j++)
            {
              proteus_index_t eN_j = eN*nDOF_trial_element+j;
              element_u[j] = u_dof.data()[u_l2g.data()[eN_j]];
            }
          calculateElementJacobian(mesh_trial_ref.data(),
                                   mesh_grad_trial_ref.data(),
                                   mesh_dof.data(),
                                   mesh_l2g.data(),
                                   dV_ref.data(),
                                   u_trial_ref.data(),
                                   u_grad_trial_ref.data(),
                                   u_test_ref.data(),
                                   u_grad_test_ref.data(),
                                   mesh_trial_trace_ref.data(),
                                   mesh_grad_trial_trace_ref.data(),
                                   dS_ref.data(),
                                   u_trial_trace_ref.data(),
                                   u_grad_trial_trace_ref.data(),
                                   u_test_trace_ref.data(),
                                   u_grad_test_trace_ref.data(),
                                   normal_ref.data(),
                                   boundaryJac_ref.data(),
                                   nElements_global,
                                   u_l2g.data(),
                                   u_dof.data(),
                                   alphaBDF,
                                   q_vf.data(),
                                   q_vs.data(),
                                   q_vos.data(),
                                   rho_s,
                                   q_rho_f.data(),
                                   rho_s_min,
                                   rho_f_min,
                                   elementJacobian_u_u,
                                   element_u,
                                   eN);
          //
          //load into element Jacobian into global Jacobian
          //
          for (int i=0;i<nDOF_test_element;i++)
            {
              proteus_index_t eN_i = eN*nDOF_test_element+i;
              for (int j=0;j<nDOF_trial_element;j++)
                {
                  proteus_index_t eN_i_j = eN_i*nDOF_trial_element+j;
                  globalJacobian.data()[csrRowIndeces_u_u.data()[eN_i] + csrColumnOffsets_u_u.data()[eN_i_j]] += elementJacobian_u_u[i*nDOF_trial_element+j];
                }//j
            }//i
        }//elements
      //
      //loop over exterior element boundaries to compute the surface integrals and load them into the global Jacobian
      //
      for (proteus_index_t ebNE = 0; ebNE < nExteriorElementBoundaries_global; ebNE++)
      	{
      	  proteus_index_t ebN = exteriorElementBoundariesArray.data()[ebNE];
      	  proteus_index_t eN  = elementBoundaryElementsArray.data()[ebN*2+0];
      	  int ebN_local = elementBoundaryLocalElementBoundariesArray.data()[ebN*2+0];
      	  proteus_index_t eN_nDOF_trial_element = eN*nDOF_trial_element;
      	  for  (int kb=0;kb<nQuadraturePoints_elementBoundary;kb++)
      	    {
      	      proteus_index_t ebNE_kb = ebNE*nQuadraturePoints_elementBoundary+kb,
      		    ebNE_kb_nSpace = ebNE_kb*nSpace;
      	      int ebN_local_kb = ebN_local*nQuadraturePoints_elementBoundary+kb,
      		    ebN_local_kb_nSpace = ebN_local_kb*nSpace;
              double h_b=0.0,
                u_ext=0.0,
                grad_u_ext[nSpace],
                a_ext=0.0,
                f_ext[nSpace],
                jac_ext[nSpace*nSpace],
                jacDet_ext,
                jacInv_ext[nSpace*nSpace],
                boundaryJac[nSpace*(nSpace-1)],
                metricTensor[(nSpace-1)*(nSpace-1)],
                metricTensorDetSqrt,
                dS,
                u_test_dS[nDOF_test_element],
                u_grad_trial_trace[nDOF_trial_element*nSpace],
                u_grad_test_dS[nDOF_test_element*nSpace],
                normal[nSpace],x_ext,y_ext,z_ext,
                penalty=0.0,
                //
                G[nSpace*nSpace],G_dd_G,tr_G;
              //
              //calculate the solution and gradients at quadrature points
              //
              u_ext=0.0;
              for (int I=0;I<nSpace;I++)
                {
                  grad_u_ext[I] = 0.0;
                }
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
              ck.calculateG(jacInv_ext,G,G_dd_G,tr_G);
              ck.calculateGScale(G,normal,h_b);
              penalty=10.0/h_b;
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
                  for (int I=0;I<nSpace;I++)
                    u_grad_test_dS[j*nSpace+I] = u_grad_trial_trace[j*nSpace+I]*dS;//cek hack, using trial
                }
              //
              //calculate the internal and external trace of the pde coefficients
              //
              evaluateCoefficients(alphaBDF,
                                   &ebqe_vf.data()[ebNE_kb_nSpace],
                                   &ebqe_vs.data()[ebNE_kb_nSpace],
                                   ebqe_vos.data()[ebNE_kb],
                                   rho_s_min,
                                   rho_f_min,
                                   f_ext,
                                   a_ext);
      	      //
      	      //update the global Jacobian from the flux Jacobian
      	      //
      	      for (int i=0;i<nDOF_test_element;i++)
      		{
      		  proteus_index_t eN_i = eN*nDOF_test_element+i;
      		  //int ebNE_kb_i = ebNE_kb*nDOF_test_element+i;
      		  for (int j=0;j<nDOF_trial_element;j++)
      		    {
      		      proteus_index_t ebN_i_j = ebN*4*nDOF_test_X_trial_element + i*nDOF_trial_element + j;
      		      int ebN_local_kb_j=ebN_local_kb*nDOF_trial_element+j,
                        j_nSpace = j*nSpace;

      		      globalJacobian.data()[csrRowIndeces_u_u.data()[eN_i] + csrColumnOffsets_eb_u_u.data()[ebN_i_j]] +=
      			ExteriorNumericalDiffusiveFluxJacobian(isDOFBoundary.data()[ebNE_kb],
      							       isFluxBoundary.data()[ebNE_kb],
      							       normal,
      							       a_ext,
      							       u_trial_trace_ref.data()[ebN_local_kb_j],
      							       &u_grad_trial_trace[j_nSpace],
      							       penalty)*u_test_dS[i]
                        +
      		        ck.ExteriorElementBoundaryScalarDiffusionAdjointJacobian
      			(isDOFBoundary.data()[ebNE_kb],
      			 isFluxBoundary.data()[ebNE_kb],
      			 1.0,
      			 u_trial_trace_ref.data()[ebN_local_kb_j],
      			 normal,
      			 a_ext,
      			 &u_grad_test_dS[i*nSpace]);
      		    }//j
      		}//i
      	    }//kb
      	}//ebNE
    }//computeJacobian
  };//cppPresInc

  inline cppPresInc_base* newPresInc(int nSpaceIn,
                                 int nQuadraturePoints_elementIn,
                                 int nDOF_mesh_trial_elementIn,
                                 int nDOF_trial_elementIn,
                                 int nDOF_test_elementIn,
                                 int nQuadraturePoints_elementBoundaryIn,
                                 int CompKernelFlag)
  {
    if (nSpaceIn == 2)
      return proteus::chooseAndAllocateDiscretization2D<cppPresInc_base,cppPresInc,CompKernel>(nSpaceIn,
                                                                                               nQuadraturePoints_elementIn,
                                                                                               nDOF_mesh_trial_elementIn,
                                                                                               nDOF_trial_elementIn,
                                                                                               nDOF_test_elementIn,
                                                                                               nQuadraturePoints_elementBoundaryIn,
                                                                                               CompKernelFlag);
    else
      return proteus::chooseAndAllocateDiscretization<cppPresInc_base,cppPresInc,CompKernel>(nSpaceIn,
                                                                                             nQuadraturePoints_elementIn,
                                                                                             nDOF_mesh_trial_elementIn,
                                                                                             nDOF_trial_elementIn,
                                                                                             nDOF_test_elementIn,
                                                                                             nQuadraturePoints_elementBoundaryIn,
                                                                                             CompKernelFlag);
  }
}//proteus
#endif
