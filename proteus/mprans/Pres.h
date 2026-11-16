#ifndef PRES_H
#define PRES_H
#include <cmath>
#include <iostream>
#include "CompKernel.h"
#include "ModelFactory.h"
#include "ArgumentsDict.h"
#include "pyarray.h"

namespace py = pybind11;

namespace proteus
{
  class cppPres_base
  {
  public:
    virtual ~cppPres_base(){}
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
  class cppPres : public cppPres_base
  {
  public:
    CompKernelType ck;
    cppPres():ck()
    {}
    inline
      void evaluateCoefficients(const double& p,
                                const double& p_last,
                                const double& p_inc,
                                const double massFlux[nSpace],
                                double f[nSpace],
                                double& r)
    {
      for (int I=0;I<nSpace;I++)
        f[I] = massFlux[I];
      r = p - p_last - p_inc;
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
                                         double* q_u,
                                         double* q_grad_u,
                                         double* q_p_last,
                                         double* q_p_inc,
                                         double* q_massFlux,
                                         double* ebqe_massFlux,
                                         double* ebqe_u,
                                         double* ebqe_grad_u,
                                         proteus_index_t offset_u, proteus_index_t stride_u,
                                         double* elementResidual_u,
                                         proteus_index_t nExteriorElementBoundaries_global,
                                         proteus_index_t* exteriorElementBoundariesArray,
                                         proteus_index_t* elementBoundaryElementsArray,
                                         int* elementBoundaryLocalElementBoundariesArray,
                                         double* element_u,
                                         proteus_index_t eN)
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
            f[nSpace],
            r=0.0,
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
          evaluateCoefficients(u,//u is p
                               q_p_last[eN_k],
                               q_p_inc[eN_k],
                               &q_massFlux[eN_k_nSpace],
                               f,
                               r);
          //
          //update element residual
          //
          for(int i=0;i<nDOF_test_element;i++)
            {
              int  i_nSpace=i*nSpace;
              elementResidual_u[i] += ck.Advection_weak(f,&u_grad_test_dV[i_nSpace]) +
                ck.Reaction_weak(r,u_test_dV[i]);
            }//i
          q_u[eN_k] = u;
          for(int I=0;I<nSpace;I++)
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
        proteus::pyarray<proteus_index_t>& u_l2g = args.array<proteus_index_t>("u_l2g");
        proteus::pyarray<double>& u_dof = args.array<double>("u_dof");
        proteus::pyarray<double>& q_u = args.array<double>("q_u");
        proteus::pyarray<double>& q_grad_u = args.array<double>("q_grad_u");
        proteus::pyarray<double>& q_p_last = args.array<double>("q_p_last");
        proteus::pyarray<double>& q_p_inc = args.array<double>("q_p_inc");
        proteus::pyarray<double>& q_massFlux = args.array<double>("q_massFlux");
        proteus::pyarray<double>& ebqe_massFlux = args.array<double>("ebqe_massFlux");
        proteus::pyarray<double>& ebqe_u = args.array<double>("ebqe_u");
        proteus::pyarray<double>& ebqe_grad_u = args.array<double>("ebqe_grad_u");
        proteus_index_t offset_u = args.scalar<proteus_index_t>("offset_u");
        proteus_index_t stride_u = args.scalar<proteus_index_t>("stride_u");
        proteus::pyarray<double>& globalResidual = args.array<double>("globalResidual");
        proteus_index_t nExteriorElementBoundaries_global = args.scalar<proteus_index_t>("nExteriorElementBoundaries_global");
        proteus::pyarray<proteus_index_t>& exteriorElementBoundariesArray = args.array<proteus_index_t>("exteriorElementBoundariesArray");
        proteus::pyarray<proteus_index_t>& elementBoundaryElementsArray = args.array<proteus_index_t>("elementBoundaryElementsArray");
        proteus::pyarray<int>& elementBoundaryLocalElementBoundariesArray = args.array<int>("elementBoundaryLocalElementBoundariesArray");
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
                                   q_u.data(),
                                   q_grad_u.data(),
                                   q_p_last.data(),
                                   q_p_inc.data(),
                                   q_massFlux.data(),
                                   ebqe_massFlux.data(),
                                   ebqe_u.data(),
                                   ebqe_grad_u.data(),
                                   offset_u,stride_u,
                                   elementResidual_u,
                                   nExteriorElementBoundaries_global,
                                   exteriorElementBoundariesArray.data(),
                                   elementBoundaryElementsArray.data(),
                                   elementBoundaryLocalElementBoundariesArray.data(),
                                   element_u,
                                   eN);
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
              double u_ext=0.0,
                adv_flux_ext=0.0,
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
                }
              //
              //load the boundary values
              //
              ebqe_u.data()[ebNE_kb] = u_ext;
              for (int I=0;I<nSpace;I++)
                ebqe_grad_u.data()[ebNE_kb_nSpace+I] = grad_u_ext[I];
              adv_flux_ext=0.0;
              for (int I=0;I<nSpace;I++)
                adv_flux_ext += ebqe_massFlux.data()[ebNE_kb_nSpace+I]*normal[I];
              //
              //update residuals
              //
              for (int i=0;i<nDOF_test_element;i++)
                {
                  elementResidual_u[i] += ck.ExteriorElementBoundaryFlux(adv_flux_ext,u_test_dS[i]);
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
                                         double* elementJacobian_u_u,
                                         proteus_index_t eN)
    {
      for (int i=0;i<nDOF_test_element;i++)
        for (int j=0;j<nDOF_trial_element;j++)
          {
            elementJacobian_u_u[i*nDOF_trial_element+j]=0.0;
          }
      for  (int k=0;k<nQuadraturePoints_element;k++)
        {
          //declare local storage
          double jac[nSpace*nSpace],
            jacDet,
            jacInv[nSpace*nSpace],
            dV,
            u_test_dV[nDOF_test_element],
            x,y,z;
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
          for (int j=0;j<nDOF_trial_element;j++)
            {
              u_test_dV[j] = u_test_ref[k*nDOF_trial_element+j]*dV;
            }
          for(int i=0;i<nDOF_test_element;i++)
            {
              for(int j=0;j<nDOF_trial_element;j++)
                {
                  elementJacobian_u_u[i*nDOF_trial_element+j] += ck.ReactionJacobian_weak(1.0,
                                                                                          u_trial_ref[k*nDOF_trial_element+j],
                                                                                          u_test_dV[i]);
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
        proteus::pyarray<proteus_csr_index_t>& csrRowIndeces_u_u = args.array<proteus_csr_index_t>("csrRowIndeces_u_u");
        proteus::pyarray<proteus_csr_index_t>& csrColumnOffsets_u_u = args.array<proteus_csr_index_t>("csrColumnOffsets_u_u");
        proteus::pyarray<double>& globalJacobian = args.array<double>("globalJacobian");
      //
      //loop over elements to compute volume integrals and load them into the element Jacobians and global Jacobian
      //
      for(proteus_index_t eN=0;eN<nElements_global;eN++)
        {
          double  elementJacobian_u_u[nDOF_test_element*nDOF_trial_element];
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
                                   elementJacobian_u_u,
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
                  globalJacobian.data()[csrRowIndeces_u_u[eN_i] + csrColumnOffsets_u_u[eN_i_j]] += elementJacobian_u_u[i*nDOF_trial_element+j];
                }//j
            }//i
        }//elements
    }//computeJacobian
  };//cppPres

  inline cppPres_base* newPres(int nSpaceIn,
                                 int nQuadraturePoints_elementIn,
                                 int nDOF_mesh_trial_elementIn,
                                 int nDOF_trial_elementIn,
                                 int nDOF_test_elementIn,
                                 int nQuadraturePoints_elementBoundaryIn,
                                 int CompKernelFlag)
  {
    if (nSpaceIn == 2)
      return proteus::chooseAndAllocateDiscretization2D<cppPres_base,cppPres,CompKernel>(nSpaceIn,
                                                                                               nQuadraturePoints_elementIn,
                                                                                               nDOF_mesh_trial_elementIn,
                                                                                               nDOF_trial_elementIn,
                                                                                               nDOF_test_elementIn,
                                                                                               nQuadraturePoints_elementBoundaryIn,
                                                                                               CompKernelFlag);
    else
      return proteus::chooseAndAllocateDiscretization<cppPres_base,cppPres,CompKernel>(nSpaceIn,
                                                                                             nQuadraturePoints_elementIn,
                                                                                             nDOF_mesh_trial_elementIn,
                                                                                             nDOF_trial_elementIn,
                                                                                             nDOF_test_elementIn,
                                                                                             nQuadraturePoints_elementBoundaryIn,
                                                                                             CompKernelFlag);
  }
}//proteus
#endif
