#ifndef ADDEDMASS_H
#define ADDEDMASS_H
#include <cmath>
#include <iostream>
#include "CompKernel.h"
#include "ModelFactory.h"
#include "ArgumentsDict.h"

#include "pyarray.h"

using namespace std;

namespace proteus
{
  class cppAddedMass_base
  {
  public:
    virtual ~cppAddedMass_base(){}
    virtual void calculateResidual(arguments_dict& args) = 0;
    /*virtual void calculateResidual(//element
                                   proteus::pyarray<double>& mesh_trial_ref,
                                   proteus::pyarray<double>& mesh_grad_trial_ref,
                                   proteus::pyarray<double>& mesh_dof,
                                   proteus::pyarray<proteus_index_t>& mesh_l2g,
                                   proteus::pyarray<double>& dV_ref,
                                   proteus::pyarray<double>& u_trial_ref,
                                   proteus::pyarray<double>& u_grad_trial_ref,
                                   proteus::pyarray<double>& u_test_ref,
                                   proteus::pyarray<double>& u_grad_test_ref,
                                   //element boundary
                                   proteus::pyarray<double>& mesh_trial_trace_ref,
                                   proteus::pyarray<double>& mesh_grad_trial_trace_ref,
                                   proteus::pyarray<double>& dS_ref,
                                   proteus::pyarray<double>& u_trial_trace_ref,
                                   proteus::pyarray<double>& u_grad_trial_trace_ref,
                                   proteus::pyarray<double>& u_test_trace_ref,
                                   proteus::pyarray<double>& u_grad_test_trace_ref,
                                   proteus::pyarray<double>& normal_ref,
                                   proteus::pyarray<double>& boundaryJac_ref,
                                   //physics
                                   proteus_index_t nElements_global,
                                   proteus_index_t nElementBoundaries_owned,
                                   proteus::pyarray<proteus_index_t>& u_l2g,
                                   proteus::pyarray<double>& u_dof,
                                   proteus::pyarray<double>& q_rho,
                                   proteus_index_t offset_u,
                                   proteus_index_t stride_u,
                                   proteus::pyarray<double>& globalResidual,
                                   proteus_index_t nExteriorElementBoundaries_global,
                                   proteus::pyarray<proteus_index_t>& exteriorElementBoundariesArray,
                                   proteus::pyarray<proteus_index_t>& elementBoundaryElementsArray,
                                   proteus::pyarray<int>& elementBoundaryLocalElementBoundariesArray,
                                   proteus::pyarray<int>& elementBoundaryMaterialTypesArray,
                                   proteus::pyarray<double>& Aij,
                                   int added_mass_i,
                                   proteus::pyarray<double>& barycenters,
                                   proteus::pyarray<int>& flags_rigidbody)=0;*/
    virtual void calculateJacobian(//element
                                   proteus::pyarray<double>& mesh_trial_ref,
                                   proteus::pyarray<double>& mesh_grad_trial_ref,
                                   proteus::pyarray<double>& mesh_dof,
                                   proteus::pyarray<proteus_index_t>& mesh_l2g,
                                   proteus::pyarray<double>& dV_ref,
                                   proteus::pyarray<double>& u_trial_ref,
                                   proteus::pyarray<double>& u_grad_trial_ref,
                                   proteus::pyarray<double>& u_test_ref,
                                   proteus::pyarray<double>& u_grad_test_ref,
                                   //element boundary
                                   proteus::pyarray<double>& mesh_trial_trace_ref,
                                   proteus::pyarray<double>& mesh_grad_trial_trace_ref,
                                   proteus::pyarray<double>& dS_ref,
                                   proteus::pyarray<double>& u_trial_trace_ref,
                                   proteus::pyarray<double>& u_grad_trial_trace_ref,
                                   proteus::pyarray<double>& u_test_trace_ref,
                                   proteus::pyarray<double>& u_grad_test_trace_ref,
                                   proteus::pyarray<double>& normal_ref,
                                   proteus::pyarray<double>& boundaryJac_ref,
                                   //physics
                                   proteus_index_t nElements_global,
                                   proteus::pyarray<proteus_index_t>& u_l2g,
                                   proteus::pyarray<double>& u_dof,
                                   proteus::pyarray<double>& q_rho,
                                   proteus::pyarray<proteus_csr_index_t>& csrRowIndeces_u_u,
                                   proteus::pyarray<proteus_csr_index_t>& csrColumnOffsets_u_u,
                                   proteus::pyarray<double>& globalJacobian,
                                   proteus_index_t nExteriorElementBoundaries_global,
                                   proteus::pyarray<proteus_index_t>& exteriorElementBoundariesArray,
                                   proteus::pyarray<proteus_index_t>& elementBoundaryElementsArray,
                                   proteus::pyarray<int>& elementBoundaryLocalElementBoundariesArray,
                                   proteus::pyarray<proteus_csr_index_t>& csrColumnOffsets_eb_u_u)=0;
  };

  template<class CompKernelType,
           int nSpace,
           int nQuadraturePoints_element,
           int nDOF_mesh_trial_element,
           int nDOF_trial_element,
           int nDOF_test_element,
           int nQuadraturePoints_elementBoundary>
  class cppAddedMass : public cppAddedMass_base
  {
  public:
    const int nDOF_test_X_trial_element;
    CompKernelType ck;
    cppAddedMass():
      nDOF_test_X_trial_element(nDOF_test_element*nDOF_trial_element),
      ck()
    {}
    inline
      void evaluateCoefficients(const double& rho,
                                double& a)
    {
      a = 1.0/rho;
    }

    inline
      void exteriorNumericalDiffusiveFlux(const double n[nSpace],
                                          const double a[nSpace],
                                          int isBodyBoundary,
                                          double& flux)
    {
      flux=0.0;
      if (isBodyBoundary == 1) {
        for (int I=0;I<nSpace;I++) {
          flux -= a[I]*n[I];
        }
      }
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
                                         double* q_rho,
                                         proteus_index_t offset_u,
                                         proteus_index_t stride_u,
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
          proteus_index_t eN_k = eN*nQuadraturePoints_element+k;
          double u=0.0,grad_u[nSpace],
            a=0.0,
            jac[nSpace*nSpace],
            jacDet,
            jacInv[nSpace*nSpace],
            u_grad_trial[nDOF_trial_element*nSpace],
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
              for (int I=0;I<nSpace;I++)
                {
                  u_grad_test_dV[j*nSpace+I]   = u_grad_trial[j*nSpace+I]*dV;//cek warning won't work for Petrov-Galerkin
                }
            }
          //
          //calculate pde coefficients at quadrature points
          //
          evaluateCoefficients(q_rho[eN_k], a);
          //
          //update element residual
          //
          for(int i=0;i<nDOF_test_element;i++)
            {
              //int eN_k_i=eN_k*nDOF_test_element+i;
              //int eN_k_i_nSpace = eN_k_i*nSpace;
              int  i_nSpace=i*nSpace;
              elementResidual_u[i] += ck.NumericalDiffusion(a,grad_u,&u_grad_test_dV[i_nSpace]);
            }//i
        }
    }

    /*void calculateResidual(//element
                           proteus::pyarray<double>& mesh_trial_ref,
                           proteus::pyarray<double>& mesh_grad_trial_ref,
                           proteus::pyarray<double>& mesh_dof,
                           proteus::pyarray<proteus_index_t>& mesh_l2g,
                           proteus::pyarray<double>& dV_ref,
                           proteus::pyarray<double>& u_trial_ref,
                           proteus::pyarray<double>& u_grad_trial_ref,
                           proteus::pyarray<double>& u_test_ref,
                           proteus::pyarray<double>& u_grad_test_ref,
                           //element boundary
                           proteus::pyarray<double>& mesh_trial_trace_ref,
                           proteus::pyarray<double>& mesh_grad_trial_trace_ref,
                           proteus::pyarray<double>& dS_ref,
                           proteus::pyarray<double>& u_trial_trace_ref,
                           proteus::pyarray<double>& u_grad_trial_trace_ref,
                           proteus::pyarray<double>& u_test_trace_ref,
                           proteus::pyarray<double>& u_grad_test_trace_ref,
                           proteus::pyarray<double>& normal_ref,
                           proteus::pyarray<double>& boundaryJac_ref,
                           //physics
                           proteus_index_t nElements_global,
                           proteus_index_t nElementBoundaries_owned,
                           proteus::pyarray<proteus_index_t>& u_l2g,
                           proteus::pyarray<double>& u_dof,
                           proteus::pyarray<double>& q_rho,
                           proteus_index_t offset_u,
                           proteus_index_t stride_u,
                           proteus::pyarray<double>& globalResidual,
                           proteus_index_t nExteriorElementBoundaries_global,
                           proteus::pyarray<proteus_index_t>& exteriorElementBoundariesArray,
                           proteus::pyarray<proteus_index_t>& elementBoundaryElementsArray,
                           proteus::pyarray<int>& elementBoundaryLocalElementBoundariesArray,
                           proteus::pyarray<int>& elementBoundaryMaterialTypesArray,
                           proteus::pyarray<double>& Aij,
                           int added_mass_i,
                           proteus::pyarray<double>& barycenters,
                           proteus::pyarray<int>& flags_rigidbody)*/
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
        //element boundary
        proteus::pyarray<double>& mesh_trial_trace_ref = args.array<double>("mesh_trial_trace_ref");
        proteus::pyarray<double>& mesh_grad_trial_trace_ref = args.array<double>("mesh_grad_trial_trace_ref");
        proteus::pyarray<double>& dS_ref = args.array<double>("dS_ref");
        proteus::pyarray<double>& u_trial_trace_ref = args.array<double>("u_trial_trace_ref");
        proteus::pyarray<double>& u_grad_trial_trace_ref = args.array<double>("u_grad_trial_trace_ref");
        proteus::pyarray<double>& u_test_trace_ref = args.array<double>("u_test_trace_ref");
        proteus::pyarray<double>& u_grad_test_trace_ref = args.array<double>("u_grad_test_trace_ref");
        proteus::pyarray<double>& normal_ref = args.array<double>("normal_ref");
        proteus::pyarray<double>& boundaryJac_ref = args.array<double>("boundaryJac_ref");
        //physics
        proteus_index_t nElements_global = args.scalar<proteus_index_t>("nElements_global");
        proteus_index_t nElementBoundaries_owned = args.scalar<proteus_index_t>("nElementBoundaries_owned");
        proteus::pyarray<proteus_index_t>& u_l2g = args.array<proteus_index_t>("u_l2g");
        proteus::pyarray<double>& u_dof = args.array<double>("u_dof"); 
        proteus::pyarray<double>& q_rho = args.array<double>("q_rho");
        proteus_index_t offset_u = args.scalar<proteus_index_t>("offset_u");
        proteus_index_t stride_u = args.scalar<proteus_index_t>("stride_u");
        proteus::pyarray<double>& globalResidual = args.array<double>("globalResidual");
        proteus_index_t nExteriorElementBoundaries_global = args.scalar<proteus_index_t>("nExteriorElementBoundaries_global");
        proteus::pyarray<proteus_index_t>& exteriorElementBoundariesArray = args.array<proteus_index_t>("exteriorElementBoundariesArray");
        proteus::pyarray<proteus_index_t>& elementBoundaryElementsArray = args.array<proteus_index_t>("elementBoundaryElementsArray");
        proteus::pyarray<int>& elementBoundaryLocalElementBoundariesArray = args.array<int>("elementBoundaryLocalElementBoundariesArray");
        proteus::pyarray<int>& elementBoundaryMaterialTypesArray = args.array<int>("elementBoundaryMaterialTypesArray");
        proteus::pyarray<double>& Aij = args.array<double>("Aij");
        int added_mass_i = args.scalar<int>("added_mass_i");
        proteus::pyarray<double>& barycenters = args.array<double>("barycenters");
        proteus::pyarray<int>& flags_rigidbody = args.array<int>("flags_rigidbody");
      for(proteus_index_t eN=0;eN<nElements_global;eN++)
        {
          for  (int k=0;k<nQuadraturePoints_element;k++)
            {
              double
                jac[nSpace*nSpace],
                jacDet,
                jacInv[nSpace*nSpace],
                x,y,z;
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
            }
        }
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
              element_u[i] = u_dof[u_l2g[eN_i]];
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
                                   q_rho.data(),
                                   offset_u,
                                   stride_u,
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
              globalResidual[offset_u+stride_u*u_l2g[eN_i]]+=elementResidual_u[i];
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
          proteus_index_t ebN = exteriorElementBoundariesArray[ebNE],
            eN  = elementBoundaryElementsArray[ebN*2+0];
          int ebN_local = elementBoundaryLocalElementBoundariesArray[ebN*2+0];
            //eN_nDOF_trial_element = eN*nDOF_trial_element;
          double elementResidual_u[nDOF_test_element];
          double element_u[nDOF_trial_element];
          for (int i=0;i<nDOF_test_element;i++)
            {
              proteus_index_t eN_i=eN*nDOF_test_element+i;
              element_u[i] = u_dof[u_l2g[eN_i]];
              elementResidual_u[i] = 0.0;
            }//i
          for  (int kb=0;kb<nQuadraturePoints_elementBoundary;kb++)
            {
              int ebN_local_kb = ebN_local*nQuadraturePoints_elementBoundary+kb,
                ebN_local_kb_nSpace = ebN_local_kb*nSpace;
              double penalty=0.0,
                u_ext=0.0,
                diff_flux_ext=0.0,
                jac_ext[nSpace*nSpace],
                jacDet_ext,
                jacInv_ext[nSpace*nSpace],
                boundaryJac[nSpace*(nSpace-1)],
                metricTensor[(nSpace-1)*(nSpace-1)],
                metricTensorDetSqrt,
                dS,
                u_test_dS[nDOF_test_element],
                u_grad_trial_trace[nDOF_trial_element*nSpace],
                normal[nSpace],x_ext,y_ext,z_ext=0.0,
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
              dS = metricTensorDetSqrt*dS_ref[kb];
              //get the metric tensor
              //cek todo use symmetry
              ck.calculateG(jacInv_ext,G,G_dd_G,tr_G);
              ck.calculateGScale(G,normal,penalty);
              //compute shape and solution information
              //shape
              ck.gradTrialFromRef(&u_grad_trial_trace_ref[ebN_local_kb_nSpace*nDOF_trial_element],jacInv_ext,u_grad_trial_trace);
              //solution and gradients
              ck.valFromElementDOF(element_u,&u_trial_trace_ref[ebN_local_kb*nDOF_test_element],u_ext);
              //ck.gradFromElementDOF(element_u,u_grad_trial_trace,grad_u_ext);
              //precalculate test function products with integration weights
              for (int j=0;j<nDOF_trial_element;j++)
                {
                  u_test_dS[j] = u_test_trace_ref[ebN_local_kb*nDOF_test_element+j]*dS;
                }
              //
              //calculate the numerical fluxes
              //
              int eBMT = elementBoundaryMaterialTypesArray[ebN];
              double rx, ry, rz;
              rx = x_ext-barycenters[3*eBMT+0];
              ry = y_ext-barycenters[3*eBMT+1];
              rz = z_ext-barycenters[3*eBMT+2];
              double added_mass_a[3] = {0.0, 0.0, 0.0};
	      if (eBMT > 0)
		{
		  switch (added_mass_i)
		    {
		    case 0:
		      added_mass_a[0] = 1.0;
		      break;
		    case 1:
		      added_mass_a[1] = 1.0;
		      break;
		    case 2:
		      added_mass_a[2] = 1.0;
		      break;
		    case 3:
		      added_mass_a[1] = -rz;
		      added_mass_a[2] =  ry;
		      break;
		    case 4:
		      added_mass_a[0] =  rz;
		      added_mass_a[2] = -rx;
		      break;
		    case 5:
		      added_mass_a[0] = -ry;
		      added_mass_a[1] =  rx;
		      break;
		    default:
		      assert(0);
		    }
		}
              // normalise unit accelerations (necessary for angular ones)
	      //I think we want the angular acceleration to be 1
	      //but the flux uses whatever the linear acceleration works
	      //out to be, so I'm commenting this out for now
              /* double added_mass_a_tot = sqrt(added_mass_a[0]*added_mass_a[0]+ */
              /*                                added_mass_a[1]*added_mass_a[1]+ */
              /*                                added_mass_a[2]*added_mass_a[2]); */
              /* added_mass_a[0] = added_mass_a[0]/added_mass_a_tot; */
              /* added_mass_a[1] = added_mass_a[1]/added_mass_a_tot; */
              /* added_mass_a[2] = added_mass_a[2]/added_mass_a_tot; */
	      
              exteriorNumericalDiffusiveFlux(normal,
                                             added_mass_a,
                                             flags_rigidbody[eBMT],
                                             diff_flux_ext);
              //
              //update residuals
              //
              for (int i=0;i<nDOF_test_element;i++)
                {
                  elementResidual_u[i] +=
                    + ck.ExteriorElementBoundaryFlux(diff_flux_ext,u_test_dS[i]);
                }//i
              //calculate Aij
              if (ebN < nElementBoundaries_owned)
                {
                  double px, py, pz;
                  px = u_ext*normal[0];
                  py = u_ext*normal[1];
                  if (nSpace==3)
                    pz = u_ext*normal[2];
                  else
                    pz=0.0;
                  Aij[36*eBMT+added_mass_i+6*0] += px*dS;
                  Aij[36*eBMT+added_mass_i+6*1] += py*dS;
                  Aij[36*eBMT+added_mass_i+6*2] += pz*dS;
                  Aij[36*eBMT+added_mass_i+6*3] += (ry*pz-rz*py)*dS;
                  Aij[36*eBMT+added_mass_i+6*4] += (rz*px-rx*pz)*dS;
                  Aij[36*eBMT+added_mass_i+6*5] += (rx*py-ry*px)*dS;
                }
            }//kb
          //
          //update the element and global residual storage
          //
          for (int i=0;i<nDOF_test_element;i++)
            {
              proteus_index_t eN_i = eN*nDOF_test_element+i;
              globalResidual[offset_u+stride_u*u_l2g[eN_i]] += elementResidual_u[i];
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
                                         double* q_rho,
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
          proteus_index_t eN_k = eN*nQuadraturePoints_element+k; //index to a scalar at a quadrature point

          //declare local storage
          double u=0.0,
            grad_u[nSpace],
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
          evaluateCoefficients(q_rho[eN_k], a);
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

    void calculateJacobian(//element
                           proteus::pyarray<double>& mesh_trial_ref,
                           proteus::pyarray<double>& mesh_grad_trial_ref,
                           proteus::pyarray<double>& mesh_dof,
                           proteus::pyarray<proteus_index_t>& mesh_l2g,
                           proteus::pyarray<double>& dV_ref,
                           proteus::pyarray<double>& u_trial_ref,
                           proteus::pyarray<double>& u_grad_trial_ref,
                           proteus::pyarray<double>& u_test_ref,
                           proteus::pyarray<double>& u_grad_test_ref,
                           //element boundary
                           proteus::pyarray<double>& mesh_trial_trace_ref,
                           proteus::pyarray<double>& mesh_grad_trial_trace_ref,
                           proteus::pyarray<double>& dS_ref,
                           proteus::pyarray<double>& u_trial_trace_ref,
                           proteus::pyarray<double>& u_grad_trial_trace_ref,
                           proteus::pyarray<double>& u_test_trace_ref,
                           proteus::pyarray<double>& u_grad_test_trace_ref,
                           proteus::pyarray<double>& normal_ref,
                           proteus::pyarray<double>& boundaryJac_ref,
                           //physics
                           proteus_index_t nElements_global,
                           proteus::pyarray<proteus_index_t>& u_l2g,
                           proteus::pyarray<double>& u_dof,
                           proteus::pyarray<double>& q_rho,
                           proteus::pyarray<proteus_csr_index_t>& csrRowIndeces_u_u,proteus::pyarray<proteus_csr_index_t>& csrColumnOffsets_u_u,
                           proteus::pyarray<double>& globalJacobian,
                           proteus_index_t nExteriorElementBoundaries_global,
                           proteus::pyarray<proteus_index_t>& exteriorElementBoundariesArray,
                           proteus::pyarray<proteus_index_t>& elementBoundaryElementsArray,
                           proteus::pyarray<int>& elementBoundaryLocalElementBoundariesArray,
                           proteus::pyarray<proteus_csr_index_t>& csrColumnOffsets_eb_u_u)
    {
      //
      //loop over elements to compute volume integrals and load them into the element Jacobians and global Jacobian
      //
      for(proteus_index_t eN=0;eN<nElements_global;eN++)
        {
          double  elementJacobian_u_u[nDOF_test_element*nDOF_trial_element],element_u[nDOF_trial_element];
          for (int j=0;j<nDOF_trial_element;j++)
            {
              proteus_index_t eN_j = eN*nDOF_trial_element+j;
              element_u[j] = u_dof[u_l2g[eN_j]];
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
                                   q_rho.data(),
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
                  globalJacobian[csrRowIndeces_u_u[eN_i] + csrColumnOffsets_u_u[eN_i_j]] += elementJacobian_u_u[i*nDOF_trial_element+j];
                }//j
            }//i
        }//elements
    }//computeJacobian
  };//cppAddedMass

  inline cppAddedMass_base* newAddedMass(int nSpaceIn,
					 int nQuadraturePoints_elementIn,
					 int nDOF_mesh_trial_elementIn,
					 int nDOF_trial_elementIn,
					 int nDOF_test_elementIn,
					 int nQuadraturePoints_elementBoundaryIn,
					 int CompKernelFlag)
  {
    if (nSpaceIn == 2)
      return proteus::chooseAndAllocateDiscretization2D<cppAddedMass_base,cppAddedMass,CompKernel>(nSpaceIn,
												   nQuadraturePoints_elementIn,
												   nDOF_mesh_trial_elementIn,
												   nDOF_trial_elementIn,
												   nDOF_test_elementIn,
												   nQuadraturePoints_elementBoundaryIn,
												   CompKernelFlag);
    else
      return proteus::chooseAndAllocateDiscretization<cppAddedMass_base,cppAddedMass,CompKernel>(nSpaceIn,
												 nQuadraturePoints_elementIn,
												 nDOF_mesh_trial_elementIn,
												 nDOF_trial_elementIn,
												 nDOF_test_elementIn,
												 nQuadraturePoints_elementBoundaryIn,
												 CompKernelFlag);
  }
}//proteus
#endif
