#ifndef MCorr_H
#define MCorr_H
#include <cmath>
#include <iostream>
#include <set>
#include <map>
#include <valarray>
#include "CompKernel.h"
#include "ModelFactory.h"
#include "equivalent_polynomials.h"
#include PROTEUS_LAPACK_H
#include "ArgumentsDict.h"
#include "pyarray.h"

namespace py = pybind11;

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
  //using GeneralizedFunctions = equivalent_polynomials::Regularized<nSpace, nP, nQ>;
  //using GeneralizedFunctions = equivalent_polynomials::EquivalentPolynomials<nSpace, nP, nQ>;

  class MCorr_base
  {
  public:
    std::valarray<double> Rpos, Rneg;
    std::valarray<double> FluxCorrectionMatrix;
    virtual ~MCorr_base(){}
    virtual void calculateResidual(arguments_dict& args, bool useExact)=0;
    virtual void calculateJacobian(arguments_dict& args, bool useExact)=0;
    virtual void elementSolve(arguments_dict& args)=0;
    virtual void elementConstantSolve(arguments_dict& args)=0;
    virtual std::tuple<double, double> globalConstantRJ(arguments_dict& args)=0;
    virtual double calculateMass(arguments_dict& args, bool useExact)=0;
    virtual void setMassQuadrature(arguments_dict& args, bool useExact)=0;
    virtual void FCTStep(arguments_dict& args)=0;
    virtual void calculateMassMatrix(arguments_dict& args)=0;
    virtual void setMassQuadratureEdgeBasedStabilizationMethods(arguments_dict& args,
                                                                bool useExact)=0;
  };

  template<class CompKernelType,
    int nSpace,
    int nQuadraturePoints_element,
    int nDOF_mesh_trial_element,
    int nDOF_trial_element,
    int nDOF_test_element,
    int nQuadraturePoints_elementBoundary>
    class MCorr : public MCorr_base
    {
    public:
      std::set<int> cutfem_boundaries;
      std::map<int, int> cutfem_local_boundaries;
      const int nDOF_test_X_trial_element;
      CompKernelType ck;
      GeneralizedFunctions<nSpace,2,nQuadraturePoints_element,nQuadraturePoints_elementBoundary> gf;
      GeneralizedFunctions<nSpace,2,nDOF_trial_element,nQuadraturePoints_elementBoundary> gf_nodes;
      GeneralizedFunctions<nSpace,4,nQuadraturePoints_element,nQuadraturePoints_elementBoundary> gf_s;
    MCorr():
      nDOF_test_X_trial_element(nDOF_test_element*nDOF_trial_element),
      ck()
        {}

      inline
        void evaluateCoefficients(const double& epsHeaviside,
                                  const double& epsDirac,
                                  const double& phi,
                                  const double& H,
                                  const double& u,
                                  const double& porosity,
                                  double& r,
                                  double& dr)
      {
        r = porosity*(gf.H(epsHeaviside,phi+u) - H);
        // the Jacobian is the derivative of r, so the Dirac must be the derivative of the SAME regularized Heaviside: epsHeaviside, not epsDirac
        dr = porosity*gf.D(epsHeaviside,phi+u);
      }

      inline void calculateElementResidual(//element
                                           double* mesh_trial_ref,
                                           double* mesh_grad_trial_ref,
                                           double* mesh_dof,
                                           int* mesh_l2g,
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
                                           int nElements_global,
                                           double useMetrics,
                                           double epsFactHeaviside,
                                           double epsFactDirac,
                                           double epsFactDiffusion,
                                           int* u_l2g,
                                           int* r_l2g,
                                           double* elementDiameter,
                                           double* nodeDiametersArray,
                                           double* u_dof,
                                           double* q_phi,
                                           double* q_normal_phi,
                                           double* ebqe_phi,
                                           double* ebqe_normal_phi,
                                           double* q_H,
                                           double* q_u,
                                           double* q_n,
                                           double* ebqe_u,
                                           double* ebqe_n,
                                           double* q_r,
                                           double* q_porosity,
                                           int offset_u, int stride_u,
                                           double* elementResidual_u,
                                           int nExteriorElementBoundaries_global,
                                           int* exteriorElementBoundariesArray,
                                           int* elementBoundaryElementsArray,
                                           int* elementBoundaryLocalElementBoundariesArray,
                                           double* element_u,
                                           int eN,
					   bool element_active,
					   double* isActiveR,
					   double* isActiveDOF,
					   const double* phi_solid)
      {
        for (int i=0;i<nDOF_test_element;i++)
          {
            elementResidual_u[i]=0.0;
          }//i
        double epsHeaviside,epsDirac,epsDiffusion,norm;
        //loop over quadrature points and compute integrands
        for  (int k=0;k<nQuadraturePoints_element;k++)
          {
            //compute indeces and declare local storage
            int eN_k = eN*nQuadraturePoints_element+k,
              eN_k_nSpace = eN_k*nSpace;
            //eN_nDOF_trial_element = eN*nDOF_trial_element;
            double u=0.0,grad_u[nSpace],
              r=0.0,dr=0.0,
              jac[nSpace*nSpace],
              jacDet,
              jacInv[nSpace*nSpace],
              u_grad_trial[nDOF_trial_element*nSpace],
              u_test_dV[nDOF_trial_element],
              u_grad_test_dV[nDOF_test_element*nSpace],
              dV,x,y,z,
              G[nSpace*nSpace],G_dd_G,tr_G,h_phi;
            gf.set_quad(k);
	    gf_s.set_quad(k);
	    const double H_s = gf_s.H(0.0,phi_solid[eN_k]);
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
            ck.calculateH_element(eN,
                                  k,
                                  nodeDiametersArray,
                                  mesh_l2g,
                                  mesh_trial_ref,
                                  h_phi);
            //get the physical integration weight
            dV = fabs(jacDet)*dV_ref[k];
            ck.calculateG(jacInv,G,G_dd_G,tr_G);

            /* double dir[nSpace]; */
            /* double norm = 1.0e-8; */
            /* for (int I=0;I<nSpace;I++) */
            /*   norm += q_normal_phi[eN_k_nSpace+I]*q_normal_phi[eN_k_nSpace+I]; */
            /* norm = sqrt(norm);    */
            /* for (int I=0;I<nSpace;I++) */
            /*   dir[I] = q_normal_phi[eN_k_nSpace+I]/norm; */
            /* ck.calculateGScale(G,dir,h_phi); */

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
            epsHeaviside = epsFactHeaviside*(useMetrics*h_phi+(1.0-useMetrics)*elementDiameter[eN]);
            epsDirac     = epsFactDirac*    (useMetrics*h_phi+(1.0-useMetrics)*elementDiameter[eN]);
            epsDiffusion = epsFactDiffusion*(useMetrics*h_phi+(1.0-useMetrics)*elementDiameter[eN]);
            // *(useMetrics*h_phi+(1.0-useMetrics)*elementDiameter[eN]);
            evaluateCoefficients(epsHeaviside,
                                 epsDirac,
                                 q_phi[eN_k],
                                 q_H[eN_k],
                                 u,
                                 q_porosity[eN_k],
                                 r,
                                 dr);
            //
            //update element residual
            //
            for(int i=0;i<nDOF_test_element;i++)
              {
		int eN_i = eN*nDOF_test_element+i;
                //int eN_k_i=eN_k*nDOF_test_element+i;
                //int eN_k_i_nSpace = eN_k_i*nSpace;
                int  i_nSpace=i*nSpace;

                elementResidual_u[i] += H_s*(ck.Reaction_weak(r,u_test_dV[i]) +
					     ck.NumericalDiffusion(epsDiffusion,grad_u,&u_grad_test_dV[i_nSpace]));
		if (element_active)
		  {
		    isActiveR[offset_u + stride_u*r_l2g[eN_i]] = 1.0;
		    isActiveDOF[u_l2g[eN_i]] = 1.0;
		  }
              }//i
            //
            //save momentum for time history and velocity for subgrid error
            //save solution for other models
            //

            q_r[eN_k] = r;
            q_u[eN_k] = u;


            norm = 1.0e-8;
            for (int I=0;I<nSpace;I++)
              norm += grad_u[I]*grad_u[I];
            norm = sqrt(norm);
            for(int I=0;I<nSpace;I++)
              q_n[eN_k_nSpace+I] = grad_u[I]/norm;
          }
      }
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
        double epsFactHeaviside = args.scalar<double>("epsFactHeaviside");
        double epsFactDirac = args.scalar<double>("epsFactDirac");
        double epsFactDiffusion = args.scalar<double>("epsFactDiffusion");
        proteus::pyarray<int>& u_l2g = args.array<int>("u_l2g");
        proteus::pyarray<int>& r_l2g = args.array<int>("r_l2g");
        proteus::pyarray<double>& elementDiameter = args.array<double>("elementDiameter");
	proteus::pyarray<double>& elementBoundaryDiameter = args.array<double>("elementBoundaryDiameter");
        proteus::pyarray<double>& nodeDiametersArray = args.array<double>("nodeDiametersArray");
        proteus::pyarray<double>& u_dof = args.array<double>("u_dof");
        proteus::pyarray<double>& phi_dof = args.array<double>("phi_dof");
        proteus::pyarray<double>& q_phi = args.array<double>("q_phi");
        proteus::pyarray<double>& q_normal_phi = args.array<double>("q_normal_phi");
        proteus::pyarray<double>& ebqe_phi = args.array<double>("ebqe_phi");
        proteus::pyarray<double>& ebqe_normal_phi = args.array<double>("ebqe_normal_phi");
        proteus::pyarray<double>& q_H = args.array<double>("q_H");
        proteus::pyarray<double>& q_u = args.array<double>("q_u");
        proteus::pyarray<double>& q_n = args.array<double>("q_n");
        proteus::pyarray<double>& ebqe_u = args.array<double>("ebqe_u");
        proteus::pyarray<double>& ebqe_n = args.array<double>("ebqe_n");
        proteus::pyarray<double>& q_r = args.array<double>("q_r");
        proteus::pyarray<double>& q_porosity = args.array<double>("q_porosity");
        int offset_u = args.scalar<int>("offset_u");
        int stride_u = args.scalar<int>("stride_u");
        proteus::pyarray<double>& globalResidual = args.array<double>("globalResidual");
        int nExteriorElementBoundaries_global = args.scalar<int>("nExteriorElementBoundaries_global");
        proteus::pyarray<int>& exteriorElementBoundariesArray = args.array<int>("exteriorElementBoundariesArray");
	proteus::pyarray<int>& elementBoundariesArray = args.array<int>("elementBoundariesArray");
        proteus::pyarray<int>& elementBoundaryElementsArray = args.array<int>("elementBoundaryElementsArray");
        proteus::pyarray<int>& elementBoundaryLocalElementBoundariesArray = args.array<int>("elementBoundaryLocalElementBoundariesArray");
	proteus::pyarray<double>& ebqe_phi_s = args.array<double>("ebqe_phi_s");
	double ghost_penalty_constant = args.scalar<double>("ghost_penalty_constant");
	const proteus::pyarray<double>& phi_solid = args.array<double>("phi_solid");
	proteus::pyarray<double>& phi_solid_nodes = args.array<double>("phi_solid_nodes");
	bool useExact_s = args.scalar<int>("useExact_s");
	proteus::pyarray<double>& isActiveR = args.array<double>("isActiveR");
	proteus::pyarray<double>& isActiveDOF = args.array<double>("isActiveDOF");
	proteus::pyarray<int>& isActiveElement = args.array<int>("isActiveElement");
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
        gf.useExact = useExact;
        gf_s.useExact = useExact_s;
	cutfem_boundaries.clear();
	cutfem_local_boundaries.clear();
        for(int eN=0;eN<nElements_global;eN++)
          {
            //declare local storage for element residual and initialize
            double elementResidual_u[nDOF_test_element], element_u[nDOF_trial_element], element_phi[nDOF_trial_element], element_phi_s[nDOF_mesh_trial_element];
	    bool element_active=false;
	    isActiveElement[eN]=0;
            for (int i=0;i<nDOF_test_element;i++)
              {
                int eN_i=eN*nDOF_test_element+i;
                element_u[i] = u_dof.data()[u_l2g.data()[eN_i]];
                element_phi[i] = phi_dof.data()[u_l2g.data()[eN_i]] + element_u[i];
		element_phi_s[i] = phi_solid_nodes.data()[u_l2g.data()[eN_i]];
              }//i
            double element_nodes[nDOF_mesh_trial_element*3];
            for (int i=0;i<nDOF_mesh_trial_element;i++)
              {
                int eN_i=eN*nDOF_mesh_trial_element+i;
                for(int I=0;I<3;I++)
                  element_nodes[i*3 + I] = mesh_dof.data()[mesh_l2g.data()[eN_i]*3 + I];
	      }//i
            gf.calculate(element_phi, element_nodes, x_ref.data(),false);
	    int icase_s = gf_s.calculate(element_phi_s, element_nodes, x_ref.data(),false);
	    if (icase_s == 0)
	      {
		element_active=true;
		isActiveElement[eN]=1;
		//only works for simplices
		for (int ebN_element=0;ebN_element < nDOF_mesh_trial_element; ebN_element++)
		  {
		    const int ebN = elementBoundariesArray.data()[eN*nDOF_mesh_trial_element+ebN_element];
		    //internal and actually a cut edge
		    //if (elementBoundaryElementsArray.data()[ebN*2+1] != -1 && (ebN < nElementBoundaries_owned) && element_phi_s[(ebN_element+1)%nDOF_mesh_trial_element]*element_phi_s[(ebN_element+2)%nDOF_mesh_trial_element] < 0.0)
		    if (elementBoundaryElementsArray[ebN*2+1] != -1 && element_phi_s[(ebN_element+1)%nDOF_mesh_trial_element]*element_phi_s[(ebN_element+2)%nDOF_mesh_trial_element] <= 0.0)
		      {
		     cutfem_boundaries.insert(ebN);
		     if (elementBoundaryElementsArray[ebN*2 + 0] == eN)
		       cutfem_local_boundaries[ebN] = ebN_element;
		      }
		  }
	      }
	    else if (icase_s == 1)
	      {
		element_active=true;
		isActiveElement[eN]=1;
	      }
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
				     useMetrics,
				     epsFactHeaviside,
				     epsFactDirac,
				     epsFactDiffusion,
				     u_l2g.data(),
				     r_l2g.data(),
				     elementDiameter.data(),
				     nodeDiametersArray.data(),
				     u_dof.data(),
				     q_phi.data(),
				     q_normal_phi.data(),
				     ebqe_phi.data(),
				     ebqe_normal_phi.data(),
				     q_H.data(),
				     q_u.data(),
				     q_n.data(),
				     ebqe_u.data(),
				     ebqe_n.data(),
				     q_r.data(),
				     q_porosity.data(),
				     offset_u,stride_u,
				     elementResidual_u,
				     nExteriorElementBoundaries_global,
				     exteriorElementBoundariesArray.data(),
				     elementBoundaryElementsArray.data(),
				     elementBoundaryLocalElementBoundariesArray.data(),
				     element_u,
				     eN,
				     element_active,
				     isActiveR.data(),
				     isActiveDOF.data(),
				     phi_solid.data());
	    //
	    //load element into global residual and save element residual
	    //
	    for(int i=0;i<nDOF_test_element;i++)
	      {
		int eN_i=eN*nDOF_test_element+i;

                globalResidual.data()[offset_u+stride_u*r_l2g.data()[eN_i]]+=elementResidual_u[i];
              }//i
          }//elements
	std::set<int>::iterator it=cutfem_boundaries.begin();
	while(it!=cutfem_boundaries.end())
	  {
	    if(isActiveElement[elementBoundaryElementsArray[(*it)*2+0]] && isActiveElement[elementBoundaryElementsArray[(*it)*2+1]])
	      {
		std::map<int,double> DW_Dn_jump;
		double gamma_cutfem=ghost_penalty_constant,
		  h_cutfem=elementBoundaryDiameter.data()[*it];
		int eN_nDOF_trial_element  = elementBoundaryElementsArray.data()[(*it)*2+0]*nDOF_trial_element;
		//See Massing Schott Wall 2018
		//double norm_v=0.0;
		//for (int i_offset=1;i_offset<nDOF_trial_element;i_offset++)//MSW18 is just on face, so trying to just use face dof
		//  {
		//		  int i = (cutfem_local_boundaries[*it] + i_offset)%nDOF_trial_element;
		//    double u=u_old_dof.data()[vel_l2g.data()[eN_nDOF_trial_element+i]];
		//      v=v_old_dof.data()[vel_l2g.data()[eN_nDOF_v_trial_element+i]],
		//      w=w_old_dof.data()[vel_l2g.data()[eN_nDOF_v_trial_element+i]];
		//   norm_v=fmax(norm_v,sqrt(u*u+v*v+w*w));
		//  }
		//double gamma_v_dim = rho_0*(nu_0 + norm_v*h_cutfem + alphaBDF*h_cutfem*h_cutfem);
		//gamma_cutfem_p *= h_cutfem*h_cutfem/gamma_v_dim;
		//if (NONCONSERVATIVE_FORM)
		//  gamma_cutfem*=gamma_v_dim;
		//else
		//  gamma_cutfem*=(gamma_v_dim/rho_0);
		for (int kb=0;kb<nQuadraturePoints_elementBoundary;kb++)
		  {
		    double Du_Dn_jump=0.0, dS;
		    for (int eN_side=0;eN_side < 2; eN_side++)
		      {
			int ebN = *it,
			  eN  = elementBoundaryElementsArray.data()[ebN*2+eN_side];
			for (int i=0;i<nDOF_test_element;i++)
			  {
			    DW_Dn_jump[r_l2g.data()[eN*nDOF_test_element+i]] = 0.0;
			  }
		      }
		    for (int eN_side=0;eN_side < 2; eN_side++)
		      {
			int ebN = *it,
			  eN  = elementBoundaryElementsArray[ebN*2+eN_side],
			  ebN_local = elementBoundaryLocalElementBoundariesArray[ebN*2+eN_side],
			  eN_nDOF_trial_element = eN*nDOF_trial_element,
			  ebN_local_kb = ebN_local*nQuadraturePoints_elementBoundary+kb,
			  ebN_local_kb_nSpace = ebN_local_kb*nSpace;
			double u_int=0.0,
			  grad_u_int[nSpace],
			  jac_int[nSpace*nSpace],
			  jacDet_int,
			  jacInv_int[nSpace*nSpace],
			  boundaryJac[nSpace*(nSpace-1)],
			  metricTensor[(nSpace-1)*(nSpace-1)],
			  metricTensorDetSqrt,
			  u_test_dS[nDOF_test_element],
			  u_grad_trial_trace[nDOF_trial_element*nSpace],
			  u_grad_test_dS[nDOF_trial_element*nSpace],
			  normal[nSpace],x_int,y_int,z_int,xt_int,yt_int,zt_int,integralScaling,
			  G[nSpace*nSpace],G_dd_G,tr_G,h_phi,h_penalty,penalty;
			for (int I=0; I<nSpace;I++)
			  grad_u_int[I] = 0.0;
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
							    jac_int,
							    jacDet_int,
							    jacInv_int,
							    boundaryJac,
							    metricTensor,
							    metricTensorDetSqrt,
							    normal_ref.data(),
							    normal,
							    x_int,y_int,z_int);
			dS = metricTensorDetSqrt*dS_ref.data()[kb];
			//compute shape and solution information
			//shape
			ck.gradTrialFromRef(&u_grad_trial_trace_ref.data()[ebN_local_kb_nSpace*nDOF_trial_element],jacInv_int,u_grad_trial_trace);
			//solution and gradients
			ck.valFromDOF(u_dof.data(),&u_l2g.data()[eN_nDOF_trial_element],&u_trial_trace_ref.data()[ebN_local_kb*nDOF_test_element],u_int);
			ck.gradFromDOF(u_dof.data(),&u_l2g.data()[eN_nDOF_trial_element],u_grad_trial_trace,grad_u_int);
			for (int I=0;I<nSpace;I++)
			  {
			    Du_Dn_jump += grad_u_int[I]*normal[I];
			  }
			for (int i=0;i<nDOF_test_element;i++)
			  {
			    int eN_i = eN*nDOF_test_element + i;
			    for (int I=0;I<nSpace;I++)
			      DW_Dn_jump[r_l2g[eN_i]] += u_grad_trial_trace[i*nSpace+I]*normal[I];
			  }
		      }//eN_side
		    for (std::map<int,double>::iterator W_it=DW_Dn_jump.begin(); W_it!=DW_Dn_jump.end(); ++W_it)
		      {
			int i_global = W_it->first;
			double DW_Dn_jump_i = W_it->second;
			globalResidual.data()[offset_u+stride_u*i_global]+=gamma_cutfem*h_cutfem*Du_Dn_jump*DW_Dn_jump_i*dS;
		      }
		  }//kb
		it++;
	      }
	    else
	      {
		it = cutfem_boundaries.erase(it);
	      }
	  }//cutfem element boundaries
        //
        //loop over exterior element boundaries to calculate levelset gradient
        //
        //ebNE is the Exterior element boundary INdex
        //ebN is the element boundary INdex
        //eN is the element index
        for (int ebNE = 0; ebNE < nExteriorElementBoundaries_global; ebNE++)
          {
            int ebN = exteriorElementBoundariesArray.data()[ebNE],
              eN  = elementBoundaryElementsArray.data()[ebN*2+0],
              ebN_local = elementBoundaryLocalElementBoundariesArray.data()[ebN*2+0];
            //eN_nDOF_trial_element = eN*nDOF_trial_element;
            //double elementResidual_u[nDOF_test_element];
            double element_u[nDOF_trial_element];
            for (int i=0;i<nDOF_test_element;i++)
              {
                int eN_i=eN*nDOF_test_element+i;
                element_u[i] = u_dof.data()[u_l2g.data()[eN_i]];
              }//i
            for  (int kb=0;kb<nQuadraturePoints_elementBoundary;kb++)
              {
                int ebNE_kb = ebNE*nQuadraturePoints_elementBoundary+kb,
                  ebNE_kb_nSpace = ebNE_kb*nSpace,
                  ebN_local_kb = ebN_local*nQuadraturePoints_elementBoundary+kb,
                  ebN_local_kb_nSpace = ebN_local_kb*nSpace;
                double u_ext=0.0,
                  grad_u_ext[nSpace],
                  //m_ext=0.0,
                  //dm_ext=0.0,
                  //H_ext=0.0,
                  //dH_ext[nSpace],
                  //flux_ext=0.0,
                  //bc_u_ext=0.0,
                  //bc_grad_u_ext[nSpace],
                  //bc_m_ext=0.0,
                  //bc_dm_ext=0.0,
                  //bc_H_ext=0.0,
                  //bc_dH_ext[nSpace],
                  jac_ext[nSpace*nSpace],
                  jacDet_ext,
                  jacInv_ext[nSpace*nSpace],
                  boundaryJac[nSpace*(nSpace-1)],
                  metricTensor[(nSpace-1)*(nSpace-1)],
                  metricTensorDetSqrt,
                  dS,
                  //u_test_dS[nDOF_test_element],
                  u_grad_trial_trace[nDOF_trial_element*nSpace],
                  normal[nSpace],x_ext,y_ext,z_ext,
                  G[nSpace*nSpace],G_dd_G,tr_G,norm;
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

                ebqe_u.data()[ebNE_kb] = u_ext;
                norm = 1.0e-8;
                for (int I=0;I<nSpace;I++)
                  norm += grad_u_ext[I]*grad_u_ext[I];
                norm = sqrt(norm);
                for (int I=0;I<nSpace;I++)
                  ebqe_n.data()[ebNE_kb_nSpace+I] = grad_u_ext[I]/norm;
              }//kb
          }//ebNE
      }

      inline void calculateElementJacobian(//element
                                           double* mesh_trial_ref,
                                           double* mesh_grad_trial_ref,
                                           double* mesh_dof,
                                           int* mesh_l2g,
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
                                           int nElements_global,
                                           double useMetrics,
                                           double epsFactHeaviside,
                                           double epsFactDirac,
                                           double epsFactDiffusion,
                                           int* u_l2g,
                                           double* elementDiameter,
                                           double* nodeDiametersArray,
                                           double* u_dof,
                                           // double* u_trial,
                                           // double* u_grad_trial,
                                           // double* u_test_dV,
                                           // double* u_grad_test_dV,
                                           double* q_phi,
                                           double* q_normal_phi,
                                           double* q_H,
                                           double* q_porosity,
                                           double* elementJacobian_u_u,
                                           double* element_u,
					   const double* phi_solid,
                                           int eN,
                                           const double* dHdphi = nullptr,
                                           const double* phi_element = nullptr)
  // dHdphi (nQuadraturePoints_element x nDOF_trial_element), when not null, is d(gf.H at quadrature point k)/d(element phi_j): the exact derivative of the
  // sharp-Heaviside weights (cut elements, useExact), used instead of the Dirac weights times the trial function (exact only for linear integrands).
      {
        for (int i=0;i<nDOF_test_element;i++)
          for (int j=0;j<nDOF_trial_element;j++)
            {
              elementJacobian_u_u[i*nDOF_trial_element+j]=0.0;
            }
        double epsHeaviside,epsDirac,epsDiffusion;
        for  (int k=0;k<nQuadraturePoints_element;k++)
          {
            int eN_k = eN*nQuadraturePoints_element+k, //index to a scalar at a quadrature point
              eN_k_nSpace = eN_k*nSpace;
            //eN_nDOF_trial_element = eN*nDOF_trial_element; //index to a vector at a quadrature point
            gf.set_quad(k);
	    gf_s.set_quad(k);
            //declare local storage
            double u=0.0,
              grad_u[nSpace],
              r=0.0,dr=0.0,
              jac[nSpace*nSpace],
              jacDet,
              jacInv[nSpace*nSpace],
              u_grad_trial[nDOF_trial_element*nSpace],
              dV,
              u_test_dV[nDOF_test_element],
              u_grad_test_dV[nDOF_test_element*nSpace],
              x,y,z,
              G[nSpace*nSpace],G_dd_G,tr_G,h_phi;
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
            ck.calculateH_element(eN,
                                  k,
                                  nodeDiametersArray,
                                  mesh_l2g,
                                  mesh_trial_ref,
                                  h_phi);
            //get the physical integration weight
            dV = fabs(jacDet)*dV_ref[k];
            ck.calculateG(jacInv,G,G_dd_G,tr_G);

            /* double dir[nSpace]; */
            /* double norm = 1.0e-8; */
            /* for (int I=0;I<nSpace;I++) */
            /*   norm += q_normal_phi[eN_k_nSpace+I]*q_normal_phi[eN_k_nSpace+I]; */
            /* norm = sqrt(norm); */
            /* for (int I=0;I<nSpace;I++) */
            /*   dir[I] = q_normal_phi[eN_k_nSpace+I]/norm; */
            /* ck.calculateGScale(G,dir,h_phi); */


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
            //calculate pde coefficients and derivatives at quadrature points
            //
            epsHeaviside=epsFactHeaviside*(useMetrics*h_phi+(1.0-useMetrics)*elementDiameter[eN]);
            epsDirac    =epsFactDirac*    (useMetrics*h_phi+(1.0-useMetrics)*elementDiameter[eN]);
            epsDiffusion=epsFactDiffusion*(useMetrics*h_phi+(1.0-useMetrics)*elementDiameter[eN]);
            //    *(useMetrics*h_phi+(1.0-useMetrics)*elementDiameter[eN]);
	    const double H_s = gf_s.H(0.0, phi_solid[eN_k]);
            evaluateCoefficients(epsHeaviside,
                                 epsDirac,
                                 q_phi[eN_k],
                                 q_H[eN_k],
                                 u,
                                 q_porosity[eN_k],
                                 r,
                                 dr);
            if (gf.useExact && dHdphi == nullptr && phi_element != nullptr)
              {
                // gf.D is the SURFACE Dirac (integrates to the interface measure); the derivative of the Heaviside with respect to the level set
                // is delta(phi) = D/|grad phi|.  |grad phi| of the element's (P1) level set phi + u from its nodal values.  Exact for the sharp
                // Heaviside when D N_i N_j is integrated exactly: equivalent polynomials of order nP need a quadrature of degree nP + 2 (= 4 here).
                double gphi[nSpace], ngphi = 0.0;
                for (int I=0;I<nSpace;I++)
                  {
                    gphi[I] = 0.0;
                    for (int jj=0;jj<nDOF_trial_element;jj++)
                      gphi[I] += phi_element[jj]*u_grad_trial[jj*nSpace+I];
                    ngphi += gphi[I]*gphi[I];
                  }
                dr /= std::max(sqrt(ngphi), 1.0e-12);
              }
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
                      H_s*((dHdphi != nullptr ? q_porosity[eN_k]*dHdphi[k*nDOF_trial_element+j]*u_test_dV[i]
                                                 : ck.ReactionJacobian_weak(dr,u_trial_ref[k*nDOF_trial_element+j],u_test_dV[i])) +
			   ck.NumericalDiffusionJacobian(epsDiffusion,&u_grad_trial[j_nSpace],&u_grad_test_dV[i_nSpace]));
                  }//j
              }//i
          }//k
      }

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
        double epsFactHeaviside = args.scalar<double>("epsFactHeaviside");
        double epsFactDirac = args.scalar<double>("epsFactDirac");
        double epsFactDiffusion = args.scalar<double>("epsFactDiffusion");
        proteus::pyarray<int>& u_l2g = args.array<int>("u_l2g");
        proteus::pyarray<int>& r_l2g = args.array<int>("r_l2g");
	proteus::pyarray<int>& elementBoundariesArray = args.array<int>("elementBoundariesArray");
        proteus::pyarray<int>& elementBoundaryElementsArray = args.array<int>("elementBoundaryElementsArray");
        proteus::pyarray<int>& elementBoundaryLocalElementBoundariesArray = args.array<int>("elementBoundaryLocalElementBoundariesArray");
        proteus::pyarray<double>& elementDiameter = args.array<double>("elementDiameter");
	proteus::pyarray<double>& elementBoundaryDiameter = args.array<double>("elementBoundaryDiameter");
        proteus::pyarray<double>& nodeDiametersArray = args.array<double>("nodeDiametersArray");
        proteus::pyarray<double>& u_dof = args.array<double>("u_dof");
        proteus::pyarray<double>& phi_dof = args.array<double>("phi_dof");
        proteus::pyarray<double>& q_phi = args.array<double>("q_phi");
        proteus::pyarray<double>& q_normal_phi = args.array<double>("q_normal_phi");
        proteus::pyarray<double>& q_H = args.array<double>("q_H");
        proteus::pyarray<double>& q_porosity = args.array<double>("q_porosity");
        proteus::pyarray<int>& csrRowIndeces_u_u = args.array<int>("csrRowIndeces_u_u");
        proteus::pyarray<int>& csrColumnOffsets_u_u = args.array<int>("csrColumnOffsets_u_u");
        proteus::pyarray<int>& csrColumnOffsets_eb_u_u = args.array<int>("csrColumnOffsets_eb_u_u");
        proteus::pyarray<double>& globalJacobian = args.array<double>("globalJacobian");
	proteus::pyarray<double>& ebqe_phi_s = args.array<double>("ebqe_phi_s");
	const proteus::pyarray<double>& phi_solid = args.array<double>("phi_solid");
	double ghost_penalty_constant = args.scalar<double>("ghost_penalty_constant");
	proteus::pyarray<double>& phi_solid_nodes = args.array<double>("phi_solid_nodes");
	bool useExact_s = args.scalar<int>("useExact_s");
	// exact (sharp) Heaviside only: 0 = analytic Jacobian (D/|grad phi| with a quadrature of degree >= nP + 2), 1 = finite difference of the
	// equivalent-polynomial weights (exact for the discrete residual at any quadrature order, 2 nDOF extra weight evaluations per cut element)
	int exact_jacobian_fd = args.scalar<int>("exact_jacobian_fd");
	proteus::pyarray<double>& isActiveR = args.array<double>("isActiveR");
	proteus::pyarray<double>& isActiveDOF = args.array<double>("isActiveDOF");
	proteus::pyarray<int>& isActiveElement = args.array<int>("isActiveElement");
        //
        //loop over elements to compute volume integrals and load them into the element Jacobians and global Jacobian
        //
        gf.useExact = useExact;
        gf_s.useExact = useExact_s;
        for(int eN=0;eN<nElements_global;eN++)
          {
            double  elementJacobian_u_u[nDOF_test_element*nDOF_trial_element],element_u[nDOF_trial_element],element_phi[nDOF_trial_element],element_phi_s[nDOF_mesh_trial_element];
            for (int j=0;j<nDOF_trial_element;j++)
              {
                int eN_j = eN*nDOF_trial_element+j;
                element_u[j] = u_dof.data()[u_l2g.data()[eN_j]];
                element_phi[j] = phi_dof.data()[u_l2g.data()[eN_j]] + element_u[j];
		element_phi_s[j] = phi_solid_nodes.data()[u_l2g.data()[eN_j]];
              }
            double element_nodes[nDOF_mesh_trial_element*3];
            for (int i=0;i<nDOF_mesh_trial_element;i++)
              {
                int eN_i=eN*nDOF_mesh_trial_element+i;
                for(int I=0;I<3;I++)
                  element_nodes[i*3 + I] = mesh_dof.data()[mesh_l2g.data()[eN_i]*3 + I];
	      }//i
	    gf.calculate(element_phi, element_nodes, x_ref.data(),false);
	    int icase_s = gf_s.calculate(element_phi_s, element_nodes, x_ref.data(), false);
	    // exact derivative of the sharp-Heaviside weights on cut elements (central difference of the equivalent-polynomial weights; the step is
	    // far below the distance of any node to the interface that matters, so the topology of the cut does not change)
	    double dHdphi_element[nQuadraturePoints_element*nDOF_trial_element];
	    const double* dHdphi_ptr = nullptr;
	    if (useExact && exact_jacobian_fd == 1)
	      {
	        int icase_e = gf.calculate(element_phi, element_nodes, x_ref.data(), false);
	        if (icase_e != 1 && icase_e != -1)
	          {
	            const double delta = 1.0e-7*elementDiameter.data()[eN];
	            double Hp[nQuadraturePoints_element], Hm[nQuadraturePoints_element], phi_pert[nDOF_trial_element];
	            for (int j=0;j<nDOF_trial_element;j++)
	              {
	                for (int jj=0;jj<nDOF_trial_element;jj++)
	                  phi_pert[jj] = element_phi[jj];
	                phi_pert[j] = element_phi[j] + delta;
	                gf.calculate(phi_pert, element_nodes, x_ref.data(), false);
	                for (int kq=0;kq<nQuadraturePoints_element;kq++) { gf.set_quad(kq); Hp[kq] = gf.H(0.0,0.0); }
	                phi_pert[j] = element_phi[j] - delta;
	                gf.calculate(phi_pert, element_nodes, x_ref.data(), false);
	                for (int kq=0;kq<nQuadraturePoints_element;kq++) { gf.set_quad(kq); Hm[kq] = gf.H(0.0,0.0); }
	                for (int kq=0;kq<nQuadraturePoints_element;kq++)
	                  dHdphi_element[kq*nDOF_trial_element+j] = (Hp[kq] - Hm[kq])/(2.0*delta);
	              }
	            gf.calculate(element_phi, element_nodes, x_ref.data(), false); // restore the unperturbed weights
	            dHdphi_ptr = dHdphi_element;
	          }
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
				     useMetrics,
				     epsFactHeaviside,
				     epsFactDirac,
				     epsFactDiffusion,
				     u_l2g.data(),
				     elementDiameter.data(),
				     nodeDiametersArray.data(),
				     u_dof.data(),
				     q_phi.data(),
				     q_normal_phi.data(),
				     q_H.data(),
				     q_porosity.data(),
				     elementJacobian_u_u,
				     element_u,
				     phi_solid.data(),
				     eN,
				     dHdphi_ptr,
				     element_phi);
	    //
	    //load into element Jacobian into global Jacobian
	    //
	    for (int i=0;i<nDOF_test_element;i++)
	      {
		int eN_i = eN*nDOF_test_element+i;
		for (int j=0;j<nDOF_trial_element;j++)
		  {
		    int eN_i_j = eN_i*nDOF_trial_element+j;

                    globalJacobian.data()[csrRowIndeces_u_u.data()[eN_i] + csrColumnOffsets_u_u.data()[eN_i_j]] += elementJacobian_u_u[i*nDOF_trial_element+j];
                  }//j
              }//i
          }//elements
	std::set<int>::iterator it=cutfem_boundaries.begin();
	while(it!=cutfem_boundaries.end())
	  {
	    std::map<int,double> DW_Dn_jump;
	    std::map<std::pair<int, int>, int> u_u_nz;
	    double gamma_cutfem=ghost_penalty_constant,
	      h_cutfem=elementBoundaryDiameter.data()[*it];
	    int eN_nDOF_trial_element  = elementBoundaryElementsArray.data()[(*it)*2+0]*nDOF_trial_element;
	    //See Massing Schott Wall 2018
	    //double norm_v=0.0;
	    //for (int i_offset=1;i_offset<nDOF_v_trial_element;i_offset++)//MSW18 is just on face
	    //  {
	    //    int i = (cutfem_local_boundaries[*it] + i_offset)%nDOF_v_trial_element;//cek hack only works for P1
	    //    double u=u_old_dof.data()[vel_l2g.data()[eN_nDOF_v_trial_element+i]],
	    //      v=v_old_dof.data()[vel_l2g.data()[eN_nDOF_v_trial_element+i]],
	    //      w=w_old_dof.data()[vel_l2g.data()[eN_nDOF_v_trial_element+i]];
	    //    norm_v=fmax(norm_v,sqrt(u*u+v*v+w*w));
	    //  }
	    ///double gamma_v_dim = rho_0*(nu_0 + norm_v*h_cutfem + alphaBDF*h_cutfem*h_cutfem);
	    //gamma_cutfem_p *= h_cutfem*h_cutfem/gamma_v_dim;
	    //if (NONCONSERVATIVE_FORM)
	    //  gamma_cutfem*=gamma_v_dim;
	    //else
	    //  gamma_cutfem*=(gamma_v_dim/rho_0);
	    for (int kb=0;kb<nQuadraturePoints_elementBoundary;kb++)
	      {
		double Du_Dn_jump=0.0, dS;
		for (int eN_side=0;eN_side < 2; eN_side++)
		  {
		    int ebN = *it,
		      eN  = elementBoundaryElementsArray.data()[ebN*2+eN_side];
		    for (int i=0;i<nDOF_test_element;i++)
		      {
			DW_Dn_jump[r_l2g.data()[eN*nDOF_test_element+i]] = 0.0;
		      }
		  }
		for (int eN_side=0;eN_side < 2; eN_side++)
		  {
		    int ebN = *it,
		      eN  = elementBoundaryElementsArray.data()[ebN*2+eN_side],
		      ebN_local = elementBoundaryLocalElementBoundariesArray.data()[ebN*2+eN_side],
		      eN_nDOF_trial_element = eN*nDOF_trial_element,
		      ebN_local_kb = ebN_local*nQuadraturePoints_elementBoundary+kb,
		      ebN_local_kb_nSpace = ebN_local_kb*nSpace;
		    double u_int=0.0,
		      grad_u_int[nSpace],
		      jac_int[nSpace*nSpace],
		      jacDet_int,
		      jacInv_int[nSpace*nSpace],
		      boundaryJac[nSpace*(nSpace-1)],
		      metricTensor[(nSpace-1)*(nSpace-1)],
		      metricTensorDetSqrt,
		      u_test_dS[nDOF_test_element],
		      u_grad_trial_trace[nDOF_trial_element*nSpace],
		      u_grad_test_dS[nDOF_trial_element*nSpace],
		      normal[nSpace],x_int,y_int,z_int,xt_int,yt_int,zt_int,integralScaling,
		      G[nSpace*nSpace],G_dd_G,tr_G,h_phi,h_penalty,penalty;
		    for (int I=0; I<nSpace;I++)
		      grad_u_int[I] = 0.0;
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
							jac_int,
							jacDet_int,
							jacInv_int,
							boundaryJac,
							metricTensor,
							metricTensorDetSqrt,
							normal_ref.data(),
							normal,
							x_int,y_int,z_int);
		    dS = metricTensorDetSqrt*dS_ref.data()[kb];
		    //compute shape and solution information
		    //shape
		    ck.gradTrialFromRef(&u_grad_trial_trace_ref.data()[ebN_local_kb_nSpace*nDOF_trial_element],jacInv_int,u_grad_trial_trace);
		    for (int i=0;i<nDOF_test_element;i++)
		      {
			int eN_i = eN*nDOF_test_element + i;
			for (int I=0;I<nSpace;I++)
			  DW_Dn_jump[r_l2g.data()[eN_i]] += u_grad_trial_trace[i*nSpace+I]*normal[I];
		      }
		  }//eN_side
		for (int eN_side=0;eN_side < 2; eN_side++)
		  {
		    int ebN = *it,
		      eN  = elementBoundaryElementsArray.data()[ebN*2+eN_side];
		    for (int i=0;i<nDOF_test_element;i++)
		      {
			int eN_i = eN*nDOF_test_element+i;
			for (int eN_side2=0;eN_side2 < 2; eN_side2++)
			  {
			    int eN2  = elementBoundaryElementsArray.data()[ebN*2+eN_side2];
			    for (int j=0;j<nDOF_test_element;j++)
			      {
				int eN_i_j = eN_i*nDOF_test_element + j;
				int eN2_j = eN2*nDOF_test_element + j;
				int ebN_i_j = ebN*4*nDOF_test_X_trial_element +
				  eN_side*2*nDOF_test_X_trial_element +
				  eN_side2*nDOF_test_X_trial_element +
				  i*nDOF_trial_element +
				  j;
				std::pair<int,int> ij = std::make_pair(u_l2g.data()[eN_i], u_l2g.data()[eN2_j]);
				if (u_u_nz.count(ij))
				  {
				    assert(u_u_nz[ij] == csrRowIndeces_u_u.data()[eN_i] + csrColumnOffsets_eb_u_u.data()[ebN_i_j]);
				  }
				else
				  u_u_nz[ij] =  csrRowIndeces_u_u.data()[eN_i] + csrColumnOffsets_eb_u_u.data()[ebN_i_j];
			      }
			  }
		      }
		  }
		for (std::map<int,double>::iterator Wi_it=DW_Dn_jump.begin(); Wi_it!=DW_Dn_jump.end(); ++Wi_it)
		  for (std::map<int,double>::iterator Wj_it=DW_Dn_jump.begin(); Wj_it!=DW_Dn_jump.end(); ++Wj_it)
		    {
		      int i_global = Wi_it->first,
			j_global = Wj_it->first;
		      double DW_Dn_jump_i = Wi_it->second,
			DW_Dn_jump_j = Wj_it->second;
		      std::pair<int,int> ij = std::make_pair(i_global, j_global);
		      globalJacobian.data()[u_u_nz.at(ij)] += gamma_cutfem*h_cutfem*DW_Dn_jump_j*DW_Dn_jump_i*dS;
		    }//i,j
	      }//kb
	    it++;
	  }//cutfem element boundaries
      }//computeJacobian
      void elementSolve(arguments_dict& args)
      {
        proteus::pyarray<double>& mesh_trial_ref = args.array<double>("mesh_trial_ref");
        proteus::pyarray<double>& mesh_grad_trial_ref = args.array<double>("mesh_grad_trial_ref");
        proteus::pyarray<double>& mesh_dof = args.array<double>("mesh_dof");
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
        double epsFactHeaviside = args.scalar<double>("epsFactHeaviside");
        double epsFactDirac = args.scalar<double>("epsFactDirac");
        double epsFactDiffusion = args.scalar<double>("epsFactDiffusion");
        proteus::pyarray<int>& u_l2g = args.array<int>("u_l2g");
        proteus::pyarray<int>& r_l2g = args.array<int>("r_l2g");
        proteus::pyarray<double>& elementDiameter = args.array<double>("elementDiameter");
        proteus::pyarray<double>& nodeDiametersArray = args.array<double>("nodeDiametersArray");
        proteus::pyarray<double>& u_dof = args.array<double>("u_dof");
        proteus::pyarray<double>& q_phi = args.array<double>("q_phi");
        proteus::pyarray<double>& q_normal_phi = args.array<double>("q_normal_phi");
        proteus::pyarray<double>& ebqe_phi = args.array<double>("ebqe_phi");
        proteus::pyarray<double>& ebqe_normal_phi = args.array<double>("ebqe_normal_phi");
        proteus::pyarray<double>& q_H = args.array<double>("q_H");
        proteus::pyarray<double>& q_u = args.array<double>("q_u");
        proteus::pyarray<double>& q_n = args.array<double>("q_n");
        proteus::pyarray<double>& ebqe_u = args.array<double>("ebqe_u");
        proteus::pyarray<double>& ebqe_n = args.array<double>("ebqe_n");
        proteus::pyarray<double>& q_r = args.array<double>("q_r");
        proteus::pyarray<double>& q_porosity = args.array<double>("q_porosity");
        int offset_u = args.scalar<int>("offset_u");
        int stride_u = args.scalar<int>("stride_u");
        proteus::pyarray<double>& globalResidual = args.array<double>("globalResidual");
        int nExteriorElementBoundaries_global = args.scalar<int>("nExteriorElementBoundaries_global");
        proteus::pyarray<int>& exteriorElementBoundariesArray = args.array<int>("exteriorElementBoundariesArray");
	proteus::pyarray<int>& elementBoundariesArray = args.array<int>("elementBoundariesArray");
        proteus::pyarray<int>& elementBoundaryElementsArray = args.array<int>("elementBoundaryElementsArray");
        proteus::pyarray<int>& elementBoundaryLocalElementBoundariesArray = args.array<int>("elementBoundaryLocalElementBoundariesArray");
	proteus::pyarray<double>& ebqe_phi_s = args.array<double>("ebqe_phi_s");
	const proteus::pyarray<double>& phi_solid = args.array<double>("phi_solid");
	proteus::pyarray<double>& phi_solid_nodes = args.array<double>("phi_solid_nodes");
	bool useExact_s = args.scalar<int>("useExact_s");
	proteus::pyarray<double>& isActiveR = args.array<double>("isActiveR");
	proteus::pyarray<double>& isActiveDOF = args.array<double>("isActiveDOF");
	proteus::pyarray<int>& isActiveElement = args.array<int>("isActiveElement");
        int maxIts = args.scalar<int>("maxIts");
        double atol = args.scalar<double>("atol");
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
            double element_u[nDOF_test_element],
              element_du[nDOF_test_element],
              elementResidual_u[nDOF_test_element],
              elementJacobian_u_u[nDOF_test_element*nDOF_trial_element],scale=1.0;
            PROTEUS_LAPACK_INTEGER elementPivots[nDOF_test_element],
              elementColPivots[nDOF_test_element];
            //double epsHeaviside,epsDirac,epsDiffusion;
	    bool element_active=true;
            for (int i=0;i<nDOF_test_element;i++)
              {
                element_u[i]=0.0;
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
                                     useMetrics,
                                     epsFactHeaviside,
                                     epsFactDirac,
                                     epsFactDiffusion,
                                     u_l2g.data(),
                                     r_l2g.data(),
                                     elementDiameter.data(),
                                     nodeDiametersArray.data(),
                                     u_dof.data(),
                                     q_phi.data(),
                                     q_normal_phi.data(),
                                     ebqe_phi.data(),
                                     ebqe_normal_phi.data(),
                                     q_H.data(),
                                     q_u.data(),
                                     q_n.data(),
                                     ebqe_u.data(),
                                     ebqe_n.data(),
                                     q_r.data(),
                                     q_porosity.data(),
                                     offset_u,stride_u,
                                     elementResidual_u,
                                     nExteriorElementBoundaries_global,
                                     exteriorElementBoundariesArray.data(),
                                     elementBoundaryElementsArray.data(),
                                     elementBoundaryLocalElementBoundariesArray.data(),
                                     element_u,
                                     eN,
				     element_active,
				     isActiveR.data(),
				     isActiveDOF.data(),
				     phi_solid.data());
            //compute l2 norm
            double resNorm=0.0;
            for (int i=0;i<nDOF_test_element;i++)
              {
                resNorm += elementResidual_u[i];
              }//i
            resNorm = fabs(resNorm);
            //now do Newton
            int its=0;
            //std::cout<<"element "<<eN<<std::endl;
            //std::cout<<"resNorm0 "<<resNorm<<std::endl;
            while (resNorm  >= atol && its < maxIts)
              {
                its+=1;
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
                                         useMetrics,
                                         epsFactHeaviside,
                                         epsFactDirac,
                                         epsFactDiffusion,
                                         u_l2g.data(),
                                         elementDiameter.data(),
                                         nodeDiametersArray.data(),
                                         u_dof.data(),
                                         q_phi.data(),
                                         q_normal_phi.data(),
                                         q_H.data(),
                                         q_porosity.data(),
                                         elementJacobian_u_u,
                                         element_u,
					 phi_solid.data(),
                                         eN);
                for (int i=0;i<nDOF_test_element;i++)
                  {
                    element_du[i] = -elementResidual_u[i];
                    elementPivots[i] = ((PROTEUS_LAPACK_INTEGER)0);
                    elementColPivots[i]=((PROTEUS_LAPACK_INTEGER)0);
                    /* std::cout<<"element jacobian"<<std::endl; */
                    /* for (int j=0;j<nDOF_test_element;j++) */
                    /*   { */
                    /*     std::cout<<elementJacobian_u_u[i*nDOF_trial_element+j]<<'\t'; */
                    /*   } */
                    /* std::cout<<std::endl; */
                  }//i
                //factor
                PROTEUS_LAPACK_INTEGER La_N=((PROTEUS_LAPACK_INTEGER)nDOF_test_element),
                  INFO=0;
                dgetc2_(&La_N,
                        elementJacobian_u_u,
                        &La_N,
                        elementPivots,
                        elementColPivots,
                        &INFO);
                //solve
                dgesc2_(&La_N,
                        elementJacobian_u_u,
                        &La_N,
                        element_du,
                        elementPivots,
                        elementColPivots,
                        &scale);
                double resNormNew = resNorm,lambda=1.0;
                int lsIts=0;
                while (resNormNew > 0.99*resNorm && lsIts < 100)
                  {
                    //apply correction
                    for (int i=0;i<nDOF_test_element;i++)
                      {
                        element_u[i] += lambda*element_du[i];
                      }//i
                    lambda /= 2.0;
                    //compute new residual
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
                                             useMetrics,
                                             epsFactHeaviside,
                                             epsFactDirac,
                                             epsFactDiffusion,
                                             u_l2g.data(),
                                             r_l2g.data(),
                                             elementDiameter.data(),
                                             nodeDiametersArray.data(),
                                             u_dof.data(),
                                             q_phi.data(),
                                             q_normal_phi.data(),
                                             ebqe_phi.data(),
                                             ebqe_normal_phi.data(),
                                             q_H.data(),
                                             q_u.data(),
                                             q_n.data(),
                                             ebqe_u.data(),
                                             ebqe_n.data(),
                                             q_r.data(),
                                             q_porosity.data(),
                                             offset_u,stride_u,
                                             elementResidual_u,
                                             nExteriorElementBoundaries_global,
                                             exteriorElementBoundariesArray.data(),
                                             elementBoundaryElementsArray.data(),
                                             elementBoundaryLocalElementBoundariesArray.data(),
                                             element_u,
                                             eN,
					     element_active,
					     isActiveR.data(),
					     isActiveDOF.data(),
					     phi_solid.data());

                    lsIts +=1;
                    //compute l2 norm
                    resNormNew=0.0;
                    for (int i=0;i<nDOF_test_element;i++)
                      {
                        resNormNew += elementResidual_u[i];
                        std::cout<<"element_u["<<i<<"] "<<element_u[i]<<std::endl;
                        std::cout<<"elementResidual_u["<<i<<"] "<<elementResidual_u[i]<<std::endl;
                      }//i
                    resNormNew = fabs(resNormNew);
                  }
                resNorm = resNormNew;
                std::cout<<"INFO "<<INFO<<std::endl;
                std::cout<<"resNorm["<<its<<"] "<<resNorm<<std::endl;
              }
          }//elements
      }
      void elementConstantSolve(arguments_dict& args)
      {
        proteus::pyarray<double>& mesh_trial_ref = args.array<double>("mesh_trial_ref");
        proteus::pyarray<double>& mesh_grad_trial_ref = args.array<double>("mesh_grad_trial_ref");
        proteus::pyarray<double>& mesh_dof = args.array<double>("mesh_dof");
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
        double epsFactHeaviside = args.scalar<double>("epsFactHeaviside");
        double epsFactDirac = args.scalar<double>("epsFactDirac");
        double epsFactDiffusion = args.scalar<double>("epsFactDiffusion");
        proteus::pyarray<int>& u_l2g = args.array<int>("u_l2g");
        proteus::pyarray<int>& r_l2g = args.array<int>("r_l2g");
        proteus::pyarray<double>& elementDiameter = args.array<double>("elementDiameter");
        proteus::pyarray<double>& nodeDiametersArray = args.array<double>("nodeDiametersArray");
        proteus::pyarray<double>& u_dof = args.array<double>("u_dof");
        proteus::pyarray<double>& q_phi = args.array<double>("q_phi");
        proteus::pyarray<double>& q_normal_phi = args.array<double>("q_normal_phi");
        proteus::pyarray<double>& ebqe_phi = args.array<double>("ebqe_phi");
        proteus::pyarray<double>& ebqe_normal_phi = args.array<double>("ebqe_normal_phi");
        proteus::pyarray<double>& q_H = args.array<double>("q_H");
        proteus::pyarray<double>& q_u = args.array<double>("q_u");
        proteus::pyarray<double>& q_n = args.array<double>("q_n");
        proteus::pyarray<double>& ebqe_u = args.array<double>("ebqe_u");
        proteus::pyarray<double>& ebqe_n = args.array<double>("ebqe_n");
        proteus::pyarray<double>& q_r = args.array<double>("q_r");
        proteus::pyarray<double>& q_porosity = args.array<double>("q_porosity");
        int offset_u = args.scalar<int>("offset_u");
        int stride_u = args.scalar<int>("stride_u");
        proteus::pyarray<double>& globalResidual = args.array<double>("globalResidual");
        int nExteriorElementBoundaries_global = args.scalar<int>("nExteriorElementBoundaries_global");
        proteus::pyarray<int>& exteriorElementBoundariesArray = args.array<int>("exteriorElementBoundariesArray");
        proteus::pyarray<int>& elementBoundaryElementsArray = args.array<int>("elementBoundaryElementsArray");
        proteus::pyarray<int>& elementBoundaryLocalElementBoundariesArray = args.array<int>("elementBoundaryLocalElementBoundariesArray");
	const proteus::pyarray<double>& phi_solid = args.array<double>("phi_solid");
	proteus::pyarray<double>& isActiveR = args.array<double>("isActiveR");
	proteus::pyarray<double>& isActiveDOF = args.array<double>("isActiveDOF");
	proteus::pyarray<int>& isActiveElement = args.array<int>("isActiveElement");
        int maxIts = args.scalar<int>("maxIts");
        double atol = args.scalar<double>("atol");
        for(int eN=0;eN<nElements_global;eN++)
          {
            //declare local storage for element residual and initialize
            double element_u[nDOF_test_element],elementConstant_u,
              elementResidual_u[nDOF_test_element],elementConstantResidual,
              elementJacobian_u_u[nDOF_test_element*nDOF_trial_element],elementConstantJacobian,resNorm;
            elementConstant_u=0.0;
	    bool element_active=true;
            for (int i=0;i<nDOF_test_element;i++)
              {
                element_u[i]=elementConstant_u;
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
                                     useMetrics,
                                     epsFactHeaviside,
                                     epsFactDirac,
                                     epsFactDiffusion,
                                     u_l2g.data(),
                                     r_l2g.data(),
                                     elementDiameter.data(),
                                     nodeDiametersArray.data(),
                                     u_dof.data(),
                                     q_phi.data(),
                                     q_normal_phi.data(),
                                     ebqe_phi.data(),
                                     ebqe_normal_phi.data(),
                                     q_H.data(),
                                     q_u.data(),
                                     q_n.data(),
                                     ebqe_u.data(),
                                     ebqe_n.data(),
                                     q_r.data(),
                                     q_porosity.data(),
                                     offset_u,stride_u,
                                     elementResidual_u,
                                     nExteriorElementBoundaries_global,
                                     exteriorElementBoundariesArray.data(),
                                     elementBoundaryElementsArray.data(),
                                     elementBoundaryLocalElementBoundariesArray.data(),
                                     element_u,
                                     eN,
				     element_active,
				     isActiveR.data(),
				     isActiveDOF.data(),
				     phi_solid.data());

            //compute l2 norm
            elementConstantResidual=0.0;
            for (int i=0;i<nDOF_test_element;i++)
              {
                elementConstantResidual += elementResidual_u[i];
              }//i
            resNorm = fabs(elementConstantResidual);
            //now do Newton
            int its=0;
            //std::cout<<"element "<<eN<<std::endl;
            //std::cout<<"resNorm0 "<<resNorm<<std::endl;
            while (resNorm >= atol && its < maxIts)
              {
                its+=1;
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
                                         useMetrics,
                                         epsFactHeaviside,
                                         epsFactDirac,
                                         epsFactDiffusion,
                                         u_l2g.data(),
                                         elementDiameter.data(),
                                         nodeDiametersArray.data(),
                                         u_dof.data(),
                                         q_phi.data(),
                                         q_normal_phi.data(),
                                         q_H.data(),
                                         q_porosity.data(),
                                         elementJacobian_u_u,
                                         element_u,
					 phi_solid.data(),
                                         eN);
                elementConstantJacobian=0.0;
                for (int i=0;i<nDOF_test_element;i++)
                  {
                    for (int j=0;j<nDOF_test_element;j++)
                      {
                        elementConstantJacobian += elementJacobian_u_u[i*nDOF_trial_element+j];
                      }
                  }//i
                std::cout<<"elementConstantJacobian "<<elementConstantJacobian<<std::endl;
                //apply correction
                elementConstant_u -= elementConstantResidual/(elementConstantJacobian+1.0e-8);
                for (int i=0;i<nDOF_test_element;i++)
                  {
                    element_u[i] = elementConstant_u;
                  }//i
                //compute new residual
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
                                         useMetrics,
                                         epsFactHeaviside,
                                         epsFactDirac,
                                         epsFactDiffusion,
                                         u_l2g.data(),
                                         r_l2g.data(),
                                         elementDiameter.data(),
                                         nodeDiametersArray.data(),
                                         u_dof.data(),
                                         q_phi.data(),
                                         q_normal_phi.data(),
                                         ebqe_phi.data(),
                                         ebqe_normal_phi.data(),
                                         q_H.data(),
                                         q_u.data(),
                                         q_n.data(),
                                         ebqe_u.data(),
                                         ebqe_n.data(),
                                         q_r.data(),
                                         q_porosity.data(),
                                         offset_u,stride_u,
                                         elementResidual_u,
                                         nExteriorElementBoundaries_global,
                                         exteriorElementBoundariesArray.data(),
                                         elementBoundaryElementsArray.data(),
                                         elementBoundaryLocalElementBoundariesArray.data(),
                                         element_u,
                                         eN,
					 element_active,
					 isActiveR.data(),
					 isActiveDOF.data(),
					 phi_solid.data());

                //compute l2 norm
                elementConstantResidual=0.0;
                for (int i=0;i<nDOF_test_element;i++)
                  {
                    elementConstantResidual += elementResidual_u[i];
                  }//i
                resNorm = fabs(elementConstantResidual);
                std::cout<<"resNorm["<<its<<"] "<<resNorm<<std::endl;
              }
          }//elements
      }

      std::tuple<double, double> globalConstantRJ(arguments_dict& args)
      {
        proteus::pyarray<double>& mesh_trial_ref = args.array<double>("mesh_trial_ref");
        proteus::pyarray<double>& mesh_grad_trial_ref = args.array<double>("mesh_grad_trial_ref");
        proteus::pyarray<double>& mesh_dof = args.array<double>("mesh_dof");
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
        int nElements_owned = args.scalar<int>("nElements_owned");
        double useMetrics = args.scalar<double>("useMetrics");
        double epsFactHeaviside = args.scalar<double>("epsFactHeaviside");
        double epsFactDirac = args.scalar<double>("epsFactDirac");
        double epsFactDiffusion = args.scalar<double>("epsFactDiffusion");
        proteus::pyarray<int>& u_l2g = args.array<int>("u_l2g");
        proteus::pyarray<int>& r_l2g = args.array<int>("r_l2g");
        proteus::pyarray<double>& elementDiameter = args.array<double>("elementDiameter");
        proteus::pyarray<double>& nodeDiametersArray = args.array<double>("nodeDiametersArray");
        proteus::pyarray<double>& u_dof = args.array<double>("u_dof");
        proteus::pyarray<double>& q_phi = args.array<double>("q_phi");
        proteus::pyarray<double>& q_normal_phi = args.array<double>("q_normal_phi");
        proteus::pyarray<double>& ebqe_phi = args.array<double>("ebqe_phi");
        proteus::pyarray<double>& ebqe_normal_phi = args.array<double>("ebqe_normal_phi");
        proteus::pyarray<double>& q_H = args.array<double>("q_H");
        proteus::pyarray<double>& q_u = args.array<double>("q_u");
        proteus::pyarray<double>& q_n = args.array<double>("q_n");
        proteus::pyarray<double>& ebqe_u = args.array<double>("ebqe_u");
        proteus::pyarray<double>& ebqe_n = args.array<double>("ebqe_n");
        proteus::pyarray<double>& q_r = args.array<double>("q_r");
        proteus::pyarray<double>& q_porosity = args.array<double>("q_porosity");
        int offset_u = args.scalar<int>("offset_u");
        int stride_u = args.scalar<int>("stride_u");
        proteus::pyarray<double>& globalResidual = args.array<double>("globalResidual");
        int nExteriorElementBoundaries_global = args.scalar<int>("nExteriorElementBoundaries_global");
        proteus::pyarray<int>& exteriorElementBoundariesArray = args.array<int>("exteriorElementBoundariesArray");
        proteus::pyarray<int>& elementBoundaryElementsArray = args.array<int>("elementBoundaryElementsArray");
        proteus::pyarray<int>& elementBoundaryLocalElementBoundariesArray = args.array<int>("elementBoundaryLocalElementBoundariesArray");
	const proteus::pyarray<double>& phi_solid = args.array<double>("phi_solid");
	proteus::pyarray<double>& isActiveR = args.array<double>("isActiveR");
	proteus::pyarray<double>& isActiveDOF = args.array<double>("isActiveDOF");
	proteus::pyarray<int>& isActiveElement = args.array<int>("isActiveElement");
        int maxIts = args.scalar<int>("maxIts");
        double atol = args.scalar<double>("atol");
        double constant_u = args.scalar<double>("constant_u");
        double element_u[nDOF_test_element],
          elementResidual_u[nDOF_test_element],
          elementJacobian_u_u[nDOF_test_element*nDOF_trial_element];
        double constantResidual = 0.0;
        double constantJacobian = 0.0;
        for (int i=0;i<nDOF_trial_element;i++)
          {
            element_u[i]=constant_u;
          }//i
        //compute residual and Jacobian
        for(int eN=0;eN<nElements_owned;eN++)
          {
	    bool element_active=true;
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
                                     nElements_owned,
                                     useMetrics,
                                     epsFactHeaviside,
                                     epsFactDirac,
                                     epsFactDiffusion,
                                     u_l2g.data(),
                                     r_l2g.data(),
                                     elementDiameter.data(),
                                     nodeDiametersArray.data(),
                                     u_dof.data(),
                                     q_phi.data(),
                                     q_normal_phi.data(),
                                     ebqe_phi.data(),
                                     ebqe_normal_phi.data(),
                                     q_H.data(),
                                     q_u.data(),
                                     q_n.data(),
                                     ebqe_u.data(),
                                     ebqe_n.data(),
                                     q_r.data(),
                                     q_porosity.data(),
                                     offset_u,stride_u,
                                     elementResidual_u,
                                     nExteriorElementBoundaries_global,
                                     exteriorElementBoundariesArray.data(),
                                     elementBoundaryElementsArray.data(),
                                     elementBoundaryLocalElementBoundariesArray.data(),
                                     element_u,
                                     eN,
				     element_active,
				     isActiveR.data(),
				     isActiveDOF.data(),
				     phi_solid.data());

            //compute l2 norm
            for (int i=0;i<nDOF_test_element;i++)
              {
                constantResidual += elementResidual_u[i];
              }//i
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
                                     nElements_owned,
                                     useMetrics,
                                     epsFactHeaviside,
                                     epsFactDirac,
                                     epsFactDiffusion,
                                     u_l2g.data(),
                                     elementDiameter.data(),
                                     nodeDiametersArray.data(),
                                     u_dof.data(),
                                     q_phi.data(),
                                     q_normal_phi.data(),
                                     q_H.data(),
                                     q_porosity.data(),
                                     elementJacobian_u_u,
                                     element_u,
				     phi_solid.data(),
                                     eN);
            for (int i=0;i<nDOF_test_element;i++)
              {
                for (int j=0;j<nDOF_test_element;j++)
                  {
                    constantJacobian += elementJacobian_u_u[i*nDOF_trial_element+j];
                  }
              }//i
          }
          return std::tuple<double, double>(constantResidual, constantJacobian);
      }

      double calculateMass(arguments_dict& args,
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
        int nElements_owned = args.scalar<int>("nElements_owned");
        double useMetrics = args.scalar<double>("useMetrics");
        double epsFactHeaviside = args.scalar<double>("epsFactHeaviside");
        double epsFactDirac = args.scalar<double>("epsFactDirac");
        double epsFactDiffusion = args.scalar<double>("epsFactDiffusion");
        proteus::pyarray<int>& u_l2g = args.array<int>("u_l2g");
        proteus::pyarray<double>& elementDiameter = args.array<double>("elementDiameter");
        proteus::pyarray<double>& nodeDiametersArray = args.array<double>("nodeDiametersArray");
        proteus::pyarray<double>& u_dof = args.array<double>("u_dof");
        proteus::pyarray<double>& phi_dof = args.array<double>("phi_dof");
        proteus::pyarray<double>& q_phi = args.array<double>("q_phi");
        proteus::pyarray<double>& q_normal_phi = args.array<double>("q_normal_phi");
        proteus::pyarray<double>& ebqe_phi = args.array<double>("ebqe_phi");
        proteus::pyarray<double>& ebqe_normal_phi = args.array<double>("ebqe_normal_phi");
        proteus::pyarray<double>& q_H = args.array<double>("q_H");
        proteus::pyarray<double>& q_u = args.array<double>("q_u");
        proteus::pyarray<double>& q_n = args.array<double>("q_n");
        proteus::pyarray<double>& ebqe_u = args.array<double>("ebqe_u");
        proteus::pyarray<double>& ebqe_n = args.array<double>("ebqe_n");
        proteus::pyarray<double>& q_r = args.array<double>("q_r");
        proteus::pyarray<double>& q_porosity = args.array<double>("q_porosity");
        int offset_u = args.scalar<int>("offset_u");
        int stride_u = args.scalar<int>("stride_u");
        proteus::pyarray<double>& globalResidual = args.array<double>("globalResidual");
        int nExteriorElementBoundaries_global = args.scalar<int>("nExteriorElementBoundaries_global");
        proteus::pyarray<int>& exteriorElementBoundariesArray = args.array<int>("exteriorElementBoundariesArray");
        proteus::pyarray<int>& elementBoundaryElementsArray = args.array<int>("elementBoundaryElementsArray");
        proteus::pyarray<int>& elementBoundaryLocalElementBoundariesArray = args.array<int>("elementBoundaryLocalElementBoundariesArray");
	const proteus::pyarray<double>& phi_solid = args.array<double>("phi_solid");
	proteus::pyarray<double>& phi_solid_nodes = args.array<double>("phi_solid_nodes");
	bool useExact_s = args.scalar<int>("useExact_s");
        double globalMass = 0.0;
        gf.useExact=useExact;
        gf_s.useExact = useExact_s;
        for(int eN=0;eN<nElements_owned;eN++)
          {
            double epsHeaviside;
            //loop over quadrature points and compute integrands
            //declare local storage for element residual and initialize
            double element_phi[nDOF_trial_element],element_phi_s[nDOF_trial_element];
            for (int i=0;i<nDOF_test_element;i++)
              {
                int eN_i=eN*nDOF_test_element+i;
                element_phi[i] = phi_dof.data()[u_l2g.data()[eN_i]];
                element_phi_s[i] = phi_solid_nodes.data()[u_l2g.data()[eN_i]];
              }//i
            double element_nodes[nDOF_mesh_trial_element*3];
            for (int i=0;i<nDOF_mesh_trial_element;i++)
              {
                int eN_i=eN*nDOF_mesh_trial_element+i;
                for(int I=0;I<3;I++)
                  element_nodes[i*3 + I] = mesh_dof.data()[mesh_l2g.data()[eN_i]*3 + I];
	      }//i
            gf.calculate(element_phi, element_nodes, x_ref.data(),false);
	    int icase_s = gf_s.calculate(element_phi_s, element_nodes, x_ref.data(),false);
	    for  (int k=0;k<nQuadraturePoints_element;k++)
	      {
		//compute indeces and declare local storage
		int eN_k = eN*nQuadraturePoints_element+k,
		  eN_k_nSpace = eN_k*nSpace;
		//eN_nDOF_trial_element = eN*nDOF_trial_element;
		//double u=0.0,grad_u[nSpace],r=0.0,dr=0.0;
		double jac[nSpace*nSpace],
		  jacDet,
		  jacInv[nSpace*nSpace],
		  //u_grad_trial[nDOF_trial_element*nSpace],
		  //u_test_dV[nDOF_trial_element],
		  //u_grad_test_dV[nDOF_test_element*nSpace],
		  dV,x,y,z,
		  G[nSpace*nSpace],G_dd_G,tr_G,h_phi;
                gf.set_quad(k);
                gf_s.set_quad(k);
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
                ck.calculateH_element(eN,
                                      k,
                                      nodeDiametersArray.data(),
                                      mesh_l2g.data(),
                                      mesh_trial_ref.data(),
                                      h_phi);
                //get the physical integration weight
                dV = fabs(jacDet)*dV_ref.data()[k];
                ck.calculateG(jacInv,G,G_dd_G,tr_G);
                /* double dir[nSpace]; */
                /* double norm = 1.0e-8; */
                /* for (int I=0;I<nSpace;I++) */
                /*        norm += q_normal_phi.data()[eN_k_nSpace+I]*q_normal_phi.data()[eN_k_nSpace+I]; */
                /* norm = sqrt(norm); */
                /* for (int I=0;I<nSpace;I++) */
                /*        dir[I] = q_normal_phi.data()[eN_k_nSpace+I]/norm; */
                /* ck.calculateGScale(G,dir,h_phi); */
                epsHeaviside=epsFactHeaviside*(useMetrics*h_phi+(1.0-useMetrics)*elementDiameter.data()[eN]);
                globalMass += gf_s.H(epsHeaviside,phi_solid.data()[eN_k])*q_porosity[eN_k]*gf.H(epsHeaviside,q_phi.data()[eN_k])*dV;
              }//k
          }//elements
          return globalMass;
      }

      void setMassQuadrature(arguments_dict& args,
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
        double epsFactHeaviside = args.scalar<double>("epsFactHeaviside");
        double epsFactDirac = args.scalar<double>("epsFactDirac");
        double epsFactDiffusion = args.scalar<double>("epsFactDiffusion");
        proteus::pyarray<int>& phi_l2g = args.array<int>("phi_l2g");
        proteus::pyarray<double>& elementDiameter = args.array<double>("elementDiameter");
        proteus::pyarray<double>& nodeDiametersArray = args.array<double>("nodeDiametersArray");
        proteus::pyarray<double>& phi_dof = args.array<double>("phi_dof");
        proteus::pyarray<double>& q_phi = args.array<double>("q_phi");
        proteus::pyarray<double>& q_normal_phi = args.array<double>("q_normal_phi");
        proteus::pyarray<double>& ebqe_phi = args.array<double>("ebqe_phi");
        proteus::pyarray<double>& ebqe_normal_phi = args.array<double>("ebqe_normal_phi");
        proteus::pyarray<double>& q_H = args.array<double>("q_H");
        proteus::pyarray<double>& q_u = args.array<double>("q_u");
        proteus::pyarray<double>& q_n = args.array<double>("q_n");
        proteus::pyarray<double>& ebqe_u = args.array<double>("ebqe_u");
        proteus::pyarray<double>& ebqe_n = args.array<double>("ebqe_n");
        proteus::pyarray<double>& q_r = args.array<double>("q_r");
        proteus::pyarray<double>& q_porosity = args.array<double>("q_porosity");
        int offset_u = args.scalar<int>("offset_u");
        int stride_u = args.scalar<int>("stride_u");
        proteus::pyarray<double>& globalResidual = args.array<double>("globalResidual");
        int nExteriorElementBoundaries_global = args.scalar<int>("nExteriorElementBoundaries_global");
        proteus::pyarray<int>& exteriorElementBoundariesArray = args.array<int>("exteriorElementBoundariesArray");
        proteus::pyarray<int>& elementBoundaryElementsArray = args.array<int>("elementBoundaryElementsArray");
        proteus::pyarray<int>& elementBoundaryLocalElementBoundariesArray = args.array<int>("elementBoundaryLocalElementBoundariesArray");
        proteus::pyarray<double>& H_dof = args.array<double>("H_dof");
        gf.useExact=useExact;
        gf_nodes.useExact=useExact;
        for(int eN=0;eN<nElements_global;eN++)
          {
            double epsHeaviside;
            //loop over quadrature points and compute integrands
            //declare local storage for element residual and initialize
            double element_phi[nDOF_trial_element];
            for (int i=0;i<nDOF_test_element;i++)
              {
                int eN_i=eN*nDOF_test_element+i;
                element_phi[i] = phi_dof.data()[phi_l2g.data()[eN_i]];
              }//i
            double element_nodes[nDOF_mesh_trial_element*3];
            for (int i=0;i<nDOF_mesh_trial_element;i++)
              {
                int eN_i=eN*nDOF_mesh_trial_element+i;
                for(int I=0;I<3;I++)
                  element_nodes[i*3 + I] = mesh_dof.data()[mesh_l2g.data()[eN_i]*3 + I];
	      }//i
            gf.calculate(element_phi, element_nodes, x_ref.data(),false);
            gf_nodes.calculate(element_phi, element_nodes, element_nodes,false);
	    for  (int k=0;k<nQuadraturePoints_element;k++)
	      {
		//compute indeces and declare local storage
		int eN_k = eN*nQuadraturePoints_element+k,
		  eN_k_nSpace = eN_k*nSpace;
		//eN_nDOF_trial_element = eN*nDOF_trial_element;
		//double u=0.0,grad_u[nSpace],r=0.0,dr=0.0;
		double jac[nSpace*nSpace],
		  jacDet,
		  jacInv[nSpace*nSpace],
		  //u_grad_trial[nDOF_trial_element*nSpace],
		  //u_test_dV[nDOF_trial_element],
		  //u_grad_test_dV[nDOF_test_element*nSpace],
		  dV,x,y,z,
		  G[nSpace*nSpace],G_dd_G,tr_G,h_phi;
		//
		//compute solution and gradients at quadrature points
		//
                gf.set_quad(k);
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

                /* double dir[nSpace]; */
                /* double norm = 1.0e-8; */
                /* for (int I=0;I<nSpace;I++) */
                /*        norm += q_normal_phi.data()[eN_k_nSpace+I]*q_normal_phi.data()[eN_k_nSpace+I]; */
                /* norm = sqrt(norm); */
                /* for (int I=0;I<nSpace;I++) */
                /*        dir[I] = q_normal_phi.data()[eN_k_nSpace+I]/norm; */

                /* ck.calculateGScale(G,dir,h_phi); */
                epsHeaviside=epsFactHeaviside*(useMetrics*h_phi+(1.0-useMetrics)*elementDiameter.data()[eN]);
                q_H.data()[eN_k] = gf.H(epsHeaviside,q_phi.data()[eN_k]);
              }//k
            // distribute rhs for mass correction
            for (int i=0;i<nDOF_trial_element;i++)
              {
                gf_nodes.set_quad(i);
                int eN_i = eN*nDOF_trial_element + i;
                int gi = phi_l2g.data()[eN_i];
                epsHeaviside = epsFactHeaviside*nodeDiametersArray.data()[mesh_l2g.data()[eN_i]];//cek hack, only works if isoparametric, but we can fix by including interpolation points
                H_dof.data() [gi] = gf_nodes.H(epsHeaviside,phi_dof.data()[gi]);
              }
          }//elements
      }

      void FCTStep(arguments_dict& args)
      {
        int NNZ = args.scalar<int>("NNZ");
        int numDOFs = args.scalar<int>("numDOFs");
        proteus::pyarray<double>& lumped_mass_matrix = args.array<double>("lumped_mass_matrix");
        proteus::pyarray<double>& solH = args.array<double>("solH");
        proteus::pyarray<double>& solL = args.array<double>("solL");
        proteus::pyarray<double>& limited_solution = args.array<double>("limited_solution");
        proteus::pyarray<int>& csrRowIndeces_DofLoops = args.array<int>("csrRowIndeces_DofLoops");
        proteus::pyarray<int>& csrColumnOffsets_DofLoops = args.array<int>("csrColumnOffsets_DofLoops");
        proteus::pyarray<double>& MassMatrix = args.array<double>("matrix");
        // phase 0 computes the limiter coefficients R of every dof, phase 1 applies them. The limiter of an
        // owned dof reads R of its neighbours, whose graph rows are truncated when they are ghosts, so
        // Python scatters R from the owners between the two phases.
        int phase = args.scalar<int>("phase");
        proteus::pyarray<double>& Rpos_dof = args.array<double>("Rpos_dof");
        proteus::pyarray<double>& Rneg_dof = args.array<double>("Rneg_dof");
        if (phase == 0)
          {
            // resize also zeroes, so phase 1 must not do it: it needs the matrix phase 0 filled
            Rpos.resize(numDOFs,0.0), Rneg.resize(numDOFs,0.0);
            FluxCorrectionMatrix.resize(NNZ,0.0);
          }
        //////////////////
        // LOOP in DOFs //
        //////////////////
        int ij=0;
        if (phase == 0)
        for (int i=0; i<numDOFs; i++)
          {
            //read some vectors
            double solHi = solH.data()[i];
            double solLi = solL.data()[i];
            double mi = lumped_mass_matrix.data()[i];

            double mini=0., maxi=1.0;
            double Pposi=0, Pnegi=0;
            // LOOP OVER THE SPARSITY PATTERN (j-LOOP)//
            for (int offset=csrRowIndeces_DofLoops.data()[i]; offset<csrRowIndeces_DofLoops.data()[i+1]; offset++)
              {
                int j = csrColumnOffsets_DofLoops.data()[offset];
                // i-th row of flux correction matrix
                FluxCorrectionMatrix[ij] = ((i==j ? 1. : 0.)*mi - MassMatrix.data()[ij]) * (solH.data()[j]-solHi);

                ///////////////////////
                // COMPUTE P VECTORS //
                ///////////////////////
                Pposi += FluxCorrectionMatrix[ij]*((FluxCorrectionMatrix[ij] > 0) ? 1. : 0.);
                Pnegi += FluxCorrectionMatrix[ij]*((FluxCorrectionMatrix[ij] < 0) ? 1. : 0.);

                //update ij
                ij+=1;
              }
            ///////////////////////
            // COMPUTE Q VECTORS //
            ///////////////////////
            double Qposi = mi*(maxi-solLi);
            double Qnegi = mi*(mini-solLi);

            ///////////////////////
            // COMPUTE R VECTORS //
            ///////////////////////
            Rpos[i] = ((Pposi==0) ? 1. : std::min(1.0,Qposi/Pposi));
            Rneg[i] = ((Pnegi==0) ? 1. : std::min(1.0,Qnegi/Pnegi));
            Rpos_dof.data()[i] = Rpos[i];
            Rneg_dof.data()[i] = Rneg[i];
          } // i DOFs

        //////////////////////
        // COMPUTE LIMITERS //
        //////////////////////
        ij=0;
        if (phase == 1)
          for (int i=0; i<numDOFs; i++)
            {
              Rpos[i] = Rpos_dof.data()[i];
              Rneg[i] = Rneg_dof.data()[i];
            }
        if (phase == 1)
        for (int i=0; i<numDOFs; i++)
          {
            double ith_Limiter_times_FluxCorrectionMatrix = 0.;
            double Rposi = Rpos[i], Rnegi = Rneg[i];
            // LOOP OVER THE SPARSITY PATTERN (j-LOOP)//
            for (int offset=csrRowIndeces_DofLoops.data()[i]; offset<csrRowIndeces_DofLoops.data()[i+1]; offset++)
              {
                int j = csrColumnOffsets_DofLoops.data()[offset];
                ith_Limiter_times_FluxCorrectionMatrix +=
                  ((FluxCorrectionMatrix[ij]>0) ? std::min(Rposi,Rneg[j]) : std::min(Rnegi,Rpos[j]))
                  * FluxCorrectionMatrix[ij];
                //ith_Limiter_times_FluxCorrectionMatrix += FluxCorrectionMatrix[ij];
                //update ij
                ij+=1;
              }
            limited_solution.data()[i] = fmax(0.0,solL.data()[i] + 1./lumped_mass_matrix.data()[i]*ith_Limiter_times_FluxCorrectionMatrix);
          }
      }

      // mql. copied from calculateElementJacobian. NOTE: there are some not necessary computations!!!
      inline void calculateElementMassMatrix(//element
                                             double* mesh_trial_ref,
                                             double* mesh_grad_trial_ref,
                                             double* mesh_dof,
                                             int* mesh_l2g,
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
                                             int nElements_global,
                                             double useMetrics,
                                             double epsFactHeaviside,
                                             double epsFactDirac,
                                             double epsFactDiffusion,
                                             int* u_l2g,
                                             double* elementDiameter,
                                             double* nodeDiametersArray,
                                             double* u_dof,
                                             // double* u_trial,
                                             // double* u_grad_trial,
                                             // double* u_test_dV,
                                             // double* u_grad_test_dV,
                                             double* q_phi,
                                             double* q_normal_phi,
                                             double* q_H,
                                             double* q_porosity,
                                             double* elementMassMatrix,
                                             double* elementLumpedMassMatrix,
                                             double* element_u,
                                             int eN)
      {
        for (int i=0;i<nDOF_test_element;i++)
          {
            elementLumpedMassMatrix[i] = 0.0;
            for (int j=0;j<nDOF_trial_element;j++)
              {
                elementMassMatrix[i*nDOF_trial_element+j]=0.0;
              }
          }
        double epsHeaviside,epsDirac,epsDiffusion;
        for  (int k=0;k<nQuadraturePoints_element;k++)
          {
            int eN_k = eN*nQuadraturePoints_element+k, //index to a scalar at a quadrature point
              eN_k_nSpace = eN_k*nSpace;
            //eN_nDOF_trial_element = eN*nDOF_trial_element; //index to a vector at a quadrature point

            //declare local storage
            double u=0.0,
              grad_u[nSpace],
              r=0.0,dr=0.0,
              jac[nSpace*nSpace],
              jacDet,
              jacInv[nSpace*nSpace],
              u_grad_trial[nDOF_trial_element*nSpace],
              dV,
              u_test_dV[nDOF_test_element],
              u_grad_test_dV[nDOF_test_element*nSpace],
              x,y,z,
              G[nSpace*nSpace],G_dd_G,tr_G,h_phi;
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
            ck.calculateH_element(eN,
                                  k,
                                  nodeDiametersArray,
                                  mesh_l2g,
                                  mesh_trial_ref,
                                  h_phi);
            //get the physical integration weight
            dV = fabs(jacDet)*dV_ref[k];
            ck.calculateG(jacInv,G,G_dd_G,tr_G);

            /* double dir[nSpace]; */
            /* double norm = 1.0e-8; */
            /* for (int I=0;I<nSpace;I++) */
            /*   norm += q_normal_phi[eN_k_nSpace+I]*q_normal_phi[eN_k_nSpace+I]; */
            /* norm = sqrt(norm); */
            /* for (int I=0;I<nSpace;I++) */
            /*   dir[I] = q_normal_phi[eN_k_nSpace+I]/norm; */
            /* ck.calculateGScale(G,dir,h_phi); */


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
            //calculate pde coefficients and derivatives at quadrature points
            //
            epsHeaviside=epsFactHeaviside*(useMetrics*h_phi+(1.0-useMetrics)*elementDiameter[eN]);
            epsDirac    =epsFactDirac*    (useMetrics*h_phi+(1.0-useMetrics)*elementDiameter[eN]);
            epsDiffusion=epsFactDiffusion*(useMetrics*h_phi+(1.0-useMetrics)*elementDiameter[eN]);
            //    *(useMetrics*h_phi+(1.0-useMetrics)*elementDiameter[eN]);
            evaluateCoefficients(epsHeaviside,
                                 epsDirac,
                                 q_phi[eN_k],
                                 q_H[eN_k],
                                 u,
                                 q_porosity[eN_k],
                                 r,
                                 dr);
            for(int i=0;i<nDOF_test_element;i++)
              {
                //int eN_k_i=eN_k*nDOF_test_element+i;
                //int eN_k_i_nSpace=eN_k_i*nSpace;
                elementLumpedMassMatrix[i] += u_test_dV[i];
                for(int j=0;j<nDOF_trial_element;j++)
                  {
                    elementMassMatrix[i*nDOF_trial_element+j] += u_trial_ref[k*nDOF_trial_element+j]*u_test_dV[i];
                  }//j
              }//i
          }//k
      }

      void calculateMassMatrix(arguments_dict& args)
      {
        proteus::pyarray<double>& mesh_trial_ref = args.array<double>("mesh_trial_ref");
        proteus::pyarray<double>& mesh_grad_trial_ref = args.array<double>("mesh_grad_trial_ref");
        proteus::pyarray<double>& mesh_dof = args.array<double>("mesh_dof");
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
        double epsFactHeaviside = args.scalar<double>("epsFactHeaviside");
        double epsFactDirac = args.scalar<double>("epsFactDirac");
        double epsFactDiffusion = args.scalar<double>("epsFactDiffusion");
        proteus::pyarray<int>& u_l2g = args.array<int>("u_l2g");
        proteus::pyarray<double>& elementDiameter = args.array<double>("elementDiameter");
        proteus::pyarray<double>& nodeDiametersArray = args.array<double>("nodeDiametersArray");
        proteus::pyarray<double>& u_dof = args.array<double>("u_dof");
        proteus::pyarray<double>& q_phi = args.array<double>("q_phi");
        proteus::pyarray<double>& q_normal_phi = args.array<double>("q_normal_phi");
        proteus::pyarray<double>& q_H = args.array<double>("q_H");
        proteus::pyarray<double>& q_porosity = args.array<double>("q_porosity");
        proteus::pyarray<int>& csrRowIndeces_u_u = args.array<int>("csrRowIndeces_u_u");
        proteus::pyarray<int>& csrColumnOffsets_u_u = args.array<int>("csrColumnOffsets_u_u");
        proteus::pyarray<double>& globalMassMatrix = args.array<double>("globalMassMatrix");
        proteus::pyarray<double>& globalLumpedMassMatrix = args.array<double>("globalLumpedMassMatrix");
        //
        //loop over elements to compute volume integrals and load them into the element Jacobians and global Jacobian
        //
        for(int eN=0;eN<nElements_global;eN++)
          {
            double  elementMassMatrix[nDOF_test_element*nDOF_trial_element],element_u[nDOF_trial_element], elementLumpedMassMatrix[nDOF_trial_element];
            for (int j=0;j<nDOF_trial_element;j++)
              {
                int eN_j = eN*nDOF_trial_element+j;
                element_u[j] = u_dof.data()[u_l2g.data()[eN_j]];
              }
            calculateElementMassMatrix(mesh_trial_ref.data(),
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
                                       useMetrics,
                                       epsFactHeaviside,
                                       epsFactDirac,
                                       epsFactDiffusion,
                                       u_l2g.data(),
                                       elementDiameter.data(),
                                       nodeDiametersArray.data(),
                                       u_dof.data(),
                                       q_phi.data(),
                                       q_normal_phi.data(),
                                       q_H.data(),
                                       q_porosity.data(),
                                       elementMassMatrix,
                                       elementLumpedMassMatrix,
                                       element_u,
                                       eN);
            //
            //load into element Jacobian into global Jacobian
            //
            for (int i=0;i<nDOF_test_element;i++)
              {
                int eN_i = eN*nDOF_test_element+i;
                int gi = u_l2g.data()[eN_i];
                globalLumpedMassMatrix.data()[gi] += elementLumpedMassMatrix[i];
                for (int j=0;j<nDOF_trial_element;j++)
                  {
                    int eN_i_j = eN_i*nDOF_trial_element+j;
                    globalMassMatrix.data()[csrRowIndeces_u_u.data()[eN_i] + csrColumnOffsets_u_u.data()[eN_i_j]] +=
                      elementMassMatrix[i*nDOF_trial_element+j];
                  }//j
              }//i
          }//elements
      }//calculate mass matrix

      void setMassQuadratureEdgeBasedStabilizationMethods(arguments_dict& args,
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
        double epsFactHeaviside = args.scalar<double>("epsFactHeaviside");
        double epsFactDirac = args.scalar<double>("epsFactDirac");
        double epsFactDiffusion = args.scalar<double>("epsFactDiffusion");
        proteus::pyarray<int>& phi_l2g = args.array<int>("phi_l2g");
        proteus::pyarray<double>& elementDiameter = args.array<double>("elementDiameter");
        proteus::pyarray<double>& nodeDiametersArray = args.array<double>("nodeDiametersArray");
        proteus::pyarray<double>& phi_dof = args.array<double>("phi_dof");
        proteus::pyarray<double>& q_phi = args.array<double>("q_phi");
        proteus::pyarray<double>& q_normal_phi = args.array<double>("q_normal_phi");
        proteus::pyarray<double>& ebqe_phi = args.array<double>("ebqe_phi");
        proteus::pyarray<double>& ebqe_normal_phi = args.array<double>("ebqe_normal_phi");
        proteus::pyarray<double>& q_H = args.array<double>("q_H");
        proteus::pyarray<double>& q_u = args.array<double>("q_u");
        proteus::pyarray<double>& q_n = args.array<double>("q_n");
        proteus::pyarray<double>& ebqe_u = args.array<double>("ebqe_u");
        proteus::pyarray<double>& ebqe_n = args.array<double>("ebqe_n");
        proteus::pyarray<double>& q_r = args.array<double>("q_r");
        proteus::pyarray<double>& q_porosity = args.array<double>("q_porosity");
        int offset_u = args.scalar<int>("offset_u");
        int stride_u = args.scalar<int>("stride_u");
        proteus::pyarray<double>& globalResidual = args.array<double>("globalResidual");
        int nExteriorElementBoundaries_global = args.scalar<int>("nExteriorElementBoundaries_global");
        proteus::pyarray<int>& exteriorElementBoundariesArray = args.array<int>("exteriorElementBoundariesArray");
        proteus::pyarray<int>& elementBoundaryElementsArray = args.array<int>("elementBoundaryElementsArray");
        proteus::pyarray<int>& elementBoundaryLocalElementBoundariesArray = args.array<int>("elementBoundaryLocalElementBoundariesArray");
        proteus::pyarray<double>& rhs_mass_correction = args.array<double>("rhs_mass_correction");
        proteus::pyarray<double>& lumped_L2p_vof_mass_correction = args.array<double>("lumped_L2p_vof_mass_correction");
        proteus::pyarray<double>& lumped_mass_matrix = args.array<double>("lumped_mass_matrix");
        int numDOFs = args.scalar<int>("numDOFs");
        gf.useExact=useExact;
        for(int eN=0;eN<nElements_global;eN++)
          {
            double element_rhs_mass_correction[nDOF_test_element];
            for (int i=0;i<nDOF_test_element;i++)
              element_rhs_mass_correction[i] = 0.;
            double epsHeaviside;
            //loop over quadrature points and compute integrands
            double element_phi[nDOF_trial_element];
            for (int i=0;i<nDOF_test_element;i++)
              {
                int eN_i=eN*nDOF_test_element+i;
                element_phi[i] = phi_dof.data()[phi_l2g.data()[eN_i]];
              }//i
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
		//compute indeces and declare local storage
		int eN_k = eN*nQuadraturePoints_element+k,
		  eN_k_nSpace = eN_k*nSpace;
		//eN_nDOF_trial_element = eN*nDOF_trial_element;
		//double u=0.0,grad_u[nSpace],r=0.0,dr=0.0;
		double jac[nSpace*nSpace],
		  jacDet,
		  jacInv[nSpace*nSpace],
		  //u_grad_trial[nDOF_trial_element*nSpace],
		  //u_test_dV[nDOF_trial_element],
		  //u_grad_test_dV[nDOF_test_element*nSpace],
		  dV,x,y,z,
		  u_test_dV[nDOF_test_element],
		  G[nSpace*nSpace],G_dd_G,tr_G,h_phi;
		gf.set_quad(k);
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

                // precalculate test function times integration weight
                for (int j=0;j<nDOF_trial_element;j++)
                  u_test_dV[j] = u_test_ref.data()[k*nDOF_trial_element+j]*dV;

                /* double dir[nSpace]; */
                /* double norm = 1.0e-8; */
                /* for (int I=0;I<nSpace;I++) */
                /*        norm += q_normal_phi.data()[eN_k_nSpace+I]*q_normal_phi.data()[eN_k_nSpace+I]; */
                /* norm = sqrt(norm); */
                /* for (int I=0;I<nSpace;I++) */
                /*        dir[I] = q_normal_phi.data()[eN_k_nSpace+I]/norm; */

                /* ck.calculateGScale(G,dir,h_phi); */
                epsHeaviside=epsFactHeaviside*(useMetrics*h_phi+(1.0-useMetrics)*elementDiameter.data()[eN]);
                q_H.data()[eN_k] = gf.H(epsHeaviside,q_phi.data()[eN_k]);

                for (int i=0;i<nDOF_trial_element;i++)
                  element_rhs_mass_correction [i] += q_porosity.data()[eN_k]*q_H.data()[eN_k]*u_test_dV[i];
              }//k
            // distribute rhs for mass correction
            for (int i=0;i<nDOF_trial_element;i++)
              {
                int eN_i = eN*nDOF_trial_element + i;
                int gi = phi_l2g.data()[eN_i];
                rhs_mass_correction.data()[gi] += element_rhs_mass_correction[i];
              }
          }//elements
        // COMPUTE LUMPED L2 PROYJECTION
        for (int i=0; i<numDOFs; i++)
          {
            double mi = lumped_mass_matrix.data()[i];
            lumped_L2p_vof_mass_correction.data()[i] = 1./mi*rhs_mass_correction.data()[i];
          }
      }
    };//MCorr

  inline MCorr_base* newMCorr(int nSpaceIn,
                              int nQuadraturePoints_elementIn,
                              int nDOF_mesh_trial_elementIn,
                              int nDOF_trial_elementIn,
                              int nDOF_test_elementIn,
                              int nQuadraturePoints_elementBoundaryIn,
                              int CompKernelFlag)
  {
    if (nSpaceIn == 2)
      return proteus::chooseAndAllocateDiscretization2D<MCorr_base,MCorr,CompKernel>(nSpaceIn,
                                                                                     nQuadraturePoints_elementIn,
                                                                                     nDOF_mesh_trial_elementIn,
                                                                                     nDOF_trial_elementIn,
                                                                                     nDOF_test_elementIn,
                                                                                     nQuadraturePoints_elementBoundaryIn,
                                                                                     CompKernelFlag);
    else
      return proteus::chooseAndAllocateDiscretization<MCorr_base,MCorr,CompKernel>(nSpaceIn,
                                                                                   nQuadraturePoints_elementIn,
                                                                                   nDOF_mesh_trial_elementIn,
                                                                                   nDOF_trial_elementIn,
                                                                                   nDOF_test_elementIn,
                                                                                   nQuadraturePoints_elementBoundaryIn,
                                                                                   CompKernelFlag);
  }
}//proteus
#endif
