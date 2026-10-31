#ifndef VOF_H
#define VOF_H
#include <cmath>
#include <iostream>
#include <set>
#include <map>
#include <valarray>
#include "CompKernel.h"
#include "ModelFactory.h"
#include "equivalent_polynomials.h"
#include "ArgumentsDict.h"
#include "pyarray.h"

namespace py = pybind11;

#define POWER_SMOOTHNESS_INDICATOR 2
#define IS_BETAij_ONE 0
#define GLOBAL_FCT 0

// Cell based methods:
//    * SUPG with BDF1 or BDF2 time integration
//    * Explicit Taylor Galerkin with EV stabilization
// Edge based methods.
//    Low order via D. Kuzmin's
//    High order methods: Smoothness indicator with MC, EV commutator with MC, D.K with ML
//    Zalesak's FCT

namespace proteus
{
  // Power entropy //
  inline double ENTROPY(const double& phi, const double& phiL, const double& phiR){
    return 1./2.*std::pow(fabs(phi),2.);
  }
  inline double DENTROPY(const double& phi, const double& phiL, const double& phiR){
    return fabs(phi)*(phi>=0 ? 1 : -1);
  }
  // Log entropy // for level set from 0 to 1
  inline double ENTROPY_LOG(const double& phi, const double& phiL, const double& phiR){
    return std::log(fabs((phi-phiL)*(phiR-phi))+1E-14);
  }
  inline double DENTROPY_LOG(const double& phi, const double& phiL, const double& phiR){
    return (phiL+phiR-2*phi)*((phi-phiL)*(phiR-phi)>=0 ? 1 : -1)/(fabs((phi-phiL)*(phiR-phi))+1E-14);
  }
}

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

  class VOF_base
  {
    //The base class defining the interface
  public:
    std::valarray<double> Rpos, Rneg;
    std::valarray<double> FluxCorrectionMatrix;
    std::valarray<double> TransportMatrix, TransposeTransportMatrix;
    std::valarray<double> psi, eta, global_entropy_residual, boundary_integral;
    std::valarray<double> maxVel,maxEntRes;
    virtual ~VOF_base(){}
    virtual void calculateResidualElementBased(arguments_dict& args)=0;
    virtual void calculateJacobian(arguments_dict& args)=0;
    virtual void FCTStep(arguments_dict& args)=0;
    virtual void calculateResidualEdgeBased(arguments_dict& args)=0;
  };

  template<class CompKernelType,
    int nSpace,
    int nQuadraturePoints_element,
    int nDOF_mesh_trial_element,
    int nDOF_trial_element,
    int nDOF_test_element,
    int nQuadraturePoints_elementBoundary>
    class VOF : public VOF_base
    {
    public:
      std::set<int> cutfem_boundaries;
      std::map<int, int> cutfem_local_boundaries;
      const int nDOF_test_X_trial_element;
      CompKernelType ck;
      GeneralizedFunctions<nSpace,2,nQuadraturePoints_element,nQuadraturePoints_elementBoundary> gf_s;
  VOF():
      nDOF_test_X_trial_element(nDOF_test_element*nDOF_trial_element),
        ck()
          {}

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
        void evaluateCoefficients(const double v[nSpace],
                                  const double& u,
                                  const double& porosity, //VRANS specific
                                  double& m,
                                  double& dm,
                                  double f[nSpace],
                                  double df[nSpace])
      {
        m = porosity*u;
        dm= porosity;
        for (int I=0; I < nSpace; I++)
          {
            f[I] = v[I]*porosity*u;
            df[I] = v[I]*porosity;
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

      inline
        void calculateNumericalDiffusion(const double& shockCapturingDiffusion,
                                         const double& elementDiameter,
                                         const double& strong_residual,
                                         const double grad_u[nSpace],
                                         double& numDiff)
      {
        double h,
          num,
          den,
          n_grad_u;
        h = elementDiameter;
        n_grad_u = 0.0;
        for (int I=0;I<nSpace;I++)
          n_grad_u += grad_u[I]*grad_u[I];
        num = shockCapturingDiffusion*0.5*h*fabs(strong_residual);
        den = sqrt(n_grad_u) + 1.0e-8;
        numDiff = num/den;
      }

      inline
        void exteriorNumericalAdvectiveFlux(const int& isDOFBoundary_u,
                                            const int& isFluxBoundary_u,
                                            const double n[nSpace],
                                            const double& bc_u,
                                            const double& bc_flux_u,
                                            const double& u,
                                            const double velocity[nSpace],
                                            double& flux)
      {

        double flow=0.0;
        for (int I=0; I < nSpace; I++)
          flow += n[I]*velocity[I];
        if (isDOFBoundary_u == 1)
          {
            if (flow >= 0.0)
              {
                flux = u*flow;
              }
            else
              {
                flux = bc_u*flow;
              }
          }
        else if (isFluxBoundary_u == 1)
          {
            flux = bc_flux_u;
          }
        else
          {
            if (flow >= 0.0)
              {
                flux = u*flow;
              }
            else
              {
                std::cout<<"warning: VOF open boundary with no external trace, setting to zero for inflow"<<std::endl;
                flux = 0.0;
              }

          }
      }

      inline
        void exteriorNumericalAdvectiveFluxDerivative(const int& isDOFBoundary_u,
                                                      const int& isFluxBoundary_u,
                                                      const double n[nSpace],
                                                      const double velocity[nSpace],
                                                      double& dflux)
      {
        double flow=0.0;
        for (int I=0; I < nSpace; I++)
          flow += n[I]*velocity[I];
        dflux=0.0;//default to no flux
        if (isDOFBoundary_u == 1)
          {
            if (flow >= 0.0)
              {
                dflux = flow;
              }
            else
              {
                dflux = 0.0;
              }
          }
        else if (isFluxBoundary_u == 1)
          {
            dflux = 0.0;
          }
        else
          {
            if (flow >= 0.0)
              {
                dflux = flow;
              }
          }
      }

      void calculateResidualElementBased(arguments_dict& args)
      {
        double dt = args.scalar<double>("dt");
        double sc_drop_time = args.scalar<double>("sc_drop_time");
        proteus::pyarray<double>& mesh_trial_ref = args.array<double>("mesh_trial_ref");
        proteus::pyarray<double>& mesh_grad_trial_ref = args.array<double>("mesh_grad_trial_ref");
        proteus::pyarray<double>& mesh_dof = args.array<double>("mesh_dof");
        proteus::pyarray<double>& mesh_velocity_dof = args.array<double>("mesh_velocity_dof");
        double MOVING_DOMAIN = args.scalar<double>("MOVING_DOMAIN");
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
        int lag_shockCapturing = args.scalar<int>("lag_shockCapturing");
        double shockCapturingDiffusion = args.scalar<double>("shockCapturingDiffusion");
        double sc_uref = args.scalar<double>("sc_uref");
        double sc_alpha = args.scalar<double>("sc_alpha");
        const proteus::pyarray<double>& q_porosity = args.array<double>("q_porosity");
        const proteus::pyarray<double>& porosity_dof = args.array<double>("porosity_dof");
        proteus::pyarray<int>& u_l2g = args.array<int>("u_l2g");
        proteus::pyarray<int>& r_l2g = args.array<int>("r_l2g");
        proteus::pyarray<double>& elementDiameter = args.array<double>("elementDiameter");
	proteus::pyarray<double>& elementBoundaryDiameter = args.array<double>("elementBoundaryDiameter");
        double degree_polynomial = args.scalar<double>("degree_polynomial");
        proteus::pyarray<double>& u_dof = args.array<double>("u_dof");
        proteus::pyarray<double>& u_dof_old = args.array<double>("u_dof_old");
        proteus::pyarray<double>& velocity = args.array<double>("velocity");
        proteus::pyarray<double>& q_m = args.array<double>("q_m");
        proteus::pyarray<double>& q_u = args.array<double>("q_u");
        proteus::pyarray<double>& q_m_betaBDF = args.array<double>("q_m_betaBDF");
        proteus::pyarray<double>& q_dV = args.array<double>("q_dV");
        proteus::pyarray<double>& q_dV_last = args.array<double>("q_dV_last");
        proteus::pyarray<double>& cfl = args.array<double>("cfl");
        proteus::pyarray<double>& edge_based_cfl = args.array<double>("edge_based_cfl");
        proteus::pyarray<double>& q_numDiff_u = args.array<double>("q_numDiff_u");
        proteus::pyarray<double>& q_numDiff_u_last = args.array<double>("q_numDiff_u_last");
        int offset_u = args.scalar<int>("offset_u");
        int stride_u = args.scalar<int>("stride_u");
        proteus::pyarray<int>& csrRowIndeces_u_u = args.array<int>("csrRowIndeces_u_u");
        proteus::pyarray<int>& csrColumnOffsets_u_u = args.array<int>("csrColumnOffsets_u_u");
        proteus::pyarray<int>& csrColumnOffsets_eb_u_u = args.array<int>("csrColumnOffsets_eb_u_u");
        proteus::pyarray<double>& globalResidual = args.array<double>("globalResidual");
        int nExteriorElementBoundaries_global = args.scalar<int>("nExteriorElementBoundaries_global");
        proteus::pyarray<int>& exteriorElementBoundariesArray = args.array<int>("exteriorElementBoundariesArray");
	proteus::pyarray<int>& elementBoundariesArray = args.array<int>("elementBoundariesArray");
        proteus::pyarray<int>& elementBoundaryElementsArray = args.array<int>("elementBoundaryElementsArray");
        proteus::pyarray<int>& elementBoundaryLocalElementBoundariesArray = args.array<int>("elementBoundaryLocalElementBoundariesArray");
        proteus::pyarray<double>& ebqe_velocity_ext = args.array<double>("ebqe_velocity_ext");
        const proteus::pyarray<double>& ebqe_porosity_ext = args.array<double>("ebqe_porosity_ext");
        proteus::pyarray<int>& isDOFBoundary_u = args.array<int>("isDOFBoundary_u");
        proteus::pyarray<double>& ebqe_bc_u_ext = args.array<double>("ebqe_bc_u_ext");
        proteus::pyarray<int>& isFluxBoundary_u = args.array<int>("isFluxBoundary_u");
        proteus::pyarray<double>& ebqe_bc_flux_u_ext = args.array<double>("ebqe_bc_flux_u_ext");
        proteus::pyarray<double>& ebqe_phi = args.array<double>("ebqe_phi");
        double epsFact = args.scalar<double>("epsFact");
        proteus::pyarray<double>& ebqe_u = args.array<double>("ebqe_u");
        proteus::pyarray<double>& ebqe_flux = args.array<double>("ebqe_flux");
        int stage = args.scalar<int>("stage");
        proteus::pyarray<double>&  uTilde_dof = args.array<double>("uTilde_dof");
        double cE = args.scalar<double>("cE");
        double cMax = args.scalar<double>("cMax");
        double cK = args.scalar<double>("cK");
        double uL = args.scalar<double>("uL");
        double uR = args.scalar<double>("uR");
        int numDOFs = args.scalar<int>("numDOFs");
        int NNZ = args.scalar<int>("NNZ");
        proteus::pyarray<int>& csrRowIndeces_DofLoops = args.array<int>("csrRowIndeces_DofLoops");
        proteus::pyarray<int>& csrColumnOffsets_DofLoops = args.array<int>("csrColumnOffsets_DofLoops");
        proteus::pyarray<int>& csrRowIndeces_CellLoops = args.array<int>("csrRowIndeces_CellLoops");
        proteus::pyarray<int>& csrColumnOffsets_CellLoops = args.array<int>("csrColumnOffsets_CellLoops");
        proteus::pyarray<int>& csrColumnOffsets_eb_CellLoops = args.array<int>("csrColumnOffsets_eb_CellLoops");
        proteus::pyarray<double>& ML = args.array<double>("ML");
        int LUMPED_MASS_MATRIX = args.scalar<int>("LUMPED_MASS_MATRIX");
        int STABILIZATION_TYPE = args.scalar<int>("STABILIZATION_TYPE");
        int ENTROPY_TYPE = args.scalar<int>("ENTROPY_TYPE");
        proteus::pyarray<double>& uLow = args.array<double>("uLow");
        proteus::pyarray<double>& dLow = args.array<double>("dLow");
        proteus::pyarray<double>& dt_times_dH_minus_dL = args.array<double>("dt_times_dH_minus_dL");
        proteus::pyarray<double>& min_u_bc = args.array<double>("min_u_bc");
        proteus::pyarray<double>& max_u_bc = args.array<double>("max_u_bc");
        proteus::pyarray<double>& quantDOFs = args.array<double>("quantDOFs");
	proteus::pyarray<double>& ebqe_phi_s = args.array<double>("ebqe_phi_s");
	double ghost_penalty_constant = args.scalar<double>("ghost_penalty_constant");
	const proteus::pyarray<double>& phi_solid = args.array<double>("phi_solid");
	proteus::pyarray<double>& phi_solid_nodes = args.array<double>("phi_solid_nodes");
	bool useExact = args.scalar<int>("useExact");
	proteus::pyarray<double>& isActiveR = args.array<double>("isActiveR");
	proteus::pyarray<double>& isActiveDOF = args.array<double>("isActiveDOF");
	proteus::pyarray<int>& isActiveElement = args.array<int>("isActiveElement");
        double meanEntropy = 0., meanOmega = 0., maxEntropy = -1E10, minEntropy = 1E10;
        maxVel.resize(nElements_global, 0.0);
        maxEntRes.resize(nElements_global, 0.0);
        double Ct_sge = 4.0;
	gf_s.useExact = useExact;
	cutfem_boundaries.clear();
	cutfem_local_boundaries.clear();
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
            double elementResidual_u[nDOF_test_element];
	    bool element_active=false;
	    isActiveElement[eN]=0;
            for (int i=0;i<nDOF_test_element;i++)
              {
                elementResidual_u[i]=0.0;
              }//i
	    double element_phi_s[nDOF_mesh_trial_element];
	    for (int j=0;j<nDOF_mesh_trial_element;j++)
	      {
		int eN_j = eN*nDOF_mesh_trial_element+j;
		element_phi_s[j] = phi_solid_nodes.data()[u_l2g.data()[eN_j]];
	      }
	    double element_nodes[nDOF_mesh_trial_element*3];
	    for (int i=0;i<nDOF_mesh_trial_element;i++)
	      {
		int eN_i=eN*nDOF_mesh_trial_element+i;
		for(int I=0;I<3;I++)
		  element_nodes[i*3 + I] = mesh_dof.data()[mesh_l2g.data()[eN_i]*3 + I];
	      }//i
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
            //loop over quadrature points and compute integrands
            for  (int k=0;k<nQuadraturePoints_element;k++)
              {
                //compute indeces and declare local storage
                int eN_k = eN*nQuadraturePoints_element+k,
                  eN_k_nSpace = eN_k*nSpace,
                  eN_nDOF_trial_element = eN*nDOF_trial_element;
                double
                  entVisc_minus_artComp,
                  u=0.0,un=0.0,
                  grad_u[nSpace],grad_u_old[nSpace],grad_uTilde[nSpace],
                  m=0.0,dm=0.0,
                  H=0.0,Hn=0.0,HTilde=0.0,
                  f[nSpace],fn[nSpace],df[nSpace],
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
                  //VRANS
                  porosity,
                  //
                  G[nSpace*nSpace],G_dd_G,tr_G;//norm_Rv;
		gf_s.set_quad(k);
		const double H_s = gf_s.H(0.0,phi_solid.data()[eN_k]);

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
                ck.gradTrialFromRef(&u_grad_trial_ref.data()[k*nDOF_trial_element*nSpace],
                                    jacInv,
                                    u_grad_trial);
                //get the solution
                ck.valFromDOF(u_dof.data(),
                              &u_l2g.data()[eN_nDOF_trial_element],
                              &u_trial_ref.data()[k*nDOF_trial_element],
                              u);
                ck.valFromDOF(u_dof_old.data(),
                              &u_l2g.data()[eN_nDOF_trial_element],
                              &u_trial_ref.data()[k*nDOF_trial_element],
                              un);
                //get the solution gradients
                ck.gradFromDOF(u_dof.data(),
                               &u_l2g.data()[eN_nDOF_trial_element],
                               u_grad_trial,
                               grad_u);
                ck.gradFromDOF(u_dof_old.data(),
                               &u_l2g.data()[eN_nDOF_trial_element],
                               u_grad_trial,
                               grad_u_old);
                ck.gradFromDOF(uTilde_dof.data(),
                               &u_l2g.data()[eN_nDOF_trial_element],
                               u_grad_trial,
                               grad_uTilde);
                //precalculate test function products with integration weights
                for (int j=0;j<nDOF_trial_element;j++)
                  {
                    u_test_dV[j] = u_test_ref.data()[k*nDOF_trial_element+j]*dV;
                    for (int I=0;I<nSpace;I++)
                      {
                        u_grad_test_dV[j*nSpace+I]   = u_grad_trial[j*nSpace+I]*dV;//cek warning won't work for Petrov-Galerkin
                      }
                  }
                //VRANS
                porosity = q_porosity.data()[eN_k];
                //
                //
                //calculate pde coefficients at quadrature points
                //
                evaluateCoefficients(&velocity.data()[eN_k_nSpace],
                                     u,
                                     //VRANS
                                     porosity,
                                     //
                                     m,
                                     dm,
                                     f,
                                     df);
                //
                //moving mesh
                //
                double mesh_velocity[3];
                mesh_velocity[0] = xt;
                mesh_velocity[1] = yt;
                mesh_velocity[2] = zt;

                for (int I=0;I<nSpace;I++)
                  {
                    f[I] -= MOVING_DOMAIN*m*mesh_velocity[I];
                    df[I] -= MOVING_DOMAIN*dm*mesh_velocity[I];
                  }
                //
                //calculate time derivative at quadrature points
                //
                if (q_dV_last.data()[eN_k] <= -100)
                  q_dV_last.data()[eN_k] = dV;
                q_dV.data()[eN_k] = dV;
                ck.bdf(alphaBDF,
                       q_m_betaBDF.data()[eN_k]*q_dV_last.data()[eN_k]/dV,//ensure prior mass integral is correct for  m_t with BDF1
                       m,
                       dm,
                       m_t,
                       dm_t);

                if (STABILIZATION_TYPE==1)
                  {
                    double normVel=0., norm_grad_un=0.;
                    for (int I=0;I<nSpace;I++)
                      {
                        Hn += df[I]*grad_u_old[I];
                        HTilde += df[I]*grad_uTilde[I];
                        fn[I] = porosity*df[I]*un-MOVING_DOMAIN*m*mesh_velocity[I];
                        H += df[I]*grad_u[I];
                        normVel += df[I]*df[I];
                        norm_grad_un += grad_u_old[I]*grad_u_old[I];
                      }
                    normVel = std::sqrt(normVel);
                    norm_grad_un = std::sqrt(norm_grad_un)+1E-10;

                    // calculate CFL
                    calculateCFL(elementDiameter.data()[eN]/degree_polynomial,df,cfl.data()[eN_k]);


                    // compute max velocity at cell
                    maxVel[eN] = fmax(normVel,maxVel[eN]);

                    // Strong entropy residual
                    double entRes = (ENTROPY(u,0,1)-ENTROPY(un,0,1))/dt + 0.5*(DENTROPY(u,0,1)*H +
                                                                               DENTROPY(un,0,1)*Hn);
                    maxEntRes[eN] = fmax(maxEntRes[eN],fabs(entRes));

                    // Quantities for normalization factor //
                    meanEntropy += ENTROPY(u,0,1)*dV;
                    meanOmega += dV;
                    maxEntropy = fmax(maxEntropy,ENTROPY(u,0,1));
                    minEntropy = fmin(minEntropy,ENTROPY(u,0,1));

                    // artificial compression
                    double hK=elementDiameter.data()[eN]/degree_polynomial;
                    entVisc_minus_artComp = fmax(1-cK*fmax(un*(1-un),0)/hK/norm_grad_un,0);
                  }
                else
                  {
                    //
                    //calculate subgrid error (strong residual and adjoint)
                    //
                    //calculate strong residual
                    pdeResidual_u = ck.Mass_strong(m_t) + ck.Advection_strong(df,grad_u);
                    //calculate adjoint
                    for (int i=0;i<nDOF_test_element;i++)
                      {
                        // int eN_k_i_nSpace = (eN_k*nDOF_trial_element+i)*nSpace;
                        // Lstar_u[i]  = ck.Advection_adjoint(df,&u_grad_test_dV[eN_k_i_nSpace]);
                        int i_nSpace = i*nSpace;
                        Lstar_u[i]  = ck.Advection_adjoint(df,&u_grad_test_dV[i_nSpace]);
                      }
                    //calculate tau and tau*Res
                    calculateSubgridError_tau(elementDiameter.data()[eN],dm_t,df,cfl.data()[eN_k],tau0);
                    calculateSubgridError_tau(Ct_sge,
                                              G,
                                              dm_t,
                                              df,
                                              tau1,
                                              cfl.data()[eN_k]);
                    tau = useMetrics*tau1+(1.0-useMetrics)*tau0;

                    subgridError_u = -tau*pdeResidual_u;
                    //
                    //calculate shock capturing diffusion
                    //

                    ck.calculateNumericalDiffusion(shockCapturingDiffusion,
                                                   elementDiameter.data()[eN],
                                                   (pdeResidual_u - (sc_drop_time*ck.Mass_strong(m_t))),
                                                   grad_u,
                                                   numDiff0);
                    //ck.calculateNumericalDiffusion(shockCapturingDiffusion,G,pdeResidual_u,grad_u_old,numDiff1);
                    ck.calculateNumericalDiffusion(shockCapturingDiffusion,
                                                   sc_uref,
                                                   sc_alpha,
                                                   G,
                                                   G_dd_G,
                                                   (pdeResidual_u - (sc_drop_time*ck.Mass_strong(m_t))),
                                                   grad_u,
                                                   numDiff1);
                    q_numDiff_u.data()[eN_k] = useMetrics*numDiff1+(1.0-useMetrics)*numDiff0;
                    //std::cout<<tau<<"   "<<q_numDiff_u.data()[eN_k]<<'\t'<<numDiff0<<'\t'<<numDiff1<<'\t'<<pdeResidual_u<<std::endl;

                    //
                    //update element residual
                    //


                    /*              std::cout<<m_t<<'\t'
                                    <<f[0]<<'\t'
                                    <<f[1]<<'\t'
                                    <<df[0]<<'\t'
                                    <<df[1]<<'\t'
                                    <<subgridError_u<<'\t'
                                    <<q_numDiff_u_last.data()[eN_k]<<std::endl;*/
                  }

                for(int i=0;i<nDOF_test_element;i++)
                  {
		    int eN_i=eN*nDOF_test_element+i;
                    //int eN_k_i=eN_k*nDOF_test_element+i,
                    //eN_k_i_nSpace = eN_k_i*nSpace,
                    int i_nSpace=i*nSpace;
                    if (STABILIZATION_TYPE==1)
                      {
                        if (stage == 1)
                          elementResidual_u[i] +=
                            ck.Mass_weak(dt*m_t,u_test_dV[i]) +  // time derivative
                            1./3*dt*ck.Advection_weak(fn,&u_grad_test_dV[i_nSpace]) +
                            1./9*dt*dt*ck.NumericalDiffusion(Hn,df,&u_grad_test_dV[i_nSpace]) +
                            1./3*dt*entVisc_minus_artComp*ck.NumericalDiffusion(q_numDiff_u_last.data()[eN_k],
                                                                                grad_u_old,
                                                                                &u_grad_test_dV[i_nSpace]);
                        // TODO: Add part about moving mesh
                        else //stage 2
                          elementResidual_u[i] +=
                            ck.Mass_weak(dt*m_t,u_test_dV[i]) +  // time derivative
                            dt*ck.Advection_weak(fn,&u_grad_test_dV[i_nSpace]) +
                            0.5*dt*dt*ck.NumericalDiffusion(HTilde,df,&u_grad_test_dV[i_nSpace]) +
                            dt*entVisc_minus_artComp*ck.NumericalDiffusion(q_numDiff_u_last.data()[eN_k],
                                                                           grad_u_old,
                                                                           &u_grad_test_dV[i_nSpace]);
                      }
                    else //supg
                      {
                        elementResidual_u[i] +=
                          H_s*(ck.Mass_weak(m_t,u_test_dV[i]) +
			       ck.Advection_weak(f,&u_grad_test_dV[i_nSpace]) +
			       ck.SubgridError(subgridError_u,Lstar_u[i]) +
			       ck.NumericalDiffusion(q_numDiff_u_last.data()[eN_k],
						     grad_u,
						     &u_grad_test_dV[i_nSpace]));
                      }
		    if (element_active)
		      {
			isActiveR.data()[offset_u + stride_u*r_l2g.data()[eN_i]] = 1.0;
			isActiveDOF.data()[u_l2g.data()[eN_i]] = 1.0;
		      }
                  }//i
                //
                //todo, get rid of m, since u=m
                //save momentum for time history and velocity for subgrid error
                //save solution for other models
                //
                q_u.data()[eN_k] = u;
                q_m.data()[eN_k] = m;
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
			//todo: check that physical coordinates match
			ck.calculateMappingVelocity_elementBoundary(eN,
								    ebN_local,
								    kb,
								    ebN_local_kb,
								    mesh_velocity_dof.data(),
								    mesh_l2g.data(),
								    mesh_trial_trace_ref.data(),
								    xt_int,yt_int,zt_int,
								    normal,
								    boundaryJac,
								    metricTensor,
								    integralScaling);
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
	// std::set<int>::iterator it=cutfem_boundaries.begin();
	// while(it!=cutfem_boundaries.end())
	//   {
	//     std::map<int,double> DW_Dn_jump;
	//     std::map<std::pair<int, int>, int> u_u_nz;
	//     double gamma_cutfem=ghost_penalty_constant,
	//       h_cutfem=elementBoundaryDiameter.data()[*it];
	//     int eN_nDOF_trial_element  = elementBoundaryElementsArray.data()[(*it)*2+0]*nDOF_trial_element;
	//     //See Massing Schott Wall 2018
	//     //double norm_v=0.0;
	//     //for (int i_offset=1;i_offset<nDOF_v_trial_element;i_offset++)//MSW18 is just on face
	//     //  {
	//     //    int i = (cutfem_local_boundaries[*it] + i_offset)%nDOF_v_trial_element;//cek hack only works for P1
	//     //    double u=u_old_dof.data()[vel_l2g.data()[eN_nDOF_v_trial_element+i]],
	//     //      v=v_old_dof.data()[vel_l2g.data()[eN_nDOF_v_trial_element+i]],
	//     //      w=w_old_dof.data()[vel_l2g.data()[eN_nDOF_v_trial_element+i]];
	//     //    norm_v=fmax(norm_v,sqrt(u*u+v*v+w*w));
	//     //  }
	//     ///double gamma_v_dim = rho_0*(nu_0 + norm_v*h_cutfem + alphaBDF*h_cutfem*h_cutfem);
	//     //gamma_cutfem_p *= h_cutfem*h_cutfem/gamma_v_dim;
	//     //if (NONCONSERVATIVE_FORM)
	//     //  gamma_cutfem*=gamma_v_dim;
	//     //else
	//     //  gamma_cutfem*=(gamma_v_dim/rho_0);
	//     for (int kb=0;kb<nQuadraturePoints_elementBoundary;kb++)
	//       {
	// 	double Du_Dn_jump=0.0, dS;
	// 	for (int eN_side=0;eN_side < 2; eN_side++)
	// 	  {
	// 	    int ebN = *it,
	// 	      eN  = elementBoundaryElementsArray.data()[ebN*2+eN_side];
	// 	    for (int i=0;i<nDOF_test_element;i++)
	// 	      {
	// 		DW_Dn_jump[r_l2g.data()[eN*nDOF_test_element+i]] = 0.0;
	// 	      }
	// 	  }
	// 	for (int eN_side=0;eN_side < 2; eN_side++)
	// 	  {
	// 	    int ebN = *it,
	// 	      eN  = elementBoundaryElementsArray.data()[ebN*2+eN_side],
	// 	      ebN_local = elementBoundaryLocalElementBoundariesArray.data()[ebN*2+eN_side],
	// 	      eN_nDOF_trial_element = eN*nDOF_trial_element,
	// 	      ebN_local_kb = ebN_local*nQuadraturePoints_elementBoundary+kb,
	// 	      ebN_local_kb_nSpace = ebN_local_kb*nSpace;
	// 	    double u_int=0.0,
	// 	      grad_u_int[nSpace],
	// 	      jac_int[nSpace*nSpace],
	// 	      jacDet_int,
	// 	      jacInv_int[nSpace*nSpace],
	// 	      boundaryJac[nSpace*(nSpace-1)],
	// 	      metricTensor[(nSpace-1)*(nSpace-1)],
	// 	      metricTensorDetSqrt,
	// 	      u_test_dS[nDOF_test_element],
	// 	      u_grad_trial_trace[nDOF_trial_element*nSpace],
	// 	      u_grad_test_dS[nDOF_trial_element*nSpace],
	// 	      normal[nSpace],x_int,y_int,z_int,xt_int,yt_int,zt_int,integralScaling,
	// 	      G[nSpace*nSpace],G_dd_G,tr_G,h_phi,h_penalty,penalty;
	// 	    for (int I=0; I<nSpace;I++)
	// 	      grad_u_int[I] = 0.0;
	// 	    //compute information about mapping from reference element to physical element

	// 	    ck.calculateMapping_elementBoundary(eN,
	// 						ebN_local,
	// 						kb,
	// 						ebN_local_kb,
	// 						mesh_dof.data(),
	// 						mesh_l2g.data(),
	// 						mesh_trial_trace_ref.data(),
	// 						mesh_grad_trial_trace_ref.data(),
	// 						boundaryJac_ref.data(),
	// 						jac_int,
	// 						jacDet_int,
	// 						jacInv_int,
	// 						boundaryJac,
	// 						metricTensor,
	// 						metricTensorDetSqrt,
	// 						normal_ref.data(),
	// 						normal,
	// 						x_int,y_int,z_int);
	// 	    //todo: check that physical coordinates match
	// 	    ck.calculateMappingVelocity_elementBoundary(eN,
	// 							ebN_local,
	// 							kb,
	// 							ebN_local_kb,
	// 							mesh_velocity_dof.data(),
	// 							mesh_l2g.data(),
	// 							mesh_trial_trace_ref.data(),
	// 							xt_int,yt_int,zt_int,
	// 							normal,
	// 							boundaryJac,
	// 							metricTensor,
	// 							integralScaling);
	// 	    dS = metricTensorDetSqrt*dS_ref.data()[kb];
	// 	    //compute shape and solution information
	// 	    //shape
	// 	    ck.gradTrialFromRef(&u_grad_trial_trace_ref.data()[ebN_local_kb_nSpace*nDOF_trial_element],jacInv_int,u_grad_trial_trace);
	// 	    for (int i=0;i<nDOF_test_element;i++)
	// 	      {
	// 		int eN_i = eN*nDOF_test_element + i;
	// 		for (int I=0;I<nSpace;I++)
	// 		  DW_Dn_jump[r_l2g.data()[eN_i]] += u_grad_trial_trace[i*nSpace+I]*normal[I];
	// 	      }
	// 	  }//eN_side
	// 	for (int eN_side=0;eN_side < 2; eN_side++)
	// 	  {
	// 	    int ebN = *it,
	// 	      eN  = elementBoundaryElementsArray.data()[ebN*2+eN_side];
	// 	    for (int i=0;i<nDOF_test_element;i++)
	// 	      {
	// 		int eN_i = eN*nDOF_test_element+i;
	// 		for (int eN_side2=0;eN_side2 < 2; eN_side2++)
	// 		  {
	// 		    int eN2  = elementBoundaryElementsArray.data()[ebN*2+eN_side2];
	// 		    for (int j=0;j<nDOF_test_element;j++)
	// 		      {
	// 			int eN_i_j = eN_i*nDOF_test_element + j;
	// 			int eN2_j = eN2*nDOF_test_element + j;
	// 			int ebN_i_j = ebN*4*nDOF_test_X_trial_element +
	// 			  eN_side*2*nDOF_test_X_trial_element +
	// 			  eN_side2*nDOF_test_X_trial_element +
	// 			  i*nDOF_trial_element +
	// 			  j;
	// 			std::pair<int,int> ij = std::make_pair(u_l2g.data()[eN_i], u_l2g.data()[eN2_j]);
	// 			if (u_u_nz.count(ij))
	// 			  {
	// 			    assert(u_u_nz[ij] == csrRowIndeces_u_u.data()[eN_i] + csrColumnOffsets_eb_u_u.data()[ebN_i_j]);
	// 			  }
	// 			else
	// 			  u_u_nz[ij] =  csrRowIndeces_u_u.data()[eN_i] + csrColumnOffsets_eb_u_u.data()[ebN_i_j];
	// 		      }
	// 		  }
	// 	      }
	// 	  }
	// 	for (std::map<int,double>::iterator Wi_it=DW_Dn_jump.begin(); Wi_it!=DW_Dn_jump.end(); ++Wi_it)
	// 	  for (std::map<int,double>::iterator Wj_it=DW_Dn_jump.begin(); Wj_it!=DW_Dn_jump.end(); ++Wj_it)
	// 	    {
	// 	      int i_global = Wi_it->first,
	// 		j_global = Wj_it->first;
	// 	      double DW_Dn_jump_i = Wi_it->second,
	// 		DW_Dn_jump_j = Wj_it->second;
	// 	      std::pair<int,int> ij = std::make_pair(i_global, j_global);
	// 	      //globalJacobian.data()[u_u_nz.at(ij)] += gamma_cutfem*h_cutfem*DW_Dn_jump_j*DW_Dn_jump_i*dS;
	// 	      globalResidual.data()[offset_u+stride_u*i_global]+=u_dof.data()[j_global]*gamma_cutfem*h_cutfem*DW_Dn_jump_j*DW_Dn_jump_i*dS;
	// 	    }//i,j
	//       }//kb
	//     it++;
	//   }//cutfem element boundaries
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
            double elementResidual_u[nDOF_test_element];
            for (int i=0;i<nDOF_test_element;i++)
              {
                elementResidual_u[i]=0.0;
              }
          double element_phi_s[nDOF_mesh_trial_element];
          for (int j=0;j<nDOF_mesh_trial_element;j++)
            {
              int eN_j = eN*nDOF_mesh_trial_element+j;
              element_phi_s[j] = phi_solid_nodes[u_l2g.data()[eN_j]];
            }
          double element_nodes[nDOF_mesh_trial_element*3];
          for (int i=0;i<nDOF_mesh_trial_element;i++)
            {
              int eN_i=eN*nDOF_mesh_trial_element+i;
              for(int I=0;I<3;I++)
                element_nodes[i*3 + I] = mesh_dof[mesh_l2g.data()[eN_i]*3 + I];
            }//i
          double mesh_dof_ref[12]={0.,0.,0.,1.,0.,0.,0.,1.,0.,0.,0.,1.};
          double xb_ref_calc[nQuadraturePoints_elementBoundary*3];
          for  (int kb=0;kb<nQuadraturePoints_elementBoundary;kb++)
            {
              double x=0.0,y=0.0,z=0.0;
              for (int j=0;j<nDOF_mesh_trial_element;j++)
                {
                  int ebN_local_kb = ebN_local*nQuadraturePoints_elementBoundary+kb;
                  int ebN_local_kb_j = ebN_local_kb*nDOF_mesh_trial_element+j;
                  x += mesh_dof_ref[j*3+0]*mesh_trial_trace_ref.data()[ebN_local_kb_j]; 
                  y += mesh_dof_ref[j*3+1]*mesh_trial_trace_ref.data()[ebN_local_kb_j]; 
                  z += mesh_dof_ref[j*3+2]*mesh_trial_trace_ref.data()[ebN_local_kb_j];
                }
              xb_ref_calc[3*kb+0] = x;
              xb_ref_calc[3*kb+1] = y;
              xb_ref_calc[3*kb+2] = z;
            }
	    int icase_s = gf_s.calculate(element_phi_s, element_nodes, xb_ref_calc, true);
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
                  f_ext[nSpace],
                  df_ext[nSpace],
                  flux_ext=0.0,
                  bc_u_ext=0.0,
                  //bc_grad_u_ext[nSpace],
                  bc_m_ext=0.0,
                  bc_dm_ext=0.0,
                  bc_f_ext[nSpace],
                  bc_df_ext[nSpace],
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
                  //VRANS
                  porosity_ext,
                  //
                  G[nSpace*nSpace],G_dd_G,tr_G;
                //
                //calculate the solution and gradients at quadrature points
                //
                //compute information about mapping from reference element to physical element
		gf_s.set_boundary_quad(kb);
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
                //std::cout<<"metricTensorDetSqrt "<<metricTensorDetSqrt<<" integralScaling "<<integralScaling<<std::endl;
                dS = ((1.0-MOVING_DOMAIN)*metricTensorDetSqrt + MOVING_DOMAIN*integralScaling)*dS_ref.data()[kb];
                //get the metric tensor
                //cek todo use symmetry
                ck.calculateG(jacInv_ext,G,G_dd_G,tr_G);
                //compute shape and solution information
                //shape
                ck.gradTrialFromRef(&u_grad_trial_trace_ref.data()[ebN_local_kb_nSpace*nDOF_trial_element],
                                    jacInv_ext,
                                    u_grad_trial_trace);
                //solution and gradients
                if (STABILIZATION_TYPE==1) //explicit
                  {
                    ck.valFromDOF(u_dof_old.data(),
                                  &u_l2g.data()[eN_nDOF_trial_element],
                                  &u_trial_trace_ref.data()[ebN_local_kb*nDOF_test_element],
                                  u_ext);
                    ck.gradFromDOF(u_dof_old.data(),
                                   &u_l2g.data()[eN_nDOF_trial_element],
                                   u_grad_trial_trace,
                                   grad_u_ext);
                  }
                else
                  {
                    ck.valFromDOF(u_dof.data(),
                                  &u_l2g.data()[eN_nDOF_trial_element],
                                  &u_trial_trace_ref.data()[ebN_local_kb*nDOF_test_element],
                                  u_ext);
                    ck.gradFromDOF(u_dof.data(),
                                   &u_l2g.data()[eN_nDOF_trial_element],
                                   u_grad_trial_trace,
                                   grad_u_ext);
                  }
                //precalculate test function products with integration weights
                for (int j=0;j<nDOF_trial_element;j++)
                  {
                    u_test_dS[j] = u_test_trace_ref.data()[ebN_local_kb*nDOF_test_element+j]*dS;
                  }
                //
                //load the boundary values
                //
                bc_u_ext = isDOFBoundary_u.data()[ebNE_kb]*ebqe_bc_u_ext.data()[ebNE_kb]+(1-isDOFBoundary_u.data()[ebNE_kb])*u_ext;
                //VRANS
                porosity_ext = ebqe_porosity_ext.data()[ebNE_kb];
                //
                //
                //calculate the pde coefficients using the solution and the boundary values for the solution
                //
                evaluateCoefficients(&ebqe_velocity_ext.data()[ebNE_kb_nSpace],
                                     u_ext,
                                     //VRANS
                                     porosity_ext,
                                     //
                                     m_ext,
                                     dm_ext,
                                     f_ext,
                                     df_ext);
                evaluateCoefficients(&ebqe_velocity_ext.data()[ebNE_kb_nSpace],
                                     bc_u_ext,
                                     //VRANS
                                     porosity_ext,
                                     //
                                     bc_m_ext,
                                     bc_dm_ext,
                                     bc_f_ext,
                                     bc_df_ext);
                //
                //moving mesh
                //
                double mesh_velocity[3];
                mesh_velocity[0] = xt_ext;
                mesh_velocity[1] = yt_ext;
                mesh_velocity[2] = zt_ext;
                //std::cout<<"mesh_velocity ext"<<std::endl;
                for (int I=0;I<nSpace;I++)
                  {
                    //std::cout<<mesh_velocity[I]<<std::endl;
                    f_ext[I] -= MOVING_DOMAIN*m_ext*mesh_velocity[I];
                    df_ext[I] -= MOVING_DOMAIN*dm_ext*mesh_velocity[I];
                    bc_f_ext[I] -= MOVING_DOMAIN*bc_m_ext*mesh_velocity[I];
                    bc_df_ext[I] -= MOVING_DOMAIN*bc_dm_ext*mesh_velocity[I];
                  }
                //
                //calculate the numerical fluxes
                //
                exteriorNumericalAdvectiveFlux(isDOFBoundary_u.data()[ebNE_kb],
                                               isFluxBoundary_u.data()[ebNE_kb],
                                               normal,
                                               bc_u_ext,
                                               ebqe_bc_flux_u_ext.data()[ebNE_kb],
                                               u_ext,
                                               df_ext,//VRANS includes porosity
                                               flux_ext);
                ebqe_flux.data()[ebNE_kb] = flux_ext;
                //save for other models? cek need to be consistent with numerical flux
                if(flux_ext >=0.0)
                  ebqe_u.data()[ebNE_kb] = u_ext;
                else
                  ebqe_u.data()[ebNE_kb] = bc_u_ext;

                if (STABILIZATION_TYPE==1)
                  if (stage==1)
                    flux_ext *= 1./3*dt;
                  else
                    flux_ext *= dt;

                //
                //update residuals
                //
		const double H_s = gf_s.H(0.0, ebqe_phi_s.data()[ebNE_kb]);
		if (isActiveElement[eN])
		  {
		    for (int i=0;i<nDOF_test_element;i++)
		      {
			//int ebNE_kb_i = ebNE_kb*nDOF_test_element+i;
			elementResidual_u[i] += H_s*ck.ExteriorElementBoundaryFlux(flux_ext,u_test_dS[i]);
		      }//i
		  }
              }//kb
            //
            //update the element and global residual storage
            //
            for (int i=0;i<nDOF_test_element;i++)
              {
                int eN_i = eN*nDOF_test_element+i;
                globalResidual.data()[offset_u+stride_u*r_l2g.data()[eN_i]] += elementResidual_u[i];
              }//i
          }//ebNE
        if (STABILIZATION_TYPE==1)
          {
            meanEntropy /= meanOmega;
            double norm_factor = fmax(fabs(maxEntropy - meanEntropy), fabs(meanEntropy-minEntropy));
            for(int eN=0;eN<nElements_global;eN++)
              {
                double hK=elementDiameter.data()[eN]/degree_polynomial;
                double linear_viscosity = cMax*hK*maxVel[eN];
                double entropy_viscosity = cE*hK*hK*maxEntRes[eN]/norm_factor;
                for  (int k=0;k<nQuadraturePoints_element;k++)
                  {
                    int eN_k = eN*nQuadraturePoints_element+k;
                    q_numDiff_u.data()[eN_k] = fmin(linear_viscosity,entropy_viscosity);
                  }
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
        int lag_shockCapturing = args.scalar<int>("lag_shockCapturing");
        double shockCapturingDiffusion = args.scalar<double>("shockCapturingDiffusion");
        const proteus::pyarray<double>& q_porosity = args.array<double>("q_porosity");
        proteus::pyarray<int>& u_l2g = args.array<int>("u_l2g");
        proteus::pyarray<int>& r_l2g = args.array<int>("r_l2g");
        proteus::pyarray<double>& elementDiameter = args.array<double>("elementDiameter");
	proteus::pyarray<double>& elementBoundaryDiameter = args.array<double>("elementBoundaryDiameter");
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
        const proteus::pyarray<double>& ebqe_porosity_ext = args.array<double>("ebqe_porosity_ext");
        proteus::pyarray<int>& isDOFBoundary_u = args.array<int>("isDOFBoundary_u");
        proteus::pyarray<double>& ebqe_bc_u_ext = args.array<double>("ebqe_bc_u_ext");
        proteus::pyarray<int>& isFluxBoundary_u = args.array<int>("isFluxBoundary_u");
        proteus::pyarray<double>& ebqe_bc_flux_u_ext = args.array<double>("ebqe_bc_flux_u_ext");
        proteus::pyarray<int>& csrColumnOffsets_eb_u_u = args.array<int>("csrColumnOffsets_eb_u_u");
        int STABILIZATION_TYPE = args.scalar<int>("STABILIZATION_TYPE");
	proteus::pyarray<double>& ebqe_phi_s = args.array<double>("ebqe_phi_s");
	const proteus::pyarray<double>& phi_solid = args.array<double>("phi_solid");
	double ghost_penalty_constant = args.scalar<double>("ghost_penalty_constant");
	proteus::pyarray<double>& phi_solid_nodes = args.array<double>("phi_solid_nodes");
	bool useExact = args.scalar<int>("useExact");
	proteus::pyarray<double>& isActiveR = args.array<double>("isActiveR");
	proteus::pyarray<double>& isActiveDOF = args.array<double>("isActiveDOF");
	proteus::pyarray<int>& isActiveElement = args.array<int>("isActiveElement");
        //std::cout<<"ndjaco  address "<<q_numDiff_u_last.data()<<std::endl;
        double Ct_sge = 4.0;
	gf_s.useExact = useExact;
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
	    double element_phi_s[nDOF_mesh_trial_element];
	    for (int j=0;j<nDOF_mesh_trial_element;j++)
	      {
		int eN_j = eN*nDOF_mesh_trial_element+j;
		element_phi_s[j] = phi_solid_nodes.data()[u_l2g.data()[eN_j]];
	      }
	    double element_nodes[nDOF_mesh_trial_element*3];
	    for (int i=0;i<nDOF_mesh_trial_element;i++)
	      {
		int eN_i=eN*nDOF_mesh_trial_element+i;
		for(int I=0;I<3;I++)
		  element_nodes[i*3 + I] = mesh_dof.data()[mesh_l2g.data()[eN_i]*3 + I];
	      }//i
	    int icase_s = gf_s.calculate(element_phi_s, element_nodes, x_ref.data(), false);
            for  (int k=0;k<nQuadraturePoints_element;k++)
              {
                int eN_k = eN*nQuadraturePoints_element+k, //index to a scalar at a quadrature point
                  eN_k_nSpace = eN_k*nSpace,
                  eN_nDOF_trial_element = eN*nDOF_trial_element; //index to a vector at a quadrature point

                //declare local storage
                double u=0.0,
                  grad_u[nSpace],
                  m=0.0,dm=0.0,
                  f[nSpace],df[nSpace],
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
                  //VRANS
                  porosity,
                  //
                  G[nSpace*nSpace],G_dd_G,tr_G;
		gf_s.set_quad(k);
                //
                //calculate solution and gradients at quadrature points
                //
                // u=0.0;
                // for (int I=0;I<nSpace;I++)
                //   {
                //     grad_u[I]=0.0;
                //   }
                // for (int j=0;j<nDOF_trial_element;j++)
                //   {
                //     int eN_j=eN*nDOF_trial_element+j;
                //     int eN_k_j=eN_k*nDOF_trial_element+j;
                //     int eN_k_j_nSpace = eN_k_j*nSpace;

                //     u += valFromDOF_c(u_dof.data()[u_l2g.data()[eN_j]],u_trial[eN_k_j]);
                //     for (int I=0;I<nSpace;I++)
                //       {
                //         grad_u[I] += gradFromDOF_c(u_dof.data()[u_l2g.data()[eN_j]],u_grad_trial[eN_k_j_nSpace+I]);
                //       }
                //   }
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
                //VRANS
                porosity = q_porosity.data()[eN_k];
                //
                //
                //calculate pde coefficients and derivatives at quadrature points
                //
		const double H_s = gf_s.H(0.0, phi_solid.data()[eN_k]);
                evaluateCoefficients(&velocity.data()[eN_k_nSpace],
                                     u,
                                     //VRANS
                                     porosity,
                                     //
                                     m,
                                     dm,
                                     f,
                                     df);
                //
                //moving mesh
                //
                double mesh_velocity[3];
                mesh_velocity[0] = xt;
                mesh_velocity[1] = yt;
                mesh_velocity[2] = zt;
                //std::cout<<"qj mesh_velocity"<<std::endl;
                for(int I=0;I<nSpace;I++)
                  {
                    //std::cout<<mesh_velocity[I]<<std::endl;
                    f[I] -= MOVING_DOMAIN*m*mesh_velocity[I];
                    df[I] -= MOVING_DOMAIN*dm*mesh_velocity[I];
                  }
                //
                //calculate time derivatives
                //
                ck.bdf(alphaBDF,
                       q_m_betaBDF.data()[eN_k],//since m_t isn't used, we don't have to correct mass
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
                    // int eN_k_i_nSpace = (eN_k*nDOF_trial_element+i)*nSpace;
                    // Lstar_u[i]=ck.Advection_adjoint(df,&u_grad_test_dV[eN_k_i_nSpace]);
                    int i_nSpace = i*nSpace;
                    Lstar_u[i]=ck.Advection_adjoint(df,&u_grad_test_dV[i_nSpace]);
                  }
                //calculate the Jacobian of strong residual
                for (int j=0;j<nDOF_trial_element;j++)
                  {
                    //int eN_k_j=eN_k*nDOF_trial_element+j;
                    //int eN_k_j_nSpace = eN_k_j*nSpace;
                    int j_nSpace = j*nSpace;
                    dpdeResidual_u_u[j]= ck.MassJacobian_strong(dm_t,u_trial_ref.data()[k*nDOF_trial_element+j]) +
                      ck.AdvectionJacobian_strong(df,&u_grad_trial[j_nSpace]);
                  }
                //tau and tau*Res
                calculateSubgridError_tau(elementDiameter.data()[eN],
                                          dm_t,
                                          df,
                                          cfl.data()[eN_k],
                                          tau0);

                calculateSubgridError_tau(Ct_sge,
                                          G,
                                          dm_t,
                                          df,
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
                        if (STABILIZATION_TYPE==0)
                          {
                            elementJacobian_u_u[i][j] +=
                              H_s*(ck.MassJacobian_weak(dm_t,
							u_trial_ref.data()[k*nDOF_trial_element+j],
							u_test_dV[i]) +
				   ck.AdvectionJacobian_weak(df,
							     u_trial_ref.data()[k*nDOF_trial_element+j],
							     &u_grad_test_dV[i_nSpace]) +
				   ck.SubgridErrorJacobian(dsubgridError_u_u[j],Lstar_u[i]) +
				   ck.NumericalDiffusionJacobian(q_numDiff_u_last.data()[eN_k],
								 &u_grad_trial[j_nSpace],
								 &u_grad_test_dV[i_nSpace])); //implicit
                          }
                        else
                          {
                            elementJacobian_u_u[i][j] +=
                              ck.MassJacobian_weak(1.0,
                                                   u_trial_ref.data()[k*nDOF_trial_element+j],
                                                   u_test_dV[i]);
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
		    //todo: check that physical coordinates match
		    ck.calculateMappingVelocity_elementBoundary(eN,
								ebN_local,
								kb,
								ebN_local_kb,
								mesh_velocity_dof.data(),
								mesh_l2g.data(),
								mesh_trial_trace_ref.data(),
								xt_int,yt_int,zt_int,
								normal,
								boundaryJac,
								metricTensor,
								integralScaling);
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
        //
        //loop over exterior element boundaries to compute the surface integrals and load them into the global Jacobian
        //
        if (STABILIZATION_TYPE==0)
          for (int ebNE = 0; ebNE < nExteriorElementBoundaries_global; ebNE++)
            {
              int ebN = exteriorElementBoundariesArray.data()[ebNE];
              int eN  = elementBoundaryElementsArray.data()[ebN*2+0],
                ebN_local = elementBoundaryLocalElementBoundariesArray.data()[ebN*2+0],
                eN_nDOF_trial_element = eN*nDOF_trial_element;
	      double element_phi_s[nDOF_mesh_trial_element];
	      for (int j=0;j<nDOF_mesh_trial_element;j++)
		{
		  int eN_j = eN*nDOF_mesh_trial_element+j;
		  element_phi_s[j] = phi_solid_nodes.data()[u_l2g.data()[eN_j]];
		}
	      double element_nodes[nDOF_mesh_trial_element*3];
	      for (int i=0;i<nDOF_mesh_trial_element;i++)
		{
		  int eN_i=eN*nDOF_mesh_trial_element+i;
		  for(int I=0;I<3;I++)
		    element_nodes[i*3 + I] = mesh_dof[mesh_l2g.data()[eN_i]*3 + I];
		}//i
	      double mesh_dof_ref[12]={0.,0.,0.,1.,0.,0.,0.,1.,0.,0.,0.,1.};
	      double xb_ref_calc[nQuadraturePoints_elementBoundary*3];
	      for  (int kb=0;kb<nQuadraturePoints_elementBoundary;kb++)
		{
		  double x=0.0,y=0.0,z=0.0;
		  for (int j=0;j<nDOF_mesh_trial_element;j++)
		    {
		      int ebN_local_kb = ebN_local*nQuadraturePoints_elementBoundary+kb;
		      int ebN_local_kb_j = ebN_local_kb*nDOF_mesh_trial_element+j;
		      x += mesh_dof_ref[j*3+0]*mesh_trial_trace_ref.data()[ebN_local_kb_j]; 
		      y += mesh_dof_ref[j*3+1]*mesh_trial_trace_ref.data()[ebN_local_kb_j]; 
		      z += mesh_dof_ref[j*3+2]*mesh_trial_trace_ref.data()[ebN_local_kb_j];
		    }
		  xb_ref_calc[3*kb+0] = x;
		  xb_ref_calc[3*kb+1] = y;
		  xb_ref_calc[3*kb+2] = z;
		}
	      int icase_s = gf_s.calculate(element_phi_s, element_nodes, xb_ref_calc,true);
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
                    f_ext[nSpace],
                    df_ext[nSpace],
                    dflux_u_u_ext=0.0,
                    bc_u_ext=0.0,
                    //bc_grad_u_ext[nSpace],
                    bc_m_ext=0.0,
                    bc_dm_ext=0.0,
                    bc_f_ext[nSpace],
                    bc_df_ext[nSpace],
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
                    //VRANS
                    porosity_ext,
                    //
                    G[nSpace*nSpace],G_dd_G,tr_G;
		  gf_s.set_boundary_quad(kb);
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
                  //std::cout<<"J mtsqrdet "<<metricTensorDetSqrt<<" integralScaling "<<integralScaling<<std::endl;
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
                  bc_u_ext = isDOFBoundary_u.data()[ebNE_kb]*ebqe_bc_u_ext.data()[ebNE_kb]+(1-isDOFBoundary_u.data()[ebNE_kb])*u_ext;
                  //VRANS
                  porosity_ext = ebqe_porosity_ext.data()[ebNE_kb];
                  //
                  //
                  //calculate the internal and external trace of the pde coefficients
                  //
                  evaluateCoefficients(&ebqe_velocity_ext.data()[ebNE_kb_nSpace],
                                       u_ext,
                                       //VRANS
                                       porosity_ext,
                                       //
                                       m_ext,
                                       dm_ext,
                                       f_ext,
                                       df_ext);
                  evaluateCoefficients(&ebqe_velocity_ext.data()[ebNE_kb_nSpace],
                                       bc_u_ext,
                                       //VRANS
                                       porosity_ext,
                                       //
                                       bc_m_ext,
                                       bc_dm_ext,
                                       bc_f_ext,
                                       bc_df_ext);
                  //
                  //moving domain
                  //
                  double mesh_velocity[3];
                  mesh_velocity[0] = xt_ext;
                  mesh_velocity[1] = yt_ext;
                  mesh_velocity[2] = zt_ext;
                  //std::cout<<"ext J mesh_velocity"<<std::endl;
                  for (int I=0;I<nSpace;I++)
                    {
                      //std::cout<<mesh_velocity[I]<<std::endl;
                      f_ext[I] -= MOVING_DOMAIN*m_ext*mesh_velocity[I];
                      df_ext[I] -= MOVING_DOMAIN*dm_ext*mesh_velocity[I];
                      bc_f_ext[I] -= MOVING_DOMAIN*bc_m_ext*mesh_velocity[I];
                      bc_df_ext[I] -= MOVING_DOMAIN*bc_dm_ext*mesh_velocity[I];
                    }
                  //
                  //calculate the numerical fluxes
                  //
                  exteriorNumericalAdvectiveFluxDerivative(isDOFBoundary_u.data()[ebNE_kb],
                                                           isFluxBoundary_u.data()[ebNE_kb],
                                                           normal,
                                                           df_ext,//VRANS holds porosity
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
		  const double H_s = gf_s.H(0.0, ebqe_phi_s[ebNE_kb]);
		  if (isActiveElement[eN])
		    {
		      for (int i=0;i<nDOF_test_element;i++)
			{
			  int eN_i = eN*nDOF_test_element+i;
			  //int ebNE_kb_i = ebNE_kb*nDOF_test_element+i;
			  for (int j=0;j<nDOF_trial_element;j++)
			    {
			      int ebN_i_j = ebN*4*nDOF_test_X_trial_element + i*nDOF_trial_element + j;
			      globalJacobian.data()[csrRowIndeces_u_u.data()[eN_i] + csrColumnOffsets_eb_u_u.data()[ebN_i_j]] += H_s*fluxJacobian_u_u[j]*u_test_dS[i];
			    }//j
			}//i
		    }
                }//kb
            }//ebNE
      }//computeJacobian

      void FCTStep(arguments_dict& args)
      {
        double dt = args.scalar<double>("dt");
        int NNZ = args.scalar<int>("NNZ");
        int numDOFs = args.scalar<int>("numDOFs");
        proteus::pyarray<double>& lumped_mass_matrix = args.array<double>("lumped_mass_matrix");
        proteus::pyarray<double>& soln = args.array<double>("soln");
        proteus::pyarray<double>& solH = args.array<double>("solH");
        proteus::pyarray<double>& uLow = args.array<double>("uLow");
        proteus::pyarray<double>& dLow = args.array<double>("dLow");
        proteus::pyarray<double>& limited_solution = args.array<double>("limited_solution");
        proteus::pyarray<int>& csrRowIndeces_DofLoops = args.array<int>("csrRowIndeces_DofLoops");
        proteus::pyarray<int>& csrColumnOffsets_DofLoops = args.array<int>("csrColumnOffsets_DofLoops");
        proteus::pyarray<double>& MassMatrix = args.array<double>("MassMatrix");
        proteus::pyarray<double>& dt_times_dH_minus_dL = args.array<double>("dt_times_dH_minus_dL");
        proteus::pyarray<double>& min_u_bc = args.array<double>("min_u_bc");
        proteus::pyarray<double>& max_u_bc = args.array<double>("max_u_bc");
        int LUMPED_MASS_MATRIX = args.scalar<int>("LUMPED_MASS_MATRIX");
        int STABILIZATION_TYPE = args.scalar<int>("STABILIZATION_TYPE");
        Rpos.resize(numDOFs,0.0);
        Rneg.resize(numDOFs,0.0);
        FluxCorrectionMatrix.resize(NNZ,0.0);
        //////////////////
        // LOOP in DOFs //
        //////////////////
        int ij=0;
        for (int i=0; i<numDOFs; i++)
          {
            //read some vectors
            double solHi = solH.data()[i];
            double solni = soln.data()[i];
            double mi = lumped_mass_matrix.data()[i];
            double uLowi = uLow.data()[i];
            double uDotLowi = (uLowi - solni)/dt;
            double mini=min_u_bc.data()[i], maxi=max_u_bc.data()[i]; // init min/max with value at BCs (NOTE: if no boundary then min=1E10, max=-1E10)
            if (GLOBAL_FCT==1)
              {
                mini = 0.;
                maxi = 1.;
              }

            double Pposi=0, Pnegi=0;
            // LOOP OVER THE SPARSITY PATTERN (j-LOOP)//
            for (int offset=csrRowIndeces_DofLoops.data()[i]; offset<csrRowIndeces_DofLoops.data()[i+1]; offset++)
              {
                int j = csrColumnOffsets_DofLoops.data()[offset];
                double solnj = soln.data()[j];
                ////////////////////////
                // COMPUTE THE BOUNDS //
                ////////////////////////
                if (GLOBAL_FCT == 0)
                  {
                    mini = fmin(mini,solnj);
                    maxi = fmax(maxi,solnj);
                  }
                double uLowj = uLow.data()[j];
                double uDotLowj = (uLowj - solnj)/dt;
                // i-th row of flux correction matrix
                if (STABILIZATION_TYPE==4) // DK high-order, linearly stable anti-dif. flux
                  {
                    FluxCorrectionMatrix[ij] = dt*(MassMatrix.data()[ij]*(uDotLowi-uDotLowj)
                                                   + dLow.data()[ij]*(uLowi-uLowj));
                  }
                else
                  {
                    double ML_minus_MC =
                      (LUMPED_MASS_MATRIX == 1 ? 0. : (i==j ? 1. : 0.)*mi - MassMatrix.data()[ij]);
                    FluxCorrectionMatrix[ij] = ML_minus_MC * (solH.data()[j]-solnj - (solHi-solni))
                      + dt_times_dH_minus_dL.data()[ij]*(solnj-solni);
                  }

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
            double Qposi = mi*(maxi-uLow.data()[i]);
            double Qnegi = mi*(mini-uLow.data()[i]);

            ///////////////////////
            // COMPUTE R VECTORS //
            ///////////////////////
            Rpos[i] = ((Pposi==0) ? 1. : fmin(1.0,Qposi/Pposi));
            Rneg[i] = ((Pnegi==0) ? 1. : fmin(1.0,Qnegi/Pnegi));
          } // i DOFs

        //////////////////////
        // COMPUTE LIMITERS //
        //////////////////////
        ij=0;
        for (int i=0; i<numDOFs; i++)
          {
            double ith_Limiter_times_FluxCorrectionMatrix = 0.;
            double Rposi = Rpos[i], Rnegi = Rneg[i];
            // LOOP OVER THE SPARSITY PATTERN (j-LOOP)//
            for (int offset=csrRowIndeces_DofLoops.data()[i]; offset<csrRowIndeces_DofLoops.data()[i+1]; offset++)
              {
                int j = csrColumnOffsets_DofLoops.data()[offset];
                double Lij = 1;
                Lij = ((FluxCorrectionMatrix[ij]>0) ? fmin(Rposi,Rneg[j]) : fmin(Rnegi,Rpos[j]));
                ith_Limiter_times_FluxCorrectionMatrix += Lij * FluxCorrectionMatrix[ij];
                //update ij
                ij+=1;
              }
            limited_solution.data()[i] = uLow.data()[i] + 1./lumped_mass_matrix.data()[i]*ith_Limiter_times_FluxCorrectionMatrix;
          }
      }

      void calculateResidualEdgeBased(arguments_dict& args)
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
        const proteus::pyarray<double>& q_porosity = args.array<double>("q_porosity");
        const proteus::pyarray<double>& porosity_dof = args.array<double>("porosity_dof");
        proteus::pyarray<int>& u_l2g = args.array<int>("u_l2g");
        proteus::pyarray<int>& r_l2g = args.array<int>("r_l2g");
        proteus::pyarray<double>& elementDiameter = args.array<double>("elementDiameter");
        double degree_polynomial = args.scalar<double>("degree_polynomial");
        proteus::pyarray<double>& u_dof = args.array<double>("u_dof");
        proteus::pyarray<double>& u_dof_old = args.array<double>("u_dof_old");
        proteus::pyarray<double>& velocity = args.array<double>("velocity");
        proteus::pyarray<double>& q_m = args.array<double>("q_m");
        proteus::pyarray<double>& q_u = args.array<double>("q_u");
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
        const proteus::pyarray<double>& ebqe_porosity_ext = args.array<double>("ebqe_porosity_ext");
        proteus::pyarray<int>& isDOFBoundary_u = args.array<int>("isDOFBoundary_u");
        proteus::pyarray<double>& ebqe_bc_u_ext = args.array<double>("ebqe_bc_u_ext");
        proteus::pyarray<int>& isFluxBoundary_u = args.array<int>("isFluxBoundary_u");
        proteus::pyarray<double>& ebqe_bc_flux_u_ext = args.array<double>("ebqe_bc_flux_u_ext");
        proteus::pyarray<double>& ebqe_phi = args.array<double>("ebqe_phi");
        double epsFact = args.scalar<double>("epsFact");
        proteus::pyarray<double>& ebqe_u = args.array<double>("ebqe_u");
        proteus::pyarray<double>& ebqe_flux = args.array<double>("ebqe_flux");
        int stage = args.scalar<int>("stage");
        proteus::pyarray<double>&  uTilde_dof = args.array<double>("uTilde_dof");
        double cE = args.scalar<double>("cE");
        double cMax = args.scalar<double>("cMax");
        double cK = args.scalar<double>("cK");
        double uL = args.scalar<double>("uL");
        double uR = args.scalar<double>("uR");
        int numDOFs = args.scalar<int>("numDOFs");
        int NNZ = args.scalar<int>("NNZ");
        proteus::pyarray<int>& csrRowIndeces_DofLoops = args.array<int>("csrRowIndeces_DofLoops");
        proteus::pyarray<int>& csrColumnOffsets_DofLoops = args.array<int>("csrColumnOffsets_DofLoops");
        proteus::pyarray<int>& csrRowIndeces_CellLoops = args.array<int>("csrRowIndeces_CellLoops");
        proteus::pyarray<int>& csrColumnOffsets_CellLoops = args.array<int>("csrColumnOffsets_CellLoops");
        proteus::pyarray<int>& csrColumnOffsets_eb_CellLoops = args.array<int>("csrColumnOffsets_eb_CellLoops");
        proteus::pyarray<double>& ML = args.array<double>("ML");
        int LUMPED_MASS_MATRIX = args.scalar<int>("LUMPED_MASS_MATRIX");
        int STABILIZATION_TYPE = args.scalar<int>("STABILIZATION_TYPE");
        int ENTROPY_TYPE = args.scalar<int>("ENTROPY_TYPE");
        proteus::pyarray<double>& uLow = args.array<double>("uLow");
        proteus::pyarray<double>& dLow = args.array<double>("dLow");
        proteus::pyarray<double>& dt_times_dH_minus_dL = args.array<double>("dt_times_dH_minus_dL");
        proteus::pyarray<double>& min_u_bc = args.array<double>("min_u_bc");
        proteus::pyarray<double>& max_u_bc = args.array<double>("max_u_bc");
        proteus::pyarray<double>& quantDOFs = args.array<double>("quantDOFs");
        // NOTE: This function follows a different (but equivalent) implementation of the smoothness based indicator than NCLS.h
        // Allocate space for the transport matrices
        // This is used for first order KUZMIN'S METHOD
        TransportMatrix.resize(NNZ,0.0);
        TransposeTransportMatrix.resize(NNZ,0.0);
        // compute entropy and init global_entropy_residual and boundary_integral
        psi.resize(numDOFs,0.0);
        eta.resize(numDOFs,0.0);
        global_entropy_residual.resize(numDOFs,0.0);
        boundary_integral.resize(numDOFs,0.0);

        for (int i=0; i<numDOFs; i++)
          {
            // NODAL ENTROPY //
            if (STABILIZATION_TYPE==2) //EV stab
              {
                double porosity_times_solni = porosity_dof.data()[i]*u_dof_old.data()[i];
                eta[i] = ENTROPY_TYPE == 0 ? ENTROPY(porosity_times_solni,uL,uR) : ENTROPY_LOG(porosity_times_solni,uL,uR);
                global_entropy_residual[i]=0.;
              }
            boundary_integral[i]=0.;
          }

        //////////////////////////////////////////////
        // ** LOOP IN CELLS FOR CELL BASED TERMS ** //
        //////////////////////////////////////////////
        // HERE WE COMPUTE:
        //    * Time derivative term. porosity*u_t
        //    * cell based CFL (for reference)
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
            for (int i=0;i<nDOF_test_element;i++)
              {
                elementResidual_u[i]=0.0;
                element_entropy_residual[i]=0.0;
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
                  // for entropy residual
                  aux_entropy_residual=0., DENTROPY_un, DENTROPY_uni,
                  //for mass matrix contributions
                  u=0.0, un=0.0, grad_un[nSpace], porosity_times_velocity[nSpace],
                  u_test_dV[nDOF_trial_element],
                  u_grad_trial[nDOF_trial_element*nSpace],
                  u_grad_test_dV[nDOF_test_element*nSpace],
                  //for general use
                  jac[nSpace*nSpace], jacDet, jacInv[nSpace*nSpace],
                  dV,x,y,z,xt,yt,zt,
                  //VRANS
                  porosity;
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
                ck.calculateMappingVelocity_element(eN,
                                                    k,
                                                    mesh_velocity_dof.data(),
                                                    mesh_l2g.data(),
                                                    mesh_trial_ref.data(),
                                                    xt,yt,zt);
                dV = fabs(jacDet)*dV_ref.data()[k];
                //get the solution (of Newton's solver). To compute time derivative term
                ck.valFromDOF(u_dof.data(),&u_l2g.data()[eN_nDOF_trial_element],&u_trial_ref.data()[k*nDOF_trial_element],u);
                //get the solution at quad point at tn and tnm1 for entropy viscosity
                ck.valFromDOF(u_dof_old.data(),&u_l2g.data()[eN_nDOF_trial_element],&u_trial_ref.data()[k*nDOF_trial_element],un);
                //get the solution gradients at tn for entropy viscosity
                ck.gradTrialFromRef(&u_grad_trial_ref.data()[k*nDOF_trial_element*nSpace],jacInv,u_grad_trial);
                ck.gradFromDOF(u_dof_old.data(),&u_l2g.data()[eN_nDOF_trial_element],u_grad_trial,grad_un);

                //precalculate test function products with integration weights for mass matrix terms
                for (int j=0;j<nDOF_trial_element;j++)
                  {
                    u_test_dV[j] = u_test_ref.data()[k*nDOF_trial_element+j]*dV;
                    for (int I=0;I<nSpace;I++)
                      u_grad_test_dV[j*nSpace+I] = u_grad_trial[j*nSpace+I]*dV;//cek warning won't work for Petrov-Galerkin
                  }

                //calculate time derivative at quadrature points
                if (q_dV_last.data()[eN_k] <= -100)
                  q_dV_last.data()[eN_k] = dV;
                q_dV.data()[eN_k] = dV;
                //VRANS
                porosity = q_porosity.data()[eN_k];
                //
                //moving mesh
                //
                double mesh_velocity[3];
                mesh_velocity[0] = xt;
                mesh_velocity[1] = yt;
                mesh_velocity[2] = zt;
                //relative velocity at tn
                for (int I=0;I<nSpace;I++)
                  porosity_times_velocity[I] = porosity*(velocity.data()[eN_k_nSpace+I]-MOVING_DOMAIN*mesh_velocity[I]);

                //////////////////////////////
                // CALCULATE CELL BASED CFL //
                //////////////////////////////
                calculateCFL(elementDiameter.data()[eN]/degree_polynomial,porosity_times_velocity,cfl.data()[eN_k]);

                //////////////////////////////////////////////
                // CALCULATE ENTROPY RESIDUAL AT QUAD POINT //
                //////////////////////////////////////////////
                if (STABILIZATION_TYPE==2) // EV stab
                  {
                    for (int I=0;I<nSpace;I++)
                      aux_entropy_residual += porosity_times_velocity[I]*grad_un[I];
                    DENTROPY_un = ENTROPY_TYPE==0 ? DENTROPY(porosity*un,uL,uR) : DENTROPY_LOG(porosity*un,uL,uR);
                  }
                //////////////
                // ith-LOOP //
                //////////////
                for(int i=0;i<nDOF_test_element;i++)
                  {
                    // VECTOR OF ENTROPY RESIDUAL //
                    int eN_i=eN*nDOF_test_element+i;
                    if (STABILIZATION_TYPE==2) // EV stab
                      {
                        int gi = offset_u+stride_u*u_l2g.data()[eN_i]; //global i-th index
                        double porosity_times_uni = porosity_dof.data()[gi]*u_dof_old.data()[gi];
                        DENTROPY_uni = ENTROPY_TYPE == 0 ? DENTROPY(porosity_times_uni,uL,uR) : DENTROPY_LOG(porosity_times_uni,uL,uR);
                        element_entropy_residual[i] += (DENTROPY_un - DENTROPY_uni)*aux_entropy_residual*u_test_dV[i];
                      }
                    elementResidual_u[i] += porosity*(u-un)*u_test_dV[i];
                    ///////////////
                    // j-th LOOP // To construct transport matrices
                    ///////////////
                    for(int j=0;j<nDOF_trial_element;j++)
                      {
                        int j_nSpace = j*nSpace;
                        int i_nSpace = i*nSpace;
                        elementTransport[i][j] += // -int[(vel.grad_wi)*wj*dx]
                          ck.AdvectionJacobian_weak(porosity_times_velocity,
                                                    u_trial_ref.data()[k*nDOF_trial_element+j],&u_grad_test_dV[i_nSpace]);
                        elementTransposeTransport[i][j] += // -int[(vel.grad_wj)*wi*dx]
                          ck.AdvectionJacobian_weak(porosity_times_velocity,
                                                    u_trial_ref.data()[k*nDOF_trial_element+i],&u_grad_test_dV[j_nSpace]);
                      }
                  }//i
                //save solution for other models
                q_u.data()[eN_k] = u;
                q_m.data()[eN_k] = porosity*u;
              }
            /////////////////
            // DISTRIBUTE // load cell based element into global residual
            ////////////////
            for(int i=0;i<nDOF_test_element;i++)
              {
                int eN_i=eN*nDOF_test_element+i;
                int gi = offset_u+stride_u*u_l2g.data()[eN_i]; //global i-th index

                // distribute global residual for (lumped) mass matrix
                globalResidual.data()[gi] += elementResidual_u[i];
                // distribute entropy_residual
                if (STABILIZATION_TYPE==2) // EV Stab
                  global_entropy_residual[gi] += element_entropy_residual[i];

                // distribute transport matrices
                for (int j=0;j<nDOF_trial_element;j++)
                  {
                    int eN_i_j = eN_i*nDOF_trial_element+j;
                    TransportMatrix[csrRowIndeces_CellLoops.data()[eN_i] +
                                    csrColumnOffsets_CellLoops.data()[eN_i_j]] += elementTransport[i][j];
                    TransposeTransportMatrix[csrRowIndeces_CellLoops.data()[eN_i] +
                                             csrColumnOffsets_CellLoops.data()[eN_i_j]]
                      += elementTransposeTransport[i][j];
                  }//j
              }//i
          }//elements

        //////////////////////////////////////////////////////////////////////////////////////////
        // ADD OUTFLOW BOUNDARY TERM TO TRANSPORT MATRICES AND COMPUTE INFLOW BOUNDARY INTEGRAL //
        //////////////////////////////////////////////////////////////////////////////////////////
        //   * Compute outflow boundary integral as a matrix; i.e., int_B[ (vel.normal)*wi*wj*dx]
        for (int ebNE = 0; ebNE < nExteriorElementBoundaries_global; ebNE++)
          {
            double min_u_bc_local = 1E10, max_u_bc_local = -1E10;
            int ebN = exteriorElementBoundariesArray.data()[ebNE];
            int eN  = elementBoundaryElementsArray.data()[ebN*2+0],
              ebN_local = elementBoundaryLocalElementBoundariesArray.data()[ebN*2+0],
              eN_nDOF_trial_element = eN*nDOF_trial_element;
            double elementResidual_u[nDOF_test_element];
            for (int i=0;i<nDOF_test_element;i++)
              elementResidual_u[i]=0.0;
            // loop on quad points
            for  (int kb=0;kb<nQuadraturePoints_elementBoundary;kb++)
              {
                int ebNE_kb = ebNE*nQuadraturePoints_elementBoundary+kb,
                  ebNE_kb_nSpace = ebNE_kb*nSpace,
                  ebN_local_kb = ebN_local*nQuadraturePoints_elementBoundary+kb,
                  ebN_local_kb_nSpace = ebN_local_kb*nSpace;
                double
                  u_ext=0.0, bc_u_ext=0.0,
                  porosity_times_velocity[nSpace],
                  flux_ext=0.0, dflux_ext=0.0,
                  fluxTransport[nDOF_trial_element],
                  jac_ext[nSpace*nSpace],
                  jacDet_ext,
                  jacInv_ext[nSpace*nSpace],
                  boundaryJac[nSpace*(nSpace-1)],
                  metricTensor[(nSpace-1)*(nSpace-1)],
                  metricTensorDetSqrt,
                  dS,
                  u_test_dS[nDOF_test_element],
                  normal[nSpace],x_ext,y_ext,z_ext,xt_ext,yt_ext,zt_ext,integralScaling,porosity_ext;
                // calculate mappings
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
                //compute shape and solution information
                ck.valFromDOF(u_dof.data(),&u_l2g.data()[eN_nDOF_trial_element],&u_trial_trace_ref.data()[ebN_local_kb*nDOF_test_element],u_ext);
                //precalculate test function products with integration weights
                for (int j=0;j<nDOF_trial_element;j++)
                  u_test_dS[j] = u_test_trace_ref.data()[ebN_local_kb*nDOF_test_element+j]*dS;

                //VRANS
                porosity_ext = ebqe_porosity_ext.data()[ebNE_kb];
                //
                //moving mesh
                //
                double mesh_velocity[3];
                mesh_velocity[0] = xt_ext;
                mesh_velocity[1] = yt_ext;
                mesh_velocity[2] = zt_ext;
                //std::cout<<"mesh_velocity ext"<<std::endl;
                for (int I=0;I<nSpace;I++)
                  porosity_times_velocity[I] = porosity_ext*(ebqe_velocity_ext.data()[ebNE_kb_nSpace+I] - MOVING_DOMAIN*mesh_velocity[I]);
                //
                //calculate the fluxes
                //
                double flow = 0.;
                for (int I=0; I < nSpace; I++)
                  flow += normal[I]*porosity_times_velocity[I];

                if (flow >= 0) //outflow. This is handled via the transport matrices. Then flux_ext=0 and dflux_ext!=0
                  {
                    dflux_ext = flow;
                    flux_ext = 0;
                    // save external u
                    ebqe_u.data()[ebNE_kb] = u_ext;
                  }
                else // inflow. This is handled via the boundary integral. Then flux_ext!=0 and dflux_ext=0
                  {
                    dflux_ext = 0;
                    // save external u
                    ebqe_u.data()[ebNE_kb] = isDOFBoundary_u.data()[ebNE_kb]*ebqe_bc_u_ext.data()[ebNE_kb]+(1-isDOFBoundary_u.data()[ebNE_kb])*u_ext;
                    if (isDOFBoundary_u.data()[ebNE_kb] == 1)
                      flux_ext = ebqe_bc_u_ext.data()[ebNE_kb]*flow;
                    else if (isFluxBoundary_u.data()[ebNE_kb] == 1)
                      flux_ext = ebqe_bc_flux_u_ext.data()[ebNE_kb];
                    else
                      {
                        std::cout<<"warning: VOF open boundary with no external trace, setting to zero for inflow"<<std::endl;
                        flux_ext = 0.0;
                      }
                  }

                for (int j=0;j<nDOF_trial_element;j++)
                  {
                    // elementResidual. This is to include the inflow boundary integral.
                    // NOTE: here I assume that we use a Galerkin approach st nDOF_test_element = nDOF_trial_element
                    elementResidual_u[j] += flux_ext*u_test_dS[j];
                    int ebN_local_kb_j=ebN_local_kb*nDOF_trial_element+j;
                    fluxTransport[j] = dflux_ext*u_trial_trace_ref.data()[ebN_local_kb_j];
                  }//j
                ///////////////////////////////////////////////////////
                // DISTRIBUTE OUTFLOW BOUNDARY TO TRANSPORT MATRICES //
                ///////////////////////////////////////////////////////
                for (int i=0;i<nDOF_test_element;i++)
                  {
                    int eN_i = eN*nDOF_test_element+i;
                    for (int j=0;j<nDOF_trial_element;j++)
                      {
                        int ebN_i_j = ebN*4*nDOF_test_X_trial_element + i*nDOF_trial_element + j;
                        TransportMatrix[csrRowIndeces_CellLoops.data()[eN_i] + csrColumnOffsets_eb_CellLoops.data()[ebN_i_j]]
                          += fluxTransport[j]*u_test_dS[i];
                        TransposeTransportMatrix[csrRowIndeces_CellLoops.data()[eN_i] + csrColumnOffsets_eb_CellLoops.data()[ebN_i_j]]
                          += fluxTransport[i]*u_test_dS[j];
                      }//j
                  }//i
                // local min/max at boundary
                min_u_bc_local = fmin(ebqe_u.data()[ebNE_kb], min_u_bc_local);
                max_u_bc_local = fmax(ebqe_u.data()[ebNE_kb], max_u_bc_local);
              }//kb
            // global min/max at boundary
            for (int i=0;i<nDOF_test_element;i++)
              {
                int eN_i = eN*nDOF_test_element+i;
                int gi = offset_u+stride_u*u_l2g.data()[eN_i]; //global i-th index
                globalResidual.data()[gi] += dt*elementResidual_u[i];
                boundary_integral[gi] += elementResidual_u[i];
                min_u_bc[gi] = fmin(min_u_bc_local,min_u_bc[gi]);
                max_u_bc[gi] = fmax(max_u_bc_local,max_u_bc[gi]);
              }
          }//ebNE
        // END OF ADDING BOUNDARY TERM TO TRANSPORT MATRICES and COMPUTING BOUNDARY INTEGRAL //

        /////////////////////////////////////////////////////////////////
        // COMPUTE SMOOTHNESS INDICATOR and NORMALIZE ENTROPY RESIDUAL //
        /////////////////////////////////////////////////////////////////
        // NOTE: see NCLS.h for a different but equivalent implementation of this.
        int ij = 0;
        for (int i=0; i<numDOFs; i++)
          {
            double etaMaxi, etaMini;
            if (STABILIZATION_TYPE==2) //EV
              {
                // For eta min and max
                etaMaxi = fabs(eta[i]);
                etaMini = fabs(eta[i]);
              }
            double porosity_times_solni = porosity_dof.data()[i]*u_dof_old.data()[i];
            // for smoothness indicator //
            double alpha_numerator = 0., alpha_denominator = 0.;
            for (int offset=csrRowIndeces_DofLoops.data()[i]; offset<csrRowIndeces_DofLoops.data()[i+1]; offset++)
              { // First loop in j (sparsity pattern)
                int j = csrColumnOffsets_DofLoops.data()[offset];
                if (STABILIZATION_TYPE==2) //EV Stabilization
                  {
                    // COMPUTE ETA MIN AND ETA MAX //
                    etaMaxi = fmax(etaMaxi,fabs(eta[j]));
                    etaMini = fmin(etaMini,fabs(eta[j]));
                  }
                double porosity_times_solnj = porosity_dof.data()[j]*u_dof_old.data()[j];
                alpha_numerator += porosity_times_solni - porosity_times_solnj;
                alpha_denominator += fabs(porosity_times_solni - porosity_times_solnj);
                //update ij
                ij+=1;
              }
            if (STABILIZATION_TYPE==2) //EV Stab
              {
                // Normalize entropy residual
                global_entropy_residual[i] *= etaMini == etaMaxi ? 0. : 2*cE/(etaMaxi-etaMini);
                quantDOFs[i] = fabs(global_entropy_residual[i]);
              }

            double alphai = alpha_numerator/(alpha_denominator+1E-15);
            quantDOFs[i] = alphai;

            if (POWER_SMOOTHNESS_INDICATOR==0)
              psi[i] = 1.0;
            else
              psi[i] = std::pow(alphai,POWER_SMOOTHNESS_INDICATOR); //NOTE: they use alpha^2 in the paper
          }
        /////////////////////////////////////////////
        // ** LOOP IN DOFs FOR EDGE BASED TERMS ** //
        /////////////////////////////////////////////
        ij=0;
        for (int i=0; i<numDOFs; i++)
          {
            // NOTE: Transport matrices already have the porosity considered. ---> Dissipation matrices as well.
            double solni = u_dof_old.data()[i]; // solution at time tn for the ith DOF
            double porosityi = porosity_dof.data()[i];
            double ith_dissipative_term = 0;
            double ith_low_order_dissipative_term = 0;
            double ith_flux_term = 0;
            double dLii = 0.;

            // loop over the sparsity pattern of the i-th DOF
            for (int offset=csrRowIndeces_DofLoops.data()[i]; offset<csrRowIndeces_DofLoops.data()[i+1]; offset++)
              {
                int j = csrColumnOffsets_DofLoops.data()[offset];
                double solnj = u_dof_old.data()[j]; // solution at time tn for the jth DOF
                double porosityj = porosity_dof.data()[j];
                double dLowij, dLij, dEVij, dHij;

                ith_flux_term += TransportMatrix[ij]*solnj;
                if (i != j) //NOTE: there is really no need to check for i!=j (see formula for ith_dissipative_term)
                  {
                    // artificial compression
                    double solij = 0.5*(porosityi*solni+porosityj*solnj);
                    double Compij = cK*fmax(solij*(1.0-solij),0.0)/(fabs(porosityi*solni-porosityj*solnj)+1E-14);
                    // first-order dissipative operator
                    dLowij = fmax(fabs(TransportMatrix[ij]),fabs(TransposeTransportMatrix[ij]));
                    //dLij = fmax(0.,fmax(psi[i]*TransportMatrix[ij], // Approach by S. Badia
                    //              psi[j]*TransposeTransportMatrix[ij]));
                    dLij = dLowij*fmax(psi[i],psi[j]); // Approach by JLG & BP
                    if (STABILIZATION_TYPE==2) //EV Stab
                      {
                        // high-order (entropy viscosity) dissipative operator
                        dEVij = fmax(fabs(global_entropy_residual[i]),fabs(global_entropy_residual[j]));
                        dHij = fmin(dLowij,dEVij) * fmax(1.0-Compij,0.0); // artificial compression
                      }
                    else // smoothness based indicator
                      {
                        dHij = dLij * fmax(1.0-Compij,0.0); // artificial compression
                      }
                    //dissipative terms
                    ith_dissipative_term += dHij*(solnj-solni);
                    ith_low_order_dissipative_term += dLowij*(solnj-solni);
                    //dHij - dLij. This matrix is needed during FCT step
                    dt_times_dH_minus_dL[ij] = dt*(dHij - dLowij);
                    dLii -= dLij;
                    dLow[ij] = dLowij;
                  }
                else //i==j
                  {
                    // NOTE: this is incorrect. Indeed, dLii = -sum_{j!=i}(dLij) and similarly for dCii.
                    // However, it is irrelevant since during the FCT step we do (dL-dC)*(solnj-solni)
                    dt_times_dH_minus_dL[ij]=0;
                    dLow[ij]=0;
                  }
                //update ij
                ij+=1;
              }
            double mi = ML.data()[i];
            // compute edge_based_cfl
            edge_based_cfl.data()[i] = 2.*fabs(dLii)/mi;
            uLow[i] = u_dof_old.data()[i] - dt/mi*(ith_flux_term
                                            + boundary_integral[i]
                                            - ith_low_order_dissipative_term);

            // update residual
            if (LUMPED_MASS_MATRIX==1)
              globalResidual.data()[i] = u_dof_old.data()[i] - dt/mi*(ith_flux_term
                                                                      + boundary_integral[i]
                                                                      - ith_dissipative_term);
            else
              globalResidual.data()[i] += dt*(ith_flux_term - ith_dissipative_term);
          }
      }
    };//VOF

  inline VOF_base* newVOF(int nSpaceIn,
                          int nQuadraturePoints_elementIn,
                          int nDOF_mesh_trial_elementIn,
                          int nDOF_trial_elementIn,
                          int nDOF_test_elementIn,
                          int nQuadraturePoints_elementBoundaryIn,
                          int CompKernelFlag)
  {
    if (nSpaceIn == 1)
      return proteus::chooseAndAllocateDiscretization1D<VOF_base,VOF,CompKernel>(nSpaceIn,
                                                                                 nQuadraturePoints_elementIn,
                                                                                 nDOF_mesh_trial_elementIn,
                                                                                 nDOF_trial_elementIn,
                                                                                 nDOF_test_elementIn,
                                                                                 nQuadraturePoints_elementBoundaryIn,
                                                                                 CompKernelFlag);
    else if (nSpaceIn == 2)
      return proteus::chooseAndAllocateDiscretization2D<VOF_base,VOF,CompKernel>(nSpaceIn,
                                                                                 nQuadraturePoints_elementIn,
                                                                                 nDOF_mesh_trial_elementIn,
                                                                                 nDOF_trial_elementIn,
                                                                                 nDOF_test_elementIn,
                                                                                 nQuadraturePoints_elementBoundaryIn,
                                                                                 CompKernelFlag);
    else
      return proteus::chooseAndAllocateDiscretization<VOF_base,VOF,CompKernel>(nSpaceIn,
                                                                               nQuadraturePoints_elementIn,
                                                                               nDOF_mesh_trial_elementIn,
                                                                               nDOF_trial_elementIn,
                                                                               nDOF_test_elementIn,
                                                                               nQuadraturePoints_elementBoundaryIn,
                                                                               CompKernelFlag);
  }
}//proteus
#endif
