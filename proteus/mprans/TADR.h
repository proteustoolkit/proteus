#ifndef TADR_H
#define TADR_H
#include <cmath>
#include <iomanip>
#include <iostream>
#include <valarray>
#include "CompKernel.h"
#include "ModelFactory.h"
#include "ArgumentsDict.h"
#include "pyarray.h"
#define nnz nSpace

namespace py = pybind11;

#define POWER_SMOOTHNESS_INDICATOR 2
#define IS_BETAij_ONE 0

// Cell based methods:
//    * Galerkin (unstabilized)
//    * VMS(SUPG) with BDF1 or BDF2 time integration
//    * Explicit Taylor Galerkin with EV stabilization
// Edge based methods.
//    Low order via D. Kuzmin's
//    High order methods: Smoothness indicator with MC, EV commutator with MC, D.K with ML
//    Zalesak's FCT

namespace proteus
{
  // ImplicitEV=5: backward-Euler implicit edge-based scheme (low-order graph
  // dissipation, no FCT) modeled on Richards STABILIZATION_TYPE==2.  Unlike the
  // explicit edge-based types (2,3,4) the advection/diffusion are evaluated at
  // the current Newton iterate and contribute to the Jacobian, so it is NOT
  // CFL-limited.  The Jacobian is assembled in calculateJacobian (a separate
  // function), NOT inside calculateResidual as Richards does.
  // Compatible=6: implicit transport driven by the flow model's realized exchanges; see
  // calculateResidual_compatible.
  enum class STABILIZATION : int { Galerkin=-1, VMS=0, TaylorGalerkinEV=1, EntropyViscosity=2, SmoothnessIndicator=3, Kuzmin=4, ImplicitEV=5, Compatible=6};
  enum class ENTROPY : int { POWER=0, LOG=1};
  enum class DISPERSION : int { Constant=0, PowerLawSaturation=1, VelocityBased=2};
  // Power entropy //
  inline double EPOWER(const double& phi, const double& phiL, const double& phiR)
  {
    return 1./2.*std::pow(fabs(phi),2.);
  }
  inline double DEPOWER(const double& phi, const double& phiL, const double& phiR)
  {
    return fabs(phi)*(phi>=0 ? 1 : -1);
  }
  // Log entropy // for level set from 0 to 1
  inline double ELOG(const double& phi, const double& phiL, const double& phiR)
  {
    return std::log(fabs((phi-phiL)*(phiR-phi))+1E-14);
  }
  inline double DELOG(const double& phi, const double& phiL, const double& phiR)
  {
    return (phiL+phiR-2*phi)*((phi-phiL)*(phiR-phi)>=0 ? 1 : -1)/(fabs((phi-phiL)*(phiR-phi))+1E-14);
  }
}

namespace proteus
{
  class TADR_base
  {
    //The base class defining the interface
  public:
    std::valarray<double> Rpos, Rneg;
    std::valarray<double> FluxCorrectionMatrix;
    std::valarray<double> TransportMatrix, DiffusionMatrix, TransposeTransportMatrix;
    std::valarray<double> psi, eta, global_entropy_residual, boundary_integral;
    std::valarray<double> m_dof, theta_dof_proj, rho_dof_proj, ML_mass_proj;
    std::valarray<double> maxVel,maxEntRes;
    virtual ~TADR_base(){}
    virtual void calculateResidual(arguments_dict& args)=0;
    virtual void calculateJacobian(arguments_dict& args)=0;
    virtual void invert(arguments_dict& args)=0;
    virtual void FCTStep(arguments_dict& args)=0;
    virtual void calculateResidual_compatible(arguments_dict& args)=0;
    virtual void reconstructFlux_compatible(arguments_dict& args)=0;
    virtual void FCTStep_compatible(arguments_dict& args)=0;
  };

  template<class CompKernelType,
           int nSpace,
           int nQuadraturePoints_element,
           int nDOF_mesh_trial_element,
           int nDOF_trial_element,
           int nDOF_test_element,
           int nQuadraturePoints_elementBoundary>
  class TADR : public TADR_base
  {
  public:
    const int nDOF_test_X_trial_element;
    CompKernelType ck;
    TADR():
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
void evaluateCoefficients(const int rowptr[nSpace],
				                      const int colind[nnz],
                              const double v[nSpace],
                              const double alpha_L,
                              const double alpha_T,
                              const double Dm,
                              const double thetaW,
                              const double rho_f,
                              const double rho_s,
                              const double& u,
                              double& rho_out,
                              double& m,
                              double& dm,
                              double f[nSpace],
                              double df[nSpace], 
                              double a[nnz],
                              double da[nnz])
    {
      const double epsilon = (rho_s - rho_f)/rho_f;
      rho_out = rho_f*(1.0 + epsilon*u);
      const double drho_du = rho_f*epsilon;
      const double rho_transport = rho_out;
      m = thetaW*rho_transport*u;
      dm = thetaW*(rho_transport + u*drho_du);
      double v_mag = 0.0;
      for (int I=0; I<nSpace; I++)
        v_mag += v[I]*v[I];
      v_mag = std::sqrt(v_mag);
      const double v_pore_mag = (thetaW > 1.0e-8) ? v_mag / thetaW : 0.0;

      double alpha_L_eff = alpha_L;
      double alpha_T_eff = alpha_T;
      double v_unit[nSpace] = {0.0};
      if (v_mag > 1.0e-10)
        for (int I=0; I<nSpace; I++)
          v_unit[I] = v[I]/v_mag;

      for (int I=0; I < nSpace; I++)
        {
          f[I] = rho_transport*v[I]*u;
          df[I] = (rho_transport + u*drho_du)*v[I];
          for (int ii = rowptr[I]; ii < rowptr[I + 1]; ii++)
            {
              const int J = colind[ii];
              const double deltaIJ = (I == J) ? 1.0 : 0.0;
              const double dispersion_tensor =
                Dm*deltaIJ +
                alpha_L_eff*v_unit[I]*v_unit[J]*v_pore_mag +
                alpha_T_eff*v_pore_mag*(deltaIJ - v_unit[I]*v_unit[J]);
              a[ii] = thetaW*rho_transport*dispersion_tensor;
              da[ii] = thetaW*drho_du*dispersion_tensor;
            }

        }
    }
inline
double inversevaluateCoefficients(const double storage,
                             const double porosity,
                             const double rho_f,
                             const double rho_s) 
    {
      const double mass_scale = std::max(porosity*rho_f, 1.0e-14);
      const double epsilon = (rho_s - rho_f)/rho_f;
      const double rhs = storage/mass_scale;
      if (std::fabs(epsilon) < 1.0e-14)
        return rhs;
      const double discriminant = std::max(1.0 + 4.0*epsilon*rhs, 0.0);
      const double sqrt_discriminant = std::sqrt(discriminant);
      if (epsilon > 0.0)
        return 2.0*rhs/(1.0 + sqrt_discriminant);
      return (-1.0 + sqrt_discriminant)/(2.0*epsilon);
    }

inline
    void exteriorNumericalDiffusiveFlux(int* rowptr,
					int* colind,
					const int& isDOFBoundary,
					const int& isDiffusiveFluxBoundary,
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
      double diffusiveVelocityComponent_I;
      double penaltyFlux;
      double max_a;
      if (isDiffusiveFluxBoundary == 1)
	{
	  flux = bc_flux;
	}
      else if (isDOFBoundary == 1)
	{
	  flux = 0.0;
	  max_a = 0.0;
	  for (int I = 0; I < nSpace; I++)
	    {
	      diffusiveVelocityComponent_I = 0.0;
	      for (int m = rowptr[I]; m < rowptr[I+1]; m++)
		{
		  diffusiveVelocityComponent_I -= a[m] * grad_potential[colind[m]];
		  max_a = fmax(max_a, a[m]);
		}
	      flux += diffusiveVelocityComponent_I * n[I];
	    }
	  penaltyFlux = max_a * penalty * (u - bc_u);
	  flux += penaltyFlux;
	}
      else
	{
	  //std::cerr << "warning, diffusion term with no boundary condition set, setting diffusive flux to 0.0" << std::endl;
	  flux = 0.0;
	}
    }

    inline
    double ExteriorNumericalDiffusiveFluxJacobian(int* rowptr,
						  int* colind,
						  const int& isDOFBoundary,
						  const int& isDiffusiveFluxBoundary,
						  const double n[nSpace],
						  double* a,
						  const double& v,
						  const double grad_v[nSpace],
						  const double& penalty)
    {
      double dvel_I;
      double tmp = 0.0;
      double max_a = 0.0;
      if ((isDiffusiveFluxBoundary == 0) && (isDOFBoundary == 1))
	{
	  for (int I = 0; I < nSpace; I++)
	    {
	      dvel_I = 0.0;
	      for (int m = rowptr[I]; m < rowptr[I + 1]; m++)
		{
		  dvel_I -= a[m] * grad_v[colind[m]];
		  max_a = fmax(max_a, a[m]);
		}
	      tmp += dvel_I * n[I];
	    }
	  tmp += max_a * penalty * v;
	}
      return tmp;
    }

    inline
    void calculateSubgridError_tau(const double& elementDiameter,
                                   const double& dmt,
                                   const double df[nSpace],
                                   double& cfl,
                                   double& tau)
    {
      //regular elements
      double h,nrm_v,oneByAbsdt;
      h = elementDiameter;
      nrm_v=0.0;
      for(int I=0;I<nSpace;I++)
        nrm_v+=df[I]*df[I];
      nrm_v = sqrt(nrm_v);
      cfl = nrm_v/h;
      oneByAbsdt =  fabs(dmt);
      tau = 1.0/(2.0*nrm_v/h + oneByAbsdt + 1.0e-8);
    }

    inline
    void calculateSubgridError_tau(const double&  Ct_sge,
                                   const double   G[nSpace*nSpace],
                                   const double&  A0,
                                   const double   Ai[nSpace],
                                   double& tau_v,
                                   double& cfl)
    {
      //metric-based tau for arbitrarily shaped elements
      double v_d_Gv=0.0;
      for(int I=0;I<nSpace;I++)
        {for (int J=0;J<nSpace;J++)
            v_d_Gv += Ai[I]*G[I*nSpace+J]*Ai[J];
          v_d_Gv += Ai[I]*G[I*nSpace+I];
          for(int J=0;J<nSpace;J++)
            {
              if(J!=I)
                v_d_Gv += 2.0*Ai[I]*G[I*nSpace+J];
            }
        }
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
                                        const int& forceStrongConditions,
                                        const double n[nSpace],
                                        const double& bc_flux_u,
                                        const double f[nSpace],
                                        const double bc_f[nSpace],
                                        const double velocity[nSpace],
                                        double& flux)
    {
      
      double flow=0.0;
      for (int I=0; I < nSpace; I++)
        flow += n[I]*velocity[I];

      if (isDOFBoundary_u == 1)
        {
          flux = 0.0;
          if (forceStrongConditions == 1)
            for (int I=0; I < nSpace; I++) flux += n[I]*bc_f[I];
          else if (flow >= 0.0)
            for (int I=0; I < nSpace; I++) flux += n[I]*f[I];
          else
            for (int I=0; I < nSpace; I++) flux += n[I]*bc_f[I];
        }
      else if (isFluxBoundary_u == 1)
        {
          flux = bc_flux_u;
        }
      else
        {
          flux = 0.0;
          if (flow >= 0.0)
            for (int I=0; I < nSpace; I++) flux += n[I]*f[I];
          // else: open boundary with inflow, no external trace — flux = 0
        }
    }

    inline
    void exteriorNumericalAdvectiveFluxDerivative(const int& isDOFBoundary_u,
                                                  const int& isFluxBoundary_u,
                                                  const int& forceStrongConditions,
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
          if (forceStrongConditions == 1)
            {
              dflux = 0.0;
            }
          else if (flow >= 0.0)
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
inline

  void exteriorNumericalDiffusiveFluxDerivative(const int& isDOFBoundary,
                                                const int& isDiffusiveFluxBoundary,
                                                const int rowptr[nSpace],
                                                const int colind[nnz],
                                                const double n[nSpace],
                                                const double a[nnz],
                                                const double da[nnz],
                                                const double grad_psi[nSpace],
                                                const double grad_v[nSpace],
                                                const double& v,
                                                const double penalty,
                                                double& fluxJacobian)
{
    if (isDiffusiveFluxBoundary == 0 && isDOFBoundary == 1)
    {
        fluxJacobian = 0.0;
        double max_a = 0.0;
        for (int I = 0; I < nSpace; I++) {
            for(int m=rowptr[I]; m<rowptr[I+1]; m++)
        {
            max_a = fmax(max_a, a[m]);
            fluxJacobian -= (a[m] * grad_v[colind[m]] + da[m] * v * grad_psi[colind[m]]) * n[I];
        }
            fluxJacobian += max_a * penalty * v;
        }
    }
    else
    {
        fluxJacobian = 0.0;
    }
}

    
 void calculateResidual(arguments_dict& args)
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
      double degree_polynomial = args.scalar<double>("degree_polynomial");
      proteus::pyarray<double>& u_dof = args.array<double>("u_dof");
      proteus::pyarray<double>& u_dof_old = args.array<double>("u_dof_old");
      proteus::pyarray<double>& velocity = args.array<double>("velocity");
      proteus::pyarray<double>& velocity_old = args.array<double>("velocity_old");
      proteus::pyarray<double>& q_m = args.array<double>("q_m");
      proteus::pyarray<double>& q_u = args.array<double>("q_u");
      proteus::pyarray<double>& q_porosity = args.array<double>("q_porosity");
      proteus::pyarray<double>& q_porosity_old = args.array<double>("q_porosity_old");
      proteus::pyarray<double>& q_rho = args.array<double>("q_rho");
      proteus::pyarray<double>& q_rho_old = args.array<double>("q_rho_old");

      proteus::pyarray<double>& q_r = args.array<double>("q_r");
      const double alpha_L = args.scalar<double>("alpha_L");
      const double alpha_T = args.scalar<double>("alpha_T");
      const double Dm = args.scalar<double>("Dm");
      // const int dispersion_type_int = args.scalar<int>("dispersion_type");
      // const double theta_s = args.scalar<double>("theta_s");
      // const double theta_r = args.scalar<double>("theta_r");
      // const double power_law_exponent = args.scalar<double>("power_law_exponent");
      // const double velocity_exponent = args.scalar<double>("velocity_exponent");
      const double rho_f = args.scalar<double>("rho_f");
      const double rho_s = args.scalar<double>("rho_s");
      int forceStrongConditions = args.scalar<int>("forceStrongConditions");
      // DISPERSION DISPERSION_TYPE = static_cast<DISPERSION>(dispersion_type_int);
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
      proteus::pyarray<int>& elementBoundaryMaterialTypes = args.array<int>("elementBoundaryMaterialTypes");
      proteus::pyarray<int>& isExteriorBoundaryPhysical = args.array<int>("isExteriorBoundaryPhysical");
      proteus::pyarray<int>& elementBoundaryElementsArray = args.array<int>("elementBoundaryElementsArray");
      proteus::pyarray<int>& elementBoundaryLocalElementBoundariesArray = args.array<int>("elementBoundaryLocalElementBoundariesArray");
      proteus::pyarray<double>& ebqe_velocity_ext = args.array<double>("ebqe_velocity_ext");
      proteus::pyarray<int>& isDOFBoundary_u = args.array<int>("isDOFBoundary_u");
      proteus::pyarray<double>& ebqe_bc_u_ext = args.array<double>("ebqe_bc_u_ext");
      proteus::pyarray<int>& isFluxBoundary_u = args.array<int>("isFluxBoundary_u");
      proteus::pyarray<double>& ebqe_bc_flux_u_ext = args.array<double>("ebqe_bc_flux_u_ext");
      proteus::pyarray<double>& ebqe_bc_diffusiveFlux_u_ext = args.array<double>("ebqe_bc_diffusiveFlux_u_ext");
      proteus::pyarray<double>& ebqe_porosity = args.array<double>("ebqe_porosity");
      proteus::pyarray<double>& ebqe_rho = args.array<double>("ebqe_rho");
      
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
      STABILIZATION STABILIZATION_TYPE = static_cast<STABILIZATION>(args.scalar<int>("STABILIZATION_TYPE"));
      ENTROPY ENTROPY_TYPE = static_cast<ENTROPY>(args.scalar<int>("ENTROPY_TYPE"));    
      //STABILIZATION STABILIZATION_TYPE{args.scalar<int>("STABILIZATION_TYPE")};
      //ENTROPY ENTROPY_TYPE{args.scalar<int>("ENTROPY_TYPE")};
      proteus::pyarray<double>& uLow = args.array<double>("uLow");
      proteus::pyarray<double>& dLow = args.array<double>("dLow");
      proteus::pyarray<double>& dt_times_dH_minus_dL = args.array<double>("dt_times_dH_minus_dL");
      proteus::pyarray<double>& min_u_bc = args.array<double>("min_u_bc");
      proteus::pyarray<double>& max_u_bc = args.array<double>("max_u_bc");
      proteus::pyarray<double>& quantDOFs = args.array<double>("quantDOFs");
      // Stage 3 (kinetic dissolution).  Adds R_diss = k_d * S_n * S_w *
      // (c_sat - c) per DOF to the mass update, scaled by theta_w * rho_w
      // (so it has mass-rate units).  When the flow model is single-phase
      // (Richards), Sn_dof is zeros and R_diss vanishes.
      proteus::pyarray<double>& Sn_dof = args.array<double>("Sn_dof");
      const double k_d   = args.scalar<double>("k_d");
      const double c_sat = args.scalar<double>("c_sat");
      /////////////////////////////////////////////////////////////////////////
      proteus::pyarray<int>& a_rowptr = args.array<int>("a_rowptr");
      proteus::pyarray<int>& a_colind = args.array<int>("a_colind");
      //proteus::pyarray<double>& D = args.array<double>("D");
      //initializeDToZero(D);
      ///////////////////////////////////////////
      proteus::pyarray<int>& isDiffusiveFluxBoundary_u = args.array<int>("isDiffusiveFluxBoundary_u");
      proteus::pyarray<int>& isAdvectiveFluxBoundary_u = args.array<int>("isAdvectiveFluxBoundary_u");
      proteus::pyarray<double>& ebqe_bc_advectiveFlux_u_ext = args.array<double>("ebqe_bc_advectiveFlux_u_ext");
      proteus::pyarray<double>& ebqe_penalty_ext = args.array<double>("ebqe_penalty_ext");
      //////////////////////////////////////////////////////////////////////////
      double physicalDiffusion = args.scalar<double>("physicalDiffusion");
      const double eb_adjoint_sigma = args.scalar<double>("eb_adjoint_sigma");
      // The dissipation of an owned dof reads psi and the normalized entropy residual of its graph
      // neighbours: ghosts whose rows are truncated when across a rank cut. Phase 0 stops after computing
      // them (writing psi_dof and global_entropy_residual_dof); Python scatters them from the owners and
      // calls again with phase 1, which recomputes everything up to there (the element loops add to
      // globalResidual, which Python zeroes in between) and takes the scattered values.
      const int phase = args.scalar<int>("phase");
      proteus::pyarray<double>& psi_dof = args.array<double>("psi_dof");
      proteus::pyarray<double>& global_entropy_residual_dof = args.array<double>("global_entropy_residual_dof");
      // The projected nodal porosity of a ghost dof sums only the elements this rank has, which do not cover its
      // support, and the edge loops of an owned dof read it at the neighbours (m_dof[j], theta_dof_proj[j]).
      // Phase -1 stops after the projection and writes it to theta_dof; Python scatters it from the owners and
      // the later phases take those values (the owned ones are the numbers this rank computed).
      proteus::pyarray<double>& theta_dof = args.array<double>("theta_dof");
      
      double meanEntropy = 0., meanOmega = 0., maxEntropy = -1E10, minEntropy = 1E10;
      const double eps_rho = (rho_s - rho_f)/rho_f;
      const double zL_mass = uL + eps_rho*uL*uL;
      const double zR_mass = uR + eps_rho*uR*uR;
      maxVel.resize(nElements_global, 0.0);
      maxEntRes.resize(nElements_global, 0.0);
      double Ct_sge = 4.0;
      if (STABILIZATION_TYPE==STABILIZATION::EntropyViscosity or
          STABILIZATION_TYPE==STABILIZATION::SmoothnessIndicator or
          STABILIZATION_TYPE==STABILIZATION::Kuzmin or
          STABILIZATION_TYPE==STABILIZATION::ImplicitEV)
        {
          TransportMatrix.resize(NNZ,0.0);
          DiffusionMatrix.resize(NNZ,0.0);
          TransposeTransportMatrix.resize(NNZ,0.0);
          m_dof.resize(numDOFs,0.0);
          theta_dof_proj.resize(numDOFs,0.0);
          rho_dof_proj.resize(numDOFs,0.0);
          ML_mass_proj.resize(numDOFs,0.0);
          // ONLY the porosity gets a quadrature->DOF projection: theta lives at
          // quadrature points and has no nodal representation.  rho and the
          // conservative variable m are evaluated DIRECTLY at the DOF from
          // u_dof_old.  Projecting them would apply (M*u)_i/ML_i -- a smoothing
          // filter, because ML_i is exactly the row sum of the consistent mass
          // matrix -- so m_dof would NOT equal u_dof_old even at rho=theta=1 and
          // the scheme would not reduce to the constant-density one.  Evaluating
          // at the DOF also makes inversevaluateCoefficients(m_dof[i],...) return
          // u_dof_old[i] exactly, and makes this m^n consistent with the nodal
          // m^{n+1} = theta_i*rho(c_i)*c_i that the ImplicitEV branch builds
          // (otherwise ML_i*(m^{n+1}-m^n)/dt is nonzero at steady state).
          for (int eN=0; eN<nElements_global; eN++)
            for (int k=0; k<nQuadraturePoints_element; k++)
              {
                int eN_k = eN*nQuadraturePoints_element + k;
                double jac[nSpace*nSpace], jacDet, jacInv[nSpace*nSpace], x, y, z;
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
                const double dV = fabs(jacDet)*dV_ref.data()[k];
                const double theta_k = q_porosity_old.data()[eN_k];
                for (int i=0; i<nDOF_test_element; i++)
                  {
                    int eN_i = eN*nDOF_test_element+i;
                    const int gi = u_l2g.data()[eN_i];
                    const double w = u_test_ref.data()[k*nDOF_trial_element+i]*dV;
                    theta_dof_proj[gi] += theta_k*w;
                    ML_mass_proj[gi] += w;
                  }
              }
          for (int i=0; i<numDOFs; i++)
            {
              if (ML_mass_proj[i] > 1.0e-14)
                theta_dof_proj[i] /= ML_mass_proj[i];
              else
                theta_dof_proj[i] = 1.0;
              // nodal density and conservative variable at t^n
              const double un_i = u_dof_old.data()[i];
              rho_dof_proj[i] = rho_f*(1.0 + eps_rho*un_i);
              m_dof[i] = theta_dof_proj[i]*rho_dof_proj[i]*un_i;
            }
          if (phase == -1)
            {
              for (int i=0; i<numDOFs; i++)
                theta_dof.data()[i] = theta_dof_proj[i];
              return;
            }
          for (int i=0; i<numDOFs; i++)
            {
              theta_dof_proj[i] = theta_dof.data()[i];
              m_dof[i] = theta_dof_proj[i]*rho_dof_proj[i]*u_dof_old.data()[i];
            }
          // compute entropy and init global_entropy_residual and boundary_integral
          psi.resize(numDOFs,0.0);
          eta.resize(numDOFs,0.0);
          global_entropy_residual.resize(numDOFs,0.0);
          boundary_integral.resize(numDOFs,0.0);
          for (int i=0; i<numDOFs; i++)
            {
              // NODAL ENTROPY //
              if (STABILIZATION_TYPE==STABILIZATION::EntropyViscosity) //EV stab
                {
                  if (ENTROPY_TYPE == ENTROPY::POWER)
                    eta[i] = EPOWER(m_dof[i],uL,uR);
                  else
                    {
                      const double mass_scale_i = fmax(theta_dof_proj[i]*rho_f, 1.0e-14);
                      const double z_i = m_dof[i]/mass_scale_i;
                      eta[i] = ELOG(z_i,zL_mass,zR_mass);
                    }
                  global_entropy_residual[i]=0.;
                }
              boundary_integral[i]=0.;
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
      for(int eN=0;eN<nElements_global;eN++)
        {
          //declare local storage for element residual and initialize
          double
            elementResidual_u[nDOF_test_element],
            element_entropy_residual[nDOF_test_element];
          double  elementTransport[nDOF_test_element][nDOF_trial_element];
          double  elementDiffusion[nDOF_test_element][nDOF_trial_element];
          double  elementTransposeTransport[nDOF_test_element][nDOF_trial_element];
          for (int i=0;i<nDOF_test_element;i++)
            {
              elementResidual_u[i]=0.0;
            }//i
          if (STABILIZATION_TYPE==STABILIZATION::EntropyViscosity or
              STABILIZATION_TYPE==STABILIZATION::SmoothnessIndicator or
              STABILIZATION_TYPE==STABILIZATION::Kuzmin or
              STABILIZATION_TYPE==STABILIZATION::ImplicitEV)
            {
              for (int i=0;i<nDOF_test_element;i++)
                {
                  element_entropy_residual[i]=0.0;
                  for (int j=0;j<nDOF_trial_element;j++)
                    {
                      elementTransport[i][j]=0.0;
                      elementDiffusion[i][j]=0.0;
                      elementTransposeTransport[i][j]=0.0;
                    }
                }
            }
          //loop over quadrature points and compute integrands
          for  (int k=0;k<nQuadraturePoints_element;k++)
            {
              //compute indeces and declare local storage
              int eN_k = eN*nQuadraturePoints_element+k,
                eN_k_nSpace = eN_k*nSpace,
                eN_nDOF_trial_element = eN*nDOF_trial_element;
                //int index_D = eN_k * a_rowptr.data()[nSpace];
              double
                entVisc_minus_artComp,
                u=0.0,un=0.0,
                grad_u[nSpace],grad_u_old[nSpace],grad_uTilde[nSpace],
                rho_out=0.0,rho_out_old=0.0,
                m=0.0,dm=0.0,mn=0.0,dmn=0.0,
                H=0.0,Hn=0.0,HTilde=0.0,
                f[nSpace],fn[nSpace],df[nSpace],dfn[nSpace],
                ////////////////////////////////////////////
                //a[nSpace], da[nSpace], an[nSpace], dan[nSpace],
                a[nnz], da[nnz], an[nnz], dan[nnz],
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
                G[nSpace*nSpace],G_dd_G,tr_G,
                // for entropy residual
                aux_entropy_residual=0.0, DENTROPY_un, DENTROPY_uni;//norm_Rv;

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
                      u_grad_test_dV[j*nSpace+I] = u_grad_trial[j*nSpace+I]*dV;//cek warning won't work for Petrov-Galerkin
                    }
                }

              //
              //
              //calculate pde coefficients at quadrature points

              evaluateCoefficients(a_rowptr.data(),
				                           a_colind.data(),
                                   &velocity.data()[eN_k_nSpace],
                                   alpha_L,
                                   alpha_T,
                                   Dm,
                                   q_porosity.data()[eN*nQuadraturePoints_element+k],
                                   rho_f,
                                   rho_s,
                                   u,
                                   rho_out,
                                   m,
                                   dm,
                                   f,
                                   df,
                                   a,
                                   da);
              q_rho.data()[eN_k] = rho_out;

              evaluateCoefficients(a_rowptr.data(),
				                           a_colind.data(),
                                   &velocity_old.data()[eN_k_nSpace],
                                   alpha_L,
                                   alpha_T,
                                   Dm,
                                   q_porosity_old.data()[eN*nQuadraturePoints_element+k],
                                   rho_f,
                                   rho_s,
                                   un,
                                   rho_out_old,
                                   mn,
                                   dmn,
                                   fn,
                                   dfn, 
                                   an, 
                                   dan);
              //an= &q_a.data()[eN_k * sd_rowptr.data()[nSpace]];

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
                  fn[I] -= MOVING_DOMAIN*mn*mesh_velocity[I];
                  dfn[I] -= MOVING_DOMAIN*dmn*mesh_velocity[I];
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

              const double thetaW_k = std::max(q_porosity_old.data()[eN_k], 1.0e-8);
              double dfn_pore[nSpace];
              for (int I=0; I<nSpace; I++) dfn_pore[I] = dfn[I] / thetaW_k;

              if (STABILIZATION_TYPE==STABILIZATION::TaylorGalerkinEV)
                {
                  double normVel=0., norm_grad_un=0.;
                  for (int I=0;I<nSpace;I++)
                    {
                      Hn += dfn[I]*grad_u_old[I];
                      HTilde += dfn[I]*grad_uTilde[I];
                      fn[I] = dfn[I]*un-MOVING_DOMAIN*m*mesh_velocity[I];//cek check this for moving domain
                      H += dfn[I]*grad_u[I];
                      normVel += dfn[I]*df[I];
                      norm_grad_un += grad_u_old[I]*grad_u_old[I];
                    }
                  normVel = std::sqrt(normVel);
                  norm_grad_un = std::sqrt(norm_grad_un)+1E-10;

                  // calculate CFL
                  calculateCFL(elementDiameter.data()[eN]/degree_polynomial,dfn_pore,cfl.data()[eN_k]);


                  // compute max velocity at cell
                  maxVel[eN] = fmax(normVel,maxVel[eN]);

                  // Strong entropy residual
                  double entRes = (EPOWER(u,0,1)-EPOWER(un,0,1))/dt + 0.5*(DEPOWER(u,0,1)*H + DEPOWER(un,0,1)*Hn);
                  maxEntRes[eN] = fmax(maxEntRes[eN],fabs(entRes));

                  // Quantities for normalization factor //
                  meanEntropy += EPOWER(u,0,1)*dV;
                  meanOmega += dV;
                  maxEntropy = fmax(maxEntropy,EPOWER(u,0,1));
                  minEntropy = fmin(minEntropy,EPOWER(u,0,1));

                  // artificial compression
                  double hK=elementDiameter.data()[eN]/degree_polynomial;
                  entVisc_minus_artComp = fmax(1-cK*fmax(un*(1-un),0)/hK/norm_grad_un,0);
                }
              else if (STABILIZATION_TYPE==STABILIZATION::VMS)
                {
                  //
                  //calculate subgrid error (strong residual and adjoint)
                  //
                  //calculate strong residual
                  pdeResidual_u = ck.Mass_strong(m_t) + ck.Advection_strong(df,grad_u);
                  //calculate adjoint
                  for (int i=0;i<nDOF_test_element;i++)
                    {
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
                                                 pdeResidual_u,
                                                 grad_u,
                                                 numDiff0);
                  ck.calculateNumericalDiffusion(shockCapturingDiffusion,
                                                 sc_uref,
                                                 sc_alpha,
                                                 G,
                                                 G_dd_G,
                                                 pdeResidual_u,
                                                 grad_u,
                                                 numDiff1);
                  q_numDiff_u.data()[eN_k] = useMetrics*numDiff1+(1.0-useMetrics)*numDiff0;
                }
              else if (STABILIZATION_TYPE==STABILIZATION::EntropyViscosity)
              {
                aux_entropy_residual = m_t;
                for (int I=0;I<nSpace;I++)
                  aux_entropy_residual += dfn[I]*grad_u_old[I];
                if (ENTROPY_TYPE==ENTROPY::POWER)
                  DENTROPY_un = DEPOWER(mn,uL,uR);
                else
                  {
                    const double mass_scale_k = fmax(q_porosity_old.data()[eN_k]*rho_f, 1.0e-14);
                    const double z_n = mn/mass_scale_k;
                    DENTROPY_un = DELOG(z_n,zL_mass,zR_mass)/mass_scale_k;
                  }
                calculateCFL(elementDiameter.data()[eN]/degree_polynomial,dfn_pore,cfl.data()[eN_k]);
              }
              else
                calculateCFL(elementDiameter.data()[eN]/degree_polynomial,dfn_pore,cfl.data()[eN_k]);

              for(int i=0;i<nDOF_test_element;i++)
                {
                  int i_nSpace=i*nSpace;
                  if (STABILIZATION_TYPE==STABILIZATION::TaylorGalerkinEV)
                    {
                      if (stage == 1)
                        elementResidual_u[i] +=
                          ck.Mass_weak(dt*m_t,u_test_dV[i]) +  // time derivative
                          1./3*dt*(ck.Advection_weak(fn,&u_grad_test_dV[i_nSpace]) +
                                   ck.Diffusion_weak(a_rowptr.data(),a_colind.data(),a,grad_u,&u_grad_test_dV[i_nSpace])+ 
                                   ck.NumericalDiffusion(physicalDiffusion, grad_u_old, &u_grad_test_dV[i_nSpace])) +
                          1./9*dt*dt*ck.NumericalDiffusion(Hn,dfn,&u_grad_test_dV[i_nSpace]) +
                          1./3*dt*entVisc_minus_artComp*ck.NumericalDiffusion(q_numDiff_u_last.data()[eN_k]+physicalDiffusion,
                                                                              grad_u_old,
                                                                              &u_grad_test_dV[i_nSpace]);
                      // TODO: Add part about moving mesh
                      else //stage 2
                        elementResidual_u[i] +=
                          ck.Mass_weak(dt*m_t,u_test_dV[i]) +  // time derivative
                          dt*(ck.Advection_weak(fn,&u_grad_test_dV[i_nSpace]) + 
                              ck.Diffusion_weak(a_rowptr.data(),a_colind.data(),an,grad_u,&u_grad_test_dV[i_nSpace])+
                              ck.NumericalDiffusion(physicalDiffusion, grad_u_old, &u_grad_test_dV[i_nSpace])) +
                          0.5*dt*dt*ck.NumericalDiffusion(HTilde,dfn,&u_grad_test_dV[i_nSpace]) +
                          dt*entVisc_minus_artComp*ck.NumericalDiffusion(q_numDiff_u_last.data()[eN_k]+physicalDiffusion,
                                                                         grad_u_old,
                                                                         &u_grad_test_dV[i_nSpace]);
                    }
                  else if (STABILIZATION_TYPE==STABILIZATION::VMS)
                    {
                      elementResidual_u[i] +=
                        ck.Mass_weak(m_t,u_test_dV[i]) +
                        ck.Advection_weak(f,&u_grad_test_dV[i_nSpace]) +
                        ck.Diffusion_weak(a_rowptr.data(),a_colind.data(),a,grad_u,&u_grad_test_dV[i_nSpace]) +    
                        ck.SubgridError(subgridError_u,Lstar_u[i]) +
                        ck.NumericalDiffusion(q_numDiff_u_last.data()[eN_k] + physicalDiffusion,//todo add full sparse diffusion terms
                                              grad_u,
                                              &u_grad_test_dV[i_nSpace]);
                    }
                  else if(STABILIZATION_TYPE==STABILIZATION::EntropyViscosity or
                          STABILIZATION_TYPE==STABILIZATION::SmoothnessIndicator or
                          STABILIZATION_TYPE==STABILIZATION::Kuzmin or
                          STABILIZATION_TYPE==STABILIZATION::ImplicitEV)
                    {
                      int eN_i=eN*nDOF_test_element+i;
                      if (STABILIZATION_TYPE==STABILIZATION::EntropyViscosity) // EV stab
                        {
                          element_entropy_residual[i] += DENTROPY_un*aux_entropy_residual*u_test_dV[i];
                        }
                      elementResidual_u[i] += (u-un)*u_test_dV[i];
                      double* adv_df = (STABILIZATION_TYPE==STABILIZATION::ImplicitEV) ? df : dfn;
                      for(int j=0;j<nDOF_trial_element;j++)
                        {
                          int j_nSpace = j*nSpace;
                          int i_nSpace = i*nSpace;
                          elementTransport[i][j] +=
                            ck.AdvectionJacobian_weak(adv_df,
                                                      u_trial_ref.data()[k*nDOF_trial_element+j],
                                                      &u_grad_test_dV[i_nSpace])
                                                      +
                            ck.SimpleDiffusionJacobian_weak(a_rowptr.data(),
										                                        a_colind.data(),
                                                            a,
                                                            &u_grad_trial[j_nSpace],
                                                            &u_grad_test_dV[i_nSpace]);




                           elementDiffusion[i][j] += ck.NumericalDiffusionJacobian(physicalDiffusion,
                                                                                   &u_grad_trial[j_nSpace],
                                                                                   &u_grad_test_dV[i_nSpace]);
                          elementTransposeTransport[i][j] += ck.AdvectionJacobian_weak(adv_df,
                                                                                       u_trial_ref.data()[k*nDOF_trial_element+i],
                                                                                       &u_grad_test_dV[j_nSpace])+
                                                             ck.SimpleDiffusionJacobian_weak(a_rowptr.data(),
                                                                                              a_colind.data(),
                                                                                              a,
                                                                                              &u_grad_trial[j_nSpace],
                                                                                              &u_grad_test_dV[i_nSpace]);
                                                                                       
                                                            
                                                          }
                    }
                  else
                    {
                      elementResidual_u[i] +=
                        ck.Mass_weak(m_t,u_test_dV[i]) +
                        ck.Advection_weak(f,&u_grad_test_dV[i_nSpace])+
                        ck.Diffusion_weak(a_rowptr.data(),a_colind.data(),a,grad_u,&u_grad_test_dV[i_nSpace]);
                        //  +                       
                        // ck.NumericalDiffusion(physicalDiffusion,//todo add full sparse diffusion terms
                        //                       grad_u,
                        //                       &u_grad_test_dV[i_nSpace]);
                    }
                }//i
              //
              //save solution for other models
              //
              q_u.data()[eN_k] = u;
              q_m.data()[eN_k] = m;
              //logInteriorState("storeQuadrature", eN, k, x, y, z, u, m, f, a);
        
              }//k
          //
          //load element into global residual and save element residual
          //
          for(int i=0;i<nDOF_test_element;i++)
            {
              int eN_i=eN*nDOF_test_element+i;
              int gi = offset_u+stride_u*u_l2g.data()[eN_i]; //global i-th index
              globalResidual.data()[gi] += elementResidual_u[i];
              if (STABILIZATION_TYPE==STABILIZATION::EntropyViscosity or
                  STABILIZATION_TYPE==STABILIZATION::SmoothnessIndicator or
                  STABILIZATION_TYPE==STABILIZATION::Kuzmin or
                  STABILIZATION_TYPE==STABILIZATION::ImplicitEV)
                {

                  // distribute entropy_residual
                  if (STABILIZATION_TYPE==STABILIZATION::EntropyViscosity) // EV Stab
                    global_entropy_residual[gi] += element_entropy_residual[i];
                  // distribute transport matrices
                  for (int j=0;j<nDOF_trial_element;j++)
                    {
                      int eN_i_j = eN_i*nDOF_trial_element+j;
                      TransportMatrix[csrRowIndeces_CellLoops.data()[eN_i] +
                                      csrColumnOffsets_CellLoops.data()[eN_i_j]] += elementTransport[i][j];
                      DiffusionMatrix[csrRowIndeces_CellLoops.data()[eN_i] +
                                      csrColumnOffsets_CellLoops.data()[eN_i_j]] += elementDiffusion[i][j];
                      TransposeTransportMatrix[csrRowIndeces_CellLoops.data()[eN_i] +
                                               csrColumnOffsets_CellLoops.data()[eN_i_j]]
                        += elementTransposeTransport[i][j];
                    }//j
                }//edge-based
            }//i
        }//eN
      //
      //loop over exterior element boundaries to calculate surface integrals and load into element and global residuals
      //
      //ebNE is the Exterior element boundary INdex
      //ebN is the element boundary INdex
      //eN is the element index
      for (int ebNE = 0; ebNE < nExteriorElementBoundaries_global; ebNE++)
        {
          double min_u_bc_local = 1E10, max_u_bc_local = -1E10;
          int ebN = exteriorElementBoundariesArray.data()[ebNE],
            eN  = elementBoundaryElementsArray.data()[ebN*2+0],
            ebN_local = elementBoundaryLocalElementBoundariesArray.data()[ebN*2+0],
            eN_nDOF_trial_element = eN*nDOF_trial_element;
          const int eN_out = elementBoundaryElementsArray.data()[ebN*2+1];
          const int ebFlag = elementBoundaryMaterialTypes.data()[ebN];
          // Only integrate true physical exterior boundaries; skip partition/non-physical faces.
          if (ebFlag <= 0 || isExteriorBoundaryPhysical.data()[ebNE] == 0 || eN_out >= 0)
            {
              continue;
            }
          double elementResidual_u[nDOF_test_element],
            fluxTransport[nDOF_test_element][nDOF_trial_element];
          for (int i=0;i<nDOF_test_element;i++)
            {
              elementResidual_u[i]=0.0;
              for (int j=0;j<nDOF_trial_element;j++)
                fluxTransport[i][j] = 0.0;
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
                f_ext[nSpace],
                df_ext[nSpace],
                /////////////////////

                a_ext[nnz],
		            da_ext[nnz],

                bc_a_ext[nnz],
		            bc_da_ext[nnz],

                /////////////////////////////
                flux_ext=0.0,
                dflux_u_u_ext=0.0,
                bc_u_ext=0.0,
                bc_m_ext=0.0,
                bc_dm_ext=0.0,

                flux_adv_ext=0.0,
                flux_diff_ext=0.0,
                difffluxjacobian_ext=0.0,
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
                u_grad_test_dS[nDOF_trial_element*nSpace],
                normal[nSpace],x_ext,y_ext,z_ext,xt_ext,yt_ext,zt_ext,integralScaling,
                
                G[nSpace*nSpace],G_dd_G,tr_G;
                
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
              ck.gradTrialFromRef(&u_grad_trial_trace_ref.data()[ebN_local_kb_nSpace*nDOF_trial_element],
                                  jacInv_ext,
                                  u_grad_trial_trace);
              //solution and gradients
              if (STABILIZATION_TYPE==STABILIZATION::TaylorGalerkinEV) //explicit
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
                  for (int I=0;I<nSpace;I++)
		                u_grad_test_dS[j*nSpace+I] = u_grad_trial_trace[j*nSpace+I]*dS;//cek hack, using trial
	
                }
              //
              //load the boundary values
              //
              bc_u_ext = isDOFBoundary_u.data()[ebNE_kb]*ebqe_bc_u_ext.data()[ebNE_kb]+
                          (1-isDOFBoundary_u.data()[ebNE_kb])*u_ext;
             

                      //
              //
              //calculate the pde coefficients using the solution and the boundary values for the solution
              //
              double rho_out_ext=0.0,rho_out_bc=0.0;
              evaluateCoefficients(a_rowptr.data(),
				                           a_colind.data(),
                                   &ebqe_velocity_ext.data()[ebNE_kb_nSpace],
                                   alpha_L,
                                   alpha_T,
                                   Dm,
                                   ebqe_porosity.data()[ebNE_kb],
                                   rho_f,
                                   rho_s,
                                   u_ext,
                                   rho_out_ext,
                                   m_ext,
                                   dm_ext,
                                   f_ext,
                                   df_ext,
                                   a_ext,
                                   da_ext);
              ebqe_rho.data()[ebNE_kb] = rho_out_ext;
              
              evaluateCoefficients(a_rowptr.data(),
				                           a_colind.data(),
                                   &ebqe_velocity_ext.data()[ebNE_kb_nSpace],
                                   alpha_L,
                                   alpha_T,
                                   Dm,
                                   ebqe_porosity.data()[ebNE_kb],
                                   rho_f,
                                   rho_s,
                                   bc_u_ext,
                                   rho_out_bc,
                                   bc_m_ext,
                                   bc_dm_ext,
                                   bc_f_ext,
                                   bc_df_ext,
                                   bc_a_ext,
                                   bc_da_ext);         
              //moving mesh
              //
              double mesh_velocity[3];
              mesh_velocity[0] = xt_ext;
              mesh_velocity[1] = yt_ext;
              mesh_velocity[2] = zt_ext;

              for (int I=0;I<nSpace;I++)
                {
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
                                             forceStrongConditions,
                                             normal,
                                             ebqe_bc_flux_u_ext.data()[ebNE_kb],
                                             f_ext,
                                             bc_f_ext,
                                             df_ext,
                                             flux_adv_ext);
              exteriorNumericalDiffusiveFlux(a_rowptr.data(),
                                             a_colind.data(),
                                             isDOFBoundary_u.data()[ebNE_kb],
                                             isDiffusiveFluxBoundary_u.data()[ebNE_kb],
                                             normal,
                                             a_ext,
                                             bc_u_ext,
                                             ebqe_bc_diffusiveFlux_u_ext.data()[ebNE_kb],
                                             a_ext,
                                             grad_u_ext,
                                             u_ext,
                                             ebqe_penalty_ext.data()[ebNE_kb],
                                             flux_diff_ext);
              flux_ext = flux_adv_ext + flux_diff_ext;
              double boundary_flow = 0.0;
              for (int I=0; I<nSpace; I++)
                boundary_flow += normal[I]*df_ext[I];
              
              ebqe_flux.data()[ebNE_kb] = flux_ext;
              if (isDOFBoundary_u.data()[ebNE_kb] == 1)
                ebqe_u.data()[ebNE_kb] = bc_u_ext;
              else if (boundary_flow >= 0.0)
                ebqe_u.data()[ebNE_kb] = u_ext;
              else
                ebqe_u.data()[ebNE_kb] = bc_u_ext;
              if (STABILIZATION_TYPE==STABILIZATION::TaylorGalerkinEV)
                {
                  if (stage == 1)
                    flux_ext *= 1./3*dt;
                  else
                    flux_ext *= dt;
                }

              //
              //update residuals
              //
              for (int i=0;i<nDOF_test_element;i++)
                {
                  if (STABILIZATION_TYPE == STABILIZATION::Galerkin or STABILIZATION_TYPE == STABILIZATION::VMS or
                      STABILIZATION_TYPE == STABILIZATION::TaylorGalerkinEV or STABILIZATION_TYPE == STABILIZATION::ImplicitEV)
                      {
                    elementResidual_u[i] += ck.ExteriorElementBoundaryFlux(flux_ext,u_test_dS[i])+
                                            ck.ExteriorElementBoundaryDiffusionAdjoint(isDOFBoundary_u.data()[ebNE_kb],
                                                                                            isDiffusiveFluxBoundary_u.data()[ebNE_kb],
                                                                                            eb_adjoint_sigma,
                                                                                            u_ext,
                                                                                            bc_u_ext,
                                                                                            normal,
                                                                                            a_rowptr.data(),
                                                                                            a_colind.data(),
                                                                                            a_ext,
                                                                                            &u_grad_test_dS[i*nSpace]);
                      }
                  else if (STABILIZATION_TYPE == STABILIZATION::EntropyViscosity or 
                      STABILIZATION_TYPE == STABILIZATION::SmoothnessIndicator or 
                      STABILIZATION_TYPE == STABILIZATION::Kuzmin)
                    {
                      const double boundaryAdvectiveContribution =
                        ck.ExteriorElementBoundaryFlux(flux_ext,u_test_dS[i]);
                      const double boundaryDiffusiveContribution =
                        ck.ExteriorElementBoundaryDiffusionAdjoint(isDOFBoundary_u.data()[ebNE_kb],
                                                                   isDiffusiveFluxBoundary_u.data()[ebNE_kb],
                                                                   eb_adjoint_sigma,
                                                                   u_ext,
                                                                   bc_u_ext,
                                                                   normal,
                                                                   a_rowptr.data(),
                                                                   a_colind.data(),
                                                                   a_ext,
                                                                   &u_grad_test_dS[i*nSpace]);
                      const double boundaryResidualContribution =
                        boundaryAdvectiveContribution + boundaryDiffusiveContribution;
                      exteriorNumericalAdvectiveFluxDerivative(isDOFBoundary_u.data()[ebNE_kb],
                                                           isFluxBoundary_u.data()[ebNE_kb],
                                                           forceStrongConditions,
                                                           normal,
                                                           df_ext,
                                                           dflux_u_u_ext);  
                      
                      if (dflux_u_u_ext> 0.0)
                      {
                        double boundaryTransportContribution = 0.0;
                        for (int j=0;j<nDOF_trial_element;j++)
                        {
                          int ebN_local_kb_j=ebN_local_kb*nDOF_trial_element+j;
                          double advJacobian_ext = 0.0, diffJacobian_ext = 0.0;
                          exteriorNumericalAdvectiveFluxDerivative(isDOFBoundary_u.data()[ebNE_kb],
                                                                   isFluxBoundary_u.data()[ebNE_kb],
                                                                   forceStrongConditions,
                                                                   normal,
                                                                   df_ext,
                                                                   advJacobian_ext);
                          exteriorNumericalDiffusiveFluxDerivative(isDOFBoundary_u.data()[ebNE_kb],
                                                                   isDiffusiveFluxBoundary_u.data()[ebNE_kb],
                                                                   a_rowptr.data(),
                                                                   a_colind.data(),
                                                                   normal,
                                                                   a_ext,
                                                                   da_ext,
                                                                   grad_u_ext,
                                                                   &u_grad_trial_trace[j*nSpace],
                                                                   u_trial_trace_ref.data()[ebN_local_kb_j],
                                                                   ebqe_penalty_ext.data()[ebNE_kb],
                                                                   diffJacobian_ext);
                          difffluxjacobian_ext = advJacobian_ext*u_trial_trace_ref.data()[ebN_local_kb_j]
                                                 + diffJacobian_ext;
                          const double localFluxTransportContribution =
                            difffluxjacobian_ext*u_test_dS[i];
                          fluxTransport[i][j] += localFluxTransportContribution;
                          boundaryTransportContribution +=
                            localFluxTransportContribution*u_dof_old.data()[u_l2g.data()[eN_nDOF_trial_element+j]];
                        }
                        elementResidual_u[i] += boundaryResidualContribution - boundaryTransportContribution;
                      }
                      else
                      {
                        elementResidual_u[i] += boundaryResidualContribution;
                        // Upwind Nitsche penalty for advection-dominated Dirichlet
                        // inflow.  At inflow (boundary_flow = v.n < 0) the upwind
                        // advective flux uses bc_u_ext and has zero derivative wrt
                        // the interior u_ext -- so the existing IIPG penalty is the
                        // only thing pulling u_ext toward bc_u_ext, and it scales as
                        // max_a*penalty/h which becomes vanishingly small when D_m
                        // is small.  Add a Nitsche term that scales with |v.n| so
                        // BC enforcement is independent of the diffusion coefficient.
                        // Sign: at inflow with u_ext > bc_u_ext (overshoot), this
                        // contributes positively to elementResidual_u[i] (= positive
                        // boundary flux out of node i), which reduces mLow at the BC
                        // DOF and pulls c back to bc_u_ext.  Mass-conservative: the
                        // term is integrated weakly with u_test_dS like the rest of
                        // the boundary residual; sum over all faces telescopes the
                        // weak Dirichlet to a consistent transport balance.
                        if (isDOFBoundary_u.data()[ebNE_kb] == 1 && boundary_flow < 0.0)
                        {
                          const double upwind_penalty_rate = -boundary_flow; // |v.n|
                          elementResidual_u[i] += upwind_penalty_rate
                                                * (u_ext - bc_u_ext)
                                                * u_test_dS[i];
                        }
                      }
                    }
                }//i
              // local min/max at boundary.
              // At Dirichlet faces use the BC value (ebqe_bc_u_ext), not the
              // current solution trace (ebqe_u): the trace can drift off c_sat
              // under weak Nitsche enforcement, and feeding that drifted value
              // into min/max_u_bc pollutes the FCT bound at every interior
              // neighbor.  Using the BC value keeps the bound tight at c_sat
              // (combined with bc_mask in the FCT step, this gives the bounded
              // + mass-conservative recipe that mphase_co2 uses).
              const double u_for_bound =
                  isDOFBoundary_u.data()[ebNE_kb]
                      ? ebqe_bc_u_ext.data()[ebNE_kb]
                      : ebqe_u.data()[ebNE_kb];
              min_u_bc_local = fmin(u_for_bound, min_u_bc_local);
              max_u_bc_local = fmax(u_for_bound, max_u_bc_local);
            }//kb
          //
          //update the element and global residual storage
          //
          for (int i=0;i<nDOF_test_element;i++)
            {
              int eN_i = eN*nDOF_test_element+i;
              int gi = offset_u+stride_u*u_l2g.data()[eN_i]; //global i-th index
              if (STABILIZATION_TYPE==STABILIZATION::EntropyViscosity or STABILIZATION_TYPE==STABILIZATION::SmoothnessIndicator or STABILIZATION_TYPE==STABILIZATION::Kuzmin)
                {
                  globalResidual.data()[gi] += dt*elementResidual_u[i];
                  boundary_integral[gi] += elementResidual_u[i];
                  min_u_bc[gi] = fmin(min_u_bc_local,min_u_bc[gi]);
                  max_u_bc[gi] = fmax(max_u_bc_local,max_u_bc[gi]);
                  for (int j=0;j<nDOF_trial_element;j++)
                    {
                      int ebN_i_j = ebN*4*nDOF_test_X_trial_element + i*nDOF_trial_element + j;
                      TransportMatrix[csrRowIndeces_CellLoops.data()[eN_i] + csrColumnOffsets_eb_CellLoops.data()[ebN_i_j]]
                        += fluxTransport[i][j];
                      TransposeTransportMatrix[csrRowIndeces_CellLoops.data()[eN_i] + csrColumnOffsets_eb_CellLoops.data()[ebN_i_j]]
                        += fluxTransport[j][i];
                    }//j
                }
              else if (STABILIZATION_TYPE==STABILIZATION::ImplicitEV)
                {
                  // Implicit boundary: stash the consistent boundary residual in
                  // boundary_integral (the edge loop adds + boundary_integral[i]
                  // and OVERWRITES globalResidual[i], so we must NOT add here).
                  // The boundary flux Jacobian is assembled in calculateJacobian.
                  boundary_integral[gi] += elementResidual_u[i];
                  // FCT bounds at the boundary (used by the explicit FCT
                  // post-step when FCT=True; harmless otherwise).
                  min_u_bc[gi] = fmin(min_u_bc_local,min_u_bc[gi]);
                  max_u_bc[gi] = fmax(max_u_bc_local,max_u_bc[gi]);
                }
              else
                {
                  globalResidual.data()[offset_u+stride_u*r_l2g.data()[eN_i]] += elementResidual_u[i];
                }
            }//i
        }//ebNE
      if (STABILIZATION_TYPE==STABILIZATION::TaylorGalerkinEV)
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
      //edge based stabilization
      else if (STABILIZATION_TYPE==STABILIZATION::EntropyViscosity or 
               STABILIZATION_TYPE==STABILIZATION::SmoothnessIndicator or 
               STABILIZATION_TYPE==STABILIZATION::Kuzmin)
        {
          /////////////////////////////////////////////////////////////////
          // COMPUTE SMOOTHNESS INDICATOR and NORMALIZE ENTROPY RESIDUAL //
          /////////////////////////////////////////////////////////////////
          // NOTE: see NCLS.h for a different but equivalent implementation of this.
          //cek todo: can these loops over numDOFs be collapsed?
          int ij = 0;
          for (int i=0; i<numDOFs; i++)
            {
              double etaMaxi, etaMini;
              if (STABILIZATION_TYPE==STABILIZATION::EntropyViscosity) //EV
                {
                  // For eta min and max
                  etaMaxi = fabs(eta[i]);
                  etaMini = fabs(eta[i]);
                }
              // for smoothness indicator //
              double alpha_numerator = 0., alpha_denominator = 0.;
              for (int offset=csrRowIndeces_DofLoops.data()[i]; offset<csrRowIndeces_DofLoops.data()[i+1]; offset++)
                { // First loop in j (sparsity pattern)
                  int j = csrColumnOffsets_DofLoops.data()[offset];
                  if (STABILIZATION_TYPE==STABILIZATION::EntropyViscosity) //EV Stabilization
                    {
                      // COMPUTE ETA MIN AND ETA MAX //
                      etaMaxi = fmax(etaMaxi,fabs(eta[j]));
                      etaMini = fmin(etaMini,fabs(eta[j]));
                    }
                  // Sense smoothness on the projected nodal conservative variable
                  // m = theta * rho(u) * u so the edge indicator is aligned with
                  // the variable-density storage used by the PDE residual.
                  const double mi = m_dof[i];
                  const double mj = m_dof[j];
                  alpha_numerator += mi - mj;
                  alpha_denominator += fabs(mi - mj);
                  //update ij
                  ij+=1;
                }
              if (STABILIZATION_TYPE==STABILIZATION::EntropyViscosity) //EV Stab
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
          if (phase == 0)
            {
              for (int i=0; i<numDOFs; i++)
                {
                  psi_dof.data()[i] = psi[i];
                  global_entropy_residual_dof.data()[i] = global_entropy_residual[i];
                }
              return;
            }
          for (int i=0; i<numDOFs; i++)
            {
              psi[i] = psi_dof.data()[i];
              if (STABILIZATION_TYPE==STABILIZATION::EntropyViscosity)
                global_entropy_residual[i] = global_entropy_residual_dof.data()[i];
            }
          /////////////////////////////////////////////
          // ** LOOP IN DOFs FOR EDGE BASED TERMS ** //
          /////////////////////////////////////////////
          ij=0;
          for (int i=0; i<numDOFs; i++)
            {
              const double mi_mass = m_dof[i];
              const double theta_i = fmax(theta_dof_proj[i], 1.0e-14);
              const double ui_mass = inversevaluateCoefficients(mi_mass, theta_i, rho_f, rho_s);
              const double rho_i = fmax(rho_dof_proj[i], rho_f);
              const double drho_du = rho_s - rho_f;
              const double dmdu_i = theta_i * (rho_i + ui_mass*drho_du);
              double ith_dissipative_term_mass = 0;
              double ith_low_order_dissipative_term_mass = 0;
              double ith_flux_term_mass = 0;
              // Row sum of the high-order graph viscosity, dLii = -sum_{j!=i} dLij.
              // Feeds edge_based_cfl below.
              double dLii = 0.;

              // loop over the sparsity pattern of the i-th DOF
              for (int offset=csrRowIndeces_DofLoops.data()[i]; offset<csrRowIndeces_DofLoops.data()[i+1]; offset++)
                {
                  int j = csrColumnOffsets_DofLoops.data()[offset];
                  const double mj_mass = m_dof[j];
                  const double theta_j = fmax(theta_dof_proj[j], 1.0e-14);
                  const double uj_mass = inversevaluateCoefficients(mj_mass, theta_j, rho_f, rho_s);
                  double dLowij, dLij, dEVij, dHij;

                  ith_flux_term_mass += (TransportMatrix[ij] + DiffusionMatrix[ij])*uj_mass;

                  if (i != j)
                    {
                      double solij = 0.5*(ui_mass+uj_mass);
                      double Compij = cK*fmax(solij*(1.0-solij),0.0)/(fabs(ui_mass-uj_mass)+1E-14);
                      dLowij = fmax(fabs(TransportMatrix[ij]),fabs(TransposeTransportMatrix[ij]));

                      dLij = dLowij*fmax(psi[i],psi[j]); // Approach by JLG & BP
                      
                      if (STABILIZATION_TYPE==STABILIZATION::EntropyViscosity) //EV Stab
                        {
                          // high-order (entropy viscosity) dissipative operator
                          dEVij = fmax(fabs(global_entropy_residual[i]),fabs(global_entropy_residual[j]));
                          dHij = fmin(dLowij,dEVij) * fmax(1.0-Compij,0.0); // artificial compression
                        }
                      else // smoothness based indicator
                        {
                          dHij = dLij * fmax(1.0-Compij,0.0); // artificial compression
                        }
                      // Dissipative terms.  The DIFFERENCES are taken in c, not in
                      // m: dLow comes from the transport matrix, which already
                      // carries rho (df/du = (rho + u*drho/du)*v), so dLow*(c_j-c_i)
                      // is a mass flux, dimensionally consistent with
                      // ith_flux_term_mass.  Differencing m instead multiplies the
                      // dissipation by a spurious extra theta*rho (~1e3 in SI).
                      // Note this pair IS already the monotone upwind operator:
                      //   sum_j (T+D)_ij c_j - sum_{j!=i} dLow_ij (c_j - c_i)
                      //     = -sum_{j!=i} (dLow_ij - T_ij - D_ij)(c_j - c_i)
                      // with dLow_ij - T_ij - D_ij >= 0 because dLow_ij >= |T_ij|,
                      // i.e. an M-matrix.  It must NOT be replaced by
                      // -sum max(0,-T_ij)(c_j - c_i): for the skew-symmetric
                      // advection matrix that is exactly HALF the upwind flux.
                      ith_dissipative_term_mass += dHij*(uj_mass-ui_mass);
                      ith_low_order_dissipative_term_mass += dLowij*(uj_mass-ui_mass);
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
              // boundary_integral was assembled from m-space fluxes (rho*v*u and
              // a=theta*rho*Disp); no dmdu_i lifting is needed.
              const double boundary_integral_mass = boundary_integral[i];
              // compute edge_based_cfl
              // 2|dLii|/mi is the constant-density edge CFL.  The low-order step
              // advances m = theta*rho(c)*c, so recovering c divides the mass
              // change by the storage Jacobian dm/dc = dmdu_i; without that
              // factor the stable dt is over-predicted in low-water-content (high
              // gas saturation) zones where dmdu_i -> 0, and uLow overshoots out
              // of [0, c_sat].  dmdu_i = 1 at rho=theta=1, so this reduces to the
              // original 2|dLii|/mi.
              edge_based_cfl.data()[i] = 2.*fabs(dLii)/(mi * fmax(dmdu_i, 1.0e-14));
              
              // Stage 3 kinetic dissolution source at node i (mass-rate form):
              //   R_diss_i = theta_w_i * rho_w(u_i) * k_d * S_n_i * (c_sat - u_i)
              // Added directly to mLow_i / mHigh_i since the lumped-mass time
              // discretization gives dm/dt = R_diss with no further scaling.
              // Sign of (c_sat - u_i) ensures R_diss > 0 when undersaturated
              // (dissolution) and < 0 when supersaturated (exsolution).
              // The S_w (interfacial-area) factor a_gw ~ S_n*S_w was dropped:
              // it makes R_diss vanish at S_n = 1, so a gas pool stops
              // dissolving once it consolidates.  theta_w is kept so the CO2
              // still goes into the brine that exists (theta_w > 0 down to
              // residual S_wr).  MUST stay consistent with mphase_co2.h.
              const double S_n_i  = Sn_dof.data()[i];
              const double rho_w_i = rho_f * (1.0 + ((rho_s - rho_f)/rho_f) * ui_mass);
              const double R_diss_i = theta_i * rho_w_i * k_d * S_n_i
                                    * (c_sat - ui_mass);

              const double mLow_i = mi_mass - dt/mi*(ith_flux_term_mass
                                                     + boundary_integral_mass
                                                     - ith_low_order_dissipative_term_mass)
                                            + dt * R_diss_i;
              uLow[i] = inversevaluateCoefficients(mLow_i, theta_i, rho_f, rho_s);

              // update residual
              if (LUMPED_MASS_MATRIX==1)
                {
                  const double mHigh_i = mi_mass - dt/mi*(ith_flux_term_mass
                                                          + boundary_integral_mass
                                                          - ith_dissipative_term_mass)
                                                 + dt * R_diss_i;
                  globalResidual.data()[i] = inversevaluateCoefficients(mHigh_i, theta_i, rho_f, rho_s);
                }
              else
                globalResidual.data()[i] += dt*(ith_flux_term_mass - ith_dissipative_term_mass - R_diss_i);//cek todo: shouldn't this have boundaryIntegral?
            }//i
        }//edge-based
      else if (STABILIZATION_TYPE==STABILIZATION::ImplicitEV)
        {
          //////////////////////////////////////////////////////////////////
          // IMPLICIT edge-based scheme (backward Euler, low-order graph    //
          // dissipation, no FCT).  Mirrors Richards STABILIZATION_TYPE==2  //
          // but evaluates the advection/diffusion at the CURRENT Newton    //
          // iterate (u_dof) so they contribute to the Jacobian (assembled  //
          // separately in calculateJacobian) and the scheme is NOT         //
          // CFL-limited.  Per-DOF residual:                                //
          //                                                                //
          //   R_i = ML_i*(m_i^{n+1} - m_i^n)/dt                            //
          //         + sum_j (T_ij + D_ij)*c_j            (consistent flux)  //
          //         - sum_{j!=i} dLow_ij*(m_j^{n+1}-m_i^{n+1}) (graph diss) //
          //         + boundary_integral_i               (0 for closed BCs) //
          //         - ML_i*R_diss_i                      (dissolution)      //
          //                                                                //
          // m^{n+1}_i = theta_i*rho(c_i)*c_i  (nodal, porosity lagged to   //
          // theta_dof_proj); m^n_i = m_dof[i] (old projected mass).        //
          // The advection coefficient in T uses the CURRENT-time velocity  //
          // (q_v, refreshed from mphase_co2 before TADR solves) and the    //
          // current iterate u -- fully implicit backward Euler.            //
          //                                                                //
          // FLUX IS UPWINDED: instead of the central sum_j(T_ij+D_ij)c_j   //
          // plus symmetric dissipation, the advective/diffusive flux is the//
          // first-order upwind low-order operator (the same one TADR's     //
          // explicit branch builds for uLow), evaluated at the current c:  //
          //                                                                //
          //   F_i = sum_{j!=i} -a_ij*(c_j - c_i),                          //
          //   a_ij = max(0,-T_ij)*(rho_up/rho_f) + max(0,-D_ij) >= 0,       //
          //   rho_up = rho(c_i) if j->i is inflow (-T_ij*(c_j-c_i)<=0)      //
          //            else rho(c_j).                                      //
          //                                                                //
          // a_ij>=0 makes the spatial operator an M-matrix (monotone, DMP);//
          // diffusion (D symmetric, row-sum 0) is reproduced exactly since //
          // -max(0,-D_ij)(c_j-c_i)=D_ij(c_j-c_i).  Conservative by global  //
          // telescoping of the edge fluxes.  a_ij (frozen) is stashed in   //
          // dLow[ij] for calculateJacobian.  m^{n+1}_i=theta_i*rho(c_i)*c_i //
          // (nodal, porosity lagged); m^n_i=m_dof[i].  boundary_integral   //
          // stays 0 (closed system); implicit boundary flux is a TODO.     //
          //////////////////////////////////////////////////////////////////
          const double drho_du = rho_s - rho_f;
          // nodal current conservative variable m^{n+1}_i = theta_i*rho(c_i)*c_i
          std::valarray<double> m_new(numDOFs);
          for (int i=0; i<numDOFs; i++)
            {
              const double ci = u_dof.data()[i];
              const double theta_i = fmax(theta_dof_proj[i], 1.0e-14);
              const double rho_ci = rho_f*(1.0 + (drho_du/rho_f)*ci);
              m_new[i] = theta_i*rho_ci*ci;
            }
          int ij=0;
          for (int i=0; i<numDOFs; i++)
            {
              const double ci = u_dof.data()[i];
              const double theta_i = fmax(theta_dof_proj[i], 1.0e-14);
              const double rho_ci = rho_f*(1.0 + (drho_du/rho_f)*ci);
              const double mi_new = m_new[i];
              const double mn_i = m_dof[i];
              const double MLi = ML.data()[i];

              double ith_upwind_flux_term_mass = 0.0;
              for (int offset=csrRowIndeces_DofLoops.data()[i]; offset<csrRowIndeces_DofLoops.data()[i+1]; offset++)
                {
                  int j = csrColumnOffsets_DofLoops.data()[offset];
                  if (i != j)
                    {
                      const double cj      = u_dof.data()[j];
                      const double rho_cj  = rho_f*(1.0 + (drho_du/rho_f)*cj);
                      const double T_ij     = TransportMatrix[ij];
                      const double D_ij     = DiffusionMatrix[ij];
                      const double delta_c  = cj - ci;
                      const double T_neg    = fmax(0.0, -T_ij);
                      const double D_neg    = fmax(0.0, -D_ij);
                      const double rho_up   = (-T_ij*delta_c <= 0.0) ? rho_ci : rho_cj;
                      // frozen edge coefficient a_ij >= 0 (advection upwind + diffusion)
                      const double a_ij     = T_neg*(rho_up/rho_f) + D_neg;
                      ith_upwind_flux_term_mass += -a_ij*delta_c;
                      // dLow stores the SYMMETRIC graph viscosity used by the
                      // explicit FCT post-step (Kuzmin antidiffusive flux, which
                      // must be antisymmetric per edge -> needs symmetric dLow,
                      // NOT the directional a_ij).  calculateJacobian recomputes
                      // a_ij directly from T,D, so it no longer reads dLow.
                      dLow.data()[ij] = fmax(fabs(T_ij), fabs(TransposeTransportMatrix[ij]));
                      // High-order (smoothness-compressed) graph dissipation for the
                      // FCT post-step, mirroring the explicit EntropyViscosity branch
                      // (dHij = dLowij*(1-Compij)).  Store dt*(dH - dLow) <= 0 so the
                      // FCTStep removes ONLY the EXCESS low-order dissipation
                      // (dLow - dH) and KEEPS dH.  Removing the full dLow (as the
                      // explicit Kuzmin flux does) makes the antidiffusion target the
                      // dissipation-free Galerkin solution, which over-sharpens the
                      // Dirichlet source front back to the initial condition -> the
                      // observed FCT=True freeze.  globalResidual is NOT touched here,
                      // so calculateJacobian needs no change.
                      {
                        // Standard Kuzmin implicit FEM-FCT high-order target: ZERO
                        // artificial dissipation (d^H = 0).  The full low-order graph
                        // viscosity dLow is antidiffused; the Zalesak limiter
                        // (min/max_u_bc bounds) provides boundedness.  The earlier
                        // dH = dLow*(1-Comp) "smoothness compression" throttled the
                        // antidiffusion so the consistent-mass term dominated and the
                        // front stayed diffuse (verified in the 1D FCT numpy replica);
                        // removing it recovers the textbook antidiffusive flux
                        //   F_ij = M~_ij[(m_j^H-m_j^n)-(m_i^H-m_i^n)]
                        //          + dt (d^H-d^L)_ij (m_j^n-m_i^n),  d^H=0.
                        const double dHij = 0.0;
                        dt_times_dH_minus_dL.data()[ij] = dt*(dHij - dLow.data()[ij]);
                      }
                    }
                  else
                    {
                      dLow.data()[ij] = 0.0;
                      dt_times_dH_minus_dL.data()[ij] = 0.0;
                    }
                  ij += 1;
                }
              // Stage-3 kinetic dissolution source (mass-rate form); MUST match
              // mphase_co2.h and the explicit edge-based branch above.
              const double S_n_i   = Sn_dof.data()[i];
              const double rho_w_i = rho_ci;   // rho_w(c_i) = rho_f*(1 + eps*c_i)
              const double R_diss_i = theta_i * rho_w_i * k_d * S_n_i * (c_sat - ci);

              globalResidual.data()[i] = MLi*(mi_new - mn_i)/dt
                                       + ith_upwind_flux_term_mass
                                       + boundary_integral[i]
                                       - MLi*R_diss_i;
            }//i
        }//implicit edge-based
    }

  void invert(arguments_dict& args)
  {
    int numDOFs = args.scalar<int>("numDOFs");
    proteus::pyarray<double>& mIn = args.array<double>("mIn");
    proteus::pyarray<double>& uOut = args.array<double>("uOut");
    proteus::pyarray<double>& nodal_porosity = args.array<double>("nodal_porosity");
    const double rho_f = args.scalar<double>("rho_f");
    const double rho_s = args.scalar<double>("rho_s");
    for (int i=0; i<numDOFs; i++)
      uOut.data()[i] = inversevaluateCoefficients(mIn.data()[i], nodal_porosity.data()[i], rho_f, rho_s);
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
      proteus::pyarray<double>& u_dof = args.array<double>("u_dof");
      proteus::pyarray<double>& velocity = args.array<double>("velocity");
      proteus::pyarray<double>& q_porosity = args.array<double>("q_porosity");
      proteus::pyarray<double>& q_rho = args.array<double>("q_rho");
      proteus::pyarray<double>& q_m_betaBDF = args.array<double>("q_m_betaBDF");
      proteus::pyarray<double>& cfl = args.array<double>("cfl");
      proteus::pyarray<double>& q_numDiff_u_last = args.array<double>("q_numDiff_u_last");
      proteus::pyarray<int>& csrRowIndeces_u_u = args.array<int>("csrRowIndeces_u_u");
      proteus::pyarray<int>& csrColumnOffsets_u_u = args.array<int>("csrColumnOffsets_u_u");
      proteus::pyarray<double>& globalJacobian = args.array<double>("globalJacobian");
      int nExteriorElementBoundaries_global = args.scalar<int>("nExteriorElementBoundaries_global");
      proteus::pyarray<int>& exteriorElementBoundariesArray = args.array<int>("exteriorElementBoundariesArray");
      proteus::pyarray<int>& elementBoundaryMaterialTypes = args.array<int>("elementBoundaryMaterialTypes");
      proteus::pyarray<int>& isExteriorBoundaryPhysical = args.array<int>("isExteriorBoundaryPhysical");
      proteus::pyarray<int>& elementBoundaryElementsArray = args.array<int>("elementBoundaryElementsArray");
      proteus::pyarray<int>& elementBoundaryLocalElementBoundariesArray = args.array<int>("elementBoundaryLocalElementBoundariesArray");
      proteus::pyarray<double>& ebqe_velocity_ext = args.array<double>("ebqe_velocity_ext");
      proteus::pyarray<int>& isDOFBoundary_u = args.array<int>("isDOFBoundary_u");
      proteus::pyarray<double>& ebqe_bc_u_ext = args.array<double>("ebqe_bc_u_ext");
      proteus::pyarray<int>& isFluxBoundary_u = args.array<int>("isFluxBoundary_u");
      proteus::pyarray<double>& ebqe_bc_flux_u_ext = args.array<double>("ebqe_bc_flux_u_ext");
      proteus::pyarray<double>& ebqe_porosity = args.array<double>("ebqe_porosity");
      proteus::pyarray<double>& ebqe_rho = args.array<double>("ebqe_rho");
      proteus::pyarray<int>& csrColumnOffsets_eb_u_u = args.array<int>("csrColumnOffsets_eb_u_u");
      STABILIZATION STABILIZATION_TYPE = static_cast<STABILIZATION>(args.scalar<int>("STABILIZATION_TYPE"));
//      ENTROPY ENTROPY_TYPE = static_cast<ENTROPY>(args.scalar<int>("ENTROPY_TYPE"));    
//      STABILIZATION STABILIZATION_TYPE{args.scalar<int>("STABILIZATION_TYPE")};
      double physicalDiffusion = args.scalar<double>("physicalDiffusion");
      const double alpha_L = args.scalar<double>("alpha_L");
      const double alpha_T = args.scalar<double>("alpha_T");
      const double Dm = args.scalar<double>("Dm");
      int forceStrongConditions = args.scalar<int>("forceStrongConditions");
      // const int dispersion_type_int = args.scalar<int>("dispersion_type");
      // const double theta_s = args.scalar<double>("theta_s");
      // const double theta_r = args.scalar<double>("theta_r");
      // const double power_law_exponent = args.scalar<double>("power_law_exponent");
      // const double velocity_exponent = args.scalar<double>("velocity_exponent");
      const double rho_f = args.scalar<double>("rho_f");
      const double rho_s = args.scalar<double>("rho_s");
      
      // DISPERSION DISPERSION_TYPE = static_cast<DISPERSION>(dispersion_type_int);


      double Ct_sge = 4.0;



            /////////////////////////////////////////////////////////////////////////
      proteus::pyarray<int>& a_rowptr = args.array<int>("a_rowptr");
      proteus::pyarray<int>& a_colind = args.array<int>("a_colind");
      //proteus::pyarray<double>& D = args.array<double>("D");
      //////////////////////////////////////////////////////////////////////////
      proteus::pyarray<int>& isDiffusiveFluxBoundary_u = args.array<int>("isDiffusiveFluxBoundary_u");
      proteus::pyarray<double>& ebqe_penalty_ext = args.array<double>("ebqe_penalty_ext");

      //////////////////////////////////////////////////////////////////////
      // ImplicitEV (STAB=5): assemble the implicit edge-based Jacobian in  //
      // THIS function (unlike Richards, which builds it inside the         //
      // residual).  It reads the transport/diffusion matrices and nodal    //
      // projections (theta_dof_proj) that the immediately preceding        //
      // calculateResidual call filled -- Proteus Newton evaluates F(u)     //
      // then J(u) at the same iterate -- plus the frozen per-edge upwind   //
      // coefficient a_ij the residual stashed in dLow[ij].  Single-comp P1 //
      // => the DOF-loop CSR coincides with the matrix CSR, so globalJacobian//
      // is indexed by the same running offset 'ij' as the residual         //
      // (identical to the Richards STAB==2 convention).                    //
      //                                                                    //
      // Differentiating the upwind flux  F_i = sum_{j!=i} -a_ij*(c_j-c_i)  //
      // with a_ij frozen gives a clean M-matrix:                           //
      //   J_ij = -a_ij                                  (j != i, <= 0)     //
      //   J_ii = ML_i*dmdu_i/dt + sum_{j!=i} a_ij  - ML_i*dRdiss_i/dc_i     //
      // dmdu_i = theta_i*(rho(c_i)+c_i*drho_du), rho(c)=rho_f(1+eps*c).     //
      // a_ij carries the upwind density/diffusion weighting; freezing it   //
      // (and the O(eps) rho_up dependence) is the only inexactness          //
      // (standard inexact-Newton, as Richards freezes Kr/psi).             //
      //////////////////////////////////////////////////////////////////////
      if (STABILIZATION_TYPE==STABILIZATION::ImplicitEV)
        {
          const double dt = args.scalar<double>("dt");
          const int numDOFs = args.scalar<int>("numDOFs");
          proteus::pyarray<double>& ML = args.array<double>("ML");
          proteus::pyarray<double>& dLow = args.array<double>("dLow");
          proteus::pyarray<double>& Sn_dof = args.array<double>("Sn_dof");
          const double k_d   = args.scalar<double>("k_d");
          const double c_sat = args.scalar<double>("c_sat");
          proteus::pyarray<int>& csrRowIndeces_DofLoops = args.array<int>("csrRowIndeces_DofLoops");
          proteus::pyarray<int>& csrColumnOffsets_DofLoops = args.array<int>("csrColumnOffsets_DofLoops");
          const double drho_du = rho_s - rho_f;

          // nodal storage Jacobian dmdu_i = theta_i*(rho(c_i) + c_i*drho_du)
          std::valarray<double> dmdu(numDOFs);
          for (int i=0; i<numDOFs; i++)
            {
              const double ci = u_dof.data()[i];
              const double theta_i = fmax(theta_dof_proj[i], 1.0e-14);
              const double rho_ci = rho_f*(1.0 + (drho_du/rho_f)*ci);
              dmdu[i] = theta_i*(rho_ci + ci*drho_du);
            }
          int ij=0;
          for (int i=0; i<numDOFs; i++)
            {
              const double ci = u_dof.data()[i];
              const double theta_i = fmax(theta_dof_proj[i], 1.0e-14);
              const double rho_ci = rho_f*(1.0 + (drho_du/rho_f)*ci);
              const double MLi = ML.data()[i];
              const double S_n_i = Sn_dof.data()[i];
              // dR_diss_i/dc_i = theta_i*k_d*S_n_i*[ drho_du*(c_sat-c_i) - rho(c_i) ]
              const double dRdiss_dci = theta_i*k_d*S_n_i*( drho_du*(c_sat - ci) - rho_ci );

              int diag_ij = -1;
              double sum_a = 0.0;
              for (int offset=csrRowIndeces_DofLoops.data()[i]; offset<csrRowIndeces_DofLoops.data()[i+1]; offset++)
                {
                  int j = csrColumnOffsets_DofLoops.data()[offset];
                  if (i != j)
                    {
                      // off-diagonal upwind flux: J_ij = -a_ij.  Recompute a_ij
                      // from the (frozen) transport/diffusion matrices and the
                      // current iterate -- dLow now stores the symmetric FCT
                      // viscosity, not a_ij.  Matches the residual exactly.
                      const double T_ij  = TransportMatrix[ij];
                      const double D_ij  = DiffusionMatrix[ij];
                      const double cj    = u_dof.data()[j];
                      const double rho_cj= rho_f*(1.0 + (drho_du/rho_f)*cj);
                      const double T_neg = fmax(0.0, -T_ij);
                      const double D_neg = fmax(0.0, -D_ij);
                      const double rho_up= (-T_ij*(cj - ci) <= 0.0) ? rho_ci : rho_cj;
                      const double a_ij  = T_neg*(rho_up/rho_f) + D_neg;
                      globalJacobian.data()[ij] += -a_ij;
                      sum_a += a_ij;
                    }
                  else
                    diag_ij = ij;
                  ij += 1;
                }
              // diagonal: storage + upwind self term (sum a_ij) - dissolution
              globalJacobian.data()[diag_ij] += MLi*dmdu[i]/dt
                                              + sum_a
                                              - MLi*dRdiss_dci;
            }//i

          //
          // Implicit boundary flux Jacobian: d(boundary flux)/du -> globalJacobian
          // so the boundary is implicit (like Richards), consistent with the
          // boundary residual the calculateResidual ImplicitEV path stashes in
          // boundary_integral.  fluxJacobian accumulates over the face
          // quadrature and is loaded ONCE per face (the residual side likewise
          // distributes once).  For closed BCs (zero adv/diff flux, no
          // Dirichlet) every term here is identically 0.
          //
          for (int ebNE = 0; ebNE < nExteriorElementBoundaries_global; ebNE++)
            {
              int ebN = exteriorElementBoundariesArray.data()[ebNE];
              const int eN_out = elementBoundaryElementsArray.data()[ebN*2+1];
              const int ebFlag = elementBoundaryMaterialTypes.data()[ebN];
              if (ebFlag <= 0 || isExteriorBoundaryPhysical.data()[ebNE] == 0 || eN_out >= 0)
                continue;
              int eN  = elementBoundaryElementsArray.data()[ebN*2+0],
                ebN_local = elementBoundaryLocalElementBoundariesArray.data()[ebN*2+0],
                eN_nDOF_trial_element = eN*nDOF_trial_element;
              double fluxJacobian_u_u[nDOF_test_element][nDOF_trial_element];
              for (int i=0;i<nDOF_test_element;i++)
                for (int j=0;j<nDOF_trial_element;j++)
                  fluxJacobian_u_u[i][j]=0.0;
              for (int kb=0;kb<nQuadraturePoints_elementBoundary;kb++)
                {
                  int ebNE_kb = ebNE*nQuadraturePoints_elementBoundary+kb,
                    ebNE_kb_nSpace = ebNE_kb*nSpace,
                    ebN_local_kb = ebN_local*nQuadraturePoints_elementBoundary+kb,
                    ebN_local_kb_nSpace = ebN_local_kb*nSpace;
                  double u_ext=0.0, grad_u_ext[nSpace], m_ext=0.0, dm_ext=0.0,
                    f_ext[nSpace], df_ext[nSpace], a_ext[nnz], da_ext[nnz],
                    bc_a_ext[nnz], bc_da_ext[nnz], difffluxjacobian_ext=0.0,
                    bc_u_ext=0.0, bc_m_ext=0.0, bc_dm_ext=0.0,
                    bc_f_ext[nSpace], bc_df_ext[nSpace],
                    jac_ext[nSpace*nSpace], jacDet_ext, jacInv_ext[nSpace*nSpace],
                    boundaryJac[nSpace*(nSpace-1)], metricTensor[(nSpace-1)*(nSpace-1)],
                    metricTensorDetSqrt, dS, u_test_dS[nDOF_test_element],
                    u_grad_trial_trace[nDOF_trial_element*nSpace],
                    u_grad_test_dS[nDOF_trial_element*nSpace], normal[nSpace],
                    x_ext,y_ext,z_ext,xt_ext,yt_ext,zt_ext,integralScaling,
                    G[nSpace*nSpace],G_dd_G,tr_G;
                  ck.calculateMapping_elementBoundary(eN,ebN_local,kb,ebN_local_kb,mesh_dof.data(),mesh_l2g.data(),mesh_trial_trace_ref.data(),mesh_grad_trial_trace_ref.data(),boundaryJac_ref.data(),jac_ext,jacDet_ext,jacInv_ext,boundaryJac,metricTensor,metricTensorDetSqrt,normal_ref.data(),normal,x_ext,y_ext,z_ext);
                  ck.calculateMappingVelocity_elementBoundary(eN,ebN_local,kb,ebN_local_kb,mesh_velocity_dof.data(),mesh_l2g.data(),mesh_trial_trace_ref.data(),xt_ext,yt_ext,zt_ext,normal,boundaryJac,metricTensor,integralScaling);
                  dS = ((1.0-MOVING_DOMAIN)*metricTensorDetSqrt + MOVING_DOMAIN*integralScaling)*dS_ref.data()[kb];
                  ck.calculateG(jacInv_ext,G,G_dd_G,tr_G);
                  ck.gradTrialFromRef(&u_grad_trial_trace_ref.data()[ebN_local_kb_nSpace*nDOF_trial_element],jacInv_ext,u_grad_trial_trace);
                  ck.valFromDOF(u_dof.data(),&u_l2g.data()[eN_nDOF_trial_element],&u_trial_trace_ref.data()[ebN_local_kb*nDOF_test_element],u_ext);
                  ck.gradFromDOF(u_dof.data(),&u_l2g.data()[eN_nDOF_trial_element],u_grad_trial_trace,grad_u_ext);
                  for (int j=0;j<nDOF_trial_element;j++)
                    {
                      u_test_dS[j] = u_test_trace_ref.data()[ebN_local_kb*nDOF_test_element+j]*dS;
                      for (int I=0;I<nSpace;I++)
                        u_grad_test_dS[j*nSpace+I]= u_grad_trial_trace[j*nSpace+I]*dS;
                    }
                  bc_u_ext = isDOFBoundary_u.data()[ebNE_kb]*ebqe_bc_u_ext.data()[ebNE_kb]+(1-isDOFBoundary_u.data()[ebNE_kb])*u_ext;
                  double rho_out_ext=0.0,rho_out_bc=0.0;
                  evaluateCoefficients(a_rowptr.data(),a_colind.data(),&ebqe_velocity_ext.data()[ebNE_kb_nSpace],alpha_L,alpha_T,Dm,ebqe_porosity.data()[ebNE_kb],rho_f,rho_s,u_ext,rho_out_ext,m_ext,dm_ext,f_ext,df_ext,a_ext,da_ext);
                  evaluateCoefficients(a_rowptr.data(),a_colind.data(),&ebqe_velocity_ext.data()[ebNE_kb_nSpace],alpha_L,alpha_T,Dm,ebqe_porosity.data()[ebNE_kb],rho_f,rho_s,bc_u_ext,rho_out_bc,bc_m_ext,bc_dm_ext,bc_f_ext,bc_df_ext,bc_a_ext,bc_da_ext);
                  double mesh_velocity[3]; mesh_velocity[0]=xt_ext; mesh_velocity[1]=yt_ext; mesh_velocity[2]=zt_ext;
                  for (int I=0;I<nSpace;I++)
                    {
                      f_ext[I] -= MOVING_DOMAIN*m_ext*mesh_velocity[I];
                      df_ext[I] -= MOVING_DOMAIN*dm_ext*mesh_velocity[I];
                      bc_f_ext[I] -= MOVING_DOMAIN*bc_m_ext*mesh_velocity[I];
                      bc_df_ext[I] -= MOVING_DOMAIN*bc_dm_ext*mesh_velocity[I];
                    }
                  for (int i=0;i<nDOF_test_element;i++)
                    for (int j=0;j<nDOF_trial_element;j++)
                      {
                        int ebN_local_kb_j=ebN_local_kb*nDOF_trial_element+j;
                        double advJacobian_ext = 0.0, diffJacobian_ext = 0.0;
                        exteriorNumericalAdvectiveFluxDerivative(isDOFBoundary_u.data()[ebNE_kb],isFluxBoundary_u.data()[ebNE_kb],forceStrongConditions,normal,df_ext,advJacobian_ext);
                        exteriorNumericalDiffusiveFluxDerivative(isDOFBoundary_u.data()[ebNE_kb],isDiffusiveFluxBoundary_u.data()[ebNE_kb],a_rowptr.data(),a_colind.data(),normal,a_ext,da_ext,grad_u_ext,&u_grad_trial_trace[j*nSpace],u_trial_trace_ref.data()[ebN_local_kb_j],ebqe_penalty_ext.data()[ebNE_kb],diffJacobian_ext);
                        difffluxjacobian_ext = advJacobian_ext*u_trial_trace_ref.data()[ebN_local_kb_j] + diffJacobian_ext;
                        fluxJacobian_u_u[i][j] += difffluxjacobian_ext*u_test_dS[i];
                      }//j
                }//kb
              for (int i=0;i<nDOF_test_element;i++)
                {
                  int eN_i = eN*nDOF_test_element+i;
                  for (int j=0;j<nDOF_trial_element;j++)
                    {
                      int ebN_i_j = ebN*4*nDOF_test_X_trial_element + i*nDOF_trial_element + j;
                      globalJacobian.data()[csrRowIndeces_u_u.data()[eN_i] + csrColumnOffsets_eb_u_u.data()[ebN_i_j]] += fluxJacobian_u_u[i][j];
                    }//j
                }//i
            }//ebNE
          return;  // skip the element/boundary Jacobian loops below
        }

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
                f[nSpace],df[nSpace],
                a[nnz],da[nnz],
                
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
              
              double rho_out=0.0;
              evaluateCoefficients(a_rowptr.data(),
				                           a_colind.data(),
                                   &velocity.data()[eN_k_nSpace],
                                   alpha_L,
                                   alpha_T,
                                   Dm,
                                   q_porosity.data()[eN*nQuadraturePoints_element+k],
                                   rho_f,
                                   rho_s,
                                   u,
                                   rho_out,
                                   m,
                                   dm,
                                   f,
                                   df,
                                   a,
                                   da);
              //
              //moving mesh
              //
              double mesh_velocity[3];
              mesh_velocity[0] = xt;
              mesh_velocity[1] = yt;
              mesh_velocity[2] = zt;
            
              for(int I=0;I<nSpace;I++)
                {
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
              if (STABILIZATION_TYPE == STABILIZATION::VMS)
                {
                  //
                  //calculate subgrid error contribution to the Jacobian (strong residual, adjoint, jacobian of strong residual)
                  //
                  //calculate the adjoint times the test functions
                  for (int i=0;i<nDOF_test_element;i++)
                    {
                      int i_nSpace = i*nSpace;
                      Lstar_u[i]=ck.Advection_adjoint(df,&u_grad_test_dV[i_nSpace]);
                    }
                  //calculate the Jacobian of strong residual
                  for (int j=0;j<nDOF_trial_element;j++)
                    {
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
                }
              for(int i=0;i<nDOF_test_element;i++)
                {
                  for(int j=0;j<nDOF_trial_element;j++)
                    {
                      int j_nSpace = j*nSpace;
                      int i_nSpace = i*nSpace;
                      if (STABILIZATION_TYPE==STABILIZATION::Galerkin)
                        {
                          elementJacobian_u_u[i][j] +=
                            ck.MassJacobian_weak(dm_t,
                                                 u_trial_ref.data()[k*nDOF_trial_element+j],
                                                 u_test_dV[i]) +
                            ck.AdvectionJacobian_weak(df,
                                                      u_trial_ref.data()[k*nDOF_trial_element+j],
                                                      &u_grad_test_dV[i_nSpace]) +
                            ck.DiffusionJacobian_weak(a_rowptr.data(),a_colind.data(),a,da,
						                                          grad_u,&u_grad_test_dV[i_nSpace],1.0,
						                                          u_trial_ref.data()[k*nDOF_trial_element+j],&u_grad_trial[j_nSpace])
                                                      +
                            ck.NumericalDiffusionJacobian(physicalDiffusion,
                                                          &u_grad_trial[j_nSpace],
                                                          &u_grad_test_dV[i_nSpace]); //implicit
                        }
                      else if (STABILIZATION_TYPE==STABILIZATION::VMS)
                        {
                          elementJacobian_u_u[i][j] +=
                            ck.MassJacobian_weak(dm_t,
                                                 u_trial_ref.data()[k*nDOF_trial_element+j],
                                                 u_test_dV[i]) +
                            ck.AdvectionJacobian_weak(df,
                                                      u_trial_ref.data()[k*nDOF_trial_element+j],
                                                      &u_grad_test_dV[i_nSpace]) +
                            ck.DiffusionJacobian_weak(a_rowptr.data(),a_colind.data(),a,da,
                                                      grad_u,&u_grad_test_dV[i_nSpace],1.0,
                                                      u_trial_ref.data()[k*nDOF_trial_element+j],&u_grad_trial[j_nSpace])+

                            ck.SubgridErrorJacobian(dsubgridError_u_u[j],Lstar_u[i]) +
                            ck.NumericalDiffusionJacobian(q_numDiff_u_last.data()[eN_k] + physicalDiffusion,
                                                          &u_grad_trial[j_nSpace],
                                                          &u_grad_test_dV[i_nSpace]); //implicit
                        }
                      else if (STABILIZATION_TYPE==STABILIZATION::TaylorGalerkinEV or 
                               STABILIZATION_TYPE==STABILIZATION::EntropyViscosity or
                               STABILIZATION_TYPE==STABILIZATION::SmoothnessIndicator or 
                               STABILIZATION_TYPE==STABILIZATION::Kuzmin)
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
      //
      //loop over exterior element boundaries to compute the surface integrals and load them into the global Jacobian
      //
      if (STABILIZATION_TYPE==STABILIZATION::VMS or STABILIZATION_TYPE==STABILIZATION::Galerkin)
        {
          for (int ebNE = 0; ebNE < nExteriorElementBoundaries_global; ebNE++)
            {
              int ebN = exteriorElementBoundariesArray.data()[ebNE];
              const int eN_out = elementBoundaryElementsArray.data()[ebN*2+1];
              const int ebFlag = elementBoundaryMaterialTypes.data()[ebN];
              if (ebFlag <= 0 || isExteriorBoundaryPhysical.data()[ebNE] == 0 || eN_out >= 0)
                continue;
              int eN  = elementBoundaryElementsArray.data()[ebN*2+0],
                ebN_local = elementBoundaryLocalElementBoundariesArray.data()[ebN*2+0],
                eN_nDOF_trial_element = eN*nDOF_trial_element;
              double fluxJacobian_u_u[nDOF_test_element][nDOF_trial_element];
              for (int i=0;i<nDOF_test_element;i++)
                for (int j=0;j<nDOF_trial_element;j++)
                  {
                    fluxJacobian_u_u[i][j]=0.0;
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
                    f_ext[nSpace],
                    df_ext[nSpace],

                    a_ext[nnz],
                    da_ext[nnz],
                    bc_a_ext[nnz],
                    bc_da_ext[nnz],

                    dflux_u_u_ext=0.0,
                    difffluxjacobian_ext=0.0,
                    bc_u_ext=0.0,
                    //bc_grad_u_ext[nSpace],
                    bc_m_ext=0.0,
                    bc_dm_ext=0.0,
                    bc_f_ext[nSpace],
                    bc_df_ext[nSpace],
                    //////////////
                    diffusiveFluxJacobian_u_u[nDOF_trial_element],
                    ////////////////////////////
                    jac_ext[nSpace*nSpace],
                    jacDet_ext,
                    jacInv_ext[nSpace*nSpace],
                    boundaryJac[nSpace*(nSpace-1)],
                    metricTensor[(nSpace-1)*(nSpace-1)],
                    metricTensorDetSqrt,
                    dS,
                    u_test_dS[nDOF_test_element],
                    u_grad_trial_trace[nDOF_trial_element*nSpace],
                    u_grad_test_dS[nDOF_trial_element*nSpace],
                    normal[nSpace],x_ext,y_ext,z_ext,xt_ext,yt_ext,zt_ext,integralScaling,
                    //
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
                      for (int I=0;I<nSpace;I++)
                      {
                        u_grad_test_dS[j*nSpace+I]= u_grad_trial_trace[j*nSpace+I]*dS;
                      }
                    }
                  //
                  //load the boundary values
                  //
                  bc_u_ext = isDOFBoundary_u.data()[ebNE_kb]*ebqe_bc_u_ext.data()[ebNE_kb]+(1-isDOFBoundary_u.data()[ebNE_kb])*u_ext;
          
                  //
                  //
                  //calculate the internal and external trace of the pde coefficients
                  //

                  double rho_out_ext=0.0,rho_out_bc=0.0;
                  evaluateCoefficients(a_rowptr.data(),
                                       a_colind.data(),
                                       &ebqe_velocity_ext.data()[ebNE_kb_nSpace],
                                       alpha_L,
                                       alpha_T,
                                       Dm,
                                   ebqe_porosity.data()[ebNE_kb],
                                       rho_f,
                                       rho_s,
                                       u_ext,
                                       rho_out_ext,
                                       m_ext,
                                       dm_ext,
                                       f_ext,
                                       df_ext,
                                       a_ext,
                                       da_ext
                                       );

                  evaluateCoefficients(a_rowptr.data(),
                                       a_colind.data(),
                                       &ebqe_velocity_ext.data()[ebNE_kb_nSpace],
                                       alpha_L,
                                       alpha_T,
                                       Dm,
                                   ebqe_porosity.data()[ebNE_kb],
                                       rho_f,
                                       rho_s,
                                       bc_u_ext,
                                       rho_out_bc,
                                       bc_m_ext,
                                       bc_dm_ext,
                                       bc_f_ext,
                                       bc_df_ext,
                                       bc_a_ext,
                                       bc_da_ext);
                  //
                  //moving domain
                  //
                  double mesh_velocity[3];
                  mesh_velocity[0] = xt_ext;
                  mesh_velocity[1] = yt_ext;
                  mesh_velocity[2] = zt_ext;
                  for (int I=0;I<nSpace;I++)
                    {
                      f_ext[I] -= MOVING_DOMAIN*m_ext*mesh_velocity[I];
                      df_ext[I] -= MOVING_DOMAIN*dm_ext*mesh_velocity[I];
                      bc_f_ext[I] -= MOVING_DOMAIN*bc_m_ext*mesh_velocity[I];
                      bc_df_ext[I] -= MOVING_DOMAIN*bc_dm_ext*mesh_velocity[I];
                    }
                  //
                  //calculate the numerical fluxes
                  //
                  //
                  //calculate the flux jacobian
                  //
                  for (int i=0;i<nDOF_test_element;i++)
                    for (int j=0;j<nDOF_trial_element;j++)
                      {
                        int ebN_local_kb_j=ebN_local_kb*nDOF_trial_element+j;
  
                        double advJacobian_ext = 0.0, diffJacobian_ext = 0.0;
                        exteriorNumericalAdvectiveFluxDerivative(isDOFBoundary_u.data()[ebNE_kb],
                                                                 isFluxBoundary_u.data()[ebNE_kb],
                                                                 forceStrongConditions,
                                                                 normal,
                                                                 df_ext,
                                                                 advJacobian_ext);
                        exteriorNumericalDiffusiveFluxDerivative(isDOFBoundary_u.data()[ebNE_kb],
                                                                 isDiffusiveFluxBoundary_u.data()[ebNE_kb],
                                                                 a_rowptr.data(),
                                                                 a_colind.data(),
                                                                 normal,
                                                                 a_ext,
                                                                 da_ext,
                                                                 grad_u_ext,
                                                                 &u_grad_trial_trace[j*nSpace],
                                                                 u_trial_trace_ref.data()[ebN_local_kb_j],
                                                                 ebqe_penalty_ext.data()[ebNE_kb],
                                                                 diffJacobian_ext);
                        difffluxjacobian_ext = advJacobian_ext*u_trial_trace_ref.data()[ebN_local_kb_j]
                                               + diffJacobian_ext;
                        fluxJacobian_u_u[i][j] += difffluxjacobian_ext*u_test_dS[i];
                      }//j

              //
              //update the global Jacobian from the flux Jacobian
              //
              for (int i=0;i<nDOF_test_element;i++)
                {
                  int eN_i = eN*nDOF_test_element+i;
                  for (int j=0;j<nDOF_trial_element;j++)
                    {
                      int ebN_i_j = ebN*4*nDOF_test_X_trial_element + i*nDOF_trial_element + j;
                      globalJacobian.data()[csrRowIndeces_u_u.data()[eN_i] + csrColumnOffsets_eb_u_u.data()[ebN_i_j]] += fluxJacobian_u_u[i][j];
//                                                                                                                         
                    }//j
                }//i
              }//kb
            }//ebNE
        }//VMS and Galerkin
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
    // bc_mask: 0.0 at Dirichlet DOFs, 1.0 elsewhere.  Mirrors mphase_co2's
    // pattern.  Multiplied into the antidiffusive flux correction below so
    // Dirichlet DOFs stay at their low-order value (which already carries the
    // Nitsche BC contribution from the boundary residual) instead of being
    // antidiffused past the BC value.
    proteus::pyarray<double>& bc_mask = args.array<double>("bc_mask");
    proteus::pyarray<int>& csrRowIndeces_DofLoops = args.array<int>("csrRowIndeces_DofLoops");
    proteus::pyarray<int>& csrColumnOffsets_DofLoops = args.array<int>("csrColumnOffsets_DofLoops");
    proteus::pyarray<double>& MassMatrix = args.array<double>("MassMatrix");
    proteus::pyarray<double>& dt_times_dH_minus_dL = args.array<double>("dt_times_dH_minus_dL");
    proteus::pyarray<double>& min_u_bc = args.array<double>("min_u_bc");
    proteus::pyarray<double>& max_u_bc = args.array<double>("max_u_bc");
    int LUMPED_MASS_MATRIX = args.scalar<int>("LUMPED_MASS_MATRIX");
    // projection arrays replacing nodal_porosity
    proteus::pyarray<double>& q_porosity_old_fct = args.array<double>("q_porosity_old_fct");
    proteus::pyarray<double>& q_rho_fct        = args.array<double>("q_rho_fct");
    proteus::pyarray<double>& q_dV_fct         = args.array<double>("q_dV_fct");
    proteus::pyarray<int>&    u_l2g_fct        = args.array<int>("u_l2g_fct");
    proteus::pyarray<double>& u_test_ref_fct   = args.array<double>("u_test_ref_fct");
    proteus::pyarray<double>& theta_dof_out    = args.array<double>("theta_dof_out");
    int nElements_global_fct              = args.scalar<int>("nElements_global_fct");
    int nQuadraturePoints_element_fct     = args.scalar<int>("nQuadraturePoints_element_fct");
    int nDOF_trial_element_fct            = args.scalar<int>("nDOF_trial_element_fct");
    const double rho_f = args.scalar<double>("rho_f");
    const double rho_s = args.scalar<double>("rho_s");
    // phase 0 computes the limiter coefficients R of every dof, phase 1 applies them. The limiter of an owned dof
    // reads R of its neighbours, whose graph rows are truncated when they are ghosts, so Python scatters R from
    // the owners between the two phases.
    const int phase = args.scalar<int>("phase");
    proteus::pyarray<double>& Rpos_dof = args.array<double>("Rpos_dof");
    proteus::pyarray<double>& Rneg_dof = args.array<double>("Rneg_dof");
//    STABILIZATION STABILIZATION_TYPE{args.scalar<int>("STABILIZATION_TYPE")};
    STABILIZATION STABILIZATION_TYPE = static_cast<STABILIZATION>(args.scalar<int>("STABILIZATION_TYPE"));
//      ENTROPY ENTROPY_TYPE = static_cast<ENTROPY>(args.scalar<int>("ENTROPY_TYPE"));    
    // --- L2 projection: quadrature → DOF ---
    std::vector<double> theta_dof(numDOFs, 0.0);
    std::vector<double> rho_dof(numDOFs, 0.0);
    std::vector<double> ML_proj(numDOFs, 0.0);
    for (int eN = 0; eN < nElements_global_fct; eN++) {
      for (int k = 0; k < nQuadraturePoints_element_fct; k++) {
        const int eN_k = eN * nQuadraturePoints_element_fct + k;
        const double dV_k    = q_dV_fct.data()[eN_k];
        const double theta_k = q_porosity_old_fct.data()[eN_k];
        const double rho_k   = q_rho_fct.data()[eN_k];
        for (int i = 0; i < nDOF_trial_element_fct; i++) {
          const int gi = u_l2g_fct.data()[eN * nDOF_trial_element_fct + i];
          const double w = u_test_ref_fct.data()[k * nDOF_trial_element_fct + i] * dV_k;
          theta_dof[gi] += theta_k * w;
          rho_dof[gi]   += rho_k   * w;
          ML_proj[gi]   += w;
        }
      }
    }
    for (int i = 0; i < numDOFs; i++) {
      if (ML_proj[i] > 1.0e-14) { theta_dof[i] /= ML_proj[i]; rho_dof[i] /= ML_proj[i]; }
      else { theta_dof[i] = 1.0; rho_dof[i] = rho_f; }
    }
    // The projection of a ghost dof sums only the elements this rank has, and the limiter of an owned dof reads
    // theta_j. Phase -1 writes the projection to theta_dof_out and returns; Python scatters it from the owners and
    // phases 0 and 1 take those values.
    if (phase == -1)
      {
        for (int i=0; i<numDOFs; i++)
          theta_dof_out.data()[i] = theta_dof[i];
        return;
      }
    for (int i=0; i<numDOFs; i++)
      theta_dof[i] = theta_dof_out.data()[i];
    if (phase == 0)
      {
        // resize also zeroes, so phase 1 must not do it: it needs the matrix phase 0 filled
        Rpos.resize(numDOFs,0.0);
        Rneg.resize(numDOFs,0.0);
        FluxCorrectionMatrix.resize(NNZ,0.0);
      }
    int ij=0;
    //loop over nodes (i)
    if (phase == 0)
    for (int i=0; i<numDOFs; i++)
      {
        //read some vectors
        double solHi = solH.data()[i];
        double solni = soln.data()[i];
        const double lumped_volume = lumped_mass_matrix.data()[i];
        double uLowi = uLow.data()[i];
        const double theta_i = theta_dof[i];
        double mLowi = theta_i*rho_f*(1.0 + ((rho_s-rho_f)/rho_f)*uLowi)*uLowi;
        double solHmi = theta_i*rho_f*(1.0 + ((rho_s-rho_f)/rho_f)*solHi)*solHi;
        double solnmi = theta_i*rho_f*(1.0 + ((rho_s-rho_f)/rho_f)*solni)*solni;
        double uDotLowi = (mLowi - solnmi)/dt;
        double mini=min_u_bc.data()[i], maxi=max_u_bc.data()[i]; // init min/max with value at BCs (NOTE: if no boundary then min=1E10, max=-1E10)
        double Pposi=0, Pnegi=0;
        // Loop over neighbors (j)
        for (int offset=csrRowIndeces_DofLoops.data()[i]; offset<csrRowIndeces_DofLoops.data()[i+1]; offset++)
          {
            assert(offset == ij); // (CSR matrix consistency)
            int j = csrColumnOffsets_DofLoops.data()[offset];
            double solnj = soln.data()[j];
            double uLowj = uLow.data()[j];
            ////////////////////////
            // COMPUTE THE BOUNDS //
            ////////////////////////
            
            // Explicit paths bound against soln (as VOF.h/VOF3P.h do);
            // ImplicitEV, whose limiter works on mass, bounds against uLow.
            const double bound_j =
              (STABILIZATION_TYPE == STABILIZATION::ImplicitEV) ? uLowj : solnj;
            mini = fmin(mini,bound_j);
            maxi = fmax(maxi,bound_j);
            const double theta_j = theta_dof[j];
            double mLowj = theta_j*rho_f*(1.0 + ((rho_s-rho_f)/rho_f)*uLowj)*uLowj;
            double solHmj = theta_j*rho_f*(1.0 + ((rho_s-rho_f)/rho_f)*solH.data()[j])*solH.data()[j];
            double solnmj = theta_j*rho_f*(1.0 + ((rho_s-rho_f)/rho_f)*solnj)*solnj;
            double uDotLowj = (mLowj - solnmj)/dt;
            // i-th row of flux correction matrix
            if (STABILIZATION_TYPE == STABILIZATION::Kuzmin)
              {
                // Explicit Kuzmin antidiffusive flux: consistent-mass correction +
                // full removal of the symmetric graph viscosity dLow*(mLow_i-mLow_j).
                FluxCorrectionMatrix[ij] = dt*(MassMatrix.data()[ij]*(uDotLowi-uDotLowj)
                                               + dLow.data()[ij]*(mLowi-mLowj));
              }
            else if (STABILIZATION_TYPE == STABILIZATION::ImplicitEV)
              {
                // Standard Kuzmin implicit FEM-FCT antidiffusive flux:
                //   F_ij = M~_ij[(m_j^H - m_j^n) - (m_i^H - m_i^n)]
                //          + dt (d^H - d^L)_ij (m_j^n - m_i^n)
                // For ImplicitEV the Newton solve gives the low-order implicit
                // solution, so the high-order state is taken as m^H := m^L (uLow):
                //   first term  -M~_ij[(m_j^L-m_j^n)-(m_i^L-m_i^n)]
                //               = dt*MC_ij*(uDotLow_i - uDotLow_j)   (consistent mass)
                //   second term  dt_times_dH_minus_dL_ij*(m_j^n - m_i^n)
                //               with dt_times_dH_minus_dL = dt*(d^H-d^L) = -dt*dLow.
                // The dissipation antidiffusion now acts on the OLD-time mass m^n
                // (solnm), matching the textbook Kuzmin form and the STAB=2
                // (EntropyViscosity) `else` branch below.  The previous code used the
                // diffused low-order m^L here and threw away most of dLow via the Comp
                // throttle, so the front stayed under-sharpened.  solnm carries the
                // density nonlinearity m^s(c)=theta*rho_f*(c + eps c^2); the inverse
                // c = 2r/(1+sqrt(1+4 eps r)) is applied in invert().
                FluxCorrectionMatrix[ij] = dt*MassMatrix.data()[ij]*(uDotLowi-uDotLowj)
                                           + dt_times_dH_minus_dL.data()[ij]*(solnmj-solnmi);
              }
            else
              {
                double ML_minus_MC =
                  (LUMPED_MASS_MATRIX == 1 ? 0. : (i==j ? 1. : 0.)*lumped_volume - MassMatrix.data()[ij]);
                FluxCorrectionMatrix[ij] = ML_minus_MC * (solHmj-solnmj - (solHmi-solnmi))
                  + dt_times_dH_minus_dL.data()[ij]*(solnmj-solnmi);
              }
            Pposi += FluxCorrectionMatrix[ij]*((FluxCorrectionMatrix[ij] > 0) ? 1. : 0.);
            Pnegi += FluxCorrectionMatrix[ij]*((FluxCorrectionMatrix[ij] < 0) ? 1. : 0.);
            ij+=1;
          }//j
        const double Qposi = lumped_volume*(theta_i*rho_f*(1.0 + ((rho_s-rho_f)/rho_f)*maxi)*maxi - mLowi);
        const double Qnegi = lumped_volume*(theta_i*rho_f*(1.0 + ((rho_s-rho_f)/rho_f)*mini)*mini - mLowi);
        Rpos[i] = ((Pposi==0) ? 1. : fmax(0.0, fmin(1.0,Qposi/Pposi)));
        Rneg[i] = ((Pnegi==0) ? 1. : fmax(0.0, fmin(1.0,Qnegi/Pnegi)));
        Rpos_dof.data()[i] = Rpos[i];
        Rneg_dof.data()[i] = Rneg[i];
      }//i
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
        const double lumped_volume = lumped_mass_matrix.data()[i];
        for (int offset=csrRowIndeces_DofLoops.data()[i]; offset<csrRowIndeces_DofLoops.data()[i+1]; offset++)
          {
            assert(offset == ij); // (CSR matrix consistency
            int j = csrColumnOffsets_DofLoops.data()[offset];
            double Lij = 1;
            Lij = ((FluxCorrectionMatrix[ij]>0) ? fmin(Rposi,Rneg[j]) : fmin(Rnegi,Rpos[j]));
            ith_Limiter_times_FluxCorrectionMatrix += Lij * FluxCorrectionMatrix[ij];
            ij+=1;
          }
        const double uLowi = uLow.data()[i];
        const double theta_i = theta_dof[i];
        const double mLowi = theta_i*rho_f*(1.0 + ((rho_s-rho_f)/rho_f)*uLowi)*uLowi;
        // bc_mask.data()[i] is 0 at Dirichlet DOFs (freeze at mLow) and 1 elsewhere.
        limited_solution.data()[i] = mLowi + bc_mask.data()[i] * (1./lumped_volume * ith_Limiter_times_FluxCorrectionMatrix);
        theta_dof_out.data()[i] = theta_dof[i];
      }
    }//FCTStep

    //////////////////////////////////////////////////////////////////////
    // STABILIZATION_TYPE=Compatible: transport driven by the realized   //
    // fluid exchanges of the flow step (Barua & Kees, Sec. 2.2).        //
    //                                                                   //
    // Flow exports, per step, the antisymmetric edge exchanges          //
    // Fhat_ij = E_ij (fluid mass rate into i from j) and the nodal      //
    // masses m^n, m^{n+1}. The transport carries the storage            //
    //   M^{n+1} = M^n + (m^{n+1} - m^n),   M^0 = m^0,                   //
    // and B_i = M_L,i (M^{n+1}_i - M^n_i)/dt - sum_j Fhat_ij closes the  //
    // fluid balance exactly (boundary exchange plus solver residual).   //
    // Low order (backward Euler), cancelled form at free nodes:         //
    //   R_i = M_L,i M^n_i (c_i - c^n_i)/dt - sum_j gamma_ij (c_j - c_i)  //
    //         - B^+_i (c_in,i - c_i),                                    //
    //   gamma_ij = max(Fhat_ij,0) + max(0, -d_ij - |Fhat_ij|/2) >= 0,    //
    // d_ij = int theta rho_T(c_h) D(v_h) grad w_j . grad w_i, so a       //
    // constant c is reproduced exactly whatever the flow does. Strong   //
    // Dirichlet rows: R_i = c_i - c_D,i. The Jacobian (computeJacobian) //
    // freezes gamma (Picard in the dispersion's rho_T(c)).              //
    //////////////////////////////////////////////////////////////////////
    void calculateResidual_compatible(arguments_dict& args)
    {
      const double dt = args.scalar<double>("dt");
      const int numDOFs = args.scalar<int>("numDOFs");
      const int nElements_global = args.scalar<int>("nElements_global");
      proteus::pyarray<double>& mesh_trial_ref = args.array<double>("mesh_trial_ref");
      proteus::pyarray<double>& mesh_grad_trial_ref = args.array<double>("mesh_grad_trial_ref");
      proteus::pyarray<double>& mesh_dof = args.array<double>("mesh_dof");
      proteus::pyarray<int>& mesh_l2g = args.array<int>("mesh_l2g");
      proteus::pyarray<double>& dV_ref = args.array<double>("dV_ref");
      proteus::pyarray<double>& u_trial_ref = args.array<double>("u_trial_ref");
      proteus::pyarray<double>& u_grad_trial_ref = args.array<double>("u_grad_trial_ref");
      proteus::pyarray<int>& u_l2g = args.array<int>("u_l2g");
      proteus::pyarray<int>& r_l2g = args.array<int>("r_l2g");
      proteus::pyarray<int>& a_rowptr = args.array<int>("a_rowptr");
      proteus::pyarray<int>& a_colind = args.array<int>("a_colind");
      const double alpha_L = args.scalar<double>("alpha_L");
      const double alpha_T = args.scalar<double>("alpha_T");
      const double Dm = args.scalar<double>("Dm");
      const double rho_f = args.scalar<double>("rho_f");
      const double rho_s = args.scalar<double>("rho_s");
      proteus::pyarray<double>& q_porosity = args.array<double>("q_porosity");
      proteus::pyarray<double>& q_velocity_dispersion = args.array<double>("q_velocity_dispersion");
      proteus::pyarray<double>& u_dof = args.array<double>("u_dof");
      proteus::pyarray<double>& u_dof_old = args.array<double>("u_dof_old");
      proteus::pyarray<int>& csrRowIndeces_DofLoops = args.array<int>("csrRowIndeces_DofLoops");
      proteus::pyarray<int>& csrColumnOffsets_DofLoops = args.array<int>("csrColumnOffsets_DofLoops");
      proteus::pyarray<int>& csrRowIndeces_CellLoops = args.array<int>("csrRowIndeces_CellLoops");
      proteus::pyarray<int>& csrColumnOffsets_CellLoops = args.array<int>("csrColumnOffsets_CellLoops");
      proteus::pyarray<double>& ML = args.array<double>("ML");
      proteus::pyarray<double>& Fhat = args.array<double>("Fhat");
      proteus::pyarray<double>& Mn = args.array<double>("Mn");
      proteus::pyarray<double>& B = args.array<double>("B");
      proteus::pyarray<double>& cin = args.array<double>("cin");
      proteus::pyarray<double>& bc_mask = args.array<double>("bc_mask");
      proteus::pyarray<double>& cD = args.array<double>("cD");
      proteus::pyarray<double>& dDispersion = args.array<double>("dDispersion");
      proteus::pyarray<double>& globalResidual = args.array<double>("globalResidual");
      const int computeJacobian = args.scalar<int>("computeJacobian");
      proteus::pyarray<double>& globalJacobian = args.array<double>("globalJacobian");
      std::valarray<double> c(numDOFs), cn(numDOFs);
      for (int eN=0; eN<nElements_global; eN++)
        for (int j=0; j<nDOF_trial_element; j++)
          {
            const int eN_j = eN*nDOF_trial_element + j;
            c[r_l2g.data()[eN_j]] = u_dof.data()[u_l2g.data()[eN_j]];
            cn[r_l2g.data()[eN_j]] = u_dof_old.data()[u_l2g.data()[eN_j]];
          }
      // dispersion matrix d_ij(c)
      std::fill(dDispersion.begin(), dDispersion.end(), 0.0);
      for (int eN=0; eN<nElements_global; eN++)
        {
          double elementDispersion[nDOF_test_element][nDOF_trial_element];
          for (int i=0; i<nDOF_test_element; i++)
            for (int j=0; j<nDOF_trial_element; j++)
              elementDispersion[i][j] = 0.0;
          for (int k=0; k<nQuadraturePoints_element; k++)
            {
              const int eN_k = eN*nQuadraturePoints_element + k;
              double jac[nSpace*nSpace], jacDet, jacInv[nSpace*nSpace], x, y, z;
              double u_grad_trial[nDOF_trial_element*nSpace], u_grad_test_dV[nDOF_test_element*nSpace];
              ck.calculateMapping_element(eN, k, mesh_dof.data(), mesh_l2g.data(), mesh_trial_ref.data(), mesh_grad_trial_ref.data(),
                                          jac, jacDet, jacInv, x, y, z);
              const double dV = fabs(jacDet)*dV_ref.data()[k];
              ck.gradTrialFromRef(&u_grad_trial_ref.data()[k*nDOF_trial_element*nSpace], jacInv, u_grad_trial);
              double u = 0.0;
              ck.valFromDOF(u_dof.data(), &u_l2g.data()[eN*nDOF_trial_element], &u_trial_ref.data()[k*nDOF_trial_element], u);
              for (int j=0; j<nDOF_trial_element; j++)
                for (int I=0; I<nSpace; I++)
                  u_grad_test_dV[j*nSpace+I] = u_grad_trial[j*nSpace+I]*dV;
              double rho_out, m, dm, f[nSpace], df[nSpace], a[nnz], da[nnz];
              evaluateCoefficients(a_rowptr.data(), a_colind.data(), &q_velocity_dispersion.data()[eN_k*nSpace],
                                   alpha_L, alpha_T, Dm, q_porosity.data()[eN_k], rho_f, rho_s, u,
                                   rho_out, m, dm, f, df, a, da);
              for (int i=0; i<nDOF_test_element; i++)
                for (int j=0; j<nDOF_trial_element; j++)
                  elementDispersion[i][j] += ck.SimpleDiffusionJacobian_weak(a_rowptr.data(), a_colind.data(), a,
                                                                              &u_grad_trial[j*nSpace], &u_grad_test_dV[i*nSpace]);
            }
          for (int i=0; i<nDOF_test_element; i++)
            {
              const int eN_i = eN*nDOF_test_element + i;
              for (int j=0; j<nDOF_trial_element; j++)
                dDispersion.data()[csrRowIndeces_CellLoops.data()[eN_i] + csrColumnOffsets_CellLoops.data()[eN_i*nDOF_trial_element + j]]
                  += elementDispersion[i][j];
            }
        }
      // edge loop
      for (int i=0; i<numDOFs; i++)
        {
          if (bc_mask.data()[i] == 0.0)
            {
              globalResidual.data()[i] = c[i] - cD.data()[i];
              if (computeJacobian)
                for (int offset=csrRowIndeces_DofLoops.data()[i]; offset<csrRowIndeces_DofLoops.data()[i+1]; offset++)
                  globalJacobian.data()[offset] = (csrColumnOffsets_DofLoops.data()[offset] == i) ? 1.0 : 0.0;
              continue;
            }
          const double Bpos = fmax(B.data()[i], 0.0);
          const double storage = ML.data()[i]*Mn.data()[i]/dt;
          double res = storage*(c[i] - cn[i]) - Bpos*(cin.data()[i] - c[i]);
          double jac_ii = storage + Bpos;
          int ii = -1;
          for (int offset=csrRowIndeces_DofLoops.data()[i]; offset<csrRowIndeces_DofLoops.data()[i+1]; offset++)
            {
              const int j = csrColumnOffsets_DofLoops.data()[offset];
              if (j == i)
                {
                  ii = offset;
                  continue;
                }
              const double F = Fhat.data()[offset];
              const double gamma = fmax(F, 0.0) + fmax(0.0, -dDispersion.data()[offset] - 0.5*fabs(F));
              res -= gamma*(c[j] - c[i]);
              jac_ii += gamma;
              if (computeJacobian)
                globalJacobian.data()[offset] = -gamma;
            }
          globalResidual.data()[i] = res;
          if (computeJacobian)
            globalJacobian.data()[ii] = jac_ii;
        }
    }

    // Piecewise constant mass flux q_h with int q_h . grad w_i = sum_j E_ij (Sec. 2.2, compatibility
    // conditions). Each edge exchange is split among the elements sharing the edge in proportion to
    // |K^e_ij|, K^e the element Laplacian (equal shares where every |K^e_ij| vanishes); within a simplex
    // the d+1 element moments then sum to zero and determine q_e uniquely. Used for the dispersion only.
    void reconstructFlux_compatible(arguments_dict& args)
    {
      const int nElements_global = args.scalar<int>("nElements_global");
      proteus::pyarray<double>& mesh_trial_ref = args.array<double>("mesh_trial_ref");
      proteus::pyarray<double>& mesh_grad_trial_ref = args.array<double>("mesh_grad_trial_ref");
      proteus::pyarray<double>& mesh_dof = args.array<double>("mesh_dof");
      proteus::pyarray<int>& mesh_l2g = args.array<int>("mesh_l2g");
      proteus::pyarray<double>& dV_ref = args.array<double>("dV_ref");
      proteus::pyarray<double>& u_grad_trial_ref = args.array<double>("u_grad_trial_ref");
      proteus::pyarray<int>& csrRowIndeces_CellLoops = args.array<int>("csrRowIndeces_CellLoops");
      proteus::pyarray<int>& csrColumnOffsets_CellLoops = args.array<int>("csrColumnOffsets_CellLoops");
      proteus::pyarray<double>& Espatial = args.array<double>("Espatial");
      proteus::pyarray<double>& edgeWeight = args.array<double>("edgeWeight");
      proteus::pyarray<double>& edgeCount = args.array<double>("edgeCount");
      proteus::pyarray<double>& element_q = args.array<double>("element_q");
      std::vector<double> gradInt(nElements_global*nDOF_trial_element*nSpace, 0.0), K(nElements_global*nDOF_test_X_trial_element, 0.0);
      for (int eN=0; eN<nElements_global; eN++)
        for (int k=0; k<nQuadraturePoints_element; k++)
          {
            double jac[nSpace*nSpace], jacDet, jacInv[nSpace*nSpace], x, y, z, u_grad_trial[nDOF_trial_element*nSpace];
            ck.calculateMapping_element(eN, k, mesh_dof.data(), mesh_l2g.data(), mesh_trial_ref.data(), mesh_grad_trial_ref.data(),
                                        jac, jacDet, jacInv, x, y, z);
            const double dV = fabs(jacDet)*dV_ref.data()[k];
            ck.gradTrialFromRef(&u_grad_trial_ref.data()[k*nDOF_trial_element*nSpace], jacInv, u_grad_trial);
            for (int i=0; i<nDOF_trial_element; i++)
              {
                for (int I=0; I<nSpace; I++)
                  gradInt[(eN*nDOF_trial_element + i)*nSpace + I] += u_grad_trial[i*nSpace+I]*dV;
                for (int j=0; j<nDOF_trial_element; j++)
                  for (int I=0; I<nSpace; I++)
                    K[eN*nDOF_test_X_trial_element + i*nDOF_trial_element + j] += u_grad_trial[i*nSpace+I]*u_grad_trial[j*nSpace+I]*dV;
              }
          }
      std::fill(edgeWeight.begin(), edgeWeight.end(), 0.0);
      std::fill(edgeCount.begin(), edgeCount.end(), 0.0);
      for (int eN=0; eN<nElements_global; eN++)
        for (int i=0; i<nDOF_test_element; i++)
          {
            const int eN_i = eN*nDOF_test_element + i;
            for (int j=0; j<nDOF_trial_element; j++)
              {
                const int ij = csrRowIndeces_CellLoops.data()[eN_i] + csrColumnOffsets_CellLoops.data()[eN_i*nDOF_trial_element + j];
                edgeWeight.data()[ij] += fabs(K[eN*nDOF_test_X_trial_element + i*nDOF_trial_element + j]);
                edgeCount.data()[ij] += 1.0;
              }
          }
      for (int eN=0; eN<nElements_global; eN++)
        {
          double r[nDOF_test_element];
          for (int i=0; i<nDOF_test_element; i++)
            {
              const int eN_i = eN*nDOF_test_element + i;
              r[i] = 0.0;
              for (int j=0; j<nDOF_trial_element; j++)
                {
                  if (j == i)
                    continue;
                  const int ij = csrRowIndeces_CellLoops.data()[eN_i] + csrColumnOffsets_CellLoops.data()[eN_i*nDOF_trial_element + j];
                  const double Ke = fabs(K[eN*nDOF_test_X_trial_element + i*nDOF_trial_element + j]);
                  const double share = (edgeWeight.data()[ij] > 0.0) ? Ke/edgeWeight.data()[ij] : 1.0/edgeCount.data()[ij];
                  r[i] += share*Espatial.data()[ij];
                }
            }
          // least squares G q = r, G_iI = int_e d w_i / d x_I
          double A[nSpace][nSpace+1];
          for (int I=0; I<nSpace; I++)
            {
              for (int J=0; J<nSpace; J++)
                {
                  A[I][J] = 0.0;
                  for (int i=0; i<nDOF_test_element; i++)
                    A[I][J] += gradInt[(eN*nDOF_trial_element + i)*nSpace + I]*gradInt[(eN*nDOF_trial_element + i)*nSpace + J];
                }
              A[I][nSpace] = 0.0;
              for (int i=0; i<nDOF_test_element; i++)
                A[I][nSpace] += gradInt[(eN*nDOF_trial_element + i)*nSpace + I]*r[i];
            }
          for (int I=0; I<nSpace; I++)
            {
              int p = I;
              for (int J=I+1; J<nSpace; J++)
                if (fabs(A[J][I]) > fabs(A[p][I]))
                  p = J;
              for (int J=0; J<=nSpace; J++)
                std::swap(A[I][J], A[p][J]);
              for (int J=I+1; J<nSpace; J++)
                {
                  const double f = A[J][I]/A[I][I];
                  for (int L=I; L<=nSpace; L++)
                    A[J][L] -= f*A[I][L];
                }
            }
          for (int I=nSpace-1; I>=0; I--)
            {
              double s = A[I][nSpace];
              for (int J=I+1; J<nSpace; J++)
                s -= A[I][J]*element_q.data()[eN*nSpace + J];
              element_q.data()[eN*nSpace + I] = s/A[I][I];
            }
        }
    }

    // Zalesak limiting of the solute mass S = M c with concentration bounds (Algorithm 3).
    //   f_ij = dt m_ij (Sdot_i - Sdot_j) - dt nu_ij (c^n_j - c^n_i),  nu_ij = max(0, |Fhat_ij|/2 + d_ij),
    //   S^{n+1}_i = M^{n+1}_i c^L_i + (1/M_L,i) sum_j alpha_ij f_ij,  c^{n+1}_i = S^{n+1}_i / M^{n+1}_i,
    // with S^max/min_i = M^{n+1}_i max/min_{j in N_i} c^L_j. Dirichlet nodes keep c_D (R^+- = 1 there).
    // limitedFluxMatrix_ij = alpha_ij f_ij (row i) for the solute budget.
    void FCTStep_compatible(arguments_dict& args)
    {
      const double dt = args.scalar<double>("dt");
      const int numDOFs = args.scalar<int>("numDOFs");
      proteus::pyarray<int>& csrRowIndeces_DofLoops = args.array<int>("csrRowIndeces_DofLoops");
      proteus::pyarray<int>& csrColumnOffsets_DofLoops = args.array<int>("csrColumnOffsets_DofLoops");
      proteus::pyarray<double>& ML = args.array<double>("ML");
      proteus::pyarray<double>& MC = args.array<double>("MC");
      proteus::pyarray<double>& Mn = args.array<double>("Mn");
      proteus::pyarray<double>& Mnp1 = args.array<double>("Mnp1");
      proteus::pyarray<double>& cn = args.array<double>("cn");
      proteus::pyarray<double>& cL = args.array<double>("cL");
      proteus::pyarray<double>& Fhat = args.array<double>("Fhat");
      proteus::pyarray<double>& dDispersion = args.array<double>("dDispersion");
      proteus::pyarray<double>& bc_mask = args.array<double>("bc_mask");
      proteus::pyarray<double>& Rpos = args.array<double>("Rpos");
      proteus::pyarray<double>& Rneg = args.array<double>("Rneg");
      proteus::pyarray<double>& FluxCorrectionMatrix = args.array<double>("FluxCorrectionMatrix");
      proteus::pyarray<double>& limitedFluxMatrix = args.array<double>("limitedFluxMatrix");
      proteus::pyarray<double>& c_new = args.array<double>("c_new");
      std::vector<double> SL(numDOFs), SdotL(numDOFs);
      for (int i=0; i<numDOFs; i++)
        {
          SL[i] = Mnp1.data()[i]*cL.data()[i];
          SdotL[i] = (SL[i] - Mn.data()[i]*cn.data()[i])/dt;
        }
      for (int i=0; i<numDOFs; i++)
        {
          double cmax = cL.data()[i], cmin = cL.data()[i], Ppos = 0.0, Pneg = 0.0;
          for (int offset=csrRowIndeces_DofLoops.data()[i]; offset<csrRowIndeces_DofLoops.data()[i+1]; offset++)
            {
              const int j = csrColumnOffsets_DofLoops.data()[offset];
              cmax = fmax(cmax, cL.data()[j]);
              cmin = fmin(cmin, cL.data()[j]);
              double f = 0.0;
              if (j != i)
                {
                  const double nu = fmax(0.0, 0.5*fabs(Fhat.data()[offset]) + dDispersion.data()[offset]);
                  f = dt*MC.data()[offset]*(SdotL[i] - SdotL[j]) - dt*nu*(cn.data()[j] - cn.data()[i]);
                }
              FluxCorrectionMatrix.data()[offset] = f;
              Ppos += fmax(f, 0.0);
              Pneg += fmin(f, 0.0);
            }
          if (bc_mask.data()[i] == 0.0)
            {
              Rpos.data()[i] = 1.0;
              Rneg.data()[i] = 1.0;
              continue;
            }
          const double Qpos = ML.data()[i]*(Mnp1.data()[i]*cmax - SL[i]);
          const double Qneg = ML.data()[i]*(Mnp1.data()[i]*cmin - SL[i]);
          Rpos.data()[i] = (Ppos == 0.0) ? 1.0 : fmax(0.0, fmin(1.0, Qpos/Ppos));
          Rneg.data()[i] = (Pneg == 0.0) ? 1.0 : fmax(0.0, fmin(1.0, Qneg/Pneg));
        }
      for (int i=0; i<numDOFs; i++)
        {
          double sum = 0.0;
          for (int offset=csrRowIndeces_DofLoops.data()[i]; offset<csrRowIndeces_DofLoops.data()[i+1]; offset++)
            {
              const int j = csrColumnOffsets_DofLoops.data()[offset];
              const double f = FluxCorrectionMatrix.data()[offset];
              double alpha = (f > 0.0) ? fmin(Rpos.data()[i], Rneg.data()[j]) : fmin(Rneg.data()[i], Rpos.data()[j]);
              if (bc_mask.data()[i] == 0.0 && bc_mask.data()[j] == 0.0)
                alpha = 0.0;
              limitedFluxMatrix.data()[offset] = alpha*f;
              sum += alpha*f;
            }
          if (bc_mask.data()[i] == 0.0)
            c_new.data()[i] = cL.data()[i];
          else
            c_new.data()[i] = (SL[i] + sum/ML.data()[i])/Mnp1.data()[i];
        }
    }
  };//TADR

inline TADR_base* newTADR(int nSpaceIn,
                          int nQuadraturePoints_elementIn,
                          int nDOF_mesh_trial_elementIn,
                          int nDOF_trial_elementIn,
                          int nDOF_test_elementIn,
                          int nQuadraturePoints_elementBoundaryIn,
                          int CompKernelFlag)
{
  if (nSpaceIn == 1)
    return proteus::chooseAndAllocateDiscretization1D<TADR_base,TADR,CompKernel>(nSpaceIn,
                                                                                 nQuadraturePoints_elementIn,
                                                                                 nDOF_mesh_trial_elementIn,
                                                                                 nDOF_trial_elementIn,
                                                                                 nDOF_test_elementIn,
                                                                                 nQuadraturePoints_elementBoundaryIn,
                                                                                 CompKernelFlag);
  else if (nSpaceIn == 2)
    return proteus::chooseAndAllocateDiscretization2D<TADR_base,TADR,CompKernel>(nSpaceIn,
                                                                                 nQuadraturePoints_elementIn,
                                                                                 nDOF_mesh_trial_elementIn,
                                                                                 nDOF_trial_elementIn,
                                                                                 nDOF_test_elementIn,
                                                                                 nQuadraturePoints_elementBoundaryIn,
                                                                                 CompKernelFlag);
  else
    return proteus::chooseAndAllocateDiscretization<TADR_base,TADR,CompKernel>(nSpaceIn,
                                                                               nQuadraturePoints_elementIn,
                                                                               nDOF_mesh_trial_elementIn,
                                                                               nDOF_trial_elementIn,
                                                                               nDOF_test_elementIn,
                                                                               nQuadraturePoints_elementBoundaryIn,
                                                                               CompKernelFlag);
}
}//proteus
#endif
