#ifndef TADR_H
#define TADR_H
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <valarray>
#include <vector>
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
    // USE_EDGE_VELOCITY: A_ij = int w_j (rho v).grad(w_i) and its transpose A_ji,
    // assembled from the flow model's edge velocity; edgeFlux = A - A^T
    std::valarray<double> AdvectionMatrix, TransposeAdvectionMatrix;
    std::valarray<double> psi, eta, global_entropy_residual, boundary_integral;
    std::valarray<double> m_dof, theta_dof_proj, rho_dof_proj, ML_mass_proj;
    // SCHEME B: boundary flux of the FLUID mass, the counterpart of
    // boundary_integral (which carries the solute).  Advective only -- physical
    // dispersion moves solute relative to the fluid, it does not move fluid.
    std::valarray<double> boundary_integral_fluid;
    // FLOW-MODEL FLUXES: per node, the boundary concentration that inflowing
    // water carries -- the dS-weighted average of this model's Dirichlet value
    // over the node's Dirichlet faces (cin_den = 0: no Dirichlet face, the water
    // enters with the node's own c).
    std::valarray<double> cin_num, cin_den;
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

    
  // RECORDED BOUNDARY FLUX with the flow model's fluxes (USE_EDGE_FLUX).  The
  // edge loop applies the solute's advective boundary exchange at the nodes:
  // B_i c_i + max(B_i, 0)(c_in_i - c_i) into node i (the conservative form of the
  // cancelled row's max(B_i,0)(c_in - c_i)), B_i = nodalBoundaryFlux.  ebqe_flux
  // recorded the trace flux n.f(c_h) with this model's own velocity instead
  // (levee deck, ImplicitEV, cin = 0.7: -2.6 against an applied -68 at the first
  // step).  The nodal exchange a_i (outward) of each boundary node is spread
  // over its boundary faces by the lumped boundary mass mb_i = int_dOmega w_i,
  //   flux_adv(x) = sum_i w_i(x) a_i / mb_i,
  // so int flux_adv dS = sum_i a_i over the nodes with boundary faces.  B_i at an
  // interior node (the flow model's Newton residual there) has no boundary
  // face and is not recorded.  ebqe_flux already holds the applied dispersive
  // flux.  c is the concentration the edge loop used (c^{n+1} for ImplicitEV,
  // c^n for the explicit branches).
  inline void recordEdgeFluxBoundaryExchange(int nExteriorElementBoundaries_global,
                                             int numDOFs,
                                             const int* exteriorElementBoundariesArray,
                                             const int* elementBoundaryElementsArray,
                                             const int* elementBoundaryLocalElementBoundariesArray,
                                             const int* elementBoundaryMaterialTypes,
                                             const int* isExteriorBoundaryPhysical,
                                             const int* u_l2g,
                                             const double* u_test_trace_ref,
                                             const double* ebqe_dS_ext,
                                             const double* nodalBoundaryFlux,
                                             const double* c,
                                             double* ebqe_flux,
                                             double* ebqe_flux_adv)
  {
    std::valarray<double> mb(0.0, numDOFs), a(0.0, numDOFs);
    for (int pass=0; pass<2; pass++)
      {
        for (int ebNE=0; ebNE<nExteriorElementBoundaries_global; ebNE++)
          {
            const int ebN = exteriorElementBoundariesArray[ebNE],
              eN = elementBoundaryElementsArray[ebN*2+0],
              ebN_local = elementBoundaryLocalElementBoundariesArray[ebN*2+0];
            // the faces the exterior loop integrates
            if (elementBoundaryMaterialTypes[ebN] <= 0 || isExteriorBoundaryPhysical[ebNE] == 0 ||
                elementBoundaryElementsArray[ebN*2+1] >= 0)
              continue;
            for (int kb=0; kb<nQuadraturePoints_elementBoundary; kb++)
              {
                const int ebNE_kb = ebNE*nQuadraturePoints_elementBoundary+kb,
                  ebN_local_kb = ebN_local*nQuadraturePoints_elementBoundary+kb;
                double adv = 0.0;
                for (int i=0; i<nDOF_test_element; i++)
                  {
                    const int gi = u_l2g[eN*nDOF_test_element+i];
                    const double w = u_test_trace_ref[ebN_local_kb*nDOF_test_element+i];
                    if (pass == 0)
                      mb[gi] += w*ebqe_dS_ext[ebNE_kb];
                    else
                      adv += w*a[gi];
                  }
                if (pass == 1)
                  {
                    ebqe_flux_adv[ebNE_kb] = adv;
                    ebqe_flux[ebNE_kb] += adv;
                  }
              }
          }
        if (pass == 0)
          for (int i=0; i<numDOFs; i++)
            if (mb[i] > 0.0)
              {
                const double B_i = nodalBoundaryFlux[i];
                const double c_in = (cin_den[i] > 0.0) ? cin_num[i]/cin_den[i] : c[i];
                a[i] = -(B_i*c[i] + fmax(B_i, 0.0)*(c_in - c[i]))/mb[i];
              }
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
      // OUT: viscosity ratio mu/mu_0 at the quadrature points, for the flow model,
      // exported the same way as q_rho.  omega_max maps u to the solute mass
      // fraction omega = omega_max*u; 0 gives mu/mu_0 = 1 exactly.
      proteus::pyarray<double>& q_mu = args.array<double>("q_mu");
      const double omega_max = args.scalar<double>("omega_max");

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
      proteus::pyarray<double>& ebqe_mu = args.array<double>("ebqe_mu");

      double epsFact = args.scalar<double>("epsFact");
      proteus::pyarray<double>& ebqe_u = args.array<double>("ebqe_u");
      proteus::pyarray<double>& ebqe_flux = args.array<double>("ebqe_flux");
      // BOUNDARY-FLUX DIAGNOSTIC.  ebqe_flux is the SUM of the advective and
      // dispersive solute fluxes, which is the one number that cannot answer
      // "does solute reach this part of the boundary by being carried there, or
      // only by spreading?".  Splitting them is what distinguishes a stagnant
      // Dirichlet strip (advection ~ 0, dispersion only) from a live one.
      // ebqe_dS is the quadrature weight, so Python can integrate rather than
      // sum bare quadrature values.  See TADR_BC_DBG in TADR.py.
      proteus::pyarray<double>& ebqe_flux_adv = args.array<double>("ebqe_flux_adv");
      proteus::pyarray<double>& ebqe_dS_ext = args.array<double>("ebqe_dS_ext");
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
      //////////////////////////////////////////////////////////////////////
      // SCHEME B -- the concentration is the RATIO of two masses.         //
      //                                                                   //
      // M_dof[i] = (theta*rho)_i is the FLUID mass and m_dof[i]           //
      // = M_dof[i]*u_i the SOLUTE mass, and u = m/M.  The identity        //
      //   m^L_i - u^n_i M^L_i = dt/ML_i sum_{j!=i} gamma_ij (M_j/theta_j) //
      //                                            * (u_j - u_i)          //
      // (gamma_ij = d_ij - K_ij >= 0) makes u^L a convex combination of   //
      // {u^n_i, u^n_j} for ANY velocity, divergent or not.                //
      //                                                                   //
      // Where M comes from differs by branch, and so does what is         //
      // conserved:                                                        //
      //  * EXPLICIT edge-based (STAB 2/3/4): M is CARRIED here and        //
      //    advected with EXACTLY the same operator (same K, same graph    //
      //    viscosity d) as the solute mass.  sum_i ML_i m_i is then       //
      //    conserved to round-off, because the operator itself is         //
      //    untouched (column sums of K vanish, d is symmetric).           //
      //  * ImplicitEV (STAB 5), OPTION C: M is READ FROM RICHARDS every   //
      //    step.  Boundedness is unchanged (any positive M gives an       //
      //    M-matrix), but conservation now holds only up to the           //
      //    flow/transport continuity defect -- see the option C note in   //
      //    the ImplicitEV branch, and the [TADR-MASS] log.                //
      //                                                                   //
      // This replaces the row-sum premise entirely.  The previous form    //
      //   sum_j (T+D)_ij w_j - sum_j d_ij (w_j - w_i)                     //
      // is a convex combination only when sum_j (T+D)_ij = 0, i.e. only   //
      // for a discretely divergence-free velocity -- which Elder's        //
      // velocity_couple is NOT, and physically cannot be: with a          //
      // solution-dependent density the fluid mass equation gives          //
      //   div v = -(1/rho)[ d_t(theta rho) + v.grad rho ] != 0.           //
      // Measured in the numpy replica (1D periodic, v = v0(1+A sin 2pi x),//
      // so every defect is interior): at A=1 the old form reaches         //
      // max u = 7.33 from a u <= 1 IC, this one stays in [0, 0.64] with a //
      // solute-mass drift of 6.7e-16.                                     //
      //                                                                   //
      // KNOWN SECOND-ORDER MODELLING ERROR: the dispersion stiffness sits //
      // inside TransportMatrix (see the element assembly), so the fluid   //
      // mass picks up a spurious dispersion of size eps_rho*D.  Physically//
      // dispersion moves solute relative to the fluid and should not move //
      // the fluid at all.  Splitting the dispersion into its own matrix   //
      // acting on u (rather than on w) is the clean fix; it is a separate //
      // restructuring of the element loop and does NOT affect boundedness //
      // or conservation, both of which hold for the combined operator     //
      // because the SAME K is applied to m and to M.                      //
      //////////////////////////////////////////////////////////////////////
      proteus::pyarray<double>& M_dof = args.array<double>("M_dof");
      proteus::pyarray<double>& M_low = args.array<double>("M_low");
      // CONTINUITY-DEFECT PROBE (option C only): the fluid flux out of node i under
      // TADR's OWN operator, which the cancelled solute form assumes balances the
      // fluid storage.  Python pairs it with Richards' M^{n+1} to form the defect.
      proteus::pyarray<double>& fluidResidual = args.array<double>("fluidResidual");
      // OUT (ImplicitEV only): rho(c_i) at the current iterate, for the flow
      // model's nodal density.  invert() overwrites it with rho of the limited c.
      proteus::pyarray<double>& rho_nodal = args.array<double>("rho_nodal");
      // FLOW-MODEL FLUXES (ImplicitEV, USE_EDGE_FLUX=1): the water the flow model
      // actually moved in this step, as rates.  edgeFlux[ij] is the mass flux into
      // row i from column j (antisymmetric), nodalBoundaryFlux[i] the rest of
      // node i's storage change (boundary exchange + the flow model's Newton
      // residual).  They REPLACE this model's own advection operator, so the fluid
      // balance the cancelled form assumes is the flow model's, exactly.
      proteus::pyarray<double>& edgeFlux = args.array<double>("edgeFlux");
      proteus::pyarray<double>& nodalBoundaryFlux = args.array<double>("nodalBoundaryFlux");
      const int USE_EDGE_FLUX = args.scalar<int>("USE_EDGE_FLUX");
      // EDGE VELOCITY (USE_EDGE_FLUX=1 only): edgeFlux is not read but WRITTEN here,
      // F_ij = A_ij - A_ji with A_ij = int w_j (rho v).grad(w_i), v the flow model's
      // edge velocity (q_v) and rho = q_rho_flow, the density that velocity was
      // divided by.  sum_j A_ji = 0, so sum_j F_ij = int rho v.grad(w_i), which the
      // flow model built to equal its realized edge-flux row sum.  The symmetric
      // part of A is fluid storage, supplied by the carried M instead.
      // nodalBoundaryFlux is written too: B_i = ML_i (M_low_i - M_dof_i)/dt - sum_j F_ij.
      const int USE_EDGE_VELOCITY = args.scalar<int>("USE_EDGE_VELOCITY");
      proteus::pyarray<double>& q_rho_flow = args.array<double>("q_rho_flow");
      // IN (USE_EDGE_FLUX only): nodal source of the CANCELLED form [mass/(volume
      // time)], added to row i as ML_i*nodalSource_i (lumped).  For a conservative
      // solute source f_s this is f_s - c*f_w: the flow model's water source f_w
      // reaches this model through B_i with the node's own c and cancels.  Zero
      // unless the deck passes Coefficients(mms_source=...).
      proteus::pyarray<double>& nodalSource = args.array<double>("nodalSource");
      // Richards' NODAL theta(psi^n).  Zero => no flow model.  Under ImplicitEV
      // it no longer recovers rho from M_dof: M_dof is Richards' old-time mass
      // theta(psi^n)*rho(c^{n-1}) (the mass its previous step produced), not
      // theta*rho(c^n).
      proteus::pyarray<double>& theta_nodal = args.array<double>("theta_nodal");
      const int init_M_dof = args.scalar<int>("init_M_dof");
      // OPTION C: for ImplicitEV, M_dof is Richards' old-time mass, refreshed on
      // the Python side before this kernel runs.  It is the old-time storage
      // coefficient ONLY; the flux density is rho(c^n) from the EOS at the node
      // (rho_dof_proj), the same nodal density Richards advects with.  There is
      // no separate M^{n+1} array and no fluid linear system to assemble.  M_low is written only by the
      // explicit edge-based branches, which still advect their own copy.
      //////////////////////////////////////////////////////////////////////
      // OPTION C -- why there is no fluid-mass Dirichlet set here.         //
      //                                                                    //
      // The fluid mass needs the SAME Dirichlet condition as the           //
      // concentration: rho is a function of u, so where u is strongly      //
      // imposed the density is imposed too and M = theta*rho(u_bc) is the  //
      // only value consistent with the row Python replaces.                //
      //                                                                    //
      // Leaving M free there is NOT a small error.  The solute flux        //
      // coefficient is gamma_ij*rho_j with rho_j = M_j/theta_j, so an M    //
      // that drifts at a Dirichlet node re-scales every edge leaving that  //
      // node.  When M was advected here as its own free unknown, the Elder //
      // source strip (u pinned to 1, v.n = 0, and the dispersion inside K  //
      // draining M toward the lighter interior with no boundary term to    //
      // hold it) saw M fall to ~4% of theta*rho: the first edge below the  //
      // strip ended up ~25x weaker than the interior edges, the profile    //
      // went 1.000 -> 0.299 -> 0.272 in one cell instead of                //
      // 1.000 -> 0.907 -> 0.836, and the whole benchmark took up 3x too    //
      // little salt (mean u 0.071 vs 0.211 at 20 yr) so convection never   //
      // developed.  Invisible in test_tadr (eps_rho = 0 => rho constant    //
      // => M cannot drift) and invisible to the u-bounds, which stayed in  //
      // [0,1] the whole time.                                              //
      //                                                                    //
      // Under option C that whole failure mode is gone by construction:    //
      // ImplicitEV reads M from Richards, which imposes its own flow BCs   //
      // at that strip and whose density is the q_rho this model pushes to  //
      // it -- so M there IS theta*rho(u_bc) without anything pinning it.   //
      // The explicit branches still advect their own M, but they also      //
      // carry the matching fluid boundary flux (boundary_integral_fluid),  //
      // which is what holds it.                                            //
      //////////////////////////////////////////////////////////////////////
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
      
      // the entropy normalization is over the domain: accumulate it over owned elements and hand it out with each
      // element's viscosities, so that in parallel the caller can reduce it over the ranks and recompute q_numDiff_u
      int nElements_owned = args.scalar<int>("nElements_owned");
      proteus::pyarray<double>& entropy_stats = args.array<double>("entropy_stats");
      proteus::pyarray<double>& element_viscosity = args.array<double>("element_viscosity");
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
          AdvectionMatrix.resize(USE_EDGE_VELOCITY ? NNZ : 0, 0.0);
          TransposeAdvectionMatrix.resize(USE_EDGE_VELOCITY ? NNZ : 0, 0.0);
          m_dof.resize(numDOFs,0.0);
          theta_dof_proj.resize(numDOFs,0.0);
          rho_dof_proj.resize(numDOFs,0.0);
          ML_mass_proj.resize(numDOFs,0.0);
          // ONLY theta is projected from quadrature -- it is a material property
          // with no nodal value of its own.  rho and u are NOT projected: rho is
          // an explicit function of the nodal u (see the rho_dof_proj assignment
          // below), and projecting u would apply the (M*u)_i/ML_i smoothing
          // filter.
          //
          // rho MUST be evaluated at the DOF, not projected.  The edge-based
          // scheme recovers the concentration from the stored mass with
          //   ui_mass = inversevaluateCoefficients(m_dof[i], theta_i)
          // which returns u_dof_old[i] IF AND ONLY IF
          //   rho_dof_proj[i] == rho_f*(1 + eps_rho*u_dof_old[i])
          // exactly.  With a projected (smoothed) rho that identity fails by
          // O(eps_rho * (rho~ - rho)), so ui_mass != u_dof_old[i] and the whole
          // edge-based pipeline runs on a field that differs from the one FCT
          // takes its bounds from (FCTStep reads the true nodal soln=u_dof_old).
          // Zalesak's precondition m(mini) <= mLow_i <= m(maxi) is then not
          // guaranteed, and when it fails Rpos clamps to 0 -- which returns
          // mLow unchanged instead of enforcing the bound.
          //
          // It also destroys the steady state: with velocity identically zero
          // the projected form is NOT a fixed point (numpy replica: 3.1% drift
          // after one step, 23% after ten).  The nodal form is exact.
          for (int eN=0; eN<nElements_global; eN++)
            for (int k=0; k<nQuadraturePoints_element; k++)
              {
                int eN_k = eN*nQuadraturePoints_element + k,
                  eN_nDOF_trial_element = eN*nDOF_trial_element;
                double jac[nSpace*nSpace], jacDet, jacInv[nSpace*nSpace], x, y, z, un_k=0.0;
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
                (void) un_k;
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
              // PREFER THE FLOW MODEL'S NODAL THETA.  M_dof_i = theta_i*rho_i was
              // formed at a DOF, with THAT DOF's soil (freeDOFMaterialTypes); the
              // quadrature theta above is evaluated per ELEMENT, with the element's
              // soil, so at a material interface the projection averages two soils
              // and M_dof/theta is a density Richards never used.  (With the vertex
              // rule the two agree everywhere else, so this only bites interfaces.)
              // Zero = no flow model: keep the projection.
              if (theta_nodal.data()[i] > 0.0)
                theta_dof_proj[i] = theta_nodal.data()[i];
              // conservative variable at t^n; u AND rho stay nodal, so that
              // inversevaluateCoefficients(m_dof[i], theta_dof_proj[i]) returns
              // u_dof_old[i] to round-off (see the note above).
              const double un_i = u_dof_old.data()[i];
              // The fluid mass M = theta*rho is NOT re-derived from the EOS
              // here.  init_M_dof=1 only on the first residual call of a deck
              // with no flow model to read M from (test_tadr), where it is the
              // one place rho(u) is used for it.  Re-seeding M := theta*rho(u)
              // every step would put the continuity defect back in as a
              // per-step projection -- measured at -0.98 relative solute mass
              // over 400 steps in the replica, and -1.1e-2 even at div v == 0.
              // With a flow model, Python has already put Richards' theta*rho
              // (ImplicitEV) or the advected value (explicit branches) in
              // M_dof before this loop runs.
              if (init_M_dof)
                M_dof.data()[i] = theta_dof_proj[i]*rho_f*(1.0 + eps_rho*un_i);
              const double Mi = fmax(M_dof.data()[i], 1.0e-14);
              // ImplicitEV: rho(c^n) from the EOS at the DOF -- the value Richards'
              // nodal density is built from (Richards.Coefficients.preStep), so both
              // models advect with one density.  M_dof is storage only here.
              rho_dof_proj[i] = (STABILIZATION_TYPE == STABILIZATION::ImplicitEV)
                ? rho_f*(1.0 + eps_rho*un_i)
                : Mi/theta_dof_proj[i];
              m_dof[i] = Mi*un_i;
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
              // rho = M/theta was formed above from the rank-local projection, which is partial at a ghost
              // dof; form it again from the owners' theta (ImplicitEV takes rho from the EOS and is unaffected)
              if (STABILIZATION_TYPE != STABILIZATION::ImplicitEV)
                rho_dof_proj[i] = fmax(M_dof.data()[i], 1.0e-14)/theta_dof_proj[i];
              // the conservative variable at t^n is the CARRIED storage times u^n, as formed
              // before the phase split.  theta*rho equals it for the explicit branches
              // (rho = M/theta), but not for ImplicitEV, whose rho is the EOS rho(u^n) while M
              // carries the flow model's rho(u^{n-1})*theta: ImplicitEV recovers u^n as
              // m_dof/M, which theta*rho(u^n)*u^n inflates by rho(u^n)/rho(u^{n-1}) wherever
              // the density just rose (UHD level 0: u up to 1.042).
              m_dof[i] = fmax(M_dof.data()[i], 1.0e-14)*u_dof_old.data()[i];
            }
          // compute entropy and init global_entropy_residual and boundary_integral
          psi.resize(numDOFs,0.0);
          eta.resize(numDOFs,0.0);
          global_entropy_residual.resize(numDOFs,0.0);
          boundary_integral.resize(numDOFs,0.0);
          boundary_integral_fluid.resize(numDOFs,0.0);
          boundary_integral_fluid = 0.0;
          cin_num.resize(numDOFs,0.0);
          cin_num = 0.0;
          cin_den.resize(numDOFs,0.0);
          cin_den = 0.0;
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
          double  elementAdvection[nDOF_test_element][nDOF_trial_element];
          double  elementTransposeAdvection[nDOF_test_element][nDOF_trial_element];
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
                      elementAdvection[i][j]=0.0;
                      elementTransposeAdvection[i][j]=0.0;
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
              // mu/mu_0 = 1 + 1.85 w - 4.1 w^2 + 44.5 w^3, Eq. (9) of Guevara
              // Morel and Graf (2023) after Holzbecher (1998), w = omega_max*u
              const double omega_k = omega_max*u;
              q_mu.data()[eN_k] = 1.0 + omega_k*(1.85 + omega_k*(-4.1 + omega_k*44.5));

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
                  if (eN < nElements_owned)
                    {
                      meanEntropy += EPOWER(u,0,1)*dV;
                      meanOmega += dV;
                      maxEntropy = fmax(maxEntropy,EPOWER(u,0,1));
                      minEntropy = fmin(minEntropy,EPOWER(u,0,1));
                    }

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
                      //////////////////////////////////////////////////////////
                      // GROUP VARIABLE w = rho(u)*u.                          //
                      //                                                       //
                      // The edge-based low-order operator is monotone only if //
                      // the transport matrix has ZERO ROW SUMS, because the   //
                      // DMP argument (see the comment at the dissipative-term //
                      // accumulation below) rewrites                          //
                      //   sum_j T_ij c_j = sum_{j!=i} T_ij (c_j - c_i)        //
                      // which needs sum_j T_ij = 0.  Since                    //
                      //   T_ij = int (df/dvar) phi_j dw_i,                    //
                      // the row sum is int (df/dvar) dw_i, and that vanishes  //
                      // by the partition of unity ONLY when df/dvar is        //
                      // constant over the element.                            //
                      //                                                       //
                      // With the variable-density flux f = rho(u)*v*u we have //
                      //   df/du = (rho + u drho/du)*v,                        //
                      // which DEPENDS ON u -- so assembling T against u leaves //
                      // a residue (sum_j T_ij)*c_i in the low-order update.    //
                      // That residue is an unbounded source, and it is largest //
                      // exactly at the front the limiter has to protect        //
                      // (numpy replica: +4.8% overshoot at eps_rho = 0.235,    //
                      // growing monotonically with eps_rho, exactly 0 at       //
                      // eps_rho = 0).                                          //
                      //                                                       //
                      // The cure is to transport the GROUP VARIABLE            //
                      //   w = rho(u)*u,     so that   f = v*w                  //
                      // is LINEAR in w with df/dw = v, the Darcy velocity,     //
                      // which does not depend on the solution.  Row sums are   //
                      // then exactly zero and Kuzmin's theory applies verbatim.//
                      // Storage becomes m = theta*w, so the update base point  //
                      // and the differencing variable are the SAME variable    //
                      // and the low-order update is a true convex combination. //
                      // w(u) = rho_f*(u + eps_rho*u^2) is strictly increasing  //
                      // for u >= 0, so bounds map through unchanged:           //
                      //   w in [w(min u), w(max u)]  <=>  u in [min u, max u]. //
                      //                                                       //
                      // Dispersion is rescaled to match: the physical flux is  //
                      // -a grad(u) = -(a/(dw/du)) grad(w), so the tensor seen  //
                      // by the w-operator is a_w = a/(dw/du).  a_w is still a  //
                      // positive multiple of the dispersion tensor, and the    //
                      // diffusion operator has zero row sums for free          //
                      // (sum_j grad(phi_j) = 0), so it never broke the DMP.    //
                      //                                                       //
                      // NOTE this changes only the EDGE-BASED assembly.  The   //
                      // cell-based branches (Galerkin, VMS, TaylorGalerkinEV)  //
                      // keep df/du, which is what they need.                   //
                      //                                                       //
                      // BEHAVIOUR CHANGE beyond the group variable itself: the //
                      // explicit branches now take the dispersion tensor from  //
                      // `an` (old time) instead of `a` (current iterate), so   //
                      // it matches the time level of both the advection        //
                      // coefficient (velocity_old) and the dwdu used to        //
                      // rescale it.  Mixing a current-time `a` with an         //
                      // old-time dwdu would give a rescaling that corresponds  //
                      // to no consistent dispersion at all.                    //
                      //////////////////////////////////////////////////////////
                      const bool implicit_ev =
                        (STABILIZATION_TYPE==STABILIZATION::ImplicitEV);
                      // dw/du = rho + u*drho/du, at the matching time level
                      const double drho_du_k = rho_f*eps_rho;
                      const double dwdu_k = implicit_ev
                        ? fmax(rho_out     + u *drho_du_k, 1.0e-14)
                        : fmax(rho_out_old + un*drho_du_k, 1.0e-14);
                      // df/dw = v, the Darcy velocity at the matching level
                      const double* adv_df = implicit_ev
                        ? &velocity.data()[eN_k_nSpace]
                        : &velocity_old.data()[eN_k_nSpace];
                      double a_w[nnz];
                      for (int ii=0; ii<a_rowptr.data()[nSpace]; ii++)
                        a_w[ii] = (implicit_ev ? a[ii] : an[ii])/dwdu_k;
                      // FLOW-MODEL FLUXES: advection comes from the flow model's
                      // edge fluxes in the edge loop, so the element matrix is
                      // the DISPERSION alone, acting on c with a = theta*rho*D
                      // (no w, no dw/du rescaling) -- and it never touches the
                      // fluid, which only the flow model's fluxes move.  Every
                      // edge-based branch; the explicit ones take the tensor at
                      // the old time, the level their fluxes are evaluated at.
                      const bool edge_flux = USE_EDGE_FLUX;
                      for(int j=0;j<nDOF_trial_element;j++)
                        {
                          int j_nSpace = j*nSpace;
                          int i_nSpace = i*nSpace;
                          if (edge_flux)
                            {
                              const double Dij = ck.SimpleDiffusionJacobian_weak(a_rowptr.data(),
                                                                                 a_colind.data(),
                                                                                 implicit_ev ? a : an,
                                                                                 &u_grad_trial[j_nSpace],
                                                                                 &u_grad_test_dV[i_nSpace]);
                              elementTransport[i][j] += Dij;
                              elementTransposeTransport[i][j] += Dij;
                              elementDiffusion[i][j] += ck.NumericalDiffusionJacobian(physicalDiffusion/dwdu_k,
                                                                                      &u_grad_trial[j_nSpace],
                                                                                      &u_grad_test_dV[i_nSpace]);
                              if (USE_EDGE_VELOCITY)
                                {
                                  // mass flux of the flow model's edge velocity at
                                  // this quadrature point; AdvectionJacobian_weak
                                  // returns -w (q.grad(w_test)) dV, hence the -=.
                                  // THIS step's velocity for every branch: it is
                                  // rebuilt from the step's own flow fluxes, and
                                  // velocity_old is the previous step's.
                                  double q_flow[nSpace];
                                  for (int I=0;I<nSpace;I++)
                                    q_flow[I] = q_rho_flow.data()[eN_k]*velocity.data()[eN_k_nSpace+I];
                                  elementAdvection[i][j] -=
                                    ck.AdvectionJacobian_weak(q_flow,
                                                              u_trial_ref.data()[k*nDOF_trial_element+j],
                                                              &u_grad_test_dV[i_nSpace]);
                                  elementTransposeAdvection[i][j] -=
                                    ck.AdvectionJacobian_weak(q_flow,
                                                              u_trial_ref.data()[k*nDOF_trial_element+i],
                                                              &u_grad_test_dV[j_nSpace]);
                                }
                              continue;
                            }
                          elementTransport[i][j] +=
                            ck.AdvectionJacobian_weak(adv_df,
                                                      u_trial_ref.data()[k*nDOF_trial_element+j],
                                                      &u_grad_test_dV[i_nSpace])
                                                      +
                            ck.SimpleDiffusionJacobian_weak(a_rowptr.data(),
										                                        a_colind.data(),
                                                            a_w,
                                                            &u_grad_trial[j_nSpace],
                                                            &u_grad_test_dV[i_nSpace]);




                           elementDiffusion[i][j] += ck.NumericalDiffusionJacobian(physicalDiffusion/dwdu_k,
                                                                                   &u_grad_trial[j_nSpace],
                                                                                   &u_grad_test_dV[i_nSpace]);
                          elementTransposeTransport[i][j] += ck.AdvectionJacobian_weak(adv_df,
                                                                                       u_trial_ref.data()[k*nDOF_trial_element+i],
                                                                                       &u_grad_test_dV[j_nSpace])+
                                                             ck.SimpleDiffusionJacobian_weak(a_rowptr.data(),
                                                                                              a_colind.data(),
                                                                                              a_w,
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
                      if (USE_EDGE_VELOCITY)
                        {
                          AdvectionMatrix[csrRowIndeces_CellLoops.data()[eN_i] +
                                          csrColumnOffsets_CellLoops.data()[eN_i_j]] += elementAdvection[i][j];
                          TransposeAdvectionMatrix[csrRowIndeces_CellLoops.data()[eN_i] +
                                                   csrColumnOffsets_CellLoops.data()[eN_i_j]] += elementTransposeAdvection[i][j];
                        }
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
            elementResidual_fluid[nDOF_test_element],   // SCHEME B: fluid mass
            fluxTransport[nDOF_test_element][nDOF_trial_element];
          for (int i=0;i<nDOF_test_element;i++)
            {
              elementResidual_u[i]=0.0;
              elementResidual_fluid[i]=0.0;
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
              // Eq. (9), as for q_mu
              const double omega_ext = omega_max*u_ext;
              ebqe_mu.data()[ebNE_kb] = 1.0 + omega_ext*(1.85 + omega_ext*(-4.1 + omega_ext*44.5));
              
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

              //////////////////////////////////////////////////////////////
              // GROUP-VARIABLE traces, for the edge-based and ImplicitEV  //
              // branches.  The INTERIOR assembly builds T against         //
              // w = rho(u)*u (df/dw = v), so every boundary quantity that //
              // ends up multiplying w -- the flux Jacobian rows that go   //
              // into TransportMatrix, the Nitsche adjoint, the upwind     //
              // penalty -- has to be in w as well.  Leaving them in u was //
              // what broke the STAB=2 DMP: the boundary rows of T were    //
              // u-scaled while the interior was w-scaled, so sum_j T_ij   //
              // stopped vanishing exactly at boundary-adjacent nodes, and //
              // uLow overshot in the row below the Dirichlet source strip //
              // (max u = 6.58 on Elder).                                  //
              //                                                           //
              // The RESIDUAL flux_ext needs no conversion: f = rho v u    //
              // = v*w is already the physical flux, and the diffusive     //
              // flux -a grad(u) = -a_w grad(w) is the same number either  //
              // way.  Only the linearised/penalty pieces are rescaled.    //
              //////////////////////////////////////////////////////////////
              const double drho_du_b  = rho_f*eps_rho;
              const double dwdu_ext   = fmax(rho_out_ext + u_ext*drho_du_b, 1.0e-14);
              const double w_ext      = rho_out_ext*u_ext;
              const double w_bc_ext   = rho_out_bc*bc_u_ext;
              // dispersion w.r.t. w:  -a grad(u) = -(a/(dw/du)) grad(w)
              double a_w_ext[nnz], zero_da_ext[nnz];
              for (int ii=0; ii<a_rowptr.data()[nSpace]; ii++)
                {
                  a_w_ext[ii]    = a_ext[ii]/dwdu_ext;
                  // a_w is treated as frozen when differentiating w.r.t. w
                  // (the explicit branch stores a fixed matrix anyway), so the
                  // da*v*grad_psi term of the diffusive Jacobian is dropped.
                  zero_da_ext[ii] = 0.0;
                }
              // df/dw = v, the Darcy velocity trace (no density factor)
              const double* v_w_ext = &ebqe_velocity_ext.data()[ebNE_kb_nSpace];
              double boundary_flow_w = 0.0;
              for (int I=0; I<nSpace; I++)
                boundary_flow_w += normal[I]*v_w_ext[I];

              //////////////////////////////////////////////////////////////
              // SCHEME B: advective boundary flux of the FLUID mass,      //
              //   F^f = (v.n) * rho_up,                                   //
              // upwinded exactly as the solute flux is.  The interior     //
              // trace of rho comes from the CARRIED fluid mass M/theta    //
              // (NOT from the EOS) so the boundary sees the same fluid    //
              // mass the interior transports; on inflow there is no       //
              // carried M outside the domain, so the EOS value at the BC  //
              // is the only choice.                                       //
              //                                                           //
              // Why this keeps u bounded at the boundary: the identity    //
              // subtracts u_i times the fluid flux from the solute flux,  //
              //   F^s - u_i F^f = (v.n) rho_up (u_up - u_i),              //
              // which on INFLOW ((v.n) < 0) is +|v.n| rho_bc (u_bc - u_i),//
              // a convex pull toward u_bc, and on OUTFLOW is ~0 because   //
              // u_up is the node's own trace.  min_u_bc/max_u_bc already  //
              // include u_bc in the FCT bounds, so the two agree.         //
              //                                                           //
              // NO diffusive counterpart: physical dispersion moves solute//
              // relative to the fluid, it does not move fluid.  This      //
              // leaves the weak-Dirichlet Nitsche term in the solute      //
              // equation without a fluid partner -- pre-existing weak-BC  //
              // behaviour, not a regression, and identically zero on a    //
              // closed (v.n = 0) boundary.                                //
              //////////////////////////////////////////////////////////////
              // GATED: theta_dof_proj / m_dof are only RESIZED for the branches
              // that carry a fluid mass (see the projection block above); the
              // cell-based branches leave them at size 0, so touching them here
              // unconditionally reads out of bounds.
              double flux_fluid_ext = 0.0;
              if (STABILIZATION_TYPE == STABILIZATION::EntropyViscosity or
                  STABILIZATION_TYPE == STABILIZATION::SmoothnessIndicator or
                  STABILIZATION_TYPE == STABILIZATION::Kuzmin or
                  STABILIZATION_TYPE == STABILIZATION::ImplicitEV)
                {
                  double rho_trace_ext = 0.0;
                  for (int j=0;j<nDOF_trial_element;j++)
                    {
                      const int gj = u_l2g.data()[eN_nDOF_trial_element+j];
                      // ImplicitEV: the nodal EOS density, as in the interior
                      // edge loop (M_dof is storage only there).
                      rho_trace_ext += u_trial_trace_ref.data()[ebN_local_kb*nDOF_test_element+j]
                                     * ((STABILIZATION_TYPE == STABILIZATION::ImplicitEV)
                                        ? rho_dof_proj[gj]
                                        : (fmax(M_dof.data()[gj], 1.0e-14)
                                           / fmax(theta_dof_proj[gj], 1.0e-14)));
                    }
                  const double rho_bc_fluid = rho_f*(1.0 + eps_rho*bc_u_ext);
                  const double rho_up_fluid = (boundary_flow_w >= 0.0) ? rho_trace_ext
                                                                       : rho_bc_fluid;
                  flux_fluid_ext = boundary_flow_w*rho_up_fluid;
                }

              // ImplicitEV without the flow model's edge fluxes: on an upwinded
              // OUTFLOW face the residual applies c_i times the fluid flux to
              // row i (see the residual update below), not the trace flux.
              const bool outflow_upwind =
                STABILIZATION_TYPE == STABILIZATION::ImplicitEV && !USE_EDGE_FLUX &&
                boundary_flow_w >= 0.0 &&
                ((isDOFBoundary_u.data()[ebNE_kb] == 1 && forceStrongConditions == 0) ||
                 (isDOFBoundary_u.data()[ebNE_kb] != 1 && isFluxBoundary_u.data()[ebNE_kb] != 1));
              // Record the flux the residual applies.  On an upwinded outflow face
              // sum_i c_i F^f w_i = c_h F^f at each quadrature point, so c_h F^f is
              // that flux pointwise.  n.f(c_h) = (v.n) rho(c_h) c_h differs from it
              // through the density, which F^f freezes at rho(c^n).
              const double flux_adv_applied = outflow_upwind ? u_ext*flux_fluid_ext : flux_adv_ext;
              ebqe_flux.data()[ebNE_kb] = outflow_upwind ? flux_adv_applied + flux_diff_ext : flux_ext;
              // advective/dispersive split + quadrature weight, see the note at
              // the ebqe_flux_adv declaration
              ebqe_flux_adv.data()[ebNE_kb] = flux_adv_applied;
              ebqe_dS_ext.data()[ebNE_kb]   = dS;
              if (USE_EDGE_FLUX)
                {
                  // FLOW-MODEL FLUXES: the advective exchange the residual applies is
                  // the flow model's NODAL boundary flux (edge loop), not n.f(c_h), so
                  // it is recorded after the edge loop (recordEdgeFluxBoundaryExchange).
                  // Here only the dispersive flux the update below applies: on a weak
                  // Dirichlet face the lumped penalty max_a*penalty*(c_i - g) w_i,
                  // whose pointwise sum over the face's test functions is
                  // max_a*penalty*(c_h - g); elsewhere flux_diff_ext.
                  double flux_disp_applied = flux_diff_ext;
                  if (isDOFBoundary_u.data()[ebNE_kb] == 1 &&
                      isDiffusiveFluxBoundary_u.data()[ebNE_kb] == 0)
                    {
                      double max_a = 0.0, c_h = 0.0;
                      for (int m=0; m<a_rowptr.data()[nSpace]; m++)
                        max_a = fmax(max_a, a_ext[m]);
                      for (int j=0; j<nDOF_trial_element; j++)
                        {
                          const int gj = u_l2g.data()[eN_nDOF_trial_element+j];
                          c_h += u_trial_trace_ref.data()[ebN_local_kb*nDOF_test_element+j]
                            *((STABILIZATION_TYPE == STABILIZATION::ImplicitEV)
                              ? u_dof.data()[gj] : u_dof_old.data()[gj]);
                        }
                      flux_disp_applied = max_a*ebqe_penalty_ext.data()[ebNE_kb]*(c_h - bc_u_ext);
                    }
                  ebqe_flux.data()[ebNE_kb] = flux_disp_applied;
                  ebqe_flux_adv.data()[ebNE_kb] = 0.0;
                }
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
                  // SCHEME B: fluid-mass boundary flux, for every branch that
                  // transports a fluid mass (the edge-based ones + ImplicitEV).
                  // The cell-based branches (Galerkin/VMS/TaylorGalerkinEV) do
                  // not carry M and are untouched.
                  if (STABILIZATION_TYPE == STABILIZATION::EntropyViscosity or
                      STABILIZATION_TYPE == STABILIZATION::SmoothnessIndicator or
                      STABILIZATION_TYPE == STABILIZATION::Kuzmin or
                      STABILIZATION_TYPE == STABILIZATION::ImplicitEV)
                    elementResidual_fluid[i] +=
                      ck.ExteriorElementBoundaryFlux(flux_fluid_ext, u_test_dS[i]);
                  if (USE_EDGE_FLUX)
                      {
                    // FLOW-MODEL FLUXES (every edge-based branch): the advective
                    // boundary exchange is the flow model's nodalBoundaryFlux
                    // (edge loop), so only the DISPERSIVE flux stays here, on c.
                    if (isDOFBoundary_u.data()[ebNE_kb] == 1 &&
                        isDiffusiveFluxBoundary_u.data()[ebNE_kb] == 0)
                      {
                        // WEAK DIRICHLET, bound preserving: a LUMPED penalty
                        //   gamma_i (c_i - g),  gamma_i = max_a*penalty*int_e w_i >= 0,
                        // which pulls c_i toward a w_i-weighted average of g and
                        // couples to no other DOF.  The IIPG terms are dropped on
                        // these faces: the consistency term -a grad(c).n couples c_i
                        // to the far nodes of the face's element with a positive
                        // off-diagonal (it is the dispersive flux through the face,
                        // whatever c_i is), the consistent-mass penalty couples c_i
                        // to its boundary neighbours with the wrong sign, and the
                        // adjoint uses grad(w_i).  Each breaks the convex-combination
                        // form of the low-order row, which put Henry's Dirichlet
                        // nodes at c = 1.006 and -7e-8.  The price is a penalty
                        // (Robin, -a grad(c).n = gamma (c - g)) instead of a
                        // consistent BC, O(h) at the boundary since penalty ~ 1/h.
                        // The explicit branches advance from c^n, so the penalty
                        // acts on c^n there.
                        const int gi_d = u_l2g.data()[eN*nDOF_test_element+i];
                        const double c_d = (STABILIZATION_TYPE == STABILIZATION::ImplicitEV)
                          ? u_dof.data()[gi_d] : u_dof_old.data()[gi_d];
                        double max_a = 0.0;
                        for (int m=0; m<a_rowptr.data()[nSpace]; m++)
                          max_a = fmax(max_a, a_ext[m]);
                        elementResidual_u[i] += max_a*ebqe_penalty_ext.data()[ebNE_kb]
                                              *(c_d - bc_u_ext)*u_test_dS[i];
                      }
                    else
                      elementResidual_u[i] += ck.ExteriorElementBoundaryFlux(flux_diff_ext,u_test_dS[i]);
                    // the concentration inflowing water carries at this node
                    if (isDOFBoundary_u.data()[ebNE_kb] == 1)
                      {
                        const int gi_b = u_l2g.data()[eN*nDOF_test_element+i];
                        cin_num[gi_b] += bc_u_ext*u_test_dS[i];
                        cin_den[gi_b] += u_test_dS[i];
                      }
                      }
                  else if (STABILIZATION_TYPE == STABILIZATION::ImplicitEV)
                      {
                    // OUTFLOW carries the node's own concentration: on an upwinded
                    // outflow face the solute flux is c_i times the fluid flux of
                    // the same face, so B^s_i - c_i B^f_i of the cancelled row is
                    // exactly zero there.  The trace flux n.f(c_h) coupled c_i to
                    // the other nodes of the face with a positive off-diagonal,
                    // which broke the M-matrix of the low-order row (Zalesak level
                    // 4 FCT: c^L = -2.9e-9 at the outflow wall node (0, 0.625)).
                    // outflow_upwind: decided once per quadrature point, above,
                    // where the recorded flux is set from the same condition
                    const double flux_i = outflow_upwind
                      ? u_dof.data()[u_l2g.data()[eN*nDOF_test_element+i]]*flux_fluid_ext + flux_diff_ext
                      : flux_ext;
                    // ImplicitEV also transports the group variable, so its
                    // boundary_integral must be the adjoint of the w-operator
                    // (see the GROUP-VARIABLE trace block above).
                    elementResidual_u[i] += ck.ExteriorElementBoundaryFlux(flux_i,u_test_dS[i])+
                                            ck.ExteriorElementBoundaryDiffusionAdjoint(isDOFBoundary_u.data()[ebNE_kb],
                                                                                            isDiffusiveFluxBoundary_u.data()[ebNE_kb],
                                                                                            eb_adjoint_sigma,
                                                                                            w_ext,
                                                                                            w_bc_ext,
                                                                                            normal,
                                                                                            a_rowptr.data(),
                                                                                            a_colind.data(),
                                                                                            a_w_ext,
                                                                                            &u_grad_test_dS[i*nSpace]);
                      }
                  else if (STABILIZATION_TYPE == STABILIZATION::Galerkin or STABILIZATION_TYPE == STABILIZATION::VMS or
                      STABILIZATION_TYPE == STABILIZATION::TaylorGalerkinEV)
                      {
                    // cell-based branches keep the u-form throughout
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
                      // GROUP VARIABLE: Nitsche adjoint on (w, w_bc) with a_w,
                      // so it is the adjoint of the SAME operator the interior
                      // assembles.  a_w = a/(dw/du) makes the physical value
                      // identical; only the variable it is expressed in changes.
                      const double boundaryDiffusiveContribution =
                        ck.ExteriorElementBoundaryDiffusionAdjoint(isDOFBoundary_u.data()[ebNE_kb],
                                                                   isDiffusiveFluxBoundary_u.data()[ebNE_kb],
                                                                   eb_adjoint_sigma,
                                                                   w_ext,
                                                                   w_bc_ext,
                                                                   normal,
                                                                   a_rowptr.data(),
                                                                   a_colind.data(),
                                                                   a_w_ext,
                                                                   &u_grad_test_dS[i*nSpace]);
                      const double boundaryResidualContribution =
                        boundaryAdvectiveContribution + boundaryDiffusiveContribution;
                      // df/dw = v, NOT df/du -- these rows go into TransportMatrix,
                      // which the edge loop multiplies by w_j.
                      exteriorNumericalAdvectiveFluxDerivative(isDOFBoundary_u.data()[ebNE_kb],
                                                           isFluxBoundary_u.data()[ebNE_kb],
                                                           forceStrongConditions,
                                                           normal,
                                                           v_w_ext,
                                                           dflux_u_u_ext);

                      if (dflux_u_u_ext> 0.0)
                      {
                        double boundaryTransportContribution = 0.0;
                        double boundaryTransportContributionFluid = 0.0;  // SCHEME B
                        for (int j=0;j<nDOF_trial_element;j++)
                        {
                          int ebN_local_kb_j=ebN_local_kb*nDOF_trial_element+j;
                          double advJacobian_ext = 0.0, diffJacobian_ext = 0.0;
                          exteriorNumericalAdvectiveFluxDerivative(isDOFBoundary_u.data()[ebNE_kb],
                                                                   isFluxBoundary_u.data()[ebNE_kb],
                                                                   forceStrongConditions,
                                                                   normal,
                                                                   v_w_ext,
                                                                   advJacobian_ext);
                          exteriorNumericalDiffusiveFluxDerivative(isDOFBoundary_u.data()[ebNE_kb],
                                                                   isDiffusiveFluxBoundary_u.data()[ebNE_kb],
                                                                   a_rowptr.data(),
                                                                   a_colind.data(),
                                                                   normal,
                                                                   a_w_ext,
                                                                   zero_da_ext,
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
                          // ... and the value it multiplies must be w^n_j, not u^n_j.
                          // SCHEME B: w^n and rho^n come from the CARRIED masses
                          // (m_dof = M_dof*u^n, M_dof), not from the EOS -- the
                          // interior operator acts on exactly those, so an EOS
                          // value here would make the subtraction below fail to
                          // cancel what actually went into TransportMatrix.
                          {
                            const int gj = u_l2g.data()[eN_nDOF_trial_element+j];
                            const double theta_j = fmax(theta_dof_proj[gj], 1.0e-14);
                            const double wn_j   = m_dof[gj]/theta_j;
                            const double rhon_j = fmax(M_dof.data()[gj],1.0e-14)/theta_j;
                            boundaryTransportContribution += localFluxTransportContribution*wn_j;
                            boundaryTransportContributionFluid += localFluxTransportContribution*rhon_j;
                          }
                        }
                        elementResidual_u[i] += boundaryResidualContribution - boundaryTransportContribution;
                        // SCHEME B: the SAME correction on the fluid mass.  For
                        // these branches fluxTransport is loaded into
                        // TransportMatrix below, so the edge loop's
                        // sum_j (T+D)_ij rho_j ALREADY contains the linearised
                        // boundary flux; adding the true boundary flux on top
                        // without subtracting it back out double-counts it.
                        // That is exactly what drained the fluid mass to 0.4 in
                        // test_tadr (it must stay at 1 for rho = theta = 1).
                        // ImplicitEV does NOT load fluxTransport into T, so it
                        // takes the unsubtracted flux -- see the i-loop head.
                        elementResidual_fluid[i] -= boundaryTransportContributionFluid;
                      }
                      else
                      {
                        elementResidual_u[i] += boundaryResidualContribution;

                        if (isDOFBoundary_u.data()[ebNE_kb] == 1 && boundary_flow_w < 0.0)
                        {
                          // inflow upwind penalty, in the conserved variable
                          const double upwind_penalty_rate = -boundary_flow_w; // |v.n|
                          elementResidual_u[i] += upwind_penalty_rate
                                                * (w_ext - w_bc_ext)
                                                * u_test_dS[i];
                        }
                      }
                    }
                }//i


              const double u_for_bound =    isDOFBoundary_u.data()[ebNE_kb]
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
                  boundary_integral_fluid[gi] += elementResidual_fluid[i];  // SCHEME B
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
                  boundary_integral_fluid[gi] += elementResidual_fluid[i];  // SCHEME B
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
          entropy_stats[0] = meanEntropy;
          entropy_stats[1] = meanOmega;
          entropy_stats[2] = maxEntropy;
          entropy_stats[3] = minEntropy;
          meanEntropy /= meanOmega;
          double norm_factor = fmax(fabs(maxEntropy - meanEntropy), fabs(meanEntropy-minEntropy));
          for(int eN=0;eN<nElements_global;eN++)
            {
              double hK=elementDiameter.data()[eN]/degree_polynomial;
              double linear_viscosity = cMax*hK*maxVel[eN];
              element_viscosity[2*eN] = linear_viscosity;
              element_viscosity[2*eN+1] = cE*hK*hK*maxEntRes[eN];
              double entropy_viscosity = cE*hK*hK*maxEntRes[eN]/norm_factor;
              for  (int k=0;k<nQuadraturePoints_element;k++)
                {
                  int eN_k = eN*nQuadraturePoints_element+k;
                  q_numDiff_u.data()[eN_k] = fmin(linear_viscosity,entropy_viscosity);
                }
            }
        }
      //edge based stabilization, advecting with the flow model's fluxes
      else if (USE_EDGE_FLUX and
               (STABILIZATION_TYPE==STABILIZATION::EntropyViscosity or
                STABILIZATION_TYPE==STABILIZATION::SmoothnessIndicator or
                STABILIZATION_TYPE==STABILIZATION::Kuzmin))
        {
          //////////////////////////////////////////////////////////////////
          // EXPLICIT edge-based stage with the FLOW MODEL'S FLUXES.        //
          // One forward-Euler step from (c^n, M^n) = (u_dof_old, M_dof) to //
          // M^{n+1} = M_low, which Python set to M_dof plus the flow       //
          // model's storage change, so that                                //
          //   ML_i (M^{n+1}_i - M^n_i)/dt = sum_j F_ij + B_i   exactly.     //
          // The conservative solute update                                 //
          //   ML_i (M^{n+1} c^{n+1} - M^n c^n)_i/dt                         //
          //     = sum_{j!=i} [F_ij (c_i + c_j)/2 + (d_ij - D_ij)(c_j - c_i)] //
          //       + B_i c_up - (boundary dispersion) + sources              //
          // minus c^n_i times that fluid balance is the CANCELLED form     //
          //   ML_i M^{n+1}_i (c^{n+1}_i - c^n_i)/dt                         //
          //     = sum_{j!=i} (d_ij - K_ij)(c_j - c_i)                       //
          //       + max(B_i,0)(c_in - c_i) - (boundary dispersion) + ...    //
          // with K_ij = -F_ij/2 + D_ij, which is what is assembled below.  //
          // It is the conservative update rewritten, so solute mass is     //
          // conserved to round-off; with d = dLow = max(0,K_ij,K_ji) every  //
          // gamma_ij = dLow_ij - K_ij >= 0, and c^L is a convex combination //
          // of c^n_i, its neighbours and c_in when                         //
          //   dt (sum_j gamma_ij + max(B_i,0)) <= ML_i M^{n+1}_i,           //
          // which is the edge_based_cfl written here.  The high-order state //
          // replaces dLow by the branch's d^H; FCT limits the difference.  //
          // Same operator as the ImplicitEV edge-flux branch, evaluated at  //
          // c^n instead of c^{n+1}.  The fluid mass is not advanced here: M //
          // is the flow model's, carried by Python (RKEV.updateStage and   //
          // Coefficients.postStep).                                        //
          //////////////////////////////////////////////////////////////////
          // INDICATORS, on c (the variable the operator acts on).
          //  * EV: there is no velocity to form d_t eta + v.grad eta with, so
          //    the residual is the entropy commutator of the centered edge flux,
          //      R_i = sum_{j!=i} (F_ij/2) [(eta_j - eta_i) - eta'_i (c_j - c_i)],
          //    the edge form of eta'(c) v.grad c - v.grad eta(c) (Guermond,
          //    Nazarov, Popov, Tomas, SISC 2018): zero for c linear along every
          //    edge, O(F) at a discontinuity.  Normalized as the element-based
          //    residual is, by 2 cE over the local entropy range.
          //  * SI/Kuzmin: the smoothness indicator on c.
          int ij = 0;
          for (int i=0; i<numDOFs; i++)
            {
              const double ci = u_dof_old.data()[i];
              const double eta_i  = (ENTROPY_TYPE == ENTROPY::POWER) ? EPOWER(ci,uL,uR) : ELOG(ci,uL,uR);
              const double deta_i = (ENTROPY_TYPE == ENTROPY::POWER) ? DEPOWER(ci,uL,uR) : DELOG(ci,uL,uR);
              double etaMaxi = eta_i, etaMini = eta_i, entropy_residual = 0.0;
              double alpha_numerator = 0., alpha_denominator = 0.;
              for (int offset=csrRowIndeces_DofLoops.data()[i]; offset<csrRowIndeces_DofLoops.data()[i+1]; offset++)
                {
                  const int j = csrColumnOffsets_DofLoops.data()[offset];
                  const double cj = u_dof_old.data()[j];
                  if (STABILIZATION_TYPE==STABILIZATION::EntropyViscosity && i != j)
                    {
                      const double eta_j = (ENTROPY_TYPE == ENTROPY::POWER) ? EPOWER(cj,uL,uR) : ELOG(cj,uL,uR);
                      etaMaxi = fmax(etaMaxi, eta_j);
                      etaMini = fmin(etaMini, eta_j);
                      const double Fij = USE_EDGE_VELOCITY
                        ? AdvectionMatrix[ij] - TransposeAdvectionMatrix[ij]
                        : edgeFlux.data()[ij];
                      entropy_residual += 0.5*Fij*((eta_j - eta_i) - deta_i*(cj - ci));
                    }
                  alpha_numerator += ci - cj;
                  alpha_denominator += fabs(ci - cj);
                  ij += 1;
                }
              if (STABILIZATION_TYPE==STABILIZATION::EntropyViscosity)
                {
                  global_entropy_residual[i] = (etaMaxi == etaMini) ? 0.0
                    : 2*cE*entropy_residual/(etaMaxi - etaMini);
                  quantDOFs.data()[i] = fabs(global_entropy_residual[i]);
                }
              const double alphai = alpha_numerator/(alpha_denominator+1E-15);
              if (STABILIZATION_TYPE!=STABILIZATION::EntropyViscosity)
                quantDOFs.data()[i] = alphai;
              psi[i] = (POWER_SMOOTHNESS_INDICATOR==0) ? 1.0 : std::pow(alphai,POWER_SMOOTHNESS_INDICATOR);
            }
          // EDGE LOOP: low- and high-order stage, cancelled form.
          ij = 0;
          for (int i=0; i<numDOFs; i++)
            {
              const double ci      = u_dof_old.data()[i];
              const double MLi     = ML.data()[i];
              const double Mnew_i  = fmax(M_low.data()[i], 1.0e-14);
              const double theta_i = fmax(theta_dof_proj[i], 1.0e-14);
              double low_term = 0.0, high_term = 0.0, sum_gamma = 0.0;
              double ith_flux_term_fluid = 0.0;   // -sum_j F_ij, for the defect probe
              for (int offset=csrRowIndeces_DofLoops.data()[i]; offset<csrRowIndeces_DofLoops.data()[i+1]; offset++)
                {
                  const int j = csrColumnOffsets_DofLoops.data()[offset];
                  if (i != j)
                    {
                      const double cj  = u_dof_old.data()[j];
                      const double Fij = USE_EDGE_VELOCITY
                        ? AdvectionMatrix[ij] - TransposeAdvectionMatrix[ij]
                        : edgeFlux.data()[ij];
                      if (USE_EDGE_VELOCITY)
                        edgeFlux.data()[ij] = Fij;
                      const double Kij = -0.5*Fij + TransportMatrix[ij] + DiffusionMatrix[ij];
                      const double Kji =  0.5*Fij + TransposeTransportMatrix[ij] + DiffusionMatrix[ij];
                      const double dLowij = fmax(0.0, fmax(Kij, Kji));
                      // artificial compression, as in the branch below
                      const double solij  = 0.5*(ci + cj);
                      const double Compij = cK*fmax(solij*(1.0-solij),0.0)/(fabs(ci-cj)+1E-14);
                      const double dHij = (STABILIZATION_TYPE==STABILIZATION::EntropyViscosity)
                        ? fmin(dLowij, fmax(fabs(global_entropy_residual[i]),
                                            fabs(global_entropy_residual[j])))*fmax(1.0-Compij,0.0)
                        : dLowij*fmax(psi[i],psi[j])*fmax(1.0-Compij,0.0);
                      const double gamma = dLowij - Kij;     // >= 0 by construction
                      low_term  += gamma*(cj - ci);
                      high_term += (dHij - Kij)*(cj - ci);
                      sum_gamma += gamma;
                      ith_flux_term_fluid -= Fij;
                      dLow.data()[ij] = dLowij;
                      dt_times_dH_minus_dL.data()[ij] = dt*(dHij - dLowij);
                    }
                  else
                    {
                      dLow.data()[ij] = 0.0;
                      dt_times_dH_minus_dL.data()[ij] = 0.0;
                      if (USE_EDGE_VELOCITY)
                        edgeFlux.data()[ij] = 0.0;
                    }
                  ij += 1;
                }
              // The flow model's boundary exchange; with the edge velocity it is
              // what the assembled F leaves of the carried mass change.
              if (USE_EDGE_VELOCITY)
                nodalBoundaryFlux.data()[i] = MLi*(M_low.data()[i] - M_dof.data()[i])/dt
                                              + ith_flux_term_fluid;
              const double B_i  = nodalBoundaryFlux.data()[i];
              const double c_in = (cin_den[i] > 0.0) ? cin_num[i]/cin_den[i] : ci;
              // Stage-3 kinetic dissolution source, as in the ImplicitEV edge-flux
              // branch (solute only; the fluid is the flow model's)
              const double S_n_i    = Sn_dof.data()[i];
              const double R_diss_i = theta_i*rho_f*(1.0 + eps_rho*ci)*k_d*S_n_i*(c_sat - ci);
              const double rhs_rest = - boundary_integral[i]
                                      + fmax(B_i, 0.0)*(c_in - ci)
                                      + MLi*R_diss_i
                                      + MLi*nodalSource.data()[i];
              const double scale = dt/(MLi*Mnew_i);
              uLow.data()[i] = ci + scale*(low_term + rhs_rest);
              globalResidual.data()[i] = ci + scale*(high_term + rhs_rest);
              edge_based_cfl.data()[i] = (sum_gamma + fmax(B_i, 0.0))/(MLi*Mnew_i);
              // the flow model's fluid flux, so the continuity defect against the
              // carried M^{n+1} is round-off by construction
              fluidResidual.data()[i] = ith_flux_term_fluid - B_i;
            }
          // the recorded boundary flux: what this update applied, at c^n
          recordEdgeFluxBoundaryExchange(nExteriorElementBoundaries_global,
                                         numDOFs, exteriorElementBoundariesArray.data(),
                                         elementBoundaryElementsArray.data(),
                                         elementBoundaryLocalElementBoundariesArray.data(),
                                         elementBoundaryMaterialTypes.data(), isExteriorBoundaryPhysical.data(),
                                         u_l2g.data(), u_test_trace_ref.data(), ebqe_dS_ext.data(),
                                         nodalBoundaryFlux.data(), u_dof_old.data(),
                                         ebqe_flux.data(), ebqe_flux_adv.data());
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
                  // Sense smoothness on the GROUP VARIABLE w = rho(u)*u = m/theta,
                  // the variable the edge operator actually transports.  Using m
                  // instead folds the porosity jump at a material interface into
                  // the smoothness sensor, which drives psi -> 1 (and so the full
                  // low-order viscosity) across an interface even where the
                  // transported field is perfectly smooth.
                  const double wi = m_dof[i]/fmax(theta_dof_proj[i], 1.0e-14);
                  const double wj = m_dof[j]/fmax(theta_dof_proj[j], 1.0e-14);
                  alpha_numerator += wi - wj;
                  alpha_denominator += fabs(wi - wj);
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
          // DMP diagnostic, Coefficients(diagnostics=("dmp",)) (off by default,
          // and the branch is a single bool test when off).
          const bool TADR_DMP_DBG = args.scalar<int>("DMP_DIAGNOSTICS") != 0;
          // total lines emitted over the whole run, so the first violation is
          // captured and the log does not explode over ~12000 substeps x 64 ranks
          static int dbg_budget = 40;
          int    dbg_reported = 0, dbg_argmax_coef = -1;
          double dbg_global_max_coef = -1.0e300, dbg_global_min_gamma = 1.0e300;
          ij=0;
          for (int i=0; i<numDOFs; i++)
            {
              const double mi_mass = m_dof[i];
              const double theta_i = fmax(theta_dof_proj[i], 1.0e-14);
              // SCHEME B: u is the RATIO of the two transported masses, so the
              // EOS inverse is not used here at all.  m_dof[i] = M_dof[i]*u^n_i
              // was set from the carried fluid mass above, so this returns
              // u^n_i exactly (no projection/EOS round-trip, which is what
              // used to break the FCT seam).
              const double Mi_fluid = fmax(M_dof.data()[i], 1.0e-14);
              const double ui_mass = mi_mass/Mi_fluid;
              const double rhoi_fluid = Mi_fluid/theta_i;
              // GROUP VARIABLE w = rho(u)*u = m/theta.  This is what the
              // transport matrix now acts on (see the assembly comment in the
              // element loop), so every flux and dissipation difference below is
              // taken in w, and the storage is simply m = theta*w.
              const double wi_mass = mi_mass/theta_i;
              double ith_dissipative_term_mass = 0;
              double ith_low_order_dissipative_term_mass = 0;
              double ith_flux_term_mass = 0;
              // SCHEME B: the SAME operator, applied to the fluid mass.  These
              // must mirror their solute counterparts term for term -- any
              // asymmetry between the two breaks the cancellation that makes
              // u = m/M a convex combination.
              double ith_flux_term_fluid = 0;
              double ith_low_order_dissipative_term_fluid = 0;
              // --- DMP diagnostic accumulators (diagnostics "dmp") ---
              // gamma_ij = dLow_ij - T_ij - D_ij is the coefficient of
              // (w_j - w_i) in the low-order update.  The update is a convex
              // combination iff every gamma_ij >= 0 AND
              //   coef_i = dt*sum_j gamma_ij/(theta_i*ML_i) <= 1.
              // dLow_from_transpose counts edges where dLow is set by |T_ji|
              // rather than |T_ij| -- i.e. by the OTHER row, which for an edge
              // touching the boundary carries the boundary flux Jacobian.
              double dbg_sum_gamma = 0.0, dbg_min_gamma = 1.0e300;
              double dbg_max_dLow = 0.0, dbg_max_absT = 0.0, dbg_max_absTt = 0.0;
              int    dbg_n_from_transpose = 0;
              // Row sum of the high-order graph viscosity, dLii = -sum_{j!=i} dLij.
              // Feeds edge_based_cfl below.
              double dLii = 0.;

              // loop over the sparsity pattern of the i-th DOF
              for (int offset=csrRowIndeces_DofLoops.data()[i]; offset<csrRowIndeces_DofLoops.data()[i+1]; offset++)
                {
                  int j = csrColumnOffsets_DofLoops.data()[offset];
                  const double mj_mass = m_dof[j];
                  const double theta_j = fmax(theta_dof_proj[j], 1.0e-14);
                  const double Mj_fluid = fmax(M_dof.data()[j], 1.0e-14);
                  const double uj_mass = mj_mass/Mj_fluid;
                  const double wj_mass = mj_mass/theta_j;
                  const double rhoj_fluid = Mj_fluid/theta_j;
                  double dLowij, dLij, dEVij, dHij;

                  ith_flux_term_mass  += (TransportMatrix[ij] + DiffusionMatrix[ij])*wj_mass;
                  ith_flux_term_fluid += (TransportMatrix[ij] + DiffusionMatrix[ij])*rhoj_fluid;

                  if (i != j)
                    {
                      double solij = 0.5*(ui_mass+uj_mass);
                      double Compij = cK*fmax(solij*(1.0-solij),0.0)/(fabs(ui_mass-uj_mass)+1E-14);
                      // MINIMAL Kuzmin graph viscosity, built from the COMBINED
                      // operator K = T + D.  The old max(|K_ij|,|K_ji|) is an
                      // over-estimate: T carries the dispersion stiffness (see
                      // the element assembly, ck.SimpleDiffusionJacobian_weak
                      // goes into elementTransport, not into elementDiffusion),
                      // and for a symmetric negative off-diagonal entry
                      // max(|K_ij|,|K_ji|) - K_ij = -2 K_ij applies the physical
                      // dispersion TWICE.  Verified in the numpy replica at
                      // v = 0: ratio to the exact D/h is 2.000 with the old
                      // form, 1.000 with this one.  max(0,K_ij,K_ji) is still
                      // symmetric (so the FCT antidiffusive flux stays
                      // antisymmetric and the scheme stays conservative) and
                      // still gives gamma_ij = dLow - K_ij >= 0 by definition,
                      // including for positive off-diagonals on obtuse elements.
                      // Where the mesh Peclet number allows it this collapses to
                      // dLow = 0, i.e. central differencing with the physical
                      // dispersion only -- no artificial diffusion at all.
                      {
                        const double Kij = TransportMatrix[ij] + DiffusionMatrix[ij];
                        const double Kji = TransposeTransportMatrix[ij] + DiffusionMatrix[ij];
                        dLowij = fmax(0.0, fmax(Kij, Kji));
                      }

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
                      // Dissipative terms.  The DIFFERENCES are taken in the GROUP
                      // VARIABLE w = rho(u)*u -- the same variable the transport
                      // matrix now acts on -- so the pair telescopes into the
                      // monotone operator:
                      //   sum_j (T+D)_ij w_j - sum_{j!=i} dLow_ij (w_j - w_i)
                      //     = -sum_{j!=i} (dLow_ij - T_ij - D_ij)(w_j - w_i)
                      // with dLow_ij - T_ij - D_ij >= 0 because dLow_ij >= |T_ij|,
                      // i.e. an M-matrix.  The rewrite of the left side into the
                      // right side is valid ONLY because T now has zero row sums
                      // (df/dw = v, independent of the solution); differencing
                      // against u with the u-assembled T left a residue
                      // (sum_j T_ij)*c_i that broke boundedness at every
                      // rho_s != rho_f.
                      //
                      // It must NOT be replaced by -sum max(0,-T_ij)(w_j - w_i):
                      // for the skew-symmetric advection matrix that is exactly
                      // HALF the upwind flux.
                      ith_dissipative_term_mass += dHij*(wj_mass-wi_mass);
                      ith_low_order_dissipative_term_mass += dLowij*(wj_mass-wi_mass);
                      // SAME dLow on the fluid mass -- symmetric, so it is
                      // conservative, and identical between the two equations,
                      // which is what the convex-combination identity needs.
                      ith_low_order_dissipative_term_fluid += dLowij*(rhoj_fluid-rhoi_fluid);
                      {
                        const double aT  = fabs(TransportMatrix[ij]);
                        const double aTt = fabs(TransposeTransportMatrix[ij]);
                        const double gam = dLowij - TransportMatrix[ij] - DiffusionMatrix[ij];
                        dbg_sum_gamma += gam;
                        dbg_min_gamma = fmin(dbg_min_gamma, gam);
                        dbg_max_dLow  = fmax(dbg_max_dLow, dLowij);
                        dbg_max_absT  = fmax(dbg_max_absT, aT);
                        dbg_max_absTt = fmax(dbg_max_absTt, aTt);
                        if (aTt > aT) dbg_n_from_transpose += 1;
                      }
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
              // In the group variable the low-order update is
              //   w_i^L = w_i + dt/(theta_i*mi) sum_{j!=i} gamma_ij (w_j - w_i),
              // a convex combination of {w_i, w_j} as long as
              //   dt/(theta_i*mi) sum_j gamma_ij <= 1,  gamma_ij = dLow-T-D.
              // So the storage Jacobian that has to divide 2|dLii|/mi is just
              // theta_i (dm/dw = theta), NOT the old dm/du = theta*(rho+u drho/du).
              // Using dm/du here was also one-sided: it is evaluated at node i,
              // while the undershoot bound needs the stencil minimum, which made
              // dt up to 19% too permissive at eps_rho = 0.235.  theta_i has no
              // such solution dependence, so the bound is now exact.
              // theta_i = 1 recovers the original 2|dLii|/mi.
              edge_based_cfl.data()[i] = 2.*fabs(dLii)/(mi * theta_i);
              
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
              // SCHEME B: advance the fluid mass with the SAME operator, then
              // take the concentration as the ratio.  R_diss is added to BOTH:
              // dissolved CO2 mass joins the aqueous phase, so it raises the
              // solute mass and the fluid mass by the same amount, and
              // (m+d)/(M+d) stays in [u, 1] for d >= 0.  (k_d = 0 by default,
              // so this is inert unless kinetic dissolution is switched on;
              // for EXSOLUTION, d < 0, the bound is not guaranteed.)
              const double MLow_i = fmax(M_dof.data()[i]
                                         - dt/mi*(ith_flux_term_fluid
                                                  + boundary_integral_fluid[i]
                                                  - ith_low_order_dissipative_term_fluid)
                                         + dt * R_diss_i,
                                         1.0e-14);
              M_low.data()[i] = MLow_i;
              uLow[i] = mLow_i/MLow_i;

              // --- DMP diagnostic: report the first few violating nodes ---
              // Distinguishes the three possible causes in one run:
              //   min_gamma < 0        -> a coefficient is negative (not an M-matrix)
              //   coef > 1             -> dt exceeds the LOCAL stability limit
              //   both fine, still >1  -> the violation is not from this update
              if (TADR_DMP_DBG && uLow[i] > 1.0 + 1.0e-10 && dbg_reported < 6 && dbg_budget > 0)
                {
                  const double coef = dt*dbg_sum_gamma/fmax(theta_i*mi, 1.0e-300);
                  std::cout << "TADR_DMP i=" << i
                            << " uLow=" << uLow[i]
                            << " wi=" << wi_mass
                            << " theta=" << theta_i
                            << " ML=" << mi
                            << " dt=" << dt
                            << " min_gamma=" << dbg_min_gamma
                            << " coef=" << coef
                            << " max_dLow=" << dbg_max_dLow
                            << " max|T_ij|=" << dbg_max_absT
                            << " max|T_ji|=" << dbg_max_absTt
                            << " n_dLow_from_transpose=" << dbg_n_from_transpose
                            << " bndry_int=" << boundary_integral_mass
                            << std::endl;
                  dbg_reported += 1; dbg_budget -= 1;
                }
              if (TADR_DMP_DBG)
                {
                  const double coef = dt*dbg_sum_gamma/fmax(theta_i*mi, 1.0e-300);
                  if (coef > dbg_global_max_coef) { dbg_global_max_coef = coef; dbg_argmax_coef = i; }
                  if (dbg_min_gamma < dbg_global_min_gamma) dbg_global_min_gamma = dbg_min_gamma;
                }

              // update residual
              if (LUMPED_MASS_MATRIX==1)
                {
                  const double mHigh_i = mi_mass - dt/mi*(ith_flux_term_mass
                                                          + boundary_integral_mass
                                                          - ith_dissipative_term_mass)
                                                 + dt * R_diss_i;
                  // SCHEME B: the fluid mass is a single state, advanced ONCE
                  // with the low-order operator.  Only the SOLUTE mass carries
                  // a high-order counterpart and gets the FCT treatment, so the
                  // high-order concentration divides by the same MLow_i.  Using
                  // two different denominators for solH and uLow would make the
                  // antidiffusive flux inconsistent with the bounds.
                  globalResidual.data()[i] = mHigh_i/MLow_i;
                  // Probe the HIGH-order solution too.  uLow is known bounded;
                  // if this one violates while uLow does not, the archived field
                  // is the high-order solution and the FCT/low-order result is
                  // being discarded downstream of FCTStep.
                  if (TADR_DMP_DBG && globalResidual.data()[i] > 1.0 + 1.0e-10 && dbg_budget > 0)
                    {
                      std::cout << "TADR_DMP_HIGH i=" << i
                                << " solH=" << globalResidual.data()[i]
                                << " uLow=" << uLow[i]
                                << " theta=" << theta_i
                                << std::endl;
                      dbg_budget -= 1;
                    }
                }
              else
                globalResidual.data()[i] += dt*(ith_flux_term_mass - ith_dissipative_term_mass - R_diss_i);//cek todo: shouldn't this have boundaryIntegral?
            }//i
          if (TADR_DMP_DBG && dbg_reported > 0)
            std::cout << "TADR_DMP_SUMMARY dt=" << dt
                      << " max_coef=" << dbg_global_max_coef
                      << " at_dof=" << dbg_argmax_coef
                      << " min_gamma_anywhere=" << dbg_global_min_gamma
                      << " (coef<=1 and gamma>=0 => convex combination)"
                      << std::endl;
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
          // LOW-ORDER FLUX: the FULL-SPEED Kuzmin operator, identical in   //
          // form to the one the explicit edge-based branch builds for uLow,//
          // but in the GROUP VARIABLE w = rho(u)*u and at the current       //
          // Newton iterate:                                                //
          //                                                                //
          //   F_i = sum_j (T_ij + D_ij) w_j                                //
          //         - sum_{j!=i} dLow_ij (w_j - w_i),                      //
          //   dLow_ij = max(|T_ij|, |T_ji|)     (symmetric)                //
          //                                                                //
          // which telescopes to  -sum_{j!=i} (dLow - T - D)_ij (w_j - w_i) //
          // with (dLow - T - D)_ij >= 0, i.e. an M-matrix (monotone, DMP), //
          // and is conservative because dLow is SYMMETRIC.                 //
          //                                                                //
          // THIS REPLACES the old  F_i = -sum_{j!=i} a_ij (c_j - c_i)  with//
          // a_ij = max(0,-T_ij)*(rho_up/rho_f) + max(0,-D_ij).  That form  //
          // is also an M-matrix, but for the skew-symmetric P1 advection   //
          // matrix it is EXACTLY HALF the Kuzmin coefficient: on an edge   //
          // with T_ij < 0 it applies |T_ij| where the Kuzmin operator      //
          // applies dLow_ij - T_ij = 2|T_ij|.  Verified node-for-node in a //
          // numpy replica (ratio 0.500 at every node).  The consequence is //
          // not extra diffusion but HALVED TRANSPORT SPEED -- the scheme   //
          // advects the front at half the physical rate, which is why the  //
          // Elder plume reached the bottom later than it should.  It also  //
          // meant the FCT post-step antidiffused the symmetric dLow it     //
          // never actually applied, a factor-2 mismatch.                   //
          //                                                                //
          // dLow[ij] (symmetric) is stashed for the FCT post-step and is   //
          // now ALSO what calculateJacobian differentiates, so residual    //
          // and Jacobian use one and the same operator.                    //
          // m^{n+1}_i=theta_i*rho(c_i)*c_i (nodal, porosity lagged);       //
          // m^n_i=m_dof[i].  boundary_integral stays 0 (closed system);    //
          // implicit boundary flux is a TODO.                              //
          //////////////////////////////////////////////////////////////////
          //////////////////////////////////////////////////////////////////
          // SCHEME B, implicit form.                                      //
          //                                                               //
          // The fluid mass is KNOWN before the solute solve -- under      //
          // option C it is Richards' theta*rho, supplied in M_dof -- and  //
          // the solute is solved implicitly for u against it.  Writing    //
          // the two conservative equations                                //
          //   ML(M^{n+1}u^{n+1} - M^n u^n)/dt + sum_j K_ij rho^n_j u^{n+1}_j
          //        - sum_j d_ij rho^n_j (u^{n+1}_j - u^{n+1}_i) + B^s = 0 //
          //   ML(M^{n+1} - M^n)/dt + sum_j K_ij rho^n_j                   //
          //        - sum_j d_ij (rho^n_j - rho^n_i) + B^f = 0             //
          // and subtracting u^{n+1}_i times the second from the first     //
          // cancels every row-sum term and leaves                         //
          //   ML M^n (u_i - u^n_i)/dt                                     //
          //     - sum_{j!=i} gamma_ij rho^n_j (u_j - u_i)                 //
          //     + (B^s_i - u_i B^f_i)  -  ML R_diss_i  = 0                //
          // with gamma_ij = d_ij - K_ij >= 0.  That is an M-matrix with a //
          // positive diagonal, so backward Euler is UNCONDITIONALLY       //
          // bounded -- no CFL, and no divergence-free assumption on v.    //
          //                                                               //
          // The cancellation is REVERSIBLE, so solving this form for u is //
          // algebraically identical to solving the two conservative       //
          // equations -- PROVIDED the M it is handed satisfies the second //
          // one.  Under option C it does not exactly (it satisfies        //
          // Richards' continuity instead), and that gap, not the          //
          // cancellation, is where conservation is lost.  See the option  //
          // C note below.                                                 //
          //                                                               //
          // This replaces BOTH the half-upwind stopgap (a_ij =            //
          // max(0,-T_ij), bounded but exactly HALF the physical transport //
          // speed) and the full-speed consistent form that preceded it    //
          // (correct speed, but max u = 4.25 on Elder from a u <= 1 IC).  //
          // gamma_ij is the FULL-SPEED coefficient: for a skew T it is    //
          // exactly 2*max(0,-T_ij), i.e. twice the stopgap.               //
          //////////////////////////////////////////////////////////////////
          // --- fluid mass: OPTION C, read from the flow model --------------
          //
          // There is no fluid system assembled or solved here any more.  M_dof
          // arrives already holding Richards' old-time mass, refreshed on the
          // Python side (TADR.LevelModel._refreshFluidMassFromFlow) before this
          // kernel runs, and it serves as the old-time storage coefficient M^n_i
          // ONLY.  Richards evaluates it with the OLD density (the mass its
          // previous step produced), so it is theta(psi^n)*rho(c^{n-1}), not
          // theta*rho(c^n), and M_j/theta_j is no longer this step's density.  The flux density is
          // rho_j = rho(c^n_j) from the EOS at the node (rho_dof_proj), which is
          // exactly the nodal density Richards advected with this step.
          //
          // What this replaces: the fluid mass used to be advanced here as its
          // own unknown.  Explicitly it inherited an explicit CFL (past it M
          // went NEGATIVE, rho = M/theta flipped sign, the solute off-diagonals
          // -gamma_ij*rho_j turned POSITIVE and boundedness died -- replica at
          // 10x the limit: min M = -6.5e52, max u = 2.2e25).  Implicitly it
          // needed a linear solve, and the sub-cycler that supplied one cost
          // S = 100+ matvec + ghost-scatter collectives per step on the 20 yr
          // Elder run.  Richards is already solving this equation; solving a
          // second, slightly different copy of it was the expensive part.
          //
          // WHAT IT COSTS.  The cancelled solute form below is algebraically
          // equivalent to the conservative pair only when M satisfies THIS
          // model's discrete fluid equation (same K, same d, same B^f).
          // Richards' m satisfies Richards' discrete continuity equation
          // instead.  The gap is the flow/transport continuity defect, which on
          // the 20 yr Elder run grew monotonically from 1.4e-16 to 3.6e-2.  So
          // conservation here is to that defect, NOT to round-off; the
          // [TADR-MASS] log reports the consequence every step.
          //
          // BOUNDEDNESS is untouched and still unconditional: the form below is
          // an M-matrix for ANY strictly positive M and rho, since gamma_ij >= 0,
          // rho_j = rho(c^n_j) > 0, and M^n > 0 whenever Richards keeps theta*rho
          // positive.  The cancellation needs the SAME rho in the solute and the
          // fluid flux, not rho = M/theta.
          //
          // --- solute: implicit, in the cancelled (bounded) form ---------
          int ij=0;
          for (int i=0; i<numDOFs; i++)
            {
              const double ci      = u_dof.data()[i];
              const double theta_i = fmax(theta_dof_proj[i], 1.0e-14);
              const double Mn_i    = fmax(M_dof.data()[i], 1.0e-14);
              const double un_i    = m_dof[i]/Mn_i;      // = u_dof_old[i] exactly
              const double MLi     = ML.data()[i];
              const double rho_i   = rho_dof_proj[i];

              double ith_upwind_flux_term_mass = 0.0;
              double ith_flux_term_fluid       = 0.0;   // continuity-defect probe
              for (int offset=csrRowIndeces_DofLoops.data()[i]; offset<csrRowIndeces_DofLoops.data()[i+1]; offset++)
                {
                  int j = csrColumnOffsets_DofLoops.data()[offset];
                  // rho(c^n_j) from the EOS at the node -- the same nodal density
                  // Richards advected with.  The solute flux and the fluid flux
                  // below use this ONE density, which is what the cancellation
                  // needs.
                  const double rhoj    = rho_dof_proj[j];
                  if (USE_EDGE_FLUX)
                    {
                      // FLOW-MODEL FLUXES: K_ij = A_ij + D_ij with A_ij = -F_ij/2,
                      // F the flow model's mass flux into i from j (F_ji = -F_ij,
                      // so A_ji = +F_ij/2), and D the dispersion on c assembled in
                      // the element loop.  F is already a mass flux, so there is
                      // no rho_j factor.  The same graph viscosity then gives
                      // gamma_ij = max(F_ij,0) + (dispersion part): upwinding on
                      // the flow model's own fluxes.
                      if (i != j)
                        {
                          const double cj      = u_dof.data()[j];
                          const double Fij     = USE_EDGE_VELOCITY
                            ? AdvectionMatrix[ij] - TransposeAdvectionMatrix[ij]
                            : edgeFlux.data()[ij];
                          // calculateJacobian reads the flux the residual applied
                          if (USE_EDGE_VELOCITY)
                            edgeFlux.data()[ij] = Fij;
                          const double Kij     = -0.5*Fij + TransportMatrix[ij] + DiffusionMatrix[ij];
                          const double Kji     =  0.5*Fij + TransposeTransportMatrix[ij] + DiffusionMatrix[ij];
                          const double dLow_ij = fmax(0.0, fmax(Kij, Kji));
                          const double gamma   = dLow_ij - Kij;     // >= 0 by construction
                          ith_upwind_flux_term_mass += -gamma*(cj - ci);
                          ith_flux_term_fluid       -= Fij;
                          dLow.data()[ij] = dLow_ij;
                          dt_times_dH_minus_dL.data()[ij] = dt*(0.0 - dLow_ij);
                        }
                      else
                        {
                          dLow.data()[ij] = 0.0;
                          dt_times_dH_minus_dL.data()[ij] = 0.0;
                          // A_ii - A_ii
                          if (USE_EDGE_VELOCITY)
                            edgeFlux.data()[ij] = 0.0;
                        }
                      ij += 1;
                      continue;
                    }
                  // Fluid flux, assembled EXACTLY as the explicit branch does it:
                  // K over every j (diagonal included), dLow only off-diagonal.
                  ith_flux_term_fluid += (TransportMatrix[ij] + DiffusionMatrix[ij])*rhoj;
                  if (i != j)
                    {
                      const double cj      = u_dof.data()[j];
                      const double Kij     = TransportMatrix[ij] + DiffusionMatrix[ij];
                      const double Kji     = TransposeTransportMatrix[ij] + DiffusionMatrix[ij];
                      const double dLow_ij = fmax(0.0, fmax(Kij, Kji));
                      const double gamma   = dLow_ij - Kij;     // >= 0 by construction
                      ith_upwind_flux_term_mass += -gamma*rhoj*(cj - ci);
                      ith_flux_term_fluid       -= dLow_ij*(rhoj - rho_i);
                      // dLow (symmetric) is what the FCT post-step antidiffuses;
                      // it must be the viscosity the residual ACTUALLY applied,
                      // which is now the case (the half-upwind form applied a_ij
                      // while FCT removed dLow -- a factor-2 mismatch).
                      dLow.data()[ij] = dLow_ij;
                      dt_times_dH_minus_dL.data()[ij] = dt*(0.0 - dLow_ij);
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
              const double rho_w_i = rho_f*(1.0 + ((rho_s - rho_f)/rho_f)*ci);
              const double R_diss_i = theta_i * rho_w_i * k_d * S_n_i * (c_sat - ci);
              // Exported for the flow model: at the converged iterate this is
              // rho(c^{n+1}_i), the nodal density Richards uses next step.
              rho_nodal.data()[i] = rho_w_i;

              if (USE_EDGE_FLUX)
                {
                  // The flow model's boundary exchange.  Outflow (B < 0) leaves
                  // with c_i and cancels against the fluid equation; inflow
                  // brings the boundary concentration, a convex pull toward it.
                  // With the edge velocity, B_i is what the assembled F leaves of
                  // the carried mass change (ith_flux_term_fluid = -sum_j F_ij).
                  if (USE_EDGE_VELOCITY)
                    nodalBoundaryFlux.data()[i] = MLi*(M_low.data()[i] - M_dof.data()[i])/dt
                                                  + ith_flux_term_fluid;
                  const double B_i  = nodalBoundaryFlux.data()[i];
                  const double c_in = (cin_den[i] > 0.0) ? cin_num[i]/cin_den[i] : ci;
                  globalResidual.data()[i] = MLi*Mn_i*(ci - un_i)/dt
                                           + ith_upwind_flux_term_mass
                                           + boundary_integral[i]
                                           - fmax(B_i, 0.0)*(c_in - ci)
                                           - MLi*R_diss_i
                                           - MLi*nodalSource.data()[i];
                  // the flow model's fluid flux, so the continuity defect against
                  // the carried M^{n+1} is round-off by construction
                  fluidResidual.data()[i] = ith_flux_term_fluid - B_i;
                  continue;
                }
              globalResidual.data()[i] = MLi*Mn_i*(ci - un_i)/dt
                                       + ith_upwind_flux_term_mass
                                       + boundary_integral[i]
                                       - ci*boundary_integral_fluid[i]
                                       - MLi*R_diss_i;
              // Everything in TADR's fluid equation EXCEPT the storage term; the
              // storage needs M^{n+1}, which only the flow model has.
              fluidResidual.data()[i] = ith_flux_term_fluid
                                      + boundary_integral_fluid[i]
                                      - MLi*R_diss_i;
            }//i
          // the recorded boundary flux: what this residual applied, at c^{n+1}
          if (USE_EDGE_FLUX)
            recordEdgeFluxBoundaryExchange(nExteriorElementBoundaries_global,
                                           numDOFs, exteriorElementBoundariesArray.data(),
                                           elementBoundaryElementsArray.data(),
                                           elementBoundaryLocalElementBoundariesArray.data(),
                                           elementBoundaryMaterialTypes.data(), isExteriorBoundaryPhysical.data(),
                                           u_l2g.data(), u_test_trace_ref.data(), ebqe_dS_ext.data(),
                                           nodalBoundaryFlux.data(), u_dof.data(),
                                           ebqe_flux.data(), ebqe_flux_adv.data());
        }//implicit edge-based
    }

  void invert(arguments_dict& args)
  {
    int numDOFs = args.scalar<int>("numDOFs");
    proteus::pyarray<double>& mIn = args.array<double>("mIn");
    proteus::pyarray<double>& uOut = args.array<double>("uOut");
    proteus::pyarray<double>& nodal_porosity = args.array<double>("nodal_porosity");
    proteus::pyarray<double>& M_dof_fct = args.array<double>("M_dof_fct");
    proteus::pyarray<double>& M_low_fct = args.array<double>("M_low_fct");
    // OUT (ImplicitEV only): the residual left rho of the UNLIMITED iterate here;
    // the flow model must see rho of the limited c.
    proteus::pyarray<double>& rho_nodal = args.array<double>("rho_nodal");
    const double rho_f = args.scalar<double>("rho_f");
    const double rho_s = args.scalar<double>("rho_s");
    STABILIZATION STABILIZATION_TYPE = static_cast<STABILIZATION>(args.scalar<int>("STABILIZATION_TYPE"));
    const bool implicitM = (STABILIZATION_TYPE == STABILIZATION::ImplicitEV);
    // ImplicitEV with the flow model's fluxes carries M^{n+1} in M_low; without
    // them it has no new-time mass and uses M^n.  Must match FCTStep.
    const bool newM_is_old = implicitM && !args.scalar<int>("USE_EDGE_FLUX");
    // The scheme-B branches limited S = M*c, so the inverse is a division by the
    // SAME new-time mass -- no EOS round-trip.  Must match FCTStep's useM set.
    if (implicitM or
        STABILIZATION_TYPE == STABILIZATION::EntropyViscosity or
        STABILIZATION_TYPE == STABILIZATION::SmoothnessIndicator or
        STABILIZATION_TYPE == STABILIZATION::Kuzmin)
      for (int i=0; i<numDOFs; i++)
        uOut.data()[i] = mIn.data()[i]/fmax((newM_is_old ? M_dof_fct : M_low_fct).data()[i], 1.0e-14);
    else
      for (int i=0; i<numDOFs; i++)
        uOut.data()[i] = inversevaluateCoefficients(mIn.data()[i], nodal_porosity.data()[i], rho_f, rho_s);
    if (implicitM)
      for (int i=0; i<numDOFs; i++)
        rho_nodal.data()[i] = rho_f*(1.0 + ((rho_s - rho_f)/rho_f)*uOut.data()[i]);
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
          // OPTION C: the old-time fluid mass, read from Richards.  The
          // immediately preceding calculateResidual filled theta_dof_proj,
          // rho_dof_proj and boundary_integral_fluid (all class members), and
          // Proteus Newton evaluates F(u) then J(u) at the same iterate, so they
          // are current.
          proteus::pyarray<double>& M_dof = args.array<double>("M_dof");
          // FLOW-MODEL FLUXES, see calculateResidual; cin_num/cin_den are the
          // members the preceding residual call filled.
          proteus::pyarray<double>& edgeFlux = args.array<double>("edgeFlux");
          proteus::pyarray<double>& nodalBoundaryFlux = args.array<double>("nodalBoundaryFlux");
          const int USE_EDGE_FLUX = args.scalar<int>("USE_EDGE_FLUX");
          const double drho_du = rho_s - rho_f;
          // SCHEME B Jacobian, matching the cancelled residual term for term:
          //   R_i = ML M^n_i (c_i - c^n_i)/dt
          //         - sum_{j!=i} gamma_ij rho^n_j (c_j - c_i)
          //         + B^s_i - c_i B^f_i - ML R_diss_i
          // so
          //   J_ij = -gamma_ij rho^n_j                        (j != i, <= 0)
          //   J_ii = ML M^n_i/dt + sum_j gamma_ij rho^n_j
          //          - B^f_i - ML dR_diss/dc_i
          // gamma_ij = dLow_ij - K_ij >= 0 and rho^n_j > 0, so this is an
          // M-matrix UNCONDITIONALLY -- it does not depend on the row sums,
          // on dt, or on the velocity being divergence free.  gamma and rho^n
          // are frozen under differentiation (rho^n = rho(c^n) is genuinely
          // independent of the iterate: it is evaluated from the old-time
          // concentration, so freezing it is EXACT here, not an inexact-Newton
          // approximation).
          int ij=0;
          for (int i=0; i<numDOFs; i++)
            {
              const double ci = u_dof.data()[i];
              const double theta_i = fmax(theta_dof_proj[i], 1.0e-14);
              const double rho_ci = rho_f*(1.0 + (drho_du/rho_f)*ci);
              const double MLi = ML.data()[i];
              const double Mn_i = fmax(M_dof.data()[i], 1.0e-14);
              const double S_n_i = Sn_dof.data()[i];
              // dR_diss_i/dc_i = theta_i*k_d*S_n_i*[ drho_du*(c_sat-c_i) - rho(c_i) ]
              const double dRdiss_dci = theta_i*k_d*S_n_i*( drho_du*(c_sat - ci) - rho_ci );

              int diag_ij = -1;
              double sum_gamma_rho = 0.0;
              for (int offset=csrRowIndeces_DofLoops.data()[i]; offset<csrRowIndeces_DofLoops.data()[i+1]; offset++)
                {
                  int j = csrColumnOffsets_DofLoops.data()[offset];
                  if (i != j)
                    {
                      // rho(c^n_j) at the node, matching the residual's flux
                      // coefficient; the flow-model fluxes are mass fluxes (rho 1)
                      const double rhoj    = USE_EDGE_FLUX ? 1.0 : rho_dof_proj[j];
                      const double Fij     = USE_EDGE_FLUX ? edgeFlux.data()[ij] : 0.0;
                      const double Kij     = -0.5*Fij + TransportMatrix[ij] + DiffusionMatrix[ij];
                      const double Kji     =  0.5*Fij + TransposeTransportMatrix[ij] + DiffusionMatrix[ij];
                      const double dLow_ij = fmax(0.0, fmax(Kij, Kji));
                      const double gamma   = dLow_ij - Kij;
                      globalJacobian.data()[ij] += -gamma*rhoj;
                      sum_gamma_rho += gamma*rhoj;
                    }
                  else
                    diag_ij = ij;
                  ij += 1;
                }
              // inflow at a node with a boundary concentration: d/dc_i of
              // -max(B_i,0)*(c_in - c_i); zero where c_in is c_i itself
              const double dBoundary_dci = USE_EDGE_FLUX
                ? ((cin_den[i] > 0.0) ? fmax(nodalBoundaryFlux.data()[i], 0.0) : 0.0)
                : -boundary_integral_fluid[i];
              globalJacobian.data()[diag_ij] += MLi*Mn_i/dt
                                              + sum_gamma_rho
                                              + dBoundary_dci
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
                  // GROUP VARIABLE: the ImplicitEV boundary residual is now the
                  // adjoint of the w-operator, so differentiate w.r.t. w (using
                  // df/dw = v and a_w) and apply the chain rule dw/dc at the end.
                  // dw/dc is frozen at the quadrature trace -- the same inexact-
                  // Newton simplification used for the interior a_ij.
                  const double drho_du_jb = rho_f*((rho_s - rho_f)/rho_f);
                  const double dwdu_ext_j = fmax(rho_out_ext + u_ext*drho_du_jb, 1.0e-14);
                  double a_w_ext_j[nnz], zero_da_ext_j[nnz];
                  for (int ii=0; ii<a_rowptr.data()[nSpace]; ii++)
                    { a_w_ext_j[ii] = a_ext[ii]/dwdu_ext_j; zero_da_ext_j[ii] = 0.0; }
                  const double* v_w_ext_j = &ebqe_velocity_ext.data()[ebNE_kb_nSpace];
                  // Upwinded outflow face of calculateResidual: the solute flux is
                  // c_i (v.n) rho_trace with rho_trace from the frozen nodal
                  // rho(c^n), so its derivative is diagonal and replaces the
                  // consistent trace derivative (v.n) dw/du w_j.
                  double flow_w_j = 0.0;
                  for (int I=0;I<nSpace;I++)
                    flow_w_j += normal[I]*v_w_ext_j[I];
                  const bool outflow_upwind_j = flow_w_j >= 0.0 &&
                    ((isDOFBoundary_u.data()[ebNE_kb] == 1 && forceStrongConditions == 0) ||
                     (isDOFBoundary_u.data()[ebNE_kb] != 1 && isFluxBoundary_u.data()[ebNE_kb] != 1));
                  double rho_trace_j = 0.0;
                  if (outflow_upwind_j && !USE_EDGE_FLUX)
                    for (int j=0;j<nDOF_trial_element;j++)
                      rho_trace_j += u_trial_trace_ref.data()[ebN_local_kb*nDOF_test_element+j]
                                   * rho_dof_proj[u_l2g.data()[eN_nDOF_trial_element+j]];
                  for (int i=0;i<nDOF_test_element;i++)
                    for (int j=0;j<nDOF_trial_element;j++)
                      {
                        int ebN_local_kb_j=ebN_local_kb*nDOF_trial_element+j;
                        double advJacobian_ext = 0.0, diffJacobian_ext = 0.0;
                        if (USE_EDGE_FLUX)
                          {
                            // FLOW-MODEL FLUXES: only the dispersive flux is left
                            // at the boundary, and it acts on c directly.  On a
                            // weak-Dirichlet face that is the lumped penalty
                            // gamma_i (c_i - g) of calculateResidual: diagonal only,
                            // with a = theta*rho(c)*D frozen as zero_da freezes it
                            // in the IIPG Jacobian this replaces.  A flux BC is
                            // data and a free face carries none.
                            if (j == i &&
                                isDOFBoundary_u.data()[ebNE_kb] == 1 &&
                                isDiffusiveFluxBoundary_u.data()[ebNE_kb] == 0)
                              {
                                double max_a = 0.0;
                                for (int m=0; m<a_rowptr.data()[nSpace]; m++)
                                  max_a = fmax(max_a, a_ext[m]);
                                fluxJacobian_u_u[i][j] += max_a*ebqe_penalty_ext.data()[ebNE_kb]*u_test_dS[i];
                              }
                            continue;
                          }
                        if (outflow_upwind_j)
                          {
                            if (j == i)
                              fluxJacobian_u_u[i][j] += flow_w_j*rho_trace_j*u_test_dS[i];
                          }
                        else
                          exteriorNumericalAdvectiveFluxDerivative(isDOFBoundary_u.data()[ebNE_kb],isFluxBoundary_u.data()[ebNE_kb],forceStrongConditions,normal,v_w_ext_j,advJacobian_ext);
                        exteriorNumericalDiffusiveFluxDerivative(isDOFBoundary_u.data()[ebNE_kb],isDiffusiveFluxBoundary_u.data()[ebNE_kb],a_rowptr.data(),a_colind.data(),normal,a_w_ext_j,zero_da_ext_j,grad_u_ext,&u_grad_trial_trace[j*nSpace],u_trial_trace_ref.data()[ebN_local_kb_j],ebqe_penalty_ext.data()[ebNE_kb],diffJacobian_ext);
                        difffluxjacobian_ext = (advJacobian_ext*u_trial_trace_ref.data()[ebN_local_kb_j] + diffJacobian_ext)*dwdu_ext_j;
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
    proteus::pyarray<double>& M_dof_fct        = args.array<double>("M_dof_fct");
    proteus::pyarray<double>& M_low_fct        = args.array<double>("M_low_fct");
    proteus::pyarray<double>& theta_nodal_fct  = args.array<double>("theta_nodal");
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
      // Same substitution as the residual: the group variable w = m/theta below has
      // to divide by the theta the residual's rho_j came from, or the antidiffusion
      // does not undo the viscosity that was applied.  Zero = no flow model.
      if (theta_nodal_fct.data()[i] > 0.0) theta_dof[i] = theta_nodal_fct.data()[i];
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
    // The scheme-B branches advance S = M*c (m_dof[i] = M_i*u^n_i), so the limiter
    // must use that same mass; STAB 0/1 carry the EOS mass and keep it.  STAB 2/3/4
    // carry M themselves and divide the new-time states by M_low = M^{n+1}; STAB=5
    // reads Richards' M^n and uses it at both time levels (the cancelled form),
    // unless it advects with the flow model's fluxes, in which case it carries
    // M^{n+1} in M_low as well.
    const bool useM = (STABILIZATION_TYPE == STABILIZATION::EntropyViscosity or
                       STABILIZATION_TYPE == STABILIZATION::SmoothnessIndicator or
                       STABILIZATION_TYPE == STABILIZATION::Kuzmin or
                       STABILIZATION_TYPE == STABILIZATION::ImplicitEV);
    const bool implicitM = (STABILIZATION_TYPE == STABILIZATION::ImplicitEV);
    // ImplicitEV with the flow model's fluxes: M_low is the carried M^{n+1}, and
    // the operator dLow acts on c itself (the fluxes are already mass fluxes).
    const int USE_EDGE_FLUX = args.scalar<int>("USE_EDGE_FLUX");
    const bool newM_is_old = implicitM && !USE_EDGE_FLUX;
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
        const double Mi_old = fmax(M_dof_fct.data()[i], 1.0e-14);
        const double Mi_new = newM_is_old ? Mi_old : fmax(M_low_fct.data()[i], 1.0e-14);
        double mLowi  = useM ? Mi_new*uLowi : theta_i*rho_f*(1.0 + ((rho_s-rho_f)/rho_f)*uLowi)*uLowi;
        double solHmi = useM ? Mi_new*solHi : theta_i*rho_f*(1.0 + ((rho_s-rho_f)/rho_f)*solHi)*solHi;
        double solnmi = useM ? Mi_old*solni : theta_i*rho_f*(1.0 + ((rho_s-rho_f)/rho_f)*solni)*solni;
        double uDotLowi = (mLowi - solnmi)/dt;
        // GROUP VARIABLE w = rho(u)*u = m/theta.  dt_times_dH_minus_dL and dLow
        // are coefficients of the w-operator (the transport matrix is assembled
        // against w), so the DISSIPATION part of the antidiffusive flux must
        // difference w, not m.  The consistent-MASS part stays in m -- it is
        // M~_ij times d(m)/dt and carries its own theta.
        // ImplicitEV: M_dof is storage only (Richards' old-time mass), so
        // M/theta is not the density the residual applied dLow with; that is
        // rho(c^n) at the node, and soln IS c^n.
        // With the flow model's fluxes every branch's dLow acts on c itself.
        const double solnwi = USE_EDGE_FLUX ? solni
                                            : (implicitM ? rho_f*(1.0 + ((rho_s-rho_f)/rho_f)*solni)*solni
                                                         : solnmi/theta_i);
        const double wLowi  = USE_EDGE_FLUX ? uLowi : mLowi/theta_i;
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
            const double Mj_old = fmax(M_dof_fct.data()[j], 1.0e-14);
            const double Mj_new = newM_is_old ? Mj_old : fmax(M_low_fct.data()[j], 1.0e-14);
            double mLowj  = useM ? Mj_new*uLowj : theta_j*rho_f*(1.0 + ((rho_s-rho_f)/rho_f)*uLowj)*uLowj;
            double solHmj = useM ? Mj_new*solH.data()[j] : theta_j*rho_f*(1.0 + ((rho_s-rho_f)/rho_f)*solH.data()[j])*solH.data()[j];
            double solnmj = useM ? Mj_old*solnj : theta_j*rho_f*(1.0 + ((rho_s-rho_f)/rho_f)*solnj)*solnj;
            double uDotLowj = (mLowj - solnmj)/dt;
            const double solnwj = USE_EDGE_FLUX ? solnj
                                                : (implicitM ? rho_f*(1.0 + ((rho_s-rho_f)/rho_f)*solnj)*solnj
                                                             : solnmj/theta_j);   // group variable, old time
            const double wLowj  = USE_EDGE_FLUX ? uLowj : mLowj/theta_j;    // group variable, low order
            // i-th row of flux correction matrix
            if (STABILIZATION_TYPE == STABILIZATION::Kuzmin)
              {
                // Explicit Kuzmin antidiffusive flux: consistent-mass correction +
                // full removal of the symmetric graph viscosity dLow*(w_i-w_j).
                FluxCorrectionMatrix[ij] = dt*(MassMatrix.data()[ij]*(uDotLowi-uDotLowj)
                                               + dLow.data()[ij]*(wLowi-wLowj));
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
                // With the flow model's edge fluxes, dLow acts on c itself and was
                // applied implicitly to c^{n+1}, so its antidiffusion has to difference
                // c^{n+1} = uLow, not c^n.  On c^n it is a forward-Euler correction of a
                // backward-Euler step: unlimited it is unstable (coupled MMS, N=40: c
                // overshoots to 0.93 against an exact maximum of 0.80), and limited it
                // costs the max-norm rate (test/salinity/mms_coupled/MMS_PLAN.md).
                const double dw_ij = USE_EDGE_FLUX ? (uLowj - uLowi) : (solnwj - solnwi);
                FluxCorrectionMatrix[ij] = dt*MassMatrix.data()[ij]*(uDotLowi-uDotLowj)
                                           + dt_times_dH_minus_dL.data()[ij]*dw_ij;
              }
            else
              {
                double ML_minus_MC =
                  (LUMPED_MASS_MATRIX == 1 ? 0. : (i==j ? 1. : 0.)*lumped_volume - MassMatrix.data()[ij]);
                FluxCorrectionMatrix[ij] = ML_minus_MC * (solHmj-solnmj - (solHmi-solnmi))
                  + dt_times_dH_minus_dL.data()[ij]*(solnwj-solnwi);
              }
            Pposi += FluxCorrectionMatrix[ij]*((FluxCorrectionMatrix[ij] > 0) ? 1. : 0.);
            Pnegi += FluxCorrectionMatrix[ij]*((FluxCorrectionMatrix[ij] < 0) ? 1. : 0.);
            ij+=1;
          }//j
        // Bounds on the DERIVED variable c, mapped to the mass being limited.
        const double Qposi = lumped_volume*((useM ? Mi_new*maxi : theta_i*rho_f*(1.0 + ((rho_s-rho_f)/rho_f)*maxi)*maxi) - mLowi);
        const double Qnegi = lumped_volume*((useM ? Mi_new*mini : theta_i*rho_f*(1.0 + ((rho_s-rho_f)/rho_f)*mini)*mini) - mLowi);
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
        const double mLowi = useM ? fmax((newM_is_old ? M_dof_fct : M_low_fct).data()[i], 1.0e-14)*uLowi
                                  : theta_i*rho_f*(1.0 + ((rho_s-rho_f)/rho_f)*uLowi)*uLowi;
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
