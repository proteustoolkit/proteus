#ifndef ADR_H
#define ADR_H
#include <cmath>
#include <iomanip>
#include <iostream>
#include <set>
#include <stdexcept>
#include <string>
#include <map>
#include <valarray>
#include <vector>
#include "CompKernel.h"
#include "ModelFactory.h"
#include "equivalent_polynomials.h"
#include "mprans/ArgumentsDict.h"
#include "xtensor-python/pyarray.hpp"

namespace py = pybind11;

namespace proteus
{
	template <int nSpace, int nP_ifem, int nP, int nQ, int nEBQ>
	using GeneralizedFunctions = equivalent_polynomials::GeneralizedFunctions_mix<nSpace, nP_ifem, nP, nQ, nEBQ, true>;

	class cADR_base
	{
	public:
		virtual ~cADR_base() {}
		virtual void calculateResidual(arguments_dict &args) = 0;
		virtual void calculateJacobian(arguments_dict &args) = 0;
	};

	template <class CompKernelType,
			  int nSpace,
			  int nQuadraturePoints_element,
			  int nDOF_mesh_trial_element,
			  int nDOF_trial_element,
			  int nDOF_test_element,
			  int nQuadraturePoints_elementBoundary>
	class cADR : public cADR_base
	{
	public:
		std::set<int> ifem_boundaries, ifem_boundary_elements,
			cutfem_boundaries, cutfem_boundary_elements;
		std::valarray<bool> elementIsActive;
		const int nDOF_test_X_trial_element;
		CompKernelType ck;
		// Debug-only per-element split of elementResidual_u (ADR_DEBUG_ROW): 0 bulk, 1 Nitsche
		// penalty, 2 Nitsche consistency flux, 3 Nitsche adjoint consistency. Written every call.
		double dbgElem[4][nDOF_test_element];
		// TC_EDGE dead-cell experiment (env ADR_TC_EDGE_INCLUDE_DEAD): 0/unset = off, 1 = every
		// fully-solid Gamma_f-cut cell, 2 = only those sharing a node with a cell that has fluid.
		static int tcDeadMode()
		{
			const char *e = std::getenv("ADR_TC_EDGE_INCLUDE_DEAD");
			return e ? std::atoi(e) : 0;
		}
		static void tcBuildNodeFluid(std::vector<char> &nodeFluid, int nElements, const xt::pyarray<int> &l2g,
									 const xt::pyarray<double> &phi_s_nodes, size_t nNodes)
		{
			nodeFluid.assign(nNodes, 0);
			for (int e = 0; e < nElements; e++)
			{
				bool dead = true;
				for (int i = 0; i < nDOF_trial_element; i++)
					if (!(phi_s_nodes.data()[l2g.data()[e * nDOF_trial_element + i]] < -1.0e-8))
						dead = false;
				if (!dead)
					for (int i = 0; i < nDOF_trial_element; i++)
						nodeFluid[l2g.data()[e * nDOF_trial_element + i]] = 1;
			}
		}
		static bool tcDeadOK(int mode, const std::vector<char> &nodeFluid, const xt::pyarray<int> &l2g, int eN)
		{
			if (mode == 0)
				return false;
			if (mode == 1)
				return true;
			for (int i = 0; i < nDOF_trial_element; i++)
				if (nodeFluid[l2g.data()[eN * nDOF_trial_element + i]])
					return true;
			return false;
		}
		// nEBQ is sized for ALL faces (nDOF_mesh_trial_element * nQuadraturePoints_elementBoundary),
		// not one face, so the immersed-interface facet terms can evaluate va/vb at the quadrature
		// points of the face they are actually integrating -- see the xB_ref_faces comment in
		// calculateResidual for why passing the reference *boundary* points directly is wrong.
		// Degree k of the fitted H/ImH/D polynomials (GfType, the triply-cut composite fits, the edge
		// fits). Each cut-cell term is sum_q w_q*fit*F, exact when deg F <= k AND k + deg F <= the
		// element rule's degree. The Gamma_s Nitsche penalty / Robin terms are degree 2p. P1: k=3 with
		// the 12-point (degree-6) rule. P2: k=4 needs the 16-point (degree-8) rule (qOrder=8); with
		// the 12-point rule k=3 is kept, which is exact through degree 2p-1=3 and still rate-optimal.
		static const int nP_fit = (nDOF_trial_element == 6 && nQuadraturePoints_element >= 16) ? 4 : 3;
		using GfType = GeneralizedFunctions<nSpace, nDOF_trial_element, nP_fit, nQuadraturePoints_element, nDOF_mesh_trial_element * nQuadraturePoints_elementBoundary>;
		// Degree of the edge-restricted moment-fit Heaviside used by the edge terms (SCIFEM, the
		// triply-cut correction). The consistency terms are degree 2p-1 and the interior penalty 2p;
		// the 1D boundary rule (qOrder points) is exact far beyond nP_edge + 2p.
		static const int nP_edge = nP_fit;
		// Per-element cache of the equivalent-polynomial/IFEM reconstruction
		// (gf_s.calculate()/gf_f.calculate()): permutation, cut classification,
		// the IFEM basis coefficient solve, and the H/ImH/D + VA/VB (+
		// gradients) evaluated at every quadrature point. None of this depends
		// on the current solution u -- only on the level-set geometry
		// (element_phi_s/element_phi_f) and mua/mub/jf -- so it is only
		// recomputed when ifemGeometryGeneration advances (see
		// recomputeIFEMGeometry / markIFEMGeometryDirty on the Python side).
		// Interior (isBoundary=false) and boundary (isBoundary=true) evaluations
		// populate different internal storage within the same GfType object, so
		// they are tracked with separate generation/icase arrays.
		std::vector<GfType> gf_f_cache, gf_s_cache;
		std::vector<int> gf_s_interior_gen, gf_f_interior_gen, gf_f_boundary_gen;
		std::vector<int> gf_s_interior_icase, gf_f_interior_icase, gf_f_boundary_icase;
		// Composite equivalent polynomials for elements cut by BOTH level sets. The product
		// ImH_f*H_s is not moment-exact on such an element (see .notes/EQUIVALENT_POLYNOMIALS.md
		// section 7.1); these replace it with one fit per intersected region. Only populated when
		// icase_s == 0 && icase_f == 0, so every single-level-set element keeps the old path
		// bit-for-bit.
		static const int nCompMon = (nP_fit + 1) * (nP_fit + 2) / 2; // monomials up to degree nP_fit
		std::vector<double> comp_A_cache, comp_B_cache;
		std::vector<double> comp_DA_cache, comp_DB_cache;
		std::vector<char> comp_valid;
		// Gamma_f's OWN Dirac measure, split by Gamma_s's sign: comp_DFm = the masked (solid,
		// phi_s<0) part, comp_DFa = the active (fluid, phi_s>0) part. Needed for any Dirac-delta
		// term integrated over Gamma_f on a triply-cut element (e.g. the prescribed-flux-jump
		// source, [beta du/dn]*D_f) -- using the plain single-curve gf_f.D(0.,0.) there
		// double-counts whatever part of Gamma_f's chord falls inside the masked-out solid.
		// Tracked as its own validity flag, not folded into comp_valid: bulk area weights, Gamma_s's
		// own Dirac measure, and Gamma_f's own Dirac measure are three independently-fitted
		// quantities that can, in principle, succeed or fail on different degenerate geometries
		// (see .notes/EQUIVALENT_POLYNOMIALS.md's "track independent flags, not one" rule).
		std::vector<double> comp_DFm_cache, comp_DFa_cache;
		std::vector<char> comp_DF_valid;
		int ifemGeometryGeneration = 0;
		cADR() : nDOF_test_X_trial_element(nDOF_test_element * nDOF_trial_element), ck()
		{
		}
		inline void ensureIFEMCacheSized(int nElements_global)
		{
			if ((int)gf_f_cache.size() == nElements_global)
				return;
			gf_f_cache.assign(nElements_global, GfType());
			gf_s_cache.assign(nElements_global, GfType());
			gf_s_interior_gen.assign(nElements_global, -1);
			gf_f_interior_gen.assign(nElements_global, -1);
			gf_f_boundary_gen.assign(nElements_global, -1);
			gf_s_interior_icase.assign(nElements_global, 0);
			gf_f_interior_icase.assign(nElements_global, 0);
			gf_f_boundary_icase.assign(nElements_global, 0);
			comp_A_cache.assign((std::size_t)nElements_global * nCompMon, 0.0);
			comp_B_cache.assign((std::size_t)nElements_global * nCompMon, 0.0);
			comp_DA_cache.assign((std::size_t)nElements_global * nCompMon, 0.0);
			comp_DB_cache.assign((std::size_t)nElements_global * nCompMon, 0.0);
			comp_valid.assign(nElements_global, 0);
			comp_DFm_cache.assign((std::size_t)nElements_global * nCompMon, 0.0);
			comp_DFa_cache.assign((std::size_t)nElements_global * nCompMon, 0.0);
			comp_DF_valid.assign(nElements_global, 0);
			for (auto &gf : gf_f_cache)
				gf.useExact = true;
			for (auto &gf : gf_s_cache)
				gf.useExact = true;
		}

		inline void exteriorNumericalDiffusiveFlux(int *rowptr,
												   int *colind,
												   const int &isDOFBoundary,
												   const int &isDiffusiveFluxBoundary,
												   const double n[nSpace],
												   double *bc_a,
												   const double &bc_u,
												   const double &bc_flux,
												   double *a,
												   const double grad_potential[nSpace],
												   const double &u,
												   const double &penalty,
												   double &flux)
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
					for (int m = rowptr[I]; m < rowptr[I + 1]; m++)
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
				std::cerr << "warning, diffusion term with no boundary condition set, setting diffusive flux to 0.0" << std::endl;
				flux = 0.0;
			}
		}

		inline double ExteriorNumericalDiffusiveFluxJacobian(int *rowptr,
															 int *colind,
															 const int &isDOFBoundary,
															 const int &isDiffusiveFluxBoundary,
															 const double n[nSpace],
															 double *a,
															 const double &v,
															 const double grad_v[nSpace],
															 const double &penalty)
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

		inline void calculateSubgridError_tau(const double &elementDiameter,
											  const double &dmt,
											  const double dH[nSpace],
											  double &cfl,
											  double &tau)
		{
			double h;
			double nrm_v;
			double oneByAbsdt;
			h = elementDiameter;
			nrm_v = 0.0;
			for (int I = 0; I < nSpace; I++)
				nrm_v += dH[I] * dH[I];
			nrm_v = sqrt(nrm_v);
			cfl = nrm_v / h;
			oneByAbsdt = fabs(dmt);
			tau = 1.0 / (2.0 * nrm_v / h + oneByAbsdt + 1.0e-8);
		}

		inline void calculateSubgridError_tau(const double &Ct_sge,
											  const double G[nSpace * nSpace],
											  const double &A0,
											  const double Ai[nSpace],
											  double &tau_v,
											  double &cfl)
		{
			double v_d_Gv = 0.0;
			for (int I = 0; I < nSpace; I++)
				for (int J = 0; J < nSpace; J++)
					v_d_Gv += Ai[I] * G[I * nSpace + J] * Ai[J];
			tau_v = 1.0 / sqrt(Ct_sge * A0 * A0 + v_d_Gv + 1.0e-8);
		}

		inline void calculateNumericalDiffusion(const double &shockCapturingDiffusion,
												const double &elementDiameter,
												const double &strong_residual,
												const double grad_u[nSpace],
												double &numDiff)
		{
			double h;
			double num;
			double den;
			double n_grad_u;
			h = elementDiameter;
			n_grad_u = 0.0;
			for (int I = 0; I < nSpace; I++)
				n_grad_u += grad_u[I] * grad_u[I];
			num = shockCapturingDiffusion * 0.5 * h * fabs(strong_residual);
			den = sqrt(n_grad_u) + 1.0e-8;
			numDiff = num / den;
		}

		inline void exteriorNumericalAdvectiveFlux(const int &isDOFBoundary_u,
												   const int &isFluxBoundary_u,
												   const double n[nSpace],
												   const double &bc_u,
												   const double &bc_flux_u,
												   const double &u,
												   const double velocity[nSpace],
												   double &flux)
		{

			double flow = 0.0;
			for (int I = 0; I < nSpace; I++)
				flow += n[I] * velocity[I];
			if (isDOFBoundary_u == 1)
			{
				if (flow >= 0.0)
				{
					flux = u * flow;
					// flux = flow;
				}
				else
				{
					flux = bc_u * flow;
					// flux = flow;
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
					flux = u * flow;
				}
				else
				{
					flux = 0.0;
				}
			}
			// flux = flow;
		}

		inline void exteriorNumericalAdvectiveFluxDerivative(const int &isDOFBoundary_u,
															 const int &isFluxBoundary_u,
															 const double n[nSpace],
															 const double velocity[nSpace],
															 double &dflux)
		{
			double flow = 0.0;
			for (int I = 0; I < nSpace; I++)
			{
				flow += n[I] * velocity[I];
			}
			// double flow=n[0]*velocity[0]+n[1]*velocity[1]+n[2]*velocity[2];
			dflux = 0.0; // default to no flux
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

		inline void updateEmbeddedBoundaryTerms(const double embeddedBoundary_penalty,
												const double dV,
												double *embeddedBoundary_normal,
												const double u_s,
												const double u,
												const double grad_u[nSpace],
												const double a,
												double &r,
												double &dr,
												double &ham,
												double *dham,
												double *f,
												double *df,
												const double D_s)
		{
			// todo this doesn't look 1d/3d
			double outward_normal[nSpace];
			for (int I = 0; I < nSpace; I++)
				outward_normal[I] = -embeddedBoundary_normal[I];

			// diffusive flux
			for (int I = 0; I < nSpace; I++)
			{
				ham -= outward_normal[I] * grad_u[I];
				dham[I] -= D_s * a * outward_normal[I];
				// Nitsche adjoint consistency
				f[I] += D_s * a * outward_normal[I] * (u - u_s);
				df[I] += D_s * a * outward_normal[I];
			}
			ham *= D_s * a;

			// Nitsche Dirichlet penalty
			r += D_s * a * embeddedBoundary_penalty * (u - u_s);
			dr += D_s * a * embeddedBoundary_penalty;
		}

		inline void updateEmbeddedBoundaryRobinTerms(const double h_coeff,
													const double u_infinity,
													const double u,
													double &r,
													double &dr,
													const double D_s)
		{
			// Robin/convective condition  a * du/dn = -h * (u - u_infinity)  on Gamma_s (n the
			// fluid's own outward normal, into the solid): substitute this physical law directly
			// for the natural boundary term -int a du/dn v ds that integration by parts leaves
			// behind. Unlike updateEmbeddedBoundaryTerms above, a Robin law is already a natural
			// condition -- nothing is being enforced weakly -- so it needs no consistency,
			// symmetry, or penalty construction, only this one reaction-like term.
			r += D_s * h_coeff * (u - u_infinity);
			dr += D_s * h_coeff;
		}

		inline void updateImmersedBoundaryTerms(const double immersedBoundary_penalty,
												const double dV,
												double *immersedBoundary_normal,
												double x,
												double y,
												double z,
												const double u_s,
												const double u,
												const double grad_u[nSpace],
												const double a,
												double &r,
												double &dr,
												double &ham,
												double *dham,
												double *f,
												double *df,
												const double fluxJump,
												const double *fluxJumpVector,
												const double D_f)
		{
			// todo this doesn't look 1d/3d
			/*       double outward_normal[nSpace];
				  for (int I=0;I<nSpace;I++)
					outward_normal[I] = immersedBoundary_normal[I];



				  //diffusive flux
				  for (int I=0;I<nSpace;I++)
					{
					ham -= outward_normal[I] * grad_u[I];
					dham[I] -= D_f * a * outward_normal[I];
					//Nitsche adjoint consistency
					f[I] += D_f * a * outward_normal[I] * (u - u_s);
					df[I] += D_f * a * outward_normal[I];
					}
				  ham *= D_f * a;
				  //Nitsche Dirichlet penalty
				  r  += D_f*a*immersedBoundary_penalty * (u - u_s);
				  dr += D_f*a*immersedBoundary_penalty; */
			// std::cout << "D_f = " << D_f << std::endl;
			// Prescribed flux jump across the interface, supplied by the physics file as
			// [beta du/dn] = fluxJump(x) + fluxJumpVector(x) . n. The vector part carries
			// the cases whose jump depends on the interface normal. Both default to zero,
			// so a problem with no prescribed flux jump contributes nothing here.
			double jump_flux = fluxJump;
			for (int I = 0; I < nSpace; I++)
				jump_flux += fluxJumpVector[I] * immersedBoundary_normal[I];
			r += jump_flux * D_f;
			dr = 0.0;
			ham = 0.0;
			dham[0] = 0.0;
			dham[1] = 0.0;
			f[0] = 0.0;
			f[1] = 0.0;
			df[0] = 0.0;
			df[1] = 0.0;
		}

		inline void calculateElementResidual(int icase_f,
											 // nonzero when the buried-Gamma_f logic demoted this element:
											 // -1 = fluid all Omega^- (mua), +1 = all Omega^+ (mub)
											 const int demoted_f,
											 // element
											 xt::pyarray<double> &mesh_trial_ref,
											 xt::pyarray<double> &mesh_grad_trial_ref,
											 xt::pyarray<double> &mesh_dof,
											 xt::pyarray<int> &mesh_l2g,
											 xt::pyarray<double> &x_ref,
											 xt::pyarray<double> &dV_ref,
											 xt::pyarray<double> &u_trial_ref,
											 xt::pyarray<double> &u_grad_trial_ref,
											 xt::pyarray<double> &u_test_ref,
											 xt::pyarray<double> &u_grad_test_ref,
											 xt::pyarray<double> &elementDiameter,
											 xt::pyarray<double> &elementBoundaryDiameter,
											 xt::pyarray<double> &nodeDiametersArray,
											 xt::pyarray<double> &cfl,
											 double Ct_sge,
											 double sc_uref,
											 double sc_alpha,
											 double useMetrics,
											 // element boundary
											 xt::pyarray<double> &mesh_trial_trace_ref,
											 xt::pyarray<double> &mesh_grad_trial_trace_ref,
											 xt::pyarray<double> &dS_ref,
											 xt::pyarray<double> &u_trial_trace_ref,
											 xt::pyarray<double> &u_grad_trial_trace_ref,
											 xt::pyarray<double> &u_test_trace_ref,
											 xt::pyarray<double> &u_grad_test_trace_ref,
											 xt::pyarray<double> &normal_ref,
											 xt::pyarray<double> &boundaryJac_ref,
											 // physics
											 int nElements_global,
											 int nElementBoundaries_owned,
											 xt::pyarray<int> &u_l2g,
											 xt::pyarray<double> &u_dof,
											 xt::pyarray<int> &sd_rowptr,
											 xt::pyarray<int> &sd_colind,
											 xt::pyarray<double> &q_a,
											 const double *comp_A,
											 const double *comp_B,
											 const double *comp_DA,
											 const double *comp_DB,
											 const double *comp_DFa,
											 xt::pyarray<double> &q_v,
											 xt::pyarray<double> &q_r,
											 int lag_shockCapturingDiffusion,
											 double shockCapturingDiffusion,
											 xt::pyarray<double> &q_numDiff_u,
											 xt::pyarray<double> &q_numDiff_u_last,
											 int offset_u,
											 int stride_u,
											 xt::pyarray<double> &elementResidual_u,
											 int nExteriorElementBoundaries_global,
											 xt::pyarray<int> &exteriorElementBoundariesArray,
											 xt::pyarray<int> &elementBoundariesArray,
											 xt::pyarray<int> &elementBoundaryElementsArray,
											 xt::pyarray<int> &elementBoundaryLocalElementBoundariesArray,
											 xt::pyarray<double> &element_u,
											// Triply-cut correction: -1 if not applicable to this element (see
											// ADR.py Coefficients.initializeSDF and the comment on
											// triplyCut_overrideValue in calculateResidual).
											int triplyCut_maskedLocalNode_eN,
											double triplyCut_overrideValue_eN,
											 int eN,
											 const bool embeddedBoundary,
											 const double embeddedBoundary_penalty,
											 xt::pyarray<double> &embeddedBoundary_normal_q,
											 xt::pyarray<double> &embeddedBoundary_u_q,
											 xt::pyarray<double> &embeddedBoundary_u_inner_q,
											 xt::pyarray<double> &embeddedBoundary_u_outer_q,
											 const bool embeddedBoundary_robin,
											 const double embeddedBoundary_robin_h_inner,
											 const double embeddedBoundary_robin_h_outer,
											 xt::pyarray<double> &embeddedBoundary_robin_uInfinity_inner_q,
											 xt::pyarray<double> &embeddedBoundary_robin_uInfinity_outer_q,
											 xt::pyarray<double> &embeddedBoundary_sdf_q,
											 const bool immersedBoundary,
											 const double immersedBoundary_penalty,
											 xt::pyarray<double> &immersedBoundary_sdf_q,
											 xt::pyarray<double> &immersedBoundary_normal_q,
											 xt::pyarray<double> &immersedBoundary_u_q,
											 xt::pyarray<double> &immersedBoundary_fluxJump_q,
											 xt::pyarray<double> &immersedBoundary_fluxJumpVector_q,
											 xt::pyarray<double> &immersedBoundary_solutionJump_nodes,
											 double *element_phi_f,
											 bool &element_active,
											 std::valarray<bool> &elementIsActive,
											 double *JA,
											 double *JB,
											 const double fluxLiftJ,
											 double &L2_error,
											 double &Linfty_error,
											 double mua,
											 double mub,
											 xt::pyarray<double> &q_u_exact_inner,
											 xt::pyarray<double> &q_u_exact_outer,
											 const bool PG,
											 const bool pgElem)
		{
			// per-element cached equivalent-polynomial/IFEM reconstruction
			// (see ensureIFEMCacheSized / ifemGeometryGeneration)
			GfType &gf_f = gf_f_cache[eN];
			GfType &gf_s = gf_s_cache[eN];
			for (int i = 0; i < nDOF_test_element; i++)
			{
				elementResidual_u.data()[i] = 0.0;
				for (int b = 0; b < 4; b++)
					dbgElem[b][i] = 0.0;
			}
			// loop over quadrature points and compute integrands
			for (int k = 0; k < nQuadraturePoints_element; k++)
			{
				gf_s.set_quad(k);
				gf_f.set_quad(k);
				// compute indeces and declare local storage
				int eN_k = eN * nQuadraturePoints_element + k;
				int eN_k_3d = eN_k * 3;
				double h_phi;
				double u = 0.0;
				double grad_u[nSpace];
				double ua = 0.0;
				double grad_ua[nSpace];
				double ub = 0.0;
				double grad_ub[nSpace];
				double uja = 0.0;
				double grad_uja[nSpace];
				double ujb = 0.0;
				double grad_ujb[nSpace];
				double m = 0.0;
				double dm = 0.0;
				double f[nSpace];
				double df[nSpace];
				double f_s[nSpace] = {0., 0.};
				double df_s[nSpace] = {0., 0.};
				double ham_s = 0.0;
				double dham_s[nSpace] = {0., 0.};
				double f_f[nSpace] = {0., 0.};
				double df_f[nSpace] = {0., 0.};
				double ham_f = 0.0;
				double dham_f[nSpace] = {0., 0.};
				double m_t = 0.0;
				double dm_t = 0.0;
				double pdeResidual_u = 0.0;
				double Lstar_u[nDOF_test_element];
				double subgridError_u = 0.0;
				double tau = 0.0;
				double tau0 = 0.0;
				double tau1 = 0.0;
				double numDiff0 = 0.0;
				double numDiff1 = 0.0;
				double *a = NULL;
				double r = 0.0;
				double r_s = 0.0;
				// updateEmbeddedBoundaryTerms ACCUMULATES into its outputs (+=, -=, then ham *= D_s*a),
				// so one private zero-initialised set per side. Sharing them silently double-counts.
				double r_s_a = 0.0, dr_s_a = 0.0, ham_s_a = 0.0;
				double dham_s_a[nSpace] = {0., 0.}, f_s_a[nSpace] = {0., 0.}, df_s_a[nSpace] = {0., 0.};
				double r_s_b = 0.0, dr_s_b = 0.0, ham_s_b = 0.0;
				double dham_s_b[nSpace] = {0., 0.}, f_s_b[nSpace] = {0., 0.}, df_s_b[nSpace] = {0., 0.};
				double dr_s = 0.0;
				double r_f = 0.0;
				double dr_f = 0.0;
				double jac[nSpace * nSpace];
				double jacDet;
				double jacInv[nSpace * nSpace];
				double u_grad_trial[nDOF_trial_element * nSpace];
				double u_test_dV[nDOF_trial_element];
				double u_grad_test_dV[nDOF_test_element * nSpace];
				double ua_grad_trial[nDOF_trial_element * nSpace];
				double ua_test_dV[nDOF_trial_element];
				double ua_grad_test_dV[nDOF_test_element * nSpace];
				double ub_grad_trial[nDOF_trial_element * nSpace];
				double ub_test_dV[nDOF_trial_element];
				double ub_grad_test_dV[nDOF_test_element * nSpace];
				double dV;
				double x;
				double y;
				double z;
				double G[nSpace * nSpace];
				double G_dd_G;
				double tr_G;
				//
				// compute solution and gradients at quadrature points
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
											x,
											y,
											z);
				ck.calculateH_element(eN,
									  k,
									  nodeDiametersArray.data(),
									  mesh_l2g.data(),
									  mesh_trial_ref.data(),
									  h_phi);
				// get the physical integration weight
				dV = fabs(jacDet) * dV_ref.data()[k];
				// get the metric tensor and friends
				ck.calculateG(jacInv, G, G_dd_G, tr_G);
				// get the trial function gradients
				// std::cout << "Calculating gradTrialFromRef from calculateElementResidual()" << std::endl;
				ck.gradTrialFromRef(&u_grad_trial_ref.data()[k * nDOF_trial_element * nSpace], jacInv, u_grad_trial);
				// get the solution
				// std::cout << "element_u data: " << std::endl;
				ck.valFromElementDOF(element_u.data(), &u_trial_ref.data()[k * nDOF_trial_element], u);
				// get the solution gradients
				ck.gradFromElementDOF(element_u.data(), u_grad_trial, grad_u);
				// precalculate test function products with integration weights
				for (int j = 0; j < nDOF_trial_element; j++)
				{
					u_test_dV[j] = u_test_ref.data()[k * nDOF_trial_element + j] * dV;
					for (int I = 0; I < nSpace; I++)
					{
						u_grad_test_dV[j * nSpace + I] = u_grad_trial[j * nSpace + I] * dV; // cek warning won't work for Petrov-Galerkin
						//PGIFEM
						//ua_grad_test_dV[j * nSpace + I] = u_grad_trial[j * nSpace + I] * dV; // cek warning won't work for Petrov-Galerkin
						//ub_grad_test_dV[j * nSpace + I] = u_grad_trial[j * nSpace + I] * dV; // cek warning won't work for Petrov-Galerkin
					}
				}
				if (icase_f == 0)
				{
					double va[nDOF_trial_element], va_grad_trial[nDOF_trial_element * nSpace], vb[nDOF_trial_element], vb_grad_trial[nDOF_trial_element * nSpace];
					for (int i = 0; i < nDOF_trial_element; i++)
					{
						va[i] = gf_f.VA(i);
						//assert(fabs(va[i] - u_trial_ref.data()[k * nDOF_trial_element+i])< 1.0e-8);
						va_grad_trial[i * nSpace + 0] = gf_f.VA_x(i);
						//assert(fabs(va_grad_trial[i * nSpace + 0] - u_grad_trial[i * nSpace + 0]) < 1.0e-8);
						va_grad_trial[i * nSpace + 1] = gf_f.VA_y(i);
						//assert(fabs(va_grad_trial[i * nSpace + 1] - u_grad_trial[i * nSpace + 1]) < 1.0e-8);
						vb[i] = gf_f.VB(i);
						//assert(fabs(vb[i] - u_trial_ref.data()[k * nDOF_trial_element+i])< 1.0e-8);
						vb_grad_trial[i * nSpace + 0] = gf_f.VB_x(i);
						//assert(fabs(vb_grad_trial[i * nSpace + 0] - u_grad_trial[i * nSpace + 0]) < 1.0e-8);
						vb_grad_trial[i * nSpace + 1] = gf_f.VB_y(i);
						//assert(fabs(vb_grad_trial[i * nSpace + 0] - u_grad_trial[i * nSpace + 0]) < 1.0e-8);
						// std::cout << "\ni: " << i << ", va: " << va[i] << ", vb: " << vb[i] << std::endl;
						// std::cout << "i: " << i << ", va_x: " << va_grad_trial[i * nSpace + 0] << ", va_y: " << va_grad_trial[i * nSpace + 1] << std::endl;
						// std::cout << "i: " << i << ", vb_x: " << vb_grad_trial[i * nSpace + 0] << ", vb_y: " << vb_grad_trial[i * nSpace + 1] << std::endl;
					}	
					// std::cout << "--------------------------------------------------------------------------------" << std::endl;
					// std::cout << nDOF_trial_element << std::endl;
					// std::cout << "va: " << va[0] << ", " << va[1] << ", " << va[2] << ", " << va[3] << ", " << va[4] << ", " << va[5] << std::endl;
					// std::cout << "vb: " << vb[0] << ", " << vb[1] << ", " << vb[2] << ", " << vb[3] << ", " << vb[4] << ", " << vb[5] << std::endl;
					
					
					// Triply-cut: use the analytic boundary value for the masked node, not its
					// meaningless raw DOF (see ADR.py Coefficients.initializeSDF); no-op elsewhere.
					double element_u_ua[nDOF_trial_element];
					for (int i_tc = 0; i_tc < nDOF_trial_element; i_tc++)
						element_u_ua[i_tc] = element_u.data()[i_tc];
					if (triplyCut_maskedLocalNode_eN >= 0)
						element_u_ua[triplyCut_maskedLocalNode_eN] = triplyCut_overrideValue_eN;
					ck.valFromElementDOF(element_u_ua, va, ua);
					ck.gradFromElementDOF(element_u_ua, va_grad_trial, grad_ua);
					ck.valFromElementDOF(element_u_ua, vb, ub);
					ck.gradFromElementDOF(element_u_ua, vb_grad_trial, grad_ub);
					ck.valFromElementDOF(JA, va, uja);
					ck.gradFromElementDOF(JA, va_grad_trial, grad_uja);
					ck.valFromElementDOF(JB, vb, ujb);
					ck.gradFromElementDOF(JB, vb_grad_trial, grad_ujb);
					// Flux-jump lift J_f*psi, added once as known data (zero unless Gamma_f cuts the cell
					// and a flux jump is prescribed). psi is the unit lift of the basis constructor.
					uja += fluxLiftJ * gf_f.PSIA();
					ujb += fluxLiftJ * gf_f.PSIB();
					grad_uja[0] += fluxLiftJ * gf_f.PSIA_x();
					grad_uja[1] += fluxLiftJ * gf_f.PSIA_y();
					grad_ujb[0] += fluxLiftJ * gf_f.PSIB_x();
					grad_ujb[1] += fluxLiftJ * gf_f.PSIB_y();
					if (getenv("ADR_DEBUG_JAJB3") && eN == atoi(getenv("ADR_DEBUG_JAJB3")))
					{
						fprintf(stderr, "ADR_DEBUG_JAJB3 eN=%d k=%d maskedLocalNode=%d "
								"vb_grad=[(%.6f,%.6f),(%.6f,%.6f),(%.6f,%.6f)] "
								"va_grad=[(%.6f,%.6f),(%.6f,%.6f),(%.6f,%.6f)] "
								"JB_node_contrib=[(%.6e,%.6e),(%.6e,%.6e),(%.6e,%.6e)] "
								"JA_node_contrib=[(%.6e,%.6e),(%.6e,%.6e),(%.6e,%.6e)] "
								"grad_ujb_total=(%.6e,%.6e) grad_uja_total=(%.6e,%.6e)\n",
								eN, k, triplyCut_maskedLocalNode_eN,
								vb_grad_trial[0 * nSpace + 0], vb_grad_trial[0 * nSpace + 1],
								vb_grad_trial[1 * nSpace + 0], vb_grad_trial[1 * nSpace + 1],
								vb_grad_trial[2 * nSpace + 0], vb_grad_trial[2 * nSpace + 1],
								va_grad_trial[0 * nSpace + 0], va_grad_trial[0 * nSpace + 1],
								va_grad_trial[1 * nSpace + 0], va_grad_trial[1 * nSpace + 1],
								va_grad_trial[2 * nSpace + 0], va_grad_trial[2 * nSpace + 1],
								JB[0] * vb_grad_trial[0 * nSpace + 0], JB[0] * vb_grad_trial[0 * nSpace + 1],
								JB[1] * vb_grad_trial[1 * nSpace + 0], JB[1] * vb_grad_trial[1 * nSpace + 1],
								JB[2] * vb_grad_trial[2 * nSpace + 0], JB[2] * vb_grad_trial[2 * nSpace + 1],
								JA[0] * va_grad_trial[0 * nSpace + 0], JA[0] * va_grad_trial[0 * nSpace + 1],
								JA[1] * va_grad_trial[1 * nSpace + 0], JA[1] * va_grad_trial[1 * nSpace + 1],
								JA[2] * va_grad_trial[2 * nSpace + 0], JA[2] * va_grad_trial[2 * nSpace + 1],
								grad_ujb[0], grad_ujb[1], grad_uja[0], grad_uja[1]);
						// uja/ujb are affine (constant gradient), so extrapolate their VALUE from
						// this quadrature point straight to each node's own physical position --
						// answers "what does the jump lift evaluate to AT the masked node itself".
						for (int i_n = 0; i_n < nDOF_trial_element; i_n++)
						{
							double xi_n = mesh_dof.data()[u_l2g.data()[eN * nDOF_trial_element + i_n] * 3 + 0];
							double yi_n = mesh_dof.data()[u_l2g.data()[eN * nDOF_trial_element + i_n] * 3 + 1];
							double uja_at_i = uja + grad_uja[0] * (xi_n - x) + grad_uja[1] * (yi_n - y);
							double ujb_at_i = ujb + grad_ujb[0] * (xi_n - x) + grad_ujb[1] * (yi_n - y);
							fprintf(stderr, "ADR_DEBUG_JAJB3_NODE eN=%d node=%d x=%.10f y=%.10f "
									"uja_at_node=%.10e ujb_at_node=%.10e is_masked=%d\n",
									eN, i_n, xi_n, yi_n, uja_at_i, ujb_at_i,
									i_n == triplyCut_maskedLocalNode_eN ? 1 : 0);
						}
					}
					for (int i = 0; i < nDOF_test_element; i++)
					{
						ua_test_dV[i] = va[i] * dV;
						ub_test_dV[i] = vb[i] * dV;
						for (int I = 0; I < nSpace; I++)
						{
							ua_grad_test_dV[i * nSpace + I] = va_grad_trial[i * nSpace + I] * dV;
							ub_grad_test_dV[i * nSpace + I] = vb_grad_trial[i * nSpace + I] * dV;
						}
					}
					for (int j = 0; j < nDOF_trial_element; j++)
					{
						for (int I = 0; I < nSpace; I++)
						{
							ua_grad_trial[j * nSpace + I] = va_grad_trial[j * nSpace + I];
							ub_grad_trial[j * nSpace + I] = vb_grad_trial[j * nSpace + I];
						}
					}
					if (PG && pgElem)
					{
						// Petrov-Galerkin: test with ordinary P1 hat functions instead of
						// the enriched va/vb branches; trial/solution side is unchanged.
						for (int i = 0; i < nDOF_test_element; i++)
						{
							ua_test_dV[i] = u_test_dV[i];
							ub_test_dV[i] = u_test_dV[i];
							for (int I = 0; I < nSpace; I++)
							{
								ua_grad_test_dV[i * nSpace + I] = u_grad_test_dV[i * nSpace + I];
								ub_grad_test_dV[i * nSpace + I] = u_grad_test_dV[i * nSpace + I];
							}
						}
					}
				}
				//
				// calculate pde coefficients at quadrature points
				//
				// evaluateCoefficients();
				// just set from pre-evaluated quadrature point values for now
				a = &q_a.data()[eN_k * sd_rowptr.data()[nSpace]];
				// On a demoted element the fluid is entirely one phase, but the pre-evaluated
				// pointwise data still switches branch across the buried Gamma_f -- which lies inside
				// the solid part of this element. H_s is only moment-exact against polynomials, and a
				// coefficient that jumps mid-element is not one, so those quadrature points inject a
				// contribution from a fluid that is not physically present. Substitute the surviving
				// phase's constants, which makes the integrand polynomial again and H_s exact.
				double a_dem[nSpace * nSpace];
				double u_s_dem = embeddedBoundary_u_q.data()[eN_k];
				if (demoted_f != 0)
				{
					for (int I = 0; I < nSpace * nSpace; I++) a_dem[I] = 0.0;
					const double mu_dem = (demoted_f < 0) ? mua : mub;
					for (int I = 0; I < nSpace; I++) a_dem[I * nSpace + I] = mu_dem;
					a = a_dem;
					u_s_dem = (demoted_f < 0) ? embeddedBoundary_u_inner_q.data()[eN_k]
						                          : embeddedBoundary_u_outer_q.data()[eN_k];
				}
				r = q_r.data()[eN_k];
				for (int I = 0; I < nSpace; I++)
				{
					f[I] = q_v.data()[eN_k * nSpace + I] * u;
					df[I] = q_v.data()[eN_k * nSpace + I];
				}
				const double H_s = gf_s.H(0., 0.);
				const double D_s = gf_s.D(0., 0.);
				// Per-side Dirac for the solid surface. Weighting the single-cut D_s by a region fit
				// (ImH_f * D_s) is the product-of-two-fits error again, and a large one -- about 31% on
				// the segment length. Declared here because the Nitsche terms below consume it.
				// Defaults to D_s, so elements cut by only one level set are untouched.
				double D_s_a = D_s, D_s_b = D_s;
				if (demoted_f != 0)
				{
					// The whole solid surface in this element borders the one surviving fluid region.
					D_s_a = (demoted_f < 0) ? D_s : 0.0;
					D_s_b = (demoted_f < 0) ? 0.0 : D_s;
				}
				else if (comp_DA != nullptr)
				{
					const double xr_d = x_ref.data()[k * 3 + 0], yr_d = x_ref.data()[k * 3 + 1];
					D_s_a = equivalent_polynomials::composite::evaluate<nP_fit>(comp_DA, xr_d, yr_d);
					D_s_b = equivalent_polynomials::composite::evaluate<nP_fit>(comp_DB, xr_d, yr_d);
				}
				if (embeddedBoundary)
				{
					double level_set_normal[nSpace];
					double sign = 0.0;
					double norm_exact = 0.0, norm_cut = 0.0;
					for (int I = 0; I < nSpace; I++)
					{
						sign += embeddedBoundary_normal_q.data()[eN_k_3d + I] * gf_s.get_normal()[I];
						level_set_normal[I] = gf_s.get_normal()[I];
						norm_cut += level_set_normal[I] * level_set_normal[I];
						norm_exact += embeddedBoundary_normal_q.data()[eN_k_3d + I] * embeddedBoundary_normal_q.data()[eN_k_3d + I];
					}
					assert(std::fabs(1.0 - norm_cut) < 1.0e-8);
					assert(std::fabs(1.0 - norm_exact) < 1.0e-8);
					if (sign < 0.0)
						for (int I = 0; I < nSpace; I++)
							level_set_normal[I] *= -1.0;
					if (icase_f == 0)
					{
						// The solution here is ua on the mua side and ub on the mub side; the plain `u` is
						// neither. Handing `u` to the penalty makes (u - u_s) nonzero even when the discrete
						// solution is exactly right, and the O(100/h) penalty amplifies that.

						if (embeddedBoundary_robin)
						{
							// Robin needs no jump-corrected gradient (no flux term of its own),
							// but the jump-corrected VALUE still matters: u_infinity was derived
							// from the exact ua/ub, not from ua+uja's jump-free counterpart.
							// Deliberately NOT embeddedBoundary_u_inner_q/_outer_q -- those are
							// also read directly (in ADR.py's initializeSDF) to pin masked
							// triply-cut DOFs to the true solution, unrelated to Gamma_s's BC
							// type; reusing them here would silently repoint that pinning too.
							updateEmbeddedBoundaryRobinTerms(embeddedBoundary_robin_h_inner,
								embeddedBoundary_robin_uInfinity_inner_q.data()[eN_k], ua + uja, r_s_a, dr_s_a, D_s_a);
							updateEmbeddedBoundaryRobinTerms(embeddedBoundary_robin_h_outer,
								embeddedBoundary_robin_uInfinity_outer_q.data()[eN_k], ub + ujb, r_s_b, dr_s_b, D_s_b);
						}
						else
						{
							double ga[nSpace], gb[nSpace];
							for (int I = 0; I < nSpace; I++)
							{
								ga[I] = grad_ua[I] + grad_uja[I];
								gb[I] = grad_ub[I] + grad_ujb[I];
							}
							updateEmbeddedBoundaryTerms(embeddedBoundary_penalty / h_phi, dV, level_set_normal,
								embeddedBoundary_u_inner_q.data()[eN_k], ua + uja, ga, mua,
								r_s_a, dr_s_a, ham_s_a, dham_s_a, f_s_a, df_s_a, D_s_a);
							updateEmbeddedBoundaryTerms(embeddedBoundary_penalty / h_phi, dV, level_set_normal,
								embeddedBoundary_u_outer_q.data()[eN_k], ub + ujb, gb, mub,
								r_s_b, dr_s_b, ham_s_b, dham_s_b, f_s_b, df_s_b, D_s_b);
						}
					}
					else
					{
						// icase_f != 0: this element is wholly on one side of Gamma_f (icase_f < 0
						// water/mua, icase_f > 0 air/mub -- the same sign convention as demoted_f
						// and mu_dem above), so a single scalar h and a single per-phase target
						// apply throughout -- NOT u_s_dem, which (like embeddedBoundary_u_inner_q
						// above) carries the true Dirichlet solution, not the Robin target.
						if (embeddedBoundary_robin)
						{
							const double h_here = (icase_f < 0) ? embeddedBoundary_robin_h_inner
								                                 : embeddedBoundary_robin_h_outer;
							const double u_inf_here = (icase_f < 0)
								? embeddedBoundary_robin_uInfinity_inner_q.data()[eN_k]
								: embeddedBoundary_robin_uInfinity_outer_q.data()[eN_k];
							updateEmbeddedBoundaryRobinTerms(h_here, u_inf_here, u, r_s, dr_s, D_s);
						}
						else
						{
							updateEmbeddedBoundaryTerms(embeddedBoundary_penalty / h_phi, dV, level_set_normal,
								u_s_dem, u, grad_u, a[0],
								r_s, dr_s, ham_s, dham_s, f_s, df_s, D_s);
						}
					}
				}
				const double ImH_f = gf_f.ImH(0., 0.);
				const double H_f = gf_f.H(0., 0.);
				// Region weights for the bulk terms. On an element cut by both level sets these
				// come from the composite fit; otherwise they are the usual products, unchanged.
				double wA = ImH_f * H_s, wB = H_f * H_s;
				if (demoted_f != 0)
				{
					// Only one fluid region exists here; give it the whole fluid weight and the other none.
					wA = (demoted_f < 0) ? H_s : 0.0;
					wB = (demoted_f < 0) ? 0.0 : H_s;
				}
				else if (comp_A != nullptr)
				{
					const double xr = x_ref.data()[k * 3 + 0], yr = x_ref.data()[k * 3 + 1];
					wA = equivalent_polynomials::composite::evaluate<nP_fit>(comp_A, xr, yr);
					wB = equivalent_polynomials::composite::evaluate<nP_fit>(comp_B, xr, yr);
				}
				double D_f = gf_f.D(0., 0.);
				if (comp_DFa != nullptr)
				{
					// Gamma_f's own Dirac measure, restricted to the part still inside the active
					// fluid domain (see comp_DFa_cache's declaration) -- the plain, single-curve
					// gf_f.D(0.,0.) above double-counts whatever part of Gamma_f's chord falls
					// inside Gamma_s's masked-out solid on a triply-cut element.
					const double xr_f = x_ref.data()[k * 3 + 0], yr_f = x_ref.data()[k * 3 + 1];
					D_f = equivalent_polynomials::composite::evaluate<nP_fit>(comp_DFa, xr_f, yr_f);
				}
				// A demoted element's Gamma_f lies entirely inside the solid: no part of it borders
				// fluid, so the flux-jump load must vanish exactly there (any fit would leave a
				// spurious source).
				if (demoted_f != 0)
					D_f = 0.0;
				if (getenv("ADR_DEBUG_ELEM") && eN == atoi(getenv("ADR_DEBUG_ELEM")))
				{
					fprintf(stderr, "ADR_DEBUG_ELEM eN=%d k=%d xi=%.10f eta=%.10f x=%.10f y=%.10f "
							"D_f=%.10e dV=%.10e D_f_dV=%.10e comp_DFa_null=%d\n",
							eN, k, x_ref.data()[k * 3 + 0], x_ref.data()[k * 3 + 1], x, y,
							D_f, dV, D_f * dV, comp_DFa == nullptr ? 1 : 0);
				}
				if (getenv("ADR_DEBUG_JAJB2") && eN == atoi(getenv("ADR_DEBUG_JAJB2")))
				{
					fprintf(stderr, "ADR_DEBUG_JAJB2 eN=%d k=%d x=%.10f y=%.10f dV=%.10e "
							"JA=[%.6f,%.6f,%.6f] JB=[%.6f,%.6f,%.6f] "
							"uja=%.10e ujb=%.10e grad_uja=(%.10e,%.10e) grad_ujb=(%.10e,%.10e) "
							"ua=%.10e ub=%.10e H_s=%.6f ImH_f=%.6f H_f=%.6f wA=%.6f wB=%.6f "
							"wA_raw=%.6f wB_raw=%.6f D_s=%.10e D_s_a=%.10e D_s_b=%.10e\n",
							eN, k, x, y, dV,
							JA[0], JA[1], JA[2], JB[0], JB[1], JB[2],
							uja, ujb, grad_uja[0], grad_uja[1], grad_ujb[0], grad_ujb[1],
							ua, ub, H_s, ImH_f, H_f, wA, wB, ImH_f * H_s, H_f * H_s,
							D_s, D_s_a, D_s_b);
				}
				// if ( H_s*ImH_f != 0.0 || D_s != 0.0 || D_f != 0.0) //for two embedded interfaces
				if (H_s != 0.0 || D_s != 0.0 || D_f != 0.0) // for one embedded interface and one immersed interface
				{
					element_active = true;
					elementIsActive[eN] = true;
				}
				if (immersedBoundary)
				{
					double level_set_normal[nSpace];
					double sign = 0.0;
					double norm_exact = 0.0, norm_cut = 0.0;
					for (int I = 0; I < nSpace; I++)
					{
						sign += immersedBoundary_normal_q.data()[eN_k_3d + I] * gf_f.get_normal()[I];
						level_set_normal[I] = gf_f.get_normal()[I];
						norm_cut += level_set_normal[I] * level_set_normal[I];
						norm_exact += immersedBoundary_normal_q.data()[eN_k_3d + I] * immersedBoundary_normal_q.data()[eN_k_3d + I];
					}
					assert(std::fabs(1.0 - norm_cut) < 1.0e-8);
					assert(std::fabs(1.0 - norm_exact) < 1.0e-8);
					if (sign < 0.0)
					  for (int I = 0; I < nSpace; I++)
					    level_set_normal[I] *= -1.0;
					updateImmersedBoundaryTerms(immersedBoundary_penalty / h_phi, // penalty,
												dV,
												level_set_normal,
												x,
												y,
												z,
												immersedBoundary_u_q.data()[eN_k],
												u,
												grad_u,
												a[0], // assume scalar diffusion for now
												r_f,
												dr_f,
												ham_f,
												dham_f,
												f_f,
												df_f,
												immersedBoundary_fluxJump_q.data()[eN_k],
												&immersedBoundary_fluxJumpVector_q.data()[eN_k_3d],
												// Gamma_f's jump physics only applies in the fluid: on an element that
												// sits (locally, at this quadrature point) entirely inside the embedded
												// solid, H_s==0 and D_f alone (gf_f.D, or the raw single-curve fallback
												// when comp_DFa isn't available) does not know that -- it only encodes
												// where Gamma_f itself is, not whether the fluid actually exists there.
												// Without this factor an element wholly inside the solid but crossed by
												// Gamma_f gets a fixed, u-independent Dirac source with nothing in the
												// Jacobian to balance it (the bulk terms are already correctly zero via
												// wA=wB=0), so Newton can never drive that row's residual to zero.
												// On a triply-cut cell comp_DFa is already restricted to the fluid part of
												// Gamma_f; multiplying it by the H_s fit again would restrict twice (a
												// product of two fits), so H_s is applied only to the single-curve D_f.
												(comp_DFa != nullptr) ? D_f : D_f * H_s);
				}
				//
				// moving mesh
				//
				/* double mesh_velocity[3]; */
				/* mesh_velocity[0] = xt; */
				/* mesh_velocity[1] = yt; */
				/* mesh_velocity[2] = zt; */
				/* for (int I=0;I<nSpace;I++) */
				/*   { */
				/*     f[I] -= MOVING_DOMAIN*m*mesh_velocity[I]; */
				/*     df[I] -= MOVING_DOMAIN*dm*mesh_velocity[I]; */
				/*   } */
				//
				// calculate time derivative at quadrature points
				//
				/* ck.bdf(alphaBDF, */
				/*          q_m_betaBDF.data()[eN_k], */
				/*          m, */
				/*          dm, */
				/*          m_t, */
				/*          dm_t); */
				//
				// calculate subgrid error (strong residual and adjoint)
				//
				// calculate strong residual
				pdeResidual_u = ck.Advection_strong(df, grad_u) + ck.Reaction_strong(r); // ck.Mass_strong(m_t) + ck.Advection_strong(df,grad_u) + ck.Reaction_strong(r);
				// calculate adjoint
				for (int i = 0; i < nDOF_test_element; i++)
				{
					// int eN_k_i_nSpace = (eN_k*nDOF_trial_element+i)*nSpace;
					// Lstar_u[i]  = ck.Advection_adjoint(df,&u_grad_test_dV.data()[eN_k_i_nSpace]);
					int i_nSpace = i * nSpace;
					Lstar_u[i] = ck.Advection_adjoint(df, &u_grad_test_dV[i_nSpace]);
				}
				// calculate tau and tau*Res
				calculateSubgridError_tau(elementDiameter.data()[eN], dm_t, df, cfl.data()[eN_k], tau0);
				calculateSubgridError_tau(Ct_sge,
										  G,
										  dm_t,
										  df,
										  tau1,
										  cfl.data()[eN_k]);

				tau = useMetrics * tau1 + (1.0 - useMetrics) * tau0;

				subgridError_u = -tau * pdeResidual_u;
				//
				// calculate shock capturing diffusion
				//
				ck.calculateNumericalDiffusion(shockCapturingDiffusion, elementDiameter.data()[eN], pdeResidual_u, grad_u, numDiff0);
				ck.calculateNumericalDiffusion(shockCapturingDiffusion, sc_uref, sc_alpha, G, G_dd_G, pdeResidual_u, grad_u, numDiff1);
				q_numDiff_u.data()[eN_k] = useMetrics * numDiff1 + (1.0 - useMetrics) * numDiff0;
				//
				// update element residual
				//
				// Leveque & Li 1994, Examples 1, 3, 4, PWC, PWL, PWQ, PWcubic
				double a_loc[nSpace * nSpace];
				for (int I = 0; I < nSpace * nSpace; I++) a_loc[I] = 0.0;

				for (int i = 0; i < nDOF_test_element; i++)
				{
					int i_nSpace = i * nSpace;
					const double dbg_e0 = elementResidual_u.data()[i];
					if (icase_f == 0)
					{
						if(!gf_f.exact.edge  && !gf_f.exact.corner)//full cut or cut on boundary of negative cell
					    {
							for (int I = 0; I < nSpace; I++) a_loc[I * nSpace + I] = mua;
							elementResidual_u.data()[i] += wA * (ck.Advection_weak(f, &ua_grad_test_dV[i_nSpace]) +
							ck.Diffusion_weak(sd_rowptr.data(), sd_colind.data(), a_loc, grad_ua, &ua_grad_test_dV[i_nSpace]) +
							ck.Diffusion_weak(sd_rowptr.data(), sd_colind.data(), a_loc, grad_uja, &ua_grad_test_dV[i_nSpace]) +
							ck.Reaction_weak(r, ua_test_dV[i]) +
							ck.NumericalDiffusion(q_numDiff_u_last.data()[eN_k], grad_ua, &ua_grad_test_dV[i_nSpace]));

							for (int I = 0; I < nSpace; I++) a_loc[I * nSpace + I] = mub;
							elementResidual_u.data()[i] += wB * (ck.Advection_weak(f, &ub_grad_test_dV[i_nSpace]) +
							ck.Diffusion_weak(sd_rowptr.data(), sd_colind.data(), a_loc, grad_ub, &ub_grad_test_dV[i_nSpace]) +
							ck.Diffusion_weak(sd_rowptr.data(), sd_colind.data(), a_loc, grad_ujb, &ub_grad_test_dV[i_nSpace]) +
							ck.Reaction_weak(r, ub_test_dV[i]) +
							ck.NumericalDiffusion(q_numDiff_u_last.data()[eN_k], grad_ub, &ub_grad_test_dV[i_nSpace]));
							// Flux-jump load int_Gamma_f J_f v: D_f times the IFE test function, NOT weighted by
							// wA/wB. On Gamma_f va = vb = v, and D_f integrates va and vb (polynomials of degree
							// <= nP_fit) exactly; weighting by the Heaviside fits would multiply two fits and
							// move part of the load between branch-a and branch-b nodes.
							elementResidual_u.data()[i] += r_f * 0.5 * (ua_test_dV[i] + ub_test_dV[i]);
						}
						else if (gf_f.exact.edge == -1 || gf_f.exact.corner == -1)
						{
							for (int I = 0; I < nSpace; I++) a_loc[I * nSpace + I] = mua;
							elementResidual_u.data()[i] += wA * (ck.Advection_weak(f, &ua_grad_test_dV[i_nSpace]) +
							ck.Diffusion_weak(sd_rowptr.data(), sd_colind.data(), a_loc, grad_ua, &ua_grad_test_dV[i_nSpace]) +
							ck.Diffusion_weak(sd_rowptr.data(), sd_colind.data(), a_loc, grad_uja, &ua_grad_test_dV[i_nSpace]) +
							ck.Reaction_weak(r, ua_test_dV[i]) +
							ck.NumericalDiffusion(q_numDiff_u_last.data()[eN_k], grad_ua, &ua_grad_test_dV[i_nSpace]));
							elementResidual_u.data()[i] += r_f * ua_test_dV[i];
						}
						else if (gf_f.exact.edge == 1 || gf_f.exact.corner == 1)
						{
							for (int I = 0; I < nSpace; I++) a_loc[I * nSpace + I] = mub;
							elementResidual_u.data()[i] += wB * (ck.Advection_weak(f, &ub_grad_test_dV[i_nSpace]) +
								ck.Diffusion_weak(sd_rowptr.data(), sd_colind.data(), a_loc, grad_ub, &ub_grad_test_dV[i_nSpace]) +
								ck.Diffusion_weak(sd_rowptr.data(), sd_colind.data(), a_loc, grad_ujb, &ub_grad_test_dV[i_nSpace]) +
								ck.Reaction_weak(r, ub_test_dV[i]) +
								ck.NumericalDiffusion(q_numDiff_u_last.data()[eN_k], grad_ub, &ub_grad_test_dV[i_nSpace]));
							elementResidual_u.data()[i] += r_f * ub_test_dV[i];
						}
						else assert(false && "Invalid gf_f.exact.edge/corner values. Should be -1, 0 or +1.");
					}
					else
					{

						elementResidual_u.data()[i] += H_s * (ck.Advection_weak(f, &u_grad_test_dV[i_nSpace]) +
															  ck.Diffusion_weak(sd_rowptr.data(), sd_colind.data(), a, grad_u, &u_grad_test_dV[i_nSpace]) +
															  ck.Reaction_weak(r, u_test_dV[i]) +
															  ck.SubgridError(subgridError_u, Lstar_u[i]) +
															  ck.NumericalDiffusion(q_numDiff_u_last.data()[eN_k], grad_u, &u_grad_test_dV[i_nSpace]));
					}
					dbgElem[0][i] += elementResidual_u.data()[i] - dbg_e0;
					if (embeddedBoundary)
					{
						if (gf_s.exact.edge >= 0 && !gf_s.exact.corner)
						{
							if (icase_f == 0)
							{
								// Test against the reconstructed basis the bulk uses, split by which
								// fluid region each part of the solid surface sits in. ImH_f + H_f = 1,
								// so the total contribution is preserved.
								// Each side already carries its own piece of the solid surface via
								// D_s_a / D_s_b, so no ImH_f / H_f weighting here.
								elementResidual_u.data()[i] +=
									(ck.Advection_weak(f_s_a, &ua_grad_test_dV[i_nSpace]) +
									 ck.Reaction_weak(r_s_a, ua_test_dV[i]) +
									 ck.Hamiltonian_weak(ham_s_a, ua_test_dV[i])) +
									(ck.Advection_weak(f_s_b, &ub_grad_test_dV[i_nSpace]) +
									 ck.Reaction_weak(r_s_b, ub_test_dV[i]) +
									 ck.Hamiltonian_weak(ham_s_b, ub_test_dV[i]));
							}
							else
							{
								elementResidual_u.data()[i] += (ck.Advection_weak(f_s, &u_grad_test_dV[i_nSpace]) +
																ck.Reaction_weak(r_s, u_test_dV[i]) +
																ck.Hamiltonian_weak(ham_s, u_test_dV[i]));
							}
						}
					}
					if (embeddedBoundary && gf_s.exact.edge >= 0 && !gf_s.exact.corner)
					{
						if (icase_f == 0)
						{
							dbgElem[1][i] += ck.Reaction_weak(r_s_a, ua_test_dV[i]) + ck.Reaction_weak(r_s_b, ub_test_dV[i]);
							dbgElem[2][i] += ck.Hamiltonian_weak(ham_s_a, ua_test_dV[i]) + ck.Hamiltonian_weak(ham_s_b, ub_test_dV[i]);
							dbgElem[3][i] += ck.Advection_weak(f_s_a, &ua_grad_test_dV[i_nSpace]) + ck.Advection_weak(f_s_b, &ub_grad_test_dV[i_nSpace]);
						}
						else
						{
							dbgElem[1][i] += ck.Reaction_weak(r_s, u_test_dV[i]);
							dbgElem[2][i] += ck.Hamiltonian_weak(ham_s, u_test_dV[i]);
							dbgElem[3][i] += ck.Advection_weak(f_s, &u_grad_test_dV[i_nSpace]);
						}
					}
					// immersedBoundary's r_f (the flux-jump Dirac reaction) is tested above against the
					// IFEM test function on Gamma_f (0.5*(va+vb), = va = vb there), not against the plain,
					// unenriched u_test_dV (va != vb != u_test away from the vertices) and not inside the
					// wA/wB-weighted branch terms (a product of two fits that moves load between branches). f_f/ham_f/df_f/dham_f are always zero (see
					// updateImmersedBoundaryTerms), so nothing else from that call belongs here.
				}
				// Error is only meaningful where the PDE is actually assembled -- strictly, in the
				// active fluid domain phi_s >= 0. The old gate used H_s (the moment-fit area
				// weight), which is only guaranteed correct in an *integrated* sense: pointwise it
				// can overshoot [0,1] or go negative near a cut (confirmed directly: values like
				// H_s=1.14 and H_s=-0.066 were measured on a real triply-cut element), so "H_s>0"
				// let some genuinely-inside-the-solid quadrature points compete for the L^infty
				// max, and weighting L2 by H_s (rather than gating it) let their squared error
				// contribute in proportion to an overshot/undershot weight instead of either the
				// full contribution or none. Gate by the raw, pointwise level-set value instead --
				// exact, not a moment-fit approximation -- so no solid-side point (node or
				// quadrature point alike) ever enters either norm, regardless of how H_s's fit
				// behaves right at that point.
				const double phi_s_q = embeddedBoundary_sdf_q.data()[eN_k];
				double L2_contrib = 0.0;
				if (icase_f == 0)
				{
					double sol_in = q_u_exact_inner.data()[eN_k];
					double err_in = fabs(ua + uja - sol_in);
					L2_contrib += ImH_f * err_in * err_in * dV;
					double sol_out = q_u_exact_outer.data()[eN_k];
					double err_out = fabs(ub + ujb - sol_out);
					L2_contrib += H_f * err_out * err_out * dV;
					if (phi_s_q >= 0.0)
					{
						double _cand = (ImH_f >= H_f) ? err_in : err_out;
						if (getenv("ADR_DEBUG_LINF") && _cand > Linfty_error)
						{
							fprintf(stderr, "ADR_DEBUG_LINF NEWMAX eN=%d x=%.10f y=%.10f err=%.6e "
									"branch=%s H_s=%.6f ImH_f=%.6f H_f=%.6f wA=%.6f wB=%.6f D_f=%.6e "
									"icase_f=%d\n",
									eN, x, y, _cand, (ImH_f >= H_f) ? "inner" : "outer",
									H_s, ImH_f, H_f, wA, wB, D_f, icase_f);
						}
						if (ImH_f >= H_f)
							Linfty_error = std::max(Linfty_error, err_in);
						else
							Linfty_error = std::max(Linfty_error, err_out);
					}
				}
				else
				{
					if (icase_f == -1)
					{
						double sol = q_u_exact_inner.data()[eN_k];
						double err = fabs(u - sol);
						L2_contrib += err * err * dV;
						if (phi_s_q >= 0.0)
						{
							if (getenv("ADR_DEBUG_LINF") && err > Linfty_error)
							{
								fprintf(stderr, "ADR_DEBUG_LINF NEWMAX2 eN=%d x=%.10f y=%.10f err=%.6e branch=inner H_s=%.6f phi_s_q=%.6e icase_f=-1\n", eN, x, y, err, H_s, phi_s_q);
								// vertices from the quad point's reference coordinates (valid for P1 and P2)
								const double xi0 = x_ref.data()[k * 3 + 0], xi1 = x_ref.data()[k * 3 + 1];
								const double v0x = x - (jac[0] * xi0 + jac[1] * xi1), v0y = y - (jac[2] * xi0 + jac[3] * xi1);
								fprintf(stderr, "ADR_DEBUG_LINF   v0=(%.10f,%.10f) v1=(%.10f,%.10f) v2=(%.10f,%.10f) xi=(%.6f,%.6f) uh_q=%.10e exact_q=%.10e u=",
										v0x, v0y, v0x + jac[0], v0y + jac[2], v0x + jac[1], v0y + jac[3], xi0, xi1, u, sol);
								for (int i = 0; i < nDOF_trial_element; i++)
									fprintf(stderr, "%s%.10e", i ? "," : "", element_u.data()[i]);
								fprintf(stderr, "\n");
							}
							Linfty_error = std::max(Linfty_error, err);
						}
					}
					if (icase_f == 1)
					{
						double sol = q_u_exact_outer.data()[eN_k];
						double err = fabs(u - sol);
						L2_contrib += err * err * dV;
						if (phi_s_q >= 0.0)
						{
							if (getenv("ADR_DEBUG_LINF") && err > Linfty_error)
								fprintf(stderr, "ADR_DEBUG_LINF NEWMAX2 eN=%d x=%.10f y=%.10f err=%.6e branch=outer H_s=%.6f phi_s_q=%.6e icase_f=1\n", eN, x, y, err, H_s, phi_s_q);
							if (getenv("ADR_DEBUG_LINF") && err > Linfty_error)
							{
								// vertices recovered from the quad point, the Jacobian and the P1 reference basis
								const double b1 = u_trial_ref.data()[k * nDOF_trial_element + 1];
								const double b2 = u_trial_ref.data()[k * nDOF_trial_element + 2];
								const double v0x = x - (jac[0] * b1 + jac[1] * b2), v0y = y - (jac[2] * b1 + jac[3] * b2);
								fprintf(stderr, "ADR_DEBUG_LINF   v0=(%.10f,%.10f) v1=(%.10f,%.10f) v2=(%.10f,%.10f) u=(%.10e,%.10e,%.10e) uh_q=%.10e exact_q=%.10e\n",
										v0x, v0y, v0x + jac[0], v0y + jac[2], v0x + jac[1], v0y + jac[3],
										element_u.data()[0], element_u.data()[1], element_u.data()[2], u, sol);
							}
							Linfty_error = std::max(Linfty_error, err);
						}
					}
				}
				if (phi_s_q >= 0.0)
					L2_error += L2_contrib;
			}
		}

		void calculateResidual(arguments_dict &args)
		{
			xt::pyarray<double> &mesh_trial_ref = args.array<double>("mesh_trial_ref");
			xt::pyarray<double> &mesh_grad_trial_ref = args.array<double>("mesh_grad_trial_ref");
			xt::pyarray<double> &mesh_dof = args.array<double>("mesh_dof");
			xt::pyarray<int> &mesh_l2g = args.array<int>("mesh_l2g");
			xt::pyarray<double> &dV_ref = args.array<double>("dV_ref");
			xt::pyarray<double> &u_trial_ref = args.array<double>("u_trial_ref");
			xt::pyarray<double> &u_grad_trial_ref = args.array<double>("u_grad_trial_ref");
			xt::pyarray<double> &u_test_ref = args.array<double>("u_test_ref");
			xt::pyarray<double> &u_grad_test_ref = args.array<double>("u_grad_test_ref");
			xt::pyarray<double> &elementDiameter = args.array<double>("elementDiameter");
			xt::pyarray<double> &cfl = args.array<double>("cfl");
			double Ct_sge = args.scalar<double>("Ct_sge");
			double sc_uref = args.scalar<double>("sc_uref");
			double sc_alpha = args.scalar<double>("sc_alpha");
			double useMetrics = args.scalar<double>("useMetrics");
			xt::pyarray<double> &mesh_trial_trace_ref = args.array<double>("mesh_trial_trace_ref");
			xt::pyarray<double> &mesh_grad_trial_trace_ref = args.array<double>("mesh_grad_trial_trace_ref");
			xt::pyarray<double> &dS_ref = args.array<double>("dS_ref");
			xt::pyarray<double> &u_trial_trace_ref = args.array<double>("u_trial_trace_ref");
			xt::pyarray<double> &u_grad_trial_trace_ref = args.array<double>("u_grad_trial_trace_ref");
			xt::pyarray<double> &u_test_trace_ref = args.array<double>("u_test_trace_ref");
			xt::pyarray<double> &u_grad_test_trace_ref = args.array<double>("u_grad_test_trace_ref");
			xt::pyarray<double> &normal_ref = args.array<double>("normal_ref");
			xt::pyarray<double> &boundaryJac_ref = args.array<double>("boundaryJac_ref");
			int nElements_global = args.scalar<int>("nElements_global");
			xt::pyarray<int> &u_l2g = args.array<int>("u_l2g");
			xt::pyarray<double> &u_dof = args.array<double>("u_dof");
			// Triply-cut correction (see ADR.py Coefficients.initializeSDF): for a P1 element
			// where exactly one local node is solid-interior AND the fluid-fluid interface also
			// cuts it, that node's raw DOF value is physically meaningless (u_ex is not defined
			// inside the solid) and corrupts BOTH reconstructed IFEM branches through the coupled
			// continuity/flux rows in ifemBasisCoefficients_wrapper.h. triplyCut_overrideValue is
			// the analytically-continued embeddedBoundary_u_inner/_outer formula evaluated at that
			// node's own coordinates instead. -1 in triplyCut_maskedLocalNode means "not applicable".
			xt::pyarray<int> &triplyCut_maskedLocalNode = args.array<int>("triplyCut_maskedLocalNode");
			xt::pyarray<double> &triplyCut_overrideValue = args.array<double>("triplyCut_overrideValue");
			xt::pyarray<int> &sd_rowptr = args.array<int>("sd_rowptr");
			xt::pyarray<int> &sd_colind = args.array<int>("sd_colind");
			xt::pyarray<double> &q_a = args.array<double>("q_a");
			xt::pyarray<double> &q_v = args.array<double>("q_v");
			xt::pyarray<double> &q_r = args.array<double>("q_r");
			int lag_shockCapturing = args.scalar<int>("lag_shockCapturing");
			double shockCapturingDiffusion = args.scalar<double>("shockCapturingDiffusion");
			xt::pyarray<double> &q_numDiff_u = args.array<double>("q_numDiff_u");
			xt::pyarray<double> &q_numDiff_u_last = args.array<double>("q_numDiff_u_last");
			int offset_u = args.scalar<int>("offset_u");
			int stride_u = args.scalar<int>("stride_u");
			xt::pyarray<double> &globalResidual = args.array<double>("globalResidual");
			int nExteriorElementBoundaries_global = args.scalar<int>("nExteriorElementBoundaries_global");
			xt::pyarray<int> &exteriorElementBoundariesArray = args.array<int>("exteriorElementBoundariesArray");
			xt::pyarray<int> &elementBoundaryElementsArray = args.array<int>("elementBoundaryElementsArray");
			xt::pyarray<int> &elementBoundaryLocalElementBoundariesArray = args.array<int>("elementBoundaryLocalElementBoundariesArray");
			xt::pyarray<double> &ebqe_a = args.array<double>("ebqe_a");
			xt::pyarray<double> &ebqe_v = args.array<double>("ebqe_v");
			xt::pyarray<int> &isDOFBoundary_u = args.array<int>("isDOFBoundary_u");
			xt::pyarray<double> &ebqe_bc_u_ext = args.array<double>("ebqe_bc_u_ext");
			xt::pyarray<int> &isDiffusiveFluxBoundary_u = args.array<int>("isDiffusiveFluxBoundary_u");
			xt::pyarray<int> &isAdvectiveFluxBoundary_u = args.array<int>("isAdvectiveFluxBoundary_u");
			xt::pyarray<double> &ebqe_bc_flux_u_ext = args.array<double>("ebqe_bc_flux_u_ext");
			xt::pyarray<double> &ebqe_bc_advectiveFlux_u_ext = args.array<double>("ebqe_bc_advectiveFlux_u_ext");
			xt::pyarray<double> &ebqe_penalty_ext = args.array<double>("ebqe_penalty_ext");
			const bool embeddedBoundary = args.scalar<int>("embeddedBoundary");
			const double embeddedBoundary_penalty = args.scalar<double>("embeddedBoundary_penalty");
			const double embeddedBoundary_ghost_penalty = args.scalar<double>("embeddedBoundary_ghost_penalty");
			xt::pyarray<double> &embeddedBoundary_sdf_nodes = args.array<double>("embeddedBoundary_sdf_nodes");
			xt::pyarray<double> &embeddedBoundary_sdf_q = args.array<double>("embeddedBoundary_sdf_q");
			xt::pyarray<double> &embeddedBoundary_normal_q = args.array<double>("embeddedBoundary_normal_q");
			xt::pyarray<double> &embeddedBoundary_u_q = args.array<double>("embeddedBoundary_u_q");
			xt::pyarray<double> &embeddedBoundary_u_inner_q = args.array<double>("embeddedBoundary_u_inner_q");
			xt::pyarray<double> &embeddedBoundary_u_outer_q = args.array<double>("embeddedBoundary_u_outer_q");
			const bool embeddedBoundary_robin = args.scalar<int>("embeddedBoundary_robin");
			const double embeddedBoundary_robin_h_inner = args.scalar<double>("embeddedBoundary_robin_h_inner");
			const double embeddedBoundary_robin_h_outer = args.scalar<double>("embeddedBoundary_robin_h_outer");
			xt::pyarray<double> &embeddedBoundary_robin_uInfinity_inner_q = args.array<double>("embeddedBoundary_robin_uInfinity_inner_q");
			xt::pyarray<double> &embeddedBoundary_robin_uInfinity_outer_q = args.array<double>("embeddedBoundary_robin_uInfinity_outer_q");
			const bool immersedBoundary = args.scalar<int>("immersedBoundary");
			const double immersedBoundary_penalty = args.scalar<double>("immersedBoundary_penalty");
			const double immersedSCIFEM_switch = args.scalar<double>("immersedSCIFEM_switch");
			const double immersedSCIFEM_penalty = args.scalar<double>("immersedSCIFEM_penalty");
			const bool PG = args.scalar<int>("PG");
			xt::pyarray<double> &immersedBoundary_sdf_nodes = args.array<double>("immersedBoundary_sdf_nodes");
			xt::pyarray<double> &immersedBoundary_sdf_q = args.array<double>("immersedBoundary_sdf_q");
			xt::pyarray<double> &immersedBoundary_normal_q = args.array<double>("immersedBoundary_normal_q");
			xt::pyarray<double> &immersedBoundary_u_q = args.array<double>("immersedBoundary_u_q");
			xt::pyarray<double> &immersedBoundary_fluxJump_q = args.array<double>("immersedBoundary_fluxJump_q");
			xt::pyarray<double> &immersedBoundary_fluxJumpVector_q = args.array<double>("immersedBoundary_fluxJumpVector_q");
			xt::pyarray<double> &immersedBoundary_fluxJump_element = args.array<double>("immersedBoundary_fluxJump_element");
			xt::pyarray<double> &immersedBoundary_solutionJump_nodes = args.array<double>("immersedBoundary_solutionJump_nodes");
			xt::pyarray<double> &isActiveDOF = args.array<double>("isActiveDOF");
			const double eb_adjoint_sigma = args.scalar<double>("eb_adjoint_sigma");
			xt::pyarray<double> &x_ref = args.array<double>("x_ref");
			xt::pyarray<double> &xB_ref = args.array<double>("xB_ref");
			xt::pyarray<int> &elementBoundariesArray = args.array<int>("elementBoundariesArray");
			const int nElementBoundaries_owned = args.scalar<int>("nElementBoundaries_owned");
			xt::pyarray<double> &elementBoundaryDiameter = args.array<double>("elementBoundaryDiameter");
			xt::pyarray<double> &nodeDiametersArray = args.array<double>("nodeDiametersArray");
			xt::pyarray<double> &L2_error = args.array<double>("L2_error");
			xt::pyarray<double> &Linfty_error = args.array<double>("Linfty_error");
			const double mua = args.scalar<double>("mua");
			const double mub = args.scalar<double>("mub");
			const double jf = args.scalar<double>("jf");
			xt::pyarray<double> &q_u_exact_inner = args.array<double>("q_u_exact_inner");
			xt::pyarray<double> &q_u_exact_outer = args.array<double>("q_u_exact_outer");
			const bool recomputeIFEMGeometry = args.scalar<int>("recomputeIFEMGeometry");
			ensureIFEMCacheSized(nElements_global); // also (re)asserts useExact = true on (re)allocation
			if (recomputeIFEMGeometry)
				ifemGeometryGeneration++;
			ifem_boundaries.clear();
			ifem_boundary_elements.clear();
			cutfem_boundaries.clear();
			cutfem_boundary_elements.clear();
			elementIsActive.resize(nElements_global);
			// Debug-only (ADR_DEBUG_ROW=<global dof>): decompose that row's residual by mechanism.
			const int dbgRow = getenv("ADR_DEBUG_ROW") ? atoi(getenv("ADR_DEBUG_ROW")) : -1;
			double dbgDelta[8] = {0., 0., 0., 0., 0., 0., 0., 0.};
			auto dbgRowVal = [&]() { return dbgRow >= 0 ? globalResidual.data()[offset_u + stride_u * dbgRow] : 0.0; };
			const int tcDeadModeV = tcDeadMode();
			const bool tcFluidMode = std::getenv("ADR_TC_EDGE_FLUID") != nullptr;
			// ADR_TC_EDGE_BOUNDARY_ONLY=1 (use with SCIFEM on): keep the one-sided fluid-side correction only
			// on root-adjacent edges that lie on the exterior boundary. Interior ones are then handled by
			// SCIFEM's two-sided consistency term, which has no counterpart on an exterior edge (no neighbour).
			const bool tcBoundaryOnly = std::getenv("ADR_TC_EDGE_BOUNDARY_ONLY") != nullptr;
			const double tcSgn = tcFluidMode ? -1.0 : 1.0;
			// ADR_EXT_NITSCHE=1 (needs ADR_TC_EDGE_BOUNDARY_ONLY): on the exterior Gamma_f-cut edges add to the
			// one-sided consistency term the Nitsche adjoint term  -int mu grad(v).n (u+j_h-g)  and, with
			// ADR_EXT_NITSCHE_PEN=gamma, the penalty  gamma*mu/h int (u+j_h-g) v,  on the fluid pieces of the edge.
			// g is the Dirichlet data of each phase, ext_edge_u_inner/outer, at 4 Gauss points per exterior edge.
			const bool extNitsche = tcBoundaryOnly && std::getenv("ADR_EXT_NITSCHE") != nullptr;
			// ADR_GP_MU=1: weight the ghost penalty of each fluid phase by its coefficient (mu_a, mu_b),
			// like the Nitsche and interior-penalty terms; the combined fallback uses the mean.
			const bool gpMu = std::getenv("ADR_GP_MU") != nullptr;
			const double gpMuA = gpMu ? mua : 1.0, gpMuB = gpMu ? mub : 1.0, gpMuAB = gpMu ? 0.5 * (mua + mub) : 1.0;
			const double extNitschePen = std::getenv("ADR_EXT_NITSCHE_PEN") ? std::atof(std::getenv("ADR_EXT_NITSCHE_PEN")) : 0.0;
			xt::pyarray<double> &ext_edge_u_inner = args.array<double>("ext_edge_u_inner");
			xt::pyarray<double> &ext_edge_u_outer = args.array<double>("ext_edge_u_outer");
			std::vector<int> ebN2ebNE;
			if (extNitsche)
			{
				ebN2ebNE.assign(elementBoundaryElementsArray.size() / 2, -1);
				for (int ie = 0; ie < nExteriorElementBoundaries_global; ie++)
					ebN2ebNE[exteriorElementBoundariesArray.data()[ie]] = ie;
			}
			const double extT[4] = {0.5 - 0.5 * 0.8611363115940526, 0.5 - 0.5 * 0.3399810435848563,
									0.5 + 0.5 * 0.3399810435848563, 0.5 + 0.5 * 0.8611363115940526};
			auto extInterp = [&](const double *gk, double t) {
				double r = 0.0;
				for (int i = 0; i < 4; i++)
				{
					double w = 1.0;
					for (int j = 0; j < 4; j++)
						if (j != i) w *= (t - extT[j]) / (extT[i] - extT[j]);
					r += gk[i] * w;
				}
				return r; };
			const double extG3[3] = {0.5 - 0.5 * std::sqrt(0.6), 0.5, 0.5 + 0.5 * std::sqrt(0.6)};
			const double extW3[3] = {5.0 / 18.0, 8.0 / 18.0, 5.0 / 18.0};
			std::vector<char> tcNodeFluid;
			if (tcDeadModeV == 2 && embeddedBoundary)
				tcBuildNodeFluid(tcNodeFluid, nElements_global, u_l2g, embeddedBoundary_sdf_nodes, u_dof.size());

			//
			// loop over elements to compute volume integrals and load them into element and global residual
			//
			// eN is the element index
			// eN_k is the quadrature point index for a scalar
			// eN_k_nSpace is the quadrature point index for a vector
			// eN_i is the element test function index
			// eN_j is the element trial function index
			// eN_k_j is the quadrature point index for a trial function
			// eN_k_i is the quadrature point index for a trial function
			// Per-face reference-ELEMENT quadrature points, taken from proteus' own trace tables.
			// Simplex::calculate(..., isBoundary=true) reads its xi_r argument as reference *element*
			// coordinates (it forms x = node0 + Jac_0*xi_r), but xB_ref holds reference *boundary*
			// points (t,0,0) -- passing those directly puts every face's points on the local
			// node0->node1 edge, i.e. correct only for local face 2. mesh_trial_trace_ref holds the P1
			// mesh shape functions at (face, quadrature point) and for a triangle those barycentric
			// functions ARE the reference coordinates (phi_0=1-xi-eta, phi_1=xi, phi_2=eta), so this
			// is exactly consistent with calculateMapping_elementBoundary's own convention.
			double xB_ref_faces[nDOF_mesh_trial_element * nQuadraturePoints_elementBoundary * 3];
			for (int f = 0; f < nDOF_mesh_trial_element; f++)
				for (int kb = 0; kb < nQuadraturePoints_elementBoundary; kb++)
				{
					const int fkb = f * nQuadraturePoints_elementBoundary + kb;
					const double *phi_m = &mesh_trial_trace_ref.data()[fkb * nDOF_mesh_trial_element];
					xB_ref_faces[fkb * 3 + 0] = phi_m[1];
					xB_ref_faces[fkb * 3 + 1] = phi_m[2];
					xB_ref_faces[fkb * 3 + 2] = 0.0;
				}
			for (int eN = 0; eN < nElements_global; eN++)
			{
				// std::cout << "########################\n element: " << eN << " \n########################" << std::endl;
				// declare local storage for element residual and initialize
				// double elementResidual_u[nDOF_test_element],element_u[nDOF_trial_element];
				auto elementResidual_u = xt::pyarray<double>::from_shape({nDOF_test_element});
				auto element_u = xt::pyarray<double>::from_shape({nDOF_trial_element});
				bool element_active = false;
				elementIsActive[eN] = false;
				for (int i = 0; i < nDOF_trial_element; i++)
				{
					int eN_i = eN * nDOF_trial_element + i;
					element_u.data()[i] = u_dof.data()[u_l2g.data()[eN_i]];
					// std::cout << "element_u[" << i << "]:" << element_u.data()[i] << std::endl;
				} // i
				double element_phi_s[nDOF_trial_element];
				for (int j = 0; j < nDOF_trial_element; j++)
				{
					int eN_j = eN * nDOF_trial_element + j;
					element_phi_s[j] = embeddedBoundary_sdf_nodes.data()[u_l2g.data()[eN_j]];
				}
				double element_phi_f[nDOF_trial_element];
				for (int j = 0; j < nDOF_trial_element; j++)
				{
					int eN_j = eN * nDOF_trial_element + j;
					element_phi_f[j] = immersedBoundary_sdf_nodes.data()[u_l2g.data()[eN_j]];
				}
				// std::cout << std::endl;
				double element_nodes[nDOF_trial_element * 3];
				for (int i = 0; i < nDOF_trial_element; i++)
				{
					int eN_i = eN * nDOF_trial_element + i;
					for (int I = 0; I < 3; I++)
						// element_nodes[i * 3 + I] = mesh_dof.data()[mesh_l2g.data()[eN_i] * 3 + I];
						element_nodes[i * 3 + I] = mesh_dof.data()[u_l2g.data()[eN_i] * 3 + I];
					// std::cout << "element node[" << i << "]:" << element_nodes[i * 3 + 0] << " " << element_nodes[i * 3 + 1] << " " << element_nodes[i * 3 + 2] << std::endl;
					// std::cout << "element node[" << i << "]:" << element_nodes[i * 3 + 0] << " " << element_nodes[i * 3 + 1] << " " << element_nodes[i * 3 + 2] << std::endl;
					// std::cout << "element phi_f[" << i << "]:" << element_phi_f[i] << std::endl << std::endl;
				} // i
				if (gf_s_interior_gen[eN] != ifemGeometryGeneration)
				{
					gf_s_interior_icase[eN] = gf_s_cache[eN].calculate(element_phi_s, element_nodes, x_ref.data(), false);
					gf_s_interior_gen[eN] = ifemGeometryGeneration;
				}
				int icase_s = gf_s_interior_icase[eN];
				const bool pgElem = !getenv("ADR_PG_TC_ONLY") || (icase_s == 0 && gf_f_interior_icase[eN] == 0);
				if (icase_s == 0)
				{
					// only works for simplices
					for (int ebN_element = 0; ebN_element < nDOF_mesh_trial_element; ebN_element++)
					{
						const int ebN = elementBoundariesArray.data()[eN * nDOF_mesh_trial_element + ebN_element];
						// internal and actually a cut edge
						if (elementBoundaryElementsArray.data()[ebN * 2 + 1] != -1 && (ebN < nElementBoundaries_owned))
							cutfem_boundaries.insert(ebN);
					}
				}
				if (gf_f_interior_gen[eN] != ifemGeometryGeneration)
				{
					gf_f_interior_icase[eN] = gf_f_cache[eN].calculate(element_phi_f, element_nodes, x_ref.data(), mua, mub, jf, false, false);
					gf_f_interior_gen[eN] = ifemGeometryGeneration;
				}
				int icase_f = gf_f_interior_icase[eN];
				// Records that the buried-Gamma_f logic below demoted this element, and to which
				// phase: -1 = fluid is all Omega^- (mua), +1 = all Omega^+ (mub), 0 = not demoted.
				int demoted_f = 0;
				// Both level sets genuinely cut this element: the product weight ImH_f*H_s is not
				// moment-exact here, so fit one polynomial per intersected region instead. Any
				// element cut by at most one level set is left entirely alone.
				comp_valid[eN] = 0;
				comp_DF_valid[eN] = 0;
				if (icase_s == 0 && icase_f == 0 && nSpace == 2)
				{
					const double phi_s3[3] = {element_phi_s[0], element_phi_s[1], element_phi_s[2]};
					const double phi_f3[3] = {element_phi_f[0], element_phi_f[1], element_phi_f[2]};
					// The fluid interface can cross this element and still lie entirely inside the
					// solid, leaving the fluid undivided. There is then no fluid-fluid interface to
					// resolve *in the fluid*, so the two-sided IFEM reconstruction has nothing to
					// do -- and building it anyway means enforcing interface conditions along a
					// segment where no fluid exists. Demote the element to ordinary two-phase by
					// reporting the fluid's own side, which routes every downstream branch (bulk
					// terms, the solid's Nitsche condition, the error norm) to the standard basis.
					double areaA = 0.0, areaB = 0.0;
					equivalent_polynomials::composite::fluid_region_areas(phi_s3, phi_f3, areaA, areaB);
					// NOTE: we deliberately do NOT change icase_f here. Switching this element to the
					// standard basis would make it disagree with its neighbours about what their shared
					// DOFs mean: the neighbour (genuinely cut in the fluid) needs the pointwise nodal
					// value of u, while this element would need the surviving branch continued to all of
					// its nodes -- including nodes on the other side of Gamma_f. One DOF cannot be both,
					// so the exact solution would drop out of the discrete space entirely. Keeping the
					// IFEM basis costs nothing (u_ex satisfies the interface conditions globally, so
					// imposing them along a buried stretch of Gamma_f is harmless) and keeps the space
					// conforming. What we DO fix is the weights: the empty region must get exactly zero
					// rather than whatever a fit over an empty region would produce.
					//
					// Tolerance, not exact equality: a genuine tangency between Gamma_f and Gamma_s
					// (their zero-sets crossing exactly on a shared mesh edge) makes one region's true
					// area a single point, but floating-point roundoff in the clip almost never lands
					// on bit-exact 0.0 -- it lands a few ULPs away (observed: ~1e-33 against a
					// reference-triangle scale of 0.5). An exact-equality check misses that case
					// entirely and falls through to fitting a degree-4 polynomial to a numerically
					// degenerate sliver, which is unstable (observed: off by 64x on its own area).
					// areaTol is far above that roundoff floor and far below any physically meaningful
					// sub-area, so it can't misclassify a genuinely small-but-real fluid region.
					const double areaTol = 1.0e-9;
					if (areaB <= areaTol && areaA > areaTol)
						demoted_f = -1;   // fluid is all Omega^- (mua side)
					else if (areaA <= areaTol && areaB > areaTol)
						demoted_f = 1;    // fluid is all Omega^+ (mub side)
					else
					{
						const double nodes2[6] = {element_nodes[0], element_nodes[1],
												  element_nodes[3], element_nodes[4],
												  element_nodes[6], element_nodes[7]};
						comp_valid[eN] = (equivalent_polynomials::composite::weights<nP_fit>(
							phi_s3, phi_f3,
							&comp_A_cache[(std::size_t)eN * nCompMon],
							&comp_B_cache[(std::size_t)eN * nCompMon]) &&
						                  equivalent_polynomials::composite::dirac_weights<nP_fit>(
							phi_s3, phi_f3, nodes2,
							&comp_DA_cache[(std::size_t)eN * nCompMon],
							&comp_DB_cache[(std::size_t)eN * nCompMon])) ? 1 : 0;
						// Gamma_f's own Dirac measure, split by Gamma_s's sign (see comp_DFm_cache's
						// declaration): comp_DFa is the part of Gamma_f's chord still inside the
						// active fluid, the only part any Gamma_f Dirac-delta term (e.g. the
						// prescribed flux jump) should ever see on this element.
						comp_DF_valid[eN] = equivalent_polynomials::composite::edge_weights<nP_fit>(
							phi_f3, phi_s3, nodes2,
							&comp_DFm_cache[(std::size_t)eN * nCompMon],
							&comp_DFa_cache[(std::size_t)eN * nCompMon]) ? 1 : 0;
					}
				}
				double JA[nDOF_trial_element];
				double JB[nDOF_trial_element];
				std::fill(JA, JA + nDOF_trial_element, 0.0);
				std::fill(JB, JB + nDOF_trial_element, 0.0);
				// Prescribed flux jump of this cell for the flux-jump lift (0 if the lift is off).
				const double fluxLiftJ = (icase_f == 0) ? immersedBoundary_fluxJump_element.data()[eN] : 0.0;
				if (getenv("ADR_DEBUG_JAJB") && eN == atoi(getenv("ADR_DEBUG_JAJB")))
				{
					fprintf(stderr, "ADR_DEBUG_JAJB eN=%d icase_s=%d icase_f=%d comp_valid=%d comp_DF_valid=%d "
							"maskedLocalNode=%d phi_f=[%.6f,%.6f,%.6f] phi_s=[%.6f,%.6f,%.6f]\n",
							eN, icase_s, icase_f, comp_valid[eN], comp_DF_valid[eN],
							triplyCut_maskedLocalNode.data()[eN],
							element_phi_f[0], element_phi_f[1], element_phi_f[2],
							element_phi_s[0], element_phi_s[1], element_phi_s[2]);
				}
				if (icase_f == 0)
				{
					// std::cout << "Active element" << std::endl;
					// only works for simplices
					for (int ebN_element = 0; ebN_element < nDOF_mesh_trial_element; ebN_element++)
					{
						// std::cout << "ebN_element =" << ebN_element << "\t ebN=" << ebN << std::endl;
						// internal and actually a cut edge
						const int ebN = elementBoundariesArray.data()[eN * nDOF_mesh_trial_element + ebN_element];
						// if (elementBoundaryElementsArray.data()[ebN * 2 + 1] != -1 && (ebN < nElementBoundaries_owned)) // This gives all the internal edges instead of just the cut edges.
						
						// indexing convention for P1: edge opposite to node i is given by (i+1)%3 and (i+2)%3 nodes. 
						// nathawani: P2 needs different indexing convention.
						// Do we need corner cases? (<=0) or just (<0) for cut edge detection?
						// The SCIFEM facet term is an inter-element consistency term: it needs the
						// trace from BOTH elements sharing the face. On an exterior face the second
						// entry of elementBoundaryElementsArray is -1, and the loops below index
						// elementIsActive[] and gf_f_cache[] with it before any validity test, which
						// is out of bounds. test=8's circle lies wholly inside the domain so this
						// never triggered; test=12's diagonal interface exits through the boundary
						// and segfaulted on arm64. Only interior faces belong in this set.
						if (elementBoundaryElementsArray.data()[ebN * 2 + 1] != -1 &&
							element_phi_f[(ebN_element + 1) % 3] * element_phi_f[(ebN_element + 2) % 3] < 0.0)
						{
							// This should give just the cut edges for simplices.
							ifem_boundaries.insert(ebN);
						}
					}
					// Leveque & Li 1994, Example 1, 2, 3, 4; Ji et. al. 2014
					// The jump function is, in general, not constant over the cut element, so it
					// must be evaluated at each node's own coordinates (not once at the cut
					// barycenter) -- otherwise JA/JB are constant over the element and their
					// interpolated gradients (grad_uja, grad_ujb) vanish identically.
					const double eps_phi = 1.0e-12;
					auto assign_jump_side = [&](int i, bool isOuter, double jump) {
						if (isOuter)
						{
							JA[i] = -jump;
							JB[i] = 0.0;
						}
						else
						{
							JA[i] = 0.0;
							JB[i] = jump;
						}
					};
						for (int i = 0; i < nDOF_trial_element; i++)
						{

							int eN_i = eN * nDOF_trial_element + i;
							const double jump_i = immersedBoundary_solutionJump_nodes.data()[u_l2g.data()[eN_i]];
							if (element_phi_f[i] > 0.0)
							{
								assign_jump_side(i, true, jump_i);
							}
							else if (element_phi_f[i] <= 0.0)
							{
								assign_jump_side(i, false, jump_i);
							}
						}
				}
				else if (icase_f == -1)
				{
				}
				else if (icase_f == 1)
				{
				}
				calculateElementResidual(icase_f,
										 demoted_f,
										 mesh_trial_ref,
										 mesh_grad_trial_ref,
										 mesh_dof,
										 mesh_l2g,
										 x_ref,
										 dV_ref,
										 u_trial_ref,
										 u_grad_trial_ref,
										 u_test_ref,
										 u_grad_test_ref,
										 elementDiameter,
										 elementBoundaryDiameter,
										 nodeDiametersArray,
										 cfl,
										 Ct_sge,
										 sc_uref,
										 sc_alpha,
										 useMetrics,
										 mesh_trial_trace_ref,
										 mesh_grad_trial_trace_ref,
										 dS_ref,
										 u_trial_trace_ref,
										 u_grad_trial_trace_ref,
										 u_test_trace_ref,
										 u_grad_test_trace_ref,
										 normal_ref,
										 boundaryJac_ref,
										 nElements_global,
										 nElementBoundaries_owned,
										 u_l2g,
										 u_dof,
										 sd_rowptr,
										 sd_colind,
										 q_a,
										 comp_valid[eN] ? &comp_A_cache[(std::size_t)eN * nCompMon] : nullptr,
										 comp_valid[eN] ? &comp_B_cache[(std::size_t)eN * nCompMon] : nullptr,
										 comp_valid[eN] ? &comp_DA_cache[(std::size_t)eN * nCompMon] : nullptr,
										 comp_valid[eN] ? &comp_DB_cache[(std::size_t)eN * nCompMon] : nullptr,
										 comp_DF_valid[eN] ? &comp_DFa_cache[(std::size_t)eN * nCompMon] : nullptr,
										 q_v,
										 q_r,
										 lag_shockCapturing,
										 shockCapturingDiffusion,
										 q_numDiff_u,
										 q_numDiff_u_last,
										 offset_u, stride_u,
										 elementResidual_u,
										 nExteriorElementBoundaries_global,
										 exteriorElementBoundariesArray,
										 elementBoundariesArray,
										 elementBoundaryElementsArray,
										 elementBoundaryLocalElementBoundariesArray,
										 element_u,
										triplyCut_maskedLocalNode.data()[eN],
										triplyCut_overrideValue.data()[eN],
										 eN,
										 embeddedBoundary,
										 embeddedBoundary_penalty,
										 embeddedBoundary_normal_q,
										 embeddedBoundary_u_q,
										 embeddedBoundary_u_inner_q,
										 embeddedBoundary_u_outer_q,
										 embeddedBoundary_robin,
										 embeddedBoundary_robin_h_inner,
										 embeddedBoundary_robin_h_outer,
										 embeddedBoundary_robin_uInfinity_inner_q,
										 embeddedBoundary_robin_uInfinity_outer_q,
										 embeddedBoundary_sdf_q,
										 immersedBoundary,
										 immersedBoundary_penalty,
										 immersedBoundary_sdf_q,
										 immersedBoundary_normal_q,
										 immersedBoundary_u_q,
										 immersedBoundary_fluxJump_q,
										 immersedBoundary_fluxJumpVector_q,
										 immersedBoundary_solutionJump_nodes,
										 element_phi_f,
										 element_active,
										 elementIsActive,
										 JA,
										 JB,
										 fluxLiftJ,
										 L2_error.data()[0],
										 Linfty_error.data()[0],
												 mua,
											 mub,
											 q_u_exact_inner,
											 q_u_exact_outer,
											 PG,
											 pgElem);
				//
				// load element into global residual and save element residual
				//
				for (int i = 0; i < nDOF_test_element; i++)
				{
					int eN_i = eN * nDOF_test_element + i;
					globalResidual.data()[offset_u + stride_u * u_l2g.data()[eN_i]] += elementResidual_u.data()[i];
					if (element_active)
						isActiveDOF.data()[offset_u + stride_u * u_l2g.data()[eN_i]] = 1.0;
					// std::cout << "globalResidual[" << offset_u + stride_u * u_l2g.data()[eN_i] << "] += " << elementResidual_u.data()[i] << std::endl;
					if (dbgRow >= 0 && u_l2g.data()[eN_i] == dbgRow)
					{
						for (int b = 0; b < 4; b++)
							dbgDelta[b] += dbgElem[b][i];
						fprintf(stderr, "ADR_ROW_ELEM eN=%d i=%d icase_s=%d icase_f=%d bulk=%.6e nitsche_pen=%.6e nitsche_cons=%.6e nitsche_adj=%.6e total=%.6e\n",
								eN, i, icase_s, icase_f, dbgElem[0][i], dbgElem[1][i], dbgElem[2][i], dbgElem[3][i], elementResidual_u.data()[i]);
					}
				} // i
				if (getenv("ADR_DEBUG_TC_RESIDUAL") && comp_valid[eN])
				{
					fprintf(stderr, "ADR_DEBUG_TC_RESIDUAL eN=%d maskedLocalNode=%d "
							"elementResidual_u=[%.10e,%.10e,%.10e]\n",
							eN, triplyCut_maskedLocalNode.data()[eN],
							elementResidual_u.data()[0], elementResidual_u.data()[1], elementResidual_u.data()[2]);
				}
				if (std::getenv("ADR_DUMP_BASIS") && embeddedBoundary && immersedBoundary && nDOF_trial_element == 3)
				{
					// Inert unless ADR_DUMP_BASIS is set: one line per element with its geometry and
					// (for Gamma_f-cut elements) the affine two-branch basis, for offline analysis.
					fprintf(stderr, "ADR_BASIS e=%d is=%d if=%d g=%d,%d,%d xy=%.12e,%.12e,%.12e,%.12e,%.12e,%.12e pf=%.12e,%.12e,%.12e ps=%.12e,%.12e,%.12e",
							eN, icase_s, icase_f, u_l2g.data()[eN * 3 + 0], u_l2g.data()[eN * 3 + 1], u_l2g.data()[eN * 3 + 2],
							element_nodes[0], element_nodes[1], element_nodes[3], element_nodes[4], element_nodes[6], element_nodes[7],
							element_phi_f[0], element_phi_f[1], element_phi_f[2], element_phi_s[0], element_phi_s[1], element_phi_s[2]);
					if (icase_f == 0)
					{
						GfType &gf_dump = gf_f_cache[eN];
						gf_dump.set_quad(0);
						double jacd[nSpace * nSpace], jacDetd, jacInvd[nSpace * nSpace], xdp, ydp, zdp;
						ck.calculateMapping_element(eN, 0, mesh_dof.data(), mesh_l2g.data(),
													mesh_trial_ref.data(), mesh_grad_trial_ref.data(),
													jacd, jacDetd, jacInvd, xdp, ydp, zdp);
						fprintf(stderr, " ref=%.12e,%.12e", xdp, ydp);
						for (int ii = 0; ii < 3; ii++)
							fprintf(stderr, " VA%d=%.12e,%.12e,%.12e VB%d=%.12e,%.12e,%.12e", ii,
									gf_dump.VA(ii), gf_dump.VA_x(ii), gf_dump.VA_y(ii), ii,
									gf_dump.VB(ii), gf_dump.VB_x(ii), gf_dump.VB_y(ii));
					}
					fprintf(stderr, "\n");
				}
				// Masked root-adjacent edges leak boundary flux H_s doesn't remove (ordinary
				// hat functions vanish there; two-sided branches don't). Add it back for all 3
				// local dofs. ADR_DISABLE_TC_EDGE=1 disables (residual+Jacobian).
				// icase_s == -1 means this element has NO fluid anywhere (H_s == 0 identically) --
				// Gamma_f may still cross it geometrically, but with zero real fluid area there is
				// no bulk contribution to correct and no physically-meaningful va/vb reconstruction
				// to add back. Without this check TC_EDGE still fires there (icase_f==0 alone
				// doesn't rule it out), injecting a real, nonzero correction from a "dead" element
				// into whichever physical dofs it shares with genuinely active neighbors.
				if (embeddedBoundary && immersedBoundary && (nDOF_trial_element == 3 || nDOF_trial_element == 6) && icase_f == 0
					&& (icase_s != -1 || tcDeadOK(tcDeadModeV, tcNodeFluid, u_l2g, eN))
					&& std::getenv("ADR_DISABLE_TC_EDGE") == nullptr)
				{
					// P1 and P2. Integrated on the two Gamma_f-cut (root-adjacent) faces with the element-boundary
					// quadrature and the boundary evaluation of the two-branch basis, as in the SCIFEM loop. Each
					// branch is restricted to its piece of the face (its side of Gamma_f, and the fluid side of
					// Gamma_s -- or the solid side for the default TC_EDGE) by an edge moment-fit indicator, exact
					// for integrands of degree <= nP_edge (flux*v: degree 1 for P1, 3 for P2).
					const int nD = nDOF_trial_element;
					bool neg[3] = {element_phi_f[0] < 0.0, element_phi_f[1] < 0.0, element_phi_f[2] < 0.0};
					int cnt_neg = (neg[0] ? 1 : 0) + (neg[1] ? 1 : 0) + (neg[2] ? 1 : 0);
					int root_i = -1;
					for (int ii = 0; ii < 3; ii++)
						if ((neg[ii] && cnt_neg == 1) || (!neg[ii] && cnt_neg == 2)) { root_i = ii; break; }
					if (root_i >= 0)
					{
						if (gf_f_boundary_gen[eN] != ifemGeometryGeneration)
						{
							gf_f_boundary_icase[eN] = gf_f_cache[eN].calculate(element_phi_f, element_nodes, xB_ref_faces, mua, mub, jf, true, false);
							gf_f_boundary_gen[eN] = ifemGeometryGeneration;
						}
						GfType &gf_f_tc = gf_f_cache[eN];
						// Branch coefficients u_i + JA_i / u_i + JB_i: the solution-jump field (JA/JB) is
						// reconstructed through the same branch basis as everywhere else in this file, so the true
						// branch value is sum (u_i+JA_i) va_i (cf. line ~784, ~931).
						double cuA[nDOF_trial_element], cuB[nDOF_trial_element];
						for (int ii = 0; ii < nD; ii++)
						{
							// Triply-cut: reuse the same analytic override the ordinary bulk
							// assembly uses (see ADR.py Coefficients.initializeSDF) instead of a
							// masked node's meaningless raw DOF.
							const double u_ii = (triplyCut_maskedLocalNode.data()[eN] == ii)
													? triplyCut_overrideValue.data()[eN]
													: u_dof.data()[u_l2g.data()[eN * nD + ii]];
							cuA[ii] = u_ii + JA[ii];
							cuB[ii] = u_ii + JB[ii];
						}
						// Default (TC_EDGE): the solid pieces {phi_s<0}. ADR_TC_EDGE_FLUID: the fluid pieces {phi_s>0}, opposite sign.
						const double sgn_s = tcFluidMode ? 1.0 : -1.0;
						for (int pj = 0; pj < 3; pj++)
						{
							if (pj == root_i)
								continue; // the face opposite the root is not cut by Gamma_f
							const int ebN_tc = elementBoundariesArray.data()[eN * nDOF_mesh_trial_element + pj];
							if (tcBoundaryOnly && elementBoundaryElementsArray.data()[ebN_tc * 2 + 1] != -1)
								continue; // interior face: left to SCIFEM
							// face pj runs from local node (pj+1)%3 (t=0) to (pj+2)%3 (t=1), as in the SCIFEM loop
							const int n0 = (pj + 1) % 3, n1 = (pj + 2) % 3;
							double C_a[nP_edge + 1], C_b[nP_edge + 1];
							equivalent_polynomials::calculate_edge_interval_H<nP_edge>(-element_phi_f[n0], -element_phi_f[n1],
																					   sgn_s * element_phi_s[n0], sgn_s * element_phi_s[n1], C_a);
							equivalent_polynomials::calculate_edge_interval_H<nP_edge>(element_phi_f[n0], element_phi_f[n1],
																					   sgn_s * element_phi_s[n0], sgn_s * element_phi_s[n1], C_b);
							const double ex0 = element_nodes[n0 * 3 + 0], ey0 = element_nodes[n0 * 3 + 1];
							const double evx = element_nodes[n1 * 3 + 0] - ex0, evy = element_nodes[n1 * 3 + 1] - ey0;
							const double elen2 = evx * evx + evy * evy;
							// ADR_EXT_NITSCHE: Dirichlet data of each phase at 4 Gauss points of the exterior edge,
							// parametrized from its lower global node.
							const int ebNE_x = extNitsche ? ebN2ebNE[ebN_tc] : -1;
							const bool fwd = u_l2g.data()[eN * nD + n0] < u_l2g.data()[eN * nD + n1];
							const double h_e = std::sqrt(elen2);
							for (int kb = 0; kb < nQuadraturePoints_elementBoundary; kb++)
							{
								const int ebN_local_kb = pj * nQuadraturePoints_elementBoundary + kb;
								double jac_b[nSpace * nSpace], jacDet_b, jacInv_b[nSpace * nSpace],
									bJac_b[nSpace * (nSpace - 1)], mT_b[(nSpace - 1) * (nSpace - 1)], mTDS_b, n_b[2], xb, yb, zb;
								ck.calculateMapping_elementBoundary(eN, pj, kb, ebN_local_kb,
																	mesh_dof.data(), mesh_l2g.data(),
																	mesh_trial_trace_ref.data(), mesh_grad_trial_trace_ref.data(),
																	boundaryJac_ref.data(), jac_b, jacDet_b, jacInv_b,
																	bJac_b, mT_b, mTDS_b, normal_ref.data(), n_b, xb, yb, zb);
								const double dS = mTDS_b * dS_ref.data()[kb];
								const double t = ((xb - ex0) * evx + (yb - ey0) * evy) / elen2;
								const double Wa = equivalent_polynomials::evaluate_edge_poly<nP_edge>(C_a, t);
								const double Wb = equivalent_polynomials::evaluate_edge_poly<nP_edge>(C_b, t);
								gf_f_tc.set_boundary_quad(ebN_local_kb);
								double uA = 0.0, uB = 0.0, fluxA = 0.0, fluxB = 0.0;
								for (int kk = 0; kk < nD; kk++)
								{
									uA += cuA[kk] * gf_f_tc.VA(kk);
									uB += cuB[kk] * gf_f_tc.VB(kk);
									fluxA += cuA[kk] * (gf_f_tc.VA_x(kk) * n_b[0] + gf_f_tc.VA_y(kk) * n_b[1]);
									fluxB += cuB[kk] * (gf_f_tc.VB_x(kk) * n_b[0] + gf_f_tc.VB_y(kk) * n_b[1]);
								}
								uA += fluxLiftJ * gf_f_tc.PSIA();
								uB += fluxLiftJ * gf_f_tc.PSIB();
								fluxA += fluxLiftJ * (gf_f_tc.PSIA_x() * n_b[0] + gf_f_tc.PSIA_y() * n_b[1]);
								fluxB += fluxLiftJ * (gf_f_tc.PSIB_x() * n_b[0] + gf_f_tc.PSIB_y() * n_b[1]);
								fluxA *= mua;
								fluxB *= mub;
								double gA = 0.0, gB = 0.0;
								if (ebNE_x >= 0)
								{
									gA = extInterp(ext_edge_u_inner.data() + 4 * ebNE_x, fwd ? t : 1.0 - t);
									gB = extInterp(ext_edge_u_outer.data() + 4 * ebNE_x, fwd ? t : 1.0 - t);
								}
								for (int jj = 0; jj < nD; jj++)
								{
									const int gdof = u_l2g.data()[eN * nD + jj];
									const double vA = gf_f_tc.VA(jj), vB = gf_f_tc.VB(jj);
									double contrib = tcSgn * (Wa * fluxA * vA + Wb * fluxB * vB) * dS;
									if (ebNE_x >= 0)
									{
										const double dvnA = mua * (gf_f_tc.VA_x(jj) * n_b[0] + gf_f_tc.VA_y(jj) * n_b[1]);
										const double dvnB = mub * (gf_f_tc.VB_x(jj) * n_b[0] + gf_f_tc.VB_y(jj) * n_b[1]);
										// Nitsche-type: adjoint term and penalty gamma*mu/h, as on Gamma_s
										contrib += (Wa * (-dvnA * (uA - gA) + extNitschePen * mua / h_e * (uA - gA) * vA)
													+ Wb * (-dvnB * (uB - gB) + extNitschePen * mub / h_e * (uB - gB) * vB)) * dS;
									}
									globalResidual.data()[offset_u + stride_u * gdof] += contrib;
									if (gdof == dbgRow)
									{
										dbgDelta[4] += contrib;
										fprintf(stderr, "ADR_ROW_TCEDGE eN=%d face=%d kb=%d t=%.4f Wa=%.6e Wb=%.6e jj=%d fluxA=%.6e fluxB=%.6e contrib=%.6e\n",
												eN, pj, kb, t, Wa, Wb, jj, fluxA, fluxB, contrib);
									}
								}
							}
						}
					}
				}
			} // elements
			if (std::getenv("ADR_DEBUG_DEAD") && embeddedBoundary)
			{
				std::vector<char> nf;
				tcBuildNodeFluid(nf, nElements_global, u_l2g, embeddedBoundary_sdf_nodes, u_dof.size());
				int nDeadF = 0, nTouch = 0, nDeadAll = 0;
				const bool dumpCells = std::atoi(std::getenv("ADR_DEBUG_DEAD")) == 2;
				for (int e = 0; e < nElements_global; e++)
				{
					if (dumpCells)
					{
						fprintf(stderr, "ADR_CELL e=%d icase_s=%d icase_f=%d touch=%d", e, gf_s_interior_icase[e], gf_f_interior_icase[e],
								(int)tcDeadOK(2, nf, u_l2g, e));
						for (int i = 0; i < nDOF_trial_element; i++)
						{
							const int g = u_l2g.data()[e * nDOF_trial_element + i];
							fprintf(stderr, " | g=%d (%.4f,%.4f) ps=%.5f pf=%.5f", g, mesh_dof.data()[g * 3 + 0], mesh_dof.data()[g * 3 + 1],
									embeddedBoundary_sdf_nodes.data()[g], immersedBoundary_sdf_nodes.data()[g]);
						}
						fprintf(stderr, "\n");
					}
					if (gf_s_interior_icase[e] == -1)
						nDeadAll++;
					if (gf_s_interior_icase[e] == -1 && gf_f_interior_icase[e] == 0)
					{
						nDeadF++;
						if (tcDeadOK(2, nf, u_l2g, e))
							nTouch++;
					}
				}
				fprintf(stderr, "ADR_DEAD total_cells=%d fully_solid=%d fully_solid_and_cut_by_Gf=%d of_which_touch_a_fluid_bearing_cell=%d never_touch=%d\n",
						nElements_global, nDeadAll, nDeadF, nTouch, nDeadF - nTouch);
			}
			const double dbg_g0 = dbgRowVal();
			for (std::set<int>::iterator it = cutfem_boundaries.begin(); it != cutfem_boundaries.end();)
			{
				if (elementIsActive[elementBoundaryElementsArray[(*it) * 2 + 0]] && elementIsActive[elementBoundaryElementsArray[(*it) * 2 + 1]])
				{
					std::map<int, double> Dwp_Dn_jump, Dw_Dn_jump;
					std::map<int, double> Dw_Dn_jump_a, Dw_Dn_jump_b;
					double gamma_cutfem = embeddedBoundary_ghost_penalty, h_cutfem = elementBoundaryDiameter.data()[*it];
					// The two elements need not parametrize the shared face in the same direction, so
					// index kb can be DIFFERENT physical points on the two sides (see the SCIFEM loop's
					// orientation map). Harmless for P1, where the normal derivative is constant on a
					// face; for P2 it turns the variation along the face into a spurious jump. Match
					// side 1 to side 0 by position.
					int gp_kmap[nQuadraturePoints_elementBoundary];
					{
						double gp_xq[2][nQuadraturePoints_elementBoundary][3] = {}; // 2D mapping never writes z
						for (int s = 0; s < 2; s++)
						{
							const int eN_g = elementBoundaryElementsArray.data()[(*it) * 2 + s],
									  ebN_local_g = elementBoundaryLocalElementBoundariesArray.data()[(*it) * 2 + s];
							for (int kb = 0; kb < nQuadraturePoints_elementBoundary; kb++)
							{
								double jac_g[nSpace * nSpace], jacDet_g, jacInv_g[nSpace * nSpace],
									bJac_g[nSpace * (nSpace - 1)], mT_g[(nSpace - 1) * (nSpace - 1)], mTDS_g, nrm_g[2];
								ck.calculateMapping_elementBoundary(eN_g, ebN_local_g, kb,
																	ebN_local_g * nQuadraturePoints_elementBoundary + kb,
																	mesh_dof.data(), mesh_l2g.data(),
																	mesh_trial_trace_ref.data(), mesh_grad_trial_trace_ref.data(),
																	boundaryJac_ref.data(), jac_g, jacDet_g, jacInv_g,
																	bJac_g, mT_g, mTDS_g, normal_ref.data(), nrm_g,
																	gp_xq[s][kb][0], gp_xq[s][kb][1], gp_xq[s][kb][2]);
							}
						}
						for (int k = 0; k < nQuadraturePoints_elementBoundary; k++)
						{
							int best = 0;
							double bestd = 1.0e300;
							for (int j = 0; j < nQuadraturePoints_elementBoundary; j++)
							{
								double d = 0.0;
								for (int I = 0; I < nSpace; I++)
									d += std::fabs(gp_xq[1][j][I] - gp_xq[0][k][I]);
								if (d < bestd) { bestd = d; best = j; }
							}
							gp_kmap[k] = best;
						}
					}
					for (int kb = 0; kb < nQuadraturePoints_elementBoundary; kb++)
					{
						double Du_Dn_jump = 0.0, dS;
						double Du_Dn_jump_a = 0.0, Du_Dn_jump_b = 0.0;
						bool phaseA_ok = true, phaseB_ok = true;
						for (int eN_side = 0; eN_side < 2; eN_side++)
						{
							int ebN = *it,
								eN = elementBoundaryElementsArray.data()[ebN * 2 + eN_side];
							for (int i = 0; i < nDOF_test_element; i++)
							{
								Dw_Dn_jump[u_l2g.data()[eN * nDOF_test_element + i]] = 0.0;
								Dw_Dn_jump_a[u_l2g.data()[eN * nDOF_test_element + i]] = 0.0;
								Dw_Dn_jump_b[u_l2g.data()[eN * nDOF_test_element + i]] = 0.0;
							}
						}
						for (int eN_side = 0; eN_side < 2; eN_side++)
						{
							int ebN = *it,
								eN = elementBoundaryElementsArray.data()[ebN * 2 + eN_side],
								ebN_local = elementBoundaryLocalElementBoundariesArray.data()[ebN * 2 + eN_side],
								eN_nDOF_trial_element = eN * nDOF_trial_element,
								kb_s = (eN_side == 0) ? kb : gp_kmap[kb],
								ebN_local_kb = ebN_local * nQuadraturePoints_elementBoundary + kb_s,
								ebN_local_kb_nSpace = ebN_local_kb * nSpace;
							double u_int = 0.0,
								   grad_u_int[nSpace] = {0., 0.},
								   jac_int[nSpace * nSpace],
								   jacDet_int,
								   jacInv_int[nSpace * nSpace],
								   boundaryJac[nSpace * (nSpace - 1)],
								   metricTensor[(nSpace - 1) * (nSpace - 1)],
								   metricTensorDetSqrt,
								   u_test_dS[nDOF_test_element],
								   u_grad_trial_trace[nDOF_trial_element * nSpace],
								   u_grad_test_dS[nDOF_trial_element * nSpace],
								   normal[2], x_int, y_int, z_int, xt_int, yt_int, zt_int, integralScaling,
								   G[nSpace * nSpace], G_dd_G, tr_G, h_phi, h_penalty, penalty,
								   force_x, force_y, force_z, force_p_x, force_p_y, force_p_z, force_v_x, force_v_y, force_v_z, r_x, r_y, r_z;
							// compute information about mapping from reference element to physical element
							ck.calculateMapping_elementBoundary(eN,
																ebN_local,
																kb_s,
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
																x_int, y_int, z_int);
							dS = metricTensorDetSqrt * dS_ref.data()[kb_s]; // kb_s mirrors kb: same Gauss weight
							// compute shape and solution information
							// shape
							// std::cout << "Calculating gradTrialFromRef from calculateResidual() 1" << std::endl;
							ck.gradTrialFromRef(&u_grad_trial_trace_ref.data()[ebN_local_kb_nSpace * nDOF_trial_element], jacInv_int, u_grad_trial_trace);
							// solution and gradients
							ck.valFromDOF(u_dof.data(), &u_l2g.data()[eN_nDOF_trial_element], &u_trial_trace_ref.data()[ebN_local_kb * nDOF_test_element], u_int);
							ck.gradFromDOF(u_dof.data(), &u_l2g.data()[eN_nDOF_trial_element], u_grad_trial_trace, grad_u_int);
							for (int I = 0; I < nSpace; I++)
							{
								Du_Dn_jump += grad_u_int[I] * normal[I];
							}
							for (int i = 0; i < nDOF_test_element; i++)
							{
								for (int I = 0; I < nSpace; I++)
									Dw_Dn_jump[u_l2g.data()[eN_nDOF_trial_element + i]] += u_grad_trial_trace[i * nSpace + I] * normal[I];
							}
							// --- Ghost penalty, per fluid phase. The penalty is meant to kill *spurious*
							//     normal-derivative jumps across a cut element's face. On an element that the
							//     fluid interface also cuts, the combined IFEM solution genuinely kinks at
							//     Gamma_f, so penalising the combined gradient penalises physics and destroys
							//     exactness. Each phase's own polynomial extension IS smooth across the face,
							//     so we penalise [[d(u_a)/dn]] and [[d(u_b)/dn]] separately -- the same per-side
							//     treatment the bulk (wA/wB) and Nitsche (D_s_a/D_s_b) terms already get.
							int gp_icase = -2; // -2: no fluid interface in play, single phase
							double gp_va_grad[nDOF_trial_element * nSpace], gp_vb_grad[nDOF_trial_element * nSpace];
							// Jump-lift arrays (assign_jump_side's own node pattern, ~line 1499-1524) for
							// THIS element eN, needed below to add grad_uja/grad_ujb to the ghost penalty's
							// per-phase normal derivative. Zero unless eN is genuinely Gamma_f-cut.
							double gp_JA[nDOF_trial_element], gp_JB[nDOF_trial_element];
							std::fill(gp_JA, gp_JA + nDOF_trial_element, 0.0);
							std::fill(gp_JB, gp_JB + nDOF_trial_element, 0.0);
							if (immersedBoundary)
							{
								double gp_phi_f[nDOF_trial_element], gp_nodes[nDOF_trial_element * 3];
								for (int i = 0; i < nDOF_trial_element; i++)
								{
									const int gp_eN_i = eN * nDOF_trial_element + i;
									gp_phi_f[i] = immersedBoundary_sdf_nodes.data()[u_l2g.data()[gp_eN_i]];
									for (int I = 0; I < 3; I++)
										gp_nodes[i * 3 + I] = mesh_dof.data()[u_l2g.data()[gp_eN_i] * 3 + I];
								}
								if (gf_f_boundary_gen[eN] != ifemGeometryGeneration)
								{
									gf_f_boundary_icase[eN] = gf_f_cache[eN].calculate(gp_phi_f, gp_nodes, xB_ref_faces, mua, mub, jf, true, false);
									gf_f_boundary_gen[eN] = ifemGeometryGeneration;
								}
								gp_icase = gf_f_boundary_icase[eN];
								if (gp_icase == 0)
								{
									for (int i = 0; i < nDOF_trial_element; i++)
									{
										const double jump_i = immersedBoundary_solutionJump_nodes.data()[u_l2g.data()[eN * nDOF_trial_element + i]];
										if (gp_phi_f[i] > 0.0)
											gp_JA[i] = -jump_i;
										else
											gp_JB[i] = jump_i;
									}
								}
							}
							if (gp_icase == 0)
							{
								gf_f_cache[eN].set_boundary_quad(ebN_local_kb);
								for (int i = 0; i < nDOF_trial_element; i++)
								{
									gp_va_grad[i * nSpace + 0] = gf_f_cache[eN].VA_x(i);
									gp_va_grad[i * nSpace + 1] = gf_f_cache[eN].VA_y(i);
									gp_vb_grad[i * nSpace + 0] = gf_f_cache[eN].VB_x(i);
									gp_vb_grad[i * nSpace + 1] = gf_f_cache[eN].VB_y(i);
								}
							}
							else
								for (int i = 0; i < nDOF_trial_element * nSpace; i++)
								{
									gp_va_grad[i] = u_grad_trial_trace[i];
									gp_vb_grad[i] = u_grad_trial_trace[i];
								}
							// Which phases does this element actually carry? A face only gets a phase's penalty
							// if BOTH its elements have that phase, otherwise there is no extension to compare.
							phaseA_ok = phaseA_ok && (gp_icase == 0 || gp_icase == -1 || gp_icase == -2);
							phaseB_ok = phaseB_ok && (gp_icase == 0 || gp_icase == 1);
							for (int i = 0; i < nDOF_test_element; i++)
							{
								const int gp_dof = u_l2g.data()[eN * nDOF_trial_element + i];
								for (int I = 0; I < nSpace; I++)
								{
									Dw_Dn_jump_a[gp_dof] += gp_va_grad[i * nSpace + I] * normal[I];
									Dw_Dn_jump_b[gp_dof] += gp_vb_grad[i * nSpace + I] * normal[I];
								}
							}
							{
								double gp_eu[nDOF_trial_element], gp_gua[nSpace] = {0., 0.}, gp_gub[nSpace] = {0., 0.};
								for (int i = 0; i < nDOF_trial_element; i++)
									gp_eu[i] = u_dof.data()[u_l2g.data()[eN_nDOF_trial_element + i]];
								// Triply-cut: substitute the analytic boundary value for the masked node's
								// otherwise-meaningless raw DOF, matching element_u_ua's own override (~line
								// 691-695); no-op elsewhere.
								const int gp_masked = triplyCut_maskedLocalNode.data()[eN];
								if (gp_masked >= 0)
									gp_eu[gp_masked] = triplyCut_overrideValue.data()[eN];
								ck.gradFromElementDOF(gp_eu, gp_va_grad, gp_gua);
								ck.gradFromElementDOF(gp_eu, gp_vb_grad, gp_gub);
								// Add the jump-lift gradient (grad_uja/grad_ujb): the ghost penalty must
								// compare the TOTAL branch field (u_a+u_ja / u_b+u_jb, see ga/gb ~line
								// 865-866) across the shared face, not the jump-free piece alone, or a
								// genuine prescribed solution jump reads as a spurious normal-derivative
								// kink between elements. Zero (no-op) unless eN is Gamma_f-cut.
								double gp_gja[nSpace] = {0., 0.}, gp_gjb[nSpace] = {0., 0.};
								ck.gradFromElementDOF(gp_JA, gp_va_grad, gp_gja);
								ck.gradFromElementDOF(gp_JB, gp_vb_grad, gp_gjb);
								if (gp_icase == 0) // flux-jump lift; set_boundary_quad(ebN_local_kb) is current
								{
									const double J = immersedBoundary_fluxJump_element.data()[eN];
									gp_gja[0] += J * gf_f_cache[eN].PSIA_x();
									gp_gja[1] += J * gf_f_cache[eN].PSIA_y();
									gp_gjb[0] += J * gf_f_cache[eN].PSIB_x();
									gp_gjb[1] += J * gf_f_cache[eN].PSIB_y();
								}
								for (int I = 0; I < nSpace; I++)
								{
									Du_Dn_jump_a += (gp_gua[I] + gp_gja[I]) * normal[I];
									Du_Dn_jump_b += (gp_gub[I] + gp_gjb[I]) * normal[I];
								}
							}
						} // eN_side
						if (phaseA_ok || phaseB_ok)
						{
							if (phaseA_ok)
								for (std::map<int, double>::iterator w_it = Dw_Dn_jump_a.begin(); w_it != Dw_Dn_jump_a.end(); ++w_it)
									globalResidual.data()[offset_u + stride_u * w_it->first] += gpMuA * gamma_cutfem * h_cutfem * Du_Dn_jump_a * w_it->second * dS;
							if (phaseB_ok)
								for (std::map<int, double>::iterator w_it = Dw_Dn_jump_b.begin(); w_it != Dw_Dn_jump_b.end(); ++w_it)
									globalResidual.data()[offset_u + stride_u * w_it->first] += gpMuB * gamma_cutfem * h_cutfem * Du_Dn_jump_b * w_it->second * dS;
						}
						else
						{
							// Neither phase is shared by both elements (Gamma_f runs along the face itself).
							// Keep the original combined penalty so the face is still stabilised.
							for (std::map<int, double>::iterator w_it = Dw_Dn_jump.begin(); w_it != Dw_Dn_jump.end(); ++w_it)
								globalResidual.data()[offset_u + stride_u * w_it->first] += gpMuAB * gamma_cutfem * h_cutfem * Du_Dn_jump * w_it->second * dS;
						} // i
					} // kb
					++it;
				}
				else
				{
					it = cutfem_boundaries.erase(it);
				}
			} // cutfem element boundaries
			dbgDelta[5] += dbgRowVal() - dbg_g0;
			const double dbg_s0 = dbgRowVal();
			// SCIFEM (Ji et al. 2014, eq. 3.7/3.8) symmetric consistency + adjoint-consistency terms
			// on cut edges: -switch*({{beta grad u}}.n_e*[[v]] + {{beta grad v}}.n_e*[[u]]).
			// A genuine 1D facet term -- the edge carries its own arc-length measure -- so no Dirac
			// surrogate is involved. The two region-wise integrands are formed separately and blended
			// by the edge Heaviside fit (see the prologue below), which is what makes this exact for
			// P1 and P2 without splitting the quadrature.
			for (std::set<int>::iterator it = ifem_boundaries.begin(); it != ifem_boundaries.end();)
			{
				if (elementIsActive[elementBoundaryElementsArray[(*it) * 2 + 0]] && elementIsActive[elementBoundaryElementsArray[(*it) * 2 + 1]])
				{
				int ebN = *it;
					int eN_s[2], ebN_local_s[2];
					for (int s = 0; s < 2; s++)
					{
						eN_s[s] = elementBoundaryElementsArray.data()[ebN * 2 + s];
						ebN_local_s[s] = elementBoundaryLocalElementBoundariesArray.data()[ebN * 2 + s];
					}

					// --- Edge-restricted phi (both level sets) --------------------------------
					double edge_nodes[2][3], phi_edge[2], phi_s_edge[2];
					{
						const int n0l = (ebN_local_s[0] + 1) % 3, n1l = (ebN_local_s[0] + 2) % 3;
						const int ln[2] = {n0l, n1l};
						for (int e = 0; e < 2; e++)
						{
							int eN_i = eN_s[0] * nDOF_trial_element + ln[e];
							phi_edge[e] = immersedBoundary_sdf_nodes.data()[u_l2g.data()[eN_i]];
							phi_s_edge[e] = embeddedBoundary ? embeddedBoundary_sdf_nodes.data()[u_l2g.data()[eN_i]] : 1.0;
							for (int I = 0; I < 3; I++)
								edge_nodes[e][I] = mesh_dof.data()[u_l2g.data()[eN_i] * 3 + I];
						}
					}
					// edge diameter for the interior-penalty scaling (same measure the cutfem ghost
					// penalty uses); note gamma/h here for a VALUE jump, vs gamma*h there for a
					// gradient jump.
					const double h_edge = elementBoundaryDiameter.data()[ebN];
					double edge_vec[3];
					double edge_len2 = 0.0;
					for (int I = 0; I < 3; I++)
					{
						edge_vec[I] = edge_nodes[1][I] - edge_nodes[0][I];
						edge_len2 += edge_vec[I] * edge_vec[I];
					}

					// --- Orientation map ------------------------------------------------------
					// The two elements sharing an interior face do NOT necessarily parametrize it in
					// the same direction, so quadrature index kb can denote DIFFERENT physical points
					// on the two sides (measured: 4 of 14 cut edges at test=8.0/refinement=1, with
					// exactly the mirror offsets |1-2t_k|*L of the Gauss rule). Pairing the sides by
					// raw kb then differences u at two different points, so [[u]] and {{beta grad u}}
					// pick up u's variation ALONG the edge instead of the genuine inter-element jump.
					// Build an explicit map instead: kmap[s][k] is side s's index for the physical
					// point that side 0 calls k. Matched by position, so no assumption about the
					// rule being symmetric or ascending.
					double xq[2][nQuadraturePoints_elementBoundary][3];
					for (int s = 0; s < 2; s++)
						for (int kb = 0; kb < nQuadraturePoints_elementBoundary; kb++)
						{
							double jac_t[nSpace * nSpace], jacDet_t, jacInv_t[nSpace * nSpace],
								   bJac_t[nSpace * (nSpace - 1)], mT_t[(nSpace - 1) * (nSpace - 1)],
								   mTDS_t, nrm_t[2], xt_ = 0.0, yt_ = 0.0, zt_ = 0.0;
							ck.calculateMapping_elementBoundary(eN_s[s], ebN_local_s[s], kb,
																ebN_local_s[s] * nQuadraturePoints_elementBoundary + kb,
																mesh_dof.data(), mesh_l2g.data(),
																mesh_trial_trace_ref.data(), mesh_grad_trial_trace_ref.data(),
																boundaryJac_ref.data(), jac_t, jacDet_t, jacInv_t,
																bJac_t, mT_t, mTDS_t, normal_ref.data(), nrm_t,
																xt_, yt_, zt_);
							xq[s][kb][0] = xt_; xq[s][kb][1] = yt_; xq[s][kb][2] = zt_;
						}
					// edge diameter: the length scale for the pairing tolerance just below,
					// and for the interior-penalty scaling further down (same measure the
					// cutfem ghost penalty uses); note gamma/h here for a VALUE jump, vs
					// gamma*h there for a gradient jump.
					// Match over nSpace components only: a component the mapping does not
					// define contributes nothing but noise to the distance, and the paired
					// points agree EXACTLY in the components it does define, so the winning
					// distance is 0 and the runner-up is a full quadrature spacing away.
					// Verify that outcome rather than trust it -- a mis-pick here silently
					// differences u at two different points, and assert() is compiled out by
					// the -DNDEBUG that reaches these translation units.
					int kmap[2][nQuadraturePoints_elementBoundary];
					bool kmap_used[nQuadraturePoints_elementBoundary] = {};
					for (int k = 0; k < nQuadraturePoints_elementBoundary; k++)
					{
						kmap[0][k] = k;
						int best = -1;
						double bestd = 1.0e300, nextd = 1.0e300;
						for (int j = 0; j < nQuadraturePoints_elementBoundary; j++)
						{
							double d = 0.0;
							for (int I = 0; I < nSpace; I++) // the 2D mapping never writes z
								d += std::fabs(xq[1][j][I] - xq[0][k][I]);
							if (d < bestd) { nextd = bestd; bestd = d; best = j; }
							else if (d < nextd) { nextd = d; }
						}
						if (best < 0 || kmap_used[best] || bestd >= 1.0e-8 * h_edge || nextd <= 2.0 * bestd)
						{
							std::cerr << "ADR immersedSCIFEM: quadrature points on face " << ebN
									  << " between elements " << eN_s[0] << " and " << eN_s[1]
									  << " do not pair up (k=" << k << ", best=" << best
									  << ", d=" << bestd << ", runner-up=" << nextd
									  << ", h=" << h_edge << "); the mesh is non-conforming or the"
									  << " boundary mapping is inconsistent." << std::endl;
							throw std::runtime_error("ADR immersedSCIFEM: face quadrature pairing failed");
						}
						kmap_used[best] = true;
						kmap[1][k] = best;
					}

					// --- Edge Heaviside moment fit -------------------------------------------
					// The facet integrand is DISCONTINUOUS at the interface crossing theta, so a fixed
					// whole-edge Gauss rule cannot integrate it directly. Weight the two one-sided
					// integrands by an edge-restricted moment-fit Heaviside instead: exact for any
					// polynomial integrand of degree <= nP_edge, hence exact for P1 (degree <= 1) and
					// P2 (degree <= 3) alike -- no linearity assumption, unlike a split-quadrature
					// scheme. ONE fit per edge, in side 0's parametrization: the two sides' fits are
					// mirror images, moment-equivalent but not pointwise equal, so they must not be
					// mixed. t below is therefore always measured along side 0's edge direction.
					// Branch a = {phi_f<0}, branch b = {phi_f>0}, each intersected with the fluid side
					// {phi_s>0} (phi_s_edge is 1 without an embedded solid, so this is then exactly the
					// single Gamma_f fit and its complement).
					double C_a_edge[nP_edge + 1], C_b_edge[nP_edge + 1];
					equivalent_polynomials::calculate_edge_interval_H<nP_edge>(-phi_edge[0], -phi_edge[1], phi_s_edge[0], phi_s_edge[1], C_a_edge);
					equivalent_polynomials::calculate_edge_interval_H<nP_edge>(phi_edge[0], phi_edge[1], phi_s_edge[0], phi_s_edge[1], C_b_edge);

					for (int k = 0; k < nQuadraturePoints_elementBoundary; k++)
					{
						double dS = 0.0;
						double ua_s[2], ub_s[2], flux_a_s[2], flux_b_s[2];
						std::map<int, double> va_m[2], vb_m[2], fta_m[2], ftb_m[2];
						for (int eN_side = 0; eN_side < 2; eN_side++)
						{
							int eN = eN_s[eN_side],
								ebN_local = ebN_local_s[eN_side],
								kb = kmap[eN_side][k],
								eN_nDOF_trial_element = eN * nDOF_trial_element,
								ebN_local_kb = ebN_local * nQuadraturePoints_elementBoundary + kb;
							double ua = 0.0, ub = 0.0,
								   grad_ua[nSpace] = {0., 0.}, grad_ub[nSpace] = {0., 0.},
								   jac_int[nSpace * nSpace], jacDet_int, jacInv_int[nSpace * nSpace],
								   boundaryJac[nSpace * (nSpace - 1)],
								   metricTensor[(nSpace - 1) * (nSpace - 1)], metricTensorDetSqrt,
								   normal[2], x_int, y_int, z_int;
							ck.calculateMapping_elementBoundary(eN, ebN_local, kb, ebN_local_kb,
																mesh_dof.data(), mesh_l2g.data(),
																mesh_trial_trace_ref.data(), mesh_grad_trial_trace_ref.data(),
																boundaryJac_ref.data(), jac_int, jacDet_int, jacInv_int,
																boundaryJac, metricTensor, metricTensorDetSqrt,
																normal_ref.data(), normal, x_int, y_int, z_int);
							// dS and the reference normal n_e are fixed from side 0; side 1's own
							// outward normal is -n_e, which sign_ne converts back.
							if (eN_side == 0)
								dS = metricTensorDetSqrt * dS_ref.data()[kb];
							double sign_ne = (eN_side == 0) ? 1.0 : -1.0;

							auto element_u = xt::pyarray<double>::from_shape({nDOF_trial_element});
							double element_phi_f[nDOF_trial_element], element_nodes[nDOF_trial_element * 3];
							for (int i = 0; i < nDOF_trial_element; i++)
							{
								int eN_i = eN * nDOF_trial_element + i;
								element_u.data()[i] = u_dof.data()[u_l2g.data()[eN_i]];
								element_phi_f[i] = immersedBoundary_sdf_nodes.data()[u_l2g.data()[eN_i]];
								for (int I = 0; I < 3; I++)
									element_nodes[i * 3 + I] = mesh_dof.data()[u_l2g.data()[eN_i] * 3 + I];
							}
							if (gf_f_boundary_gen[eN] != ifemGeometryGeneration)
							{
								gf_f_boundary_icase[eN] = gf_f_cache[eN].calculate(element_phi_f, element_nodes, xB_ref_faces, mua, mub, jf, true, false);
								gf_f_boundary_gen[eN] = ifemGeometryGeneration;
							}
							GfType &gf_f = gf_f_cache[eN];
							gf_f.set_boundary_quad(ebN_local_kb);

							double va[nDOF_trial_element], va_grad_trial[nDOF_trial_element * nSpace],
								   vb[nDOF_trial_element], vb_grad_trial[nDOF_trial_element * nSpace];
							for (int i = 0; i < nDOF_trial_element; i++)
							{
								va[i] = gf_f.VA(i);
								va_grad_trial[i * nSpace + 0] = gf_f.VA_x(i);
								va_grad_trial[i * nSpace + 1] = gf_f.VA_y(i);
								vb[i] = gf_f.VB(i);
								vb_grad_trial[i * nSpace + 0] = gf_f.VB_x(i);
								vb_grad_trial[i * nSpace + 1] = gf_f.VB_y(i);
							}
							// Solution-jump lift (same node pattern as assign_jump_side / the ghost penalty's
							// gp_JA/gp_JB): branch a is u + JA, branch b is u + JB. Without it the edge terms
							// see u^h instead of u^h + j_h and are inconsistent when [u] != 0.
							double ca_s[nDOF_trial_element], cb_s[nDOF_trial_element];
							for (int i = 0; i < nDOF_trial_element; i++)
							{
								ca_s[i] = element_u.data()[i];
								cb_s[i] = element_u.data()[i];
							}
							if (gf_f_boundary_icase[eN] == 0)
							{
								for (int i = 0; i < nDOF_trial_element; i++)
								{
									const double jump_i = immersedBoundary_solutionJump_nodes.data()[u_l2g.data()[eN * nDOF_trial_element + i]];
									if (element_phi_f[i] > 0.0)
										ca_s[i] -= jump_i;
									else
										cb_s[i] += jump_i;
								}
							}
							ck.valFromElementDOF(ca_s, va, ua);
							ck.gradFromElementDOF(ca_s, va_grad_trial, grad_ua);
							ck.valFromElementDOF(cb_s, vb, ub);
							ck.gradFromElementDOF(cb_s, vb_grad_trial, grad_ub);
							if (gf_f_boundary_icase[eN] == 0) // flux-jump lift, as for the solution jump
							{
								const double J = immersedBoundary_fluxJump_element.data()[eN];
								ua += J * gf_f.PSIA();
								ub += J * gf_f.PSIB();
								grad_ua[0] += J * gf_f.PSIA_x();
								grad_ua[1] += J * gf_f.PSIA_y();
								grad_ub[0] += J * gf_f.PSIB_x();
								grad_ub[1] += J * gf_f.PSIB_y();
							}

							ua_s[eN_side] = ua;
							ub_s[eN_side] = ub;
							double fa = 0.0, fb = 0.0;
							for (int I = 0; I < nSpace; I++)
							{
								fa += mua * grad_ua[I] * normal[I];
								fb += mub * grad_ub[I] * normal[I];
							}
							flux_a_s[eN_side] = sign_ne * fa;
							flux_b_s[eN_side] = sign_ne * fb;

							for (int i = 0; i < nDOF_test_element; i++)
							{
								int dof_i = u_l2g.data()[eN_nDOF_trial_element + i];
								va_m[eN_side][dof_i] = va[i];
								vb_m[eN_side][dof_i] = vb[i];
								double fta = 0.0, ftb = 0.0;
								for (int I = 0; I < nSpace; I++)
								{
									fta += mua * va_grad_trial[i * nSpace + I] * normal[I];
									ftb += mub * vb_grad_trial[i * nSpace + I] * normal[I];
								}
								fta_m[eN_side][dof_i] = sign_ne * fta;
								ftb_m[eN_side][dof_i] = sign_ne * ftb;
							}
						} // eN_side

						// t of this physical point along side 0's edge direction, then H_hat(t)
						double proj = 0.0;
						for (int I = 0; I < 3; I++)
							proj += (xq[0][k][I] - edge_nodes[0][I]) * edge_vec[I];
						double t_edge = (edge_len2 > 1.0e-24) ? (proj / edge_len2) : 0.0;
						// H_e / ImH_e: weights of branch b / branch a on its fluid piece of the edge
						double H_e = equivalent_polynomials::evaluate_edge_poly<nP_edge>(C_b_edge, t_edge);
						double ImH_e = equivalent_polynomials::evaluate_edge_poly<nP_edge>(C_a_edge, t_edge);

						double avg_flux_a = 0.5 * (flux_a_s[0] + flux_a_s[1]);
						double avg_flux_b = 0.5 * (flux_b_s[0] + flux_b_s[1]);
						double jump_ua = ua_s[0] - ua_s[1];
						double jump_ub = ub_s[0] - ub_s[1];

						// Blend the region-wise PRODUCTS, never the individual factors:
						// (ImH*A_f + H*B_f)*(ImH*A_v + H*B_v) != ImH*A_f*A_v + H*B_f*B_v.
						for (int side = 0; side < 2; side++)
						{
							double v_sign = (side == 0) ? 1.0 : -1.0;
							for (std::map<int, double>::iterator w = va_m[side].begin(); w != va_m[side].end(); ++w)
							{
								int dof_i = w->first;
								double jva_i = v_sign * va_m[side][dof_i], jvb_i = v_sign * vb_m[side][dof_i];
								double Pa = avg_flux_a * jva_i + 0.5 * fta_m[side][dof_i] * jump_ua;
								double Pb = avg_flux_b * jvb_i + 0.5 * ftb_m[side][dof_i] * jump_ub;
								globalResidual.data()[offset_u + stride_u * dof_i] -=
									immersedSCIFEM_switch * (ImH_e * Pa + H_e * Pb) * dS;
								// Interior-penalty stabilization gamma/h * int_e [[u]][[v]] ds. eq (3.7) is
								// consistency + adjoint only, so it carries no coercivity guarantee; this is
								// the standard remedy. Added with a PLUS sign (like the bulk stiffness) so it
								// is positive semi-definite, and blended by the same edge Heaviside so it
								// stays exact for P1 (integrand degree <= 2) and P2 (degree <= 4 = nP_edge).
								// Weighted by mu of each branch (like the Nitsche term), so the penalty is
								// of the size of the bulk term on both sides of a high-contrast interface.
								globalResidual.data()[offset_u + stride_u * dof_i] +=
									(immersedSCIFEM_penalty / h_edge) *
									(ImH_e * mua * jump_ua * jva_i + H_e * mub * jump_ub * jvb_i) * dS;
							}
						}
					} // k
					++it;
				}
				else
				{
					it = ifem_boundaries.erase(it);
				}
			} // ifem element boundaries
			dbgDelta[6] += dbgRowVal() - dbg_s0;
			const double dbg_x0 = dbgRowVal();
			//
			// loop over exterior element boundaries to calculate surface integrals and load into element and global residuals
			//
			// ebNE is the Exterior element boundary INdex
			// ebN is the element boundary INdex
			// eN is the element index
			for (int ebNE = 0; ebNE < nExteriorElementBoundaries_global; ebNE++)
			{
				int ebN = exteriorElementBoundariesArray.data()[ebNE],
					eN = elementBoundaryElementsArray.data()[ebN * 2 + 0],
					ebN_local = elementBoundaryLocalElementBoundariesArray.data()[ebN * 2 + 0],
					eN_nDOF_trial_element = eN * nDOF_trial_element;
				double elementResidual_u[nDOF_test_element];
				for (int i = 0; i < nDOF_test_element; i++)
				{
					elementResidual_u[i] = 0.0;
				}
				for (int kb = 0; kb < nQuadraturePoints_elementBoundary; kb++)
				{
					int ebNE_kb = ebNE * nQuadraturePoints_elementBoundary + kb,
						ebNE_kb_nSpace = ebNE_kb * nSpace,
						ebN_local_kb = ebN_local * nQuadraturePoints_elementBoundary + kb,
						ebN_local_kb_nSpace = ebN_local_kb * nSpace;
					double u_ext = 0.0,
						   grad_u_ext[nSpace],
						   m_ext = 0.0,
						   dm_ext = 0.0,
						   *a_ext,
						   /* *da_exxt, */
						f_ext[nSpace],
						   df_ext[nSpace],
						   r_ext = 0.0,
						   /* dr_ext=0.0, */
						flux_diff_ext = 0.0,
						   flux_advect_ext = 0.0,
						   bc_u_ext = 0.0,
						   // bc_grad_u_ext[nSpace],
						bc_m_ext = 0.0,
						   bc_dm_ext = 0.0,
						   bc_f_ext[nSpace],
						   bc_df_ext[nSpace],
						   jac_ext[nSpace * nSpace],
						   jacDet_ext,
						   jacInv_ext[nSpace * nSpace],
						   boundaryJac[nSpace * (nSpace - 1)],
						   metricTensor[(nSpace - 1) * (nSpace - 1)],
						   metricTensorDetSqrt,
						   dS,
						   u_test_dS[nDOF_test_element],
						   u_grad_trial_trace[nDOF_trial_element * nSpace],
						   u_grad_test_dS[nDOF_trial_element * nSpace],
						   normal[nSpace], x_ext, y_ext, z_ext, xt_ext, yt_ext, zt_ext, integralScaling,
						   //
						G[nSpace * nSpace], G_dd_G, tr_G;
					//
					// calculate the solution and gradients at quadrature points
					//
					// compute information about mapping from reference element to physical element
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
														x_ext, y_ext, z_ext);
					/* ck.calculateMappingVelocity_elementBoundary(eN, */
					/*                           ebN_local, */
					/*                           kb, */
					/*                           ebN_local_kb, */
					/*                           mesh_velocity_dof, */
					/*                           mesh_l2g, */
					/*                           mesh_trial_trace_ref, */
					/*                           xt_ext,yt_ext,zt_ext, */
					/*                           normal, */
					/*                           boundaryJac, */
					/*                           metricTensor, */
					/*                           integralScaling); */
					/*dS = ((1.0-MOVING_DOMAIN)*metricTensorDetSqrt + MOVING_DOMAIN*integralScaling)*dS_ref.data()[kb];*/
					dS = metricTensorDetSqrt * dS_ref.data()[kb];
					// get the metric tensor
					// cek todo use symmetry
					ck.calculateG(jacInv_ext, G, G_dd_G, tr_G);
					// compute shape and solution information
					// shape
					// std::cout << "Calculating gradTrialFromRef from calculateResidual() 3" << std::endl;
					ck.gradTrialFromRef(&u_grad_trial_trace_ref.data()[ebN_local_kb_nSpace * nDOF_trial_element], jacInv_ext, u_grad_trial_trace);
					// solution and gradients
					ck.valFromDOF(u_dof.data(), &u_l2g.data()[eN_nDOF_trial_element], &u_trial_trace_ref.data()[ebN_local_kb * nDOF_test_element], u_ext);
					ck.gradFromDOF(u_dof.data(), &u_l2g.data()[eN_nDOF_trial_element], u_grad_trial_trace, grad_u_ext);
					// precalculate test function products with integration weights
					for (int j = 0; j < nDOF_trial_element; j++)
					{
						u_test_dS[j] = u_test_trace_ref.data()[ebN_local_kb * nDOF_test_element + j] * dS;
						for (int I = 0; I < nSpace; I++)
							u_grad_test_dS[j * nSpace + I] = u_grad_trial_trace[j * nSpace + I] * dS; // cek hack, using trial
					}
					//
					// load the boundary values
					//
					bc_u_ext = isDOFBoundary_u.data()[ebNE_kb] * ebqe_bc_u_ext.data()[ebNE_kb] + (1 - isDOFBoundary_u.data()[ebNE_kb]) * u_ext;
					//
					//
					// calculate the pde coefficients using the solution and the boundary values for the solution
					//
					a_ext = &ebqe_a.data()[ebNE_kb * sd_rowptr[nSpace]];
					for (int I = 0; I < nSpace; I++)
					{
						f_ext[I] = ebqe_v.data()[ebNE_kb * nSpace + I] * u_ext;
						df_ext[I] = ebqe_v.data()[ebNE_kb * nSpace + I];
						bc_f_ext[I] = ebqe_v.data()[ebNE_kb * nSpace + I] * bc_u_ext;
						bc_df_ext[I] = ebqe_v.data()[ebNE_kb * nSpace + I];
					}
					/* evaluateCoefficients(&ebqe_velocity_ext.data()[ebNE_kb_nSpace], */
					/*                u_ext, */
					/*                //VRANS */
					/*                porosity_ext, */
					/*                // */
					/*                m_ext, */
					/*                dm_ext, */
					/*                f_ext, */
					/*                df_ext); */
					/* evaluateCoefficients(&ebqe_velocity_ext.data()[ebNE_kb_nSpace], */
					/*                bc_u_ext, */
					/*                //VRANS */
					/*                porosity_ext, */
					/*                // */
					/*                bc_m_ext, */
					/*                bc_dm_ext, */
					/*                bc_f_ext, */
					/*                bc_df_ext);     */
					//
					// moving mesh
					//
					/* double velocity_ext[nSpace]; */
					/* double mesh_velocity[3]; */
					/* mesh_velocity[0] = xt_ext; */
					/* mesh_velocity[1] = yt_ext; */
					/* mesh_velocity[2] = zt_ext; */
					/* for (int I=0;I<nSpace;I++) */
					/*     velocity_ext[I] = ebqe_velocity_ext.data()[ebNE_kb_nSpace+0] - MOVING_DOMAIN*mesh_velocity[I]; */
					//
					// calculate the numerical fluxes
					//
					exteriorNumericalDiffusiveFlux(sd_rowptr.data(),
												   sd_colind.data(),
												   isDOFBoundary_u.data()[ebNE_kb],
												   isDiffusiveFluxBoundary_u.data()[ebNE_kb],
												   normal,
												   a_ext,
												   bc_u_ext,
												   ebqe_bc_flux_u_ext.data()[ebNE_kb],
												   a_ext,
												   grad_u_ext,
												   u_ext,
												   ebqe_penalty_ext.data()[ebNE_kb],
												   flux_diff_ext);
					exteriorNumericalAdvectiveFlux(isDOFBoundary_u.data()[ebNE_kb],
												   isAdvectiveFluxBoundary_u.data()[ebNE_kb],
												   normal,
												   bc_u_ext,
												   ebqe_bc_flux_u_ext.data()[ebNE_kb],
												   u_ext,
												   df_ext,
												   flux_advect_ext);
					// ebqe_flux.data()[ebNE_kb] = flux_ext;
					//
					// update residuals
					//
					for (int i = 0; i < nDOF_test_element; i++)
					{
						// int ebNE_kb_i = ebNE_kb*nDOF_test_element+i;
						elementResidual_u[i] += ck.ExteriorElementBoundaryFlux(flux_diff_ext + flux_advect_ext, u_test_dS[i]) +
												ck.ExteriorElementBoundaryDiffusionAdjoint(isDOFBoundary_u.data()[ebNE_kb],
																						   isDiffusiveFluxBoundary_u.data()[ebNE_kb],
																						   eb_adjoint_sigma,
																						   u_ext,
																						   bc_u_ext,
																						   normal,
																						   sd_rowptr.data(),
																						   sd_colind.data(),
																						   a_ext,
																						   &u_grad_test_dS[i * nSpace]);
					} // i
				} // kb
				//
				// update the element and global residual storage
				//
				for (int i = 0; i < nDOF_test_element; i++)
				{
					int eN_i = eN * nDOF_test_element + i;
					globalResidual.data()[offset_u + stride_u * u_l2g.data()[eN_i]] += elementResidual_u[i];
				} // i
			} // ebNE
			if (dbgRow >= 0)
			{
				dbgDelta[7] = dbgRowVal() - dbg_x0;
				fprintf(stderr, "ADR_ROW_SUMMARY row=%d total=%.6e | bulk=%.6e nitsche_pen=%.6e nitsche_cons=%.6e nitsche_adj=%.6e "
						"tc_edge=%.6e ghost=%.6e scifem=%.6e exterior=%.6e\n",
						dbgRow, dbgRowVal(), dbgDelta[0], dbgDelta[1], dbgDelta[2], dbgDelta[3],
						dbgDelta[4], dbgDelta[5], dbgDelta[6], dbgDelta[7]);
			}
			if (getenv("ADR_DEBUG_ROWS"))
			{
				// Interior nodes only: exterior-boundary rows are overwritten by the strong BC anyway.
				for (int g = 0; g < (int)u_dof.size(); g++)
				{
					const double xg = mesh_dof.data()[g * 3 + 0], yg = mesh_dof.data()[g * 3 + 1];
					const double rg = globalResidual.data()[offset_u + stride_u * g];
					if (fabs(xg) < 0.999 && fabs(yg) < 0.999 && fabs(rg) > 1.0e-9)
						fprintf(stderr, "ADR_ROWS g=%d x=%.6f y=%.6f phi_s=%.6f phi_f=%.6f r=%.6e\n", g, xg, yg,
								embeddedBoundary ? embeddedBoundary_sdf_nodes.data()[g] : 0.0,
								immersedBoundary_sdf_nodes.data()[g], rg);
				}
			}
		}

		inline void calculateElementJacobian(int icase_f,
											 const int demoted_f,
											 // element
											 xt::pyarray<double> &mesh_trial_ref,
											 xt::pyarray<double> &mesh_grad_trial_ref,
											 xt::pyarray<double> &mesh_dof,
											 xt::pyarray<int> &mesh_l2g,
											 xt::pyarray<double> &x_ref,
											 xt::pyarray<double> &dV_ref,
											 xt::pyarray<double> &u_trial_ref,
											 xt::pyarray<double> &u_grad_trial_ref,
											 xt::pyarray<double> &u_test_ref,
											 xt::pyarray<double> &u_grad_test_ref,
											 xt::pyarray<double> &elementDiameter,
											 xt::pyarray<double> &elementBoundaryDiameter,
											 xt::pyarray<double> &nodeDiametersArray,
											 xt::pyarray<double> &cfl,
											 double Ct_sge,
											 double sc_uref,
											 double sc_alpha,
											 double useMetrics,
											 // element boundary
											 xt::pyarray<double> &mesh_trial_trace_ref,
											 xt::pyarray<double> &mesh_grad_trial_trace_ref,
											 xt::pyarray<double> &dS_ref,
											 xt::pyarray<double> &u_trial_trace_ref,
											 xt::pyarray<double> &u_grad_trial_trace_ref,
											 xt::pyarray<double> &u_test_trace_ref,
											 xt::pyarray<double> &u_grad_test_trace_ref,
											 xt::pyarray<double> &normal_ref,
											 xt::pyarray<double> &boundaryJac_ref,
											 // physics
											 int nElements_global,
											 int nElementBoundaries_owned,
											 xt::pyarray<int> &u_l2g,
											 xt::pyarray<double> &u_dof,
											 xt::pyarray<int> &sd_rowptr,
											 xt::pyarray<int> &sd_colind,
											 xt::pyarray<double> &q_a,
											 const double *comp_A,
											 const double *comp_B,
											 const double *comp_DA,
											 const double *comp_DB,
											 const double *comp_DFa,
											 xt::pyarray<double> &q_v,
											 xt::pyarray<double> &q_r,
											 int lag_shockCapturing,
											 double shockCapturingDiffusion,
											 xt::pyarray<double> &q_numDiff_u,
											 xt::pyarray<double> &q_numDiff_u_last,
											 xt::pyarray<double> &elementJacobian_u_u,
											 xt::pyarray<double> &element_u,
											 int eN,
											 const bool embeddedBoundary,
											 const double embeddedBoundary_penalty,
											 xt::pyarray<double> &embeddedBoundary_normal_q,
											 xt::pyarray<double> &embeddedBoundary_u_q,
											 xt::pyarray<double> &embeddedBoundary_u_inner_q,
											 xt::pyarray<double> &embeddedBoundary_u_outer_q,
											 const bool embeddedBoundary_robin,
											 const double embeddedBoundary_robin_h_inner,
											 const double embeddedBoundary_robin_h_outer,
											 xt::pyarray<double> &embeddedBoundary_robin_uInfinity_inner_q,
											 xt::pyarray<double> &embeddedBoundary_robin_uInfinity_outer_q,
											 const bool immersedBoundary,
											 const double immersedBoundary_penalty,
											 xt::pyarray<double> &immersedBoundary_sdf_q,
											 xt::pyarray<double> &immersedBoundary_normal_q,
											 xt::pyarray<double> &immersedBoundary_u_q,
											 xt::pyarray<double> &immersedBoundary_fluxJump_q,
											 xt::pyarray<double> &immersedBoundary_fluxJumpVector_q,
											 xt::pyarray<double> &immersedBoundary_solutionJump_nodes,
											 double *element_phi_f,
											 double mua,
											 double mub,
											 const bool PG,
											 const bool pgElem)
		{
			// per-element cached equivalent-polynomial/IFEM reconstruction
			// (see ensureIFEMCacheSized / ifemGeometryGeneration)
			GfType &gf_f = gf_f_cache[eN];
			GfType &gf_s = gf_s_cache[eN];
			for (int i = 0; i < nDOF_test_element; i++)
				for (int j = 0; j < nDOF_trial_element; j++)
				{
					elementJacobian_u_u.data()[i * nDOF_trial_element + j] = 0.0;
				}
			for (int k = 0; k < nQuadraturePoints_element; k++)
			{
				gf_s.set_quad(k);
				gf_f.set_quad(k);
				int eN_k = eN * nQuadraturePoints_element + k; // index to a scalar at a quadrature point
				int eN_k_3d = eN_k * 3;
				// declare local storage
				double u = 0.0,
					   grad_u[nSpace],
					   ua = 0.0,
					   grad_ua[nSpace],
					   ub = 0.0,
					   grad_ub[nSpace],
					   m = 0.0, dm = 0.0,
					   h_phi = 0.0,
					   r_s = 0.0, dr_s = 0.0,
					   r_s_a = 0.0, dr_s_a = 0.0, ham_s_a = 0.0, r_s_b = 0.0, dr_s_b = 0.0, ham_s_b = 0.0,
					   dham_s_a[nSpace] = {0., 0.}, f_s_a[nSpace] = {0., 0.}, df_s_a[nSpace] = {0., 0.},
					   dham_s_b[nSpace] = {0., 0.}, f_s_b[nSpace] = {0., 0.}, df_s_b[nSpace] = {0., 0.},
					   f[nSpace], df[nSpace],
					   f_s[nSpace] = {0., 0.}, df_s[nSpace] = {0., 0.},
					   ham_s = 0.0, dham_s[nSpace] = {0., 0.},
					   r_f = 0.0, dr_f = 0.0,
					   f_f[nSpace] = {0., 0.}, df_f[nSpace] = {0., 0.},
					   ham_f = 0.0, dham_f[nSpace] = {0., 0.},
					   m_t = 0.0, dm_t = 0.0,
					   dpdeResidual_u_u[nDOF_trial_element],
					   Lstar_u[nDOF_test_element],
					   dsubgridError_u_u[nDOF_trial_element],
					   tau = 0.0, tau0 = 0.0, tau1 = 0.0,
					   *a = NULL,
					   dr = 0.0,
					   jac[nSpace * nSpace],
					   jacDet,
					   jacInv[nSpace * nSpace],
					   u_grad_trial[nDOF_trial_element * nSpace],
					   ua_grad_trial[nDOF_trial_element * nSpace],
					   ub_grad_trial[nDOF_trial_element * nSpace],
					   dV,
					   ua_trial[nDOF_trial_element],
					   ub_trial[nDOF_trial_element],
					   u_test_dV[nDOF_test_element],
					   ua_test_dV[nDOF_test_element],
					   ub_test_dV[nDOF_test_element],
					   u_grad_test_dV[nDOF_test_element * nSpace],
					   ua_grad_test_dV[nDOF_test_element * nSpace],
					   ub_grad_test_dV[nDOF_test_element * nSpace],
					   x, y, z,
					   G[nSpace * nSpace], G_dd_G, tr_G;
				//
				// calculate solution and gradients at quadrature points
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
											x, y, z);
				ck.calculateH_element(eN,
									  k,
									  nodeDiametersArray.data(),
									  mesh_l2g.data(),
									  mesh_trial_ref.data(),
									  h_phi);
				// get the physical integration weight
				dV = fabs(jacDet) * dV_ref.data()[k];
				// get metric tensor and friends
				ck.calculateG(jacInv, G, G_dd_G, tr_G);
				// get the trial function gradients
				// std::cout << "    calculating gradTrialfromRef from calculateElementJacobian() "<< std::endl;
				ck.gradTrialFromRef(&u_grad_trial_ref.data()[k * nDOF_trial_element * nSpace], jacInv, u_grad_trial);
				// get the solution
				ck.valFromElementDOF(element_u.data(), &u_trial_ref.data()[k * nDOF_trial_element], u);
				// get the solution gradients
				ck.gradFromElementDOF(element_u.data(), u_grad_trial, grad_u);
				// precalculate test function products with integration weights
				for (int j = 0; j < nDOF_trial_element; j++)
				{
					u_test_dV[j] = u_test_ref.data()[k * nDOF_trial_element + j] * dV;
					for (int I = 0; I < nSpace; I++)
					{
						u_grad_test_dV[j * nSpace + I] = u_grad_trial[j * nSpace + I] * dV; // cek warning won't work for Petrov-Galerkin
						//PGIFEM
						//ua_grad_test_dV[j * nSpace + I] = u_grad_trial[j * nSpace + I] * dV; // cek warning won't work for Petrov-Galerkin
						//ub_grad_test_dV[j * nSpace + I] = u_grad_trial[j * nSpace + I] * dV; // cek warning won't work for Petrov-Galerkin
					}
				}
				if (icase_f == 0)
				{
					double va[nDOF_trial_element], va_grad_trial[nDOF_trial_element * nSpace], 
						vb[nDOF_trial_element], vb_grad_trial[nDOF_trial_element * nSpace];
					for (int i = 0; i < nDOF_trial_element; i++)
					{
						va[i] = gf_f.VA(i);
						va_grad_trial[i * nSpace + 0] = gf_f.VA_x(i);
						va_grad_trial[i * nSpace + 1] = gf_f.VA_y(i);
						vb[i] = gf_f.VB(i);
						vb_grad_trial[i * nSpace + 0] = gf_f.VB_x(i);
						vb_grad_trial[i * nSpace + 1] = gf_f.VB_y(i);
						// std::cout << " va[" << i << "]=" << va[i] << std::endl;
						// std::cout << " vb[" << i << "]=" << vb[i] << std::endl;
						// std::cout << " va_grad_trial[" << i << "]=" << va_grad_trial[i * nSpace + 0] << "," << va_grad_trial[i * nSpace + 1] << std::endl;
						// std::cout << " vb_grad_trial[" << i << "]=" << vb_grad_trial[i * nSpace + 0] << "," << vb_grad_trial[i * nSpace + 1] << std::endl;
					}
					ck.valFromElementDOF(element_u.data(), va, ua);
					ck.gradFromElementDOF(element_u.data(), va_grad_trial, grad_ua);
					ck.valFromElementDOF(element_u.data(), vb, ub);
					ck.gradFromElementDOF(element_u.data(), vb_grad_trial, grad_ub);
					for (int i = 0; i < nDOF_test_element; i++)
					{
						ua_test_dV[i] = va[i] * dV;
						ub_test_dV[i] = vb[i] * dV;
						for (int I = 0; I < nSpace; I++)
						{
							ua_grad_test_dV[i * nSpace + I] = va_grad_trial[i * nSpace + I] * dV;
							ub_grad_test_dV[i * nSpace + I] = vb_grad_trial[i * nSpace + I] * dV;
						}
					}
					for (int j = 0; j < nDOF_trial_element; j++)
					{
						ua_trial[j] = va[j];
						ub_trial[j] = vb[j];
						for (int I = 0; I < nSpace; I++)
						{
							ua_grad_trial[j * nSpace + I] = va_grad_trial[j * nSpace + I];
							ub_grad_trial[j * nSpace + I] = vb_grad_trial[j * nSpace + I];
						}
						// std::cout << " ua_grad_test_dV[" << j << "]=" << ua_grad_test_dV[j * nSpace + 0] << "," << ua_grad_test_dV[j * nSpace + 1] << std::endl;
						// std::cout << " ub_grad_test_dV[" << j << "]=" << ub_grad_test_dV[j * nSpace + 0] << "," << ub_grad_test_dV[j * nSpace + 1] << std::endl;
					}
					if (PG && pgElem)
					{
						// Same substitution as the residual; trial-side (j-index) arrays
						// are untouched, only the test side (i-index) changes.
						for (int i = 0; i < nDOF_test_element; i++)
						{
							ua_test_dV[i] = u_test_dV[i];
							ub_test_dV[i] = u_test_dV[i];
							for (int I = 0; I < nSpace; I++)
							{
								ua_grad_test_dV[i * nSpace + I] = u_grad_test_dV[i * nSpace + I];
								ub_grad_test_dV[i * nSpace + I] = u_grad_test_dV[i * nSpace + I];
							}
						}
					}
				}
				//
				// calculate pde coefficients and derivatives at quadrature points
				//
				// evaluateCoefficients()
				a = &q_a.data()[eN_k * sd_rowptr.data()[nSpace]];
				// On a demoted element the fluid is entirely one phase, but the pre-evaluated
				// pointwise data still switches branch across the buried Gamma_f -- which lies inside
				// the solid part of this element. H_s is only moment-exact against polynomials, and a
				// coefficient that jumps mid-element is not one, so those quadrature points inject a
				// contribution from a fluid that is not physically present. Substitute the surviving
				// phase's constants, which makes the integrand polynomial again and H_s exact.
				double a_dem[nSpace * nSpace];
				double u_s_dem = embeddedBoundary_u_q.data()[eN_k];
				if (demoted_f != 0)
				{
					for (int I = 0; I < nSpace * nSpace; I++) a_dem[I] = 0.0;
					const double mu_dem = (demoted_f < 0) ? mua : mub;
					for (int I = 0; I < nSpace; I++) a_dem[I * nSpace + I] = mu_dem;
					a = a_dem;
					u_s_dem = (demoted_f < 0) ? embeddedBoundary_u_inner_q.data()[eN_k]
						                          : embeddedBoundary_u_outer_q.data()[eN_k];
				}
				for (int I = 0; I < nSpace; I++)
					df[I] = q_v.data()[eN_k * nSpace + I];
				dr = 0.0;
				const double H_s = gf_s.H(0., 0.);
				const double D_s = gf_s.D(0., 0.);
				// Per-side Dirac for the solid surface. Weighting the single-cut D_s by a region fit
				// (ImH_f * D_s) is the product-of-two-fits error again, and a large one -- about 31% on
				// the segment length. Declared here because the Nitsche terms below consume it.
				// Defaults to D_s, so elements cut by only one level set are untouched.
				double D_s_a = D_s, D_s_b = D_s;
				if (demoted_f != 0)
				{
					// The whole solid surface in this element borders the one surviving fluid region.
					D_s_a = (demoted_f < 0) ? D_s : 0.0;
					D_s_b = (demoted_f < 0) ? 0.0 : D_s;
				}
				else if (comp_DA != nullptr)
				{
					const double xr_d = x_ref.data()[k * 3 + 0], yr_d = x_ref.data()[k * 3 + 1];
					D_s_a = equivalent_polynomials::composite::evaluate<nP_fit>(comp_DA, xr_d, yr_d);
					D_s_b = equivalent_polynomials::composite::evaluate<nP_fit>(comp_DB, xr_d, yr_d);
				}
				if (embeddedBoundary)
				{
					double level_set_normal[nSpace];
					double sign = 0.0;
					double norm_exact = 0.0, norm_cut = 0.0;
					for (int I = 0; I < nSpace; I++)
					{
						sign += embeddedBoundary_normal_q.data()[eN_k_3d + I] * gf_s.get_normal()[I];
						level_set_normal[I] = gf_s.get_normal()[I];
						norm_cut += level_set_normal[I] * level_set_normal[I];
						norm_exact += embeddedBoundary_normal_q.data()[eN_k_3d + I] * embeddedBoundary_normal_q.data()[eN_k_3d + I];
					}
					assert(std::fabs(1.0 - norm_cut) < 1.0e-8);
					assert(std::fabs(1.0 - norm_exact) < 1.0e-8);
					if (sign < 0.0)
						for (int I = 0; I < nSpace; I++)
							level_set_normal[I] *= -1.0;
					if (icase_f == 0)
					{
						// The solution here is ua on the mua side and ub on the mub side; the plain `u` is
						// neither. Handing `u` to the penalty makes (u - u_s) nonzero even when the discrete
						// solution is exactly right, and the O(100/h) penalty amplifies that.
						// The prescribed jump is a constant offset with zero derivative, so it is
						// residual-only and does not appear here.

						if (embeddedBoundary_robin)
						{
							updateEmbeddedBoundaryRobinTerms(embeddedBoundary_robin_h_inner,
								embeddedBoundary_robin_uInfinity_inner_q.data()[eN_k], ua, r_s_a, dr_s_a, D_s_a);
							updateEmbeddedBoundaryRobinTerms(embeddedBoundary_robin_h_outer,
								embeddedBoundary_robin_uInfinity_outer_q.data()[eN_k], ub, r_s_b, dr_s_b, D_s_b);
						}
						else
						{
							updateEmbeddedBoundaryTerms(embeddedBoundary_penalty / h_phi, dV, level_set_normal,
								embeddedBoundary_u_inner_q.data()[eN_k], ua, grad_ua, mua,
								r_s_a, dr_s_a, ham_s_a, dham_s_a, f_s_a, df_s_a, D_s_a);
							updateEmbeddedBoundaryTerms(embeddedBoundary_penalty / h_phi, dV, level_set_normal,
								embeddedBoundary_u_outer_q.data()[eN_k], ub, grad_ub, mub,
								r_s_b, dr_s_b, ham_s_b, dham_s_b, f_s_b, df_s_b, D_s_b);
						}
					}
					else
					{
						if (embeddedBoundary_robin)
						{
							const double h_here = (icase_f < 0) ? embeddedBoundary_robin_h_inner
								                                 : embeddedBoundary_robin_h_outer;
							const double u_inf_here = (icase_f < 0)
								? embeddedBoundary_robin_uInfinity_inner_q.data()[eN_k]
								: embeddedBoundary_robin_uInfinity_outer_q.data()[eN_k];
							updateEmbeddedBoundaryRobinTerms(h_here, u_inf_here, u, r_s, dr_s, D_s);
						}
						else
						{
							updateEmbeddedBoundaryTerms(embeddedBoundary_penalty / h_phi, dV, level_set_normal,
								u_s_dem, u, grad_u, a[0],
								r_s, dr_s, ham_s, dham_s, f_s, df_s, D_s);
						}
					}
				}
				const double ImH_f = gf_f.ImH(0., 0.);
				const double H_f = gf_f.H(0., 0.);
				// Region weights for the bulk terms. On an element cut by both level sets these
				// come from the composite fit; otherwise they are the usual products, unchanged.
				double wA = ImH_f * H_s, wB = H_f * H_s;
				if (demoted_f != 0)
				{
					// Only one fluid region exists here; give it the whole fluid weight and the other none.
					wA = (demoted_f < 0) ? H_s : 0.0;
					wB = (demoted_f < 0) ? 0.0 : H_s;
				}
				else if (comp_A != nullptr)
				{
					const double xr = x_ref.data()[k * 3 + 0], yr = x_ref.data()[k * 3 + 1];
					wA = equivalent_polynomials::composite::evaluate<nP_fit>(comp_A, xr, yr);
					wB = equivalent_polynomials::composite::evaluate<nP_fit>(comp_B, xr, yr);
				}
				double D_f = gf_f.D(0., 0.);
				if (comp_DFa != nullptr)
				{
					// Gamma_f's own Dirac measure, restricted to the part still inside the active
					// fluid domain (see comp_DFa_cache's declaration) -- the plain, single-curve
					// gf_f.D(0.,0.) above double-counts whatever part of Gamma_f's chord falls
					// inside Gamma_s's masked-out solid on a triply-cut element.
					const double xr_f = x_ref.data()[k * 3 + 0], yr_f = x_ref.data()[k * 3 + 1];
					D_f = equivalent_polynomials::composite::evaluate<nP_fit>(comp_DFa, xr_f, yr_f);
				}
				// A demoted element's Gamma_f lies entirely inside the solid: no part of it borders
				// fluid, so the flux-jump load must vanish exactly there (any fit would leave a
				// spurious source).
				if (demoted_f != 0)
					D_f = 0.0;
				if (getenv("ADR_DEBUG_ELEM") && eN == atoi(getenv("ADR_DEBUG_ELEM")))
				{
					fprintf(stderr, "ADR_DEBUG_ELEM eN=%d k=%d xi=%.10f eta=%.10f x=%.10f y=%.10f "
							"D_f=%.10e dV=%.10e D_f_dV=%.10e comp_DFa_null=%d\n",
							eN, k, x_ref.data()[k * 3 + 0], x_ref.data()[k * 3 + 1], x, y,
							D_f, dV, D_f * dV, comp_DFa == nullptr ? 1 : 0);
				}
				if (immersedBoundary)
				{
					double level_set_normal[nSpace];
					double sign = 0.0;
					double norm_exact = 0.0, norm_cut = 0.0;
					for (int I = 0; I < nSpace; I++)
					{
						sign += immersedBoundary_normal_q.data()[eN_k_3d + I] * gf_f.get_normal()[I];
						level_set_normal[I] = gf_f.get_normal()[I];
						norm_cut += level_set_normal[I] * level_set_normal[I];
						norm_exact += immersedBoundary_normal_q.data()[eN_k_3d + I] * immersedBoundary_normal_q.data()[eN_k_3d + I];
					}
					assert(std::fabs(1.0 - norm_cut) < 1.0e-8);
					assert(std::fabs(1.0 - norm_exact) < 1.0e-8);
					if (sign < 0.0)
					  for (int I = 0; I < nSpace; I++)
					    level_set_normal[I] *= -1.0;
					updateImmersedBoundaryTerms(immersedBoundary_penalty / h_phi, // penalty,
												dV,
												level_set_normal,
												x,
												y,
												z,
												immersedBoundary_u_q.data()[eN_k],
												u,
												grad_u,
												a[0], // assume scalar diffusion for now
												r_f,
												dr_f,
												ham_f,
												dham_f,
												f_f,
												df_f,
												immersedBoundary_fluxJump_q.data()[eN_k],
												&immersedBoundary_fluxJumpVector_q.data()[eN_k_3d],
												// Gamma_f's jump physics only applies in the fluid: on an element that
												// sits (locally, at this quadrature point) entirely inside the embedded
												// solid, H_s==0 and D_f alone (gf_f.D, or the raw single-curve fallback
												// when comp_DFa isn't available) does not know that -- it only encodes
												// where Gamma_f itself is, not whether the fluid actually exists there.
												// Without this factor an element wholly inside the solid but crossed by
												// Gamma_f gets a fixed, u-independent Dirac source with nothing in the
												// Jacobian to balance it (the bulk terms are already correctly zero via
												// wA=wB=0), so Newton can never drive that row's residual to zero.
												// On a triply-cut cell comp_DFa is already restricted to the fluid part of
												// Gamma_f; multiplying it by the H_s fit again would restrict twice (a
												// product of two fits), so H_s is applied only to the single-curve D_f.
												(comp_DFa != nullptr) ? D_f : D_f * H_s);
				}
				//
				// calculate subgrid error contribution to the Jacobian (strong residual, adjoint, jacobian of strong residual)
				//
				// calculate the adjoint times the test functions
				for (int i = 0; i < nDOF_test_element; i++)
				{
					// int eN_k_i_nSpace = (eN_k*nDOF_trial_element+i)*nSpace;
					// Lstar_u[i]=ck.Advection_adjoint(df,&u_grad_test_dV.data()[eN_k_i_nSpace]);
					int i_nSpace = i * nSpace;
					Lstar_u[i] = ck.Advection_adjoint(df, &u_grad_test_dV[i_nSpace]);
				}
				// calculate the Jacobian of strong residual
				for (int j = 0; j < nDOF_trial_element; j++)
				{
					// int eN_k_j=eN_k*nDOF_trial_element+j;
					// int eN_k_j_nSpace = eN_k_j*nSpace;
					int j_nSpace = j * nSpace;
					dpdeResidual_u_u[j] = ck.MassJacobian_strong(dm_t, u_trial_ref.data()[k * nDOF_trial_element + j]) +
										  ck.AdvectionJacobian_strong(df, &u_grad_trial[j_nSpace]);
				}
				// tau and tau*Res
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
				tau = useMetrics * tau1 + (1.0 - useMetrics) * tau0;

				for (int j = 0; j < nDOF_trial_element; j++)
					dsubgridError_u_u[j] = -tau * dpdeResidual_u_u[j];

				// Leveque & Li 1994, Examples 1, 3, 4, PWC, PWL, PWQ
				double a_loc[nSpace * nSpace];
				for (int I = 0; I < nSpace * nSpace; I++) a_loc[I] = 0.0;
				for (int i = 0; i < nDOF_test_element; i++)
				{
					int i_nSpace = i * nSpace;
					for (int j = 0; j < nDOF_trial_element; j++)
					{
						int j_nSpace = j * nSpace;
						if (icase_f == 0)
						{
							if(!gf_f.exact.edge && !gf_f.exact.corner)
						    {
								for (int I = 0; I < nSpace; I++) a_loc[I * nSpace + I] = mua;
								elementJacobian_u_u.data()[i * nDOF_trial_element + j] += wA * (ck.AdvectionJacobian_weak(df, ua_trial[j], &ua_grad_test_dV[i_nSpace]) +
								ck.SimpleDiffusionJacobian_weak(sd_rowptr.data(), sd_colind.data(), a_loc, &ua_grad_trial[j_nSpace], &ua_grad_test_dV[i_nSpace]) +
								ck.ReactionJacobian_weak(dr + dr_f, ua_trial[j], ua_test_dV[i]) +
								ck.NumericalDiffusionJacobian(q_numDiff_u_last.data()[eN_k], &ua_grad_trial[j_nSpace], &ua_grad_test_dV[i_nSpace]));

								for (int I = 0; I < nSpace; I++) a_loc[I * nSpace + I] = mub;
								elementJacobian_u_u.data()[i * nDOF_trial_element + j] += wB * (ck.AdvectionJacobian_weak(df, ub_trial[j], &ub_grad_test_dV[i_nSpace]) +
									ck.SimpleDiffusionJacobian_weak(sd_rowptr.data(), sd_colind.data(), a_loc, &ub_grad_trial[j_nSpace], &ub_grad_test_dV[i_nSpace]) +
									ck.ReactionJacobian_weak(dr + dr_f, ub_trial[j], ub_test_dV[i]) +
									ck.NumericalDiffusionJacobian(q_numDiff_u_last.data()[eN_k], &ub_grad_trial[j_nSpace], &ub_grad_test_dV[i_nSpace]));
							}
							else if (gf_f.exact.edge == -1 || gf_f.exact.corner == -1)
							{
								for (int I = 0; I < nSpace; I++) a_loc[I * nSpace + I] = mua;
								elementJacobian_u_u.data()[i * nDOF_trial_element + j] += wA * (ck.AdvectionJacobian_weak(df, ua_trial[j], &ua_grad_test_dV[i_nSpace]) +
								ck.SimpleDiffusionJacobian_weak(sd_rowptr.data(), sd_colind.data(), a_loc, &ua_grad_trial[j_nSpace], &ua_grad_test_dV[i_nSpace]) +
								ck.ReactionJacobian_weak(dr + dr_f, ua_trial[j], ua_test_dV[i]) +
								ck.NumericalDiffusionJacobian(q_numDiff_u_last.data()[eN_k], &ua_grad_trial[j_nSpace], &ua_grad_test_dV[i_nSpace]));
							}
							else if (gf_f.exact.edge == 1 || gf_f.exact.corner == 1)
							{
								for (int I = 0; I < nSpace; I++) a_loc[I * nSpace + I] = mub;
								elementJacobian_u_u.data()[i * nDOF_trial_element + j] += wB * (ck.AdvectionJacobian_weak(df, ub_trial[j], &ub_grad_test_dV[i_nSpace]) +
									ck.SimpleDiffusionJacobian_weak(sd_rowptr.data(), sd_colind.data(), a_loc, &ub_grad_trial[j_nSpace], &ub_grad_test_dV[i_nSpace]) +
									ck.ReactionJacobian_weak(dr + dr_f, ub_trial[j], ub_test_dV[i]) +
									ck.NumericalDiffusionJacobian(q_numDiff_u_last.data()[eN_k], &ub_grad_trial[j_nSpace], &ub_grad_test_dV[i_nSpace]));
							}
							else assert(false && "Invalid gf_f.exact.edge/corner values. Should be -1, 0 or +1.");
						}
						else
						{
							elementJacobian_u_u.data()[i * nDOF_trial_element + j] += H_s * (ck.AdvectionJacobian_weak(df, u_trial_ref.data()[k * nDOF_trial_element + j], &u_grad_test_dV[i_nSpace]) +
																							 ck.SimpleDiffusionJacobian_weak(sd_rowptr.data(), sd_colind.data(), a, &u_grad_trial[j_nSpace], &u_grad_test_dV[i_nSpace]) +
																							 ck.ReactionJacobian_weak(dr, u_trial_ref.data()[k * nDOF_trial_element + j], u_test_dV[i]) +
																							 ck.SubgridErrorJacobian(dsubgridError_u_u[j], Lstar_u[i]) +
																							 ck.NumericalDiffusionJacobian(q_numDiff_u_last.data()[eN_k], &u_grad_trial[j_nSpace], &u_grad_test_dV[i_nSpace]));
						}
						if (embeddedBoundary)
						{
							if (gf_s.exact.edge >=0 && !gf_s.exact.corner)
							{
								if (icase_f == 0)
								{
									elementJacobian_u_u.data()[i * nDOF_trial_element + j] +=
										(ck.AdvectionJacobian_weak(df_s_a, ua_trial[j], &ua_grad_test_dV[i_nSpace])
										 + ck.ReactionJacobian_weak(dr_s_a, ua_trial[j], ua_test_dV[i])
										 + ck.HamiltonianJacobian_weak(dham_s_a, &ua_grad_trial[j_nSpace], ua_test_dV[i])) +
										(ck.AdvectionJacobian_weak(df_s_b, ub_trial[j], &ub_grad_test_dV[i_nSpace])
										 + ck.ReactionJacobian_weak(dr_s_b, ub_trial[j], ub_test_dV[i])
										 + ck.HamiltonianJacobian_weak(dham_s_b, &ub_grad_trial[j_nSpace], ub_test_dV[i]));
								}
								else
								{
									elementJacobian_u_u.data()[i * nDOF_trial_element + j] += (ck.AdvectionJacobian_weak(df_s, u_trial_ref.data()[k * nDOF_trial_element + j], &u_grad_test_dV[i_nSpace])
																					+ ck.ReactionJacobian_weak(dr_s, u_trial_ref.data()[k * nDOF_trial_element + j], u_test_dV[i])
																					+ ck.HamiltonianJacobian_weak(dham_s, &u_grad_trial[j_nSpace], u_test_dV[i]));
								}
							}
						}
						// immersedBoundary's dr_f is folded into the wA/wB branch reaction terms
						// above (ck.ReactionJacobian_weak(dr + dr_f, ...)), matching the residual
						// restructuring -- see the comment there. dr_f/df_f/dham_f are always zero
						// (updateImmersedBoundaryTerms), so this was a no-op numerically either way,
						// but kept the wrong (unenriched) test function on the books.
						// std::cout << "   elementJacobian_u_u[" << i << "," << j << "] = " << elementJacobian_u_u.data()[i * nDOF_trial_element + j];
					} // j
					// std::cout << std::endl;
				} // i
				// std::cout << std::endl;
			} // k
		}

		void calculateJacobian(arguments_dict &args)
		{
			xt::pyarray<double> &mesh_trial_ref = args.array<double>("mesh_trial_ref");
			xt::pyarray<double> &mesh_grad_trial_ref = args.array<double>("mesh_grad_trial_ref");
			xt::pyarray<double> &mesh_dof = args.array<double>("mesh_dof");
			xt::pyarray<int> &mesh_l2g = args.array<int>("mesh_l2g");
			xt::pyarray<double> &dV_ref = args.array<double>("dV_ref");
			xt::pyarray<double> &u_trial_ref = args.array<double>("u_trial_ref");
			xt::pyarray<double> &u_grad_trial_ref = args.array<double>("u_grad_trial_ref");
			xt::pyarray<double> &u_test_ref = args.array<double>("u_test_ref");
			xt::pyarray<double> &u_grad_test_ref = args.array<double>("u_grad_test_ref");
			xt::pyarray<double> &elementDiameter = args.array<double>("elementDiameter");
			xt::pyarray<double> &cfl = args.array<double>("cfl");
			double Ct_sge = args.scalar<double>("Ct_sge");
			double sc_uref = args.scalar<double>("sc_uref");
			double sc_alpha = args.scalar<double>("sc_alpha");
			double useMetrics = args.scalar<double>("useMetrics");
			xt::pyarray<double> &mesh_trial_trace_ref = args.array<double>("mesh_trial_trace_ref");
			xt::pyarray<double> &mesh_grad_trial_trace_ref = args.array<double>("mesh_grad_trial_trace_ref");
			xt::pyarray<double> &dS_ref = args.array<double>("dS_ref");
			xt::pyarray<double> &u_trial_trace_ref = args.array<double>("u_trial_trace_ref");
			xt::pyarray<double> &u_grad_trial_trace_ref = args.array<double>("u_grad_trial_trace_ref");
			xt::pyarray<double> &u_test_trace_ref = args.array<double>("u_test_trace_ref");
			xt::pyarray<double> &u_grad_test_trace_ref = args.array<double>("u_grad_test_trace_ref");
			xt::pyarray<double> &normal_ref = args.array<double>("normal_ref");
			xt::pyarray<double> &boundaryJac_ref = args.array<double>("boundaryJac_ref");
			int nElements_global = args.scalar<int>("nElements_global");
			xt::pyarray<int> &u_l2g = args.array<int>("u_l2g");
			xt::pyarray<double> &u_dof = args.array<double>("u_dof");
			xt::pyarray<int> &sd_rowptr = args.array<int>("sd_rowptr");
			xt::pyarray<int> &sd_colind = args.array<int>("sd_colind");
			xt::pyarray<double> &q_a = args.array<double>("q_a");
			xt::pyarray<double> &q_v = args.array<double>("q_v");
			xt::pyarray<double> &q_r = args.array<double>("q_r");
			int lag_shockCapturing = args.scalar<int>("lag_shockCapturing");
			double shockCapturingDiffusion = args.scalar<double>("shockCapturingDiffusion");
			xt::pyarray<double> &q_numDiff_u = args.array<double>("q_numDiff_u");
			xt::pyarray<double> &q_numDiff_u_last = args.array<double>("q_numDiff_u_last");
			xt::pyarray<int> &csrRowIndeces_u_u = args.array<int>("csrRowIndeces_u_u");
			xt::pyarray<int> &csrColumnOffsets_u_u = args.array<int>("csrColumnOffsets_u_u");
			xt::pyarray<double> &globalJacobian = args.array<double>("globalJacobian");
			int nExteriorElementBoundaries_global = args.scalar<int>("nExteriorElementBoundaries_global");
			xt::pyarray<int> &exteriorElementBoundariesArray = args.array<int>("exteriorElementBoundariesArray");
			xt::pyarray<int> &elementBoundaryElementsArray = args.array<int>("elementBoundaryElementsArray");
			xt::pyarray<int> &elementBoundaryLocalElementBoundariesArray = args.array<int>("elementBoundaryLocalElementBoundariesArray");
			xt::pyarray<double> &ebqe_a = args.array<double>("ebqe_a");
			xt::pyarray<double> &ebqe_v = args.array<double>("ebqe_v");
			xt::pyarray<int> &isDOFBoundary_u = args.array<int>("isDOFBoundary_u");
			xt::pyarray<double> &ebqe_bc_u_ext = args.array<double>("ebqe_bc_u_ext");
			xt::pyarray<int> &isDiffusiveFluxBoundary_u = args.array<int>("isDiffusiveFluxBoundary_u");
			xt::pyarray<int> &isAdvectiveFluxBoundary_u = args.array<int>("isAdvectiveFluxBoundary_u");
			xt::pyarray<double> &ebqe_bc_flux_u_ext = args.array<double>("ebqe_bc_flux_u_ext");
			xt::pyarray<double> &ebqe_bc_advectiveFlux_u_ext = args.array<double>("ebqe_bc_advectiveFlux_u_ext");
			xt::pyarray<int> &csrColumnOffsets_eb_u_u = args.array<int>("csrColumnOffsets_eb_u_u");
			xt::pyarray<double> &ebqe_penalty_ext = args.array<double>("ebqe_penalty_ext");
			const bool embeddedBoundary = args.scalar<int>("embeddedBoundary");
			const double embeddedBoundary_penalty = args.scalar<double>("embeddedBoundary_penalty");
			const double embeddedBoundary_ghost_penalty = args.scalar<double>("embeddedBoundary_ghost_penalty");
			xt::pyarray<double> &embeddedBoundary_sdf_nodes = args.array<double>("embeddedBoundary_sdf_nodes");
			xt::pyarray<double> &embeddedBoundary_sdf_q = args.array<double>("embeddedBoundary_sdf_q");
			xt::pyarray<double> &embeddedBoundary_normal_q = args.array<double>("embeddedBoundary_normal_q");
			xt::pyarray<double> &embeddedBoundary_u_q = args.array<double>("embeddedBoundary_u_q");
			xt::pyarray<double> &embeddedBoundary_u_inner_q = args.array<double>("embeddedBoundary_u_inner_q");
			xt::pyarray<double> &embeddedBoundary_u_outer_q = args.array<double>("embeddedBoundary_u_outer_q");
			const bool embeddedBoundary_robin = args.scalar<int>("embeddedBoundary_robin");
			const double embeddedBoundary_robin_h_inner = args.scalar<double>("embeddedBoundary_robin_h_inner");
			const double embeddedBoundary_robin_h_outer = args.scalar<double>("embeddedBoundary_robin_h_outer");
			xt::pyarray<double> &embeddedBoundary_robin_uInfinity_inner_q = args.array<double>("embeddedBoundary_robin_uInfinity_inner_q");
			xt::pyarray<double> &embeddedBoundary_robin_uInfinity_outer_q = args.array<double>("embeddedBoundary_robin_uInfinity_outer_q");
			const bool immersedBoundary = args.scalar<int>("immersedBoundary");
			const double immersedBoundary_penalty = args.scalar<double>("immersedBoundary_penalty");
			const double immersedSCIFEM_switch = args.scalar<double>("immersedSCIFEM_switch");
			const double immersedSCIFEM_penalty = args.scalar<double>("immersedSCIFEM_penalty");
			const bool PG = args.scalar<int>("PG");
			xt::pyarray<double> &immersedBoundary_sdf_nodes = args.array<double>("immersedBoundary_sdf_nodes");
			xt::pyarray<double> &immersedBoundary_sdf_q = args.array<double>("immersedBoundary_sdf_q");
			xt::pyarray<double> &immersedBoundary_normal_q = args.array<double>("immersedBoundary_normal_q");
			xt::pyarray<double> &immersedBoundary_u_q = args.array<double>("immersedBoundary_u_q");
			xt::pyarray<double> &immersedBoundary_fluxJump_q = args.array<double>("immersedBoundary_fluxJump_q");
			xt::pyarray<double> &immersedBoundary_fluxJumpVector_q = args.array<double>("immersedBoundary_fluxJumpVector_q");
			xt::pyarray<double> &immersedBoundary_fluxJump_element = args.array<double>("immersedBoundary_fluxJump_element");
			xt::pyarray<double> &immersedBoundary_solutionJump_nodes = args.array<double>("immersedBoundary_solutionJump_nodes");
			xt::pyarray<double> &isActiveDOF = args.array<double>("isActiveDOF");
			const double eb_adjoint_sigma = args.scalar<double>("eb_adjoint_sigma");
			xt::pyarray<double> &x_ref = args.array<double>("x_ref");
			xt::pyarray<double> &xB_ref = args.array<double>("xB_ref");
			xt::pyarray<int> &elementBoundariesArray = args.array<int>("elementBoundariesArray");
			const int nElementBoundaries_owned = args.scalar<int>("nElementBoundaries_owned");
			xt::pyarray<double> &elementBoundaryDiameter = args.array<double>("elementBoundaryDiameter");
			xt::pyarray<double> &nodeDiametersArray = args.array<double>("nodeDiametersArray");
			const double mua = args.scalar<double>("mua");
			const double mub = args.scalar<double>("mub");
			const double jf = args.scalar<double>("jf");
			const bool recomputeIFEMGeometry = args.scalar<int>("recomputeIFEMGeometry");
			ensureIFEMCacheSized(nElements_global); // also (re)asserts useExact = true on (re)allocation
			if (recomputeIFEMGeometry)
				ifemGeometryGeneration++;
			//
			// loop over elements to compute volume integrals and load them into the element Jacobians and global Jacobian
			//
			const int tcDeadModeV = tcDeadMode();
			const bool tcFluidMode = std::getenv("ADR_TC_EDGE_FLUID") != nullptr;
			// ADR_TC_EDGE_BOUNDARY_ONLY=1 (use with SCIFEM on): keep the one-sided fluid-side correction only
			// on root-adjacent edges that lie on the exterior boundary. Interior ones are then handled by
			// SCIFEM's two-sided consistency term, which has no counterpart on an exterior edge (no neighbour).
			const bool tcBoundaryOnly = std::getenv("ADR_TC_EDGE_BOUNDARY_ONLY") != nullptr;
			const double tcSgn = tcFluidMode ? -1.0 : 1.0;
			// ADR_EXT_NITSCHE=1 (needs ADR_TC_EDGE_BOUNDARY_ONLY): on the exterior Gamma_f-cut edges add to the
			// one-sided consistency term the Nitsche adjoint term  -int mu grad(v).n (u+j_h-g)  and, with
			// ADR_EXT_NITSCHE_PEN=gamma, the penalty  gamma*mu/h int (u+j_h-g) v,  on the fluid pieces of the edge.
			// g is the Dirichlet data of each phase, ext_edge_u_inner/outer, at 4 Gauss points per exterior edge.
			const bool extNitsche = tcBoundaryOnly && std::getenv("ADR_EXT_NITSCHE") != nullptr;
			// ADR_GP_MU=1: weight the ghost penalty of each fluid phase by its coefficient (mu_a, mu_b),
			// like the Nitsche and interior-penalty terms; the combined fallback uses the mean.
			const bool gpMu = std::getenv("ADR_GP_MU") != nullptr;
			const double gpMuA = gpMu ? mua : 1.0, gpMuB = gpMu ? mub : 1.0, gpMuAB = gpMu ? 0.5 * (mua + mub) : 1.0;
			const double extNitschePen = std::getenv("ADR_EXT_NITSCHE_PEN") ? std::atof(std::getenv("ADR_EXT_NITSCHE_PEN")) : 0.0;
			xt::pyarray<double> &ext_edge_u_inner = args.array<double>("ext_edge_u_inner");
			xt::pyarray<double> &ext_edge_u_outer = args.array<double>("ext_edge_u_outer");
			std::vector<int> ebN2ebNE;
			if (extNitsche)
			{
				ebN2ebNE.assign(elementBoundaryElementsArray.size() / 2, -1);
				for (int ie = 0; ie < nExteriorElementBoundaries_global; ie++)
					ebN2ebNE[exteriorElementBoundariesArray.data()[ie]] = ie;
			}
			const double extT[4] = {0.5 - 0.5 * 0.8611363115940526, 0.5 - 0.5 * 0.3399810435848563,
									0.5 + 0.5 * 0.3399810435848563, 0.5 + 0.5 * 0.8611363115940526};
			auto extInterp = [&](const double *gk, double t) {
				double r = 0.0;
				for (int i = 0; i < 4; i++)
				{
					double w = 1.0;
					for (int j = 0; j < 4; j++)
						if (j != i) w *= (t - extT[j]) / (extT[i] - extT[j]);
					r += gk[i] * w;
				}
				return r; };
			const double extG3[3] = {0.5 - 0.5 * std::sqrt(0.6), 0.5, 0.5 + 0.5 * std::sqrt(0.6)};
			const double extW3[3] = {5.0 / 18.0, 8.0 / 18.0, 5.0 / 18.0};
			std::vector<char> tcNodeFluid;
			if (tcDeadModeV == 2 && embeddedBoundary)
				tcBuildNodeFluid(tcNodeFluid, nElements_global, u_l2g, embeddedBoundary_sdf_nodes, u_dof.size());
			// std::cout << "We are computing the Jacobian...\n" << std::endl;
			// Per-face reference-ELEMENT quadrature points, taken from proteus' own trace tables.
			// Simplex::calculate(..., isBoundary=true) reads its xi_r argument as reference *element*
			// coordinates (it forms x = node0 + Jac_0*xi_r), but xB_ref holds reference *boundary*
			// points (t,0,0) -- passing those directly puts every face's points on the local
			// node0->node1 edge, i.e. correct only for local face 2. mesh_trial_trace_ref holds the P1
			// mesh shape functions at (face, quadrature point) and for a triangle those barycentric
			// functions ARE the reference coordinates (phi_0=1-xi-eta, phi_1=xi, phi_2=eta), so this
			// is exactly consistent with calculateMapping_elementBoundary's own convention.
			double xB_ref_faces[nDOF_mesh_trial_element * nQuadraturePoints_elementBoundary * 3];
			for (int f = 0; f < nDOF_mesh_trial_element; f++)
				for (int kb = 0; kb < nQuadraturePoints_elementBoundary; kb++)
				{
					const int fkb = f * nQuadraturePoints_elementBoundary + kb;
					const double *phi_m = &mesh_trial_trace_ref.data()[fkb * nDOF_mesh_trial_element];
					xB_ref_faces[fkb * 3 + 0] = phi_m[1];
					xB_ref_faces[fkb * 3 + 1] = phi_m[2];
					xB_ref_faces[fkb * 3 + 2] = 0.0;
				}
			for (int eN = 0; eN < nElements_global; eN++)
			{
				// std::cout << "########################\n element: " << eN << " \n########################" << std::endl;
				// double  elementJacobian_u_u.data()[nDOF_test_element*nDOF_trial_element],element_u[nDOF_trial_element];
				auto elementJacobian_u_u = xt::pyarray<double>::from_shape({nDOF_test_element * nDOF_trial_element});
				auto element_u = xt::pyarray<double>::from_shape({nDOF_trial_element});
				for (int j = 0; j < nDOF_trial_element; j++)
				{
					int eN_j = eN * nDOF_trial_element + j;
					element_u.data()[j] = u_dof.data()[u_l2g.data()[eN_j]];
				}
				double element_phi_s[nDOF_trial_element];
				for (int j = 0; j < nDOF_trial_element; j++)
				{
					int eN_j = eN * nDOF_trial_element + j;
					element_phi_s[j] = embeddedBoundary_sdf_nodes.data()[u_l2g.data()[eN_j]];
				}
				double element_phi_f[nDOF_trial_element];
				for (int j = 0; j < nDOF_trial_element; j++)
				{
					int eN_j = eN * nDOF_trial_element + j;
					element_phi_f[j] = immersedBoundary_sdf_nodes.data()[u_l2g.data()[eN_j]];
				}
				double element_nodes[nDOF_trial_element * 3];
				for (int i = 0; i < nDOF_trial_element; i++)
				{
					int eN_i = eN * nDOF_trial_element + i;
					for (int I = 0; I < 3; I++)
						// element_nodes[i * 3 + I] = mesh_dof.data()[mesh_l2g.data()[eN_i] * 3 + I];
						element_nodes[i * 3 + I] = mesh_dof.data()[u_l2g.data()[eN_i] * 3 + I];
					// std::cout << "element_nodes[" << i << "] = (" << element_nodes[i * 3 + 0] << ", " << element_nodes[i * 3 + 1] << ", " << element_nodes[i * 3 + 2] << ")\n";
				} // i
				if (gf_s_interior_gen[eN] != ifemGeometryGeneration)
				{
					gf_s_interior_icase[eN] = gf_s_cache[eN].calculate(element_phi_s, element_nodes, x_ref.data(), false);
					gf_s_interior_gen[eN] = ifemGeometryGeneration;
				}
				int icase_s = gf_s_interior_icase[eN];
				const bool pgElem = !getenv("ADR_PG_TC_ONLY") || (icase_s == 0 && gf_f_interior_icase[eN] == 0);
				if (gf_f_interior_gen[eN] != ifemGeometryGeneration)
				{
					gf_f_interior_icase[eN] = gf_f_cache[eN].calculate(element_phi_f, element_nodes, x_ref.data(), mua, mub, jf, false, false);
					gf_f_interior_gen[eN] = ifemGeometryGeneration;
				}
				int icase_f = gf_f_interior_icase[eN];
				// Records that the buried-Gamma_f logic below demoted this element, and to which
				// phase: -1 = fluid is all Omega^- (mua), +1 = all Omega^+ (mub), 0 = not demoted.
				int demoted_f = 0;
				// Both level sets genuinely cut this element: the product weight ImH_f*H_s is not
				// moment-exact here, so fit one polynomial per intersected region instead. Any
				// element cut by at most one level set is left entirely alone.
				comp_valid[eN] = 0;
				comp_DF_valid[eN] = 0;
				if (icase_s == 0 && icase_f == 0 && nSpace == 2)
				{
					const double phi_s3[3] = {element_phi_s[0], element_phi_s[1], element_phi_s[2]};
					const double phi_f3[3] = {element_phi_f[0], element_phi_f[1], element_phi_f[2]};
					// The fluid interface can cross this element and still lie entirely inside the
					// solid, leaving the fluid undivided. There is then no fluid-fluid interface to
					// resolve *in the fluid*, so the two-sided IFEM reconstruction has nothing to
					// do -- and building it anyway means enforcing interface conditions along a
					// segment where no fluid exists. Demote the element to ordinary two-phase by
					// reporting the fluid's own side, which routes every downstream branch (bulk
					// terms, the solid's Nitsche condition, the error norm) to the standard basis.
					double areaA = 0.0, areaB = 0.0;
					equivalent_polynomials::composite::fluid_region_areas(phi_s3, phi_f3, areaA, areaB);
					// NOTE: we deliberately do NOT change icase_f here. Switching this element to the
					// standard basis would make it disagree with its neighbours about what their shared
					// DOFs mean: the neighbour (genuinely cut in the fluid) needs the pointwise nodal
					// value of u, while this element would need the surviving branch continued to all of
					// its nodes -- including nodes on the other side of Gamma_f. One DOF cannot be both,
					// so the exact solution would drop out of the discrete space entirely. Keeping the
					// IFEM basis costs nothing (u_ex satisfies the interface conditions globally, so
					// imposing them along a buried stretch of Gamma_f is harmless) and keeps the space
					// conforming. What we DO fix is the weights: the empty region must get exactly zero
					// rather than whatever a fit over an empty region would produce.
					//
					// Tolerance, not exact equality: a genuine tangency between Gamma_f and Gamma_s
					// (their zero-sets crossing exactly on a shared mesh edge) makes one region's true
					// area a single point, but floating-point roundoff in the clip almost never lands
					// on bit-exact 0.0 -- it lands a few ULPs away (observed: ~1e-33 against a
					// reference-triangle scale of 0.5). An exact-equality check misses that case
					// entirely and falls through to fitting a degree-4 polynomial to a numerically
					// degenerate sliver, which is unstable (observed: off by 64x on its own area).
					// areaTol is far above that roundoff floor and far below any physically meaningful
					// sub-area, so it can't misclassify a genuinely small-but-real fluid region.
					const double areaTol = 1.0e-9;
					if (areaB <= areaTol && areaA > areaTol)
						demoted_f = -1;   // fluid is all Omega^- (mua side)
					else if (areaA <= areaTol && areaB > areaTol)
						demoted_f = 1;    // fluid is all Omega^+ (mub side)
					else
					{
						const double nodes2[6] = {element_nodes[0], element_nodes[1],
												  element_nodes[3], element_nodes[4],
												  element_nodes[6], element_nodes[7]};
						comp_valid[eN] = (equivalent_polynomials::composite::weights<nP_fit>(
							phi_s3, phi_f3,
							&comp_A_cache[(std::size_t)eN * nCompMon],
							&comp_B_cache[(std::size_t)eN * nCompMon]) &&
						                  equivalent_polynomials::composite::dirac_weights<nP_fit>(
							phi_s3, phi_f3, nodes2,
							&comp_DA_cache[(std::size_t)eN * nCompMon],
							&comp_DB_cache[(std::size_t)eN * nCompMon])) ? 1 : 0;
						comp_DF_valid[eN] = equivalent_polynomials::composite::edge_weights<nP_fit>(
							phi_f3, phi_s3, nodes2,
							&comp_DFm_cache[(std::size_t)eN * nCompMon],
							&comp_DFa_cache[(std::size_t)eN * nCompMon]) ? 1 : 0;
					}
				}
				calculateElementJacobian(icase_f,
										 demoted_f,
										 mesh_trial_ref,
										 mesh_grad_trial_ref,
										 mesh_dof,
										 mesh_l2g,
										 x_ref,
										 dV_ref,
										 u_trial_ref,
										 u_grad_trial_ref,
										 u_test_ref,
										 u_grad_test_ref,
										 elementDiameter,
										 elementBoundaryDiameter,
										 nodeDiametersArray,
										 cfl,
										 Ct_sge,
										 sc_uref,
										 sc_alpha,
										 useMetrics,
										 mesh_trial_trace_ref,
										 mesh_grad_trial_trace_ref,
										 dS_ref,
										 u_trial_trace_ref,
										 u_grad_trial_trace_ref,
										 u_test_trace_ref,
										 u_grad_test_trace_ref,
										 normal_ref,
										 boundaryJac_ref,
										 nElements_global,
										 nElementBoundaries_owned,
										 u_l2g,
										 u_dof,
										 sd_rowptr,
										 sd_colind,
										 q_a,
										 comp_valid[eN] ? &comp_A_cache[(std::size_t)eN * nCompMon] : nullptr,
										 comp_valid[eN] ? &comp_B_cache[(std::size_t)eN * nCompMon] : nullptr,
										 comp_valid[eN] ? &comp_DA_cache[(std::size_t)eN * nCompMon] : nullptr,
										 comp_valid[eN] ? &comp_DB_cache[(std::size_t)eN * nCompMon] : nullptr,
										 comp_DF_valid[eN] ? &comp_DFa_cache[(std::size_t)eN * nCompMon] : nullptr,
										 q_v,
										 q_r,
										 lag_shockCapturing,
										 shockCapturingDiffusion,
										 q_numDiff_u,
										 q_numDiff_u_last,
										 elementJacobian_u_u,
										 element_u,
										 eN,
										 embeddedBoundary,
										 embeddedBoundary_penalty,
										 embeddedBoundary_normal_q,
										 embeddedBoundary_u_q,
										 embeddedBoundary_u_inner_q,
										 embeddedBoundary_u_outer_q,
										 embeddedBoundary_robin,
										 embeddedBoundary_robin_h_inner,
										 embeddedBoundary_robin_h_outer,
										 embeddedBoundary_robin_uInfinity_inner_q,
										 embeddedBoundary_robin_uInfinity_outer_q,
										 immersedBoundary,
										 immersedBoundary_penalty,
										 immersedBoundary_sdf_q,
										 immersedBoundary_normal_q,
										 immersedBoundary_u_q,
										 immersedBoundary_fluxJump_q,
										 immersedBoundary_fluxJumpVector_q,
										 immersedBoundary_solutionJump_nodes,
										 element_phi_f,
												 mua,
											 mub,
											 PG,
											 pgElem);
				//
				// load into element Jacobian into global Jacobian
				//
				for (int i = 0; i < nDOF_test_element; i++)
				{
					int eN_i = eN * nDOF_test_element + i;
					for (int j = 0; j < nDOF_trial_element; j++)
					{
						int eN_i_j = eN_i * nDOF_trial_element + j;
						globalJacobian.data()[csrRowIndeces_u_u.data()[eN_i] + csrColumnOffsets_u_u.data()[eN_i_j]] += elementJacobian_u_u.data()[i * nDOF_trial_element + j];
						// std::cout << "globalJacobian[" << eN_i << "," << eN * nDOF_trial_element + j << "] += " << elementJacobian_u_u.data()[i * nDOF_trial_element + j] << std::endl;
					} // j
				} // i
				// Jacobian of the correction above: exact 3x3 block, linear in u_dof through
				// the flux only. Must match the residual side's ADR_DISABLE_TC_EDGE gate,
				// including the icase_s != -1 check (see the residual side for why).
				if (embeddedBoundary && immersedBoundary && (nDOF_trial_element == 3 || nDOF_trial_element == 6) && icase_f == 0
					&& (icase_s != -1 || tcDeadOK(tcDeadModeV, tcNodeFluid, u_l2g, eN))
					&& std::getenv("ADR_DISABLE_TC_EDGE") == nullptr)
				{
					// Same faces, quadrature and piece indicators as the residual side (P1 and P2).
					const int nD = nDOF_trial_element;
					bool neg[3] = {element_phi_f[0] < 0.0, element_phi_f[1] < 0.0, element_phi_f[2] < 0.0};
					int cnt_neg = (neg[0] ? 1 : 0) + (neg[1] ? 1 : 0) + (neg[2] ? 1 : 0);
					int root_i = -1;
					for (int ii = 0; ii < 3; ii++)
						if ((neg[ii] && cnt_neg == 1) || (!neg[ii] && cnt_neg == 2)) { root_i = ii; break; }
					if (root_i >= 0)
					{
						if (gf_f_boundary_gen[eN] != ifemGeometryGeneration)
						{
							gf_f_boundary_icase[eN] = gf_f_cache[eN].calculate(element_phi_f, element_nodes, xB_ref_faces, mua, mub, jf, true, false);
							gf_f_boundary_gen[eN] = ifemGeometryGeneration;
						}
						GfType &gf_f_tc = gf_f_cache[eN];
						const double sgn_s = tcFluidMode ? 1.0 : -1.0;
						for (int pj = 0; pj < 3; pj++)
						{
							if (pj == root_i)
								continue;
							const int ebN_tc = elementBoundariesArray.data()[eN * nDOF_mesh_trial_element + pj];
							if (tcBoundaryOnly && elementBoundaryElementsArray.data()[ebN_tc * 2 + 1] != -1)
								continue; // interior face: left to SCIFEM
							const int n0 = (pj + 1) % 3, n1 = (pj + 2) % 3;
							double C_a[nP_edge + 1], C_b[nP_edge + 1];
							equivalent_polynomials::calculate_edge_interval_H<nP_edge>(-element_phi_f[n0], -element_phi_f[n1],
																					   sgn_s * element_phi_s[n0], sgn_s * element_phi_s[n1], C_a);
							equivalent_polynomials::calculate_edge_interval_H<nP_edge>(element_phi_f[n0], element_phi_f[n1],
																					   sgn_s * element_phi_s[n0], sgn_s * element_phi_s[n1], C_b);
							const double ex0 = element_nodes[n0 * 3 + 0], ey0 = element_nodes[n0 * 3 + 1];
							const double evx = element_nodes[n1 * 3 + 0] - ex0, evy = element_nodes[n1 * 3 + 1] - ey0;
							const double elen2 = evx * evx + evy * evy;
							const bool extEdge = extNitsche && ebN2ebNE[ebN_tc] >= 0;
							const double h_e = std::sqrt(elen2);
							for (int kb = 0; kb < nQuadraturePoints_elementBoundary; kb++)
							{
								const int ebN_local_kb = pj * nQuadraturePoints_elementBoundary + kb;
								double jac_b[nSpace * nSpace], jacDet_b, jacInv_b[nSpace * nSpace],
									bJac_b[nSpace * (nSpace - 1)], mT_b[(nSpace - 1) * (nSpace - 1)], mTDS_b, n_b[2], xb, yb, zb;
								ck.calculateMapping_elementBoundary(eN, pj, kb, ebN_local_kb,
																	mesh_dof.data(), mesh_l2g.data(),
																	mesh_trial_trace_ref.data(), mesh_grad_trial_trace_ref.data(),
																	boundaryJac_ref.data(), jac_b, jacDet_b, jacInv_b,
																	bJac_b, mT_b, mTDS_b, normal_ref.data(), n_b, xb, yb, zb);
								const double dS = mTDS_b * dS_ref.data()[kb];
								const double t = ((xb - ex0) * evx + (yb - ey0) * evy) / elen2;
								const double Wa = equivalent_polynomials::evaluate_edge_poly<nP_edge>(C_a, t);
								const double Wb = equivalent_polynomials::evaluate_edge_poly<nP_edge>(C_b, t);
								gf_f_tc.set_boundary_quad(ebN_local_kb);
								for (int jj = 0; jj < nD; jj++)
								{
									const int eN_jj = eN * nDOF_test_element + jj;
									const double vA = gf_f_tc.VA(jj), vB = gf_f_tc.VB(jj);
									const double dvnA = mua * (gf_f_tc.VA_x(jj) * n_b[0] + gf_f_tc.VA_y(jj) * n_b[1]);
									const double dvnB = mub * (gf_f_tc.VB_x(jj) * n_b[0] + gf_f_tc.VB_y(jj) * n_b[1]);
									for (int kk = 0; kk < nD; kk++)
									{
										// JA/JB need no Jacobian term: fixed data, not a function of u_dof. Does not
										// account for the masked-node override (calculateJacobian has no
										// triplyCut_maskedLocalNode); unused with ADR_DISABLE_TC=1.
										const double uA_k = gf_f_tc.VA(kk), uB_k = gf_f_tc.VB(kk);
										const double fA_k = mua * (gf_f_tc.VA_x(kk) * n_b[0] + gf_f_tc.VA_y(kk) * n_b[1]);
										const double fB_k = mub * (gf_f_tc.VB_x(kk) * n_b[0] + gf_f_tc.VB_y(kk) * n_b[1]);
										double d = tcSgn * (Wa * fA_k * vA + Wb * fB_k * vB) * dS;
										if (extEdge)
											d += (Wa * (-dvnA * uA_k + extNitschePen * mua / h_e * uA_k * vA)
												  + Wb * (-dvnB * uB_k + extNitschePen * mub / h_e * uB_k * vB)) * dS;
										const int eN_jj_kk = eN_jj * nDOF_trial_element + kk;
										globalJacobian.data()[csrRowIndeces_u_u.data()[eN_jj] + csrColumnOffsets_u_u.data()[eN_jj_kk]] += d;
									}
								}
							}
						}
					}
				}
			} // elements
			for (std::set<int>::iterator it = cutfem_boundaries.begin(); it != cutfem_boundaries.end(); ++it)
			{
				std::map<int, double> Dw_Dn_jump;
				std::map<int, double> Dw_Dn_jump_a, Dw_Dn_jump_b;
				std::map<std::pair<int, int>, int> u_u_nz;
				double gamma_cutfem = embeddedBoundary_ghost_penalty, h_cutfem = elementBoundaryDiameter.data()[*it];
				// The two elements need not parametrize the shared face in the same direction, so
				// index kb can be DIFFERENT physical points on the two sides (see the SCIFEM loop's
				// orientation map). Harmless for P1, where the normal derivative is constant on a
				// face; for P2 it turns the variation along the face into a spurious jump. Match
				// side 1 to side 0 by position.
				int gp_kmap[nQuadraturePoints_elementBoundary];
				{
					double gp_xq[2][nQuadraturePoints_elementBoundary][3] = {}; // 2D mapping never writes z
					for (int s = 0; s < 2; s++)
					{
						const int eN_g = elementBoundaryElementsArray.data()[(*it) * 2 + s],
								  ebN_local_g = elementBoundaryLocalElementBoundariesArray.data()[(*it) * 2 + s];
						for (int kb = 0; kb < nQuadraturePoints_elementBoundary; kb++)
						{
							double jac_g[nSpace * nSpace], jacDet_g, jacInv_g[nSpace * nSpace],
								bJac_g[nSpace * (nSpace - 1)], mT_g[(nSpace - 1) * (nSpace - 1)], mTDS_g, nrm_g[2];
							ck.calculateMapping_elementBoundary(eN_g, ebN_local_g, kb,
																ebN_local_g * nQuadraturePoints_elementBoundary + kb,
																mesh_dof.data(), mesh_l2g.data(),
																mesh_trial_trace_ref.data(), mesh_grad_trial_trace_ref.data(),
																boundaryJac_ref.data(), jac_g, jacDet_g, jacInv_g,
																bJac_g, mT_g, mTDS_g, normal_ref.data(), nrm_g,
																gp_xq[s][kb][0], gp_xq[s][kb][1], gp_xq[s][kb][2]);
						}
					}
					for (int k = 0; k < nQuadraturePoints_elementBoundary; k++)
					{
						int best = 0;
						double bestd = 1.0e300;
						for (int j = 0; j < nQuadraturePoints_elementBoundary; j++)
						{
							double d = 0.0;
							for (int I = 0; I < nSpace; I++)
								d += std::fabs(gp_xq[1][j][I] - gp_xq[0][k][I]);
							if (d < bestd) { bestd = d; best = j; }
						}
						gp_kmap[k] = best;
					}
				}
				int eN_nDOF_trial_element = elementBoundaryElementsArray.data()[(*it) * 2 + 0] * nDOF_trial_element;
				for (int kb = 0; kb < nQuadraturePoints_elementBoundary; kb++)
				{
					double Dp_Dn_jump = 0.0, Du_Dn_jump = 0.0, Dv_Dn_jump = 0.0, dS;
					bool phaseA_ok = true, phaseB_ok = true;
					for (int eN_side = 0; eN_side < 2; eN_side++)
					{
						int ebN = *it,
							eN = elementBoundaryElementsArray.data()[ebN * 2 + eN_side];
						for (int i = 0; i < nDOF_test_element; i++)
						{
							Dw_Dn_jump[u_l2g.data()[eN * nDOF_test_element + i]] = 0.0;
							Dw_Dn_jump_a[u_l2g.data()[eN * nDOF_test_element + i]] = 0.0;
							Dw_Dn_jump_b[u_l2g.data()[eN * nDOF_test_element + i]] = 0.0;
						}
					}
					for (int eN_side = 0; eN_side < 2; eN_side++)
					{
						int ebN = *it,
							eN = elementBoundaryElementsArray.data()[ebN * 2 + eN_side],
							ebN_local = elementBoundaryLocalElementBoundariesArray.data()[ebN * 2 + eN_side],
							eN_nDOF_trial_element = eN * nDOF_trial_element,
							kb_s = (eN_side == 0) ? kb : gp_kmap[kb],
								ebN_local_kb = ebN_local * nQuadraturePoints_elementBoundary + kb_s,
							ebN_local_kb_nSpace = ebN_local_kb * nSpace;
						double
							u_int = 0.0,
							grad_u_int[nSpace] = {0., 0.},
							jac_int[nSpace * nSpace],
							jacDet_int,
							jacInv_int[nSpace * nSpace],
							boundaryJac[nSpace * (nSpace - 1)],
							metricTensor[(nSpace - 1) * (nSpace - 1)],
							metricTensorDetSqrt,
							u_test_dS[nDOF_test_element],
							u_grad_trial_trace[nDOF_trial_element * nSpace],
							u_grad_test_dS[nDOF_trial_element * nSpace],
							normal[2], x_int, y_int, z_int, xt_int, yt_int, zt_int, integralScaling,
							G[nSpace * nSpace], G_dd_G, tr_G, h_phi, h_penalty, penalty;
						// compute information about mapping from reference element to physical element
						ck.calculateMapping_elementBoundary(eN,
															ebN_local,
															kb_s,
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
															x_int, y_int, z_int);
						dS = metricTensorDetSqrt * dS_ref.data()[kb_s]; // kb_s mirrors kb: same Gauss weight
						// compute shape and solution information
						// shape
						// std::cout << "Calculating gradTrialFromRef from calculateJacobian() 1" << std::endl;
						ck.gradTrialFromRef(&u_grad_trial_trace_ref.data()[ebN_local_kb_nSpace * nDOF_trial_element], jacInv_int, u_grad_trial_trace);
						for (int i = 0; i < nDOF_test_element; i++)
						{
							int eN_i = eN * nDOF_test_element + i;
							for (int I = 0; I < nSpace; I++)
								Dw_Dn_jump[u_l2g.data()[eN_i]] += u_grad_trial_trace[i * nSpace + I] * normal[I];
						}
						// Ghost penalty per fluid phase -- see the matching comment in calculateResidual.
						// Must mirror the residual exactly or Newton loses one-step consistency.
						int gp_icase = -2;
						double gp_va_grad[nDOF_trial_element * nSpace], gp_vb_grad[nDOF_trial_element * nSpace];
						if (immersedBoundary)
						{
							double gp_phi_f[nDOF_trial_element], gp_nodes[nDOF_trial_element * 3];
							for (int i = 0; i < nDOF_trial_element; i++)
							{
								const int gp_eN_i = eN * nDOF_trial_element + i;
								gp_phi_f[i] = immersedBoundary_sdf_nodes.data()[u_l2g.data()[gp_eN_i]];
								for (int I = 0; I < 3; I++)
									gp_nodes[i * 3 + I] = mesh_dof.data()[u_l2g.data()[gp_eN_i] * 3 + I];
							}
							if (gf_f_boundary_gen[eN] != ifemGeometryGeneration)
							{
								gf_f_boundary_icase[eN] = gf_f_cache[eN].calculate(gp_phi_f, gp_nodes, xB_ref_faces, mua, mub, jf, true, false);
								gf_f_boundary_gen[eN] = ifemGeometryGeneration;
							}
							gp_icase = gf_f_boundary_icase[eN];
						}
						if (gp_icase == 0)
						{
							gf_f_cache[eN].set_boundary_quad(ebN_local_kb);
							for (int i = 0; i < nDOF_trial_element; i++)
							{
								gp_va_grad[i * nSpace + 0] = gf_f_cache[eN].VA_x(i);
								gp_va_grad[i * nSpace + 1] = gf_f_cache[eN].VA_y(i);
								gp_vb_grad[i * nSpace + 0] = gf_f_cache[eN].VB_x(i);
								gp_vb_grad[i * nSpace + 1] = gf_f_cache[eN].VB_y(i);
							}
						}
						else
							for (int i = 0; i < nDOF_trial_element * nSpace; i++)
							{
								gp_va_grad[i] = u_grad_trial_trace[i];
								gp_vb_grad[i] = u_grad_trial_trace[i];
							}
						phaseA_ok = phaseA_ok && (gp_icase == 0 || gp_icase == -1 || gp_icase == -2);
						phaseB_ok = phaseB_ok && (gp_icase == 0 || gp_icase == 1);
						for (int i = 0; i < nDOF_test_element; i++)
						{
							const int gp_dof = u_l2g.data()[eN * nDOF_trial_element + i];
							for (int I = 0; I < nSpace; I++)
							{
								Dw_Dn_jump_a[gp_dof] += gp_va_grad[i * nSpace + I] * normal[I];
								Dw_Dn_jump_b[gp_dof] += gp_vb_grad[i * nSpace + I] * normal[I];
							}
						}
					} // eN_side
					for (int eN_side = 0; eN_side < 2; eN_side++)
					{
						int ebN = *it,
							eN = elementBoundaryElementsArray.data()[ebN * 2 + eN_side];
						for (int i = 0; i < nDOF_test_element; i++)
						{
							int eN_i = eN * nDOF_test_element + i;
							for (int eN_side2 = 0; eN_side2 < 2; eN_side2++)
							{
								int eN2 = elementBoundaryElementsArray.data()[ebN * 2 + eN_side2];
								for (int j = 0; j < nDOF_test_element; j++)
								{
									int eN_i_j = eN_i * nDOF_test_element + j;
									int eN2_j = eN2 * nDOF_test_element + j;
									int ebN_i_j = ebN * 4 * nDOF_test_X_trial_element +
												  eN_side * 2 * nDOF_test_X_trial_element +
												  eN_side2 * nDOF_test_X_trial_element +
												  i * nDOF_trial_element +
												  j;
									std::pair<int, int> ij = std::make_pair(u_l2g.data()[eN_i], u_l2g.data()[eN2_j]);
									if (u_u_nz.count(ij))
									{
										assert(u_u_nz[ij] == csrRowIndeces_u_u.data()[eN_i] + csrColumnOffsets_eb_u_u.data()[ebN_i_j]);
									}
									else
										u_u_nz[ij] = csrRowIndeces_u_u.data()[eN_i] + csrColumnOffsets_eb_u_u.data()[ebN_i_j];
								}
							}
						}
					}
					{
						std::map<int, double> *gp_maps[2] = {&Dw_Dn_jump_a, &Dw_Dn_jump_b};
						bool gp_ok[2] = {phaseA_ok, phaseB_ok};
						double gp_w[2] = {gpMuA, gpMuB};
						if (!phaseA_ok && !phaseB_ok)
						{
							gp_maps[0] = &Dw_Dn_jump; gp_ok[0] = true;   // fall back to the combined form
							gp_ok[1] = false;
							gp_w[0] = gpMuAB;
						}
						for (int gp_p = 0; gp_p < 2; gp_p++)
						{
							if (!gp_ok[gp_p]) continue;
							std::map<int, double> &M = *gp_maps[gp_p];
							for (std::map<int, double>::iterator wi_it = M.begin(); wi_it != M.end(); ++wi_it)
								for (std::map<int, double>::iterator wj_it = M.begin(); wj_it != M.end(); ++wj_it)
								{
									std::pair<int, int> ij = std::make_pair(wi_it->first, wj_it->first);
									globalJacobian.data()[u_u_nz.at(ij)] += gp_w[gp_p] * gamma_cutfem * h_cutfem * wj_it->second * wi_it->second * dS;
								} // i,j
						}
					}
				} // kb
			} // cutfem element boundaries
			// Jacobian of the SCIFEM (eq. 3.7) consistency terms assembled in calculateResidual --
			// same construction (orientation map + edge Heaviside blend of region-wise products),
			// differentiated w.r.t. each side's u_dof. Everything is linear in u_dof, so
			// d(avg_flux_a)/d(u_dof[s,j]) = 0.5*fta_m[s][j] and d(jump_ua)/d(u_dof[s,j]) =
			// (+/-)va_m[s][j] -- the same per-dof quantities used for the test index i.
			for (std::set<int>::iterator it = ifem_boundaries.begin(); it != ifem_boundaries.end(); ++it)
			{
				{
					std::map<std::pair<int, int>, int> u_u_nz;
					int ebN0 = *it;
					for (int eN_side = 0; eN_side < 2; eN_side++)
					{
						int eN = elementBoundaryElementsArray.data()[ebN0 * 2 + eN_side];
						for (int i = 0; i < nDOF_test_element; i++)
						{
							int eN_i = eN * nDOF_test_element + i;
							for (int eN_side2 = 0; eN_side2 < 2; eN_side2++)
							{
								int eN2 = elementBoundaryElementsArray.data()[ebN0 * 2 + eN_side2];
								for (int j = 0; j < nDOF_test_element; j++)
								{
									int eN2_j = eN2 * nDOF_test_element + j;
									int ebN_i_j = ebN0 * 4 * nDOF_test_X_trial_element +
												  eN_side * 2 * nDOF_test_X_trial_element +
												  eN_side2 * nDOF_test_X_trial_element +
												  i * nDOF_trial_element + j;
									std::pair<int, int> ij = std::make_pair(u_l2g.data()[eN_i], u_l2g.data()[eN2_j]);
									if (!u_u_nz.count(ij))
										u_u_nz[ij] = csrRowIndeces_u_u.data()[eN_i] + csrColumnOffsets_eb_u_u.data()[ebN_i_j];
								}
							}
						}
					}

				int ebN = *it;
					int eN_s[2], ebN_local_s[2];
					for (int s = 0; s < 2; s++)
					{
						eN_s[s] = elementBoundaryElementsArray.data()[ebN * 2 + s];
						ebN_local_s[s] = elementBoundaryLocalElementBoundariesArray.data()[ebN * 2 + s];
					}

					// --- Edge-restricted phi (both level sets) --------------------------------
					double edge_nodes[2][3], phi_edge[2], phi_s_edge[2];
					{
						const int n0l = (ebN_local_s[0] + 1) % 3, n1l = (ebN_local_s[0] + 2) % 3;
						const int ln[2] = {n0l, n1l};
						for (int e = 0; e < 2; e++)
						{
							int eN_i = eN_s[0] * nDOF_trial_element + ln[e];
							phi_edge[e] = immersedBoundary_sdf_nodes.data()[u_l2g.data()[eN_i]];
							phi_s_edge[e] = embeddedBoundary ? embeddedBoundary_sdf_nodes.data()[u_l2g.data()[eN_i]] : 1.0;
							for (int I = 0; I < 3; I++)
								edge_nodes[e][I] = mesh_dof.data()[u_l2g.data()[eN_i] * 3 + I];
						}
					}
					// edge diameter for the interior-penalty scaling (same measure the cutfem ghost
					// penalty uses); note gamma/h here for a VALUE jump, vs gamma*h there for a
					// gradient jump.
					const double h_edge = elementBoundaryDiameter.data()[ebN];
					double edge_vec[3];
					double edge_len2 = 0.0;
					for (int I = 0; I < 3; I++)
					{
						edge_vec[I] = edge_nodes[1][I] - edge_nodes[0][I];
						edge_len2 += edge_vec[I] * edge_vec[I];
					}

					// --- Orientation map ------------------------------------------------------
					// The two elements sharing an interior face do NOT necessarily parametrize it in
					// the same direction, so quadrature index kb can denote DIFFERENT physical points
					// on the two sides (measured: 4 of 14 cut edges at test=8.0/refinement=1, with
					// exactly the mirror offsets |1-2t_k|*L of the Gauss rule). Pairing the sides by
					// raw kb then differences u at two different points, so [[u]] and {{beta grad u}}
					// pick up u's variation ALONG the edge instead of the genuine inter-element jump.
					// Build an explicit map instead: kmap[s][k] is side s's index for the physical
					// point that side 0 calls k. Matched by position, so no assumption about the
					// rule being symmetric or ascending.
					double xq[2][nQuadraturePoints_elementBoundary][3];
					for (int s = 0; s < 2; s++)
						for (int kb = 0; kb < nQuadraturePoints_elementBoundary; kb++)
						{
							double jac_t[nSpace * nSpace], jacDet_t, jacInv_t[nSpace * nSpace],
								   bJac_t[nSpace * (nSpace - 1)], mT_t[(nSpace - 1) * (nSpace - 1)],
								   mTDS_t, nrm_t[2], xt_ = 0.0, yt_ = 0.0, zt_ = 0.0;
							ck.calculateMapping_elementBoundary(eN_s[s], ebN_local_s[s], kb,
																ebN_local_s[s] * nQuadraturePoints_elementBoundary + kb,
																mesh_dof.data(), mesh_l2g.data(),
																mesh_trial_trace_ref.data(), mesh_grad_trial_trace_ref.data(),
																boundaryJac_ref.data(), jac_t, jacDet_t, jacInv_t,
																bJac_t, mT_t, mTDS_t, normal_ref.data(), nrm_t,
																xt_, yt_, zt_);
							xq[s][kb][0] = xt_; xq[s][kb][1] = yt_; xq[s][kb][2] = zt_;
						}
					// edge diameter: the length scale for the pairing tolerance just below,
					// and for the interior-penalty scaling further down (same measure the
					// cutfem ghost penalty uses); note gamma/h here for a VALUE jump, vs
					// gamma*h there for a gradient jump.
					// Match over nSpace components only: a component the mapping does not
					// define contributes nothing but noise to the distance, and the paired
					// points agree EXACTLY in the components it does define, so the winning
					// distance is 0 and the runner-up is a full quadrature spacing away.
					// Verify that outcome rather than trust it -- a mis-pick here silently
					// differences u at two different points, and assert() is compiled out by
					// the -DNDEBUG that reaches these translation units.
					int kmap[2][nQuadraturePoints_elementBoundary];
					bool kmap_used[nQuadraturePoints_elementBoundary] = {};
					for (int k = 0; k < nQuadraturePoints_elementBoundary; k++)
					{
						kmap[0][k] = k;
						int best = -1;
						double bestd = 1.0e300, nextd = 1.0e300;
						for (int j = 0; j < nQuadraturePoints_elementBoundary; j++)
						{
							double d = 0.0;
							for (int I = 0; I < nSpace; I++) // the 2D mapping never writes z
								d += std::fabs(xq[1][j][I] - xq[0][k][I]);
							if (d < bestd) { nextd = bestd; bestd = d; best = j; }
							else if (d < nextd) { nextd = d; }
						}
						if (best < 0 || kmap_used[best] || bestd >= 1.0e-8 * h_edge || nextd <= 2.0 * bestd)
						{
							std::cerr << "ADR immersedSCIFEM: quadrature points on face " << ebN
									  << " between elements " << eN_s[0] << " and " << eN_s[1]
									  << " do not pair up (k=" << k << ", best=" << best
									  << ", d=" << bestd << ", runner-up=" << nextd
									  << ", h=" << h_edge << "); the mesh is non-conforming or the"
									  << " boundary mapping is inconsistent." << std::endl;
							throw std::runtime_error("ADR immersedSCIFEM: face quadrature pairing failed");
						}
						kmap_used[best] = true;
						kmap[1][k] = best;
					}

					// --- Edge Heaviside moment fit -------------------------------------------
					// The facet integrand is DISCONTINUOUS at the interface crossing theta, so a fixed
					// whole-edge Gauss rule cannot integrate it directly. Weight the two one-sided
					// integrands by an edge-restricted moment-fit Heaviside instead: exact for any
					// polynomial integrand of degree <= nP_edge, hence exact for P1 (degree <= 1) and
					// P2 (degree <= 3) alike -- no linearity assumption, unlike a split-quadrature
					// scheme. ONE fit per edge, in side 0's parametrization: the two sides' fits are
					// mirror images, moment-equivalent but not pointwise equal, so they must not be
					// mixed. t below is therefore always measured along side 0's edge direction.
					// Branch a = {phi_f<0}, branch b = {phi_f>0}, each intersected with the fluid side
					// {phi_s>0} (phi_s_edge is 1 without an embedded solid, so this is then exactly the
					// single Gamma_f fit and its complement).
					double C_a_edge[nP_edge + 1], C_b_edge[nP_edge + 1];
					equivalent_polynomials::calculate_edge_interval_H<nP_edge>(-phi_edge[0], -phi_edge[1], phi_s_edge[0], phi_s_edge[1], C_a_edge);
					equivalent_polynomials::calculate_edge_interval_H<nP_edge>(phi_edge[0], phi_edge[1], phi_s_edge[0], phi_s_edge[1], C_b_edge);

					for (int k = 0; k < nQuadraturePoints_elementBoundary; k++)
					{
						double dS = 0.0;
						std::map<int, double> va_m[2], vb_m[2], fta_m[2], ftb_m[2];
						for (int eN_side = 0; eN_side < 2; eN_side++)
						{
							int eN = eN_s[eN_side],
								ebN_local = ebN_local_s[eN_side],
								kb = kmap[eN_side][k],
								eN_nDOF_trial_element = eN * nDOF_trial_element,
								ebN_local_kb = ebN_local * nQuadraturePoints_elementBoundary + kb;
							double jac_int[nSpace * nSpace], jacDet_int, jacInv_int[nSpace * nSpace],
								   boundaryJac[nSpace * (nSpace - 1)],
								   metricTensor[(nSpace - 1) * (nSpace - 1)], metricTensorDetSqrt,
								   normal[2], x_int, y_int, z_int;
							ck.calculateMapping_elementBoundary(eN, ebN_local, kb, ebN_local_kb,
																mesh_dof.data(), mesh_l2g.data(),
																mesh_trial_trace_ref.data(), mesh_grad_trial_trace_ref.data(),
																boundaryJac_ref.data(), jac_int, jacDet_int, jacInv_int,
																boundaryJac, metricTensor, metricTensorDetSqrt,
																normal_ref.data(), normal, x_int, y_int, z_int);
							if (eN_side == 0)
								dS = metricTensorDetSqrt * dS_ref.data()[kb];
							double sign_ne = (eN_side == 0) ? 1.0 : -1.0;

							double element_phi_f[nDOF_trial_element], element_nodes[nDOF_trial_element * 3];
							for (int i = 0; i < nDOF_trial_element; i++)
							{
								int eN_i = eN * nDOF_trial_element + i;
								element_phi_f[i] = immersedBoundary_sdf_nodes.data()[u_l2g.data()[eN_i]];
								for (int I = 0; I < 3; I++)
									element_nodes[i * 3 + I] = mesh_dof.data()[u_l2g.data()[eN_i] * 3 + I];
							}
							if (gf_f_boundary_gen[eN] != ifemGeometryGeneration)
							{
								gf_f_boundary_icase[eN] = gf_f_cache[eN].calculate(element_phi_f, element_nodes, xB_ref_faces, mua, mub, jf, true, false);
								gf_f_boundary_gen[eN] = ifemGeometryGeneration;
							}
							GfType &gf_f = gf_f_cache[eN];
							gf_f.set_boundary_quad(ebN_local_kb);

							for (int i = 0; i < nDOF_test_element; i++)
							{
								int dof_i = u_l2g.data()[eN_nDOF_trial_element + i];
								va_m[eN_side][dof_i] = gf_f.VA(i);
								vb_m[eN_side][dof_i] = gf_f.VB(i);
								double gax = gf_f.VA_x(i), gay = gf_f.VA_y(i),
									   gbx = gf_f.VB_x(i), gby = gf_f.VB_y(i);
								fta_m[eN_side][dof_i] = sign_ne * mua * (gax * normal[0] + gay * normal[1]);
								ftb_m[eN_side][dof_i] = sign_ne * mub * (gbx * normal[0] + gby * normal[1]);
							}
						} // eN_side

						// t of this physical point along side 0's edge direction, then H_hat(t)
						double proj = 0.0;
						for (int I = 0; I < 3; I++)
							proj += (xq[0][k][I] - edge_nodes[0][I]) * edge_vec[I];
						double t_edge = (edge_len2 > 1.0e-24) ? (proj / edge_len2) : 0.0;
						// H_e / ImH_e: weights of branch b / branch a on its fluid piece of the edge
						double H_e = equivalent_polynomials::evaluate_edge_poly<nP_edge>(C_b_edge, t_edge);
						double ImH_e = equivalent_polynomials::evaluate_edge_poly<nP_edge>(C_a_edge, t_edge);

						for (int side_i = 0; side_i < 2; side_i++)
						{
							double vi_sign = (side_i == 0) ? 1.0 : -1.0;
							for (std::map<int, double>::iterator wi = va_m[side_i].begin(); wi != va_m[side_i].end(); ++wi)
							{
								int dof_i = wi->first;
								double jva_i = vi_sign * va_m[side_i][dof_i], jvb_i = vi_sign * vb_m[side_i][dof_i];
								double fta_i = fta_m[side_i][dof_i], ftb_i = ftb_m[side_i][dof_i];
								for (int side_j = 0; side_j < 2; side_j++)
								{
									double vj_sign = (side_j == 0) ? 1.0 : -1.0;
									for (std::map<int, double>::iterator wj = va_m[side_j].begin(); wj != va_m[side_j].end(); ++wj)
									{
										int dof_j = wj->first;
										double jva_j = vj_sign * va_m[side_j][dof_j], jvb_j = vj_sign * vb_m[side_j][dof_j];
										double fta_j = fta_m[side_j][dof_j], ftb_j = ftb_m[side_j][dof_j];
										double dPa = 0.5 * fta_j * jva_i + 0.5 * fta_i * jva_j;
										double dPb = 0.5 * ftb_j * jvb_i + 0.5 * ftb_i * jvb_j;
										std::pair<int, int> ij = std::make_pair(dof_i, dof_j);
										globalJacobian.data()[u_u_nz.at(ij)] -=
											immersedSCIFEM_switch * (ImH_e * dPa + H_e * dPb) * dS;
										// derivative of the interior-penalty term: d[[u]]/du_j = [[v_j]],
										// so the contribution is the (symmetric, PSD) gram term below.
										globalJacobian.data()[u_u_nz.at(ij)] +=
											(immersedSCIFEM_penalty / h_edge) *
											(ImH_e * mua * jva_j * jva_i + H_e * mub * jvb_j * jvb_i) * dS;
									}
								}
							}
						}
					} // k
				}
			} // ifem element boundaries
			//
			// loop over exterior element boundaries to compute the surface integrals and load them into the global Jacobian
			//
			for (int ebNE = 0; ebNE < nExteriorElementBoundaries_global; ebNE++)
			{
				int ebN = exteriorElementBoundariesArray.data()[ebNE];
				int eN = elementBoundaryElementsArray.data()[ebN * 2 + 0],
					ebN_local = elementBoundaryLocalElementBoundariesArray.data()[ebN * 2 + 0],
					eN_nDOF_trial_element = eN * nDOF_trial_element;
				for (int kb = 0; kb < nQuadraturePoints_elementBoundary; kb++)
				{
					int ebNE_kb = ebNE * nQuadraturePoints_elementBoundary + kb,
						ebNE_kb_nSpace = ebNE_kb * nSpace,
						ebN_local_kb = ebN_local * nQuadraturePoints_elementBoundary + kb,
						ebN_local_kb_nSpace = ebN_local_kb * nSpace;

					double u_ext = 0.0,
						   grad_u_ext[nSpace],
						   m_ext = 0.0,
						   dm_ext = 0.0,
						   *a_ext,
						   f_ext[nSpace],
						   df_ext[nSpace],
						   r_ext = 0.0,
						   dflux_u_u_ext = 0.0,
						   bc_u_ext = 0.0,
						   // bc_grad_u_ext[nSpace],
						bc_m_ext = 0.0,
						   bc_dm_ext = 0.0,
						   bc_f_ext[nSpace],
						   bc_df_ext[nSpace],
						   fluxJacobian_u_u[nDOF_trial_element],
						   jac_ext[nSpace * nSpace],
						   jacDet_ext,
						   jacInv_ext[nSpace * nSpace],
						   boundaryJac[nSpace * (nSpace - 1)],
						   metricTensor[(nSpace - 1) * (nSpace - 1)],
						   metricTensorDetSqrt,
						   dS,
						   u_test_dS[nDOF_test_element],
						   u_grad_trial_trace[nDOF_trial_element * nSpace],
						   u_grad_test_dS[nDOF_trial_element * nSpace],
						   normal[nSpace], x_ext, y_ext, z_ext, xt_ext, yt_ext, zt_ext, integralScaling,
						   //
						G[nSpace * nSpace], G_dd_G, tr_G;
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
														x_ext, y_ext, z_ext);
					dS = metricTensorDetSqrt * dS_ref.data()[kb];
					ck.calculateG(jacInv_ext, G, G_dd_G, tr_G);
					// compute shape and solution information
					// shape
					// std::cout << "Calculating gradTrialFromRef from calculateJacobian() 3" << std::endl;
					ck.gradTrialFromRef(&u_grad_trial_trace_ref.data()[ebN_local_kb_nSpace * nDOF_trial_element], jacInv_ext, u_grad_trial_trace);
					// solution and gradients
					ck.valFromDOF(u_dof.data(), &u_l2g.data()[eN_nDOF_trial_element], &u_trial_trace_ref.data()[ebN_local_kb * nDOF_test_element], u_ext);
					ck.gradFromDOF(u_dof.data(), &u_l2g.data()[eN_nDOF_trial_element], u_grad_trial_trace, grad_u_ext);
					// precalculate test function products with integration weights
					for (int j = 0; j < nDOF_trial_element; j++)
					{
						u_test_dS[j] = u_test_trace_ref.data()[ebN_local_kb * nDOF_test_element + j] * dS;
						for (int I = 0; I < nSpace; I++)
							u_grad_test_dS[j * nSpace + I] = u_grad_trial_trace[j * nSpace + I] * dS; // cek hack, using trial
					}
					//
					// load the boundary values
					//
					bc_u_ext = isDOFBoundary_u.data()[ebNE_kb] * ebqe_bc_u_ext.data()[ebNE_kb] + (1 - isDOFBoundary_u.data()[ebNE_kb]) * u_ext;
					a_ext = &ebqe_a.data()[ebNE_kb * sd_rowptr.data()[nSpace]];
					for (int I = 0; I < nSpace; I++)
					{
						df_ext[I] = ebqe_v.data()[ebNE_kb * nSpace + I];
						bc_df_ext[I] = ebqe_v.data()[ebNE_kb * nSpace + I];
					}
					//
					// calculate the numerical fluxes
					//
					exteriorNumericalAdvectiveFluxDerivative(isDOFBoundary_u.data()[ebNE_kb],
															 isAdvectiveFluxBoundary_u.data()[ebNE_kb],
															 normal,
															 df_ext,
															 dflux_u_u_ext);
					//
					// calculate the flux jacobian
					//
					for (int j = 0; j < nDOF_trial_element; j++)
					{
						// int ebNE_kb_j = ebNE_kb*nDOF_trial_element+j;
						int j_nSpace = j * nSpace, ebN_local_kb_j = ebN_local_kb * nDOF_trial_element + j;
						fluxJacobian_u_u[j] = ExteriorNumericalDiffusiveFluxJacobian(sd_rowptr.data(),
																					 sd_colind.data(),
																					 isDOFBoundary_u.data()[ebNE_kb],
																					 isDiffusiveFluxBoundary_u.data()[ebNE_kb],
																					 normal,
																					 a_ext,
																					 u_trial_trace_ref.data()[ebN_local_kb_j],
																					 &u_grad_trial_trace[j_nSpace],
																					 ebqe_penalty_ext.data()[ebNE_kb]) +
											  ck.ExteriorNumericalAdvectiveFluxJacobian(dflux_u_u_ext, u_trial_trace_ref.data()[ebN_local_kb_j]);
					} // j
					//
					// update the global Jacobian from the flux Jacobian
					//
					for (int i = 0; i < nDOF_test_element; i++)
					{
						int eN_i = eN * nDOF_test_element + i;
						int i_nSpace = i * nSpace;
						for (int j = 0; j < nDOF_trial_element; j++)
						{
							int ebN_i_j = ebN * 4 * nDOF_test_X_trial_element + i * nDOF_trial_element + j;
							int ebN_local_kb_j = ebN_local_kb * nDOF_trial_element + j;

							globalJacobian.data()[csrRowIndeces_u_u.data()[eN_i] + csrColumnOffsets_eb_u_u.data()[ebN_i_j]] += fluxJacobian_u_u[j] * u_test_dS[i] +
																															   ck.ExteriorElementBoundaryDiffusionAdjointJacobian(isDOFBoundary_u.data()[ebNE_kb],
																																												  isDiffusiveFluxBoundary_u.data()[ebNE_kb],
																																												  eb_adjoint_sigma,
																																												  u_trial_trace_ref.data()[ebN_local_kb_j],
																																												  normal,
																																												  sd_rowptr.data(),
																																												  sd_colind.data(),
																																												  a_ext,
																																												  &u_grad_test_dS[i_nSpace]);
						} // j
					} // i
				} // kb
			} // ebNE
		} // computeJacobian
	}; // cADR

	inline cADR_base *newADR(int nSpaceIn,
							 int nQuadraturePoints_elementIn,
							 int nDOF_mesh_trial_elementIn,
							 int nDOF_trial_elementIn,
							 int nDOF_test_elementIn,
							 int nQuadraturePoints_elementBoundaryIn,
							 int CompKernelFlag)
	{
		// P2 with the degree-8 (16-point) triangle rule, qOrder 8: the rule that integrates the
		// degree-4 equivalent polynomials times the degree-4 Gamma_s integrands exactly. ADR-only,
		// so it is not added to the shared ModelFactory.
		if (nSpaceIn == 2 && CompKernelFlag == 0 && nDOF_mesh_trial_elementIn == 3 && nDOF_trial_elementIn == 6 &&
			nDOF_test_elementIn == 6 && nQuadraturePoints_elementIn == 16 && nQuadraturePoints_elementBoundaryIn == 8)
			return static_cast<cADR_base *>(new cADR<CompKernel<2, 3, 6, 6>, 2, 16, 3, 6, 6, 8>());
		if (nSpaceIn == 2)
			return proteus::chooseAndAllocateDiscretization2D<cADR_base, cADR, CompKernel>(nSpaceIn,
																						   nQuadraturePoints_elementIn,
																						   nDOF_mesh_trial_elementIn,
																						   nDOF_trial_elementIn,
																						   nDOF_test_elementIn,
																						   nQuadraturePoints_elementBoundaryIn,
																						   CompKernelFlag);
		else
			return proteus::chooseAndAllocateDiscretization<cADR_base, cADR, CompKernel>(nSpaceIn,
																						 nQuadraturePoints_elementIn,
																						 nDOF_mesh_trial_elementIn,
																						 nDOF_trial_elementIn,
																						 nDOF_test_elementIn,
																						 nQuadraturePoints_elementBoundaryIn,
																						 CompKernelFlag);
	}
} // proteus
#endif