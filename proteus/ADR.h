#ifndef ADR_H
#define ADR_H
#include <algorithm>
#include <cmath>
#include <type_traits>
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
#include "cutquadric.h"
#include "UnfittedKernels.h"
#include "Unfitted.h"
#include "mprans/ArgumentsDict.h"
#include "pyarray.h"

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
		const int nDOF_test_X_trial_element;
		CompKernelType ck;
		// nEBQ is sized for ALL faces (nDOF_mesh_trial_element * nQuadraturePoints_elementBoundary),
		// not one face, so the immersed-interface facet terms can evaluate va/vb at the quadrature
		// points of the face they are actually integrating -- see the xB_ref_faces comment in
		// calculateResidual for why passing the reference *boundary* points directly is wrong.
		// Degree k of the fitted H/ImH/D polynomials (GfType, the triply-cut composite fits, the edge
		// fits). Each cut-cell term is sum_q w_q*fit*F, exact when deg F <= k AND k + deg F <= the
		// element rule's degree. The Gamma_s Nitsche penalty / Robin terms are degree 2p. P1: k=3 with
		// the 12-point (degree-6) rule. P2: k=4 needs the 16-point (degree-8) rule (qOrder=8); with
		// the 12-point rule k=3 is kept, which is exact through degree 2p-1=3 and still rate-optimal.
		// Polynomial order of the trial space, from the element's dof count in nSpace dimensions
		// (simplex Lagrange: P1 has nSpace+1 dofs, P2 has (nSpace+1)(nSpace+2)/2). Not a triangle
		// test: a 3D P1 tetrahedron has 4 dofs and a P2 one has 10.
		static const int pOrder = (nDOF_trial_element == nSpace + 1) ? 1 : ((nDOF_trial_element == (nSpace + 1) * (nSpace + 2) / 2) ? 2 : 0);
		// 2D P2: k=4 with the 16-point (degree-8) triangle rule. 3D P2: k=3 with the degree-7 rule (31 points,
		// qOrder 7, the default, decision P9 of C44 phase 1b), or k=4 with the degree-8 rule (45 points, qOrder 8;
		// both tetrahedron rules have some negative weights). The degree-4 Nitsche penalty is exact against k=3
		// only through degree 3; with the curved cut k=4 is what reaches L2 rate 3 and H1 rate 2 (measured, B4).
		static const int nP_fit = (pOrder == 2 && (nSpace == 2 ? nQuadraturePoints_element >= 16 : nQuadraturePoints_element >= 45)) ? 4 : 3;
		// The fluid interface: the moment fits of H, 1-H, D and the two-sided IFE basis of the same cut
		using GfType = unfitted::IFEFunctions<nSpace, nDOF_trial_element, nP_fit, nQuadraturePoints_element, nDOF_mesh_trial_element * nQuadraturePoints_elementBoundary>;
		// The solid level set only needs the H/D moment fits, never the two-sided IFEM basis (2D keeps the
		// useIfemBasis = true fits, bit for bit; in 3D the plain fit, C44 phase 1).
		using GfTypeS = typename std::conditional<nSpace == 2,
			GeneralizedFunctions<nSpace, nDOF_trial_element, nP_fit, nQuadraturePoints_element, nDOF_mesh_trial_element * nQuadraturePoints_elementBoundary>,
			equivalent_polynomials::GeneralizedFunctions_mix<nSpace, nDOF_trial_element, nP_fit, nQuadraturePoints_element, nDOF_mesh_trial_element * nQuadraturePoints_elementBoundary, false>>::type;
		// Degree of the edge-restricted moment-fit Heaviside used by the edge terms (SCIFEM, the
		// triply-cut correction). The consistency terms are degree 2p-1 and the interior penalty 2p;
		// the 1D boundary rule (qOrder points) is exact far beyond nP_edge + 2p.
		static const int nP_edge = nP_fit;
		// The geometry of the cuts (classification, moment fits, the IFE basis, composite fits, flux-jump lift, which elements
		// carry fluid, the faces of the face terms) is built once by Geometry::prepare() and read by the residual and the
		// Jacobian alike; none of it depends on the solution, so it is rebuilt only when the level sets change (see
		// recomputeIFEMGeometry / invalidateIFEMGeometry on the Python side). Storage is for cut elements only.
		// the 2D moment fit on a cut triangular face of a tetrahedron (3D SCIFEM); unused in 2D
		equivalent_polynomials::Simplex<2, 3, nP_edge, nQuadraturePoints_elementBoundary, 1, false> face_gf;
		static const int nCompMon = (nP_fit + 1) * (nP_fit + 2) / 2; // monomials up to degree nP_fit
		using Geometry = unfitted::CutGeometry<GfType, GfTypeS, nSpace, nDOF_trial_element, nQuadraturePoints_element,
											   nDOF_mesh_trial_element, nQuadraturePoints_elementBoundary, nP_fit>;
		Geometry geom;
		// C44 phase 1b: the curved solid (embeddedBoundary_curved) on P2 tetrahedra: moment fits of H, D and the three
		// normal-weighted Diracs (cutquadric.h) instead of the planar fit
		bool curvedSolid = false;
		cADR() : nDOF_test_X_trial_element(nDOF_test_element * nDOF_trial_element), ck()
		{
		}
		// 3D SCIFEM: moment-fit weights of the two branches on a triangular face that Gamma_f cuts.
		// H = weight of branch b (phi_f > 0), ImH = weight of branch a (phi_f < 0), at the face quadrature points
		// of side 0 (physical positions xq0), exact for polynomials of degree <= nP_edge. The fit is the 2D
		// equivalent polynomial on the reference triangle: the face's three vertex values of phi_f and the
		// barycentric coordinates of each quadrature point in the face (side 0's vertex order) are all it needs,
		// because the map from the reference triangle to the face is affine and the physical weights carry the
		// face area through dS. Face k of a tetrahedron is opposite local node k; its vertices are the other three.
		inline void cutFaceWeights(const int eN0, const int ebN_local0, const double (*xq0)[3],
								   const int *u_l2g, const double *mesh_dof, const double *phi_nodes,
								   double *H, double *ImH)
		{
			if constexpr (nSpace == 3)
			{
				double P[3][3], phi3[3];
				for (int m = 0; m < 3; m++)
				{
					const int ln = (m < ebN_local0) ? m : m + 1;
					const int dof = u_l2g[eN0 * nDOF_trial_element + ln];
					phi3[m] = phi_nodes[dof];
					for (int I = 0; I < 3; I++)
						P[m][I] = mesh_dof[dof * 3 + I];
				}
				double v0[3], v1[3];
				for (int I = 0; I < 3; I++)
				{
					v0[I] = P[1][I] - P[0][I];
					v1[I] = P[2][I] - P[0][I];
				}
				const double d00 = v0[0] * v0[0] + v0[1] * v0[1] + v0[2] * v0[2], d01 = v0[0] * v1[0] + v0[1] * v1[1] + v0[2] * v1[2],
							 d11 = v1[0] * v1[0] + v1[1] * v1[1] + v1[2] * v1[2];
				const double den = d00 * d11 - d01 * d01;
				double xi[nQuadraturePoints_elementBoundary * 3];
				for (int k = 0; k < nQuadraturePoints_elementBoundary; k++)
				{
					double v2[3];
					for (int I = 0; I < 3; I++)
						v2[I] = xq0[k][I] - P[0][I];
					const double d20 = v2[0] * v0[0] + v2[1] * v0[1] + v2[2] * v0[2], d21 = v2[0] * v1[0] + v2[1] * v1[1] + v2[2] * v1[2];
					xi[k * 3 + 0] = (d11 * d20 - d01 * d21) / den;
					xi[k * 3 + 1] = (d00 * d21 - d01 * d20) / den;
					xi[k * 3 + 2] = 0.0;
				}
				static const double ref_nodes[9] = {0., 0., 0., 1., 0., 0., 0., 1., 0.};
				const int icase = face_gf.calculate(phi3, ref_nodes, xi, false);
				for (int k = 0; k < nQuadraturePoints_elementBoundary; k++)
				{
					if (icase == 1)
					{
						H[k] = 1.0;
						ImH[k] = 0.0;
					}
					else if (icase == -1)
					{
						H[k] = 0.0;
						ImH[k] = 1.0;
					}
					else
					{
						face_gf.set_quad(k);
						H[k] = face_gf.H(0., 0.);
						ImH[k] = face_gf.ImH(0., 0.);
					}
				}
			}
		}
		// Build (or reuse) the geometry of the cuts. Called first by the residual and by the Jacobian, so neither depends on
		// the other having run.
		inline void prepareGeometry(arguments_dict &args, const bool curved)
		{
			proteus::pyarray<double> &x_ref = args.array<double>("x_ref");
			proteus::pyarray<double> &mesh_trial_trace_ref = args.array<double>("mesh_trial_trace_ref");
			proteus::pyarray<double> &mesh_dof = args.array<double>("mesh_dof");
			proteus::pyarray<int> &u_l2g = args.array<int>("u_l2g");
			proteus::pyarray<double> &dV_ref = args.array<double>("dV_ref");
			proteus::pyarray<int> &elementBoundariesArray = args.array<int>("elementBoundariesArray");
			proteus::pyarray<int> &elementBoundaryElementsArray = args.array<int>("elementBoundaryElementsArray");
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
					xB_ref_faces[fkb * 3 + 2] = (nSpace == 3) ? phi_m[3] : 0.0;
				}
			typename Geometry::Inputs in;
			in.nElements = args.scalar<int>("nElements_global");
			in.l2g = u_l2g.data();
			in.phi_s_nodes = args.array<double>("embeddedBoundary_sdf_nodes").data();
			in.phi_f_nodes = args.array<double>("immersedBoundary_sdf_nodes").data();
			in.mesh_dof = mesh_dof.data();
			in.x_ref = x_ref.data();
			in.xB_ref_faces = xB_ref_faces;
			in.dV_ref = dV_ref.data();
			in.elementBoundaries = elementBoundariesArray.data();
			in.elementBoundaryElements = elementBoundaryElementsArray.data();
			in.fluxJump_element = args.array<double>("immersedBoundary_fluxJump_element").data();
			in.solutionJump_nodes = args.array<double>("immersedBoundary_solutionJump_nodes").data();
			in.fluxJumpLift = args.scalar<int>("immersedBoundary_fluxJumpLift");
			in.mua = args.scalar<double>("mua");
			in.mub = args.scalar<double>("mub");
			in.jf = args.scalar<double>("jf");
			in.curvedSolid = curved;
			in.cut_label = args.array<int>("embeddedBoundary_cut_label").data();
			geom.prepare(in, args.scalar<int>("recomputeIFEMGeometry"));
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

		// Strong diffusion residual -div(a grad u) at a point: a_IJ u_IJ + (d_I a_IJ) u_J,
		// with the sparse tensor a (sd_rowptr/sd_colind) and a_div[J] = sum_I d_I a_IJ.
		inline double diffusionStrong(const int *rowptr, const int *colind, const double *a, const double *a_div,
									  const double *hess, const double *grad)
		{
			double tmp = 0.0;
			for (int I = 0; I < nSpace; I++)
			{
				for (int m = rowptr[I]; m < rowptr[I + 1]; m++)
					tmp += a[m] * hess[I * nSpace + colind[m]];
				tmp += a_div[I] * grad[I];
			}
			return -tmp;
		}

		// inverse-estimate constant of the diffusive time scale tau^-1 = 2|b|/h + c a/h^2 + 1/dt (Codina; measured
		// on P1 and P2: c = 4 and 16 give the same rates, test/ADR_SGS)
		static constexpr double sgeDiffusiveConstant = 4.0;
		// largest diagonal entry of the diffusion tensor: the scale of its stabilization time
		inline double diffusionScale(const int *rowptr, const int *colind, const double *a)
		{
			double amax = 0.0;
			for (int I = 0; I < nSpace; I++)
				for (int m = rowptr[I]; m < rowptr[I + 1]; m++)
					if (colind[m] == I)
						amax = std::max(amax, std::fabs(a[m]));
			return amax;
		}

		inline void calculateSubgridError_tau(const double &elementDiameter,
											  const double &dmt,
											  const double dH[nSpace],
											  const double &diffusiveScale, // c_inv * |a|: 0 turns the diffusive time scale off
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
			tau = 1.0 / (2.0 * nrm_v / h + diffusiveScale / (h * h) + oneByAbsdt + 1.0e-8);
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

		// The cut-cell penalty terms at one quadrature point go through the kernels of UnfittedKernels.h (one form: a
		// coefficient times a solution jump times a test-function jump). The caller passes the test and the trial
		// functions separately because Petrov-Galerkin tests with the ordinary hats; for a Galerkin call they are the
		// same arrays. A residual call passes J = nullptr (and the trial arrays null), a Jacobian call r = nullptr.
		//
		// Nitsche Dirichlet terms on Gamma_s: consistency -[a du/dn] v, adjoint -[a dv/dn](u - u_s) and the penalty
		// (D_s a pen) (u - u_s) v. m[I] = D_s * (outward normal of the fluid)[I] is the Dirac measure carrying the normal
		// (for the planar fit D_s times one normal per element, for the curved fit the normal-weighted Dirac -Dn_s of
		// C44 phase 1b, which is exact for polynomial integrands on a curved Gamma_s); D_s alone multiplies the penalty.
		inline void nitscheDirichlet(const double w, const double D_s, const double pen, const double a, const double *m,
									 const double u_s, const double u, const double *grad_u,
									 const double *v_test, const double *grad_v_test,
									 const double *v_trial, const double *grad_v_trial,
									 double *r, double *J)
		{
			double flux_u = 0.0;
			for (int I = 0; I < nSpace; I++)
				flux_u += m[I] * grad_u[I];
			flux_u *= a;
			double flux_v_test[nDOF_test_element], flux_v_trial[nDOF_trial_element];
			for (int i = 0; i < nDOF_test_element; i++)
			{
				double s = 0.0;
				for (int I = 0; I < nSpace; I++)
					s += m[I] * grad_v_test[i * nSpace + I];
				flux_v_test[i] = a * s;
			}
			if (J)
				for (int j = 0; j < nDOF_trial_element; j++)
				{
					double s = 0.0;
					for (int I = 0; I < nSpace; I++)
						s += m[I] * grad_v_trial[j * nSpace + I];
					flux_v_trial[j] = a * s;
				}
			unfitted::symmetricConsistency<nDOF_test_element>(w, 1.0, flux_u, u - u_s, flux_v_test, v_test, flux_v_trial, v_trial, r, J);
			unfitted::jumpPenalty<nDOF_test_element>(w, D_s * a * pen, u - u_s, v_test, v_trial, r, J);
		}

		// Robin/convective condition a du/dn = -h (u - u_inf) on Gamma_s (n the fluid's own outward normal, into the
		// solid): the physical law replaces the natural boundary term that integration by parts leaves behind, so it is
		// one reaction-like term, a jump penalty with c = D_s h and no consistency or adjoint part.
		inline void robinTerm(const double w, const double D_s, const double h_coeff, const double u_infinity, const double u,
							  const double *v_test, const double *v_trial, double *r, double *J)
		{
			unfitted::jumpPenalty<nDOF_test_element>(w, D_s * h_coeff, u - u_infinity, v_test, v_trial, r, J);
		}

		// Prescribed flux jump across Gamma_f, [beta du/dn] = fluxJump(x) + fluxJumpVector(x) . n, supplied by the
		// physics file (the vector part carries the cases whose jump depends on the interface normal; both default to
		// zero). The load is  int_Gamma_f [beta du/dn] v  = + jump_flux D_f v, i.e. unfitted::load with g = -jump_flux.
		inline double fluxJumpValue(const double fluxJump, const double *fluxJumpVector, const double *n)
		{
			double jump_flux = fluxJump;
			for (int I = 0; I < nSpace; I++)
				jump_flux += fluxJumpVector[I] * n[I];
			return jump_flux;
		}

		// ---- face terms ---------------------------------------------------------------------------------------------------
		// What the face terms (ghost penalty, SCIFEM, triply-cut edge) read about the mesh, as plain pointers.
		struct FaceCtx
		{
			double *mesh_dof, *mesh_trial_trace_ref, *mesh_grad_trial_trace_ref, *boundaryJac_ref, *normal_ref, *dS_ref,
				*u_grad_trial_trace_ref, *u_dof, *phi_f_nodes, *phi_s_nodes, *jump_nodes;
			int *mesh_l2g, *u_l2g, *eb_elements, *eb_local;
			bool immersed, embedded;
		};
		inline FaceCtx makeFaceCtx(arguments_dict &args)
		{
			FaceCtx c;
			c.mesh_dof = args.array<double>("mesh_dof").data();
			c.mesh_trial_trace_ref = args.array<double>("mesh_trial_trace_ref").data();
			c.mesh_grad_trial_trace_ref = args.array<double>("mesh_grad_trial_trace_ref").data();
			c.boundaryJac_ref = args.array<double>("boundaryJac_ref").data();
			c.normal_ref = args.array<double>("normal_ref").data();
			c.dS_ref = args.array<double>("dS_ref").data();
			c.u_grad_trial_trace_ref = args.array<double>("u_grad_trial_trace_ref").data();
			c.u_dof = args.array<double>("u_dof").data();
			c.phi_f_nodes = args.array<double>("immersedBoundary_sdf_nodes").data();
			c.phi_s_nodes = args.array<double>("embeddedBoundary_sdf_nodes").data();
			c.jump_nodes = args.array<double>("immersedBoundary_solutionJump_nodes").data();
			c.mesh_l2g = args.array<int>("mesh_l2g").data();
			c.u_l2g = args.array<int>("u_l2g").data();
			c.eb_elements = args.array<int>("elementBoundaryElementsArray").data();
			c.eb_local = args.array<int>("elementBoundaryLocalElementBoundariesArray").data();
			c.immersed = args.scalar<int>("immersedBoundary");
			c.embedded = args.scalar<int>("embeddedBoundary");
			return c;
		}

		// The two elements sharing an interior face do NOT necessarily parametrize it in the same direction, so
		// quadrature index kb can denote DIFFERENT physical points on the two sides (measured: 4 of 14 cut edges at
		// test=8.0/refinement=1, with exactly the mirror offsets |1-2t_k|*L of the Gauss rule). Pairing the sides by raw kb
		// then differences u at two different points, so [[u]] and {{beta grad u}} pick up u's variation ALONG the face
		// instead of the genuine inter-element jump. kmap[s][k] is side s's index for the physical point that side 0 calls
		// k, matched by position (no assumption about the rule being symmetric or ascending); xq are the physical points
		// of both sides. Match over nSpace components only: a component the mapping does not define contributes nothing
		// but noise to the distance. With strict, the pairing is verified (the winning distance is 0 and the runner-up is
		// a full quadrature spacing away) and a failure throws: a mis-pick silently differences u at two different points,
		// and assert() is compiled out by the -DNDEBUG that reaches these translation units. Without it the nearest point
		// is taken (the ghost penalty's pairing).
		inline void pairFacePoints(const FaceCtx &c, const int ebN, int kmap[2][nQuadraturePoints_elementBoundary],
								   double xq[2][nQuadraturePoints_elementBoundary][3], const double h_edge, const bool strict)
		{
			for (int s = 0; s < 2; s++)
			{
				const int eN_g = c.eb_elements[ebN * 2 + s], ebN_local_g = c.eb_local[ebN * 2 + s];
				for (int kb = 0; kb < nQuadraturePoints_elementBoundary; kb++)
				{
					double jac_g[nSpace * nSpace], jacDet_g, jacInv_g[nSpace * nSpace],
						bJac_g[nSpace * (nSpace - 1)], mT_g[(nSpace - 1) * (nSpace - 1)], mTDS_g, nrm_g[3], xt = 0.0, yt = 0.0, zt = 0.0;
					ck.calculateMapping_elementBoundary(eN_g, ebN_local_g, kb, ebN_local_g * nQuadraturePoints_elementBoundary + kb,
														c.mesh_dof, c.mesh_l2g, c.mesh_trial_trace_ref, c.mesh_grad_trial_trace_ref,
														c.boundaryJac_ref, jac_g, jacDet_g, jacInv_g, bJac_g, mT_g, mTDS_g, c.normal_ref, nrm_g,
														xt, yt, zt);
					xq[s][kb][0] = xt;
					xq[s][kb][1] = yt;
					xq[s][kb][2] = zt;
				}
			}
			bool used[nQuadraturePoints_elementBoundary] = {};
			for (int k = 0; k < nQuadraturePoints_elementBoundary; k++)
			{
				kmap[0][k] = k;
				int best = strict ? -1 : 0;
				double bestd = 1.0e300, nextd = 1.0e300;
				for (int j = 0; j < nQuadraturePoints_elementBoundary; j++)
				{
					double d = 0.0;
					for (int I = 0; I < nSpace; I++)
						d += std::fabs(xq[1][j][I] - xq[0][k][I]);
					if (d < bestd)
					{
						nextd = bestd;
						bestd = d;
						best = j;
					}
					else if (d < nextd)
						nextd = d;
				}
				if (strict && (best < 0 || used[best] || bestd >= 1.0e-8 * h_edge || nextd <= 2.0 * bestd))
				{
					std::cerr << "ADR immersedSCIFEM: quadrature points on face " << ebN
							  << " between elements " << c.eb_elements[ebN * 2 + 0] << " and " << c.eb_elements[ebN * 2 + 1]
							  << " do not pair up (k=" << k << ", best=" << best
							  << ", d=" << bestd << ", runner-up=" << nextd
							  << ", h=" << h_edge << "); the mesh is non-conforming or the"
							  << " boundary mapping is inconsistent." << std::endl;
					throw std::runtime_error("ADR immersedSCIFEM: face quadrature pairing failed");
				}
				used[best] = true;
				kmap[1][k] = best;
			}
		}

		// Ghost penalty at one quadrature point of an interior face of Gamma_s-cut elements, per fluid phase. The penalty
		// kills *spurious* normal-derivative jumps across a cut element's face. On an element the fluid interface also
		// cuts, the combined IFEM solution genuinely kinks at Gamma_f, so penalising the combined gradient penalises
		// physics and destroys exactness. Each phase's own polynomial extension IS smooth across the face, so [[d(u_a)/dn]]
		// and [[d(u_b)/dn]] are penalised separately -- the same per-side treatment the bulk (wA/wB) and Nitsche
		// (D_s_a/D_s_b) terms get. A face only gets a phase's penalty if BOTH its elements have that phase; when neither is
		// shared (Gamma_f runs along the face itself) the combined penalty is kept so the face is still stabilised.
		//
		// The local DOFs are those of both elements, side 0 first: jv_*[s * n + i] is the normal derivative of basis
		// function i of side s times that side's outward normal, so the sum is the jump; ju_* is the jump of the normal
		// derivative of the solution (residual only: with withSolution == false it is not formed). a and b are the
		// phases, c the combined (standard-basis) form. dS is the weight of the last side, as always here.
		struct GhostPoint
		{
			double jv_a[2 * nDOF_test_element], jv_b[2 * nDOF_test_element], jv_c[2 * nDOF_test_element];
			double ju_a = 0.0, ju_b = 0.0, ju_c = 0.0, dS = 0.0;
			bool okA = true, okB = true;
		};
		inline void ghostPoint(const FaceCtx &c, const int ebN, const int kb, const int kb1, const bool withSolution, GhostPoint &gp)
		{
			gp.ju_a = gp.ju_b = gp.ju_c = 0.0;
			gp.okA = gp.okB = true;
			for (int eN_side = 0; eN_side < 2; eN_side++)
			{
				const int eN = c.eb_elements[ebN * 2 + eN_side], ebN_local = c.eb_local[ebN * 2 + eN_side],
						  kb_s = (eN_side == 0) ? kb : kb1, ebN_local_kb = ebN_local * nQuadraturePoints_elementBoundary + kb_s,
						  ebN_local_kb_nSpace = ebN_local_kb * nSpace;
				double grad_u_int[nSpace] = {0., 0.}, jac_int[nSpace * nSpace], jacDet_int, jacInv_int[nSpace * nSpace],
					boundaryJac[nSpace * (nSpace - 1)], metricTensor[(nSpace - 1) * (nSpace - 1)], metricTensorDetSqrt,
					u_grad_trial_trace[nDOF_trial_element * nSpace], normal[3], x_int, y_int, z_int;
				ck.calculateMapping_elementBoundary(eN, ebN_local, kb_s, ebN_local_kb, c.mesh_dof, c.mesh_l2g,
													c.mesh_trial_trace_ref, c.mesh_grad_trial_trace_ref, c.boundaryJac_ref,
													jac_int, jacDet_int, jacInv_int, boundaryJac, metricTensor, metricTensorDetSqrt,
													c.normal_ref, normal, x_int, y_int, z_int);
				gp.dS = metricTensorDetSqrt * c.dS_ref[kb_s]; // kb_s mirrors kb: same Gauss weight
				ck.gradTrialFromRef(&c.u_grad_trial_trace_ref[ebN_local_kb_nSpace * nDOF_trial_element], jacInv_int, u_grad_trial_trace);
				const int l2g0 = eN * nDOF_trial_element;
				if (withSolution)
				{
					ck.gradFromDOF(c.u_dof, &c.u_l2g[l2g0], u_grad_trial_trace, grad_u_int);
					for (int I = 0; I < nSpace; I++)
						gp.ju_c += grad_u_int[I] * normal[I];
				}
				for (int i = 0; i < nDOF_test_element; i++)
				{
					double s = 0.0;
					for (int I = 0; I < nSpace; I++)
						s += u_grad_trial_trace[i * nSpace + I] * normal[I];
					gp.jv_c[eN_side * nDOF_test_element + i] = s;
				}
				// the phase fields: the branch basis on a Gamma_f-cut element, the standard basis elsewhere
				const int icaseF = c.immersed ? geom.icaseF(eN) : -2; // -2: no fluid interface in play, single phase
				double va_grad[nDOF_trial_element * nSpace], vb_grad[nDOF_trial_element * nSpace];
				double JA[nDOF_trial_element], JB[nDOF_trial_element];
				std::fill(JA, JA + nDOF_trial_element, 0.0);
				std::fill(JB, JB + nDOF_trial_element, 0.0);
				if (icaseF == 0)
				{
					// the solution-jump lift of this element (assign_jump_side's node pattern), zero unless eN is Gamma_f-cut
					for (int i = 0; i < nDOF_trial_element; i++)
					{
						const double jump_i = c.jump_nodes[c.u_l2g[eN * nDOF_trial_element + i]];
						if (c.phi_f_nodes[c.u_l2g[eN * nDOF_trial_element + i]] > 0.0)
							JA[i] = -jump_i;
						else
							JB[i] = jump_i;
					}
					GfType &gf = geom.gfF(eN);
					gf.set_boundary_quad(ebN_local_kb);
					for (int i = 0; i < nDOF_trial_element; i++)
					{
						va_grad[i * nSpace + 0] = gf.VA_x(i);
						va_grad[i * nSpace + 1] = gf.VA_y(i);
						vb_grad[i * nSpace + 0] = gf.VB_x(i);
						vb_grad[i * nSpace + 1] = gf.VB_y(i);
						if constexpr (nSpace == 3)
						{
							va_grad[i * nSpace + 2] = gf.VA_z(i);
							vb_grad[i * nSpace + 2] = gf.VB_z(i);
						}
					}
				}
				else
					for (int i = 0; i < nDOF_trial_element * nSpace; i++)
					{
						va_grad[i] = u_grad_trial_trace[i];
						vb_grad[i] = u_grad_trial_trace[i];
					}
				// Which phases does this element actually carry?
				gp.okA = gp.okA && (icaseF == 0 || icaseF == -1 || icaseF == -2);
				gp.okB = gp.okB && (icaseF == 0 || icaseF == 1);
				for (int i = 0; i < nDOF_test_element; i++)
				{
					double sa = 0.0, sb = 0.0;
					for (int I = 0; I < nSpace; I++)
					{
						sa += va_grad[i * nSpace + I] * normal[I];
						sb += vb_grad[i * nSpace + I] * normal[I];
					}
					gp.jv_a[eN_side * nDOF_test_element + i] = sa;
					gp.jv_b[eN_side * nDOF_test_element + i] = sb;
				}
				if (withSolution)
				{
					double eu[nDOF_trial_element], gua[nSpace] = {0., 0.}, gub[nSpace] = {0., 0.};
					for (int i = 0; i < nDOF_trial_element; i++)
						eu[i] = c.u_dof[c.u_l2g[l2g0 + i]];
					ck.gradFromElementDOF(eu, va_grad, gua);
					ck.gradFromElementDOF(eu, vb_grad, gub);
					// The ghost penalty must compare the TOTAL branch field (u_a+u_ja / u_b+u_jb) across the shared face, not
					// the jump-free piece alone, or a genuine prescribed solution jump reads as a spurious normal-derivative
					// kink between elements. Zero (no-op) unless eN is Gamma_f-cut.
					double gja[nSpace] = {0., 0.}, gjb[nSpace] = {0., 0.};
					ck.gradFromElementDOF(JA, va_grad, gja);
					ck.gradFromElementDOF(JB, vb_grad, gjb);
					if (icaseF == 0) // flux-jump lift; set_boundary_quad(ebN_local_kb) is current
					{
						const double J = geom.fluxLift(eN);
						gja[0] += J * geom.gfF(eN).PSIA_x();
						gja[1] += J * geom.gfF(eN).PSIA_y();
						gjb[0] += J * geom.gfF(eN).PSIB_x();
						gjb[1] += J * geom.gfF(eN).PSIB_y();
					}
					for (int I = 0; I < nSpace; I++)
					{
						gp.ju_a += (gua[I] + gja[I]) * normal[I];
						gp.ju_b += (gub[I] + gjb[I]) * normal[I];
					}
				}
			}
		}

		// The weights of the two branches on the cut face of a SCIFEM face, at side 0's quadrature points xq0: H = branch b
		// (phi_f > 0), ImH = branch a. The facet integrand is DISCONTINUOUS at the interface crossing, so a fixed whole-face
		// Gauss rule cannot integrate it directly: the two one-sided integrands are weighted by a moment-fit Heaviside
		// instead, exact for any polynomial integrand of degree <= nP_edge (P1: <= 1, P2: <= 3 for the consistency terms,
		// 2p for the penalty) -- no linearity assumption, unlike a split-quadrature scheme. ONE fit per face, in side 0's
		// parametrization: the two sides' fits are mirror images, moment-equivalent but not pointwise equal, so they must not
		// be mixed. Each branch is intersected with the fluid side {phi_s > 0} (phi_s is 1 without an embedded solid, so this
		// is then exactly the single Gamma_f fit and its complement). 2D: the edge fit along the edge; 3D: the 2D Heaviside
		// fit on the cut triangle (cutFaceWeights).
		inline void faceCutWeights(const FaceCtx &c, const int ebN, const double (*xq0)[3], double *ImH_e, double *H_e)
		{
			const int eN0 = c.eb_elements[ebN * 2], ebN_local0 = c.eb_local[ebN * 2];
			if constexpr (nSpace == 2)
			{
				const int n0l = (ebN_local0 + 1) % 3, n1l = (ebN_local0 + 2) % 3;
				const int ln[2] = {n0l, n1l};
				double edge_nodes[2][3] = {}, phi_edge[2] = {}, phi_s_edge[2] = {};
				for (int e = 0; e < 2; e++)
				{
					const int dof = c.u_l2g[eN0 * nDOF_trial_element + ln[e]];
					phi_edge[e] = c.phi_f_nodes[dof];
					phi_s_edge[e] = c.embedded ? c.phi_s_nodes[dof] : 1.0;
					for (int I = 0; I < 3; I++)
						edge_nodes[e][I] = c.mesh_dof[dof * 3 + I];
				}
				double edge_vec[3];
				double edge_len2 = 0.0;
				for (int I = 0; I < 3; I++)
				{
					edge_vec[I] = edge_nodes[1][I] - edge_nodes[0][I];
					edge_len2 += edge_vec[I] * edge_vec[I];
				}
				double C_a_edge[nP_edge + 1], C_b_edge[nP_edge + 1];
				equivalent_polynomials::calculate_edge_interval_H<nP_edge>(-phi_edge[0], -phi_edge[1], phi_s_edge[0], phi_s_edge[1], C_a_edge);
				equivalent_polynomials::calculate_edge_interval_H<nP_edge>(phi_edge[0], phi_edge[1], phi_s_edge[0], phi_s_edge[1], C_b_edge);
				for (int k = 0; k < nQuadraturePoints_elementBoundary; k++)
				{
					// t of this physical point along side 0's edge direction, then H_hat(t)
					double proj = 0.0;
					for (int I = 0; I < 3; I++)
						proj += (xq0[k][I] - edge_nodes[0][I]) * edge_vec[I];
					const double t_edge = (edge_len2 > 1.0e-24) ? (proj / edge_len2) : 0.0;
					H_e[k] = equivalent_polynomials::evaluate_edge_poly<nP_edge>(C_b_edge, t_edge);
					ImH_e[k] = equivalent_polynomials::evaluate_edge_poly<nP_edge>(C_a_edge, t_edge);
				}
			}
			else
				cutFaceWeights(eN0, ebN_local0, xq0, c.u_l2g, c.mesh_dof, c.phi_f_nodes, H_e, ImH_e);
		}

		// SCIFEM (Ji et al. 2014, eq. 3.7/3.8) at one quadrature point of a face Gamma_f cuts: the two-sided basis of both
		// elements, side 0 first (local DOF s * n + i). jv_*[l] is the jump of basis function l on branch a/b (side 1 enters
		// with the opposite sign), hf_*[l] half its normal flux (mu_branch times the normal derivative, with the sign that
		// turns each side's outward normal into side 0's), and, with the solution, avg_flux_* the average of the two sides'
		// fluxes of the branch field and jump_u* the jump of the field (the solution plus the solution-jump and flux-jump
		// lifts). dS is side 0's weight. The consistency and adjoint terms are unfitted::symmetricConsistency with the face
		// weights of the two branches, the interior penalty unfitted::jumpPenalty.
		struct ScifemPoint
		{
			double jv_a[2 * nDOF_test_element], jv_b[2 * nDOF_test_element], hf_a[2 * nDOF_test_element], hf_b[2 * nDOF_test_element];
			double avg_flux_a = 0.0, avg_flux_b = 0.0, jump_ua = 0.0, jump_ub = 0.0, dS = 0.0;
		};
		inline void scifemPoint(const FaceCtx &c, const int ebN, const int k, const int kmap[2][nQuadraturePoints_elementBoundary],
								const double mua, const double mub, const bool withSolution, ScifemPoint &p)
		{
			double ua_s[2] = {0., 0.}, ub_s[2] = {0., 0.}, flux_a_s[2] = {0., 0.}, flux_b_s[2] = {0., 0.};
			for (int eN_side = 0; eN_side < 2; eN_side++)
			{
				const int eN = c.eb_elements[ebN * 2 + eN_side], ebN_local = c.eb_local[ebN * 2 + eN_side], kb = kmap[eN_side][k],
						  ebN_local_kb = ebN_local * nQuadraturePoints_elementBoundary + kb;
				double jac_int[nSpace * nSpace], jacDet_int, jacInv_int[nSpace * nSpace], boundaryJac[nSpace * (nSpace - 1)],
					metricTensor[(nSpace - 1) * (nSpace - 1)], metricTensorDetSqrt, normal[3], x_int, y_int, z_int;
				ck.calculateMapping_elementBoundary(eN, ebN_local, kb, ebN_local_kb, c.mesh_dof, c.mesh_l2g,
													c.mesh_trial_trace_ref, c.mesh_grad_trial_trace_ref, c.boundaryJac_ref,
													jac_int, jacDet_int, jacInv_int, boundaryJac, metricTensor, metricTensorDetSqrt,
													c.normal_ref, normal, x_int, y_int, z_int);
				// dS and the reference normal n_e are fixed from side 0; side 1's own outward normal is -n_e, which sign_ne
				// converts back.
				if (eN_side == 0)
					p.dS = metricTensorDetSqrt * c.dS_ref[kb];
				const double sign_ne = (eN_side == 0) ? 1.0 : -1.0;
				const double v_sign = sign_ne;
				GfType &gf_f = geom.gfF(eN);
				gf_f.set_boundary_quad(ebN_local_kb);
				double va[nDOF_trial_element], va_grad[nDOF_trial_element * nSpace], vb[nDOF_trial_element], vb_grad[nDOF_trial_element * nSpace];
				for (int i = 0; i < nDOF_test_element; i++)
				{
					va[i] = gf_f.VA(i);
					vb[i] = gf_f.VB(i);
					va_grad[i * nSpace + 0] = gf_f.VA_x(i);
					va_grad[i * nSpace + 1] = gf_f.VA_y(i);
					vb_grad[i * nSpace + 0] = gf_f.VB_x(i);
					vb_grad[i * nSpace + 1] = gf_f.VB_y(i);
					if constexpr (nSpace == 3)
					{
						va_grad[i * nSpace + 2] = gf_f.VA_z(i);
						vb_grad[i * nSpace + 2] = gf_f.VB_z(i);
					}
					double fta = va_grad[i * nSpace + 0] * normal[0] + va_grad[i * nSpace + 1] * normal[1];
					double ftb = vb_grad[i * nSpace + 0] * normal[0] + vb_grad[i * nSpace + 1] * normal[1];
					if constexpr (nSpace == 3)
					{
						fta += va_grad[i * nSpace + 2] * normal[2];
						ftb += vb_grad[i * nSpace + 2] * normal[2];
					}
					const int l = eN_side * nDOF_test_element + i;
					p.jv_a[l] = v_sign * va[i];
					p.jv_b[l] = v_sign * vb[i];
					p.hf_a[l] = 0.5 * (sign_ne * mua * fta);
					p.hf_b[l] = 0.5 * (sign_ne * mub * ftb);
				}
				if (!withSolution)
					continue;
				// Solution-jump lift (branch a is u + JA, branch b is u + JB; same node pattern as assign_jump_side): without it
				// the face terms see u^h instead of u^h + j_h and are inconsistent when [u] != 0.
				double ca_s[nDOF_trial_element], cb_s[nDOF_trial_element];
				for (int i = 0; i < nDOF_trial_element; i++)
				{
					ca_s[i] = c.u_dof[c.u_l2g[eN * nDOF_trial_element + i]];
					cb_s[i] = ca_s[i];
				}
				if (geom.icaseF(eN) == 0)
					for (int i = 0; i < nDOF_trial_element; i++)
					{
						const int dof = c.u_l2g[eN * nDOF_trial_element + i];
						const double jump_i = c.jump_nodes[dof];
						if (c.phi_f_nodes[dof] > 0.0)
							ca_s[i] -= jump_i;
						else
							cb_s[i] += jump_i;
					}
				double ua = 0.0, ub = 0.0, grad_ua[nSpace] = {0., 0.}, grad_ub[nSpace] = {0., 0.};
				ck.valFromElementDOF(ca_s, va, ua);
				ck.gradFromElementDOF(ca_s, va_grad, grad_ua);
				ck.valFromElementDOF(cb_s, vb, ub);
				ck.gradFromElementDOF(cb_s, vb_grad, grad_ub);
				if (geom.icaseF(eN) == 0) // flux-jump lift, as for the solution jump
				{
					const double J = geom.fluxLift(eN);
					ua += J * gf_f.PSIA();
					ub += J * gf_f.PSIB();
					grad_ua[0] += J * gf_f.PSIA_x();
					grad_ua[1] += J * gf_f.PSIA_y();
					grad_ub[0] += J * gf_f.PSIB_x();
					grad_ub[1] += J * gf_f.PSIB_y();
					if constexpr (nSpace == 3)
					{
						grad_ua[2] += J * gf_f.PSIA_z();
						grad_ub[2] += J * gf_f.PSIB_z();
					}
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
			}
			if (withSolution)
			{
				p.avg_flux_a = 0.5 * (flux_a_s[0] + flux_a_s[1]);
				p.avg_flux_b = 0.5 * (flux_b_s[0] + flux_b_s[1]);
				p.jump_ua = ua_s[0] - ua_s[1];
				p.jump_ub = ub_s[0] - ub_s[1];
			}
		}

		// The one-sided edge correction on an element cut by both level sets (2D, P1 and P2). Masked root-adjacent edges
		// leak boundary flux that H_s does not remove (ordinary hat functions vanish there; two-sided branches do not), so it is
		// added back: on the two Gamma_f-cut (root-adjacent) faces of the element, with the face quadrature and the boundary
		// evaluation of the two-branch basis, each branch restricted to its piece of the face (its side of Gamma_f, and the
		// solid side of Gamma_s -- or the fluid side, tcFluidMode) by an edge moment-fit indicator, exact for integrands of
		// degree <= nP_edge (flux * v: degree 1 for P1, 3 for P2). The term is  tcSgn * W_branch * (mu_branch grad u_branch . n)
		// * v_branch  per branch: a one-sided consistency term, unfitted::symmetricConsistency with no adjoint part (s = 0),
		// local DOFs = the element's. With r (and the nodal coefficients of the two branches cuA, cuB, and the flux-jump
		// lift) the residual is added, with J the Jacobian block (linear in u_dof through the flux only; JA/JB are fixed
		// data). r and J are local (nDOF, nDOF * nDOF); either may be null.
		inline void tripleCutEdge(const FaceCtx &c, const int eN, const double *element_phi_f, const double *element_phi_s,
								  const double *element_nodes, const int *elementBoundaries, const bool tcFluidMode,
								  const bool tcBoundaryOnly, const double tcSgn, const double mua, const double mub,
								  const double *JA, const double *JB, const double fluxLiftJ, double *r, double *J)
		{
			constexpr int nD = nDOF_trial_element;
			bool neg[3] = {element_phi_f[0] < 0.0, element_phi_f[1] < 0.0, element_phi_f[2] < 0.0};
			int cnt_neg = (neg[0] ? 1 : 0) + (neg[1] ? 1 : 0) + (neg[2] ? 1 : 0);
			int root_i = -1;
			for (int ii = 0; ii < 3; ii++)
				if ((neg[ii] && cnt_neg == 1) || (!neg[ii] && cnt_neg == 2))
				{
					root_i = ii;
					break;
				}
			if (root_i < 0)
				return;
			GfType &gf_f_tc = geom.gfF(eN);
			// Branch coefficients u_i + JA_i / u_i + JB_i: the solution-jump field (JA/JB) is reconstructed through the same
			// branch basis as everywhere else in this file, so the true branch value is sum (u_i+JA_i) va_i.
			double cuA[nD], cuB[nD];
			if (r)
				for (int ii = 0; ii < nD; ii++)
				{
					const double u_ii = c.u_dof[c.u_l2g[eN * nD + ii]];
					cuA[ii] = u_ii + JA[ii];
					cuB[ii] = u_ii + JB[ii];
				}
			// Default (TC_EDGE): the solid pieces {phi_s<0}. triplyCut_edgeFluidSide: the fluid pieces {phi_s>0}, opposite sign.
			const double sgn_s = tcFluidMode ? 1.0 : -1.0;
			for (int pj = 0; pj < 3; pj++)
			{
				if (pj == root_i)
					continue; // the face opposite the root is not cut by Gamma_f
				const int ebN_tc = elementBoundaries[eN * nDOF_mesh_trial_element + pj];
				if (tcBoundaryOnly && c.eb_elements[ebN_tc * 2 + 1] != -1)
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
				for (int kb = 0; kb < nQuadraturePoints_elementBoundary; kb++)
				{
					const int ebN_local_kb = pj * nQuadraturePoints_elementBoundary + kb;
					double jac_b[nSpace * nSpace], jacDet_b, jacInv_b[nSpace * nSpace],
						bJac_b[nSpace * (nSpace - 1)], mT_b[(nSpace - 1) * (nSpace - 1)], mTDS_b, n_b[3], xb, yb, zb;
					ck.calculateMapping_elementBoundary(eN, pj, kb, ebN_local_kb, c.mesh_dof, c.mesh_l2g,
														c.mesh_trial_trace_ref, c.mesh_grad_trial_trace_ref, c.boundaryJac_ref,
														jac_b, jacDet_b, jacInv_b, bJac_b, mT_b, mTDS_b, c.normal_ref, n_b, xb, yb, zb);
					const double dS = mTDS_b * c.dS_ref[kb];
					const double t = ((xb - ex0) * evx + (yb - ey0) * evy) / elen2;
					const double Wa = equivalent_polynomials::evaluate_edge_poly<nP_edge>(C_a, t);
					const double Wb = equivalent_polynomials::evaluate_edge_poly<nP_edge>(C_b, t);
					gf_f_tc.set_boundary_quad(ebN_local_kb);
					double vA[nD], vB[nD], fA[nD], fB[nD];
					for (int kk = 0; kk < nD; kk++)
					{
						vA[kk] = gf_f_tc.VA(kk);
						vB[kk] = gf_f_tc.VB(kk);
						fA[kk] = mua * (gf_f_tc.VA_x(kk) * n_b[0] + gf_f_tc.VA_y(kk) * n_b[1]);
						fB[kk] = mub * (gf_f_tc.VB_x(kk) * n_b[0] + gf_f_tc.VB_y(kk) * n_b[1]);
					}
					double fluxA = 0.0, fluxB = 0.0;
					if (r)
					{
						for (int kk = 0; kk < nD; kk++)
						{
							fluxA += cuA[kk] * (gf_f_tc.VA_x(kk) * n_b[0] + gf_f_tc.VA_y(kk) * n_b[1]);
							fluxB += cuB[kk] * (gf_f_tc.VB_x(kk) * n_b[0] + gf_f_tc.VB_y(kk) * n_b[1]);
						}
						fluxA += fluxLiftJ * (gf_f_tc.PSIA_x() * n_b[0] + gf_f_tc.PSIA_y() * n_b[1]);
						fluxB += fluxLiftJ * (gf_f_tc.PSIB_x() * n_b[0] + gf_f_tc.PSIB_y() * n_b[1]);
						fluxA *= mua;
						fluxB *= mub;
					}
					unfitted::symmetricConsistency<nD>(-tcSgn * Wa * dS, 0.0, fluxA, 0.0, fA, vA, fA, vA, r, J);
					unfitted::symmetricConsistency<nD>(-tcSgn * Wb * dS, 0.0, fluxB, 0.0, fB, vB, fB, vB, r, J);
				}
			}
		}

		inline void calculateElementResidual(int icase_f,
											 // nonzero when the buried-Gamma_f logic demoted this element:
											 // -1 = fluid all Omega^- (mua), +1 = all Omega^+ (mub)
											 const int demoted_f,
											 // element
											 proteus::pyarray<double> &mesh_trial_ref,
											 proteus::pyarray<double> &mesh_grad_trial_ref,
											 proteus::pyarray<double> &mesh_dof,
											 proteus::pyarray<int> &mesh_l2g,
											 proteus::pyarray<double> &x_ref,
											 proteus::pyarray<double> &dV_ref,
											 proteus::pyarray<double> &u_trial_ref,
											 proteus::pyarray<double> &u_grad_trial_ref,
											 proteus::pyarray<double> &u_test_ref,
											 proteus::pyarray<double> &u_grad_test_ref,
											 proteus::pyarray<double> &elementDiameter,
											 proteus::pyarray<double> &elementBoundaryDiameter,
											 proteus::pyarray<double> &nodeDiametersArray,
											 proteus::pyarray<double> &cfl,
											 double Ct_sge,
											 double sc_uref,
											 double sc_alpha,
											 double useMetrics,
											 double sgeDiffusion,
											 proteus::pyarray<double> &u_hess_trial_ref,
											 proteus::pyarray<double> &q_a_div,
											 // element boundary
											 proteus::pyarray<double> &mesh_trial_trace_ref,
											 proteus::pyarray<double> &mesh_grad_trial_trace_ref,
											 proteus::pyarray<double> &dS_ref,
											 proteus::pyarray<double> &u_trial_trace_ref,
											 proteus::pyarray<double> &u_grad_trial_trace_ref,
											 proteus::pyarray<double> &u_test_trace_ref,
											 proteus::pyarray<double> &u_grad_test_trace_ref,
											 proteus::pyarray<double> &normal_ref,
											 proteus::pyarray<double> &boundaryJac_ref,
											 // physics
											 int nElements_global,
											 int nElementBoundaries_owned,
											 proteus::pyarray<int> &u_l2g,
											 proteus::pyarray<double> &u_dof,
											 proteus::pyarray<int> &sd_rowptr,
											 proteus::pyarray<int> &sd_colind,
											 proteus::pyarray<double> &q_a,
											 proteus::pyarray<double> &q_v,
											 proteus::pyarray<double> &q_r,
											 int lag_shockCapturingDiffusion,
											 double shockCapturingDiffusion,
											 proteus::pyarray<double> &q_numDiff_u,
											 proteus::pyarray<double> &q_numDiff_u_last,
											 int offset_u,
											 int stride_u,
											 proteus::pyarray<double> &elementResidual_u,
											 int nExteriorElementBoundaries_global,
											 proteus::pyarray<int> &exteriorElementBoundariesArray,
											 proteus::pyarray<int> &elementBoundariesArray,
											 proteus::pyarray<int> &elementBoundaryElementsArray,
											 proteus::pyarray<int> &elementBoundaryLocalElementBoundariesArray,
											 proteus::pyarray<double> &element_u,
											 int eN,
											 const bool embeddedBoundary,
											 const double embeddedBoundary_penalty,
											 proteus::pyarray<double> &embeddedBoundary_normal_q,
											 proteus::pyarray<double> &embeddedBoundary_u_q,
											 proteus::pyarray<double> &embeddedBoundary_u_inner_q,
											 proteus::pyarray<double> &embeddedBoundary_u_outer_q,
											 const bool embeddedBoundary_robin,
											 const double embeddedBoundary_robin_h_inner,
											 const double embeddedBoundary_robin_h_outer,
											 proteus::pyarray<double> &embeddedBoundary_robin_uInfinity_inner_q,
											 proteus::pyarray<double> &embeddedBoundary_robin_uInfinity_outer_q,
											 proteus::pyarray<double> &embeddedBoundary_sdf_q,
											 const bool immersedBoundary,
											 const double immersedBoundary_penalty,
											 proteus::pyarray<double> &immersedBoundary_sdf_q,
											 proteus::pyarray<double> &immersedBoundary_normal_q,
											 proteus::pyarray<double> &immersedBoundary_u_q,
											 proteus::pyarray<double> &immersedBoundary_fluxJump_q,
											 proteus::pyarray<double> &immersedBoundary_fluxJumpVector_q,
											 proteus::pyarray<double> &immersedBoundary_solutionJump_nodes,
											 double *element_phi_f,
											 double *JA,
											 double *JB,
											 const double fluxLiftJ,
											 double &L2_error,
											 double &Linfty_error,
											 double &H1_error,
											 double mua,
											 double mub,
											 proteus::pyarray<double> &q_u_exact_inner,
											 proteus::pyarray<double> &q_u_exact_outer,
											 proteus::pyarray<double> &q_grad_u_exact_inner,
											 proteus::pyarray<double> &q_grad_u_exact_outer,
											 const bool PG)
		{
			// the fits of this element's cuts (CutGeometry; a canonical constant fit where the element is uncut)
			GfType &gf_f = geom.gfF(eN);
			GfTypeS &gf_s = geom.gfS(eN);
			for (int i = 0; i < nDOF_test_element; i++)
				elementResidual_u.data()[i] = 0.0;
			// loop over quadrature points and compute integrands
			for (int k = 0; k < nQuadraturePoints_element; k++)
			{
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
				double jac[nSpace * nSpace];
				double jacDet;
				double jacInv[nSpace * nSpace];
				double u_grad_trial[nDOF_trial_element * nSpace];
				double u_hess_trial[nDOF_trial_element * nSpace * nSpace];
				double hess_u[nSpace * nSpace];
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
				ck.gradTrialFromRef(&u_grad_trial_ref.data()[k * nDOF_trial_element * nSpace], jacInv, u_grad_trial);
				// get the solution
				ck.valFromElementDOF(element_u.data(), &u_trial_ref.data()[k * nDOF_trial_element], u);
				// get the solution gradients
				ck.gradFromElementDOF(element_u.data(), u_grad_trial, grad_u);
				ck.hessTrialFromRef(&u_hess_trial_ref.data()[k * nDOF_trial_element * nSpace * nSpace], jacInv, u_hess_trial);
				for (int IJ = 0; IJ < nSpace * nSpace; IJ++)
				{
					hess_u[IJ] = 0.0;
					for (int j = 0; j < nDOF_trial_element; j++)
						hess_u[IJ] += element_u.data()[j] * u_hess_trial[j * nSpace * nSpace + IJ];
				}
				// precalculate test function products with integration weights
				for (int j = 0; j < nDOF_trial_element; j++)
				{
					u_test_dV[j] = u_test_ref.data()[k * nDOF_trial_element + j] * dV;
					for (int I = 0; I < nSpace; I++)
					{
						u_grad_test_dV[j * nSpace + I] = u_grad_trial[j * nSpace + I] * dV; // cek warning won't work for Petrov-Galerkin
						//PGIFEM
					}
				}
				// the two-sided basis of a Gamma_f-cut element and the jump lifts (see unfitted::BranchBasis)
				unfitted::BranchBasis<nSpace, nDOF_trial_element> bb;
				double(&va)[nDOF_trial_element] = bb.va;
				double(&vb)[nDOF_trial_element] = bb.vb;
				double(&va_grad_trial)[nDOF_trial_element * nSpace] = bb.va_grad;
				double(&vb_grad_trial)[nDOF_trial_element * nSpace] = bb.vb_grad;
				if (icase_f == 0)
				{
					gf_f.set_quad(k);
					bb.load(gf_f);
					bb.fieldA(element_u.data(), ua, grad_ua);
					bb.fieldB(element_u.data(), ub, grad_ub);
					bb.lift(JA, JB, fluxLiftJ, uja, grad_uja, ujb, grad_ujb);
					bb.weight(dV, ua_test_dV, ub_test_dV, ua_grad_test_dV, ub_grad_test_dV);
					for (int j = 0; j < nDOF_trial_element; j++)
					{
						for (int I = 0; I < nSpace; I++)
						{
							ua_grad_trial[j * nSpace + I] = va_grad_trial[j * nSpace + I];
							ub_grad_trial[j * nSpace + I] = vb_grad_trial[j * nSpace + I];
						}
					}
					if (PG)
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
				// the weights of the regions at this quadrature point (see unfitted::RegionWeights)
				unfitted::RegionWeights<nSpace, nP_fit> rw;
				rw.evaluate(geom, eN, k, x_ref.data(), curvedSolid);
				const double H_s = rw.H_s, D_s = rw.D_s, D_s_a = rw.D_s_a, D_s_b = rw.D_s_b;
				const double *Dn_s = rw.Dn_s;
				// Test functions without the quadrature weight, for the penalty kernels (Petrov-Galerkin tests with the
				// ordinary hats, the trial side is the branch basis)
				const double *ua_test = PG ? &u_test_ref.data()[k * nDOF_trial_element] : va;
				const double *ub_test = PG ? &u_test_ref.data()[k * nDOF_trial_element] : vb;
				const double *ua_grad_test = PG ? u_grad_trial : va_grad_trial;
				const double *ub_grad_test = PG ? u_grad_trial : vb_grad_trial;
				double *r_loc = elementResidual_u.data();
				// the surface terms are added only where Gamma_s cuts the element (or the whole curved fit is in play)
				const bool solidTerms = embeddedBoundary && (curvedSolid || (gf_s.exact.edge >= 0 && !gf_s.exact.corner));
				if (embeddedBoundary)
				{
					double level_set_normal[nSpace] = {};
					double sign = 0.0;
					double norm_exact = 0.0, norm_cut = 0.0;
					for (int I = 0; I < nSpace && !curvedSolid; I++)
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
					if (solidTerms)
					{
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
								// Deliberately NOT embeddedBoundary_u_inner_q/_outer_q: those are the
								// Dirichlet data on Gamma_s, unrelated to the Robin target.
								robinTerm(dV, D_s_a, embeddedBoundary_robin_h_inner,
										  embeddedBoundary_robin_uInfinity_inner_q.data()[eN_k], ua + uja, ua_test, nullptr, r_loc, nullptr);
								robinTerm(dV, D_s_b, embeddedBoundary_robin_h_outer,
										  embeddedBoundary_robin_uInfinity_outer_q.data()[eN_k], ub + ujb, ub_test, nullptr, r_loc, nullptr);
							}
							else
							{
								// Each side carries its own piece of the solid surface via D_s_a / D_s_b, and tests
								// against the reconstructed basis the bulk uses; ImH_f + H_f = 1, so the total is preserved.
								double ga[nSpace], gb[nSpace], ma[nSpace], mb[nSpace];
								for (int I = 0; I < nSpace; I++)
								{
									ga[I] = grad_ua[I] + grad_uja[I];
									gb[I] = grad_ub[I] + grad_ujb[I];
									ma[I] = -D_s_a * level_set_normal[I];
									mb[I] = -D_s_b * level_set_normal[I];
								}
								nitscheDirichlet(dV, D_s_a, embeddedBoundary_penalty / h_phi, mua, ma,
												 embeddedBoundary_u_inner_q.data()[eN_k], ua + uja, ga,
												 ua_test, ua_grad_test, nullptr, nullptr, r_loc, nullptr);
								nitscheDirichlet(dV, D_s_b, embeddedBoundary_penalty / h_phi, mub, mb,
												 embeddedBoundary_u_outer_q.data()[eN_k], ub + ujb, gb,
												 ub_test, ub_grad_test, nullptr, nullptr, r_loc, nullptr);
							}
						}
						else
						{
							// icase_f != 0: this element is wholly on one side of Gamma_f (icase_f < 0
							// water/mua, icase_f > 0 air/mub -- the same sign convention as demoted_f
							// and mu_dem above), so a single scalar h and a single per-phase target
							// apply throughout -- NOT u_s_dem, which (like embeddedBoundary_u_inner_q
							// above) carries the true Dirichlet solution, not the Robin target.
							const double *u_test = &u_test_ref.data()[k * nDOF_trial_element];
							if (embeddedBoundary_robin)
							{
								const double h_here = (icase_f < 0) ? embeddedBoundary_robin_h_inner
									                                 : embeddedBoundary_robin_h_outer;
								const double u_inf_here = (icase_f < 0)
									? embeddedBoundary_robin_uInfinity_inner_q.data()[eN_k]
									: embeddedBoundary_robin_uInfinity_outer_q.data()[eN_k];
								robinTerm(dV, D_s, h_here, u_inf_here, u, u_test, nullptr, r_loc, nullptr);
							}
							else
							{
								double m[nSpace];
								for (int I = 0; I < nSpace; I++)
									m[I] = curvedSolid ? -Dn_s[I] : -D_s * level_set_normal[I];
								nitscheDirichlet(dV, D_s, embeddedBoundary_penalty / h_phi, a[0], m, u_s_dem, u, grad_u,
												 u_test, u_grad_trial, nullptr, nullptr, r_loc, nullptr);
							}
						}
					}
				}
				const double ImH_f = rw.ImH_f, H_f = rw.H_f, wA = rw.wA, wB = rw.wB, D_f = rw.D_f;
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
					const double jump_flux = fluxJumpValue(immersedBoundary_fluxJump_q.data()[eN_k],
														   &immersedBoundary_fluxJumpVector_q.data()[eN_k_3d], level_set_normal);
					const double D_f_load = (geom.compDFa(eN) != nullptr) ? D_f : D_f * H_s;
					if (icase_f == 0)
					{
						// Flux-jump load int_Gamma_f J_f v: D_f times the IFE test function, NOT weighted by
						// wA/wB. On Gamma_f va = vb = v, and D_f integrates va and vb (polynomials of degree
						// <= nP_fit) exactly; weighting by the Heaviside fits would multiply two fits and
						// move part of the load between branch-a and branch-b nodes.
						double v_load[nDOF_test_element];
						for (int i = 0; i < nDOF_test_element; i++)
						{
							if (rw.branches == 0) // full cut or cut on boundary of negative cell
								v_load[i] = 0.5 * (ua_test[i] + ub_test[i]);
							else if (rw.branches == -1)
								v_load[i] = ua_test[i];
							else
								v_load[i] = ub_test[i];
						}
						unfitted::load<nDOF_test_element>(dV, -(jump_flux * D_f_load), v_load, r_loc);
					}
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
				pdeResidual_u = ck.Advection_strong(df, grad_u) + ck.Reaction_strong(r) +
								 sgeDiffusion * diffusionStrong(sd_rowptr.data(), sd_colind.data(), a, &q_a_div.data()[eN_k * nSpace], hess_u, grad_u); // ck.Mass_strong(m_t) + ck.Advection_strong(df,grad_u) + ck.Reaction_strong(r);
				// calculate adjoint
				for (int i = 0; i < nDOF_test_element; i++)
				{
					// int eN_k_i_nSpace = (eN_k*nDOF_trial_element+i)*nSpace;
					// Lstar_u[i]  = ck.Advection_adjoint(df,&u_grad_test_dV.data()[eN_k_i_nSpace]);
					int i_nSpace = i * nSpace;
					Lstar_u[i] = ck.Advection_adjoint(df, &u_grad_test_dV[i_nSpace]);
				}
				// calculate tau and tau*Res
				calculateSubgridError_tau(elementDiameter.data()[eN], dm_t, df,
										  sgeDiffusion * sgeDiffusiveConstant * diffusionScale(sd_rowptr.data(), sd_colind.data(), a), cfl.data()[eN_k], tau0);
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
					if (icase_f == 0)
					{
						if (rw.branches == 0) // full cut or cut on boundary of negative cell
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
						}
						else if (rw.branches == -1)
						{
							for (int I = 0; I < nSpace; I++) a_loc[I * nSpace + I] = mua;
							elementResidual_u.data()[i] += wA * (ck.Advection_weak(f, &ua_grad_test_dV[i_nSpace]) +
							ck.Diffusion_weak(sd_rowptr.data(), sd_colind.data(), a_loc, grad_ua, &ua_grad_test_dV[i_nSpace]) +
							ck.Diffusion_weak(sd_rowptr.data(), sd_colind.data(), a_loc, grad_uja, &ua_grad_test_dV[i_nSpace]) +
							ck.Reaction_weak(r, ua_test_dV[i]) +
							ck.NumericalDiffusion(q_numDiff_u_last.data()[eN_k], grad_ua, &ua_grad_test_dV[i_nSpace]));
						}
						else if (rw.branches == 1)
						{
							for (int I = 0; I < nSpace; I++) a_loc[I * nSpace + I] = mub;
							elementResidual_u.data()[i] += wB * (ck.Advection_weak(f, &ub_grad_test_dV[i_nSpace]) +
								ck.Diffusion_weak(sd_rowptr.data(), sd_colind.data(), a_loc, grad_ub, &ub_grad_test_dV[i_nSpace]) +
								ck.Diffusion_weak(sd_rowptr.data(), sd_colind.data(), a_loc, grad_ujb, &ub_grad_test_dV[i_nSpace]) +
								ck.Reaction_weak(r, ub_test_dV[i]) +
								ck.NumericalDiffusion(q_numDiff_u_last.data()[eN_k], grad_ub, &ub_grad_test_dV[i_nSpace]));
						}
						else assert(rw.branches != 2 && "Invalid gf_f.exact.edge/corner values. Should be -1, 0 or +1.");
					}
					else
					{

						elementResidual_u.data()[i] += H_s * (ck.Advection_weak(f, &u_grad_test_dV[i_nSpace]) +
															  ck.Diffusion_weak(sd_rowptr.data(), sd_colind.data(), a, grad_u, &u_grad_test_dV[i_nSpace]) +
															  ck.Reaction_weak(r, u_test_dV[i]) +
															  ck.SubgridError(subgridError_u, Lstar_u[i]) +
															  ck.NumericalDiffusion(q_numDiff_u_last.data()[eN_k], grad_u, &u_grad_test_dV[i_nSpace]));
					}
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
				// H1 seminorm part: |grad(u - u_h)|^2 of the phase the point belongs to, with the same
				// weights and the same fluid gate as the L2 error.
				double H1_contrib = 0.0;
				const double *gex_in = &q_grad_u_exact_inner.data()[eN_k * nSpace];
				const double *gex_out = &q_grad_u_exact_outer.data()[eN_k * nSpace];
				if (icase_f == 0)
				{
					double sol_in = q_u_exact_inner.data()[eN_k];
					double err_in = fabs(ua + uja - sol_in);
					L2_contrib += ImH_f * err_in * err_in * dV;
					double sol_out = q_u_exact_outer.data()[eN_k];
					double err_out = fabs(ub + ujb - sol_out);
					L2_contrib += H_f * err_out * err_out * dV;
					for (int I = 0; I < nSpace; I++)
					{
						const double gerr_in = grad_ua[I] + grad_uja[I] - gex_in[I];
						const double gerr_out = grad_ub[I] + grad_ujb[I] - gex_out[I];
						H1_contrib += (ImH_f * gerr_in * gerr_in + H_f * gerr_out * gerr_out) * dV;
					}
					if (phi_s_q >= 0.0)
					{
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
						for (int I = 0; I < nSpace; I++)
							H1_contrib += (grad_u[I] - gex_in[I]) * (grad_u[I] - gex_in[I]) * dV;
						if (phi_s_q >= 0.0)
						{
							Linfty_error = std::max(Linfty_error, err);
						}
					}
					if (icase_f == 1)
					{
						double sol = q_u_exact_outer.data()[eN_k];
						double err = fabs(u - sol);
						L2_contrib += err * err * dV;
						for (int I = 0; I < nSpace; I++)
							H1_contrib += (grad_u[I] - gex_out[I]) * (grad_u[I] - gex_out[I]) * dV;
						if (phi_s_q >= 0.0)
						{
							Linfty_error = std::max(Linfty_error, err);
						}
					}
				}
				if (phi_s_q >= 0.0)
				{
					L2_error += L2_contrib;
					H1_error += H1_contrib;
				}
			}
		}

		// Every cut-cell path below (the two-sided IFEM basis, the cut-face terms: Nitsche and ghost
		// penalty, SCIFEM, the triply-cut edge correction, and the composite triply-cut fit) is written
		// for triangles. assert() is compiled out (-DNDEBUG), so in 3D they would not stop, they would
		// compute garbage. Refuse before any of it runs. The phase numbers are the C44 plan's.
		static void requireSupportedDimension(bool embedded, bool robin, bool immersed, double scifem_switch, double scifem_penalty, bool PG, double ghost_penalty, bool curved = false)
		{
			// The curved solid (a P2 level set, C44 phase 1b) is 3D P2 tetrahedra with the embedded solid alone.
			// Curved cuts in 2D (where p = 2 uses straight cuts) are not part of phase 1b (decision P7).
			if (curved)
			{
				if (!embedded)
					throw std::runtime_error("ADR: embeddedBoundary_curved needs embeddedBoundary");
				if (nSpace != 3)
					throw std::runtime_error("ADR: embeddedBoundary_curved is implemented on tetrahedra only (curved cuts in 2D are not part of C44 phase 1b)");
				if (pOrder != 2)
					throw std::runtime_error("ADR: embeddedBoundary_curved needs a P2 space (the level set is the P2 interpolant; P1 cuts are planar)");
				if (immersed || robin)
					throw std::runtime_error("ADR: embeddedBoundary_curved is implemented for the embedded solid alone (not with immersedBoundary or embeddedBoundary_robin)");
			}
			if (nSpace == 2)
				return;
			std::string what;
			int phase = 0;
			auto need = [&](const char *name, int ph)
			{
				if (!what.empty())
					what += "; ";
				what += name;
				phase = std::max(phase, ph);
			};
			// C44 phase 1: the Nitsche terms and the ghost penalty of an embedded boundary on P1 and P2
			// tetrahedra (planar cut: the vertex values place the plane; P2 needs the degree-7 rule, qOrder 7).
			(void)ghost_penalty;
			if (robin)
				need("embeddedBoundary_robin (Robin condition on the cut solid surface)", 1);
			// C44 phase 2, P1: the two-sided IFEM basis and the flux-jump lift on P1 tetrahedra
			if (immersed && pOrder != 1)
				need("immersedBoundary with P2 (the two-sided P2 basis on tetrahedra)", 2);
			// SCIFEM on triangular faces of P1 tetrahedra is implemented (cutFaceWeights); P2 is refused above
			(void)scifem_switch;
			(void)scifem_penalty;
			if (PG)
				need("PG (Petrov-Galerkin test with the IFEM basis)", 2);
			if (embedded && immersed)
				need("embeddedBoundary together with immersedBoundary (triply-cut elements: composite fit, edge correction)", 3);
			if (!what.empty())
				throw std::runtime_error("ADR: not implemented in " + std::to_string(nSpace) +
										 "D (the cut-cell machinery is 2D only): " + what + ". Planned for C44 phase " +
										 std::to_string(phase) + ". Plain ADR (no embedded or immersed boundary) runs in 3D.");
		}
		void calculateResidual(arguments_dict &args)
		{
			proteus::pyarray<double> &mesh_trial_ref = args.array<double>("mesh_trial_ref");
			proteus::pyarray<double> &mesh_grad_trial_ref = args.array<double>("mesh_grad_trial_ref");
			proteus::pyarray<double> &mesh_dof = args.array<double>("mesh_dof");
			proteus::pyarray<int> &mesh_l2g = args.array<int>("mesh_l2g");
			proteus::pyarray<double> &dV_ref = args.array<double>("dV_ref");
			proteus::pyarray<double> &u_trial_ref = args.array<double>("u_trial_ref");
			proteus::pyarray<double> &u_grad_trial_ref = args.array<double>("u_grad_trial_ref");
			proteus::pyarray<double> &u_test_ref = args.array<double>("u_test_ref");
			proteus::pyarray<double> &u_grad_test_ref = args.array<double>("u_grad_test_ref");
			proteus::pyarray<double> &elementDiameter = args.array<double>("elementDiameter");
			proteus::pyarray<double> &cfl = args.array<double>("cfl");
			double Ct_sge = args.scalar<double>("Ct_sge");
			double sc_uref = args.scalar<double>("sc_uref");
			double sc_alpha = args.scalar<double>("sc_alpha");
			double useMetrics = args.scalar<double>("useMetrics");
			double sgeDiffusion = args.scalar<double>("sgeDiffusion");
			proteus::pyarray<double> &u_hess_trial_ref = args.array<double>("u_hess_trial_ref");
			proteus::pyarray<double> &q_a_div = args.array<double>("q_a_div");
			proteus::pyarray<double> &mesh_trial_trace_ref = args.array<double>("mesh_trial_trace_ref");
			proteus::pyarray<double> &mesh_grad_trial_trace_ref = args.array<double>("mesh_grad_trial_trace_ref");
			proteus::pyarray<double> &dS_ref = args.array<double>("dS_ref");
			proteus::pyarray<double> &u_trial_trace_ref = args.array<double>("u_trial_trace_ref");
			proteus::pyarray<double> &u_grad_trial_trace_ref = args.array<double>("u_grad_trial_trace_ref");
			proteus::pyarray<double> &u_test_trace_ref = args.array<double>("u_test_trace_ref");
			proteus::pyarray<double> &u_grad_test_trace_ref = args.array<double>("u_grad_test_trace_ref");
			proteus::pyarray<double> &normal_ref = args.array<double>("normal_ref");
			proteus::pyarray<double> &boundaryJac_ref = args.array<double>("boundaryJac_ref");
			int nElements_global = args.scalar<int>("nElements_global");
			proteus::pyarray<int> &u_l2g = args.array<int>("u_l2g");
			proteus::pyarray<double> &u_dof = args.array<double>("u_dof");
			proteus::pyarray<int> &sd_rowptr = args.array<int>("sd_rowptr");
			proteus::pyarray<int> &sd_colind = args.array<int>("sd_colind");
			proteus::pyarray<double> &q_a = args.array<double>("q_a");
			proteus::pyarray<double> &q_v = args.array<double>("q_v");
			proteus::pyarray<double> &q_r = args.array<double>("q_r");
			int lag_shockCapturing = args.scalar<int>("lag_shockCapturing");
			double shockCapturingDiffusion = args.scalar<double>("shockCapturingDiffusion");
			proteus::pyarray<double> &q_numDiff_u = args.array<double>("q_numDiff_u");
			proteus::pyarray<double> &q_numDiff_u_last = args.array<double>("q_numDiff_u_last");
			int offset_u = args.scalar<int>("offset_u");
			int stride_u = args.scalar<int>("stride_u");
			proteus::pyarray<double> &globalResidual = args.array<double>("globalResidual");
			int nExteriorElementBoundaries_global = args.scalar<int>("nExteriorElementBoundaries_global");
			proteus::pyarray<int> &exteriorElementBoundariesArray = args.array<int>("exteriorElementBoundariesArray");
			proteus::pyarray<int> &elementBoundaryElementsArray = args.array<int>("elementBoundaryElementsArray");
			proteus::pyarray<int> &elementBoundaryLocalElementBoundariesArray = args.array<int>("elementBoundaryLocalElementBoundariesArray");
			proteus::pyarray<double> &ebqe_a = args.array<double>("ebqe_a");
			proteus::pyarray<double> &ebqe_v = args.array<double>("ebqe_v");
			proteus::pyarray<int> &isDOFBoundary_u = args.array<int>("isDOFBoundary_u");
			proteus::pyarray<double> &ebqe_bc_u_ext = args.array<double>("ebqe_bc_u_ext");
			proteus::pyarray<int> &isDiffusiveFluxBoundary_u = args.array<int>("isDiffusiveFluxBoundary_u");
			proteus::pyarray<int> &isAdvectiveFluxBoundary_u = args.array<int>("isAdvectiveFluxBoundary_u");
			proteus::pyarray<double> &ebqe_bc_flux_u_ext = args.array<double>("ebqe_bc_flux_u_ext");
			proteus::pyarray<double> &ebqe_bc_advectiveFlux_u_ext = args.array<double>("ebqe_bc_advectiveFlux_u_ext");
			proteus::pyarray<double> &ebqe_penalty_ext = args.array<double>("ebqe_penalty_ext");
			const bool embeddedBoundary = args.scalar<int>("embeddedBoundary");
			const double embeddedBoundary_penalty = args.scalar<double>("embeddedBoundary_penalty");
			const double embeddedBoundary_ghost_penalty = args.scalar<double>("embeddedBoundary_ghost_penalty");
			proteus::pyarray<double> &embeddedBoundary_sdf_nodes = args.array<double>("embeddedBoundary_sdf_nodes");
			proteus::pyarray<double> &embeddedBoundary_sdf_q = args.array<double>("embeddedBoundary_sdf_q");
			proteus::pyarray<double> &embeddedBoundary_normal_q = args.array<double>("embeddedBoundary_normal_q");
			proteus::pyarray<double> &embeddedBoundary_u_q = args.array<double>("embeddedBoundary_u_q");
			proteus::pyarray<double> &embeddedBoundary_u_inner_q = args.array<double>("embeddedBoundary_u_inner_q");
			proteus::pyarray<double> &embeddedBoundary_u_outer_q = args.array<double>("embeddedBoundary_u_outer_q");
			const bool embeddedBoundary_robin = args.scalar<int>("embeddedBoundary_robin");
			const double embeddedBoundary_robin_h_inner = args.scalar<double>("embeddedBoundary_robin_h_inner");
			const double embeddedBoundary_robin_h_outer = args.scalar<double>("embeddedBoundary_robin_h_outer");
			proteus::pyarray<double> &embeddedBoundary_robin_uInfinity_inner_q = args.array<double>("embeddedBoundary_robin_uInfinity_inner_q");
			proteus::pyarray<double> &embeddedBoundary_robin_uInfinity_outer_q = args.array<double>("embeddedBoundary_robin_uInfinity_outer_q");
			const bool immersedBoundary = args.scalar<int>("immersedBoundary");
			const double immersedBoundary_penalty = args.scalar<double>("immersedBoundary_penalty");
			const double immersedSCIFEM_switch = args.scalar<double>("immersedSCIFEM_switch");
			const double immersedSCIFEM_penalty = args.scalar<double>("immersedSCIFEM_penalty");
			const bool PG = args.scalar<int>("PG");
			const bool embeddedBoundary_curved = args.scalar<int>("embeddedBoundary_curved");
			proteus::pyarray<int> &embeddedBoundary_cut_label = args.array<int>("embeddedBoundary_cut_label");
			requireSupportedDimension(embeddedBoundary, embeddedBoundary_robin, immersedBoundary, immersedSCIFEM_switch, immersedSCIFEM_penalty, PG, embeddedBoundary_ghost_penalty, embeddedBoundary_curved);
			curvedSolid = embeddedBoundary_curved;
			proteus::pyarray<double> &immersedBoundary_sdf_nodes = args.array<double>("immersedBoundary_sdf_nodes");
			proteus::pyarray<double> &immersedBoundary_sdf_q = args.array<double>("immersedBoundary_sdf_q");
			proteus::pyarray<double> &immersedBoundary_normal_q = args.array<double>("immersedBoundary_normal_q");
			proteus::pyarray<double> &immersedBoundary_u_q = args.array<double>("immersedBoundary_u_q");
			proteus::pyarray<double> &immersedBoundary_fluxJump_q = args.array<double>("immersedBoundary_fluxJump_q");
			proteus::pyarray<double> &immersedBoundary_fluxJumpVector_q = args.array<double>("immersedBoundary_fluxJumpVector_q");
			proteus::pyarray<double> &immersedBoundary_fluxJump_element = args.array<double>("immersedBoundary_fluxJump_element");
			proteus::pyarray<double> &immersedBoundary_solutionJump_nodes = args.array<double>("immersedBoundary_solutionJump_nodes");
			proteus::pyarray<double> &isActiveDOF = args.array<double>("isActiveDOF");
			const double eb_adjoint_sigma = args.scalar<double>("eb_adjoint_sigma");
			proteus::pyarray<double> &x_ref = args.array<double>("x_ref");
			proteus::pyarray<double> &xB_ref = args.array<double>("xB_ref");
			proteus::pyarray<int> &elementBoundariesArray = args.array<int>("elementBoundariesArray");
			const int nElementBoundaries_owned = args.scalar<int>("nElementBoundaries_owned");
			// Owned elements are numbered first on each rank; the L2 error sums them only (the
			// overlap elements are some other rank's owned ones and are summed there).
			const int nElements_owned = args.scalar<int>("nElements_owned");
			proteus::pyarray<double> &elementBoundaryDiameter = args.array<double>("elementBoundaryDiameter");
			proteus::pyarray<double> &nodeDiametersArray = args.array<double>("nodeDiametersArray");
			proteus::pyarray<double> &L2_error = args.array<double>("L2_error");
			proteus::pyarray<double> &Linfty_error = args.array<double>("Linfty_error");
			const double mua = args.scalar<double>("mua");
			const double mub = args.scalar<double>("mub");
			const double jf = args.scalar<double>("jf");
			proteus::pyarray<double> &q_u_exact_inner = args.array<double>("q_u_exact_inner");
			proteus::pyarray<double> &q_u_exact_outer = args.array<double>("q_u_exact_outer");
			proteus::pyarray<double> &q_grad_u_exact_inner = args.array<double>("q_grad_u_exact_inner");
			proteus::pyarray<double> &q_grad_u_exact_outer = args.array<double>("q_grad_u_exact_outer");
			proteus::pyarray<double> &H1_error = args.array<double>("H1_error");
			prepareGeometry(args, curvedSolid);
			// Coefficients.triplyCut_edgeFluidSide: take the fluid-side one-sided edge correction on
			// triply-cut cells (flips the sign of the correction in residual and Jacobian).
			const bool tcFluidMode = args.scalar<int>("triplyCut_edgeFluidSide");
			// Coefficients.triplyCut_edgeBoundaryOnly (use with SCIFEM on): keep the one-sided fluid-side
			// correction only on root-adjacent edges that lie on the exterior boundary. Interior ones are then
			// handled by SCIFEM's two-sided consistency term, which has no counterpart on an exterior edge.
			const bool tcBoundaryOnly = args.scalar<int>("triplyCut_edgeBoundaryOnly");
			const double tcSgn = tcFluidMode ? -1.0 : 1.0;
			const FaceCtx tcCtx = makeFaceCtx(args);

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
			for (int eN = 0; eN < nElements_global; eN++)
			{
				// declare local storage for element residual and initialize
				// double elementResidual_u[nDOF_test_element],element_u[nDOF_trial_element];
				auto elementResidual_u = proteus::pyarray<double>::from_shape({nDOF_test_element});
				auto element_u = proteus::pyarray<double>::from_shape({nDOF_trial_element});
				for (int i = 0; i < nDOF_trial_element; i++)
				{
					int eN_i = eN * nDOF_trial_element + i;
					element_u.data()[i] = u_dof.data()[u_l2g.data()[eN_i]];
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
				double element_nodes[nDOF_trial_element * 3];
				for (int i = 0; i < nDOF_trial_element; i++)
				{
					int eN_i = eN * nDOF_trial_element + i;
					for (int I = 0; I < 3; I++)
						element_nodes[i * 3 + I] = mesh_dof.data()[u_l2g.data()[eN_i] * 3 + I];
				} // i
				// how the two level sets cut this element, from the geometry (built once, see prepareGeometry)
				const int icase_s = geom.icaseS(eN);
				const int icase_f = geom.icaseF(eN);
				// the buried-Gamma_f logic demoted this element: -1 = fluid all Omega^- (mua), +1 = all Omega^+ (mub), 0 = not
				const int demoted_f = geom.demoted(eN);
				double JA[nDOF_trial_element];
				double JB[nDOF_trial_element];
				std::fill(JA, JA + nDOF_trial_element, 0.0);
				std::fill(JB, JB + nDOF_trial_element, 0.0);
				if (icase_f == 0)
				{
					// Leveque & Li 1994, Example 1, 2, 3, 4; Ji et. al. 2014
					// The jump function is, in general, not constant over the cut element, so it
					// must be evaluated at each node's own coordinates (not once at the cut
					// barycenter) -- otherwise JA/JB are constant over the element and their
					// interpolated gradients (grad_uja, grad_ujb) vanish identically.
					for (int i = 0; i < nDOF_trial_element; i++)
					{
						const double jump_i = immersedBoundary_solutionJump_nodes.data()[u_l2g.data()[eN * nDOF_trial_element + i]];
						if (element_phi_f[i] > 0.0)
						{
							JA[i] = -jump_i;
							JB[i] = 0.0;
						}
						else if (element_phi_f[i] <= 0.0)
						{
							JA[i] = 0.0;
							JB[i] = jump_i;
						}
					}
				}
				// Prescribed flux jump of this cell for the flux-jump lift (0 if the lift is off), as built by the geometry
				double fluxLiftJ = (icase_f == 0) ? geom.fluxLift(eN) : 0.0;
				double L2_elem = 0.0, H1_elem = 0.0;
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
										 sgeDiffusion,
										 u_hess_trial_ref,
										 q_a_div,
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
										 JA,
										 JB,
										 fluxLiftJ,
										 L2_elem,
										 Linfty_error.data()[0],
										 H1_elem,
												 mua,
											 mub,
											 q_u_exact_inner,
											 q_u_exact_outer,
											 q_grad_u_exact_inner,
											 q_grad_u_exact_outer,
											 PG);
				if (eN < nElements_owned)
				{
					L2_error.data()[0] += L2_elem;
					H1_error.data()[0] += H1_elem;
				}
				//
				// load element into global residual and save element residual
				//
				for (int i = 0; i < nDOF_test_element; i++)
				{
					int eN_i = eN * nDOF_test_element + i;
					globalResidual.data()[offset_u + stride_u * u_l2g.data()[eN_i]] += elementResidual_u.data()[i];
					if (geom.active(eN))
						isActiveDOF.data()[offset_u + stride_u * u_l2g.data()[eN_i]] = 1.0;
				} // i
				// Masked root-adjacent edges leak boundary flux H_s doesn't remove (ordinary
				// hat functions vanish there; two-sided branches don't). Add it back for all 3
				// local dofs.
				// icase_s == -1 means this element has NO fluid anywhere (H_s == 0 identically) --
				// Gamma_f may still cross it geometrically, but with zero real fluid area there is
				// no bulk contribution to correct and no physically-meaningful va/vb reconstruction
				// to add back. Without this check TC_EDGE still fires there (icase_f==0 alone
				// doesn't rule it out), injecting a real, nonzero correction from a "dead" element
				// into whichever physical dofs it shares with genuinely active neighbors.
				if (embeddedBoundary && immersedBoundary && (nSpace == 2 && (pOrder == 1 || pOrder == 2)) && icase_f == 0
					&& icase_s != -1)
				{
					double r_tc[nDOF_trial_element] = {};
					tripleCutEdge(tcCtx, eN, element_phi_f, element_phi_s, element_nodes, elementBoundariesArray.data(), tcFluidMode,
								  tcBoundaryOnly, tcSgn, mua, mub, JA, JB, fluxLiftJ, r_tc, nullptr);
					for (int jj = 0; jj < nDOF_trial_element; jj++)
						globalResidual.data()[offset_u + stride_u * u_l2g.data()[eN * nDOF_trial_element + jj]] += r_tc[jj];
				}
			} // elements
			// Ghost penalty on the interior faces of Gamma_s-cut elements (the faces the geometry lists: both elements carry
			// fluid): gamma_g h_F int_F [[du/dn]] [[dv/dn]], one jump penalty per phase (see ghostPoint) over the local DOFs
			// of both elements.
			{
				const FaceCtx fctx = makeFaceCtx(args);
				constexpr int n2 = 2 * nDOF_test_element;
				for (const int ebN : geom.cutfemFaces())
				{
					const double gamma_h = embeddedBoundary_ghost_penalty * elementBoundaryDiameter.data()[ebN];
					int gp_kmap[2][nQuadraturePoints_elementBoundary];
					double gp_xq[2][nQuadraturePoints_elementBoundary][3];
					pairFacePoints(fctx, ebN, gp_kmap, gp_xq, elementBoundaryDiameter.data()[ebN], false);
					for (int kb = 0; kb < nQuadraturePoints_elementBoundary; kb++)
					{
						GhostPoint gp;
						ghostPoint(fctx, ebN, kb, gp_kmap[1][kb], true, gp);
						double r2[n2] = {};
						if (gp.okA || gp.okB)
						{
							if (gp.okA)
								unfitted::jumpPenalty<n2>(gp.dS, gamma_h, gp.ju_a, gp.jv_a, r2, nullptr);
							if (gp.okB)
								unfitted::jumpPenalty<n2>(gp.dS, gamma_h, gp.ju_b, gp.jv_b, r2, nullptr);
						}
						else
							unfitted::jumpPenalty<n2>(gp.dS, gamma_h, gp.ju_c, gp.jv_c, r2, nullptr);
						for (int side = 0; side < 2; side++)
						{
							const int eN = elementBoundaryElementsArray.data()[ebN * 2 + side];
							for (int i = 0; i < nDOF_test_element; i++)
								globalResidual.data()[offset_u + stride_u * u_l2g.data()[eN * nDOF_test_element + i]] += r2[side * nDOF_test_element + i];
						}
					}
				}
			} // cutfem element boundaries
			// SCIFEM (Ji et al. 2014, eq. 3.7/3.8) symmetric consistency + adjoint-consistency terms
			// on cut edges: -switch*({{beta grad u}}.n_e*[[v]] + {{beta grad v}}.n_e*[[u]]).
			// A genuine 1D facet term -- the edge carries its own arc-length measure -- so no Dirac
			// surrogate is involved. The two region-wise integrands are formed separately and blended
			// by the edge Heaviside fit (see the prologue below), which is what makes this exact for
			// P1 and P2 without splitting the quadrature.
			{
				const FaceCtx fctx = makeFaceCtx(args);
				constexpr int n2 = 2 * nDOF_test_element;
				for (const int ebN : geom.ifemFaces())
				{
					// edge diameter for the interior-penalty scaling (same measure the ghost penalty uses); note gamma/h here
					// for a VALUE jump, vs gamma*h there for a gradient jump
					const double h_edge = elementBoundaryDiameter.data()[ebN];
					int kmap[2][nQuadraturePoints_elementBoundary];
					double xq[2][nQuadraturePoints_elementBoundary][3];
					pairFacePoints(fctx, ebN, kmap, xq, h_edge, true);
					double H_e[nQuadraturePoints_elementBoundary], ImH_e[nQuadraturePoints_elementBoundary];
					faceCutWeights(fctx, ebN, xq[0], ImH_e, H_e);
					for (int k = 0; k < nQuadraturePoints_elementBoundary; k++)
					{
						ScifemPoint sp;
						scifemPoint(fctx, ebN, k, kmap, mua, mub, true, sp);
						double r2[n2] = {};
						// Blend the region-wise PRODUCTS, never the individual factors:
						// (ImH*A_f + H*B_f)*(ImH*A_v + H*B_v) != ImH*A_f*A_v + H*B_f*B_v.
						unfitted::symmetricConsistency<n2>(immersedSCIFEM_switch * ImH_e[k] * sp.dS, 1.0, sp.avg_flux_a, sp.jump_ua, sp.hf_a, sp.jv_a, r2, nullptr);
						unfitted::symmetricConsistency<n2>(immersedSCIFEM_switch * H_e[k] * sp.dS, 1.0, sp.avg_flux_b, sp.jump_ub, sp.hf_b, sp.jv_b, r2, nullptr);
						// Interior-penalty stabilization gamma/h int_e [[u]][[v]] ds. eq (3.7) is consistency + adjoint only, so it
						// carries no coercivity guarantee; this is the standard remedy. Added with a PLUS sign (like the bulk
						// stiffness) so it is positive semi-definite, and blended by the same edge Heaviside so it stays exact
						// for P1 (integrand degree <= 2) and P2 (degree <= 4 = nP_edge). Weighted by mu of each branch (like
						// the Nitsche term), so the penalty is of the size of the bulk term on both sides of a high-contrast
						// interface.
						unfitted::jumpPenalty<n2>(ImH_e[k] * sp.dS, (immersedSCIFEM_penalty / h_edge) * mua, sp.jump_ua, sp.jv_a, r2, nullptr);
						unfitted::jumpPenalty<n2>(H_e[k] * sp.dS, (immersedSCIFEM_penalty / h_edge) * mub, sp.jump_ub, sp.jv_b, r2, nullptr);
						for (int side = 0; side < 2; side++)
						{
							const int eN = elementBoundaryElementsArray.data()[ebN * 2 + side];
							for (int i = 0; i < nDOF_test_element; i++)
								globalResidual.data()[offset_u + stride_u * u_l2g.data()[eN * nDOF_test_element + i]] += r2[side * nDOF_test_element + i];
						}
					} // k
				}
			} // ifem element boundaries
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
		}

		inline void calculateElementJacobian(int icase_f,
											 const int demoted_f,
											 // element
											 proteus::pyarray<double> &mesh_trial_ref,
											 proteus::pyarray<double> &mesh_grad_trial_ref,
											 proteus::pyarray<double> &mesh_dof,
											 proteus::pyarray<int> &mesh_l2g,
											 proteus::pyarray<double> &x_ref,
											 proteus::pyarray<double> &dV_ref,
											 proteus::pyarray<double> &u_trial_ref,
											 proteus::pyarray<double> &u_grad_trial_ref,
											 proteus::pyarray<double> &u_test_ref,
											 proteus::pyarray<double> &u_grad_test_ref,
											 proteus::pyarray<double> &elementDiameter,
											 proteus::pyarray<double> &elementBoundaryDiameter,
											 proteus::pyarray<double> &nodeDiametersArray,
											 proteus::pyarray<double> &cfl,
											 double Ct_sge,
											 double sc_uref,
											 double sc_alpha,
											 double useMetrics,
											 double sgeDiffusion,
											 proteus::pyarray<double> &u_hess_trial_ref,
											 proteus::pyarray<double> &q_a_div,
											 // element boundary
											 proteus::pyarray<double> &mesh_trial_trace_ref,
											 proteus::pyarray<double> &mesh_grad_trial_trace_ref,
											 proteus::pyarray<double> &dS_ref,
											 proteus::pyarray<double> &u_trial_trace_ref,
											 proteus::pyarray<double> &u_grad_trial_trace_ref,
											 proteus::pyarray<double> &u_test_trace_ref,
											 proteus::pyarray<double> &u_grad_test_trace_ref,
											 proteus::pyarray<double> &normal_ref,
											 proteus::pyarray<double> &boundaryJac_ref,
											 // physics
											 int nElements_global,
											 int nElementBoundaries_owned,
											 proteus::pyarray<int> &u_l2g,
											 proteus::pyarray<double> &u_dof,
											 proteus::pyarray<int> &sd_rowptr,
											 proteus::pyarray<int> &sd_colind,
											 proteus::pyarray<double> &q_a,
											 proteus::pyarray<double> &q_v,
											 proteus::pyarray<double> &q_r,
											 int lag_shockCapturing,
											 double shockCapturingDiffusion,
											 proteus::pyarray<double> &q_numDiff_u,
											 proteus::pyarray<double> &q_numDiff_u_last,
											 proteus::pyarray<double> &elementJacobian_u_u,
											 proteus::pyarray<double> &element_u,
											 int eN,
											 const bool embeddedBoundary,
											 const double embeddedBoundary_penalty,
											 proteus::pyarray<double> &embeddedBoundary_normal_q,
											 proteus::pyarray<double> &embeddedBoundary_u_q,
											 proteus::pyarray<double> &embeddedBoundary_u_inner_q,
											 proteus::pyarray<double> &embeddedBoundary_u_outer_q,
											 const bool embeddedBoundary_robin,
											 const double embeddedBoundary_robin_h_inner,
											 const double embeddedBoundary_robin_h_outer,
											 proteus::pyarray<double> &embeddedBoundary_robin_uInfinity_inner_q,
											 proteus::pyarray<double> &embeddedBoundary_robin_uInfinity_outer_q,
											 const bool immersedBoundary,
											 const double immersedBoundary_penalty,
											 proteus::pyarray<double> &immersedBoundary_sdf_q,
											 proteus::pyarray<double> &immersedBoundary_normal_q,
											 proteus::pyarray<double> &immersedBoundary_u_q,
											 proteus::pyarray<double> &immersedBoundary_fluxJump_q,
											 proteus::pyarray<double> &immersedBoundary_fluxJumpVector_q,
											 proteus::pyarray<double> &immersedBoundary_solutionJump_nodes,
											 double *element_phi_f,
											 double mua,
											 double mub,
											 const bool PG)
		{
			// the fits of this element's cuts (CutGeometry; a canonical constant fit where the element is uncut)
			GfType &gf_f = geom.gfF(eN);
			GfTypeS &gf_s = geom.gfS(eN);
			for (int i = 0; i < nDOF_test_element; i++)
				for (int j = 0; j < nDOF_trial_element; j++)
				{
					elementJacobian_u_u.data()[i * nDOF_trial_element + j] = 0.0;
				}
			for (int k = 0; k < nQuadraturePoints_element; k++)
			{
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
					   f[nSpace], df[nSpace],
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
					   u_hess_trial[nDOF_trial_element * nSpace * nSpace],
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
				ck.gradTrialFromRef(&u_grad_trial_ref.data()[k * nDOF_trial_element * nSpace], jacInv, u_grad_trial);
				// get the solution
				ck.valFromElementDOF(element_u.data(), &u_trial_ref.data()[k * nDOF_trial_element], u);
				// get the solution gradients
				ck.gradFromElementDOF(element_u.data(), u_grad_trial, grad_u);
				ck.hessTrialFromRef(&u_hess_trial_ref.data()[k * nDOF_trial_element * nSpace * nSpace], jacInv, u_hess_trial);
				// precalculate test function products with integration weights
				for (int j = 0; j < nDOF_trial_element; j++)
				{
					u_test_dV[j] = u_test_ref.data()[k * nDOF_trial_element + j] * dV;
					for (int I = 0; I < nSpace; I++)
					{
						u_grad_test_dV[j * nSpace + I] = u_grad_trial[j * nSpace + I] * dV; // cek warning won't work for Petrov-Galerkin
						//PGIFEM
					}
				}
				unfitted::BranchBasis<nSpace, nDOF_trial_element> bb;
				double(&va)[nDOF_trial_element] = bb.va;
				double(&vb)[nDOF_trial_element] = bb.vb;
				double(&va_grad_trial)[nDOF_trial_element * nSpace] = bb.va_grad;
				double(&vb_grad_trial)[nDOF_trial_element * nSpace] = bb.vb_grad;
				if (icase_f == 0)
				{
					gf_f.set_quad(k);
					bb.load(gf_f);
					bb.fieldA(element_u.data(), ua, grad_ua);
					bb.fieldB(element_u.data(), ub, grad_ub);
					bb.weight(dV, ua_test_dV, ub_test_dV, ua_grad_test_dV, ub_grad_test_dV);
					for (int j = 0; j < nDOF_trial_element; j++)
					{
						ua_trial[j] = va[j];
						ub_trial[j] = vb[j];
						for (int I = 0; I < nSpace; I++)
						{
							ua_grad_trial[j * nSpace + I] = va_grad_trial[j * nSpace + I];
							ub_grad_trial[j * nSpace + I] = vb_grad_trial[j * nSpace + I];
						}
					}
					if (PG)
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
				// the weights of the regions at this quadrature point (see unfitted::RegionWeights)
				unfitted::RegionWeights<nSpace, nP_fit> rw;
				rw.evaluate(geom, eN, k, x_ref.data(), curvedSolid);
				const double H_s = rw.H_s, D_s = rw.D_s, D_s_a = rw.D_s_a, D_s_b = rw.D_s_b;
				const double *Dn_s = rw.Dn_s;
				// Test functions without the quadrature weight (Petrov-Galerkin tests with the ordinary hats), and the
				// trial functions, for the penalty kernels
				const double *ua_test = PG ? &u_test_ref.data()[k * nDOF_trial_element] : va;
				const double *ub_test = PG ? &u_test_ref.data()[k * nDOF_trial_element] : vb;
				const double *ua_grad_test = PG ? u_grad_trial : va_grad_trial;
				const double *ub_grad_test = PG ? u_grad_trial : vb_grad_trial;
				double *J_loc = elementJacobian_u_u.data();
				const bool solidTerms = embeddedBoundary && (curvedSolid || (gf_s.exact.edge >= 0 && !gf_s.exact.corner));
				if (embeddedBoundary)
				{
					double level_set_normal[nSpace] = {};
					double sign = 0.0;
					double norm_exact = 0.0, norm_cut = 0.0;
					for (int I = 0; I < nSpace && !curvedSolid; I++)
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
					if (solidTerms)
					{
						if (icase_f == 0)
						{
							// The solution here is ua on the mua side and ub on the mub side; the plain `u` is
							// neither. The prescribed jump is a constant offset with zero derivative, so it is
							// residual-only and does not appear here.
							if (embeddedBoundary_robin)
							{
								robinTerm(dV, D_s_a, embeddedBoundary_robin_h_inner,
										  embeddedBoundary_robin_uInfinity_inner_q.data()[eN_k], ua, ua_test, ua_trial, nullptr, J_loc);
								robinTerm(dV, D_s_b, embeddedBoundary_robin_h_outer,
										  embeddedBoundary_robin_uInfinity_outer_q.data()[eN_k], ub, ub_test, ub_trial, nullptr, J_loc);
							}
							else
							{
								double ma[nSpace], mb[nSpace];
								for (int I = 0; I < nSpace; I++)
								{
									ma[I] = -D_s_a * level_set_normal[I];
									mb[I] = -D_s_b * level_set_normal[I];
								}
								nitscheDirichlet(dV, D_s_a, embeddedBoundary_penalty / h_phi, mua, ma,
												 embeddedBoundary_u_inner_q.data()[eN_k], ua, grad_ua,
												 ua_test, ua_grad_test, ua_trial, ua_grad_trial, nullptr, J_loc);
								nitscheDirichlet(dV, D_s_b, embeddedBoundary_penalty / h_phi, mub, mb,
												 embeddedBoundary_u_outer_q.data()[eN_k], ub, grad_ub,
												 ub_test, ub_grad_test, ub_trial, ub_grad_trial, nullptr, J_loc);
							}
						}
						else
						{
							const double *u_test = &u_test_ref.data()[k * nDOF_trial_element];
							const double *u_trial = &u_trial_ref.data()[k * nDOF_trial_element];
							if (embeddedBoundary_robin)
							{
								const double h_here = (icase_f < 0) ? embeddedBoundary_robin_h_inner
									                                 : embeddedBoundary_robin_h_outer;
								const double u_inf_here = (icase_f < 0)
									? embeddedBoundary_robin_uInfinity_inner_q.data()[eN_k]
									: embeddedBoundary_robin_uInfinity_outer_q.data()[eN_k];
								robinTerm(dV, D_s, h_here, u_inf_here, u, u_test, u_trial, nullptr, J_loc);
							}
							else
							{
								double m[nSpace];
								for (int I = 0; I < nSpace; I++)
									m[I] = curvedSolid ? -Dn_s[I] : -D_s * level_set_normal[I];
								nitscheDirichlet(dV, D_s, embeddedBoundary_penalty / h_phi, a[0], m, u_s_dem, u, grad_u,
												 u_test, u_grad_trial, u_trial, u_grad_trial, nullptr, J_loc);
							}
						}
					}
				}
				const double ImH_f = rw.ImH_f, H_f = rw.H_f, wA = rw.wA, wB = rw.wB;
				(void)ImH_f;
				(void)H_f;
				// immersedBoundary: the prescribed flux-jump load is data (u-independent), so it is residual-only
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
										  ck.AdvectionJacobian_strong(df, &u_grad_trial[j_nSpace]) +
										  sgeDiffusion * diffusionStrong(sd_rowptr.data(), sd_colind.data(), a, &q_a_div.data()[eN_k * nSpace], &u_hess_trial[j * nSpace * nSpace], &u_grad_trial[j_nSpace]);
				}
				// tau and tau*Res
				calculateSubgridError_tau(elementDiameter.data()[eN],
										  dm_t,
										  df,
										  sgeDiffusion * sgeDiffusiveConstant * diffusionScale(sd_rowptr.data(), sd_colind.data(), a),
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
							if (rw.branches == 0)
						    {
								for (int I = 0; I < nSpace; I++) a_loc[I * nSpace + I] = mua;
								elementJacobian_u_u.data()[i * nDOF_trial_element + j] += wA * (ck.AdvectionJacobian_weak(df, ua_trial[j], &ua_grad_test_dV[i_nSpace]) +
								ck.SimpleDiffusionJacobian_weak(sd_rowptr.data(), sd_colind.data(), a_loc, &ua_grad_trial[j_nSpace], &ua_grad_test_dV[i_nSpace]) +
								ck.ReactionJacobian_weak(dr, ua_trial[j], ua_test_dV[i]) +
								ck.NumericalDiffusionJacobian(q_numDiff_u_last.data()[eN_k], &ua_grad_trial[j_nSpace], &ua_grad_test_dV[i_nSpace]));

								for (int I = 0; I < nSpace; I++) a_loc[I * nSpace + I] = mub;
								elementJacobian_u_u.data()[i * nDOF_trial_element + j] += wB * (ck.AdvectionJacobian_weak(df, ub_trial[j], &ub_grad_test_dV[i_nSpace]) +
									ck.SimpleDiffusionJacobian_weak(sd_rowptr.data(), sd_colind.data(), a_loc, &ub_grad_trial[j_nSpace], &ub_grad_test_dV[i_nSpace]) +
									ck.ReactionJacobian_weak(dr, ub_trial[j], ub_test_dV[i]) +
									ck.NumericalDiffusionJacobian(q_numDiff_u_last.data()[eN_k], &ub_grad_trial[j_nSpace], &ub_grad_test_dV[i_nSpace]));
							}
							else if (rw.branches == -1)
							{
								for (int I = 0; I < nSpace; I++) a_loc[I * nSpace + I] = mua;
								elementJacobian_u_u.data()[i * nDOF_trial_element + j] += wA * (ck.AdvectionJacobian_weak(df, ua_trial[j], &ua_grad_test_dV[i_nSpace]) +
								ck.SimpleDiffusionJacobian_weak(sd_rowptr.data(), sd_colind.data(), a_loc, &ua_grad_trial[j_nSpace], &ua_grad_test_dV[i_nSpace]) +
								ck.ReactionJacobian_weak(dr, ua_trial[j], ua_test_dV[i]) +
								ck.NumericalDiffusionJacobian(q_numDiff_u_last.data()[eN_k], &ua_grad_trial[j_nSpace], &ua_grad_test_dV[i_nSpace]));
							}
							else if (rw.branches == 1)
							{
								for (int I = 0; I < nSpace; I++) a_loc[I * nSpace + I] = mub;
								elementJacobian_u_u.data()[i * nDOF_trial_element + j] += wB * (ck.AdvectionJacobian_weak(df, ub_trial[j], &ub_grad_test_dV[i_nSpace]) +
									ck.SimpleDiffusionJacobian_weak(sd_rowptr.data(), sd_colind.data(), a_loc, &ub_grad_trial[j_nSpace], &ub_grad_test_dV[i_nSpace]) +
									ck.ReactionJacobian_weak(dr, ub_trial[j], ub_test_dV[i]) +
									ck.NumericalDiffusionJacobian(q_numDiff_u_last.data()[eN_k], &ub_grad_trial[j_nSpace], &ub_grad_test_dV[i_nSpace]));
							}
							else assert(rw.branches != 2 && "Invalid gf_f.exact.edge/corner values. Should be -1, 0 or +1.");
						}
						else
						{
							elementJacobian_u_u.data()[i * nDOF_trial_element + j] += H_s * (ck.AdvectionJacobian_weak(df, u_trial_ref.data()[k * nDOF_trial_element + j], &u_grad_test_dV[i_nSpace]) +
																							 ck.SimpleDiffusionJacobian_weak(sd_rowptr.data(), sd_colind.data(), a, &u_grad_trial[j_nSpace], &u_grad_test_dV[i_nSpace]) +
																							 ck.ReactionJacobian_weak(dr, u_trial_ref.data()[k * nDOF_trial_element + j], u_test_dV[i]) +
																							 ck.SubgridErrorJacobian(dsubgridError_u_u[j], Lstar_u[i]) +
																							 ck.NumericalDiffusionJacobian(q_numDiff_u_last.data()[eN_k], &u_grad_trial[j_nSpace], &u_grad_test_dV[i_nSpace]));
						}
					} // j
				} // i
			} // k
		}

		void calculateJacobian(arguments_dict &args)
		{
			proteus::pyarray<double> &mesh_trial_ref = args.array<double>("mesh_trial_ref");
			proteus::pyarray<double> &mesh_grad_trial_ref = args.array<double>("mesh_grad_trial_ref");
			proteus::pyarray<double> &mesh_dof = args.array<double>("mesh_dof");
			proteus::pyarray<int> &mesh_l2g = args.array<int>("mesh_l2g");
			proteus::pyarray<double> &dV_ref = args.array<double>("dV_ref");
			proteus::pyarray<double> &u_trial_ref = args.array<double>("u_trial_ref");
			proteus::pyarray<double> &u_grad_trial_ref = args.array<double>("u_grad_trial_ref");
			proteus::pyarray<double> &u_test_ref = args.array<double>("u_test_ref");
			proteus::pyarray<double> &u_grad_test_ref = args.array<double>("u_grad_test_ref");
			proteus::pyarray<double> &elementDiameter = args.array<double>("elementDiameter");
			proteus::pyarray<double> &cfl = args.array<double>("cfl");
			double Ct_sge = args.scalar<double>("Ct_sge");
			double sc_uref = args.scalar<double>("sc_uref");
			double sc_alpha = args.scalar<double>("sc_alpha");
			double useMetrics = args.scalar<double>("useMetrics");
			double sgeDiffusion = args.scalar<double>("sgeDiffusion");
			proteus::pyarray<double> &u_hess_trial_ref = args.array<double>("u_hess_trial_ref");
			proteus::pyarray<double> &q_a_div = args.array<double>("q_a_div");
			proteus::pyarray<double> &mesh_trial_trace_ref = args.array<double>("mesh_trial_trace_ref");
			proteus::pyarray<double> &mesh_grad_trial_trace_ref = args.array<double>("mesh_grad_trial_trace_ref");
			proteus::pyarray<double> &dS_ref = args.array<double>("dS_ref");
			proteus::pyarray<double> &u_trial_trace_ref = args.array<double>("u_trial_trace_ref");
			proteus::pyarray<double> &u_grad_trial_trace_ref = args.array<double>("u_grad_trial_trace_ref");
			proteus::pyarray<double> &u_test_trace_ref = args.array<double>("u_test_trace_ref");
			proteus::pyarray<double> &u_grad_test_trace_ref = args.array<double>("u_grad_test_trace_ref");
			proteus::pyarray<double> &normal_ref = args.array<double>("normal_ref");
			proteus::pyarray<double> &boundaryJac_ref = args.array<double>("boundaryJac_ref");
			int nElements_global = args.scalar<int>("nElements_global");
			proteus::pyarray<int> &u_l2g = args.array<int>("u_l2g");
			proteus::pyarray<double> &u_dof = args.array<double>("u_dof");
			proteus::pyarray<int> &sd_rowptr = args.array<int>("sd_rowptr");
			proteus::pyarray<int> &sd_colind = args.array<int>("sd_colind");
			proteus::pyarray<double> &q_a = args.array<double>("q_a");
			proteus::pyarray<double> &q_v = args.array<double>("q_v");
			proteus::pyarray<double> &q_r = args.array<double>("q_r");
			int lag_shockCapturing = args.scalar<int>("lag_shockCapturing");
			double shockCapturingDiffusion = args.scalar<double>("shockCapturingDiffusion");
			proteus::pyarray<double> &q_numDiff_u = args.array<double>("q_numDiff_u");
			proteus::pyarray<double> &q_numDiff_u_last = args.array<double>("q_numDiff_u_last");
			proteus::pyarray<int> &csrRowIndeces_u_u = args.array<int>("csrRowIndeces_u_u");
			proteus::pyarray<int> &csrColumnOffsets_u_u = args.array<int>("csrColumnOffsets_u_u");
			proteus::pyarray<double> &globalJacobian = args.array<double>("globalJacobian");
			int nExteriorElementBoundaries_global = args.scalar<int>("nExteriorElementBoundaries_global");
			proteus::pyarray<int> &exteriorElementBoundariesArray = args.array<int>("exteriorElementBoundariesArray");
			proteus::pyarray<int> &elementBoundaryElementsArray = args.array<int>("elementBoundaryElementsArray");
			proteus::pyarray<int> &elementBoundaryLocalElementBoundariesArray = args.array<int>("elementBoundaryLocalElementBoundariesArray");
			proteus::pyarray<double> &ebqe_a = args.array<double>("ebqe_a");
			proteus::pyarray<double> &ebqe_v = args.array<double>("ebqe_v");
			proteus::pyarray<int> &isDOFBoundary_u = args.array<int>("isDOFBoundary_u");
			proteus::pyarray<double> &ebqe_bc_u_ext = args.array<double>("ebqe_bc_u_ext");
			proteus::pyarray<int> &isDiffusiveFluxBoundary_u = args.array<int>("isDiffusiveFluxBoundary_u");
			proteus::pyarray<int> &isAdvectiveFluxBoundary_u = args.array<int>("isAdvectiveFluxBoundary_u");
			proteus::pyarray<double> &ebqe_bc_flux_u_ext = args.array<double>("ebqe_bc_flux_u_ext");
			proteus::pyarray<double> &ebqe_bc_advectiveFlux_u_ext = args.array<double>("ebqe_bc_advectiveFlux_u_ext");
			proteus::pyarray<int> &csrColumnOffsets_eb_u_u = args.array<int>("csrColumnOffsets_eb_u_u");
			proteus::pyarray<double> &ebqe_penalty_ext = args.array<double>("ebqe_penalty_ext");
			const bool embeddedBoundary = args.scalar<int>("embeddedBoundary");
			const double embeddedBoundary_penalty = args.scalar<double>("embeddedBoundary_penalty");
			const double embeddedBoundary_ghost_penalty = args.scalar<double>("embeddedBoundary_ghost_penalty");
			proteus::pyarray<double> &embeddedBoundary_sdf_nodes = args.array<double>("embeddedBoundary_sdf_nodes");
			proteus::pyarray<double> &embeddedBoundary_sdf_q = args.array<double>("embeddedBoundary_sdf_q");
			proteus::pyarray<double> &embeddedBoundary_normal_q = args.array<double>("embeddedBoundary_normal_q");
			proteus::pyarray<double> &embeddedBoundary_u_q = args.array<double>("embeddedBoundary_u_q");
			proteus::pyarray<double> &embeddedBoundary_u_inner_q = args.array<double>("embeddedBoundary_u_inner_q");
			proteus::pyarray<double> &embeddedBoundary_u_outer_q = args.array<double>("embeddedBoundary_u_outer_q");
			const bool embeddedBoundary_robin = args.scalar<int>("embeddedBoundary_robin");
			const double embeddedBoundary_robin_h_inner = args.scalar<double>("embeddedBoundary_robin_h_inner");
			const double embeddedBoundary_robin_h_outer = args.scalar<double>("embeddedBoundary_robin_h_outer");
			proteus::pyarray<double> &embeddedBoundary_robin_uInfinity_inner_q = args.array<double>("embeddedBoundary_robin_uInfinity_inner_q");
			proteus::pyarray<double> &embeddedBoundary_robin_uInfinity_outer_q = args.array<double>("embeddedBoundary_robin_uInfinity_outer_q");
			const bool immersedBoundary = args.scalar<int>("immersedBoundary");
			const double immersedBoundary_penalty = args.scalar<double>("immersedBoundary_penalty");
			const double immersedSCIFEM_switch = args.scalar<double>("immersedSCIFEM_switch");
			const double immersedSCIFEM_penalty = args.scalar<double>("immersedSCIFEM_penalty");
			const bool PG = args.scalar<int>("PG");
			const bool embeddedBoundary_curved = args.scalar<int>("embeddedBoundary_curved");
			proteus::pyarray<int> &embeddedBoundary_cut_label = args.array<int>("embeddedBoundary_cut_label");
			requireSupportedDimension(embeddedBoundary, embeddedBoundary_robin, immersedBoundary, immersedSCIFEM_switch, immersedSCIFEM_penalty, PG, embeddedBoundary_ghost_penalty, embeddedBoundary_curved);
			curvedSolid = embeddedBoundary_curved;
			proteus::pyarray<double> &immersedBoundary_sdf_nodes = args.array<double>("immersedBoundary_sdf_nodes");
			proteus::pyarray<double> &immersedBoundary_sdf_q = args.array<double>("immersedBoundary_sdf_q");
			proteus::pyarray<double> &immersedBoundary_normal_q = args.array<double>("immersedBoundary_normal_q");
			proteus::pyarray<double> &immersedBoundary_u_q = args.array<double>("immersedBoundary_u_q");
			proteus::pyarray<double> &immersedBoundary_fluxJump_q = args.array<double>("immersedBoundary_fluxJump_q");
			proteus::pyarray<double> &immersedBoundary_fluxJumpVector_q = args.array<double>("immersedBoundary_fluxJumpVector_q");
			proteus::pyarray<double> &immersedBoundary_fluxJump_element = args.array<double>("immersedBoundary_fluxJump_element");
			proteus::pyarray<double> &immersedBoundary_solutionJump_nodes = args.array<double>("immersedBoundary_solutionJump_nodes");
			proteus::pyarray<double> &isActiveDOF = args.array<double>("isActiveDOF");
			const double eb_adjoint_sigma = args.scalar<double>("eb_adjoint_sigma");
			proteus::pyarray<double> &x_ref = args.array<double>("x_ref");
			proteus::pyarray<double> &xB_ref = args.array<double>("xB_ref");
			proteus::pyarray<int> &elementBoundariesArray = args.array<int>("elementBoundariesArray");
			const int nElementBoundaries_owned = args.scalar<int>("nElementBoundaries_owned");
			proteus::pyarray<double> &elementBoundaryDiameter = args.array<double>("elementBoundaryDiameter");
			proteus::pyarray<double> &nodeDiametersArray = args.array<double>("nodeDiametersArray");
			const double mua = args.scalar<double>("mua");
			const double mub = args.scalar<double>("mub");
			const double jf = args.scalar<double>("jf");
			prepareGeometry(args, curvedSolid);
			//
			// loop over elements to compute volume integrals and load them into the element Jacobians and global Jacobian
			//
			// Coefficients.triplyCut_edgeFluidSide: take the fluid-side one-sided edge correction on
			// triply-cut cells (flips the sign of the correction in residual and Jacobian).
			const bool tcFluidMode = args.scalar<int>("triplyCut_edgeFluidSide");
			// Coefficients.triplyCut_edgeBoundaryOnly (use with SCIFEM on): keep the one-sided fluid-side
			// correction only on root-adjacent edges that lie on the exterior boundary. Interior ones are then
			// handled by SCIFEM's two-sided consistency term, which has no counterpart on an exterior edge.
			const bool tcBoundaryOnly = args.scalar<int>("triplyCut_edgeBoundaryOnly");
			const double tcSgn = tcFluidMode ? -1.0 : 1.0;
			const FaceCtx tcCtx = makeFaceCtx(args);
			for (int eN = 0; eN < nElements_global; eN++)
			{
				// double  elementJacobian_u_u.data()[nDOF_test_element*nDOF_trial_element],element_u[nDOF_trial_element];
				auto elementJacobian_u_u = proteus::pyarray<double>::from_shape({nDOF_test_element * nDOF_trial_element});
				auto element_u = proteus::pyarray<double>::from_shape({nDOF_trial_element});
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
						element_nodes[i * 3 + I] = mesh_dof.data()[u_l2g.data()[eN_i] * 3 + I];
				} // i
				// how the two level sets cut this element, from the geometry (built once, see prepareGeometry)
				const int icase_s = geom.icaseS(eN);
				const int icase_f = geom.icaseF(eN);
				const int demoted_f = geom.demoted(eN);
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
										 sgeDiffusion,
										 u_hess_trial_ref,
										 q_a_div,
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
											 PG);
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
					} // j
				} // i
				// Jacobian of the correction above: exact 3x3 block, linear in u_dof through
				// the flux only. Same gate as the residual side, including the icase_s != -1 check.
				if (embeddedBoundary && immersedBoundary && (nSpace == 2 && (pOrder == 1 || pOrder == 2)) && icase_f == 0
					&& icase_s != -1)
				{
					double J_tc[nDOF_trial_element * nDOF_trial_element] = {};
					tripleCutEdge(tcCtx, eN, element_phi_f, element_phi_s, element_nodes, elementBoundariesArray.data(), tcFluidMode,
								  tcBoundaryOnly, tcSgn, mua, mub, nullptr, nullptr, 0.0, nullptr, J_tc);
					for (int jj = 0; jj < nDOF_test_element; jj++)
					{
						const int eN_jj = eN * nDOF_test_element + jj;
						for (int kk = 0; kk < nDOF_trial_element; kk++)
						{
							const int eN_jj_kk = eN_jj * nDOF_trial_element + kk;
							globalJacobian.data()[csrRowIndeces_u_u.data()[eN_jj] + csrColumnOffsets_u_u.data()[eN_jj_kk]] += J_tc[jj * nDOF_trial_element + kk];
						}
					}
				}
			} // elements
			// Jacobian of the ghost penalty assembled in calculateResidual: the same jump penalties, over the local DOFs of both
			// elements, scattered by the face's CSR offsets (everything is linear in u_dof)
			{
				const FaceCtx fctx = makeFaceCtx(args);
				constexpr int n2 = 2 * nDOF_test_element;
				for (const int ebN : geom.cutfemFaces())
				{
					const double gamma_h = embeddedBoundary_ghost_penalty * elementBoundaryDiameter.data()[ebN];
					int gp_kmap[2][nQuadraturePoints_elementBoundary];
					double gp_xq[2][nQuadraturePoints_elementBoundary][3];
					pairFacePoints(fctx, ebN, gp_kmap, gp_xq, elementBoundaryDiameter.data()[ebN], false);
					for (int kb = 0; kb < nQuadraturePoints_elementBoundary; kb++)
					{
						GhostPoint gp;
						ghostPoint(fctx, ebN, kb, gp_kmap[1][kb], false, gp);
						double J2[n2 * n2] = {};
						if (gp.okA || gp.okB)
						{
							if (gp.okA)
								unfitted::jumpPenalty<n2>(gp.dS, gamma_h, 0.0, gp.jv_a, nullptr, J2);
							if (gp.okB)
								unfitted::jumpPenalty<n2>(gp.dS, gamma_h, 0.0, gp.jv_b, nullptr, J2);
						}
						else
							unfitted::jumpPenalty<n2>(gp.dS, gamma_h, 0.0, gp.jv_c, nullptr, J2);
						for (int eN_side = 0; eN_side < 2; eN_side++)
						{
							const int eN = elementBoundaryElementsArray.data()[ebN * 2 + eN_side];
							for (int i = 0; i < nDOF_test_element; i++)
							{
								const int eN_i = eN * nDOF_test_element + i;
								for (int eN_side2 = 0; eN_side2 < 2; eN_side2++)
									for (int j = 0; j < nDOF_test_element; j++)
									{
										const int ebN_i_j = ebN * 4 * nDOF_test_X_trial_element +
															eN_side * 2 * nDOF_test_X_trial_element +
															eN_side2 * nDOF_test_X_trial_element +
															i * nDOF_trial_element + j;
										globalJacobian.data()[csrRowIndeces_u_u.data()[eN_i] + csrColumnOffsets_eb_u_u.data()[ebN_i_j]] +=
											J2[(eN_side * nDOF_test_element + i) * n2 + eN_side2 * nDOF_test_element + j];
									}
							}
						}
					}
				}
			} // cutfem element boundaries
			// Jacobian of the SCIFEM (eq. 3.7) consistency terms assembled in calculateResidual --
			// same construction (orientation map + edge Heaviside blend of region-wise products),
			// differentiated w.r.t. each side's u_dof. Everything is linear in u_dof, so
			// d(avg_flux_a)/d(u_dof[s,j]) = 0.5*fta_m[s][j] and d(jump_ua)/d(u_dof[s,j]) =
			// (+/-)va_m[s][j] -- the same per-dof quantities used for the test index i.
			{
				const FaceCtx fctx = makeFaceCtx(args);
				constexpr int n2 = 2 * nDOF_test_element;
				for (const int ebN : geom.ifemFaces())
				{
					const double h_edge = elementBoundaryDiameter.data()[ebN];
					int kmap[2][nQuadraturePoints_elementBoundary];
					double xq[2][nQuadraturePoints_elementBoundary][3];
					pairFacePoints(fctx, ebN, kmap, xq, h_edge, true);
					double H_e[nQuadraturePoints_elementBoundary], ImH_e[nQuadraturePoints_elementBoundary];
					faceCutWeights(fctx, ebN, xq[0], ImH_e, H_e);
					for (int k = 0; k < nQuadraturePoints_elementBoundary; k++)
					{
						ScifemPoint sp;
						scifemPoint(fctx, ebN, k, kmap, mua, mub, false, sp);
						double J2[n2 * n2] = {};
						// the same kernels with the residual's weights; everything is linear in u_dof, so
						// d(avg_flux)/d(u_dof[s,j]) = hf[s,j] and d(jump_u)/d(u_dof[s,j]) = jv[s,j]
						unfitted::symmetricConsistency<n2>(immersedSCIFEM_switch * ImH_e[k] * sp.dS, 1.0, 0.0, 0.0, sp.hf_a, sp.jv_a, sp.hf_a, sp.jv_a, nullptr, J2);
						unfitted::symmetricConsistency<n2>(immersedSCIFEM_switch * H_e[k] * sp.dS, 1.0, 0.0, 0.0, sp.hf_b, sp.jv_b, sp.hf_b, sp.jv_b, nullptr, J2);
						unfitted::jumpPenalty<n2>(ImH_e[k] * sp.dS, (immersedSCIFEM_penalty / h_edge) * mua, 0.0, sp.jv_a, nullptr, J2);
						unfitted::jumpPenalty<n2>(H_e[k] * sp.dS, (immersedSCIFEM_penalty / h_edge) * mub, 0.0, sp.jv_b, nullptr, J2);
						for (int eN_side = 0; eN_side < 2; eN_side++)
						{
							const int eN = elementBoundaryElementsArray.data()[ebN * 2 + eN_side];
							for (int i = 0; i < nDOF_test_element; i++)
							{
								const int eN_i = eN * nDOF_test_element + i;
								for (int eN_side2 = 0; eN_side2 < 2; eN_side2++)
									for (int j = 0; j < nDOF_test_element; j++)
									{
										const int ebN_i_j = ebN * 4 * nDOF_test_X_trial_element +
															eN_side * 2 * nDOF_test_X_trial_element +
															eN_side2 * nDOF_test_X_trial_element +
															i * nDOF_trial_element + j;
										globalJacobian.data()[csrRowIndeces_u_u.data()[eN_i] + csrColumnOffsets_eb_u_u.data()[ebN_i_j]] +=
											J2[(eN_side * nDOF_test_element + i) * n2 + eN_side2 * nDOF_test_element + j];
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
		// 3D P2 on tetrahedra with the degree-5 (15-point) rule: ADR-only for the same reason, the
		// shared ModelFactory has no P2 tetrahedron entry (commented out there).
		if (nSpaceIn == 3 && CompKernelFlag == 0 && nDOF_mesh_trial_elementIn == 4 && nDOF_trial_elementIn == 10 &&
			nDOF_test_elementIn == 10 && nQuadraturePoints_elementIn == 15 && nQuadraturePoints_elementBoundaryIn == 7)
			return static_cast<cADR_base *>(new cADR<CompKernel<3, 4, 10, 10>, 3, 15, 4, 10, 10, 7>());
		// 3D P2 with the degree-7 rule (31 points on the tetrahedron, 16 on a face; the tetrahedron rule has some
		// negative weights): the rule that integrates the fit (degree 3) times the degree-4 Nitsche integrands
		// exactly, qOrder 7 (C44 phase 1b).
		if (nSpaceIn == 3 && CompKernelFlag == 0 && nDOF_mesh_trial_elementIn == 4 && nDOF_trial_elementIn == 10 &&
			nDOF_test_elementIn == 10 && nQuadraturePoints_elementIn == 31 && nQuadraturePoints_elementBoundaryIn == 16)
			return static_cast<cADR_base *>(new cADR<CompKernel<3, 4, 10, 10>, 3, 31, 4, 10, 10, 16>());
		// 3D P2 with the degree-8 rule (qOrder 8: 45 tetrahedron points, 16 face points): fit degree 4 (C44 phase 1b)
		if (nSpaceIn == 3 && CompKernelFlag == 0 && nDOF_mesh_trial_elementIn == 4 && nDOF_trial_elementIn == 10 &&
			nDOF_test_elementIn == 10 && nQuadraturePoints_elementIn == 45 && nQuadraturePoints_elementBoundaryIn == 16)
			return static_cast<cADR_base *>(new cADR<CompKernel<3, 4, 10, 10>, 3, 45, 4, 10, 10, 16>());
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