#ifndef PROTEUS_UNFITTED_KERNELS_H
#define PROTEUS_UNFITTED_KERNELS_H
// Portable penalty kernels for unfitted (cut-cell) finite elements.
//
// Every penalty in an unfitted method has one form: a coefficient times a jump of the solution times a jump of the
// test function, integrated against a weight. These kernels add that form at ONE quadrature point to a local
// residual vector and a local row-major Jacobian block. They know nothing about meshes, cut cells, level sets,
// pyarray or any proteus type: plain data in, local contributions out, the caller owns the local storage and the
// global scatter, and the physics (the jumps, the averaged fluxes, the weight w = quadrature weight times a
// moment-fitted Dirac or Heaviside) stays with the caller.
//
// Contract: only <cmath>; no allocation, exceptions, virtuals or I/O; templated on the local DOF count n.
// PROTEUS_UNFITTED_HD (default `inline`) marks the kernels for a device build (a Kokkos build defines it as
// KOKKOS_INLINE_FUNCTION, as experimental/adrh/portable.h does for ADRH_FN).
//
// Either output may be a null pointer (a residual call passes J = nullptr, a Jacobian call r = nullptr).
// The test and trial functions may differ (Petrov-Galerkin): the overloads taking both arrays are the general
// form, the others set trial = test.
//
//   jumpPenalty          r_i += c w [u] [v_i]               J_ij += c w [v_i] [v_j]
//                        ghost penalty ([du/dn], c = gamma h), SCIFEM interior penalty ([u], c = gamma/h),
//                        Nitsche penalty on a Dirichlet surface (u - g, c = gamma a/h), Robin (u - u_inf, c = h_R)
//   symmetricConsistency r_i -= w ({sigma(u)} [v_i] + s {sigma(v_i)} [u])    and its Jacobian
//                        SCIFEM consistency and adjoint terms; Nitsche consistency and adjoint (one-sided)
//   load                 r_i -= w g v_i                     prescribed flux jump, Robin data
#include <cmath>

#ifndef PROTEUS_UNFITTED_HD
#define PROTEUS_UNFITTED_HD inline
#endif

namespace proteus
{
	namespace unfitted
	{
		// r[i] += c*w*ju*jv_test[i];   J[i*n+j] += c*w*jv_test[i]*jv_trial[j]
		template <int n>
		PROTEUS_UNFITTED_HD void jumpPenalty(double w, double c, double ju, const double *jv_test, const double *jv_trial,
											 double *r, double *J)
		{
			const double cw = c * w;
			if (r)
				for (int i = 0; i < n; i++)
					r[i] += cw * ju * jv_test[i];
			if (J)
				for (int i = 0; i < n; i++)
					for (int j = 0; j < n; j++)
						J[i * n + j] += cw * jv_test[i] * jv_trial[j];
		}
		template <int n>
		PROTEUS_UNFITTED_HD void jumpPenalty(double w, double c, double ju, const double *jv, double *r, double *J)
		{
			jumpPenalty<n>(w, c, ju, jv, jv, r, J);
		}

		// r[i] -= w*(avg_flux_u*jv_test[i] + s*avg_flux_v_test[i]*ju)
		// J[i*n+j] -= w*(avg_flux_v_trial[j]*jv_test[i] + s*avg_flux_v_test[i]*jv_trial[j])
		// s = +1 symmetric, -1 skew-symmetric
		template <int n>
		PROTEUS_UNFITTED_HD void symmetricConsistency(double w, double s, double avg_flux_u, double ju,
													  const double *avg_flux_v_test, const double *jv_test,
													  const double *avg_flux_v_trial, const double *jv_trial,
													  double *r, double *J)
		{
			if (r)
				for (int i = 0; i < n; i++)
					r[i] -= w * (avg_flux_u * jv_test[i] + s * avg_flux_v_test[i] * ju);
			if (J)
				for (int i = 0; i < n; i++)
					for (int j = 0; j < n; j++)
						J[i * n + j] -= w * (avg_flux_v_trial[j] * jv_test[i] + s * avg_flux_v_test[i] * jv_trial[j]);
		}
		template <int n>
		PROTEUS_UNFITTED_HD void symmetricConsistency(double w, double s, double avg_flux_u, double ju,
													  const double *avg_flux_v, const double *jv, double *r, double *J)
		{
			symmetricConsistency<n>(w, s, avg_flux_u, ju, avg_flux_v, jv, avg_flux_v, jv, r, J);
		}

		// r[i] -= w*g*v[i]
		template <int n>
		PROTEUS_UNFITTED_HD void load(double w, double g, const double *v, double *r)
		{
			for (int i = 0; i < n; i++)
				r[i] -= w * g * v[i];
		}
	} // namespace unfitted
} // namespace proteus
#endif
