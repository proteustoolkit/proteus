// The fourth two-phase slice of the engine: proteus's MCorr model (mprans/MCorr.h calculateResidual/calculateJacobian with
// useExact = 0 and useExact_s = 0: the damBreakLevelSet configuration) on the generic element kernel (engine.h), not on the
// SUPG template of VOF/NCLS/RDLS. MCorr is a steady nonlinear reaction-diffusion problem for the level-set correction u:
//
//   R_i = sum_q H_s [ r(u) w_i + eps_diff grad u . grad w_i ] dV,     J_ij = sum_q H_s [ dr w_j w_i + eps_diff grad w_j . grad w_i ] dV
//   r = porosity (H_epsH(phi + u) - H_vof),  dr = porosity delta_epsD(phi + u),  H_s = H_0(phi_s)
//
// with phi NCLS's level set and H_vof VOF's solution at the point, phi_s the solid's signed distance (H_0: the sharp
// Heaviside, 1 in the fluid), eps_X = epsFact_X (useMetrics h_phi + (1 - useMetrics) h_e) and h_phi the node diameters
// interpolated to the point. In proteus's form (adrh.h): no mass, advection or Hamiltonian; phi = u, a = H_s eps_diff I,
// r and dr scaled by H_s. The Jacobian is exact.
//
// The point data comes in through the functor: bindElement(eN) selects the element's block of the point fields and
// bindPoint(q) the point (engine.h), so the generic kernel runs unchanged. What MCorr hands to other models at the points
// (q_u, q_r without H_s, and grad u / (|grad u| + 1e-8) as q_n) comes from mcorrPointOutputs.
//
// The active-DOF marking: with useExact_s = 0 every element is active (GeneralizedFunctions_mix::calculate returns 1), so
// isActiveR and isActiveDOF are 1 everywhere and proteus's masking of the residual and the Jacobian rows is the identity.
// The cut-cell path (useExact_s = 1: the ghost penalty on cut faces and the inactive rows) is not reproduced.
#pragma once
#include <cmath>
#include "portable.h"
#include "masked.h"
#include "spaces.h"
#include "rdls.h"

namespace adrh
{
  // equivalent_polynomials.h Regularized::D, the smoothed Dirac delta MCorr uses when useExact = 0
  ADRH_FN ADRH_FORCE_INLINE double smoothedDirac(double eps, double phi)
  {
    if (phi > eps) return 0.0;
    if (phi < -eps) return 0.0;
    return 0.5 * (1.0 + std::cos(M_PI * phi / eps)) / eps;
  }

  // fields (nE, nQ, nFields) = (phi, H_vof, porosity, h_phi, h_e, phi_s) at each point
  template <int nS_>
  struct MCorrCoefficients : DenseMasks
  {
    static constexpr int nc = 1, nSpace = nS_, nFields = 6;
    static constexpr bool has_mass = false, has_advection = false, has_hamiltonian = false;
    static constexpr bool da_nz(int, int, int) { return false; }   // eps_diff does not depend on u
    double useMetrics, epsFactHeaviside, epsFactDirac, epsFactDiffusion;
    const double *fields;   // the whole mesh's point fields
    int nQ;
    const double *fq = nullptr;   // this point's fields, set by bindElement + bindPoint
    ADRH_FN MCorrCoefficients bindElement(int eN) const
    {
      MCorrCoefficients c = *this;
      c.fq = fields + eN * nQ * nFields;
      return c;
    }
    ADRH_FN MCorrCoefficients bindPoint(int q) const
    {
      MCorrCoefficients c = *this;
      c.fq = fq + q * nFields;
      return c;
    }
    // MCorr.h's per-point rules: eps from h_phi and h_e, H_s, r and dr (evaluateCoefficients)
    ADRH_FN ADRH_FORCE_INLINE void pointCoefficients(double u, double &r, double &dr, double &epsDiffusion, double &H_s) const
    {
      const double phi = fq[0], Hvof = fq[1], porosity = fq[2], h_phi = fq[3], h_e = fq[4], phi_s = fq[5];
      const double epsHeaviside = epsFactHeaviside * (useMetrics * h_phi + (1.0 - useMetrics) * h_e);
      const double epsDirac = epsFactDirac * (useMetrics * h_phi + (1.0 - useMetrics) * h_e);
      epsDiffusion = epsFactDiffusion * (useMetrics * h_phi + (1.0 - useMetrics) * h_e);
      H_s = smoothedHeaviside(0.0, phi_s);
      r = porosity * (smoothedHeaviside(epsHeaviside, phi + u) - Hvof);
      dr = porosity * smoothedDirac(epsDirac, phi + u);
    }
    ADRH_FN void evaluate(double, const double[3], const double u[1], const double[1][nS_], Terms<1, nS_> &T) const
    {
      double r, dr, epsDiffusion, H_s;
      pointCoefficients(u[0], r, dr, epsDiffusion, H_s);
      T.r[0] = H_s * r;
      T.dr[0][0] = H_s * dr;
      T.phi[0] = u[0];
      T.dphi[0][0] = 1.0;
      for (int I = 0; I < nS_; ++I) T.a[0][0][I][I] = H_s * epsDiffusion;
    }
  };

  // What MCorr.h writes at the points of one element for the other models: q_u, q_r (r without H_s) and
  // q_n = grad u / (sqrt(1e-8 + |grad u|^2)). C is the functor bound to the element.
  template <class Space, int nS = Space::nS>
  ADRH_FN void mcorrPointOutputs(const MCorrCoefficients<nS> &C, const double *X, const double *ue, int nQ, const double *qp,
                                 double *q_u, double *q_r, double *q_n)
  {
    constexpr int nD = Space::nDOF;
    PointGeometry<Space> G;
    for (int q = 0; q < nQ; ++q)
    {
      G.compute(X, qp + q * nS);
      double u = 0.0, gu[nS] = {};
      for (int b = 0; b < nD; ++b)
      {
        u += ue[b] * G.w[b];
        for (int I = 0; I < nS; ++I) gu[I] += ue[b] * G.gw[b][I];
      }
      double r, dr, epsDiffusion, H_s;
      C.bindPoint(q).pointCoefficients(u, r, dr, epsDiffusion, H_s);
      q_u[q] = u;
      q_r[q] = r;
      double norm = 1.0e-8;
      for (int I = 0; I < nS; ++I) norm += gu[I] * gu[I];
      norm = std::sqrt(norm);
      for (int I = 0; I < nS; ++I) q_n[q * nS + I] = gu[I] / norm;
    }
  }
} // namespace adrh
