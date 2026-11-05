// The third two-phase slice of the engine: proteus's RDLS model (mprans/RDLS.h calculateResidual/calculateJacobian with
// ELLIPTIC_REDISTANCING = 0, freezeLevelSet = 1, useExact = 0, useMetrics = 1, no time integration, shock capturing and
// subgrid error not lagged: the damBreakLevelSet configuration) on the VOF/NCLS element kernel (vof.h).
//
//   S = H_eps(phi) - (1 - H_eps(phi)),  eps = epsFact h_phi       (phi: NCLS's level set at the point; h_phi: the node
//                                                                   diameters interpolated to the point)
//   m = u (no time term: dm_t = 0),  H = S |grad u|,  dH = S grad u / (|grad u| + 1e-12),  r = -S      (RDLSCoefficients)
//   R_i = sum_q [ H w_i + r w_i + tau R(u) L*(w_i) + (nu + nu_bg) grad u . grad w_i ] dV               (scalarSUPGElement)
//
// with R(u) = dH . grad u + r, L*(w) = -dH . grad w, tau = 1/(sqrt(dH . G dH) + 1e-8) (tauForm 1), nu = c |R(u)| /
// sqrt(grad u . G grad u + 1e-12) computed at the current u (lag_sc = 0, sc_beta = 1), and the background diffusion
// nu_bg = backgroundDiffusionFactor h_e where |phi| > 2 eps. H is nonlinear in grad u: the first nonlinear Hamiltonian on the
// template. The Jacobian is proteus's, not the exact one: the H term is exact (dH/d(grad u) = dH), but the subgrid term keeps
// dH and tau frozen and nu is not differentiated, as RDLS.h does.
//
// The interface pin (rdlsPinElement): after the element's quadrature, every node J of the element whose frozen value
// u_last[J] (the level set RDLS started from) is within eps of zero gets the element's residual row replaced by
// (u_J - u_last_J) wDF h_e and its Jacobian row by wDF h_e on the diagonal. eps there is the value at the element's last
// quadrature point (RDLS.h reuses the loop variable), so a node shared by elements can be pinned in some and not in others.
// RDLS has no boundary terms (NumericalFlux.DoNothing); it hands the boundary trace and its normalized gradient on
// (rdlsTraceFace). RDLS.h also writes q[('grad(u)',0)] from an uninitialized local (`dir`); that output is not reproduced.
#pragma once
#include <cmath>
#include "portable.h"
#include "spaces.h"
#include "vof.h"

namespace adrh
{
  // equivalent_polynomials.h Regularized::H, the smoothed Heaviside RDLS uses when useExact = 0
  ADRH_FN ADRH_FORCE_INLINE double smoothedHeaviside(double eps, double phi)
  {
    if (phi > eps) return 1.0;
    if (phi < -eps) return 0.0;
    if (phi == 0.0) return 0.5;
    return 0.5 * (1.0 + phi / eps + std::sin(M_PI * phi / eps) / M_PI);
  }

  // RDLS's coefficients (RDLS.h evaluateCoefficients plus the eps and background-diffusion rules of its element loop):
  // fields = (phi, h_phi, h_e) at each point.
  template <int nS_>
  struct RDLSCoefficients
  {
    static constexpr int nSpace = nS_, nFields = 3;
    static constexpr bool hasF = false, hasH = true, hasR = true;
    static constexpr int tauForm = 1;
    double epsFact, backgroundDiffusionFactor;
    ADRH_FN ADRH_FORCE_INLINE void evaluate(const double *fld, double u, const double *gu, ScalarTerms<nS_> &T) const
    {
      const double phi = fld[0], eps = epsFact * fld[1], he = fld[2];
      const double Hphi = smoothedHeaviside(eps, phi);
      const double S = Hphi - (1.0 - Hphi);
      T.m = u;
      T.dm = 1.0;
      T.r = -S;
      double norm = 0.0;
      for (int I = 0; I < nS_; ++I) norm += gu[I] * gu[I];
      norm = std::sqrt(norm);
      T.H = S * norm;
      for (int I = 0; I < nS_; ++I) T.dH[I] = S * gu[I] / (norm + 1.0e-12);
      T.nuExtra = std::fabs(phi) > 2.0 * eps ? backgroundDiffusionFactor * he : 0.0;
    }
  };

  // The interface pin of one element (RDLS.h, freezeLevelSet): ue, uLast (nD) the current and frozen nodal values, epsLast the
  // eps of the element's last quadrature point, wDF the weakDirichletFactor, he the element diameter. R (nD), J (nD, nD) are
  // the element's contributions, overwritten in the pinned rows.
  template <int nD>
  ADRH_FN ADRH_FORCE_INLINE int rdlsPinElement(const double *ue, const double *uLast, double epsLast, double wDF, double he,
                                               double *R, double *J)
  {
    int pinned = 0;
    for (int j = 0; j < nD; ++j)
      if (std::fabs(uLast[j]) < epsLast)
      {
        R[j] = (ue[j] - uLast[j]) * wDF * he;
        for (int b = 0; b < nD; ++b) J[j * nD + b] = 0.0;
        J[j * nD + j] = wDF * he;
        ++pinned;
      }
    return pinned;
  }

  // One exterior face k: the trace u (nQf) and grad u / sqrt(|grad u|^2 + 1e-8) (nQf, nS), RDLS's ebqe_u and ebqe_n.
  template <class Space>
  ADRH_FN ADRH_FORCE_INLINE void rdlsTraceFace(const double *X, const double *ue, int nQf, const double *xi, double *uTrace,
                                               double *nGradU)
  {
    constexpr int nS = Space::nS, nD = Space::nDOF;
    PointGeometry<Space> G;
    for (int q = 0; q < nQf; ++q)
    {
      G.compute(X, xi + q * nS);
      double u = 0.0, gu[nS] = {};
      for (int b = 0; b < nD; ++b)
      {
        u += ue[b] * G.w[b];
        for (int I = 0; I < nS; ++I) gu[I] += ue[b] * G.gw[b][I];
      }
      uTrace[q] = u;
      double norm = 1.0e-8;
      for (int I = 0; I < nS; ++I) norm += gu[I] * gu[I];
      norm = std::sqrt(norm);
      for (int I = 0; I < nS; ++I) nGradU[q * nS + I] = gu[I] / norm;
    }
  }
} // namespace adrh
