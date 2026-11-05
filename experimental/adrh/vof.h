// The first two-phase slice of the engine: proteus's VOF model (mprans/VOF.h, STABILIZATION_TYPE=0, the
// damBreakLevelSet configuration) as a coefficient functor plus a scalar SUPG kernel that does not know it is VOF.
//
//   m = porosity u,  f = porosity v u          (VOFCoefficients: the part a generator writes)
//   R_i = sum_q [ m_t w_i - f . grad w_i + H w_i + tau R(u) L*(w_i) + nu_last grad u . grad w_i ] dV  (scalarSUPGElement)
//
// (NCLS, ncls.h, is the same element kernel with a Hamiltonian H = v . grad u in place of f: a = df + dH below.)
//       + sum over exterior faces of F(u) w_i dS                                                   (scalarUpwindFace)
//
// with m_t = alpha m + beta (BDF1, beta the stored history at the point), a = df + dH, the strong residual R(u) = m_t + a . grad u,
// the adjoint L*(w) = -a . grad w, the metric tau = 1/sqrt(4 dm_t^2 + a . G a + 1e-8), G = J^{-T} J^{-1}, and the shock
// capturing viscosity nu = c |R(u) - sc_drop_time m_t| (grad u . G grad u)^{-(2-beta)/2} (uref sqrt(G:G))^{1-beta}: computed
// here and returned, used one step later (lagged). Two things the Terms/elementKernelSpace path cannot express and this
// kernel adds, because every two-phase model needs them (C19_GENERATOR_GAP_NOTE_2026-11-03.md, G1 and G5-G7):
//   * per-point input fields from other models: the functor declares nFields and reads them from fields[q*nFields ...]
//     (VOF: the velocity RANS2P computed and the porosity);
//   * the mass history as a point field (beta), not the old state: MCorr rewrites VOF's history with H_eps(phi) every step,
//     so it is not m(u_old).
// The Jacobian is exact for VOF (linear in u; tau depends on v only; nu is lagged). The useMetrics = 0 forms (h-based tau and
// shock capturing) are not implemented: the deck uses the metric forms. The cfl output is |df| / h_e, as proteus's h-based tau
// leaves it.
#pragma once
#include <cmath>
#include "portable.h"
#include "spaces.h"

namespace adrh
{
  // What a scalar coefficient functor returns at a point: mass m(u), an advective flux f(u) (weak form -f . grad w) and a
  // Hamiltonian H(u, grad u) (weak form H w), with their derivatives. A functor declares which of f and H it has (hasF, hasH);
  // the kernel compiles only those terms, so VOF (f only) and NCLS (H only) cost what they did as separate kernels.
  template <int nS>
  struct ScalarTerms
  {
    double m, dm, f[nS], df[nS], H, dH[nS];
  };

  // VOF's coefficients (mprans/VOF.h evaluateCoefficients), in the form a generator emits: fields = (v_0..v_{nS-1}, porosity).
  template <int nS_>
  struct VOFCoefficients
  {
    static constexpr int nSpace = nS_, nFields = nS_ + 1;
    static constexpr bool hasF = true, hasH = false;
    ADRH_FN ADRH_FORCE_INLINE void evaluate(const double *fld, double u, const double *, ScalarTerms<nS_> &T) const
    {
      const double porosity = fld[nS_];
      T.m = porosity * u;
      T.dm = porosity;
      for (int I = 0; I < nS_; ++I)
      {
        T.f[I] = fld[I] * porosity * u;
        T.df[I] = fld[I] * porosity;
      }
    }
  };

  struct SUPGParameters
  {
    double alpha;       // BDF1: m_t = alpha m + beta
    double sc_c;        // shockCapturingFactor
    double sc_uref;
    double sc_beta;
    double sc_drop_time;
    double Ct_sge;      // 4 in proteus
  };

  // One element. X (nDOF,3) node coordinates; ue (nDOF); per point q: fields (nQ, nFields), beta (nQ), numDiffLast (nQ);
  // h the element diameter (cfl only). R (nDOF), J (nDOF, nDOF), numDiff (nQ), cfl (nQ) out.
  template <class Coefficients, class Space>
  ADRH_FN ADRH_FORCE_INLINE void scalarSUPGElement(const Coefficients &C, const SUPGParameters &P, const double *X, const double *ue,
                                                  int nQ, const double *qp, const double *qw, const double *fields,
                                                  const double *beta, const double *numDiffLast, double h,
                                                  double *R, double *J, double *numDiff, double *cfl)
  {
    constexpr int nS = Space::nS, nD = Space::nDOF, nF = Coefficients::nFields;
    static_assert(Coefficients::nSpace == nS, "coefficient space dimension");
    PointGeometry<Space> G;
    ScalarTerms<nS> T;
    double Rloc[nD] = {};
    double Jloc[nD][nD] = {};
    for (int q = 0; q < nQ; ++q)
    {
      G.compute(X, qp + q * nS);
      const double *w = G.w;
      const auto &gw = G.gw;
      const double dV = qw[q] * G.adet;
      double u = 0.0, gu[nS] = {};
      for (int b = 0; b < nD; ++b)
      {
        u += ue[b] * w[b];
        for (int I = 0; I < nS; ++I) gu[I] += ue[b] * gw[b][I];
      }
      C.evaluate(fields + q * nF, u, gu, T);
      const double m_t = P.alpha * T.m + beta[q], dm_t = P.alpha * T.dm;
      // the characteristic velocity of the strong residual, its adjoint and tau: df + dH (only the terms the functor has)
      double a[nS];
      for (int I = 0; I < nS; ++I)
      {
        if constexpr (Coefficients::hasF && Coefficients::hasH) a[I] = T.df[I] + T.dH[I];
        else if constexpr (Coefficients::hasH) a[I] = T.dH[I];
        else a[I] = T.df[I];
      }
      // the metric tensor G = Jinv^T Jinv (CompKernel calculateG)
      double Gm[nS][nS], G_dd_G = 0.0;
      for (int I = 0; I < nS; ++I)
        for (int K = 0; K < nS; ++K)
        {
          Gm[I][K] = 0.0;
          for (int L = 0; L < nS; ++L) Gm[I][K] += G.Jinv[L][I] * G.Jinv[L][K];
        }
      for (int I = 0; I < nS; ++I)
        for (int K = 0; K < nS; ++K) G_dd_G += Gm[I][K] * Gm[I][K];
      double vGv = 0.0, nrm_v = 0.0, dfgu = 0.0, guGgu = 0.0;
      for (int I = 0; I < nS; ++I)
      {
        nrm_v += a[I] * a[I];
        dfgu += a[I] * gu[I];
        for (int K = 0; K < nS; ++K)
        {
          vGv += a[I] * Gm[I][K] * a[K];
          guGgu += gu[I] * Gm[I][K] * gu[K];
        }
      }
      cfl[q] = std::sqrt(nrm_v) / h;
      const double tau = 1.0 / std::sqrt(P.Ct_sge * dm_t * dm_t + vGv + 1.0e-8);
      const double pdeResidual = m_t + dfgu;
      const double subgridError = -tau * pdeResidual;
      {
        const double h1 = 1.0 / std::sqrt(guGgu + 1.0e-12);
        const double h2 = 1.0 / (P.sc_uref * std::sqrt(G_dd_G + 1.0e-12));
        numDiff[q] = P.sc_c * std::fabs(pdeResidual - P.sc_drop_time * m_t) * std::pow(h1, 2.0 - P.sc_beta) *
                     std::pow(h2, P.sc_beta - 1.0);
      }
      const double nu = numDiffLast[q];
      double Lstar[nD], dR[nD];
      for (int i = 0; i < nD; ++i)
      {
        double agw = 0.0;
        for (int I = 0; I < nS; ++I) agw += a[I] * gw[i][I];
        Lstar[i] = -agw * dV;
        dR[i] = dm_t * w[i] + agw;
      }
      for (int i = 0; i < nD; ++i)
      {
        double gugw = 0.0;
        for (int I = 0; I < nS; ++I) gugw += gu[I] * gw[i][I];
        double weak = m_t * w[i] * dV;
        if constexpr (Coefficients::hasF)
        {
          double fgw = 0.0;
          for (int I = 0; I < nS; ++I) fgw += T.f[I] * gw[i][I];
          weak -= fgw * dV;
        }
        if constexpr (Coefficients::hasH) weak += T.H * w[i] * dV;
        Rloc[i] += weak + subgridError * Lstar[i] + nu * gugw * dV;
        for (int j = 0; j < nD; ++j)
        {
          double gwjgwi = 0.0;
          for (int I = 0; I < nS; ++I) gwjgwi += gw[j][I] * gw[i][I];
          double dweak = dm_t * w[j] * w[i] * dV;
          if constexpr (Coefficients::hasF)
          {
            double dfgwi = 0.0;
            for (int I = 0; I < nS; ++I) dfgwi += T.df[I] * gw[i][I];
            dweak -= dfgwi * w[j] * dV;
          }
          if constexpr (Coefficients::hasH)
          {
            double dHgwj = 0.0;
            for (int I = 0; I < nS; ++I) dHgwj += T.dH[I] * gw[j][I];
            dweak += dHgwj * w[i] * dV;
          }
          Jloc[i][j] += dweak - tau * dR[j] * Lstar[i] + nu * gwjgwi * dV;
        }
      }
    }
    for (int a = 0; a < nD; ++a)
    {
      R[a] = Rloc[a];
      for (int b = 0; b < nD; ++b) J[a * nD + b] = Jloc[a][b];
    }
  }

  // One exterior face k of an element: proteus's VOF exteriorNumericalAdvectiveFlux. flow = n . df at each point;
  //   Dirichlet point:  F = u flow on outflow (flow >= 0), g flow on inflow
  //   flux point:       F = the prescribed flux (no Jacobian)
  //   neither:          F = u flow on outflow, 0 on inflow
  // xi (nQf, nS): the face's points in the element's reference coordinates; qwf (nQf) reference face weights. Per point:
  // fields (nQf, nFields), isDirichlet, g, isFlux, gflux. R (nDOF) and J (nDOF, nDOF) are added to; flux (nQf) and the
  // upwinded trace uUp (nQf) out.
  template <class Coefficients, class Space>
  ADRH_FN ADRH_FORCE_INLINE void scalarUpwindFace(const Coefficients &C, const double *X, const double *ue, int k, int nQf,
                                                 const double *xi, const double *qwf, const double *fields, const int *isDirichlet,
                                                 const double *g, const int *isFlux, const double *gflux,
                                                 double *R, double *J, double *flux, double *uUp)
  {
    constexpr int nS = Space::nS, nD = Space::nDOF, nF = Coefficients::nFields;
    PointGeometry<Space> G;
    ScalarTerms<nS> T;
    for (int q = 0; q < nQf; ++q)
    {
      G.compute(X, xi + q * nS);
      const double *w = G.w;
      double nrm[3], vn;
      G.faceNormal(k, nrm, vn);
      const double dS = qwf[q] * vn;
      double u = 0.0;
      for (int b = 0; b < nD; ++b) u += ue[b] * w[b];
      C.evaluate(fields + q * nF, u, nullptr, T);
      double flow = 0.0;
      for (int I = 0; I < nS; ++I) flow += nrm[I] * T.df[I];
      double F, dF;
      if (isDirichlet[q] == 1)
      {
        F = flow >= 0.0 ? u * flow : g[q] * flow;
        dF = flow >= 0.0 ? flow : 0.0;
      }
      else if (isFlux[q] == 1)
      {
        F = gflux[q];
        dF = 0.0;
      }
      else
      {
        F = flow >= 0.0 ? u * flow : 0.0;
        dF = flow >= 0.0 ? flow : 0.0;
      }
      flux[q] = F;
      uUp[q] = F >= 0.0 ? u : (isDirichlet[q] == 1 ? g[q] : u);
      for (int a = 0; a < nD; ++a)
      {
        R[a] += F * w[a] * dS;
        for (int b = 0; b < nD; ++b) J[a * nD + b] += dF * w[b] * w[a] * dS;
      }
    }
  }
} // namespace adrh
