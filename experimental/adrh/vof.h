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
#include <type_traits>
#include "portable.h"
#include "spaces.h"

namespace adrh
{
  // Which optional slots and inputs a coefficient functor declares (static constexpr bool members; absent = false). The kernels
  // below compile a term only for the slots the functor has, so a functor that does not use one pays nothing for it.
  //   hasF, hasH, hasR   advective flux f, Hamiltonian H, reaction r (below)
  //   hasDR              dr = dr/du: the reaction depends on u (the Jacobian gets dr w_j w_i, the strong residual dr w_j)
  //   hasA               isotropic diffusion ad (and dad = d ad/du), weak form ad grad u . grad w (Galerkin: it does not enter
  //                      tau or the strong residual, which is exact for P1)
  //   hasDDH             ddH[I][K] = d dH_I / d grad u_K, from the functor's evaluateDDH, which the kernel calls only with
  //                      SUPGParameters::exactJacobian: the Jacobian of the subgrid term
  //                      follows the dependence of a = dH on grad u (RDLS's S |grad u|); without it, proteus's Jacobian
  //   usesPoint          evaluate also takes the point's physical coordinates x[3] (a coefficient that is a function of x)
#define ADRH_FUNCTOR_FLAG(name)                                                              \
  template <class C, class = void> struct Flag_##name : std::false_type {};                  \
  template <class C> struct Flag_##name<C, std::void_t<decltype(C::name)>> : std::bool_constant<C::name> {};
  ADRH_FUNCTOR_FLAG(hasDR)
  ADRH_FUNCTOR_FLAG(hasA)
  ADRH_FUNCTOR_FLAG(hasDDH)
  ADRH_FUNCTOR_FLAG(usesPoint)
#undef ADRH_FUNCTOR_FLAG
  // A generated functor exists only for the dimensions its spec was read in: the others are the primary template with
  // available = false. A hand-written functor has no such member and is available.
  template <class C, class = void> struct Available : std::true_type {};
  template <class C> struct Available<C, std::void_t<decltype(C::available)>> : std::bool_constant<C::available> {};

  // The face kernel a functor's boundary words select (generator: from the boundary kind and the form of the equation; the
  // hand-written functors state theirs): Upwind = scalarUpwindFace (a conservative flux f, strong Dirichlet on inflow),
  // InflowPenalty = scalarInflowPenaltyFace (a Hamiltonian without flux: penalize inflow towards the boundary state), None = no
  // boundary term (a Hamiltonian nonlinear in grad u: RDLS's eikonal; the faces only hand the trace on).
  enum class FaceKind : int { Upwind = 0, InflowPenalty = 1, None = 2 };

  // What a scalar coefficient functor returns at a point: mass m(u), an advective flux f(u) (weak form -f . grad w) and a
  // Hamiltonian H(u, grad u) (weak form H w), with their derivatives. A functor declares which of f and H it has (hasF, hasH);
  // the kernel compiles only those terms, so VOF (f only) and NCLS (H only) cost what they did as separate kernels.
  // RDLS (rdls.h) adds a reaction r (weak form r w, independent of u) and an extra isotropic diffusion nuExtra that the
  // functor computes from its fields (hasR); VOF and NCLS do not set them.
  template <int nS>
  struct ScalarTerms
  {
    double m, dm, f[nS], df[nS], H, dH[nS], r, nuExtra;
    double dr, ad, dad, ddH[nS][nS];   // optional slots, set only by a functor that declares hasDR, hasA, hasDDH
  };

  // The call of a functor's evaluate: with the point's coordinates when it declares usesPoint.
  template <class Coefficients, int nS>
  ADRH_FN ADRH_FORCE_INLINE void evaluateScalar(const Coefficients &C, const double *fld, double u, const double *gu, const double *x,
                                               ScalarTerms<nS> &T)
  {
    if constexpr (Flag_usesPoint<Coefficients>::value) C.evaluate(fld, u, gu, x, T);
    else C.evaluate(fld, u, gu, T);
  }

  // VOF's coefficients (mprans/VOF.h evaluateCoefficients), in the form a generator emits: fields = (v_0..v_{nS-1}, porosity).
  template <int nS_>
  struct VOFCoefficients
  {
    static constexpr int nSpace = nS_, nFields = nS_ + 1;
    static constexpr bool hasF = true, hasH = false, hasR = false;
    static constexpr int tauForm = 0;
    static constexpr FaceKind face = FaceKind::Upwind;
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
    int lag_sc = 1;     // 1: nu = numDiffLast (VOF, NCLS); 0: nu = the numDiff computed here (RDLS)
    int exactJacobian = 0;   // 1: differentiate the subgrid term through a = dH(grad u) when the functor has hasDDH (nu is never)
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
      evaluateScalar<Coefficients, nS>(C, fields + q * nF, u, gu, G.x, T);
      if constexpr (Flag_hasDDH<Coefficients>::value)
        if (P.exactJacobian)
        {
          if constexpr (Flag_usesPoint<Coefficients>::value) C.evaluateDDH(fields + q * nF, u, gu, G.x, T);
          else C.evaluateDDH(fields + q * nF, u, gu, T);
        }
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
      // tauForm 0 (VOF, NCLS): 1/sqrt(Ct dm_t^2 + a.Ga + 1e-8); 1 (RDLS, no time term): 1/(sqrt(a.Ga) + 1e-8)
      double tau;
      if constexpr (Coefficients::tauForm == 1) tau = 1.0 / (std::sqrt(vGv) + 1.0e-8);
      else tau = 1.0 / std::sqrt(P.Ct_sge * dm_t * dm_t + vGv + 1.0e-8);
      double pdeResidual = m_t + dfgu;
      if constexpr (Coefficients::hasR) pdeResidual += T.r;
      // exact Jacobian through a(grad u) = dH (hasDDH, exactJacobian): B gw_j = ddH gw_j, gu . B gw_j and a G B gw_j per node
      constexpr bool exactSlot = Flag_hasDDH<Coefficients>::value;
      double gubgw[nD], aGBgw[nD], gwBgw[nD][nD];
      if constexpr (exactSlot)
        if (P.exactJacobian)
        {
          double aG[nS];
          for (int K = 0; K < nS; ++K)
          {
            aG[K] = 0.0;
            for (int I = 0; I < nS; ++I) aG[K] += a[I] * Gm[I][K];
          }
          double Bgw[nD][nS];
          for (int j = 0; j < nD; ++j)
          {
            gubgw[j] = 0.0; aGBgw[j] = 0.0;
            for (int I = 0; I < nS; ++I)
            {
              Bgw[j][I] = 0.0;
              for (int K = 0; K < nS; ++K) Bgw[j][I] += T.ddH[I][K] * gw[j][K];
              gubgw[j] += gu[I] * Bgw[j][I];
              aGBgw[j] += aG[I] * Bgw[j][I];
            }
          }
          for (int i = 0; i < nD; ++i)
            for (int j = 0; j < nD; ++j)
            {
              gwBgw[i][j] = 0.0;
              for (int I = 0; I < nS; ++I) gwBgw[i][j] += gw[i][I] * Bgw[j][I];
            }
        }
      const double subgridError = -tau * pdeResidual;
      {
        const double h1 = 1.0 / std::sqrt(guGgu + 1.0e-12);
        const double h2 = 1.0 / (P.sc_uref * std::sqrt(G_dd_G + 1.0e-12));
        numDiff[q] = P.sc_c * std::fabs(pdeResidual - P.sc_drop_time * m_t) * std::pow(h1, 2.0 - P.sc_beta) *
                     std::pow(h2, P.sc_beta - 1.0);
      }
      double nu = P.lag_sc ? numDiffLast[q] : numDiff[q];
      if constexpr (Coefficients::hasR) nu += T.nuExtra;
      double Lstar[nD], dR[nD];
      for (int i = 0; i < nD; ++i)
      {
        double agw = 0.0;
        for (int I = 0; I < nS; ++I) agw += a[I] * gw[i][I];
        Lstar[i] = -agw * dV;
        dR[i] = dm_t * w[i] + agw;
        if constexpr (Flag_hasDR<Coefficients>::value) dR[i] += T.dr * w[i];
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
        if constexpr (Coefficients::hasR) weak += T.r * w[i] * dV;
        if constexpr (Flag_hasA<Coefficients>::value) weak += T.ad * gugw * dV;
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
          if constexpr (Flag_hasDR<Coefficients>::value) dweak += T.dr * w[j] * w[i] * dV;
          if constexpr (Flag_hasA<Coefficients>::value) dweak += (T.ad * gwjgwi + T.dad * w[j] * gugw) * dV;
          double dsub = -tau * dR[j] * Lstar[i];
          if constexpr (exactSlot)
            if (P.exactJacobian)
            {
              // d/du_j of tau R(u) (a . gw_i): dR gets gu . B gw_j, a . gw_i gets gw_i . B gw_j, tau gets -tau^2 (a G B gw_j)/|a|_G
              // (tauForm 1) or -tau^3 (a G B gw_j) (tauForm 0)
              double dtau = 0.0;
              if constexpr (Coefficients::tauForm == 1) dtau = vGv > 0.0 ? -tau * tau * aGBgw[j] / std::sqrt(vGv) : 0.0;
              else dtau = -tau * tau * tau * aGBgw[j];
              const double agwi = -Lstar[i] / dV;
              dsub += (tau * gubgw[j] * agwi + tau * pdeResidual * gwBgw[i][j] + dtau * pdeResidual * agwi) * dV;
            }
          Jloc[i][j] += dweak + dsub + nu * gwjgwi * dV;
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
      evaluateScalar<Coefficients, nS>(C, fields + q * nF, u, nullptr, G.x, T);
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
