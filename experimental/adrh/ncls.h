// The second two-phase slice of the engine: proteus's NCLS model (mprans/NCLS.h calculateResidual/calculateJacobian,
// STABILIZATION_TYPE = 0, the damBreakLevelSet configuration) on the VOF slice's element kernel (vof.h).
//
//   m = u,  H = v . grad u                     (NCLSCoefficients: the part a generator writes; no advective flux f)
//   R_i = sum_q [ m_t w_i + H w_i + tau R(u) L*(w_i) + nu_last grad u . grad w_i ] dV               (scalarSUPGElement)
//       + sum over exterior faces of F(u) w_i dS                                                    (scalarInflowPenaltyFace)
//
// The element kernel is VOF's with a = dH = v: the same metric tau, the same lagged shock capturing, the same beta history.
// The faces differ: NCLS has no flux to upwind, so proteus penalizes the inflow, F = -(v . n)(u - u_b) where v . n < 0 and
// 0 where it is outflow. u_b is the Dirichlet value where the point has one and the redistanced level set (RDLS's trace,
// ebqe_rd_u_ext) elsewhere; on a rank-interface face proteus passes the current u so the term is zero (NCLS.py,
// ebqe_rd_u_for_kernel), which this kernel sees as data. The model also hands two point fields on to the others: grad u
// (q_n, MCorr and RDLS) and the boundary trace with its normalized gradient (ebqe_u, ebqe_grad_u), and it marks the nodes of
// every element where the level set changes sign (interface_locator, RDLS's freeze set). The first two are outputs of the
// face kernel here; the sign change is signChangeElement. The Jacobian is exact: NCLS is linear in u (v from RANS2P, nu
// lagged). MOVING_DOMAIN, PURE_BDF = 1 and the useMetrics = 0 forms are not implemented (the deck uses none of them).
#pragma once
#include <cmath>
#include "portable.h"
#include "spaces.h"
#include "vof.h"

namespace adrh
{
  // NCLS's coefficients (mprans/NCLS.h evaluateCoefficients): fields = (v_0..v_{nS-1}).
  template <int nS_>
  struct NCLSCoefficients
  {
    static constexpr int nSpace = nS_, nFields = nS_;
    static constexpr bool hasF = false, hasH = true;
    ADRH_FN ADRH_FORCE_INLINE void evaluate(const double *fld, double u, const double *gu, ScalarTerms<nS_> &T) const
    {
      T.m = u;
      T.dm = 1.0;
      T.H = 0.0;
      for (int I = 0; I < nS_; ++I)
      {
        if (gu) T.H += fld[I] * gu[I];
        T.dH[I] = fld[I];
      }
    }
  };

  // One exterior face k of an element: NCLS's exteriorNumericalFlux. flow = n . dH at each point; u_b = g where the point
  // is Dirichlet, rd (the redistanced trace) elsewhere; F = -flow (u - u_b) on inflow (flow <= 0), 0 on outflow. R and J
  // are added to; the trace u (nQf) and its normalized gradient grad u / sqrt(|grad u|^2 + 1e-8) (nQf, nS) out.
  template <class Coefficients, class Space>
  ADRH_FN ADRH_FORCE_INLINE void scalarInflowPenaltyFace(const Coefficients &C, const double *X, const double *ue, int k, int nQf,
                                                        const double *xi, const double *qwf, const double *fields,
                                                        const int *isDirichlet, const double *g, const double *rd,
                                                        double *R, double *J, double *uTrace, double *nGradU)
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
      double u = 0.0, gu[nS] = {};
      for (int b = 0; b < nD; ++b)
      {
        u += ue[b] * w[b];
        for (int I = 0; I < nS; ++I) gu[I] += ue[b] * G.gw[b][I];
      }
      C.evaluate(fields + q * nF, u, gu, T);
      double flow = 0.0;
      for (int I = 0; I < nS; ++I) flow += nrm[I] * T.dH[I];
      const double ub = isDirichlet[q] == 1 ? g[q] : rd[q];
      const double F = flow > 0.0 ? 0.0 : -flow * (u - ub), dF = flow > 0.0 ? 0.0 : -flow;
      uTrace[q] = u;
      double norm = 1.0e-8;
      for (int I = 0; I < nS; ++I) norm += gu[I] * gu[I];
      norm = std::sqrt(norm);
      for (int I = 0; I < nS; ++I) nGradU[q * nS + I] = gu[I] / norm;
      for (int a = 0; a < nD; ++a)
      {
        R[a] += F * w[a] * dS;
        for (int b = 0; b < nD; ++b) J[a * nD + b] += dF * w[b] * w[a] * dS;
      }
    }
  }

  // True when the level set's nodal values on the element do not all have the same sign (proteus's Sign: >= 0 is +).
  template <int nD>
  ADRH_FN ADRH_FORCE_INLINE bool signChangeElement(const double *ue)
  {
    const bool pos = ue[0] >= 0.0;
    for (int b = 1; b < nD; ++b)
      if ((ue[b] >= 0.0) != pos) return true;
    return false;
  }
} // namespace adrh
