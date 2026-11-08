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
// The solid and its constraint pass (MCorr.h's element loop and cut-face loop, then MCorr.py's masking), reproduced as
// proteus does them, quirks included, so that proteus stays the oracle (C19 round 4; decision 9 of 2026-11-06):
//
// 1. Per element, equivalent_polynomials' GeneralizedFunctions_mix (proteus's own header, the same template MCorr.h
//    instantiates: moment fit of degree 4) classifies the element by its nodal phi_s (mcorrSolidElement): icase 1 in the
//    fluid (every node > 1e-8), -1 in the solid (every node < -1e-8), 0 cut. H_s at the points is its H: the moment fit
//    on a cut element when useExact_s = 1; with useExact_s = 0 it is the regularized H with eps = 0, and every element is
//    "cut" (Regularized::calculate returns 0). Elements with icase 0 or 1 are active, and so are their nodes' rows and DOFs.
// 2. A cut face (mcorrCutFaces) is an interior face of an icase-0 element whose two nodes (local face f opposite local node
//    f; nodes (f+1)%nD and (f+2)%nD) have phi_s(a) phi_s(b) <= 0, kept when both neighbours are active. Quirks: the
//    product test is sharp (no 1e-8), and in 3D it reads only two of the face's three nodes (MCorr.h's "only works for
//    simplices"). proteus's Jacobian reuses the set its last residual built.
// 3. The ghost penalty on a cut face (mcorrGhostFace, P1): gamma h_F sum_kb [grad u . n] [grad w_i . n] dS with gamma
//    = 10 ghost_penalty_constant (MCorr.py), h_F the face's diameter and [.] the sum over the two sides with each side's own
//    outward normal. P1 gradients are constant per side, so the face integral is |F| times the product.
// 4. After assembly MCorr.py multiplies the residual by isActiveR, replaces the Jacobian rows of inactive DOFs by identity
//    rows (in the matrix's own sparsity) and zeroes the inactive DOFs (replaceInactiveRows). This is row replacement, not a
//    Dirichlet condition on the global system; the global-constraint form is a later behaviour-change round.
#pragma once
#include <cmath>
#include <vector>
#include "portable.h"
#include "masked.h"
#include "spaces.h"
#include "rdls.h"
// proteus's header, unchanged: its dimension-generic code indexes past small arrays in branches dead for the instantiated nSpace
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Warray-bounds"
#endif
#include "../../proteus/equivalent_polynomials.h"
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif

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
    const double *Hs = nullptr;   // H_s at the points (nE, nQ) from mcorrSolidElement; null: the sharp H_0(phi_s) of the fields
    const double *fq = nullptr;   // this point's fields, set by bindElement + bindPoint
    const double *hq = nullptr;   // this point's H_s when Hs is given
    ADRH_FN MCorrCoefficients bindElement(int eN) const
    {
      MCorrCoefficients c = *this;
      c.fq = fields + eN * nQ * nFields;
      c.hq = Hs ? Hs + eN * nQ : nullptr;
      return c;
    }
    ADRH_FN MCorrCoefficients bindPoint(int q) const
    {
      MCorrCoefficients c = *this;
      c.fq = fq + q * nFields;
      c.hq = hq ? hq + q : nullptr;
      return c;
    }
    // MCorr.h's per-point rules: eps from h_phi and h_e, H_s, r and dr (evaluateCoefficients)
    ADRH_FN ADRH_FORCE_INLINE void pointCoefficients(double u, double &r, double &dr, double &epsDiffusion, double &H_s) const
    {
      const double phi = fq[0], Hvof = fq[1], porosity = fq[2], h_phi = fq[3], h_e = fq[4], phi_s = fq[5];
      const double epsHeaviside = epsFactHeaviside * (useMetrics * h_phi + (1.0 - useMetrics) * h_e);
      const double epsDirac = epsFactDirac * (useMetrics * h_phi + (1.0 - useMetrics) * h_e);
      epsDiffusion = epsFactDiffusion * (useMetrics * h_phi + (1.0 - useMetrics) * h_e);
      H_s = hq ? *hq : smoothedHeaviside(0.0, phi_s);
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

  // 1. The element's case and H_s at its points (MCorr.h: gf_s.calculate(element_phi_s, element_nodes, x_ref), then per point
  // gf_s.set_quad(k), gf_s.H(0, phi_solid[eN_k])). X: nD nodes x 3; phi_e: phi_s at the nodes; xref: nQ x 3 reference points.
  template <int nS, int nQ>
  struct MCorrSolid
  {
    equivalent_polynomials::GeneralizedFunctions_mix<nS, 4, 4, nQ, 1, false> gf;
    explicit MCorrSolid(bool useExact_s) : gf(useExact_s) {}
    int element(const double *X, const double *phi_e, const double *xref, const double *phi_q, double *Hs)
    {
      const int icase = gf.calculate(phi_e, X, xref, false);
      for (int q = 0; q < nQ; ++q)
      {
        gf.set_quad(q);
        Hs[q] = gf.H(0.0, phi_q[q]);
      }
      return icase;
    }
  };

  // 2. The cut faces, in MCorr.h's order (a std::set of face numbers). icase (nE); phi_s at the nodes; l2g (nE, nD);
  // elementBoundaries (nE, nD): the element's faces, face f opposite node f; boundaryElements (nEB, 2), -1 outside.
  inline std::vector<int> mcorrCutFaces(int nE, int nD, const int *icase, const double *phi_s, const int *l2g,
                                        const int *elementBoundaries, const int *boundaryElements)
  {
    std::vector<char> mark;
    std::vector<int> faces;
    for (int eN = 0; eN < nE; ++eN)
    {
      if (icase[eN] != 0) continue;
      for (int f = 0; f < nD; ++f)
      {
        const int ebN = elementBoundaries[eN * nD + f];
        const double a = phi_s[l2g[eN * nD + (f + 1) % nD]], b = phi_s[l2g[eN * nD + (f + 2) % nD]];
        if (boundaryElements[ebN * 2 + 1] != -1 && a * b <= 0.0)
        {
          if (ebN >= (int)mark.size()) mark.resize(ebN + 1, 0);
          mark[ebN] = 1;
        }
      }
    }
    for (int ebN = 0; ebN < (int)mark.size(); ++ebN)
    {
      if (!mark[ebN]) continue;
      const int eL = boundaryElements[ebN * 2], eR = boundaryElements[ebN * 2 + 1];
      const bool activeL = icase[eL] == 0 || icase[eL] == 1, activeR = icase[eR] == 0 || icase[eR] == 1;
      if (activeL && activeR) faces.push_back(ebN);
    }
    return faces;
  }

  // 3. The ghost penalty of one cut face, P1 simplices: the residual of each side's test functions and the Jacobian
  // blocks (side s, dof i; side t, dof j). XL, XR: the two elements' nodes; uL, uR: their DOFs; fL, fR: the face's local
  // number in each (opposite that node); gh = gamma h_F.
  template <class Space>
  void mcorrGhostFace(const double *XL, const double *XR, const double *uL, const double *uR, int fL, int fR, double gh,
                      double *R /* 2 x nD */, double *J /* 2 x nD x 2 x nD */)
  {
    constexpr int nS = Space::nS, nD = Space::nDOF;
    static_assert(nD == nS + 1, "the ghost penalty is reproduced for P1 simplices (MCorr.h's cut-face loop)");
    const double *X[2] = {XL, XR}, *ue[2] = {uL, uR};
    const int f[2] = {fL, fR};
    double dWn[2][nD], dUn = 0.0, area = 0.0;
    for (int s = 0; s < 2; ++s)
    {
      PointGeometry<Space> G;
      const double xi[3] = {0.0, 0.0, 0.0};
      G.compute(X[s], xi);
      // the outward unit normal of face f: minus the gradient of the opposite node's basis function, normalized
      double n[nS], nn = 0.0;
      for (int I = 0; I < nS; ++I) { n[I] = -G.gw[f[s]][I]; nn += n[I] * n[I]; }
      nn = std::sqrt(nn);
      for (int I = 0; I < nS; ++I) n[I] /= nn;
      for (int b = 0; b < nD; ++b)
      {
        dWn[s][b] = 0.0;
        for (int I = 0; I < nS; ++I) dWn[s][b] += G.gw[b][I] * n[I];
        dUn += ue[s][b] * dWn[s][b];
      }
      if (s == 0)
      {
        // the face's measure, from side L's nodes
        const double *A = X[0] + ((f[0] + 1) % nD) * 3, *B = X[0] + ((f[0] + 2) % nD) * 3;
        if constexpr (nS == 2)
          area = std::sqrt((B[0] - A[0]) * (B[0] - A[0]) + (B[1] - A[1]) * (B[1] - A[1]));
        else
        {
          const double *C = X[0] + ((f[0] + 3) % nD) * 3;
          const double u[3] = {B[0] - A[0], B[1] - A[1], B[2] - A[2]}, v[3] = {C[0] - A[0], C[1] - A[1], C[2] - A[2]};
          const double c[3] = {u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]};
          area = 0.5 * std::sqrt(c[0] * c[0] + c[1] * c[1] + c[2] * c[2]);
        }
      }
    }
    for (int s = 0; s < 2; ++s)
      for (int i = 0; i < nD; ++i)
      {
        R[s * nD + i] = gh * dUn * dWn[s][i] * area;
        for (int t = 0; t < 2; ++t)
          for (int j = 0; j < nD; ++j) J[((s * nD + i) * 2 + t) * nD + j] = gh * dWn[t][j] * dWn[s][i] * area;
      }
  }

  // 4. MCorr.py's masking after assembly: r *= isActiveR; the Jacobian row of an inactive DOF becomes the identity row
  // (diagonal 1, the rest of the row's stored entries 0); the inactive DOFs are zeroed (u_dof, when given).
  inline void replaceInactiveRows(int n, const int *rowptr, const int *colind, double *vals, double *r, const double *activeR,
                                  double *u_dof, const double *activeDOF)
  {
    for (int i = 0; i < n; ++i)
    {
      r[i] *= activeR[i];
      if (activeR[i] == 0.0)
        for (int k = rowptr[i]; k < rowptr[i + 1]; ++k) vals[k] = colind[k] == i ? 1.0 : 0.0;
      if (u_dof && activeDOF[i] != 1.0) u_dof[i] = 0.0;
    }
  }
} // namespace adrh
