// The fifth two-phase slice of the engine and its first vector model: proteus's RANS2P element integrals
// (mprans/RANS2P2D.h calculateResidual / calculateJacobian, the element loop) on damBreakLevelSet's configuration:
// the non-conservative form, two phases mixed by VOF's vf (useVF = 1), equal-order P1 pressure and velocity, the
// residual-based VMS stabilization with the lagged subgrid velocity, and the lagged shock-capturing viscosity.
//
// Unknowns (nc = 1 + nS): p, then the velocity components u_c. Per quadrature point, with H = clamp(vf, 0, 1):
//   rho = rho_0 (1 - H) + rho_1 H,   nu = nu_0 (1 - H) + nu_1 H,   mu = rho nu,   dm = rho porosity
//   continuity     div(porosity u)                                     weak: -porosity u . grad q
//   momentum c     dm (alpha u_c + beta_c dV_last/dV)                  (BDF1, beta from the time integrator per point)
//                  + porosity dp/dx_c + rho porosity (u . grad u_c)    (Hamiltonians: pressure gradient and convection)
//                  - div(porosity mu (grad u + grad u^T))_c            (the diffusion tensors of RANS2P2D.h's sdInfo)
//                  - porosity rho g_c                                  (source)
//                  + nu_sc,c grad u_c . grad w                         (lagged shock capturing, q_numDiff_*_last)
// VMS (RANS2P2D.h, calculateSubgridError_tau and calculateSubgridError_tauRes): with a* = dm v_sge (the lagged velocity),
//   R_p = div(porosity u),  R_c = dm_t u-terms + a* . grad u_c + porosity dp/dx_c + source,
//   tau_v = 1 / sqrt(Ct (dm alpha)^2 + a*.G a* + 1e-12 + Cd (porosity mu)^2 G:G),  tau_p = 1 / (porosity tr(G) tau_v)
//   (useMetrics = 1; the h form is evaluated too and mixed by useMetrics, as proteus does), and the subgrid terms
//   tau_v R_c L*_cp(q) + ... in the continuity rows, (-tau_v R_c)(-a* . grad w) and (-tau_p R_p)(-porosity dw/dx_c) in
//   momentum row c. The Jacobian is proteus's: exact in the Galerkin terms (convection dH/du and dH/d grad u, the mass,
//   the pressure gradient, the diffusion), Picard in the stabilization (tau, a* and the shock-capturing viscosity
//   frozen; the subgrid residual differentiated with the lagged a*), and without the cross-component subgrid terms
//   proteus leaves out (no d(-tau_v R_v)/du in row u).
//
// What proteus evaluates and this configuration multiplies by zero is left out: turbulence closures (nu_t = 0 and
// q_eddy_viscosity_last = 0), Darcy-Forchheimer drag (q_dragAlpha = q_dragBeta = 0), particles (nParticles = 0, H_s = 1),
// the moving mesh (MOVING_DOMAIN = 0), RBLES (useRBLES = 0), the pressure projection (PRESSURE_PROJECTION_STABILIZATION
// = 0), mass sources and point forces (0), the regularized phase field (useVF = 1 multiplies it by 0). The test
// (rans2p_test.py) checks those inputs on the fixture. Exterior faces (the Nitsche/SIPG boundary terms) are not reproduced yet.
//
// fields (nE, nQ, nFields): vf, porosity, dV_last, beta_c (nS), numDiff_last_c (nS), v_sge (nS), all as proteus hands
// them to the kernel. Outputs per point: the velocity (q_velocity), the next shock-capturing viscosity (q_numDiff_u)
// and the CFL (q_cfl, from the h-form tau, as proteus).
#pragma once
#include <cmath>
#include "portable.h"
#include "spaces.h"

namespace adrh
{
  struct NavierStokesParams
  {
    double rho_0, nu_0, rho_1, nu_1, g[3];
    double alphaBDF, Ct_sge, Cd_sge, C_dc, useMetrics, hFactor, useVF;
    double MOMENTUM_SGE, PRESSURE_SGE, VELOCITY_SGE;
  };

  template <int nS>
  struct NavierStokesFields
  {
    static constexpr int nFields = 3 + 3 * nS;
    static constexpr int vf = 0, porosity = 1, dV_last = 2, beta = 3, numDiff_last = 3 + nS, v_sge = 3 + 2 * nS;
  };

  // One element: X (nD, 3) the nodes, ue (nc, nD) the DOFs (p, u_0, .., u_{nS-1}), fq (nQ, nFields), h_e the element
  // diameter. R (nc, nD), J (nc, nD, nc, nD); q_vel (nQ, nS), q_numDiff (nQ), q_cfl (nQ).
  template <class Space>
  ADRH_FN void navierStokesVMSElement(const NavierStokesParams &P, const double *X, const double *ue, int nQ,
                                      const double *qp, const double *qw, const double *fq, double h_e, double *R,
                                      double *J, double *q_vel, double *q_numDiff, double *q_cfl)
  {
    constexpr int nS = Space::nS, nD = Space::nDOF, nc = 1 + nS;
    using F = NavierStokesFields<nS>;
    constexpr double inertial_term = 1.0;   // RANS2P2D.h's file constant
    double Rl[nc][nD] = {}, Jl[nc][nD][nc][nD] = {};
    for (int q = 0; q < nQ; ++q)
    {
      PointGeometry<Space> G;
      G.compute(X, qp + q * nS);
      const double *f = fq + q * F::nFields;
      const double dV = G.adet * qw[q];
      // the metric tensor G = Jinv^T Jinv (CompKernel calculateG)
      double Gm[nS][nS], G_dd_G = 0.0, tr_G = 0.0;
      for (int I = 0; I < nS; ++I)
        for (int K = 0; K < nS; ++K)
        {
          Gm[I][K] = 0.0;
          for (int L = 0; L < nS; ++L) Gm[I][K] += G.Jinv[L][I] * G.Jinv[L][K];
        }
      for (int I = 0; I < nS; ++I)
      {
        for (int K = 0; K < nS; ++K) G_dd_G += Gm[I][K] * Gm[I][K];
        tr_G += Gm[I][I];
      }
      // the solution and its gradients
      double p = 0.0, gp[nS] = {}, u[nS] = {}, gu[nS][nS] = {};
      for (int b = 0; b < nD; ++b)
      {
        p += ue[b] * G.w[b];
        for (int c = 0; c < nS; ++c) u[c] += ue[(1 + c) * nD + b] * G.w[b];
      }
      for (int b = 0; b < nD; ++b)
        for (int I = 0; I < nS; ++I)
        {
          gp[I] += ue[b] * G.gw[b][I];
          for (int c = 0; c < nS; ++c) gu[c][I] += ue[(1 + c) * nD + b] * G.gw[b][I];
        }
      double w_dV[nD], gw_dV[nD][nS];
      for (int a = 0; a < nD; ++a)
      {
        w_dV[a] = G.w[a] * dV;
        for (int I = 0; I < nS; ++I) gw_dV[a][I] = G.gw[a][I] * dV;
      }
      // the coefficients (RANS2P2D.h evaluateCoefficients, NONCONSERVATIVE_FORM > 0)
      const double porosity = f[F::porosity];
      const double Hc = std::fmin(1.0, std::fmax(0.0, f[F::vf]));
      const double H = P.useVF * Hc, ImH = P.useVF * (1.0 - Hc);
      const double rho = P.rho_0 * ImH + P.rho_1 * H, nu = P.nu_0 * ImH + P.nu_1 * H;
      const double mu = rho * nu;
      const double dm = rho * porosity;                       // dmom_c_acc_c
      const double diag2 = 2.0 * porosity * mu, pm = porosity * mu;
      double ham[nS], dham_grad_u[nS], dham_u[nS][nS], source[nS];
      for (int I = 0; I < nS; ++I) dham_grad_u[I] = inertial_term * rho * porosity * u[I];
      for (int c = 0; c < nS; ++c)
      {
        double ugu = 0.0;
        for (int I = 0; I < nS; ++I) ugu += u[I] * gu[c][I];
        ham[c] = porosity * gp[c];
        ham[c] += inertial_term * rho * porosity * ugu;
        for (int d = 0; d < nS; ++d) dham_u[c][d] = inertial_term * rho * porosity * gu[c][d];
        source[c] = -porosity * rho * P.g[c];
      }
      // BDF1 with proteus's per-point history (ck.bdf), the non-conservative scaling
      double m_t[nS];
      for (int c = 0; c < nS; ++c)
      {
        m_t[c] = P.alphaBDF * u[c] + f[F::beta + c] * f[F::dV_last] / dV;
        m_t[c] *= dm;
      }
      const double dm_t = P.alphaBDF * dm;
      // strong residuals with the lagged subgrid velocity a* = dm v_sge
      double astar[nS];
      for (int I = 0; I < nS; ++I) astar[I] = inertial_term * dm * f[F::v_sge + I];
      double pdeR_p = 0.0;
      for (int c = 0; c < nS; ++c)
        for (int I = 0; I < nS; ++I) pdeR_p += (I == c ? porosity : 0.0) * gu[c][I];
      double pdeR[nS];
      for (int c = 0; c < nS; ++c)
      {
        double hs = 0.0, hp = 0.0;
        for (int I = 0; I < nS; ++I) hs += astar[I] * gu[c][I];
        for (int I = 0; I < nS; ++I) hp += (I == c ? porosity : 0.0) * gp[I];
        pdeR[c] = m_t[c] + hs + hp + source[c];
      }
      // tau: the h form (which also gives the CFL) and the metric form, mixed by useMetrics
      const double tmpR = dm_t;
      double nrm_df = 0.0;
      for (int I = 0; I < nS; ++I) nrm_df += astar[I] * astar[I];
      nrm_df = std::sqrt(nrm_df);
      const double h = P.hFactor * h_e;
      q_cfl[q] = nrm_df / (h * dm);
      const double oneByAbsdt = std::fabs(tmpR);
      const double tau_v0 = 1.0 / (4.0 * pm / (h * h) + inertial_term * (2.0 * nrm_df / h + oneByAbsdt));
      const double tau_p0 = (4.0 * pm + inertial_term * (2.0 * nrm_df * h + oneByAbsdt * h * h)) / porosity;
      double vGv = 0.0;
      for (int I = 0; I < nS; ++I)
        for (int K = 0; K < nS; ++K) vGv += astar[I] * Gm[I][K] * astar[K];
      const double tau_v1 = 1.0 / std::sqrt(inertial_term * (P.Ct_sge * tmpR * tmpR + vGv + 1.0e-12) + P.Cd_sge * pm * pm * G_dd_G);
      const double tau_p1 = 1.0 / (porosity * tr_G * tau_v1);
      const double tau_v = P.useMetrics * tau_v1 + (1.0 - P.useMetrics) * tau_v0;
      const double tau_p = P.useMetrics * tau_p1 + (1.0 - P.useMetrics) * tau_p0;
      const double sge_p = -tau_p * pdeR_p;
      double sge[nS];
      for (int c = 0; c < nS; ++c) sge[c] = -tau_v * pdeR[c];
      // adjoints times the test functions
      double Lcp[nS][nD], Lcc[nD], Lpc[nS][nD];
      for (int a = 0; a < nD; ++a)
      {
        double l = 0.0;
        for (int I = 0; I < nS; ++I) l -= astar[I] * gw_dV[a][I];
        Lcc[a] = l;
        for (int c = 0; c < nS; ++c)
        {
          double lp = 0.0, lc = 0.0;
          for (int I = 0; I < nS; ++I)
          {
            lp -= (I == c ? porosity : 0.0) * gw_dV[a][I];   // Advection_adjoint(dmass_adv_c, grad q dV)
            lc -= (I == c ? porosity : 0.0) * gw_dV[a][I];   // Hamiltonian_adjoint(dmom_c_ham_grad_p, grad w dV)
          }
          Lcp[c][a] = lp;
          Lpc[c][a] = lc;
        }
      }
      // the next shock-capturing viscosity (lagged: used at the next step) and the velocity at the point
      double nR = 0.0;
      for (int c = 0; c < nS; ++c) nR += pdeR[c] * pdeR[c];
      nR = std::sqrt(nR);
      q_numDiff[q] = P.C_dc * nR * (P.useMetrics / std::sqrt(G_dd_G + 1.0e-12) +
                                    (1.0 - P.useMetrics) * P.hFactor * P.hFactor * h_e * h_e);
      for (int c = 0; c < nS; ++c) q_vel[q * nS + c] = u[c];
      // the diffusion tensor entry of momentum row c against grad u_d, as the sdInfo sparsity orders it
      auto diffusionWeak = [&](int c, const double *gwa) {
        double s = 0.0;
        for (int d = 0; d < nS; ++d)
        {
          double t = 0.0;
          if (d == c)
            for (int I = 0; I < nS; ++I) t += (I == c ? diag2 : pm) * gu[c][I] * gwa[I];
          else
            t += pm * gu[d][c] * gwa[d];
          s += t;
        }
        return s;
      };
      // residual
      for (int a = 0; a < nD; ++a)
      {
        double adv = 0.0;
        for (int I = 0; I < nS; ++I) adv -= porosity * u[I] * gw_dV[a][I];
        double s = adv;
        for (int c = 0; c < nS; ++c) s += sge[c] * Lcp[c][a];
        Rl[0][a] += s;
        for (int c = 0; c < nS; ++c)
        {
          double nd = 0.0;
          for (int I = 0; I < nS; ++I) nd += gu[c][I] * gw_dV[a][I];
          double rc = m_t[c] * w_dV[a] + diffusionWeak(c, gw_dV[a]) + source[c] * w_dV[a] + ham[c] * w_dV[a] +
                      P.MOMENTUM_SGE * P.VELOCITY_SGE * sge[c] * Lcc[a] + f[F::numDiff_last + c] * nd;
          rc += P.MOMENTUM_SGE * P.PRESSURE_SGE * sge_p * Lpc[c][a];
          Rl[1 + c][a] += rc;
        }
      }
      // Jacobian: derivatives of the strong residuals with the lagged a*, then the weak terms
      double dRp_u[nS][nD], dRc_u[nD], dRc_p[nS][nD];
      for (int b = 0; b < nD; ++b)
      {
        double s = 0.0;
        for (int I = 0; I < nS; ++I) s += astar[I] * G.gw[b][I];
        dRc_u[b] = dm_t * G.w[b] + s;
        for (int c = 0; c < nS; ++c)
        {
          double sp = 0.0, sc = 0.0;
          for (int I = 0; I < nS; ++I)
          {
            sp += (I == c ? porosity : 0.0) * G.gw[b][I];
            sc += (I == c ? porosity : 0.0) * G.gw[b][I];
          }
          dRp_u[c][b] = sp;
          dRc_p[c][b] = sc;
        }
      }
      for (int a = 0; a < nD; ++a)
        for (int b = 0; b < nD; ++b)
        {
          double gg = 0.0;
          for (int I = 0; I < nS; ++I) gg += G.gw[b][I] * gw_dV[a][I];
          // continuity rows
          double jpp = 0.0;
          for (int c = 0; c < nS; ++c) jpp += (-tau_v * dRc_p[c][b]) * Lcp[c][a];
          Jl[0][a][0][b] += jpp;
          for (int c = 0; c < nS; ++c)
          {
            double adv = 0.0;
            for (int I = 0; I < nS; ++I) adv -= (I == c ? porosity : 0.0) * G.w[b] * gw_dV[a][I];
            Jl[0][a][1 + c][b] += adv;
            Jl[0][a][1 + c][b] += (-tau_v * dRc_u[b]) * Lcp[c][a];
          }
          // momentum rows
          for (int c = 0; c < nS; ++c)
          {
            double hp = 0.0;
            for (int I = 0; I < nS; ++I) hp += (I == c ? porosity : 0.0) * G.gw[b][I] * w_dV[a];
            Jl[1 + c][a][0][b] += hp + P.MOMENTUM_SGE * P.VELOCITY_SGE * (-tau_v * dRc_p[c][b]) * Lcc[a];
            for (int d = 0; d < nS; ++d)
            {
              double j;
              if (d == c)
              {
                double hg = 0.0, dif = 0.0;
                for (int I = 0; I < nS; ++I) hg += dham_grad_u[I] * G.gw[b][I] * w_dV[a];
                for (int I = 0; I < nS; ++I) dif += (I == c ? diag2 : pm) * G.gw[b][I] * gw_dV[a][I];
                j = dm_t * G.w[b] * w_dV[a] + dham_u[c][c] * G.w[b] * w_dV[a] + hg + dif +
                    P.MOMENTUM_SGE * P.PRESSURE_SGE * (-tau_p * dRp_u[c][b]) * Lpc[c][a] +
                    P.MOMENTUM_SGE * P.VELOCITY_SGE * (-tau_v * dRc_u[b]) * Lcc[a] + f[F::numDiff_last + c] * gg;
              }
              else
                j = dham_u[c][d] * G.w[b] * w_dV[a] + pm * G.gw[b][c] * gw_dV[a][d] +
                    P.MOMENTUM_SGE * P.PRESSURE_SGE * (-tau_p * dRp_u[d][b]) * Lpc[c][a];
              Jl[1 + c][a][1 + d][b] += j;
            }
          }
        }
    }
    for (int k = 0; k < nc * nD; ++k) R[k] = (&Rl[0][0])[k];
    for (int k = 0; k < nc * nD * nc * nD; ++k) J[k] = (&Jl[0][0][0][0])[k];
  }
} // namespace adrh
