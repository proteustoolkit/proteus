// Two-component (H2O/CO2) compositional two-phase flow kernel.
//
// Co-authored-by: Claude Opus 5 <noreply@anthropic.com> and abarua_ce
#ifndef MPHASE_CO2_H
#define MPHASE_CO2_H
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <valarray>
#include "CompKernel.h"
#include "ModelFactory.h"
#include "../mprans/ArgumentsDict.h"
#include "pyarray.h"
#include "../pskRelations.h"
// Compositional CO2-brine EOS + analytic (p,z) flash (P3).  Standalone headers
// in global namespace ::m_comp_co2 (distinct from proteus::m_comp_co2 below);
// call sites use the fully-qualified ::m_comp_co2::flash:: form.
#include "co2_brine_flash.h"
#define nnz nSpace

namespace py = pybind11;
#define POWER_SMOOTHNESS_INDICATOR 2
#define IS_BETAij_ONE              0
#define GLOBAL_FCT                 0
namespace proteus
{
enum class STABILIZATION : int {
  Galerkin         = 0,
  EV_Stab          = 1,
  EntropyViscosity = 2,
  Implicit_FCT     = 3
};

namespace m_comp_co2
{
//cek todo: revisit entry for mass transport form
// Power entropy //
inline double ENTROPY(const double &phi, const double &phiL, const double &phiR)
{
  return 1. / 2. * std::pow(fabs(phi), 2.);
}
inline double DENTROPY(const double &phi, const double &phiL, const double &phiR)
{
  return fabs(phi) * (phi >= 0 ? 1 : -1);
}
// Log entropy // for level set from 0 to 1
inline double ENTROPY_LOG(const double &phi, const double &phiL, const double &phiR)
{
  return std::log(fabs((phi - phiL) * (phiR - phi)) + 1E-14);
}
inline double DENTROPY_LOG(const double &phi, const double &phiL, const double &phiR)
{
  return (phiL + phiR - 2 * phi) * ((phi - phiL) * (phiR - phi) >= 0 ? 1 : -1) / (fabs((phi - phiL) * (phiR - phi)) + 1E-14);
}
} // namespace m_comp_co2
} // namespace proteus
namespace proteus
{
namespace m_comp_co2
{
class M_comp_co2_base {
  //The base class defining the interface
public:
  virtual ~M_comp_co2_base() { double anb_seepage_flux = 1e-16; }
  virtual void calculateResidual(arguments_dict &args)                   = 0;
  virtual void calculateJacobian(arguments_dict &args)                   = 0;
  virtual void invert(arguments_dict &args)                              = 0;
  virtual void FCTStep(arguments_dict &args)                             = 0;
  virtual void kth_FCT_step(arguments_dict &args)                        = 0;
  virtual void calculateResidual_entropy_viscosity(arguments_dict &args) = 0;
  virtual void calculateMassMatrix(arguments_dict &args)                 = 0;
  virtual void dissolutionFlash(arguments_dict &args)                    = 0;
  virtual void calculateFlashFields(arguments_dict &args)                = 0;
};

template <class CompKernelType, int nSpace, int nQuadraturePoints_element, int nDOF_mesh_trial_element, int nDOF_trial_element, int nDOF_test_element, int nQuadraturePoints_elementBoundary>
class M_comp_co2 : public M_comp_co2_base {
public:
  const int      nDOF_test_X_trial_element;
  CompKernelType ck;
  std::vector<double> rho_dof_member;
  std::vector<double> rho_n_phi_dof_member;
  int PSK_TYPE_member = 0;
  // Compositional (p,z) state for the flash.  Isothermal / fixed-salinity per
  // solve.  T_C_member is now wired through argsDict["T_C"] and set (like
  // PSK_TYPE_member) at every top-level entry point from Coefficients.T_C, so
  // temperature can be changed from the input deck without recompiling.  The
  // 20.0 default is only the pre-first-entry fallback.  m_NaCl is still fixed
  // (SP2005 salting-out is TODO); wire it the same way if salinity is needed.
  double T_C_member    = 20.0;   // temperature [degC] (default; overwritten from argsDict["T_C"])
  double m_NaCl_member = 0.0;    // salinity [mol/kg] (SP2005 salting-out is TODO)
  bool immiscible_member = false;
  // Per-rock closure output: min / max over the rocks at each node of S_r,
  // from the last residual evaluation (Newton's last one is at the converged
  // state); NaN at one-rock nodes.  Copied out by calculateFlashFields.
  std::vector<double> Sg_rock_min_member, Sg_rock_max_member;
  M_comp_co2() : nDOF_test_X_trial_element(nDOF_test_element * nDOF_trial_element), ck() { }
  //
  // Closure functions vgm_/bc_*_from_Se expect wetting effective saturation
  //   S_e = (S_w - S_wr)/(1 - S_wr) = (1 - u_n - S_wr)/(1 - S_wr).
  // dS_e/du_n = -1/(1 - S_wr) (negative); the sign flip propagates through
  // every chain rule (k_rw, theta_w, p_c) when caller uses dSe_du_n.
  inline void evaluateCoefficients_from_Se(const int rowptr[nSpace], const int colind[nnz],
                                           const double rho0, const double rho_transport, const double beta,
                                           const double gravity[nSpace],
                                           const double alpha, const double n_vg,
                                           const double thetaR, const double thetaSR,
                                           const double KWs[nnz],
                                           const double &u_w, const double &u_n,
                                           double &m, double &dm_du_w, double &dm_du_n,
                                           double f[nSpace], double df_du_w[nSpace], double df_du_n[nSpace],
                                           double a[nnz], double da_du_w[nnz], double da_du_n[nnz],
                                           double as[nnz],
                                           double &kr, double &dkr_du_w, double &dkr_du_n,
                                           double &thetaW_out)
  {
    const double phi      = thetaR + thetaSR;                    // == thetaS
    const double S_wr     = thetaR / phi;                        // residual S_w
    const double one_m_Sr = 1.0 - S_wr;                          // = thetaSR/phi
    // Se in wetting form, expressed in terms of u_n = S_n.
    const double Se_raw   = (1.0 - u_n - S_wr) / one_m_Sr;
    // at the clips Se is held constant, so dSe/du_n=0.
    // Without this, the closure's DthetaW_DSe (returned regardless of
    // clipping) gets multiplied by +-1/(1-S_wr) and produces a non-zero (0,1)
    // mass Jacobian entry in the infeasible-u_n region, which tricks Newton
    // into overshooting further past saturation/residual.
    double Se, dSe_du_n;
    if (Se_raw <= 0.0)      { Se = 0.0;     dSe_du_n = 0.0; }
    else if (Se_raw >= 1.0) { Se = 1.0;     dSe_du_n = 0.0; }
    else                    { Se = Se_raw;  dSe_du_n = -1.0 / one_m_Sr; }

    double thetaW, DthetaW_DSe, KWr, DKWr_DSe;
    if (PSK_TYPE_member == 1) {
      proteus::m_comp_co2::psk::bc_wetting_from_Se(
          Se, alpha, n_vg, thetaR, thetaSR,
          thetaW, DthetaW_DSe, KWr, DKWr_DSe);
    } else {
      proteus::m_comp_co2::psk::vgm_wetting_from_Se(
          Se, alpha, n_vg, thetaR, thetaSR,
          thetaW, DthetaW_DSe, KWr, DKWr_DSe);
    }
    thetaW_out = thetaW;
    // Density: rho_w(p_w) = rho_transport * exp(beta * p_w); beta in 1/Pa.
    const double rhom  = rho_transport * exp(beta * u_w);
    const double drhom = beta * rhom;
    // component-0 (H2O) accumulation, compositional (p,z) form:
    //   m_0 = phi * N * (1 - z),   N = rho_g*S_g + rho_a*(1-S_g)
    // u_w = p [Pa], u_n = z [-].  Flash gives S_g, rho_a, rho_g + (p,z) derivs.
    // NOTE (P3c complete): the phase-based flux coeffs (f, kr) computed below are
    // NO LONGER used for the H2O residual -- calculateResidual assembles the
    // compositional flux F_0 directly.  `a` (and rhom/thetaW) is retained only
    // for the water-phase Darcy velocity projection (q_velocity) + diagnostics.
    {
      const double p_cl = fmax(u_w, 1.0e2);
      ::m_comp_co2::flash::FlashState fs =
          ::m_comp_co2::flash::flashPZ(p_cl, u_n, T_C_member, m_NaCl_member, ::m_comp_co2::flash::EPS_Z, immiscible_member);
      const double Sa    = 1.0 - fs.S_g;
      const double N     = fs.rho_g*fs.S_g + fs.rho_a*Sa;
      const double dN_dp = fs.drho_g_dp*fs.S_g + fs.rho_g*fs.dS_g_dp
                         + fs.drho_a_dp*Sa     - fs.rho_a*fs.dS_g_dp;
      const double dN_dz = fs.drho_g_dz*fs.S_g + fs.rho_g*fs.dS_g_dz
                         + fs.drho_a_dz*Sa     - fs.rho_a*fs.dS_g_dz;
      m       = phi * N * (1.0 - u_n);
      dm_du_w = phi * (1.0 - u_n) * dN_dp;
      dm_du_n = phi * ((1.0 - u_n) * dN_dz - N);
    }
    // Chain-rule factor for k_rw and downstream Se-derivatives.
    const double DKWr_Du_n = DKWr_DSe * dSe_du_n;
    for (int I = 0; I < nSpace; I++) {
      f[I]       = 0.0;
      df_du_w[I] = 0.0;
      df_du_n[I] = 0.0;
      for (int ii = rowptr[I]; ii < rowptr[I + 1]; ii++) {
        // Diffusion tensor a_w = rho_w * k_rw * (K/mu_w).
        a[ii]       = rhom * KWr * KWs[ii];
        da_du_w[ii] = drhom * KWr * KWs[ii];
        da_du_n[ii] = rhom * DKWr_Du_n * KWs[ii];
        // Gravity flux f_w = rho_w^2 * k_rw * (K/mu_w) * g  (pressure form;
        // no /rho0 factor).
        f[I] += rhom * rhom * KWr * KWs[ii] * gravity[colind[ii]];
        df_du_w[I] += 2.0 * drhom * rhom * KWr * KWs[ii] * gravity[colind[ii]];
        df_du_n[I] += rhom * rhom * DKWr_Du_n * KWs[ii] * gravity[colind[ii]];
        as[ii] = rhom * KWs[ii];
        kr        = KWr;
        dkr_du_w  = 0.0;            // k_rw depends on S_w (= 1-u_n) only
        dkr_du_n  = DKWr_Du_n;
      }
    }
  }

  // ---------------------------------------------------------------------------
  // Component-1 (CO2) per-node closure -- the symmetric analog of
  // evaluateCoefficients_from_Se for the gas/CO2 equation.  Where the comp-0
  // evaluator returns the wetting accumulation + tensor (a,f,kr), this returns
  // the flash+psk PRIMITIVES that BOTH compositional CO2 flux forms are built
  // from, all with analytic d/d{p,z}:
  //   * QP / tensor form  (calculateResidual STAB=0, calculateJacobian):
  //       F_1 = rho_g*Y*u_g + rho_a*X*u_a,
  //       u_a = -(krw*KWs)(grad p - rho_a_mass g),
  //       u_g = -(krn*KWs/mu_n)(grad p + pcp*grad S_a - rho_g_mass g).
  //   * edge / scalar-mobility form  (calculateResidual_entropy_viscosity P2):
  //       F_1 = tau*lam_g_up*gate*dPhi_g + tau*lam_a_up*dPhi_a,
  //       lam_g = (1/mu_n)*rho_g*Y*krn,  lam_a = rho_a*X*krw.
  // Conventions match the verified inline kernels: krn already carries krn_end
  // (so the edge mobility uses cg = 1/mu_n); krw is the bare wetting relperm;
  // KWs = K/mu_w is applied by the caller; gas molar density rho_g is treated
  // as z-independent in rho_g_mass (drho_g/dz ~ 0, as in eftest / P2).
  // Primary vars u_w = p [Pa], u_n = z [-]; PSK_TYPE_member / T_C_member /
  // m_NaCl_member are read from the class state (set at every entry point).
  // FD-verified standalone in comp1_closure_test.cpp; the lam/pc/rgm/ram values
  // match the eftest-verified CO2 closure (COMP_CO2=true) to round-off.
  struct Comp1Closure {
    double S_g, dS_g_dp, dS_g_dz;        // flash gas saturation (gate + grad S_a)
    double N, m, dm_dp, dm_dz;           // total molar density N; accumulation m_1 = phi*N*z
    double krw, dkrw_dp, dkrw_dz;        // wetting relperm (aqueous Darcy)
    double krn, dkrn_dp, dkrn_dz;        // nonwetting relperm * krn_end (gas Darcy)
    double pc, dpc_dp, dpc_dz, pcp;      // capillary pressure; pcp = dp_c/dS_a (QP cross term)
    double rho_g, rho_a, Y, X;           // flash phase props (QP weights)
    double drho_g_dp, drho_g_dz, drho_a_dp, drho_a_dz;
    double dY_dp, dY_dz, dX_dp, dX_dz;
    double rgm, drgm_dp, drgm_dz;        // rho_g_mass = rho_g * Mbar_g  (gravity)
    double ram, dram_dp, dram_dz;        // rho_a_mass = rho_a * Mbar_a  (gravity)
    double lam_g, dlam_g_dp, dlam_g_dz;  // gas molar mobility (1/mu_n)*rho_g*Y*krn (edge)
    double lam_a, dlam_a_dp, dlam_a_dz;  // aqueous molar mobility rho_a*X*krw     (edge)
  };

  inline Comp1Closure evaluateCoefficients_comp1(const double alpha, const double n_vg, const double bc_Se_min,
                                                 const double thetaR, const double thetaSR,
                                                 const double krn_end, const double mu_n,
                                                 const double u_w, const double u_n)
  {
    Comp1Closure o;
    const double phi      = thetaR + thetaSR;                    // == thetaS
    const double S_wr     = thetaR / phi;
    const double one_m_Sr = 1.0 - S_wr;
    const double cg       = 1.0 / mu_n;                          // krn_end folded into krn below
    const double dMm      = ::m_comp_co2::eos::M_CO2_KG - ::m_comp_co2::eos::M_H2O_KG;
    const double p_cl     = fmax(u_w, 1.0e2);
    ::m_comp_co2::flash::FlashState f =
        ::m_comp_co2::flash::flashPZ(p_cl, u_n, T_C_member, m_NaCl_member, ::m_comp_co2::flash::EPS_Z, immiscible_member);
    o.S_g = f.S_g; o.dS_g_dp = f.dS_g_dp; o.dS_g_dz = f.dS_g_dz;
    o.rho_g = f.rho_g; o.rho_a = f.rho_a; o.Y = f.Y; o.X = f.X;
    o.drho_g_dp = f.drho_g_dp; o.drho_g_dz = f.drho_g_dz;
    o.drho_a_dp = f.drho_a_dp; o.drho_a_dz = f.drho_a_dz;
    o.dY_dp = f.dY_dp; o.dY_dz = f.dY_dz; o.dX_dp = f.dX_dp; o.dX_dz = f.dX_dz;
    // accumulation m_1 = phi*N*z, N = rho_g*S_g + rho_a*S_a.
    const double Sa    = 1.0 - f.S_g;
    o.N = f.rho_g*f.S_g + f.rho_a*Sa;
    const double dN_dp = f.drho_g_dp*f.S_g + f.rho_g*f.dS_g_dp + f.drho_a_dp*Sa - f.rho_a*f.dS_g_dp;
    const double dN_dz = f.drho_g_dz*f.S_g + f.rho_g*f.dS_g_dz + f.drho_a_dz*Sa - f.rho_a*f.dS_g_dz;
    o.m     = phi * o.N * u_n;
    o.dm_dp = phi * dN_dp * u_n;
    o.dm_dz = phi * (dN_dz * u_n + o.N);                        // d(phi*N*z)/dz
    // wetting effective saturation from S_a; clipped derivatives.
    const double Se_raw = (Sa - S_wr) / one_m_Sr;
    double Se, dSe_dp, dSe_dz;
    if (Se_raw <= 0.0)      { Se = 0.0; dSe_dp = 0.0; dSe_dz = 0.0; }
    else if (Se_raw >= 1.0) { Se = 1.0; dSe_dp = 0.0; dSe_dz = 0.0; }
    else { Se = Se_raw; dSe_dp = -f.dS_g_dp/one_m_Sr; dSe_dz = -f.dS_g_dz/one_m_Sr; }
    double krn=0,dkrn=0,krw=0,dkrw=0,thW=0,DthW=0,pc=0,dpc_dSe=0,d2pc=0;
    if (PSK_TYPE_member == 1) {
      proteus::m_comp_co2::psk::bc_kr_nonwetting_from_Se(Se, alpha, n_vg, krn, dkrn);
      proteus::m_comp_co2::psk::bc_wetting_from_Se(Se, alpha, n_vg, thetaR, thetaSR, thW, DthW, krw, dkrw);
      proteus::m_comp_co2::psk::bc_pc_from_Se(Se, alpha, n_vg, bc_Se_min, pc, dpc_dSe, d2pc);
    } else {
      proteus::m_comp_co2::psk::vgm_kr_nonwetting_from_Se(Se, alpha, n_vg, krn, dkrn);
      proteus::m_comp_co2::psk::vgm_wetting_from_Se(Se, alpha, n_vg, thetaR, thetaSR, thW, DthW, krw, dkrw);
      proteus::m_comp_co2::psk::vgm_pc_from_Se(Se, alpha, n_vg, pc, dpc_dSe, d2pc);
    }
    krn  *= krn_end;                                             // fold krn_end into the relperm
    dkrn *= krn_end;
    o.krw = krw;  o.dkrw_dp = dkrw*dSe_dp;  o.dkrw_dz = dkrw*dSe_dz;
    o.krn = krn;  o.dkrn_dp = dkrn*dSe_dp;  o.dkrn_dz = dkrn*dSe_dz;
    o.pc  = pc;   o.dpc_dp  = dpc_dSe*dSe_dp; o.dpc_dz = dpc_dSe*dSe_dz;
    o.pcp = dpc_dSe / one_m_Sr;                                  // dp_c/dS_a (QP capillary cross term)
    const double Mbar_g = f.Y*::m_comp_co2::eos::M_CO2_KG + (1.0-f.Y)*::m_comp_co2::eos::M_H2O_KG;
    const double Mbar_a = f.X*::m_comp_co2::eos::M_CO2_KG + (1.0-f.X)*::m_comp_co2::eos::M_H2O_KG;
    o.rgm = f.rho_g*Mbar_g;  o.ram = f.rho_a*Mbar_a;
    o.drgm_dp = f.drho_g_dp*Mbar_g + f.rho_g*f.dY_dp*dMm;
    o.drgm_dz = f.drho_g_dz*Mbar_g + f.rho_g*f.dY_dz*dMm;        // drho_g/dz ~ 0
    o.dram_dp = f.drho_a_dp*Mbar_a + f.rho_a*f.dX_dp*dMm;
    o.dram_dz = f.drho_a_dz*Mbar_a + f.rho_a*f.dX_dz*dMm;
    // gas molar mobility lam_g = cg*rho_g*Y*krn (krn already carries krn_end).
    o.lam_g     = cg*f.rho_g*f.Y*o.krn;
    o.dlam_g_dp = cg*(f.drho_g_dp*f.Y*o.krn + f.rho_g*f.dY_dp*o.krn + f.rho_g*f.Y*o.dkrn_dp);
    o.dlam_g_dz = cg*(f.drho_g_dz*f.Y*o.krn + f.rho_g*f.dY_dz*o.krn + f.rho_g*f.Y*o.dkrn_dz);
    // aqueous molar mobility lam_a = rho_a*X*krw.
    o.lam_a     = f.rho_a*f.X*o.krw;
    o.dlam_a_dp = f.drho_a_dp*f.X*o.krw + f.rho_a*f.dX_dp*o.krw + f.rho_a*f.X*o.dkrw_dp;
    o.dlam_a_dz = f.drho_a_dz*f.X*o.krw + f.rho_a*f.dX_dz*o.krw + f.rho_a*f.X*o.dkrw_dz;
    return o;
  }

  // Inversion routines (vgm_invert_analytic / vgm_invert_newton) are not used by
  // this module; the maintained copies live in proteus/pskRelations.h.

  inline void calculateCFL(const double &elementDiameter, const double df[nSpace], double &cfl)
  {
    double h, nrm_v;
    h     = elementDiameter;
    nrm_v = 0.0;
    for (int I = 0; I < nSpace; I++) nrm_v += df[I] * df[I];
    nrm_v = sqrt(nrm_v);
    cfl   = nrm_v / h;
  }

  inline void calculateSubgridError_tau(const double &elementDiameter, const double &dmt, const double dH[nSpace], double &cfl, double &tau)
  {
    double h, nrm_v, oneByAbsdt;
    h     = elementDiameter;
    nrm_v = 0.0;
    for (int I = 0; I < nSpace; I++) nrm_v += dH[I] * dH[I];
    nrm_v      = sqrt(nrm_v);
    cfl        = nrm_v / h;
    oneByAbsdt = fabs(dmt);
    tau        = 1.0 / (2.0 * nrm_v / h + oneByAbsdt + 1.0e-8);
  }

  inline void calculateSubgridError_tau(const double &Ct_sge, const double G[nSpace * nSpace], const double &A0, const double Ai[nSpace], double &tau_v, double &cfl)
  {
    double v_d_Gv = 0.0;
    for (int I = 0; I < nSpace; I++)
      for (int J = 0; J < nSpace; J++) v_d_Gv += Ai[I] * G[I * nSpace + J] * Ai[J];
    tau_v = 1.0 / sqrt(Ct_sge * A0 * A0 + v_d_Gv);
  }

  inline void calculateNumericalDiffusion(const double &shockCapturingDiffusion, const double &elementDiameter, const double &strong_residual, const double grad_u[nSpace], double &numDiff)
  {
    double h, num, den, n_grad_u;
    h        = elementDiameter;
    n_grad_u = 0.0;
    for (int I = 0; I < nSpace; I++) n_grad_u += grad_u[I] * grad_u[I];
    num     = shockCapturingDiffusion * 0.5 * h * fabs(strong_residual);
    den     = sqrt(n_grad_u) + 1.0e-8;
    numDiff = num / den;
  }

  inline void exteriorNumericalFlux(const double &bc_flux, int rowptr[nSpace], int colind[nnz], int isSeepageFace, int &isDOFBoundary, double n[nSpace], double bc_u, double K[nnz], double grad_psi[nSpace], double u, double K_rho_g[nSpace], double penalty, double &flux)
  {
    double v_I, bc_u_seepage = 0.0;
    if (isSeepageFace || isDOFBoundary) {
      flux = 0.0;
      for (int I = 0; I < nSpace; I++) {
        //gravity
        v_I = K_rho_g[I];
        //pressure head
        for (int m = rowptr[I]; m < rowptr[I + 1]; m++) { v_I -= K[m] * grad_psi[colind[m]]; }
        flux += v_I * n[I];
      }
      if (isSeepageFace) bc_u = bc_u_seepage;
      flux += penalty * (u - bc_u);
      //flux -= penalty * bc_u;
      if (isSeepageFace) {
        if (flux > 0.0) {
          isDOFBoundary = 1;
          bc_u          = bc_u_seepage;
        } else {
          isDOFBoundary = 0;
          flux          = 0.0;
        }
      }
    } else flux = bc_flux;
  }

  void exteriorNumericalFluxJacobian(const int rowptr[nSpace], const int colind[nnz], const int isDOFBoundary, const double n[nSpace], const double K[nnz], const double dK[nnz], const double grad_psi[nSpace], const double grad_v[nSpace], const double dK_rho_g[nSpace], const double v, const double penalty, double &fluxJacobian)
  {
    if (isDOFBoundary) {
      fluxJacobian = 0.0;
      for (int I = 0; I < nSpace; I++) {
        //gravity
        fluxJacobian += dK_rho_g[I] * v * n[I];
        //pressure head
        for (int m = rowptr[I]; m < rowptr[I + 1]; m++) { fluxJacobian -= (K[m] * grad_v[colind[m]] + dK[m] * v * grad_psi[colind[m]]) * n[I]; }
      }
      //Dirichlet penalty
      fluxJacobian += penalty * v;
    } else fluxJacobian = 0.0;
  }

inline void exteriorNumericalFlux2(const double &bc_flux, int rowptr[nSpace], int colind[nnz], int isSeepageFace, int &isDOFBoundary, double n[nSpace], double bc_u, double K[nnz], double grad_psi[nSpace], double u, double K_rho_g[nSpace], double penalty, double &flux, double &bflux)
  {
    double v_I, bc_u_seepage = 0.0;
    if (isSeepageFace || isDOFBoundary) {
      flux = 0.0;
      bflux = 0.0;
      for (int I = 0; I < nSpace; I++) {
        //gravity
        v_I = K_rho_g[I];
        //pressure head
        for (int m = rowptr[I]; m < rowptr[I + 1]; m++) { v_I -= K[m] * grad_psi[colind[m]]; }
        flux += v_I * n[I];
      }
      if (isSeepageFace) bc_u = bc_u_seepage;
      flux += penalty * (u - bc_u);
      bflux += penalty * (u - bc_u);
      if (isSeepageFace) {
        if (flux > 0.0) {
          isDOFBoundary = 1;
        } else {
          isDOFBoundary = 0;
          flux          = 0.0;
          bflux         = 0.0;
        }
      }
    } else {
      flux = bc_flux;
      bflux = bc_flux;
    }
  }

  void exteriorNumericalFluxJacobian2(const int rowptr[nSpace], const int colind[nnz], const int isDOFBoundary, const double n[nSpace],  const double Ks[nnz], const double K[nnz], const double dK[nnz], const double grad_psi[nSpace], const double grad_v[nSpace], const double dK_rho_g[nSpace], const double v, const double penalty, double &fluxJacobian, double &bfluxJacobian)
  {
    if (isDOFBoundary) {
      fluxJacobian = 0.0;
      bfluxJacobian = 0.0;
      for (int I = 0; I < nSpace; I++) {
        for (int m = rowptr[I]; m < rowptr[I + 1]; m++) { 
          fluxJacobian -= Ks[m] * grad_v[colind[m]] * n[I]; 
        }
      }
      //Dirichlet penalty
      bfluxJacobian = penalty * v;
    } else {
      fluxJacobian = 0.0;
      bfluxJacobian = 0.0;
    }
  }

  double seepagefluxcalculator(double anb_seepage_flux, int isSeepageFace, double dS, double flux_ext)
  {
    if (isSeepageFace) { anb_seepage_flux += flux_ext * dS; }
    return anb_seepage_flux;
  }

  void calculateResidual(arguments_dict &args)
  {
    proteus::pyarray<double> &mesh_trial_ref                             = args.array<double>("mesh_trial_ref");
    proteus::pyarray<double> &mesh_grad_trial_ref                        = args.array<double>("mesh_grad_trial_ref");
    proteus::pyarray<double> &mesh_dof                                   = args.array<double>("mesh_dof");
    proteus::pyarray<double> &mesh_velocity_dof                          = args.array<double>("mesh_velocity_dof");
    double               MOVING_DOMAIN                              = args.scalar<double>("MOVING_DOMAIN");
    proteus::pyarray<int>    &mesh_l2g                                   = args.array<int>("mesh_l2g");
    proteus::pyarray<double> &dV_ref                                     = args.array<double>("dV_ref");
    proteus::pyarray<double> &u_trial_ref                                = args.array<double>("u_trial_ref");
    proteus::pyarray<double> &u_grad_trial_ref                           = args.array<double>("u_grad_trial_ref");
    proteus::pyarray<double> &u_test_ref                                 = args.array<double>("u_test_ref");
    proteus::pyarray<double> &u_grad_test_ref                            = args.array<double>("u_grad_test_ref");
    proteus::pyarray<double> &mesh_trial_trace_ref                       = args.array<double>("mesh_trial_trace_ref");
    proteus::pyarray<double> &mesh_grad_trial_trace_ref                  = args.array<double>("mesh_grad_trial_trace_ref");
    proteus::pyarray<double> &dS_ref                                     = args.array<double>("dS_ref");
    proteus::pyarray<double> &u_trial_trace_ref                          = args.array<double>("u_trial_trace_ref");
    proteus::pyarray<double> &u_grad_trial_trace_ref                     = args.array<double>("u_grad_trial_trace_ref");
    proteus::pyarray<double> &u_test_trace_ref                           = args.array<double>("u_test_trace_ref");
    proteus::pyarray<double> &u_grad_test_trace_ref                      = args.array<double>("u_grad_test_trace_ref");
    proteus::pyarray<double> &normal_ref                                 = args.array<double>("normal_ref");
    proteus::pyarray<double> &boundaryJac_ref                            = args.array<double>("boundaryJac_ref");
    int                  nElements_global                           = args.scalar<int>("nElements_global");
    proteus::pyarray<double> &ebqe_penalty_ext                           = args.array<double>("ebqe_penalty_ext");
    proteus::pyarray<int>    &elementMaterialTypes                       = args.array<int>("elementMaterialTypes");
    proteus::pyarray<int>    &isSeepageFace                              = args.array<int>("isSeepageFace");
    proteus::pyarray<int>    &a_rowptr                                   = args.array<int>("a_rowptr");
    proteus::pyarray<int>    &a_colind                                   = args.array<int>("a_colind");
    double               rho                                        = args.scalar<double>("rho");
    double               beta                                       = args.scalar<double>("beta");

    /////////////////////////////DENSITY COUPLING  >>>> USE rho from mprans model/////////////////////////
    proteus::pyarray<double> &q_rho                                     = args.array<double>("q_rho");
    proteus::pyarray<double> &ebqe_rho                                  = args.array<double>("ebqe_rho");

    proteus::pyarray<double> &gravity                                    = args.array<double>("gravity");
    proteus::pyarray<double> &alpha                                      = args.array<double>("alpha");
    proteus::pyarray<double> &n                                          = args.array<double>("n");
    proteus::pyarray<double> &thetaR                                     = args.array<double>("thetaR");
    proteus::pyarray<double> &thetaSR                                    = args.array<double>("thetaSR");
    proteus::pyarray<double> &KWs                                        = args.array<double>("KWs");
    proteus::pyarray<double> &krn_end                                    = args.array<double>("krn_end");
    proteus::pyarray<double> &S_gr                                       = args.array<double>("S_gr");
    proteus::pyarray<double> &bc_Se_min                                  = args.array<double>("bc_Se_min");
    double               mu_n                                       = args.scalar<double>("mu_n");
    double               useMetrics                                 = args.scalar<double>("useMetrics");
    double               alphaBDF                                   = args.scalar<double>("alphaBDF");
    int                  lag_shockCapturing                         = args.scalar<int>("lag_shockCapturing");
    double               shockCapturingDiffusion                    = args.scalar<double>("shockCapturingDiffusion");
    double               sc_uref                                    = args.scalar<double>("sc_uref");
    double               sc_alpha                                   = args.scalar<double>("sc_alpha");
    proteus::pyarray<int>    &u_l2g                                      = args.array<int>("u_l2g");
    proteus::pyarray<double> &elementDiameter                            = args.array<double>("elementDiameter");
    proteus::pyarray<double> &u_dof                                      = args.array<double>("u_dof");
    proteus::pyarray<double> &u_dof_old                                  = args.array<double>("u_dof_old");
    proteus::pyarray<double> &velocity                                   = args.array<double>("velocity");
    proteus::pyarray<double> &q_m                                        = args.array<double>("q_m");
    proteus::pyarray<double> &q_theta                                    = args.array<double>("q_theta");
    proteus::pyarray<double> &q_u                                        = args.array<double>("q_u");
    proteus::pyarray<double> &q_dV                                       = args.array<double>("q_dV");
    proteus::pyarray<double> &q_m_betaBDF                                = args.array<double>("q_m_betaBDF");
    proteus::pyarray<double> &cfl                                        = args.array<double>("cfl");
    proteus::pyarray<double> &q_numDiff_u                                = args.array<double>("q_numDiff_u");
    proteus::pyarray<double> &q_numDiff_u_last                           = args.array<double>("q_numDiff_u_last");
    int                  offset_u                                   = args.scalar<int>("offset_u");
    int                  stride_u                                   = args.scalar<int>("stride_u");
    // component-1 (S_n) mass equation args.
    // Used in the dedicated component-1 element loop appended at the end
    // of this function. Not consumed by the existing component-0 logic.
    const double         dt                                         = args.scalar<double>("dt");
    proteus::pyarray<double> &u_dof_n                                    = args.array<double>("u_dof_n");
    proteus::pyarray<double> &u_dof_n_old                                = args.array<double>("u_dof_n_old");
    // gas-phase density (linear EOS). rho_n is the reference density and
    // p_ref_n the reference pressure: rho_n_local(p_n) = rho_n*p_n/p_ref_n
    // when p_ref_n > 0; constant rho_n otherwise.
    const double         rho_n                                      = args.scalar<double>("rho_n");
    const double         p_ref_n                                    = args.scalar<double>("p_ref_n");
    const bool           rho_n_compressible                         = (p_ref_n > 0.0);
    const double         c_n                                        = rho_n_compressible ? (rho_n / p_ref_n) : 0.0;
    const int            offset_n                                   = args.scalar<int>("offset_n");
    const int            stride_n                                   = args.scalar<int>("stride_n");
    // Stage 3b: gas-side kinetic dissolution sink.  R_diss = k_d * S_n *
    // (1 - S_n) * theta_w * rho_w(c) * (c_sat - c) is subtracted from the
    // gas-equation residual at each quadrature point.  c is read from TADR's
    // u[0].dof aliased Python-side and passed in as c_dof.  k_d=0 disables
    // the sink (legacy behavior).
    proteus::pyarray<double> &c_dof                                      = args.array<double>("c_dof");
    const double         k_d                                        = args.scalar<double>("k_d");
    const double         c_sat                                      = args.scalar<double>("c_sat");
    // CO2 injection: per-node source field (built Python-side, schedule-gated).
    // Applied like R_diss but with opposite sign -- a source, not a sink.
    // All-zero array when no injection is configured.
    proteus::pyarray<double> &injection_dof                              = args.array<double>("injection_dof");
    proteus::pyarray<double> &globalResidual                             = args.array<double>("globalResidual");
    int                  nExteriorElementBoundaries_global          = args.scalar<int>("nExteriorElementBoundaries_global");
    proteus::pyarray<int>    &exteriorElementBoundariesArray             = args.array<int>("exteriorElementBoundariesArray");
    proteus::pyarray<int>    &elementBoundaryElementsArray               = args.array<int>("elementBoundaryElementsArray");
    proteus::pyarray<int>    &elementBoundaryLocalElementBoundariesArray = args.array<int>("elementBoundaryLocalElementBoundariesArray");
    proteus::pyarray<double> &ebqe_velocity_ext                          = args.array<double>("ebqe_velocity_ext");
    proteus::pyarray<int>    &isDOFBoundary_u                            = args.array<int>("isDOFBoundary_u");
    proteus::pyarray<double> &ebqe_bc_u_ext                              = args.array<double>("ebqe_bc_u_ext");
    // component-1 (S_n) boundary arrays.
    proteus::pyarray<int>    &isDOFBoundary_n                            = args.array<int>("isDOFBoundary_n");
    proteus::pyarray<double> &ebqe_bc_u_n_ext                            = args.array<double>("ebqe_bc_u_n_ext");
    proteus::pyarray<int>    &isFluxBoundary_u                           = args.array<int>("isFluxBoundary_u");
    proteus::pyarray<double> &ebqe_bc_flux_ext                           = args.array<double>("ebqe_bc_flux_ext");
    proteus::pyarray<double> &ebqe_phi                                   = args.array<double>("ebqe_phi");
    double               epsFact                                    = args.scalar<double>("epsFact");
    proteus::pyarray<double> &ebqe_u                                     = args.array<double>("ebqe_u");
    proteus::pyarray<double> &ebqe_theta                                 = args.array<double>("ebqe_theta");
    proteus::pyarray<double> &ebqe_flux                                  = args.array<double>("ebqe_flux");
    // VMS
    double VMS = args.scalar<double>("VMS");
    // PARAMETERS FOR EDGE BASED STABILIZATION
    double cE = args.scalar<double>("cE");
    double cK = args.scalar<double>("cK");
    // PARAMETERS FOR LOG BASED ENTROPY FUNCTION
    double uL = args.scalar<double>("uL");
    double uR = args.scalar<double>("uR");
    // PARAMETERS FOR EDGE VISCOSITY
    int               numDOFs                       = args.scalar<int>("numDOFs");
    // numDOFs is the compact component-0 free-DOF count used by the stabilized
    // DOF loops. Full-matrix slots are recovered from offset/stride-aware CSR
    // indexing against the interleaved global matrix.
    int               numDOFs_u                     = args.scalar<int>("numDOFs_u");
    int               NNZ                           = args.scalar<int>("NNZ");
    proteus::pyarray<int> &csrRowIndeces_DofLoops        = args.array<int>("csrRowIndeces_DofLoops");
    proteus::pyarray<int> &csrColumnOffsets_DofLoops     = args.array<int>("csrColumnOffsets_DofLoops");
    proteus::pyarray<int> &csrRowIndeces_Full            = args.array<int>("csrRowIndeces_Full");
    proteus::pyarray<int> &csrColumnOffsets_Full         = args.array<int>("csrColumnOffsets_Full");
    proteus::pyarray<int> &csrRowIndeces_CellLoops       = args.array<int>("csrRowIndeces_CellLoops");
    proteus::pyarray<int> &csrColumnOffsets_CellLoops    = args.array<int>("csrColumnOffsets_CellLoops");
    proteus::pyarray<int> &csrColumnOffsets_eb_CellLoops = args.array<int>("csrColumnOffsets_eb_CellLoops");
    // C matrices
    proteus::pyarray<double> &Cx         = args.array<double>("Cx");
    proteus::pyarray<double> &Cy         = args.array<double>("Cy");
    proteus::pyarray<double> &Cz         = args.array<double>("Cz");
    proteus::pyarray<double> &CTx        = args.array<double>("CTx");
    proteus::pyarray<double> &CTy        = args.array<double>("CTy");
    proteus::pyarray<double> &CTz        = args.array<double>("CTz");
    proteus::pyarray<double> &ML         = args.array<double>("ML");
    proteus::pyarray<double> &delta_x_ij = args.array<double>("delta_x_ij");
    // PARAMETERS FOR 1st or 2nd ORDER MPP METHOD
    int LUMPED_MASS_MATRIX = args.scalar<int>("LUMPED_MASS_MATRIX");
    STABILIZATION STABILIZATION_TYPE{static_cast<STABILIZATION>(args.scalar<int>("STABILIZATION_TYPE"))};
    int ENTROPY_TYPE = args.scalar<int>("ENTROPY_TYPE");
    // PSK closure selector for evaluateCoefficients (read from argsDict).
    PSK_TYPE_member = args.scalar<int>("PSK_TYPE");
    immiscible_member = (args.scalar<int>("immiscible") != 0);
    T_C_member        = args.scalar<double>("T_C");      // temperature [degC] from input
    // FOR FCT
    proteus::pyarray<double> &dLow                 = args.array<double>("dLow");
    proteus::pyarray<double> &fluxMatrix           = args.array<double>("fluxMatrix");
    // AUX QUANTITIES OF INTEREST
    proteus::pyarray<double> &quantDOFs = args.array<double>("quantDOFs");

    assert(a_rowptr.data()[nSpace] == nnz);
    assert(a_rowptr.data()[nSpace] == nSpace);
    //cek should this be read in?
    double Ct_sge = 4.0;

    proteus::pyarray<double> &anb_seepage_flux_n = args.array<double>("anb_seepage_flux_n");

    proteus::pyarray<double> &velocity_couple                            = args.array<double>("velocity_couple");
    proteus::pyarray<double> &ebqe_velocity_ext_couple                          = args.array<double>("ebqe_velocity_ext_couple");
    
    // proteus::pyarray<double> &q_x    = args.array<double>("q_x");
    // proteus::pyarray<double> &ebqe_x = args.array<double>("ebqe_x");

    //double anb_seepage_flux=0.0;
    double &anb_seepage_flux(args.scalar<double>("anb_seepage_flux"));
    proteus::pyarray<double> &q_velocity = args.array<double>("q_velocity");
    anb_seepage_flux = 0.0;

    //loop over elements to compute volume integrals and load them into element and global residual
    //
    //eN is the element index
    //eN_k is the quadrature point index for a scalar
    //eN_k_nSpace is the quadrature point index for a vector
    //eN_i is the element test function index
    //eN_j is the element trial function index
    //eN_k_j is the quadrature point index for a trial function
    //eN_k_i is the quadrature point index for a trial function
    for (int eN = 0; eN < nElements_global; eN++) {
      //declare local storage for element residual and initialize
      double elementResidual_u[nDOF_test_element];
      for (int i = 0; i < nDOF_test_element; i++) { elementResidual_u[i] = 0.0; } //i
      //loop over quadrature points and compute integrands
      for (int k = 0; k < nQuadraturePoints_element; k++) {
        //compute indeces and declare local storage
        int eN_k = eN * nQuadraturePoints_element + k, eN_k_nSpace = eN_k * nSpace, eN_nDOF_trial_element = eN * nDOF_trial_element;
        double u = 0.0, grad_u[nSpace], grad_u_old[nSpace], m = 0.0, dm = 0.0, f[nSpace], df[nSpace], a[nnz], da[nnz], as[nnz], m_t = 0.0, dm_t = 0.0, pdeResidual_u = 0.0, Lstar_u[nDOF_test_element], subgridError_u = 0.0, tau = 0.0, tau0 = 0.0, tau1 = 0.0, numDiff0 = 0.0, numDiff1 = 0.0, jac[nSpace * nSpace], jacDet, jacInv[nSpace * nSpace], u_grad_trial[nDOF_trial_element * nSpace], u_test_dV[nDOF_trial_element], u_grad_test_dV[nDOF_test_element * nSpace], dV, x, y, z, xt, yt, zt, G[nSpace * nSpace], G_dd_G, tr_G, norm_Rv;
        //
        //compute solution and gradients at quadrature points
        //
        ck.calculateMapping_element(eN, k, mesh_dof.data(), mesh_l2g.data(), mesh_trial_ref.data(), mesh_grad_trial_ref.data(), jac, jacDet, jacInv, x, y, z);
        ck.calculateMappingVelocity_element(eN, k, mesh_velocity_dof.data(), mesh_l2g.data(), mesh_trial_ref.data(), xt, yt, zt);
        //get the physical integration weight
        dV                = fabs(jacDet) * dV_ref.data()[k];
        q_dV.data()[eN_k] = dV;
        ck.calculateG(jacInv, G, G_dd_G, tr_G);
        //get the trial function gradients
        ck.gradTrialFromRef(&u_grad_trial_ref.data()[k * nDOF_trial_element * nSpace], jacInv, u_grad_trial);
        //get the solution
        ck.valFromDOF(u_dof.data(), &u_l2g.data()[eN_nDOF_trial_element], &u_trial_ref.data()[k * nDOF_trial_element], u);
        //get the solution gradients
        ck.gradFromDOF(u_dof.data(), &u_l2g.data()[eN_nDOF_trial_element], u_grad_trial, grad_u);
        // P3c: gradient of z (comp-1 DOF) at this QP -- the comp-0 component flux
        // needs grad S_a = -(dSg/dp grad p + dSg/dz grad z).
        double grad_u_n[nSpace];
        ck.gradFromDOF(u_dof_n.data(), &u_l2g.data()[eN_nDOF_trial_element], u_grad_trial, grad_u_n);

        // //populate q_x
        // const int eN_k_3d = eN_k * 3;
        // q_x.data()[eN_k_3d + 0] = x;
        // q_x.data()[eN_k_3d + 1] = y;
        // q_x.data()[eN_k_3d + 2] = z;      
        
        //precalculate test function products with integration weights
        for (int j = 0; j < nDOF_trial_element; j++) {
          u_test_dV[j] = u_test_ref.data()[k * nDOF_trial_element + j] * dV;
          for (int I = 0; I < nSpace; I++) {
            u_grad_test_dV[j * nSpace + I] = u_grad_trial[j * nSpace + I] * dV; //cek warning won't work for Petrov-Galerkin
          }
        }
        //
        //calculate pde coefficients at quadrature points
        //
        double Kr, dKr, thetaW;
        const double rho_local = q_rho.data()[eN_k];
        const double rho_velocity = std::fabs(rho_local) > 1.0e-12 ? rho_local : rho;
        // Cross-derivatives (dm_du_n, df_du_n, da_du_n, dkr_du_n) are unused in
        // the residual; the (0,1) Jacobian cross-block consumes them elsewhere.
        double dm_du_n_qp = 0.0, dkr_du_n_qp = 0.0;
        double df_du_n_qp[nSpace];
        double da_du_n_qp[nnz];
        for (int I = 0; I < nSpace; I++) df_du_n_qp[I] = 0.0;
        for (int ii = 0; ii < nnz; ii++) da_du_n_qp[ii] = 0.0;
        double u_n_qp = 0.0;
        ck.valFromDOF(u_dof_n.data(),
                      &u_l2g.data()[eN_nDOF_trial_element],
                      &u_trial_ref.data()[k * nDOF_trial_element], u_n_qp);
        evaluateCoefficients_from_Se(a_rowptr.data(), a_colind.data(), rho, rho_local, beta, gravity.data(),
                                     alpha.data()[elementMaterialTypes.data()[eN]], n.data()[elementMaterialTypes.data()[eN]],
                                     thetaR.data()[elementMaterialTypes.data()[eN]], thetaSR.data()[elementMaterialTypes.data()[eN]],
                                     &KWs.data()[elementMaterialTypes.data()[eN] * nnz], u, u_n_qp,
                                     m, dm, dm_du_n_qp, f, df, df_du_n_qp, a, da, da_du_n_qp,
                                     as, Kr, dKr, dkr_du_n_qp, thetaW);
        q_theta.data()[eN_k] = thetaW;
        

        for (int I = 0; I < nSpace; ++I) {
          q_velocity.data()[eN_k_nSpace + I] = grad_u[I];
        }
        // Darcy Velocity
        double pressure_gradient[nSpace];
        for (int J=0; J<nSpace; ++J)
          pressure_gradient[J] = grad_u[J] - rho_velocity * gravity.data()[J];
        // q_w = -(a/rho_w) * (grad p_w - rho_w g) = -(k_rw K/mu_w) * (grad p_w - rho_w g).
        for (int I=0; I<nSpace; ++I) {
          double acc = 0.0;
          for (int ii = a_rowptr.data()[I]; ii < a_rowptr.data()[I+1]; ++ii) {
            const int J = a_colind.data()[ii];
            acc += (a[ii] / rho_velocity) * pressure_gradient[J];
          }
          velocity.data()[eN_k_nSpace + I] = -acc;
          velocity_couple.data()[eN_k_nSpace + I] = -acc ;
          }
        //
        // Compositional H2O accumulation m_0 = phi*N*(1-z), N = rho_g*S_g +
        // rho_a*S_a (overrides the phase-based m from evaluateCoefficients_from_Se;
        // (p,z) formulation -- consistent with the (0,0)/(0,1) mass Jacobian).
        {
          const int    mat0   = elementMaterialTypes.data()[eN];
          const double phi0_m = thetaR.data()[mat0] + thetaSR.data()[mat0];
          const double p0_m   = fmax(u, 1.0e2);
          ::m_comp_co2::flash::FlashState fsm =
              ::m_comp_co2::flash::flashPZ(p0_m, u_n_qp, T_C_member, m_NaCl_member, ::m_comp_co2::flash::EPS_Z, immiscible_member);
          const double Nm     = fsm.rho_g*fsm.S_g + fsm.rho_a*(1.0 - fsm.S_g);
          const double dNm_dp = fsm.drho_g_dp*fsm.S_g + fsm.rho_g*fsm.dS_g_dp
                              + fsm.drho_a_dp*(1.0-fsm.S_g) - fsm.rho_a*fsm.dS_g_dp;
          m  = phi0_m * Nm * (1.0 - u_n_qp);
          dm = phi0_m * dNm_dp * (1.0 - u_n_qp);   // d m_0/dp (used by bdf/subgrid only)
        }
        //
        //calculate time derivative at quadrature points
        //
        ck.bdf(alphaBDF, q_m_betaBDF.data()[eN_k], m, dm, m_t, dm_t);
        //
        //calculate subgrid error (strong residual and adjoint)
        //
        //calculate strong residual
        // pdeResidual_u = ck.Mass_strong(m_t) + ck.Advection_strong(df, grad_u);
        // //calculate adjoint
        // for (int i = 0; i < nDOF_test_element; i++) {
        //   int i_nSpace = i * nSpace;
        //   Lstar_u[i]   = ck.Advection_adjoint(df, &u_grad_test_dV[i_nSpace]);
        // }
        // //calculate tau and tau*Res
        // calculateSubgridError_tau(elementDiameter[eN], dm_t, df, cfl[eN_k], tau0);
        // calculateSubgridError_tau(Ct_sge, G, dm_t, df, tau1, cfl[eN_k]);

        // tau = useMetrics * tau1 + (1.0 - useMetrics) * tau0;

        // subgridError_u = -tau * pdeResidual_u;
        // //
        // //calculate shock capturing diffusion
        // //
        // ck.calculateNumericalDiffusion(shockCapturingDiffusion, elementDiameter[eN], pdeResidual_u, grad_u, numDiff0);
        // ck.calculateNumericalDiffusion(shockCapturingDiffusion, sc_uref, sc_alpha, G, G_dd_G, pdeResidual_u, grad_u, numDiff1);
        // q_numDiff_u[eN_k] = useMetrics * numDiff1 + (1.0 - useMetrics) * numDiff0;
        // ===== P3c: component H2O molar flux  F_0 = rho_g*(1-Y)*u_g + rho_a*(1-X)*u_a =====
        // Same two phase Darcy velocities as the comp-1 flux, weighted by the H2O
        // mole fractions (1-Y) in gas, (1-X) in aqueous.  Props from the flash
        // saturation S_g (S_a = 1 - S_g); psk closures take wetting Se_a0.
        const int    mat_eN0   = elementMaterialTypes.data()[eN];
        const double alpha_eN0 = alpha.data()[mat_eN0];
        const double n_vg_eN0  = n.data()[mat_eN0];
        const double krn_end0  = krn_end.data()[mat_eN0];
        const double *KWs_eN0  = &KWs.data()[mat_eN0 * nnz];
        const double phi0      = thetaR.data()[mat_eN0] + thetaSR.data()[mat_eN0];
        const double S_wr0     = thetaR.data()[mat_eN0] / phi0;
        const double one_m_Sr0 = 1.0 - S_wr0;
        const double Se_trap_L771 = 1.0 - S_gr.data()[mat_eN0] / one_m_Sr0;  // gas-only residual trapping
        const double p_cl0     = fmax(u, 1.0e2);
        ::m_comp_co2::flash::FlashState fs0 =
            ::m_comp_co2::flash::flashPZ(p_cl0, u_n_qp, T_C_member, m_NaCl_member, ::m_comp_co2::flash::EPS_Z, immiscible_member);
        const double Se_a0 = fmin(fmax((1.0 - fs0.S_g - S_wr0) / one_m_Sr0, 0.0), 1.0);
        double KWr0 = 0.0, DKWr0 = 0.0, thW0 = 0.0, DthW0 = 0.0;
        double KNr0 = 0.0, DKNr0 = 0.0, pc0 = 0.0, dpc_dSe0 = 0.0, d2pc0 = 0.0;
        if (PSK_TYPE_member == 1) {
          proteus::m_comp_co2::psk::bc_wetting_from_Se(Se_a0, alpha_eN0, n_vg_eN0,
              thetaR.data()[mat_eN0], thetaSR.data()[mat_eN0], thW0, DthW0, KWr0, DKWr0);
          proteus::m_comp_co2::psk::bc_kr_nonwetting_from_Se(Se_a0, alpha_eN0, n_vg_eN0, KNr0, DKNr0, Se_trap_L771);
          proteus::m_comp_co2::psk::bc_pc_from_Se(Se_a0, alpha_eN0, n_vg_eN0, bc_Se_min.data()[mat_eN0], pc0, dpc_dSe0, d2pc0);
        } else {
          proteus::m_comp_co2::psk::vgm_wetting_from_Se(Se_a0, alpha_eN0, n_vg_eN0,
              thetaR.data()[mat_eN0], thetaSR.data()[mat_eN0], thW0, DthW0, KWr0, DKWr0);
          proteus::m_comp_co2::psk::vgm_kr_nonwetting_from_Se(Se_a0, alpha_eN0, n_vg_eN0, KNr0, DKNr0, Se_trap_L771);
          proteus::m_comp_co2::psk::vgm_pc_from_Se(Se_a0, alpha_eN0, n_vg_eN0, pc0, dpc_dSe0, d2pc0);
        }
        KNr0 *= krn_end0;
        const double pcp0 = dpc_dSe0 / one_m_Sr0;
        const double Mbar_g0 = fs0.Y*::m_comp_co2::eos::M_CO2_KG + (1.0 - fs0.Y)*::m_comp_co2::eos::M_H2O_KG;
        const double Mbar_a0 = fs0.X*::m_comp_co2::eos::M_CO2_KG + (1.0 - fs0.X)*::m_comp_co2::eos::M_H2O_KG;
        const double rho_g_mass0 = fs0.rho_g*Mbar_g0;
        const double rho_a_mass0 = fs0.rho_a*Mbar_a0;
        double F0[nSpace];
        for (int I = 0; I < nSpace; I++) {
          double ua = 0.0, ug = 0.0;
          for (int ii = a_rowptr.data()[I]; ii < a_rowptr.data()[I + 1]; ii++) {
            const int J = a_colind.data()[ii];
            const double gradSa_J = -(fs0.dS_g_dp*grad_u[J] + fs0.dS_g_dz*grad_u_n[J]);
            const double gp_a = grad_u[J]                  - rho_a_mass0*gravity.data()[J];
            const double gp_g = grad_u[J] + pcp0*gradSa_J  - rho_g_mass0*gravity.data()[J];
            ua -= (KWr0*KWs_eN0[ii])      * gp_a;
            ug -= (KNr0*KWs_eN0[ii]/mu_n) * gp_g;
          }
          F0[I] = fs0.rho_g*(1.0 - fs0.Y)*ug + fs0.rho_a*(1.0 - fs0.X)*ua;
        }
        //
        //update element residual
        //
        for (int i = 0; i < nDOF_test_element; i++) {
          int eN_k_i = eN_k * nDOF_test_element + i, eN_k_i_nSpace = eN_k_i * nSpace, i_nSpace = i * nSpace;
          // P3c: mass + component flux (divergence form, -= F_0 . grad N_i).
          elementResidual_u[i] += ck.Mass_weak(m_t, u_test_dV[i]) + VMS * ck.SubgridError(subgridError_u, Lstar_u[i]) + VMS * ck.NumericalDiffusion(q_numDiff_u_last[eN_k], grad_u, &u_grad_test_dV[i_nSpace]);
          for (int I = 0; I < nSpace; I++)
            elementResidual_u[i] -= F0[I] * u_grad_test_dV[i_nSpace + I];
        } //i
        //
        q_m.data()[eN_k] = m;
        q_u.data()[eN_k] = u;
      }
      //
      //load element into global residual and save element residual
      //
      for (int i = 0; i < nDOF_test_element; i++) {
        int eN_i = eN * nDOF_test_element + i;

        globalResidual.data()[offset_u + stride_u * u_l2g.data()[eN_i]] += elementResidual_u[i];
      } //i
    } //elements
    //
    //loop over exterior element boundaries to calculate surface integrals and load into element and global residuals
    //
    //ebNE is the Exterior element boundary INdex
    //ebN is the element boundary INdex
    //eN is the element index
    for (int ebNE = 0; ebNE < nExteriorElementBoundaries_global; ebNE++) {
      int    ebN = exteriorElementBoundariesArray.data()[ebNE], eN = elementBoundaryElementsArray.data()[ebN * 2 + 0], ebN_local = elementBoundaryLocalElementBoundariesArray.data()[ebN * 2 + 0], eN_nDOF_trial_element = eN * nDOF_trial_element;
      double elementResidual_u[nDOF_test_element];
      for (int i = 0; i < nDOF_test_element; i++) { elementResidual_u[i] = 0.0; }
      for (int kb = 0; kb < nQuadraturePoints_elementBoundary; kb++) {
        int    ebNE_kb = ebNE * nQuadraturePoints_elementBoundary + kb, ebNE_kb_nSpace = ebNE_kb * nSpace, ebN_local_kb = ebN_local * nQuadraturePoints_elementBoundary + kb, ebN_local_kb_nSpace = ebN_local_kb * nSpace;
        double u_ext = 0.0, grad_u_ext[nSpace], m_ext = 0.0, dm_ext = 0.0, f_ext[nSpace], df_ext[nSpace], a_ext[nnz], da_ext[nnz], as_ext[nnz], flux_ext = 0.0,
               //anb_seepage_flux=0.0, // for flux calculation
          bc_u_ext = 0.0, bc_grad_u_ext[nSpace], bc_m_ext = 0.0, bc_dm_ext = 0.0, bc_f_ext[nSpace], bc_df_ext[nSpace], bc_a_ext[nnz], bc_da_ext[nnz], bc_as_ext[nnz], jac_ext[nSpace * nSpace], jacDet_ext, jacInv_ext[nSpace * nSpace], boundaryJac[nSpace * (nSpace - 1)], metricTensor[(nSpace - 1) * (nSpace - 1)], metricTensorDetSqrt, dS, u_test_dS[nDOF_test_element], u_grad_trial_trace[nDOF_trial_element * nSpace], normal[3], x_ext, y_ext, z_ext, xt_ext, yt_ext, zt_ext, integralScaling, G[nSpace * nSpace], G_dd_G, tr_G;
        //
        //calculate the solution and gradients at quadrature points
        //
        //compute information about mapping from reference element to physical element
        ck.calculateMapping_elementBoundary(eN, ebN_local, kb, ebN_local_kb, mesh_dof.data(), mesh_l2g.data(), mesh_trial_trace_ref.data(), mesh_grad_trial_trace_ref.data(), boundaryJac_ref.data(), jac_ext, jacDet_ext, jacInv_ext, boundaryJac, metricTensor, metricTensorDetSqrt,
                                            normal_ref.data(), normal, x_ext, y_ext, z_ext);
        ck.calculateMappingVelocity_elementBoundary(eN, ebN_local, kb, ebN_local_kb, mesh_velocity_dof.data(), mesh_l2g.data(), mesh_trial_trace_ref.data(), xt_ext, yt_ext, zt_ext, normal, boundaryJac, metricTensor, integralScaling);
        dS = ((1.0 - MOVING_DOMAIN) * metricTensorDetSqrt + MOVING_DOMAIN * integralScaling) * dS_ref.data()[kb];
        //get the metric tensor
        //cek todo use symmetry
        ck.calculateG(jacInv_ext, G, G_dd_G, tr_G);
        //compute shape and solution information
        //shape
        ck.gradTrialFromRef(&u_grad_trial_trace_ref.data()[ebN_local_kb_nSpace * nDOF_trial_element], jacInv_ext, u_grad_trial_trace);
        //solution and gradient
        ck.valFromDOF(u_dof.data(), &u_l2g.data()[eN_nDOF_trial_element], &u_trial_trace_ref.data()[ebN_local_kb * nDOF_test_element], u_ext);
        ck.gradFromDOF(u_dof.data(), &u_l2g.data()[eN_nDOF_trial_element], u_grad_trial_trace, grad_u_ext);
        
        //populate ebqe_x
        // const int ebNE_kb_3d = ebNE_kb * 3;
        // ebqe_x.data()[ebNE_kb_3d + 0] = x_ext;
        // ebqe_x.data()[ebNE_kb_3d + 1] = y_ext;
        // ebqe_x.data()[ebNE_kb_3d + 2] = z_ext;
        
        //precalculate test function products with integration weights
        for (int j = 0; j < nDOF_trial_element; j++) { u_test_dS[j] = u_test_trace_ref.data()[ebN_local_kb * nDOF_test_element + j] * dS; }
        //
        //load the boundary values
        //
        bc_u_ext = isDOFBoundary_u.data()[ebNE_kb] * ebqe_bc_u_ext.data()[ebNE_kb] + (1 - isDOFBoundary_u.data()[ebNE_kb]) * u_ext;
        //
        //calculate the pde coefficients using the solution and the boundary values for the solution
        //
        const double rho_ext = ebqe_rho.data()[ebNE_kb];
        const double rho_velocity_ext = std::fabs(rho_ext) > 1.0e-12 ? rho_ext : rho;
        double Kr, dKr, thetaW_ext, thetaW_bc;
        // Boundary closure cross-derivatives wrt u_n; (0,1) boundary Jacobian
        // is not assembled here.
        double dm_du_n_ext = 0.0, dkr_du_n_ext = 0.0;
        double df_du_n_ext[nSpace];
        double da_du_n_ext[nnz];
        double bc_dm_du_n = 0.0, bc_dkr_du_n = 0.0;
        double bc_df_du_n[nSpace];
        double bc_da_du_n[nnz];
        for (int I = 0; I < nSpace; I++) { df_du_n_ext[I] = 0.0; bc_df_du_n[I] = 0.0; }
        for (int ii = 0; ii < nnz; ii++)  { da_du_n_ext[ii] = 0.0; bc_da_du_n[ii] = 0.0; }
        double u_n_ext_qp = 0.0;
        ck.valFromDOF(u_dof_n.data(), &u_l2g.data()[eN_nDOF_trial_element],
                      &u_trial_trace_ref.data()[ebN_local_kb * nDOF_test_element], u_n_ext_qp);
        // Component-1 boundary Dirichlet via the standard
        // mask = isDOFBoundary_n * bc + (1 - isDOFBoundary_n) * interior.
        const double bc_u_n_ext_qp = isDOFBoundary_n.data()[ebNE_kb] * ebqe_bc_u_n_ext.data()[ebNE_kb]
                                   + (1 - isDOFBoundary_n.data()[ebNE_kb]) * u_n_ext_qp;
        evaluateCoefficients_from_Se(a_rowptr.data(), a_colind.data(), rho, rho_ext, beta, gravity.data(),
                                     alpha.data()[elementMaterialTypes.data()[eN]], n.data()[elementMaterialTypes.data()[eN]],
                                     thetaR.data()[elementMaterialTypes.data()[eN]], thetaSR.data()[elementMaterialTypes.data()[eN]],
                                     &KWs.data()[elementMaterialTypes.data()[eN] * nnz], u_ext, u_n_ext_qp,
                                     m_ext, dm_ext, dm_du_n_ext, f_ext, df_ext, df_du_n_ext, a_ext, da_ext, da_du_n_ext,
                                     as_ext, Kr, dKr, dkr_du_n_ext, thetaW_ext);
        evaluateCoefficients_from_Se(a_rowptr.data(), a_colind.data(), rho, rho_ext, beta, gravity.data(),
                                     alpha.data()[elementMaterialTypes.data()[eN]], n.data()[elementMaterialTypes.data()[eN]],
                                     thetaR.data()[elementMaterialTypes.data()[eN]], thetaSR.data()[elementMaterialTypes.data()[eN]],
                                     &KWs.data()[elementMaterialTypes.data()[eN] * nnz], bc_u_ext, bc_u_n_ext_qp,
                                     bc_m_ext, bc_dm_ext, bc_dm_du_n, bc_f_ext, bc_df_ext, bc_df_du_n, bc_a_ext, bc_da_ext, bc_da_du_n,
                                     bc_as_ext, Kr, dKr, bc_dkr_du_n, thetaW_bc);
        ebqe_theta.data()[ebNE_kb] = thetaW_ext;
        
        //
        //Calculate Darcy velocity on exterior face : v_ext = -(a_ext/rho) * (grad_u_ext + gravity) ---
        //
        double ext_pressure_gradient[nSpace];
        for (int J=0; J<nSpace; ++J)
          ext_pressure_gradient[J] = grad_u_ext[J] - rho_velocity_ext * gravity.data()[J];

        for (int I=0; I<nSpace; ++I) {
          double acc = 0.0;
          for (int ii = a_rowptr.data()[I]; ii < a_rowptr.data()[I+1]; ++ii) {
            const int J = a_colind.data()[ii];
            acc += (a_ext[ii] / rho_velocity_ext) * ext_pressure_gradient[J];
          }
          ebqe_velocity_ext.data()[ebNE_kb_nSpace + I] = -acc;
          ebqe_velocity_ext_couple.data()[ebNE_kb_nSpace + I] = -acc ;  // store vector at this boundary qp
        }


        //
        // ===== P3c boundary: compositional comp-0 (H2O) flux  F_0 . n =====
        // Richards-style Nitsche trace flux on a pressure-Dirichlet (or seepage)
        // face: flux = F_0.n + penalty*(p - p_BC); no-flow faces keep bc_flux.
        // F_0 mirrors the interior H2O flux; FD-verified in boundary0_test.cpp.
        {
          double grad_u_n_ext[nSpace];
          ck.gradFromDOF(u_dof_n.data(), &u_l2g.data()[eN_nDOF_trial_element],
                         u_grad_trial_trace, grad_u_n_ext);
          const int    mat_b   = elementMaterialTypes.data()[eN];
          const double alpha_b = alpha.data()[mat_b];
          const double n_vg_b  = n.data()[mat_b];
          const double krn_end_b = krn_end.data()[mat_b];
          const double *KWs_b  = &KWs.data()[mat_b * nnz];
          const double phi_b   = thetaR.data()[mat_b] + thetaSR.data()[mat_b];
          const double S_wr_b  = thetaR.data()[mat_b] / phi_b;
          const double one_m_Sr_b = 1.0 - S_wr_b;
          const double Se_trap_L956 = 1.0 - S_gr.data()[mat_b] / one_m_Sr_b;  // gas-only residual trapping
          const double p_clb   = fmax(u_ext, 1.0e2);
          ::m_comp_co2::flash::FlashState fsb =
              ::m_comp_co2::flash::flashPZ(p_clb, u_n_ext_qp, T_C_member, m_NaCl_member, ::m_comp_co2::flash::EPS_Z, immiscible_member);
          const double Se_ab = fmin(fmax((1.0 - fsb.S_g - S_wr_b)/one_m_Sr_b, 0.0), 1.0);
          double KWrb=0,DKWrb=0,thWb=0,DthWb=0,KNrb=0,DKNrb=0,pcb=0,dpc_dSeb=0,d2pcb=0;
          if (PSK_TYPE_member == 1) {
            proteus::m_comp_co2::psk::bc_wetting_from_Se(Se_ab, alpha_b, n_vg_b, thetaR.data()[mat_b], thetaSR.data()[mat_b], thWb, DthWb, KWrb, DKWrb);
            proteus::m_comp_co2::psk::bc_kr_nonwetting_from_Se(Se_ab, alpha_b, n_vg_b, KNrb, DKNrb, Se_trap_L956);
            proteus::m_comp_co2::psk::bc_pc_from_Se(Se_ab, alpha_b, n_vg_b, bc_Se_min.data()[mat_b], pcb, dpc_dSeb, d2pcb);
          } else {
            proteus::m_comp_co2::psk::vgm_wetting_from_Se(Se_ab, alpha_b, n_vg_b, thetaR.data()[mat_b], thetaSR.data()[mat_b], thWb, DthWb, KWrb, DKWrb);
            proteus::m_comp_co2::psk::vgm_kr_nonwetting_from_Se(Se_ab, alpha_b, n_vg_b, KNrb, DKNrb, Se_trap_L956);
            proteus::m_comp_co2::psk::vgm_pc_from_Se(Se_ab, alpha_b, n_vg_b, pcb, dpc_dSeb, d2pcb);
          }
          KNrb *= krn_end_b;
          const double pcpb = dpc_dSeb / one_m_Sr_b;
          const double Mbar_gb = fsb.Y*::m_comp_co2::eos::M_CO2_KG + (1.0 - fsb.Y)*::m_comp_co2::eos::M_H2O_KG;
          const double Mbar_ab = fsb.X*::m_comp_co2::eos::M_CO2_KG + (1.0 - fsb.X)*::m_comp_co2::eos::M_H2O_KG;
          const double rho_g_mass_b = fsb.rho_g*Mbar_gb;
          const double rho_a_mass_b = fsb.rho_a*Mbar_ab;
          double F0n = 0.0;
          for (int I = 0; I < nSpace; I++) {
            double ua = 0.0, ug = 0.0;
            for (int ii = a_rowptr.data()[I]; ii < a_rowptr.data()[I+1]; ii++) {
              const int J = a_colind.data()[ii];
              const double gradSa_J = -(fsb.dS_g_dp*grad_u_ext[J] + fsb.dS_g_dz*grad_u_n_ext[J]);
              const double gp_a = grad_u_ext[J]                  - rho_a_mass_b*gravity.data()[J];
              const double gp_g = grad_u_ext[J] + pcpb*gradSa_J  - rho_g_mass_b*gravity.data()[J];
              ua -= (KWrb*KWs_b[ii])      * gp_a;
              ug -= (KNrb*KWs_b[ii]/mu_n) * gp_g;
            }
            F0n += (fsb.rho_g*(1.0 - fsb.Y)*ug + fsb.rho_a*(1.0 - fsb.X)*ua) * normal[I];
          }
          const int isSeep = isSeepageFace.data()[ebNE];
          if (isSeep || isDOFBoundary_u.data()[ebNE_kb]) {
            const double bc_u_pen = isSeep ? 0.0 : bc_u_ext;
            flux_ext = F0n + ebqe_penalty_ext.data()[ebNE_kb] * (u_ext - bc_u_pen);
            if (isSeep && flux_ext <= 0.0) flux_ext = 0.0;   // closed unless outflow
          } else {
            flux_ext = ebqe_bc_flux_ext[ebNE_kb];            // no-flow / prescribed flux
          }
        }
        ebqe_flux.data()[ebNE_kb] = flux_ext;

        anb_seepage_flux             = seepagefluxcalculator(anb_seepage_flux, isSeepageFace.data()[ebNE], dS, flux_ext);
        anb_seepage_flux_n.data()[0] = anb_seepage_flux;
        ebqe_u.data()[ebNE_kb]       = u_ext;
        //
        //update residuals
        //
        for (int i = 0; i < nDOF_test_element; i++) {
          elementResidual_u[i] += ck.ExteriorElementBoundaryFlux(flux_ext, u_test_dS[i]);
        } //i
      } //kb

      //
      //update the element and global residual storage
      //
      for (int i = 0; i < nDOF_test_element; i++) {
        int eN_i = eN * nDOF_test_element + i;
        globalResidual.data()[offset_u + stride_u * u_l2g.data()[eN_i]] += elementResidual_u[i];
      } //i
    } //ebNE

    // ============================================================================
    // component-1 (S_n) gas-mass equation.
    //
    //   d(phi*rho_n*S_n)/dt + div F_n = 0
    //     m_n     = phi * rho_n * u_n
    //     m_n_old = phi * rho_n * u_n_old
    //
    // Gas Darcy flux F_n = -(K/mu_n) k_rn (grad p_n - rho_n g)
    //                    = -(K/mu_n) k_rn (grad p_w + dp_c/dS_n grad S_n - rho_n g)
    // Splitting into Proteus form for u_w = p_w, u_n = S_n:
    //   f_n     = rho_n^2 * k_rn * (K/mu_n) * g          (gravity advection)
    //   a_n     = rho_n   * k_rn * (K/mu_n)               (diffusion against grad u_w)
    //   a_n_pc  = rho_n   * k_rn * (K/mu_n) * dp_c/dS_n   (capillary diffusion against grad u_n)
    // At u_n = 0 (fully wet) the closure returns k_rn = 0 -> all flux contributions
    // vanish, so this loop should reproduce the single-phase result for the
    // Bioswale.
    for (int eN = 0; eN < nElements_global; eN++) {
      const int   mat_eN  = elementMaterialTypes.data()[eN];
      const double phi_eN = thetaR.data()[mat_eN] + thetaSR.data()[mat_eN];
      const double alpha_eN = alpha.data()[mat_eN];
      const double krn_end_eN = krn_end.data()[mat_eN];
      const double n_vg_eN  = n.data()[mat_eN];
      const double *KWs_eN  = &KWs.data()[mat_eN * nnz];
      double elementResidual_n[nDOF_test_element];
      // Row-sum lumped weight per node: M_lump[i] = int N_i dV over the element.
      // Used to apply the CO2 injection source as a pure diagonal contribution
      // at the port nodes (no smearing onto neighbor basis functions), which
      // matches the localised disk source and avoids exciting the BC closure
      // on rim nodes where k_rn is still zero.
      double lumped_w_n[nDOF_test_element];
      for (int i = 0; i < nDOF_test_element; i++) {
        elementResidual_n[i] = 0.0;
        lumped_w_n[i]        = 0.0;
      }
      for (int k = 0; k < nQuadraturePoints_element; k++) {
        const int eN_k = eN * nQuadraturePoints_element + k;
        const int eN_nDOF_trial_element = eN * nDOF_trial_element;
        double jac[nSpace * nSpace], jacDet, jacInv[nSpace * nSpace], x_q, y_q, z_q;
        ck.calculateMapping_element(eN, k, mesh_dof.data(), mesh_l2g.data(),
                                    mesh_trial_ref.data(), mesh_grad_trial_ref.data(),
                                    jac, jacDet, jacInv, x_q, y_q, z_q);
        const double dV = std::fabs(jacDet) * dV_ref.data()[k];
        // Trial gradients in physical coords (used for grad u_w and grad N_i).
        double u_grad_trial_qp[nDOF_trial_element * nSpace];
        ck.gradTrialFromRef(&u_grad_trial_ref.data()[k * nDOF_trial_element * nSpace],
                            jacInv, u_grad_trial_qp);
        // Saturation u_n and old at QP.
        double u_n = 0.0, u_n_old = 0.0;
        ck.valFromDOF(u_dof_n.data(),
                      &u_l2g.data()[eN_nDOF_trial_element],
                      &u_trial_ref.data()[k * nDOF_trial_element], u_n);
        ck.valFromDOF(u_dof_n_old.data(),
                      &u_l2g.data()[eN_nDOF_trial_element],
                      &u_trial_ref.data()[k * nDOF_trial_element], u_n_old);
        // Wetting-pressure value and gradient at QP. u_w (and u_w_old) feed
        // the linear EOS rho_n(p_n) = c_n*(u_w + p_c(u_n)).
        double u_w_qp = 0.0, u_w_qp_old = 0.0;
        ck.valFromDOF(u_dof.data(),
                      &u_l2g.data()[eN_nDOF_trial_element],
                      &u_trial_ref.data()[k * nDOF_trial_element], u_w_qp);
        ck.valFromDOF(u_dof_old.data(),
                      &u_l2g.data()[eN_nDOF_trial_element],
                      &u_trial_ref.data()[k * nDOF_trial_element], u_w_qp_old);
        double grad_u_w[nSpace];
        ck.gradFromDOF(u_dof.data(),
                       &u_l2g.data()[eN_nDOF_trial_element],
                       u_grad_trial_qp, grad_u_w);
        // Saturation gradient at QP (Step 3d: needed for p_c(S_n) flux contribution).
        double grad_u_n[nSpace];
        ck.gradFromDOF(u_dof_n.data(),
                       &u_l2g.data()[eN_nDOF_trial_element],
                       u_grad_trial_qp, grad_u_n);
        // Rock pore fraction and residual wetting saturation; consumed by the
        // compositional flux F_1 below (Se_a = (S_a - S_wr)/(1 - S_wr)).
        // P3c (complete): the gas relperm / capillary / linear-gas-EOS closures
        // that used to be evaluated here -- phase-based, keyed on u_n as S_n --
        // are removed.  The F_1 block recomputes every saturation-dependent
        // property from the FLASH saturation S_g, so nothing downstream reads a
        // phase-based k_rn(u_n), p_c(u_n) or rho_n(p_n) any more.
        const double phi_loc      = thetaR.data()[mat_eN] + thetaSR.data()[mat_eN];
        const double S_wr_loc     = thetaR.data()[mat_eN] / phi_loc;
        const double one_m_Sr_loc = 1.0 - S_wr_loc;
        const double Se_trap_L1103 = 1.0 - S_gr.data()[mat_eN] / one_m_Sr_loc;  // gas-only residual trapping
        // component-1 (CO2) accumulation, compositional (p,z) form:
        //   m_1 = phi * N * z,   N = rho_g*S_g + rho_a*(1-S_g),  z = u_n.
        const double p_cl     = fmax(u_w_qp,     1.0e2);
        const double p_cl_old = fmax(u_w_qp_old, 1.0e2);
        ::m_comp_co2::flash::FlashState fs_n =
            ::m_comp_co2::flash::flashPZ(p_cl, u_n, T_C_member, m_NaCl_member, ::m_comp_co2::flash::EPS_Z, immiscible_member);
        ::m_comp_co2::flash::FlashState fs_n_old =
            ::m_comp_co2::flash::flashPZ(p_cl_old, u_n_old, T_C_member, m_NaCl_member, ::m_comp_co2::flash::EPS_Z, immiscible_member);
        const double N_cur = fs_n.rho_g*fs_n.S_g + fs_n.rho_a*(1.0 - fs_n.S_g);
        const double N_old = fs_n_old.rho_g*fs_n_old.S_g
                           + fs_n_old.rho_a*(1.0 - fs_n_old.S_g);
        const double m_n     = phi_eN * N_cur * u_n;
        const double m_n_old = phi_eN * N_old * u_n_old;
        const double m_n_t   = (m_n - m_n_old) / dt;
        // ===== P3c: component CO2 molar flux  F_1 = rho_g*Y*u_g + rho_a*X*u_a =====
        // Saturation-dependent props are recomputed from the FLASH saturation S_g
        // (NOT u_n, which is now z).  S_a = 1 - S_g; psk closures take wetting Se_a.
        const double S_a_qp = 1.0 - fs_n.S_g;
        const double Se_a   = fmin(fmax((S_a_qp - S_wr_loc) / one_m_Sr_loc, 0.0), 1.0);
        double KWr_a = 0.0, DKWr_a = 0.0, thW_a = 0.0, DthW_a = 0.0;
        double KNr_a = 0.0, DKNr_a = 0.0;
        double pc_a = 0.0, dpc_dSe_a = 0.0, d2pc_a = 0.0;
        if (PSK_TYPE_member == 1) {
          proteus::m_comp_co2::psk::bc_wetting_from_Se(Se_a, alpha_eN, n_vg_eN,
              thetaR.data()[mat_eN], thetaSR.data()[mat_eN], thW_a, DthW_a, KWr_a, DKWr_a);
          proteus::m_comp_co2::psk::bc_kr_nonwetting_from_Se(Se_a, alpha_eN, n_vg_eN, KNr_a, DKNr_a, Se_trap_L1103);
          proteus::m_comp_co2::psk::bc_pc_from_Se(Se_a, alpha_eN, n_vg_eN, bc_Se_min.data()[mat_eN], pc_a, dpc_dSe_a, d2pc_a);
        } else {
          proteus::m_comp_co2::psk::vgm_wetting_from_Se(Se_a, alpha_eN, n_vg_eN,
              thetaR.data()[mat_eN], thetaSR.data()[mat_eN], thW_a, DthW_a, KWr_a, DKWr_a);
          proteus::m_comp_co2::psk::vgm_kr_nonwetting_from_Se(Se_a, alpha_eN, n_vg_eN, KNr_a, DKNr_a, Se_trap_L1103);
          proteus::m_comp_co2::psk::vgm_pc_from_Se(Se_a, alpha_eN, n_vg_eN, pc_a, dpc_dSe_a, d2pc_a);
        }
        KNr_a *= krn_end_eN;
        // p_c'(S_a) = dp_c/dSe * dSe/dS_a = dpc_dSe_a / (1 - S_wr).
        const double pcp_a = dpc_dSe_a / one_m_Sr_loc;
        // phase mass densities for gravity (molar density * mean molar mass):
        const double Mbar_g = fs_n.Y*::m_comp_co2::eos::M_CO2_KG
                            + (1.0 - fs_n.Y)*::m_comp_co2::eos::M_H2O_KG;
        const double Mbar_a = fs_n.X*::m_comp_co2::eos::M_CO2_KG
                            + (1.0 - fs_n.X)*::m_comp_co2::eos::M_H2O_KG;
        const double rho_g_mass = fs_n.rho_g*Mbar_g;
        const double rho_a_mass = fs_n.rho_a*Mbar_a;
        // u_a = -(K krw/mu_w)(grad p_a - rho_a_mass g),  p_a = u_w
        // u_g = -(K krg/mu_g)(grad p_g - rho_g_mass g),  p_g = u_w + p_c(S_a)
        //   grad p_g = grad u_w + p_c'(S_a) grad S_a,  grad S_a = -(dSg/dp grad u_w + dSg/dz grad u_n)
        // KWs = K/mu_w (aqueous base); mu_n = mu_g/mu_w so KWs/mu_n = K/mu_g.
        double F1[nSpace];
        for (int I = 0; I < nSpace; I++) {
          double ua = 0.0, ug = 0.0;
          for (int ii = a_rowptr.data()[I]; ii < a_rowptr.data()[I + 1]; ii++) {
            const int J = a_colind.data()[ii];
            const double gradSa_J = -(fs_n.dS_g_dp*grad_u_w[J] + fs_n.dS_g_dz*grad_u_n[J]);
            const double gp_a = grad_u_w[J]                  - rho_a_mass*gravity.data()[J];
            const double gp_g = grad_u_w[J] + pcp_a*gradSa_J - rho_g_mass*gravity.data()[J];
            ua -= (KWr_a*KWs_eN[ii])      * gp_a;
            ug -= (KNr_a*KWs_eN[ii]/mu_n) * gp_g;
          }
          F1[I] = fs_n.rho_g*fs_n.Y*ug + fs_n.rho_a*fs_n.X*ua;
        }
        // Dissolution is handled thermodynamically by the inline flash (aqueous
        // CO2 = rho_a*X advected in F_1); the old kinetic R_diss sink is removed.
        // Residual integration: mass + component CO2 flux (- injection, lumped).
        for (int i = 0; i < nDOF_test_element; i++) {
          const double test_i = u_test_ref.data()[k * nDOF_test_element + i];
          // Mass contribution.
          elementResidual_n[i] += m_n_t * test_i * dV;
          lumped_w_n[i] += test_i * dV;
          for (int I = 0; I < nSpace; I++) {
            elementResidual_n[i] -= F1[I] * u_grad_trial_qp[i * nSpace + I] * dV;
          }
        }
      }
      // Row-sum lumped CO2 injection: subtract Q_inj at port nodes only.
      for (int i = 0; i < nDOF_test_element; i++) {
        const int gi = u_l2g.data()[eN * nDOF_test_element + i];
        elementResidual_n[i] -= injection_dof.data()[gi] * lumped_w_n[i];
      }
      for (int i = 0; i < nDOF_test_element; i++) {
        const int eN_i = eN * nDOF_test_element + i;
        globalResidual.data()[offset_n + stride_n * u_l2g.data()[eN_i]] += elementResidual_n[i];
      }
    }

    // ============================================================================
    // Comp-1 (CO2 / z) exterior boundary loop -- STAB=0 path. COMPOSITIONAL.
    //
    // P3c STATUS: BOUNDARY PORTED (2026-06-06). Slot 1 is the overall CO2
    // composition z, NOT a saturation. Computes the compositional molar CO2
    // trace flux F_1.n = rho_g*Y*u_g + rho_a*X*u_a, mirroring the FD-verified
    // interior element flux (~line 1152) and the STAB=2 boundary loop; every
    // saturation-dependent property is recomputed from the FLASH saturation
    // S_g(p,z). The surface term from div(F_1) by parts is +(F_1.n) N_i dS.
    //   isDir_n     : F_1.n = consistent compositional flux
    //                         + penalty*(z - z_BC)               [Nitsche]
    //   not isDir_n : F_1.n = 0   (no-flow / closed -- no spurious boundary flux)
    // so STAB=0 and STAB=2 now enforce identical compositional comp-1 BCs.
    // McWhorter-Sunada drives this via getDBC_z at the inlet (z_BC = z(S_n_BC)).
    // Jacobian is the matching (1,1)/(1,0) blocks in calculateJacobian.
    // ============================================================================
    for (int ebNE = 0; ebNE < nExteriorElementBoundaries_global; ebNE++) {
      const int ebN = exteriorElementBoundariesArray.data()[ebNE];
      const int eN  = elementBoundaryElementsArray.data()[ebN * 2 + 0];
      const int ebN_local = elementBoundaryLocalElementBoundariesArray.data()[ebN * 2 + 0];
      const int eN_nDOF_trial_element = eN * nDOF_trial_element;
      const int    mat_eN    = elementMaterialTypes.data()[eN];
      const double phi_eN    = thetaR.data()[mat_eN] + thetaSR.data()[mat_eN];
      const double alpha_eN  = alpha.data()[mat_eN];
      const double krn_end_eN = krn_end.data()[mat_eN];
      const double n_vg_eN   = n.data()[mat_eN];
      const double *KWs_eN   = &KWs.data()[mat_eN * nnz];
      const double S_wr_loc      = thetaR.data()[mat_eN] / phi_eN;
      const double one_m_Sr_loc  = 1.0 - S_wr_loc;
      const double Se_trap_L1235 = 1.0 - S_gr.data()[mat_eN] / one_m_Sr_loc;  // gas-only residual trapping

      double elementResidual_n_eb[nDOF_test_element];
      for (int i = 0; i < nDOF_test_element; i++) elementResidual_n_eb[i] = 0.0;

      for (int kb = 0; kb < nQuadraturePoints_elementBoundary; kb++) {
        const int ebNE_kb            = ebNE * nQuadraturePoints_elementBoundary + kb;
        const int ebN_local_kb       = ebN_local * nQuadraturePoints_elementBoundary + kb;
        const int ebN_local_kb_nSpace = ebN_local_kb * nSpace;

        double jac_ext[nSpace * nSpace], jacDet_ext, jacInv_ext[nSpace * nSpace];
        double boundaryJac_b[nSpace * (nSpace - 1)];
        double metricTensor_b[(nSpace - 1) * (nSpace - 1)];
        double metricTensorDetSqrt_b, dS_eb, normal_b[3];
        double xt_b, yt_b, zt_b, integralScaling_b;
        double x_eb, y_eb, z_eb;
        ck.calculateMapping_elementBoundary(eN, ebN_local, kb, ebN_local_kb,
            mesh_dof.data(), mesh_l2g.data(), mesh_trial_trace_ref.data(),
            mesh_grad_trial_trace_ref.data(), boundaryJac_ref.data(),
            jac_ext, jacDet_ext, jacInv_ext, boundaryJac_b, metricTensor_b,
            metricTensorDetSqrt_b, normal_ref.data(), normal_b,
            x_eb, y_eb, z_eb);
        ck.calculateMappingVelocity_elementBoundary(eN, ebN_local, kb, ebN_local_kb,
            mesh_velocity_dof.data(), mesh_l2g.data(), mesh_trial_trace_ref.data(),
            xt_b, yt_b, zt_b, normal_b, boundaryJac_b, metricTensor_b,
            integralScaling_b);
        dS_eb = ((1.0 - MOVING_DOMAIN) * metricTensorDetSqrt_b
                + MOVING_DOMAIN * integralScaling_b) * dS_ref.data()[kb];

        double u_grad_trial_trace_b[nDOF_trial_element * nSpace];
        ck.gradTrialFromRef(
            &u_grad_trial_trace_ref.data()[ebN_local_kb_nSpace * nDOF_trial_element],
            jacInv_ext, u_grad_trial_trace_b);
        double u_w_ext_b = 0.0, u_n_ext_b = 0.0;
        double grad_u_w_ext_b[nSpace], grad_u_n_ext_b[nSpace];
        ck.valFromDOF(u_dof.data(), &u_l2g.data()[eN_nDOF_trial_element],
                      &u_trial_trace_ref.data()[ebN_local_kb * nDOF_test_element],
                      u_w_ext_b);
        ck.valFromDOF(u_dof_n.data(), &u_l2g.data()[eN_nDOF_trial_element],
                      &u_trial_trace_ref.data()[ebN_local_kb * nDOF_test_element],
                      u_n_ext_b);
        ck.gradFromDOF(u_dof.data(), &u_l2g.data()[eN_nDOF_trial_element],
                       u_grad_trial_trace_b, grad_u_w_ext_b);
        ck.gradFromDOF(u_dof_n.data(), &u_l2g.data()[eN_nDOF_trial_element],
                       u_grad_trial_trace_b, grad_u_n_ext_b);

        const int    isDir_n      = isDOFBoundary_n.data()[ebNE_kb];
        const double bc_u_n_ext_b = isDir_n * ebqe_bc_u_n_ext.data()[ebNE_kb]
                                  + (1 - isDir_n) * u_n_ext_b;

        // ====================================================================
        // P3c boundary (STAB=0): compositional comp-1 (CO2) trace flux F_1.n.
        // Slot 1 is the overall CO2 composition z, NOT a saturation. Mirrors
        // the FD-verified interior element flux F_1 = rho_g*Y*u_g + rho_a*X*u_a
        // (~line 1152) and the STAB=2 boundary loop; every saturation-dependent
        // property is recomputed from the FLASH saturation S_g(p,z) (psk
        // closures on the wetting Se_a = (1-S_g-S_wr)/(1-S_wr)). Consistent flux
        // on Dirichlet-z faces + a Nitsche penalty driving z at the trace toward
        // bc_u_n_ext_b (= z_BC; McWhorter-Sunada inlet); no-flow faces contribute
        // nothing (closed-box conservation). Jacobian is in calculateJacobian.
        // ====================================================================
        const double p_clb = fmax(u_w_ext_b, 1.0e2);
        ::m_comp_co2::flash::FlashState fsb =
            ::m_comp_co2::flash::flashPZ(p_clb, u_n_ext_b, T_C_member, m_NaCl_member, ::m_comp_co2::flash::EPS_Z, immiscible_member);
        const double Se_ab = fmin(fmax((1.0 - fsb.S_g - S_wr_loc)/one_m_Sr_loc, 0.0), 1.0);
        double KWr_b=0,DKWr_b=0,thW_b=0,DthW_b=0,KNr_b=0,DKNr_b=0,pc_b=0,dpc_dSe_b=0,d2pc_b=0;
        if (PSK_TYPE_member == 1) {
          proteus::m_comp_co2::psk::bc_wetting_from_Se(Se_ab, alpha_eN, n_vg_eN, thetaR.data()[mat_eN], thetaSR.data()[mat_eN], thW_b, DthW_b, KWr_b, DKWr_b);
          proteus::m_comp_co2::psk::bc_kr_nonwetting_from_Se(Se_ab, alpha_eN, n_vg_eN, KNr_b, DKNr_b, Se_trap_L1235);
          proteus::m_comp_co2::psk::bc_pc_from_Se(Se_ab, alpha_eN, n_vg_eN, bc_Se_min.data()[mat_eN], pc_b, dpc_dSe_b, d2pc_b);
        } else {
          proteus::m_comp_co2::psk::vgm_wetting_from_Se(Se_ab, alpha_eN, n_vg_eN, thetaR.data()[mat_eN], thetaSR.data()[mat_eN], thW_b, DthW_b, KWr_b, DKWr_b);
          proteus::m_comp_co2::psk::vgm_kr_nonwetting_from_Se(Se_ab, alpha_eN, n_vg_eN, KNr_b, DKNr_b, Se_trap_L1235);
          proteus::m_comp_co2::psk::vgm_pc_from_Se(Se_ab, alpha_eN, n_vg_eN, pc_b, dpc_dSe_b, d2pc_b);
        }
        KNr_b *= krn_end_eN;
        const double pcpb = dpc_dSe_b / one_m_Sr_loc;
        const double Mbar_gb = fsb.Y*::m_comp_co2::eos::M_CO2_KG + (1.0 - fsb.Y)*::m_comp_co2::eos::M_H2O_KG;
        const double Mbar_ab = fsb.X*::m_comp_co2::eos::M_CO2_KG + (1.0 - fsb.X)*::m_comp_co2::eos::M_H2O_KG;
        const double rho_g_mass_b = fsb.rho_g*Mbar_gb;
        const double rho_a_mass_b = fsb.rho_a*Mbar_ab;
        double F_n_dot_n = 0.0;
        for (int I = 0; I < nSpace; I++) {
          double ua = 0.0, ug = 0.0;
          for (int ii = a_rowptr.data()[I]; ii < a_rowptr.data()[I + 1]; ii++) {
            const int J = a_colind.data()[ii];
            const double gradSa_J = -(fsb.dS_g_dp*grad_u_w_ext_b[J] + fsb.dS_g_dz*grad_u_n_ext_b[J]);
            const double gp_a = grad_u_w_ext_b[J]                 - rho_a_mass_b*gravity.data()[J];
            const double gp_g = grad_u_w_ext_b[J] + pcpb*gradSa_J - rho_g_mass_b*gravity.data()[J];
            ua -= (KWr_b*KWs_eN[ii])      * gp_a;
            ug -= (KNr_b*KWs_eN[ii]/mu_n) * gp_g;
          }
          F_n_dot_n += (fsb.rho_g*fsb.Y*ug + fsb.rho_a*fsb.X*ua) * normal_b[I];
        }
        if (isDir_n) {
          // IIPG penalty scaled by the local comp-1 diffusion magnitude a_n
          // (= rho_g*Y*krn/mu_n + rho_a*X*krw, times a representative K/mu_w).
          // The framework penalty is const/h, UNSCALED by the coefficient; for
          // comp-1 the ~1e4 molar density makes the bare penalty ~1e4x too weak
          // to enforce z=z_BC, so the inlet floats (the McWhorter-Sunada
          // convergence killer). Coefficient treated frozen in the Jacobian.
          double Kw_rep = 0.0;
          for (int ii = 0; ii < nnz; ii++) Kw_rep = fmax(Kw_rep, fabs(KWs_eN[ii]));
          const double a_n_scale = (fsb.rho_g*fsb.Y*KNr_b/mu_n
                                  + fsb.rho_a*fsb.X*KWr_b) * Kw_rep;
          const double pen_eff = ebqe_penalty_ext.data()[ebNE_kb] * a_n_scale;
          F_n_dot_n += pen_eff * (u_n_ext_b - bc_u_n_ext_b);   // Nitsche, drives z->z_BC
        } else {
          F_n_dot_n = 0.0;                                     // no-flow
        }

        for (int i = 0; i < nDOF_test_element; i++) {
          const double test_i_dS = u_test_trace_ref.data()[
              ebN_local_kb * nDOF_test_element + i] * dS_eb;
          elementResidual_n_eb[i] += F_n_dot_n * test_i_dS;
        }
      } // kb

      for (int i = 0; i < nDOF_test_element; i++) {
        const int gi = u_l2g.data()[eN * nDOF_test_element + i];
        globalResidual.data()[offset_n + stride_n * gi] += elementResidual_n_eb[i];
      }
    } // ebNE
  }

  void calculateJacobian(arguments_dict &args)
  {
    proteus::pyarray<double> &mesh_trial_ref            = args.array<double>("mesh_trial_ref");
    proteus::pyarray<double> &mesh_grad_trial_ref       = args.array<double>("mesh_grad_trial_ref");
    proteus::pyarray<double> &mesh_dof                  = args.array<double>("mesh_dof");
    proteus::pyarray<double> &mesh_velocity_dof         = args.array<double>("mesh_velocity_dof");
    double               MOVING_DOMAIN             = args.scalar<double>("MOVING_DOMAIN");
    proteus::pyarray<int>    &mesh_l2g                  = args.array<int>("mesh_l2g");
    proteus::pyarray<double> &dV_ref                    = args.array<double>("dV_ref");
    proteus::pyarray<double> &u_trial_ref               = args.array<double>("u_trial_ref");
    proteus::pyarray<double> &u_grad_trial_ref          = args.array<double>("u_grad_trial_ref");
    proteus::pyarray<double> &u_test_ref                = args.array<double>("u_test_ref");
    proteus::pyarray<double> &u_grad_test_ref           = args.array<double>("u_grad_test_ref");
    proteus::pyarray<double> &mesh_trial_trace_ref      = args.array<double>("mesh_trial_trace_ref");
    proteus::pyarray<double> &mesh_grad_trial_trace_ref = args.array<double>("mesh_grad_trial_trace_ref");
    proteus::pyarray<double> &dS_ref                    = args.array<double>("dS_ref");
    proteus::pyarray<double> &u_trial_trace_ref         = args.array<double>("u_trial_trace_ref");
    proteus::pyarray<double> &u_grad_trial_trace_ref    = args.array<double>("u_grad_trial_trace_ref");
    proteus::pyarray<double> &u_test_trace_ref          = args.array<double>("u_test_trace_ref");
    proteus::pyarray<double> &u_grad_test_trace_ref     = args.array<double>("u_grad_test_trace_ref");
    proteus::pyarray<double> &normal_ref                = args.array<double>("normal_ref");
    proteus::pyarray<double> &boundaryJac_ref           = args.array<double>("boundaryJac_ref");
    int                  nElements_global          = args.scalar<int>("nElements_global");
    proteus::pyarray<double> &ebqe_penalty_ext          = args.array<double>("ebqe_penalty_ext");
    proteus::pyarray<int>    &elementMaterialTypes      = args.array<int>("elementMaterialTypes");
    proteus::pyarray<int>    &isSeepageFace             = args.array<int>("isSeepageFace");
    proteus::pyarray<int>    &a_rowptr                  = args.array<int>("a_rowptr");
    proteus::pyarray<int>    &a_colind                  = args.array<int>("a_colind");
    double               rho                       = args.scalar<double>("rho");
    double               beta                      = args.scalar<double>("beta");

    /////////////////////////////DENSITY COUPLING  >>>> USE rho from mprans model/////////////////////////
    proteus::pyarray<double> &q_rho                    = args.array<double>("q_rho");
    proteus::pyarray<double> &ebqe_rho                 = args.array<double>("ebqe_rho");
    //////////////////////////////////////////////////////////////////////////////////////////////////////
    // Stage 3b: gas-side kinetic dissolution sink reads (residual-only at
    // this stage; the Jacobian contribution -d(R_diss)/du_n is small
    // (proportional to k_d * dt) and we approximate it as zero in the
    // initial port -- Newton recovers it through outer iteration since the
    // sink is also bounded.  Promote to a proper Jacobian contribution if
    // Newton stalls in the FluidFlower setup.
    proteus::pyarray<double> &c_dof_jac                = args.array<double>("c_dof");
    const double         k_d_jac                  = args.scalar<double>("k_d");
    const double         c_sat_jac                = args.scalar<double>("c_sat");

    proteus::pyarray<double> &gravity                   = args.array<double>("gravity");
    proteus::pyarray<double> &alpha                     = args.array<double>("alpha");
    proteus::pyarray<double> &n                         = args.array<double>("n");
    proteus::pyarray<double> &thetaR                    = args.array<double>("thetaR");
    proteus::pyarray<double> &thetaSR                   = args.array<double>("thetaSR");
    proteus::pyarray<double> &KWs                       = args.array<double>("KWs");
    proteus::pyarray<double> &krn_end                   = args.array<double>("krn_end");
    proteus::pyarray<double> &S_gr                      = args.array<double>("S_gr");
    proteus::pyarray<double> &bc_Se_min                 = args.array<double>("bc_Se_min");
    double               mu_n                      = args.scalar<double>("mu_n");
    double               useMetrics                = args.scalar<double>("useMetrics");
    double               alphaBDF                  = args.scalar<double>("alphaBDF");
    int                  lag_shockCapturing        = args.scalar<int>("lag_shockCapturing");
    double               shockCapturingDiffusion   = args.scalar<double>("shockCapturingDiffusion");
    // VMS
    double               VMS                                        = args.scalar<double>("VMS");
    proteus::pyarray<int>    &u_l2g                                      = args.array<int>("u_l2g");
    proteus::pyarray<double> &elementDiameter                            = args.array<double>("elementDiameter");
    proteus::pyarray<double> &u_dof                                      = args.array<double>("u_dof");
    // component-1 saturation DOFs (needed by the gas-eq
    // Jacobian element loop appended at the end of this function).
    proteus::pyarray<double> &u_dof_n                                    = args.array<double>("u_dof_n");
    proteus::pyarray<double> &velocity                                   = args.array<double>("velocity");
    proteus::pyarray<double> &q_m_betaBDF                                = args.array<double>("q_m_betaBDF");
    proteus::pyarray<double> &cfl                                        = args.array<double>("cfl");
    proteus::pyarray<double> &q_numDiff_u                                = args.array<double>("q_numDiff_u");
    proteus::pyarray<double> &q_numDiff_u_last                           = args.array<double>("q_numDiff_u_last");
    proteus::pyarray<int>    &csrRowIndeces_u_u                          = args.array<int>("csrRowIndeces_u_u");
    proteus::pyarray<int>    &csrColumnOffsets_u_u                       = args.array<int>("csrColumnOffsets_u_u");
    proteus::pyarray<double> &globalJacobian                             = args.array<double>("globalJacobian");
    // component-1 (S_n) Jacobian args. Used by
    // the dedicated component-1 element loop appended at the end of this
    // function. (1,1) block is the consistent mass matrix / dt; (0,1) and
    // (1,0) cross-blocks are zero in Step 1.
    const double         dt_n                                       = args.scalar<double>("dt");
    // gas-phase density (linear EOS rho_n*p_n/p_ref_n when p_ref_n>0,
    // constant rho_n otherwise). c_n = drho_n/dp_n is constant for the
    // linear EOS and drives the new (1,1)/(1,0) compressibility terms.
    const double         rho_n                                      = args.scalar<double>("rho_n");
    const double         p_ref_n                                    = args.scalar<double>("p_ref_n");
    const bool           rho_n_compressible                         = (p_ref_n > 0.0);
    const double         c_n                                        = rho_n_compressible ? (rho_n / p_ref_n) : 0.0;
    proteus::pyarray<int>    &csrRowIndeces_n_n                          = args.array<int>("csrRowIndeces_n_n");
    // (1,0) cross-block CSR maps for the gas-eq diffusion
    // against grad u_w. Currently allocated by the framework but unused by
    // the C++ assembly - 3c.2/3c.3 will write into them.
    proteus::pyarray<int>    &csrRowIndeces_n_w                          = args.array<int>("csrRowIndeces_n_w");
    proteus::pyarray<int>    &csrColumnOffsets_n_n                       = args.array<int>("csrColumnOffsets_n_n");
    proteus::pyarray<int>    &csrColumnOffsets_n_w                       = args.array<int>("csrColumnOffsets_n_w");
    // Exterior-boundary CSR column offsets for the comp-1 boundary Jacobian
    // loop ported from calculateResidual_entropy_viscosity (STAB=0 comp-1
    // Dirichlet enforcement). Added to getJacobian's argsDict in Python.
    proteus::pyarray<int>    &csrColumnOffsets_eb_n_n                    = args.array<int>("csrColumnOffsets_eb_n_n");
    proteus::pyarray<int>    &csrColumnOffsets_eb_n_w                    = args.array<int>("csrColumnOffsets_eb_n_w");
    // (0,1) cross-block CSR maps for the wetting eq.
    //   J_{wv,ij} = (dm/du_n)*N_i*N_j + (df/du_n)*grad N_i*N_j
    //               + (da/du_n)*grad u_w*grad N_i*N_j
    // The wetting equation row index is offset_u + stride_u * dof, and the column
    // index is offset_n + stride_n * dof.
    proteus::pyarray<int>    &csrRowIndeces_w_n                          = args.array<int>("csrRowIndeces_w_n");
    proteus::pyarray<int>    &csrColumnOffsets_w_n                       = args.array<int>("csrColumnOffsets_w_n");
    // (0,1) cross-block boundary CSR for the wetting-eq
    // exterior-flux Jacobian.
    proteus::pyarray<int>    &csrColumnOffsets_eb_w_n                    = args.array<int>("csrColumnOffsets_eb_w_n");
    // PSK closure selector for evaluateCoefficients (read from argsDict).
    PSK_TYPE_member = args.scalar<int>("PSK_TYPE");
    immiscible_member = (args.scalar<int>("immiscible") != 0);
    T_C_member        = args.scalar<double>("T_C");      // temperature [degC] from input
    int                  nExteriorElementBoundaries_global          = args.scalar<int>("nExteriorElementBoundaries_global");
    proteus::pyarray<int>    &exteriorElementBoundariesArray             = args.array<int>("exteriorElementBoundariesArray");
    proteus::pyarray<int>    &elementBoundaryElementsArray               = args.array<int>("elementBoundaryElementsArray");
    proteus::pyarray<int>    &elementBoundaryLocalElementBoundariesArray = args.array<int>("elementBoundaryLocalElementBoundariesArray");
    proteus::pyarray<double> &ebqe_velocity_ext                          = args.array<double>("ebqe_velocity_ext");
    proteus::pyarray<int>    &isDOFBoundary_u                            = args.array<int>("isDOFBoundary_u");
    proteus::pyarray<double> &ebqe_bc_u_ext                              = args.array<double>("ebqe_bc_u_ext");
    // component-1 (S_n) boundary arrays.
    proteus::pyarray<int>    &isDOFBoundary_n                            = args.array<int>("isDOFBoundary_n");
    proteus::pyarray<double> &ebqe_bc_u_n_ext                            = args.array<double>("ebqe_bc_u_n_ext");
    proteus::pyarray<int>    &isFluxBoundary_u                           = args.array<int>("isFluxBoundary_u");
    proteus::pyarray<double> &ebqe_bc_flux_ext                           = args.array<double>("ebqe_bc_flux_ext");
    proteus::pyarray<int>    &csrColumnOffsets_eb_u_u                    = args.array<int>("csrColumnOffsets_eb_u_u");
    int                  LUMPED_MASS_MATRIX                         = args.scalar<int>("LUMPED_MASS_MATRIX");
    assert(a_rowptr.data()[nSpace] == nnz);
    assert(a_rowptr.data()[nSpace] == nSpace);
    double Ct_sge = 4.0;

    //
    //loop over elements to compute volume integrals and load them into the element Jacobians and global Jacobian
    //
    for (int eN = 0; eN < nElements_global; eN++) {
      double elementJacobian_u_u[nDOF_test_element][nDOF_trial_element];
      // (0,1) cross-block element storage.
      double elementJacobian_u_n[nDOF_test_element][nDOF_trial_element];
      for (int i = 0; i < nDOF_test_element; i++) {
        for (int j = 0; j < nDOF_trial_element; j++) {
          elementJacobian_u_u[i][j] = 0.0;
          elementJacobian_u_n[i][j] = 0.0;
        }
      }
      for (int k = 0; k < nQuadraturePoints_element; k++) {
        int eN_k                  = eN * nQuadraturePoints_element + k, //index to a scalar at a quadrature point
          eN_k_nSpace             = eN_k * nSpace,
            eN_nDOF_trial_element = eN * nDOF_trial_element; //index to a vector at a quadrature point

        //declare local storage
        double u = 0.0, grad_u[nSpace], m = 0.0, dm = 0.0, f[nSpace], df[nSpace], a[nnz], da[nnz], as[nnz], m_t = 0.0, dm_t = 0.0, dpdeResidual_u_u[nDOF_trial_element], Lstar_u[nDOF_test_element], dsubgridError_u_u[nDOF_trial_element], tau = 0.0, tau0 = 0.0, tau1 = 0.0, jac[nSpace * nSpace], jacDet, jacInv[nSpace * nSpace], u_grad_trial[nDOF_trial_element * nSpace], dV, u_test_dV[nDOF_test_element], u_grad_test_dV[nDOF_test_element * nSpace], x, y, z, xt, yt, zt, G[nSpace * nSpace], G_dd_G, tr_G;
        //
        //calculate solution and gradients at quadrature points
        //
        //get jacobian, etc for mapping reference element
        ck.calculateMapping_element(eN, k, mesh_dof.data(), mesh_l2g.data(), mesh_trial_ref.data(), mesh_grad_trial_ref.data(), jac, jacDet, jacInv, x, y, z);
        ck.calculateMappingVelocity_element(eN, k, mesh_velocity_dof.data(), mesh_l2g.data(), mesh_trial_ref.data(), xt, yt, zt);
        //get the physical integration weight
        dV = fabs(jacDet) * dV_ref.data()[k];
        ck.calculateG(jacInv, G, G_dd_G, tr_G);
        //get the trial function gradients
        ck.gradTrialFromRef(&u_grad_trial_ref.data()[k * nDOF_trial_element * nSpace], jacInv, u_grad_trial);
        //get the solution
        ck.valFromDOF(u_dof.data(), &u_l2g.data()[eN_nDOF_trial_element], &u_trial_ref.data()[k * nDOF_trial_element], u);
        //get the solution gradients
        ck.gradFromDOF(u_dof.data(), &u_l2g.data()[eN_nDOF_trial_element], u_grad_trial, grad_u);
        //precalculate test function products with integration weights
        for (int j = 0; j < nDOF_trial_element; j++) {
          u_test_dV[j] = u_test_ref.data()[k * nDOF_trial_element + j] * dV;
          for (int I = 0; I < nSpace; I++) {
            u_grad_test_dV[j * nSpace + I] = u_grad_trial[j * nSpace + I] * dV; //cek warning won't work for Petrov-Galerkin
          }
        }
        //
        // Compositional (p,z) H2O-equation Jacobian (P3c).  Residual:
        //   R^w_i = (phi*N*(1-z))_t * N_i dV + VMS num.diff - F_0.gradN_i dV,
        //   F_0 = rho_g*(1-Y)*u_g + rho_a*(1-X)*u_a.  Differentiate wrt the
        //   primaries p=u_w (->(0,0)) and z=u_n (->(0,1)) and their gradients,
        //   chain-ruled through the analytic flash.  FD-verified in
        //   flux_jac_test.cpp.  (u is p, grad_u is grad p at this QP.)
        //
        const int    mat_eN0     = elementMaterialTypes.data()[eN];
        const double alpha_eN0   = alpha.data()[mat_eN0];
        const double n_vg_eN0    = n.data()[mat_eN0];
        const double krn_end_eN0 = krn_end.data()[mat_eN0];
        const double *KWs_eN0    = &KWs.data()[mat_eN0 * nnz];
        const double phi_eN0     = thetaR.data()[mat_eN0] + thetaSR.data()[mat_eN0];
        const double S_wr_loc0   = thetaR.data()[mat_eN0] / phi_eN0;
        const double one_m_Sr0   = 1.0 - S_wr_loc0;
        const double Se_trap_L1547 = 1.0 - S_gr.data()[mat_eN0] / one_m_Sr0;  // gas-only residual trapping
        double u_n_qp = 0.0, grad_u_n[nSpace];
        ck.valFromDOF(u_dof_n.data(), &u_l2g.data()[eN_nDOF_trial_element],
                      &u_trial_ref.data()[k * nDOF_trial_element], u_n_qp);
        ck.gradFromDOF(u_dof_n.data(), &u_l2g.data()[eN_nDOF_trial_element],
                       u_grad_trial, grad_u_n);
        const double p_cl0 = fmax(u, 1.0e2);
        ::m_comp_co2::flash::FlashState fs0 =
            ::m_comp_co2::flash::flashPZ(p_cl0, u_n_qp, T_C_member, m_NaCl_member, ::m_comp_co2::flash::EPS_Z, immiscible_member);
        const double S_g0 = fs0.S_g, Sa0 = 1.0 - S_g0;
        // accumulation m_0 = phi*N*(1-z), N = rho_g*S_g + rho_a*S_a:
        const double N0     = fs0.rho_g*S_g0 + fs0.rho_a*Sa0;
        const double dN0_dp = fs0.drho_g_dp*S_g0 + fs0.rho_g*fs0.dS_g_dp
                            + fs0.drho_a_dp*Sa0   - fs0.rho_a*fs0.dS_g_dp;
        const double dN0_dz = fs0.drho_g_dz*S_g0 + fs0.rho_g*fs0.dS_g_dz
                            + fs0.drho_a_dz*Sa0   - fs0.rho_a*fs0.dS_g_dz;
        const double dm0_dp = phi_eN0 * dN0_dp * (1.0 - u_n_qp);
        const double dm0_dz = phi_eN0 * (dN0_dz * (1.0 - u_n_qp) - N0);
        // wetting effective saturation from the flash saturation + (p,z) derivs.
        const double Se_raw0 = (Sa0 - S_wr_loc0) / one_m_Sr0;
        double Se_a0, dSe0_dp, dSe0_dz;
        if (Se_raw0 <= 0.0)      { Se_a0 = 0.0; dSe0_dp = 0.0; dSe0_dz = 0.0; }
        else if (Se_raw0 >= 1.0) { Se_a0 = 1.0; dSe0_dp = 0.0; dSe0_dz = 0.0; }
        else { Se_a0 = Se_raw0; dSe0_dp = -fs0.dS_g_dp/one_m_Sr0; dSe0_dz = -fs0.dS_g_dz/one_m_Sr0; }
        double KWr0=0.0, DKWr0=0.0, thW0=0.0, DthW0=0.0, KNr0=0.0, DKNr0=0.0;
        double pc0=0.0, dpc_dSe0=0.0, d2pc0=0.0;
        if (PSK_TYPE_member == 1) {
          proteus::m_comp_co2::psk::bc_wetting_from_Se(Se_a0, alpha_eN0, n_vg_eN0,
              thetaR.data()[mat_eN0], thetaSR.data()[mat_eN0], thW0, DthW0, KWr0, DKWr0);
          proteus::m_comp_co2::psk::bc_kr_nonwetting_from_Se(Se_a0, alpha_eN0, n_vg_eN0, KNr0, DKNr0, Se_trap_L1547);
          proteus::m_comp_co2::psk::bc_pc_from_Se(Se_a0, alpha_eN0, n_vg_eN0, bc_Se_min.data()[mat_eN0], pc0, dpc_dSe0, d2pc0);
        } else {
          proteus::m_comp_co2::psk::vgm_wetting_from_Se(Se_a0, alpha_eN0, n_vg_eN0,
              thetaR.data()[mat_eN0], thetaSR.data()[mat_eN0], thW0, DthW0, KWr0, DKWr0);
          proteus::m_comp_co2::psk::vgm_kr_nonwetting_from_Se(Se_a0, alpha_eN0, n_vg_eN0, KNr0, DKNr0, Se_trap_L1547);
          proteus::m_comp_co2::psk::vgm_pc_from_Se(Se_a0, alpha_eN0, n_vg_eN0, pc0, dpc_dSe0, d2pc0);
        }
        KNr0 *= krn_end_eN0;  DKNr0 *= krn_end_eN0;
        const double pcp0     = dpc_dSe0 / one_m_Sr0;            // pc'(S_a)
        const double dpcp0_dp = (d2pc0 / one_m_Sr0) * dSe0_dp;
        const double dpcp0_dz = (d2pc0 / one_m_Sr0) * dSe0_dz;
        const double dMm0 = ::m_comp_co2::eos::M_CO2_KG - ::m_comp_co2::eos::M_H2O_KG;
        const double Mbar_g0 = fs0.Y*::m_comp_co2::eos::M_CO2_KG + (1.0-fs0.Y)*::m_comp_co2::eos::M_H2O_KG;
        const double Mbar_a0 = fs0.X*::m_comp_co2::eos::M_CO2_KG + (1.0-fs0.X)*::m_comp_co2::eos::M_H2O_KG;
        const double rho_g_mass0 = fs0.rho_g*Mbar_g0, rho_a_mass0 = fs0.rho_a*Mbar_a0;
        const double drgm0_dp = fs0.drho_g_dp*Mbar_g0 + fs0.rho_g*fs0.dY_dp*dMm0;
        const double drgm0_dz =                          fs0.rho_g*fs0.dY_dz*dMm0;
        const double dram0_dp = fs0.drho_a_dp*Mbar_a0 + fs0.rho_a*fs0.dX_dp*dMm0;
        const double dram0_dz = fs0.drho_a_dz*Mbar_a0 + fs0.rho_a*fs0.dX_dz*dMm0;
        // H2O transport coefficients Ag=rho_g*(1-Y), Aa=rho_a*(1-X) + (p,z) derivs.
        const double Ag0 = fs0.rho_g*(1.0-fs0.Y), Aa0 = fs0.rho_a*(1.0-fs0.X);
        const double dAg0_dp = fs0.drho_g_dp*(1.0-fs0.Y) - fs0.rho_g*fs0.dY_dp;
        const double dAg0_dz =                            - fs0.rho_g*fs0.dY_dz;
        const double dAa0_dp = fs0.drho_a_dp*(1.0-fs0.X) - fs0.rho_a*fs0.dX_dp;
        const double dAa0_dz = fs0.drho_a_dz*(1.0-fs0.X) - fs0.rho_a*fs0.dX_dz;
        // per-direction Darcy velocities + value-block partials (gradients fixed).
        double ug0[nSpace], ua0[nSpace];
        double dug0_dp[nSpace], dug0_dz[nSpace], dua0_dp[nSpace], dua0_dz[nSpace];
        for (int I = 0; I < nSpace; I++) {
          double ugI=0.0, uaI=0.0, dugp=0.0, dugz=0.0, duap=0.0, duaz=0.0;
          for (int ii = a_rowptr.data()[I]; ii < a_rowptr.data()[I + 1]; ii++) {
            const int J = a_colind.data()[ii];
            const double Kii = KWs_eN0[ii];
            const double Mob_g = KNr0*Kii/mu_n, Mob_a = KWr0*Kii;
            const double dMobg_dp = (DKNr0*Kii/mu_n)*dSe0_dp, dMobg_dz = (DKNr0*Kii/mu_n)*dSe0_dz;
            const double dMoba_dp = (DKWr0*Kii)*dSe0_dp,        dMoba_dz = (DKWr0*Kii)*dSe0_dz;
            const double gJ = gravity.data()[J];
            const double gradSa = -(fs0.dS_g_dp*grad_u[J] + fs0.dS_g_dz*grad_u_n[J]);
            const double gp_a = grad_u[J] - rho_a_mass0*gJ;
            const double gp_g = grad_u[J] + pcp0*gradSa - rho_g_mass0*gJ;
            ugI -= Mob_g*gp_g;  uaI -= Mob_a*gp_a;
            const double dgradSa_dp = -(fs0.d2S_g_dp2 *grad_u[J] + fs0.d2S_g_dpdz*grad_u_n[J]);
            const double dgradSa_dz = -(fs0.d2S_g_dpdz*grad_u[J] + fs0.d2S_g_dz2 *grad_u_n[J]);
            const double dgpg_dp = dpcp0_dp*gradSa + pcp0*dgradSa_dp - drgm0_dp*gJ;
            const double dgpg_dz = dpcp0_dz*gradSa + pcp0*dgradSa_dz - drgm0_dz*gJ;
            dugp -= dMobg_dp*gp_g + Mob_g*dgpg_dp;
            dugz -= dMobg_dz*gp_g + Mob_g*dgpg_dz;
            duap -= dMoba_dp*gp_a + Mob_a*(-dram0_dp*gJ);
            duaz -= dMoba_dz*gp_a + Mob_a*(-dram0_dz*gJ);
          }
          ug0[I]=ugI; ua0[I]=uaI;
          dug0_dp[I]=dugp; dug0_dz[I]=dugz; dua0_dp[I]=duap; dua0_dz[I]=duaz;
        }
        // Assemble (0,0) [d/dp] and (0,1) [d/dz].
        //   dR^w_i/d(p_j) = alphaBDF*dm0/dp * N_j N_i dV
        //                 - [ (dF_0/dp)_val N_j + (dF_0/dgrad_p).gradN_j ].gradN_i dV
        //                 + VMS num.diff
        for (int i = 0; i < nDOF_test_element; i++) {
          const int i_nSpace = i * nSpace;
          double Sval_p = 0.0, Sval_z = 0.0;
          for (int I = 0; I < nSpace; I++) {
            const double gNiI = u_grad_trial[i_nSpace + I];
            Sval_p += (dAg0_dp*ug0[I] + Ag0*dug0_dp[I] + dAa0_dp*ua0[I] + Aa0*dua0_dp[I]) * gNiI;
            Sval_z += (dAg0_dz*ug0[I] + Ag0*dug0_dz[I] + dAa0_dz*ua0[I] + Aa0*dua0_dz[I]) * gNiI;
          }
          for (int j = 0; j < nDOF_trial_element; j++) {
            const int j_nSpace = j * nSpace;
            const double trial_j = u_trial_ref.data()[k * nDOF_trial_element + j];
            double Sgrad_p = 0.0, Sgrad_z = 0.0;
            for (int I = 0; I < nSpace; I++) {
              const double gNiI = u_grad_trial[i_nSpace + I];
              for (int ii = a_rowptr.data()[I]; ii < a_rowptr.data()[I + 1]; ii++) {
                const int J = a_colind.data()[ii];
                const double Kii = KWs_eN0[ii];
                const double Mob_g = KNr0*Kii/mu_n, Mob_a = KWr0*Kii;
                const double gNjJ = u_grad_trial[j_nSpace + J];
                const double dFdgp = Ag0*(-Mob_g*(1.0 - pcp0*fs0.dS_g_dp)) + Aa0*(-Mob_a);
                const double dFdgz = Ag0*(-Mob_g*(-pcp0*fs0.dS_g_dz));
                Sgrad_p += dFdgp * gNjJ * gNiI;
                Sgrad_z += dFdgz * gNjJ * gNiI;
              }
            }
            elementJacobian_u_u[i][j] += alphaBDF * dm0_dp * trial_j * u_test_dV[i];
            elementJacobian_u_u[i][j] -= (Sval_p * trial_j + Sgrad_p) * dV;
            elementJacobian_u_u[i][j] += VMS * ck.NumericalDiffusionJacobian(
                q_numDiff_u_last[eN_k], &u_grad_trial[j_nSpace], &u_grad_test_dV[i_nSpace]);
            elementJacobian_u_n[i][j] += alphaBDF * dm0_dz * trial_j * u_test_dV[i];
            elementJacobian_u_n[i][j] -= (Sval_z * trial_j + Sgrad_z) * dV;
          }
        }
      } //k
      //
      //load into element Jacobian into global Jacobian
      //
      for (int i = 0; i < nDOF_test_element; i++) {
        int eN_i = eN * nDOF_test_element + i;
        for (int j = 0; j < nDOF_trial_element; j++) {
          int eN_i_j = eN_i * nDOF_trial_element + j;
          globalJacobian.data()[csrRowIndeces_u_u[eN_i] + csrColumnOffsets_u_u[eN_i_j]] += elementJacobian_u_u[i][j];
          // load (0,1) cross-block coupling the wetting eq to S_n through
          // dm/du_n, df/du_n, da/du_n.
          globalJacobian.data()[csrRowIndeces_w_n.data()[eN_i] + csrColumnOffsets_w_n.data()[eN_i_j]] += elementJacobian_u_n[i][j];
        } //j
      } //i
    } //elements
    //
    //loop over exterior element boundaries to compute the surface integrals and load them into the global Jacobian
    //
    for (int ebNE = 0; ebNE < nExteriorElementBoundaries_global; ebNE++) {
      int ebN = exteriorElementBoundariesArray.data()[ebNE];
      int eN = elementBoundaryElementsArray.data()[ebN * 2 + 0], ebN_local = elementBoundaryLocalElementBoundariesArray.data()[ebN * 2 + 0], eN_nDOF_trial_element = eN * nDOF_trial_element;
      for (int kb = 0; kb < nQuadraturePoints_elementBoundary; kb++) {
        int ebNE_kb = ebNE * nQuadraturePoints_elementBoundary + kb, ebNE_kb_nSpace = ebNE_kb * nSpace, ebN_local_kb = ebN_local * nQuadraturePoints_elementBoundary + kb, ebN_local_kb_nSpace = ebN_local_kb * nSpace;

        double u_ext = 0.0, grad_u_ext[nSpace], m_ext = 0.0, dm_ext = 0.0, f_ext[nSpace], df_ext[nSpace], a_ext[nnz], da_ext[nnz], as_ext[nnz], dflux_u_u_ext = 0.0, bc_u_ext = 0.0,
               //bc_grad_u_ext[nSpace],
          bc_m_ext = 0.0, bc_dm_ext = 0.0, bc_f_ext[nSpace], bc_df_ext[nSpace], bc_a_ext[nnz], bc_da_ext[nnz], bc_as_ext[nnz], fluxJacobian_u_u[nDOF_trial_element], jac_ext[nSpace * nSpace], jacDet_ext, jacInv_ext[nSpace * nSpace], boundaryJac[nSpace * (nSpace - 1)], metricTensor[(nSpace - 1) * (nSpace - 1)], metricTensorDetSqrt, dS, u_test_dS[nDOF_test_element], u_grad_trial_trace[nDOF_trial_element * nSpace], normal[3], x_ext, y_ext, z_ext, xt_ext, yt_ext, zt_ext, integralScaling, G[nSpace * nSpace], G_dd_G, tr_G;
        //
        //calculate the solution and gradients at quadrature points
        //
        ck.calculateMapping_elementBoundary(eN, ebN_local, kb, ebN_local_kb, mesh_dof.data(), mesh_l2g.data(), mesh_trial_trace_ref.data(), mesh_grad_trial_trace_ref.data(), boundaryJac_ref.data(), jac_ext, jacDet_ext, jacInv_ext, boundaryJac, metricTensor, metricTensorDetSqrt,
                                            normal_ref.data(), normal, x_ext, y_ext, z_ext);
        ck.calculateMappingVelocity_elementBoundary(eN, ebN_local, kb, ebN_local_kb, mesh_velocity_dof.data(), mesh_l2g.data(), mesh_trial_trace_ref.data(), xt_ext, yt_ext, zt_ext, normal, boundaryJac, metricTensor, integralScaling);
        dS = ((1.0 - MOVING_DOMAIN) * metricTensorDetSqrt + MOVING_DOMAIN * integralScaling) * dS_ref.data()[kb];
        ck.calculateG(jacInv_ext, G, G_dd_G, tr_G);
        //compute shape and solution information
        //shape
        ck.gradTrialFromRef(&u_grad_trial_trace_ref.data()[ebN_local_kb_nSpace * nDOF_trial_element], jacInv_ext, u_grad_trial_trace);
        //solution and gradients
        ck.valFromDOF(u_dof.data(), &u_l2g.data()[eN_nDOF_trial_element], &u_trial_trace_ref.data()[ebN_local_kb * nDOF_test_element], u_ext);
        ck.gradFromDOF(u_dof.data(), &u_l2g.data()[eN_nDOF_trial_element], u_grad_trial_trace, grad_u_ext);
        //precalculate test function products with integration weights
        for (int j = 0; j < nDOF_trial_element; j++) { u_test_dS[j] = u_test_trace_ref.data()[ebN_local_kb * nDOF_test_element + j] * dS; }
        //
        //load the boundary values
        //
        bc_u_ext = isDOFBoundary_u.data()[ebNE_kb] * ebqe_bc_u_ext.data()[ebNE_kb] + (1 - isDOFBoundary_u.data()[ebNE_kb]) * u_ext;
        //
        //calculate the internal and external trace of the pde coefficients
        //
        double Kr, dKr, thetaW, thetaW_bc;
        const double rho_ext = ebqe_rho.data()[ebNE_kb];
        // Boundary closure for the wetting-equation Jacobian. (0,1) boundary
        // contribution is not assembled here.
        double dm_du_n_ext = 0.0, dkr_du_n_ext = 0.0;
        double df_du_n_ext[nSpace];
        double da_du_n_ext[nnz];
        double bc_dm_du_n = 0.0, bc_dkr_du_n = 0.0;
        double bc_df_du_n[nSpace];
        double bc_da_du_n[nnz];
        for (int I = 0; I < nSpace; I++) { df_du_n_ext[I] = 0.0; bc_df_du_n[I] = 0.0; }
        for (int ii = 0; ii < nnz; ii++)  { da_du_n_ext[ii] = 0.0; bc_da_du_n[ii] = 0.0; }
        // u_n_ext_qp & bc_u_n_ext_qp held at outer scope for the (0,1) boundary
        // Jacobian assembly below.
        double u_n_ext_qp_outer = 0.0;
        double bc_u_n_ext_qp_outer = 0.0;
        {
          double u_n_ext_qp = 0.0;
          ck.valFromDOF(u_dof_n.data(), &u_l2g.data()[eN_nDOF_trial_element],
                        &u_trial_trace_ref.data()[ebN_local_kb * nDOF_test_element], u_n_ext_qp);
          const double bc_u_n_ext_qp = isDOFBoundary_n.data()[ebNE_kb] * ebqe_bc_u_n_ext.data()[ebNE_kb]
                                     + (1 - isDOFBoundary_n.data()[ebNE_kb]) * u_n_ext_qp;
          u_n_ext_qp_outer    = u_n_ext_qp;
          bc_u_n_ext_qp_outer = bc_u_n_ext_qp;
          evaluateCoefficients_from_Se(a_rowptr.data(), a_colind.data(), rho, rho_ext, beta, gravity.data(),
                                       alpha.data()[elementMaterialTypes.data()[eN]], n.data()[elementMaterialTypes.data()[eN]],
                                       thetaR.data()[elementMaterialTypes.data()[eN]], thetaSR.data()[elementMaterialTypes.data()[eN]],
                                       &KWs.data()[elementMaterialTypes.data()[eN] * nnz], u_ext, u_n_ext_qp,
                                       m_ext, dm_ext, dm_du_n_ext, f_ext, df_ext, df_du_n_ext, a_ext, da_ext, da_du_n_ext,
                                       as_ext, Kr, dKr, dkr_du_n_ext, thetaW);
          evaluateCoefficients_from_Se(a_rowptr.data(), a_colind.data(), rho, rho_ext, beta, gravity.data(),
                                       alpha.data()[elementMaterialTypes.data()[eN]], n.data()[elementMaterialTypes.data()[eN]],
                                       thetaR.data()[elementMaterialTypes.data()[eN]], thetaSR.data()[elementMaterialTypes.data()[eN]],
                                       &KWs.data()[elementMaterialTypes.data()[eN] * nnz], bc_u_ext, bc_u_n_ext_qp,
                                       bc_m_ext, bc_dm_ext, bc_dm_du_n, bc_f_ext, bc_df_ext, bc_df_du_n, bc_a_ext, bc_da_ext, bc_da_du_n,
                                       bc_as_ext, Kr, dKr, bc_dkr_du_n, thetaW_bc);
        }
        //
        // ===== P3c boundary: compositional comp-0 (H2O) flux Jacobian =====
        // d(flux_0)/d(p_j) -> fluxJacobian_u_u, d(flux_0)/d(z_j) -> fluxJacobian_u_n,
        // where flux_0 = F_0.n + penalty*(p - p_BC).  Mirrors the interior comp-0
        // flux Jacobian (grad N_i -> normal, + penalty*trial_j); the F_0 value/grad
        // partials are FD-verified (boundary0_test.cpp / flux_jac_test.cpp).
        // Nonzero only on Dirichlet-p faces; no-flow faces contribute 0.
        double fluxJacobian_u_n[nDOF_trial_element];
        for (int j = 0; j < nDOF_trial_element; j++) { fluxJacobian_u_u[j] = 0.0; fluxJacobian_u_n[j] = 0.0; }
        if (isDOFBoundary_u.data()[ebNE_kb]) {
          double grad_u_n_ext[nSpace];
          ck.gradFromDOF(u_dof_n.data(), &u_l2g.data()[eN_nDOF_trial_element],
                         u_grad_trial_trace, grad_u_n_ext);
          const int    mat_b   = elementMaterialTypes.data()[eN];
          const double alpha_b = alpha.data()[mat_b];
          const double n_vg_b  = n.data()[mat_b];
          const double krn_end_b = krn_end.data()[mat_b];
          const double *KWs_b  = &KWs.data()[mat_b * nnz];
          const double phi_b   = thetaR.data()[mat_b] + thetaSR.data()[mat_b];
          const double S_wr_b  = thetaR.data()[mat_b] / phi_b;
          const double one_m_Sr_b = 1.0 - S_wr_b;
          const double Se_trap_L1775 = 1.0 - S_gr.data()[mat_b] / one_m_Sr_b;  // gas-only residual trapping
          const double p_clb   = fmax(u_ext, 1.0e2);
          ::m_comp_co2::flash::FlashState fsb =
              ::m_comp_co2::flash::flashPZ(p_clb, u_n_ext_qp_outer, T_C_member, m_NaCl_member, ::m_comp_co2::flash::EPS_Z, immiscible_member);
          const double Sab = 1.0 - fsb.S_g;
          const double Se_rawb = (Sab - S_wr_b)/one_m_Sr_b;
          double Se_ab, dSeb_dp, dSeb_dz;
          if (Se_rawb<=0.0){Se_ab=0.0;dSeb_dp=0.0;dSeb_dz=0.0;}
          else if (Se_rawb>=1.0){Se_ab=1.0;dSeb_dp=0.0;dSeb_dz=0.0;}
          else {Se_ab=Se_rawb;dSeb_dp=-fsb.dS_g_dp/one_m_Sr_b;dSeb_dz=-fsb.dS_g_dz/one_m_Sr_b;}
          double KWrb=0,DKWrb=0,thWb=0,DthWb=0,KNrb=0,DKNrb=0,pcb=0,dpc_dSeb=0,d2pcb=0;
          if (PSK_TYPE_member == 1) {
            proteus::m_comp_co2::psk::bc_wetting_from_Se(Se_ab, alpha_b, n_vg_b, thetaR.data()[mat_b], thetaSR.data()[mat_b], thWb, DthWb, KWrb, DKWrb);
            proteus::m_comp_co2::psk::bc_kr_nonwetting_from_Se(Se_ab, alpha_b, n_vg_b, KNrb, DKNrb, Se_trap_L1775);
            proteus::m_comp_co2::psk::bc_pc_from_Se(Se_ab, alpha_b, n_vg_b, bc_Se_min.data()[mat_b], pcb, dpc_dSeb, d2pcb);
          } else {
            proteus::m_comp_co2::psk::vgm_wetting_from_Se(Se_ab, alpha_b, n_vg_b, thetaR.data()[mat_b], thetaSR.data()[mat_b], thWb, DthWb, KWrb, DKWrb);
            proteus::m_comp_co2::psk::vgm_kr_nonwetting_from_Se(Se_ab, alpha_b, n_vg_b, KNrb, DKNrb, Se_trap_L1775);
            proteus::m_comp_co2::psk::vgm_pc_from_Se(Se_ab, alpha_b, n_vg_b, pcb, dpc_dSeb, d2pcb);
          }
          KNrb *= krn_end_b; DKNrb *= krn_end_b;
          const double pcpb     = dpc_dSeb / one_m_Sr_b;
          const double dpcpb_dp = (d2pcb / one_m_Sr_b) * dSeb_dp;
          const double dpcpb_dz = (d2pcb / one_m_Sr_b) * dSeb_dz;
          const double dMmb = ::m_comp_co2::eos::M_CO2_KG - ::m_comp_co2::eos::M_H2O_KG;
          const double Mbar_gb = fsb.Y*::m_comp_co2::eos::M_CO2_KG + (1.0-fsb.Y)*::m_comp_co2::eos::M_H2O_KG;
          const double Mbar_ab = fsb.X*::m_comp_co2::eos::M_CO2_KG + (1.0-fsb.X)*::m_comp_co2::eos::M_H2O_KG;
          const double rgmb = fsb.rho_g*Mbar_gb, ramb = fsb.rho_a*Mbar_ab;
          const double drgmb_dp = fsb.drho_g_dp*Mbar_gb + fsb.rho_g*fsb.dY_dp*dMmb;
          const double drgmb_dz =                          fsb.rho_g*fsb.dY_dz*dMmb;
          const double dramb_dp = fsb.drho_a_dp*Mbar_ab + fsb.rho_a*fsb.dX_dp*dMmb;
          const double dramb_dz = fsb.drho_a_dz*Mbar_ab + fsb.rho_a*fsb.dX_dz*dMmb;
          const double Agb = fsb.rho_g*(1.0-fsb.Y), Aab = fsb.rho_a*(1.0-fsb.X);
          const double dAgb_dp = fsb.drho_g_dp*(1.0-fsb.Y) - fsb.rho_g*fsb.dY_dp;
          const double dAgb_dz =                            - fsb.rho_g*fsb.dY_dz;
          const double dAab_dp = fsb.drho_a_dp*(1.0-fsb.X) - fsb.rho_a*fsb.dX_dp;
          const double dAab_dz = fsb.drho_a_dz*(1.0-fsb.X) - fsb.rho_a*fsb.dX_dz;
          double ugb[nSpace], uab[nSpace], dugb_dp[nSpace], dugb_dz[nSpace], duab_dp[nSpace], duab_dz[nSpace];
          for (int I=0;I<nSpace;I++){
            double ugI=0,uaI=0,dugp=0,dugz=0,duap=0,duaz=0;
            for (int ii=a_rowptr.data()[I];ii<a_rowptr.data()[I+1];ii++){
              const int J=a_colind.data()[ii];
              const double Kii=KWs_b[ii];
              const double Mob_g=KNrb*Kii/mu_n, Mob_a=KWrb*Kii;
              const double dMobg_dp=(DKNrb*Kii/mu_n)*dSeb_dp, dMobg_dz=(DKNrb*Kii/mu_n)*dSeb_dz;
              const double dMoba_dp=(DKWrb*Kii)*dSeb_dp, dMoba_dz=(DKWrb*Kii)*dSeb_dz;
              const double gJ=gravity.data()[J];
              const double gradSa=-(fsb.dS_g_dp*grad_u_ext[J]+fsb.dS_g_dz*grad_u_n_ext[J]);
              const double gp_a=grad_u_ext[J]-ramb*gJ;
              const double gp_g=grad_u_ext[J]+pcpb*gradSa-rgmb*gJ;
              ugI-=Mob_g*gp_g; uaI-=Mob_a*gp_a;
              const double dgradSa_dp=-(fsb.d2S_g_dp2*grad_u_ext[J]+fsb.d2S_g_dpdz*grad_u_n_ext[J]);
              const double dgradSa_dz=-(fsb.d2S_g_dpdz*grad_u_ext[J]+fsb.d2S_g_dz2*grad_u_n_ext[J]);
              const double dgpg_dp=dpcpb_dp*gradSa+pcpb*dgradSa_dp-drgmb_dp*gJ;
              const double dgpg_dz=dpcpb_dz*gradSa+pcpb*dgradSa_dz-drgmb_dz*gJ;
              dugp-=dMobg_dp*gp_g+Mob_g*dgpg_dp;
              dugz-=dMobg_dz*gp_g+Mob_g*dgpg_dz;
              duap-=dMoba_dp*gp_a+Mob_a*(-dramb_dp*gJ);
              duaz-=dMoba_dz*gp_a+Mob_a*(-dramb_dz*gJ);
            }
            ugb[I]=ugI;uab[I]=uaI;dugb_dp[I]=dugp;dugb_dz[I]=dugz;duab_dp[I]=duap;duab_dz[I]=duaz;
          }
          const double penb = ebqe_penalty_ext.data()[ebNE_kb];
          for (int j=0;j<nDOF_trial_element;j++){
            const int j_nSpace = j*nSpace;
            const double trial_j = u_trial_trace_ref.data()[ebN_local_kb * nDOF_test_element + j];
            double accp=0.0, accz=0.0;
            for (int I=0;I<nSpace;I++){
              const double nI = normal[I];
              const double valp = dAgb_dp*ugb[I]+Agb*dugb_dp[I]+dAab_dp*uab[I]+Aab*duab_dp[I];
              const double valz = dAgb_dz*ugb[I]+Agb*dugb_dz[I]+dAab_dz*uab[I]+Aab*duab_dz[I];
              double gradp=0.0, gradz=0.0;
              for (int ii=a_rowptr.data()[I];ii<a_rowptr.data()[I+1];ii++){
                const int J=a_colind.data()[ii];
                const double Kii=KWs_b[ii];
                const double Mob_g=KNrb*Kii/mu_n, Mob_a=KWrb*Kii;
                const double gNjJ = u_grad_trial_trace[j_nSpace + J];
                const double dFdgp = Agb*(-Mob_g*(1.0 - pcpb*fsb.dS_g_dp)) + Aab*(-Mob_a);
                const double dFdgz = Agb*(-Mob_g*(-pcpb*fsb.dS_g_dz));
                gradp += dFdgp*gNjJ;
                gradz += dFdgz*gNjJ;
              }
              accp += nI*(valp*trial_j + gradp);
              accz += nI*(valz*trial_j + gradz);
            }
            fluxJacobian_u_u[j] = accp + penb*trial_j;
            fluxJacobian_u_n[j] = accz;
          }
        }
        //
        //update the global Jacobian from the flux Jacobian
        //
        for (int i = 0; i < nDOF_test_element; i++) {
          int eN_i = eN * nDOF_test_element + i;
          for (int j = 0; j < nDOF_trial_element; j++) {
            int ebN_i_j = ebN * 4 * nDOF_test_X_trial_element + i * nDOF_trial_element + j;
            globalJacobian.data()[csrRowIndeces_u_u[eN_i] + csrColumnOffsets_eb_u_u[ebN_i_j]] += fluxJacobian_u_u[j] * u_test_dS[i];
            // (0,1) cross-block boundary contribution (nonzero only on
            // Dirichlet faces for u_w).
            globalJacobian.data()[csrRowIndeces_w_n.data()[eN_i] + csrColumnOffsets_eb_w_n.data()[ebN_i_j]] += fluxJacobian_u_n[j] * u_test_dS[i];
          } //j
        } //i
      } //kb
    } //ebNE

    // ============================================================================
    // Compositional (p,z) component-1 (CO2) Jacobian -- accumulation + component
    // molar flux F_1 = rho_g*Y*u_g + rho_a*X*u_a, all flash-derived.  u_w = p,
    // u_n = z (overall CO2 mole fraction; NOT a saturation).
    //
    //   m_1 = phi*N*z,  N = rho_g*S_g + rho_a*S_a   (S_g, rho_a, rho_g from flash)
    //
    //   J_(1,1)[i,j] = dm_1/dz/dt * N_i N_j dV
    //                - [ (dF_1/dz)_val N_j + (dF_1/dgrad_z).gradN_j ].gradN_i dV
    //   J_(1,0)[i,j] = dm_1/dp/dt * N_i N_j dV
    //                - [ (dF_1/dp)_val N_j + (dF_1/dgrad_p).gradN_j ].gradN_i dV
    //
    // (0,1) cross-block (wetting eq dependence on z) is assembled in the element
    // loop above.  Per-QP flux partials FD-verified in flux_jac_test.cpp.
    // ============================================================================
    for (int eN = 0; eN < nElements_global; eN++) {
      const int    mat_eN    = elementMaterialTypes.data()[eN];
      const double phi_eN    = thetaR.data()[mat_eN] + thetaSR.data()[mat_eN];
      const double alpha_eN  = alpha.data()[mat_eN];
      const double krn_end_eN = krn_end.data()[mat_eN];
      const double n_vg_eN   = n.data()[mat_eN];
      const double *KWs_eN   = &KWs.data()[mat_eN * nnz];
      // dm_n_du_n is no longer hoisted: ρ_n depends on the Newton iterate
      // through the linear EOS, so phi*ρ_n*S_n derivative is QP-local.
      double elementJacobian_n_n[nDOF_test_element][nDOF_trial_element];
      double elementJacobian_n_w[nDOF_test_element][nDOF_trial_element];
      for (int i = 0; i < nDOF_test_element; i++)
        for (int j = 0; j < nDOF_trial_element; j++) {
          elementJacobian_n_n[i][j] = 0.0;
          elementJacobian_n_w[i][j] = 0.0;
        }
      for (int k = 0; k < nQuadraturePoints_element; k++) {
        const int eN_k = eN * nQuadraturePoints_element + k;
        const int eN_nDOF_trial_element = eN * nDOF_trial_element;
        double jac[nSpace * nSpace], jacDet, jacInv[nSpace * nSpace], x_q, y_q, z_q;
        ck.calculateMapping_element(eN, k, mesh_dof.data(), mesh_l2g.data(),
                                    mesh_trial_ref.data(), mesh_grad_trial_ref.data(),
                                    jac, jacDet, jacInv, x_q, y_q, z_q);
        const double dV = std::fabs(jacDet) * dV_ref.data()[k];
        // Trial gradients in physical coords.
        double u_grad_trial_qp[nDOF_trial_element * nSpace];
        ck.gradTrialFromRef(&u_grad_trial_ref.data()[k * nDOF_trial_element * nSpace],
                            jacInv, u_grad_trial_qp);
        // Saturation u_n at QP.
        double u_n = 0.0;
        ck.valFromDOF(u_dof_n.data(),
                      &u_l2g.data()[eN_nDOF_trial_element],
                      &u_trial_ref.data()[k * nDOF_trial_element], u_n);
        // Wetting-pressure value (for linear EOS) and gradient at QP.
        double u_w_qp = 0.0;
        ck.valFromDOF(u_dof.data(),
                      &u_l2g.data()[eN_nDOF_trial_element],
                      &u_trial_ref.data()[k * nDOF_trial_element], u_w_qp);
        double grad_u_w[nSpace];
        ck.gradFromDOF(u_dof.data(),
                       &u_l2g.data()[eN_nDOF_trial_element],
                       u_grad_trial_qp, grad_u_w);
        // Step 3d: saturation gradient at QP for the capillary diffusion term.
        double grad_u_n[nSpace];
        ck.gradFromDOF(u_dof_n.data(),
                       &u_l2g.data()[eN_nDOF_trial_element],
                       u_grad_trial_qp, grad_u_n);
        // =====================================================================
        // Compositional (p,z) CO2-equation Jacobian (P3c). The flux is the
        // component molar flux F_1 = rho_g*Y*u_g + rho_a*X*u_a assembled in the
        // residual; here we differentiate it (and the accumulation m_1=phi*N*z)
        // wrt the primaries (p=u_w, z=u_n) AND their gradients, chain-ruled
        // through the analytic flash.  Formulas FD-verified in flux_jac_test.cpp.
        // =====================================================================
        const double p_cl_j = fmax(u_w_qp, 1.0e2);
        ::m_comp_co2::flash::FlashState fs_j =
            ::m_comp_co2::flash::flashPZ(p_cl_j, u_n, T_C_member, m_NaCl_member, ::m_comp_co2::flash::EPS_Z, immiscible_member);
        const double S_g_j   = fs_j.S_g, Sa_j = 1.0 - S_g_j;
        // accumulation m_1 = phi*N*z, N = rho_g*S_g + rho_a*S_a:
        const double N_j     = fs_j.rho_g*S_g_j + fs_j.rho_a*Sa_j;
        const double dN_dp_j = fs_j.drho_g_dp*S_g_j + fs_j.rho_g*fs_j.dS_g_dp
                             + fs_j.drho_a_dp*Sa_j   - fs_j.rho_a*fs_j.dS_g_dp;
        const double dN_dz_j = fs_j.drho_g_dz*S_g_j + fs_j.rho_g*fs_j.dS_g_dz
                             + fs_j.drho_a_dz*Sa_j   - fs_j.rho_a*fs_j.dS_g_dz;
        const double dm_n_du_n = phi_eN * (dN_dz_j * u_n + N_j);   // d m_1/dz
        const double dm_n_du_w = phi_eN * (dN_dp_j * u_n);          // d m_1/dp
        // wetting effective saturation from the FLASH saturation + (p,z) derivs.
        const double S_wr_loc     = thetaR.data()[mat_eN] / phi_eN;
        const double one_m_Sr_loc = 1.0 - S_wr_loc;
        const double Se_trap_L1965 = 1.0 - S_gr.data()[mat_eN] / one_m_Sr_loc;  // gas-only residual trapping
        const double Se_raw_j = (Sa_j - S_wr_loc) / one_m_Sr_loc;
        double Se_a, dSe_dp, dSe_dz;
        if (Se_raw_j <= 0.0)      { Se_a = 0.0; dSe_dp = 0.0; dSe_dz = 0.0; }
        else if (Se_raw_j >= 1.0) { Se_a = 1.0; dSe_dp = 0.0; dSe_dz = 0.0; }
        else { Se_a = Se_raw_j; dSe_dp = -fs_j.dS_g_dp/one_m_Sr_loc; dSe_dz = -fs_j.dS_g_dz/one_m_Sr_loc; }
        double KWr_a=0.0, DKWr_a=0.0, thW_a=0.0, DthW_a=0.0, KNr_a=0.0, DKNr_a=0.0;
        double pc_a=0.0, dpc_dSe_a=0.0, d2pc_a=0.0;
        if (PSK_TYPE_member == 1) {
          proteus::m_comp_co2::psk::bc_wetting_from_Se(Se_a, alpha_eN, n_vg_eN,
              thetaR.data()[mat_eN], thetaSR.data()[mat_eN], thW_a, DthW_a, KWr_a, DKWr_a);
          proteus::m_comp_co2::psk::bc_kr_nonwetting_from_Se(Se_a, alpha_eN, n_vg_eN, KNr_a, DKNr_a, Se_trap_L1965);
          proteus::m_comp_co2::psk::bc_pc_from_Se(Se_a, alpha_eN, n_vg_eN, bc_Se_min.data()[mat_eN], pc_a, dpc_dSe_a, d2pc_a);
        } else {
          proteus::m_comp_co2::psk::vgm_wetting_from_Se(Se_a, alpha_eN, n_vg_eN,
              thetaR.data()[mat_eN], thetaSR.data()[mat_eN], thW_a, DthW_a, KWr_a, DKWr_a);
          proteus::m_comp_co2::psk::vgm_kr_nonwetting_from_Se(Se_a, alpha_eN, n_vg_eN, KNr_a, DKNr_a, Se_trap_L1965);
          proteus::m_comp_co2::psk::vgm_pc_from_Se(Se_a, alpha_eN, n_vg_eN, pc_a, dpc_dSe_a, d2pc_a);
        }
        KNr_a *= krn_end_eN;  DKNr_a *= krn_end_eN;
        const double pcp_a   = dpc_dSe_a / one_m_Sr_loc;            // pc'(S_a)
        const double dpcp_dp = (d2pc_a / one_m_Sr_loc) * dSe_dp;    // d pc'(S_a)/dp
        const double dpcp_dz = (d2pc_a / one_m_Sr_loc) * dSe_dz;
        // mass densities for gravity (molar density * mean molar mass) + derivs.
        const double dMm = ::m_comp_co2::eos::M_CO2_KG - ::m_comp_co2::eos::M_H2O_KG;
        const double Mbar_g = fs_j.Y*::m_comp_co2::eos::M_CO2_KG + (1.0-fs_j.Y)*::m_comp_co2::eos::M_H2O_KG;
        const double Mbar_a = fs_j.X*::m_comp_co2::eos::M_CO2_KG + (1.0-fs_j.X)*::m_comp_co2::eos::M_H2O_KG;
        const double rho_g_mass = fs_j.rho_g*Mbar_g, rho_a_mass = fs_j.rho_a*Mbar_a;
        const double drgm_dp = fs_j.drho_g_dp*Mbar_g + fs_j.rho_g*fs_j.dY_dp*dMm;
        const double drgm_dz =                          fs_j.rho_g*fs_j.dY_dz*dMm;
        const double dram_dp = fs_j.drho_a_dp*Mbar_a + fs_j.rho_a*fs_j.dX_dp*dMm;
        const double dram_dz = fs_j.drho_a_dz*Mbar_a + fs_j.rho_a*fs_j.dX_dz*dMm;
        // CO2 transport coefficients Ag=rho_g*Y, Aa=rho_a*X + (p,z) derivatives.
        const double Ag = fs_j.rho_g*fs_j.Y, Aa = fs_j.rho_a*fs_j.X;
        const double dAg_dp = fs_j.drho_g_dp*fs_j.Y + fs_j.rho_g*fs_j.dY_dp;
        const double dAg_dz =                          fs_j.rho_g*fs_j.dY_dz;
        const double dAa_dp = fs_j.drho_a_dp*fs_j.X + fs_j.rho_a*fs_j.dX_dp;
        const double dAa_dz = fs_j.drho_a_dz*fs_j.X + fs_j.rho_a*fs_j.dX_dz;
        // per-direction Darcy velocities ug[I], ua[I] + their value-block partials
        // (gradients held fixed). Mobilities carry the 1/mu_n (gas) factor.
        double ug[nSpace], ua[nSpace];
        double dug_dp[nSpace], dug_dz[nSpace], dua_dp[nSpace], dua_dz[nSpace];
        for (int I = 0; I < nSpace; I++) {
          double ugI=0.0, uaI=0.0, dugp=0.0, dugz=0.0, duap=0.0, duaz=0.0;
          for (int ii = a_rowptr.data()[I]; ii < a_rowptr.data()[I + 1]; ii++) {
            const int J = a_colind.data()[ii];
            const double Kii = KWs_eN[ii];
            const double Mob_g = KNr_a*Kii/mu_n, Mob_a = KWr_a*Kii;
            const double dMobg_dp = (DKNr_a*Kii/mu_n)*dSe_dp, dMobg_dz = (DKNr_a*Kii/mu_n)*dSe_dz;
            const double dMoba_dp = (DKWr_a*Kii)*dSe_dp,        dMoba_dz = (DKWr_a*Kii)*dSe_dz;
            const double gJ = gravity.data()[J];
            const double gradSa = -(fs_j.dS_g_dp*grad_u_w[J] + fs_j.dS_g_dz*grad_u_n[J]);
            const double gp_a = grad_u_w[J] - rho_a_mass*gJ;
            const double gp_g = grad_u_w[J] + pcp_a*gradSa - rho_g_mass*gJ;
            ugI -= Mob_g*gp_g;  uaI -= Mob_a*gp_a;
            const double dgradSa_dp = -(fs_j.d2S_g_dp2 *grad_u_w[J] + fs_j.d2S_g_dpdz*grad_u_n[J]);
            const double dgradSa_dz = -(fs_j.d2S_g_dpdz*grad_u_w[J] + fs_j.d2S_g_dz2 *grad_u_n[J]);
            const double dgpg_dp = dpcp_dp*gradSa + pcp_a*dgradSa_dp - drgm_dp*gJ;
            const double dgpg_dz = dpcp_dz*gradSa + pcp_a*dgradSa_dz - drgm_dz*gJ;
            dugp -= dMobg_dp*gp_g + Mob_g*dgpg_dp;
            dugz -= dMobg_dz*gp_g + Mob_g*dgpg_dz;
            duap -= dMoba_dp*gp_a + Mob_a*(-dram_dp*gJ);
            duaz -= dMoba_dz*gp_a + Mob_a*(-dram_dz*gJ);
          }
          ug[I]=ugI; ua[I]=uaI;
          dug_dp[I]=dugp; dug_dz[I]=dugz; dua_dp[I]=duap; dua_dz[I]=duaz;
        }
        // (Kinetic R_diss dissolution sink removed -- dissolution is now handled
        // thermodynamically by the inline flash, so there is no sink Jacobian.)
        // Assemble per (i, j).  Residual:  R^c_i = m_1_t*N_i*dV - F_1.gradN_i*dV.
        // Flux Jacobian (chain rule through the flash):
        //   dR^c_i/d(z_j) = -[ (dF_1/dz)_val * N_j + (dF_1/dgrad_z) . gradN_j ] . gradN_i dV  -> (1,1)
        //   dR^c_i/d(p_j) = -[ (dF_1/dp)_val * N_j + (dF_1/dgrad_p) . gradN_j ] . gradN_i dV  -> (1,0)
        for (int i = 0; i < nDOF_test_element; i++) {
          const double test_i = u_test_ref.data()[k * nDOF_test_element + i];
          // value-block scalars: sum_I (dF_1/dval) * gradN_i[I]   (val = p, z)
          double Sval_p = 0.0, Sval_z = 0.0;
          for (int I = 0; I < nSpace; I++) {
            const double gNiI = u_grad_trial_qp[i * nSpace + I];
            Sval_p += (dAg_dp*ug[I] + Ag*dug_dp[I] + dAa_dp*ua[I] + Aa*dua_dp[I]) * gNiI;
            Sval_z += (dAg_dz*ug[I] + Ag*dug_dz[I] + dAa_dz*ua[I] + Aa*dua_dz[I]) * gNiI;
          }
          for (int j = 0; j < nDOF_trial_element; j++) {
            const double trial_j = u_trial_ref.data()[k * nDOF_trial_element + j];
            // gradient-block scalars: sum_{I,ii} (dF_1[I]/dgrad_var[J]) gradN_j[J] gradN_i[I]
            double Sgrad_p = 0.0, Sgrad_z = 0.0;
            for (int I = 0; I < nSpace; I++) {
              const double gNiI = u_grad_trial_qp[i * nSpace + I];
              for (int ii = a_rowptr.data()[I]; ii < a_rowptr.data()[I + 1]; ii++) {
                const int J = a_colind.data()[ii];
                const double Kii = KWs_eN[ii];
                const double Mob_g = KNr_a*Kii/mu_n, Mob_a = KWr_a*Kii;
                const double gNjJ = u_grad_trial_qp[j * nSpace + J];
                // d ug[I]/d grad_p[J] = -Mob_g*(1 - pcp*dSg_dp); d ua[I]/d grad_p[J] = -Mob_a
                const double dFdgp = Ag*(-Mob_g*(1.0 - pcp_a*fs_j.dS_g_dp)) + Aa*(-Mob_a);
                // d ug[I]/d grad_z[J] = -Mob_g*(-pcp*dSg_dz); d ua[I]/d grad_z[J] = 0
                const double dFdgz = Ag*(-Mob_g*(-pcp_a*fs_j.dS_g_dz));
                Sgrad_p += dFdgp * gNjJ * gNiI;
                Sgrad_z += dFdgz * gNjJ * gNiI;
              }
            }
            // (1,1): accumulation (implicit BDF) + CO2 flux d/dz.
            elementJacobian_n_n[i][j] += (dm_n_du_n * test_i * trial_j * dV) / dt_n;
            elementJacobian_n_n[i][j] -= (Sval_z * trial_j + Sgrad_z) * dV;
            // (1,0): accumulation d/dp + CO2 flux d/dp.
            elementJacobian_n_w[i][j] += (dm_n_du_w * test_i * trial_j * dV) / dt_n;
            elementJacobian_n_w[i][j] -= (Sval_p * trial_j + Sgrad_p) * dV;
          }
        }
      }
      for (int i = 0; i < nDOF_test_element; i++) {
        const int eN_i = eN * nDOF_test_element + i;
        for (int j = 0; j < nDOF_trial_element; j++) {
          const int eN_i_j = eN_i * nDOF_trial_element + j;
          globalJacobian.data()[csrRowIndeces_n_n.data()[eN_i] + csrColumnOffsets_n_n.data()[eN_i_j]]
              += elementJacobian_n_n[i][j];
          globalJacobian.data()[csrRowIndeces_n_w.data()[eN_i] + csrColumnOffsets_n_w.data()[eN_i_j]]
              += elementJacobian_n_w[i][j];
        }
      }
    }

    // ============================================================================
    // Comp-1 (CO2 / z) exterior boundary Jacobian -- STAB=0 path. COMPOSITIONAL.
    // Matching Jacobian for the compositional F_1.n boundary residual term in
    // calculateResidual. Mirrors the FD-verified interior comp-1 flux Jacobian
    // (~line 1942) chain-ruled through the analytic flash, with the interior
    // gradN_i replaced by the boundary normal n_I:
    //   (1,1): d(F_1.n)/dz  (value-block * trial_j + grad-block . gradN_j)
    //          + Nitsche penalty * trial_j   (Dirichlet faces only)
    //   (1,0): d(F_1.n)/dp  (value-block * trial_j + grad-block . gradN_j)
    // Gated on isDir_n: no-flow faces contribute nothing (matches the residual).
    // ============================================================================
    for (int ebNE = 0; ebNE < nExteriorElementBoundaries_global; ebNE++) {
      const int ebN = exteriorElementBoundariesArray.data()[ebNE];
      const int eN  = elementBoundaryElementsArray.data()[ebN * 2 + 0];
      const int ebN_local = elementBoundaryLocalElementBoundariesArray.data()[ebN * 2 + 0];
      const int eN_nDOF_trial_element = eN * nDOF_trial_element;
      const int    mat_eN    = elementMaterialTypes.data()[eN];
      const double phi_eN    = thetaR.data()[mat_eN] + thetaSR.data()[mat_eN];
      const double alpha_eN  = alpha.data()[mat_eN];
      const double krn_end_eN = krn_end.data()[mat_eN];
      const double n_vg_eN   = n.data()[mat_eN];
      const double *KWs_eN   = &KWs.data()[mat_eN * nnz];
      const double S_wr_loc      = thetaR.data()[mat_eN] / phi_eN;
      const double one_m_Sr_loc  = 1.0 - S_wr_loc;
      const double Se_trap_L2110 = 1.0 - S_gr.data()[mat_eN] / one_m_Sr_loc;  // gas-only residual trapping

      double elementJacobian_n_n_eb[nDOF_test_element][nDOF_trial_element];
      double elementJacobian_n_w_eb[nDOF_test_element][nDOF_trial_element];
      for (int i = 0; i < nDOF_test_element; i++)
        for (int j = 0; j < nDOF_trial_element; j++) {
          elementJacobian_n_n_eb[i][j] = 0.0;
          elementJacobian_n_w_eb[i][j] = 0.0;
        }

      for (int kb = 0; kb < nQuadraturePoints_elementBoundary; kb++) {
        const int ebNE_kb            = ebNE * nQuadraturePoints_elementBoundary + kb;
        const int ebN_local_kb       = ebN_local * nQuadraturePoints_elementBoundary + kb;
        const int ebN_local_kb_nSpace = ebN_local_kb * nSpace;

        double jac_ext[nSpace * nSpace], jacDet_ext, jacInv_ext[nSpace * nSpace];
        double boundaryJac_b[nSpace * (nSpace - 1)];
        double metricTensor_b[(nSpace - 1) * (nSpace - 1)];
        double metricTensorDetSqrt_b, dS_eb, normal_b[3];
        double xt_b, yt_b, zt_b, integralScaling_b;
        double x_eb, y_eb, z_eb;
        ck.calculateMapping_elementBoundary(eN, ebN_local, kb, ebN_local_kb,
            mesh_dof.data(), mesh_l2g.data(), mesh_trial_trace_ref.data(),
            mesh_grad_trial_trace_ref.data(), boundaryJac_ref.data(),
            jac_ext, jacDet_ext, jacInv_ext, boundaryJac_b, metricTensor_b,
            metricTensorDetSqrt_b, normal_ref.data(), normal_b,
            x_eb, y_eb, z_eb);
        ck.calculateMappingVelocity_elementBoundary(eN, ebN_local, kb, ebN_local_kb,
            mesh_velocity_dof.data(), mesh_l2g.data(), mesh_trial_trace_ref.data(),
            xt_b, yt_b, zt_b, normal_b, boundaryJac_b, metricTensor_b,
            integralScaling_b);
        dS_eb = ((1.0 - MOVING_DOMAIN) * metricTensorDetSqrt_b
                + MOVING_DOMAIN * integralScaling_b) * dS_ref.data()[kb];

        double u_grad_trial_trace_b[nDOF_trial_element * nSpace];
        ck.gradTrialFromRef(
            &u_grad_trial_trace_ref.data()[ebN_local_kb_nSpace * nDOF_trial_element],
            jacInv_ext, u_grad_trial_trace_b);
        double u_w_ext_b = 0.0, u_n_ext_b = 0.0;
        double grad_u_w_ext_b[nSpace], grad_u_n_ext_b[nSpace];
        ck.valFromDOF(u_dof.data(), &u_l2g.data()[eN_nDOF_trial_element],
                      &u_trial_trace_ref.data()[ebN_local_kb * nDOF_test_element], u_w_ext_b);
        ck.valFromDOF(u_dof_n.data(), &u_l2g.data()[eN_nDOF_trial_element],
                      &u_trial_trace_ref.data()[ebN_local_kb * nDOF_test_element], u_n_ext_b);
        ck.gradFromDOF(u_dof.data(), &u_l2g.data()[eN_nDOF_trial_element],
                       u_grad_trial_trace_b, grad_u_w_ext_b);
        ck.gradFromDOF(u_dof_n.data(), &u_l2g.data()[eN_nDOF_trial_element],
                       u_grad_trial_trace_b, grad_u_n_ext_b);

        const int    isDir_n  = isDOFBoundary_n.data()[ebNE_kb];
        const double penalty  = ebqe_penalty_ext.data()[ebNE_kb];

        // ====================================================================
        // P3c boundary (STAB=0): compositional comp-1 (CO2) flux Jacobian.
        // Matches the F_1.n residual boundary; mirrors the FD-verified interior
        // comp-1 Jacobian (~line 1942) chain-ruled through the analytic flash,
        // with the interior gradN_i replaced by the boundary normal n_I:
        //   (1,1) d(F_1.n)/dz  + Nitsche penalty * trial_j (Dirichlet faces),
        //   (1,0) d(F_1.n)/dp.
        // ====================================================================
        const double p_clb = fmax(u_w_ext_b, 1.0e2);
        ::m_comp_co2::flash::FlashState fsb =
            ::m_comp_co2::flash::flashPZ(p_clb, u_n_ext_b, T_C_member, m_NaCl_member, ::m_comp_co2::flash::EPS_Z, immiscible_member);
        const double Sa_b = 1.0 - fsb.S_g;
        const double Se_raw_b = (Sa_b - S_wr_loc) / one_m_Sr_loc;
        double Se_b, dSe_dp_b, dSe_dz_b;
        if (Se_raw_b <= 0.0)      { Se_b = 0.0; dSe_dp_b = 0.0; dSe_dz_b = 0.0; }
        else if (Se_raw_b >= 1.0) { Se_b = 1.0; dSe_dp_b = 0.0; dSe_dz_b = 0.0; }
        else { Se_b = Se_raw_b; dSe_dp_b = -fsb.dS_g_dp/one_m_Sr_loc; dSe_dz_b = -fsb.dS_g_dz/one_m_Sr_loc; }
        double KWr_b=0,DKWr_b=0,thW_b=0,DthW_b=0,KNr_b=0,DKNr_b=0,pc_b=0,dpc_dSe_b=0,d2pc_b=0;
        if (PSK_TYPE_member == 1) {
          proteus::m_comp_co2::psk::bc_wetting_from_Se(Se_b, alpha_eN, n_vg_eN, thetaR.data()[mat_eN], thetaSR.data()[mat_eN], thW_b, DthW_b, KWr_b, DKWr_b);
          proteus::m_comp_co2::psk::bc_kr_nonwetting_from_Se(Se_b, alpha_eN, n_vg_eN, KNr_b, DKNr_b, Se_trap_L2110);
          proteus::m_comp_co2::psk::bc_pc_from_Se(Se_b, alpha_eN, n_vg_eN, bc_Se_min.data()[mat_eN], pc_b, dpc_dSe_b, d2pc_b);
        } else {
          proteus::m_comp_co2::psk::vgm_wetting_from_Se(Se_b, alpha_eN, n_vg_eN, thetaR.data()[mat_eN], thetaSR.data()[mat_eN], thW_b, DthW_b, KWr_b, DKWr_b);
          proteus::m_comp_co2::psk::vgm_kr_nonwetting_from_Se(Se_b, alpha_eN, n_vg_eN, KNr_b, DKNr_b, Se_trap_L2110);
          proteus::m_comp_co2::psk::vgm_pc_from_Se(Se_b, alpha_eN, n_vg_eN, pc_b, dpc_dSe_b, d2pc_b);
        }
        KNr_b *= krn_end_eN;  DKNr_b *= krn_end_eN;
        const double pcp_b     = dpc_dSe_b / one_m_Sr_loc;
        const double dpcp_dp_b = (d2pc_b / one_m_Sr_loc) * dSe_dp_b;
        const double dpcp_dz_b = (d2pc_b / one_m_Sr_loc) * dSe_dz_b;
        const double dMm_b = ::m_comp_co2::eos::M_CO2_KG - ::m_comp_co2::eos::M_H2O_KG;
        const double Mbar_g_b = fsb.Y*::m_comp_co2::eos::M_CO2_KG + (1.0-fsb.Y)*::m_comp_co2::eos::M_H2O_KG;
        const double Mbar_a_b = fsb.X*::m_comp_co2::eos::M_CO2_KG + (1.0-fsb.X)*::m_comp_co2::eos::M_H2O_KG;
        const double rho_g_mass_b = fsb.rho_g*Mbar_g_b, rho_a_mass_b = fsb.rho_a*Mbar_a_b;
        const double drgm_dp_b = fsb.drho_g_dp*Mbar_g_b + fsb.rho_g*fsb.dY_dp*dMm_b;
        const double drgm_dz_b =                          fsb.rho_g*fsb.dY_dz*dMm_b;
        const double dram_dp_b = fsb.drho_a_dp*Mbar_a_b + fsb.rho_a*fsb.dX_dp*dMm_b;
        const double dram_dz_b = fsb.drho_a_dz*Mbar_a_b + fsb.rho_a*fsb.dX_dz*dMm_b;
        const double Ag = fsb.rho_g*fsb.Y, Aa = fsb.rho_a*fsb.X;
        const double dAg_dp = fsb.drho_g_dp*fsb.Y + fsb.rho_g*fsb.dY_dp;
        const double dAg_dz =                        fsb.rho_g*fsb.dY_dz;
        const double dAa_dp = fsb.drho_a_dp*fsb.X + fsb.rho_a*fsb.dX_dp;
        const double dAa_dz = fsb.drho_a_dz*fsb.X + fsb.rho_a*fsb.dX_dz;
        double ug_b[nSpace], ua_b[nSpace];
        double dug_dp_b[nSpace], dug_dz_b[nSpace], dua_dp_b[nSpace], dua_dz_b[nSpace];
        for (int I = 0; I < nSpace; I++) {
          double ugI=0.0, uaI=0.0, dugp=0.0, dugz=0.0, duap=0.0, duaz=0.0;
          for (int ii = a_rowptr.data()[I]; ii < a_rowptr.data()[I + 1]; ii++) {
            const int J = a_colind.data()[ii];
            const double Kii = KWs_eN[ii];
            const double Mob_g = KNr_b*Kii/mu_n, Mob_a = KWr_b*Kii;
            const double dMobg_dp = (DKNr_b*Kii/mu_n)*dSe_dp_b, dMobg_dz = (DKNr_b*Kii/mu_n)*dSe_dz_b;
            const double dMoba_dp = (DKWr_b*Kii)*dSe_dp_b,        dMoba_dz = (DKWr_b*Kii)*dSe_dz_b;
            const double gJ = gravity.data()[J];
            const double gradSa = -(fsb.dS_g_dp*grad_u_w_ext_b[J] + fsb.dS_g_dz*grad_u_n_ext_b[J]);
            const double gp_a = grad_u_w_ext_b[J] - rho_a_mass_b*gJ;
            const double gp_g = grad_u_w_ext_b[J] + pcp_b*gradSa - rho_g_mass_b*gJ;
            ugI -= Mob_g*gp_g;  uaI -= Mob_a*gp_a;
            const double dgradSa_dp = -(fsb.d2S_g_dp2 *grad_u_w_ext_b[J] + fsb.d2S_g_dpdz*grad_u_n_ext_b[J]);
            const double dgradSa_dz = -(fsb.d2S_g_dpdz*grad_u_w_ext_b[J] + fsb.d2S_g_dz2 *grad_u_n_ext_b[J]);
            const double dgpg_dp = dpcp_dp_b*gradSa + pcp_b*dgradSa_dp - drgm_dp_b*gJ;
            const double dgpg_dz = dpcp_dz_b*gradSa + pcp_b*dgradSa_dz - drgm_dz_b*gJ;
            dugp -= dMobg_dp*gp_g + Mob_g*dgpg_dp;
            dugz -= dMobg_dz*gp_g + Mob_g*dgpg_dz;
            duap -= dMoba_dp*gp_a + Mob_a*(-dram_dp_b*gJ);
            duaz -= dMoba_dz*gp_a + Mob_a*(-dram_dz_b*gJ);
          }
          ug_b[I]=ugI; ua_b[I]=uaI;
          dug_dp_b[I]=dugp; dug_dz_b[I]=dugz; dua_dp_b[I]=duap; dua_dz_b[I]=duaz;
        }
        // value-block scalars dotted with the normal (interior gradN_i -> n_I).
        double Sval_p_b = 0.0, Sval_z_b = 0.0;
        for (int I = 0; I < nSpace; I++) {
          Sval_p_b += (dAg_dp*ug_b[I] + Ag*dug_dp_b[I] + dAa_dp*ua_b[I] + Aa*dua_dp_b[I]) * normal_b[I];
          Sval_z_b += (dAg_dz*ug_b[I] + Ag*dug_dz_b[I] + dAa_dz*ua_b[I] + Aa*dua_dz_b[I]) * normal_b[I];
        }
        // IIPG penalty scaled by the comp-1 diffusion magnitude a_n (frozen
        // coefficient) -- MUST match the residual loop in calculateResidual.
        double Kw_rep = 0.0;
        for (int ii = 0; ii < nnz; ii++) Kw_rep = fmax(Kw_rep, fabs(KWs_eN[ii]));
        const double a_n_scale = (Ag*KNr_b/mu_n + Aa*KWr_b) * Kw_rep;
        const double pen_eff = penalty * a_n_scale;

        for (int i = 0; i < nDOF_test_element; i++) {
          const double test_i_dS = u_test_trace_ref.data()[
              ebN_local_kb * nDOF_test_element + i] * dS_eb;
          for (int j = 0; j < nDOF_trial_element; j++) {
            const double trial_j_b = u_trial_trace_ref.data()[
                ebN_local_kb * nDOF_test_element + j];
            double Sgrad_p_b = 0.0, Sgrad_z_b = 0.0;
            for (int I = 0; I < nSpace; I++) {
              for (int ii = a_rowptr.data()[I]; ii < a_rowptr.data()[I + 1]; ii++) {
                const int J = a_colind.data()[ii];
                const double Kii = KWs_eN[ii];
                const double Mob_g = KNr_b*Kii/mu_n, Mob_a = KWr_b*Kii;
                const double gNjJ = u_grad_trial_trace_b[j * nSpace + J];
                const double dFdgp = Ag*(-Mob_g*(1.0 - pcp_b*fsb.dS_g_dp)) + Aa*(-Mob_a);
                const double dFdgz = Ag*(-Mob_g*(-pcp_b*fsb.dS_g_dz));
                Sgrad_p_b += dFdgp * gNjJ * normal_b[I];
                Sgrad_z_b += dFdgz * gNjJ * normal_b[I];
              }
            }
            double jac_nn = Sval_z_b * trial_j_b + Sgrad_z_b;
            double jac_nw = Sval_p_b * trial_j_b + Sgrad_p_b;
            if (isDir_n) {
              jac_nn += pen_eff * trial_j_b;
            } else {
              jac_nn = 0.0;
              jac_nw = 0.0;
            }
            elementJacobian_n_n_eb[i][j] += jac_nn * test_i_dS;
            elementJacobian_n_w_eb[i][j] += jac_nw * test_i_dS;
          }
        }
      } // kb

      for (int i = 0; i < nDOF_test_element; i++) {
        const int eN_i = eN * nDOF_test_element + i;
        for (int j = 0; j < nDOF_trial_element; j++) {
          const int ebN_i_j = ebN * 4 * nDOF_test_X_trial_element
                            + i * nDOF_trial_element + j;
          globalJacobian.data()[csrRowIndeces_n_n.data()[eN_i]
              + csrColumnOffsets_eb_n_n.data()[ebN_i_j]] += elementJacobian_n_n_eb[i][j];
          globalJacobian.data()[csrRowIndeces_n_w.data()[eN_i]
              + csrColumnOffsets_eb_n_w.data()[ebN_i_j]] += elementJacobian_n_w_eb[i][j];
        }
      }
    } // ebNE
  } //computeJacobian

 

  // ============================================================================
  // FCTStep -- Zalesak FCT limiter, component- and pass-dispatched (ONE function).
  //
  //   if component == 0 (wetting, p_w) / else component == 1 (non-wetting, S_n);
  //   within each, if pass == 1 / else pass == 2.
  //
  // Split into two passes for MPI parallel-correctness: the limiter is
  // mass-conservative only if both ranks sharing an edge compute the SAME
  // L_ij, which needs Rpos/Rneg consistent for BOTH endpoints. With 1-layer
  // overlap a ghost DOF's stencil is incomplete, so pass 1 computes Rpos/Rneg
  // locally, Python ghost-scatters them, then pass 2 applies the limiter.
  // In serial the scatter is a no-op.
  //
  //   pass 1: inputs -> FluxCorrectionMatrix, Rpos, Rneg
  //   [Python: scatter_forward Rpos, Rneg]
  //   pass 2: Rpos, Rneg, FluxCorrectionMatrix -> limited_solution, fluxCorrection
  //
  // comp-0 indexes MC / dt_times_fH_minus_fL by the FULL CSR offset (via
  // full_offset_from_compact); comp-1 indexes MC_n / dt_times_fH_minus_fL_n by
  // the COMPACT comp-1 CSR position directly.
  // ============================================================================
  void FCTStep(arguments_dict &args)
  {
    const int component = args.scalar<int>("component");
    const int pass      = args.scalar<int>("pass");

    if (component == 0) {
      // ======================= wetting (p_w) =======================
      if (pass == 1) {
        int                  numDOFs                   = args.scalar<int>("numDOFs");
        double               dt                        = args.scalar<double>("dt");
        proteus::pyarray<double> &ML                        = args.array<double>("ML");
        proteus::pyarray<double> &mn                        = args.array<double>("mn");
        proteus::pyarray<double> &mLow                      = args.array<double>("mLow");
        proteus::pyarray<double> &mDotLow                   = args.array<double>("mDotLow");
        proteus::pyarray<int>    &csrRowIndeces_DofLoops    = args.array<int>("csrRowIndeces_DofLoops");
        proteus::pyarray<int>    &csrColumnOffsets_DofLoops = args.array<int>("csrColumnOffsets_DofLoops");
        proteus::pyarray<int>    &csrRowIndeces_Full        = args.array<int>("csrRowIndeces_Full");
        proteus::pyarray<int>    &csrColumnOffsets_Full     = args.array<int>("csrColumnOffsets_Full");
        proteus::pyarray<double> &MC                        = args.array<double>("MC");
        proteus::pyarray<double> &dt_times_fH_minus_fL      = args.array<double>("dt_times_fH_minus_fL");
        proteus::pyarray<double> &min_m_bc                  = args.array<double>("min_m_bc");
        proteus::pyarray<double> &max_m_bc                  = args.array<double>("max_m_bc");
        proteus::pyarray<double> &FluxCorrectionMatrix      = args.array<double>("FluxCorrectionMatrix");
        proteus::pyarray<double> &Rpos                      = args.array<double>("Rpos");
        proteus::pyarray<double> &Rneg                      = args.array<double>("Rneg");
        int                  LUMPED_MASS_MATRIX        = args.scalar<int>("LUMPED_MASS_MATRIX");
        int                  MONOLITHIC                = args.scalar<int>("MONOLITHIC");
        const int            offset_u                  = args.scalar<int>("offset_u");
        const int            stride_u                  = args.scalar<int>("stride_u");
        auto full_offset_from_compact = [&](int i_compact, int j_compact) -> int {
          const int full_i = offset_u + stride_u * i_compact;
          const int full_j = offset_u + stride_u * j_compact;
          for (int offset = csrRowIndeces_Full.at(full_i); offset < csrRowIndeces_Full.at(full_i + 1); ++offset)
            if (csrColumnOffsets_Full.at(offset) == full_j) return offset;
          return -1;
        };
        int ij = 0;
        for (int i = 0; i < numDOFs; i++) {
          double mini = min_m_bc.at(i);
          double maxi = max_m_bc.at(i);
          double Pposi = 0.0, Pnegi = 0.0;
          for (int offset = csrRowIndeces_DofLoops.at(i); offset < csrRowIndeces_DofLoops.at(i + 1); offset++) {
            int j = csrColumnOffsets_DofLoops.at(offset);
            const int full_offset = full_offset_from_compact(i, j);
            assert(full_offset >= 0);
            if (GLOBAL_FCT == 0) {
              if (MONOLITHIC == 0) { mini = fmin(mini, mLow.at(j)); maxi = fmax(maxi, mLow.at(j)); }
              else                 { mini = fmin(mini, mn.at(j));   maxi = fmax(maxi, mn.at(j));   }
            }
            if (MONOLITHIC == 0) {
              FluxCorrectionMatrix.at(ij) = (LUMPED_MASS_MATRIX == 1 ? 0. : 1.) * dt
                  * MC.at(full_offset) * (mDotLow.at(i) - mDotLow.at(j))
                  + dt_times_fH_minus_fL.at(full_offset);
            } else {
              FluxCorrectionMatrix.at(ij) = dt_times_fH_minus_fL.at(full_offset);
            }
            Pposi += FluxCorrectionMatrix.at(ij) * ((FluxCorrectionMatrix.at(ij) > 0) ? 1. : 0.);
            Pnegi += FluxCorrectionMatrix.at(ij) * ((FluxCorrectionMatrix.at(ij) < 0) ? 1. : 0.);
            ij += 1;
          }
          double Qposi, Qnegi;
          if (MONOLITHIC == 0) {
            Qposi = ML.at(i) * (maxi - mLow.at(i));
            Qnegi = ML.at(i) * (mini - mLow.at(i));
          } else {
            const double gamma = 10.0 * ML.at(i);
            Qposi = fmin(0.5 * ML.at(i) * (1.0 - mn.at(i)), gamma * (maxi - mn.at(i)));
            Qnegi = fmax(0.5 * ML.at(i) * (0.0 - mn.at(i)), gamma * (mini - mn.at(i)));
          }
          Rpos.at(i) = ((Pposi == 0.0) ? 1.0 : fmin(1.0, Qposi / Pposi));
          Rneg.at(i) = ((Pnegi == 0.0) ? 1.0 : fmin(1.0, Qnegi / Pnegi));
        }
      } else {
        // comp-0 pass 2
        proteus::pyarray<double> &bc_mask                   = args.array<double>("bc_mask");
        int                  numDOFs                   = args.scalar<int>("numDOFs");
        double               dt                        = args.scalar<double>("dt");
        proteus::pyarray<double> &ML                        = args.array<double>("ML");
        proteus::pyarray<double> &mn                        = args.array<double>("mn");
        proteus::pyarray<double> &mLow                      = args.array<double>("mLow");
        proteus::pyarray<int>    &csrRowIndeces_DofLoops    = args.array<int>("csrRowIndeces_DofLoops");
        proteus::pyarray<int>    &csrColumnOffsets_DofLoops = args.array<int>("csrColumnOffsets_DofLoops");
        proteus::pyarray<int>    &csrRowIndeces_Full        = args.array<int>("csrRowIndeces_Full");
        proteus::pyarray<int>    &csrColumnOffsets_Full     = args.array<int>("csrColumnOffsets_Full");
        proteus::pyarray<double> &MC                        = args.array<double>("MC");
        proteus::pyarray<double> &FluxCorrectionMatrix      = args.array<double>("FluxCorrectionMatrix");
        proteus::pyarray<double> &Rpos                      = args.array<double>("Rpos");
        proteus::pyarray<double> &Rneg                      = args.array<double>("Rneg");
        proteus::pyarray<double> &fluxCorrection            = args.array<double>("fluxCorrection");
        proteus::pyarray<double> &limited_solution          = args.array<double>("limited_solution");
        int                  LUMPED_MASS_MATRIX        = args.scalar<int>("LUMPED_MASS_MATRIX");
        int                  MONOLITHIC                = args.scalar<int>("MONOLITHIC");
        const int            offset_u                  = args.scalar<int>("offset_u");
        const int            stride_u                  = args.scalar<int>("stride_u");
        auto full_offset_from_compact = [&](int i_compact, int j_compact) -> int {
          const int full_i = offset_u + stride_u * i_compact;
          const int full_j = offset_u + stride_u * j_compact;
          for (int offset = csrRowIndeces_Full.at(full_i); offset < csrRowIndeces_Full.at(full_i + 1); ++offset)
            if (csrColumnOffsets_Full.at(offset) == full_j) return offset;
          return -1;
        };
        int ij = 0;
        for (int i = 0; i < numDOFs; i++) {
          double ith_Limiter_times_FluxCorrectionMatrix = 0.0;
          const double beta_ij = 1.0;
          const double mDot_i = (mLow.at(i) - mn.at(i)) / dt;
          for (int offset = csrRowIndeces_DofLoops.at(i); offset < csrRowIndeces_DofLoops.at(i + 1); offset++) {
            int j = csrColumnOffsets_DofLoops.at(offset);
            const int full_offset = full_offset_from_compact(i, j);
            assert(full_offset >= 0);
            const double alpha_fA = ((FluxCorrectionMatrix.at(ij) > 0.0)
                  ? fmin(Rpos.at(i), Rneg.at(j)) : fmin(Rneg.at(i), Rpos.at(j)))
                * FluxCorrectionMatrix.at(ij);
            if (MONOLITHIC == 0) {
              ith_Limiter_times_FluxCorrectionMatrix += alpha_fA;
            } else {
              const double mDot_j = (mLow.at(j) - mn.at(j)) / dt;
              const double alpha_dot = fmin(1.0, beta_ij * fabs(alpha_fA) / MC.at(full_offset)
                                                / fmax(1.0e-8, fabs(mDot_i - mDot_j)));
              ith_Limiter_times_FluxCorrectionMatrix += alpha_fA
                  + (LUMPED_MASS_MATRIX == 1 ? 0. : 1.) * dt * alpha_dot
                    * MC.at(full_offset) * (mDot_i - mDot_j);
            }
            ij += 1;
          }
          fluxCorrection.at(i)   = -ith_Limiter_times_FluxCorrectionMatrix * bc_mask.at(i) / dt;
          limited_solution.at(i) = mLow.at(i)
              + 1.0 / ML.at(i) * ith_Limiter_times_FluxCorrectionMatrix * bc_mask.at(i);
        }
      }
    } else {
      // ===================== non-wetting (S_n) =====================
      if (pass == 1) {
        int                  numDOFs_n                   = args.scalar<int>("numDOFs_n");
        double               dt                          = args.scalar<double>("dt");
        proteus::pyarray<double> &ML_n                        = args.array<double>("ML_n");
        proteus::pyarray<double> &MC_n                        = args.array<double>("MC_n");
        proteus::pyarray<double> &mLow_n                      = args.array<double>("mLow_n");
        proteus::pyarray<double> &mDotLow_n                   = args.array<double>("mDotLow_n");
        proteus::pyarray<double> &dt_times_fH_minus_fL_n      = args.array<double>("dt_times_fH_minus_fL_n");
        proteus::pyarray<double> &min_m_bc_n                  = args.array<double>("min_m_bc_n");
        proteus::pyarray<double> &max_m_bc_n                  = args.array<double>("max_m_bc_n");
        proteus::pyarray<double> &FluxCorrectionMatrix_n      = args.array<double>("FluxCorrectionMatrix_n");
        proteus::pyarray<double> &Rpos_n                      = args.array<double>("Rpos_n");
        proteus::pyarray<double> &Rneg_n                      = args.array<double>("Rneg_n");
        proteus::pyarray<int>    &csrRowIndeces_n_DofLoops    = args.array<int>("csrRowIndeces_n_DofLoops");
        proteus::pyarray<int>    &csrColumnOffsets_n_DofLoops = args.array<int>("csrColumnOffsets_n_DofLoops");
        int                  LUMPED_MASS_MATRIX          = args.scalar<int>("LUMPED_MASS_MATRIX");
        int ij = 0;
        for (int i = 0; i < numDOFs_n; i++) {
          double mini = min_m_bc_n.at(i);
          double maxi = max_m_bc_n.at(i);
          double Pposi = 0.0, Pnegi = 0.0;
          for (int offset = csrRowIndeces_n_DofLoops.at(i);
               offset < csrRowIndeces_n_DofLoops.at(i + 1); offset++) {
            const int j = csrColumnOffsets_n_DofLoops.at(offset);
            if (GLOBAL_FCT == 0) {
              mini = std::fmin(mini, mLow_n.at(j));
              maxi = std::fmax(maxi, mLow_n.at(j));
            }
            FluxCorrectionMatrix_n.at(ij) =
                (LUMPED_MASS_MATRIX == 1 ? 0.0 : 1.0)
                  * dt * MC_n.at(offset) * (mDotLow_n.at(i) - mDotLow_n.at(j))
                + dt_times_fH_minus_fL_n.at(offset);
            Pposi += (FluxCorrectionMatrix_n.at(ij) > 0.0) ? FluxCorrectionMatrix_n.at(ij) : 0.0;
            Pnegi += (FluxCorrectionMatrix_n.at(ij) < 0.0) ? FluxCorrectionMatrix_n.at(ij) : 0.0;
            ij += 1;
          }
          const double Qposi = ML_n.at(i) * (maxi - mLow_n.at(i));
          const double Qnegi = ML_n.at(i) * (mini - mLow_n.at(i));
          Rpos_n.at(i) = (Pposi == 0.0) ? 1.0 : std::fmin(1.0, Qposi / Pposi);
          Rneg_n.at(i) = (Pnegi == 0.0) ? 1.0 : std::fmin(1.0, Qnegi / Pnegi);
        }
      } else {
        // comp-1 pass 2
        int                  numDOFs_n                   = args.scalar<int>("numDOFs_n");
        double               dt                          = args.scalar<double>("dt");
        proteus::pyarray<double> &ML_n                        = args.array<double>("ML_n");
        proteus::pyarray<double> &mLow_n                      = args.array<double>("mLow_n");
        proteus::pyarray<double> &FluxCorrectionMatrix_n      = args.array<double>("FluxCorrectionMatrix_n");
        proteus::pyarray<double> &Rpos_n                      = args.array<double>("Rpos_n");
        proteus::pyarray<double> &Rneg_n                      = args.array<double>("Rneg_n");
        proteus::pyarray<double> &fluxCorrection_n            = args.array<double>("fluxCorrection_n");
        proteus::pyarray<double> &limited_solution_n          = args.array<double>("limited_solution_n");
        proteus::pyarray<double> &bc_mask_n                   = args.array<double>("bc_mask_n");
        proteus::pyarray<int>    &csrRowIndeces_n_DofLoops    = args.array<int>("csrRowIndeces_n_DofLoops");
        proteus::pyarray<int>    &csrColumnOffsets_n_DofLoops = args.array<int>("csrColumnOffsets_n_DofLoops");
        int ij = 0;
        for (int i = 0; i < numDOFs_n; i++) {
          double ith_Limited_FCM = 0.0;
          for (int offset = csrRowIndeces_n_DofLoops.at(i);
               offset < csrRowIndeces_n_DofLoops.at(i + 1); offset++) {
            const int j = csrColumnOffsets_n_DofLoops.at(offset);
            const double alpha_fA =
                ((FluxCorrectionMatrix_n.at(ij) > 0.0)
                     ? std::fmin(Rpos_n.at(i), Rneg_n.at(j))
                     : std::fmin(Rneg_n.at(i), Rpos_n.at(j)))
                * FluxCorrectionMatrix_n.at(ij);
            ith_Limited_FCM += alpha_fA;
            ij += 1;
          }
          fluxCorrection_n.at(i)   = -ith_Limited_FCM * bc_mask_n.at(i) / dt;
          limited_solution_n.at(i) = mLow_n.at(i)
                                  + (1.0 / ML_n.at(i)) * ith_Limited_FCM * bc_mask_n.at(i);
        }
      }
    }
  }


  void kth_FCT_step(arguments_dict &args)
  {
    int                  NNZ                       = args.scalar<int>("NNZ");     //number on non-zero entries on sparsity pattern
    int                  numDOFs                   = args.scalar<int>("numDOFs"); //number of DOFs
    int                  num_fct_iter              = args.scalar<int>("num_fct_iter");
    double               dt                        = args.scalar<double>("dt");
    proteus::pyarray<double> &lumped_mass_matrix        = args.array<double>("lumped_mass_matrix"); //lumped mass matrix (as vector)
    proteus::pyarray<double> &soln                      = args.array<double>("soln");               //DOFs of solution at time tn
    proteus::pyarray<double> &pn                        = args.array<double>("pn");                 //DOFs of solution at time tn
    proteus::pyarray<double> &solH                      = args.array<double>("solH");               //DOFs of high order solution at tnp1
    proteus::pyarray<double> &uLow                      = args.array<double>("uLow");
    proteus::pyarray<double> &uDotLow                   = args.array<double>("uDotLow");
    proteus::pyarray<double> &dLow                      = args.array<double>("dLow");
    proteus::pyarray<double> &solLim                    = args.array<double>("limited_solution");
    proteus::pyarray<double> &MC                        = args.array<double>("MC");
    proteus::pyarray<double> &ML                        = args.array<double>("ML");
    proteus::pyarray<double> &FluxMatrix                = args.array<double>("FluxMatrix");
    proteus::pyarray<double> &limitedFlux               = args.array<double>("limited_Flux");
    proteus::pyarray<int>    &csrRowIndeces_DofLoops    = args.array<int>("csrRowIndeces_DofLoops");    //csr row indeces
    proteus::pyarray<int>    &csrColumnOffsets_DofLoops = args.array<int>("csrColumnOffsets_DofLoops"); //csr column offsets
    proteus::pyarray<double> &MassMatrix                = args.array<double>("MassMatrix");             //mass matrix
    proteus::pyarray<double> &dt_times_fH_minus_fL      = args.array<double>("dt_times_fH_minus_fL");   //low minus high order dissipative matrices
    proteus::pyarray<double> &min_m_bc                  = args.array<double>("min_m_bc");               //min/max value at BCs. If DOF is not at boundary then min=1E10, max=-1E10
    proteus::pyarray<double> &max_m_bc                  = args.array<double>("max_m_bc");
    int                  LUMPED_MASS_MATRIX        = args.scalar<int>("LUMPED_MASS_MATRIX");
    int                  MONOLITHIC                = args.scalar<int>("MONOLITHIC");
    double               Rpos[numDOFs], Rneg[numDOFs];
    int                  ij = 0;

    //////////////////////////////////////////////////////
    // ********** COMPUTE LOW ORDER SOLUTION ********** //
    //////////////////////////////////////////////////////
    if (num_fct_iter == 0) { // No FCT for global bounds
      for (int i = 0; i < numDOFs; i++) { solLim.data()[i] = uLow.data()[i]; }
    } else // do FCT iterations (with global bounds) on low order solution
    {
      for (int iter = 0; iter < num_fct_iter; iter++) {
        ij = 0;
        for (int i = 0; i < numDOFs; i++) {
          double maxi = 1.0, Pposi = 0;
          for (int offset = csrRowIndeces_DofLoops.data()[i]; offset < csrRowIndeces_DofLoops.data()[i + 1]; offset++) {
            int j = csrColumnOffsets_DofLoops.data()[offset];
            // compute Flux correction
            double Fluxij = FluxMatrix.data()[ij] - limitedFlux.data()[ij];
            Pposi += Fluxij * ((Fluxij > 0) ? 1. : 0.);
            // update ij
            ij += 1;
          }
          // compute Q vectors
          double mi      = ML.data()[i];
          double solLimi = solLim.data()[i];
          double Qposi   = mi * (maxi - solLimi);
          // compute R vectors
          Rpos[i] = ((Pposi == 0) ? 1. : fmin(1.0, Qposi / Pposi));
        }
        ij = 0;
        for (int i = 0; i < numDOFs; i++) {
          double ith_Limiter_times_FluxCorrectionMatrix = 0.;
          double Rposi                                  = Rpos[i];
          for (int offset = csrRowIndeces_DofLoops.data()[i]; offset < csrRowIndeces_DofLoops.data()[i + 1]; offset++) {
            int j = csrColumnOffsets_DofLoops.data()[offset];
            // Flux Correction
            double Fluxij = FluxMatrix.data()[ij] - limitedFlux.data()[ij];
            // compute limiter
            double Lij = 1.0;
            Lij        = (Fluxij > 0 ? Rposi : Rpos[j]);
            // compute limited flux
            ith_Limiter_times_FluxCorrectionMatrix += Lij * Fluxij;

            // update limited flux
            limitedFlux.data()[ij] = Lij * Fluxij;

            //update FluxMatrix
            FluxMatrix.data()[ij] = Fluxij;

            //update ij
            ij += 1;
          }
          //update limited solution
          double mi = ML.data()[i];
        }
      }
    }

    // ***************************************** //
    // ********** HIGH ORDER SOLUTION ********** //
    // ***************************************** //
    ij = 0;
    for (int i = 0; i < numDOFs; i++) {
      double mini = soln.data()[i], maxi = soln.data()[i];
      double Pposi = 0, Pnegi = 0.;
      for (int offset = csrRowIndeces_DofLoops.data()[i]; offset < csrRowIndeces_DofLoops.data()[i + 1]; offset++) {
        int j = csrColumnOffsets_DofLoops.data()[offset];
        // compute local bounds //
        mini = fmin(mini, soln.data()[j]);
        maxi = fmax(maxi, soln.data()[j]);
        // compute P vectors //
        double fij = (MC.data()[ij] * (uDotLow.data()[i] - uDotLow.data()[j]) / dt + dLow.data()[ij] * (uLow.data()[i] - uLow.data()[j]));
        Pposi += fij * (fij > 0 ? 1. : 0.);
        Pnegi += fij * (fij < 0 ? 1. : 0.);
        //update ij
        ij += 1;
      }
      // compute Q vectors //
      double mi    = ML.data()[i];
      double Qposi = mi * (maxi - solLim.data()[i]);
      double Qnegi = mi * (mini - solLim.data()[i]);
      // compute R vectors //
      Rpos[i] = ((Pposi == 0) ? 1. : fmin(1.0, Qposi / Pposi));
      Rneg[i] = ((Pnegi == 0) ? 1. : fmin(1.0, Qnegi / Pnegi));
    }

    // COMPUTE LIMITERS //
    ij = 0;
    for (int i = 0; i < numDOFs; i++) {
      double ith_limited_flux_correction = 0;
      double Rposi                       = Rpos[i];
      double Rnegi                       = Rneg[i];
      for (int offset = csrRowIndeces_DofLoops.data()[i]; offset < csrRowIndeces_DofLoops.data()[i + 1]; offset++) {
        int j = csrColumnOffsets_DofLoops.data()[offset];
        // compute flux correction
        double fij = (MC.data()[ij] * (uDotLow.data()[i] - uDotLow.data()[j]) / dt + dLow.data()[ij] * (uLow.data()[i] - uLow.data()[j]));

        // compute limiters
        double Lij = 1.0;
        Lij        = fij > 0 ? fmin(Rposi, Rneg[j]) : fmin(Rnegi, Rpos[j]);
        // compute ith_limited_flux_correction
        ith_limited_flux_correction += Lij * fij;
        ij += 1;
      }
      double mi = ML.data()[i];
      solLim[i] += 1. / mi * ith_limited_flux_correction;
    }
  }

  void calculateResidual_entropy_viscosity(arguments_dict &args)
  {
    proteus::pyarray<double> &globalJacobian            = args.array<double>("globalJacobian");
    double               Theta                     = args.scalar<double>("Theta");
    double               Theta_h                   = args.scalar<double>("Theta_h");
    proteus::pyarray<double> &bc_mask                   = args.array<double>("bc_mask");
    double               dt                        = args.scalar<double>("dt");
    proteus::pyarray<double> &mesh_trial_ref            = args.array<double>("mesh_trial_ref");
    proteus::pyarray<double> &mesh_grad_trial_ref       = args.array<double>("mesh_grad_trial_ref");
    proteus::pyarray<double> &mesh_dof                  = args.array<double>("mesh_dof");
    proteus::pyarray<double> &mesh_velocity_dof         = args.array<double>("mesh_velocity_dof");
    double               MOVING_DOMAIN             = args.scalar<double>("MOVING_DOMAIN");
    proteus::pyarray<int>    &mesh_l2g                  = args.array<int>("mesh_l2g");
    proteus::pyarray<double> &dV_ref                    = args.array<double>("dV_ref");
    proteus::pyarray<double> &u_trial_ref               = args.array<double>("u_trial_ref");
    proteus::pyarray<double> &u_grad_trial_ref          = args.array<double>("u_grad_trial_ref");
    proteus::pyarray<double> &u_test_ref                = args.array<double>("u_test_ref");
    proteus::pyarray<double> &u_grad_test_ref           = args.array<double>("u_grad_test_ref");
    proteus::pyarray<double> &mesh_trial_trace_ref      = args.array<double>("mesh_trial_trace_ref");
    proteus::pyarray<double> &mesh_grad_trial_trace_ref = args.array<double>("mesh_grad_trial_trace_ref");
    proteus::pyarray<double> &dS_ref                    = args.array<double>("dS_ref");
    proteus::pyarray<double> &u_trial_trace_ref         = args.array<double>("u_trial_trace_ref");

    proteus::pyarray<double> &u_grad_trial_trace_ref                     = args.array<double>("u_grad_trial_trace_ref");
    proteus::pyarray<double> &u_test_trace_ref                           = args.array<double>("u_test_trace_ref");
    proteus::pyarray<double> &u_grad_test_trace_ref                      = args.array<double>("u_grad_test_trace_ref");
    proteus::pyarray<double> &normal_ref                                 = args.array<double>("normal_ref");
    proteus::pyarray<double> &boundaryJac_ref                            = args.array<double>("boundaryJac_ref");
    int                  nElements_global                           = args.scalar<int>("nElements_global");
    proteus::pyarray<double> &ebqe_penalty_ext                           = args.array<double>("ebqe_penalty_ext");
    proteus::pyarray<int>    &elementMaterialTypes                       = args.array<int>("elementMaterialTypes");
    proteus::pyarray<int>    &isSeepageFace                              = args.array<int>("isSeepageFace");
    proteus::pyarray<int>    &a_rowptr                                   = args.array<int>("a_rowptr");
    proteus::pyarray<int>    &a_colind                                   = args.array<int>("a_colind");
    double               rho                                        = args.scalar<double>("rho");
    double               beta                                       = args.scalar<double>("beta");
    //////////////////////////////Density Coupling ///////////////////////////////
    proteus::pyarray<double> &q_rho                                     = args.array<double>("q_rho");
    proteus::pyarray<double> &ebqe_rho                                  = args.array<double>("ebqe_rho");
    ////////////////////////////////////////////////////////////////////////////


    proteus::pyarray<double> &gravity                                    = args.array<double>("gravity");
    proteus::pyarray<double> &alpha                                      = args.array<double>("alpha");
    proteus::pyarray<double> &n                                          = args.array<double>("n");
    proteus::pyarray<double> &thetaR                                     = args.array<double>("thetaR");
    proteus::pyarray<double> &thetaSR                                    = args.array<double>("thetaSR");
    proteus::pyarray<double> &KWs                                        = args.array<double>("KWs");
    proteus::pyarray<double> &krn_end                                    = args.array<double>("krn_end");
    proteus::pyarray<double> &S_gr                                       = args.array<double>("S_gr");
    proteus::pyarray<double> &bc_Se_min                                  = args.array<double>("bc_Se_min");
    double               mu_n                                       = args.scalar<double>("mu_n");
    double               useMetrics                                 = args.scalar<double>("useMetrics");
    double               alphaBDF                                   = args.scalar<double>("alphaBDF");
    int                  lag_shockCapturing                         = args.scalar<int>("lag_shockCapturing");
    double               shockCapturingDiffusion                    = args.scalar<double>("shockCapturingDiffusion");
    double               sc_uref                                    = args.scalar<double>("sc_uref");
    double               sc_alpha                                   = args.scalar<double>("sc_alpha");
    proteus::pyarray<int>    &u_l2g                                      = args.array<int>("u_l2g");
    // Component-1 (z) element-to-DOF map (comp-1's own femSpace.dofMap.l2g).  On
    // the shared P1 mesh it equals u_l2g element-wise.
    proteus::pyarray<int>    &u_l2g_n                                    = args.array<int>("u_l2g_n");
    proteus::pyarray<int>    &r_l2g                                     = args.array<int>("r_l2g");
    proteus::pyarray<double> &elementDiameter                            = args.array<double>("elementDiameter");
    int                  degree_polynomial                          = args.scalar<int>("degree_polynomial");
    proteus::pyarray<double> &u_dof                                      = args.array<double>("u_dof");
    proteus::pyarray<double> &u_dof_old                                  = args.array<double>("u_dof_old");
    proteus::pyarray<double> &velocity                                   = args.array<double>("velocity");
    proteus::pyarray<double> &q_m                                        = args.array<double>("q_m");
    proteus::pyarray<double> &q_theta                                    = args.array<double>("q_theta");
    proteus::pyarray<double> &q_u                                        = args.array<double>("q_u");
    proteus::pyarray<double> &q_dV                                       = args.array<double>("q_dV");
    proteus::pyarray<double> &q_m_betaBDF                                = args.array<double>("q_m_betaBDF");
    proteus::pyarray<double> &cfl                                        = args.array<double>("cfl");
    proteus::pyarray<double> &q_numDiff_u                                = args.array<double>("q_numDiff_u");
    proteus::pyarray<double> &q_numDiff_u_last                           = args.array<double>("q_numDiff_u_last");
    int                  offset_u                                   = args.scalar<int>("offset_u");
    int                  stride_u                                   = args.scalar<int>("stride_u");
    // component-1 (S_n) mass equation args.
    // (dt is already declared at the top of this function for the EV path.)
    proteus::pyarray<double> &u_dof_n                                    = args.array<double>("u_dof_n");
    proteus::pyarray<double> &u_dof_n_old                                = args.array<double>("u_dof_n_old");
    // gas-phase density: EXPONENTIAL EOS mirroring comp-0's slightly-compressible
    // water (rho_w = rho*exp(beta*u_w)).  rho_n is the reference density at
    // p_n = 0 (gauge = atmospheric); p_ref_n is the e-folding pressure scale:
    //   rho_n(p_n) = rho_n * exp(p_n / p_ref_n),   p_n = u_w + p_c,
    //   drho_n/dp_n = rho_n(p_n) / p_ref_n   (state-dependent, NOT a constant).
    // beta_n = 1/p_ref_n is the constant gas compressibility.  CO2 near
    // atmospheric in a lab rig is ideal-gas-like (rho ~ P_abs), so p_ref_n ~
    // atmospheric in head ~ 10.3 m gives beta_n ~ 0.1 /m.  p_ref_n <= 0 ->
    // incompressible (constant rho_n).  Exponential (vs the old linear c_n*p_n)
    // keeps rho_n = rho_n at gauge p_n = 0 rather than collapsing to 0.
    const double         rho_n                                      = args.scalar<double>("rho_n");
    const double         p_ref_n                                    = args.scalar<double>("p_ref_n");
    const bool           rho_n_compressible                         = (p_ref_n > 0.0);
    const double         inv_p_ref_n                                = rho_n_compressible ? (1.0 / p_ref_n) : 0.0;
    const int            offset_n                                   = args.scalar<int>("offset_n");
    const int            stride_n                                   = args.scalar<int>("stride_n");
    // Consistent (Galerkin) point-source injection (MOOSE DiracKernel form):
    //   R^c_i -= Q_port * N_i(x_p)  on the element containing the port.
    // sum_i N_i = 1 (partition of unity) -> total injected mass is EXACT and
    // mesh-independent; no elementMass lumping.  Solution-independent => zero
    // Jacobian (same as the lumped path).  inj_point_mode==0 keeps the legacy
    // lumped volumetric-disk source (injection_dof) byte-identical.
    const int            inj_point_mode = args.scalar<int>("inj_point_mode");
    const int            inj_n_ports    = args.scalar<int>("inj_n_ports");
    proteus::pyarray<int>    &inj_element    = args.array<int>("inj_element");   // containing elem id / rank (-1 if absent)
    proteus::pyarray<double> &inj_weight     = args.array<double>("inj_weight"); // N_i(x_p), [port*nDOF_test_element + i]
    proteus::pyarray<double> &inj_rate       = args.array<double>("inj_rate");   // Q_port * ramp(t)  [mol/(s*m_depth)]
    // Stage 3b: gas-side kinetic dissolution sink.  R_diss = k_d * S_n *
    // (1 - S_n) * theta_w * rho_w(c) * (c_sat - c) is subtracted from the
    // gas-equation residual at each quadrature point.  c is read from TADR's
    // u[0].dof aliased Python-side and passed in as c_dof.  k_d=0 disables
    // the sink (legacy behavior).
    proteus::pyarray<double> &c_dof                                      = args.array<double>("c_dof");
    const double         k_d                                        = args.scalar<double>("k_d");
    const double         c_sat                                      = args.scalar<double>("c_sat");
    // CO2 injection: per-node source field (built Python-side, schedule-gated).
    // Applied like R_diss but with opposite sign -- a source, not a sink.
    // All-zero array when no injection is configured.
    proteus::pyarray<double> &injection_dof                              = args.array<double>("injection_dof");
    proteus::pyarray<double> &globalResidual                             = args.array<double>("globalResidual");
    int                  nExteriorElementBoundaries_global          = args.scalar<int>("nExteriorElementBoundaries_global");
    proteus::pyarray<int>    &exteriorElementBoundariesArray             = args.array<int>("exteriorElementBoundariesArray");
    proteus::pyarray<int>    &elementBoundaryElementsArray               = args.array<int>("elementBoundaryElementsArray");
    proteus::pyarray<int>    &elementBoundaryLocalElementBoundariesArray = args.array<int>("elementBoundaryLocalElementBoundariesArray");
    proteus::pyarray<double> &ebqe_velocity_ext                          = args.array<double>("ebqe_velocity_ext");
    proteus::pyarray<int>    &isDOFBoundary_u                            = args.array<int>("isDOFBoundary_u");
    proteus::pyarray<double> &ebqe_bc_u_ext                              = args.array<double>("ebqe_bc_u_ext");
    // component-1 (S_n) boundary arrays.
    proteus::pyarray<int>    &isDOFBoundary_n                            = args.array<int>("isDOFBoundary_n");
    proteus::pyarray<double> &ebqe_bc_u_n_ext                            = args.array<double>("ebqe_bc_u_n_ext");
    proteus::pyarray<int>    &isFluxBoundary_u                           = args.array<int>("isFluxBoundary_u");
    proteus::pyarray<double> &ebqe_bc_flux_ext                           = args.array<double>("ebqe_bc_flux_ext");
    proteus::pyarray<double> &ebqe_phi                                   = args.array<double>("ebqe_phi");
    double               epsFact                                    = args.scalar<double>("epsFact");
    proteus::pyarray<double> &ebqe_u                                     = args.array<double>("ebqe_u");
    proteus::pyarray<double> &ebqe_theta                                 = args.array<double>("ebqe_theta");
    proteus::pyarray<double> &ebqe_flux                                  = args.array<double>("ebqe_flux");
    // PARAMETERS FOR EDGE BASED STABILIZATION
    double cE = args.scalar<double>("cE");
    double cK = args.scalar<double>("cK");
    // PARAMETERS FOR LOG BASED ENTROPY FUNCTION
    double uL = args.scalar<double>("uL");
    double uR = args.scalar<double>("uR");
    // PARAMETERS FOR EDGE VISCOSITY
    int               numDOFs                       = args.scalar<int>("numDOFs");
    // numDOFs is the compact component-0 free-DOF count used by the stabilized
    // DOF loops. Full matrix slots are recovered from the interleaved global
    // CSR using offset/stride-aware indexing.
    int               numDOFs_u                     = args.scalar<int>("numDOFs_u");
    int               NNZ                           = args.scalar<int>("NNZ");
    proteus::pyarray<int> &csrRowIndeces_DofLoops        = args.array<int>("csrRowIndeces_DofLoops");
    proteus::pyarray<int> &csrColumnOffsets_DofLoops     = args.array<int>("csrColumnOffsets_DofLoops");
    proteus::pyarray<int> &csrRowIndeces_Full            = args.array<int>("csrRowIndeces_Full");
    proteus::pyarray<int> &csrColumnOffsets_Full         = args.array<int>("csrColumnOffsets_Full");
    proteus::pyarray<int> &csrRowIndeces_CellLoops       = args.array<int>("csrRowIndeces_CellLoops");
    proteus::pyarray<int> &csrColumnOffsets_CellLoops    = args.array<int>("csrColumnOffsets_CellLoops");
    proteus::pyarray<int> &csrColumnOffsets_eb_CellLoops = args.array<int>("csrColumnOffsets_eb_CellLoops");
    // C matrices
    proteus::pyarray<double> &Cx  = args.array<double>("Cx");
    proteus::pyarray<double> &Cy  = args.array<double>("Cy");
    proteus::pyarray<double> &Cz  = args.array<double>("Cz");
    proteus::pyarray<double> &CTx = args.array<double>("CTx");
    proteus::pyarray<double> &CTy = args.array<double>("CTy");
    proteus::pyarray<double> &CTz = args.array<double>("CTz");
    proteus::pyarray<double> &ML  = args.array<double>("ML");
    proteus::pyarray<double> &MC  = args.array<double>("MC");

    proteus::pyarray<double> &delta_x_ij = args.array<double>("delta_x_ij");
    // PARAMETERS FOR 1st or 2nd ORDER MPP METHOD
    int LUMPED_MASS_MATRIX = args.scalar<int>("LUMPED_MASS_MATRIX");
    STABILIZATION STABILIZATION_TYPE{static_cast<STABILIZATION>(args.scalar<int>("STABILIZATION_TYPE"))};

    int ENTROPY_TYPE = args.scalar<int>("ENTROPY_TYPE");
    PSK_TYPE_member = args.scalar<int>("PSK_TYPE");
    immiscible_member = (args.scalar<int>("immiscible") != 0);
    T_C_member        = args.scalar<double>("T_C");      // temperature [degC] from input
    // FOR FCT
    proteus::pyarray<double> &dLow                 = args.array<double>("dLow");
    proteus::pyarray<double> &fluxMatrix           = args.array<double>("fluxMatrix");
    proteus::pyarray<double> &mDotLow              = args.array<double>("mDotLow");
    proteus::pyarray<double> &mLow                 = args.array<double>("mLow");
    proteus::pyarray<double> &dt_times_fH_minus_fL = args.array<double>("dt_times_fH_minus_fL");
    proteus::pyarray<double> &min_m_bc             = args.array<double>("min_m_bc");
    proteus::pyarray<double> &max_m_bc             = args.array<double>("max_m_bc");
    // AUX QUANTITIES OF INTEREST
    proteus::pyarray<double> &quantDOFs = args.array<double>("quantDOFs");
    proteus::pyarray<double> &mn        = args.array<double>("mn");
    proteus::pyarray<double> &fluxCorrection        = args.array<double>("fluxCorrection");
    proteus::pyarray<double> &limited_solution          = args.array<double>("limited_solution");
    proteus::pyarray<int>    &freeDOFMaterialTypes      = args.array<int>("freeDOFMaterialTypes");
    proteus::pyarray<int>    &freeDOFToNode_u           = args.array<int>("freeDOFToNode_u");
    // Per-comp-1-DOF volume material (mesh node index == comp-1 DOF index, since
    // gas has no Dirichlet). Used by the nodal closure-eval block below to
    // evaluate krn / p_c / rho_n with the node's actual sand parameters --
    // exact at the node, so dkrn/dS_n and dp_c/dS_n in the Jacobian match the
    // lambda used in the residual (needed for quadratic Newton / the mass-leak fix).
    proteus::pyarray<int>    &nodeMaterialTypes_n       = args.array<int>("nodeMaterialTypes_n");
    // Coarsest incident capillary entry pressure p_d=1/alpha [head] per mesh
    // node, for the comp-1 element-side capillary entry-pressure barrier.
    proteus::pyarray<double> &node_pd_min               = args.array<double>("node_pd_min");
    // Full gas saturation 1-S_wr of the coarsest incident medium per node (the
    // saturation the coarse pool fills to against a seal); anchors the valve.
    proteus::pyarray<double> &node_Sn_max               = args.array<double>("node_Sn_max");
    // DIAGNOSTIC (mass-creation hunt): gas_diag[0]=max|T_ij - T_ji| (tau
    // symmetry), [1]=max|T_ij| (scale), [2]=sum_ij F_ij (flux imbalance ->
    // net mass created/destroyed by the edge flux), [3]=sum_ij|F_ij| (scale).
    proteus::pyarray<double> &gas_diag                  = args.array<double>("gas_diag");

    proteus::pyarray<double> &velocity_couple               = args.array<double>("velocity_couple");
    proteus::pyarray<double> &ebqe_velocity_ext_couple      = args.array<double>("ebqe_velocity_ext_couple");
    // proteus::pyarray<double> &q_x    = args.array<double>("q_x");
    // proteus::pyarray<double> &ebqe_x = args.array<double>("ebqe_x");
    
    proteus::pyarray<double> &anb_seepage_flux_n = args.array<double>("anb_seepage_flux_n");
    proteus::pyarray<double> &q_velocity = args.array<double>("q_velocity");
    double &anb_seepage_flux(args.scalar<double>("anb_seepage_flux"));
    anb_seepage_flux = 0.0;
    proteus::pyarray<int>    &csrRowIndeces_u_u                          = args.array<int>("csrRowIndeces_u_u");
    proteus::pyarray<int>    &csrColumnOffsets_u_u                       = args.array<int>("csrColumnOffsets_u_u");
    proteus::pyarray<int>    &csrColumnOffsets_eb_u_u                    = args.array<int>("csrColumnOffsets_eb_u_u");
    // CSR maps for the (1,1) Jacobian block. Used by the
    // dedicated component-1 element loop appended at the end to assemble
    // the gas-side mass-matrix Jacobian / dt.
    proteus::pyarray<int>    &csrRowIndeces_n_n                          = args.array<int>("csrRowIndeces_n_n");
    // (1,0) cross-block CSR maps for the gas-eq diffusion
    // against grad u_w. Currently allocated by the framework but unused by
    // the C++ assembly - 3c.2/3c.3 will write into them.
    proteus::pyarray<int>    &csrRowIndeces_n_w                          = args.array<int>("csrRowIndeces_n_w");
    proteus::pyarray<int>    &csrColumnOffsets_n_n                       = args.array<int>("csrColumnOffsets_n_n");
    proteus::pyarray<int>    &csrColumnOffsets_n_w                       = args.array<int>("csrColumnOffsets_n_w");
    // P1: comp-0 (H2O) (0,0) and (0,1) block CSR maps -- the framework's
    // authoritative (row,col)->flat-nzval offsets for the water equation's
    // dependence on p (w_w) and z (w_n).  Used by the Richards-style block
    // scatter of the two-sided water flux Jacobian (replaces the Full-CSR
    // column search that dropped the (0,1) off-diagonal coupling).
    proteus::pyarray<int>    &csrRowIndeces_w_w                          = args.array<int>("csrRowIndeces_w_w");
    proteus::pyarray<int>    &csrColumnOffsets_w_w                       = args.array<int>("csrColumnOffsets_w_w");
    proteus::pyarray<int>    &csrRowIndeces_w_n                          = args.array<int>("csrRowIndeces_w_n");
    proteus::pyarray<int>    &csrColumnOffsets_w_n                       = args.array<int>("csrColumnOffsets_w_n");
    // Comp-1 boundary CSR maps used by the exterior boundary loop appended
    // at the end of this routine.
    proteus::pyarray<int>    &csrColumnOffsets_eb_n_n                    = args.array<int>("csrColumnOffsets_eb_n_n");
    proteus::pyarray<int>    &csrColumnOffsets_eb_n_w                    = args.array<int>("csrColumnOffsets_eb_n_w");
    // COMPONENT-1 (m_n = phi*rho_n*u_n) EV plumbing. Mirrors the comp-0
    // EV scaffolding: a compact DOF graph (csrRowIndeces_n_DofLoops), per-edge
    // dLow_n / dEV_n storage, and per-DOF mLow_n / mDotLow_n. Sensor / sensor
    // bounds (u_n_L, u_n_R) operate on S_n; the stabilization itself acts on
    // m_n through chain rule dm_n/du_n = -phi*rho_n.
    int                  numDOFs_n                  = args.scalar<int>("numDOFs_n");
    int                  NNZ_n                      = args.scalar<int>("NNZ_n");
    proteus::pyarray<int>    &csrRowIndeces_n_DofLoops    = args.array<int>("csrRowIndeces_n_DofLoops");
    proteus::pyarray<int>    &csrColumnOffsets_n_DofLoops = args.array<int>("csrColumnOffsets_n_DofLoops");
    proteus::pyarray<double> &dLow_n                    = args.array<double>("dLow_n");
    proteus::pyarray<double> &dEV_n                      = args.array<double>("dEV_n");
    proteus::pyarray<double> &fluxMatrix_n               = args.array<double>("fluxMatrix_n");
    proteus::pyarray<double> &mLow_n                     = args.array<double>("mLow_n");
    proteus::pyarray<double> &mDotLow_n                  = args.array<double>("mDotLow_n");
    double               u_n_L                       = args.scalar<double>("u_n_L");
    double               u_n_R                       = args.scalar<double>("u_n_R");
    proteus::pyarray<double> &mn_n        = args.array<double>("mn_n");           // m_n at t^n (numDOFs_u)
    proteus::pyarray<double> &quantDOFs_n = args.array<double>("quantDOFs_n");     // sensor scratch (numDOFs_u)
    // Per-node gas-residual BUDGET (mass-creation hunt). Each gas-equation
    // residual contribution is accumulated into its own slot at the node i it
    // is scattered to, so Python can sum over OWNED nodes only and MPI-reduce
    // (parallel-exact, no overlap double-count -- the owned-node pattern, like
    // mLow_n/mn_n). Layout is term-major, size 6*numDOFs_n:
    //   [0]=accumulation (m_n - mn_n)/dt    [1]=interior upwind flux  -sum F
    //   [2]=dissolution sink +R_diss        [3]=injection             -Q_inj
    //   [4]=exterior boundary flux          [5]=TOTAL scattered residual (~0 @ conv)
    // At convergence slot5~0, so slot0 = -(slot1+slot2+slot3+slot4). If the gas
    // mass grows yet slots 1+4 (the only non-telescoping, non-source terms) sum
    // to ~0, the creation is POST-kernel (FCT/inversion/coupling); if slot1 or
    // slot4 is the culprit it shows up here directly -- including a per-rank
    // boundary leak (slot4 != 0 on a partition that mis-tags an interior face).
    proteus::pyarray<double> &gas_budget_node = args.array<double>("gas_budget_node");
    // Comp-1 FCT plumbing read here so the gate at the end of this routine
    // can call FCTStep_n with all args present.
    proteus::pyarray<double> &dt_times_fH_minus_fL_n = args.array<double>("dt_times_fH_minus_fL_n");
    proteus::pyarray<double> &fluxCorrection_n       = args.array<double>("fluxCorrection_n");
    int                  FCT_n                  = args.scalar<int>("FCT_n");
    // ===== Per-rock nodal closure (Z_ACROSS_ROCK_INTERFACES.md Sec. 3) =====
    // At a node shared by several rocks the flash's S_g is the pore-volume
    // average of the per-rock gas saturations: substituting c_r, w_r into the
    // design's G(pi) = 0 gives
    //   sum_r W_ir S_r(pi_i) = W_i S_g(p_i, z_i),   W_i = sum_r W_ir,
    // one capillary pressure pi_i per node, S_r = the inverse of rock r's
    // bc_pc_from_Se (bc_Se_from_pc).  dS_r/d(p,z) follow from the
    // implicit-function theorem.  Storage needs only W_i (it is W_i N z for
    // CO2 and W_i N (1-z) for H2O), so the S_r cache is read by the flux loops
    // only.  One-rock nodes keep the flash's S_g (node_multi = 0), so the
    // closure changes nothing away from rock interfaces.  Built for the current
    // iterate and the old time level.  1D prototype:
    // test/m_comp_co2/verification/rock_interface_1d/column_1d.py.
    const int per_rock = args.scalar<int>("per_rock_closure");
    proteus::pyarray<double> &node_rock_W = args.array<double>("node_rock_W");   // [node][rock]
    const int nRock = args.scalar<int>("nRockTypes");
    std::vector<char>   node_multi;
    std::vector<double> node_W;                         // W_i = sum_r W_ir
    std::vector<double> rS, rS_dp, rS_dz;               // [node * nRock + rock], current
    std::vector<double> rS_o, rS_dz_o;                  // old time level
    if (per_rock) {
      node_multi.assign(numDOFs_n, 0);
      node_W.assign(numDOFs_n, 0.0);
      rS.assign(numDOFs_n * nRock, 0.0);  rS_dp.assign(numDOFs_n * nRock, 0.0);
      rS_dz.assign(numDOFs_n * nRock, 0.0);
      rS_o.assign(numDOFs_n * nRock, 0.0);  rS_dz_o.assign(numDOFs_n * nRock, 0.0);
      for (int i = 0; i < numDOFs_n; i++) {
        const double *Wi = &node_rock_W.data()[i * nRock];
        int n_r = 0;
        for (int r = 0; r < nRock; r++) {
          node_W[i] += Wi[r];
          if (Wi[r] > 0.0) n_r++;
        }
        if (n_r < 2) continue;
        node_multi[i] = 1;
        // Bracket for pi: gas first enters the coarsest rock at its p_d; above
        // the largest pc(S_e = 0) every rock is at its S_e = 0 cap.
        double pi_lo = 1.0e300, pi_hi = 0.0;
        for (int r = 0; r < nRock; r++) {
          if (Wi[r] <= 0.0) continue;
          double pc0, dpc0, d2pc0;
          proteus::m_comp_co2::psk::bc_pc_from_Se(0.0, alpha.data()[r], n.data()[r], bc_Se_min.data()[r], pc0, dpc0, d2pc0);
          pi_lo = std::min(pi_lo, 1.0 / alpha.data()[r]);
          pi_hi = std::max(pi_hi, pc0);
        }
        for (int lvl = 0; lvl < 2; lvl++) {
          const double p_i = fmax((lvl == 0 ? u_dof.data()[i] : u_dof_old.data()[i]), 1.0e2);
          const double z_i = (lvl == 0 ? u_dof_n.data()[i] : u_dof_n_old.data()[i]);
          ::m_comp_co2::flash::FlashState f =
              ::m_comp_co2::flash::flashPZ(p_i, z_i, T_C_member, m_NaCl_member, ::m_comp_co2::flash::EPS_Z, immiscible_member);
          const double target = node_W[i] * f.S_g;
          // H(pi) = sum_r W_ir S_r(pi) - W_i S_g and dH/dpi; S_r = (1 - S_wr,r)(1 - S_e,r).
          auto H = [&](const double pi, double &dH) {
            double s = 0.0;
            dH = 0.0;
            for (int r = 0; r < nRock; r++) {
              if (Wi[r] <= 0.0) continue;
              const double om = thetaSR.data()[r] / (thetaR.data()[r] + thetaSR.data()[r]);
              double Se, dSe;
              proteus::m_comp_co2::psk::bc_Se_from_pc(pi, alpha.data()[r], n.data()[r], bc_Se_min.data()[r], Se, dSe);
              s  += Wi[r] * om * (1.0 - Se);
              dH -= Wi[r] * om * dSe;
            }
            return s - target;
          };
          double pi = pi_lo, dH = 0.0;
          bool capped = false;
          if (target <= 0.0) {
            pi = pi_lo;                         // single-phase node: every S_r = 0
          } else if (H(pi_hi, dH) < 0.0) {
            capped = true;                      // branch (d): beyond every rock's S_e = 0
          } else {
            // Safeguarded Newton from the low end of the bracket; bisection
            // whenever the step leaves it.  1e-15 relative resolves S_r to roundoff.
            double a = pi_lo, b = pi_hi;
            for (int it = 0; it < 200; it++) {
              const double h = H(pi, dH);
              if (h < 0.0) a = pi; else b = pi;
              if (std::fabs(h) <= 1.0e-15 * target || b - a <= 1.0e-15 * b) break;
              const double pn = (dH > 0.0) ? pi - h / dH : 0.5 * (a + b);
              pi = (pn > a && pn < b) ? pn : 0.5 * (a + b);
            }
            H(pi, dH);
          }
          for (int r = 0; r < nRock; r++) {
            if (Wi[r] <= 0.0) continue;
            const int o = i * nRock + r;
            const double om = thetaSR.data()[r] / (thetaR.data()[r] + thetaSR.data()[r]);
            double S_r, dS_dp, dS_dz;
            if (capped) {
              // The kernel clamps S_e at 0, so this excess changes no k_r or
              // p_c; spread it uniformly so sum_r W_ir S_r = W_i S_g still holds.
              double h_hi = H(pi_hi, dH);
              S_r = om - h_hi / node_W[i];
              dS_dp = f.dS_g_dp;  dS_dz = f.dS_g_dz;
            } else {
              double Se, dSe;
              proteus::m_comp_co2::psk::bc_Se_from_pc(pi, alpha.data()[r], n.data()[r], bc_Se_min.data()[r], Se, dSe);
              S_r = (target <= 0.0) ? 0.0 : om * (1.0 - Se);
              // dpi/d(p,z) = W_i dS_g/d(p,z) / dH/dpi; with dH = 0 (flat at the
              // root, measure zero) fall back to the average's derivative.
              const double dSr_dpi = -om * dSe;
              if (target > 0.0 && dH > 0.0) {
                dS_dp = dSr_dpi * node_W[i] * f.dS_g_dp / dH;
                dS_dz = dSr_dpi * node_W[i] * f.dS_g_dz / dH;
              } else if (target > 0.0) {
                dS_dp = f.dS_g_dp;  dS_dz = f.dS_g_dz;
              } else {
                dS_dp = 0.0;  dS_dz = 0.0;
              }
            }
            if (lvl == 0) { rS[o] = S_r; rS_dp[o] = dS_dp; rS_dz[o] = dS_dz; }
            else          { rS_o[o] = S_r; rS_dz_o[o] = dS_dz; }
          }
        }
      }
      Sg_rock_min_member.assign(numDOFs_n, std::nan(""));
      Sg_rock_max_member.assign(numDOFs_n, std::nan(""));
      for (int i = 0; i < numDOFs_n; i++) {
        if (!node_multi[i]) continue;
        double s_lo = 1.0e300, s_hi = -1.0e300;
        for (int r = 0; r < nRock; r++) {
          if (node_rock_W.data()[i * nRock + r] <= 0.0) continue;
          s_lo = std::min(s_lo, rS[i * nRock + r]);
          s_hi = std::max(s_hi, rS[i * nRock + r]);
        }
        Sg_rock_min_member[i] = s_lo;
        Sg_rock_max_member[i] = s_hi;
      }
    }
    // double Rpos[numDOFs], Rneg[numDOFs];
     std::vector<double> Rpos(numDOFs, 0.0), Rneg(numDOFs, 0.0);
     std::vector<double> TransportMatrix(NNZ, 0.0),
                        TransportMatrixConsistent(NNZ, 0.0),
                        TransportMatrixn(NNZ, 0.0),
                        TransportMatrixConsistentn(NNZ, 0.0);
    //double FluxCorrectionMatrix[NNZ];
    // NOTE: This function follows a different (but equivalent) implementation of the smoothness based indicator than NCLS.h
    // Allocate space for the transport matrices
    // This is used for first order KUZMIN'S METHOD
    // double                TransportMatrix[NNZ], TransportMatrixConsistent[NNZ];
    // double                TransportMatrixn[NNZ], TransportMatrixConsistentn[NNZ];
    std::valarray<double> u_free_dof(numDOFs);
    std::valarray<double> u_free_dof_old(numDOFs);
    std::valarray<double> ML2(numDOFs);
    // Lumped L2 projection buffers for density
    std::vector<double> rho_dof(numDOFs, 0.0);
    std::vector<double> ML_rho(numDOFs, 0.0);
    std::fill(velocity_couple.data(), velocity_couple.data() + velocity_couple.size(), 0.0);
    std::fill(ebqe_velocity_ext_couple.data(), ebqe_velocity_ext_couple.data() + ebqe_velocity_ext_couple.size(), 0.0);
    auto full_offset_from_compact = [&](int i_compact, int j_compact) -> int
    {
      const int full_i = offset_u + stride_u * i_compact;
      const int full_j = offset_u + stride_u * j_compact;
      for (int offset = csrRowIndeces_Full.data()[full_i]; offset < csrRowIndeces_Full.data()[full_i + 1]; ++offset)
        if (csrColumnOffsets_Full.data()[offset] == full_j) return offset;
      return -1;
    };
    
     for (int eN = 0; eN < nElements_global; eN++)
      for (int j = 0; j < nDOF_trial_element; j++) {
        int eN_nDOF_trial_element                               = eN * nDOF_trial_element;
        u_free_dof[r_l2g.data()[eN_nDOF_trial_element + j]]     = u_dof.data()[u_l2g.data()[eN_nDOF_trial_element + j]];
        u_free_dof_old[r_l2g.data()[eN_nDOF_trial_element + j]] = u_dof_old.data()[u_l2g.data()[eN_nDOF_trial_element + j]];
      }
    for (int i = 0; i < NNZ; i++) {
      TransportMatrix[i]            = 0.;
      TransportMatrixConsistent[i]  = 0.;
      TransportMatrixn[i]           = 0.;
      TransportMatrixConsistentn[i] = 0.;
    }

    // Project quadrature density to nodal DOFs before constructing nodal
    // stabilization potentials so Phi uses a true nodal density field.
    for (int eN = 0; eN < nElements_global; eN++) {
      const int eN_nDOF_trial_element = eN * nDOF_trial_element;
      for (int k = 0; k < nQuadraturePoints_element; k++) {
        const int eN_k = eN * nQuadraturePoints_element + k;
        double jac[nSpace * nSpace], jacDet, jacInv[nSpace * nSpace], x, y, z;
        ck.calculateMapping_element(eN, k, mesh_dof.data(), mesh_l2g.data(),
                                    mesh_trial_ref.data(), mesh_grad_trial_ref.data(),
                                    jac, jacDet, jacInv, x, y, z);
        const double dV = fabs(jacDet) * dV_ref.data()[k];
        for (int i = 0; i < nDOF_test_element; i++) {
          const int eN_i = eN * nDOF_test_element + i;
          const int free_gi = r_l2g.data()[eN_i];
          const double u_test_dV = u_test_ref.data()[k * nDOF_trial_element + i] * dV;
          rho_dof[free_gi] += q_rho.data()[eN_k] * u_test_dV;
          ML_rho[free_gi] += u_test_dV;
        }
      }
    }
    for (int i = 0; i < numDOFs; ++i) {
      if (ML_rho[i] > 0.0) rho_dof[i] /= ML_rho[i];
      else rho_dof[i] = rho;
    }
    // Cache the projected, salinity-coupled density for use in invert() so
    // the m -> u inversion is consistent with the forward residual.
    rho_dof_member = rho_dof;

    // -------- Comp-1: lumped L2 projections for the upwind potential flux --------
    // rho_n_phi_dof, rho_w_dof: as before -- invert(COMPONENT=1) and R_diss.
    // pc_dof, dpc_dof, krn_dof, dkrn_dof, rho_n_dof: nodal closure values
    
    std::vector<double> rho_n_phi_dof(numDOFs_n, 0.0);
    // Old-time phi*N for the accumulation old mass m_c_old = (phi*N_old)*z_old.
    std::vector<double> rho_n_phi_dof_old(numDOFs_n, 0.0);
    // Lumped nodal phi*dN/dz and phi*dN/dp -- the compositional accumulation
    // Jacobian needs dm_c/dz = phi*(dN/dz*z + N) and dm_c/dp = phi*dN/dp*z.
    std::vector<double> dphiN_dz_dof(numDOFs_n, 0.0);
    std::vector<double> dphiN_dp_dof(numDOFs_n, 0.0);
    std::vector<double> rho_w_dof(numDOFs_n, 0.0);
    std::vector<double> rho_n_dof(numDOFs_n, 0.0);
    std::vector<double> pc_dof(numDOFs_n, 0.0);
    std::vector<double> dpc_dof(numDOFs_n, 0.0);
    std::vector<double> krn_dof(numDOFs_n, 0.0);
    std::vector<double> dkrn_dof(numDOFs_n, 0.0);
    std::vector<double> rho_n_dof_old(numDOFs_n, 0.0);
    std::vector<double> pc_dof_old(numDOFs_n, 0.0);
    std::vector<double> krn_dof_old(numDOFs_n, 0.0);
    // Uncapped Brooks-Corey p_c (+ dp_c/dS_n and old-time sibling), used ONLY on
    // material-interface edges for entry-pressure capillary breakthrough. 
    std::vector<double> pc_uncap_dof(numDOFs_n, 0.0);
    std::vector<double> dpc_uncap_dof(numDOFs_n, 0.0);
    std::vector<double> pc_uncap_dof_old(numDOFs_n, 0.0);
    std::vector<double> ML_n(numDOFs_n, 0.0);
    for (int eN = 0; eN < nElements_global; eN++) {
      const int    mat_eN_proj = elementMaterialTypes.data()[eN];
      const double phi_eN      = thetaR.data()[mat_eN_proj] + thetaSR.data()[mat_eN_proj];
      const double alpha_eN_p  = alpha.data()[mat_eN_proj];
      const double n_vg_eN_p   = n.data()[mat_eN_proj];
      const double krn_end_p   = krn_end.data()[mat_eN_proj];
      const double S_wr_p      = thetaR.data()[mat_eN_proj] / phi_eN;
      const double one_m_Sr_p  = 1.0 - S_wr_p;
      const double Se_trap_L3043 = 1.0 - S_gr.data()[mat_eN_proj] / one_m_Sr_p;  // gas-only residual trapping
      const int    eN_nDOF_trial_element = eN * nDOF_trial_element;
      for (int k = 0; k < nQuadraturePoints_element; k++) {
        double jac[nSpace * nSpace], jacDet, jacInv[nSpace * nSpace], x_p, y_p, z_p;
        ck.calculateMapping_element(eN, k, mesh_dof.data(), mesh_l2g.data(),
                                    mesh_trial_ref.data(), mesh_grad_trial_ref.data(),
                                    jac, jacDet, jacInv, x_p, y_p, z_p);
        const double dV = std::fabs(jacDet) * dV_ref.data()[k];
        // Current iterate values at QP.
        double u_w_p = 0.0, u_n_p = 0.0;
        ck.valFromDOF(u_dof.data(),
                      &u_l2g.data()[eN_nDOF_trial_element],
                      &u_trial_ref.data()[k * nDOF_trial_element], u_w_p);
        ck.valFromDOF(u_dof_n.data(),
                      &u_l2g_n.data()[eN_nDOF_trial_element],
                      &u_trial_ref.data()[k * nDOF_trial_element], u_n_p);
        // Closure at current iterate -- Se, p_c, dp_c/dS_n, k_rn, dk_rn/dS_n.
        const double Se_p_raw = (1.0 - u_n_p - S_wr_p) / one_m_Sr_p;
        double Se_p, dSe_du_n_p;
        if (Se_p_raw <= 0.0)      { Se_p = 0.0; dSe_du_n_p = 0.0; }
        else if (Se_p_raw >= 1.0) { Se_p = 1.0; dSe_du_n_p = 0.0; }
        else                      { Se_p = Se_p_raw; dSe_du_n_p = -1.0 / one_m_Sr_p; }
        double pc_p = 0.0, dpc_dSe_p = 0.0, d2pc_p_unused = 0.0;
        if (PSK_TYPE_member == 1)
          proteus::m_comp_co2::psk::bc_pc_from_Se(Se_p, alpha_eN_p, n_vg_eN_p, bc_Se_min.data()[mat_eN_proj], pc_p, dpc_dSe_p, d2pc_p_unused);
        else
          proteus::m_comp_co2::psk::vgm_pc_from_Se(Se_p, alpha_eN_p, n_vg_eN_p, pc_p, dpc_dSe_p, d2pc_p_unused);
        const double dpc_dSn_p = dpc_dSe_p * dSe_du_n_p;
        double krn_p = 0.0, dkrn_dSe_p = 0.0;
        if (PSK_TYPE_member == 1)
          proteus::m_comp_co2::psk::bc_kr_nonwetting_from_Se(Se_p, alpha_eN_p, n_vg_eN_p, krn_p, dkrn_dSe_p, Se_trap_L3043);
        else
          proteus::m_comp_co2::psk::vgm_kr_nonwetting_from_Se(Se_p, alpha_eN_p, n_vg_eN_p, krn_p, dkrn_dSe_p, Se_trap_L3043);
        krn_p             = krn_p * krn_end_p / mu_n;
        const double dkrn_dSn_p = dkrn_dSe_p * krn_end_p * dSe_du_n_p / mu_n;
        const double rho_n_p     = rho_n_compressible ? (rho_n * exp(fmin((u_w_p + pc_p) * inv_p_ref_n, 50.0))) : rho_n;
        const double phi_rho_n_qp = phi_eN * rho_n_p;
        // Old-time-level values for the (1-Theta) part of the edge flux.
        double u_w_p_old = 0.0, u_n_p_old = 0.0;
        ck.valFromDOF(u_dof_old.data(),
                      &u_l2g.data()[eN_nDOF_trial_element],
                      &u_trial_ref.data()[k * nDOF_trial_element], u_w_p_old);
        ck.valFromDOF(u_dof_n_old.data(),
                      &u_l2g_n.data()[eN_nDOF_trial_element],
                      &u_trial_ref.data()[k * nDOF_trial_element], u_n_p_old);
        const double Se_p_old_raw = (1.0 - u_n_p_old - S_wr_p) / one_m_Sr_p;
        double Se_p_old;
        if (Se_p_old_raw <= 0.0)      Se_p_old = 0.0;
        else if (Se_p_old_raw >= 1.0) Se_p_old = 1.0;
        else                          Se_p_old = Se_p_old_raw;
        double pc_p_old = 0.0, dpc_p_old_unused = 0.0, d2pc_p_old_unused = 0.0;
        if (PSK_TYPE_member == 1)
          proteus::m_comp_co2::psk::bc_pc_from_Se(Se_p_old, alpha_eN_p, n_vg_eN_p, bc_Se_min.data()[mat_eN_proj], pc_p_old, dpc_p_old_unused, d2pc_p_old_unused);
        else
          proteus::m_comp_co2::psk::vgm_pc_from_Se(Se_p_old, alpha_eN_p, n_vg_eN_p, pc_p_old, dpc_p_old_unused, d2pc_p_old_unused);
        double krn_p_old = 0.0, dkrn_p_old_unused = 0.0;
        if (PSK_TYPE_member == 1)
          proteus::m_comp_co2::psk::bc_kr_nonwetting_from_Se(Se_p_old, alpha_eN_p, n_vg_eN_p, krn_p_old, dkrn_p_old_unused, Se_trap_L3043);
        else
          proteus::m_comp_co2::psk::vgm_kr_nonwetting_from_Se(Se_p_old, alpha_eN_p, n_vg_eN_p, krn_p_old, dkrn_p_old_unused, Se_trap_L3043);
        krn_p_old = krn_p_old * krn_end_p / mu_n;
        const double rho_n_p_old = rho_n_compressible ? (rho_n * exp(fmin((u_w_p_old + pc_p_old) * inv_p_ref_n, 50.0))) : rho_n;
        // TADR brine density at this QP.
        const int    eN_k_proj    = eN * nQuadraturePoints_element + k;
        const double rho_w_qp_proj = q_rho.data()[eN_k_proj];
        const double p_cl_pr     = fmax(u_w_p, 1.0e2);
        const double p_cl_pr_old = fmax(u_w_p_old, 1.0e2);
        ::m_comp_co2::flash::FlashState fs_pr =
            ::m_comp_co2::flash::flashPZ(p_cl_pr, u_n_p, T_C_member, m_NaCl_member, ::m_comp_co2::flash::EPS_Z, immiscible_member);
        ::m_comp_co2::flash::FlashState fs_pr_old =
            ::m_comp_co2::flash::flashPZ(p_cl_pr_old, u_n_p_old, T_C_member, m_NaCl_member, ::m_comp_co2::flash::EPS_Z, immiscible_member);
        const double Sa_pr    = 1.0 - fs_pr.S_g;
        const double N_pr     = fs_pr.rho_g*fs_pr.S_g + fs_pr.rho_a*Sa_pr;
        const double dN_dp_pr = fs_pr.drho_g_dp*fs_pr.S_g + fs_pr.rho_g*fs_pr.dS_g_dp
                              + fs_pr.drho_a_dp*Sa_pr     - fs_pr.rho_a*fs_pr.dS_g_dp;
        const double dN_dz_pr = fs_pr.drho_g_dz*fs_pr.S_g + fs_pr.rho_g*fs_pr.dS_g_dz
                              + fs_pr.drho_a_dz*Sa_pr     - fs_pr.rho_a*fs_pr.dS_g_dz;
        const double N_pr_old = fs_pr_old.rho_g*fs_pr_old.S_g
                              + fs_pr_old.rho_a*(1.0 - fs_pr_old.S_g);
        const double phiN_qp     = phi_eN * N_pr;
        const double phiN_old_qp = phi_eN * N_pr_old;
        const double dphiN_dz_qp = phi_eN * dN_dz_pr;
        const double dphiN_dp_qp = phi_eN * dN_dp_pr;
        for (int i = 0; i < nDOF_test_element; i++) {
          const int    eN_i = eN * nDOF_test_element + i;
          const int    gi   = u_l2g_n.data()[eN_i];   // comp-1 DOF (lumped phi*N projection)
          const double u_test_dV = u_test_ref.data()[k * nDOF_trial_element + i] * dV;
          // P1: phi*N (and derivatives) for the compositional CO2 accumulation.
          rho_n_phi_dof[gi]     += phiN_qp * u_test_dV;
          rho_n_phi_dof_old[gi] += phiN_old_qp * u_test_dV;
          dphiN_dz_dof[gi]      += dphiN_dz_qp * u_test_dV;
          dphiN_dp_dof[gi]      += dphiN_dp_qp * u_test_dV;
          // Phase-based closures below still feed the (P2) upwind flux pass.
          rho_w_dof[gi]     += rho_w_qp_proj * u_test_dV;
          rho_n_dof[gi]     += rho_n_p * u_test_dV;
          pc_dof[gi]        += pc_p * u_test_dV;
          dpc_dof[gi]       += dpc_dSn_p * u_test_dV;
          krn_dof[gi]       += krn_p * u_test_dV;
          dkrn_dof[gi]      += dkrn_dSn_p * u_test_dV;
          rho_n_dof_old[gi] += rho_n_p_old * u_test_dV;
          pc_dof_old[gi]    += pc_p_old * u_test_dV;
          krn_dof_old[gi]   += krn_p_old * u_test_dV;
          ML_n[gi]          += u_test_dV;
        }
      }
    }
    // Normalize over ALL comp-1 nodes (numDOFs_n), not numDOFs_u -- the arrays
    // and the projection above span every mesh node, including the top-boundary
    // p_w-Dirichlet nodes that are absent from the comp-0 free-DOF count.
    for (int i = 0; i < numDOFs_n; ++i) {
      if (ML_n[i] > 0.0) {
        rho_n_phi_dof[i] /= ML_n[i];
        rho_n_phi_dof_old[i] /= ML_n[i];
        dphiN_dz_dof[i]  /= ML_n[i];
        dphiN_dp_dof[i]  /= ML_n[i];
        rho_w_dof[i]     /= ML_n[i];
        rho_n_dof[i]     /= ML_n[i];
        pc_dof[i]        /= ML_n[i];
        dpc_dof[i]       /= ML_n[i];
        krn_dof[i]       /= ML_n[i];
        dkrn_dof[i]      /= ML_n[i];
        rho_n_dof_old[i] /= ML_n[i];
        pc_dof_old[i]    /= ML_n[i];
        krn_dof_old[i]   /= ML_n[i];
      } else {
        rho_n_phi_dof[i] = thetaR.data()[0] + thetaSR.data()[0]; // fallback
        rho_n_phi_dof_old[i] = thetaR.data()[0] + thetaSR.data()[0]; // fallback
        rho_w_dof[i]     = rho;                                  // fallback
        rho_n_dof[i]     = rho_n;                                // fallback
        rho_n_dof_old[i] = rho_n;
        // pc, dpc, krn, dkrn default to 0 -- only nodes with no element
        // contributions ever hit this branch.
      }
      rho_n_phi_dof[i] = std::max(rho_n_phi_dof[i], 1.0e-16);
    }
    rho_n_phi_dof_member = rho_n_phi_dof;

    // compute entropy and init global_entropy_residual and boundary_integral
    double psi[numDOFs], eta[numDOFs], global_entropy_residual[numDOFs], boundary_integral[numDOFs];
    for (int i = 0; i < numDOFs; i++) {
      // NODAL ENTROPY //
      if (STABILIZATION_TYPE == STABILIZATION::EV_Stab) //EV stab
      {
        double solni = 1.0 * u_free_dof_old[i];
        eta[i]                      = ENTROPY_TYPE == 1 ? ENTROPY(solni, uL, uR) : ENTROPY_LOG(solni, uL, uR);
        global_entropy_residual[i]  = 0.;
      }
      boundary_integral[i] = 0.;
      ML2[i]               = 0.0;
    }

    //////////////////////////////////////////////
    // ** LOOP IN CELLS FOR CELL BASED TERMS ** //
    //////////////////////////////////////////////
    // HERE WE COMPUTE:
    //    * Time derivative term. u_t
    //    * cell based CFL (for reference)
    //    * Entropy residual
    //    * Transport matrices

    for (int eN = 0; eN < nElements_global; eN++) {
      const int eN_nDOF_trial_element = eN * nDOF_trial_element;
      const int eN_nDOF_mesh_trial_element = eN * nDOF_mesh_trial_element;
      //declare local storage for local contributions and initialize
      double elementResidual_u[nDOF_test_element], element_entropy_residual[nDOF_test_element], Phi[nDOF_trial_element], Phi_n[nDOF_trial_element];
      double elementTransport[nDOF_test_element][nDOF_trial_element], elementTransportConsistent[nDOF_test_element][nDOF_trial_element];
      double elementTransportn[nDOF_test_element][nDOF_trial_element], elementTransportConsistentn[nDOF_test_element][nDOF_trial_element];
      for (int j = 0; j < nDOF_trial_element; j++) {
        const int u_gj = u_l2g.data()[eN_nDOF_trial_element + j];
        const int free_gj = r_l2g.data()[eN_nDOF_trial_element + j];
        const int x_gj = mesh_l2g.data()[eN_nDOF_mesh_trial_element + j];
        const double rho_node_j = rho_dof[free_gj];
        Phi[j]   = u_dof.data()[u_gj];
        Phi_n[j] = u_dof_old.data()[u_gj];
        for (int I = 0; I < nSpace; I++) {
          // Pressure form: grad(p) - rho_w g (no /rho0 factor).
          Phi[j]   -= rho_node_j * mesh_dof.data()[x_gj * 3 + I] * gravity[I];
          Phi_n[j] -= rho_node_j * mesh_dof.data()[x_gj * 3 + I] * gravity[I];
        }
      }
      for (int i = 0; i < nDOF_test_element; i++) {
        elementResidual_u[i]        = 0.0;
        element_entropy_residual[i] = 0.0;
        for (int j = 0; j < nDOF_trial_element; j++) {
          elementTransport[i][j]            = 0.0;
          elementTransportConsistent[i][j]  = 0.0;
          elementTransportn[i][j]           = 0.0;
          elementTransportConsistentn[i][j] = 0.0;
        }
      }
      //loop over quadrature points and compute integrands
      for (int k = 0; k < nQuadraturePoints_element; k++) {
        //compute indeces and declare local storage
        int eN_k = eN * nQuadraturePoints_element + k, eN_k_nSpace = eN_k * nSpace;
        double
          // for entropy residual
          aux_entropy_residual = 0.,
          DENTROPY_un, DENTROPY_uni,
          //for mass matrix contributions
          u = 0.0, un = 0.0, grad_phi[nSpace], grad_phi_n[nSpace], grad_u_velocity[nSpace], velocity_loc[nSpace], u_test_dV[nDOF_trial_element], u_grad_trial[nDOF_trial_element * nSpace], u_grad_test_dV[nDOF_test_element * nSpace],
          //for general use
          jac[nSpace * nSpace], jacDet, jacInv[nSpace * nSpace], dV, x, y, z, xt, yt, zt, m, dm, f[nSpace], df[nSpace], a[nnz], da[nnz], as[nnz], mn, dmn, fn[nSpace], dfn[nSpace], an[nnz], dan[nnz], asn[nnz];
        //get the physical integration weight
        ck.calculateMapping_element(eN, k, mesh_dof.data(), mesh_l2g.data(), mesh_trial_ref.data(), mesh_grad_trial_ref.data(), jac, jacDet, jacInv, x, y, z);
        ck.calculateMappingVelocity_element(eN, k, mesh_velocity_dof.data(), mesh_l2g.data(), mesh_trial_ref.data(), xt, yt, zt);
        dV = fabs(jacDet) * dV_ref.data()[k];
        //get the solution (of Newton's solver). To compute time derivative term
        ck.valFromDOF(u_dof.data(), &u_l2g.data()[eN_nDOF_trial_element], &u_trial_ref.data()[k * nDOF_trial_element], u);
        //get the solution at quad point at tn and tnm1 for entropy viscosity
        ck.valFromDOF(u_dof_old.data(), &u_l2g.data()[eN_nDOF_trial_element], &u_trial_ref.data()[k * nDOF_trial_element], un);
        //get the solution gradients at tn for entropy viscosity
        ck.gradTrialFromRef(&u_grad_trial_ref.data()[k * nDOF_trial_element * nSpace], jacInv, u_grad_trial);
        ck.gradFromDOF(u_dof.data(), &u_l2g.data()[eN_nDOF_trial_element], u_grad_trial, grad_u_velocity);

        //populate q_x 
        // const int eN_k_3d = eN_k * 3;
        // q_x.data()[eN_k_3d + 0] = x;
        // q_x.data()[eN_k_3d + 1] = y;
        // q_x.data()[eN_k_3d + 2] = z;
        //precalculate test function products with integration weights for mass matrix terms
        for (int I = 0; I < nSpace; I++) {
          grad_phi[I]  = 0.0;
          grad_phi_n[I] = 0.0;
        }
        for (int j = 0; j < nDOF_trial_element; j++) {
          u_test_dV[j] = u_test_ref.data()[k * nDOF_trial_element + j] * dV;
          for (int I = 0; I < nSpace; I++) {
            grad_phi_n[I] += Phi_n[j] * u_grad_trial[j * nSpace + I];
            grad_phi[I] += Phi[j] * u_grad_trial[j * nSpace + I];
            u_grad_test_dV[j * nSpace + I] = u_grad_trial[j * nSpace + I] * dV; //cek warning won't work for Petrov-Galerkin
          }
        }
        //
        //calculate pde coefficients at quadrature points
        //
        double Kr, dKr, Krn, dKrn, thetaW, thetaWn;
        const double rho_local = q_rho.data()[eN_k];
        const double rho_velocity = std::fabs(rho_local) > 1.0e-12 ? rho_local : rho;

        // Cross-derivative buffers are filled by _from_Se but ignored downstream;
        // the EV residual only uses (mn, fn, an) and (m, f, a).
        double dm_du_n_qp_n = 0.0, dkr_du_n_qp_n = 0.0;
        double dm_du_n_qp = 0.0, dkr_du_n_qp = 0.0;
        double df_du_n_qp_n[nSpace], df_du_n_qp[nSpace];
        double da_du_n_qp_n[nnz],    da_du_n_qp[nnz];
        for (int I = 0; I < nSpace; I++) { df_du_n_qp_n[I] = 0.0; df_du_n_qp[I] = 0.0; }
        for (int ii = 0; ii < nnz; ii++) { da_du_n_qp_n[ii] = 0.0; da_du_n_qp[ii] = 0.0; }
        double u_n_qp = 0.0, u_n_qp_old = 0.0;
        ck.valFromDOF(u_dof_n.data(),
                      &u_l2g.data()[eN_nDOF_trial_element],
                      &u_trial_ref.data()[k * nDOF_trial_element], u_n_qp);
        ck.valFromDOF(u_dof_n_old.data(),
                      &u_l2g.data()[eN_nDOF_trial_element],
                      &u_trial_ref.data()[k * nDOF_trial_element], u_n_qp_old);
        evaluateCoefficients_from_Se(a_rowptr.data(), a_colind.data(), rho, rho_local, beta, gravity.data(),
                                     alpha.data()[elementMaterialTypes[eN]], n.data()[elementMaterialTypes[eN]],
                                     thetaR.data()[elementMaterialTypes[eN]], thetaSR.data()[elementMaterialTypes[eN]],
                                     &KWs.data()[elementMaterialTypes[eN] * nnz], un, u_n_qp_old,
                                     mn, dmn, dm_du_n_qp_n, fn, dfn, df_du_n_qp_n, an, dan, da_du_n_qp_n,
                                     asn, Krn, dKrn, dkr_du_n_qp_n, thetaWn);
        evaluateCoefficients_from_Se(a_rowptr.data(), a_colind.data(), rho, rho_local, beta, gravity.data(),
                                     alpha.data()[elementMaterialTypes[eN]], n.data()[elementMaterialTypes[eN]],
                                     thetaR.data()[elementMaterialTypes[eN]], thetaSR.data()[elementMaterialTypes[eN]],
                                     &KWs.data()[elementMaterialTypes[eN] * nnz], u, u_n_qp,
                                     m, dm, dm_du_n_qp, f, df, df_du_n_qp, a, da, da_du_n_qp,
                                     as, Kr, dKr, dkr_du_n_qp, thetaW);
        q_theta.data()[eN_k] = thetaW;

        // Darcy velocity for coupling should use the direct FE gradient of the
        // pressure head. The Phi-based gradients are only for stabilization.
        for (int I = 0; I < nSpace; ++I) {
          q_velocity.data()[eN_k_nSpace + I] = grad_u_velocity[I];
        }

        double pressure_gradient[nSpace];
        for (int J = 0; J < nSpace; ++J)
          pressure_gradient[J] = grad_u_velocity[J] - rho_velocity * gravity.data()[J];

        for (int I = 0; I < nSpace; ++I) {
          double acc = 0.0;
          for (int ii = a_rowptr.data()[I]; ii < a_rowptr.data()[I+1]; ++ii) {
            const int J = a_colind.data()[ii];
            acc += (a[ii] / rho_velocity) * pressure_gradient[J];
          }
          velocity.data()[eN_k_nSpace + I] = -acc;
          velocity_couple.data()[eN_k_nSpace + I] = -acc;
        }
        // static int debug_ev_prints = 0;
        // if (debug_ev_prints < 6 && eN < 2 && k < 2)
        // {
        //   std::cout << "[Richards EV fixed] eN=" << eN
        //             << " k=" << k
        //             << " u=" << u
        //             << " un=" << un
        //             << " grad_phi=(" << grad_phi[0];
        //   for (int I = 1; I < nSpace; ++I)
        //     std::cout << "," << grad_phi[I];
        //   std::cout << ") grad_u_velocity=(" << grad_u_velocity[0];
        //   for (int I = 1; I < nSpace; ++I)
        //     std::cout << "," << grad_u_velocity[I];
        //   std::cout << ") velocity_couple=(" << velocity_couple.data()[eN_k_nSpace + 0];
        //   for (int I = 1; I < nSpace; ++I)
        //     std::cout << "," << velocity_couple.data()[eN_k_nSpace + I];
        //   std::cout << ")" << std::endl;
        //   for (int j = 0; j < nDOF_trial_element; ++j)
        //   {
        //     const int u_gj = u_l2g.data()[eN_nDOF_trial_element + j];
        //     const int x_gj = mesh_l2g.data()[eN_nDOF_mesh_trial_element + j];
        //     const int free_gj = r_l2g.data()[eN_nDOF_trial_element + j];
        //     std::cout << "  [EV fixed dof] j=" << j
        //               << " mapped_u=" << u_dof.data()[u_gj]
        //               << " mapped_u_old=" << u_dof_old.data()[u_gj]
        //               << " free_material=" << freeDOFMaterialTypes.data()[free_gj]
        //               << " mapped_x=(" << mesh_dof.data()[x_gj * 3 + 0]
        //               << "," << mesh_dof.data()[x_gj * 3 + 1]
        //               << "," << mesh_dof.data()[x_gj * 3 + 2]
        //               << ") Phi=" << Phi[j]
        //               << " Phi_n=" << Phi_n[j]
        //               << std::endl;
        //   }
        //   debug_ev_prints++;
        // }
       // if (nSpace != 2) {std::cout << "WARNING nSpace=" << nSpace << std::endl;}
        //
        //moving mesh
        //
        double mesh_velocity[3];
        mesh_velocity[0] = xt;
        mesh_velocity[1] = yt;
        mesh_velocity[2] = zt;
        //relative velocity at tn
        for (int I = 0; I < nSpace; I++) {
          f[I] -= MOVING_DOMAIN * m * mesh_velocity[I];
          velocity_loc[I] = df[I] * (2.0 * dm * dm / (dm * dm + fmax(1.0e-16, dm * dm)));
        }
        //////////////////////////////
        // CALCULATE CELL BASED CFL //
        //////////////////////////////
        calculateCFL(elementDiameter.data()[eN] / degree_polynomial, velocity_loc, cfl.data()[eN_k]);
        //////////////////////////////////////////////
        // CALCULATE ENTROPY RESIDUAL AT QUAD POINT //
        //////////////////////////////////////////////
        if (STABILIZATION_TYPE == STABILIZATION::EV_Stab) // EV stab
        {
          for (int I = 0; I < nSpace; I++) aux_entropy_residual += velocity_loc[I] * grad_phi_n[I];
          DENTROPY_un = ENTROPY_TYPE == 1 ? DENTROPY(un, uL, uR) : DENTROPY_LOG(un, uL, uR);
        }
        //////////////
        // ith-LOOP //
        //////////////
        for (int i = 0; i < nDOF_test_element; i++) {
          // VECTOR OF ENTROPY RESIDUAL //
          int eN_i = eN * nDOF_test_element + i;
          ML2[r_l2g.data()[eN_i]] += u_test_dV[i];
          if (STABILIZATION_TYPE == STABILIZATION::EV_Stab) // EV stab
          {
            double uni = u_dof_old.data()[u_l2g.data()[eN_i]];
            DENTROPY_uni              = ENTROPY_TYPE == 1 ? DENTROPY(uni, uL, uR) : DENTROPY_LOG(uni, uL, uR);
            element_entropy_residual[i] += (DENTROPY_un - DENTROPY_uni) * aux_entropy_residual * u_test_dV[i];
          }

          elementResidual_u[i] += m * u_test_dV[i];
          ///////////////
          // j-th LOOP // To construct transport matrices
          ///////////////
          
          for (int j = 0; j < nDOF_trial_element; j++) {
            int j_nSpace = j * nSpace;
            int i_nSpace = i * nSpace;
            elementTransport[i][j] += ck.SimpleDiffusionJacobian_weak(a_rowptr.data(), a_colind.data(), as, &u_grad_trial[j_nSpace], &u_grad_test_dV[i_nSpace]);
            elementTransportConsistent[i][j] += ck.SimpleDiffusionJacobian_weak(a_rowptr.data(), a_colind.data(), a, &u_grad_trial[j_nSpace], &u_grad_test_dV[i_nSpace]);
            elementTransportn[i][j] += ck.SimpleDiffusionJacobian_weak(a_rowptr.data(), a_colind.data(), asn, &u_grad_trial[j_nSpace], &u_grad_test_dV[i_nSpace]);
            elementTransportConsistentn[i][j] += ck.SimpleDiffusionJacobian_weak(a_rowptr.data(), a_colind.data(), an, &u_grad_trial[j_nSpace], &u_grad_test_dV[i_nSpace]);
          }
        } //i 
        //save solution for other models
        q_u.data()[eN_k] = u;
        q_m.data()[eN_k] = m;
      }
      /////////////////
      // DISTRIBUTE // load cell based element into global residual
      ////////////////
      for (int i = 0; i < nDOF_test_element; i++) {
        int eN_i = eN * nDOF_test_element + i;
        int gi   = r_l2g.data()[eN_i];
        // distribute entropy_residual
        if (STABILIZATION_TYPE == STABILIZATION::EV_Stab) // EV Stab
          global_entropy_residual[gi] += element_entropy_residual[i];
        // distribute transport matrices
        for (int j = 0; j < nDOF_trial_element; j++) {
          int eN_i_j = eN_i * nDOF_trial_element + j;
          TransportMatrix[csrRowIndeces_CellLoops.data()[eN_i] + csrColumnOffsets_CellLoops.data()[eN_i_j]] += elementTransport[i][j];
          TransportMatrixConsistent[csrRowIndeces_CellLoops.data()[eN_i] + csrColumnOffsets_CellLoops.data()[eN_i_j]] += elementTransportConsistent[i][j];
          TransportMatrixn[csrRowIndeces_CellLoops.data()[eN_i] + csrColumnOffsets_CellLoops.data()[eN_i_j]] += elementTransportn[i][j];
          TransportMatrixConsistentn[csrRowIndeces_CellLoops.data()[eN_i] + csrColumnOffsets_CellLoops.data()[eN_i_j]] += elementTransportConsistentn[i][j];
        } //j
      } //i
      
    } //elementsxw

    // double s = 0.0, sabs = 0.0;
    // double vmin = 1e300, vmax = -1e300;
    // size_t n_nonzero = 0;

    // for (size_t i = 0; i < velocity.size(); i++) {
    //   double v = velocity.data()[i];
    //   s += v;
    //   sabs += std::abs(v);
    //   vmin = std::min(vmin, v);
    //   vmax = std::max(vmax, v);
    //   if (std::abs(v) > 1e-14) n_nonzero++;
    // }

    // std::cout << "[after compute] size=" << velocity.size()
    //           << " nnz=" << n_nonzero
    //           << " min=" << vmin << " max=" << vmax
    //           << " sumabs=" << sabs << std::endl;

        //loop over exterior element boundaries to calculate surface integrals and load into element and global residuals
    //
    //ebNE is the Exterior element boundary INdex
    //ebN is the element boundary INdex
    //eN is the element index
    for (int ebNE = 0; ebNE < nExteriorElementBoundaries_global; ebNE++) {
      int    ebN = exteriorElementBoundariesArray.data()[ebNE], eN = elementBoundaryElementsArray.data()[ebN * 2 + 0], ebN_local = elementBoundaryLocalElementBoundariesArray.data()[ebN * 2 + 0], eN_nDOF_trial_element = eN * nDOF_trial_element;
      double elementResidual_u[nDOF_test_element];
      for (int i = 0; i < nDOF_test_element; i++) { elementResidual_u[i] = 0.0; }
      for (int kb = 0; kb < nQuadraturePoints_elementBoundary; kb++) {
        int    ebNE_kb = ebNE * nQuadraturePoints_elementBoundary + kb, ebNE_kb_nSpace = ebNE_kb * nSpace, ebN_local_kb = ebN_local * nQuadraturePoints_elementBoundary + kb, ebN_local_kb_nSpace = ebN_local_kb * nSpace;
        double u_ext = 0.0, un_ext, grad_u_ext[nSpace], m_ext = 0.0, dm_ext = 0.0, f_ext[nSpace], df_ext[nSpace], a_ext[nnz], da_ext[nnz], as_ext[nnz], 
        mn_ext = 0.0, dmn_ext = 0.0, fn_ext[nSpace], dfn_ext[nSpace], an_ext[nnz], dan_ext[nnz], asn_ext[nnz], flux_ext = 0.0, bflux_ext = 0.0,
               //anb_seepage_flux=0.0, // for flux calculation
          bc_u_ext = 0.0, bc_grad_u_ext[nSpace], bc_m_ext = 0.0, bc_dm_ext = 0.0, bc_f_ext[nSpace], bc_df_ext[nSpace], bc_a_ext[nnz], bc_da_ext[nnz], bc_as_ext[nnz], jac_ext[nSpace * nSpace], jacDet_ext, jacInv_ext[nSpace * nSpace], boundaryJac[nSpace * (nSpace - 1)], metricTensor[(nSpace - 1) * (nSpace - 1)], metricTensorDetSqrt, dS, u_test_dS[nDOF_test_element], u_grad_trial_trace[nDOF_trial_element * nSpace], normal[3], x_ext, y_ext, z_ext, xt_ext, yt_ext, zt_ext, integralScaling, G[nSpace * nSpace], G_dd_G, tr_G, fluxJacobian_u_u[nDOF_trial_element], bfluxJacobian_u_u[nDOF_trial_element], fluxJacobian_un_un[nDOF_trial_element];
        //
        //calculate the solution and gradients at quadrature points
        //
        //compute information about mapping from reference element to physical element
        ck.calculateMapping_elementBoundary(eN, ebN_local, kb, ebN_local_kb, mesh_dof.data(), mesh_l2g.data(), mesh_trial_trace_ref.data(), mesh_grad_trial_trace_ref.data(), boundaryJac_ref.data(), jac_ext, jacDet_ext, jacInv_ext, boundaryJac, metricTensor, metricTensorDetSqrt,
                                            normal_ref.data(), normal, x_ext, y_ext, z_ext);
        ck.calculateMappingVelocity_elementBoundary(eN, ebN_local, kb, ebN_local_kb, mesh_velocity_dof.data(), mesh_l2g.data(), mesh_trial_trace_ref.data(), xt_ext, yt_ext, zt_ext, normal, boundaryJac, metricTensor, integralScaling);
        dS = ((1.0 - MOVING_DOMAIN) * metricTensorDetSqrt + MOVING_DOMAIN * integralScaling) * dS_ref.data()[kb];
        //get the metric tensor
        //cek todo use symmetry
        ck.calculateG(jacInv_ext, G, G_dd_G, tr_G);
        //compute shape and solution information
        //shape
        ck.gradTrialFromRef(&u_grad_trial_trace_ref.data()[ebN_local_kb_nSpace * nDOF_trial_element], jacInv_ext, u_grad_trial_trace);
        //solution and gradient
        ck.valFromDOF(u_dof.data(), &u_l2g.data()[eN_nDOF_trial_element], &u_trial_trace_ref.data()[ebN_local_kb * nDOF_test_element], u_ext);
        ck.valFromDOF(u_dof_old.data(), &u_l2g.data()[eN_nDOF_trial_element], &u_trial_trace_ref.data()[ebN_local_kb * nDOF_test_element], un_ext);
        ck.gradFromDOF(u_dof.data(), &u_l2g.data()[eN_nDOF_trial_element], u_grad_trial_trace, grad_u_ext);
        
        //populate ebqe_x
        // const int ebNE_kb_3d = ebNE_kb * 3;
        // ebqe_x.data()[ebNE_kb_3d + 0] = x_ext;
        // ebqe_x.data()[ebNE_kb_3d + 1] = y_ext;
        // ebqe_x.data()[ebNE_kb_3d + 2] = z_ext;

        
        //precalculate test function products with integration weights
        for (int j = 0; j < nDOF_trial_element; j++) { u_test_dS[j] = u_test_trace_ref.data()[ebN_local_kb * nDOF_test_element + j] * dS; }
        //
        //load the boundary values
        //
        bc_u_ext = isDOFBoundary_u.data()[ebNE_kb] * ebqe_bc_u_ext.data()[ebNE_kb] + (1 - isDOFBoundary_u.data()[ebNE_kb]) * u_ext;
        //
        //calculate the pde coefficients using the solution and the boundary values for the solution
        //
        double bc_Kr, bc_dKr,bc_Kr_ext, bc_dKr_ext, bc_Krn, bc_dKrn, thetaW_ext, thetaWn_ext, thetaW_bc_ext;
        const double rho_ext = ebqe_rho.data()[ebNE_kb];
        const double rho_velocity_ext = std::fabs(rho_ext) > 1.0e-12 ? rho_ext : rho;

        // EV exterior boundary closure.
        double dm_du_n_ext = 0.0, dkr_du_n_ext = 0.0;
        double dmn_du_n_ext = 0.0, dkrn_du_n_ext = 0.0;
        double bc_dm_du_n = 0.0, bc_dkr_du_n = 0.0;
        double df_du_n_ext[nSpace], dfn_du_n_ext[nSpace], bc_df_du_n[nSpace];
        double da_du_n_ext[nnz], dan_du_n_ext[nnz], bc_da_du_n[nnz];
        for (int I = 0; I < nSpace; I++) {
          df_du_n_ext[I] = 0.0; dfn_du_n_ext[I] = 0.0; bc_df_du_n[I] = 0.0;
        }
        for (int ii = 0; ii < nnz; ii++) {
          da_du_n_ext[ii] = 0.0; dan_du_n_ext[ii] = 0.0; bc_da_du_n[ii] = 0.0;
        }
        double u_n_ext_qp = 0.0, u_n_ext_qp_old = 0.0;
        ck.valFromDOF(u_dof_n.data(), &u_l2g.data()[eN_nDOF_trial_element],
                      &u_trial_trace_ref.data()[ebN_local_kb * nDOF_test_element], u_n_ext_qp);
        ck.valFromDOF(u_dof_n_old.data(), &u_l2g.data()[eN_nDOF_trial_element],
                      &u_trial_trace_ref.data()[ebN_local_kb * nDOF_test_element], u_n_ext_qp_old);
        const double bc_u_n_ext_qp = isDOFBoundary_n.data()[ebNE_kb] * ebqe_bc_u_n_ext.data()[ebNE_kb]
                                   + (1 - isDOFBoundary_n.data()[ebNE_kb]) * u_n_ext_qp;
        evaluateCoefficients_from_Se(a_rowptr.data(), a_colind.data(), rho, rho_ext, beta, gravity.data(),
                                     alpha.data()[elementMaterialTypes.data()[eN]], n.data()[elementMaterialTypes.data()[eN]],
                                     thetaR.data()[elementMaterialTypes.data()[eN]], thetaSR.data()[elementMaterialTypes.data()[eN]],
                                     &KWs.data()[elementMaterialTypes.data()[eN] * nnz], u_ext, u_n_ext_qp,
                                     m_ext, dm_ext, dm_du_n_ext, f_ext, df_ext, df_du_n_ext, a_ext, da_ext, da_du_n_ext,
                                     as_ext, bc_Kr, bc_dKr, dkr_du_n_ext, thetaW_ext);
        evaluateCoefficients_from_Se(a_rowptr.data(), a_colind.data(), rho, rho_ext, beta, gravity.data(),
                                     alpha.data()[elementMaterialTypes.data()[eN]], n.data()[elementMaterialTypes.data()[eN]],
                                     thetaR.data()[elementMaterialTypes.data()[eN]], thetaSR.data()[elementMaterialTypes.data()[eN]],
                                     &KWs.data()[elementMaterialTypes.data()[eN] * nnz], un_ext, u_n_ext_qp_old,
                                     mn_ext, dmn_ext, dmn_du_n_ext, fn_ext, dfn_ext, dfn_du_n_ext, an_ext, dan_ext, dan_du_n_ext,
                                     asn_ext, bc_Krn, bc_dKrn, dkrn_du_n_ext, thetaWn_ext);
        evaluateCoefficients_from_Se(a_rowptr.data(), a_colind.data(), rho, rho_ext, beta, gravity.data(),
                                     alpha.data()[elementMaterialTypes.data()[eN]], n.data()[elementMaterialTypes.data()[eN]],
                                     thetaR.data()[elementMaterialTypes.data()[eN]], thetaSR.data()[elementMaterialTypes.data()[eN]],
                                     &KWs.data()[elementMaterialTypes.data()[eN] * nnz], bc_u_ext, bc_u_n_ext_qp,
                                     bc_m_ext, bc_dm_ext, bc_dm_du_n, bc_f_ext, bc_df_ext, bc_df_du_n, bc_a_ext, bc_da_ext, bc_da_du_n,
                                     bc_as_ext, bc_Kr_ext, bc_dKr_ext, bc_dkr_du_n, thetaW_bc_ext);
        ebqe_theta.data()[ebNE_kb] = thetaW_ext;
        
        //
        //Calculate Darcy Velocity at external faces
        //
        
        // double  darcy_velocity_loc_ext[nSpace];
        // for (int I = 0; I < nSpace; I++) { darcy_velocity_loc_ext[I] = 0.0; }
        
        // for (int I = 0; I < nSpace; I++) {
        //   for (int J = 0; J < nSpace; J++) { darcy_velocity_loc_ext[I] -= bc_Kr * KWs.data()[elementMaterialTypes[eN] * nSpace * nSpace + I * nSpace + J] * (grad_u_ext[J]+  gravity.data()[J]); }
        // }
        // for (int I = 0; I < nSpace; I++) { ebqe_velocity_ext_couple.data()[ebNE_kb_nSpace + I] = darcy_velocity_loc_ext[I] ; }
        
        double ext_pressure_gradient[nSpace];
        for (int J = 0; J < nSpace; ++J)
          ext_pressure_gradient[J] = grad_u_ext[J] - rho_velocity_ext * gravity.data()[J];

        for (int I = 0; I < nSpace; ++I) {
          double acc = 0.0;
          for (int ii = a_rowptr.data()[I]; ii < a_rowptr.data()[I+1]; ++ii) {
            const int J = a_colind.data()[ii];
            acc += (a_ext[ii] / rho_velocity_ext) * ext_pressure_gradient[J];
          }
          ebqe_velocity_ext.data()[ebNE_kb_nSpace + I] = -acc;
          ebqe_velocity_ext_couple.data()[ebNE_kb_nSpace + I] = -acc;
        }
        bool useConsistentFlux=false;
        double grad_u_n_ext_b[nSpace];
        ck.gradFromDOF(u_dof_n.data(), &u_l2g.data()[eN_nDOF_trial_element],
                       u_grad_trial_trace, grad_u_n_ext_b);
        const int    mat_b0  = elementMaterialTypes.data()[eN];
        const double alpha_b0 = alpha.data()[mat_b0];
        const double n_vg_b0  = n.data()[mat_b0];
        const double krn_end_b0 = krn_end.data()[mat_b0];
        const double *KWs_b0  = &KWs.data()[mat_b0 * nnz];
        const double phi_b0   = thetaR.data()[mat_b0] + thetaSR.data()[mat_b0];
        const double S_wr_b0  = thetaR.data()[mat_b0] / phi_b0;
        const double one_m_Sr_b0 = 1.0 - S_wr_b0;
        const double Se_trap_L3616 = 1.0 - S_gr.data()[mat_b0] / one_m_Sr_b0;  // gas-only residual trapping
        const double p_clb0   = fmax(u_ext, 1.0e2);
        ::m_comp_co2::flash::FlashState fsb0 =
            ::m_comp_co2::flash::flashPZ(p_clb0, u_n_ext_qp, T_C_member, m_NaCl_member, ::m_comp_co2::flash::EPS_Z, immiscible_member);
        double Sgb0 = fsb0.S_g, dSgb0_dp = fsb0.dS_g_dp, dSgb0_dz = fsb0.dS_g_dz;
        // Per-rock closure: with vertex (Lobatto) boundary quadrature this point is
        // a face vertex; at an interface node read this element's rock's S_r.
        if (per_rock) {
          int a_v = -1;
          for (int a = 0; a < nDOF_trial_element; a++)
            if (u_trial_trace_ref.data()[ebN_local_kb * nDOF_test_element + a] > 1.0 - 1.0e-12) a_v = a;
          if (a_v < 0)
            throw std::runtime_error("m_comp_co2: per_rock_closure needs vertex (Lobatto) boundary quadrature");
          const int gv = u_l2g.data()[eN_nDOF_trial_element + a_v];
          if (node_multi[gv]) {
            const int o = gv * nRock + mat_b0;
            Sgb0 = rS[o];  dSgb0_dp = rS_dp[o];  dSgb0_dz = rS_dz[o];
          }
        }
        const double Sab0 = 1.0 - Sgb0;
        const double Se_rawb0 = (Sab0 - S_wr_b0)/one_m_Sr_b0;
        double Se_ab0, dSeb0_dp, dSeb0_dz;
        if (Se_rawb0<=0.0){Se_ab0=0.0;dSeb0_dp=0.0;dSeb0_dz=0.0;}
        else if (Se_rawb0>=1.0){Se_ab0=1.0;dSeb0_dp=0.0;dSeb0_dz=0.0;}
        else {Se_ab0=Se_rawb0;dSeb0_dp=-dSgb0_dp/one_m_Sr_b0;dSeb0_dz=-dSgb0_dz/one_m_Sr_b0;}
        double KWrb0=0,DKWrb0=0,thWb0=0,DthWb0=0,KNrb0=0,DKNrb0=0,pcb0=0,dpc_dSeb0=0,d2pcb0=0;
        if (PSK_TYPE_member == 1) {
          proteus::m_comp_co2::psk::bc_wetting_from_Se(Se_ab0, alpha_b0, n_vg_b0, thetaR.data()[mat_b0], thetaSR.data()[mat_b0], thWb0, DthWb0, KWrb0, DKWrb0);
          proteus::m_comp_co2::psk::bc_kr_nonwetting_from_Se(Se_ab0, alpha_b0, n_vg_b0, KNrb0, DKNrb0, Se_trap_L3616);
          proteus::m_comp_co2::psk::bc_pc_from_Se(Se_ab0, alpha_b0, n_vg_b0, bc_Se_min.data()[mat_b0], pcb0, dpc_dSeb0, d2pcb0);
        } else {
          proteus::m_comp_co2::psk::vgm_wetting_from_Se(Se_ab0, alpha_b0, n_vg_b0, thetaR.data()[mat_b0], thetaSR.data()[mat_b0], thWb0, DthWb0, KWrb0, DKWrb0);
          proteus::m_comp_co2::psk::vgm_kr_nonwetting_from_Se(Se_ab0, alpha_b0, n_vg_b0, KNrb0, DKNrb0, Se_trap_L3616);
          proteus::m_comp_co2::psk::vgm_pc_from_Se(Se_ab0, alpha_b0, n_vg_b0, pcb0, dpc_dSeb0, d2pcb0);
        }
        KNrb0 *= krn_end_b0; DKNrb0 *= krn_end_b0;
        const double pcpb0 = dpc_dSeb0/one_m_Sr_b0;
        const double dpcpb0_dp = (d2pcb0/one_m_Sr_b0)*dSeb0_dp;
        const double dpcpb0_dz = (d2pcb0/one_m_Sr_b0)*dSeb0_dz;
        const double dMmb0 = ::m_comp_co2::eos::M_CO2_KG - ::m_comp_co2::eos::M_H2O_KG;
        const double Mbar_gb0 = fsb0.Y*::m_comp_co2::eos::M_CO2_KG + (1.0-fsb0.Y)*::m_comp_co2::eos::M_H2O_KG;
        const double Mbar_ab0 = fsb0.X*::m_comp_co2::eos::M_CO2_KG + (1.0-fsb0.X)*::m_comp_co2::eos::M_H2O_KG;
        const double rgmb0 = fsb0.rho_g*Mbar_gb0, ramb0 = fsb0.rho_a*Mbar_ab0;
        const double drgmb0_dp = fsb0.drho_g_dp*Mbar_gb0 + fsb0.rho_g*fsb0.dY_dp*dMmb0;
        const double drgmb0_dz =                            fsb0.rho_g*fsb0.dY_dz*dMmb0;
        const double dramb0_dp = fsb0.drho_a_dp*Mbar_ab0 + fsb0.rho_a*fsb0.dX_dp*dMmb0;
        const double dramb0_dz = fsb0.drho_a_dz*Mbar_ab0 + fsb0.rho_a*fsb0.dX_dz*dMmb0;
        const double Agb0 = fsb0.rho_g*(1.0-fsb0.Y), Aab0 = fsb0.rho_a*(1.0-fsb0.X);
        const double dAgb0_dp = fsb0.drho_g_dp*(1.0-fsb0.Y) - fsb0.rho_g*fsb0.dY_dp;
        const double dAgb0_dz =                              - fsb0.rho_g*fsb0.dY_dz;
        const double dAab0_dp = fsb0.drho_a_dp*(1.0-fsb0.X) - fsb0.rho_a*fsb0.dX_dp;
        const double dAab0_dz = fsb0.drho_a_dz*(1.0-fsb0.X) - fsb0.rho_a*fsb0.dX_dz;
        double F0n_b = 0.0;
        double valp_b[nSpace], valz_b[nSpace];
        for (int I=0;I<nSpace;I++){
          double ugI=0,uaI=0,dugp=0,dugz=0,duap=0,duaz=0;
          for (int ii=a_rowptr.data()[I];ii<a_rowptr.data()[I+1];ii++){
            const int J=a_colind.data()[ii];
            const double Kii=KWs_b0[ii];
            const double Mob_g=KNrb0*Kii/mu_n, Mob_a=KWrb0*Kii;
            const double dMobg_dp=(DKNrb0*Kii/mu_n)*dSeb0_dp, dMobg_dz=(DKNrb0*Kii/mu_n)*dSeb0_dz;
            const double dMoba_dp=(DKWrb0*Kii)*dSeb0_dp, dMoba_dz=(DKWrb0*Kii)*dSeb0_dz;
            const double gJ=gravity.data()[J];
            const double gradSa=-(dSgb0_dp*grad_u_ext[J]+dSgb0_dz*grad_u_n_ext_b[J]);
            const double gp_a=grad_u_ext[J]-ramb0*gJ;
            const double gp_g=grad_u_ext[J]+pcpb0*gradSa-rgmb0*gJ;
            ugI-=Mob_g*gp_g; uaI-=Mob_a*gp_a;
            const double dgradSa_dp=-(fsb0.d2S_g_dp2*grad_u_ext[J]+fsb0.d2S_g_dpdz*grad_u_n_ext_b[J]);
            const double dgradSa_dz=-(fsb0.d2S_g_dpdz*grad_u_ext[J]+fsb0.d2S_g_dz2*grad_u_n_ext_b[J]);
            const double dgpg_dp=dpcpb0_dp*gradSa+pcpb0*dgradSa_dp-drgmb0_dp*gJ;
            const double dgpg_dz=dpcpb0_dz*gradSa+pcpb0*dgradSa_dz-drgmb0_dz*gJ;
            dugp-=dMobg_dp*gp_g+Mob_g*dgpg_dp;
            dugz-=dMobg_dz*gp_g+Mob_g*dgpg_dz;
            duap-=dMoba_dp*gp_a+Mob_a*(-dramb0_dp*gJ);
            duaz-=dMoba_dz*gp_a+Mob_a*(-dramb0_dz*gJ);
          }
          F0n_b += (Agb0*ugI + Aab0*uaI) * normal[I];
          valp_b[I] = (dAgb0_dp*ugI + Agb0*dugp + dAab0_dp*uaI + Aab0*duap);
          valz_b[I] = (dAgb0_dz*ugI + Agb0*dugz + dAab0_dz*uaI + Aab0*duaz);
        }
        const double penb0 = ebqe_penalty_ext.data()[ebNE_kb];
        (void)valz_b;
        {
          const int isSeep = isSeepageFace.data()[ebNE];
          if (isSeep || isDOFBoundary_u.data()[ebNE_kb]) {
            const double bc_u_pen = isSeep ? 0.0 : bc_u_ext;
            const double pen_term = penb0*(u_ext - bc_u_pen);
            flux_ext  = F0n_b + pen_term;     // consistent -> TransportMatrix*
            bflux_ext = pen_term;             // penalty     -> globalResidual
            if (isSeep && flux_ext <= 0.0) { flux_ext = 0.0; bflux_ext = 0.0; }
          } else {
            flux_ext  = ebqe_bc_flux_ext[ebNE_kb];
            bflux_ext = ebqe_bc_flux_ext[ebNE_kb];
          }
        }

        ebqe_flux.data()[ebNE_kb] = flux_ext;

        anb_seepage_flux             = seepagefluxcalculator(anb_seepage_flux, isSeepageFace.data()[ebNE], dS, flux_ext);
        anb_seepage_flux_n.data()[0] = anb_seepage_flux;
        ebqe_u.data()[ebNE_kb]       = u_ext;
        //
        //update residuals
        //
        for (int i = 0; i < nDOF_test_element; i++) {
          if (useConsistentFlux) {
            elementResidual_u[i] += ck.ExteriorElementBoundaryFlux(flux_ext, u_test_dS[i]);
          } else {
            elementResidual_u[i] += ck.ExteriorElementBoundaryFlux(bflux_ext, u_test_dS[i]);
          }
        } //i
        for (int j = 0; j < nDOF_trial_element; j++) {
          if (useConsistentFlux) {
          exteriorNumericalFluxJacobian(a_rowptr.data(), a_colind.data(), isDOFBoundary_u.data()[ebNE_kb], normal, a_ext, da_ext, grad_u_ext, &u_grad_trial_trace[j * nSpace], df_ext, u_trial_trace_ref.data()[ebN_local_kb * nDOF_test_element + j],
                                        ebqe_penalty_ext.data()[ebNE_kb], //penalty,
                                        fluxJacobian_u_u[j]);
          } else {
            // P3c boundary (STAB=2): compositional comp-0 flux Jacobian.
            // Consistent d(F_0.n)/d(p_j) -> fluxJacobian_u_u (TransportMatrix*);
            // penalty -> bfluxJacobian_u_u (globalJacobian).  value/grad blocks
            // reuse valp_b + the stored flash/psk state (FD-verified, b0test).
            // (0,1) cross-block not assembled here (matches the original).
            if (isDOFBoundary_u.data()[ebNE_kb]) {
              const double trial_j = u_trial_trace_ref.data()[ebN_local_kb * nDOF_test_element + j];
              double accp = 0.0;
              for (int I=0;I<nSpace;I++){
                accp += normal[I]*valp_b[I]*trial_j;
                for (int ii=a_rowptr.data()[I];ii<a_rowptr.data()[I+1];ii++){
                  const int J=a_colind.data()[ii];
                  const double Kii=KWs_b0[ii];
                  const double Mob_g=KNrb0*Kii/mu_n, Mob_a=KWrb0*Kii;
                  const double dFdgp = Agb0*(-Mob_g*(1.0 - pcpb0*dSgb0_dp)) + Aab0*(-Mob_a);
                  accp += normal[I]*dFdgp*u_grad_trial_trace[j*nSpace + J];
                }
              }
              fluxJacobian_u_u[j]  = accp;            // consistent (no penalty)
              bfluxJacobian_u_u[j] = penb0*trial_j;   // penalty
            } else {
              fluxJacobian_u_u[j]  = 0.0;
              bfluxJacobian_u_u[j] = 0.0;
            }
          }
          exteriorNumericalFluxJacobian(a_rowptr.data(), a_colind.data(),
              isDOFBoundary_u.data()[ebNE_kb], normal,
              asn_ext, dan_ext, grad_u_ext,
              &u_grad_trial_trace[j * nSpace], dfn_ext,
              u_trial_trace_ref.data()[ebN_local_kb * nDOF_test_element + j],
              ebqe_penalty_ext.data()[ebNE_kb],
              fluxJacobian_un_un[j]);
        } //j
        //
        //update the element and global residual storage
        //
        for (int i = 0; i < nDOF_test_element; i++) {
          int eN_i = eN * nDOF_test_element + i;
          for (int j = 0; j < nDOF_trial_element; j++) {
            int ebN_i_j = ebN * 4 * nDOF_test_X_trial_element + i * nDOF_trial_element + j;
            if (useConsistentFlux) {
              globalJacobian.data()[csrRowIndeces_u_u[eN_i] + csrColumnOffsets_eb_u_u[ebN_i_j]] += fluxJacobian_u_u[j] * u_test_dS[i];
            } else {
              globalJacobian.data()[csrRowIndeces_u_u[eN_i] + csrColumnOffsets_eb_u_u[ebN_i_j]] += bfluxJacobian_u_u[j] * u_test_dS[i];
              TransportMatrix[csrRowIndeces_u_u[eN_i] + csrColumnOffsets_eb_u_u[ebN_i_j]] += fluxJacobian_u_u[j] * u_test_dS[i];
              TransportMatrixConsistent[csrRowIndeces_u_u[eN_i] + csrColumnOffsets_eb_u_u[ebN_i_j]] += fluxJacobian_u_u[j] * u_test_dS[i];
              TransportMatrixn[csrRowIndeces_u_u[eN_i] + csrColumnOffsets_eb_u_u[ebN_i_j]] += fluxJacobian_un_un[j] * u_test_dS[i];
              TransportMatrixConsistentn[csrRowIndeces_u_u[eN_i] + csrColumnOffsets_eb_u_u[ebN_i_j]] += fluxJacobian_un_un[j] * u_test_dS[i];
            }
          } //j
        } //i
      } //kb
      for (int i = 0; i < nDOF_test_element; i++) {
          int eN_i = eN * nDOF_test_element + i;
          globalResidual.data()[offset_u + stride_u * u_l2g.data()[eN_i]] += elementResidual_u[i];
      }//i
    } //ebNE
    /////////////////////////////////////////////////////////////////
    // COMPUTE SMOOTHNESS INDICATOR and NORMALIZE ENTROPY RESIDUAL //
    /////////////////////////////////////////////////////////////////
    // NOTE: see NCLS.h for a different but equivalent implementation of this.
    std::vector<double> cflux(numDOFs, 0.0);
    for (int i = 0; i < numDOFs_u; i++) {
      double gi[nSpace], Cij[nSpace], xi[nSpace], etaMaxi, etaMini;
      const int node_i = freeDOFToNode_u.data()[i];
      double solni = u_free_dof_old[i];
      for (int I = 0; I < nSpace; I++) {
        solni -= rho_dof[i] * gravity.data()[I] * mesh_dof.data()[node_i * 3 + I];
      }
      if (STABILIZATION_TYPE == STABILIZATION::EV_Stab) //EV Stabilization
      {
        // For eta min and max
        etaMaxi = fabs(eta[i]);
        etaMini = fabs(eta[i]);
      }
      // initialize gi and compute xi
      for (int I = 0; I < nSpace; I++) {
        gi[I] = 0.;
        xi[I] = mesh_dof.data()[node_i * 3 + I];
      }
      // for smoothness indicator //
      double alpha_numerator_pos = 0., alpha_numerator_neg = 0., alpha_denominator_pos = 0., alpha_denominator_neg = 0.;
      for (int offset = csrRowIndeces_DofLoops.data()[i]; offset < csrRowIndeces_DofLoops.data()[i + 1]; offset++) { // First loop in j (sparsity pattern)
        int j = csrColumnOffsets_DofLoops.data()[offset];
        const int full_offset = full_offset_from_compact(i, j);
        assert(full_offset >= 0);
        const int node_j = freeDOFToNode_u.data()[j];
        if (STABILIZATION_TYPE == STABILIZATION::EV_Stab) //EV Stabilization
        {
          // COMPUTE ETA MIN AND ETA MAX //
          etaMaxi = fmax(etaMaxi, fabs(eta[j]));
          etaMini = fmin(etaMini, fabs(eta[j]));
        }
        double solnj = u_free_dof_old[j];
        for (int I = 0; I < nSpace; I++) {
          solnj -= rho_dof[j] * gravity.data()[I] * mesh_dof.data()[node_j * 3 + I];
        }
        // Update Cij matrices
        Cij[0] = Cx[full_offset];
#if nSpace == 2
        Cij[1] = Cy[full_offset];
#endif
#if nSpace == 3
        Cij[2] = Cz[full_offset];
#endif
        // COMPUTE gi VECTOR. gi=1/mi*sum_j(Cij*solj)
        for (int I = 0; I < nSpace; I++) gi[I] += Cij[I] * solnj;

        // COMPUTE numerator and denominator of smoothness indicator
        double alpha_num = solni - solnj;
        if (alpha_num >= 0.) {
          alpha_numerator_pos += alpha_num;
          alpha_denominator_pos += alpha_num;
        } else {
          alpha_numerator_neg += alpha_num;
          alpha_denominator_neg += fabs(alpha_num);
        }
      }
      // scale g vector by lumped mass matrix
      //double mass_matrix_error = abs(ML.data()[i] - ML2[i]);
      //if (mass_matrix_error > 1.0e-16) std::cout << mass_matrix_error<<" ML " << ML.data()[i] << '\t' << ML2[i] << std::endl;
      for (int I = 0; I < nSpace; I++) gi[I] /= ML.data()[i];
      if (STABILIZATION_TYPE == STABILIZATION::EV_Stab) //EV Stab
      {
        // Normalizae entropy residual
        global_entropy_residual[i] *= etaMini == etaMaxi ? 0. : 2 * cE / (etaMaxi - etaMini);
        quantDOFs.data()[i] = fabs(global_entropy_residual[i]);
      }

      // Now that I have the gi vectors, I can use them for the current i-th DOF
      double SumPos = 0., SumNeg = 0.;
      for (int offset = csrRowIndeces_DofLoops.data()[i]; offset < csrRowIndeces_DofLoops.data()[i + 1]; offset++) { // second loop in j (sparsity pattern)
        int j = csrColumnOffsets_DofLoops.data()[offset];
        const int full_offset = full_offset_from_compact(i, j);
        assert(full_offset >= 0);
        // compute gi*(xi-xj)
        double gi_times_x = 0.;
        for (int I = 0; I < nSpace; I++) {
          gi_times_x += gi[I] * delta_x_ij.data()[full_offset * 3 + I];
        }
        // compute the positive and negative part of gi*(xi-xj)
        SumPos += gi_times_x > 0 ? gi_times_x : 0;
        SumNeg += gi_times_x < 0 ? gi_times_x : 0;
      }
      double sigmaPosi  = fmin(1., (fabs(SumNeg) + 1E-15) / (SumPos + 1E-15));
      double sigmaNegi  = fmin(1., (SumPos + 1E-15) / (fabs(SumNeg) + 1E-15));
      double alpha_numi = fabs(sigmaPosi * alpha_numerator_pos + sigmaNegi * alpha_numerator_neg);
      double alpha_deni = sigmaPosi * alpha_denominator_pos + sigmaNegi * alpha_denominator_neg;
      if (IS_BETAij_ONE == 1) {
        alpha_numi = fabs(alpha_numerator_pos + alpha_numerator_neg);
        alpha_deni = alpha_denominator_pos + alpha_denominator_neg;
      }
      double alphai       = alpha_numi / (alpha_deni + 1E-15);
      quantDOFs.data()[i] = alphai;

      if (POWER_SMOOTHNESS_INDICATOR == 0) psi[i] = 1.0;
      else psi[i] = std::pow(alphai, POWER_SMOOTHNESS_INDICATOR); //NOTE: they use alpha^2 in the paper
    }
    std::vector<double> w_lam_g(numDOFs_u, 0.0),  w_lam_a(numDOFs_u, 0.0);
    std::vector<double> w_dlam_g_dp(numDOFs_u, 0.0), w_dlam_g_dz(numDOFs_u, 0.0);
    std::vector<double> w_dlam_a_dp(numDOFs_u, 0.0), w_dlam_a_dz(numDOFs_u, 0.0);
    std::vector<double> w_pc(numDOFs_u, 0.0), w_dpc_dp(numDOFs_u, 0.0), w_dpc_dz(numDOFs_u, 0.0);
    std::vector<double> w_rgm(numDOFs_u, 0.0), w_ram(numDOFs_u, 0.0);
    std::vector<double> w_drgm_dp(numDOFs_u, 0.0), w_drgm_dz(numDOFs_u, 0.0);
    std::vector<double> w_dram_dp(numDOFs_u, 0.0), w_dram_dz(numDOFs_u, 0.0);
    std::vector<double> w_lam_g_old(numDOFs_u, 0.0), w_lam_a_old(numDOFs_u, 0.0);
    std::vector<double> w_pc_old(numDOFs_u, 0.0);
    std::vector<double> w_rgm_old(numDOFs_u, 0.0), w_ram_old(numDOFs_u, 0.0);
    {
      const double dMm_w = ::m_comp_co2::eos::M_CO2_KG - ::m_comp_co2::eos::M_H2O_KG;
      for (int i = 0; i < numDOFs_u; i++) {
        const int    node_i   = freeDOFToNode_u.data()[i];
        const int    mat_i   = freeDOFMaterialTypes.data()[i];
        const double alpha_i  = alpha.data()[mat_i];
        const double n_vg_i   = n.data()[mat_i];
        const double krn_end_i= krn_end.data()[mat_i];
        const double phi_i    = thetaR.data()[mat_i] + thetaSR.data()[mat_i];
        const double S_wr_i   = thetaR.data()[mat_i] / phi_i;
        const double one_m_Sr_i = 1.0 - S_wr_i;
        const double Se_trap_L3923 = 1.0 - S_gr.data()[mat_i] / one_m_Sr_i;  // gas-only residual trapping
        const double cg_i     = krn_end_i / mu_n;
        // --- current iterate ---
        const double p_cl = fmax(u_free_dof[i], 1.0e2);
        ::m_comp_co2::flash::FlashState f =
            ::m_comp_co2::flash::flashPZ(p_cl, u_dof_n.data()[node_i], T_C_member, m_NaCl_member, ::m_comp_co2::flash::EPS_Z, immiscible_member);
        const double Sa = 1.0 - f.S_g;
        const double Se_raw = (Sa - S_wr_i) / one_m_Sr_i;
        double Se, dSe_dp, dSe_dz;
        if (Se_raw <= 0.0)      { Se = 0.0; dSe_dp = 0.0; dSe_dz = 0.0; }
        else if (Se_raw >= 1.0) { Se = 1.0; dSe_dp = 0.0; dSe_dz = 0.0; }
        else { Se = Se_raw; dSe_dp = -f.dS_g_dp/one_m_Sr_i; dSe_dz = -f.dS_g_dz/one_m_Sr_i; }
        double krn=0,dkrn=0,krw=0,dkrw=0,thW=0,DthW=0,pc=0,dpc_dSe=0,d2pc=0;
        if (PSK_TYPE_member == 1) {
          proteus::m_comp_co2::psk::bc_kr_nonwetting_from_Se(Se, alpha_i, n_vg_i, krn, dkrn, Se_trap_L3923);
          proteus::m_comp_co2::psk::bc_wetting_from_Se(Se, alpha_i, n_vg_i, thetaR.data()[mat_i], thetaSR.data()[mat_i], thW, DthW, krw, dkrw);
          proteus::m_comp_co2::psk::bc_pc_from_Se(Se, alpha_i, n_vg_i, bc_Se_min.data()[mat_i], pc, dpc_dSe, d2pc);
        } else {
          proteus::m_comp_co2::psk::vgm_kr_nonwetting_from_Se(Se, alpha_i, n_vg_i, krn, dkrn, Se_trap_L3923);
          proteus::m_comp_co2::psk::vgm_wetting_from_Se(Se, alpha_i, n_vg_i, thetaR.data()[mat_i], thetaSR.data()[mat_i], thW, DthW, krw, dkrw);
          proteus::m_comp_co2::psk::vgm_pc_from_Se(Se, alpha_i, n_vg_i, pc, dpc_dSe, d2pc);
        }
        w_pc[i] = pc;  w_dpc_dp[i] = dpc_dSe*dSe_dp;  w_dpc_dz[i] = dpc_dSe*dSe_dz;
        const double Mg = f.Y*::m_comp_co2::eos::M_CO2_KG + (1.0-f.Y)*::m_comp_co2::eos::M_H2O_KG;
        const double Ma = f.X*::m_comp_co2::eos::M_CO2_KG + (1.0-f.X)*::m_comp_co2::eos::M_H2O_KG;
        w_rgm[i] = f.rho_g*Mg;  w_ram[i] = f.rho_a*Ma;
        w_drgm_dp[i] = f.drho_g_dp*Mg + f.rho_g*f.dY_dp*dMm_w;  w_drgm_dz[i] = f.rho_g*f.dY_dz*dMm_w;
        w_dram_dp[i] = f.drho_a_dp*Ma + f.rho_a*f.dX_dp*dMm_w;  w_dram_dz[i] = f.drho_a_dz*Ma + f.rho_a*f.dX_dz*dMm_w;
        // H2O molar mobilities: gas weight (1-Y), aqueous weight (1-X).
        const double yw = 1.0 - f.Y, dyw_dp = -f.dY_dp, dyw_dz = -f.dY_dz;
        const double xw = 1.0 - f.X, dxw_dp = -f.dX_dp, dxw_dz = -f.dX_dz;
        w_lam_g[i]     = cg_i*f.rho_g*yw*krn;
        w_dlam_g_dp[i] = cg_i*(f.drho_g_dp*yw*krn + f.rho_g*dyw_dp*krn + f.rho_g*yw*dkrn*dSe_dp);
        w_dlam_g_dz[i] = cg_i*(                      f.rho_g*dyw_dz*krn + f.rho_g*yw*dkrn*dSe_dz);
        w_lam_a[i]     = f.rho_a*xw*krw;
        w_dlam_a_dp[i] = f.drho_a_dp*xw*krw + f.rho_a*dxw_dp*krw + f.rho_a*xw*dkrw*dSe_dp;
        w_dlam_a_dz[i] = f.drho_a_dz*xw*krw + f.rho_a*dxw_dz*krw + f.rho_a*xw*dkrw*dSe_dz;
        // --- old time level (frozen; feeds the (1-Theta) part, no derivatives) ---
        const double p_cl_o = fmax(u_free_dof_old[i], 1.0e2);
        ::m_comp_co2::flash::FlashState fo =
            ::m_comp_co2::flash::flashPZ(p_cl_o, u_dof_n_old.data()[node_i], T_C_member, m_NaCl_member, ::m_comp_co2::flash::EPS_Z, immiscible_member);
        const double Sa_o = 1.0 - fo.S_g;
        const double Se_o_raw = (Sa_o - S_wr_i) / one_m_Sr_i;
        const double Se_o = Se_o_raw <= 0.0 ? 0.0 : (Se_o_raw >= 1.0 ? 1.0 : Se_o_raw);
        double krn_o=0,dkrn_o=0,krw_o=0,dkrw_o=0,thW_o=0,DthW_o=0,pc_o=0,dpc_o=0,d2pc_o=0;
        if (PSK_TYPE_member == 1) {
          proteus::m_comp_co2::psk::bc_kr_nonwetting_from_Se(Se_o, alpha_i, n_vg_i, krn_o, dkrn_o, Se_trap_L3923);
          proteus::m_comp_co2::psk::bc_wetting_from_Se(Se_o, alpha_i, n_vg_i, thetaR.data()[mat_i], thetaSR.data()[mat_i], thW_o, DthW_o, krw_o, dkrw_o);
          proteus::m_comp_co2::psk::bc_pc_from_Se(Se_o, alpha_i, n_vg_i, bc_Se_min.data()[mat_i], pc_o, dpc_o, d2pc_o);
        } else {
          proteus::m_comp_co2::psk::vgm_kr_nonwetting_from_Se(Se_o, alpha_i, n_vg_i, krn_o, dkrn_o, Se_trap_L3923);
          proteus::m_comp_co2::psk::vgm_wetting_from_Se(Se_o, alpha_i, n_vg_i, thetaR.data()[mat_i], thetaSR.data()[mat_i], thW_o, DthW_o, krw_o, dkrw_o);
          proteus::m_comp_co2::psk::vgm_pc_from_Se(Se_o, alpha_i, n_vg_i, pc_o, dpc_o, d2pc_o);
        }
        w_pc_old[i] = pc_o;
        const double Mg_o = fo.Y*::m_comp_co2::eos::M_CO2_KG + (1.0-fo.Y)*::m_comp_co2::eos::M_H2O_KG;
        const double Ma_o = fo.X*::m_comp_co2::eos::M_CO2_KG + (1.0-fo.X)*::m_comp_co2::eos::M_H2O_KG;
        w_rgm_old[i] = fo.rho_g*Mg_o;  w_ram_old[i] = fo.rho_a*Ma_o;
        w_lam_g_old[i] = cg_i*fo.rho_g*(1.0-fo.Y)*krn_o;
        w_lam_a_old[i] = fo.rho_a*(1.0-fo.X)*krw_o;
      }
    }
    /////////////////////////////////////////////
    // ** LOOP IN DOFs FOR EDGE BASED TERMS ** //
    /////////////////////////////////////////////
    // only iterate component-0 DOFs here. Component 1
    // (trivial gas eq) is assembled by the dedicated element loop appended
    // at the end of this function.
    for (int i = 0; i < numDOFs_u; i++) {
      int    ii = -1;
      double sum_abs_dt_times_fH_minus_fL = 0.0, MLi = ML.data()[i];
      double Kr, dKr, Krn, dKrn;
      double J_ii = 0.0;
      double ith_dissipative_term           = 0;
      double ith_low_order_dissipative_term = 0;
      double ith_flux_term                  = 0;
      double ith_consistent_flux_term       = 0;
      double dLii                           = 0.;
      double m, dm, f[nSpace], df[nSpace], a[nnz], da[nnz], as[nnz];
      double dmn, fn[nSpace], dfn[nSpace], an[nnz], dan[nnz], asn[nnz];
      // Cross-derivatives used by the (0,1) coupling wired into the low-order
      // wetting operator below: dm/du_n is added to the lumped-mass coupling
      // (row_w, col_n_i), and dkr/du_n[upwind] is added to the Kuzmin upwind
      // term at (row_w, col_n_upwind). The 'n_old' variants describe the
      // frozen time history -- they don't contribute to the Jacobian.
      double dm_du_n_fct, dkr_du_n_fct, df_du_n_fct[nSpace], da_du_n_fct[nnz];
      double dmn_du_n_fct, dkrn_du_n_fct, dfn_du_n_fct[nSpace], dan_du_n_fct[nnz];

      const double rho_i = rho_dof[i];
      const int node_i = freeDOFToNode_u.data()[i];

      double thetaW_tmp = 0.0;
      // loop over the sparsity pattern of the i-th DOF
      for (int offset = csrRowIndeces_DofLoops.data()[i]; offset < csrRowIndeces_DofLoops.data()[i + 1]; offset++) {
        int j = csrColumnOffsets_DofLoops.data()[offset];
        const int full_offset = full_offset_from_compact(i, j);
        assert(full_offset >= 0);
        if (i == j) ii = full_offset;
        const double rho_j = rho_dof[j];
        const int node_j = freeDOFToNode_u.data()[j];
        const double rho_edge = 0.5 * (rho_i + rho_j);
        // ===== P3c: compositional H2O edge flux  F_0 = F_g^w + F_a^w =====
        // Replaces the single-phase Kuzmin wetting flux fL = krw*tau*delta_phi.
        //   F_g^w = tau * lam_g^w_up * dPhi_g,  lam_g^w = (krn_end/mu_n)*rho_g*(1-Y)*krn
        //   F_a^w = tau * lam_a^w_up * dPhi_a,  lam_a^w = rho_a*(1-X)*krw
        //   dPhi_g = d(p+pc) - rho_g_mass_edge*g.dx,  dPhi_a = dp - rho_a_mass_edge*g.dx
        // The two-phase potentials, mobilities and densities come from the
        // per-free-DOF flash cache (w_*) built above; node closures use the
        // single-nodal rock freeDOFMaterialTypes[i] (comp-0 skeleton kept).
        // BARE transmissibility tau = -TransportMatrix[ij]/rho_edge strips the
        // rho_w baked into as=rhom*KWs (TransportMatrix = int rhom*KWs gg) so
        // density/mobility live in lam^w/dPhi (the memory's tau/rho_edge option).
        // No high-order term, no EV graph dissipation, no capillary entry-pressure
        // gate (comp-0 gas branch carries only water vapor 1-Y ~ tiny; the DOF-graph
        // loop has no per-element context for the two-sided barrier).  comp-0 FCT
        // is a pass-through: dt_times_fH_minus_fL is zeroed and never consumed
        // (postStep runs only FCTStep(component=1)).
        dt_times_fH_minus_fL.data()[full_offset] = 0.0;
        if (i == j) continue;                       // ii captured above; no self-flux
        const double tau = fmax(0.0, -TransportMatrix[full_offset]) / rho_edge;
        if (tau == 0.0) continue;
        double g_dot_dx = 0.0;
        for (int I = 0; I < nSpace; I++)
          g_dot_dx += gravity.data()[I]
                    * (mesh_dof.data()[node_j * 3 + I] - mesh_dof.data()[node_i * 3 + I]);
        // ---- gas branch (water vapor, ungated) ----
        const double rgm_edge     = 0.5 * (w_rgm[i]     + w_rgm[j]);
        const double rgm_edge_old = 0.5 * (w_rgm_old[i] + w_rgm_old[j]);
        const double dPhi_g     = (u_free_dof[j]     + w_pc[j])     - (u_free_dof[i]     + w_pc[i])
                                - rgm_edge     * g_dot_dx;
        const double dPhi_g_old = (u_free_dof_old[j] + w_pc_old[j]) - (u_free_dof_old[i] + w_pc_old[i])
                                - rgm_edge_old * g_dot_dx;
        const bool   up_i_g     = (dPhi_g     <= 0.0);
        const bool   up_i_g_old = (dPhi_g_old <= 0.0);
        const double lam_g_up     = up_i_g     ? w_lam_g[i]     : w_lam_g[j];
        const double lam_g_up_old = up_i_g_old ? w_lam_g_old[i] : w_lam_g_old[j];
        const double Fg = Theta         * tau * lam_g_up     * dPhi_g
                        + (1.0 - Theta) * tau * lam_g_up_old * dPhi_g_old;
        // ---- aqueous branch (H2O in brine, no gate) ----
        const double ram_edge     = 0.5 * (w_ram[i]     + w_ram[j]);
        const double ram_edge_old = 0.5 * (w_ram_old[i] + w_ram_old[j]);
        const double dPhi_a     = u_free_dof[j]     - u_free_dof[i]     - ram_edge     * g_dot_dx;
        const double dPhi_a_old = u_free_dof_old[j] - u_free_dof_old[i] - ram_edge_old * g_dot_dx;
        const bool   up_i_a     = (dPhi_a     <= 0.0);
        const bool   up_i_a_old = (dPhi_a_old <= 0.0);
        const double lam_a_up     = up_i_a     ? w_lam_a[i]     : w_lam_a[j];
        const double lam_a_up_old = up_i_a_old ? w_lam_a_old[i] : w_lam_a_old[j];
        const double Fa = Theta         * tau * lam_a_up     * dPhi_a
                        + (1.0 - Theta) * tau * lam_a_up_old * dPhi_a_old;
        // R_w[i] -= F_0 (the residual subtracts ith_flux_term below).  Same +sign
        // convention as the old fL (~ +delta_phi = p_j - p_i).
        // P1: the single-nodal-rock comp-0 water flux is RETIRED here -- it is now
        // assembled TWO-SIDED per element-side (elementMaterialTypes[eN]) in the
        // comp-1 element loop below (elementResidual_w + full-CSR scatter).  Keeping
        // ith_flux_term = 0 makes this loop comp-0 ACCUMULATION-ONLY:
        // R_w[i] = MLi*(m-mn)/dt.  (The w_* per-DOF cache above is now unused by the
        // residual; left in place for this correctness pass, TODO: drop for speed.)
        (void)Fg; (void)Fa;
        // ===== Theta-part Jacobian wrt (p_i,z_i,p_j,z_j).  Mirrors comp-1 (eftest). =====
        const double ddPhig_dpi = -1.0 - w_dpc_dp[i] - 0.5 * w_drgm_dp[i] * g_dot_dx;
        const double ddPhig_dzi =      - w_dpc_dz[i] - 0.5 * w_drgm_dz[i] * g_dot_dx;
        const double ddPhig_dpj = +1.0 + w_dpc_dp[j] - 0.5 * w_drgm_dp[j] * g_dot_dx;
        const double ddPhig_dzj =      + w_dpc_dz[j] - 0.5 * w_drgm_dz[j] * g_dot_dx;
        const double ddPhia_dpi = -1.0 - 0.5 * w_dram_dp[i] * g_dot_dx;
        const double ddPhia_dzi =      - 0.5 * w_dram_dz[i] * g_dot_dx;
        const double ddPhia_dpj = +1.0 - 0.5 * w_dram_dp[j] * g_dot_dx;
        const double ddPhia_dzj =      - 0.5 * w_dram_dz[j] * g_dot_dx;
        const double Tt = Theta * tau;
        double dF_dpi = 0.0, dF_dzi = 0.0, dF_dpj = 0.0, dF_dzj = 0.0;
        // gas potential part (all four DOFs)
        dF_dpi += Tt*lam_g_up*ddPhig_dpi; dF_dzi += Tt*lam_g_up*ddPhig_dzi;
        dF_dpj += Tt*lam_g_up*ddPhig_dpj; dF_dzj += Tt*lam_g_up*ddPhig_dzj;
        // gas mobility part (upstream node only)
        if (up_i_g) { dF_dpi += Tt*w_dlam_g_dp[i]*dPhi_g; dF_dzi += Tt*w_dlam_g_dz[i]*dPhi_g; }
        else        { dF_dpj += Tt*w_dlam_g_dp[j]*dPhi_g; dF_dzj += Tt*w_dlam_g_dz[j]*dPhi_g; }
        // aqueous potential part (all four DOFs)
        dF_dpi += Tt*lam_a_up*ddPhia_dpi; dF_dzi += Tt*lam_a_up*ddPhia_dzi;
        dF_dpj += Tt*lam_a_up*ddPhia_dpj; dF_dzj += Tt*lam_a_up*ddPhia_dzj;
        // aqueous mobility part (upstream node only)
        if (up_i_a) { dF_dpi += Tt*w_dlam_a_dp[i]*dPhi_a; dF_dzi += Tt*w_dlam_a_dz[i]*dPhi_a; }
        else        { dF_dpj += Tt*w_dlam_a_dp[j]*dPhi_a; dF_dzj += Tt*w_dlam_a_dz[j]*dPhi_a; }
        // P1: the comp-0 flux Jacobian is RETIRED here too -- its (0,0)/(0,1)
        // blocks are assembled two-sided in the element loop below.  J_ii stays 0,
        // so globalJacobian[ii] below carries only the mass diagonal MLi*dm/dt.
        (void)dF_dpi; (void)dF_dpj; (void)dF_dzi; (void)dF_dzj;
      }
      mDotLow.data()[i] = ith_flux_term/MLi;
      cflux[i] = ith_consistent_flux_term;
      // Final per-DOF coefficient evaluations at (u_w, u_n) and (u_w_old, u_n_old):
      //   m  -> mLow.data()[i]      (current low-order mass)
      //   mn -> mn.data()[i]        (time-history mass for the dt difference)
      //   dm_du_n_fct -> consumed by the (0,1) lumped-mass coupling below.
      evaluateCoefficients_from_Se(a_rowptr.data(), a_colind.data(), rho, rho_i, beta, gravity.data(),
                                   alpha.data()[freeDOFMaterialTypes.data()[i]],
                                   n.data()[freeDOFMaterialTypes.data()[i]], thetaR.data()[freeDOFMaterialTypes.data()[i]], thetaSR.data()[freeDOFMaterialTypes.data()[i]], &KWs.data()[freeDOFMaterialTypes.data()[i] * nnz],
                                   u_free_dof[i], u_dof_n.data()[node_i],
                                   m, dm, dm_du_n_fct, f, df, df_du_n_fct, a, da, da_du_n_fct, as, Kr, dKr, dkr_du_n_fct, thetaW_tmp);
      evaluateCoefficients_from_Se(a_rowptr.data(), a_colind.data(), rho, rho_i, beta, gravity.data(),
                                   alpha.data()[freeDOFMaterialTypes.data()[i]],
                                   n.data()[freeDOFMaterialTypes.data()[i]], thetaR.data()[freeDOFMaterialTypes.data()[i]], thetaSR.data()[freeDOFMaterialTypes.data()[i]], &KWs.data()[freeDOFMaterialTypes.data()[i] * nnz],
                                   u_free_dof_old[i], u_dof_n_old.data()[node_i],
                                   mn.data()[i], dmn, dmn_du_n_fct, fn, dfn, dfn_du_n_fct, an, dan, dan_du_n_fct, asn, Krn, dKrn, dkrn_du_n_fct, thetaW_tmp);
      if (per_rock && node_multi[node_i]) {
        // H2O storage on the node's pore volume W_i (Sec. 4), as the CO2
        // storage already is: MLi * phi(first element visited) -> W_i.  At a
        // one-rock node the two agree (up to roundoff), so they are left as is.
        const int    mat_s = freeDOFMaterialTypes.data()[i];
        const double s_W   = node_W[node_i] / (MLi * (thetaR.data()[mat_s] + thetaSR.data()[mat_s]));
        m *= s_W;  dm *= s_W;  dm_du_n_fct *= s_W;  mn.data()[i] *= s_W;
      }
      mLow.data()[i] = m;
      globalResidual.data()[offset_u + stride_u * i] += bc_mask.data()[i] * (MLi * (m - mn.data()[i]) / dt - ith_flux_term);
      globalJacobian.data()[ii] += bc_mask.data()[i] * (MLi * dm / dt + J_ii) + (1.0 - bc_mask.data()[i]);
      // (0,1) cross-block: lumped-mass contribution MLi * dm/du_n / dt on the
      // (row_w=i, col_n=node_i) entry. This complements the per-edge
      // dkr/du_n * max(0,-T) * delta_phi contributions wired into (row_w=i,
      // col_n=upwind_node) inside the j-loop above. Together they make the
      // low-order operator fully consistent with d R_w / d u_n.
      {
        const int row_w   = offset_u + stride_u * i;
        const int col_n   = offset_n + stride_n * node_i;
        int off_wv = -1;
        for (int o = csrRowIndeces_Full.data()[row_w];
             o < csrRowIndeces_Full.data()[row_w + 1]; o++) {
          if (csrColumnOffsets_Full.data()[o] == col_n) { off_wv = o; break; }
        }
        if (off_wv >= 0) {
          globalJacobian.data()[off_wv] += bc_mask.data()[i] * MLi * dm_du_n_fct / dt;
        }
      }
    }
    // FCT limiter execution.
    //
    // Two operating modes:
    //   * STABILIZATION_TYPE == Implicit_FCT  (Richards-style, in-Newton FCT):
    //         comp-0's fluxCorrection is injected into globalResidual so
    //         Newton sees the limited residual. This is the legacy path; it
    //         relies on small enough alpha-derivative deviations to converge.
    //   * FCT_n == 1 with STABILIZATION_TYPE == EntropyViscosity (TADR-style
    //         defect-correction):
    //         Newton solves the LOW-ORDER R_low cleanly (no Zalesak
    //         contribution to globalResidual). The limiter is computed here
    //         so limited_solution_n and fluxCorrection_n are available, but
    //         the actual scatter to self.u[1].dof happens in Python after
    //         Newton convergence (Coefficients.postStep). This matches TADR /
    //         Richards Newton.solve flow where FCT is a post-step.
    //
    // FCTStep is NOT called here. Both components' FCT run as a Python-
    // orchestrated post-step (Coefficients.postStep -> LevelModel.FCTStep):
    // FCTStep(component, pass=1) -> ghost-scatter Rpos/Rneg -> FCTStep(pass=2),
    // which is the requirement for MPI-parallel mass conservation. This
    // routine just leaves the comp-0 and comp-1 FCT predictor arrays
    // (mLow, mDotLow, dt_times_fH_minus_fL, dLow, min/max_m_bc, ...) populated
    // from the converged iterate.
    if (STABILIZATION_TYPE == STABILIZATION::Implicit_FCT) {
      // Legacy in-Newton injection (kept for STAB=Implicit_FCT only; the
      // Python Coefficients class rejects Implicit_FCT, so this is dead).
      for (int i = 0; i < numDOFs; i++) {
        globalResidual.data()[offset_u + stride_u * i] += fluxCorrection.data()[i];
      }
      for (int i_n = 0; i_n < numDOFs_n; i_n++) {
        globalResidual.data()[offset_n + stride_n * i_n] += fluxCorrection_n.data()[i_n];
      }
    }

    // ============================================================================
    // Component-1 (S_n) -- upwind gas-potential flux on the conserved m_n.
    //
    // The gas equation is recast in potential form:
    //   d(phi*rho_n*S_n)/dt + div F_n = source/sink
    //   F_n = -lambda_n K grad(Phi_n)
    //   Phi_n   = u_w + p_c(S_n) - rho_n g . x
    //   lambda_n(S_n,p_n) = rho_n(p_n) * k_rn(S_n) / mu_n
    // Darcy, capillary, and buoyancy collapse into the single edge flux.
    // ============================================================================

    // -------- Per-DOF nodal CO2 mass (m_c, mc_old for the lumped mass). --------
    // rho_n_phi_dof (= lumped phi*N) and ML_n were projected above for the
    // invert(COMPONENT=1) path; reuse them here.  Compositional (p,z):
    //   m_c = (phi*N) * z (= u_n).  invert: z = m_c / (phi*N).
    std::vector<double> m_n_DOF(numDOFs_n, 0.0);
    std::vector<double> mn_n_DOF(numDOFs_n, 0.0);
    for (int i_n = 0; i_n < numDOFs_n; i_n++) {
      const double sat     = u_dof_n.data()[i_n];      // z at DOF i (current)
      const double sat_old = u_dof_n_old.data()[i_n];  // z at DOF i (t^n)
      m_n_DOF[i_n]  = rho_n_phi_dof[i_n] * sat;
      mn_n_DOF[i_n] = rho_n_phi_dof_old[i_n] * sat_old;   // old mass uses OLD phi*N
      mn_n.data()[i_n]        = mn_n_DOF[i_n];     // diagnostic
      quantDOFs_n.data()[i_n] = 0.0;                // reset
    }

    double diag_sumF = 0.0, diag_absF = 0.0;

    // Zero the per-node gas-residual budget (6 slots, term-major over numDOFs_n).
    const bool have_gas_budget = (gas_budget_node.size() >= (size_t)(6 * numDOFs_n));
    if (have_gas_budget)
      for (int s = 0; s < 6 * numDOFs_n; ++s) gas_budget_node.data()[s] = 0.0;

    std::vector<double> psi_n(numDOFs_n, 1.0);
    if (STABILIZATION_TYPE == STABILIZATION::EV_Stab) {
      for (int i_n = 0; i_n < numDOFs_n; i_n++) {
        const double zi = u_dof_n_old.data()[i_n];
        double num = 0.0, den = 0.0;
        for (int offset = csrRowIndeces_n_DofLoops.data()[i_n];
             offset < csrRowIndeces_n_DofLoops.data()[i_n + 1]; offset++) {
          const int j_n = csrColumnOffsets_n_DofLoops.data()[offset];
          if (j_n == i_n) continue;
          const double d = zi - u_dof_n_old.data()[j_n];
          num += d;
          den += fabs(d);
        }
        const double alpha_i = fabs(num) / (den + 1.0e-15);
        psi_n[i_n] = (POWER_SMOOTHNESS_INDICATOR == 0)
                   ? 1.0 : std::pow(alpha_i, POWER_SMOOTHNESS_INDICATOR);
      }
    }
    std::vector<int> node_iface(numDOFs_n, 0);
    {
      std::vector<int> node_mat0(numDOFs_n, -1);
      for (int eN = 0; eN < nElements_global; eN++) {
        const int mat_eN_if = elementMaterialTypes.data()[eN];
        for (int a = 0; a < nDOF_trial_element; a++) {
          const int gN = u_l2g.data()[eN * nDOF_trial_element + a];
          if (node_mat0[gN] < 0)            node_mat0[gN] = mat_eN_if;
          else if (node_mat0[gN] != mat_eN_if) node_iface[gN] = 1;
        }
      }
    }
    auto comp1_offset = [&](int i_n, int j_n) -> int {
      for (int off = csrRowIndeces_n_DofLoops.data()[i_n];
           off < csrRowIndeces_n_DofLoops.data()[i_n + 1]; ++off)
        if (csrColumnOffsets_n_DofLoops.data()[off] == j_n) return off;
      return -1;
    };
    for (int off = 0; off < NNZ_n; ++off) {
      dLow_n.data()[off]                 = 0.0;
      dEV_n.data()[off]                  = 0.0;
      dt_times_fH_minus_fL_n.data()[off] = 0.0;
    }
    std::vector<double> node_Kdiag(numDOFs_n, 0.0);

    for (int eN = 0; eN < nElements_global; eN++) {
      const int    mat_eN    = elementMaterialTypes.data()[eN];
      const double phi_eN    = thetaR.data()[mat_eN] + thetaSR.data()[mat_eN];
      const double alpha_eN  = alpha.data()[mat_eN];
      const double krn_end_eN = krn_end.data()[mat_eN];
      const double n_vg_eN   = n.data()[mat_eN];
      const double *KWs_eN   = &KWs.data()[mat_eN * nnz];
      // dm_n_du_n is no longer hoisted: when rho_n is compressible the
      // per-DOF lumped diagonal is taken from rho_n_phi_dof, so the
      // (1,1) lumped contribution lives inside the per-i loop below.
      double elementResidual_n[nDOF_test_element];
      // P1: comp-0 (H2O) two-sided water flux residual assembled in THIS element
      // loop (replaces the single-nodal-rock DOF-graph water flux).  Its (0,0)/
      // (0,1) Jacobian scatters directly into globalJacobian via the full CSR
      // inside the edge loop (same mapping the DOF-graph loop used).
      double elementResidual_w[nDOF_test_element];
      double elementMass_n[nDOF_test_element];
      double u_n_local[nDOF_trial_element];
      double u_n_old_local[nDOF_trial_element];
      double elementJacobian_n_n[nDOF_test_element][nDOF_trial_element];
      double elementJacobian_n_w[nDOF_test_element][nDOF_trial_element];
      // P1 (Richards-style block assembly): the comp-0 (H2O) water-flux
      // Jacobian collects into dedicated element arrays and scatters via the
      // framework's (0,0)/(0,1) block CSR maps (csr*_w_w / csr*_w_n) with the
      // element-local eN_i_j offset -- EXACTLY mirroring the comp-1 (1,1)/(1,0)
      // scatter below and Richards.h's direct globalJacobian[ij] write.  This
      // replaces the old Full-CSR column SEARCH (col==col_n_j ...), which
      // silently dropped the (0,1) off-diagonal water<-neighbor-z coupling when
      // the hand-computed column index didn't match the matrix layout (the
      // "structural misses" in the FD Jacobian probe -> stalled Newton).
      double elementJacobian_w_w[nDOF_test_element][nDOF_trial_element];
      double elementJacobian_w_n[nDOF_test_element][nDOF_trial_element];
      // elementTransport_n collects the mobility-free gas transmissibility
      //   tau_ij = int K . grad N_j . grad N_i dV
      // consumed by the post-element-loop upwind potential-flux edge pass.
      // NO rho_n, NO k_rn, NO p_c factors -- those go into lambda_up and
      // delta_Phi at the edge level.
      double elementTransport_n[nDOF_test_element][nDOF_trial_element];
      const int eN_nDOF_trial_element = eN * nDOF_trial_element;
      for (int i = 0; i < nDOF_test_element; i++) {
        elementResidual_n[i] = 0.0;
        elementResidual_w[i] = 0.0;
        elementMass_n[i]     = 0.0;
        for (int j = 0; j < nDOF_trial_element; j++) {
          elementJacobian_n_n[i][j] = 0.0;
          elementJacobian_n_w[i][j] = 0.0;
          elementJacobian_w_w[i][j] = 0.0;
          elementJacobian_w_n[i][j] = 0.0;
          elementTransport_n[i][j]  = 0.0;
        }
      }
      for (int j = 0; j < nDOF_trial_element; j++) {
        u_n_local[j]     = u_dof_n.data()[u_l2g_n.data()[eN_nDOF_trial_element + j]];
        u_n_old_local[j] = u_dof_n_old.data()[u_l2g_n.data()[eN_nDOF_trial_element + j]];
      }
      for (int k = 0; k < nQuadraturePoints_element; k++) {
        double jac[nSpace * nSpace], jacDet, jacInv[nSpace * nSpace], x_q, y_q, z_q;
        ck.calculateMapping_element(eN, k, mesh_dof.data(), mesh_l2g.data(),
                                    mesh_trial_ref.data(), mesh_grad_trial_ref.data(),
                                    jac, jacDet, jacInv, x_q, y_q, z_q);
        const double dV = std::fabs(jacDet) * dV_ref.data()[k];
        double u_grad_trial_qp[nDOF_trial_element * nSpace];
        ck.gradTrialFromRef(&u_grad_trial_ref.data()[k * nDOF_trial_element * nSpace],
                            jacInv, u_grad_trial_qp);
        // Cell loop only accumulates:
        //   elementMass_n[i] = int N_i dV         (lumped row volume, consumed
        //                                          below by time-derivative,
        //                                          R_diss, Q_inj at element scope)
        //   elementTransport_n[i][j] = int K . grad N_j . grad N_i dV
        //                                          (tau_ij used by the
        //                                          post-element-loop upwind
        //                                          potential-flux edge pass)
        // No closure evaluation, no flux residual, no coefficient sensitivities:
        // those collapse into the edge flux F_ij = tau_ij * lambda_up * delta_Phi.
        for (int i = 0; i < nDOF_test_element; i++) {
          const double test_i = u_test_ref.data()[k * nDOF_test_element + i];
          elementMass_n[i] += test_i * dV;
          for (int j = 0; j < nDOF_trial_element; j++) {
            double K_trial_ij = 0.0;
            for (int I = 0; I < nSpace; I++) {
              const double grad_Ni_I = u_grad_trial_qp[i * nSpace + I];
              for (int ii = a_rowptr.data()[I]; ii < a_rowptr.data()[I + 1]; ii++) {
                const int J = a_colind.data()[ii];
                K_trial_ij += KWs_eN[ii] * u_grad_trial_qp[j * nSpace + J] * grad_Ni_I;
              }
            }
            elementTransport_n[i][j] += K_trial_ij * dV;
          }
        }
      } // end QP loop

      // -------- Lumped mass: ML_n[i] * (m_c - mc_old)/dt on the diagonal. --------
      // Applied at element level so it sums to the global lumped mass.
      // Compositional (p,z): m_c = (phi*N)*z, with phi*N (and d/dz, d/dp) taken
      // from the lumped projections rho_n_phi_dof / dphiN_dz_dof / dphiN_dp_dof.
      //   dm_c/dz = phi*N + z*phi*dN/dz      (1,1) diagonal
      //   dm_c/dp = z*phi*dN/dp              (1,0) diagonal
      // Accumulation Jacobian formulas FD-verified in kernel_math_test.cpp (kmtest).
      for (int i = 0; i < nDOF_test_element; i++) {
        const int    gi        = u_l2g_n.data()[eN * nDOF_test_element + i];   // comp-1 DOF
        node_Kdiag[gi]        += fmax(0.0, elementTransport_n[i][i]);          // K-weighted nodal conductance
        const double phiN_i    = rho_n_phi_dof[gi];       // lumped phi*N (current)
        const double phiN_old_i= rho_n_phi_dof_old[gi];   // lumped phi*N (t^n)
        const double z_i       = u_n_local[i];
        const double m_n_loc     = phiN_i     * z_i;                // m_c
        const double m_n_old_loc = phiN_old_i * u_n_old_local[i];   // m_c_old
        elementResidual_n[i]      += elementMass_n[i] * (m_n_loc - m_n_old_loc) / dt;
        if (have_gas_budget)
          gas_budget_node.data()[0 * numDOFs_n + gi] += elementMass_n[i] * (m_n_loc - m_n_old_loc) / dt;
        // (1,1): dm_c/dz = phi*N + z*phi*dN/dz.
        elementJacobian_n_n[i][i] += elementMass_n[i] * (phiN_i + z_i * dphiN_dz_dof[gi]) / dt;
        // (1,0): dm_c/dp = z*phi*dN/dp.
        elementJacobian_n_w[i][i] += elementMass_n[i] * (z_i * dphiN_dp_dof[gi]) / dt;
        if (inj_point_mode == 0) {
          const double Q_inj_n    = injection_dof.data()[gi];
          elementResidual_n[i]   -= elementMass_n[i] * Q_inj_n;
          if (have_gas_budget)
            gas_budget_node.data()[3 * numDOFs_n + gi] -= elementMass_n[i] * Q_inj_n;
        }
      }

      if (inj_point_mode == 1) {
        for (int p = 0; p < inj_n_ports; p++) {
          if (inj_element.data()[p] == eN && inj_rate.data()[p] != 0.0) {
            const double qp = inj_rate.data()[p];
            for (int i = 0; i < nDOF_test_element; i++) {
              const double w = inj_weight.data()[p * nDOF_test_element + i];
              elementResidual_n[i] -= qp * w;
              if (have_gas_budget) {
                const int gii = u_l2g_n.data()[eN * nDOF_test_element + i];   // comp-1 DOF
                gas_budget_node.data()[3 * numDOFs_n + gii] -= qp * w;
              }
            }
          }
        }
      }

      {
        const double S_wr_eN     = thetaR.data()[mat_eN] / phi_eN;
        const double one_m_Sr_eN = 1.0 - S_wr_eN;
        const double Se_trap_L4415 = 1.0 - S_gr.data()[mat_eN] / one_m_Sr_eN;  // gas-only residual trapping
        // Capillary entry pressure of THIS element's rock, p_d_e = 1/alpha [head].
        // The barrier below charges the gas flux the entry-pressure JUMP
        // (p_d_e - p_d_coarsest_neighbor) when gas crosses an edge INTO this
        // element from a coarser-medium (lower p_d) node.
        const double p_d_e = (alpha_eN > 0.0) ? (1.0 / alpha_eN) : 0.0;
        // P2 (compositional): per-node closure for the TWO-PHASE CO2 edge flux
        //   F_1 = F_g + F_a,  F_g = tau*lam_g_up*gate*dPhi_g,  F_a = tau*lam_a_up*dPhi_a
        //   lam_g = (krn_end/mu_n)*rho_g*Y*krn,   lam_a = rho_a*X*krw   (molar)
        //   dPhi_g = d(p+pc) - rho_g_mass_edge*g.dx,  dPhi_a = dp - rho_a_mass_edge*g.dx
        // All flash-derived; FD-verified in edge_flux_test.cpp.
        int    gN_e[nDOF_trial_element];   // mesh-node id (p, geometry)
        int    zN_e[nDOF_trial_element];   // comp-1 z-DOF
        double uw_e[nDOF_trial_element],   uw_old_e[nDOF_trial_element];
        double pc_e[nDOF_trial_element],   pc_old_e[nDOF_trial_element];
        double dpc_dp_e[nDOF_trial_element], dpc_dz_e[nDOF_trial_element];
        double lam_g_e[nDOF_trial_element], lam_a_e[nDOF_trial_element];
        double lam_g_old_e[nDOF_trial_element], lam_a_old_e[nDOF_trial_element];
        double dlam_g_dp_e[nDOF_trial_element], dlam_g_dz_e[nDOF_trial_element];
        double dlam_a_dp_e[nDOF_trial_element], dlam_a_dz_e[nDOF_trial_element];
        // P1: H2O-weighted molar mobilities for the two-sided comp-0 water flux.
        // lam_g^w = cg*rho_g*(1-Y)*krn, lam_a^w = rho_a*(1-X)*krw -- same flash /
        // krn / krw primitives as the CO2 mobilities, weights flipped Y->(1-Y),
        // X->(1-X), no gate.  FD-verified in comp0_elem_test.cpp (c0etest).
        double lwg_e[nDOF_trial_element], lwa_e[nDOF_trial_element];
        double lwg_old_e[nDOF_trial_element], lwa_old_e[nDOF_trial_element];
        double dlwg_dp_e[nDOF_trial_element], dlwg_dz_e[nDOF_trial_element];
        double dlwa_dp_e[nDOF_trial_element], dlwa_dz_e[nDOF_trial_element];
        double rgm_e[nDOF_trial_element], ram_e[nDOF_trial_element];
        double rgm_old_e[nDOF_trial_element], ram_old_e[nDOF_trial_element];
        double drgm_dp_e[nDOF_trial_element], drgm_dz_e[nDOF_trial_element];
        double dram_dp_e[nDOF_trial_element], dram_dz_e[nDOF_trial_element];
        double Sg_e[nDOF_trial_element], Sg_old_e[nDOF_trial_element];
        double dSg_dp_e[nDOF_trial_element], dSg_dz_e[nDOF_trial_element];
        double dlg_dz_o[nDOF_trial_element], dla_dz_o[nDOF_trial_element], dpc_dz_o[nDOF_trial_element]; // old-time z-derivs for lagged graph viscosity
        const double cg_eN  = krn_end_eN / mu_n;
        const double dMm_e  = ::m_comp_co2::eos::M_CO2_KG - ::m_comp_co2::eos::M_H2O_KG;
        for (int a = 0; a < nDOF_trial_element; a++) {
          const int gN = u_l2g.data()[eN_nDOF_trial_element + a];      // mesh node: p + geometry
          const int zN = u_l2g_n.data()[eN_nDOF_trial_element + a];    // comp-1 z-DOF
          gN_e[a]      = gN;
          zN_e[a]      = zN;
          const double p_a   = u_dof.data()[gN];
          const double z_a   = u_dof_n.data()[zN];        // u_n = z (compositional)
          const double p_a_o = u_dof_old.data()[gN];
          const double z_a_o = u_dof_n_old.data()[zN];
          uw_e[a] = p_a;  uw_old_e[a] = p_a_o;
          // --- current iterate: flash + closures ---
          const double p_cl = fmax(p_a, 1.0e2);
          ::m_comp_co2::flash::FlashState f =
              ::m_comp_co2::flash::flashPZ(p_cl, z_a, T_C_member, m_NaCl_member, ::m_comp_co2::flash::EPS_Z, immiscible_member);
          // Gas saturation of THIS element's rock at the vertex: the per-rock
          // closure's S_r at interface nodes, the flash's S_g elsewhere.
          double Sg_v = f.S_g, dSg_dp_v = f.dS_g_dp, dSg_dz_v = f.dS_g_dz;
          if (per_rock && node_multi[zN]) {
            const int o = zN * nRock + mat_eN;
            Sg_v = rS[o];  dSg_dp_v = rS_dp[o];  dSg_dz_v = rS_dz[o];
          }
          Sg_e[a] = Sg_v;  dSg_dp_e[a] = dSg_dp_v;  dSg_dz_e[a] = dSg_dz_v;
          const double Sa = 1.0 - Sg_v;
          const double Se_raw = (Sa - S_wr_eN) / one_m_Sr_eN;
          double Se, dSe_dp, dSe_dz;
          if (Se_raw <= 0.0)      { Se = 0.0; dSe_dp = 0.0; dSe_dz = 0.0; }
          else if (Se_raw >= 1.0) { Se = 1.0; dSe_dp = 0.0; dSe_dz = 0.0; }
          else { Se = Se_raw; dSe_dp = -dSg_dp_v/one_m_Sr_eN; dSe_dz = -dSg_dz_v/one_m_Sr_eN; }
          double krn=0,dkrn=0,krw=0,dkrw=0,thW=0,DthW=0,pc=0,dpc_dSe=0,d2pc=0;
          if (PSK_TYPE_member == 1) {
            proteus::m_comp_co2::psk::bc_kr_nonwetting_from_Se(Se, alpha_eN, n_vg_eN, krn, dkrn, Se_trap_L4415);
            proteus::m_comp_co2::psk::bc_wetting_from_Se(Se, alpha_eN, n_vg_eN, thetaR.data()[mat_eN], thetaSR.data()[mat_eN], thW, DthW, krw, dkrw);
            proteus::m_comp_co2::psk::bc_pc_from_Se(Se, alpha_eN, n_vg_eN, bc_Se_min.data()[mat_eN], pc, dpc_dSe, d2pc);
          } else {
            proteus::m_comp_co2::psk::vgm_kr_nonwetting_from_Se(Se, alpha_eN, n_vg_eN, krn, dkrn, Se_trap_L4415);
            proteus::m_comp_co2::psk::vgm_wetting_from_Se(Se, alpha_eN, n_vg_eN, thetaR.data()[mat_eN], thetaSR.data()[mat_eN], thW, DthW, krw, dkrw);
            proteus::m_comp_co2::psk::vgm_pc_from_Se(Se, alpha_eN, n_vg_eN, pc, dpc_dSe, d2pc);
          }
          pc_e[a] = pc;  dpc_dp_e[a] = dpc_dSe*dSe_dp;  dpc_dz_e[a] = dpc_dSe*dSe_dz;
          const double Mg = f.Y*::m_comp_co2::eos::M_CO2_KG + (1.0-f.Y)*::m_comp_co2::eos::M_H2O_KG;
          const double Ma = f.X*::m_comp_co2::eos::M_CO2_KG + (1.0-f.X)*::m_comp_co2::eos::M_H2O_KG;
          rgm_e[a] = f.rho_g*Mg;  ram_e[a] = f.rho_a*Ma;
          drgm_dp_e[a] = f.drho_g_dp*Mg + f.rho_g*f.dY_dp*dMm_e;  drgm_dz_e[a] = f.rho_g*f.dY_dz*dMm_e;
          dram_dp_e[a] = f.drho_a_dp*Ma + f.rho_a*f.dX_dp*dMm_e;  dram_dz_e[a] = f.drho_a_dz*Ma + f.rho_a*f.dX_dz*dMm_e;
          // molar mobilities lam_g = cg*rho_g*Y*krn, lam_a = rho_a*X*krw
          lam_g_e[a]     = cg_eN*f.rho_g*f.Y*krn;
          dlam_g_dp_e[a] = cg_eN*(f.drho_g_dp*f.Y*krn + f.rho_g*f.dY_dp*krn + f.rho_g*f.Y*dkrn*dSe_dp);
          dlam_g_dz_e[a] = cg_eN*(                       f.rho_g*f.dY_dz*krn + f.rho_g*f.Y*dkrn*dSe_dz);
          lam_a_e[a]     = f.rho_a*f.X*krw;
          dlam_a_dp_e[a] = f.drho_a_dp*f.X*krw + f.rho_a*f.dX_dp*krw + f.rho_a*f.X*dkrw*dSe_dp;
          dlam_a_dz_e[a] = f.drho_a_dz*f.X*krw + f.rho_a*f.dX_dz*krw + f.rho_a*f.X*dkrw*dSe_dz;
          // P1: H2O-weighted siblings (1-Y),(1-X) -- same krn/krw/dSe, no gate.
          {
            const double yw=1.0-f.Y, dyw_dp=-f.dY_dp, dyw_dz=-f.dY_dz;
            const double xw=1.0-f.X, dxw_dp=-f.dX_dp, dxw_dz=-f.dX_dz;
            lwg_e[a]     = cg_eN*f.rho_g*yw*krn;
            dlwg_dp_e[a] = cg_eN*(f.drho_g_dp*yw*krn + f.rho_g*dyw_dp*krn + f.rho_g*yw*dkrn*dSe_dp);
            dlwg_dz_e[a] = cg_eN*(                      f.rho_g*dyw_dz*krn + f.rho_g*yw*dkrn*dSe_dz);
            lwa_e[a]     = f.rho_a*xw*krw;
            dlwa_dp_e[a] = f.drho_a_dp*xw*krw + f.rho_a*dxw_dp*krw + f.rho_a*xw*dkrw*dSe_dp;
            dlwa_dz_e[a] = f.drho_a_dz*xw*krw + f.rho_a*dxw_dz*krw + f.rho_a*xw*dkrw*dSe_dz;
          }
          // --- old time level (frozen; feeds the (1-Theta) part, no derivatives) ---
          const double p_cl_o = fmax(p_a_o, 1.0e2);
          ::m_comp_co2::flash::FlashState fo =
              ::m_comp_co2::flash::flashPZ(p_cl_o, z_a_o, T_C_member, m_NaCl_member, ::m_comp_co2::flash::EPS_Z, immiscible_member);
          double Sg_v_o = fo.S_g, dSg_dz_v_o = fo.dS_g_dz;
          if (per_rock && node_multi[zN]) {
            const int o = zN * nRock + mat_eN;
            Sg_v_o = rS_o[o];  dSg_dz_v_o = rS_dz_o[o];
          }
          Sg_old_e[a] = Sg_v_o;
          const double Sa_o = 1.0 - Sg_v_o;
          const double Se_o_raw = (Sa_o - S_wr_eN) / one_m_Sr_eN;
          const double Se_o = Se_o_raw <= 0.0 ? 0.0 : (Se_o_raw >= 1.0 ? 1.0 : Se_o_raw);
          double krn_o=0,dkrn_o=0,krw_o=0,dkrw_o=0,thW_o=0,DthW_o=0,pc_o=0,dpc_o=0,d2pc_o=0;
          if (PSK_TYPE_member == 1) {
            proteus::m_comp_co2::psk::bc_kr_nonwetting_from_Se(Se_o, alpha_eN, n_vg_eN, krn_o, dkrn_o, Se_trap_L4415);
            proteus::m_comp_co2::psk::bc_wetting_from_Se(Se_o, alpha_eN, n_vg_eN, thetaR.data()[mat_eN], thetaSR.data()[mat_eN], thW_o, DthW_o, krw_o, dkrw_o);
            proteus::m_comp_co2::psk::bc_pc_from_Se(Se_o, alpha_eN, n_vg_eN, bc_Se_min.data()[mat_eN], pc_o, dpc_o, d2pc_o);
          } else {
            proteus::m_comp_co2::psk::vgm_kr_nonwetting_from_Se(Se_o, alpha_eN, n_vg_eN, krn_o, dkrn_o, Se_trap_L4415);
            proteus::m_comp_co2::psk::vgm_wetting_from_Se(Se_o, alpha_eN, n_vg_eN, thetaR.data()[mat_eN], thetaSR.data()[mat_eN], thW_o, DthW_o, krw_o, dkrw_o);
            proteus::m_comp_co2::psk::vgm_pc_from_Se(Se_o, alpha_eN, n_vg_eN, pc_o, dpc_o, d2pc_o);
          }
          pc_old_e[a] = pc_o;
          const double Mg_o = fo.Y*::m_comp_co2::eos::M_CO2_KG + (1.0-fo.Y)*::m_comp_co2::eos::M_H2O_KG;
          const double Ma_o = fo.X*::m_comp_co2::eos::M_CO2_KG + (1.0-fo.X)*::m_comp_co2::eos::M_H2O_KG;
          rgm_old_e[a] = fo.rho_g*Mg_o;  ram_old_e[a] = fo.rho_a*Ma_o;
          lam_g_old_e[a] = cg_eN*fo.rho_g*fo.Y*krn_o;
          lam_a_old_e[a] = fo.rho_a*fo.X*krw_o;
          lwg_old_e[a]   = cg_eN*fo.rho_g*(1.0-fo.Y)*krn_o;   // P1: H2O old-time
          lwa_old_e[a]   = fo.rho_a*(1.0-fo.X)*krw_o;
          const double dSe_o = (Se_o_raw>0.0 && Se_o_raw<1.0) ? -dSg_dz_v_o/one_m_Sr_eN : 0.0;
          dlg_dz_o[a] = cg_eN*(fo.rho_g*fo.dY_dz*krn_o + fo.rho_g*fo.Y*dkrn_o*dSe_o);
          dla_dz_o[a] = fo.drho_a_dz*fo.X*krw_o + fo.rho_a*fo.dX_dz*krw_o + fo.rho_a*fo.X*dkrw_o*dSe_o;
          dpc_dz_o[a] = dpc_o*dSe_o;
        }
        // Edge pair flux. tau^e is symmetric and lambda_up picks the same
        // physical upstream node for (i,j) and (j,i), so F^e_ij = -F^e_ji
        // (per-element conservation; summing the <=2 elements sharing an edge
        // reconstructs the full two-sided edge flux).
        for (int i = 0; i < nDOF_test_element; i++) {
          for (int j = 0; j < nDOF_trial_element; j++) {
            if (i == j) continue;
            const double tau = fmax(0.0, -elementTransport_n[i][j]);
            if (tau == 0.0) continue;
            double g_dot_dx = 0.0;
            for (int I = 0; I < nSpace; I++) {
              g_dot_dx += gravity.data()[I] * (mesh_dof.data()[gN_e[j] * 3 + I]
                                             - mesh_dof.data()[gN_e[i] * 3 + I]);
            }
            // ===================== GAS branch (free CO2, gated) ================
            // lam_g = rho_g*Y*krn*krn_end/mu_n ; dPhi_g = d(p+pc) - rho_g_mass*g.dx
            const double rgm_edge     = 0.5 * (rgm_e[i]     + rgm_e[j]);
            const double rgm_edge_old = 0.5 * (rgm_old_e[i] + rgm_old_e[j]);
            const double dPhi_g     = (uw_e[j]     + pc_e[j])     - (uw_e[i]     + pc_e[i])
                                    - rgm_edge     * g_dot_dx;
            const double dPhi_g_old = (uw_old_e[j] + pc_old_e[j]) - (uw_old_e[i] + pc_old_e[i])
                                    - rgm_edge_old * g_dot_dx;
            const bool   up_i_g     = (dPhi_g     <= 0.0);
            const bool   up_i_g_old = (dPhi_g_old <= 0.0);
            const double lam_g_up     = up_i_g     ? lam_g_e[i]     : lam_g_e[j];
            const double lam_g_up_old = up_i_g_old ? lam_g_old_e[i] : lam_g_old_e[j];
            // GATE-FREE gas branch -- the capillary entry-pressure valve is DELETED.
            // With one z (hence one S_g) per node, a node shared by several rocks
            // does not represent the capillary-pressure jump; see
            // Z_ACROSS_ROCK_INTERFACES.md.  No node_pd_min / node_Sn_max valve, no
            // gate Jacobian, no smoothstep.
            const double Fg = Theta         * tau * lam_g_up     * dPhi_g
                            + (1.0 - Theta) * tau * lam_g_up_old * dPhi_g_old;
            // ================= AQUEOUS branch (dissolved CO2, no gate) =========
            // lam_a = rho_a*X*krw ; dPhi_a = dp - rho_a_mass*g.dx
            const double ram_edge     = 0.5 * (ram_e[i]     + ram_e[j]);
            const double ram_edge_old = 0.5 * (ram_old_e[i] + ram_old_e[j]);
            const double dPhi_a     = uw_e[j]     - uw_e[i]     - ram_edge     * g_dot_dx;
            const double dPhi_a_old = uw_old_e[j] - uw_old_e[i] - ram_edge_old * g_dot_dx;
            const bool   up_i_a     = (dPhi_a     <= 0.0);
            const bool   up_i_a_old = (dPhi_a_old <= 0.0);
            const double lam_a_up     = up_i_a     ? lam_a_e[i]     : lam_a_e[j];
            const double lam_a_up_old = up_i_a_old ? lam_a_old_e[i] : lam_a_old_e[j];
            const double Fa = Theta         * tau * lam_a_up     * dPhi_a
                            + (1.0 - Theta) * tau * lam_a_up_old * dPhi_a_old;
            // R_n[i] -= F_g + F_a.
            elementResidual_n[i] -= Fg + Fa;
            if (have_gas_budget)
              gas_budget_node.data()[1 * numDOFs_n + zN_e[i]] -= Fg + Fa;
            diag_sumF += Fg + Fa;
            diag_absF += std::fabs(Fg + Fa);
            // ===== Theta-part Jacobian wrt (p_i,z_i,p_j,z_j). FD-verified. =====
            const double ddPhig_dpi = -1.0 - dpc_dp_e[i] - 0.5 * drgm_dp_e[i] * g_dot_dx;
            const double ddPhig_dzi =      - dpc_dz_e[i] - 0.5 * drgm_dz_e[i] * g_dot_dx;
            const double ddPhig_dpj = +1.0 + dpc_dp_e[j] - 0.5 * drgm_dp_e[j] * g_dot_dx;
            const double ddPhig_dzj =      + dpc_dz_e[j] - 0.5 * drgm_dz_e[j] * g_dot_dx;
            const double ddPhia_dpi = -1.0 - 0.5 * dram_dp_e[i] * g_dot_dx;
            const double ddPhia_dzi =      - 0.5 * dram_dz_e[i] * g_dot_dx;
            const double ddPhia_dpj = +1.0 - 0.5 * dram_dp_e[j] * g_dot_dx;
            const double ddPhia_dzj =      - 0.5 * dram_dz_e[j] * g_dot_dx;
            const double Tt = Theta * tau;
            double dF_dpi = 0.0, dF_dzi = 0.0, dF_dpj = 0.0, dF_dzj = 0.0;
            // gas potential part (all four DOFs) -- gate-free
            dF_dpi += Tt*lam_g_up*ddPhig_dpi; dF_dzi += Tt*lam_g_up*ddPhig_dzi;
            dF_dpj += Tt*lam_g_up*ddPhig_dpj; dF_dzj += Tt*lam_g_up*ddPhig_dzj;
            // gas mobility part (upstream node only)
            if (up_i_g) { dF_dpi += Tt*dlam_g_dp_e[i]*dPhi_g; dF_dzi += Tt*dlam_g_dz_e[i]*dPhi_g; }
            else        { dF_dpj += Tt*dlam_g_dp_e[j]*dPhi_g; dF_dzj += Tt*dlam_g_dz_e[j]*dPhi_g; }
            // (gate Jacobian removed with the gate)
            // aqueous potential part (all four DOFs)
            dF_dpi += Tt*lam_a_up*ddPhia_dpi; dF_dzi += Tt*lam_a_up*ddPhia_dzi;
            dF_dpj += Tt*lam_a_up*ddPhia_dpj; dF_dzj += Tt*lam_a_up*ddPhia_dzj;
            // aqueous mobility part (upstream node only)
            if (up_i_a) { dF_dpi += Tt*dlam_a_dp_e[i]*dPhi_a; dF_dzi += Tt*dlam_a_dz_e[i]*dPhi_a; }
            else        { dF_dpj += Tt*dlam_a_dp_e[j]*dPhi_a; dF_dzj += Tt*dlam_a_dz_e[j]*dPhi_a; }
            // dR_n[i]/dx = -dF/dx.  (1,1) = z columns, (1,0) = p columns.
            elementJacobian_n_n[i][i] += -dF_dzi;
            elementJacobian_n_n[i][j] += -dF_dzj;
            elementJacobian_n_w[i][i] += -dF_dpi;
            elementJacobian_n_w[i][j] += -dF_dpj;
            // ===== P1: comp-0 (H2O) two-sided water flux  F_0^e = F_g^w + F_a^w.
            // Reuses THIS edge's dPhi_g/dPhi_a/tau and the SAME upstream switches
            // (phase potentials are composition-independent); only the mobilities
            // change to the H2O weights lwg=cg*rho_g*(1-Y)*krn, lwa=rho_a*(1-X)*krw.
            // NO gate (matches the verified comp-0 kernel c0test/c0etest; the
            // (1-Y) gas branch is water vapor ~ 0 anyway).  R_w[i] -= F_0^e and the
            // (0,0)/(0,1) Jacobian scatter into comp-0 rows via the full CSR -- the
            // SAME mapping the retired single-nodal DOF-graph water loop used.  The
            // water MASS + Dirichlet identity stay in the DOF-graph accumulation.
            {
              const double lwg_up     = up_i_g     ? lwg_e[i]     : lwg_e[j];
              const double lwg_up_old = up_i_g_old ? lwg_old_e[i] : lwg_old_e[j];
              const double lwa_up     = up_i_a     ? lwa_e[i]     : lwa_e[j];
              const double lwa_up_old = up_i_a_old ? lwa_old_e[i] : lwa_old_e[j];
              const double Fwg = Theta         * tau * lwg_up     * dPhi_g
                               + (1.0 - Theta) * tau * lwg_up_old * dPhi_g_old;
              const double Fwa = Theta         * tau * lwa_up     * dPhi_a
                               + (1.0 - Theta) * tau * lwa_up_old * dPhi_a_old;
              elementResidual_w[i] -= Fwg + Fwa;
              // Theta-part Jacobian wrt (p_i,z_i,p_j,z_j) (gate=1, no gate deriv).
              double dFw_dpi=0.0, dFw_dzi=0.0, dFw_dpj=0.0, dFw_dzj=0.0;
              dFw_dpi += Tt*lwg_up*ddPhig_dpi; dFw_dzi += Tt*lwg_up*ddPhig_dzi;
              dFw_dpj += Tt*lwg_up*ddPhig_dpj; dFw_dzj += Tt*lwg_up*ddPhig_dzj;
              if (up_i_g) { dFw_dpi += Tt*dlwg_dp_e[i]*dPhi_g; dFw_dzi += Tt*dlwg_dz_e[i]*dPhi_g; }
              else        { dFw_dpj += Tt*dlwg_dp_e[j]*dPhi_g; dFw_dzj += Tt*dlwg_dz_e[j]*dPhi_g; }
              dFw_dpi += Tt*lwa_up*ddPhia_dpi; dFw_dzi += Tt*lwa_up*ddPhia_dzi;
              dFw_dpj += Tt*lwa_up*ddPhia_dpj; dFw_dzj += Tt*lwa_up*ddPhia_dzj;
              if (up_i_a) { dFw_dpi += Tt*dlwa_dp_e[i]*dPhi_a; dFw_dzi += Tt*dlwa_dz_e[i]*dPhi_a; }
              else        { dFw_dpj += Tt*dlwa_dp_e[j]*dPhi_a; dFw_dzj += Tt*dlwa_dz_e[j]*dPhi_a; }
              // Richards-style block scatter: accumulate into the element
              // (0,0)/(0,1) arrays; the end-of-element loop loads them into the
              // global Jacobian via the framework's csr*_w_w / csr*_w_n block
              // maps (eN_i_j offset), so the off-diagonal water<-neighbor-z
              // slot is ALWAYS hit (no column search, no dropped coupling).
              // Dirichlet mask uses the comp-0 free-DOF tag of the test node.
              const int fi = r_l2g.data()[eN_nDOF_trial_element + i];
              const double mwi = bc_mask.data()[fi];
              elementJacobian_w_w[i][i] += mwi * (-dFw_dpi);
              elementJacobian_w_w[i][j] += mwi * (-dFw_dpj);
              elementJacobian_w_n[i][i] += mwi * (-dFw_dzi);
              elementJacobian_w_n[i][j] += mwi * (-dFw_dzj);
            }
            // Consistent STAB=2: lagged, PHASE-SPLIT graph viscosity on z.
            // Split into a GAS part and an AQUEOUS (dissolved-CO2) part so each
            // is gated like the flux branch it stabilizes:
            //   * gas part   -> multiplied by gate_old, so it vanishes across a
            //     sand->seal entry-pressure jump and does NOT diffuse free CO2
            //     through the seal (gas pools & spreads laterally as intended).
            //   * aqueous part -> NOT gated.  The aqueous flux Fa = tau*lam_a*dPhi_a
            //     has no gate (dissolved CO2 freely crosses the interface with the
            //     brine), so its stabilization must stay ON across interfaces too.
            //     Gating it (the old `*gate_old` on the whole dvg) switched off all
            //     z-dissipation exactly at sand interfaces, leaving the ungated
            //     aqueous advection unstabilized -> interface-aligned z (=> c)
            //     fingers in the dissolved tongues.  This split removes them while
            //     preserving the seal barrier for the gas.
            // Old-time coeff => constant in Newton => the +d/-d Jacobian is exact;
            // antisymmetric => global CO2 mass conserved.
            const double si_g = std::fabs(dlg_dz_o[i])*std::fabs(dPhi_g_old)
                              + std::fabs(lam_g_old_e[i])*std::fabs(dpc_dz_o[i]);
            const double si_a = std::fabs(dla_dz_o[i])*std::fabs(dPhi_a_old);
            const double sj_g = std::fabs(dlg_dz_o[j])*std::fabs(dPhi_g_old)
                              + std::fabs(lam_g_old_e[j])*std::fabs(dpc_dz_o[j]);
            const double sj_a = std::fabs(dla_dz_o[j])*std::fabs(dPhi_a_old);
            const double psi_edge = fmax(psi_n[zN_e[i]], psi_n[zN_e[j]]);
            const double dEV = tau * ( cE*psi_edge*fmax(si_g,sj_g)
                                     + cE*psi_edge*fmax(si_a,sj_a) );
            const double dLow = tau * ( fmax(si_g,sj_g) + fmax(si_a,sj_a) );
            const double dHi  = fmin(dEV, dLow);   // high-order target, clamped <= dLow
            const double dResid = (FCT_n == 1) ? dLow : dEV;
            if (dResid > 0.0) {
              const double Fv = dResid*(u_dof_n.data()[zN_e[j]] - u_dof_n.data()[zN_e[i]]);
              elementResidual_n[i] -= Fv;
              elementJacobian_n_n[i][i] += dResid;
              elementJacobian_n_n[i][j] -= dResid;
            }
            if (FCT_n == 1) {
              const int off_n = comp1_offset(zN_e[i], zN_e[j]);
              if (off_n >= 0) {
                const double dz = u_dof_n.data()[zN_e[j]] - u_dof_n.data()[zN_e[i]];
                dLow_n.data()[off_n]                 += dLow;
                dEV_n.data()[off_n]                  += dHi;
                dt_times_fH_minus_fL_n.data()[off_n] += dt * (dLow - dHi) * dz;
              }
            }
          }
        }
      }

      // -------- Distribute element arrays to global storage. --------
      for (int i = 0; i < nDOF_test_element; i++) {
        const int eN_i = eN * nDOF_test_element + i;
        const int gi   = u_l2g_n.data()[eN_i];   // comp-1 DOF (residual + budget)
        globalResidual.data()[offset_n + stride_n * gi] += elementResidual_n[i];
        if (have_gas_budget)
          gas_budget_node.data()[5 * numDOFs_n + gi] += elementResidual_n[i];
        // P1: comp-0 (H2O) two-sided water flux residual (Dirichlet-masked).
        const int fi_w = r_l2g.data()[eN_nDOF_trial_element + i];
        globalResidual.data()[offset_u + stride_u * fi_w] += bc_mask.data()[fi_w] * elementResidual_w[i];
        for (int j = 0; j < nDOF_trial_element; j++) {
          const int eN_i_j = eN_i * nDOF_trial_element + j;
          globalJacobian.data()[csrRowIndeces_n_n.data()[eN_i] + csrColumnOffsets_n_n.data()[eN_i_j]]
              += elementJacobian_n_n[i][j];
          globalJacobian.data()[csrRowIndeces_n_w.data()[eN_i] + csrColumnOffsets_n_w.data()[eN_i_j]]
              += elementJacobian_n_w[i][j];
          globalJacobian.data()[csrRowIndeces_w_w.data()[eN_i] + csrColumnOffsets_w_w.data()[eN_i_j]]
              += elementJacobian_w_w[i][j];
          globalJacobian.data()[csrRowIndeces_w_n.data()[eN_i] + csrColumnOffsets_w_n.data()[eN_i_j]]
              += elementJacobian_w_n[i][j];
        }
      }
    }

    for (int i_n = 0; i_n < numDOFs_n; i_n++) {
      mLow_n.data()[i_n]    = m_n_DOF[i_n];
      mDotLow_n.data()[i_n] = (m_n_DOF[i_n] - mn_n.data()[i_n]) / dt;
    }

    // DIAG: net gas-flux imbalance (Python prints + MPI-reduces). The T-asymmetry
    // probes [0]/[1] retired with the DOF-graph loop; [2]=sum F, [3]=sum|F|.
    if (gas_diag.size() >= 4) {
      gas_diag.data()[0] = 0.0;
      gas_diag.data()[1] = 0.0;
      gas_diag.data()[2] = diag_sumF;
      gas_diag.data()[3] = diag_absF;
    }


    // ============================================================================
    // Comp-1 (CO2 / z) exterior boundary loop -- STAB=2 path. COMPOSITIONAL.
    //
    // P3c STATUS: BOUNDARY PORTED (2026-06-06). Slot 1 is the overall CO2
    // composition z, NOT a saturation. This loop computes the compositional
    // molar CO2 trace flux F_1.n, mirroring the FD-verified interior element
    // flux (calculateResidual / calculateJacobian, lines ~1119/1942):
    //   F_1 = rho_g*Y*u_g + rho_a*X*u_a,
    //   u_a = -(K krw/mu_w)(grad p - rho_a_mass g),    p_a = p
    //   u_g = -(K krg/mu_g)(grad p + pc'(S_a) grad S_a - rho_g_mass g),
    //   grad S_a = -(dSg/dp grad p + dSg/dz grad z),
    // with every saturation-dependent property recomputed from the FLASH
    // saturation S_g(p,z) (psk closures take the wetting Se_a = (1-S_g-S_wr)/
    // (1-S_wr)). The surface term from integrating div(F_1) by parts is
    // +(F_1.n) N_i dS, so the residual mirrors the interior with gradN_i -> n.
    //
    // BC handling: a Nitsche-style penalty drives z at the trace toward the
    // prescribed bc_u_n_ext_b (= z_BC) on Dirichlet-z faces (isDir_n != 0;
    // McWhorter-Sunada inlet). No-Dirichlet faces are no-flow (F_1.n = 0) so a
    // closed box conserves mass.
    //
    // Jacobian contributions (chain rule through the analytic flash; interior
    // gradN_i replaced by the boundary normal n_I):
    //   (1,1) self  : d(F_1.n)/dz  (value-block * trial_j + grad-block . gradN_j)
    //                 + Dirichlet penalty * trial_j.
    //   (1,0) cross : d(F_1.n)/dp  (value-block * trial_j + grad-block . gradN_j).
    //
    // NOTE: structurally a consistent CG/Nitsche trace flux (lambda at the
    // trace, no upwind), not the interior edge-based upwind potential flux. A
    // ghost-node TPFA boundary (F_b = tau_b lambda_up (Phi_BC - Phi_trace)) is
    // the eventual sharper form; see [[m_comp_co2_stab2_upwind_potential_flux]].
    // Closure dispatch uses PSK_TYPE_member set at the top of this routine.
    // ============================================================================
    for (int ebNE = 0; ebNE < nExteriorElementBoundaries_global; ebNE++) {
      const int ebN = exteriorElementBoundariesArray.data()[ebNE];
      const int eN  = elementBoundaryElementsArray.data()[ebN * 2 + 0];
      const int ebN_local = elementBoundaryLocalElementBoundariesArray.data()[ebN * 2 + 0];
      const int eN_nDOF_trial_element = eN * nDOF_trial_element;
      const int    mat_eN    = elementMaterialTypes.data()[eN];
      const double phi_eN    = thetaR.data()[mat_eN] + thetaSR.data()[mat_eN];
      const double alpha_eN  = alpha.data()[mat_eN];
      const double krn_end_eN = krn_end.data()[mat_eN];
      const double n_vg_eN   = n.data()[mat_eN];
      const double *KWs_eN   = &KWs.data()[mat_eN * nnz];
      const double S_wr_loc      = thetaR.data()[mat_eN] / phi_eN;
      const double one_m_Sr_loc  = 1.0 - S_wr_loc;
      const double Se_trap_L4846 = 1.0 - S_gr.data()[mat_eN] / one_m_Sr_loc;  // gas-only residual trapping

      double elementResidual_n_eb[nDOF_test_element];
      double elementJacobian_n_n_eb[nDOF_test_element][nDOF_trial_element];
      double elementJacobian_n_w_eb[nDOF_test_element][nDOF_trial_element];
      for (int i = 0; i < nDOF_test_element; i++) {
        elementResidual_n_eb[i] = 0.0;
        for (int j = 0; j < nDOF_trial_element; j++) {
          elementJacobian_n_n_eb[i][j] = 0.0;
          elementJacobian_n_w_eb[i][j] = 0.0;
        }
      }

      for (int kb = 0; kb < nQuadraturePoints_elementBoundary; kb++) {
        const int ebNE_kb            = ebNE * nQuadraturePoints_elementBoundary + kb;
        const int ebN_local_kb       = ebN_local * nQuadraturePoints_elementBoundary + kb;
        const int ebN_local_kb_nSpace = ebN_local_kb * nSpace;

        double jac_ext[nSpace * nSpace], jacDet_ext, jacInv_ext[nSpace * nSpace];
        double boundaryJac_b[nSpace * (nSpace - 1)];
        double metricTensor_b[(nSpace - 1) * (nSpace - 1)];
        double metricTensorDetSqrt_b, dS_eb, normal_b[3];
        double xt_b, yt_b, zt_b, integralScaling_b;
        double x_eb, y_eb, z_eb;
        ck.calculateMapping_elementBoundary(eN, ebN_local, kb, ebN_local_kb,
            mesh_dof.data(), mesh_l2g.data(), mesh_trial_trace_ref.data(),
            mesh_grad_trial_trace_ref.data(), boundaryJac_ref.data(),
            jac_ext, jacDet_ext, jacInv_ext, boundaryJac_b, metricTensor_b,
            metricTensorDetSqrt_b, normal_ref.data(), normal_b,
            x_eb, y_eb, z_eb);
        ck.calculateMappingVelocity_elementBoundary(eN, ebN_local, kb, ebN_local_kb,
            mesh_velocity_dof.data(), mesh_l2g.data(), mesh_trial_trace_ref.data(),
            xt_b, yt_b, zt_b, normal_b, boundaryJac_b, metricTensor_b,
            integralScaling_b);
        dS_eb = ((1.0 - MOVING_DOMAIN) * metricTensorDetSqrt_b
                + MOVING_DOMAIN * integralScaling_b) * dS_ref.data()[kb];

        // Trace solution and gradients.
        double u_grad_trial_trace_b[nDOF_trial_element * nSpace];
        ck.gradTrialFromRef(
            &u_grad_trial_trace_ref.data()[ebN_local_kb_nSpace * nDOF_trial_element],
            jacInv_ext, u_grad_trial_trace_b);
        double u_w_ext_b = 0.0, u_n_ext_b = 0.0;
        double grad_u_w_ext_b[nSpace], grad_u_n_ext_b[nSpace];
        ck.valFromDOF(u_dof.data(), &u_l2g.data()[eN_nDOF_trial_element],
                      &u_trial_trace_ref.data()[ebN_local_kb * nDOF_test_element],
                      u_w_ext_b);
        ck.valFromDOF(u_dof_n.data(), &u_l2g.data()[eN_nDOF_trial_element],
                      &u_trial_trace_ref.data()[ebN_local_kb * nDOF_test_element],
                      u_n_ext_b);
        ck.gradFromDOF(u_dof.data(), &u_l2g.data()[eN_nDOF_trial_element],
                       u_grad_trial_trace_b, grad_u_w_ext_b);
        ck.gradFromDOF(u_dof_n.data(), &u_l2g.data()[eN_nDOF_trial_element],
                       u_grad_trial_trace_b, grad_u_n_ext_b);

        // Dirichlet target for u_n (no-op when isDOFBoundary_n == 0).
        const int    isDir_n      = isDOFBoundary_n.data()[ebNE_kb];
        const double bc_u_n_ext_b = isDir_n * ebqe_bc_u_n_ext.data()[ebNE_kb]
                                  + (1 - isDir_n) * u_n_ext_b;

        
        const double p_clb = fmax(u_w_ext_b, 1.0e2);
        ::m_comp_co2::flash::FlashState fsb =
            ::m_comp_co2::flash::flashPZ(p_clb, u_n_ext_b, T_C_member, m_NaCl_member, ::m_comp_co2::flash::EPS_Z, immiscible_member);
        double S_g_b = fsb.S_g, dSgb_dp = fsb.dS_g_dp, dSgb_dz = fsb.dS_g_dz;
        // Per-rock closure: with vertex (Lobatto) boundary quadrature this point is
        // a face vertex; at an interface node read this element's rock's S_r.
        if (per_rock) {
          int a_v = -1;
          for (int a = 0; a < nDOF_trial_element; a++)
            if (u_trial_trace_ref.data()[ebN_local_kb * nDOF_test_element + a] > 1.0 - 1.0e-12) a_v = a;
          if (a_v < 0)
            throw std::runtime_error("m_comp_co2: per_rock_closure needs vertex (Lobatto) boundary quadrature");
          const int gv = u_l2g.data()[eN_nDOF_trial_element + a_v];
          if (node_multi[gv]) {
            const int o = gv * nRock + mat_eN;
            S_g_b = rS[o];  dSgb_dp = rS_dp[o];  dSgb_dz = rS_dz[o];
          }
        }
        const double Sa_b = 1.0 - S_g_b;
        // wetting effective saturation from the FLASH saturation + (p,z) derivs.
        const double Se_raw_b = (Sa_b - S_wr_loc) / one_m_Sr_loc;
        double Se_b, dSe_dp_b, dSe_dz_b;
        if (Se_raw_b <= 0.0)      { Se_b = 0.0; dSe_dp_b = 0.0; dSe_dz_b = 0.0; }
        else if (Se_raw_b >= 1.0) { Se_b = 1.0; dSe_dp_b = 0.0; dSe_dz_b = 0.0; }
        else { Se_b = Se_raw_b; dSe_dp_b = -dSgb_dp/one_m_Sr_loc; dSe_dz_b = -dSgb_dz/one_m_Sr_loc; }
        double KWr_b=0.0, DKWr_b=0.0, thW_b=0.0, DthW_b=0.0, KNr_b=0.0, DKNr_b=0.0;
        double pc_b=0.0, dpc_dSe_b=0.0, d2pc_b=0.0;
        if (PSK_TYPE_member == 1) {
          proteus::m_comp_co2::psk::bc_wetting_from_Se(Se_b, alpha_eN, n_vg_eN,
              thetaR.data()[mat_eN], thetaSR.data()[mat_eN], thW_b, DthW_b, KWr_b, DKWr_b);
          proteus::m_comp_co2::psk::bc_kr_nonwetting_from_Se(Se_b, alpha_eN, n_vg_eN, KNr_b, DKNr_b, Se_trap_L4846);
          proteus::m_comp_co2::psk::bc_pc_from_Se(Se_b, alpha_eN, n_vg_eN, bc_Se_min.data()[mat_eN], pc_b, dpc_dSe_b, d2pc_b);
        } else {
          proteus::m_comp_co2::psk::vgm_wetting_from_Se(Se_b, alpha_eN, n_vg_eN,
              thetaR.data()[mat_eN], thetaSR.data()[mat_eN], thW_b, DthW_b, KWr_b, DKWr_b);
          proteus::m_comp_co2::psk::vgm_kr_nonwetting_from_Se(Se_b, alpha_eN, n_vg_eN, KNr_b, DKNr_b, Se_trap_L4846);
          proteus::m_comp_co2::psk::vgm_pc_from_Se(Se_b, alpha_eN, n_vg_eN, pc_b, dpc_dSe_b, d2pc_b);
        }
        KNr_b *= krn_end_eN;  DKNr_b *= krn_end_eN;
        const double pcp_b     = dpc_dSe_b / one_m_Sr_loc;            // pc'(S_a)
        const double dpcp_dp_b = (d2pc_b / one_m_Sr_loc) * dSe_dp_b;  // d pc'(S_a)/dp
        const double dpcp_dz_b = (d2pc_b / one_m_Sr_loc) * dSe_dz_b;
        // mass densities for gravity (molar density * mean molar mass) + derivs.
        const double dMm_b = ::m_comp_co2::eos::M_CO2_KG - ::m_comp_co2::eos::M_H2O_KG;
        const double Mbar_g_b = fsb.Y*::m_comp_co2::eos::M_CO2_KG + (1.0-fsb.Y)*::m_comp_co2::eos::M_H2O_KG;
        const double Mbar_a_b = fsb.X*::m_comp_co2::eos::M_CO2_KG + (1.0-fsb.X)*::m_comp_co2::eos::M_H2O_KG;
        const double rho_g_mass_b = fsb.rho_g*Mbar_g_b, rho_a_mass_b = fsb.rho_a*Mbar_a_b;
        const double drgm_dp_b = fsb.drho_g_dp*Mbar_g_b + fsb.rho_g*fsb.dY_dp*dMm_b;
        const double drgm_dz_b =                          fsb.rho_g*fsb.dY_dz*dMm_b;
        const double dram_dp_b = fsb.drho_a_dp*Mbar_a_b + fsb.rho_a*fsb.dX_dp*dMm_b;
        const double dram_dz_b = fsb.drho_a_dz*Mbar_a_b + fsb.rho_a*fsb.dX_dz*dMm_b;
        // CO2 transport coefficients Ag=rho_g*Y, Aa=rho_a*X + (p,z) derivatives.
        const double Ag = fsb.rho_g*fsb.Y, Aa = fsb.rho_a*fsb.X;
        const double dAg_dp = fsb.drho_g_dp*fsb.Y + fsb.rho_g*fsb.dY_dp;
        const double dAg_dz =                        fsb.rho_g*fsb.dY_dz;
        const double dAa_dp = fsb.drho_a_dp*fsb.X + fsb.rho_a*fsb.dX_dp;
        const double dAa_dz = fsb.drho_a_dz*fsb.X + fsb.rho_a*fsb.dX_dz;
        // per-direction Darcy velocities ug[I], ua[I] + value-block partials
        // (gradients held fixed). Mobilities carry the 1/mu_n (gas) factor.
        double ug_b[nSpace], ua_b[nSpace];
        double dug_dp_b[nSpace], dug_dz_b[nSpace], dua_dp_b[nSpace], dua_dz_b[nSpace];
        for (int I = 0; I < nSpace; I++) {
          double ugI=0.0, uaI=0.0, dugp=0.0, dugz=0.0, duap=0.0, duaz=0.0;
          for (int ii = a_rowptr.data()[I]; ii < a_rowptr.data()[I + 1]; ii++) {
            const int J = a_colind.data()[ii];
            const double Kii = KWs_eN[ii];
            const double Mob_g = KNr_b*Kii/mu_n, Mob_a = KWr_b*Kii;
            const double dMobg_dp = (DKNr_b*Kii/mu_n)*dSe_dp_b, dMobg_dz = (DKNr_b*Kii/mu_n)*dSe_dz_b;
            const double dMoba_dp = (DKWr_b*Kii)*dSe_dp_b,        dMoba_dz = (DKWr_b*Kii)*dSe_dz_b;
            const double gJ = gravity.data()[J];
            const double gradSa = -(dSgb_dp*grad_u_w_ext_b[J] + dSgb_dz*grad_u_n_ext_b[J]);
            const double gp_a = grad_u_w_ext_b[J] - rho_a_mass_b*gJ;
            const double gp_g = grad_u_w_ext_b[J] + pcp_b*gradSa - rho_g_mass_b*gJ;
            ugI -= Mob_g*gp_g;  uaI -= Mob_a*gp_a;
            const double dgradSa_dp = -(fsb.d2S_g_dp2 *grad_u_w_ext_b[J] + fsb.d2S_g_dpdz*grad_u_n_ext_b[J]);
            const double dgradSa_dz = -(fsb.d2S_g_dpdz*grad_u_w_ext_b[J] + fsb.d2S_g_dz2 *grad_u_n_ext_b[J]);
            const double dgpg_dp = dpcp_dp_b*gradSa + pcp_b*dgradSa_dp - drgm_dp_b*gJ;
            const double dgpg_dz = dpcp_dz_b*gradSa + pcp_b*dgradSa_dz - drgm_dz_b*gJ;
            dugp -= dMobg_dp*gp_g + Mob_g*dgpg_dp;
            dugz -= dMobg_dz*gp_g + Mob_g*dgpg_dz;
            duap -= dMoba_dp*gp_a + Mob_a*(-dram_dp_b*gJ);
            duaz -= dMoba_dz*gp_a + Mob_a*(-dram_dz_b*gJ);
          }
          ug_b[I]=ugI; ua_b[I]=uaI;
          dug_dp_b[I]=dugp; dug_dz_b[I]=dugz; dua_dp_b[I]=duap; dua_dz_b[I]=duaz;
        }

        // F_1 . n at this QP (consistent flux, before the penalty term which
        // depends only on the test-function row).
        double F_n_dot_n = 0.0;
        for (int I = 0; I < nSpace; I++) {
          F_n_dot_n += (Ag*ug_b[I] + Aa*ua_b[I]) * normal_b[I];
        }
        // value-block scalars dotted with the normal (interior gradN_i -> n_I).
        double Sval_p_b = 0.0, Sval_z_b = 0.0;
        for (int I = 0; I < nSpace; I++) {
          Sval_p_b += (dAg_dp*ug_b[I] + Ag*dug_dp_b[I] + dAa_dp*ua_b[I] + Aa*dua_dp_b[I]) * normal_b[I];
          Sval_z_b += (dAg_dz*ug_b[I] + Ag*dug_dz_b[I] + dAa_dz*ua_b[I] + Aa*dua_dz_b[I]) * normal_b[I];
        }
        // IIPG penalty scaled by the comp-1 diffusion magnitude a_n (= rho_g*Y*
        // krn/mu_n + rho_a*X*krw, times a representative K/mu_w; frozen in the
        // Jacobian). The bare framework penalty (const/h) is ~1e4x too weak vs
        // the molar-density-scaled comp-1 equation, so z floats off the BC.
        double Kw_rep = 0.0;
        for (int ii = 0; ii < nnz; ii++) Kw_rep = fmax(Kw_rep, fabs(KWs_eN[ii]));
        const double a_n_scale = (Ag*KNr_b/mu_n + Aa*KWr_b) * Kw_rep;
        const double penalty = ebqe_penalty_ext.data()[ebNE_kb] * a_n_scale;
        if (isDir_n) {
          // Nitsche penalty drives z at the trace toward the prescribed BC.
          F_n_dot_n += penalty * (u_n_ext_b - bc_u_n_ext_b);
        } else {
          // No Dirichlet on z => no-flow / closed boundary for the CO2 eq.
          // The consistent interior-trace flux must NOT be applied here: it
          // is generally nonzero (gravity + capillary at the trace) and would
          // inject a spurious boundary flux that breaks mass conservation on
          // a closed box. This mirrors STAB=0's calculateResidual, whose
          // exterior loop adds nothing to the gas equation. Only true
          // Dirichlet faces get the consistent flux + Nitsche penalty.
          F_n_dot_n = 0.0;
        }

        for (int i = 0; i < nDOF_test_element; i++) {
          const double test_i_dS = u_test_trace_ref.data()[
              ebN_local_kb * nDOF_test_element + i] * dS_eb;
          // Residual contribution: +(F_1.n) N_i dS.
          elementResidual_n_eb[i] += F_n_dot_n * test_i_dS;
          // Jacobian (per trial j).  d(F_1.n)/du_j = value-block * trial_j +
          // gradient-block . gradN_j, the interior chain rule with gradN_i->n.
          for (int j = 0; j < nDOF_trial_element; j++) {
            const double trial_j_b = u_trial_trace_ref.data()[
                ebN_local_kb * nDOF_test_element + j];
            // gradient-block scalars dotted with the normal:
            //   sum_{I,ii} (dF_1[I]/dgrad_var[J]) gradN_j[J] n_I.
            double Sgrad_p_b = 0.0, Sgrad_z_b = 0.0;
            for (int I = 0; I < nSpace; I++) {
              for (int ii = a_rowptr.data()[I]; ii < a_rowptr.data()[I + 1]; ii++) {
                const int J = a_colind.data()[ii];
                const double Kii = KWs_eN[ii];
                const double Mob_g = KNr_b*Kii/mu_n, Mob_a = KWr_b*Kii;
                const double gNjJ = u_grad_trial_trace_b[j * nSpace + J];
                // d ug[I]/d grad_p[J] = -Mob_g*(1 - pcp*dSg_dp); d ua[I]/d grad_p[J] = -Mob_a
                const double dFdgp = Ag*(-Mob_g*(1.0 - pcp_b*dSgb_dp)) + Aa*(-Mob_a);
                // d ug[I]/d grad_z[J] = -Mob_g*(-pcp*dSg_dz); d ua[I]/d grad_z[J] = 0
                const double dFdgz = Ag*(-Mob_g*(-pcp_b*dSgb_dz));
                Sgrad_p_b += dFdgp * gNjJ * normal_b[I];
                Sgrad_z_b += dFdgz * gNjJ * normal_b[I];
              }
            }
            // (1,1) self: d(F_1.n)/dz ; (1,0) cross: d(F_1.n)/dp.
            double jac_nn = Sval_z_b * trial_j_b + Sgrad_z_b;
            double jac_nw = Sval_p_b * trial_j_b + Sgrad_p_b;
            // Same no-flow gate as the residual: only Dirichlet faces
            // contribute a boundary flux (consistent flux + Nitsche penalty).
            if (isDir_n) {
              jac_nn += penalty * trial_j_b;
            } else {
              jac_nn = 0.0;
              jac_nw = 0.0;
            }
            elementJacobian_n_n_eb[i][j] += jac_nn * test_i_dS;
            elementJacobian_n_w_eb[i][j] += jac_nw * test_i_dS;
          }
        }
      } // kb

      // Scatter element-boundary contributions to global storage.
      for (int i = 0; i < nDOF_test_element; i++) {
        const int eN_i = eN * nDOF_test_element + i;
        const int gi   = u_l2g.data()[eN_i];
        globalResidual.data()[offset_n + stride_n * gi] += elementResidual_n_eb[i];
        if (have_gas_budget) {
          gas_budget_node.data()[4 * numDOFs_n + gi] += elementResidual_n_eb[i];
          gas_budget_node.data()[5 * numDOFs_n + gi] += elementResidual_n_eb[i];
        }
        for (int j = 0; j < nDOF_trial_element; j++) {
          const int ebN_i_j = ebN * 4 * nDOF_test_X_trial_element
                            + i * nDOF_trial_element + j;
          globalJacobian.data()[csrRowIndeces_n_n.data()[eN_i]
              + csrColumnOffsets_eb_n_n.data()[ebN_i_j]]
              += elementJacobian_n_n_eb[i][j];
          globalJacobian.data()[csrRowIndeces_n_w.data()[eN_i]
              + csrColumnOffsets_eb_n_w.data()[ebN_i_j]]
              += elementJacobian_n_w_eb[i][j];
        }
      }
    } // ebNE
  }

  void invert(arguments_dict &args)
  {
    proteus::pyarray<int>    &a_rowptr             = args.array<int>("a_rowptr");
    proteus::pyarray<int>    &a_colind             = args.array<int>("a_colind");
    double               rho                  = args.scalar<double>("rho");        // freshwater reference (fallback)
    double               beta                 = args.scalar<double>("beta");
    proteus::pyarray<double> &gravity              = args.array<double>("gravity");
    proteus::pyarray<double> &alpha                = args.array<double>("alpha");
    proteus::pyarray<double> &n                    = args.array<double>("n");
    proteus::pyarray<double> &thetaR               = args.array<double>("thetaR");
    proteus::pyarray<double> &thetaSR              = args.array<double>("thetaSR");
    proteus::pyarray<double> &KWs                  = args.array<double>("KWs");
    proteus::pyarray<double> &krn_end              = args.array<double>("krn_end");
    proteus::pyarray<double> &S_gr                 = args.array<double>("S_gr");
    double               mu_n                 = args.scalar<double>("mu_n");
    proteus::pyarray<int>    &elementMaterialTypes = args.array<int>("elementMaterialTypes");
    proteus::pyarray<int>    &freeDOFMaterialTypes = args.array<int>("freeDOFMaterialTypes");
    int                  numDOFs              = args.scalar<int>("numDOFs");
    proteus::pyarray<double> &mIn  = args.array<double>("limited_solution");
    proteus::pyarray<double> &pOut = args.array<double>("u_dof");
    int                  USE_NEWTON_INVERT = args.scalar<int>("USE_NEWTON_INVERT");
    const int            PSK_TYPE          = args.scalar<int>("PSK_TYPE");
    const int            COMPONENT         = args.scalar<int>("COMPONENT");

    if (COMPONENT == 1) {
      // -------- Component-1 inverse (compositional): m_c -> u_n = z. --------
      // m_c = (phi*N)*z  =>  z = m_c / (phi*N).  rho_n_phi_dof_member caches the
      // lumped phi*N built by the residual / MassMatrix.  Clamp z to [0,1]
      // (overall CO2 mole fraction), NOT a saturation range.
      const int numDOFs_u = static_cast<int>(pOut.size());
      const bool have_rho_n_phi = (rho_n_phi_dof_member.size() ==
                                   static_cast<std::size_t>(numDOFs_u));
      // Fallback divisor (safety net only -- the residual/MassMatrix always
      // populate the cache first). A representative aqueous-rich molar density
      // keeps the magnitude sane if an orphan node ever reaches here.
      const int    mat0    = elementMaterialTypes.data()[0];
      const double phi_mat = thetaR.data()[mat0] + thetaSR.data()[mat0];
      const double phiN_fallback = phi_mat * 5.0e4;   // ~brine molar density [mol/m^3]
      for (int i = 0; i < numDOFs_u; i++) {
        const double phiN_i = have_rho_n_phi ? rho_n_phi_dof_member[i]
                                             : phiN_fallback;
        double z = mIn.data()[i] / std::max(phiN_i, 1.0e-16);
        // Safety clamp to [0,1]. A correct conservative FCT keeps
        // limited_solution_n in-bounds, so this should not trigger; it guards
        // against round-off / misconfiguration.
        if (z < 0.0) z = 0.0;
        if (z > 1.0) z = 1.0;
        pOut.data()[i] = z;
      }
      return;
    }

    // -------- Component-0 inverse: m_w -> p_w (Richards-style, 2-stage). --------
    // m_w = rho_w(p_w) * thetaW(Se(S_n)),  rho_w(p_w) = rho * exp(beta*p_w).
    // "Ill-posed for p_w ALONE" only means: you cannot recover p_w without S_n.
    // But S_n is the OTHER primary variable -- always available -- so:
    //   stage 1:  rho_w = m_w / thetaW(Se(S_n))     (S_n known => thetaW known)
    //   stage 2:  p_w   = ln(rho_w / rho) / beta    (analytic inverse of EOS)
    // Well-posed whenever beta != 0 and thetaW > 0 (always: S_n <= 1-S_wr).
    // The (limited) S_n field is passed as "u_dof_n".
    if (COMPONENT == 0) {
      if (beta == 0.0)
        throw std::runtime_error(
            "m_comp_co2::invert COMPONENT=0: beta == 0 makes rho_w(p_w) "
            "constant, so m_w carries no p_w information. Use beta > 0.");
      proteus::pyarray<double> &u_dof_n = args.array<double>("u_dof_n");
      const int numDOFs_w = static_cast<int>(pOut.size());
      // Material 0 fallback -- matches the COMPONENT==1 convention above.
      const int    mat0     = elementMaterialTypes.data()[0];
      const double alpha0   = alpha.data()[mat0];
      const double n_vg0    = n.data()[mat0];
      const double thetaR0  = thetaR.data()[mat0];
      const double thetaSR0 = thetaSR.data()[mat0];
      const double S_wr0    = thetaR0 / (thetaR0 + thetaSR0);
      const double one_m_Sr = 1.0 - S_wr0;
      for (int i = 0; i < numDOFs_w; i++) {
        const double S_n = u_dof_n.data()[i];
        double Se = (1.0 - S_n - S_wr0) / one_m_Sr;
        if (Se < 0.0) Se = 0.0; else if (Se > 1.0) Se = 1.0;
        double thetaW, DthetaW_DSe, KWr, DKWr_DSe;
        if (PSK_TYPE == 1)
          proteus::m_comp_co2::psk::bc_wetting_from_Se(
              Se, alpha0, n_vg0, thetaR0, thetaSR0, thetaW, DthetaW_DSe, KWr, DKWr_DSe);
        else
          proteus::m_comp_co2::psk::vgm_wetting_from_Se(
              Se, alpha0, n_vg0, thetaR0, thetaSR0, thetaW, DthetaW_DSe, KWr, DKWr_DSe);
        const double rho_w = (thetaW > 1.0e-14) ? mIn.data()[i] / thetaW : rho;
        pOut.data()[i] = std::log(std::max(rho_w / rho, 1.0e-300)) / beta;
      }
      (void)a_rowptr; (void)a_colind; (void)gravity; (void)KWs;
      (void)freeDOFMaterialTypes; (void)numDOFs; (void)USE_NEWTON_INVERT;
      return;
    }

    throw std::runtime_error(
        "m_comp_co2::invert: COMPONENT must be 0 (m_w -> p_w) or 1 (m_n -> S_n).");
  }

  void calculateMassMatrix(arguments_dict &args)
  {
    //element
    double               dt                  = args.scalar<double>("dt");
    proteus::pyarray<double> &mesh_trial_ref      = args.array<double>("mesh_trial_ref");
    proteus::pyarray<double> &mesh_grad_trial_ref = args.array<double>("mesh_grad_trial_ref");
    proteus::pyarray<double> &mesh_dof            = args.array<double>("mesh_dof");
    proteus::pyarray<double> &mesh_velocity_dof   = args.array<double>("mesh_velocity_dof");
    double               MOVING_DOMAIN       = args.scalar<double>("MOVING_DOMAIN");
    proteus::pyarray<int>    &mesh_l2g            = args.array<int>("mesh_l2g");
    proteus::pyarray<double> &dV_ref              = args.array<double>("dV_ref");
    proteus::pyarray<double> &u_trial_ref         = args.array<double>("u_trial_ref");
    proteus::pyarray<double> &u_grad_trial_ref    = args.array<double>("u_grad_trial_ref");
    proteus::pyarray<double> &u_test_ref          = args.array<double>("u_test_ref");
    proteus::pyarray<double> &u_grad_test_ref     = args.array<double>("u_grad_test_ref");
    //element boundary
    proteus::pyarray<double> &mesh_trial_trace_ref      = args.array<double>("mesh_trial_trace_ref");
    proteus::pyarray<double> &mesh_grad_trial_trace_ref = args.array<double>("mesh_grad_trial_trace_ref");
    proteus::pyarray<double> &dS_ref                    = args.array<double>("dS_ref");
    proteus::pyarray<double> &u_trial_trace_ref         = args.array<double>("u_trial_trace_ref");
    proteus::pyarray<double> &u_grad_trial_trace_ref    = args.array<double>("u_grad_trial_trace_ref");
    proteus::pyarray<double> &u_test_trace_ref          = args.array<double>("u_test_trace_ref");
    proteus::pyarray<double> &u_grad_test_trace_ref     = args.array<double>("u_grad_test_trace_ref");
    proteus::pyarray<double> &normal_ref                = args.array<double>("normal_ref");
    proteus::pyarray<double> &boundaryJac_ref           = args.array<double>("boundaryJac_ref");
    //physics
    int nElements_global = args.scalar<int>("nElements_global");
    //new
    proteus::pyarray<double> &ebqe_penalty_ext     = args.array<double>("ebqe_penalty_ext");
    proteus::pyarray<int>    &elementMaterialTypes = args.array<int>("elementMaterialTypes");
    proteus::pyarray<int>    &isSeepageFace        = args.array<int>("isSeepageFace");
    proteus::pyarray<int>    &a_rowptr             = args.array<int>("a_rowptr");
    proteus::pyarray<int>    &a_colind             = args.array<int>("a_colind");
    double               rho                  = args.scalar<double>("rho");
    double               beta                 = args.scalar<double>("beta");
    // gas-phase reference density and reference pressure for the linear EOS
    // rho_n_local(p_n) = rho_n * p_n / p_ref_n (p_ref_n>0) or constant rho_n.
    const double         rho_n_mm             = args.scalar<double>("rho_n");
    const double         p_ref_n              = args.scalar<double>("p_ref_n");
    const bool           rho_n_compressible   = (p_ref_n > 0.0);

    proteus::pyarray<double> &q_rho                = args.array<double>("q_rho");

    proteus::pyarray<double> &gravity              = args.array<double>("gravity");
    proteus::pyarray<double> &alpha                = args.array<double>("alpha");
    proteus::pyarray<double> &n                    = args.array<double>("n");
    proteus::pyarray<double> &thetaR               = args.array<double>("thetaR");
    proteus::pyarray<double> &thetaSR              = args.array<double>("thetaSR");
    proteus::pyarray<double> &KWs                  = args.array<double>("KWs");
    proteus::pyarray<double> &krn_end              = args.array<double>("krn_end");
    proteus::pyarray<double> &S_gr                 = args.array<double>("S_gr");
    double               mu_n                 = args.scalar<double>("mu_n");
    //end new
    double               useMetrics                                 = args.scalar<double>("useMetrics");
    double               alphaBDF                                   = args.scalar<double>("alphaBDF");
    int                  lag_shockCapturing                         = args.scalar<int>("lag_shockCapturing");
    double               shockCapturingDiffusion                    = args.scalar<double>("shockCapturingDiffusion");
    proteus::pyarray<int>    &u_l2g                                      = args.array<int>("u_l2g");
    // Comp-1 (z) map -- MUST match the residual's u_l2g_n so the mass / (1,1)
    // tangent is assembled on the same DOFs.
    proteus::pyarray<int>    &u_l2g_n                                   = args.array<int>("u_l2g_n");
    proteus::pyarray<int>    &r_l2g                                      = args.array<int>("r_l2g");
    proteus::pyarray<double> &elementDiameter                            = args.array<double>("elementDiameter");
    int                  degree_polynomial                          = args.scalar<int>("degree_polynomial");
    proteus::pyarray<double> &u_dof                                      = args.array<double>("u_dof");
    // u_dof_n always present in argsDict (getJacobian sets it from
    // self.u[1].dof in m_comp_co2.py).
    proteus::pyarray<double> &u_dof_n                                    = args.array<double>("u_dof_n");
    proteus::pyarray<double> &velocity                                   = args.array<double>("velocity");
    proteus::pyarray<double> &q_m_betaBDF                                = args.array<double>("q_m_betaBDF");
    proteus::pyarray<double> &cfl                                        = args.array<double>("cfl");
    proteus::pyarray<double> &q_numDiff_u_last                           = args.array<double>("q_numDiff_u_last");
    proteus::pyarray<int>    &csrRowIndeces_u_u                          = args.array<int>("csrRowIndeces_u_u");
    proteus::pyarray<int>    &csrColumnOffsets_u_u                       = args.array<int>("csrColumnOffsets_u_u");
    proteus::pyarray<int>    &csrRowIndeces_n_n                          = args.array<int>("csrRowIndeces_n_n");
    // (1,0) cross-block CSR maps for the gas-eq diffusion
    // against grad u_w. Currently allocated by the framework but unused by
    // the C++ assembly - 3c.2/3c.3 will write into them.
    proteus::pyarray<int>    &csrRowIndeces_n_w                          = args.array<int>("csrRowIndeces_n_w");
    proteus::pyarray<int>    &csrColumnOffsets_n_n                       = args.array<int>("csrColumnOffsets_n_n");
    proteus::pyarray<int>    &csrColumnOffsets_n_w                       = args.array<int>("csrColumnOffsets_n_w");
    proteus::pyarray<double> &globalJacobian                             = args.array<double>("globalJacobian");
    proteus::pyarray<double> &delta_x_ij                                 = args.array<double>("delta_x_ij");
    int                  nExteriorElementBoundaries_global          = args.scalar<int>("nExteriorElementBoundaries_global");
    proteus::pyarray<int>    &exteriorElementBoundariesArray             = args.array<int>("exteriorElementBoundariesArray");
    proteus::pyarray<int>    &elementBoundaryElementsArray               = args.array<int>("elementBoundaryElementsArray");
    proteus::pyarray<int>    &elementBoundaryLocalElementBoundariesArray = args.array<int>("elementBoundaryLocalElementBoundariesArray");
    proteus::pyarray<double> &ebqe_velocity_ext                          = args.array<double>("ebqe_velocity_ext");
    proteus::pyarray<int>    &isDOFBoundary_u                            = args.array<int>("isDOFBoundary_u");
    proteus::pyarray<double> &ebqe_bc_u_ext                              = args.array<double>("ebqe_bc_u_ext");
    // component-1 (S_n) boundary arrays.
    proteus::pyarray<int>    &isDOFBoundary_n                            = args.array<int>("isDOFBoundary_n");
    proteus::pyarray<double> &ebqe_bc_u_n_ext                            = args.array<double>("ebqe_bc_u_n_ext");
    proteus::pyarray<int>    &isFluxBoundary_u                           = args.array<int>("isFluxBoundary_u");
    proteus::pyarray<double> &ebqe_bc_flux_ext                           = args.array<double>("ebqe_bc_flux_ext");
    proteus::pyarray<int>    &csrColumnOffsets_eb_u_u                    = args.array<int>("csrColumnOffsets_eb_u_u");
    int                  LUMPED_MASS_MATRIX                         = args.scalar<int>("LUMPED_MASS_MATRIX");
    // PSK closure selector for evaluateCoefficients (read from argsDict).
    PSK_TYPE_member = args.scalar<int>("PSK_TYPE");
    immiscible_member = (args.scalar<int>("immiscible") != 0);
    T_C_member        = args.scalar<double>("T_C");      // temperature [degC] from input
    double Ct_sge = 4.0;
    //
    //loop over elements to compute volume integrals and load them into the element Jacobians and global Jacobian
    //
    for (int eN = 0; eN < nElements_global; eN++) {
      double elementJacobian_u_u[nDOF_test_element][nDOF_trial_element];
      for (int i = 0; i < nDOF_test_element; i++)
        for (int j = 0; j < nDOF_trial_element; j++) { elementJacobian_u_u[i][j] = 0.0; }
      for (int k = 0; k < nQuadraturePoints_element; k++) {
        int eN_k                  = eN * nQuadraturePoints_element + k, //index to a scalar at a quadrature point
          eN_k_nSpace             = eN_k * nSpace,
            eN_nDOF_trial_element = eN * nDOF_trial_element; //index to a vector at a quadrature point
        //declare local storage
        double u = 0.0, grad_u[nSpace], m = 0.0, dm = 0.0, f[nSpace], df[nSpace], a[nnz], da[nnz], as[nnz], m_t = 0.0, dm_t = 0.0, dpdeResidual_u_u[nDOF_trial_element], Lstar_u[nDOF_test_element], dsubgridError_u_u[nDOF_trial_element], tau = 0.0, tau0 = 0.0, tau1 = 0.0, jac[nSpace * nSpace], jacDet, jacInv[nSpace * nSpace], u_grad_trial[nDOF_trial_element * nSpace], dV, u_test_dV[nDOF_test_element], u_grad_test_dV[nDOF_test_element * nSpace], x, y, z, xt, yt, zt,
          G[nSpace * nSpace], G_dd_G, tr_G;

        //get jacobian, etc for mapping reference element
        ck.calculateMapping_element(eN, k, mesh_dof.data(), mesh_l2g.data(), mesh_trial_ref.data(), mesh_grad_trial_ref.data(), jac, jacDet, jacInv, x, y, z);
        ck.calculateMappingVelocity_element(eN, k, mesh_velocity_dof.data(), mesh_l2g.data(), mesh_trial_ref.data(), xt, yt, zt);
        //get the physical integration weight
        dV = fabs(jacDet) * dV_ref.data()[k];
        ck.calculateG(jacInv, G, G_dd_G, tr_G);
        //get the trial function gradients
        ck.gradTrialFromRef(&u_grad_trial_ref.data()[k * nDOF_trial_element * nSpace], jacInv, u_grad_trial);
        //get the solution
        ck.valFromDOF(u_dof.data(), &u_l2g.data()[eN_nDOF_trial_element], &u_trial_ref.data()[k * nDOF_trial_element], u);
        //get the solution gradients
        ck.gradFromDOF(u_dof.data(), &u_l2g.data()[eN_nDOF_trial_element], u_grad_trial, grad_u);
        //precalculate test function products with integration weights
        for (int j = 0; j < nDOF_trial_element; j++) {
          u_test_dV[j] = u_test_ref.data()[k * nDOF_trial_element + j] * dV;
          for (int I = 0; I < nSpace; I++) {
            u_grad_test_dV[j * nSpace + I] = u_grad_trial[j * nSpace + I] * dV; //cek warning won't work for Petrov-Galerkin
          }
        }
        //
        //calculate pde coefficients and derivatives at quadrature points
        //
        double Kr, dKr, thetaW;
        //const double rho_local = q_rho.data()[eN_k];
        // NOTE: the cek hack at line ~3170 (`dm = 1.0`) and the dm_t=1.0
        // override at the Jacobian assembly below make the final mass matrix
        // output independent of the coefficients here (result is just
        // M = integral N_i N_j dV regardless). Coefficient call is kept for
        // moving-mesh / VMS consistency.
        double dm_du_n_mm = 0.0, dkr_du_n_mm = 0.0;
        double df_du_n_mm[nSpace], da_du_n_mm[nnz];
        for (int I = 0; I < nSpace; I++) df_du_n_mm[I] = 0.0;
        for (int ii = 0; ii < nnz; ii++) da_du_n_mm[ii] = 0.0;
        double u_n_qp = 0.0;
        ck.valFromDOF(u_dof_n.data(),
                      &u_l2g_n.data()[eN_nDOF_trial_element],
                      &u_trial_ref.data()[k * nDOF_trial_element], u_n_qp);
        evaluateCoefficients_from_Se(a_rowptr.data(), a_colind.data(), rho, q_rho.data()[eN_k], beta, gravity.data(),
                                     alpha.data()[elementMaterialTypes.data()[eN]], n.data()[elementMaterialTypes.data()[eN]],
                                     thetaR.data()[elementMaterialTypes.data()[eN]], thetaSR.data()[elementMaterialTypes.data()[eN]],
                                     &KWs.data()[elementMaterialTypes.data()[eN] * nnz], u, u_n_qp,
                                     m, dm, dm_du_n_mm, f, df, df_du_n_mm, a, da, da_du_n_mm,
                                     as, Kr, dKr, dkr_du_n_mm, thetaW);
        //
        //moving mesh
        //
        double mesh_velocity[3];
        mesh_velocity[0] = xt;
        mesh_velocity[1] = yt;
        mesh_velocity[2] = zt;
        for (int I = 0; I < nSpace; I++) {
          f[I] -= MOVING_DOMAIN * m * mesh_velocity[I];
          df[I] -= MOVING_DOMAIN * dm * mesh_velocity[I];
        }
        //
        //calculate time derivatives
        //
        //cek hack
        dm = 1.0;
        ck.bdf(alphaBDF,
               q_m_betaBDF.data()[eN_k], //since m_t isn't used, we don't have to correct mass
               m, dm, m_t, dm_t);
        //
        //calculate subgrid error contribution to the Jacobian (strong residual, adjoint, jacobian of strong residual)
        //
        //calculate the adjoint times the test functions
        for (int i = 0; i < nDOF_test_element; i++) {
          int i_nSpace = i * nSpace;
          Lstar_u[i]   = ck.Advection_adjoint(df, &u_grad_test_dV[i_nSpace]);
        }
        //calculate the Jacobian of strong residual
        for (int j = 0; j < nDOF_trial_element; j++) {
          int j_nSpace        = j * nSpace;
          dpdeResidual_u_u[j] = ck.MassJacobian_strong(dm_t, u_trial_ref.data()[k * nDOF_trial_element + j]) + ck.AdvectionJacobian_strong(df, &u_grad_trial[j_nSpace]);
        }
        //tau and tau*Res
        calculateSubgridError_tau(elementDiameter.data()[eN], dm_t, df, cfl.data()[eN_k], tau0);

        calculateSubgridError_tau(Ct_sge, G, dm_t, df, tau1, cfl.data()[eN_k]);
        tau = useMetrics * tau1 + (1.0 - useMetrics) * tau0;

        for (int j = 0; j < nDOF_trial_element; j++) dsubgridError_u_u[j] = -tau * dpdeResidual_u_u[j];
        for (int i = 0; i < nDOF_test_element; i++) {
          for (int j = 0; j < nDOF_trial_element; j++) {
            if (LUMPED_MASS_MATRIX == 1) {
              if (i == j) elementJacobian_u_u[i][j] += u_test_dV[i];
            } else {
              int j_nSpace = j * nSpace;
              int i_nSpace = i * nSpace;
              dm_t = 1.0; //we are solving for continuum density explicitly
              elementJacobian_u_u[i][j] += ck.MassJacobian_weak(dm_t, u_trial_ref.data()[k * nDOF_trial_element + j], u_test_dV[i]);
            }
          } //j
        } //i
      } //k
      //
      //load into element Jacobian into global Jacobian
      //
      for (int i = 0; i < nDOF_test_element; i++) {
        int eN_i = eN * nDOF_test_element + i;
        int I    = u_l2g.data()[eN_i];
        for (int j = 0; j < nDOF_trial_element; j++) {
          int eN_i_j = eN_i * nDOF_trial_element + j;
          int J      = u_l2g.data()[eN * nDOF_trial_element + j];
          //globalJacobian.data()[csrRowIndeces_u_u.data()[eN_i] + csrColumnOffsets_u_u.data()[eN_i_j]] += elementJacobian_u_u[i][j];
          delta_x_ij.data()[3 * (csrRowIndeces_u_u.data()[eN_i] + csrColumnOffsets_u_u.data()[eN_i_j]) + 0] = mesh_dof.data()[I * 3 + 0] - mesh_dof.data()[J * 3 + 0];
          delta_x_ij.data()[3 * (csrRowIndeces_u_u.data()[eN_i] + csrColumnOffsets_u_u.data()[eN_i_j]) + 1] = mesh_dof.data()[I * 3 + 1] - mesh_dof.data()[J * 3 + 1];
          delta_x_ij.data()[3 * (csrRowIndeces_u_u.data()[eN_i] + csrColumnOffsets_u_u.data()[eN_i_j]) + 2] = mesh_dof.data()[I * 3 + 2] - mesh_dof.data()[J * 3 + 2];
        } //j
      } //i
    } //elements
    // (1,1) block: unit consistent mass / dt. Used by the time integrator.
    // Also refresh the cached nodal phi*rho_n so invert(COMPONENT=1) sees the
    // correct projection if mass-matrix assembly runs more recently than the
    // residual.
    const int numDOFs_u_mm = static_cast<int>(u_dof.size());
    // These are indexed by the comp-1 DOF (u_l2g_n), so they are sized by the
    // comp-1 DOF count u_dof_n.size(), not the comp-0 count u_dof.size().
    const int numDOFs_n_mm = static_cast<int>(u_dof_n.size());
    std::vector<double> rho_n_phi_mm(numDOFs_n_mm, 0.0);
    std::vector<double> ML_n_mm(numDOFs_n_mm, 0.0);
    for (int eN = 0; eN < nElements_global; eN++) {
      double elementJacobian_n_n[nDOF_test_element][nDOF_trial_element];
      for (int i = 0; i < nDOF_test_element; i++)
        for (int j = 0; j < nDOF_trial_element; j++) { elementJacobian_n_n[i][j] = 0.0; }
      const int    mat_eN_mm = elementMaterialTypes.data()[eN];
      const double phi_eN_mm = thetaR.data()[mat_eN_mm] + thetaSR.data()[mat_eN_mm];
      const double alpha_eN_mm = alpha.data()[mat_eN_mm];
      const double n_vg_eN_mm  = n.data()[mat_eN_mm];
      const double S_wr_mm     = thetaR.data()[mat_eN_mm] / phi_eN_mm;
      const double one_m_Sr_mm = 1.0 - S_wr_mm;
      for (int k = 0; k < nQuadraturePoints_element; k++) {
        const int eN_nDOF_trial_element = eN * nDOF_trial_element;
        double jac[nSpace * nSpace], jacDet, jacInv[nSpace * nSpace], x, y, z;
        ck.calculateMapping_element(eN, k, mesh_dof.data(), mesh_l2g.data(),
                                    mesh_trial_ref.data(), mesh_grad_trial_ref.data(),
                                    jac, jacDet, jacInv, x, y, z);
        const double dV = fabs(jacDet) * dV_ref.data()[k];
        // P1 (compositional): project phi*N (total molar density * porosity) from
        // the flash, IDENTICAL to the residual's rho_n_phi_dof, so invert(COMP=1)
        // recovers z = m_c/(phi*N) consistently.  N = rho_g*S_g + rho_a*S_a.
        double u_w_p = 0.0, u_n_p = 0.0;
        ck.valFromDOF(u_dof.data(),
                      &u_l2g.data()[eN_nDOF_trial_element],
                      &u_trial_ref.data()[k * nDOF_trial_element], u_w_p);
        ck.valFromDOF(u_dof_n.data(),
                      &u_l2g_n.data()[eN_nDOF_trial_element],
                      &u_trial_ref.data()[k * nDOF_trial_element], u_n_p);
        const double p_cl_mm = fmax(u_w_p, 1.0e2);
        ::m_comp_co2::flash::FlashState fs_mm =
            ::m_comp_co2::flash::flashPZ(p_cl_mm, u_n_p, T_C_member, m_NaCl_member, ::m_comp_co2::flash::EPS_Z, immiscible_member);
        const double phi_rho_n_qp = phi_eN_mm * (fs_mm.rho_g*fs_mm.S_g
                                               + fs_mm.rho_a*(1.0 - fs_mm.S_g));
        for (int i = 0; i < nDOF_test_element; i++) {
          const double test_i = u_test_ref.data()[k * nDOF_test_element + i];
          const int    gi     = u_l2g_n.data()[eN * nDOF_test_element + i];   // comp-1 DOF
          rho_n_phi_mm[gi] += phi_rho_n_qp * test_i * dV;
          ML_n_mm[gi]      += test_i * dV;
          for (int j = 0; j < nDOF_trial_element; j++) {
            const double trial_j = u_trial_ref.data()[k * nDOF_trial_element + j];
            elementJacobian_n_n[i][j] += (test_i * trial_j * dV) / dt;
          }
        }
      }
      for (int i = 0; i < nDOF_test_element; i++) {
        const int eN_i = eN * nDOF_test_element + i;
        for (int j = 0; j < nDOF_trial_element; j++) {
          const int eN_i_j = eN_i * nDOF_trial_element + j;
          const int full_offset = csrRowIndeces_n_n.data()[eN_i] + csrColumnOffsets_n_n.data()[eN_i_j];
          if (std::fabs(globalJacobian.data()[full_offset]) < 1.0e-14)
            globalJacobian.data()[full_offset] += elementJacobian_n_n[i][j];
        }
      }
    }
    for (int i = 0; i < numDOFs_n_mm; ++i) {
      if (ML_n_mm[i] > 0.0) rho_n_phi_mm[i] /= ML_n_mm[i];
      else rho_n_phi_mm[i] = thetaR.data()[0] + thetaSR.data()[0];
      rho_n_phi_mm[i] = std::max(rho_n_phi_mm[i], 1.0e-16);
    }
    rho_n_phi_dof_member = rho_n_phi_mm;
  } //computeMassMatrix

  void dissolutionFlash(arguments_dict &args)
  {
    proteus::pyarray<double> &c_dof   = args.array<double>("c_dof");    // c   (in/out)
    proteus::pyarray<double> &Sn_dof  = args.array<double>("Sn_dof");   // S_n (in/out)
    const double rho_n           = args.scalar<double>("rho_n");
    const double X_sat           = args.scalar<double>("X_sat");
    const double c_sat           = args.scalar<double>("c_sat");
    const double k_d             = args.scalar<double>("k_d");     // dissolution rate [1/time]
    const double dt              = args.scalar<double>("dt");      // step size
    const int    numDOFs         = args.scalar<int>("numDOFs");
    if (X_sat <= 0.0 || k_d <= 0.0)
      return;                                    // dissolution disabled
    const double a = X_sat / c_sat;              // dissolved CO2 per unit c
    for (int i = 0; i < numDOFs; i++)
      {
        const double Sn = Sn_dof.data()[i];
        const double c  = c_dof.data()[i];
        if (Sn > 0.0 && c < c_sat)
          {
            const double M    = rho_n * Sn + a * (1.0 - Sn) * c;
            const double c_eq = (M <= X_sat) ? (M / a) : c_sat;   // gas-limited; in [c, c_sat]
            const double r    = k_d * Sn * dt;
            const double frac = (r > 0.0) ? r / (1.0 + r) : 0.0;
            const double c_new = c + frac * (c_eq - c);           // in [c, c_eq]
            const double denom = rho_n - a * c_new;               // > 0 since a*c_new <= X_sat < rho_n
            double Sn_new = (denom != 0.0) ? (M - a * c_new) / denom : Sn;
            if (Sn_new < 0.0) Sn_new = 0.0;                       // safety (denom>0 keeps it >=0)
            c_dof.data()[i]  = c_new;
            Sn_dof.data()[i] = Sn_new;
          }
      }
    // --- PART 2: rebuild TADR's quadrature old-mass from the flashed fields ---
    proteus::pyarray<double> &q_m_tadr             = args.array<double>("q_m_tadr"); // TADR q[('m',0)] (out)
    proteus::pyarray<int>    &u_l2g                = args.array<int>("u_l2g");
    proteus::pyarray<double> &u_trial_ref          = args.array<double>("u_trial_ref");
    proteus::pyarray<int>    &elementMaterialTypes = args.array<int>("elementMaterialTypes");
    proteus::pyarray<double> &thetaR               = args.array<double>("thetaR");
    proteus::pyarray<double> &thetaSR              = args.array<double>("thetaSR");
    const double rho_f                        = args.scalar<double>("rho_f");
    const double rho_s                        = args.scalar<double>("rho_s");
    const int nElements_global                = args.scalar<int>("nElements_global");
    // Renamed (nQP_flash / nDOF_flash) to avoid shadowing the struct's
    // template parameters nQuadraturePoints_element / nDOF_trial_element.
    // These come from Python (derived from q[('m',0)].shape and the l2g),
    // so the q_m_tadr write stays in bounds.
    const int nQP_flash                       = args.scalar<int>("nQuadraturePoints_element");
    const int nDOF_flash                      = args.scalar<int>("nDOF_trial_element");
    const double eps = (rho_f != 0.0) ? (rho_s - rho_f) / rho_f : 0.0;
    for (int eN = 0; eN < nElements_global; eN++)
      {
        const int    mat    = elementMaterialTypes.data()[eN];
        const double phi_eN  = thetaR.data()[mat] + thetaSR.data()[mat];
        const int    eN_nDOF = eN * nDOF_flash;
        for (int k = 0; k < nQP_flash; k++)
          {
            double c_k = 0.0, Sn_k = 0.0;
            ck.valFromDOF(c_dof.data(),  &u_l2g.data()[eN_nDOF],
                          &u_trial_ref.data()[k * nDOF_flash], c_k);
            ck.valFromDOF(Sn_dof.data(), &u_l2g.data()[eN_nDOF],
                          &u_trial_ref.data()[k * nDOF_flash], Sn_k);
            const double thetaW = phi_eN * (1.0 - Sn_k);
            const double rho    = rho_f * (1.0 + eps * c_k);
            q_m_tadr.data()[eN * nQP_flash + k] = thetaW * rho * c_k;
          }
      }
  } //dissolutionFlash

  // ---- Post-step derived-field export ---------------------------------------
  // Per-node value-only flash of the PRIMARY (p,z) state into the three
  // human-facing compositional fields that the archiver writes to the XDMF:
  //   Sg = free-gas saturation               (FlashState.S_g)
  //   X  = CO2 mole fraction dissolved in brine  (FlashState.X)
  //   c  = brine CO2 MASS concentration [kg/m^3] = rho_a * X * M_CO2
  void calculateFlashFields(arguments_dict &args)
  {
    proteus::pyarray<double> &p_dof  = args.array<double>("p_dof");   // comp-0 (pressure) [Pa]
    proteus::pyarray<double> &z_dof  = args.array<double>("z_dof");   // comp-1 (overall CO2 mole frac)
    proteus::pyarray<double> &Sg_dof = args.array<double>("Sg_dof");  // out: gas saturation
    proteus::pyarray<double> &X_dof  = args.array<double>("X_dof");   // out: CO2 mole frac in brine
    proteus::pyarray<double> &c_dof  = args.array<double>("c_dof");   // out: brine CO2 mass conc [kg/m^3]
    const int numDOFs           = args.scalar<int>("numDOFs");
    immiscible_member = (args.scalar<int>("immiscible") != 0);
    T_C_member        = args.scalar<double>("T_C");      // temperature [degC] from input
    const double M_CO2 = 0.04401;                                // CO2 molar mass [kg/mol]
    for (int i = 0; i < numDOFs; i++)
      {
        const double p_i = (p_dof.data()[i] > 1.0e2) ? p_dof.data()[i] : 1.0e2;
        double z_i = z_dof.data()[i];
        if (z_i < 1.0e-12)        z_i = 1.0e-12;                  // match plot-side clamp
        if (z_i > 1.0 - 1.0e-12)  z_i = 1.0 - 1.0e-12;
        ::m_comp_co2::flash::FlashState fs =
            ::m_comp_co2::flash::flashPZ(p_i, z_i, T_C_member, m_NaCl_member,
                                         ::m_comp_co2::flash::EPS_Z,
                                         immiscible_member);
        Sg_dof.data()[i] = fs.S_g;
        X_dof.data()[i]  = (fs.X > 0.0) ? fs.X : 0.0;
        c_dof.data()[i]  = (fs.rho_a * fs.X > 0.0) ? fs.rho_a * fs.X * M_CO2 : 0.0;
      }
    // Per-rock closure: min / max of S_r over the rocks at each node (the
    // flash's S_g at one-rock nodes and before the first residual).
    if (args.scalar<int>("per_rock_closure")) {
      proteus::pyarray<double> &Sg_min = args.array<double>("Sg_rock_min_dof");
      proteus::pyarray<double> &Sg_max = args.array<double>("Sg_rock_max_dof");
      const bool have = (static_cast<int>(Sg_rock_min_member.size()) >= numDOFs);
      for (int i = 0; i < numDOFs; i++) {
        const bool multi = have && !std::isnan(Sg_rock_min_member[i]);
        Sg_min.data()[i] = multi ? Sg_rock_min_member[i] : Sg_dof.data()[i];
        Sg_max.data()[i] = multi ? Sg_rock_max_member[i] : Sg_dof.data()[i];
      }
    }
  } //calculateFlashFields
}; //M_comp_co2

inline M_comp_co2_base *newm_comp_co2(int nSpaceIn, int nQuadraturePoints_elementIn, int nDOF_mesh_trial_elementIn, int nDOF_trial_elementIn, int nDOF_test_elementIn, int nQuadraturePoints_elementBoundaryIn, int CompKernelFlag)
{
  if (nSpaceIn == 1)
    return proteus::chooseAndAllocateDiscretization1D<M_comp_co2_base, M_comp_co2, CompKernel>(nSpaceIn, nQuadraturePoints_elementIn, nDOF_mesh_trial_elementIn, nDOF_trial_elementIn, nDOF_test_elementIn, nQuadraturePoints_elementBoundaryIn, CompKernelFlag);
  else if (nSpaceIn == 2)
    return proteus::chooseAndAllocateDiscretization2D<M_comp_co2_base, M_comp_co2, CompKernel>(nSpaceIn, nQuadraturePoints_elementIn, nDOF_mesh_trial_elementIn, nDOF_trial_elementIn, nDOF_test_elementIn, nQuadraturePoints_elementBoundaryIn, CompKernelFlag);
  else {
    assert(nSpaceIn == 3);
    return proteus::chooseAndAllocateDiscretization<M_comp_co2_base, M_comp_co2, CompKernel>(nSpaceIn, nQuadraturePoints_elementIn, nDOF_mesh_trial_elementIn, nDOF_trial_elementIn, nDOF_test_elementIn, nQuadraturePoints_elementBoundaryIn, CompKernelFlag);
  }
}
} // namespace m_comp_co2
} // namespace proteus
#endif
