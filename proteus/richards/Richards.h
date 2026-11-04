#ifndef Richards_H
#define Richards_H
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <valarray>
#include <vector>
#include "CompKernel.h"
#include "ModelFactory.h"
#include "../pskRelations.h"
#include "../mprans/ArgumentsDict.h"
#include "pyarray.h"
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

namespace richards
{
//A/B switch for the weak-Dirichlet boundary block that feeds the HIGH order FCT flux fH.
//1 (default) -> fH's boundary block uses a_ext = rho*k_rw*Ks, matching the cell loop's `a`,
//               so it cancels against fL's boundary block (as_ext scaled by the nodal
//               upwind k_rw) in fA = fH - fL, exactly as the interior blocks do.
//0           -> the old behaviour: fH's boundary block uses as_ext = rho*Ks, leaving a
//               residue in fA of order 1/k_rw times the interior antidiffusion.
//Read once; only TransportMatrixConsistent depends on it, so FCT=False is unaffected either way.
inline bool richards_env_on(const char *name)
{
  const char *e = std::getenv(name);
  return (e == nullptr) || (std::strcmp(e, "0") != 0);
}
static const bool RICHARDS_FCT_BND_KRW = richards_env_on("RICHARDS_FCT_BND_KRW");
//Step B: keep the weak-Dirichlet block in its own matrix TB instead of folding it into
//TransportMatrix where the interior fmax(0,-T_ij) clip mangles it, and apply it
//identically in the low and the high order flux,
//  fH_ij = fL_ij = -TB_ij dPhi   (boundary block),
//so it cancels out of fA: the antidiffusive flux is the interior one alone, antisymmetric,
//and the limited correction conserves mass.  TB gets NO graph viscosity: one acting on
//dPhi (gravity included) is an O(1) inconsistency at the boundary rows that only the
//limiter could remove, and the limiter clips exactly there
//(test/salinity/mms_coupled/MMS_PLAN.md).
//The price is that the low order boundary rows are not an M-matrix.
//RICHARDS_FCT_BND_DIFF=0 restores the old routing (TB folded into TransportMatrix).
static const bool RICHARDS_FCT_BND_DIFF  = richards_env_on("RICHARDS_FCT_BND_DIFF");
static const bool RICHARDS_FCT_BND_RESET = richards_env_on("RICHARDS_FCT_BND_RESET");
//Hand the transport model the water flux this scheme really applies on the exterior
//boundary instead of the pointwise interior trace -(a/rho)(grad psi - rho_r g).n.  See
//the consBoundaryFlux comment in calculateResidual_entropy_viscosity for the algebra and
//for the numbers that motivated it.  RICHARDS_BC_CONS=0 restores the pointwise trace.
static const bool RICHARDS_BC_CONS       = richards_env_on("RICHARDS_BC_CONS");

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
} // namespace richards
} // namespace proteus
namespace proteus
{
namespace richards
{
class Richards_base {
  //The base class defining the interface
public:
  virtual ~Richards_base() { double anb_seepage_flux = 1e-16; }
  virtual void calculateResidual(arguments_dict &args)                   = 0;
  virtual void calculateJacobian(arguments_dict &args)                   = 0;
  virtual void invert(arguments_dict &args)                              = 0;
  virtual void FCTStep(arguments_dict &args)                             = 0;
  virtual void ImplicitFCTStep(arguments_dict &args)                     = 0;
  virtual void kth_FCT_step(arguments_dict &args)                        = 0;
  virtual void calculateResidual_entropy_viscosity(arguments_dict &args) = 0;
  virtual void calculateMassMatrix(arguments_dict &args)                 = 0;
  virtual void calculateEdgeVelocity(arguments_dict &args)               = 0;
};

template <class CompKernelType, int nSpace, int nQuadraturePoints_element, int nDOF_mesh_trial_element, int nDOF_trial_element, int nDOF_test_element, int nQuadraturePoints_elementBoundary>
class Richards : public Richards_base {
public:
  const int      nDOF_test_X_trial_element;
  CompKernelType ck;
  // Per-DOF density projected from q_rho in calculateResidual_entropy_viscosity.
  // Reused by invert() so the m -> u inversion uses the same variable density
  // that built the forward mass.
  std::vector<double> rho_dof_member;
  // Scratch arrays sized by the GLOBAL DOF count, held as members and resized
  // in place rather than declared as stack VLAs inside the kernels.
  //
  // They used to be `double psi[numDOFs], ...` etc.  numDOFs is global, so on a
  // refined mesh those are megabytes of automatic storage: at numDOFs = 361501
  // (Elder 1201x301) the four in calculateResidual_entropy_viscosity are
  // 4 x 361501 x 8 = 11.6 MB against a default `ulimit -s` of 8 MB, and the
  // function segfaulted on its very first call with no other symptom.  The
  // cliff sits at 8 MB/(4 x 8 B) = 262144 DOFs -- below it everything works and
  // nothing is ever noticed, which is why it only bites after a refinement, and
  // why a case can run under a scheduler (which usually raises or unlimits the
  // stack) yet crash on a workstation with identical code.
  //
  // resize() only reallocates when the size grows, so after the first call this
  // costs nothing; the explicit zeroing at each use site is kept because eta and
  // global_entropy_residual are written only under EV_Stab, and a reused buffer
  // would otherwise hand the next call the previous one's values.
  std::vector<double> psi_member, eta_member, global_entropy_residual_member,
                      boundary_integral_member, cflux_member,
                      Rpos_member, Rneg_member;
  // Pore size distribution / relative permeability model selected from Python
  // (Coefficients.PSK_type). 0 = van Genuchten-Mualem (default), 1 = Brooks-
  // Corey-Burdine, 2 = Brooks-Corey-Mualem, 3 = Gardner, 4 = Hornung-Messing.
  // Refreshed from args at
  // the top of every kernel that evaluates the closure.
  int PSK_TYPE_member = 0;
  // Air-entry head h_e >= 0 of the van Genuchten-Mualem closure (PSK_TYPE 0), refreshed with
  // PSK_TYPE. h_e > 0 selects the modification of Ippisch, Vogel & Bastian (2006): saturated for
  // psiC <= h_e, Se = sBar(psiC)/S_c and k_r = sqrt(Se) (vBar/V_c)^2 above, S_c = sBar(h_e),
  // V_c = vBar(h_e). theta and k_r then have bounded slopes at saturation; for n < 2 the
  // unmodified closure's are unbounded there, which stalls Newton at nodes crossing psi = 0.
  double VG_AIR_ENTRY_member = 0.0;
  inline void vgAirEntryScales(const double alpha, const double n_vg, double &Sc, double &Vc)
  {
    const double m_vg  = 1.0 - 1.0 / n_vg;
    const double pcBar = alpha * VG_AIR_ENTRY_member;
    Sc = pow(1.0 + pow(pcBar, n_vg), -m_vg);
    Vc = 1.0 - pow(pcBar, n_vg - 1.0) * Sc;
  }
  Richards() : nDOF_test_X_trial_element(nDOF_test_element * nDOF_trial_element), ck() { }
  inline void evaluateCoefficients(const int rowptr[nSpace], const int colind[nnz], const double rho0, const double rho_transport, const double beta, const double gravity[nSpace], const double alpha, const double n_vg, const double thetaR, const double thetaSR, const double KWs[nnz], const double &u, double &m, double &dm, double f[nSpace], double df[nSpace], double a[nnz], double da[nnz], double as[nnz], double &kr, double &dkr, double &thetaW_out)
  {
    const double psiC = -u;
    double thetaW, DthetaW_DpsiC, KWr, DKWr_DpsiC;
    if (PSK_TYPE_member == 1 || PSK_TYPE_member == 2) {
      // Same Brooks-Corey retention curve either way; PSK_TYPE only picks which
      // k_rw closure supplies the exponent (see pskRelations.h).
      proteus::richards::psk::bc_wetting(
          psiC, alpha, n_vg, thetaR, thetaSR,
          thetaW, DthetaW_DpsiC, KWr, DKWr_DpsiC,
          PSK_TYPE_member == 2 ? proteus::richards::psk::bc_kr::mualem
                               : proteus::richards::psk::bc_kr::burdine);
    } else if (PSK_TYPE_member == 3) {
      proteus::richards::psk::gardner_wetting(
          psiC, alpha, n_vg, thetaR, thetaSR,
          thetaW, DthetaW_DpsiC, KWr, DKWr_DpsiC);
    } else if (PSK_TYPE_member == 4) {
      proteus::richards::psk::hm_wetting(
          psiC, alpha, n_vg, thetaR, thetaSR,
          thetaW, DthetaW_DpsiC, KWr, DKWr_DpsiC);
    } else if (VG_AIR_ENTRY_member > 0.0) {
      if (psiC <= VG_AIR_ENTRY_member) {
        thetaW        = thetaR + thetaSR;
        DthetaW_DpsiC = 0.0;
        KWr           = 1.0;
        DKWr_DpsiC    = 0.0;
      } else {
        double Sc, Vc;
        vgAirEntryScales(alpha, n_vg, Sc, Vc);
        proteus::richards::psk::vgm_wetting(
            psiC, alpha, n_vg, thetaR, thetaSR,
            thetaW, DthetaW_DpsiC, KWr, DKWr_DpsiC);
        thetaW         = thetaR + (thetaW - thetaR) / Sc;
        DthetaW_DpsiC /= Sc;
        const double krScale = 1.0 / (sqrt(Sc) * Vc * Vc);
        KWr        *= krScale;
        DKWr_DpsiC *= krScale;
      }
    } else {
      proteus::richards::psk::vgm_wetting(
          psiC, alpha, n_vg, thetaR, thetaSR,
          thetaW, DthetaW_DpsiC, KWr, DKWr_DpsiC);
    }
    thetaW_out = thetaW;
    // Density uses transported salinity scaled by the compressibility factor.
    const double rhom  = rho_transport * exp(beta * u);
    const double drhom = beta * rhom;
    m     = rhom * thetaW;
    dm    = -rhom * DthetaW_DpsiC + drhom * thetaW;
    const double rho_ratio = rhom / rho0;
    for (int I = 0; I < nSpace; I++) {
      f[I]  = 0.0;
      df[I] = 0.0;
      for (int ii = rowptr[I]; ii < rowptr[I + 1]; ii++) {
        a[ii]  = rhom * KWr * KWs[ii];
        da[ii] = (drhom * KWr - rhom * DKWr_DpsiC) * KWs[ii];
        f[I] += rhom * rho_ratio * KWr * KWs[ii] * gravity[colind[ii]];
        df[I] += (drhom * rho_ratio * KWr +
                  rhom * (drhom / rho0) * KWr -
                  rhom * rho_ratio * DKWr_DpsiC) * KWs[ii] * gravity[colind[ii]];
        as[ii] = rhom * KWs[ii];
        kr     = KWr;
        dkr    = -DKWr_DpsiC;
      }
    }
  }

  inline void evaluateInverseCoefficients(const int rowptr[nSpace], const int colind[nnz], const double rho, const double beta, const double gravity[nSpace], const double alpha, const double n_vg, const double thetaR, const double thetaSR, const double KWs[nnz], double &u, const double &m, const double &dm, const double f[nSpace], const double df[nSpace], const double a[nnz], const double da[nnz])
  {
    (void)rowptr; (void)colind; (void)beta; (void)gravity; (void)KWs;
    (void)dm; (void)f; (void)df; (void)a; (void)da;

    // Both Brooks-Corey codes share one retention curve, so both invert with
    // bc_*; PSK_TYPE 1 vs 2 differs only in k_rw, which plays no part here.
    if (PSK_TYPE_member == 1 || PSK_TYPE_member == 2) {
      proteus::richards::psk::bc_invert_analytic(m, rho, alpha, n_vg, thetaR, thetaSR, u);
    } else if (PSK_TYPE_member == 3) {
      proteus::richards::psk::gardner_invert_analytic(m, rho, alpha, n_vg, thetaR, thetaSR, u);
    } else if (PSK_TYPE_member == 4) {
      proteus::richards::psk::hm_invert_analytic(m, rho, alpha, n_vg, thetaR, thetaSR, u);
    } else if (VG_AIR_ENTRY_member > 0.0) {
      // theta_vg = thetaR + S_c (theta - thetaR) is the unmodified curve's water content at the same psiC
      double Sc, Vc;
      vgAirEntryScales(alpha, n_vg, Sc, Vc);
      proteus::richards::psk::vgm_invert_analytic(Sc * m + (1.0 - Sc) * rho * thetaR, rho, alpha, n_vg, thetaR, thetaSR, u);
    } else {
      proteus::richards::psk::vgm_invert_analytic(m, rho, alpha, n_vg, thetaR, thetaSR, u);
    }
  }

inline void evaluateInverseCoefficients_Newton(const int rowptr[nSpace],
                                               const int colind[nnz],
                                               const double rho,
                                               const double beta,
                                               const double gravity[nSpace],
                                               const double alpha,
                                               const double n_vg,
                                               const double thetaR,
                                               const double thetaSR,
                                               const double KWs[nnz],
                                               double &u,               // in/out
                                               const double &m,
                                               const double &dm,
                                               const double f[nSpace],
                                               const double df[nSpace],
                                               const double a[nnz],
                                               const double da[nnz])
{
  (void)rowptr; (void)colind; (void)gravity; (void)KWs;
  (void)dm; (void)f; (void)df; (void)a; (void)da;

  // Newton inversion of the full forward mass m = rho(u)*theta_w(u), so the
  // exp(beta*u) factor the analytic inverse drops is included. The retention
  // curve it solves against is the selected PSK model.
  if (PSK_TYPE_member == 1 || PSK_TYPE_member == 2) {
    proteus::richards::psk::bc_invert_newton(m, rho, beta, alpha, n_vg, thetaR, thetaSR, u);
  } else if (PSK_TYPE_member == 3) {
    proteus::richards::psk::gardner_invert_newton(m, rho, beta, alpha, n_vg, thetaR, thetaSR, u);
  } else if (PSK_TYPE_member == 4) {
    proteus::richards::psk::hm_invert_newton(m, rho, beta, alpha, n_vg, thetaR, thetaSR, u);
  } else if (VG_AIR_ENTRY_member > 0.0) {
    // as in evaluateInverseCoefficients; the shift uses rho, not rho exp(beta u) (O(beta) in thetaR)
    double Sc, Vc;
    vgAirEntryScales(alpha, n_vg, Sc, Vc);
    proteus::richards::psk::vgm_invert_newton(Sc * m + (1.0 - Sc) * rho * thetaR, rho, beta, alpha, n_vg, thetaR, thetaSR, u);
  } else {
    proteus::richards::psk::vgm_invert_newton(m, rho, beta, alpha, n_vg, thetaR, thetaSR, u);
  }
}

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
    // Floored like the 1.0e-8 in the elementDiameter version above: in a
    // saturated region without gravity A0 = dm_t = 0 and Ai = df = 0, and an
    // infinite tau_v turns useMetrics*tau1 = 0*inf into NaN in the residual.
    tau_v = 1.0 / sqrt(Ct_sge * A0 * A0 + v_d_Gv + 1.0e-16);
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

  //cfluxJacobian is the SAME IIPG consistency term as fluxJacobian but built from the
  //relative-permeability-weighted conductivity K = rho*k_rw*Ks instead of the saturated
  //Ks.  It feeds TransportMatrixConsistent, i.e. the HIGH order flux fH, exactly the way
  //the cell loop feeds it from `a` while TransportMatrix gets `as`.  Without the split the
  //boundary block enters fH with weight 1 and enters fL multiplied by the nodal upwind
  //k_rw, so it does not cancel in fA = fH - fL and leaves a spurious antidiffusive flux
  //~1/k_rw times the interior one at every weakly imposed Dirichlet row.
  void exteriorNumericalFluxJacobian2(const int rowptr[nSpace], const int colind[nnz], const int isDOFBoundary, const double n[nSpace],  const double Ks[nnz], const double K[nnz], const double dK[nnz], const double grad_psi[nSpace], const double grad_v[nSpace], const double dK_rho_g[nSpace], const double v, const double penalty, double &fluxJacobian, double &bfluxJacobian, double &cfluxJacobian)
  {
    if (isDOFBoundary) {
      fluxJacobian = 0.0;
      bfluxJacobian = 0.0;
      cfluxJacobian = 0.0;
      for (int I = 0; I < nSpace; I++) {
        for (int m = rowptr[I]; m < rowptr[I + 1]; m++) {
          fluxJacobian -= Ks[m] * grad_v[colind[m]] * n[I];
          cfluxJacobian -= K[m] * grad_v[colind[m]] * n[I];
        }
      }
      //Dirichlet penalty
      bfluxJacobian = penalty * v;
    } else {
      fluxJacobian = 0.0;
      bfluxJacobian = 0.0;
      cfluxJacobian = 0.0;
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
    // mu/mu_0 from the transport model: the conductivity at a quadrature point is K_s/mu
    proteus::pyarray<double> &q_mu                                      = args.array<double>("q_mu");
    proteus::pyarray<double> &ebqe_mu                                   = args.array<double>("ebqe_mu");

    proteus::pyarray<double> &gravity                                    = args.array<double>("gravity");
    proteus::pyarray<double> &alpha                                      = args.array<double>("alpha");
    proteus::pyarray<double> &n                                          = args.array<double>("n");
    proteus::pyarray<double> &thetaR                                     = args.array<double>("thetaR");
    proteus::pyarray<double> &thetaSR                                    = args.array<double>("thetaSR");
    PSK_TYPE_member = args.scalar<int>("PSK_TYPE");
    VG_AIR_ENTRY_member = args.scalar<double>("VG_AIR_ENTRY");
    proteus::pyarray<double> &KWs                                        = args.array<double>("KWs");
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
    proteus::pyarray<double> &globalResidual                             = args.array<double>("globalResidual");
    int                  nExteriorElementBoundaries_global          = args.scalar<int>("nExteriorElementBoundaries_global");
    proteus::pyarray<int>    &exteriorElementBoundariesArray             = args.array<int>("exteriorElementBoundariesArray");
    proteus::pyarray<int>    &elementBoundaryElementsArray               = args.array<int>("elementBoundaryElementsArray");
    proteus::pyarray<int>    &elementBoundaryLocalElementBoundariesArray = args.array<int>("elementBoundaryLocalElementBoundariesArray");
    proteus::pyarray<double> &ebqe_velocity_ext                          = args.array<double>("ebqe_velocity_ext");
    proteus::pyarray<int>    &isDOFBoundary_u                            = args.array<int>("isDOFBoundary_u");
    proteus::pyarray<double> &ebqe_bc_u_ext                              = args.array<double>("ebqe_bc_u_ext");
    proteus::pyarray<int>    &isFluxBoundary_u                           = args.array<int>("isFluxBoundary_u");
    proteus::pyarray<double> &ebqe_bc_flux_ext                           = args.array<double>("ebqe_bc_flux_ext");
    proteus::pyarray<double> &ebqe_phi                                   = args.array<double>("ebqe_phi");
    double               epsFact                                    = args.scalar<double>("epsFact");
    proteus::pyarray<double> &ebqe_u                                     = args.array<double>("ebqe_u");
    proteus::pyarray<double> &ebqe_theta                                 = args.array<double>("ebqe_theta");
    proteus::pyarray<double> &ebqe_flux                                  = args.array<double>("ebqe_flux");
    // VMS
    double VMS = args.scalar<double>("VMS");
    // Time-weighting of the flux terms (ThetaScheme): 1 = backward Euler,
    // 1/2 = Crank-Nicolson. The mass term is always (m - m^n)/dt.
    const double Theta = args.scalar<double>("Theta");
    // PARAMETERS FOR EDGE BASED STABILIZATION
    double cE = args.scalar<double>("cE");
    double cK = args.scalar<double>("cK");
    // PARAMETERS FOR LOG BASED ENTROPY FUNCTION
    double uL = args.scalar<double>("uL");
    double uR = args.scalar<double>("uR");
    // PARAMETERS FOR EDGE VISCOSITY
    int               numDOFs                       = args.scalar<int>("numDOFs");
    int               NNZ                           = args.scalar<int>("NNZ");
    proteus::pyarray<int> &csrRowIndeces_DofLoops        = args.array<int>("csrRowIndeces_DofLoops");
    proteus::pyarray<int> &csrColumnOffsets_DofLoops     = args.array<int>("csrColumnOffsets_DofLoops");
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
        //old-time solution and gradient for the (1-Theta) flux terms
        double un = 0.0;
        ck.valFromDOF(u_dof_old.data(), &u_l2g.data()[eN_nDOF_trial_element], &u_trial_ref.data()[k * nDOF_trial_element], un);
        ck.gradFromDOF(u_dof_old.data(), &u_l2g.data()[eN_nDOF_trial_element], u_grad_trial, grad_u_old);

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
        double KWs_mu[nnz];
        for (int ii = 0; ii < nnz; ii++) KWs_mu[ii] = KWs.data()[elementMaterialTypes.data()[eN] * nnz + ii] / q_mu.data()[eN_k];
        evaluateCoefficients(a_rowptr.data(), a_colind.data(), rho, rho_local, beta, gravity.data(), alpha.data()[elementMaterialTypes.data()[eN]], n.data()[elementMaterialTypes.data()[eN]], thetaR.data()[elementMaterialTypes.data()[eN]],
                             thetaSR.data()[elementMaterialTypes.data()[eN]], KWs_mu, u, m, dm, f, df, a, da, as, Kr, dKr, thetaW);
        q_theta.data()[eN_k] = thetaW;
        double mn_q, dmn_q, fn[nSpace], dfn[nSpace], an[nnz], dan[nnz], asn[nnz], Krn, dKrn, thetaWn;
        if (Theta < 1.0) {
          evaluateCoefficients(a_rowptr.data(), a_colind.data(), rho, rho_local, beta, gravity.data(), alpha.data()[elementMaterialTypes.data()[eN]], n.data()[elementMaterialTypes.data()[eN]], thetaR.data()[elementMaterialTypes.data()[eN]],
                               thetaSR.data()[elementMaterialTypes.data()[eN]], &KWs.data()[elementMaterialTypes.data()[eN] * nnz], un, mn_q, dmn_q, fn, dfn, an, dan, asn, Krn, dKrn, thetaWn);
        }

        for (int I = 0; I < nSpace; ++I) {
          q_velocity.data()[eN_k_nSpace + I] = grad_u[I];
        }
        // Darcy Velocity
        double pressure_gradient[nSpace];
        const double rho_ratio = rho_velocity / rho;
        for (int J=0; J<nSpace; ++J)
          pressure_gradient[J] = grad_u[J] - rho_ratio * gravity.data()[J];
        // for each row I, acc = sum_j (a_{Ij}/rho) * gp[j]
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
        //calculate time derivative at quadrature points
        //
        ck.bdf(alphaBDF, q_m_betaBDF.data()[eN_k], m, dm, m_t, dm_t);
        //
        //calculate subgrid error (strong residual and adjoint)
        //
        //calculate strong residual
        pdeResidual_u = ck.Mass_strong(m_t) + ck.Advection_strong(df, grad_u);
        //calculate adjoint
        for (int i = 0; i < nDOF_test_element; i++) {
          int i_nSpace = i * nSpace;
          Lstar_u[i]   = ck.Advection_adjoint(df, &u_grad_test_dV[i_nSpace]);
        }
        //calculate tau and tau*Res
        calculateSubgridError_tau(elementDiameter[eN], dm_t, df, cfl[eN_k], tau0);
        calculateSubgridError_tau(Ct_sge, G, dm_t, df, tau1, cfl[eN_k]);

        tau = useMetrics * tau1 + (1.0 - useMetrics) * tau0;

        subgridError_u = -tau * pdeResidual_u;
        // //
        // //calculate shock capturing diffusion
        // //
        // ck.calculateNumericalDiffusion(shockCapturingDiffusion, elementDiameter[eN], pdeResidual_u, grad_u, numDiff0);
        // ck.calculateNumericalDiffusion(shockCapturingDiffusion, sc_uref, sc_alpha, G, G_dd_G, pdeResidual_u, grad_u, numDiff1);
        // q_numDiff_u[eN_k] = useMetrics * numDiff1 + (1.0 - useMetrics) * numDiff0;
        //
        //update element residual
        //
        for (int i = 0; i < nDOF_test_element; i++) {
          int eN_k_i = eN_k * nDOF_test_element + i, eN_k_i_nSpace = eN_k_i * nSpace, i_nSpace = i * nSpace;

          elementResidual_u[i] += ck.Mass_weak(m_t, u_test_dV[i]) + Theta * (ck.Advection_weak(f, &u_grad_test_dV[i_nSpace]) + ck.Diffusion_weak(a_rowptr.data(), a_colind.data(), a, grad_u, &u_grad_test_dV[i_nSpace])) + VMS * ck.SubgridError(subgridError_u, Lstar_u[i]) + VMS * ck.NumericalDiffusion(q_numDiff_u_last[eN_k], grad_u, &u_grad_test_dV[i_nSpace]);
          if (Theta < 1.0)
            elementResidual_u[i] += (1.0 - Theta) * (ck.Advection_weak(fn, &u_grad_test_dV[i_nSpace]) + ck.Diffusion_weak(a_rowptr.data(), a_colind.data(), an, grad_u_old, &u_grad_test_dV[i_nSpace]));
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
        double KWs_mu[nnz];
        for (int ii = 0; ii < nnz; ii++) KWs_mu[ii] = KWs.data()[elementMaterialTypes.data()[eN] * nnz + ii] / ebqe_mu.data()[ebNE_kb];
        evaluateCoefficients(a_rowptr.data(), a_colind.data(), rho, rho_ext, beta, gravity.data(), alpha.data()[elementMaterialTypes.data()[eN]], n.data()[elementMaterialTypes.data()[eN]], thetaR.data()[elementMaterialTypes.data()[eN]],
                             thetaSR.data()[elementMaterialTypes.data()[eN]], KWs_mu, u_ext, m_ext, dm_ext, f_ext, df_ext, a_ext, da_ext, as_ext, Kr, dKr, thetaW_ext);
        evaluateCoefficients(a_rowptr.data(), a_colind.data(), rho, rho_ext, beta, gravity.data(), alpha.data()[elementMaterialTypes.data()[eN]], n.data()[elementMaterialTypes.data()[eN]], thetaR.data()[elementMaterialTypes.data()[eN]],
                             thetaSR.data()[elementMaterialTypes.data()[eN]], KWs_mu, bc_u_ext, bc_m_ext, bc_dm_ext, bc_f_ext, bc_df_ext, bc_a_ext, bc_da_ext, bc_as_ext, Kr, dKr, thetaW_bc);
        ebqe_theta.data()[ebNE_kb] = thetaW_ext;
        
        //
        //Calculate Darcy velocity on exterior face : v_ext = -(a_ext/rho) * (grad_u_ext + gravity) ---
        //
        double ext_pressure_gradient[nSpace];
        const double rho_ratio_ext = rho_velocity_ext / rho;
        for (int J=0; J<nSpace; ++J)
          ext_pressure_gradient[J] = grad_u_ext[J] - rho_ratio_ext * gravity.data()[J];

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
        //calculate the numerical fluxes
        //
        exteriorNumericalFlux(ebqe_bc_flux_ext[ebNE_kb], a_rowptr.data(), a_colind.data(),
                              isSeepageFace.data()[ebNE], //tricky, this is a face flag not face quad
                              isDOFBoundary_u.data()[ebNE_kb], normal, bc_u_ext, a_ext, grad_u_ext, u_ext, f_ext,
                              ebqe_penalty_ext.data()[ebNE_kb], // penalty,
                              flux_ext);
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
    proteus::pyarray<double> &q_mu                     = args.array<double>("q_mu");
    proteus::pyarray<double> &ebqe_mu                  = args.array<double>("ebqe_mu");
    //////////////////////////////////////////////////////////////////////////////////////////////////////

    proteus::pyarray<double> &gravity                   = args.array<double>("gravity");
    proteus::pyarray<double> &alpha                     = args.array<double>("alpha");
    proteus::pyarray<double> &n                         = args.array<double>("n");
    proteus::pyarray<double> &thetaR                    = args.array<double>("thetaR");
    proteus::pyarray<double> &thetaSR                   = args.array<double>("thetaSR");
    PSK_TYPE_member = args.scalar<int>("PSK_TYPE");
    VG_AIR_ENTRY_member = args.scalar<double>("VG_AIR_ENTRY");
    proteus::pyarray<double> &KWs                       = args.array<double>("KWs");
    double               useMetrics                = args.scalar<double>("useMetrics");
    double               alphaBDF                  = args.scalar<double>("alphaBDF");
    int                  lag_shockCapturing        = args.scalar<int>("lag_shockCapturing");
    double               shockCapturingDiffusion   = args.scalar<double>("shockCapturingDiffusion");
    // VMS
    double               VMS                                        = args.scalar<double>("VMS");
    const double         Theta                                      = args.scalar<double>("Theta"); // see calculateResidual
    proteus::pyarray<int>    &u_l2g                                      = args.array<int>("u_l2g");
    proteus::pyarray<double> &elementDiameter                            = args.array<double>("elementDiameter");
    proteus::pyarray<double> &u_dof                                      = args.array<double>("u_dof");
    proteus::pyarray<double> &velocity                                   = args.array<double>("velocity");
    proteus::pyarray<double> &q_m_betaBDF                                = args.array<double>("q_m_betaBDF");
    proteus::pyarray<double> &cfl                                        = args.array<double>("cfl");
    proteus::pyarray<double> &q_numDiff_u                                = args.array<double>("q_numDiff_u");
    proteus::pyarray<double> &q_numDiff_u_last                           = args.array<double>("q_numDiff_u_last");
    proteus::pyarray<int>    &csrRowIndeces_u_u                          = args.array<int>("csrRowIndeces_u_u");
    proteus::pyarray<int>    &csrColumnOffsets_u_u                       = args.array<int>("csrColumnOffsets_u_u");
    proteus::pyarray<double> &globalJacobian                             = args.array<double>("globalJacobian");
    int                  nExteriorElementBoundaries_global          = args.scalar<int>("nExteriorElementBoundaries_global");
    proteus::pyarray<int>    &exteriorElementBoundariesArray             = args.array<int>("exteriorElementBoundariesArray");
    proteus::pyarray<int>    &elementBoundaryElementsArray               = args.array<int>("elementBoundaryElementsArray");
    proteus::pyarray<int>    &elementBoundaryLocalElementBoundariesArray = args.array<int>("elementBoundaryLocalElementBoundariesArray");
    proteus::pyarray<double> &ebqe_velocity_ext                          = args.array<double>("ebqe_velocity_ext");
    proteus::pyarray<int>    &isDOFBoundary_u                            = args.array<int>("isDOFBoundary_u");
    proteus::pyarray<double> &ebqe_bc_u_ext                              = args.array<double>("ebqe_bc_u_ext");
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
      for (int i = 0; i < nDOF_test_element; i++) {
        for (int j = 0; j < nDOF_trial_element; j++) { elementJacobian_u_u[i][j] = 0.0; }
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
        //calculate pde coefficients and derivatives at quadrature points
        //
        double Kr, dKr, thetaW;
        //const double rho_local = q_rho.data()[eN_k];
        double KWs_mu[nnz];
        for (int ii = 0; ii < nnz; ii++) KWs_mu[ii] = KWs.data()[elementMaterialTypes.data()[eN] * nnz + ii] / q_mu.data()[eN_k];

        evaluateCoefficients(a_rowptr.data(), a_colind.data(), rho, q_rho.data()[eN_k], beta, gravity.data(), alpha.data()[elementMaterialTypes.data()[eN]], n.data()[elementMaterialTypes.data()[eN]], thetaR.data()[elementMaterialTypes.data()[eN]],
                             thetaSR.data()[elementMaterialTypes.data()[eN]], KWs_mu, u, m, dm, f, df, a, da, as, Kr, dKr, thetaW);
        //
        //calculate time derivatives
        //
        ck.bdf(alphaBDF, q_m_betaBDF.data()[eN_k], m, dm, m_t, dm_t);
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
          dpdeResidual_u_u[j] = ck.MassJacobian_strong(dm_t, u_trial_ref[k * nDOF_trial_element + j]) + ck.AdvectionJacobian_strong(df, &u_grad_trial[j_nSpace]);
        }
        //tau and tau*Res
        calculateSubgridError_tau(elementDiameter[eN], dm_t, df, cfl[eN_k], tau0);
        calculateSubgridError_tau(Ct_sge, G, dm_t, df, tau1, cfl[eN_k]);
        tau = useMetrics * tau1 + (1.0 - useMetrics) * tau0;
        for (int j = 0; j < nDOF_trial_element; j++) dsubgridError_u_u[j] = -tau * dpdeResidual_u_u[j];
        for (int i = 0; i < nDOF_test_element; i++) {
          for (int j = 0; j < nDOF_trial_element; j++) {
            int j_nSpace = j * nSpace;
            int i_nSpace = i * nSpace;
            elementJacobian_u_u[i][j] += ck.MassJacobian_weak(dm_t, u_trial_ref.data()[k * nDOF_trial_element + j], u_test_dV[i]) + Theta * (ck.AdvectionJacobian_weak(df, u_trial_ref.data()[k * nDOF_trial_element + j], &u_grad_test_dV[i_nSpace]) +
                                         ck.DiffusionJacobian_weak(a_rowptr.data(), a_colind.data(), a, da, grad_u, &u_grad_test_dV[i_nSpace], 1.0, u_trial_ref.data()[k * nDOF_trial_element + j], &u_grad_trial[j_nSpace])) + VMS * ck.SubgridErrorJacobian(dsubgridError_u_u[j], Lstar_u[i]) + VMS * ck.NumericalDiffusionJacobian(q_numDiff_u_last[eN_k], &u_grad_trial[j_nSpace], &u_grad_test_dV[i_nSpace]);
          } //j
        } //i
      } //k
      //
      //load into element Jacobian into global Jacobian
      //
      for (int i = 0; i < nDOF_test_element; i++) {
        int eN_i = eN * nDOF_test_element + i;
        for (int j = 0; j < nDOF_trial_element; j++) {
          int eN_i_j = eN_i * nDOF_trial_element + j;
          globalJacobian.data()[csrRowIndeces_u_u[eN_i] + csrColumnOffsets_u_u[eN_i_j]] += elementJacobian_u_u[i][j];
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
        double KWs_mu[nnz];
        for (int ii = 0; ii < nnz; ii++) KWs_mu[ii] = KWs.data()[elementMaterialTypes.data()[eN] * nnz + ii] / ebqe_mu.data()[ebNE_kb];

        evaluateCoefficients(a_rowptr.data(), a_colind.data(), rho, rho_ext, beta, gravity.data(), alpha.data()[elementMaterialTypes.data()[eN]], n.data()[elementMaterialTypes.data()[eN]], thetaR.data()[elementMaterialTypes.data()[eN]],
                             thetaSR.data()[elementMaterialTypes.data()[eN]], KWs_mu, u_ext, m_ext, dm_ext, f_ext, df_ext, a_ext, da_ext, as_ext, Kr, dKr, thetaW);
        evaluateCoefficients(a_rowptr.data(), a_colind.data(), rho, rho_ext, beta, gravity.data(), alpha.data()[elementMaterialTypes.data()[eN]], n.data()[elementMaterialTypes.data()[eN]], thetaR.data()[elementMaterialTypes.data()[eN]],
                             thetaSR.data()[elementMaterialTypes.data()[eN]], KWs_mu, bc_u_ext, bc_m_ext, bc_dm_ext, bc_f_ext, bc_df_ext, bc_a_ext, bc_da_ext, bc_as_ext, Kr, dKr, thetaW_bc);
        //
        //calculate the flux jacobian
        //
        for (int j = 0; j < nDOF_trial_element; j++) {
          exteriorNumericalFluxJacobian(a_rowptr.data(), a_colind.data(), isDOFBoundary_u.data()[ebNE_kb], normal, a_ext, da_ext, grad_u_ext, &u_grad_trial_trace[j * nSpace], df_ext, u_trial_trace_ref.data()[ebN_local_kb * nDOF_test_element + j],
                                        ebqe_penalty_ext.data()[ebNE_kb], //penalty,
                                        fluxJacobian_u_u[j]);
        } //j
        //
        //update the global Jacobian from the flux Jacobian
        //
        for (int i = 0; i < nDOF_test_element; i++) {
          int eN_i = eN * nDOF_test_element + i;
          for (int j = 0; j < nDOF_trial_element; j++) {
            int ebN_i_j = ebN * 4 * nDOF_test_X_trial_element + i * nDOF_trial_element + j;
            globalJacobian.data()[csrRowIndeces_u_u[eN_i] + csrColumnOffsets_eb_u_u[ebN_i_j]] += fluxJacobian_u_u[j] * u_test_dS[i];
          } //j
        } //i
      } //kb
    } //ebNE
  } //computeJacobian

  void FCTStep(arguments_dict &args)
{
  proteus::pyarray<double> &bc_mask                   = args.array<double>("bc_mask");
  int                  NNZ                       = args.scalar<int>("NNZ");     // number of non-zero entries
  int                  numDOFs                   = args.scalar<int>("numDOFs"); // number of DOFs
  double               dt                        = args.scalar<double>("dt");
  proteus::pyarray<double> &ML                        = args.array<double>("ML");    // lumped mass matrix (as vector)
  proteus::pyarray<double> &mn                        = args.array<double>("mn");    // DOFs at time tn
  proteus::pyarray<double> &mHigh                     = args.array<double>("mHigh"); // high-order mass at t^{n+1}
  proteus::pyarray<double> &mLow                      = args.array<double>("mLow");  // low-order mass at t^{n+1}
  proteus::pyarray<double> &mDotLow                   = args.array<double>("mDotLow");
  proteus::pyarray<double> &limited_solution          = args.array<double>("limited_solution");
  proteus::pyarray<int>    &csrRowIndeces_DofLoops    = args.array<int>("csrRowIndeces_DofLoops");
  proteus::pyarray<int>    &csrColumnOffsets_DofLoops = args.array<int>("csrColumnOffsets_DofLoops");
  proteus::pyarray<double> &MC                        = args.array<double>("MC");              // consistent mass matrix
  proteus::pyarray<double> &dt_times_fH_minus_fL      = args.array<double>("dt_times_fH_minus_fL");
  proteus::pyarray<double> &min_m_bc                  = args.array<double>("min_m_bc");
  proteus::pyarray<double> &max_m_bc                  = args.array<double>("max_m_bc");
  proteus::pyarray<double> &thetaR                    = args.array<double>("thetaR");
  proteus::pyarray<double> &thetaSR                   = args.array<double>("thetaSR");
  proteus::pyarray<int>    &freeDOFMaterialTypes      = args.array<int>("freeDOFMaterialTypes");
  proteus::pyarray<double> &fluxCorrection            = args.array<double>("fluxCorrection");
  //owned by Python so they survive between the two passes below
  proteus::pyarray<double> &Rpos                      = args.array<double>("Rpos");
  proteus::pyarray<double> &Rneg                      = args.array<double>("Rneg");
  proteus::pyarray<double> &FluxCorrectionMatrix      = args.array<double>("FluxCorrectionMatrix");
  // realized corrections alpha_ij f_ij (row i, times bc_mask_i) and their spatial part
  // alpha_ij dt (f^H - f^L)_ij, i.e. without the consistent-mass term; read by the transport coupling
  proteus::pyarray<double> &limitedFluxMatrix         = args.array<double>("limitedFluxMatrix");
  proteus::pyarray<double> &limitedSpatialFluxMatrix  = args.array<double>("limitedSpatialFluxMatrix");
  // OUT: the limited correction applied on each edge (mass*ML), exported to a
  // coupled transport model; written in pass 2.
  proteus::pyarray<double> &limitedFlux               = args.array<double>("limitedFlux");
  // flags
  int                  LUMPED_MASS_MATRIX        = args.scalar<int>("LUMPED_MASS_MATRIX");
  int                  MONOLITHIC                = args.scalar<int>("MONOLITHIC");
  int                  fct_pass                  = args.scalar<int>("fct_pass");
  // 1 at DOFs whose low-order predictor is saturated (Coefficients.FCT_SATURATED_LOW_ORDER)
  proteus::pyarray<double> &satLow                    = args.array<double>("satLow");
  //Bound the WATER CONTENT, not the mass m = theta*rho: a neighbour's low-order
  //mass is compared at THIS node's density, mLow_j*rho_i/rho_j = theta_j*rho_i.  With
  //a variable density a heavier neighbour's m exceeds rho_i*theta_s, so bounding m
  //let the limiter hand node i more water than it can hold; invert() then clipped
  //it and the water was lost (up to 24% at a node on UHD).  With a uniform density
  //rho_i/rho_j = 1 and this is the plain mass bound.
  const bool           rhoBounds                 = rho_dof_member.size() == static_cast<std::size_t>(numDOFs);

  //PARALLEL. The limiter is L_ij = min(Rpos_i, Rneg_j), so an owned row i reads the ratio of a column
  //j that may be a ghost, and everything a ratio is built from -- the row's own sparsity, ML, mDotLow,
  //min/max_m_bc -- is INCOMPLETE at a ghost DOF: with one layer of overlap a ghost's element star and
  //its boundary faces are only partly on this rank. Both ranks sharing a cut edge would then apply a
  //different L_ij, f_ij = -f_ji dies and the correction manufactures mass along the partition.
  //The cure is to let the owners' values cross the cut in the middle of the limiter:
  //  fct_pass = 1 -> FluxCorrectionMatrix + the local ratios Rpos/Rneg
  //  [Python forward-inserts Rpos/Rneg, owner -> ghost; the inputs above are scattered before pass 1]
  //  fct_pass = 2 -> apply the limiter
  //  fct_pass = 0 -> both back to back; this is the serial path and is byte-identical to the old
  //                  single-pass function.
  //mDot is pointwise in (mLow, mn) so it is rebuilt in whichever pass needs it rather than carried.
  const bool doPass1 = (fct_pass != 2);
  const bool doPass2 = (fct_pass != 1);

  std::vector<double> mDot(numDOFs, 0.0);
  for (int i = 0; i < numDOFs; i++)
    mDot.at(i) = (mLow.at(i) - mn.at(i)) / dt; // local time derivative from low-order mass

  ///////////////////////////////////////////////////////
  // PASS 1: antidiffusive fluxes and Zalesak's ratios  //
  ///////////////////////////////////////////////////////
  if (doPass1) {
  int ij = 0;
  for (int i = 0; i < numDOFs; i++) {

    // initialize local min/max from BC
    double mini = min_m_bc.at(i);
    double maxi = max_m_bc.at(i);

    double Pposi = 0.0, Pnegi = 0.0;

    // LOOP OVER THE SPARSITY PATTERN (j-LOOP)
    for (int offset = csrRowIndeces_DofLoops.at(i); offset < csrRowIndeces_DofLoops.at(i + 1);offset++)
    {
      int j = csrColumnOffsets_DofLoops.at(offset);

      ////////////////////////
      // COMPUTE THE BOUNDS //
      ////////////////////////
      if (GLOBAL_FCT == 0) {
        if (MONOLITHIC == 0) {
          const double mLow_j = rhoBounds ? mLow.at(j) * (rho_dof_member[i] / rho_dof_member[j]) : mLow.at(j);
          mini = fmin(mini, mLow_j);
          maxi = fmax(maxi, mLow_j);
        } else {
          mini = fmin(mini, mn.at(j));
          maxi = fmax(maxi, mn.at(j));
        }
      }

      if (MONOLITHIC == 0) {
        FluxCorrectionMatrix.at(ij) = (LUMPED_MASS_MATRIX == 1 ? 0. : 1.) * dt * MC.at(ij) * (mDotLow.at(i) - mDotLow.at(j)) + dt_times_fH_minus_fL.at(ij);
      } else {
        FluxCorrectionMatrix.at(ij) = dt_times_fH_minus_fL.at(ij);
      }

      ///////////////////////
      // COMPUTE P VECTORS //
      ///////////////////////
      Pposi += FluxCorrectionMatrix.at(ij) * ((FluxCorrectionMatrix.at(ij) > 0) ? 1. : 0.);
      Pnegi += FluxCorrectionMatrix.at(ij) * ((FluxCorrectionMatrix.at(ij) < 0) ? 1. : 0.);

      // update ij
      ij += 1;
    } // j-loop

    //Cap the bounds at what node i can hold, rho_i*[theta_r, theta_s] of its own soil.  The
    //neighbour bound above is rho_i*max_j theta_j, and in a layered soil theta_j can lie in a
    //soil with a larger theta_s: the limiter then handed node i more water than its retention
    //curve can take, invert() clipped it at theta_s, and the water was lost (4.4% of the fluid
    //on the bioswale, nil with one theta_s).  mLow_i is kept inside the cap so the bounds stay
    //feasible where compressible storage (beta > 0) has already put it above rho_i*theta_s.
    if (rhoBounds && GLOBAL_FCT == 0 && MONOLITHIC == 0) {
      const int    mat_i = freeDOFMaterialTypes.at(i);
      const double rho_i = rho_dof_member[i];
      mini = fmax(mini, fmin(rho_i * thetaR.at(mat_i), mLow.at(i)));
      maxi = fmin(maxi, fmax(rho_i * (thetaR.at(mat_i) + thetaSR.at(mat_i)), mLow.at(i)));
    }

    ///////////////////////
    // COMPUTE Q VECTORS //
    ///////////////////////
    double gamma;
    double Qposi;
    double Qnegi;

    if (MONOLITHIC == 0) {
      Qposi = ML.at(i) * (maxi - mLow.at(i));
      Qnegi = ML.at(i) * (mini - mLow.at(i));
    } else {
      // cek todo: don't think this is right for Richards
      gamma = 10.0 * ML.at(i);
      Qposi = fmin(0.5 * ML.at(i) * (1.0 - mn.at(i)), gamma * (maxi - mn.at(i)));
      Qnegi = fmax(0.5 * ML.at(i) * (0.0 - mn.at(i)), gamma * (mini - mn.at(i)));
    }

    ///////////////////////
    // COMPUTE R VECTORS //
    ///////////////////////
    // Saturated predictor: no admissible change of theta at this DOF, so every
    // flux touching it is limited to zero (alpha_ij = alpha_ji, still conservative).
    if (satLow.at(i) > 0.0) { Qposi = 0.0; Qnegi = 0.0; }
    Rpos.at(i) = ((Pposi == 0.0) ? 1.0 : fmin(1.0, Qposi / Pposi));
    Rneg.at(i) = ((Pnegi == 0.0) ? 1.0 : fmin(1.0, Qnegi / Pnegi));
  } // i DOFs
  } // pass 1

  //////////////////////
  // COMPUTE LIMITERS //
  //////////////////////
  if (doPass2) {
  int ij = 0;
 // std::cout << "FCT: entering second DOF loop (applying limiters)...\n";

  for (int i = 0; i < numDOFs; i++) {
    double ith_Limiter_times_FluxCorrectionMatrix = 0.0;
    double alpha_fA, alpha_dot, beta_ij = 1.0;
    // LOOP OVER THE SPARSITY PATTERN (j-LOOP)
    for (int offset = csrRowIndeces_DofLoops.at(i);  offset < csrRowIndeces_DofLoops.at(i + 1); offset++)
    {
      int j = csrColumnOffsets_DofLoops.at(offset);
      const double alpha_ij = (FluxCorrectionMatrix.at(ij) > 0.0) ? fmin(Rpos.at(i), Rneg.at(j)) : fmin(Rneg.at(i), Rpos.at(j));
      alpha_fA = alpha_ij * FluxCorrectionMatrix.at(ij);
      limitedFluxMatrix.at(ij)        = alpha_fA * bc_mask.at(i);
      limitedSpatialFluxMatrix.at(ij) = alpha_ij * dt_times_fH_minus_fL.at(ij) * bc_mask.at(i);
      alpha_dot = fmin(1.0, beta_ij * fabs(alpha_fA) / MC.at(ij) / fmax(1.0e-8, fabs(mDot.at(i) - mDot.at(j))));

      if (MONOLITHIC == 0) {
        ith_Limiter_times_FluxCorrectionMatrix += alpha_fA;
        limitedFlux.at(ij) = alpha_fA * bc_mask.at(i);
      } else {
        ith_Limiter_times_FluxCorrectionMatrix += alpha_fA + (LUMPED_MASS_MATRIX == 1 ? 0. : 1.) * dt * alpha_dot * MC.at(ij) * (mDot.at(i) - mDot.at(j));
        limitedFlux.at(ij) = (alpha_fA + (LUMPED_MASS_MATRIX == 1 ? 0. : 1.) * dt * alpha_dot * MC.at(ij) * (mDot.at(i) - mDot.at(j))) * bc_mask.at(i);
      }
      ij += 1;
    } 
    fluxCorrection.at(i) = -ith_Limiter_times_FluxCorrectionMatrix * bc_mask.at(i) / dt;
    limited_solution.at(i) = mLow.at(i) + 1.0 / ML.at(i) * ith_Limiter_times_FluxCorrectionMatrix * bc_mask.at(i);
  }
  }
}

  //EDGE VELOCITY FOR THE COUPLED TRANSPORT MODEL.  A velocity at the quadrature
  //points whose nodal fluxes are exactly the water this step moved along the edges,
  //
  //  int rho v.grad(w_i) dV = sum_j E_ij,   E_ij = edgeFluxLow_ij + limitedFlux_ij/dt
  //
  //(E = fL with FCT off; low order plus the limited correction with FCT on), so a
  //transport model advecting with v satisfies this model's discrete continuity
  //equation node by node.  No global solve; element by element:
  //  1. split E_ij among the elements sharing edge ij with the symmetric weights
  //     |S^e_ij| / sum_e |S^e_ij|, S^e_ij = int_e rho K_s grad(w_j).grad(w_i) (equal
  //     shares where every S^e_ij is zero), so f^e_ij = -f^e_ji and
  //     sum_e f^e_ij = E_ij;
  //  2. R^e_a = sum_b f^e_ab sums to zero over the element, and on a P1 simplex the
  //     G_a = int_e grad(w_a) dV span exactly the zero-sum vectors, so one constant
  //     mass flux q_e solves q_e.G_a = R^e_a for every vertex a;
  //  3. v = q_e/rho at each quadrature point.
  //With uniform kr on an acute mesh and no FCT correction, q_e is the element mean
  //of the Darcy mass flux -rho kr K_s (grad psi - (rho/rho0) g).
  //
  //Must run after the Newton solve and FCTStep, since it needs the final
  //limitedFlux.
  //
  //PARALLEL: an owned row's edges and their element stars are all local, so its sum
  //is exact.  Each edge takes its value from its lower-numbered row, and owned DOFs
  //are numbered before ghosts, so a ghost row never overrides an owned one and every
  //element stays antisymmetric.
  void calculateEdgeVelocity(arguments_dict &args)
  {
    proteus::pyarray<double> &mesh_trial_ref             = args.array<double>("mesh_trial_ref");
    proteus::pyarray<double> &mesh_grad_trial_ref        = args.array<double>("mesh_grad_trial_ref");
    proteus::pyarray<double> &mesh_dof                   = args.array<double>("mesh_dof");
    proteus::pyarray<int>    &mesh_l2g                   = args.array<int>("mesh_l2g");
    proteus::pyarray<double> &dV_ref                     = args.array<double>("dV_ref");
    proteus::pyarray<double> &u_grad_trial_ref           = args.array<double>("u_grad_trial_ref");
    int                  nElements_global           = args.scalar<int>("nElements_global");
    int                  numDOFs                    = args.scalar<int>("numDOFs");
    int                  NNZ                        = args.scalar<int>("NNZ");
    proteus::pyarray<int>    &a_rowptr                   = args.array<int>("a_rowptr");
    proteus::pyarray<int>    &a_colind                   = args.array<int>("a_colind");
    proteus::pyarray<int>    &elementMaterialTypes       = args.array<int>("elementMaterialTypes");
    proteus::pyarray<double> &KWs                        = args.array<double>("KWs");
    proteus::pyarray<double> &q_rho                      = args.array<double>("q_rho");
    proteus::pyarray<double> &q_mu                       = args.array<double>("q_mu");
    double               dt                         = args.scalar<double>("dt");
    proteus::pyarray<double> &edgeFluxLow                = args.array<double>("edgeFluxLow");
    proteus::pyarray<double> &limitedFlux                = args.array<double>("limitedFlux");
    proteus::pyarray<int>    &csrRowIndeces_DofLoops     = args.array<int>("csrRowIndeces_DofLoops");
    proteus::pyarray<int>    &csrColumnOffsets_DofLoops  = args.array<int>("csrColumnOffsets_DofLoops");
    proteus::pyarray<int>    &csrTransposeIndices        = args.array<int>("csrTransposeIndices");
    proteus::pyarray<int>    &csrRowIndeces_CellLoops    = args.array<int>("csrRowIndeces_CellLoops");
    proteus::pyarray<int>    &csrColumnOffsets_CellLoops = args.array<int>("csrColumnOffsets_CellLoops");
    proteus::pyarray<double> &velocity_edge              = args.array<double>("velocity_edge");
    //step 2 relies on the vertex gradients of a P1 simplex
    assert(nDOF_trial_element == nSpace + 1);

    //antisymmetric realized edge flux; the lower-numbered row wins
    std::vector<double> E(NNZ, 0.0);
    for (int i = 0; i < numDOFs; i++)
      for (int ij = csrRowIndeces_DofLoops.data()[i]; ij < csrRowIndeces_DofLoops.data()[i + 1]; ij++) {
        const int j  = csrColumnOffsets_DofLoops.data()[ij];
        const int ji = csrTransposeIndices.data()[ij];
        E[ij] = (i <= j) ?  (edgeFluxLow.data()[ij] + limitedFlux.data()[ij] / dt)
                         : -(edgeFluxLow.data()[ji] + limitedFlux.data()[ji] / dt);
      }
    //per edge: sum over its elements of |S^e_ij|, and the number of elements
    std::vector<double> W_sum(NNZ, 0.0), n_e(NNZ, 0.0);
    //pass 0 fills W_sum and n_e; pass 1 splits E, solves for q_e and writes v
    for (int pass = 0; pass < 2; pass++)
      for (int eN = 0; eN < nElements_global; eN++) {
        const int mat = elementMaterialTypes.data()[eN];
        double S[nDOF_test_element][nDOF_trial_element], G[nDOF_test_element][nSpace];
        for (int a = 0; a < nDOF_test_element; a++) {
          for (int b = 0; b < nDOF_trial_element; b++) S[a][b] = 0.0;
          for (int I = 0; I < nSpace; I++) G[a][I] = 0.0;
        }
        for (int k = 0; k < nQuadraturePoints_element; k++) {
          const int eN_k = eN * nQuadraturePoints_element + k;
          double jac[nSpace * nSpace], jacDet, jacInv[nSpace * nSpace], x, y, z;
          double u_grad_trial[nDOF_trial_element * nSpace], as_w[nnz];
          ck.calculateMapping_element(eN, k, mesh_dof.data(), mesh_l2g.data(), mesh_trial_ref.data(), mesh_grad_trial_ref.data(), jac, jacDet, jacInv, x, y, z);
          const double dV = fabs(jacDet) * dV_ref.data()[k];
          ck.gradTrialFromRef(&u_grad_trial_ref.data()[k * nDOF_trial_element * nSpace], jacInv, u_grad_trial);
          //the same K_s/mu the residual built T from
          for (int ii = 0; ii < nnz; ii++) as_w[ii] = q_rho.data()[eN_k] * KWs.data()[mat * nnz + ii] / q_mu.data()[eN_k];
          for (int a = 0; a < nDOF_test_element; a++) {
            double grad_w_dV[nSpace];
            for (int I = 0; I < nSpace; I++) {
              grad_w_dV[I] = u_grad_trial[a * nSpace + I] * dV;
              G[a][I] += grad_w_dV[I];
            }
            for (int b = 0; b < nDOF_trial_element; b++)
              S[a][b] += ck.SimpleDiffusionJacobian_weak(a_rowptr.data(), a_colind.data(), as_w, &u_grad_trial[b * nSpace], grad_w_dV);
          }
        }
        double R[nDOF_test_element];
        for (int a = 0; a < nDOF_test_element; a++) {
          const int eN_a = eN * nDOF_test_element + a;
          R[a] = 0.0;
          for (int b = 0; b < nDOF_trial_element; b++) {
            if (a == b) continue;
            const int ab = csrRowIndeces_CellLoops.data()[eN_a] + csrColumnOffsets_CellLoops.data()[eN_a * nDOF_trial_element + b];
            if (pass == 0) {
              W_sum[ab] += fabs(S[a][b]);
              n_e[ab]   += 1.0;
            } else {
              const double share = (W_sum[ab] > 0.0) ? fabs(S[a][b]) / W_sum[ab] : 1.0 / n_e[ab];
              R[a] += share * E[ab];
            }
          }
        }
        if (pass == 0) continue;
        //G q = R is consistent (sum_a R_a = 0 = sum_a G_a), so the normal equations
        //(G^T G) q = G^T R give its exact solution
        double N[nSpace][nSpace], q[nSpace];
        for (int I = 0; I < nSpace; I++) {
          q[I] = 0.0;
          for (int a = 0; a < nDOF_test_element; a++) q[I] += G[a][I] * R[a];
          for (int J = 0; J < nSpace; J++) {
            N[I][J] = 0.0;
            for (int a = 0; a < nDOF_test_element; a++) N[I][J] += G[a][I] * G[a][J];
          }
        }
        //N is symmetric positive definite, so elimination needs no pivoting
        for (int I = 0; I < nSpace; I++)
          for (int J = I + 1; J < nSpace; J++) {
            const double l = N[J][I] / N[I][I];
            for (int K = I; K < nSpace; K++) N[J][K] -= l * N[I][K];
            q[J] -= l * q[I];
          }
        for (int I = nSpace - 1; I >= 0; I--) {
          for (int K = I + 1; K < nSpace; K++) q[I] -= N[I][K] * q[K];
          q[I] /= N[I][I];
        }
        for (int k = 0; k < nQuadraturePoints_element; k++) {
          const int eN_k = eN * nQuadraturePoints_element + k;
          for (int I = 0; I < nSpace; I++)
            velocity_edge.data()[eN_k * nSpace + I] = q[I] / q_rho.data()[eN_k];
        }
      }
  }


  // Limiter of the fully implicit FCT (Dielmann, Kuzmin & Kees, JNMA), steps 2-5:
  // raw fluxes at the high-order predictor p^H, nodal-variation prelimiter, Zalesak with GLOBAL
  // bounds [s_min, s_max] relative to s^n, and the corrected datum
  //   s^{n,FCT}_i = s^n_i + (1/m_i) sum_j alpha_ij fbar_ij.
  // It returns fluxCorrection_i = -bc_mask_i (s^{n,FCT}_i - s^n_i) m_i / dt, which the low-order
  // residual adds under STABILIZATION_TYPE=Implicit_FCT, i.e. the final solve
  //   M_L (s(p) - s^{n,FCT})/dt + A~(p) u(p) = 0.
  // Inputs are evaluated by calculateResidual_entropy_viscosity at p^H: mLow = s(p^H), mn = s(p^n),
  // dt_times_fH_minus_fL = dt (theta a^H + (1-theta) a^n - a~^H)_ij (u_i - u_j), rho_dof_member.
  // fct_pass (parallel; ghost values of the outputs must be scattered between the passes):
  //   1 -> alpha_nv;  2 -> FluxCorrectionMatrix (= dt fbar), Rpos, Rneg;  3 -> fluxCorrection;
  //   0 -> all three back to back (serial).
  void ImplicitFCTStep(arguments_dict &args)
  {
    proteus::pyarray<double> &bc_mask                   = args.array<double>("bc_mask");
    int                  numDOFs                   = args.scalar<int>("numDOFs");
    double               dt                        = args.scalar<double>("dt");
    double               rho                       = args.scalar<double>("rho");
    proteus::pyarray<double> &gravity                   = args.array<double>("gravity");
    proteus::pyarray<double> &mesh_dof                  = args.array<double>("mesh_dof");
    proteus::pyarray<double> &u_dof                     = args.array<double>("u_dof");   // p^H
    proteus::pyarray<double> &ML                        = args.array<double>("ML");
    proteus::pyarray<double> &MC                        = args.array<double>("MC");
    proteus::pyarray<double> &mn                        = args.array<double>("mn");      // s(p^n)
    proteus::pyarray<double> &mLow                      = args.array<double>("mLow");    // s(p^H)
    proteus::pyarray<double> &dt_times_fH_minus_fL      = args.array<double>("dt_times_fH_minus_fL");
    proteus::pyarray<int>    &csrRowIndeces_DofLoops    = args.array<int>("csrRowIndeces_DofLoops");
    proteus::pyarray<int>    &csrColumnOffsets_DofLoops = args.array<int>("csrColumnOffsets_DofLoops");
    proteus::pyarray<double> &thetaR                    = args.array<double>("thetaR");
    proteus::pyarray<double> &thetaSR                   = args.array<double>("thetaSR");
    proteus::pyarray<int>    &freeDOFMaterialTypes      = args.array<int>("freeDOFMaterialTypes");
    double               prelimiter_power          = args.scalar<double>("prelimiter_power");
    // detector variable: 0 = pressure head p^H (JNMA MFEM), 1 = hydraulic head p^H + z
    int                  prelimiter_head           = args.scalar<int>("prelimiter_head");
    int                  fct_pass                  = args.scalar<int>("fct_pass");
    proteus::pyarray<double> &alpha_nv                  = args.array<double>("alpha_nv");
    proteus::pyarray<double> &Rpos                      = args.array<double>("Rpos");
    proteus::pyarray<double> &Rneg                      = args.array<double>("Rneg");
    proteus::pyarray<double> &FluxCorrectionMatrix      = args.array<double>("FluxCorrectionMatrix");
    proteus::pyarray<double> &fluxCorrection            = args.array<double>("fluxCorrection");
    proteus::pyarray<double> &limited_solution          = args.array<double>("limited_solution"); // s^{n,FCT}
    proteus::pyarray<double> &limitedFluxMatrix         = args.array<double>("limitedFluxMatrix");
    proteus::pyarray<double> &limitedSpatialFluxMatrix  = args.array<double>("limitedSpatialFluxMatrix");
    const bool have_rho_dof = (rho_dof_member.size() == static_cast<std::size_t>(numDOFs));

    if (fct_pass == 0 || fct_pass == 1) {
      // nodal-variation prelimiter over N_i* = {j != i : m_ij > 0}, on the pressure head p^H
      // (prelimiter_head = 0) or on the hydraulic head p - (rho_i/rho) g.x as in the residual's
      // edge loop (prelimiter_head = 1). prelimiter_power <= 0 switches it off.
      std::vector<double> head(numDOFs);
      for (int i = 0; i < numDOFs; i++) {
        const double rho_i = have_rho_dof ? rho_dof_member[i] : rho;
        head[i] = u_dof.data()[i];
        if (prelimiter_head)
          for (int I = 0; I < nSpace; I++) head[i] -= (rho_i / rho) * gravity.data()[I] * mesh_dof.data()[i * 3 + I];
      }
      for (int i = 0; i < numDOFs; i++) {
        double num = 0.0, den = 0.0;
        for (int offset = csrRowIndeces_DofLoops.data()[i]; offset < csrRowIndeces_DofLoops.data()[i + 1]; offset++) {
          const int j = csrColumnOffsets_DofLoops.data()[offset];
          if (j == i || MC.data()[offset] <= 0.0) continue;
          num += head[i] - head[j];
          den += fabs(head[i] - head[j]);
        }
        alpha_nv.data()[i] = (den > 0.0 && prelimiter_power > 0.0) ? 1.0 - std::pow(fabs(num) / den, prelimiter_power) : 1.0;
      }
    }
    if (fct_pass == 0 || fct_pass == 2) {
      int ij = 0;
      for (int i = 0; i < numDOFs; i++) {
        const int    mat_i = freeDOFMaterialTypes.data()[i];
        const double rho_i = have_rho_dof ? rho_dof_member[i] : rho;
        // global bounds of the retention curve (beta = 0): s in [rho thetaR, rho (thetaR + thetaSR)]
        const double s_min  = rho_i * thetaR.data()[mat_i];
        const double s_max  = rho_i * (thetaR.data()[mat_i] + thetaSR.data()[mat_i]);
        const double sDot_i = (mLow.data()[i] - mn.data()[i]) / dt;
        double Pposi = 0.0, Pnegi = 0.0;
        for (int offset = csrRowIndeces_DofLoops.data()[i]; offset < csrRowIndeces_DofLoops.data()[i + 1]; offset++) {
          const int    j      = csrColumnOffsets_DofLoops.data()[offset];
          const double sDot_j = (mLow.data()[j] - mn.data()[j]) / dt;
          const double f      = dt * MC.data()[ij] * (sDot_i - sDot_j) + dt_times_fH_minus_fL.data()[ij];
          FluxCorrectionMatrix.data()[ij] = (j == i) ? 0.0 : fmin(alpha_nv.data()[i], alpha_nv.data()[j]) * f;
          Pposi += fmax(0.0, FluxCorrectionMatrix.data()[ij]);
          Pnegi += fmin(0.0, FluxCorrectionMatrix.data()[ij]);
          ij += 1;
        }
        const double Qposi = ML.data()[i] * fmax(0.0, s_max - mn.data()[i]);
        const double Qnegi = ML.data()[i] * fmin(0.0, s_min - mn.data()[i]);
        Rpos.data()[i] = (Pposi == 0.0) ? 1.0 : fmin(1.0, Qposi / Pposi);
        Rneg.data()[i] = (Pnegi == 0.0) ? 1.0 : fmin(1.0, Qnegi / Pnegi);
      }
    }
    if (fct_pass == 0 || fct_pass == 3) {
      int ij = 0;
      for (int i = 0; i < numDOFs; i++) {
        double sum = 0.0;
        for (int offset = csrRowIndeces_DofLoops.data()[i]; offset < csrRowIndeces_DofLoops.data()[i + 1]; offset++) {
          const int    j = csrColumnOffsets_DofLoops.data()[offset];
          const double f       = FluxCorrectionMatrix.data()[ij];
          const double L_ij    = (f > 0.0) ? fmin(Rpos.data()[i], Rneg.data()[j]) : fmin(Rneg.data()[i], Rpos.data()[j]);
          const double alpha_ij = (j == i) ? 0.0 : L_ij * fmin(alpha_nv.data()[i], alpha_nv.data()[j]);
          sum += L_ij * f;
          limitedFluxMatrix.data()[ij]        = bc_mask.data()[i] * L_ij * f;
          limitedSpatialFluxMatrix.data()[ij] = bc_mask.data()[i] * alpha_ij * dt_times_fH_minus_fL.data()[ij];
          ij += 1;
        }
        limited_solution.data()[i] = mn.data()[i] + bc_mask.data()[i] * sum / ML.data()[i];
        fluxCorrection.data()[i]   = -bc_mask.data()[i] * sum / dt;
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
    //Heap-backed members, not stack VLAs -- see the declarations in the class body.
    Rpos_member.assign(numDOFs, 0.0);
    Rneg_member.assign(numDOFs, 0.0);
    std::vector<double> &Rpos = Rpos_member;
    std::vector<double> &Rneg = Rneg_member;
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
    // L-scheme Jacobian (Pop, Radu & Knabner; List & Radu; JNMA defect correction): LSCHEME_L > 0 assembles
    // J = (L/dt) M_L + A~(u^k) -- dm/du replaced by L, upwind k_r frozen (no dk_r terms). The residual is unchanged.
    const double         LSCHEME_L                 = args.scalar<double>("LSCHEME_L");
    const double         dKr_on                    = (LSCHEME_L > 0.0) ? 0.0 : 1.0;
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
    proteus::pyarray<double> &q_mu                                      = args.array<double>("q_mu");
    proteus::pyarray<double> &ebqe_mu                                   = args.array<double>("ebqe_mu");
    proteus::pyarray<double> &gravity                                    = args.array<double>("gravity");
    proteus::pyarray<double> &alpha                                      = args.array<double>("alpha");
    proteus::pyarray<double> &n                                          = args.array<double>("n");
    proteus::pyarray<double> &thetaR                                     = args.array<double>("thetaR");
    proteus::pyarray<double> &thetaSR                                    = args.array<double>("thetaSR");
    PSK_TYPE_member = args.scalar<int>("PSK_TYPE");
    VG_AIR_ENTRY_member = args.scalar<double>("VG_AIR_ENTRY");
    proteus::pyarray<double> &KWs                                        = args.array<double>("KWs");
    double               useMetrics                                 = args.scalar<double>("useMetrics");
    double               alphaBDF                                   = args.scalar<double>("alphaBDF");
    int                  lag_shockCapturing                         = args.scalar<int>("lag_shockCapturing");
    double               shockCapturingDiffusion                    = args.scalar<double>("shockCapturingDiffusion");
    double               sc_uref                                    = args.scalar<double>("sc_uref");
    double               sc_alpha                                   = args.scalar<double>("sc_alpha");
    proteus::pyarray<int>    &u_l2g                                      = args.array<int>("u_l2g");
    proteus::pyarray<int>    &r_l2g                                      = args.array<int>("r_l2g");
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
    proteus::pyarray<double> &globalResidual                             = args.array<double>("globalResidual");
    int                  nExteriorElementBoundaries_global          = args.scalar<int>("nExteriorElementBoundaries_global");
    proteus::pyarray<int>    &exteriorElementBoundariesArray             = args.array<int>("exteriorElementBoundariesArray");
    proteus::pyarray<int>    &elementBoundaryElementsArray               = args.array<int>("elementBoundaryElementsArray");
    proteus::pyarray<int>    &elementBoundaryLocalElementBoundariesArray = args.array<int>("elementBoundaryLocalElementBoundariesArray");
    proteus::pyarray<double> &ebqe_velocity_ext                          = args.array<double>("ebqe_velocity_ext");
    proteus::pyarray<int>    &isDOFBoundary_u                            = args.array<int>("isDOFBoundary_u");
    proteus::pyarray<double> &ebqe_bc_u_ext                              = args.array<double>("ebqe_bc_u_ext");
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
    int               NNZ                           = args.scalar<int>("NNZ");
    proteus::pyarray<int> &csrRowIndeces_DofLoops        = args.array<int>("csrRowIndeces_DofLoops");
    proteus::pyarray<int> &csrColumnOffsets_DofLoops     = args.array<int>("csrColumnOffsets_DofLoops");
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
    // FOR FCT
    proteus::pyarray<double> &dLow                 = args.array<double>("dLow");
    proteus::pyarray<double> &fluxMatrix           = args.array<double>("fluxMatrix");
    proteus::pyarray<double> &mDotLow              = args.array<double>("mDotLow");
    proteus::pyarray<double> &mLow                 = args.array<double>("mLow");
    proteus::pyarray<double> &dt_times_fH_minus_fL = args.array<double>("dt_times_fH_minus_fL");
    proteus::pyarray<double> &min_m_bc             = args.array<double>("min_m_bc");
    proteus::pyarray<double> &max_m_bc             = args.array<double>("max_m_bc");
    //weak-Dirichlet block of the transport operator, used only when RICHARDS_FCT_BND_DIFF
    //is on.  See the flag's comment above.
    proteus::pyarray<double> &TransportBoundary    = args.array<double>("TransportBoundary");
    //Step B changes the RESIDUAL (TB unclipped instead of through the fmax clip), so it is
    //tied to FCT, where it keeps the limited correction conservative; an FCT=False run keeps
    //the old routing.  Python passes coefficients.FCT here; the env var can still force it
    //off for an A/B but can never turn it on for an FCT=False run.
    const bool useBndDiff = (args.scalar<int>("FCT_BND_DIFF") != 0) && RICHARDS_FCT_BND_DIFF;
    const bool useBndKrw  = (args.scalar<int>("FCT_BND_KRW") != 0) && RICHARDS_FCT_BND_KRW;
    // AUX QUANTITIES OF INTEREST
    proteus::pyarray<double> &quantDOFs = args.array<double>("quantDOFs");
    proteus::pyarray<double> &mn        = args.array<double>("mn");
    // 1 => rho_nodal holds the transport model's nodal rho(c_i) and replaces the
    // L2 projection of q_rho below.
    proteus::pyarray<double> &rho_nodal = args.array<double>("rho_nodal");
    const int USE_RHO_NODAL = args.scalar<int>("USE_RHO_NODAL");
    // 1 => rho_nodal_old holds the nodal rho of the previous accepted step, and
    // the old-time mass mn is evaluated with it (see the DOF loop).
    proteus::pyarray<double> &rho_nodal_old = args.array<double>("rho_nodal_old");
    const int USE_RHO_NODAL_OLD = args.scalar<int>("USE_RHO_NODAL_OLD");
    // NODAL water content theta(psi^n), the theta of mn.  Exported for coupled
    // transport models.  Not read by this model.
    proteus::pyarray<double> &thetaW_dof = args.array<double>("thetaW_dof");
    proteus::pyarray<double> &fluxCorrection        = args.array<double>("fluxCorrection");
    proteus::pyarray<double> &limited_solution          = args.array<double>("limited_solution");
    proteus::pyarray<int>    &freeDOFMaterialTypes      = args.array<int>("freeDOFMaterialTypes");

    proteus::pyarray<double> &velocity_couple               = args.array<double>("velocity_couple");
    proteus::pyarray<double> &ebqe_velocity_ext_couple      = args.array<double>("ebqe_velocity_ext_couple");
    // BOUNDARY-FLUX DIAGNOSTIC.  ebqe_flux gets the FULL flux from
    // exteriorNumericalFlux2, but with useConsistentFlux = false the residual
    // integrates bflux_ext, which that routine sets to the PENALTY TERM ALONE
    // (Richards.h:320-321) -- the Darcy consistency term v.n is dropped.  So
    // ebqe_flux is NOT the flux this scheme applies, and neither is the
    // velocity_couple trace handed to TADR, which is the consistency term and
    // nothing else.  Python cannot see bflux_ext at all, so export it, plus the
    // quadrature weight, and let the diagnostic in Richards.py do the split
    // exactly instead of estimating it from penalty*(u - bc_u).
    proteus::pyarray<double> &ebqe_bflux                    = args.array<double>("ebqe_bflux");
    proteus::pyarray<double> &ebqe_dS                       = args.array<double>("ebqe_dS");
    // CONSERVATIVE NODAL BOUNDARY FLUX, exported for the transport coupling.
    // At a boundary DOF i the water flux this scheme applies is the row's
    // penalty part (bflux, integrated below) PLUS the IIPG consistency rows
    // that are folded into TransportMatrix (see the exterior loop) and reach
    // the residual as non-telescoping edge fluxes.  At convergence their sum
    // equals the flux of the SYMMETRIC interior operator out of node i,
    //     F_i = sum_j Kr_up max(0, -T^int_ij) dphi_ij,    T^int = T - TB,
    // i.e. the Hughes consistent flux, which telescopes exactly with the
    // interior edge fluxes.  The pointwise trace -(a/rho)(grad psi - rho_r g).n
    // that ebqe_velocity_ext_couple carries is NOT that number: on Simmons SMD
    // level 4 it reads 0.29x of the corner singularity, 1.5-3x the flux across a
    // one-cell density jump, and 0.83-0.86x the open-top outflow.  The second
    // exterior pass at the end of this function therefore replaces the normal
    // component of the coupled trace by F_i/(rho S_i), the same way the cell
    // loop builds the interior trace -- no Python post-processing, one extra
    // sweep over the boundary quadrature points per residual call.  Units are
    // the residual's (rho-weighted); outward positive.  Only the Theta part is
    // carried -- Theta is hardwired to 1 in Richards.py.
    proteus::pyarray<double> &consBoundaryFlux              = args.array<double>("consBoundaryFlux");
    // OUT: per edge, the low-order water flux into row i from column j on the
    // INTERIOR (symmetric) part of the transport matrix, same units as the
    // residual's edge flux.  Antisymmetric by construction, so a coupled
    // transport model can advect with it and conserve.  Theta part only
    // (Theta is hardwired to 1).
    proteus::pyarray<double> &edgeFluxLow                   = args.array<double>("edgeFluxLow");
    // the interior low-order flux fL_int of edge ij alone (boundary part excluded); read by the
    // edge-flux TADR path (STABILIZATION_TYPE 2-5) and calculateEdgeVelocity
    proteus::pyarray<double> &edgeFluxInterior              = args.array<double>("edgeFluxInterior");
    // IN: nodal water source [mass/(volume time)], added to row i as ML_i*nodalSource_i
    // (lumped).  Zero unless the deck passes Coefficients(mms_source=...), e.g. the
    // flow source of a manufactured solution.
    proteus::pyarray<double> &nodalSource                   = args.array<double>("nodalSource");
    // proteus::pyarray<double> &q_x    = args.array<double>("q_x");
    // proteus::pyarray<double> &ebqe_x = args.array<double>("ebqe_x");
    
    proteus::pyarray<double> &anb_seepage_flux_n = args.array<double>("anb_seepage_flux_n");
    proteus::pyarray<double> &q_velocity = args.array<double>("q_velocity");
    double &anb_seepage_flux(args.scalar<double>("anb_seepage_flux"));
    anb_seepage_flux = 0.0;
    proteus::pyarray<int>    &csrRowIndeces_u_u                          = args.array<int>("csrRowIndeces_u_u");
    proteus::pyarray<int>    &csrColumnOffsets_u_u                       = args.array<int>("csrColumnOffsets_u_u");
    proteus::pyarray<int>    &csrColumnOffsets_eb_u_u                    = args.array<int>("csrColumnOffsets_eb_u_u");
    // Compatible flow -> transport coupling (TADR STABILIZATION_TYPE=Compatible):
    //   USE_NODAL_RHO    -> rho_dof = rho_dof_in, the transport's nodal rho_T(c_i), instead of the
    //                       lumped projection of q_rho (which is not the nodal value at boundaries)
    //   edgeFluxLow_ij   <- the low-order edge flux of row i (what the residual sums over j)
    //   boundaryFluxDOF_i<- the boundary integral of row i, positive into the domain
    const int            USE_NODAL_RHO    = args.scalar<int>("USE_NODAL_RHO");
    proteus::pyarray<double> &rho_dof_in       = args.array<double>("rho_dof_in");
    proteus::pyarray<double> &boundaryFluxDOF  = args.array<double>("boundaryFluxDOF");
    std::fill(boundaryFluxDOF.begin(), boundaryFluxDOF.end(), 0.0);
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
    std::fill(consBoundaryFlux.data(), consBoundaryFlux.data() + consBoundaryFlux.size(), 0.0);
    // rows that own at least one exterior-face quadrature point; only these get
    // the consBoundaryFlux accumulation in the DOF loop (it needs an extra
    // closure evaluation per edge, so it is not done on interior rows)
    std::vector<char> is_bdry_dof(numDOFs, 0);
    // Bookkeeping for the conservative boundary velocity below: per DOF, the
    // measure of the weakly imposed (Dirichlet) part of its boundary star and
    // the flux the prescribed-flux faces already pin there; per boundary
    // quadrature point, the outward normal, cached so the second pass does not
    // have to redo the element boundary mapping.
    std::vector<double> bdryDirichletArea(numDOFs, 0.0);
    std::vector<double> bdryPrescribedFlux(numDOFs, 0.0);
    std::vector<double> ebqe_normal_cache(RICHARDS_BC_CONS ? nExteriorElementBoundaries_global * nQuadraturePoints_elementBoundary * nSpace : 0, 0.0);
    
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
      TransportBoundary.data()[i]   = 0.;
    }
    //min/max_m_bc are only ever fmin/fmax'd into below, so without this they would be the
    //widest window ever seen rather than this step's: a time dependent Dirichlet or a
    //moving seepage active set would keep loosening the FCT bound and never tighten it.
    //RICHARDS_FCT_BND_RESET=0 restores the old accumulate-forever behaviour.
    if (RICHARDS_FCT_BND_RESET)
      for (int i = 0; i < numDOFs; i++) {
        min_m_bc.data()[i] =  1.0e10;
        max_m_bc.data()[i] = -1.0e10;
      }

    // Project quadrature density to nodal DOFs before constructing nodal
    // stabilization potentials so Phi uses a true nodal density field.
    // With USE_RHO_NODAL the transport model's nodal rho(c_i) is used instead:
    // lumped mass evaluates storage AT the node, where the value is rho(c_i),
    // and the projection smooths it.  (Linear EOS: interpolating rho(c_i)
    // reproduces q_rho exactly, so nodes and quadrature points then agree.)
    if (USE_RHO_NODAL) {
      for (int i = 0; i < numDOFs; ++i) rho_dof[i] = rho_nodal.data()[i];
    } else {
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
    } // USE_RHO_NODAL
    if (USE_NODAL_RHO)
      for (int i = 0; i < numDOFs; ++i) rho_dof[i] = rho_dof_in.data()[i];
    // Cache the projected, salinity-coupled density for use in invert() so
    // the m -> u inversion is consistent with the forward residual.
    rho_dof_member = rho_dof;

    // compute entropy and init global_entropy_residual and boundary_integral
    //Heap-backed members, not stack VLAs -- see the declarations in the class body.
    psi_member.assign(numDOFs, 0.0);
    eta_member.assign(numDOFs, 0.0);
    global_entropy_residual_member.assign(numDOFs, 0.0);
    boundary_integral_member.assign(numDOFs, 0.0);
    std::vector<double> &psi                     = psi_member;
    std::vector<double> &eta                     = eta_member;
    std::vector<double> &global_entropy_residual = global_entropy_residual_member;
    std::vector<double> &boundary_integral       = boundary_integral_member;
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
          // Match the main variable-density operator: grad(u) - (rho/rho0) g.
          Phi[j]   -= (rho_node_j / rho) * mesh_dof.data()[x_gj * 3 + I] * gravity[I];
          Phi_n[j] -= (rho_node_j / rho) * mesh_dof.data()[x_gj * 3 + I] * gravity[I];
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
        // conductivity K_s/mu with the transported viscosity: T, a and the coupled
        // velocity below all carry it, the nodal upwind k_r and storage do not
        double KWs_mu[nnz];
        for (int ii = 0; ii < nnz; ii++) KWs_mu[ii] = KWs.data()[elementMaterialTypes[eN] * nnz + ii] / q_mu.data()[eN_k];

        evaluateCoefficients(a_rowptr.data(), a_colind.data(), rho, rho_local, beta, gravity.data(), alpha.data()[elementMaterialTypes[eN]], n.data()[elementMaterialTypes[eN]], thetaR.data()[elementMaterialTypes[eN]], thetaSR.data()[elementMaterialTypes[eN]],
                             KWs_mu, un, mn, dmn, fn, dfn, an, dan, asn, Krn, dKrn, thetaWn);
        evaluateCoefficients(a_rowptr.data(), a_colind.data(), rho, rho_local, beta, gravity.data(), alpha.data()[elementMaterialTypes[eN]], n.data()[elementMaterialTypes[eN]], thetaR.data()[elementMaterialTypes[eN]], thetaSR.data()[elementMaterialTypes[eN]],
                             KWs_mu, u, m, dm, f, df, a, da, as, Kr, dKr, thetaW);
        q_theta.data()[eN_k] = thetaW;

        // Darcy velocity for coupling should use the direct FE gradient of the
        // pressure head. The Phi-based gradients are only for stabilization.
        for (int I = 0; I < nSpace; ++I) {
          q_velocity.data()[eN_k_nSpace + I] = grad_u_velocity[I];
        }

        double pressure_gradient[nSpace];
        const double rho_ratio = rho_velocity / rho;
        for (int J = 0; J < nSpace; ++J)
          pressure_gradient[J] = grad_u_velocity[J] - rho_ratio * gravity.data()[J];

        for (int I = 0; I < nSpace; ++I) {
          double acc = 0.0;
          for (int ii = a_rowptr.data()[I]; ii < a_rowptr.data()[I+1]; ++ii) {
            const int J = a_colind.data()[ii];
            acc += (a[ii] / rho_velocity) * pressure_gradient[J];
          }
          velocity.data()[eN_k_nSpace + I] = -acc;
          velocity_couple.data()[eN_k_nSpace + I] = -acc;
        }
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
          ML2[u_l2g.data()[eN_i]] += u_test_dV[i];
          if (STABILIZATION_TYPE == STABILIZATION::EV_Stab) // EV stab
          {
            int    gi                 = offset_u + stride_u * u_l2g.data()[eN_i]; //global i-th index
            double uni = u_dof_old.data()[gi];
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
        int gi   = offset_u + stride_u * r_l2g.data()[eN_i]; //global i-th index
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
      
    } 

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
          bc_u_ext = 0.0, bc_grad_u_ext[nSpace], bc_m_ext = 0.0, bc_dm_ext = 0.0, bc_f_ext[nSpace], bc_df_ext[nSpace], bc_a_ext[nnz], bc_da_ext[nnz], bc_as_ext[nnz], jac_ext[nSpace * nSpace], jacDet_ext, jacInv_ext[nSpace * nSpace], boundaryJac[nSpace * (nSpace - 1)], metricTensor[(nSpace - 1) * (nSpace - 1)], metricTensorDetSqrt, dS, u_test_dS[nDOF_test_element], u_grad_trial_trace[nDOF_trial_element * nSpace], normal[3], x_ext, y_ext, z_ext, xt_ext, yt_ext, zt_ext, integralScaling, G[nSpace * nSpace], G_dd_G, tr_G, fluxJacobian_u_u[nDOF_trial_element], bfluxJacobian_u_u[nDOF_trial_element], fluxJacobian_un_un[nDOF_trial_element], cfluxJacobian_u_u[nDOF_trial_element];
        for (int j = 0; j < nDOF_trial_element; j++) {
          fluxJacobian_u_u[j] = 0.0; bfluxJacobian_u_u[j] = 0.0; fluxJacobian_un_un[j] = 0.0; cfluxJacobian_u_u[j] = 0.0;
        }
        //\
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
        double KWs_mu[nnz];
        for (int ii = 0; ii < nnz; ii++) KWs_mu[ii] = KWs.data()[elementMaterialTypes.data()[eN] * nnz + ii] / ebqe_mu.data()[ebNE_kb];

        evaluateCoefficients(a_rowptr.data(), a_colind.data(), rho, rho_ext, beta, gravity.data(), alpha.data()[elementMaterialTypes.data()[eN]], n.data()[elementMaterialTypes.data()[eN]], thetaR.data()[elementMaterialTypes.data()[eN]],
                             thetaSR.data()[elementMaterialTypes.data()[eN]], KWs_mu, u_ext, m_ext, dm_ext, f_ext, df_ext, a_ext, da_ext, as_ext, bc_Kr, bc_dKr, thetaW_ext);
        evaluateCoefficients(a_rowptr.data(), a_colind.data(), rho, rho_ext, beta, gravity.data(), alpha.data()[elementMaterialTypes.data()[eN]], n.data()[elementMaterialTypes.data()[eN]], thetaR.data()[elementMaterialTypes.data()[eN]],
                             thetaSR.data()[elementMaterialTypes.data()[eN]], KWs_mu, un_ext, mn_ext, dmn_ext, fn_ext, dfn_ext, an_ext, dan_ext, asn_ext, bc_Krn, bc_dKrn, thetaWn_ext);
        evaluateCoefficients(a_rowptr.data(), a_colind.data(), rho, rho_ext, beta, gravity.data(), alpha.data()[elementMaterialTypes.data()[eN]], n.data()[elementMaterialTypes.data()[eN]], thetaR.data()[elementMaterialTypes.data()[eN]],
                             thetaSR.data()[elementMaterialTypes.data()[eN]], KWs_mu, bc_u_ext, bc_m_ext, bc_dm_ext, bc_f_ext, bc_df_ext, bc_a_ext, bc_da_ext, bc_as_ext, bc_Kr_ext,bc_dKr_ext, thetaW_bc_ext);
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
        const double rho_ratio_ext = rho_velocity_ext / rho;
        for (int J = 0; J < nSpace; ++J)
          ext_pressure_gradient[J] = grad_u_ext[J] - rho_ratio_ext * gravity.data()[J];

        for (int I = 0; I < nSpace; ++I) {
          double acc = 0.0;
          for (int ii = a_rowptr.data()[I]; ii < a_rowptr.data()[I+1]; ++ii) {
            const int J = a_colind.data()[ii];
            acc += (a_ext[ii] / rho_velocity_ext) * ext_pressure_gradient[J];
          }
          ebqe_velocity_ext.data()[ebNE_kb_nSpace + I] = -acc;
          ebqe_velocity_ext_couple.data()[ebNE_kb_nSpace + I] = -acc;
        }


        //
        //calculate the numerical fluxes
        //
        bool useConsistentFlux=false;
        if (useConsistentFlux) {
          exteriorNumericalFlux(ebqe_bc_flux_ext[ebNE_kb], a_rowptr.data(), a_colind.data(),
                                isSeepageFace.data()[ebNE], //tricky, this is a face flag not face quad
                                isDOFBoundary_u.data()[ebNE_kb], normal, bc_u_ext, a_ext, grad_u_ext, u_ext, f_ext,
                                ebqe_penalty_ext.data()[ebNE_kb], // penalty,
                                flux_ext);
        } else {
          exteriorNumericalFlux2(ebqe_bc_flux_ext[ebNE_kb], a_rowptr.data(), a_colind.data(),
                              isSeepageFace.data()[ebNE], //tricky, this is a face flag not face quad
                              isDOFBoundary_u.data()[ebNE_kb], normal, bc_u_ext, a_ext, grad_u_ext, u_ext, f_ext,
                              ebqe_penalty_ext.data()[ebNE_kb], // penalty,
                              flux_ext, bflux_ext);
        }

        ebqe_flux.data()[ebNE_kb] = flux_ext;

        anb_seepage_flux             = seepagefluxcalculator(anb_seepage_flux, isSeepageFace.data()[ebNE], dS, flux_ext);
        anb_seepage_flux_n.data()[0] = anb_seepage_flux;
        ebqe_u.data()[ebNE_kb]       = u_ext;
        // see the note at the ebqe_bflux declaration: bflux_ext, not flux_ext,
        // is what the residual below integrates when useConsistentFlux = false
        ebqe_bflux.data()[ebNE_kb] = useConsistentFlux ? flux_ext : bflux_ext;
        ebqe_dS.data()[ebNE_kb]    = dS;
        //the coupled-velocity pass below needs the outward normal, and ebqe here
        //is hand-built and carries none
        if (RICHARDS_BC_CONS)
          for (int I = 0; I < nSpace; I++) ebqe_normal_cache[ebNE_kb_nSpace + I] = normal[I];
        //seed the FCT bounds with the imposed value. min_m_bc/max_m_bc are otherwise left at +-1e10, so
        //Zalesak's bounds at a weakly imposed Dirichlet DOF come only from the neighbours' low order masses
        //and the limiter clamps the boundary layer. theta(bc_u) uses the DOF's own soil and density so the
        //bound lies on the same retention curve as mLow in the edge loop below. done after the flux call so
        //the seepage active set is current
        if (isDOFBoundary_u.data()[ebNE_kb]) {
          double bc_u_dof = isSeepageFace.data()[ebNE] ? 0.0 : ebqe_bc_u_ext.data()[ebNE_kb];
          for (int i = 0; i < nDOF_test_element; i++) {
            if (u_test_trace_ref.data()[ebN_local_kb * nDOF_test_element + i] <= 1.0e-12) continue; //DOF not on this face
            int    free_gi = r_l2g.data()[eN * nDOF_test_element + i], mat_gi = freeDOFMaterialTypes.data()[free_gi];
            double m_bc, dm_bc, f_bc[nSpace], df_bc[nSpace], a_bc[nnz], da_bc[nnz], as_bc[nnz], Kr_bc, dKr_bc, thetaW_bc;
            evaluateCoefficients(a_rowptr.data(), a_colind.data(), rho, rho_dof[free_gi], beta, gravity.data(), alpha.data()[mat_gi],
                                 n.data()[mat_gi], thetaR.data()[mat_gi], thetaSR.data()[mat_gi], &KWs.data()[mat_gi * nnz], bc_u_dof,
                                 m_bc, dm_bc, f_bc, df_bc, a_bc, da_bc, as_bc, Kr_bc, dKr_bc, thetaW_bc);
            min_m_bc.data()[free_gi] = fmin(min_m_bc.data()[free_gi], m_bc);
            max_m_bc.data()[free_gi] = fmax(max_m_bc.data()[free_gi], m_bc);
          }
        }
        //
        //update residuals
        //
        for (int i = 0; i < nDOF_test_element; i++) {
          if (useConsistentFlux) {
            elementResidual_u[i] += ck.ExteriorElementBoundaryFlux(flux_ext, u_test_dS[i]);
          } else {
            elementResidual_u[i] += ck.ExteriorElementBoundaryFlux(bflux_ext, u_test_dS[i]);
          }
          // see the consBoundaryFlux note: mark the rows the DOF loop must
          // accumulate the conservative boundary flux for (face DOFs only, the
          // off-face vertex has a zero trace), and split this DOF's boundary
          // star into the weakly imposed part, whose flux the conservative pass
          // rewrites, and the part a flux BC already pins exactly.  isDOFBoundary
          // is read AFTER the flux call, so a seepage face that just shut counts
          // as prescribed (zero), which is what the residual got.
          if (u_test_trace_ref.data()[ebN_local_kb * nDOF_test_element + i] > 1.0e-12) {
            const int gi = r_l2g.data()[eN * nDOF_test_element + i];
            is_bdry_dof[gi] = 1;
            if (isDOFBoundary_u.data()[ebNE_kb])
              bdryDirichletArea[gi] += u_test_dS[i];
            else
              //prescribed: a flux BC, a seepage face that shut, or the
              //subdomain's artificial boundary -- all three carry bflux_ext
              //into the residual, and it is zero for the last two
              bdryPrescribedFlux[gi] += bflux_ext * u_test_dS[i];
          }
        } //i
        for (int j = 0; j < nDOF_trial_element; j++) {
          if (useConsistentFlux) {
          exteriorNumericalFluxJacobian(a_rowptr.data(), a_colind.data(), isDOFBoundary_u.data()[ebNE_kb], normal, a_ext, da_ext, grad_u_ext, &u_grad_trial_trace[j * nSpace], df_ext, u_trial_trace_ref.data()[ebN_local_kb * nDOF_test_element + j],
                                        ebqe_penalty_ext.data()[ebNE_kb], //penalty,
                                        fluxJacobian_u_u[j]);
          } else {
            exteriorNumericalFluxJacobian2(a_rowptr.data(), a_colind.data(), isDOFBoundary_u.data()[ebNE_kb], normal, as_ext, a_ext, da_ext, grad_u_ext, &u_grad_trial_trace[j * nSpace], df_ext, u_trial_trace_ref.data()[ebN_local_kb * nDOF_test_element + j],
                                        ebqe_penalty_ext.data()[ebNE_kb], //penalty,
                                        fluxJacobian_u_u[j],bfluxJacobian_u_u[j],cfluxJacobian_u_u[j]);
          }
          //probably need isDOFBoundary_un here
          //exteriorNumericalFluxJacobian(a_rowptr.data(), a_colind.data(), isDOFBoundary_u.data()[ebNE_kb], normal, asn_ext, dan_ext, grad_u_ext, &u_grad_trial_trace[j * nSpace], dfn_ext, u_trial_trace_ref.data()[ebN_local_kb * nDOF_test_element + j],
          //                              ebqe_penalty_ext.data()[ebNE_kb], //penalty,
          //                              fluxJacobian_un_un[j]);
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
              const int ij_eb = csrRowIndeces_u_u[eN_i] + csrColumnOffsets_eb_u_u[ebN_i_j];
              if (useBndDiff) {
                //Step B: park the boundary block in TB. TransportMatrix/Consistent stay pure
                //interior, so they keep the symmetry and zero row sums the edge loop relies on.
                TransportBoundary.data()[ij_eb] += cfluxJacobian_u_u[j] * u_test_dS[i];
              } else {
                TransportMatrix[ij_eb] += fluxJacobian_u_u[j] * u_test_dS[i];
                //also keep the IIPG block on its own, so the DOF loop can recover the
                //SYMMETRIC interior operator T = TransportMatrix - TransportBoundary for
                //the conservative boundary flux export (consBoundaryFlux).  Nothing else
                //reads TransportBoundary when useBndDiff is off, so the residual, the
                //Jacobian and the FCT=False solution are unchanged by this line.
                TransportBoundary.data()[ij_eb] += fluxJacobian_u_u[j] * u_test_dS[i];
                //k_rw-weighted: this array is read ONLY by fH in the edge loop below, so this
                //line cannot change the residual, the Jacobian or the FCT=False solution.
                //RICHARDS_FCT_BND_KRW=0 restores the old Ks-based boundary block for an A/B.
                TransportMatrixConsistent[ij_eb] +=
                  (useBndKrw ? cfluxJacobian_u_u[j] : fluxJacobian_u_u[j]) * u_test_dS[i];
              }
              TransportMatrixn[ij_eb] += fluxJacobian_un_un[j] * u_test_dS[i];
              TransportMatrixConsistentn[ij_eb] += fluxJacobian_un_un[j] * u_test_dS[i];
            }
          } //j
        } //i
        //Step B Jacobian, dK part.  TB_ij = int -(K grad(w_j)).n w_i dS with K = a_ext(psi_h),
        //so row i's boundary term Theta*sum_j TB_ij*dphi_ij also depends on psi through K.
        //The edge loop sees TB only as assembled numbers and adds Theta*TB_ij alone; the
        //derivative of K needs da_ext and the trace w_k at this quadrature point, so it is
        //added here:  d/du_k = Theta * sum_q w_k w_i dS * sum_j G_j dphi_ij,
        //G_j = -(dK grad(w_j)).n, with dphi_ij built exactly as in the edge loop.
        if (useBndDiff && !useConsistentFlux && isDOFBoundary_u.data()[ebNE_kb]) {
          for (int i = 0; i < nDOF_test_element; i++) {
            const int eN_i = eN * nDOF_test_element + i;
            const int gi   = r_l2g.data()[eN_i];
            double dK_flux = 0.0;
            for (int j = 0; j < nDOF_trial_element; j++) {
              const int gj = r_l2g.data()[eN_nDOF_trial_element + j];
              double G_j = 0.0;
              for (int I = 0; I < nSpace; I++)
                for (int m = a_rowptr.data()[I]; m < a_rowptr.data()[I + 1]; m++)
                  G_j -= da_ext[m] * u_grad_trial_trace[j * nSpace + a_colind.data()[m]] * normal[I];
              const double rho_edge = 0.5 * (rho_dof[gi] + rho_dof[gj]);
              double dphi = u_free_dof[gj] - u_free_dof[gi];
              for (int I = 0; I < nSpace; I++)
                dphi -= (rho_edge / rho) * gravity.data()[I] * (mesh_dof.data()[gj * 3 + I] - mesh_dof.data()[gi * 3 + I]);
              dK_flux += G_j * dphi;
            }
            for (int k = 0; k < nDOF_trial_element; k++) {
              const int ebN_i_k = ebN * 4 * nDOF_test_X_trial_element + i * nDOF_trial_element + k;
              globalJacobian.data()[csrRowIndeces_u_u[eN_i] + csrColumnOffsets_eb_u_u[ebN_i_k]] +=
                bc_mask.data()[gi] * Theta * dK_flux * u_trial_trace_ref.data()[ebN_local_kb * nDOF_test_element + k] * u_test_dS[i];
            }
          }
        }
      } //kb
      for (int i = 0; i < nDOF_test_element; i++) {
          int eN_i = eN * nDOF_test_element + i;
          globalResidual.data()[offset_u + stride_u * u_l2g.data()[eN_i]] += elementResidual_u[i];
          boundaryFluxDOF.data()[r_l2g.data()[eN_i]] -= elementResidual_u[i];
      }//i
    } //ebNE
    /////////////////////////////////////////////////////////////////
    // COMPUTE SMOOTHNESS INDICATOR and NORMALIZE ENTROPY RESIDUAL //
    /////////////////////////////////////////////////////////////////
    // NOTE: see NCLS.h for a different but equivalent implementation of this.
    int ij = 0;
    //Heap-backed member, not a stack VLA -- see the declarations in the class body.
    cflux_member.assign(numDOFs, 0.0);
    std::vector<double> &cflux = cflux_member;
    for (int i = 0; i < numDOFs; i++) {
      double gi[nSpace], Cij[nSpace], xi[nSpace], etaMaxi, etaMini;
      double solni = u_free_dof_old[i];
      for (int I = 0; I < nSpace; I++) {
        solni -= (rho_dof[i] / rho) * gravity.data()[I] * mesh_dof.data()[i * 3 + I];
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
        xi[I] = mesh_dof.data()[i * 3 + I];
      }
      // for smoothness indicator //
      double alpha_numerator_pos = 0., alpha_numerator_neg = 0., alpha_denominator_pos = 0., alpha_denominator_neg = 0.;
      for (int offset = csrRowIndeces_DofLoops.data()[i]; offset < csrRowIndeces_DofLoops.data()[i + 1]; offset++) { // First loop in j (sparsity pattern)
        int j = csrColumnOffsets_DofLoops.data()[offset];
        if (STABILIZATION_TYPE == STABILIZATION::EV_Stab) //EV Stabilization
        {
          // COMPUTE ETA MIN AND ETA MAX //
          etaMaxi = fmax(etaMaxi, fabs(eta[j]));
          etaMini = fmin(etaMini, fabs(eta[j]));
        }
        double solnj = u_free_dof_old[j];
        for (int I = 0; I < nSpace; I++) {
          solnj -= (rho_dof[j] / rho) * gravity.data()[I] * mesh_dof.data()[j * 3 + I];
        }
        // Update Cij matrices
        Cij[0] = Cx[ij];
#if nSpace == 2
        Cij[1] = Cy[ij];
#endif
#if nSpace == 3
        Cij[2] = Cz[ij];
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
        //update ij
        ij += 1;
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
        // compute xj
        double xj[nSpace];
        for (int I = 0; I < nSpace; I++) xj[I] = mesh_dof.data()[j * 3 + I];
        // compute gi*(xi-xj)
        double gi_times_x = 0.;
        for (int I = 0; I < nSpace; I++) {
          gi_times_x += gi[I] * delta_x_ij.data()[offset * 3 + I];
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
    /////////////////////////////////////////////
    // ** LOOP IN DOFs FOR EDGE BASED TERMS ** //
    /////////////////////////////////////////////
    ij = 0;
    for (int i = 0; i < numDOFs; i++) {
      int    ii;
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

      const double rho_i = rho_dof[i];
      // Rock region of each DOF.  The nodal m, mn and upwind k_rw below are
      // evaluated at a DOF, so they take that DOF's soil; a single
      // elementMaterialTypes[0] here would put the whole mesh on one retention
      // curve and, since invert() maps m back with the per-DOF soil, would make
      // psi -> m -> psi something other than the identity.
      const int mat_i = freeDOFMaterialTypes.data()[i];

      double thetaW_tmp = 0.0;
      // loop over the sparsity pattern of the i-th DOF
      for (int offset = csrRowIndeces_DofLoops.data()[i]; offset < csrRowIndeces_DofLoops.data()[i + 1]; offset++) {
        int j = csrColumnOffsets_DofLoops.data()[offset];
        if (i == j) ii = ij;
        const double ith_flux_term_before = ith_flux_term;
        const double rho_j = rho_dof[j];
        const int mat_j = freeDOFMaterialTypes.data()[j];
        const double rho_edge = 0.5 * (rho_i + rho_j);
        double delta_phi = u_free_dof[j] - u_free_dof[i];
        double delta_phin = u_free_dof_old[j] - u_free_dof_old[i];
        // Match evaluateCoefficients(): use an edge-based hydrostatic jump
        // with local density scaling rho/rho0, rather than grad(rho g·x),
        // which would introduce a spurious (g·x) grad(rho) term.
        for (int I = 0; I < nSpace; I++) {
          const double delta_x = mesh_dof.data()[j * 3 + I] - mesh_dof.data()[i * 3 + I];
          const double hydrostatic_jump = (rho_edge / rho) * gravity.data()[I] * delta_x;
          delta_phi -= hydrostatic_jump;
          delta_phin -= hydrostatic_jump;
        }
        double dLowij, dLij, dEVij, dHij, fH, fL, fA=0.0;
        double fL_CN =0.0, fA_CN=0.0 ; 
        // High-order (Galerkin) flux with its own time weight Theta_h (ThetaScheme.theta);
        // the low-order fL below keeps Theta, which LevelModel sets to 1 (backward Euler).
        fH = -Theta_h * TransportMatrixConsistent[ij] * delta_phi - (1 - Theta_h) * TransportMatrixConsistentn[ij] * delta_phin;
        ith_consistent_flux_term += fH;
        fA = fH;
        fA_CN = fH;

        if (-TransportMatrix[ij] * delta_phi <= 0.0) {
          evaluateCoefficients(a_rowptr.data(), a_colind.data(), rho, rho_i, beta, gravity.data(),
                               alpha.data()[mat_i],
                               n.data()[mat_i], thetaR.data()[mat_i], thetaSR.data()[mat_i], &KWs.data()[mat_i * nnz], u_free_dof[i], m, dm, f, df, a, da, as, Kr, dKr, thetaW_tmp);
          fL = Theta * Kr * fmax(0.0, -TransportMatrix[ij]) * delta_phi;
          fL_CN = Theta_h * Kr * fmax(0.0, -TransportMatrix[ij]) * delta_phi;

          if (i != j) {
            globalJacobian.data()[ij] -= Theta * Kr * fmax(0.0, -TransportMatrix[ij]);
            J_ii -= -Theta * Kr * fmax(0.0, -TransportMatrix[ij]) + dKr_on * Theta * dKr * fmax(0.0, -TransportMatrix[ij]) * delta_phi;
          }
          ith_flux_term += fL;
          fA -= fL;
        } else {
          evaluateCoefficients(a_rowptr.data(), a_colind.data(), rho, rho_j, beta, gravity.data(),
                               alpha.data()[mat_j],
                               n.data()[mat_j], thetaR.data()[mat_j], thetaSR.data()[mat_j], &KWs.data()[mat_j * nnz], u_free_dof[j], m, dm, f, df, a, da, as, Kr, dKr, thetaW_tmp);
          fL = Theta * Kr * fmax(0.0, -TransportMatrix[ij]) * delta_phi;
          fL_CN = Theta_h * Kr * fmax(0.0, -TransportMatrix[ij]) * delta_phi;

          if (i != j) {
            globalJacobian.data()[ij] -= Theta * Kr * fmax(0.0, -TransportMatrix[ij]) + dKr_on * Theta * dKr * fmax(0.0, -TransportMatrix[ij]) * delta_phi;
            J_ii -= -Theta * Kr * fmax(0.0, -TransportMatrix[ij]);
          }
          ith_flux_term += fL;
          fA -= fL;
        }
        // new-time low-order flux; on a boundary row it is replaced below by its
        // interior part (the weak-Dirichlet block is not an edge exchange)
        double fL_int = fL;
        if (-TransportMatrixn[ij] * delta_phin <= 0.0) {
          evaluateCoefficients(a_rowptr.data(), a_colind.data(), rho, rho_i, beta, gravity.data(),
                               alpha.data()[mat_i],
                               n.data()[mat_i], thetaR.data()[mat_i], thetaSR.data()[mat_i], &KWs.data()[mat_i * nnz], u_free_dof_old[i], m, dm, f, df, a, da, as, Kr, dKr, thetaW_tmp);
          fL = (1 - Theta) * Kr * fmax(0.0, -TransportMatrixn[ij]) * delta_phin;
          fL_CN += (1 - Theta_h) * Kr * fmax(0.0, -TransportMatrixn[ij]) * delta_phin;
          ith_flux_term += fL;
          fA -= fL;
          fA_CN -= fL_CN;
        } else {
          evaluateCoefficients(a_rowptr.data(), a_colind.data(), rho, rho_j, beta, gravity.data(),
                               alpha.data()[mat_j],
                               n.data()[mat_j], thetaR.data()[mat_j], thetaSR.data()[mat_j], &KWs.data()[mat_j * nnz], u_free_dof_old[j], m, dm, f, df, a, da, as, Kr, dKr, thetaW_tmp);
          fL = (1 - Theta) * Kr * fmax(0.0, -TransportMatrixn[ij]) * delta_phin;
          fL_CN += (1 - Theta_h) * Kr * fmax(0.0, -TransportMatrixn[ij]) * delta_phin;
          ith_flux_term += fL;
          fA -= fL;
          fA_CN -= fL_CN;
        }
        if (useBndDiff) {
          //Weak-Dirichlet block, kept out of the fmax() clip above and applied identically
          //in the low and the high order flux, so it never enters fA (see Step B above).
          const double TB_ij = TransportBoundary.data()[ij];
          const double f_b   = -Theta * TB_ij * delta_phi;
          ith_consistent_flux_term += f_b;
          ith_flux_term           += f_b;
          if (i != j) {
            //the low order boundary operator is what the residual and hence Newton sees
            globalJacobian.data()[ij] -= Theta * (-TB_ij);
            J_ii                      -= Theta * TB_ij;
          }
        }
        // CONSERVATIVE BOUNDARY FLUX (see the consBoundaryFlux note at the top):
        // the symmetric part of this row's edge flux, i.e. fL rebuilt on
        // T^int = T - TB with the same upwind rule.  T^int is symmetric and
        // dphi antisymmetric, so this flux telescopes exactly against the
        // interior and is the water node i really exchanges with the domain.
        // With useBndDiff on TransportMatrix is already pure interior.
        if (is_bdry_dof[i] && i != j) {
          const double Tint_ij = TransportMatrix[ij] - (useBndDiff ? 0.0 : TransportBoundary.data()[ij]);
          double Kr_c = 0.0, dKr_c = 0.0;
          if (-Tint_ij * delta_phi <= 0.0)
            evaluateCoefficients(a_rowptr.data(), a_colind.data(), rho, rho_i, beta, gravity.data(),
                                 alpha.data()[mat_i],
                                 n.data()[mat_i], thetaR.data()[mat_i], thetaSR.data()[mat_i], &KWs.data()[mat_i * nnz], u_free_dof[i], m, dm, f, df, a, da, as, Kr_c, dKr_c, thetaW_tmp);
          else
            evaluateCoefficients(a_rowptr.data(), a_colind.data(), rho, rho_j, beta, gravity.data(),
                                 alpha.data()[mat_j],
                                 n.data()[mat_j], thetaR.data()[mat_j], thetaSR.data()[mat_j], &KWs.data()[mat_j * nnz], u_free_dof[j], m, dm, f, df, a, da, as, Kr_c, dKr_c, thetaW_tmp);
          fL_int = Theta * Kr_c * fmax(0.0, -Tint_ij) * delta_phi;
          consBoundaryFlux.data()[i] += fL_int;
        }
        edgeFluxInterior.data()[ij] = fL_int;
        edgeFluxLow.data()[ij] = ith_flux_term - ith_flux_term_before;
        dt_times_fH_minus_fL.data()[ij] = dt * fA;
        //dt_times_fH_minus_fL.data()[ij] = dt * fA_CN;
        ij += 1;
      }
      // The nodal source is part of the low-order rate, so the consistent-mass FCT
      // term built from mDotLow sees it as well as the residual.  It is not an edge
      // exchange: it stays out of edgeFluxLow, and a coupled transport model finds
      // it in its nodal boundary flux B_i.
      ith_flux_term += MLi * nodalSource.data()[i];
      cflux[i] = ith_consistent_flux_term;
      evaluateCoefficients(a_rowptr.data(), a_colind.data(), rho, rho_i, beta, gravity.data(),
                           alpha.data()[mat_i],
                           n.data()[mat_i], thetaR.data()[mat_i], thetaSR.data()[mat_i], &KWs.data()[mat_i * nnz], u_free_dof[i], m, dm, f, df, a, da, as, Kr, dKr, thetaW_tmp);
      // Old-time mass with the OLD density: mn = theta(psi^n)*rho^{n-1}, the mass
      // the previous step produced.  With the current rho_i here, both levels of
      // the storage term shared one rho and theta*d(rho)/dt was dropped.
      const double rho_old_i = USE_RHO_NODAL_OLD ? rho_nodal_old.data()[i] : rho_i;
      evaluateCoefficients(a_rowptr.data(), a_colind.data(), rho, rho_old_i, beta, gravity.data(),
                           alpha.data()[mat_i],
                           n.data()[mat_i], thetaR.data()[mat_i], thetaSR.data()[mat_i], &KWs.data()[mat_i * nnz], u_free_dof_old[i], mn.data()[i], dmn, fn, dfn, an, dan, asn, Krn, dKrn, thetaW_tmp);
      // mDotLow enters the FCT mass antidiffusion MC_ij (mDotLow_i - mDotLow_j). At a strong-Dirichlet
      // DOF the row is masked, so the flux divergence is not dm/dt there; use the time difference of the
      // imposed data instead. (Without this the WRR FCT is only first order on Q1, Hornung-Messing.)
      mDotLow.data()[i] = (bc_mask.data()[i] == 0.0) ? (m - mn.data()[i]) / dt : ith_flux_term / MLi;
      mLow.data()[i] = m;
      // thetaW_tmp is whatever the LAST evaluateCoefficients left, i.e. the old-time
      // call immediately above -- so this is theta(psi^n)_i, the partner of mn_i.
      thetaW_dof.data()[i] = thetaW_tmp;
      globalResidual.data()[i] += bc_mask.data()[i] * (MLi * (m - mn.data()[i]) / dt - ith_flux_term);
      globalJacobian.data()[ii] += bc_mask.data()[i] * (MLi * ((LSCHEME_L > 0.0) ? LSCHEME_L : dm) / dt + J_ii) + (1.0 - bc_mask.data()[i]);
    }
    if (STABILIZATION_TYPE == STABILIZATION::Implicit_FCT) {
      //FCTStep(args);
      for (int i = 0; i < numDOFs; i++) {
        globalResidual.data()[i] += fluxCorrection.data()[i];
      }
    }
    ////////////////////////////////////////////////////////////////////////////
    // CONSERVATIVE BOUNDARY VELOCITY FOR THE COUPLED TRANSPORT MODEL
    //
    // The cell loop above hands the transport model a pointwise Darcy trace
    // -(a/rho)(grad psi - rho_r g); on the boundary that is NOT the water this
    // scheme moves, because the head is imposed weakly (penalty + IIPG rows in
    // TransportMatrix).  consBoundaryFlux, assembled in the DOF loop, IS that
    // water: F_i = sum_j Kr_up max(0,-T^int_ij) dphi_ij, the flux of the
    // symmetric interior operator out of node i, which telescopes exactly with
    // the interior edge fluxes and equals bflux_i minus the IIPG rows at
    // convergence.  Keep the tangential trace, replace the normal component:
    //
    //   v.n = (F_i - prescribed part) / (rho S_i)   weakly imposed (Dirichlet)
    //   v.n = bc_flux / rho                          prescribed flux faces
    //
    // interpolated back to the quadrature points with the trace basis, S_i
    // being the Dirichlet measure of node i.  Faces with neither flag (the
    // subdomain's artificial boundary with its neighbours) are left alone.
    // Nothing in the flow model reads this array, so the Richards solution and
    // its Jacobian are untouched -- this only changes what TADR is told.
    //
    // PARALLEL: F_i, S_i and the prescribed part are assembled from this rank's
    // elements/faces, so they are complete for OWNED rows only.  With the nodal
    // (vertex) boundary rule both models use -- checkCoupledQuadrature ties them
    // together -- a boundary quadrature point feeds only its own node's row, so
    // every value that reaches an owned transport row is complete.  A non-nodal
    // boundary rule would need the three arrays ghost-summed first.
    ////////////////////////////////////////////////////////////////////////////
    if (RICHARDS_BC_CONS) {
      for (int ebNE = 0; ebNE < nExteriorElementBoundaries_global; ebNE++) {
        const int ebN       = exteriorElementBoundariesArray.data()[ebNE],
                  eN        = elementBoundaryElementsArray.data()[ebN * 2 + 0],
                  ebN_local = elementBoundaryLocalElementBoundariesArray.data()[ebN * 2 + 0];
        for (int kb = 0; kb < nQuadraturePoints_elementBoundary; kb++) {
          const int ebNE_kb       = ebNE * nQuadraturePoints_elementBoundary + kb,
                    ebNE_kb_nSpace = ebNE_kb * nSpace,
                    ebN_local_kb  = ebN_local * nQuadraturePoints_elementBoundary + kb;
          const bool isDir = isDOFBoundary_u.data()[ebNE_kb] != 0;
          //A qp with neither flag is the subdomain's artificial boundary with a
          //neighbouring rank, not a physical one: leave it alone.  (It only ever
          //touches ghost nodes, whose transport rows are overwritten by the
          //scatter, but zeroing a velocity there would be a lie either way.)
          if (!isDir && !isFluxBoundary_u.data()[ebNE_kb] && !isSeepageFace.data()[ebNE]) continue;
          double vn_cons = 0.0;
          if (isDir) {
            double w_sum = 0.0;
            for (int i = 0; i < nDOF_test_element; i++) {
              const double w = u_test_trace_ref.data()[ebN_local_kb * nDOF_test_element + i];
              if (w <= 1.0e-12) continue;                       //DOF not on this face
              const int gi = r_l2g.data()[eN * nDOF_test_element + i];
              if (bdryDirichletArea[gi] <= 0.0) continue;
              vn_cons += w * (consBoundaryFlux.data()[gi] - bdryPrescribedFlux[gi]) / bdryDirichletArea[gi];
              w_sum   += w;
            }
            if (w_sum <= 1.0e-12) continue;
            vn_cons /= w_sum;                                   //= 1 with a full face star
          } else {
            //not weakly imposed: the residual integrated exactly ebqe_bflux here
            //(the prescribed flux, or 0 on a seepage face that shut), so that is
            //the water the transport model must see
            vn_cons = ebqe_bflux.data()[ebNE_kb];
          }
          //the residual's fluxes are rho-weighted; the coupled trace is a Darcy velocity
          const double rho_ext_c = ebqe_rho.data()[ebNE_kb];
          vn_cons /= (std::fabs(rho_ext_c) > 1.0e-12 ? rho_ext_c : rho);
          double vn_old = 0.0;
          for (int I = 0; I < nSpace; I++)
            vn_old += ebqe_velocity_ext_couple.data()[ebNE_kb_nSpace + I] * ebqe_normal_cache[ebNE_kb_nSpace + I];
          for (int I = 0; I < nSpace; I++)
            ebqe_velocity_ext_couple.data()[ebNE_kb_nSpace + I] += (vn_cons - vn_old) * ebqe_normal_cache[ebNE_kb_nSpace + I];
        } //kb
      } //ebNE
    }

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
    PSK_TYPE_member = args.scalar<int>("PSK_TYPE");
    VG_AIR_ENTRY_member = args.scalar<double>("VG_AIR_ENTRY");
    proteus::pyarray<double> &KWs                  = args.array<double>("KWs");
    proteus::pyarray<int>    &elementMaterialTypes = args.array<int>("elementMaterialTypes");
    proteus::pyarray<int>    &freeDOFMaterialTypes = args.array<int>("freeDOFMaterialTypes");
    int                  numDOFs              = args.scalar<int>("numDOFs");
    proteus::pyarray<double> &mIn  = args.array<double>("limited_solution");
    proteus::pyarray<double> &pOut = args.array<double>("u_dof");
    // Per-DOF density projected from q_rho in
    // calculateResidual_entropy_viscosity (cached as a class member there).
    // If the cache is empty (e.g. invert called before the first residual
    // evaluation, or coupling disabled) fall back to the scalar `rho`.
    const bool have_rho_dof = (rho_dof_member.size() == static_cast<std::size_t>(numDOFs));
    int USE_NEWTON_INVERT = args.scalar<int>("USE_NEWTON_INVERT");

    for (int i = 0; i < numDOFs; i++) {
      const int material_i = freeDOFMaterialTypes.data()[i];
      // Use the salinity-coupled density at this DOF so the m -> u inversion
      // is consistent with the variable-density forward residual
      //     m_forward = theta * rho_local * (...).
      const double rho_i = have_rho_dof ? rho_dof_member[i] : rho;
      double dm, f[nSpace], df[nSpace], a[nnz], da[nnz];
      if (USE_NEWTON_INVERT){
        evaluateInverseCoefficients_Newton(a_rowptr.data(), a_colind.data(), rho_i, beta, gravity.data(), alpha.data()[material_i], n.data()[material_i], thetaR.data()[material_i],
                                          thetaSR.data()[material_i], &KWs.data()[material_i * nnz],
                                          pOut.data()[i], mIn.data()[i],
                                          dm, f, df, a, da);
      }
        else{
        evaluateInverseCoefficients(a_rowptr.data(), a_colind.data(), rho_i, beta, gravity.data(), alpha.data()[material_i], n.data()[material_i], thetaR.data()[material_i],
                                          thetaSR.data()[material_i], &KWs.data()[material_i * nnz],
                                          pOut.data()[i], mIn.data()[i],
                                          dm, f, df, a, da);
        }
      }
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

    proteus::pyarray<double> &q_rho                = args.array<double>("q_rho");
    proteus::pyarray<double> &q_mu                 = args.array<double>("q_mu");

    proteus::pyarray<double> &gravity              = args.array<double>("gravity");
    proteus::pyarray<double> &alpha                = args.array<double>("alpha");
    proteus::pyarray<double> &n                    = args.array<double>("n");
    proteus::pyarray<double> &thetaR               = args.array<double>("thetaR");
    proteus::pyarray<double> &thetaSR              = args.array<double>("thetaSR");
    PSK_TYPE_member = args.scalar<int>("PSK_TYPE");
    VG_AIR_ENTRY_member = args.scalar<double>("VG_AIR_ENTRY");
    proteus::pyarray<double> &KWs                  = args.array<double>("KWs");
    //end new
    double               useMetrics                                 = args.scalar<double>("useMetrics");
    double               alphaBDF                                   = args.scalar<double>("alphaBDF");
    int                  lag_shockCapturing                         = args.scalar<int>("lag_shockCapturing");
    double               shockCapturingDiffusion                    = args.scalar<double>("shockCapturingDiffusion");
    proteus::pyarray<int>    &u_l2g                                      = args.array<int>("u_l2g");
    proteus::pyarray<int>    &r_l2g                                      = args.array<int>("r_l2g");
    proteus::pyarray<double> &elementDiameter                            = args.array<double>("elementDiameter");
    int                  degree_polynomial                          = args.scalar<int>("degree_polynomial");
    proteus::pyarray<double> &u_dof                                      = args.array<double>("u_dof");
    proteus::pyarray<double> &velocity                                   = args.array<double>("velocity");
    proteus::pyarray<double> &q_m_betaBDF                                = args.array<double>("q_m_betaBDF");
    proteus::pyarray<double> &cfl                                        = args.array<double>("cfl");
    proteus::pyarray<double> &q_numDiff_u_last                           = args.array<double>("q_numDiff_u_last");
    proteus::pyarray<int>    &csrRowIndeces_u_u                          = args.array<int>("csrRowIndeces_u_u");
    proteus::pyarray<int>    &csrColumnOffsets_u_u                       = args.array<int>("csrColumnOffsets_u_u");
    proteus::pyarray<double> &globalJacobian                             = args.array<double>("globalJacobian");
    proteus::pyarray<double> &delta_x_ij                                 = args.array<double>("delta_x_ij");
    int                  nExteriorElementBoundaries_global          = args.scalar<int>("nExteriorElementBoundaries_global");
    proteus::pyarray<int>    &exteriorElementBoundariesArray             = args.array<int>("exteriorElementBoundariesArray");
    proteus::pyarray<int>    &elementBoundaryElementsArray               = args.array<int>("elementBoundaryElementsArray");
    proteus::pyarray<int>    &elementBoundaryLocalElementBoundariesArray = args.array<int>("elementBoundaryLocalElementBoundariesArray");
    proteus::pyarray<double> &ebqe_velocity_ext                          = args.array<double>("ebqe_velocity_ext");
    proteus::pyarray<int>    &isDOFBoundary_u                            = args.array<int>("isDOFBoundary_u");
    proteus::pyarray<double> &ebqe_bc_u_ext                              = args.array<double>("ebqe_bc_u_ext");
    proteus::pyarray<int>    &isFluxBoundary_u                           = args.array<int>("isFluxBoundary_u");
    proteus::pyarray<double> &ebqe_bc_flux_ext                           = args.array<double>("ebqe_bc_flux_ext");
    proteus::pyarray<int>    &csrColumnOffsets_eb_u_u                    = args.array<int>("csrColumnOffsets_eb_u_u");
    int                  LUMPED_MASS_MATRIX                         = args.scalar<int>("LUMPED_MASS_MATRIX");
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
        double KWs_mu[nnz];
        for (int ii = 0; ii < nnz; ii++) KWs_mu[ii] = KWs.data()[elementMaterialTypes.data()[eN] * nnz + ii] / q_mu.data()[eN_k];
        evaluateCoefficients(a_rowptr.data(), a_colind.data(), rho, q_rho.data()[eN_k], beta, gravity.data(), alpha.data()[elementMaterialTypes.data()[eN]], n.data()[elementMaterialTypes.data()[eN]], thetaR.data()[elementMaterialTypes.data()[eN]],
                             thetaSR.data()[elementMaterialTypes.data()[eN]], KWs_mu, u, m, dm, f, df, a, da, as, Kr, dKr, thetaW);
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
  } //computeMassMatrix
}; //Richards

inline Richards_base *newRichards(int nSpaceIn, int nQuadraturePoints_elementIn, int nDOF_mesh_trial_elementIn, int nDOF_trial_elementIn, int nDOF_test_elementIn, int nQuadraturePoints_elementBoundaryIn, int CompKernelFlag)
{
  if (nSpaceIn == 1)
    return proteus::chooseAndAllocateDiscretization1D<Richards_base, Richards, CompKernel>(nSpaceIn, nQuadraturePoints_elementIn, nDOF_mesh_trial_elementIn, nDOF_trial_elementIn, nDOF_test_elementIn, nQuadraturePoints_elementBoundaryIn, CompKernelFlag);
  else if (nSpaceIn == 2)
    return proteus::chooseAndAllocateDiscretization2D<Richards_base, Richards, CompKernel>(nSpaceIn, nQuadraturePoints_elementIn, nDOF_mesh_trial_elementIn, nDOF_trial_elementIn, nDOF_test_elementIn, nQuadraturePoints_elementBoundaryIn, CompKernelFlag);
  else {
    assert(nSpaceIn == 3);
    return proteus::chooseAndAllocateDiscretization<Richards_base, Richards, CompKernel>(nSpaceIn, nQuadraturePoints_elementIn, nDOF_mesh_trial_elementIn, nDOF_trial_elementIn, nDOF_test_elementIn, nQuadraturePoints_elementBoundaryIn, CompKernelFlag);
  }
}
} // namespace richards
} // namespace proteus
#endif
