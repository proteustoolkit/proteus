#ifndef CO2_BRINE_FLASH_H
#define CO2_BRINE_FLASH_H
// Analytic (p,z) flash for the 2-component CO2-brine model -- C++ port of
// co2_brine_flash.py.  Header-only; the kernel includes this.
//
//   flashPZ(p_Pa, z, T_C, m_NaCl, eps) -> FlashState
//
// All first AND second derivatives are exact, computed by 2nd-order forward-mode
// AD (Jet2) -- NO finite differencing.  The composition clamps are hyperbolically
// smoothed so S_g is C1 across phase appearance/disappearance and the capillary
// value-block Jacobian (which needs d2S_g) is fully consistent.
#include "co2_brine_eos.h"
#include "jet2.h"
#include <cmath>
#include <algorithm>

namespace m_comp_co2 {
namespace flash {

constexpr double PA_PER_BAR = 1.0e5;
// composition-clamp smoothing band [mole fraction].  See co2_brine_flash.py for
// the full rationale: smooth_min undershoots X below z, and the lever rule
// amplifies that by rho_a/rho_g (~1400x) into S_g.  At SI scale (Xeq~6e-4) the old
// 5e-4 leaked S_g~0.10 at z=1e-5; 1e-5 keeps spurious S_g ~5e-5 while staying C1.
constexpr double EPS_Z      = 1.0e-5;   // composition-clamp smoothing band
// z is evaluated inside [Z_MIN, 1 - Z_MIN] (CO2-free brine starts at z = 0 and
// Newton iterates can leave [0, 1]).  Outside it the flash returns the bound's
// state with ZERO z-derivatives: the result does not change with z there, so a
// nonzero dS_g/dz, dX/dz, ... would put a derivative in the Jacobian that the
// residual does not have (FD probe of the 1D prototype: 515 vs 0 at CO2-free
// nodes, test/m_comp_co2/verification/rock_interface_1d).  Callers pass the raw z.
constexpr double Z_MIN      = 1.0e-8;

// Immiscible / incompressible limit (verification only) -- must match
// co2_brine_flash.py.  When enabled (flashPZ immiscible=true, set from the
// Coefficients 'immiscible' option) mutual solubility is suppressed (Xeq=0,
// Yeq=1) and both phase densities are pinned to these constants, so the species
// decouple into the classical immiscible two-phase saturation equations.  The
// McWhorter-Sunada S_n similarity profile is invariant to the density values
// (they divide out of each saturation equation); any IC/BC converting S_n <-> z
// MUST use the SAME two constants.
constexpr double RHO_A_IMM = 5.55e4;    // constant aqueous molar density [mol/m^3]
constexpr double RHO_G_IMM = 2.00e4;    // constant gas     molar density [mol/m^3]

struct FlashState {
    double S_g, X, Y, rho_a, rho_g, f_g;
    double dS_g_dp, dS_g_dz, dX_dp, dX_dz, dY_dp, dY_dz;
    double drho_a_dp, drho_a_dz, drho_g_dp, drho_g_dz;
    double d2S_g_dp2, d2S_g_dpdz, d2S_g_dz2;     // flash Hessian (capillary Jac)
    int phase;   // 0 aqueous, 1 two-phase, 2 gas
};

// CO2-rich molar volume as a 2-jet: value from the float cubic solver, then
// Newton-refined in jet space against the RK residual so (p,z) derivs are exact.
inline Jet2 _V_jet(double T_C, const Jet2& P_jet, bool liquid) {
    using namespace m_comp_co2::eos;
    double T_K = T_C + 273.15;
    double a = A_CO2_0 - A_CO2_1*T_K, b = B_CO2;
    double RTK = R_CM3_BAR*T_K, T05 = std::sqrt(T_K);
    Jet2 V = Jet2::cst(rk_molar_volume(T_C, P_jet.v, liquid));
    for (int it = 0; it < 4; ++it) {
        Jet2 VVb = V*(V + b);
        Jet2 F = P_jet - RTK*recip(V - b) + (a/T05)*recip(VVb);
        Jet2 dFdV = RTK*recip((V - b)*(V - b))
                  - (a/T05)*(2.0*V + b)*recip(VVb*VVb);
        V = V - F*recip(dFdV);
    }
    return V;
}

inline Jet2 _ln_phi_jet(double T_C, const Jet2& P_jet, const Jet2& V,
                        double b_k, double sum_aik) {
    using namespace m_comp_co2::eos;
    double T_K = T_C + 273.15;
    double a = A_CO2_0 - A_CO2_1*T_K, b = B_CO2;
    double RTK = R_CM3_BAR*T_K, RT15 = R_CM3_BAR*std::pow(T_K, 1.5);
    Jet2 lnr = jlog((V + b)/V);
    return jlog(V/(V - b))
         + b_k*recip(V - b)
         - (2.0*sum_aik/(RT15*b))*lnr
         + (a*b_k/(RT15*b*b))*(lnr - b*recip(V + b))
         - jlog(P_jet*V/RTK);
}

inline void _solubility_jet(double T_C, const Jet2& P_jet, bool liquid,
                            Jet2& x_CO2, Jet2& y_H2O, Jet2& V) {
    using namespace m_comp_co2::eos;
    double T_K = T_C + 273.15, RTK = R_CM3_BAR*T_K;
    V = _V_jet(T_C, P_jet, liquid);
    Jet2 phi_CO2 = jexp(_ln_phi_jet(T_C, P_jet, V, B_CO2, A_CO2_0 - A_CO2_1*T_K));
    Jet2 phi_H2O = jexp(_ln_phi_jet(T_C, P_jet, V, B_H2O, A_H2O_CO2));
    double K0H = std::pow(10.0, poly_logK0(K0_H2O, T_C));
    double K0C = std::pow(10.0, poly_logK0(liquid ? K0_CO2_L : K0_CO2_G, T_C));
    Jet2 dP = P_jet - P0_BAR;
    Jet2 A = (K0H*recip(phi_H2O*P_jet))*jexp(dP*(V_H2O/RTK));
    Jet2 B = (phi_CO2*P_jet*(1.0/(M_PER_KG_H2O*K0C)))*jexp(dP*(-V_CO2/RTK));
    y_H2O = (1.0 - B)*recip(recip(A) - B);
    x_CO2 = B*(1.0 - y_H2O);
}

inline FlashState flashPZ(double p_Pa, double z, double T_C,
                          double /*m_NaCl*/ = 0.0, double eps = EPS_Z,
                          bool immiscible = false) {
    using namespace m_comp_co2::eos;
    // Immiscible/incompressible verification limit (Xeq=0, Yeq=1, const
    // densities); driven by the Coefficients 'immiscible' option via argsDict.
    Jet2 p_jet = Jet2::var_p(p_Pa);
    const double z_in = std::min(std::max(z, Z_MIN), 1.0 - Z_MIN);
    Jet2 z_jet = (z_in == z) ? Jet2::var_z(z) : Jet2::cst(z_in);

    Jet2 X, Y, rho_a, rho_g;
    if (immiscible) {
        // No mutual solubility (Xeq=0, Yeq=1), constant densities: f_g = z and
        // the lever rule maps z -> S_g with all p-derivatives exactly zero, so
        // the species balances reduce to the two saturation equations.
        X = Jet2::cst(0.0);
        Y = Jet2::cst(1.0);
        rho_a = Jet2::cst(RHO_A_IMM);
        rho_g = Jet2::cst(RHO_G_IMM);
    } else {
        Jet2 pb_jet = p_jet*(1.0/PA_PER_BAR);
        bool liquid = co2_is_liquid(T_C, pb_jet.v);

        Jet2 Xeq, yH2O, V_jet;
        _solubility_jet(T_C, pb_jet, liquid, Xeq, yH2O, V_jet);
        Jet2 Yeq = 1.0 - yH2O;
        rho_g = 1.0e6*recip(V_jet);

        Jet2 sx = jsqrt((z_jet - Xeq)*(z_jet - Xeq) + eps*eps);
        X = 0.5*(z_jet + Xeq - sx);
        Jet2 sy = jsqrt((z_jet - Yeq)*(z_jet - Yeq) + eps*eps);
        Y = 0.5*(z_jet + Yeq + sy);

        double rho_w = pure_water_density_gcc(T_C);
        double Vphi = 37.51 - 9.585e-2*T_C + 8.740e-4*T_C*T_C - 5.044e-7*T_C*T_C*T_C;
        double c0 = 18.015/rho_w;
        rho_a = (1.0e6*recip(c0 + (Vphi - c0)*X))
              * jexp((p_jet - P_REF_BRINE)*C_F_BRINE);   // brine compressibility
    }

    Jet2 f_g = (z_jet - X)*recip(Y - X);
    Jet2 G = f_g*recip(rho_g);
    Jet2 L = (1.0 - f_g)*recip(rho_a);
    Jet2 S_g = G*recip(G + L);

    FlashState s;
    s.S_g = S_g.v; s.X = X.v; s.Y = Y.v; s.rho_a = rho_a.v; s.rho_g = rho_g.v;
    s.f_g = f_g.v;
    s.dS_g_dp = S_g.dp; s.dS_g_dz = S_g.dz;
    s.dX_dp = X.dp; s.dX_dz = X.dz; s.dY_dp = Y.dp; s.dY_dz = Y.dz;
    s.drho_a_dp = rho_a.dp; s.drho_a_dz = rho_a.dz;
    s.drho_g_dp = rho_g.dp; s.drho_g_dz = rho_g.dz;
    s.d2S_g_dp2 = S_g.dpp; s.d2S_g_dpdz = S_g.dpz; s.d2S_g_dz2 = S_g.dzz;
    s.phase = (f_g.v < 1.0e-6) ? 0 : (f_g.v > 1.0 - 1.0e-6 ? 2 : 1);
    return s;
}

}  // namespace flash
}  // namespace m_comp_co2
#endif
