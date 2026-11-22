#ifndef CO2_BRINE_EOS_H
#define CO2_BRINE_EOS_H
// Spycher-Pruess-Ennis-King (2003) CO2-brine equilibrium EOS  -- C++ port of
// co2_brine_eos.py (P2).  Header-only, no proteus deps so it is
// standalone-testable.  All constants/equations match the numpy reference.
//
//   Spycher, Pruess & Ennis-King (2003), GCA 67(16) 3015-3031 (LBNL-50991)
//
// Salt-free (m_NaCl wired but the SP2005 salting-out is a follow-up).
#include <cmath>
#include <algorithm>

namespace m_comp_co2 {
namespace eos {

// --- SP2003 constants (verbatim, == co2_brine_eos.py) ----------------------
constexpr double R_CM3_BAR    = 83.1446;   // cm^3 bar /(mol K)
constexpr double P0_BAR       = 1.0;
constexpr double M_PER_KG_H2O = 55.508;

constexpr double A_CO2_0   = 7.54e7;       // a_CO2(T) = A_CO2_0 - A_CO2_1*T(K)
constexpr double A_CO2_1   = 4.02e4;
constexpr double A_H2O_CO2 = 7.89e7;
constexpr double B_CO2     = 27.80;
constexpr double B_H2O     = 18.10;

constexpr double V_H2O = 18.5, V_CO2 = 32.1;   // partial molar volumes cm^3/mol

// Brine compressibility: rho_a(p) = rho_a(T,X)*exp(C_F_BRINE*(p - P_REF_BRINE)),
// p in Pa.  Gives the aqueous phase a pressure dependence so the comp-0 pressure
// equation is non-degenerate in single-phase brine.
constexpr double C_F_BRINE   = 4.5e-10;        // isothermal compressibility [1/Pa]
constexpr double P_REF_BRINE = 1.01325e5;      // reference pressure [Pa]

constexpr double M_CO2_KG = 0.04401;           // molar mass [kg/mol]
constexpr double M_H2O_KG = 0.018015;          // molar mass [kg/mol]

// log10(K0) = a + bT + cT^2 + dT^3 + eT^4 , T in CELSIUS
constexpr double K0_H2O[5]   = {-2.215, 3.162e-2, -1.294e-4, 4.187e-7, -7.331e-10};
constexpr double K0_CO2_G[5] = { 1.188, 1.307e-2, -5.445e-5, 0.0, 0.0};
constexpr double K0_CO2_L[5] = { 1.168, 1.361e-2, -5.135e-5, 0.0, 0.0};

constexpr double TC_CO2_K = 304.1282, PC_CO2_BAR = 73.773;
constexpr double SW_A[4]  = {-7.0602087, 1.9391218, -1.6463597, -3.2995634};

inline double poly_logK0(const double c[5], double T_C) {
    return c[0] + T_C*(c[1] + T_C*(c[2] + T_C*(c[3] + T_C*c[4])));
}

inline double co2_saturation_pressure_bar(double T_C) {
    double T_K = T_C + 273.15;
    if (T_K >= TC_CO2_K) return INFINITY;
    double tau = 1.0 - T_K/TC_CO2_K;
    double s = SW_A[0]*tau + SW_A[1]*std::pow(tau,1.5)
             + SW_A[2]*tau*tau + SW_A[3]*std::pow(tau,4.0);
    return PC_CO2_BAR*std::exp((TC_CO2_K/T_K)*s);
}

inline bool co2_is_liquid(double T_C, double P_bar) {
    if (T_C + 273.15 >= TC_CO2_K) return false;
    return P_bar >= co2_saturation_pressure_bar(T_C);
}

// real roots of  x^3 + b x^2 + c x + d = 0 ; returns count, fills r[3]
inline int solve_cubic_real(double b, double c, double d, double r[3]) {
    double p = c - b*b/3.0;
    double q = 2.0*b*b*b/27.0 - b*c/3.0 + d;
    double shift = -b/3.0;
    double disc = q*q/4.0 + p*p*p/27.0;
    if (disc > 1e-14) {                                   // one real root
        double s = std::sqrt(disc);
        r[0] = std::cbrt(-q/2.0 + s) + std::cbrt(-q/2.0 - s) + shift;
        return 1;
    } else if (disc < -1e-14) {                           // three real roots
        double m = 2.0*std::sqrt(-p/3.0);
        double theta = std::acos(std::max(-1.0, std::min(1.0,
                         3.0*q/(p*m))))/3.0;
        for (int k = 0; k < 3; ++k)
            r[k] = m*std::cos(theta - 2.0*M_PI*k/3.0) + shift;
        return 3;
    } else {                                              // (near) repeated
        double u = std::cbrt(-q/2.0);
        r[0] = 2.0*u + shift;
        r[1] = -u + shift;
        return 2;
    }
}

// molar volume [cm^3/mol] of the CO2-rich phase (RK, infinite-dilution a,b=CO2)
inline double rk_molar_volume(double T_C, double P_bar, bool liquid) {
    double T_K = T_C + 273.15;
    double a = A_CO2_0 - A_CO2_1*T_K;
    double b = B_CO2;
    double A = a*P_bar/(R_CM3_BAR*R_CM3_BAR*std::pow(T_K,2.5));
    double B = b*P_bar/(R_CM3_BAR*T_K);
    double roots[3];
    int n = solve_cubic_real(-1.0, (A - B - B*B), -A*B, roots);
    double Zsel = liquid ? INFINITY : -INFINITY;
    bool found = false;
    for (int i = 0; i < n; ++i) {
        if (roots[i] <= B) continue;                      // physical: Z > B
        if (liquid) { if (roots[i] < Zsel) { Zsel = roots[i]; found = true; } }
        else        { if (roots[i] > Zsel) { Zsel = roots[i]; found = true; } }
    }
    if (!found) {                                         // fall back: any root
        for (int i = 0; i < n; ++i)
            if (liquid ? roots[i] < Zsel : roots[i] > Zsel) Zsel = roots[i];
    }
    return Zsel*R_CM3_BAR*T_K/P_bar;
}

inline double ln_phi(double T_C, double P_bar, double V,
                     double b_k, double sum_aik) {
    double T_K = T_C + 273.15;
    double a = A_CO2_0 - A_CO2_1*T_K;
    double b = B_CO2;
    double RT15 = R_CM3_BAR*std::pow(T_K,1.5);
    double ln_ratio = std::log((V + b)/V);
    return std::log(V/(V - b))
         + b_k/(V - b)
         - (2.0*sum_aik/(RT15*b))*ln_ratio
         + (a*b_k/(RT15*b*b))*(ln_ratio - b/(V + b))
         - std::log(P_bar*V/(R_CM3_BAR*T_K));
}

// mutual solubilities at (T_C [C], P_bar [bar]) -> x_CO2 (aqueous), y_H2O (gas)
inline void spycher_pruess_solubility(double T_C, double P_bar, double /*m_NaCl*/,
                                      double& x_CO2, double& y_H2O) {
    bool liquid = co2_is_liquid(T_C, P_bar);
    double T_K = T_C + 273.15;
    double V = rk_molar_volume(T_C, P_bar, liquid);
    double phi_CO2 = std::exp(ln_phi(T_C, P_bar, V, B_CO2, A_CO2_0 - A_CO2_1*T_K));
    double phi_H2O = std::exp(ln_phi(T_C, P_bar, V, B_H2O, A_H2O_CO2));
    double K0H = std::pow(10.0, poly_logK0(K0_H2O, T_C));
    double K0C = std::pow(10.0, poly_logK0(liquid ? K0_CO2_L : K0_CO2_G, T_C));
    double dP = P_bar - P0_BAR;
    double A = (K0H/(phi_H2O*P_bar))*std::exp(dP*V_H2O/(R_CM3_BAR*T_K));
    double B = (phi_CO2*P_bar/(M_PER_KG_H2O*K0C))*std::exp(-dP*V_CO2/(R_CM3_BAR*T_K));
    y_H2O = (1.0 - B)/(1.0/A - B);
    x_CO2 = B*(1.0 - y_H2O);
}

inline double co2_rich_molar_density(double T_C, double P_bar) {
    return 1.0e6/rk_molar_volume(T_C, P_bar, co2_is_liquid(T_C, P_bar));
}

inline double pure_water_density_gcc(double T_C) {   // Kell (1975), g/cm^3
    double num = 999.83952 + 16.945176*T_C - 7.9870401e-3*T_C*T_C
               - 46.170461e-6*T_C*T_C*T_C + 105.56302e-9*std::pow(T_C,4)
               - 280.54253e-12*std::pow(T_C,5);
    return num/(1.0 + 16.879850e-3*T_C)/1000.0;
}

inline double brine_molar_density(double T_C, double P_bar, double x_CO2) {
    double Vphi = 37.51 - 9.585e-2*T_C + 8.740e-4*T_C*T_C - 5.044e-7*T_C*T_C*T_C;
    double rho_w = pure_water_density_gcc(T_C);
    double n_H2O = M_PER_KG_H2O;
    double n_CO2 = x_CO2/(1.0 - x_CO2)*n_H2O;
    double vol_cm3 = (n_H2O*18.015)/rho_w + n_CO2*Vphi;
    double rho0 = ((n_H2O + n_CO2)/vol_cm3)*1.0e6;
    return rho0*std::exp(C_F_BRINE*(P_bar*1.0e5 - P_REF_BRINE));
}

}  // namespace eos
}  // namespace m_comp_co2
#endif
