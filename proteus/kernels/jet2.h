#ifndef JET2_H
#define JET2_H
// Second-order forward-mode AD in two independent variables (p, z).  A Jet2
// carries (v, dp, dz, dpp, dpz, dzz) -- value plus exact first and second
// partials -- so threading it through the EOS/flash arithmetic yields
// machine-precision analytic derivatives with NO finite differencing.
//
// References:
//   Griewank, A. and Walther, A., "Evaluating Derivatives: Principles and
//     Techniques of Algorithmic Differentiation", 2nd ed., SIAM, Philadelphia,
//     2008 -- forward mode and the propagation of truncated Taylor (jet)
//     coefficients that this struct implements for order 2 in two variables.
//   Fike, J.A. and Alonso, J.J., "The Development of Hyper-Dual Numbers for
//     Exact Second-Derivative Calculations", AIAA Paper 2011-886, 49th AIAA
//     Aerospace Sciences Meeting, Orlando, 2011 -- the same second-order
//     forward-mode idea, specialised to one independent variable.
#include <cmath>

namespace m_comp_co2 {

struct Jet2 {
    double v, dp, dz, dpp, dpz, dzz;
    Jet2(double v_=0, double dp_=0, double dz_=0,
         double dpp_=0, double dpz_=0, double dzz_=0)
        : v(v_), dp(dp_), dz(dz_), dpp(dpp_), dpz(dpz_), dzz(dzz_) {}
    static Jet2 cst(double c)   { return Jet2(c); }
    static Jet2 var_p(double p) { return Jet2(p, 1.0, 0.0); }
    static Jet2 var_z(double z) { return Jet2(z, 0.0, 1.0); }
};

inline Jet2 operator+(const Jet2& a, const Jet2& b) {
    return Jet2(a.v+b.v, a.dp+b.dp, a.dz+b.dz, a.dpp+b.dpp, a.dpz+b.dpz, a.dzz+b.dzz);
}
inline Jet2 operator+(const Jet2& a, double c) { return Jet2(a.v+c, a.dp, a.dz, a.dpp, a.dpz, a.dzz); }
inline Jet2 operator+(double c, const Jet2& a) { return a + c; }
inline Jet2 operator-(const Jet2& a) { return Jet2(-a.v, -a.dp, -a.dz, -a.dpp, -a.dpz, -a.dzz); }
inline Jet2 operator-(const Jet2& a, const Jet2& b) { return a + (-b); }
inline Jet2 operator-(const Jet2& a, double c) { return Jet2(a.v-c, a.dp, a.dz, a.dpp, a.dpz, a.dzz); }
inline Jet2 operator-(double c, const Jet2& a) { return Jet2(c-a.v, -a.dp, -a.dz, -a.dpp, -a.dpz, -a.dzz); }

inline Jet2 operator*(const Jet2& a, const Jet2& b) {
    return Jet2(a.v*b.v,
                a.dp*b.v + a.v*b.dp,
                a.dz*b.v + a.v*b.dz,
                a.dpp*b.v + 2.0*a.dp*b.dp + a.v*b.dpp,
                a.dpz*b.v + a.dp*b.dz + a.dz*b.dp + a.v*b.dpz,
                a.dzz*b.v + 2.0*a.dz*b.dz + a.v*b.dzz);
}
inline Jet2 operator*(const Jet2& a, double c) { return Jet2(a.v*c, a.dp*c, a.dz*c, a.dpp*c, a.dpz*c, a.dzz*c); }
inline Jet2 operator*(double c, const Jet2& a) { return a*c; }

inline Jet2 recip(const Jet2& s) {
    double b = s.v, r = 1.0/b, b2 = b*b, b3 = b2*b;
    return Jet2(r, -s.dp/b2, -s.dz/b2,
                (2.0*s.dp*s.dp - b*s.dpp)/b3,
                (2.0*s.dp*s.dz - b*s.dpz)/b3,
                (2.0*s.dz*s.dz - b*s.dzz)/b3);
}
inline Jet2 operator/(const Jet2& a, const Jet2& b) { return a*recip(b); }
inline Jet2 operator/(const Jet2& a, double c) { return Jet2(a.v/c, a.dp/c, a.dz/c, a.dpp/c, a.dpz/c, a.dzz/c); }
inline Jet2 operator/(double c, const Jet2& a) { return Jet2::cst(c)*recip(a); }

inline Jet2 jsqrt(const Jet2& s) {
    double a = s.v, rt = std::sqrt(a), t2 = 2.0*rt, t3 = 4.0*rt*rt*rt;
    return Jet2(rt, s.dp/t2, s.dz/t2,
                s.dpp/t2 - s.dp*s.dp/t3,
                s.dpz/t2 - s.dp*s.dz/t3,
                s.dzz/t2 - s.dz*s.dz/t3);
}
inline Jet2 jexp(const Jet2& s) {
    double e = std::exp(s.v);
    return Jet2(e, e*s.dp, e*s.dz,
                e*(s.dpp + s.dp*s.dp),
                e*(s.dpz + s.dp*s.dz),
                e*(s.dzz + s.dz*s.dz));
}
inline Jet2 jlog(const Jet2& s) {
    double a = s.v, a2 = a*a;
    return Jet2(std::log(a), s.dp/a, s.dz/a,
                s.dpp/a - s.dp*s.dp/a2,
                s.dpz/a - s.dp*s.dz/a2,
                s.dzz/a - s.dz*s.dz/a2);
}

}  // namespace m_comp_co2
#endif
