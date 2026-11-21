// Forward-mode dual numbers for generated coefficient functors (C19 slices round).
//
// ymf's ADR form gives a coefficient's first derivative with respect to each unknown as code (dm, df, dr, dH). Two derivatives it
// does not give, and a generated functor needs, are taken here by the functor itself, from the same expressions, exactly (no
// finite differences): d(dH_I)/d(grad u_K) (the slot hasDDH of vof.h: RDLS's S |grad u| has a Hamiltonian nonlinear in grad u),
// and the Jacobian of a problem with a lagged coefficient (the coefficient's dependence on u is dropped by evaluating it as a
// plain double). The generator emits every function through `fn::`, which is std for double and the chain rule for Dual, so
// one expression text serves both and the double instantiation is the std call (bitwise what it was).
#pragma once
#include <cmath>
#include "portable.h"

namespace adrh
{
  template <int N>
  struct Dual
  {
    double v;
    double d[N];
    ADRH_FN ADRH_FORCE_INLINE Dual() : v(0.0) { for (int i = 0; i < N; ++i) d[i] = 0.0; }
    ADRH_FN ADRH_FORCE_INLINE Dual(double c) : v(c) { for (int i = 0; i < N; ++i) d[i] = 0.0; }   // a constant
    ADRH_FN ADRH_FORCE_INLINE static Dual seed(double value, int i) { Dual r(value); r.d[i] = 1.0; return r; }
  };

#define ADRH_DUAL_BIN(op, vexpr, dexpr)                                                                          \
  template <int N> ADRH_FN ADRH_FORCE_INLINE Dual<N> operator op(const Dual<N> &a, const Dual<N> &b)            \
  { Dual<N> r; r.v = vexpr; for (int i = 0; i < N; ++i) r.d[i] = dexpr; return r; }                              \
  template <int N> ADRH_FN ADRH_FORCE_INLINE Dual<N> operator op(const Dual<N> &a, double bc)                   \
  { return a op Dual<N>(bc); }                                                                                   \
  template <int N> ADRH_FN ADRH_FORCE_INLINE Dual<N> operator op(double ac, const Dual<N> &b)                   \
  { return Dual<N>(ac) op b; }
  ADRH_DUAL_BIN(+, a.v + b.v, a.d[i] + b.d[i])
  ADRH_DUAL_BIN(-, a.v - b.v, a.d[i] - b.d[i])
  ADRH_DUAL_BIN(*, a.v * b.v, a.d[i] * b.v + a.v * b.d[i])
  ADRH_DUAL_BIN(/, a.v / b.v, (a.d[i] * b.v - a.v * b.d[i]) / (b.v * b.v))
#undef ADRH_DUAL_BIN

  template <int N> ADRH_FN ADRH_FORCE_INLINE Dual<N> operator-(const Dual<N> &a)
  { Dual<N> r; r.v = -a.v; for (int i = 0; i < N; ++i) r.d[i] = -a.d[i]; return r; }

#define ADRH_DUAL_CMP(op)                                                                                             \
  template <int N> ADRH_FN ADRH_FORCE_INLINE bool operator op(const Dual<N> &a, const Dual<N> &b) { return a.v op b.v; } \
  template <int N> ADRH_FN ADRH_FORCE_INLINE bool operator op(const Dual<N> &a, double b) { return a.v op b; }         \
  template <int N> ADRH_FN ADRH_FORCE_INLINE bool operator op(double a, const Dual<N> &b) { return a op b.v; }
  ADRH_DUAL_CMP(<)
  ADRH_DUAL_CMP(>)
  ADRH_DUAL_CMP(<=)
  ADRH_DUAL_CMP(>=)
#undef ADRH_DUAL_CMP

  // The functions a generated expression may use: std for double, the chain rule for Dual.
  namespace fn
  {
#define ADRH_FN1(name, stdname, dfun)                                                                                  \
    ADRH_FN ADRH_FORCE_INLINE double name(double a) { return std::stdname(a); }                                        \
    template <int N> ADRH_FN ADRH_FORCE_INLINE Dual<N> name(const Dual<N> &a)                                          \
    { Dual<N> r; r.v = std::stdname(a.v); const double s = dfun; for (int i = 0; i < N; ++i) r.d[i] = s * a.d[i]; return r; }
    ADRH_FN1(sqrt, sqrt, 0.5 / r.v)
    ADRH_FN1(sin, sin, std::cos(a.v))
    ADRH_FN1(cos, cos, -std::sin(a.v))
    ADRH_FN1(tan, tan, 1.0 + r.v * r.v)
    ADRH_FN1(exp, exp, r.v)
    ADRH_FN1(log, log, 1.0 / a.v)
    ADRH_FN1(fabs, fabs, (a.v < 0.0 ? -1.0 : 1.0))
    ADRH_FN1(tanh, tanh, 1.0 - r.v * r.v)
    ADRH_FN1(sinh, sinh, std::cosh(a.v))
    ADRH_FN1(cosh, cosh, std::sinh(a.v))
#undef ADRH_FN1
    ADRH_FN ADRH_FORCE_INLINE double pow(double a, double b) { return std::pow(a, b); }
    template <int N> ADRH_FN ADRH_FORCE_INLINE Dual<N> pow(const Dual<N> &a, double b)
    {
      Dual<N> r; r.v = std::pow(a.v, b);
      const double s = b * std::pow(a.v, b - 1.0);
      for (int i = 0; i < N; ++i) r.d[i] = s * a.d[i];
      return r;
    }
    ADRH_FN ADRH_FORCE_INLINE double sq(double a) { return a * a; }
    template <int N> ADRH_FN ADRH_FORCE_INLINE Dual<N> sq(const Dual<N> &a) { return a * a; }
  } // namespace fn
} // namespace adrh
