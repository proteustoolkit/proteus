#!/bin/env python
import math
import sympy
from sympy.integrals import intpoly
import numpy as np
from sympy.abc import I, J, K, N
from sympy import Sum
from sympy.geometry import Triangle, Point
try:
    from sympy.printing.cxxcode import cxxcode   # sympy < 1.7
except ImportError:
    from sympy.printing.cxx import cxxcode       # sympy >= 1.7
x,y,z,x_0,y_0,z_0 = sympy.symbols('x,y,z,x_0,y_0,z_0')

f=open("equivalent_polynomials_coefficients.h",'w')
f.write("""#ifndef EQUIVALENT_POLYNOMIALS_COEFFICIENTS_H
#define EQUIVALENT_POLYNOMIALS_COEFFICIENTS_H
#include <cmath>

namespace equivalent_polynomials
{
  template<int nSpace, int nP>
  inline void _set_Ainv(double* Ainv);

  template<int nSpace, int nP>
  inline void _calculate_b(double* X_0, double* b_H, double* b_ImH, double* b_dH);

""")

for nSpace in range(1,4):
    for order in range(1,5):
        nDOF1D=order + 1
        if nSpace == 3:
            basis = [x**i*y**j*z**k for i in range(nDOF1D) for j in range(nDOF1D-i) for k in range(nDOF1D-i-j)]
            nDOF = len(basis)
            test_nDOF = sympy.factor(Sum(Sum(Sum(1,(K,0,N-I-J-1)),(J,0,N-I-1)),(I,0,N-1)).doit())
            unit_tet = [[(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)],
                        [1, 2, 3], [2, 3, 0], [3, 0, 1], [0, 1, 2]]
        elif nSpace == 2:
            basis = [x**i*y**j for i in range(nDOF1D) for j in range(nDOF1D-i)]
            nDOF = len(basis)
            test_nDOF = sympy.factor(Sum(Sum(1,(J,0,N-I-1)),(I,0,N-1)).doit())
            unit_triangle = Triangle(Point(0, 0), Point(0, 1), Point(1, 0))
        elif nSpace == 1:
            basis = [x**i for i in range(nDOF1D)]
            nDOF = len(basis)
            test_nDOF = sympy.factor(Sum(1,(I,0,N-1)).doit())
        # sympy >= 1.7 returns a Float here, which no longer compares equal to a python
        # int; the check is only a sanity check on the basis count, so compare as integers.
        assert(int(round(float(test_nDOF.evalf(subs={'N':nDOF1D})))) == nDOF)
        A = np.zeros((nDOF,nDOF),'d')
        for i in range(nDOF):
            for j in range(nDOF):
                if nSpace == 3:
                    A[i,j] = intpoly.polytope_integrate(unit_tet,sympy.expand(basis[j]*basis[i])).evalf()
                elif nSpace == 2:
                    A[i,j] = intpoly.polytope_integrate(unit_triangle,sympy.expand(basis[j]*basis[i])).evalf()
                elif nSpace == 1:
                    A[i,j] = sympy.integrate(basis[j]*basis[i],(x,0,1)).evalf()
        Ainv = np.linalg.inv(A)
        b_H=[]
        b_dH_x=[]
        b_dH_y=[]
        b_dH_z=[]
        b_1mH=[]
        if nSpace == 3:
            sub_tet = [[(0, 0, 0), (x_0, 0, 0), (0, y_0, 0), (0, 0, z_0)],
                       [1, 2, 3], [2, 3, 0], [3, 0, 1], [0, 1, 2]]
            for i in range(nDOF):
                b_1mH.append(sympy.simplify(intpoly.polytope_integrate(sub_tet, basis[i])))
                b_H.append(sympy.simplify(intpoly.polytope_integrate(unit_tet, basis[i]) 
                                          - sympy.simplify(intpoly.polytope_integrate(sub_tet, basis[i]))))
                b_dH_x.append(sympy.simplify(b_H[i].diff(x_0)))
                b_dH_y.append(sympy.simplify(b_H[i].diff(y_0)))
                b_dH_z.append(sympy.simplify(b_H[i].diff(z_0)))
        elif nSpace == 2:
            sub_triangle = Triangle(Point(0, 0), Point(0, y_0), Point(x_0, 0))
            for i in range(nDOF):
                b_1mH.append(intpoly.polytope_integrate(sub_triangle, basis[i]))
                b_H.append(intpoly.polytope_integrate(unit_triangle, basis[i]) 
                           -intpoly.polytope_integrate(sub_triangle, basis[i]))
                b_dH_x.append(sympy.simplify(b_H[i].diff(x_0)))
                b_dH_y.append(sympy.simplify(b_H[i].diff(y_0)))
        elif nSpace == 1:
            for i in range(nDOF):
                b_1mH.append(sympy.integrate(basis[i],(x,0,x_0)))
                b_H.append(sympy.integrate(basis[i],(x,x_0,1)))
                b_dH_x.append(b_H[i].diff(x_0))

        f.write("""  template<>
  inline void _set_Ainv<{0:d},{1:d}>(double* Ainv)
  {{
""".format(nSpace, order))
        for i in range(nDOF):
            for j in range(nDOF):
                # repr() of a numpy scalar is "np.float64(...)" under numpy >= 2, which is not
                # valid C++; go through a python float so the literal is emitted bare.
                f.write("    Ainv[{0:d}] = {1};\n".format(i*len(b_H)+j,repr(float(Ainv[i,j]))))
        f.write("""  }}

  template<>
  inline void _calculate_b<{0:d},{1:d}>(double* X_0, double* b_H, double* b_ImH, double* b_dH)
  {{
""".format(nSpace, order))
        if nSpace == 3:
            f.write("""    const double x_0(X_0[0]), y_0(X_0[1]), z_0(X_0[2]);
""")
        if nSpace == 2:
            f.write("""    const double x_0(X_0[0]), y_0(X_0[1]);
""")
        if nSpace == 1:
            f.write("""    const double x_0(X_0[0]);
""")
        for i in range(len(b_H)):
            f.write("    b_H[{0:d}] = {1:s};\n".format(i,cxxcode(b_H[i])))
            f.write("    b_ImH[{0:d}] = {1:s};\n".format(i,cxxcode(b_1mH[i])))
            f.write("    b_dH[{0:d}] = {1:s};\n".format(i*nSpace,cxxcode(b_dH_x[i])))
            if nSpace > 1:
                f.write("    b_dH[{0:d}] = {1:s};\n".format(i*nSpace+1,cxxcode(b_dH_y[i])))
            if nSpace > 2:
                f.write("    b_dH[{0:d}] = {1:s};\n".format(i*nSpace+2,cxxcode(b_dH_z[i])))
        f.write("""  }

""")
# ---------------------------------------------------------------------------------------------
# Composite equivalent polynomials: one fit per region of an element cut by TWO level sets.
#
# Same construction as _calculate_b/_set_Ainv above -- a vector of region moments against the
# monomial basis, multiplied by the inverse Gram matrix of that basis over the reference element.
# The only difference is where the moments come from: _calculate_b has closed forms for a single
# cut, while two cuts produce a polygon whose moments are computed at run time. So this part is
# emitted as generic code rather than per-(nSpace,nP) tables, and reuses _set_Ainv<2,nP> for the
# solve instead of duplicating the Gram matrix.
#
# Motivation: the product ImH_f*H_s of two separate single-cut fits is NOT moment-exact on such an
# element -- it lies in P^{2nP}, which nothing constrains. See .notes/EQUIVALENT_POLYNOMIALS.md
# section 7.1 and test/test_equivalent_polynomials.py.
#
# Basis ordering below MUST match the `basis` list above: x**i * y**j with i outer, j inner.
# ---------------------------------------------------------------------------------------------
# Edge-restricted moment fit. Same construction as the composite fit below and as
# _calculate_C: moments times the inverse Gram matrix. It consumes the generated 1D tables
# (_set_Ainv<1,nP> / _calculate_b<1,nP>) and nothing else, so it belongs with them rather
# than in the hand-written header.
f.write("""
  // Edge-restricted equivalent-polynomial (moment-fit) Heaviside for a single mesh edge, treated
  // as its own 1D domain with reference parameter t in [0,1] running from the edge's node 0 to its
  // node 1. Purpose: let an integrand that is DISCONTINUOUS at the interface crossing point theta
  // be integrated with the ordinary whole-edge quadrature rule. The fit satisfies
  //     int_0^1 H_hat(t) p(t) dt = int_theta^1 p(t) dt   exactly for every p in P^nP,
  // so (1-H_hat)*P_a + H_hat*P_b integrates to exactly int_0^theta P_a + int_theta^1 P_b whenever
  // P_a,P_b are polynomials of degree <= nP -- which covers the facet integrands here (degree <= 1
  // for P1, <= 3 for P2, against nP = 4).
  //
  // NOTE this is the *Heaviside*, used purely as an integration weight for an indicator function.
  // It is deliberately NOT the Dirac: a mesh facet already carries its own arc-length measure, so
  // no delta surrogate is needed (an earlier attempt that used one was wrong -- see
  // .notes/SCIFEM_INTERFACE_CONSISTENCY.md). The pointwise overshoot inherent to moment fits is harmless
  // here because H_hat only ever multiplies genuine polynomials, which is exactly the regime the
  // moment guarantee covers.
  //
  // The 2D volume fit (_calculate_C()) evaluated at edge points is NOT a substitute: matching 2D
  // volume moments places no constraint on 1D edge-trace moments. phi0/phi1 must have opposite
  // signs (the edge must actually be cut). Orientation-safe: returns the fit for the indicator of
  // {phi > 0} whichever way round the edge is parametrized.
  template <int nP>
  inline void calculate_edge_H(double phi0, double phi1, double C_H[nP + 1])
  {
    double theta = -phi0 / (phi1 - phi0);
    double Ainv[(nP + 1) * (nP + 1)];
    _set_Ainv<1, nP>(Ainv);
    double b_H[nP + 1], b_ImH[nP + 1], b_dH[nP + 1];
    _calculate_b<1, nP>(&theta, b_H, b_ImH, b_dH);
    // _calculate_b assumes H=1 for t>theta (phi increasing along t); use the complementary
    // moments when phi decreases instead, so C_H always fits the indicator of {phi>0}.
    const double *b = (phi1 > phi0) ? b_H : b_ImH;
    for (int i = 0; i <= nP; i++)
    {
      C_H[i] = 0.0;
      for (int j = 0; j <= nP; j++)
        C_H[i] += Ainv[i * (nP + 1) + j] * b[j];
    }
  }

  template <int nP>
  inline double evaluate_edge_poly(const double C[nP + 1], double t)
  {
    double val = 0.0, tpow = 1.0;
    for (int i = 0; i <= nP; i++)
    {
      val += C[i] * tpow;
      tpow *= t;
    }
    return val;
  }
""")

f.write("""
  // -------------------------------------------------------------------------------------
  // Composite fits for an element cut by two level sets (2D simplices only).
  // Generated by scripts/gen_eqp.py -- edit there, not here.
  // -------------------------------------------------------------------------------------
  namespace composite
  {
    static const int MAX_POLY_VERTS = 8;   // a triangle clipped by two half-planes

    struct Poly
    {
      double x[MAX_POLY_VERTS];
      double y[MAX_POLY_VERTS];
      int n;
      Poly() : n(0) {}
      inline void push(double px, double py)
      {
        if (n < MAX_POLY_VERTS) { x[n] = px; y[n] = py; ++n; }
      }
    };

    // Sutherland-Hodgman: keep the vertices on the wanted side of the affine functional s,
    // inserting the crossing point on every edge that changes side.
    template <class S>
    inline Poly clip(const Poly &in, const S &s, bool keep_positive)
    {
      Poly out;
      for (int i = 0; i < in.n; ++i)
      {
        const int j = (i + 1) % in.n;
        const double si = s(in.x[i], in.y[i]);
        const double sj = s(in.x[j], in.y[j]);
        const bool ini = keep_positive ? (si >= 0.0) : (si <= 0.0);
        const bool inj = keep_positive ? (sj >= 0.0) : (sj <= 0.0);
        if (ini) out.push(in.x[i], in.y[i]);
        if (ini != inj)
        {
          const double d = si - sj;
          const double t = (std::fabs(d) > 0.0) ? si / d : 0.0;
          out.push(in.x[i] + t * (in.x[j] - in.x[i]),
                   in.y[i] + t * (in.y[j] - in.y[i]));
        }
      }
      if (out.n < 3) out.n = 0;   // degenerate: zero measure
      return out;
    }

    // 6-point Gauss-Legendre on [0,1]. With the Duffy map below this is exact for polynomials of
    // degree <= 11 over a triangle, well above the nP in use.
    inline const double *gl_nodes()
    {
      static const double a[6] = {0.03376524289842399, 0.16939530676686776, 0.38069040695840155,
                                  0.61930959304159845, 0.83060469323313224, 0.96623475710157601};
      return a;
    }
    inline const double *gl_weights()
    {
      static const double w[6] = {0.08566224618958517, 0.18038078652406930, 0.23395696728634552,
                                  0.23395696728634552, 0.18038078652406930, 0.08566224618958517};
      return w;
    }

    // Monomials in the same order as the generated basis: x**i * y**j, i outer.
    template <int nP>
    inline void monomials(double x, double y, double *m)
    {
      int idx = 0;
      for (int i = 0; i <= nP; ++i)
        for (int j = 0; j <= nP - i; ++j, ++idx)
        {
          double t = 1.0;
          for (int q = 0; q < i; ++q) t *= x;
          for (int q = 0; q < j; ++q) t *= y;
          m[idx] = t;
        }
    }

    template <int nP>
    inline double evaluate(const double *c, double x, double y)
    {
      double m[((nP + 1) * (nP + 2)) / 2];
      monomials<nP>(x, y, m);
      double s = 0.0;
      for (int i = 0; i < ((nP + 1) * (nP + 2)) / 2; ++i) s += c[i] * m[i];
      return s;
    }

    template <int nP>
    inline void triangle_moments(const double p0[2], const double p1[2], const double p2[2],
                                 double *moments)
    {
      const int nDOF = ((nP + 1) * (nP + 2)) / 2;
      const double J00 = p1[0] - p0[0], J01 = p2[0] - p0[0];
      const double J10 = p1[1] - p0[1], J11 = p2[1] - p0[1];
      const double detJ = std::fabs(J00 * J11 - J01 * J10);
      if (detJ == 0.0) return;
      const double *gp = gl_nodes(), *gw = gl_weights();
      double m[((nP + 1) * (nP + 2)) / 2];
      for (int a = 0; a < 6; ++a)
        for (int b = 0; b < 6; ++b)
        {
          const double u = gp[a], v = gp[b];              // Duffy map, jacobian (1-u)
          const double xi = u, eta = (1.0 - u) * v;
          const double w = gw[a] * gw[b] * (1.0 - u) * detJ;
          monomials<nP>(p0[0] + J00 * xi + J01 * eta,
                        p0[1] + J10 * xi + J11 * eta, m);
          for (int i = 0; i < nDOF; ++i) moments[i] += w * m[i];
        }
    }

    template <int nP>
    inline void polygon_moments(const Poly &p, double *moments)
    {
      const int nDOF = ((nP + 1) * (nP + 2)) / 2;
      for (int i = 0; i < nDOF; ++i) moments[i] = 0.0;
      for (int i = 1; i + 1 < p.n; ++i)
      {
        const double p0[2] = {p.x[0], p.y[0]};
        const double p1[2] = {p.x[i], p.y[i]};
        const double p2[2] = {p.x[i + 1], p.y[i + 1]};
        triangle_moments<nP>(p0, p1, p2, moments);
      }
    }

    // Composite weights for an element cut by both level sets.
    //
    //   phi_s, phi_f : level set values at the triangle's three vertices. Convention as in ADR.h:
    //                  phi_s > 0 is the active fluid, phi_f < 0 is Omega^-.
    //   cA           : fit for {phi_f < 0} intersect {phi_s > 0}   (the mua side)
    //   cB           : fit for {phi_f > 0} intersect {phi_s > 0}   (the mub side)
    //
    // Both are computed on the reference triangle, so the |J| carried by the Gram matrix and by
    // the moments cancels and the coefficients are directly usable.
    template <int nP>
    inline bool weights(const double phi_s[3], const double phi_f[3], double *cA, double *cB)
    {
      const int nDOF = ((nP + 1) * (nP + 2)) / 2;
      Poly T;
      T.push(0.0, 0.0); T.push(1.0, 0.0); T.push(0.0, 1.0);
      const double s0 = phi_s[0], sx = phi_s[1] - phi_s[0], sy = phi_s[2] - phi_s[0];
      const double f0 = phi_f[0], fx = phi_f[1] - phi_f[0], fy = phi_f[2] - phi_f[0];
      const Poly fluid = clip(T, [&](double x, double y) { return s0 + sx * x + sy * y; }, true);
      if (fluid.n == 0) return false;
      const Poly A = clip(fluid, [&](double x, double y) { return f0 + fx * x + fy * y; }, false);
      const Poly B = clip(fluid, [&](double x, double y) { return f0 + fx * x + fy * y; }, true);

      double bA[((nP + 1) * (nP + 2)) / 2], bB[((nP + 1) * (nP + 2)) / 2];
      polygon_moments<nP>(A, bA);
      polygon_moments<nP>(B, bB);

      // Same solve as _calculate_C: coefficients = Ainv * moments.
      double Ainv[(((nP + 1) * (nP + 2)) / 2) * (((nP + 1) * (nP + 2)) / 2)];
      _set_Ainv<2, nP>(Ainv);
      for (int i = 0; i < nDOF; ++i)
      {
        cA[i] = 0.0;
        cB[i] = 0.0;
        for (int j = 0; j < nDOF; ++j)
        {
          cA[i] += Ainv[i * nDOF + j] * bA[j];
          cB[i] += Ainv[i * nDOF + j] * bB[j];
        }
      }
      return true;
    }
  }//composite

""")
f.write("""}//equivalent_polynomials
#endif
""")
f.close()