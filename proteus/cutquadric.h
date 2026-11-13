#ifndef PROTEUS_CUTQUADRIC_H
#define PROTEUS_CUTQUADRIC_H
// Curved cuts of the reference tetrahedron (and triangle) by a quadratic (P2) level set: classification, exact
// monomial moments of {u > 0}, and a surface quadrature rule on {u = 0}.
//
// C44 phase 1b (part A) owns this file. cutclip.h stays the planar kernel and the oracle; this header includes it for
// the monomial order and for the planar fast path (a quadric whose quadratic part is round-off is exactly a plane).
//
// Method (ported from the July cutquad.cpp tet_quad_u / tri_quad_u, restructured to one sweep): the inner coordinate x is
// integrated analytically, the two outer coordinates (y, z) by Gauss-Legendre on the segments between the critical
// values of the level set (discriminants, facet restrictions, branch line), with the cubic smoothstep map that turns the
// inverse-square-root endpoint behaviour into a regular integrand. One sweep gives every monomial moment.
//
// Conventions
//   * Everything is on the reference tet {x,y,z >= 0, x+y+z <= 1} (triangle {s,t >= 0, s+t <= 1}). Physical elements are
//     affine images; the caller applies |J| (or, for the surface rule, the J-dependent weights below).
//   * P2 nodal values u[10] are in proteus order: vertices 0..3, then the mid-edge nodes of edges
//     (0,1) (1,2) (2,3) (0,2) (1,3) (0,3). The triangle's six are vertices 0..2 then (0,1) (1,2) (0,2).
//   * Monomial coefficients and moments use cutclip::monomialIndex(): degree ascending, then x power descending, then
//     y power descending. For a quadric: [1, x, y, z, xx, xy, xz, yy, yz, zz]. Triangle: [1, s, t, ss, st, tt].
#include <algorithm>
#include <array>
#include <cmath>
#include <map>
#include <mutex>
#include <stdexcept>
#include <vector>

#include "cutclip.h"

namespace proteus
{
namespace cutquadric
{
using cutclip::monomialIndex;
using cutclip::numMonomials;

using P2Tet = std::array<double, 10>;
using P2Tri = std::array<double, 6>;
using Coef3 = std::array<double, 10>;  // 1, x, y, z, xx, xy, xz, yy, yz, zz
using Coef2 = std::array<double, 6>;   // 1, s, t, ss, st, tt

// proteus mid-edge node order on a tet, and its face/edge tables
static const int kEdge[6][2] = {{0, 1}, {1, 2}, {2, 3}, {0, 2}, {1, 3}, {0, 3}};

namespace detail
{
// linear form c + a x + b y + d z as {c, a, b, d}
using Lin = std::array<double, 4>;
inline Coef3 mulLin(const Lin &p, const Lin &q)
{
  return {p[0] * q[0],
          p[0] * q[1] + p[1] * q[0],
          p[0] * q[2] + p[2] * q[0],
          p[0] * q[3] + p[3] * q[0],
          p[1] * q[1],
          p[1] * q[2] + p[2] * q[1],
          p[1] * q[3] + p[3] * q[1],
          p[2] * q[2],
          p[2] * q[3] + p[3] * q[2],
          p[3] * q[3]};
}
using Lin2 = std::array<double, 3>;  // c + s a + t b
inline Coef2 mulLin2(const Lin2 &p, const Lin2 &q)
{
  return {p[0] * q[0], p[0] * q[1] + p[1] * q[0], p[0] * q[2] + p[2] * q[0], p[1] * q[1], p[1] * q[2] + p[2] * q[1],
          p[2] * q[2]};
}
}  // namespace detail

// Nodal values (proteus order) -> monomial coefficients. Closed form from u = sum_v u_v l_v (2 l_v - 1) + sum_e 4 u_e l_a l_b.
inline Coef3 p2ToMonomial(const P2Tet &u)
{
  using detail::Lin;
  const Lin lam[4] = {Lin{1, -1, -1, -1}, Lin{0, 1, 0, 0}, Lin{0, 0, 1, 0}, Lin{0, 0, 0, 1}};
  Coef3 c{};
  for (int v = 0; v < 4; ++v)
  {
    Lin t = lam[v];
    for (int k = 0; k < 4; ++k)
      t[k] *= 2.0;
    t[0] -= 1.0;
    const Coef3 pr = detail::mulLin(lam[v], t);
    for (int k = 0; k < 10; ++k)
      c[k] += u[v] * pr[k];
  }
  for (int e = 0; e < 6; ++e)
  {
    const Coef3 pr = detail::mulLin(lam[kEdge[e][0]], lam[kEdge[e][1]]);
    for (int k = 0; k < 10; ++k)
      c[k] += 4.0 * u[4 + e] * pr[k];
  }
  return c;
}

// Triangle: vertices 0,1,2 then mid-edge nodes (0,1) (1,2) (0,2).
inline Coef2 p2TriToMonomial(const P2Tri &u)
{
  using detail::Lin2;
  const Lin2 lam[3] = {Lin2{1, -1, -1}, Lin2{0, 1, 0}, Lin2{0, 0, 1}};
  const int ed[3][2] = {{0, 1}, {1, 2}, {0, 2}};
  Coef2 c{};
  for (int v = 0; v < 3; ++v)
  {
    Lin2 t = lam[v];
    for (int k = 0; k < 3; ++k)
      t[k] *= 2.0;
    t[0] -= 1.0;
    const Coef2 pr = detail::mulLin2(lam[v], t);
    for (int k = 0; k < 6; ++k)
      c[k] += u[v] * pr[k];
  }
  for (int e = 0; e < 3; ++e)
  {
    const Coef2 pr = detail::mulLin2(lam[ed[e][0]], lam[ed[e][1]]);
    for (int k = 0; k < 6; ++k)
      c[k] += 4.0 * u[3 + e] * pr[k];
  }
  return c;
}

inline double evalQuadric(const Coef3 &c, double x, double y, double z)
{
  return c[0] + x * (c[1] + c[4] * x + c[5] * y + c[6] * z) + y * (c[2] + c[7] * y + c[8] * z) + z * (c[3] + c[9] * z);
}
inline double evalQuadric(const Coef2 &c, double s, double t)
{
  return c[0] + s * (c[1] + c[3] * s + c[4] * t) + t * (c[2] + c[5] * t);
}

// ------------------------------------------------------------------------------------------------ classification
enum class CutLabel
{
  UncutPositive,
  UncutNegative,
  Simple,        // vertex signs see the cut and no edge is cut twice
  EdgeCutTwice,  // some edge has two roots in (0,1)
  SubNodal       // all ten nodal values of one sign and the element is still cut (a cap through a face, a closed component)
};

inline const char *labelName(CutLabel l)
{
  switch (l)
  {
  case CutLabel::UncutPositive:
    return "uncut_positive";
  case CutLabel::UncutNegative:
    return "uncut_negative";
  case CutLabel::Simple:
    return "simple";
  case CutLabel::EdgeCutTwice:
    return "edge_cut_twice";
  default:
    return "sub_nodal";
  }
}

struct Classification
{
  CutLabel label = CutLabel::UncutNegative;
  bool cut = false;
  double bernsteinMin = 0, bernsteinMax = 0;  // convex-hull bounds of the P2 function
  double rangeMin = 0, rangeMax = 0;          // exact range over the closed element
  int doubleEdges = 0;                        // edges with two roots in (0,1)
};

namespace detail
{
// roots of c0 + c1 t + c2 t^2 strictly inside (0,1) that change sign, as a count
inline int rootsInUnitInterval(double c0, double c1, double c2, double scale)
{
  const double tiny = 1e-14 * scale;
  auto inside = [](double r) { return r > 0.0 && r < 1.0; };
  if (std::fabs(c2) <= tiny)
  {
    if (std::fabs(c1) <= tiny)
      return 0;
    return inside(-c0 / c1) ? 1 : 0;
  }
  const double disc = c1 * c1 - 4.0 * c2 * c0;
  if (disc <= 0.0)
    return 0;
  const double s = std::sqrt(disc);
  const double q = -0.5 * (c1 + (c1 >= 0.0 ? s : -s));
  int n = 0;
  if (inside(q / c2))
    ++n;
  if (q != 0.0 && inside(c0 / q))
    ++n;
  return n;
}
}  // namespace detail

// Exact range of a P2 tetrahedron function: vertices, the critical point of each edge restriction (6), face restriction
// (4) and the interior, each kept when it lies in its entity.
inline void rangeOfP2(const P2Tet &u, double &lo, double &hi)
{
  const Coef3 c = p2ToMonomial(u);
  lo = hi = u[0];
  auto take = [&](double v) {
    lo = std::min(lo, v);
    hi = std::max(hi, v);
  };
  for (int i = 1; i < 10; ++i)
    take(u[i]);  // vertices and mid-edge nodes (all inside)
  double scale = 0.0;
  for (double v : c)
    scale = std::max(scale, std::fabs(v));
  const double tiny = 1e-14 * scale;
  // edges: f(t) = ua + (4 um - 3 ua - ub) t + (2 ua + 2 ub - 4 um) t^2
  for (int e = 0; e < 6; ++e)
  {
    const double ua = u[kEdge[e][0]], ub = u[kEdge[e][1]], um = u[4 + e];
    const double A = 2 * ua + 2 * ub - 4 * um, B = 4 * um - 3 * ua - ub;
    if (std::fabs(A) > tiny)
    {
      const double t = -B / (2 * A);
      if (t > 0.0 && t < 1.0)
        take(ua + t * (B + t * A));
    }
  }
  // faces: restriction is a 2D quadratic in the face parameters (s,t), vertices v0<v1<v2 of the face opposite vertex k
  for (int k = 0; k < 4; ++k)
  {
    int v[3], n = 0;
    for (int m = 0; m < 4; ++m)
      if (m != k)
        v[n++] = m;
    auto midIndex = [&](int a, int b) {
      for (int e = 0; e < 6; ++e)
        if (kEdge[e][0] == a && kEdge[e][1] == b)
          return 4 + e;
      return -1;
    };
    const P2Tri f = {u[v[0]], u[v[1]], u[v[2]], u[midIndex(v[0], v[1])], u[midIndex(v[1], v[2])],
                     u[midIndex(v[0], v[2])]};
    const Coef2 q = p2TriToMonomial(f);
    const double det = 4 * q[3] * q[5] - q[4] * q[4];
    if (std::fabs(det) > 1e-14 * scale * scale)
    {
      // grad = 0: [2 ss, st; st, 2 tt] (s,t) = -(q1, q2)
      const double s = (-q[1] * 2 * q[5] + q[2] * q[4]) / det;
      const double t = (-q[2] * 2 * q[3] + q[1] * q[4]) / det;
      if (s > 0.0 && t > 0.0 && s + t < 1.0)
        take(evalQuadric(q, s, t));
    }
  }
  // interior: Hessian solve
  {
    const double H[3][3] = {{2 * c[4], c[5], c[6]}, {c[5], 2 * c[7], c[8]}, {c[6], c[8], 2 * c[9]}};
    const double g[3] = {-c[1], -c[2], -c[3]};
    const double det = H[0][0] * (H[1][1] * H[2][2] - H[1][2] * H[2][1]) - H[0][1] * (H[1][0] * H[2][2] - H[1][2] * H[2][0]) +
                       H[0][2] * (H[1][0] * H[2][1] - H[1][1] * H[2][0]);
    if (std::fabs(det) > 1e-14 * scale * scale * scale)
    {
      const double x = (g[0] * (H[1][1] * H[2][2] - H[1][2] * H[2][1]) - H[0][1] * (g[1] * H[2][2] - H[1][2] * g[2]) +
                        H[0][2] * (g[1] * H[2][1] - H[1][1] * g[2])) /
                       det;
      const double y = (H[0][0] * (g[1] * H[2][2] - H[1][2] * g[2]) - g[0] * (H[1][0] * H[2][2] - H[1][2] * H[2][0]) +
                        H[0][2] * (H[1][0] * g[2] - g[1] * H[2][0])) /
                       det;
      const double z = (H[0][0] * (H[1][1] * g[2] - g[1] * H[2][1]) - H[0][1] * (H[1][0] * g[2] - g[1] * H[2][0]) +
                        g[0] * (H[1][0] * H[2][1] - H[1][1] * H[2][0])) /
                       det;
      if (x > 0.0 && y > 0.0 && z > 0.0 && x + y + z < 1.0)
        take(evalQuadric(c, x, y, z));
    }
  }
}

// Bernstein bounds, exact range, and the label. Step 1 (Bernstein) is sufficient for "uncut"; step 2 (exact range)
// decides; step 3 labels the configuration for counting and policy (the integration treats all labels alike).
inline Classification classify(const P2Tet &u)
{
  Classification r;
  r.bernsteinMin = r.bernsteinMax = u[0];
  auto bern = [&](double b) {
    r.bernsteinMin = std::min(r.bernsteinMin, b);
    r.bernsteinMax = std::max(r.bernsteinMax, b);
  };
  for (int i = 1; i < 4; ++i)
    bern(u[i]);
  for (int e = 0; e < 6; ++e)
    bern(2.0 * u[4 + e] - 0.5 * (u[kEdge[e][0]] + u[kEdge[e][1]]));
  double scale = 0.0;
  for (double v : u)
    scale = std::max(scale, std::fabs(v));
  rangeOfP2(u, r.rangeMin, r.rangeMax);
  if (r.bernsteinMin >= 0.0 && r.bernsteinMax > 0.0)  // step 1: convex hull positive
  {
    r.label = CutLabel::UncutPositive;
    return r;
  }
  if (r.bernsteinMax <= 0.0)
  {
    r.label = CutLabel::UncutNegative;
    return r;
  }
  if (r.rangeMin >= 0.0)  // step 2: exact range
  {
    r.label = CutLabel::UncutPositive;
    return r;
  }
  if (r.rangeMax <= 0.0)
  {
    r.label = CutLabel::UncutNegative;
    return r;
  }
  r.cut = true;
  for (int e = 0; e < 6; ++e)
  {
    const double ua = u[kEdge[e][0]], ub = u[kEdge[e][1]], um = u[4 + e];
    if (detail::rootsInUnitInterval(ua, 4 * um - 3 * ua - ub, 2 * ua + 2 * ub - 4 * um, scale) == 2)
      ++r.doubleEdges;
  }
  bool allPos = true, allNeg = true;
  for (double v : u)
  {
    allPos = allPos && v > 0.0;
    allNeg = allNeg && v < 0.0;
  }
  if (allPos || allNeg)
    r.label = CutLabel::SubNodal;
  else if (r.doubleEdges > 0)
    r.label = CutLabel::EdgeCutTwice;
  else
    r.label = CutLabel::Simple;
  return r;
}
}  // namespace cutquadric
}  // namespace proteus
#endif
