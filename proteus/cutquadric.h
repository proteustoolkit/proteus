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

// ------------------------------------------------------------------------------------------------ the sweep
struct Options
{
  int order = 16;           // Gauss points per outer segment per outer coordinate
  double planarTol = 1e-12; // quadratic part below this fraction of the largest coefficient: plane, use cutclip (<= 0: never)
  double deflateTol = 1e-10; // drop leading polynomial coefficients below this fraction of the largest before a root solve
};

namespace detail
{
struct GL
{
  std::vector<double> x, w;  // on [0,1]
};

inline const GL &gauss01(int n)
{
  static std::map<int, GL> cache;
  static std::mutex mtx;
  std::lock_guard<std::mutex> lock(mtx);
  auto it = cache.find(n);
  if (it != cache.end())
    return it->second;
  GL g;
  g.x.resize(n);
  g.w.resize(n);
  for (int i = 0; i < n; ++i)
  {
    double x = std::cos(M_PI * (i + 0.75) / (n + 0.5));
    double p0 = 1.0, p1 = 0.0;
    for (int it2 = 0; it2 < 100; ++it2)
    {
      p0 = 1.0;
      p1 = 0.0;
      for (int k = 0; k < n; ++k)
      {
        const double p2 = p1;
        p1 = p0;
        p0 = ((2.0 * k + 1.0) * x * p1 - k * p2) / (k + 1.0);
      }
      const double dp = n * (x * p0 - p1) / (x * x - 1.0);
      const double dx = p0 / dp;
      x -= dx;
      if (std::fabs(dx) < 1e-15)
      {
        p0 = 1.0;
        p1 = 0.0;
        for (int k = 0; k < n; ++k)
        {
          const double p2 = p1;
          p1 = p0;
          p0 = ((2.0 * k + 1.0) * x * p1 - k * p2) / (k + 1.0);
        }
        break;
      }
    }
    const double dp = n * (x * p0 - p1) / (x * x - 1.0);
    g.x[i] = 0.5 * (x + 1.0);
    g.w[i] = 0.5 * 2.0 / ((1.0 - x * x) * dp * dp);
  }
  return cache.emplace(n, std::move(g)).first->second;
}

// stable real roots of a x^2 + b x + c; a tangential double root is returned once (it is a split point)
inline int quadRoots(double a, double b, double c, double eps, double r[2])
{
  if (std::fabs(a) <= eps)
  {
    if (std::fabs(b) <= eps)
      return 0;
    r[0] = -c / b;
    return 1;
  }
  const double disc = b * b - 4.0 * a * c;
  if (disc <= 0.0)
  {
    if (disc > -eps * eps)
    {
      r[0] = -b / (2.0 * a);
      return 1;
    }
    return 0;
  }
  const double s = std::sqrt(disc);
  const double q = -0.5 * (b + (b >= 0.0 ? s : -s));
  r[0] = q / a;
  r[1] = (q != 0.0 ? c / q : -b / (2.0 * a));
  if (r[0] > r[1])
    std::swap(r[0], r[1]);
  return 2;
}

// real roots in (lo-eps, hi+eps) of c0 + c1 z + c2 z^2. Leading coefficients below deflate * (largest) are round-off and
// are dropped first: a root finder handed 1e-17 z^2 + b z + c otherwise returns a root of the wrong polynomial.
inline void polyRoots(double c0, double c1, double c2, double lo, double hi, double eps, double deflate,
                      std::vector<double> &out)
{
  const double scale = std::max({std::fabs(c0), std::fabs(c1), std::fabs(c2)});
  if (scale == 0.0)
    return;
  if (std::fabs(c2) < deflate * scale)
    c2 = 0.0;
  if (c2 == 0.0 && std::fabs(c1) < deflate * scale)
    return;
  double r[2];
  const int n = quadRoots(c2, c1, c0, 1e-13 * scale, r);
  for (int i = 0; i < n; ++i)
    if (lo - eps < r[i] && r[i] < hi + eps)
      out.push_back(r[i]);
}

// sorted, clamped, de-duplicated critical values with the two end points
inline std::vector<double> segments(std::vector<double> crit, double hiEnd, double eps)
{
  std::sort(crit.begin(), crit.end());
  std::vector<double> segs;
  for (double t : crit)
  {
    if (t < -eps || t > hiEnd + eps)
      continue;
    t = std::min(std::max(t, 0.0), hiEnd);
    if (segs.empty() || t - segs.back() > 1e-14)
      segs.push_back(t);
  }
  if (segs.empty() || segs.front() > 0.0)
    segs.insert(segs.begin(), 0.0);
  if (segs.back() < hiEnd)
    segs.push_back(hiEnd);
  return segs;
}

// cubic smoothstep t -> s(t), with ds/dt
inline void smooth(double t, double &s, double &j)
{
  s = t * t * (3.0 - 2.0 * t);
  j = 6.0 * t * (1.0 - t);
}

inline double maxAbs(const Coef3 &c)
{
  double m = 0.0;
  for (double v : c)
    m = std::max(m, std::fabs(v));
  return m;
}

// Visit every outer Gauss point (y, z) of the sweep with its weight (segment length, Gauss weight and smoothstep
// Jacobian in both outer directions) and the inner polynomial a2 x^2 + a1 x + a0 on x in [0, L], L = 1 - y - z.
// The outer partition is by the critical values of the level set (the x-discriminant, the two facet restrictions, the
// branch line) so that every outer integrand is smooth on a segment up to the endpoint square roots the map removes.
template <class Visitor>
inline void sweepOuter(const Coef3 &c, const Options &opt, Visitor &&visit)
{
  const double scale = maxAbs(c);
  if (scale == 0.0)
    return;
  const double eps = 1e-12 * scale;
  const double c1 = c[0], cx = c[1], cy = c[2], cz = c[3], cxx = c[4], cxy = c[5], cxz = c[6], cyy = c[7], cyz = c[8],
               czz = c[9];
  const double a2 = cxx;
  // quadratic-in-y families alpha, beta(z) linear, gamma(z) quadratic
  struct Fam
  {
    double al;
    double be[2];
    double ga[3];
  };
  std::vector<Fam> fams;
  fams.push_back({cxy * cxy - 4 * a2 * cyy,
                  {2 * cxy * cx - 4 * a2 * cy, 2 * cxy * cxz - 4 * a2 * cyz},
                  {cx * cx - 4 * a2 * c1, 2 * cx * cxz - 4 * a2 * cz, cxz * cxz - 4 * a2 * czz}});
  fams.push_back({cyy, {cy, cyz}, {c1, cz, czz}});
  fams.push_back({cxx + cyy - cxy,
                  {cy - cx + cxy - 2 * cxx, 2 * cxx - cxy + cyz - cxz},
                  {c1 + cx + cxx, cz - cx - 2 * cxx + cxz, cxx + czz - cxz}});
  if (std::fabs(cxy) > eps)
    fams.push_back({0.0, {cxy, 0.0}, {cx, cxz, 0.0}});

  std::vector<double> zcrit = {0.0, 1.0};
  for (const Fam &F : fams)
  {
    // beta^2 - 4 alpha gamma = 0 (birth/death), gamma = 0 (exit through y = 0), value at y = 1 - z (exit through the facet)
    polyRoots(F.be[0] * F.be[0] - 4 * F.al * F.ga[0], 2 * F.be[0] * F.be[1] - 4 * F.al * F.ga[1],
              F.be[1] * F.be[1] - 4 * F.al * F.ga[2], 0.0, 1.0, eps, opt.deflateTol, zcrit);
    polyRoots(F.ga[0], F.ga[1], F.ga[2], 0.0, 1.0, eps, opt.deflateTol, zcrit);
    polyRoots(F.al + F.be[0] + F.ga[0], -2 * F.al + F.be[1] - F.be[0] + F.ga[1], F.al - F.be[1] + F.ga[2], 0.0, 1.0, eps,
              opt.deflateTol, zcrit);
  }

  const GL &g = gauss01(opt.order);
  const std::vector<double> zsegs = segments(zcrit, 1.0, eps);
  std::vector<double> ycrit;
  for (std::size_t zs = 0; zs + 1 < zsegs.size(); ++zs)
  {
    const double za = zsegs[zs], zh = zsegs[zs + 1] - za;
    if (zh < 1e-14)
      continue;
    for (int zi = 0; zi < opt.order; ++zi)
    {
      double sz, jz;
      smooth(g.x[zi], sz, jz);
      const double z = za + zh * sz;
      const double wz = g.w[zi] * zh * jz;
      const double Ly = 1.0 - z;
      ycrit.clear();
      for (const Fam &F : fams)
      {
        double r[2];
        const int n = quadRoots(F.al, F.be[0] + F.be[1] * z, F.ga[0] + z * (F.ga[1] + z * F.ga[2]), eps, r);
        for (int i = 0; i < n; ++i)
          if (eps < r[i] && r[i] < Ly - eps)
            ycrit.push_back(r[i]);
      }
      const std::vector<double> ysegs = segments(ycrit, Ly, eps);
      for (std::size_t ys = 0; ys + 1 < ysegs.size(); ++ys)
      {
        const double ya = ysegs[ys], yh = ysegs[ys + 1] - ya;
        if (yh < 1e-14)
          continue;
        for (int yi = 0; yi < opt.order; ++yi)
        {
          double sy, jy;
          smooth(g.x[yi], sy, jy);
          const double y = ya + yh * sy;
          const double L = 1.0 - y - z;
          if (L <= eps)
            continue;
          const double b1 = cxy * y + cxz * z + cx;
          const double b0 = cyy * y * y + czz * z * z + cyz * y * z + cy * y + cz * z + c1;
          visit(y, z, wz * g.w[yi] * yh * jy, a2, b1, b0, L, eps);
        }
      }
    }
  }
}

// intervals of [0, L] where a2 x^2 + a1 x + a0 > 0 (at most two); returns their count
inline int positiveIntervals(double a2, double a1, double a0, double L, double eps, double lo[2], double hi[2])
{
  double r[2];
  const int n = quadRoots(a2, a1, a0, eps, r);
  double pts[4] = {0.0, L, L, L};
  int np = 1;
  for (int i = 0; i < n; ++i)
    if (eps < r[i] && r[i] < L - eps)
      pts[np++] = r[i];
  pts[np++] = L;
  int count = 0;
  for (int k = 0; k + 1 < np; ++k)
  {
    if (pts[k + 1] - pts[k] < 1e-15)
      continue;
    const double xm = 0.5 * (pts[k] + pts[k + 1]);
    if ((a2 * xm + a1) * xm + a0 > 0.0)
    {
      lo[count] = pts[k];
      hi[count] = pts[k + 1];
      ++count;
    }
  }
  return count;
}

inline bool isPlanar(const Coef3 &c, double planarTol)
{
  if (planarTol <= 0.0)
    return false;
  const double scale = maxAbs(c);
  double q = 0.0;
  for (int k = 4; k < 10; ++k)
    q = std::max(q, std::fabs(c[k]));
  return q <= planarTol * scale;
}

inline cutclip::Tet refTet()
{
  return {cutclip::Vec3{0, 0, 0}, cutclip::Vec3{1, 0, 0}, cutclip::Vec3{0, 1, 0}, cutclip::Vec3{0, 0, 1}};
}
// {x : a.x + b > 0} for the linear part of the quadric
inline cutclip::Plane planeOf(const Coef3 &c) { return {c[1], c[2], c[3], c[0]}; }
}  // namespace detail

// Monomial moments of {u > 0} over the reference tet, every monomial of total degree <= degree in monomialIndex order.
// Exactly planar data (quadratic part round-off) goes to cutclip, which is exact and cheaper; otherwise one slicing
// sweep builds the interval list once per outer point and accumulates every antiderivative x^(a+1)/(a+1) y^b z^c.
inline std::vector<double> volumeMoments(const Coef3 &c, int degree, const Options &opt = Options())
{
  if (degree < 0 || degree > 7)
    throw std::invalid_argument("cutquadric: moments are supported to degree 7 (cutclip's limit)");
  if (detail::maxAbs(c) == 0.0)
    return std::vector<double>(numMonomials(degree), 0.0);
  if (detail::isPlanar(c, opt.planarTol))
    return cutclip::moments(cutclip::clipTet(detail::refTet(), std::vector<cutclip::Plane>{detail::planeOf(c)}), degree);
  const auto idx = monomialIndex(degree);
  std::vector<double> m(idx.size(), 0.0);
  std::vector<double> yp(degree + 1), zp(degree + 1), lp(degree + 2), hp(degree + 2);
  detail::sweepOuter(c, opt, [&](double y, double z, double w, double a2, double a1, double a0, double L, double eps) {
    double lo[2], hi[2];
    const int ni = detail::positiveIntervals(a2, a1, a0, L, eps, lo, hi);
    if (ni == 0)
      return;
    yp[0] = zp[0] = 1.0;
    for (int k = 1; k <= degree; ++k)
    {
      yp[k] = yp[k - 1] * y;
      zp[k] = zp[k - 1] * z;
    }
    for (int i = 0; i < ni; ++i)
    {
      lp[0] = hp[0] = 1.0;
      for (int k = 1; k <= degree + 1; ++k)
      {
        lp[k] = lp[k - 1] * lo[i];
        hp[k] = hp[k - 1] * hi[i];
      }
      for (std::size_t n = 0; n < idx.size(); ++n)
      {
        const int a = idx[n][0];
        m[n] += w * (hp[a + 1] - lp[a + 1]) / (a + 1) * yp[idx[n][1]] * zp[idx[n][2]];
      }
    }
  });
  return m;
}
// Coef3 and P2Tet are both std::array<double,10>: the nodal-value entry point has its own name
inline std::vector<double> volumeMomentsNodal(const P2Tet &u, int degree, const Options &opt = Options())
{
  return volumeMoments(p2ToMonomial(u), degree, opt);
}


// ------------------------------------------------------------------------------------------------ triangles and faces
// 2D monomial order of a triangle's moments: degree ascending, s power descending.
inline std::vector<std::array<int, 2>> monomialIndex2(int degree)
{
  std::vector<std::array<int, 2>> idx;
  for (int n = 0; n <= degree; ++n)
    for (int p = n; p >= 0; --p)
      idx.push_back({p, n - p});
  return idx;
}
inline int numMonomials2(int degree) { return (degree + 1) * (degree + 2) / 2; }
inline int index2(int p, int q) { return (p + q) * (p + q + 1) / 2 + q; }

// P2 values of face k (opposite vertex k) from the element's ten: (v0, v1, v2, (v0,v1), (v1,v2), (v0,v2)) with v the
// other three vertices in increasing order. Depends only on the six nodal values the two neighbours share.
inline P2Tri faceNodal(const P2Tet &u, int k)
{
  int v[3], n = 0;
  for (int m = 0; m < 4; ++m)
    if (m != k)
      v[n++] = m;
  auto mid = [&](int a, int b) {
    for (int e = 0; e < 6; ++e)
      if (kEdge[e][0] == a && kEdge[e][1] == b)
        return u[4 + e];
    return 0.0;
  };
  return {u[v[0]], u[v[1]], u[v[2]], mid(v[0], v[1]), mid(v[1], v[2]), mid(v[0], v[2])};
}

enum class FaceLabel
{
  UncutPositive,
  UncutNegative,
  Cut
};
inline const char *faceLabelName(FaceLabel l)
{
  return l == FaceLabel::UncutPositive ? "uncut_positive" : (l == FaceLabel::UncutNegative ? "uncut_negative" : "cut");
}
struct FaceClassification
{
  FaceLabel label = FaceLabel::UncutNegative;
  bool cut = false;
  double bernsteinMin = 0, bernsteinMax = 0, rangeMin = 0, rangeMax = 0;
};

// Exact range of a P2 triangle function: 3 vertices, 3 edge critical points, the interior critical point.
inline void rangeOfP2Tri(const P2Tri &u, double &lo, double &hi)
{
  const Coef2 q = p2TriToMonomial(u);
  lo = hi = u[0];
  auto take = [&](double v) {
    lo = std::min(lo, v);
    hi = std::max(hi, v);
  };
  for (int i = 1; i < 6; ++i)
    take(u[i]);
  double scale = 0.0;
  for (double v : q)
    scale = std::max(scale, std::fabs(v));
  const double tiny = 1e-14 * scale;
  const int ed[3][2] = {{0, 1}, {1, 2}, {0, 2}};
  for (int e = 0; e < 3; ++e)
  {
    const double ua = u[ed[e][0]], ub = u[ed[e][1]], um = u[3 + e];
    const double A = 2 * ua + 2 * ub - 4 * um, B = 4 * um - 3 * ua - ub;
    if (std::fabs(A) > tiny)
    {
      const double t = -B / (2 * A);
      if (t > 0.0 && t < 1.0)
        take(ua + t * (B + t * A));
    }
  }
  const double det = 4 * q[3] * q[5] - q[4] * q[4];
  if (std::fabs(det) > 1e-14 * scale * scale)
  {
    const double s = (-q[1] * 2 * q[5] + q[2] * q[4]) / det;
    const double t = (-q[2] * 2 * q[3] + q[1] * q[4]) / det;
    if (s > 0.0 && t > 0.0 && s + t < 1.0)
      take(evalQuadric(q, s, t));
  }
}

inline FaceClassification classifyTri(const P2Tri &u)
{
  FaceClassification r;
  r.bernsteinMin = r.bernsteinMax = u[0];
  auto bern = [&](double b) {
    r.bernsteinMin = std::min(r.bernsteinMin, b);
    r.bernsteinMax = std::max(r.bernsteinMax, b);
  };
  bern(u[1]);
  bern(u[2]);
  const int ed[3][2] = {{0, 1}, {1, 2}, {0, 2}};
  for (int e = 0; e < 3; ++e)
    bern(2.0 * u[3 + e] - 0.5 * (u[ed[e][0]] + u[ed[e][1]]));
  rangeOfP2Tri(u, r.rangeMin, r.rangeMax);
  if ((r.bernsteinMin >= 0.0 && r.bernsteinMax > 0.0) || r.rangeMin >= 0.0)
    r.label = FaceLabel::UncutPositive;
  else if (r.bernsteinMax <= 0.0 || r.rangeMax <= 0.0)
    r.label = FaceLabel::UncutNegative;
  else
  {
    r.label = FaceLabel::Cut;
    r.cut = true;
  }
  return r;
}

namespace detail
{
inline bool isPlanar(const Coef2 &c, double planarTol)
{
  if (planarTol <= 0.0)
    return false;
  double scale = 0.0;
  for (double v : c)
    scale = std::max(scale, std::fabs(v));
  return std::max({std::fabs(c[3]), std::fabs(c[4]), std::fabs(c[5])}) <= planarTol * scale;
}
}  // namespace detail

// Monomial moments s^p t^q of {u > 0} over the reference triangle, in monomialIndex2 order. Same slicing as the tet:
// inner coordinate s integrated exactly, outer t by Gauss-Legendre between the critical values with the smoothstep map.
inline std::vector<double> triMoments(const Coef2 &c, int degree, const Options &opt = Options())
{
  if (degree < 0 || degree > 7)
    throw std::invalid_argument("cutquadric: moments are supported to degree 7 (cutclip's limit)");
  const auto idx = monomialIndex2(degree);
  std::vector<double> m(idx.size(), 0.0);
  double scale = 0.0;
  for (double v : c)
    scale = std::max(scale, std::fabs(v));
  if (scale == 0.0)
    return m;
  if (detail::isPlanar(c, opt.planarTol))
  {
    // plane c0 + c1 s + c2 t in the z = 0 embedding of the reference triangle
    const cutclip::Tri tri = {cutclip::Vec3{0, 0, 0}, cutclip::Vec3{1, 0, 0}, cutclip::Vec3{0, 1, 0}};
    const auto m3 = cutclip::moments(cutclip::clipTri(tri, std::vector<cutclip::Plane>{{c[1], c[2], 0.0, c[0]}}), degree);
    const auto idx3 = monomialIndex(degree);
    for (std::size_t i = 0; i < idx3.size(); ++i)
      if (idx3[i][2] == 0)
        m[index2(idx3[i][0], idx3[i][1])] = m3[i];
    return m;
  }
  const double eps = 1e-12 * scale;
  const double F = c[0], D = c[1], E = c[2], A = c[3], B = c[4], C = c[5];
  std::vector<double> crit = {0.0, 1.0};
  detail::polyRoots(D * D - 4 * A * F, 2 * B * D - 4 * A * E, B * B - 4 * A * C, 0.0, 1.0, eps, opt.deflateTol, crit);  // discriminant
  detail::polyRoots(F, E, C, 0.0, 1.0, eps, opt.deflateTol, crit);                                                      // s = 0
  detail::polyRoots(A + D + F, -2 * A + B - D + E, A - B + C, 0.0, 1.0, eps, opt.deflateTol, crit);                       // s = 1 - t
  if (std::fabs(B) > eps)
    crit.push_back(-D / B);                                                                                            // branch
  const detail::GL &g = detail::gauss01(opt.order);
  const std::vector<double> segs = detail::segments(crit, 1.0, eps);
  std::vector<double> tp(degree + 1), lp(degree + 2), hp(degree + 2);
  for (std::size_t sg = 0; sg + 1 < segs.size(); ++sg)
  {
    const double a = segs[sg], h = segs[sg + 1] - a;
    if (h < 1e-14)
      continue;
    for (int i = 0; i < opt.order; ++i)
    {
      double sm, j;
      detail::smooth(g.x[i], sm, j);
      const double t = a + h * sm, w = g.w[i] * h * j;
      const double L = 1.0 - t;
      if (L <= eps)
        continue;
      double lo[2], hi[2];
      const int ni = detail::positiveIntervals(A, B * t + D, (C * t + E) * t + F, L, eps, lo, hi);
      tp[0] = 1.0;
      for (int k = 1; k <= degree; ++k)
        tp[k] = tp[k - 1] * t;
      for (int q = 0; q < ni; ++q)
      {
        lp[0] = hp[0] = 1.0;
        for (int k = 1; k <= degree + 1; ++k)
        {
          lp[k] = lp[k - 1] * lo[q];
          hp[k] = hp[k - 1] * hi[q];
        }
        for (std::size_t n = 0; n < idx.size(); ++n)
          m[n] += w * (hp[idx[n][0] + 1] - lp[idx[n][0] + 1]) / (idx[n][0] + 1) * tp[idx[n][1]];
      }
    }
  }
  return m;
}
inline std::vector<double> triMomentsNodal(const P2Tri &u, int degree, const Options &opt = Options())
{
  return triMoments(p2TriToMonomial(u), degree, opt);
}

// Moments of the 3D monomials x^p y^q z^r over F cap {u > 0}, F = face k of the reference tet, with respect to the face's
// parameter measure ds dt (full face: 1/2). The physical area element is |(x1-x0) x (x2-x0)| ds dt, the outward normal
// times it is the (outward) cross product: for the reference tet (1,1,1), (-1,0,0), (0,-1,0), (0,0,-1) for faces 0..3.
inline std::vector<double> faceMoments(const P2Tet &u, int k, int degree, const Options &opt = Options())
{
  const std::vector<double> m2 = triMomentsNodal(faceNodal(u, k), degree, opt);
  static const double X[4][3] = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
  int v[3], n = 0;
  for (int m = 0; m < 4; ++m)
    if (m != k)
      v[n++] = m;
  const auto idx = monomialIndex(degree);
  std::vector<double> out(idx.size(), 0.0);
  const int D = degree + 1;
  for (std::size_t i = 0; i < idx.size(); ++i)
  {
    // x_c = X0 + s (X1 - X0) + t (X2 - X0), as a polynomial in (s,t): product of (a + b s + c t)^e_c
    std::vector<double> poly(D * D, 0.0), tmp(D * D);
    poly[0] = 1.0;
    for (int comp = 0; comp < 3; ++comp)
    {
      const double a = X[v[0]][comp], b = X[v[1]][comp] - X[v[0]][comp], c = X[v[2]][comp] - X[v[0]][comp];
      for (int rep = 0; rep < idx[i][comp]; ++rep)
      {
        std::fill(tmp.begin(), tmp.end(), 0.0);
        for (int p = 0; p < D; ++p)
          for (int q = 0; q + p < D; ++q)
          {
            const double cur = poly[p * D + q];
            if (cur == 0.0)
              continue;
            tmp[p * D + q] += a * cur;
            if (p + 1 < D)
              tmp[(p + 1) * D + q] += b * cur;
            if (q + 1 < D)
              tmp[p * D + q + 1] += c * cur;
          }
        poly.swap(tmp);
      }
    }
    double s = 0.0;
    for (int p = 0; p < D; ++p)
      for (int q = 0; q + p < D; ++q)
        if (poly[p * D + q] != 0.0)
          s += poly[p * D + q] * m2[index2(p, q)];
    out[i] = s;
  }
  return out;
}

}  // namespace cutquadric
}  // namespace proteus
#endif
