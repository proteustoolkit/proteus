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
  int n;
  if (c2 != 0.0 && c1 * c1 - 4.0 * c2 * c0 > 0.0 && c1 * c1 - 4.0 * c2 * c0 <= 1e-13 * (c1 * c1 + std::fabs(4.0 * c2 * c0)))
  {
    // a double root computed from sqrt(discriminant) is only good to ~1e-8 and shows up as two events 1e-8 apart, which
    // leaves an outer segment of that width next to a singular end. Report one event.
    r[0] = -c1 / (2.0 * c2);
    n = 1;
  }
  else
    n = quadRoots(c2, c1, c0, 1e-13 * scale, r);
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

// cubic smoothstep t -> s(t) = t^2 (3 - 2t), with ds/dt = 6 t (1 - t): contact of order 2 at both ends turns an end
// singularity (z - a)^(-1/2) or (z - a)^(1/2) of the outer integrand into an analytic one. Gauss-Legendre nodes then lie at
// distances O(t^2) from the end, far above the resolution of z, so an inverse-square-root end loses nothing.
inline void smooth(double t, double &s, double &j)
{
  s = t * t * (3.0 - 2.0 * t);
  j = 6.0 * t * (1.0 - t);
}

// Double-exponential (tanh-sinh) nodes on [0,1] for segment ends where the outer integrand is logarithmic (the turning
// points of the silhouette curve, where two silhouettes merge: a run that ends on one of them integrates 1/sqrt(disc) to
// log(1/R) with R^2 proportional to the distance to the turning point). The cubic map gives only n^-4 there, tanh-sinh
// converges exponentially. It cannot be used at an inverse-square-root end: it truncates the tail at the resolution of z
// and loses sqrt(ulp) ~ 1e-8. Step h = 2/order. frac is the distance from the nearest end as a fraction of the segment.
struct TSNode
{
  double frac, w;
  bool left;
};
inline const std::vector<TSNode> &tanhSinh(int order)
{
  static std::map<int, std::vector<TSNode>> cache;
  static std::mutex mtx;
  std::lock_guard<std::mutex> lock(mtx);
  auto it = cache.find(order);
  if (it != cache.end())
    return it->second;
  std::vector<TSNode> nodes;
  const double h = 2.0 / std::max(order, 1), tmax = 3.2;
  const int K = (int)std::floor(tmax / h);
  for (int k = -K; k <= K; ++k)
  {
    const double t = k * h, e = std::exp(-M_PI * std::fabs(std::sinh(t)));
    const double frac = e / (1.0 + e);
    nodes.push_back({frac, h * M_PI * std::cosh(t) * frac * (1.0 - frac), k <= 0});
  }
  return cache.emplace(order, std::move(nodes)).first->second;
}

inline double maxAbs(const Coef3 &c)
{
  double m = 0.0;
  for (double v : c)
    m = std::max(m, std::fabs(v));
  return m;
}

// Integrate a root term over one run [A, B] of an outer line. Along the line the root is r = (-a1 +- sqrt(disc))/(2 a2)
// with disc(y) = al y^2 + be y + ga (the x-discriminant, a quadratic in the outer coordinate) and the only non-analytic
// dependence on y is sqrt(disc). Substituting y = m + R sin(phi) (al < 0), y = m +- R cosh(psi) (al > 0) or
// y = y1 + sigma tau^2 (al ~ 0) makes sqrt(disc) analytic on the whole natural domain of the branch, so a run that is
// truncated near a silhouette by some other critical value (a kink) is still integrated to round-off by plain
// Gauss-Legendre: no clustering of nodes toward its end can resolve a singularity that sits just outside it.
// emit(y, w, wOverRoot): w is the weight dy-measure x outer weight, wOverRoot = w / sqrt(disc(y)) formed analytically
// (the surface weight |grad u| / |du/dx| = |grad u| / sqrt(disc) needs it without a division of two small numbers).
template <class Emit>
inline void integrateRootRun(double A, double B, const GL &g, double al, double be, double ga, double wOuter, Emit &&emit)
{
  const int n = (int)g.x.size();
  const double S = std::max({std::fabs(al), std::fabs(be), std::fabs(ga)});
  if (S == 0.0 || B <= A)
    return;
  auto plain = [&]() {
    const double h = B - A;
    for (int i = 0; i < n; ++i)
    {
      const double y = A + h * g.x[i];
      const double d = (al * y + be) * y + ga;
      if (d <= 0.0)
        continue;
      const double w = wOuter * g.w[i] * h;
      emit(y, w, w / std::sqrt(d));
    }
  };
  if (std::fabs(al) <= 1e-10 * S)
  {
    if (std::fabs(be) <= 1e-14 * S)
    {
      plain();
      return;
    }
    // disc = be (y - y1): y = y1 + sigma tau^2
    const double y1 = -ga / be, sigma = be > 0.0 ? 1.0 : -1.0;
    if (std::fabs(y1 - 0.5 * (A + B)) > 20.0)
    {
      plain();  // the silhouette is far outside the unit range
      return;
    }
    const double tA = std::sqrt(std::max(sigma * (A - y1), 0.0)), tB = std::sqrt(std::max(sigma * (B - y1), 0.0));
    const double lo = std::min(tA, tB), hi = std::max(tA, tB), d = hi - lo, sb = std::sqrt(std::fabs(be));
    for (int i = 0; i < n; ++i)
    {
      const double tau = lo + d * g.x[i];
      emit(y1 + sigma * tau * tau, wOuter * g.w[i] * d * 2.0 * tau, wOuter * g.w[i] * d * 2.0 / sb);
    }
    return;
  }
  const double D2 = be * be - 4.0 * al * ga;
  if (al > 0.0 && D2 < 0.0)
  {
    // disc = al ((y - m)^2 + R^2) > 0 on the whole line, but the complex silhouettes m +- iR can lie close to the real
    // axis (near a turning point of the silhouette curve): y = m + R sinh(psi) makes sqrt(disc) = sqrt(al) R cosh(psi)
    const double m = -be / (2.0 * al), R = std::sqrt(-D2) / (2.0 * al);
    if (R <= 20.0)
    {
      const double pA = std::asinh((A - m) / R), pB = std::asinh((B - m) / R);
      const int pieces = std::max(1, (int)std::ceil(std::fabs(pB - pA)));
      const double d = (pB - pA) / pieces, sa = std::sqrt(al);
      for (int pc = 0; pc < pieces; ++pc)
        for (int i = 0; i < n; ++i)
        {
          const double psi = pA + d * (pc + g.x[i]);
          const double wp = wOuter * g.w[i] * std::fabs(d);
          emit(m + R * std::sinh(psi), wp * R * std::cosh(psi), wp / sa);
        }
      return;
    }
  }
  double r[2];
  const int nr = D2 > 0.0 ? quadRoots(al, be, ga, 1e-14 * S, r) : 0;
  if (nr < 2)
  {
    plain();  // disc keeps one sign on the line, no silhouette in play
    return;
  }
  const double y1 = r[0], y2 = r[1], m = 0.5 * (y1 + y2), R = 0.5 * (y2 - y1);
  const double tol = 1e-13 * std::max(1.0, std::fabs(R));
  if (R > 20.0)
  {
    // the other silhouette is far outside the unit range: one-sided map at the near root, disc = al (y - yn)(y - yf) with
    // the second factor analytic over the run
    const double ym = 0.5 * (A + B);
    const bool nearLow = std::fabs(ym - y1) < std::fabs(ym - y2);
    const double yn = nearLow ? y1 : y2, yf = nearLow ? y2 : y1, sigma = ym > yn ? 1.0 : -1.0;
    if (std::fabs(ym - yn) > 20.0)
    {
      plain();
      return;
    }
    const double tA = std::sqrt(std::max(sigma * (A - yn), 0.0)), tB = std::sqrt(std::max(sigma * (B - yn), 0.0));
    const double lo = std::min(tA, tB), hi = std::max(tA, tB), d = hi - lo;
    for (int i = 0; i < n; ++i)
    {
      const double tau = lo + d * g.x[i];
      const double y = yn + sigma * tau * tau;
      emit(y, wOuter * g.w[i] * d * 2.0 * tau, wOuter * g.w[i] * d * 2.0 / std::sqrt(std::fabs(al * (y - yf))));
    }
    return;
  }
  if (al < 0.0)
  {
    const double sa = std::sqrt(-al);
    const bool atLo = A <= y1 + tol, atHi = B >= y2 - tol;
    const double pA = atLo ? -M_PI / 2 : std::asin(std::min(1.0, std::max(-1.0, (A - m) / R)));
    const double pB = atHi ? M_PI / 2 : std::asin(std::min(1.0, std::max(-1.0, (B - m) / R)));
    const double d = pB - pA;
    for (int i = 0; i < n; ++i)
    {
      const double phi = pA + d * g.x[i];
      double cs;  // cos(phi), accurate at an end that sits on the silhouette
      if (atHi && g.x[i] > 0.5)
        cs = std::sin(d * g.x[n - 1 - i] + (M_PI / 2 - pB));
      else if (atLo && g.x[i] <= 0.5)
        cs = std::sin(d * g.x[i] + (pA + M_PI / 2));
      else
        cs = std::cos(phi);
      const double wp = wOuter * g.w[i] * d;
      emit(m + R * std::sin(phi), wp * R * cs, wp / sa);
    }
    return;
  }
  // al > 0: the run lies outside [y1, y2]
  const double sa = std::sqrt(al);
  const bool lower = 0.5 * (A + B) < m;
  const double ya = lower ? m - B : A - m, yb = lower ? m - A : B - m;  // distances from the centre, ya < yb
  const bool atRoot = ya <= R + tol;
  const double pA = atRoot ? 0.0 : std::acosh(std::max(1.0, ya / R)), pB = std::acosh(std::max(1.0, yb / R));
  // y - m = R cosh(psi) grows like e^psi: when the two silhouettes are close (R small) the run is long in psi and the
  // integrand is a polynomial in e^psi, so integrate in pieces of length <= 1 (each analytic and mild)
  const int pieces = std::max(1, (int)std::ceil(pB - pA));
  const double d = (pB - pA) / pieces;
  for (int pc = 0; pc < pieces; ++pc)
    for (int i = 0; i < n; ++i)
    {
      const double psi = pA + d * (pc + g.x[i]);
      const bool nearRoot = atRoot && pc == 0 && g.x[i] <= 0.5;
      const double sh = nearRoot ? std::sinh(d * g.x[i]) : std::sinh(psi);
      const double wp = wOuter * g.w[i] * d, ch = std::cosh(psi);
      emit(lower ? m - R * ch : m + R * ch, wp * R * sh, wp / sa);
    }
}

// Inner structure of one outer line. The indicator of {q > 0} on [0, L], q = a2 x^2 + a1 x + a0, integrates any x^m as
//   sum over roots r in (0,L) of s_r r^(m+1)/(m+1)  +  [q(L) > 0] L^(m+1)/(m+1)       (the x = 0 end contributes nothing)
// with s_r = +1 where q goes from positive to negative at r. Each term (term 0: the smaller root, 1: the larger root,
// 2: the end x = L) is smooth along the outer line except where its own validity changes, so it is integrated over its own
// maximal runs of validity (integrateRootRun for the roots, the smoothstep map for the end term). Splitting at the
// breakpoints of the *other* branches would leave runs whose integrand is singular just outside an end.
// coeffs(y, a1, a0, L) gives the inner polynomial at outer point y; disc = (al, be, ga) is the discriminant a1^2 - 4 a2 a0
// as a quadratic in y; emit(term, y, w, wOverD, r, s, a1, a0, L) receives the weight (times wOuter), w / |dq/dx(r)| for
// the root terms, and for term 2 r = L, s = 1.
template <class Coeffs, class Emit>
inline void integrateTerms(const std::vector<double> &segs, const GL &g, double a2, double eps, bool withL, double wOuter,
                           double al, double be, double ga, Coeffs &&coeffs, Emit &&emit)
{
  struct Seg
  {
    double a, b;
    bool v[3];
  };
  std::vector<Seg> sg;
  for (std::size_t k = 0; k + 1 < segs.size(); ++k)
  {
    Seg q;
    q.a = segs[k];
    q.b = segs[k + 1];
    if (q.b - q.a < 1e-14)
      continue;
    double a1, a0, L, r[2];
    coeffs(0.5 * (q.a + q.b), a1, a0, L);
    const int n = quadRoots(a2, a1, a0, eps, r);
    q.v[0] = n >= 1 && r[0] > 0.0 && r[0] < L;
    q.v[1] = n == 2 && r[1] > 0.0 && r[1] < L;
    q.v[2] = withL && (a2 * L + a1) * L + a0 > 0.0;
    sg.push_back(q);
  }
  for (int term = 0; term < (withL ? 3 : 2); ++term)
  {
    bool open = false;
    double start = 0.0, end = 0.0;
    auto point = [&](double y, double w, double wOverRoot) {
      double a1, a0, L, r[2];
      coeffs(y, a1, a0, L);
      if (L <= eps)
        return;
      const int n = quadRoots(a2, a1, a0, eps, r);
      if (term >= n)
        return;
      double s, wod = wOverRoot;
      if (std::fabs(a2) <= eps)
      {
        s = a1 > 0.0 ? -1.0 : 1.0;
        wod = w / std::fabs(a1);
      }
      else if (n == 2)
        s = (a2 > 0.0) == (term == 0) ? 1.0 : -1.0;
      else
        s = 0.0;  // tangential double root: the two ends cancel
      emit(term, y, w, wod, r[term], s, a1, a0, L);
    };
    auto flush = [&]() {
      if (term == 2)
      {
        const double h = end - start;  // L^(m+1) y^b z^c is analytic: only the run ends are kinks
        for (std::size_t i = 0; i < g.x.size(); ++i)
        {
          const double y = start + h * g.x[i];
          double a1, a0, L;
          coeffs(y, a1, a0, L);
          if (L > eps)
            emit(2, y, wOuter * g.w[i] * h, 0.0, L, 1.0, a1, a0, L);
        }
      }
      else if (std::fabs(a2) <= eps)
      {
        // linear inner polynomial: the root is analytic in y wherever it is valid
        const double h = end - start;
        for (std::size_t i = 0; i < g.x.size(); ++i)
          point(start + h * g.x[i], wOuter * g.w[i] * h, 0.0);
      }
      else
        integrateRootRun(start, end, g, al, be, ga, wOuter, point);
    };
    for (const Seg &q : sg)
    {
      if (q.v[term])
      {
        if (!open)
        {
          open = true;
          start = q.a;
        }
        end = q.b;
      }
      else if (open)
      {
        flush();
        open = false;
      }
    }
    if (open)
      flush();
  }
}

// z-values at which the line p y + q z + r = 0 meets the conic alpha y^2 + (beta0 + beta1 z) y + gamma0 + gamma1 z + gamma2 z^2 = 0
// inside the outer range [0,1]. The conic families of the sweep are the curves in the outer (y,z) plane on which the
// interval structure of {u > 0} changes; where two of them cross, the outer integrand has a corner or a square-root
// singularity that Gauss-Legendre on a smooth segment cannot resolve, so the crossing z must be a segment boundary.
inline void lineConicCrossings(double p, double q, double r, double alpha, double beta0, double beta1, double gamma0,
                               double gamma1, double gamma2, double eps, double deflate, std::vector<double> &zOut)
{
  const double ls = std::max({std::fabs(p), std::fabs(q), std::fabs(r)});
  if (ls == 0.0)
    return;
  if (std::fabs(p) >= std::fabs(q) && std::fabs(p) > 1e-13 * ls)
  {
    // y = e0 + e1 z
    const double e0 = -r / p, e1 = -q / p;
    polyRoots(alpha * e0 * e0 + beta0 * e0 + gamma0, 2 * alpha * e0 * e1 + beta0 * e1 + beta1 * e0 + gamma1,
              alpha * e1 * e1 + beta1 * e1 + gamma2, 0.0, 1.0, eps, deflate, zOut);
  }
  else if (std::fabs(q) > 1e-13 * ls)
  {
    // z = e0 + e1 y: solve for y, report z
    const double e0 = -r / q, e1 = -p / q;
    std::vector<double> ys;
    polyRoots(gamma0 + gamma1 * e0 + gamma2 * e0 * e0, beta0 + beta1 * e0 + gamma1 * e1 + 2 * gamma2 * e0 * e1,
              alpha + beta1 * e1 + gamma2 * e1 * e1, -1.0, 2.0, eps, deflate, ys);
    for (double y : ys)
    {
      const double z = e0 + e1 * y;
      if (z > -eps && z < 1.0 + eps)
        zOut.push_back(z);
    }
  }
}

// Visit every term of every outer Gauss point (y, z) of the sweep, visit(term, y, z, w, wOverD, r, s, a2, a1, a0, L): w is the
// weight (segment length, Gauss weight and smoothstep Jacobian in both outer directions), the inner polynomial is
// a2 x^2 + a1 x + a0 on x in [0, L], L = 1 - y - z, and the term is as in integrateTerms.
// The outer partition is by the critical values of the level set (the x-discriminant, the two facet restrictions, the
// branch line) so that every outer integrand is smooth on a segment up to the endpoint square roots the map removes.
template <class Visitor>
inline void sweepOuter(const Coef3 &c, const Options &opt, bool withL, Visitor &&visit)
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

  std::vector<double> zcrit = {0.0, 1.0}, zlog;
  for (const Fam &F : fams)
  {
    // beta^2 - 4 alpha gamma = 0 (birth/death), gamma = 0 (exit through y = 0), value at y = 1 - z (exit through the facet)
    {
      const std::size_t before = zcrit.size();
      polyRoots(F.be[0] * F.be[0] - 4 * F.al * F.ga[0], 2 * F.be[0] * F.be[1] - 4 * F.al * F.ga[1],
                F.be[1] * F.be[1] - 4 * F.al * F.ga[2], 0.0, 1.0, eps, opt.deflateTol, zcrit);
      if (&F == &fams[0])  // turning points of the silhouette curve: logarithmic ends of the outer integrand
        zlog.insert(zlog.end(), zcrit.begin() + before, zcrit.end());
    }
    polyRoots(F.ga[0], F.ga[1], F.ga[2], 0.0, 1.0, eps, opt.deflateTol, zcrit);
    polyRoots(F.al + F.be[0] + F.ga[0], -2 * F.al + F.be[1] - F.be[0] + F.ga[1], F.al - F.be[1] + F.ga[2], 0.0, 1.0, eps,
              opt.deflateTol, zcrit);
  }
  // crossings of the curves with each other: silhouette x* = 0 (a0 = 0 and a1 = 0), root at both ends (a0 = 0, a1 + a2 L = 0)
  // and silhouette on the facet (u(L) = 0, a1 + 2 a2 L = 0)
  {
    const Fam &F1 = fams[1], &F2 = fams[2];
    lineConicCrossings(cxy, cxz, cx, F1.al, F1.be[0], F1.be[1], F1.ga[0], F1.ga[1], F1.ga[2], eps, opt.deflateTol, zcrit);
    lineConicCrossings(cxy - a2, cxz - a2, cx + a2, F1.al, F1.be[0], F1.be[1], F1.ga[0], F1.ga[1], F1.ga[2], eps,
                       opt.deflateTol, zcrit);
    lineConicCrossings(cxy - 2 * a2, cxz - 2 * a2, cx + 2 * a2, F2.al, F2.be[0], F2.be[1], F2.ga[0], F2.ga[1], F2.ga[2], eps,
                       opt.deflateTol, zcrit);
  }

  const GL &g = gauss01(opt.order);
  const std::vector<double> zsegs = segments(zcrit, 1.0, eps);
  std::vector<double> ycrit;
  // the whole inner problem at one outer value z, with outer weight wz
  auto atZ = [&](double z, double wz) {
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
    auto coeffs = [&](double y, double &a1, double &a0, double &L) {
      L = 1.0 - y - z;
      a1 = cxy * y + cxz * z + cx;
      a0 = cyy * y * y + czz * z * z + cyz * y * z + cy * y + cz * z + c1;
    };
    const Fam &S0 = fams[0];
    integrateTerms(ysegs, g, a2, eps, withL, wz, S0.al, S0.be[0] + S0.be[1] * z, S0.ga[0] + z * (S0.ga[1] + z * S0.ga[2]),
                   coeffs, [&](int term, double y, double w, double wod, double r, double s, double a1, double a0, double L) {
                     visit(term, y, z, w, wod, r, s, a2, a1, a0, L);
                   });
  };
  auto smoothRun = [&](double a, double b) {
    const double h = b - a;
    for (int zi = 0; zi < opt.order; ++zi)
    {
      double sz, jz;
      smooth(g.x[zi], sz, jz);
      atZ(a + h * sz, g.w[zi] * h * jz);
    }
  };
  auto logRun = [&](double a, double b) {
    const double h = b - a;
    for (const TSNode &tn : tanhSinh(opt.order))
    {
      const double z = tn.left ? a + h * tn.frac : b - h * tn.frac;
      if (z > a && z < b)  // nodes closer to an end than z can resolve carry negligible weight
        atZ(z, tn.w * h);
    }
  };
  auto isLog = [&](double z) {
    for (double q : zlog)
      if (std::fabs(q - z) <= 1e-9)
        return true;
    return false;
  };
  for (std::size_t zs = 0; zs + 1 < zsegs.size(); ++zs)
  {
    const double za = zsegs[zs], zb = zsegs[zs + 1];
    if (zb - za < 1e-14)
      continue;
    const bool ll = isLog(za), lr = isLog(zb);
    if (!ll && !lr)
      smoothRun(za, zb);
    else
    {
      // split so that each half has at most one special end; the middle is a regular point where clustering is harmless
      const double c = 0.5 * (za + zb);
      ll ? logRun(za, c) : smoothRun(za, c);
      lr ? logRun(c, zb) : smoothRun(c, zb);
    }
  }
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
  std::vector<double> yp(degree + 1), zp(degree + 1), rp(degree + 2);
  detail::sweepOuter(c, opt, true, [&](int, double y, double z, double w, double, double r, double s, double, double, double, double) {
    if (s == 0.0)
      return;
    yp[0] = zp[0] = rp[0] = 1.0;
    for (int k = 1; k <= degree; ++k)
    {
      yp[k] = yp[k - 1] * y;
      zp[k] = zp[k - 1] * z;
    }
    for (int k = 1; k <= degree + 1; ++k)
      rp[k] = rp[k - 1] * r;
    const double ws = w * s;
    for (std::size_t n = 0; n < idx.size(); ++n)
    {
      const int a = idx[n][0];
      m[n] += ws * rp[a + 1] / (a + 1) * yp[idx[n][1]] * zp[idx[n][2]];
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
  std::vector<double> tp(degree + 1), rp(degree + 2);
  auto coeffs = [&](double t, double &a1, double &a0, double &L) {
    L = 1.0 - t;
    a1 = B * t + D;
    a0 = (C * t + E) * t + F;
  };
  detail::integrateTerms(segs, g, A, eps, true, 1.0, B * B - 4 * A * C, 2 * B * D - 4 * A * E, D * D - 4 * A * F, coeffs,
                         [&](int, double t, double w, double, double r, double sg, double, double, double) {
                           if (sg == 0.0)
                             return;
                           tp[0] = rp[0] = 1.0;
                           for (int k = 1; k <= degree; ++k)
                             tp[k] = tp[k - 1] * t;
                           for (int k = 1; k <= degree + 1; ++k)
                             rp[k] = rp[k - 1] * r;
                           for (std::size_t n = 0; n < idx.size(); ++n)
                             m[n] += w * sg * rp[idx[n][0] + 1] / (idx[n][0] + 1) * tp[idx[n][1]];
                         });
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


// ------------------------------------------------------------------------------------------------ surface rule
// A point rule on Gamma cap K = {u = 0} with positive weights and the discrete normal grad u / |grad u| (pointing into
// {u > 0}) at every point, from the same slicing as the volume sweep. With the inner coordinate x' along a chosen axis,
// Gamma cap K is a graph over the two outer coordinates: each root of a2 x'^2 + a1 x' + a0 is a point, with the weight
//   dS = |grad u| / |du/dx'| dy dz.
// 1/|du/dx'| blows up like an inverse square root at the x'-discriminant (the silhouette); the smoothstep map in the
// sweep regularises it. A surface with du/dx' == 0 has no x'-roots at all, so the axis is chosen as the coordinate with the
// largest mean |du/dx_i| over the element (the reference tet is symmetric under permutation of the coordinates).
struct SurfaceRule
{
  std::vector<std::array<double, 3>> x;  // points on Gamma, reference coordinates
  std::vector<double> w;                 // weights (reference measure, or physical measure when a Jacobian was given)
  std::vector<std::array<double, 3>> n;  // unit normals (reference, or physical when a Jacobian was given)
};

// coordinate with the largest mean |du/dx_i| over the ten P2 nodes
inline int chooseAxis(const Coef3 &c)
{
  static const double nodes[10][3] = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {.5, 0, 0},
                                      {.5, .5, 0}, {0, .5, .5}, {0, .5, 0}, {.5, 0, .5}, {0, 0, .5}};
  double sum[3] = {0, 0, 0};
  for (const auto &p : nodes)
  {
    sum[0] += std::fabs(c[1] + 2 * c[4] * p[0] + c[5] * p[1] + c[6] * p[2]);
    sum[1] += std::fabs(c[2] + c[5] * p[0] + 2 * c[7] * p[1] + c[8] * p[2]);
    sum[2] += std::fabs(c[3] + c[6] * p[0] + c[8] * p[1] + 2 * c[9] * p[2]);
  }
  return (sum[0] >= sum[1] && sum[0] >= sum[2]) ? 0 : (sum[1] >= sum[2] ? 1 : 2);
}

namespace detail
{
// u'(x') = u(x) with x_{perm[k]} = x'_k
inline Coef3 permuteQuadric(const Coef3 &c, const int perm[3])
{
  int inv[3];
  for (int k = 0; k < 3; ++k)
    inv[perm[k]] = k;
  // symmetric storage: pair (i,j), i <= j -> index
  static const int pairIdx[3][3] = {{4, 5, 6}, {5, 7, 8}, {6, 8, 9}};
  Coef3 r{};
  r[0] = c[0];
  for (int i = 0; i < 3; ++i)
    r[1 + inv[i]] = c[1 + i];
  for (int i = 0; i < 3; ++i)
    for (int j = i; j < 3; ++j)
    {
      const int a = std::min(inv[i], inv[j]), b = std::max(inv[i], inv[j]);
      r[pairIdx[a][b]] += c[pairIdx[i][j]];
    }
  return r;
}

inline void invert3(const double J[9], double Ji[9], double &det)
{
  det = J[0] * (J[4] * J[8] - J[5] * J[7]) - J[1] * (J[3] * J[8] - J[5] * J[6]) + J[2] * (J[3] * J[7] - J[4] * J[6]);
  Ji[0] = (J[4] * J[8] - J[5] * J[7]) / det;
  Ji[1] = (J[2] * J[7] - J[1] * J[8]) / det;
  Ji[2] = (J[1] * J[5] - J[2] * J[4]) / det;
  Ji[3] = (J[5] * J[6] - J[3] * J[8]) / det;
  Ji[4] = (J[0] * J[8] - J[2] * J[6]) / det;
  Ji[5] = (J[2] * J[3] - J[0] * J[5]) / det;
  Ji[6] = (J[3] * J[7] - J[4] * J[6]) / det;
  Ji[7] = (J[1] * J[6] - J[0] * J[7]) / det;
  Ji[8] = (J[0] * J[4] - J[1] * J[3]) / det;
}

// reference point, reference weight and the reference gradient of u there -> stored point, weight, normal; with a
// Jacobian J (x = x0 + J xi, row-major) the weight is w |det J| |J^{-T} n| and the normal J^{-T} grad / |J^{-T} grad|
struct Scaling
{
  bool physical = false;
  double Ji[9], det = 1.0;
  void add(SurfaceRule &out, const double xr[3], double wRef, const double g[3]) const
  {
    const double ng = std::sqrt(g[0] * g[0] + g[1] * g[1] + g[2] * g[2]);
    if (ng == 0.0)
      return;
    double nn[3] = {g[0] / ng, g[1] / ng, g[2] / ng};
    double w = wRef;
    if (physical)
    {
      double gp[3];
      for (int i = 0; i < 3; ++i)
        gp[i] = Ji[0 * 3 + i] * g[0] + Ji[1 * 3 + i] * g[1] + Ji[2 * 3 + i] * g[2];
      const double ngp = std::sqrt(gp[0] * gp[0] + gp[1] * gp[1] + gp[2] * gp[2]);
      w = wRef * std::fabs(det) * ngp / ng;
      for (int i = 0; i < 3; ++i)
        nn[i] = gp[i] / ngp;
    }
    out.x.push_back({xr[0], xr[1], xr[2]});
    out.w.push_back(w);
    out.n.push_back({nn[0], nn[1], nn[2]});
  }
};
}  // namespace detail

// axis < 0: choose automatically. J (row-major 3x3, J[i][j] = dx_i/dxi_j) or nullptr for reference measure.
inline SurfaceRule surfaceRule(const Coef3 &c, const Options &opt = Options(), int axis = -1, const double *J = nullptr)
{
  SurfaceRule out;
  if (detail::maxAbs(c) == 0.0)
    return out;
  detail::Scaling sc;
  if (J)
  {
    sc.physical = true;
    detail::invert3(J, sc.Ji, sc.det);
  }
  if (detail::isPlanar(c, opt.planarTol))
  {
    // the cut is a plane: triangles from cutclip, a collapsed Gauss rule on each (positive weights, exact to degree 7)
    const double gn = std::sqrt(c[1] * c[1] + c[2] * c[2] + c[3] * c[3]);
    if (gn == 0.0)
      return out;
    const auto tris = cutclip::cutSurface(detail::refTet(), detail::planeOf(c), std::vector<cutclip::Plane>{});
    const int ng = std::min(std::max(opt.order, 4), 8);
    const detail::GL &g = detail::gauss01(ng);
    const double grad[3] = {c[1], c[2], c[3]};
    for (const cutclip::Tri &t : tris)
    {
      const double area2 = 2.0 * cutclip::triArea(t);
      for (int i = 0; i < ng; ++i)
        for (int j = 0; j < ng; ++j)
        {
          const double u1 = g.x[i], v1 = g.x[j];
          const double l1 = u1, l2 = v1 * (1.0 - u1), l0 = 1.0 - l1 - l2;
          double x[3];
          for (int k = 0; k < 3; ++k)
            x[k] = l0 * t[0][k] + l1 * t[1][k] + l2 * t[2][k];
          sc.add(out, x, g.w[i] * g.w[j] * (1.0 - u1) * area2, grad);
        }
    }
    return out;
  }
  if (axis < 0)
    axis = chooseAxis(c);
  const int perm[3] = {axis, (axis + 1) % 3, (axis + 2) % 3};
  const Coef3 cp = detail::permuteQuadric(c, perm);
  detail::sweepOuter(cp, opt, false, [&](int, double y, double z, double, double wod, double r, double, double, double, double, double) {
    double x[3];
    x[perm[0]] = r;
    x[perm[1]] = y;
    x[perm[2]] = z;
    const double g[3] = {c[1] + 2 * c[4] * x[0] + c[5] * x[1] + c[6] * x[2], c[2] + c[5] * x[0] + 2 * c[7] * x[1] + c[8] * x[2],
                         c[3] + c[6] * x[0] + c[8] * x[1] + 2 * c[9] * x[2]};
    const double ng = std::sqrt(g[0] * g[0] + g[1] * g[1] + g[2] * g[2]);
    sc.add(out, x, wod * ng, g);  // wod = w / |du/dx'|
  });
  return out;
}
inline SurfaceRule surfaceRuleNodal(const P2Tet &u, const Options &opt = Options(), int axis = -1, const double *J = nullptr)
{
  return surfaceRule(p2ToMonomial(u), opt, axis, J);
}

// Scalar and normal-weighted monomial moments of the rule, in reference monomials (cutclip order):
//   S[a]    = sum_q w_q x_q^a               = int_Gamma x^a dS            (Dirac fit, penalty, Robin, data on Gamma)
//   Sn[i][a] = sum_q w_q n_i,q x_q^a        = int_Gamma x^a n_i dS        (consistency terms through three Dirac fits)
inline void surfaceMoments(const SurfaceRule &r, int degree, std::vector<double> &S, std::vector<double> Sn[3])
{
  if (degree < 0 || degree > 7)
    throw std::invalid_argument("cutquadric: moments are supported to degree 7 (cutclip's limit)");
  const auto idx = monomialIndex(degree);
  S.assign(idx.size(), 0.0);
  for (int i = 0; i < 3; ++i)
    Sn[i].assign(idx.size(), 0.0);
  std::vector<double> px(degree + 1), py(degree + 1), pz(degree + 1);
  for (std::size_t q = 0; q < r.w.size(); ++q)
  {
    px[0] = py[0] = pz[0] = 1.0;
    for (int k = 1; k <= degree; ++k)
    {
      px[k] = px[k - 1] * r.x[q][0];
      py[k] = py[k - 1] * r.x[q][1];
      pz[k] = pz[k - 1] * r.x[q][2];
    }
    for (std::size_t n = 0; n < idx.size(); ++n)
    {
      const double mono = r.w[q] * px[idx[n][0]] * py[idx[n][1]] * pz[idx[n][2]];
      S[n] += mono;
      for (int i = 0; i < 3; ++i)
        Sn[i][n] += mono * r.n[q][i];
    }
  }
}


// ------------------------------------------------------------------------------------------------ the fits
// Equivalent polynomials of degree k from the moments (the curved-cut counterpart of equivalent_polynomials.h):
//   H   in P_k(K):  int_K H q  = int_{K cap {u>0}} q          for every q in P_k      (so ImH = 1 - H)
//   D   in P_k(K):  int_K D q  = int_Gamma q dS / |J|
//   Dn_i in P_k(K): int_K Dn_i q = int_Gamma q n_i dS / |J|   (n the unit normal into {u>0}, physical when J is given)
// The element kernel multiplies each quadrature weight by |J| and by H, ImH, D or Dn_i at the point, so dividing the surface
// moments by |det J| here makes the Dirac fits integrate to the physical surface measure. A flux term
// int_Gamma (a grad v.n) w dS = sum_q |J| w_q sum_I Dn_I(x_q) (a d_I v w)(x_q) then needs no normal off the surface and is
// exact for polynomial integrands of degree <= k.
namespace detail
{
// Cholesky factor and Gram matrix G_ab = int_K x^(a+b) = (a+b)! / (|a+b|+3)! componentwise, monomial order
struct Gram
{
  int n = 0;
  std::vector<double> G, L;
};
inline const Gram &gram(int k)
{
  static std::map<int, Gram> cache;
  static std::mutex mtx;
  std::lock_guard<std::mutex> lock(mtx);
  auto it = cache.find(k);
  if (it != cache.end())
    return it->second;
  const auto idx = monomialIndex(k);
  Gram g;
  g.n = (int)idx.size();
  g.G.assign(g.n * g.n, 0.0);
  auto fact = [](int m) {
    double f = 1.0;
    for (int i = 2; i <= m; ++i)
      f *= i;
    return f;
  };
  for (int a = 0; a < g.n; ++a)
    for (int b = 0; b < g.n; ++b)
    {
      const int p = idx[a][0] + idx[b][0], q = idx[a][1] + idx[b][1], r = idx[a][2] + idx[b][2];
      g.G[a * g.n + b] = fact(p) * fact(q) * fact(r) / fact(p + q + r + 3);
    }
  g.L.assign(g.n * g.n, 0.0);
  for (int i = 0; i < g.n; ++i)
    for (int j = 0; j <= i; ++j)
    {
      double sum = g.G[i * g.n + j];
      for (int m = 0; m < j; ++m)
        sum -= g.L[i * g.n + m] * g.L[j * g.n + m];
      if (i == j)
      {
        if (sum <= 0.0)
          throw std::runtime_error("cutquadric: Gram matrix is not positive definite");
        g.L[i * g.n + i] = std::sqrt(sum);
      }
      else
        g.L[i * g.n + j] = sum / g.L[j * g.n + j];
    }
  return cache.emplace(k, std::move(g)).first->second;
}
// solve G c = m (Cholesky, then one step of iterative refinement: the monomial Gram matrix is ill-conditioned)
inline std::vector<double> gramSolve(int k, const std::vector<double> &m)
{
  const Gram &g = gram(k);
  const int n = g.n;
  auto solve = [&](const std::vector<double> &rhs) {
    std::vector<double> y(n), x(n);
    for (int i = 0; i < n; ++i)
    {
      double s = rhs[i];
      for (int j = 0; j < i; ++j)
        s -= g.L[i * n + j] * y[j];
      y[i] = s / g.L[i * n + i];
    }
    for (int i = n - 1; i >= 0; --i)
    {
      double s = y[i];
      for (int j = i + 1; j < n; ++j)
        s -= g.L[j * n + i] * x[j];
      x[i] = s / g.L[i * n + i];
    }
    return x;
  };
  std::vector<double> c = solve(m);
  std::vector<double> res(n);
  for (int i = 0; i < n; ++i)
  {
    double s = m[i];
    for (int j = 0; j < n; ++j)
      s -= g.G[i * n + j] * c[j];
    res[i] = s;
  }
  const std::vector<double> dc = solve(res);
  for (int i = 0; i < n; ++i)
    c[i] += dc[i];
  return c;
}
}  // namespace detail

struct QuadricFit
{
  int degree = 0;
  std::vector<double> cH, cD, cDn[3];  // monomial coefficients in monomialIndex(degree) order
  // quadrature-point accessors of the kernel (set_quad, then H(), ImH(), D(), Dn(I)), as GeneralizedFunctions_mix
  std::vector<std::array<double, 3>> points;
  std::vector<double> qH, qD, qDn[3];
  unsigned int q = 0;

  // u: P2 nodal values (proteus order); J: row-major 3x3 or nullptr (identity: reference measure)
  void calculate(const P2Tet &u, int k, const Options &opt = Options(), int axis = -1, const double *J = nullptr)
  {
    degree = k;
    const Coef3 c = p2ToMonomial(u);
    cH = detail::gramSolve(k, volumeMoments(c, k, opt));
    const SurfaceRule rule = surfaceRule(c, opt, axis, J);
    std::vector<double> S, Sn[3];
    surfaceMoments(rule, k, S, Sn);
    double detJ = 1.0;
    if (J)
    {
      double Ji[9];
      detail::invert3(J, Ji, detJ);
      detJ = std::fabs(detJ);
    }
    for (double &v : S)
      v /= detJ;
    cD = detail::gramSolve(k, S);
    for (int i = 0; i < 3; ++i)
    {
      for (double &v : Sn[i])
        v /= detJ;
      cDn[i] = detail::gramSolve(k, Sn[i]);
    }
  }
  static double eval(const std::vector<double> &coef, int k, const double x[3])
  {
    const auto idx = monomialIndex(k);
    double px[8], py[8], pz[8];
    px[0] = py[0] = pz[0] = 1.0;
    for (int i = 1; i <= k; ++i)
    {
      px[i] = px[i - 1] * x[0];
      py[i] = py[i - 1] * x[1];
      pz[i] = pz[i - 1] * x[2];
    }
    double s = 0.0;
    for (std::size_t n = 0; n < idx.size(); ++n)
      s += coef[n] * px[idx[n][0]] * py[idx[n][1]] * pz[idx[n][2]];
    return s;
  }
  double evalH(const double x[3]) const { return eval(cH, degree, x); }
  double evalImH(const double x[3]) const { return 1.0 - eval(cH, degree, x); }
  double evalD(const double x[3]) const { return eval(cD, degree, x); }
  double evalDn(int I, const double x[3]) const { return eval(cDn[I], degree, x); }
  void setPoints(const std::vector<std::array<double, 3>> &pts)
  {
    points = pts;
    qH.resize(pts.size());
    qD.resize(pts.size());
    for (int i = 0; i < 3; ++i)
      qDn[i].resize(pts.size());
    for (std::size_t p = 0; p < pts.size(); ++p)
    {
      qH[p] = evalH(pts[p].data());
      qD[p] = evalD(pts[p].data());
      for (int i = 0; i < 3; ++i)
        qDn[i][p] = evalDn(i, pts[p].data());
    }
    q = 0;
  }
  void set_quad(unsigned int qq) { q = qq; }
  double H() const { return qH[q]; }
  double ImH() const { return 1.0 - qH[q]; }
  double D() const { return qD[q]; }
  double Dn(int I) const { return qDn[I][q]; }
};


// The embedded solid of one P2 tetrahedron as the ADR kernel reads it (C44 phase 1b, B2): classification from the
// exact range test (not the vertex signs), then, on a cut element, the degree-k fits of the Heaviside of {u > 0},
// the Dirac of {u = 0} and the three normal-weighted Diracs at the element's quadrature points. u > 0 is the fluid.
// The element is affine, so the Jacobian comes from its four vertices (`nodes`: 3 doubles per node, vertices first).
// An uncut element stores nothing: H is 1 or 0, D and Dn are 0.
struct SolidFit
{
  int icase = 0;                      // +1 uncut, u > 0; -1 uncut, u < 0; 0 cut
  CutLabel label = CutLabel::UncutPositive;
  int nQ = 0;
  std::vector<double> qH, qD, qDn;    // per point; qDn has 3 per point

  int calculate(const double *phi, const double *nodes, const double *xRef, int nPoints, int k, const Options &opt = Options())
  {
    P2Tet u;
    for (int i = 0; i < 10; ++i)
      u[i] = phi[i];
    const Classification cl = classify(u);
    label = cl.label;
    nQ = nPoints;
    if (cl.label == CutLabel::UncutPositive || cl.label == CutLabel::UncutNegative)
    {
      icase = (cl.label == CutLabel::UncutPositive) ? 1 : -1;
      qH.clear();
      qD.clear();
      qDn.clear();
      return icase;
    }
    icase = 0;
    double J[9];
    for (int i = 0; i < 3; ++i)
      for (int I = 0; I < 3; ++I)
        J[I * 3 + i] = nodes[(i + 1) * 3 + I] - nodes[I];
    QuadricFit fit;
    fit.calculate(u, k, opt, -1, J);
    std::vector<std::array<double, 3>> pts(nPoints);
    for (int p = 0; p < nPoints; ++p)
      pts[p] = {xRef[3 * p], xRef[3 * p + 1], xRef[3 * p + 2]};
    fit.setPoints(pts);
    qH = fit.qH;
    qD = fit.qD;
    qDn.resize(3 * nPoints);
    for (int p = 0; p < nPoints; ++p)
      for (int I = 0; I < 3; ++I)
        qDn[3 * p + I] = fit.qDn[I][p];
    return icase;
  }
  double H(int p) const { return icase == 0 ? qH[p] : (icase > 0 ? 1.0 : 0.0); }
  double D(int p) const { return icase == 0 ? qD[p] : 0.0; }
  double Dn(int p, int I) const { return icase == 0 ? qDn[3 * p + I] : 0.0; }
};
}  // namespace cutquadric
}  // namespace proteus
#endif
