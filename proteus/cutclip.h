#ifndef PROTEUS_CUTCLIP_H
#define PROTEUS_CUTCLIP_H
// Exact clipping of a tetrahedron or triangle by half-spaces, and exact polynomial moments of the clipped region.
//
// C44 (3D CutFEM-IFEM) owns this file. It is the linear (planar-cut) kernel of the cut quadrature; it is the oracle the
// equivalent-polynomial fits in equivalent_polynomials.h are checked against, and the clipped faces it produces are the
// quadrature domains for the ghost-penalty and Nitsche terms on cut triangular faces.
//
// Conventions
//   * A half-space is {x : a.x + b > 0} given as double[4] = {a0, a1, a2, b}. The kept region is where it is positive.
//   * A clipped tetrahedron is a list of tetrahedra (vertex lists) that tile the kept region; a clipped triangle in 3D is
//     a list of triangles. Degenerate (zero volume or area) pieces are dropped.
//   * moments(): integral of x^p y^q z^r over the region for every p+q+r <= degree, in the order of monomialIndex().
//     Gauss-Legendre with 5 points per collapsed direction is exact for total degree <= 7.
#include <array>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace proteus
{
namespace cutclip
{
using Vec3 = std::array<double, 3>;
using Tet = std::array<Vec3, 4>;
using Tri = std::array<Vec3, 3>;
using Plane = std::array<double, 4>;

inline double signedDistance(const Plane &h, const Vec3 &x)
{
  return h[0] * x[0] + h[1] * x[1] + h[2] * x[2] + h[3];
}

inline Vec3 crossingPoint(const Vec3 &p, double dp, const Vec3 &q, double dq)
{
  const double t = dp / (dp - dq);
  return {p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1]), p[2] + t * (q[2] - p[2])};
}

inline double tetVolume(const Tet &t)
{
  const double ax = t[1][0] - t[0][0], ay = t[1][1] - t[0][1], az = t[1][2] - t[0][2];
  const double bx = t[2][0] - t[0][0], by = t[2][1] - t[0][1], bz = t[2][2] - t[0][2];
  const double cx = t[3][0] - t[0][0], cy = t[3][1] - t[0][1], cz = t[3][2] - t[0][2];
  return (ax * (by * cz - bz * cy) - ay * (bx * cz - bz * cx) + az * (bx * cy - by * cx)) / 6.0;
}

inline double triArea(const Tri &t)
{
  const double ax = t[1][0] - t[0][0], ay = t[1][1] - t[0][1], az = t[1][2] - t[0][2];
  const double bx = t[2][0] - t[0][0], by = t[2][1] - t[0][1], bz = t[2][2] - t[0][2];
  const double nx = ay * bz - az * by, ny = az * bx - ax * bz, nz = ax * by - ay * bx;
  return 0.5 * std::sqrt(nx * nx + ny * ny + nz * nz);
}

// Clip one tetrahedron by one half-space; appends the tiling of the kept part to out. A vertex with distance exactly
// zero counts as kept, so a plane through a vertex, an edge or a face never creates sliver pieces: such a piece has
// zero volume and is dropped.
inline void clipTet(const Tet &t, const Plane &h, std::vector<Tet> &out)
{
  double d[4];
  int pos[4], neg[4], np = 0, nn = 0;
  for (int i = 0; i < 4; ++i)
  {
    d[i] = signedDistance(h, t[i]);
    if (d[i] >= 0.0)
      pos[np++] = i;
    else
      neg[nn++] = i;
  }
  auto push = [&out](const Tet &s) {
    if (std::fabs(tetVolume(s)) > 0.0)
      out.push_back(s);
  };
  if (nn == 0)
  {
    push(t);
    return;
  }
  if (np == 0)
    return;
  if (np == 1)
  {
    const int a = pos[0];
    Tet s;
    s[0] = t[a];
    for (int k = 0; k < 3; ++k)
      s[k + 1] = crossingPoint(t[a], d[a], t[neg[k]], d[neg[k]]);
    push(s);
    return;
  }
  if (np == 3)
  {
    // kept region is a triangular prism: base (a,b,c), top = crossings with the single negative vertex n
    const int a = pos[0], b = pos[1], c = pos[2], n = neg[0];
    const Vec3 pa = crossingPoint(t[a], d[a], t[n], d[n]);
    const Vec3 pb = crossingPoint(t[b], d[b], t[n], d[n]);
    const Vec3 pc = crossingPoint(t[c], d[c], t[n], d[n]);
    push({t[a], t[b], t[c], pa});
    push({t[b], t[c], pa, pb});
    push({t[c], pa, pb, pc});
    return;
  }
  // np == 2: wedge with edge (a,b) and the four crossings
  const int a = pos[0], b = pos[1], c = neg[0], e = neg[1];
  const Vec3 pac = crossingPoint(t[a], d[a], t[c], d[c]);
  const Vec3 pae = crossingPoint(t[a], d[a], t[e], d[e]);
  const Vec3 pbc = crossingPoint(t[b], d[b], t[c], d[c]);
  const Vec3 pbe = crossingPoint(t[b], d[b], t[e], d[e]);
  push({t[a], t[b], pac, pae});
  push({t[b], pac, pae, pbc});
  push({t[b], pae, pbc, pbe});
}

inline std::vector<Tet> clipTet(const Tet &t, const std::vector<Plane> &planes)
{
  std::vector<Tet> cur{t}, next;
  for (const Plane &h : planes)
  {
    next.clear();
    for (const Tet &s : cur)
      clipTet(s, h, next);
    cur.swap(next);
  }
  return cur;
}

// Clip a triangle (in 3D) by one half-space; appends the kept part, triangulated, to out.
inline void clipTri(const Tri &t, const Plane &h, std::vector<Tri> &out)
{
  double d[3];
  int pos[3], neg[3], np = 0, nn = 0;
  for (int i = 0; i < 3; ++i)
  {
    d[i] = signedDistance(h, t[i]);
    if (d[i] >= 0.0)
      pos[np++] = i;
    else
      neg[nn++] = i;
  }
  auto push = [&out](const Tri &s) {
    if (triArea(s) > 0.0)
      out.push_back(s);
  };
  if (nn == 0)
  {
    push(t);
    return;
  }
  if (np == 0)
    return;
  if (np == 1)
  {
    const int a = pos[0];
    push({t[a], crossingPoint(t[a], d[a], t[neg[0]], d[neg[0]]), crossingPoint(t[a], d[a], t[neg[1]], d[neg[1]])});
    return;
  }
  const int a = pos[0], b = pos[1], c = neg[0];
  const Vec3 pa = crossingPoint(t[a], d[a], t[c], d[c]);
  const Vec3 pb = crossingPoint(t[b], d[b], t[c], d[c]);
  push({t[a], t[b], pa});
  push({t[b], pb, pa});
}

inline std::vector<Tri> clipTri(const Tri &t, const std::vector<Plane> &planes)
{
  std::vector<Tri> cur{t}, next;
  for (const Plane &h : planes)
  {
    next.clear();
    for (const Tri &s : cur)
      clipTri(s, h, next);
    cur.swap(next);
  }
  return cur;
}

// The portion of the plane {h = 0} inside tet t and inside every plane of `others`, triangulated: the cut surface of
// the element. A plane that does not meet the tet gives an empty list.
inline std::vector<Tri> cutSurface(const Tet &t, const Plane &h, const std::vector<Plane> &others)
{
  // same classification as clipTet: distance >= 0 is "positive", so a plane through a vertex gives that vertex
  double d[4];
  int pos[4], neg[4], np = 0, nn = 0;
  for (int i = 0; i < 4; ++i)
  {
    d[i] = signedDistance(h, t[i]);
    if (d[i] >= 0.0)
      pos[np++] = i;
    else
      neg[nn++] = i;
  }
  std::vector<Tri> tris;
  auto X = [&](int p, int n) { return crossingPoint(t[p], d[p], t[n], d[n]); };
  if (np == 1)
    tris.push_back({X(pos[0], neg[0]), X(pos[0], neg[1]), X(pos[0], neg[2])});
  else if (np == 3)
    tris.push_back({X(pos[0], neg[0]), X(pos[1], neg[0]), X(pos[2], neg[0])});
  else if (np == 2)
  {
    // quadrilateral a-c, a-e, b-e, b-c split along the diagonal (a-c, b-e)
    const Vec3 ac = X(pos[0], neg[0]), ae = X(pos[0], neg[1]), be = X(pos[1], neg[1]), bc = X(pos[1], neg[0]);
    tris.push_back({ac, ae, be});
    tris.push_back({ac, be, bc});
  }
  std::vector<Tri> cur, next;
  for (const Tri &s : tris)
    if (triArea(s) > 0.0)
      cur.push_back(s);
  for (const Plane &o : others)
  {
    next.clear();
    for (const Tri &s : cur)
      clipTri(s, o, next);
    cur.swap(next);
  }
  return cur;
}

inline int numMonomials(int degree) { return (degree + 1) * (degree + 2) * (degree + 3) / 6; }

// Order of the monomials x^p y^q z^r in moments(): total degree ascending, then p descending, then q descending.
inline std::vector<std::array<int, 3>> monomialIndex(int degree)
{
  std::vector<std::array<int, 3>> idx;
  for (int n = 0; n <= degree; ++n)
    for (int p = n; p >= 0; --p)
      for (int q = n - p; q >= 0; --q)
        idx.push_back({p, q, n - p - q});
  return idx;
}

namespace detail
{
// 5-point Gauss-Legendre on [0,1]
static const double gx[5] = {0.04691007703066802, 0.23076534494715845, 0.5, 0.7692346550528415, 0.9530899229693319};
static const double gw[5] = {0.11846344252809454, 0.2393143352496832, 0.28444444444444444, 0.2393143352496832,
                             0.11846344252809454};
inline double ipow(double x, int n)
{
  double r = 1.0;
  for (int i = 0; i < n; ++i)
    r *= x;
  return r;
}
}  // namespace detail

inline void addMoments(const Tet &t, int degree, const std::vector<std::array<int, 3>> &idx, std::vector<double> &m)
{
  if (degree > 7)
    throw std::invalid_argument("cutclip: moments are exact to degree 7");
  const double vol6 = std::fabs(tetVolume(t)) * 6.0;
  for (int i = 0; i < 5; ++i)
    for (int j = 0; j < 5; ++j)
      for (int k = 0; k < 5; ++k)
      {
        // collapsed coordinates: barycentric l1 = u, l2 = v(1-u), l3 = w(1-u)(1-v); Jacobian (1-u)^2 (1-v) (volume 1/6 ref)
        const double u = detail::gx[i], v = detail::gx[j], w = detail::gx[k];
        const double l1 = u, l2 = v * (1.0 - u), l3 = w * (1.0 - u) * (1.0 - v);
        const double l0 = 1.0 - l1 - l2 - l3;
        const double wt = detail::gw[i] * detail::gw[j] * detail::gw[k] * (1.0 - u) * (1.0 - u) * (1.0 - v) * vol6;
        double x[3];
        for (int c = 0; c < 3; ++c)
          x[c] = l0 * t[0][c] + l1 * t[1][c] + l2 * t[2][c] + l3 * t[3][c];
        for (std::size_t n = 0; n < idx.size(); ++n)
          m[n] += wt * detail::ipow(x[0], idx[n][0]) * detail::ipow(x[1], idx[n][1]) * detail::ipow(x[2], idx[n][2]);
      }
}

inline void addMoments(const Tri &t, int degree, const std::vector<std::array<int, 3>> &idx, std::vector<double> &m)
{
  if (degree > 7)
    throw std::invalid_argument("cutclip: moments are exact to degree 7");
  const double area2 = 2.0 * triArea(t);
  for (int i = 0; i < 5; ++i)
    for (int j = 0; j < 5; ++j)
    {
      const double u = detail::gx[i], v = detail::gx[j];
      const double l1 = u, l2 = v * (1.0 - u), l0 = 1.0 - l1 - l2;
      const double wt = detail::gw[i] * detail::gw[j] * (1.0 - u) * area2;
      double x[3];
      for (int c = 0; c < 3; ++c)
        x[c] = l0 * t[0][c] + l1 * t[1][c] + l2 * t[2][c];
      for (std::size_t n = 0; n < idx.size(); ++n)
        m[n] += wt * detail::ipow(x[0], idx[n][0]) * detail::ipow(x[1], idx[n][1]) * detail::ipow(x[2], idx[n][2]);
    }
}

template <class Piece>
inline std::vector<double> moments(const std::vector<Piece> &pieces, int degree)
{
  const auto idx = monomialIndex(degree);
  std::vector<double> m(idx.size(), 0.0);
  for (const Piece &p : pieces)
    addMoments(p, degree, idx, m);
  return m;
}
}  // namespace cutclip
}  // namespace proteus
#endif
