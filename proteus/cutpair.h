#ifndef PROTEUS_CUTPAIR_H
#define PROTEUS_CUTPAIR_H
// A tetrahedron cut by two P1 level sets: the sub-polytopes, their moment-exact fits, the split Dirac fits of each cut
// surface, the triple line, and the same on a triangular face cut by two lines.
//
// C44 (3D CutFEM-IFEM) owns this file. It is the geometry of the "triply cut" element in 3D (phase 3, part A): the
// counterpart of equivalent_polynomials::composite (2D). It adds nothing to cutclip.h or cutquadric.h; it calls
// cutclip for the clipping and the moments and cutquadric for the Gram solve. No kernel reads it yet.
//
// Conventions (as in ADR.h and the 2D composite fits)
//   * Reference tetrahedron (0,0,0), (1,0,0), (0,1,0), (0,0,1); a level set is its four vertex values, so every cut
//     is a plane inside the element.
//   * phi_s is the embedded solid: phi_s > 0 is the active fluid, phi_s < 0 the masked solid.
//     phi_f is the fluid interface: phi_f < 0 is Omega^- (the mua side), phi_f > 0 is Omega^+ (the mub side).
//   * Regions:  A = {phi_s > 0, phi_f < 0},  B = {phi_s > 0, phi_f > 0},  S = {phi_s < 0}.  A + B + S tile K.
//   * Surfaces: Gamma_s = {phi_s = 0} split by the fluid that borders it (GsA, GsB);
//               Gamma_f = {phi_f = 0} split into the part in the fluid (GfActive) and the part in the solid (GfMasked).
//   * Triple line: L = Gamma_s cap Gamma_f cap K, a segment whose two ends lie on faces of K. On each such face the
//     end is the triple point of that face's own two cut lines.
//   * A vertex value of exactly zero counts as positive (as in cutclip): pieces of zero measure are dropped, so a cut
//     through a vertex, an edge or a face needs no special case.
//   * Fits: for a region R, c_R in P_k(K) with  int_K c_R q = int_R q  for every q in P_k, monomial coefficients in
//     cutclip::monomialIndex(k) order. For a surface piece G,  int_K c_G q = int_G q dS / |det J|  with dS the physical
//     area when the Jacobian is given, so the kernel's |det J| w_q c_G(x_q) integrates over the physical piece. The
//     cuts are planar and the map affine, so the physical unit normal of each surface is one constant vector
//     (normalS, normalF, pointing into {phi > 0}); a normal-weighted Dirac is that vector times the fit.
//     The same for the triple line with arc length in place of area (cL).
//   * Exactness: cutclip::moments is exact to degree 7, so k <= 7.
#include <array>
#include <cmath>
#include <stdexcept>
#include <vector>
#include "cutclip.h"
#include "cutquadric.h"

namespace proteus
{
namespace cutpair
{
using cutclip::Plane;
using cutclip::Tet;
using cutclip::Tri;
using cutclip::Vec3;
using P1Tet = std::array<double, 4>;
using P1Tri = std::array<double, 3>;

inline Tet refTet() { return {{{0., 0., 0.}, {1., 0., 0.}, {0., 1., 0.}, {0., 0., 1.}}}; }

// {phi > 0} on the reference tetrahedron as a cutclip half-space
inline Plane planeOf(const P1Tet &phi) { return {phi[1] - phi[0], phi[2] - phi[0], phi[3] - phi[0], phi[0]}; }
inline Plane negated(const Plane &h) { return {-h[0], -h[1], -h[2], -h[3]}; }

// +1: every vertex value >= 0; -1: every vertex value < 0; 0: cut. The rule of cutclip::clipTet.
template <std::size_t N>
inline int side(const std::array<double, N> &phi)
{
  std::size_t np = 0;
  for (double v : phi)
    if (v >= 0.0)
      ++np;
  return np == N ? 1 : (np == 0 ? -1 : 0);
}

// ---------------------------------------------------------------------------------------------------------- regions
struct Regions
{
  std::vector<Tet> A, B, S;              // tilings of the three regions by tetrahedra
  std::vector<Tri> GsA, GsB;             // Gamma_s bordering A, bordering B
  std::vector<Tri> GfActive, GfMasked;   // Gamma_f inside {phi_s > 0}, inside {phi_s < 0}
};

inline Regions regions(const P1Tet &phi_s, const P1Tet &phi_f)
{
  const Tet K = refTet();
  const Plane hs = planeOf(phi_s), hf = planeOf(phi_f);
  const Plane ms = negated(hs), mf = negated(hf);
  Regions r;
  r.A = cutclip::clipTet(K, std::vector<Plane>{hs, mf});
  r.B = cutclip::clipTet(K, std::vector<Plane>{hs, hf});
  r.S = cutclip::clipTet(K, std::vector<Plane>{ms});
  r.GsA = cutclip::cutSurface(K, hs, std::vector<Plane>{mf});
  r.GsB = cutclip::cutSurface(K, hs, std::vector<Plane>{hf});
  r.GfActive = cutclip::cutSurface(K, hf, std::vector<Plane>{hs});
  r.GfMasked = cutclip::cutSurface(K, hf, std::vector<Plane>{ms});
  return r;
}

// ------------------------------------------------------------------------------------------------------ triple line
struct TripleLine
{
  bool exists = false;
  Vec3 x0{{0., 0., 0.}}, x1{{0., 0., 0.}};  // end points (reference coordinates), x1 - x0 along grad phi_s x grad phi_f
  int face0 = -1, face1 = -1;               // the face of K (opposite vertex face_i) each end lies on
  double length = 0.0;                      // reference length
  Vec3 tangent{{0., 0., 0.}};               // reference unit tangent, (x1 - x0) / length
};

// L = {phi_s = 0} cap {phi_f = 0} cap K. Parallel or coincident planes, and a line that misses K or touches it in one
// point, give exists = false. The line is x(tau) = p + tau t with t = grad phi_s x grad phi_f; each barycentric
// coordinate is linear in tau, so K cap L is the interval where all four are >= 0.
inline TripleLine tripleLine(const P1Tet &phi_s, const P1Tet &phi_f)
{
  TripleLine L;
  const Plane hs = planeOf(phi_s), hf = planeOf(phi_f);
  const double gs[3] = {hs[0], hs[1], hs[2]}, gf[3] = {hf[0], hf[1], hf[2]};
  const double t[3] = {gs[1] * gf[2] - gs[2] * gf[1], gs[2] * gf[0] - gs[0] * gf[2], gs[0] * gf[1] - gs[1] * gf[0]};
  const double tt = t[0] * t[0] + t[1] * t[1] + t[2] * t[2];
  const double ss = gs[0] * gs[0] + gs[1] * gs[1] + gs[2] * gs[2], ff = gf[0] * gf[0] + gf[1] * gf[1] + gf[2] * gf[2];
  if (!(tt > 1.0e-28 * ss * ff))
    return L;  // parallel (sine of the angle below 1e-14) or a constant level set
  // the point of the line closest to the origin: p = (-b_s (g_f x t) - b_f (t x g_s)) / |t|^2
  const double fxt[3] = {gf[1] * t[2] - gf[2] * t[1], gf[2] * t[0] - gf[0] * t[2], gf[0] * t[1] - gf[1] * t[0]};
  const double txs[3] = {t[1] * gs[2] - t[2] * gs[1], t[2] * gs[0] - t[0] * gs[2], t[0] * gs[1] - t[1] * gs[0]};
  double p[3];
  for (int c = 0; c < 3; ++c)
    p[c] = (-hs[3] * fxt[c] - hf[3] * txs[c]) / tt;
  // barycentric coordinates lambda_i = a_i + tau b_i; face i is {lambda_i = 0}
  const double a[4] = {1.0 - p[0] - p[1] - p[2], p[0], p[1], p[2]};
  const double b[4] = {-t[0] - t[1] - t[2], t[0], t[1], t[2]};
  double lo = -1.0e300, hi = 1.0e300;
  int flo = -1, fhi = -1;
  for (int i = 0; i < 4; ++i)
  {
    if (b[i] > 0.0)
    {
      const double tau = -a[i] / b[i];
      if (tau > lo)
      {
        lo = tau;
        flo = i;
      }
    }
    else if (b[i] < 0.0)
    {
      const double tau = -a[i] / b[i];
      if (tau < hi)
      {
        hi = tau;
        fhi = i;
      }
    }
    else if (a[i] < 0.0)
      return L;  // parallel to face i and outside it
  }
  if (!(hi > lo))
    return L;
  L.exists = true;
  L.face0 = flo;
  L.face1 = fhi;
  for (int c = 0; c < 3; ++c)
  {
    L.x0[c] = p[c] + lo * t[c];
    L.x1[c] = p[c] + hi * t[c];
  }
  const double nt = std::sqrt(tt);
  L.length = (hi - lo) * nt;
  for (int c = 0; c < 3; ++c)
    L.tangent[c] = t[c] / nt;
  return L;
}

// int_L x^p y^q z^r dl (reference arc length), cutclip::monomialIndex(degree) order; 5-point Gauss, exact to degree 9
inline std::vector<double> lineMoments(const TripleLine &L, int degree)
{
  if (degree > 9)
    throw std::invalid_argument("cutpair: line moments are exact to degree 9");
  const auto idx = cutclip::monomialIndex(degree);
  std::vector<double> m(idx.size(), 0.0);
  if (!L.exists)
    return m;
  for (int g = 0; g < 5; ++g)
  {
    const double u = cutclip::detail::gx[g], w = cutclip::detail::gw[g] * L.length;
    const double x[3] = {L.x0[0] + u * (L.x1[0] - L.x0[0]), L.x0[1] + u * (L.x1[1] - L.x0[1]),
                         L.x0[2] + u * (L.x1[2] - L.x0[2])};
    for (std::size_t n = 0; n < idx.size(); ++n)
      m[n] += w * cutclip::detail::ipow(x[0], idx[n][0]) * cutclip::detail::ipow(x[1], idx[n][1]) *
              cutclip::detail::ipow(x[2], idx[n][2]);
  }
  return m;
}

// ----------------------------------------------------------------------------------------------- affine map scaling
namespace detail
{
// J is row-major, J[I*3+i] = d x_I / d xi_i (as cutquadric::SolidFit builds it). Returns det J and Ji = J^{-1}.
inline double inverse3(const double *J, double Ji[9])
{
  const double a = J[0], b = J[1], c = J[2], d = J[3], e = J[4], f = J[5], g = J[6], h = J[7], i = J[8];
  const double det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
  Ji[0] = (e * i - f * h) / det;
  Ji[1] = (c * h - b * i) / det;
  Ji[2] = (b * f - c * e) / det;
  Ji[3] = (f * g - d * i) / det;
  Ji[4] = (a * i - c * g) / det;
  Ji[5] = (c * d - a * f) / det;
  Ji[6] = (d * h - e * g) / det;
  Ji[7] = (b * g - a * h) / det;
  Ji[8] = (a * e - b * d) / det;
  return det;
}
}  // namespace detail

// Physical unit normal (into {phi > 0}) of the reference plane h, and kappa = dS_phys / (|det J| dS_ref): by Nanson's
// formula n dS = det J J^{-T} n_ref dS_ref, so kappa = |J^{-T} g| / |g| with g the reference gradient. J == nullptr is
// the identity (reference measure). A constant level set has no surface: n = 0, kappa = 0.
inline void planeScale(const Plane &h, const double *J, double n[3], double &kappa)
{
  const double g[3] = {h[0], h[1], h[2]};
  const double ng = std::sqrt(g[0] * g[0] + g[1] * g[1] + g[2] * g[2]);
  n[0] = n[1] = n[2] = 0.0;
  kappa = 0.0;
  if (!(ng > 0.0))
    return;
  double G[3] = {g[0], g[1], g[2]};
  if (J)
  {
    double Ji[9];
    detail::inverse3(J, Ji);
    for (int I = 0; I < 3; ++I)
      G[I] = Ji[0 * 3 + I] * g[0] + Ji[1 * 3 + I] * g[1] + Ji[2 * 3 + I] * g[2];  // (J^{-T} g)_I
  }
  const double nG = std::sqrt(G[0] * G[0] + G[1] * G[1] + G[2] * G[2]);
  for (int I = 0; I < 3; ++I)
    n[I] = G[I] / nG;
  kappa = nG / ng;
}

// Physical unit tangent of the triple line and kappa = dl_phys / (|det J| dl_ref) = |J t| / |det J|.
inline void lineScale(const TripleLine &L, const double *J, double t[3], double &kappa)
{
  t[0] = t[1] = t[2] = 0.0;
  kappa = 0.0;
  if (!L.exists)
    return;
  double T[3] = {L.tangent[0], L.tangent[1], L.tangent[2]}, det = 1.0;
  if (J)
  {
    double Ji[9];
    det = std::fabs(detail::inverse3(J, Ji));
    for (int I = 0; I < 3; ++I)
      T[I] = J[I * 3 + 0] * L.tangent[0] + J[I * 3 + 1] * L.tangent[1] + J[I * 3 + 2] * L.tangent[2];
  }
  const double nT = std::sqrt(T[0] * T[0] + T[1] * T[1] + T[2] * T[2]);
  for (int I = 0; I < 3; ++I)
    t[I] = T[I] / nT;
  kappa = nT / det;
}

// ------------------------------------------------------------------------------------------------------------- fits
struct PairFit
{
  int degree = 0;
  int sideS = 0, sideF = 0;                 // side() of each level set on this element
  double volA = 0.0, volB = 0.0, volS = 0.0;  // reference volumes (they sum to 1/6)
  TripleLine line;
  std::vector<double> cA, cB;               // region fits; the solid is 1 - cA - cB
  std::vector<double> cDsA, cDsB;           // Dirac of Gamma_s bordering A, bordering B
  std::vector<double> cDfActive, cDfMasked; // Dirac of Gamma_f in the fluid, in the solid
  std::vector<double> cL;                   // Dirac of the triple line
  double normalS[3] = {0., 0., 0.}, normalF[3] = {0., 0., 0.};  // physical unit normals into {phi > 0}
  double tangentL[3] = {0., 0., 0.};        // physical unit tangent of the triple line
  double kappaS = 0.0, kappaF = 0.0, kappaL = 0.0;

  // phi_s, phi_f: vertex values; k: fit degree; J: row-major 3x3 d x_I / d xi_i, or nullptr for the reference measure
  void calculate(const P1Tet &phi_s, const P1Tet &phi_f, int k, const double *J = nullptr)
  {
    degree = k;
    sideS = side(phi_s);
    sideF = side(phi_f);
    const Regions r = regions(phi_s, phi_f);
    const std::vector<double> mA = cutclip::moments(r.A, k), mB = cutclip::moments(r.B, k);
    volA = mA[0];
    volB = mB[0];
    volS = cutclip::moments(r.S, 0)[0];
    cA = cutquadric::detail::gramSolve(k, mA);
    cB = cutquadric::detail::gramSolve(k, mB);
    planeScale(planeOf(phi_s), J, normalS, kappaS);
    planeScale(planeOf(phi_f), J, normalF, kappaF);
    auto dirac = [k](const std::vector<Tri> &pieces, double kappa) {
      std::vector<double> m = cutclip::moments(pieces, k);
      for (double &v : m)
        v *= kappa;
      return cutquadric::detail::gramSolve(k, m);
    };
    cDsA = dirac(r.GsA, kappaS);
    cDsB = dirac(r.GsB, kappaS);
    cDfActive = dirac(r.GfActive, kappaF);
    cDfMasked = dirac(r.GfMasked, kappaF);
    line = tripleLine(phi_s, phi_f);
    lineScale(line, J, tangentL, kappaL);
    std::vector<double> mL = lineMoments(line, k);
    for (double &v : mL)
      v *= kappaL;
    cL = cutquadric::detail::gramSolve(k, mL);
  }

  // The 2D kernel's rule for a fluid interface that crosses the element but lies in the solid (ADR.h, areaTol): -1 if
  // the fluid is all Omega^- (B empty), +1 if all Omega^+ (A empty), 0 if the fluid is divided or absent.
  int demoted(double volTol = 1.0e-9) const
  {
    if (volB <= volTol && volA > volTol)
      return -1;
    if (volA <= volTol && volB > volTol)
      return 1;
    return 0;
  }

  double eval(const std::vector<double> &coef, const double x[3]) const
  {
    return cutquadric::QuadricFit::eval(coef, degree, x);
  }
};

// ------------------------------------------------------------------------------------------------------------ faces
// Vertex values on face k (opposite vertex k), the other three vertices in increasing order: (v0, v1, v2). The face
// parameters are x = X_v0 + s (X_v1 - X_v0) + t (X_v2 - X_v0), as cutquadric::faceNodal. They depend only on the three
// values the two neighbours share.
inline P1Tri faceNodal(const P1Tet &phi, int k)
{
  P1Tri f{{0., 0., 0.}};
  int n = 0;
  for (int m = 0; m < 4; ++m)
    if (m != k)
      f[n++] = phi[m];
  return f;
}

namespace detail
{
// solve G c = m, G_ab = int_F s^(pa+pb) t^(qa+qb) ds dt = p! q! / (p+q+2)! on the unit triangle (Cholesky + one refinement)
inline std::vector<double> gram2Solve(int k, const std::vector<double> &m)
{
  const auto idx = cutquadric::monomialIndex2(k);
  const int n = (int)idx.size();
  auto fact = [](int j) {
    double f = 1.0;
    for (int i = 2; i <= j; ++i)
      f *= i;
    return f;
  };
  std::vector<double> G(n * n), Lc(n * n, 0.0);
  for (int a = 0; a < n; ++a)
    for (int b = 0; b < n; ++b)
    {
      const int p = idx[a][0] + idx[b][0], q = idx[a][1] + idx[b][1];
      G[a * n + b] = fact(p) * fact(q) / fact(p + q + 2);
    }
  for (int i = 0; i < n; ++i)
    for (int j = 0; j <= i; ++j)
    {
      double sum = G[i * n + j];
      for (int l = 0; l < j; ++l)
        sum -= Lc[i * n + l] * Lc[j * n + l];
      if (i == j)
      {
        if (sum <= 0.0)
          throw std::runtime_error("cutpair: face Gram matrix is not positive definite");
        Lc[i * n + i] = std::sqrt(sum);
      }
      else
        Lc[i * n + j] = sum / Lc[j * n + j];
    }
  auto solve = [&](const std::vector<double> &rhs) {
    std::vector<double> y(n), x(n);
    for (int i = 0; i < n; ++i)
    {
      double s = rhs[i];
      for (int j = 0; j < i; ++j)
        s -= Lc[i * n + j] * y[j];
      y[i] = s / Lc[i * n + i];
    }
    for (int i = n - 1; i >= 0; --i)
    {
      double s = y[i];
      for (int j = i + 1; j < n; ++j)
        s -= Lc[j * n + i] * x[j];
      x[i] = s / Lc[i * n + i];
    }
    return x;
  };
  std::vector<double> c = solve(m), res(n);
  for (int i = 0; i < n; ++i)
  {
    double s = m[i];
    for (int j = 0; j < n; ++j)
      s -= G[i * n + j] * c[j];
    res[i] = s;
  }
  const std::vector<double> dc = solve(res);
  for (int i = 0; i < n; ++i)
    c[i] += dc[i];
  return c;
}
}  // namespace detail

// A triangular face cut by the two lines {phi_s = 0}, {phi_f = 0}: the regions A, B, S of the face in its own
// parameters (s, t), measure ds dt (the whole face is 1/2), moments and fits in cutquadric::monomialIndex2 order.
struct FacePair
{
  int degree = 0;
  int sideS = 0, sideF = 0;
  double areaA = 0.0, areaB = 0.0, areaS = 0.0;
  std::vector<double> mA, mB, mS;   // int_R s^p t^q ds dt
  std::vector<double> cA, cB;       // int_F c_R q = int_R q for q in P_k(F); the solid is 1 - cA - cB
  bool hasTriplePoint = false;      // the two lines meet inside the face
  double triplePoint[2] = {0., 0.}; // (s, t)

  void calculate(const P1Tri &phi_s, const P1Tri &phi_f, int k)
  {
    degree = k;
    sideS = side(phi_s);
    sideF = side(phi_f);
    // the face as the triangle (0,0,0), (1,0,0), (0,1,0) of the plane z = 0, so cutclip clips and integrates it
    const Tri F{{{0., 0., 0.}, {1., 0., 0.}, {0., 1., 0.}}};
    const Plane hs{phi_s[1] - phi_s[0], phi_s[2] - phi_s[0], 0.0, phi_s[0]};
    const Plane hf{phi_f[1] - phi_f[0], phi_f[2] - phi_f[0], 0.0, phi_f[0]};
    auto planar = [k](const std::vector<Tri> &pieces) {
      const std::vector<double> m3 = cutclip::moments(pieces, k);
      const auto idx = cutclip::monomialIndex(k);
      std::vector<double> m2;
      for (std::size_t n = 0; n < idx.size(); ++n)
        if (idx[n][2] == 0)
          m2.push_back(m3[n]);  // (p, q, 0): degree ascending, p descending = monomialIndex2
      return m2;
    };
    mA = planar(cutclip::clipTri(F, std::vector<Plane>{hs, negated(hf)}));
    mB = planar(cutclip::clipTri(F, std::vector<Plane>{hs, hf}));
    mS = planar(cutclip::clipTri(F, std::vector<Plane>{negated(hs)}));
    areaA = mA[0];
    areaB = mB[0];
    areaS = mS[0];
    cA = detail::gram2Solve(k, mA);
    cB = detail::gram2Solve(k, mB);
    // triple point: hs = hf = 0 in (s, t)
    hasTriplePoint = false;
    const double det = hs[0] * hf[1] - hs[1] * hf[0];
    const double ns = hs[0] * hs[0] + hs[1] * hs[1], nf = hf[0] * hf[0] + hf[1] * hf[1];
    if (det * det > 1.0e-28 * ns * nf)
    {
      const double s = (-hs[3] * hf[1] + hs[1] * hf[3]) / det, t = (-hs[0] * hf[3] + hs[3] * hf[0]) / det;
      if (s >= 0.0 && t >= 0.0 && s + t <= 1.0)
      {
        hasTriplePoint = true;
        triplePoint[0] = s;
        triplePoint[1] = t;
      }
    }
  }
  double eval(const std::vector<double> &coef, double s, double t) const
  {
    const auto idx = cutquadric::monomialIndex2(degree);
    double v = 0.0;
    for (std::size_t n = 0; n < idx.size(); ++n)
      v += coef[n] * cutclip::detail::ipow(s, idx[n][0]) * cutclip::detail::ipow(t, idx[n][1]);
    return v;
  }
};
}  // namespace cutpair
}  // namespace proteus
#endif
