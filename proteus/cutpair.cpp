// Python binding for proteus/cutpair.h (C44 phase 3, part A): a tetrahedron cut by two P1 level sets.
// Test binding; no kernel calls it. The last function evaluates the 2D composite fit the ADR kernel uses on a triply cut
// triangle, so the face fits here can be compared with it.
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include "cutpair.h"
#include "equivalent_polynomials.h"

namespace py = pybind11;
using namespace proteus::cutpair;
using Arr = py::array_t<double, py::array::c_style | py::array::forcecast>;

template <std::size_t N>
static std::array<double, N> toValues(const Arr &a)
{
  if (a.ndim() != 1 || a.shape(0) != (py::ssize_t)N)
    throw std::invalid_argument("level-set values must be one per vertex");
  std::array<double, N> v;
  for (std::size_t i = 0; i < N; ++i)
    v[i] = a.at(i);
  return v;
}

template <std::size_t N>
static Arr piecesToArray(const std::vector<std::array<std::array<double, 3>, N>> &v)
{
  Arr out({(py::ssize_t)v.size(), (py::ssize_t)N, (py::ssize_t)3});
  for (std::size_t i = 0; i < v.size(); ++i)
    for (std::size_t k = 0; k < N; ++k)
      for (int c = 0; c < 3; ++c)
        out.mutable_at(i, k, c) = v[i][k][c];
  return out;
}

static std::vector<double> vec3(const double *x) { return {x[0], x[1], x[2]}; }

static py::dict lineDict(const TripleLine &L)
{
  py::dict d;
  d["exists"] = L.exists;
  d["x0"] = std::vector<double>(L.x0.begin(), L.x0.end());
  d["x1"] = std::vector<double>(L.x1.begin(), L.x1.end());
  d["face0"] = L.face0;
  d["face1"] = L.face1;
  d["length"] = L.length;
  d["tangent"] = std::vector<double>(L.tangent.begin(), L.tangent.end());
  return d;
}

static Arr evalMany(const PairFit &f, const std::vector<double> &coef, const Arr &x)
{
  Arr out(x.shape(0));
  for (py::ssize_t i = 0; i < x.shape(0); ++i)
  {
    const double p[3] = {x.at(i, 0), x.at(i, 1), x.at(i, 2)};
    out.mutable_at(i) = f.eval(coef, p);
  }
  return out;
}

template <int nP>
static py::dict composite2d(const Arr &phi_s, const Arr &phi_f, const Arr &st)
{
  const auto ps = toValues<3>(phi_s), pf = toValues<3>(phi_f);
  constexpr int n = ((nP + 1) * (nP + 2)) / 2;
  double cA[n], cB[n];
  const bool ok = equivalent_polynomials::composite::weights<nP>(ps.data(), pf.data(), cA, cB);
  Arr a(st.shape(0)), b(st.shape(0));
  for (py::ssize_t i = 0; i < st.shape(0); ++i)
  {
    a.mutable_at(i) = ok ? equivalent_polynomials::composite::evaluate<nP>(cA, st.at(i, 0), st.at(i, 1)) : 0.0;
    b.mutable_at(i) = ok ? equivalent_polynomials::composite::evaluate<nP>(cB, st.at(i, 0), st.at(i, 1)) : 0.0;
  }
  py::dict d;
  d["valid"] = ok;
  d["A"] = a;
  d["B"] = b;
  return d;
}

PYBIND11_MODULE(cutpair, m)
{
  m.doc() = "A tetrahedron cut by two P1 level sets: regions, fits, split Diracs, triple line (C44 phase 3, part A)";
  m.def("monomials", [](int degree) { return proteus::cutclip::monomialIndex(degree); });
  m.def("monomials2", [](int degree) { return proteus::cutquadric::monomialIndex2(degree); });
  m.def("side", [](Arr phi) {
    if (phi.ndim() == 1 && phi.shape(0) == 3)
      return side(toValues<3>(phi));
    return side(toValues<4>(phi));
  });
  m.def("regions", [](Arr phi_s, Arr phi_f) {
    const Regions r = regions(toValues<4>(phi_s), toValues<4>(phi_f));
    py::dict d;
    d["A"] = piecesToArray<4>(r.A);
    d["B"] = piecesToArray<4>(r.B);
    d["S"] = piecesToArray<4>(r.S);
    d["GsA"] = piecesToArray<3>(r.GsA);
    d["GsB"] = piecesToArray<3>(r.GsB);
    d["GfActive"] = piecesToArray<3>(r.GfActive);
    d["GfMasked"] = piecesToArray<3>(r.GfMasked);
    return d;
  });
  // moments of x^p y^q z^r over every piece, reference measure (volume, area, arc length)
  m.def("moments", [](Arr phi_s, Arr phi_f, int degree) {
    const P1Tet ps = toValues<4>(phi_s), pf = toValues<4>(phi_f);
    const Regions r = regions(ps, pf);
    py::dict d;
    d["A"] = proteus::cutclip::moments(r.A, degree);
    d["B"] = proteus::cutclip::moments(r.B, degree);
    d["S"] = proteus::cutclip::moments(r.S, degree);
    d["GsA"] = proteus::cutclip::moments(r.GsA, degree);
    d["GsB"] = proteus::cutclip::moments(r.GsB, degree);
    d["GfActive"] = proteus::cutclip::moments(r.GfActive, degree);
    d["GfMasked"] = proteus::cutclip::moments(r.GfMasked, degree);
    d["L"] = lineMoments(tripleLine(ps, pf), degree);
    return d;
  });
  m.def("triple_line", [](Arr phi_s, Arr phi_f) { return lineDict(tripleLine(toValues<4>(phi_s), toValues<4>(phi_f))); });
  // J: (3,3) with J[I,i] = d x_I / d xi_i, or None for the reference measure; points: (n,3) reference points
  m.def(
    "fit",
    [](Arr phi_s, Arr phi_f, int k, py::object J, Arr points) {
      if (points.ndim() != 2 || points.shape(1) != 3)
        throw std::invalid_argument("points must be (n,3)");
      double Jm[9];
      const double *Jp = nullptr;
      if (!J.is_none())
      {
        Arr Ja = J.cast<Arr>();
        if (Ja.ndim() != 2 || Ja.shape(0) != 3 || Ja.shape(1) != 3)
          throw std::invalid_argument("J must be (3,3)");
        for (int I = 0; I < 3; ++I)
          for (int i = 0; i < 3; ++i)
            Jm[I * 3 + i] = Ja.at(I, i);
        Jp = Jm;
      }
      PairFit f;
      f.calculate(toValues<4>(phi_s), toValues<4>(phi_f), k, Jp);
      py::dict d;
      d["side_s"] = f.sideS;
      d["side_f"] = f.sideF;
      d["vol_a"] = f.volA;
      d["vol_b"] = f.volB;
      d["vol_s"] = f.volS;
      d["demoted"] = f.demoted();
      d["line"] = lineDict(f.line);
      d["normal_s"] = vec3(f.normalS);
      d["normal_f"] = vec3(f.normalF);
      d["tangent_l"] = vec3(f.tangentL);
      d["kappa_s"] = f.kappaS;
      d["kappa_f"] = f.kappaF;
      d["kappa_l"] = f.kappaL;
      d["cA"] = f.cA;
      d["cB"] = f.cB;
      d["cDsA"] = f.cDsA;
      d["cDsB"] = f.cDsB;
      d["cDfActive"] = f.cDfActive;
      d["cDfMasked"] = f.cDfMasked;
      d["cL"] = f.cL;
      d["A"] = evalMany(f, f.cA, points);
      d["B"] = evalMany(f, f.cB, points);
      d["DsA"] = evalMany(f, f.cDsA, points);
      d["DsB"] = evalMany(f, f.cDsB, points);
      d["DfActive"] = evalMany(f, f.cDfActive, points);
      d["DfMasked"] = evalMany(f, f.cDfMasked, points);
      d["L"] = evalMany(f, f.cL, points);
      return d;
    },
    py::arg("phi_s"), py::arg("phi_f"), py::arg("k"), py::arg("J"), py::arg("points"));
  m.def("face_nodal", [](Arr phi, int k) {
    const P1Tri f = faceNodal(toValues<4>(phi), k);
    return std::vector<double>(f.begin(), f.end());
  });
  // st: (n,2) face parameters
  m.def("face_pair", [](Arr phi_s, Arr phi_f, int k, Arr st) {
    if (st.ndim() != 2 || st.shape(1) != 2)
      throw std::invalid_argument("st must be (n,2)");
    FacePair f;
    f.calculate(toValues<3>(phi_s), toValues<3>(phi_f), k);
    py::dict d;
    d["side_s"] = f.sideS;
    d["side_f"] = f.sideF;
    d["area_a"] = f.areaA;
    d["area_b"] = f.areaB;
    d["area_s"] = f.areaS;
    d["mA"] = f.mA;
    d["mB"] = f.mB;
    d["mS"] = f.mS;
    d["cA"] = f.cA;
    d["cB"] = f.cB;
    d["has_triple_point"] = f.hasTriplePoint;
    d["triple_point"] = std::vector<double>{f.triplePoint[0], f.triplePoint[1]};
    Arr a(st.shape(0)), b(st.shape(0));
    for (py::ssize_t i = 0; i < st.shape(0); ++i)
    {
      a.mutable_at(i) = f.eval(f.cA, st.at(i, 0), st.at(i, 1));
      b.mutable_at(i) = f.eval(f.cB, st.at(i, 0), st.at(i, 1));
    }
    d["A"] = a;
    d["B"] = b;
    return d;
  });
  // the 2D kernel's composite fit (equivalent_polynomials::composite::weights<nP>) evaluated at (s,t) points
  m.def("composite2d", [](Arr phi_s, Arr phi_f, int nP, Arr st) {
    if (st.ndim() != 2 || st.shape(1) != 2)
      throw std::invalid_argument("st must be (n,2)");
    if (nP == 2)
      return composite2d<2>(phi_s, phi_f, st);
    if (nP == 3)
      return composite2d<3>(phi_s, phi_f, st);
    if (nP == 4)
      return composite2d<4>(phi_s, phi_f, st);
    throw std::invalid_argument("composite2d: nP must be 2, 3 or 4");
  });
}
