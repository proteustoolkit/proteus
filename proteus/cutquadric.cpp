// Python binding for proteus/cutquadric.h (C44 phase 1b): curved cuts of a tetrahedron/triangle by a P2 level set.
// Test binding (the ADR kernel includes the header directly); the oracle for the fits is cutclip.
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include "cutquadric.h"

namespace py = pybind11;
using namespace proteus::cutquadric;
using Arr = py::array_t<double, py::array::c_style | py::array::forcecast>;

static P2Tet toP2Tet(const Arr &a)
{
  if (a.ndim() != 1 || a.shape(0) != 10)
    throw std::invalid_argument("P2 tet values must be (10,) in proteus node order");
  P2Tet u;
  for (int i = 0; i < 10; ++i)
    u[i] = a.at(i);
  return u;
}

static Arr evalMany(const QuadricFit &f, const Arr &x, int kind)
{
  if (x.ndim() != 2 || x.shape(1) != 3)
    throw std::invalid_argument("points must be (n,3)");
  Arr out(x.shape(0));
  for (py::ssize_t i = 0; i < x.shape(0); ++i)
  {
    const double p[3] = {x.at(i, 0), x.at(i, 1), x.at(i, 2)};
    out.mutable_at(i) = kind == 0 ? f.evalH(p) : (kind == 1 ? f.evalImH(p) : f.evalD(p));
  }
  return out;
}

PYBIND11_MODULE(cutquadric, m)
{
  m.doc() = "Curved cuts of a tetrahedron by a quadratic level set (C44 phase 1b)";
  m.def("monomials", [](int degree) { return monomialIndex(degree); });
  m.def("p2_to_monomial", [](Arr u) {
    const Coef3 c = p2ToMonomial(toP2Tet(u));
    return std::vector<double>(c.begin(), c.end());
  });
  m.def("classify", [](Arr u) {
    const Classification c = classify(toP2Tet(u));
    py::dict d;
    d["label"] = labelName(c.label);
    d["cut"] = c.cut;
    d["bernstein_min"] = c.bernsteinMin;
    d["bernstein_max"] = c.bernsteinMax;
    d["range_min"] = c.rangeMin;
    d["range_max"] = c.rangeMax;
    d["double_edges"] = c.doubleEdges;
    return d;
  });
  m.def("volume_moments",
        [](Arr u, int degree, int order, double planar_tol, double deflate_tol) {
          Options opt;
          opt.order = order;
          opt.planarTol = planar_tol;
          opt.deflateTol = deflate_tol;
          return volumeMomentsNodal(toP2Tet(u), degree, opt);
        },
        py::arg("u"), py::arg("degree"), py::arg("order") = 16, py::arg("planar_tol") = 1e-12,
        py::arg("deflate_tol") = 1e-10);
  m.def("monomials2", [](int degree) { return monomialIndex2(degree); });
  m.def("p2_tri_to_monomial", [](Arr u) {
    if (u.ndim() != 1 || u.shape(0) != 6)
      throw std::invalid_argument("P2 triangle values must be (6,)");
    P2Tri t;
    for (int i = 0; i < 6; ++i)
      t[i] = u.at(i);
    const Coef2 c = p2TriToMonomial(t);
    return std::vector<double>(c.begin(), c.end());
  });
  m.def("face_nodal", [](Arr u, int k) {
    if (k < 0 || k > 3)
      throw std::invalid_argument("face must be 0..3");
    const P2Tri f = faceNodal(toP2Tet(u), k);
    return std::vector<double>(f.begin(), f.end());
  });
  m.def("classify_face", [](Arr u) {
    if (u.ndim() != 1 || u.shape(0) != 6)
      throw std::invalid_argument("P2 triangle values must be (6,)");
    P2Tri t;
    for (int i = 0; i < 6; ++i)
      t[i] = u.at(i);
    const FaceClassification c = classifyTri(t);
    py::dict d;
    d["label"] = faceLabelName(c.label);
    d["cut"] = c.cut;
    d["bernstein_min"] = c.bernsteinMin;
    d["bernstein_max"] = c.bernsteinMax;
    d["range_min"] = c.rangeMin;
    d["range_max"] = c.rangeMax;
    return d;
  });
  m.def("tri_moments",
        [](Arr u, int degree, int order, double planar_tol, double deflate_tol) {
          if (u.ndim() != 1 || u.shape(0) != 6)
            throw std::invalid_argument("P2 triangle values must be (6,)");
          P2Tri t;
          for (int i = 0; i < 6; ++i)
            t[i] = u.at(i);
          Options opt;
          opt.order = order;
          opt.planarTol = planar_tol;
          opt.deflateTol = deflate_tol;
          return triMomentsNodal(t, degree, opt);
        },
        py::arg("u"), py::arg("degree"), py::arg("order") = 16, py::arg("planar_tol") = 1e-12,
        py::arg("deflate_tol") = 1e-10);
  m.def("face_moments",
        [](Arr u, int face, int degree, int order, double planar_tol, double deflate_tol) {
          if (face < 0 || face > 3)
            throw std::invalid_argument("face must be 0..3");
          Options opt;
          opt.order = order;
          opt.planarTol = planar_tol;
          opt.deflateTol = deflate_tol;
          return faceMoments(toP2Tet(u), face, degree, opt);
        },
        py::arg("u"), py::arg("face"), py::arg("degree"), py::arg("order") = 16, py::arg("planar_tol") = 1e-12,
        py::arg("deflate_tol") = 1e-10);
  m.def("choose_axis", [](Arr u) { return chooseAxis(p2ToMonomial(toP2Tet(u))); });
  m.def("surface_rule",
        [](Arr u, int order, int axis, py::object J, double planar_tol, double deflate_tol) {
          Options opt;
          opt.order = order;
          opt.planarTol = planar_tol;
          opt.deflateTol = deflate_tol;
          double Jm[9];
          const bool hasJ = !J.is_none();
          if (hasJ)
          {
            Arr a = J.cast<Arr>();
            if (a.ndim() != 2 || a.shape(0) != 3 || a.shape(1) != 3)
              throw std::invalid_argument("J must be (3,3)");
            for (int i = 0; i < 3; ++i)
              for (int j = 0; j < 3; ++j)
                Jm[3 * i + j] = a.at(i, j);
          }
          const SurfaceRule r = surfaceRuleNodal(toP2Tet(u), opt, axis, hasJ ? Jm : nullptr);
          const py::ssize_t n = (py::ssize_t)r.w.size();
          Arr P({n, (py::ssize_t)3}), W(n), N({n, (py::ssize_t)3});
          for (py::ssize_t i = 0; i < n; ++i)
          {
            W.mutable_at(i) = r.w[i];
            for (int c = 0; c < 3; ++c)
            {
              P.mutable_at(i, c) = r.x[i][c];
              N.mutable_at(i, c) = r.n[i][c];
            }
          }
          return py::make_tuple(P, W, N);
        },
        py::arg("u"), py::arg("order") = 16, py::arg("axis") = -1, py::arg("J") = py::none(), py::arg("planar_tol") = 1e-12,
        py::arg("deflate_tol") = 1e-10);
  m.def("surface_moments",
        [](Arr u, int degree, int order, int axis, py::object J, double planar_tol, double deflate_tol) {
          Options opt;
          opt.order = order;
          opt.planarTol = planar_tol;
          opt.deflateTol = deflate_tol;
          double Jm[9];
          const bool hasJ = !J.is_none();
          if (hasJ)
          {
            Arr a = J.cast<Arr>();
            if (a.ndim() != 2 || a.shape(0) != 3 || a.shape(1) != 3)
              throw std::invalid_argument("J must be (3,3)");
            for (int i = 0; i < 3; ++i)
              for (int j = 0; j < 3; ++j)
                Jm[3 * i + j] = a.at(i, j);
          }
          const SurfaceRule r = surfaceRuleNodal(toP2Tet(u), opt, axis, hasJ ? Jm : nullptr);
          std::vector<double> S, Sn[3];
          surfaceMoments(r, degree, S, Sn);
          return py::make_tuple(S, std::vector<std::vector<double>>{Sn[0], Sn[1], Sn[2]});
        },
        py::arg("u"), py::arg("degree"), py::arg("order") = 16, py::arg("axis") = -1, py::arg("J") = py::none(),
        py::arg("planar_tol") = 1e-12, py::arg("deflate_tol") = 1e-10);

  py::class_<QuadricFit>(m, "QuadricFit")
      .def(py::init([](Arr u, int degree, int order, int axis, py::object J, double planar_tol, double deflate_tol) {
             Options opt;
             opt.order = order;
             opt.planarTol = planar_tol;
             opt.deflateTol = deflate_tol;
             double Jm[9];
             const bool hasJ = !J.is_none();
             if (hasJ)
             {
               Arr a = J.cast<Arr>();
               if (a.ndim() != 2 || a.shape(0) != 3 || a.shape(1) != 3)
                 throw std::invalid_argument("J must be (3,3)");
               for (int i = 0; i < 3; ++i)
                 for (int j = 0; j < 3; ++j)
                   Jm[3 * i + j] = a.at(i, j);
             }
             auto f = std::make_unique<QuadricFit>();
             f->calculate(toP2Tet(u), degree, opt, axis, hasJ ? Jm : nullptr);
             return f;
           }),
           py::arg("u"), py::arg("degree"), py::arg("order") = 16, py::arg("axis") = -1, py::arg("J") = py::none(),
           py::arg("planar_tol") = 1e-12, py::arg("deflate_tol") = 1e-10)
      .def_readonly("degree", &QuadricFit::degree)
      .def_readonly("coef_H", &QuadricFit::cH)
      .def_readonly("coef_D", &QuadricFit::cD)
      .def("coef_Dn", [](const QuadricFit &f, int i) { return f.cDn[i]; })
      .def("H", [](const QuadricFit &f, Arr x) { return evalMany(f, x, 0); })
      .def("ImH", [](const QuadricFit &f, Arr x) { return evalMany(f, x, 1); })
      .def("D", [](const QuadricFit &f, Arr x) { return evalMany(f, x, 2); })
      .def("Dn", [](const QuadricFit &f, Arr x) {
        if (x.ndim() != 2 || x.shape(1) != 3)
          throw std::invalid_argument("points must be (n,3)");
        Arr out({x.shape(0), (py::ssize_t)3});
        for (py::ssize_t i = 0; i < x.shape(0); ++i)
        {
          const double p[3] = {x.at(i, 0), x.at(i, 1), x.at(i, 2)};
          for (int I = 0; I < 3; ++I)
            out.mutable_at(i, I) = f.evalDn(I, p);
        }
        return out;
      });
}
