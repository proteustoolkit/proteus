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
}
