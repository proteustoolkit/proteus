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
}
