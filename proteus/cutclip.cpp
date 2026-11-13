// Python binding for proteus/cutclip.h (C44): exact planar clipping of tetrahedra and triangles and their moments.
// Used as the oracle in test/ADR3D and by the fit-versus-clip checks; not called from the ADR kernel.
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include "cutclip.h"

namespace py = pybind11;
using namespace proteus::cutclip;
using Arr = py::array_t<double, py::array::c_style | py::array::forcecast>;

static Tet toTet(const Arr &a)
{
  if (a.ndim() != 2 || a.shape(0) != 4 || a.shape(1) != 3)
    throw std::invalid_argument("tet must be (4,3)");
  Tet t;
  for (int i = 0; i < 4; ++i)
    for (int c = 0; c < 3; ++c)
      t[i][c] = a.at(i, c);
  return t;
}
static Tri toTri(const Arr &a)
{
  if (a.ndim() != 2 || a.shape(0) != 3 || a.shape(1) != 3)
    throw std::invalid_argument("tri must be (3,3)");
  Tri t;
  for (int i = 0; i < 3; ++i)
    for (int c = 0; c < 3; ++c)
      t[i][c] = a.at(i, c);
  return t;
}
static std::vector<Plane> toPlanes(const Arr &a)
{
  if (a.ndim() != 2 || a.shape(1) != 4)
    throw std::invalid_argument("planes must be (m,4): a.x + b > 0 is kept");
  std::vector<Plane> p(a.shape(0));
  for (py::ssize_t i = 0; i < a.shape(0); ++i)
    for (int c = 0; c < 4; ++c)
      p[i][c] = a.at(i, c);
  return p;
}
template <class P, std::size_t N>
static Arr piecesToArray(const std::vector<std::array<std::array<double, 3>, N>> &v)
{
  Arr out({(py::ssize_t)v.size(), (py::ssize_t)N, (py::ssize_t)3});
  for (std::size_t i = 0; i < v.size(); ++i)
    for (std::size_t k = 0; k < N; ++k)
      for (int c = 0; c < 3; ++c)
        out.mutable_at(i, k, c) = v[i][k][c];
  return out;
}

PYBIND11_MODULE(cutclip, m)
{
  m.doc() = "Exact planar clipping of a tetrahedron/triangle and its polynomial moments (C44 oracle)";
  m.def("clip_tet", [](Arr tet, Arr planes) { return piecesToArray<Tet, 4>(clipTet(toTet(tet), toPlanes(planes))); });
  m.def("clip_tri", [](Arr tri, Arr planes) { return piecesToArray<Tri, 3>(clipTri(toTri(tri), toPlanes(planes))); });
  m.def("cut_surface", [](Arr tet, Arr plane, Arr others) {
    Arr pl = plane;
    if (pl.ndim() != 1 || pl.shape(0) != 4)
      throw std::invalid_argument("plane must be (4,)");
    Plane h{pl.at(0), pl.at(1), pl.at(2), pl.at(3)};
    return piecesToArray<Tri, 3>(cutSurface(toTet(tet), h, toPlanes(others)));
  });
  m.def("tet_moments", [](Arr tet, Arr planes, int degree) {
    return moments(clipTet(toTet(tet), toPlanes(planes)), degree);
  });
  m.def("tri_moments", [](Arr tri, Arr planes, int degree) {
    return moments(clipTri(toTri(tri), toPlanes(planes)), degree);
  });
  m.def("surface_moments", [](Arr tet, Arr plane, Arr others, int degree) {
    Arr pl = plane;
    Plane h{pl.at(0), pl.at(1), pl.at(2), pl.at(3)};
    return moments(cutSurface(toTet(tet), h, toPlanes(others)), degree);
  });
  m.def("monomials", [](int degree) { return monomialIndex(degree); });
}
