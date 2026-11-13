// Python binding for proteus/cutclip.h (C44): exact planar clipping of tetrahedra and triangles and their moments.
// Used as the oracle in test/ADR3D and by the fit-versus-clip checks; not called from the ADR kernel.
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <memory>
#include <vector>
#include "cutclip.h"
#include "equivalent_polynomials.h"

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

// The real equivalent_polynomials::Simplex with the two-sided IFEM basis (the object ADR.h holds per
// element), evaluated at up to 64 reference points of one simplex. Test hook for the 2D and 3D basis
// (C44 phase 2): returns the cut classification, the Heaviside/Dirac fits, VA/VB and their gradients and the
// unit flux-jump lift at each point, exactly as the kernel reads them through set_quad().
template <int nSpace>
static py::dict ifemBasis(const Arr &nodes, const Arr &phi, const Arr &pts, double ma, double mb)
{
  constexpr int nN = nSpace + 1, NQ = 64;
  if (nodes.ndim() != 2 || nodes.shape(0) != nN || nodes.shape(1) != 3)
    throw std::invalid_argument("nodes must be (nSpace+1, 3)");
  if (phi.ndim() != 1 || phi.shape(0) != nN)
    throw std::invalid_argument("phi must be (nSpace+1,)");
  if (pts.ndim() != 2 || pts.shape(1) != 3 || pts.shape(0) > NQ)
    throw std::invalid_argument("pts must be (m <= 64, 3) reference points");
  const int m = (int)pts.shape(0);
  using S = equivalent_polynomials::Simplex<nSpace, nN, 3, NQ, NQ, true>;
  std::unique_ptr<S> sx(new S());
  std::vector<double> xi(NQ * 3, 0.0);
  for (int q = 0; q < m; ++q)
    for (int c = 0; c < 3; ++c)
      xi[q * 3 + c] = pts.at(q, c);
  std::vector<double> ph(nN), nd(nN * 3);
  for (int i = 0; i < nN; ++i)
  {
    ph[i] = phi.at(i);
    for (int c = 0; c < 3; ++c)
      nd[i * 3 + c] = nodes.at(i, c);
  }
  const int icase = sx->calculate(ph.data(), nd.data(), xi.data(), ma, mb, 0.0, false, false);
  py::dict out;
  out["icase"] = icase;
  out["edge"] = sx->edge;
  out["corner"] = sx->corner;
  out["inside_out"] = sx->inside_out;
  py::array_t<double> H({m}), ImH({m}), D({m}), va({m, nN}), vb({m, nN}), ga({m, nN, 3}), gb({m, nN, 3}), psi({m, 2, 4});
  py::array_t<double> normal({nSpace});
  if (icase == 0)
  {
    for (int c = 0; c < nSpace; ++c)
      normal.mutable_at(c) = sx->get_normal()[c];
    for (int q = 0; q < m; ++q)
    {
      sx->set_quad(q);
      H.mutable_at(q) = sx->H(0., 0.);
      ImH.mutable_at(q) = sx->ImH(0., 0.);
      D.mutable_at(q) = sx->D(0., 0.);
      for (int i = 0; i < nN; ++i)
      {
        va.mutable_at(q, i) = sx->VA(i);
        vb.mutable_at(q, i) = sx->VB(i);
        ga.mutable_at(q, i, 0) = sx->VA_x(i);
        ga.mutable_at(q, i, 1) = sx->VA_y(i);
        ga.mutable_at(q, i, 2) = sx->VA_z(i);
        gb.mutable_at(q, i, 0) = sx->VB_x(i);
        gb.mutable_at(q, i, 1) = sx->VB_y(i);
        gb.mutable_at(q, i, 2) = sx->VB_z(i);
      }
      const double pa[4] = {sx->PSIA(), sx->PSIA_x(), sx->PSIA_y(), sx->PSIA_z()};
      const double pb[4] = {sx->PSIB(), sx->PSIB_x(), sx->PSIB_y(), sx->PSIB_z()};
      for (int c = 0; c < 4; ++c)
      {
        psi.mutable_at(q, 0, c) = pa[c];
        psi.mutable_at(q, 1, c) = pb[c];
      }
    }
  }
  out["normal"] = normal;
  out["H"] = H;
  out["ImH"] = ImH;
  out["D"] = D;
  out["va"] = va;
  out["vb"] = vb;
  out["va_grad"] = ga;
  out["vb_grad"] = gb;
  out["psi"] = psi;
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
  m.def("ifem_basis_2d", &ifemBasis<2>, py::arg("nodes"), py::arg("phi"), py::arg("pts"), py::arg("ma"), py::arg("mb"));
  m.def("ifem_basis_3d", &ifemBasis<3>, py::arg("nodes"), py::arg("phi"), py::arg("pts"), py::arg("ma"), py::arg("mb"));
  m.def("monomials", [](int degree) { return monomialIndex(degree); });
}
