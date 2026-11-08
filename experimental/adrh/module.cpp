#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <type_traits>
#include "coefficients.h"
#include "boundary.h"
#include "engine.h"
#include "richards.h"
#include "lowOrder.h"
#include "vof.h"
#include "ncls.h"
#include "rdls.h"
#include "mcorr.h"
#include "rans2p.h"
#include <stdexcept>
#include <vector>
#include <string>

namespace py = pybind11;
using Arr = py::array_t<double, py::array::c_style | py::array::forcecast>;

template <class C, int nS>
py::tuple run(const C &coef, Arr xn, Arr u, py::object uold, Arr qp, Arr qw, double t, double dt)
{
  constexpr int nc = C::nc;
  const int nE = xn.shape(0), nQ = qw.shape(0), nD = nS + 1;
  Arr uo;
  const double *uop = nullptr;
  if (!uold.is_none())
  {
    uo = uold.cast<Arr>();
    uop = uo.data();
  }
  py::array_t<double> R({nE, nc, nD}), J({nE, nc, nD, nc, nD});
  {
    py::gil_scoped_release release;
    adrh::elementKernel<C, nS>(coef, nE, nQ, xn.data(), u.data(), uop, qp.data(), qw.data(), t, dt,
                               R.mutable_data(), J.mutable_data());
  }
  return py::make_tuple(R, J);
}

template <class C, int nS>
py::tuple runMasked(const C &coef, Arr xn, Arr u, py::object uold, Arr qp, Arr qw, double t, double dt)
{
  constexpr int nc = C::nc;
  const int nE = xn.shape(0), nQ = qw.shape(0), nD = nS + 1;
  Arr uo;
  const double *uop = nullptr;
  if (!uold.is_none())
  {
    uo = uold.cast<Arr>();
    uop = uo.data();
  }
  py::array_t<double> R({nE, nc, nD}), J({nE, nc, nD, nc, nD});
  {
    py::gil_scoped_release release;
    adrh::elementKernelMasked<C, nS>(coef, nE, nQ, xn.data(), u.data(), uop, qp.data(), qw.data(), t, dt,
                                     R.mutable_data(), J.mutable_data());
  }
  return py::make_tuple(R, J);
}

// Lagrange spaces by name and dimension; f gets a default-constructed space as a type tag and returns a py::tuple.
template <class F>
py::tuple dispatchSpace(int nd, const std::string &space, F &&f)
{
  if (nd == 1 && space == "p1") return f(adrh::P1Simplex<1>{});
  if (nd == 1 && space == "p2") return f(adrh::P2Simplex<1>{});
  if (nd == 2 && space == "p1") return f(adrh::P1Simplex<2>{});
  if (nd == 3 && space == "p1") return f(adrh::P1Simplex<3>{});
  if (nd == 2 && space == "p2") return f(adrh::P2Simplex<2>{});
  if (nd == 3 && space == "p2") return f(adrh::P2Simplex<3>{});
  if (nd == 2 && space == "q1") return f(adrh::Q1Cube<2>{});
  if (nd == 3 && space == "q1") return f(adrh::Q1Cube<3>{});
  throw std::invalid_argument("no space '" + space + "' in " + std::to_string(nd) + "D (p1, p2, q1; nd 1 to 3, q1 only 2 or 3)");
}

PYBIND11_MODULE(adrh_proto, m)
{
  // exterior faces of the advection-diffusion sine problem in nd = 2 (segments) or 3 (triangles) dimensions, for the
  // space "p1" or "p2"; faceKind (nF) per face: 1 = flux face (diffusive flux given, advection by the interior value on
  // outflow), 2 = flux face with the advective flux given too, 3 = weak Dirichlet (penalty_constant / h^penalty_power,
  // adjoint sigma: 0 IIPG, -1 NIPG, +1 SIPG), 4 = Robin alpha u + beta q.n = g with (alpha, beta) given
  m.def("advection_diffusion_sine_faces", [](int nd, const std::string &space, Arr b, double kappa, Arr omega, double omega0,
                                              py::array_t<int, py::array::c_style | py::array::forcecast> faceLocal,
                                              py::array_t<int, py::array::c_style | py::array::forcecast> faceKind,
                                              Arr xn, Arr u, Arr qpf, Arr qwf,
                                              double penalty_constant, double penalty_power, double sigma,
                                              double robin_alpha, double robin_beta) {
    return dispatchSpace(nd, space, [&](auto spaceTag) {
      using S = decltype(spaceTag);
      constexpr int nS = S::nS, nD = S::nDOF;
      adrh::AdvectionDiffusionSine<nS> c(b.data(), kappa, omega.data(), omega0);
      adrh::SineBC<nS> bc(b.data(), kappa, omega.data(), omega0, faceKind.data(), robin_alpha, robin_beta);
      adrh::WeakDirichletParams wd{penalty_constant, penalty_power, sigma};
      const int nF = xn.shape(0), nQf = qwf.shape(0);
      py::array_t<double> Rf({nF, 1, nD}), Jf({nF, 1, nD, 1, nD});
      {
        py::gil_scoped_release release;
        adrh::exteriorFaceKernel<adrh::AdvectionDiffusionSine<nS>, adrh::SineBC<nS>, S>(
            c, bc, wd, nF, nQf, faceLocal.data(), xn.data(), u.data(), qpf.data(), qwf.data(), 0.0,
            Rf.mutable_data(), Jf.mutable_data());
      }
      return py::make_tuple(Rf, Jf);
    });
  });
  // element residual and Jacobian of the same problem, same nd and space
  m.def("advection_diffusion_sine", [](int nd, const std::string &space, Arr b, double kappa, Arr omega, double omega0,
                                        Arr xn, Arr u, Arr qp, Arr qw) {
    return dispatchSpace(nd, space, [&](auto spaceTag) {
      using S = decltype(spaceTag);
      constexpr int nS = S::nS, nD = S::nDOF;
      adrh::AdvectionDiffusionSine<nS> c(b.data(), kappa, omega.data(), omega0);
      const int nE = xn.shape(0), nQ = qw.shape(0);
      py::array_t<double> R({nE, 1, nD}), J({nE, 1, nD, 1, nD});
      {
        py::gil_scoped_release release;
        adrh::elementKernelSpace<adrh::AdvectionDiffusionSine<nS>, S>(c, nE, nQ, xn.data(), u.data(), nullptr,
                                                                      qp.data(), qw.data(), 0.0, 0.0,
                                                                      R.mutable_data(), J.mutable_data());
      }
      return py::make_tuple(R, J);
    });
  });
  // (nDOF, nVertices, faces (nFaces, nFaceVertices) as local vertex numbers) of a space
  m.def("space_info", [](int nd, const std::string &space) {
    return dispatchSpace(nd, space, [&](auto spaceTag) {
      using S = decltype(spaceTag);
      py::array_t<int> faces({(int)S::nFaces, (int)S::nFaceVertices});
      for (int k = 0; k < S::nFaces; ++k)
        for (int m = 0; m < S::nFaceVertices; ++m) faces.mutable_data()[k * S::nFaceVertices + m] = S::faceVertex(k, m);
      return py::make_tuple((int)S::nDOF, (int)S::nVertices, faces);
    });
  });
  // the space's isoparametric geometry at reference points: x (nE, nQ, 3) and |det J| (nE, nQ), for the driver's norms
  m.def("quadrature_geometry", [](int nd, const std::string &space, Arr xn, Arr xi) {
    return dispatchSpace(nd, space, [&](auto spaceTag) {
      using S = decltype(spaceTag);
      constexpr int nS = S::nS, nD = S::nDOF;
      const int nE = xn.shape(0), nQ = xi.shape(0);
      py::array_t<double> x({nE, nQ, 3}), adet({nE, nQ});
      adrh::PointGeometry<S> G;
      for (int e = 0; e < nE; ++e)
        for (int q = 0; q < nQ; ++q)
        {
          G.compute(xn.data() + e * nD * 3, xi.data() + q * nS);
          for (int I = 0; I < 3; ++I) x.mutable_data()[(e * nQ + q) * 3 + I] = G.x[I];
          adet.mutable_data()[e * nQ + q] = G.adet;
        }
      return py::make_tuple(x, adet);
    });
  });
  // transient linear ADR u_t + div(b u - kappa grad u) + c u = 0, backward Euler: element residual and Jacobian at
  // time t with the step dt and the old solution uold (nE, nc, nDOF)
  m.def("transient_adr", [](int nd, const std::string &space, Arr b, double kappa, double c,
                             Arr xn, Arr u, Arr uold, Arr qp, Arr qw, double t, double dt) {
    return dispatchSpace(nd, space, [&](auto spaceTag) {
      using S = decltype(spaceTag);
      constexpr int nS = S::nS, nD = S::nDOF;
      adrh::TransientLinearADR<nS> co(b.data(), kappa, c);
      const int nE = xn.shape(0), nQ = qw.shape(0);
      py::array_t<double> R({nE, 1, nD}), J({nE, 1, nD, 1, nD});
      {
        py::gil_scoped_release release;
        adrh::elementKernelSpace<adrh::TransientLinearADR<nS>, S>(co, nE, nQ, xn.data(), u.data(), uold.data(),
                                                                  qp.data(), qw.data(), t, dt,
                                                                  R.mutable_data(), J.mutable_data());
      }
      return py::make_tuple(R, J);
    });
  });
  // its exterior faces at time t, with the boundary data from the Dirac-IC Gaussian (tStart, u0, x0) and the
  // same face kinds as advection_diffusion_sine_faces
  m.def("transient_adr_faces", [](int nd, const std::string &space, Arr b, double kappa, double c,
                                   double tStart, double u0, Arr x0,
                                   py::array_t<int, py::array::c_style | py::array::forcecast> faceLocal,
                                   py::array_t<int, py::array::c_style | py::array::forcecast> faceKind,
                                   Arr xn, Arr u, Arr qpf, Arr qwf, double t,
                                   double penalty_constant, double penalty_power, double sigma,
                                   double robin_alpha, double robin_beta) {
    return dispatchSpace(nd, space, [&](auto spaceTag) {
      using S = decltype(spaceTag);
      constexpr int nS = S::nS, nD = S::nDOF;
      adrh::TransientLinearADR<nS> co(b.data(), kappa, c);
      adrh::DiracExact<nS> exact(b.data(), kappa, c, tStart, u0, x0.data());
      adrh::ExactBC<nS, adrh::DiracExact<nS>> bc(exact, b.data(), kappa, faceKind.data(), robin_alpha, robin_beta);
      adrh::WeakDirichletParams wd{penalty_constant, penalty_power, sigma};
      const int nF = xn.shape(0), nQf = qwf.shape(0);
      py::array_t<double> Rf({nF, 1, nD}), Jf({nF, 1, nD, 1, nD});
      {
        py::gil_scoped_release release;
        adrh::exteriorFaceKernel<adrh::TransientLinearADR<nS>, adrh::ExactBC<nS, adrh::DiracExact<nS>>, S>(
            co, bc, wd, nF, nQf, faceLocal.data(), xn.data(), u.data(), qpf.data(), qwf.data(), t,
            Rf.mutable_data(), Jf.mutable_data());
      }
      return py::make_tuple(Rf, Jf);
    });
  });
  // basis values of the space at reference points xi (nQ, nd): (nQ, nDOF), for the driver's error norms
  m.def("basis_values", [](int nd, const std::string &space, Arr xi) {
    return dispatchSpace(nd, space, [&](auto spaceTag) {
      using S = decltype(spaceTag);
      constexpr int nS = S::nS, nD = S::nDOF;
      const int nQ = xi.shape(0);
      py::array_t<double> W({nQ, nD});
      double gw[nD][nS];
      for (int q = 0; q < nQ; ++q) S::basis(xi.data() + q * nS, W.mutable_data() + q * nD, gw);
      return py::make_tuple(W);
    });
  });
  // Richards' equation in pressure-head form (richards.h): element residual and Jacobian with the backward Euler mass term, for nm
  // materials (alpha, n_vg, thetaR, thetaSR of length nm; Ks (nm, nd), the diagonal conductivities) and a material number per element
  using IArrR = py::array_t<int, py::array::c_style | py::array::forcecast>;
  m.def("richards_vgm", [](int nd, const std::string &space, Arr alpha, Arr n_vg, Arr thetaR, Arr thetaSR, Arr Ks, Arr g, double rho,
                            double beta, IArrR material, Arr xn, Arr u, py::object uold, Arr qp, Arr qw, double t, double dt, int psk, int mode) {
    return dispatchSpace(nd, space, [&](auto spaceTag) {
      using S = decltype(spaceTag);
      constexpr int nS = S::nS, nD = S::nDOF;
      adrh::RichardsMaterials<nS> co((int)alpha.size(), alpha.data(), n_vg.data(), thetaR.data(), thetaSR.data(), Ks.data(), g.data(),
                                     material.data(), rho, beta, 1.0, psk, mode);
      const int nE = xn.shape(0), nQ = qw.shape(0);
      Arr uo;
      const double *uop = nullptr;
      if (!uold.is_none() && dt > 0.0) { uo = uold.cast<Arr>(); uop = uo.data(); }
      py::array_t<double> R({nE, 1, nD}), J({nE, 1, nD, 1, nD});
      {
        py::gil_scoped_release release;
        adrh::elementKernelSpace<adrh::RichardsMaterials<nS>, S>(co, nE, nQ, xn.data(), u.data(), uop, qp.data(), qw.data(), t, dt,
                                                                 R.mutable_data(), J.mutable_data());
      }
      return py::make_tuple(R, J);
    });
  });
  // its exterior faces: weak Dirichlet data gD (one value per face) as proteus's Richards_IIPG_exterior, the advective flux from the
  // interior state, the penalty penalty_constant / h^penalty_power (h = 1 for a point), no adjoint term; faceElem the face's element
  m.def("richards_vgm_faces", [](int nd, const std::string &space, Arr alpha, Arr n_vg, Arr thetaR, Arr thetaSR, Arr Ks, Arr g, double rho,
                                  double beta, IArrR material, IArrR faceElem, IArrR faceLocal, Arr gD, Arr xn, Arr u, Arr qpf, Arr qwf,
                                  double t, double penalty_constant, double penalty_power, int psk, int mode) {
    return dispatchSpace(nd, space, [&](auto spaceTag) {
      using S = decltype(spaceTag);
      constexpr int nS = S::nS, nD = S::nDOF;
      adrh::RichardsMaterials<nS> co((int)alpha.size(), alpha.data(), n_vg.data(), thetaR.data(), thetaSR.data(), Ks.data(), g.data(),
                                     material.data(), rho, beta, 1.0, psk, mode);
      adrh::FixedValueBC bc(gD.data());
      adrh::WeakDirichletParams wd{penalty_constant, penalty_power, 0.0};
      const int nF = xn.shape(0), nQf = qwf.shape(0);
      py::array_t<double> Rf({nF, 1, nD}), Jf({nF, 1, nD, 1, nD});
      {
        py::gil_scoped_release release;
        adrh::exteriorFaceKernel<adrh::RichardsMaterials<nS>, adrh::FixedValueBC, S>(
            co, bc, wd, nF, nQf, faceLocal.data(), xn.data(), u.data(), qpf.data(), qwf.data(), t, Rf.mutable_data(), Jf.mutable_data(),
            faceElem.data());
      }
      return py::make_tuple(Rf, Jf);
    });
  });
  // ---- the K-upwinded low-order operator (lowOrder.h) ----
  using IArrL = py::array_t<int, py::array::c_style | py::array::forcecast>;
  // nodal relative permeability and its derivative with respect to u: material number per node, parameters per material
  m.def("richards_kr_nodes", [](Arr alpha, Arr n_vg, Arr thetaR, Arr thetaSR, int psk, IArrL nodeMat, Arr u) {
    const int n = (int)u.size();
    py::array_t<double> kr(n), dkr(n);
    for (int i = 0; i < n; ++i)
    {
      const int s = nodeMat.data()[i];
      adrh::krNode(u.data()[i], alpha.data()[s], n_vg.data()[s], thetaR.data()[s], thetaSR.data()[s], psk, kr.mutable_data()[i], dkr.mutable_data()[i]);
    }
    return py::make_tuple(kr, dkr);
  });
  // nodal storage m = rho theta_w and dm/du with each node's own material (the low-order scheme's lumped mass is nodal)
  m.def("richards_m_nodes", [](Arr alpha, Arr n_vg, Arr thetaR, Arr thetaSR, int psk, IArrL nodeMat, Arr u, double rho, double beta) {
    const int n = (int)u.size();
    py::array_t<double> mm(n), dm(n);
    for (int i = 0; i < n; ++i)
    {
      const int s = nodeMat.data()[i];
      adrh::mNode(u.data()[i], alpha.data()[s], n_vg.data()[s], thetaR.data()[s], thetaSR.data()[s], psk, rho, beta, mm.mutable_data()[i], dm.mutable_data()[i]);
    }
    return py::make_tuple(mm, dm);
  }, py::arg("alpha"), py::arg("n_vg"), py::arg("thetaR"), py::arg("thetaSR"), py::arg("psk"), py::arg("nodeMat"), py::arg("u"), py::arg("rho") = 1.0,
     py::arg("beta") = 0.0);
  // the flux part of the residual (r) and its Jacobian values on the pattern: S the stiffness (mode 1), phi the total head, kr/dkr nodal
  m.def("upwind_flux", [](IArrL indptr, IArrL indices, Arr S, Arr phi, Arr kr, Arr dkr) {
    const int n = (int)indptr.size() - 1;
    py::array_t<double> r(n), J(S.size());
    for (int i = 0; i < n; ++i)
      adrh::upwindFluxRow(i, indptr.data(), indices.data(), S.data(), phi.data(), kr.data(), dkr.data(), r.mutable_data(), J.mutable_data());
    return py::make_tuple(r, J);
  });
  m.def("upwind_flux_edges", [](IArrL indptr, IArrL indices, Arr S, Arr dphi, Arr ddi, Arr ddj, Arr kr, Arr dkr) {
    const int n = (int)indptr.size() - 1;
    py::array_t<double> r(n), J(S.size());
    for (int i = 0; i < n; ++i)
      adrh::upwindFluxRowE(i, indptr.data(), indices.data(), S.data(), dphi.data(), ddi.data(), ddj.data(), kr.data(), dkr.data(), r.mutable_data(), J.mutable_data());
    return py::make_tuple(r, J);
  });
  m.def("antidiffusive_flux_edges", [](IArrL indptr, IArrL indices, Arr C, Arr S, Arr dphi, Arr kr) {
    const int n = (int)indptr.size() - 1;
    py::array_t<double> A(S.size());
    for (int i = 0; i < n; ++i)
      adrh::antidiffusiveFluxRowE(i, indptr.data(), indices.data(), C.data(), S.data(), dphi.data(), kr.data(), A.mutable_data());
    return A;
  });
  m.def("antidiffusive_flux", [](IArrL indptr, IArrL indices, Arr C, Arr S, Arr phi, Arr kr) {
    const int n = (int)indptr.size() - 1;
    py::array_t<double> A(S.size());
    for (int i = 0; i < n; ++i)
      adrh::antidiffusiveFluxRow(i, indptr.data(), indices.data(), C.data(), S.data(), phi.data(), kr.data(), A.mutable_data());
    return A;
  });
  // bounds_old: take the local extrema over mold as well as mL (False: over mL alone, as proteus's FCTStep does when it is not monolithic)
  m.def("fct_limit_flux", [](IArrL indptr, IArrL indices, IArrL tpos, Arr Mc, Arr A, Arr ML, Arr mL, Arr mold, double dt, bool bounds_old) {
    py::array_t<double> out(mL.size());
    adrh::fctLimitFlux((int)indptr.size() - 1, indptr.data(), indices.data(), tpos.data(), Mc.data(), A.data(), ML.data(), mL.data(), mold.data(),
                       bounds_old ? mold.data() : mL.data(), dt, out.mutable_data());
    return out;
  }, py::arg("indptr"), py::arg("indices"), py::arg("tpos"), py::arg("Mc"), py::arg("A"), py::arg("ML"), py::arg("mL"), py::arg("mold"), py::arg("dt"),
     py::arg("bounds_old") = true);
  // FCT with the bounds given (the head's local extrema mapped to moisture content by each node's own curve; see headExtremaRow)
  m.def("head_extrema", [](IArrL indptr, IArrL indices, Arr a, Arr b) {
    const int n = (int)indptr.size() - 1;
    py::array_t<double> pmax(n), pmin(n);
    for (int i = 0; i < n; ++i) adrh::headExtremaRow(i, indptr.data(), indices.data(), a.data(), b.data(), pmax.mutable_data(), pmin.mutable_data());
    return py::make_tuple(pmax, pmin);
  });
  m.def("fct_limit_flux_bounded", [](IArrL indptr, IArrL indices, IArrL tpos, Arr Mc, Arr A, Arr ML, Arr mL, Arr mold, Arr mmax, Arr mmin, double dt) {
    py::array_t<double> out(mL.size());
    adrh::fctLimitFluxBounded((int)indptr.size() - 1, indptr.data(), indices.data(), tpos.data(), Mc.data(), A.data(), ML.data(), mL.data(), mold.data(),
                              mmax.data(), mmin.data(), dt, out.mutable_data());
    return out;
  });
  // implicit FCT (lowOrder.h): the nodal-variation prelimiter and the limited datum s^{n,FCT}
  m.def("nodal_variation", [](IArrL indptr, IArrL indices, Arr Mc, Arr head, double power) {
    const int n = (int)indptr.size() - 1;
    py::array_t<double> alpha(n);
    for (int i = 0; i < n; ++i) adrh::nodalVariationRow(i, indptr.data(), indices.data(), Mc.data(), head.data(), power, alpha.mutable_data());
    return alpha;
  });
  m.def("implicit_fct_limit", [](IArrL indptr, IArrL indices, IArrL tpos, Arr Mc, Arr A, Arr ML, Arr sH, Arr sn, Arr smax, Arr smin, Arr alpha, double dt) {
    py::array_t<double> datum(sH.size());
    adrh::implicitFctLimit((int)indptr.size() - 1, indptr.data(), indices.data(), tpos.data(), Mc.data(), A.data(), ML.data(), sH.data(), sn.data(),
                           smax.data(), smin.data(), alpha.data(), dt, datum.mutable_data());
    return datum;
  });
  // ---- algebraic stabilization (lowOrder.h) ----
  m.def("transpose_map", [](py::array_t<int, py::array::c_style | py::array::forcecast> indptr, py::array_t<int, py::array::c_style | py::array::forcecast> indices) {
    const std::vector<int> t = adrh::buildTransposeMap((int)indptr.size() - 1, indptr.data(), indices.data());
    py::array_t<int> out((py::ssize_t)t.size());
    std::copy(t.begin(), t.end(), out.mutable_data());
    return out;
  });
  // the discrete upwinding of lowOrder.h applied to the residual r and the CSR values of the spatial Jacobian: returns (values, r)
  m.def("algebraic_dissipation", [](py::array_t<int, py::array::c_style | py::array::forcecast> indptr,
                                     py::array_t<int, py::array::c_style | py::array::forcecast> indices,
                                     py::array_t<int, py::array::c_style | py::array::forcecast> tpos, Arr values, Arr u, Arr r) {
    py::array_t<double> J(values.size()), R(r.size());
    std::copy(values.data(), values.data() + values.size(), J.mutable_data());
    std::copy(r.data(), r.data() + r.size(), R.mutable_data());
    adrh::algebraicDissipation((int)indptr.size() - 1, indptr.data(), indices.data(), tpos.data(), J.mutable_data(), u.data(), R.mutable_data());
    return py::make_tuple(J, R);
  });
  // lumped mass of every element: (w, m, dm, mold), each (nE, nDOF)
  m.def("richards_vgm_lumped", [](int nd, const std::string &space, Arr alpha, Arr n_vg, Arr thetaR, Arr thetaSR, Arr Ks, Arr g, double rho,
                                   double beta, py::array_t<int, py::array::c_style | py::array::forcecast> material, Arr xn, Arr u,
                                   py::object uold, Arr qp, Arr qw, double t, double dt, int psk, int mode) {
    return dispatchSpace(nd, space, [&](auto spaceTag) {
      using S = decltype(spaceTag);
      constexpr int nD = S::nDOF;
      adrh::RichardsMaterials<S::nS> co((int)alpha.size(), alpha.data(), n_vg.data(), thetaR.data(), thetaSR.data(), Ks.data(), g.data(),
                                        material.data(), rho, beta, 1.0, psk, mode);
      const int nE = xn.shape(0), nQ = qw.shape(0);
      Arr uo;
      const double *uop = nullptr;
      if (!uold.is_none() && dt > 0.0) { uo = uold.cast<Arr>(); uop = uo.data(); }
      py::array_t<double> w({nE, nD}), mm({nE, nD}), dm({nE, nD}), mold({nE, nD});
      adrh::lumpedMassKernel<adrh::RichardsMaterials<S::nS>, S>(co, nE, nQ, xn.data(), u.data(), uop, qp.data(), qw.data(), t, dt,
                                                                w.mutable_data(), mm.mutable_data(), dm.mutable_data(), mold.mutable_data());
      return py::make_tuple(w, mm, dm, mold);
    });
  });
  m.def("transient_adr_lumped", [](int nd, const std::string &space, Arr b, double kappa, double c, Arr xn, Arr u, py::object uold, Arr qp,
                                    Arr qw, double t, double dt) {
    return dispatchSpace(nd, space, [&](auto spaceTag) {
      using S = decltype(spaceTag);
      constexpr int nD = S::nDOF;
      adrh::TransientLinearADR<S::nS> co(b.data(), kappa, c);
      const int nE = xn.shape(0), nQ = qw.shape(0);
      Arr uo;
      const double *uop = nullptr;
      if (!uold.is_none() && dt > 0.0) { uo = uold.cast<Arr>(); uop = uo.data(); }
      py::array_t<double> w({nE, nD}), mm({nE, nD}), dm({nE, nD}), mold({nE, nD});
      adrh::lumpedMassKernel<adrh::TransientLinearADR<S::nS>, S>(co, nE, nQ, xn.data(), u.data(), uop, qp.data(), qw.data(), t, dt,
                                                                 w.mutable_data(), mm.mutable_data(), dm.mutable_data(), mold.mutable_data());
      return py::make_tuple(w, mm, dm, mold);
    });
  });
  using IArrF = py::array_t<int, py::array::c_style | py::array::forcecast>;
  m.def("dissipation_values", [](IArrF indptr, IArrF indices, IArrF tpos, Arr values) {
    py::array_t<double> d(values.size());
    adrh::dissipationValues((int)indptr.size() - 1, indptr.data(), indices.data(), tpos.data(), values.data(), d.mutable_data());
    return d;
  });
  m.def("consistent_mass", [](int nd, const std::string &space, Arr xn, Arr qp, Arr qw) {
    return dispatchSpace(nd, space, [&](auto spaceTag) {
      using S = decltype(spaceTag);
      constexpr int nD = S::nDOF;
      const int nE = xn.shape(0), nQ = qw.shape(0);
      py::array_t<double> M({nE, nD, nD});
      adrh::massMatrixKernel<S>(nE, nQ, xn.data(), qp.data(), qw.data(), M.mutable_data());
      return py::make_tuple(M);
    });
  });
  // vL: the variable the dissipation acts on; mL, mold: the conserved variable at the low-order solution and at the old time; returns the
  // limited conserved variable (for a linear storage term all three are u)
  m.def("fct_limit", [](IArrF indptr, IArrF indices, IArrF tpos, Arr Mc, Arr d, Arr ML, Arr vL, Arr mL, Arr mold, double dt) {
    py::array_t<double> out(mL.size());
    adrh::fctLimit((int)indptr.size() - 1, indptr.data(), indices.data(), tpos.data(), Mc.data(), d.data(), ML.data(), vL.data(), mL.data(),
                   mold.data(), dt, out.mutable_data());
    return out;
  });
  m.def("poisson3d_masked", [](Arr xn, Arr u, Arr qp, Arr qw) {
    return runMasked<adrh::Poisson3D, 3>(adrh::Poisson3D(), xn, u, py::none(), qp, qw, 0.0, 0.0);
  });
  m.def("coupled2_masked", [](Arr xn, Arr u, py::object uold, Arr qp, Arr qw, double t, double dt) {
    return runMasked<adrh::Coupled2, 3>(adrh::Coupled2(), xn, u, uold, qp, qw, t, dt);
  });
  // chain5: (masked kernel | runtime-loop kernel) x (sparse masks | all-true masks)
  m.def("chain5", [](Arr xn, Arr u, py::object uold, Arr qp, Arr qw, double t, double dt, bool masked, bool sparse) {
    if (masked && sparse) return runMasked<adrh::Chain5<true>, 3>(adrh::Chain5<true>(), xn, u, uold, qp, qw, t, dt);
    if (masked) return runMasked<adrh::Chain5<false>, 3>(adrh::Chain5<false>(), xn, u, uold, qp, qw, t, dt);
    return run<adrh::Chain5<false>, 3>(adrh::Chain5<false>(), xn, u, uold, qp, qw, t, dt);
  });
  m.def("poisson3d", [](Arr xn, Arr u, Arr qp, Arr qw) {
    return run<adrh::Poisson3D, 3>(adrh::Poisson3D(), xn, u, py::none(), qp, qw, 0.0, 0.0);
  });
  m.def("coupled2", [](Arr xn, Arr u, py::object uold, Arr qp, Arr qw, double t, double dt) {
    return run<adrh::Coupled2, 3>(adrh::Coupled2(), xn, u, uold, qp, qw, t, dt);
  });
  // VOF (vof.h): the SUPG element kernel and the upwind exterior faces, P1 in 2 or 3 dimensions. Element-local inputs:
  // xn (nE, nS+1, 3), u (nE, nS+1); per point fields (nE, nQ, nS+1) = velocity and porosity, beta (nE, nQ) the BDF history,
  // numDiffLast (nE, nQ); h (nE). Returns R (nE, nD), J (nE, nD, nD), numDiff (nE, nQ), cfl (nE, nQ).
  using IArrV = py::array_t<int, py::array::c_style | py::array::forcecast>;
  m.def("vof_supg", [](int nd, Arr xn, Arr u, Arr qp, Arr qw, Arr fields, Arr beta, Arr numDiffLast, Arr h,
                       double alpha, double sc_c, double sc_uref, double sc_beta, double sc_drop_time) {
    return dispatchSpace(nd, "p1", [&](auto spaceTag) {
      using S = decltype(spaceTag);
      constexpr int nS = S::nS, nD = S::nDOF;
      using C = adrh::VOFCoefficients<nS>;
      const int nE = xn.shape(0), nQ = qw.shape(0);
      const adrh::SUPGParameters P{alpha, sc_c, sc_uref, sc_beta, sc_drop_time, 4.0};
      py::array_t<double> R({nE, nD}), J({nE, nD, nD}), nd_({nE, nQ}), cfl({nE, nQ});
      double *Rp = R.mutable_data(), *Jp = J.mutable_data(), *np = nd_.mutable_data(), *cp = cfl.mutable_data();
      const double *X = xn.data(), *U = u.data(), *F = fields.data(), *B = beta.data(), *NL = numDiffLast.data(), *H = h.data();
      {
        py::gil_scoped_release release;
        for (int eN = 0; eN < nE; ++eN)
          adrh::scalarSUPGElement<C, S>(C(), P, X + eN * nD * 3, U + eN * nD, nQ, qp.data(), qw.data(), F + eN * nQ * C::nFields,
                                        B + eN * nQ, NL + eN * nQ, H[eN], Rp + eN * nD, Jp + eN * nD * nD, np + eN * nQ, cp + eN * nQ);
      }
      return py::make_tuple(R, J, nd_, cfl);
    });
  });
  // exterior faces: per face f the element-local xn (nF, nD, 3), u (nF, nD), the local face number faceLocal (nF); xi (nD, nQf, nS)
  // the face points of each local face in reference coordinates, qwf (nQf); per point fields (nF, nQf, nS+1), isDirichlet, g,
  // isFlux, gflux (nF, nQf). Returns R (nF, nD), J (nF, nD, nD), flux and the upwinded trace (nF, nQf).
  m.def("vof_faces", [](int nd, Arr xn, Arr u, IArrV faceLocal, Arr xi, Arr qwf, Arr fields, IArrV isDirichlet, Arr g,
                        IArrV isFlux, Arr gflux) {
    return dispatchSpace(nd, "p1", [&](auto spaceTag) {
      using S = decltype(spaceTag);
      constexpr int nS = S::nS, nD = S::nDOF;
      using C = adrh::VOFCoefficients<nS>;
      const int nF = xn.shape(0), nQf = qwf.shape(0);
      py::array_t<double> R({nF, nD}), J({nF, nD, nD}), flux({nF, nQf}), uUp({nF, nQf});
      double *Rp = R.mutable_data(), *Jp = J.mutable_data();
      for (int i = 0; i < nF * nD; ++i) Rp[i] = 0.0;
      for (int i = 0; i < nF * nD * nD; ++i) Jp[i] = 0.0;
      const int *fl = faceLocal.data();
      for (int f = 0; f < nF; ++f)
        adrh::scalarUpwindFace<C, S>(C(), xn.data() + f * nD * 3, u.data() + f * nD, fl[f], nQf, xi.data() + fl[f] * nQf * nS, qwf.data(),
                                     fields.data() + f * nQf * C::nFields, isDirichlet.data() + f * nQf, g.data() + f * nQf,
                                     isFlux.data() + f * nQf, gflux.data() + f * nQf, Rp + f * nD, Jp + f * nD * nD,
                                     flux.mutable_data() + f * nQf, uUp.mutable_data() + f * nQf);
      return py::make_tuple(R, J, flux, uUp);
    });
  });

  // NCLS (ncls.h): the same element kernel with H = v . grad u, fields (nE, nQ, nS) = the velocity; returns R, J, numDiff, cfl
  // as vof_supg does.
  m.def("ncls_supg", [](int nd, Arr xn, Arr u, Arr qp, Arr qw, Arr fields, Arr beta, Arr numDiffLast, Arr h,
                        double alpha, double sc_c, double sc_uref, double sc_beta, double sc_drop_time) {
    return dispatchSpace(nd, "p1", [&](auto spaceTag) {
      using S = decltype(spaceTag);
      constexpr int nS = S::nS, nD = S::nDOF;
      using C = adrh::NCLSCoefficients<nS>;
      const int nE = xn.shape(0), nQ = qw.shape(0);
      const adrh::SUPGParameters P{alpha, sc_c, sc_uref, sc_beta, sc_drop_time, 4.0};
      py::array_t<double> R({nE, nD}), J({nE, nD, nD}), nd_({nE, nQ}), cfl({nE, nQ});
      py::array_t<int> cut(nE);
      double *Rp = R.mutable_data(), *Jp = J.mutable_data(), *np = nd_.mutable_data(), *cp = cfl.mutable_data();
      int *cutp = cut.mutable_data();
      const double *X = xn.data(), *U = u.data(), *F = fields.data(), *B = beta.data(), *NL = numDiffLast.data(), *H = h.data();
      {
        py::gil_scoped_release release;
        for (int eN = 0; eN < nE; ++eN)
        {
          adrh::scalarSUPGElement<C, S>(C(), P, X + eN * nD * 3, U + eN * nD, nQ, qp.data(), qw.data(), F + eN * nQ * C::nFields,
                                        B + eN * nQ, NL + eN * nQ, H[eN], Rp + eN * nD, Jp + eN * nD * nD, np + eN * nQ, cp + eN * nQ);
          cutp[eN] = adrh::signChangeElement<nD>(U + eN * nD) ? 1 : 0;
        }
      }
      return py::make_tuple(R, J, nd_, cfl, cut);
    });
  });
  // NCLS exterior faces: as vof_faces, with rd (nF, nQf) the redistanced trace in place of the flux condition. Returns R, J,
  // the trace u and its normalized gradient (nF, nQf, nS).
  m.def("ncls_faces", [](int nd, Arr xn, Arr u, IArrV faceLocal, Arr xi, Arr qwf, Arr fields, IArrV isDirichlet, Arr g, Arr rd) {
    return dispatchSpace(nd, "p1", [&](auto spaceTag) {
      using S = decltype(spaceTag);
      constexpr int nS = S::nS, nD = S::nDOF;
      using C = adrh::NCLSCoefficients<nS>;
      const int nF = xn.shape(0), nQf = qwf.shape(0);
      py::array_t<double> R({nF, nD}), J({nF, nD, nD}), uT({nF, nQf}), gN({nF, nQf, nS});
      double *Rp = R.mutable_data(), *Jp = J.mutable_data();
      for (int i = 0; i < nF * nD; ++i) Rp[i] = 0.0;
      for (int i = 0; i < nF * nD * nD; ++i) Jp[i] = 0.0;
      const int *fl = faceLocal.data();
      for (int f = 0; f < nF; ++f)
        adrh::scalarInflowPenaltyFace<C, S>(C(), xn.data() + f * nD * 3, u.data() + f * nD, fl[f], nQf, xi.data() + fl[f] * nQf * nS,
                                            qwf.data(), fields.data() + f * nQf * C::nFields, isDirichlet.data() + f * nQf,
                                            g.data() + f * nQf, rd.data() + f * nQf, Rp + f * nD, Jp + f * nD * nD,
                                            uT.mutable_data() + f * nQf, gN.mutable_data() + f * nQf * nS);
      return py::make_tuple(R, J, uT, gN);
    });
  });

  // RDLS (rdls.h): the same element kernel with H = S |grad u|, r = -S, no time term, shock capturing at the current u, and
  // the interface pin. Per element xn (nE, nD, 3), u and uLast (nE, nD); per point fields (nE, nQ, 3) = (phi, h_phi, h_e);
  // h (nE) the element diameter. Returns R (nE, nD), J (nE, nD, nD), numDiff, cfl (nE, nQ) and the pinned count per element.
  m.def("rdls_supg", [](int nd, Arr xn, Arr u, Arr uLast, Arr qp, Arr qw, Arr fields, Arr h, double sc_c, double epsFact,
                        double backgroundDiffusionFactor, double weakDirichletFactor) {
    return dispatchSpace(nd, "p1", [&](auto spaceTag) {
      using S = decltype(spaceTag);
      constexpr int nS = S::nS, nD = S::nDOF;
      using C = adrh::RDLSCoefficients<nS>;
      const C coef{epsFact, backgroundDiffusionFactor};
      const int nE = xn.shape(0), nQ = qw.shape(0);
      adrh::SUPGParameters P{0.0, sc_c, 1.0, 1.0, 0.0, 4.0};
      P.lag_sc = 0;
      py::array_t<double> R({nE, nD}), J({nE, nD, nD}), nd_({nE, nQ}), cfl({nE, nQ});
      py::array_t<int> pinned(nE);
      double *Rp = R.mutable_data(), *Jp = J.mutable_data(), *np = nd_.mutable_data(), *cp = cfl.mutable_data();
      int *pp = pinned.mutable_data();
      const double *X = xn.data(), *U = u.data(), *UL = uLast.data(), *F = fields.data(), *H = h.data();
      std::vector<double> zeros(nQ, 0.0);
      {
        py::gil_scoped_release release;
        for (int eN = 0; eN < nE; ++eN)
        {
          const double *Fe = F + eN * nQ * C::nFields;
          adrh::scalarSUPGElement<C, S>(coef, P, X + eN * nD * 3, U + eN * nD, nQ, qp.data(), qw.data(), Fe, zeros.data(),
                                        zeros.data(), H[eN], Rp + eN * nD, Jp + eN * nD * nD, np + eN * nQ, cp + eN * nQ);
          const double epsLast = epsFact * Fe[(nQ - 1) * C::nFields + 1];
          pp[eN] = adrh::rdlsPinElement<nD>(U + eN * nD, UL + eN * nD, epsLast, weakDirichletFactor, H[eN], Rp + eN * nD,
                                            Jp + eN * nD * nD);
        }
      }
      return py::make_tuple(R, J, nd_, cfl, pinned);
    });
  });
  // RDLS exterior faces: the trace and its normalized gradient only (no boundary terms). Returns uT (nF, nQf), gN (nF, nQf, nS).
  m.def("rdls_faces", [](int nd, Arr xn, Arr u, IArrV faceLocal, Arr xi) {
    return dispatchSpace(nd, "p1", [&](auto spaceTag) {
      using S = decltype(spaceTag);
      constexpr int nS = S::nS, nD = S::nDOF;
      const int nF = xn.shape(0), nQf = xi.shape(1);
      py::array_t<double> uT({nF, nQf}), gN({nF, nQf, nS});
      const int *fl = faceLocal.data();
      for (int f = 0; f < nF; ++f)
        adrh::rdlsTraceFace<S>(xn.data() + f * nD * 3, u.data() + f * nD, nQf, xi.data() + fl[f] * nQf * nS,
                               uT.mutable_data() + f * nQf, gN.mutable_data() + f * nQf * nS);
      return py::make_tuple(uT, gN);
    });
  });
  // MCorr (mcorr.h): the generic element kernel (engine.h) with MCorrCoefficients, steady. Per element xn (nE, nD, 3), u (nE, nD);
  // per point fields (nE, nQ, 6) = (phi, H_vof, porosity, h_phi, h_e, phi_s). Returns R (nE, nD), J (nE, nD, nD) and the point
  // outputs q_u, q_r (nE, nQ), q_n (nE, nQ, nS).
  m.def("mcorr", [](int nd, Arr xn, Arr u, Arr qp, Arr qw, Arr fields, double useMetrics, double epsFactHeaviside,
                    double epsFactDirac, double epsFactDiffusion) {
    return dispatchSpace(nd, "p1", [&](auto spaceTag) {
      using S = decltype(spaceTag);
      constexpr int nS = S::nS, nD = S::nDOF;
      using C = adrh::MCorrCoefficients<nS>;
      const int nE = xn.shape(0), nQ = qw.shape(0);
      if (fields.ndim() != 3 || fields.shape(0) != nE || fields.shape(1) != nQ || fields.shape(2) != C::nFields)
        throw std::invalid_argument("mcorr: fields must be (nE, nQ, 6)");
      C coef;
      coef.useMetrics = useMetrics; coef.epsFactHeaviside = epsFactHeaviside; coef.epsFactDirac = epsFactDirac;
      coef.epsFactDiffusion = epsFactDiffusion; coef.fields = fields.data(); coef.nQ = nQ;
      py::array_t<double> R({nE, nD}), J({nE, nD, nD}), qu({nE, nQ}), qr({nE, nQ}), qn({nE, nQ, nS});
      {
        py::gil_scoped_release release;
        adrh::elementKernelSpace<C, S>(coef, nE, nQ, xn.data(), u.data(), nullptr, qp.data(), qw.data(), 0.0, 0.0,
                                       R.mutable_data(), J.mutable_data());
        for (int eN = 0; eN < nE; ++eN)
          adrh::mcorrPointOutputs<S>(coef.bindElement(eN), xn.data() + eN * nD * 3, u.data() + eN * nD, nQ, qp.data(),
                                     qu.mutable_data() + eN * nQ, qr.mutable_data() + eN * nQ, qn.mutable_data() + eN * nQ * nS);
      }
      return py::make_tuple(R, J, qu, qr, qn);
    });
  });

  // MCorr with a solid (mcorr.h steps 1-3, MCorr.h's cut-cell path): mcorr's arguments, then MCorr.h's solid data: xref
  // (nQ, 3) the reference points as MCorr.h hands them to equivalent_polynomials, phi_s at the nodes (nN) and at the points
  // (nE, nQ), useExact_s, the mesh's l2g (nE, nD), element faces (nE, nD), face elements and local face numbers (nEB, 2),
  // face diameters (nEB) and gamma = 10 ghost_penalty_constant. Returns the element residuals and Jacobians (H_s from
  // the solid), the point outputs, icase (nE), H_s (nE, nQ), the cut faces kept (nF) and their ghost-penalty residuals
  // (nF, 2, nD) and Jacobian blocks (nF, 2, nD, 2, nD), side 0 being the face's first element.
  m.def("mcorr_solid", [](int nd, Arr xn, Arr u, Arr qp, Arr qw, Arr fields, double useMetrics, double epsFactHeaviside,
                          double epsFactDirac, double epsFactDiffusion, Arr xref, Arr phi_s, Arr phi_q, int useExact_s,
                          IArrV l2g, IArrV elementBoundaries, IArrV boundaryElements, IArrV boundaryLocal, Arr boundaryDiameter,
                          double gamma) {
    return dispatchSpace(nd, "p1", [&](auto spaceTag) {
      using S = decltype(spaceTag);
      constexpr int nS = S::nS, nD = S::nDOF;
      using C = adrh::MCorrCoefficients<nS>;
      const int nE = xn.shape(0), nQ = qw.shape(0);
      if (fields.ndim() != 3 || fields.shape(0) != nE || fields.shape(1) != nQ || fields.shape(2) != C::nFields)
        throw std::invalid_argument("mcorr_solid: fields must be (nE, nQ, 6)");
      if (xref.ndim() != 2 || xref.shape(0) != nQ || xref.shape(1) != 3 || phi_q.shape(0) != nE || phi_q.shape(1) != nQ)
        throw std::invalid_argument("mcorr_solid: xref must be (nQ, 3) and phi_q (nE, nQ)");
      if constexpr (nS == 1)
        throw std::invalid_argument("mcorr_solid: 2D and 3D only");
      py::array_t<int> icase(nE);
      py::array_t<double> Hs({nE, nQ});
      auto solid = [&](auto nQtag) {
        constexpr int NQ = decltype(nQtag)::value;
        adrh::MCorrSolid<nS, NQ> gf(useExact_s != 0);
        for (int eN = 0; eN < nE; ++eN)
        {
          double phi_e[nD];
          for (int a = 0; a < nD; ++a) phi_e[a] = phi_s.data()[l2g.data()[eN * nD + a]];
          icase.mutable_data()[eN] = gf.element(xn.data() + eN * nD * 3, phi_e, xref.data(), phi_q.data() + eN * nQ,
                                                Hs.mutable_data() + eN * nQ);
        }
      };
      switch (nQ)
      {
        case 3: solid(std::integral_constant<int, 3>{}); break;
        case 4: solid(std::integral_constant<int, 4>{}); break;
        case 5: solid(std::integral_constant<int, 5>{}); break;
        case 6: solid(std::integral_constant<int, 6>{}); break;
        case 7: solid(std::integral_constant<int, 7>{}); break;
        default: throw std::invalid_argument("mcorr_solid: nQ must be 3-7");
      }
      C coef;
      coef.useMetrics = useMetrics; coef.epsFactHeaviside = epsFactHeaviside; coef.epsFactDirac = epsFactDirac;
      coef.epsFactDiffusion = epsFactDiffusion; coef.fields = fields.data(); coef.nQ = nQ; coef.Hs = Hs.data();
      py::array_t<double> R({nE, nD}), J({nE, nD, nD}), qu({nE, nQ}), qr({nE, nQ}), qn({nE, nQ, nS});
      adrh::elementKernelSpace<C, S>(coef, nE, nQ, xn.data(), u.data(), nullptr, qp.data(), qw.data(), 0.0, 0.0,
                                     R.mutable_data(), J.mutable_data());
      for (int eN = 0; eN < nE; ++eN)
        adrh::mcorrPointOutputs<S>(coef.bindElement(eN), xn.data() + eN * nD * 3, u.data() + eN * nD, nQ, qp.data(),
                                   qu.mutable_data() + eN * nQ, qr.mutable_data() + eN * nQ, qn.mutable_data() + eN * nQ * nS);
      const std::vector<int> cut = adrh::mcorrCutFaces(nE, nD, icase.data(), phi_s.data(), l2g.data(), elementBoundaries.data(),
                                                       boundaryElements.data());
      const int nF = (int)cut.size();
      py::array_t<int> faces(nF);
      py::array_t<double> Rf({nF, 2, nD}), Jf({nF, 2, nD, 2, nD});
      for (int k = 0; k < nF; ++k)
      {
        const int ebN = cut[k], eL = boundaryElements.data()[ebN * 2], eR = boundaryElements.data()[ebN * 2 + 1];
        faces.mutable_data()[k] = ebN;
        if constexpr (nD != nS + 1)
          throw std::invalid_argument("mcorr_solid: the ghost penalty is reproduced for P1 simplices");
        else
          adrh::mcorrGhostFace<S>(xn.data() + eL * nD * 3, xn.data() + eR * nD * 3, u.data() + eL * nD, u.data() + eR * nD,
                                boundaryLocal.data()[ebN * 2], boundaryLocal.data()[ebN * 2 + 1],
                                gamma * boundaryDiameter.data()[ebN], Rf.mutable_data() + k * 2 * nD,
                                Jf.mutable_data() + k * 4 * nD * nD);
      }
      return py::make_tuple(R, J, qu, qr, qn, icase, Hs, faces, Rf, Jf);
    });
  });

  // RANS2P's element integrals (rans2p.h, the two-phase Navier-Stokes VMS element): xn (nE, nD, 3), u (nE, 1 + nS, nD)
  // with the components p, u_0, .., qp (nQ, nS), qw (nQ), fields (nE, nQ, 3 + 3 nS) as NavierStokesFields orders them,
  // h_e (nE), and params = (rho_0, nu_0, rho_1, nu_1, g_0, g_1, g_2, alphaBDF, Ct_sge, Cd_sge, C_dc, useMetrics, hFactor,
  // useVF, MOMENTUM_SGE, PRESSURE_SGE, VELOCITY_SGE). Returns R (nE, nc, nD), J (nE, nc, nD, nc, nD), q_velocity
  // (nE, nQ, nS), q_numDiff (nE, nQ), q_cfl (nE, nQ).
  m.def("rans2p_element", [](int nd, Arr xn, Arr u, Arr qp, Arr qw, Arr fields, Arr h_e, Arr params) {
    return dispatchSpace(nd, "p1", [&](auto spaceTag) {
      using S = decltype(spaceTag);
      constexpr int nS = S::nS, nD = S::nDOF, nc = 1 + nS;
      constexpr int nF = adrh::NavierStokesFields<nS>::nFields;
      const int nE = xn.shape(0), nQ = qw.shape(0);
      if (u.ndim() != 3 || u.shape(0) != nE || u.shape(1) != nc || u.shape(2) != nD)
        throw std::invalid_argument("rans2p_element: u must be (nE, 1 + nS, nD)");
      if (fields.ndim() != 3 || fields.shape(0) != nE || fields.shape(1) != nQ || fields.shape(2) != nF)
        throw std::invalid_argument("rans2p_element: fields must be (nE, nQ, 3 + 3 nS)");
      if (params.shape(0) != 17) throw std::invalid_argument("rans2p_element: 17 params");
      const double *q = params.data();
      adrh::NavierStokesParams P{q[0], q[1], q[2], q[3], {q[4], q[5], q[6]}, q[7], q[8], q[9], q[10], q[11], q[12], q[13],
                                 q[14], q[15], q[16]};
      py::array_t<double> R({nE, nc, nD}), J({nE, nc, nD, nc, nD}), qv({nE, nQ, nS}), qnd({nE, nQ}), qcfl({nE, nQ});
      {
        py::gil_scoped_release release;
        for (int eN = 0; eN < nE; ++eN)
          adrh::navierStokesVMSElement<S>(P, xn.data() + eN * nD * 3, u.data() + eN * nc * nD, nQ, qp.data(), qw.data(),
                                          fields.data() + eN * nQ * nF, h_e.data()[eN], R.mutable_data() + eN * nc * nD,
                                          J.mutable_data() + eN * nc * nD * nc * nD, qv.mutable_data() + eN * nQ * nS,
                                          qnd.mutable_data() + eN * nQ, qcfl.mutable_data() + eN * nQ);
      }
      return py::make_tuple(R, J, qv, qnd, qcfl);
    });
  });

  // MCorr.py's masking after assembly (mcorr.h step 4), in place: r *= activeR, the CSR rows of inactive DOFs become identity
  // rows, and the DOFs with activeDOF != 1 are zeroed.
  m.def("replace_inactive_rows", [](py::array_t<int, py::array::c_style> rowptr, py::array_t<int, py::array::c_style> colind,
                                    py::array_t<double, py::array::c_style> vals, py::array_t<double, py::array::c_style> r,
                                    Arr activeR, py::array_t<double, py::array::c_style> u_dof, Arr activeDOF) {
    const int n = r.shape(0);
    if (rowptr.shape(0) != n + 1 || activeR.shape(0) != n || u_dof.shape(0) != n || activeDOF.shape(0) != n)
      throw std::invalid_argument("replace_inactive_rows: sizes");
    adrh::replaceInactiveRows(n, rowptr.data(), colind.data(), vals.mutable_data(), r.mutable_data(), activeR.data(),
                              u_dof.mutable_data(), activeDOF.data());
  });
}
