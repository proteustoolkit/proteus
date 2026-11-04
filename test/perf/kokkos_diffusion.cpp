// A research test for the generated kernels (C19): the diffusion residual on a structured
// hexahedral mesh, in Kokkos, assembled with element coloring.
//
// What it tests, in the terms of the C30 baseline (ON_NODE_PERFORMANCE_BASELINE.md):
//   * what the dependency costs: compile time and size of a kernel that uses it;
//   * what a kernel that computes its geometry and coefficients in registers does per element
//     (no stored quadrature arrays), against the legacy path's passes;
//   * whether coloring gives a threaded assembly that does not depend on the thread count:
//     the residual must be bit-identical at every number of threads;
//   * how the kernel scales over the cores of a node.
//
// The problem is the Poisson test's: -div(a(x) grad u) with a = diag(x+5, y+5, z+10) on a
// regular grid of Q1 hexahedra, 2x2x2 Gauss points, evaluated at u = x^2+y^2+z^2. The residual
// is r_a = sum_q w_q |J| sum_d a_d(x_q) (du/dx_d)(q) (dphi_a/dx_d)(q), computed per element from
// its eight nodal values, and scattered into the nodal vector. Eight colors (the parity of the
// element's i, j, k) mean that the elements of a color share no node, so the scatter within a
// color needs neither atomics nor locks; colors run one after the other.
//
// Usage: kokkos_diffusion NN [repeats] [--kokkos-num-threads=N]
#include <Kokkos_Core.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

struct RefHex {
  // reference gradients of the eight Q1 shape functions at the eight Gauss points, d/dxi
  double dphi[8][8][3];
  double w;
  double xi[8][3];
  RefHex() {
    const double g = 1.0 / std::sqrt(3.0);
    int sgn[8][3] = {{-1,-1,-1},{1,-1,-1},{1,1,-1},{-1,1,-1},{-1,-1,1},{1,-1,1},{1,1,1},{-1,1,1}};
    for (int q = 0; q < 8; q++) {
      for (int d = 0; d < 3; d++) xi[q][d] = sgn[q][d] * g;
      for (int a = 0; a < 8; a++)
        for (int d = 0; d < 3; d++) {
          double v = 0.5 * sgn[a][d];
          for (int e = 0; e < 3; e++)
            if (e != d) v *= 0.5 * (1.0 + sgn[a][e] * xi[q][e]);
          dphi[q][a][d] = v;
        }
    }
    w = 1.0;   // Gauss weights of the 2-point rule
  }
};

// One element: nodal values in, residual contribution out. Geometry (a cube of side h) and the
// coefficient are computed here, from the element's origin: nothing is read but the eight values.
KOKKOS_INLINE_FUNCTION
void element_residual(const RefHex& R, const double* u, double ox, double oy, double oz, double h,
                      double* r) {
  const double s = 2.0 / h, detw = (h * 0.5) * (h * 0.5) * (h * 0.5);
  for (int a = 0; a < 8; a++) r[a] = 0.0;
  for (int q = 0; q < 8; q++) {
    double gu[3] = {0.0, 0.0, 0.0};
    for (int a = 0; a < 8; a++)
      for (int d = 0; d < 3; d++) gu[d] += u[a] * R.dphi[q][a][d];
    const double x = ox + 0.5 * h * (1.0 + R.xi[q][0]);
    const double y = oy + 0.5 * h * (1.0 + R.xi[q][1]);
    const double z = oz + 0.5 * h * (1.0 + R.xi[q][2]);
    const double coef[3] = {x + 5.0, y + 5.0, z + 10.0};
    double flux[3];
    for (int d = 0; d < 3; d++) flux[d] = coef[d] * (gu[d] * s) * R.w * detw;
    for (int a = 0; a < 8; a++)
      for (int d = 0; d < 3; d++) r[a] += flux[d] * (R.dphi[q][a][d] * s);
  }
}

// One color: its elements share no node, so the scatter needs neither atomics nor locks. A named
// function, because a device lambda must be defined inside one (nvcc).
void sweep_color(Kokkos::View<int*> base, Kokkos::View<const double*> u, Kokkos::View<double*> r,
                 RefHex ref, long nn, long strideY, long strideZ, double x0, double h) {
  Kokkos::parallel_for("diffusion residual", Kokkos::RangePolicy<>(0, base.extent(0)),
                       KOKKOS_LAMBDA(const long e) {
    const long n0 = base(e);
    const long node[8] = {n0, n0 + 1, n0 + 1 + strideY, n0 + strideY,
                          n0 + strideZ, n0 + 1 + strideZ, n0 + 1 + strideY + strideZ,
                          n0 + strideY + strideZ};
    const long i = n0 % nn, j = (n0 / nn) % nn, k = n0 / (nn * nn);
    double ue[8], re[8];
    for (int a = 0; a < 8; a++) ue[a] = u(node[a]);
    element_residual(ref, ue, x0 + i * h, x0 + j * h, x0 + k * h, h, re);
    for (int a = 0; a < 8; a++) r(node[a]) += re[a];       // no atomics: the color shares no node
  });
}

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  int rc = 0;
  {
    const int nn = argc > 1 ? std::atoi(argv[1]) : 65;
    const int repeats = argc > 2 && argv[2][0] != '-' ? std::atoi(argv[2]) : 10;
    const int ne = nn - 1;
    const double L = 6.0, h = L / ne, x0 = -3.0;
    const long nNodes = (long)nn * nn * nn, nElem = (long)ne * ne * ne;
    const RefHex ref;

    // nodal values and the eight colors' element lists (the base node of each element)
    Kokkos::View<double*> u("u", nNodes), r("r", nNodes);
    auto hu = Kokkos::create_mirror_view(u);
    for (int k = 0; k < nn; k++)
      for (int j = 0; j < nn; j++)
        for (int i = 0; i < nn; i++) {
          double x = x0 + i * h, y = x0 + j * h, z = x0 + k * h;
          hu(i + (long)nn * (j + (long)nn * k)) = x * x + y * y + z * z;
        }
    Kokkos::deep_copy(u, hu);
    std::vector<std::vector<int>> by_color(8);
    for (int k = 0; k < ne; k++)
      for (int j = 0; j < ne; j++)
        for (int i = 0; i < ne; i++)
          by_color[(i & 1) + 2 * (j & 1) + 4 * (k & 1)].push_back(i + nn * (j + nn * k));
    Kokkos::View<int*> colors[8];
    for (int c = 0; c < 8; c++) {
      colors[c] = Kokkos::View<int*>("color", by_color[c].size());
      auto hc = Kokkos::create_mirror_view(colors[c]);
      for (size_t e = 0; e < by_color[c].size(); e++) hc(e) = by_color[c][e];
      Kokkos::deep_copy(colors[c], hc);
    }
    const long strideY = nn, strideZ = (long)nn * nn;

    auto assemble = [&]() {
      Kokkos::deep_copy(r, 0.0);
      for (int c = 0; c < 8; c++)
        sweep_color(colors[c], u, r, ref, nn, strideY, strideZ, x0, h);
      Kokkos::fence();
    };

    assemble();                                    // warm up
    double best = 1e30;
    for (int rep = 0; rep < repeats; rep++) {
      auto t0 = std::chrono::steady_clock::now();
      assemble();
      double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
      best = std::min(best, dt);
    }

    // checksum of the residual's bits, to compare across thread counts; and a plausibility check
    auto hr = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), r);
    unsigned long long h64 = 1469598103934665603ULL;
    double sum = 0.0;
    for (long n = 0; n < nNodes; n++) {
      unsigned long long bits;
      std::memcpy(&bits, &hr(n), 8);
      h64 = (h64 ^ bits) * 1099511628211ULL;
      sum += hr(n);
    }
    // the weak residual of -div(a grad u) = f at an interior node is int f phi_i dV, which for this
    // f (the Poisson test's) and the nodal basis is f(x_i) h^3 to the quadrature's accuracy
    const long mid = nn / 2;
    const long inode = mid + nn * (mid + (long)nn * mid);
    const double xm = x0 + mid * h, ym = xm, zm = xm;
    const double f = -2.0 * xm - 2.0 * (5.0 + xm) - 2.0 * ym - 2.0 * (5.0 + ym) - 2.0 * zm - 2.0 * (10.0 + zm);
    const double expected = f * h * h * h;
    std::printf("kokkos %s  threads=%d  nn=%d  elements=%ld  %.1f ns/element  %.3f s  "
                "%.1f Melement/s  checksum=%016llx  interior r=%.6e (~%.6e)\n",
                Kokkos::DefaultExecutionSpace::name(), (int)Kokkos::DefaultExecutionSpace().concurrency(), nn,
                nElem, 1e9 * best / nElem, best, nElem / best / 1e6, h64, hr(inode), expected);
  }
  Kokkos::finalize();
  return rc;
}
