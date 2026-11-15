#ifndef PROTEUS_UNFITTED_H
#define PROTEUS_UNFITTED_H
// Unfitted (cut-cell) finite element building blocks that sit on top of the moment fits of
// equivalent_polynomials.h and know nothing of ADR's storage:
//
//   IFEBasis    the two-sided immersed-finite-element basis of one element cut by an interface (with its unit flux-jump
//               lift): coefficient solve, evaluation at quadrature points, gradients. Built from the cut frame of an
//               equivalent_polynomials::Simplex (the permutation of the nodes and the geometry of the cut), it replaces
//               what used to live inside Simplex behind `useIfemBasis`, so the models that include
//               equivalent_polynomials.h (MCorr, VOF, RDLS, RANS*) no longer carry its storage.
//   IFEFunctions  GeneralizedFunctions_mix (moment fits of H, 1-H, D) plus an IFEBasis, calculated and queried together:
//               the object ADR holds per cut element.
//
// Includes equivalent_polynomials.h and UnfittedKernels.h; uses no pyarray or ADR type, plain pointers and sizes.
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include "equivalent_polynomials.h"
#include "UnfittedKernels.h"
#include "cutquadric.h"

namespace proteus
{
  namespace unfitted
  {
    // ---- small dense solves for the two-sided basis coefficients (2D) ---------------------------------------------
    // (formerly ifemBasisCoefficients_wrapper.h)

    // Solve A*x = b for a small, fixed-size N x N system via Gauss-Jordan
    // elimination with partial pivoting. A is row-major and destroyed.
    template <int N>
    inline void solveSmallLinearSystem(double A[N * N], double b[N], double x[N])
    {
      for (int col = 0; col < N; col++)
      {
        int piv = col;
        double maxval = fabs(A[col * N + col]);
        for (int r = col + 1; r < N; r++)
        {
          if (fabs(A[r * N + col]) > maxval)
          {
            maxval = fabs(A[r * N + col]);
            piv = r;
          }
        }
        if (piv != col)
        {
          for (int j = 0; j < N; j++)
            std::swap(A[col * N + j], A[piv * N + j]);
          std::swap(b[col], b[piv]);
        }
        const double pivval = A[col * N + col];
        assert(fabs(pivval) > 1.0e-14 && "Singular system in solveSmallLinearSystem");
        const double invPiv = 1.0 / pivval;
        for (int j = col; j < N; j++)
          A[col * N + j] *= invPiv;
        b[col] *= invPiv;
        for (int r = 0; r < N; r++)
        {
          if (r == col)
            continue;
          const double factor = A[r * N + col];
          if (factor != 0.0)
          {
            for (int j = col; j < N; j++)
              A[r * N + j] -= factor * A[col * N + j];
            b[r] -= factor * b[col];
          }
        }
      }
      for (int i = 0; i < N; i++)
        x[i] = b[i];
    }

    // P1 linear basis solver (overload for 3 nodal values)
    //
    // Solves the same constraint system as
    // proteus.ifemBasisCoefficients._solveCoefficients_P1 (nodal
    // interpolation + interface continuity + flux jump), but numerically:
    // the constraint matrix depends only on (x0,y0,nx,ny,ma,mb) -- not on
    // nodal_values -- so this is a plain 6x6 linear solve with no symbolic
    // differentiation and no Python round-trip.
    inline std::array<double, 6> solve_ifem_basis_coefficients(
        int basis_order,
        double x0,
        double y0,
        double nx,
        double ny,
        double ma,
        double mb,
        double jf,
        double Jit00,
        double Jit01,
        double Jit10,
        double Jit11,
        const std::array<double, 3> &nodal_values)
    {
      (void)basis_order; // always 1 for this overload
      const double tx = Jit00 * nx + Jit10 * ny;
      const double ty = Jit01 * nx + Jit11 * ny;

      // unknowns: [a1,a2,a3,b1,b2,b3]
      double A[36] = {
          1., 0., 0., 0., 0., 0.,
          0., 0., 0., 1., 1., 0.,
          0., 0., 0., 1., 0., 1.,
          -1., -x0, 0., 1., x0, 0.,
          -1., 0., -y0, 1., 0., y0,
          0., -ma * tx, -ma * ty, 0., mb * tx, mb * ty};
      double rhs[6] = {nodal_values[0], nodal_values[1], nodal_values[2], 0.0, 0.0, jf};

      double coeffs[6];
      solveSmallLinearSystem<6>(A, rhs, coeffs);

      std::array<double, 6> out;
      for (int i = 0; i < 6; ++i)
      {
        out[i] = coeffs[i];
        if (std::isnan(out[i]) || std::isinf(out[i]))
          throw std::runtime_error("solve_ifem_basis_coefficients (P1) returned NaN or Inf");
      }
      return out;
    }

    // P2 quadratic basis solver (overload for 6 nodal values)
    //
    // Solves the same constraint system as
    // proteus.ifemBasisCoefficients._solveCoefficients_P2 numerically. va/vb
    // are full 2D quadratics, so all constraints (nodal interpolation,
    // interface continuity, flux jump, normal-Laplacian jump) reduce to
    // linear equations in the 12 coefficients with coefficients that are
    // plain algebraic functions of (x0,y0,nx,ny,ma,mb) -- the "symbolic
    // differentiation" in the Python version is just picking off constant
    // polynomial coefficients, so it never needed to happen at runtime.
    inline std::array<double, 12> solve_ifem_basis_coefficients(
        int basis_order,
        double x0,
        double y0,
        double nx,
        double ny,
        double ma,
        double mb,
        double jf,
        double Jit00,
        double Jit01,
        double Jit10,
        double Jit11,
        const std::array<double, 6> &nodal_values)
    {
      (void)basis_order; // always 2 for this overload
      assert(!((x0 == 0.0 && y0 == 0.0) || (x0 == 1.0 && y0 == 0.0) || (x0 == 0.0 && y0 == 1.0)) &&
             "Interface passes through a triangle vertex");
      assert(!(x0 > 0.5 && y0 <= 0.5) &&
             "Invalid interface location for quadratic basis functions (should have been flipped upstream)");

      const double tx = Jit00 * nx + Jit10 * ny;
      const double ty = Jit01 * nx + Jit11 * ny;
      const double v1 = nodal_values[0], v2 = nodal_values[1], v3 = nodal_values[2],
                   v4 = nodal_values[3], v5 = nodal_values[4], v6 = nodal_values[5];

      // unknowns: [a1,a2,a3,a4,a5,a6, b1,b2,b3,b4,b5,b6]
      double A[144] = {0.0};
      double rhs[12] = {0.0};

      // c1: va(0,0) = v1
      A[0 * 12 + 0] = 1.0;
      rhs[0] = v1;

      // c2: vb(1,0) = v2  (x^2 term is 1 at x=1, so b5 contributes)
      A[1 * 12 + 6] = 1.0;
      A[1 * 12 + 7] = 1.0;
      A[1 * 12 + 10] = 1.0;
      rhs[1] = v2;

      // c3: vb(0,1) = v3  (y^2 term is 1 at y=1, so b6 contributes)
      A[2 * 12 + 6] = 1.0;
      A[2 * 12 + 8] = 1.0;
      A[2 * 12 + 11] = 1.0;
      rhs[2] = v3;

      // c4: vb(1/2,0) = v4 if x0<=1/2, else va(1/2,0) = v4
      if (x0 <= 0.5)
      {
        A[3 * 12 + 6] = 1.0;
        A[3 * 12 + 7] = 0.5;
        A[3 * 12 + 10] = 0.25;
      }
      else
      {
        A[3 * 12 + 0] = 1.0;
        A[3 * 12 + 1] = 0.5;
        A[3 * 12 + 4] = 0.25;
      }
      rhs[3] = v4;

      // c5: vb(1/2,1/2) = v5
      A[4 * 12 + 6] = 1.0;
      A[4 * 12 + 7] = 0.5;
      A[4 * 12 + 8] = 0.5;
      A[4 * 12 + 9] = 0.25;
      A[4 * 12 + 10] = 0.25;
      A[4 * 12 + 11] = 0.25;
      rhs[4] = v5;

      // c6: vb(0,1/2) = v6 if y0<=1/2, else va(0,1/2) = v6
      if (y0 <= 0.5)
      {
        A[5 * 12 + 6] = 1.0;
        A[5 * 12 + 8] = 0.5;
        A[5 * 12 + 11] = 0.25;
      }
      else
      {
        A[5 * 12 + 0] = 1.0;
        A[5 * 12 + 2] = 0.5;
        A[5 * 12 + 5] = 0.25;
      }
      rhs[5] = v6;

      // c7: vb(x0,0) - va(x0,0) = 0
      A[6 * 12 + 0] = -1.0;
      A[6 * 12 + 1] = -x0;
      A[6 * 12 + 4] = -x0 * x0;
      A[6 * 12 + 6] = 1.0;
      A[6 * 12 + 7] = x0;
      A[6 * 12 + 10] = x0 * x0;

      // c8: vb(0,y0) - va(0,y0) = 0
      A[7 * 12 + 0] = -1.0;
      A[7 * 12 + 2] = -y0;
      A[7 * 12 + 5] = -y0 * y0;
      A[7 * 12 + 6] = 1.0;
      A[7 * 12 + 8] = y0;
      A[7 * 12 + 11] = y0 * y0;

      // c9: vb(x0/2,y0/2) - va(x0/2,y0/2) = 0
      A[8 * 12 + 0] = -1.0;
      A[8 * 12 + 1] = -0.5 * x0;
      A[8 * 12 + 2] = -0.5 * y0;
      A[8 * 12 + 3] = -0.25 * x0 * y0;
      A[8 * 12 + 4] = -0.25 * x0 * x0;
      A[8 * 12 + 5] = -0.25 * y0 * y0;
      A[8 * 12 + 6] = 1.0;
      A[8 * 12 + 7] = 0.5 * x0;
      A[8 * 12 + 8] = 0.5 * y0;
      A[8 * 12 + 9] = 0.25 * x0 * y0;
      A[8 * 12 + 10] = 0.25 * x0 * x0;
      A[8 * 12 + 11] = 0.25 * y0 * y0;

      // c10: mb*flux_b(x0,0) - ma*flux_a(x0,0) = jf
      A[9 * 12 + 1] = -ma * tx;
      A[9 * 12 + 2] = -ma * ty;
      A[9 * 12 + 3] = -ma * x0 * ty;
      A[9 * 12 + 4] = -ma * 2.0 * x0 * tx;
      A[9 * 12 + 7] = mb * tx;
      A[9 * 12 + 8] = mb * ty;
      A[9 * 12 + 9] = mb * x0 * ty;
      A[9 * 12 + 10] = mb * 2.0 * x0 * tx;
      rhs[9] = jf;

      // c11: mb*flux_b(0,y0) - ma*flux_a(0,y0) = jf
      A[10 * 12 + 1] = -ma * tx;
      A[10 * 12 + 2] = -ma * ty;
      A[10 * 12 + 3] = -ma * y0 * tx;
      A[10 * 12 + 5] = -ma * 2.0 * y0 * ty;
      A[10 * 12 + 7] = mb * tx;
      A[10 * 12 + 8] = mb * ty;
      A[10 * 12 + 9] = mb * y0 * tx;
      A[10 * 12 + 11] = mb * 2.0 * y0 * ty;
      rhs[10] = jf;

      // c12: mb*vb_nn(x0/2,y0/2) - ma*va_nn(x0/2,y0/2) = 0
      // (va_nn/vb_nn are constant over the element for a quadratic, so the
      // evaluation point doesn't actually matter)
      A[11 * 12 + 3] = -ma * 2.0 * tx * ty;
      A[11 * 12 + 4] = -ma * 2.0 * tx * tx;
      A[11 * 12 + 5] = -ma * 2.0 * ty * ty;
      A[11 * 12 + 9] = mb * 2.0 * tx * ty;
      A[11 * 12 + 10] = mb * 2.0 * tx * tx;
      A[11 * 12 + 11] = mb * 2.0 * ty * ty;

      double coeffs[12];
      solveSmallLinearSystem<12>(A, rhs, coeffs);

      std::array<double, 12> out;
      for (int i = 0; i < 12; ++i)
      {
        out[i] = coeffs[i];
        if (std::isnan(out[i]) || std::isinf(out[i]))
          throw std::runtime_error("solve_ifem_basis_coefficients (P2) returned NaN or Inf");
      }
      return out;
    }

    // The two-sided P1 IFEM basis on a tetrahedron, in closed form (C44 phase 2; math plan section 2.1).
    //
    // With phi the P1 level set (phi < 0 is branch a, coefficient ma; phi > 0 is branch b, coefficient mb),
    // psi^+ = sum_{phi_j > 0} phi_j lambda_j and psi^- = phi - psi^+, the basis function of node i is
    //     v_i^a = lambda_i - c_i psi^+,      v_i^b = lambda_i + c_i psi^-,
    //     c_i   = (ma - mb) grad(lambda_i).grad(phi) / D,
    //     D     = mb grad(psi^-).grad(phi) + ma grad(psi^+).grad(phi).
    // Both pieces take the value delta_ij at node j (psi^+ vanishes at a negative node, psi^- at a positive
    // one), they differ by c_i phi (so [v] = 0 on the whole planar cut, for the 1-3, 3-1 and 2-2 splits),
    // and mb dv^b/dn - ma dv^a/dn = 0 by construction; sum_i c_i = 0 gives a partition of unity. The unit
    // flux-jump lift (zero nodal values, mb d(l^b)/dn - ma d(l^a)/dn = 1, n = grad(phi)/|grad(phi)|) has
    // the same denominator: l^a = -cJ psi^+, l^b = cJ psi^-, cJ = |grad(phi)| / D.
    //
    // The basis exists iff D != 0. D > 0 whenever every dihedral angle between a positive and a negative
    // vertex's faces is at most 90 degrees; on an obtuse tetrahedron D vanishes at particular contrasts.
    // That case throws (ifem_singular_error); assert() is compiled out and would give garbage. A zero
    // vertex value counts for neither side; the result does not depend on which side it is given.
    struct ifem_singular_error : public std::runtime_error
    {
      ifem_singular_error(const std::string &what) : std::runtime_error(what) {}
    };
    // |D| below this times max(ma, mb) |grad phi|^2 is "singular". On a tetrahedron without obtuse dihedral
    // angles D/|grad phi|^2 >= min(ma, mb), so the test only trips for a contrast above 1e10 there.
    static constexpr double ifem_singular_tol = 1.0e-10;

    template <int nSpace, int nP_ifem, int nQ, int nEBQ>
    class IFEBasis
    {
    public:
      using CutFrame = equivalent_polynomials::CutFrame;

      // Coefficients of the basis and of the lift for the cut described by `fr`; ma, mb are the coefficients of branch
      // a (phi < 0) and b (phi > 0). Cells the interface only touches (fr.corner, fr.edge) get the ordinary basis.
      inline void calculate_coefficients(const CutFrame &fr, const double ma, const double mb)
      {
        if constexpr (nSpace == 3)
          calculate_coefficients_tet(fr, ma, mb);
        else
          calculate_coefficients_2d(fr, ma, mb);
      }

      // Evaluate the basis (and the lift) at the quadrature points of an element (or of its faces) and make point 0
      // current. xi_q holds nPoints permuted reference points (nSpace each, packed).
      inline void evaluate_points(const CutFrame &fr, const double *xi_q, const int nPoints, const bool isBoundary)
      {
        for (int q = 0; q < nPoints; q++)
        {
          const double *xi = &xi_q[q * nSpace];
          double *va = isBoundary ? &_va_ebq[q * nP_ifem] : &_va[q * nP_ifem];
          double *vb = isBoundary ? &_vb_ebq[q * nP_ifem] : &_vb[q * nP_ifem];
          calculate_basis(xi, va, vb);
          if (isBoundary)
            calculate_basis_gradients(fr, xi, &_va_x_ebq[q * nP_ifem], &_va_y_ebq[q * nP_ifem], &_va_z_ebq[q * nP_ifem], &_vb_x_ebq[q * nP_ifem], &_vb_y_ebq[q * nP_ifem], &_vb_z_ebq[q * nP_ifem]);
          else
            calculate_basis_gradients(fr, xi, &_va_x[q * nP_ifem], &_va_y[q * nP_ifem], &_va_z[q * nP_ifem], &_vb_x[q * nP_ifem], &_vb_y[q * nP_ifem], &_vb_z[q * nP_ifem]);
          calculate_lift(fr, xi, isBoundary ? &_psi_ebq[q * 8] : &_psi[q * 8]);
        }
        cut = !(fr.edge == -1 || fr.corner == -1 || fr.edge == 1 || fr.corner == 1);
        if (isBoundary)
          set_boundary_quad(0);
        else
          set_quad(0);
      }

      inline void set_quad(unsigned int q)
      {
        assert(q < nQ);
        // basis functions already adjusted for inside_out
        for (int i = 0; i < nP_ifem; i++)
        {
          _va_q[i] = _va[q * nP_ifem + i];
          _vb_q[i] = _vb[q * nP_ifem + i];
          _va_x_q[i] = _va_x[q * nP_ifem + i];
          _va_y_q[i] = _va_y[q * nP_ifem + i];
          _va_z_q[i] = _va_z[q * nP_ifem + i];
          _vb_x_q[i] = _vb_x[q * nP_ifem + i];
          _vb_y_q[i] = _vb_y[q * nP_ifem + i];
          _vb_z_q[i] = _vb_z[q * nP_ifem + i];
        }
        for (int i = 0; i < 8; i++)
          _psi_q[i] = cut ? _psi[q * 8 + i] : 0.0;
      }

      inline void set_boundary_quad(unsigned int ebq)
      {
        assert(ebq < nEBQ);
        for (int i = 0; i < nP_ifem; i++)
        {
          _va_q[i] = _va_ebq[ebq * nP_ifem + i];
          _vb_q[i] = _vb_ebq[ebq * nP_ifem + i];
          _va_x_q[i] = _va_x_ebq[ebq * nP_ifem + i];
          _va_y_q[i] = _va_y_ebq[ebq * nP_ifem + i];
          _vb_x_q[i] = _vb_x_ebq[ebq * nP_ifem + i];
          _vb_y_q[i] = _vb_y_ebq[ebq * nP_ifem + i];
          _va_z_q[i] = _va_z_ebq[ebq * nP_ifem + i];
          _vb_z_q[i] = _vb_z_ebq[ebq * nP_ifem + i];
        }
        for (int i = 0; i < 8; i++)
          _psi_q[i] = cut ? _psi_ebq[ebq * 8 + i] : 0.0;
      }

      inline double VA(int i) { return _va_q[i]; };
      inline double VA_x(int i) { return _va_x_q[i]; };
      inline double VA_y(int i) { return _va_y_q[i]; };
      inline double VA_z(int i) { return _va_z_q[i]; };
      inline double VB(int i) { return _vb_q[i]; };
      inline double VB_x(int i) { return _vb_x_q[i]; };
      inline double VB_y(int i) { return _vb_y_q[i]; };
      inline double VB_z(int i) { return _vb_z_q[i]; };
      // Unit flux-jump lift psi (branch a / branch b, value and physical gradient) at the current
      // quadrature point: zero at every node, continuous across the interface, and
      // mb*dpsi_b/dn - ma*dpsi_a/dn = 1 with n the interface normal (from branch a to branch b).
      // The basis itself is always built for a homogeneous flux jump; a prescribed jump J_f is
      // represented by adding J_f*psi to the trial function once (like the solution-jump lift).
      inline double PSIA() { return _psi_q[0]; };
      inline double PSIA_x() { return _psi_q[1]; };
      inline double PSIA_y() { return _psi_q[2]; };
      inline double PSIA_z() { return _psi_q[3]; };
      inline double PSIB() { return _psi_q[4]; };
      inline double PSIB_x() { return _psi_q[5]; };
      inline double PSIB_y() { return _psi_q[6]; };
      inline double PSIB_z() { return _psi_q[7]; };

    private:
      // false on a cell the interface only touches (edge/corner): the lift is then zero
      bool cut = true;
      // the arrays VA()/VB()/VA_x()/... hand straight back to the caller
      double _va_q[nP_ifem] = {}, _vb_q[nP_ifem] = {}, _va_x_q[nP_ifem] = {}, _va_y_q[nP_ifem] = {}, _va_z_q[nP_ifem] = {},
             _vb_x_q[nP_ifem] = {}, _vb_y_q[nP_ifem] = {}, _vb_z_q[nP_ifem] = {};
      // coefficients in the monomials 1, xi, eta, xi*eta, xi^2, eta^2 (2D) or 1, xi, eta, zeta (3D P1, the first four)
      double _a1[nP_ifem] = {}, _a2[nP_ifem] = {}, _a3[nP_ifem] = {}, _b1[nP_ifem] = {}, _b2[nP_ifem] = {}, _b3[nP_ifem] = {};
      double _a4[nP_ifem] = {}, _a5[nP_ifem] = {}, _a6[nP_ifem] = {}, _b4[nP_ifem] = {}, _b5[nP_ifem] = {}, _b6[nP_ifem] = {};
      // unit flux-jump lift: coefficients (same monomial order) and values/gradients
      // [psi_a, psi_a_x, psi_a_y, psi_a_z, psi_b, psi_b_x, psi_b_y, psi_b_z] at the quadrature points (z is zero in 2D)
      double _lift_a[6] = {0., 0., 0., 0., 0., 0.}, _lift_b[6] = {0., 0., 0., 0., 0., 0.};
      double _psi[nQ * 8] = {}, _psi_ebq[nEBQ * 8] = {}, _psi_q[8] = {0., 0., 0., 0., 0., 0., 0., 0.};
      double _va[nQ * nP_ifem] = {}, _vb[nQ * nP_ifem] = {};
      double _va_ebq[nEBQ * nP_ifem] = {}, _vb_ebq[nEBQ * nP_ifem] = {};
      double _va_x[nQ * nP_ifem] = {}, _va_y[nQ * nP_ifem] = {}, _va_z[nQ * nP_ifem] = {}, _vb_x[nQ * nP_ifem] = {}, _vb_y[nQ * nP_ifem] = {}, _vb_z[nQ * nP_ifem] = {};
      double _va_x_ebq[nEBQ * nP_ifem] = {}, _va_y_ebq[nEBQ * nP_ifem] = {}, _va_z_ebq[nEBQ * nP_ifem] = {}, _vb_x_ebq[nEBQ * nP_ifem] = {}, _vb_y_ebq[nEBQ * nP_ifem] = {}, _vb_z_ebq[nEBQ * nP_ifem] = {};

      inline void calculate_coefficients_2d(const CutFrame &fr, const double ma, const double mb)
      {
    assert(nSpace == 2);
    assert(nSpace + 1 == 3);
    // fr.normal is the normal after Simplex's inside_out flip; for inside_out the solve below wants the flipped
    // one and for not inside_out the plain one, i.e. exactly fr.normal
    double nx = fr.normal[0], ny = fr.normal[1];
    double Jit00 = fr.inv_Jac[0 * nSpace + 0],
           Jit01 = fr.inv_Jac[1 * nSpace + 0],
           Jit10 = fr.inv_Jac[0 * nSpace + 1],
           Jit11 = fr.inv_Jac[1 * nSpace + 1];
    double x0 = fr.X_0[0],
           y0 = fr.X_0[1];
    const double *vall = nullptr;


    switch (nP_ifem)
    {
    case 3:
    {
      static const double vall_p1[9] = {
          1., 0., 0.,
          0., 1., 0.,
          0., 0., 1.};
      vall = vall_p1;
      for (int j = 0; j < 3; j++)
      {
        int i = fr.permutation[j];
        double v[3] = {0.0, 0.0, 0.0};
        v[0] = vall[j * 3 + 0];
        v[1] = vall[j * 3 + 1];
        v[2] = vall[j * 3 + 2];

        if (fr.corner || fr.edge)
        {
          _a1[i] = v[0];
          _a2[i] = -v[0] + v[1];
          _a3[i] = -v[0] + v[2];
          _b1[i] = v[0];
          _b2[i] = -v[0] + v[1];
          _b3[i] = -v[0] + v[2];
        }
        else
        {
          // nathawani: Implement inside out case directly here.
          const std::array<double, 3> nodal_values = {v[0], v[1], v[2]};
          if (fr.inside_out)
          {
            const std::array<double, 6> coeffs = solve_ifem_basis_coefficients(
              1, x0, y0, nx, ny, mb, ma, 0.0, Jit00, Jit01, Jit10, Jit11, nodal_values);
              
            _b1[i] = coeffs[0];
            _b2[i] = coeffs[1];
            _b3[i] = coeffs[2];
            _a1[i] = coeffs[3];
            _a2[i] = coeffs[4];
            _a3[i] = coeffs[5];
          }
          else
          {

            const std::array<double, 6> coeffs = solve_ifem_basis_coefficients(
              1, x0, y0, nx, ny, ma, mb, 0.0, Jit00, Jit01, Jit10, Jit11, nodal_values);
            
            _a1[i] = coeffs[0];
            _a2[i] = coeffs[1];
            _a3[i] = coeffs[2];
            _b1[i] = coeffs[3];
            _b2[i] = coeffs[4];
            _b3[i] = coeffs[5];
          }
        }
      }
      break;
    }
    case 6:
    {
      static const double vall_p2[36] = {
          1., 0., 0., 0., 0., 0.,
          0., 1., 0., 0., 0., 0.,
          0., 0., 1., 0., 0., 0.,
          0., 0., 0., 1., 0., 0.,
          0., 0., 0., 0., 1., 0.,
          0., 0., 0., 0., 0., 1.};
      vall = vall_p2;
      
      for (int j = 0; j < 6; j++)
      {
        int i = fr.permutation[j];
        double v[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};//, grad_va[2] = {0.0, 0.0}, grad_vb[2] = {0.0, 0.0}, grad_va_ref[2] = {0.0, 0.0}, grad_vb_ref[2] = {0.0, 0.0};

        v[0] = vall[j * 6 + 0];
        v[1] = vall[j * 6 + 1];
        v[2] = vall[j * 6 + 2];
        v[3] = vall[j * 6 + 3];
        v[4] = vall[j * 6 + 4];
        v[5] = vall[j * 6 + 5];

         if (fr.corner || fr.edge)
        {
          _a1[i] = v[0];
            _a2[i] = - 3 * v[0] - v[1] + 4 * v[3];
            _a3[i] = - 3 * v[0] - v[2] + 4 * v[5];
            _a4[i] = 4 * v[0] - 4 * v[3] + 4 * v[4] - 4 * v[5];
            _a5[i] = 2 * v[0] + 2 * v[1] - 4 * v[3];
            _a6[i] = 2 * v[0] + 2 * v[2] - 4 * v[5];
          _b1[i] = v[0];
            _b2[i] = - 3 * v[0] - v[1] + 4 * v[3];
            _b3[i] = - 3 * v[0] - v[2] + 4 * v[5];
            _b4[i] = 4 * v[0] - 4 * v[3] + 4 * v[4] - 4 * v[5];
            _b5[i] = 2 * v[0] + 2 * v[1] - 4 * v[3];
            _b6[i] = 2 * v[0] + 2 * v[2] - 4 * v[5]; 
        }
        else
        {
          if (fr.inside_out)
          {
             const std::array<double, 6> nodal_values = {v[0], v[1], v[2], v[3], v[4], v[5]};
             const std::array<double, 12> coeffs = solve_ifem_basis_coefficients(
              2, x0, y0, nx, ny, mb, ma, 0.0, Jit00, Jit01, Jit10, Jit11, nodal_values);
            
            _b1[i] = coeffs[0];
            _b2[i] = coeffs[1];
            _b3[i] = coeffs[2];
            _b4[i] = coeffs[3];
            _b5[i] = coeffs[4];
            _b6[i] = coeffs[5];
            _a1[i] = coeffs[6];
            _a2[i] = coeffs[7];
            _a3[i] = coeffs[8];
            _a4[i] = coeffs[9];
            _a5[i] = coeffs[10];
            _a6[i] = coeffs[11];
          }
          else
          {
            const std::array<double, 6> nodal_values = {v[0], v[1], v[2], v[3], v[4], v[5]};
            const std::array<double, 12> coeffs = solve_ifem_basis_coefficients(
              2, x0, y0, nx, ny, ma, mb, 0.0, Jit00, Jit01, Jit10, Jit11, nodal_values);

              _a1[i] = coeffs[0];
              _a2[i] = coeffs[1];
              _a3[i] = coeffs[2];
              _a4[i] = coeffs[3];
              _a5[i] = coeffs[4];
              _a6[i] = coeffs[5];
              _b1[i] = coeffs[6];
              _b2[i] = coeffs[7];
              _b3[i] = coeffs[8];
              _b4[i] = coeffs[9];
              _b5[i] = coeffs[10];
              _b6[i] = coeffs[11];
          }
        }
      }
      break;
    }
    default:
      throw std::runtime_error("Simplex::_calculate_coefficients not implemented for order > 2");
    }
    // Unit flux-jump lift: zero nodal values, flux jump 1. The solve is linear in (nodal values,
    // jf), so a basis built with jf != 0 would carry one copy of this function per basis function;
    // it is kept separate and added once by the caller, scaled by the prescribed jump.
    for (int i = 0; i < 6; i++)
    {
      _lift_a[i] = 0.0;
      _lift_b[i] = 0.0;
    }
    if (!(fr.corner || fr.edge))
    {
      // The solver imposes  m_second*dpsi_second/dn_s - m_first*dpsi_first/dn_s = jf  with
      // n_s = +-level_set_normal, and level_set_normal is the cut tangent rotated by 90 degrees: its
      // sign follows the order of the cut points, not the direction of phi. (The homogeneous basis
      // never notices.) In both fr.inside_out branches the physical jump from a (phi<0) to b (phi>0) is
      // jf*sign(level_set_normal . grad phi), so that sign is passed as jf for a unit jump.
      // the cut normal before the inside_out flip (the lift sign is defined against it)
      const double n0[2] = {fr.inside_out ? -fr.normal[0] : fr.normal[0], fr.inside_out ? -fr.normal[1] : fr.normal[1]};
      const double gphi_ref[2] = {fr.phi[1] - fr.phi[0], fr.phi[2] - fr.phi[0]};
      double n_dot_gphi = 0.0;
      for (int I = 0; I < 2; I++)
      {
        double g = 0.0;
        for (int J = 0; J < 2; J++)
          g += fr.inv_Jac[J * nSpace + I] * gphi_ref[J];
        n_dot_gphi += g * n0[I];
      }
      const double jf_unit = (n_dot_gphi >= 0.0) ? 1.0 : -1.0;
      double *first = fr.inside_out ? _lift_b : _lift_a, *second = fr.inside_out ? _lift_a : _lift_b;
      const double m_first = fr.inside_out ? mb : ma, m_second = fr.inside_out ? ma : mb;
      if (nP_ifem == 3)
      {
        const std::array<double, 3> zero = {0., 0., 0.};
        const std::array<double, 6> c = solve_ifem_basis_coefficients(
            1, x0, y0, nx, ny, m_first, m_second, jf_unit, Jit00, Jit01, Jit10, Jit11, zero);
        for (int i = 0; i < 3; i++)
        {
          first[i] = c[i];
          second[i] = c[3 + i];
        }
      }
      else if (nP_ifem == 6)
      {
        const std::array<double, 6> zero = {0., 0., 0., 0., 0., 0.};
        const std::array<double, 12> c = solve_ifem_basis_coefficients(
            2, x0, y0, nx, ny, m_first, m_second, jf_unit, Jit00, Jit01, Jit10, Jit11, zero);
        for (int i = 0; i < 6; i++)
        {
          first[i] = c[i];
          second[i] = c[6 + i];
        }
      }
    }
      }

      inline void calculate_coefficients_tet(const CutFrame &fr, const double ma, const double mb)
      {
    static_assert(nSpace == 3, "the closed-form tetrahedron basis");
    if (nP_ifem != 4)
      throw std::runtime_error("Simplex: the two-sided IFEM basis on tetrahedra is implemented for P1 only");
    for (int i = 0; i < 6; i++)
    {
      _lift_a[i] = 0.0;
      _lift_b[i] = 0.0;
    }
    if (fr.corner || fr.edge)
    {
      // not cut: the interface touches the element only at a vertex, edge or face; both branches are the
      // ordinary linear basis
      for (int j = 0; j < 4; j++)
      {
        const int i = fr.permutation[j];
        const double v[4] = {j == 0 ? 1.0 : 0.0, j == 1 ? 1.0 : 0.0, j == 2 ? 1.0 : 0.0, j == 3 ? 1.0 : 0.0};
        _a1[i] = v[0];
        _a2[i] = v[1] - v[0];
        _a3[i] = v[2] - v[0];
        _a4[i] = v[3] - v[0];
        _b1[i] = _a1[i];
        _b2[i] = _a2[i];
        _b3[i] = _a3[i];
        _b4[i] = _a4[i];
      }
      return;
    }
    // physical gradients of the barycentric coordinates in the permuted ordering: lambda_{k+1} = xi_k
    double g[4][3];
    for (int k = 0; k < 3; k++)
      for (int I = 0; I < 3; I++)
        g[k + 1][I] = fr.inv_Jac[k * 3 + I];
    for (int I = 0; I < 3; I++)
      g[0][I] = -(g[1][I] + g[2][I] + g[3][I]);
    double gphi[3] = {0., 0., 0.}, gpsi[3] = {0., 0., 0.}, psi[4];
    for (int j = 0; j < 4; j++)
    {
      psi[j] = fr.phi[j] > 0.0 ? fr.phi[j] : 0.0;
      for (int I = 0; I < 3; I++)
      {
        gphi[I] += fr.phi[j] * g[j][I];
        gpsi[I] += psi[j] * g[j][I];
      }
    }
    const double gg = gphi[0] * gphi[0] + gphi[1] * gphi[1] + gphi[2] * gphi[2];
    const double s_plus = gpsi[0] * gphi[0] + gpsi[1] * gphi[1] + gpsi[2] * gphi[2]; // grad(psi^+).grad(phi)
    const double D = mb * (gg - s_plus) + ma * s_plus;
    if (!(gg > 0.0) || !(std::fabs(D) >= ifem_singular_tol * std::max(std::fabs(ma), std::fabs(mb)) * gg))
    {
      std::ostringstream msg;
      msg << std::setprecision(6) << "the two-sided P1 IFEM basis does not exist on this tetrahedron: D = mb grad(psi-).grad(phi) + ma "
          << "grad(psi+).grad(phi) = " << D << " (|grad phi|^2 = " << gg << ", s = grad(psi+).grad(phi)/|grad phi|^2 = "
          << (gg > 0.0 ? s_plus / gg : 0.0) << ", ma = " << ma << ", mb = " << mb
          << "). A tetrahedron with a dihedral angle above 90 degrees is singular at the contrast ma/mb = 1 - 1/s; use a mesh without obtuse "
          << "dihedral angles (e.g. structured tetrahedra) or a different contrast.";
      throw ifem_singular_error(msg.str());
    }
    const double cJ = std::sqrt(gg) / D;
    for (int j = 0; j < 4; j++)
    {
      const int i = fr.permutation[j];
      const double c_j = (ma - mb) * (g[j][0] * gphi[0] + g[j][1] * gphi[1] + g[j][2] * gphi[2]) / D;
      // nodal values of the two pieces at the permuted nodes k
      double wa[4], wb[4];
      for (int k = 0; k < 4; k++)
      {
        wa[k] = (k == j ? 1.0 : 0.0) - c_j * psi[k];
        wb[k] = (k == j ? 1.0 : 0.0) + c_j * (fr.phi[k] - psi[k]);
      }
      _a1[i] = wa[0];
      _a2[i] = wa[1] - wa[0];
      _a3[i] = wa[2] - wa[0];
      _a4[i] = wa[3] - wa[0];
      _b1[i] = wb[0];
      _b2[i] = wb[1] - wb[0];
      _b3[i] = wb[2] - wb[0];
      _b4[i] = wb[3] - wb[0];
    }
    // unit flux-jump lift, coefficients in 1, xi, eta, zeta
    double la[4], lb[4];
    for (int k = 0; k < 4; k++)
    {
      la[k] = -cJ * psi[k];
      lb[k] = cJ * (fr.phi[k] - psi[k]);
    }
    _lift_a[0] = la[0];
    _lift_a[1] = la[1] - la[0];
    _lift_a[2] = la[2] - la[0];
    _lift_a[3] = la[3] - la[0];
    _lift_b[0] = lb[0];
    _lift_b[1] = lb[1] - lb[0];
    _lift_b[2] = lb[2] - lb[0];
    _lift_b[3] = lb[3] - lb[0];
      }

      inline void calculate_lift(const CutFrame &fr, const double *xi, double *psi)
      {
    const double *c[2] = {_lift_a, _lift_b};
    for (int s = 0; s < 2; s++)
    {
      const double *a = c[s];
      double val, g_ref[3] = {0.0, 0.0, 0.0};
      if constexpr (nSpace == 3)
      {
        val = a[0] + a[1] * xi[0] + a[2] * xi[1] + a[3] * xi[2];
        g_ref[0] = a[1];
        g_ref[1] = a[2];
        g_ref[2] = a[3];
      }
      else
      {
        val = a[0] + a[1] * xi[0] + a[2] * xi[1] + a[3] * xi[0] * xi[1] + a[4] * xi[0] * xi[0] + a[5] * xi[1] * xi[1];
        g_ref[0] = a[1] + a[3] * xi[1] + 2.0 * a[4] * xi[0];
        g_ref[1] = a[2] + a[3] * xi[0] + 2.0 * a[5] * xi[1];
      }
      double g[3] = {0.0, 0.0, 0.0};
      for (int I = 0; I < nSpace && I < 3; I++)
        for (int J = 0; J < nSpace && J < 3; J++)
          g[I] += fr.inv_Jac[J * nSpace + I] * g_ref[J];
      psi[4 * s + 0] = val;
      psi[4 * s + 1] = g[0];
      psi[4 * s + 2] = g[1];
      psi[4 * s + 3] = g[2];
    }
      }

      inline void calculate_basis(const double *xi, double *va, double *vb)
      {
    // Switch on nP_ifem to compute basis functions at given xi
    if constexpr (nSpace == 3)
    {
      // P1 on a tetrahedron (the only 3D order implemented)
      for (int i = 0; i < nP_ifem; i++)
      {
        va[i] = _a1[i] + _a2[i] * xi[0] + _a3[i] * xi[1] + _a4[i] * xi[2];
        vb[i] = _b1[i] + _b2[i] * xi[0] + _b3[i] * xi[1] + _b4[i] * xi[2];
      }
      return;
    }
    switch (nP_ifem)
    {
    case 3:
      for (int i = 0; i < nP_ifem; i++)
      {
        va[i] = _a1[i] + _a2[i] * xi[0] + _a3[i] * xi[1];
        vb[i] = _b1[i] + _b2[i] * xi[0] + _b3[i] * xi[1];

                 
      }
      break;
    case 6:
      for (int i = 0; i < nP_ifem; i++)
      {
        va[i] = _a1[i] + _a2[i] * xi[0] + _a3[i] * xi[1] + _a4[i] * xi[0] * xi[1] + _a5[i] * xi[0] * xi[0] + _a6[i] * xi[1] * xi[1];
        vb[i] = _b1[i] + _b2[i] * xi[0] + _b3[i] * xi[1] + _b4[i] * xi[0] * xi[1] + _b5[i] * xi[0] * xi[0] + _b6[i] * xi[1] * xi[1];

                 

      }
      break;
    default:
      throw std::runtime_error("Simplex::_calculate_basis not implemented for order > 2");
    }
      }

      inline void calculate_basis_gradients(const CutFrame &fr, const double *xi, double *va_x, double *va_y, double *va_z, double *vb_x, double *vb_y, double *vb_z)
      {
    // Switch on nP_ifem to compute basis functions gradients at given xi
    if constexpr (nSpace == 3)
    {
      // P1 on a tetrahedron: constant reference gradients (a2, a3, a4)
      for (int i = 0; i < nP_ifem; i++)
      {
        const double ga_ref[3] = {_a2[i], _a3[i], _a4[i]}, gb_ref[3] = {_b2[i], _b3[i], _b4[i]};
        double ga[3] = {0.0, 0.0, 0.0}, gb[3] = {0.0, 0.0, 0.0};
        for (int I = 0; I < 3; I++)
          for (int J = 0; J < 3; J++)
          {
            ga[I] += fr.inv_Jac[J * 3 + I] * ga_ref[J];
            gb[I] += fr.inv_Jac[J * 3 + I] * gb_ref[J];
          }
        va_x[i] = ga[0];
        va_y[i] = ga[1];
        va_z[i] = ga[2];
        vb_x[i] = gb[0];
        vb_y[i] = gb[1];
        vb_z[i] = gb[2];
      }
      return;
    }
    switch (nP_ifem)
    {
      case 3:
      for (int i = 0; i < nP_ifem; i++)
      {
        double grad_va[2] = {0.0, 0.0}, grad_vb[2] = {0.0, 0.0}, grad_va_ref[2] = {0.0, 0.0}, grad_vb_ref[2] = {0.0, 0.0};

        grad_va_ref[0] = _a2[i];
        grad_va_ref[1] = _a3[i];
        grad_vb_ref[0] = _b2[i];
        grad_vb_ref[1] = _b3[i];
        for (int I = 0; I < nSpace; I++)
        {
          for (int J = 0; J < nSpace; J++)
          {
            grad_va[I] += fr.inv_Jac[J * nSpace + I] * grad_va_ref[J];
            grad_vb[I] += fr.inv_Jac[J * nSpace + I] * grad_vb_ref[J];
          }
        }
        va_x[i] = grad_va[0];
        va_y[i] = grad_va[1];
        va_z[i] = 0.0;
        vb_x[i] = grad_vb[0];
        vb_y[i] = grad_vb[1];
        vb_z[i] = 0.0;


      }
      break;
    case 6:
      for (int i = 0; i < nP_ifem; i++)
      {
        double grad_va[2] = {0.0, 0.0}, grad_vb[2] = {0.0, 0.0}, grad_va_ref[2] = {0.0, 0.0}, grad_vb_ref[2] = {0.0, 0.0};

        grad_va_ref[0] = _a2[i] + _a4[i] * xi[1] + 2.0 * _a5[i] * xi[0];
        grad_va_ref[1] = _a3[i] + _a4[i] * xi[0] + 2.0 * _a6[i] * xi[1];
        grad_vb_ref[0] = _b2[i] + _b4[i] * xi[1] + 2.0 * _b5[i] * xi[0];
        grad_vb_ref[1] = _b3[i] + _b4[i] * xi[0] + 2.0 * _b6[i] * xi[1];


        for (int I = 0; I < nSpace; I++)
        {
          for (int J = 0; J < nSpace; J++)
          {
            grad_va[I] += fr.inv_Jac[J * nSpace + I] * grad_va_ref[J];
            grad_vb[I] += fr.inv_Jac[J * nSpace + I] * grad_vb_ref[J];
          }
        }
        va_x[i] = grad_va[0];
        va_y[i] = grad_va[1];
        va_z[i] = 0.0;
        vb_x[i] = grad_vb[0];
        vb_y[i] = grad_vb[1];
        vb_z[i] = 0.0;
      }
      break;
    default:
      throw std::runtime_error("Simplex::_calculate_basis_gradients not implemented for order > 2");
    }
      }
    };

    // GeneralizedFunctions_mix (the moment fits of H, 1-H, D of the cut) together with the two-sided IFE basis of the
    // same cut, calculated and queried as one object: calculate() fits and then builds the basis, set_quad() and
    // set_boundary_quad() select a quadrature point for both. This is the object ADR holds per cut element; the models
    // that need only the moment fit use GeneralizedFunctions_mix alone and carry no basis storage.
    template <int nSpace, int nP_ifem, int nP, int nQ, int nEBQ>
    class IFEFunctions : public equivalent_polynomials::GeneralizedFunctions_mix<nSpace, nP_ifem, nP, nQ, nEBQ, true>
    {
    public:
      using Base = equivalent_polynomials::GeneralizedFunctions_mix<nSpace, nP_ifem, nP, nQ, nEBQ, true>;
      IFEBasis<nSpace, nP_ifem, nQ, nEBQ> basis;

      inline int calculate(const double *phi_dof, const double *phi_nodes, const double *xi_r, double ma, double mb, double jf, bool isBoundary, bool scale)
      {
        (void)jf; // the lift is built for a unit jump and scaled by the caller
        // the 2D pressure-basis scaling of ma, mb (scale) is the moment fit's business only; ADR never asks for it
        if (scale)
          throw std::runtime_error("IFEFunctions: the scaled coefficients (scale = true) are not available with the IFE basis");
        const int icase = Base::calculate(phi_dof, phi_nodes, xi_r, ma, mb, 0.0, isBoundary, false);
        if (icase != 0)
          return icase;
        const auto fr = this->exact.cut_frame();
        basis.calculate_coefficients(fr, ma, mb);
        const int n = isBoundary ? nEBQ : nQ;
        double xi[(nQ > nEBQ ? nQ : nEBQ) * nSpace];
        for (int q = 0; q < n; q++)
          this->exact.permuted_reference_point(phi_nodes, &xi_r[q * 3], &xi[q * nSpace]);
        basis.evaluate_points(fr, xi, n, isBoundary);
        // make the moment fit's point 0 current as well (calculate() did; the basis has just done the same)
        return icase;
      }
      inline int calculate(const double *phi_dof, const double *phi_nodes, const double *xi_r, double ma, double mb, bool isBoundary, bool scale)
      {
        return calculate(phi_dof, phi_nodes, xi_r, ma, mb, 0.0, isBoundary, false);
      }
      inline int calculate(const double *phi_dof, const double *phi_nodes, const double *xi_r, bool isBoundary)
      {
        return calculate(phi_dof, phi_nodes, xi_r, 1.0, 1.0, 0.0, isBoundary, false);
      }
      inline void set_quad(unsigned int q)
      {
        Base::set_quad(q);
        basis.set_quad(q);
      }
      inline void set_boundary_quad(unsigned int ebq)
      {
        Base::set_boundary_quad(ebq);
        basis.set_boundary_quad(ebq);
      }
      inline double VA(int i) { return basis.VA(i); }
      inline double VA_x(int i) { return basis.VA_x(i); }
      inline double VA_y(int i) { return basis.VA_y(i); }
      inline double VA_z(int i) { return basis.VA_z(i); }
      inline double VB(int i) { return basis.VB(i); }
      inline double VB_x(int i) { return basis.VB_x(i); }
      inline double VB_y(int i) { return basis.VB_y(i); }
      inline double VB_z(int i) { return basis.VB_z(i); }
      inline double PSIA() { return basis.PSIA(); }
      inline double PSIA_x() { return basis.PSIA_x(); }
      inline double PSIA_y() { return basis.PSIA_y(); }
      inline double PSIA_z() { return basis.PSIA_z(); }
      inline double PSIB() { return basis.PSIB(); }
      inline double PSIB_x() { return basis.PSIB_x(); }
      inline double PSIB_y() { return basis.PSIB_y(); }
      inline double PSIB_z() { return basis.PSIB_z(); }
    };

    // The geometry of the cuts of a mesh, built once by prepare() and read by the residual and the Jacobian alike.
    //
    // For every element: how the solid level set (phi_s, Gamma_s) and the fluid-interface level set (phi_f, Gamma_f) cut
    // it (icase: +1 uncut with phi > 0, -1 uncut with phi < 0, 0 cut), the moment fits of H/1-H/D (and the two-sided
    // IFE basis for Gamma_f, and for the curved solid the quadric fits), the composite fits of an element cut by both
    // (2D), the flux-jump lift amplitude, whether the element carries any fluid, and the interior faces the ghost
    // penalty and the SCIFEM terms are integrated on. None of it depends on the solution.
    //
    // Storage is for cut elements only: an uncut element has only its icase (and a per-element slot index), and reads
    // from two canonical objects per fit type. Before this, two fit objects (8.5 KB each in 2D P1, 16.7 KB in 2D P2,
    // about 80 KB in 3D P2) were allocated for every element, cut or not.
    //
    // prepare() is idempotent and cheap when the geometry is unchanged (it returns at once unless `recompute` is set,
    // the mesh size changed or this is the first call), so the residual and the Jacobian each call it and neither
    // depends on the other having run first.
    template <class GfF, class GfS, int nSpace, int nDOF, int nQ, int nFaces, int nFaceQ, int nP_fit>
    class CutGeometry
    {
    public:
      static constexpr int nCompMon = (nP_fit + 1) * (nP_fit + 2) / 2; // monomials up to degree nP_fit
      static constexpr int pOrder = (nDOF == nSpace + 1) ? 1 : ((nDOF == (nSpace + 1) * (nSpace + 2) / 2) ? 2 : 0);

      // Everything prepare() reads, as plain pointers. Arrays are indexed as in the ADR arguments.
      struct Inputs
      {
        int nElements = 0;
        const int *l2g = nullptr;               // nElements * nDOF
        const double *phi_s_nodes = nullptr;     // per global dof: the solid level set
        const double *phi_f_nodes = nullptr;     // per global dof: the fluid-interface level set
        const double *mesh_dof = nullptr;        // 3 per global dof
        const double *x_ref = nullptr;           // nQ * 3 reference quadrature points
        const double *xB_ref_faces = nullptr;    // nFaces * nFaceQ * 3 reference-element face quadrature points
        const double *dV_ref = nullptr;          // nQ reference weights
        const int *elementBoundaries = nullptr;  // nElements * nFaces
        const int *elementBoundaryElements = nullptr; // 2 per face, -1 if exterior
        const double *fluxJump_element = nullptr; // prescribed flux jump of each cell (the lift amplitude)
        const double *solutionJump_nodes = nullptr; // per global dof
        bool fluxJumpLift = false;
        double mua = 1.0, mub = 1.0, jf = 0.0;
        bool curvedSolid = false;
        int *cut_label = nullptr;                // optional output, nElements: the class of the solid cut (curved solid)
      };

      // Element classification and weights
      inline int icaseS(int eN) const { return icase_s_[eN]; }
      inline int icaseF(int eN) const { return icase_f_[eN]; }
      // -1 / +1: the buried-Gamma_f logic demoted the element to the fluid on one side of Gamma_f (0: not demoted)
      inline int demoted(int eN) const { return demoted_[eN]; }
      inline bool active(int eN) const { return active_[eN] != 0; }
      // (with the curved solid the fits live in cfS(); this object is then only the canonical one the element routines touch)
      inline GfS &gfS(int eN) { return (slot_s_[eN] >= 0 && !curved_) ? gf_s_[slot_s_[eN]] : (icase_s_[eN] > 0 ? gf_s_pos_ : gf_s_neg_); }
      inline GfF &gfF(int eN) { return slot_f_[eN] >= 0 ? gf_f_[slot_f_[eN]] : (icase_f_[eN] > 0 ? gf_f_pos_ : gf_f_neg_); }
      inline cutquadric::SolidFit &cfS(int eN) { return slot_s_[eN] >= 0 ? cf_s_[slot_s_[eN]] : (icase_s_[eN] > 0 ? cf_pos_ : cf_neg_); }
      // composite fits of an element cut by both level sets (null where not valid)
      inline const double *compA(int eN) const { return comp_valid_[eN] ? &comp_A_[(std::size_t)comp_slot_[eN] * nCompMon] : nullptr; }
      inline const double *compB(int eN) const { return comp_valid_[eN] ? &comp_B_[(std::size_t)comp_slot_[eN] * nCompMon] : nullptr; }
      inline const double *compDA(int eN) const { return comp_valid_[eN] ? &comp_DA_[(std::size_t)comp_slot_[eN] * nCompMon] : nullptr; }
      inline const double *compDB(int eN) const { return comp_valid_[eN] ? &comp_DB_[(std::size_t)comp_slot_[eN] * nCompMon] : nullptr; }
      inline const double *compDFa(int eN) const { return comp_DF_valid_[eN] ? &comp_DFa_[(std::size_t)comp_slot_[eN] * nCompMon] : nullptr; }
      // amplitude of the flux-jump lift J*psi on a Gamma_f-cut cell: the per-cell J_f minus the flux jump the
      // solution-jump lift already carries (nonzero only when the solution jump varies over the cell); 0 elsewhere
      inline double fluxLift(int eN) const { return slot_f_[eN] >= 0 ? flux_lift_[slot_f_[eN]] : 0.0; }
      // interior faces, ascending, both elements carrying fluid: the faces of Gamma_s-cut elements (ghost penalty)
      // and the faces Gamma_f cuts (SCIFEM)
      inline const std::vector<int> &cutfemFaces() const { return cutfem_faces_; }
      inline const std::vector<int> &ifemFaces() const { return ifem_faces_; }
      inline int nCutS() const { return (int)gf_s_.size() + (int)cf_s_.size(); }
      inline int nCutF() const { return (int)gf_f_.size(); }
      inline bool valid() const { return valid_; }

      inline void prepare(const Inputs &in, const bool recompute)
      {
        if (valid_ && !recompute && in.nElements == nElements_ && in.curvedSolid == curved_)
          return;
        nElements_ = in.nElements;
        curved_ = in.curvedSolid;
        const int nE = in.nElements;
        icase_s_.assign(nE, 0);
        icase_f_.assign(nE, 0);
        demoted_.assign(nE, 0);
        active_.assign(nE, 0);
        slot_s_.assign(nE, -1);
        slot_f_.assign(nE, -1);
        comp_slot_.assign(nE, -1);
        comp_valid_.assign(nE, 0);
        comp_DF_valid_.assign(nE, 0);
        gf_s_.clear();
        cf_s_.clear();
        gf_f_.clear();
        flux_lift_.clear();
        comp_A_.clear(); comp_B_.clear(); comp_DA_.clear(); comp_DB_.clear(); comp_DFm_.clear(); comp_DFa_.clear();
        std::vector<int> cutfem, ifem;
        // canonical objects an uncut element reads (an uncut cell's fits are the constants 1 or 0; D and the normal
        // are never used there)
        init_canonical(in);
        GfS gfs_scratch;
        gfs_scratch.useExact = true;
        GfF gff_scratch;
        gff_scratch.useExact = true;
        cutquadric::SolidFit cf_scratch;
        for (int eN = 0; eN < nE; eN++)
        {
          double phi_s[nDOF], phi_f[nDOF], nodes[nDOF * 3];
          for (int j = 0; j < nDOF; j++)
          {
            const int dof = in.l2g[eN * nDOF + j];
            phi_s[j] = in.phi_s_nodes[dof];
            phi_f[j] = in.phi_f_nodes[dof];
            for (int I = 0; I < 3; I++)
              nodes[j * 3 + I] = in.mesh_dof[dof * 3 + I];
          }
          // --- the solid
          if (curved_)
          {
            icase_s_[eN] = cf_scratch.calculate(phi_s, nodes, in.x_ref, nQ, nP_fit);
            if (in.cut_label)
              in.cut_label[eN] = static_cast<int>(cf_scratch.label);
            if (icase_s_[eN] == 0)
            {
              slot_s_[eN] = (int)cf_s_.size();
              cf_s_.push_back(cf_scratch);
            }
          }
          else
          {
            gfs_scratch.useExact = true;
            icase_s_[eN] = gfs_scratch.calculate(phi_s, nodes, in.x_ref, false);
            if (icase_s_[eN] == 0)
            {
              slot_s_[eN] = (int)gf_s_.size();
              gf_s_.push_back(gfs_scratch);
            }
          }
          const int icase_s = icase_s_[eN];
          if (icase_s == 0)
          {
            // every interior face of a Gamma_s-cut element, owned or not (the residual is assembled with full overlap, so a
            // face term is summed on every rank that has both of its elements)
            for (int f = 0; f < nFaces; f++)
            {
              const int ebN = in.elementBoundaries[eN * nFaces + f];
              if (in.elementBoundaryElements[ebN * 2 + 1] != -1)
                cutfem.push_back(ebN);
            }
          }
          // --- the fluid interface
          gff_scratch.useExact = true;
          icase_f_[eN] = fluid_calculate(eN, gff_scratch, phi_f, nodes, in.x_ref, in.mua, in.mub, in.jf);
          const int icase_f = icase_f_[eN];
          if (icase_f == 0)
          {
            // the same fit at the quadrature points of every face (two-sided basis on the faces, for the face terms)
            fluid_calculate_faces(eN, gff_scratch, phi_f, nodes, in.xB_ref_faces, in.mua, in.mub, in.jf);
            slot_f_[eN] = (int)gf_f_.size();
            gf_f_.push_back(gff_scratch);
            flux_lift_.push_back(0.0);
          }
          // --- an element cut by both: composite fits per intersected region (2D only)
          int demoted = 0;
          if (icase_s == 0 && icase_f == 0 && nSpace == 2)
            demoted = composite_setup(eN, phi_s, phi_f, nodes);
          demoted_[eN] = (signed char)demoted;
          if (icase_f == 0)
          {
            // faces Gamma_f cuts. Interior faces only: the SCIFEM facet term needs the trace from both elements.
            for (int f = 0; f < nFaces; f++)
            {
              const int ebN = in.elementBoundaries[eN * nFaces + f];
              if (in.elementBoundaryElements[ebN * 2 + 1] == -1)
                continue;
              if (nSpace == 2)
              {
                // indexing convention for P1/P2 triangles: the face opposite vertex f joins vertices (f+1)%3 and (f+2)%3
                if (phi_f[(f + 1) % 3] * phi_f[(f + 2) % 3] < 0.0)
                  ifem.push_back(ebN);
              }
              else
              {
                // a triangular face Gamma_f cuts: its three vertices (all but the opposite node) carry both signs. Decided
                // from the face's own vertex values, so both elements, and both ranks, agree.
                double fmin = 1.0e300, fmax = -1.0e300;
                for (int m = 0; m < 4; m++)
                  if (m != f)
                  {
                    fmin = std::min(fmin, phi_f[m]);
                    fmax = std::max(fmax, phi_f[m]);
                  }
                if (fmin < 0.0 && fmax > 0.0)
                  ifem.push_back(ebN);
              }
            }
            flux_lift_[slot_f_[eN]] = fluxLiftAmplitude(eN, in, phi_f, nodes);
          }
          // --- does the element carry any fluid? (H_s, D_s or D_f nonzero at a quadrature point)
          bool act = false;
          {
            GfS &gs = gfS(eN);
            GfF &gf = gfF(eN);
            for (int k = 0; k < nQ && !act; k++)
            {
              double H_s, D_s;
              if (curved_)
              {
                H_s = cfS(eN).H(k);
                D_s = cfS(eN).D(k);
              }
              else
              {
                gs.set_quad(k);
                H_s = gs.H(0., 0.);
                D_s = gs.D(0., 0.);
              }
              gf.set_quad(k);
              double D_f = gf.D(0., 0.);
              if (compDFa(eN) != nullptr)
              {
                const double xr = in.x_ref[k * 3 + 0], yr = in.x_ref[k * 3 + 1];
                D_f = equivalent_polynomials::composite::evaluate<nP_fit>(compDFa(eN), xr, yr);
              }
              if (demoted != 0)
                D_f = 0.0;
              act = (H_s != 0.0 || D_s != 0.0 || D_f != 0.0);
            }
          }
          active_[eN] = act ? 1 : 0;
        }
        // the face sets: ascending, unique, and only faces whose two elements both carry fluid
        for (auto *list : {&cutfem, &ifem})
        {
          std::sort(list->begin(), list->end());
          list->erase(std::unique(list->begin(), list->end()), list->end());
        }
        auto filter = [&](const std::vector<int> &in_list, std::vector<int> &out)
        {
          out.clear();
          for (int ebN : in_list)
            if (active_[in.elementBoundaryElements[ebN * 2 + 0]] && active_[in.elementBoundaryElements[ebN * 2 + 1]])
              out.push_back(ebN);
        };
        filter(cutfem, cutfem_faces_);
        filter(ifem, ifem_faces_);
        valid_ = true;
      }

    private:
      bool valid_ = false, curved_ = false;
      int nElements_ = -1;
      std::vector<int> icase_s_, icase_f_, slot_s_, slot_f_, comp_slot_;
      std::vector<signed char> demoted_;
      std::vector<char> active_, comp_valid_, comp_DF_valid_;
      std::vector<GfS> gf_s_;
      std::vector<cutquadric::SolidFit> cf_s_;
      std::vector<GfF> gf_f_;
      std::vector<double> flux_lift_;
      std::vector<double> comp_A_, comp_B_, comp_DA_, comp_DB_, comp_DFm_, comp_DFa_;
      GfS gf_s_pos_, gf_s_neg_;
      GfF gf_f_pos_, gf_f_neg_;
      cutquadric::SolidFit cf_pos_, cf_neg_;
      std::vector<int> cutfem_faces_, ifem_faces_;

      // The reference element's own geometry, with the level set all positive or all negative: calculate() takes its
      // early exit and leaves the constants H = 1 / 0, D = 0
      inline void init_canonical(const Inputs &in)
      {
        double nodes[nDOF * 3] = {};
        for (int i = 0; i < nSpace + 1 && i < nDOF; i++)
          if (i > 0)
            nodes[i * 3 + (i - 1)] = 1.0;
        double phi[nDOF];
        for (int sgn = 1; sgn >= -1; sgn -= 2)
        {
          for (int j = 0; j < nDOF; j++)
            phi[j] = sgn;
          GfS &gs = sgn > 0 ? gf_s_pos_ : gf_s_neg_;
          gs = GfS();
          gs.useExact = true;
          gs.calculate(phi, nodes, in.x_ref, false);
          GfF &gf = sgn > 0 ? gf_f_pos_ : gf_f_neg_;
          gf = GfF();
          gf.useExact = true;
          gf.calculate(phi, nodes, in.x_ref, 1.0, 1.0, 0.0, false, false);
          if (pOrder == 2 && nSpace == 3)
          {
            double u10[10];
            for (int j = 0; j < 10; j++)
              u10[j] = sgn;
            (sgn > 0 ? cf_pos_ : cf_neg_).calculate(u10, nodes, in.x_ref, nQ, nP_fit);
          }
        }
      }

      // gf_f.calculate() for the interior quadrature points, naming the element if the two-sided P1 basis on a
      // tetrahedron does not exist there (obtuse tetrahedron at a particular contrast, C44 decision C4). Only that error
      // is rethrown; the rank that throws stops, so run such a mesh in serial to read it.
      inline int fluid_calculate(int eN, GfF &gf, const double *phi_f, const double *nodes, const double *x_ref, double mua, double mub, double jf)
      {
        try
        {
          const int icase = gf.calculate(phi_f, nodes, x_ref, mua, mub, jf, false, false);
          return icase;
        }
        catch (const ifem_singular_error &e)
        {
          double c[3] = {0.0, 0.0, 0.0};
          for (int j = 0; j < nSpace + 1; j++)
            for (int I = 0; I < nSpace; I++)
              c[I] += nodes[j * 3 + I] / (nSpace + 1.0);
          throw std::runtime_error("ADR immersedBoundary: element " + std::to_string(eN) + " (local number, centroid " +
                                   std::to_string(c[0]) + ", " + std::to_string(c[1]) + ", " + std::to_string(c[2]) + "): " + e.what());
        }
      }

      inline void fluid_calculate_faces(int eN, GfF &gf, const double *phi_f, const double *nodes, const double *xB_ref_faces, double mua, double mub, double jf)
      {
        try
        {
          gf.calculate(phi_f, nodes, xB_ref_faces, mua, mub, jf, true, false);
        }
        catch (const ifem_singular_error &e)
        {
          throw std::runtime_error("ADR immersedBoundary: element " + std::to_string(eN) + ": " + e.what());
        }
      }

      // Both level sets genuinely cut this element: the product weight ImH_f*H_s is not moment-exact here, so fit one
      // polynomial per intersected region instead. Returns the demotion (-1/+1) when the fluid interface lies entirely
      // inside the solid and the fluid is undivided (the empty region then gets exactly zero), else 0 with the
      // composite fits stored.
      inline int composite_setup(int eN, const double *phi_s, const double *phi_f, const double *nodes)
      {
        if constexpr (nSpace == 2)
        {
          const double phi_s3[3] = {phi_s[0], phi_s[1], phi_s[2]};
          const double phi_f3[3] = {phi_f[0], phi_f[1], phi_f[2]};
          // The fluid interface can cross this element and still lie entirely inside the solid, leaving the fluid
          // undivided. There is then no fluid-fluid interface to resolve *in the fluid*, so the two-sided IFEM
          // reconstruction has nothing to do. NOTE: icase_f is deliberately NOT changed: switching this element to the
          // standard basis would make it disagree with its neighbours about what their shared DOFs mean (one DOF cannot
          // be both a pointwise nodal value and a continued branch value). What is fixed are the weights: the empty
          // region gets exactly zero rather than whatever a fit over an empty region would produce.
          //
          // Tolerance, not exact equality: a genuine tangency between Gamma_f and Gamma_s makes one region's true area a
          // single point, but roundoff in the clip lands a few ULPs away (observed: ~1e-33 against a reference-triangle
          // scale of 0.5); fitting a degree-4 polynomial to such a sliver is unstable (off by 64x on its own area).
          // areaTol is far above that floor and far below any physically meaningful sub-area.
          double areaA = 0.0, areaB = 0.0;
          equivalent_polynomials::composite::fluid_region_areas(phi_s3, phi_f3, areaA, areaB);
          const double areaTol = 1.0e-9;
          if (areaB <= areaTol && areaA > areaTol)
            return -1; // fluid is all Omega^- (mua side)
          if (areaA <= areaTol && areaB > areaTol)
            return 1; // fluid is all Omega^+ (mub side)
          const int slot = (int)(comp_A_.size() / nCompMon);
          comp_slot_[eN] = slot;
          comp_A_.resize((std::size_t)(slot + 1) * nCompMon, 0.0);
          comp_B_.resize((std::size_t)(slot + 1) * nCompMon, 0.0);
          comp_DA_.resize((std::size_t)(slot + 1) * nCompMon, 0.0);
          comp_DB_.resize((std::size_t)(slot + 1) * nCompMon, 0.0);
          comp_DFm_.resize((std::size_t)(slot + 1) * nCompMon, 0.0);
          comp_DFa_.resize((std::size_t)(slot + 1) * nCompMon, 0.0);
          const double nodes2[6] = {nodes[0], nodes[1], nodes[3], nodes[4], nodes[6], nodes[7]};
          const std::size_t o = (std::size_t)slot * nCompMon;
          comp_valid_[eN] = (equivalent_polynomials::composite::weights<nP_fit>(phi_s3, phi_f3, &comp_A_[o], &comp_B_[o]) &&
                             equivalent_polynomials::composite::dirac_weights<nP_fit>(phi_s3, phi_f3, nodes2, &comp_DA_[o], &comp_DB_[o]))
                                ? 1 : 0;
          // Gamma_f's own Dirac measure, split by Gamma_s's sign: comp_DFa is the part of Gamma_f's chord still inside
          // the active fluid, the only part any Gamma_f Dirac-delta term (the prescribed flux jump) should see here.
          comp_DF_valid_[eN] = equivalent_polynomials::composite::edge_weights<nP_fit>(phi_f3, phi_s3, nodes2, &comp_DFm_[o], &comp_DFa_[o]) ? 1 : 0;
        }
        (void)eN; (void)phi_s; (void)phi_f; (void)nodes;
        return 0;
      }

      // Amplitude of the flux-jump lift of a Gamma_f-cut cell. j_h carries the flux jump
      // [mu dj_h/dn] = sum_i (mub JB_i grad phi_i^b - mua JA_i grad phi_i^a).n, which vanishes when J_s is the same at
      // every node of the cell (partition of unity) but not when J_s varies. The lift supplies only the rest,
      // J_f - [mu dj_h/dn], so that the trial function has the prescribed flux jump either way. n is the normal of the
      // cell's Gamma_f chord from phi_f < 0 to phi_f > 0. Cells with a constant J_s are untouched; only cells ADR.py
      // gives a lift (both signs at the vertices, none exactly zero) are corrected.
      inline double fluxLiftAmplitude(int eN, const Inputs &in, const double *phi_f, const double *nodes)
      {
        double fluxLiftJ = in.fluxJump_element[eN];
        bool liftCell = false;
        {
          int nneg = 0, npos = 0, nzero = 0;
          for (int i = 0; i < nSpace + 1; i++)
          {
            if (phi_f[i] < 0.0) nneg++;
            else if (phi_f[i] > 0.0) npos++;
            else nzero++;
          }
          liftCell = (nneg > 0 && npos > 0 && nzero == 0);
        }
        if (!(in.fluxJumpLift && liftCell))
          return fluxLiftJ;
        // the solution-jump lift (assign_jump_side's node pattern): branch a is u + JA, branch b is u + JB
        double JA[nDOF], JB[nDOF];
        for (int i = 0; i < nDOF; i++)
        {
          const double jump_i = in.solutionJump_nodes[in.l2g[eN * nDOF + i]];
          if (phi_f[i] > 0.0)
          {
            JA[i] = -jump_i;
            JB[i] = 0.0;
          }
          else
          {
            JA[i] = 0.0;
            JB[i] = jump_i;
          }
        }
        bool jumpVaries = false;
        const double jump0 = in.solutionJump_nodes[in.l2g[eN * nDOF]];
        for (int i = 1; i < nDOF; i++)
          if (in.solutionJump_nodes[in.l2g[eN * nDOF + i]] != jump0)
            jumpVaries = true;
        if (!jumpVaries)
          return fluxLiftJ;
        // unit normal of Gamma_f in the cell, from phi_f < 0 to phi_f > 0: grad of the linear interpolant
        double nf[3] = {0.0, 0.0, 0.0};
        if constexpr (nSpace == 2)
        {
          const double ex0 = nodes[3] - nodes[0], ey0 = nodes[4] - nodes[1];
          const double ex1 = nodes[6] - nodes[0], ey1 = nodes[7] - nodes[1];
          const double dp0 = phi_f[1] - phi_f[0], dp1 = phi_f[2] - phi_f[0];
          const double det = ex0 * ey1 - ex1 * ey0;
          nf[0] = (dp0 * ey1 - dp1 * ey0) / det;
          nf[1] = (ex0 * dp1 - ex1 * dp0) / det;
        }
        else
        {
          // E g = dp with E's rows the edge vectors from node 0: g = E^{-1} dp by cofactors
          double E[3][3], dp[3];
          for (int k = 0; k < 3; k++)
          {
            dp[k] = phi_f[k + 1] - phi_f[0];
            for (int I = 0; I < 3; I++)
              E[k][I] = nodes[(k + 1) * 3 + I] - nodes[I];
          }
          const double det = E[0][0] * (E[1][1] * E[2][2] - E[1][2] * E[2][1]) -
                             E[0][1] * (E[1][0] * E[2][2] - E[1][2] * E[2][0]) +
                             E[0][2] * (E[1][0] * E[2][1] - E[1][1] * E[2][0]);
          for (int I = 0; I < 3; I++)
          {
            // Cramer: replace column I of E by dp
            double M[3][3];
            for (int k = 0; k < 3; k++)
              for (int J = 0; J < 3; J++)
                M[k][J] = (J == I) ? dp[k] : E[k][J];
            nf[I] = (M[0][0] * (M[1][1] * M[2][2] - M[1][2] * M[2][1]) -
                     M[0][1] * (M[1][0] * M[2][2] - M[1][2] * M[2][0]) +
                     M[0][2] * (M[1][0] * M[2][1] - M[1][1] * M[2][0])) / det;
          }
        }
        const double nnorm = std::sqrt(nf[0] * nf[0] + nf[1] * nf[1] + nf[2] * nf[2]);
        nf[0] /= nnorm;
        nf[1] /= nnorm;
        nf[2] /= nnorm;
        GfF &gfj = gfF(eN);
        auto jhFluxJump = [&](int k)
        {
          gfj.set_quad(k);
          double s = 0.0;
          for (int i = 0; i < nDOF; i++)
            s += in.mub * JB[i] * (gfj.VB_x(i) * nf[0] + gfj.VB_y(i) * nf[1] + gfj.VB_z(i) * nf[2]) -
                 in.mua * JA[i] * (gfj.VA_x(i) * nf[0] + gfj.VA_y(i) * nf[1] + gfj.VA_z(i) * nf[2]);
          return s;
        };
        double Fj = 0.0;
        if (pOrder == 1)
          Fj = jhFluxJump(0); // P1: the branch gradients are constant on the cell
        else
        {
          // P2: average along Gamma_f, weighted by the Dirac fit
          double wsum = 0.0, fsum = 0.0, usum = 0.0;
          for (int k = 0; k < nQ; k++)
          {
            const double Fk = jhFluxJump(k);
            const double wk = gfj.D(0., 0.) * in.dV_ref[k];
            wsum += wk;
            fsum += wk * Fk;
            usum += Fk;
          }
          Fj = (std::fabs(wsum) > 1.0e-14) ? fsum / wsum : usum / nQ;
        }
        fluxLiftJ -= Fj;
        return fluxLiftJ;
      }
    };

    // The two-sided basis of a Gamma_f-cut element at the current quadrature point (after gf.set_quad): the branch
    // functions a (phi_f < 0) and b (phi_f > 0), their physical gradients and the unit flux-jump lift. It carries the
    // solution on each branch and the jump lift in one place, for the residual and the Jacobian alike.
    template <int nSpace, int nDOF>
    struct BranchBasis
    {
      double va[nDOF], vb[nDOF], va_grad[nDOF * nSpace], vb_grad[nDOF * nSpace];
      double psi_a = 0.0, psi_b = 0.0, psi_a_grad[nSpace] = {}, psi_b_grad[nSpace] = {};

      template <class GfF>
      inline void load(GfF &gf)
      {
        for (int i = 0; i < nDOF; i++)
        {
          va[i] = gf.VA(i);
          va_grad[i * nSpace + 0] = gf.VA_x(i);
          va_grad[i * nSpace + 1] = gf.VA_y(i);
          vb[i] = gf.VB(i);
          vb_grad[i * nSpace + 0] = gf.VB_x(i);
          vb_grad[i * nSpace + 1] = gf.VB_y(i);
          if constexpr (nSpace == 3)
          {
            va_grad[i * nSpace + 2] = gf.VA_z(i);
            vb_grad[i * nSpace + 2] = gf.VB_z(i);
          }
        }
        psi_a = gf.PSIA();
        psi_b = gf.PSIB();
        psi_a_grad[0] = gf.PSIA_x();
        psi_a_grad[1] = gf.PSIA_y();
        psi_b_grad[0] = gf.PSIB_x();
        psi_b_grad[1] = gf.PSIB_y();
        if constexpr (nSpace == 3)
        {
          psi_a_grad[2] = gf.PSIA_z();
          psi_b_grad[2] = gf.PSIB_z();
        }
      }
      // value and gradient of the field with nodal coefficients c on branch a / branch b
      inline void fieldA(const double *c, double &val, double *grad) const { field(c, va, va_grad, val, grad); }
      inline void fieldB(const double *c, double &val, double *grad) const { field(c, vb, vb_grad, val, grad); }
      // the solution-jump lift (nodal jumps JA on branch a, JB on branch b) and the flux-jump lift fluxLift*psi,
      // added once as known data (zero unless Gamma_f cuts the cell and a flux jump is prescribed)
      inline void lift(const double *JA, const double *JB, const double fluxLift,
                       double &uja, double *grad_uja, double &ujb, double *grad_ujb) const
      {
        fieldA(JA, uja, grad_uja);
        fieldB(JB, ujb, grad_ujb);
        uja += fluxLift * psi_a;
        ujb += fluxLift * psi_b;
        for (int I = 0; I < nSpace; I++)
        {
          grad_uja[I] += fluxLift * psi_a_grad[I];
          grad_ujb[I] += fluxLift * psi_b_grad[I];
        }
      }
      // test functions times the quadrature weight dV
      inline void weight(const double dV, double *ua_test_dV, double *ub_test_dV, double *ua_grad_test_dV, double *ub_grad_test_dV) const
      {
        for (int i = 0; i < nDOF; i++)
        {
          ua_test_dV[i] = va[i] * dV;
          ub_test_dV[i] = vb[i] * dV;
          for (int I = 0; I < nSpace; I++)
          {
            ua_grad_test_dV[i * nSpace + I] = va_grad[i * nSpace + I] * dV;
            ub_grad_test_dV[i * nSpace + I] = vb_grad[i * nSpace + I] * dV;
          }
        }
      }

    private:
      static inline void field(const double *c, const double *v, const double *gv, double &val, double *grad)
      {
        val = 0.0;
        for (int j = 0; j < nDOF; j++)
          val += c[j] * v[j];
        for (int I = 0; I < nSpace; I++)
          grad[I] = 0.0;
        for (int j = 0; j < nDOF; j++)
          for (int I = 0; I < nSpace; I++)
            grad[I] += c[j] * gv[j * nSpace + I];
      }
    };

    // The weights of the regions at one quadrature point of one element, from the geometry: the solid's Heaviside H_s,
    // Dirac D_s (and the normal-weighted Dirac Dn_s of the curved fit); the weights wA, wB of the two fluid regions
    // (Omega^- and Omega^+) for the bulk terms; the Dirac measures of Gamma_s on each side (D_s_a, D_s_b) and of Gamma_f in
    // the fluid (D_f). On an element cut by both level sets these come from the composite fit (the product ImH_f*H_s is
    // not moment-exact there), on a demoted element (Gamma_f buried in the solid, the fluid undivided) the empty region
    // gets exactly zero, and otherwise they are the usual products, unchanged. `branches` says which branch of the
    // two-sided basis a Gamma_f-cut element has (0 both, -1 only a, +1 only b: the interface touches it at a vertex or
    // an edge, or an invalid classification, +2).
    template <int nSpace, int nP_fit>
    struct RegionWeights
    {
      double H_s = 0.0, D_s = 0.0, Dn_s[nSpace] = {}, D_s_a = 0.0, D_s_b = 0.0;
      double ImH_f = 0.0, H_f = 0.0, D_f = 0.0, wA = 0.0, wB = 0.0;
      int branches = 0;

      template <class Geom>
      inline void evaluate(Geom &geom, const int eN, const int k, const double *x_ref, const bool curvedSolid)
      {
        auto &gf_s = geom.gfS(eN);
        auto &gf_f = geom.gfF(eN);
        gf_s.set_quad(k);
        gf_f.set_quad(k);
        H_s = curvedSolid ? geom.cfS(eN).H(k) : gf_s.H(0., 0.);
        D_s = curvedSolid ? geom.cfS(eN).D(k) : gf_s.D(0., 0.);
        for (int I = 0; I < nSpace; I++)
          Dn_s[I] = curvedSolid ? geom.cfS(eN).Dn(k, I) : 0.0;
        const int demoted_f = geom.demoted(eN);
        // Per-side Dirac for the solid surface. Weighting the single-cut D_s by a region fit (ImH_f * D_s) is the
        // product-of-two-fits error again, and a large one -- about 31% on the segment length. Defaults to D_s, so
        // elements cut by only one level set are untouched.
        D_s_a = D_s;
        D_s_b = D_s;
        if (demoted_f != 0)
        {
          // The whole solid surface in this element borders the one surviving fluid region.
          D_s_a = (demoted_f < 0) ? D_s : 0.0;
          D_s_b = (demoted_f < 0) ? 0.0 : D_s;
        }
        else if (geom.compDA(eN) != nullptr)
        {
          const double xr_d = x_ref[k * 3 + 0], yr_d = x_ref[k * 3 + 1];
          D_s_a = equivalent_polynomials::composite::evaluate<nP_fit>(geom.compDA(eN), xr_d, yr_d);
          D_s_b = equivalent_polynomials::composite::evaluate<nP_fit>(geom.compDB(eN), xr_d, yr_d);
        }
        ImH_f = gf_f.ImH(0., 0.);
        H_f = gf_f.H(0., 0.);
        // Region weights for the bulk terms. On an element cut by both level sets these come from the composite fit;
        // otherwise they are the usual products, unchanged.
        wA = ImH_f * H_s;
        wB = H_f * H_s;
        if (demoted_f != 0)
        {
          // Only one fluid region exists here; give it the whole fluid weight and the other none.
          wA = (demoted_f < 0) ? H_s : 0.0;
          wB = (demoted_f < 0) ? 0.0 : H_s;
        }
        else if (geom.compA(eN) != nullptr)
        {
          const double xr = x_ref[k * 3 + 0], yr = x_ref[k * 3 + 1];
          wA = equivalent_polynomials::composite::evaluate<nP_fit>(geom.compA(eN), xr, yr);
          wB = equivalent_polynomials::composite::evaluate<nP_fit>(geom.compB(eN), xr, yr);
        }
        D_f = gf_f.D(0., 0.);
        if (geom.compDFa(eN) != nullptr)
        {
          // Gamma_f's own Dirac measure, restricted to the part still inside the active fluid domain -- the plain,
          // single-curve gf_f.D(0.,0.) double-counts whatever part of Gamma_f's chord falls inside Gamma_s's masked-out
          // solid on a triply-cut element.
          const double xr_f = x_ref[k * 3 + 0], yr_f = x_ref[k * 3 + 1];
          D_f = equivalent_polynomials::composite::evaluate<nP_fit>(geom.compDFa(eN), xr_f, yr_f);
        }
        // A demoted element's Gamma_f lies entirely inside the solid: no part of it borders fluid, so the flux-jump
        // load must vanish exactly there (any fit would leave a spurious source).
        if (demoted_f != 0)
          D_f = 0.0;
        if (geom.icaseF(eN) == 0)
        {
          if (!gf_f.exact.edge && !gf_f.exact.corner) // full cut or cut on boundary of negative cell
            branches = 0;
          else if (gf_f.exact.edge == -1 || gf_f.exact.corner == -1)
            branches = -1;
          else if (gf_f.exact.edge == 1 || gf_f.exact.corner == 1)
            branches = 1;
          else
            branches = 2; // invalid: neither branch contributes
        }
        else
          branches = 0;
      }
    };
  } // namespace unfitted
} // namespace proteus
#endif
