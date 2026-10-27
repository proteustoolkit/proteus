#ifndef EQUIVALENT_POLYNOMIALS_H
#define EQUIVALENT_POLYNOMIALS_H
#include <array>
#include <cmath>
#include <cassert>
#include <cstring>
#include <iostream>
#include <iomanip>
#include <stdexcept>
#include "ifemBasisCoefficients_wrapper.h"
#include "equivalent_polynomials_coefficients.h"
#include "equivalent_polynomials_coefficients_quad.h"
#include "equivalent_polynomials_utils.h"

namespace equivalent_polynomials
{

  template <int nSpace, int nP_ifem, int nP, int nQ, int nEBQ, bool useIfemBasis = false>
  class Regularized
  {
  public:
    Regularized(bool useExact = false)
    {
      (void)useExact;
    }
    inline int calculate(const double *phi_dof, const double *phi_nodes, const double *xi_r, double ma, double mb, double jf, bool isBoundary, bool scale)
    {
      (void)phi_dof;
      (void)phi_nodes;
      (void)xi_r;
      (void)ma;
      (void)mb;
      (void)jf;
      (void)isBoundary;
      (void)scale;
      return 0;
    }
    inline int calculate(const double *phi_dof, const double *phi_nodes, const double *xi_r, double ma, double mb, bool isBoundary, bool scale)
    {
      return calculate(phi_dof, phi_nodes, xi_r, ma, mb, 0.0, isBoundary, scale);
    }
    inline int calculate(const double *phi_dof, const double *phi_nodes, const double *xi_r, bool isBoundary)
    {
      return calculate(phi_dof, phi_nodes, xi_r, 1.0, 1.0, 0.0, isBoundary, false);
    }
    inline double *get_normal()
    {
      return nullptr;
    }
    inline void set_quad(unsigned int q)
    {
      (void)q;
    }
    inline void set_boundary_quad(unsigned int ebq)
    {
      (void)ebq;
    }
    inline double H(double eps, double phi)
    {
      double h;
      if (phi > eps)
        h = 1.0;
      else if (phi < -eps)
        h = 0.0;
      else if (phi == 0.0)
        h = 0.5;
      else
        h = 0.5 * (1.0 + phi / eps + sin(M_PI * phi / eps) / M_PI);
      return h;
    }
    inline double ImH(double eps, double phi)
    {
      return 1.0 - H(eps, phi);
    }
    inline double D(double eps, double phi)
    {
      double d;
      if (phi > eps)
        d = 0.0;
      else if (phi < -eps)
        d = 0.0;
      else
        d = 0.5 * (1.0 + cos(M_PI * phi / eps)) / eps;
      return d;
    }
    inline double VA(int i) { return -1.0; };
    inline double VA_x(int i) { return -1.0; };
    inline double VA_y(int i) { return -1.0; };
    inline double VA_z(int i) { return -1.0; };
    inline double VB(int i) { return -1.0; };
    inline double VB_x(int i) { return -1.0; };
    inline double VB_y(int i) { return -1.0; };
    inline double VB_z(int i) { return -1.0; };
    inline double PSIA() { return 0.0; };
    inline double PSIA_x() { return 0.0; };
    inline double PSIA_y() { return 0.0; };
    inline double PSIB() { return 0.0; };
    inline double PSIB_x() { return 0.0; };
    inline double PSIB_y() { return 0.0; };
  };

  template <int nSpace, int nP_ifem, int nP, int nQ, int nEBQ, bool useIfemBasis = false>
  class Simplex
  {
  public:
    Simplex(bool useExact = true)
    {
      if (nSpace == 1)
        assert(nDOF == nP + 1);
      else if (nSpace == 2)
        assert(nDOF == (nP + 1) * (nP + 2) / 2);
      else if (nSpace == 3)
        assert(nDOF == (nP + 1) * (nP + 2) * (nP + 3) / 6);
      else
        assert(false);
      _set_Ainv<nSpace, nP>(Ainv);
      for (int i = 0; i < nSpace; i++)
        level_set_normal[i] = 0.0;
      level_set_normal[0] = 1.0;
    }

    inline int calculate(const double *phi_dof, const double *phi_nodes, const double *xi_r, double ma, double mb, double jf, bool isBoundary, bool scale);

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
      assert(q >= 0);
      assert(q < nQ);
      // The edge/corner special cases are an IFEM addition: pre-IFEM these degenerate
      // cells still took their H/ImH/D straight from the moment fit, and forcing D = 0
      // here removes the interface measure that MCorr's mass correction integrates
      // over (vortex2D_exactHeaviside). Keep them on the IFEM path only.
      if (useIfemBasis && (edge == -1 || corner == -1))
      {
        _H_q = 0.0;
        _ImH_q = 1.0;
        _D_q = 0.0;
      }
      else if (useIfemBasis && (edge == 1 || corner == 1))
      {
        _H_q = 1.0;
        _ImH_q = 0.0;
        _D_q = 0.0;
      }
      else if (inside_out)
      {
        // std::cout << "Inside out \t corner " << corner << "\t edge " << edge << std::endl;
        _H_q = _ImH[q];
        _ImH_q = _H[q];
        _D_q = _D[q];
      }
      else
      {
        // std::cout << "corner " << corner << "\t edge " << edge << std::endl;
        _H_q = _H[q];
        _ImH_q = _ImH[q];
        _D_q = _D[q];
      }
      // basis functions already adjusted for inside_out
      for (int i = 0; i < nP_ifem; i++)
      {
        _va_q[i] = _va[q * nP_ifem + i];
        _vb_q[i] = _vb[q * nP_ifem + i];
      }
      for (int i = 0; i < nP_ifem; i++)
      {
        _va_x_q[i] = _va_x[q * nP_ifem + i];
        _va_y_q[i] = _va_y[q * nP_ifem + i];
        _va_z_q[i] = _va_z[q * nP_ifem + i];
        _vb_x_q[i] = _vb_x[q * nP_ifem + i];
        _vb_y_q[i] = _vb_y[q * nP_ifem + i];
        _vb_z_q[i] = _vb_z[q * nP_ifem + i];
      }
      const bool cut = !(edge == -1 || corner == -1 || edge == 1 || corner == 1);
      for (int i = 0; i < 6; i++)
        _psi_q[i] = cut ? _psi[q * 6 + i] : 0.0;
    }

    inline void set_boundary_quad(unsigned int ebq)
    {
      assert(ebq >= 0);
      assert(ebq < nEBQ);
      // The edge/corner special cases are an IFEM addition: pre-IFEM these degenerate
      // cells still took their H/ImH/D straight from the moment fit, and forcing D = 0
      // here removes the interface measure that MCorr's mass correction integrates
      // over (vortex2D_exactHeaviside). Keep them on the IFEM path only.
      if (useIfemBasis && (edge == -1 || corner == -1))
      {
        _H_q = 0.0;
        _ImH_q = 1.0;
        _D_q = 0.0;
      }
      else if (useIfemBasis && (edge == 1 || corner == 1))
      {
        _H_q = 1.0;
        _ImH_q = 0.0;
        _D_q = 0.0;
      }
      else if (inside_out)
      {
        _H_q = _ImH_ebq[ebq];
        _ImH_q = _H_ebq[ebq];
        _D_q = _D_ebq[ebq];
      }
      else
      {
        _H_q = _H_ebq[ebq];
        _ImH_q = _ImH_ebq[ebq];
        _D_q = _D_ebq[ebq];
      }
      // basis functions already adjusted for inside_out
      for (int i = 0; i < nP_ifem; i++)
      {
        _va_q[i] = _va_ebq[ebq * nP_ifem + i];
        _vb_q[i] = _vb_ebq[ebq * nP_ifem + i];
      }
      for (int i = 0; i < nP_ifem; i++)
      {
        _va_x_q[i] = _va_x_ebq[ebq * nP_ifem + i];
        _va_y_q[i] = _va_y_ebq[ebq * nP_ifem + i];
        _vb_x_q[i] = _vb_x_ebq[ebq * nP_ifem + i];
        _vb_y_q[i] = _vb_y_ebq[ebq * nP_ifem + i];
      }
      const bool cut = !(edge == -1 || corner == -1 || edge == 1 || corner == 1);
      for (int i = 0; i < 6; i++)
        _psi_q[i] = cut ? _psi_ebq[ebq * 6 + i] : 0.0;
    }

    inline double *get_H() { return _H; };
    inline double *get_ImH() { return _ImH; };
    inline double *get_D() { return _D; };
    inline double H(double eps, double phi) { return _H_q; };
    inline double ImH(double eps, double phi) { return _ImH_q; };
    inline double D(double eps, double phi) { return _D_q; };
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
    inline double PSIB() { return _psi_q[3]; };
    inline double PSIB_x() { return _psi_q[4]; };
    inline double PSIB_y() { return _psi_q[5]; };
    inline double *get_normal()
    {
      return level_set_normal;
    }
    // Value-initialised: gf_f_cache/gf_s_cache in ADR.h are built once with
    // assign(n, GfType()) and reused, so any entry whose calculate() was not
    // invoked in the current pass is read with whatever the default-constructed
    // temporary happened to hold.  edge/corner in particular are *branched on*
    // in set_quad, which turns indeterminate bytes into a discretely different
    // H/ImH/D and a discretely different Jacobian.
    bool inside_out = false, quad_cut = false;
    static const unsigned int nN = nSpace + 1;
    // Sized for both of its roles: the cut geometry needs one corrected value per
    // topological vertex (nN), while the P2 IFEM basis needs one per trial dof
    // (nP_ifem = 6). Non-ADR callers instantiate with nP_ifem < nN -- VOF/MCorr
    // with 2, RANS3PF with 1 -- so sizing by nP_ifem alone left the vertex entries
    // past the end of the array, and _calculate_cuts read them back as garbage.
    static const unsigned int nDOF_phi = (useIfemBasis && nP_ifem > nN) ? nP_ifem : nN;
    double phi_dof_corrected[nDOF_phi] = {};
    double cut_barycenter[3] = {0., 0., 0.};
    int edge = 0, corner = 0;
    bool split = false;
    bool flip_the_cell = false;

  private:
    int P2_ifem_case = 0;
    // These eight are the arrays VA()/VB()/VA_x()/... hand straight back to the
    // caller, so they are the read path for the SCIFEM facet terms in ADR.h.
    // 3567c2a3 initialised the storage arrays (_va, _va_ebq, _a*, _b*) but left
    // this scratch row alone -- only the three scalars ahead of it got values.
    double _H_q = 0., _ImH_q = 0., _D_q = 0., _va_q[nP_ifem] = {}, _vb_q[nP_ifem] = {},
        _va_x_q[nP_ifem] = {}, _va_y_q[nP_ifem] = {}, _va_z_q[nP_ifem] = {},
        _vb_x_q[nP_ifem] = {}, _vb_y_q[nP_ifem] = {}, _vb_z_q[nP_ifem] = {};
    // Sized nDOF_phi for the same reason as phi_dof_corrected: the cut geometry
    // permutes nN topological vertices, while the P2 IFEM basis also permutes its
    // mid-side dofs (nP_ifem = 6). Sizing by nP_ifem alone left permutation[nN-1]
    // unwritten for the nP_ifem < nN callers (VOF/MCorr 2, RANS3PF 1), and
    // _calculate_cuts then indexed phi_dof_corrected with that garbage.
    unsigned int root_node = 0, permutation[nDOF_phi] = {};
    // Same nN-vs-nP_ifem split as phi_dof_corrected/permutation: the cut geometry
    // needs one entry per topological vertex (the Jacobian reads nodes[(1+i)*3+I]
    // for i < nN-1), while the P2 IFEM basis needs one per trial dof. Sizing by
    // nP_ifem alone left the last vertex unfilled for nP_ifem < nN callers and the
    // Jacobian was built from memory past the end of nodes[].
    double phi[nDOF_phi] = {}, nodes[nDOF_phi * 3] = {};
    double _a1[nP_ifem] = {}, _a2[nP_ifem] = {}, _a3[nP_ifem] = {}, _b1[nP_ifem] = {}, _b2[nP_ifem] = {}, _b3[nP_ifem] = {};
    double _a4[nP_ifem] = {}, _a5[nP_ifem] = {}, _a6[nP_ifem] = {}, _b4[nP_ifem] = {}, _b5[nP_ifem] = {}, _b6[nP_ifem] = {};
    // unit flux-jump lift: coefficients (same monomial order as _a1.._a6) and values/gradients
    // [psi_a, psi_a_x, psi_a_y, psi_b, psi_b_x, psi_b_y] at the quadrature points
    double _lift_a[6] = {0., 0., 0., 0., 0., 0.}, _lift_b[6] = {0., 0., 0., 0., 0., 0.};
    double _psi[nQ * 6] = {}, _psi_ebq[nEBQ * 6] = {}, _psi_q[6] = {0., 0., 0., 0., 0., 0.};
    inline void _calculate_lift(const double *xi, double *psi);
    double Jac[nSpace * nSpace] = {}, inv_Jac[nSpace * nSpace] = {}, det_Jac = 0.;
    double level_set_normal[nSpace], X_0[nSpace], phys_nodes_cut[(nN - 1) * 3], THETA_01, THETA_02, THETA_31, THETA_32, phys_nodes_cut_quad_01[3], phys_nodes_cut_quad_02[3], phys_nodes_cut_quad_31[3], phys_nodes_cut_quad_32[3];
    static const unsigned int nDOF = ((nSpace - 1) / 2) * (nSpace - 2) * (nP + 1) * (nP + 2) * (nP + 3) / 6 + (nSpace - 1) * (3 - nSpace) * (nP + 1) * (nP + 2) / 2 + (2 - nSpace) * ((3 - nSpace) / 2) * (nP + 1);
    double Ainv[nDOF * nDOF];
    double C_H[nDOF] = {}, C_ImH[nDOF] = {}, C_D[nDOF] = {};
    inline int _calculate_permutation(const double *phi_dof, const double *phi_nodes);
    inline void _calculate_cuts();
    inline void _calculate_cuts_quad();
    inline void _calculate_C();
    inline void _correct_phi(const double *phi_dof, const double *phi_nodes);
    double _H[nQ] = {}, _ImH[nQ] = {}, _D[nQ] = {}, _va[nQ * nP_ifem] = {}, _vb[nQ * nP_ifem] = {};
    double _H_ebq[nEBQ] = {}, _ImH_ebq[nEBQ] = {}, _D_ebq[nEBQ] = {}, _va_ebq[nEBQ * nP_ifem] = {}, _vb_ebq[nEBQ * nP_ifem] = {}; // cek hack: this is confusing because we use no suffice for the q arrays and _ebq for the ebq arrays, then use _q above for generic quad point
    double _va_x[nQ * nP_ifem] = {}, _va_y[nQ * nP_ifem] = {}, _va_z[nQ * nP_ifem] = {}, _vb_x[nQ * nP_ifem] = {}, _vb_y[nQ * nP_ifem] = {}, _vb_z[nQ * nP_ifem] = {};
    double _va_x_ebq[nEBQ * nP_ifem] = {}, _va_y_ebq[nEBQ * nP_ifem] = {}, _va_z_ebq[nEBQ * nP_ifem] = {}, _vb_x_ebq[nEBQ * nP_ifem] = {}, _vb_y_ebq[nEBQ * nP_ifem] = {}, _vb_z_ebq[nEBQ * nP_ifem] = {};
    inline void _calculate_basis_coefficients(const double ma, const double mb, const double jf);
    inline void _calculate_basis(const double *xi, double *va, double *vb);
    inline void _calculate_basis_gradients(const double *xi, double *va_x, double *va_y, double *vb_x, double *vb_y);
  };

  template <int nSpace, int nP_ifem, int nP, int nQ, int nEBQ, bool useIfemBasis>
  inline void Simplex<nSpace, nP_ifem, nP, nQ, nEBQ, useIfemBasis>::_calculate_C()
  {
    double b_H[nDOF], b_ImH[nDOF], b_dH[nDOF * nSpace], b_D[nDOF * nSpace];
    if (quad_cut)
    {
      _calculate_b<nP>(THETA_01, THETA_02, THETA_31, THETA_32,
                       phi_dof_corrected[permutation[0]],
                       phi_dof_corrected[permutation[1]],
                       phi_dof_corrected[permutation[2]],
                       phi_dof_corrected[permutation[3]],
                       b_H, b_ImH, b_D);
      if (inside_out) // todo handle insdie out for H/ImH/D in a simplified/unified way
      {
        for (unsigned int i = 0; i < nDOF; i++)
        {
          b_D[i] = -b_D[i];
        }
      }
      for (unsigned int i = 0; i < nDOF; i++)
      {
        C_H[i] = 0.0;
        C_ImH[i] = 0.0;
        C_D[i] = 0.0;
        for (unsigned int j = 0; j < nDOF; j++)
        {
          assert(!std::isnan(Ainv[i * nDOF + j]));
          assert(!std::isnan(b_H[j]));
          assert(!std::isnan(b_ImH[j]));
          assert(!std::isnan(b_D[j]));
          C_H[i] += Ainv[i * nDOF + j] * b_H[j];
          C_ImH[i] += Ainv[i * nDOF + j] * b_ImH[j];
          C_D[i] += Ainv[i * nDOF + j] * b_D[j];
        }
        // only if direct boundary integral is used
        // C_D[i] /= det_Jac;
      }
    }
    else
    {
      _calculate_b<nSpace, nP>(X_0, b_H, b_ImH, b_dH);

      double Jt_dphi_dx[nSpace];
      for (unsigned int I = 0; I < nSpace; I++)
      {
        Jt_dphi_dx[I] = 0.0;
        for (unsigned int J = 0; J < nSpace; J++)
          Jt_dphi_dx[I] += Jac[J * nSpace + I] * level_set_normal[J];
      }
      for (unsigned int i = 0; i < nDOF; i++)
      {
        C_H[i] = 0.0;
        C_ImH[i] = 0.0;
        C_D[i] = 0.0;
        for (unsigned int j = 0; j < nDOF; j++)
        {
          C_H[i] += Ainv[i * nDOF + j] * b_H[j];
          C_ImH[i] += Ainv[i * nDOF + j] * b_ImH[j];
          for (unsigned int I = 0; I < nSpace; I++)
          {
            if (fabs(Jt_dphi_dx[I]) > 0.0)
              C_D[i] -= Ainv[i * nDOF + j] * b_dH[j * nSpace + I] / (Jt_dphi_dx[I]);
          }
        }
      }
      // The Dirac is a surface measure and must be independent of the caller's node ordering,
      // but C_D above inherits sign(det_Jac) through Jt_dphi_dx (built from Jac). Elements
      // handed in with negative orientation therefore produced a sign-flipped D (test_2D/3D).
      // Correct the sign here rather than reordering the permutation: the permutation feeds
      // solve_ifem_basis_coefficients, and reordering it changes the P2 IFEM basis.
      if (det_Jac < 0.0)
        for (unsigned int i = 0; i < nDOF; i++)
          C_D[i] = -C_D[i];
    }
  }

  template <int nSpace, int nP_ifem, int nP, int nQ, int nEBQ, bool useIfemBasis>
  inline int Simplex<nSpace, nP_ifem, nP, nQ, nEBQ, useIfemBasis>::_calculate_permutation(const double *phi_dof, const double *phi_nodes)
  {
    if (flip_the_cell)
    {
      //  std::cout << "Flipping the permutation from: " << std::endl;
      // for (unsigned int i = 0; i < nP_ifem; i++)
      // {
      //   std::cout << permutation[i] << " ";
      // }
      // std::cout << std::endl;
      int temp = permutation[1];
      permutation[1] = permutation[2];
      permutation[2] = temp;
      temp = permutation[3];
      permutation[3] = permutation[5];
      permutation[5] = temp;
      flip_the_cell = false;

      // std::cout << "Flipping the permutation to: " << std::endl;
      // for (unsigned int i = 0; i < nP_ifem; i++)
      // {
      //   std::cout << permutation[i] << " ";
      // }
      // std::cout << std::endl;
    }
    else
    {
      int p_i, pcount = 0, n_i, ncount = 0, z_i, zcount = 0;
      corner = 0;
      edge = 0;
      root_node = 0;
      inside_out = false;
      quad_cut = false;

      const double eps = 1.0e-8;

      for (unsigned int i = 0; i < nN; i++)
      { 
        // std::cout << "phi_nodes[" << i << "] = " << phi_nodes[i*3+0] << ", " << phi_nodes[i*3+1] << ", " << phi_nodes[i*3+2] << std::endl;
        // std::cout << "phi_dof[" << i << "] = " << phi_dof[i] << std::endl << std::endl;
        if (phi_dof[i] > eps)
        {
          if (pcount == 0)
            p_i = i;
          pcount += 1;
        }
        else if (phi_dof[i] < -eps)
        {
          if (ncount == 0)
            n_i = i;
          ncount += 1;
        }
        else
        {
          if (zcount == 0)
            z_i = i;
          zcount += 1;
        }
      }
      // std::cout << "zcount " << zcount << "\t pcount " << pcount << "\t ncount " << ncount << std::endl;
      if (pcount == nN)
      {
        // All positive: element is fully in the +1 domain
        // std::cout << "This is a fully positive element." << std::endl;
        return 1;
      }
      else if (ncount == nN)
      {
        // All negative: element is fully in the -1 domain
        // std::cout << "This is a fully negative element." << std::endl;
        return -1;
      }
      else if (ncount == 1)
      {
        if (zcount == nN - 1) // for P1 ifem, interface is on an element boundary and the element is fully in the -1 domain.
        {
          edge = -1;
          // std::cout << "This is a edge case with negative side element." << std::endl;
        }
        else if (zcount == 1 && pcount == 1)
        {
          // std::cout << "This is a true split element" << std::endl;
          split = true;
        }
        root_node = n_i;
      }
      else if (pcount == 1)
      {
        if (zcount == nN - 1) // for P1 ifem, interface is on an element boundary and the element is fully in the +1 domain.
        {
          edge = 1;
          // std::cout << "This is a edge case with positive side element." << std::endl;
        }
        root_node = p_i;
        inside_out = true;
      }
      else if (nSpace == 3 && pcount == 2 && ncount == 2)
      {
        // special case only in 3D
        quad_cut = true;
        root_node = n_i;
      }
      else
      {
        assert(zcount < nN - 1);
        // std::cout << "corner case: zcount " << zcount << "\t pcount " << pcount << "\t ncount " << ncount << std::endl;
        if (pcount && !ncount)
        {
          corner = 1; // The interface passes through a corner node and element is in + side
          assert(pcount == nN - 1);
          root_node = z_i;
          // ADR's P2 IFEM basis needs the ma/mb swap here, but pre-IFEM this branch
          // left inside_out false, and set_quad uses it to exchange H and ImH -- which
          // inverts the phase fractions MCorr/VOF integrate. IFEM path only.
          if (useIfemBasis)
            inside_out = true;
        }
        else if (ncount && !pcount)
        {
          corner = -1; // The interface passes through a corner node and element is in - side
          assert(ncount == nN - 1);
          // Pre-IFEM this rooted at n_i; the IFEM basis wants z_i. Keep the old root for
          // the non-IFEM callers so their permutation/Jacobian are unchanged.
          root_node = useIfemBasis ? z_i : n_i;
        }
        else
          assert(false);
      }
      // std::cout << "root_node: " << root_node << "\t inside_out: " << inside_out << "\t quad_cut: " << quad_cut << std::endl;
      for (unsigned int i = 0; i < nDOF_phi; i++)
      {
        // Works for both P1 and P2 IFEM. The loop bound is nDOF_phi = max(nN, nP_ifem):
        // the geometry always needs all nN corner entries, while P2 IFEM needs its
        // mid-side dofs on top of them. Bounding by nP_ifem alone left the corner
        // entries past nP_ifem unwritten whenever nP_ifem < nN (VOF/MCorr 2,
        // RANS3PF 1), and _calculate_cuts read them back as garbage indices.
        // the cycling modulus is nN (topological vertices): the first nN entries rotate the
        // corner nodes starting at the root, and the P2 mid-side DOFs shadow them at offset nN.
        // The modulus must NOT be a literal 3 -- that is only nN for a triangle, and on a
        // tetrahedron (nN=4) it produced out-of-range indices such as 3+(2+3)%3 = 5, giving a
        // garbage Jacobian and nan H/ImH/D (test_3D, phi = [1,1,-1,-1]).
        if (i < nN)
          permutation[i] = (root_node + i) % nN;
        else
          permutation[i] = nN + (root_node + i) % nN;
      }
      if (quad_cut)
      {
        if (phi_dof[permutation[nN - 1]] > 0.0)
        {
          int tmp = permutation[nN - 1];
          if (phi_dof[permutation[nN - 2]] < 0.0)
          {
            permutation[nN - 1] = permutation[nN - 2];
            permutation[nN - 2] = tmp;
          }
          else if (phi_dof[permutation[nN - 3]] < 0.0)
          {
            permutation[nN - 1] = permutation[nN - 3];
            permutation[nN - 3] = tmp;
          }
          else
            assert(false);
        }
        assert(phi_dof[permutation[0]] < 0.0);
        assert(phi_dof[permutation[3]] < 0.0);
        assert(phi_dof[permutation[1]] > 0.0);
        assert(phi_dof[permutation[2]] > 0.0);
      }
      // std::cout << "pcount " << pcount << "\t ncount " << ncount << "\t zcount " << zcount << "\t root node = " << root_node << std::endl;
    }
    for (unsigned int i = 0; i < nDOF_phi; i++)
    {
      phi[i] = phi_dof[permutation[i]];
      // // std::cout << "ref idx: " << i << "\t real idx(permutation[i]): " << permutation[i] << std::endl;
      for (unsigned int I = 0; I < 3; I++)
      {
        nodes[i * 3 + I] = phi_nodes[permutation[i] * 3 + I]; // nodes always 3D

        // std::cout << "nodes[" << i * 3 + I << "] = " << nodes[i * 3 + I] << std::endl;
      }
    }
    double JacTest[nSpace * nSpace];
    for (unsigned int I = 0; I < nSpace; I++)
    {
      for (unsigned int i = 0; i < nN - 1; i++)
      {
        Jac[I * nSpace + i] = nodes[(1 + i) * 3 + I] - nodes[I];
        JacTest[I * nSpace + i] = phi_nodes[(1 + i) * 3 + I] - phi_nodes[I];
        // std::cout << "Jac[" << I * nSpace + i << "] = " << Jac[I * nSpace + i] << std::endl;
        // std::cout << "JacTest[" << I * nSpace + i << "] = " << JacTest[I * nSpace + i] << std::endl;
      }
    }
    det_Jac = det<nSpace>(Jac);
    double det_JacTest = det<nSpace>(JacTest);
    /* assert(det_JacTest >= 0.0); */
    /* assert(det_Jac >= 0.0); */
    
    if (det_Jac < 0.0 && flip_the_cell)
    {
      if (quad_cut) // flip the two internal positive nodes
      {
        double tmp = permutation[2];
        permutation[2] = permutation[1];
        permutation[1] = tmp;
      }
      else // flip the last two nodes
      {
        double tmp = permutation[nN - 1];
        permutation[nN - 1] = permutation[nN - 2];
        permutation[nN - 2] = tmp;
      }
      for (unsigned int i = 0; i < nDOF_phi; i++)
      {
        phi[i] = phi_dof[permutation[i]];
        for (unsigned int I = 0; I < 3; I++)
        {
          nodes[i * 3 + I] = phi_nodes[permutation[i] * 3 + I]; // nodes always 3D
        }
      }
      for (unsigned int i = 0; i < nN - 1; i++)
        for (unsigned int I = 0; I < nSpace; I++)
          Jac[I * nSpace + i] = nodes[(1 + i) * 3 + I] - nodes[I];
      det_Jac = det<nSpace>(Jac);
      assert(det_Jac > 0);
      if (nSpace == 1)
        inside_out = true;
    }
    inv<nSpace>(Jac, inv_Jac);
    return 0;
  }

  template <int nSpace, int nP_ifem, int nP, int nQ, int nEBQ, bool useIfemBasis>
  inline void Simplex<nSpace, nP_ifem, nP, nQ, nEBQ, useIfemBasis>::_calculate_cuts()
  {
    const double eps = 1.0e-8;
    for (unsigned int i = 0; i < nN - 1; i++)
    {
      if (useIfemBasis && (corner == 1 || corner == -1))
      {
        X_0[i] = 0.0;
        for (unsigned int I = 0; I < 3; I++)
        {
          phys_nodes_cut[i * 3 + I] = nodes[I];
        }
      }
      else if (phi[i + 1] * phi[0] < 0.0)
      {
        X_0[i] = 0.5 - 0.5 * (phi[i + 1] + phi[0]) / (phi[i + 1] - phi[0]);
        assert(X_0[i] <= 1.0);
        assert(X_0[i] >= 0.0);
        for (unsigned int I = 0; I < 3; I++)
        {
          phys_nodes_cut[i * 3 + I] = (1 - X_0[i]) * nodes[I] + X_0[i] * nodes[(1 + i) * 3 + I];
          // std::cout << "nodes[" << I << "] = " << nodes[I] << std::endl;
          // std::cout << "phys_nodes_cut[" << i*3 + I << "] = " << phys_nodes_cut[i*3 + I] << std::endl << std::endl;
        }
      }
      else
      {
        // assert(phi[i+1] < eps);
        if (phi[i + 1] < eps)
        {
          X_0[i] = 1.0;
          for (unsigned int I = 0; I < 3; I++)
          {
            phys_nodes_cut[i * 3 + I] = nodes[(1 + i) * 3 + I];
          }
        }
        else
        {
          X_0[i] = 0.0;
          for (unsigned int I = 0; I < 3; I++)
          {
            phys_nodes_cut[i * 3 + I] = nodes[I];
          }
        }
      }
    }
    // std::cout << "X_0: \t" << X_0[0] << ", " << X_0[1] << std::endl;
    // std::cout << "phys_nodes_cut: " << std::endl;
    // for (unsigned int i = 0; i < nN - 1; i++)
    // {
    //   std::cout << "phys_nodes_cut[" << i << "]: \t" << phys_nodes_cut[i * 3 + 0] << ", " << phys_nodes_cut[i * 3 + 1] << ", " << phys_nodes_cut[i * 3 + 2] << std::endl;
    // }
    // Redo the permutation for the case: (x0>0.5 and y0<=0.5)
    if (nP_ifem == 6 && (X_0[0] > 0.5 && X_0[1] <= 0.5))
    {
      // std::cout << "Case (X_0[0] > 0.5 && X_0[1] <= 0.5) detected in _calculate_cuts(), flipping the cell" << std::endl;
      flip_the_cell = true;
    }
  }

  template <int nSpace, int nP_ifem, int nP, int nQ, int nEBQ, bool useIfemBasis>
  inline void Simplex<nSpace, nP_ifem, nP, nQ, nEBQ, useIfemBasis>::_calculate_cuts_quad()
  {
    const double eps = 1.0e-8, Imeps = 1.0 - eps;
    THETA_01 = 0.5 - 0.5 * (phi[1] + phi[0]) / (phi[1] - phi[0]);
    THETA_02 = 0.5 - 0.5 * (phi[2] + phi[0]) / (phi[2] - phi[0]);
    THETA_31 = 0.5 - 0.5 * (phi[1] + phi[3]) / (phi[1] - phi[3]);
    THETA_32 = 0.5 - 0.5 * (phi[2] + phi[3]) / (phi[2] - phi[3]);
    if ((THETA_01 < eps || THETA_01 > Imeps) || (THETA_02 < eps || THETA_02 > Imeps) || (THETA_31 < eps || THETA_31 > Imeps) || (THETA_32 < eps || THETA_32 > Imeps))
    {
      THETA_01 = fmin(Imeps, fmax(eps, 0.5 - 0.5 * (phi[1] + phi[0]) / (phi[1] - phi[0])));
      THETA_02 = fmin(Imeps, fmax(eps, 0.5 - 0.5 * (phi[2] + phi[0]) / (phi[2] - phi[0])));
      THETA_31 = fmin(Imeps, fmax(eps, 0.5 - 0.5 * (phi[1] + phi[3]) / (phi[1] - phi[3])));
      THETA_32 = fmin(Imeps, fmax(eps, 0.5 - 0.5 * (phi[2] + phi[3]) / (phi[2] - phi[3])));
    }
    for (unsigned int I = 0; I < 3; I++)
    {
      phys_nodes_cut_quad_01[I] = (1 - THETA_01) * nodes[I] + THETA_01 * nodes[1 * 3 + I];
      phys_nodes_cut_quad_02[I] = (1 - THETA_02) * nodes[I] + THETA_02 * nodes[2 * 3 + I];
      phys_nodes_cut_quad_31[I] = (1 - THETA_31) * nodes[3 * 3 + I] + THETA_31 * nodes[1 * 3 + I];
      phys_nodes_cut_quad_32[I] = (1 - THETA_32) * nodes[3 * 3 + I] + THETA_32 * nodes[2 * 3 + I];
    }
  }

  template <int nSpace, int nP_ifem, int nP, int nQ, int nEBQ, bool useIfemBasis>
  inline void Simplex<nSpace, nP_ifem, nP, nQ, nEBQ, useIfemBasis>::_correct_phi(const double *phi_dof, const double *phi_nodes)
  {
    memset(cut_barycenter, 0, 3 * sizeof(double));
    const double one_by_nNm1 = 1.0 / (nN - 1.0);
    if (quad_cut)
    {
      for (unsigned int I = 0; I < nSpace; I++)
        cut_barycenter[I] += 0.25 * (phys_nodes_cut_quad_01[I] +
                                     phys_nodes_cut_quad_02[I] +
                                     phys_nodes_cut_quad_31[I] +
                                     phys_nodes_cut_quad_32[I]);
    }
    else
    {
      for (unsigned int i = 0; i < nN - 1; i++)
      {
        assert(!std::isnan(phys_nodes_cut[i * 3 + 0]));
        assert(!std::isnan(phys_nodes_cut[i * 3 + 1]));
        assert(!std::isnan(phys_nodes_cut[i * 3 + 2]));
        for (unsigned int I = 0; I < nSpace; I++)
          cut_barycenter[I] += phys_nodes_cut[i * 3 + I] * one_by_nNm1;
      }
    }
    for (unsigned int i = 0; i < nDOF_phi; i++)
    {
      phi_dof_corrected[i] = 0.0;
      for (unsigned int I = 0; I < nSpace; I++)
      {
        phi_dof_corrected[i] += level_set_normal[I] * (phi_nodes[i * 3 + I] - cut_barycenter[I]);
      }
      // ensure sdf sign convention consistent with input phi
      if (phi_dof_corrected[i] * phi_dof[i] < 0.0)
      {
        phi_dof_corrected[i] *= -1.0;
      }
    }
  }

  template <int nSpace, int nP_ifem, int nP, int nQ, int nEBQ, bool useIfemBasis>
  inline void Simplex<nSpace, nP_ifem, nP, nQ, nEBQ, useIfemBasis>::_calculate_basis_coefficients(const double ma, const double mb, const double jf)
  {
    assert(nSpace == 2);
    assert(nN == 3);
    double nx = 0.0, ny = 0.0;
    if (inside_out)
    {
      nx = -level_set_normal[0];
      ny = -level_set_normal[1];
    }
    else
    {
      nx = level_set_normal[0];
      ny = level_set_normal[1];
    }
    double Jit00 = inv_Jac[0 * nSpace + 0],
           Jit01 = inv_Jac[1 * nSpace + 0],
           Jit10 = inv_Jac[0 * nSpace + 1],
           Jit11 = inv_Jac[1 * nSpace + 1];
    double x0 = X_0[0],
           y0 = X_0[1];
    const double *vall = nullptr;

    // std::cout << "X0 = " << x0 << std::endl << "Y0 = " << y0 << std::endl;
    // std::cout << "NX = " << nx << std::endl << "NY = " << ny << std::endl;
    // std::cout << "MUA = " << ma << std::endl << "MUB = " << mb << std::endl << "jf = " << jf << std::endl;
    // std::cout << "Jit00 = " << Jit00 << std::endl << "Jit01 = " << Jit01 << std::endl << "Jit10 = " << Jit10 << std::endl << "Jit11 = " << Jit11 << std::endl;
    // std::cout << "nx: " << nx << "\t ny: " << ny << std::endl;

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
        int i = permutation[j];
        double v[3] = {0.0, 0.0, 0.0};
        v[0] = vall[j * 3 + 0];
        v[1] = vall[j * 3 + 1];
        v[2] = vall[j * 3 + 2];

        if (corner || edge)
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
          if (inside_out)
          {
            const std::array<double, 6> coeffs = proteus::solve_ifem_basis_coefficients(
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

            const std::array<double, 6> coeffs = proteus::solve_ifem_basis_coefficients(
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
        int i = permutation[j];
        double v[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};//, grad_va[2] = {0.0, 0.0}, grad_vb[2] = {0.0, 0.0}, grad_va_ref[2] = {0.0, 0.0}, grad_vb_ref[2] = {0.0, 0.0};

        v[0] = vall[j * 6 + 0];
        v[1] = vall[j * 6 + 1];
        v[2] = vall[j * 6 + 2];
        v[3] = vall[j * 6 + 3];
        v[4] = vall[j * 6 + 4];
        v[5] = vall[j * 6 + 5];

        // std::cout << "v: " << v[0] << "\t" << v[1] << "\t" << v[2] << "\t" << v[3] << "\t" << v[4] << "\t" << v[5] << std::endl;
         if (corner || edge)
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
          if (inside_out)
          {
             const std::array<double, 6> nodal_values = {v[0], v[1], v[2], v[3], v[4], v[5]};
             const std::array<double, 12> coeffs = proteus::solve_ifem_basis_coefficients(
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
            const std::array<double, 12> coeffs = proteus::solve_ifem_basis_coefficients(
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
    (void)jf;
    for (int i = 0; i < 6; i++)
    {
      _lift_a[i] = 0.0;
      _lift_b[i] = 0.0;
    }
    if (!(corner || edge))
    {
      // The solver imposes  m_second*dpsi_second/dn_s - m_first*dpsi_first/dn_s = jf  with
      // n_s = +-level_set_normal, and level_set_normal is the cut tangent rotated by 90 degrees: its
      // sign follows the order of the cut points, not the direction of phi. (The homogeneous basis
      // never notices.) In both inside_out branches the physical jump from a (phi<0) to b (phi>0) is
      // jf*sign(level_set_normal . grad phi), so that sign is passed as jf for a unit jump.
      const double gphi_ref[2] = {phi[1] - phi[0], phi[2] - phi[0]};
      double n_dot_gphi = 0.0;
      for (int I = 0; I < 2; I++)
      {
        double g = 0.0;
        for (int J = 0; J < 2; J++)
          g += inv_Jac[J * nSpace + I] * gphi_ref[J];
        n_dot_gphi += g * level_set_normal[I];
      }
      const double jf_unit = (n_dot_gphi >= 0.0) ? 1.0 : -1.0;
      double *first = inside_out ? _lift_b : _lift_a, *second = inside_out ? _lift_a : _lift_b;
      const double m_first = inside_out ? mb : ma, m_second = inside_out ? ma : mb;
      if (nP_ifem == 3)
      {
        const std::array<double, 3> zero = {0., 0., 0.};
        const std::array<double, 6> c = proteus::solve_ifem_basis_coefficients(
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
        const std::array<double, 12> c = proteus::solve_ifem_basis_coefficients(
            2, x0, y0, nx, ny, m_first, m_second, jf_unit, Jit00, Jit01, Jit10, Jit11, zero);
        for (int i = 0; i < 6; i++)
        {
          first[i] = c[i];
          second[i] = c[6 + i];
        }
      }
    }
  }

  template <int nSpace, int nP_ifem, int nP, int nQ, int nEBQ, bool useIfemBasis>
  inline void Simplex<nSpace, nP_ifem, nP, nQ, nEBQ, useIfemBasis>::_calculate_lift(const double *xi, double *psi)
  {
    const double *c[2] = {_lift_a, _lift_b};
    for (int s = 0; s < 2; s++)
    {
      const double *a = c[s];
      const double val = a[0] + a[1] * xi[0] + a[2] * xi[1] + a[3] * xi[0] * xi[1] + a[4] * xi[0] * xi[0] + a[5] * xi[1] * xi[1];
      const double g_ref[2] = {a[1] + a[3] * xi[1] + 2.0 * a[4] * xi[0], a[2] + a[3] * xi[0] + 2.0 * a[5] * xi[1]};
      double g[2] = {0.0, 0.0};
      for (int I = 0; I < nSpace && I < 2; I++)
        for (int J = 0; J < nSpace && J < 2; J++)
          g[I] += inv_Jac[J * nSpace + I] * g_ref[J];
      psi[3 * s + 0] = val;
      psi[3 * s + 1] = g[0];
      psi[3 * s + 2] = g[1];
    }
  }

  template <int nSpace, int nP_ifem, int nP, int nQ, int nEBQ, bool useIfemBasis>
  inline void Simplex<nSpace, nP_ifem, nP, nQ, nEBQ, useIfemBasis>::_calculate_basis(const double *xi, double *va, double *vb)
  {
    // Switch on nP_ifem to compute basis functions at given xi
    switch (nP_ifem)
    {
    case 3:
      for (int i = 0; i < nP_ifem; i++)
      {
        va[i] = _a1[i] + _a2[i] * xi[0] + _a3[i] * xi[1];
        vb[i] = _b1[i] + _b2[i] * xi[0] + _b3[i] * xi[1];

        // std::cout << "\nCoefficients for real node " << i << " at quadrature point: " << xi[0] << ", " << xi[1] << std::endl;
        // std::cout << "a: \t [" << _a1[i] << ", " << _a2[i] << ", " << _a3[i] << "]" << std::endl;
        // std::cout << "b: \t [" << _b1[i] << ", " << _b2[i] << ", " << _b3[i] << "]" << std::endl;
                 
        // std::cout << "va[" << i << "] = " << va[i] << "\t vb[" << i << "] = " << vb[i] << std::endl;
      }
      break;
    case 6:
      for (int i = 0; i < nP_ifem; i++)
      {
        va[i] = _a1[i] + _a2[i] * xi[0] + _a3[i] * xi[1] + _a4[i] * xi[0] * xi[1] + _a5[i] * xi[0] * xi[0] + _a6[i] * xi[1] * xi[1];
        vb[i] = _b1[i] + _b2[i] * xi[0] + _b3[i] * xi[1] + _b4[i] * xi[0] * xi[1] + _b5[i] * xi[0] * xi[0] + _b6[i] * xi[1] * xi[1];

        // std::cout << "\nCoefficients for real node " << i << " at quadrature point: " << xi[0] << ", " << xi[1] << std::endl;
        // std::cout << "a: \t [" << _a1[i] << ", " << _a2[i] << ", " << _a3[i] << ", " << _a4[i] << ", " << _a5[i] << ", " << _a6[i] << "]" << std::endl;
        // std::cout << "b: \t [" << _b1[i] << ", " << _b2[i] << ", " << _b3[i] << ", " << _b4[i] << ", " << _b5[i] << ", " << _b6[i] << "]" << std::endl;
                 
        // std::cout << "va[" << i << "] = " << va[i] << "\t vb[" << i << "] = " << vb[i] << std::endl;

      }
      break;
    default:
      throw std::runtime_error("Simplex::_calculate_basis not implemented for order > 2");
    }
  }

  template <int nSpace, int nP_ifem, int nP, int nQ, int nEBQ, bool useIfemBasis>
  inline void Simplex<nSpace, nP_ifem, nP, nQ, nEBQ, useIfemBasis>::_calculate_basis_gradients(const double *xi, double *va_x, double *va_y, double *vb_x, double *vb_y)
  {
    // Switch on nP_ifem to compute basis functions gradients at given xi

    switch (nP_ifem)
    {
      case 3:
      for (int i = 0; i < nP_ifem; i++)
      {
        double grad_va[2] = {0.0, 0.0}, grad_vb[2] = {0.0, 0.0}, grad_va_ref[2] = {0.0, 0.0}, grad_vb_ref[2] = {0.0, 0.0};

        // int i = permutation[j];
        grad_va_ref[0] = _a2[i];
        grad_va_ref[1] = _a3[i];
        grad_vb_ref[0] = _b2[i];
        grad_vb_ref[1] = _b3[i];
        for (int I = 0; I < nSpace; I++)
        {
          for (int J = 0; J < nSpace; J++)
          {
            // // std::cout << "inv_Jac[" << I * nSpace + J << "] = " << inv_Jac[I * nSpace + J] << "\t grad_va_ref[" << J << "] = " << grad_va_ref[J] << std::endl;
            grad_va[I] += inv_Jac[J * nSpace + I] * grad_va_ref[J];
            grad_vb[I] += inv_Jac[J * nSpace + I] * grad_vb_ref[J];
          }
        }
        // if (inside_out){
        //   va_x[i] = grad_vb[0];
        //   va_y[i] = grad_vb[1];
        //   vb_x[i] = grad_va[0];
        //   vb_y[i] = grad_va[1];
        // }
        // else
        // {
          va_x[i] = grad_va[0];
          va_y[i] = grad_va[1];
          vb_x[i] = grad_vb[0];
          vb_y[i] = grad_vb[1];
        // }

        // // std::cout << "Inverse jacobian: [" << inv_Jac[0] << ", " << inv_Jac[1] << ", " << inv_Jac[2] << ", " << inv_Jac[3] << "]\n"; 

        // // std::cout << "quadrature point: (" << xi[0] << ", " << xi[1] << ")\n";
        // // std::cout << i << "\t Ref: \t va_x: " << grad_va_ref[0] << ", va_y: " << grad_va_ref[1] 
        //           << ", vb_x: " << grad_vb_ref[0] << ", vb_y: " << grad_vb_ref[1] << std::endl;
        // // std::cout << j << "\t Real: \t va_x: " << grad_va[0] << ", va_y: " << grad_va[1] 
        //           << ", vb_x: " << grad_vb[0] << ", vb_y: " << grad_vb[1] << std::endl;
      }
      break;
    case 6:
      for (int i = 0; i < nP_ifem; i++)
      {
        double grad_va[2] = {0.0, 0.0}, grad_vb[2] = {0.0, 0.0}, grad_va_ref[2] = {0.0, 0.0}, grad_vb_ref[2] = {0.0, 0.0};

        // int i = permutation[j];
        grad_va_ref[0] = _a2[i] + _a4[i] * xi[1] + 2.0 * _a5[i] * xi[0];
        grad_va_ref[1] = _a3[i] + _a4[i] * xi[0] + 2.0 * _a6[i] * xi[1];
        grad_vb_ref[0] = _b2[i] + _b4[i] * xi[1] + 2.0 * _b5[i] * xi[0];
        grad_vb_ref[1] = _b3[i] + _b4[i] * xi[0] + 2.0 * _b6[i] * xi[1];

        // std::cout << i << "Ref: \t va_x: " << grad_va_ref[0] << ", va_y: " << grad_va_ref[1] << std::endl
        //           << ", vb_x: " << grad_vb_ref[0] << ", vb_y: " << grad_vb_ref[1] << std::endl;

        for (int I = 0; I < nSpace; I++)
        {
          for (int J = 0; J < nSpace; J++)
          {
            // std::cout << "Jac[" << I * nSpace + J << "] = " << Jac[I * nSpace + J] << std::endl;
            // std::cout << "inv_Jac[" << I * nSpace + J << "] = " << inv_Jac[I * nSpace + J] << std::endl;
            grad_va[I] += inv_Jac[J * nSpace + I] * grad_va_ref[J];
            grad_vb[I] += inv_Jac[J * nSpace + I] * grad_vb_ref[J];
          }
        }
        // va_x[i] = grad_va[0];
        // va_y[i] = grad_va[1];
        // vb_x[i] = grad_vb[0];
        // vb_y[i] = grad_vb[1];
        // if (inside_out){
        //   va_x[i] = grad_vb[0];
        //   va_y[i] = grad_vb[1];
        //   vb_x[i] = grad_va[0];
        //   vb_y[i] = grad_va[1];
        // }
        // else
        // {
          va_x[i] = grad_va[0];
          va_y[i] = grad_va[1];
          vb_x[i] = grad_vb[0];
          vb_y[i] = grad_vb[1];
        // }
      }
      break;
    default:
      throw std::runtime_error("Simplex::_calculate_basis_gradients not implemented for order > 2");
    }
  }

  template <int nSpace, int nP_ifem, int nP, int nQ, int nEBQ, bool useIfemBasis>
  inline int Simplex<nSpace, nP_ifem, nP, nQ, nEBQ, useIfemBasis>::calculate(const double *phi_dof, const double *phi_nodes, const double *xi_r, double ma, double mb, double jf, bool isBoundary, bool scale)
  {
    // initialize phi_dof_corrected -- correction can only be actually computed on cut cells
    for (unsigned int i = 0; i < nDOF_phi; i++)
      phi_dof_corrected[i] = phi_dof[i];
    int icase = _calculate_permutation(phi_dof, phi_nodes); // permuation, Jac,inv_Jac...
    if (icase == 1)
    {
      for (unsigned int q = 0; q < nQ; q++)
      {
        _H[q] = 1.0;
        _ImH[q] = 0.0;
        _D[q] = 0.0;
      }
      for (unsigned int ebq = 0; ebq < nEBQ; ebq++)
      {
        _H_ebq[ebq] = 1.0;
        _ImH_ebq[ebq] = 0.0;
        _D_ebq[ebq] = 0.0;
      }
      return icase;
    }
    else if (icase == -1)
    {
      for (unsigned int q = 0; q < nQ; q++)
      {
        _H[q] = 0.0;
        _ImH[q] = 1.0;
        _D[q] = 0.0;
      }
      for (unsigned int ebq = 0; ebq < nEBQ; ebq++)
      {
        _H_ebq[ebq] = 0.0;
        _ImH_ebq[ebq] = 1.0;
        _D_ebq[ebq] = 0.0;
      }
      return icase;
    }
    else if (nSpace == 1 && edge == 1)
    {
      // 1D only: the "edge" here is a single node, so touching zero there
      // is a measure-zero point, not a genuine interface -- treat this the
      // same as the fully-positive icase==1 case. (In 2D/3D "edge" means an
      // entire element edge/face has phi==0, a real nonzero-measure
      // interface, so this special case does not apply there.)
      // inside_out was set true by the pcount==1 branch above (it flips the
      // dir vector for the interior-cut machinery this case never reaches);
      // reset it so set_quad()/set_boundary_quad() don't swap H and ImH.
      inside_out = false;
      for (unsigned int q = 0; q < nQ; q++)
      {
        _H[q] = 1.0;
        _ImH[q] = 0.0;
        _D[q] = 0.0;
      }
      for (unsigned int ebq = 0; ebq < nEBQ; ebq++)
      {
        _H_ebq[ebq] = 1.0;
        _ImH_ebq[ebq] = 0.0;
        _D_ebq[ebq] = 0.0;
      }
      return edge;
    }
    else if (nSpace == 1 && edge == -1)
    {
      // same as above, but the cell is entirely non-positive.
      inside_out = false;
      for (unsigned int q = 0; q < nQ; q++)
      {
        _H[q] = 0.0;
        _ImH[q] = 1.0;
        _D[q] = 0.0;
      }
      for (unsigned int ebq = 0; ebq < nEBQ; ebq++)
      {
        _H_ebq[ebq] = 0.0;
        _ImH_ebq[ebq] = 1.0;
        _D_ebq[ebq] = 0.0;
      }
      return edge;
    }
    if (quad_cut)
    {
      _calculate_cuts_quad(); // THETA_* for quad cut in 3D
      _calculate_normal_quad(phys_nodes_cut_quad_01,
                             phys_nodes_cut_quad_02,
                             phys_nodes_cut_quad_31,
                             phys_nodes_cut_quad_32,
                             level_set_normal); // normal to interface
    }
    else
    {
      _calculate_cuts();                                           // X_0, array of interface cuts on reference simplex
      _calculate_normal<nSpace>(phys_nodes_cut, level_set_normal); // normal to interface
    }
    _correct_phi(phi_dof, phi_nodes);
    if (flip_the_cell)
    {
      _calculate_permutation(phi_dof, phi_nodes);
      _calculate_cuts();                                           // X_0, array of interface cuts on reference simplex
    }
    _calculate_C(); // coefficients of equiv poly

    double ma_scale, mb_scale;
    if (scale)
    {
      // cek hack - 2D, pressure basis for discontinuous density
      double jump_scale = level_set_normal[1],
             m_average = 0.5 * (ma + mb),
             m_jump = 0.5 * (mb - ma);
      mb_scale = m_average + jump_scale * m_jump; // mb when jump_scale=1
      ma_scale = m_average - jump_scale * m_jump; // ma when jump_scale=1
      //    double mb_scale=mb, ma_scale=ma;
      // cek hack end
    }
    else
    {
      ma_scale = ma;
      mb_scale = mb;
    }
    // The two-sided IFEM basis is only implemented for 2D simplices with a P1
    // (nP_ifem 3) or P2 (nP_ifem 6) trial space; _calculate_basis_coefficients
    // asserts nSpace==2 and throws for any other nP_ifem. Every non-ADR caller
    // aliases GeneralizedFunctions_mix<nSpace, nP, nP, ...>, so VOF/MCorr reach
    // here with nP_ifem 2 and RANS3PF with 1: they need only the H/ImH/D moment
    // fit and must not enter the basis solve. Guard on nP_ifem, not just nSpace.
    if (useIfemBasis)
      _calculate_basis_coefficients(ma_scale, mb_scale, jf);
    // compute the default affine map based on phi_nodes[0]
    double Jac_0[nSpace * nSpace];
    for (unsigned int i = 0; i < nN - 1; i++)
      for (unsigned int I = 0; I < nSpace; I++)
        Jac_0[I * nSpace + i] = phi_nodes[(1 + i) * 3 + I] - phi_nodes[I];

    if (!isBoundary)
    {
      for (unsigned int q = 0; q < nQ; q++)
      {
        // Due to the permutation, the quadrature points on the reference may be rotated
        // map reference to physical simplex, then back to permuted reference
        double x[nSpace], xi[nSpace];
        // to physical coordinates
        for (unsigned int I = 0; I < nSpace; I++)
        {
          x[I] = phi_nodes[I];
          for (unsigned int J = 0; J < nSpace; J++)
          {
            x[I] += Jac_0[I * nSpace + J] * xi_r[q * 3 + J];
          }
        }
        // back to reference coordinates on possibly permuted
        for (unsigned int I = 0; I < nSpace; I++)
        {
          xi[I] = 0.0;
          for (unsigned int J = 0; J < nSpace; J++)
          {
            xi[I] += inv_Jac[I * nSpace + J] * (x[J] - nodes[J]);
          }
        }
          if (nSpace == 1)
            _calculate_polynomial_1D<nP>(xi, C_H, C_ImH, C_D, _H[q], _ImH[q], _D[q]);
          else if (nSpace == 2)
          {
            _calculate_polynomial_2D<nP>(xi, C_H, C_ImH, C_D, _H[q], _ImH[q], _D[q]);
            // Only the IFEM instantiations have a two-sided basis; see the guard on
            // _calculate_basis_coefficients above.
            if (useIfemBasis)
            {
              _calculate_basis(xi, &_va[q * nP_ifem], &_vb[q * nP_ifem]);
              _calculate_basis_gradients(xi, &_va_x[q * nP_ifem], &_va_y[q * nP_ifem], &_vb_x[q * nP_ifem], &_vb_y[q * nP_ifem]);
            }
            _calculate_lift(xi, &_psi[q * 6]);
          }
          else if (nSpace == 3)
            _calculate_polynomial_3D<nP>(xi, C_H, C_ImH, C_D, _H[q], _ImH[q], _D[q]);
      }
      set_quad(0);
    }
    else
    {
      for (unsigned int ebq = 0; ebq < nEBQ; ebq++)
      {
        // Due to the permutation, the quadrature points on the reference may be rotated
        // map reference to physical simplex, then back to permuted reference
        double x[nSpace], xi[nSpace];
        // to physical coordinates
        for (unsigned int I = 0; I < nSpace; I++)
        {
          x[I] = phi_nodes[I];
          for (unsigned int J = 0; J < nSpace; J++)
          {
            x[I] += Jac_0[I * nSpace + J] * xi_r[ebq * 3 + J];
          }
        }
        // back to reference coordinates on possibly permuted
        for (unsigned int I = 0; I < nSpace; I++)
        {
          xi[I] = 0.0;
          for (unsigned int J = 0; J < nSpace; J++)
          {
            xi[I] += inv_Jac[I * nSpace + J] * (x[J] - nodes[J]);
          }
        }
        if (nSpace == 1)
          _calculate_polynomial_1D<nP>(xi, C_H, C_ImH, C_D, _H_ebq[ebq], _ImH_ebq[ebq], _D_ebq[ebq]);
        else if (nSpace == 2)
        {
          _calculate_polynomial_2D<nP>(xi, C_H, C_ImH, C_D, _H_ebq[ebq], _ImH_ebq[ebq], _D_ebq[ebq]);
          // Only the IFEM instantiations have a two-sided basis; see the guard on
          // _calculate_basis_coefficients above.
          if (useIfemBasis)
          {
            _calculate_basis(xi, &_va_ebq[ebq * nP_ifem], &_vb_ebq[ebq * nP_ifem]);
            _calculate_basis_gradients(xi, &_va_x_ebq[ebq * nP_ifem], &_va_y_ebq[ebq * nP_ifem], &_vb_x_ebq[ebq * nP_ifem], &_vb_y_ebq[ebq * nP_ifem]);
          }
          _calculate_lift(xi, &_psi_ebq[ebq * 6]);
        }
        else if (nSpace == 3)
          _calculate_polynomial_3D<nP>(xi, C_H, C_ImH, C_D, _H_ebq[ebq], _ImH_ebq[ebq], _D_ebq[ebq]);
      }
      set_boundary_quad(0);
    }
    // IFEM needs the interface normal to follow its own inside_out convention, but
    // pre-IFEM this flip was deliberately disabled (it sat commented out inside the
    // old normal_sign block) and every other consumer -- VOF/MCorr/RDLS/RANS* --
    // still reads get_normal() expecting the unflipped normal. Re-enabling it for
    // them broke MCorr's mass correction. Gate it with the rest of the IFEM path.
    if (useIfemBasis && inside_out)
      for (unsigned int I = 0; I < nSpace; I++)
        level_set_normal[I] *= -1.0;
    return icase;
  }

  template <int nSpace, int nP_ifem, int nP, int nQ, int nEBQ, bool useIfemBasis = false>
  class GeneralizedFunctions_mix
  {
  public:
    Regularized<nSpace, nP_ifem, nP, nQ, nEBQ, useIfemBasis> regularized;
    Simplex<nSpace, nP_ifem, nP, nQ, nEBQ, useIfemBasis> exact;
    bool useExact;
    GeneralizedFunctions_mix(bool useExact = true) : useExact(useExact)
    {
    }

    inline int calculate(const double *phi_dof, const double *phi_nodes, const double *xi_r, double ma, double mb, double jf, bool isBoundary, bool scale)
    {

      if (useExact)
        return exact.calculate(phi_dof, phi_nodes, xi_r, ma, mb, jf, isBoundary, scale);
      else // for inexact just copy over local phi_dof
      {
        for (int i = 0; i < exact.nN; i++)
          exact.phi_dof_corrected[i] = phi_dof[i];
        return 1;
      }
    }

    inline int calculate(const double *phi_dof, const double *phi_nodes, const double *xi_r, double ma, double mb, bool isBoundary, bool scale)
    {
      return calculate(phi_dof, phi_nodes, xi_r, ma, mb, 0.0, isBoundary, false);
    }

    inline int calculate(const double *phi_dof, const double *phi_nodes, const double *xi_r, bool isBoundary)
    {
      return calculate(phi_dof, phi_nodes, xi_r, 1.0, 1.0, 0.0, isBoundary, false);
    }

    inline double *get_normal()
    {
      if (useExact)
        return exact.get_normal();
      else
        return regularized.get_normal();
    }

    inline void set_quad(unsigned int q)
    {
      if (useExact)
        exact.set_quad(q);
    }

    inline void set_boundary_quad(unsigned int ebq)
    {
      if (useExact)
        exact.set_boundary_quad(ebq);
    }

    inline double H(double eps, double phi)
    {
      if (useExact)
        return exact.H(eps, phi);
      else
        return regularized.H(eps, phi);
    }

    inline double ImH(double eps, double phi)
    {
      if (useExact)
        return exact.ImH(eps, phi);
      else
        return regularized.ImH(eps, phi);
    }

    inline double D(double eps, double phi)
    {
      if (useExact)
        return exact.D(eps, phi);
      else
        return regularized.D(eps, phi);
    }
    inline double VA(int i)
    {
      if (useExact)
        return exact.VA(i);
      else
        return regularized.VA(i);
    }
    inline double VA_x(int i)
    {
      if (useExact)
        return exact.VA_x(i);
      else
        return regularized.VA_x(i);
    }
    inline double VA_y(int i)
    {
      if (useExact)
        return exact.VA_y(i);
      else
        return regularized.VA_y(i);
    }
    inline double VA_z(int i)
    {
      if (useExact)
        return exact.VA_z(i);
      else
        return regularized.VA_z(i);
    }
    inline double VB(int i)
    {
      if (useExact)
        return exact.VB(i);
      else
        return regularized.VB(i);
    }
    inline double VB_x(int i)
    {
      if (useExact)
        return exact.VB_x(i);
      else
        return regularized.VB_x(i);
    }
    inline double VB_y(int i)
    {
      if (useExact)
        return exact.VB_y(i);
      else
        return regularized.VB_y(i);
    }
    inline double VB_z(int i)
    {
      if (useExact)
        return exact.VB_z(i);
      else
        return regularized.VB_z(i);
    }
    inline double PSIA() { return useExact ? exact.PSIA() : regularized.PSIA(); }
    inline double PSIA_x() { return useExact ? exact.PSIA_x() : regularized.PSIA_x(); }
    inline double PSIA_y() { return useExact ? exact.PSIA_y() : regularized.PSIA_y(); }
    inline double PSIB() { return useExact ? exact.PSIB() : regularized.PSIB(); }
    inline double PSIB_x() { return useExact ? exact.PSIB_x() : regularized.PSIB_x(); }
    inline double PSIB_y() { return useExact ? exact.PSIB_y() : regularized.PSIB_y(); }
  };
} // equivalent_polynomials

#endif
