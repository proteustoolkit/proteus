#ifndef EQUIVALENT_POYNOMIALS_UTILS
#define EQUIVALENT_POYNOMIALS_UTILS
namespace equivalent_polynomials
{
  template<int nSpace>
  inline void _calculate_normal(double* phys_nodes_cut, double* level_set_normal);

  template<>
  inline void _calculate_normal<1>(double* phys_nodes_cut, double* level_set_normal)
  {
    const unsigned int nSpace(1);
    level_set_normal[0] = 1.0;
  }

  template<>
  inline void _calculate_normal<2>(double* phys_nodes_cut, double* level_set_normal)
  {
    const unsigned int nSpace(2);
    double level_set_tangent[2];
    for(unsigned int I=0; I < nSpace; I++)
      {
        level_set_tangent[I] = phys_nodes_cut[3+I] - phys_nodes_cut[I];//nodes always 3D
      }
    level_set_normal[0] =   level_set_tangent[1];
    level_set_normal[1] = - level_set_tangent[0];
    double norm = std::sqrt(level_set_normal[0]*level_set_normal[0] +
                            level_set_normal[1]*level_set_normal[1]);
    if (norm > 0.0)
      for(unsigned int I=0; I < nSpace; I++)
        level_set_normal[I] /=  norm;
    else
      {
	for(unsigned int I=0; I < nSpace; I++)
	  level_set_normal[I] =0.0;
	level_set_normal[1] = 1.0;
      }
  }

  template<>
  inline void _calculate_normal<3>(double* phys_nodes_cut, double* level_set_normal)
  {
    const unsigned int nSpace(3);
    double level_set_tangent_a[3], level_set_tangent_b[3];
    for(unsigned int I=0; I < nSpace; I++)
      {
        level_set_tangent_a[I] = phys_nodes_cut[3+I] - phys_nodes_cut[I];
        level_set_tangent_b[I] = phys_nodes_cut[6+I] - phys_nodes_cut[I];
      }
    level_set_normal[0] =   level_set_tangent_a[1]*level_set_tangent_b[2]
      - level_set_tangent_a[2]*level_set_tangent_b[1];
    level_set_normal[1] = - level_set_tangent_a[0]*level_set_tangent_b[2]
      + level_set_tangent_a[2]*level_set_tangent_b[0];
    level_set_normal[2] =   level_set_tangent_a[0]*level_set_tangent_b[1]
      - level_set_tangent_a[1]*level_set_tangent_b[0];
    double norm = std::sqrt(level_set_normal[0]*level_set_normal[0] +
                            level_set_normal[1]*level_set_normal[1] +
                            level_set_normal[2]*level_set_normal[2]);
    if (norm > 0.0)
      for(unsigned int I=0; I < nSpace; I++)
        level_set_normal[I] /=  norm;
    else
      {
	for(unsigned int I=0; I < nSpace; I++)
	  level_set_normal[I] =0.0;
	level_set_normal[2] =1.0;
      }
    assert(std::fabs(1.0-level_set_normal[0]*level_set_normal[0] - level_set_normal[1]*level_set_normal[1] - level_set_normal[2]*level_set_normal[2]) < 1.0e-8);
  }

  inline void _calculate_normal_quad(double* phys_nodes_cut_quad_01,
				     double* phys_nodes_cut_quad_02,
				     double* phys_nodes_cut_quad_31,
				     double* phys_nodes_cut_quad_32,
				     double* level_set_normal)
  {
    const unsigned int nSpace(3);
    double level_set_tangent_a[3], level_set_tangent_b[3], level_set_tangent_c[3], level_set_normal_2[3];
    for(unsigned int I=0; I < nSpace; I++)
      {
        level_set_tangent_a[I] = phys_nodes_cut_quad_02[I] - phys_nodes_cut_quad_01[I];
        level_set_tangent_b[I] = phys_nodes_cut_quad_32[I] - phys_nodes_cut_quad_01[I];
        level_set_tangent_c[I] = phys_nodes_cut_quad_31[I] - phys_nodes_cut_quad_01[I];
      }
    level_set_normal[0] =   level_set_tangent_a[1]*level_set_tangent_b[2] - level_set_tangent_a[2]*level_set_tangent_b[1];
    level_set_normal[1] = - level_set_tangent_a[0]*level_set_tangent_b[2] + level_set_tangent_a[2]*level_set_tangent_b[0];
    level_set_normal[2] =   level_set_tangent_a[0]*level_set_tangent_b[1] - level_set_tangent_a[1]*level_set_tangent_b[0];
    double norm = std::sqrt(level_set_normal[0]*level_set_normal[0] +
                            level_set_normal[1]*level_set_normal[1] +
                            level_set_normal[2]*level_set_normal[2]);
    if (norm > 0.0)
      {
	for(unsigned int I=0; I < nSpace; I++)
	  level_set_normal[I] =  level_set_normal[I] / norm;
      }
    else
      {
	for(unsigned int I=0; I < nSpace; I++)
	  level_set_normal[I] =0.0;
	level_set_normal[2] = 1.0;
      }
    assert(std::fabs(1.0-level_set_normal[0]*level_set_normal[0] - level_set_normal[1]*level_set_normal[1] - level_set_normal[2]*level_set_normal[2]) < 1.0e-8);
  }

  template<int nP>
  inline void _calculate_polynomial_1D(double* xi, double* C_H, double* C_ImH, double* C_D, double& _H, double& _ImH, double& _D)
  {
    _H   = 0.0;
    _ImH = 0.0;
    _D   = 0.0;
    unsigned int iDOF    = 0;
    double x_pow_i=1.0;
    for(unsigned int i=0; i < nP+1; i++, iDOF++)
      {
        double psi=x_pow_i;
        _H   += C_H[iDOF]*psi;
        _ImH += C_ImH[iDOF]*psi;
        _D   += C_D[iDOF]*psi;
        x_pow_i *= xi[0];
      }
  }

  // calculate_edge_H / evaluate_edge_poly: the SCIFEM edge-restricted moment-fit Heaviside
  // (ADR.h's ifem_boundaries facet term). Not a duplicate of _calculate_polynomial_1D above,
  // despite the similar Horner loop: that function EVALUATES an already-computed (C_H,C_ImH,C_D)
  // triple for a genuinely 1D *element* (Simplex<1,...>); calculate_edge_H instead COMPUTES a
  // single H-only fit (via _set_Ainv<1,nP>/_calculate_b<1,nP> from
  // equivalent_polynomials_coefficients.h) for one *edge* of a 2D/3D element, with the
  // orientation-safe sign handling a facet needs. Orientation-safe: returns the fit for the
  // indicator of {phi > 0} whichever way round the edge is parametrized. phi0/phi1 must have
  // opposite signs (the edge must actually be cut).
  template <int nP>
  inline void calculate_edge_H(double phi0, double phi1, double C_H[nP + 1])
  {
    double theta = -phi0 / (phi1 - phi0);
    double Ainv[(nP + 1) * (nP + 1)];
    _set_Ainv<1, nP>(Ainv);
    double b_H[nP + 1], b_ImH[nP + 1], b_dH[nP + 1];
    _calculate_b<1, nP>(&theta, b_H, b_ImH, b_dH);
    // _calculate_b assumes H=1 for t>theta (phi increasing along t); use the complementary
    // moments when phi decreases instead, so C_H always fits the indicator of {phi>0}.
    const double *b = (phi1 > phi0) ? b_H : b_ImH;
    for (int i = 0; i <= nP; i++)
    {
      C_H[i] = 0.0;
      for (int j = 0; j <= nP; j++)
        C_H[i] += Ainv[i * (nP + 1) + j] * b[j];
    }
  }

  // Indicator of {t in [0,1] : f(t) > 0 and g(t) > 0} for affine f, g given by their values at the
  // edge ends -- e.g. one Gamma_f branch AND the fluid side of Gamma_s on a triply-cut edge. The set
  // is an interval [lo,hi], so its fit is chi(t>lo) - chi(t>hi), each a calculate_edge_H fit of
  // phi = t - c: exact for the same integrand degrees (<= nP). All zeros if the interval is empty.
  template <int nP>
  inline void calculate_edge_interval_H(double f0, double f1, double g0, double g1, double C[nP + 1])
  {
    double lo = 0.0, hi = 1.0;
    bool empty = false;
    for (int k = 0; k < 2; k++)
    {
      const double p0 = (k == 0) ? f0 : g0, p1 = (k == 0) ? f1 : g1;
      if (p0 > 0.0 && p1 > 0.0)
        continue;
      if (p0 <= 0.0 && p1 <= 0.0)
      {
        empty = true;
        continue;
      }
      const double theta = p0 / (p0 - p1);
      if (p0 > 0.0 && theta < hi)
        hi = theta;
      else if (p0 <= 0.0 && theta > lo)
        lo = theta;
    }
    for (int i = 0; i <= nP; i++)
      C[i] = 0.0;
    if (empty || hi <= lo)
      return;
    double C_lo[nP + 1], C_hi[nP + 1];
    if (lo > 0.0)
      calculate_edge_H<nP>(-lo, 1.0 - lo, C_lo);
    else
      for (int i = 0; i <= nP; i++)
        C_lo[i] = (i == 0) ? 1.0 : 0.0;
    if (hi < 1.0)
      calculate_edge_H<nP>(-hi, 1.0 - hi, C_hi);
    else
      for (int i = 0; i <= nP; i++)
        C_hi[i] = 0.0;
    for (int i = 0; i <= nP; i++)
      C[i] = C_lo[i] - C_hi[i];
  }

  template <int nP>
  inline double evaluate_edge_poly(const double C[nP + 1], double t)
  {
    double val = 0.0, tpow = 1.0;
    for (int i = 0; i <= nP; i++)
    {
      val += C[i] * tpow;
      tpow *= t;
    }
    return val;
  }

  template<int nP>
  inline void _calculate_polynomial_2D(double* xi, double* C_H, double* C_ImH, double* C_D, double& _H, double& _ImH, double& _D)
  {
    _H   = 0.0;
    _ImH = 0.0;
    _D   = 0.0;
    unsigned int iDOF    = 0;
    double x_pow_i=1.0;
    for(unsigned int i=0; i < nP+1; i++)
      {
        double y_pow_j=1.0;
        for(unsigned int j=0; j < nP+1-i; j++, iDOF++)
          {
            double psi=x_pow_i*y_pow_j;
            _H   += C_H[iDOF]*psi;
            _ImH += C_ImH[iDOF]*psi;
            _D   += C_D[iDOF]*psi;
            y_pow_j *= xi[1];
          }
        x_pow_i *= xi[0];
      }
  }

  template<int nP>
  inline void _calculate_polynomial_3D(double* xi, double* C_H, double* C_ImH, double* C_D, double& _H, double& _ImH, double& _D)
  {
    _H   = 0.0;
    _ImH = 0.0;
    _D   = 0.0;
    unsigned int iDOF    = 0;
    double x_pow_i=1.0;
    for(unsigned int i=0; i < nP+1; i++)
      {
        double y_pow_j=1.0;
        for(unsigned int j=0; j < nP+1-i; j++)
          {
            double x_pow_i_y_pow_j=x_pow_i*y_pow_j, z_pow_k=1.0;
            for(unsigned int k=0; k < nP+1-i-j; k++, iDOF++)
              {
                double psi=x_pow_i_y_pow_j*z_pow_k;
                _H   += C_H[iDOF]*psi;
                _ImH += C_ImH[iDOF]*psi;
                _D   += C_D[iDOF]*psi;
                z_pow_k *= xi[2];
              }
            y_pow_j *= xi[1];
          }
        x_pow_i *= xi[0];
      }
  }
  
  const int XX(0),XY(1),XZ(2),YX(3),YY(4),YZ(5),ZX(6),ZY(7),ZZ(8);

  template<int nSpace>
  inline double det(const double* A);

  template<>
  inline double det<1>(const double* A)
  {
    return A[0];
  }

  template<>
  inline double det<2>(const double* A)
  {
    return A[0]*A[3] - A[1]*A[2];
  }

  template<>
  inline double det<3>(const double* A)
  {
    return A[XX]*(A[YY]*A[ZZ] - A[YZ]*A[ZY]) -
      A[XY]*(A[YX]*A[ZZ] - A[YZ]*A[ZX]) +
      A[XZ]*(A[YX]*A[ZY] - A[YY]*A[ZX]);
  }

  template<int nSpace>
  inline void inv(const double* A, double* Ainv);

  template<>
  inline void inv<1>(const double* A, double* Ainv)
  {
    Ainv[0] = 1.0/A[0];
  }

  template<>
  inline void inv<2>(const double* A, double* Ainv)
  {
    double oneOverDet = 1.0/det<2>(A);
    Ainv[0] = oneOverDet*A[3];
    Ainv[1] = -oneOverDet*A[1];
    Ainv[2] = -oneOverDet*A[2];
    Ainv[3] = oneOverDet*A[0];
    assert(fabs(Ainv[0]*A[0]+Ainv[1]*A[2]-1.0) < 1.0e-10);
    assert(fabs(Ainv[0]*A[1]+Ainv[1]*A[3]) < 1.0e-10);
    assert(fabs(Ainv[2]*A[0]+Ainv[3]*A[2]) < 1.0e-10);
    assert(fabs(Ainv[2]*A[1]+Ainv[3]*A[3]-1.0) < 1.0e-10);
  }

  template<>
  inline void inv<3>(const double* A, double* Ainv)
  {
    double oneOverDet = 1.0/det<3>(A);
    Ainv[XX] = oneOverDet*(A[YY]*A[ZZ] - A[YZ]*A[ZY]);
    Ainv[YX] = oneOverDet*(A[YZ]*A[ZX] - A[YX]*A[ZZ]);
    Ainv[ZX] = oneOverDet*(A[YX]*A[ZY] - A[YY]*A[ZX]);
    Ainv[XY] = oneOverDet*(A[ZY]*A[XZ] - A[ZZ]*A[XY]);
    Ainv[YY] = oneOverDet*(A[ZZ]*A[XX] - A[ZX]*A[XZ]);
    Ainv[ZY] = oneOverDet*(A[ZX]*A[XY] - A[ZY]*A[XX]);
    Ainv[XZ] = oneOverDet*(A[XY]*A[YZ] - A[XZ]*A[YY]);
    Ainv[YZ] = oneOverDet*(A[XZ]*A[YX] - A[XX]*A[YZ]);
    Ainv[ZZ] = oneOverDet*(A[XX]*A[YY] - A[XY]*A[YX]);
  }
}//equivalent_polynomials
#endif
