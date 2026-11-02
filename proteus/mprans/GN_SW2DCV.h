#ifndef GN_SW2DCV_H
#define GN_SW2DCV_H
#include "ArgumentsDict.h"
#include "CompKernel.h"
#include "ModelFactory.h"
#include "xtensor-python/pyarray.hpp"
#include <assert.h>
#include <cmath>
#include <iostream>
#include <valarray>

namespace py = pybind11;

#define VEL_FIX_POWER 2.
#define LAMBDA_MGN 1.
#define IF_BOTH_GAMMA_BRANCHES 0
#define LIMITING_ITERATION 2
#define IF_DEBUGGING 0
#define IF_LIMITING_DEBUGGING 0

/* inline functions */
namespace proteus {

// for Serre-Green-Naghdi "max" wave speeds
inline double GN_nu1(const double &g, const double &hL, const double &uL,
                     const double &etaL, const double &inv_meshSizeL) {

  double augL = LAMBDA_MGN / 3. * inv_meshSizeL * (6. * hL + 12. * (hL - etaL));
  const double meshSizeL = 1. / inv_meshSizeL;

  if (etaL >= hL) {
    augL = LAMBDA_MGN / 3. * inv_meshSizeL * (6. * hL);
  }
  augL = augL * std::pow(meshSizeL / fmax(meshSizeL, hL), 2);

  return uL - sqrt(g * hL) * sqrt(1. + augL);
}
inline double GN_nu3(const double &g, const double &hR, const double &uR,
                     const double &etaR, const double &inv_meshSizeR) {
  double augR = LAMBDA_MGN / 3. * inv_meshSizeR * (6. * hR + 12. * (hR - etaR));
  const double meshSizeR = 1. / inv_meshSizeR;

  if (etaR >= hR) {
    augR = LAMBDA_MGN / 3. * inv_meshSizeR * (6. * hR);
  }
  augR = augR * std::pow(meshSizeR / fmax(meshSizeR, hR), 2);
  return uR + sqrt(g * hR) * sqrt(1 + augR);
}
// FOR CELL BASED ENTROPY VISCOSITY
inline double ENTROPY(const double &g, const double &h, const double &hu,
                      const double &hv, const double &z,
                      const double &one_over_hReg) {
  return 0.5 *
         (g * h * h + one_over_hReg * (hu * hu + hv * hv) + 2. * g * h * z);
}
inline double DENTROPY_DH(const double &g, const double &h, const double &hu,
                          const double &hv, const double &z,
                          const double &one_over_hReg) {
  return g * h - 0.5 * (hu * hu + hv * hv) * std::pow(one_over_hReg, 2) + g * z;
}
inline double DENTROPY_DHU(const double &g, const double &h, const double &hu,
                           const double &hv, const double &z,
                           const double &one_over_hReg) {
  return hu * one_over_hReg;
}
inline double DENTROPY_DHV(const double &g, const double &h, const double &hu,
                           const double &hv, const double &z,
                           const double &one_over_hReg) {
  return hv * one_over_hReg;
}
inline double ENTROPY_FLUX1(const double &g, const double &h, const double &hu,
                            const double &hv, const double &z,
                            const double &one_over_hReg) {
  return (ENTROPY(g, h, hu, hv, z, one_over_hReg) + 0.5 * g * h * h +
          g * h * z) *
         hu * one_over_hReg;
}
inline double ENTROPY_FLUX2(const double &g, const double &h, const double &hu,
                            const double &hv, const double &z,
                            const double &one_over_hReg) {
  return (ENTROPY(g, h, hu, hv, z, one_over_hReg) + 0.5 * g * h * h +
          g * h * z) *
         hv * one_over_hReg;
}
inline double relaxation(const double &xi, const double &alpha) {
  double function = 0.;
  if (xi < 1.) {
    const double den = 1.0 - alpha;
    const double num = std::exp(-std::abs(std::log(alpha)) * xi * xi) - alpha;
    function = num / den;
  }
  return function;
}
} // namespace proteus

namespace proteus {

class GN_SW2DCV_base {
public:
  virtual ~GN_SW2DCV_base() {}
  virtual void convexLimiting(arguments_dict &args) = 0;
  virtual double calculateEdgeBasedCFL(arguments_dict &args) = 0;
  virtual void calculatePreStep(arguments_dict &args) = 0;
  virtual void calculateEV(arguments_dict &args) = 0;
  virtual void calculateBoundsAndHighOrderRHS(arguments_dict &args) = 0;
  virtual void calculateResidual(arguments_dict &args) = 0;
  virtual void calculateMassMatrix(arguments_dict &args) = 0;
  virtual void calculateLumpedMassMatrix(arguments_dict &args) = 0;
};

template <class CompKernelType, int nSpace, int nQuadraturePoints_element,
          int nDOF_mesh_trial_element, int nDOF_trial_element,
          int nDOF_test_element, int nQuadraturePoints_elementBoundary>
class GN_SW2DCV : public GN_SW2DCV_base {
public:
  const int nDOF_test_X_trial_element;
  CompKernelType ck;
  GN_SW2DCV()
      : nDOF_test_X_trial_element(nDOF_test_element * nDOF_trial_element),
        ck() {
    std::cout << "Constructing GN_SW2DCV<CompKernelTemplate<" << nSpace << ","
              << nQuadraturePoints_element << "," << nDOF_mesh_trial_element
              << "," << nDOF_trial_element << "," << nDOF_test_element << ","
              << nQuadraturePoints_elementBoundary << ">());" << std::endl
              << std::flush;
  }

  inline double maxWaveSpeedSharpInitialGuess(
      const double g, const double nx, const double ny, const double hL,
      const double huL, double hvL, const double hetaL, const double inv_MeshL,
      const double hR, const double huR, const double hvR, const double hetaR,
      const double inv_MeshR, const double hEps) {
    const double one_over_hL =
        2.0 * hL / (hL * hL + std::pow(fmax(hL, hEps), 2.0));
    const double one_over_hR =
        2.0 * hR / (hR * hR + std::pow(fmax(hR, hEps), 2.0));
    const double hVelL = nx * huL + ny * hvL;
    const double hVelR = nx * huR + ny * hvR;
    const double velL = one_over_hL * hVelL;
    const double velR = one_over_hR * hVelR;
    const double etaL = one_over_hL * hetaL;
    const double etaR = one_over_hR * hetaR;

    /* See equation 4.12 JCP 2019 paper:
      1-eigenvalue: uL-sqrt(g*hL)*sqrt(1 + augL)
      3-eigenvalue: uR+sqrt(g*hR)*sqrt(1 + augR)
    */
    const double lambda1 = GN_nu1(g, hL, velL, etaL, inv_MeshL);
    const double lambda3 = GN_nu3(g, hR, velR, etaR, inv_MeshR);

    return fmax(fabs(lambda1), fabs(lambda3));
  }

  inline void calculateCFL(const double &elementDiameter, const double &g,
                           const double &h, const double &hu, const double &hv,
                           const double hEps, double &cfl) {
    double cflx, cfly, c = sqrt(fmax(g * hEps, g * h));
    const double u = 2 * h / (h * h + std::pow(fmax(h, hEps), 2)) * hu;
    const double v = 2 * h / (h * h + std::pow(fmax(h, hEps), 2)) * hv;

    if (u > 0.0)
      cflx = (u + c) / elementDiameter;
    else
      cflx = fabs(u - c) / elementDiameter;

    if (v > 0.0)
      cfly = (v + c) / elementDiameter;
    else
      cfly = fabs(v - c) / elementDiameter;
    cfl = sqrt(cflx * cflx + cfly * cfly); // hack, conservative estimate
  }

  void convexLimiting(arguments_dict &args) {
    const int numDOFs = args.scalar<int>("numDOFs");
    const xt::pyarray<int> &csrRowIndeces_DofLoops =
        args.array<int>("csrRowIndeces_DofLoops");
    const xt::pyarray<int> &csrColumnOffsets_DofLoops =
        args.array<int>("csrColumnOffsets_DofLoops");
    const xt::pyarray<double> &MassMatrix = args.array<double>("MassMatrix");
    const xt::pyarray<double> &lumped_mass_matrix =
        args.array<double>("lumped_mass_matrix");
    const double dt = args.scalar<double>("dt");
    const xt::pyarray<double> &h_old = args.array<double>("h_old");
    const xt::pyarray<double> &hu_old = args.array<double>("hu_old");
    const xt::pyarray<double> &hv_old = args.array<double>("hv_old");
    const xt::pyarray<double> &heta_old = args.array<double>("heta_old");
    const xt::pyarray<double> &hw_old = args.array<double>("hw_old");
    const xt::pyarray<double> &hbeta_old = args.array<double>("hbeta_old");
    const xt::pyarray<double> &b_dof = args.array<double>("b_dof");
    xt::pyarray<double> &limited_hnp1 = args.array<double>("limited_hnp1");
    xt::pyarray<double> &limited_hunp1 = args.array<double>("limited_hunp1");
    xt::pyarray<double> &limited_hvnp1 = args.array<double>("limited_hvnp1");
    xt::pyarray<double> &limited_hetanp1 =
        args.array<double>("limited_hetanp1");
    xt::pyarray<double> &limited_hwnp1 = args.array<double>("limited_hwnp1");
    xt::pyarray<double> &limited_hbetanp1 =
        args.array<double>("limited_hbetanp1");
    const double hEps = args.scalar<double>("hEps");
    xt::pyarray<double> &hLow = args.array<double>("hLow");
    xt::pyarray<double> &huLow = args.array<double>("huLow");
    xt::pyarray<double> &hvLow = args.array<double>("hvLow");
    xt::pyarray<double> &hetaLow = args.array<double>("hetaLow");
    xt::pyarray<double> &hwLow = args.array<double>("hwLow");
    xt::pyarray<double> &hbetaLow = args.array<double>("hbetaLow");
    const xt::pyarray<double> &h_min = args.array<double>("h_min");
    const xt::pyarray<double> &h_max = args.array<double>("h_max");
    const xt::pyarray<double> &heta_min = args.array<double>("heta_min");
    const xt::pyarray<double> &heta_max = args.array<double>("heta_max");
    const xt::pyarray<double> &kin_max = args.array<double>("kin_max");
    const double KE_tiny = args.scalar<double>("KE_tiny");
    const xt::pyarray<double> &SourceTerm_h =
        args.array<double>("SourceTerm_h");
    const xt::pyarray<double> &SourceTerm_hu =
        args.array<double>("SourceTerm_hu");
    const xt::pyarray<double> &SourceTerm_hv =
        args.array<double>("SourceTerm_hv");
    const xt::pyarray<double> &SourceTerm_heta =
        args.array<double>("SourceTerm_heta");
    const xt::pyarray<double> &SourceTerm_hw =
        args.array<double>("SourceTerm_hw");
    const xt::pyarray<double> &SourceTerm_hbeta =
        args.array<double>("SourceTerm_hbeta");
    const xt::pyarray<double> &global_entropy_residual =
        args.array<double>("global_entropy_residual");
    const xt::pyarray<double> &Cx = args.array<double>("Cx");
    const xt::pyarray<double> &Cy = args.array<double>("Cy");
    const xt::pyarray<double> &CTx = args.array<double>("CTx");
    const xt::pyarray<double> &CTy = args.array<double>("CTy");
    const xt::pyarray<double> &RHS_high_h = args.array<double>("RHS_high_h");
    const xt::pyarray<double> &RHS_high_hu = args.array<double>("RHS_high_hu");
    const xt::pyarray<double> &RHS_high_hv = args.array<double>("RHS_high_hv");
    const xt::pyarray<double> &RHS_high_heta =
        args.array<double>("RHS_high_heta");
    const xt::pyarray<double> &RHS_high_hw = args.array<double>("RHS_high_hw");
    const xt::pyarray<double> &RHS_high_hbeta =
        args.array<double>("RHS_high_hbeta");
    const xt::pyarray<double> &extendedSourceTerm_hu =
        args.array<double>("extendedSourceTerm_hu");
    const xt::pyarray<double> &extendedSourceTerm_hv =
        args.array<double>("extendedSourceTerm_hv");
    const xt::pyarray<double> &thetaj_inv = args.array<double>("thetaj_inv");
    const xt::pyarray<double> &wall_dof = args.array<double>("wall_dof");
    const xt::pyarray<double> &porosity_dof = args.array<double>("porosity_dof");
    const double g = args.scalar<double>("g");
    const xt::pyarray<double> &inverse_mesh =
        args.array<double>("inverse_mesh");

    // Create FCT component matrices in vector form
    std::valarray<double> FCT_h(0., Cx.size()), FCT_hu(0., Cx.size()),
        FCT_hv(0., Cx.size()), FCT_heta(0., Cx.size()), FCT_hw(0., Cx.size()),
        FCT_hbeta(0., Cx.size());

    // One limiting pass per call. Between passes the caller exchanges the
    // low-order states with the ghost DOFs: pass 2 reads hLow[j] etc. at
    // neighbours, and a ghost row's own pass-1 update uses an incomplete
    // stencil. The FCT matrices persist in FCT_all (6 x nnz) across calls.
    const int limit_iter_arg = args.scalar<int>("limit_iter");
    const int n_limit_iter = args.scalar<int>("n_limit_iter");
    xt::pyarray<double> &FCT_all = args.array<double>("FCT_all");
    const std::size_t nnz_fct = Cx.size();

    ////////////////////////////////////////////////////
    // Loop to define FCT matrices for each component //
    ////////////////////////////////////////////////////
    int ij = 0;
    for (int i = 0; i < numDOFs && limit_iter_arg == 0; i++) {
      // Read un at ith node
      const double hi = h_old[i];
      const double one_over_hi =
          2. * hi / (hi * hi + std::pow(fmax(hi, hEps), 2)); // hEps
      const double hui = hu_old[i];
      const double ui = hui * one_over_hi;
      const double hvi = hv_old[i];
      const double vi = hvi * one_over_hi;
      const double hetai = heta_old[i];
      const double hwi = hw_old[i];
      const double hbetai = hbeta_old[i];
      const double Zi = b_dof[i];
      const double mi = lumped_mass_matrix[i];
      const double inv_meshSizei = inverse_mesh[i];

      // LOOP OVER THE SPARSITY PATTERN (j-LOOP)//
      for (int offset = csrRowIndeces_DofLoops[i];
           offset < csrRowIndeces_DofLoops[i + 1]; offset++) {
        int j = csrColumnOffsets_DofLoops[offset];

        if (j == i) {
          FCT_h[ij] = 0.;
          FCT_hu[ij] = 0.;
          FCT_hv[ij] = 0.;
          FCT_heta[ij] = 0.;
          FCT_hw[ij] = 0.;
          FCT_hbeta[ij] = 0.;
        } else {
          // Read un stuff at jth node
          const double hj = h_old[j];
          const double one_over_hj =
              2. * hj / (hj * hj + std::pow(fmax(hj, hEps), 2));
          const double huj = hu_old[j];
          const double uj = huj * one_over_hj;
          const double hvj = hv_old[j];
          const double vj = hvj * one_over_hj;
          const double hetaj = heta_old[j];
          const double hwj = hw_old[j];
          const double hbetaj = hbeta_old[j];
          const double Zj = b_dof[j];
          const double mj = lumped_mass_matrix[j];
          const double inv_meshSizej = inverse_mesh[j];

          // Compute star states
          const double hStarij = fmax(0., hi + Zi - fmax(Zi, Zj));
          const double hStarji = fmax(0., hj + Zj - fmax(Zi, Zj));
          //
          const double hStar_ratio_i = hStarij * one_over_hi;
          const double hStar_ratio_j = hStarji * one_over_hj;
          //
          const double huStarij = hui * hStar_ratio_i;
          const double hvStarij = hvi * hStar_ratio_i;
          const double hetaStarij = hetai * std::pow(hStar_ratio_i, 2);
          const double hwStarij = hwi * hStar_ratio_i;
          const double hbetaStarij = hbetai * hStar_ratio_i;
          //
          const double huStarji = huj * hStar_ratio_j;
          const double hvStarji = hvj * hStar_ratio_j;
          const double hetaStarji = hetaj * std::pow(hStar_ratio_j, 2);
          const double hwStarji = hwj * hStar_ratio_j;
          const double hbetaStarji = hbetaj * hStar_ratio_j;

          const double b_ij = 0. - MassMatrix[ij] / mj;
          const double b_ji = 0. - MassMatrix[ij] / mi;

          /* Redefine high-order viscosity. Note that this is not so expensive
          since we are just accessing and multipying. */
          const double cij_norm = sqrt(Cx[ij] * Cx[ij] + Cy[ij] * Cy[ij]);
          const double cji_norm = sqrt(CTx[ij] * CTx[ij] + CTy[ij] * CTy[ij]);
          const double nxij = Cx[ij] / cij_norm;
          const double nyij = Cy[ij] / cij_norm;
          const double nxji = CTx[ij] / cji_norm;
          const double nyji = CTy[ij] / cji_norm;

          double cxa, cya, cxaT, cyaT;
          wallCoefficients(i, j, ij, wall_dof, Cx, Cy, CTx, CTy, cxa, cya,
                           cxaT, cyaT);
          const double phibar_ij = phiBar(i, j, porosity_dof);
          const double muijL = phibar_ij * fmax(std::abs(ui * cxa + vi * cya),
                                                std::abs(uj * cxaT + vj * cyaT));
          double dijL = phibar_ij * fmax(
              maxWaveSpeedSharpInitialGuess(g, nxij, nyij, hi, hui, hvi, hetai,
                                            inv_meshSizei, hj, huj, hvj, hetaj,
                                            inv_meshSizej, hEps) *
                  cij_norm,
              maxWaveSpeedSharpInitialGuess(g, nxji, nyji, hj, huj, hvj, hetaj,
                                            inv_meshSizej, hi, hui, hvi, hetai,
                                            inv_meshSizei, hEps) *
                  cji_norm);

          // Take max with muij
          dijL = std::max(dijL, muijL);

          // Define high-order graph viscosity coefficients
          const double dEVij =
              phibar_ij *
              std::max(global_entropy_residual[i], global_entropy_residual[j]);

          const double dijH = std::min(dijL, dEVij);
          const double muijH = std::min(muijL, dEVij);

          const double diff_dij_muij = (dijH - dijL) - (muijH - muijL);
          const double diff_muij = (muijH - muijL);

          // h
          double viscous_terms =
              diff_dij_muij * (hStarji - hStarij) + diff_muij * (hj - hi);
          FCT_h[ij] = dt * (b_ij * RHS_high_h[j] - b_ji * RHS_high_h[i] +
                            viscous_terms);
          // hu
          viscous_terms =
              diff_dij_muij * (huStarji - huStarij) + diff_muij * (huj - hui);
          FCT_hu[ij] = dt * (b_ij * RHS_high_hu[j] - b_ji * RHS_high_hu[i] +
                             viscous_terms);
          // hv
          viscous_terms =
              diff_dij_muij * (hvStarji - hvStarij) + diff_muij * (hvj - hvi);
          FCT_hv[ij] = dt * (b_ij * RHS_high_hv[j] - b_ji * RHS_high_hv[i] +
                             viscous_terms);
          // heta
          viscous_terms = diff_dij_muij * (hetaStarji - hetaStarij) +
                          diff_muij * (hetaj - hetai);
          FCT_heta[ij] = dt * (b_ij * RHS_high_heta[j] -
                               b_ji * RHS_high_heta[i] + viscous_terms);
          // hw
          viscous_terms =
              diff_dij_muij * (hwStarji - hwStarij) + diff_muij * (hwj - hwi);
          FCT_hw[ij] = dt * (b_ij * RHS_high_hw[j] - b_ji * RHS_high_hw[i] +
                             viscous_terms);
          // hbeta
          viscous_terms = diff_dij_muij * (hbetaStarji - hbetaStarij) +
                          diff_muij * (hbetaj - hbetai);
          FCT_hbeta[ij] = dt * (b_ij * RHS_high_hbeta[j] -
                                b_ji * RHS_high_hbeta[i] + viscous_terms);
        }

        // UPDATE ij //
        ij += 1;
      } // j loop ends here
    }   // i loop ends here

    ////////////////////////////////////////////////////////////////////
    // Main loop to define limiters and compute limited solution //////
    ////////////////////////////////////////////////////////////////////

    if (limit_iter_arg > 0) {
      for (std::size_t k = 0; k < nnz_fct; k++) {
        FCT_h[k] = FCT_all[0 * nnz_fct + k];
        FCT_hu[k] = FCT_all[1 * nnz_fct + k];
        FCT_hv[k] = FCT_all[2 * nnz_fct + k];
        FCT_heta[k] = FCT_all[3 * nnz_fct + k];
        FCT_hw[k] = FCT_all[4 * nnz_fct + k];
        FCT_hbeta[k] = FCT_all[5 * nnz_fct + k];
      }
    }

    // Create Lij_array and initialize with 1
    std::valarray<double> Lij_array(1., Cx.size());

    // define tolerance for limiting
    const double eps = 1e-14;

    /* Loop over # of limiting iterations */
    for (int limit_iter = limit_iter_arg; limit_iter < limit_iter_arg + 1;
         limit_iter++) {

      /* Loop to compute limiter l^j_i and l^i_j */
      ij = 0;
      for (int i = 0; i < numDOFs; i++) {

        // Get low order solution at ith node
        const double hLowi = hLow[i];
        const double huLowi = huLow[i];
        const double hvLowi = hvLow[i];
        const double hetaLowi = hetaLow[i];
        const double kinMaxi = kin_max[i];
        const double mi = lumped_mass_matrix[i];

        // LOOP OVER THE SPARSITY PATTERN (j-LOOP)
        for (int offset = csrRowIndeces_DofLoops[i];
             offset < csrRowIndeces_DofLoops[i + 1]; offset++) {
          int j = csrColumnOffsets_DofLoops[offset];

          Lij_array[ij] = 1.;

          // Get low order solution at jth node
          const double hLowj = hLow[j];
          const double huLowj = huLow[j];
          const double hvLowj = hvLow[j];
          const double hetaLowj = hetaLow[j];
          const double kinMaxj = kin_max[j];
          const double mj = lumped_mass_matrix[j];

          // Compute Pij matrix and Pji
          double denom = 1. / (mi * thetaj_inv[i]);
          const double P_h = FCT_h[ij] * denom;
          const double P_hu = FCT_hu[ij] * denom;
          const double P_hv = FCT_hv[ij] * denom;
          const double P_heta = FCT_heta[ij] * denom;

          denom = 1. / (mj * thetaj_inv[j]);
          const double P_h_tr = -FCT_h[ij] * denom;
          const double P_hu_tr = -FCT_hu[ij] * denom;
          const double P_hv_tr = -FCT_hv[ij] * denom;
          const double P_heta_tr = -FCT_heta[ij] * denom;

          /* h limiting -- to defne l_ji_h */
          double l_ji_h = Lij_array[ij];
          {
            const double denominator = 1. / (std::abs(P_h) + eps * h_max[i]);

            // Define limiter here
            if (hLowi + P_h < h_min[i]) {
              l_ji_h = std::min((std::abs(h_min[i] - hLowi) + eps * h_min[i]) *
                                    denominator,
                                1.);
            } else if (h_max[i] < hLowi + P_h) {
              l_ji_h = std::min((std::abs(h_max[i] - hLowi) + eps * h_min[i]) *
                                    denominator,
                                1.);
            }

            // Set limiter to 0 if water depth is close to 0
            l_ji_h = (hLowi <= hEps) ? 0. : l_ji_h;

            /* Box limiter to be safe? */
            l_ji_h = std::min(l_ji_h, 1.);
            l_ji_h = std::max(l_ji_h, 0.);

#if IF_LIMITING_DEBUGGING
            if ((hLowi + l_ji_h * P_h) - h_min[i] < -1e-12) {
              std::cout << " MAJOR BUG 1a " << std::setprecision(15) << " \n "
                        << " Diff   =  " << (hLowi + l_ji_h * P_h) - h_min[i]
                        << " \n "
                        << " hLowi  = " << hLowi << " \n "
                        << " h_min  = " << h_min[i] << " \n "
                        << " h_max  = " << h_max[i] << " \n "
                        << " l_ji_h = " << l_ji_h << std::endl;
              std::cout << "LIMIT_ITER " << limit_iter << std::endl;
              abort();
            }

            if (h_max[i] - (hLowi + l_ji_h * P_h) < -1e-12) {
              std::cout << " MAJOR BUG 1b " << std::setprecision(15) << " \n "
                        << " Diff   = " << h_max[i] - (hLowi + l_ji_h * P_h)
                        << " \n "
                        << " Soln   = " << (hLowi + P_h) << " \n "
                        << " hLowi  = " << hLowi << " \n "
                        << " h_min  = " << h_min[i] << " \n "
                        << " h_max  = " << h_max[i] << " \n "
                        << " P_h    = " << P_h << " \n "
                        << " l_ji_h = " << l_ji_h << std::endl;
              std::cout << "LIMIT_ITER " << limit_iter << std::endl;
              abort();
            }
#endif
          }

          /* h limiting -- to define l_ij_h */
          double l_ij_h = Lij_array[ij];
          {
            const double denominator_test =
                1. / (std::abs(P_h_tr) + eps * h_max[j]);

            if (hLowj + P_h_tr < h_min[j]) {
              l_ij_h = std::min((std::abs(h_min[j] - hLowj) + eps * h_min[j]) *
                                    denominator_test,
                                1.);
            } else if (h_max[j] < hLowj + P_h_tr) {
              l_ij_h = std::min((std::abs(h_max[j] - hLowj) + eps * h_min[j]) *
                                    denominator_test,
                                1.);
            }

            // set limiter to 0 if water depth is close to 0
            l_ij_h = (hLowj <= hEps) ? 0. : l_ij_h;

            /* Box limiter to be safe? */
            l_ij_h = std::min(l_ij_h, 1.);
            l_ij_h = std::max(l_ij_h, 0.);

#if IF_LIMITING_DEBUGGING
            if ((hLowj + l_ij_h * P_h_tr) - h_min[j] < -1e-12) {
              std::cout << " MAJOR BUG 2a " << std::setprecision(15) << " \n "
                        << " Diff   =  " << h_min[j] - (hLowj + l_ij_h * P_h_tr)
                        << " \n "
                        << " Soln   = " << (hLowj + P_h_tr) << " \n "
                        << " hLowj  = " << hLowj << " \n "
                        << " h_min  = " << h_min[j] << " \n "
                        << " h_max  = " << h_max[j] << " \n "
                        << " P_h_tr = " << P_h_tr << " \n "
                        << " l_ij_h = " << l_ij_h << std::endl;
              std::cout << "LIMIT_ITER " << limit_iter << std::endl;
              abort();
            }

            if (h_max[j] - (hLowj + l_ij_h * P_h_tr) < -1e-12) {
              std::cout << " MAJOR BUG 2b " << std::setprecision(15) << " \n "
                        << " Diff   =  " << h_max[j] - (hLowj + l_ij_h * P_h_tr)
                        << " \n "
                        << " Soln   = " << (hLowj + P_h_tr) << " \n "
                        << " hLowj  = " << hLowj << " \n "
                        << " h_min  = " << h_min[j] << " \n "
                        << " h_max  = " << h_max[j] << " \n "
                        << " P_h_tr = " << P_h_tr << " \n "
                        << " l_ij_h = " << l_ij_h << std::endl;
              std::cout << "LIMIT_ITER " << limit_iter << std::endl;
              abort();
            }
#endif
          }

          /* q1 limiting -- to define l_ji_q1 */
          double l_ji_q1 = l_ji_h;
          {
            const double denominator =
                1. / (std::abs(P_heta) + eps * heta_max[i]);

            // Define limiter here
            if (hetaLowi + P_heta < heta_min[i]) {
              l_ji_q1 = std::min(
                  (std::abs(heta_min[i] - hetaLowi) + eps * heta_min[i]) *
                      denominator,
                  1.);
            } else if (heta_max[i] < hetaLowi + P_heta) {
              l_ji_q1 = std::min(
                  (std::abs(heta_max[i] - hetaLowi) + eps * heta_min[i]) *
                      denominator,
                  1.);
            }

            // set limiter to 0 if water depth is close to 0
            l_ji_q1 = (hLowi <= hEps) ? 0. : l_ji_q1;

            // get min of l_ji_q1 and previous limiter
            l_ji_q1 = std::min(l_ji_q1, l_ji_h);

            /* Box limiter to be safe? */
            l_ji_q1 = std::min(l_ji_q1, 1.);
            l_ji_q1 = std::max(l_ji_q1, 0.);

#if IF_LIMITING_DEBUGGING
            if ((hetaLowi + l_ji_q1 * P_heta) - heta_min[i] < -hEps * hEps) {
              std::cout << " MAJOR BUG 3a " << std::setprecision(15) << " \n "
                        << "New soln " << hetaLowi + l_ji_q1 * P_heta << " \n "
                        << "hetaLowi " << hetaLowi << " \n "
                        << "heta_min " << heta_min[i] << " \n "
                        << "heta_max " << heta_max[i] << " \n "
                        << "l_ji_q1  " << l_ji_q1 << " \n "
                        << "test hi  " << hLowi << std::endl;
              if (heta_max[i] > -hEps)
                abort();
            }

            if (heta_max[i] - (hetaLowi + l_ji_q1 * P_heta) < -hEps * hEps) {
              std::cout << " MAJOR BUG 3b " << std::setprecision(15) << " \n "
                        << "New soln " << hetaLowi + l_ji_q1 * P_heta << " \n "
                        << "hetaLowi " << hetaLowi << " \n "
                        << "heta_min " << heta_min[i] << " \n "
                        << "heta_max " << heta_max[i] << " \n "
                        << "l_ji_q1  " << l_ji_q1 << " \n "
                        << "test hi  " << hLowi << std::endl;
              if (heta_max[i] > -hEps)
                abort();
            }
#endif
          }

          /* q1 limiting -- to define l_ij_q1 */
          double l_ij_q1 = l_ij_h;
          {
            const double denominator =
                1. / (std::abs(P_heta_tr) + eps * heta_max[j]);

            if (hetaLowj + P_heta_tr < heta_min[j]) {
              l_ij_q1 = std::min(
                  (std::abs(heta_min[j] - hetaLowj) + eps * heta_min[j]) *
                      denominator,
                  1.);
            } else if (heta_max[j] < hetaLowj + P_heta_tr) {
              l_ij_q1 = std::min(
                  (std::abs(heta_max[j] - hetaLowj) + eps * heta_min[j]) *
                      denominator,
                  1.);
            }

            // set limiter to 0 if water depth is close to 0
            l_ij_q1 = (hLowj <= hEps) ? 0. : l_ij_q1;

            // get min of l_ij_q1 and previous limiter
            l_ij_q1 = std::min(l_ij_q1, l_ij_h);

            /* Box limiter to be safe? */
            l_ij_q1 = std::min(l_ij_q1, 1.0);
            l_ij_q1 = std::max(l_ij_q1, 0.0);
          }

#if IF_LIMITING_DEBUGGING
          if ((hetaLowj + l_ij_q1 * P_heta_tr) - heta_min[j] < -hEps * hEps) {
            std::cout << " MAJOR BUG 4a " << std::setprecision(15) << " \n "
                      << "New soln " << hetaLowj + l_ij_q1 * P_heta_tr << " \n "
                      << "hetaLowj " << hetaLowj << " \n "
                      << "heta_min " << heta_min[j] << " \n "
                      << "heta_max " << heta_max[j] << " \n "
                      << "l_ij_q1  " << l_ij_q1 << " \n "
                      << "test hj  " << hLowj << std::endl;
            if (heta_max[j] > -hEps)
              abort();
          }

          if (heta_max[j] - (hetaLowj + l_ij_q1 * P_heta_tr) < -hEps * hEps) {
            std::cout << " MAJOR BUG 4b " << std::setprecision(15) << " \n "
                      << "New soln " << hetaLowj + l_ij_q1 * P_heta_tr << " \n "
                      << "hetaLowj " << hetaLowj << " \n "
                      << "heta_min " << heta_min[j] << " \n "
                      << "heta_max " << heta_max[j] << " \n "
                      << "l_ij_q1  " << l_ij_q1 << " \n "
                      << "test hj  " << hLowj << std::endl;
            if (heta_max[j] > -hEps)
              abort();
          }
#endif

          /* kinetic energy limiting -- to define l_ji_K */
          double l_tmp = 0.;
          double l_ji_K = l_ji_q1;
          {
            // We first check if current l_ji_K is a good state
            const double h_r = hLowi + l_ji_K * P_h;
            const double hu_r = huLowi + l_ji_K * P_hu;
            const double hv_r = hvLowi + l_ji_K * P_hv;
            const double psi =
                kin_max[i] * h_r - 0.5 * (hu_r * hu_r + hv_r * hv_r);

            l_tmp = (psi > -KE_tiny) ? l_ji_K : l_tmp;

            const double ai = -0.5 * (P_hu * P_hu + P_hv * P_hv);
            const double ai_nudged = std::min(ai, -KE_tiny);
            const double bi = kinMaxi * P_h - (huLowi * P_hu + hvLowi * P_hv);
            const double ci =
                hLowi * kinMaxi - 0.5 * (huLowi * huLowi + hvLowi * hvLowi);

            const double delta_i = bi * bi - 4. * ai * ci;
            const double root_i =
                0.5 / ai_nudged * (-bi - std::sqrt(std::abs(delta_i)));

            // define l_ji_K
            l_ji_K = (root_i > 0.) ? std::min(root_i, l_ji_K)
                                   : std::min(l_tmp, l_ji_K);

            /* if previous bound was already satisfied, we should take
             l_tmp instead of l_ji_K */
            l_tmp = std::max(l_tmp, l_ji_K);

            // If we are in a "dry state", limiter should be 0
            l_tmp = (hLowi * kinMaxi <= KE_tiny) ? 0. : l_tmp;

            /* Then box to be safe */
            l_tmp = std::min(l_tmp, 1.);
            l_tmp = std::max(l_tmp, 0.);
          }

          /* Define final l_ji_K */
          l_ji_K = std::min(l_tmp, l_ji_q1);

          /* kinetic energy limiting -- to define l_ij_K */
          l_tmp = 0.;
          double l_ij_K = l_ij_q1;
          {
            // We first check if current l_ij_K is a good state
            const double h_r = hLowj + l_ij_K * P_h_tr;
            const double hu_r = huLowj + l_ij_K * P_hu_tr;
            const double hv_r = hvLowj + l_ij_K * P_hv_tr;
            const double psi =
                kinMaxj * h_r - 0.5 * (hu_r * hu_r + hv_r * hv_r);

            l_tmp = (psi > -KE_tiny) ? l_ij_K : l_tmp;

            const double aj = -0.5 * (P_hu_tr * P_hu_tr + P_hv_tr * P_hv_tr);
            const double aj_nudged = std::min(aj, -KE_tiny);
            const double bj =
                kinMaxj * P_h_tr - (huLowj * P_hu_tr + hvLowj * P_hv_tr);
            const double cj =
                hLowj * kinMaxj - 0.5 * (huLowj * huLowj + hvLowj * hvLowj);

            const double delta_j = bj * bj - 4. * aj * cj;
            const double root_j =
                0.5 / aj_nudged * (-bj - std::sqrt(std::abs(delta_j)));

            // define l_ij_K
            l_ij_K = (root_j > 0.) ? std::min(root_j, l_ij_K)
                                   : std::min(l_tmp, l_ij_K);

            /* if previous bound was already satisfied, we should take l_tmp
            instead of l_ij_K */
            l_tmp = std::max(l_tmp, l_ij_K);

            // If we are in a "dry state", limiter should be 0
            l_tmp = (hLowj * kinMaxj <= KE_tiny) ? 0. : l_tmp;

            /* Box limiter */
            l_tmp = std::min(l_tmp, 1.);
            l_tmp = std::max(l_tmp, 0.);
          }

          /* Get final l_ij_K*/
          l_ij_K = std::min(l_tmp, l_ij_q1);

          /* Then we get the final limiter lij */
          Lij_array[ij] = std::min(l_ji_K, l_ij_K);

#if IF_LIMITING_DEBUGGING
          if (Lij_array[ij] > 1. || Lij_array[ij] < 0.) {
            std::cout << "\n Problem with limiter! \n " << Lij_array[ij]
                      << "\n Aborting! " << std::endl;
          }
#endif
          ij += 1;
        } // end j loop
      }   // end i loop

      /* Loop to define limited solution */
      ij = 0;
      for (int i = 0; i < numDOFs; i++) {

        const double one_over_mi = 1. / lumped_mass_matrix[i];
        double ith_Limiter_times_FCT_matrix1 = 0.;
        double ith_Limiter_times_FCT_matrix2 = 0.;
        double ith_Limiter_times_FCT_matrix3 = 0.;
        double ith_Limiter_times_FCT_matrix4 = 0.;
        double ith_Limiter_times_FCT_matrix5 = 0.;
        double ith_Limiter_times_FCT_matrix6 = 0.;

        // LOOP OVER THE SPARSITY PATTERN (j-LOOP)//
        for (int offset = csrRowIndeces_DofLoops[i];
             offset < csrRowIndeces_DofLoops[i + 1]; offset++) {
          int j = csrColumnOffsets_DofLoops[offset];

          // COMPUTE LIMITED FLUX //
          ith_Limiter_times_FCT_matrix1 += Lij_array[ij] * FCT_h[ij];
          ith_Limiter_times_FCT_matrix2 += Lij_array[ij] * FCT_hu[ij];
          ith_Limiter_times_FCT_matrix3 += Lij_array[ij] * FCT_hv[ij];
          ith_Limiter_times_FCT_matrix4 += Lij_array[ij] * FCT_heta[ij];
          ith_Limiter_times_FCT_matrix5 += Lij_array[ij] * FCT_hw[ij];
          ith_Limiter_times_FCT_matrix6 += Lij_array[ij] * FCT_hbeta[ij];

          // update ij
          ij += 1;
        } // end j loop

        // then we add lij*Aij to uLow
        hLow[i] += one_over_mi * ith_Limiter_times_FCT_matrix1;
        huLow[i] += one_over_mi * ith_Limiter_times_FCT_matrix2;
        hvLow[i] += one_over_mi * ith_Limiter_times_FCT_matrix3;
        hetaLow[i] += one_over_mi * ith_Limiter_times_FCT_matrix4;
        hwLow[i] += one_over_mi * ith_Limiter_times_FCT_matrix5;
        hbetaLow[i] += one_over_mi * ith_Limiter_times_FCT_matrix6;

#if IF_LIMITING_DEBUGGING
        if (hLow[i] < -hEps) {
          std::cout
              << " \n New intermediated/limited water depth is negative! \n"
              << "new hLow[i] = " << hLow[i] << " \n "
              << "-hEps   = " << -hEps << "\n"
              << "h_min = " << h_min[i] << "\n"
              << "h_max = " << h_max[i] << "\n"
              << "... Aborting! \n"
              << std::endl;
          abort();
        }
        if (h_max[i] - hLow[i] < -1e-12 || hLow[i] - h_min[i] < -1e-12) {
          std::cout
              << std::setprecision(15)
              << " --- We have a major problem (h limiting bounds) --- \n "
              << "i        = " << i << " \n "
              << "hLow[i]  = " << hLow[i] << " \n "
              << "h_min[i] = " << h_min[i] << " \n "
              << "h_max[i] = " << h_max[i] << " \n "
              << "Diff max = " << h_max[i] - hLow[i] << " \n "
              << "Diff min = " << hLow[i] - h_min[i] << " \n " << std::endl;
          std::cout << "LIMIT_ITER " << limit_iter << std::endl;
          abort();
        }

        if (heta_max[i] - hetaLow[i] < -hEps * hEps ||
            hetaLow[i] - heta_min[i] < -hEps * hEps) {
          std::cout
              << std::setprecision(15)
              << " --- We have a major problem (heta limiting bounds) --- \n
                 "
              << "hetaLow[i]  = " << hetaLow[i] << " \n "
              << "heta_min[i] = " << heta_min[i] << " \n "
              << "heta_max[i] = " << heta_max[i] << " \n "
              << "Diff max = " << heta_max[i] - hetaLow[i] << " \n "
              << "Diff min = " << hetaLow[i] - heta_min[i] << " \n "
              << std::endl;
          abort();
        }
#endif
      } // end i loop

      // update FCT matrices as Fct = (1 - Lij)*Fct
      FCT_h = (1. - Lij_array) * FCT_h;
      FCT_hu = (1. - Lij_array) * FCT_hu;
      FCT_hv = (1. - Lij_array) * FCT_hv;
      FCT_heta = (1. - Lij_array) * FCT_heta;
      FCT_hw = (1. - Lij_array) * FCT_hw;
      FCT_hbeta = (1. - Lij_array) * FCT_hbeta;
    } // end loop for limiting iteration

    if (limit_iter_arg < n_limit_iter - 1) {
      for (std::size_t k = 0; k < nnz_fct; k++) {
        FCT_all[0 * nnz_fct + k] = FCT_h[k];
        FCT_all[1 * nnz_fct + k] = FCT_hu[k];
        FCT_all[2 * nnz_fct + k] = FCT_hv[k];
        FCT_all[3 * nnz_fct + k] = FCT_heta[k];
        FCT_all[4 * nnz_fct + k] = FCT_hw[k];
        FCT_all[5 * nnz_fct + k] = FCT_hbeta[k];
      }
      return; // the caller exchanges hLow etc. and calls the next pass
    }

    /* Update final solution */
    for (int i = 0; i < numDOFs; i++) {

      const double one_over_mi = 1. / lumped_mass_matrix[i];

      limited_hnp1[i] = hLow[i] + dt * one_over_mi * SourceTerm_h[i];
      limited_hunp1[i] = huLow[i] + dt * one_over_mi * extendedSourceTerm_hu[i];
      limited_hvnp1[i] = hvLow[i] + dt * one_over_mi * extendedSourceTerm_hv[i];
      limited_hetanp1[i] = hetaLow[i] + dt * one_over_mi * SourceTerm_heta[i];
      limited_hwnp1[i] = hwLow[i] + dt * one_over_mi * SourceTerm_hw[i];
      limited_hbetanp1[i] =
          hbetaLow[i] + dt * one_over_mi * SourceTerm_hbeta[i];

      if (limited_hnp1[i] < -hEps) {
        std::cout << " \n "
                  << " !!!! Limited water height is negative: !!!! \n"
                  << "     hLim  = " << limited_hnp1[i] << "\n"
                  << "     hEps  = " << hEps << "\n"
                  << "     h_min = " << h_min[i] << "\n"
                  << "     h_max = " << h_max[i] << "\n"
                  << " !!!!             ABORTING              !!!! \n"
                  << std::endl;
        abort();
      } else {
        // clean up uHigh from round off error
        if (limited_hnp1[i] < hEps) {
          limited_hnp1[i] = 0.;
        }

        const double aux = fmax(limited_hnp1[i], hEps);
        limited_hunp1[i] *= 2. * std::pow(limited_hnp1[i], VEL_FIX_POWER) /
                            (std::pow(limited_hnp1[i], VEL_FIX_POWER) +
                             std::pow(aux, VEL_FIX_POWER));
        limited_hvnp1[i] *= 2. * std::pow(limited_hnp1[i], VEL_FIX_POWER) /
                            (std::pow(limited_hnp1[i], VEL_FIX_POWER) +
                             std::pow(aux, VEL_FIX_POWER));
        limited_hwnp1[i] *= 2. * std::pow(limited_hnp1[i], VEL_FIX_POWER) /
                            (std::pow(limited_hnp1[i], VEL_FIX_POWER) +
                             std::pow(aux, VEL_FIX_POWER));
        limited_hbetanp1[i] *= 2. * std::pow(limited_hnp1[i], VEL_FIX_POWER) /
                               (std::pow(limited_hnp1[i], VEL_FIX_POWER) +
                                std::pow(aux, VEL_FIX_POWER));
      }
    }

  } // end convex limiting function

  // Walls: on rows with wall_dof = 1 the advective (inertial) terms are
  // integrated by parts with zero advective flux through the wall, i.e. row i
  // uses c'_ij = -c_ji in place of c_ij (they differ only on boundary edges).
  // The pressure and bathymetry terms stay on c_ij. See SW2DCV.h.
  static inline void wallCoefficients(int i, int j, int ij,
                                      const xt::pyarray<double> &wall_dof,
                                      const xt::pyarray<double> &Cx,
                                      const xt::pyarray<double> &Cy,
                                      const xt::pyarray<double> &CTx,
                                      const xt::pyarray<double> &CTy,
                                      double &cx_ij, double &cy_ij,
                                      double &cx_ji, double &cy_ji) {
    cx_ij = wall_dof[i] > 0.5 ? -CTx[ij] : Cx[ij];
    cy_ij = wall_dof[i] > 0.5 ? -CTy[ij] : Cy[ij];
    cx_ji = wall_dof[j] > 0.5 ? -Cx[ij] : CTx[ij];
    cy_ji = wall_dof[j] > 0.5 ? -Cy[ij] : CTy[ij];
  }

  // Porous edges: graph viscosities carry max(phi_i, phi_j) (see SW2DCV.h).
  static inline double phiBar(int i, int j,
                              const xt::pyarray<double> &porosity_dof) {
    return fmax(porosity_dof[i], porosity_dof[j]);
  }

  double calculateEdgeBasedCFL(arguments_dict &args) {
    const double g = args.scalar<double>("g");
    const int numDOFsPerEqn = args.scalar<int>("numDOFsPerEqn");
    const xt::pyarray<double> &lumped_mass_matrix =
        args.array<double>("lumped_mass_matrix");
    const xt::pyarray<double> &h_dof_old = args.array<double>("h_dof_old");
    const xt::pyarray<double> &hu_dof_old = args.array<double>("hu_dof_old");
    const xt::pyarray<double> &hv_dof_old = args.array<double>("hv_dof_old");
    const xt::pyarray<double> &heta_dof_old =
        args.array<double>("heta_dof_old");
    const xt::pyarray<int> &csrRowIndeces_DofLoops =
        args.array<int>("csrRowIndeces_DofLoops");
    const xt::pyarray<int> &csrColumnOffsets_DofLoops =
        args.array<int>("csrColumnOffsets_DofLoops");
    const double hEps = args.scalar<double>("hEps");
    const xt::pyarray<double> &Cx = args.array<double>("Cx");
    const xt::pyarray<double> &Cy = args.array<double>("Cy");
    const xt::pyarray<double> &CTx = args.array<double>("CTx");
    const xt::pyarray<double> &CTy = args.array<double>("CTy");
    const xt::pyarray<double> &inverse_mesh =
        args.array<double>("inverse_mesh");
    xt::pyarray<double> &edge_based_cfl = args.array<double>("edge_based_cfl");
    const xt::pyarray<double> &wall_dof = args.array<double>("wall_dof");
    const xt::pyarray<double> &porosity_dof = args.array<double>("porosity_dof");

    double max_edge_based_cfl = 0.;
    double dLowij = 0.;

    int ij = 0;
    for (int i = 0; i < numDOFsPerEqn; i++) {

      // solution at time tn for the ith DOF
      const double hi = h_dof_old[i];
      const double one_over_hi =
          2. * hi / (hi * hi + std::pow(fmax(hi, hEps), 2));
      const double hui = hu_dof_old[i];
      const double hvi = hv_dof_old[i];
      const double ui = hui * one_over_hi;
      const double vi = hvi * one_over_hi;
      const double hetai = heta_dof_old[i];
      const double mi = lumped_mass_matrix[i];
      const double inv_meshSizei = inverse_mesh[i];

      // Initialize diagonal entry
      double dLowii = 0.;
      // boundary vector n_i = sum_j c_ji = int_{dOmega} phi_i n (zero inside)
      double nix = 0., niy = 0.;

      for (int offset = csrRowIndeces_DofLoops[i];
           offset < csrRowIndeces_DofLoops[i + 1]; offset++) {

        // loop in j (sparsity pattern)
        const int j = csrColumnOffsets_DofLoops[offset];

        // solution at time tn for the jth DOF
        const double hj = h_dof_old[j];
        const double one_over_hj =
            2. * hj / (hj * hj + std::pow(fmax(hj, hEps), 2));
        const double huj = hu_dof_old[j];
        const double hvj = hv_dof_old[j];
        const double uj = huj * one_over_hj;
        const double vj = hvj * one_over_hj;
        const double hetaj = heta_dof_old[j];
        const double mj = lumped_mass_matrix[j];
        const double inv_meshSizej = inverse_mesh[j];

        if (j != i) {
          ////////////////////////
          // DISSIPATIVE MATRIX //
          ////////////////////////
          const double cij_norm = sqrt(Cx[ij] * Cx[ij] + Cy[ij] * Cy[ij]);
          const double cji_norm = sqrt(CTx[ij] * CTx[ij] + CTy[ij] * CTy[ij]);
          const double nxij = Cx[ij] / cij_norm, nyij = Cy[ij] / cij_norm;
          const double nxji = CTx[ij] / cji_norm, nyji = CTy[ij] / cji_norm;

          double cxa, cya, cxaT, cyaT;
          wallCoefficients(i, j, ij, wall_dof, Cx, Cy, CTx, CTy, cxa, cya,
                           cxaT, cyaT);
          const double muijL = fmax(std::abs(ui * cxa + vi * cya),
                                    std::abs(uj * cxaT + vj * cyaT));
          dLowij = fmax(maxWaveSpeedSharpInitialGuess(
                            g, nxij, nyij, hi, hui, hvi, hetai, inv_meshSizei,
                            hj, huj, hvj, hetaj, inv_meshSizej, hEps) *
                            cij_norm,
                        maxWaveSpeedSharpInitialGuess(
                            g, nxji, nyji, hj, huj, hvj, hetaj, inv_meshSizej,
                            hi, hui, hvi, hetai, inv_meshSizei, hEps) *
                            cji_norm);

          // Take max of dij and muij
          dLowij = fmax(dLowij, muijL) * phiBar(i, j, porosity_dof);

          // Define diagonal entry
          dLowii -= dLowij;
        }
        nix += CTx[ij];
        niy += CTy[ij];

        // update ij
        ij += 1;
      }
      //////////////////////////////
      // CALCULATE EDGE BASED CFL //
      //////////////////////////////
      // A wall row's boundary term adds (u_i.n_i) U_i to the update; flow
      // leaving the wall (u_i.n_i < 0) needs that much more room.
      double wall_margin = 0.;
      if (wall_dof[i] > 0.5)
        wall_margin =
            0.5 * fmax(0., -porosity_dof[i] * (ui * nix + vi * niy));
      edge_based_cfl[i] = (fabs(dLowii) + wall_margin) / mi;
      max_edge_based_cfl = fmax(max_edge_based_cfl, edge_based_cfl[i]);
    }

    return max_edge_based_cfl;
  } // End calculateEdgeBasedCFL

  void calculatePreStep(arguments_dict &args) {
    const double g = args.scalar<double>("g");
    const xt::pyarray<double> &h_dof_old = args.array<double>("h_dof_old");
    const xt::pyarray<double> &hu_dof_old = args.array<double>("hu_dof_old");
    const xt::pyarray<double> &hv_dof_old = args.array<double>("hv_dof_old");
    const xt::pyarray<double> &heta_dof_old =
        args.array<double>("heta_dof_old");
    const double hEps = args.scalar<double>("hEps");
    const int numDOFsPerEqn = args.scalar<int>("numDOFsPerEqn");
    const xt::pyarray<int> &csrRowIndeces_DofLoops =
        args.array<int>("csrRowIndeces_DofLoops");
    const xt::pyarray<int> &csrColumnOffsets_DofLoops =
        args.array<int>("csrColumnOffsets_DofLoops");
    xt::pyarray<double> &entropy = args.array<double>("entropy");
    xt::pyarray<double> &delta_Sqd_h = args.array<double>("delta_Sqd_h");
    xt::pyarray<double> &delta_Sqd_heta = args.array<double>("delta_Sqd_heta");
    xt::pyarray<double> &thetaj_inv = args.array<double>("thetaj_inv");
    double &dij_small = args.scalar<double>("dij_small");
    const double h0_max = args.scalar<double>("h0_max");
    const xt::pyarray<double> &Cx = args.array<double>("Cx");
    const xt::pyarray<double> &Cy = args.array<double>("Cy");

    const double speed = std::sqrt(g * h0_max);
    dij_small = 0.;

    ////////////////////////////////////////
    // ********** LOOP ON DOFs ********** //
    ////////////////////////////////////////
    // To compute:
    //     * Entropy    at i-th nodes
    //     * thetaj_inv at i-th nodes
    //     * second order differences at i-th node
    //     * dij_small

    int ij = 0;
    for (int i = 0; i < numDOFsPerEqn; i++) {

      // Define things at ith node
      const double hi = h_dof_old[i];
      const double hu_i = hu_dof_old[i];
      const double hv_i = hv_dof_old[i];
      const double hetai = heta_dof_old[i];
      const double one_over_hi =
          2. * hi / (hi * hi + std::pow(fmax(hi, hEps), 2)); // hEps
      const double u_i = hu_i * one_over_hi;
      const double v_i = hv_i * one_over_hi;
      const double kin_i = 0.5 * hi * (u_i * u_i + v_i * v_i);

      // Compute entropy (with a flat bottom)
      entropy[i] = ENTROPY(g, hi, hu_i, hv_i, 0., one_over_hi);

      // Define convex coefficients for each i
      thetaj_inv[i] =
          1. / (csrRowIndeces_DofLoops[i + 1] - csrRowIndeces_DofLoops[i] - 1);

      // Initialize sum of differences
      double h_diff = 0.;
      double heta_diff = 0.;
      double kin_diff = 0.;

      for (int offset = csrRowIndeces_DofLoops[i];
           offset < csrRowIndeces_DofLoops[i + 1]; offset++) {
        const int j = csrColumnOffsets_DofLoops[offset];

        // Define things at jth node
        const double hj = h_dof_old[j];
        const double hu_j = hu_dof_old[j];
        const double hv_j = hv_dof_old[j];
        const double hetaj = heta_dof_old[j];
        const double one_over_hj =
            2. * hj / (hj * hj + std::pow(fmax(hj, hEps), 2));
        const double u_j = hu_j * one_over_hj;
        const double v_j = hv_j * one_over_hj;
        const double kin_j = 0.5 * hj * (u_j * u_j + v_j * v_j);

        /* To compute:
         *    Second-difference relaxation quantities
         */
        if (j != i) {
          h_diff += hi - hj;
          heta_diff += hetai - hetaj;
          kin_diff += kin_i - kin_j;
        }

        // Compute dij_small
        const double x = fabs(Cx[ij]) + fabs(Cy[ij]);
        dij_small = fmax(dij_small, x * speed);

        ij += 1;
      } // j loop ends here

      // Set differences
      delta_Sqd_h[i] = h_diff;
      delta_Sqd_heta[i] = heta_diff;

    } // i loop ends here

    // Define final dij_small
    dij_small = 1E-14 * dij_small;

  } // end calculatePreStep

  void calculateEV(arguments_dict &args) {
    const double g = args.scalar<double>("g");
    const xt::pyarray<double> &h_dof_old = args.array<double>("h_dof_old");
    const xt::pyarray<double> &hu_dof_old = args.array<double>("hu_dof_old");
    const xt::pyarray<double> &hv_dof_old = args.array<double>("hv_dof_old");
    const xt::pyarray<double> &Cx = args.array<double>("Cx");
    const xt::pyarray<double> &Cy = args.array<double>("Cy");
    const xt::pyarray<double> &CTx = args.array<double>("CTx");
    const xt::pyarray<double> &CTy = args.array<double>("CTy");
    const int numDOFsPerEqn = args.scalar<int>("numDOFsPerEqn");
    const xt::pyarray<int> &csrRowIndeces_DofLoops =
        args.array<int>("csrRowIndeces_DofLoops");
    const xt::pyarray<int> &csrColumnOffsets_DofLoops =
        args.array<int>("csrColumnOffsets_DofLoops");
    const xt::pyarray<double> &lumped_mass_matrix =
        args.array<double>("lumped_mass_matrix");
    const double hEps = args.scalar<double>("hEps");
    xt::pyarray<double> &global_entropy_residual =
        args.array<double>("global_entropy_residual");
    const xt::pyarray<double> &entropy = args.array<double>("entropy");
    const double h0_max = args.scalar<double>("h0_max");

    ///////////////////////////////////////////////
    // ********** FIRST LOOP ON DOFs ********** //
    ///////////////////////////////////////////////
    // To compute:
    //     * global entropy residual

    int ij = 0;
    for (int i = 0; i < numDOFsPerEqn; i++) {

      // solution at time tn for the ith DOF
      const double hi = h_dof_old[i];
      const double hui = hu_dof_old[i];
      const double hvi = hv_dof_old[i];
      const double one_over_hiReg =
          2. * hi / (hi * hi + std::pow(fmax(hi, hEps), 2)); // hEps
      const double ui = hui * one_over_hiReg;
      const double vi = hvi * one_over_hiReg;
      const double mi = lumped_mass_matrix[i];

      // initialize some things for entropy residual
      double etaMax_i = fabs(entropy[i]);
      double etaMin_i = fabs(entropy[i]);

      double ith_flux_term1 = 0., ith_flux_term2 = 0., ith_flux_term3 = 0.;
      double entropy_flux = 0.;
      double sum_entprime_flux = 0.;
      const double eta_prime1 =
          DENTROPY_DH(g, hi, hui, hvi, 0., one_over_hiReg);
      const double eta_prime2 =
          DENTROPY_DHU(g, hi, hui, hvi, 0., one_over_hiReg);
      const double eta_prime3 =
          DENTROPY_DHV(g, hi, hui, hvi, 0., one_over_hiReg);

      // loop in j (sparsity pattern)
      for (int offset = csrRowIndeces_DofLoops[i];
           offset < csrRowIndeces_DofLoops[i + 1]; offset++) {

        int j = csrColumnOffsets_DofLoops[offset];

        // solution at time tn for the jth DOF
        const double hj = h_dof_old[j];
        const double huj = hu_dof_old[j];
        const double hvj = hv_dof_old[j];
        const double one_over_hjReg =
            2. * hj / (hj * hj + std::pow(fmax(hj, hEps), 2));
        const double uj = huj * one_over_hjReg;
        const double vj = hvj * one_over_hjReg;

        // auxiliary functions to compute fluxes
        const double aux_h =
            (uj * hj - ui * hi) * Cx[ij] + (vj * hj - vi * hi) * Cy[ij];
        const double aux_hu =
            (uj * huj - ui * hui) * Cx[ij] + (vj * huj - vi * hui) * Cy[ij];
        const double aux_hv =
            (uj * hvj - ui * hvi) * Cx[ij] + (vj * hvj - vi * hvi) * Cy[ij];

        // get regular flux
        ith_flux_term1 += aux_h;
        ith_flux_term2 += aux_hu + 0.5 * g * hj * hj * Cx[ij];
        ith_flux_term3 += aux_hv + 0.5 * g * hj * hj * Cy[ij];

        // define entropy flux with flat topography
        entropy_flux +=
            (Cx[ij] * ENTROPY_FLUX1(g, hj, huj, hvj, 0., one_over_hjReg) +
             Cy[ij] * ENTROPY_FLUX2(g, hj, huj, hvj, 0., one_over_hjReg));

        // compute eta_max and eta_min
        etaMax_i = fmax(etaMax_i, fabs(entropy[j]));
        etaMin_i = fmin(etaMin_i, fabs(entropy[j]));

        // update ij
        ij += 1;
      } // end j loop

      // define sum of entprime*flux
      sum_entprime_flux =
          (ith_flux_term1 * eta_prime1 + ith_flux_term2 * eta_prime2 +
           ith_flux_term3 * eta_prime3);

      // define rescale for normalization
      const double small_rescale = g * hEps * h0_max;
      const double rescale =
          fmax(fabs(etaMax_i - etaMin_i) / 2., small_rescale);

      // compute entropy residual
      const double one_over_entNormFactori = 1. / rescale;
      global_entropy_residual[i] =
          one_over_entNormFactori * fabs(entropy_flux - sum_entprime_flux);

      // if "dry" state, set residual to 1
      if (hi <= hEps) {
        global_entropy_residual[i] = 1.;
      }
    } // end i loop

    // ********** END OF LOOP IN DOFs ********** //
  } // end calculateEV

  void calculateBoundsAndHighOrderRHS(arguments_dict &args) {
    const double g = args.scalar<double>("g");
    const xt::pyarray<double> &h_dof_old = args.array<double>("h_dof_old");
    const xt::pyarray<double> &hu_dof_old = args.array<double>("hu_dof_old");
    const xt::pyarray<double> &hv_dof_old = args.array<double>("hv_dof_old");
    const xt::pyarray<double> &heta_dof_old =
        args.array<double>("heta_dof_old");
    const xt::pyarray<double> &hw_dof_old = args.array<double>("hw_dof_old");
    const xt::pyarray<double> &hbeta_dof_old =
        args.array<double>("hbeta_dof_old");
    const xt::pyarray<double> &b_dof = args.array<double>("b_dof");
    const xt::pyarray<double> &Cx = args.array<double>("Cx");
    const xt::pyarray<double> &Cy = args.array<double>("Cy");
    const xt::pyarray<double> &CTx = args.array<double>("CTx");
    const xt::pyarray<double> &CTy = args.array<double>("CTy");
    const int numDOFsPerEqn = args.scalar<int>("numDOFsPerEqn");
    const xt::pyarray<int> &csrRowIndeces_DofLoops =
        args.array<int>("csrRowIndeces_DofLoops");
    const xt::pyarray<int> &csrColumnOffsets_DofLoops =
        args.array<int>("csrColumnOffsets_DofLoops");
    const xt::pyarray<double> &lumped_mass_matrix =
        args.array<double>("lumped_mass_matrix");
    const double hEps = args.scalar<double>("hEps");
    xt::pyarray<double> &SourceTerm_h = args.array<double>("SourceTerm_h");
    xt::pyarray<double> &SourceTerm_hu = args.array<double>("SourceTerm_hu");
    xt::pyarray<double> &SourceTerm_hv = args.array<double>("SourceTerm_hv");
    xt::pyarray<double> &SourceTerm_heta =
        args.array<double>("SourceTerm_heta");
    xt::pyarray<double> &SourceTerm_hw = args.array<double>("SourceTerm_hw");
    xt::pyarray<double> &SourceTerm_hbeta =
        args.array<double>("SourceTerm_hbeta");
    const double dt = args.scalar<double>("dt");
    const double mannings = args.scalar<double>("mannings");
    const int lstage = args.scalar<int>("lstage");
    xt::pyarray<double> &global_entropy_residual =
        args.array<double>("global_entropy_residual");
    const double dij_small = args.scalar<double>("dij_small");
    xt::pyarray<double> &hLow = args.array<double>("hLow");
    xt::pyarray<double> &huLow = args.array<double>("huLow");
    xt::pyarray<double> &hvLow = args.array<double>("hvLow");
    xt::pyarray<double> &hetaLow = args.array<double>("hetaLow");
    xt::pyarray<double> &hwLow = args.array<double>("hwLow");
    xt::pyarray<double> &hbetaLow = args.array<double>("hbetaLow");
    xt::pyarray<double> &h_min = args.array<double>("h_min");
    xt::pyarray<double> &h_max = args.array<double>("h_max");
    xt::pyarray<double> &heta_min = args.array<double>("heta_min");
    xt::pyarray<double> &heta_max = args.array<double>("heta_max");
    xt::pyarray<double> &kin_max = args.array<double>("kin_max");
    const xt::pyarray<double> &x_values = args.array<double>("x_values");
    const xt::pyarray<double> &inverse_mesh =
        args.array<double>("inverse_mesh");
    const double h0_max = args.scalar<double>("h0_max");
    xt::pyarray<double> &RHS_high_h = args.array<double>("RHS_high_h");
    xt::pyarray<double> &RHS_high_hu = args.array<double>("RHS_high_hu");
    xt::pyarray<double> &RHS_high_hv = args.array<double>("RHS_high_hv");
    xt::pyarray<double> &RHS_high_heta = args.array<double>("RHS_high_heta");
    xt::pyarray<double> &RHS_high_hw = args.array<double>("RHS_high_hw");
    xt::pyarray<double> &RHS_high_hbeta = args.array<double>("RHS_high_hbeta");
    xt::pyarray<double> &extendedSourceTerm_hu =
        args.array<double>("extendedSourceTerm_hu");
    xt::pyarray<double> &extendedSourceTerm_hv =
        args.array<double>("extendedSourceTerm_hv");
    double size_of_domain = args.scalar<double>("size_of_domain");
    xt::pyarray<double> &delta_Sqd_h = args.array<double>("delta_Sqd_h");
    xt::pyarray<double> &delta_Sqd_heta = args.array<double>("delta_Sqd_heta");
    const double gen_length = args.scalar<double>("gen_length");
    const double gen_start = args.scalar<double>("gen_start");
    const double abs_length = args.scalar<double>("abs_length");
    const double abs_start = args.scalar<double>("abs_start");
    // xt::pyarray<int> &relaxation_zone_nodes =
    //     args.array<int>("relaxation_zone_nodes");
    xt::pyarray<double> &h_wave = args.array<double>("h_wave");
    xt::pyarray<double> &h_u_wave = args.array<double>("h_u_wave");
    xt::pyarray<double> &h_v_wave = args.array<double>("h_v_wave");
    xt::pyarray<double> &h_eta_wave = args.array<double>("h_eta_wave");
    xt::pyarray<double> &h_w_wave = args.array<double>("h_w_wave");
    xt::pyarray<double> &h_beta_wave = args.array<double>("h_beta_wave");
    // 1 on rows whose advective terms are integrated by parts (no-flux walls)
    const xt::pyarray<double> &wall_dof = args.array<double>("wall_dof");
    // Canopy: porosity (storage phi U, fluxes phi F, pressures phi_i grad p)
    // and Darcy-Forchheimer drag -(alpha + beta|u|) h u per unit plan area
    const xt::pyarray<double> &porosity_dof = args.array<double>("porosity_dof");
    const xt::pyarray<double> &drag_alpha_dof =
        args.array<double>("drag_alpha_dof");
    const xt::pyarray<double> &drag_beta_dof = args.array<double>("drag_beta_dof");

    /* Define constants for sources here. Note that these do not depend on the
     * DOFs so we should only compute once */
    const double n2 = std::pow(mannings, 2.);
    const double gamma = 4. / 3;
    const double xi = 10.;
    const double alpha = 0.005;
    /* ---------------------------------- */

    /////////////////////////////////////////////////////
    // ********** FIRST SET OF LOOP ON DOFs ********** //
    /////////////////////////////////////////////////////
    // To compute:
    //     * Soure terms
    //     * Low order solution (in terms of bar states)
    //     * Local bounds for limiting
    //     * High-order right hand side

    /* ----------- Here we do some initial declaration --------------- */
    // Bar states
    std::valarray<double> hBT(0., Cx.size()), huBT(0., Cx.size()),
        hvBT(0., Cx.size()), hetaBT(0., Cx.size()), hwBT(0., Cx.size()),
        hbetaBT(0., Cx.size()), dLow(0., Cx.size());

    // Relaxation quantities
    std::valarray<double> bar_deltaSqd_h(0., numDOFsPerEqn),
        bar_deltaSqd_heta(0., numDOFsPerEqn);

    double high_viscosity_h, high_viscosity_hu, high_viscosity_hv,
        high_viscosity_heta, high_viscosity_hw, high_viscosity_hbeta;

    // Boundary vectors n_i = sum_j c_ji = int_{dOmega} phi_i n on wall rows.
    // There R_i = -sum_j F_j.c_ji = sum_j (F_j - F_i).c'_ij - F_i.n_i, and
    // F_adv(U).n = (u.n) U, so the boundary term only rescales U_i.
    std::valarray<double> wall_nx(0., numDOFsPerEqn), wall_ny(0., numDOFsPerEqn);
    {
      int ij_w = 0;
      for (int i = 0; i < numDOFsPerEqn; i++)
        for (int offset = csrRowIndeces_DofLoops[i];
             offset < csrRowIndeces_DofLoops[i + 1]; offset++) {
          if (wall_dof[i] > 0.5) {
            wall_nx[i] += CTx[ij_w];
            wall_ny[i] += CTy[ij_w];
          }
          ij_w++;
        }
    }

    /* First loop to compute:
      1. bar states
      2. Sources
      3 high-order RHS
     */
    int ij = 0;
    for (int i = 0; i < numDOFsPerEqn; i++) {

      // define things at ith node
      const double hi = h_dof_old[i];
      const double hui = hu_dof_old[i];
      const double hvi = hv_dof_old[i];
      const double hetai = heta_dof_old[i];
      const double hwi = hw_dof_old[i];
      const double hbetai = hbeta_dof_old[i];
      const double Zi = b_dof[i];
      const double mi = lumped_mass_matrix[i];
      const double one_over_hiReg =
          2. * hi / (hi * hi + std::pow(fmax(hi, hEps), 2));
      const double ui = hui * one_over_hiReg;
      const double vi = hvi * one_over_hiReg;
      const double etai = hetai * one_over_hiReg;
      const double inv_meshSizei = inverse_mesh[i];
      const double x_i = x_values[i]; // x value at ith dof

      // Initialize Sources to 0
      SourceTerm_h[i] = 0.;
      SourceTerm_hu[i] = 0.;
      SourceTerm_hv[i] = 0.;
      SourceTerm_heta[i] = 0.;
      SourceTerm_hw[i] = 0.;
      SourceTerm_hbeta[i] = 0.;

      // Define pTilde at ith node here
      double pTildei =
          -(LAMBDA_MGN * g / 3. * inv_meshSizei) * 6. * hi * (hetai - hi * hi);
      if (IF_BOTH_GAMMA_BRANCHES) {
        double diff_over_h_i = (hetai - hi * hi) * one_over_hiReg;
        if (hetai > std::pow(hi, 2.0)) {
          pTildei = -(LAMBDA_MGN * g / 3.0 * inv_meshSizei) * 2.0 *
                    diff_over_h_i * (etai * etai + etai * hi + hi * hi);
        }
      }

      // Define full pressure
      const double pressure_i = 0.5 * g * hi * hi + pTildei;

      // For extended topography source for bar state verison of low-order
      extendedSourceTerm_hu[i] = 0.;
      extendedSourceTerm_hv[i] = 0.;

      /* ---- For nodal sources, part 1 (assuming source is on RHS) ---------*/

      // Friction terms
      const double veli_norm = std::sqrt(ui * ui + vi * vi);
      const double hi_to_the_gamma = std::pow(fmax(hi, hEps), gamma);
      const double friction_aux =
          veli_norm == 0.
              ? 0.
              : (2 * g * n2 * veli_norm * mi /
                 (hi_to_the_gamma +
                  fmax(hi_to_the_gamma, xi * g * n2 * dt * veli_norm)));
      SourceTerm_hu[i] = -friction_aux * hui;
      SourceTerm_hv[i] = -friction_aux * hvi;

      // Canopy drag, semi-implicit like the Manning term: K hu per unit
      // storage with K = (alpha + beta |u|) / phi
      const double phi_i = porosity_dof[i];
      {
        const double K =
            (drag_alpha_dof[i] + drag_beta_dof[i] * veli_norm) / phi_i;
        if (K > 0.) {
          const double drag_aux = 2. * K * mi / (1. + fmax(1., xi * dt * K));
          SourceTerm_hu[i] -= drag_aux * hui;
          SourceTerm_hv[i] -= drag_aux * hvi;
        }
      }
      // The limited update adds extendedSourceTerm (not SourceTerm) to hu and
      // hv, so friction and drag must be in it too; before this they only
      // reached the high-order RHS and had no net effect.
      extendedSourceTerm_hu[i] += SourceTerm_hu[i];
      extendedSourceTerm_hv[i] += SourceTerm_hv[i];

      // Set flux sum terms to 0.
      double sum_flux_h = 0.;
      double sum_flux_hu = 0.;
      double sum_flux_hv = 0.;
      double sum_flux_heta = 0.;
      double sum_flux_hw = 0.;
      double sum_flux_hbeta = 0.;

      // Initialize sum for high-order viscosity terms
      high_viscosity_h = 0.;
      high_viscosity_hu = 0.;
      high_viscosity_hv = 0.;
      high_viscosity_heta = 0.;
      high_viscosity_hw = 0.;
      high_viscosity_hbeta = 0.;

      // Set some values to 0 for ith node
      double grad_Z_x_i = 0.;
      double grad_Z_y_i = 0.;

      const double card_inv =
          1. / (csrRowIndeces_DofLoops[i + 1] - csrRowIndeces_DofLoops[i]);

      // loop over the sparsity pattern of the i-th DOF
      for (int offset = csrRowIndeces_DofLoops[i];
           offset < csrRowIndeces_DofLoops[i + 1]; offset++) {
        const int j = csrColumnOffsets_DofLoops[offset];

        // define things at jth node
        const double hj = h_dof_old[j];
        const double huj = hu_dof_old[j];
        const double hvj = hv_dof_old[j];
        const double hetaj = heta_dof_old[j];
        const double hwj = hw_dof_old[j];
        const double hbetaj = hbeta_dof_old[j];
        const double Zj = b_dof[j];
        const double one_over_hjReg =
            2. * hj / (hj * hj + std::pow(fmax(hj, hEps), 2));
        const double uj = huj * one_over_hjReg;
        const double vj = hvj * one_over_hjReg;
        const double etaj = hetaj * one_over_hjReg;
        const double mj = lumped_mass_matrix[j];
        const double inv_meshSizej = inverse_mesh[j]; // local mesh size in 2d

        // Define grad_Z_x and grad_Z_y
        grad_Z_x_i += Zj * Cx[ij];
        grad_Z_y_i += Zj * Cy[ij];

        // Define pTilde at jth node
        double pTildej = -(LAMBDA_MGN * g / 3. * inv_meshSizej) * 6. * hj *
                         (hetaj - hj * hj);
        if (IF_BOTH_GAMMA_BRANCHES) {
          double diff_over_h_j = (hetaj - hj * hj) * one_over_hjReg;
          if (hetaj > std::pow(hj, 2.0)) {
            pTildej = -(LAMBDA_MGN * g / 3.0 * inv_meshSizej) * 2.0 *
                      diff_over_h_j * (etaj * etaj + etaj * hj + hj * hj);
          }
        }

        // Define full pressure at jth node
        const double pressure_j = 0.5 * g * hj * hj + pTildej;

        // Define extended source term for hu and hv (porous: times phi_i)
        extendedSourceTerm_hu[i] +=
            phi_i * g * (-hi * Zj + 0.5 * (hj - hi) * (hj - hi)) * Cx[ij];
        extendedSourceTerm_hv[i] +=
            phi_i * g * (-hi * Zj + 0.5 * (hj - hi) * (hj - hi)) * Cy[ij];
        const double phi_j = porosity_dof[j];

        // advective coefficients: c'_ij for row i and c'_ji for row j
        double cxa, cya, cxaT, cyaT;
        wallCoefficients(i, j, ij, wall_dof, Cx, Cy, CTx, CTy, cxa, cya, cxaT,
                         cyaT);

        /*  -------  Define hyperbolic fluxes (for bar states) ------------ */
        // (porous: advective fluxes phi_j F_j, pressure phi_i (p_j - p_i))
        const double flux_h = (phi_j * hj * uj - phi_i * hi * ui) * cxa +
                              (phi_j * hj * vj - phi_i * hi * vi) * cya;
        //
        const double aux_hu = (phi_j * uj * huj - phi_i * ui * hui) * cxa +
                              (phi_j * vj * huj - phi_i * vi * hui) * cya;
        const double flux_hu =
            aux_hu + phi_i * (pressure_j - pressure_i) * Cx[ij];
        //
        const double aux_hv = (phi_j * uj * hvj - phi_i * ui * hvi) * cxa +
                              (phi_j * vj * hvj - phi_i * vi * hvi) * cya;
        const double flux_hv =
            aux_hv + phi_i * (pressure_j - pressure_i) * Cy[ij];
        //
        const double flux_heta =
            (phi_j * uj * hj * etaj - phi_i * ui * hi * etai) * cxa +
            (phi_j * vj * hj * etaj - phi_i * vi * hi * etai) * cya;
        const double flux_hw = (phi_j * uj * hwj - phi_i * ui * hwi) * cxa +
                               (phi_j * vj * hwj - phi_i * vi * hwi) * cya;
        //
        const double flux_hbeta =
            (phi_j * uj * hbetaj - phi_i * ui * hbetai) * cxa +
            (phi_j * vj * hbetaj - phi_i * vi * hbetai) * cya;
        /* --------------------------------------------------------- */

        /* ------ Sum flux terms with well-balancing term on hu,hv ---------- */
        sum_flux_h += flux_h;
        sum_flux_hu += aux_hu + phi_i * (g * hi * (hj + Zj) + pTildej) * Cx[ij];
        sum_flux_hv += aux_hv + phi_i * (g * hi * (hj + Zj) + pTildej) * Cy[ij];
        sum_flux_heta += flux_heta;
        sum_flux_hw += flux_hw;
        sum_flux_hbeta += flux_hbeta;
        /* -------------------------------------------- */

        // Initialize low and high graph-viscosity coefficients to 0
        double muLij = 0., muHij = 0.;
        double dLowij = 0., dLij = 0., dHij = 0.;

        if (j != i) {
          // Put these computations first  before it gets messy
          bar_deltaSqd_h[i] += 0.5 * delta_Sqd_h[j] + 0.5 * delta_Sqd_h[i];
          bar_deltaSqd_heta[i] +=
              0.5 * delta_Sqd_heta[j] + 0.5 * delta_Sqd_heta[i];

          /* ------------ Define viscosity  ------------ */
          const double cij_norm = sqrt(Cx[ij] * Cx[ij] + Cy[ij] * Cy[ij]);
          const double cji_norm = sqrt(CTx[ij] * CTx[ij] + CTy[ij] * CTy[ij]);
          const double nxij = Cx[ij] / cij_norm;
          const double nyij = Cy[ij] / cij_norm;
          const double nxji = CTx[ij] / cji_norm;
          const double nyji = CTy[ij] / cji_norm;
          dLowij = fmax(maxWaveSpeedSharpInitialGuess(
                            g, nxij, nyij, hi, hui, hvi, hetai, inv_meshSizei,
                            hj, huj, hvj, hetaj, inv_meshSizej, hEps) *
                            cij_norm,
                        maxWaveSpeedSharpInitialGuess(
                            g, nxji, nyji, hj, huj, hvj, hetaj, inv_meshSizej,
                            hi, hui, hvi, hetai, inv_meshSizei, hEps) *
                            cji_norm);

          // Low-order graph viscoscity coefficients (symmetric in i, j)
          muLij = fmax(std::abs(ui * cxa + vi * cya),
                       std::abs(uj * cxaT + vj * cyaT));
          const double phibar_ij = phiBar(i, j, porosity_dof);
          dLowij *= phibar_ij;
          muLij *= phibar_ij;
          dLij = fmax(dLowij, muLij);

          // Safe for computation of bounds below (do not pass to others)
          dLow[ij] = dLij;

          // High-order graph viscosity coefficients
          const double dEVij =
              phibar_ij *
              fmax(global_entropy_residual[i], global_entropy_residual[j]);
          dHij = fmin(dLij, dEVij);
          muHij = fmin(muLij, dEVij);
          /* --------------------------------------------- */

          /* ---------- Compute star states ------------ */
          const double hStarij = fmax(0., hi + Zi - fmax(Zi, Zj));
          const double hStarji = fmax(0., hj + Zj - fmax(Zi, Zj));
          const double hStar_ratio_i = hStarij * one_over_hiReg;
          const double hStar_ratio_j = hStarji * one_over_hjReg;
          //
          const double huStarij = hui * hStar_ratio_i;
          const double hvStarij = hvi * hStar_ratio_i;
          const double hetaStarij = hetai * std::pow(hStar_ratio_i, 2);
          const double hwStarij = hwi * hStar_ratio_i;
          const double hbetaStarij = hbetai * hStar_ratio_i;
          //
          const double huStarji = huj * hStar_ratio_j;
          const double hvStarji = hvj * hStar_ratio_j;
          const double hetaStarji = hetaj * std::pow(hStar_ratio_j, 2);
          const double hwStarji = hwj * hStar_ratio_j;
          const double hbetaStarji = hbetaj * hStar_ratio_j;
          /* ------------------------------------------- */

          /* --------- Bar states for i not equal j --------- */
          double hBar_ij = 0., hTilde_ij = 0., huBar_ij = 0., huTilde_ij = 0.,
                 hvBar_ij = 0., hvTilde_ij = 0., hetaBar_ij = 0.,
                 hetaTilde_ij = 0., hwBar_ij = 0., hwTilde_ij = 0.,
                 hbetaBar_ij = 0., hbetaTilde_ij = 0.;

          const double half_dij_inv = -0.5 / fmax(dLij, dij_small);
          const double visc_ratio = -half_dij_inv * (dLij - muLij);

          // h component
          hBar_ij = half_dij_inv * flux_h + 0.5 * (hj + hi);
          hTilde_ij = visc_ratio * (hStarji - hj - (hStarij - hi));

          // hu component
          huBar_ij = half_dij_inv * flux_hu + 0.5 * (huj + hui);
          huTilde_ij = visc_ratio * (huStarji - huj - (huStarij - hui));

          // hv component
          hvBar_ij = half_dij_inv * flux_hv + 0.5 * (hvj + hvi);
          hvTilde_ij = visc_ratio * (hvStarji - hvj - (hvStarij - hvi));

          // heta component
          hetaBar_ij = half_dij_inv * flux_heta + 0.5 * (hetaj + hetai);
          hetaTilde_ij =
              visc_ratio * (hetaStarji - hetaj - (hetaStarij - hetai));

          // hw component
          hwBar_ij = half_dij_inv * flux_hw + 0.5 * (hwj + hwi);
          hwTilde_ij = visc_ratio * (hwStarji - hwj - (hwStarij - hwi));

          // hbeta component
          hbetaBar_ij = half_dij_inv * flux_hbeta + 0.5 * (hbetaj + hbetai);
          hbetaTilde_ij =
              visc_ratio * (hbetaStarji - hbetaj - (hbetaStarij - hbetai));

          /* Define bar states here. Note we take max with 0 for hBT due to
           potential round off */
          hBT[ij] = std::max(hBar_ij + hTilde_ij, 0.);
          huBT[ij] = huBar_ij + huTilde_ij;
          hvBT[ij] = hvBar_ij + hvTilde_ij;
          hetaBT[ij] = hetaBar_ij + hetaTilde_ij;
          hwBT[ij] = hwBar_ij + hwTilde_ij;
          hbetaBT[ij] = hbetaBar_ij + hbetaTilde_ij;
          /* ------------------------------------------- */

          /* --------- Sum high viscosity terms ------- */
          high_viscosity_h +=
              (dHij - muHij) * (hStarji - hStarij) + muHij * (hj - hi);

          high_viscosity_hu +=
              (dHij - muHij) * (huStarji - huStarij) + muHij * (huj - hui);

          high_viscosity_hv +=
              (dHij - muHij) * (hvStarji - hvStarij) + muHij * (hvj - hvi);

          high_viscosity_heta += (dHij - muHij) * (hetaStarji - hetaStarij) +
                                 muHij * (hetaj - hetai);

          high_viscosity_hw +=
              (dHij - muHij) * (hwStarji - hwStarij) + muHij * (hwj - hwi);

          high_viscosity_hbeta += (dHij - muHij) * (hbetaStarji - hbetaStarij) +
                                  muHij * (hbetaj - hbetai);
          /* ------------------------------------------- */

        } else {
          // Bar states by definition satisfy Utilde_ii + Ubar_ii = U_i
          hBT[ij] = hi;
          huBT[ij] = hui;
          hvBT[ij] = hvi;
          hetaBT[ij] = hetai;
          hwBT[ij] = hwi;
          hbetaBT[ij] = hbetai;
        }

        // UPDATE ij //
        ij += 1;
      } // j loop ends here

      // wall rows: advective boundary term -F_i . n_i = -(u_i . n_i) U_i
      if (wall_dof[i] > 0.5) {
        const double un_i = phi_i * (ui * wall_nx[i] + vi * wall_ny[i]);
        sum_flux_h -= un_i * hi;
        sum_flux_hu -= un_i * hui;
        sum_flux_hv -= un_i * hvi;
        sum_flux_heta -= un_i * hi * etai;
        sum_flux_hw -= un_i * hwi;
        sum_flux_hbeta -= un_i * hbetai;
      }

      // for bar_deltaSqd_h, bar_deltaSqd_heta
      bar_deltaSqd_h[i] *= card_inv * 0.5;
      bar_deltaSqd_heta[i] *= card_inv * 0.5;

      /* ---- For nodal sources, part 2 (assuming source is on RHS) ---------*/
      // PDE source terms
      double hSqd_GammaPi = 6.0 * (hetai - hi * hi);
      if (IF_BOTH_GAMMA_BRANCHES) {
        const double diff_over_h_i = (hetai - hi * hi) * one_over_hiReg;
        if (hetai > std::pow(hi, 2.0)) {
          hSqd_GammaPi = 6.0 * etai * diff_over_h_i;
        }
      }

      // grad_Z_i / m_i must be per unit plan area: m_i is phi-weighted, so
      // scale by phi_i; the topography-dispersion force is a pressure term
      // and carries phi_i as well
      const double q_dot_gradZ =
          phi_i * (hui * grad_Z_x_i + hvi * grad_Z_y_i);
      const double R1 = hwi - 1.5 * q_dot_gradZ / mi;
      const double R2 = LAMBDA_MGN * g * inv_meshSizei * hSqd_GammaPi;
      const double R3 = LAMBDA_MGN * std::sqrt(g * h0_max) * inv_meshSizei *
                        (q_dot_gradZ / mi - hbetai);

      SourceTerm_h[i] += 0.;
      // topography-dispersion force, the momentum part of the PDE source R.
      // The limited update must put back all of tau (R_i + S_i) (Guermond et
      // al. 2022, Water Waves, eq. 6.13); it was only in the high-order RHS.
      const double topo_disp_x = phi_i * (0.5 * R2 - 0.25 * R3) * grad_Z_x_i;
      const double topo_disp_y = phi_i * (0.5 * R2 - 0.25 * R3) * grad_Z_y_i;
      SourceTerm_hu[i] += topo_disp_x;
      SourceTerm_hv[i] += topo_disp_y;
      extendedSourceTerm_hu[i] += topo_disp_x;
      extendedSourceTerm_hv[i] += topo_disp_y;
      SourceTerm_heta[i] += mi * R1;
      SourceTerm_hw[i] += -mi * R2;
      SourceTerm_hbeta[i] += mi * R3;

      /* Generation/absorption sources */
      if (gen_length > 0.) {
        const double shift = -(gen_start - gen_length);
        const double xhat = (x_i + shift) / gen_length;
        const double function_gen = relaxation(xhat, alpha);

        SourceTerm_h[i] += -mi * std::sqrt(g * h0_max) * inv_meshSizei *
                           function_gen * (hi - h_wave[i]);
        SourceTerm_hu[i] += -mi * std::sqrt(g * h0_max) * inv_meshSizei *
                            function_gen * (hui - h_u_wave[i]);
        SourceTerm_hv[i] += -mi * std::sqrt(g * h0_max) * inv_meshSizei *
                            function_gen * (hvi - h_v_wave[i]);
        SourceTerm_heta[i] += -mi * std::sqrt(g * h0_max) * inv_meshSizei *
                              function_gen * (hetai - h_eta_wave[i]);
        SourceTerm_hw[i] += -mi * std::sqrt(g * h0_max) * inv_meshSizei *
                            function_gen * (hwi - h_w_wave[i]);
        SourceTerm_hbeta[i] += -mi * std::sqrt(g * h0_max) * inv_meshSizei *
                               function_gen * (hbetai - 0.);
        // for hu and hv
        extendedSourceTerm_hu[i] += -mi * std::sqrt(g * h0_max) *
                                    inv_meshSizei * function_gen *
                                    (hui - h_u_wave[i]);
        extendedSourceTerm_hv[i] += -mi * std::sqrt(g * h0_max) *
                                    inv_meshSizei * function_gen *
                                    (hvi - h_v_wave[i]);
      }

      if (abs_length > 0.) {
        const double shift = (abs_start + abs_length);
        const double xhat = (shift - x_i) / abs_length;
        const double function_gen = relaxation(xhat, alpha);

        SourceTerm_hu[i] += -mi * std::sqrt(g * h0_max) * inv_meshSizei *
                            function_gen * (hui - 0.);
        SourceTerm_hv[i] += -mi * std::sqrt(g * h0_max) * inv_meshSizei *
                            function_gen * (hvi - 0.);
        SourceTerm_hw[i] += -mi * std::sqrt(g * h0_max) * inv_meshSizei *
                            function_gen * (hwi - 0.);
        SourceTerm_hbeta[i] += -mi * std::sqrt(g * h0_max) * inv_meshSizei *
                               function_gen * (hbetai - 0.);
        // for hu and hv
        extendedSourceTerm_hu[i] += -mi * std::sqrt(g * h0_max) *
                                    inv_meshSizei * function_gen * (hui - 0.);
        extendedSourceTerm_hv[i] += -mi * std::sqrt(g * h0_max) *
                                    inv_meshSizei * function_gen * (hvi - 0.);
      }
      /* ----------------------------------------------------- */

      /* ----------- Save high order RHS -------------------- */
      RHS_high_h[i] = SourceTerm_h[i] - sum_flux_h + high_viscosity_h;
      RHS_high_hu[i] = SourceTerm_hu[i] - sum_flux_hu + high_viscosity_hu;
      RHS_high_hv[i] = SourceTerm_hv[i] - sum_flux_hv + high_viscosity_hv;
      RHS_high_heta[i] =
          SourceTerm_heta[i] - sum_flux_heta + high_viscosity_heta;
      RHS_high_hw[i] = SourceTerm_hw[i] - sum_flux_hw + high_viscosity_hw;
      RHS_high_hbeta[i] =
          SourceTerm_hbeta[i] - sum_flux_hbeta + high_viscosity_hbeta;
      /* ----------------------------------------------------*/

    } // i loops ends here

    /* Then final loop to get low order solution and local bounds */
    ij = 0;
    for (int i = 0; i < numDOFsPerEqn; i++) {

      /* Define things at ith node */
      const double hi = h_dof_old[i];
      const double hu_i = hu_dof_old[i];
      const double hv_i = hv_dof_old[i];
      const double hetai = heta_dof_old[i];
      const double hwi = hw_dof_old[i];
      const double hbetai = hbeta_dof_old[i];
      const double one_over_hi =
          2. * hi / (hi * hi + std::pow(fmax(hi, hEps), 2)); // hEps
      const double u_i = hu_i * one_over_hi;
      const double v_i = hv_i * one_over_hi;
      const double kin_i = 0.5 * hi * (u_i * u_i + v_i * v_i);
      const double mi = lumped_mass_matrix[i];

      /* Initialize bounds */
      h_min[i] = hi;
      h_max[i] = hi;
      heta_min[i] = hetai;
      heta_max[i] = hetai;
      kin_max[i] = kin_i;

      // For convex low-order update
      double sum_dij = 0.;
      double sum_dij_hbar = 0.;
      double sum_dij_hubar = 0.;
      double sum_dij_hvbar = 0.;
      double sum_dij_hetabar = 0.;
      double sum_dij_hwbar = 0.;
      double sum_dij_hbetabar = 0.;

      // loop in j (sparsity pattern)
      for (int offset = csrRowIndeces_DofLoops[i];
           offset < csrRowIndeces_DofLoops[i + 1]; offset++) {
        const int j = csrColumnOffsets_DofLoops[offset];

        const double one_over_hBT =
            2. * hBT[ij] /
            (hBT[ij] * hBT[ij] + std::pow(fmax(hBT[ij], hEps), 2));
        const double psi_ij =
            0.5 * one_over_hBT * (huBT[ij] * huBT[ij] + hvBT[ij] * hvBT[ij]);

        // compute local bounds
        h_min[i] = std::min(h_min[i], hBT[ij]);
        h_max[i] = std::max(h_max[i], hBT[ij]);
        heta_min[i] = std::min(heta_min[i], hetaBT[ij]);
        heta_max[i] = std::max(heta_max[i], hetaBT[ij]);
        kin_max[i] = fmax(psi_ij, kin_max[i]);

        if (j != i) {
          sum_dij += dLow[ij];
          sum_dij_hbar += dLow[ij] * hBT[ij];
          sum_dij_hubar += dLow[ij] * huBT[ij];
          sum_dij_hvbar += dLow[ij] * hvBT[ij];
          sum_dij_hetabar += dLow[ij] * hetaBT[ij];
          sum_dij_hwbar += dLow[ij] * hwBT[ij];
          sum_dij_hbetabar += dLow[ij] * hbetaBT[ij];
        }

        // update ij
        ij += 1;
      } // j loop ends here

      const double cfl_condition = (1. - dt / mi * 2. * sum_dij);

      // Here we define low-order convex update (WITHOUT SOURCES)
      hLow[i] = hi * cfl_condition + dt / mi * (2. * sum_dij_hbar);
      huLow[i] = hu_i * cfl_condition + dt / mi * (2. * sum_dij_hubar);
      hvLow[i] = hv_i * cfl_condition + dt / mi * (2. * sum_dij_hvbar);
      hetaLow[i] = hetai * cfl_condition + dt / mi * (2. * sum_dij_hetabar);
      hwLow[i] = hwi * cfl_condition + dt / mi * (2. * sum_dij_hwbar);
      hbetaLow[i] = hbetai * cfl_condition + dt / mi * (2. * sum_dij_hbetabar);

      // wall rows: + dt/m_i (u_i . n_i) U_i from the advective boundary term.
      // This rescales U_i, so the low-order state is a positive (not convex)
      // combination; widen the local bounds to contain it.
      if (wall_dof[i] > 0.5) {
        const double un_i =
            porosity_dof[i] * (u_i * wall_nx[i] + v_i * wall_ny[i]);
        const double etai_w =
            hetai * one_over_hi; // F_heta = u h eta, as in flux_heta
        hLow[i] += dt / mi * un_i * hi;
        huLow[i] += dt / mi * un_i * hu_i;
        hvLow[i] += dt / mi * un_i * hv_i;
        hetaLow[i] += dt / mi * un_i * hi * etai_w;
        hwLow[i] += dt / mi * un_i * hwi;
        hbetaLow[i] += dt / mi * un_i * hbetai;
        h_min[i] = std::min(h_min[i], hLow[i]);
        h_max[i] = std::max(h_max[i], hLow[i]);
        heta_min[i] = std::min(heta_min[i], hetaLow[i]);
        heta_max[i] = std::max(heta_max[i], hetaLow[i]);
        const double one_over_hLow =
            2. * hLow[i] /
            (hLow[i] * hLow[i] + std::pow(fmax(hLow[i], hEps), 2));
        kin_max[i] = fmax(kin_max[i],
                          0.5 * one_over_hLow *
                              (huLow[i] * huLow[i] + hvLow[i] * hvLow[i]));
      }

      // clean up hLow from round off error
      if (dt < 1e-2 && hLow[i] < -hEps) {
        std::cout << "Low-order water depth is negative !!! " << hLow[i] << "\n"
                  << "hLow[i] + hEps                    !!! " << hLow[i] + hEps
                  << "\n"
                  << "hEps                              !!! " << hEps << "\n"
                  << " ... aborting!" << std::endl;
        abort();
      } else {
        if (hLow[i] <= hEps) {
          // Clean up low-order h and heta
          hLow[i] = 0.;
        }
      }

#if IF_DEBUGGING
      if (h_min[i] < 0.) {
        std::cout << " \n "
                  << " Minimum water depth is negative !!! " << h_min[i]
                  << "\n "
                  << " hLow[i]                         !!! " << hLow[i] << "\n "
                  << " ... aborting!"
                  << "\n"
                  << std::endl;
      } else {
        if (h_min[i] <= hEps) {
          // Clean up low-order h and heta
          h_min[i] = 0.;
        }
      }

      if (h_max[i] < 0.) {
        std::cout << " Maximum water depth is negative !!! " << h_max[i] << "\n"
                  << " hLow[i]                         !!! " << hLow[i] << "\n "
                  << " ... aborting!" << std::endl;
        abort();
      } else {
        if (h_max[i] <= hEps) {
          // Clean up low-order h and heta
          h_max[i] = 0.;
        }
      }
#endif

      /* Relaxation of bounds */
      const double s_i = std::pow(std::sqrt(std::sqrt(mi / size_of_domain)), 3);
      const double urelax_i = 1. + s_i;
      const double drelax_i = 1. - s_i;

      kin_max[i] =
          std::max((1. + std::sqrt(mi / size_of_domain)) * kin_max[i], 0.);
      h_min[i] =
          std::max(drelax_i * h_min[i], h_min[i] - std::abs(bar_deltaSqd_h[i]));
      h_max[i] =
          std::min(urelax_i * h_max[i], h_max[i] + std::abs(bar_deltaSqd_h[i]));
      heta_min[i] = std::max(drelax_i * heta_min[i],
                             heta_min[i] - std::abs(bar_deltaSqd_heta[i]));
      heta_max[i] = std::min(urelax_i * heta_max[i],
                             heta_max[i] + std::abs(bar_deltaSqd_heta[i]));

#if IF_DEBUGGING
      if (hLow[i] > h_max[i] || hLow[i] < h_min[i]) {
        std::cout << " --- We have a major problem (h bounds) --- "
                  << std::setprecision(15) << std::endl;
        std::cout << "hLow[i]  = " << hLow[i] << " \n "
                  << "h_min[i] = " << h_min[i] << " \n "
                  << "h_max[i] = " << h_max[i] << " \n "
                  << "Diff max = " << h_max[i] - hLow[i] << " \n "
                  << "Diff min = " << hLow[i] - h_min[i] << " \n " << std::endl;
      }

      if (heta_max[i] - hetaLow[i] < -hEps * hEps ||
          hetaLow[i] - heta_min[i] < -hEps * hEps) {
        std::cout << " --- We have a major problem (heta bounds) --- "
                  << std::setprecision(15) << std::endl;
        std::cout << "hetaLow[i]  = " << hetaLow[i] << " \n "
                  << "heta_min[i] = " << heta_min[i] << " \n "
                  << "heta_max[i] = " << heta_max[i] << " \n "
                  << "Diff max = " << heta_max[i] - hetaLow[i] << " \n "
                  << "Diff min = " << hetaLow[i] - heta_min[i] << " \n "
                  << std::endl;
      }
#endif
    } // i loop ends here

    // ********** END OF LOOP IN DOFs ********** //
  } // end calculateBoundsAndHighOrderRHS

  void calculateResidual(arguments_dict &args) {
    xt::pyarray<double> &mesh_trial_ref = args.array<double>("mesh_trial_ref");
    xt::pyarray<double> &mesh_grad_trial_ref =
        args.array<double>("mesh_grad_trial_ref");
    xt::pyarray<double> &mesh_dof = args.array<double>("mesh_dof");
    xt::pyarray<int> &mesh_l2g = args.array<int>("mesh_l2g");
    xt::pyarray<double> &dV_ref = args.array<double>("dV_ref");
    xt::pyarray<double> &h_trial_ref = args.array<double>("h_trial_ref");
    xt::pyarray<double> &h_grad_trial_ref =
        args.array<double>("h_grad_trial_ref");
    xt::pyarray<double> &h_test_ref = args.array<double>("h_test_ref");
    xt::pyarray<double> &h_grad_test_ref =
        args.array<double>("h_grad_test_ref");
    xt::pyarray<double> &vel_trial_ref = args.array<double>("vel_trial_ref");
    xt::pyarray<double> &vel_grad_trial_ref =
        args.array<double>("vel_grad_trial_ref");
    xt::pyarray<double> &vel_test_ref = args.array<double>("vel_test_ref");
    xt::pyarray<double> &vel_grad_test_ref =
        args.array<double>("vel_grad_test_ref");
    xt::pyarray<double> &mesh_trial_trace_ref =
        args.array<double>("mesh_trial_trace_ref");
    xt::pyarray<double> &mesh_grad_trial_trace_ref =
        args.array<double>("mesh_grad_trial_trace_ref");
    xt::pyarray<double> &h_trial_trace_ref =
        args.array<double>("h_trial_trace_ref");
    xt::pyarray<double> &h_grad_trial_trace_ref =
        args.array<double>("h_grad_trial_trace_ref");
    xt::pyarray<double> &h_test_trace_ref =
        args.array<double>("h_test_trace_ref");
    xt::pyarray<double> &h_grad_test_trace_ref =
        args.array<double>("h_grad_test_trace_ref");
    xt::pyarray<double> &vel_trial_trace_ref =
        args.array<double>("vel_trial_trace_ref");
    xt::pyarray<double> &vel_grad_trial_trace_ref =
        args.array<double>("vel_grad_trial_trace_ref");
    xt::pyarray<double> &vel_test_trace_ref =
        args.array<double>("vel_test_trace_ref");
    xt::pyarray<double> &vel_grad_test_trace_ref =
        args.array<double>("vel_grad_test_trace_ref");
    xt::pyarray<double> &normal_ref = args.array<double>("normal_ref");
    xt::pyarray<double> &boundaryJac_ref =
        args.array<double>("boundaryJac_ref");
    xt::pyarray<double> &elementDiameter =
        args.array<double>("elementDiameter");
    int nElements_global = args.scalar<int>("nElements_global");
    double g = args.scalar<double>("g");
    xt::pyarray<double> &porosity_dof = args.array<double>("porosity_dof");
    xt::pyarray<int> &h_l2g = args.array<int>("h_l2g");
    xt::pyarray<int> &vel_l2g = args.array<int>("vel_l2g");
    xt::pyarray<double> &h_dof_old = args.array<double>("h_dof_old");
    xt::pyarray<double> &hu_dof_old = args.array<double>("hu_dof_old");
    xt::pyarray<double> &hv_dof_old = args.array<double>("hv_dof_old");
    xt::pyarray<double> &heta_dof_old = args.array<double>("heta_dof_old");
    xt::pyarray<double> &hw_dof_old = args.array<double>("hw_dof_old");
    xt::pyarray<double> &hbeta_dof_old = args.array<double>("hbeta_dof_old");
    xt::pyarray<double> &b_dof = args.array<double>("b_dof");
    xt::pyarray<double> &h_dof = args.array<double>("h_dof");
    xt::pyarray<double> &hu_dof = args.array<double>("hu_dof");
    xt::pyarray<double> &hv_dof = args.array<double>("hv_dof");
    xt::pyarray<double> &heta_dof = args.array<double>("heta_dof");
    xt::pyarray<double> &hw_dof = args.array<double>("hw_dof");
    xt::pyarray<double> &hbeta_dof = args.array<double>("hbeta_dof");
    xt::pyarray<double> &q_cfl = args.array<double>("q_cfl");
    xt::pyarray<int> &sdInfo_hu_hu_rowptr =
        args.array<int>("sdInfo_hu_hu_rowptr");
    xt::pyarray<int> &sdInfo_hu_hu_colind =
        args.array<int>("sdInfo_hu_hu_colind");
    xt::pyarray<int> &sdInfo_hu_hv_rowptr =
        args.array<int>("sdInfo_hu_hv_rowptr");
    xt::pyarray<int> &sdInfo_hu_hv_colind =
        args.array<int>("sdInfo_hu_hv_colind");
    xt::pyarray<int> &sdInfo_hv_hv_rowptr =
        args.array<int>("sdInfo_hv_hv_rowptr");
    xt::pyarray<int> &sdInfo_hv_hv_colind =
        args.array<int>("sdInfo_hv_hv_colind");
    xt::pyarray<int> &sdInfo_hv_hu_rowptr =
        args.array<int>("sdInfo_hv_hu_rowptr");
    xt::pyarray<int> &sdInfo_hv_hu_colind =
        args.array<int>("sdInfo_hv_hu_colind");
    int offset_h = args.scalar<int>("offset_h");
    int offset_hu = args.scalar<int>("offset_hu");
    int offset_hv = args.scalar<int>("offset_hv");
    int offset_heta = args.scalar<int>("offset_heta");
    int offset_hw = args.scalar<int>("offset_hw");
    int offset_hbeta = args.scalar<int>("offset_hbeta");
    int stride_h = args.scalar<int>("stride_h");
    int stride_hu = args.scalar<int>("stride_hu");
    int stride_hv = args.scalar<int>("stride_hv");
    int stride_heta = args.scalar<int>("stride_heta");
    int stride_hw = args.scalar<int>("stride_hw");
    int stride_hbeta = args.scalar<int>("stride_hbeta");
    xt::pyarray<double> &globalResidual = args.array<double>("globalResidual");
    int nExteriorElementBoundaries_global =
        args.scalar<int>("nExteriorElementBoundaries_global");
    xt::pyarray<int> &exteriorElementBoundariesArray =
        args.array<int>("exteriorElementBoundariesArray");
    xt::pyarray<int> &elementBoundaryElementsArray =
        args.array<int>("elementBoundaryElementsArray");
    xt::pyarray<int> &elementBoundaryLocalElementBoundariesArray =
        args.array<int>("elementBoundaryLocalElementBoundariesArray");
    xt::pyarray<int> &isDOFBoundary_h = args.array<int>("isDOFBoundary_h");
    xt::pyarray<int> &isDOFBoundary_hu = args.array<int>("isDOFBoundary_hu");
    xt::pyarray<int> &isDOFBoundary_hv = args.array<int>("isDOFBoundary_hv");
    xt::pyarray<int> &isAdvectiveFluxBoundary_h =
        args.array<int>("isAdvectiveFluxBoundary_h");
    xt::pyarray<int> &isAdvectiveFluxBoundary_hu =
        args.array<int>("isAdvectiveFluxBoundary_hu");
    xt::pyarray<int> &isAdvectiveFluxBoundary_hv =
        args.array<int>("isAdvectiveFluxBoundary_hv");
    xt::pyarray<int> &isDiffusiveFluxBoundary_hu =
        args.array<int>("isDiffusiveFluxBoundary_hu");
    xt::pyarray<int> &isDiffusiveFluxBoundary_hv =
        args.array<int>("isDiffusiveFluxBoundary_hv");
    xt::pyarray<double> &ebqe_bc_h_ext = args.array<double>("ebqe_bc_h_ext");
    xt::pyarray<double> &ebqe_bc_flux_mass_ext =
        args.array<double>("ebqe_bc_flux_mass_ext");
    xt::pyarray<double> &ebqe_bc_flux_mom_hu_adv_ext =
        args.array<double>("ebqe_bc_flux_mom_hu_adv_ext");
    xt::pyarray<double> &ebqe_bc_flux_mom_hv_adv_ext =
        args.array<double>("ebqe_bc_flux_mom_hv_adv_ext");
    xt::pyarray<double> &ebqe_bc_hu_ext = args.array<double>("ebqe_bc_hu_ext");
    xt::pyarray<double> &ebqe_bc_flux_hu_diff_ext =
        args.array<double>("ebqe_bc_flux_hu_diff_ext");
    xt::pyarray<double> &ebqe_penalty_ext =
        args.array<double>("ebqe_penalty_ext");
    xt::pyarray<double> &ebqe_bc_hv_ext = args.array<double>("ebqe_bc_hv_ext");
    xt::pyarray<double> &ebqe_bc_flux_hv_diff_ext =
        args.array<double>("ebqe_bc_flux_hv_diff_ext");
    xt::pyarray<double> &q_velocity = args.array<double>("q_velocity");
    xt::pyarray<double> &ebqe_velocity = args.array<double>("ebqe_velocity");
    xt::pyarray<double> &flux = args.array<double>("flux");
    xt::pyarray<double> &elementResidual_h_save =
        args.array<double>("elementResidual_h_save");
    const xt::pyarray<double> &Cx = args.array<double>("Cx");
    const xt::pyarray<double> &Cy = args.array<double>("Cy");
    const xt::pyarray<double> &CTx = args.array<double>("CTx");
    const xt::pyarray<double> &CTy = args.array<double>("CTy");
    const int numDOFsPerEqn = args.scalar<int>("numDOFsPerEqn");
    const xt::pyarray<int> &csrRowIndeces_DofLoops =
        args.array<int>("csrRowIndeces_DofLoops");
    const xt::pyarray<int> &csrColumnOffsets_DofLoops =
        args.array<int>("csrColumnOffsets_DofLoops");
    const xt::pyarray<double> &lumped_mass_matrix =
        args.array<double>("lumped_mass_matrix");
    const double cfl_run = args.scalar<double>("cfl_run");
    const double hEps = args.scalar<double>("hEps");
    xt::pyarray<double> &hnp1_at_quad_point =
        args.array<double>("hnp1_at_quad_point");
    xt::pyarray<double> &hunp1_at_quad_point =
        args.array<double>("hunp1_at_quad_point");
    xt::pyarray<double> &hvnp1_at_quad_point =
        args.array<double>("hvnp1_at_quad_point");
    xt::pyarray<double> &hetanp1_at_quad_point =
        args.array<double>("hetanp1_at_quad_point");
    xt::pyarray<double> &hwnp1_at_quad_point =
        args.array<double>("hwnp1_at_quad_point");
    xt::pyarray<double> &hbetanp1_at_quad_point =
        args.array<double>("hbetanp1_at_quad_point");
    int LUMPED_MASS_MATRIX = args.scalar<int>("LUMPED_MASS_MATRIX");
    double dt = args.scalar<double>("dt");
    xt::pyarray<double> &quantDOFs = args.array<double>("quantDOFs");
    int SECOND_CALL_CALCULATE_RESIDUAL =
        args.scalar<int>("SECOND_CALL_CALCULATE_RESIDUAL");
    int COMPUTE_NORMALS = args.scalar<int>("COMPUTE_NORMALS");
    xt::pyarray<double> &normalx = args.array<double>("normalx");
    xt::pyarray<double> &normaly = args.array<double>("normaly");
    const int lstage = args.scalar<int>("lstage");
    const xt::pyarray<double> &MassMatrix = args.array<double>("MassMatrix");
    const xt::pyarray<double> &RHS_high_h = args.array<double>("RHS_high_h");
    const xt::pyarray<double> &RHS_high_hu = args.array<double>("RHS_high_hu");
    const xt::pyarray<double> &RHS_high_hv = args.array<double>("RHS_high_hv");
    const xt::pyarray<double> &RHS_high_heta =
        args.array<double>("RHS_high_heta");
    const xt::pyarray<double> &RHS_high_hw = args.array<double>("RHS_high_hw");
    const xt::pyarray<double> &RHS_high_hbeta =
        args.array<double>("RHS_high_hbeta");

    //////////////////////////////////////
    // ********** CELL LOOPS ********** //
    //////////////////////////////////////
    // To compute:
    //      * Time derivative term
    //      * Cell based CFL
    //      * Velocity and soln at quad points (for other models)
    for (int eN = 0; eN < nElements_global; eN++) {
      // declare local storage for element residual and initialize
      double elementResidual_h[nDOF_test_element],
          elementResidual_hu[nDOF_test_element],
          elementResidual_hv[nDOF_test_element],
          elementResidual_heta[nDOF_test_element],
          elementResidual_hw[nDOF_test_element],
          elementResidual_hbeta[nDOF_test_element];

      for (int i = 0; i < nDOF_test_element; i++) {
        elementResidual_h[i] = 0.0;
        elementResidual_hu[i] = 0.0;
        elementResidual_hv[i] = 0.0;
        elementResidual_heta[i] = 0.0;
        elementResidual_hw[i] = 0.0;
        elementResidual_hbeta[i] = 0.0;
      }
      //
      // loop over quadrature points and compute integrands
      //
      for (int k = 0; k < nQuadraturePoints_element; k++) {
        // compute indices and declare local storage
        int eN_k = eN * nQuadraturePoints_element + k,
                     eN_k_nSpace = eN_k * nSpace,
                     eN_nDOF_trial_element = eN * nDOF_trial_element;
        double h = 0.0, hu = 0.0, hv = 0.0, heta = 0.0, hw = 0.0,
                        hbeta = 0.0, // solution at current time
            h_old = 0.0, hu_old = 0.0, hv_old = 0.0, heta_old = 0.0,
                        hw_old = 0.0,
                        hbeta_old = 0.0, // solution at lstage
            jac[nSpace * nSpace], jacDet, jacInv[nSpace * nSpace],
                        h_test_dV[nDOF_trial_element], dV, x, y, xt, yt;
        // get jacobian, etc for mapping reference element
        ck.calculateMapping_element(
            eN, k, mesh_dof.data(), mesh_l2g.data(), mesh_trial_ref.data(),
            mesh_grad_trial_ref.data(), jac, jacDet, jacInv, x, y);
        // get the physical integration weight
        dV = fabs(jacDet) * dV_ref[k];
        {
          // porosity weighting: storage is phi U
          double phi_q = 0.;
          ck.valFromDOF(porosity_dof.data(),
                        &h_l2g.data()[eN * nDOF_trial_element],
                        &h_trial_ref.data()[k * nDOF_trial_element], phi_q);
          dV *= phi_q;
        }
        // get the solution at current time
        ck.valFromDOF(h_dof.data(), &h_l2g.data()[eN_nDOF_trial_element],
                      &h_trial_ref.data()[k * nDOF_trial_element], h);
        ck.valFromDOF(hu_dof.data(), &vel_l2g[eN_nDOF_trial_element],
                      &vel_trial_ref.data()[k * nDOF_trial_element], hu);
        ck.valFromDOF(hv_dof.data(), &vel_l2g[eN_nDOF_trial_element],
                      &vel_trial_ref.data()[k * nDOF_trial_element], hv);
        ck.valFromDOF(heta_dof.data(), &vel_l2g.data()[eN_nDOF_trial_element],
                      &vel_trial_ref.data()[k * nDOF_trial_element], heta);
        ck.valFromDOF(hw_dof.data(), &vel_l2g.data()[eN_nDOF_trial_element],
                      &vel_trial_ref.data()[k * nDOF_trial_element], hw);
        ck.valFromDOF(hbeta_dof.data(), &vel_l2g.data()[eN_nDOF_trial_element],
                      &vel_trial_ref.data()[k * nDOF_trial_element], hbeta);
        // get the solution at the lstage
        ck.valFromDOF(h_dof_old.data(), &h_l2g.data()[eN_nDOF_trial_element],
                      &h_trial_ref.data()[k * nDOF_trial_element], h_old);
        ck.valFromDOF(hu_dof_old.data(), &vel_l2g.data()[eN_nDOF_trial_element],
                      &vel_trial_ref.data()[k * nDOF_trial_element], hu_old);
        ck.valFromDOF(hv_dof_old.data(), &vel_l2g.data()[eN_nDOF_trial_element],
                      &vel_trial_ref.data()[k * nDOF_trial_element], hv_old);
        ck.valFromDOF(heta_dof_old.data(), &vel_l2g[eN_nDOF_trial_element],
                      &vel_trial_ref.data()[k * nDOF_trial_element], heta_old);
        ck.valFromDOF(hw_dof_old.data(), &vel_l2g.data()[eN_nDOF_trial_element],
                      &vel_trial_ref.data()[k * nDOF_trial_element], hw_old);
        ck.valFromDOF(hbeta_dof_old.data(),
                      &vel_l2g.data()[eN_nDOF_trial_element],
                      &vel_trial_ref.data()[k * nDOF_trial_element], hbeta_old);
        // calculate cell based CFL to keep a reference
        calculateCFL(elementDiameter.data()[eN], g, h_old, hu_old, hv_old, hEps,
                     q_cfl[eN_k]);
        // precalculate test function products with integration weights
        for (int j = 0; j < nDOF_trial_element; j++)
          h_test_dV[j] = h_test_ref[k * nDOF_trial_element + j] * dV;

        // SAVE VELOCITY // at quadrature points for other models to use
        q_velocity[eN_k_nSpace + 0] =
            2 * h / (h * h + std::pow(fmax(h, hEps), 2)) * hu;
        q_velocity[eN_k_nSpace + 1] =
            2 * h / (h * h + std::pow(fmax(h, hEps), 2)) * hv;
        hnp1_at_quad_point[eN_k] = h;
        hunp1_at_quad_point[eN_k] = hu;
        hvnp1_at_quad_point[eN_k] = hv;
        hetanp1_at_quad_point[eN_k] = heta;
        hwnp1_at_quad_point[eN_k] = hw;
        hbetanp1_at_quad_point[eN_k] = hbeta;

        for (int i = 0; i < nDOF_test_element; i++) {
          // compute time derivative part of global residual. NOTE: no lumping
          elementResidual_h[i] += (h - h_old) * h_test_dV[i];
          elementResidual_hu[i] += (hu - hu_old) * h_test_dV[i];
          elementResidual_hv[i] += (hv - hv_old) * h_test_dV[i];
          elementResidual_heta[i] += (heta - heta_old) * h_test_dV[i];
          elementResidual_hw[i] += (hw - hw_old) * h_test_dV[i];
          elementResidual_hbeta[i] += (hbeta - hbeta_old) * h_test_dV[i];
        }
      }
      // distribute
      for (int i = 0; i < nDOF_test_element; i++) {
        int eN_i = eN * nDOF_test_element + i;

        // global i-th index for h (this is same for vel_l2g)
        int h_gi = h_l2g[eN_i];

        // distribute time derivative to global residual
        globalResidual[offset_h + stride_h * h_gi] += elementResidual_h[i];
        globalResidual[offset_hu + stride_hu * h_gi] += elementResidual_hu[i];
        globalResidual[offset_hv + stride_hv * h_gi] += elementResidual_hv[i];
        globalResidual[offset_heta + stride_heta * h_gi] +=
            elementResidual_heta[i];
        globalResidual[offset_hw + stride_hw * h_gi] += elementResidual_hw[i];
        globalResidual[offset_hbeta + stride_hbeta * h_gi] +=
            elementResidual_hbeta[i];
      }
    }
    // ********** END OF CELL LOOPS ********** //

    if (SECOND_CALL_CALCULATE_RESIDUAL == 0) // This is to save some time
    {
      ///////////////////////////////////////////////////////
      // ********** FIRST AND ONLY LOOP ON DOFs ********** //
      ///////////////////////////////////////////////////////
      // To compute:
      //     * High-order update with neumann series approximation

      int ij = 0;
      for (int i = 0; i < numDOFsPerEqn; i++) {

        // get things at ith node
        const double hi = h_dof_old[i];
        const double hui = hu_dof_old[i];
        const double hvi = hv_dof_old[i];
        const double hetai = heta_dof_old[i];
        const double hwi = hw_dof_old[i];
        const double hbetai = hbeta_dof_old[i];
        const double mi = lumped_mass_matrix[i];

        double sum_RHS_h = 0.;
        double sum_RHS_hu = 0.;
        double sum_RHS_hv = 0.;
        double sum_RHS_heta = 0.;
        double sum_RHS_hw = 0.;
        double sum_RHS_hbeta = 0.;

        double b_ij = 0., b_ji = 0.;

        // loop over the sparsity pattern of the i-th DOF
        for (int offset = csrRowIndeces_DofLoops[i];
             offset < csrRowIndeces_DofLoops[i + 1]; offset++) {

          int j = csrColumnOffsets_DofLoops[offset];

          // get things at jth node
          const double mj = lumped_mass_matrix[j];

          // define b_ij and b_ji
          if (j != i) {
            b_ij = (0. - MassMatrix[ij] / mj);
            b_ji = (0. - MassMatrix[ij] / mi);
          } else {
            b_ij = (1. - MassMatrix[ij] / mj);
            b_ji = (1. - MassMatrix[ij] / mi);
          }

          // define sum for high-order RHS
          sum_RHS_h += b_ij * RHS_high_h[j] - b_ji * RHS_high_h[i];
          sum_RHS_hu += b_ij * RHS_high_hu[j] - b_ji * RHS_high_hu[i];
          sum_RHS_hv += b_ij * RHS_high_hv[j] - b_ji * RHS_high_hv[i];
          sum_RHS_heta += b_ij * RHS_high_heta[j] - b_ji * RHS_high_heta[i];
          sum_RHS_hw += b_ij * RHS_high_hw[j] - b_ji * RHS_high_hw[i];
          sum_RHS_hbeta += b_ij * RHS_high_hbeta[j] - b_ji * RHS_high_hbeta[i];

          // update ij
          ij += 1;
        } // j loop ends here

        /* Define global residual */
        // if (LUMPED_MASS_MATRIX == 1) {
        globalResidual[offset_h + stride_h * i] =
            hi + dt / mi * (RHS_high_h[i] + sum_RHS_h);

        globalResidual[offset_hu + stride_hu * i] =
            hui + dt / mi * (RHS_high_hu[i] + sum_RHS_hu);

        globalResidual[offset_hv + stride_hv * i] =
            hvi + dt / mi * (RHS_high_hv[i] + sum_RHS_hv);

        globalResidual[offset_heta + stride_heta * i] =
            hetai + dt / mi * (RHS_high_heta[i] + sum_RHS_heta);

        globalResidual[offset_hw + stride_hw * i] =
            hwi + dt / mi * (RHS_high_hw[i] + sum_RHS_hw);

        globalResidual[offset_hbeta + stride_hbeta * i] =
            hbetai + dt / mi * (RHS_high_hbeta[i] + sum_RHS_hbeta);

        // clean up potential negative water height due to machine precision
        if (globalResidual[offset_h + stride_h * i] >= -hEps &&
            globalResidual[offset_h + stride_h * i] < hEps) {
          globalResidual[offset_h + stride_h * i] = 0.;
        }
      }
      // ********** END OF LOOP IN DOFs ********** //
    } // end SECOND_CALL_CALCULATE_RESIDUAL

    // ********** COMPUTE NORMALS ********** //
    if (COMPUTE_NORMALS == 1) {
      // This is to identify the normals and create a vector of normal
      // components
      for (int ebNE = 0; ebNE < nExteriorElementBoundaries_global; ebNE++) {
        int
            ebN = exteriorElementBoundariesArray[ebNE],
            eN = elementBoundaryElementsArray[ebN * 2 + 0],
            ebN_local = elementBoundaryLocalElementBoundariesArray[ebN * 2 + 0];
        double normal[3];
        {             // "Loop" in quad points
          int kb = 0; // NOTE: I need to consider just one quad point since
                      // the element is not curved so the normal is constant
                      // per element
          int ebN_local_kb =
              ebN_local * nQuadraturePoints_elementBoundary + kb;
          double jac_ext[nSpace * nSpace], jacDet_ext,
              jacInv_ext[nSpace * nSpace], boundaryJac[nSpace * (nSpace - 1)],
              metricTensor[(nSpace - 1) * (nSpace - 1)], metricTensorDetSqrt,
              x_ext, y_ext;
          /* compute information about mapping from reference element to
           * physical element */
          ck.calculateMapping_elementBoundary(
              eN, ebN_local, kb, ebN_local_kb, mesh_dof.data(), mesh_l2g.data(),
              mesh_trial_trace_ref.data(), mesh_grad_trial_trace_ref.data(),
              boundaryJac_ref.data(), jac_ext, jacDet_ext, jacInv_ext,
              boundaryJac, metricTensor, metricTensorDetSqrt, normal_ref.data(),
              normal, x_ext, y_ext);
        }
        // distribute the normal vectors
        for (int i = 0; i < nDOF_test_element; i++) {
          int eN_i = eN * nDOF_test_element + i;
          int gi = h_l2g[eN_i];
          normalx[gi] += 0.5 * normal[0] * (i == ebN_local ? 0. : 1.);
          normaly[gi] += 0.5 * normal[1] * (i == ebN_local ? 0. : 1.);
        }
      }
      // normalize
      for (int gi = 0; gi < numDOFsPerEqn; gi++) {
        double norm_factor =
            sqrt(std::pow(normalx[gi], 2) + std::pow(normaly[gi], 2));
        if (norm_factor != 0) {
          normalx[gi] /= norm_factor;
          normaly[gi] /= norm_factor;
        }
      }
    }
    // ********** END OF COMPUTING NORMALS ********** //
  } // end calculateResidual

  void calculateMassMatrix(arguments_dict &args) {
    xt::pyarray<double> &mesh_trial_ref = args.array<double>("mesh_trial_ref");
    xt::pyarray<double> &mesh_grad_trial_ref =
        args.array<double>("mesh_grad_trial_ref");
    xt::pyarray<double> &mesh_dof = args.array<double>("mesh_dof");
    xt::pyarray<double> &mesh_velocity_dof =
        args.array<double>("mesh_velocity_dof");
    double MOVING_DOMAIN = args.scalar<double>("MOVING_DOMAIN");
    xt::pyarray<int> &mesh_l2g = args.array<int>("mesh_l2g");
    xt::pyarray<double> &dV_ref = args.array<double>("dV_ref");
    xt::pyarray<double> &h_trial_ref = args.array<double>("h_trial_ref");
    xt::pyarray<double> &h_grad_trial_ref =
        args.array<double>("h_grad_trial_ref");
    xt::pyarray<double> &h_test_ref = args.array<double>("h_test_ref");
    xt::pyarray<double> &h_grad_test_ref =
        args.array<double>("h_grad_test_ref");
    xt::pyarray<double> &vel_trial_ref = args.array<double>("vel_trial_ref");
    xt::pyarray<double> &vel_grad_trial_ref =
        args.array<double>("vel_grad_trial_ref");
    xt::pyarray<double> &vel_test_ref = args.array<double>("vel_test_ref");
    xt::pyarray<double> &vel_grad_test_ref =
        args.array<double>("vel_grad_test_ref");
    xt::pyarray<double> &mesh_trial_trace_ref =
        args.array<double>("mesh_trial_trace_ref");
    xt::pyarray<double> &mesh_grad_trial_trace_ref =
        args.array<double>("mesh_grad_trial_trace_ref");
    xt::pyarray<double> &dS_ref = args.array<double>("dS_ref");
    xt::pyarray<double> &h_trial_trace_ref =
        args.array<double>("h_trial_trace_ref");
    xt::pyarray<double> &h_grad_trial_trace_ref =
        args.array<double>("h_grad_trial_trace_ref");
    xt::pyarray<double> &h_test_trace_ref =
        args.array<double>("h_test_trace_ref");
    xt::pyarray<double> &h_grad_test_trace_ref =
        args.array<double>("h_grad_test_trace_ref");
    xt::pyarray<double> &vel_trial_trace_ref =
        args.array<double>("vel_trial_trace_ref");
    xt::pyarray<double> &vel_grad_trial_trace_ref =
        args.array<double>("vel_grad_trial_trace_ref");
    xt::pyarray<double> &vel_test_trace_ref =
        args.array<double>("vel_test_trace_ref");
    xt::pyarray<double> &vel_grad_test_trace_ref =
        args.array<double>("vel_grad_test_trace_ref");
    xt::pyarray<double> &normal_ref = args.array<double>("normal_ref");
    xt::pyarray<double> &boundaryJac_ref =
        args.array<double>("boundaryJac_ref");
    xt::pyarray<double> &elementDiameter =
        args.array<double>("elementDiameter");
    int nElements_global = args.scalar<int>("nElements_global");
    double g = args.scalar<double>("g");
    xt::pyarray<int> &h_l2g = args.array<int>("h_l2g");
    xt::pyarray<int> &vel_l2g = args.array<int>("vel_l2g");
    xt::pyarray<double> &b_dof = args.array<double>("b_dof");
    xt::pyarray<double> &h_dof = args.array<double>("h_dof");
    xt::pyarray<double> &hu_dof = args.array<double>("hu_dof");
    xt::pyarray<double> &hv_dof = args.array<double>("hv_dof");
    xt::pyarray<double> &heta_dof = args.array<double>("heta_dof");
    xt::pyarray<double> &hw_dof = args.array<double>("hw_dof");
    xt::pyarray<double> &hbeta_dof = args.array<double>("hbeta_dof");
    xt::pyarray<double> &q_cfl = args.array<double>("q_cfl");
    xt::pyarray<int> &sdInfo_hu_hu_rowptr =
        args.array<int>("sdInfo_hu_hu_rowptr");
    xt::pyarray<int> &sdInfo_hu_hu_colind =
        args.array<int>("sdInfo_hu_hu_colind");
    xt::pyarray<int> &sdInfo_hu_hv_rowptr =
        args.array<int>("sdInfo_hu_hv_rowptr");
    xt::pyarray<int> &sdInfo_hu_hv_colind =
        args.array<int>("sdInfo_hu_hv_colind");
    xt::pyarray<int> &sdInfo_hv_hv_rowptr =
        args.array<int>("sdInfo_hv_hv_rowptr");
    xt::pyarray<int> &sdInfo_hv_hv_colind =
        args.array<int>("sdInfo_hv_hv_colind");
    xt::pyarray<int> &sdInfo_hv_hu_rowptr =
        args.array<int>("sdInfo_hv_hu_rowptr");
    xt::pyarray<int> &sdInfo_hv_hu_colind =
        args.array<int>("sdInfo_hv_hu_colind");
    xt::pyarray<int> &csrRowIndeces_h_h = args.array<int>("csrRowIndeces_h_h");
    xt::pyarray<int> &csrColumnOffsets_h_h =
        args.array<int>("csrColumnOffsets_h_h");
    xt::pyarray<int> &csrRowIndeces_h_hu =
        args.array<int>("csrRowIndeces_h_hu");
    xt::pyarray<int> &csrColumnOffsets_h_hu =
        args.array<int>("csrColumnOffsets_h_hu");
    xt::pyarray<int> &csrRowIndeces_h_hv =
        args.array<int>("csrRowIndeces_h_hv");
    xt::pyarray<int> &csrColumnOffsets_h_hv =
        args.array<int>("csrColumnOffsets_h_hv");
    xt::pyarray<int> &csrRowIndeces_h_heta =
        args.array<int>("csrRowIndeces_h_heta");
    xt::pyarray<int> &csrColumnOffsets_h_heta =
        args.array<int>("csrColumnOffsets_h_heta");
    xt::pyarray<int> &csrRowIndeces_h_hw =
        args.array<int>("csrRowIndeces_h_hw");
    xt::pyarray<int> &csrColumnOffsets_h_hw =
        args.array<int>("csrColumnOffsets_h_hw");
    xt::pyarray<int> &csrRowIndeces_h_hbeta =
        args.array<int>("csrRowIndeces_h_hbeta");
    xt::pyarray<int> &csrColumnOffsets_h_hbeta =
        args.array<int>("csrColumnOffsets_h_hbeta");
    xt::pyarray<int> &csrRowIndeces_hu_h =
        args.array<int>("csrRowIndeces_hu_h");
    xt::pyarray<int> &csrColumnOffsets_hu_h =
        args.array<int>("csrColumnOffsets_hu_h");
    xt::pyarray<int> &csrRowIndeces_hu_hu =
        args.array<int>("csrRowIndeces_hu_hu");
    xt::pyarray<int> &csrColumnOffsets_hu_hu =
        args.array<int>("csrColumnOffsets_hu_hu");
    xt::pyarray<int> &csrRowIndeces_hu_hv =
        args.array<int>("csrRowIndeces_hu_hv");
    xt::pyarray<int> &csrColumnOffsets_hu_hv =
        args.array<int>("csrColumnOffsets_hu_hv");
    xt::pyarray<int> &csrRowIndeces_hu_heta =
        args.array<int>("csrRowIndeces_hu_heta");
    xt::pyarray<int> &csrColumnOffsets_hu_heta =
        args.array<int>("csrColumnOffsets_hu_heta");
    xt::pyarray<int> &csrRowIndeces_hu_hw =
        args.array<int>("csrRowIndeces_hu_hw");
    xt::pyarray<int> &csrColumnOffsets_hu_hw =
        args.array<int>("csrColumnOffsets_hu_hw");
    xt::pyarray<int> &csrRowIndeces_hu_hbeta =
        args.array<int>("csrRowIndeces_hu_hbeta");
    xt::pyarray<int> &csrColumnOffsets_hu_hbeta =
        args.array<int>("csrColumnOffsets_hu_hbeta");
    xt::pyarray<int> &csrRowIndeces_hv_h =
        args.array<int>("csrRowIndeces_hv_h");
    xt::pyarray<int> &csrColumnOffsets_hv_h =
        args.array<int>("csrColumnOffsets_hv_h");
    xt::pyarray<int> &csrRowIndeces_hv_hu =
        args.array<int>("csrRowIndeces_hv_hu");
    xt::pyarray<int> &csrColumnOffsets_hv_hu =
        args.array<int>("csrColumnOffsets_hv_hu");
    xt::pyarray<int> &csrRowIndeces_hv_hv =
        args.array<int>("csrRowIndeces_hv_hv");
    xt::pyarray<int> &csrColumnOffsets_hv_hv =
        args.array<int>("csrColumnOffsets_hv_hv");
    xt::pyarray<int> &csrRowIndeces_hv_heta =
        args.array<int>("csrRowIndeces_hv_heta");
    xt::pyarray<int> &csrColumnOffsets_hv_heta =
        args.array<int>("csrColumnOffsets_hv_heta");
    xt::pyarray<int> &csrRowIndeces_hv_hw =
        args.array<int>("csrRowIndeces_hv_hw");
    xt::pyarray<int> &csrColumnOffsets_hv_hw =
        args.array<int>("csrColumnOffsets_hv_hw");
    xt::pyarray<int> &csrRowIndeces_hv_hbeta =
        args.array<int>("csrRowIndeces_hv_hbeta");
    xt::pyarray<int> &csrColumnOffsets_hv_hbeta =
        args.array<int>("csrColumnOffsets_hv_hbeta");
    xt::pyarray<int> &csrRowIndeces_heta_h =
        args.array<int>("csrRowIndeces_heta_h");
    xt::pyarray<int> &csrColumnOffsets_heta_h =
        args.array<int>("csrColumnOffsets_heta_h");
    xt::pyarray<int> &csrRowIndeces_heta_hu =
        args.array<int>("csrRowIndeces_heta_hu");
    xt::pyarray<int> &csrColumnOffsets_heta_hu =
        args.array<int>("csrColumnOffsets_heta_hu");
    xt::pyarray<int> &csrRowIndeces_heta_hv =
        args.array<int>("csrRowIndeces_heta_hv");
    xt::pyarray<int> &csrColumnOffsets_heta_hv =
        args.array<int>("csrColumnOffsets_heta_hv");
    xt::pyarray<int> &csrRowIndeces_heta_heta =
        args.array<int>("csrRowIndeces_heta_heta");
    xt::pyarray<int> &csrColumnOffsets_heta_heta =
        args.array<int>("csrColumnOffsets_heta_heta");
    xt::pyarray<int> &csrRowIndeces_heta_hw =
        args.array<int>("csrRowIndeces_heta_hw");
    xt::pyarray<int> &csrColumnOffsets_heta_hw =
        args.array<int>("csrColumnOffsets_heta_hw");
    xt::pyarray<int> &csrRowIndeces_heta_hbeta =
        args.array<int>("csrRowIndeces_heta_hbeta");
    xt::pyarray<int> &csrColumnOffsets_heta_hbeta =
        args.array<int>("csrColumnOffsets_heta_hbeta");
    xt::pyarray<int> &csrRowIndeces_hw_h =
        args.array<int>("csrRowIndeces_hw_h");
    xt::pyarray<int> &csrColumnOffsets_hw_h =
        args.array<int>("csrColumnOffsets_hw_h");
    xt::pyarray<int> &csrRowIndeces_hw_hu =
        args.array<int>("csrRowIndeces_hw_hu");
    xt::pyarray<int> &csrColumnOffsets_hw_hu =
        args.array<int>("csrColumnOffsets_hw_hu");
    xt::pyarray<int> &csrRowIndeces_hw_hv =
        args.array<int>("csrRowIndeces_hw_hv");
    xt::pyarray<int> &csrColumnOffsets_hw_hv =
        args.array<int>("csrColumnOffsets_hw_hv");
    xt::pyarray<int> &csrRowIndeces_hw_heta =
        args.array<int>("csrRowIndeces_hw_heta");
    xt::pyarray<int> &csrColumnOffsets_hw_heta =
        args.array<int>("csrColumnOffsets_hw_heta");
    xt::pyarray<int> &csrRowIndeces_hw_hw =
        args.array<int>("csrRowIndeces_hw_hw");
    xt::pyarray<int> &csrColumnOffsets_hw_hw =
        args.array<int>("csrColumnOffsets_hw_hw");
    xt::pyarray<int> &csrRowIndeces_hw_hbeta =
        args.array<int>("csrRowIndeces_hw_hbeta");
    xt::pyarray<int> &csrColumnOffsets_hw_hbeta =
        args.array<int>("csrColumnOffsets_hw_hbeta");
    //
    xt::pyarray<int> &csrRowIndeces_hbeta_h =
        args.array<int>("csrRowIndeces_hbeta_h");
    xt::pyarray<int> &csrColumnOffsets_hbeta_h =
        args.array<int>("csrColumnOffsets_hbeta_h");
    xt::pyarray<int> &csrRowIndeces_hbeta_hu =
        args.array<int>("csrRowIndeces_hbeta_hu");
    xt::pyarray<int> &csrColumnOffsets_hbeta_hu =
        args.array<int>("csrColumnOffsets_hbeta_hu");
    xt::pyarray<int> &csrRowIndeces_hbeta_hv =
        args.array<int>("csrRowIndeces_hbeta_hv");
    xt::pyarray<int> &csrColumnOffsets_hbeta_hv =
        args.array<int>("csrColumnOffsets_hbeta_hv");
    xt::pyarray<int> &csrRowIndeces_hbeta_heta =
        args.array<int>("csrRowIndeces_hbeta_heta");
    xt::pyarray<int> &csrColumnOffsets_hbeta_heta =
        args.array<int>("csrColumnOffsets_hbeta_heta");
    xt::pyarray<int> &csrRowIndeces_hbeta_hw =
        args.array<int>("csrRowIndeces_hbeta_hw");
    xt::pyarray<int> &csrColumnOffsets_hbeta_hw =
        args.array<int>("csrColumnOffsets_hbeta_hw");
    xt::pyarray<int> &csrRowIndeces_hbeta_hbeta =
        args.array<int>("csrRowIndeces_hbeta_hbeta");
    xt::pyarray<int> &csrColumnOffsets_hbeta_hbeta =
        args.array<int>("csrColumnOffsets_hbeta_hbeta");
    xt::pyarray<double> &globalJacobian = args.array<double>("globalJacobian");
    xt::pyarray<double> &porosity_dof = args.array<double>("porosity_dof");
    int nExteriorElementBoundaries_global =
        args.scalar<int>("nExteriorElementBoundaries_global");
    xt::pyarray<int> &exteriorElementBoundariesArray =
        args.array<int>("exteriorElementBoundariesArray");
    xt::pyarray<int> &elementBoundaryElementsArray =
        args.array<int>("elementBoundaryElementsArray");
    xt::pyarray<int> &elementBoundaryLocalElementBoundariesArray =
        args.array<int>("elementBoundaryLocalElementBoundariesArray");
    xt::pyarray<int> &isDOFBoundary_h = args.array<int>("isDOFBoundary_h");
    xt::pyarray<int> &isDOFBoundary_hu = args.array<int>("isDOFBoundary_hu");
    xt::pyarray<int> &isDOFBoundary_hv = args.array<int>("isDOFBoundary_hv");
    xt::pyarray<int> &isAdvectiveFluxBoundary_h =
        args.array<int>("isAdvectiveFluxBoundary_h");
    xt::pyarray<int> &isAdvectiveFluxBoundary_hu =
        args.array<int>("isAdvectiveFluxBoundary_hu");
    xt::pyarray<int> &isAdvectiveFluxBoundary_hv =
        args.array<int>("isAdvectiveFluxBoundary_hv");
    xt::pyarray<int> &isDiffusiveFluxBoundary_hu =
        args.array<int>("isDiffusiveFluxBoundary_hu");
    xt::pyarray<int> &isDiffusiveFluxBoundary_hv =
        args.array<int>("isDiffusiveFluxBoundary_hv");
    xt::pyarray<double> &ebqe_bc_h_ext = args.array<double>("ebqe_bc_h_ext");
    xt::pyarray<double> &ebqe_bc_flux_mass_ext =
        args.array<double>("ebqe_bc_flux_mass_ext");
    xt::pyarray<double> &ebqe_bc_flux_mom_hu_adv_ext =
        args.array<double>("ebqe_bc_flux_mom_hu_adv_ext");
    xt::pyarray<double> &ebqe_bc_flux_mom_hv_adv_ext =
        args.array<double>("ebqe_bc_flux_mom_hv_adv_ext");
    xt::pyarray<double> &ebqe_bc_hu_ext = args.array<double>("ebqe_bc_hu_ext");
    xt::pyarray<double> &ebqe_bc_flux_hu_diff_ext =
        args.array<double>("ebqe_bc_flux_hu_diff_ext");
    xt::pyarray<double> &ebqe_penalty_ext =
        args.array<double>("ebqe_penalty_ext");
    xt::pyarray<double> &ebqe_bc_hv_ext = args.array<double>("ebqe_bc_hv_ext");
    xt::pyarray<double> &ebqe_bc_flux_hv_diff_ext =
        args.array<double>("ebqe_bc_flux_hv_diff_ext");
    xt::pyarray<int> &csrColumnOffsets_eb_h_h =
        args.array<int>("csrColumnOffsets_eb_h_h");
    xt::pyarray<int> &csrColumnOffsets_eb_h_hu =
        args.array<int>("csrColumnOffsets_eb_h_hu");
    xt::pyarray<int> &csrColumnOffsets_eb_h_hv =
        args.array<int>("csrColumnOffsets_eb_h_hv");
    xt::pyarray<int> &csrColumnOffsets_eb_hu_h =
        args.array<int>("csrColumnOffsets_eb_hu_h");
    xt::pyarray<int> &csrColumnOffsets_eb_hu_hu =
        args.array<int>("csrColumnOffsets_eb_hu_hu");
    xt::pyarray<int> &csrColumnOffsets_eb_hu_hv =
        args.array<int>("csrColumnOffsets_eb_hu_hv");
    xt::pyarray<int> &csrColumnOffsets_eb_hv_h =
        args.array<int>("csrColumnOffsets_eb_hv_h");
    xt::pyarray<int> &csrColumnOffsets_eb_hv_hu =
        args.array<int>("csrColumnOffsets_eb_hv_hu");
    xt::pyarray<int> &csrColumnOffsets_eb_hv_hv =
        args.array<int>("csrColumnOffsets_eb_hv_hv");
    double dt = args.scalar<double>("dt");
    //
    // loop over elements to compute volume integrals and load them into the
    // element Jacobians and global Jacobian
    //
    for (int eN = 0; eN < nElements_global; eN++) {
      double elementJacobian_h_h[nDOF_test_element]
                                         [nDOF_trial_element],
          elementJacobian_hu_hu[nDOF_test_element][nDOF_trial_element],
          elementJacobian_hv_hv[nDOF_test_element][nDOF_trial_element],
          elementJacobian_heta_heta[nDOF_test_element][nDOF_trial_element],
          elementJacobian_hw_hw[nDOF_test_element][nDOF_trial_element],
          elementJacobian_hbeta_hbeta[nDOF_test_element][nDOF_trial_element];
      for (int i = 0; i < nDOF_test_element; i++)
        for (int j = 0; j < nDOF_trial_element; j++) {
          elementJacobian_h_h[i][j] = 0.0;
          elementJacobian_hu_hu[i][j] = 0.0;
          elementJacobian_hv_hv[i][j] = 0.0;
          elementJacobian_heta_heta[i][j] = 0.0;
          elementJacobian_hw_hw[i][j] = 0.0;
          elementJacobian_hbeta_hbeta[i][j] = 0.0;
        }
      for (int k = 0; k < nQuadraturePoints_element; k++) {
        int eN_k = eN * nQuadraturePoints_element +
                   k, // index to a scalar at a quadrature point
            eN_k_nSpace = eN_k * nSpace,
            eN_nDOF_trial_element =
                eN * nDOF_trial_element; // index to a vector at a
                                         // quadrature point

        // declare local storage
        double jac[nSpace * nSpace], jacDet, jacInv[nSpace * nSpace],
            dV, h_test_dV[nDOF_test_element], vel_test_dV[nDOF_test_element], x,
            y, xt, yt;
        // get jacobian, etc for mapping reference element
        ck.calculateMapping_element(
            eN, k, mesh_dof.data(), mesh_l2g.data(), mesh_trial_ref.data(),
            mesh_grad_trial_ref.data(), jac, jacDet, jacInv, x, y);
        // get the physical integration weight
        dV = fabs(jacDet) * dV_ref[k];
        {
          // porosity weighting: storage is phi U
          double phi_q = 0.;
          ck.valFromDOF(porosity_dof.data(),
                        &h_l2g.data()[eN * nDOF_trial_element],
                        &h_trial_ref.data()[k * nDOF_trial_element], phi_q);
          dV *= phi_q;
        }
        // precalculate test function products with integration weights
        for (int j = 0; j < nDOF_trial_element; j++) {
          h_test_dV[j] = h_test_ref[k * nDOF_trial_element + j] * dV;
          vel_test_dV[j] = vel_test_ref[k * nDOF_trial_element + j] * dV;
        }
        for (int i = 0; i < nDOF_test_element; i++) {
          int i_nSpace = i * nSpace;
          for (int j = 0; j < nDOF_trial_element; j++) {
            int j_nSpace = j * nSpace;
            elementJacobian_h_h[i][j] +=
                h_trial_ref[k * nDOF_trial_element + j] * h_test_dV[i];
            elementJacobian_hu_hu[i][j] +=
                vel_trial_ref[k * nDOF_trial_element + j] * vel_test_dV[i];
            elementJacobian_hv_hv[i][j] +=
                vel_trial_ref[k * nDOF_trial_element + j] * vel_test_dV[i];
            elementJacobian_heta_heta[i][j] +=
                vel_trial_ref[k * nDOF_trial_element + j] * vel_test_dV[i];
            elementJacobian_hw_hw[i][j] +=
                vel_trial_ref[k * nDOF_trial_element + j] * vel_test_dV[i];
            elementJacobian_hbeta_hbeta[i][j] +=
                vel_trial_ref[k * nDOF_trial_element + j] * vel_test_dV[i];
          } // j
        }   // i
      }     // k
      //
      // load into element Jacobian into global Jacobian
      //
      for (int i = 0; i < nDOF_test_element; i++) {
        int eN_i = eN * nDOF_test_element + i;
        for (int j = 0; j < nDOF_trial_element; j++) {
          int eN_i_j = eN_i * nDOF_trial_element + j;
          globalJacobian[csrRowIndeces_h_h[eN_i] +
                         csrColumnOffsets_h_h[eN_i_j]] +=
              elementJacobian_h_h[i][j];
          globalJacobian[csrRowIndeces_hu_hu[eN_i] +
                         csrColumnOffsets_hu_hu[eN_i_j]] +=
              elementJacobian_hu_hu[i][j];
          globalJacobian[csrRowIndeces_hv_hv[eN_i] +
                         csrColumnOffsets_hv_hv[eN_i_j]] +=
              elementJacobian_hv_hv[i][j];
          globalJacobian[csrRowIndeces_heta_heta[eN_i] +
                         csrColumnOffsets_heta_heta[eN_i_j]] +=
              elementJacobian_heta_heta[i][j];
          globalJacobian[csrRowIndeces_hw_hw[eN_i] +
                         csrColumnOffsets_hw_hw[eN_i_j]] +=
              elementJacobian_hw_hw[i][j];
          globalJacobian[csrRowIndeces_hbeta_hbeta[eN_i] +
                         csrColumnOffsets_hbeta_hbeta[eN_i_j]] +=
              elementJacobian_hbeta_hbeta[i][j];
        } // j
      }   // i
    }     // elements
  }

  void calculateLumpedMassMatrix(arguments_dict &args) {
    xt::pyarray<double> &mesh_trial_ref = args.array<double>("mesh_trial_ref");
    xt::pyarray<double> &mesh_grad_trial_ref =
        args.array<double>("mesh_grad_trial_ref");
    xt::pyarray<double> &mesh_dof = args.array<double>("mesh_dof");
    xt::pyarray<double> &mesh_velocity_dof =
        args.array<double>("mesh_velocity_dof");
    double MOVING_DOMAIN = args.scalar<double>("MOVING_DOMAIN");
    xt::pyarray<int> &mesh_l2g = args.array<int>("mesh_l2g");
    xt::pyarray<double> &dV_ref = args.array<double>("dV_ref");
    xt::pyarray<double> &h_trial_ref = args.array<double>("h_trial_ref");
    xt::pyarray<double> &h_grad_trial_ref =
        args.array<double>("h_grad_trial_ref");
    xt::pyarray<double> &h_test_ref = args.array<double>("h_test_ref");
    xt::pyarray<double> &h_grad_test_ref =
        args.array<double>("h_grad_test_ref");
    xt::pyarray<double> &vel_trial_ref = args.array<double>("vel_trial_ref");
    xt::pyarray<double> &vel_grad_trial_ref =
        args.array<double>("vel_grad_trial_ref");
    xt::pyarray<double> &vel_test_ref = args.array<double>("vel_test_ref");
    xt::pyarray<double> &vel_grad_test_ref =
        args.array<double>("vel_grad_test_ref");
    xt::pyarray<double> &mesh_trial_trace_ref =
        args.array<double>("mesh_trial_trace_ref");
    xt::pyarray<double> &mesh_grad_trial_trace_ref =
        args.array<double>("mesh_grad_trial_trace_ref");
    xt::pyarray<double> &dS_ref = args.array<double>("dS_ref");
    xt::pyarray<double> &h_trial_trace_ref =
        args.array<double>("h_trial_trace_ref");
    xt::pyarray<double> &h_grad_trial_trace_ref =
        args.array<double>("h_grad_trial_trace_ref");
    xt::pyarray<double> &h_test_trace_ref =
        args.array<double>("h_test_trace_ref");
    xt::pyarray<double> &h_grad_test_trace_ref =
        args.array<double>("h_grad_test_trace_ref");
    xt::pyarray<double> &vel_trial_trace_ref =
        args.array<double>("vel_trial_trace_ref");
    xt::pyarray<double> &vel_grad_trial_trace_ref =
        args.array<double>("vel_grad_trial_trace_ref");
    xt::pyarray<double> &vel_test_trace_ref =
        args.array<double>("vel_test_trace_ref");
    xt::pyarray<double> &vel_grad_test_trace_ref =
        args.array<double>("vel_grad_test_trace_ref");
    xt::pyarray<double> &normal_ref = args.array<double>("normal_ref");
    xt::pyarray<double> &boundaryJac_ref =
        args.array<double>("boundaryJac_ref");
    xt::pyarray<double> &elementDiameter =
        args.array<double>("elementDiameter");
    int nElements_global = args.scalar<int>("nElements_global");
    double g = args.scalar<double>("g");
    xt::pyarray<int> &h_l2g = args.array<int>("h_l2g");
    xt::pyarray<int> &vel_l2g = args.array<int>("vel_l2g");
    xt::pyarray<double> &b_dof = args.array<double>("b_dof");
    xt::pyarray<double> &h_dof = args.array<double>("h_dof");
    xt::pyarray<double> &hu_dof = args.array<double>("hu_dof");
    xt::pyarray<double> &hv_dof = args.array<double>("hv_dof");
    xt::pyarray<double> &q_cfl = args.array<double>("q_cfl");
    xt::pyarray<int> &sdInfo_hu_hu_rowptr =
        args.array<int>("sdInfo_hu_hu_rowptr");
    xt::pyarray<int> &sdInfo_hu_hu_colind =
        args.array<int>("sdInfo_hu_hu_colind");
    xt::pyarray<int> &sdInfo_hu_hv_rowptr =
        args.array<int>("sdInfo_hu_hv_rowptr");
    xt::pyarray<int> &sdInfo_hu_hv_colind =
        args.array<int>("sdInfo_hu_hv_colind");
    xt::pyarray<int> &sdInfo_hv_hv_rowptr =
        args.array<int>("sdInfo_hv_hv_rowptr");
    xt::pyarray<int> &sdInfo_hv_hv_colind =
        args.array<int>("sdInfo_hv_hv_colind");
    xt::pyarray<int> &sdInfo_hv_hu_rowptr =
        args.array<int>("sdInfo_hv_hu_rowptr");
    xt::pyarray<int> &sdInfo_hv_hu_colind =
        args.array<int>("sdInfo_hv_hu_colind");
    // h
    xt::pyarray<int> &csrRowIndeces_h_h = args.array<int>("csrRowIndeces_h_h");
    xt::pyarray<int> &csrColumnOffsets_h_h =
        args.array<int>("csrColumnOffsets_h_h");
    xt::pyarray<int> &csrRowIndeces_h_hu =
        args.array<int>("csrRowIndeces_h_hu");
    xt::pyarray<int> &csrColumnOffsets_h_hu =
        args.array<int>("csrColumnOffsets_h_hu");
    xt::pyarray<int> &csrRowIndeces_h_hv =
        args.array<int>("csrRowIndeces_h_hv");
    xt::pyarray<int> &csrColumnOffsets_h_hv =
        args.array<int>("csrColumnOffsets_h_hv");
    xt::pyarray<int> &csrRowIndeces_h_heta =
        args.array<int>("csrRowIndeces_h_heta");
    xt::pyarray<int> &csrColumnOffsets_h_heta =
        args.array<int>("csrColumnOffsets_h_heta");
    xt::pyarray<int> &csrRowIndeces_h_hw =
        args.array<int>("csrRowIndeces_h_hw");
    xt::pyarray<int> &csrColumnOffsets_h_hw =
        args.array<int>("csrColumnOffsets_h_hw");
    // hu
    xt::pyarray<int> &csrRowIndeces_hu_h =
        args.array<int>("csrRowIndeces_hu_h");
    xt::pyarray<int> &csrColumnOffsets_hu_h =
        args.array<int>("csrColumnOffsets_hu_h");
    xt::pyarray<int> &csrRowIndeces_hu_hu =
        args.array<int>("csrRowIndeces_hu_hu");
    xt::pyarray<int> &csrColumnOffsets_hu_hu =
        args.array<int>("csrColumnOffsets_hu_hu");
    xt::pyarray<int> &csrRowIndeces_hu_hv =
        args.array<int>("csrRowIndeces_hu_hv");
    xt::pyarray<int> &csrColumnOffsets_hu_hv =
        args.array<int>("csrColumnOffsets_hu_hv");
    xt::pyarray<int> &csrRowIndeces_hu_heta =
        args.array<int>("csrRowIndeces_hu_heta");
    xt::pyarray<int> &csrColumnOffsets_hu_heta =
        args.array<int>("csrColumnOffsets_hu_heta");
    xt::pyarray<int> &csrRowIndeces_hu_hw =
        args.array<int>("csrRowIndeces_hu_hw");
    xt::pyarray<int> &csrColumnOffsets_hu_hw =
        args.array<int>("csrColumnOffsets_hu_hw");
    // hv
    xt::pyarray<int> &csrRowIndeces_hv_h =
        args.array<int>("csrRowIndeces_hv_h");
    xt::pyarray<int> &csrColumnOffsets_hv_h =
        args.array<int>("csrColumnOffsets_hv_h");
    xt::pyarray<int> &csrRowIndeces_hv_hu =
        args.array<int>("csrRowIndeces_hv_hu");
    xt::pyarray<int> &csrColumnOffsets_hv_hu =
        args.array<int>("csrColumnOffsets_hv_hu");
    xt::pyarray<int> &csrRowIndeces_hv_hv =
        args.array<int>("csrRowIndeces_hv_hv");
    xt::pyarray<int> &csrColumnOffsets_hv_hv =
        args.array<int>("csrColumnOffsets_hv_hv");
    xt::pyarray<int> &csrRowIndeces_hv_heta =
        args.array<int>("csrRowIndeces_hv_heta");
    xt::pyarray<int> &csrColumnOffsets_hv_heta =
        args.array<int>("csrColumnOffsets_hv_heta");
    xt::pyarray<int> &csrRowIndeces_hv_hw =
        args.array<int>("csrRowIndeces_hv_hw");
    xt::pyarray<int> &csrColumnOffsets_hv_hw =
        args.array<int>("csrColumnOffsets_hv_hw");
    // heta
    xt::pyarray<int> &csrRowIndeces_heta_h =
        args.array<int>("csrRowIndeces_heta_h");
    xt::pyarray<int> &csrColumnOffsets_heta_h =
        args.array<int>("csrColumnOffsets_heta_h");
    xt::pyarray<int> &csrRowIndeces_heta_hu =
        args.array<int>("csrRowIndeces_heta_hu");
    xt::pyarray<int> &csrColumnOffsets_heta_hu =
        args.array<int>("csrColumnOffsets_heta_hu");
    xt::pyarray<int> &csrRowIndeces_heta_hv =
        args.array<int>("csrRowIndeces_heta_hv");
    xt::pyarray<int> &csrColumnOffsets_heta_hv =
        args.array<int>("csrColumnOffsets_heta_hv");
    xt::pyarray<int> &csrRowIndeces_heta_heta =
        args.array<int>("csrRowIndeces_heta_heta");
    xt::pyarray<int> &csrColumnOffsets_heta_heta =
        args.array<int>("csrColumnOffsets_heta_heta");
    xt::pyarray<int> &csrRowIndeces_heta_hw =
        args.array<int>("csrRowIndeces_heta_hw");
    xt::pyarray<int> &csrColumnOffsets_heta_hw =
        args.array<int>("csrColumnOffsets_heta_hw");
    // hw
    xt::pyarray<int> &csrRowIndeces_hw_h =
        args.array<int>("csrRowIndeces_hw_h");
    xt::pyarray<int> &csrColumnOffsets_hw_h =
        args.array<int>("csrColumnOffsets_hw_h");
    xt::pyarray<int> &csrRowIndeces_hw_hu =
        args.array<int>("csrRowIndeces_hw_hu");
    xt::pyarray<int> &csrColumnOffsets_hw_hu =
        args.array<int>("csrColumnOffsets_hw_hu");
    xt::pyarray<int> &csrRowIndeces_hw_hv =
        args.array<int>("csrRowIndeces_hw_hv");
    xt::pyarray<int> &csrColumnOffsets_hw_hv =
        args.array<int>("csrColumnOffsets_hw_hv");
    xt::pyarray<int> &csrRowIndeces_hw_heta =
        args.array<int>("csrRowIndeces_hw_heta");
    xt::pyarray<int> &csrColumnOffsets_hw_heta =
        args.array<int>("csrColumnOffsets_hw_heta");
    xt::pyarray<int> &csrRowIndeces_hw_hw =
        args.array<int>("csrRowIndeces_hw_hw");
    xt::pyarray<int> &csrColumnOffsets_hw_hw =
        args.array<int>("csrColumnOffsets_hw_hw");
    xt::pyarray<int> &csrRowIndeces_hbeta_hbeta =
        args.array<int>("csrRowIndeces_hbeta_hbeta");
    xt::pyarray<int> &csrColumnOffsets_hbeta_hbeta =
        args.array<int>("csrColumnOffsets_hbeta_hbeta");
    xt::pyarray<double> &globalJacobian = args.array<double>("globalJacobian");
    xt::pyarray<double> &porosity_dof = args.array<double>("porosity_dof");
    int nExteriorElementBoundaries_global =
        args.scalar<int>("nExteriorElementBoundaries_global");
    xt::pyarray<int> &exteriorElementBoundariesArray =
        args.array<int>("exteriorElementBoundariesArray");
    xt::pyarray<int> &elementBoundaryElementsArray =
        args.array<int>("elementBoundaryElementsArray");
    xt::pyarray<int> &elementBoundaryLocalElementBoundariesArray =
        args.array<int>("elementBoundaryLocalElementBoundariesArray");
    xt::pyarray<int> &isDOFBoundary_h = args.array<int>("isDOFBoundary_h");
    xt::pyarray<int> &isDOFBoundary_hu = args.array<int>("isDOFBoundary_hu");
    xt::pyarray<int> &isDOFBoundary_hv = args.array<int>("isDOFBoundary_hv");
    xt::pyarray<int> &isAdvectiveFluxBoundary_h =
        args.array<int>("isAdvectiveFluxBoundary_h");
    xt::pyarray<int> &isAdvectiveFluxBoundary_hu =
        args.array<int>("isAdvectiveFluxBoundary_hu");
    xt::pyarray<int> &isAdvectiveFluxBoundary_hv =
        args.array<int>("isAdvectiveFluxBoundary_hv");
    xt::pyarray<int> &isDiffusiveFluxBoundary_hu =
        args.array<int>("isDiffusiveFluxBoundary_hu");
    xt::pyarray<int> &isDiffusiveFluxBoundary_hv =
        args.array<int>("isDiffusiveFluxBoundary_hv");
    xt::pyarray<double> &ebqe_bc_h_ext = args.array<double>("ebqe_bc_h_ext");
    xt::pyarray<double> &ebqe_bc_flux_mass_ext =
        args.array<double>("ebqe_bc_flux_mass_ext");
    xt::pyarray<double> &ebqe_bc_flux_mom_hu_adv_ext =
        args.array<double>("ebqe_bc_flux_mom_hu_adv_ext");
    xt::pyarray<double> &ebqe_bc_flux_mom_hv_adv_ext =
        args.array<double>("ebqe_bc_flux_mom_hv_adv_ext");
    xt::pyarray<double> &ebqe_bc_hu_ext = args.array<double>("ebqe_bc_hu_ext");
    xt::pyarray<double> &ebqe_bc_flux_hu_diff_ext =
        args.array<double>("ebqe_bc_flux_hu_diff_ext");
    xt::pyarray<double> &ebqe_penalty_ext =
        args.array<double>("ebqe_penalty_ext");
    xt::pyarray<double> &ebqe_bc_hv_ext = args.array<double>("ebqe_bc_hv_ext");
    xt::pyarray<double> &ebqe_bc_flux_hv_diff_ext =
        args.array<double>("ebqe_bc_flux_hv_diff_ext");
    xt::pyarray<int> &csrColumnOffsets_eb_h_h =
        args.array<int>("csrColumnOffsets_eb_h_h");
    xt::pyarray<int> &csrColumnOffsets_eb_h_hu =
        args.array<int>("csrColumnOffsets_eb_h_hu");
    xt::pyarray<int> &csrColumnOffsets_eb_h_hv =
        args.array<int>("csrColumnOffsets_eb_h_hv");
    xt::pyarray<int> &csrColumnOffsets_eb_hu_h =
        args.array<int>("csrColumnOffsets_eb_hu_h");
    xt::pyarray<int> &csrColumnOffsets_eb_hu_hu =
        args.array<int>("csrColumnOffsets_eb_hu_hu");
    xt::pyarray<int> &csrColumnOffsets_eb_hu_hv =
        args.array<int>("csrColumnOffsets_eb_hu_hv");
    xt::pyarray<int> &csrColumnOffsets_eb_hv_h =
        args.array<int>("csrColumnOffsets_eb_hv_h");
    xt::pyarray<int> &csrColumnOffsets_eb_hv_hu =
        args.array<int>("csrColumnOffsets_eb_hv_hu");
    xt::pyarray<int> &csrColumnOffsets_eb_hv_hv =
        args.array<int>("csrColumnOffsets_eb_hv_hv");
    double dt = args.scalar<double>("dt");
    //
    // loop over elements to compute volume integrals and load them into the
    // element Jacobians and global Jacobian
    //
    for (int eN = 0; eN < nElements_global; eN++) {
      double elementJacobian_h_h[nDOF_test_element]
                                         [nDOF_trial_element],
          elementJacobian_hu_hu[nDOF_test_element][nDOF_trial_element],
          elementJacobian_hv_hv[nDOF_test_element][nDOF_trial_element],
          elementJacobian_heta_heta[nDOF_test_element][nDOF_trial_element],
          elementJacobian_hw_hw[nDOF_test_element][nDOF_trial_element],
          elementJacobian_hbeta_hbeta[nDOF_test_element][nDOF_trial_element];
      for (int i = 0; i < nDOF_test_element; i++)
        for (int j = 0; j < nDOF_trial_element; j++) {
          elementJacobian_h_h[i][j] = 0.0;
          elementJacobian_hu_hu[i][j] = 0.0;
          elementJacobian_hv_hv[i][j] = 0.0;
          elementJacobian_heta_heta[i][j] = 0.0;
          elementJacobian_hw_hw[i][j] = 0.0;
          elementJacobian_hbeta_hbeta[i][j] = 0.0;
        }
      for (int k = 0; k < nQuadraturePoints_element; k++) {
        int eN_k = eN * nQuadraturePoints_element +
                   k, // index to a scalar at a quadrature point
            eN_k_nSpace = eN_k * nSpace,
            eN_nDOF_trial_element =
                eN * nDOF_trial_element; // index to a vector at a
                                         // quadrature point

        // declare local storage
        double jac[nSpace * nSpace], jacDet, jacInv[nSpace * nSpace],
            dV, h_test_dV[nDOF_test_element], vel_test_dV[nDOF_test_element], x,
            y, xt, yt;
        // get jacobian, etc for mapping reference element
        ck.calculateMapping_element(
            eN, k, mesh_dof.data(), mesh_l2g.data(), mesh_trial_ref.data(),
            mesh_grad_trial_ref.data(), jac, jacDet, jacInv, x, y);
        // get the physical integration weight
        dV = fabs(jacDet) * dV_ref[k];
        {
          // porosity weighting: storage is phi U
          double phi_q = 0.;
          ck.valFromDOF(porosity_dof.data(),
                        &h_l2g.data()[eN * nDOF_trial_element],
                        &h_trial_ref.data()[k * nDOF_trial_element], phi_q);
          dV *= phi_q;
        }
        // precalculate test function products with integration weights
        for (int j = 0; j < nDOF_trial_element; j++) {
          h_test_dV[j] = h_test_ref[k * nDOF_trial_element + j] * dV;
          vel_test_dV[j] = vel_test_ref[k * nDOF_trial_element + j] * dV;
        }

        for (int i = 0; i < nDOF_test_element; i++) {
          int i_nSpace = i * nSpace;
          for (int j = 0; j < nDOF_trial_element; j++) {
            int j_nSpace = j * nSpace;
            elementJacobian_h_h[i][j] += (i == j ? 1.0 : 0.0) * h_test_dV[i];
            elementJacobian_hu_hu[i][j] +=
                (i == j ? 1.0 : 0.0) * vel_test_dV[i];
            elementJacobian_hv_hv[i][j] +=
                (i == j ? 1.0 : 0.0) * vel_test_dV[i];
            elementJacobian_heta_heta[i][j] +=
                (i == j ? 1.0 : 0.0) * vel_test_dV[i];
            elementJacobian_hw_hw[i][j] +=
                (i == j ? 1.0 : 0.0) * vel_test_dV[i];
            elementJacobian_hbeta_hbeta[i][j] +=
                (i == j ? 1.0 : 0.0) * vel_test_dV[i];
          } // j
        }   // i
      }     // k
      //
      // load into element Jacobian into global Jacobian
      //
      for (int i = 0; i < nDOF_test_element; i++) {
        int eN_i = eN * nDOF_test_element + i;
        for (int j = 0; j < nDOF_trial_element; j++) {
          int eN_i_j = eN_i * nDOF_trial_element + j;
          globalJacobian[csrRowIndeces_h_h[eN_i] +
                         csrColumnOffsets_h_h[eN_i_j]] +=
              elementJacobian_h_h[i][j];
          globalJacobian[csrRowIndeces_hu_hu[eN_i] +
                         csrColumnOffsets_hu_hu[eN_i_j]] +=
              elementJacobian_hu_hu[i][j];
          globalJacobian[csrRowIndeces_hv_hv[eN_i] +
                         csrColumnOffsets_hv_hv[eN_i_j]] +=
              elementJacobian_hv_hv[i][j];
          globalJacobian[csrRowIndeces_heta_heta[eN_i] +
                         csrColumnOffsets_heta_heta[eN_i_j]] +=
              elementJacobian_heta_heta[i][j];
          globalJacobian[csrRowIndeces_hw_hw[eN_i] +
                         csrColumnOffsets_hw_hw[eN_i_j]] +=
              elementJacobian_hw_hw[i][j];
          globalJacobian[csrRowIndeces_hbeta_hbeta[eN_i] +
                         csrColumnOffsets_hbeta_hbeta[eN_i_j]] +=
              elementJacobian_hbeta_hbeta[i][j];
        } // j
      }   // i
    }     // elements
  }
}; // namespace proteus

inline GN_SW2DCV_base *
newGN_SW2DCV(int nSpaceIn, int nQuadraturePoints_elementIn,
             int nDOF_mesh_trial_elementIn, int nDOF_trial_elementIn,
             int nDOF_test_elementIn, int nQuadraturePoints_elementBoundaryIn,
             int CompKernelFlag) {
  return proteus::chooseAndAllocateDiscretization2D<GN_SW2DCV_base, GN_SW2DCV,
                                                    CompKernel>(
      nSpaceIn, nQuadraturePoints_elementIn, nDOF_mesh_trial_elementIn,
      nDOF_trial_elementIn, nDOF_test_elementIn,
      nQuadraturePoints_elementBoundaryIn, CompKernelFlag);
}
} // end namespace proteus

#endif
