#ifndef SW2DCV_H
#define SW2DCV_H
#include "ArgumentsDict.h"
#include "CompKernel.h"
#include "ModelFactory.h"
#include "xtensor-python/pyarray.hpp"
#include <assert.h>
#include <cmath>
#include <iostream>
#include <valarray>

namespace py = pybind11;

#define POWER_SMOOTHNESS_INDICATOR 2
#define VEL_FIX_POWER 2.
#define REESTIMATE_MAX_EDGE_BASED_CFL 0
#define LIMITING_ITERATION 2

/* inline functions */
namespace proteus {
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
// FOR ESTIMATING MAX WAVE SPEEDS
inline double f(const double &g, const double &h, const double &hZ) {
  return ((h <= hZ) ? 2. * (sqrt(g * h) - sqrt(g * hZ))
                    : (h - hZ) * sqrt(0.5 * g * (h + hZ) / h / hZ));
}
inline double phi(const double &g, const double &h, const double &hL,
                  const double &hR, const double &uL, const double &uR) {
  return (f(g, h, hL) + f(g, h, hR) + uR - uL);
}
inline double fp(const double &g, const double &h, const double &hZ) {
  return ((h <= hZ)
              ? sqrt(g / h)
              : g * (2 * h * h + h * hZ + hZ * hZ) /
                    (2 * sqrt(2 * g) * h * h * hZ * sqrt(1 / h + 1 / hZ)));
}
inline double phip(const double &g, const double &h, const double &hL,
                   const double &hR) {
  return (fp(g, h, hL) + fp(g, h, hR));
}
inline double nu1(const double &g, const double &hStar, const double &hL,
                  const double &uL, const double &one_over_hL) {
  return (uL - sqrt(g * hL) *
                   sqrt((1. + fmax((hStar - hL) / 2. * one_over_hL, 0.0)) *
                        (1. + fmax((hStar - hL) * one_over_hL, 0.))));
}
inline double nu3(const double &g, const double &hStar, const double &hR,
                  const double &uR, const double &one_over_hR) {
  return (uR + sqrt(g * hR) *
                   sqrt((1. + fmax((hStar - hR) / 2. * one_over_hR, 0.0)) *
                        (1. + fmax((hStar - hR) * one_over_hR, 0.))));
}
inline double phiDiff(const double &g, const double &h1k, const double &h2k,
                      const double &hL, const double &hR, const double &uL,
                      const double &uR) {
  return ((phi(g, h2k, hL, hR, uL, uR) - phi(g, h1k, hL, hR, uL, uR)) /
          (h2k - h1k));
}
inline double phiDDiff1(const double &g, const double &h1k, const double &h2k,
                        const double &hL, const double &hR, const double &uL,
                        const double &uR) {
  return ((phiDiff(g, h1k, h2k, hL, hR, uL, uR) - phip(g, h1k, hL, hR)) /
          (h2k - h1k));
}
inline double phiDDiff2(const double &g, const double &h1k, const double &h2k,
                        const double &hL, const double &hR, const double &uL,
                        const double &uR) {
  return ((phip(g, h2k, hL, hR) - phiDiff(g, h1k, h2k, hL, hR, uL, uR)) /
          (h2k - h1k));
}
inline double hStarLFromQuadPhiFromAbove(const double &g, const double &hStarL,
                                         const double &hStarR, const double &hL,
                                         const double &hR, const double &uL,
                                         const double &uR) {
  return (hStarL -
          2 * phi(g, hStarL, hL, hR, uL, uR) /
              (phip(g, hStarL, hL, hR) +
               sqrt(std::pow(phip(g, hStarL, hL, hR), 2) -
                    4 * phi(g, hStarL, hL, hR, uL, uR) *
                        phiDDiff1(g, hStarL, hStarR, hL, hR, uL, uR))));
}
inline double hStarRFromQuadPhiFromBelow(const double &g, const double &hStarL,
                                         const double &hStarR, const double &hL,
                                         const double &hR, const double &uL,
                                         const double &uR) {
  return (hStarR -
          2 * phi(g, hStarR, hL, hR, uL, uR) /
              (phip(g, hStarR, hL, hR) +
               sqrt(std::pow(phip(g, hStarR, hL, hR), 2) -
                    4 * phi(g, hStarR, hL, hR, uL, uR) *
                        phiDDiff2(g, hStarL, hStarR, hL, hR, uL, uR))));
}
} // namespace proteus

namespace proteus {

class SW2DCV_base {
public:
  virtual ~SW2DCV_base() {}
  virtual void convexLimiting(arguments_dict &args) = 0;
  virtual double calculateEdgeBasedCFL(arguments_dict &args) = 0;
  virtual void calculateEV(arguments_dict &args) = 0;
  virtual void calculateResidual(arguments_dict &args) = 0;
  virtual void calculateMassMatrix(arguments_dict &args) = 0;
  virtual void calculateLumpedMassMatrix(arguments_dict &args) = 0;
};

template <class CompKernelType, int nSpace, int nQuadraturePoints_element,
          int nDOF_mesh_trial_element, int nDOF_trial_element,
          int nDOF_test_element, int nQuadraturePoints_elementBoundary>
class SW2DCV : public SW2DCV_base {
public:
  const int nDOF_test_X_trial_element;
  CompKernelType ck;
  SW2DCV()
      : nDOF_test_X_trial_element(nDOF_test_element * nDOF_trial_element),
        ck() {
    std::cout << "Constructing SW2DCV<CompKernelTemplate<" << nSpace << ","
              << nQuadraturePoints_element << "," << nDOF_mesh_trial_element
              << "," << nDOF_trial_element << "," << nDOF_test_element << ","
              << nQuadraturePoints_elementBoundary << ">());" << std::endl
              << std::flush;
  }

  inline double maxWaveSpeedSharpInitialGuess(double g, double nx, double ny,
                                              double hL, double huL, double hvL,
                                              double hR, double huR, double hvR,
                                              double hEps, bool debugging) {
    double lambda1, lambda3;
    // 1-eigenvalue: uL-sqrt(g*hL)
    // 3-eigenvalue: uR+sqrt(g*hR)

    // To avoid division by 0
    double one_over_hL = 2.0 * hL / (hL * hL + std::pow(fmax(hL, hEps), 2.0));
    double one_over_hR = 2.0 * hR / (hR * hR + std::pow(fmax(hR, hEps), 2.0));

    double hVelL = nx * huL + ny * hvL;
    double hVelR = nx * huR + ny * hvR;
    double velL = one_over_hL * hVelL;
    double velR = one_over_hR * hVelR;

    double x0 = std::pow(2. * sqrt(2.) - 1., 2.);
    double hMin = fmin(hL, hR);
    double hMax = fmax(hL, hR);

    double hStar;
    double fMin = phi(g, x0 * hMin, hL, hR, velL, velR);
    double fMax = phi(g, x0 * hMax, hL, hR, velL, velR);

    double sqrMin = sqrt(hMin);
    double sqrMax = sqrt(hMax);

    if (0. <= fMin) {
      hStar = fmin(
          x0 * hMin,
          std::pow(fmax(0., velL - velR + 2. * sqrt(g) * (sqrt(hL) + sqrt(hR))),
                   2) /
              16. / g);
    } else if (0. <= fMax) {
      double a = 1.0 / (2.0 * sqrt(2.0));
      double c = -hMin * a - sqrMin * sqrMax +
                 sqrMin * (velR - velL) / (2.0 * sqrt(g));
      double delta = hMin - 4.0 * a * c;
      if (delta < 0.0) {
        std::cout << "Bug in computing lambda. Exiting." << std::endl;
        abort();
      }
      hStar = fmin(x0 * hMax, std::pow((-sqrMin + sqrt(delta)) / (2. * a), 2));
    } else {
      hStar = sqrMin * sqrMax *
              (1.0 + sqrt(2.0 / g) * (velL - velR) / (sqrMin + sqrMax));
    }

    // return lambda_max
    lambda1 = nu1(g, hStar, hL, velL, one_over_hL);
    lambda3 = nu3(g, hStar, hR, velR, one_over_hR);
    return fmax(fabs(lambda1), fabs(lambda3));
  }

  inline void calculateCFL(const double &elementDiameter, const double &g,
                           const double &h, const double &hu, const double &hv,
                           const double hEps, double &cfl) {
    double cflx, cfly, c = sqrt(fmax(g * hEps, g * h));
    double u = 2 * h / (h * h + std::pow(fmax(h, hEps), 2)) * hu;
    double v = 2 * h / (h * h + std::pow(fmax(h, hEps), 2)) * hv;

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
    double dt = args.scalar<double>("dt");
    int NNZ = args.scalar<int>("NNZ");
    int numDOFs = args.scalar<int>("numDOFs");
    xt::pyarray<double> &lumped_mass_matrix =
        args.array<double>("lumped_mass_matrix");
    xt::pyarray<double> &h_old = args.array<double>("h_old");
    xt::pyarray<double> &hu_old = args.array<double>("hu_old");
    xt::pyarray<double> &hv_old = args.array<double>("hv_old");
    xt::pyarray<double> &b_dof = args.array<double>("b_dof");
    xt::pyarray<double> &high_order_hnp1 =
        args.array<double>("high_order_hnp1");
    xt::pyarray<double> &high_order_hunp1 =
        args.array<double>("high_order_hunp1");
    xt::pyarray<double> &high_order_hvnp1 =
        args.array<double>("high_order_hvnp1");
    xt::pyarray<double> &extendedSourceTerm_hu =
        args.array<double>("extendedSourceTerm_hu");
    xt::pyarray<double> &extendedSourceTerm_hv =
        args.array<double>("extendedSourceTerm_hv");
    xt::pyarray<double> &limited_hnp1 = args.array<double>("limited_hnp1");
    xt::pyarray<double> &limited_hunp1 = args.array<double>("limited_hunp1");
    xt::pyarray<double> &limited_hvnp1 = args.array<double>("limited_hvnp1");
    xt::pyarray<int> &csrRowIndeces_DofLoops =
        args.array<int>("csrRowIndeces_DofLoops");
    xt::pyarray<int> &csrColumnOffsets_DofLoops =
        args.array<int>("csrColumnOffsets_DofLoops");
    xt::pyarray<double> &MassMatrix = args.array<double>("MassMatrix");
    xt::pyarray<double> &dH_minus_dL = args.array<double>("dH_minus_dL");
    xt::pyarray<double> &muH_minus_muL = args.array<double>("muH_minus_muL");
    double hEps = args.scalar<double>("hEps");
    int LUMPED_MASS_MATRIX = args.scalar<int>("LUMPED_MASS_MATRIX");
    // diagnostic: no bed in the star states (well-balanced dissipation off)
    int NO_STAR_STATES = args.scalar<int>("NO_STAR_STATES");
    xt::pyarray<double> &dLow = args.array<double>("dLow");
    xt::pyarray<double> &new_SourceTerm_hu =
        args.array<double>("new_SourceTerm_hu");
    xt::pyarray<double> &new_SourceTerm_hv =
        args.array<double>("new_SourceTerm_hv");
    xt::pyarray<double> &hLow = args.array<double>("hLow");
    xt::pyarray<double> &huLow = args.array<double>("huLow");
    xt::pyarray<double> &hvLow = args.array<double>("hvLow");
    xt::pyarray<double> &h_min = args.array<double>("h_min");
    xt::pyarray<double> &h_max = args.array<double>("h_max");
    xt::pyarray<double> &kin_max = args.array<double>("kin_max");
    double KE_tiny = args.scalar<double>("KE_tiny");

    // The limiter runs as a sequence of calls so the caller can exchange
    // ghost-DOF values in between (each ghost row only sees part of its
    // stencil): per pass, stage 0 computes R+-, the caller exchanges them,
    // stage 1 computes l_ij, applies the limited fluxes and updates the FCT
    // matrices, and the caller exchanges hLow/huLow/hvLow before the next
    // pass. FCT_h, FCT_hu, FCT_hv and l_ij persist in FCT_all (4 x nnz).
    const int limit_iter_arg = args.scalar<int>("limit_iter");
    const int limit_stage = args.scalar<int>("limit_stage");
    xt::pyarray<double> &FCT_all = args.array<double>("FCT_all");
    xt::pyarray<double> &Rneg = args.array<double>("Rneg_dof");
    xt::pyarray<double> &Rpos = args.array<double>("Rpos_dof");
    // number of neighbours of each DOF, exchanged with the ghosts: a ghost
    // DOF's row in the local CSR pattern is truncated
    xt::pyarray<double> &num_neighbours = args.array<double>("num_neighbours");
    const bool first_call = (limit_iter_arg == 0 && limit_stage == 0);

    // Create FCT component matrices in vector form
    std::valarray<double> FCT_h(0.0, dH_minus_dL.size()),
        FCT_hu(0.0, dH_minus_dL.size()), FCT_hv(0.0, dH_minus_dL.size());
    const std::size_t nnz_fct = dH_minus_dL.size();

    ////////////////////////////////////////////////////
    // Loop to define FCT matrices for each component //
    ////////////////////////////////////////////////////
    int ij = 0;
    for (int i = 0; i < numDOFs && first_call; i++) {
      // Read some vectors
      double high_order_hnp1i = high_order_hnp1[i];
      double high_order_hunp1i = high_order_hunp1[i];
      double high_order_hvnp1i = high_order_hvnp1[i];
      double hi = h_old[i];
      double huni = hu_old[i];
      double hvni = hv_old[i];
      double Zi = b_dof[i];
      double mi = lumped_mass_matrix[i];
      double one_over_hiReg =
          2 * hi / (hi * hi + std::pow(fmax(hi, hEps), 2)); // hEps

      // LOOP OVER THE SPARSITY PATTERN (j-LOOP)//
      for (int offset = csrRowIndeces_DofLoops[i];
           offset < csrRowIndeces_DofLoops[i + 1]; offset++) {

        int j = csrColumnOffsets_DofLoops[offset];

        if (i != j) {
          // Read some vectors
          double hj = h_old[j];
          double hunj = hu_old[j];
          double hvnj = hv_old[j];
          double Zj = b_dof[j];
          double one_over_hjReg =
              2. * hj / (hj * hj + std::pow(fmax(hj, hEps), 2));

          // Compute star states
          double hStarij = NO_STAR_STATES ? hi : fmax(0., hi + Zi - fmax(Zi, Zj));
          double huStarij = huni * hStarij * one_over_hiReg;
          double hvStarij = hvni * hStarij * one_over_hiReg;

          double hStarji = NO_STAR_STATES ? hj : fmax(0., hj + Zj - fmax(Zi, Zj));
          double huStarji = hunj * hStarji * one_over_hjReg;
          double hvStarji = hvnj * hStarji * one_over_hjReg;

          // i-th row of flux correction matrix
          double ML_minus_MC = (LUMPED_MASS_MATRIX == 1
                                    ? 0.
                                    : (i == j ? 1. : 0.) * mi - MassMatrix[ij]);

          FCT_h[ij] =
              ML_minus_MC *
                  (high_order_hnp1[j] - hj - (high_order_hnp1i - hi)) +
              dt * (dH_minus_dL[ij] - muH_minus_muL[ij]) * (hStarji - hStarij) +
              dt * muH_minus_muL[ij] * (hj - hi);

          FCT_hu[ij] = ML_minus_MC * (high_order_hunp1[j] - hunj -
                                      (high_order_hunp1i - huni)) +
                       dt * (dH_minus_dL[ij] - muH_minus_muL[ij]) *
                           (huStarji - huStarij) +
                       dt * muH_minus_muL[ij] * (hunj - huni);

          FCT_hv[ij] = ML_minus_MC * (high_order_hvnp1[j] - hvnj -
                                      (high_order_hvnp1i - hvni)) +
                       dt * (dH_minus_dL[ij] - muH_minus_muL[ij]) *
                           (hvStarji - hvStarij) +
                       dt * muH_minus_muL[ij] * (hvnj - hvni);

        } else {
          FCT_h[ij] = 0.0;
          FCT_hu[ij] = 0.0;
          FCT_hv[ij] = 0.0;
        }

        // UPDATE ij //
        ij += 1;
      } // j loop ends here
    }   // i loop ends here

    ////////////////////////////////////////////////////////////////////
    // Main loop to define limiters and computed limited solution //////
    ////////////////////////////////////////////////////////////////////

    // Create Lij_array and initialize with 1
    std::valarray<double> Lij_array(1.0, dH_minus_dL.size());
    if (!first_call) {
      for (std::size_t k = 0; k < nnz_fct; k++) {
        FCT_h[k] = FCT_all[0 * nnz_fct + k];
        FCT_hu[k] = FCT_all[1 * nnz_fct + k];
        FCT_hv[k] = FCT_all[2 * nnz_fct + k];
        Lij_array[k] = FCT_all[3 * nnz_fct + k];
      }
    }

    /* One stage of one limiting pass per call */
    for (int limit_iter = limit_iter_arg; limit_iter < limit_iter_arg + 1;
         limit_iter++) {

      /* Loop to define FCT Rpos and Rneg values */
      if (limit_stage == 0) {
      ij = 0;
      for (int i = 0; i < numDOFs; i++) {

        double hi = h_old[i];
        double mi = lumped_mass_matrix[i];
        // Initialize Pneg and Ppos quantities at ith node
        double Pnegi = 0., Pposi = 0.;

        // LOOP OVER THE SPARSITY PATTERN (j-LOOP)
        for (int offset = csrRowIndeces_DofLoops[i];
             offset < csrRowIndeces_DofLoops[i + 1]; offset++) {

          int j = csrColumnOffsets_DofLoops[offset];

          // COMPUTE P VECTORS
          Pnegi += FCT_h[ij] * ((FCT_h[ij] < 0) ? 1. : 0.);
          Pposi += FCT_h[ij] * ((FCT_h[ij] > 0) ? 1. : 0.);

          // UPDATE ij
          ij += 1;
        } // j loop ends here

        double psmall_h = 1E-14 * fmax(fabs(Pnegi), fabs(Pposi));

        ///////////////////////
        // COMPUTE Q VECTORS //
        ///////////////////////
        double Qnegi = std::min(mi * (h_min[i] - hLow[i]), 0.0);
        double Qposi = std::max(mi * (h_max[i] - hLow[i]), 0.0);

        ///////////////////////
        // COMPUTE R VECTORS //
        ///////////////////////
        if (hi <= hEps) {
          Rneg[i] = 0.;
          Rpos[i] = 0.;
        } else {
          // for h
          if (Pnegi >= -psmall_h) {
            Rneg[i] = 1.0;
          } else {
            Rneg[i] = std::min(1.0, Qnegi / Pnegi);
          }
          if (Pposi <= psmall_h) {
            Rpos[i] = 1.0;
          } else {
            Rpos[i] = std::min(1.0, Qposi / Pposi);
          }
        }
      } // i loop ends here
      break; // stage 0 done: the caller exchanges Rneg/Rpos
      }

      /* Here we compute the limiters */
      ij = 0;
      for (int i = 0; i < numDOFs; i++) {

        double mi = lumped_mass_matrix[i];
        double ci =
            kin_max[i] * hLow[i] -
            0.5 * (huLow[i] * huLow[i] + hvLow[i] * hvLow[i]); // for KE lim.

        // LOOP OVER THE SPARSITY PATTERN (j-LOOP)//
        for (int offset = csrRowIndeces_DofLoops[i];
             offset < csrRowIndeces_DofLoops[i + 1]; offset++) {

          int j = csrColumnOffsets_DofLoops[offset];

          if (j != i) {

            // Compute limiter based on water height
            if (FCT_h[ij] >= 0.) {
              Lij_array[ij] = fmin(Lij_array[ij], std::min(Rneg[j], Rpos[i]));
            } else {
              Lij_array[ij] = fmin(Lij_array[ij], std::min(Rneg[i], Rpos[j]));
            }

            /*======================================================*/
            /*            Kinetic Energy limiting                   */
            double lambdaj =
                num_neighbours[i];
            double Ph_ij = FCT_h[ij] / mi / lambdaj;
            double Phu_ij = FCT_hu[ij] / mi / lambdaj;
            double Phv_ij = FCT_hv[ij] / mi / lambdaj;

            // Here we initialize limiter based on kinetic energy
            double KE_limiter = 1;
            double neg_root_i = 1., neg_root_j = 1.;

            // We first check if local kinetic energy > 0
            if (kin_max[i] * hLow[i] <= KE_tiny)
              KE_limiter = 0.;

            // We then check if KE bound is already satisfied
            double hi_with_lijPij = hLow[i] + Lij_array[ij] * Ph_ij;
            double hui_with_lijPij = huLow[i] + Lij_array[ij] * Phu_ij;
            double hvi_with_lijPij = hvLow[i] + Lij_array[ij] * Phv_ij;
            double psi = kin_max[i] * hi_with_lijPij -
                         0.5 * (hui_with_lijPij * hui_with_lijPij +
                                hvi_with_lijPij * hvi_with_lijPij);
            if (psi > -KE_tiny) {
              KE_limiter = fmin(KE_limiter, Lij_array[ij]);
            }

            /*======================================================*/

            double ai = -0.5 * (Phu_ij * Phu_ij + Phv_ij * Phv_ij);
            double bi =
                kin_max[i] * Ph_ij - (huLow[i] * Phu_ij + hvLow[i] * Phv_ij);
            double delta_i = bi * bi - 4. * ai * ci;

            if (delta_i < 0. || ai >= -0.) {
              KE_limiter = fmin(KE_limiter, Lij_array[ij]);
            } else {
              neg_root_i = (-bi - std::sqrt(delta_i)) / 2. / ai;
            }

            // root of jth-DOF (To compute transpose component)
            double lambdai =
                num_neighbours[j];
            double mj = lumped_mass_matrix[j];
            double cj = kin_max[j] * hLow[j] -
                        0.5 * (huLow[j] * huLow[j] + hvLow[j] * hvLow[j]);
            double Ph_ji = -FCT_h[ij] / mj / lambdai; // Aij=-Aji
            double Phu_ji = -FCT_hu[ij] / mj / lambdai;
            double Phv_ji = -FCT_hv[ij] / mj / lambdai;
            double aj = -0.5 * (Phu_ji * Phu_ji + Phv_ji * Phv_ji);
            double bj =
                kin_max[j] * Ph_ji - (huLow[j] * Phu_ji + hvLow[j] * Phv_ji);
            double delta_j = bj * bj - 4. * aj * cj;

            if (delta_j < 0. || aj >= -0.) {
              KE_limiter = fmin(KE_limiter, Lij_array[ij]);
            } else {
              neg_root_j = (-bj - std::sqrt(delta_j)) / 2. / aj;
            }

            // define final limiter based on KE
            KE_limiter =
                fmin(KE_limiter, fmin(fabs(neg_root_i), fabs(neg_root_j)));

            // Here we set final limiter
            Lij_array[ij] = fmin(KE_limiter, Lij_array[ij]);

          } else {
            // if i = j then lij = 0
            Lij_array[ij] = 0.;
          }

          // update ij
          ij += 1;
        } // end j loop
      }   // end i loop

      // The kinetic-energy checks above use row-i data only, so l_ij and
      // l_ji can differ; the limited flux conserves mass only if they agree.
      for (int i = 0; i < numDOFs; i++) {
        for (int offset = csrRowIndeces_DofLoops[i];
             offset < csrRowIndeces_DofLoops[i + 1]; offset++) {
          int j = csrColumnOffsets_DofLoops[offset];
          if (j <= i)
            continue;
          for (int offset_ji = csrRowIndeces_DofLoops[j];
               offset_ji < csrRowIndeces_DofLoops[j + 1]; offset_ji++) {
            if (csrColumnOffsets_DofLoops[offset_ji] == i) {
              double l = fmin(Lij_array[offset], Lij_array[offset_ji]);
              Lij_array[offset] = l;
              Lij_array[offset_ji] = l;
              break;
            }
          }
        }
      }

      /* Final loop to apply limiting and then define limited solution */
      ij = 0;
      for (int i = 0; i < numDOFs; i++) {

        double one_over_mi = 1.0 / lumped_mass_matrix[i];
        double ith_Limiter_times_FluxCorrectionMatrix1 = 0.;
        double ith_Limiter_times_FluxCorrectionMatrix2 = 0.;
        double ith_Limiter_times_FluxCorrectionMatrix3 = 0.;

        // LOOP OVER THE SPARSITY PATTERN (j-LOOP)//
        for (int offset = csrRowIndeces_DofLoops[i];
             offset < csrRowIndeces_DofLoops[i + 1]; offset++) {

          int j = csrColumnOffsets_DofLoops[offset];

          // COMPUTE LIMITED FLUX //
          ith_Limiter_times_FluxCorrectionMatrix1 += Lij_array[ij] * FCT_h[ij];
          ith_Limiter_times_FluxCorrectionMatrix2 += Lij_array[ij] * FCT_hu[ij];
          ith_Limiter_times_FluxCorrectionMatrix3 += Lij_array[ij] * FCT_hv[ij];

          // update ij
          ij += 1;
        } // end j loop

        // then we add lij*Aij to uLow
        hLow[i] += one_over_mi * ith_Limiter_times_FluxCorrectionMatrix1;
        huLow[i] += one_over_mi * ith_Limiter_times_FluxCorrectionMatrix2;
        hvLow[i] += one_over_mi * ith_Limiter_times_FluxCorrectionMatrix3;

        // Finally define the limited solution
        limited_hnp1[i] = hLow[i];
        limited_hunp1[i] = huLow[i] + dt * one_over_mi * new_SourceTerm_hu[i];
        limited_hvnp1[i] = hvLow[i] + dt * one_over_mi * new_SourceTerm_hv[i];

        if (limited_hnp1[i] < -hEps && dt < 1.0) {
          std::cout << "Limited water height is negative: \n "
                    << "hLow: " << hLow[i] << "\n"
                    << "hHigh: " << limited_hnp1[i] << "\n"
                    << "hEps:  " << hEps << "\n"
                    << " ... aborting!" << std::endl;
          abort();
        } else {
          // clean up uHigh from round off error
          if (limited_hnp1[i] < hEps)
            limited_hnp1[i] = 0.0;
          double aux = fmax(limited_hnp1[i], hEps);
          limited_hunp1[i] *= 2 * std::pow(limited_hnp1[i], VEL_FIX_POWER) /
                              (std::pow(limited_hnp1[i], VEL_FIX_POWER) +
                               std::pow(aux, VEL_FIX_POWER));
          limited_hvnp1[i] *= 2 * std::pow(limited_hnp1[i], VEL_FIX_POWER) /
                              (std::pow(limited_hnp1[i], VEL_FIX_POWER) +
                               std::pow(aux, VEL_FIX_POWER));
        }
      } // end i loop

      // update FCT matrices as Fct = (1 - Lij)*Fct
      FCT_h = (1.0 - Lij_array) * FCT_h;
      FCT_hu = (1.0 - Lij_array) * FCT_hu;
      FCT_hv = (1.0 - Lij_array) * FCT_hv;
    } // end loop for limiting iteration

    for (std::size_t k = 0; k < nnz_fct; k++) {
      FCT_all[0 * nnz_fct + k] = FCT_h[k];
      FCT_all[1 * nnz_fct + k] = FCT_hu[k];
      FCT_all[2 * nnz_fct + k] = FCT_hv[k];
      FCT_all[3 * nnz_fct + k] = Lij_array[k];
    }
  }   // end convex limiting function

  // Porous edges: the graph viscosity on edge ij carries max(phi_i, phi_j).
  // With fluxes phi_j F_j and storage m_i phi_i this keeps the bar states
  // positive (phi_i/phibar <= 1) and the dissipation symmetric (conservative).
  static inline double phiBar(int i, int j, xt::pyarray<double> &porosity_dof) {
    return fmax(porosity_dof[i], porosity_dof[j]);
  }

  // Wall rows with wall_dof = 2 (WALL_RIEMANN): the boundary flux is the Rusanov
  // flux against the mirror state, whose normal momentum is reversed. With
  // nu = n_i/|n_i|, u_n = u.nu and lambda = |u_n| + sqrt(g h) the mass flux is
  // zero and the momentum flux is (p + h u_n^2 + lambda h u_n) nu. The
  // hydrostatic p is already carried by the pressure on c_ij, so only
  //   dp = h u_n (u_n + lambda)
  // is added (times n_i). It raises the wall pressure when the flow runs into
  // the wall and lowers it when the flow leaves, the impedance of the exact
  // wall Riemann problem. rate = (u_n + lambda)|n_i| >= 0 is its damping rate
  // on the normal momentum (for the CFL).
  inline void wallRiemann(double g, double hi, double hui, double hvi,
                          double nx, double ny, double hEps, double &dp,
                          double &rate) {
    dp = 0.;
    rate = 0.;
    const double nn = std::sqrt(nx * nx + ny * ny);
    if (nn <= 0. || hi <= hEps)
      return;
    const double one_over_h = 2.0 * hi / (hi * hi + std::pow(fmax(hi, hEps), 2));
    const double qn = (hui * nx + hvi * ny) / nn;
    const double un = qn * one_over_h;
    const double lambda = std::fabs(un) + std::sqrt(g * hi);
    dp = qn * (un + lambda);
    rate = (un + lambda) * nn;
  }

  // Tailwater rows (tailwater_dof = 1): exact Godunov flux against a
  // characteristic ghost state. The ghost has depth hb (the stage minus the
  // bed, equal to h_i at rest when the stage matches), the tangential velocity
  // of the interior and the outgoing Riemann invariant kept,
  //   u_n,b = u_n,i + 2 (sqrt(g h_i) - sqrt(g h_b)),
  // so the Riemann problem (U_i, U_b) contains a single wave of the u_n - c
  // family and the state at the interface is one of three:
  //   shock (hb > hi): speed s = (hb u_b - hi u_i)/(hb - hi); U_b if s < 0,
  //     else U_i (the bore is swept out of the domain);
  //   rarefaction (hb < hi): U_b if u_b - c_b <= 0, U_i if u_i - c_i >= 0,
  //     else the critical state inside the fan (c = (u_i + 2 c_i)/3, u_n = c).
  // Supercritical outflow therefore ignores the data unless the tailwater
  // is high enough to send a wave upstream. The natural (open) row already
  // carries F(U_i).nu through c_ij, so only
  //   dF = F(U*).nu - F(U_i).nu                    (zero when U_b = U_i)
  // times the boundary length |n_i| is added to the low-order and high-order
  // updates. rate = (|u_n,i| + c_i + |u_n,b| + c_b)|n_i| / 2 bounds the
  // damping rate (for the CFL).
  inline void tailwaterFlux(double g, double hi, double hui, double hvi,
                            double hb, double nx, double ny, double hEps,
                            double &dFh, double &dFhu, double &dFhv,
                            double &rate) {
    dFh = dFhu = dFhv = rate = 0.;
    const double nn = std::sqrt(nx * nx + ny * ny);
    if (nn <= 0. || (hi <= hEps && hb <= hEps))
      return;
    const double nux = nx / nn, nuy = ny / nn;
    const double tx = -nuy, ty = nux;
    const double one_over_h = 2.0 * hi / (hi * hi + std::pow(fmax(hi, hEps), 2));
    const double ui = hui * one_over_h, vi = hvi * one_over_h;
    const double un_i = ui * nux + vi * nuy, ut = ui * tx + vi * ty;
    const double c_i = std::sqrt(g * fmax(hi, 0.));
    const double c_b = std::sqrt(g * fmax(hb, 0.));
    const double R = un_i + 2.0 * c_i;
    const double un_b = R - 2.0 * c_b;
    rate = 0.5 * (std::fabs(un_i) + c_i + std::fabs(un_b) + c_b) * nn;
    double hs, uns;
    bool ghost_state = false;
    if (hb >= hi) {
      if (hb - hi <= 1e-14) // ghost = interior
        return;
      const double s = (hb * un_b - hi * un_i) / (hb - hi);
      if (s < 0.) { hs = hb; uns = un_b; ghost_state = true; } else { hs = hi; uns = un_i; }
    } else {
      if (un_b - c_b <= 0.) { hs = hb; uns = un_b; ghost_state = true; }
      else if (un_i - c_i >= 0.) { hs = hi; uns = un_i; }
      else { const double c = R / 3.0; hs = c * c / g; uns = c; }
    }
    // inflow brings the tangential velocity of the outside (zero), outflow keeps
    // the interior one
    const double ut_s = (ghost_state && uns < 0.) ? 0. : ut;
    const double p_s = 0.5 * g * hs * hs, p_i = 0.5 * g * hi * hi;
    const double mass_s = hs * uns, mass_i = hi * un_i;
    dFh = (mass_s - mass_i) * nn;
    dFhu = (mass_s * (uns * nux + ut_s * tx) + p_s * nux -
            (mass_i * (un_i * nux + ut * tx) + p_i * nux)) * nn;
    dFhv = (mass_s * (uns * nuy + ut_s * ty) + p_s * nuy -
            (mass_i * (un_i * nuy + ut * ty) + p_i * nuy)) * nn;
  }

  // On an edge touching a wall row, c'_ij = -c_ji no longer equals -c'_ji,
  // so mu_ij computed from row i alone differs from mu_ji. The dissipation
  // sum_j mu_ij (U_j - U_i) is conservative only if mu is symmetric, so take
  // the max over both orientations; this returns the row-j orientation.
  inline double muWallTranspose(int i, int j, int ij, double ui, double vi,
                                double uj, double vj,
                                xt::pyarray<double> &wall_dof,
                                xt::pyarray<double> &Cx,
                                xt::pyarray<double> &Cy,
                                xt::pyarray<double> &CTx,
                                xt::pyarray<double> &CTy) {
    // c'_ji: c_ji = CTx[ij] on an ordinary row j, -c_ij on a wall row j
    double cx_ji = wall_dof[j] > 0.5 ? -Cx[ij] : CTx[ij];
    double cy_ji = wall_dof[j] > 0.5 ? -Cy[ij] : CTy[ij];
    return fmax(fmax(0., -(uj * cx_ji + vj * cy_ji)),
                fmax(0., (ui * cx_ji + vi * cy_ji)));
  }

  double calculateEdgeBasedCFL(arguments_dict &args) {
    double g = args.scalar<double>("g");
    int numDOFsPerEqn = args.scalar<int>("numDOFsPerEqn");
    xt::pyarray<double> &lumped_mass_matrix =
        args.array<double>("lumped_mass_matrix");
    xt::pyarray<double> &h_dof_old = args.array<double>("h_dof_old");
    xt::pyarray<double> &hu_dof_old = args.array<double>("hu_dof_old");
    xt::pyarray<double> &hv_dof_old = args.array<double>("hv_dof_old");
    xt::pyarray<double> &b_dof = args.array<double>("b_dof");
    xt::pyarray<int> &csrRowIndeces_DofLoops =
        args.array<int>("csrRowIndeces_DofLoops");
    xt::pyarray<int> &csrColumnOffsets_DofLoops =
        args.array<int>("csrColumnOffsets_DofLoops");
    double hEps = args.scalar<double>("hEps");
    xt::pyarray<double> &Cx = args.array<double>("Cx");
    xt::pyarray<double> &Cy = args.array<double>("Cy");
    xt::pyarray<double> &CTx = args.array<double>("CTx");
    xt::pyarray<double> &CTy = args.array<double>("CTy");
    xt::pyarray<double> &dLow = args.array<double>("dLow");
    double run_cfl = args.scalar<double>("run_cfl");
    xt::pyarray<double> &edge_based_cfl = args.array<double>("edge_based_cfl");
    int debug = args.scalar<int>("debug");
    xt::pyarray<double> &wall_dof = args.array<double>("wall_dof");
    xt::pyarray<double> &tailwater_dof = args.array<double>("tailwater_dof");
    xt::pyarray<double> &tailwater_depth_dof =
        args.array<double>("tailwater_depth_dof");
    xt::pyarray<double> &porosity_dof = args.array<double>("porosity_dof");

    double max_edge_based_cfl = 0.;

    int ij = 0;
    for (int i = 0; i < numDOFsPerEqn; i++) {
      // solution at time tn for the ith DOF
      double hi = h_dof_old[i];
      double hui = hu_dof_old[i];
      double hvi = hv_dof_old[i];
      double dLowii = 0.;
      // boundary vector n_i = sum_j c_ji = int_{dOmega} phi_i n (zero inside)
      double nix = 0., niy = 0.;

      for (int offset = csrRowIndeces_DofLoops[i];
           offset < csrRowIndeces_DofLoops[i + 1]; offset++) {

        // loop in j (sparsity pattern)
        int j = csrColumnOffsets_DofLoops[offset];

        // solution at time tn for the jth DOF
        double hj = h_dof_old[j];
        double huj = hu_dof_old[j];
        double hvj = hv_dof_old[j];

        if (i != j) {
          ////////////////////////
          // DISSIPATIVE MATRIX //
          ////////////////////////
          double cij_norm = sqrt(Cx[ij] * Cx[ij] + Cy[ij] * Cy[ij]);
          double cji_norm = sqrt(CTx[ij] * CTx[ij] + CTy[ij] * CTy[ij]);
          double nxij = Cx[ij] / cij_norm, nyij = Cy[ij] / cij_norm;
          double nxji = CTx[ij] / cji_norm, nyji = CTy[ij] / cji_norm;
          dLow[ij] =
              fmax(maxWaveSpeedSharpInitialGuess(g, nxij, nyij, hi, hui, hvi,
                                                 hj, huj, hvj, hEps, debug) *
                       cij_norm, // hEps
                   maxWaveSpeedSharpInitialGuess(g, nxji, nyji, hj, huj, hvj,
                                                 hi, hui, hvi, hEps, debug) *
                       cji_norm); // hEps
          dLow[ij] *= phiBar(i, j, porosity_dof);
          dLowii -= dLow[ij];
        } else
          dLow[ij] = 0.;
        nix += CTx[ij];
        niy += CTy[ij];
        // update ij
        ij += 1;
      }
      //////////////////////////////
      // CALCULATE EDGE BASED CFL //
      //////////////////////////////
      double mi = lumped_mass_matrix[i];
      // On a wall row the advective boundary term adds (u_i.n_i) U_i to the
      // update (see calculateResidual); when the flow leaves the wall
      // (u_i.n_i < 0) it needs that much more room for positivity.
      double wall_margin = 0.;
      if (wall_dof[i] > 0.5) {
        double one_over_hiReg =
            2.0 * hi / (hi * hi + std::pow(fmax(hi, hEps), 2));
        wall_margin = 0.5 * fmax(0., -porosity_dof[i] * (hui * nix + hvi * niy) *
                                         one_over_hiReg);
      }
      if (wall_dof[i] > 1.5) {
        double dp_w, rate_w;
        wallRiemann(g, hi, hui, hvi, nix, niy, hEps, dp_w, rate_w);
        wall_margin += 0.5 * porosity_dof[i] * rate_w;
      }
      if (tailwater_dof[i] > 0.5) {
        double dFh_t, dFhu_t, dFhv_t, rate_t;
        tailwaterFlux(g, hi, hui, hvi, tailwater_depth_dof[i], nix, niy, hEps,
                      dFh_t, dFhu_t, dFhv_t, rate_t);
        wall_margin += 0.5 * porosity_dof[i] * rate_t;
      }
      edge_based_cfl[i] = (fabs(dLowii) + wall_margin) / mi;
      max_edge_based_cfl = fmax(max_edge_based_cfl, edge_based_cfl[i]);
    }
    return max_edge_based_cfl;
  } // End calculateEdgeBasedCFL

  void calculateEV(arguments_dict &args) {
    double g = args.scalar<double>("g");
    xt::pyarray<double> &h_dof_old = args.array<double>("h_dof_old");
    xt::pyarray<double> &hu_dof_old = args.array<double>("hu_dof_old");
    xt::pyarray<double> &hv_dof_old = args.array<double>("hv_dof_old");
    xt::pyarray<double> &b_dof = args.array<double>("b_dof");
    xt::pyarray<double> &Cx = args.array<double>("Cx");
    xt::pyarray<double> &Cy = args.array<double>("Cy");
    xt::pyarray<double> &CTx = args.array<double>("CTx");
    xt::pyarray<double> &CTy = args.array<double>("CTy");
    int numDOFsPerEqn = args.scalar<int>("numDOFsPerEqn");
    xt::pyarray<int> &csrRowIndeces_DofLoops =
        args.array<int>("csrRowIndeces_DofLoops");
    xt::pyarray<int> &csrColumnOffsets_DofLoops =
        args.array<int>("csrColumnOffsets_DofLoops");
    xt::pyarray<double> &lumped_mass_matrix =
        args.array<double>("lumped_mass_matrix");
    double eps = args.scalar<double>("eps");
    double hEps = args.scalar<double>("hEps");
    xt::pyarray<double> &global_entropy_residual =
        args.array<double>("global_entropy_residual");
    double &dij_small = args.scalar<double>("dij_small");

    //////////////////////////////////////////////
    // ********** FIRST LOOP ON DOFs ********** //
    //////////////////////////////////////////////

    // To compute:
    //     * Entropy at i-th node
    std::valarray<double> eta(numDOFsPerEqn);
    for (int i = 0; i < numDOFsPerEqn; i++) {
      // COMPUTE ENTROPY. NOTE: WE CONSIDER A FLAT BOTTOM
      double hi = h_dof_old[i];
      double one_over_hiReg =
          2 * hi / (hi * hi + std::pow(fmax(hi, hEps), 2)); // hEps
      eta[i] = ENTROPY(g, hi, hu_dof_old[i], hv_dof_old[i], 0., one_over_hiReg);
    }

    // ********** END OF FIRST LOOP ON DOFs ********** //

    ///////////////////////////////////////////////
    // ********** SECOND LOOP ON DOFs ********** //
    ///////////////////////////////////////////////
    // To compute:
    //     * global entropy residual
    //     * dij_small to avoid division by 0

    int ij = 0;
    std::valarray<double> etaMax(numDOFsPerEqn), etaMin(numDOFsPerEqn);

    // speed = sqrt(g max(h_0)), I divide by h_epsilon to get max(h_0)
    double speed = std::sqrt(g * hEps / eps);
    dij_small = 0.0;

    for (int i = 0; i < numDOFsPerEqn; i++) {

      // solution at time tn for the ith DOF
      double hi = h_dof_old[i];
      double hui = hu_dof_old[i];
      double hvi = hv_dof_old[i];
      double Zi = b_dof[i];

      // Define some things using above
      double one_over_hiReg =
          2 * hi / (hi * hi + std::pow(fmax(hi, hEps), 2)); // hEps
      double ui = hui * one_over_hiReg;
      double vi = hvi * one_over_hiReg;
      double mi = lumped_mass_matrix[i];

      // initialize etaMax and etaMin
      etaMax[i] = fabs(eta[i]);
      etaMin[i] = fabs(eta[i]);

      // FOR ENTROPY RESIDUAL, NOTE: FLAT BOTTOM //
      double ith_flux_term1 = 0., ith_flux_term2 = 0., ith_flux_term3 = 0.;
      double entropy_flux = 0.;
      double sum_entprime_flux = 0.;
      double eta_prime1 = DENTROPY_DH(g, hi, hui, hvi, 0., one_over_hiReg);
      double eta_prime2 = DENTROPY_DHU(g, hi, hui, hvi, 0., one_over_hiReg);
      double eta_prime3 = DENTROPY_DHV(g, hi, hui, hvi, 0., one_over_hiReg);

      // loop in j (sparsity pattern)
      for (int offset = csrRowIndeces_DofLoops[i];
           offset < csrRowIndeces_DofLoops[i + 1]; offset++) {

        int j = csrColumnOffsets_DofLoops[offset];

        // solution at time tn for the jth DOF
        double hj = h_dof_old[j];
        double huj = hu_dof_old[j];
        double hvj = hv_dof_old[j];
        double Zj = b_dof[j];

        // Then define some things here using above
        double one_over_hjReg =
            2.0 * hj / (hj * hj + std::pow(fmax(hj, hEps), 2));
        double uj = huj * one_over_hjReg;
        double vj = hvj * one_over_hjReg;

        // auxiliary functions to compute fluxes
        double aux_h =
            (uj * hj - ui * hi) * Cx[ij] + (vj * hj - vi * hi) * Cy[ij];
        double aux_hu =
            (uj * huj - ui * hui) * Cx[ij] + (vj * huj - vi * hui) * Cy[ij];
        double aux_hv =
            (uj * hvj - ui * hvi) * Cx[ij] + (vj * hvj - vi * hvi) * Cy[ij];

        // flux for entropy
        ith_flux_term1 += aux_h;
        ith_flux_term2 += aux_hu + 0.5 * g * hj * hj * Cx[ij];
        ith_flux_term3 += aux_hv + 0.5 * g * hj * hj * Cy[ij];

        // NOTE: WE CONSIDER FLAT BOTTOM
        entropy_flux +=
            (Cx[ij] * ENTROPY_FLUX1(g, hj, huj, hvj, 0., one_over_hjReg) +
             Cy[ij] * ENTROPY_FLUX2(g, hj, huj, hvj, 0., one_over_hjReg));

        // COMPUTE ETA MIN AND ETA MAX //
        etaMax[i] = fmax(etaMax[i], fabs(eta[j]));
        etaMin[i] = fmin(etaMin[i], fabs(eta[j]));

        // define dij_small in j loop
        double x = fabs(Cx[ij]) + fabs(Cy[ij]);
        dij_small = fmax(dij_small, x * speed);

        // update ij
        ij += 1;
      } // end j loop

      // define sum of entprime*flux
      sum_entprime_flux =
          (ith_flux_term1 * eta_prime1 + ith_flux_term2 * eta_prime2 +
           ith_flux_term3 * eta_prime3);

      // define rescale for normalization
      double small_rescale = g * hEps * hEps / eps;
      double rescale = fmax(fabs(etaMax[i] - etaMin[i]) / 2., small_rescale);

      // COMPUTE ENTROPY RESIDUAL //
      double one_over_entNormFactori = 1.0 / rescale;
      global_entropy_residual[i] =
          one_over_entNormFactori * fabs(entropy_flux - sum_entprime_flux);
      if (hi <= hEps) {
        global_entropy_residual[i] = 1.0;
      }
    } // end i loop

    // Finally dij_small here
    dij_small = 1E-14 * dij_small;
    // ********** END OF LOOP IN DOFs ********** //
  } // end calculateEV

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
    xt::pyarray<int> &h_l2g = args.array<int>("h_l2g");
    xt::pyarray<int> &vel_l2g = args.array<int>("vel_l2g");
    xt::pyarray<double> &h_dof_old = args.array<double>("h_dof_old");
    xt::pyarray<double> &hu_dof_old = args.array<double>("hu_dof_old");
    xt::pyarray<double> &hv_dof_old = args.array<double>("hv_dof_old");
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
    int offset_h = args.scalar<int>("offset_h");
    int offset_hu = args.scalar<int>("offset_hu");
    int offset_hv = args.scalar<int>("offset_hv");
    int stride_h = args.scalar<int>("stride_h");
    int stride_hu = args.scalar<int>("stride_hu");
    int stride_hv = args.scalar<int>("stride_hv");
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
    xt::pyarray<double> &elementResidual_h =
        args.array<double>("elementResidual_h");
    xt::pyarray<double> &Cx = args.array<double>("Cx");
    xt::pyarray<double> &Cy = args.array<double>("Cy");
    xt::pyarray<double> &CTx = args.array<double>("CTx");
    xt::pyarray<double> &CTy = args.array<double>("CTy");
    int numDOFsPerEqn = args.scalar<int>("numDOFsPerEqn");
    int NNZ = args.scalar<int>("NNZ");
    xt::pyarray<int> &csrRowIndeces_DofLoops =
        args.array<int>("csrRowIndeces_DofLoops");
    xt::pyarray<int> &csrColumnOffsets_DofLoops =
        args.array<int>("csrColumnOffsets_DofLoops");
    xt::pyarray<double> &lumped_mass_matrix =
        args.array<double>("lumped_mass_matrix");
    double cfl_run = args.scalar<double>("cfl_run");
    double eps = args.scalar<double>("eps");
    double hEps = args.scalar<double>("hEps");
    xt::pyarray<double> &hnp1_at_quad_point =
        args.array<double>("hnp1_at_quad_point");
    xt::pyarray<double> &hunp1_at_quad_point =
        args.array<double>("hunp1_at_quad_point");
    xt::pyarray<double> &hvnp1_at_quad_point =
        args.array<double>("hvnp1_at_quad_point");
    xt::pyarray<double> &extendedSourceTerm_hu =
        args.array<double>("extendedSourceTerm_hu");
    xt::pyarray<double> &extendedSourceTerm_hv =
        args.array<double>("extendedSourceTerm_hv");
    xt::pyarray<double> &dH_minus_dL = args.array<double>("dH_minus_dL");
    xt::pyarray<double> &muH_minus_muL = args.array<double>("muH_minus_muL");
    double cE = args.scalar<double>("cE");
    int LUMPED_MASS_MATRIX = args.scalar<int>("LUMPED_MASS_MATRIX");
    // diagnostic: no bed in the star states (well-balanced dissipation off)
    int NO_STAR_STATES = args.scalar<int>("NO_STAR_STATES");
    double dt = args.scalar<double>("dt");
    int LINEAR_FRICTION = args.scalar<int>("LINEAR_FRICTION");
    // 1: local bounds from the neighbours' bar states only (ubar_ii = u_i
    // excluded), so an existing odd-even mode is not within bounds; the
    // low-order state is still added to the bounds below.
    int BOUNDS_EXCLUDE_SELF = args.scalar<int>("BOUNDS_EXCLUDE_SELF");
    int NODAL_H_BOUNDS = args.scalar<int>("NODAL_H_BOUNDS");
    double mannings = args.scalar<double>("mannings");
    // 1 on rows whose advective terms are integrated by parts (no-flux walls)
    xt::pyarray<double> &wall_dof = args.array<double>("wall_dof");
    xt::pyarray<double> &tailwater_dof = args.array<double>("tailwater_dof");
    xt::pyarray<double> &tailwater_depth_dof =
        args.array<double>("tailwater_depth_dof");
    // Canopy: porosity phi (storage phi h, flux phi h u, pressure
    // phi g h grad(h+Z)) and Darcy-Forchheimer drag -(alpha + beta|u|) h u per
    // unit plan area. All nodal; phi = 1, alpha = beta = 0 outside canopies.
    xt::pyarray<double> &porosity_dof = args.array<double>("porosity_dof");
    xt::pyarray<double> &drag_alpha_dof = args.array<double>("drag_alpha_dof");
    xt::pyarray<double> &drag_beta_dof = args.array<double>("drag_beta_dof");
    xt::pyarray<double> &quantDOFs = args.array<double>("quantDOFs");
    int SECOND_CALL_CALCULATE_RESIDUAL =
        args.scalar<int>("SECOND_CALL_CALCULATE_RESIDUAL");
    int COMPUTE_NORMALS = args.scalar<int>("COMPUTE_NORMALS");
    xt::pyarray<double> &normalx = args.array<double>("normalx");
    xt::pyarray<double> &normaly = args.array<double>("normaly");
    xt::pyarray<double> &dLow = args.array<double>("dLow");
    int lstage = args.scalar<int>("lstage");
    xt::pyarray<double> &new_SourceTerm_hu =
        args.array<double>("new_SourceTerm_hu");
    xt::pyarray<double> &new_SourceTerm_hv =
        args.array<double>("new_SourceTerm_hv");
    xt::pyarray<double> &global_entropy_residual =
        args.array<double>("global_entropy_residual");
    double dij_small = args.scalar<double>("dij_small");
    xt::pyarray<double> &hLow = args.array<double>("hLow");
    xt::pyarray<double> &huLow = args.array<double>("huLow");
    xt::pyarray<double> &hvLow = args.array<double>("hvLow");
    xt::pyarray<double> &h_min = args.array<double>("h_min");
    xt::pyarray<double> &h_max = args.array<double>("h_max");
    xt::pyarray<double> &kin_max = args.array<double>("kin_max");
    xt::pyarray<double> &urelax = args.array<double>("urelax");
    xt::pyarray<double> &drelax = args.array<double>("drelax");
    // FOR FRICTION//
    double n2 = std::pow(mannings, 2.);
    double gamma = 4. / 3;
    double xi = 10.;

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
          elementResidual_hv[nDOF_test_element];

      for (int i = 0; i < nDOF_test_element; i++) {
        elementResidual_h[i] = 0.0;
        elementResidual_hu[i] = 0.0;
        elementResidual_hv[i] = 0.0;
      }
      //
      // loop over quadrature points and compute integrands
      //
      for (int k = 0; k < nQuadraturePoints_element; k++) {
        // compute indices and declare local storage
        int eN_k = eN * nQuadraturePoints_element + k,
                     eN_k_nSpace = eN_k * nSpace,
                     eN_nDOF_trial_element = eN * nDOF_trial_element;
        double h = 0.0, hu = 0.0,
                        hv = 0.0,                    // solution at current time
            h_old = 0.0, hu_old = 0.0, hv_old = 0.0, // solution at lstage
            jac[nSpace * nSpace], jacDet, jacInv[nSpace * nSpace],
                        h_test_dV[nDOF_trial_element], dV, x, y, xt, yt;
        // get jacobian, etc for mapping reference element
        ck.calculateMapping_element(
            eN, k, mesh_dof.data(), mesh_l2g.data(), mesh_trial_ref.data(),
            mesh_grad_trial_ref.data(), jac, jacDet, jacInv, x, y);
        // get the physical integration weight, times the porosity (storage is
        // phi h, phi hu, phi hv)
        dV = fabs(jacDet) * dV_ref[k];
        {
          double phi_q = 0.;
          ck.valFromDOF(porosity_dof.data(),
                        &h_l2g.data()[eN_nDOF_trial_element],
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
        // get the solution at the lstage
        ck.valFromDOF(h_dof_old.data(), &h_l2g.data()[eN_nDOF_trial_element],
                      &h_trial_ref.data()[k * nDOF_trial_element], h_old);
        ck.valFromDOF(hu_dof_old.data(), &vel_l2g.data()[eN_nDOF_trial_element],
                      &vel_trial_ref.data()[k * nDOF_trial_element], hu_old);
        ck.valFromDOF(hv_dof_old.data(), &vel_l2g.data()[eN_nDOF_trial_element],
                      &vel_trial_ref.data()[k * nDOF_trial_element], hv_old);
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

        for (int i = 0; i < nDOF_test_element; i++) {
          // compute time derivative part of global residual. NOTE: no lumping
          elementResidual_h[i] += (h - h_old) * h_test_dV[i];
          elementResidual_hu[i] += (hu - hu_old) * h_test_dV[i];
          elementResidual_hv[i] += (hv - hv_old) * h_test_dV[i];
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
      }
    }
    // ********** END OF CELL LOOPS ********** //

    if (SECOND_CALL_CALCULATE_RESIDUAL == 0) // This is to save some time
    {
      /////////////////////////////////////////////////////
      // ********** FIRST SET OF LOOP ON DOFs ********** //
      /////////////////////////////////////////////////////
      // To compute:
      //     * Local bounds for limiting
      //     * Low order solution (in terms of bar states)

      // Walls: on rows with wall_dof = 1 the advective (inertial) terms are
      // integrated by parts with zero advective flux through the wall,
      //   R_i = -sum_j F_j . c_ji = sum_j (F_j - F_i) . c'_ij - F_i . n_i,
      // with c'_ij = -c_ji and n_i = sum_j c_ji = int_{dOmega} phi_i n.
      // F_adv(U).n = (u.n) U, so the boundary term only rescales U_i and the
      // low-order update stays a positive combination of U_i and bar states.
      // The pressure stays in the non-conservative, well-balanced form on
      // c_ij, which carries the wall's pressure reaction itself.
      std::valarray<double> wall_nx(0.0, numDOFsPerEqn),
          wall_ny(0.0, numDOFsPerEqn);
      {
        int ij_w = 0;
        for (int i = 0; i < numDOFsPerEqn; i++)
          for (int offset = csrRowIndeces_DofLoops[i];
               offset < csrRowIndeces_DofLoops[i + 1]; offset++) {
            if (wall_dof[i] > 0.5 || tailwater_dof[i] > 0.5) {
              wall_nx[i] += CTx[ij_w];
              wall_ny[i] += CTy[ij_w];
            }
            ij_w++;
          }
      }

      // Here we declare some arrays for local bounds. delta_Sqd_h and
      // delta_Sqd_kin (sum over neighbours of the difference) are computed
      // by the caller and exchanged with the ghost DOFs: bar_deltaSqd at row
      // i reads them at the neighbours, and a ghost row's own sum would miss
      // part of its stencil.
      xt::pyarray<double> &delta_Sqd_h = args.array<double>("delta_Sqd_h");
      xt::pyarray<double> &delta_Sqd_kin = args.array<double>("delta_Sqd_kin");
      std::valarray<double> bar_deltaSqd_h(0.0, numDOFsPerEqn),
          bar_deltaSqd_kin(0.0, numDOFsPerEqn);
      xt::pyarray<double> kin(numDOFsPerEqn), max_of_h_and_hEps(numDOFsPerEqn);

      // Define kinetic energy, kin = 1/2 q^2 / h
      max_of_h_and_hEps = xt::where(h_dof_old > hEps, h_dof_old, hEps);
      kin = 0.5 * (hu_dof_old * hu_dof_old + hv_dof_old * hv_dof_old);
      kin = kin * (2.0 * h_dof_old /
                   (h_dof_old * h_dof_old + max_of_h_and_hEps * max_of_h_and_hEps));

      // Stuff for bar states here (BT = BarTilde)
      std::valarray<double> hBT(0.0, dH_minus_dL.size()),
          huBT(0.0, dH_minus_dL.size()), hvBT(0.0, dH_minus_dL.size());

      /* Second loop to compute bar states (variable)BT */
      int ij = 0;
      for (int i = 0; i < numDOFsPerEqn; i++) {

        // define things at ith node
        double hi = h_dof_old[i];
        double hui = hu_dof_old[i];
        double hvi = hv_dof_old[i];
        double Zi = b_dof[i];
        double mi = lumped_mass_matrix[i];
        double one_over_hiReg =
            2.0 * hi / (hi * hi + std::pow(fmax(hi, hEps), 2));
        double ui = hui * one_over_hiReg;
        double vi = hvi * one_over_hiReg;

        // define full pressure at ith node for definition of bar states
        double pressure_i = 0.5 * g * hi * hi;
        double phi_i = porosity_dof[i];

        // loop over the sparsity pattern of the i-th DOF
        for (int offset = csrRowIndeces_DofLoops[i];
             offset < csrRowIndeces_DofLoops[i + 1]; offset++) {
          int j = csrColumnOffsets_DofLoops[offset];

          // define things at jth node
          double hj = h_dof_old[j];
          double huj = hu_dof_old[j];
          double hvj = hv_dof_old[j];
          double Zj = b_dof[j];
          double one_over_hjReg =
              2.0 * hj / (hj * hj + std::pow(fmax(hj, hEps), 2));
          double uj = huj * one_over_hjReg;
          double vj = hvj * one_over_hjReg;
          // advective coefficient for row i: c'_ij = -c_ji on wall rows
          double cxa = wall_dof[i] > 0.5 ? -CTx[ij] : Cx[ij];
          double cya = wall_dof[i] > 0.5 ? -CTy[ij] : Cy[ij];

          // for bar_deltaSqd_h, bar_deltaSqd_heta, bar_deltaSqd_kin
          double muLowij = 0., muLij = 0., dLowij = 0., dLij = 0.;

          if (i != j) {
            // Put these computations first  before it gets messy
            bar_deltaSqd_h[i] += 0.5 * delta_Sqd_h[j] + 0.5 * delta_Sqd_h[i];
            bar_deltaSqd_kin[i] +=
                0.5 * delta_Sqd_kin[j] + 0.5 * delta_Sqd_kin[i];

            if (lstage == 0)
              dLowij = dLow[ij];
            else {
              double cij_norm = sqrt(Cx[ij] * Cx[ij] + Cy[ij] * Cy[ij]);
              double cji_norm = sqrt(CTx[ij] * CTx[ij] + CTy[ij] * CTy[ij]);
              double nxij = Cx[ij] / cij_norm, nyij = Cy[ij] / cij_norm;
              double nxji = CTx[ij] / cji_norm, nyji = CTy[ij] / cji_norm;
              dLowij = fmax(
                  maxWaveSpeedSharpInitialGuess(g, nxij, nyij, hi, hui, hvi, hj,
                                                huj, hvj, hEps, false) *
                      cij_norm,
                  maxWaveSpeedSharpInitialGuess(g, nxji, nyji, hj, huj, hvj, hi,
                                                hui, hvi, hEps, false) *
                      cji_norm);
              dLowij *= phiBar(i, j, porosity_dof);
            }
            // save dLij
            dLij = dLowij;

            // compute muij
            muLij = fmax(fmax(0., -(ui * cxa + vi * cya)),
                         fmax(0., (uj * cxa + vj * cya)));
            if (wall_dof[i] > 0.5 || wall_dof[j] > 0.5)
              muLij = fmax(muLij, muWallTranspose(i, j, ij, ui, vi, uj, vj,
                                                  wall_dof, Cx, Cy, CTx, CTy));
            muLij *= phiBar(i, j, porosity_dof);

            // Define dLij as max of dLij and muLij
            dLij = fmax(dLowij, muLij);
            dLow[ij] = fmax(dLij, muLij);

            ////////////////////////
            // COMPUTE BAR STATES //
            ////////////////////////

            // define pressure at jth node for bar states
            double pressure_j = 0.5 * g * hj * hj;

            // Compute star states
            double hStarij = NO_STAR_STATES ? hi : fmax(0., hi + Zi - fmax(Zi, Zj));
            double huStarij = hui * hStarij * one_over_hiReg;
            double hvStarij = hvi * hStarij * one_over_hiReg;
            double hStarji = NO_STAR_STATES ? hj : fmax(0., hj + Zj - fmax(Zi, Zj));
            double huStarji = huj * hStarji * one_over_hjReg;
            double hvStarji = hvj * hStarji * one_over_hjReg;

            double hBar_ij = 0., hTilde_ij = 0., huBar_ij = 0., huTilde_ij = 0.,
                   hvBar_ij = 0., hvTilde_ij = 0.;

            // h component (advective fluxes carry phi_j F_j, pressure phi_i)
            double phi_j = porosity_dof[j];
            hBar_ij = -1. / (2.0 * fmax(dLij, dij_small)) *
                          ((phi_j * uj * hj - phi_i * ui * hi) * cxa +
                           (phi_j * vj * hj - phi_i * vi * hi) * cya) +
                      0.5 * (hj + hi);
            hTilde_ij = (dLij - muLij) / (2.0 * fmax(dLij, dij_small)) *
                        (hStarji - hj - (hStarij - hi));
            // hu component
            huBar_ij =
                -1. / (2.0 * fmax(dLij, dij_small)) *
                    ((phi_j * uj * huj - phi_i * ui * hui) * cxa +
                     (phi_j * vj * huj - phi_i * vi * hui) * cya +
                     phi_i * (pressure_j - pressure_i) * Cx[ij]) +
                0.5 * (huj + hui);
            huTilde_ij = (dLij - muLij) / (2.0 * fmax(dLij, dij_small)) *
                         (huStarji - huj - (huStarij - hui));
            // hv component
            hvBar_ij =
                -1. / (2.0 * fmax(dLij, dij_small)) *
                    ((phi_j * uj * hvj - phi_i * ui * hvi) * cxa +
                     (phi_j * vj * hvj - phi_i * vi * hvi) * cya +
                     phi_i * (pressure_j - pressure_i) * Cy[ij]) +
                0.5 * (hvj + hvi);
            hvTilde_ij = (dLij - muLij) / (2.0 * fmax(dLij, dij_small)) *
                         (hvStarji - hvj - (hvStarij - hvi));

            // Here we define uBar + uTilde
            hBT[ij] = hBar_ij + hTilde_ij;
            huBT[ij] = huBar_ij + huTilde_ij;
            hvBT[ij] = hvBar_ij + hvTilde_ij;
          } else {
            // i==j
            // Bar states by definition satisfy Utilde_ii + Ubar_ii = U_i
            hBT[ij] = hi;
            huBT[ij] = hui;
            hvBT[ij] = hvi;
          }

          // UPDATE ij //
          ij += 1;
        } // j loop ends here

        // for bar_deltaSqd_h, bar_deltaSqd_heta, bar_deltaSqd_kin
        bar_deltaSqd_h[i] =
            bar_deltaSqd_h[i] /
            (csrRowIndeces_DofLoops[i + 1] - csrRowIndeces_DofLoops[i] - 1) /
            2.0;
        bar_deltaSqd_kin[i] =
            bar_deltaSqd_kin[i] /
            (csrRowIndeces_DofLoops[i + 1] - csrRowIndeces_DofLoops[i] - 1) /
            2.0;

      } // i loops ends here

      /* Then final loop of first set to get local bounds */
      ij = 0;
      for (int i = 0; i < numDOFsPerEqn; i++) {

        // define m_i
        double mi = lumped_mass_matrix[i];

        /* Initialize hmin, hmax */
        h_min[i] = h_dof_old[i];
        h_max[i] = h_dof_old[i];

        /* Initialize low order solution */
        hLow[i] = h_dof_old[i];
        huLow[i] = hu_dof_old[i];
        hvLow[i] = hv_dof_old[i];
        kin_max[i] = kin[i];
        if (BOUNDS_EXCLUDE_SELF) {
          h_min[i] = 1.0e300;
          h_max[i] = -1.0e300;
          kin_max[i] = 0.0;
        }

        // loop in j (sparsity pattern)
        for (int offset = csrRowIndeces_DofLoops[i];
             offset < csrRowIndeces_DofLoops[i + 1]; offset++) {

          int j = csrColumnOffsets_DofLoops[offset];

          double one_over_hBT =
              2.0 * hBT[ij] /
              (hBT[ij] * hBT[ij] + std::pow(fmax(hBT[ij], hEps), 2));
          double psi_ij = one_over_hBT *
                          (huBT[ij] * huBT[ij] + hvBT[ij] * hvBT[ij]) /
                          2.0; // Eqn (6.31)

          // COMPUTE LOCAL BOUNDS //
          if (!(BOUNDS_EXCLUDE_SELF && j == i)) {
            kin_max[i] = fmax(psi_ij, kin_max[i]);
            if (!NODAL_H_BOUNDS) {
              h_min[i] = std::min(h_min[i], hBT[ij]);
              h_max[i] = std::max(h_max[i], hBT[ij]);
            }
          }
          // diagnostic: bounds on h from the nodal values of the stencil
          // (local extremum diminishing test), no bar states, no relaxation
          if (NODAL_H_BOUNDS) {
            h_min[i] = std::min(h_min[i], h_dof_old[j]);
            h_max[i] = std::max(h_max[i], h_dof_old[j]);
          }

          /* COMPUTE LOW ORDER SOLUTION. See EQN 6.23 in SW friction paper */
          // This is low order solution WITHOUT sources
          if (i != j) {
            hLow[i] += h_dof_old[i] * (-dt / mi * 2 * dLow[ij]) +
                       dt / mi * (2 * dLow[ij] * hBT[ij]);
            huLow[i] += hu_dof_old[i] * (-dt / mi * 2 * dLow[ij]) +
                        dt / mi * (2 * dLow[ij] * huBT[ij]);
            hvLow[i] += hv_dof_old[i] * (-dt / mi * 2 * dLow[ij]) +
                        dt / mi * (2 * dLow[ij] * hvBT[ij]);
          }

          // UPDATE ij //
          ij += 1;
        } // j loop ends here

        // wall rows: + dt/m_i (u_i . n_i) U_i from the advective boundary term
        if (wall_dof[i] > 0.5) {
          double one_over_hiReg_w =
              2.0 * h_dof_old[i] /
              (h_dof_old[i] * h_dof_old[i] +
               std::pow(fmax(h_dof_old[i], hEps), 2));
          double un_i = (hu_dof_old[i] * wall_nx[i] + hv_dof_old[i] * wall_ny[i]) *
                        one_over_hiReg_w;
          un_i *= porosity_dof[i]; // boundary flux phi_i (u_i.n_i) U_i
          hLow[i] += dt / mi * un_i * h_dof_old[i];
          huLow[i] += dt / mi * un_i * hu_dof_old[i];
          hvLow[i] += dt / mi * un_i * hv_dof_old[i];
          if (wall_dof[i] > 1.5) {
            double dp_w, rate_w;
            wallRiemann(g, h_dof_old[i], hu_dof_old[i], hv_dof_old[i],
                        wall_nx[i], wall_ny[i], hEps, dp_w, rate_w);
            huLow[i] -= dt / mi * porosity_dof[i] * dp_w * wall_nx[i];
            hvLow[i] -= dt / mi * porosity_dof[i] * dp_w * wall_ny[i];
          }
          // the rescaled U_i makes the low-order state a positive, not
          // convex, combination; widen the local bounds to contain it
          h_min[i] = std::min(h_min[i], hLow[i]);
          h_max[i] = std::max(h_max[i], hLow[i]);
          const double one_over_hLow =
              2.0 * hLow[i] /
              (hLow[i] * hLow[i] + std::pow(fmax(hLow[i], hEps), 2));
          kin_max[i] = fmax(kin_max[i],
                            0.5 * one_over_hLow *
                                (huLow[i] * huLow[i] + hvLow[i] * hvLow[i]));
        }

        // tailwater rows: Rusanov flux against the ghost state, as a
        // correction to the natural outflow carried by c_ij
        if (tailwater_dof[i] > 0.5) {
          double dFh_t, dFhu_t, dFhv_t, rate_t;
          tailwaterFlux(g, h_dof_old[i], hu_dof_old[i], hv_dof_old[i],
                        tailwater_depth_dof[i], wall_nx[i], wall_ny[i], hEps,
                        dFh_t, dFhu_t, dFhv_t, rate_t);
          hLow[i] -= dt / mi * porosity_dof[i] * dFh_t;
          huLow[i] -= dt / mi * porosity_dof[i] * dFhu_t;
          hvLow[i] -= dt / mi * porosity_dof[i] * dFhv_t;
          hLow[i] = fmax(hLow[i], 0.);
          h_min[i] = std::min(h_min[i], hLow[i]);
          h_max[i] = std::max(h_max[i], hLow[i]);
          const double one_over_hLow_t =
              2.0 * hLow[i] /
              (hLow[i] * hLow[i] + std::pow(fmax(hLow[i], hEps), 2));
          kin_max[i] = fmax(kin_max[i],
                            0.5 * one_over_hLow_t *
                                (huLow[i] * huLow[i] + hvLow[i] * hvLow[i]));
        }

        // Then do relaxation of bounds here. If confused, see convex
        // limiting paper
        kin_max[i] = std::min(urelax[i] * kin_max[i],
                              kin_max[i] + std::abs(bar_deltaSqd_kin[i]));
        if (!NODAL_H_BOUNDS) {
          h_min[i] = std::max(drelax[i] * h_min[i],
                              h_min[i] - std::abs(bar_deltaSqd_h[i]));
          h_max[i] = std::min(urelax[i] * h_max[i],
                              h_max[i] + std::abs(bar_deltaSqd_h[i]));
        }

        // clean up hLow from round off error
        if (hLow[i] < hEps)
          hLow[i] = 0.0;
      } // i loop ends here

      ////////////////////////////////////////////////////////
      // ********** Second set of loops on dofs ********** //
      ///////////////////////////////////////////////////////
      // To compute:
      //     * Hyperbolic part of the flux
      //     * Extended source terms
      //     * Smoothness indicator

      ij = 0;
      std::valarray<double> hyp_flux_h(numDOFsPerEqn),
          hyp_flux_hu(numDOFsPerEqn), hyp_flux_hv(numDOFsPerEqn),
          psi(numDOFsPerEqn), etaMax(numDOFsPerEqn), etaMin(numDOFsPerEqn);

      for (int i = 0; i < numDOFsPerEqn; i++) {
        // solution at time tn for the ith DOF
        double hi = h_dof_old[i];
        double hui = hu_dof_old[i];
        double hvi = hv_dof_old[i];
        double Zi = b_dof[i];
        // Define some things using above
        double one_over_hiReg =
            2 * hi / (hi * hi + std::pow(fmax(hi, hEps), 2)); // hEps
        double ui = hui * one_over_hiReg;
        double vi = hvi * one_over_hiReg;
        double mi = lumped_mass_matrix[i];

        /* COMPUTE EXTENDED SOURCE TERMS for all equations:
         * Friction terms
         * NOTE: Be careful with sign of source terms. Extended sources are on
         * left side of equations. "new_SourceTerm" variables are on right
         */

        double phi_i = porosity_dof[i];

        // FRICTION
        if (LINEAR_FRICTION == 1) {
          extendedSourceTerm_hu[i] = mannings * hui * mi;
          extendedSourceTerm_hv[i] = mannings * hvi * mi;
          // For use in the convex limiting function
          // actually didn't need to do this but it helps with signs
          new_SourceTerm_hu[i] = -mannings * hui * mi;
          new_SourceTerm_hv[i] = -mannings * hvi * mi;
        } else {
          double veli_norm = std::sqrt(ui * ui + vi * vi);
          double hi_to_the_gamma = std::pow(fmax(hi, hEps), gamma);
          double friction_aux =
              veli_norm == 0.
                  ? 0.
                  : (2 * g * n2 * veli_norm * mi /
                     (hi_to_the_gamma +
                      fmax(hi_to_the_gamma, xi * g * n2 * dt * veli_norm)));
          extendedSourceTerm_hu[i] = friction_aux * hui;
          extendedSourceTerm_hv[i] = friction_aux * hvi;
          // For use in the convex limiting function
          new_SourceTerm_hu[i] = -friction_aux * hui;
          new_SourceTerm_hv[i] = -friction_aux * hvi;
        }

        // CANOPY DRAG: -(alpha + beta |u|) h u per unit plan area, i.e. a
        // sink K hu per unit storage with K = (alpha + beta |u|) / phi. Same
        // semi-implicit form as the Manning term (2 K m / (1 + max(1, xi dt
        // K))), so dt * drag_aux / m_i <= 2 / xi and hu never changes sign.
        {
          double veli_norm = std::sqrt(ui * ui + vi * vi);
          double K = (drag_alpha_dof[i] + drag_beta_dof[i] * veli_norm) / phi_i;
          if (K > 0.) {
            double drag_aux = 2. * K * mi / (1. + fmax(1., xi * dt * K));
            extendedSourceTerm_hu[i] += drag_aux * hui;
            extendedSourceTerm_hv[i] += drag_aux * hvi;
            new_SourceTerm_hu[i] -= drag_aux * hui;
            new_SourceTerm_hv[i] -= drag_aux * hvi;
          }
        }

        /* HYPERBOLIC FLUXES */
        hyp_flux_h[i] = 0;
        hyp_flux_hu[i] = 0;
        hyp_flux_hv[i] = 0;

        // FOR SMOOTHNESS INDICATOR //
        double alphai;
        double alpha_numerator = 0;
        double alpha_denominator = 0;
        double alpha_zero = 0.5; // if only want smoothness
        double alpha_factor = 1.0 / (1.0 - alpha_zero);

        // loop in j (sparsity pattern)
        for (int offset = csrRowIndeces_DofLoops[i];
             offset < csrRowIndeces_DofLoops[i + 1]; offset++) {

          int j = csrColumnOffsets_DofLoops[offset];

          // solution at time tn for the jth DOF
          double hj = h_dof_old[j];
          double huj = hu_dof_old[j];
          double hvj = hv_dof_old[j];
          double Zj = b_dof[j];
          // Then define some things here using above
          double one_over_hjReg =
              2.0 * hj / (hj * hj + std::pow(fmax(hj, hEps), 2));
          double uj = huj * one_over_hjReg;
          double vj = hvj * one_over_hjReg;

          // auxiliary functions to compute fluxes (advective part on c')
          double cxa = wall_dof[i] > 0.5 ? -CTx[ij] : Cx[ij];
          double cya = wall_dof[i] > 0.5 ? -CTy[ij] : Cy[ij];
          double phi_j = porosity_dof[j];
          double aux_h = (phi_j * uj * hj - phi_i * ui * hi) * cxa +
                         (phi_j * vj * hj - phi_i * vi * hi) * cya;
          double aux_hu = (phi_j * uj * huj - phi_i * ui * hui) * cxa +
                          (phi_j * vj * huj - phi_i * vi * hui) * cya;
          double aux_hv = (phi_j * uj * hvj - phi_i * ui * hvi) * cxa +
                          (phi_j * vj * hvj - phi_i * vi * hvi) * cya;

          /* HYPERBOLIC FLUX */
          hyp_flux_h[i] += aux_h;
          hyp_flux_hu[i] += aux_hu;
          hyp_flux_hv[i] += aux_hv;

          // EXTENDED SOURCE, USING 6.13 //
          // (porous: phi_i g h_i grad(h + Z), non-conservative at node i)
          extendedSourceTerm_hu[i] += phi_i * g * hi * (hj + Zj) * Cx[ij];
          extendedSourceTerm_hv[i] += phi_i * g * hi * (hj + Zj) * Cy[ij];

          new_SourceTerm_hu[i] += phi_i * g *
                                  (-hi * (Zj - Zi) + 0.5 * std::pow(hj - hi, 2)) *
                                  Cx[ij];
          new_SourceTerm_hv[i] += phi_i * g *
                                  (-hi * (Zj - Zi) + 0.5 * std::pow(hj - hi, 2)) *
                                  Cy[ij];

          // FOR SMOOTHNESS INDICATOR //
          alpha_numerator += hj - hi;
          alpha_denominator += fabs(hj - hi);

          // update ij
          ij += 1;
        } // end j loop

        // wall rows: advective boundary term -F_i . n_i = -(u_i . n_i) U_i
        if (wall_dof[i] > 0.5) {
          double un_i = phi_i * (ui * wall_nx[i] + vi * wall_ny[i]);
          hyp_flux_h[i] -= un_i * hi;
          hyp_flux_hu[i] -= un_i * hui;
          hyp_flux_hv[i] -= un_i * hvi;
          if (wall_dof[i] > 1.5) {
            double dp_w, rate_w;
            wallRiemann(g, hi, hui, hvi, wall_nx[i], wall_ny[i], hEps, dp_w,
                        rate_w);
            hyp_flux_hu[i] += phi_i * dp_w * wall_nx[i];
            hyp_flux_hv[i] += phi_i * dp_w * wall_ny[i];
          }
        }

        // tailwater rows: same boundary-flux correction in the high-order update
        if (tailwater_dof[i] > 0.5) {
          double dFh_t, dFhu_t, dFhv_t, rate_t;
          tailwaterFlux(g, hi, hui, hvi, tailwater_depth_dof[i], wall_nx[i],
                        wall_ny[i], hEps, dFh_t, dFhu_t, dFhv_t, rate_t);
          hyp_flux_h[i] += phi_i * dFh_t;
          hyp_flux_hu[i] += phi_i * dFhu_t;
          hyp_flux_hv[i] += phi_i * dFhv_t;
        }

        // COMPUTE SMOOTHNESS INDICATOR //
        if (hi <= hEps) {
          alphai = 1.0;
          psi[i] = 1.0;
        } else {
          // Force alphai=0 in constant states
          if (fabs(alpha_numerator) <= hEps) {
            alphai = 0.;
          } else {
            alphai =
                (fabs(alpha_numerator) - hEps) / fabs(alpha_denominator - hEps);
          }
          alphai = fmax(alphai - alpha_zero, 0.0) * alpha_factor;
          if (POWER_SMOOTHNESS_INDICATOR == 0)
            psi[i] = 1.0;
          else
            psi[i] = std::pow(alphai, POWER_SMOOTHNESS_INDICATOR);
        }
      }
      // ********** END OF 2nd LOOP ON DOFS ********** //

      /////////////////////////////////////////////
      // ********** MAIN LOOP ON DOFs **********
      // To compute:
      //      * dissipative terms
      //      * bar states
      /////////////////////////////////////////////

      ij = 0;
      for (int i = 0; i < numDOFsPerEqn; i++) {
        double hi = h_dof_old[i];
        double hui = hu_dof_old[i];
        double hvi = hv_dof_old[i];
        double Zi = b_dof[i];
        double mi = lumped_mass_matrix[i];

        double one_over_hiReg =
            2.0 * hi / (hi * hi + std::pow(fmax(hi, hEps), 2));
        double ui = hui * one_over_hiReg;
        double vi = hvi * one_over_hiReg;

        // Define full pressure at ith node for definition of bar states below
        double pressure_i = 0.5 * g * hi * hi;

        // HIGH ORDER DISSIPATIVE TERMS, for Aij matrix
        double ith_dHij_minus_muHij_times_hStarStates = 0.,
               ith_dHij_minus_muHij_times_huStarStates = 0.,
               ith_dHij_minus_muHij_times_hvStarStates = 0.,
               ith_muHij_times_hStates = 0., ith_muHij_times_huStates = 0.,
               ith_muHij_times_hvStates = 0.;

        // loop over the sparsity pattern of the i-th DOF
        for (int offset = csrRowIndeces_DofLoops[i];
             offset < csrRowIndeces_DofLoops[i + 1]; offset++) {
          int j = csrColumnOffsets_DofLoops[offset];
          double hj = h_dof_old[j];
          double huj = hu_dof_old[j];
          double hvj = hv_dof_old[j];
          double Zj = b_dof[j];
          double one_over_hjReg =
              2.0 * hj / (hj * hj + std::pow(fmax(hj, hEps), 2));
          double uj = huj * one_over_hjReg;
          double vj = hvj * one_over_hjReg;

          // define pressure at jth node
          double pressure_j = 0.5 * g * hj * hj;

          // COMPUTE STAR STATES
          double hStarij = NO_STAR_STATES ? hi : fmax(0., hi + Zi - fmax(Zi, Zj));
          double huStarij = hui * hStarij * one_over_hiReg;
          double hvStarij = hvi * hStarij * one_over_hiReg;

          double hStarji = NO_STAR_STATES ? hj : fmax(0., hj + Zj - fmax(Zi, Zj));
          double huStarji = huj * hStarji * one_over_hjReg;
          double hvStarji = hvj * hStarji * one_over_hjReg;

          // Dissipative well balancing term
          double muLowij = 0., muLij = 0., muHij = 0.;
          double dLowij = 0., dLij = 0., dHij = 0.;
          if (i != j) // This is not necessary. See formula for
                      // ith_dissipative_terms
          {
            ////////////////////////
            // DISSIPATIVE MATRIX //
            ////////////////////////
            if (lstage == 0)
              dLowij = dLow[ij];
            else {
              double cij_norm = sqrt(Cx[ij] * Cx[ij] + Cy[ij] * Cy[ij]);
              double cji_norm = sqrt(CTx[ij] * CTx[ij] + CTy[ij] * CTy[ij]);
              double nxij = Cx[ij] / cij_norm, nyij = Cy[ij] / cij_norm;
              double nxji = CTx[ij] / cji_norm, nyji = CTy[ij] / cji_norm;
              dLowij = fmax(
                  maxWaveSpeedSharpInitialGuess(g, nxij, nyij, hi, hui, hvi, hj,
                                                huj, hvj, hEps, false) *
                      cij_norm,
                  maxWaveSpeedSharpInitialGuess(g, nxji, nyji, hj, huj, hvj, hi,
                                                hui, hvi, hEps, false) *
                      cji_norm);
              dLowij *= phiBar(i, j, porosity_dof);
            }
            // this is standard low-order dij, can you use dLij =
            // dLowij*fmax(psi[i],psi[j]) if want smoothness based as low order
            dLij = dLowij;

            ///////////////////////////////////////
            // WELL BALANCING DISSIPATIVE MATRIX //
            ///////////////////////////////////////
            // advective coefficient for row i: c'_ij = -c_ji on wall rows
            double cxa = wall_dof[i] > 0.5 ? -CTx[ij] : Cx[ij];
            double cya = wall_dof[i] > 0.5 ? -CTy[ij] : Cy[ij];
            muLowij = fmax(fmax(0., -(ui * cxa + vi * cya)),
                           fmax(0., (uj * cxa + vj * cya)));
            if (wall_dof[i] > 0.5 || wall_dof[j] > 0.5)
              muLowij = fmax(muLowij,
                             muWallTranspose(i, j, ij, ui, vi, uj, vj, wall_dof,
                                             Cx, Cy, CTx, CTy));
            muLowij *= phiBar(i, j, porosity_dof);
            muLij = muLowij;

            // Define dLij as low order dijs
            muLij = muLowij;
            dLij = fmax(dLowij, muLij);

            // Then save dLow for limiting step, maybe a bit confusing
            dLow[ij] = fmax(dLij, muLij);

            ///////////////////////
            // ENTROPY VISCOSITY //
            ///////////////////////
            double dEVij = cE * fmax(global_entropy_residual[i],
                                     global_entropy_residual[j]) *
                           phiBar(i, j, porosity_dof);
            dHij = fmin(dLowij, dEVij);
            muHij = fmin(muLowij, dEVij);

            // compute dij_minus_muij times star solution terms
            // see: eqn (6.13)
            ith_dHij_minus_muHij_times_hStarStates +=
                (dHij - muHij) * (hStarji - hStarij);
            ith_dHij_minus_muHij_times_huStarStates +=
                (dHij - muHij) * (huStarji - huStarij);
            ith_dHij_minus_muHij_times_hvStarStates +=
                (dHij - muHij) * (hvStarji - hvStarij);

            // compute muij times solution terms
            ith_muHij_times_hStates += muHij * (hj - hi);
            ith_muHij_times_huStates += muHij * (huj - hui);
            ith_muHij_times_hvStates += muHij * (hvj - hvi);

            // compute dH_minus_dL
            dH_minus_dL[ij] = dHij - dLij;
            muH_minus_muL[ij] = muHij - muLij;
          } else // i==j
          {
            dH_minus_dL[ij] =
                0.; // Not true but the prod of this times Uj-Ui will be zero
            muH_minus_muL[ij] =
                0.; // Not true but the prod of this times Uj-Ui will be zero
          }
          // update ij
          ij += 1;
        } // j loop ends here

        /* Define global residual */
        if (LUMPED_MASS_MATRIX == 1) {
          globalResidual[offset_h + stride_h * i] =
              hi - dt / mi *
                       (hyp_flux_h[i] - ith_dHij_minus_muHij_times_hStarStates -
                        ith_muHij_times_hStates);
          globalResidual[offset_hu + stride_hu * i] =
              hui - dt / mi *
                        ((hyp_flux_hu[i] + extendedSourceTerm_hu[i]) -
                         ith_dHij_minus_muHij_times_huStarStates -
                         ith_muHij_times_huStates);
          globalResidual[offset_hv + stride_hv * i] =
              hvi - dt / mi *
                        ((hyp_flux_hv[i] + extendedSourceTerm_hv[i]) -
                         ith_dHij_minus_muHij_times_hvStarStates -
                         ith_muHij_times_hvStates);
          // clean up potential negative water height due to machine precision
          if (globalResidual[offset_h + stride_h * i] >= -hEps &&
              globalResidual[offset_h + stride_h * i] < hEps)
            globalResidual[offset_h + stride_h * i] = 0;
        } else {
          // Distribute residual
          // NOTE: MASS MATRIX IS CONSISTENT
          globalResidual[offset_h + stride_h * i] +=
              dt * (hyp_flux_h[i] - ith_dHij_minus_muHij_times_hStarStates -
                    ith_muHij_times_hStates);
          globalResidual[offset_hu + stride_hu * i] +=
              dt * (hyp_flux_hu[i] + extendedSourceTerm_hu[i] -
                    ith_dHij_minus_muHij_times_huStarStates -
                    ith_muHij_times_huStates);
          globalResidual[offset_hv + stride_hv * i] +=
              dt * (hyp_flux_hv[i] + extendedSourceTerm_hv[i] -
                    ith_dHij_minus_muHij_times_hvStarStates -
                    ith_muHij_times_hvStates);
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
          elementJacobian_hv_hv[nDOF_test_element][nDOF_trial_element];
      for (int i = 0; i < nDOF_test_element; i++)
        for (int j = 0; j < nDOF_trial_element; j++) {
          elementJacobian_h_h[i][j] = 0.0;
          elementJacobian_hu_hu[i][j] = 0.0;
          elementJacobian_hv_hv[i][j] = 0.0;
        }
      for (int k = 0; k < nQuadraturePoints_element; k++) {
        int eN_k = eN * nQuadraturePoints_element +
                   k, // index to a scalar at a quadrature point
            eN_k_nSpace = eN_k * nSpace,
            eN_nDOF_trial_element =
                eN *
                nDOF_trial_element; // index to a vector at a quadrature point

        // declare local storage
        double jac[nSpace * nSpace], jacDet, jacInv[nSpace * nSpace],
            dV, h_test_dV[nDOF_test_element], vel_test_dV[nDOF_test_element], x,
            y, xt, yt;
        // get jacobian, etc for mapping reference element
        ck.calculateMapping_element(
            eN, k, mesh_dof.data(), mesh_l2g.data(), mesh_trial_ref.data(),
            mesh_grad_trial_ref.data(), jac, jacDet, jacInv, x, y);
        // get the physical integration weight, times the porosity: storage
        // is phi h and phi hu, so the mass matrix is int phi_h w_i w_j
        dV = fabs(jacDet) * dV_ref[k];
        {
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
          elementJacobian_hv_hv[nDOF_test_element][nDOF_trial_element];
      for (int i = 0; i < nDOF_test_element; i++)
        for (int j = 0; j < nDOF_trial_element; j++) {
          elementJacobian_h_h[i][j] = 0.0;
          elementJacobian_hu_hu[i][j] = 0.0;
          elementJacobian_hv_hv[i][j] = 0.0;
        }
      for (int k = 0; k < nQuadraturePoints_element; k++) {
        int eN_k = eN * nQuadraturePoints_element +
                   k, // index to a scalar at a quadrature point
            eN_k_nSpace = eN_k * nSpace,
            eN_nDOF_trial_element =
                eN *
                nDOF_trial_element; // index to a vector at a quadrature point

        // declare local storage
        double jac[nSpace * nSpace], jacDet, jacInv[nSpace * nSpace],
            dV, h_test_dV[nDOF_test_element], vel_test_dV[nDOF_test_element], x,
            y, xt, yt;
        // get jacobian, etc for mapping reference element
        ck.calculateMapping_element(
            eN, k, mesh_dof.data(), mesh_l2g.data(), mesh_trial_ref.data(),
            mesh_grad_trial_ref.data(), jac, jacDet, jacInv, x, y);
        // get the physical integration weight, times the porosity: storage
        // is phi h and phi hu, so the mass matrix is int phi_h w_i w_j
        dV = fabs(jacDet) * dV_ref[k];
        {
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
        } // j
      }   // i
    }     // elements
  }
}; // namespace proteus

inline SW2DCV_base *newSW2DCV(int nSpaceIn, int nQuadraturePoints_elementIn,
                              int nDOF_mesh_trial_elementIn,
                              int nDOF_trial_elementIn, int nDOF_test_elementIn,
                              int nQuadraturePoints_elementBoundaryIn,
                              int CompKernelFlag) {
  return proteus::chooseAndAllocateDiscretization2D<SW2DCV_base, SW2DCV,
                                                    CompKernel>(
      nSpaceIn, nQuadraturePoints_elementIn, nDOF_mesh_trial_elementIn,
      nDOF_trial_elementIn, nDOF_test_elementIn,
      nQuadraturePoints_elementBoundaryIn, CompKernelFlag);
}
} // end namespace proteus

#endif
