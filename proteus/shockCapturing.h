#ifndef SHOCKCAPTURING_H
#define SHOCKCAPTURING_H
#include "index_type.h"
/*!
 \file shockCapturing.h
 \brief C implementations of shock capturing diffusion calculations.
*/

/**
   \defgroup shockCapturing shockCapturing
 \brief C implementations of shock capturing diffusion calculations
 @{
*/
extern void calculateNumericalDiffusionResGrad(proteus_index_t nElements_global,
                                               int nQuadraturePoints_element,
                                               int nSpace,
                                               double shockCapturingDiffusion,
                                               double* elementDiameter,
                                               double* strong_residual,
                                               double* grad_u,
                                               double* numDiff);
extern void calculateNumericalDiffusionResGradQuad(proteus_index_t nElements_global,
                                                   int nQuadraturePoints_element,
                                                   int nSpace,
                                                   double shockCapturingDiffusion,
                                                   double *elementDiameter,
                                                   double *strong_residual,
                                                   double *grad_u,
                                                   double *numDiff);
extern void calculateNumericalDiffusionHJ(proteus_index_t nElements_global,
                                          int nQuadraturePoints_element,
                                          char shockCapturing,
                                          double shockCapturingDiffusion,
                                          double* elementDiameter,
                                          double* strong_residual,
                                          double* mt,
                                          double* H,
                                          double* numDiff);
extern void calculateNumericalDiffusionHJV2(proteus_index_t nElements_global,
                                            int nQuadraturePoints_element,
                                            char shockCapturing,
                                            double shockCapturingDiffusion,
                                            double* elementDiameter,
                                            double* strong_residual,
                                            double* mt,
                                            double* H,
                                            double* numDiff);
extern void calculateNumericalDiffusion_A_1(proteus_index_t nElements_global,
                                            int nQuadraturePoints_element,
                                            int nSpace,
                                            double shockCapturingFactor,
                                            double* elementDiameter,
                                            double* strong_residual,
                                            double* mt,
                                            double* df,
                                            double* numDiff);
extern void calculateNumericalDiffusionResGradJuanes(proteus_index_t nElements_global,
                                                     int nQuadraturePoints_element,
                                                     int nSpace,
                                                     double shockCapturingDiffusion,
                                                     double uSC,
                                                     double *elementDiameter,
                                                     double *strong_residual,
                                                     double *grad_u,
                                                     double *numDiff);
extern void calculateNumericalDiffusionResGrad(proteus_index_t nElements_global,
                                        int nQuadraturePoints_element,
                                        int nSpace,
                                        double shockCapturingDiffusion,
                                        double* elementDiameter,
                                        double* strong_residual,
                                        double* grad_u,
                                        double* numDiff);
extern void calculateNumericalDiffusionResGradQuad(proteus_index_t nElements_global,
                                            int nQuadraturePoints_element,
                                            int nSpace,
                                            double shockCapturingDiffusion,
                                            double* elementDiameter,
                                            double* strong_residual,
                                            double* grad_u,
                                            double* numDiff);
extern void calculateNumericalDiffusionEikonal(proteus_index_t nElements_global,
					int nQuadraturePoints_element,
					double shockCapturingDiffusion,
					double* elementDiameter,
					double* strong_residual,
					double* numDiff);
extern void calculateNumericalDiffusionHJ(proteus_index_t nElements_global,
				   int nQuadraturePoints_element,
				   char shockCapturing,
				   double shockCapturingDiffusion,
				   double* elementDiameter,
				   double* strong_residual,
				   double* mt,
				   double* H,
				   double* numDiff);
extern void calculateNumericalDiffusionHJV2(proteus_index_t nElements_global,
				     int nQuadraturePoints_element,
				     char shockCapturing,
				     double shockCapturingDiffusion,
				     double* elementDiameter,
				     double* strong_residual,
				     double* mt,
				     double* H,
				     double* numDiff);
extern void calculateNumericalDiffusion_A_1(proteus_index_t nElements_global,
                                     int nQuadraturePoints_element,
                                     int nSpace,
                                     double shockCapturingFactor,
                                     double* elementDiameter,
                                     double* strong_residual,
                                     double* mt,
                                     double* df,
                                     double* numDiff);
extern void calculateNumericalDiffusionResGradJuanes(proteus_index_t nElements_global,
					      int nQuadraturePoints_element,
					      int nSpace,
					      double shockCapturingDiffusion,
					      double uSC,
					      double* elementDiameter,
					      double* strong_residual,
					      double* grad_u,
					      double* numDiff);
extern void calculateNumericalDiffusionJaffre(proteus_index_t nElements_global,
				       int nQuadraturePoints_element,
				       int nSpace,
				       double shockCapturingDiffusion,
				       double beta,
				       double* elementDiameter,
				       double* strong_residual,
				       double* grad_u,
				       double* numDiff);
/** @} */
#endif
