#ifndef FEMINTEGRALS_H
#define FEMINTEGRALS_H
#include "index_type.h"

/*!
 \file femIntegrals.h
 \brief C implementation of fem integral calculations
*/

/**
   \defgroup femIntegrals femIntegrals
 \brief C implementation of fem integral calculations
   @{ 
*/

extern void parametricFiniteElementSpace_getHessianValues(proteus_index_t nElements_global, int nQuadraturePoints_element, int nDOF_element, int nSpace_global, double *Hessian_psi, double *inverseJacobianArray, double *Hessian_vArray);
extern void     updateDiffusion2_strong(proteus_index_t nElements_global, int nQuadraturePoints_element, int nSpace, double *a, double *Hess_phi, double *strong_residual);
extern void updateDiffusion2_strong_sd(proteus_index_t nElements_global, int nQuadraturePoints_element,int nSpace, proteus_csr_index_t* rowptr, proteus_csr_index_t* colind, double* a, double* Hess_phi, double* strong_residual);
extern void     updateDiffusionJacobian2_strong(proteus_index_t nElements_global, int nQuadraturePoints_element, int nDOF_trial_element, int nSpace, proteus_index_t *l2g, double *a, double *da, double *v, double *Hess_phi, double *dphi, double *Hess_v, double *dstrong_residual);
extern void updateDiffusionJacobian2_strong_sd(proteus_index_t nElements_global, 
					       int nQuadraturePoints_element,
					       int nDOF_trial_element,
					       int nSpace,
					       proteus_csr_index_t* rowptr,
					       proteus_csr_index_t* colind,
					       proteus_index_t* l2g,
					       double* a,
					       double* da,
					       double* v,
					       double* Hess_phi,
					       double* dphi,
					       double* Hess_v,
					       double* dstrong_residual);
extern void     updateDiffusion2_adjoint(proteus_index_t nElements_global, int nQuadraturePoints_element, int nDOF_test_element, int nSpace, double *a, double *Hess_w_dV, double *Lstar_w_dV);
void updateDiffusion2_adjoint_sd(proteus_index_t nElements_global, int nQuadraturePoints_element, int nDOF_test_element, int nSpace, proteus_csr_index_t* rowptr, proteus_csr_index_t* colind, double* a, double* Hess_w_dV, double* Lstar_w_dV);
extern void     calculateWeightedShapeHessians(proteus_index_t nElements_global, int nQuadraturePoints_element, int nDOF_test_element, int nSpace, double *dVR, double *abs_det_jac, double *Hess_w, double *Hess_w_dV);
extern void     calculateFiniteElementFunctionHessianValues(proteus_index_t nElements_global, int nQuadraturePoints_element, int nDOF_trial_element, int nComponents, int nSpace, proteus_index_t *l2g, double *dof, double *Hessian_v, double *Hessian_u);
extern void     updateGlobalJacobianFromInteriorElementBoundaryFluxJacobian_2sided_CSR(proteus_index_t nInteriorElementBoundaries_global, int nElementBoundaries_element, int nQuadraturePoints_elementBoundary, int nDOF_test_element, int nDOF_trial_element, proteus_index_t *interiorElementBoundaries, proteus_index_t *elementBoundaryElements, int *elementBoundaryLocalElementBoundaries, int *nFreeDOF_element_r, int *freeLocal_r, int *nFreeDOF_element_u, int *freeLocal_u, proteus_csr_index_t *csrRowIndeces_ru, proteus_csr_index_t *csrColumnOffsets_eb_ru, double *elementBoundaryFluxJacobian_2sided, double *w_dS, double *jac);
extern void     updateGlobalJacobianFromInteriorElementBoundaryFluxJacobian_2sided_dense(proteus_index_t nInteriorElementBoundaries_global, int nElementBoundaries_element, int nQuadraturePoints_elementBoundary, int nDOF_test_element, int nDOF_trial_element, proteus_index_t offset_r, proteus_index_t stride_r, proteus_index_t offset_u, proteus_index_t stride_u, proteus_index_t nFreeVDOF_global, proteus_index_t *interiorElementBoundaries, proteus_index_t *elementBoundaryElements, int *elementBoundaryLocalElementBoundaries, int *nFreeDOF_element_r, int *freeLocal_r, proteus_index_t *freeGlobal_r, int *nFreeDOF_element_u, int *freeLocal_u, proteus_index_t *freeGlobal_u, double *elementBoundaryFluxJacobian_2sided, double *w_dS, double *jac);
extern void     updateInteriorTwoSidedElementBoundaryFlux(proteus_index_t nInteriorElementBoundaries_global, int nElementBoundaries_element, int nQuadraturePoints_elementBoundary, int nDOF_test_element, proteus_index_t *interiorElementBoundaries, proteus_index_t *elementBoundaryElements, int *elementBoundaryLocalElementBoundaries, double *flux, double *w_dS, double *residual);
extern void     calculateCFLADR2speeds(proteus_index_t nElements_global, int nQuadraturePoints_element, int nSpace, double *elementDiameter, double *dm, double *df1, double *df2, double *cfl);
extern int      checkElementBoundaryAndExteriorElementBoundaryArraysSame(int nElementBoundaries_element, proteus_index_t nExteriorElementBoundaries_global, int nQuadraturePoints_elementBoundary, int nValuesPerQuadraturePoint, double tolerance, const proteus_index_t *exteriorElementBoundariesArray, const proteus_index_t *elementBoundaryElementsArray, const int *elementBoundaryLocalElementBoundariesArray, const double *ebq_val, const double *ebqe_val, proteus_index_t *firstBadIndex);
extern int      checkGlobalElementBoundaryAndExteriorElementBoundaryArraysSame(proteus_index_t nExteriorElementBoundaries_global, int nQuadraturePoints_elementBoundary, int nValuesPerQuadraturePoint, double tolerance, const proteus_index_t *exteriorElementBoundariesArray, const proteus_index_t *elementBoundaryElementsArray, const int *elementBoundaryLocalElementBoundariesArray, const double *ebq_global_val, const double *ebqe_val, proteus_index_t *firstBadIndex);
extern void     calculateExteriorElementBoundaryStress3D(proteus_index_t nExteriorElementBoundaries_global, int nQuadraturePoints_elementBoundary, int *elementBoundaryMaterialTypes, proteus_index_t *exteriorElementBoundaries, proteus_index_t *elementBoundaryElements, int *elementBoundaryLocalElementBoundaries, double *p, double *mom_flux_vec_u, double *mom_flux_vec_v, double *mom_flux_vec_w, double *dS, double *n, double *F);
extern void     calculateExteriorElementBoundaryStress2D(proteus_index_t nExteriorElementBoundaries_global, int nQuadraturePoints_elementBoundary, int *elementBoundaryMaterialTypes, proteus_index_t *exteriorElementBoundaries, proteus_index_t *elementBoundaryElements, int *elementBoundaryLocalElementBoundaries, double *p, double *mom_flux_vec_u, double *mom_flux_vec_v, double *dS, double *n, double *F);

extern void copyLeftElementBoundaryInfo(
  int nElementBoundaries_element,
  int nElementBoundaryQuadraturePoints_elementBoundary,
  int nSpace_global,
  proteus_index_t nExteriorElementBoundaries_global,
  proteus_index_t nInteriorElementBoundaries_global,
  proteus_index_t *elementBoundaryElementsArray,
  int *elementBoundaryLocalElementBoundariesArray,
  proteus_index_t *exteriorElementBoundariesArray,
  proteus_index_t *interiorElementBoundariesArray,
  double *x,
  double *n,
  double *xg,
  double *ng
);
extern void parametricFiniteElementSpace_getValues(
  proteus_index_t nElements_global,
  int nQuadraturePoints_element,
  int nDOF_element,
  double *psi,
  double *vArray
);
extern void parametricFiniteElementSpace_getValuesTrace(
  proteus_index_t nElements_global,
  int nElementBoundaries_element,
  int nElementBoundaryQuadraturePoints_elementBoundary,
  int nDOF_element,
  double *psi,
  int *permutations,
  double *vArray
);
extern void parametricFiniteElementSpace_getGradientValues(
  proteus_index_t nElements_global,
  int nQuadraturePoints_element,
  int nDOF_element,
  int nSpace_global,
  double *grad_psi,
  double *inverseJacobianArray,
  double *grad_vArray
);
extern void parametricFiniteElementSpace_getGradientValuesTrace(
  proteus_index_t nElements_global,
  int nElementBoundaries_element,
  int nElementBoundaryQuadraturePoints_elementBoundary,
  int nDOF_element,
  int nSpace_global,
  double *grad_psi,
  int *permutations,
  double *inverseJacobianArray,
  double *grad_vArray
);
extern void parametricMaps_getPermutations(
  proteus_index_t nElements_global,
  int nElementBoundaries_element,
  int nElementBoundaryQuadraturePoints_elementBoundary,
  int nSpace_global,
  double *xiArray,
  int *permutations
);
extern void parametricMaps_getPermutationsGlobalExterior(int nElementBoundaryQuadraturePoints_elementBoundary,
							 int nSpace_global,
							 proteus_index_t nExteriorElementBoundaries_global,
							 const proteus_index_t * exteriorElementBoundariesArray,
							 const proteus_index_t * elementBoundaryElementsArray,
							 const int * elementBoundaryLocalElementBoundariesArray,
							 double* xiArray,
							 int* permutations);
extern void getPermutationsGlobal(proteus_index_t nElementBoundaries_global,
				  int nElementBoundaryQuadraturePoints_elementBoundary,
				  double* xArray,
				  double* xArrayNew,
				  int* permutations);
extern void parametricMaps_getValues(
  proteus_index_t nElements_global,
  int nQuadraturePoints_element,
  int nDOF_element,
  int nSpace_global,
  double *psi,
  proteus_index_t *l2g,
  double *nodeArray,
  double *xArray
);
extern void parametricMaps_getValuesTrace(
  proteus_index_t nElements_global,
  int nElementBoundaries_element,
  int nQuadraturePoints_element,
  int nDOF_element,
  int nSpace_global,
  double *psi,
  proteus_index_t *l2g,
  double *nodeArray,
  double *xArray
);
extern void parametricMaps_getInverseValues(
  proteus_index_t nElements_global,
  int nQuadraturePoints_element,
  int nDOF_element,
  int nSpace_global,
  double *inverseJacobian,
  proteus_index_t *l2g,
  double *nodeArray,
  double *xArray,
  double *xiArray
);
extern void parametricMaps_getInverseValuesTrace(
  proteus_index_t nElements_global,
  int nElementBoundaries_element,
  int nElementBoundaryQuadraturePoints_elementBoundary,
  int nDOF_element,
  int nSpace_global,
  double *inverseJacobian,
  proteus_index_t *l2g,
  double *nodeArray,
  double *xArray,
  double *xiArray
);
extern void parametricMaps_getJacobianValues3D(
  proteus_index_t nElements_global,
  int nQuadraturePoints_element,
  int nDOF_element,
  double *grad_psi,
  proteus_index_t *l2g,
  double *nodeArray,
  double *jacobianArray,
  double *jacobianDeterminantArray,
  double *jacobianInverseArray
);
extern void parametricMaps_getJacobianValues2D(
  proteus_index_t nElements_global,
  int nQuadraturePoints_element,
  int nDOF_element,
  double *grad_psi,
  proteus_index_t *l2g,
  double *nodeArray,
  double *jacobianArray,
  double *jacobianDeterminantArray,
  double *jacobianInverseArray
);
extern void parametricMaps_getJacobianValues1D(
  proteus_index_t nElements_global,
  int nQuadraturePoints_element,
  int nDOF_element,
  double *grad_psi,
  proteus_index_t *l2g,
  double *nodeArray,
  double *jacobianArray,
  double *jacobianDeterminantArray,
  double *jacobianInverseArray
);
extern void parametricMaps_getJacobianValuesTrace3D(
  proteus_index_t nElements_global,
  int nElementBoundaries_element,
  int nQuadraturePoints_element,
  int nDOF_element,
  double *grad_psi,
  double *boundaryNormals,
  double *boundaryJacobians,
  proteus_index_t *l2g,
  double *nodeArray,
  double *jacobianInverseArray,
  double *metricTensorArray,
  double *metricTensorDeterminantSqrtArray,
  double *unitNormalArray
);
extern void parametricMaps_getJacobianValuesTrace2D(
  proteus_index_t nElements_global,
  int nElementBoundaries_element,
  int nQuadraturePoints_element,
  int nDOF_element,
  double *grad_psi,
  double *boundaryNormals,
  double *boundaryJacobians,
  proteus_index_t *l2g,
  double *nodeArray,
  double *jacobianInverseArray,
  double *metricTensorArray,
  double *metricTensorDeterminantSqrtArray,
  double *unitNormalArray
);
extern void parametricMaps_getJacobianValuesTrace1D(
  proteus_index_t nElements_global,
  int nElementBoundaries_element,
  int nQuadraturePoints_element,
  int nDOF_element,
  double *grad_psi,
  double *boundaryNormals,
  double *boundaryJacobians,
  proteus_index_t *l2g,
  double *nodeArray,
  double *jacobianInverseArray,
  double *metricTensorArray,
  double *metricTensorDeterminantSqrtArray,
  double *unitNormalArray
);
extern void updateMass_weak(
  proteus_index_t nElements_global,
  int nQuadraturePoints_element,
  int nDOF_test_element,
  double *mt,
  double *w_dV,
  double *weak_residual
);
extern void updateMassJacobian_weak(
  proteus_index_t nElements_global,
  int nQuadraturePoints_element,
  int nDOF_trial_element,
  int nDOF_test_element,
  double *dmt,
  double *v_X_w_dV,
  double *jacobian_weak_residual
);
extern void updateMassJacobian_weak_lowmem(proteus_index_t nElements_global,
                                           int nQuadraturePoints_element,
                                           int nDOF_trial_element,
                                           int nDOF_test_element,
                                           double* dmt,
                                           double* v,
                                           double* w_dV,
                                           double* jacobian_weak_residual);
extern void updateMass_strong(
  proteus_index_t nElements_global,
  int nQuadraturePoints_element,
  double *mt,
  double *strong_residual
);
extern void updateMassJacobian_strong(
  proteus_index_t nElements_global,
  int nQuadraturePoints_element,
  int nDOF_trial_element,
  double *dmt,
  double *v,
  double *dstrong_residual
);
extern void updateMass_adjoint(
  proteus_index_t nElements_global,
  int nQuadraturePoints_element,
  int nDOF_test_element,
  double *dmt,
  double *w_dV,
  double *Lstar_w_dV
);
extern void updateAdvection_weak(
  proteus_index_t nElements_global,
  int nQuadraturePoints_element,
  int nDOF_test_element,
  int nSpace,
  double *f,
  double *grad_w_dV,
  double *weak_residual
);
extern void updateAdvectionJacobian_weak(
  proteus_index_t nElements_global,
  int nQuadraturePoints_element,
  int nDOF_trial_element,
  int nDOF_test_element,
  int nSpace,
  double *df,
  double *v_X_grad_w_dV,
  double *jacobian_weak_residual
);
extern void updateAdvectionJacobian_weak_lowmem(proteus_index_t nElements_global,
                                                int nQuadraturePoints_element,
                                                int nDOF_trial_element,
                                                int nDOF_test_element,
                                                int nSpace,
                                                double* df,
                                                double* v,
                                                double* grad_w_dV,
                                                double* jacobian_weak_residual);
extern void updateAdvection_strong(
  proteus_index_t nElements_global,
  int nQuadraturePoints_element,
  int nSpace,
  double *df,
  double *grad_u,
  double *strong_residual
);
extern void updateAdvectionJacobian_strong(
  proteus_index_t nElements_global,
  int nQuadraturePoints_element,
  int nDOF_trial_element,
  int nSpace,
  double *df,
  double *grad_v,
  double *dstrong_residual
);
extern void updateAdvection_adjoint(
  proteus_index_t nElements_global,
  int nQuadraturePoints_element,
  int nDOF_test_element,
  int nSpace,
  double *df,
  double *grad_w_dV,
  double *Lstar_w_dV
);
extern void updateHamiltonian_weak(
  proteus_index_t nElements_global,
  int nQuadraturePoints_element,
  int nDOF_test_element,
  double *H,
  double *w_dV,
  double *weak_residual
);
extern void updateHamiltonianJacobian_weak(
  proteus_index_t nElements_global,
  int nQuadraturePoints_element,
  int nDOF_trial_element,
  int nDOF_test_element,
  int nSpace,
  double *dH,
  double *grad_v_X_w_dV,
  double *jacobian_weak_residual
);
extern void updateHamiltonianJacobian_weak_lowmem(proteus_index_t nElements_global,
                                                  int nQuadraturePoints_element,
                                                  int nDOF_trial_element,
                                                  int nDOF_test_element,
                                                  int nSpace,
                                                  double* dH,
                                                  double* grad_v,
                                                  double* w_dV,
                                                  double* jacobian_weak_residual);
extern void updateHamiltonian_strong(
  proteus_index_t nElements_global,
  int nQuadraturePoints_element,
  int nSpace,
  double *dH,
  double *grad_u,
  double *strong_residual
);
extern void updateHamiltonianJacobian_strong(
  proteus_index_t nElements_global,
  int nQuadraturePoints_element,
  int nDOF_trial_element,
  int nSpace,
  double *dH,
  double *grad_v,
  double *dstrong_residual
);
extern void updateHamiltonian_adjoint(
  proteus_index_t nElements_global,
  int nQuadraturePoints_element,
  int nDOF_test_element,
  int nSpace,
  double *dH,
  double *grad_w_dV,
  double *Lstar_w_dV
);
extern void updateDiffusion_weak(
  proteus_index_t nElements_global,
  int nQuadraturePoints_element,
  int nDOF_test_element,
  int nSpace,
  double *a,
  double *grad_phi_X_grad_w_dV,
  double *weak_residual
);
extern void updateDiffusion_weak_lowmem(proteus_index_t nElements_global,
                                 int nQuadraturePoints_element,
                                 int nDOF_test_element,
                                 int nSpace,
                                 double* a,
                                 double* grad_phi,
                                 double* grad_w_dV,
                                 double* weak_residual);
extern void updateDiffusion_weak_sd(proteus_index_t nElements_global,
				    int nQuadraturePoints_element,
				    int nDOF_test_element,
				    int nSpace,
				    proteus_csr_index_t* rowptr,
				    proteus_csr_index_t* colind,
				    double* a,
				    double* grad_phi,
				    double* grad_w_dV,
				    double* weak_residual);
extern void updateDiffusionJacobian_weak(
  proteus_index_t nElements_global,
  int nQuadraturePoints_element,
  int nDOF_trial_element,
  int nDOF_test_element,
  int nSpace,
  proteus_index_t *l2g,
  double *a,
  double *da,
  double *grad_phi_X_grad_w_dV,
  double *dphi,
  double *v,
  double *grad_v_X_grad_w_dV,
  double *jacobian_weak_residual
);
extern void updateDiffusionJacobian_weak_lowmem(proteus_index_t nElements_global,
                                                int nQuadraturePoints_element,
                                                int nDOF_trial_element,
                                                int nDOF_test_element,
                                                int nSpace,
                                                proteus_index_t* l2g,
                                                double* a,
                                                double* da,
                                                double* grad_phi,
                                                double* grad_w_dV,
                                                double* dphi,
                                                double* v,
                                                double* grad_v,
                                                double* jacobian_weak_residual);
extern void updateDiffusionJacobian_weak_sd(proteus_index_t nElements_global,
					    int nQuadraturePoints_element,
					    int nDOF_trial_element,
					    int nDOF_test_element,
					    int nSpace,
					    proteus_csr_index_t* rowptr,
					    proteus_csr_index_t* colind,
					    proteus_index_t* l2g,
					    double* a,
					    double* da,
					    double* grad_phi,
					    double* grad_w_dV,
					    double* dphi,
					    double* v,
					    double* grad_v,
					    double* jacobian_weak_residual);
extern void updateDiffusion_strong(
  proteus_index_t nElements_global,
  int nQuadraturePoints_element,
  int nSpace,
  double *da,
  double *grad_phi,
  double *grad_u,
  double *strong_residual
);
extern void updateDiffusion_strong_sd(proteus_index_t nElements_global,
				      int nQuadraturePoints_element,
				      int nSpace,
				      proteus_csr_index_t* rowptr,
				      proteus_csr_index_t* colind,
				      double* da,
				      double* grad_phi,
				      double* grad_u,
				      double* strong_residual);
extern void updateDiffusionJacobian_strong(
  proteus_index_t nElements_global,
  int nQuadraturePoints_element,
  int nDOF_trial_element,
  int nSpace,
  proteus_index_t *l2g,
  double *da,
  double *dphi,
  double *grad_phi,
  double *grad_u,
  double *grad_v,
  double *dstrong_residual
);
extern void updateDiffusionJacobian_strong_sd(proteus_index_t nElements_global,
					      int nQuadraturePoints_element,
					      int nDOF_trial_element,
					      int nSpace,
					      proteus_csr_index_t* rowptr,
					      proteus_csr_index_t* colind,
					      proteus_index_t* l2g,
					      double* da,
					      double* dphi,
					      double* grad_phi,
					      double* grad_u,
					      double* grad_v,
					      double* dstrong_residual);
extern void updateDiffusion_adjoint(
  proteus_index_t nElements_global,
  int nQuadraturePoints_element,
  int nDOF_test_element,
  int nSpace,
  double *da,
  double *grad_phi,
  double *grad_w_dV,
  double *Lstar_w_dV
);
extern void updateDiffusion_adjoint_sd(proteus_index_t nElements_global,
				       int nQuadraturePoints_element,
				       int nDOF_test_element,
				       int nSpace,
				       proteus_csr_index_t* rowptr,
				       proteus_csr_index_t* colind,
				       double* da,
				       double* grad_phi,
				       double* grad_w_dV,
				       double* Lstar_w_dV);
extern void updateReaction_weak(
  proteus_index_t nElements_global,
  int nQuadraturePoints_element,
  int nDOF_test_element,
  double *r,
  double *w_dV,
  double *weak_residual
);
extern void updateReactionJacobian_weak(
  proteus_index_t nElements_global,
  int nQuadraturePoints_element,
  int nDOF_trial_element,
  int nDOF_test_element,
  double *dr,
  double *v_X_w_dV,
  double *jacobian_weak_residual
);
extern void updateReactionJacobian_weak_lowmem(proteus_index_t nElements_global,
                                               int nQuadraturePoints_element,
                                               int nDOF_trial_element,
                                               int nDOF_test_element,
                                               double* dr,
                                               double* v,
                                               double* w_dV,
                                               double* jacobian_weak_residual);
extern void updateReaction_strong(
  proteus_index_t nElements_global,
  int nQuadraturePoints_element,
  double *r,
  double *strong_residual
);
extern void updateReactionJacobian_strong(
  proteus_index_t nElements_global,
  int nQuadraturePoints_element,
  int nDOF_trial_element,
  double *dr,
  double *v,
  double *dstrong_residual
);
extern void updateReaction_adjoint(
  proteus_index_t nElements_global,
  int nQuadraturePoints_element,
  int nDOF_test_element,
  double *dr,
  double *w_dV,
  double *Lstar_w_dV
);
extern void updateSubgridError(
  proteus_index_t nElements_global,
  int nQuadraturePoints_element,
  int nDOF_test_element,
  double *error,
  double *Lstar_w_dV,
  double *weak_residual
);
extern void updateSubgridErrorJacobian(
  proteus_index_t nElements_global,
  int nQuadraturePoints_element,
  int nDOF_trial_element,
  int nDOF_test_element,
  double *derror,
  double *Lstar_w_dV,
  double *jacobian_weak_residual
);
extern void updateNumericalDiffusion(
  proteus_index_t nElements_global,
  int nQuadraturePoints_element,
  int nDOF_test_element,
  int nSpace,
  double *numDiff,
  double *grad_u_X_grad_w_dV,
  double *weak_residual
);
extern void updateNumericalDiffusion_lowmem(proteus_index_t nElements_global,
                                     int nQuadraturePoints_element,
                                     int nDOF_test_element,
                                     int nSpace,
                                     double* numDiff,
                                     double* grad_u,
                                     double* grad_w_dV,
                                     double* weak_residual);
extern void updateNumericalDiffusionJacobian(
  proteus_index_t nElements_global,
  int nQuadraturePoints_element,
  int nDOF_trial_element,
  int nDOF_test_element,
  int nSpace,
  double *numDiff,
  double *grad_v_X_grad_w_dV,
  double *jacobian_weak_residual
);
extern void updateNumericalDiffusionJacobian_lowmem(proteus_index_t nElements_global,
                                                    int nQuadraturePoints_element,
                                                    int nDOF_trial_element,
                                                    int nDOF_test_element,
                                                    int nSpace,
                                                    double* numDiff,
                                                    double* grad_v,
                                                    double* grad_w_dV,
                                                    double* jacobian_weak_residual);
extern void calculateScalarScalarProduct(
  proteus_index_t nElements_global,
  int nQuadraturePoints_element,
  double *s1,
  double *s2,
  double *sResult
);
extern void calculateVectorScalarProduct(
  proteus_index_t nElements_global,
  int nQuadraturePoints_element,
  int nSpace,
  double *v,
  double *s,
  double *vResult
);
extern void calculateTensorScalarProduct(
  proteus_index_t nElements_global,
  int nQuadraturePoints_element,
  int nSpace,
  double *t,
  double *s,
  double *tResult
);
extern void updateInteriorElementBoundaryFlux(
  proteus_index_t nInteriorElementBoundaries_global,
  int nElementBoundaries_element,
  int nQuadraturePoints_elementBoundary,
  int nDOF_test_element,
  proteus_index_t *interiorElementBoundaries,
  proteus_index_t *elementBoundaryElements,
  int *elementBoundaryLocalElementBoundaries,
  double *flux,
  double *w_dS,
  double *residual
);
extern void updateExteriorElementBoundaryFlux(
  proteus_index_t nExteriorElementBoundaries_global,
  int nElementBoundaries_element,
  int nQuadraturePoints_elementBoundary,
  int nDOF_test_element,
  proteus_index_t *exteriorElementBoundaries,
  proteus_index_t *elementBoundaryElements,
  int *elementBoundaryLocalElementBoundaries,
  double *flux,
  double *w_dS,
  double *residual
);
extern void updateGlobalResidualFromElementResidual(
  proteus_index_t nElements_global,
  int nDOF_test_element,
  proteus_index_t offset_r,
  proteus_index_t stride_r,
  int *nFreeDOF_element_r,
  int *freeLocal_r,
  proteus_index_t *freeGlobal_r,
  double *elementResidual,
  double *globalResidual
);
extern void updateGlobalJacobianFromElementJacobian_dense(
  proteus_index_t nElements_global,
  int nDOF_test_element,
  int nDOF_trial_element,
  proteus_index_t offset_r,
  proteus_index_t stride_r,
  proteus_index_t offset_u,
  proteus_index_t stride_u,
  proteus_index_t nFreeVDOF_global,
  int *nFreeDOF_element_r,
  int *freeLocal_r,
  proteus_index_t *freeGlobal_r,
  int *nFreeDOF_element_u,
  int *freeLocal_u,
  proteus_index_t *freeGlobal_u,
  double *elementJacobian,
  double *globalJacobian
);
extern void updateGlobalJacobianFromElementJacobian_eb_dense(
  proteus_index_t *elementNeighbors,
  proteus_index_t nElements_global,
  int nElementBoundaries_element,
  int nDOF_test_element,
  int nDOF_trial_element,
  proteus_index_t offset_r,
  proteus_index_t stride_r,
  proteus_index_t offset_u,
  proteus_index_t stride_u,
  proteus_index_t nFreeVDOF_global,
  int *nFreeDOF_element_r,
  int *freeLocal_r,
  proteus_index_t *freeGlobal_r,
  int *nFreeDOF_element_u,
  int *freeLocal_u,
  proteus_index_t *freeGlobal_u,
  double *elementJacobian_eb,
  double *globalJacobian
);
extern void updateGlobalJacobianFromInteriorElementBoundaryFluxJacobian_dense(
  proteus_index_t nInteriorElementBoundaries_global,
  int nElementBoundaries_element,
  int nQuadraturePoints_elementBoundary,
  int nDOF_test_element,
  int nDOF_trial_element,
  proteus_index_t offset_r,
  proteus_index_t stride_r,
  proteus_index_t offset_u,
  proteus_index_t stride_u,
  proteus_index_t nFreeVDOF_global,
  proteus_index_t *interiorElementBoundaries,
  proteus_index_t *elementBoundaryElements,
  int *elementBoundaryLocalElementBoundaries,
  int *nFreeDOF_element_r,
  int *freeLocal_r,
  proteus_index_t *freeGlobal_r,
  int *nFreeDOF_element_u,
  int *freeLocal_u,
  proteus_index_t *freeGlobal_u,
  double *elementBoundaryFluxJacobian,
  double *w_dS,
  double *jac
);
extern void
updateGlobalJacobianFromInteriorElementBoundaryFluxJacobian_eb_dense(
  proteus_index_t *elementNeighbors,
  proteus_index_t nElements_global,
  proteus_index_t nInteriorElementBoundaries_global,
  int nElementBoundaries_element,
  int nQuadraturePoints_elementBoundary,
  int nDOF_test_element,
  int nDOF_trial_element,
  proteus_index_t offset_r,
  proteus_index_t stride_r,
  proteus_index_t offset_u,
  proteus_index_t stride_u,
  proteus_index_t nFreeVDOF_global,
  proteus_index_t *interiorElementBoundaries,
  proteus_index_t *elementBoundaryElements,
  int *elementBoundaryLocalElementBoundaries,
  int *nFreeDOF_element_r,
  int *freeLocal_r,
  proteus_index_t *freeGlobal_r,
  int *nFreeDOF_element_u,
  int *freeLocal_u,
  proteus_index_t *freeGlobal_u,
  double *elementBoundaryFluxJacobian_eb,
  double *w_dS,
  double *jac
);
extern void updateGlobalJacobianFromExteriorElementBoundaryFluxJacobian_dense(
  proteus_index_t nExteriorElementBoundaries_global,
  int nElementBoundaries_element,
  int nQuadraturePoints_elementBoundary,
  int nDOF_test_element,
  int nDOF_trial_element,
  proteus_index_t offset_r,
  proteus_index_t stride_r,
  proteus_index_t offset_u,
  proteus_index_t stride_u,
  proteus_index_t nFreeVDOF_global,
  proteus_index_t *exteriorElementBoundaries,
  proteus_index_t *elementBoundaryElements,
  int *elementBoundaryLocalElementBoundaries,
  int *nFreeDOF_element_r,
  int *freeLocal_r,
  proteus_index_t *freeGlobal_r,
  int *nFreeDOF_element_u,
  int *freeLocal_u,
  proteus_index_t *freeGlobal_u,
  double *elementBoundaryFluxJacobian,
  double *w_dS,
  double *jac
);
extern void
updateGlobalJacobianFromExteriorElementBoundaryFluxJacobian_eb_dense(
  proteus_index_t *elementNeighbors,
  proteus_index_t nElements_global,
  proteus_index_t nExteriorElementBoundaries_global,
  int nElementBoundaries_element,
  int nQuadraturePoints_elementBoundary,
  int nDOF_test_element,
  int nDOF_trial_element,
  proteus_index_t offset_r,
  proteus_index_t stride_r,
  proteus_index_t offset_u,
  proteus_index_t stride_u,
  proteus_index_t nFreeVDOF_global,
  proteus_index_t *exteriorElementBoundaries,
  proteus_index_t *elementBoundaryElements,
  int *elementBoundaryLocalElementBoundaries,
  int *nFreeDOF_element_r,
  int *freeLocal_r,
  proteus_index_t *freeGlobal_r,
  int *nFreeDOF_element_u,
  int *freeLocal_u,
  proteus_index_t *freeGlobal_u,
  double *elementBoundaryFluxJacobian_eb,
  double *w_dS,
  double *jac
);
extern void updateGlobalJacobianFromElementJacobian_CSR(
  proteus_index_t nElements_global,
  int nDOF_test_element,
  int nDOF_trial_element,
  int *nFreeDOF_element_r,
  int *freeLocal_r,
  int *nFreeDOF_element_u,
  int *freeLocal_u,
  proteus_csr_index_t *csrRowIndeces_ru,
  proteus_csr_index_t *csrColumnOffsets_ru,
  double *elementJacobian,
  double *globalJacobian
);
extern void updateGlobalJacobianFromElementJacobian_eb_CSR(
  proteus_index_t *elementNeighbors,
  proteus_index_t nElements_global,
  int nElementBoundaries_element,
  int nDOF_test_element,
  int nDOF_trial_element,
  int *nFreeDOF_element_r,
  int *freeLocal_r,
  int *nFreeDOF_element_u,
  int *freeLocal_u,
  proteus_csr_index_t *csrRowIndeces_ru,
  proteus_csr_index_t *csrColumnOffsets_eb_ru,
  double *elementJacobian_eb,
  double *globalJacobian
);
extern void updateGlobalJacobianFromInteriorElementBoundaryFluxJacobian_CSR(
  proteus_index_t nInteriorElementBoundaries_global,
  int nElementBoundaries_element,
  int nQuadraturePoints_elementBoundary,
  int nDOF_test_element,
  int nDOF_trial_element,
  proteus_index_t *interiorElementBoundaries,
  proteus_index_t *elementBoundaryElements,
  int *elementBoundaryLocalElementBoundaries,
  int *nFreeDOF_element_r,
  int *freeLocal_r,
  int *nFreeDOF_element_u,
  int *freeLocal_u,
  proteus_csr_index_t *csrRowIndeces_ru,
  proteus_csr_index_t *csrColumnOffsets_eb_ru,
  double *elementBoundaryFluxJacobian,
  double *w_dS,
  double *jac
);
extern void updateGlobalJacobianFromExteriorElementBoundaryFluxJacobian_CSR(
  proteus_index_t nExteriorElementBoundaries_global,
  int nElementBoundaries_element,
  int nQuadraturePoints_elementBoundary,
  int nDOF_test_element,
  int nDOF_trial_element,
  proteus_index_t *exteriorElementBoundaries,
  proteus_index_t *elementBoundaryElements,
  int *elementBoundaryLocalElementBoundaries,
  int *nFreeDOF_element_r,
  int *freeLocal_r,
  int *nFreeDOF_element_u,
  int *freeLocal_u,
  proteus_csr_index_t *csrRowIndeces_ru,
  proteus_csr_index_t *csrColumnOffsets_eb_ru,
  double *elementBoundaryFluxJacobian,
  double *w_dS,
  double *jac
);
extern void updateGlobalJacobianFromInteriorElementBoundaryFluxJacobian_eb_CSR(
  proteus_index_t *elementNeighbors,
  proteus_index_t nInteriorElementBoundaries_global,
  int nElementBoundaries_element,
  int nQuadraturePoints_elementBoundary,
  int nDOF_test_element,
  int nDOF_trial_element,
  proteus_index_t *interiorElementBoundaries,
  proteus_index_t *elementBoundaryElements,
  int *elementBoundaryLocalElementBoundaries,
  int *nFreeDOF_element_r,
  int *freeLocal_r,
  int *nFreeDOF_element_u,
  int *freeLocal_u,
  proteus_csr_index_t *csrRowIndeces_ru,
  proteus_csr_index_t *csrColumnOffsets_eb_eNebN_ru,
  double *elementBoundaryFluxJacobian_eb,
  double *w_dS,
  double *jac
);
extern void updateGlobalJacobianFromExteriorElementBoundaryFluxJacobian_eb_CSR(
  proteus_index_t *elementNeighbors,
  proteus_index_t nExteriorElementBoundaries_global,
  int nElementBoundaries_element,
  int nQuadraturePoints_elementBoundary,
  int nDOF_test_element,
  int nDOF_trial_element,
  proteus_index_t *exteriorElementBoundaries,
  proteus_index_t *elementBoundaryElements,
  int *elementBoundaryLocalElementBoundaries,
  int *nFreeDOF_element_r,
  int *freeLocal_r,
  int *nFreeDOF_element_u,
  int *freeLocal_u,
  proteus_csr_index_t *csrRowIndeces_ru,
  proteus_csr_index_t *csrColumnOffsets_eb_eNebN_ru,
  double *elementBoundaryFluxJacobian_eb,
  double *w_dS,
  double *jac
);
extern void calculateWeightedShape(
  proteus_index_t nElements_global,
  int nQuadraturePoints_element,
  int nDOF_test_element,
  double *dVR,
  double *abs_det_jac,
  double *w,
  double *w_dV
);
extern void calculateWeightedShapeGradients(
  proteus_index_t nElements_global,
  int nQuadraturePoints_element,
  int nDOF_test_element,
  int nSpace,
  double *dVR,
  double *abs_det_jac,
  double *grad_w,
  double *grad_w_dV
);
extern void calculateShape_X_weightedShape(
  proteus_index_t nElements_global,
  int nQuadraturePoints_element,
  int nDOF_trial_element,
  int nDOF_test_element,
  double *v,
  double *w_dV,
  double *v_X_w_dV
);
extern void calculateShape_X_weightedGradShape(
  proteus_index_t nElements_global,
  int nQuadraturePoints_element,
  int nDOF_trial_element,
  int nDOF_test_element,
  int nSpace,
  double *v,
  double *grad_w_dV,
  double *v_X_grad_w_dV
);
extern void calculateGradShape_X_weightedShape(
  proteus_index_t nElements_global,
  int nQuadraturePoints_element,
  int nDOF_trial_element,
  int nDOF_test_element,
  int nSpace,
  double *grad_v,
  double *w_dV,
  double *grad_v_X_w_dV
);
extern void calculateGradShape_X_weightedGradShape(
  proteus_index_t nElements_global,
  int nQuadraturePoints_element,
  int nDOF_trial_element,
  int nDOF_test_element,
  int nSpace,
  double *grad_v,
  double *grad_w_dV,
  double *grad_v_X_grad_w_dV
);
extern void calculateWeightedShapeTrace(
  proteus_index_t nElements_global,
  int nElementBoundaries_element,
  int nElementBoundaryQuadraturePoints_elementBoundary,
  int nDOF_test_element,
  double *dSR,
  double *sqrt_det_g,
  double *w,
  double *w_dS
);
extern void calculateShape_X_weightedShapeTrace(
  proteus_index_t nElements_global,
  int nElementBoundaries_element,
  int nElementBoundaryQuadraturePoints_elementBoundary,
  int nDOF_trial_element,
  int nDOF_test_element,
  double *v,
  double *w_dS,
  double *v_X_w_dS
);
extern void calculateGradShape_X_weightedShapeTrace(
  proteus_index_t nElements_global,
  int nElementBoundaries_element,
  int nElementBoundaryQuadraturePoints_elementBoundary,
  int nDOF_trial_element,
  int nDOF_test_element,
  int nSpace,
  double *grad_v,
  double *w_dS,
  double *grad_v_X_w_dS
);
extern void calculateIntegrationWeights(
  proteus_index_t nElements_global,
  int nQuadraturePoints_element,
  double *abs_det_J,
  double *referenceWeights,
  double *weights
);
extern void calculateElementBoundaryIntegrationWeights(
  proteus_index_t nElements_global,
  int nElementBoundaries_element,
  int nQuadraturePoints_elementBoundary,
  double *sqrt_det_g,
  double *referenceWeights,
  double *weights
);
extern void calculateFiniteElementFunctionValues(
  proteus_index_t nElements_global,
  int nQuadraturePoints_element,
  int nDOF_trial_element,
  int nComponents,
  proteus_index_t *l2g,
  double *dof,
  double *v,
  double *u
);
extern void calculateFiniteElementFunctionGradientValues(
  proteus_index_t nElements_global,
  int nQuadraturePoints_element,
  int nDOF_trial_element,
  int nComponents,
  int nSpace,
  proteus_index_t *l2g,
  double *dof,
  double *grad_v,
  double *grad_u
);
extern void calculateFiniteElementFunctionGradientTensorValues(
  proteus_index_t nElements_global,
  int nQuadraturePoints_element,
  int nDOF_trial_element,
  int nDOF_test_element,
  int nComponents,
  int nSpace,
  proteus_index_t *l2g,
  double *dof,
  double *grad_v_X_grad_w_dV,
  double *grad_u_X_grad_w_dV
);
extern void calculateFiniteElementFunctionValuesTrace(
  proteus_index_t nElements_global,
  int nElementBoundaries_element,
  int nQuadraturePoints_elementBoundary,
  int nDOF_trial_element,
  int nComponents,
  proteus_index_t *l2g,
  double *dof,
  double *v,
  double *u
);
extern void calculateFiniteElementFunctionGradientValuesTrace(
  proteus_index_t nElements_global,
  int nElementBoundaries_element,
  int nQuadraturePoints_elementBoundary,
  int nDOF_trial_element,
  int nComponents,
  int nSpace,
  proteus_index_t *l2g,
  double *dof,
  double *grad_v,
  double *grad_u
);
extern void calculateFlowVelocity(
  proteus_index_t nElements_global,
  int nQuadraturePoints_element,
  int nSpace,
  double *f,
  double *a,
  double *grad_phi,
  double *v
);
extern void updateAddJacobian_CSR(
  int jacIndex,
  double val,
  double *jac
);
extern void zeroJacobian_CSR(
  proteus_index_t nNonzeros,
  double *jac
);
extern void setInflowFlux(
  proteus_index_t nExteriorElementBoundaries_global,
  int nQuadraturePoints_elementBoundary,
  proteus_index_t *exteriorElementBoundaries,
  double *inflowFlux,
  double *flux
);
extern void calculateInteriorElementBoundaryVelocities(
  proteus_index_t nInteriorElementBoundaries_global,
  int nElementBoundaries_element,
  int nQuadraturePoints_elementBoundary,
  int nSpace,
  proteus_index_t *interiorElementBoundaries,
  proteus_index_t *elementBoundaryElements,
  int *elementBoundaryLocalElementBoundaries,
  double *m,
  double *a,
  double *grad_phi,
  double *f,
  double *vAverage,
  double *vJump,
  double *mAverage,
  double *mJump
);
extern void calculateExteriorElementBoundaryVelocities(
  proteus_index_t nExteriorElementBoundaries_global,
  int nElementBoundaries_element,
  int nQuadraturePoints_elementBoundary,
  int nSpace,
  proteus_index_t *exteriorElementBoundaries,
  proteus_index_t *elementBoundaryElements,
  int *elementBoundaryLocalElementBoundaries,
  double *m,
  double *a,
  double *grad_phi,
  double *f,
  double *vAverage,
  double *vJump,
  double *mAverage,
  double *mJump
);
extern void calculateConservationResidualPWL(
  proteus_index_t nElements_global,
  proteus_index_t nInteriorElementBoundaries_global,
  proteus_index_t nExteriorElementBoundaries_global,
  int nElementBoundaries_element,
  int nQuadraturePoints_elementBoundary,
  int nNodes_element,
  int nSpace,
  proteus_index_t *interiorElementBoundaries,
  proteus_index_t *exteriorElementBoundaries,
  proteus_index_t *elementBoundaryElements,
  int *elementBoundaryLocalElementBoundaries,
  proteus_index_t *elementNodes,
  proteus_index_t *nodeStarElements,
  proteus_index_t *nodeStarElementNeighbors,
  proteus_index_t *nodeStarOffsets,
  int *nElements_node,
  double *elementResidual,
  double *vAverage,
  double *starU,
  double *dX,
  double *w,
  double *normal,
  double *conservationResidual,
  double *starR,
  double *vConservative,
  double *vConservative_element
);
extern void calculateConservationJacobianPWL(
  proteus_index_t nNodes_global,
  proteus_index_t nNodes_internal,
  proteus_index_t nElements_global,
  proteus_index_t nInteriorElementBoundaries_global,
  proteus_index_t nExteriorElementBoundaries_global,
  int nElementBoundaries_element,
  int nQuadraturePoints_elementBoundary,
  int nNodes_element,
  int nSpace,
  proteus_index_t *interiorElementBoundaries,
  proteus_index_t *exteriorElementBoundaries,
  proteus_index_t *elementBoundaryElements,
  int *elementBoundaryLocalElementBoundaries,
  proteus_index_t *elementNodes,
  proteus_index_t *nodeStarElements,
  proteus_index_t *nodeStarElementNeighbors,
  proteus_index_t *nodeStarOffsets,
  proteus_index_t *nodeStarJacobianOffsets,
  int *nElements_node,
  proteus_index_t *internalNodes,
  double *w,
  double *normal,
  double *starJacobian
);
extern void calculateConservationFluxPWL(
  proteus_index_t nNodes_global,
  proteus_index_t nNodes_internal,
  int *nElements_node,
  proteus_index_t *nodeStarOffsets,
  proteus_index_t *nodeStarJacobianOffsets,
  proteus_index_t *internalNodes,
  double *starR,
  double *starJ,
  double *starU
);
extern void setExteriorGlobalElementBoundaryVelocityValues(
  int updateFluxValues,
  proteus_index_t nExteriorElementBoundaries_global,
  int nQuadraturePoints_elementBoundary,
  int nSpace,
  proteus_index_t *exteriorElementBoundaries,
  proteus_index_t *elementBoundaryElements,
  int *elementBoundaryLocalElementBoundaries,
  double *n,
  double *vn_in,
  double *v_out
);
extern void calculateDimensionlessNumbersADR(
  proteus_index_t nElements_global,
  int nQuadraturePoints_element,
  int nSpace,
  int computeDiffusiveTimeStepLimit,
  double *elementDiameter,
  double *df,
  double *a,
  double *dphi,
  double *dr,
  double *dmt,
  double *pe,
  double *cfl
);
extern void calculateDimensionlessNumbersADR_sd(proteus_index_t nElements_global,
                                         int nQuadraturePoints_element,
                                         int nSpace,
                                         int computeDiffusiveTimeStepLimit,
                                         proteus_csr_index_t* rowptr,
                                         proteus_csr_index_t* colind,
                                         double* elementDiameter,
                                         double* df,
                                         double* a,
                                         double* dphi,
                                         double* dr,
                                         double* dmt,
                                         double* pe,
                                         double* cfl);
extern void calculateCFLADR(
  proteus_index_t nElements_global,
  int nQuadraturePoints_element,
  int nSpace,
  double *elementDiameter,
  double *dm,
  double *df,
  double *cfl
);
extern void updateInteriorElementBoundaryDiffusiveVelocity(
  proteus_index_t nInteriorElementBoundaries_global,
  int nElementBoundaries_element,
  int nQuadraturePoints_elementBoundary,
  int nSpace,
  proteus_index_t *interiorElementBoundaries,
  proteus_index_t *elementBoundaryElements,
  int *elementBoundaryLocalElementBoundaries,
  double *a,
  double *grad_phi,
  double *velocity
);
extern void updateInteriorElementBoundaryDiffusiveVelocity_sd(proteus_index_t nInteriorElementBoundaries_global,
							      int nElementBoundaries_element,
							      int nQuadraturePoints_elementBoundary,
							      int nSpace,
							      proteus_csr_index_t* rowptr,
							      proteus_csr_index_t* colind,
							      proteus_index_t* interiorElementBoundaries,
							      proteus_index_t* elementBoundaryElements,
							      int* elementBoundaryLocalElementBoundaries,
							      double* a,
							      double* grad_phi,
							      double* velocity);
extern void updateExteriorElementBoundaryDiffusiveVelocity(
  proteus_index_t nExteriorElementBoundaries_global,
  int nElementBoundaries_element,
  int nQuadraturePoints_elementBoundary,
  int nSpace,
  proteus_index_t *exteriorElementBoundaries,
  proteus_index_t *elementBoundaryElements,
  int *elementBoundaryLocalElementBoundaries,
  double *a,
  double *grad_phi,
  double *velocity
);
extern void updateExteriorElementBoundaryDiffusiveVelocity_sd(proteus_index_t nExteriorElementBoundaries_global,
							      int nElementBoundaries_element,
							      int nQuadraturePoints_elementBoundary,
							      int nSpace,
							      proteus_csr_index_t* rowptr,
							      proteus_csr_index_t* colind,
							      proteus_index_t* exteriorElementBoundaries,
							      proteus_index_t* elementBoundaryElements,
							      int* elementBoundaryLocalElementBoundaries,
							      double* a,
							      double* grad_phi,
							      double* velocity);
extern void updateInteriorElementBoundaryAdvectiveVelocity(
  proteus_index_t nInteriorElementBoundaries_global,
  int nElementBoundaries_element,
  int nQuadraturePoints_elementBoundary,
  int nSpace,
  proteus_index_t *interiorElementBoundaries,
  proteus_index_t *elementBoundaryElements,
  int *elementBoundaryLocalElementBoundaries,
  double *f,
  double *velocity
);
extern void updateExteriorElementBoundaryAdvectiveVelocity(
  proteus_index_t nExteriorElementBoundaries_global,
  int nElementBoundaries_element,
  int nQuadraturePoints_elementBoundary,
  int nSpace,
  proteus_index_t *exteriorElementBoundaries,
  proteus_index_t *elementBoundaryElements,
  int *elementBoundaryLocalElementBoundaries,
  double *f,
  double *velocity
);
extern void updateInteriorElementBoundaryShockCapturingVelocity(
  proteus_index_t nInteriorElementBoundaries_global,
  int nElementBoundaries_element,
  int nQuadraturePoints_elementBoundary,
  int nQuadraturePoints_element,
  int nSpace,
  proteus_index_t *interiorElementBoundaries,
  proteus_index_t *elementBoundaryElements,
  int *elementBoundaryLocalElementBoundaries,
  double *numDiff,
  double *grad_u,
  double *velocity
);
extern void updateExteriorElementBoundaryShockCapturingVelocity(
  proteus_index_t nExteriorElementBoundaries_global,
  int nElementBoundaries_element,
  int nQuadraturePoints_elementBoundary,
  int nQuadraturePoints_element,
  int nSpace,
  proteus_index_t *exteriorElementBoundaries,
  proteus_index_t *elementBoundaryElements,
  int *elementBoundaryLocalElementBoundaries,
  double *numDiff,
  double *grad_u,
  double *velocity
);
extern void calculateInteriorElementBoundaryAverageVelocity(
  proteus_index_t nInteriorElementBoundaries_global,
  int nElementBoundaries_element,
  int nQuadraturePoints_elementBoundary,
  int nSpace,
  proteus_index_t *interiorElementBoundaries,
  proteus_index_t *elementBoundaryElements,
  int *elementBoundaryLocalElementBoundaries,
  double *v,
  double *vAverage
);
extern void calculateExteriorElementBoundaryAverageVelocity(
  proteus_index_t nExteriorElementBoundaries_global,
  int nElementBoundaries_element,
  int nQuadraturePoints_elementBoundary,
  int nSpace,
  proteus_index_t *exteriorElementBoundaries,
  proteus_index_t *elementBoundaryElements,
  int *elementBoundaryLocalElementBoundaries,
  double *v,
  double *vAverage
);
extern void calculateConservationResidualDG(
  proteus_index_t nElements_global,
  int nDOF_test_element,
  double *elementResidual,
  double *conservationResidual
);
extern void calculateConservationResidual(
  proteus_index_t nElements_global,
  int nDOF_test_element,
  int nElementBoundaries_element,
  int nQuadraturePoints_elementBoundary,
  int nSpace,
  double *n,
  double *dS_u,
  double *elementResidual,
  double *velocity,
  double *conservationResidual
);
extern void calculateConservationResidualGlobalBoundaries(
  proteus_index_t nElements_global,
  proteus_index_t nInteriorElementBoundaries_global,
  proteus_index_t nExteriorElementBoundaries_global,
  int nElementBoundaries_element,
  int nQuadraturePoints_elementBoundary,
  int nNodes_element,
  int nSpace,
  proteus_index_t *interiorElementBoundaries,
  proteus_index_t *exteriorElementBoundaries,
  proteus_index_t *elementBoundaryElements,
  int *elementBoundaryLocalElementBoundaries,
  double *dS,
  double *normal,
  double *elementResidual,
  double *velocity,
  double *conservationResidual
);
extern void copyGlobalElementBoundaryVelocityToElementBoundary(
  proteus_index_t nElements_global,
  proteus_index_t nInteriorElementBoundaries_global,
  proteus_index_t nExteriorElementBoundaries_global,
  proteus_index_t nElementBoundaries_global,
  int nElementBoundaries_element,
  int nQuadraturePoints_elementBoundary,
  int nSpace,
  proteus_index_t *interiorElementBoundaries,
  proteus_index_t *exteriorElementBoundaries,
  proteus_index_t *elementBoundaryElementsArray,
  int *elementBoundaryLocalElementBoundariesArray,
  double *velocityBoundary_global,
  double *velocityBoundary_element
);
extern void loadBoundaryFluxIntoGlobalElementBoundaryVelocity(
  proteus_index_t nExteriorElementBoundaries_global,
  int nQuadraturePoints_elementBoundary,
  int nSpace,
  proteus_index_t *exteriorElementBoundaries,
  proteus_index_t *fluxElementBoundaries,
  double *normal,
  double *flux,
  double updateCoef,
  double *velocity
);
extern void calculateInteriorNumericalTrace_Potential(
  proteus_index_t nInteriorElementBoundaries_global,
  int nElementBoundaries_element,
  int nQuadraturePoints_elementBoundary,
  proteus_index_t *interiorElementBoundaries,
  proteus_index_t *elementBoundaryElements,
  int *elementBoundaryLocalElementBoundaries,
  double *phi,
  double *dphi,
  double *phi_trace,
  double *dphi_trace_left,
  double *dphi_trace_right
);
extern void calculateExteriorNumericalTrace_Potential(
  int *isDOFBoundary,
  proteus_index_t nExteriorElementBoundaries_global,
  int nElementBoundaries_element,
  int nQuadraturePoints_elementBoundary,
  proteus_index_t *exteriorElementBoundaries,
  proteus_index_t *elementBoundaryElements,
  int *elementBoundaryLocalElementBoundaries,
  double *phi_bc,
  double *phi,
  double *dphi,
  double *phi_trace,
  double *dphi_trace_left
);
extern void updateInteriorElementBoundary_MixedForm_weak(
  proteus_index_t nInteriorElementBoundaries_global,
  int nElementBoundaries_element,
  int nQuadraturePoints_elementBoundary,
  int nDOF_test_element,
  int nSpace,
  proteus_index_t *interiorElementBoundaries,
  proteus_index_t *elementBoundaryElements,
  int *elementBoundaryLocalElementBoundaries,
  double *n,
  double *phi_trace,
  double *w_dS,
  double *b
);
extern void updateInteriorElementBoundary_MixedForm_weakJacobian(
  proteus_index_t nInteriorElementBoundaries_global,
  int nElementBoundaries_element,
  int nQuadraturePoints_elementBoundary,
  int nDOF_test_element,
  int nSpace,
  proteus_index_t *interiorElementBoundaries,
  proteus_index_t *elementBoundaryElements,
  int *elementBoundaryLocalElementBoundaries,
  double *n,
  double *dphi_trace_left,
  double *dphi_trace_right,
  double *v,
  double *w_dS,
  double *db,
  double *db_eb
);
extern void updateExteriorElementBoundary_MixedForm_weak(
  proteus_index_t nExteriorElementBoundaries_global,
  int nElementBoundaries_element,
  int nQuadraturePoints_elementBoundary,
  int nDOF_test_element,
  int nSpace,
  proteus_index_t *exteriorElementBoundaries,
  proteus_index_t *elementBoundaryElements,
  int *elementBoundaryLocalElementBoundaries,
  double *n,
  double *phi_trace,
  double *w_dS,
  double *b
);
extern void updateExteriorElementBoundary_MixedForm_weakJacobian(
  proteus_index_t nExteriorElementBoundaries_global,
  int nElementBoundaries_element,
  int nQuadraturePoints_elementBoundary,
  int nDOF_test_element,
  int nSpace,
  proteus_index_t *exteriorElementBoundaries,
  proteus_index_t *elementBoundaryElements,
  int *elementBoundaryLocalElementBoundaries,
  double *n,
  double *dphi_trace_left,
  double *v,
  double *w_dS,
  double *db,
  double *db_eb
);
extern void updatePotential_MixedForm_weak(
  proteus_index_t nElements_global,
  int nQuadraturePoints_element,
  int nDOF_test_element,
  int nSpace,
  double *phi,
  double *grad_w_dV,
  double *b
);
extern void updatePotential_MixedForm_weakJacobian(
  proteus_index_t nElements_global,
  int nQuadraturePoints_element,
  int nDOF_test_element,
  int nSpace,
  double *dphi,
  double *v,
  double *grad_w_dV,
  double *db
);
extern void calculateVelocityQuadrature_MixedForm(
  proteus_index_t nElements_global,
  int nElementBoundaries_element,
  int nElementBoundaryQuadraturePoints_elementBoundary,
  int nDOF_element,
  int nSpace,
  int nQuadraturePoints_element,
  double *A_inv,
  double *b,
  double *v,
  double *V,
  double *qv,
  double *qV
);
extern void calculateVelocityQuadrature_MixedForm_Jacobian(
  proteus_index_t nElements_global,
  int nElementBoundaries_element,
  int nElementBoundaryQuadraturePoints_elementBoundary,
  int nDOF_element,
  int nSpace,
  int nQuadraturePoints_element,
  double *A_inv,
  double *db,
  double *db_eb,
  double *v,
  double *DV,
  double *DV_eb,
  double *qv,
  double *qDV,
  double *qDV_eb
);
extern void calculateVelocityProjectionMatrixLDG(
  proteus_index_t nElements_global,
  int nQuadraturePoints_element,
  int nDOF_element,
  double *vXw_dV,
  double *A_inv
);
extern void updateDiffusion_MixedForm_weak(
  proteus_index_t nElements_global,
  int nQuadraturePoints_element,
  int nDOF_test_element,
  int nSpace,
  double *a,
  double *qV,
  double *grad_w_dV,
  double *weak_residual
);
extern void updateDiffusionJacobian_MixedForm_weak(
  proteus_index_t nElements_global,
  int nElementBoundaries_element,
  int nQuadraturePoints_element,
  int nDOF_trial_element,
  int nDOF_test_element,
  int nSpace,
  double *a,
  double *da,
  double *qV,
  double *qDV,
  double *qDV_eb,
  double *grad_w_dV,
  double *v,
  double *jacobian_weak_residual,
  double *jacobian_weak_residual_eb
);
extern void estimate_mt(
  proteus_index_t nElements_global,
  int nQuadraturePoints_element,
  int nDOF_element,
  double *v,
  double *vXw_dV,
  double *elementSpatialResidual,
  double *mt
  );
extern void estimate_mt_lowmem(proteus_index_t nElements_global,
                               int nQuadraturePoints_element,
                               int nDOF_element,
                               double* v,
                               double* w_dV,
                               double* elementSpatialResidual,
                               double* mt);
extern double scalarDomainIntegral(proteus_index_t nElements_global,
                                   int nQuadraturePoints_element,
                                   double* dV,
                                   double* nValueArray);
extern double scalarHeavisideDomainIntegral(proteus_index_t nElements_global,
                                            int nQuadraturePoints_element,
                                            double* dV,
                                            double* nValueArray);
extern double scalarSmoothedHeavisideDomainIntegral(proteus_index_t nElements_global,
						    int nQuadraturePoints_element,
						    double epsFact,
						    double* elementDiameter,
						    double* dV,
						    double* nValueArray);
extern double fluxDomainBoundaryIntegral(proteus_index_t nExteriorElementBoundaries,
					 proteus_index_t nElementBoundaries_owned,
                                         int nQuadraturePoints_elementBoundary,
                                         int* flag,
                                         proteus_index_t* exteriorElementBoundariesArray,
                                         double* dS,
                                         double* nValueArray);

extern double fluxDomainBoundaryIntegralFromVector(proteus_index_t nExteriorElementBoundaries,
						   proteus_index_t nElementBoundaries_owned,
						   int nQuadraturePoints_elementBoundary,
						   int nSpace,
						   int* flag,
						   proteus_index_t* exteriorElementBoundaries,
						   double* dS,
						   double* nValueArray,
						   double* normal);
extern
void copyExteriorElementBoundaryValuesFromElementBoundaryValues(proteus_index_t nExteriorElementBoundaries_global,
								proteus_index_t nElements_global,
								int nElementBoundaries_element,
								int nQuadraturePoints_elementBoundary,
								int nValuesPerQuadraturePoint,
								const proteus_index_t * exteriorElementBoundaries,
								const proteus_index_t* elementBoundaryElements,
								const int * elementBoundaryLocalElementBoundaries,
								const double * ebq_val,
								double * ebqe_val);

extern
void copyExteriorElementBoundaryValuesToElementBoundaryValues(proteus_index_t nExteriorElementBoundaries_global,
							      proteus_index_t nElements_global,
							      int nElementBoundaries_element,
							      int nQuadraturePoints_elementBoundary,
							      int nValuesPerQuadraturePoint,
							      const proteus_index_t * exteriorElementBoundaries,
							      const proteus_index_t* elementBoundaryElements,
							      const int * elementBoundaryLocalElementBoundaries,
							      const double * ebqe_val,
							      double * ebq_val);

extern
void copyExteriorElementBoundaryValuesToGlobalElementBoundaryValues(proteus_index_t nExteriorElementBoundaries_global,
								    int nQuadraturePoints_elementBoundary,
								    int nValuesPerQuadraturePoint,
								    const proteus_index_t * exteriorElementBoundaries,
								    const proteus_index_t* elementBoundaryElements,
								    const int * elementBoundaryLocalElementBoundaries,
								    const double * ebqe_val,
								    double * ebq_global_val);



extern
void copyExteriorElementBoundaryValuesFromGlobalElementBoundaryValues(proteus_index_t nExteriorElementBoundaries_global,
								      int nQuadraturePoints_elementBoundary,
								      int nValuesPerQuadraturePoint,
								      const proteus_index_t * exteriorElementBoundaries,
								      const proteus_index_t* elementBoundaryElements,
								      const int * elementBoundaryLocalElementBoundaries,
								      const double * ebq_global_val,
								      double * ebqe_val);




extern
void computeC0P1InterpolantDGP0(proteus_index_t nElements_global,
				proteus_index_t nNodes_global,
				int nNodes_element,
				int nDOF_element,
				int dim_dof,
				const proteus_index_t* elementNodesArray,
				const proteus_index_t* nodeElementOffsets,
				const proteus_index_t* nodeElementsArray,
				const proteus_index_t* l2g,
				const double * dof,
				double* nodalAverage);

extern
void computeC0P1InterpolantNCP1(proteus_index_t nElements_global,
				proteus_index_t nNodes_global,
				int nNodes_element,
				int nDOF_element,
				int dim_dof,
				const proteus_index_t* elementNodesArray,
				const proteus_index_t* nodeElementOffsets,
				const proteus_index_t* nodeElementsArray,
				const proteus_index_t* l2g,
				const double * dof,
				double* nodalAverage);

extern
void computeC0P1InterpolantDGP12(proteus_index_t nElements_global,
				 proteus_index_t nNodes_global,
				 int nNodes_element,
				 int nDOF_element,
				 int dim_dof,
				 const proteus_index_t* elementNodesArray,
				 const proteus_index_t* nodeElementOffsets,
				 const proteus_index_t* nodeElementsArray,
				 const proteus_index_t* l2g,
				 const double * dof,
				 double* nodalAverage);




extern
void parametricFiniteElementSpace_getValuesGlobalExteriorTrace(int nElementBoundaries_element,
							       int nElementBoundaryQuadraturePoints_elementBoundary,
							       int nDOF_element,
							       proteus_index_t nExteriorElementBoundaries_global,
							       const proteus_index_t* exteriorElementBoundariesArray,
							       const proteus_index_t* elementBoundaryElementsArray,
							       const int* elementBoundaryLocalElementBoundariesArray,
							       double* psi,
							       double* vArray);


extern
void parametricFiniteElementSpace_getGradientValuesGlobalExteriorTrace(int nElementBoundaries_element,
								       int nElementBoundaryQuadraturePoints_elementBoundary,
								       int nDOF_element,
								       int nSpace_global,
								       proteus_index_t nExteriorElementBoundaries_global,
								       const proteus_index_t *exteriorElementBoundariesArray,
								       const proteus_index_t *elementBoundaryElementsArray,
								       const int *elementBoundaryLocalElementBoundariesArray,
								       double* grad_psi,
								       double* inverseJacobianArray,
								       double* grad_vArray);


extern
void parametricMaps_getValuesGlobalExteriorTrace(int nQuadraturePoints_elementBoundary,
						 int nDOF_element,
						 int nSpace_global,
						 proteus_index_t nExteriorElementBoundaries_global,
						 const proteus_index_t* exteriorElementBoundariesArray,
						 const proteus_index_t* elementBoundaryElementsArray,
						 const int* elementBoundaryLocalElementBoundariesArray,
						 double* psi,
						 proteus_index_t* l2g,
						 double* nodeArray,
						 double* xArray);


extern
void parametricMaps_getInverseValuesGlobalExteriorTrace(int nElementBoundaryQuadraturePoints_elementBoundary,
							int nDOF_element,
							int nSpace_global,
							proteus_index_t nExteriorElementBoundaries_global,
							const proteus_index_t* exteriorElementBoundariesArray,
							const proteus_index_t* elementBoundaryElementsArray,
							const int* elementBoundaryLocalElementBoundariesArray,
							double* inverseJacobian,
							proteus_index_t* l2g,
							double* nodeArray,
							double* xArray,
							double* xiArray);



extern
void parametricMaps_getJacobianValuesGlobalExteriorTrace1D(int nQuadraturePoints_element,
							   int nDOF_element,
							   proteus_index_t nExteriorElementBoundaries_global,
							   const proteus_index_t * exteriorElementBoundariesArray,
							   const proteus_index_t * elementBoundaryElementsArray,
							   const int * elementBoundaryLocalElementBoundariesArray,
							   double* grad_psi,
							   double* boundaryNormals,
							   double* boundaryJacobians,
							   proteus_index_t* l2g,
							   double* nodeArray,
							   double* jacobianInverseArray,
							   double* metricTensorArray,
							   double* metricTensorDeterminantSqrtArray,
							   double* unitNormalArray);





extern
void parametricMaps_getJacobianValuesGlobalExteriorTrace2D(int nQuadraturePoints_element,
							   int nDOF_element,
							   proteus_index_t nExteriorElementBoundaries_global,
							   const proteus_index_t * exteriorElementBoundariesArray,
							   const proteus_index_t * elementBoundaryElementsArray,
							   const int * elementBoundaryLocalElementBoundariesArray,
							   double* grad_psi,
							   double* boundaryNormals,
							   double* boundaryJacobians,
							   proteus_index_t* l2g,
							   double* nodeArray,
							   double* jacobianInverseArray,
							   double* metricTensorArray,
							   double* metricTensorDeterminantSqrtArray,
							   double* unitNormalArray);
extern
void parametricMaps_getJacobianValuesGlobalExteriorTrace2D_movingDomain(int nQuadraturePoints_element,
									int nDOF_element,
									proteus_index_t nExteriorElementBoundaries_global,
									const proteus_index_t * exteriorElementBoundariesArray,
									const proteus_index_t * elementBoundaryElementsArray,
									const int * elementBoundaryLocalElementBoundariesArray,
									double* xtArray,
									double* grad_psi,
									double* boundaryNormals,
									double* boundaryJacobians,
									proteus_index_t* l2g,
									double* nodeArray,
									double* jacobianInverseArray,
									double* metricTensorArray,
									double* metricTensorDeterminantSqrtArray,
									double* unitNormalArray);
extern
void parametricMaps_getJacobianValuesGlobalExteriorTrace3D(int nQuadraturePoints_element,
							   int nDOF_element,
							   proteus_index_t nExteriorElementBoundaries_global,
							   const proteus_index_t * exteriorElementBoundariesArray,
							   const proteus_index_t * elementBoundaryElementsArray,
							   const int * elementBoundaryLocalElementBoundariesArray,
							   double* grad_psi,
							   double* boundaryNormals,
							   double* boundaryJacobians,
							   proteus_index_t* l2g,
							   double* nodeArray,
							   double* jacobianInverseArray,
							   double* metricTensorArray,
							   double* metricTensorDeterminantSqrtArray,
							   double* unitNormalArray);


extern
void updateGlobalExteriorElementBoundaryFlux(proteus_index_t nExteriorElementBoundaries_global,
					     int nQuadraturePoints_elementBoundary,
					     int nDOF_test_element,
					     proteus_index_t* exteriorElementBoundaries,
					     proteus_index_t* elementBoundaryElements,
					     int* elementBoundaryLocalElementBoundaries,
					     double* flux,
					     double* w_dS,
					     double* residual);



extern
void updateGlobalJacobianFromGlobalExteriorElementBoundaryFluxJacobian_eb_dense(proteus_index_t* elementNeighbors,
										proteus_index_t nElements_global,
										proteus_index_t nExteriorElementBoundaries_global,
										int nElementBoundaries_element,
										int nQuadraturePoints_elementBoundary,
										int nDOF_test_element,
										int nDOF_trial_element,
										proteus_index_t offset_r,
										proteus_index_t stride_r,
										proteus_index_t offset_u,
										proteus_index_t stride_u,
										proteus_index_t nFreeVDOF_global,
										proteus_index_t* exteriorElementBoundaries,
										proteus_index_t* elementBoundaryElements,
										int* elementBoundaryLocalElementBoundaries,
										int* nFreeDOF_element_r,
										int* freeLocal_r,
										proteus_index_t* freeGlobal_r,
										int* nFreeDOF_element_u,
										int* freeLocal_u,
										proteus_index_t* freeGlobal_u,
										double* elementBoundaryFluxJacobian_eb,
										double* w_dS,
										double* jac);

extern
void updateGlobalJacobianFromGlobalExteriorElementBoundaryFluxJacobian_dense(proteus_index_t nExteriorElementBoundaries_global,
									     int nQuadraturePoints_elementBoundary,
									     int nDOF_test_element,
									     int nDOF_trial_element,
									     proteus_index_t offset_r,
									     proteus_index_t stride_r,
									     proteus_index_t offset_u,
									     proteus_index_t stride_u,
									     proteus_index_t nFreeVDOF_global,
									     proteus_index_t* exteriorElementBoundaries,
									     proteus_index_t* elementBoundaryElements,
									     int* elementBoundaryLocalElementBoundaries,
									     int* nFreeDOF_element_r,
									     int* freeLocal_r,
									     proteus_index_t* freeGlobal_r,
									     int* nFreeDOF_element_u,
									     int* freeLocal_u,
									     proteus_index_t* freeGlobal_u,
									     double* elementBoundaryFluxJacobian,
									     double* w_dS,
									     double* jac);

extern
void updateGlobalJacobianFromGlobalExteriorElementBoundaryFluxJacobian_CSR(proteus_index_t nExteriorElementBoundaries_global,
									   int nQuadraturePoints_elementBoundary,
									   int nDOF_test_element,
									   int nDOF_trial_element,
									   proteus_index_t* exteriorElementBoundaries,
									   proteus_index_t* elementBoundaryElements,
									   int* elementBoundaryLocalElementBoundaries,
									   int* nFreeDOF_element_r,
									   int* freeLocal_r,
									   int* nFreeDOF_element_u,
									   int* freeLocal_u,
									   proteus_csr_index_t* csrRowIndeces_ru,
									   proteus_csr_index_t* csrColumnOffsets_eb_ru,
									   double* elementBoundaryFluxJacobian,
									   double* w_dS,
									   double* jac);



extern
void updateGlobalJacobianFromGlobalExteriorElementBoundaryFluxJacobian_eb_CSR(proteus_index_t* elementNeighbors,
									      proteus_index_t nExteriorElementBoundaries_global,
									      int nElementBoundaries_element,
									      int nQuadraturePoints_elementBoundary,
									      int nDOF_test_element,
									      int nDOF_trial_element,
									      proteus_index_t* exteriorElementBoundaries,
									      proteus_index_t* elementBoundaryElements,
									      int* elementBoundaryLocalElementBoundaries,
									      int* nFreeDOF_element_r,
									      int* freeLocal_r,
									      int* nFreeDOF_element_u,
									      int* freeLocal_u,
									      proteus_csr_index_t* csrRowIndeces_ru,
									      proteus_csr_index_t* csrColumnOffsets_eb_eNebN_ru,
									      double* elementBoundaryFluxJacobian_eb,
									      double* w_dS,
									      double* jac);

extern
void calculateWeightedShapeGlobalExteriorTrace(int nElementBoundaryQuadraturePoints_elementBoundary,
					       int nDOF_test_element,
					       proteus_index_t nExteriorElementBoundaries_global,
					       const proteus_index_t* exteriorElementBoundariesArray,
					       const proteus_index_t* elementBoundaryElementsArray,
					       const int* elementBoundaryLocalElementBoundariesArray,
					       double* dSR,
					       double* sqrt_det_g,
					       double* w,
					       double* w_dS);


extern
void calculateShape_X_weightedShapeGlobalExteriorTrace(int nElementBoundaryQuadraturePoints_elementBoundary,
						       int nDOF_trial_element,
						       int nDOF_test_element,
						       proteus_index_t nExteriorElementBoundaries_global,
						       const proteus_index_t* exteriorElementBoundariesArray,
						       const proteus_index_t* elementBoundaryElementsArray,
						       const int* elementBoundaryLocalElementBoundariesArray,
						       double* v,
						       double* w_dS,
						       double* v_X_w_dS);

extern
void calculateGradShape_X_weightedShapeGlobalExteriorTrace(int nElementBoundaryQuadraturePoints_elementBoundary,
							   int nDOF_trial_element,
							   int nDOF_test_element,
							   int nSpace,
							   proteus_index_t nExteriorElementBoundaries_global,
							   const proteus_index_t* exteriorElementBoundariesArray,
							   const proteus_index_t* elementBoundaryElementsArray,
							   const int* elementBoundaryLocalElementBoundariesArray,
							   double* grad_v,
							   double* w_dS,
							   double* grad_v_X_w_dS);


extern
void calculateGlobalExteriorElementBoundaryIntegrationWeights(int nQuadraturePoints_elementBoundary,
							      proteus_index_t nExteriorElementBoundaries_global,
							      double* sqrt_det_g,
							      double* referenceWeights,
							      double* weights);

extern
void calculateFiniteElementFunctionValuesGlobalExteriorTrace(int nQuadraturePoints_elementBoundary,
							     int nDOF_trial_element,
							     int nComponents,
							     proteus_index_t nExteriorElementBoundaries_global,
							     const proteus_index_t * exteriorElementBoundariesArray,
							     const proteus_index_t * elementBoundaryElementsArray,
							     const int * elementBoundaryLocalElementBoundariesArray,
							     proteus_index_t* l2g,
							     double* dof,
							     double* v,
							     double* u);

extern
void calculateFiniteElementFunctionGradientValuesGlobalExteriorTrace(int nQuadraturePoints_elementBoundary,
								     int nDOF_trial_element,
								     int nComponents,
								     int nSpace,
								     proteus_index_t nExteriorElementBoundaries_global,
								     const proteus_index_t * exteriorElementBoundariesArray,
								     const proteus_index_t * elementBoundaryElementsArray,
								     const int * elementBoundaryLocalElementBoundariesArray,
								     proteus_index_t* l2g,
								     double* dof,
								     double* grad_v,
								     double* grad_u);


extern
void updateGlobalExteriorElementBoundaryDiffusiveVelocity(proteus_index_t nExteriorElementBoundaries_global,
							  int nQuadraturePoints_elementBoundary,
							  int nSpace,
							  proteus_index_t* exteriorElementBoundaries,
							  proteus_index_t* elementBoundaryElements,
							  int* elementBoundaryLocalElementBoundaries,
							  double* a,
							  double* grad_phi,
							  double* velocity);


extern
void updateGlobalExteriorElementBoundaryAdvectiveVelocity(proteus_index_t nExteriorElementBoundaries_global,
							  int nQuadraturePoints_elementBoundary,
							  int nSpace,
							  proteus_index_t* exteriorElementBoundaries,
							  proteus_index_t* elementBoundaryElements,
							  int* elementBoundaryLocalElementBoundaries,
							  double* f,
							  double* velocity);


extern
void updateGlobalExteriorElementBoundaryShockCapturingVelocity(proteus_index_t nExteriorElementBoundaries_global,
							       int nQuadraturePoints_elementBoundary,
							       int nSpace,
							       proteus_index_t* exteriorElementBoundaries,
							       proteus_index_t* elementBoundaryElements,
							       int* elementBoundaryLocalElementBoundaries,
							       double* numDiff,
							       double* grad_u,
							       double* velocity);



extern
void copyFreeUnknownsToGlobalUnknowns(proteus_index_t nDOF2set,
				      proteus_index_t offset,
				      proteus_index_t stride,
				      const proteus_index_t* globalDOFids,
				      const proteus_index_t* freeDOFids,
				      const double * free_u,
				      double * u);
extern
void copyGlobalUnknownsToFreeUnknowns(proteus_index_t nDOF2set,
				      proteus_index_t offset,
				      proteus_index_t stride,
				      const proteus_index_t* globalDOFids,
				      const proteus_index_t* freeDOFids,
				      const double * u,
				      double * free_u);
extern
void updateInteriorElementBoundaryDiffusionAdjoint(proteus_index_t nInteriorElementBoundaries_global,
						   int nElementBoundaries_element,
						   int nQuadraturePoints_elementBoundary,
						   int nDOF_test_element,
						   int nSpace,
						   proteus_index_t* interiorElementBoundaries,
						   proteus_index_t* elementBoundaryElements,
						   int* elementBoundaryLocalElementBoundaries,
						   double sigma,
						   double* u,
						   double* n,
						   double* a,
						   double* grad_w,
						   double* dS,
						   double* residual);
extern
void updateExteriorElementBoundaryDiffusionAdjoint(proteus_index_t nExteriorElementBoundaries_global,
						   int nQuadraturePoints_elementBoundary,
						   int nDOF_test_element,
						   int nSpace,
						   int* isDOFBoundary,
						   proteus_index_t* exteriorElementBoundaries,
						   proteus_index_t* elementBoundaryElements,
						   int* elementBoundaryLocalElementBoundaries,
						   double sigma,
						   double* u,
						   double* ub,
						   double* n,
						   double* a,
						   double* grad_w,
						   double* dS,
						   double* residual);
extern
void updateGlobalJacobianFromInteriorElementBoundaryDiffusionAdjoint_dense(proteus_index_t nInteriorElementBoundaries_global,
									   int nElementBoundaries_element,
									   int nQuadraturePoints_elementBoundary,
									   int nDOF_test_element,
									   int nDOF_trial_element,
									   int nSpace,
									   proteus_index_t offset_r,
									   proteus_index_t stride_r,
									   proteus_index_t offset_u,
									   proteus_index_t stride_u,
									   proteus_index_t nFreeVDOF_global,
									   proteus_index_t* interiorElementBoundaries,
									   proteus_index_t* elementBoundaryElements,
									   int* elementBoundaryLocalElementBoundaries,
									   int* nFreeDOF_element_r,
									   int* freeLocal_r,
									   proteus_index_t* freeGlobal_r,
									   int* nFreeDOF_element_u,
									   int* freeLocal_u,
									   proteus_index_t* freeGlobal_u,
									   double sigma,
									   double* v,
									   double* n,
									   double* a,
									   double* grad_w,
									   double* dS,
									   double* jac);
extern
void updateGlobalJacobianFromExteriorElementBoundaryDiffusionAdjoint_dense(proteus_index_t nExteriorElementBoundaries_global,
									   int nQuadraturePoints_elementBoundary,
									   int nDOF_test_element,
									   int nDOF_trial_element,
									   int nSpace,
									   proteus_index_t offset_r,
									   proteus_index_t stride_r,
									   proteus_index_t offset_u,
									   proteus_index_t stride_u,
									   proteus_index_t nFreeVDOF_global,
									   proteus_index_t* exteriorElementBoundaries,
									   proteus_index_t* elementBoundaryElements,
									   int* elementBoundaryLocalElementBoundaries,
									   int* nFreeDOF_element_r,
									   int* freeLocal_r,
									   proteus_index_t* freeGlobal_r,
									   int* nFreeDOF_element_u,
									   int* freeLocal_u,
									   proteus_index_t* freeGlobal_u,
									   int* isDOFBoundary,
									   double sigma,
									   double* v,
									   double* n,
									   double* a,
									   double* grad_w,
									   double* dS,
									   double* jac);
extern
void updateInteriorElementBoundaryDiffusionAdjoint_sd(proteus_index_t nInteriorElementBoundaries_global,
						      int nElementBoundaries_element,
						      int nQuadraturePoints_elementBoundary,
						      int nDOF_test_element,
						      int nSpace,
						      proteus_csr_index_t* rowptr,
						      proteus_csr_index_t* colind,
						      proteus_index_t* interiorElementBoundaries,
						      proteus_index_t* elementBoundaryElements,
						      int* elementBoundaryLocalElementBoundaries,
						      double sigma,
						      double* u,
						      double* n,
						      double* a,
						      double* grad_w,
						      double* dS,
						      double* residual);
extern
void updateExteriorElementBoundaryDiffusionAdjoint_sd(proteus_index_t nExteriorElementBoundaries_global,
						      int nQuadraturePoints_elementBoundary,
						      int nDOF_test_element,
						      int nSpace,
						      proteus_csr_index_t* rowptr,
						      proteus_csr_index_t* colind,
						      int* isDOFBoundary,
						      proteus_index_t* exteriorElementBoundaries,
						      proteus_index_t* elementBoundaryElements,
						      int* elementBoundaryLocalElementBoundaries,
						      double sigma,
						      double* u,
						      double* ub,
						      double* n,
						      double* a,
						      double* grad_w,
						      double* dS,
						      double* residual);
extern
void updateGlobalJacobianFromInteriorElementBoundaryDiffusionAdjoint_dense_sd(proteus_index_t nInteriorElementBoundaries_global,
									      int nElementBoundaries_element,
									      int nQuadraturePoints_elementBoundary,
									      int nDOF_test_element,
									      int nDOF_trial_element,
									      int nSpace,
									      proteus_csr_index_t* rowptr,
									      proteus_csr_index_t* colind,
									      proteus_index_t offset_r,
									      proteus_index_t stride_r,
									      proteus_index_t offset_u,
									      proteus_index_t stride_u,
									      proteus_index_t nFreeVDOF_global,
									      proteus_index_t* interiorElementBoundaries,
									      proteus_index_t* elementBoundaryElements,
									      int* elementBoundaryLocalElementBoundaries,
									      int* nFreeDOF_element_r,
									      int* freeLocal_r,
									      proteus_index_t* freeGlobal_r,
									      int* nFreeDOF_element_u,
									      int* freeLocal_u,
									      proteus_index_t* freeGlobal_u,
									      double sigma,
									      double* v,
									      double* n,
									      double* a,
									      double* grad_w,
									      double* dS,
									      double* jac);
extern
void updateGlobalJacobianFromExteriorElementBoundaryDiffusionAdjoint_dense_sd(proteus_index_t nExteriorElementBoundaries_global,
									      int nQuadraturePoints_elementBoundary,
									      int nDOF_test_element,
									      int nDOF_trial_element,
									      int nSpace,
									      proteus_csr_index_t* rowptr,
									      proteus_csr_index_t* colind,
									      proteus_index_t offset_r,
									      proteus_index_t stride_r,
									      proteus_index_t offset_u,
									      proteus_index_t stride_u,
									      proteus_index_t nFreeVDOF_global,
									      proteus_index_t* exteriorElementBoundaries,
									      proteus_index_t* elementBoundaryElements,
									      int* elementBoundaryLocalElementBoundaries,
									      int* nFreeDOF_element_r,
									      int* freeLocal_r,
									      proteus_index_t* freeGlobal_r,
									      int* nFreeDOF_element_u,
									      int* freeLocal_u,
									      proteus_index_t* freeGlobal_u,
									      int* isDOFBoundary,
									      double sigma,
									      double* v,
									      double* n,
									      double* a,
									      double* grad_w,
									      double* dS,
									      double* jac);
extern
void updateGlobalJacobianFromInteriorElementBoundaryDiffusionAdjoint_CSR_sd(proteus_index_t nInteriorElementBoundaries_global,
									    int nElementBoundaries_element,
									    int nQuadraturePoints_elementBoundary,
									    int nDOF_test_element,
									    int nDOF_trial_element,
									    int nSpace,
									    proteus_csr_index_t* rowptr,
									    proteus_csr_index_t* colind,
									    proteus_index_t offset_r,
									    proteus_index_t stride_r,
									    proteus_index_t offset_u,
									    proteus_index_t stride_u,
									    proteus_index_t nFreeVDOF_global,
									    proteus_index_t* interiorElementBoundaries,
									    proteus_index_t* elementBoundaryElements,
									    int* elementBoundaryLocalElementBoundaries,
									    int* nFreeDOF_element_r,
									    int* freeLocal_r,
									    proteus_index_t* freeGlobal_r,
									    int* nFreeDOF_element_u,
									    int* freeLocal_u,
									    proteus_index_t* freeGlobal_u,
									    proteus_csr_index_t* csrRowIndeces_ru,
									    proteus_csr_index_t* csrColumnOffsets_eb_ru,
									    double sigma,
									    double* v,
									    double* n,
									    double* a,
									    double* grad_w,
									    double* dS,
									    double* jac);
extern
void updateGlobalJacobianFromExteriorElementBoundaryDiffusionAdjoint_CSR_sd(proteus_index_t nExteriorElementBoundaries_global,
									    int nQuadraturePoints_elementBoundary,
									    int nDOF_test_element,
									    int nDOF_trial_element,
									    int nSpace,
									    proteus_csr_index_t* rowptr,
									    proteus_csr_index_t* colind,
									    proteus_index_t offset_r,
									    proteus_index_t stride_r,
									    proteus_index_t offset_u,
									    proteus_index_t stride_u,
									    proteus_index_t nFreeVDOF_global,
									    proteus_index_t* exteriorElementBoundaries,
									    proteus_index_t* elementBoundaryElements,
									    int* elementBoundaryLocalElementBoundaries,
									    int* nFreeDOF_element_r,
									    int* freeLocal_r,
									    proteus_index_t* freeGlobal_r,
									    int* nFreeDOF_element_u,
									    int* freeLocal_u,
									    proteus_index_t* freeGlobal_u,
									    proteus_csr_index_t* csrRowIndeces_ru,
									    proteus_csr_index_t* csrColumnOffsets_eb_ru,
									    int* isDOFBoundary,
									    double sigma,
									    double* v,
									    double* n,
									    double* a,
									    double* grad_w,
									    double* dS,
									    double* jac);
extern
void update_f_movingDomain_q(proteus_index_t nElements_global,
			     int nQuadraturePoints_element,
			     int nSpace,
			     double* xt,
			     double* m,
			     double* f);
extern
void update_f_movingDomain_constantMass_q(proteus_index_t nElements_global,
					  int nQuadraturePoints_element,
					  int nSpace,
					  double* xt,
					  double* f);
extern
void update_f_movingDomain_ebq(proteus_index_t nElements_global,
			       int nElementBoundaries_element,
			       int nQuadraturePoints_elementBoundary,
			       int nSpace,
			       double* xt,
			       double* m,
			       double* f);
extern
void update_f_movingDomain_constantMass_ebq(proteus_index_t nElements_global,
					    int nElementBoundaries_element,
					    int nQuadraturePoints_elementBoundary,
					    int nSpace,
					    double* xt,
					    double* f);
extern
void updateStress_weak(proteus_index_t nElements_global,
		       int nQuadraturePoints_element,
		       int nDOF_test_element,
		       int nSpace,
		       double* sigma,
		       double* grad_w_dV,
		       double* weak_residual_x,
		       double* weak_residual_y,
		       double* weak_residual_z);
extern
void updateStressJacobian_weak(proteus_index_t nElements_global,
			       int nQuadraturePoints_element,
			       int nDOF_trial_element,
			       int nDOF_test_element,
			       int nSpace,
			       double* dsigma_xx,
			       double* dsigma_xy,
			       double* dsigma_xz,
			       double* dsigma_yx,
			       double* dsigma_yy,
			       double* dsigma_yz,
			       double* dsigma_zx,
			       double* dsigma_zy,
			       double* dsigma_zz,
			       double* grad_v,
			       double* grad_w_dV,
			       double* jacobian_weak_residual_xx,
			       double* jacobian_weak_residual_xy,
			       double* jacobian_weak_residual_xz,
			       double* jacobian_weak_residual_yx,
			       double* jacobian_weak_residual_yy,
			       double* jacobian_weak_residual_yz,
			       double* jacobian_weak_residual_zx,
			       double* jacobian_weak_residual_zy,
			       double* jacobian_weak_residual_zz);
extern
void projectFromNodalInterpolationConditions(proteus_index_t nElements_global,
					     int nDOF_element,
					     int dim_dof,
					     const proteus_index_t * l2g,
					     const int * functional_map_element,
					     const double * interpolationValues,
					     double * dofs);
extern void calculateVelocityQuadrature_MixedForm2_sd(proteus_index_t nElements_global,
                                               int nElementBoundaries_element,
                                               int nElementBoundaryQuadraturePoints_elementBoundary,
                                               int nDOF_element,
                                               int nSpace,
                                               int nQuadraturePoints_element,
                                               const proteus_csr_index_t * rowptr,
                                               const proteus_csr_index_t * colind,
                                               double* qa,
                                               double* qw_dV,
                                               double* b,
                                               double* v,
                                               double* V,
                                               double* qv,
                                               double* qV);
extern void updateDiffusion_MixedForm_weak_sd(proteus_index_t nElements_global,
                                       int nQuadraturePoints_element,
                                       int nDOF_test_element,
                                       int nSpace,
                                       int rho_split,
                                       proteus_csr_index_t* rowptr,
                                       proteus_csr_index_t* colind,
                                       double* a,
                                       double* qV,
                                       double* grad_w_dV,
                                       double* velocity, /* added for ldg coupling */
                                       double* weak_residual);
extern void calculateVelocityQuadrature_MixedForm2_Jacobian_sd(proteus_index_t nElements_global,
                                                        int nElementBoundaries_element,
                                                        int nElementBoundaryQuadraturePoints_elementBoundary,
                                                        int nDOF_element,
                                                        int nSpace,
                                                        int nQuadraturePoints_element,
                                                        const proteus_csr_index_t *rowptr,
                                                        const proteus_csr_index_t *colind,
                                                        double* qa,
                                                        double* qw_dV,
                                                        double* db,
                                                        double* db_eb,
                                                        double* v,
                                                        double* DV,
                                                        double* DV_eb,
                                                        double* qv,
                                                        double* qDV,
                                                        double* qDV_eb);
extern void updateDiffusionJacobian_MixedForm_weak_sd(proteus_index_t nElements_global,
                                               int nElementBoundaries_element,
                                               int nQuadraturePoints_element,
                                               int nDOF_trial_element,
                                               int nDOF_test_element,
                                               int nSpace,
                                               proteus_csr_index_t* rowptr,
                                               proteus_csr_index_t* colind,
                                               double* a,
                                               double* da,
                                               double* qV,
                                               double* qDV,
                                               double* qDV_eb,
                                               double* grad_w_dV,
                                               double* v,
                                               double* jacobian_weak_residual,
                                               double* jacobian_weak_residual_eb);
/** @} */
#endif
