# A type of -*- python -*- file
# cython: language_level=3
import numpy as np
cimport numpy as np

cdef extern from "index_type.h":
     ctypedef int proteus_index_t
     ctypedef int proteus_csr_index_t

cdef extern from "femIntegrals.h":
     void cparametricFiniteElementSpace_getHessianValues "parametricFiniteElementSpace_getHessianValues"(proteus_index_t nElements_global,
     							                                                 int nQuadraturePoints_element,
							                                                 int nDOF_element,
							                                                 int nSpace_global,
							                                                 double* Hessian_psi,
							                                                 double* inverseJacobianArray,
							                                                 double* Hessian_vArray)
     void cupdateDiffusion2_strong "updateDiffusion2_strong"(proteus_index_t nElements_global,
				                             int nQuadraturePoints_element,
				                             int nSpace,
				                             double *a,
				                             double *Hess_phi,
				                             double *strong_residual)
     void cupdateDiffusion2_strong_sd "updateDiffusion2_strong_sd"(proteus_index_t nElements_global,
                                                                   int nQuadraturePoints_element,
                                                                   int nSpace,
                                                                   proteus_csr_index_t* rowptr,
                                                                   proteus_csr_index_t* colind,
                                                                   double* a,
                                                                   double* Hess_phi,
                                                                   double* strong_residual)
     void cupdateDiffusionJacobian2_strong "updateDiffusionJacobian2_strong"(proteus_index_t nElements_global,
					                                     int nQuadraturePoints_element,
					                                     int nDOF_trial_element,
					                                     int nSpace,
					                                     proteus_index_t *l2g,
					                                     double *a,
					                                     double *da,
					                                     double *v,
					                                     double *Hess_phi,
					                                     double *dphi,
					                                     double *Hess_v,
					                                     double *dstrong_residual)
     void cupdateDiffusionJacobian2_strong_sd "updateDiffusionJacobian2_strong_sd"(proteus_index_t nElements_global,
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
                                                                                   double* dstrong_residual)
     void cupdateDiffusion2_adjoint "updateDiffusion2_adjoint"(proteus_index_t nElements_global,
				                               int nQuadraturePoints_element,
				                               int nDOF_test_element,
				                               int nSpace,
				                               double *a,
				                               double *Hess_w_dV,
				                               double *Lstar_w_dV)
     void cupdateDiffusion2_adjoint_sd "updateDiffusion2_adjoint_sd"(proteus_index_t nElements_global,
                                                                     int nQuadraturePoints_element,
                                                                     int nDOF_test_element,
                                                                     int nSpace,
                                                                     proteus_csr_index_t* rowptr,
                                                                     proteus_csr_index_t* colind,
                                                                     double* a,
                                                                     double* Hess_w_dV,
                                                                     double* Lstar_w_dV)
     void ccalculateWeightedShapeHessians "calculateWeightedShapeHessians"(proteus_index_t nElements_global,
					                                   int nQuadraturePoints_element,
					                                   int nDOF_test_element,
					                                   int nSpace,
					                                   double *dVR,
					                                   double *abs_det_jac,
					                                   double *Hess_w,
					                                   double *Hess_w_dV)
     void ccalculateFiniteElementFunctionHessianValues "calculateFiniteElementFunctionHessianValues"(proteus_index_t nElements_global,
						                                                     int nQuadraturePoints_element,
						                                                     int nDOF_trial_element,
						                                                     int nComponents,
						                                                     int nSpace,
						                                                     proteus_index_t *l2g,
						                                                     double *dof,
						                                                     double *Hessian_v,
						                                                     double *Hessian_u)
     void cupdateGlobalJacobianFromInteriorElementBoundaryFluxJacobian_2sided_CSR "updateGlobalJacobianFromInteriorElementBoundaryFluxJacobian_2sided_CSR"(proteus_index_t nInteriorElementBoundaries_global,
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
										                                                                           double *elementBoundaryFluxJacobian_2sided,
										                                                                           double *w_dS,
										                                                                           double *jac)
     void cupdateGlobalJacobianFromInteriorElementBoundaryFluxJacobian_2sided_dense "updateGlobalJacobianFromInteriorElementBoundaryFluxJacobian_2sided_dense"(proteus_index_t nInteriorElementBoundaries_global,
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
										                                                                               double *elementBoundaryFluxJacobian_2sided,
										                                                                               double *w_dS,
										                                                                               double *jac)
     void cupdateInteriorTwoSidedElementBoundaryFlux "updateInteriorTwoSidedElementBoundaryFlux"(proteus_index_t nInteriorElementBoundaries_global,
						                                                 int nElementBoundaries_element,
						                                                 int nQuadraturePoints_elementBoundary,
						                                                 int nDOF_test_element,
						                                                 proteus_index_t *interiorElementBoundaries,
						                                                 proteus_index_t *elementBoundaryElements,
						                                                 int *elementBoundaryLocalElementBoundaries,
						                                                 double *flux,
						                                                 double *w_dS,
						                                                 double *residual)
     void ccalculateCFLADR2speeds "calculateCFLADR2speeds"(proteus_index_t nElements_global,
				                           int nQuadraturePoints_element,
				                           int nSpace,
				                           double *elementDiameter,
				                           double *dm,
				                           double *df1,
				                           double *df2,
				                           double *cfl)
     int ccheckElementBoundaryAndExteriorElementBoundaryArraysSame "checkElementBoundaryAndExteriorElementBoundaryArraysSame"(int nElementBoundaries_element,
								                                                              proteus_index_t nExteriorElementBoundaries_global,
								                                                              int nQuadraturePoints_elementBoundary,
								                                                              int nValuesPerQuadraturePoint,
								                                                              double tolerance,
								                                                              proteus_index_t *exteriorElementBoundariesArray,
								                                                              proteus_index_t *elementBoundaryElementsArray,
								                                                              int *elementBoundaryLocalElementBoundariesArray,
								                                                              double *ebq_val,
								                                                              double *ebqe_val,
								                                                              proteus_index_t *firstBadIndex)
     int ccheckGlobalElementBoundaryAndExteriorElementBoundaryArraysSame "checkGlobalElementBoundaryAndExteriorElementBoundaryArraysSame"(proteus_index_t nExteriorElementBoundaries_global,
									                                                                  int nQuadraturePoints_elementBoundary,
									                                                                  int nValuesPerQuadraturePoint,
									                                                                  double tolerance,
									                                                                  proteus_index_t *exteriorElementBoundariesArray,
									                                                                  proteus_index_t *elementBoundaryElementsArray,
									                                                                  int *elementBoundaryLocalElementBoundariesArray,
									                                                                  double *ebq_global_val,
									                                                                  double *ebqe_val,
									                                                                  proteus_index_t *firstBadIndex)
     void ccalculateExteriorElementBoundaryStress3D "calculateExteriorElementBoundaryStress3D"(proteus_index_t nExteriorElementBoundaries_global,
						                                               int nQuadraturePoints_elementBoundary,
						                                               int *elementBoundaryMaterialTypes,
						                                               proteus_index_t *exteriorElementBoundaries,
						                                               proteus_index_t *elementBoundaryElements,
						                                               int *elementBoundaryLocalElementBoundaries,
						                                               double *p,
						                                               double *mom_flux_vec_u,
						                                               double *mom_flux_vec_v,
						                                               double *mom_flux_vec_w,
						                                               double *dS,
						                                               double *n,
						                                               double *F)
     void ccalculateExteriorElementBoundaryStress2D "calculateExteriorElementBoundaryStress2D"(proteus_index_t nExteriorElementBoundaries_global,
						                                               int nQuadraturePoints_elementBoundary,
						                                               int *elementBoundaryMaterialTypes,
						                                               proteus_index_t *exteriorElementBoundaries,
						                                               proteus_index_t *elementBoundaryElements,
						                                               int *elementBoundaryLocalElementBoundaries,
						                                               double *p,
						                                               double *mom_flux_vec_u,
						                                               double *mom_flux_vec_v,
						                                               double *dS,
						                                               double *n,
						                                               double *F)
     void ccopyLeftElementBoundaryInfo "copyLeftElementBoundaryInfo"(int nElementBoundaries_element,
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
                                                                     double *ng)
     void cparametricFiniteElementSpace_getValues "parametricFiniteElementSpace_getValues"(proteus_index_t nElements_global,
                                                                                           int nQuadraturePoints_element,
                                                                                           int nDOF_element,
                                                                                           double *psi,
                                                                                           double *vArray)
     void cparametricFiniteElementSpace_getValuesTrace "parametricFiniteElementSpace_getValuesTrace"(proteus_index_t nElements_global,
                                                                                                     int nElementBoundaries_element,
                                                                                                     int nElementBoundaryQuadraturePoints_elementBoundary,
                                                                                                     int nDOF_element,
                                                                                                     double *psi,
                                                                                                     int *permutations,
                                                                                                     double *vArray)
     void cparametricFiniteElementSpace_getGradientValues "parametricFiniteElementSpace_getGradientValues"(proteus_index_t nElements_global,
                                                                                                           int nQuadraturePoints_element,
                                                                                                           int nDOF_element,
                                                                                                           int nSpace_global,
                                                                                                           double *grad_psi,
                                                                                                           double *inverseJacobianArray,
                                                                                                           double *grad_vArray)
     void cparametricFiniteElementSpace_getGradientValuesTrace "parametricFiniteElementSpace_getGradientValuesTrace"(proteus_index_t nElements_global,
                                                                                                                     int nElementBoundaries_element,
                                                                                                                     int nElementBoundaryQuadraturePoints_elementBoundary,
                                                                                                                     int nDOF_element,
                                                                                                                     int nSpace_global,
                                                                                                                     double *grad_psi,
                                                                                                                     int *permutations,
                                                                                                                     double *inverseJacobianArray,
                                                                                                                     double *grad_vArray)
     void cparametricMaps_getPermutations "parametricMaps_getPermutations"(proteus_index_t nElements_global,
                                                                           int nElementBoundaries_element,
                                                                           int nElementBoundaryQuadraturePoints_elementBoundary,
                                                                           int nSpace_global,
                                                                           double *xiArray,
                                                                           int *permutations)
     void cparametricMaps_getPermutationsGlobalExterior "parametricMaps_getPermutationsGlobalExterior"(int nElementBoundaryQuadraturePoints_elementBoundary,
						                                                       int nSpace_global,
						                                                       proteus_index_t nExteriorElementBoundaries_global,
						                                                       proteus_index_t* exteriorElementBoundariesArray,
						                                                       proteus_index_t* elementBoundaryElementsArray,
						                                                       int* elementBoundaryLocalElementBoundariesArray,
						                                                       double* xiArray,
						                                                       int* permutations);
     void cgetPermutationsGlobal "getPermutationsGlobal"(proteus_index_t nElementBoundaries_global,
				                         int nElementBoundaryQuadraturePoints_elementBoundary,
				                         double* xArray,
				                         double* xArrayNew,
				                         int* permutations);
     void cparametricMaps_getValues "parametricMaps_getValues"(proteus_index_t nElements_global,
                                                               int nQuadraturePoints_element,
                                                               int nDOF_element,
                                                               int nSpace_global,
                                                               double *psi,
                                                               proteus_index_t *l2g,
                                                               double *nodeArray,
                                                               double *xArray)
     void cparametricMaps_getValuesTrace "parametricMaps_getValuesTrace"(proteus_index_t nElements_global,
                                                                         int nElementBoundaries_element,
                                                                         int nQuadraturePoints_element,
                                                                         int nDOF_element,
                                                                         int nSpace_global,
                                                                         double *psi,
                                                                         proteus_index_t *l2g,
                                                                         double *nodeArray,
                                                                         double *xArray)
     void cparametricMaps_getInverseValues "parametricMaps_getInverseValues"(proteus_index_t nElements_global,
                                                                             int nQuadraturePoints_element,
                                                                             int nDOF_element,
                                                                             int nSpace_global,
                                                                             double *inverseJacobian,
                                                                             proteus_index_t *l2g,
                                                                             double *nodeArray,
                                                                             double *xArray,
                                                                             double *xiArray)
     void cparametricMaps_getInverseValuesTrace "parametricMaps_getInverseValuesTrace"(proteus_index_t nElements_global,
                                                                                       int nElementBoundaries_element,
                                                                                       int nElementBoundaryQuadraturePoints_elementBoundary,
                                                                                       int nDOF_element,
                                                                                       int nSpace_global,
                                                                                       double *inverseJacobian,
                                                                                       proteus_index_t *l2g,
                                                                                       double *nodeArray,
                                                                                       double *xArray,
                                                                                       double *xiArray)
     void cparametricMaps_getJacobianValues3D "parametricMaps_getJacobianValues3D"(proteus_index_t nElements_global,
                                                                                   int nQuadraturePoints_element,
                                                                                   int nDOF_element,
                                                                                   double *grad_psi,
                                                                                   proteus_index_t *l2g,
                                                                                   double *nodeArray,
                                                                                   double *jacobianArray,
                                                                                   double *jacobianDeterminantArray,
                                                                                   double *jacobianInverseArray)
     void cparametricMaps_getJacobianValues2D "parametricMaps_getJacobianValues2D"(proteus_index_t nElements_global,
                                                                                   int nQuadraturePoints_element,
                                                                                   int nDOF_element,
                                                                                   double *grad_psi,
                                                                                   proteus_index_t *l2g,
                                                                                   double *nodeArray,
                                                                                   double *jacobianArray,
                                                                                   double *jacobianDeterminantArray,
                                                                                   double *jacobianInverseArray)
     void cparametricMaps_getJacobianValues1D "parametricMaps_getJacobianValues1D"(proteus_index_t nElements_global,
                                                                                   int nQuadraturePoints_element,
                                                                                   int nDOF_element,
                                                                                   double *grad_psi,
                                                                                   proteus_index_t *l2g,
                                                                                   double *nodeArray,
                                                                                   double *jacobianArray,
                                                                                   double *jacobianDeterminantArray,
                                                                                   double *jacobianInverseArray)
     void cparametricMaps_getJacobianValuesTrace3D "parametricMaps_getJacobianValuesTrace3D"(proteus_index_t nElements_global,
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
                                                                                             double *unitNormalArray)
     void cparametricMaps_getJacobianValuesTrace2D "parametricMaps_getJacobianValuesTrace2D"(proteus_index_t nElements_global,
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
                                                                                             double *unitNormalArray)
     void cparametricMaps_getJacobianValuesTrace1D "parametricMaps_getJacobianValuesTrace1D"(proteus_index_t nElements_global,
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
                                                                                             double *unitNormalArray)
     void cupdateMass_weak "updateMass_weak"(proteus_index_t nElements_global,
                                             int nQuadraturePoints_element,
                                             int nDOF_test_element,
                                             double *mt,
                                             double *w_dV,
                                             double *weak_residual)
     void cupdateMassJacobian_weak "updateMassJacobian_weak"(proteus_index_t nElements_global,
                                                             int nQuadraturePoints_element,
                                                             int nDOF_trial_element,
                                                             int nDOF_test_element,
                                                             double *dmt,
                                                             double *v_X_w_dV,
                                                             double *jacobian_weak_residual)
     void cupdateMassJacobian_weak_lowmem "updateMassJacobian_weak_lowmem"(proteus_index_t nElements_global,
                                                                           int nQuadraturePoints_element,
                                                                           int nDOF_trial_element,
                                                                           int nDOF_test_element,
                                                                           double* dmt,
                                                                           double* v,
                                                                           double* w_dV,
                                                                           double* jacobian_weak_residual)
     void cupdateMass_strong "updateMass_strong"(proteus_index_t nElements_global,
                                                 int nQuadraturePoints_element,
                                                 double *mt,
                                                 double *strong_residual)
     void cupdateMassJacobian_strong "updateMassJacobian_strong"(proteus_index_t nElements_global,
                                                                 int nQuadraturePoints_element,
                                                                 int nDOF_trial_element,
                                                                 double *dmt,
                                                                 double *v,
                                                                 double *dstrong_residual)
     void cupdateMass_adjoint "updateMass_adjoint"(proteus_index_t nElements_global,
                                                   int nQuadraturePoints_element,
                                                   int nDOF_test_element,
                                                   double *dmt,
                                                   double *w_dV,
                                                   double *Lstar_w_dV)
     void cupdateAdvection_weak "updateAdvection_weak"(proteus_index_t nElements_global,
                                                       int nQuadraturePoints_element,
                                                       int nDOF_test_element,
                                                       int nSpace,
                                                       double *f,
                                                       double *grad_w_dV,
                                                       double *weak_residual)
     void cupdateAdvectionJacobian_weak "updateAdvectionJacobian_weak"(proteus_index_t nElements_global,
                                                                       int nQuadraturePoints_element,
                                                                       int nDOF_trial_element,
                                                                       int nDOF_test_element,
                                                                       int nSpace,
                                                                       double *df,
                                                                       double *v_X_grad_w_dV,
                                                                       double *jacobian_weak_residual)
     void cupdateAdvectionJacobian_weak_lowmem "updateAdvectionJacobian_weak_lowmem"(proteus_index_t nElements_global,
                                                                                     int nQuadraturePoints_element,
                                                                                     int nDOF_trial_element,
                                                                                     int nDOF_test_element,
                                                                                     int nSpace,
                                                                                     double* df,
                                                                                     double* v,
                                                                                     double* grad_w_dV,
                                                                                     double* jacobian_weak_residual)
     void cupdateAdvection_strong "updateAdvection_strong"(proteus_index_t nElements_global,
                                                           int nQuadraturePoints_element,
                                                           int nSpace,
                                                           double *df,
                                                           double *grad_u,
                                                           double *strong_residual)
     void cupdateAdvectionJacobian_strong "updateAdvectionJacobian_strong"(proteus_index_t nElements_global,
                                                                           int nQuadraturePoints_element,
                                                                           int nDOF_trial_element,
                                                                           int nSpace,
                                                                           double *df,
                                                                           double *grad_v,
                                                                           double *dstrong_residual)
     void cupdateAdvection_adjoint "updateAdvection_adjoint"(proteus_index_t nElements_global,
                                                             int nQuadraturePoints_element,
                                                             int nDOF_test_element,
                                                             int nSpace,
                                                             double *df,
                                                             double *grad_w_dV,
                                                             double *Lstar_w_dV)
     void cupdateHamiltonian_weak "updateHamiltonian_weak"(proteus_index_t nElements_global,
                                                           int nQuadraturePoints_element,
                                                           int nDOF_test_element,
                                                           double *H,
                                                           double *w_dV,
                                                           double *weak_residual)
     void cupdateHamiltonianJacobian_weak "updateHamiltonianJacobian_weak"(proteus_index_t nElements_global,
                                                                           int nQuadraturePoints_element,
                                                                           int nDOF_trial_element,
                                                                           int nDOF_test_element,
                                                                           int nSpace,
                                                                           double *dH,
                                                                           double *grad_v_X_w_dV,
                                                                           double *jacobian_weak_residual)
     void cupdateHamiltonianJacobian_weak_lowmem "updateHamiltonianJacobian_weak_lowmem"(proteus_index_t nElements_global,
                                                                                         int nQuadraturePoints_element,
                                                                                         int nDOF_trial_element,
                                                                                         int nDOF_test_element,
                                                                                         int nSpace,
                                                                                         double* dH,
                                                                                         double* grad_v,
                                                                                         double* w_dV,
                                                                                         double* jacobian_weak_residual)
     void cupdateHamiltonian_strong "updateHamiltonian_strong"(proteus_index_t nElements_global,
                                                               int nQuadraturePoints_element,
                                                               int nSpace,
                                                               double *dH,
                                                               double *grad_u,
                                                               double *strong_residual)
     void cupdateHamiltonianJacobian_strong "updateHamiltonianJacobian_strong"(proteus_index_t nElements_global,
                                                                               int nQuadraturePoints_element,
                                                                               int nDOF_trial_element,
                                                                               int nSpace,
                                                                               double *dH,
                                                                               double *grad_v,
                                                                               double *dstrong_residual)
     void cupdateHamiltonian_adjoint "updateHamiltonian_adjoint"(proteus_index_t nElements_global,
                                                                 int nQuadraturePoints_element,
                                                                 int nDOF_test_element,
                                                                 int nSpace,
                                                                 double *dH,
                                                                 double *grad_w_dV,
                                                                 double *Lstar_w_dV)
     void cupdateDiffusion_weak "updateDiffusion_weak"(proteus_index_t nElements_global,
                                                       int nQuadraturePoints_element,
                                                       int nDOF_test_element,
                                                       int nSpace,
                                                       double *a,
                                                       double *grad_phi_X_grad_w_dV,
                                                       double *weak_residual)
     void cupdateDiffusion_weak_lowmem "updateDiffusion_weak_lowmem"(proteus_index_t nElements_global,
                                                                     int nQuadraturePoints_element,
                                                                     int nDOF_test_element,
                                                                     int nSpace,
                                                                     double* a,
                                                                     double* grad_phi,
                                                                     double* grad_w_dV,
                                                                     double* weak_residual)
     void cupdateDiffusion_weak_sd "updateDiffusion_weak_sd"(proteus_index_t nElements_global,
                                                             int nQuadraturePoints_element,
                                                             int nDOF_test_element,
                                                             int nSpace,
                                                             proteus_csr_index_t* rowptr,
                                                             proteus_csr_index_t* colind,
                                                             double* a,
                                                             double* grad_phi,
                                                             double* grad_w_dV,
                                                             double* weak_residual)
     void cupdateDiffusionJacobian_weak "updateDiffusionJacobian_weak"(proteus_index_t nElements_global,
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
                                                                       double *jacobian_weak_residual)
     void cupdateDiffusionJacobian_weak_lowmem "updateDiffusionJacobian_weak_lowmem"(proteus_index_t nElements_global,
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
                                                                                     double* jacobian_weak_residual)
     void cupdateDiffusionJacobian_weak_sd "updateDiffusionJacobian_weak_sd"(proteus_index_t nElements_global,
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
					                                     double* jacobian_weak_residual)
     void cupdateDiffusion_strong "updateDiffusion_strong"(proteus_index_t nElements_global,
                                                           int nQuadraturePoints_element,
                                                           int nSpace,
                                                           double *da,
                                                           double *grad_phi,
                                                           double *grad_u,
                                                           double *strong_residual)
     void cupdateDiffusion_strong_sd "updateDiffusion_strong_sd"(proteus_index_t nElements_global,
                                                                 int nQuadraturePoints_element,
                                                                 int nSpace,
                                                                 proteus_csr_index_t* rowptr,
                                                                 proteus_csr_index_t* colind,
                                                                 double* da,
                                                                 double* grad_phi,
                                                                 double* grad_u,
                                                                 double* strong_residual)
     void cupdateDiffusionJacobian_strong "updateDiffusionJacobian_strong"(proteus_index_t nElements_global,
                                                                           int nQuadraturePoints_element,
                                                                           int nDOF_trial_element,
                                                                           int nSpace,
                                                                           proteus_index_t *l2g,
                                                                           double *da,
                                                                           double *dphi,
                                                                           double *grad_phi,
                                                                           double *grad_u,
                                                                           double *grad_v,
                                                                           double *dstrong_residual)
     void cupdateDiffusionJacobian_strong_sd "updateDiffusionJacobian_strong_sd"(proteus_index_t nElements_global,
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
                                                                                 double* dstrong_residual)
     void cupdateDiffusion_adjoint "updateDiffusion_adjoint"(proteus_index_t nElements_global,
                                                             int nQuadraturePoints_element,
                                                             int nDOF_test_element,
                                                             int nSpace,
                                                             double *da,
                                                             double *grad_phi,
                                                             double *grad_w_dV,
                                                             double *Lstar_w_dV)
     void cupdateDiffusion_adjoint_sd "updateDiffusion_adjoint_sd"(proteus_index_t nElements_global,
				                                   int nQuadraturePoints_element,
				                                   int nDOF_test_element,
				                                   int nSpace,
				                                   proteus_csr_index_t* rowptr,
				                                   proteus_csr_index_t* colind,
				                                   double* da,
				                                   double* grad_phi,
				                                   double* grad_w_dV,
				                                   double* Lstar_w_dV)
     void cupdateReaction_weak "updateReaction_weak"(proteus_index_t nElements_global,
                                                     int nQuadraturePoints_element,
                                                     int nDOF_test_element,
                                                     double *r,
                                                     double *w_dV,
                                                     double *weak_residual)
     void cupdateReactionJacobian_weak "updateReactionJacobian_weak"(proteus_index_t nElements_global,
                                                                     int nQuadraturePoints_element,
                                                                     int nDOF_trial_element,
                                                                     int nDOF_test_element,
                                                                     double *dr,
                                                                     double *v_X_w_dV,
                                                                     double *jacobian_weak_residual)
     void cupdateReactionJacobian_weak_lowmem "updateReactionJacobian_weak_lowmem"(proteus_index_t nElements_global,
                                                                                   int nQuadraturePoints_element,
                                                                                   int nDOF_trial_element,
                                                                                   int nDOF_test_element,
                                                                                   double* dr,
                                                                                   double* v,
                                                                                   double* w_dV,
                                                                                   double* jacobian_weak_residual)
     void cupdateReaction_strong "updateReaction_strong"(proteus_index_t nElements_global,
                                                         int nQuadraturePoints_element,
                                                         double *r,
                                                         double *strong_residual)
     void cupdateReactionJacobian_strong "updateReactionJacobian_strong"(proteus_index_t nElements_global,
                                                                         int nQuadraturePoints_element,
                                                                         int nDOF_trial_element,
                                                                         double *dr,
                                                                         double *v,
                                                                         double *dstrong_residual)
     void cupdateReaction_adjoint "updateReaction_adjoint"(proteus_index_t nElements_global,
                                                           int nQuadraturePoints_element,
                                                           int nDOF_test_element,
                                                           double *dr,
                                                           double *w_dV,
                                                           double *Lstar_w_dV)
     void cupdateSubgridError "updateSubgridError"(proteus_index_t nElements_global,
                                                   int nQuadraturePoints_element,
                                                   int nDOF_test_element,
                                                   double *error,
                                                   double *Lstar_w_dV,
                                                   double *weak_residual)
     void cupdateSubgridErrorJacobian "updateSubgridErrorJacobian"(proteus_index_t nElements_global,
                                                                   int nQuadraturePoints_element,
                                                                   int nDOF_trial_element,
                                                                   int nDOF_test_element,
                                                                   double *derror,
                                                                   double *Lstar_w_dV,
                                                                   double *jacobian_weak_residual)
     void cupdateNumericalDiffusion "updateNumericalDiffusion"(proteus_index_t nElements_global,
                                                               int nQuadraturePoints_element,
                                                               int nDOF_test_element,
                                                               int nSpace,
                                                               double *numDiff,
                                                               double *grad_u_X_grad_w_dV,
                                                               double *weak_residual)
     void cupdateNumericalDiffusion_lowmem "updateNumericalDiffusion_lowmem"(proteus_index_t nElements_global,
                                                                             int nQuadraturePoints_element,
                                                                             int nDOF_test_element,
                                                                             int nSpace,
                                                                             double* numDiff,
                                                                             double* grad_u,
                                                                             double* grad_w_dV,
                                                                             double* weak_residual)
     void cupdateNumericalDiffusionJacobian "updateNumericalDiffusionJacobian"(proteus_index_t nElements_global,
                                                                               int nQuadraturePoints_element,
                                                                               int nDOF_trial_element,
                                                                               int nDOF_test_element,
                                                                               int nSpace,
                                                                               double *numDiff,
                                                                               double *grad_v_X_grad_w_dV,
                                                                               double *jacobian_weak_residual)
     void cupdateNumericalDiffusionJacobian_lowmem "updateNumericalDiffusionJacobian_lowmem"(proteus_index_t nElements_global,
                                                                                             int nQuadraturePoints_element,
                                                                                             int nDOF_trial_element,
                                                                                             int nDOF_test_element,
                                                                                             int nSpace,
                                                                                             double* numDiff,
                                                                                             double* grad_v,
                                                                                             double* grad_w_dV,
                                                                                             double* jacobian_weak_residual)
     void ccalculateScalarScalarProduct "calculateScalarScalarProduct"(proteus_index_t nElements_global,
                                                                       int nQuadraturePoints_element,
                                                                       double *s1,
                                                                       double *s2,
                                                                       double *sResult)
     void ccalculateVectorScalarProduct "calculateVectorScalarProduct"(proteus_index_t nElements_global,
                                                                       int nQuadraturePoints_element,
                                                                       int nSpace,
                                                                       double *v,
                                                                       double *s,
                                                                       double *vResult)
     void ccalculateTensorScalarProduct "calculateTensorScalarProduct"(proteus_index_t nElements_global,
                                                                       int nQuadraturePoints_element,
                                                                       int nSpace,
                                                                       double *t,
                                                                       double *s,
                                                                       double *tResult)
     void cupdateInteriorElementBoundaryFlux "updateInteriorElementBoundaryFlux"(proteus_index_t nInteriorElementBoundaries_global,
                                                                                 int nElementBoundaries_element,
                                                                                 int nQuadraturePoints_elementBoundary,
                                                                                 int nDOF_test_element,
                                                                                 proteus_index_t *interiorElementBoundaries,
                                                                                 proteus_index_t *elementBoundaryElements,
                                                                                 int *elementBoundaryLocalElementBoundaries,
                                                                                 double *flux,
                                                                                 double *w_dS,
                                                                                 double *residual)
     void cupdateExteriorElementBoundaryFlux "updateExteriorElementBoundaryFlux"(proteus_index_t nExteriorElementBoundaries_global,
                                                                                 int nElementBoundaries_element,
                                                                                 int nQuadraturePoints_elementBoundary,
                                                                                 int nDOF_test_element,
                                                                                 proteus_index_t *exteriorElementBoundaries,
                                                                                 proteus_index_t *elementBoundaryElements,
                                                                                 int *elementBoundaryLocalElementBoundaries,
                                                                                 double *flux,
                                                                                 double *w_dS,
                                                                                 double *residual)
     void cupdateGlobalResidualFromElementResidual "updateGlobalResidualFromElementResidual"(proteus_index_t nElements_global,
                                                                                             int nDOF_test_element,
                                                                                             proteus_index_t offset_r,
                                                                                             proteus_index_t stride_r,
                                                                                             int *nFreeDOF_element_r,
                                                                                             int *freeLocal_r,
                                                                                             proteus_index_t *freeGlobal_r,
                                                                                             double *elementResidual,
                                                                                             double *globalResidual)
     void cupdateGlobalJacobianFromElementJacobian_dense "updateGlobalJacobianFromElementJacobian_dense"(proteus_index_t nElements_global,
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
                                                                                                         double *globalJacobian)
     void cupdateGlobalJacobianFromElementJacobian_eb_dense "updateGlobalJacobianFromElementJacobian_eb_dense"(proteus_index_t *elementNeighbors,
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
                                                                                                               double *globalJacobian)
     void cupdateGlobalJacobianFromInteriorElementBoundaryFluxJacobian_dense "updateGlobalJacobianFromInteriorElementBoundaryFluxJacobian_dense"(proteus_index_t nInteriorElementBoundaries_global,
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
                                                                                                                                                 double *jac)
     void cupdateGlobalJacobianFromInteriorElementBoundaryFluxJacobian_eb_dense "updateGlobalJacobianFromInteriorElementBoundaryFluxJacobian_eb_dense"(proteus_index_t *elementNeighbors,
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
                                                                                                                                                       double *jac)
     void cupdateGlobalJacobianFromExteriorElementBoundaryFluxJacobian_dense "updateGlobalJacobianFromExteriorElementBoundaryFluxJacobian_dense"(proteus_index_t nExteriorElementBoundaries_global,
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
                                                                                                                                                 double *jac)
     void cupdateGlobalJacobianFromExteriorElementBoundaryFluxJacobian_eb_dense "updateGlobalJacobianFromExteriorElementBoundaryFluxJacobian_eb_dense"(proteus_index_t *elementNeighbors,
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
                                                                                                                                                       double *jac)
     void cupdateGlobalJacobianFromElementJacobian_CSR "updateGlobalJacobianFromElementJacobian_CSR"(proteus_index_t nElements_global,
                                                                                                     int nDOF_test_element,
                                                                                                     int nDOF_trial_element,
                                                                                                     int *nFreeDOF_element_r,
                                                                                                     int *freeLocal_r,
                                                                                                     int *nFreeDOF_element_u,
                                                                                                     int *freeLocal_u,
                                                                                                     proteus_csr_index_t *csrRowIndeces_ru,
                                                                                                     proteus_csr_index_t *csrColumnOffsets_ru,
                                                                                                     double *elementJacobian,
                                                                                                     double *globalJacobian)
     void cupdateGlobalJacobianFromElementJacobian_eb_CSR "updateGlobalJacobianFromElementJacobian_eb_CSR"(proteus_index_t *elementNeighbors,
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
                                                                                                           double *globalJacobian)
     void cupdateGlobalJacobianFromInteriorElementBoundaryFluxJacobian_CSR "updateGlobalJacobianFromInteriorElementBoundaryFluxJacobian_CSR"(proteus_index_t nInteriorElementBoundaries_global,
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
                                                                                                                                             double *jac)
     void cupdateGlobalJacobianFromExteriorElementBoundaryFluxJacobian_CSR "updateGlobalJacobianFromExteriorElementBoundaryFluxJacobian_CSR"(proteus_index_t nExteriorElementBoundaries_global,
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
                                                                                                                                             double *jac)
     void cupdateGlobalJacobianFromInteriorElementBoundaryFluxJacobian_eb_CSR "updateGlobalJacobianFromInteriorElementBoundaryFluxJacobian_eb_CSR"(proteus_index_t *elementNeighbors,
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
                                                                                                                                                   double *jac)
     void cupdateGlobalJacobianFromExteriorElementBoundaryFluxJacobian_eb_CSR "updateGlobalJacobianFromExteriorElementBoundaryFluxJacobian_eb_CSR"(proteus_index_t *elementNeighbors,
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
                                                                                                                                                   double *jac)
     void ccalculateWeightedShape "calculateWeightedShape"(proteus_index_t nElements_global,
                                                           int nQuadraturePoints_element,
                                                           int nDOF_test_element,
                                                           double *dVR,
                                                           double *abs_det_jac,
                                                           double *w,
                                                           double *w_dV)
     void ccalculateWeightedShapeGradients "calculateWeightedShapeGradients"(proteus_index_t nElements_global,
                                                                             int nQuadraturePoints_element,
                                                                             int nDOF_test_element,
                                                                             int nSpace,
                                                                             double *dVR,
                                                                             double *abs_det_jac,
                                                                             double *grad_w,
                                                                             double *grad_w_dV)
     void ccalculateShape_X_weightedShape "calculateShape_X_weightedShape"(proteus_index_t nElements_global,
                                                                           int nQuadraturePoints_element,
                                                                           int nDOF_trial_element,
                                                                           int nDOF_test_element,
                                                                           double *v,
                                                                           double *w_dV,
                                                                           double *v_X_w_dV)
     void ccalculateShape_X_weightedGradShape "calculateShape_X_weightedGradShape"(proteus_index_t nElements_global,
                                                                                   int nQuadraturePoints_element,
                                                                                   int nDOF_trial_element,
                                                                                   int nDOF_test_element,
                                                                                   int nSpace,
                                                                                   double *v,
                                                                                   double *grad_w_dV,
                                                                                   double *v_X_grad_w_dV)
     void ccalculateGradShape_X_weightedShape "calculateGradShape_X_weightedShape"(proteus_index_t nElements_global,
                                                                                   int nQuadraturePoints_element,
                                                                                   int nDOF_trial_element,
                                                                                   int nDOF_test_element,
                                                                                   int nSpace,
                                                                                   double *grad_v,
                                                                                   double *w_dV,
                                                                                   double *grad_v_X_w_dV)
     void ccalculateGradShape_X_weightedGradShape "calculateGradShape_X_weightedGradShape"(proteus_index_t nElements_global,
                                                                                           int nQuadraturePoints_element,
                                                                                           int nDOF_trial_element,
                                                                                           int nDOF_test_element,
                                                                                           int nSpace,
                                                                                           double *grad_v,
                                                                                           double *grad_w_dV,
                                                                                           double *grad_v_X_grad_w_dV)
     void ccalculateWeightedShapeTrace "calculateWeightedShapeTrace"(proteus_index_t nElements_global,
                                                                     int nElementBoundaries_element,
                                                                     int nElementBoundaryQuadraturePoints_elementBoundary,
                                                                     int nDOF_test_element,
                                                                     double *dSR,
                                                                     double *sqrt_det_g,
                                                                     double *w,
                                                                     double *w_dS)
     void ccalculateShape_X_weightedShapeTrace "calculateShape_X_weightedShapeTrace"(proteus_index_t nElements_global,
                                                                                     int nElementBoundaries_element,
                                                                                     int nElementBoundaryQuadraturePoints_elementBoundary,
                                                                                     int nDOF_trial_element,
                                                                                     int nDOF_test_element,
                                                                                     double *v,
                                                                                     double *w_dS,
                                                                                     double *v_X_w_dS)
     void ccalculateGradShape_X_weightedShapeTrace "calculateGradShape_X_weightedShapeTrace"(proteus_index_t nElements_global,
                                                                                             int nElementBoundaries_element,
                                                                                             int nElementBoundaryQuadraturePoints_elementBoundary,
                                                                                             int nDOF_trial_element,
                                                                                             int nDOF_test_element,
                                                                                             int nSpace,
                                                                                             double *grad_v,
                                                                                             double *w_dS,
                                                                                             double *grad_v_X_w_dS)
     void ccalculateIntegrationWeights "calculateIntegrationWeights"(proteus_index_t nElements_global,
                                                                     int nQuadraturePoints_element,
                                                                     double *abs_det_J,
                                                                     double *referenceWeights,
                                                                     double *weights)
     void ccalculateElementBoundaryIntegrationWeights "calculateElementBoundaryIntegrationWeights"(proteus_index_t nElements_global,
                                                                                                   int nElementBoundaries_element,
                                                                                                   int nQuadraturePoints_elementBoundary,
                                                                                                   double *sqrt_det_g,
                                                                                                   double *referenceWeights,
                                                                                                   double *weights)
     void ccalculateFiniteElementFunctionValues "calculateFiniteElementFunctionValues"(proteus_index_t nElements_global,
                                                                                       int nQuadraturePoints_element,
                                                                                       int nDOF_trial_element,
                                                                                       int nComponents,
                                                                                       proteus_index_t *l2g,
                                                                                       double *dof,
                                                                                       double *v,
                                                                                       double *u)
     void ccalculateFiniteElementFunctionGradientValues "calculateFiniteElementFunctionGradientValues"(proteus_index_t nElements_global,
                                                                                                       int nQuadraturePoints_element,
                                                                                                       int nDOF_trial_element,
                                                                                                       int nComponents,
                                                                                                       int nSpace,
                                                                                                       proteus_index_t *l2g,
                                                                                                       double *dof,
                                                                                                       double *grad_v,
                                                                                                       double *grad_u)
     void ccalculateFiniteElementFunctionGradientTensorValues "calculateFiniteElementFunctionGradientTensorValues"(proteus_index_t nElements_global,
                                                                                                                   int nQuadraturePoints_element,
                                                                                                                   int nDOF_trial_element,
                                                                                                                   int nDOF_test_element,
                                                                                                                   int nComponents,
                                                                                                                   int nSpace,
                                                                                                                   proteus_index_t *l2g,
                                                                                                                   double *dof,
                                                                                                                   double *grad_v_X_grad_w_dV,
                                                                                                                   double *grad_u_X_grad_w_dV)
     void ccalculateFiniteElementFunctionValuesTrace "calculateFiniteElementFunctionValuesTrace"(proteus_index_t nElements_global,
                                                                                                 int nElementBoundaries_element,
                                                                                                 int nQuadraturePoints_elementBoundary,
                                                                                                 int nDOF_trial_element,
                                                                                                 int nComponents,
                                                                                                 proteus_index_t *l2g,
                                                                                                 double *dof,
                                                                                                 double *v,
                                                                                                 double *u)
     void ccalculateFiniteElementFunctionGradientValuesTrace "calculateFiniteElementFunctionGradientValuesTrace"(proteus_index_t nElements_global,
                                                                                                                 int nElementBoundaries_element,
                                                                                                                 int nQuadraturePoints_elementBoundary,
                                                                                                                 int nDOF_trial_element,
                                                                                                                 int nComponents,
                                                                                                                 int nSpace,
                                                                                                                 proteus_index_t *l2g,
                                                                                                                 double *dof,
                                                                                                                 double *grad_v,
                                                                                                                 double *grad_u)
     void ccalculateFlowVelocity "calculateFlowVelocity"(proteus_index_t nElements_global,
                                                         int nQuadraturePoints_element,
                                                         int nSpace,
                                                         double *f,
                                                         double *a,
                                                         double *grad_phi,
                                                         double *v)
     void cupdateAddJacobian_CSR "updateAddJacobian_CSR"(int jacIndex,
                                                         double val,
                                                         double *jac)
     void czeroJacobian_CSR "zeroJacobian_CSR"(proteus_index_t nNonzeros,
                                               double *jac)
     #void csetInflowFlux "setInflowFlux"(proteus_index_t nExteriorElementBoundaries_global,
     #                                    int nQuadraturePoints_elementBoundary,
     #                                    proteus_index_t *exteriorElementBoundaries,
     #                                    double *inflowFlux,
     #                                    double *flux)
     void ccalculateInteriorElementBoundaryVelocities "calculateInteriorElementBoundaryVelocities"(proteus_index_t nInteriorElementBoundaries_global,
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
                                                                                                   double *mJump)
     void ccalculateExteriorElementBoundaryVelocities "calculateExteriorElementBoundaryVelocities"(proteus_index_t nExteriorElementBoundaries_global,
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
                                                                                                   double *mJump)
     void ccalculateConservationResidualPWL "calculateConservationResidualPWL"(proteus_index_t nElements_global,
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
                                                                               double *vConservative_element)
     void ccalculateConservationJacobianPWL "calculateConservationJacobianPWL"(proteus_index_t nNodes_global,
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
                                                                               double *starJacobian)
     void ccalculateConservationFluxPWL "calculateConservationFluxPWL"(proteus_index_t nNodes_global,
                                                                       proteus_index_t nNodes_internal,
                                                                       int *nElements_node,
                                                                       proteus_index_t *nodeStarOffsets,
                                                                       proteus_index_t *nodeStarJacobianOffsets,
                                                                       proteus_index_t *internalNodes,
                                                                       double *starR,
                                                                       double *starJ,
                                                                       double *starU)
     void csetExteriorGlobalElementBoundaryVelocityValues "setExteriorGlobalElementBoundaryVelocityValues"(int updateFluxValues,
                                                                                                           proteus_index_t nExteriorElementBoundaries_global,
                                                                                                           int nQuadraturePoints_elementBoundary,
                                                                                                           int nSpace,
                                                                                                           proteus_index_t *exteriorElementBoundaries,
                                                                                                           proteus_index_t *elementBoundaryElements,
                                                                                                           int *elementBoundaryLocalElementBoundaries,
                                                                                                           double *n,
                                                                                                           double *vn_in,
                                                                                                           double *v_out)
     void ccalculateDimensionlessNumbersADR "calculateDimensionlessNumbersADR"(proteus_index_t nElements_global,
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
                                                                               double *cfl)
     void ccalculateDimensionlessNumbersADR_sd "calculateDimensionlessNumbersADR_sd"(proteus_index_t nElements_global,
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
                                                                                     double* cfl)
     void ccalculateCFLADR "calculateCFLADR"(proteus_index_t nElements_global,
                                             int nQuadraturePoints_element,
                                             int nSpace,
                                             double *elementDiameter,
                                             double *dm,
                                             double *df,
                                             double *cfl)
     void cupdateInteriorElementBoundaryDiffusiveVelocity "updateInteriorElementBoundaryDiffusiveVelocity"(proteus_index_t nInteriorElementBoundaries_global,
                                                                                                           int nElementBoundaries_element,
                                                                                                           int nQuadraturePoints_elementBoundary,
                                                                                                           int nSpace,
                                                                                                           proteus_index_t *interiorElementBoundaries,
                                                                                                           proteus_index_t *elementBoundaryElements,
                                                                                                           int *elementBoundaryLocalElementBoundaries,
                                                                                                           double *a,
                                                                                                           double *grad_phi,
                                                                                                           double *velocity)
     void cupdateInteriorElementBoundaryDiffusiveVelocity_sd "updateInteriorElementBoundaryDiffusiveVelocity_sd"(proteus_index_t nInteriorElementBoundaries_global,
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
                                                                                                                 double* velocity)
     void cupdateExteriorElementBoundaryDiffusiveVelocity "updateExteriorElementBoundaryDiffusiveVelocity"(proteus_index_t nExteriorElementBoundaries_global,
                                                                                                           int nElementBoundaries_element,
                                                                                                           int nQuadraturePoints_elementBoundary,
                                                                                                           int nSpace,
                                                                                                           proteus_index_t *exteriorElementBoundaries,
                                                                                                           proteus_index_t *elementBoundaryElements,
                                                                                                           int *elementBoundaryLocalElementBoundaries,
                                                                                                           double *a,
                                                                                                           double *grad_phi,
                                                                                                           double *velocity)
     void cupdateExteriorElementBoundaryDiffusiveVelocity_sd "updateExteriorElementBoundaryDiffusiveVelocity_sd"(proteus_index_t nExteriorElementBoundaries_global,
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
                                                                                                                 double* velocity)
     void cupdateInteriorElementBoundaryAdvectiveVelocity "updateInteriorElementBoundaryAdvectiveVelocity"(proteus_index_t nInteriorElementBoundaries_global,
                                                                                                           int nElementBoundaries_element,
                                                                                                           int nQuadraturePoints_elementBoundary,
                                                                                                           int nSpace,
                                                                                                           proteus_index_t *interiorElementBoundaries,
                                                                                                           proteus_index_t *elementBoundaryElements,
                                                                                                           int *elementBoundaryLocalElementBoundaries,
                                                                                                           double *f,
                                                                                                           double *velocity)
     void cupdateExteriorElementBoundaryAdvectiveVelocity "updateExteriorElementBoundaryAdvectiveVelocity"(proteus_index_t nExteriorElementBoundaries_global,
                                                                                                           int nElementBoundaries_element,
                                                                                                           int nQuadraturePoints_elementBoundary,
                                                                                                           int nSpace,
                                                                                                           proteus_index_t *exteriorElementBoundaries,
                                                                                                           proteus_index_t *elementBoundaryElements,
                                                                                                           int *elementBoundaryLocalElementBoundaries,
                                                                                                           double *f,
                                                                                                           double *velocity)
     void cupdateInteriorElementBoundaryShockCapturingVelocity "updateInteriorElementBoundaryShockCapturingVelocity"(proteus_index_t nInteriorElementBoundaries_global,
                                                                                                                     int nElementBoundaries_element,
                                                                                                                     int nQuadraturePoints_elementBoundary,
                                                                                                                     int nQuadraturePoints_element,
                                                                                                                     int nSpace,
                                                                                                                     proteus_index_t *interiorElementBoundaries,
                                                                                                                     proteus_index_t *elementBoundaryElements,
                                                                                                                     int *elementBoundaryLocalElementBoundaries,
                                                                                                                     double *numDiff,
                                                                                                                     double *grad_u,
                                                                                                                     double *velocity)
     void cupdateExteriorElementBoundaryShockCapturingVelocity "updateExteriorElementBoundaryShockCapturingVelocity"(proteus_index_t nExteriorElementBoundaries_global,
                                                                                                                     int nElementBoundaries_element,
                                                                                                                     int nQuadraturePoints_elementBoundary,
                                                                                                                     int nQuadraturePoints_element,
                                                                                                                     int nSpace,
                                                                                                                     proteus_index_t *exteriorElementBoundaries,
                                                                                                                     proteus_index_t *elementBoundaryElements,
                                                                                                                     int *elementBoundaryLocalElementBoundaries,
                                                                                                                     double *numDiff,
                                                                                                                     double *grad_u,
                                                                                                                     double *velocity)
     void ccalculateInteriorElementBoundaryAverageVelocity "calculateInteriorElementBoundaryAverageVelocity"(proteus_index_t nInteriorElementBoundaries_global,
                                                                                                             int nElementBoundaries_element,
                                                                                                             int nQuadraturePoints_elementBoundary,
                                                                                                             int nSpace,
                                                                                                             proteus_index_t *interiorElementBoundaries,
                                                                                                             proteus_index_t *elementBoundaryElements,
                                                                                                             int *elementBoundaryLocalElementBoundaries,
                                                                                                             double *v,
                                                                                                             double *vAverage)
     void ccalculateExteriorElementBoundaryAverageVelocity "calculateExteriorElementBoundaryAverageVelocity"(proteus_index_t nExteriorElementBoundaries_global,
                                                                                                             int nElementBoundaries_element,
                                                                                                             int nQuadraturePoints_elementBoundary,
                                                                                                             int nSpace,
                                                                                                             proteus_index_t *exteriorElementBoundaries,
                                                                                                             proteus_index_t *elementBoundaryElements,
                                                                                                             int *elementBoundaryLocalElementBoundaries,
                                                                                                             double *v,
                                                                                                             double *vAverage)
     void ccalculateConservationResidualDG "calculateConservationResidualDG"(proteus_index_t nElements_global,
                                                                             int nDOF_test_element,
                                                                             double *elementResidual,
                                                                             double *conservationResidual)
     void ccalculateConservationResidual "calculateConservationResidual"(proteus_index_t nElements_global,
                                                                         int nDOF_test_element,
                                                                         int nElementBoundaries_element,
                                                                         int nQuadraturePoints_elementBoundary,
                                                                         int nSpace,
                                                                         double *n,
                                                                         double *dS_u,
                                                                         double *elementResidual,
                                                                         double *velocity,
                                                                         double *conservationResidual)
     void ccalculateConservationResidualGlobalBoundaries "calculateConservationResidualGlobalBoundaries"(proteus_index_t nElements_global,
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
                                                                                                         double *conservationResidual)
     void ccopyGlobalElementBoundaryVelocityToElementBoundary "copyGlobalElementBoundaryVelocityToElementBoundary"(proteus_index_t nElements_global,
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
                                                                                                                   double *velocityBoundary_element)
     void cloadBoundaryFluxIntoGlobalElementBoundaryVelocity "loadBoundaryFluxIntoGlobalElementBoundaryVelocity"(proteus_index_t nExteriorElementBoundaries_global,
                                                                                                                 int nQuadraturePoints_elementBoundary,
                                                                                                                 int nSpace,
                                                                                                                 proteus_index_t *exteriorElementBoundaries,
                                                                                                                 proteus_index_t *fluxElementBoundaries,
                                                                                                                 double *normal,
                                                                                                                 double *flux,
                                                                                                                 double updateCoef,
                                                                                                                 double *velocity)
     void ccalculateInteriorNumericalTrace_Potential "calculateInteriorNumericalTrace_Potential"(proteus_index_t nInteriorElementBoundaries_global,
                                                                                                 int nElementBoundaries_element,
                                                                                                 int nQuadraturePoints_elementBoundary,
                                                                                                 proteus_index_t *interiorElementBoundaries,
                                                                                                 proteus_index_t *elementBoundaryElements,
                                                                                                 int *elementBoundaryLocalElementBoundaries,
                                                                                                 double *phi,
                                                                                                 double *dphi,
                                                                                                 double *phi_trace,
                                                                                                 double *dphi_trace_left,
                                                                                                 double *dphi_trace_right)
     void ccalculateExteriorNumericalTrace_Potential "calculateExteriorNumericalTrace_Potential"(int *isDOFBoundary,
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
                                                                                                 double *dphi_trace_left)
     void cupdateInteriorElementBoundary_MixedForm_weak "updateInteriorElementBoundary_MixedForm_weak"(proteus_index_t nInteriorElementBoundaries_global,
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
                                                                                                       double *b)
     void cupdateInteriorElementBoundary_MixedForm_weakJacobian "updateInteriorElementBoundary_MixedForm_weakJacobian"(proteus_index_t nInteriorElementBoundaries_global,
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
                                                                                                                       double *db_eb)
     void cupdateExteriorElementBoundary_MixedForm_weak "updateExteriorElementBoundary_MixedForm_weak"(proteus_index_t nExteriorElementBoundaries_global,
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
                                                                                                       double *b)
     void cupdateExteriorElementBoundary_MixedForm_weakJacobian "updateExteriorElementBoundary_MixedForm_weakJacobian"(proteus_index_t nExteriorElementBoundaries_global,
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
                                                                                                                       double *db_eb)
     void cupdatePotential_MixedForm_weak "updatePotential_MixedForm_weak"(proteus_index_t nElements_global,
                                                                           int nQuadraturePoints_element,
                                                                           int nDOF_test_element,
                                                                           int nSpace,
                                                                           double *phi,
                                                                           double *grad_w_dV,
                                                                           double *b)
     void cupdatePotential_MixedForm_weakJacobian "updatePotential_MixedForm_weakJacobian"(proteus_index_t nElements_global,
                                                                                           int nQuadraturePoints_element,
                                                                                           int nDOF_test_element,
                                                                                           int nSpace,
                                                                                           double *dphi,
                                                                                           double *v,
                                                                                           double *grad_w_dV,
                                                                                           double *db)
     void ccalculateVelocityQuadrature_MixedForm "calculateVelocityQuadrature_MixedForm"(proteus_index_t nElements_global,
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
                                                                                         double *qV)
     void ccalculateVelocityQuadrature_MixedForm_Jacobian "calculateVelocityQuadrature_MixedForm_Jacobian"(proteus_index_t nElements_global,
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
                                                                                                           double *qDV_eb)
     void ccalculateVelocityProjectionMatrixLDG "calculateVelocityProjectionMatrixLDG"(proteus_index_t nElements_global,
                                                                                       int nQuadraturePoints_element,
                                                                                       int nDOF_element,
                                                                                       double *vXw_dV,
                                                                                       double *A_inv)
     void cupdateDiffusion_MixedForm_weak "updateDiffusion_MixedForm_weak"(proteus_index_t nElements_global,
                                                                           int nQuadraturePoints_element,
                                                                           int nDOF_test_element,
                                                                           int nSpace,
                                                                           double *a,
                                                                           double *qV,
                                                                           double *grad_w_dV,
                                                                           double *weak_residual)
     void cupdateDiffusionJacobian_MixedForm_weak "updateDiffusionJacobian_MixedForm_weak"(proteus_index_t nElements_global,
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
                                                                                           double *jacobian_weak_residual_eb)
     void cestimate_mt "estimate_mt"(proteus_index_t nElements_global,
                                     int nQuadraturePoints_element,
                                     int nDOF_element,
                                     double *v,
                                     double *vXw_dV,
                                     double *elementSpatialResidual,
                                     double *mt)
     void cestimate_mt_lowmem "estimate_mt_lowmem"(proteus_index_t nElements_global,
                                                   int nQuadraturePoints_element,
                                                   int nDOF_element,
                                                   double* v,
                                                   double* w_dV,
                                                   double* elementSpatialResidual,
                                                   double* mt)
     double cscalarDomainIntegral "scalarDomainIntegral"(proteus_index_t nElements_global,
                                                         int nQuadraturePoints_element,
                                                         double* dV,
                                                         double* nValueArray)
     double cscalarHeavisideDomainIntegral "scalarHeavisideDomainIntegral"(proteus_index_t nElements_global,
                                                                           int nQuadraturePoints_element,
                                                                           double* dV,
                                                                           double* nValueArray)
     double cscalarSmoothedHeavisideDomainIntegral "scalarSmoothedHeavisideDomainIntegral"(proteus_index_t nElements_global,
						                                           int nQuadraturePoints_element,
						                                           double epsFact,
						                                           double* elementDiameter,
						                                           double* dV,
						                                           double* nValueArray)
     double cfluxDomainBoundaryIntegral "fluxDomainBoundaryIntegral"(proteus_index_t nExteriorElementBoundaries,
				                                     proteus_index_t nElementBoundaries_owned,
                                                                     int nQuadraturePoints_elementBoundary,
                                                                     int* flag,
                                                                     proteus_index_t* exteriorElementBoundariesArray,
                                                                     double* dS,
                                                                     double* nValueArray)
     double cfluxDomainBoundaryIntegralFromVector "fluxDomainBoundaryIntegralFromVector"(proteus_index_t nExteriorElementBoundaries,
						  proteus_index_t nElementBoundaries_owned,
						  int nQuadraturePoints_elementBoundary,
						  int nSpace,
						  int* flag,
						  proteus_index_t* exteriorElementBoundaries,
						  double* dS,
						  double* nValueArray,
						  double* normal)
     void ccopyExteriorElementBoundaryValuesFromElementBoundaryValues "copyExteriorElementBoundaryValuesFromElementBoundaryValues"(proteus_index_t nExteriorElementBoundaries_global,
								                                                                   proteus_index_t nElements_global,
								                                                                   int nElementBoundaries_element,
								                                                                   int nQuadraturePoints_elementBoundary,
								                                                                   int nValuesPerQuadraturePoint,
								                                                                   proteus_index_t * exteriorElementBoundaries,
								                                                                   proteus_index_t* elementBoundaryElements,
								                                                                   int * elementBoundaryLocalElementBoundaries,
								                                                                   double * ebq_val,
								                                                                   double * ebqe_val)
     void ccopyExteriorElementBoundaryValuesToElementBoundaryValues "copyExteriorElementBoundaryValuesToElementBoundaryValues"(proteus_index_t nExteriorElementBoundaries_global,
							                                                                       proteus_index_t nElements_global,
							                                                                       int nElementBoundaries_element,
							                                                                       int nQuadraturePoints_elementBoundary,
							                                                                       int nValuesPerQuadraturePoint,
							                                                                       proteus_index_t * exteriorElementBoundaries,
							                                                                       proteus_index_t* elementBoundaryElements,
							                                                                       int * elementBoundaryLocalElementBoundaries,
							                                                                       double * ebqe_val,
							                                                                       double * ebq_val)
     void ccopyExteriorElementBoundaryValuesToGlobalElementBoundaryValues "copyExteriorElementBoundaryValuesToGlobalElementBoundaryValues"(proteus_index_t nExteriorElementBoundaries_global,
								                                                                           int nQuadraturePoints_elementBoundary,
								                                                                           int nValuesPerQuadraturePoint,
								                                                                           proteus_index_t * exteriorElementBoundaries,
								                                                                           proteus_index_t* elementBoundaryElements,
								                                                                           int * elementBoundaryLocalElementBoundaries,
								                                                                           double * ebqe_val,
								                                                                           double * ebq_global_val)
     void ccopyExteriorElementBoundaryValuesFromGlobalElementBoundaryValues "copyExteriorElementBoundaryValuesFromGlobalElementBoundaryValues"(proteus_index_t nExteriorElementBoundaries_global,
								                                                                               int nQuadraturePoints_elementBoundary,
								                                                                               int nValuesPerQuadraturePoint,
								                                                                               proteus_index_t * exteriorElementBoundaries,
								                                                                               proteus_index_t* elementBoundaryElements,
								                                                                               int * elementBoundaryLocalElementBoundaries,
								                                                                               double * ebq_global_val,
								                                                                               double * ebqe_val)
     void ccomputeC0P1InterpolantDGP0 "computeC0P1InterpolantDGP0"(proteus_index_t nElements_global,
				                                   proteus_index_t nNodes_global,
				                                   int nNodes_element,
				                                   int nDOF_element,
				                                   int dim_dof,
				                                   proteus_index_t* elementNodesArray,
				                                   proteus_index_t* nodeElementOffsets,
				                                   proteus_index_t* nodeElementsArray,
				                                   proteus_index_t* l2g,
				                                   double * dof,
				                                   double* nodalAverage)
     void ccomputeC0P1InterpolantNCP1 "computeC0P1InterpolantNCP1"(proteus_index_t nElements_global,
				                                   proteus_index_t nNodes_global,
				                                   int nNodes_element,
				                                   int nDOF_element,
				                                   int dim_dof,
				                                   proteus_index_t* elementNodesArray,
				                                   proteus_index_t* nodeElementOffsets,
				                                   proteus_index_t* nodeElementsArray,
				                                   proteus_index_t* l2g,
				                                   double * dof,
				                                   double* nodalAverage)
     void ccomputeC0P1InterpolantDGP12 "computeC0P1InterpolantDGP12"(proteus_index_t nElements_global,
				                                     proteus_index_t nNodes_global,
				                                     int nNodes_element,
				                                     int nDOF_element,
				                                     int dim_dof,
				                                     proteus_index_t* elementNodesArray,
				                                     proteus_index_t* nodeElementOffsets,
				                                     proteus_index_t* nodeElementsArray,
				                                     proteus_index_t* l2g,
				                                     double * dof,
				                                     double* nodalAverage)
     void cparametricFiniteElementSpace_getValuesGlobalExteriorTrace "parametricFiniteElementSpace_getValuesGlobalExteriorTrace"(int nElementBoundaries_element,
							                                                                         int nElementBoundaryQuadraturePoints_elementBoundary,
							                                                                         int nDOF_element,
							                                                                         proteus_index_t nExteriorElementBoundaries_global,
							                                                                         proteus_index_t* exteriorElementBoundariesArray,
							                                                                         proteus_index_t* elementBoundaryElementsArray,
							                                                                         int* elementBoundaryLocalElementBoundariesArray,
							                                                                         double* psi,
							                                                                         double* vArray)
     void cparametricFiniteElementSpace_getGradientValuesGlobalExteriorTrace "parametricFiniteElementSpace_getGradientValuesGlobalExteriorTrace"(int nElementBoundaries_element,
								                                                                                 int nElementBoundaryQuadraturePoints_elementBoundary,
								                                                                                 int nDOF_element,
								                                                                                 int nSpace_global,
								                                                                                 proteus_index_t nExteriorElementBoundaries_global,
								                                                                                 proteus_index_t *exteriorElementBoundariesArray,
								                                                                                 proteus_index_t *elementBoundaryElementsArray,
								                                                                                 int *elementBoundaryLocalElementBoundariesArray,
								                                                                                 double* grad_psi,
								                                                                                 double* inverseJacobianArray,
								                                                                                 double* grad_vArray)
     void cparametricMaps_getValuesGlobalExteriorTrace "parametricMaps_getValuesGlobalExteriorTrace"(int nQuadraturePoints_elementBoundary,
						                                                     int nDOF_element,
						                                                     int nSpace_global,
						                                                     proteus_index_t nExteriorElementBoundaries_global,
						                                                     proteus_index_t* exteriorElementBoundariesArray,
						                                                     proteus_index_t* elementBoundaryElementsArray,
						                                                     int* elementBoundaryLocalElementBoundariesArray,
						                                                     double* psi,
						                                                     proteus_index_t* l2g,
						                                                     double* nodeArray,
						                                                     double* xArray)
     void cparametricMaps_getInverseValuesGlobalExteriorTrace "parametricMaps_getInverseValuesGlobalExteriorTrace"(int nElementBoundaryQuadraturePoints_elementBoundary,
							                                                           int nDOF_element,
							                                                           int nSpace_global,
							                                                           proteus_index_t nExteriorElementBoundaries_global,
							                                                           proteus_index_t* exteriorElementBoundariesArray,
							                                                           proteus_index_t* elementBoundaryElementsArray,
							                                                           int* elementBoundaryLocalElementBoundariesArray,
							                                                           double* inverseJacobian,
							                                                           proteus_index_t* l2g,
							                                                           double* nodeArray,
							                                                           double* xArray,
							                                                           double* xiArray)
     void cparametricMaps_getJacobianValuesGlobalExteriorTrace1D "parametricMaps_getJacobianValuesGlobalExteriorTrace1D"(int nQuadraturePoints_element,
							                                                                 int nDOF_element,
							                                                                 proteus_index_t nExteriorElementBoundaries_global,
							                                                                 proteus_index_t * exteriorElementBoundariesArray,
							                                                                 proteus_index_t * elementBoundaryElementsArray,
							                                                                 int * elementBoundaryLocalElementBoundariesArray,
							                                                                 double* grad_psi,
							                                                                 double* boundaryNormals,
							                                                                 double* boundaryJacobians,
							                                                                 proteus_index_t* l2g,
							                                                                 double* nodeArray,
							                                                                 double* jacobianInverseArray,
							                                                                 double* metricTensorArray,
							                                                                 double* metricTensorDeterminantSqrtArray,
							                                                                 double* unitNormalArray)
     void cparametricMaps_getJacobianValuesGlobalExteriorTrace2D "parametricMaps_getJacobianValuesGlobalExteriorTrace2D"(int nQuadraturePoints_element,
							                                                                 int nDOF_element,
							                                                                 proteus_index_t nExteriorElementBoundaries_global,
							                                                                 proteus_index_t * exteriorElementBoundariesArray,
							                                                                 proteus_index_t * elementBoundaryElementsArray,
							                                                                 int * elementBoundaryLocalElementBoundariesArray,
							                                                                 double* grad_psi,
							                                                                 double* boundaryNormals,
							                                                                 double* boundaryJacobians,
							                                                                 proteus_index_t* l2g,
							                                                                 double* nodeArray,
							                                                                 double* jacobianInverseArray,
							                                                                 double* metricTensorArray,
							                                                                 double* metricTensorDeterminantSqrtArray,
							                                                                 double* unitNormalArray)
     void cparametricMaps_getJacobianValuesGlobalExteriorTrace2D_movingDomain "parametricMaps_getJacobianValuesGlobalExteriorTrace2D_movingDomain"(int nQuadraturePoints_element,
                                                                                                                                                   int nDOF_element,
                                                                                                                                                   proteus_index_t nExteriorElementBoundaries_global,
								                                                                                   proteus_index_t* exteriorElementBoundariesArray,
								                                                                                   proteus_index_t* elementBoundaryElementsArray,
								                                                                                   int* elementBoundaryLocalElementBoundariesArray,
								                                                                                   double* xtArray,
								                                                                                   double* grad_psi,
								                                                                                   double* boundaryNormals,
								                                                                                   double* boundaryJacobians,
								                                                                                   proteus_index_t* l2g,
								                                                                                   double* nodeArray,
								                                                                                   double* jacobianInverseArray,
								                                                                                   double* metricTensorArray,
								                                                                                   double* metricTensorDeterminantSqrtArray,
								                                                                                   double* unitNormalArray)
     void cparametricMaps_getJacobianValuesGlobalExteriorTrace3D "parametricMaps_getJacobianValuesGlobalExteriorTrace3D"(int nQuadraturePoints_element,
							                                                                 int nDOF_element,
							                                                                 proteus_index_t nExteriorElementBoundaries_global,
							                                                                 proteus_index_t * exteriorElementBoundariesArray,
							                                                                 proteus_index_t * elementBoundaryElementsArray,
							                                                                 int * elementBoundaryLocalElementBoundariesArray,
							                                                                 double* grad_psi,
							                                                                 double* boundaryNormals,
							                                                                 double* boundaryJacobians,
							                                                                 proteus_index_t* l2g,
							                                                                 double* nodeArray,
							                                                                 double* jacobianInverseArray,
							                                                                 double* metricTensorArray,
							                                                                 double* metricTensorDeterminantSqrtArray,
							                                                                 double* unitNormalArray)
     void cupdateGlobalExteriorElementBoundaryFlux "updateGlobalExteriorElementBoundaryFlux"(proteus_index_t nExteriorElementBoundaries_global,
					                                                     int nQuadraturePoints_elementBoundary,
					                                                     int nDOF_test_element,
					                                                     proteus_index_t* exteriorElementBoundaries,
					                                                     proteus_index_t* elementBoundaryElements,
					                                                     int* elementBoundaryLocalElementBoundaries,
					                                                     double* flux,
					                                                     double* w_dS,
					                                                     double* residual)
     void cupdateGlobalJacobianFromGlobalExteriorElementBoundaryFluxJacobian_eb_dense "updateGlobalJacobianFromGlobalExteriorElementBoundaryFluxJacobian_eb_dense"(proteus_index_t* elementNeighbors,
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
										                                                                                   double* jac)
     void cupdateGlobalJacobianFromGlobalExteriorElementBoundaryFluxJacobian_dense "updateGlobalJacobianFromGlobalExteriorElementBoundaryFluxJacobian_dense"(proteus_index_t nExteriorElementBoundaries_global,
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
									                                                                                     double* jac)
     void cupdateGlobalJacobianFromGlobalExteriorElementBoundaryFluxJacobian_CSR "updateGlobalJacobianFromGlobalExteriorElementBoundaryFluxJacobian_CSR"(proteus_index_t nExteriorElementBoundaries_global,
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
									                                                                                 double* jac)
     void cupdateGlobalJacobianFromGlobalExteriorElementBoundaryFluxJacobian_eb_CSR "updateGlobalJacobianFromGlobalExteriorElementBoundaryFluxJacobian_eb_CSR"(proteus_index_t* elementNeighbors,
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
									                                                                                       double* jac)
     void ccalculateWeightedShapeGlobalExteriorTrace "calculateWeightedShapeGlobalExteriorTrace"(int nElementBoundaryQuadraturePoints_elementBoundary,
					                                                         int nDOF_test_element,
					                                                         proteus_index_t nExteriorElementBoundaries_global,
					                                                         proteus_index_t* exteriorElementBoundariesArray,
					                                                         proteus_index_t* elementBoundaryElementsArray,
					                                                         int* elementBoundaryLocalElementBoundariesArray,
					                                                         double* dSR,
					                                                         double* sqrt_det_g,
					                                                         double* w,
					                                                         double* w_dS)
     void ccalculateShape_X_weightedShapeGlobalExteriorTrace "calculateShape_X_weightedShapeGlobalExteriorTrace"(int nElementBoundaryQuadraturePoints_elementBoundary,
						                                                                 int nDOF_trial_element,
						                                                                 int nDOF_test_element,
						                                                                 proteus_index_t nExteriorElementBoundaries_global,
						                                                                 proteus_index_t* exteriorElementBoundariesArray,
						                                                                 proteus_index_t* elementBoundaryElementsArray,
						                                                                 int* elementBoundaryLocalElementBoundariesArray,
						                                                                 double* v,
						                                                                 double* w_dS,
						                                                                 double* v_X_w_dS)
     void ccalculateGradShape_X_weightedShapeGlobalExteriorTrace "calculateGradShape_X_weightedShapeGlobalExteriorTrace"(int nElementBoundaryQuadraturePoints_elementBoundary,
							                                                                 int nDOF_trial_element,
							                                                                 int nDOF_test_element,
							                                                                 int nSpace,
							                                                                 proteus_index_t nExteriorElementBoundaries_global,
							                                                                 proteus_index_t* exteriorElementBoundariesArray,
							                                                                 proteus_index_t* elementBoundaryElementsArray,
							                                                                 int* elementBoundaryLocalElementBoundariesArray,
							                                                                 double* grad_v,
							                                                                 double* w_dS,
							                                                                 double* grad_v_X_w_dS)
     void ccalculateGlobalExteriorElementBoundaryIntegrationWeights "calculateGlobalExteriorElementBoundaryIntegrationWeights"(int nQuadraturePoints_elementBoundary,
							                                                                       proteus_index_t nExteriorElementBoundaries_global,
							                                                                       double* sqrt_det_g,
							                                                                       double* referenceWeights,
							                                                                       double* weights)
     void ccalculateFiniteElementFunctionValuesGlobalExteriorTrace "calculateFiniteElementFunctionValuesGlobalExteriorTrace"(int nQuadraturePoints_elementBoundary,
							                                                                     int nDOF_trial_element,
							                                                                     int nComponents,
							                                                                     proteus_index_t nExteriorElementBoundaries_global,
							                                                                     proteus_index_t * exteriorElementBoundariesArray,
							                                                                     proteus_index_t * elementBoundaryElementsArray,
							                                                                     int * elementBoundaryLocalElementBoundariesArray,
							                                                                     proteus_index_t* l2g,
							                                                                     double* dof,
							                                                                     double* v,
							                                                                     double* u)
     void ccalculateFiniteElementFunctionGradientValuesGlobalExteriorTrace "calculateFiniteElementFunctionGradientValuesGlobalExteriorTrace"(int nQuadraturePoints_elementBoundary,
								                                                                             int nDOF_trial_element,
								                                                                             int nComponents,
								                                                                             int nSpace,
								                                                                             proteus_index_t nExteriorElementBoundaries_global,
								                                                                             proteus_index_t * exteriorElementBoundariesArray,
								                                                                             proteus_index_t * elementBoundaryElementsArray,
								                                                                             int * elementBoundaryLocalElementBoundariesArray,
								                                                                             proteus_index_t* l2g,
								                                                                             double* dof,
								                                                                             double* grad_v,
								                                                                             double* grad_u)
     void cupdateGlobalExteriorElementBoundaryDiffusiveVelocity "updateGlobalExteriorElementBoundaryDiffusiveVelocity"(proteus_index_t nExteriorElementBoundaries_global,
							                                                               int nQuadraturePoints_elementBoundary,
							                                                               int nSpace,
							                                                               proteus_index_t* exteriorElementBoundaries,
							                                                               proteus_index_t* elementBoundaryElements,
							                                                               int* elementBoundaryLocalElementBoundaries,
							                                                               double* a,
							                                                               double* grad_phi,
							                                                               double* velocity)
     void cupdateGlobalExteriorElementBoundaryAdvectiveVelocity "updateGlobalExteriorElementBoundaryAdvectiveVelocity"(proteus_index_t nExteriorElementBoundaries_global,
							                                                               int nQuadraturePoints_elementBoundary,
							                                                               int nSpace,
							                                                               proteus_index_t* exteriorElementBoundaries,
							                                                               proteus_index_t* elementBoundaryElements,
							                                                               int* elementBoundaryLocalElementBoundaries,
							                                                               double* f,
							                                                               double* velocity)
     void cupdateGlobalExteriorElementBoundaryShockCapturingVelocity "updateGlobalExteriorElementBoundaryShockCapturingVelocity"(proteus_index_t nExteriorElementBoundaries_global,
							                                                                         int nQuadraturePoints_elementBoundary,
							                                                                         int nSpace,
							                                                                         proteus_index_t* exteriorElementBoundaries,
							                                                                         proteus_index_t* elementBoundaryElements,
							                                                                         int* elementBoundaryLocalElementBoundaries,
							                                                                         double* numDiff,
							                                                                         double* grad_u,
							                                                                         double* velocity)
     void ccopyFreeUnknownsToGlobalUnknowns "copyFreeUnknownsToGlobalUnknowns"(proteus_index_t nDOF2set,
				                                               proteus_index_t offset,
				                                               proteus_index_t stride,
				                                               proteus_index_t* globalDOFids,
				                                               proteus_index_t* freeDOFids,
				                                               double * free_u,
				                                               double * u)
     void ccopyGlobalUnknownsToFreeUnknowns "copyGlobalUnknownsToFreeUnknowns"(proteus_index_t nDOF2set,
				                                               proteus_index_t offset,
				                                               proteus_index_t stride,
				                                               proteus_index_t* globalDOFids,
				                                               proteus_index_t* freeDOFids,
				                                               double * u,
				                                               double * free_u)
     void cupdateInteriorElementBoundaryDiffusionAdjoint "updateInteriorElementBoundaryDiffusionAdjoint"(proteus_index_t nInteriorElementBoundaries_global,
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
						                                                         double* residual)
     void cupdateExteriorElementBoundaryDiffusionAdjoint "updateExteriorElementBoundaryDiffusionAdjoint"(proteus_index_t nExteriorElementBoundaries_global,
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
						                                                         double* residual)
     void cupdateGlobalJacobianFromInteriorElementBoundaryDiffusionAdjoint_dense "updateGlobalJacobianFromInteriorElementBoundaryDiffusionAdjoint_dense"(proteus_index_t nInteriorElementBoundaries_global,
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
									                                                                                 double* jac)
     void cupdateGlobalJacobianFromExteriorElementBoundaryDiffusionAdjoint_dense "updateGlobalJacobianFromExteriorElementBoundaryDiffusionAdjoint_dense"(proteus_index_t nExteriorElementBoundaries_global,
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
									                                                                                 double* jac)
     void cupdateInteriorElementBoundaryDiffusionAdjoint_sd "updateInteriorElementBoundaryDiffusionAdjoint_sd"(proteus_index_t nInteriorElementBoundaries_global,
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
						                                                               double* residual)
     void cupdateExteriorElementBoundaryDiffusionAdjoint_sd "updateExteriorElementBoundaryDiffusionAdjoint_sd"(proteus_index_t nExteriorElementBoundaries_global,
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
						                                                               double* residual)
     void cupdateGlobalJacobianFromInteriorElementBoundaryDiffusionAdjoint_dense_sd "updateGlobalJacobianFromInteriorElementBoundaryDiffusionAdjoint_dense_sd"(proteus_index_t nInteriorElementBoundaries_global,
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
									                                                                                       double* jac)
     void cupdateGlobalJacobianFromExteriorElementBoundaryDiffusionAdjoint_dense_sd "updateGlobalJacobianFromExteriorElementBoundaryDiffusionAdjoint_dense_sd"(proteus_index_t nExteriorElementBoundaries_global,
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
									                                                                                       double* jac)
     void cupdateGlobalJacobianFromInteriorElementBoundaryDiffusionAdjoint_CSR_sd "updateGlobalJacobianFromInteriorElementBoundaryDiffusionAdjoint_CSR_sd"(proteus_index_t nInteriorElementBoundaries_global,
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
									                                                                                   double* jac)
     void cupdateGlobalJacobianFromExteriorElementBoundaryDiffusionAdjoint_CSR_sd "updateGlobalJacobianFromExteriorElementBoundaryDiffusionAdjoint_CSR_sd"(proteus_index_t nExteriorElementBoundaries_global,
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
									                                                                                   double* jac)
     void cupdate_f_movingDomain_q "update_f_movingDomain_q"(proteus_index_t nElements_global,
			                                     int nQuadraturePoints_element,
			                                     int nSpace,
			                                     double* xt,
			                                     double* m,
			                                     double* f)
     void cupdate_f_movingDomain_constantMass_q "update_f_movingDomain_constantMass_q"(proteus_index_t nElements_global,
					                                               int nQuadraturePoints_element,
					                                               int nSpace,
					                                               double* xt,
					                                               double* f)
     void cupdate_f_movingDomain_ebq "update_f_movingDomain_ebq"(proteus_index_t nElements_global,
			                                         int nElementBoundaries_element,
			                                         int nQuadraturePoints_elementBoundary,
			                                         int nSpace,
			                                         double* xt,
			                                         double* m,
			                                         double* f)
     void cupdate_f_movingDomain_constantMass_ebq "update_f_movingDomain_constantMass_ebq"(proteus_index_t nElements_global,
					                                                   int nElementBoundaries_element,
					                                                   int nQuadraturePoints_elementBoundary,
					                                                   int nSpace,
					                                                   double* xt,
					                                                   double* f)
     void cupdateStress_weak "updateStress_weak"(proteus_index_t nElements_global,
		                                 int nQuadraturePoints_element,
		                                 int nDOF_test_element,
		                                 int nSpace,
		                                 double* sigma,
		                                 double* grad_w_dV,
		                                 double* weak_residual_x,
		                                 double* weak_residual_y,
		                                 double* weak_residual_z)
     void cupdateStressJacobian_weak "updateStressJacobian_weak"(proteus_index_t nElements_global,
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
			                                         double* jacobian_weak_residual_zz)
     void cprojectFromNodalInterpolationConditions "projectFromNodalInterpolationConditions"(proteus_index_t nElements_global,
					                                                     int nDOF_element,
					                                                     int dim_dof,
					                                                     const proteus_index_t * l2g,
					                                                     const int * functional_map_element,
					                                                     const double * interpolationValues,
					                                                     double * dofs)
     void cccalculateVelocityQuadrature_MixedForm2_sd "calculateVelocityQuadrature_MixedForm2_sd"(proteus_index_t nElements_global,
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
                                               double* qV)
     void cupdateDiffusion_MixedForm_weak_sd "updateDiffusion_MixedForm_weak_sd"(proteus_index_t nElements_global,
                                       int nQuadraturePoints_element,
                                       int nDOF_test_element,
                                       int nSpace,
                                       int rho_split,
                                       proteus_csr_index_t* rowptr,
                                       proteus_csr_index_t* colind,
                                       double* a,
                                       double* qV,
                                       double* grad_w_dV,
                                       double* velocity,
                                       double* weak_residual)
     void ccalculateVelocityQuadrature_MixedForm2_Jacobian_sd "calculateVelocityQuadrature_MixedForm2_Jacobian_sd"(proteus_index_t nElements_global,
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
                        double* qDV_eb)
     void cupdateDiffusionJacobian_MixedForm_weak_sd "updateDiffusionJacobian_MixedForm_weak_sd"(proteus_index_t nElements_global,
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
                                               double* jacobian_weak_residual_eb)
def parametricFiniteElementSpace_getHessianValues(np.ndarray Hessian_psi,
						  np.ndarray inverseJacobianArray,
						  np.ndarray Hessian_vArray):
    cdef int nElements_global = Hessian_vArray.shape[0]
    cdef int nQuadraturePoints_element = Hessian_vArray.shape[1]
    cdef int nDOF_element = Hessian_vArray.shape[2]
    cdef int nSpace_global = Hessian_vArray.shape[3]
    cparametricFiniteElementSpace_getHessianValues(nElements_global,
     						   nQuadraturePoints_element,
						   nDOF_element,
						   nSpace_global,
						   <double*> Hessian_psi.data,
						   <double*> inverseJacobianArray.data,
						   <double*> Hessian_vArray.data)
def updateDiffusion2_strong(np.ndarray a,
			    np.ndarray Hess_phi,
			    np.ndarray strong_residual):
    cdef int nElements_global = Hess_phi.shape[0]
    cdef int nQuadraturePoints_element = Hess_phi.shape[1]
    cdef int nSpace = Hess_phi.shape[2]
    cupdateDiffusion2_strong(nElements_global,
                             nQuadraturePoints_element,
                             nSpace,
                             <double*> a.data,
			     <double*> Hess_phi.data,
			     <double*> strong_residual.data)
def updateDiffusion2_strong_sd(np.ndarray rowptr,
			       np.ndarray colind,
			       np.ndarray a,
			       np.ndarray Hess_phi,
			       np.ndarray strong_residual):
    cdef int nElements_global = Hess_phi.shape[0]
    cdef int nQuadraturePoints_element = Hess_phi.shape[1]
    cdef int nSpace = Hess_phi.shape[2]
    cupdateDiffusion2_strong_sd(nElements_global,
				nQuadraturePoints_element,
				nSpace,
				<proteus_csr_index_t*> rowptr.data,
				<proteus_csr_index_t*> colind.data,
				<double*> a.data,
				<double*> Hess_phi.data,
				<double*> strong_residual.data)
def updateDiffusionJacobian2_strong(np.ndarray l2g,
				    np.ndarray a,
				    np.ndarray da,
				    np.ndarray v,
				    np.ndarray Hess_phi,
				    np.ndarray dphi,
				    np.ndarray Hess_v,
				    np.ndarray dstrong_residual):
    cdef int nElements_global = Hess_v.shape[0]
    cdef int nQuadraturePoints_element = Hess_v.shape[1]
    cdef int nDOF_trial_element = Hess_v.shape[2]
    cdef int nSpace = Hess_v.shape[3]
    cupdateDiffusionJacobian2_strong(nElements_global,
                                     nQuadraturePoints_element,
                                     nDOF_trial_element,
                                     nSpace,
                                     <proteus_index_t*> l2g.data,
				     <double*> a.data,
				     <double*> da.data,
				     <double*> v.data,
				     <double*> Hess_phi.data,
				     <double*> dphi.data,
				     <double*> Hess_v.data,
				     <double*> dstrong_residual.data)
def updateDiffusionJacobian2_strong_sd(np.ndarray rowptr,
                                       np.ndarray colind,
                                       np.ndarray l2g,
                                       np.ndarray a,
                                       np.ndarray da,
                                       np.ndarray v,
                                       np.ndarray Hess_phi,
                                       np.ndarray dphi,
                                       np.ndarray Hess_v,
                                       np.ndarray dstrong_residual):
    cdef int nElements_global = Hess_v.shape[0]
    cdef int nQuadraturePoints_element = Hess_v.shape[1]
    cdef int nDOF_trial_element = Hess_v.shape[2]
    cdef int nSpace = Hess_v.shape[3]
    cupdateDiffusionJacobian2_strong_sd(nElements_global,
                                        nQuadraturePoints_element,
                                        nDOF_trial_element,
                                        nSpace,
                                        <proteus_csr_index_t*> rowptr.data,
                                        <proteus_csr_index_t*> colind.data,
                                        <proteus_index_t*> l2g.data,
                                        <double*> a.data,
                                        <double*> da.data,
                                        <double*> v.data,
                                        <double*> Hess_phi.data,
                                        <double*> dphi.data,
                                        <double*> Hess_v.data,
                                        <double*> dstrong_residual.data)
def updateDiffusion2_adjoint(np.ndarray a,
			     np.ndarray Hess_w_dV,
			     np.ndarray Lstar_w_dV):
    cdef int nElements_global = Hess_w_dV.shape[0]
    cdef int nQuadraturePoints_element = Hess_w_dV.shape[1]
    cdef int nDOF_test_element  = Hess_w_dV.shape[2]
    cdef int nSpace = Hess_w_dV.shape[3]
    cupdateDiffusion2_adjoint(nElements_global,
                              nQuadraturePoints_element,
                              nDOF_test_element,
                              nSpace,
                              <double*> a.data,
			      <double*> Hess_w_dV.data,
			      <double*> Lstar_w_dV.data)
def updateDiffusion2_adjoint_sd(np.ndarray rowptr,
                                np.ndarray colind,
                                np.ndarray a,
                                np.ndarray Hess_w_dV,
                                np.ndarray Lstar_w_dV):
    cdef int nElements_global = Hess_w_dV.shape[0]
    cdef int nQuadraturePoints_element = Hess_w_dV.shape[1]
    cdef int nDOF_test_element = Hess_w_dV.shape[2]
    cdef int nSpace = Hess_w_dV.shape[3]
    cupdateDiffusion2_adjoint_sd(nElements_global,
                                 nQuadraturePoints_element,
                                 nDOF_test_element,
                                 nSpace,
                                 <proteus_csr_index_t*> rowptr.data,
                                 <proteus_csr_index_t*> colind.data,
                                 <double*> a.data,
                                 <double*> Hess_w_dV.data,
                                 <double*> Lstar_w_dV.data)
def calculateWeightedShapeHessians(np.ndarray dVR,
				   np.ndarray abs_det_jac,
				   np.ndarray Hess_w,
				   np.ndarray Hess_w_dV):
    cdef int nElements_global = Hess_w_dV.shape[0]
    cdef int nQuadraturePoints_element = Hess_w_dV.shape[1]
    cdef int nDOF_test_element = Hess_w_dV.shape[2]
    cdef int nSpace = Hess_w_dV.shape[3]
    ccalculateWeightedShapeHessians(nElements_global,
				    nQuadraturePoints_element,
				    nDOF_test_element,
				    nSpace,
				    <double*> dVR.data,
				    <double*> abs_det_jac.data,
				    <double*> Hess_w.data,
				    <double*> Hess_w_dV.data)
def calculateFiniteElementFunctionHessianValues(np.ndarray l2g,
						np.ndarray dof,
						np.ndarray Hessian_v,
						np.ndarray Hessian_u):
    cdef int nElements_global = Hessian_v.shape[0]
    cdef int nQuadraturePoints_element = Hessian_v.shape[1]
    cdef int nDOF_trial_element = Hessian_v.shape[2]
    cdef int nComponents = 1
    cdef int nSpace = 1
    cdef int nd = Hessian_u.ndim
    if nd == 5:
        nComponents = Hessian_u.shape[2]
        nSpace = Hessian_u.shape[3]
    else:
        nComponents = 1
        nSpace = Hessian_u.shape[2]
    ccalculateFiniteElementFunctionHessianValues(nElements_global,
						 nQuadraturePoints_element,
						 nDOF_trial_element,
						 nComponents,
						 nSpace,
						 <proteus_index_t*> l2g.data,
						 <double*> dof.data,
						 <double*> Hessian_v.data,
						 <double*> Hessian_u.data)
def updateGlobalJacobianFromInteriorElementBoundaryFluxJacobian_2sided_CSR(np.ndarray interiorElementBoundaries,
									   np.ndarray elementBoundaryElements,
									   np.ndarray elementBoundaryLocalElementBoundaries,
									   np.ndarray nFreeDOF_element_r,
									   np.ndarray freeLocal_r,
									   np.ndarray nFreeDOF_element_u,
									   np.ndarray freeLocal_u,
									   np.ndarray csrRowIndeces_ru,
									   np.ndarray csrColumnOffsets_eb_ru,
									   np.ndarray elementBoundaryFluxJacobian_2sided,
									   np.ndarray w_dS,
									   jac):
    cdef np.ndarray rowptr, colind, jac_array
    (rowptr,colind,jac_array) = jac.getCSRrepresentation()
    cdef int nInteriorElementBoundaries_global = interiorElementBoundaries.shape[0]
    cdef int nElementBoundaries_element = w_dS.shape[1]
    cdef int nQuadraturePoints_elementBoundary = w_dS.shape[2]
    cdef int nDOF_test_element = w_dS.shape[3]
    cdef int nDOF_trial_element = elementBoundaryFluxJacobian_2sided.shape[4]
    cupdateGlobalJacobianFromInteriorElementBoundaryFluxJacobian_2sided_CSR(nInteriorElementBoundaries_global,
									    nElementBoundaries_element,
									    nQuadraturePoints_elementBoundary,
									    nDOF_test_element,
									    nDOF_trial_element,
								            <proteus_index_t*> interiorElementBoundaries.data,
									    <proteus_index_t*> elementBoundaryElements.data,
									    <int*> elementBoundaryLocalElementBoundaries.data,
									    <int*> nFreeDOF_element_r.data,
									    <int*> freeLocal_r.data,
									    <int*> nFreeDOF_element_u.data,
									    <int*> freeLocal_u.data,
									    <proteus_csr_index_t*> csrRowIndeces_ru.data,
									    <proteus_csr_index_t*> csrColumnOffsets_eb_ru.data,
									    <double*> elementBoundaryFluxJacobian_2sided.data,
									    <double*> w_dS.data,
									    <double*> jac_array.data)
def updateGlobalJacobianFromInteriorElementBoundaryFluxJacobian_2sided_dense(proteus_index_t offset_r,
									     proteus_index_t stride_r,
									     proteus_index_t offset_u,
									     proteus_index_t stride_u,
									     proteus_index_t nFreeVDOF_global,
									     np.ndarray interiorElementBoundaries,
									     np.ndarray elementBoundaryElements,
									     np.ndarray elementBoundaryLocalElementBoundaries,
									     np.ndarray nFreeDOF_element_r,
									     np.ndarray freeLocal_r,
									     np.ndarray freeGlobal_r,
									     np.ndarray nFreeDOF_element_u,
									     np.ndarray freeLocal_u,
									     np.ndarray freeGlobal_u,
									     np.ndarray elementBoundaryFluxJacobian_2sided,
									     np.ndarray w_dS,
									     np.ndarray jac):
    cdef int nInteriorElementBoundaries_global = interiorElementBoundaries.shape[0]
    cdef int nElementBoundaries_element = w_dS.shape[1]
    cdef int nQuadraturePoints_elementBoundary = w_dS.shape[2]
    cdef int nDOF_test_element = w_dS.shape[3]
    cdef int nDOF_trial_element = elementBoundaryFluxJacobian_2sided.shape[4]
    cupdateGlobalJacobianFromInteriorElementBoundaryFluxJacobian_2sided_dense(nInteriorElementBoundaries_global,
									      nElementBoundaries_element,
									      nQuadraturePoints_elementBoundary,
									      nDOF_test_element,
									      nDOF_trial_element,
									      offset_r,
									      stride_r,
									      offset_u,
									      stride_u,
									      nFreeVDOF_global,
									      <proteus_index_t*> interiorElementBoundaries.data,
									      <proteus_index_t*> elementBoundaryElements.data,
									      <int*> elementBoundaryLocalElementBoundaries.data,
									      <int*> nFreeDOF_element_r.data,
									      <int*> freeLocal_r.data,
									      <proteus_index_t*> freeGlobal_r.data,
									      <int*> nFreeDOF_element_u.data,
									      <int*> freeLocal_u.data,
									      <proteus_index_t*> freeGlobal_u.data,
									      <double*> elementBoundaryFluxJacobian_2sided.data,
									      <double*> w_dS.data,
									      <double*> jac.data)
def updateInteriorTwoSidedElementBoundaryFlux(np.ndarray interiorElementBoundaries,
					      np.ndarray elementBoundaryElements,
					      np.ndarray elementBoundaryLocalElementBoundaries,
					      np.ndarray flux,
					      np.ndarray w_dS,
					      np.ndarray residual):
    cdef int nInteriorElementBoundaries_global = interiorElementBoundaries.shape[0]
    cdef int nElementBoundaries_element = w_dS.shape[1]
    cdef int nQuadraturePoints_elementBoundary = w_dS.shape[2]
    cdef int nDOF_test_element = w_dS.shape[3]
    cupdateInteriorTwoSidedElementBoundaryFlux(nInteriorElementBoundaries_global,
					       nElementBoundaries_element,
					       nQuadraturePoints_elementBoundary,
					       nDOF_test_element,
					       <proteus_index_t*> interiorElementBoundaries.data,
					       <proteus_index_t*> elementBoundaryElements.data,
					       <int*> elementBoundaryLocalElementBoundaries.data,
					       <double*> flux.data,
					       <double*> w_dS.data,
					       <double*> residual.data)
def calculateCFLADR2speeds(np.ndarray elementDiameter,
			   np.ndarray dm,
			   np.ndarray df1,
			   np.ndarray df2,
			   np.ndarray cfl):
    cdef int nElements_global = df1.shape[0]
    cdef int nQuadraturePoints_element = df1.shape[1]
    cdef int nSpace = df1.shape[2]
    ccalculateCFLADR2speeds(nElements_global,
                            nQuadraturePoints_element,
                            nSpace,
                            <double*> elementDiameter.data,
			    <double*> dm.data,
			    <double*> df1.data,
			    <double*> df2.data,
			    <double*> cfl.data)
def checkElementBoundaryAndExteriorElementBoundaryArraysSame(double tolerance,
							     np.ndarray exteriorElementBoundariesArray,
							     np.ndarray elementBoundaryElementsArray,
							     np.ndarray elementBoundaryLocalElementBoundariesArray,
							     np.ndarray ebq_val,
							     np.ndarray ebqe_val,
							     np.ndarray firstBadIndex):
    cdef int output
    cdef int nElementBoundaries_element = ebq_val.shape[1]
    cdef int nExteriorElementBoundaries_global = exteriorElementBoundariesArray.shape[0]
    cdef int nQuadraturePoints_elementBoundary = ebqe_val.shape[1]
    cdef int nValuesPerQuadraturePoint = 1
    cdef int nd = ebqe_val.ndim
    if nd > 2:
        for i in range(2,nd):
            nValuesPerQuadraturePoint *= ebqe_val.shape[i]
    output = ccheckElementBoundaryAndExteriorElementBoundaryArraysSame(nElementBoundaries_element,
								       nExteriorElementBoundaries_global,
								       nQuadraturePoints_elementBoundary,
								       nValuesPerQuadraturePoint,
								       tolerance,
								       <proteus_index_t*> exteriorElementBoundariesArray.data,
								       <proteus_index_t*> elementBoundaryElementsArray.data,
								       <int*> elementBoundaryLocalElementBoundariesArray.data,
								       <double*> ebq_val.data,
								       <double*> ebqe_val.data,
								       <proteus_index_t*> firstBadIndex.data)
    return output
def checkGlobalElementBoundaryAndExteriorElementBoundaryArraysSame(double tolerance,
								   np.ndarray exteriorElementBoundariesArray,
								   np.ndarray elementBoundaryElementsArray,
								   np.ndarray elementBoundaryLocalElementBoundariesArray,
								   np.ndarray ebq_global_val,
								   np.ndarray ebqe_val,
								   np.ndarray firstBadIndex):
    cdef int output
    cdef int nExteriorElementBoundaries_global = exteriorElementBoundariesArray.shape[0]
    cdef int nQuadraturePoints_elementBoundary = ebqe_val.shape[1]
    cdef int nValuesPerQuadraturePoint = 1
    cdef int nd = ebqe_val
    if nd > 2:
        for i in range(2,nd):
            nValuesPerQuadraturePoint *= ebqe_val.shape[i]
    output = ccheckGlobalElementBoundaryAndExteriorElementBoundaryArraysSame(nExteriorElementBoundaries_global,
									     nQuadraturePoints_elementBoundary,
									     nValuesPerQuadraturePoint,
									     tolerance,
									     <proteus_index_t*> exteriorElementBoundariesArray.data,
									     <proteus_index_t*> elementBoundaryElementsArray.data,
									     <int*> elementBoundaryLocalElementBoundariesArray.data,
									     <double*> ebq_global_val.data,
									     <double*> ebqe_val.data,
									     <proteus_index_t*> firstBadIndex.data)
def calculateExteriorElementBoundaryStress3D(np.ndarray elementBoundaryMaterialTypes,
					     np.ndarray exteriorElementBoundaries,
					     np.ndarray elementBoundaryElements,
					     np.ndarray elementBoundaryLocalElementBoundaries,
					     np.ndarray p,
					     np.ndarray mom_flux_vec_u,
					     np.ndarray mom_flux_vec_v,
					     np.ndarray mom_flux_vec_w,
					     np.ndarray dS,
					     np.ndarray n,
					     np.ndarray F):
    cdef int nExteriorElementBoundaries_global = p.shape[0]
    cdef int nQuadraturePoints_elementBoundary = p.shape[1]
    ccalculateExteriorElementBoundaryStress3D(nExteriorElementBoundaries_global,
					      nQuadraturePoints_elementBoundary,
                                              <int*> elementBoundaryMaterialTypes.data,
					      <proteus_index_t*> exteriorElementBoundaries.data,
					      <proteus_index_t*> elementBoundaryElements.data,
					      <int*> elementBoundaryLocalElementBoundaries.data,
					      <double*> p.data,
					      <double*> mom_flux_vec_u.data,
					      <double*> mom_flux_vec_v.data,
					      <double*> mom_flux_vec_w.data,
					      <double*> dS.data,
					      <double*> n.data,
					      <double*> F.data)
def calculateExteriorElementBoundaryStress2D(np.ndarray elementBoundaryMaterialTypes,
					     np.ndarray exteriorElementBoundaries,
					     np.ndarray elementBoundaryElements,
					     np.ndarray elementBoundaryLocalElementBoundaries,
					     np.ndarray p,
					     np.ndarray mom_flux_vec_u,
					     np.ndarray mom_flux_vec_v,
					     np.ndarray dS,
					     np.ndarray n,
					     np.ndarray F):
    cdef int nExteriorElementBoundaries_global = p.shape[0]
    cdef int nQuadraturePoints_elementBoundary = p.shape[1]
    ccalculateExteriorElementBoundaryStress2D(nExteriorElementBoundaries_global,
					      nQuadraturePoints_elementBoundary,
					      <int*> elementBoundaryMaterialTypes.data,
					      <proteus_index_t*> exteriorElementBoundaries.data,
					      <proteus_index_t*> elementBoundaryElements.data,
					      <int*> elementBoundaryLocalElementBoundaries.data,
					      <double*> p.data,
					      <double*> mom_flux_vec_u.data,
					      <double*> mom_flux_vec_v.data,
					      <double*> dS.data,
					      <double*> n.data,
					      <double*> F.data)
def copyLeftElementBoundaryInfo(np.ndarray elementBoundaryElementsArray,
                                np.ndarray elementBoundaryLocalElementBoundariesArray,
                                np.ndarray exteriorElementBoundariesArray,
                                np.ndarray interiorElementBoundariesArray,
                                np.ndarray x,
                                np.ndarray n,
                                np.ndarray xg,
                                np.ndarray ng):
    cdef int nElementBoundaries_element = n.shape[1]
    cdef int nElementBoundaryQuadraturePoints_elementBoundary = n.shape[2]
    cdef int nSpace_global = n.shape[3]
    cdef int nExteriorElementBoundaries_global = exteriorElementBoundariesArray.shape[0]
    cdef int nInteriorElementBoundaries_global = interiorElementBoundariesArray.shape[0]
    ccopyLeftElementBoundaryInfo(nElementBoundaries_element,
                                 nElementBoundaryQuadraturePoints_elementBoundary,
                                 nSpace_global,
                                 nExteriorElementBoundaries_global,
                                 nInteriorElementBoundaries_global,
                                 <proteus_index_t*>elementBoundaryElementsArray.data,
                                 <int *>elementBoundaryLocalElementBoundariesArray.data,
                                 <proteus_index_t*>exteriorElementBoundariesArray.data,
                                 <proteus_index_t*>interiorElementBoundariesArray.data,
                                 <double*>x.data,
                                 <double*>n.data,
                                 <double*>xg.data,
                                 <double*>ng.data)
def parametricFiniteElementSpace_getValues(np.ndarray psi,
                                           np.ndarray vArray):
    cdef int nElements_global = vArray.shape[0]
    cdef int nQuadraturePoints_element = vArray.shape[1]
    cdef int nDOF_element = vArray.shape[2]
    cparametricFiniteElementSpace_getValues(nElements_global,
                                            nQuadraturePoints_element,
                                            nDOF_element,
                                            <double*>psi.data,
                                            <double*>vArray.data)
def parametricFiniteElementSpace_getValuesTrace(np.ndarray psi,
                                                np.ndarray permutations,
                                                np.ndarray vArray):
    cdef int nElements_global = vArray.shape[0]
    cdef int nElementBoundaries_element = vArray.shape[1]
    cdef int nElementBoundaryQuadraturePoints_elementBoundary = vArray.shape[2]
    cdef int nDOF_element = vArray.shape[3]
    cparametricFiniteElementSpace_getValuesTrace(nElements_global,
                                                 nElementBoundaries_element,
                                                 nElementBoundaryQuadraturePoints_elementBoundary,
                                                 nDOF_element,
                                                 <double*>psi.data,
                                                 <int *>permutations.data,
                                                 <double*>vArray.data)
def parametricFiniteElementSpace_getGradientValues(np.ndarray grad_psi,
                                                   np.ndarray inverseJacobianArray,
                                                   np.ndarray grad_vArray):
    cdef int nElements_global = grad_vArray.shape[0]
    cdef int nQuadraturePoints_element = grad_vArray.shape[1]
    cdef int nDOF_element = grad_vArray.shape[2]
    cdef int nSpace_global = grad_vArray.shape[3]
    cparametricFiniteElementSpace_getGradientValues(nElements_global,
                                                    nQuadraturePoints_element,
                                                    nDOF_element,
                                                    nSpace_global,
                                                    <double*>grad_psi.data,
                                                    <double*>inverseJacobianArray.data,
                                                    <double*>grad_vArray.data)
def parametricFiniteElementSpace_getGradientValuesTrace(np.ndarray grad_psi,
                                                        np.ndarray permutations,
                                                        np.ndarray inverseJacobianArray,
                                                        np.ndarray grad_vArray):
    cdef int nElements_global = grad_vArray.shape[0]
    cdef int nElementBoundaries_element = grad_vArray.shape[1]
    cdef int nElementBoundaryQuadraturePoints_elementBoundary = grad_vArray.shape[2]
    cdef int nDOF_element = grad_vArray.shape[3]
    cdef int nSpace_global = grad_vArray.shape[4]
    cparametricFiniteElementSpace_getGradientValuesTrace(nElements_global,
                                                         nElementBoundaries_element,
                                                         nElementBoundaryQuadraturePoints_elementBoundary,
                                                         nDOF_element,
                                                         nSpace_global,
                                                         <double*>grad_psi.data,
                                                         <int *>permutations.data,
                                                         <double*>inverseJacobianArray.data,
                                                         <double*>grad_vArray.data)
def parametricMaps_getPermutations(np.ndarray xiArray,
                                   np.ndarray permutations):
    cdef int nElements_global = xiArray.shape[0]
    cdef int nElementBoundaries_element = xiArray.shape[1]
    cdef int nElementBoundaryQuadraturePoints_elementBoundary = xiArray.shape[2]
    cdef int nSpace_global = xiArray.shape[3]
    cparametricMaps_getPermutations(nElements_global,
                                    nElementBoundaries_element,
                                    nElementBoundaryQuadraturePoints_elementBoundary,
                                    nSpace_global,
                                    <double*>xiArray.data,
                                    <int *>permutations.data)
def parametricMaps_getPermutationsGlobalExterior(np.ndarray exteriorElementBoundariesArray,
						 np.ndarray elementBoundaryElementsArray,
						 np.ndarray elementBoundaryLocalElementBoundariesArray,
						 np.ndarray xiArray,
						 np.ndarray permutations):
    nElementBoundaryQuadraturePoints_elementBoundary = xiArray.shape[1]
    nSpace_global = xiArray.shape[2]
    nExteriorElementBoundaries_global = exteriorElementBoundariesArray.shape[0]
    cparametricMaps_getPermutationsGlobalExterior(nElementBoundaryQuadraturePoints_elementBoundary,
						  nSpace_global,
						  nExteriorElementBoundaries_global,
                                                  <proteus_index_t*> exteriorElementBoundariesArray.data,
						  <proteus_index_t*> elementBoundaryElementsArray.data,
						  <int*> elementBoundaryLocalElementBoundariesArray.data,
						  <double*> xiArray.data,
						  <int*> permutations.data)
def getPermutationsGlobal(np.ndarray xArray,
			  np.ndarray xArrayNew,
			  np.ndarray permutations):
    cdef int nElementBoundaries_global = xArray.shape[0]
    cdef int nElementBoundaryQuadraturePoints_elementBoundary = xArray.shape[1]
    cgetPermutationsGlobal(nElementBoundaries_global,
                           nElementBoundaryQuadraturePoints_elementBoundary,
                           <double*> xArray.data,
			   <double*> xArrayNew.data,
			   <int*> permutations.data)
def parametricMaps_getValues(np.ndarray psi,
                             np.ndarray l2g,
                             np.ndarray nodeArray,
                             np.ndarray xArray):
    cdef int nElements_global = xArray.shape[0]
    cdef int nQuadraturePoints_element = xArray.shape[1]
    cdef int nDOF_element = l2g.shape[1]
    cdef int nSpace_global = xArray.shape[2]
    cparametricMaps_getValues(nElements_global,
                              nQuadraturePoints_element,
                              nDOF_element,
                              nSpace_global,
                              <double*>psi.data,
                              <proteus_index_t*>l2g.data,
                              <double*>nodeArray.data,
                              <double*>xArray.data)
def parametricMaps_getValuesTrace(np.ndarray psi,
                                  np.ndarray l2g,
                                  np.ndarray nodeArray,
                                  np.ndarray xArray):
    cdef int nElements_global = xArray.shape[0]
    cdef int nElementBoundaries_element = xArray.shape[1]
    cdef int nQuadraturePoints_element = xArray.shape[2]
    cdef int nDOF_element = l2g.shape[1]
    cdef int nSpace_global = xArray.shape[3]
    cparametricMaps_getValuesTrace(nElements_global,
                                   nElementBoundaries_element,
                                   nQuadraturePoints_element,
                                   nDOF_element,
                                   nSpace_global,
                                   <double*>psi.data,
                                   <proteus_index_t*>l2g.data,
                                   <double*>nodeArray.data,
                                   <double*>xArray.data)
def parametricMaps_getInverseValues(np.ndarray inverseJacobian,
                                    np.ndarray l2g,
                                    np.ndarray nodeArray,
                                    np.ndarray xArray,
                                    np.ndarray xiArray):
    cdef int nElements_global = xArray.shape[0]
    cdef int nQuadraturePoints_element = xArray.shape[1]
    cdef int nDOF_element = l2g.shape[1]
    cdef int nSpace_global = inverseJacobian.shape[2]
    cparametricMaps_getInverseValues(nElements_global,
                                     nQuadraturePoints_element,
                                     nDOF_element,
                                     nSpace_global,
                                     <double*>inverseJacobian.data,
                                     <proteus_index_t*>l2g.data,
                                     <double*>nodeArray.data,
                                     <double*>xArray.data,
                                     <double*>xiArray.data)
def parametricMaps_getInverseValuesTrace(np.ndarray inverseJacobian,
                                         np.ndarray l2g,
                                         np.ndarray nodeArray,
                                         np.ndarray xArray,
                                         np.ndarray xiArray):
    cdef int nElements_global = xArray.shape[0]
    cdef int nElementBoundaries_element = xArray.shape[1]
    cdef int nElementBoundaryQuadraturePoints_elementBoundary = xArray.shape[2]
    cdef int nDOF_element = l2g.shape[1]
    cdef int nSpace_global = inverseJacobian.shape[3]
    cparametricMaps_getInverseValuesTrace(nElements_global,
                                          nElementBoundaries_element,
                                          nElementBoundaryQuadraturePoints_elementBoundary,
                                          nDOF_element,
                                          nSpace_global,
                                          <double*>inverseJacobian.data,
                                          <proteus_index_t*>l2g.data,
                                          <double*>nodeArray.data,
                                          <double*>xArray.data,
                                          <double*>xiArray.data)
def parametricMaps_getJacobianValues(np.ndarray grad_psi,
                                     np.ndarray l2g,
                                     np.ndarray nodeArray,
                                     np.ndarray jacobianArray,
                                     np.ndarray jacobianDeterminantArray,
                                     np.ndarray jacobianInverseArray):
    cdef int nd = jacobianArray.shape[2]
    if nd == 1:
        cparametricMaps_getJacobianValues1D(jacobianArray.shape[0],
                                            jacobianArray.shape[1],
                                            l2g.shape[1],
                                            <double*>grad_psi.data,
                                            <proteus_index_t*>l2g.data,
                                            <double*>nodeArray.data,
                                            <double*>jacobianArray.data,
                                            <double*>jacobianDeterminantArray.data,
                                            <double*>jacobianInverseArray.data)
    elif nd == 2:
        cparametricMaps_getJacobianValues2D(jacobianArray.shape[0],
                                            jacobianArray.shape[1],
                                            l2g.shape[1],
                                            <double*>grad_psi.data,
                                            <proteus_index_t*>l2g.data,
                                            <double*>nodeArray.data,
                                            <double*>jacobianArray.data,
                                            <double*>jacobianDeterminantArray.data,
                                            <double*>jacobianInverseArray.data)
    elif nd == 3:
        cparametricMaps_getJacobianValues3D(jacobianArray.shape[0],
                                            jacobianArray.shape[1],
                                            l2g.shape[1],
                                            <double*>grad_psi.data,
                                            <proteus_index_t*>l2g.data,
                                            <double*>nodeArray.data,
                                            <double*>jacobianArray.data,
                                            <double*>jacobianDeterminantArray.data,
                                            <double*>jacobianInverseArray.data)
    else:
        print("error in getJacobianValues...jacobian not sized properly")
def parametricMaps_getJacobianValuesTrace(np.ndarray grad_psi,
                                          np.ndarray boundaryNormals,
                                          np.ndarray boundaryJacobians,
                                          np.ndarray l2g,
                                          np.ndarray nodeArray,
                                          np.ndarray jacobianInverseArray,
                                          np.ndarray metricTensorArray,
                                          np.ndarray metricTensorDeterminantSqrtArray,
                                          np.ndarray unitNormalArray):
    cdef int nd = jacobianInverseArray.shape[3]
    if nd == 1:
        cparametricMaps_getJacobianValuesTrace1D(jacobianInverseArray.shape[0],
                                                 jacobianInverseArray.shape[1],
                                                 jacobianInverseArray.shape[2],
                                                 l2g.shape[1],
                                                 <double*>grad_psi.data,
                                                 <double*>boundaryNormals.data,
                                                 <double*>boundaryJacobians.data,
                                                 <proteus_index_t*>l2g.data,
                                                 <double*>nodeArray.data,
                                                 <double*>jacobianInverseArray.data,
                                                 <double*>metricTensorArray.data,
                                                 <double*>metricTensorDeterminantSqrtArray.data,
                                                 <double*>unitNormalArray.data)
    elif nd == 2:
        cparametricMaps_getJacobianValuesTrace2D(jacobianInverseArray.shape[0],
                                                 jacobianInverseArray.shape[1],
                                                 jacobianInverseArray.shape[2],
                                                 l2g.shape[1],
                                                 <double*>grad_psi.data,
                                                 <double*>boundaryNormals.data,
                                                 <double*>boundaryJacobians.data,
                                                 <proteus_index_t*>l2g.data,
                                                 <double*>nodeArray.data,
                                                 <double*>jacobianInverseArray.data,
                                                 <double*>metricTensorArray.data,
                                                 <double*>metricTensorDeterminantSqrtArray.data,
                                                 <double*>unitNormalArray.data)
    elif nd == 3:
        cparametricMaps_getJacobianValuesTrace3D(jacobianInverseArray.shape[0],
                                                 jacobianInverseArray.shape[1],
                                                 jacobianInverseArray.shape[2],
                                                 l2g.shape[1],
                                                 <double*>grad_psi.data,
                                                 <double*>boundaryNormals.data,
                                                 <double*>boundaryJacobians.data,
                                                 <proteus_index_t*>l2g.data,
                                                 <double*>nodeArray.data,
                                                 <double*>jacobianInverseArray.data,
                                                 <double*>metricTensorArray.data,
                                                 <double*>metricTensorDeterminantSqrtArray.data,
                                                 <double*>unitNormalArray.data)
    else:
        print("error in getJacobianValuesTrace...jacobianInverse not sized properly")
def updateMass_weak(np.ndarray mt,
                    np.ndarray w_dV,
                    np.ndarray weak_residual):
    cdef int nElements_global = w_dV.shape[0]
    cdef int nQuadraturePoints_element = w_dV.shape[1]
    cdef int nDOF_test_element = w_dV.shape[2]
    cupdateMass_weak(nElements_global,
                     nQuadraturePoints_element,
                     nDOF_test_element,
                     <double*>mt.data,
                     <double*>w_dV.data,
                     <double*>weak_residual.data)
def updateMassJacobian_weak(np.ndarray dmt,
                            np.ndarray v_X_w_dV,
                            np.ndarray jacobian_weak_residual):
    cdef int nElements_global = v_X_w_dV.shape[0]
    cdef int nQuadraturePoints_element = v_X_w_dV.shape[1]
    cdef int nDOF_trial_element = v_X_w_dV.shape[2]
    cdef int nDOF_test_element = v_X_w_dV.shape[3]
    cupdateMassJacobian_weak(nElements_global,
                             nQuadraturePoints_element,
                             nDOF_trial_element,
                             nDOF_test_element,
                             <double*>dmt.data,
                             <double*>v_X_w_dV.data,
                             <double*>jacobian_weak_residual.data)
def updateMassJacobian_weak_lowmem(np.ndarray dmt,
                                   np.ndarray v,
                                   np.ndarray w_dV,
                                   np.ndarray jacobian_weak_residual):
    cdef int nElements_global = v.shape[0]
    cdef int nQuadraturePoints_element = v.shape[1]
    cdef int nDOF_trial_element = v.shape[2]
    cdef int nDOF_test_element = w_dV.shape[2]
    cupdateMassJacobian_weak_lowmem(nElements_global,
                                    nQuadraturePoints_element,
                                    nDOF_trial_element,
                                    nDOF_test_element,
                                    <double*> dmt.data,
                                    <double*> v.data,
                                    <double*> w_dV.data,
                                    <double*> jacobian_weak_residual.data)
def updateMass_strong(np.ndarray mt,
                      np.ndarray strong_residual):
    cdef int nElements_global = mt.shape[0]
    cdef int nQuadraturePoints_element = mt.shape[1]
    cupdateMass_strong(nElements_global,
                       nQuadraturePoints_element,
                       <double*>mt.data,
                       <double*>strong_residual.data)
def updateMassJacobian_strong(np.ndarray dmt,
                              np.ndarray v,
                              np.ndarray dstrong_residual):
    cdef int nElements_global = v.shape[0]
    cdef int nQuadraturePoints_element = v.shape[1]
    cdef int nDOF_trial_element = v.shape[2]
    cupdateMassJacobian_strong(nElements_global,
                               nQuadraturePoints_element,
                               nDOF_trial_element,
                               <double*>dmt.data,
                               <double*>v.data,
                               <double*>dstrong_residual.data)
def updateMass_adjoint(np.ndarray dmt,
                       np.ndarray w_dV,
                       np.ndarray Lstar_w_dV):
    cdef int nElements_global = w_dV.shape[0]
    cdef int nQuadraturePoints_element = w_dV.shape[1]
    cdef int nDOF_test_element = w_dV.shape[2]
    cupdateMass_adjoint(nElements_global,
                        nQuadraturePoints_element,
                        nDOF_test_element,
                        <double*>dmt.data,
                        <double*>w_dV.data,
                        <double*>Lstar_w_dV.data)
def updateAdvection_weak(np.ndarray f,
                         np.ndarray grad_w_dV,
                         np.ndarray weak_residual):
    cdef int nElements_global = grad_w_dV.shape[0]
    cdef int nQuadraturePoints_element = grad_w_dV.shape[1]
    cdef int nDOF_test_element = grad_w_dV.shape[2]
    cdef int nSpace = grad_w_dV.shape[3]
    cupdateAdvection_weak(nElements_global,
                          nQuadraturePoints_element,
                          nDOF_test_element,
                          nSpace,
                          <double*>f.data,
                          <double*>grad_w_dV.data,
                          <double*>weak_residual.data)
def updateAdvectionJacobian_weak(np.ndarray df,
                                 np.ndarray v_X_grad_w_dV,
                                 np.ndarray jacobian_weak_residual):
    cdef int nElements_global = v_X_grad_w_dV.shape[0]
    cdef int nQuadraturePoints_element = v_X_grad_w_dV.shape[1]
    cdef int nDOF_trial_element = v_X_grad_w_dV.shape[2]
    cdef int nDOF_test_element = v_X_grad_w_dV.shape[3]
    cdef int nSpace = v_X_grad_w_dV.shape[4]
    cupdateAdvectionJacobian_weak(nElements_global,
                                  nQuadraturePoints_element,
                                  nDOF_trial_element,
                                  nDOF_test_element,
                                  nSpace,
                                  <double*>df.data,
                                  <double*>v_X_grad_w_dV.data,
                                  <double*>jacobian_weak_residual.data)
def updateAdvectionJacobian_weak_lowmem(np.ndarray df,
                                        np.ndarray v,
                                        np.ndarray grad_w_dV,
                                        np.ndarray jacobian_weak_residual):
    cdef int nElements_global = v.shape[0]
    cdef int nQuadraturePoints_element = v.shape[1]
    cdef int nDOF_trial_element = v.shape[2]
    cdef int nDOF_test_element = grad_w_dV.shape[2]
    cdef int nSpace = grad_w_dV.shape[3]
    cupdateAdvectionJacobian_weak_lowmem(nElements_global,
                                         nQuadraturePoints_element,
                                         nDOF_trial_element,
                                         nDOF_test_element,
                                         nSpace,
                                         <double*> df.data,
                                         <double*> v.data,
                                         <double*> grad_w_dV.data,
                                         <double*> jacobian_weak_residual.data)
def updateAdvection_strong(np.ndarray df,
                           np.ndarray grad_u,
                           np.ndarray strong_residual):
    cdef int nElements_global = grad_u.shape[0]
    cdef int nQuadraturePoints_element = grad_u.shape[1]
    cdef int nSpace = grad_u.shape[2]
    cupdateAdvection_strong(nElements_global,
                            nQuadraturePoints_element,
                            nSpace,
                            <double*>df.data,
                            <double*>grad_u.data,
                            <double*>strong_residual.data)
def updateAdvectionJacobian_strong(np.ndarray df,
                                   np.ndarray grad_v,
                                   np.ndarray dstrong_residual):
    cdef int nElements_global = grad_v.shape[0]
    cdef int nQuadraturePoints_element = grad_v.shape[1]
    cdef int nDOF_trial_element = grad_v.shape[2]
    cdef int nSpace = grad_v.shape[3]
    cupdateAdvectionJacobian_strong(nElements_global,
                                    nQuadraturePoints_element,
                                    nDOF_trial_element,
                                    nSpace,
                                    <double*>df.data,
                                    <double*>grad_v.data,
                                    <double*>dstrong_residual.data)
def updateAdvection_adjoint(np.ndarray df,
                            np.ndarray grad_w_dV,
                            np.ndarray Lstar_w_dV):
    cdef int nElements_global = grad_w_dV.shape[0]
    cdef int nQuadraturePoints_element = grad_w_dV.shape[1]
    cdef int nDOF_test_element = grad_w_dV.shape[2]
    cdef int nSpace = grad_w_dV.shape[3]
    cupdateAdvection_adjoint(nElements_global,
                             nQuadraturePoints_element,
                             nDOF_test_element,
                             nSpace,
                             <double*>df.data,
                             <double*>grad_w_dV.data,
                             <double*>Lstar_w_dV.data)
def updateHamiltonian_weak(np.ndarray H,
                           np.ndarray w_dV,
                           np.ndarray weak_residual):
    cdef int nElements_global = w_dV.shape[0]
    cdef int nQuadraturePoints_element = w_dV.shape[1]
    cdef int nDOF_test_element = w_dV.shape[2]
    cupdateHamiltonian_weak(nElements_global,
                            nQuadraturePoints_element,
                            nDOF_test_element,
                            <double*>H.data,
                            <double*>w_dV.data,
                            <double*>weak_residual.data)
def updateHamiltonianJacobian_weak(np.ndarray dH,
                                   np.ndarray grad_v_X_w_dV,
                                   np.ndarray jacobian_weak_residual):
    cdef int nElements_global = grad_v_X_w_dV.shape[0]
    cdef int nQuadraturePoints_element = grad_v_X_w_dV.shape[1]
    cdef int nDOF_trial_element = grad_v_X_w_dV.shape[2]
    cdef int nDOF_test_element = grad_v_X_w_dV.shape[3]
    cdef int nSpace = grad_v_X_w_dV.shape[4]
    cupdateHamiltonianJacobian_weak(nElements_global,
                                    nQuadraturePoints_element,
                                    nDOF_trial_element,
                                    nDOF_test_element,
                                    nSpace,
                                    <double*>dH.data,
                                    <double*>grad_v_X_w_dV.data,
                                    <double*>jacobian_weak_residual.data)
def updateHamiltonianJacobian_weak_lowmem(np.ndarray dH,
                                          np.ndarray grad_v,
                                          np.ndarray w_dV,
                                          np.ndarray jacobian_weak_residual):
    cdef int nElements_global = grad_v.shape[0]
    cdef int nQuadraturePoints_element = grad_v.shape[1]
    cdef int nDOF_trial_element = grad_v.shape[2]
    cdef int nDOF_test_element = w_dV.shape[2]
    cdef int nSpace = grad_v.shape[3]
    cupdateHamiltonianJacobian_weak_lowmem(nElements_global,
                                           nQuadraturePoints_element,
                                           nDOF_trial_element,
                                           nDOF_test_element,
                                           nSpace,
                                           <double*> dH.data,
                                           <double*> grad_v.data,
                                           <double*> w_dV.data,
                                           <double*> jacobian_weak_residual.data)
def updateHamiltonian_strong(np.ndarray dH,
                             np.ndarray grad_u,
                             np.ndarray strong_residual):
    cdef int nElements_global = grad_u.shape[0]
    cdef int nQuadraturePoints_element = grad_u.shape[1]
    cdef int nSpace = grad_u.shape[2]
    cupdateHamiltonian_strong(nElements_global,
                              nQuadraturePoints_element,
                              nSpace,
                              <double*>dH.data,
                              <double*>grad_u.data,
                              <double*>strong_residual.data)
def updateHamiltonianJacobian_strong(np.ndarray dH,
                                     np.ndarray grad_v,
                                     np.ndarray dstrong_residual):
    cdef int nElements_global = grad_v.shape[0]
    cdef int nQuadraturePoints_element = grad_v.shape[1]
    cdef int nDOF_trial_element = grad_v.shape[2]
    cdef int nSpace = grad_v.shape[3]
    cupdateHamiltonianJacobian_strong(nElements_global,
                                      nQuadraturePoints_element,
                                      nDOF_trial_element,
                                      nSpace,
                                      <double*>dH.data,
                                      <double*>grad_v.data,
                                      <double*>dstrong_residual.data)
def updateHamiltonian_adjoint(np.ndarray dH,
                              np.ndarray grad_w_dV,
                              np.ndarray Lstar_w_dV):
    cdef int nElements_global = grad_w_dV.shape[0]
    cdef int nQuadraturePoints_element = grad_w_dV.shape[1]
    cdef int nDOF_test_element = grad_w_dV.shape[2]
    cdef int nSpace = grad_w_dV.shape[3]
    cupdateHamiltonian_adjoint(nElements_global,
                               nQuadraturePoints_element,
                               nDOF_test_element,
                               nSpace,
                               <double*>dH.data,
                               <double*>grad_w_dV.data,
                               <double*>Lstar_w_dV.data)
def updateDiffusion_weak(np.ndarray a,
                         np.ndarray grad_phi_X_grad_w_dV,
                         np.ndarray weak_residual):
    cdef int nElements_global = grad_phi_X_grad_w_dV.shape[0]
    cdef int nQuadraturePoints_element = grad_phi_X_grad_w_dV.shape[1]
    cdef int nDOF_test_element = grad_phi_X_grad_w_dV.shape[2]
    cdef int nSpace = grad_phi_X_grad_w_dV.shape[3]
    cupdateDiffusion_weak(nElements_global,
                          nQuadraturePoints_element,
                          nDOF_test_element,
                          nSpace,
                          <double*>a.data,
                          <double*>grad_phi_X_grad_w_dV.data,
                          <double*>weak_residual.data)
def updateDiffusion_weak_lowmem(np.ndarray a,
                                np.ndarray grad_phi,
                                np.ndarray grad_w_dV,
                                np.ndarray weak_residual):
    nElements_global = grad_w_dV.shape[0]
    nQuadraturePoints_element = grad_w_dV.shape[1]
    nDOF_test_element = grad_w_dV.shape[2]
    nSpace = grad_w_dV.shape[3]
    cupdateDiffusion_weak_lowmem(nElements_global,
                                 nQuadraturePoints_element,
                                 nDOF_test_element,
                                 nSpace,
                                 <double*> a.data,
                                 <double*> grad_phi.data,
                                 <double*> grad_w_dV.data,
                                 <double*> weak_residual.data)
def updateDiffusion_weak_sd(np.ndarray rowptr,
                            np.ndarray colind,
                            np.ndarray a,
                            np.ndarray grad_phi,
                            np.ndarray grad_w_dV,
                            np.ndarray weak_residual):
    cdef int nElements_global = grad_w_dV.shape[0]
    cdef int nQuadraturePoints_element = grad_w_dV.shape[1]
    cdef int nDOF_test_element = grad_w_dV.shape[2]
    cdef int nSpace = grad_w_dV.shape[3]
    cupdateDiffusion_weak_sd(nElements_global,
                             nQuadraturePoints_element,
                             nDOF_test_element,
                             nSpace,
                             <proteus_csr_index_t*> rowptr.data,
                             <proteus_csr_index_t*> colind.data,
                             <double*> a.data,
                             <double*> grad_phi.data,
                             <double*> grad_w_dV.data,
                             <double*> weak_residual.data)
def updateDiffusionJacobian_weak(np.ndarray l2g,
                                 np.ndarray a,
                                 np.ndarray da,
                                 np.ndarray grad_phi_X_grad_w_dV,
                                 np.ndarray dphi,
                                 np.ndarray v,
                                 np.ndarray grad_v_X_grad_w_dV,
                                 np.ndarray jacobian_weak_residual):
    cdef int nElements_global = grad_v_X_grad_w_dV.shape[0]
    cdef int nQuadraturePoints_element = grad_v_X_grad_w_dV.shape[1]
    cdef int nDOF_trial_element = grad_v_X_grad_w_dV.shape[2]
    cdef int nDOF_test_element = grad_v_X_grad_w_dV.shape[3]
    cdef int nSpace = grad_v_X_grad_w_dV.shape[4]
    cupdateDiffusionJacobian_weak(nElements_global,
                                  nQuadraturePoints_element,
                                  nDOF_trial_element,
                                  nDOF_test_element,
                                  nSpace,
                                  <proteus_index_t*>l2g.data,
                                  <double*>a.data,
                                  <double*>da.data,
                                  <double*>grad_phi_X_grad_w_dV.data,
                                  <double*>dphi.data,
                                  <double*>v.data,
                                  <double*>grad_v_X_grad_w_dV.data,
                                  <double*>jacobian_weak_residual.data)
def updateDiffusionJacobian_weak_lowmem(np.ndarray l2g,
                                        np.ndarray a,
                                        np.ndarray da,
                                        np.ndarray grad_phi,
                                        np.ndarray grad_w_dV,
                                        np.ndarray dphi,
                                        np.ndarray v,
                                        np.ndarray grad_v,
                                        np.ndarray jacobian_weak_residual):
    cdef int nElements_global = grad_w_dV.shape[0]
    cdef int nQuadraturePoints_element = grad_w_dV.shape[1]
    cdef int nDOF_trial_element = grad_v.shape[2]
    cdef int nDOF_test_element = grad_w_dV.shape[2]
    cdef int nSpace = grad_w_dV.shape[3]
    cupdateDiffusionJacobian_weak_lowmem(nElements_global,
                                         nQuadraturePoints_element,
                                         nDOF_trial_element,
                                         nDOF_test_element,
                                         nSpace,
                                         <proteus_index_t*> l2g.data,
                                         <double*> a.data,
                                         <double*> da.data,
                                         <double*> grad_phi.data,
                                         <double*> grad_w_dV.data,
                                         <double*> dphi.data,
                                         <double*> v.data,
                                         <double*> grad_v.data,
                                         <double*> jacobian_weak_residual.data)
def updateDiffusionJacobian_weak_sd(np.ndarray rowptr,
				    np.ndarray colind,
				    np.ndarray l2g,
				    np.ndarray a,
				    np.ndarray da,
				    np.ndarray grad_phi,
				    np.ndarray grad_w_dV,
				    np.ndarray dphi,
				    np.ndarray v,
				    np.ndarray grad_v,
				    np.ndarray jacobian_weak_residual):
    cdef int nElements_global = grad_w_dV.shape[0]
    cdef int nQuadraturePoints_element = grad_w_dV.shape[1]
    cdef int nDOF_trial_element = grad_v.shape[2]
    cdef int nDOF_test_element = grad_w_dV.shape[2]
    cdef int nSpace = grad_w_dV.shape[3]
    cupdateDiffusionJacobian_weak_sd(nElements_global,
				     nQuadraturePoints_element,
				     nDOF_trial_element,
				     nDOF_test_element,
				     nSpace,
				     <proteus_csr_index_t*> rowptr.data,
				     <proteus_csr_index_t*> colind.data,
				     <proteus_index_t*> l2g.data,
				     <double*> a.data,
				     <double*> da.data,
				     <double*> grad_phi.data,
				     <double*> grad_w_dV.data,
				     <double*> dphi.data,
				     <double*> v.data,
				     <double*> grad_v.data,
				     <double*> jacobian_weak_residual.data)
def updateDiffusion_strong(np.ndarray da,
                           np.ndarray grad_phi,
                           np.ndarray grad_u,
                           np.ndarray strong_residual):
    cdef int nElements_global = grad_u.shape[0]
    cdef int nQuadraturePoints_element = grad_u.shape[1]
    cdef int nSpace = grad_u.shape[2]
    cupdateDiffusion_strong(nElements_global,
                            nQuadraturePoints_element,
                            nSpace,
                            <double*>da.data,
                            <double*>grad_phi.data,
                            <double*>grad_u.data,
                            <double*>strong_residual.data)
def updateDiffusion_strong_sd(np.ndarray rowptr,
                              np.ndarray colind,
                              np.ndarray da,
                              np.ndarray grad_phi,
                              np.ndarray grad_u,
                              np.ndarray strong_residual):
    cdef int nElements_global = grad_u.shape[0]
    cdef int nQuadraturePoints_element = grad_u.shape[1]
    cdef int nSpace = grad_u.shape[2]
    cupdateDiffusion_strong_sd(nElements_global,
                               nQuadraturePoints_element,
                               nSpace,
                               <proteus_csr_index_t*> rowptr.data,
                               <proteus_csr_index_t*> colind.data,
                               <double*> da.data,
                               <double*> grad_phi.data,
                               <double*> grad_u.data,
                               <double*> strong_residual.data)
def updateDiffusionJacobian_strong(np.ndarray l2g,
                                   np.ndarray da,
                                   np.ndarray dphi,
                                   np.ndarray grad_phi,
                                   np.ndarray grad_u,
                                   np.ndarray grad_v,
                                   np.ndarray dstrong_residual):
    cdef int nElements_global = grad_v.shape[0]
    cdef int nQuadraturePoints_element = grad_v.shape[1]
    cdef int nDOF_trial_element = grad_v.shape[2]
    cdef int nSpace = grad_v.shape[3]
    cupdateDiffusionJacobian_strong(nElements_global,
                                    nQuadraturePoints_element,
                                    nDOF_trial_element,
                                    nSpace,
                                    <proteus_index_t*>l2g.data,
                                    <double*>da.data,
                                    <double*>dphi.data,
                                    <double*>grad_phi.data,
                                    <double*>grad_u.data,
                                    <double*>grad_v.data,
                                    <double*>dstrong_residual.data)
def updateDiffusionJacobian_strong_sd(np.ndarray rowptr,
				      np.ndarray colind,
				      np.ndarray l2g,
				      np.ndarray da,
				      np.ndarray dphi,
				      np.ndarray grad_phi,
				      np.ndarray grad_u,
				      np.ndarray grad_v,
				      np.ndarray dstrong_residual):
    cdef int nElements_global = grad_v.shape[0]
    cdef int nQuadraturePoints_element = grad_v.shape[1]
    cdef int nDOF_trial_element = grad_v.shape[2]
    cdef int nSpace = grad_v.shape[3]
    cupdateDiffusionJacobian_strong_sd(nElements_global,
				       nQuadraturePoints_element,
				       nDOF_trial_element,
				       nSpace,
				       <proteus_csr_index_t*> rowptr.data,
				       <proteus_csr_index_t*> colind.data,
				       <proteus_index_t*> l2g.data,
				       <double*> da.data,
				       <double*> dphi.data,
				       <double*> grad_phi.data,
				       <double*> grad_u.data,
				       <double*> grad_v.data,
				       <double*> dstrong_residual.data)
def updateDiffusion_adjoint(np.ndarray da,
                            np.ndarray grad_phi,
                            np.ndarray grad_w_dV,
                            np.ndarray Lstar_w_dV):
    cdef int nElements_global = grad_w_dV.shape[0]
    cdef int nQuadraturePoints_element = grad_w_dV.shape[1]
    cdef int nDOF_test_element = grad_w_dV.shape[2]
    cdef int nSpace = grad_w_dV.shape[3]
    cupdateDiffusion_adjoint(nElements_global,
                             nQuadraturePoints_element,
                             nDOF_test_element,
                             nSpace,
                             <double*>da.data,
                             <double*>grad_phi.data,
                             <double*>grad_w_dV.data,
                             <double*>Lstar_w_dV.data)
def updateDiffusion_adjoint_sd(np.ndarray rowptr,
			       np.ndarray colind,
			       np.ndarray da,
			       np.ndarray grad_phi,
			       np.ndarray grad_w_dV,
			       np.ndarray Lstar_w_dV):
    nElements_global = grad_w_dV.shape[0]
    nQuadraturePoints_element = grad_w_dV.shape[1]
    nDOF_test_element = grad_w_dV.shape[2]
    nSpace = grad_w_dV.shape[3]
    cupdateDiffusion_adjoint_sd(nElements_global,
				nQuadraturePoints_element,
				nDOF_test_element,
				nSpace,
				<proteus_csr_index_t*> rowptr.data,
				<proteus_csr_index_t*> colind.data,
				<double*> da.data,
				<double*> grad_phi.data,
				<double*> grad_w_dV.data,
				<double*> Lstar_w_dV.data)
def updateReaction_weak(np.ndarray r,
                        np.ndarray w_dV,
                        np.ndarray weak_residual):
    cdef int nElements_global = w_dV.shape[0]
    cdef int nQuadraturePoints_element = w_dV.shape[1]
    cdef int nDOF_test_element = w_dV.shape[2]
    cupdateReaction_weak(nElements_global,
                         nQuadraturePoints_element,
                         nDOF_test_element,
                         <double*>r.data,
                         <double*>w_dV.data,
                         <double*>weak_residual.data)
def updateReactionJacobian_weak(np.ndarray dr,
                                np.ndarray v_X_w_dV,
                                np.ndarray jacobian_weak_residual):
    cdef int nElements_global = v_X_w_dV.shape[0]
    cdef int nQuadraturePoints_element = v_X_w_dV.shape[1]
    cdef int nDOF_trial_element = v_X_w_dV.shape[2]
    cdef int nDOF_test_element = v_X_w_dV.shape[3]
    cupdateReactionJacobian_weak(nElements_global,
                                 nQuadraturePoints_element,
                                 nDOF_trial_element,
                                 nDOF_test_element,
                                 <double*>dr.data,
                                 <double*>v_X_w_dV.data,
                                 <double*>jacobian_weak_residual.data)
def updateReactionJacobian_weak_lowmem(np.ndarray dr,
                                       np.ndarray v,
                                       np.ndarray w_dV,
                                       np.ndarray jacobian_weak_residual):
    cdef int nElements_global = v.shape[0]
    cdef int nQuadraturePoints_element = v.shape[1]
    cdef int nDOF_trial_element = v.shape[2]
    cdef int nDOF_test_element = w_dV.shape[2]
    cupdateReactionJacobian_weak_lowmem(nElements_global,
                                        nQuadraturePoints_element,
                                        nDOF_trial_element,
                                        nDOF_test_element,
                                        <double*> dr.data,
                                        <double*> v.data,
                                        <double*> w_dV.data,
                                        <double*> jacobian_weak_residual.data)
def updateReaction_strong(np.ndarray r,
                          np.ndarray strong_residual):
    cdef int nElements_global = r.shape[0]
    cdef int nQuadraturePoints_element = r.shape[1]
    cupdateReaction_strong(nElements_global,
                           nQuadraturePoints_element,
                           <double*>r.data,
                           <double*>strong_residual.data)
def updateReactionJacobian_strong(np.ndarray dr,
                                  np.ndarray v,
                                  np.ndarray dstrong_residual):
    cdef int nElements_global = v.shape[0]
    cdef int nQuadraturePoints_element = v.shape[1]
    cdef int nDOF_trial_element = v.shape[2]
    cupdateReactionJacobian_strong(nElements_global,
                                   nQuadraturePoints_element,
                                   nDOF_trial_element,
                                   <double*>dr.data,
                                   <double*>v.data,
                                   <double*>dstrong_residual.data)
def updateReaction_adjoint(np.ndarray dr,
                           np.ndarray w_dV,
                           np.ndarray Lstar_w_dV):
    cdef int nElements_global = w_dV.shape[0]
    cdef int nQuadraturePoints_element = w_dV.shape[1]
    cdef int nDOF_test_element = w_dV.shape[2]
    cupdateReaction_adjoint(nElements_global,
                            nQuadraturePoints_element,
                            nDOF_test_element,
                            <double*>dr.data,
                            <double*>w_dV.data,
                            <double*>Lstar_w_dV.data)
def updateSubgridError(np.ndarray error,
                       np.ndarray Lstar_w_dV,
                       np.ndarray weak_residual):
    cdef int nElements_global = Lstar_w_dV.shape[0]
    cdef int nQuadraturePoints_element = Lstar_w_dV.shape[1]
    cdef int nDOF_test_element = Lstar_w_dV.shape[2]
    cupdateSubgridError(nElements_global,
                        nQuadraturePoints_element,
                        nDOF_test_element,
                        <double*>error.data,
                        <double*>Lstar_w_dV.data,
                        <double*>weak_residual.data)
def updateSubgridErrorJacobian(np.ndarray derror,
                               np.ndarray Lstar_w_dV,
                               np.ndarray jacobian_weak_residual):
    cdef int nElements_global = Lstar_w_dV.shape[0]
    cdef int nQuadraturePoints_element = Lstar_w_dV.shape[1]
    cdef int nDOF_trial_element = derror.shape[2]
    cdef int nDOF_test_element = Lstar_w_dV.shape[2]
    cupdateSubgridErrorJacobian(nElements_global,
                                nQuadraturePoints_element,
                                nDOF_trial_element,
                                nDOF_test_element,
                                <double*>derror.data,
                                <double*>Lstar_w_dV.data,
                                <double*>jacobian_weak_residual.data)
def updateNumericalDiffusion(np.ndarray numDiff,
                             np.ndarray grad_u_X_grad_w_dV,
                             np.ndarray weak_residual):
    cdef int nElements_global = grad_u_X_grad_w_dV.shape[0]
    cdef int nQuadraturePoints_element = grad_u_X_grad_w_dV.shape[1]
    cdef int nDOF_test_element = grad_u_X_grad_w_dV.shape[2]
    cdef int nSpace = grad_u_X_grad_w_dV.shape[3]
    cupdateNumericalDiffusion(nElements_global,
                              nQuadraturePoints_element,
                              nDOF_test_element,
                              nSpace,
                              <double*>numDiff.data,
                              <double*>grad_u_X_grad_w_dV.data,
                              <double*>weak_residual.data)
def updateNumericalDiffusion_lowmem(np.ndarray numDiff,
                                    np.ndarray grad_u,
                                    np.ndarray grad_w_dV,
                                    np.ndarray weak_residual):
    cdef int nElements_global = grad_w_dV.shape[0]
    cdef int nQuadraturePoints_element = grad_w_dV.shape[1]
    cdef int nDOF_test_element = grad_w_dV.shape[2]
    cdef int nSpace = grad_w_dV.shape[3]
    cupdateNumericalDiffusion_lowmem(nElements_global,
                                     nQuadraturePoints_element,
                                     nDOF_test_element,
                                     nSpace,
                                     <double*> numDiff.data,
                                     <double*> grad_u.data,
                                     <double*> grad_w_dV.data,
                                     <double*> weak_residual.data)
def updateNumericalDiffusionJacobian(np.ndarray numDiff,
                                     np.ndarray grad_v_X_grad_w_dV,
                                     np.ndarray jacobian_weak_residual):
    cdef int nElements_global = grad_v_X_grad_w_dV.shape[0]
    cdef int nQuadraturePoints_element = grad_v_X_grad_w_dV.shape[1]
    cdef int nDOF_trial_element = grad_v_X_grad_w_dV.shape[2]
    cdef int nDOF_test_element = grad_v_X_grad_w_dV.shape[3]
    cdef int nSpace = grad_v_X_grad_w_dV.shape[4]
    cupdateNumericalDiffusionJacobian(nElements_global,
                                      nQuadraturePoints_element,
                                      nDOF_trial_element,
                                      nDOF_test_element,
                                      nSpace,
                                      <double*>numDiff.data,
                                      <double*>grad_v_X_grad_w_dV.data,
                                      <double*>jacobian_weak_residual.data)
def updateNumericalDiffusionJacobian_lowmem(np.ndarray numDiff,
                                            np.ndarray grad_v,
                                            np.ndarray grad_w_dV,
                                            np.ndarray jacobian_weak_residual):
    nElements_global = grad_v.shape[0]
    nQuadraturePoints_element = grad_v.shape[1]
    nDOF_trial_element = grad_v.shape[2]
    nDOF_test_element = grad_w_dV.shape[2]
    nSpace = grad_w_dV.shape[3]
    cupdateNumericalDiffusionJacobian_lowmem(nElements_global,
                                             nQuadraturePoints_element,
                                             nDOF_trial_element,
                                             nDOF_test_element,
                                             nSpace,
                                             <double*> numDiff.data,
                                             <double*> grad_v.data,
                                             <double*> grad_w_dV.data,
                                             <double*> jacobian_weak_residual.data)
def calculateScalarScalarProduct(np.ndarray s1,
                                 np.ndarray s2,
                                 np.ndarray sResult):
    cdef int nElements_global = s1.shape[0]
    cdef int nQuadraturePoints_element = s1.shape[1]
    ccalculateScalarScalarProduct(nElements_global,
                                  nQuadraturePoints_element,
                                  <double*>s1.data,
                                  <double*>s2.data,
                                  <double*>sResult.data)
def calculateVectorScalarProduct(np.ndarray v,
                                 np.ndarray s,
                                 np.ndarray vResult):
    cdef int nElements_global = v.shape[0]
    cdef int nQuadraturePoints_element = v.shape[1]
    cdef int nSpace = v.shape[2]
    ccalculateVectorScalarProduct(nElements_global,
                                  nQuadraturePoints_element,
                                  nSpace,
                                  <double*>v.data,
                                  <double*>s.data,
                                  <double*>vResult.data)
def calculateTensorScalarProduct(np.ndarray t,
                                 np.ndarray s,
                                 np.ndarray tResult):
    cdef int nElements_global = t.shape[0]
    cdef int nQuadraturePoints_element = t.shape[1]
    cdef int nSpace = t.shape[2]
    ccalculateTensorScalarProduct(nElements_global,
                                  nQuadraturePoints_element,
                                  nSpace,
                                  <double*>t.data,
                                  <double*>s.data,
                                  <double*>tResult.data)
def updateInteriorElementBoundaryFlux(np.ndarray interiorElementBoundaries,
                                      np.ndarray elementBoundaryElements,
                                      np.ndarray elementBoundaryLocalElementBoundaries,
                                      np.ndarray flux,
                                      np.ndarray w_dS,
                                      np.ndarray residual):
    cdef int nInteriorElementBoundaries_global = interiorElementBoundaries.shape[0]
    cdef int nElementBoundaries_element = w_dS.shape[1]
    cdef int nQuadraturePoints_elementBoundary = w_dS.shape[2]
    cdef int nDOF_test_element = w_dS.shape[3]
    cupdateInteriorElementBoundaryFlux(nInteriorElementBoundaries_global,
                                       nElementBoundaries_element,
                                       nQuadraturePoints_elementBoundary,
                                       nDOF_test_element,
                                       <proteus_index_t*>interiorElementBoundaries.data,
                                       <proteus_index_t*>elementBoundaryElements.data,
                                       <int *>elementBoundaryLocalElementBoundaries.data,
                                       <double*>flux.data,
                                       <double*>w_dS.data,
                                       <double*>residual.data)
def updateExteriorElementBoundaryFlux(np.ndarray exteriorElementBoundaries,
                                      np.ndarray elementBoundaryElements,
                                      np.ndarray elementBoundaryLocalElementBoundaries,
                                      np.ndarray flux,
                                      np.ndarray w_dS,
                                      np.ndarray residual):
    cdef int nd = w_dS.ndim
    if nd >= 4:
        cupdateExteriorElementBoundaryFlux(exteriorElementBoundaries.shape[0],
                                           w_dS.shape[1],
                                           w_dS.shape[2],
                                           w_dS.shape[3],
                                           <proteus_index_t*>exteriorElementBoundaries.data,
                                           <proteus_index_t*>elementBoundaryElements.data,
                                           <int *>elementBoundaryLocalElementBoundaries.data,
                                           <double*>flux.data,
                                           <double*>w_dS.data,
                                           <double*>residual.data)
    else:
        cupdateGlobalExteriorElementBoundaryFlux(exteriorElementBoundaries.shape[0],
                                                 w_dS.shape[1],
                                                 w_dS.shape[2],
					         <proteus_index_t*> exteriorElementBoundaries.data,
					         <proteus_index_t*> elementBoundaryElements.data,
					         <int*> elementBoundaryLocalElementBoundaries.data,
					         <double*> flux.data,
					         <double*> w_dS.data,
					         <double*> residual.data)
def updateGlobalResidualFromElementResidual(proteus_index_t offset_r,
                                            proteus_index_t stride_r,
                                            np.ndarray nFreeDOF_element_r,
                                            np.ndarray freeLocal_r,
                                            np.ndarray freeGlobal_r,
                                            np.ndarray elementResidual,
                                            np.ndarray globalResidual):
    cdef int nElements_global = elementResidual.shape[0]
    cdef int nDOF_test_element = elementResidual.shape[1]
    cupdateGlobalResidualFromElementResidual(nElements_global,
                                             nDOF_test_element,
                                             offset_r,
                                             stride_r,
                                             <int *>nFreeDOF_element_r.data,
                                             <int *>freeLocal_r.data,
                                             <proteus_index_t*>freeGlobal_r.data,
                                             <double*>elementResidual.data,
                                             <double*>globalResidual.data)
def updateGlobalJacobianFromElementJacobian_dense(proteus_index_t offset_r,
                                                  proteus_index_t stride_r,
                                                  proteus_index_t offset_u,
                                                  proteus_index_t stride_u,
                                                  proteus_index_t nFreeVDOF_global,
                                                  np.ndarray nFreeDOF_element_r,
                                                  np.ndarray freeLocal_r,
                                                  np.ndarray freeGlobal_r,
                                                  np.ndarray nFreeDOF_element_u,
                                                  np.ndarray freeLocal_u,
                                                  np.ndarray freeGlobal_u,
                                                  np.ndarray elementJacobian,
                                                  np.ndarray globalJacobian):
    cdef int nElements_global = elementJacobian.shape[0]
    cdef int nDOF_test_element = elementJacobian.shape[1]
    cdef int nDOF_trial_element = elementJacobian.shape[2]
    cupdateGlobalJacobianFromElementJacobian_dense(nElements_global,
                                                   nDOF_test_element,
                                                   nDOF_trial_element,
                                                   offset_r,
                                                   stride_r,
                                                   offset_u,
                                                   stride_u,
                                                   nFreeVDOF_global,
                                                   <int *>nFreeDOF_element_r.data,
                                                   <int *>freeLocal_r.data,
                                                   <proteus_index_t*>freeGlobal_r.data,
                                                   <int *>nFreeDOF_element_u.data,
                                                   <int *>freeLocal_u.data,
                                                   <proteus_index_t*>freeGlobal_u.data,
                                                   <double*>elementJacobian.data,
                                                   <double*>globalJacobian.data)
def updateGlobalJacobianFromElementJacobian_eb_dense(np.ndarray elementNeighbors,
                                                     proteus_index_t offset_r,
                                                     proteus_index_t stride_r,
                                                     proteus_index_t offset_u,
                                                     proteus_index_t stride_u,
                                                     proteus_index_t nFreeVDOF_global,
                                                     np.ndarray nFreeDOF_element_r,
                                                     np.ndarray freeLocal_r,
                                                     np.ndarray freeGlobal_r,
                                                     np.ndarray nFreeDOF_element_u,
                                                     np.ndarray freeLocal_u,
                                                     np.ndarray freeGlobal_u,
                                                     np.ndarray elementJacobian_eb,
                                                     np.ndarray globalJacobian):
    cdef int nElements_global = elementJacobian_eb.shape[0]
    cdef int nElementBoundaries_element = elementJacobian_eb.shape[1]
    cdef int nDOF_test_element = elementJacobian_eb.shape[2]
    cdef int nDOF_trial_element = elementJacobian_eb.shape[3]
    cupdateGlobalJacobianFromElementJacobian_eb_dense(<proteus_index_t*>elementNeighbors.data,
                                                      nElements_global,
                                                      nElementBoundaries_element,
                                                      nDOF_test_element,
                                                      nDOF_trial_element,
                                                      offset_r,
                                                      stride_r,
                                                      offset_u,
                                                      stride_u,
                                                      nFreeVDOF_global,
                                                      <int *>nFreeDOF_element_r.data,
                                                      <int *>freeLocal_r.data,
                                                      <proteus_index_t*>freeGlobal_r.data,
                                                      <int *>nFreeDOF_element_u.data,
                                                      <int *>freeLocal_u.data,
                                                      <proteus_index_t*>freeGlobal_u.data,
                                                      <double*>elementJacobian_eb.data,
                                                      <double*>globalJacobian.data)
def updateGlobalJacobianFromInteriorElementBoundaryFluxJacobian_dense(proteus_index_t offset_r,
                                                                      proteus_index_t stride_r,
                                                                      proteus_index_t offset_u,
                                                                      proteus_index_t stride_u,
                                                                      proteus_index_t nFreeVDOF_global,
                                                                      np.ndarray interiorElementBoundaries,
                                                                      np.ndarray elementBoundaryElements,
                                                                      np.ndarray elementBoundaryLocalElementBoundaries,
                                                                      np.ndarray nFreeDOF_element_r,
                                                                      np.ndarray freeLocal_r,
                                                                      np.ndarray freeGlobal_r,
                                                                      np.ndarray nFreeDOF_element_u,
                                                                      np.ndarray freeLocal_u,
                                                                      np.ndarray freeGlobal_u,
                                                                      np.ndarray elementBoundaryFluxJacobian,
                                                                      np.ndarray w_dS,
                                                                      np.ndarray jac):
    cdef int nInteriorElementBoundaries_global = interiorElementBoundaries.shape[0]
    cdef int nElementBoundaries_element = w_dS.shape[1]
    cdef int nQuadraturePoints_elementBoundary = w_dS.shape[2]
    cdef int nDOF_test_element = w_dS.shape[3]
    cdef int nDOF_trial_element = elementBoundaryFluxJacobian.shape[3]
    cupdateGlobalJacobianFromInteriorElementBoundaryFluxJacobian_dense(nInteriorElementBoundaries_global,
                                                                       nElementBoundaries_element,
                                                                       nQuadraturePoints_elementBoundary,
                                                                       nDOF_test_element,
                                                                       nDOF_trial_element,
                                                                       offset_r,
                                                                       stride_r,
                                                                       offset_u,
                                                                       stride_u,
                                                                       nFreeVDOF_global,
                                                                       <proteus_index_t*>interiorElementBoundaries.data,
                                                                       <proteus_index_t*>elementBoundaryElements.data,
                                                                       <int *>elementBoundaryLocalElementBoundaries.data,
                                                                       <int *>nFreeDOF_element_r.data,
                                                                       <int *>freeLocal_r.data,
                                                                       <proteus_index_t*>freeGlobal_r.data,
                                                                       <int *>nFreeDOF_element_u.data,
                                                                       <int *>freeLocal_u.data,
                                                                       <proteus_index_t*>freeGlobal_u.data,
                                                                       <double*>elementBoundaryFluxJacobian.data,
                                                                       <double*>w_dS.data,
                                                                       <double*>jac.data)
def updateGlobalJacobianFromInteriorElementBoundaryFluxJacobian_eb_dense(np.ndarray elementNeighbors,
                                                                         proteus_index_t nElements_global,
                                                                         proteus_index_t offset_r,
                                                                         proteus_index_t stride_r,
                                                                         proteus_index_t offset_u,
                                                                         proteus_index_t stride_u,
                                                                         proteus_index_t nFreeVDOF_global,
                                                                         np.ndarray interiorElementBoundaries,
                                                                         np.ndarray elementBoundaryElements,
                                                                         np.ndarray elementBoundaryLocalElementBoundaries,
                                                                         np.ndarray nFreeDOF_element_r,
                                                                         np.ndarray freeLocal_r,
                                                                         np.ndarray freeGlobal_r,
                                                                         np.ndarray nFreeDOF_element_u,
                                                                         np.ndarray freeLocal_u,
                                                                         np.ndarray freeGlobal_u,
                                                                         np.ndarray elementBoundaryFluxJacobian_eb,
                                                                         np.ndarray w_dS,
                                                                         np.ndarray jac):
    cdef int nInteriorElementBoundaries_global = interiorElementBoundaries.shape[0]
    cdef int nElementBoundaries_element = w_dS.shape[1]
    cdef int nQuadraturePoints_elementBoundary = w_dS.shape[2]
    cdef int nDOF_test_element = w_dS.shape[3]
    cdef int nDOF_trial_element = elementBoundaryFluxJacobian_eb.shape[4]
    cupdateGlobalJacobianFromInteriorElementBoundaryFluxJacobian_eb_dense(<proteus_index_t*>elementNeighbors.data,
                                                                          nElements_global,
                                                                          nInteriorElementBoundaries_global,
                                                                          nElementBoundaries_element,
                                                                          nQuadraturePoints_elementBoundary,
                                                                          nDOF_test_element,
                                                                          nDOF_trial_element,
                                                                          offset_r,
                                                                          stride_r,
                                                                          offset_u,
                                                                          stride_u,
                                                                          nFreeVDOF_global,
                                                                          <proteus_index_t*>interiorElementBoundaries.data,
                                                                          <proteus_index_t*>elementBoundaryElements.data,
                                                                          <int *>elementBoundaryLocalElementBoundaries.data,
                                                                          <int *>nFreeDOF_element_r.data,
                                                                          <int *>freeLocal_r.data,
                                                                          <proteus_index_t*>freeGlobal_r.data,
                                                                          <int *>nFreeDOF_element_u.data,
                                                                          <int *>freeLocal_u.data,
                                                                          <proteus_index_t*>freeGlobal_u.data,
                                                                          <double*>elementBoundaryFluxJacobian_eb.data,
                                                                          <double*>w_dS.data,
                                                                          <double*>jac.data)
def updateGlobalJacobianFromExteriorElementBoundaryFluxJacobian_dense(proteus_index_t offset_r,
                                                                      proteus_index_t stride_r,
                                                                      proteus_index_t offset_u,
                                                                      proteus_index_t stride_u,
                                                                      proteus_index_t nFreeVDOF_global,
                                                                      np.ndarray exteriorElementBoundaries,
                                                                      np.ndarray elementBoundaryElements,
                                                                      np.ndarray elementBoundaryLocalElementBoundaries,
                                                                      np.ndarray nFreeDOF_element_r,
                                                                      np.ndarray freeLocal_r,
                                                                      np.ndarray freeGlobal_r,
                                                                      np.ndarray nFreeDOF_element_u,
                                                                      np.ndarray freeLocal_u,
                                                                      np.ndarray freeGlobal_u,
                                                                      np.ndarray elementBoundaryFluxJacobian,
                                                                      np.ndarray w_dS,
                                                                      np.ndarray jac):
    cdef int nd = w_dS.ndim
    if nd > 3:
        assert elementBoundaryFluxJacobian.ndim == 4
        cupdateGlobalJacobianFromExteriorElementBoundaryFluxJacobian_dense(exteriorElementBoundaries.shape[0],
                                                                           w_dS.shape[1],
                                                                           w_dS.shape[2],
                                                                           w_dS.shape[3],
                                                                           elementBoundaryFluxJacobian.shape[3],
                                                                           offset_r,
                                                                           stride_r,
                                                                           offset_u,
                                                                           stride_u,
                                                                           nFreeVDOF_global,
                                                                           <proteus_index_t*>exteriorElementBoundaries.data,
                                                                           <proteus_index_t*>elementBoundaryElements.data,
                                                                           <int *>elementBoundaryLocalElementBoundaries.data,
                                                                           <int *>nFreeDOF_element_r.data,
                                                                           <int *>freeLocal_r.data,
                                                                           <proteus_index_t*>freeGlobal_r.data,
                                                                           <int *>nFreeDOF_element_u.data,
                                                                           <int *>freeLocal_u.data,
                                                                           <proteus_index_t*>freeGlobal_u.data,
                                                                           <double*>elementBoundaryFluxJacobian.data,
                                                                           <double*>w_dS.data,
                                                                           <double*>jac.data)
    else:
        assert elementBoundaryFluxJacobian.ndim == 3
        cupdateGlobalJacobianFromGlobalExteriorElementBoundaryFluxJacobian_dense(exteriorElementBoundaries.shape[0],
                                                                                 w_dS.shape[1],
                                                                                 w_dS.shape[2],
                                                                                 elementBoundaryFluxJacobian.shape[2],
									         offset_r,
									         stride_r,
									         offset_u,
									         stride_u,
									         nFreeVDOF_global,
									         <proteus_index_t*> exteriorElementBoundaries.data,
									         <proteus_index_t*> elementBoundaryElements.data,
									         <int*> elementBoundaryLocalElementBoundaries.data,
									         <int*> nFreeDOF_element_r.data,
									         <int*> freeLocal_r.data,
									         <proteus_index_t*> freeGlobal_r.data,
									         <int*> nFreeDOF_element_u.data,
									         <int*> freeLocal_u.data,
									         <proteus_index_t*> freeGlobal_u.data,
									         <double*> elementBoundaryFluxJacobian.data,
									         <double*> w_dS.data,
									         <double*> jac.data)
def updateGlobalJacobianFromExteriorElementBoundaryFluxJacobian_eb_dense(np.ndarray elementNeighbors,
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
                                                                         np.ndarray exteriorElementBoundaries,
                                                                         np.ndarray elementBoundaryElements,
                                                                         np.ndarray elementBoundaryLocalElementBoundaries,
                                                                         np.ndarray nFreeDOF_element_r,
                                                                         np.ndarray freeLocal_r,
                                                                         np.ndarray freeGlobal_r,
                                                                         np.ndarray nFreeDOF_element_u,
                                                                         np.ndarray freeLocal_u,
                                                                         np.ndarray freeGlobal_u,
                                                                         np.ndarray elementBoundaryFluxJacobian_eb,
                                                                         np.ndarray w_dS,
                                                                         np.ndarray jac):
    if w_dS.ndim > 3:
        cupdateGlobalJacobianFromExteriorElementBoundaryFluxJacobian_eb_dense(<proteus_index_t*>elementNeighbors.data,
                                                                              nElements_global,
                                                                              exteriorElementBoundaries.shape[0],
                                                                              w_dS.shape[1],
                                                                              w_dS.shape[2],
                                                                              w_dS.shape[3],
                                                                              elementBoundaryFluxJacobian_eb.shape[4],
                                                                              offset_r,
                                                                              stride_r,
                                                                              offset_u,
                                                                              stride_u,
                                                                              nFreeVDOF_global,
                                                                              <proteus_index_t*>exteriorElementBoundaries.data,
                                                                              <proteus_index_t*>elementBoundaryElements.data,
                                                                              <int *>elementBoundaryLocalElementBoundaries.data,
                                                                              <int *>nFreeDOF_element_r.data,
                                                                              <int *>freeLocal_r.data,
                                                                              <proteus_index_t*>freeGlobal_r.data,
                                                                              <int *>nFreeDOF_element_u.data,
                                                                              <int *>freeLocal_u.data,
                                                                              <proteus_index_t*>freeGlobal_u.data,
                                                                              <double*>elementBoundaryFluxJacobian_eb.data,
                                                                              <double*>w_dS.data,
                                                                              <double*>jac.data)
    else:
        cupdateGlobalJacobianFromGlobalExteriorElementBoundaryFluxJacobian_eb_dense(<proteus_index_t*> elementNeighbors.data,
									            nElements_global,
									            exteriorElementBoundaries.shape[0],
                                                                                    elementBoundaryFluxJacobian_eb.shape[2],
                                                                                    w_dS.shape[1],
                                                                                    w_dS.shape[2],
                                                                                    elementBoundaryFluxJacobian_eb.shape[4],
									            offset_r,
									            stride_r,
									            offset_u,
									            stride_u,
									            nFreeVDOF_global,
									            <proteus_index_t*> exteriorElementBoundaries.data,
									            <proteus_index_t*> elementBoundaryElements.data,
									            <int*> elementBoundaryLocalElementBoundaries.data,
									            <int*> nFreeDOF_element_r.data,
									            <int*> freeLocal_r.data,
									            <proteus_index_t*> freeGlobal_r.data,
									            <int*> nFreeDOF_element_u.data,
									            <int*> freeLocal_u.data,
									            <proteus_index_t*> freeGlobal_u.data,
									            <double*> elementBoundaryFluxJacobian_eb.data,
									            <double*> w_dS.data,
									            <double*> jac.data)
def updateGlobalJacobianFromElementJacobian_CSR(np.ndarray nFreeDOF_element_r,
                                                np.ndarray freeLocal_r,
                                                np.ndarray nFreeDOF_element_u,
                                                np.ndarray freeLocal_u,
                                                np.ndarray csrRowIndeces_ru,
                                                np.ndarray csrColumnOffsets_ru,
                                                np.ndarray elementJacobian,
                                                globalJacobian):
    cdef np.ndarray rowptr, colind, globalJacobian_array
    (rowptr,colind,globalJacobian_array) = globalJacobian.getCSRrepresentation()
    cdef int nElements_global = elementJacobian.shape[0]
    cdef int nDOF_test_element = elementJacobian.shape[1]
    cdef int nDOF_trial_element = elementJacobian.shape[2]
    cupdateGlobalJacobianFromElementJacobian_CSR(nElements_global,
                                                 nDOF_test_element,
                                                 nDOF_trial_element,
                                                 <int *>nFreeDOF_element_r.data,
                                                 <int *>freeLocal_r.data,
                                                 <int *>nFreeDOF_element_u.data,
                                                 <int *>freeLocal_u.data,
                                                 <proteus_csr_index_t*>csrRowIndeces_ru.data,
                                                 <proteus_csr_index_t*>csrColumnOffsets_ru.data,
                                                 <double*>elementJacobian.data,
                                                 <double*>globalJacobian_array.data)
def updateGlobalJacobianFromElementJacobian_eb_CSR(np.ndarray elementNeighbors,
                                                   np.ndarray nFreeDOF_element_r,
                                                   np.ndarray freeLocal_r,
                                                   np.ndarray nFreeDOF_element_u,
                                                   np.ndarray freeLocal_u,
                                                   np.ndarray csrRowIndeces_ru,
                                                   np.ndarray csrColumnOffsets_eb_ru,
                                                   np.ndarray elementJacobian_eb,
                                                   globalJacobian):
    cdef np.ndarray rowptr, colind, globalJacobian_array
    (rowptr,colind,globalJacobian_array) = globalJacobian.getCSRrepresentation()
    cdef int nElements_global = elementJacobian_eb.shape[0]
    cdef int nElementBoundaries_element = elementJacobian_eb.shape[1]
    cdef int nDOF_test_element = elementJacobian_eb.shape[2]
    cdef int nDOF_trial_element = elementJacobian_eb.shape[3]
    cupdateGlobalJacobianFromElementJacobian_eb_CSR(<proteus_index_t*>elementNeighbors.data,
                                                    nElements_global,
                                                    nElementBoundaries_element,
                                                    nDOF_test_element,
                                                    nDOF_trial_element,
                                                    <int *>nFreeDOF_element_r.data,
                                                    <int *>freeLocal_r.data,
                                                    <int *>nFreeDOF_element_u.data,
                                                    <int *>freeLocal_u.data,
                                                    <proteus_csr_index_t*>csrRowIndeces_ru.data,
                                                    <proteus_csr_index_t*>csrColumnOffsets_eb_ru.data,
                                                    <double*>elementJacobian_eb.data,
                                                    <double*>globalJacobian_array.data)
def updateGlobalJacobianFromInteriorElementBoundaryFluxJacobian_CSR(np.ndarray interiorElementBoundaries,
                                                                    np.ndarray elementBoundaryElements,
                                                                    np.ndarray elementBoundaryLocalElementBoundaries,
                                                                    np.ndarray nFreeDOF_element_r,
                                                                    np.ndarray freeLocal_r,
                                                                    np.ndarray nFreeDOF_element_u,
                                                                    np.ndarray freeLocal_u,
                                                                    np.ndarray csrRowIndeces_ru,
                                                                    np.ndarray csrColumnOffsets_eb_ru,
                                                                    np.ndarray elementBoundaryFluxJacobian,
                                                                    np.ndarray w_dS,
                                                                    jac):
    cdef np.ndarray rowptr, colind, jac_array
    (rowptr,colind,jac_array) = jac.getCSRrepresentation()
    cdef int nInteriorElementBoundaries_global = interiorElementBoundaries.shape[0]
    cdef int nElementBoundaries_element = w_dS.shape[1]
    cdef int nQuadraturePoints_elementBoundary = w_dS.shape[2]
    cdef int nDOF_test_element = w_dS.shape[3]
    cdef int nDOF_trial_element = elementBoundaryFluxJacobian.shape[3]
    cupdateGlobalJacobianFromInteriorElementBoundaryFluxJacobian_CSR(nInteriorElementBoundaries_global,
                                                                     nElementBoundaries_element,
                                                                     nQuadraturePoints_elementBoundary,
                                                                     nDOF_test_element,
                                                                     nDOF_trial_element,
                                                                     <proteus_index_t*>interiorElementBoundaries.data,
                                                                     <proteus_index_t*>elementBoundaryElements.data,
                                                                     <int *>elementBoundaryLocalElementBoundaries.data,
                                                                     <int *>nFreeDOF_element_r.data,
                                                                     <int *>freeLocal_r.data,
                                                                     <int *>nFreeDOF_element_u.data,
                                                                     <int *>freeLocal_u.data,
                                                                     <proteus_csr_index_t*>csrRowIndeces_ru.data,
                                                                     <proteus_csr_index_t*>csrColumnOffsets_eb_ru.data,
                                                                     <double*>elementBoundaryFluxJacobian.data,
                                                                     <double*>w_dS.data,
                                                                     <double*>jac_array.data)
def updateGlobalJacobianFromExteriorElementBoundaryFluxJacobian_CSR(np.ndarray exteriorElementBoundaries,
                                                                    np.ndarray elementBoundaryElements,
                                                                    np.ndarray elementBoundaryLocalElementBoundaries,
                                                                    np.ndarray nFreeDOF_element_r,
                                                                    np.ndarray freeLocal_r,
                                                                    np.ndarray nFreeDOF_element_u,
                                                                    np.ndarray freeLocal_u,
                                                                    np.ndarray csrRowIndeces_ru,
                                                                    np.ndarray csrColumnOffsets_eb_ru,
                                                                    np.ndarray elementBoundaryFluxJacobian,
                                                                    np.ndarray w_dS,
                                                                    jac):
    cdef np.ndarray rowptr, colind, jac_array
    (rowptr,colind,jac_array) = jac.getCSRrepresentation()
    cdef int nd = w_dS.ndim
    if nd > 3:
        assert elementBoundaryFluxJacobian.ndim == 4
        cupdateGlobalJacobianFromExteriorElementBoundaryFluxJacobian_CSR(exteriorElementBoundaries.shape[0],
                                                                         w_dS.shape[1],
                                                                         w_dS.shape[2],
                                                                         w_dS.shape[3],
                                                                         elementBoundaryFluxJacobian.shape[3],
                                                                         <proteus_index_t*>exteriorElementBoundaries.data,
                                                                         <proteus_index_t*>elementBoundaryElements.data,
                                                                         <int *>elementBoundaryLocalElementBoundaries.data,
                                                                         <int *>nFreeDOF_element_r.data,
                                                                         <int *>freeLocal_r.data,
                                                                         <int *>nFreeDOF_element_u.data,
                                                                         <int *>freeLocal_u.data,
                                                                         <proteus_csr_index_t*>csrRowIndeces_ru.data,
                                                                         <proteus_csr_index_t*>csrColumnOffsets_eb_ru.data,
                                                                         <double*>elementBoundaryFluxJacobian.data,
                                                                         <double*>w_dS.data,
                                                                         <double*>jac_array.data)
    else:
        assert elementBoundaryFluxJacobian.ndim == 3
        cupdateGlobalJacobianFromGlobalExteriorElementBoundaryFluxJacobian_CSR(exteriorElementBoundaries.shape[0],
                                                                               w_dS.shape[1],
                                                                               w_dS.shape[2],
                                                                               elementBoundaryFluxJacobian.shape[2],
									       <proteus_index_t*> exteriorElementBoundaries.data,
									       <proteus_index_t*> elementBoundaryElements.data,
									       <int*> elementBoundaryLocalElementBoundaries.data,
									       <int*> nFreeDOF_element_r.data,
									       <int*> freeLocal_r.data,
									       <int*> nFreeDOF_element_u.data,
									       <int*> freeLocal_u.data,
									       <proteus_csr_index_t*> csrRowIndeces_ru.data,
									       <proteus_csr_index_t*> csrColumnOffsets_eb_ru.data,
									       <double*> elementBoundaryFluxJacobian.data,
									       <double*> w_dS.data,
									       <double*> jac_array.data)
def updateGlobalJacobianFromInteriorElementBoundaryFluxJacobian_eb_CSR(np.ndarray elementNeighbors,
                                                                       np.ndarray interiorElementBoundaries,
                                                                       np.ndarray elementBoundaryElements,
                                                                       np.ndarray elementBoundaryLocalElementBoundaries,
                                                                       np.ndarray nFreeDOF_element_r,
                                                                       np.ndarray freeLocal_r,
                                                                       np.ndarray nFreeDOF_element_u,
                                                                       np.ndarray freeLocal_u,
                                                                       np.ndarray csrRowIndeces_ru,
                                                                       np.ndarray csrColumnOffsets_eb_eNebN_ru,
                                                                       np.ndarray elementBoundaryFluxJacobian_eb,
                                                                       np.ndarray w_dS,
                                                                       jac):
    cdef np.ndarray rowptr, colind, jac_array
    (rowptr,colind,jac_array) = jac.getCSRrepresentation()
    cdef int nInteriorElementBoundaries_global = interiorElementBoundaries.shape[0]
    cdef int nElementBoundaries_element = w_dS.shape[1]
    cdef int nQuadraturePoints_elementBoundary = w_dS.shape[2]
    cdef int nDOF_test_element = w_dS.shape[3]
    cdef int nDOF_trial_element = elementBoundaryFluxJacobian_eb.shape[4]
    cupdateGlobalJacobianFromInteriorElementBoundaryFluxJacobian_eb_CSR(<proteus_index_t*>elementNeighbors.data,
                                                                        nInteriorElementBoundaries_global,
                                                                        nElementBoundaries_element,
                                                                        nQuadraturePoints_elementBoundary,
                                                                        nDOF_test_element,
                                                                        nDOF_trial_element,
                                                                        <proteus_index_t*>interiorElementBoundaries.data,
                                                                        <proteus_index_t*>elementBoundaryElements.data,
                                                                        <int *>elementBoundaryLocalElementBoundaries.data,
                                                                        <int *>nFreeDOF_element_r.data,
                                                                        <int *>freeLocal_r.data,
                                                                        <int *>nFreeDOF_element_u.data,
                                                                        <int *>freeLocal_u.data,
                                                                        <proteus_csr_index_t*>csrRowIndeces_ru.data,
                                                                        <proteus_csr_index_t*>csrColumnOffsets_eb_eNebN_ru.data,
                                                                        <double*>elementBoundaryFluxJacobian_eb.data,
                                                                        <double*>w_dS.data,
                                                                        <double*>jac_array.data)
def updateGlobalJacobianFromExteriorElementBoundaryFluxJacobian_eb_CSR(np.ndarray elementNeighbors,
                                                                       np.ndarray exteriorElementBoundaries,
                                                                       np.ndarray elementBoundaryElements,
                                                                       np.ndarray elementBoundaryLocalElementBoundaries,
                                                                       np.ndarray nFreeDOF_element_r,
                                                                       np.ndarray freeLocal_r,
                                                                       np.ndarray nFreeDOF_element_u,
                                                                       np.ndarray freeLocal_u,
                                                                       np.ndarray csrRowIndeces_ru,
                                                                       np.ndarray csrColumnOffsets_eb_eNebN_ru,
                                                                       np.ndarray elementBoundaryFluxJacobian_eb,
                                                                       np.ndarray w_dS,
                                                                       jac):
    cdef np.ndarray rowptr, colind, jac_array
    (rowptr,colind,jac_array) = jac.getCSRrepresentation()
    if w_dS.ndim > 3:
        cupdateGlobalJacobianFromExteriorElementBoundaryFluxJacobian_eb_CSR(<proteus_index_t*>elementNeighbors.data,
                                                                            exteriorElementBoundaries.shape[0],
                                                                            w_dS.shape[1],
                                                                            w_dS.shape[2],
                                                                            w_dS.shape[3],
                                                                            elementBoundaryFluxJacobian_eb.shape[4],
                                                                            <proteus_index_t*>exteriorElementBoundaries.data,
                                                                            <proteus_index_t*>elementBoundaryElements.data,
                                                                            <int *>elementBoundaryLocalElementBoundaries.data,
                                                                            <int *>nFreeDOF_element_r.data,
                                                                            <int *>freeLocal_r.data,
                                                                            <int *>nFreeDOF_element_u.data,
                                                                            <int *>freeLocal_u.data,
                                                                            <proteus_csr_index_t*>csrRowIndeces_ru.data,
                                                                            <proteus_csr_index_t*>csrColumnOffsets_eb_eNebN_ru.data,
                                                                            <double*>elementBoundaryFluxJacobian_eb.data,
                                                                            <double*>w_dS.data,
                                                                            <double*>jac_array.data)
    else:
        cupdateGlobalJacobianFromGlobalExteriorElementBoundaryFluxJacobian_eb_CSR(<proteus_index_t*> elementNeighbors.data,
                                                                                  exteriorElementBoundaries.shape[0],
                                                                                  elementBoundaryFluxJacobian_eb.shape[2],
                                                                                  w_dS.shape[1],
                                                                                  w_dS.shape[2],
                                                                                  elementBoundaryFluxJacobian_eb.shape[4],
									          <proteus_index_t*> exteriorElementBoundaries.data,
									          <proteus_index_t*> elementBoundaryElements.data,
									          <int*> elementBoundaryLocalElementBoundaries.data,
									          <int*> nFreeDOF_element_r.data,
									          <int*> freeLocal_r.data,
									          <int*> nFreeDOF_element_u.data,
									          <int*> freeLocal_u.data,
									          <proteus_csr_index_t*> csrRowIndeces_ru.data,
									          <proteus_csr_index_t*> csrColumnOffsets_eb_eNebN_ru.data,
									          <double*> elementBoundaryFluxJacobian_eb.data,
									          <double*> w_dS.data,
									          <double*> jac_array.data)
def calculateWeightedShape(np.ndarray dVR,
                           np.ndarray abs_det_jac,
                           np.ndarray w,
                           np.ndarray w_dV):
    cdef int nElements_global = w_dV.shape[0]
    cdef int nQuadraturePoints_element = w_dV.shape[1]
    cdef int nDOF_test_element = w_dV.shape[2]
    ccalculateWeightedShape(nElements_global,
                            nQuadraturePoints_element,
                            nDOF_test_element,
                            <double*>dVR.data,
                            <double*>abs_det_jac.data,
                            <double*>w.data,
                            <double*>w_dV.data)
def calculateWeightedShapeGradients(np.ndarray dVR,
                                    np.ndarray abs_det_jac,
                                    np.ndarray grad_w,
                                    np.ndarray grad_w_dV):
    cdef int nElements_global = grad_w_dV.shape[0]
    cdef int nQuadraturePoints_element = grad_w_dV.shape[1]
    cdef int nDOF_test_element = grad_w_dV.shape[2]
    cdef int nSpace = grad_w_dV.shape[3]
    ccalculateWeightedShapeGradients(nElements_global,
                                     nQuadraturePoints_element,
                                     nDOF_test_element,
                                     nSpace,
                                     <double*>dVR.data,
                                     <double*>abs_det_jac.data,
                                     <double*>grad_w.data,
                                     <double*>grad_w_dV.data)
def calculateShape_X_weightedShape(np.ndarray v,
                                   np.ndarray w_dV,
                                   np.ndarray v_X_w_dV):
    cdef int nElements_global = v_X_w_dV.shape[0]
    cdef int nQuadraturePoints_element = v_X_w_dV.shape[1]
    cdef int nDOF_trial_element = v_X_w_dV.shape[2]
    cdef int nDOF_test_element = v_X_w_dV.shape[3]
    ccalculateShape_X_weightedShape(nElements_global,
                                    nQuadraturePoints_element,
                                    nDOF_trial_element,
                                    nDOF_test_element,
                                    <double*>v.data,
                                    <double*>w_dV.data,
                                    <double*>v_X_w_dV.data)
def calculateShape_X_weightedGradShape(np.ndarray v,
                                       np.ndarray grad_w_dV,
                                       np.ndarray v_X_grad_w_dV):
    cdef int nElements_global = v_X_grad_w_dV.shape[0]
    cdef int nQuadraturePoints_element = v_X_grad_w_dV.shape[1]
    cdef int nDOF_trial_element = v_X_grad_w_dV.shape[2]
    cdef int nDOF_test_element = v_X_grad_w_dV.shape[3]
    cdef int nSpace = v_X_grad_w_dV.shape[4]
    ccalculateShape_X_weightedGradShape(nElements_global,
                                        nQuadraturePoints_element,
                                        nDOF_trial_element,
                                        nDOF_test_element,
                                        nSpace,
                                        <double*>v.data,
                                        <double*>grad_w_dV.data,
                                        <double*>v_X_grad_w_dV.data)
def calculateGradShape_X_weightedShape(np.ndarray grad_v,
                                       np.ndarray w_dV,
                                       np.ndarray grad_v_X_w_dV):
    cdef int nElements_global = grad_v_X_w_dV.shape[0]
    cdef int nQuadraturePoints_element = grad_v_X_w_dV.shape[1]
    cdef int nDOF_trial_element = grad_v_X_w_dV.shape[2]
    cdef int nDOF_test_element = grad_v_X_w_dV.shape[3]
    cdef int nSpace = grad_v_X_w_dV.shape[4]
    ccalculateGradShape_X_weightedShape(nElements_global,
                                        nQuadraturePoints_element,
                                        nDOF_trial_element,
                                        nDOF_test_element,
                                        nSpace,
                                        <double*>grad_v.data,
                                        <double*>w_dV.data,
                                        <double*>grad_v_X_w_dV.data)
def calculateGradShape_X_weightedGradShape(np.ndarray grad_v,
                                           np.ndarray grad_w_dV,
                                           np.ndarray grad_v_X_grad_w_dV):
    cdef int nElements_global = grad_v_X_grad_w_dV.shape[0]
    cdef int nQuadraturePoints_element = grad_v_X_grad_w_dV.shape[1]
    cdef int nDOF_trial_element = grad_v_X_grad_w_dV.shape[2]
    cdef int nDOF_test_element = grad_v_X_grad_w_dV.shape[3]
    cdef int nSpace = grad_v_X_grad_w_dV.shape[4]
    ccalculateGradShape_X_weightedGradShape(nElements_global,
                                            nQuadraturePoints_element,
                                            nDOF_trial_element,
                                            nDOF_test_element,
                                            nSpace,
                                            <double*>grad_v.data,
                                            <double*>grad_w_dV.data,
                                            <double*>grad_v_X_grad_w_dV.data)
def calculateWeightedShapeTrace(np.ndarray dSR,
                                np.ndarray sqrt_det_g,
                                np.ndarray w,
                                np.ndarray w_dS):
    cdef int nElements_global = w_dS.shape[0]
    cdef int nElementBoundaries_element = w_dS.shape[1]
    cdef int nElementBoundaryQuadraturePoints_elementBoundary = w_dS.shape[2]
    cdef int nDOF_test_element = w_dS.shape[3]
    ccalculateWeightedShapeTrace(nElements_global,
                                 nElementBoundaries_element,
                                 nElementBoundaryQuadraturePoints_elementBoundary,
                                 nDOF_test_element,
                                 <double*>dSR.data,
                                 <double*>sqrt_det_g.data,
                                 <double*>w.data,
                                 <double*>w_dS.data)
def calculateShape_X_weightedShapeTrace(np.ndarray v,
                                        np.ndarray w_dS,
                                        np.ndarray v_X_w_dS):
    cdef int nElements_global = v_X_w_dS.shape[0]
    cdef int nElementBoundaries_element = v_X_w_dS.shape[1]
    cdef int nElementBoundaryQuadraturePoints_elementBoundary = v_X_w_dS.shape[2]
    cdef int nDOF_trial_element = v_X_w_dS.shape[3]
    cdef int nDOF_test_element = v_X_w_dS.shape[4]
    ccalculateShape_X_weightedShapeTrace(nElements_global,
                                         nElementBoundaries_element,
                                         nElementBoundaryQuadraturePoints_elementBoundary,
                                         nDOF_trial_element,
                                         nDOF_test_element,
                                         <double*>v.data,
                                         <double*>w_dS.data,
                                         <double*>v_X_w_dS.data)
def calculateGradShape_X_weightedShapeTrace(np.ndarray grad_v,
                                            np.ndarray w_dS,
                                            np.ndarray grad_v_X_w_dS):
    cdef int nElements_global = grad_v_X_w_dS.shape[0]
    cdef int nElementBoundaries_element = grad_v_X_w_dS.shape[1]
    cdef int nElementBoundaryQuadraturePoints_elementBoundary = grad_v_X_w_dS.shape[2]
    cdef int nDOF_trial_element = grad_v_X_w_dS.shape[3]
    cdef int nDOF_test_element = grad_v_X_w_dS.shape[4]
    cdef int nSpace = grad_v_X_w_dS.shape[5]
    ccalculateGradShape_X_weightedShapeTrace(nElements_global,
                                             nElementBoundaries_element,
                                             nElementBoundaryQuadraturePoints_elementBoundary,
                                             nDOF_trial_element,
                                             nDOF_test_element,
                                             nSpace,
                                             <double*>grad_v.data,
                                             <double*>w_dS.data,
                                             <double*>grad_v_X_w_dS.data)
def calculateIntegrationWeights(np.ndarray abs_det_J,
                                np.ndarray referenceWeights,
                                np.ndarray weights):
    cdef int nElements_global = abs_det_J.shape[0]
    cdef int nQuadraturePoints_element = abs_det_J.shape[1]
    ccalculateIntegrationWeights(nElements_global,
                                 nQuadraturePoints_element,
                                 <double*>abs_det_J.data,
                                 <double*>referenceWeights.data,
                                 <double*>weights.data)
def calculateElementBoundaryIntegrationWeights(np.ndarray sqrt_det_g,
                                               np.ndarray referenceWeights,
                                               np.ndarray weights):
    cdef int nElements_global = sqrt_det_g.shape[0]
    cdef int nElementBoundaries_element = sqrt_det_g.shape[1]
    cdef int nQuadraturePoints_elementBoundary = sqrt_det_g.shape[2]
    ccalculateElementBoundaryIntegrationWeights(nElements_global,
                                                nElementBoundaries_element,
                                                nQuadraturePoints_elementBoundary,
                                                <double*>sqrt_det_g.data,
                                                <double*>referenceWeights.data,
                                                <double*>weights.data)
def calculateFiniteElementFunctionValues(np.ndarray l2g,
                                         np.ndarray dof,
                                         np.ndarray v,
                                         np.ndarray u):
    cdef int nElements_global = v.shape[0]
    cdef int nQuadraturePoints_element = v.shape[1]
    cdef int nDOF_trial_element = v.shape[2]
    cdef int nd = u.ndim
    cdef int nComponents = 1
    if nd == 3:
        nComponents = u.shape[2]
    else:
        nComponents = 1
    ccalculateFiniteElementFunctionValues(nElements_global,
                                          nQuadraturePoints_element,
                                          nDOF_trial_element,
                                          nComponents,
                                          <proteus_index_t*>l2g.data,
                                          <double*>dof.data,
                                          <double*>v.data,
                                          <double*>u.data)
def calculateFiniteElementFunctionGradientValues(np.ndarray l2g,
                                                 np.ndarray dof,
                                                 np.ndarray grad_v,
                                                 np.ndarray grad_u):
    cdef int nElements_global = grad_v.shape[0]
    cdef int nQuadraturePoints_element = grad_v.shape[1]
    cdef int nDOF_trial_element = grad_v.shape[2]
    cdef int nd = grad_u.ndim
    cdef int nComponents = 1
    if nd == 4:
        nComponents = grad_u.shape[2]
        nSpace = grad_u.shape[3]
    else:
        nComponents = 1
        nSpace = grad_u.shape[2]
    ccalculateFiniteElementFunctionGradientValues(nElements_global,
                                                  nQuadraturePoints_element,
                                                  nDOF_trial_element,
                                                  nComponents,
                                                  nSpace,
                                                  <proteus_index_t*>l2g.data,
                                                  <double*>dof.data,
                                                  <double*>grad_v.data,
                                                  <double*>grad_u.data)
def calculateFiniteElementFunctionGradientTensorValues(np.ndarray l2g,
                                                       np.ndarray dof,
                                                       np.ndarray grad_v_X_grad_w_dV,
                                                       np.ndarray grad_u_X_grad_w_dV):
    cdef int nElements_global = grad_v_X_grad_w_dV.shape[0]
    cdef int nQuadraturePoints_element = grad_v_X_grad_w_dV.shape[1]
    cdef int nDOF_trial_element = grad_v_X_grad_w_dV.shape[2]
    cdef int nDOF_test_element = grad_v_X_grad_w_dV.shape[3]
    cdef int nComponents = 1
    cdef int nSpace = 1
    if grad_u_X_grad_w_dV.ndim == 6:
        nComponents = grad_u_X_grad_w_dV.shape[3]
        nSpace = grad_u_X_grad_w_dV.shape[4]
    else:
        nComponents = 1
        nSpace = grad_u_X_grad_w_dV.shape[3]
    ccalculateFiniteElementFunctionGradientTensorValues(nElements_global,
                                                        nQuadraturePoints_element,
                                                        nDOF_trial_element,
                                                        nDOF_test_element,
                                                        nComponents,
                                                        nSpace,
                                                        <proteus_index_t*>l2g.data,
                                                        <double*>dof.data,
                                                        <double*>grad_v_X_grad_w_dV.data,
                                                        <double*>grad_u_X_grad_w_dV.data)
def calculateFiniteElementFunctionValuesTrace(np.ndarray l2g,
                                              np.ndarray dof,
                                              np.ndarray v,
                                              np.ndarray u):
    cdef int nElements_global = v.shape[0]
    cdef int nElementBoundaries_element = v.shape[1]
    cdef int nQuadraturePoints_elementBoundary = v.shape[2]
    cdef int nDOF_trial_element = v.shape[3]
    cdef int nComponents = 1
    cdef int nd = u.ndim
    if nd == 4:
        nComponents = u.shape[3]
    else:
        nComponents = 1
    ccalculateFiniteElementFunctionValuesTrace(nElements_global,
                                               nElementBoundaries_element,
                                               nQuadraturePoints_elementBoundary,
                                               nDOF_trial_element,
                                               nComponents,
                                               <proteus_index_t*>l2g.data,
                                               <double*>dof.data,
                                               <double*>v.data,
                                               <double*>u.data)
def calculateFiniteElementFunctionGradientValuesTrace(np.ndarray l2g,
                                                      np.ndarray dof,
                                                      np.ndarray grad_v,
                                                      np.ndarray grad_u):
    cdef int nElements_global = grad_v.shape[0]
    cdef int nElementBoundaries_element = grad_v.shape[1]
    cdef int nQuadraturePoints_elementBoundary = grad_v.shape[2]
    cdef int nDOF_trial_element = grad_v.shape[3]
    cdef int nComponents =1
    cdef int nSpace = 1
    cdef int nd = grad_u.ndim
    if nd == 5:
        nComponents = grad_u.shape[3]
        nSpace = grad_u.shape[4]
    else:
        nComponents = 1
        nSpace = grad_u.shape[3]
    ccalculateFiniteElementFunctionGradientValuesTrace(nElements_global,
                                                       nElementBoundaries_element,
                                                       nQuadraturePoints_elementBoundary,
                                                       nDOF_trial_element,
                                                       nComponents,
                                                       nSpace,
                                                       <proteus_index_t*>l2g.data,
                                                       <double*>dof.data,
                                                       <double*>grad_v.data,
                                                       <double*>grad_u.data)
def calculateFlowVelocity(np.ndarray f,
                          np.ndarray a,
                          np.ndarray grad_phi,
                          np.ndarray v):
    cdef int nElements_global = f.shape[0]
    cdef int nQuadraturePoints_element = f.shape[1]
    cdef int nSpace = f.shape[2]
    ccalculateFlowVelocity(nElements_global,
                           nQuadraturePoints_element,
                           nSpace,
                           <double*>f.data,
                           <double*>a.data,
                           <double*>grad_phi.data,
                           <double*>v.data)
def updateAddJacobian_CSR(int jacIndex,
                          double val,
                          jac):
    cdef np.ndarray rowptr, colind, jac_array
    (rowptr,colind,jac_array) = jac.getCSRrepresentation()
    cupdateAddJacobian_CSR(jacIndex,
                           val,
                           <double*>jac_array.data)
def zeroJacobian_CSR(proteus_index_t nNonzeros,
                     jac):
    cdef np.ndarray rowptr, colind, jac_array
    (rowptr,colind,jac_array) = jac.getCSRrepresentation()
    czeroJacobian_CSR(nNonzeros,
                      <double*>jac_array.data)
def calculateInteriorElementBoundaryVelocities(proteus_index_t nInteriorElementBoundaries_global,
                                               int nElementBoundaries_element,
                                               int nQuadraturePoints_elementBoundary,
                                               int nSpace,
                                               np.ndarray interiorElementBoundaries,
                                               np.ndarray elementBoundaryElements,
                                               np.ndarray elementBoundaryLocalElementBoundaries,
                                               np.ndarray m,
                                               np.ndarray a,
                                               np.ndarray grad_phi,
                                               np.ndarray f,
                                               np.ndarray vAverage,
                                               np.ndarray vJump,
                                               np.ndarray mAverage,
                                               np.ndarray mJump):
    ccalculateInteriorElementBoundaryVelocities(nInteriorElementBoundaries_global,
                                                nElementBoundaries_element,
                                                nQuadraturePoints_elementBoundary,
                                                nSpace,
                                                <proteus_index_t*>interiorElementBoundaries.data,
                                                <proteus_index_t*>elementBoundaryElements.data,
                                                <int *>elementBoundaryLocalElementBoundaries.data,
                                                <double*>m.data,
                                                <double*>a.data,
                                                <double*>grad_phi.data,
                                                <double*>f.data,
                                                <double*>vAverage.data,
                                                <double*>vJump.data,
                                                <double*>mAverage.data,
                                                <double*>mJump.data)
def calculateExteriorElementBoundaryVelocities(proteus_index_t nExteriorElementBoundaries_global,
                                               int nElementBoundaries_element,
                                               int nQuadraturePoints_elementBoundary,
                                               int nSpace,
                                               np.ndarray exteriorElementBoundaries,
                                               np.ndarray elementBoundaryElements,
                                               np.ndarray elementBoundaryLocalElementBoundaries,
                                               np.ndarray m,
                                               np.ndarray a,
                                               np.ndarray grad_phi,
                                               np.ndarray f,
                                               np.ndarray vAverage,
                                               np.ndarray vJump,
                                               np.ndarray mAverage,
                                               np.ndarray mJump):
    ccalculateExteriorElementBoundaryVelocities(nExteriorElementBoundaries_global,
                                                nElementBoundaries_element,
                                                nQuadraturePoints_elementBoundary,
                                                nSpace,
                                                <proteus_index_t*>exteriorElementBoundaries.data,
                                                <proteus_index_t*>elementBoundaryElements.data,
                                                <int *>elementBoundaryLocalElementBoundaries.data,
                                                <double*>m.data,
                                                <double*>a.data,
                                                <double*>grad_phi.data,
                                                <double*>f.data,
                                                <double*>vAverage.data,
                                                <double*>vJump.data,
                                                <double*>mAverage.data,
                                                <double*>mJump.data)
def calculateConservationResidualPWL(proteus_index_t nElements_global,
                                     proteus_index_t nInteriorElementBoundaries_global,
                                     proteus_index_t nExteriorElementBoundaries_global,
                                     int nElementBoundaries_element,
                                     int nQuadraturePoints_elementBoundary,
                                     int nNodes_element,
                                     int nSpace,
                                     np.ndarray interiorElementBoundaries,
                                     np.ndarray exteriorElementBoundaries,
                                     np.ndarray elementBoundaryElements,
                                     np.ndarray elementBoundaryLocalElementBoundaries,
                                     np.ndarray elementNodes,
                                     np.ndarray nodeStarElements,
                                     np.ndarray nodeStarElementNeighbors,
                                     np.ndarray nodeStarOffsets,
                                     np.ndarray nElements_node,
                                     np.ndarray elementResidual,
                                     np.ndarray vAverage,
                                     np.ndarray starU,
                                     np.ndarray dX,
                                     np.ndarray w,
                                     np.ndarray normal,
                                     np.ndarray conservationResidual,
                                     np.ndarray starR,
                                     np.ndarray vConservative,
                                     np.ndarray vConservative_element):
    ccalculateConservationResidualPWL(nElements_global,
                                      nInteriorElementBoundaries_global,
                                      nExteriorElementBoundaries_global,
                                      nElementBoundaries_element,
                                      nQuadraturePoints_elementBoundary,
                                      nNodes_element,
                                      nSpace,
                                      <proteus_index_t*>interiorElementBoundaries.data,
                                      <proteus_index_t*>exteriorElementBoundaries.data,
                                      <proteus_index_t*>elementBoundaryElements.data,
                                      <int *>elementBoundaryLocalElementBoundaries.data,
                                      <proteus_index_t*>elementNodes.data,
                                      <proteus_index_t*>nodeStarElements.data,
                                      <proteus_index_t*>nodeStarElementNeighbors.data,
                                      <proteus_index_t*>nodeStarOffsets.data,
                                      <int *>nElements_node.data,
                                      <double*>elementResidual.data,
                                      <double*>vAverage.data,
                                      <double*>starU.data,
                                      <double*>dX.data,
                                      <double*>w.data,
                                      <double*>normal.data,
                                      <double*>conservationResidual.data,
                                      <double*>starR.data,
                                      <double*>vConservative.data,
                                      <double*>vConservative_element.data)
def calculateConservationJacobianPWL(proteus_index_t nNodes_global,
                                     proteus_index_t nNodes_internal,
                                     proteus_index_t nElements_global,
                                     proteus_index_t nInteriorElementBoundaries_global,
                                     proteus_index_t nExteriorElementBoundaries_global,
                                     int nElementBoundaries_element,
                                     int nQuadraturePoints_elementBoundary,
                                     int nNodes_element,
                                     int nSpace,
                                     np.ndarray interiorElementBoundaries,
                                     np.ndarray exteriorElementBoundaries,
                                     np.ndarray elementBoundaryElements,
                                     np.ndarray elementBoundaryLocalElementBoundaries,
                                     np.ndarray elementNodes,
                                     np.ndarray nodeStarElements,
                                     np.ndarray nodeStarElementNeighbors,
                                     np.ndarray nodeStarOffsets,
                                     np.ndarray nodeStarJacobianOffsets,
                                     np.ndarray nElements_node,
                                     np.ndarray internalNodes,
                                     np.ndarray w,
                                     np.ndarray normal,
                                     np.ndarray starJacobian):
    ccalculateConservationJacobianPWL(nNodes_global,
                                      nNodes_internal,
                                      nElements_global,
                                      nInteriorElementBoundaries_global,
                                      nExteriorElementBoundaries_global,
                                      nElementBoundaries_element,
                                      nQuadraturePoints_elementBoundary,
                                      nNodes_element,
                                      nSpace,
                                      <proteus_index_t*>interiorElementBoundaries.data,
                                      <proteus_index_t*>exteriorElementBoundaries.data,
                                      <proteus_index_t*>elementBoundaryElements.data,
                                      <int *>elementBoundaryLocalElementBoundaries.data,
                                      <proteus_index_t*>elementNodes.data,
                                      <proteus_index_t*>nodeStarElements.data,
                                      <proteus_index_t*>nodeStarElementNeighbors.data,
                                      <proteus_index_t*>nodeStarOffsets.data,
                                      <proteus_index_t*>nodeStarJacobianOffsets.data,
                                      <int *>nElements_node.data,
                                      <proteus_index_t*>internalNodes.data,
                                      <double*>w.data,
                                      <double*>normal.data,
                                      <double*>starJacobian.data)
def calculateConservationFluxPWL(proteus_index_t nNodes_global,
                                 proteus_index_t nNodes_internal,
                                 np.ndarray nElements_node,
                                 np.ndarray nodeStarOffsets,
                                 np.ndarray nodeStarJacobianOffsets,
                                 np.ndarray internalNodes,
                                 np.ndarray starR,
                                 np.ndarray starJ,
                                 np.ndarray starU):
    ccalculateConservationFluxPWL(nNodes_global,
                                  nNodes_internal,
                                  <int *>nElements_node.data,
                                  <proteus_index_t*>nodeStarOffsets.data,
                                  <proteus_index_t*>nodeStarJacobianOffsets.data,
                                  <proteus_index_t*>internalNodes.data,
                                  <double*>starR.data,
                                  <double*>starJ.data,
                                  <double*>starU.data)
def setExteriorGlobalElementBoundaryVelocityValues(int updateFluxValues,
                                                   proteus_index_t nExteriorElementBoundaries_global,
                                                   int nQuadraturePoints_elementBoundary,
                                                   int nSpace,
                                                   np.ndarray exteriorElementBoundaries,
                                                   np.ndarray elementBoundaryElements,
                                                   np.ndarray elementBoundaryLocalElementBoundaries,
                                                   np.ndarray n,
                                                   np.ndarray vn_in,
                                                   np.ndarray v_out):
    csetExteriorGlobalElementBoundaryVelocityValues(updateFluxValues,
                                                    nExteriorElementBoundaries_global,
                                                    nQuadraturePoints_elementBoundary,
                                                    nSpace,
                                                    <proteus_index_t*>exteriorElementBoundaries.data,
                                                    <proteus_index_t*>elementBoundaryElements.data,
                                                    <int *>elementBoundaryLocalElementBoundaries.data,
                                                    <double*>n.data,
                                                    <double*>vn_in.data,
                                                    <double*>v_out.data)
def calculateDimensionlessNumbersADR(proteus_index_t nElements_global,
                                     int nQuadraturePoints_element,
                                     int nSpace,
                                     np.ndarray elementDiameter,
                                     np.ndarray df,
                                     np.ndarray a,
                                     np.ndarray dphi,
                                     np.ndarray dr,
                                     np.ndarray dmt,
                                     np.ndarray pe,
                                     np.ndarray cfl):
    cdef int computeDiffusiveTimeStepLimit = 0
    ccalculateDimensionlessNumbersADR(nElements_global,
                                      nQuadraturePoints_element,
                                      nSpace,
                                      computeDiffusiveTimeStepLimit,
                                      <double*>elementDiameter.data,
                                      <double*>df.data,
                                      <double*>a.data,
                                      <double*>dphi.data,
                                      <double*>dr.data,
                                      <double*>dmt.data,
                                      <double*>pe.data,
                                      <double*>cfl.data)
def calculateDimensionlessNumbersADR_sd(proteus_index_t nElements_global,
                                        int nQuadraturePoints_element,
                                        int nSpace,
                                        np.ndarray rowptr,
                                        np.ndarray colind,
                                        np.ndarray elementDiameter,
                                        np.ndarray df,
                                        np.ndarray a,
                                        np.ndarray dphi,
                                        np.ndarray dr,
                                        np.ndarray dmt,
                                        np.ndarray pe,
                                        np.ndarray cfl):
    cdef int computeDiffusiveTimeStepLimit = 0
    ccalculateDimensionlessNumbersADR_sd(nElements_global,
                                         nQuadraturePoints_element,
                                         nSpace,
					 computeDiffusiveTimeStepLimit,
                                         <proteus_csr_index_t*> rowptr.data,
                                         <proteus_csr_index_t*> colind.data,
                                         <double*> elementDiameter.data,
                                         <double*> df.data,
                                         <double*> a.data,
                                         <double*> dphi.data,
                                         <double*> dr.data,
                                         <double*> dmt.data,
                                         <double*> pe.data,
                                         <double*> cfl.data)
def calculateCFLADR(np.ndarray elementDiameter,
                    np.ndarray dm,
                    np.ndarray df,
                    np.ndarray cfl):
    cdef int nElements_global = df.shape[0]
    cdef int nQuadraturePoints_element = df.shape[1]
    cdef int nSpace = df.shape[2]
    ccalculateCFLADR(nElements_global,
                     nQuadraturePoints_element,
                     nSpace,
                     <double*>elementDiameter.data,
                     <double*>dm.data,
                     <double*>df.data,
                     <double*>cfl.data)
def updateInteriorElementBoundaryDiffusiveVelocity(np.ndarray interiorElementBoundaries,
                                                   np.ndarray elementBoundaryElements,
                                                   np.ndarray elementBoundaryLocalElementBoundaries,
                                                   np.ndarray a,
                                                   np.ndarray grad_phi,
                                                   np.ndarray velocity):
    cdef int nInteriorElementBoundaries_global = interiorElementBoundaries.shape[0]
    cdef int nElementBoundaries_element = grad_phi.shape[1]
    cdef int nQuadraturePoints_elementBoundary = grad_phi.shape[2]
    cdef int nSpace = grad_phi.shape[3]
    cupdateInteriorElementBoundaryDiffusiveVelocity(nInteriorElementBoundaries_global,
                                                    nElementBoundaries_element,
                                                    nQuadraturePoints_elementBoundary,
                                                    nSpace,
                                                    <proteus_index_t*>interiorElementBoundaries.data,
                                                    <proteus_index_t*>elementBoundaryElements.data,
                                                    <int *>elementBoundaryLocalElementBoundaries.data,
                                                    <double*>a.data,
                                                    <double*>grad_phi.data,
                                                    <double*>velocity.data)
def updateInteriorElementBoundaryDiffusiveVelocity_sd(np.ndarray rowptr,
                                                      np.ndarray colind,
                                                      np.ndarray interiorElementBoundaries,
                                                      np.ndarray elementBoundaryElements,
                                                      np.ndarray elementBoundaryLocalElementBoundaries,
                                                      np.ndarray a,
                                                      np.ndarray grad_phi,
                                                      np.ndarray velocity):
    cdef int nInteriorElementBoundaries_global = interiorElementBoundaries.shape[0]
    cdef int nElementBoundaries_element = grad_phi.shape[1]
    cdef int nQuadraturePoints_elementBoundary = grad_phi.shape[2]
    cdef int nSpace = grad_phi.shape[3]
    cupdateInteriorElementBoundaryDiffusiveVelocity_sd(nInteriorElementBoundaries_global,
                                                       nElementBoundaries_element,
                                                       nQuadraturePoints_elementBoundary,
                                                       nSpace,
                                                       <proteus_csr_index_t*> rowptr.data,
                                                       <proteus_csr_index_t*> colind.data,
                                                       <proteus_index_t*> interiorElementBoundaries.data,
                                                       <proteus_index_t*> elementBoundaryElements.data,
                                                       <int*> elementBoundaryLocalElementBoundaries.data,
                                                       <double*> a.data,
                                                       <double*> grad_phi.data,
                                                       <double*> velocity.data)
def updateExteriorElementBoundaryDiffusiveVelocity(np.ndarray exteriorElementBoundaries,
                                                   np.ndarray elementBoundaryElements,
                                                   np.ndarray elementBoundaryLocalElementBoundaries,
                                                   np.ndarray a,
                                                   np.ndarray grad_phi,
                                                   np.ndarray velocity):
    cdef int nd = grad_phi.ndim
    if nd > 3:
        assert nd == velocity.ndim
        cupdateExteriorElementBoundaryDiffusiveVelocity(exteriorElementBoundaries.shape[0],
                                                        grad_phi.shape[1],
                                                        grad_phi.shape[2],
                                                        grad_phi.shape[3],
                                                        <proteus_index_t*>exteriorElementBoundaries.data,
                                                        <proteus_index_t*>elementBoundaryElements.data,
                                                        <int *>elementBoundaryLocalElementBoundaries.data,
                                                        <double*>a.data,
                                                        <double*>grad_phi.data,
                                                        <double*>velocity.data)
    else:
        assert nd == velocity.ndim
        cupdateGlobalExteriorElementBoundaryDiffusiveVelocity(exteriorElementBoundaries.shape[0],
                                                              grad_phi.shape[1],
                                                              grad_phi.shape[2],
							      <proteus_index_t*> exteriorElementBoundaries.data,
							      <proteus_index_t*> elementBoundaryElements.data,
							      <int*> elementBoundaryLocalElementBoundaries.data,
							      <double*> a.data,
							      <double*> grad_phi.data,
							      <double*> velocity.data)
def updateExteriorElementBoundaryDiffusiveVelocity_sd(np.ndarray rowptr,
                                                      np.ndarray colind,
                                                      np.ndarray exteriorElementBoundaries,
                                                      np.ndarray elementBoundaryElements,
                                                      np.ndarray elementBoundaryLocalElementBoundaries,
                                                      np.ndarray a,
                                                      np.ndarray grad_phi,
                                                      np.ndarray velocity):
    cdef int nExteriorElementBoundaries_global = exteriorElementBoundaries.shape[0]
    cdef int nElementBoundaries_element = grad_phi.shape[1]
    cdef int nQuadraturePoints_elementBoundary = grad_phi.shape[2]
    cdef int nSpace = grad_phi.shape[3]
    cupdateExteriorElementBoundaryDiffusiveVelocity_sd(nExteriorElementBoundaries_global,
                                                       nElementBoundaries_element,
                                                       nQuadraturePoints_elementBoundary,
                                                       nSpace,
                                                       <proteus_csr_index_t*> rowptr.data,
                                                       <proteus_csr_index_t*> colind.data,
                                                       <proteus_index_t*> exteriorElementBoundaries.data,
                                                       <proteus_index_t*> elementBoundaryElements.data,
                                                       <int*> elementBoundaryLocalElementBoundaries.data,
                                                       <double*> a.data,
                                                       <double*> grad_phi.data,
                                                       <double*> velocity.data)
def updateInteriorElementBoundaryAdvectiveVelocity(np.ndarray interiorElementBoundaries,
                                                   np.ndarray elementBoundaryElements,
                                                   np.ndarray elementBoundaryLocalElementBoundaries,
                                                   np.ndarray f,
                                                   np.ndarray velocity):
    cdef int nInteriorElementBoundaries_global = interiorElementBoundaries.shape[0]
    cdef int nElementBoundaries_element = f.shape[1]
    cdef int nQuadraturePoints_elementBoundary = f.shape[2]
    cdef int nSpace = f.shape[3]
    cupdateInteriorElementBoundaryAdvectiveVelocity(nInteriorElementBoundaries_global,
                                                    nElementBoundaries_element,
                                                    nQuadraturePoints_elementBoundary,
                                                    nSpace,
                                                    <proteus_index_t*>interiorElementBoundaries.data,
                                                    <proteus_index_t*>elementBoundaryElements.data,
                                                    <int *>elementBoundaryLocalElementBoundaries.data,
                                                    <double*>f.data,
                                                    <double*>velocity.data)
def updateExteriorElementBoundaryAdvectiveVelocity(np.ndarray exteriorElementBoundaries,
                                                   np.ndarray elementBoundaryElements,
                                                   np.ndarray elementBoundaryLocalElementBoundaries,
                                                   np.ndarray f,
                                                   np.ndarray velocity):
    cdef int nd = f.ndim
    if nd > 3:
        assert nd == velocity.ndim
        cupdateExteriorElementBoundaryAdvectiveVelocity(exteriorElementBoundaries.shape[0],
                                                        f.shape[1],
                                                        f.shape[2],
                                                        f.shape[3],
                                                        <proteus_index_t*>exteriorElementBoundaries.data,
                                                        <proteus_index_t*>elementBoundaryElements.data,
                                                        <int *>elementBoundaryLocalElementBoundaries.data,
                                                        <double*>f.data,
                                                        <double*>velocity.data)
    else:
        assert nd == velocity.ndim
        cupdateGlobalExteriorElementBoundaryAdvectiveVelocity(exteriorElementBoundaries.shape[0],
                                                              f.shape[1],
                                                              f.shape[2],
							      <proteus_index_t*> exteriorElementBoundaries.data,
							      <proteus_index_t*> elementBoundaryElements.data,
							      <int*> elementBoundaryLocalElementBoundaries.data,
							      <double*> f.data,
							      <double*> velocity.data)
def updateInteriorElementBoundaryShockCapturingVelocity(np.ndarray interiorElementBoundaries,
                                                        np.ndarray elementBoundaryElements,
                                                        np.ndarray elementBoundaryLocalElementBoundaries,
                                                        np.ndarray numDiff,
                                                        np.ndarray grad_u,
                                                        np.ndarray velocity):
    cdef int nInteriorElementBoundaries_global = interiorElementBoundaries.shape[0]
    cdef int nElementBoundaries_element = grad_u.shape[1]
    cdef int nQuadraturePoints_elementBoundary = grad_u.shape[2]
    cdef int nQuadraturePoints_element = numDiff.shape[1]
    cdef int nSpace = grad_u.shape[3]
    cupdateInteriorElementBoundaryShockCapturingVelocity(nInteriorElementBoundaries_global,
                                                         nElementBoundaries_element,
                                                         nQuadraturePoints_elementBoundary,
                                                         nQuadraturePoints_element,
                                                         nSpace,
                                                         <proteus_index_t*>interiorElementBoundaries.data,
                                                         <proteus_index_t*>elementBoundaryElements.data,
                                                         <int *>elementBoundaryLocalElementBoundaries.data,
                                                         <double*>numDiff.data,
                                                         <double*>grad_u.data,
                                                         <double*>velocity.data)
def updateExteriorElementBoundaryShockCapturingVelocity(np.ndarray exteriorElementBoundaries,
                                                        np.ndarray elementBoundaryElements,
                                                        np.ndarray elementBoundaryLocalElementBoundaries,
                                                        np.ndarray numDiff,
                                                        np.ndarray grad_u,
                                                        np.ndarray velocity):
    if grad_u.ndim > 3:
        assert numDiff.shape[0] == grad_u.shape[0]
        cupdateExteriorElementBoundaryShockCapturingVelocity(exteriorElementBoundaries.shape[0],
                                                             grad_u.shape[1],
                                                             grad_u.shape[2],
                                                             numDiff.shape[1],
                                                             grad_u.shape[3],
                                                             <proteus_index_t*>exteriorElementBoundaries.data,
                                                             <proteus_index_t*>elementBoundaryElements.data,
                                                             <int *>elementBoundaryLocalElementBoundaries.data,
                                                             <double*>numDiff.data,
                                                             <double*>grad_u.data,
                                                             <double*>velocity.data)
    else:
        assert numDiff.shape[0] == grad_u.shape[0]
        cupdateGlobalExteriorElementBoundaryShockCapturingVelocity(exteriorElementBoundaries.shape[0],
                                                                   grad_u.shape[1],
                                                                   grad_u.shape[2],
							           <proteus_index_t*> exteriorElementBoundaries.data,
							           <proteus_index_t*> elementBoundaryElements.data,
							           <int*> elementBoundaryLocalElementBoundaries.data,
							           <double*> numDiff.data,
							           <double*> grad_u.data,
							           <double*> velocity.data)
def calculateInteriorElementBoundaryAverageVelocity(np.ndarray interiorElementBoundaries,
                                                    np.ndarray elementBoundaryElements,
                                                    np.ndarray elementBoundaryLocalElementBoundaries,
                                                    np.ndarray v,
                                                    np.ndarray vAverage):
    cdef int nInteriorElementBoundaries_global = interiorElementBoundaries.shape[0]
    cdef int nElementBoundaries_element = v.shape[1]
    cdef int nQuadraturePoints_elementBoundary = v.shape[2]
    cdef int nSpace = v.shape[3]
    ccalculateInteriorElementBoundaryAverageVelocity(nInteriorElementBoundaries_global,
                                                     nElementBoundaries_element,
                                                     nQuadraturePoints_elementBoundary,
                                                     nSpace,
                                                     <proteus_index_t*>interiorElementBoundaries.data,
                                                     <proteus_index_t*>elementBoundaryElements.data,
                                                     <int *>elementBoundaryLocalElementBoundaries.data,
                                                     <double*>v.data,
                                                     <double*>vAverage.data)
def calculateExteriorElementBoundaryAverageVelocity(np.ndarray exteriorElementBoundaries,
                                                    np.ndarray elementBoundaryElements,
                                                    np.ndarray elementBoundaryLocalElementBoundaries,
                                                    np.ndarray v,
                                                    np.ndarray vAverage):
    cdef int nExteriorElementBoundaries_global = exteriorElementBoundaries.shape[0]
    cdef int nElementBoundaries_element = v.shape[1]
    cdef int nQuadraturePoints_elementBoundary = v.shape[2]
    cdef int nSpace = v.shape[3]
    ccalculateExteriorElementBoundaryAverageVelocity(nExteriorElementBoundaries_global,
                                                     nElementBoundaries_element,
                                                     nQuadraturePoints_elementBoundary,
                                                     nSpace,
                                                     <proteus_index_t*>exteriorElementBoundaries.data,
                                                     <proteus_index_t*>elementBoundaryElements.data,
                                                     <int *>elementBoundaryLocalElementBoundaries.data,
                                                     <double*>v.data,
                                                     <double*>vAverage.data)
def calculateConservationResidualDG(np.ndarray elementResidual,
                                    np.ndarray conservationResidual):
    cdef int nElements_global = elementResidual.shape[0]
    cdef int nDOF_test_element = elementResidual.shape[1]
    ccalculateConservationResidualDG(nElements_global,
                                     nDOF_test_element,
                                     <double*>elementResidual.data,
                                     <double*>conservationResidual.data)
def calculateConservationResidual(np.ndarray n,
                                  np.ndarray dS_u,
                                  np.ndarray elementResidual,
                                  np.ndarray velocity,
                                  np.ndarray conservationResidual):
    cdef int nElements_global = elementResidual.shape[0]
    cdef int nDOF_test_element = elementResidual.shape[1]
    cdef int nElementBoundaries_element = n.shape[1]
    cdef int nQuadraturePoints_elementBoundary = n.shape[2]
    cdef int nSpace = n.shape[3]
    ccalculateConservationResidual(nElements_global,
                                   nDOF_test_element,
                                   nElementBoundaries_element,
                                   nQuadraturePoints_elementBoundary,
                                   nSpace,
                                   <double*>n.data,
                                   <double*>dS_u.data,
                                   <double*>elementResidual.data,
                                   <double*>velocity.data,
                                   <double*>conservationResidual.data)
def calculateConservationResidualGlobalBoundaries(proteus_index_t nElements_global,
                                                  proteus_index_t nInteriorElementBoundaries_global,
                                                  proteus_index_t nExteriorElementBoundaries_global,
                                                  int nElementBoundaries_element,
                                                  int nQuadraturePoints_elementBoundary,
                                                  int nNodes_element,
                                                  int nSpace,
                                                  np.ndarray interiorElementBoundaries,
                                                  np.ndarray exteriorElementBoundaries,
                                                  np.ndarray elementBoundaryElements,
                                                  np.ndarray elementBoundaryLocalElementBoundaries,
                                                  np.ndarray dS,
                                                  np.ndarray normal,
                                                  np.ndarray elementResidual,
                                                  np.ndarray velocity,
                                                  np.ndarray conservationResidual):
    ccalculateConservationResidualGlobalBoundaries(nElements_global,
                                                   nInteriorElementBoundaries_global,
                                                   nExteriorElementBoundaries_global,
                                                   nElementBoundaries_element,
                                                   nQuadraturePoints_elementBoundary,
                                                   nNodes_element,
                                                   nSpace,
                                                   <proteus_index_t*>interiorElementBoundaries.data,
                                                   <proteus_index_t*>exteriorElementBoundaries.data,
                                                   <proteus_index_t*>elementBoundaryElements.data,
                                                   <int *>elementBoundaryLocalElementBoundaries.data,
                                                   <double*>dS.data,
                                                   <double*>normal.data,
                                                   <double*>elementResidual.data,
                                                   <double*>velocity.data,
                                                   <double*>conservationResidual.data)
def copyGlobalElementBoundaryVelocityToElementBoundary(np.ndarray interiorElementBoundaries,
                                                       np.ndarray exteriorElementBoundaries,
                                                       np.ndarray elementBoundaryElementsArray,
                                                       np.ndarray elementBoundaryLocalElementBoundariesArray,
                                                       np.ndarray velocityBoundary_global,
                                                       np.ndarray velocityBoundary_element):
    cdef int nElements_global = velocityBoundary_element.shape[0]
    cdef int nInteriorElementBoundaries_global = interiorElementBoundaries.shape[0]
    cdef int nExteriorElementBoundaries_global = exteriorElementBoundaries.shape[0]
    cdef int nElementBoundaries_global = velocityBoundary_global.shape[0]
    cdef int nElementBoundaries_element = velocityBoundary_element.shape[1]
    cdef int nQuadraturePoints_elementBoundary = velocityBoundary_element.shape[2]
    cdef int nSpace = velocityBoundary_element.shape[3]
    ccopyGlobalElementBoundaryVelocityToElementBoundary(nElements_global,
                                                        nInteriorElementBoundaries_global,
                                                        nExteriorElementBoundaries_global,
                                                        nElementBoundaries_global,
                                                        nElementBoundaries_element,
                                                        nQuadraturePoints_elementBoundary,
                                                        nSpace,
                                                        <proteus_index_t*>interiorElementBoundaries.data,
                                                        <proteus_index_t*>exteriorElementBoundaries.data,
                                                        <proteus_index_t*>elementBoundaryElementsArray.data,
                                                        <int *>elementBoundaryLocalElementBoundariesArray.data,
                                                        <double*>velocityBoundary_global.data,
                                                        <double*>velocityBoundary_element.data)
def loadBoundaryFluxIntoGlobalElementBoundaryVelocity(np.ndarray exteriorElementBoundaries,
                                                      np.ndarray fluxElementBoundaries,
                                                      np.ndarray normal,
                                                      np.ndarray flux,
                                                      double updateCoef,
                                                      np.ndarray velocity):
    cdef int nExteriorElementBoundaries_global = exteriorElementBoundaries.shape[0]
    cdef int nQuadraturePoints_elementBoundary = normal.shape[1]
    cdef int nSpace = normal.shape[2]
    cloadBoundaryFluxIntoGlobalElementBoundaryVelocity(nExteriorElementBoundaries_global,
                                                       nQuadraturePoints_elementBoundary,
                                                       nSpace,
                                                       <proteus_index_t*>exteriorElementBoundaries.data,
                                                       <proteus_index_t*>fluxElementBoundaries.data,
                                                       <double*>normal.data,
                                                       <double*>flux.data,
                                                       updateCoef,
                                                       <double*>velocity.data)
def calculateInteriorNumericalTrace_Potential(np.ndarray interiorElementBoundaries,
                                              np.ndarray elementBoundaryElements,
                                              np.ndarray elementBoundaryLocalElementBoundaries,
                                              np.ndarray phi,
                                              np.ndarray dphi,
                                              np.ndarray phi_trace,
                                              np.ndarray dphi_trace_left,
                                              np.ndarray dphi_trace_right):
    cdef int nInteriorElementBoundaries_global = interiorElementBoundaries.shape[0]
    cdef int nElementBoundaries_element = phi.shape[1]
    cdef int nQuadraturePoints_elementBoundary = phi.shape[2]
    ccalculateInteriorNumericalTrace_Potential(nInteriorElementBoundaries_global,
                                               nElementBoundaries_element,
                                               nQuadraturePoints_elementBoundary,
                                               <proteus_index_t*>interiorElementBoundaries.data,
                                               <proteus_index_t*>elementBoundaryElements.data,
                                               <int *>elementBoundaryLocalElementBoundaries.data,
                                               <double*>phi.data,
                                               <double*>dphi.data,
                                               <double*>phi_trace.data,
                                               <double*>dphi_trace_left.data,
                                               <double*>dphi_trace_right.data)
def calculateExteriorNumericalTrace_Potential(np.ndarray isDOFBoundary,
                                              np.ndarray exteriorElementBoundaries,
                                              np.ndarray elementBoundaryElements,
                                              np.ndarray elementBoundaryLocalElementBoundaries,
                                              np.ndarray phi_bc,
                                              np.ndarray phi,
                                              np.ndarray dphi,
                                              np.ndarray phi_trace,
                                              np.ndarray dphi_trace_left):
    cdef int nExteriorElementBoundaries_global = exteriorElementBoundaries.shape[0]
    cdef int nElementBoundaries_element = phi.shape[1]
    cdef int nQuadraturePoints_elementBoundary = phi.shape[2]
    ccalculateExteriorNumericalTrace_Potential(<int *>isDOFBoundary.data,
                                               nExteriorElementBoundaries_global,
                                               nElementBoundaries_element,
                                               nQuadraturePoints_elementBoundary,
                                               <proteus_index_t*>exteriorElementBoundaries.data,
                                               <proteus_index_t*>elementBoundaryElements.data,
                                               <int *>elementBoundaryLocalElementBoundaries.data,
                                               <double*>phi_bc.data,
                                               <double*>phi.data,
                                               <double*>dphi.data,
                                               <double*>phi_trace.data,
                                               <double*>dphi_trace_left.data)
def updateInteriorElementBoundary_MixedForm_weak(np.ndarray interiorElementBoundaries,
                                                 np.ndarray elementBoundaryElements,
                                                 np.ndarray elementBoundaryLocalElementBoundaries,
                                                 np.ndarray n,
                                                 np.ndarray phi_trace,
                                                 np.ndarray w_dS,
                                                 np.ndarray b):
    cdef int nInteriorElementBoundaries_global = interiorElementBoundaries.shape[0]
    cdef int nElementBoundaries_element = w_dS.shape[1]
    cdef int nQuadraturePoints_elementBoundary = w_dS.shape[2]
    cdef int nDOF_test_element = w_dS.shape[3]
    cdef int nSpace = n.shape[3]
    cupdateInteriorElementBoundary_MixedForm_weak(nInteriorElementBoundaries_global,
                                                  nElementBoundaries_element,
                                                  nQuadraturePoints_elementBoundary,
                                                  nDOF_test_element,
                                                  nSpace,
                                                  <proteus_index_t*>interiorElementBoundaries.data,
                                                  <proteus_index_t*>elementBoundaryElements.data,
                                                  <int *>elementBoundaryLocalElementBoundaries.data,
                                                  <double*>n.data,
                                                  <double*>phi_trace.data,
                                                  <double*>w_dS.data,
                                                  <double*>b.data)
def updateInteriorElementBoundary_MixedForm_weakJacobian(np.ndarray interiorElementBoundaries,
                                                         np.ndarray elementBoundaryElements,
                                                         np.ndarray elementBoundaryLocalElementBoundaries,
                                                         np.ndarray n,
                                                         np.ndarray dphi_trace_left,
                                                         np.ndarray dphi_trace_right,
                                                         np.ndarray v,
                                                         np.ndarray w_dS,
                                                         np.ndarray db,
                                                         np.ndarray db_eb):
    cdef int nInteriorElementBoundaries_global = interiorElementBoundaries.shape[0]
    cdef int nElementBoundaries_element = v.shape[1]
    cdef int nQuadraturePoints_elementBoundary = v.shape[2]
    cdef int nDOF_test_element = v.shape[3]
    cdef int nSpace = n.shape[3]
    cupdateInteriorElementBoundary_MixedForm_weakJacobian(nInteriorElementBoundaries_global,
                                                          nElementBoundaries_element,
                                                          nQuadraturePoints_elementBoundary,
                                                          nDOF_test_element,
                                                          nSpace,
                                                          <proteus_index_t*>interiorElementBoundaries.data,
                                                          <proteus_index_t*>elementBoundaryElements.data,
                                                          <int *>elementBoundaryLocalElementBoundaries.data,
                                                          <double*>n.data,
                                                          <double*>dphi_trace_left.data,
                                                          <double*>dphi_trace_right.data,
                                                          <double*>v.data,
                                                          <double*>w_dS.data,
                                                          <double*>db.data,
                                                          <double*>db_eb.data)
def updateExteriorElementBoundary_MixedForm_weak(np.ndarray exteriorElementBoundaries,
                                                 np.ndarray elementBoundaryElements,
                                                 np.ndarray elementBoundaryLocalElementBoundaries,
                                                 np.ndarray n,
                                                 np.ndarray phi_trace,
                                                 np.ndarray w_dS,
                                                 np.ndarray b):
    cdef int nExteriorElementBoundaries_global = exteriorElementBoundaries.shape[0]
    cdef int nElementBoundaries_element = w_dS.shape[1]
    cdef int nQuadraturePoints_elementBoundary = w_dS.shape[2]
    cdef int nDOF_test_element = w_dS.shape[3]
    cdef int nSpace = n.shape[3]
    cupdateExteriorElementBoundary_MixedForm_weak(nExteriorElementBoundaries_global,
                                                  nElementBoundaries_element,
                                                  nQuadraturePoints_elementBoundary,
                                                  nDOF_test_element,
                                                  nSpace,
                                                  <proteus_index_t*>exteriorElementBoundaries.data,
                                                  <proteus_index_t*>elementBoundaryElements.data,
                                                  <int *>elementBoundaryLocalElementBoundaries.data,
                                                  <double*>n.data,
                                                  <double*>phi_trace.data,
                                                  <double*>w_dS.data,
                                                  <double*>b.data)
def updateExteriorElementBoundary_MixedForm_weakJacobian(np.ndarray exteriorElementBoundaries,
                                                         np.ndarray elementBoundaryElements,
                                                         np.ndarray elementBoundaryLocalElementBoundaries,
                                                         np.ndarray n,
                                                         np.ndarray dphi_trace_left,
                                                         np.ndarray v,
                                                         np.ndarray w_dS,
                                                         np.ndarray db,
                                                         np.ndarray db_eb):
    cdef int nExteriorElementBoundaries_global = exteriorElementBoundaries.shape[0]
    cdef int nElementBoundaries_element = v.shape[1]
    cdef int nQuadraturePoints_elementBoundary = v.shape[2]
    cdef int nDOF_test_element = v.shape[3]
    cdef int nSpace = n.shape[3]
    cupdateExteriorElementBoundary_MixedForm_weakJacobian(nExteriorElementBoundaries_global,
                                                          nElementBoundaries_element,
                                                          nQuadraturePoints_elementBoundary,
                                                          nDOF_test_element,
                                                          nSpace,
                                                          <proteus_index_t*>exteriorElementBoundaries.data,
                                                          <proteus_index_t*>elementBoundaryElements.data,
                                                          <int *>elementBoundaryLocalElementBoundaries.data,
                                                          <double*>n.data,
                                                          <double*>dphi_trace_left.data,
                                                          <double*>v.data,
                                                          <double*>w_dS.data,
                                                          <double*>db.data,
                                                          <double*>db_eb.data)
def updatePotential_MixedForm_weak(np.ndarray phi,
                                   np.ndarray grad_w_dV,
                                   np.ndarray b):
    cdef int nElements_global = grad_w_dV.shape[0]
    cdef int nQuadraturePoints_element = grad_w_dV.shape[1]
    cdef int nDOF_test_element = grad_w_dV.shape[2]
    cdef int nSpace = grad_w_dV.shape[3]
    cupdatePotential_MixedForm_weak(nElements_global,
                                    nQuadraturePoints_element,
                                    nDOF_test_element,
                                    nSpace,
                                    <double*>phi.data,
                                    <double*>grad_w_dV.data,
                                    <double*>b.data)
def updatePotential_MixedForm_weakJacobian(np.ndarray dphi,
                                           np.ndarray v,
                                           np.ndarray grad_w_dV,
                                           np.ndarray db):
    cdef int nElements_global = grad_w_dV.shape[0]
    cdef int nQuadraturePoints_element = grad_w_dV.shape[1]
    cdef int nDOF_test_element = grad_w_dV.shape[2]
    cdef int nSpace = grad_w_dV.shape[3]
    cupdatePotential_MixedForm_weakJacobian(nElements_global,
                                            nQuadraturePoints_element,
                                            nDOF_test_element,
                                            nSpace,
                                            <double*>dphi.data,
                                            <double*>v.data,
                                            <double*>grad_w_dV.data,
                                            <double*>db.data)
def calculateVelocityQuadrature_MixedForm(np.ndarray A_inv,
                                          np.ndarray b,
                                          np.ndarray v,
                                          np.ndarray V,
                                          np.ndarray qv,
                                          np.ndarray qV):
    cdef int nElements_global = v.shape[0]
    cdef int nElementBoundaries_element = v.shape[1]
    cdef int nElementBoundaryQuadraturePoints_elementBoundary = v.shape[2]
    cdef int nDOF_element = v.shape[3]
    cdef int nSpace = b.shape[1]
    cdef int nQuadraturePoints_element = qv.shape[1]
    ccalculateVelocityQuadrature_MixedForm(nElements_global,
                                           nElementBoundaries_element,
                                           nElementBoundaryQuadraturePoints_elementBoundary,
                                           nDOF_element,
                                           nSpace,
                                           nQuadraturePoints_element,
                                           <double*>A_inv.data,
                                           <double*>b.data,
                                           <double*>v.data,
                                           <double*>V.data,
                                           <double*>qv.data,
                                           <double*>qV.data)
def calculateVelocityQuadrature_MixedForm_Jacobian(np.ndarray A_inv,
                                                   np.ndarray db,
                                                   np.ndarray db_eb,
                                                   np.ndarray v,
                                                   np.ndarray DV,
                                                   np.ndarray DV_eb,
                                                   np.ndarray qv,
                                                   np.ndarray qDV,
                                                   np.ndarray qDV_eb):
    cdef int nElements_global = v.shape[0]
    cdef int nElementBoundaries_element = v.shape[1]
    cdef int nElementBoundaryQuadraturePoints_elementBoundary = v.shape[2]
    cdef int nDOF_element = v.shape[3]
    cdef int nSpace = db.shape[1]
    cdef int nQuadraturePoints_element = qv.shape[1]
    ccalculateVelocityQuadrature_MixedForm_Jacobian(nElements_global,
                                                    nElementBoundaries_element,
                                                    nElementBoundaryQuadraturePoints_elementBoundary,
                                                    nDOF_element,
                                                    nSpace,
                                                    nQuadraturePoints_element,
                                                    <double*>A_inv.data,
                                                    <double*>db.data,
                                                    <double*>db_eb.data,
                                                    <double*>v.data,
                                                    <double*>DV.data,
                                                    <double*>DV_eb.data,
                                                    <double*>qv.data,
                                                    <double*>qDV.data,
                                                    <double*>qDV_eb.data)
def calculateVelocityProjectionMatrixLDG(np.ndarray vXw_dV,
                                         np.ndarray A_inv):
    cdef int nElements_global = vXw_dV.shape[0]
    cdef int nQuadraturePoints_element = vXw_dV.shape[1]
    cdef int nDOF_element = vXw_dV.shape[2]
    ccalculateVelocityProjectionMatrixLDG(nElements_global,
                                          nQuadraturePoints_element,
                                          nDOF_element,
                                          <double*>vXw_dV.data,
                                          <double*>A_inv.data)
def updateDiffusion_MixedForm_weak(np.ndarray a,
                                   np.ndarray qV,
                                   np.ndarray grad_w_dV,
                                   np.ndarray weak_residual):
    cdef int nElements_global = grad_w_dV.shape[0]
    cdef int nQuadraturePoints_element = grad_w_dV.shape[1]
    cdef int nDOF_test_element = grad_w_dV.shape[2]
    cdef int nSpace = grad_w_dV.shape[3]
    cupdateDiffusion_MixedForm_weak(nElements_global,
                                    nQuadraturePoints_element,
                                    nDOF_test_element,
                                    nSpace,
                                    <double*>a.data,
                                    <double*>qV.data,
                                    <double*>grad_w_dV.data,
                                    <double*>weak_residual.data)
def updateDiffusionJacobian_MixedForm_weak(np.ndarray a,
                                           np.ndarray da,
                                           np.ndarray qV,
                                           np.ndarray qDV,
                                           np.ndarray qDV_eb,
                                           np.ndarray grad_w_dV,
                                           np.ndarray v,
                                           np.ndarray jacobian_weak_residual,
                                           np.ndarray jacobian_weak_residual_eb):
    cdef int nElements_global = qDV_eb.shape[0]
    cdef int nElementBoundaries_element = qDV_eb.shape[1]
    cdef int nQuadraturePoints_element = qDV_eb.shape[2]
    cdef int nDOF_trial_element = qDV_eb.shape[3]
    cdef int nDOF_test_element = grad_w_dV.shape[2]
    cdef int nSpace = grad_w_dV.shape[3]
    cupdateDiffusionJacobian_MixedForm_weak(nElements_global,
                                            nElementBoundaries_element,
                                            nQuadraturePoints_element,
                                            nDOF_trial_element,
                                            nDOF_test_element,
                                            nSpace,
                                            <double*>a.data,
                                            <double*>da.data,
                                            <double*>qV.data,
                                            <double*>qDV.data,
                                            <double*>qDV_eb.data,
                                            <double*>grad_w_dV.data,
                                            <double*>v.data,
                                            <double*>jacobian_weak_residual.data,
                                            <double*>jacobian_weak_residual_eb.data)
def estimate_mt(np.ndarray v,
                np.ndarray vXw_dV,
                np.ndarray elementSpatialResidual,
                np.ndarray mt):
    cdef int nElements_global = vXw_dV.shape[0]
    cdef int nQuadraturePoints_element = vXw_dV.shape[1]
    cdef int nDOF_element = vXw_dV.shape[2]
    cestimate_mt(nElements_global,
                 nQuadraturePoints_element,
                 nDOF_element,
                 <double*>v.data,
                 <double*>vXw_dV.data,
                 <double*>elementSpatialResidual.data,
                 <double*>mt.data)
def estimate_mt_lowmem(np.ndarray v,
                       np.ndarray w_dV,
                       np.ndarray elementSpatialResidual,
                       np.ndarray mt):
    cdef int nElements_global = w_dV.shape[0]
    cdef int nQuadraturePoints_element = w_dV.shape[1]
    cdef int nDOF_element = w_dV.shape[2]
    cestimate_mt_lowmem(nElements_global,
                        nQuadraturePoints_element,
                        nDOF_element,
                        <double*> v.data,
                        <double*> w_dV.data,
                        <double*> elementSpatialResidual.data,
                        <double*> mt.data)
def scalarDomainIntegral(np.ndarray dV,
                         np.ndarray nValueArray,
                         proteus_index_t nElements_global):
    cdef double output
    cdef int nQuadraturePoints_element = dV.shape[1]
    output = cscalarDomainIntegral(nElements_global,
                                   nQuadraturePoints_element,
                                   <double*> dV.data,
                                   <double*> nValueArray.data)
    return output
def scalarHeavisideDomainIntegral(proteus_index_t nElements_global,
                                  int nQuadraturePoints_element,
                                  np.ndarray dV,
                                  np.ndarray nValueArray):
    cdef double output
    output = cscalarHeavisideDomainIntegral(nElements_global,
                                            nQuadraturePoints_element,
                                            <double*> dV.data,
                                            <double*> nValueArray.data)
    return output
def scalarSmoothedHeavisideDomainIntegral(double epsFact,
					  np.ndarray elementDiameter,
					  np.ndarray dV,
					  np.ndarray nValueArray,
                                          proteus_index_t nElements_global):
    cdef double output
    cdef int nQuadraturePoints_element = dV.shape[1]
    output = cscalarSmoothedHeavisideDomainIntegral(nElements_global,
						    nQuadraturePoints_element,
						    epsFact,
						    <double*> elementDiameter.data,
						    <double*> dV.data,
						    <double*> nValueArray.data)
    return output
def fluxDomainBoundaryIntegral(proteus_index_t nElementBoundaries_owned,
                               np.ndarray flag,
                               np.ndarray exteriorElementBoundariesArray,
                               np.ndarray dS,
                               np.ndarray nValueArray):
    nExteriorElementBoundaries = nValueArray.shape[0]
    nQuadraturePoints_elementBoundary = nValueArray.shape[1]
    cdef double output
    output = cfluxDomainBoundaryIntegral(nExteriorElementBoundaries,
				         nElementBoundaries_owned,
                                         nQuadraturePoints_elementBoundary,
                                         <int*> flag.data,
                                         <proteus_index_t*> exteriorElementBoundariesArray.data,
                                         <double*> dS.data,
                                         <double*> nValueArray.data)
    return output
def fluxDomainBoundaryIntegralFromVector(proteus_index_t nElementBoundaries_owned,
					 np.ndarray flag,
					 np.ndarray exteriorElementBoundaries,
					 np.ndarray dS,
					 np.ndarray nValueArray,
					 np.ndarray normal):
    cdef double output
    cdef int nExteriorElementBoundaries = normal.shape[0]
    cdef int nQuadraturePoints_elementBoundary = normal.shape[1]
    cdef int nSpace = normal.shape[2]
    output = cfluxDomainBoundaryIntegralFromVector(nExteriorElementBoundaries,
						   nElementBoundaries_owned,
						   nQuadraturePoints_elementBoundary,
						   nSpace,
						   <int*> flag.data,
						   <proteus_index_t*> exteriorElementBoundaries.data,
						   <double*> dS.data,
						   <double*> nValueArray.data,
						   <double*> normal.data)
    return output
def copyExteriorElementBoundaryValuesFromElementBoundaryValues(np.ndarray exteriorElementBoundaries,
							       np.ndarray elementBoundaryElements,
							       np.ndarray elementBoundaryLocalElementBoundaries,
							       np.ndarray  ebq_val,
							       np.ndarray  ebqe_val):
    cdef int nExteriorElementBoundaries_global = exteriorElementBoundaries.shape[0]
    cdef int nElements_global = ebq_val.shape[0]
    cdef int nElementBoundaries_element = ebq_val.shape[1]
    cdef int nQuadraturePoints_elementBoundary = ebq_val.shape[2]
    cdef int nValuesPerQuadraturePoint = 1
    cdef int nd = ebq_val.ndim
    if nd > 3:
        for i in range(3,nd):
            nValuesPerQuadraturePoint *= ebq_val.shape[i]
    ccopyExteriorElementBoundaryValuesFromElementBoundaryValues(nExteriorElementBoundaries_global,
							        nElements_global,
							        nElementBoundaries_element,
							        nQuadraturePoints_elementBoundary,
							        nValuesPerQuadraturePoint,
							        <proteus_index_t*> exteriorElementBoundaries.data,
							        <proteus_index_t*> elementBoundaryElements.data,
							        <int*> elementBoundaryLocalElementBoundaries.data,
							        <double*> ebq_val.data,
							        <double*> ebqe_val.data)
def copyExteriorElementBoundaryValuesToElementBoundaryValues(np.ndarray exteriorElementBoundaries,
							     np.ndarray elementBoundaryElements,
							     np.ndarray elementBoundaryLocalElementBoundaries,
							     np.ndarray  ebqe_val,
							     np.ndarray  ebq_val):
    cdef int nExteriorElementBoundaries_global = exteriorElementBoundaries.shape[0]
    cdef int nElements_global = ebq_val.shape[0]
    cdef int nElementBoundaries_element = ebq_val.shape[1]
    cdef int nQuadraturePoints_elementBoundary = ebq_val.shape[2]
    cdef int nValuesPerQuadraturePoint = 1
    cdef int nd = ebq_val.ndim
    if nd > 3:
        for i in range(3,nd):
            nValuesPerQuadraturePoint *= ebq_val.shape[i]
    ccopyExteriorElementBoundaryValuesToElementBoundaryValues(nExteriorElementBoundaries_global,
							      nElements_global,
							      nElementBoundaries_element,
							      nQuadraturePoints_elementBoundary,
							      nValuesPerQuadraturePoint,
							      <proteus_index_t*> exteriorElementBoundaries.data,
							      <proteus_index_t*> elementBoundaryElements.data,
							      <int*> elementBoundaryLocalElementBoundaries.data,
							      <double*> ebqe_val.data,
							      <double*> ebq_val.data)
def copyExteriorElementBoundaryValuesToGlobalElementBoundaryValues(np.ndarray exteriorElementBoundaries,
								   np.ndarray elementBoundaryElements,
								   np.ndarray elementBoundaryLocalElementBoundaries,
								   np.ndarray ebqe_val,
								   np.ndarray ebq_global_val):
    cdef int nExteriorElementBoundaries_global = exteriorElementBoundaries.shape[0]
    cdef int nQuadraturePoints_elementBoundary = ebq_global_val.shape[1]
    cdef int nValuesPerQuadraturePoint = 1
    cdef int nd = ebq_global_val.ndim
    if nd > 2:
        for i in range(2,nd):
            nValuesPerQuadraturePoint *= ebq_global_val.shape[i]
    ccopyExteriorElementBoundaryValuesToGlobalElementBoundaryValues(nExteriorElementBoundaries_global,
								    nQuadraturePoints_elementBoundary,
								    nValuesPerQuadraturePoint,
								    <proteus_index_t*> exteriorElementBoundaries.data,
								    <proteus_index_t*> elementBoundaryElements.data,
								    <int*> elementBoundaryLocalElementBoundaries.data,
								    <double*> ebqe_val.data,
								    <double*> ebq_global_val.data)
def copyExteriorElementBoundaryValuesFromGlobalElementBoundaryValues(np.ndarray exteriorElementBoundaries,
								     np.ndarray elementBoundaryElements,
								     np.ndarray elementBoundaryLocalElementBoundaries,
								     np.ndarray ebq_global_val,
								     np.ndarray ebqe_val):
    cdef int nExteriorElementBoundaries_global = exteriorElementBoundaries.shape[0]
    cdef int nQuadraturePoints_elementBoundary = ebq_global_val.shape[1]
    cdef int nValuesPerQuadraturePoint = 1;
    cdef int nd = ebq_global_val.ndim
    if nd > 2:
        for i in range(2,nd):
            nValuesPerQuadraturePoint *= ebq_global_val.shape[i]
    ccopyExteriorElementBoundaryValuesFromGlobalElementBoundaryValues(nExteriorElementBoundaries_global,
								      nQuadraturePoints_elementBoundary,
								      nValuesPerQuadraturePoint,
								      <proteus_index_t*> exteriorElementBoundaries.data,
								      <proteus_index_t*> elementBoundaryElements.data,
								      <int*> elementBoundaryLocalElementBoundaries.data,
								      <double*> ebq_global_val.data,
								      <double*> ebqe_val.data)
def computeC0P1InterpolantDGP0(np.ndarray elementNodesArray,
			       np.ndarray nodeElementOffsets,
			       np.ndarray nodeElementsArray,
			       np.ndarray l2g,
			       np.ndarray  dof,
			       np.ndarray nodalAverage,
                               dim_dof):
    cdef int nElements_global = elementNodesArray.shape[0]
    cdef int nNodes_global = nodeElementOffsets.shape[0]-1
    cdef int nNodes_element = elementNodesArray.shape[1]
    cdef int nDOF_element = l2g.shape[1]
    ccomputeC0P1InterpolantDGP0(nElements_global,
			        nNodes_global,
			        nNodes_element,
			        nDOF_element,
			        dim_dof,
			        <proteus_index_t*> elementNodesArray.data,
			        <proteus_index_t*> nodeElementOffsets.data,
			        <proteus_index_t*> nodeElementsArray.data,
			        <proteus_index_t*> l2g.data,
			        <double*> dof.data,
			        <double*> nodalAverage.data)
def computeC0P1InterpolantNCP1(int dim_dof,
			       np.ndarray elementNodesArray,
			       np.ndarray nodeElementOffsets,
			       np.ndarray nodeElementsArray,
			       np.ndarray l2g,
			       np.ndarray  dof,
			       np.ndarray nodalAverage):
    cdef int nElements_global = elementNodesArray.shape[0]
    cdef int nNodes_global = nodeElementOffsets.shape[0]-1
    cdef int nNodes_element = elementNodesArray.shape[1]
    cdef int nDOF_element = l2g.shape[1]
    ccomputeC0P1InterpolantNCP1(nElements_global,
			        nNodes_global,
			        nNodes_element,
			        nDOF_element,
			        dim_dof,
			        <proteus_index_t*> elementNodesArray.data,
			        <proteus_index_t*> nodeElementOffsets.data,
			        <proteus_index_t*> nodeElementsArray.data,
			        <proteus_index_t*> l2g.data,
			        <double*> dof.data,
			        <double*> nodalAverage.data)
def computeC0P1InterpolantDGP12(int dim_dof,
				np.ndarray elementNodesArray,
				np.ndarray nodeElementOffsets,
				np.ndarray nodeElementsArray,
				np.ndarray l2g,
				np.ndarray  dof,
				np.ndarray nodalAverage):
    cdef int nElements_global = elementNodesArray.shape[0]
    cdef int nNodes_global = nodeElementOffsets.shape[0]-1
    cdef int nNodes_element = elementNodesArray.shape[1]
    cdef int nDOF_element = l2g.shape[1]
    ccomputeC0P1InterpolantDGP12(nElements_global,
				 nNodes_global,
				 nNodes_element,
				 nDOF_element,
				 dim_dof,
				 <proteus_index_t*> elementNodesArray.data,
				 <proteus_index_t*> nodeElementOffsets.data,
				 <proteus_index_t*> nodeElementsArray.data,
				 <proteus_index_t*> l2g.data,
				 <double*> dof.data,
				 <double*> nodalAverage.data)
def parametricFiniteElementSpace_getValuesGlobalExteriorTrace(np.ndarray exteriorElementBoundariesArray,
							      np.ndarray elementBoundaryElementsArray,
							      np.ndarray elementBoundaryLocalElementBoundariesArray,
							      np.ndarray psi,
							      np.ndarray vArray):
    cdef int nElementBoundaries_element = psi.shape[0]
    cdef int nElementBoundaryQuadraturePoints_elementBoundary = psi.shape[1]
    cdef int nDOF_element = psi.shape[2]
    cdef int nExteriorElementBoundaries_global = exteriorElementBoundariesArray.shape[0]
    cparametricFiniteElementSpace_getValuesGlobalExteriorTrace(nElementBoundaries_element,
							       nElementBoundaryQuadraturePoints_elementBoundary,
							       nDOF_element,
							       nExteriorElementBoundaries_global,
							       <proteus_index_t*> exteriorElementBoundariesArray.data,
							       <proteus_index_t*> elementBoundaryElementsArray.data,
							       <int*> elementBoundaryLocalElementBoundariesArray.data,
							       <double*> psi.data,
							       <double*> vArray.data)
def parametricFiniteElementSpace_getGradientValuesGlobalExteriorTrace(np.ndarray exteriorElementBoundariesArray,
								      np.ndarray elementBoundaryElementsArray,
								      np.ndarray elementBoundaryLocalElementBoundariesArray,
								      np.ndarray grad_psi,
								      np.ndarray inverseJacobianArray,
								      np.ndarray grad_vArray):
    cdef int nElementBoundaries_element = grad_psi.shape[0]
    cdef int nElementBoundaryQuadraturePoints_elementBoundary = grad_psi.shape[1]
    cdef int nDOF_element = grad_psi.shape[2]
    cdef int nSpace_global = grad_vArray.shape[3]
    cdef int nExteriorElementBoundaries_global = exteriorElementBoundariesArray.shape[0]
    cparametricFiniteElementSpace_getGradientValuesGlobalExteriorTrace(nElementBoundaries_element,
								       nElementBoundaryQuadraturePoints_elementBoundary,
								       nDOF_element,
								       nSpace_global,
								       nExteriorElementBoundaries_global,
								       <proteus_index_t*>exteriorElementBoundariesArray.data,
								       <proteus_index_t*>elementBoundaryElementsArray.data,
								       <int *>elementBoundaryLocalElementBoundariesArray.data,
								       <double*> grad_psi.data,
								       <double*> inverseJacobianArray.data,
								       <double*> grad_vArray.data)
def parametricMaps_getValuesGlobalExteriorTrace(np.ndarray exteriorElementBoundariesArray,
						np.ndarray elementBoundaryElementsArray,
						np.ndarray elementBoundaryLocalElementBoundariesArray,
						np.ndarray psi,
						np.ndarray l2g,
						np.ndarray nodeArray,
						np.ndarray xArray):
    cdef int nQuadraturePoints_elementBoundary = xArray.shape[1]
    cdef int nDOF_element = l2g.shape[1]
    cdef int nSpace_global = xArray.shape[2]
    cdef int nExteriorElementBoundaries_global = exteriorElementBoundariesArray.shape[0]
    cparametricMaps_getValuesGlobalExteriorTrace(nQuadraturePoints_elementBoundary,
						 nDOF_element,
						 nSpace_global,
						 nExteriorElementBoundaries_global,
						 <proteus_index_t*> exteriorElementBoundariesArray.data,
						 <proteus_index_t*> elementBoundaryElementsArray.data,
						 <int*> elementBoundaryLocalElementBoundariesArray.data,
						 <double*> psi.data,
						 <proteus_index_t*> l2g.data,
						 <double*> nodeArray.data,
						 <double*> xArray.data)
def parametricMaps_getInverseValuesGlobalExteriorTrace(np.ndarray exteriorElementBoundariesArray,
						       np.ndarray elementBoundaryElementsArray,
						       np.ndarray elementBoundaryLocalElementBoundariesArray,
						       np.ndarray inverseJacobian,
						       np.ndarray l2g,
						       np.ndarray nodeArray,
						       np.ndarray xArray,
						       np.ndarray xiArray):
    cdef int nElementBoundaryQuadraturePoints_elementBoundary = xArray.shape[1]
    cdef int nDOF_element = l2g.shape[1]
    cdef int nSpace_global = inverseJacobian.shape[3]
    cdef int nExteriorElementBoundaries_global = exteriorElementBoundariesArray.shape[0]
    cparametricMaps_getInverseValuesGlobalExteriorTrace(nElementBoundaryQuadraturePoints_elementBoundary,
						        nDOF_element,
						        nSpace_global,
						        nExteriorElementBoundaries_global,
						        <proteus_index_t*> exteriorElementBoundariesArray.data,
						        <proteus_index_t*> elementBoundaryElementsArray.data,
						        <int*> elementBoundaryLocalElementBoundariesArray.data,
						        <double*> inverseJacobian.data,
						        <proteus_index_t*> l2g.data,
						        <double*> nodeArray.data,
						        <double*> xArray.data,
						        <double*> xiArray.data)
def parametricMaps_getJacobianValuesGlobalExteriorTrace(np.ndarray exteriorElementBoundariesArray,
							np.ndarray elementBoundaryElementsArray,
							np.ndarray elementBoundaryLocalElementBoundariesArray,
							np.ndarray grad_psi,
							np.ndarray boundaryNormals,
							np.ndarray boundaryJacobians,
							np.ndarray l2g,
							np.ndarray nodeArray,
							np.ndarray jacobianInverseArray,
							np.ndarray metricTensorArray,
							np.ndarray metricTensorDeterminantSqrtArray,
							np.ndarray unitNormalArray):
    cdef int nd = jacobianInverseArray.shape[2]
    if nd == 1:
        cparametricMaps_getJacobianValuesGlobalExteriorTrace1D(jacobianInverseArray.shape[1],
                                                               l2g.shape[1],
                                                               exteriorElementBoundariesArray.shape[0],
							       <proteus_index_t*> exteriorElementBoundariesArray.data,
							       <proteus_index_t*> elementBoundaryElementsArray.data,
							       <int *> elementBoundaryLocalElementBoundariesArray.data,
							       <double*> grad_psi.data,
							       <double*> boundaryNormals.data,
							       <double*> boundaryJacobians.data,
							       <proteus_index_t*> l2g.data,
							       <double*> nodeArray.data,
							       <double*> jacobianInverseArray.data,
							       <double*> metricTensorArray.data,
							       <double*> metricTensorDeterminantSqrtArray.data,
							       <double*> unitNormalArray.data)
    elif nd == 2:
        cparametricMaps_getJacobianValuesGlobalExteriorTrace2D(jacobianInverseArray.shape[1],
                                                               l2g.shape[1],
                                                               exteriorElementBoundariesArray.shape[0],
							       <proteus_index_t*> exteriorElementBoundariesArray.data,
							       <proteus_index_t*> elementBoundaryElementsArray.data,
							       <int *> elementBoundaryLocalElementBoundariesArray.data,
							       <double*> grad_psi.data,
							       <double*> boundaryNormals.data,
							       <double*> boundaryJacobians.data,
							       <proteus_index_t*> l2g.data,
							       <double*> nodeArray.data,
							       <double*> jacobianInverseArray.data,
							       <double*> metricTensorArray.data,
							       <double*> metricTensorDeterminantSqrtArray.data,
							       <double*> unitNormalArray.data)
    elif nd == 3:
        cparametricMaps_getJacobianValuesGlobalExteriorTrace3D(jacobianInverseArray.shape[1],
                                                               l2g.shape[1],
                                                               exteriorElementBoundariesArray.shape[0],
							       <proteus_index_t*> exteriorElementBoundariesArray.data,
							       <proteus_index_t*> elementBoundaryElementsArray.data,
							       <int *> elementBoundaryLocalElementBoundariesArray.data,
							       <double*> grad_psi.data,
							       <double*> boundaryNormals.data,
							       <double*> boundaryJacobians.data,
							       <proteus_index_t*> l2g.data,
							       <double*> nodeArray.data,
							       <double*> jacobianInverseArray.data,
							       <double*> metricTensorArray.data,
							       <double*> metricTensorDeterminantSqrtArray.data,
							       <double*> unitNormalArray.data)
    else:
        print("error in getJacobianValuesTrace...jacobianInverse not sized properly")
def parametricMaps_getJacobianValuesGlobalExteriorTrace_movingDomain(np.ndarray exteriorElementBoundariesArray,
								     np.ndarray elementBoundaryElementsArray,
								     np.ndarray elementBoundaryLocalElementBoundariesArray,
								     np.ndarray xtArray,
								     np.ndarray grad_psi,
								     np.ndarray boundaryNormals,
								     np.ndarray boundaryJacobians,
								     np.ndarray l2g,
								     np.ndarray nodeArray,
								     np.ndarray jacobianInverseArray,
								     np.ndarray metricTensorArray,
								     np.ndarray metricTensorDeterminantSqrtArray,
								     np.ndarray unitNormalArray):
    cdef int nd = jacobianInverseArray.shape[2]
    if nd == 1:
        cparametricMaps_getJacobianValuesGlobalExteriorTrace1D(jacobianInverseArray.shape[1],
                                                               l2g.shape[1],
                                                               exteriorElementBoundariesArray.shape[0],
							       <proteus_index_t*> exteriorElementBoundariesArray.data,
							       <proteus_index_t*> elementBoundaryElementsArray.data,
							       <int*> elementBoundaryLocalElementBoundariesArray.data,
							       <double*> grad_psi.data,
							       <double*> boundaryNormals.data,
							       <double*> boundaryJacobians.data,
							       <proteus_index_t*> l2g.data,
							       <double*> nodeArray.data,
							       <double*> jacobianInverseArray.data,
							       <double*> metricTensorArray.data,
							       <double*> metricTensorDeterminantSqrtArray.data,
							       <double*> unitNormalArray.data)
    elif nd == 2:
        cparametricMaps_getJacobianValuesGlobalExteriorTrace2D_movingDomain(jacobianInverseArray.shape[1],
                                                                            l2g.shape[1],
                                                                            exteriorElementBoundariesArray.shape[0],
								            <proteus_index_t*> exteriorElementBoundariesArray.data,
								            <proteus_index_t*> elementBoundaryElementsArray.data,
								            <int*> elementBoundaryLocalElementBoundariesArray.data,
								            <double*> xtArray.data,
								            <double*> grad_psi.data,
								            <double*> boundaryNormals.data,
								            <double*> boundaryJacobians.data,
								            <proteus_index_t*> l2g.data,
								            <double*> nodeArray.data,
								            <double*> jacobianInverseArray.data,
								            <double*> metricTensorArray.data,
								            <double*> metricTensorDeterminantSqrtArray.data,
								            <double*> unitNormalArray.data)
    elif nd == 3:
        cparametricMaps_getJacobianValuesGlobalExteriorTrace3D(jacobianInverseArray.shape[1],
                                                               l2g.shape[1],
                                                               exteriorElementBoundariesArray.shape[0],
							       <proteus_index_t*> exteriorElementBoundariesArray.data,
							       <proteus_index_t*> elementBoundaryElementsArray.data,
							       <int*> elementBoundaryLocalElementBoundariesArray.data,
							       <double*> grad_psi.data,
							       <double*> boundaryNormals.data,
							       <double*> boundaryJacobians.data,
							       <proteus_index_t*> l2g.data,
							       <double*> nodeArray.data,
							       <double*> jacobianInverseArray.data,
							       <double*> metricTensorArray.data,
							       <double*> metricTensorDeterminantSqrtArray.data,
							       <double*> unitNormalArray.data)
    else:
        print("error in getJacobianValuesTrace...jacobianInverse not sized properly")
def calculateWeightedShapeGlobalExteriorTrace(np.ndarray exteriorElementBoundariesArray,
					      np.ndarray elementBoundaryElementsArray,
					      np.ndarray elementBoundaryLocalElementBoundariesArray,
					      np.ndarray dSR,
					      np.ndarray sqrt_det_g,
					      np.ndarray w,
					      np.ndarray w_dS):
    cdef int nElementBoundaryQuadraturePoints_elementBoundary = w_dS.shape[1]
    cdef int nDOF_test_element = w_dS.shape[2]
    cdef int nExteriorElementBoundaries_global = exteriorElementBoundariesArray.shape[0]
    ccalculateWeightedShapeGlobalExteriorTrace(nElementBoundaryQuadraturePoints_elementBoundary,
					       nDOF_test_element,
					       nExteriorElementBoundaries_global,
					       <proteus_index_t*> exteriorElementBoundariesArray.data,
					       <proteus_index_t*> elementBoundaryElementsArray.data,
					       <int*> elementBoundaryLocalElementBoundariesArray.data,
					       <double*> dSR.data,
					       <double*> sqrt_det_g.data,
					       <double*> w.data,
					       <double*> w_dS.data)
def calculateShape_X_weightedShapeGlobalExteriorTrace(np.ndarray exteriorElementBoundariesArray,
						      np.ndarray elementBoundaryElementsArray,
						      np.ndarray elementBoundaryLocalElementBoundariesArray,
						      np.ndarray v,
						      np.ndarray w_dS,
						      np.ndarray v_X_w_dS):
    cdef int nElementBoundaryQuadraturePoints_elementBoundary = v_X_w_dS.shape[1]
    cdef int nDOF_trial_element = v_X_w_dS.shape[2]
    cdef int nDOF_test_element = v_X_w_dS.shape[3]
    cdef int nExteriorElementBoundaries_global = exteriorElementBoundariesArray.shape[0]
    ccalculateShape_X_weightedShapeGlobalExteriorTrace(nElementBoundaryQuadraturePoints_elementBoundary,
						       nDOF_trial_element,
						       nDOF_test_element,
						       nExteriorElementBoundaries_global,
						       <proteus_index_t*> exteriorElementBoundariesArray.data,
						       <proteus_index_t*> elementBoundaryElementsArray.data,
						       <int*> elementBoundaryLocalElementBoundariesArray.data,
						       <double*> v.data,
						       <double*> w_dS.data,
						       <double*> v_X_w_dS.data)
def calculateGradShape_X_weightedShapeGlobalExteriorTrace(np.ndarray exteriorElementBoundariesArray,
							  np.ndarray elementBoundaryElementsArray,
							  np.ndarray elementBoundaryLocalElementBoundariesArray,
							  np.ndarray grad_v,
							  np.ndarray w_dS,
							  np.ndarray grad_v_X_w_dS):
    cdef int nElementBoundaryQuadraturePoints_elementBoundary = grad_v_X_w_dS.shape[1]
    cdef int nDOF_trial_element = grad_v_X_w_dS.shape[2]
    cdef int nDOF_test_element = grad_v_X_w_dS.shape[3]
    cdef int nSpace = grad_v_X_w_dS.shape[4]
    cdef int nExteriorElementBoundaries_global = exteriorElementBoundariesArray.shape[0]
    ccalculateGradShape_X_weightedShapeGlobalExteriorTrace(nElementBoundaryQuadraturePoints_elementBoundary,
							   nDOF_trial_element,
							   nDOF_test_element,
							   nSpace,
							   nExteriorElementBoundaries_global,
							   <proteus_index_t*> exteriorElementBoundariesArray.data,
							   <proteus_index_t*> elementBoundaryElementsArray.data,
							   <int*> elementBoundaryLocalElementBoundariesArray.data,
							   <double*> grad_v.data,
							   <double*> w_dS.data,
							   <double*> grad_v_X_w_dS.data)
def calculateGlobalExteriorElementBoundaryIntegrationWeights(int nQuadraturePoints_elementBoundary,
							     proteus_index_t nExteriorElementBoundaries_global,
							     np.ndarray sqrt_det_g,
							     np.ndarray referenceWeights,
							     np.ndarray weights):
    ccalculateGlobalExteriorElementBoundaryIntegrationWeights(nQuadraturePoints_elementBoundary,
							      nExteriorElementBoundaries_global,
							      <double*> sqrt_det_g.data,
							      <double*> referenceWeights.data,
							      <double*> weights.data)
def calculateFiniteElementFunctionValuesGlobalExteriorTrace(np.ndarray exteriorElementBoundariesArray,
							    np.ndarray elementBoundaryElementsArray,
							    np.ndarray elementBoundaryLocalElementBoundariesArray,
							    np.ndarray l2g,
							    np.ndarray dof,
							    np.ndarray v,
							    np.ndarray u):
    cdef int nQuadraturePoints_elementBoundary = v.shape[1]
    cdef int nDOF_trial_element = v.shape[2]
    cdef int nExteriorElementBoundaries_global = exteriorElementBoundariesArray.shape[0]
    cdef int nComponents = 1
    cdef int nd = u.ndim
    if nd == 4:
        nComponents = u.shape[3]
    else:
        nComponents = 1
    ccalculateFiniteElementFunctionValuesGlobalExteriorTrace(nQuadraturePoints_elementBoundary,
							     nDOF_trial_element,
							     nComponents,
							     nExteriorElementBoundaries_global,
							     <proteus_index_t*> exteriorElementBoundariesArray.data,
							     <proteus_index_t*> elementBoundaryElementsArray.data,
							     <int *> elementBoundaryLocalElementBoundariesArray.data,
							     <proteus_index_t*> l2g.data,
							     <double*> dof.data,
							     <double*> v.data,
							     <double*> u.data)
def calculateFiniteElementFunctionGradientValuesGlobalExteriorTrace(np.ndarray exteriorElementBoundariesArray,
								    np.ndarray elementBoundaryElementsArray,
								    np.ndarray elementBoundaryLocalElementBoundariesArray,
								    np.ndarray l2g,
								    np.ndarray dof,
								    np.ndarray grad_v,
								    np.ndarray grad_u):
    cdef int nQuadraturePoints_elementBoundary = grad_v.shape[1]
    cdef int nDOF_trial_element = grad_v.shape[2]
    cdef int nExteriorElementBoundaries_global = exteriorElementBoundariesArray.shape[0]
    cdef int nComponents = 1
    cdef int nSpace = 1
    cdef int nd = grad_u.ndim
    if nd == 4:
        nComponents = grad_u.shape[2]
        nSpace = grad_u.shape[3]
    else:
        nComponents = 1
        nSpace = grad_u.shape[2]
    ccalculateFiniteElementFunctionGradientValuesGlobalExteriorTrace(nQuadraturePoints_elementBoundary,
								     nDOF_trial_element,
								     nComponents,
								     nSpace,
								     nExteriorElementBoundaries_global,
								     <proteus_index_t*> exteriorElementBoundariesArray.data,
								     <proteus_index_t*> elementBoundaryElementsArray.data,
								     <int *> elementBoundaryLocalElementBoundariesArray.data,
								     <proteus_index_t*> l2g.data,
								     <double*> dof.data,
								     <double*> grad_v.data,
								     <double*> grad_u.data)
def copyBetweenFreeUnknownsAndGlobalUnknowns(proteus_index_t nDOF2set,
				             proteus_index_t offset,
				             proteus_index_t stride,
				             np.ndarray globalDOFids,
				             np.ndarray freeDOFids,
				             np.ndarray  free_u,
				             np.ndarray  u):
    if nDOF2set > 0:
        ccopyFreeUnknownsToGlobalUnknowns(globalDOFids.shape[0],
				          offset,
				          stride,
				          <proteus_index_t*> globalDOFids.data,
				          <proteus_index_t*> freeDOFids.data,
				          <double*> free_u.data,
				          <double*> u.data)
    elif nDOF2set == 0:
        ccopyGlobalUnknownsToFreeUnknowns(globalDOFids.shape[0],
				          offset,
				          stride,
				          <proteus_index_t*> globalDOFids.data,
				          <proteus_index_t*> freeDOFids.data,
				          <double*> u.data,
				          <double*> free_u.data)
    else:
        print("error copyFromFreeToGlobal = ", nDOF2set, " not recognized quitting\n")
def updateInteriorElementBoundaryDiffusionAdjoint(np.ndarray interiorElementBoundaries,
						  np.ndarray elementBoundaryElements,
						  np.ndarray elementBoundaryLocalElementBoundaries,
						  double sigma,
						  np.ndarray u,
						  np.ndarray n,
						  np.ndarray a,
						  np.ndarray grad_w,
						  np.ndarray dS,
						  np.ndarray residual):
    cdef int nInteriorElementBoundaries_global = interiorElementBoundaries.shape[0]
    cdef int nElementBoundaries_element = grad_w.shape[1]
    cdef int nQuadraturePoints_elementBoundary = grad_w.shape[2]
    cdef int nDOF_test_element = grad_w.shape[3]
    cdef int nSpace = grad_w.shape[4]
    cupdateInteriorElementBoundaryDiffusionAdjoint(nInteriorElementBoundaries_global,
						   nElementBoundaries_element,
						   nQuadraturePoints_elementBoundary,
						   nDOF_test_element,
						   nSpace,
						   <proteus_index_t*> interiorElementBoundaries.data,
						   <proteus_index_t*> elementBoundaryElements.data,
						   <int*> elementBoundaryLocalElementBoundaries.data,
						   sigma,
						   <double*> u.data,
						   <double*> n.data,
						   <double*> a.data,
						   <double*> grad_w.data,
						   <double*> dS.data,
						   <double*> residual.data)
def updateExteriorElementBoundaryDiffusionAdjoint(np.ndarray isDOFBoundary,
						  np.ndarray exteriorElementBoundaries,
						  np.ndarray elementBoundaryElements,
						  np.ndarray elementBoundaryLocalElementBoundaries,
						  double sigma,
						  np.ndarray u,
						  np.ndarray ub,
						  np.ndarray n,
						  np.ndarray a,
						  np.ndarray grad_w,
						  np.ndarray dS,
						  np.ndarray residual):
    cdef int nExteriorElementBoundaries_global = exteriorElementBoundaries.shape[0]
    cdef int nQuadraturePoints_elementBoundary = grad_w.shape[1]
    cdef int nDOF_test_element = grad_w.shape[2]
    cdef int nSpace = grad_w.shape[3]
    cupdateExteriorElementBoundaryDiffusionAdjoint(nExteriorElementBoundaries_global,
						   nQuadraturePoints_elementBoundary,
						   nDOF_test_element,
						   nSpace,
						   <int*> isDOFBoundary.data,
						   <proteus_index_t*> exteriorElementBoundaries.data,
						   <proteus_index_t*> elementBoundaryElements.data,
						   <int*> elementBoundaryLocalElementBoundaries.data,
						   sigma,
						   <double*> u.data,
						   <double*> ub.data,
						   <double*> n.data,
						   <double*> a.data,
						   <double*> grad_w.data,
						   <double*> dS.data,
						   <double*> residual.data)
def updateGlobalJacobianFromInteriorElementBoundaryDiffusionAdjoint_dense(proteus_index_t offset_r,
									  proteus_index_t stride_r,
									  proteus_index_t offset_u,
									  proteus_index_t stride_u,
									  proteus_index_t nFreeVDOF_global,
									  np.ndarray interiorElementBoundaries,
									  np.ndarray elementBoundaryElements,
									  np.ndarray elementBoundaryLocalElementBoundaries,
									  np.ndarray nFreeDOF_element_r,
									  np.ndarray freeLocal_r,
									  np.ndarray freeGlobal_r,
									  np.ndarray nFreeDOF_element_u,
									  np.ndarray freeLocal_u,
									  np.ndarray freeGlobal_u,
									  double sigma,
									  np.ndarray v,
									  np.ndarray n,
									  np.ndarray a,
									  np.ndarray grad_w,
									  np.ndarray dS,
									  np.ndarray jac):
    cdef int nInteriorElementBoundaries_global = interiorElementBoundaries.shape[0]
    cdef int nElementBoundaries_element = grad_w.shape[1]
    cdef int nQuadraturePoints_elementBoundary = grad_w.shape[2]
    cdef int nDOF_test_element = grad_w.shape[3]
    cdef int nDOF_trial_element = v.shape[3]
    cdef int nSpace = n.shape[3]
    cupdateGlobalJacobianFromInteriorElementBoundaryDiffusionAdjoint_dense(nInteriorElementBoundaries_global,
									   nElementBoundaries_element,
									   nQuadraturePoints_elementBoundary,
									   nDOF_test_element,
									   nDOF_trial_element,
									   nSpace,
									   offset_r,
									   stride_r,
									   offset_u,
									   stride_u,
									   nFreeVDOF_global,
									   <proteus_index_t*> interiorElementBoundaries.data,
									   <proteus_index_t*> elementBoundaryElements.data,
									   <int*> elementBoundaryLocalElementBoundaries.data,
									   <int*> nFreeDOF_element_r.data,
									   <int*> freeLocal_r.data,
									   <proteus_index_t*> freeGlobal_r.data,
									   <int*> nFreeDOF_element_u.data,
									   <int*> freeLocal_u.data,
									   <proteus_index_t*> freeGlobal_u.data,
									   sigma,
									   <double*> v.data,
									   <double*> n.data,
									   <double*> a.data,
									   <double*> grad_w.data,
									   <double*> dS.data,
									   <double*> jac.data)
def updateGlobalJacobianFromExteriorElementBoundaryDiffusionAdjoint_dense(proteus_index_t offset_r,
									  proteus_index_t stride_r,
									  proteus_index_t offset_u,
									  proteus_index_t stride_u,
									  proteus_index_t nFreeVDOF_global,
									  np.ndarray exteriorElementBoundaries,
									  np.ndarray elementBoundaryElements,
									  np.ndarray elementBoundaryLocalElementBoundaries,
									  np.ndarray nFreeDOF_element_r,
									  np.ndarray freeLocal_r,
									  np.ndarray freeGlobal_r,
									  np.ndarray nFreeDOF_element_u,
									  np.ndarray freeLocal_u,
									  np.ndarray freeGlobal_u,
									  np.ndarray isDOFBoundary,
									  double sigma,
									  np.ndarray v,
									  np.ndarray n,
									  np.ndarray a,
									  np.ndarray grad_w,
									  np.ndarray dS,
									  np.ndarray jac):
    cdef int nExteriorElementBoundaries_global = exteriorElementBoundaries.shape[0]
    cdef int nQuadraturePoints_elementBoundary = grad_w.shape[1]
    cdef int nDOF_test_element = grad_w.shape[2]
    cdef int nDOF_trial_element = v.shape[2]
    cdef int nSpace = n.shape[2]
    cupdateGlobalJacobianFromExteriorElementBoundaryDiffusionAdjoint_dense(nExteriorElementBoundaries_global,
									   nQuadraturePoints_elementBoundary,
									   nDOF_test_element,
									   nDOF_trial_element,
									   nSpace,
									   offset_r,
									   stride_r,
									   offset_u,
									   stride_u,
									   nFreeVDOF_global,
									   <proteus_index_t*> exteriorElementBoundaries.data,
									   <proteus_index_t*> elementBoundaryElements.data,
									   <int*> elementBoundaryLocalElementBoundaries.data,
									   <int*> nFreeDOF_element_r.data,
									   <int*> freeLocal_r.data,
									   <proteus_index_t*> freeGlobal_r.data,
									   <int*> nFreeDOF_element_u.data,
									   <int*> freeLocal_u.data,
									   <proteus_index_t*> freeGlobal_u.data,
									   <int*> isDOFBoundary.data,
									   sigma,
									   <double*> v.data,
									   <double*> n.data,
									   <double*> a.data,
									   <double*> grad_w.data,
									   <double*> dS.data,
									   <double*> jac.data)
def updateInteriorElementBoundaryDiffusionAdjoint_sd(np.ndarray rowptr,
						     np.ndarray colind,
						     np.ndarray interiorElementBoundaries,
						     np.ndarray elementBoundaryElements,
						     np.ndarray elementBoundaryLocalElementBoundaries,
						     double sigma,
						     np.ndarray u,
						     np.ndarray n,
						     np.ndarray a,
						     np.ndarray grad_w,
						     np.ndarray dS,
						     np.ndarray residual):
    cdef int nInteriorElementBoundaries_global = interiorElementBoundaries.shape[0]
    cdef int nElementBoundaries_element = grad_w.shape[1]
    cdef int nQuadraturePoints_elementBoundary = grad_w.shape[2]
    cdef int nDOF_test_element = grad_w.shape[3]
    cdef int nSpace = grad_w.shape[4]
    cupdateInteriorElementBoundaryDiffusionAdjoint_sd(nInteriorElementBoundaries_global,
						      nElementBoundaries_element,
						      nQuadraturePoints_elementBoundary,
						      nDOF_test_element,
						      nSpace,
						      <proteus_csr_index_t*> rowptr.data,
						      <proteus_csr_index_t*> colind.data,
						      <proteus_index_t*> interiorElementBoundaries.data,
						      <proteus_index_t*> elementBoundaryElements.data,
						      <int*> elementBoundaryLocalElementBoundaries.data,
						      sigma,
						      <double*> u.data,
						      <double*> n.data,
						      <double*> a.data,
						      <double*> grad_w.data,
						      <double*> dS.data,
						      <double*> residual.data)
def updateExteriorElementBoundaryDiffusionAdjoint_sd(np.ndarray rowptr,
						     np.ndarray colind,
						     np.ndarray isDOFBoundary,
						     np.ndarray exteriorElementBoundaries,
						     np.ndarray elementBoundaryElements,
						     np.ndarray elementBoundaryLocalElementBoundaries,
						     double sigma,
						     np.ndarray u,
						     np.ndarray ub,
						     np.ndarray n,
						     np.ndarray a,
						     np.ndarray grad_w,
						     np.ndarray dS,
						     np.ndarray residual):
    cdef int nExteriorElementBoundaries_global = exteriorElementBoundaries.shape[0]
    cdef int nQuadraturePoints_elementBoundary = grad_w.shape[1]
    cdef int nDOF_test_element = grad_w.shape[2]
    cdef int nSpace = grad_w.shape[3]
    cupdateExteriorElementBoundaryDiffusionAdjoint_sd(nExteriorElementBoundaries_global,
                                                      nQuadraturePoints_elementBoundary,
                                                      nDOF_test_element,
                                                      nSpace,
                                                      <proteus_csr_index_t*> rowptr.data,
						      <proteus_csr_index_t*> colind.data,
						      <int*> isDOFBoundary.data,
						      <proteus_index_t*> exteriorElementBoundaries.data,
						      <proteus_index_t*> elementBoundaryElements.data,
						      <int*> elementBoundaryLocalElementBoundaries.data,
						      sigma,
						      <double*> u.data,
						      <double*> ub.data,
						      <double*> n.data,
						      <double*> a.data,
						      <double*> grad_w.data,
						      <double*> dS.data,
						      <double*> residual.data)
def updateGlobalJacobianFromInteriorElementBoundaryDiffusionAdjoint_dense_sd(np.ndarray rowptr,
									     np.ndarray colind,
									     proteus_index_t offset_r,
									     proteus_index_t stride_r,
									     proteus_index_t offset_u,
									     proteus_index_t stride_u,
									     proteus_index_t nFreeVDOF_global,
									     np.ndarray interiorElementBoundaries,
									     np.ndarray elementBoundaryElements,
									     np.ndarray elementBoundaryLocalElementBoundaries,
									     np.ndarray nFreeDOF_element_r,
									     np.ndarray freeLocal_r,
									     np.ndarray freeGlobal_r,
									     np.ndarray nFreeDOF_element_u,
									     np.ndarray freeLocal_u,
									     np.ndarray freeGlobal_u,
									     double sigma,
									     np.ndarray v,
									     np.ndarray n,
									     np.ndarray a,
									     np.ndarray grad_w,
									     np.ndarray dS,
									     np.ndarray jac):
    cdef int nInteriorElementBoundaries_global = interiorElementBoundaries.shape[0]
    cdef int nElementBoundaries_element = grad_w.shape[1]
    cdef int nQuadraturePoints_elementBoundary = grad_w.shape[2]
    cdef int nDOF_test_element = grad_w.shape[3]
    cdef int nDOF_trial_element = v.shape[3]
    cdef int nSpace = n.shape[3]
    cupdateGlobalJacobianFromInteriorElementBoundaryDiffusionAdjoint_dense_sd(nInteriorElementBoundaries_global,
									      nElementBoundaries_element,
									      nQuadraturePoints_elementBoundary,
									      nDOF_test_element,
									      nDOF_trial_element,
									      nSpace,
									      <proteus_csr_index_t*> rowptr.data,
									      <proteus_csr_index_t*> colind.data,
									      offset_r,
									      stride_r,
									      offset_u,
									      stride_u,
									      nFreeVDOF_global,
									      <proteus_index_t*> interiorElementBoundaries.data,
									      <proteus_index_t*> elementBoundaryElements.data,
									      <int*> elementBoundaryLocalElementBoundaries.data,
									      <int*> nFreeDOF_element_r.data,
									      <int*> freeLocal_r.data,
									      <proteus_index_t*> freeGlobal_r.data,
									      <int*> nFreeDOF_element_u.data,
									      <int*> freeLocal_u.data,
									      <proteus_index_t*> freeGlobal_u.data,
									      sigma,
									      <double*> v.data,
									      <double*> n.data,
									      <double*> a.data,
									      <double*> grad_w.data,
									      <double*> dS.data,
									      <double*> jac.data)
def updateGlobalJacobianFromExteriorElementBoundaryDiffusionAdjoint_dense_sd(np.ndarray rowptr,
									     np.ndarray colind,
									     proteus_index_t offset_r,
									     proteus_index_t stride_r,
									     proteus_index_t offset_u,
									     proteus_index_t stride_u,
									     proteus_index_t nFreeVDOF_global,
									     np.ndarray exteriorElementBoundaries,
									     np.ndarray elementBoundaryElements,
									     np.ndarray elementBoundaryLocalElementBoundaries,
									     np.ndarray nFreeDOF_element_r,
									     np.ndarray freeLocal_r,
									     np.ndarray freeGlobal_r,
									     np.ndarray nFreeDOF_element_u,
									     np.ndarray freeLocal_u,
									     np.ndarray freeGlobal_u,
									     np.ndarray isDOFBoundary,
									     double sigma,
									     np.ndarray v,
									     np.ndarray n,
									     np.ndarray a,
									     np.ndarray grad_w,
									     np.ndarray dS,
									     np.ndarray jac):
    cdef int nExteriorElementBoundaries_global = exteriorElementBoundaries.shape[0]
    cdef int nQuadraturePoints_elementBoundary = grad_w.shape[1]
    cdef int nDOF_test_element = grad_w.shape[2]
    cdef int nDOF_trial_element = v.shape[2]
    cdef int nSpace = n.shape[2]
    cupdateGlobalJacobianFromExteriorElementBoundaryDiffusionAdjoint_dense_sd(nExteriorElementBoundaries_global,
									      nQuadraturePoints_elementBoundary,
									      nDOF_test_element,
									      nDOF_trial_element,
									      nSpace,
									      <proteus_csr_index_t*> rowptr.data,
									      <proteus_csr_index_t*> colind.data,
									      offset_r,
									      stride_r,
									      offset_u,
									      stride_u,
									      nFreeVDOF_global,
									      <proteus_index_t*> exteriorElementBoundaries.data,
									      <proteus_index_t*> elementBoundaryElements.data,
									      <int*> elementBoundaryLocalElementBoundaries.data,
									      <int*> nFreeDOF_element_r.data,
									      <int*> freeLocal_r.data,
									      <proteus_index_t*> freeGlobal_r.data,
									      <int*> nFreeDOF_element_u.data,
									      <int*> freeLocal_u.data,
									      <proteus_index_t*> freeGlobal_u.data,
									      <int*> isDOFBoundary.data,
									      sigma,
									      <double*> v.data,
									      <double*> n.data,
									      <double*> a.data,
									      <double*> grad_w.data,
									      <double*> dS.data,
									      <double*> jac.data)
def updateGlobalJacobianFromInteriorElementBoundaryDiffusionAdjoint_CSR_sd(np.ndarray rowptr,
									   np.ndarray colind,
									   proteus_index_t offset_r,
									   proteus_index_t stride_r,
									   proteus_index_t offset_u,
									   proteus_index_t stride_u,
									   proteus_index_t nFreeVDOF_global,
									   np.ndarray interiorElementBoundaries,
									   np.ndarray elementBoundaryElements,
									   np.ndarray elementBoundaryLocalElementBoundaries,
									   np.ndarray nFreeDOF_element_r,
									   np.ndarray freeLocal_r,
									   np.ndarray freeGlobal_r,
									   np.ndarray nFreeDOF_element_u,
									   np.ndarray freeLocal_u,
									   np.ndarray freeGlobal_u,
									   np.ndarray csrRowIndeces_ru,
									   np.ndarray csrColumnOffsets_eb_ru,
									   double sigma,
									   np.ndarray v,
									   np.ndarray n,
									   np.ndarray a,
									   np.ndarray grad_w,
									   np.ndarray dS,
									   jac):
    cdef np.ndarray rowptr_dummy, colind_dummy, jac_array
    (rowptr_dummy,colind_dummy,jac_array) = jac.getCSRrepresentation()
    cdef int nInteriorElementBoundaries_global = interiorElementBoundaries.shape[0]
    cdef int nElementBoundaries_element = grad_w.shape[1]
    cdef int nQuadraturePoints_elementBoundary = grad_w.shape[2]
    cdef int nDOF_test_element = grad_w.shape[3]
    cdef int nDOF_trial_element = v.shape[3]
    cdef int nSpace = n.shape[3]
    cupdateGlobalJacobianFromInteriorElementBoundaryDiffusionAdjoint_CSR_sd(nInteriorElementBoundaries_global,
									    nElementBoundaries_element,
									    nQuadraturePoints_elementBoundary,
									    nDOF_test_element,
									    nDOF_trial_element,
									    nSpace,
									    <proteus_csr_index_t*> rowptr.data,
									    <proteus_csr_index_t*> colind.data,
									    offset_r,
									    stride_r,
									    offset_u,
									    stride_u,
									    nFreeVDOF_global,
									    <proteus_index_t*> interiorElementBoundaries.data,
									    <proteus_index_t*> elementBoundaryElements.data,
									    <int*> elementBoundaryLocalElementBoundaries.data,
									    <int*> nFreeDOF_element_r.data,
									    <int*> freeLocal_r.data,
									    <proteus_index_t*> freeGlobal_r.data,
									    <int*> nFreeDOF_element_u.data,
									    <int*> freeLocal_u.data,
									    <proteus_index_t*> freeGlobal_u.data,
									    <proteus_csr_index_t*> csrRowIndeces_ru.data,
									    <proteus_csr_index_t*> csrColumnOffsets_eb_ru.data,
									    sigma,
									    <double*> v.data,
									    <double*> n.data,
									    <double*> a.data,
									    <double*> grad_w.data,
									    <double*> dS.data,
									    <double*> jac_array.data)
def updateGlobalJacobianFromExteriorElementBoundaryDiffusionAdjoint_CSR_sd(np.ndarray rowptr,
									   np.ndarray colind,
									   proteus_index_t offset_r,
									   proteus_index_t stride_r,
									   proteus_index_t offset_u,
									   proteus_index_t stride_u,
									   proteus_index_t nFreeVDOF_global,
									   np.ndarray exteriorElementBoundaries,
									   np.ndarray elementBoundaryElements,
									   np.ndarray elementBoundaryLocalElementBoundaries,
									   np.ndarray nFreeDOF_element_r,
									   np.ndarray freeLocal_r,
									   np.ndarray freeGlobal_r,
									   np.ndarray nFreeDOF_element_u,
									   np.ndarray freeLocal_u,
									   np.ndarray freeGlobal_u,
									   np.ndarray csrRowIndeces_ru,
									   np.ndarray csrColumnOffsets_eb_ru,
									   np.ndarray isDOFBoundary,
									   double sigma,
									   np.ndarray v,
									   np.ndarray n,
									   np.ndarray a,
									   np.ndarray grad_w,
									   np.ndarray dS,
									   jac):
    cdef np.ndarray rowptr_dummy, colind_dummy, jac_array
    (rowptr_dummy,colind_dummy,jac_array) = jac.getCSRrepresentation()
    cdef int nExteriorElementBoundaries_global = exteriorElementBoundaries.shape[0]
    cdef int nQuadraturePoints_elementBoundary = grad_w.shape[1]
    cdef int nDOF_test_element = grad_w.shape[2]
    cdef int nDOF_trial_element = v.shape[2]
    cdef int nSpace = n.shape[2]
    cupdateGlobalJacobianFromExteriorElementBoundaryDiffusionAdjoint_CSR_sd(nExteriorElementBoundaries_global,
									    nQuadraturePoints_elementBoundary,
									    nDOF_test_element,
									    nDOF_trial_element,
									    nSpace,
									    <proteus_csr_index_t*> rowptr.data,
									    <proteus_csr_index_t*> colind.data,
									    offset_r,
									    stride_r,
									    offset_u,
									    stride_u,
									    nFreeVDOF_global,
									    <proteus_index_t*> exteriorElementBoundaries.data,
									    <proteus_index_t*> elementBoundaryElements.data,
									    <int*> elementBoundaryLocalElementBoundaries.data,
									    <int*> nFreeDOF_element_r.data,
									    <int*> freeLocal_r.data,
									    <proteus_index_t*> freeGlobal_r.data,
									    <int*> nFreeDOF_element_u.data,
									    <int*> freeLocal_u.data,
									    <proteus_index_t*> freeGlobal_u.data,
									    <proteus_csr_index_t*> csrRowIndeces_ru.data,
									    <proteus_csr_index_t*> csrColumnOffsets_eb_ru.data,
									    <int*> isDOFBoundary.data,
									    sigma,
									    <double*> v.data,
									    <double*> n.data,
									    <double*> a.data,
									    <double*> grad_w.data,
									    <double*> dS.data,
									    <double*> jac_array.data)
def update_f_movingDomain(np.ndarray xt,
			  np.ndarray m,
			  np.ndarray f):
    if m.ndim == 2:
        cupdate_f_movingDomain_q(f.shape[0],
                                 f.shape[1],
                                 f.shape[2],
			         <double*> xt.data,
			         <double*> m.data,
			         <double*> f.data)
    elif m.ndim == 3:
        cupdate_f_movingDomain_ebq(f.shape[0],
                                   f.shape[1],
                                   f.shape[2],
                                   f.shape[3],
			           <double*> xt.data,
			           <double*> m.data,
			           <double*> f.data)
def update_f_movingDomain_constantMass(np.ndarray xt,
				       np.ndarray f):
    if f.ndim == 3:
        cupdate_f_movingDomain_constantMass_q(f.shape[0],
                                              f.shape[1],
                                              f.shape[2],
					      <double*> xt.data,
					      <double*> f.data)
    elif f.ndim == 4:
        cupdate_f_movingDomain_constantMass_ebq(f.shape[0],
                                                f.shape[1],
                                                f.shape[2],
                                                f.shape[3],
					        <double*> xt.data,
					        <double*> f.data)
def updateStress_weak(np.ndarray sigma,
		      np.ndarray grad_w_dV,
		      np.ndarray weak_residual_x,
		      np.ndarray weak_residual_y,
		      np.ndarray weak_residual_z):
    cdef int nElements_global = grad_w_dV.shape[0]
    cdef int nQuadraturePoints_element = grad_w_dV.shape[1]
    cdef int nDOF_test_element = grad_w_dV.shape[2]
    cdef int nSpace = grad_w_dV.shape[3]
    cupdateStress_weak(nElements_global,
		       nQuadraturePoints_element,
		       nDOF_test_element,
		       nSpace,
		       <double*> sigma.data,
		       <double*> grad_w_dV.data,
		       <double*> weak_residual_x.data,
		       <double*> weak_residual_y.data,
		       <double*> weak_residual_z.data)
def updateStressJacobian_weak(np.ndarray dsigma_xx,
			      np.ndarray dsigma_xy,
			      np.ndarray dsigma_xz,
			      np.ndarray dsigma_yx,
			      np.ndarray dsigma_yy,
			      np.ndarray dsigma_yz,
			      np.ndarray dsigma_zx,
			      np.ndarray dsigma_zy,
			      np.ndarray dsigma_zz,
			      np.ndarray grad_v,
			      np.ndarray grad_w_dV,
			      np.ndarray jacobian_weak_residual_xx,
			      np.ndarray jacobian_weak_residual_xy,
			      np.ndarray jacobian_weak_residual_xz,
			      np.ndarray jacobian_weak_residual_yx,
			      np.ndarray jacobian_weak_residual_yy,
			      np.ndarray jacobian_weak_residual_yz,
			      np.ndarray jacobian_weak_residual_zx,
			      np.ndarray jacobian_weak_residual_zy,
			      np.ndarray jacobian_weak_residual_zz):
    cdef int nElements_global = grad_w_dV.shape[0]
    cdef int nQuadraturePoints_element = grad_w_dV.shape[1]
    cdef int nDOF_trial_element = grad_w_dV.shape[2]
    cdef int nDOF_test_element = grad_w_dV.shape[2]
    cdef int nSpace = grad_w_dV.shape[3]
    cupdateStressJacobian_weak(nElements_global,
			       nQuadraturePoints_element,
			       nDOF_trial_element,
			       nDOF_test_element,
			       nSpace,
			       <double*> dsigma_xx.data,
			       <double*> dsigma_xy.data,
			       <double*> dsigma_xz.data,
			       <double*> dsigma_yx.data,
			       <double*> dsigma_yy.data,
			       <double*> dsigma_yz.data,
			       <double*> dsigma_zx.data,
			       <double*> dsigma_zy.data,
			       <double*> dsigma_zz.data,
			       <double*> grad_v.data,
			       <double*> grad_w_dV.data,
			       <double*> jacobian_weak_residual_xx.data,
			       <double*> jacobian_weak_residual_xy.data,
			       <double*> jacobian_weak_residual_xz.data,
			       <double*> jacobian_weak_residual_yx.data,
			       <double*> jacobian_weak_residual_yy.data,
			       <double*> jacobian_weak_residual_yz.data,
			       <double*> jacobian_weak_residual_zx.data,
			       <double*> jacobian_weak_residual_zy.data,
			       <double*> jacobian_weak_residual_zz.data)
def projectFromNodalInterpolationConditions(int dim_dof,
					    np.ndarray l2g,
					    np.ndarray functional_map_element,
					    np.ndarray interpolationValues,
					    np.ndarray dofs):
    cdef int nElements_global = l2g.shape[0]
    cdef int nDOF_element = l2g.shape[1]
    cprojectFromNodalInterpolationConditions(nElements_global,
					     nDOF_element,
					     dim_dof,
					     <proteus_index_t*> l2g.data,
					     <int*> functional_map_element.data,
					     <double*> interpolationValues.data,
					     <double*> dofs.data)
def calculateVelocityQuadrature_MixedForm2_sd(np.ndarray rowptr,
                                            np.ndarray colind,
                                            np.ndarray qa,
                                            np.ndarray qw_dV,
                                            np.ndarray b,
                                            np.ndarray v,
                                            np.ndarray V,
                                            np.ndarray qv,
                                            np.ndarray qV):
    cccalculateVelocityQuadrature_MixedForm2_sd(v.shape[0],
                                            v.shape[1],
                                            v.shape[2],
                                            v.shape[3],
                                            b.shape[1],
                                            qv.shape[1],
                                            <proteus_csr_index_t*> rowptr.data,
                                            <proteus_csr_index_t*> colind.data,
                                            <double*> qa.data,
                                            <double*> qw_dV.data,
                                            <double*> b.data,
                                            <double*> v.data,
                                            <double*> V.data,
                                            <double*> qv.data,
                                            <double*> qV.data)

def updateDiffusion_MixedForm_weak_sd(np.ndarray rowptr,
                    np.ndarray colind,
                    np.ndarray a,
                    np.ndarray qV,
                    np.ndarray grad_w_dV,
                    np.ndarray velocity,
                    np.ndarray residual,
                    int rho_split):
    cupdateDiffusion_MixedForm_weak_sd(grad_w_dV.shape[0],
                    grad_w_dV.shape[1],
                    grad_w_dV.shape[2],
                    grad_w_dV.shape[3],
                    rho_split,
                    <proteus_csr_index_t*> rowptr.data,
                    <proteus_csr_index_t*> colind.data,
                    <double*> a.data,
                    <double*> qV.data,
                    <double*> grad_w_dV.data,
                    <double*> velocity.data,
                    <double*> residual.data)

def calculateVelocityQuadrature_MixedForm2_Jacobian_sd(np.ndarray rowptr,
                    np.ndarray colind,
                    np.ndarray qa,
                    np.ndarray qw_dV,
                    np.ndarray db,
                    np.ndarray db_eb,
                    np.ndarray v,
                    np.ndarray DV,
                    np.ndarray DV_eb,
                    np.ndarray qv,
                    np.ndarray qDV,
                    np.ndarray qDV_eb):
    ccalculateVelocityQuadrature_MixedForm2_Jacobian_sd(v.shape[0],
                    v.shape[1],
                    v.shape[2],
                    v.shape[3],
                    db.shape[1],
                    qv.shape[1],
                    <proteus_csr_index_t*> rowptr.data,
                    <proteus_csr_index_t*> colind.data,
                    <double*> qa.data,
                    <double*> qw_dV.data,
                    <double*> db.data,
                    <double*> db_eb.data,
                    <double*> v.data,
                    <double*> DV.data,
                    <double*> DV_eb.data,
                    <double*> qv.data,
                    <double*> qDV.data,
                    <double*> qDV_eb.data)
def updateDiffusionJacobian_MixedForm_weak_sd(np.ndarray rowptr,
                       np.ndarray colind,
                       np.ndarray a,
                       np.ndarray da,
                       np.ndarray qV,
                       np.ndarray qDV,
                       np.ndarray qDV_eb,
                       np.ndarray grad_w_dV,
                       np.ndarray v,
                       np.ndarray jacobian,
                       np.ndarray jacobian_eb):
    cupdateDiffusionJacobian_MixedForm_weak_sd(qDV_eb.shape[0],
                        qDV_eb.shape[1],
                        qDV_eb.shape[2],
                        qDV_eb.shape[3],
                        grad_w_dV.shape[2],
                        grad_w_dV.shape[3],
                        <proteus_csr_index_t*> rowptr.data,
                        <proteus_csr_index_t*> colind.data,
                        <double*>  a.data,
                        <double*> da.data,
                        <double*> qV.data,
                        <double*> qDV.data,
                        <double*> qDV_eb.data,
                        <double*> grad_w_dV.data,
                        <double*> v.data,
                        <double*> jacobian.data,
                        <double*> jacobian_eb.data)
