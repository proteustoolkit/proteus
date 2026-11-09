# A type of -*- python -*- file
# cython: language_level=3
cdef extern from "index_type.h":
    ctypedef int proteus_index_t

cdef extern from "sparsity.h" namespace "proteus":
    cdef cppclass SparsityInfo:
        proteus_index_t nrows
        proteus_index_t nnz
        proteus_index_t *rowptr
        proteus_index_t *colind
        double *nzval
        SparsityInfo()
        void findNonzeros(proteus_index_t nElements_global,
                          int nDOF_test_element,
                          int nDOF_trial_element,
                          int* nFreeDOF_test,
                          proteus_index_t* freeGlobal_test,
                          int* nFreeDOF_trial,
                          proteus_index_t* freeGlobal_trial,
                          proteus_index_t offset_test,
                          proteus_index_t stride_test,
                          proteus_index_t offset_trial,
                          proteus_index_t stride_trial,
                          int hasNumericalFlux,
                          int hasDiffusionInMixedForm,
                          int needNumericalFluxJacobian,
                          int nElementBoundaries_element,
                          proteus_index_t* elementNeighborsArray,
                          proteus_index_t nInteriorElementBoundaries_global,
                          proteus_index_t* interiorElementBoundariesArray,
                          proteus_index_t* elementBoundaryElementsArray,
                          int* elementBoundaryLocalElementBoundariesArray,
                          int hasFluxBoundaryConditions,
                          proteus_index_t nExteriorElementBoundaries_global,
                          proteus_index_t* exteriorElementBoundariesArray,
                          int hasOutflowBoundary,
                          int needOutflowJacobian);
        void getOffsets_CSR(proteus_index_t nElements_global,
                            int nDOF_test_element,
                            int nDOF_trial_element,
                            int* nFreeDOF_test,
                            proteus_index_t* freeGlobal_test,
                            int* nFreeDOF_trial,
                            proteus_index_t* freeGlobal_trial,
                            proteus_index_t offset_test,
                            proteus_index_t stride_test,
                            proteus_index_t offset_trial,
                            proteus_index_t stride_trial,
                            int hasNumericalFlux,
                            int hasDiffusionInMixedForm,
                            int needNumericalFluxJacobian,
                            int nElementBoundaries_element,
                            proteus_index_t* elementNeighborsArray,
                            proteus_index_t nInteriorElementBoundaries_global,
                            proteus_index_t* interiorElementBoundariesArray,
                            proteus_index_t* elementBoundaryElementsArray,
                            int* elementBoundaryLocalElementBoundariesArray,
                            int hasFluxBoundaryConditions,
                            proteus_index_t nExteriorElementBoundaries_global,
                            proteus_index_t* exteriorElementBoundariesArray,
                            int hasOutflowBoundary,
                            int needOutflowJacobian,
                            proteus_index_t* rowptr,
                            proteus_index_t* csrRowIndeces,
                            proteus_index_t* csrColumnOffsets,
                            proteus_index_t* csrColumnOffsets_eNebN,
                            proteus_index_t* csrColumnOffsets_eb,
                            proteus_index_t* csrColumnOffsets_eb_eNebN)
        void getCSR()
