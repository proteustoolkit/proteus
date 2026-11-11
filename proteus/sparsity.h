#ifndef SPARSITYINFO_H
#define SPARSITYINFO_H
#include "index_type.h"
#include <map>
#include <set>
#include <cassert>
#include <algorithm>

namespace proteus
{
  class SparsityInfo
  {
  public:
    SparsityInfo();
    ~SparsityInfo();
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
                        proteus_csr_index_t* rowptr,
                        proteus_csr_index_t* csrRowIndeces,
                        proteus_csr_index_t* csrColumnOffsets,
                        proteus_csr_index_t* csrColumnOffsets_eNebN,
                        proteus_csr_index_t* csrColumnOffsets_eb,
                        proteus_csr_index_t* csrColumnOffsets_eb_eNebN);
    void getCSR();
    proteus_index_t nrows;
    proteus_csr_index_t nnz;
    proteus_csr_index_t *rowptr, *colind;
    double* nzval;
  private:
    std::map<proteus_index_t,std::set<proteus_index_t> > columnIndecesMap;
    std::map<proteus_index_t,std::map<proteus_index_t,proteus_csr_index_t> > columnOffsetsMap;
  };
}
#endif
