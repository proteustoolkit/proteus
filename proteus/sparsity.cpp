#include "sparsity.h"

namespace proteus
{
  void SparsityInfo::findNonzeros(proteus_index_t nElements_global,
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
                                  int needOutflowJacobian)
  {      
    //elements
    for(int eN=0;eN<nElements_global;eN++)
      {
        for(int ii=0;ii<nFreeDOF_test[eN];ii++)
          {
            proteus_index_t I = offset_test + stride_test*freeGlobal_test[eN*nDOF_test_element+ii];
            for(int jj=0;jj<nFreeDOF_trial[eN];jj++)
              {
                proteus_index_t J = offset_trial + stride_trial*freeGlobal_trial[eN*nDOF_trial_element+jj];
                columnIndecesMap[I].insert(J);
              }
          }
        //get element neighbor DOF for mixed form diffusion
        if (hasNumericalFlux &&
            hasDiffusionInMixedForm &&
            needNumericalFluxJacobian)
          {
            for(proteus_index_t ebN=0;ebN<nElementBoundaries_element;ebN++)
              {
                proteus_index_t eN_ebN = elementNeighborsArray[eN*nElementBoundaries_element + ebN];
                if (eN_ebN >= 0)
                  {
                    for (int ii=0;ii<nFreeDOF_test[eN];ii++)
                      {
                        proteus_index_t I = offset_test + stride_test*freeGlobal_test[eN*nDOF_test_element+ii];
                        for (int jj=0;jj<nFreeDOF_trial[eN_ebN];jj++)
                          {
                            proteus_index_t J = offset_trial + stride_trial*freeGlobal_trial[eN_ebN*nDOF_trial_element+jj];
                            columnIndecesMap[I].insert(J);
                          }
                      }
                  }
              }
          }
      }
    //interior element boundaries
    if (hasNumericalFlux &&
        needNumericalFluxJacobian)
      {
        for(int ebNI=0;ebNI<nInteriorElementBoundaries_global;ebNI++)
          {
            proteus_index_t ebN = interiorElementBoundariesArray[ebNI],
              left_eN_global   = elementBoundaryElementsArray[ebN*2 + 0],
              right_eN_global  = elementBoundaryElementsArray[ebN*2 + 1];
            //left_ebN_element  = elementBoundaryLocalElementBoundariesArray[ebN*2 + 0],
            //right_ebN_element = elementBoundaryLocalElementBoundariesArray[ebN*2 + 1];
            for(int ii=0;ii<nFreeDOF_test[left_eN_global];ii++)
              {
                proteus_index_t left_I = offset_test+stride_test*freeGlobal_test[left_eN_global*nDOF_test_element + ii];
                for (int jj=0;jj<nFreeDOF_trial[left_eN_global];jj++)
                  {
                    proteus_index_t left_J = offset_trial+stride_trial*freeGlobal_trial[left_eN_global*nDOF_trial_element + jj];
                    columnIndecesMap[left_I].insert(left_J);
                  }
                for (int jj=0;jj<nFreeDOF_trial[right_eN_global];jj++)
                  {
                    proteus_index_t right_J = offset_trial+stride_trial*freeGlobal_trial[right_eN_global*nDOF_trial_element + jj];
                    columnIndecesMap[left_I].insert(right_J);
                  }
              }
            for(int ii=0;ii<nFreeDOF_test[right_eN_global];ii++)
              {
                proteus_index_t right_I = offset_test+stride_test*freeGlobal_test[right_eN_global*nDOF_test_element+ii];
                for(int jj=0;jj<nFreeDOF_trial[left_eN_global];jj++)
                  {
                    proteus_index_t left_J = offset_trial+stride_trial*freeGlobal_trial[left_eN_global*nDOF_trial_element+jj];
                    columnIndecesMap[right_I].insert(left_J);
                  }
                for(int jj=0;jj<nFreeDOF_trial[right_eN_global];jj++)
                  {
                    proteus_index_t right_J = offset_trial+stride_trial*freeGlobal_trial[right_eN_global*nDOF_trial_element+jj];
                    columnIndecesMap[right_I].insert(right_J);
                  }
              }
            if(hasDiffusionInMixedForm)
              {
                for(int ebN_eN=0;ebN_eN<nElementBoundaries_element;ebN_eN++)
                  {
                    proteus_index_t left_eN_ebN = elementNeighborsArray[left_eN_global*nElementBoundaries_element+ebN_eN],
                      right_eN_ebN = elementNeighborsArray[right_eN_global*nElementBoundaries_element+ebN_eN];
                    for(int ii=0;ii<nFreeDOF_test[left_eN_global];ii++)
                      {
                        proteus_index_t left_I = offset_test+stride_test*freeGlobal_test[left_eN_global*nDOF_test_element+ii];
                        if(left_eN_ebN >= 0)
                          {
                            for (int jj=0;jj<nFreeDOF_trial[left_eN_ebN];jj++)
                              {
                                proteus_index_t left_J = offset_trial+stride_trial*freeGlobal_trial[left_eN_ebN*nDOF_trial_element+jj];
                                columnIndecesMap[left_I].insert(left_J);
                              }
                          }
                        if(right_eN_ebN >= 0)
                          {
                            for(int jj=0;jj<nFreeDOF_trial[right_eN_ebN];jj++)
                              {
                                proteus_index_t right_J = offset_trial+stride_trial*freeGlobal_trial[right_eN_ebN*nDOF_trial_element+jj];
                                columnIndecesMap[left_I].insert(right_J);
                              }
                          }
                      }
                    for(int ii=0;ii<nFreeDOF_test[right_eN_global];ii++)
                      {
                        proteus_index_t right_I = offset_test+stride_test*freeGlobal_test[right_eN_global*nDOF_test_element+ii];
                        if(left_eN_ebN >= 0)
                          {
                            for(int jj=0;jj<nFreeDOF_trial[left_eN_ebN];jj++)
                              {
                                proteus_index_t left_J = offset_trial+stride_trial*freeGlobal_trial[left_eN_ebN*nDOF_trial_element+jj];
                                columnIndecesMap[right_I].insert(left_J);
                              }
                          }
                        if(right_eN_ebN >= 0)
                          {
                            for(int jj=0;jj<nFreeDOF_trial[right_eN_ebN];jj++)
                              {
                                proteus_index_t right_J = offset_trial+stride_trial*freeGlobal_trial[right_eN_ebN*nDOF_trial_element+jj];
                                columnIndecesMap[right_I].insert(right_J);
                              }
                          }
                      }
                  }
              }
          }
      }
    //exterior element boundaries
    if ((hasNumericalFlux &&
         needNumericalFluxJacobian) ||
        (hasOutflowBoundary &&
         needOutflowJacobian))
      {
        for(int ebNE=0;ebNE<nExteriorElementBoundaries_global;ebNE++)
          {
            int ebN = exteriorElementBoundariesArray[ebNE],
              eN_global   = elementBoundaryElementsArray[ebN*2+0];
            //ebN_element  = elementBoundaryLocalElementBoundariesArray[ebN*2+0];
            for (int ii=0;ii<nFreeDOF_test[eN_global];ii++)
              {
                proteus_index_t I = offset_test+stride_test*freeGlobal_test[eN_global*nDOF_test_element+ii];
                for (int jj=0;jj<nFreeDOF_trial[eN_global];jj++)
                  {
                    proteus_index_t J = offset_trial+stride_trial*freeGlobal_trial[eN_global*nDOF_trial_element+jj];
                    columnIndecesMap[I].insert(J);
                  }
              }
            if(hasNumericalFlux &&
               hasDiffusionInMixedForm)
              {
                for(int ebN_eN=0;ebN_eN < nElementBoundaries_element;ebN_eN++)
                  {
                    proteus_index_t eN_ebN = elementNeighborsArray[eN_global*nElementBoundaries_element+ebN_eN];
                    for (int ii=0;ii<nFreeDOF_test[eN_global];ii++)
                      {
                        proteus_index_t I = offset_test + stride_test*freeGlobal_test[eN_global*nDOF_test_element+ii];
                        if(eN_ebN >= 0)
                          {
                            for(int jj=0;jj<nFreeDOF_trial[eN_ebN];jj++)
                              {
                                proteus_index_t J = offset_trial+stride_trial*freeGlobal_trial[eN_ebN*nDOF_trial_element+jj];
                                columnIndecesMap[I].insert(J);
                              }
                          }
                      }
                  }
              }
          }
      }
    //debug
    //   for (std::map<proteus_index_t, std::set<proteus_index_t> >::iterator mit = columnIndecesMap.begin();mit != columnIndecesMap.end();mit++)
    //     {
    //       for(std::set<proteus_index_t>::iterator sit = mit->second.begin();sit!=mit->second.end();sit++)
    //         std::cout<<*sit<<'\t';
    //       std::cout<<std::endl;
    //     }
  }

  void SparsityInfo::getOffsets_CSR(proteus_index_t nElements_global,
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
  {
    //elements
    for(int eN=0;eN<nElements_global;eN++)
      {
        for(int ii=0;ii<nFreeDOF_test[eN];ii++)
          {
            proteus_index_t I = offset_test + stride_test*freeGlobal_test[eN*nDOF_test_element+ii];
            csrRowIndeces[eN*nDOF_test_element+ii] = rowptr[I];
            for(int jj=0;jj<nFreeDOF_trial[eN];jj++)
              {
                proteus_index_t J = offset_trial + stride_trial*freeGlobal_trial[eN*nDOF_trial_element+jj];
                csrColumnOffsets[eN*nDOF_test_element*nDOF_trial_element+
                                 ii*nDOF_trial_element+
                                 jj] = columnOffsetsMap[I][J];
              }
          }
        //get element neighbor DOF for mixed form diffusion
        if (hasNumericalFlux &&
            hasDiffusionInMixedForm &&
            needNumericalFluxJacobian)
          {
            for(proteus_index_t ebN=0;ebN<nElementBoundaries_element;ebN++)
              {
                proteus_index_t eN_ebN = elementNeighborsArray[eN*nElementBoundaries_element + ebN];
                if (eN_ebN >= 0)
                  {
                    for (int ii=0;ii<nFreeDOF_test[eN];ii++)
                      {
                        proteus_index_t I = offset_test + stride_test*freeGlobal_test[eN*nDOF_test_element+ii];
                        for (int jj=0;jj<nFreeDOF_trial[eN_ebN];jj++)
                          {
                            proteus_index_t J = offset_trial + stride_trial*freeGlobal_trial[eN_ebN*nDOF_trial_element+jj];
                            csrColumnOffsets_eNebN[eN*nElementBoundaries_element*nDOF_test_element*nDOF_trial_element+
                                                   ebN*nDOF_test_element*nDOF_trial_element+
                                                   ii*nDOF_test_element+
                                                   jj] = columnOffsetsMap[I][J];
                          }
                      }
                  }
              }
          }
      }
    //interior element boundaries
    if (hasNumericalFlux &&
        needNumericalFluxJacobian)
      {
        for(int ebNI=0;ebNI<nInteriorElementBoundaries_global;ebNI++)
          {
            int ebN = interiorElementBoundariesArray[ebNI],
              left_eN_global   = elementBoundaryElementsArray[ebN*2 + 0],
              right_eN_global  = elementBoundaryElementsArray[ebN*2 + 1];
            //left_ebN_element  = elementBoundaryLocalElementBoundariesArray[ebN*2 + 0],
            //right_ebN_element = elementBoundaryLocalElementBoundariesArray[ebN*2 + 1];
            for(int ii=0;ii<nFreeDOF_test[left_eN_global];ii++)
              {
                proteus_index_t left_I = offset_test+stride_test*freeGlobal_test[left_eN_global*nDOF_test_element + ii];
                for (int jj=0;jj<nFreeDOF_trial[left_eN_global];jj++)
                  {
                    proteus_index_t left_J = offset_trial+stride_trial*freeGlobal_trial[left_eN_global*nDOF_trial_element + jj];
                    csrColumnOffsets_eb[ebN*2*2*nDOF_test_element*nDOF_trial_element+
                                        0*2*nDOF_test_element*nDOF_trial_element+
                                        0*nDOF_test_element*nDOF_trial_element+
                                        ii*nDOF_trial_element+
                                        jj] = columnOffsetsMap[left_I][left_J];
                  }
                for (int jj=0;jj<nFreeDOF_trial[right_eN_global];jj++)
                  {
                    proteus_index_t right_J = offset_trial+stride_trial*freeGlobal_trial[right_eN_global*nDOF_trial_element + jj];
                    csrColumnOffsets_eb[ebN*2*2*nDOF_test_element*nDOF_trial_element+
                                        0*2*nDOF_test_element*nDOF_trial_element+
                                        1*nDOF_test_element*nDOF_trial_element+
                                        ii*nDOF_trial_element+
                                        jj] = columnOffsetsMap[left_I][right_J];
                  }
              }
            for(int ii=0;ii<nFreeDOF_test[right_eN_global];ii++)
              {
                proteus_index_t right_I = offset_test+stride_test*freeGlobal_test[right_eN_global*nDOF_test_element+ii];
                for(int jj=0;jj<nFreeDOF_trial[left_eN_global];jj++)
                  {
                    proteus_index_t left_J = offset_trial+stride_trial*freeGlobal_trial[left_eN_global*nDOF_trial_element+jj];
                    csrColumnOffsets_eb[ebN*2*2*nDOF_test_element*nDOF_trial_element+
                                        1*2*nDOF_test_element*nDOF_trial_element+
                                        0*nDOF_test_element*nDOF_trial_element+
                                        ii*nDOF_trial_element+
                                        jj] = columnOffsetsMap[right_I][left_J];
                  }
                for(int jj=0;jj<nFreeDOF_trial[right_eN_global];jj++)
                  {
                    proteus_index_t right_J = offset_trial+stride_trial*freeGlobal_trial[right_eN_global*nDOF_trial_element+jj];
                    csrColumnOffsets_eb[ebN*2*2*nDOF_test_element*nDOF_trial_element+
                                        1*2*nDOF_test_element*nDOF_trial_element+
                                        1*nDOF_test_element*nDOF_trial_element+
                                        ii*nDOF_trial_element+
                                        jj] = columnOffsetsMap[right_I][right_J];
                  }
              }
            if(hasDiffusionInMixedForm)
              {
                for(int ebN_eN=0;ebN_eN<nElementBoundaries_element;ebN_eN++)
                  {
                    int left_eN_ebN = elementNeighborsArray[left_eN_global*nElementBoundaries_element+ebN_eN],
                      right_eN_ebN = elementNeighborsArray[right_eN_global*nElementBoundaries_element+ebN_eN];
                    for(int ii=0;ii<nFreeDOF_test[left_eN_global];ii++)
                      {
                        proteus_index_t left_I = offset_test+stride_test*freeGlobal_test[left_eN_global*nDOF_test_element+ii];
                        if(left_eN_ebN >= 0)
                          {
                            for (int jj=0;jj<nFreeDOF_trial[left_eN_ebN];jj++)
                              {
                                proteus_index_t left_J = offset_trial+stride_trial*freeGlobal_trial[left_eN_ebN*nDOF_trial_element+jj];
                                csrColumnOffsets_eb_eNebN[ebN*2*2*nElementBoundaries_element*nDOF_test_element*nDOF_trial_element+
                                                          0*2*nElementBoundaries_element*nDOF_test_element*nDOF_trial_element+
                                                          0*nElementBoundaries_element*nDOF_test_element*nDOF_trial_element+
                                                          ebN_eN*nDOF_test_element*nDOF_trial_element+
                                                          ii*nDOF_trial_element+
                                                          jj] = columnOffsetsMap[left_I][left_J];
                              }
                          }
                        if(right_eN_ebN >= 0)
                          {
                            for(int jj=0;jj<nFreeDOF_trial[right_eN_ebN];jj++)
                              {
                                proteus_index_t right_J = offset_trial+stride_trial*freeGlobal_trial[right_eN_ebN*nDOF_trial_element+jj];
                                csrColumnOffsets_eb_eNebN[ebN*2*2*nElementBoundaries_element*nDOF_test_element*nDOF_trial_element+
                                                          0*2*nElementBoundaries_element*nDOF_test_element*nDOF_trial_element+
                                                          1*nElementBoundaries_element*nDOF_test_element*nDOF_trial_element+
                                                          ebN_eN*nDOF_test_element*nDOF_trial_element+
                                                          ii*nDOF_trial_element+
                                                          jj] = columnOffsetsMap[left_I][right_J];
                              }
                          }
                      }
                    for(int ii=0;ii<nFreeDOF_test[right_eN_global];ii++)
                      {
                        proteus_index_t right_I = offset_test+stride_test*freeGlobal_test[right_eN_global*nDOF_test_element+ii];
                        if(left_eN_ebN >= 0)
                          {
                            for(int jj=0;jj<nFreeDOF_trial[left_eN_ebN];jj++)
                              {
                                proteus_index_t left_J = offset_trial+stride_trial*freeGlobal_trial[left_eN_ebN*nDOF_trial_element+jj];
                                csrColumnOffsets_eb_eNebN[ebN*2*2*nElementBoundaries_element*nDOF_test_element*nDOF_trial_element+
                                                          1*2*nElementBoundaries_element*nDOF_test_element*nDOF_trial_element+
                                                          0*nElementBoundaries_element*nDOF_test_element*nDOF_trial_element+
                                                          ebN_eN*nDOF_test_element*nDOF_trial_element+
                                                          ii*nDOF_trial_element+
                                                          jj] = columnOffsetsMap[right_I][left_J];
                              }
                          }
                        if(right_eN_ebN >= 0)
                          {
                            for(int jj=0;jj<nFreeDOF_trial[right_eN_ebN];jj++)
                              {
                                proteus_index_t right_J = offset_trial+stride_trial*freeGlobal_trial[right_eN_ebN*nDOF_trial_element+jj];
                                csrColumnOffsets_eb_eNebN[ebN*2*2*nElementBoundaries_element*nDOF_test_element*nDOF_trial_element+
                                                          1*2*nElementBoundaries_element*nDOF_test_element*nDOF_trial_element+
                                                          1*nElementBoundaries_element*nDOF_test_element*nDOF_trial_element+
                                                          ebN_eN*nDOF_test_element*nDOF_trial_element+
                                                          ii*nDOF_trial_element+
                                                          jj] = columnOffsetsMap[right_I][right_J];
                              }
                          }
                      }
                  }
              }
          }
      }
    //exterior element boundaries
    if ((hasNumericalFlux &&
         needNumericalFluxJacobian) ||
        (hasOutflowBoundary &&
         needOutflowJacobian))
      {
        for(int ebNE=0;ebNE<nExteriorElementBoundaries_global;ebNE++)
          {
            int ebN = exteriorElementBoundariesArray[ebNE],
              eN_global   = elementBoundaryElementsArray[ebN*2+0];
            //ebN_element  = elementBoundaryLocalElementBoundariesArray[ebN*2+0];
            for (int ii=0;ii<nFreeDOF_test[eN_global];ii++)
              {
                proteus_index_t I = offset_test+stride_test*freeGlobal_test[eN_global*nDOF_test_element+ii];
                for (int jj=0;jj<nFreeDOF_trial[eN_global];jj++)
                  {
                    proteus_index_t J = offset_trial+stride_trial*freeGlobal_trial[eN_global*nDOF_trial_element+jj];
                    csrColumnOffsets_eb[ebN*2*2*nDOF_test_element*nDOF_trial_element+
                                        0*2*nDOF_test_element*nDOF_trial_element+
                                        0*nDOF_test_element*nDOF_trial_element+
                                        ii*nDOF_trial_element+
                                        jj] = columnOffsetsMap[I][J];
                  }
              }
            if(hasNumericalFlux &&
               hasDiffusionInMixedForm)
              {
                for(int ebN_eN=0;ebN_eN < nElementBoundaries_element;ebN_eN++)
                  {
                    proteus_index_t eN_ebN = elementNeighborsArray[eN_global*nElementBoundaries_element+ebN_eN];
                    for (int ii=0;ii<nFreeDOF_test[eN_global];ii++)
                      {
                        proteus_index_t I = offset_test + stride_test*freeGlobal_test[eN_global*nDOF_test_element+ii];
                        if(eN_ebN >= 0)
                          {
                            for(int jj=0;jj<nFreeDOF_trial[eN_ebN];jj++)
                              {
                                proteus_index_t J = offset_trial+stride_trial*freeGlobal_trial[eN_ebN*nDOF_trial_element+jj];
                                csrColumnOffsets_eb_eNebN[ebN*2*2*nElementBoundaries_element*nDOF_test_element*nDOF_trial_element+
                                                          0*2*nElementBoundaries_element*nDOF_test_element*nDOF_trial_element+
                                                          0*nElementBoundaries_element*nDOF_test_element*nDOF_trial_element+
                                                          ebN_eN*nDOF_test_element*nDOF_trial_element+
                                                          ii*nDOF_trial_element+
                                                          jj] = columnOffsetsMap[I][J];
                              }
                          }
                      }
                  }
              }
          }
      }
  }
  
  void SparsityInfo::getCSR()
  {
    //debug
    //   for (std::map<proteus_index_t, std::set<proteus_index_t> >::iterator mit = self->columnIndecesMap.begin();mit != self->columnIndecesMap.end();mit++)
    //     {
    //       for(std::set<proteus_index_t>::iterator sit = mit->second.begin();sit!=mit->second.end();sit++)
    //         std:://cout<<*sit<<'\t';
    //       std::cout<<std::endl;
    //     }
    nrows = int(columnIndecesMap.size())+1;
    rowptr = new proteus_index_t[nrows];
    rowptr[0] = 0;
    for(proteus_index_t I=1;I< columnIndecesMap.size()+1;I++)
      rowptr[I]=rowptr[I-1] + int(columnIndecesMap[I-1].size());
    nnz = rowptr[columnIndecesMap.size()];
    colind = new proteus_index_t[nnz];
    nzval = new double[nnz];
    proteus_index_t max_nonzeros=0;
    for(proteus_index_t I=0;I< columnIndecesMap.size();I++)
      {
        int offset=0;
        for(std::set<proteus_index_t>::iterator sit=columnIndecesMap[I].begin();sit != columnIndecesMap[I].end();sit++)
          {
            columnOffsetsMap[I][*sit] = offset;
            assert(rowptr[I]+offset < nnz);
            colind[rowptr[I]+offset]=*sit;
            offset++;
          }
        std::sort(&colind[rowptr[I]],&colind[rowptr[I+1]]);
        max_nonzeros = std::max(max_nonzeros,rowptr[I+1] - rowptr[I]);
      }
    //std::cout<<"Proteus: Maximum nonzeros in any row is "<<max_nonzeros<<std::endl;
  }
  
  SparsityInfo::SparsityInfo():
    rowptr(NULL),
    colind(NULL),
    nzval(NULL)
  {}
  
  SparsityInfo::~SparsityInfo()
  {
    if (rowptr != NULL)
      delete [] rowptr;
    if (colind != NULL)
      delete [] colind;
    if (nzval != NULL)
      delete [] nzval;
  }
}//proteus
