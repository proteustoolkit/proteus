#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <strings.h>
#include <assert.h>
#include "femIntegrals.h"
#include PROTEUS_LAPACK_H

/**
   \file femIntegrals.h
   \ingroup femIntegrals

   \brief A library of functions for computing the discrete finite element
   formulations.

@{
*/

void copyLeftElementBoundaryInfo(int nElementBoundaries_element,
                                 int nElementBoundaryQuadraturePoints_elementBoundary,
                                 int nSpace_global,
                                 proteus_index_t nExteriorElementBoundaries_global,
                                 proteus_index_t nInteriorElementBoundaries_global,
                                 proteus_index_t* elementBoundaryElementsArray,
                                 int* elementBoundaryLocalElementBoundariesArray,
                                 proteus_index_t* exteriorElementBoundariesArray,
                                 proteus_index_t* interiorElementBoundariesArray,
                                 double* x,
                                 double* n,
                                 double* xg,
                                 double* ng)
{
  proteus_index_t ebN, left_eN_global, right_eN_global, I;  int ebNE, ebNI, k, left_ebN_element, right_ebN_element;
  for (ebNE=0;ebNE<nExteriorElementBoundaries_global;ebNE++)
    {
      ebN = exteriorElementBoundariesArray[ebNE];
      left_eN_global   = elementBoundaryElementsArray[ebN*2+0];
      left_ebN_element  = elementBoundaryLocalElementBoundariesArray[ebN*2+0];
      for (k=0;k<nElementBoundaryQuadraturePoints_elementBoundary;k++)
        {
          for (I=0;I<3;I++)
            xg[ebN*nElementBoundaryQuadraturePoints_elementBoundary*3+
               k*3+
               I] =
              x[left_eN_global*nElementBoundaries_element*nElementBoundaryQuadraturePoints_elementBoundary*3+
                left_ebN_element*nElementBoundaryQuadraturePoints_elementBoundary*3+
                k*3+
                I];
          for (I=0;I<nSpace_global;I++)
            ng[ebN*nElementBoundaryQuadraturePoints_elementBoundary*nSpace_global+
               k*nSpace_global+
               I] =
              n[left_eN_global*nElementBoundaries_element*nElementBoundaryQuadraturePoints_elementBoundary*nSpace_global+
                left_ebN_element*nElementBoundaryQuadraturePoints_elementBoundary*nSpace_global+
                k*nSpace_global+
                I];
        }
    }
  for (ebNI=0;ebNI<nInteriorElementBoundaries_global;ebNI++)
    {
      ebN = interiorElementBoundariesArray[ebNI];
      left_eN_global   = elementBoundaryElementsArray[ebN*2+0];
      left_ebN_element  = elementBoundaryLocalElementBoundariesArray[ebN*2+0];
      right_eN_global  = elementBoundaryElementsArray[ebN*2+1];
      right_ebN_element = elementBoundaryLocalElementBoundariesArray[ebN*2+1];
      for (k=0;k<nElementBoundaryQuadraturePoints_elementBoundary;k++)
        {
          for (I=0;I<3;I++)
            {
              xg[ebN*nElementBoundaryQuadraturePoints_elementBoundary*3+
                 k*3+
                 I] =
                x[left_eN_global*nElementBoundaries_element*nElementBoundaryQuadraturePoints_elementBoundary*3+
                  left_ebN_element*nElementBoundaryQuadraturePoints_elementBoundary*3+
                  k*3+
                  I];
              x[right_eN_global*nElementBoundaries_element*nElementBoundaryQuadraturePoints_elementBoundary*3+
                right_ebN_element*nElementBoundaryQuadraturePoints_elementBoundary*3+
                k*3+
                I]=
                x[left_eN_global*nElementBoundaries_element*nElementBoundaryQuadraturePoints_elementBoundary*3+
                  left_ebN_element*nElementBoundaryQuadraturePoints_elementBoundary*3+
                  k*3+
                  I];
            }
          for (I=0;I<nSpace_global;I++)
            {
              ng[ebN*nElementBoundaryQuadraturePoints_elementBoundary*nSpace_global+
                 k*nSpace_global+
                 I] =
                n[left_eN_global*nElementBoundaries_element*nElementBoundaryQuadraturePoints_elementBoundary*nSpace_global+
                  left_ebN_element*nElementBoundaryQuadraturePoints_elementBoundary*nSpace_global+
                  k*nSpace_global+
                  I];
              n[right_eN_global*nElementBoundaries_element*nElementBoundaryQuadraturePoints_elementBoundary*nSpace_global+
                right_ebN_element*nElementBoundaryQuadraturePoints_elementBoundary*nSpace_global+
                k*nSpace_global+
                I]=
                -n[left_eN_global*nElementBoundaries_element*nElementBoundaryQuadraturePoints_elementBoundary*nSpace_global+
                   left_ebN_element*nElementBoundaryQuadraturePoints_elementBoundary*nSpace_global+
                   k*nSpace_global+
                   I];
            }
        }
    }
}

void copyGlobalElementBoundaryInfo(int nElementBoundaries_element,
                                 int nElementBoundaryQuadraturePoints_elementBoundary,
                                 int nSpace_global,
                                 proteus_index_t nExteriorElementBoundaries_global,
                                 proteus_index_t nInteriorElementBoundaries_global,
                                 proteus_index_t* elementBoundaryElementsArray,
                                 int* elementBoundaryLocalElementBoundariesArray,
                                 proteus_index_t* exteriorElementBoundariesArray,
                                 proteus_index_t* interiorElementBoundariesArray,
                                 double* x,
                                 double* n,
                                 double* ebqe_x,
                                 double* ebqe_n,
                                 double* xg,
                                 double* ng)
{
  proteus_index_t ebN, left_eN_global, right_eN_global, I, J;  int ebNE, ebNI, k, left_ebN_element, right_ebN_element;
  double dot,sign;
  for (ebNE=0;ebNE<nExteriorElementBoundaries_global;ebNE++)
    {
      ebN = exteriorElementBoundariesArray[ebNE];
      left_eN_global   = elementBoundaryElementsArray[ebN*2+0];
      left_ebN_element  = elementBoundaryLocalElementBoundariesArray[ebN*2+0];
      for (k=0;k<nElementBoundaryQuadraturePoints_elementBoundary;k++)
        {
          dot=0.0;
          for (J=0;J<nSpace_global;J++)
            dot+= ebqe_n[ebNE*nElementBoundaryQuadraturePoints_elementBoundary*nSpace_global+
                         k*nSpace_global+
                         J]*
              ng[ebN*nElementBoundaryQuadraturePoints_elementBoundary*nSpace_global+
                 k*nSpace_global+
                 J];
          if(dot < 0.0)
            sign=-1.0;
          else
            sign=1.0;

        for (I=0;I<3;I++)
          {
            x[left_eN_global*nElementBoundaries_element*nElementBoundaryQuadraturePoints_elementBoundary*3+
              left_ebN_element*nElementBoundaryQuadraturePoints_elementBoundary*3+
              k*3+
              I] = xg[ebN*nElementBoundaryQuadraturePoints_elementBoundary*3+
                      k*3+
                      I];
            ebqe_x[ebNE*nElementBoundaryQuadraturePoints_elementBoundary*3+
                      k*3+
                      I] = xg[ebN*nElementBoundaryQuadraturePoints_elementBoundary*3+
                      k*3+
                      I];
          }
        for (I=0;I<nSpace_global;I++)
          {
            n[left_eN_global*nElementBoundaries_element*nElementBoundaryQuadraturePoints_elementBoundary*nSpace_global+
              left_ebN_element*nElementBoundaryQuadraturePoints_elementBoundary*nSpace_global+
              k*nSpace_global+
              I] =
              ng[ebN*nElementBoundaryQuadraturePoints_elementBoundary*nSpace_global+
                 k*nSpace_global+
                 I];
            ebqe_n[ebNE*nElementBoundaryQuadraturePoints_elementBoundary*nSpace_global+
                   k*nSpace_global+
                   I]=
              sign*ng[ebN*nElementBoundaryQuadraturePoints_elementBoundary*nSpace_global+
                      k*nSpace_global+
                      I];
          }
        }
    }
  for (ebNI=0;ebNI<nInteriorElementBoundaries_global;ebNI++)
    {
      ebN = interiorElementBoundariesArray[ebNI];
      left_eN_global   = elementBoundaryElementsArray[ebN*2+0];
      left_ebN_element  = elementBoundaryLocalElementBoundariesArray[ebN*2+0];
      right_eN_global  = elementBoundaryElementsArray[ebN*2+1];
      right_ebN_element = elementBoundaryLocalElementBoundariesArray[ebN*2+1];
      for (k=0;k<nElementBoundaryQuadraturePoints_elementBoundary;k++)
        {
          for (I=0;I<3;I++)
            {
              x[left_eN_global*nElementBoundaries_element*nElementBoundaryQuadraturePoints_elementBoundary*3+
                left_ebN_element*nElementBoundaryQuadraturePoints_elementBoundary*3+
                k*3+
                I] = xg[ebN*nElementBoundaryQuadraturePoints_elementBoundary*3+
                        k*3+
                        I];
              x[right_eN_global*nElementBoundaries_element*nElementBoundaryQuadraturePoints_elementBoundary*3+
                right_ebN_element*nElementBoundaryQuadraturePoints_elementBoundary*3+
                k*3+
                I] = xg[ebN*nElementBoundaryQuadraturePoints_elementBoundary*3+
                        k*3+
                        I];
            }
          for (I=0;I<nSpace_global;I++)
            {
              n[left_eN_global*nElementBoundaries_element*nElementBoundaryQuadraturePoints_elementBoundary*nSpace_global+
                left_ebN_element*nElementBoundaryQuadraturePoints_elementBoundary*nSpace_global+
                k*nSpace_global+
                I] = ng[ebN*nElementBoundaryQuadraturePoints_elementBoundary*nSpace_global+
                        k*nSpace_global+
                        I];
              n[right_eN_global*nElementBoundaries_element*nElementBoundaryQuadraturePoints_elementBoundary*nSpace_global+
                right_ebN_element*nElementBoundaryQuadraturePoints_elementBoundary*nSpace_global+
                k*nSpace_global+
                I] = -ng[ebN*nElementBoundaryQuadraturePoints_elementBoundary*nSpace_global+
                         k*nSpace_global+
                         I];
            }
        }
    }
}

void copyLeftElementBoundaryInfo_movingDomain(int nElementBoundaries_element,
                                              int nElementBoundaryQuadraturePoints_elementBoundary,
                                              proteus_index_t nExteriorElementBoundaries_global,
                                              proteus_index_t nInteriorElementBoundaries_global,
                                              proteus_index_t* elementBoundaryElementsArray,
                                              int* elementBoundaryLocalElementBoundariesArray,
                                              proteus_index_t* exteriorElementBoundariesArray,
                                              proteus_index_t* interiorElementBoundariesArray,
                                              double* xt)
{
  proteus_index_t ebN, left_eN_global, right_eN_global, I;  int ebNI, k, left_ebN_element, right_ebN_element;
  for (ebNI=0;ebNI<nInteriorElementBoundaries_global;ebNI++)
    {
      ebN = interiorElementBoundariesArray[ebNI];
      left_eN_global   = elementBoundaryElementsArray[ebN*2+0];
      left_ebN_element  = elementBoundaryLocalElementBoundariesArray[ebN*2+0];
      right_eN_global  = elementBoundaryElementsArray[ebN*2+1];
      right_ebN_element = elementBoundaryLocalElementBoundariesArray[ebN*2+1];
      for (k=0;k<nElementBoundaryQuadraturePoints_elementBoundary;k++)
        for (I=0;I<3;I++)
          {
            xt[right_eN_global*nElementBoundaries_element*nElementBoundaryQuadraturePoints_elementBoundary*3+
              right_ebN_element*nElementBoundaryQuadraturePoints_elementBoundary*3+
              k*3+
              I]=
              xt[left_eN_global*nElementBoundaries_element*nElementBoundaryQuadraturePoints_elementBoundary*3+
                left_ebN_element*nElementBoundaryQuadraturePoints_elementBoundary*3+
                k*3+
                I];
          }
    }
}

void parametricFiniteElementSpace_getValues(proteus_index_t nElements_global,
                                            int nQuadraturePoints_element,
                                            int nDOF_element,
                                            double* psi,
                                            double* vArray)
{
  proteus_index_t eN;  int k, j;
  for(eN=0;eN<nElements_global;eN++)
    for(k=0;k<nQuadraturePoints_element;k++)
      for(j=0;j<nDOF_element;j++)
        vArray[eN*nQuadraturePoints_element*nDOF_element+
               k*nDOF_element+
               j]
          =
          psi[k*nDOF_element+
              j];
}

void parametricFiniteElementSpace_getValuesTrace(proteus_index_t nElements_global,
                                                 int nElementBoundaries_element,
                                                 int nElementBoundaryQuadraturePoints_elementBoundary,
                                                 int nDOF_element,
                                                 double* psi,
                                                 int* permutations,
                                                 double* vArray)
{
  proteus_index_t eN, ebN;  int k, j;
  for(eN=0;eN<nElements_global;eN++)
    for (ebN=0;ebN<nElementBoundaries_element;ebN++)
      for(k=0;k<nElementBoundaryQuadraturePoints_elementBoundary;k++)
        for(j=0;j<nDOF_element;j++)
          vArray[eN*nElementBoundaries_element*nElementBoundaryQuadraturePoints_elementBoundary*nDOF_element+
                 ebN*nElementBoundaryQuadraturePoints_elementBoundary*nDOF_element+
                 k*nDOF_element+
                 j]
            =
            psi[permutations[eN*nElementBoundaries_element*nElementBoundaryQuadraturePoints_elementBoundary+
                             ebN*nElementBoundaryQuadraturePoints_elementBoundary+
                             k]*nDOF_element+
                j];
}

void parametricFiniteElementSpace_getValuesGlobalExteriorTrace(int nElementBoundaries_element,
                                                               int nElementBoundaryQuadraturePoints_elementBoundary,
                                                               int nDOF_element,
                                                               proteus_index_t nExteriorElementBoundaries_global,
                                                               const proteus_index_t* exteriorElementBoundariesArray,
                                                               const proteus_index_t* elementBoundaryElementsArray,
                                                               const int* elementBoundaryLocalElementBoundariesArray,
                                                               double* psi,
                                                               double* vArray)
{
  proteus_index_t eN, ebN;  int ebNE, ebN_local, k, j;
  for (ebNE = 0; ebNE < nExteriorElementBoundaries_global; ebNE++)
    {
      ebN = exteriorElementBoundariesArray[ebNE];
      eN  = elementBoundaryElementsArray[ebN*2+0];
      ebN_local = elementBoundaryLocalElementBoundariesArray[ebN*2+0];
      for(k=0;k<nElementBoundaryQuadraturePoints_elementBoundary;k++)
        for(j=0;j<nDOF_element;j++)
          vArray[ebNE*nElementBoundaryQuadraturePoints_elementBoundary*nDOF_element+
                 k*nDOF_element+
                 j]
            =
            psi[ebN_local*nElementBoundaryQuadraturePoints_elementBoundary*nDOF_element +
                k*nDOF_element +
                j];
    }
}

void parametricFiniteElementSpace_getGradientValues(proteus_index_t nElements_global,
                                                    int nQuadraturePoints_element,
                                                    int nDOF_element,
                                                    int nSpace_global,
                                                    double* grad_psi,
                                                    double* inverseJacobianArray,
                                                    double* grad_vArray)
{
  proteus_index_t eN, I, J;  int k, j, nSpace_global2 = nSpace_global*nSpace_global;
  for(eN=0;eN<nElements_global;eN++)
    for(k=0;k<nQuadraturePoints_element;k++)
      for(j=0;j<nDOF_element;j++)
        for (I=0;I<nSpace_global;I++)
          for (J=0;J<nSpace_global;J++)
            grad_vArray[eN*nQuadraturePoints_element*nDOF_element*nSpace_global+
                        k*nDOF_element*nSpace_global+
                        j*nSpace_global+
                        I]
              +=
              grad_psi[k*nDOF_element*nSpace_global+
                       j*nSpace_global+
                       J]
              *
              inverseJacobianArray[eN*nQuadraturePoints_element*nSpace_global2+
                                   k*nSpace_global2+
                                   J*nSpace_global+
                                   I];
}

void parametricFiniteElementSpace_getHessianValues(proteus_index_t nElements_global,
                                                   int nQuadraturePoints_element,
                                                   int nDOF_element,
                                                   int nSpace_global,
                                                   double* Hessian_psi,
                                                   double* inverseJacobianArray,
                                                   double* Hessian_vArray)
{
  proteus_index_t eN, I, J;  int k, j, K, L, nSpace_global2 = nSpace_global*nSpace_global;
  for(eN=0;eN<nElements_global;eN++)
    for(k=0;k<nQuadraturePoints_element;k++)
      for(j=0;j<nDOF_element;j++)
        for (I=0;I<nSpace_global;I++)
          for (J=0;J<nSpace_global;J++)
            for (K=0;K<nSpace_global;K++)
              for (L=0;L<nSpace_global;L++)
                Hessian_vArray[eN*nQuadraturePoints_element*nDOF_element*nSpace_global2+
                               k*nDOF_element*nSpace_global2+
                               j*nSpace_global2+
                               I*nSpace_global+
                               J]
                  +=
                  Hessian_psi[k*nDOF_element*nSpace_global2+
                              j*nSpace_global2+
                              K*nSpace_global+
                              L]
                  *
                  inverseJacobianArray[eN*nQuadraturePoints_element*nSpace_global2+
                                       k*nSpace_global2+
                                       L*nSpace_global+
                                       J]
                  *
                  inverseJacobianArray[eN*nQuadraturePoints_element*nSpace_global2+
                                       k*nSpace_global2+
                                       K*nSpace_global+
                                       I];
}

void parametricFiniteElementSpace_getGradientValuesTrace(proteus_index_t nElements_global,
                                                         int nElementBoundaries_element,
                                                         int nElementBoundaryQuadraturePoints_elementBoundary,
                                                         int nDOF_element,
                                                         int nSpace_global,
                                                         double* grad_psi,
                                                         int* permutations,
                                                         double* inverseJacobianArray,
                                                         double* grad_vArray)
{
  proteus_index_t eN, ebN, I, J;  int k, j, nSpace_global2 = nSpace_global*nSpace_global;
  for(eN=0;eN<nElements_global;eN++)
    for (ebN=0;ebN<nElementBoundaries_element;ebN++)
      for(k=0;k<nElementBoundaryQuadraturePoints_elementBoundary;k++)
        for(j=0;j<nDOF_element;j++)
          for (I=0;I<nSpace_global;I++)
            for (J=0;J<nSpace_global;J++)
              grad_vArray[eN*nElementBoundaries_element*nElementBoundaryQuadraturePoints_elementBoundary*nDOF_element*nSpace_global+
                          ebN*nElementBoundaryQuadraturePoints_elementBoundary*nDOF_element*nSpace_global+
                          k*nDOF_element*nSpace_global+
                          j*nSpace_global+
                          I]
                +=
                grad_psi[permutations[eN*nElementBoundaries_element*nElementBoundaryQuadraturePoints_elementBoundary+
                                      ebN*nElementBoundaryQuadraturePoints_elementBoundary+
                                      k]*nDOF_element*nSpace_global+
                         j*nSpace_global+
                         J]
                *
                inverseJacobianArray[eN*nElementBoundaries_element*nElementBoundaryQuadraturePoints_elementBoundary*nSpace_global2+
                                     ebN*nElementBoundaryQuadraturePoints_elementBoundary*nSpace_global2+
                                     k*nSpace_global2+
                                     J*nSpace_global+
                                     I];
}

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
                                                                       double* grad_vArray)
{
  proteus_index_t ebN, eN, I, J;  int ebNE, ebN_local, k, j, nSpace_global2 = nSpace_global*nSpace_global;
  for (ebNE = 0; ebNE < nExteriorElementBoundaries_global; ebNE++)
    {
      ebN = exteriorElementBoundariesArray[ebNE];
      eN  = elementBoundaryElementsArray[ebN*2+0];
      ebN_local = elementBoundaryLocalElementBoundariesArray[ebN*2+0];
      for(k=0;k<nElementBoundaryQuadraturePoints_elementBoundary;k++)
        for(j=0;j<nDOF_element;j++)
          for (I=0;I<nSpace_global;I++)
            for (J=0;J<nSpace_global;J++)
              grad_vArray[ebNE*nElementBoundaryQuadraturePoints_elementBoundary*nDOF_element*nSpace_global+
                          k*nDOF_element*nSpace_global+
                          j*nSpace_global+
                          I]
                +=
                grad_psi[ebN_local*nElementBoundaryQuadraturePoints_elementBoundary*nDOF_element*nSpace_global+
                         k*nDOF_element*nSpace_global+
                         j*nSpace_global+
                         J]
                *
                inverseJacobianArray[ebNE*nElementBoundaryQuadraturePoints_elementBoundary*nSpace_global2+
                                     k*nSpace_global2+
                                     J*nSpace_global+
                                     I];
    }
}

void parametricMaps_getPermutations(proteus_index_t nElements_global,
                                    int nElementBoundaries_element,
                                    int nElementBoundaryQuadraturePoints_elementBoundary,
                                    int nSpace_global,
                                    double* xiArray,
                                    int* permutations)
{
  const int kTot=(nElementBoundaries_element*nElementBoundaryQuadraturePoints_elementBoundary);
  proteus_index_t eN, ebN, I;  int k, k0;
  register double errorNorm,errorNormMin;
  /* permutations are relative  to the ordering on the  first element so the first entries used are the identity*/
  for (k0=0;k0<kTot;k0++)
    permutations[k0]=k0;
  /* now  loop over  remaining elements and find  the permutation used to get to reference points to match*/
  for (eN=1;eN<nElements_global;eN++)
    {
      for (ebN=0;ebN<nElementBoundaries_element;ebN++)
        for (k=0;k<nElementBoundaryQuadraturePoints_elementBoundary;k++)
          {
            errorNormMin=1.0;
            for (k0=0;k0<kTot;k0++)
              {
                errorNorm=0.0;
                for (I=0;I<nSpace_global;I++)
                  {
                    errorNorm += fabs(xiArray[k0*3+I]
                                      -
                                      xiArray[eN*nElementBoundaries_element*nElementBoundaryQuadraturePoints_elementBoundary*3+
                                              ebN*nElementBoundaryQuadraturePoints_elementBoundary*3+
                                              k*3+
                                              I]);
                  }
                if (errorNorm < errorNormMin)
                  {
                    permutations[eN*nElementBoundaries_element*nElementBoundaryQuadraturePoints_elementBoundary+
                                 ebN*nElementBoundaryQuadraturePoints_elementBoundary+
                                 k] = k0;
                    errorNormMin = errorNorm;
                  }
              }
          }
    }
}
void parametricMaps_getPermutationsGlobalExterior(int nElementBoundaryQuadraturePoints_elementBoundary,
                                                  int nSpace_global,
                                                  proteus_index_t nExteriorElementBoundaries_global,
                                                  const proteus_index_t * exteriorElementBoundariesArray,
                                                  const proteus_index_t * elementBoundaryElementsArray,
                                                  const int * elementBoundaryLocalElementBoundariesArray,
                                                  double* xiArray,
                                                  int* permutations)
{
  const int kTot=nElementBoundaryQuadraturePoints_elementBoundary;
  proteus_index_t eN, ebN, I;  int ebNE, ebN_local, k, k0;
  register double errorNorm,errorNormMin;
  /* permutations are relative  to the ordering on the  first elementBoundary so the first entries used are the identity*/
  for (k0=0;k0<kTot;k0++)
    permutations[k0]=k0;
  /* loop over exterior elementBoundaries and setup permutation used to get to reference points to match*/
  for (ebNE = 1; ebNE < nExteriorElementBoundaries_global; ebNE++)
    {
      ebN = exteriorElementBoundariesArray[ebNE];
      eN  = elementBoundaryElementsArray[ebN*2+0];
      ebN_local = elementBoundaryLocalElementBoundariesArray[ebN*2+0];
      for (k=0;k<nElementBoundaryQuadraturePoints_elementBoundary;k++)
        {
          errorNormMin=1.0;
          for (k0=0;k0<kTot;k0++)
            {
              errorNorm=0.0;
              for (I=0;I<nSpace_global;I++)
                {
                  errorNorm += fabs(xiArray[k0*3+I]
                                    -
                                    xiArray[ebNE*nElementBoundaryQuadraturePoints_elementBoundary*3+
                                            k*3+
                                            I]);
                  }
                if (errorNorm < errorNormMin)
                  {
                    permutations[ebNE*nElementBoundaryQuadraturePoints_elementBoundary+
                                 k] = k0;
                    errorNormMin = errorNorm;
                  }
            }
        }
    }
}

void getPermutationsGlobal(proteus_index_t nElementBoundaries_global,
                           int nElementBoundaryQuadraturePoints_elementBoundary,
                           double* xArray,
                           double* xArrayNew,
                           int* permutations)
{
  proteus_index_t ebN, I;  int k, k0;
  register double errorNorm,errorNormMin;
  /* now  loop over  remaining elements and find  the permutation used to get to reference points to match*/
  for (ebN=0;ebN<nElementBoundaries_global;ebN++)
    {
      for (k=0;k<nElementBoundaryQuadraturePoints_elementBoundary;k++)
        {
          errorNormMin=1.0;
          for (k0=0;k0<nElementBoundaryQuadraturePoints_elementBoundary;k0++)
            {
              errorNorm=0.0;
              for (I=0;I<3;I++)
                {
                  errorNorm += fabs(xArrayNew[ebN*nElementBoundaryQuadraturePoints_elementBoundary*3+
                                              k0*3+
                                              I]
                                    -
                                    xArray[ebN*nElementBoundaryQuadraturePoints_elementBoundary*3+
                                           k*3+
                                           I]);
                }
              if (errorNorm < errorNormMin)
                {
                  permutations[ebN*nElementBoundaryQuadraturePoints_elementBoundary+
                               k] = k0;
                  errorNormMin = errorNorm;
                }
            }
        }
    }
}

void parametricMaps_getValues(proteus_index_t nElements_global,
                              int nQuadraturePoints_element,
                              int nDOF_element,
                              int nSpace_global,
                              double* psi,
                              proteus_index_t* l2g,
                              double* nodeArray,
                              double* xArray)
{
  memset(xArray,0,sizeof(double)*nElements_global*nQuadraturePoints_element*3);
  proteus_index_t eN, j_global, I;  int k, j;
  for(eN=0;eN<nElements_global;eN++)
    for(k=0;k<nQuadraturePoints_element;k++)
      for(j=0;j<nDOF_element;j++)
        {
          j_global = l2g[eN*nDOF_element+
                         j];
          for(I=0;I<3;I++)
            xArray[eN*nQuadraturePoints_element*3+
                   k*3+
                   I]
              +=
              psi[k*nDOF_element+
                  j]
              *
              nodeArray[j_global*3+
                        I];
        }
}

void parametricMaps_getValuesTrace(proteus_index_t nElements_global,
                                   int nElementBoundaries_element,
                                   int nQuadraturePoints_element,
                                   int nDOF_element,
                                   int nSpace_global,
                                   double* psi,
                                   proteus_index_t* l2g,
                                   double* nodeArray,
                                   double* xArray)
{
  memset(xArray,0,sizeof(double)*nElements_global*nElementBoundaries_element*nQuadraturePoints_element*3);
  proteus_index_t eN, ebN, j_global, I;  int k, j;
  for(eN=0;eN<nElements_global;eN++)
    for(ebN=0;ebN<nElementBoundaries_element;ebN++)
      for(k=0;k<nQuadraturePoints_element;k++)
        for(j=0;j<nDOF_element;j++)
          {
            j_global = l2g[eN*nDOF_element+
                           j];
            for(I=0;I<3;I++)
              xArray[eN*nElementBoundaries_element*nQuadraturePoints_element*3+
                     ebN*nQuadraturePoints_element*3+
                     k*3+
                     I]
                +=
                psi[ebN*nQuadraturePoints_element*nDOF_element+
                    k*nDOF_element+
                    j]
                *
                nodeArray[j_global*3+
                          I];
          }
}

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
                                                 double* xArray)
{
  memset(xArray,0,sizeof(double)*nExteriorElementBoundaries_global*nQuadraturePoints_elementBoundary*3);
  proteus_index_t eN, ebN, j_global, I;  int ebNE, ebN_local, k, j;
  for (ebNE = 0; ebNE < nExteriorElementBoundaries_global; ebNE++)
    {
      ebN = exteriorElementBoundariesArray[ebNE];
      eN  = elementBoundaryElementsArray[ebN*2 + 0];
      ebN_local = elementBoundaryLocalElementBoundariesArray[ebN*2 + 0];
      for(k=0;k<nQuadraturePoints_elementBoundary;k++)
        for(j=0;j<nDOF_element;j++)
          {
            j_global = l2g[eN*nDOF_element+
                           j];
            for(I=0;I<3;I++)
              xArray[ebNE*nQuadraturePoints_elementBoundary*3+
                     k*3+
                     I]
                +=
                psi[ebN_local*nQuadraturePoints_elementBoundary*nDOF_element+
                    k*nDOF_element+
                    j]
                *
                nodeArray[j_global*3+
                          I];
          }
    }
}

void parametricMaps_getInverseValues(proteus_index_t nElements_global,
                                     int nQuadraturePoints_element,
                                     int nDOF_element,
                                     int nSpace_global,
                                     double* inverseJacobian,
                                     proteus_index_t* l2g,
                                     double* nodeArray,
                                     double* xArray,
                                     double* xiArray)
{
  memset(xiArray,0,sizeof(double)*nElements_global*nQuadraturePoints_element*3);
  proteus_index_t eN, node0, I, J;  int k, nSpace_global2=nSpace_global*nSpace_global;
  double dx[nSpace_global];
  for(eN=0;eN<nElements_global;eN++)
    for(k=0;k<nQuadraturePoints_element;k++)
      {
        node0 = l2g[eN*nDOF_element];
        for(I=0;I<nSpace_global;I++)
          dx[I] =
            xArray[eN*nQuadraturePoints_element*3+
                   k*3+
                   I]
            -
            nodeArray[node0*3+I];
        for(I=0;I<nSpace_global;I++)
          for(J=0;J<nSpace_global;J++)
            xiArray[eN*nQuadraturePoints_element*3+
                    k*3+
                    I]
              +=
              inverseJacobian[eN*nQuadraturePoints_element*nSpace_global2+
                              k*nSpace_global2+
                              I*nSpace_global+
                              J]
              *
              dx[J];
      }
}

void parametricMaps_getInverseValuesTrace(proteus_index_t nElements_global,
                                          int nElementBoundaries_element,
                                          int nElementBoundaryQuadraturePoints_elementBoundary,
                                          int nDOF_element,
                                          int nSpace_global,
                                          double* inverseJacobian,
                                          proteus_index_t* l2g,
                                          double* nodeArray,
                                          double* xArray,
                                          double* xiArray)
{
  memset(xiArray,0,sizeof(double)*nElements_global*nElementBoundaries_element*nElementBoundaryQuadraturePoints_elementBoundary*3);
  proteus_index_t eN, ebN, node0, I, J;  int k, nSpace_global2=nSpace_global*nSpace_global;
  double dx[nSpace_global];
  for(eN=0;eN<nElements_global;eN++)
    for (ebN=0;ebN<nElementBoundaries_element;ebN++)
      for(k=0;k<nElementBoundaryQuadraturePoints_elementBoundary;k++)
        {
          node0 = l2g[eN*nDOF_element];
          for(I=0;I<nSpace_global;I++)
            dx[I] =
              xArray[eN*nElementBoundaries_element*nElementBoundaryQuadraturePoints_elementBoundary*3+
                     ebN*nElementBoundaryQuadraturePoints_elementBoundary*3+
                     k*3+
                     I]
              -
              nodeArray[node0*3+I];
          for(I=0;I<nSpace_global;I++)
            for(J=0;J<nSpace_global;J++)
              xiArray[eN*nElementBoundaries_element*nElementBoundaryQuadraturePoints_elementBoundary*3+
                      ebN*nElementBoundaryQuadraturePoints_elementBoundary*3+
                      k*3+
                      I]
                +=
                inverseJacobian[eN*nElementBoundaries_element*nElementBoundaryQuadraturePoints_elementBoundary*nSpace_global2+
                                ebN*nElementBoundaryQuadraturePoints_elementBoundary*nSpace_global2+
                                k*nSpace_global2+
                                I*nSpace_global+
                                J]
                *
                dx[J];
        }
}

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
                                                        double* xiArray)
{
  memset(xiArray,0,sizeof(double)*nExteriorElementBoundaries_global*nElementBoundaryQuadraturePoints_elementBoundary*3);
  proteus_index_t eN, ebN, node0, I, J;  int ebNE, ebN_local, k, nSpace_global2=nSpace_global*nSpace_global;
  double dx[nSpace_global];
  for (ebNE = 0; ebNE < nExteriorElementBoundaries_global; ebNE++)
    {
      ebN = exteriorElementBoundariesArray[ebNE];
      eN  = elementBoundaryElementsArray[ebN*2+0];
      ebN_local = elementBoundaryLocalElementBoundariesArray[ebN*2+0];

      for(k=0;k<nElementBoundaryQuadraturePoints_elementBoundary;k++)
        {
          node0 = l2g[eN*nDOF_element];
          for(I=0;I<nSpace_global;I++)
            dx[I] =
              xArray[ebNE*nElementBoundaryQuadraturePoints_elementBoundary*3+
                     k*3+
                     I]
              -
              nodeArray[node0*3+I];
          for(I=0;I<nSpace_global;I++)
            for(J=0;J<nSpace_global;J++)
              xiArray[ebNE*nElementBoundaryQuadraturePoints_elementBoundary*3+
                      k*3+
                      I]
                +=
                inverseJacobian[ebNE*nElementBoundaryQuadraturePoints_elementBoundary*nSpace_global2+
                                k*nSpace_global2+
                                I*nSpace_global+
                                J]
                *
                dx[J];
        }
    }
}

void parametricMaps_getJacobianValues3D(proteus_index_t nElements_global,
                                        int nQuadraturePoints_element,
                                        int nDOF_element,
                                        double* grad_psi,
                                        proteus_index_t* l2g,
                                        double* nodeArray,
                                        double* jacobianArray,
                                        double* jacobianDeterminantArray,
                                        double* jacobianInverseArray)
{
  proteus_index_t eN, j_global;  int k, j;
  const int X=0,Y=1,Z=2,
    XX=0,XY=1,XZ=2,
    YX=3,YY=4,YZ=5,
    ZX=6,ZY=7,ZZ=8;
  double *jac=NULL,*jacDet=NULL,*jacInv=NULL,*grad=NULL,*node=NULL;
  register double oneOverJacDet=0.0;
  for(eN=0;eN<nElements_global;eN++)
    for(k=0;k<nQuadraturePoints_element;k++)
      {
        jac = jacobianArray +
          eN*nQuadraturePoints_element*9+
          k*9;
        jacDet = jacobianDeterminantArray +
          eN*nQuadraturePoints_element+
          k;
        jacInv = jacobianInverseArray +
          eN*nQuadraturePoints_element*9+
          k*9;
        for(j=0;j<nDOF_element;j++)
          {
            j_global = l2g[eN*nDOF_element+
                           j];
            grad = grad_psi + k*nDOF_element*3 + j*3;
            node = nodeArray + j_global*3;
            jac[XX] += node[X]*grad[X];
            jac[XY] += node[X]*grad[Y];
            jac[XZ] += node[X]*grad[Z];
            jac[YX] += node[Y]*grad[X];
            jac[YY] += node[Y]*grad[Y];
            jac[YZ] += node[Y]*grad[Z];
            jac[ZX] += node[Z]*grad[X];
            jac[ZY] += node[Z]*grad[Y];
            jac[ZZ] += node[Z]*grad[Z];
          }
        *jacDet
          =
          jac[XX]*(jac[YY]*jac[ZZ] - jac[YZ]*jac[ZY]) -
          jac[XY]*(jac[YX]*jac[ZZ] - jac[YZ]*jac[ZX]) +
          jac[XZ]*(jac[YX]*jac[ZY] - jac[YY]*jac[ZX]);
        oneOverJacDet = 1.0/(*jacDet);
        jacInv[XX] = oneOverJacDet*(jac[YY]*jac[ZZ] - jac[YZ]*jac[ZY]);
        jacInv[YX] = oneOverJacDet*(jac[YZ]*jac[ZX] - jac[YX]*jac[ZZ]);
        jacInv[ZX] = oneOverJacDet*(jac[YX]*jac[ZY] - jac[YY]*jac[ZX]);
        jacInv[XY] = oneOverJacDet*(jac[ZY]*jac[XZ] - jac[ZZ]*jac[XY]);
        jacInv[YY] = oneOverJacDet*(jac[ZZ]*jac[XX] - jac[ZX]*jac[XZ]);
        jacInv[ZY] = oneOverJacDet*(jac[ZX]*jac[XY] - jac[ZY]*jac[XX]);
        jacInv[XZ] = oneOverJacDet*(jac[XY]*jac[YZ] - jac[XZ]*jac[YY]);
        jacInv[YZ] = oneOverJacDet*(jac[XZ]*jac[YX] - jac[XX]*jac[YZ]);
        jacInv[ZZ] = oneOverJacDet*(jac[XX]*jac[YY] - jac[XY]*jac[YX]);
      }
}

void parametricMaps_getJacobianValues2D(proteus_index_t nElements_global,
                                        int nQuadraturePoints_element,
                                        int nDOF_element,
                                        double* grad_psi,
                                        proteus_index_t* l2g,
                                        double* nodeArray,
                                        double* jacobianArray,
                                        double* jacobianDeterminantArray,
                                        double* jacobianInverseArray)
{
  proteus_index_t eN, j_global;  int k, j;
  const int X=0,Y=1,
    XX=0,XY=1,
    YX=2,YY=3;
  double *jac=NULL,*jacDet=NULL,*jacInv=NULL,*grad=NULL,*node=NULL;
  register double oneOverJacDet=0.0;
  for(eN=0;eN<nElements_global;eN++)
    for(k=0;k<nQuadraturePoints_element;k++)
      {
        jac = jacobianArray +
          eN*nQuadraturePoints_element*4+
          k*4;
        jacDet = jacobianDeterminantArray +
          eN*nQuadraturePoints_element+
          k;
        jacInv = jacobianInverseArray +
          eN*nQuadraturePoints_element*4+
          k*4;
        for(j=0;j<nDOF_element;j++)
          {
            j_global = l2g[eN*nDOF_element+
                           j];
            grad = grad_psi + k*nDOF_element*2 + j*2;
            /* nodes are always 3D  */
            node = nodeArray + j_global*3;
            jac[XX] += node[X]*grad[X];
            jac[XY] += node[X]*grad[Y];
            jac[YX] += node[Y]*grad[X];
            jac[YY] += node[Y]*grad[Y];
          }
        *jacDet
          =
          jac[XX]*jac[YY]- jac[XY]*jac[YX];
        oneOverJacDet = 1.0/(*jacDet);
        jacInv[XX] = oneOverJacDet*jac[YY];
        jacInv[YX] = -oneOverJacDet*jac[YX];
        jacInv[XY] = -oneOverJacDet*jac[XY];
        jacInv[YY] = oneOverJacDet*jac[XX];
      }
}

void parametricMaps_getJacobianValues1D(proteus_index_t nElements_global,
                                        int nQuadraturePoints_element,
                                        int nDOF_element,
                                        double* grad_psi,
                                        proteus_index_t* l2g,
                                        double* nodeArray,
                                        double* jacobianArray,
                                        double* jacobianDeterminantArray,
                                        double* jacobianInverseArray)
{
  proteus_index_t eN, j_global;  int k, j;
  const int X=0,
    XX=0;
  double *jac=NULL,*jacDet=NULL,*jacInv=NULL,*grad=NULL,*node=NULL;
  register double oneOverJacDet=0.0;
  for(eN=0;eN<nElements_global;eN++)
    for(k=0;k<nQuadraturePoints_element;k++)
      {
        jac = jacobianArray +
          eN*nQuadraturePoints_element+
          k;
        jacDet = jacobianDeterminantArray +
          eN*nQuadraturePoints_element+
          k;
        jacInv = jacobianInverseArray +
          eN*nQuadraturePoints_element+
          k;
        for(j=0;j<nDOF_element;j++)
          {
            j_global = l2g[eN*nDOF_element+
                           j];
            grad = grad_psi + k*nDOF_element + j;
            /* nodes are always 3D  */
            node = nodeArray + j_global*3;
            jac[XX] += node[X]*grad[X];
          }
        *jacDet
          =
          jac[XX];
        oneOverJacDet = 1.0/(*jacDet);
        jacInv[XX] = oneOverJacDet;
      }
}

void parametricMaps_getJacobianValuesTrace3D(proteus_index_t nElements_global,
                                             int nElementBoundaries_element,
                                             int nQuadraturePoints_element,
                                             int nDOF_element,
                                             double* grad_psi,
                                             double* boundaryNormals,
                                             double* boundaryJacobians,
                                             proteus_index_t* l2g,
                                             double* nodeArray,
                                             double* jacobianInverseArray,
                                             double* metricTensorArray,
                                             double* metricTensorDeterminantSqrtArray,
                                             double* unitNormalArray)
{
  proteus_index_t eN, ebN, j_global;  int k, j;
  const int
    X=0,Y=1,Z=2,
    XX=0,XY=1,XZ=2,
    YX=3,YY=4,YZ=5,
    ZX=6,ZY=7,ZZ=8,
    XHX=0,XHY=1,
    YHX=2,YHY=3,
    ZHX=4,ZHY=5,
    HXHX=0,HXHY=1,
    HYHX=2,HYHY=3;
  double *jacInv=NULL,*mt=NULL,*mtDetSqrt=NULL,*n=NULL,*grad=NULL,*node=NULL,*bn,*bj;
  register double oneOverJacDet=0.0,oneOverNbn=0.0;
  double emj[9],ebmj[6];
  for(eN=0;eN<nElements_global;eN++)
    for(ebN=0;ebN<nElementBoundaries_element;ebN++)
      for(k=0;k<nQuadraturePoints_element;k++)
        {
          emj[XX] = 0.0;
          emj[XY] = 0.0;
          emj[XZ] = 0.0;
          emj[YX] = 0.0;
          emj[YY] = 0.0;
          emj[YZ] = 0.0;
          emj[ZX] = 0.0;
          emj[ZY] = 0.0;
          emj[ZZ] = 0.0;
          ebmj[XHX] = 0.0;
          ebmj[XHY] = 0.0;
          ebmj[YHX] = 0.0;
          ebmj[YHY] = 0.0;
          ebmj[ZHX] = 0.0;
          ebmj[ZHY] = 0.0;
          jacInv = jacobianInverseArray +
            eN*nElementBoundaries_element*nQuadraturePoints_element*9+
            ebN*nQuadraturePoints_element*9+
            k*9;
          mt = metricTensorArray +
            eN*nElementBoundaries_element*nQuadraturePoints_element*4+
            ebN*nQuadraturePoints_element*4+
            k*4;
          mtDetSqrt = metricTensorDeterminantSqrtArray +
            eN*nElementBoundaries_element*nQuadraturePoints_element+
            ebN*nQuadraturePoints_element+
            k;
          n = unitNormalArray+
            eN*nElementBoundaries_element*nQuadraturePoints_element*3+
            ebN*nQuadraturePoints_element*3+
            k*3;
          bn = boundaryNormals + ebN*3;
          bj = boundaryJacobians + ebN*6;
          for(j=0;j<nDOF_element;j++)
            {
              j_global = l2g[eN*nDOF_element+
                             j];
              grad = grad_psi + ebN*nQuadraturePoints_element*nDOF_element*3+k*nDOF_element*3 + j*3;
              /* nodes are always 3D  */
              node = nodeArray + j_global*3;
              emj[XX] += node[X]*grad[X];
              emj[XY] += node[X]*grad[Y];
              emj[XZ] += node[X]*grad[Z];
              emj[YX] += node[Y]*grad[X];
              emj[YY] += node[Y]*grad[Y];
              emj[YZ] += node[Y]*grad[Z];
              emj[ZX] += node[Z]*grad[X];
              emj[ZY] += node[Z]*grad[Y];
              emj[ZZ] += node[Z]*grad[Z];
             }
          oneOverJacDet = 1.0/(emj[XX]*(emj[YY]*emj[ZZ] - emj[YZ]*emj[ZY]) -
                               emj[XY]*(emj[YX]*emj[ZZ] - emj[YZ]*emj[ZX]) +
                               emj[XZ]*(emj[YX]*emj[ZY] - emj[YY]*emj[ZX]));
          jacInv[XX] = oneOverJacDet*(emj[YY]*emj[ZZ] - emj[YZ]*emj[ZY]);
          jacInv[YX] = oneOverJacDet*(emj[YZ]*emj[ZX] - emj[YX]*emj[ZZ]);
          jacInv[ZX] = oneOverJacDet*(emj[YX]*emj[ZY] - emj[YY]*emj[ZX]);
          jacInv[XY] = oneOverJacDet*(emj[ZY]*emj[XZ] - emj[ZZ]*emj[XY]);
          jacInv[YY] = oneOverJacDet*(emj[ZZ]*emj[XX] - emj[ZX]*emj[XZ]);
          jacInv[ZY] = oneOverJacDet*(emj[ZX]*emj[XY] - emj[ZY]*emj[XX]);
          jacInv[XZ] = oneOverJacDet*(emj[XY]*emj[YZ] - emj[XZ]*emj[YY]);
          jacInv[YZ] = oneOverJacDet*(emj[XZ]*emj[YX] - emj[XX]*emj[YZ]);
          jacInv[ZZ] = oneOverJacDet*(emj[XX]*emj[YY] - emj[XY]*emj[YX]);

          ebmj[XHX] = emj[XX]*bj[XHX]+emj[XY]*bj[YHX]+emj[XZ]*bj[ZHX];
          ebmj[XHY] = emj[XX]*bj[XHY]+emj[XY]*bj[YHY]+emj[XZ]*bj[ZHY];
          ebmj[YHX] = emj[YX]*bj[XHX]+emj[YY]*bj[YHX]+emj[YZ]*bj[ZHX];
          ebmj[YHY] = emj[YX]*bj[XHY]+emj[YY]*bj[YHY]+emj[YZ]*bj[ZHY];
          ebmj[ZHX] = emj[ZX]*bj[XHX]+emj[ZY]*bj[YHX]+emj[ZZ]*bj[ZHX];
          ebmj[ZHY] = emj[ZX]*bj[XHY]+emj[ZY]*bj[YHY]+emj[ZZ]*bj[ZHY];

          mt[HXHX] = ebmj[XHX]*ebmj[XHX]+ebmj[YHX]*ebmj[YHX]+ebmj[ZHX]*ebmj[ZHX];
          mt[HXHY] = ebmj[XHX]*ebmj[XHY]+ebmj[YHX]*ebmj[YHY]+ebmj[ZHX]*ebmj[ZHY];
          mt[HYHX] = ebmj[XHY]*ebmj[XHX]+ebmj[YHY]*ebmj[YHX]+ebmj[ZHY]*ebmj[ZHX];
          mt[HYHY] = ebmj[XHY]*ebmj[XHY]+ebmj[YHY]*ebmj[YHY]+ebmj[ZHY]*ebmj[ZHY];

          *mtDetSqrt=sqrt(mt[HXHX]*mt[HYHY]- mt[HXHY]*mt[HYHX]);


          n[X] = (jacInv[XX]*bn[X]+jacInv[YX]*bn[Y]+jacInv[ZX]*bn[Z]);
          n[Y] = (jacInv[XY]*bn[X]+jacInv[YY]*bn[Y]+jacInv[ZY]*bn[Z]);
          n[Z] = (jacInv[XZ]*bn[X]+jacInv[YZ]*bn[Y]+jacInv[ZZ]*bn[Z]);

          oneOverNbn = 1.0/sqrt(n[X]*n[X]+n[Y]*n[Y]+n[Z]*n[Z]);

          n[X] *= oneOverNbn;
          n[Y] *= oneOverNbn;
          n[Z] *= oneOverNbn;

        }
}

void parametricMaps_getJacobianValuesTrace2D(proteus_index_t nElements_global,
                                             int nElementBoundaries_element,
                                             int nQuadraturePoints_element,
                                             int nDOF_element,
                                             double* grad_psi,
                                             double* boundaryNormals,
                                             double* boundaryJacobians,
                                             proteus_index_t* l2g,
                                             double* nodeArray,
                                             double* jacobianInverseArray,
                                             double* metricTensorArray,
                                             double* metricTensorDeterminantSqrtArray,
                                             double* unitNormalArray)
{
  proteus_index_t eN, ebN, j_global;  int k, j;
  const int X=0,Y=1,
    XX=0,XY=1,
    YX=2,YY=3;
  double *jacInv=NULL,*mt,*mtDetSqrt,*n,*grad=NULL,*node=NULL,*bn,*bj;
  register double oneOverJacDet=0.0,oneOverNbn=0.0;
  double emj[4],ebmj[2];
  for(eN=0;eN<nElements_global;eN++)
    for(ebN=0;ebN<nElementBoundaries_element;ebN++)
      for(k=0;k<nQuadraturePoints_element;k++)
        {
          emj[XX] = 0.0;
          emj[XY] = 0.0;
          emj[YX] = 0.0;
          emj[YY] = 0.0;
          ebmj[XX] = 0.0;
          ebmj[XY] = 0.0;
          jacInv = jacobianInverseArray +
            eN*nElementBoundaries_element*nQuadraturePoints_element*4+
            ebN*nQuadraturePoints_element*4+
            k*4;
          mt = metricTensorArray +
            eN*nElementBoundaries_element*nQuadraturePoints_element+
            ebN*nQuadraturePoints_element+
            k;
          mtDetSqrt = metricTensorDeterminantSqrtArray +
            eN*nElementBoundaries_element*nQuadraturePoints_element+
            ebN*nQuadraturePoints_element+
            k;
          n = unitNormalArray+
            eN*nElementBoundaries_element*nQuadraturePoints_element*2+
            ebN*nQuadraturePoints_element*2+
            k*2;
          bn = boundaryNormals + ebN*2;
          bj = boundaryJacobians + ebN*2;
          for(j=0;j<nDOF_element;j++)
            {
              j_global = l2g[eN*nDOF_element+
                             j];
              grad = grad_psi + ebN*nQuadraturePoints_element*nDOF_element*2+k*nDOF_element*2 + j*2;
              /* nodes are always 3D  */
              node = nodeArray + j_global*3;
              emj[XX] += node[X]*grad[X];
              emj[XY] += node[X]*grad[Y];
              emj[YX] += node[Y]*grad[X];
              emj[YY] += node[Y]*grad[Y];
            }
          oneOverJacDet = 1.0/(emj[XX]*emj[YY]- emj[XY]*emj[YX]);
          jacInv[XX] = oneOverJacDet*emj[YY];
          jacInv[YX] = -oneOverJacDet*emj[YX];
          jacInv[XY] = -oneOverJacDet*emj[XY];
          jacInv[YY] = oneOverJacDet*emj[XX];
          ebmj[X] = emj[XX]*bj[XX]+emj[XY]*bj[Y];
          ebmj[Y] = emj[YX]*bj[X]+emj[YY]*bj[Y];
          mt[X] = ebmj[X]*ebmj[X]+ebmj[Y]*ebmj[Y];
          *mtDetSqrt=sqrt(mt[XX]);
          n[X] = jacInv[XX]*bn[X]+jacInv[YX]*bn[Y];
          n[Y] = jacInv[XY]*bn[X]+jacInv[YY]*bn[Y];
          oneOverNbn= 1.0/sqrt(n[X]*n[X]+n[Y]*n[Y]);
          n[X] *= oneOverNbn;
          n[Y] *= oneOverNbn;
        }
}

void parametricMaps_getJacobianValuesTrace2D_movingDomain(proteus_index_t nElements_global,
                                                          int nElementBoundaries_element,
                                                          int nQuadraturePoints_element,
                                                          int nDOF_element,
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
{
  proteus_index_t eN, ebN, j_global;  int k, j;
  const int X=0,Y=1,
    XX=0,XY=1,
    YX=2,YY=3;
  double *jacInv=NULL,*mt,*mtDetSqrt,*n,*grad=NULL,*node=NULL,*bn,*bj,*xt;
  register double oneOverJacDet=0.0,oneOverNbn=0.0,xt_dot_xt,xt_dot_n,xt_dot_bj;
  double emj[4],ebmj[2];
  for(eN=0;eN<nElements_global;eN++)
    for(ebN=0;ebN<nElementBoundaries_element;ebN++)
      for(k=0;k<nQuadraturePoints_element;k++)
        {
          emj[XX] = 0.0;
          emj[XY] = 0.0;
          emj[YX] = 0.0;
          emj[YY] = 0.0;
          ebmj[XX] = 0.0;
          ebmj[XY] = 0.0;
          jacInv = jacobianInverseArray +
            eN*nElementBoundaries_element*nQuadraturePoints_element*4+
            ebN*nQuadraturePoints_element*4+
            k*4;
          mt = metricTensorArray +
            eN*nElementBoundaries_element*nQuadraturePoints_element+
            ebN*nQuadraturePoints_element+
            k;
          mtDetSqrt = metricTensorDeterminantSqrtArray +
            eN*nElementBoundaries_element*nQuadraturePoints_element+
            ebN*nQuadraturePoints_element+
            k;
          n = unitNormalArray+
            eN*nElementBoundaries_element*nQuadraturePoints_element*2+
            ebN*nQuadraturePoints_element*2+
            k*2;
          xt = xtArray + eN*nElementBoundaries_element*nQuadraturePoints_element*3+
            ebN*nQuadraturePoints_element*3+
            k*3;
          bn = boundaryNormals + ebN*2;
          bj = boundaryJacobians + ebN*2;
          for(j=0;j<nDOF_element;j++)
            {
              j_global = l2g[eN*nDOF_element+
                             j];
              grad = grad_psi + ebN*nQuadraturePoints_element*nDOF_element*2+k*nDOF_element*2 + j*2;
              /* nodes are always 3D  */
              node = nodeArray + j_global*3;
              emj[XX] += node[X]*grad[X];
              emj[XY] += node[X]*grad[Y];
              emj[YX] += node[Y]*grad[X];
              emj[YY] += node[Y]*grad[Y];
            }
          oneOverJacDet = 1.0/(emj[XX]*emj[YY]- emj[XY]*emj[YX]);
          jacInv[XX] = oneOverJacDet*emj[YY];
          jacInv[YX] = -oneOverJacDet*emj[YX];
          jacInv[XY] = -oneOverJacDet*emj[XY];
          jacInv[YY] = oneOverJacDet*emj[XX];
          ebmj[X] = emj[XX]*bj[X]+emj[XY]*bj[Y];
          ebmj[Y] = emj[YX]*bj[X]+emj[YY]*bj[Y];
          mt[X] =  ebmj[X]*ebmj[X]+ebmj[Y]*ebmj[Y];
          n[X] = jacInv[XX]*bn[X]+jacInv[YX]*bn[Y];
          n[Y] = jacInv[XY]*bn[X]+jacInv[YY]*bn[Y];
          oneOverNbn= 1.0/sqrt(n[X]*n[X]+n[Y]*n[Y]);
          n[X] *= oneOverNbn;
          n[Y] *= oneOverNbn;
          xt_dot_xt = xt[X]*xt[X]+xt[Y]*xt[Y];
          xt_dot_bj = xt[X]*ebmj[X]+xt[Y]*ebmj[Y];
          xt_dot_n  = xt[X]*n[X]+xt[Y]*n[Y];
          //printf("%12.5e %12.5e %12.5e\n",xt_dot_xt,xt_dot_bj,xt_dot_n);
          //printf("%12.5e \n",((1.0+xt_dot_xt)*mt[X] - xt_dot_bj*xt_dot_bj)/(1.0+xt_dot_n*xt_dot_n));
          *mtDetSqrt=sqrt(((1.0+xt_dot_xt)*mt[X] - xt_dot_bj*xt_dot_bj)/(1.0+xt_dot_n*xt_dot_n));// sqrt(det(G^t G))/sqrt(1 + (xt^t n)^2)
        }
}

void parametricMaps_getJacobianValuesTrace1D(proteus_index_t nElements_global,
                                             int nElementBoundaries_element,
                                             int nQuadraturePoints_element,
                                             int nDOF_element,
                                             double* grad_psi,
                                             double* boundaryNormals,
                                             double* boundaryJacobians,
                                             proteus_index_t* l2g,
                                             double* nodeArray,
                                             double* jacobianInverseArray,
                                             double* metricTensorArray,
                                             double* metricTensorDeterminantSqrtArray,
                                             double* unitNormalArray)
{
  proteus_index_t eN, ebN, j_global;  int k, j;
  const int X=0,
    XX=0;
  double *jacInv=NULL,*mt,*mtDetSqrt,*n,*grad=NULL,*node=NULL;
  register double emj=0.0,ebmj=0.0;
  for(eN=0;eN<nElements_global;eN++)
    for(ebN=0;ebN<nElementBoundaries_element;ebN++)
      for(k=0;k<nQuadraturePoints_element;k++)
        {
          emj = 0.0;
          ebmj = 0.0;
          jacInv = jacobianInverseArray +
            eN*nElementBoundaries_element*nQuadraturePoints_element+
            ebN*nQuadraturePoints_element+
            k;
          mt = metricTensorArray +
            eN*nElementBoundaries_element*nQuadraturePoints_element+
            ebN*nQuadraturePoints_element+
            k;
          mtDetSqrt = metricTensorDeterminantSqrtArray +
            eN*nElementBoundaries_element*nQuadraturePoints_element+
            ebN*nQuadraturePoints_element+
            k;
          n = unitNormalArray+
            eN*nElementBoundaries_element*nQuadraturePoints_element+
            ebN*nQuadraturePoints_element+
            k;
          for(j=0;j<nDOF_element;j++)
            {
              j_global = l2g[eN*nDOF_element+
                             j];
              grad = grad_psi + ebN*nQuadraturePoints_element*nDOF_element+k*nDOF_element + j;
              /* nodes are always 3D  */
              node = nodeArray + j_global*3;
              emj += node[X]*grad[X];
            }
          jacInv[XX] = 1.0/(emj);
          mt[XX] = 1.0;
          *mtDetSqrt=1.0;
          n[X] = boundaryNormals[ebN];
        }
}

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
                                                           double* unitNormalArray)
{
  proteus_index_t eN, ebN, j_global;  int ebNE, ebN_local, k, j;
  const int X=0,
    XX=0;
  double *jacInv=NULL,*mt,*mtDetSqrt,*n,*grad=NULL,*node=NULL;
  register double emj=0.0,ebmj=0.0;
  for(ebNE = 0; ebNE < nExteriorElementBoundaries_global; ebNE++)
    {
      ebN = exteriorElementBoundariesArray[ebNE];
      eN  = elementBoundaryElementsArray[ebN*2+0];
      ebN_local = elementBoundaryLocalElementBoundariesArray[ebN*2+0];
      for(k=0;k<nQuadraturePoints_element;k++)
        {
          emj = 0.0;
          ebmj = 0.0;
          jacInv = jacobianInverseArray +
            ebNE*nQuadraturePoints_element+
            k;
          mt = metricTensorArray +
            ebNE*nQuadraturePoints_element+
            k;
          mtDetSqrt = metricTensorDeterminantSqrtArray +
            ebNE*nQuadraturePoints_element+
            k;
          n = unitNormalArray+
            ebNE*nQuadraturePoints_element+
            k;
          for(j=0;j<nDOF_element;j++)
            {
              j_global = l2g[eN*nDOF_element+
                             j];
              grad = grad_psi + ebN_local*nQuadraturePoints_element*nDOF_element+k*nDOF_element + j;
              /* nodes are always 3D  */
              node = nodeArray + j_global*3;
              emj += node[X]*grad[X];
            }
          jacInv[XX] = 1.0/(emj);
          mt[XX] = 1.0;
          *mtDetSqrt=1.0;
          n[X] = boundaryNormals[ebN_local];
        }
    }
}
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
                                                           double* unitNormalArray)
{
  proteus_index_t eN, ebN, j_global;  int ebNE, ebN_local, k, j;
  const int X=0,Y=1,
    XX=0,XY=1,
    YX=2,YY=3;
  double *jacInv=NULL,*mt,*mtDetSqrt,*n,*grad=NULL,*node=NULL,*bn,*bj;
  register double oneOverJacDet=0.0,oneOverNbn=0.0;
  double emj[4],ebmj[2];
  for(ebNE = 0; ebNE < nExteriorElementBoundaries_global; ebNE++)
    {
      ebN = exteriorElementBoundariesArray[ebNE];
      eN  = elementBoundaryElementsArray[ebN*2+0];
      ebN_local = elementBoundaryLocalElementBoundariesArray[ebN*2+0];
      for(k=0;k<nQuadraturePoints_element;k++)
        {
          emj[XX] = 0.0;
          emj[XY] = 0.0;
          emj[YX] = 0.0;
          emj[YY] = 0.0;
          ebmj[XX] = 0.0;
          ebmj[XY] = 0.0;
          jacInv = jacobianInverseArray +
            ebNE*nQuadraturePoints_element*4+
            k*4;
          mt = metricTensorArray +
            ebNE*nQuadraturePoints_element+
            k;
          mtDetSqrt = metricTensorDeterminantSqrtArray +
            ebNE*nQuadraturePoints_element+
            k;
          n = unitNormalArray+
            ebNE*nQuadraturePoints_element*2+
            k*2;
          bn = boundaryNormals + ebN_local*2;
          bj = boundaryJacobians + ebN_local*2;
          for(j=0;j<nDOF_element;j++)
            {
              j_global = l2g[eN*nDOF_element+
                             j];
              grad = grad_psi + ebN_local*nQuadraturePoints_element*nDOF_element*2+k*nDOF_element*2 + j*2;
              /* nodes are always 3D  */
              node = nodeArray + j_global*3;
              emj[XX] += node[X]*grad[X];
              emj[XY] += node[X]*grad[Y];
              emj[YX] += node[Y]*grad[X];
              emj[YY] += node[Y]*grad[Y];
            }
          oneOverJacDet = 1.0/(emj[XX]*emj[YY]- emj[XY]*emj[YX]);
          jacInv[XX] = oneOverJacDet*emj[YY];
          jacInv[YX] = -oneOverJacDet*emj[YX];
          jacInv[XY] = -oneOverJacDet*emj[XY];
          jacInv[YY] = oneOverJacDet*emj[XX];
          ebmj[X] = emj[XX]*bj[XX]+emj[XY]*bj[Y];
          ebmj[Y] = emj[YX]*bj[X]+emj[YY]*bj[Y];
          mt[X] = ebmj[X]*ebmj[X]+ebmj[Y]*ebmj[Y];
          *mtDetSqrt=sqrt(mt[XX]);
          n[X] = jacInv[XX]*bn[X]+jacInv[YX]*bn[Y];
          n[Y] = jacInv[XY]*bn[X]+jacInv[YY]*bn[Y];
          oneOverNbn= 1.0/sqrt(n[X]*n[X]+n[Y]*n[Y]);
          n[X] *= oneOverNbn;
          n[Y] *= oneOverNbn;
        }
    }
}

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
                                                                        double* unitNormalArray)
{
  proteus_index_t eN, ebN, j_global;  int ebNE, ebN_local, k, j;
  const int X=0,Y=1,
    XX=0,XY=1,
    YX=2,YY=3;
  double *jacInv=NULL,*mt,*mtDetSqrt,*n,*grad=NULL,*node=NULL,*bn,*bj,*xt;
  register double oneOverJacDet=0.0,oneOverNbn=0.0,xt_dot_xt,xt_dot_n,xt_dot_bj;
  double emj[4],ebmj[2];
  for(ebNE = 0; ebNE < nExteriorElementBoundaries_global; ebNE++)
    {
      ebN = exteriorElementBoundariesArray[ebNE];
      eN  = elementBoundaryElementsArray[ebN*2+0];
      ebN_local = elementBoundaryLocalElementBoundariesArray[ebN*2+0];
      for(k=0;k<nQuadraturePoints_element;k++)
        {
          emj[XX] = 0.0;
          emj[XY] = 0.0;
          emj[YX] = 0.0;
          emj[YY] = 0.0;
          ebmj[XX] = 0.0;
          ebmj[XY] = 0.0;
          jacInv = jacobianInverseArray +
            ebNE*nQuadraturePoints_element*4+
            k*4;
          mt = metricTensorArray +
            ebNE*nQuadraturePoints_element+
            k;
          mtDetSqrt = metricTensorDeterminantSqrtArray +
            ebNE*nQuadraturePoints_element+
            k;
          n = unitNormalArray+
            ebNE*nQuadraturePoints_element*2+
            k*2;
          xt = xtArray+
            ebNE*nQuadraturePoints_element*3+
            k*3;
          bn = boundaryNormals + ebN_local*2;
          bj = boundaryJacobians + ebN_local*2;
          for(j=0;j<nDOF_element;j++)
            {
              j_global = l2g[eN*nDOF_element+
                             j];
              grad = grad_psi + ebN_local*nQuadraturePoints_element*nDOF_element*2+k*nDOF_element*2 + j*2;
              /* nodes are always 3D  */
              node = nodeArray + j_global*3;
              emj[XX] += node[X]*grad[X];
              emj[XY] += node[X]*grad[Y];
              emj[YX] += node[Y]*grad[X];
              emj[YY] += node[Y]*grad[Y];
            }
          oneOverJacDet = 1.0/(emj[XX]*emj[YY]- emj[XY]*emj[YX]);
          jacInv[XX] = oneOverJacDet*emj[YY];
          jacInv[YX] = -oneOverJacDet*emj[YX];
          jacInv[XY] = -oneOverJacDet*emj[XY];
          jacInv[YY] = oneOverJacDet*emj[XX];
          ebmj[X] = emj[XX]*bj[XX]+emj[XY]*bj[Y];
          ebmj[Y] = emj[YX]*bj[X]+emj[YY]*bj[Y];
          mt[X] = ebmj[X]*ebmj[X]+ebmj[Y]*ebmj[Y];
          n[X] = jacInv[XX]*bn[X]+jacInv[YX]*bn[Y];
          n[Y] = jacInv[XY]*bn[X]+jacInv[YY]*bn[Y];
          oneOverNbn= 1.0/sqrt(n[X]*n[X]+n[Y]*n[Y]);
          n[X] *= oneOverNbn;
          n[Y] *= oneOverNbn;
          xt_dot_xt = xt[X]*xt[X]+xt[Y]*xt[Y];
          xt_dot_bj = xt[X]*ebmj[X]+xt[Y]*ebmj[Y];
          xt_dot_n  = xt[X]*n[X]+xt[Y]*n[Y];
          //printf("%12.5e %12.5e %12.5e\n",xt_dot_xt,xt_dot_bj,xt_dot_n);
          *mtDetSqrt=sqrt(((1.0+xt_dot_xt)*mt[X] - xt_dot_bj*xt_dot_bj)/(1.0+xt_dot_n*xt_dot_n));// sqrt(det(G^t G))/sqrt(1 + (xt^t n)^2)
        }
    }
}
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
                                                           double* unitNormalArray)
{
  proteus_index_t eN, ebN, j_global;  int ebNE, ebN_local, k, j;
  const int
    X=0,Y=1,Z=2,
    XX=0,XY=1,XZ=2,
    YX=3,YY=4,YZ=5,
    ZX=6,ZY=7,ZZ=8,
    XHX=0,XHY=1,
    YHX=2,YHY=3,
    ZHX=4,ZHY=5,
    HXHX=0,HXHY=1,
    HYHX=2,HYHY=3;
  double *jacInv=NULL,*mt=NULL,*mtDetSqrt=NULL,*n=NULL,*grad=NULL,*node=NULL,*bn,*bj;
  register double oneOverJacDet=0.0,oneOverNbn=0.0;
  double emj[9],ebmj[6];
  for(ebNE = 0; ebNE < nExteriorElementBoundaries_global; ebNE++)
    {
      ebN = exteriorElementBoundariesArray[ebNE];
      eN  = elementBoundaryElementsArray[ebN*2+0];
      ebN_local = elementBoundaryLocalElementBoundariesArray[ebN*2+0];
      for(k=0;k<nQuadraturePoints_element;k++)
        {
          emj[XX] = 0.0;
          emj[XY] = 0.0;
          emj[XZ] = 0.0;
          emj[YX] = 0.0;
          emj[YY] = 0.0;
          emj[YZ] = 0.0;
          emj[ZX] = 0.0;
          emj[ZY] = 0.0;
          emj[ZZ] = 0.0;
          ebmj[XHX] = 0.0;
          ebmj[XHY] = 0.0;
          ebmj[YHX] = 0.0;
          ebmj[YHY] = 0.0;
          ebmj[ZHX] = 0.0;
          ebmj[ZHY] = 0.0;
          jacInv = jacobianInverseArray +
            ebNE*nQuadraturePoints_element*9+
            k*9;
          mt = metricTensorArray +
            ebNE*nQuadraturePoints_element*4+
            k*4;
          mtDetSqrt = metricTensorDeterminantSqrtArray +
            ebNE*nQuadraturePoints_element+
            k;
          n = unitNormalArray+
            ebNE*nQuadraturePoints_element*3+
            k*3;
          bn = boundaryNormals + ebN_local*3;
          bj = boundaryJacobians + ebN_local*6;
          for(j=0;j<nDOF_element;j++)
            {
              j_global = l2g[eN*nDOF_element+
                             j];
              grad = grad_psi + ebN_local*nQuadraturePoints_element*nDOF_element*3+k*nDOF_element*3 + j*3;
              /* nodes are always 3D  */
              node = nodeArray + j_global*3;
              emj[XX] += node[X]*grad[X];
              emj[XY] += node[X]*grad[Y];
              emj[XZ] += node[X]*grad[Z];
              emj[YX] += node[Y]*grad[X];
              emj[YY] += node[Y]*grad[Y];
              emj[YZ] += node[Y]*grad[Z];
              emj[ZX] += node[Z]*grad[X];
              emj[ZY] += node[Z]*grad[Y];
              emj[ZZ] += node[Z]*grad[Z];
             }
          oneOverJacDet = 1.0/(emj[XX]*(emj[YY]*emj[ZZ] - emj[YZ]*emj[ZY]) -
                               emj[XY]*(emj[YX]*emj[ZZ] - emj[YZ]*emj[ZX]) +
                               emj[XZ]*(emj[YX]*emj[ZY] - emj[YY]*emj[ZX]));
          jacInv[XX] = oneOverJacDet*(emj[YY]*emj[ZZ] - emj[YZ]*emj[ZY]);
          jacInv[YX] = oneOverJacDet*(emj[YZ]*emj[ZX] - emj[YX]*emj[ZZ]);
          jacInv[ZX] = oneOverJacDet*(emj[YX]*emj[ZY] - emj[YY]*emj[ZX]);
          jacInv[XY] = oneOverJacDet*(emj[ZY]*emj[XZ] - emj[ZZ]*emj[XY]);
          jacInv[YY] = oneOverJacDet*(emj[ZZ]*emj[XX] - emj[ZX]*emj[XZ]);
          jacInv[ZY] = oneOverJacDet*(emj[ZX]*emj[XY] - emj[ZY]*emj[XX]);
          jacInv[XZ] = oneOverJacDet*(emj[XY]*emj[YZ] - emj[XZ]*emj[YY]);
          jacInv[YZ] = oneOverJacDet*(emj[XZ]*emj[YX] - emj[XX]*emj[YZ]);
          jacInv[ZZ] = oneOverJacDet*(emj[XX]*emj[YY] - emj[XY]*emj[YX]);

          ebmj[XHX] = emj[XX]*bj[XHX]+emj[XY]*bj[YHX]+emj[XZ]*bj[ZHX];
          ebmj[XHY] = emj[XX]*bj[XHY]+emj[XY]*bj[YHY]+emj[XZ]*bj[ZHY];
          ebmj[YHX] = emj[YX]*bj[XHX]+emj[YY]*bj[YHX]+emj[YZ]*bj[ZHX];
          ebmj[YHY] = emj[YX]*bj[XHY]+emj[YY]*bj[YHY]+emj[YZ]*bj[ZHY];
          ebmj[ZHX] = emj[ZX]*bj[XHX]+emj[ZY]*bj[YHX]+emj[ZZ]*bj[ZHX];
          ebmj[ZHY] = emj[ZX]*bj[XHY]+emj[ZY]*bj[YHY]+emj[ZZ]*bj[ZHY];

          mt[HXHX] = ebmj[XHX]*ebmj[XHX]+ebmj[YHX]*ebmj[YHX]+ebmj[ZHX]*ebmj[ZHX];
          mt[HXHY] = ebmj[XHX]*ebmj[XHY]+ebmj[YHX]*ebmj[YHY]+ebmj[ZHX]*ebmj[ZHY];
          mt[HYHX] = ebmj[XHY]*ebmj[XHX]+ebmj[YHY]*ebmj[YHX]+ebmj[ZHY]*ebmj[ZHX];
          mt[HYHY] = ebmj[XHY]*ebmj[XHY]+ebmj[YHY]*ebmj[YHY]+ebmj[ZHY]*ebmj[ZHY];

          *mtDetSqrt=sqrt(mt[HXHX]*mt[HYHY]- mt[HXHY]*mt[HYHX]);


          n[X] = (jacInv[XX]*bn[X]+jacInv[YX]*bn[Y]+jacInv[ZX]*bn[Z]);
          n[Y] = (jacInv[XY]*bn[X]+jacInv[YY]*bn[Y]+jacInv[ZY]*bn[Z]);
          n[Z] = (jacInv[XZ]*bn[X]+jacInv[YZ]*bn[Y]+jacInv[ZZ]*bn[Z]);

          oneOverNbn = 1.0/sqrt(n[X]*n[X]+n[Y]*n[Y]+n[Z]*n[Z]);

          n[X] *= oneOverNbn;
          n[Y] *= oneOverNbn;
          n[Z] *= oneOverNbn;

        }
    }
}

/**
   \brief Loop over all the elements and update the element weak_residual
   with the numerical quadrature approximation of the mass integral.

   @param nElements_global The number of elements in the mesh

   @param nQuadraturePoints_element The number of quadrature points
   per element.

   @param nDOF_test_element The number of degrees of freedom in the
   test space per element.

   @param mt (nElements_global x nQuadraturePoints_element). The
   change in mass associated with each quadrature point.

   @param w_dV (nElements_global x nQuadraturePoints_element x
   nDOF_test_element). The weighted finite element test function values.

   @param weak_residual (nElements_global x nDOF_test_element). The element
   weak_residual, which is updated upon return.

   The result of calling this function is

   \f[ residual_{e,i} \mapsto residual_{e,i} + \int_{\Omega_e} \rho
   w_i dV \quad \forall e,i \f]

   where

   \f[ \int_{\Omega_e} \rho w_i dV = \int_{\Omega_r} \rho w_i |J_e|
   d\hat{V} \approx \sum_k m_k (w dV)_{k,i} \f]
*/
void updateMass_weak(proteus_index_t nElements_global,
                     int nQuadraturePoints_element,
                     int nDOF_test_element,
                     double* mt,
                     double* w_dV,
                     double* weak_residual)
{
  proteus_index_t eN;  int i, k;
  for(eN=0;eN<nElements_global;eN++)
    for (i=0;i<nDOF_test_element;i++)
      for (k=0;k<nQuadraturePoints_element;k++)
        weak_residual[eN*nDOF_test_element +
                 i]
          +=
          mt[eN*nQuadraturePoints_element +
            k]
          *
          w_dV[eN*nQuadraturePoints_element*nDOF_test_element +
               k*nDOF_test_element +
               i];
}

/**
   \brief Loop over all the elements and update the element Jacobian
   with the numerical quadrature approximation of the mass integral
   Jacobian.

   @param nElements_global The number of elements in the mesh

   @param nQuadraturePoints_element The number of quadrature points
   per element.

   @param nDOF_test_element The number of degrees of freedom per
   element in the test space.

   @param nDOF_trial_element The number of degrees of freedom per
   element in the trial space.

   @param dmt (nElements_global x nQuadraturePoints_element). The
   derivate of the change in mass associated with each quadrature
   point.

   @param v_X_w_dV (nElements_global x nQuadraturePoints_element x
   nDOF_trial_element). The tensor product of the weighted finite element
   trial and test function values.

   @param jacobian_weak_residual (nElements_global x nDOF_test_element x
   nDOF_trial_element). The element jacobian, which is updated upon
   return.

   The result of calling this function is

   \f[ jacobian_weak_residual_{e,i,j} \mapsto jacobian_weak_residual_{e,i,j} +\int_{\Omega_e}
   \frac{\partial \rho}{\partial u} v_j w_i dV \quad \forall e,i,j \f]

   where

   \f[ \int_{\Omega_e} \frac{\partial \rho}{\partial u} v_j w_i dV =
   \int_{\Omega_r} \frac{\partial \rho}{\partial u} v_j w_i |J_e|
   d\hat{V} \approx \sum_k dm_k (v \otimes w dV)_{k,j,i} \f]
*/
void updateMassJacobian_weak(proteus_index_t nElements_global,
                             int nQuadraturePoints_element,
                             int nDOF_trial_element,
                             int nDOF_test_element,
                             double* dmt,
                             double* v_X_w_dV,
                             double* jacobian_weak_residual)
{
  proteus_index_t nDOF_test_X_trial_element=nDOF_trial_element*nDOF_test_element;  int eN, i, j, k;
  for(eN=0;eN<nElements_global;eN++)
    for (i=0;i<nDOF_test_element;i++)
      for (k=0;k<nQuadraturePoints_element;k++)
        for (j=0;j<nDOF_trial_element;j++)
          jacobian_weak_residual[eN*nDOF_test_X_trial_element +
                   i*nDOF_trial_element+
                   j]
            +=
            dmt[eN*nQuadraturePoints_element +
               k]
            *
            v_X_w_dV[eN*nQuadraturePoints_element*nDOF_test_X_trial_element +
                     k*nDOF_test_X_trial_element +
                     j*nDOF_test_element+
                     i];
}

void updateMassJacobian_weak_lowmem(proteus_index_t nElements_global,
                                    int nQuadraturePoints_element,
                                    int nDOF_trial_element,
                                    int nDOF_test_element,
                                    double* dmt,
                                    double* v,
                                    double* w_dV,
                                    double* jacobian_weak_residual)
{
  proteus_index_t nDOF_test_X_trial_element=nDOF_trial_element*nDOF_test_element;  int eN, i, j, k;
  for(eN=0;eN<nElements_global;eN++)
    for (i=0;i<nDOF_test_element;i++)
      for (k=0;k<nQuadraturePoints_element;k++)
        for (j=0;j<nDOF_trial_element;j++)
          jacobian_weak_residual[eN*nDOF_test_X_trial_element +
                   i*nDOF_trial_element+
                   j]
            +=
            dmt[eN*nQuadraturePoints_element +
               k]
            *
            v[eN*nQuadraturePoints_element*nDOF_trial_element +
              k*nDOF_trial_element +
              j]
            *
            w_dV[eN*nQuadraturePoints_element*nDOF_test_element +
                 k*nDOF_test_element +
                 i];
}

/**
   \brief Loop over all the elements and update the strong from of the
   residual at the quadrature points with the mass term

   @param nElements_global The number of elements in the mesh

   @param nQuadraturePoints_element The number of quadrature points
   per element.

   @param mt (nElements_global x nQuadraturePoints_element) The change
   in mass with respect to time

   @param strong_residual (nElements_global x nQuadraturePoints_element)
   The strong form of the residual at the quadrature points

   The result of calling this function is

   \f[ \mathcal{R}_k \mapsto \mathcal{R}_k + m_k \f]
*/
void updateMass_strong(proteus_index_t nElements_global,
                       int nQuadraturePoints_element,
                       double* mt,
                       double* strong_residual)
{
  proteus_index_t eN;  int k;
  for(eN=0;eN<nElements_global;eN++)
    for (k=0;k<nQuadraturePoints_element;k++)
      strong_residual[eN*nQuadraturePoints_element+
                  k]
        +=
        mt[eN*nQuadraturePoints_element+
          k];
}

/**
   \brief Loop over all the elements and update the Jacobian of the
   strong from of the residual at the quadrature points with the mass
   term

   @param nElements_global The number of elements in the mesh

   @param nQuadraturePoints_element The number of quadrature points
   per element.

   @param nDOF_test_element The number  of trial functions per  element

   @param dmt (nElements_global x nQuadraturePoints_element) The
   derivative with respect to u of the change in mass with respect to
   time

   @param dstrong_residual (nElements_global x nQuadraturePoints_element)
   The strong form of the residual at the quadrature points

   The result of calling this function is

   \f[ \partial{\mathcal{R}}{u}_{k,j} \mapsto
   \partial{\mathcal{R}}{u}_{k,j} + dm_k v_{k,j} \f]
*/
void updateMassJacobian_strong(proteus_index_t nElements_global,
                               int nQuadraturePoints_element,
                               int nDOF_trial_element,
                               double* dmt,
                               double* v,
                               double* dstrong_residual)
{
  proteus_index_t eN;  int k, j;
  for(eN=0;eN<nElements_global;eN++)
    for(j=0;j<nDOF_trial_element;j++)
      for (k=0;k<nQuadraturePoints_element;k++)
        dstrong_residual[eN*nQuadraturePoints_element*nDOF_trial_element+
                     k*nDOF_trial_element +
                     j]
          +=
          dmt[eN*nQuadraturePoints_element+
              k]
          *
          v[eN*nQuadraturePoints_element*nDOF_trial_element+
            k*nDOF_trial_element +
            j];
}

/**
   \brief Loop over all the elements and update the linearized
   adjoint, applied to the weighted test functions, with the mass term

   @param nElements_global The number of elements in the mesh

   @param nQuadraturePoints_element The number of quadrature points
   per element.

   @param nDOF_test_element The number of degrees of freedom in the
   test space per element.

   @param dmt (nElements_global x nQuadraturePoints_element). The
   derivative with respect to u of the change in mass associated with
   each quadrature point.

   @Lstar_w_dV (nElements_global x nQuadraturePoints_element x
   nDOF_test_element). The linearized adjoint applied to the weighted
   element test functions.

   The result of calling the function is

   \f[ \mathcal{L}^* (w dV)_{k,i} \mapsto \mathcal{L}^* (w dV)_{k,i} + \frac{\partial m_t}{\partial u}_{k} (w dV)_{k,i}
   \f]
*/
void updateMass_adjoint(proteus_index_t nElements_global,
                        int nQuadraturePoints_element,
                        int nDOF_test_element,
                        double* dmt,
                        double* w_dV,
                        double* Lstar_w_dV)
{
  proteus_index_t eN;  int i, k;
  for(eN=0;eN<nElements_global;eN++)
    for (i=0;i<nDOF_test_element;i++)
      for (k=0;k<nQuadraturePoints_element;k++)
        Lstar_w_dV[eN*nQuadraturePoints_element*nDOF_test_element +
                   k*nDOF_test_element +
                   i]
          +=
          dmt[eN*nQuadraturePoints_element +
              k]
          *
          w_dV[eN*nQuadraturePoints_element*nDOF_test_element +
               k*nDOF_test_element +
               i];
}


/**
   \brief Loop over all the elements and update the element weak_residual
   with the numerical quadrature approximation of the advection
   integral.

   @param nElements_global The number of elements in the mesh

   @param nQuadraturePoints_element The number of quadrature points
   per element.

   @param nDOF_test_element The number of degrees of freedom per
   element in the test space.

   @param nDOF_trial_element The number of degrees of freedom per
   element in the trial space.

   @param nSpace The number of spatial dimensions.

   @param f (nElements_global x nQuadraturePoints_element x
   nSpace). The advection associated with each quadrature point
   (includes integration weight).

   @param grad_w_dV (nElements_global x nQuadraturePoints_element x
   nDOF_test_element x nSpace). The weighted finite element test
   function gradient values.

   @param weak_residual (nElements_global x nDOF_test_element). The element
   weak_residual, which is updated upon return.

   The result of calling this function is

   \f[ weak_residual_{e,i} \mapsto weak_residual_{e,i} - \int_{\Omega_e}
   \mathbf{f} \cdot \nabla w_i dV \quad \forall e,i \f]

   where

   \f[
   \int_{\Omega_e} \mathbf{f} \cdot \nabla w_i dV =  \int_{\Omega_r} \mathbf{f} \cdot \nabla w_i |J_e| d\hat{V} \int_{\Omega_r} \sum_k \mathbf{f}_k \cdot (\nabla w dV)_i
   \f]
*/
void updateAdvection_weak(proteus_index_t nElements_global,
                          int nQuadraturePoints_element,
                          int nDOF_test_element,
                          int nSpace,
                          double* f,
                          double* grad_w_dV,
                          double* weak_residual)
{
  proteus_index_t eN, I;  int i, k;
  for(eN=0;eN<nElements_global;eN++)
    for (i=0;i<nDOF_test_element;i++)
      for (k=0;k<nQuadraturePoints_element;k++)
        for (I=0;I<nSpace;I++)
          weak_residual[eN*nDOF_test_element +
                   i]
            -=
            f[eN*nQuadraturePoints_element*nSpace +
              k*nSpace +
              I]
            *
            grad_w_dV[eN*nQuadraturePoints_element*nDOF_test_element*nSpace +
                      k*nDOF_test_element*nSpace +
                      i*nSpace +
                      I];
}

/**
   \brief Loop over all the elements and update the element Jacobian
   with the numerical quadrature approximation of the advection
   integral Jacobian.

   @param nElements_global The number of elements in the mesh

   @param nQuadraturePoints_element The number of quadrature points
   per element.

   @param nDOF_test_element The number of degrees of freedom per
   element.

   @param nDOF_trial_element The number of degrees of freedom per
   element.

   @param nSpace The number of spatial dimensions.

   @param df (nElements_global x nQuadraturePoints_element x
   nSpace). The derivative of the advection associated with each
   quadrature point (includes integration weight).

   @param v_X_grad_w_dV (nElements_global x nQuadraturePoints_element
   x nDOF_trial_element x nDOF_test_element x nSpace). The tensor
   product of the finite element trial and test function gradient
   values.

   @param jacobian_weak_residual (nElements_global x nDOF_test_element x
   nDOF_trial_element). The element jacobian, which is updated upon
   return.

   The result of calling this function is

   \f[ jacobian_weak_residual_{e,i,j} \mapsto jacobian_weak_residual_{e,i,j} - \int_{\Omega_e}
   \frac{\partial \mathbf{f}}{\partial u} v_j \cdot \nabla w_i dV
   \quad \forall e,i,j \f]

   where

   \f[ \int_{\Omega_e} \frac{\partial \mathbf{f}}{\partial u} v_j
   \cdot \nabla w_i dV = \int_{\Omega_r} \frac{\partial
   \mathbf{f}}{\partial u} v_j \cdot \nabla w_i |J_e| d\hat{V} \approx
   \sum_k df_k \cdot (v \otimes \nabla w dV)_{j,i} \f]
*/
void updateAdvectionJacobian_weak(proteus_index_t nElements_global,
                                  int nQuadraturePoints_element,
                                  int nDOF_trial_element,
                                  int nDOF_test_element,
                                  int nSpace,
                                  double* df,
                                  double* v_X_grad_w_dV,
                                  double* jacobian_weak_residual)
{
  proteus_index_t nDOF_test_X_trial_element=nDOF_trial_element*nDOF_test_element;  proteus_index_t I; int eN, i, j, k;
  for(eN=0;eN<nElements_global;eN++)
    for (i=0;i<nDOF_test_element;i++)
      for (k=0;k<nQuadraturePoints_element;k++)
        for (j=0;j<nDOF_trial_element;j++)
          for (I=0;I<nSpace;I++)
            jacobian_weak_residual[eN*nDOF_test_X_trial_element +
                     i*nDOF_trial_element+
                     j]
              -=
              df[eN*nQuadraturePoints_element*nSpace +
                 k*nSpace +
                 I]
              *
              v_X_grad_w_dV[eN*nQuadraturePoints_element*nDOF_test_X_trial_element*nSpace +
                            k*nDOF_test_X_trial_element*nSpace +
                            j*nDOF_test_element*nSpace +
                            i*nSpace +
                            I];
}

void updateAdvectionJacobian_weak_lowmem(proteus_index_t nElements_global,
                                         int nQuadraturePoints_element,
                                         int nDOF_trial_element,
                                         int nDOF_test_element,
                                         int nSpace,
                                         double* df,
                                         double* v,
                                         double* grad_w_dV,
                                         double* jacobian_weak_residual)
{
  proteus_index_t nDOF_test_X_trial_element=nDOF_trial_element*nDOF_test_element;  proteus_index_t I; int eN, i, j, k;
  for(eN=0;eN<nElements_global;eN++)
    for (i=0;i<nDOF_test_element;i++)
      for (k=0;k<nQuadraturePoints_element;k++)
        for (j=0;j<nDOF_trial_element;j++)
          for (I=0;I<nSpace;I++)
            jacobian_weak_residual[eN*nDOF_test_X_trial_element +
                     i*nDOF_trial_element+
                     j]
              -=
              df[eN*nQuadraturePoints_element*nSpace +
                 k*nSpace +
                 I]
              *
              v[eN*nQuadraturePoints_element*nDOF_trial_element +
                k*nDOF_trial_element +
                j]
              *
              grad_w_dV[eN*nQuadraturePoints_element*nDOF_test_element*nSpace +
                        k*nDOF_test_element*nSpace +
                        i*nSpace +
                        I];
}

/**
   \brief Loop over all the elements and update the strong form of the
   residual with the advection term

   @param nElements_global The number of elements in the mesh

   @param nQuadraturePoints_element The number of quadrature points
   per element.

   @param nSpace The number of spatial dimensions.

   @param df (nElements_global x nQuadraturePoints_element) The
   derivative of f with respect to u at the quadrature points

   @param grad_u (nElements_global x nQuadraturePoints_element) The
   gradient of u at the quadrature points

   @param strong_residual (nElements_global x nQuadraturePoints_element)
   The strong form of the residual

   The result of calling this  function is

   \f[ \mathcal{R}_k \mapsto \mathcal{R}_k + \frac{\mathbf{f}}{u}_k \cdot \nabla u_k \f]
*/
void updateAdvection_strong(proteus_index_t nElements_global,
                            int nQuadraturePoints_element,
                            int nSpace,
                            double* df,
                            double* grad_u,
                            double* strong_residual)
{
  proteus_index_t eN, I;  int k;
  for(eN=0;eN<nElements_global;eN++)
    for (k=0;k<nQuadraturePoints_element;k++)
      for(I=0;I<nSpace;I++)
        strong_residual[eN*nQuadraturePoints_element+
                    k]
          +=
          df[eN*nQuadraturePoints_element*nSpace +
             k*nSpace +
             I]
          *
          grad_u[eN*nQuadraturePoints_element*nSpace +
                 k*nSpace +
                 I];
}

/**
   \brief Loop over all the elements and update the Jacobian of the strong form of the
   residual with the advection term

   @param nElements_global The number of elements in the mesh

   @param nQuadraturePoints_element The number of quadrature points
   per element.

   @param nSpace The number of spatial dimensions.

   @param df (nElements_global x nQuadraturePoints_element) The
   derivative of f with respect to u at the quadrature points

   @param grad_u (nElements_global x nQuadraturePoints_element) The
   gradient of u at the quadrature points

   @param strong_residual (nElements_global x nQuadraturePoints_element)
   The strong form of the residual

   The result of calling this  function is

   \f[ \mathcal{R}_k \mapsto \mathcal{R}_k + \frac{\mathbf{f}}{u}_k \cdot \nabla u_k \f]
*/
void updateAdvectionJacobian_strong(proteus_index_t nElements_global,
                                    int nQuadraturePoints_element,
                                    int nDOF_trial_element,
                                    int nSpace,
                                    double* df,
                                    double* grad_v,
                                    double* dstrong_residual)
{
  proteus_index_t eN, I;  int k, j;
  for(eN=0;eN<nElements_global;eN++)
    for(j=0;j<nDOF_trial_element;j++)
      for (k=0;k<nQuadraturePoints_element;k++)
        for (I=0;I<nSpace;I++)
          dstrong_residual[eN*nQuadraturePoints_element*nDOF_trial_element+
                           k*nDOF_trial_element +
                           j]
            +=
            df[eN*nQuadraturePoints_element*nSpace +
               k*nSpace +
               I]
            *
            grad_v[eN*nQuadraturePoints_element*nSpace*nDOF_trial_element +
                   k*nSpace*nDOF_trial_element +
                   j*nSpace+
                   I];
}

/**
   \brief Loop over all the elements and update the linearized adjoint
   applied to the weighted test functions with the advection term

   @param nElements_global The number of elements in the mesh

   @param nQuadraturePoints_element The number of quadrature points
   per element.

   @param nDOF_test_element The number of degrees of freedom in the
   test space per element.

   @param nSpace The number of spatial dimensions.

   @param df (nElements_global x nQuadraturePoints_element) The
   derivative of f with respect to u at the quadrature points

   @param grad_w_dV (nElements_global x nQuadraturePoints_element x
   nDOF_test_element) The gradient  of the weighted
   test functions

   @param Lstar_w_dV (nElements_global x nQuadraturePoints_element x
   nDOF_test_element) The linearized adjoint applied to the weighted
   test functions

   The result of  calling this  funciton is

   \f[ \mathcal{L}^* (w dV)_{k,i} \mapsto \mathcal{L}^* (w dV)_{k,i} -
   \frac{\partial \mathbf{f}}{\partial u}_k \cdot \nabla (w dV)_{k,i} \f]
*/
void updateAdvection_adjoint(proteus_index_t nElements_global,
                             int nQuadraturePoints_element,
                             int nDOF_test_element,
                             int nSpace,
                             double* df,
                             double* grad_w_dV,
                             double* Lstar_w_dV)
{
  proteus_index_t eN, I;  int i, k;
  for(eN=0;eN<nElements_global;eN++)
    for (i=0;i<nDOF_test_element;i++)
      for (k=0;k<nQuadraturePoints_element;k++)
        for (I=0;I<nSpace;I++)
          Lstar_w_dV[eN*nQuadraturePoints_element*nDOF_test_element +
                    k*nDOF_test_element +
                    i]
            -=
            df[eN*nQuadraturePoints_element*nSpace +
               k*nSpace +
               I]
            *
            grad_w_dV[eN*nQuadraturePoints_element*nDOF_test_element*nSpace +
                      k*nDOF_test_element*nSpace +
                      i*nSpace +
                      I];
}

/**
   \brief Loop over all the elements and update the element weak_residual
   with the numerical quadrature approximation of the Hamiltonian
   integral.

   @param nElements_global The number of elements in the mesh

   @param nQuadraturePoints_element The number of quadrature points
   per element.

   @param nDOF_test_element The number of degrees of freedom per element.

   @param h (nElements_global x nQuadraturePoints_element). The
   Hamiltonian associated with each quadrature point (includes
   integration weight).

   @param w (nElements_global x nQuadraturePoints_element x
   nDOF_test_element). The finite element test function values.

   @param weak_residual (nElements_global x nDOF_test_element). The element
   weak_residual, which is updated upon return.

   The result of calling this function is

   \f[ weak_residual_{e,i} \mapsto weak_residual_{e,i} - \int_{\Omega_e} h w_i
   dV \quad \forall e,i \f]

   where

   \f[ \int_{\Omega_e} h w_i dV = \int_{\Omega_r} h w_i |J_e| d\hat{V}
   \int_{\Omega_r} \sum_k h_k (w dV)_{k,i} \f]
*/
void updateHamiltonian_weak(proteus_index_t nElements_global,
                            int nQuadraturePoints_element,
                            int nDOF_test_element,
                            double* H,
                            double* w_dV,
                            double* weak_residual)
{
  proteus_index_t eN;  int i, k;
  for(eN=0;eN<nElements_global;eN++)
    for (i=0;i<nDOF_test_element;i++)
      for (k=0;k<nQuadraturePoints_element;k++)
        weak_residual[eN*nDOF_test_element + i]
          +=
          H[eN*nQuadraturePoints_element +
            k]
          *
          w_dV[eN*nQuadraturePoints_element*nDOF_test_element +
            k*nDOF_test_element +
            i];
}

/**
   \brief Loop over all the elements and update the element Jacobian
   with the numerical quadrature approximation of the Hamiltonian
   integral Jacobian.

   @param nElements_global The number of elements in the mesh

   @param nQuadraturePoints_element The number of quadrature points
   per element.

   @param nDOF_test_element The number of degrees of freedom per
   element in the test space.

   @param nDOF_trial_element The number of degrees of freedom per
   element in the trial space.

   @param nSpace The number of spatial dimensions.

   @param dH (nElements_global x nQuadraturePoints_element x
   nSpace). The derivative of the Hamiltonian (with respect to the
   solution gradient) associated with each quadrature point (includes
   integration weight).

   @param grad_v_X_w_dV (nElements_global x nQuadraturePoints_element
   x nDOF_trial_element x nDOF_test_element x nSpace). The tensor
   product of the finite element trial and weighted test function
   gradient values.

   @param jacobian_weak_residual (nElements_global x nDOF_test_element x
   nDOF_trial_element). The element jacobian, which is updated upon
   return.

   The result of calling this function is

   \f[ jacobian_{e,i,j} \mapsto jacobian_{e,i,j} - \int_{\Omega_e}
   \frac{\partial \mathbf{h}}{\partial \nabla u} \cdot \nabla v_j w_i
   dV \quad \forall e,i,j \f]

   where

   \f[ \int_{\Omega_e} \frac{\partial \mathbf{h}}{\partial \nabla u}
   \cdot \nabla v_j \nabla w_i dV = \int_{\Omega_r} \frac{\partial
   \mathbf{h}}{\partial \nabla u} \nabla v_j \cdot w_i |J_e| d\hat{V}
   \approx \sum_k dh_k \cdot (\nabla v \otimes w dV)_{k,j,i} \f]
*/
void updateHamiltonianJacobian_weak(proteus_index_t nElements_global,
                                    int nQuadraturePoints_element,
                                    int nDOF_trial_element,
                                    int nDOF_test_element,
                                    int nSpace,
                                    double* dH,
                                    double* grad_v_X_w_dV,
                                    double* jacobian_weak_residual)
{
  proteus_index_t nDOF_test_X_trial_element=nDOF_test_element*nDOF_trial_element;  proteus_index_t I; int eN, i, j, k;
  for(eN=0;eN<nElements_global;eN++)
    for (i=0;i<nDOF_test_element;i++)
      for (k=0;k<nQuadraturePoints_element;k++)
        for (j=0;j<nDOF_trial_element;j++)
          for (I=0;I<nSpace;I++)
            jacobian_weak_residual[eN*nDOF_test_X_trial_element +
                     i*nDOF_trial_element +
                     j]
              +=
              dH[eN*nQuadraturePoints_element*nSpace +
                 k*nSpace+
                 I]
              *
              grad_v_X_w_dV[eN*nQuadraturePoints_element*nDOF_test_X_trial_element*nSpace +
                            k*nDOF_test_X_trial_element*nSpace +
                            j*nDOF_test_element*nSpace +
                            i*nSpace+
                            I];
}

void updateHamiltonianJacobian_weak_lowmem(proteus_index_t nElements_global,
                                           int nQuadraturePoints_element,
                                           int nDOF_trial_element,
                                           int nDOF_test_element,
                                           int nSpace,
                                           double* dH,
                                           double* grad_v,
                                           double* w_dV,
                                           double* jacobian_weak_residual)
{
  proteus_index_t nDOF_test_X_trial_element=nDOF_test_element*nDOF_trial_element;  proteus_index_t I; int eN, i, j, k;
  for(eN=0;eN<nElements_global;eN++)
    for (i=0;i<nDOF_test_element;i++)
      for (k=0;k<nQuadraturePoints_element;k++)
        for (j=0;j<nDOF_trial_element;j++)
          for (I=0;I<nSpace;I++)
            jacobian_weak_residual[eN*nDOF_test_X_trial_element +
                     i*nDOF_trial_element +
                     j]
              +=
              dH[eN*nQuadraturePoints_element*nSpace +
                 k*nSpace+
                 I]
              *
              grad_v[eN*nQuadraturePoints_element*nDOF_trial_element*nSpace +
                     k*nDOF_trial_element*nSpace +
                     j*nSpace +
                     I]
              *
              w_dV[eN*nQuadraturePoints_element*nDOF_test_element +
                   k*nDOF_test_element +
                   i];
}

/**
   \brief Loop over all the elements and update the strong form of the
   residual with the advection term

   @param nElements_global The number of elements in the mesh

   @param nQuadraturePoints_element The number of quadrature points
   per element.

   @param nSpace The number of spatial dimensions.

   @param dH (nElements_global x nQuadraturePoints_element) The
   derivative of the Hamiltonian at the quadrature points

   @param grad_u (nElements_global x nQuadraturePoints_element) The
   gradient of u at the quadrature points

   @param strong_residual (nElements_global x nQuadraturePoints_element)
   The strong form of the residual

   The result of calling this  function is

   \f[ \mathcal{R}_k \mapsto \mathcal{R}_k + H_k \f]

   \todo  cek check  on whether there isn't a better way to do the strong residual. Why not just  use H?
*/
void updateHamiltonian_strong(proteus_index_t nElements_global,
                              int nQuadraturePoints_element,
                              int nSpace,
                              double* dH,
                              double* grad_u,
                              double* strong_residual)
{
  proteus_index_t eN, I;  int k;
  for(eN=0;eN<nElements_global;eN++)
    for (k=0;k<nQuadraturePoints_element;k++)
      for (I=0;I<nSpace;I++)
        strong_residual[eN*nQuadraturePoints_element+k]
          +=
          dH[eN*nQuadraturePoints_element*nSpace+
             k*nSpace+
             I]
          *
          grad_u[eN*nQuadraturePoints_element*nSpace+
                 k*nSpace+
                 I];
}

/**
   \brief Loop over all the elements and update the Jacobian of the
   strong form of the residual with the advection term

   @param nElements_global The number of elements in the mesh

   @param nQuadraturePoints_element The number of quadrature points
   per element.

   @param nSpace The number of spatial dimensions.

   @param dH (nElements_global x nQuadraturePoints_element) The
   derivative of the Hamiltonian at the quadrature points

   @param grad_v (nElements_global x nQuadraturePoints_element) The
   gradient of the test functions at the quadrature points

   @param strong_residual (nElements_global x nQuadraturePoints_element)
   The strong form of the residual

   The result of calling this  function is

   \f[ \mathcal{R}_k \mapsto \mathcal{R}_k + H_k \f]
*/
void updateHamiltonianJacobian_strong(proteus_index_t nElements_global,
                                      int nQuadraturePoints_element,
                                      int nDOF_trial_element,
                                      int nSpace,
                                      double* dH,
                                      double* grad_v,
                                      double* dstrong_residual)
{
  proteus_index_t eN, I;  int k, j;
  for(eN=0;eN<nElements_global;eN++)
    for(j=0;j<nDOF_trial_element;j++)
      for (k=0;k<nQuadraturePoints_element;k++)
        for (I=0;I<nSpace;I++)
          dstrong_residual[eN*nQuadraturePoints_element*nDOF_trial_element+
                       k*nDOF_trial_element +
                       j]
            +=
            dH[eN*nQuadraturePoints_element*nSpace+
               k*nSpace+
               I]
            *
            grad_v[eN*nQuadraturePoints_element*nDOF_trial_element*nSpace+
                   k*nDOF_trial_element*nSpace +
                   j*nSpace +
                   I];
}

/**
   \brief Loop over all the elements and update the linearized adjoint
   applied to the weighted test functions with the Hamiltonian term

   @param nElements_global The number of elements in the mesh

   @param nQuadraturePoints_element The number of quadrature points
   per element.

   @param nDOF_test_element The number of degrees of freedom in the
   test space per element.

   @param nSpace The number of spatial dimensions.

   @param dH (nElements_global x nQuadraturePoints_element) The
   derivative of H with respect to the gradient of u at the quadrature
   points

   @param grad_w_dV (nElements_global x nQuadraturePoints_element x
   nDOF_test_element x nSpace) The gradient of the weighted
   test functions

   @param Lstar_w_dV (nElements_global x nQuadraturePoints_element x
   nDOF_test_element) The linearized adjoint applied to the weighted
   test functions

   The result of  calling this  funciton is

   \f[ \mathcal{L}^* (w dV)_{k,i} \mapsto \mathcal{L}^* (w dV)_{k,i}
   \frac{\partial \mathbf{H}}{\partial \nabla u}_k \cdot \nabla (w dV)_{k,i} \f]
*/
void updateHamiltonian_adjoint(proteus_index_t nElements_global,
                               int nQuadraturePoints_element,
                               int nDOF_test_element,
                               int nSpace,
                               double* dH,
                               double* grad_w_dV,
                               double* Lstar_w_dV)
{
  proteus_index_t eN, I;  int i, k;
  for(eN=0;eN<nElements_global;eN++)
    for (i=0;i<nDOF_test_element;i++)
      for (k=0;k<nQuadraturePoints_element;k++)
        for (I=0;I<nSpace;I++)
          Lstar_w_dV[eN*nQuadraturePoints_element*nDOF_test_element +
                    k*nDOF_test_element +
                    i]
            -=
            dH[eN*nQuadraturePoints_element*nSpace +
               k*nSpace +
               I]
            *
            grad_w_dV[eN*nQuadraturePoints_element*nDOF_test_element*nSpace +
                      k*nDOF_test_element*nSpace +
                      i*nSpace +
                      I];
}

/**
   \brief Loop over all the elements and update the element weak_residual
   with the numerical quadrature approximation of the diffusion
   integral.

   @param nElements_global The number of elements in the mesh

   @param nQuadraturePoints_element The number of quadrature points
   per element.

   @param nDOF_test_element The number of degrees of freedom per
   element in the test space.

   @param nSpace The number of spatial dimensions.

   @param a (nElements_global x nQuadraturePoints_element x nSpace x
   nSpace). The diffusion coefficient associated with each quadrature
   point (includes integration weight).

   @param grad_phi_X_grad_w_dV (nElements_global x
   nQuadraturePoints_element x nDOF_test_element x nSpace x
   nSpace). The tensor product of potential gradient values and finite
   element test function gradient values.

   @param weak_residual (nElements_global x nDOF_test_element). The element
   weak_residual, which is updated upon return.

   The result of calling this function is

   \f[ weak_residual_{e,i} \mapsto weak_residual_{e,i} + \int_{\Omega_e}
   \bar{\mathbf{a}} \nabla \phi \cdot \nabla w_i dV \quad \forall e,i
   \f]

   where

   \f[ \int_{\Omega_e} \bar{\mathbf{a}} \nabla \phi \cdot \nabla w_i
   dV = \int_{\Omega_r} \bar{\mathbf{a}} \nabla \phi \cdot \nabla w_i
   |J_e| d\hat{V} = \sum_k \bar{\mathbf{a}}_k \cdot (\nabla \phi
   \otimes \nabla w dV)_i \f]
*/
/*#define SCALAR_DIFFUSION*/
void updateDiffusion_weak(proteus_index_t nElements_global,
                          int nQuadraturePoints_element,
                          int nDOF_test_element,
                          int nSpace,
                          double* a,
                          double* grad_phi_X_grad_w_dV,
                          double* weak_residual)
{
  proteus_index_t nSpace2=nSpace*nSpace;  proteus_index_t I, J; int eN, i, k;
  for(eN=0;eN<nElements_global;eN++)
    for (i=0;i<nDOF_test_element;i++)
      for (k=0;k<nQuadraturePoints_element;k++)
        for (I=0;I<nSpace;I++)
          for (J=0;J<nSpace;J++)
            weak_residual[eN*nDOF_test_element + i]
              +=
              a[eN*nQuadraturePoints_element*nSpace2 +
                k*nSpace2 +
                I*nSpace +
                J]
              *
              grad_phi_X_grad_w_dV[eN*nQuadraturePoints_element*nDOF_test_element*nSpace2 +
                                   k*nDOF_test_element*nSpace2 +
                                   i*nSpace2+
                                   J*nSpace +
                                   I];
}

void updateDiffusion_weak_lowmem(proteus_index_t nElements_global,
                                 int nQuadraturePoints_element,
                                 int nDOF_test_element,
                                 int nSpace,
                                 double* a,
                                 double* grad_phi,
                                 double* grad_w_dV,
                                 double* weak_residual)
{
  proteus_index_t nSpace2=nSpace*nSpace;  proteus_index_t I, J; int eN, i, k;
  for(eN=0;eN<nElements_global;eN++)
    for (i=0;i<nDOF_test_element;i++)
      for (k=0;k<nQuadraturePoints_element;k++)
#ifdef SCALAR_DIFFUSION
        for (I=0;I<nSpace;I++)
          {
            J=I;
            weak_residual[eN*nDOF_test_element + i]
              +=
              a[eN*nQuadraturePoints_element*nSpace2 +
                k*nSpace2]
              *
              grad_phi[eN*nQuadraturePoints_element*nSpace +
                       k*nSpace +
                       J]
              *
              grad_w_dV[eN*nQuadraturePoints_element*nDOF_test_element*nSpace +
                        k*nDOF_test_element*nSpace +
                        i*nSpace+
                        I];
          }
#else
        for (I=0;I<nSpace;I++)
          for (J=0;J<nSpace;J++)
            weak_residual[eN*nDOF_test_element + i]
              +=
              a[eN*nQuadraturePoints_element*nSpace2 +
                k*nSpace2 +
                I*nSpace +
                J]
              *
              grad_phi[eN*nQuadraturePoints_element*nSpace +
                       k*nSpace +
                       J]
              *
              grad_w_dV[eN*nQuadraturePoints_element*nDOF_test_element*nSpace +
                        k*nDOF_test_element*nSpace +
                        i*nSpace+
                        I];
#endif
}

void updateDiffusion_weak_sd(proteus_index_t nElements_global,
                             int nQuadraturePoints_element,
                             int nDOF_test_element,
                             int nSpace,
                             proteus_index_t* rowptr,
                             proteus_index_t* colind,
                             double* a,
                             double* grad_phi,
                             double* grad_w_dV,
                             double* weak_residual)
{
  proteus_index_t m, nnz=rowptr[nSpace];  proteus_index_t I; int eN, i, k;
  for(eN=0;eN<nElements_global;eN++)
    for (i=0;i<nDOF_test_element;i++)
      for (k=0;k<nQuadraturePoints_element;k++)
        for (I=0;I<nSpace;I++)
          for (m=rowptr[I];m<rowptr[I+1];m++)
            weak_residual[eN*nDOF_test_element + i]
              +=
              a[eN*nQuadraturePoints_element*nnz+
                k*nnz +
                m]
              *
              grad_phi[eN*nQuadraturePoints_element*nSpace +
                       k*nSpace +
                       colind[m]]
              *
              grad_w_dV[eN*nQuadraturePoints_element*nDOF_test_element*nSpace +
                        k*nDOF_test_element*nSpace +
                        i*nSpace+
                        I];
}

/**
   \brief Loop over all the elements and update the element Jacobian
   with the numerical quadrature approximation of the diffusion
   integral Jacobian.

   @param nElements_global The number of elements in the mesh

   @param nQuadraturePoints_element The number of quadrature points
   per element.

   @param nDOF_test_element The number of degrees of freedom per
   element in the test space.

   @param nDOF_trial_element The number of degrees of freedom per
   element in the trial space.

   @param l2g (nElements_global x nDOF_test_element) The mapping between
   element degrees of freedom and global degrees of freedom.

   @param nSpace The number of spatial dimensions.

   @param a (nElements_global x nQuadraturePoints_element x nSpace x
   nSpace). The diffusion coefficient associated with each quadrature
   point (includes integration weight).

   @param da (nElements_global x nQuadraturePoints_element x nSpace x
   nSpace). The derivative of the diffusion coefficient associated
   with each quadrature point (includes integration weight).

   @param grad_phi_X_grad_w_dV (nElements_global x
   nQuadraturePoints_element x nDOF_test_element x nSpace x
   nSpace). The tensor product of the finite element trial and test
   function gradient values.

   @param dphi (nDOF_global). The global degrees of freedom of phi.

   @param v (nElements_global x nQuadraturePoints_element x
   nDOF_trial_element). The trial function values at each quadrature
   point.

   @param grad_v_X_grad_w_dV (nElements_global x
   nQuadraturePoints_element x nDOF_trial_element x nDOF_test_element
   x nSpace x nSpace). The tensor product of trial function gradient
   values and test function gradient values at each quadrature point.

   @param jacobian_weak_residual (nElements_global x nDOF_test_element x
   nDOF_trial_element). The element jacobian, which is updated upon
   return.

   The result of calling this function is

   \f[ jacobian_{e,i,j} \mapsto jacobian_{e,i,j} + \int_{\Omega_e}
   (\frac{\partial \bar{\mathbf{a}}}{\partial u} v_j \nabla \phi +
   \bar{\mathbf{a}} \frac{\partial \phi}{\partial u} \nabla v_j )
   \cdot \nabla w_i dV \quad \forall e,i,j \f]

   where

   \f[ \int_{\Omega_e} \frac{\partial \bar{\mathbf{a}}}{\partial u}
   v_j \nabla \phi \cdot \nabla w_i dV = \int_{\Omega_r}
   \frac{\partial \bar{\mathbf{a}}}{\partial u} v_j \nabla \phi \cdot
   \nabla w_i |J_e| d\hat{V} \approx \sum_k da_k \cdot (\nabla \phi
   \otimes \nabla w dV)_{j,i} \f]

   \f[ \int_{\Omega_e} \bar{\mathbf{a}}\frac{\partial \phi}{\partial
   u} \nabla v_j \cdot \nabla w_i dV = \int_{\Omega_r}
   \bar{\mathbf{a}} \frac{\partial \phi}{\partial u} \nabla v_j \cdot
   \nabla w_i |J_e| d\hat{V} \approx \sum_k a_k dphi_{j} \cdot (\nabla
   v \otimes \nabla w dV)_{j,i} \f]

   \f[ dphi_j = dphi[l2g[e,j]] \f]
*/
void updateDiffusionJacobian_weak(proteus_index_t nElements_global,
                                  int nQuadraturePoints_element,
                                  int nDOF_trial_element,
                                  int nDOF_test_element,
                                  int nSpace,
                                  proteus_index_t* l2g,
                                  double* a,
                                  double* da,
                                  double* grad_phi_X_grad_w_dV,
                                  double* dphi,
                                  double* v,
                                  double* grad_v_X_grad_w_dV,
                                  double* jacobian_weak_residual)
{
  proteus_index_t nSpace2=nSpace*nSpace, nDOF_test_X_trial_element=nDOF_test_element*nDOF_trial_element;  proteus_index_t I, J; int eN, i, j, k;
  double daProduct,dphiProduct;
  for(eN=0;eN<nElements_global;eN++)
    for (i=0;i<nDOF_test_element;i++)
      for (k=0;k<nQuadraturePoints_element;k++)
        {
          daProduct=0.0;
          for (I=0;I<nSpace;I++)
            for (J=0;J<nSpace;J++)
              daProduct
                +=
                da[eN*nQuadraturePoints_element*nSpace2 +
                   k*nSpace2 +
                   I*nSpace +
                   J]
                *
                grad_phi_X_grad_w_dV[eN*nQuadraturePoints_element*nDOF_test_element*nSpace2 +
                                     k*nDOF_test_element*nSpace2 +
                                     i*nSpace2+
                                     J*nSpace +
                                     I];
          for (j=0;j<nDOF_trial_element;j++)
            {
              dphiProduct=0.0;
              for (I=0;I<nSpace;I++)
                for (J=0;J<nSpace;J++)
                  dphiProduct
                    +=
                    a[eN*nQuadraturePoints_element*nSpace2 +
                      k*nSpace2+
                      I*nSpace +
                      J]
                    *
                    grad_v_X_grad_w_dV[eN*nQuadraturePoints_element*nDOF_test_X_trial_element*nSpace2 +
                                       k*nDOF_test_X_trial_element*nSpace2 +
                                       j*nDOF_test_element*nSpace2 +
                                       i*nSpace2 +
                                       J*nSpace +
                                       I];
              jacobian_weak_residual[eN*nDOF_test_X_trial_element +
                       i*nDOF_trial_element +
                       j]
                +=
                daProduct
                *
                v[eN*nQuadraturePoints_element*nDOF_trial_element+
                  k*nDOF_trial_element+
                  j]
                +
                dphiProduct
                *
                dphi[l2g[eN*nDOF_trial_element +
                         j]];
            }
        }
}

void updateDiffusionJacobian_weak_lowmem(proteus_index_t nElements_global,
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
{
  proteus_index_t nSpace2=nSpace*nSpace, nDOF_test_X_trial_element=nDOF_test_element*nDOF_trial_element;  proteus_index_t I, J; int eN, i, j, k;
  double daProduct,dphiProduct;
  for(eN=0;eN<nElements_global;eN++)
    for (i=0;i<nDOF_test_element;i++)
      for (k=0;k<nQuadraturePoints_element;k++)
        {
          daProduct=0.0;
#ifdef SCALAR_DIFFUSION
          for (I=0;I<nSpace;I++)
            {
              J=I;
              daProduct
                +=
                da[eN*nQuadraturePoints_element*nSpace2 +
                   k*nSpace2]
                *
                grad_phi[eN*nQuadraturePoints_element*nSpace +
                         k*nSpace +
                         J]
                *
                grad_w_dV[eN*nQuadraturePoints_element*nDOF_test_element*nSpace +
                          k*nDOF_test_element*nSpace +
                          i*nSpace+
                          I];
            }
          for (j=0;j<nDOF_trial_element;j++)
            {
              dphiProduct=0.0;
              for (I=0;I<nSpace;I++)
                {
                  J=I;
                  dphiProduct
                    +=
                    a[eN*nQuadraturePoints_element*nSpace2 +
                      k*nSpace2]
                    *
                    grad_v[eN*nQuadraturePoints_element*nDOF_trial_element*nSpace +
                           k*nDOF_trial_element*nSpace +
                           j*nSpace +
                           J]
                    *
                    grad_w_dV[eN*nQuadraturePoints_element*nDOF_test_element*nSpace +
                              k*nDOF_test_element*nSpace +
                              i*nSpace +
                              I];
                }
              jacobian_weak_residual[eN*nDOF_test_X_trial_element +
                       i*nDOF_trial_element +
                       j]
                +=
                daProduct
                *
                v[eN*nQuadraturePoints_element*nDOF_trial_element+
                  k*nDOF_trial_element+
                  j]
                +
                dphiProduct
                *
                dphi[l2g[eN*nDOF_trial_element +
                         j]];
            }
#else
          for (I=0;I<nSpace;I++)
            for (J=0;J<nSpace;J++)
              daProduct
                +=
                da[eN*nQuadraturePoints_element*nSpace2 +
                   k*nSpace2 +
                   I*nSpace +
                   J]
                *
                grad_phi[eN*nQuadraturePoints_element*nSpace +
                         k*nSpace +
                         J]
                *
                grad_w_dV[eN*nQuadraturePoints_element*nDOF_test_element*nSpace +
                          k*nDOF_test_element*nSpace +
                          i*nSpace+
                          I];
          for (j=0;j<nDOF_trial_element;j++)
            {
              dphiProduct=0.0;
              for (I=0;I<nSpace;I++)
                for (J=0;J<nSpace;J++)
                  dphiProduct
                    +=
                    a[eN*nQuadraturePoints_element*nSpace2 +
                      k*nSpace2+
                      I*nSpace +
                      J]
                    *
                    grad_v[eN*nQuadraturePoints_element*nDOF_trial_element*nSpace +
                           k*nDOF_trial_element*nSpace +
                           j*nSpace +
                           J]
                    *
                    grad_w_dV[eN*nQuadraturePoints_element*nDOF_test_element*nSpace +
                              k*nDOF_test_element*nSpace +
                              i*nSpace +
                              I];
              jacobian_weak_residual[eN*nDOF_test_X_trial_element +
                       i*nDOF_trial_element +
                       j]
                +=
                daProduct
                *
                v[eN*nQuadraturePoints_element*nDOF_trial_element+
                  k*nDOF_trial_element+
                  j]
                +
                dphiProduct
                *
                dphi[l2g[eN*nDOF_trial_element +
                         j]];
            }
#endif
        }
}
void updateDiffusionJacobian_weak_sd(proteus_index_t nElements_global,
                                     int nQuadraturePoints_element,
                                     int nDOF_trial_element,
                                     int nDOF_test_element,
                                     int nSpace,
                                     proteus_index_t* rowptr,
                                     proteus_index_t* colind,
                                     proteus_index_t* l2g,
                                     double* a,
                                     double* da,
                                     double* grad_phi,
                                     double* grad_w_dV,
                                     double* dphi,
                                     double* v,
                                     double* grad_v,
                                     double* jacobian_weak_residual)
{
  proteus_index_t m, nDOF_test_X_trial_element=nDOF_test_element*nDOF_trial_element, nnz=rowptr[nSpace];  proteus_index_t I; int eN, i, j, k;
  double daProduct,dphiProduct;
  for(eN=0;eN<nElements_global;eN++)
    for (i=0;i<nDOF_test_element;i++)
      for (k=0;k<nQuadraturePoints_element;k++)
        {
          daProduct=0.0;
          for (I=0;I<nSpace;I++)
            for (m=rowptr[I];m<rowptr[I+1];m++)
              daProduct
                +=
                da[eN*nQuadraturePoints_element*nnz+
                   k*nnz +
                   m]
                *
                grad_phi[eN*nQuadraturePoints_element*nSpace +
                         k*nSpace +
                         colind[m]]
                *
                grad_w_dV[eN*nQuadraturePoints_element*nDOF_test_element*nSpace +
                          k*nDOF_test_element*nSpace +
                          i*nSpace+
                          I];
          for (j=0;j<nDOF_trial_element;j++)
            {
              dphiProduct=0.0;
              for (I=0;I<nSpace;I++)
                for(m=rowptr[I];m<rowptr[I+1];m++)
                  dphiProduct
                    +=
                    a[eN*nQuadraturePoints_element*nnz +
                      k*nnz+
                      m]
                    *
                    grad_v[eN*nQuadraturePoints_element*nDOF_trial_element*nSpace +
                           k*nDOF_trial_element*nSpace +
                           j*nSpace +
                           colind[m]]
                    *
                    grad_w_dV[eN*nQuadraturePoints_element*nDOF_test_element*nSpace +
                              k*nDOF_test_element*nSpace +
                              i*nSpace +
                              I];
              jacobian_weak_residual[eN*nDOF_test_X_trial_element +
                       i*nDOF_trial_element +
                       j]
                +=
                daProduct
                *
                v[eN*nQuadraturePoints_element*nDOF_trial_element+
                  k*nDOF_trial_element+
                  j]
                +
                dphiProduct
                *
                dphi[l2g[eN*nDOF_trial_element +
                         j]];
            }
        }
}

/**
   \brief Loop over all the elements and update the strong form of the
   residual with the diffusion term at the quadrature points

   @param nElements_global The number of elements in the mesh

   @param nQuadraturePoints_element The number of quadrature points
   per element.

   @param nSpace The number of spatial dimensions.

   @param da (nElements_global x nQuadraturePoints_element x nSpace x
   nSpace) The derivative of the diffusion tensor with respect to u

   @param grad_phi (nElements_global x nQuadraturePoints_element x
   nSpace) The gradient of phi

   @param grad_u (nElements_global x nQuadraturePoints_element x
   nSpace) The gradient of u

   The result of calling this function is

   \f[ \mathcal{R}_k \mapsto \mathcal{R}_k - \partial{\mathbf{\:a}}{u}_k \nabla \phi_k \cdot \nabla u_k \f]
*/
void updateDiffusion_strong(proteus_index_t nElements_global,
                            int nQuadraturePoints_element,
                            int nSpace,
                            double* da,
                            double* grad_phi,
                            double* grad_u,
                            double* strong_residual)
{
  proteus_index_t nSpace2=nSpace*nSpace;  proteus_index_t I, J; int eN, k;
  for(eN=0;eN<nElements_global;eN++)
    for (k=0;k<nQuadraturePoints_element;k++)
      for(I=0;I<nSpace;I++)
        for (J=0;J<nSpace;J++)
          strong_residual[eN*nQuadraturePoints_element+
                      k]
            -=
            da[eN*nQuadraturePoints_element*nSpace2 +
               k*nSpace2 +
               I*nSpace+
               J]
            *
            grad_phi[eN*nQuadraturePoints_element*nSpace +
                     k*nSpace +
                     J]
            *grad_u[eN*nQuadraturePoints_element*nSpace +
                    k*nSpace +
                    I];
}
void updateDiffusion_strong_sd(proteus_index_t nElements_global,
                               int nQuadraturePoints_element,
                               int nSpace,
                               proteus_index_t* rowptr,
                               proteus_index_t* colind,
                               double* da,
                               double* grad_phi,
                               double* grad_u,
                               double* strong_residual)
{
  proteus_index_t m, nnz=rowptr[nSpace];  proteus_index_t I; int eN, k;
  for(eN=0;eN<nElements_global;eN++)
    for (k=0;k<nQuadraturePoints_element;k++)
      for(I=0;I<nSpace;I++)
        for(m=rowptr[I];m<rowptr[I+1];m++)
          strong_residual[eN*nQuadraturePoints_element+
                      k]
            -=
            da[eN*nQuadraturePoints_element*nnz +
               k*nnz+
               m]
            *
            grad_phi[eN*nQuadraturePoints_element*nSpace +
                     k*nSpace +
                     colind[m]]
            *grad_u[eN*nQuadraturePoints_element*nSpace +
                    k*nSpace +
                    I];
}

/**
   \brief Loop over all the elements and update the strong form of the
   residual with the diffusion term at the quadrature points

   @param nElements_global The number of elements in the mesh

   @param nQuadraturePoints_element The number of quadrature points
   per element.

   @param nSpace The number of spatial dimensions.

   @param da (nElements_global x nQuadraturePoints_element x nSpace x
   nSpace) The derivative of the diffusion tensor with respect to u

   @param grad_phi (nElements_global x nQuadraturePoints_element x
   nSpace) The gradient of phi

   @param grad_u (nElements_global x nQuadraturePoints_element x
   nSpace) The gradient of u

   The result of calling this function is

   \f[ \mathcal{R}_k \mapsto \mathcal{R}_k - \partial{\mathbf{\:a}}{u}_k \nabla \phi_k \cdot \nabla u_k \f]
*/
void updateDiffusionJacobian_strong(proteus_index_t nElements_global,
                                    int nQuadraturePoints_element,
                                    int nDOF_trial_element,
                                    int nSpace,
                                    proteus_index_t* l2g,
                                    double* da,
                                    double* dphi,
                                    double* grad_phi,
                                    double* grad_u,
                                    double* grad_v,
                                    double* dstrong_residual)
{
  proteus_index_t nSpace2=nSpace*nSpace;  proteus_index_t I, J; int eN, k, j;
  for(eN=0;eN<nElements_global;eN++)
    for (k=0;k<nQuadraturePoints_element;k++)
      for(j=0;j<nDOF_trial_element;j++)
        for(I=0;I<nSpace;I++)
          for(J=0;J<nSpace;J++)
            {
             dstrong_residual[eN*nQuadraturePoints_element*nDOF_trial_element+
                             k*nDOF_trial_element +
                             j]
              -=
               da[eN*nQuadraturePoints_element*nSpace2 +
                  k*nSpace2 +
                  I*nSpace+
                  J]
              *
              (
               grad_phi[eN*nQuadraturePoints_element*nSpace +
                        k*nSpace +
                        J]
               *
               grad_v[eN*nQuadraturePoints_element*nSpace*nDOF_trial_element +
                      k*nSpace*nDOF_trial_element +
                      j*nSpace +
                      I]
               +
               dphi[l2g[eN*nDOF_trial_element +
                        j]]*
               grad_v[eN*nQuadraturePoints_element*nSpace*nDOF_trial_element +
                      k*nSpace*nDOF_trial_element +
                      j*nSpace +
                      J]
               *
               grad_u[eN*nQuadraturePoints_element*nSpace+
                      k*nSpace+
                      I]
               );
            }
}
void updateDiffusionJacobian_strong_sd(proteus_index_t nElements_global,
                                       int nQuadraturePoints_element,
                                       int nDOF_trial_element,
                                       int nSpace,
                                       proteus_index_t* rowptr,
                                       proteus_index_t* colind,
                                       proteus_index_t* l2g,
                                       double* da,
                                       double* dphi,
                                       double* grad_phi,
                                       double* grad_u,
                                       double* grad_v,
                                       double* dstrong_residual)
{
  proteus_index_t m, nnz=rowptr[nSpace];  proteus_index_t I; int eN, k, j;
  for(eN=0;eN<nElements_global;eN++)
    for (k=0;k<nQuadraturePoints_element;k++)
      for(j=0;j<nDOF_trial_element;j++)
        for(I=0;I<nSpace;I++)
          for(m=rowptr[I];m<rowptr[I+1];m++)
            {
             dstrong_residual[eN*nQuadraturePoints_element*nDOF_trial_element+
                             k*nDOF_trial_element +
                             j]
              -=
               da[eN*nQuadraturePoints_element*nnz +
                  k*nnz+
                  m]
              *
              (
               grad_phi[eN*nQuadraturePoints_element*nSpace +
                        k*nSpace +
                        colind[m]]
               *
               grad_v[eN*nQuadraturePoints_element*nSpace*nDOF_trial_element +
                      k*nSpace*nDOF_trial_element +
                      j*nSpace +
                      I]
               +
               dphi[l2g[eN*nDOF_trial_element +
                        j]]*
               grad_v[eN*nQuadraturePoints_element*nSpace*nDOF_trial_element +
                      k*nSpace*nDOF_trial_element +
                      j*nSpace +
                      colind[m]]
               *
               grad_u[eN*nQuadraturePoints_element*nSpace+
                      k*nSpace+
                      I]
               );
            }
}


/**
   \brief Loop over all the elements and update the linearized
   adjoint applied to the weighted test function with the diffusion
   term

   @param nElements_global The number of elements in the mesh

   @param nQuadraturePoints_element The number of quadrature points
   per element.

   @param nDOF_test_element The number of degrees of freedom in the
   test space per element.

   @param nSpace The number of spatial dimensions.

   @param da (nElements_global x nQuadraturePoints_element x nSpace x
   nSpace) The derivative of the diffusion tensor with respect to u

   @param grad_phi (nElements_global x nQuadraturePoints_element  x nSpace) The gradient of  phi

   @param grad_w_dV (nElements_global x nQuadraturePoints_element x
   nDOF_test_element x nSpace) The gradient of the weighted test
   functions

   @param Lstar_w_dV (nElements_global x nQuadraturePoints_element x
   nDOF_test_element) The linearized adjoint applied to the weighted
   test functions

   The result of  calling this  function is

   \f[ \mathcal{L}^* (w dV)_{k,i} \mapsto \mathcal{L}^* (w dV)_{k,i} + \frac{\mathbf{\:a}}{u} \nabla \phi \cdot \nabla (w dV)_{k,i} \f]
 */
void updateDiffusion_adjoint(proteus_index_t nElements_global,
                             int nQuadraturePoints_element,
                             int nDOF_test_element,
                             int nSpace,
                             double* da,
                             double* grad_phi,
                             double* grad_w_dV,
                             double* Lstar_w_dV)
{
  proteus_index_t nSpace2=nSpace*nSpace;  proteus_index_t I, J; int eN, i, k;
  for(eN=0;eN<nElements_global;eN++)
    for (i=0;i<nDOF_test_element;i++)
      for (k=0;k<nQuadraturePoints_element;k++)
        for (I=0;I<nSpace;I++)
          for(J=0;J<nSpace;J++)
            Lstar_w_dV[eN*nQuadraturePoints_element*nDOF_test_element +
                      k*nDOF_test_element +
                      i]
              +=
              da[eN*nQuadraturePoints_element*nSpace2 +
                 k*nSpace2 +
                 I*nSpace+
                 J]
              *
              grad_phi[eN*nQuadraturePoints_element*nSpace+
                       k*nSpace+
                       J]
              *
              grad_w_dV[eN*nQuadraturePoints_element*nDOF_test_element*nSpace +
                        k*nDOF_test_element*nSpace +
                        i*nSpace +
                        I];
}

void updateDiffusion_adjoint_sd(proteus_index_t nElements_global,
                                int nQuadraturePoints_element,
                                int nDOF_test_element,
                                int nSpace,
                                proteus_index_t* rowptr,
                                proteus_index_t* colind,
                                double* da,
                                double* grad_phi,
                                double* grad_w_dV,
                                double* Lstar_w_dV)
{
  proteus_index_t m, nnz=rowptr[nSpace];  proteus_index_t I; int eN, i, k;
  for(eN=0;eN<nElements_global;eN++)
    for (i=0;i<nDOF_test_element;i++)
      for (k=0;k<nQuadraturePoints_element;k++)
        for (I=0;I<nSpace;I++)
          for(m=rowptr[I];m<rowptr[I+1];m++)
            Lstar_w_dV[eN*nQuadraturePoints_element*nDOF_test_element +
                      k*nDOF_test_element +
                      i]
              +=
              da[eN*nQuadraturePoints_element*nnz +
                 k*nnz+
                 m]
              *
              grad_phi[eN*nQuadraturePoints_element*nSpace+
                       k*nSpace+
                       colind[m]]
              *
              grad_w_dV[eN*nQuadraturePoints_element*nDOF_test_element*nSpace +
                        k*nDOF_test_element*nSpace +
                        i*nSpace +
                        I];
}

void updateDiffusion2_strong(proteus_index_t nElements_global,
                             int nQuadraturePoints_element,
                             int nSpace,
                             double* a,
                             double* Hess_phi,
                             double* strong_residual)
{
  proteus_index_t nSpace2=nSpace*nSpace;  proteus_index_t I, J; int eN, k;
  for(eN=0;eN<nElements_global;eN++)
    for (k=0;k<nQuadraturePoints_element;k++)
      for(I=0;I<nSpace;I++)
        for (J=0;J<nSpace;J++)
          {
            strong_residual[eN*nQuadraturePoints_element+
                            k]
              -=
              a[eN*nQuadraturePoints_element*nSpace2 +
                k*nSpace2 +
                I*nSpace+
                J]
              *
              Hess_phi[eN*nQuadraturePoints_element*nSpace2 +
                       k*nSpace2 +
                       J*nSpace+
                       I];
          }
}

void updateDiffusion2_strong_sd(proteus_index_t nElements_global,
                                int nQuadraturePoints_element,
                                int nSpace,
                                proteus_index_t* rowptr,
                                proteus_index_t* colind,
                                double* a,
                                double* Hess_phi,
                                double* strong_residual)
{
  proteus_index_t m, nSpace2=nSpace*nSpace, nnz=rowptr[nSpace];  proteus_index_t I; int eN, k;
  for(eN=0;eN<nElements_global;eN++)
    for (k=0;k<nQuadraturePoints_element;k++)
      for(I=0;I<nSpace;I++)
        for (m=rowptr[I];m<rowptr[I+1];m++)
          {
            strong_residual[eN*nQuadraturePoints_element+
                            k]
              -=
              a[eN*nQuadraturePoints_element*nnz +
                k*nnz+
                m]
              *
              Hess_phi[eN*nQuadraturePoints_element*nSpace2 +
                       k*nSpace2 +
                       colind[m]*nSpace+
                       I];
          }
}

void updateDiffusionJacobian2_strong(proteus_index_t nElements_global,
                                     int nQuadraturePoints_element,
                                     int nDOF_trial_element,
                                     int nSpace,
                                     proteus_index_t* l2g,
                                     double* a,
                                     double* da,
                                     double* v,
                                     double* Hess_phi,
                                     double* dphi,
                                     double* Hess_v,
                                     double* dstrong_residual)
{
  proteus_index_t nSpace2=nSpace*nSpace;  proteus_index_t I, J; int eN, k, j;
  /*double tmp;*/
  for(eN=0;eN<nElements_global;eN++)
    for (k=0;k<nQuadraturePoints_element;k++)
      for (j=0;j<nDOF_trial_element;j++)
        for(I=0;I<nSpace;I++)
          for (J=0;J<nSpace;J++)
            {
              dstrong_residual[eN*nQuadraturePoints_element*nDOF_trial_element+
                               k*nDOF_trial_element+
                               j]
                -=
                (da[eN*nQuadraturePoints_element*nSpace2 +
                    k*nSpace2 +
                    I*nSpace+
                    J]
                 *
                 v[eN*nQuadraturePoints_element*nDOF_trial_element+
                   k*nDOF_trial_element+
                   j]
                 *
                 Hess_phi[eN*nQuadraturePoints_element*nSpace2 +
                          k*nSpace2 +
                          J*nSpace+
                          I]
                 +
                 a[eN*nQuadraturePoints_element*nSpace2 +
                   k*nSpace2 +
                   I*nSpace+
                   J]
                 *
                 dphi[l2g[eN*nDOF_trial_element +
                          j]]
                 *
                 Hess_v[eN*nQuadraturePoints_element*nDOF_trial_element*nSpace2 +
                        k*nDOF_trial_element*nSpace2 +
                        j*nSpace2+
                        J*nSpace+
                        I]);
            }
}

void updateDiffusionJacobian2_strong_sd(proteus_index_t nElements_global,
                                        int nQuadraturePoints_element,
                                        int nDOF_trial_element,
                                        int nSpace,
                                        proteus_index_t* rowptr,
                                        proteus_index_t* colind,
                                        proteus_index_t* l2g,
                                        double* a,
                                        double* da,
                                        double* v,
                                        double* Hess_phi,
                                        double* dphi,
                                        double* Hess_v,
                                        double* dstrong_residual)
{
  proteus_index_t m, nSpace2=nSpace*nSpace, nnz=rowptr[nSpace];  proteus_index_t I; int eN, k, j;
  /*double tmp;*/
  for(eN=0;eN<nElements_global;eN++)
    for (k=0;k<nQuadraturePoints_element;k++)
      for (j=0;j<nDOF_trial_element;j++)
        for(I=0;I<nSpace;I++)
          for (m=rowptr[I];m<rowptr[I+1];m++)
            {
              dstrong_residual[eN*nQuadraturePoints_element*nDOF_trial_element+
                               k*nDOF_trial_element+
                               j]
                -=
                (da[eN*nQuadraturePoints_element*nnz+
                    k*nnz+
                    m]
                 *
                 v[eN*nQuadraturePoints_element*nDOF_trial_element+
                   k*nDOF_trial_element+
                   j]
                 *
                 Hess_phi[eN*nQuadraturePoints_element*nSpace2 +
                          k*nSpace2 +
                          colind[m]*nSpace+
                          I]
                 +
                 a[eN*nQuadraturePoints_element*nnz+
                   k*nnz+
                   m]
                 *
                 dphi[l2g[eN*nDOF_trial_element +
                          j]]
                 *
                 Hess_v[eN*nQuadraturePoints_element*nDOF_trial_element*nSpace2 +
                        k*nDOF_trial_element*nSpace2 +
                        j*nSpace2+
                        colind[m]*nSpace+
                        I]);
            }
}

void updateDiffusion2_adjoint(proteus_index_t nElements_global,
                              int nQuadraturePoints_element,
                              int nDOF_test_element,
                              int nSpace,
                              double* a,
                              double* Hess_w_dV,
                              double* Lstar_w_dV)
{
  proteus_index_t nSpace2=nSpace*nSpace;  proteus_index_t I, J; int eN, i, k;
  for(eN=0;eN<nElements_global;eN++)
    for (i=0;i<nDOF_test_element;i++)
      for (k=0;k<nQuadraturePoints_element;k++)
        for (I=0;I<nSpace;I++)
          for(J=0;J<nSpace;J++)
            {
              Lstar_w_dV[eN*nQuadraturePoints_element*nDOF_test_element +
                         k*nDOF_test_element +
                         i]
                -=
                a[eN*nQuadraturePoints_element*nSpace2 +
                  k*nSpace2 +
                  I*nSpace+
                  J]
                *
                Hess_w_dV[eN*nQuadraturePoints_element*nDOF_test_element*nSpace2 +
                          k*nDOF_test_element*nSpace2 +
                          i*nSpace2+
                          I*nSpace +
                          J];
            }
}


void updateDiffusion2_adjoint_sd(proteus_index_t nElements_global,
                                 int nQuadraturePoints_element,
                                 int nDOF_test_element,
                                 int nSpace,
                                 proteus_index_t* rowptr,
                                 proteus_index_t* colind,
                                 double* a,
                                 double* Hess_w_dV,
                                 double* Lstar_w_dV)
{
  proteus_index_t m, nSpace2=nSpace*nSpace, nnz=rowptr[nSpace];  proteus_index_t I; int eN, i, k;
  for(eN=0;eN<nElements_global;eN++)
    for (i=0;i<nDOF_test_element;i++)
      for (k=0;k<nQuadraturePoints_element;k++)
        for (I=0;I<nSpace;I++)
          for(m=rowptr[I];m<rowptr[I+1];m++)
            {
              Lstar_w_dV[eN*nQuadraturePoints_element*nDOF_test_element +
                         k*nDOF_test_element +
                         i]
                -=
                a[eN*nQuadraturePoints_element*nnz+
                  k*nnz+
                  m]
                *
                Hess_w_dV[eN*nQuadraturePoints_element*nDOF_test_element*nSpace2 +
                          k*nDOF_test_element*nSpace2 +
                          i*nSpace2+
                          I*nSpace +
                          colind[m]];
            }
}


/**
   \brief Loop over all the elements and update the element weak_residual
   with the numerical quadrature approximation of the reaction
   integral.

   @param nElements_global The number of elements in the mesh

   @param nQuadraturePoints_element The number of quadrature points
   per element.

   @param nDOF_test_element The number of degrees of freedom per
   element.

   @param r (nElements_global x nQuadraturePoints_element). The
   reaction associated with each quadrature point (includes
   integration weight).

   @param w (nElements_global x nQuadraturePoints_element x
   nDOF_test_element). The finite element test function values.

   @param weak_residual (nElements_global x nDOF_test_element). The element
   weak_residual, which is updated upon return.

   The result of calling this function is

   \f[ weak_residual_{e,i} \mapsto weak_residual_{e,i} + \int_{\Omega_e} r w_i
   dV \quad \forall e,i \f]

   where

   \f[ \int_{\Omega_e} r w_i dV = \int_{\Omega_r} r w_i |J_e| d\hat{V}
   \approx \sum_k r_k (w dV)_i \f]
*/
void updateReaction_weak(proteus_index_t nElements_global,
                         int nQuadraturePoints_element,
                         int nDOF_test_element,
                         double* r,
                         double* w_dV,
                         double* weak_residual)
{
  updateMass_weak(nElements_global,nQuadraturePoints_element,nDOF_test_element,r,w_dV,weak_residual);
}

/**
   \brief Loop over all the elements and update the element Jacobian
   with the numerical quadrature approximation of the reaction
   integral Jacobian.

   @param nElements_global The number of elements in the mesh

   @param nQuadraturePoints_element The number of quadrature points
   per element.

   @param nDOF_test_element The number of degrees of freedom per
   element in the test space.

   @param nDOF_trial_element The number of degrees of freedom per
   element in the trial space.

   @param dr (nElements_global x nQuadraturePoints_element). The
   derivate of the reaction associated with each quadrature point
   (includes integration weight).

   @param v_X_w_dV (nElements_global x nQuadraturePoints_element x
   nDOF_trial_element x nDOF_test_element). The tensor product of the
   finite element trial and test function values.

   @param jacobian_weak_residual (nElements_global x nDOF_test_element x
   nDOF_trial_element). The element jacobian, which is updated upon
   return.

   The result of calling this function is

   \f[ jacobian_{e,i,j} \mapsto jacobian_{e,i,j} +\int_{\Omega_e}
   \frac{\partial r}{\partial u} v_j w_i dV \quad \forall e,i,j \f]

   where

   \f[ \int_{\Omega_e} \frac{\partial r}{\partial u} v_j w_i dV =
   \int_{\Omega_r} \frac{\partial r}{\partial u} v_j w_i |J_e|
   d\hat{V} \approx \sum_k dr_k (v \otimes w dV)_{j,i} \f]
*/
void updateReactionJacobian_weak(proteus_index_t nElements_global,
                                 int nQuadraturePoints_element,
                                 int nDOF_trial_element,
                                 int nDOF_test_element,
                                 double* dr,
                                 double* v_X_w_dV,
                                 double* jacobian_weak_residual)
{
  updateMassJacobian_weak(nElements_global,nQuadraturePoints_element,nDOF_trial_element,nDOF_test_element,dr,v_X_w_dV,jacobian_weak_residual);
}

void updateReactionJacobian_weak_lowmem(proteus_index_t nElements_global,
                                        int nQuadraturePoints_element,
                                        int nDOF_trial_element,
                                        int nDOF_test_element,
                                        double* dr,
                                        double* v,
                                        double* w_dV,
                                        double* jacobian_weak_residual)
{
  updateMassJacobian_weak_lowmem(nElements_global,nQuadraturePoints_element,nDOF_trial_element,nDOF_test_element,dr,
                                 v,w_dV,jacobian_weak_residual);
}

/**
   \brief Loop over all the elements and update the strong from of the
   residual at the quadrature points with the reaction term

   @param nElements_global The number of elements in the mesh

   @param nQuadraturePoints_element The number of quadrature points
   per element.

   @param r (nElements_global x nQuadraturePoints_element) The reaction term

   @param strong_residual (nElements_global x nQuadraturePoints_element)
   The strong form of the residual at the quadrature points

   The result of calling this function is

   \f[ \mathcal{R}_k \mapsto \mathcal{R}_k + r_k \f]
*/
void updateReaction_strong(proteus_index_t nElements_global,
                           int nQuadraturePoints_element,
                           double* r,
                           double* strong_residual)
{
  proteus_index_t eN;  int k;
  for(eN=0;eN<nElements_global;eN++)
    for (k=0;k<nQuadraturePoints_element;k++)
      strong_residual[eN*nQuadraturePoints_element+
                  k]
        +=
        r[eN*nQuadraturePoints_element+
          k];
}

/**
   \brief Loop over all the elements and update the strong from of the
   residual at the quadrature points with the reaction term

   @param nElements_global The number of elements in the mesh

   @param nQuadraturePoints_element The number of quadrature points
   per element.

   @param r (nElements_global x nQuadraturePoints_element) The reaction term

   @param strong_residual (nElements_global x nQuadraturePoints_element)
   The strong form of the residual at the quadrature points

   The result of calling this function is

   \f[ \frac{\partial \mathcal{R}}{\partial u}_{k,j} \mapsto
   \frac{\partial \mathcal{R}}{\partial u}_{k,j} + dr_k (v dV)_{k,j} \f]
*/
void updateReactionJacobian_strong(proteus_index_t nElements_global,
                                   int nQuadraturePoints_element,
                                   int nDOF_trial_element,
                                   double* dr,
                                   double* v,
                                   double* dstrong_residual)
{
  proteus_index_t eN;  int k, j;
  for(eN=0;eN<nElements_global;eN++)
    for(j=0;j<nDOF_trial_element;j++)
      for (k=0;k<nQuadraturePoints_element;k++)
        dstrong_residual[eN*nQuadraturePoints_element*nDOF_trial_element+
                     k*nDOF_trial_element +
                     j]
          +=
          dr[eN*nQuadraturePoints_element+
             k]
          *
          v[eN*nQuadraturePoints_element*nDOF_trial_element+
            k*nDOF_trial_element +
            j];
}

/**
   \brief Loop over all the elements and update the linearized
   adjoint, applied to the weighted test functions, with the mass term

   @param nElements_global The number of elements in the mesh

   @param nQuadraturePoints_element The number of quadrature points
   per element.

   @param nDOF_test_element The number of degrees of freedom in the
   test space per element.

   @param dr (nElements_global x nQuadraturePoints_element). The
   derivative of  r with respect to u

   @Lstar_w_dV (nElements_global x nQuadraturePoints_element x
   nDOF_test_element). The linearized adjoint applied to the weighted
   element test functions.

   The result of calling the function is

   \f[ \mathcal{L}^* (w dV)_{k,i} \mapsto \mathcal{L}^* (w dV)_{k,i} + \frac{\partial r}{\partial u}_{k} (w dV)_{k,i}
   \f]
*/
void updateReaction_adjoint(proteus_index_t nElements_global,
                            int nQuadraturePoints_element,
                            int nDOF_test_element,
                            double* dr,
                            double* w_dV,
                            double* Lstar_w_dV)
{
  proteus_index_t eN;  int i, k;
  for(eN=0;eN<nElements_global;eN++)
    for (i=0;i<nDOF_test_element;i++)
      for (k=0;k<nQuadraturePoints_element;k++)
        Lstar_w_dV[eN*nQuadraturePoints_element*nDOF_test_element +
                   k*nDOF_test_element +
                   i]
          +=
          dr[eN*nQuadraturePoints_element +
             k]
          *
          w_dV[eN*nQuadraturePoints_element*nDOF_test_element +
               k*nDOF_test_element +
               i];
}


/**
   \brief Loop over all the elements and update the element weak_residual
   with the numerical quadrature approximation of the stabilization
   integral.

   @param nElements_global The number of elements in the mesh

   @param nQuadraturePoints_element The number of quadrature points
   per element.

   @param nDOF_test_element The number of degrees of freedom per element.

   @param error (nElements_global x nQuadraturePoints_element). The
   error approximation associated with each quadrature point.

   @param LstarW (nElements_global x nQuadraturePoints_element x
   nDOF_test_element). The linearized adjoint applied to the weighted
   finite element test function values.

   @param weak_residual (nElements_global x nDOF_test_element). The element
   weak_residual, which is updated upon return.

   The result of calling this function is

   \f[ weak_residual_{e,i} \mapsto weak_residual_{e,i} - \int_{\Omega_e}
   \mathcal{e} \mathcal{L^*}(w_i) dV \quad \forall e,i \f]

   where

   \f[ \int_{\Omega_e} \tau \mathcal{R} \mathcal{L^*}(w_i) dV =
   \int_{\Omega_r} \mathcal{e} \mathcal{L^*}(w_i) |J_e| d\hat{V}
   \approx \sum_k \tau_k \mathcal{R}_k \mathcal{L^*}(w dV)_i \f]
*/
void updateSubgridError(proteus_index_t nElements_global,
                         int nQuadraturePoints_element,
                         int nDOF_test_element,
                         double* error,
                         double* Lstar_w_dV,
                         double* weak_residual)
{
  proteus_index_t eN;  int i, k;
  for(eN=0;eN<nElements_global;eN++)
    for (i=0;i<nDOF_test_element;i++)
      for (k=0;k<nQuadraturePoints_element;k++)
        weak_residual[eN*nDOF_test_element + i]
          +=
          error[eN*nQuadraturePoints_element +
                k]
          *
          Lstar_w_dV[eN*nQuadraturePoints_element*nDOF_test_element +
                     k*nDOF_test_element +
                     i];
}

/**
   \brief Loop over all the elements and update the element Jacobian
   with the numerical quadrature approximation of the stabilization
   integral Jacobian.

   @param nElements_global The number of elements in the mesh

   @param nQuadraturePoints_element The number of quadrature points
   per element.

   @param nDOF_test_element The number of degrees of freedom per
   element in the test space.

   @param nDOF_trial_element The number of degrees of freedom per
   element in the trial space.

   @param error (nElements_global x nQuadraturePoints_element). The
   error approximation associated with each quadrature point

   @param Lstar_w_dV (nElements_global x nQuadraturePoints_element x
   nDOF_test_element). The linearized adjoint applied to the weighted
   finite element test function values.

   @param jacobian_weak_residual (nElements_global x nDOF_test_element x
   nDOF_trial_element). The element jacobian, which is updated upon return.

   The result of calling this function is
   \f[
   jacobian_{e,i,j} \mapsto jacobian_{e,i,j} - \int_{\Omega_e} \tau \frac{\partial \mathcal{R}}{\partial u_j} \mathcal{L^*}(w dV){k,i} \quad \forall e,i
   \f]

   where

   \f[ \int_{\Omega_e} \frac{\partial \mathcal{e}}{\partial u_j}
   \mathcal{L^*}(w_i) dV = \int_{\Omega_r} \frac{\partial
   \mathcal{e}}{\partial u_j} \mathcal{L^*}(w_i) |J_e| d\hat{V}
   \approx \sum_k (\frac{\partial \mathcal{e}}{\partial u_j})_k
   \mathcal{L^*}(w dV)_i \f]
*/
void updateSubgridErrorJacobian(proteus_index_t nElements_global,
                                 int nQuadraturePoints_element,
                                 int nDOF_trial_element,
                                 int nDOF_test_element,
                                 double* derror,
                                 double* Lstar_w_dV,
                                 double* jacobian_weak_residual)
{
  proteus_index_t nDOF_test_X_trial_element=nDOF_test_element*nDOF_trial_element;  int eN, i, j, k;
  for(eN=0;eN<nElements_global;eN++)
    for (i=0;i<nDOF_test_element;i++)
      for (k=0;k<nQuadraturePoints_element;k++)
        for (j=0;j<nDOF_trial_element;j++)
          jacobian_weak_residual[eN*nDOF_test_X_trial_element +
                   i*nDOF_trial_element +
                   j]
            +=
            derror[eN*nQuadraturePoints_element*nDOF_trial_element+
                   k*nDOF_trial_element+
                   j]
            *
            Lstar_w_dV[eN*nQuadraturePoints_element*nDOF_test_element +
                   k*nDOF_test_element +
                   i];
}

/**
   \brief Loop over all the elements and update the element weak_residual
   with the numerical quadrature approximation of the shock capturing
   integral.

   @param nElements_global The number of elements in the mesh

   @param nQuadraturePoints_element The number of quadrature points
   per element.

   @param nDOF_test_element The number of degrees of freedom per
   element.

   @param nSpace The number of spatial dimensions.

   @param numDiff (nElements_global x nQuadraturePoints_element). The
   shock capturing diffusion associated with each quadrature point
   (includes integration weight).

   @param grad_u_X_grad_w_dV (nElements_global x
   nQuadraturePoints_element x nDOF_test_element x nSpace x
   nSpace). The tensor product of the solution gradient values and the
   weighted test function gradient values.

   @param weak_residual (nElements_global x nDOF_test_element). The element
   weak_residual, which is updated upon return.

   The result of calling this function is

   \f[ weak_residual_{e,i} \mapsto weak_residual_{e,i} + \int_{\Omega_e}
   \epsilon \nabla u \cdot \nabla w_i dV \quad \forall e,i \f]

   where

   \f[ \int_{\Omega_e} \epsilon \nabla u \cdot \nabla w_i dV =
   \int_{\Omega_r} \epsilon \nabla u \cdot \nabla w_i |J_e| d\hat{V}
   \approx \sum_k numDiff_k \bar{\mathbf{I}} \cdot (\nabla u \otimes
   \nabla w dV)_i \f]
*/
void updateNumericalDiffusion(proteus_index_t nElements_global,
                              int nQuadraturePoints_element,
                              int nDOF_test_element,
                              int nSpace,
                              double* numDiff,
                              double* grad_u_X_grad_w_dV,
                              double* weak_residual)
{
  proteus_index_t nSpace2=nSpace*nSpace;  proteus_index_t I; int eN, i, k;
  for(eN=0;eN<nElements_global;eN++)
    for (i=0;i<nDOF_test_element;i++)
      for (k=0;k<nQuadraturePoints_element;k++)
        for (I=0;I<nSpace;I++)
          weak_residual[eN*nDOF_test_element + i]
            +=
            numDiff[eN*nQuadraturePoints_element +
                    k]
            *
            grad_u_X_grad_w_dV[eN*nQuadraturePoints_element*nDOF_test_element*nSpace2 +
                               k*nDOF_test_element*nSpace2 +
                               i*nSpace2 +
                               I*nSpace +
                               I];
}

void updateNumericalDiffusion_lowmem(proteus_index_t nElements_global,
                                     int nQuadraturePoints_element,
                                     int nDOF_test_element,
                                     int nSpace,
                                     double* numDiff,
                                     double* grad_u,
                                     double* grad_w_dV,
                                     double* weak_residual)
{
  proteus_index_t eN, I;  int i, k;
  for(eN=0;eN<nElements_global;eN++)
    for (i=0;i<nDOF_test_element;i++)
      for (k=0;k<nQuadraturePoints_element;k++)
        for (I=0;I<nSpace;I++)
          weak_residual[eN*nDOF_test_element + i]
            +=
            numDiff[eN*nQuadraturePoints_element +
                    k]
            *
            grad_u[eN*nQuadraturePoints_element*nSpace +
                   k*nSpace +
                   I]
            *
            grad_w_dV[eN*nQuadraturePoints_element*nDOF_test_element*nSpace +
                      k*nDOF_test_element*nSpace +
                      i*nSpace +
                      I];
}

/**
   \brief Loop over all the elements and update the element Jacobian
   with the numerical quadrature approximation of the shock capturing
   integral Jacobian.

   @param nElements_global The number of elements in the mesh

   @param nQuadraturePoints_element The number of quadrature points
   per element.

   @param nDOF_test_element The number of degrees of freedom per
   element in the test space.

   @param nDOF_trial_element The number of degrees of freedom per
   element in the trial space.

   @param nSpace The number of spatial dimensions.

   @param numDiff (nElements_global x nQuadraturePoints_element). The
   shock capturing diffusion associated with each quadrature point
   (includes integration weight).

   @param grad_v_X_grad_w_dV (nElements_global x
   nQuadraturePoints_element x nDOF_trial_element x nDOF_test_element x nSpace x
   nSpace). The tensor product of the trial function gradient values
   and the test function gradient values.

   @param jacobian_weak_residual (nElements_global x nDOF_test_element x
   nDOF_trial_element). The element jacobian, which is updated upon
   return.

   The result of calling this function is

   \f[ jacobian_{e,i,j} \mapsto jacobian_{e,i,j} + \int_{\Omega_e}
   \epsilon \nabla v_j \cdot \nabla w_i dV \quad \forall e,i,j \f]

   where

   \f[ \int_{\Omega_e} \epsilon \nabla v_j \cdot \nabla w_i dV =
   \int_{\Omega_r} \epsilon \nabla v_j \cdot \nabla w_i |J_e| d\hat{V}
   \approx \sum_k numDiff_k \bar{\mathbf{I}} \cdot (\nabla v \otimes
   \nabla w)_{j,i} \f]
*/
void updateNumericalDiffusionJacobian(proteus_index_t nElements_global,
                                      int nQuadraturePoints_element,
                                      int nDOF_trial_element,
                                      int nDOF_test_element,
                                      int nSpace,
                                      double* numDiff,
                                      double* grad_v_X_grad_w_dV,
                                      double* jacobian_weak_residual)
{
  proteus_index_t nSpace2=nSpace*nSpace, nDOF_test_X_trial_element=nDOF_test_element*nDOF_trial_element;  proteus_index_t I; int eN, i, j, k;
  for(eN=0;eN<nElements_global;eN++)
    for (i=0;i<nDOF_test_element;i++)
      for (k=0;k<nQuadraturePoints_element;k++)
        for (j=0;j<nDOF_trial_element;j++)
          for (I=0;I<nSpace;I++)
            jacobian_weak_residual[eN*nDOF_test_X_trial_element +
                     i*nDOF_trial_element +
                     j]
              +=
              numDiff[eN*nQuadraturePoints_element +
                      k]
              *
              grad_v_X_grad_w_dV[eN*nQuadraturePoints_element*nDOF_test_X_trial_element*nSpace2 +
                                 k*nDOF_test_X_trial_element*nSpace2 +
                                 j*nDOF_test_element*nSpace2 +
                                 i*nSpace2 +
                                 I*nSpace +
                                 I];
}

void updateNumericalDiffusionJacobian_lowmem(proteus_index_t nElements_global,
                                             int nQuadraturePoints_element,
                                             int nDOF_trial_element,
                                             int nDOF_test_element,
                                             int nSpace,
                                             double* numDiff,
                                             double* grad_v,
                                             double* grad_w_dV,
                                             double* jacobian_weak_residual)
{
  proteus_index_t nDOF_test_X_trial_element=nDOF_test_element*nDOF_trial_element;  proteus_index_t I; int eN, i, j, k;
  for(eN=0;eN<nElements_global;eN++)
    for (i=0;i<nDOF_test_element;i++)
      for (k=0;k<nQuadraturePoints_element;k++)
        for (j=0;j<nDOF_trial_element;j++)
          for (I=0;I<nSpace;I++)
            jacobian_weak_residual[eN*nDOF_test_X_trial_element +
                     i*nDOF_trial_element +
                     j]
              +=
              numDiff[eN*nQuadraturePoints_element +
                      k]
              *
              grad_v[eN*nQuadraturePoints_element*nDOF_trial_element*nSpace +
                     k*nDOF_trial_element*nSpace +
                     j*nSpace +
                     I]
              *
              grad_w_dV[eN*nQuadraturePoints_element*nDOF_test_element*nSpace +
                        k*nDOF_test_element*nSpace +
                        i*nSpace +
                        I];
}

/**
    \brief Calculate the product of two scalars at the quadrature points
*/
void calculateScalarScalarProduct(proteus_index_t nElements_global,
                                  int nQuadraturePoints_element,
                                  double* s1,
                                  double* s2,
                                  double* sResult)
{
  proteus_index_t eN;  int k;
  for(eN=0;eN<nElements_global;eN++)
    for (k=0;k<nQuadraturePoints_element;k++)
      sResult[eN*nQuadraturePoints_element +
              k]
        =
        s1[eN*nQuadraturePoints_element +
           k]
        *
        s2[eN*nQuadraturePoints_element +
           k];
}

/**
   \brief Calculate the product of a vector and a scalar at the quadrature points.
*/
void calculateVectorScalarProduct(proteus_index_t nElements_global,
                                  int nQuadraturePoints_element,
                                  int nSpace,
                                  double* v,
                                  double* s,
                                  double* vResult)
{
  proteus_index_t eN, I;  int k;
  for(eN=0;eN<nElements_global;eN++)
    for (k=0;k<nQuadraturePoints_element;k++)
      for (I=0;I<nSpace;I++)
        vResult[eN*nQuadraturePoints_element*nSpace +
                k*nSpace +
                I]
          =
          v[eN*nQuadraturePoints_element*nSpace +
            k*nSpace +
            I]
          *
          s[eN*nQuadraturePoints_element +
            k];
}

/**
   \brief Calculate the product of a tensor and scalar at the quadrature points
*/
void calculateTensorScalarProduct(proteus_index_t nElements_global,
                                  int nQuadraturePoints_element,
                                  int nSpace,
                                  double* t,
                                  double* s,
                                  double* tResult)
{
  proteus_index_t nSpace2=nSpace*nSpace;  proteus_index_t I, J; int eN, k;
  for(eN=0;eN<nElements_global;eN++)
    for (k=0;k<nQuadraturePoints_element;k++)
      for (I=0;I<nSpace;I++)
        for (J=0;J<nSpace;J++)
          tResult[eN*nQuadraturePoints_element*nSpace2 +
                  k*nSpace2 +
                  I*nSpace +
                  J]
            =
            t[eN*nQuadraturePoints_element*nSpace2 +
              k*nSpace2 +
              I*nSpace +
              J]
            *
            s[eN*nQuadraturePoints_element +
              k];
}






/**
   \brief Update the element boundary flux on interior element boundaries
*/
void updateInteriorElementBoundaryFlux(proteus_index_t nInteriorElementBoundaries_global,
                                       int nElementBoundaries_element,
                                       int nQuadraturePoints_elementBoundary,
                                       int nDOF_test_element,
                                       proteus_index_t* interiorElementBoundaries,
                                       proteus_index_t* elementBoundaryElements,
                                       int* elementBoundaryLocalElementBoundaries,
                                       double* flux,
                                       double* w_dS,
                                       double* residual)
{
  proteus_index_t ebN, left_eN_global, right_eN_global;  int ebNI, left_ebN_element, right_ebN_element, i, k;
  for(ebNI=0;ebNI<nInteriorElementBoundaries_global;ebNI++)
    {
      ebN = interiorElementBoundaries[ebNI];
      left_eN_global = elementBoundaryElements[ebN*2+0];
      right_eN_global = elementBoundaryElements[ebN*2+1];
      left_ebN_element = elementBoundaryLocalElementBoundaries[ebN*2+0];
      right_ebN_element = elementBoundaryLocalElementBoundaries[ebN*2+1];
      for(i=0;i<nDOF_test_element;i++)
        for(k=0;k<nQuadraturePoints_elementBoundary;k++)
          {
            residual[left_eN_global*nDOF_test_element+
                     i]
              +=
              flux[ebN*nQuadraturePoints_elementBoundary+
                   k]*
              w_dS[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_test_element+
                   left_ebN_element*nQuadraturePoints_elementBoundary*nDOF_test_element+
                   k*nDOF_test_element+
                   i];
            residual[right_eN_global*nDOF_test_element+
                     i]
              -=
              flux[ebN*nQuadraturePoints_elementBoundary+
                   k]*
              w_dS[right_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_test_element+
                   right_ebN_element*nQuadraturePoints_elementBoundary*nDOF_test_element+
                   k*nDOF_test_element+
                   i];
          }
    }
}

/**
   \brief Update the element boundary flux on exterior element boundaries
*/
void updateExteriorElementBoundaryFlux(proteus_index_t nExteriorElementBoundaries_global,
                                       int nElementBoundaries_element,
                                       int nQuadraturePoints_elementBoundary,
                                       int nDOF_test_element,
                                       proteus_index_t* exteriorElementBoundaries,
                                       proteus_index_t* elementBoundaryElements,
                                       int* elementBoundaryLocalElementBoundaries,
                                       double* flux,
                                       double* w_dS,
                                       double* residual)
{
  proteus_index_t ebN, eN_global;  int ebNE, ebN_element, i, k;
  for(ebNE=0;ebNE<nExteriorElementBoundaries_global;ebNE++)
    {
      ebN = exteriorElementBoundaries[ebNE];
      eN_global = elementBoundaryElements[ebN*2+0];
      ebN_element = elementBoundaryLocalElementBoundaries[ebN*2+0];
      for(i=0;i<nDOF_test_element;i++)
        for(k=0;k<nQuadraturePoints_elementBoundary;k++)
          {
            residual[eN_global*nDOF_test_element+
                     i]
              +=
              flux[ebN*nQuadraturePoints_elementBoundary+
                   k]
              *
              w_dS[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_test_element+
                   ebN_element*nQuadraturePoints_elementBoundary*nDOF_test_element+
                   k*nDOF_test_element+
                   i];
          }
    }
}

void updateGlobalExteriorElementBoundaryFlux(proteus_index_t nExteriorElementBoundaries_global,
                                             int nQuadraturePoints_elementBoundary,
                                             int nDOF_test_element,
                                             proteus_index_t* exteriorElementBoundaries,
                                             proteus_index_t* elementBoundaryElements,
                                             int* elementBoundaryLocalElementBoundaries,
                                             double* flux,
                                             double* w_dS,
                                             double* residual)
{
  proteus_index_t ebN, eN_global;  int ebNE, ebN_element, i, k;
  for(ebNE=0;ebNE<nExteriorElementBoundaries_global;ebNE++)
    {
      ebN = exteriorElementBoundaries[ebNE];
      eN_global = elementBoundaryElements[ebN*2+0];
      ebN_element = elementBoundaryLocalElementBoundaries[ebN*2+0];
      for(i=0;i<nDOF_test_element;i++)
        for(k=0;k<nQuadraturePoints_elementBoundary;k++)
          {
            residual[eN_global*nDOF_test_element+
                     i]
              +=
              flux[ebNE*nQuadraturePoints_elementBoundary+
                   k]
              *
              w_dS[ebNE*nQuadraturePoints_elementBoundary*nDOF_test_element+
                   k*nDOF_test_element+
                   i];
          }
    }
}
void updateGlobalExteriorElementBoundaryStressFlux(proteus_index_t nExteriorElementBoundaries_global,
                                                   int nQuadraturePoints_elementBoundary,
                                                   int nDOF_test_element,
                                                   proteus_index_t* exteriorElementBoundaries,
                                                   proteus_index_t* elementBoundaryElements,
                                                   int* elementBoundaryLocalElementBoundaries,
                                                   double* stressFlux,
                                                   double* w_dS,
                                                   double* residual)
{
  proteus_index_t ebN, eN_global, I;  int ebNE, ebN_element, i, k;
  for(ebNE=0;ebNE<nExteriorElementBoundaries_global;ebNE++)
    {
      ebN = exteriorElementBoundaries[ebNE];
      eN_global = elementBoundaryElements[ebN*2+0];
      ebN_element = elementBoundaryLocalElementBoundaries[ebN*2+0];
      for(i=0;i<nDOF_test_element;i++)
        for(k=0;k<nQuadraturePoints_elementBoundary;k++)
          residual[eN_global*nDOF_test_element+
                   i]
            +=
            stressFlux[ebNE*nQuadraturePoints_elementBoundary+
                       k]
            *
            w_dS[ebNE*nQuadraturePoints_elementBoundary*nDOF_test_element+
                 k*nDOF_test_element+
                 i];
    }
}
/**
   \brief Update a two-sided (say nonconservative HJ flux) element boundary flux on interior element boundaries
*/
void updateInteriorTwoSidedElementBoundaryFlux(proteus_index_t nInteriorElementBoundaries_global,
                                               int nElementBoundaries_element,
                                               int nQuadraturePoints_elementBoundary,
                                               int nDOF_test_element,
                                               proteus_index_t* interiorElementBoundaries,
                                               proteus_index_t* elementBoundaryElements,
                                               int* elementBoundaryLocalElementBoundaries,
                                               double* flux,
                                               double* w_dS,
                                               double* residual)
{
  proteus_index_t ebN, left_eN_global, right_eN_global;  int ebNI, left_ebN_element, right_ebN_element, i, k;
  for(ebNI=0;ebNI<nInteriorElementBoundaries_global;ebNI++)
    {
      ebN = interiorElementBoundaries[ebNI];
      left_eN_global = elementBoundaryElements[ebN*2+0];
      right_eN_global = elementBoundaryElements[ebN*2+1];
      left_ebN_element = elementBoundaryLocalElementBoundaries[ebN*2+0];
      right_ebN_element = elementBoundaryLocalElementBoundaries[ebN*2+1];
      for(i=0;i<nDOF_test_element;i++)
        for(k=0;k<nQuadraturePoints_elementBoundary;k++)
          {
            residual[left_eN_global*nDOF_test_element+
                     i]
              +=
              flux[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary+
                   left_ebN_element*nQuadraturePoints_elementBoundary+ /*left flux*/
                   k]*
              w_dS[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_test_element+
                   left_ebN_element*nQuadraturePoints_elementBoundary*nDOF_test_element+
                   k*nDOF_test_element+
                   i];
            residual[right_eN_global*nDOF_test_element+
                     i]
              +=
              flux[right_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary+
                   right_ebN_element*nQuadraturePoints_elementBoundary+ /*right flux*/
                   k]*
              w_dS[right_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_test_element+
                   right_ebN_element*nQuadraturePoints_elementBoundary*nDOF_test_element+
                   k*nDOF_test_element+
                   i];
          }
    }
}

void calculateExteriorElementBoundaryStress3D(proteus_index_t nExteriorElementBoundaries_global,
                                              int nQuadraturePoints_elementBoundary,
                                              int* elementBoundaryMaterialTypes,
                                              proteus_index_t* exteriorElementBoundaries,
                                              proteus_index_t* elementBoundaryElements,
                                              int* elementBoundaryLocalElementBoundaries,
                                              double* p,
                                              double* mom_flux_vec_u,
                                              double* mom_flux_vec_v,
                                              double* mom_flux_vec_w,
                                              double* dS,
                                              double* n,
                                              double* F)
{
  proteus_index_t ebN, eN_global, I;  int ebNE, ebN_element, k;
  double stress_x,stress_y,stress_z;
  for(ebNE=0;ebNE<nExteriorElementBoundaries_global;ebNE++)
    {
      ebN = exteriorElementBoundaries[ebNE];
      eN_global = elementBoundaryElements[ebN*2+0];
      ebN_element = elementBoundaryLocalElementBoundaries[ebN*2+0];
      for(k=0;k<nQuadraturePoints_elementBoundary;k++)
        {
          stress_x = 0.0;
          stress_y = 0.0;
          stress_z = 0.0;
          stress_x += p[ebNE*nQuadraturePoints_elementBoundary+
                        k]
            *
            n[ebNE*nQuadraturePoints_elementBoundary*3+
              k*3+
              0];
          stress_y += p[ebNE*nQuadraturePoints_elementBoundary+
                        k]
            *
            n[ebNE*nQuadraturePoints_elementBoundary*3+
              k*3+
              1];
          stress_z += p[ebNE*nQuadraturePoints_elementBoundary+
                        k]
            *
            n[ebNE*nQuadraturePoints_elementBoundary*3+
              k*3+
              2];
          for(I=0;I<3;I++)
            {
              stress_x += mom_flux_vec_u[ebNE*nQuadraturePoints_elementBoundary*3+
                                         k*3+
                                         I]
                *
                n[ebNE*nQuadraturePoints_elementBoundary*3+
                  k*3+
                  I];
              stress_y += mom_flux_vec_v[ebNE*nQuadraturePoints_elementBoundary*3+
                                         k*3+
                                         I]
                *
                n[ebNE*nQuadraturePoints_elementBoundary*3+
                  k*3+
                  I];
              stress_z += mom_flux_vec_w[ebNE*nQuadraturePoints_elementBoundary*3+
                                         k*3+
                                         I]
                *
                n[ebNE*nQuadraturePoints_elementBoundary*3+
                  k*3+
                  I];
            }
          F[elementBoundaryMaterialTypes[ebN]*3 + 0] += stress_x*
            dS[ebNE*nQuadraturePoints_elementBoundary+
               k];
          F[elementBoundaryMaterialTypes[ebN]*3 + 1] += stress_y*
            dS[ebNE*nQuadraturePoints_elementBoundary+
               k];
          F[elementBoundaryMaterialTypes[ebN]*3 + 2] += stress_z*
            dS[ebNE*nQuadraturePoints_elementBoundary+
               k];
        }
    }
}

void calculateExteriorElementBoundaryStress2D(proteus_index_t nExteriorElementBoundaries_global,
                                              int nQuadraturePoints_elementBoundary,
                                              int* elementBoundaryMaterialTypes,
                                              proteus_index_t* exteriorElementBoundaries,
                                              proteus_index_t* elementBoundaryElements,
                                              int* elementBoundaryLocalElementBoundaries,
                                              double* p,
                                              double* mom_flux_vec_u,
                                              double* mom_flux_vec_v,
                                              double* dS,
                                              double* n,
                                              double* F)
{
  proteus_index_t ebN, eN_global, I;  int ebNE, ebN_element, k;
  double stress_x,stress_y;
  for(ebNE=0;ebNE<nExteriorElementBoundaries_global;ebNE++)
    {
      ebN = exteriorElementBoundaries[ebNE];
      eN_global = elementBoundaryElements[ebN*2+0];
      ebN_element = elementBoundaryLocalElementBoundaries[ebN*2+0];
      for(k=0;k<nQuadraturePoints_elementBoundary;k++)
        {
          stress_x = 0.0;
          stress_y = 0.0;
          stress_x += p[ebNE*nQuadraturePoints_elementBoundary+
                        k]
            *
            n[ebNE*nQuadraturePoints_elementBoundary*2+
              k*2+
              0];
          stress_y += p[ebNE*nQuadraturePoints_elementBoundary+
                        k]
            *
            n[ebNE*nQuadraturePoints_elementBoundary*2+
              k*2+
              1];
          /* for(I=0;I<2;I++) */
          /*   { */
          /*     stress_x += mom_flux_vec_u[ebNE*nQuadraturePoints_elementBoundary*2+ */
          /*                                k*2+ */
          /*                                I] */
          /*       * */
          /*       n[ebNE*nQuadraturePoints_elementBoundary*2+ */
          /*         k*2+ */
          /*         I]; */
          /*     stress_y += mom_flux_vec_v[ebNE*nQuadraturePoints_elementBoundary*2+ */
          /*                                k*2+ */
          /*                                I] */
          /*       * */
          /*       n[ebNE*nQuadraturePoints_elementBoundary*2+ */
          /*         k*2+ */
          /*         I]; */
          /*   } */
          F[elementBoundaryMaterialTypes[ebN]*2 + 0] += stress_x*
            dS[ebNE*nQuadraturePoints_elementBoundary+
               k];
          F[elementBoundaryMaterialTypes[ebN]*2 + 1] += stress_y*
            dS[ebNE*nQuadraturePoints_elementBoundary+
               k];
        }
    }
}
/**
   compute average pressure over different exterior element boundaries
   requires pressure and boundary measure to be zeroed outside call
   if that is desired
 */
void accumulateExteriorElementPressureIntegrals(proteus_index_t nExteriorElementBoundaries_global,
                                                int nQuadraturePoints_elementBoundary,
                                                int* elementBoundaryMaterialTypes,
                                                proteus_index_t* exteriorElementBoundaries,
                                                double* p,
                                                double* dS,
                                                double* P,
                                                double* boundaryMeasure)
{
  proteus_index_t ebN;  int ebNE, k, elementBoundaryFlag;
  /*loop through exterior element boundaries*/
  for(ebNE=0;ebNE<nExteriorElementBoundaries_global;ebNE++)
    {
      /*get the global element boundary number for the current exterior element boundary*/
      ebN = exteriorElementBoundaries[ebNE];
      /*integer flag for this boundary*/
      elementBoundaryFlag = elementBoundaryMaterialTypes[ebN];
      /*loop through element boundary quadrature points and accumulate pressure and boundary area/length */
      for(k=0;k<nQuadraturePoints_elementBoundary;k++)
        {
          P[elementBoundaryFlag] += p[ebNE*nQuadraturePoints_elementBoundary+k]
            *
            dS[ebNE*nQuadraturePoints_elementBoundary+k];
          boundaryMeasure[elementBoundaryFlag] +=
            dS[ebNE*nQuadraturePoints_elementBoundary+k];
        }
    }
}


/**
   \brief Update the global residuals from the element residuals.
*/
void updateGlobalResidualFromElementResidual(proteus_index_t nElements_global,
                                             int nDOF_test_element,
                                             proteus_index_t offset_r,
                                             proteus_index_t stride_r,
                                             int* nFreeDOF_element_r,
                                             int* freeLocal_r,
                                             proteus_index_t* freeGlobal_r,
                                             double* elementResidual,
                                             double* globalResidual)
{
  proteus_index_t eN;  int ii;
  for (eN=0;eN<nElements_global;eN++)
    for (ii=0;ii<nFreeDOF_element_r[eN];ii++)
      globalResidual[offset_r +
                     stride_r*freeGlobal_r[eN*nDOF_test_element+
                                         ii]]
        +=
        elementResidual[eN*nDOF_test_element +
                        freeLocal_r[eN*nDOF_test_element+
                                    ii]];
}

/**
   \brief  Update the global dense jacobian  from the element Jacobians.
*/
void updateGlobalJacobianFromElementJacobian_dense(proteus_index_t nElements_global,
                                                   int nDOF_test_element,
                                                   int nDOF_trial_element,
                                                   proteus_index_t offset_r,
                                                   proteus_index_t stride_r,
                                                   proteus_index_t offset_u,
                                                   proteus_index_t stride_u,
                                                   proteus_index_t nFreeVDOF_global,
                                                   int* nFreeDOF_element_r,
                                                   int* freeLocal_r,
                                                   proteus_index_t* freeGlobal_r,
                                                   int* nFreeDOF_element_u,
                                                   int* freeLocal_u,
                                                   proteus_index_t* freeGlobal_u,
                                                   double* elementJacobian,
                                                   double* globalJacobian)
{
  proteus_index_t nDOF_test_X_trial_element=nDOF_test_element*nDOF_trial_element;  proteus_index_t jacIndex, I, J; int eN, ii, jj, i, j;
  for (eN=0;eN<nElements_global;eN++)
    for (ii=0;ii<nFreeDOF_element_r[eN];ii++)
      {
        i = freeLocal_r[eN*nDOF_test_element+
                        ii];
        I = offset_r + stride_r*freeGlobal_r[eN*nDOF_test_element+
                                             ii];
        for (jj=0;jj<nFreeDOF_element_u[eN];jj++)
          {
            j = freeLocal_u[eN*nDOF_trial_element+
                            jj];
            J = offset_u + stride_u*freeGlobal_u[eN*nDOF_trial_element+
                                                 jj];
            jacIndex = I + J*nFreeVDOF_global;
            globalJacobian[jacIndex]
              +=
              elementJacobian[eN*nDOF_test_X_trial_element +
                              i*nDOF_trial_element+
                              j];
          }
      }
}

/**
   \brief  Update the global dense jacobian  from the element Jacobians.
*/
void updateGlobalJacobianFromElementJacobian_eb_dense(proteus_index_t* elementNeighbors,
                                                      proteus_index_t nElements_global,
                                                      int nElementBoundaries_element,
                                                      int nDOF_test_element,
                                                      int nDOF_trial_element,
                                                      proteus_index_t offset_r,
                                                      proteus_index_t stride_r,
                                                      proteus_index_t offset_u,
                                                      proteus_index_t stride_u,
                                                      proteus_index_t nFreeVDOF_global,
                                                      int* nFreeDOF_element_r,
                                                      int* freeLocal_r,
                                                      proteus_index_t* freeGlobal_r,
                                                      int* nFreeDOF_element_u,
                                                      int* freeLocal_u,
                                                      proteus_index_t* freeGlobal_u,
                                                      double* elementJacobian_eb,
                                                      double* globalJacobian)
{
  proteus_index_t nDOF_test_X_trial_element=nDOF_test_element*nDOF_trial_element;  proteus_index_t ebN, eN_ebN, jacIndex, I, J; int eN, ii, jj, i, j;
  for (eN=0;eN<nElements_global;eN++)
    for (ebN=0;ebN<nElementBoundaries_element;ebN++)
      {
/*         /\*cek 1D debugging *\/ */
/*         if(ebN == 0) */
/*           eN_ebN = eN+1; */
/*         if(ebN == 1) */
/*           eN_ebN = eN-1; */
/*         if(eN_ebN >= 0 && eN_ebN < nElements_global) */
        eN_ebN = elementNeighbors[eN*nElementBoundaries_element+ebN];
        if (eN_ebN >= 0)
          for (ii=0;ii<nFreeDOF_element_r[eN];ii++)
            {
              i = freeLocal_r[eN*nDOF_test_element+
                              ii];
              I = offset_r + stride_r*freeGlobal_r[eN*nDOF_test_element+
                                                   ii];
              for (jj=0;jj<nFreeDOF_element_u[eN_ebN];jj++)
                {
                  j = freeLocal_u[eN_ebN*nDOF_trial_element+
                                  jj];
                  J = offset_u + stride_u*freeGlobal_u[eN_ebN*nDOF_trial_element+
                                                       jj];
                  jacIndex = I + J*nFreeVDOF_global;
                  globalJacobian[jacIndex]
                    +=
                    elementJacobian_eb[eN*nElementBoundaries_element*nDOF_test_X_trial_element +
                                       ebN*nDOF_test_X_trial_element+
                                       i*nDOF_trial_element+
                                       j];
                }
            }
      }
}

/**
   \brief Update the global dense Jacobian from the element boundary flux Jacobians at interior boundaries
*/
void updateGlobalJacobianFromInteriorElementBoundaryFluxJacobian_dense(proteus_index_t nInteriorElementBoundaries_global,
                                                                       int nElementBoundaries_element,
                                                                       int nQuadraturePoints_elementBoundary,
                                                                       int nDOF_test_element,
                                                                       int nDOF_trial_element,
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
                                                                       double* elementBoundaryFluxJacobian,
                                                                       double* w_dS,
                                                                       double* jac)
{
  proteus_index_t ebN, left_eN_global, right_eN_global, jacIndex, I, J;  int ebNI, left_ebN_element, right_ebN_element, ii, i, k, jj, j;
  for (ebNI=0;ebNI<nInteriorElementBoundaries_global;ebNI++)
    {
      ebN = interiorElementBoundaries[ebNI];
      left_eN_global   = elementBoundaryElements[ebN*2+0];
      right_eN_global  = elementBoundaryElements[ebN*2+1];
      left_ebN_element  = elementBoundaryLocalElementBoundaries[ebN*2+0];
      right_ebN_element = elementBoundaryLocalElementBoundaries[ebN*2+1];
      for(ii=0;ii<nFreeDOF_element_r[left_eN_global];ii++)
        {
          i = freeLocal_r[left_eN_global*nDOF_test_element+
                        ii];
          I = offset_r + stride_r*freeGlobal_r[left_eN_global*nDOF_test_element+
                                             ii];
          for(k=0;k<nQuadraturePoints_elementBoundary;k++)
            {
              for(jj=0;jj<nFreeDOF_element_u[left_eN_global];jj++)
                {
                  j = freeLocal_u[left_eN_global*nDOF_trial_element+
                                jj];
                  J = offset_u + stride_u*freeGlobal_u[left_eN_global*nDOF_trial_element+
                                                     jj];
                  jacIndex = I+J*nFreeVDOF_global;
                  jac[jacIndex]
                    +=
                    elementBoundaryFluxJacobian[ebN*2*nQuadraturePoints_elementBoundary*nDOF_trial_element +
                                                0*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                                                k*nDOF_trial_element+
                                                j]
                    *
                    w_dS[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_test_element +
                         left_ebN_element*nQuadraturePoints_elementBoundary*nDOF_test_element+
                         k*nDOF_test_element+
                         i];
                }
              for(jj=0;jj<nFreeDOF_element_u[right_eN_global];jj++)
                {
                  j = freeLocal_u[right_eN_global*nDOF_trial_element+
                                  jj];
                  J = offset_u + stride_u*freeGlobal_u[right_eN_global*nDOF_trial_element+
                                                       jj];
                  jacIndex = I+J*nFreeVDOF_global;
                  jac[jacIndex]
                    +=
                    elementBoundaryFluxJacobian[ebN*2*nQuadraturePoints_elementBoundary*nDOF_trial_element +
                                                1*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                                                k*nDOF_trial_element+
                                                j]*
                    w_dS[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_test_element +
                         left_ebN_element*nQuadraturePoints_elementBoundary*nDOF_test_element+
                         k*nDOF_test_element+i];
                }
            }
        }
      for(ii=0;ii<nFreeDOF_element_r[right_eN_global];ii++)
        {
          i = freeLocal_r[right_eN_global*nDOF_test_element+
                        ii];
          I = offset_r + stride_r*freeGlobal_r[right_eN_global*nDOF_test_element+
                                             ii];
          for(k=0;k<nQuadraturePoints_elementBoundary;k++)
            {
              for(jj=0;jj<nFreeDOF_element_u[left_eN_global];jj++)
                {
                  j = freeLocal_u[left_eN_global*nDOF_trial_element+
                                jj];
                  J = offset_u + stride_u*freeGlobal_u[left_eN_global*nDOF_trial_element+
                                                     jj];
                  jacIndex = I+J*nFreeVDOF_global;
                  jac[jacIndex]
                    -=
                    elementBoundaryFluxJacobian[ebN*2*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                                                0*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                                                k*nDOF_trial_element+
                                                j]
                    *
                    w_dS[right_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_test_element+
                         right_ebN_element*nQuadraturePoints_elementBoundary*nDOF_test_element+
                         k*nDOF_test_element+
                         i];
                }
              for(jj=0;jj<nFreeDOF_element_u[right_eN_global];jj++)
                {
                  j = freeLocal_u[right_eN_global*nDOF_trial_element+
                                jj];
                  J = offset_u + stride_u*freeGlobal_u[right_eN_global*nDOF_trial_element+
                                                     jj];
                  jacIndex = I+J*nFreeVDOF_global;
                  jac[jacIndex]
                    -=
                    elementBoundaryFluxJacobian[ebN*2*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                                                1*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                                                k*nDOF_trial_element+
                                                j]*
                    w_dS[right_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_test_element+
                         right_ebN_element*nQuadraturePoints_elementBoundary*nDOF_test_element+
                         k*nDOF_test_element+
                         i];
                }
            }
        }
    }
}

/**
   \brief Update the global dense Jacobian from the element boundary flux Jacobians at interior boundaries
*/
void updateGlobalJacobianFromInteriorElementBoundaryFluxJacobian_eb_dense(proteus_index_t* elementNeighbors,
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
                                                                          proteus_index_t* interiorElementBoundaries,
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
{
  proteus_index_t ebN, left_eN_ebN, right_eN_ebN, left_eN_global, right_eN_global, jacIndex, I, J;  int ebNI, ebN_element, left_ebN_element, right_ebN_element, ii, i, k, jj, j;
  for (ebNI=0;ebNI<nInteriorElementBoundaries_global;ebNI++)
    {
      ebN = interiorElementBoundaries[ebNI];
      left_eN_global   = elementBoundaryElements[ebN*2+0];
      right_eN_global  = elementBoundaryElements[ebN*2+1];
      left_ebN_element  = elementBoundaryLocalElementBoundaries[ebN*2+0];
      right_ebN_element = elementBoundaryLocalElementBoundaries[ebN*2+1];
      for(ebN_element=0;ebN_element<nElementBoundaries_element;ebN_element++)
        {
/*           /\* cek debugging 1D *\/ */
/*           if(ebN_element == 0) */
/*             left_eN_ebN = left_eN_global+1; */
/*           if(ebN_element == 1) */
/*             left_eN_ebN = left_eN_global-1; */
/*           if(ebN_element == 0) */
/*             right_eN_ebN = right_eN_global+1; */
/*           if(ebN_element == 1) */
/*             right_eN_ebN = right_eN_global-1; */
          left_eN_ebN = elementNeighbors[left_eN_global*nElementBoundaries_element+ebN_element];
          right_eN_ebN = elementNeighbors[right_eN_global*nElementBoundaries_element+ebN_element];
          for(ii=0;ii<nFreeDOF_element_r[left_eN_global];ii++)
            {
              i = freeLocal_r[left_eN_global*nDOF_test_element+
                              ii];
              I = offset_r + stride_r*freeGlobal_r[left_eN_global*nDOF_test_element+
                                                   ii];
              for(k=0;k<nQuadraturePoints_elementBoundary;k++)
                {
/*                   if(left_eN_ebN >= 0 && left_eN_ebN < nElements_global) */
                  if(left_eN_ebN >= 0)
                    for(jj=0;jj<nFreeDOF_element_u[left_eN_ebN];jj++)
                      {
                        j = freeLocal_u[left_eN_ebN*nDOF_trial_element+
                                        jj];
                        J = offset_u + stride_u*freeGlobal_u[left_eN_ebN*nDOF_trial_element+
                                                             jj];
                        jacIndex = I+J*nFreeVDOF_global;
                        jac[jacIndex]
                          +=
                          elementBoundaryFluxJacobian_eb[ebN*2*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_trial_element +
                                                         0*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                                                         ebN_element*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                                                         k*nDOF_trial_element+
                                                         j]
                          *
                          w_dS[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_test_element +
                               left_ebN_element*nQuadraturePoints_elementBoundary*nDOF_test_element+
                               k*nDOF_test_element+
                               i];
/*                         printf("ind=%i,ebN = %i,left_eN_global=%i,left_ebN_element=%i,ebN_element=%i,i=%i,j=%i,jac=%12.5e,fjac=%12.5e,w=%12.5e\n", */
/*                                jacIndex,ebN,left_eN_global,left_ebN_element,ebN_element,i,j,jac[jacIndex], */
/*                                elementBoundaryFluxJacobian_eb[ebN*2*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_trial_element +  */
/*                                                               0*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_trial_element+ */
/*                                                               ebN_element*nQuadraturePoints_elementBoundary*nDOF_trial_element+ */
/*                                                               k*nDOF_trial_element+ */
/*                                                               j], */
/*                                w_dS[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_test_element +  */
/*                                     left_ebN_element*nQuadraturePoints_elementBoundary*nDOF_test_element+ */
/*                                     k*nDOF_test_element+ */
/*                                     i]); */
                      }
/*                   if(right_eN_ebN >= 0 && right_eN_ebN < nElements_global) */
                  if(right_eN_ebN >= 0)
                    for(jj=0;jj<nFreeDOF_element_u[right_eN_ebN];jj++)
                      {
                        j = freeLocal_u[right_eN_ebN*nDOF_trial_element+
                                        jj];
                        J = offset_u + stride_u*freeGlobal_u[right_eN_ebN*nDOF_trial_element+
                                                             jj];
                        jacIndex = I+J*nFreeVDOF_global;
                        jac[jacIndex]
                          +=
                          elementBoundaryFluxJacobian_eb[ebN*2*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_trial_element +
                                                         1*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                                                         ebN_element*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                                                         k*nDOF_trial_element+
                                                         j]*
                          w_dS[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_test_element +
                               left_ebN_element*nQuadraturePoints_elementBoundary*nDOF_test_element+
                               k*nDOF_test_element+i];
/*                         printf("ind=%i,ebN = %i,left_eN_global=%i,left_ebN_element=%i,ebN_element=%i,i=%i,j=%i,jac=%12.5e,fjac=%12.5e,w=%12.5e\n", */
/*                                jacIndex,ebN,left_eN_global,left_ebN_element,ebN_element,i,j,jac[jacIndex], */
/*                                elementBoundaryFluxJacobian_eb[ebN*2*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_trial_element +  */
/*                                                               1*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_trial_element+ */
/*                                                               ebN_element*nQuadraturePoints_elementBoundary*nDOF_trial_element+ */
/*                                                               k*nDOF_trial_element+ */
/*                                                               j], */
/*                                w_dS[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_test_element +  */
/*                                     left_ebN_element*nQuadraturePoints_elementBoundary*nDOF_test_element+ */
/*                                     k*nDOF_test_element+ */
/*                                     i]); */
                      }
                }
            }
          for(ii=0;ii<nFreeDOF_element_r[right_eN_global];ii++)
            {
              i = freeLocal_r[right_eN_global*nDOF_test_element+
                              ii];
              I = offset_r + stride_r*freeGlobal_r[right_eN_global*nDOF_test_element+
                                                   ii];
              for(k=0;k<nQuadraturePoints_elementBoundary;k++)
                {
/*                   if(left_eN_ebN >= 0 && left_eN_ebN < nElements_global) */
                  if(left_eN_ebN >= 0)
                    for(jj=0;jj<nFreeDOF_element_u[left_eN_ebN];jj++)
                      {
                        j = freeLocal_u[left_eN_ebN*nDOF_trial_element+
                                        jj];
                        J = offset_u + stride_u*freeGlobal_u[left_eN_ebN*nDOF_trial_element+
                                                             jj];
                        jacIndex = I+J*nFreeVDOF_global;
                        jac[jacIndex]
                          -=
                          elementBoundaryFluxJacobian_eb[ebN*2*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                                                         0*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                                                         ebN_element*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                                                         k*nDOF_trial_element+
                                                         j]
                          *
                          w_dS[right_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_test_element+
                               right_ebN_element*nQuadraturePoints_elementBoundary*nDOF_test_element+
                               k*nDOF_test_element+
                               i];
/*                         printf("ind=%i,ebN = %i,left_eN_global=%i,left_ebN_element=%i,ebN_element=%i,i=%i,j=%i,jac=%12.5e,fjac=%12.5e,w=%12.5e\n", */
/*                                jacIndex,ebN,right_eN_global,right_ebN_element,ebN_element,i,j,jac[jacIndex], */
/*                                elementBoundaryFluxJacobian_eb[ebN*2*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_trial_element +  */
/*                                                               0*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_trial_element+ */
/*                                                               ebN_element*nQuadraturePoints_elementBoundary*nDOF_trial_element+ */
/*                                                               k*nDOF_trial_element+ */
/*                                                               j], */
/*                                w_dS[right_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_test_element +  */
/*                                     right_ebN_element*nQuadraturePoints_elementBoundary*nDOF_test_element+ */
/*                                     k*nDOF_test_element+ */
/*                                     i]); */
                      }
/*                   if(right_eN_ebN >= 0 && right_eN_ebN < nElements_global) */
                  if(right_eN_ebN >= 0)
                    for(jj=0;jj<nFreeDOF_element_u[right_eN_ebN];jj++)
                      {
                        j = freeLocal_u[right_eN_ebN*nDOF_trial_element+
                                        jj];
                        J = offset_u + stride_u*freeGlobal_u[right_eN_ebN*nDOF_trial_element+
                                                             jj];
                        jacIndex = I+J*nFreeVDOF_global;
                        jac[jacIndex]
                          -=
                          elementBoundaryFluxJacobian_eb[ebN*2*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                                                         1*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                                                         ebN_element*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                                                         k*nDOF_trial_element+
                                                         j]*
                          w_dS[right_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_test_element+
                               right_ebN_element*nQuadraturePoints_elementBoundary*nDOF_test_element+
                               k*nDOF_test_element+
                               i];
/*                         printf("ind=%i,ebN = %i,left_eN_global=%i,left_ebN_element=%i,ebN_element=%i,i=%i,j=%i,jac=%12.5e,fjac=%12.5e,w=%12.5e\n", */
/*                                jacIndex,ebN,right_eN_global,right_ebN_element,ebN_element,i,j,jac[jacIndex], */
/*                                elementBoundaryFluxJacobian_eb[ebN*2*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_trial_element +  */
/*                                                               1*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_trial_element+ */
/*                                                               ebN_element*nQuadraturePoints_elementBoundary*nDOF_trial_element+ */
/*                                                               k*nDOF_trial_element+ */
/*                                                               j], */
/*                                w_dS[right_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_test_element +  */
/*                                     right_ebN_element*nQuadraturePoints_elementBoundary*nDOF_test_element+ */
/*                                     k*nDOF_test_element+ */
/*                                     i]); */
                      }
                }
            }
        }
    }
}
/**
   \brief Update the global dense Jacobian from the element boundary two-sided Hamiltonflux Jacobians at interior boundaries
*/
void updateGlobalJacobianFromInteriorElementBoundaryFluxJacobian_2sided_dense(proteus_index_t nInteriorElementBoundaries_global,
                                                                          int nElementBoundaries_element,
                                                                          int nQuadraturePoints_elementBoundary,
                                                                          int nDOF_test_element,
                                                                          int nDOF_trial_element,
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
                                                                          double* elementBoundaryFluxJacobian_2sided,
                                                                          double* w_dS,
                                                                          double* jac)
{
  proteus_index_t ebN, left_eN_global, right_eN_global, jacIndex, I, J;  int ebNI, left_ebN_element, right_ebN_element, ii, i, k, jj, j;
  for (ebNI=0;ebNI<nInteriorElementBoundaries_global;ebNI++)
    {
      ebN = interiorElementBoundaries[ebNI];
      left_eN_global   = elementBoundaryElements[ebN*2+0];
      right_eN_global  = elementBoundaryElements[ebN*2+1];
      left_ebN_element  = elementBoundaryLocalElementBoundaries[ebN*2+0];
      right_ebN_element = elementBoundaryLocalElementBoundaries[ebN*2+1];
      /*left  flux*/
      for(ii=0;ii<nFreeDOF_element_r[left_eN_global];ii++)
        {
          i = freeLocal_r[left_eN_global*nDOF_test_element+
                          ii];
          I = offset_r + stride_r*freeGlobal_r[left_eN_global*nDOF_test_element+
                                               ii];
          for(k=0;k<nQuadraturePoints_elementBoundary;k++)
            {
              for(jj=0;jj<nFreeDOF_element_u[left_eN_global];jj++)
                {
                  j = freeLocal_u[left_eN_global*nDOF_trial_element+
                                  jj];
                  J = offset_u + stride_u*freeGlobal_u[left_eN_global*nDOF_trial_element+
                                                       jj];
                  jacIndex = I+J*nFreeVDOF_global;
                  jac[jacIndex]
                    +=
                    elementBoundaryFluxJacobian_2sided[ebN*2*2*nQuadraturePoints_elementBoundary*nDOF_trial_element +
                                                       0*2*nQuadraturePoints_elementBoundary*nDOF_trial_element + /*left flux*/
                                                       0*nQuadraturePoints_elementBoundary*nDOF_trial_element+ /*left  neig. dep*/
                                                       k*nDOF_trial_element+
                                                       j]
                    *
                    w_dS[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_test_element +
                         left_ebN_element*nQuadraturePoints_elementBoundary*nDOF_test_element+
                         k*nDOF_test_element+
                         i];
                }
              for(jj=0;jj<nFreeDOF_element_u[right_eN_global];jj++)
                {
                  j = freeLocal_u[right_eN_global*nDOF_trial_element+
                                  jj];
                  J = offset_u + stride_u*freeGlobal_u[right_eN_global*nDOF_trial_element+
                                                       jj];
                  jacIndex = I+J*nFreeVDOF_global;
                  jac[jacIndex]
                    +=
                    elementBoundaryFluxJacobian_2sided[ebN*2*2*nQuadraturePoints_elementBoundary*nDOF_trial_element +
                                                       0*2*nQuadraturePoints_elementBoundary*nDOF_trial_element + /*left flux*/
                                                       1*nQuadraturePoints_elementBoundary*nDOF_trial_element+ /*right  neig. dep*/
                                                       k*nDOF_trial_element+
                                                       j]*
                    w_dS[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_test_element +
                         left_ebN_element*nQuadraturePoints_elementBoundary*nDOF_test_element+
                         k*nDOF_test_element+i];
                }
            }/*k*/
        }/*ii*/
      /*right flux*/
      for(ii=0;ii<nFreeDOF_element_r[right_eN_global];ii++)
        {
          i = freeLocal_r[right_eN_global*nDOF_test_element+
                        ii];
          I = offset_r + stride_r*freeGlobal_r[right_eN_global*nDOF_test_element+
                                             ii];
          for(k=0;k<nQuadraturePoints_elementBoundary;k++)
            {
              for(jj=0;jj<nFreeDOF_element_u[left_eN_global];jj++)
                {
                  j = freeLocal_u[left_eN_global*nDOF_trial_element+
                                jj];
                  J = offset_u + stride_u*freeGlobal_u[left_eN_global*nDOF_trial_element+
                                                     jj];
                  jacIndex = I+J*nFreeVDOF_global;
                  jac[jacIndex]
                    +=
                    elementBoundaryFluxJacobian_2sided[ebN*2*2*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                                                       1*2*nQuadraturePoints_elementBoundary*nDOF_trial_element + /*right flux*/
                                                       0*nQuadraturePoints_elementBoundary*nDOF_trial_element+ /*left neighbor*/
                                                       k*nDOF_trial_element+
                                                       j]
                    *
                    w_dS[right_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_test_element+
                         right_ebN_element*nQuadraturePoints_elementBoundary*nDOF_test_element+
                         k*nDOF_test_element+
                         i];
                }
              for(jj=0;jj<nFreeDOF_element_u[right_eN_global];jj++)
                {
                  j = freeLocal_u[right_eN_global*nDOF_trial_element+
                                jj];
                  J = offset_u + stride_u*freeGlobal_u[right_eN_global*nDOF_trial_element+
                                                     jj];
                  jacIndex = I+J*nFreeVDOF_global;
                  jac[jacIndex]
                    +=
                    elementBoundaryFluxJacobian_2sided[ebN*2*2*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                                                       1*2*nQuadraturePoints_elementBoundary*nDOF_trial_element + /*right flux*/
                                                       1*nQuadraturePoints_elementBoundary*nDOF_trial_element+ /*right neig dep*/
                                                       k*nDOF_trial_element+
                                                       j]*
                    w_dS[right_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_test_element+
                         right_ebN_element*nQuadraturePoints_elementBoundary*nDOF_test_element+
                         k*nDOF_test_element+
                         i];
                }
            }
        }
    }
}

/**
   \brief Update the global dense Jacobian from the element boundary flux Jacobians at exterior boundaries
*/
void updateGlobalJacobianFromExteriorElementBoundaryFluxJacobian_dense(proteus_index_t nExteriorElementBoundaries_global,
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
                                                                       double* elementBoundaryFluxJacobian,
                                                                       double* w_dS,
                                                                       double* jac)
{
  proteus_index_t ebN, eN_global, I, J, jacIndex;  int ebNE, ebN_element, ii, i, k, jj, j;
  for (ebNE=0;ebNE<nExteriorElementBoundaries_global;ebNE++)
    {
      ebN = exteriorElementBoundaries[ebNE];
      eN_global   = elementBoundaryElements[ebN*2+0];
      ebN_element  = elementBoundaryLocalElementBoundaries[ebN*2+0];
      for(ii=0;ii<nFreeDOF_element_r[eN_global];ii++)
        {
          i = freeLocal_r[eN_global*nDOF_test_element+
                          ii];
          I = offset_r + stride_r*freeGlobal_r[eN_global*nDOF_test_element+
                                               ii];
          for(k=0;k<nQuadraturePoints_elementBoundary;k++)
            {
              for(jj=0;jj<nFreeDOF_element_u[eN_global];jj++)
                {
                  j = freeLocal_u[eN_global*nDOF_trial_element+
                                  jj];
                  J = offset_u + stride_u*freeGlobal_u[eN_global*nDOF_trial_element+
                                                       jj];
                  jacIndex = I+J*nFreeVDOF_global;
                  jac[jacIndex]
                    +=
                    elementBoundaryFluxJacobian[ebN*2*nQuadraturePoints_elementBoundary*nDOF_trial_element +
                                                0*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                                                k*nDOF_trial_element+
                                                j]
                    *
                    w_dS[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_test_element +
                         ebN_element*nQuadraturePoints_elementBoundary*nDOF_test_element+
                         k*nDOF_test_element+
                         i];
                }
            }
        }
    }
}

/**
   \brief Update the global dense Jacobian from the element boundary flux Jacobians at exterior boundaries
*/
void updateGlobalJacobianFromExteriorElementBoundaryFluxJacobian_eb_dense(proteus_index_t* elementNeighbors,
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
{
  proteus_index_t ebN, eN_global, eN_ebN, I, J, jacIndex;  int ebNE, ebN_element, ebN_eN, ii, i, k, jj, j;
  for (ebNE=0;ebNE<nExteriorElementBoundaries_global;ebNE++)
    {
      ebN = exteriorElementBoundaries[ebNE];
      eN_global   = elementBoundaryElements[ebN*2+0];
      ebN_element  = elementBoundaryLocalElementBoundaries[ebN*2+0];
      for (ebN_eN=0;ebN_eN<nElementBoundaries_element;ebN_eN++)
        {
/*           /\* cek debugging 1D *\/ */
/*           if(ebN_eN == 0) */
/*             eN_ebN = eN_global+1; */
/*           if(ebN_eN == 1) */
/*             eN_ebN = eN_global-1; */
          eN_ebN = elementNeighbors[eN_global*nElementBoundaries_element+ebN_eN];
          for(ii=0;ii<nFreeDOF_element_r[eN_global];ii++)
            {
              i = freeLocal_r[eN_global*nDOF_test_element+
                              ii];
              I = offset_r + stride_r*freeGlobal_r[eN_global*nDOF_test_element+
                                                   ii];
              for(k=0;k<nQuadraturePoints_elementBoundary;k++)
                {
/*                   if(eN_ebN >= 0 && eN_ebN < nElements_global) */
                  if(eN_ebN >= 0)
                    for(jj=0;jj<nFreeDOF_element_u[eN_ebN];jj++)
                      {
                        j = freeLocal_u[eN_ebN*nDOF_trial_element+
                                        jj];
                        J = offset_u + stride_u*freeGlobal_u[eN_ebN*nDOF_trial_element+
                                                             jj];
                        jacIndex = I+J*nFreeVDOF_global;
                        jac[jacIndex]
                          +=
                          elementBoundaryFluxJacobian_eb[ebN*2*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_trial_element +
                                                         0*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                                                         ebN_eN*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                                                         k*nDOF_trial_element+
                                                         j]
                          *
                          w_dS[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_test_element +
                               ebN_element*nQuadraturePoints_elementBoundary*nDOF_test_element+
                               k*nDOF_test_element+
                               i];
/*                         printf("ind=%i,ebN = %i,eN_global=%i,ebN_element=%i,ebN_element=%i,i=%i,j=%i,jac=%12.5e,fjac=%12.5e,w=%12.5e\n", */
/*                                jacIndex,ebN,eN_global,ebN_element,ebN_element,i,j,jac[jacIndex], */
/*                                elementBoundaryFluxJacobian_eb[ebN*2*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_trial_element + */
/*                                                               0*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_trial_element+ */
/*                                                               ebN_element*nQuadraturePoints_elementBoundary*nDOF_trial_element+ */
/*                                                               k*nDOF_trial_element+ */
/*                                                               j], */
/*                                w_dS[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_test_element + */
/*                                     ebN_element*nQuadraturePoints_elementBoundary*nDOF_test_element+ */
/*                                     k*nDOF_test_element+ */
/*                                     i]); */
                      }
                }
            }
        }
    }
}
/**
   \brief Update the global dense Jacobian from the element boundary flux Jacobians at exterior boundaries
          only difference from updateGlobalJacobianFromGlobalExteriorElementBoundaryFluxJacobian
          is test function dimensionality right now
*/
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
                                                                                double* jac)
{
  proteus_index_t ebN, eN_global, eN_ebN, I, J, jacIndex;  int ebNE, ebN_element, ebN_eN, ii, i, k, jj, j;
  for (ebNE=0;ebNE<nExteriorElementBoundaries_global;ebNE++)
    {
      ebN = exteriorElementBoundaries[ebNE];
      eN_global   = elementBoundaryElements[ebN*2+0];
      ebN_element  = elementBoundaryLocalElementBoundaries[ebN*2+0];
      for (ebN_eN=0;ebN_eN<nElementBoundaries_element;ebN_eN++)
        {
/*           /\* cek debugging 1D *\/ */
/*           if(ebN_eN == 0) */
/*             eN_ebN = eN_global+1; */
/*           if(ebN_eN == 1) */
/*             eN_ebN = eN_global-1; */
          eN_ebN = elementNeighbors[eN_global*nElementBoundaries_element+ebN_eN];
          for(ii=0;ii<nFreeDOF_element_r[eN_global];ii++)
            {
              i = freeLocal_r[eN_global*nDOF_test_element+
                              ii];
              I = offset_r + stride_r*freeGlobal_r[eN_global*nDOF_test_element+
                                                   ii];
              for(k=0;k<nQuadraturePoints_elementBoundary;k++)
                {
/*                   if(eN_ebN >= 0 && eN_ebN < nElements_global) */
                  if(eN_ebN >= 0)
                    for(jj=0;jj<nFreeDOF_element_u[eN_ebN];jj++)
                      {
                        j = freeLocal_u[eN_ebN*nDOF_trial_element+
                                        jj];
                        J = offset_u + stride_u*freeGlobal_u[eN_ebN*nDOF_trial_element+
                                                             jj];
                        jacIndex = I+J*nFreeVDOF_global;
                        jac[jacIndex]
                          +=
                          elementBoundaryFluxJacobian_eb[ebN*2*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_trial_element +
                                                         0*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                                                         ebN_eN*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                                                         k*nDOF_trial_element+
                                                         j]
                          *
                          w_dS[ebNE*nQuadraturePoints_elementBoundary*nDOF_test_element+
                               k*nDOF_test_element+
                               i];
/*                         printf("ind=%i,ebN = %i,eN_global=%i,ebN_element=%i,ebN_element=%i,i=%i,j=%i,jac=%12.5e,fjac=%12.5e,w=%12.5e\n", */
/*                                jacIndex,ebN,eN_global,ebN_element,ebN_element,i,j,jac[jacIndex], */
/*                                elementBoundaryFluxJacobian_eb[ebN*2*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_trial_element + */
/*                                                               0*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_trial_element+ */
/*                                                               ebN_element*nQuadraturePoints_elementBoundary*nDOF_trial_element+ */
/*                                                               k*nDOF_trial_element+ */
/*                                                               j], */
/*                                w_dS[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_test_element + */
/*                                     ebN_element*nQuadraturePoints_elementBoundary*nDOF_test_element+ */
/*                                     k*nDOF_test_element+ */
/*                                     i]); */
                      }
                }
            }
        }
    }
}
/**
   \brief Update the global dense Jacobian from the element boundary flux Jacobians at exterior boundaries
*/
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
                                                                             double* jac)
{
  proteus_index_t ebN, eN_global, I, J, jacIndex;  int ebNE, ebN_element, ii, i, k, jj, j;
  for (ebNE=0;ebNE<nExteriorElementBoundaries_global;ebNE++)
    {
      ebN = exteriorElementBoundaries[ebNE];
      eN_global   = elementBoundaryElements[ebN*2+0];
      ebN_element  = elementBoundaryLocalElementBoundaries[ebN*2+0];
      for(ii=0;ii<nFreeDOF_element_r[eN_global];ii++)
        {
          i = freeLocal_r[eN_global*nDOF_test_element+
                          ii];
          I = offset_r + stride_r*freeGlobal_r[eN_global*nDOF_test_element+
                                               ii];
          for(k=0;k<nQuadraturePoints_elementBoundary;k++)
            {
              for(jj=0;jj<nFreeDOF_element_u[eN_global];jj++)
                {
                  j = freeLocal_u[eN_global*nDOF_trial_element+
                                  jj];
                  J = offset_u + stride_u*freeGlobal_u[eN_global*nDOF_trial_element+
                                                       jj];
                  jacIndex = I+J*nFreeVDOF_global;
                  jac[jacIndex]
                    +=
                    elementBoundaryFluxJacobian[ebNE*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                                                k*nDOF_trial_element+
                                                j]
                    *
                    w_dS[ebNE*nQuadraturePoints_elementBoundary*nDOF_test_element+
                         k*nDOF_test_element+
                         i];
                }
            }
        }
    }
}

/**
   \brief  Update the global CSR jacobian  from the element Jacobians.
*/
void updateGlobalJacobianFromElementJacobian_CSR(proteus_index_t nElements_global,
                                                 int nDOF_test_element,
                                                 int nDOF_trial_element,
                                                 int* nFreeDOF_element_r,
                                                 int* freeLocal_r,
                                                 int* nFreeDOF_element_u,
                                                 int* freeLocal_u,
                                                 proteus_index_t* csrRowIndeces_ru,
                                                 proteus_index_t* csrColumnOffsets_ru,
                                                 double* elementJacobian,
                                                 double* globalJacobian)
{
  proteus_index_t nDOF_test_X_trial_element=nDOF_test_element*nDOF_trial_element;  proteus_index_t jacIndex; int eN, ii, jj, i, j;
  for (eN=0;eN<nElements_global;eN++)
    for (ii=0;ii<nFreeDOF_element_r[eN];ii++)
      {
        i = freeLocal_r[eN*nDOF_test_element+
                      ii];
        for (jj=0;jj<nFreeDOF_element_u[eN];jj++)
          {
            j = freeLocal_u[eN*nDOF_trial_element+
                          jj];
            jacIndex = csrRowIndeces_ru[eN*nDOF_test_element+
                                     ii]
              +
              csrColumnOffsets_ru[eN*nDOF_test_X_trial_element+
                                  ii*nDOF_trial_element+
                                  jj];
            globalJacobian[jacIndex]
              +=
              elementJacobian[eN*nDOF_test_X_trial_element +
                              i*nDOF_trial_element +
                              j];
          }
      }
}

/**
   \brief  Update the global CSR jacobian  from the element Jacobians.
*/
void updateGlobalJacobianFromElementJacobian_eb_CSR(proteus_index_t* elementNeighbors,
                                                    proteus_index_t nElements_global,
                                                    int nElementBoundaries_element,
                                                    int nDOF_test_element,
                                                    int nDOF_trial_element,
                                                    int* nFreeDOF_element_r,
                                                    int* freeLocal_r,
                                                    int* nFreeDOF_element_u,
                                                    int* freeLocal_u,
                                                    proteus_index_t* csrRowIndeces_ru,
                                                    proteus_index_t* csrColumnOffsets_eb_ru,
                                                    double* elementJacobian_eb,
                                                    double* globalJacobian)
{
  proteus_index_t nDOF_test_X_trial_element=nDOF_test_element*nDOF_trial_element;  proteus_index_t ebN, eN_ebN, jacIndex; int eN, ii, jj, i, j;
  for (eN=0;eN<nElements_global;eN++)
    for (ebN=0;ebN<nElementBoundaries_element;ebN++)
      {
/*         /\*cek 1D debugging *\/ */
/*         /\** \todo cek Need to add mesh structure for getting element's corresponding to element boundaries and  */
/*            fix sparse matrix load of off element jacobian terms in the element Jacobians for LDG methods *\/ */
/*         if(ebN == 0) */
/*           eN_ebN = eN+1; */
/*         if(ebN == 1) */
/*           eN_ebN = eN-1; */
        eN_ebN = elementNeighbors[eN*nElementBoundaries_element+ebN];
        if (eN_ebN >= 0)
          for (ii=0;ii<nFreeDOF_element_r[eN];ii++)
            {
              i = freeLocal_r[eN*nDOF_test_element+
                              ii];
              for (jj=0;jj<nFreeDOF_element_u[eN_ebN];jj++)
                {
                  j = freeLocal_u[eN_ebN*nDOF_trial_element+
                                  jj];
                  jacIndex = csrRowIndeces_ru[eN*nDOF_test_element+
                                              ii]
                    +
                    csrColumnOffsets_eb_ru[eN*nElementBoundaries_element*nDOF_test_X_trial_element+
                                           ebN*nDOF_test_X_trial_element+
                                           ii*nDOF_trial_element+
                                           jj];
                  globalJacobian[jacIndex]
                    +=
                    elementJacobian_eb[eN*nElementBoundaries_element*nDOF_test_X_trial_element +
                                       ebN*nDOF_test_X_trial_element+
                                       i*nDOF_trial_element +
                                       j];
                }
            }
      }
}

/**
   \brief Update the global CSR Jacobian from the element boundary flux Jacobians at interior boundaries
*/
void updateGlobalJacobianFromInteriorElementBoundaryFluxJacobian_CSR(proteus_index_t nInteriorElementBoundaries_global,
                                                                     int nElementBoundaries_element,
                                                                     int nQuadraturePoints_elementBoundary,
                                                                     int nDOF_test_element,
                                                                     int nDOF_trial_element,
                                                                     proteus_index_t* interiorElementBoundaries,
                                                                     proteus_index_t* elementBoundaryElements,
                                                                     int* elementBoundaryLocalElementBoundaries,
                                                                     int* nFreeDOF_element_r,
                                                                     int* freeLocal_r,
                                                                     int* nFreeDOF_element_u,
                                                                     int* freeLocal_u,
                                                                     proteus_index_t* csrRowIndeces_ru,
                                                                     proteus_index_t* csrColumnOffsets_eb_ru,
                                                                     double* elementBoundaryFluxJacobian,
                                                                     double* w_dS,
                                                                     double* jac)
{
  proteus_index_t nDOF_test_X_trial_element=nDOF_test_element*nDOF_trial_element;  proteus_index_t ebN, left_eN_global, right_eN_global, jacIndex; int ebNI, left_ebN_element, right_ebN_element, ii, i, k, jj, j;
  for (ebNI=0;ebNI<nInteriorElementBoundaries_global;ebNI++)
    {
      ebN = interiorElementBoundaries[ebNI];
      left_eN_global   = elementBoundaryElements[ebN*2+0];
      right_eN_global  = elementBoundaryElements[ebN*2+1];
      left_ebN_element  = elementBoundaryLocalElementBoundaries[ebN*2+0];
      right_ebN_element = elementBoundaryLocalElementBoundaries[ebN*2+1];
      for(ii=0;ii<nFreeDOF_element_r[left_eN_global];ii++)
        {
          i = freeLocal_r[left_eN_global*nDOF_test_element+ii];
          for(k=0;k<nQuadraturePoints_elementBoundary;k++)
            {
              for(jj=0;jj<nFreeDOF_element_u[left_eN_global];jj++)
                {
                  j = freeLocal_u[left_eN_global*nDOF_trial_element+
                                  jj];
                  jacIndex = csrRowIndeces_ru[left_eN_global*nDOF_test_element+
                                              ii]
                    +
                    csrColumnOffsets_eb_ru[ebN*4*nDOF_test_X_trial_element +
                                           0*2*nDOF_test_X_trial_element +
                                           0*nDOF_test_X_trial_element +
                                           ii*nDOF_trial_element +
                                           jj];
                  jac[jacIndex]
                    +=
                    elementBoundaryFluxJacobian[ebN*2*nQuadraturePoints_elementBoundary*nDOF_trial_element +
                                                0*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                                                k*nDOF_trial_element+
                                                j]
                    *
                    w_dS[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_test_element +
                         left_ebN_element*nQuadraturePoints_elementBoundary*nDOF_test_element+
                         k*nDOF_test_element+
                         i];
                }
              for(jj=0;jj<nFreeDOF_element_u[right_eN_global];jj++)
                {
                  j = freeLocal_u[right_eN_global*nDOF_trial_element+
                                jj];
                  jacIndex = csrRowIndeces_ru[left_eN_global*nDOF_test_element+
                                           ii]
                    +
                    csrColumnOffsets_eb_ru[ebN*4*nDOF_test_X_trial_element+
                                        0*2*nDOF_test_X_trial_element+
                                        1*nDOF_test_X_trial_element+
                                        ii*nDOF_trial_element+
                                        jj];
                  jac[jacIndex]
                    +=
                    elementBoundaryFluxJacobian[ebN*2*nQuadraturePoints_elementBoundary*nDOF_trial_element +
                                                1*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                                                k*nDOF_trial_element+
                                                j]
                    *
                    w_dS[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_test_element +
                         left_ebN_element*nQuadraturePoints_elementBoundary*nDOF_test_element+
                         k*nDOF_test_element+
                         i];
                }
            }
        }
      for(ii=0;ii<nFreeDOF_element_r[right_eN_global];ii++)
        {
          i = freeLocal_r[right_eN_global*nDOF_test_element+
                        ii];
          for(k=0;k<nQuadraturePoints_elementBoundary;k++)
            {
              for(jj=0;jj<nFreeDOF_element_u[left_eN_global];jj++)
                {
                  j = freeLocal_u[left_eN_global*nDOF_trial_element+
                                jj];
                  jacIndex = csrRowIndeces_ru[right_eN_global*nDOF_test_element+
                                           ii]
                    +
                    csrColumnOffsets_eb_ru[ebN*4*nDOF_test_X_trial_element+
                                        1*2*nDOF_test_X_trial_element+
                                        0*nDOF_test_X_trial_element+
                                        ii*nDOF_trial_element+
                                        jj];
                  jac[jacIndex]
                    -=
                    elementBoundaryFluxJacobian[ebN*2*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                                                0*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                                                k*nDOF_trial_element+
                                                j]
                    *
                    w_dS[right_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_test_element+
                         right_ebN_element*nQuadraturePoints_elementBoundary*nDOF_test_element+
                         k*nDOF_test_element+
                         i];
                }
              for(jj=0;jj<nFreeDOF_element_u[right_eN_global];jj++)
                {
                  j = freeLocal_u[right_eN_global*nDOF_trial_element+
                                jj];
                  jacIndex = csrRowIndeces_ru[right_eN_global*nDOF_test_element+
                                           ii]
                    +
                    csrColumnOffsets_eb_ru[ebN*4*nDOF_test_X_trial_element+
                                        1*2*nDOF_test_X_trial_element+
                                        1*nDOF_test_X_trial_element+
                                        ii*nDOF_trial_element+
                                        jj];
                  jac[jacIndex]
                    -=
                    elementBoundaryFluxJacobian[ebN*2*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                                                1*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                                                k*nDOF_trial_element+
                                                j]*
                    w_dS[right_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_test_element+
                         right_ebN_element*nQuadraturePoints_elementBoundary*nDOF_test_element+
                         k*nDOF_test_element+
                         i];
                }
            }
        }
    }
}

/**
   \brief Update the global CSR Jacobian from the element boundary flux Jacobians at exterior boundaries
*/
void updateGlobalJacobianFromExteriorElementBoundaryFluxJacobian_CSR(proteus_index_t nExteriorElementBoundaries_global,
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
                                                                     proteus_index_t* csrRowIndeces_ru,
                                                                     proteus_index_t* csrColumnOffsets_eb_ru,
                                                                     double* elementBoundaryFluxJacobian,
                                                                     double* w_dS,
                                                                     double* jac)
{
  proteus_index_t nDOF_test_X_trial_element=nDOF_test_element*nDOF_trial_element;  proteus_index_t ebN, eN_global, jacIndex; int ebNE, ebN_element, ii, i, k, jj, j;
  for (ebNE=0;ebNE<nExteriorElementBoundaries_global;ebNE++)
    {
      ebN = exteriorElementBoundaries[ebNE];
      eN_global   = elementBoundaryElements[ebN*2+0];
      ebN_element  = elementBoundaryLocalElementBoundaries[ebN*2+0];
      for(ii=0;ii<nFreeDOF_element_r[eN_global];ii++)
        {
          i = freeLocal_r[eN_global*nDOF_test_element+ii];
          for(k=0;k<nQuadraturePoints_elementBoundary;k++)
            {
              for(jj=0;jj<nFreeDOF_element_u[eN_global];jj++)
                {
                  j = freeLocal_u[eN_global*nDOF_trial_element+
                                  jj];
                  jacIndex = csrRowIndeces_ru[eN_global*nDOF_test_element+
                                              ii]
                    +
                    csrColumnOffsets_eb_ru[ebN*4*nDOF_test_X_trial_element +
                                           0*2*nDOF_test_X_trial_element +
                                           0*nDOF_test_X_trial_element +
                                           ii*nDOF_trial_element +
                                           jj];
                  jac[jacIndex]
                    +=
                    elementBoundaryFluxJacobian[ebN*2*nQuadraturePoints_elementBoundary*nDOF_trial_element +
                                                0*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                                                k*nDOF_trial_element+
                                                j]
                    *
                    w_dS[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_test_element +
                         ebN_element*nQuadraturePoints_elementBoundary*nDOF_test_element+
                         k*nDOF_test_element+
                         i];
                }
            }
        }
    }
}
/**
   \brief Update the global CSR Jacobian from the element boundary flux Jacobians at exterior boundaries
*/
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
                                                                           proteus_index_t* csrRowIndeces_ru,
                                                                           proteus_index_t* csrColumnOffsets_eb_ru,
                                                                           double* elementBoundaryFluxJacobian,
                                                                           double* w_dS,
                                                                           double* jac)
{
  proteus_index_t nDOF_test_X_trial_element=nDOF_test_element*nDOF_trial_element;  proteus_index_t ebN, eN_global, jacIndex; int ebNE, ii, i, k, jj, j;
  for (ebNE=0;ebNE<nExteriorElementBoundaries_global;ebNE++)
    {
      ebN = exteriorElementBoundaries[ebNE];
      eN_global   = elementBoundaryElements[ebN*2+0];
      for(ii=0;ii<nFreeDOF_element_r[eN_global];ii++)
        {
          i = freeLocal_r[eN_global*nDOF_test_element+ii];
          for(k=0;k<nQuadraturePoints_elementBoundary;k++)
            {
              for(jj=0;jj<nFreeDOF_element_u[eN_global];jj++)
                {
                  j = freeLocal_u[eN_global*nDOF_trial_element+
                                  jj];
                  jacIndex = csrRowIndeces_ru[eN_global*nDOF_test_element+
                                              ii]
                    +
                    csrColumnOffsets_eb_ru[ebN*4*nDOF_test_X_trial_element +
                                           0*2*nDOF_test_X_trial_element +
                                           0*nDOF_test_X_trial_element +
                                           ii*nDOF_trial_element +
                                           jj];
                  jac[jacIndex]
                    +=
                    elementBoundaryFluxJacobian[ebNE*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                                                k*nDOF_trial_element+
                                                j]
                    *
                    w_dS[ebNE*nQuadraturePoints_elementBoundary*nDOF_test_element+
                         k*nDOF_test_element+
                         i];
                }
            }
        }
    }
}

/**
   \brief Update the global CSR Jacobian from the element boundary flux Jacobians at interior boundaries
*/
void updateGlobalJacobianFromInteriorElementBoundaryFluxJacobian_eb_CSR(proteus_index_t* elementNeighbors,
                                                                        proteus_index_t nInteriorElementBoundaries_global,
                                                                     int nElementBoundaries_element,
                                                                     int nQuadraturePoints_elementBoundary,
                                                                     int nDOF_test_element,
                                                                     int nDOF_trial_element,
                                                                     proteus_index_t* interiorElementBoundaries,
                                                                     proteus_index_t* elementBoundaryElements,
                                                                     int* elementBoundaryLocalElementBoundaries,
                                                                     int* nFreeDOF_element_r,
                                                                     int* freeLocal_r,
                                                                     int* nFreeDOF_element_u,
                                                                     int* freeLocal_u,
                                                                     proteus_index_t* csrRowIndeces_ru,
                                                                     proteus_index_t* csrColumnOffsets_eb_eNebN_ru,
                                                                     double* elementBoundaryFluxJacobian_eb,
                                                                     double* w_dS,
                                                                     double* jac)
{
  proteus_index_t nDOF_test_X_trial_element=nDOF_test_element*nDOF_trial_element;  proteus_index_t ebN, left_eN_ebN, right_eN_ebN, left_eN_global, right_eN_global, jacIndex; int ebNI, ebN_element, left_ebN_element, right_ebN_element, ii, i, k, jj, j;
  for (ebNI=0;ebNI<nInteriorElementBoundaries_global;ebNI++)
    {
      ebN = interiorElementBoundaries[ebNI];
      left_eN_global   = elementBoundaryElements[ebN*2+0];
      right_eN_global  = elementBoundaryElements[ebN*2+1];
      left_ebN_element  = elementBoundaryLocalElementBoundaries[ebN*2+0];
      right_ebN_element = elementBoundaryLocalElementBoundaries[ebN*2+1];
      for(ebN_element=0;ebN_element<nElementBoundaries_element;ebN_element++)
        {
          left_eN_ebN = elementNeighbors[left_eN_global*nElementBoundaries_element+ebN_element];
          right_eN_ebN = elementNeighbors[right_eN_global*nElementBoundaries_element+ebN_element];
          for(ii=0;ii<nFreeDOF_element_r[left_eN_global];ii++)
            {
              i = freeLocal_r[left_eN_global*nDOF_test_element+ii];
              for(k=0;k<nQuadraturePoints_elementBoundary;k++)
                {
                  if(left_eN_ebN >= 0)
                    for(jj=0;jj<nFreeDOF_element_u[left_eN_ebN];jj++)
                      {
                        j = freeLocal_u[left_eN_ebN*nDOF_trial_element+
                                        jj];
                        jacIndex = csrRowIndeces_ru[left_eN_global*nDOF_test_element+
                                                    ii]
                          +
                          csrColumnOffsets_eb_eNebN_ru[ebN*4*nElementBoundaries_element*nDOF_test_X_trial_element +
                                                       0*2*nElementBoundaries_element*nDOF_test_X_trial_element +
                                                       0*nElementBoundaries_element*nDOF_test_X_trial_element +
                                                       ebN_element*nDOF_test_X_trial_element+
                                                       ii*nDOF_trial_element +
                                                       jj];
                        jac[jacIndex]
                          +=
                          elementBoundaryFluxJacobian_eb[ebN*2*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_trial_element +
                                                         0*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                                                         ebN_element*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                                                         k*nDOF_trial_element+
                                                         j]
                          *
                          w_dS[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_test_element +
                               left_ebN_element*nQuadraturePoints_elementBoundary*nDOF_test_element+
                               k*nDOF_test_element+
                               i];
/*                         printf("ind=%i,ebN = %i,left_eN_global=%i,left_ebN_element=%i,ebN_element=%i,i=%i,j=%i,jac=%12.5e,fjac=%12.5e,w=%12.5e\n", */
/*                                jacIndex,ebN,left_eN_global,left_ebN_element,ebN_element,i,j,jac[jacIndex], */
/*                                elementBoundaryFluxJacobian_eb[ebN*2*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_trial_element +  */
/*                                                               0*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_trial_element+ */
/*                                                               ebN_element*nQuadraturePoints_elementBoundary*nDOF_trial_element+ */
/*                                                               k*nDOF_trial_element+ */
/*                                                               j], */
/*                                w_dS[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_test_element +  */
/*                                     left_ebN_element*nQuadraturePoints_elementBoundary*nDOF_test_element+ */
/*                                     k*nDOF_test_element+ */
/*                                     i]); */
                      }
                  if(right_eN_ebN >= 0)
                    for(jj=0;jj<nFreeDOF_element_u[right_eN_ebN];jj++)
                      {
                        j = freeLocal_u[right_eN_ebN*nDOF_trial_element+
                                        jj];
                        jacIndex = csrRowIndeces_ru[left_eN_global*nDOF_test_element+
                                                    ii]
                          +
                          csrColumnOffsets_eb_eNebN_ru[ebN*4*nElementBoundaries_element*nDOF_test_X_trial_element+
                                                       0*2*nElementBoundaries_element*nDOF_test_X_trial_element+
                                                       1*nElementBoundaries_element*nDOF_test_X_trial_element+
                                                       ebN_element*nDOF_test_X_trial_element+
                                                       ii*nDOF_trial_element+
                                                       jj];
                        jac[jacIndex]
                          +=
                          elementBoundaryFluxJacobian_eb[ebN*2*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_trial_element +
                                                         1*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                                                         ebN_element*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                                                         k*nDOF_trial_element+
                                                         j]
                          *
                          w_dS[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_test_element +
                               left_ebN_element*nQuadraturePoints_elementBoundary*nDOF_test_element+
                               k*nDOF_test_element+
                               i];
/*                         printf("ind=%i,ebN = %i,left_eN_global=%i,left_ebN_element=%i,ebN_element=%i,i=%i,j=%i,jac=%12.5e,fjac=%12.5e,w=%12.5e\n", */
/*                                jacIndex,ebN,left_eN_global,left_ebN_element,ebN_element,i,j,jac[jacIndex], */
/*                                elementBoundaryFluxJacobian_eb[ebN*2*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_trial_element +  */
/*                                                               1*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_trial_element+ */
/*                                                               ebN_element*nQuadraturePoints_elementBoundary*nDOF_trial_element+ */
/*                                                               k*nDOF_trial_element+ */
/*                                                               j], */
/*                                w_dS[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_test_element +  */
/*                                     left_ebN_element*nQuadraturePoints_elementBoundary*nDOF_test_element+ */
/*                                     k*nDOF_test_element+ */
/*                                     i]); */
                      }
                }
            }
          for(ii=0;ii<nFreeDOF_element_r[right_eN_global];ii++)
            {
              i = freeLocal_r[right_eN_global*nDOF_test_element+
                              ii];
              for(k=0;k<nQuadraturePoints_elementBoundary;k++)
                {
                  if(left_eN_ebN >= 0)
                    for(jj=0;jj<nFreeDOF_element_u[left_eN_ebN];jj++)
                      {
                        j = freeLocal_u[left_eN_ebN*nDOF_trial_element+
                                        jj];
                        jacIndex = csrRowIndeces_ru[right_eN_global*nDOF_test_element+
                                                    ii]
                          +
                          csrColumnOffsets_eb_eNebN_ru[ebN*4*nElementBoundaries_element*nDOF_test_X_trial_element+
                                                       1*2*nElementBoundaries_element*nDOF_test_X_trial_element+
                                                       0*nElementBoundaries_element*nDOF_test_X_trial_element+
                                                       ebN_element*nDOF_test_X_trial_element+
                                                       ii*nDOF_trial_element+
                                                       jj];
                        jac[jacIndex]
                          -=
                          elementBoundaryFluxJacobian_eb[ebN*2*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                                                         0*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                                                         ebN_element*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                                                         k*nDOF_trial_element+
                                                         j]
                          *
                          w_dS[right_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_test_element+
                               right_ebN_element*nQuadraturePoints_elementBoundary*nDOF_test_element+
                               k*nDOF_test_element+
                               i];
/*                         printf("ind=%i,ebN = %i,left_eN_global=%i,left_ebN_element=%i,ebN_element=%i,i=%i,j=%i,jac=%12.5e,fjac=%12.5e,w=%12.5e\n", */
/*                                jacIndex,ebN,right_eN_global,right_ebN_element,ebN_element,i,j,jac[jacIndex], */
/*                                elementBoundaryFluxJacobian_eb[ebN*2*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_trial_element +  */
/*                                                               0*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_trial_element+ */
/*                                                               ebN_element*nQuadraturePoints_elementBoundary*nDOF_trial_element+ */
/*                                                               k*nDOF_trial_element+ */
/*                                                               j], */
/*                                w_dS[right_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_test_element +  */
/*                                     right_ebN_element*nQuadraturePoints_elementBoundary*nDOF_test_element+ */
/*                                     k*nDOF_test_element+ */
/*                                     i]); */
                      }
                  if(right_eN_ebN >= 0)
                    for(jj=0;jj<nFreeDOF_element_u[right_eN_ebN];jj++)
                      {
                        j = freeLocal_u[right_eN_ebN*nDOF_trial_element+
                                        jj];
                        jacIndex = csrRowIndeces_ru[right_eN_global*nDOF_test_element+
                                                    ii]
                          +
                          csrColumnOffsets_eb_eNebN_ru[ebN*4*nElementBoundaries_element*nDOF_test_X_trial_element+
                                                       1*2*nElementBoundaries_element*nDOF_test_X_trial_element+
                                                       1*nElementBoundaries_element*nDOF_test_X_trial_element+
                                                       ebN_element*nDOF_test_X_trial_element+
                                                       ii*nDOF_trial_element+
                                                       jj];
                        jac[jacIndex]
                          -=
                          elementBoundaryFluxJacobian_eb[ebN*2*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                                                         1*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                                                         ebN_element*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                                                         k*nDOF_trial_element+
                                                         j]*
                          w_dS[right_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_test_element+
                               right_ebN_element*nQuadraturePoints_elementBoundary*nDOF_test_element+
                               k*nDOF_test_element+
                               i];
/*                         printf("ind=%i,ebN = %i,left_eN_global=%i,left_ebN_element=%i,ebN_element=%i,i=%i,j=%i,jac=%12.5e,fjac=%12.5e,w=%12.5e\n", */
/*                                jacIndex,ebN,right_eN_global,right_ebN_element,ebN_element,i,j,jac[jacIndex], */
/*                                elementBoundaryFluxJacobian_eb[ebN*2*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_trial_element +  */
/*                                                               1*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_trial_element+ */
/*                                                               ebN_element*nQuadraturePoints_elementBoundary*nDOF_trial_element+ */
/*                                                               k*nDOF_trial_element+ */
/*                                                               j], */
/*                                w_dS[right_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_test_element +  */
/*                                     right_ebN_element*nQuadraturePoints_elementBoundary*nDOF_test_element+ */
/*                                     k*nDOF_test_element+ */
/*                                     i]); */
                      }
                }
            }
        }
    }
}

/**
   \brief Update the global CSR Jacobian from the element boundary flux Jacobians at exterior boundaries
*/
void updateGlobalJacobianFromExteriorElementBoundaryFluxJacobian_eb_CSR(proteus_index_t* elementNeighbors,
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
                                                                     proteus_index_t* csrRowIndeces_ru,
                                                                     proteus_index_t* csrColumnOffsets_eb_eNebN_ru,
                                                                     double* elementBoundaryFluxJacobian_eb,
                                                                     double* w_dS,
                                                                     double* jac)
{
  proteus_index_t nDOF_test_X_trial_element=nDOF_test_element*nDOF_trial_element;  proteus_index_t ebN, eN_ebN, eN_global, jacIndex; int ebNE, ebN_eN, ebN_element, ii, i, k, jj, j;
  for (ebNE=0;ebNE<nExteriorElementBoundaries_global;ebNE++)
    {
      ebN = exteriorElementBoundaries[ebNE];
      eN_global   = elementBoundaryElements[ebN*2+0];
      ebN_element  = elementBoundaryLocalElementBoundaries[ebN*2+0];
      for(ebN_eN=0;ebN_eN<nElementBoundaries_element;ebN_eN++)
        {
          eN_ebN = elementNeighbors[eN_global*nElementBoundaries_element+ebN_eN];
          for(ii=0;ii<nFreeDOF_element_r[eN_global];ii++)
            {
              i = freeLocal_r[eN_global*nDOF_test_element+ii];
              for(k=0;k<nQuadraturePoints_elementBoundary;k++)
                {
                  if(eN_ebN >= 0)
                    for(jj=0;jj<nFreeDOF_element_u[eN_ebN];jj++)
                      {
                        j = freeLocal_u[eN_ebN*nDOF_trial_element+
                                        jj];
                        jacIndex = csrRowIndeces_ru[eN_global*nDOF_test_element+
                                                    ii]
                          +
                          csrColumnOffsets_eb_eNebN_ru[ebN*4*nElementBoundaries_element*nDOF_test_X_trial_element +
                                                       0*2*nElementBoundaries_element*nDOF_test_X_trial_element +
                                                       0*nElementBoundaries_element*nDOF_test_X_trial_element +
                                                       ebN_eN*nDOF_test_X_trial_element +
                                                       ii*nDOF_trial_element +
                                                       jj];
                        jac[jacIndex]
                          +=
                          elementBoundaryFluxJacobian_eb[ebN*2*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_trial_element +
                                                      0*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                                                      ebN_eN*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                                                      k*nDOF_trial_element+
                                                      j]
                          *
                          w_dS[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_test_element +
                               ebN_element*nQuadraturePoints_elementBoundary*nDOF_test_element+
                               k*nDOF_test_element+
                               i];
/*                         printf("ind=%i,ebN = %i,eN_global=%i,ebN_element=%i,ebN_element=%i,i=%i,j=%i,jac=%12.5e,fjac=%12.5e,w=%12.5e\n", */
/*                                jacIndex,ebN,eN_global,ebN_element,ebN_element,i,j,jac[jacIndex], */
/*                                elementBoundaryFluxJacobian_eb[ebN*2*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_trial_element + */
/*                                                               0*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_trial_element+ */
/*                                                               ebN_element*nQuadraturePoints_elementBoundary*nDOF_trial_element+ */
/*                                                               k*nDOF_trial_element+ */
/*                                                               j], */
/*                                w_dS[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_test_element + */
/*                                     ebN_element*nQuadraturePoints_elementBoundary*nDOF_test_element+ */
/*                                     k*nDOF_test_element+ */
/*                                     i]); */
                      }
                }
            }
        }
    }
}
/**
   \brief Update the global CSR Jacobian from the element boundary flux Jacobians at exterior boundaries
          only difference from updateGlobalJacobianFromGlobalExteriorElementBoundaryFluxJacobian
          is test function dimensionality right now
*/
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
                                                                              proteus_index_t* csrRowIndeces_ru,
                                                                              proteus_index_t* csrColumnOffsets_eb_eNebN_ru,
                                                                              double* elementBoundaryFluxJacobian_eb,
                                                                              double* w_dS,
                                                                              double* jac)
{
  proteus_index_t nDOF_test_X_trial_element=nDOF_test_element*nDOF_trial_element;  proteus_index_t ebN, eN_ebN, eN_global, jacIndex; int ebNE, ebN_eN, ebN_element, ii, i, k, jj, j;
  for (ebNE=0;ebNE<nExteriorElementBoundaries_global;ebNE++)
    {
      ebN = exteriorElementBoundaries[ebNE];
      eN_global   = elementBoundaryElements[ebN*2+0];
      ebN_element  = elementBoundaryLocalElementBoundaries[ebN*2+0];
      for(ebN_eN=0;ebN_eN<nElementBoundaries_element;ebN_eN++)
        {
          eN_ebN = elementNeighbors[eN_global*nElementBoundaries_element+ebN_eN];
          for(ii=0;ii<nFreeDOF_element_r[eN_global];ii++)
            {
              i = freeLocal_r[eN_global*nDOF_test_element+ii];
              for(k=0;k<nQuadraturePoints_elementBoundary;k++)
                {
                  if(eN_ebN >= 0)
                    for(jj=0;jj<nFreeDOF_element_u[eN_ebN];jj++)
                      {
                        j = freeLocal_u[eN_ebN*nDOF_trial_element+
                                        jj];
                        jacIndex = csrRowIndeces_ru[eN_global*nDOF_test_element+
                                                    ii]
                          +
                          csrColumnOffsets_eb_eNebN_ru[ebN*4*nElementBoundaries_element*nDOF_test_X_trial_element +
                                                       0*2*nElementBoundaries_element*nDOF_test_X_trial_element +
                                                       0*nElementBoundaries_element*nDOF_test_X_trial_element +
                                                       ebN_eN*nDOF_test_X_trial_element +
                                                       ii*nDOF_trial_element +
                                                       jj];
                        jac[jacIndex]
                          +=
                          elementBoundaryFluxJacobian_eb[ebN*2*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_trial_element +
                                                      0*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                                                      ebN_eN*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                                                      k*nDOF_trial_element+
                                                      j]
                          *
                          w_dS[ebNE*nQuadraturePoints_elementBoundary*nDOF_test_element+
                               k*nDOF_test_element+
                               i];
/*                         printf("ind=%i,ebN = %i,eN_global=%i,ebN_element=%i,ebN_element=%i,i=%i,j=%i,jac=%12.5e,fjac=%12.5e,w=%12.5e\n", */
/*                                jacIndex,ebN,eN_global,ebN_element,ebN_element,i,j,jac[jacIndex], */
/*                                elementBoundaryFluxJacobian_eb[ebN*2*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_trial_element + */
/*                                                               0*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_trial_element+ */
/*                                                               ebN_element*nQuadraturePoints_elementBoundary*nDOF_trial_element+ */
/*                                                               k*nDOF_trial_element+ */
/*                                                               j], */
/*                                w_dS[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_test_element + */
/*                                     ebN_element*nQuadraturePoints_elementBoundary*nDOF_test_element+ */
/*                                     k*nDOF_test_element+ */
/*                                     i]); */
                      }
                }
            }
        }
    }
}
/**
   \brief Update the global CSR Jacobian from the element boundary two-sided Hamiltonian flux Jacobians at interior boundaries
*/
void updateGlobalJacobianFromInteriorElementBoundaryFluxJacobian_2sided_CSR(proteus_index_t nInteriorElementBoundaries_global,
                                                                            int nElementBoundaries_element,
                                                                            int nQuadraturePoints_elementBoundary,
                                                                            int nDOF_test_element,
                                                                            int nDOF_trial_element,
                                                                            proteus_index_t* interiorElementBoundaries,
                                                                            proteus_index_t* elementBoundaryElements,
                                                                            int* elementBoundaryLocalElementBoundaries,
                                                                            int* nFreeDOF_element_r,
                                                                            int* freeLocal_r,
                                                                            int* nFreeDOF_element_u,
                                                                            int* freeLocal_u,
                                                                            proteus_index_t* csrRowIndeces_ru,
                                                                            proteus_index_t* csrColumnOffsets_eb_ru,
                                                                            double* elementBoundaryFluxJacobian_2sided,
                                                                            double* w_dS,
                                                                            double* jac)
{
  proteus_index_t nDOF_test_X_trial_element=nDOF_test_element*nDOF_trial_element;  proteus_index_t ebN, left_eN_global, right_eN_global, jacIndex; int ebNI, left_ebN_element, right_ebN_element, ii, i, k, jj, j;
  for (ebNI=0;ebNI<nInteriorElementBoundaries_global;ebNI++)
    {
      ebN = interiorElementBoundaries[ebNI];
      left_eN_global   = elementBoundaryElements[ebN*2+0];
      right_eN_global  = elementBoundaryElements[ebN*2+1];
      left_ebN_element  = elementBoundaryLocalElementBoundaries[ebN*2+0];
      right_ebN_element = elementBoundaryLocalElementBoundaries[ebN*2+1];
      /*left flux*/
      for(ii=0;ii<nFreeDOF_element_r[left_eN_global];ii++)
        {
          i = freeLocal_r[left_eN_global*nDOF_test_element+ii];
          for(k=0;k<nQuadraturePoints_elementBoundary;k++)
            {
              for(jj=0;jj<nFreeDOF_element_u[left_eN_global];jj++)
                {
                  j = freeLocal_u[left_eN_global*nDOF_trial_element+
                                  jj];
                  jacIndex = csrRowIndeces_ru[left_eN_global*nDOF_test_element+
                                              ii]
                    +
                    csrColumnOffsets_eb_ru[ebN*4*nDOF_test_X_trial_element +
                                           0*2*nDOF_test_X_trial_element +
                                           0*nDOF_test_X_trial_element +
                                           ii*nDOF_trial_element +
                                           jj];
                  jac[jacIndex]
                    +=
                    elementBoundaryFluxJacobian_2sided[ebN*2*2*nQuadraturePoints_elementBoundary*nDOF_trial_element +
                                                       0*2*nQuadraturePoints_elementBoundary*nDOF_trial_element + /*left flux*/
                                                       0*nQuadraturePoints_elementBoundary*nDOF_trial_element+ /*left neig dep*/
                                                       k*nDOF_trial_element+
                                                       j]
                    *
                    w_dS[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_test_element +
                         left_ebN_element*nQuadraturePoints_elementBoundary*nDOF_test_element+
                         k*nDOF_test_element+
                         i];
                }
              for(jj=0;jj<nFreeDOF_element_u[right_eN_global];jj++)
                {
                  j = freeLocal_u[right_eN_global*nDOF_trial_element+
                                jj];
                  jacIndex = csrRowIndeces_ru[left_eN_global*nDOF_test_element+
                                           ii]
                    +
                    csrColumnOffsets_eb_ru[ebN*4*nDOF_test_X_trial_element+
                                        0*2*nDOF_test_X_trial_element+
                                        1*nDOF_test_X_trial_element+
                                        ii*nDOF_trial_element+
                                        jj];
                  jac[jacIndex]
                    +=
                    elementBoundaryFluxJacobian_2sided[ebN*2*2*nQuadraturePoints_elementBoundary*nDOF_trial_element +
                                                       0*2*nQuadraturePoints_elementBoundary*nDOF_trial_element + /*left flux*/
                                                       1*nQuadraturePoints_elementBoundary*nDOF_trial_element+ /*right neig dep*/
                                                       k*nDOF_trial_element+
                                                       j]
                    *
                    w_dS[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_test_element +
                         left_ebN_element*nQuadraturePoints_elementBoundary*nDOF_test_element+
                         k*nDOF_test_element+
                         i];
                }
            }
        }
      /*right flux*/
      for(ii=0;ii<nFreeDOF_element_r[right_eN_global];ii++)
        {
          i = freeLocal_r[right_eN_global*nDOF_test_element+
                        ii];
          for(k=0;k<nQuadraturePoints_elementBoundary;k++)
            {
              for(jj=0;jj<nFreeDOF_element_u[left_eN_global];jj++)
                {
                  j = freeLocal_u[left_eN_global*nDOF_trial_element+
                                jj];
                  jacIndex = csrRowIndeces_ru[right_eN_global*nDOF_test_element+
                                           ii]
                    +
                    csrColumnOffsets_eb_ru[ebN*4*nDOF_test_X_trial_element+
                                        1*2*nDOF_test_X_trial_element+
                                        0*nDOF_test_X_trial_element+
                                        ii*nDOF_trial_element+
                                        jj];
                  jac[jacIndex]
                    +=
                    elementBoundaryFluxJacobian_2sided[ebN*2*2*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                                                       1*2*nQuadraturePoints_elementBoundary*nDOF_trial_element + /*right flux*/
                                                       0*nQuadraturePoints_elementBoundary*nDOF_trial_element+ /*left neig dep*/
                                                       k*nDOF_trial_element+
                                                       j]
                    *
                    w_dS[right_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_test_element+
                         right_ebN_element*nQuadraturePoints_elementBoundary*nDOF_test_element+
                         k*nDOF_test_element+
                         i];
                }
              for(jj=0;jj<nFreeDOF_element_u[right_eN_global];jj++)
                {
                  j = freeLocal_u[right_eN_global*nDOF_trial_element+
                                jj];
                  jacIndex = csrRowIndeces_ru[right_eN_global*nDOF_test_element+
                                           ii]
                    +
                    csrColumnOffsets_eb_ru[ebN*4*nDOF_test_X_trial_element+
                                        1*2*nDOF_test_X_trial_element+
                                        1*nDOF_test_X_trial_element+
                                        ii*nDOF_trial_element+
                                        jj];
                  jac[jacIndex]
                    +=
                    elementBoundaryFluxJacobian_2sided[ebN*2*2*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                                                       1*2*nQuadraturePoints_elementBoundary*nDOF_trial_element + /*right flux*/
                                                       1*nQuadraturePoints_elementBoundary*nDOF_trial_element+ /*right neig dep*/
                                                       k*nDOF_trial_element+
                                                       j]*
                    w_dS[right_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_test_element+
                         right_ebN_element*nQuadraturePoints_elementBoundary*nDOF_test_element+
                         k*nDOF_test_element+
                         i];
                }
            }
        }
    }
}

/**
   \brief Weight the test function with the integration weights
*/
void calculateWeightedShape(proteus_index_t nElements_global,
                            int nQuadraturePoints_element,
                            int nDOF_test_element,
                            double* dVR,
                            double* abs_det_jac,
                            double* w,
                            double* w_dV)
{
  proteus_index_t eN;  int i, k;
  for (eN=0;eN<nElements_global;eN++)
    for (k=0;k<nQuadraturePoints_element;k++)
      for (i=0;i<nDOF_test_element;i++)
        w_dV[eN*nQuadraturePoints_element*nDOF_test_element+
             k*nDOF_test_element+
             i]
          =
          w[eN*nQuadraturePoints_element*nDOF_test_element+
            k*nDOF_test_element+
            i]
          *
          dVR[k]
          *
          abs_det_jac[eN*nQuadraturePoints_element+
                      k];
}

/**
   \brief Weight the test function with the integration weights
*/
void calculateWeightedShapeGradients(proteus_index_t nElements_global,
                                     int nQuadraturePoints_element,
                                     int nDOF_test_element,
                                     int nSpace,
                                     double* dVR,
                                     double* abs_det_jac,
                                     double* grad_w,
                                     double* grad_w_dV)
{
  proteus_index_t eN, I;  int i, k;
  for (eN=0;eN<nElements_global;eN++)
    for (k=0;k<nQuadraturePoints_element;k++)
      for (i=0;i<nDOF_test_element;i++)
        for (I=0;I<nSpace;I++)
          grad_w_dV[eN*nQuadraturePoints_element*nDOF_test_element*nSpace+
                    k*nDOF_test_element*nSpace+
                    i*nSpace+
                    I]
            =
            grad_w[eN*nQuadraturePoints_element*nDOF_test_element*nSpace+
                   k*nDOF_test_element*nSpace+
                   i*nSpace+
                   I]
            *
            dVR[k]
            *
            abs_det_jac[eN*nQuadraturePoints_element+
                        k];
}

/// THIS SHOULD BE REMOVED BEFORE MERGE
void calculateWeightedPiolaShapeGradients(proteus_index_t nElements_global,
                                          int nQuadraturePoints_element,
                                          int nDOF_test_element,
                                          int nSpace,
                                          double* dVR,
                                          double* abs_det_jac,
                                          double* grad_w,
                                          double* grad_w_dV)
{
  proteus_index_t eN, I;  int i, k;
  for (eN=0;eN<nElements_global;eN++)
    for (k=0;k<nQuadraturePoints_element;k++)
      for (i=0;i<nDOF_test_element;i++)
        for (I=0;I<nSpace;I++)
          grad_w_dV[eN*nQuadraturePoints_element*nDOF_test_element*nSpace+
                    k*nDOF_test_element*nSpace+
                    i*nSpace+
                    I]
            =
            grad_w[eN*nQuadraturePoints_element*nDOF_test_element*nSpace+
                   k*nDOF_test_element*nSpace+
                   i*nSpace+
                   I]
            *
            dVR[k]
            ;
}

void calculateWeightedShapeHessians(proteus_index_t nElements_global,
                                     int nQuadraturePoints_element,
                                     int nDOF_test_element,
                                     int nSpace,
                                     double* dVR,
                                     double* abs_det_jac,
                                     double* Hess_w,
                                     double* Hess_w_dV)
{
  proteus_index_t nSpace2=nSpace*nSpace;  proteus_index_t I, J; int eN, i, k;
  for (eN=0;eN<nElements_global;eN++)
    for (k=0;k<nQuadraturePoints_element;k++)
      for (i=0;i<nDOF_test_element;i++)
        for (I=0;I<nSpace;I++)
          for (J=0;J<nSpace;J++)
            Hess_w_dV[eN*nQuadraturePoints_element*nDOF_test_element*nSpace2+
                      k*nDOF_test_element*nSpace2+
                      i*nSpace2+
                      I*nSpace+
                      J]
              =
              Hess_w[eN*nQuadraturePoints_element*nDOF_test_element*nSpace2+
                     k*nDOF_test_element*nSpace2+
                     i*nSpace2+
                     I*nSpace+
                     J]
              *
              dVR[k]
              *
              abs_det_jac[eN*nQuadraturePoints_element+
                          k];
}

/**
   \brief Calcualte the tensor product of trial and test functions at the quadrature  points
*/
void calculateShape_X_weightedShape(proteus_index_t nElements_global,
                                    int nQuadraturePoints_element,
                                    int nDOF_trial_element,
                                    int nDOF_test_element,
                                    double* v,
                                    double* w_dV,
                                    double* v_X_w_dV)
{
  proteus_index_t nDOF_test_X_trial_element=nDOF_test_element*nDOF_trial_element;  int eN, k, i, j;
  for (eN=0;eN<nElements_global;eN++)
    for (k=0;k<nQuadraturePoints_element;k++)
      for (i=0;i<nDOF_test_element;i++)
        for (j=0;j<nDOF_trial_element;j++)
          v_X_w_dV[eN*nQuadraturePoints_element*nDOF_test_X_trial_element +
                   k*nDOF_test_X_trial_element+
                   j*nDOF_test_element+
                   i]
            =
            v[eN*nQuadraturePoints_element*nDOF_trial_element +
              k*nDOF_trial_element+
              j]
            *
            w_dV[eN*nQuadraturePoints_element*nDOF_test_element +
                 k*nDOF_test_element+
                 i];
}

/**
   \brief Calculate  the tensor  product of trial functions and test function gradients at the quadrature  points.
*/
void calculateShape_X_weightedGradShape(proteus_index_t nElements_global,
                                        int nQuadraturePoints_element,
                                        int nDOF_trial_element,
                                        int nDOF_test_element,
                                        int nSpace,
                                        double* v,
                                        double* grad_w_dV,
                                        double* v_X_grad_w_dV)
{
  proteus_index_t nDOF_test_X_trial_element=nDOF_test_element*nDOF_trial_element;  proteus_index_t I; int eN, k, i, j;
  for (eN=0;eN<nElements_global;eN++)
    for (k=0;k<nQuadraturePoints_element;k++)
      for (i=0;i<nDOF_test_element;i++)
        for (j=0;j<nDOF_trial_element;j++)
          for (I=0;I<nSpace;I++)
            v_X_grad_w_dV[eN*nQuadraturePoints_element*nDOF_test_X_trial_element*nSpace +
                          k*nDOF_test_X_trial_element*nSpace+
                          j*nDOF_test_element*nSpace+
                          i*nSpace+
                          I]
              =
              v[eN*nQuadraturePoints_element*nDOF_trial_element +
                k*nDOF_trial_element+
                j]
              *
              grad_w_dV[eN*nQuadraturePoints_element*nDOF_test_element*nSpace +
                        k*nDOF_test_element*nSpace+
                        i*nSpace+
                        I];
}

/**
   \brief Calculate  the tensor  product of trial function gradients and test functions at the quadrature points.
*/
void calculateGradShape_X_weightedShape(proteus_index_t nElements_global,
                                        int nQuadraturePoints_element,
                                        int nDOF_trial_element,
                                        int nDOF_test_element,
                                        int nSpace,
                                        double* grad_v,
                                        double* w_dV,
                                        double* grad_v_X_w_dV)
{
  proteus_index_t nDOF_test_X_trial_element=nDOF_test_element*nDOF_trial_element;  proteus_index_t I; int eN, k, i, j;
  for (eN=0;eN<nElements_global;eN++)
    for (k=0;k<nQuadraturePoints_element;k++)
      for (i=0;i<nDOF_test_element;i++)
        for (j=0;j<nDOF_trial_element;j++)
          for (I=0;I<nSpace;I++)
            grad_v_X_w_dV[eN*nQuadraturePoints_element*nDOF_test_X_trial_element*nSpace +
                          k*nDOF_test_X_trial_element*nSpace+
                          j*nDOF_test_element*nSpace+
                          i*nSpace+
                          I]
              =
              grad_v[eN*nQuadraturePoints_element*nDOF_trial_element*nSpace +
                     k*nDOF_trial_element*nSpace+
                     j*nSpace+
                     I]
              *
              w_dV[eN*nQuadraturePoints_element*nDOF_test_element +
                   k*nDOF_test_element+
                   i];
}

/**
   \brief Calculate  the tensor  product of trial function gradients and test function gradients at the quadrature points.
*/
void calculateGradShape_X_weightedGradShape(proteus_index_t nElements_global,
                                            int nQuadraturePoints_element,
                                            int nDOF_trial_element,
                                            int nDOF_test_element,
                                            int nSpace,
                                            double* grad_v,
                                            double* grad_w_dV,
                                            double* grad_v_X_grad_w_dV)
{
  proteus_index_t nDOF_test_X_trial_element=nDOF_test_element*nDOF_trial_element, nSpace2=nSpace*nSpace;  proteus_index_t I, J; int eN, k, i, j;
  for (eN=0;eN<nElements_global;eN++)
    for (k=0;k<nQuadraturePoints_element;k++)
      for (i=0;i<nDOF_test_element;i++)
        for (j=0;j<nDOF_trial_element;j++)
          for (I=0;I<nSpace;I++)
            for (J=0;J<nSpace;J++)
              grad_v_X_grad_w_dV[eN*nQuadraturePoints_element*nDOF_test_X_trial_element*nSpace2 +
                                 k*nDOF_test_X_trial_element*nSpace2+
                                 j*nDOF_test_element*nSpace2+
                                 i*nSpace2+
                                 I*nSpace+
                                 J]
                =
                grad_v[eN*nQuadraturePoints_element*nDOF_trial_element*nSpace +
                       k*nDOF_trial_element*nSpace+
                       j*nSpace+
                       I]
                *
                grad_w_dV[eN*nQuadraturePoints_element*nDOF_test_element*nSpace +
                          k*nDOF_test_element*nSpace+
                          i*nSpace+
                          J];
}

/**
   \brief Weight the traces of the test function with the element boundary integration weights
*/
void calculateWeightedShapeTrace(proteus_index_t nElements_global,
                                 int nElementBoundaries_element,
                                 int nElementBoundaryQuadraturePoints_elementBoundary,
                                 int nDOF_test_element,
                                 double* dSR,
                                 double* sqrt_det_g,
                                 double* w,
                                 double* w_dS)
{
  proteus_index_t eN, ebN;  int i, k;
  for (eN=0;eN<nElements_global;eN++)
    for (ebN=0;ebN<nElementBoundaries_element;ebN++)
      for (k=0;k<nElementBoundaryQuadraturePoints_elementBoundary;k++)
        for (i=0;i<nDOF_test_element;i++)
          {
            w_dS[eN*nElementBoundaries_element*nElementBoundaryQuadraturePoints_elementBoundary*nDOF_test_element+
                 ebN*nElementBoundaryQuadraturePoints_elementBoundary*nDOF_test_element +
                 k*nDOF_test_element+
                 i]
              =
              w[eN*nElementBoundaries_element*nElementBoundaryQuadraturePoints_elementBoundary*nDOF_test_element+
                ebN*nElementBoundaryQuadraturePoints_elementBoundary*nDOF_test_element +
                k*nDOF_test_element+
                i]
              *
              dSR[k]
              *
              sqrt_det_g[eN*nElementBoundaries_element*nElementBoundaryQuadraturePoints_elementBoundary+
                         ebN*nElementBoundaryQuadraturePoints_elementBoundary+
                         k];
          }
}
void calculateWeightedPiolaShapeTrace(proteus_index_t nElements_global,
                                      int nElementBoundaries_element,
                                      int nElementBoundaryQuadraturePoints_elementBoundary,
                                      int nDOF_test_element,
                                      double* dSR,
                                      double* sqrt_det_g,
                                      double* w,
                                      double* w_dS)
{
  proteus_index_t eN, ebN;  int i, k;
  for (eN=0;eN<nElements_global;eN++)
    for (ebN=0;ebN<nElementBoundaries_element;ebN++)
      for (k=0;k<nElementBoundaryQuadraturePoints_elementBoundary;k++)
        for (i=0;i<nDOF_test_element;i++)
          {
            w_dS[eN*nElementBoundaries_element*nElementBoundaryQuadraturePoints_elementBoundary*nDOF_test_element+
                 ebN*nElementBoundaryQuadraturePoints_elementBoundary*nDOF_test_element +
                 k*nDOF_test_element+
                 i]
              =
              w[eN*nElementBoundaries_element*nElementBoundaryQuadraturePoints_elementBoundary*nDOF_test_element+
                ebN*nElementBoundaryQuadraturePoints_elementBoundary*nDOF_test_element +
                k*nDOF_test_element+
                i]
              *
              dSR[k]
              *
              sqrt_det_g[eN*nElementBoundaries_element*nElementBoundaryQuadraturePoints_elementBoundary+
                         ebN*nElementBoundaryQuadraturePoints_elementBoundary+
                         k];
          }
}

/**
   \brief Calcualte the tensor product of trial and test functions at the quadrature  points
*/
void calculateShape_X_weightedShapeTrace(proteus_index_t nElements_global,
                                         int nElementBoundaries_element,
                                         int nElementBoundaryQuadraturePoints_elementBoundary,
                                         int nDOF_trial_element,
                                         int nDOF_test_element,
                                         double* v,
                                         double* w_dS,
                                         double* v_X_w_dS)
{
  proteus_index_t nDOF_test_X_trial_element=nDOF_test_element*nDOF_trial_element;  proteus_index_t ebN; int eN, k, i, j;
  for (eN=0;eN<nElements_global;eN++)
    for (ebN=0;ebN<nElementBoundaries_element;ebN++)
      for (k=0;k<nElementBoundaryQuadraturePoints_elementBoundary;k++)
        for (i=0;i<nDOF_test_element;i++)
          for (j=0;j<nDOF_trial_element;j++)
            v_X_w_dS[eN*nElementBoundaries_element*nElementBoundaryQuadraturePoints_elementBoundary*nDOF_test_X_trial_element +
                     ebN*nElementBoundaryQuadraturePoints_elementBoundary*nDOF_test_X_trial_element +
                     k*nDOF_test_X_trial_element+
                     j*nDOF_test_element+
                     i]
              =
              v[eN*nElementBoundaries_element*nElementBoundaryQuadraturePoints_elementBoundary*nDOF_trial_element +
                ebN*nElementBoundaryQuadraturePoints_elementBoundary*nDOF_trial_element +
                k*nDOF_trial_element+
                j]
              *
              w_dS[eN*nElementBoundaries_element*nElementBoundaryQuadraturePoints_elementBoundary*nDOF_test_element +
                   ebN*nElementBoundaryQuadraturePoints_elementBoundary*nDOF_test_element +
                   k*nDOF_trial_element+
                   i];
}

/**
   \brief Calculate  the tensor  product of trial function gradients and test functions at the quadrature points.
*/
void calculateGradShape_X_weightedShapeTrace(proteus_index_t nElements_global,
                                        int nElementBoundaries_element,
                                        int nElementBoundaryQuadraturePoints_elementBoundary,
                                        int nDOF_trial_element,
                                        int nDOF_test_element,
                                        int nSpace,
                                        double* grad_v,
                                        double* w_dS,
                                        double* grad_v_X_w_dS)
{
  proteus_index_t nDOF_test_X_trial_element=nDOF_test_element*nDOF_trial_element;  proteus_index_t ebN, I; int eN, k, i, j;
  for (eN=0;eN<nElements_global;eN++)
    for (ebN=0;ebN<nElementBoundaries_element;ebN++)
      for (k=0;k<nElementBoundaryQuadraturePoints_elementBoundary;k++)
        for (i=0;i<nDOF_test_element;i++)
          for (j=0;j<nDOF_trial_element;j++)
            for (I=0;I<nSpace;I++)
              grad_v_X_w_dS[eN*nElementBoundaries_element*nElementBoundaryQuadraturePoints_elementBoundary*nDOF_test_X_trial_element*nSpace +
                            ebN*nElementBoundaryQuadraturePoints_elementBoundary*nDOF_test_X_trial_element*nSpace +
                            k*nDOF_test_X_trial_element*nSpace+
                            j*nDOF_test_element*nSpace+
                            i*nSpace+
                            I]
                =
                grad_v[eN*nElementBoundaries_element*nElementBoundaryQuadraturePoints_elementBoundary*nDOF_trial_element*nSpace +
                       ebN*nElementBoundaryQuadraturePoints_elementBoundary*nDOF_trial_element*nSpace +
                       k*nDOF_trial_element*nSpace+
                       j*nSpace+
                       I]
                *
                w_dS[eN*nElementBoundaries_element*nElementBoundaryQuadraturePoints_elementBoundary*nDOF_test_element +
                     ebN*nElementBoundaryQuadraturePoints_elementBoundary*nDOF_test_element +
                     k*nDOF_test_element+
                     i];
}
/**
   \brief Weight the traces of the test function with the element boundary integration weights
   global exterior boundary version
*/
void calculateWeightedShapeGlobalExteriorTrace(int nElementBoundaryQuadraturePoints_elementBoundary,
                                               int nDOF_test_element,
                                               proteus_index_t nExteriorElementBoundaries_global,
                                               const proteus_index_t* exteriorElementBoundariesArray,
                                               const proteus_index_t* elementBoundaryElementsArray,
                                               const int* elementBoundaryLocalElementBoundariesArray,
                                               double* dSR,
                                               double* sqrt_det_g,
                                               double* w,
                                               double* w_dS)
{
  int ebNE,i,k;
  for (ebNE = 0; ebNE < nExteriorElementBoundaries_global; ebNE++)
    {
      for (k=0;k<nElementBoundaryQuadraturePoints_elementBoundary;k++)
        for (i=0;i<nDOF_test_element;i++)
          {
            w_dS[ebNE*nElementBoundaryQuadraturePoints_elementBoundary*nDOF_test_element +
                 k*nDOF_test_element+
                 i]
              =
              w[ebNE*nElementBoundaryQuadraturePoints_elementBoundary*nDOF_test_element +
                k*nDOF_test_element+
                i]
              *
              dSR[k]
              *
              sqrt_det_g[ebNE*nElementBoundaryQuadraturePoints_elementBoundary+
                         k];
          }
    }
}
/**
   \brief Calcualte the tensor product of trial and test functions at the quadrature  points
   global exterior boundary version
*/
void calculateShape_X_weightedShapeGlobalExteriorTrace(int nElementBoundaryQuadraturePoints_elementBoundary,
                                                       int nDOF_trial_element,
                                                       int nDOF_test_element,
                                                       proteus_index_t nExteriorElementBoundaries_global,
                                                       const proteus_index_t* exteriorElementBoundariesArray,
                                                       const proteus_index_t* elementBoundaryElementsArray,
                                                       const int* elementBoundaryLocalElementBoundariesArray,
                                                       double* v,
                                                       double* w_dS,
                                                       double* v_X_w_dS)
{
  proteus_index_t nDOF_test_X_trial_element=nDOF_test_element*nDOF_trial_element;  int ebNE, k, i, j;
  for (ebNE=0;ebNE<nExteriorElementBoundaries_global;ebNE++)
    for (k=0;k<nElementBoundaryQuadraturePoints_elementBoundary;k++)
      for (i=0;i<nDOF_test_element;i++)
        for (j=0;j<nDOF_trial_element;j++)
          v_X_w_dS[ebNE*nElementBoundaryQuadraturePoints_elementBoundary*nDOF_test_X_trial_element +
                   k*nDOF_test_X_trial_element+
                   j*nDOF_test_element+
                   i]
            =
            v[ebNE*nElementBoundaryQuadraturePoints_elementBoundary*nDOF_trial_element +
              k*nDOF_trial_element+
              j]
            *
            w_dS[ebNE*nElementBoundaryQuadraturePoints_elementBoundary*nDOF_test_element +
                 k*nDOF_trial_element+
                 i];
}

/**
   \brief Calculate  the tensor  product of trial function gradients and test functions at the quadrature points.
*/
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
                                                           double* grad_v_X_w_dS)
{
  proteus_index_t nDOF_test_X_trial_element=nDOF_test_element*nDOF_trial_element;  proteus_index_t I; int ebNE, k, i, j;
  for (ebNE=0;ebNE<nExteriorElementBoundaries_global;ebNE++)
    for (k=0;k<nElementBoundaryQuadraturePoints_elementBoundary;k++)
      for (i=0;i<nDOF_test_element;i++)
        for (j=0;j<nDOF_trial_element;j++)
          for (I=0;I<nSpace;I++)
            grad_v_X_w_dS[ebNE*nElementBoundaryQuadraturePoints_elementBoundary*nDOF_test_X_trial_element*nSpace +
                          k*nDOF_test_X_trial_element*nSpace+
                          j*nDOF_test_element*nSpace+
                          i*nSpace+
                          I]
              =
              grad_v[ebNE*nElementBoundaryQuadraturePoints_elementBoundary*nDOF_trial_element*nSpace +
                     k*nDOF_trial_element*nSpace+
                     j*nSpace+
                     I]
              *
              w_dS[ebNE*nElementBoundaryQuadraturePoints_elementBoundary*nDOF_test_element +
                   k*nDOF_test_element+
                   i];
}


/**
   \brief Calculate the physical space integration weights from the reference element weights and Jacobian determinants.
*/
void calculateIntegrationWeights(proteus_index_t nElements_global,
                                 int nQuadraturePoints_element,
                                 double* abs_det_J,
                                 double* referenceWeights,
                                 double* weights)
{
  proteus_index_t eN;  int k;
  for (eN=0;eN<nElements_global;eN++)
    for (k=0;k<nQuadraturePoints_element;k++)
      weights[eN*nQuadraturePoints_element+
              k]
        =
        abs_det_J[eN*nQuadraturePoints_element+
                  k]
        *
        referenceWeights[k];
}

void calculateElementBoundaryIntegrationWeights(proteus_index_t nElements_global,
                                                int nElementBoundaries_element,
                                                int nQuadraturePoints_elementBoundary,
                                                double* sqrt_det_g,
                                                double* referenceWeights,
                                                double* weights)
{
  proteus_index_t ebN, eN;  int k;
  for (eN=0;eN<nElements_global;eN++)
    for (ebN=0; ebN < nElementBoundaries_element; ebN++)
      for (k=0;k<nQuadraturePoints_elementBoundary;k++)
        weights[eN*nElementBoundaries_element*nQuadraturePoints_elementBoundary+
                ebN*nQuadraturePoints_elementBoundary +
                k]
          =
          sqrt_det_g[eN*nElementBoundaries_element*nQuadraturePoints_elementBoundary+
                     ebN*nQuadraturePoints_elementBoundary +
                     k]
          *
          referenceWeights[k];
}

void calculateGlobalExteriorElementBoundaryIntegrationWeights(int nQuadraturePoints_elementBoundary,
                                                              proteus_index_t nExteriorElementBoundaries_global,
                                                              double* sqrt_det_g,
                                                              double* referenceWeights,
                                                              double* weights)
{
  int ebNE,k;
  for (ebNE=0;ebNE<nExteriorElementBoundaries_global;ebNE++)
    for (k=0;k<nQuadraturePoints_elementBoundary;k++)
      weights[ebNE*nQuadraturePoints_elementBoundary +
              k]
        =
        sqrt_det_g[ebNE*nQuadraturePoints_elementBoundary +
                   k]
        *
        referenceWeights[k];
}

/**
   \brief Calculate the values of a multicomponent finite element function at the quadrature  points from the degrees of freedom and the test function values at the quadrature points
*/
void calculateFiniteElementFunctionValues(proteus_index_t nElements_global,
                                          int nQuadraturePoints_element,
                                          int nDOF_trial_element,
                                          int nComponents,
                                          proteus_index_t* l2g,
                                          double* dof,
                                          double* v,
                                          double* u)
{
  proteus_index_t eN;  int k, j, t;
  memset(u,0,sizeof(double)*nElements_global*nQuadraturePoints_element*nComponents);
  for (eN=0;eN<nElements_global;eN++)
    for  (k=0;k<nQuadraturePoints_element;k++)
      for (j=0;j<nDOF_trial_element;j++)
        for (t=0;t<nComponents;t++)
          u[eN*nQuadraturePoints_element*nComponents+
            k*nComponents+
            t]
            +=
            dof[l2g[eN*nDOF_trial_element+
                    j]*nComponents+
                t]
            *
            v[eN*nQuadraturePoints_element*nDOF_trial_element+
              k*nDOF_trial_element+
              j];
}

/**
   \brief Calculate the gradient values of a multicomponent finite element function at the quadrature  points from the degrees of freedom and the test function gradient values at the quadrature points
*/
void calculateFiniteElementFunctionGradientValues(proteus_index_t nElements_global,
                                                  int nQuadraturePoints_element,
                                                  int nDOF_trial_element,
                                                  int nComponents,
                                                  int nSpace,
                                                  proteus_index_t* l2g,
                                                  double* dof,
                                                  double* grad_v,
                                                  double* grad_u)
{
  proteus_index_t eN, I;  int k, j, t;
  memset(grad_u,0,sizeof(double)*nElements_global*nQuadraturePoints_element*nComponents*nSpace);
  for (eN=0;eN<nElements_global;eN++)
    for  (k=0;k<nQuadraturePoints_element;k++)
      for (j=0;j<nDOF_trial_element;j++)
        for (t=0;t<nComponents;t++)
          for  (I=0;I<nSpace;I++)
            grad_u[eN*nQuadraturePoints_element*nComponents*nSpace+
                   k*nComponents*nSpace+
                   t*nSpace+
                   I]
              +=
              dof[l2g[eN*nDOF_trial_element+
                      j]*nComponents+
                  t]
              *
              grad_v[eN*nQuadraturePoints_element*nDOF_trial_element*nSpace+
                     k*nDOF_trial_element*nSpace+
                     j*nSpace+
                     I];
}

void calculateFiniteElementFunctionHessianValues(proteus_index_t nElements_global,
                                                 int nQuadraturePoints_element,
                                                 int nDOF_trial_element,
                                                 int nComponents,
                                                 int nSpace,
                                                 proteus_index_t* l2g,
                                                 double* dof,
                                                 double* Hessian_v,
                                                 double* Hessian_u)
{
  proteus_index_t nSpace2=nSpace*nSpace;  proteus_index_t I, J; int eN, k, j, t;
  memset(Hessian_u,0,sizeof(double)*nElements_global*nQuadraturePoints_element*nComponents*nSpace2);
  for (eN=0;eN<nElements_global;eN++)
    for  (k=0;k<nQuadraturePoints_element;k++)
      for (j=0;j<nDOF_trial_element;j++)
        for (t=0;t<nComponents;t++)
          for  (I=0;I<nSpace;I++)
            for  (J=0;J<nSpace;J++)
              Hessian_u[eN*nQuadraturePoints_element*nComponents*nSpace2+
                        k*nComponents*nSpace2+
                        t*nSpace2+
                        I*nSpace+
                        J]
                +=
                dof[l2g[eN*nDOF_trial_element+
                        j]*nComponents+
                    t]
                *
                Hessian_v[eN*nQuadraturePoints_element*nDOF_trial_element*nSpace2+
                          k*nDOF_trial_element*nSpace2+
                          j*nSpace2+
                          I*nSpace+
                          J];
}

/**
   \brief  Loop over all the quadrature points and calculate the tensor product of the solution gradient with the test functions.

   @param nElements_global The number of elements in the mesh

   @param nQuadraturePoints_element The number of quadrature points
   per element.

   @param nDOF_test_element The number of degrees of freedom per
   element.

   @param nSpace The number of spatial dimensions.

   @param l2g (nElements_global x nDOF_trial_element) The mapping
   between element degrees of freedom and global degrees of freedom.

   @param dof (nDOF_global). The global degrees of freedom of u.

   @param grad_v_X_grad_w (nElements_global x
   nQuadraturePoints_element x x nDOF_test_element x nSpace x nSpace)
   The tensor product of the solution gradient values and trial
   function gradient values at the quadrature points.

   @param grad_u_X_grad_w (nElements_global x
   nQuadraturePoints_element x x nDOF_test_element x nSpace x nSpace)
   The tensor product of the solution gradient values and trial
   function gradient values at the quadrature points.

   \f[
   u_{e,q} \mapsto \sum_j u_j v_{e,q,j}
   \f]
   \f[
   \nabla u_{e,q} \mapsto \sum_j u_j \nabla v_{e,q,j}
   \f]
   \f[
   (\nabla u \otimes \nabla w)_{e,q,j,i} \mapsto \sum_j u_j (\nabla v \otimes \nabla w)_{e,q,j,i}
   \f]

   where

   \f[
   u_j = u[l2g[e,j]]
   \f]
*/
void calculateFiniteElementFunctionGradientTensorValues(proteus_index_t nElements_global,
                                                        int nQuadraturePoints_element,
                                                        int nDOF_trial_element,
                                                        int nDOF_test_element,
                                                        int nComponents,
                                                        int nSpace,
                                                        proteus_index_t* l2g,
                                                        double* dof,
                                                        double* grad_v_X_grad_w_dV,
                                                        double* grad_u_X_grad_w_dV)
{
  proteus_index_t nSpace2=nSpace*nSpace;  proteus_index_t I, J; int eN, k, j, i, t;
  proteus_index_t nDOF_test_X_trial_element = nDOF_test_element*nDOF_trial_element;
  memset(grad_u_X_grad_w_dV,0,sizeof(double)*nElements_global*nQuadraturePoints_element*nDOF_test_element*nComponents*nSpace2);
  for(eN=0;eN<nElements_global;eN++)
    for(k=0;k<nQuadraturePoints_element;k++)
      for(j=0;j<nDOF_trial_element;j++)
        for(i=0;i<nDOF_test_element;i++)
          for (t=0;t<nComponents;t++)
            for(I=0;I<nSpace;I++)
              for(J=0;J<nSpace;J++)
                grad_u_X_grad_w_dV[eN*nQuadraturePoints_element*nDOF_test_element*nComponents*nSpace2+
                                   k*nDOF_test_element*nComponents*nSpace2 +
                                   i*nComponents*nSpace2 +
                                   t*nSpace2 +
                                   I*nSpace +
                                   J]
                  +=
                  dof[l2g[eN*nDOF_trial_element+
                          j]*nComponents+
                      t]
                  *
                  grad_v_X_grad_w_dV[eN*nQuadraturePoints_element*nDOF_test_X_trial_element*nSpace2 +
                                     k*nDOF_test_X_trial_element*nSpace2 +
                                     j*nDOF_test_element*nSpace2 +
                                     i*nSpace2 +
                                     I*nSpace +
                                     J];
}

/**
   \brief Calculate the values of a multi-component finite element function at element boundary quadrature points from the degrees of freedom and the trace of the trial functions at the element boundary quadrature points
*/
void calculateFiniteElementFunctionValuesTrace(proteus_index_t nElements_global,
                                               int nElementBoundaries_element,
                                               int nQuadraturePoints_elementBoundary,
                                               int nDOF_trial_element,
                                               int nComponents,
                                               proteus_index_t* l2g,
                                               double* dof,
                                               double* v,
                                               double* u)
{
  proteus_index_t eN, ebN;  int k, j, t;
  memset(u,0,sizeof(double)*nElements_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nComponents);
  for(eN=0;eN<nElements_global;eN++)
    for(ebN=0;ebN<nElementBoundaries_element;ebN++)
      for(k=0;k<nQuadraturePoints_elementBoundary;k++)
        for(j=0;j<nDOF_trial_element;j++)
          for(t=0;t<nComponents;t++)
            u[eN*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nComponents+
              ebN*nQuadraturePoints_elementBoundary*nComponents+
              k*nComponents+
              t]
              +=
              dof[l2g[eN*nDOF_trial_element+j]*nComponents+
                  t]
              *
              v[eN*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                ebN*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                k*nDOF_trial_element+
                j];
}


/**
   \brief Calculate the gradients of a multi-component finite element function at the element boundary quadrature points from the degress of freedom and the trace of the trial functions at the element boundary quadrature  points.
*/
void calculateFiniteElementFunctionGradientValuesTrace(proteus_index_t nElements_global,
                                                       int nElementBoundaries_element,
                                                       int nQuadraturePoints_elementBoundary,
                                                       int nDOF_trial_element,
                                                       int nComponents,
                                                       int nSpace,
                                                       proteus_index_t* l2g,
                                                       double* dof,
                                                       double* grad_v,
                                                       double* grad_u)
{
  proteus_index_t eN, ebN, I;  int k, j, t;
  memset(grad_u,0,sizeof(double)*nElements_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nComponents*nSpace);
  for(eN=0;eN<nElements_global;eN++)
    for(ebN=0;ebN<nElementBoundaries_element;ebN++)
      for(k=0;k<nQuadraturePoints_elementBoundary;k++)
        for(j=0;j<nDOF_trial_element;j++)
          for(t=0;t<nComponents;t++)
            for(I=0;I<nSpace;I++)
              grad_u[eN*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nComponents*nSpace+
                     ebN*nQuadraturePoints_elementBoundary*nComponents*nSpace+
                     k*nComponents*nSpace+
                     t*nSpace+
                     I]
                +=
                dof[l2g[eN*nDOF_trial_element+j]*nComponents+
                    t]
                *
                grad_v[eN*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_trial_element*nSpace+
                       ebN*nQuadraturePoints_elementBoundary*nDOF_trial_element*nSpace+
                       k*nDOF_trial_element*nSpace+
                       j*nSpace+
                       I];
}
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
                                                             double* u)
{
  proteus_index_t eN, ebN;  int ebNE, ebN_local, k, j, t;
  memset(u,0,sizeof(double)*nExteriorElementBoundaries_global*nQuadraturePoints_elementBoundary*nComponents);
  for (ebNE = 0; ebNE < nExteriorElementBoundaries_global; ebNE++)
    {
      ebN = exteriorElementBoundariesArray[ebNE];
      eN  = elementBoundaryElementsArray[ebN*2+0];
      ebN_local = elementBoundaryLocalElementBoundariesArray[ebN*2+0];
      for(k=0;k<nQuadraturePoints_elementBoundary;k++)
        for(j=0;j<nDOF_trial_element;j++)
          for(t=0;t<nComponents;t++)
            u[ebNE*nQuadraturePoints_elementBoundary*nComponents+
              k*nComponents+
              t]
              +=
              dof[l2g[eN*nDOF_trial_element+j]*nComponents+
                  t]
              *
              v[ebNE*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                k*nDOF_trial_element+
                j];
    }
}

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
                                                                     double* grad_u)
{
  proteus_index_t eN, ebN, I;  int ebNE, ebN_local, k, j, t;
  memset(grad_u,0,sizeof(double)*nExteriorElementBoundaries_global*nQuadraturePoints_elementBoundary*nComponents*nSpace);
  for (ebNE = 0; ebNE < nExteriorElementBoundaries_global; ebNE++)
    {
      ebN = exteriorElementBoundariesArray[ebNE];
      eN  = elementBoundaryElementsArray[ebN*2+0];
      ebN_local = elementBoundaryLocalElementBoundariesArray[ebN*2+0];
      for(k=0;k<nQuadraturePoints_elementBoundary;k++)
        for(j=0;j<nDOF_trial_element;j++)
          for(t=0;t<nComponents;t++)
            for(I=0;I<nSpace;I++)
              grad_u[ebNE*nQuadraturePoints_elementBoundary*nComponents*nSpace+
                     k*nComponents*nSpace+
                     t*nSpace+
                     I]
                +=
                dof[l2g[eN*nDOF_trial_element+j]*nComponents+
                    t]
                *
                grad_v[ebNE*nQuadraturePoints_elementBoundary*nDOF_trial_element*nSpace+
                       k*nDOF_trial_element*nSpace+
                       j*nSpace+
                       I];
    }
}


/**
   \brief Calculate the total (advective + diffusive) flow  velocity.
*/
void calculateFlowVelocity(proteus_index_t nElements_global,
                           int nQuadraturePoints_element,
                           int nSpace,
                           double* f,
                           double* a,
                           double* grad_phi,
                           double* v)
{
  proteus_index_t nSpace2=nSpace*nSpace;  proteus_index_t I, J; int eN, k;
  for  (eN=0;eN<nElements_global;eN++)
    for (k=0;k<nQuadraturePoints_element;k++)
      for (I=0;I<nSpace;I++)
        {
          v[eN*nQuadraturePoints_element*nSpace+
            k*nSpace+
            I]
            =
            f[eN*nQuadraturePoints_element*nSpace+
              k*nSpace+
              I];
          for (J=0;J<nSpace;J++)
            v[eN*nQuadraturePoints_element*nSpace+
              k*nSpace+
              I]
              -=
              a[eN*nQuadraturePoints_element*nSpace2+
                k*nSpace2+
                I*nSpace+
                J]
              *
              grad_phi[eN*nQuadraturePoints_element*nSpace+
                       k*nSpace+
                       J];
        }
}

/**
   \brief Update a single  element of the Jacobian
*/
void updateAddJacobian_CSR(int jacIndex,
                           double val,
                           double* jac)
{
  jac[jacIndex] += val;
}

/**
   \brief  Set all the Jacobian entries  to 0.0
*/
void zeroJacobian_CSR(proteus_index_t nNonzeros,
                      double* jac)
{
  memset(jac,0,sizeof(double)*nNonzeros);
}

void calculateInteriorElementBoundaryVelocities(proteus_index_t nInteriorElementBoundaries_global,
                                                int nElementBoundaries_element,
                                                int nQuadraturePoints_elementBoundary,
                                                int nSpace,
                                                proteus_index_t* interiorElementBoundaries,
                                                proteus_index_t* elementBoundaryElements,
                                                int* elementBoundaryLocalElementBoundaries,
                                                double* m,
                                                double* a,
                                                double* grad_phi,
                                                double* f,
                                                double* vAverage,
                                                double* vJump,
                                                double* mAverage,
                                                double* mJump)
{
  proteus_index_t nSpace2=nSpace*nSpace;  proteus_index_t ebN, left_eN_global, right_eN_global, I, J; int ebNI, left_ebN_element, right_ebN_element, k;
  register double vLeft,vRight,mLeft,mRight;
  for(ebNI=0;ebNI<nInteriorElementBoundaries_global;ebNI++)
    {
      ebN = interiorElementBoundaries[ebNI];
      left_eN_global = elementBoundaryElements[ebN*2+0];
      right_eN_global = elementBoundaryElements[ebN*2+1];
      left_ebN_element = elementBoundaryLocalElementBoundaries[ebN*2+0];
      right_ebN_element = elementBoundaryLocalElementBoundaries[ebN*2+1];
      for(k=0;k<nQuadraturePoints_elementBoundary;k++)
        {
          mLeft
            =
            m[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary+
              left_ebN_element*nQuadraturePoints_elementBoundary+
              k];
          mRight
            =
            m[right_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary+
              right_ebN_element*nQuadraturePoints_elementBoundary+
              k];
          for (I=0;I<nSpace;I++)
            {
              vLeft
                =
                f[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                  left_ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                  k*nSpace+
                  I];
              vRight
                =
                f[right_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                  right_ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                  k*nSpace+
                  I];
              for (J=0;J<nSpace;J++)
                {
                  vLeft
                    -=
                    a[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace2+
                      left_ebN_element*nQuadraturePoints_elementBoundary*nSpace2+
                      k*nSpace2+
                      I*nSpace+
                      J]
                    *
                    grad_phi[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                             left_ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                             k*nSpace+
                             J];
                  vRight -=
                    a[right_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace2+
                      right_ebN_element*nQuadraturePoints_elementBoundary*nSpace2+
                      k*nSpace2+
                      I*nSpace+
                      J]
                    *
                    grad_phi[right_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                             right_ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                             k*nSpace+
                             J];
                }
              vAverage[ebN*nQuadraturePoints_elementBoundary*nSpace+
                       k*nSpace+
                       I] = 0.5*(vLeft + vRight);
              vJump[ebN*nQuadraturePoints_elementBoundary*nSpace+
                    k*nSpace+
                    I] = (vLeft - vRight);
            }
          mAverage[ebN*nQuadraturePoints_elementBoundary+
                   k] = 0.5*(mLeft+mRight);
          mJump[ebN*nQuadraturePoints_elementBoundary+
                k] = mLeft - mRight;
        }
    }
}

void calculateExteriorElementBoundaryVelocities(proteus_index_t nExteriorElementBoundaries_global,
                                                int nElementBoundaries_element,
                                                int nQuadraturePoints_elementBoundary,
                                                int nSpace,
                                                proteus_index_t* exteriorElementBoundaries,
                                                proteus_index_t* elementBoundaryElements,
                                                int* elementBoundaryLocalElementBoundaries,
                                                double* m,
                                                double* a,
                                                double* grad_phi,
                                                double* f,
                                                double* vAverage,
                                                double* vJump,
                                                double* mAverage,
                                                double* mJump)
{
  proteus_index_t nSpace2=nSpace*nSpace;  proteus_index_t ebN, left_eN_global, I, J; int ebNE, left_ebN_element, k;
  register double vLeft,mLeft;
  for(ebNE=0;ebNE<nExteriorElementBoundaries_global;ebNE++)
    {
      ebN = exteriorElementBoundaries[ebNE];
      left_eN_global = elementBoundaryElements[ebN*2+0];
      left_ebN_element = elementBoundaryLocalElementBoundaries[ebN*2+0];
      for(k=0;k<nQuadraturePoints_elementBoundary;k++)
        {
          mLeft
            =
            m[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary+
              left_ebN_element*nQuadraturePoints_elementBoundary+
              k];
          for (I=0;I<nSpace;I++)
            {
              vLeft
                =
                f[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                  left_ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                  k*nSpace+
                  I];
              for (J=0;J<nSpace;J++)
                {
                  vLeft
                    -=
                    a[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace2+
                      left_ebN_element*nQuadraturePoints_elementBoundary*nSpace2+
                      k*nSpace2+
                      I*nSpace+
                      J]
                    *
                    grad_phi[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                             left_ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                             k*nSpace+
                             J];
                }
              vAverage[ebN*nQuadraturePoints_elementBoundary*nSpace+
                       k*nSpace+
                       I] = vLeft;
              vJump[ebN*nQuadraturePoints_elementBoundary*nSpace+
                    k*nSpace+
                    I] = vLeft;
            }
          mAverage[ebN*nQuadraturePoints_elementBoundary+
                   k] = mLeft;
          mJump[ebN*nQuadraturePoints_elementBoundary+
                k] = mLeft;
        }
    }
}

/*mwf hack
  load in boundary condition velocity values directy into velocity vector
*/
void setExteriorGlobalElementBoundaryVelocityValues(int updateFluxValues,
                                                    proteus_index_t nExteriorElementBoundaries_global,
                                                    int nQuadraturePoints_elementBoundary,
                                                    int nSpace,
                                                    proteus_index_t* exteriorElementBoundaries,
                                                    proteus_index_t* elementBoundaryElements,
                                                    int* elementBoundaryLocalElementBoundaries,
                                                    double* n,
                                                    double* vn_in,
                                                    double* v_out)
{
  proteus_index_t ebN, I;  int ebNE, k;
  double multiple = 0.0;
  if (updateFluxValues > 0)
    multiple = 1.0;
  /*mwf debug*/
  printf("setExteriorGlobalElementBoundaryVelocityValues update= %d nExtElmBndr_global= %d nQuadElmBndr= %d nSpace=%d \n",
         updateFluxValues,nExteriorElementBoundaries_global,nQuadraturePoints_elementBoundary,nSpace);
  for(ebNE=0;ebNE<nExteriorElementBoundaries_global;ebNE++)
    {
      ebN = exteriorElementBoundaries[ebNE];
      for(k=0;k<nQuadraturePoints_elementBoundary;k++)
        for (I=0;I<nSpace;I++)
          {
            v_out[ebN*nQuadraturePoints_elementBoundary*nSpace+
                 k*nSpace+
                 I] =
              multiple*v_out[ebN*nQuadraturePoints_elementBoundary*nSpace+
                             k*nSpace+
                             I]
              +
              n[ebN*nQuadraturePoints_elementBoundary*nSpace+
                k*nSpace+
                I]
              *
              vn_in[ebN*nQuadraturePoints_elementBoundary+
                    k];

            /*mwf debug*/
            printf("setExtGlobElmVel ebN=%d k=%d I=%d vn_in=%g v_out=%g \n",
                   ebN,k,I,
                   vn_in[ebN*nQuadraturePoints_elementBoundary+
                                 k],
                   v_out[ebN*nQuadraturePoints_elementBoundary*nSpace+
                         k*nSpace+
                         I]);

          }
    }

}


/**
   \brief Calculate the Peclet and Courant-Friedrichs-Lewy numbers for the scalar advection-diffusion-reaction equation
*/
void calculateDimensionlessNumbersADR(proteus_index_t nElements_global,
                                      int nQuadraturePoints_element,
                                      int nSpace,
                                      int computeDiffusiveTimeStepLimit,
                                      double* elementDiameter,
                                      double* df,
                                      double* a,
                                      double* dphi,
                                      double* dr,
                                      double* dmt,
                                      double* pe,
                                      double* cfl)
{
  proteus_index_t nSpace2=nSpace*nSpace;  proteus_index_t I; int eN, k;
  double h,Vlin,Alin,A_II,num,den,cfl1,cfl2;
  for(eN=0;eN<nElements_global;eN++)
    {
      h = elementDiameter[eN];
      for (k=0;k<nQuadraturePoints_element;k++)
        {
          Vlin = 0.0;
          Alin = 0.0;
          for(I=0;I<nSpace;I++)
            Vlin
              +=
              df[eN*nQuadraturePoints_element*nSpace +
                 k*nSpace +
                 I]
              *
              df[eN*nQuadraturePoints_element*nSpace +
                 k*nSpace +
                 I];
          Vlin = sqrt(Vlin);
          /*max diagonal entry for A. This choice needs to be looked into*/
          for(I=0;I<nSpace;I++)
            {
              A_II = a[eN*nQuadraturePoints_element*nSpace2 +
                       k*nSpace2 +
                       I*nSpace +
                       I];
              Alin = (A_II > Alin) ? A_II : Alin;
            }
          Alin*=dphi[eN*nQuadraturePoints_element +
                     k];
          cfl1 = Vlin/h;
          cfl2 = Alin/(h*h);
          if (computeDiffusiveTimeStepLimit)
            cfl[eN*nQuadraturePoints_element +
                k] = cfl2;
          else
            cfl[eN*nQuadraturePoints_element +
                k] = cfl1;
          num = 0.5*Vlin*h + 1.0e-8;
          den = Alin + num*1.0e-8;
          pe[eN*nQuadraturePoints_element + k] = num/den;
        }
    }
}

void calculateDimensionlessNumbersADR_sd(proteus_index_t nElements_global,
                                         int nQuadraturePoints_element,
                                         int nSpace,
                                         int computeDiffusiveTimeStepLimit,
                                         proteus_index_t* rowptr,
                                         proteus_index_t* colind,
                                         double* elementDiameter,
                                         double* df,
                                         double* a,
                                         double* dphi,
                                         double* dr,
                                         double* dmt,
                                         double* pe,
                                         double* cfl)
{
  proteus_index_t m, nnz=rowptr[nSpace];  proteus_index_t I; int eN, k;
  double h,Vlin,Alin,A_II,num,den,cfl1,cfl2;
  for(eN=0;eN<nElements_global;eN++)
    {
      h = elementDiameter[eN];
      for (k=0;k<nQuadraturePoints_element;k++)
        {
          Vlin = 0.0;
          Alin = 0.0;
          for(I=0;I<nSpace;I++)
            {
              Vlin
                +=
                df[eN*nQuadraturePoints_element*nSpace +
                   k*nSpace +
                   I]
                *
                df[eN*nQuadraturePoints_element*nSpace +
                   k*nSpace +
                   I];
            }
          Vlin = sqrt(Vlin);
          /*max diagonal entry for A. This choice needs to be looked into*/
          for(I=0;I<nSpace;I++)
            {
              for(m=rowptr[I];m<rowptr[I+1];m++)
                {
                  if (I == colind[m])
                    {
                      A_II = a[eN*nQuadraturePoints_element*nnz+
                               k*nnz+
                               m];
                      Alin = (A_II > Alin) ? A_II : Alin;
                    }
                }
            }
          Alin*=dphi[eN*nQuadraturePoints_element +
                     k];
          cfl1 = Vlin/h;
          cfl2 = Alin/(h*h);
          if (computeDiffusiveTimeStepLimit)
            cfl[eN*nQuadraturePoints_element +
                k] = cfl2;
          else
            cfl[eN*nQuadraturePoints_element +
                k] = cfl1;
          num = 0.5*Vlin*h + 1.0e-8;
          den = Alin + num*1.0e-8;
          pe[eN*nQuadraturePoints_element + k] = num/den;
        }
    }
}

/*just compute CFL*/
void calculateCFLADR(proteus_index_t nElements_global,
                     int nQuadraturePoints_element,
                     int nSpace,
                     double* elementDiameter,
                     double* dm,
                     double* df,
                     double* cfl)
{
  proteus_index_t eN, I;  int k;
  double h,Vlin,dmdu,cfl1;
  for(eN=0;eN<nElements_global;eN++)
    {
      h = elementDiameter[eN];
      for (k=0;k<nQuadraturePoints_element;k++)
        {
          dmdu = dm[eN*nQuadraturePoints_element + k];
          Vlin = 0.0;
          for(I=0;I<nSpace;I++)
            Vlin
              +=
              df[eN*nQuadraturePoints_element*nSpace +
                 k*nSpace +
                 I]
              *
              df[eN*nQuadraturePoints_element*nSpace +
                 k*nSpace +
                 I];
          Vlin = sqrt(Vlin)/(dmdu+1.0e-10);
          cfl1 = Vlin/h;
          cfl[eN*nQuadraturePoints_element +
              k] = cfl1;

        }
    }
}
/*not really likely but in case have two separate characteristic speeds to add?*/
void calculateCFLADR2speeds(proteus_index_t nElements_global,
                            int nQuadraturePoints_element,
                            int nSpace,
                            double* elementDiameter,
                            double* dm,
                            double* df1,
                            double* df2,
                            double* cfl)
{
  proteus_index_t eN, I;  int k;
  double h,Vlin,dmdu,cfl1;
  for(eN=0;eN<nElements_global;eN++)
    {
      h = elementDiameter[eN];
      for (k=0;k<nQuadraturePoints_element;k++)
        {
          dmdu = dm[eN*nQuadraturePoints_element + k];
          Vlin = 0.0;
          for(I=0;I<nSpace;I++)
            Vlin
              +=
              df1[eN*nQuadraturePoints_element*nSpace +
                 k*nSpace +
                 I]
              *
              df1[eN*nQuadraturePoints_element*nSpace +
                  k*nSpace +
                  I]
              +
              df2[eN*nQuadraturePoints_element*nSpace +
                  k*nSpace +
                  I]
              *
              df2[eN*nQuadraturePoints_element*nSpace +
                  k*nSpace +
                  I];

          Vlin = sqrt(Vlin)/(dmdu+1.0e-10);
          cfl1 = Vlin/h;
          cfl[eN*nQuadraturePoints_element +
              k] = cfl1;

        }
    }
}


/**
   \brief Calculate the diffusive flux at interior element boundary quadrature points
*/
void updateInteriorElementBoundaryDiffusiveVelocity(proteus_index_t nInteriorElementBoundaries_global,
                                                    int nElementBoundaries_element,
                                                    int nQuadraturePoints_elementBoundary,
                                                    int nSpace,
                                                    proteus_index_t* interiorElementBoundaries,
                                                    proteus_index_t* elementBoundaryElements,
                                                    int* elementBoundaryLocalElementBoundaries,
                                                    double* a,
                                                    double* grad_phi,
                                                    double* velocity)
{
  proteus_index_t nSpace2=nSpace*nSpace;  proteus_index_t ebN, left_eN_global, right_eN_global, J, I; int ebNI, left_ebN_element, right_ebN_element, k;
  for(ebNI=0;ebNI<nInteriorElementBoundaries_global;ebNI++)
    {
      ebN = interiorElementBoundaries[ebNI];
      left_eN_global = elementBoundaryElements[ebN*2+0];
      right_eN_global = elementBoundaryElements[ebN*2+1];
      left_ebN_element = elementBoundaryLocalElementBoundaries[ebN*2+0];
      right_ebN_element = elementBoundaryLocalElementBoundaries[ebN*2+1];
      for(k=0;k<nQuadraturePoints_elementBoundary;k++)
        for(I=0;I<nSpace;I++)
          for(J=0;J<nSpace;J++)
            {
              velocity[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                       left_ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                       k*nSpace+I]
                -=
                a[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace2+
                  left_ebN_element*nQuadraturePoints_elementBoundary*nSpace2+
                  k*nSpace2+
                  I*nSpace+
                  J]
                *
                grad_phi[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                         left_ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                         k*nSpace+J];
              velocity[right_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                       right_ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                       k*nSpace+I]
                -=
                a[right_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace2+
                  right_ebN_element*nQuadraturePoints_elementBoundary*nSpace2+
                  k*nSpace2+
                  I*nSpace+
                  J]
                *
                grad_phi[right_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                         right_ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                         k*nSpace+J];
            }
    }
}

void updateInteriorElementBoundaryDiffusiveVelocity_sd(proteus_index_t nInteriorElementBoundaries_global,
                                                       int nElementBoundaries_element,
                                                       int nQuadraturePoints_elementBoundary,
                                                       int nSpace,
                                                       proteus_index_t* rowptr,
                                                       proteus_index_t* colind,
                                                       proteus_index_t* interiorElementBoundaries,
                                                       proteus_index_t* elementBoundaryElements,
                                                       int* elementBoundaryLocalElementBoundaries,
                                                       double* a,
                                                       double* grad_phi,
                                                       double* velocity)
{
  proteus_index_t m, nnz=rowptr[nSpace];  proteus_index_t ebN, left_eN_global, right_eN_global, I; int ebNI, left_ebN_element, right_ebN_element, k;
  for(ebNI=0;ebNI<nInteriorElementBoundaries_global;ebNI++)
    {
      ebN = interiorElementBoundaries[ebNI];
      left_eN_global = elementBoundaryElements[ebN*2+0];
      right_eN_global = elementBoundaryElements[ebN*2+1];
      left_ebN_element = elementBoundaryLocalElementBoundaries[ebN*2+0];
      right_ebN_element = elementBoundaryLocalElementBoundaries[ebN*2+1];
      for(k=0;k<nQuadraturePoints_elementBoundary;k++)
        for(I=0;I<nSpace;I++)
          for(m=rowptr[I];m<rowptr[I+1];m++)
            {
              velocity[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                       left_ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                       k*nSpace+I]
                -=
                a[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nnz+
                  left_ebN_element*nQuadraturePoints_elementBoundary*nnz+
                  k*nnz+
                  m]
                *
                grad_phi[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                         left_ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                         k*nSpace+colind[m]];
              velocity[right_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                       right_ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                       k*nSpace+I]
                -=
                a[right_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nnz+
                  right_ebN_element*nQuadraturePoints_elementBoundary*nnz+
                  k*nnz+
                  m]
                *
                grad_phi[right_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                         right_ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                         k*nSpace+colind[m]];
            }
    }
}

/**
   \brief Calculate the diffusive flux at exterior element boundary quadrature points
*/
void updateExteriorElementBoundaryDiffusiveVelocity(proteus_index_t nExteriorElementBoundaries_global,
                                                    int nElementBoundaries_element,
                                                    int nQuadraturePoints_elementBoundary,
                                                    int nSpace,
                                                    proteus_index_t* exteriorElementBoundaries,
                                                    proteus_index_t* elementBoundaryElements,
                                                    int* elementBoundaryLocalElementBoundaries,
                                                    double* a,
                                                    double* grad_phi,
                                                    double* velocity)
{
  proteus_index_t nSpace2=nSpace*nSpace;  proteus_index_t ebN, eN_global, J, I; int ebNE, ebN_element, k;
  for(ebNE=0;ebNE<nExteriorElementBoundaries_global;ebNE++)
    {
      ebN = exteriorElementBoundaries[ebNE];
      eN_global = elementBoundaryElements[ebN*2+0];
      ebN_element = elementBoundaryLocalElementBoundaries[ebN*2+0];
      for(k=0;k<nQuadraturePoints_elementBoundary;k++)
        for(I=0;I<nSpace;I++)
          {
            for(J=0;J<nSpace;J++)
              {
                velocity[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                         ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                         k*nSpace+I]
                  -=
                  a[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace2+
                    ebN_element*nQuadraturePoints_elementBoundary*nSpace2+
                    k*nSpace2+
                    I*nSpace+
                    J]
                  *
                  grad_phi[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                           ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                           k*nSpace+J];
              }
          }/*I*/
    }
}

void updateExteriorElementBoundaryDiffusiveVelocity_sd(proteus_index_t nExteriorElementBoundaries_global,
                                                       int nElementBoundaries_element,
                                                       int nQuadraturePoints_elementBoundary,
                                                       int nSpace,
                                                       proteus_index_t* rowptr,
                                                       proteus_index_t* colind,
                                                       proteus_index_t* exteriorElementBoundaries,
                                                       proteus_index_t* elementBoundaryElements,
                                                       int* elementBoundaryLocalElementBoundaries,
                                                       double* a,
                                                       double* grad_phi,
                                                       double* velocity)
{
  proteus_index_t m, nnz=rowptr[nSpace];  proteus_index_t ebN, eN_global, I; int ebNE, ebN_element, k;
  for(ebNE=0;ebNE<nExteriorElementBoundaries_global;ebNE++)
    {
      ebN = exteriorElementBoundaries[ebNE];
      eN_global = elementBoundaryElements[ebN*2+0];
      ebN_element = elementBoundaryLocalElementBoundaries[ebN*2+0];
      for(k=0;k<nQuadraturePoints_elementBoundary;k++)
        for(I=0;I<nSpace;I++)
          {
            for(m=rowptr[I];m<rowptr[I+1];m++)
              {
                velocity[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                         ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                         k*nSpace+I]
                  -=
                  a[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nnz+
                    ebN_element*nQuadraturePoints_elementBoundary*nnz+
                    k*nnz+
                    m]
                  *
                  grad_phi[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                           ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                           k*nSpace+colind[m]];
              }
          }/*I*/
    }
}

/**
   \brief Calculate the diffusive flux at exterior element boundary quadrature points
*/
void updateGlobalExteriorElementBoundaryDiffusiveVelocity(proteus_index_t nExteriorElementBoundaries_global,
                                                          int nQuadraturePoints_elementBoundary,
                                                          int nSpace,
                                                          proteus_index_t* exteriorElementBoundaries,
                                                          proteus_index_t* elementBoundaryElements,
                                                          int* elementBoundaryLocalElementBoundaries,
                                                          double* a,
                                                          double* grad_phi,
                                                          double* velocity)
{
  proteus_index_t nSpace2=nSpace*nSpace;  proteus_index_t J, I; int ebNE, k;
  for(ebNE=0;ebNE<nExteriorElementBoundaries_global;ebNE++)
    {
      for(k=0;k<nQuadraturePoints_elementBoundary;k++)
        for(I=0;I<nSpace;I++)
          {
            for(J=0;J<nSpace;J++)
              {
                velocity[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                         k*nSpace+I]
                  -=
                  a[ebNE*nQuadraturePoints_elementBoundary*nSpace2+
                    k*nSpace2+
                    I*nSpace+
                    J]
                  *
                  grad_phi[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                           k*nSpace+J];
              }
          }/*I*/
    }
}

void updateGlobalExteriorElementBoundaryDiffusiveVelocity_sd(proteus_index_t nExteriorElementBoundaries_global,
                                                             int nQuadraturePoints_elementBoundary,
                                                             int nSpace,
                                                             proteus_index_t* rowptr,
                                                             proteus_index_t* colind,
                                                             proteus_index_t* exteriorElementBoundaries,
                                                             proteus_index_t* elementBoundaryElements,
                                                             int* elementBoundaryLocalElementBoundaries,
                                                             double* a,
                                                             double* grad_phi,
                                                             double* velocity)
{
  proteus_index_t m, nnz=rowptr[nSpace];  proteus_index_t I; int ebNE, k;
  for(ebNE=0;ebNE<nExteriorElementBoundaries_global;ebNE++)
    {
      for(k=0;k<nQuadraturePoints_elementBoundary;k++)
        for(I=0;I<nSpace;I++)
          {
            for(m=rowptr[I];m<rowptr[I+1];m++)
              {
                velocity[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                         k*nSpace+I]
                  -=
                  a[ebNE*nQuadraturePoints_elementBoundary*nnz+
                    k*nnz+
                    m]
                  *
                  grad_phi[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                           k*nSpace+colind[m]];
              }
          }/*I*/
    }
}


/**
   \brief Calculate the advective flux at at interior element boundaries
*/
void updateInteriorElementBoundaryAdvectiveVelocity(proteus_index_t nInteriorElementBoundaries_global,
                                                    int nElementBoundaries_element,
                                                    int nQuadraturePoints_elementBoundary,
                                                    int nSpace,
                                                    proteus_index_t* interiorElementBoundaries,
                                                    proteus_index_t* elementBoundaryElements,
                                                    int* elementBoundaryLocalElementBoundaries,
                                                    double* f,
                                                    double* velocity)
{
  proteus_index_t ebN, left_eN_global, right_eN_global, J;  int ebNI, left_ebN_element, right_ebN_element, k;
  for(ebNI=0;ebNI<nInteriorElementBoundaries_global;ebNI++)
    {
      ebN = interiorElementBoundaries[ebNI];
      left_eN_global = elementBoundaryElements[ebN*2+0];
      right_eN_global = elementBoundaryElements[ebN*2+1];
      left_ebN_element = elementBoundaryLocalElementBoundaries[ebN*2+0];
      right_ebN_element = elementBoundaryLocalElementBoundaries[ebN*2+1];
      for(k=0;k<nQuadraturePoints_elementBoundary;k++)
        for (J=0;J<nSpace;J++)
          {
            velocity[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                     left_ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                     k*nSpace+
                     J]
              +=
              f[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                left_ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                k*nSpace+
                J];
            velocity[right_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                     right_ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                     k*nSpace+
                     J]
              +=
              f[right_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                right_ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                k*nSpace+
                J];
          }
    }
}


/**
   \brief Update the advective flux at exterior element boundaries.
*/
void updateExteriorElementBoundaryAdvectiveVelocity(proteus_index_t nExteriorElementBoundaries_global,
                                                    int nElementBoundaries_element,
                                                    int nQuadraturePoints_elementBoundary,
                                                    int nSpace,
                                                    proteus_index_t* exteriorElementBoundaries,
                                                    proteus_index_t* elementBoundaryElements,
                                                    int* elementBoundaryLocalElementBoundaries,
                                                    double* f,
                                                    double* velocity)
{
  proteus_index_t ebN, eN_global, J;  int ebNE, ebN_element, k;
  for(ebNE=0;ebNE<nExteriorElementBoundaries_global;ebNE++)
    {
      ebN = exteriorElementBoundaries[ebNE];
      eN_global = elementBoundaryElements[ebN*2+0];
      ebN_element = elementBoundaryLocalElementBoundaries[ebN*2+0];
      for(k=0;k<nQuadraturePoints_elementBoundary;k++)
        for(J=0;J<nSpace;J++)
          {
            velocity[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                     ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                     k*nSpace+
                     J]
              +=
              f[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                k*nSpace+
                J];
          }
    }
}
/**
   \brief Update the advective flux at exterior element boundaries.
*/
void updateGlobalExteriorElementBoundaryAdvectiveVelocity(proteus_index_t nExteriorElementBoundaries_global,
                                                          int nQuadraturePoints_elementBoundary,
                                                          int nSpace,
                                                          proteus_index_t* exteriorElementBoundaries,
                                                          proteus_index_t* elementBoundaryElements,
                                                          int* elementBoundaryLocalElementBoundaries,
                                                          double* f,
                                                          double* velocity)
{
  proteus_index_t J;  int ebNE, k;
  for(ebNE=0;ebNE<nExteriorElementBoundaries_global;ebNE++)
    {
      for(k=0;k<nQuadraturePoints_elementBoundary;k++)
        for(J=0;J<nSpace;J++)
          {
            velocity[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                     k*nSpace+
                     J]
              +=
              f[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                k*nSpace+
                J];
          }
    }
}

/**
   \brief Calculate the velocity from shock capturing at interior element boundary quadrature points
*/
void updateInteriorElementBoundaryShockCapturingVelocity(proteus_index_t nInteriorElementBoundaries_global,
                                                         int nElementBoundaries_element,
                                                         int nQuadraturePoints_elementBoundary,
                                                         int nQuadraturePoints_element,
                                                         int nSpace,
                                                         proteus_index_t* interiorElementBoundaries,
                                                         proteus_index_t* elementBoundaryElements,
                                                         int* elementBoundaryLocalElementBoundaries,
                                                         double* numDiff,
                                                         double* grad_u,
                                                         double* velocity)
{
  proteus_index_t ebN, left_eN_global, right_eN_global, I;  int ebNI, left_ebN_element, right_ebN_element, k;
  double numDiffAvg_left,numDiffAvg_right;
  for(ebNI=0;ebNI<nInteriorElementBoundaries_global;ebNI++)
    {
      ebN = interiorElementBoundaries[ebNI];
      left_eN_global = elementBoundaryElements[ebN*2+0];
      right_eN_global = elementBoundaryElements[ebN*2+1];
      left_ebN_element = elementBoundaryLocalElementBoundaries[ebN*2+0];
      right_ebN_element = elementBoundaryLocalElementBoundaries[ebN*2+1];
      /*get scalar numDiff coefficient from the left and right assuming constant*/
      numDiffAvg_left  = numDiff[left_eN_global*nQuadraturePoints_element+0];
      numDiffAvg_right = numDiff[right_eN_global*nQuadraturePoints_element+0];
      for(k=0;k<nQuadraturePoints_elementBoundary;k++)
        for(I=0;I<nSpace;I++)
          {
            velocity[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                     left_ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                     k*nSpace+I]
              -=
              numDiffAvg_left
              *
              grad_u[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                     left_ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                     k*nSpace+I];
            velocity[right_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                     right_ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                     k*nSpace+I]
              -=
                numDiffAvg_right
                *
                grad_u[right_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                       right_ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                       k*nSpace+I];
          }
    }
}

/**
   \brief Calculate the shock capturing flux at exterior element boundary quadrature points
*/
void updateExteriorElementBoundaryShockCapturingVelocity(proteus_index_t nExteriorElementBoundaries_global,
                                                         int nElementBoundaries_element,
                                                         int nQuadraturePoints_elementBoundary,
                                                         int nQuadraturePoints_element,
                                                         int nSpace,
                                                         proteus_index_t* exteriorElementBoundaries,
                                                         proteus_index_t* elementBoundaryElements,
                                                         int* elementBoundaryLocalElementBoundaries,
                                                         double* numDiff,
                                                         double* grad_u,
                                                         double* velocity)
{
  proteus_index_t ebN, eN_global, I;  int ebNE, ebN_element, k;
  double numDiffAvg;
  for(ebNE=0;ebNE<nExteriorElementBoundaries_global;ebNE++)
    {
      ebN = exteriorElementBoundaries[ebNE];
      eN_global = elementBoundaryElements[ebN*2+0];
      ebN_element = elementBoundaryLocalElementBoundaries[ebN*2+0];
      /*get scalar numDiff coefficient from the left and right assuming constant*/
      numDiffAvg  = numDiff[eN_global*nQuadraturePoints_element+0];
      for(k=0;k<nQuadraturePoints_elementBoundary;k++)
        for(I=0;I<nSpace;I++)
          {
            velocity[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                     ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                     k*nSpace+I]
                  -=
                  numDiffAvg
                  *
                  grad_u[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                         ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                         k*nSpace+I];

          }/*I*/
    }/*ebN*/
}


/**
   \brief Calculate the shock capturing flux at exterior element boundary quadrature points
*/
void updateGlobalExteriorElementBoundaryShockCapturingVelocity(proteus_index_t nExteriorElementBoundaries_global,
                                                               int nQuadraturePoints_elementBoundary,
                                                               int nSpace,
                                                               proteus_index_t* exteriorElementBoundaries,
                                                               proteus_index_t* elementBoundaryElements,
                                                               int* elementBoundaryLocalElementBoundaries,
                                                               double* numDiff,
                                                               double* grad_u,
                                                               double* velocity)
{
  proteus_index_t I;  int ebNE, k;
  double numDiffAvg;
  for(ebNE=0;ebNE<nExteriorElementBoundaries_global;ebNE++)
    {
      for(k=0;k<nQuadraturePoints_elementBoundary;k++)
        for(I=0;I<nSpace;I++)
          {
            numDiffAvg  = numDiff[ebNE*nQuadraturePoints_elementBoundary+k];
            velocity[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                     k*nSpace+I]
                  -=
                  numDiffAvg
                  *
                  grad_u[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                         k*nSpace+I];

          }/*I*/
    }/*ebN*/
}
void calculateInteriorElementBoundaryAverageVelocity(proteus_index_t nInteriorElementBoundaries_global,
                                                     int nElementBoundaries_element,
                                                     int nQuadraturePoints_elementBoundary,
                                                     int nSpace,
                                                     proteus_index_t* interiorElementBoundaries,
                                                     proteus_index_t* elementBoundaryElements,
                                                     int* elementBoundaryLocalElementBoundaries,
                                                     double* v,
                                                     double* vAverage)
{
  proteus_index_t ebN, left_eN_global, right_eN_global, I;  int ebNI, left_ebN_element, right_ebN_element, k;
  for(ebNI=0;ebNI<nInteriorElementBoundaries_global;ebNI++)
    {
      ebN = interiorElementBoundaries[ebNI];
      left_eN_global = elementBoundaryElements[ebN*2+0];
      right_eN_global = elementBoundaryElements[ebN*2+1];
      left_ebN_element = elementBoundaryLocalElementBoundaries[ebN*2+0];
      right_ebN_element = elementBoundaryLocalElementBoundaries[ebN*2+1];
      for(k=0;k<nQuadraturePoints_elementBoundary;k++)
        for (I=0;I<nSpace;I++)
          vAverage[ebN*nQuadraturePoints_elementBoundary*nSpace+
                   k*nSpace+
                   I]
            = 0.5*(v[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                     left_ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                     k*nSpace+
                     I]
                   +
                   v[right_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                     right_ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                     k*nSpace+
                     I]);
    }
}

void calculateExteriorElementBoundaryAverageVelocity(proteus_index_t nExteriorElementBoundaries_global,
                                                     int nElementBoundaries_element,
                                                     int nQuadraturePoints_elementBoundary,
                                                     int nSpace,
                                                     proteus_index_t* exteriorElementBoundaries,
                                                     proteus_index_t* elementBoundaryElements,
                                                     int* elementBoundaryLocalElementBoundaries,
                                                     double* v,
                                                     double* vAverage)
{
  proteus_index_t ebN, left_eN_global, I;  int ebNE, left_ebN_element, k;
  for(ebNE=0;ebNE<nExteriorElementBoundaries_global;ebNE++)
    {
      ebN = exteriorElementBoundaries[ebNE];
      left_eN_global = elementBoundaryElements[ebN*2+0];
      left_ebN_element = elementBoundaryLocalElementBoundaries[ebN*2+0];
      for(k=0;k<nQuadraturePoints_elementBoundary;k++)
        for (I=0;I<nSpace;I++)
          {

            vAverage[ebN*nQuadraturePoints_elementBoundary*nSpace+
                     k*nSpace+
                     I]
              = v[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                  left_ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                  k*nSpace+
                  I];
            /*mwf debug
            printf("vfem.ext.vavg ebN=%d eN=%d ebN_element=%d k=%d v[%d]= %g \n",ebN,left_eN_global,
                   left_ebN_element,k,I,
                   v[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                     left_ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                     k*nSpace+
                     I]);
            */
          }
    }
}




void calculateConservationResidualDG(proteus_index_t nElements_global,
                                     int nDOF_test_element,
                                     double * elementResidual,
                                     double * conservationResidual)
{
  proteus_index_t eN;  int i;
  for (eN = 0; eN < nElements_global; eN++)
    {
      conservationResidual[eN] = 0.0;
      for (i = 0; i < nDOF_test_element; i++)
        conservationResidual[eN] += elementResidual[eN*nDOF_test_element + i];
    }
}

/**
 \brief calculate mass conservation error as
\f[
   \sum_{i} res_{E,i} + \sum_{E,e} \int_{e} \vec v \cdot \vec n_e \ds
\f]
  where \f$res_{E,i}\f$ is the \f$i\f$'th test function's residual on element \f$E\f$
*/
void calculateConservationResidual(proteus_index_t nElements_global,
                                   int nDOF_test_element,
                                   int nElementBoundaries_element,
                                   int nQuadraturePoints_elementBoundary,
                                   int nSpace,
                                   double * n,
                                   double * dS_u,
                                   double * elementResidual,
                                   double * velocity,
                                   double * conservationResidual)
{
  proteus_index_t eN, ebN, I;  int i, k;
  double boundaryFlux;
  for (eN = 0; eN < nElements_global; eN++)
    {
      conservationResidual[eN] = 0.0;

      for (i = 0; i < nDOF_test_element; i++)
        conservationResidual[eN] += elementResidual[eN*nDOF_test_element + i];

      boundaryFlux = 0.0;
      for (ebN = 0; ebN < nElementBoundaries_element; ebN++)
        {
          for (k = 0; k < nQuadraturePoints_elementBoundary; k++)
            {
              for (I = 0; I < nSpace; I++)
                {
                  boundaryFlux +=
                    n[eN*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace +
                      ebN*nQuadraturePoints_elementBoundary*nSpace +
                      k*nSpace +
                      I]
                    *
                    velocity[eN*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace +
                             ebN*nQuadraturePoints_elementBoundary*nSpace +
                             k*nSpace +
                             I]
                    *
                    dS_u[eN*nElementBoundaries_element*nQuadraturePoints_elementBoundary +
                         ebN*nQuadraturePoints_elementBoundary +
                         k];
                }/*I*/
            }/*k*/
        }/*ebN*/
      /*mwf debug
      printf("calcConsRes eN=%d accum= %g boundaryFlux= %g diff=%g \n",eN,conservationResidual[eN],boundaryFlux,
             conservationResidual[eN] -boundaryFlux);

      */
      conservationResidual[eN] += boundaryFlux;

    }/*eN*/
}
/* void calculateConservationResidualGlobalBoundaries(proteus_index_t nElements_global, */
/*                                                 int nInteriorElementBoundaries_global, */
/*                                                 int nExteriorElementBoundaries_global, */
/*                                                 int nElementBoundaries_element, */
/*                                                 int nQuadraturePoints_elementBoundary, */
/*                                                 int nNodes_element, */
/*                                                 int nSpace, */
/*                                                 proteus_index_t* interiorElementBoundaries, */
/*                                                 proteus_index_t* exteriorElementBoundaries, */
/*                                                 proteus_index_t* elementBoundaryElements, */
/*                                                 int* elementBoundaryLocalElementBoundaries, */
/*                                                 double* dS, */
/*                                                 double* normal, */
/*                                                 double* elementResidual, */
/*                                                 double* velocity, */
/*                                                 double* conservationResidual) */
/* { */
/*   proteus_index_t ebN, I; int ebNI, ebNE, eN, nN, left_eN, right_eN, ebN_element, left_ebN_element, right_ebN_element, k; */
/*   register double flux,ds; */
/* /\*   /\\*mwf debug*\\/ *\/ */
/* /\*   register double signDebug = -1.0; *\/ */
/*   /\*first loop through and get element residual sums*\/ */
/*   for (eN = 0; eN < nElements_global; eN++) */
/*     { */
/*       for (nN = 0; nN < nNodes_element; nN++) */
/*      { */
/*        conservationResidual[eN] += elementResidual[eN*nNodes_element + nN]; */
/*      } */
/*     } */
/*   /\*now loop through element boundaries and update element sums*\/ */
/*   /\*interior*\/ */
/*   for (ebNI = 0; ebNI < nInteriorElementBoundaries_global; ebNI++) */
/*     { */
/*       ebN = interiorElementBoundaries[ebNI]; */
/*       left_eN = elementBoundaryElements[ebN*2+0]; */
/*       right_eN = elementBoundaryElements[ebN*2+1]; */
/*       left_ebN_element = elementBoundaryLocalElementBoundaries[ebN*2+0]; */
/*       right_ebN_element = elementBoundaryLocalElementBoundaries[ebN*2+1]; */

/*       flux = 0.0; */
/*       for(k=0;k<nQuadraturePoints_elementBoundary;k++) */
/*      { */
/*           ds = dS[left_eN*nElementBoundaries_element*nQuadraturePoints_elementBoundary+ */
/*                left_ebN_element*nQuadraturePoints_elementBoundary+ */
/*                k]; */
/*        for (I = 0; I < nSpace; I++) */
/*          { */
/*            flux+= velocity[ebN*nQuadraturePoints_elementBoundary*nSpace+ */
/*                            k*nSpace+ */
/*                            I] */
/*              * */
/*              normal[ebN*nQuadraturePoints_elementBoundary*nSpace+ */
/*                     k*nSpace+ */
/*                     I]  */
/*              *  */
/*              ds; */
/*          } */
/*      }/\*k*\/ */
/*       conservationResidual[left_eN] += flux; */
/*       conservationResidual[right_eN]-= flux; */

/*     }/\*ebNI*\/ */
/*   /\*exterior*\/ */
/*   for (ebNE = 0; ebNE < nExteriorElementBoundaries_global; ebNE++) */
/*     { */
/*       ebN = exteriorElementBoundaries[ebNE]; */
/*       eN = elementBoundaryElements[ebN*2+0]; */
/*       ebN_element = elementBoundaryLocalElementBoundaries[ebN*2+0]; */
/*       flux = 0.0; */
/*       for(k=0;k<nQuadraturePoints_elementBoundary;k++) */
/*      { */
/*           ds = dS[eN*nElementBoundaries_element*nQuadraturePoints_elementBoundary+ */
/*                ebN_element*nQuadraturePoints_elementBoundary+ */
/*                k]; */
/*        for (I = 0; I < nSpace; I++) */
/*          { */
/*            flux+= velocity[ebN*nQuadraturePoints_elementBoundary*nSpace+ */
/*                            k*nSpace+ */
/*                            I] */
/*              * */
/*              normal[ebN*nQuadraturePoints_elementBoundary*nSpace+ */
/*                     k*nSpace+ */
/*                     I] */
/*              * */
/*              ds; */
/*          } */
/*      }/\*k*\/ */
/*       conservationResidual[eN] += flux; */
/*     }/\*ebNE*\/ */
/* } */


void copyGlobalElementBoundaryVelocityToElementBoundary(proteus_index_t nElements_global,
                                                        proteus_index_t nInteriorElementBoundaries_global,
                                                        proteus_index_t nExteriorElementBoundaries_global,
                                                        proteus_index_t nElementBoundaries_global,
                                                        int nElementBoundaries_element,
                                                        int nQuadraturePoints_elementBoundary,
                                                        int nSpace,
                                                        proteus_index_t * interiorElementBoundaries,
                                                        proteus_index_t * exteriorElementBoundaries,
                                                        proteus_index_t * elementBoundaryElementsArray,
                                                        int * elementBoundaryLocalElementBoundariesArray,
                                                        double * velocityBoundary_global,
                                                        double * velocityBoundary_element)
{
  proteus_index_t ebN, eN_left, eN_right, I;  int ebNE, ebNI, ebN_left_element, ebN_right_element, k;

  for (ebNI = 0; ebNI < nInteriorElementBoundaries_global; ebNI++)
    {
      ebN = interiorElementBoundaries[ebNI];
      eN_left = elementBoundaryElementsArray[ebN*2 + 0];
      eN_right= elementBoundaryElementsArray[ebN*2 + 1];
      ebN_left_element = elementBoundaryLocalElementBoundariesArray[ebN*2 + 0];
      ebN_right_element= elementBoundaryLocalElementBoundariesArray[ebN*2 + 1];

      for (k = 0; k < nQuadraturePoints_elementBoundary; k++)
        for (I = 0; I < nSpace; I++)
          {
            velocityBoundary_element[eN_left*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace +
                                     ebN_left_element*nQuadraturePoints_elementBoundary*nSpace +
                                     k*nSpace +
                                     I] =
              velocityBoundary_global[ebN*nQuadraturePoints_elementBoundary*nSpace +
                                      k*nSpace +
                                      I];
            velocityBoundary_element[eN_right*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace +
                                     ebN_right_element*nQuadraturePoints_elementBoundary*nSpace +
                                     k*nSpace +
                                     I] =
              velocityBoundary_global[ebN*nQuadraturePoints_elementBoundary*nSpace +
                                      k*nSpace +
                                      I];

          }
    }/*ebNI*/

  for (ebNE = 0; ebNE < nExteriorElementBoundaries_global; ebNE++)
    {
      ebN = exteriorElementBoundaries[ebNE];
      eN_left = elementBoundaryElementsArray[ebN*2 + 0];
      ebN_left_element = elementBoundaryLocalElementBoundariesArray[ebN*2 + 0];

      for (k = 0; k < nQuadraturePoints_elementBoundary; k++)
        for (I = 0; I < nSpace; I++)
          {
            velocityBoundary_element[eN_left*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace +
                                     ebN_left_element*nQuadraturePoints_elementBoundary*nSpace +
                                     k*nSpace +
                                     I] =
              velocityBoundary_global[ebN*nQuadraturePoints_elementBoundary*nSpace +
                                      k*nSpace +
                                      I];
          }
    }/*ebNE*/
}
void loadBoundaryFluxIntoGlobalElementBoundaryVelocity(proteus_index_t nExteriorElementBoundaries_global,
                                                       int nQuadraturePoints_elementBoundary,
                                                       int nSpace,
                                                       proteus_index_t* exteriorElementBoundaries,
                                                       proteus_index_t* fluxElementBoundaries,
                                                       double* normal,
                                                       double* flux,
                                                       double updateCoef,
                                                       double* velocity)
{
  proteus_index_t ebN, I;  int ebNE, k;
  double val;
  for (ebNE = 0; ebNE < nExteriorElementBoundaries_global; ebNE++)
    {
      ebN = exteriorElementBoundaries[ebNE];
      if (fluxElementBoundaries[ebNE] > 0)
        {
          for (k = 0; k < nQuadraturePoints_elementBoundary; k++)
            {
              for (I=0; I < nSpace; I++)
                {
                  val = velocity[ebN*nQuadraturePoints_elementBoundary*nSpace +
                                 k*nSpace + I];
                  velocity[ebN*nQuadraturePoints_elementBoundary*nSpace +
                           k*nSpace + I]
                    = val*updateCoef +
                    flux[ebN*nQuadraturePoints_elementBoundary +
                         k]
                    *
                    normal[ebN*nQuadraturePoints_elementBoundary*nSpace +
                           k*nSpace + I];
                  /*mwf debug
                  printf("load fluxes ebNE=%d ebN=%d k=%d I=%d val=%g flux=%g n=%g vel=%g\n",
                         ebNE,ebN,k,I,val,
                         flux[ebN*nQuadraturePoints_elementBoundary+k],
                         normal[ebN*nQuadraturePoints_elementBoundary*nSpace + k*nSpace + I],
                         velocity[ebN*nQuadraturePoints_elementBoundary*nSpace +
                                  k*nSpace + I]);
                  */
                }
            }
        }
    }

}

#define TR_ALPHA 0.5
#define TR_ALPHA_EXT 1.0

/**
   \brief Calculate the trace of the potential on interior element boundaries. Use the arithmetic average
*/
void calculateInteriorNumericalTrace_Potential(proteus_index_t nInteriorElementBoundaries_global,
                                               int nElementBoundaries_element,
                                               int nQuadraturePoints_elementBoundary,
                                               proteus_index_t* interiorElementBoundaries,
                                               proteus_index_t* elementBoundaryElements,
                                               int* elementBoundaryLocalElementBoundaries,
                                               double* phi,
                                               double* dphi,
                                               double* phi_trace,
                                               double* dphi_trace_left,
                                               double* dphi_trace_right)
{
  proteus_index_t ebN, left_eN_global, right_eN_global;  int ebNI, left_ebN_element, right_ebN_element, k;
  for(ebNI=0;ebNI<nInteriorElementBoundaries_global;ebNI++)
    {
      ebN = interiorElementBoundaries[ebNI];
      left_eN_global = elementBoundaryElements[ebN*2+0];
      right_eN_global = elementBoundaryElements[ebN*2+1];
      left_ebN_element = elementBoundaryLocalElementBoundaries[ebN*2+0];
      right_ebN_element = elementBoundaryLocalElementBoundaries[ebN*2+1];
      for(k=0;k<nQuadraturePoints_elementBoundary;k++)
        {
          phi_trace[ebN*nQuadraturePoints_elementBoundary+
                    k]
            = (1.0-TR_ALPHA)*phi[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary+
                              left_ebN_element*nQuadraturePoints_elementBoundary+
                              k]
            +
            TR_ALPHA*phi[right_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary+
                      right_ebN_element*nQuadraturePoints_elementBoundary+
                      k];
          dphi_trace_left[ebN*nQuadraturePoints_elementBoundary+
                          k]
            = (1.0-TR_ALPHA)*
            dphi[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary+
                 left_ebN_element*nQuadraturePoints_elementBoundary+
                 k];
          dphi_trace_right[ebN*nQuadraturePoints_elementBoundary+
                           k]
            = TR_ALPHA*dphi[right_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary+
                         right_ebN_element*nQuadraturePoints_elementBoundary+
                         k];
        }
    }
}

/**
   \brief Calculate the trace of the potential on interior element boundaries. Use the arithmetic average
*/
void calculateExteriorNumericalTrace_Potential(int* isDOFBoundary,
                                               proteus_index_t nExteriorElementBoundaries_global,
                                               int nElementBoundaries_element,
                                               int nQuadraturePoints_elementBoundary,
                                               proteus_index_t* exteriorElementBoundaries,
                                               proteus_index_t* elementBoundaryElements,
                                               int* elementBoundaryLocalElementBoundaries,
                                               double* phi_bc,
                                               double* phi,
                                               double* dphi,
                                               double* phi_trace,
                                               double* dphi_trace_left)
{
  proteus_index_t ebN, eN_global;  int ebNE, ebN_element, k;
  for(ebNE=0;ebNE<nExteriorElementBoundaries_global;ebNE++)
    {
      ebN = exteriorElementBoundaries[ebNE];
      eN_global = elementBoundaryElements[ebN*2+0];
      ebN_element = elementBoundaryLocalElementBoundaries[ebN*2+0];
      for(k=0;k<nQuadraturePoints_elementBoundary;k++)
        {
          phi_trace[ebN*nQuadraturePoints_elementBoundary+
                    k]
            = (1.0-TR_ALPHA_EXT)*phi[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary+
                              ebN_element*nQuadraturePoints_elementBoundary+
                              k]
            +
            TR_ALPHA_EXT*phi_bc[ebNE*nQuadraturePoints_elementBoundary+
                                k];
          if(isDOFBoundary[ebNE*nQuadraturePoints_elementBoundary+k] == 1)
            dphi_trace_left[ebN*nQuadraturePoints_elementBoundary+
                            k]
              = (1.0-TR_ALPHA_EXT)*dphi[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary+
                                        ebN_element*nQuadraturePoints_elementBoundary+
                                        k];
          else
            dphi_trace_left[ebN*nQuadraturePoints_elementBoundary+
                            k]
              = dphi[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary+
                     ebN_element*nQuadraturePoints_elementBoundary+
                     k];
        }
    }
}
/**
   \brief Update the element boundary flux on interior element boundaries
*/
void updateInteriorElementBoundary_MixedForm_weak(proteus_index_t nInteriorElementBoundaries_global,
                                                  int nElementBoundaries_element,
                                                  int nQuadraturePoints_elementBoundary,
                                                  int nDOF_test_element,
                                                  int nSpace,
                                                  proteus_index_t* interiorElementBoundaries,
                                                  proteus_index_t* elementBoundaryElements,
                                                  int* elementBoundaryLocalElementBoundaries,
                                                  double* n,
                                                  double* phi_trace,
                                                  double* w_dS,
                                                  double* b)
{
  proteus_index_t ebN, left_eN_global, right_eN_global, I;  int ebNI, left_ebN_element, right_ebN_element, i, k;
  for(ebNI=0;ebNI<nInteriorElementBoundaries_global;ebNI++)
    {
      ebN = interiorElementBoundaries[ebNI];
      left_eN_global = elementBoundaryElements[ebN*2+0];
      right_eN_global = elementBoundaryElements[ebN*2+1];
      left_ebN_element = elementBoundaryLocalElementBoundaries[ebN*2+0];
      right_ebN_element = elementBoundaryLocalElementBoundaries[ebN*2+1];
      for(i=0;i<nDOF_test_element;i++)
        for(k=0;k<nQuadraturePoints_elementBoundary;k++)
          for(I=0;I<nSpace;I++)
            {
              b[left_eN_global*nDOF_test_element*nSpace+
                I*nDOF_test_element+
                i]
                -=
                phi_trace[ebN*nQuadraturePoints_elementBoundary+
                          k]
                *
                n[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                  left_ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                  k*nSpace+
                  I]
                *
                w_dS[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_test_element+
                     left_ebN_element*nQuadraturePoints_elementBoundary*nDOF_test_element+
                     k*nDOF_test_element+
                     i];
              b[right_eN_global*nDOF_test_element*nSpace+
                I*nDOF_test_element+
                i]
                -=
                phi_trace[ebN*nQuadraturePoints_elementBoundary+
                          k]
                *
                n[right_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                  right_ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                  k*nSpace+
                  I]
                *
                w_dS[right_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_test_element+
                     right_ebN_element*nQuadraturePoints_elementBoundary*nDOF_test_element+
                     k*nDOF_test_element+
                     i];
            }
    }
}

/**
   \brief Update the element boundary flux on interior element boundaries
*/
void updateInteriorElementBoundary_MixedForm_weakJacobian(proteus_index_t nInteriorElementBoundaries_global,
                                                           int nElementBoundaries_element,
                                                           int nQuadraturePoints_elementBoundary,
                                                           int nDOF_test_element,
                                                           int nSpace,
                                                           proteus_index_t* interiorElementBoundaries,
                                                           proteus_index_t* elementBoundaryElements,
                                                           int* elementBoundaryLocalElementBoundaries,
                                                           double* n,
                                                           double* dphi_trace_left,
                                                           double* dphi_trace_right,
                                                           double* v,
                                                           double* w_dS,
                                                           double* db,
                                                           double* db_eb)
{
  proteus_index_t ebN, left_eN_global, right_eN_global, I;  int ebNI, left_ebN_element, right_ebN_element, i, j, k, nDOF_test_element2=nDOF_test_element*nDOF_test_element;
  for(ebNI=0;ebNI<nInteriorElementBoundaries_global;ebNI++)
    {
      ebN = interiorElementBoundaries[ebNI];
      left_eN_global = elementBoundaryElements[ebN*2+0];
      right_eN_global = elementBoundaryElements[ebN*2+1];
      left_ebN_element = elementBoundaryLocalElementBoundaries[ebN*2+0];
      right_ebN_element = elementBoundaryLocalElementBoundaries[ebN*2+1];
      for(i=0;i<nDOF_test_element;i++)
        for(j=0;j<nDOF_test_element;j++)
          for(k=0;k<nQuadraturePoints_elementBoundary;k++)
            for(I=0;I<nSpace;I++)
              {
                db[left_eN_global*nDOF_test_element2*nSpace+
                   I*nDOF_test_element2+
                   i*nDOF_test_element+
                   j]
                  -=
                  dphi_trace_left[ebN*nQuadraturePoints_elementBoundary+
                                  k]*

                  v[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_test_element+
                    left_ebN_element*nQuadraturePoints_elementBoundary*nDOF_test_element+
                    k*nDOF_test_element+
                    j]
                  *
                  n[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                    left_ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                    k*nSpace+
                    I]
                  *
                  w_dS[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_test_element+
                       left_ebN_element*nQuadraturePoints_elementBoundary*nDOF_test_element+
                       k*nDOF_test_element+
                       i];
                db_eb[left_eN_global*nElementBoundaries_element*nDOF_test_element2*nSpace+
                      left_ebN_element*nDOF_test_element2*nSpace+
                      I*nDOF_test_element2+
                      i*nDOF_test_element+
                      j]
                  -=
                  dphi_trace_right[ebN*nQuadraturePoints_elementBoundary+
                                   k]
                  *
                  v[right_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_test_element+
                    right_ebN_element*nQuadraturePoints_elementBoundary*nDOF_test_element+
                    k*nDOF_test_element+
                    j]
                  *
                  n[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                    left_ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                    k*nSpace+
                    I]
                  *
                  w_dS[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_test_element+
                       left_ebN_element*nQuadraturePoints_elementBoundary*nDOF_test_element+
                       k*nDOF_test_element+
                       i];
                db_eb[right_eN_global*nElementBoundaries_element*nDOF_test_element2*nSpace+
                      right_ebN_element*nDOF_test_element2*nSpace+
                      I*nDOF_test_element2+
                      i*nDOF_test_element+
                      j]
                  -=
                  dphi_trace_left[ebN*nQuadraturePoints_elementBoundary+
                                  k]*

                  v[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_test_element+
                    left_ebN_element*nQuadraturePoints_elementBoundary*nDOF_test_element+
                    k*nDOF_test_element+
                    j]
                  *
                  n[right_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                    right_ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                    k*nSpace+
                    I]
                  *
                  w_dS[right_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_test_element+
                       right_ebN_element*nQuadraturePoints_elementBoundary*nDOF_test_element+
                       k*nDOF_test_element+
                       i];
                db[right_eN_global*nDOF_test_element2*nSpace+
                   I*nDOF_test_element2+
                   i*nDOF_test_element+
                   j]
                  -=
                  dphi_trace_right[ebN*nQuadraturePoints_elementBoundary+
                                   k]
                  *
                  v[right_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_test_element+
                    right_ebN_element*nQuadraturePoints_elementBoundary*nDOF_test_element+
                    k*nDOF_test_element+
                    j]
                  *
                  n[right_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                    right_ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                    k*nSpace+
                    I]
                  *
                  w_dS[right_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_test_element+
                       right_ebN_element*nQuadraturePoints_elementBoundary*nDOF_test_element+
                       k*nDOF_test_element+
                       i];
              }
    }
}

/**
   \brief Update the element boundary flux on exterior element boundaries
*/
void updateExteriorElementBoundary_MixedForm_weak(proteus_index_t nExteriorElementBoundaries_global,
                                                  int nElementBoundaries_element,
                                                  int nQuadraturePoints_elementBoundary,
                                                  int nDOF_test_element,
                                                  int nSpace,
                                                  proteus_index_t* exteriorElementBoundaries,
                                                  proteus_index_t* elementBoundaryElements,
                                                  int* elementBoundaryLocalElementBoundaries,
                                                  double* n,
                                                  double* phi_trace,
                                                  double* w_dS,
                                                  double* b)
{
  proteus_index_t ebN, eN_global, I;  int ebNE, ebN_element, i, k;
  for(ebNE=0;ebNE<nExteriorElementBoundaries_global;ebNE++)
    {
      ebN = exteriorElementBoundaries[ebNE];
      eN_global = elementBoundaryElements[ebN*2+0];
      ebN_element = elementBoundaryLocalElementBoundaries[ebN*2+0];
      for(i=0;i<nDOF_test_element;i++)
        for(k=0;k<nQuadraturePoints_elementBoundary;k++)
          for (I=0;I<nSpace;I++)
            b[eN_global*nDOF_test_element*nSpace+
              I*nDOF_test_element+
              i]
              -=
              phi_trace[ebN*nQuadraturePoints_elementBoundary+
                        k]
              *
              n[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                k*nSpace+
                I]
              *
              w_dS[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_test_element+
                   ebN_element*nQuadraturePoints_elementBoundary*nDOF_test_element+
                   k*nDOF_test_element+
                   i];
    }
}

/**
   \brief Update the element boundary flux on interior element boundaries
*/
void updateExteriorElementBoundary_MixedForm_weakJacobian(proteus_index_t nExteriorElementBoundaries_global,
                                                           int nElementBoundaries_element,
                                                           int nQuadraturePoints_elementBoundary,
                                                           int nDOF_test_element,
                                                           int nSpace,
                                                           proteus_index_t* exteriorElementBoundaries,
                                                           proteus_index_t* elementBoundaryElements,
                                                           int* elementBoundaryLocalElementBoundaries,
                                                           double* n,
                                                           double* dphi_trace_left,
                                                           double* v,
                                                           double* w_dS,
                                                           double* db,
                                                           double* db_eb)
{
  proteus_index_t ebN, eN_global, I;  int ebNE, ebN_element, i, j, k, nDOF_test_element2=nDOF_test_element*nDOF_test_element;
  for(ebNE=0;ebNE<nExteriorElementBoundaries_global;ebNE++)
    {
      ebN = exteriorElementBoundaries[ebNE];
      eN_global = elementBoundaryElements[ebN*2+0];
      ebN_element = elementBoundaryLocalElementBoundaries[ebN*2+0];
      for(i=0;i<nDOF_test_element;i++)
        for(j=0;j<nDOF_test_element;j++)
          for(k=0;k<nQuadraturePoints_elementBoundary;k++)
            for(I=0;I<nSpace;I++)
              {
                db[eN_global*nDOF_test_element2*nSpace+
                   I*nDOF_test_element2+
                   i*nDOF_test_element+
                   j]
                  -=
                  dphi_trace_left[ebN*nQuadraturePoints_elementBoundary+
                                  k]
                  *
                  v[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_test_element+
                    ebN_element*nQuadraturePoints_elementBoundary*nDOF_test_element+
                    k*nDOF_test_element+
                    j]
                  *
                  n[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                    ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                    k*nSpace+
                    I]
                  *
                  w_dS[eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_test_element+
                       ebN_element*nQuadraturePoints_elementBoundary*nDOF_test_element+
                       k*nDOF_test_element+
                       i];
              }
    }
}

void updatePotential_MixedForm_weak(proteus_index_t nElements_global,
                                    int nQuadraturePoints_element,
                                    int nDOF_test_element,
                                    int nSpace,
                                    double* phi,
                                    double* grad_w_dV,
                                    double* b)
{
  proteus_index_t eN, I;  int i, k;
  for(eN=0;eN<nElements_global;eN++)
    for (I=0;I<nSpace;I++)
      for (i=0;i<nDOF_test_element;i++)
        for (k=0;k<nQuadraturePoints_element;k++)
          b[eN*nDOF_test_element*nSpace +
            I*nDOF_test_element+
            i]
            +=
            phi[eN*nQuadraturePoints_element+
                k]
            *
            grad_w_dV[eN*nQuadraturePoints_element*nDOF_test_element*nSpace +
                      k*nDOF_test_element*nSpace +
                      i*nSpace +
                      I];
}

void updatePotential_MixedForm_weak_gwvd(proteus_index_t nElements_global,
                                    int nQuadraturePoints_element,
                                    int nDOF_test_element,
                                    int nSpace,
                                    double  epsilon,
                                    double* phi,
                                    double* w_dV,
                                    double* grad_w_dV,
                                    double* b,
                                    double* mf)
{
  proteus_index_t eN, I;  int i, k;
  for(eN=0;eN<nElements_global;eN++)
    for (I=0;I<nSpace;I++)
      for (i=0;i<nDOF_test_element;i++)
        for (k=0;k<nQuadraturePoints_element;k++)
          {
            b[eN*nDOF_test_element*nSpace +
            I*nDOF_test_element+i]
              +=
            phi[eN*nQuadraturePoints_element+
                k]
            *
            grad_w_dV[eN*nQuadraturePoints_element*nDOF_test_element*nSpace +
                      k*nDOF_test_element*nSpace +
                      i*nSpace +
                      I];
                if (I==nSpace-1)
              {
                b[eN*nDOF_test_element*nSpace +
                I*nDOF_test_element+i]
               -= epsilon*mf[eN*nQuadraturePoints_element+
                             k ]*w_dV[eN*nQuadraturePoints_element*nDOF_test_element+ k*nDOF_test_element + i ];
                             }
          }
}

void updatePotential_MixedForm_weakJacobian(proteus_index_t nElements_global,
                                            int nQuadraturePoints_element,
                                            int nDOF_test_element,
                                            int nSpace,
                                            double* dphi,
                                            double* v,
                                            double* grad_w_dV,
                                            double* db)
{
  proteus_index_t eN, I;  int i, j, k, nDOF_test_element2=nDOF_test_element*nDOF_test_element;
  for(eN=0;eN<nElements_global;eN++)
    for (I=0;I<nSpace;I++)
      for (i=0;i<nDOF_test_element;i++)
        for (j=0;j<nDOF_test_element;j++)
          for (k=0;k<nQuadraturePoints_element;k++)
            db[eN*nDOF_test_element2*nSpace +
               I*nDOF_test_element2+
               i*nDOF_test_element+
               j]
              +=
              dphi[eN*nQuadraturePoints_element+
                   k]
              *
              v[eN*nQuadraturePoints_element*nDOF_test_element+
                k*nDOF_test_element+
                j]
              *
              grad_w_dV[eN*nQuadraturePoints_element*nDOF_test_element*nSpace +
                        k*nDOF_test_element*nSpace +
                        i*nSpace +
                        I];
}

void calculateVelocityQuadrature_MixedForm(proteus_index_t nElements_global,
                                           int nElementBoundaries_element,
                                           int nElementBoundaryQuadraturePoints_elementBoundary,
                                           int nDOF_element,
                                           int nSpace,
                                           int nQuadraturePoints_element,
                                           double* A_inv,
                                           double* b,
                                           double* v,
                                           double* V,
                                           double* qv,
                                           double* qV)
{
  proteus_index_t eN, ebN, I;  int k, i, j, nDOF_element2=nDOF_element*nDOF_element;
  double V_dof[nSpace][nDOF_element];
  memset(V,0,sizeof(double)*
         nElements_global*
         nElementBoundaries_element*
         nElementBoundaryQuadraturePoints_elementBoundary*
         nSpace);
  memset(qV,0,sizeof(double)*
         nElements_global*
         nQuadraturePoints_element*
         nSpace);
  for(eN=0;eN<nElements_global;eN++)
    {
      /* velocity DOF */
      for(I=0;I<nSpace;I++)
        for(i=0;i<nDOF_element;i++)
          {
            V_dof[I][i]=0.0;
            for(j=0;j<nDOF_element;j++)
              V_dof[I][i]
                +=
                A_inv[eN*nDOF_element2+
                      i*nDOF_element+
                      j]
                *
                b[eN*nSpace*nDOF_element+
                  I*nDOF_element+
                  j];
          }
      /* evaluate at element quadrature */
      for(k=0;k<nQuadraturePoints_element;k++)
        for(j=0;j<nDOF_element;j++)
          for(I=0;I<nSpace;I++)
            qV[eN*nQuadraturePoints_element*nSpace+
               k*nSpace+
               I]
              +=
              V_dof[I][j]
              *
              qv[eN*nQuadraturePoints_element*nDOF_element+
                 k*nDOF_element+
                 j];
      /* evaluate at element boundary quadrature*/
      for (ebN=0;ebN<nElementBoundaries_element;ebN++)
        for(k=0;k<nElementBoundaryQuadraturePoints_elementBoundary;k++)
          for(j=0;j<nDOF_element;j++)
            for(I=0;I<nSpace;I++)
              V[eN*nElementBoundaries_element*nElementBoundaryQuadraturePoints_elementBoundary*nSpace+
                ebN*nElementBoundaryQuadraturePoints_elementBoundary*nSpace+
                k*nSpace+
                I]
                +=
                V_dof[I][j]
                *
                v[eN*nElementBoundaries_element*nElementBoundaryQuadraturePoints_elementBoundary*nDOF_element+
                  ebN*nElementBoundaryQuadraturePoints_elementBoundary*nDOF_element+
                  k*nDOF_element+
                  j];
    }
}

void calculateVelocityQuadrature_MixedForm2(proteus_index_t nElements_global,
                                            int nElementBoundaries_element,
                                            int nElementBoundaryQuadraturePoints_elementBoundary,
                                            int nDOF_element,
                                            int nSpace,
                                            int nQuadraturePoints_element,
                                            double* qa,
                                            double* qw_dV,
                                            double* b,
                                            double* v,
                                            double* V,
                                            double* qv,
                                            double* qV)
{
  proteus_index_t nSpace2=nSpace*nSpace;  proteus_index_t ebN, I; int eN, k, i, j, nDOF_element2=nDOF_element*nDOF_element;
  PROTEUS_LAPACK_INTEGER ipiv[nDOF_element],lwork=((PROTEUS_LAPACK_INTEGER)nDOF_element),dim=((PROTEUS_LAPACK_INTEGER)nDOF_element),info=0;
  double work[nDOF_element],A_inv[nDOF_element2];
  double V_dof[nSpace][nDOF_element];
  memset(V,0,sizeof(double)*
         nElements_global*
         nElementBoundaries_element*
         nElementBoundaryQuadraturePoints_elementBoundary*
         nSpace);
  memset(qV,0,sizeof(double)*
         nElements_global*
         nQuadraturePoints_element*
         nSpace);
  for(eN=0;eN<nElements_global;eN++)
    {
      for(I=0;I<nSpace;I++)
        {
          memset(A_inv,0,sizeof(double)*nDOF_element2);
          for(i=0;i<nDOF_element;i++)
            for(j=0;j<nDOF_element;j++)
              {
                for(k=0;k<nQuadraturePoints_element;k++)
                  {
                    //cek hack do diagonal only for now
                    A_inv[i*nDOF_element+j] += (1.0/qa[eN*nQuadraturePoints_element*nSpace2+
                                                       k*nSpace2+
                                                       I*nSpace+
                                                       I])
                      *qv[eN*nQuadraturePoints_element*nDOF_element+
                          k*nDOF_element+
                          j]
                      *
                      qw_dV[eN*nQuadraturePoints_element*nDOF_element+
                            k*nDOF_element+
                            i];
                  }
              }
          info=0;
          dgetrf_(&dim,&dim,A_inv,&dim,ipiv,&info);
          dgetri_(&dim,A_inv,&dim,ipiv,work,&lwork,&info);

          /* velocity DOF */
          for(i=0;i<nDOF_element;i++)
            {
              V_dof[I][i]=0.0;
              for(j=0;j<nDOF_element;j++)
                V_dof[I][i]
                  +=
                  A_inv[i*nDOF_element+
                        j]
                  *
                  b[eN*nSpace*nDOF_element+
                    I*nDOF_element+
                    j];
            }
        }
      /* evaluate at element quadrature */
      for(k=0;k<nQuadraturePoints_element;k++)
        for(j=0;j<nDOF_element;j++)
          for(I=0;I<nSpace;I++)
            qV[eN*nQuadraturePoints_element*nSpace+
               k*nSpace+
               I]
              +=
              V_dof[I][j]
              *
              qv[eN*nQuadraturePoints_element*nDOF_element+
                 k*nDOF_element+
                 j];
      /* evaluate at element boundary quadrature*/
      for (ebN=0;ebN<nElementBoundaries_element;ebN++)
        for(k=0;k<nElementBoundaryQuadraturePoints_elementBoundary;k++)
          for(j=0;j<nDOF_element;j++)
            for(I=0;I<nSpace;I++)
              V[eN*nElementBoundaries_element*nElementBoundaryQuadraturePoints_elementBoundary*nSpace+
                ebN*nElementBoundaryQuadraturePoints_elementBoundary*nSpace+
                k*nSpace+
                I]
                +=
                V_dof[I][j]
                *
                v[eN*nElementBoundaries_element*nElementBoundaryQuadraturePoints_elementBoundary*nDOF_element+
                  ebN*nElementBoundaryQuadraturePoints_elementBoundary*nDOF_element+
                  k*nDOF_element+
                  j];
    }
}

void calculateVelocityQuadrature_MixedForm2_sd(proteus_index_t nElements_global,
                                               int nElementBoundaries_element,
                                               int nElementBoundaryQuadraturePoints_elementBoundary,
                                               int nDOF_element,
                                               int nSpace,
                                               int nQuadraturePoints_element,
                                               const proteus_index_t * rowptr,
                                               const proteus_index_t * colind,
                                               double* qa,
                                               double* qw_dV,
                                               double* b,
                                               double* v,
                                               double* V,
                                               double* qv,
                                               double* qV)
{
  proteus_index_t nSpace2=nSpace*nSpace;  proteus_index_t ebN, I; int eN, k, i, j, nDOF_element2=nDOF_element*nDOF_element;
  proteus_index_t m, nnz=rowptr[nSpace];
  PROTEUS_LAPACK_INTEGER ipiv[nDOF_element],lwork=((PROTEUS_LAPACK_INTEGER)nDOF_element),dim=((PROTEUS_LAPACK_INTEGER)nDOF_element),info=0;
  double work[nDOF_element],A_inv[nDOF_element2];
  double V_dof[nSpace][nDOF_element];
  memset(V,0,sizeof(double)*
         nElements_global*
         nElementBoundaries_element*
         nElementBoundaryQuadraturePoints_elementBoundary*
         nSpace);
  memset(qV,0,sizeof(double)*
         nElements_global*
         nQuadraturePoints_element*
         nSpace);
  for(eN=0;eN<nElements_global;eN++)
    {
      for(I=0;I<nSpace;I++)
        {
          memset(A_inv,0,sizeof(double)*nDOF_element2);
          for(i=0;i<nDOF_element;i++)
            for(j=0;j<nDOF_element;j++)
              {
                for(k=0;k<nQuadraturePoints_element;k++)
                  {
                    //cek hack do diagonal only for now
                    for (m=rowptr[I]; m < rowptr[I+1];m++)
                      if (colind[m] == I)
                        {
                          /*mwf debug
                          printf("mixedform2 eN=%d I=%d m=%d colind[m]=%d a=%g \n",
                                 eN,I,m,colind[m],qa[eN*nQuadraturePoints_element*nnz+
                                                     k*nnz + m]);
                          */
                          A_inv[i*nDOF_element+j] += (1.0/qa[eN*nQuadraturePoints_element*nnz+
                                                             k*nnz + m])
                            *qv[eN*nQuadraturePoints_element*nDOF_element+
                                k*nDOF_element+
                                j]
                            *
                            qw_dV[eN*nQuadraturePoints_element*nDOF_element+
                                  k*nDOF_element+
                                  i];
                        }
                  }
              }
          info=0;
          dgetrf_(&dim,&dim,A_inv,&dim,ipiv,&info);
          dgetri_(&dim,A_inv,&dim,ipiv,work,&lwork,&info);

          /* velocity DOF */
          for(i=0;i<nDOF_element;i++)
            {
              V_dof[I][i]=0.0;
              for(j=0;j<nDOF_element;j++)
                V_dof[I][i]
                  +=
                  A_inv[i*nDOF_element+
                        j]
                  *
                  b[eN*nSpace*nDOF_element+
                    I*nDOF_element+
                    j];
            }
        }
      /* evaluate at element quadrature */
      for(k=0;k<nQuadraturePoints_element;k++)
        for(j=0;j<nDOF_element;j++)
          for(I=0;I<nSpace;I++)
            qV[eN*nQuadraturePoints_element*nSpace+
               k*nSpace+
               I]
              +=
              V_dof[I][j]
              *
              qv[eN*nQuadraturePoints_element*nDOF_element+
                 k*nDOF_element+
                 j];
      /* evaluate at element boundary quadrature*/
      for (ebN=0;ebN<nElementBoundaries_element;ebN++)
        for(k=0;k<nElementBoundaryQuadraturePoints_elementBoundary;k++)
          for(j=0;j<nDOF_element;j++)
            for(I=0;I<nSpace;I++)
              V[eN*nElementBoundaries_element*nElementBoundaryQuadraturePoints_elementBoundary*nSpace+
                ebN*nElementBoundaryQuadraturePoints_elementBoundary*nSpace+
                k*nSpace+
                I]
                +=
                V_dof[I][j]
                *
                v[eN*nElementBoundaries_element*nElementBoundaryQuadraturePoints_elementBoundary*nDOF_element+
                  ebN*nElementBoundaryQuadraturePoints_elementBoundary*nDOF_element+
                  k*nDOF_element+
                  j];
    }
}

/* Velocity Quadrature_MixedForm2 function that saves the velocity degrees of freedom, tjp added*/

void calculateVelocityQuadrature_MixedForm2_vdof_sd(proteus_index_t nElements_global,
                                               int nElementBoundaries_element,
                                               int nElementBoundaryQuadraturePoints_elementBoundary,
                                               int nDOF_element,
                                               int nSpace,
                                               int nQuadraturePoints_element,
                                               const proteus_index_t * rowptr,
                                               const proteus_index_t * colind,
                                               double* qa,
                                               double* qw_dV,
                                               double* b,
                                               double* v,
                                               double* V,
                                               double* qv,
                                               double* qV,
                                               double* vel_dofs)
{
  proteus_index_t nSpace2=nSpace*nSpace;  proteus_index_t ebN, I; int eN, k, i, j, nDOF_element2=nDOF_element*nDOF_element;
  proteus_index_t m, nnz=rowptr[nSpace];
  PROTEUS_LAPACK_INTEGER ipiv[nDOF_element],lwork=((PROTEUS_LAPACK_INTEGER)nDOF_element),dim=((PROTEUS_LAPACK_INTEGER)nDOF_element),info=0;
  double work[nDOF_element],A_inv[nDOF_element2];
  double V_dof[nSpace][nDOF_element];
  double vel_dofs_temp[nElements_global][nSpace][nDOF_element];
  memset(V,0,sizeof(double)*
         nElements_global*
         nElementBoundaries_element*
         nElementBoundaryQuadraturePoints_elementBoundary*
         nSpace);
  memset(qV,0,sizeof(double)*
         nElements_global*
         nQuadraturePoints_element*
         nSpace);
  for(eN=0;eN<nElements_global;eN++)
    {
      for(I=0;I<nSpace;I++)
        {
          memset(A_inv,0,sizeof(double)*nDOF_element2);
          for(i=0;i<nDOF_element;i++)
            for(j=0;j<nDOF_element;j++)
              {
                for(k=0;k<nQuadraturePoints_element;k++)
                  {
                    //cek hack do diagonal only for now
                    for (m=rowptr[I]; m < rowptr[I+1];m++)
                      if (colind[m] == I)
                        {
                          /*mwf debug
                          printf("mixedform2 eN=%d I=%d m=%d colind[m]=%d a=%g \n",
                                 eN,I,m,colind[m],qa[eN*nQuadraturePoints_element*nnz+
                                                     k*nnz + m]);
                          */
                          A_inv[i*nDOF_element+j] += (1.0/qa[eN*nQuadraturePoints_element*nnz+
                                                             k*nnz + m])
                            *qv[eN*nQuadraturePoints_element*nDOF_element+
                                k*nDOF_element+
                                j]
                            *
                            qw_dV[eN*nQuadraturePoints_element*nDOF_element+
                                  k*nDOF_element+
                                  i];
                        }
                  }
              }
          info=0;
          dgetrf_(&dim,&dim,A_inv,&dim,ipiv,&info);
          dgetri_(&dim,A_inv,&dim,ipiv,work,&lwork,&info);

          /* velocity DOF */
          for(i=0;i<nDOF_element;i++)
            {
              V_dof[I][i]=0.0;
              vel_dofs_temp[eN][I][i]=0.0;
              for(j=0;j<nDOF_element;j++)
                {
                V_dof[I][i]
                  +=
                  A_inv[i*nDOF_element+
                        j]
                  *
                  b[eN*nSpace*nDOF_element+
                    I*nDOF_element+
                    j];
                vel_dofs_temp[eN][I][i]
                += A_inv[i*nDOF_element+
                        j]
                  *
                  b[eN*nSpace*nDOF_element+
                    I*nDOF_element+
                    j];
                }
            }
        }

      /* Change the shape of the velocity degrees of freedom */
        for(j=0;j<nDOF_element;j++)
          for(I=0;I<nSpace;I++)
            vel_dofs[eN*nDOF_element*nSpace+
               j*nSpace+
               I]
              =vel_dofs_temp[eN][I][j];

      /* evaluate at element quadrature */
      for(k=0;k<nQuadraturePoints_element;k++)
        for(j=0;j<nDOF_element;j++)
          for(I=0;I<nSpace;I++)
            qV[eN*nQuadraturePoints_element*nSpace+
               k*nSpace+
               I]
              +=
              V_dof[I][j]
              *
              qv[eN*nQuadraturePoints_element*nDOF_element+
                 k*nDOF_element+
                 j];


      /* evaluate at element boundary quadrature*/
      for (ebN=0;ebN<nElementBoundaries_element;ebN++)
        for(k=0;k<nElementBoundaryQuadraturePoints_elementBoundary;k++)
          for(j=0;j<nDOF_element;j++)
            for(I=0;I<nSpace;I++)
              V[eN*nElementBoundaries_element*nElementBoundaryQuadraturePoints_elementBoundary*nSpace+
                ebN*nElementBoundaryQuadraturePoints_elementBoundary*nSpace+
                k*nSpace+
                I]
                +=
                V_dof[I][j]
                *
                v[eN*nElementBoundaries_element*nElementBoundaryQuadraturePoints_elementBoundary*nDOF_element+
                  ebN*nElementBoundaryQuadraturePoints_elementBoundary*nDOF_element+
                  k*nDOF_element+
                  j];
    }
}


void calculateVelocityQuadrature_MixedForm_Jacobian(proteus_index_t nElements_global,
                                                    int nElementBoundaries_element,
                                                    int nElementBoundaryQuadraturePoints_elementBoundary,
                                                    int nDOF_element,
                                                    int nSpace,
                                                    int nQuadraturePoints_element,
                                                    double* A_inv,
                                                    double* db,
                                                    double* db_eb,
                                                    double* v,
                                                    double* DV,
                                                    double* DV_eb,
                                                    double* qv,
                                                    double* qDV,
                                                    double* qDV_eb)
{
  proteus_index_t eN, ebN, I;  int ebN_ebN, k, i, j, jj, nDOF_element2=nDOF_element*nDOF_element;
  double DV_dof[nSpace][nDOF_element][nDOF_element];
  memset(DV,0,sizeof(double)*
         nElements_global*
         nElementBoundaries_element*
         nElementBoundaryQuadraturePoints_elementBoundary*
         nDOF_element*
         nSpace);
  memset(DV_eb,0,sizeof(double)*
         nElements_global*
         nElementBoundaries_element*
         nElementBoundaries_element*
         nElementBoundaryQuadraturePoints_elementBoundary*
         nDOF_element*
         nSpace);
  memset(qDV,0,sizeof(double)*
         nElements_global*
         nQuadraturePoints_element*
         nDOF_element*
         nSpace);
  memset(qDV_eb,0,sizeof(double)*
         nElements_global*
         nElementBoundaries_element*
         nQuadraturePoints_element*
         nDOF_element*
         nSpace);
  for(eN=0;eN<nElements_global;eN++)
    {
      /* get derivatives of velocity DOF w.r.t. u DOF*/
      for(I=0;I<nSpace;I++)
        for(jj=0;jj<nDOF_element;jj++)
          for(i=0;i<nDOF_element;i++)
            {
              DV_dof[I][i][jj]=0.0;
              for(j=0;j<nDOF_element;j++)
                DV_dof[I][i][jj]
                  +=
                  A_inv[eN*nDOF_element2+
                        i*nDOF_element+
                        j]
                  *
                  db[eN*nSpace*nDOF_element2+
                     I*nDOF_element2+
                     j*nDOF_element+
                     jj];
            }
      /* get derivatives of velocity at element quadrature  w.r.t u DOF on element*/
      for(k=0;k<nQuadraturePoints_element;k++)
        for(j=0;j<nDOF_element;j++)
          for(jj=0;jj<nDOF_element;jj++)
            for(I=0;I<nSpace;I++)
              qDV[eN*nQuadraturePoints_element*nDOF_element*nSpace+
                  k*nDOF_element*nSpace+
                  jj*nSpace+
                  I]
                +=
                DV_dof[I][j][jj]
                *
                qv[eN*nQuadraturePoints_element*nDOF_element+
                   k*nDOF_element+
                   j];
      /* get derivatives of velocity at element boundary quadrature w.r.t. u DOF on element */
      for (ebN=0;ebN<nElementBoundaries_element;ebN++)
        for(k=0;k<nElementBoundaryQuadraturePoints_elementBoundary;k++)
          for(j=0;j<nDOF_element;j++)
            for(jj=0;jj<nDOF_element;jj++)
              for(I=0;I<nSpace;I++)
                DV[eN*nElementBoundaries_element*nElementBoundaryQuadraturePoints_elementBoundary*nDOF_element*nSpace+
                   ebN*nElementBoundaryQuadraturePoints_elementBoundary*nDOF_element*nSpace+
                   k*nDOF_element*nSpace+
                   jj*nSpace+
                   I]
                  +=
                  DV_dof[I][j][jj]
                  *
                  v[eN*nElementBoundaries_element*nElementBoundaryQuadraturePoints_elementBoundary*nDOF_element+
                    ebN*nElementBoundaryQuadraturePoints_elementBoundary*nDOF_element+
                    k*nDOF_element+
                    j];
      /* get derivatives at element neighbors */
      for (ebN=0;ebN<nElementBoundaries_element;ebN++)
        {
          /* DOF calculations */
          for(I=0;I<nSpace;I++)
            for(jj=0;jj<nDOF_element;jj++)
              for(i=0;i<nDOF_element;i++)
                {
                  DV_dof[I][i][jj] = 0.0;
                  for(j=0;j<nDOF_element;j++)
                    DV_dof[I][i][jj]
                      +=
                      A_inv[eN*nDOF_element2+
                            i*nDOF_element+
                            j]
                      *
                      db_eb[eN*nElementBoundaries_element*nSpace*nDOF_element2+
                            ebN*nSpace*nDOF_element2+
                            I*nDOF_element2+
                            j*nDOF_element+
                            jj];
                }
          /* quadrature calculations */
          for(k=0;k<nQuadraturePoints_element;k++)
            for(j=0;j<nDOF_element;j++)
              for(jj=0;jj<nDOF_element;jj++)
                for(I=0;I<nSpace;I++)
                  {
                    qDV_eb[eN*nElementBoundaries_element*nQuadraturePoints_element*nDOF_element*nSpace+
                           ebN*nQuadraturePoints_element*nDOF_element*nSpace+
                           k*nDOF_element*nSpace+
                           jj*nSpace+
                           I]
                      +=
                      DV_dof[I][j][jj]
                      *
                      qv[eN*nQuadraturePoints_element*nDOF_element+
                         k*nDOF_element+
                         j];
                  }
          for (ebN_ebN=0;ebN_ebN<nElementBoundaries_element;ebN_ebN++)
            {
              /* element boundary quadrature calculations */
              for(k=0;k<nElementBoundaryQuadraturePoints_elementBoundary;k++)
                for(j=0;j<nDOF_element;j++)
                  for(jj=0;jj<nDOF_element;jj++)
                    for(I=0;I<nSpace;I++)
                      {
                        DV_eb[eN*nElementBoundaries_element*nElementBoundaries_element*nElementBoundaryQuadraturePoints_elementBoundary*nDOF_element*nSpace+
                              ebN_ebN*nElementBoundaries_element*nElementBoundaryQuadraturePoints_elementBoundary*nDOF_element*nSpace+
                              ebN*nElementBoundaryQuadraturePoints_elementBoundary*nDOF_element*nSpace+
                              k*nDOF_element*nSpace+
                              jj*nSpace+
                              I]
                          +=
                          DV_dof[I][j][jj]
                          *
                          v[eN*nElementBoundaries_element*nElementBoundaryQuadraturePoints_elementBoundary*nDOF_element+
                            ebN_ebN*nElementBoundaryQuadraturePoints_elementBoundary*nDOF_element+
                            k*nDOF_element+
                            j];
                      }
            }
        }
    }
}
void calculateVelocityQuadrature_MixedForm2_Jacobian(proteus_index_t nElements_global,
                                                    int nElementBoundaries_element,
                                                    int nElementBoundaryQuadraturePoints_elementBoundary,
                                                    int nDOF_element,
                                                    int nSpace,
                                                    int nQuadraturePoints_element,
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
{
  proteus_index_t nSpace2=nSpace*nSpace;  proteus_index_t ebN, I; int eN, ebN_ebN, k, i, j, jj, nDOF_element2=nDOF_element*nDOF_element;
  PROTEUS_LAPACK_INTEGER ipiv[nDOF_element],lwork=((PROTEUS_LAPACK_INTEGER)nDOF_element),dim=((PROTEUS_LAPACK_INTEGER)nDOF_element),info=0;
  double work[nDOF_element],A_inv[nSpace][nDOF_element2];
  double DV_dof[nSpace][nDOF_element][nDOF_element];
  memset(DV,0,sizeof(double)*
         nElements_global*
         nElementBoundaries_element*
         nElementBoundaryQuadraturePoints_elementBoundary*
         nDOF_element*
         nSpace);
  memset(DV_eb,0,sizeof(double)*
         nElements_global*
         nElementBoundaries_element*
         nElementBoundaries_element*
         nElementBoundaryQuadraturePoints_elementBoundary*
         nDOF_element*
         nSpace);
  memset(qDV,0,sizeof(double)*
         nElements_global*
         nQuadraturePoints_element*
         nDOF_element*
         nSpace);
  memset(qDV_eb,0,sizeof(double)*
         nElements_global*
         nElementBoundaries_element*
         nQuadraturePoints_element*
         nDOF_element*
         nSpace);
  for(eN=0;eN<nElements_global;eN++)
    {
      for(I=0;I<nSpace;I++)
        {
          memset(A_inv[I],0,sizeof(double)*nDOF_element2);
          for(i=0;i<nDOF_element;i++)
            for(j=0;j<nDOF_element;j++)
              {
                for(k=0;k<nQuadraturePoints_element;k++)
                  {
                    //cek hack do diagonal only for now
                    A_inv[I][i*nDOF_element+j] += (1.0/qa[eN*nQuadraturePoints_element*nSpace2+
                                                          k*nSpace2+
                                                          I*nSpace+
                                                          I])
                      *qv[eN*nQuadraturePoints_element*nDOF_element+
                          k*nDOF_element+
                          j]
                      *
                      qw_dV[eN*nQuadraturePoints_element*nDOF_element+
                            k*nDOF_element+
                            i];
                  }
              }
          info=0;
          dgetrf_(&dim,&dim,A_inv[I],&dim,ipiv,&info);
          dgetri_(&dim,A_inv[I],&dim,ipiv,work,&lwork,&info);
          /* get derivatives of velocity DOF w.r.t. u DOF*/
          for(jj=0;jj<nDOF_element;jj++)
            for(i=0;i<nDOF_element;i++)
              {
                DV_dof[I][i][jj]=0.0;
                for(j=0;j<nDOF_element;j++)
                  DV_dof[I][i][jj]
                    +=
                    A_inv[I][i*nDOF_element+
                             j]
                    *
                    db[eN*nSpace*nDOF_element2+
                       I*nDOF_element2+
                       j*nDOF_element+
                       jj];
              }
        }
      /* get derivatives of velocity at element quadrature  w.r.t u DOF on element*/
      for(k=0;k<nQuadraturePoints_element;k++)
        for(j=0;j<nDOF_element;j++)
          for(jj=0;jj<nDOF_element;jj++)
            for(I=0;I<nSpace;I++)
              qDV[eN*nQuadraturePoints_element*nDOF_element*nSpace+
                  k*nDOF_element*nSpace+
                  jj*nSpace+
                  I]
                +=
                DV_dof[I][j][jj]
                *
                qv[eN*nQuadraturePoints_element*nDOF_element+
                   k*nDOF_element+
                   j];
      /* get derivatives of velocity at element boundary quadrature w.r.t. u DOF on element */
      for (ebN=0;ebN<nElementBoundaries_element;ebN++)
        for(k=0;k<nElementBoundaryQuadraturePoints_elementBoundary;k++)
          for(j=0;j<nDOF_element;j++)
            for(jj=0;jj<nDOF_element;jj++)
              for(I=0;I<nSpace;I++)
                DV[eN*nElementBoundaries_element*nElementBoundaryQuadraturePoints_elementBoundary*nDOF_element*nSpace+
                   ebN*nElementBoundaryQuadraturePoints_elementBoundary*nDOF_element*nSpace+
                   k*nDOF_element*nSpace+
                   jj*nSpace+
                   I]
                  +=
                  DV_dof[I][j][jj]
                  *
                  v[eN*nElementBoundaries_element*nElementBoundaryQuadraturePoints_elementBoundary*nDOF_element+
                    ebN*nElementBoundaryQuadraturePoints_elementBoundary*nDOF_element+
                    k*nDOF_element+
                    j];
      /* get derivatives at element neighbors */
      for (ebN=0;ebN<nElementBoundaries_element;ebN++)
        {
          /* DOF calculations */
          for(I=0;I<nSpace;I++)
            for(jj=0;jj<nDOF_element;jj++)
              for(i=0;i<nDOF_element;i++)
                {
                  DV_dof[I][i][jj] = 0.0;
                  for(j=0;j<nDOF_element;j++)
                    DV_dof[I][i][jj]
                      +=
                      A_inv[I][i*nDOF_element+
                               j]
                      *
                      db_eb[eN*nElementBoundaries_element*nSpace*nDOF_element2+
                            ebN*nSpace*nDOF_element2+
                            I*nDOF_element2+
                            j*nDOF_element+
                            jj];
                }
          /* quadrature calculations */
          for(k=0;k<nQuadraturePoints_element;k++)
            for(j=0;j<nDOF_element;j++)
              for(jj=0;jj<nDOF_element;jj++)
                for(I=0;I<nSpace;I++)
                  {
                    qDV_eb[eN*nElementBoundaries_element*nQuadraturePoints_element*nDOF_element*nSpace+
                           ebN*nQuadraturePoints_element*nDOF_element*nSpace+
                           k*nDOF_element*nSpace+
                           jj*nSpace+
                           I]
                      +=
                      DV_dof[I][j][jj]
                      *
                      qv[eN*nQuadraturePoints_element*nDOF_element+
                         k*nDOF_element+
                         j];
                  }
          for (ebN_ebN=0;ebN_ebN<nElementBoundaries_element;ebN_ebN++)
            {
              /* element boundary quadrature calculations */
              for(k=0;k<nElementBoundaryQuadraturePoints_elementBoundary;k++)
                for(j=0;j<nDOF_element;j++)
                  for(jj=0;jj<nDOF_element;jj++)
                    for(I=0;I<nSpace;I++)
                      {
                        DV_eb[eN*nElementBoundaries_element*nElementBoundaries_element*nElementBoundaryQuadraturePoints_elementBoundary*nDOF_element*nSpace+
                              ebN_ebN*nElementBoundaries_element*nElementBoundaryQuadraturePoints_elementBoundary*nDOF_element*nSpace+
                              ebN*nElementBoundaryQuadraturePoints_elementBoundary*nDOF_element*nSpace+
                              k*nDOF_element*nSpace+
                              jj*nSpace+
                              I]
                          +=
                          DV_dof[I][j][jj]
                          *
                          v[eN*nElementBoundaries_element*nElementBoundaryQuadraturePoints_elementBoundary*nDOF_element+
                            ebN_ebN*nElementBoundaryQuadraturePoints_elementBoundary*nDOF_element+
                            k*nDOF_element+
                            j];
                      }
            }
        }
    }
}

void calculateVelocityQuadrature_MixedForm2_Jacobian_sd(proteus_index_t nElements_global,
                                                        int nElementBoundaries_element,
                                                        int nElementBoundaryQuadraturePoints_elementBoundary,
                                                        int nDOF_element,
                                                        int nSpace,
                                                        int nQuadraturePoints_element,
                                                        const proteus_index_t *rowptr,
                                                        const proteus_index_t *colind,
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
{
  proteus_index_t nSpace2=nSpace*nSpace;  proteus_index_t ebN, I; int eN, ebN_ebN, k, i, j, jj, nDOF_element2=nDOF_element*nDOF_element;
  proteus_index_t m, nnz=rowptr[nSpace];
  PROTEUS_LAPACK_INTEGER ipiv[nDOF_element],lwork=((PROTEUS_LAPACK_INTEGER)nDOF_element),dim=((PROTEUS_LAPACK_INTEGER)nDOF_element),info=0;
  double work[nDOF_element],A_inv[nSpace][nDOF_element2];
  double DV_dof[nSpace][nDOF_element][nDOF_element];
  memset(DV,0,sizeof(double)*
         nElements_global*
         nElementBoundaries_element*
         nElementBoundaryQuadraturePoints_elementBoundary*
         nDOF_element*
         nSpace);
  memset(DV_eb,0,sizeof(double)*
         nElements_global*
         nElementBoundaries_element*
         nElementBoundaries_element*
         nElementBoundaryQuadraturePoints_elementBoundary*
         nDOF_element*
         nSpace);
  memset(qDV,0,sizeof(double)*
         nElements_global*
         nQuadraturePoints_element*
         nDOF_element*
         nSpace);
  memset(qDV_eb,0,sizeof(double)*
         nElements_global*
         nElementBoundaries_element*
         nQuadraturePoints_element*
         nDOF_element*
         nSpace);
  for(eN=0;eN<nElements_global;eN++)
    {
      for(I=0;I<nSpace;I++)
        {
          memset(A_inv[I],0,sizeof(double)*nDOF_element2);
          for(i=0;i<nDOF_element;i++)
            for(j=0;j<nDOF_element;j++)
              {
                for(k=0;k<nQuadraturePoints_element;k++)
                  {
                    //cek hack do diagonal only for now
                    for (m=rowptr[I]; m < rowptr[I+1]; m++)
                      if (colind[m] == I)
                        {
                          A_inv[I][i*nDOF_element+j] += (1.0/qa[eN*nQuadraturePoints_element*nnz+
                                                             k*nnz+m])
                            *qv[eN*nQuadraturePoints_element*nDOF_element+
                                k*nDOF_element+
                                j]
                            *
                            qw_dV[eN*nQuadraturePoints_element*nDOF_element+
                                  k*nDOF_element+
                                  i];
                        }
                  }
              }
          info=0;
          dgetrf_(&dim,&dim,A_inv[I],&dim,ipiv,&info);
          dgetri_(&dim,A_inv[I],&dim,ipiv,work,&lwork,&info);
          /* get derivatives of velocity DOF w.r.t. u DOF*/
          for(jj=0;jj<nDOF_element;jj++)
            {
              for(i=0;i<nDOF_element;i++)
                {
                  DV_dof[I][i][jj]=0.0;
                  for(j=0;j<nDOF_element;j++)
                    DV_dof[I][i][jj]
                      +=
                      A_inv[I][i*nDOF_element+
                            j]
                      *
                      db[eN*nSpace*nDOF_element2+
                         I*nDOF_element2+
                         j*nDOF_element+
                         jj];
                }
            }
        }
      for(k=0;k<nQuadraturePoints_element;k++)
        for(j=0;j<nDOF_element;j++)
          for(jj=0;jj<nDOF_element;jj++)
            for(I=0;I<nSpace;I++)
              qDV[eN*nQuadraturePoints_element*nDOF_element*nSpace+
                  k*nDOF_element*nSpace+
                  jj*nSpace+
                  I]
                +=
                DV_dof[I][j][jj]
                *
                qv[eN*nQuadraturePoints_element*nDOF_element+
                   k*nDOF_element+
                   j];
      /* get derivatives of velocity at element boundary quadrature w.r.t. u DOF on element */
      for (ebN=0;ebN<nElementBoundaries_element;ebN++)
        for(k=0;k<nElementBoundaryQuadraturePoints_elementBoundary;k++)
          for(j=0;j<nDOF_element;j++)
            for(jj=0;jj<nDOF_element;jj++)
              for(I=0;I<nSpace;I++)
                DV[eN*nElementBoundaries_element*nElementBoundaryQuadraturePoints_elementBoundary*nDOF_element*nSpace+
                   ebN*nElementBoundaryQuadraturePoints_elementBoundary*nDOF_element*nSpace+
                   k*nDOF_element*nSpace+
                   jj*nSpace+
                   I]
                  +=
                  DV_dof[I][j][jj]
                  *
                  v[eN*nElementBoundaries_element*nElementBoundaryQuadraturePoints_elementBoundary*nDOF_element+
                    ebN*nElementBoundaryQuadraturePoints_elementBoundary*nDOF_element+
                    k*nDOF_element+
                    j];
      /* get derivatives at element neighbors */
      for (ebN=0;ebN<nElementBoundaries_element;ebN++)
        {
          /* DOF calculations */
          for(I=0;I<nSpace;I++)
            for(jj=0;jj<nDOF_element;jj++)
              {
                for(i=0;i<nDOF_element;i++)
                  {
                    DV_dof[I][i][jj] = 0.0;
                    for(j=0;j<nDOF_element;j++)
                      DV_dof[I][i][jj]
                        +=
                        A_inv[I][i*nDOF_element+
                              j]
                        *
                        db_eb[eN*nElementBoundaries_element*nSpace*nDOF_element2+
                              ebN*nSpace*nDOF_element2+
                              I*nDOF_element2+
                              j*nDOF_element+
                              jj];
                  }
              }
          /* quadrature calculations */
          for(k=0;k<nQuadraturePoints_element;k++)
            for(j=0;j<nDOF_element;j++)
              for(jj=0;jj<nDOF_element;jj++)
                for(I=0;I<nSpace;I++)
                  {
                    qDV_eb[eN*nElementBoundaries_element*nQuadraturePoints_element*nDOF_element*nSpace+
                           ebN*nQuadraturePoints_element*nDOF_element*nSpace+
                           k*nDOF_element*nSpace+
                           jj*nSpace+
                           I]
                      +=
                      DV_dof[I][j][jj]
                      *
                      qv[eN*nQuadraturePoints_element*nDOF_element+
                         k*nDOF_element+
                         j];
                  }
          for (ebN_ebN=0;ebN_ebN<nElementBoundaries_element;ebN_ebN++)
            {
              /* element boundary quadrature calculations */
              for(k=0;k<nElementBoundaryQuadraturePoints_elementBoundary;k++)
                for(j=0;j<nDOF_element;j++)
                  for(jj=0;jj<nDOF_element;jj++)
                    for(I=0;I<nSpace;I++)
                      {
                        DV_eb[eN*nElementBoundaries_element*nElementBoundaries_element*nElementBoundaryQuadraturePoints_elementBoundary*nDOF_element*nSpace+
                              ebN_ebN*nElementBoundaries_element*nElementBoundaryQuadraturePoints_elementBoundary*nDOF_element*nSpace+
                              ebN*nElementBoundaryQuadraturePoints_elementBoundary*nDOF_element*nSpace+
                              k*nDOF_element*nSpace+
                              jj*nSpace+
                              I]
                          +=
                          DV_dof[I][j][jj]
                          *
                          v[eN*nElementBoundaries_element*nElementBoundaryQuadraturePoints_elementBoundary*nDOF_element+
                            ebN_ebN*nElementBoundaryQuadraturePoints_elementBoundary*nDOF_element+
                            k*nDOF_element+
                            j];
                      }
            }
        }
    }
}



void calculateVelocityProjectionMatrixLDG(proteus_index_t nElements_global,
                                          int nQuadraturePoints_element,
                                          int nDOF_element,
                                          double* vXw_dV,
                                          double* A_inv)
{
  proteus_index_t eN;  int i, j, k, nDOF_element2=nDOF_element*nDOF_element;
  PROTEUS_LAPACK_INTEGER ipiv[nDOF_element],lwork=((PROTEUS_LAPACK_INTEGER)nDOF_element),dim=((PROTEUS_LAPACK_INTEGER)nDOF_element),info=0;
  double work[nDOF_element];
  memset(A_inv,0,sizeof(double)*nElements_global*nDOF_element2);
  for(eN=0;eN<nElements_global;eN++)
    {
      for(i=0;i<nDOF_element;i++)
        for(j=0;j<nDOF_element;j++)
          {
            for(k=0;k<nQuadraturePoints_element;k++)
              {
                A_inv[eN*nDOF_element2+i*nDOF_element+j] += vXw_dV[eN*nQuadraturePoints_element*nDOF_element2+
                                                                   k*nDOF_element2+
                                                                   j*nDOF_element+
                                                                   i];
              }
          }
      dgetrf_(&dim,&dim,&A_inv[eN*nDOF_element2],&dim,ipiv,&info);
      dgetri_(&dim,&A_inv[eN*nDOF_element2],&dim,ipiv,work,&lwork,&info);
    }
}

void updateDiffusion_MixedForm_weak(proteus_index_t nElements_global,
                                    int nQuadraturePoints_element,
                                    int nDOF_test_element,
                                    int nSpace,
                                    double* a,
                                    double* qV,
                                    double* grad_w_dV,
                                    double* weak_residual)
{
  proteus_index_t nSpace2=nSpace*nSpace;  proteus_index_t I, J; int eN, i, k;
  for(eN=0;eN<nElements_global;eN++)
    for (i=0;i<nDOF_test_element;i++)
      for (k=0;k<nQuadraturePoints_element;k++)
        for (I=0;I<nSpace;I++)
          for (J=0;J<nSpace;J++)
            weak_residual[eN*nDOF_test_element + i]
              -=
              a[eN*nQuadraturePoints_element*nSpace2 +
                k*nSpace2 +
                I*nSpace +
                J]
              *
              qV[eN*nQuadraturePoints_element*nSpace +
                 k*nSpace +
                 J]
              *
              grad_w_dV[eN*nQuadraturePoints_element*nDOF_test_element*nSpace +
                        k*nDOF_test_element*nSpace +
                        i*nSpace+
                        I];
}

void updateDiffusion_MixedForm_weak_sd(proteus_index_t nElements_global,
                                       int nQuadraturePoints_element,
                                       int nDOF_test_element,
                                       int nSpace,
                                       int rho_split,
                                       proteus_index_t* rowptr,
                                       proteus_index_t* colind,
                                       double* a,
                                       double* qV,
                                       double* grad_w_dV,
                                       double* velocity, /* added for ldg coupling */
                                       double* weak_residual)
{
  proteus_index_t m, nnz=rowptr[nSpace];  proteus_index_t I; int eN, i, k;
  for(eN=0;eN<nElements_global;eN++)
    for (i=0;i<nDOF_test_element;i++)
      for (k=0;k<nQuadraturePoints_element;k++)
        for (I=0;I<nSpace;I++)
          for (m=rowptr[I];m<rowptr[I+1];m++)
            {
            weak_residual[eN*nDOF_test_element + i]
              -=
             a[eN*nQuadraturePoints_element*nnz+
                k*nnz+
                m]
              *
              qV[eN*nQuadraturePoints_element*nSpace +
                 k*nSpace +
                 colind[m]]
              *
              grad_w_dV[eN*nQuadraturePoints_element*nDOF_test_element*nSpace +
                        k*nDOF_test_element*nSpace +
                        i*nSpace+
                        I];
            }

  for(eN=0;eN<nElements_global;eN++)
      for (k=0;k<nQuadraturePoints_element;k++)
        for (I=0;I<nSpace;I++)
          if (rho_split==0)
            {
              for (m=rowptr[I];m<rowptr[I+1];m++)
                {
                  velocity[eN*nQuadraturePoints_element*nSpace +
                           k*nSpace + I]
                    += a[eN*nQuadraturePoints_element*nnz+
                         k*nnz+ m]
                    *
                    qV[eN*nQuadraturePoints_element*nSpace +
                       k*nSpace + colind[m]];
                }
            }
          else if (rho_split==1)
            {
              velocity[eN*nQuadraturePoints_element*nSpace +
                           k*nSpace + I]
                    =
                    qV[eN*nQuadraturePoints_element*nSpace +
                       k*nSpace + I];
            }
}

void updateDiffusionJacobian_MixedForm_weak(proteus_index_t nElements_global,
                                            int nElementBoundaries_element,
                                            int nQuadraturePoints_element,
                                            int nDOF_trial_element,
                                            int nDOF_test_element,
                                            int nSpace,
                                            double* a,
                                            double* da,
                                            double* qV,
                                            double* qDV,
                                            double* qDV_eb,
                                            double* grad_w_dV,
                                            double* v,
                                            double* jacobian_weak_residual,
                                            double* jacobian_weak_residual_eb)
{
  proteus_index_t nSpace2=nSpace*nSpace, nDOF_test_X_trial_element=nDOF_test_element*nDOF_trial_element;  proteus_index_t ebN, I, J; int eN, i, j, k;
  double daProduct,dphiProduct;
  for(eN=0;eN<nElements_global;eN++)
    {
      for (i=0;i<nDOF_test_element;i++)
        for (k=0;k<nQuadraturePoints_element;k++)
          {
            daProduct=0.0;
            for (I=0;I<nSpace;I++)
              for (J=0;J<nSpace;J++)
                daProduct
                  -=
                  da[eN*nQuadraturePoints_element*nSpace2 +
                     k*nSpace2 +
                     I*nSpace +
                     J]
                  *
                  qV[eN*nQuadraturePoints_element*nSpace +
                     k*nSpace +
                     J]
                  *
                  grad_w_dV[eN*nQuadraturePoints_element*nDOF_test_element*nSpace +
                            k*nDOF_test_element*nSpace +
                            i*nSpace+
                            I];
            for (j=0;j<nDOF_trial_element;j++)
              {
                dphiProduct=0.0;
                for (I=0;I<nSpace;I++)
                  for (J=0;J<nSpace;J++)
                    dphiProduct
                      -=
                      a[eN*nQuadraturePoints_element*nSpace2 +
                        k*nSpace2+
                        I*nSpace +
                        J]
                      *
                      qDV[eN*nQuadraturePoints_element*nDOF_trial_element*nSpace +
                          k*nDOF_trial_element*nSpace +
                          j*nSpace+
                          J]
                      *
                      grad_w_dV[eN*nQuadraturePoints_element*nDOF_test_element*nSpace +
                                k*nDOF_test_element*nSpace +
                                i*nSpace+
                                I];
                jacobian_weak_residual[eN*nDOF_test_X_trial_element +
                                       i*nDOF_trial_element +
                                       j]
                  +=
                  daProduct
                  *
                  v[eN*nQuadraturePoints_element*nDOF_trial_element+
                    k*nDOF_trial_element+
                    j]
                  +
                  dphiProduct;
              }
          }
      for (ebN=0;ebN<nElementBoundaries_element;ebN++)
        for (i=0;i<nDOF_test_element;i++)
          for (k=0;k<nQuadraturePoints_element;k++)
            {
              for (j=0;j<nDOF_trial_element;j++)
                {
                  dphiProduct=0.0;
                  for (I=0;I<nSpace;I++)
                    for (J=0;J<nSpace;J++)
                      dphiProduct
                        -=
                        a[eN*nQuadraturePoints_element*nSpace2 +
                          k*nSpace2+
                          I*nSpace +
                          J]
                        *
                        qDV_eb[eN*nElementBoundaries_element*nQuadraturePoints_element*nDOF_trial_element*nSpace +
                               ebN*nQuadraturePoints_element*nDOF_trial_element*nSpace+
                               k*nDOF_trial_element*nSpace +
                               j*nSpace+
                               J]
                        *
                        grad_w_dV[eN*nQuadraturePoints_element*nDOF_test_element*nSpace +
                                  k*nDOF_test_element*nSpace +
                                  i*nSpace+
                                  I];
                  jacobian_weak_residual_eb[eN*nElementBoundaries_element*nDOF_test_X_trial_element +
                                            ebN*nDOF_test_X_trial_element +
                                            i*nDOF_trial_element +
                                            j]
                    +=
                    dphiProduct;
                }
            }
    }
}

void updateDiffusionJacobian_MixedForm_weak_sd(proteus_index_t nElements_global,
                                               int nElementBoundaries_element,
                                               int nQuadraturePoints_element,
                                               int nDOF_trial_element,
                                               int nDOF_test_element,
                                               int nSpace,
                                               proteus_index_t* rowptr,
                                               proteus_index_t* colind,
                                               double* a,
                                               double* da,
                                               double* qV,
                                               double* qDV,
                                               double* qDV_eb,
                                               double* grad_w_dV,
                                               double* v,
                                               double* jacobian_weak_residual,
                                               double* jacobian_weak_residual_eb)
{
  proteus_index_t m, nDOF_test_X_trial_element=nDOF_test_element*nDOF_trial_element, nnz=rowptr[nSpace];  proteus_index_t ebN, I; int eN, i, j, k;
  double daProduct,dphiProduct;
  for(eN=0;eN<nElements_global;eN++)
    {
      for (i=0;i<nDOF_test_element;i++)
        for (k=0;k<nQuadraturePoints_element;k++)
          {
            daProduct=0.0;
            for (I=0;I<nSpace;I++)
              for (m=rowptr[I];m<rowptr[I+1];m++)
                daProduct
                  -=
                  da[eN*nQuadraturePoints_element*nnz+
                     k*nnz+
                     m]
                  *
                  qV[eN*nQuadraturePoints_element*nSpace +
                     k*nSpace +
                     colind[m]]
                  *
                  grad_w_dV[eN*nQuadraturePoints_element*nDOF_test_element*nSpace +
                            k*nDOF_test_element*nSpace +
                            i*nSpace+
                            I];
            for (j=0;j<nDOF_trial_element;j++)
              {
                dphiProduct=0.0;
                for (I=0;I<nSpace;I++)
                  for(m=rowptr[I];m<rowptr[I+1];m++)
                    dphiProduct
                      -=
                      a[eN*nQuadraturePoints_element*nnz+
                        k*nnz+
                        m]
                      *
                      qDV[eN*nQuadraturePoints_element*nDOF_trial_element*nSpace +
                          k*nDOF_trial_element*nSpace +
                          j*nSpace+
                          colind[m]]
                      *
                      grad_w_dV[eN*nQuadraturePoints_element*nDOF_test_element*nSpace +
                                k*nDOF_test_element*nSpace +
                                i*nSpace+
                                I];
                jacobian_weak_residual[eN*nDOF_test_X_trial_element +
                                       i*nDOF_trial_element +
                                       j]
                  +=
                  daProduct
                  *
                  v[eN*nQuadraturePoints_element*nDOF_trial_element+
                    k*nDOF_trial_element+
                    j]
                  +
                  dphiProduct;
              }
          }
      for (ebN=0;ebN<nElementBoundaries_element;ebN++)
        for (i=0;i<nDOF_test_element;i++)
          for (k=0;k<nQuadraturePoints_element;k++)
            {
              for (j=0;j<nDOF_trial_element;j++)
                {
                  dphiProduct=0.0;
                  for (I=0;I<nSpace;I++)
                    for(m=rowptr[I];m<rowptr[I+1];m++)
                      dphiProduct
                        -=
                        a[eN*nQuadraturePoints_element*nnz+
                          k*nnz+
                          m]
                        *
                        qDV_eb[eN*nElementBoundaries_element*nQuadraturePoints_element*nDOF_trial_element*nSpace +
                               ebN*nQuadraturePoints_element*nDOF_trial_element*nSpace+
                               k*nDOF_trial_element*nSpace +
                               j*nSpace+
                               colind[m]]
                        *
                        grad_w_dV[eN*nQuadraturePoints_element*nDOF_test_element*nSpace +
                                  k*nDOF_test_element*nSpace +
                                  i*nSpace+
                                  I];
                  jacobian_weak_residual_eb[eN*nElementBoundaries_element*nDOF_test_X_trial_element +
                                            ebN*nDOF_test_X_trial_element +
                                            i*nDOF_trial_element +
                                            j]
                    +=
                    dphiProduct;
                }
            }
    }
}

void estimate_mt(proteus_index_t nElements_global,
                 int nQuadraturePoints_element,
                 int nDOF_element,
                 double* v,
                 double* vXw_dV,
                 double* elementSpatialResidual,
                 double* mt)
{
  proteus_index_t eN;  int i, j, k, nDOF_element2=nDOF_element*nDOF_element;
  char trans='N';
  PROTEUS_LAPACK_INTEGER nrhs=1,dim=((PROTEUS_LAPACK_INTEGER)nDOF_element),info=0,ipiv[nDOF_element];
  double massMatrix[nDOF_element2],b[nDOF_element];
  /* loop over elements and solve for the local polynomial \pi(mt) such that (\pi(mt),w) = - elementSpatialResidual = b(u,w) -a(u,w) */
  /* then evaluate \pi(mt) at the quadrature points to recover the mt estimate */
  memset(mt,0,sizeof(double)*nElements_global*nQuadraturePoints_element);
  for(eN=0;eN<nElements_global;eN++)
    {
      memset(massMatrix,0,sizeof(double)*nDOF_element2);
      memset(ipiv,0,sizeof(PROTEUS_LAPACK_INTEGER)*nDOF_element);
      for(i=0;i<nDOF_element;i++)
        for(j=0;j<nDOF_element;j++)
          {
            for(k=0;k<nQuadraturePoints_element;k++)
              {
                mt[eN*nQuadraturePoints_element+
                   k] = 0.0;
                massMatrix[i*nDOF_element+j] += vXw_dV[eN*nQuadraturePoints_element*nDOF_element2+
                                                       k*nDOF_element2+
                                                       j*nDOF_element+
                                                       i];
              }
            b[j] = -elementSpatialResidual[eN*nDOF_element+j];
          }
      dgetrf_(&dim,&dim,massMatrix,&dim,ipiv,&info);
      dgetrs_(&trans,&dim,&nrhs,massMatrix,&dim,ipiv,b,&dim,&info);
      for (j=0;j<nDOF_element;j++)
        for(k=0;k<nQuadraturePoints_element;k++)
          mt[eN*nQuadraturePoints_element+
             k] += b[j]*v[eN*nQuadraturePoints_element*nDOF_element+
                          k*nDOF_element+
                          j];
    }
}
void estimate_mt_lowmem(proteus_index_t nElements_global,
                        int nQuadraturePoints_element,
                        int nDOF_element,
                        double* v,
                        double* w_dV,
                        double* elementSpatialResidual,
                        double* mt)
{
  proteus_index_t eN;  int i, j, k, nDOF_element2=nDOF_element*nDOF_element;
  char trans='N';
  PROTEUS_LAPACK_INTEGER nrhs=1,dim=((PROTEUS_LAPACK_INTEGER)nDOF_element),info=0,ipiv[nDOF_element];
  double massMatrix[nDOF_element2],b[nDOF_element];
  /* loop over elements and solve for the local polynomial \pi(mt) such that (\pi(mt),w) = - elementSpatialResidual = b(u,w) -a(u,w) */
  /* then evaluate \pi(mt) at the quadrature points to recover the mt estimate */
  for(eN=0;eN<nElements_global;eN++)
    {
      memset(massMatrix,0,sizeof(double)*nDOF_element2);
      memset(ipiv,0,sizeof(PROTEUS_LAPACK_INTEGER)*nDOF_element);
      for(i=0;i<nDOF_element;i++)
        for(j=0;j<nDOF_element;j++)
          {
            for(k=0;k<nQuadraturePoints_element;k++)
              {
                mt[eN*nQuadraturePoints_element+
                   k] = 0.0;
                massMatrix[i*nDOF_element+j] +=
                  v[eN*nQuadraturePoints_element*nDOF_element+
                    k*nDOF_element+
                    j]
                  *w_dV[eN*nQuadraturePoints_element*nDOF_element+
                        k*nDOF_element+
                        i];
              }
            b[j] = -elementSpatialResidual[eN*nDOF_element+j];
          }
      dgetrf_(&dim,&dim,massMatrix,&dim,ipiv,&info);
      dgetrs_(&trans,&dim,&nrhs,massMatrix,&dim,ipiv,b,&dim,&info);
      for (j=0;j<nDOF_element;j++)
        for(k=0;k<nQuadraturePoints_element;k++)
          mt[eN*nQuadraturePoints_element+
             k] += b[j]*v[eN*nQuadraturePoints_element*nDOF_element+
                          k*nDOF_element+
                          j];
    }
}
/**
\brief copy quantity in an elementBoundary quadrature array to one that sits only on exterior boundaries
*/
void copyExteriorElementBoundaryValuesFromElementBoundaryValues(proteus_index_t nExteriorElementBoundaries_global,
                                                                proteus_index_t nElements_global,
                                                                int nElementBoundaries_element,
                                                                int nQuadraturePoints_elementBoundary,
                                                                int nValuesPerQuadraturePoint,
                                                                const proteus_index_t * exteriorElementBoundaries,
                                                                const proteus_index_t* elementBoundaryElements,
                                                                const int * elementBoundaryLocalElementBoundaries,
                                                                const double * ebq_val,
                                                                double * ebqe_val)
{
  proteus_index_t ebN, eN;  int ebNE, ebN_element, k, i;
  for (ebNE = 0; ebNE < nExteriorElementBoundaries_global; ebNE++)
    {
      ebN = exteriorElementBoundaries[ebNE];
      eN  = elementBoundaryElements[ebN*2+0];
      ebN_element = elementBoundaryLocalElementBoundaries[ebN*2+0];
      for (k = 0; k < nQuadraturePoints_elementBoundary; k++)
        for (i=0; i < nValuesPerQuadraturePoint; i++)
          {
            ebqe_val[ebNE*nQuadraturePoints_elementBoundary*nValuesPerQuadraturePoint +
                   k*nValuesPerQuadraturePoint + i]
              =
              ebq_val[eN*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nValuesPerQuadraturePoint+
                      ebN_element*nQuadraturePoints_elementBoundary*nValuesPerQuadraturePoint+
                      k*nValuesPerQuadraturePoint+
                      i];
          }/*i,k*/
    }/*ebNE*/
}
/**
\brief copy quantity that sits only on exterior boundaries into an elementBoundary quadrature array
*/
void copyExteriorElementBoundaryValuesToElementBoundaryValues(proteus_index_t nExteriorElementBoundaries_global,
                                                              proteus_index_t nElements_global,
                                                              int nElementBoundaries_element,
                                                              int nQuadraturePoints_elementBoundary,
                                                              int nValuesPerQuadraturePoint,
                                                              const proteus_index_t * exteriorElementBoundaries,
                                                              const proteus_index_t* elementBoundaryElements,
                                                              const int * elementBoundaryLocalElementBoundaries,
                                                              const double * ebqe_val,
                                                              double * ebq_val)
{
  proteus_index_t ebN, eN;  int ebNE, ebN_element, k, i;
  for (ebNE = 0; ebNE < nExteriorElementBoundaries_global; ebNE++)
    {
      ebN = exteriorElementBoundaries[ebNE];
      eN  = elementBoundaryElements[ebN*2+0];
      ebN_element = elementBoundaryLocalElementBoundaries[ebN*2+0];
      for (k = 0; k < nQuadraturePoints_elementBoundary; k++)
        for (i=0; i < nValuesPerQuadraturePoint; i++)
          {
            ebq_val[eN*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nValuesPerQuadraturePoint+
                    ebN_element*nQuadraturePoints_elementBoundary*nValuesPerQuadraturePoint+
                    k*nValuesPerQuadraturePoint+
                    i]
              =
              ebqe_val[ebNE*nQuadraturePoints_elementBoundary*nValuesPerQuadraturePoint +
                       k*nValuesPerQuadraturePoint + i];
          }/*i,k*/
    }/*ebNE*/
}
/**
\brief copy quantity that sits only on exterior boundaries into a global elementBoundary quadrature array
*/
void copyExteriorElementBoundaryValuesToGlobalElementBoundaryValues(proteus_index_t nExteriorElementBoundaries_global,
                                                                    int nQuadraturePoints_elementBoundary,
                                                                    int nValuesPerQuadraturePoint,
                                                                    const proteus_index_t * exteriorElementBoundaries,
                                                                    const proteus_index_t* elementBoundaryElements,
                                                                    const int * elementBoundaryLocalElementBoundaries,
                                                                    const double * ebqe_val,
                                                                    double * ebq_global_val)
{
  proteus_index_t ebN, eN;  int ebNE, k, i;
  for (ebNE = 0; ebNE < nExteriorElementBoundaries_global; ebNE++)
    {
      ebN = exteriorElementBoundaries[ebNE];
      eN  = elementBoundaryElements[ebN*2+0];
      for (k = 0; k < nQuadraturePoints_elementBoundary; k++)
        for (i=0; i < nValuesPerQuadraturePoint; i++)
          {
            ebq_global_val[ebN*nQuadraturePoints_elementBoundary*nValuesPerQuadraturePoint+
                           k*nValuesPerQuadraturePoint+
                           i]
              =
              ebqe_val[ebNE*nQuadraturePoints_elementBoundary*nValuesPerQuadraturePoint +
                       k*nValuesPerQuadraturePoint + i];
          }/*i,k*/
    }/*ebNE*/
}

/**
\brief copy quantity that sits only on exterior boundaries from a global elementBoundary quadrature array
*/
void copyExteriorElementBoundaryValuesFromGlobalElementBoundaryValues(proteus_index_t nExteriorElementBoundaries_global,
                                                                      int nQuadraturePoints_elementBoundary,
                                                                      int nValuesPerQuadraturePoint,
                                                                      const proteus_index_t * exteriorElementBoundaries,
                                                                      const proteus_index_t* elementBoundaryElements,
                                                                      const int * elementBoundaryLocalElementBoundaries,
                                                                      const double * ebq_global_val,
                                                                      double * ebqe_val)
{
  proteus_index_t ebN, eN;  int ebNE, k, i;
  for (ebNE = 0; ebNE < nExteriorElementBoundaries_global; ebNE++)
    {
      ebN = exteriorElementBoundaries[ebNE];
      eN  = elementBoundaryElements[ebN*2+0];
      for (k = 0; k < nQuadraturePoints_elementBoundary; k++)
        for (i=0; i < nValuesPerQuadraturePoint; i++)
          {
            ebqe_val[ebNE*nQuadraturePoints_elementBoundary*nValuesPerQuadraturePoint +
                     k*nValuesPerQuadraturePoint + i]
              =
              ebq_global_val[ebN*nQuadraturePoints_elementBoundary*nValuesPerQuadraturePoint+
                             k*nValuesPerQuadraturePoint+
                             i];
          }/*i,k*/
    }/*ebNE*/
}

double scalarDomainIntegral(proteus_index_t nElements,
                            int nQuadraturePoints_element,
                            double* dV,
                            double* nValueArray)
{
  proteus_index_t eN;  int k;
  register double integral=0.0;
  for(eN=0;eN<nElements;eN++)
    for(k=0;k<nQuadraturePoints_element;k++)
      {
        integral += nValueArray[eN*nQuadraturePoints_element + k]*
          dV[eN*nQuadraturePoints_element+k];
      }
  return integral;
}

double scalarHeavisideDomainIntegral(proteus_index_t nElements,
                                     int nQuadraturePoints_element,
                                     double* dV,
                                     double* nValueArray)
{
  proteus_index_t eN;  int k;
  register double integral=0.0;
  for(eN=0;eN<nElements;eN++)
    for(k=0;k<nQuadraturePoints_element;k++)
      {
        if(nValueArray[eN*nQuadraturePoints_element + k] > 0.0)
          integral += dV[eN*nQuadraturePoints_element+k];
        else if (nValueArray[eN*nQuadraturePoints_element + k] == 0.0)
          integral += 0.5*dV[eN*nQuadraturePoints_element+k];
      }
  return integral;
}

double scalarSmoothedHeavisideDomainIntegral(proteus_index_t nElements,
                                             int nQuadraturePoints_element,
                                             double epsFact,
                                             double* elementDiameter,
                                             double* dV,
                                             double* nValueArray)
{
  proteus_index_t eN;  int k;
  register double integral=0.0,eps,H,phi;
  for(eN=0;eN<nElements;eN++)
    {
      eps = elementDiameter[eN]*epsFact;
      for(k=0;k<nQuadraturePoints_element;k++)
        {
          phi = nValueArray[eN*nQuadraturePoints_element + k];
          if (phi > eps)
            H=1.0;
          else if (phi < -eps)
            H=0.0;
          else if (phi==0.0)
            H=0.5;
          else
            H = 0.5*(1.0 + phi/eps + sin(M_PI*phi/eps)/M_PI);
          integral += H*dV[eN*nQuadraturePoints_element+k];
        }
    }
  return integral;
}

double fluxDomainBoundaryIntegral(proteus_index_t nExteriorElementBoundaries,
                                  proteus_index_t nElementBoundaries_owned,
                                  int nQuadraturePoints_elementBoundary,
                                  int* flag,
                                  proteus_index_t* exteriorElementBoundariesArray,
                                  double* dS,
                                  double* nValueArray)
{
  proteus_index_t ebN;  int ebNE, k;
  register double integral=0.0;
  for(ebNE=0;ebNE<nExteriorElementBoundaries;ebNE++)
    {
      ebN = exteriorElementBoundariesArray[ebNE];
      if (ebN < nElementBoundaries_owned && flag[ebN] > 0)
        {
          for(k=0;k<nQuadraturePoints_elementBoundary;k++)
            {
              integral += nValueArray[ebNE*nQuadraturePoints_elementBoundary + k]*
                dS[ebNE*nQuadraturePoints_elementBoundary+k];
            }
        }
    }
  return integral;
}

double fluxDomainBoundaryIntegralFromVector(proteus_index_t nExteriorElementBoundaries,
                                            proteus_index_t nElementBoundaries_owned,
                                            int nQuadraturePoints_elementBoundary,
                                            int nSpace,
                                            int* flag,
                                            proteus_index_t* exteriorElementBoundaries,
                                            double* dS,
                                            double* nValueArray,
                                            double* normal)
{
  proteus_index_t ebN, I;  int ebNE, k;
  register double integral=0.0,flux=0.0;
  for(ebNE=0;ebNE<nExteriorElementBoundaries;ebNE++)
    {
      ebN = exteriorElementBoundaries[ebNE];
      if (ebN < nElementBoundaries_owned && flag[ebN] > 0)
        {
          for(k=0;k<nQuadraturePoints_elementBoundary;k++)
            {
              flux=0.0;
              for (I=0;I<nSpace;I++)
                {
                  flux += nValueArray[ebNE*nQuadraturePoints_elementBoundary*nSpace + k*nSpace+I]*
                    normal[ebNE*nQuadraturePoints_elementBoundary*nSpace + k*nSpace+I];
                }
              integral += flux*dS[ebNE*nQuadraturePoints_elementBoundary+k];
            }
        }
    }
  return integral;
}


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
                                 double* nodalAverage)
{
  proteus_index_t eN, nN, ig;  int i, iv;
  proteus_index_t nElements_nN;
  memset(nodalAverage,0,sizeof(double)*nNodes_global*dim_dof);
  for (iv = 0; iv < dim_dof; iv++)
    {
      for (eN = 0; eN < nElements_global; eN++)
        {
          for (i = 0; i < nNodes_element; i++)
            {
              nN = elementNodesArray[eN*nNodes_element + i];
              ig = l2g[eN*nDOF_element + i];
              nodalAverage[nN*dim_dof + iv] += dof[ig*dim_dof + iv];
            }
        }
      for (nN = 0; nN < nNodes_global; nN++)
        {
          nElements_nN = nodeElementOffsets[nN+1] - nodeElementOffsets[nN];
          nodalAverage[nN*dim_dof + iv] /= nElements_nN;
        }
    }/*iv*/
}

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
                                double* nodalAverage)
{
  proteus_index_t eN, nN, ig;  int i, iv;
  proteus_index_t nElements_nN;
  memset(nodalAverage,0,sizeof(double)*nNodes_global*dim_dof);
  for (iv = 0; iv < dim_dof; iv++)
    {
      for (eN = 0; eN < nElements_global; eN++)
        {
          ig = l2g[eN*nDOF_element+0];
          for (i = 0; i < nNodes_element; i++)
            {
              nN = elementNodesArray[eN*nNodes_element+i];
              nodalAverage[nN*dim_dof + iv] += dof[ig*dim_dof + iv];
            }
        }
      for (nN = 0; nN < nNodes_global; nN++)
        {
          nElements_nN =  nodeElementOffsets[nN+1] - nodeElementOffsets[nN];
          nodalAverage[nN*dim_dof + iv] /= nElements_nN;
        }
    }
}

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
                                double* nodalAverage)
{
  proteus_index_t eN, nN, ig, jg;  int i, j, iv;
  proteus_index_t nElements_nN;
  double val,nd;
  memset(nodalAverage,0,sizeof(double)*nNodes_global*dim_dof);
  nd = nNodes_element - 1.0;
  for (iv = 0; iv < dim_dof; iv++)
    {
      for (eN = 0; eN < nElements_global; eN++)
        {
          /*
            recall N_i = n_d(1/n_d - \lambda_i)
            lambda_i = barycentric coord for node i

          */
          for (i = 0; i < nNodes_element; i++)
            {
              val = 0.0;
              for (j=0; j < i; j++)
                {
                  jg = l2g[eN*nDOF_element+j];
                  val += dof[jg*dim_dof + iv];
                }
              for (j=i+1; j < nNodes_element; j++)
                {
                  jg = l2g[eN*nDOF_element+j];
                  val += dof[jg*dim_dof + iv];
                }
              ig = l2g[eN*nDOF_element+i];
              val += (1.0-nd)*dof[ig*dim_dof + iv];
              nN = elementNodesArray[eN*nNodes_element+i];
              nodalAverage[nN*dim_dof + iv] += val;
            }
        }
      for (nN = 0; nN < nNodes_global; nN++)
        {
          nElements_nN = nodeElementOffsets[nN+1] - nodeElementOffsets[nN];
          nodalAverage[nN*dim_dof + iv] /= nElements_nN;
        }
    }/*iv*/
}

/**
  for debugging
 */
int checkElementBoundaryAndExteriorElementBoundaryArraysSame(int nElementBoundaries_element,
                                                             proteus_index_t nExteriorElementBoundaries_global,
                                                             int nQuadraturePoints_elementBoundary,
                                                             int nValuesPerQuadraturePoint,
                                                             double tolerance,
                                                             const proteus_index_t * exteriorElementBoundariesArray,
                                                             const proteus_index_t * elementBoundaryElementsArray,
                                                             const int * elementBoundaryLocalElementBoundariesArray,
                                                             const double * ebq_val,
                                                             const double * ebqe_val,
                                                             proteus_index_t* firstBadIndex)
{
  proteus_index_t eN, ebN;  int ebN_local, ebNE, k, i;
  double val1,val2;
  int failed = 0;
  *firstBadIndex = -1;

  for (ebNE = 0; ebNE < nExteriorElementBoundaries_global; ebNE++)
    {
      ebN = exteriorElementBoundariesArray[ebNE];
      eN  = elementBoundaryElementsArray[ebN*2+0];
      ebN_local = elementBoundaryLocalElementBoundariesArray[ebN*2+0];

      for (i=0; i < nValuesPerQuadraturePoint; i++)
        {
          val1 = ebq_val[eN*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nValuesPerQuadraturePoint +
                         ebN_local*nQuadraturePoints_elementBoundary*nValuesPerQuadraturePoint +
                         k*nValuesPerQuadraturePoint +
                         i];
          val2 = ebqe_val[ebNE*nQuadraturePoints_elementBoundary*nValuesPerQuadraturePoint +
                          k*nValuesPerQuadraturePoint +
                          i];
          if (fabs(val1-val2) > tolerance)
            {
              failed = 1;
              *firstBadIndex = ebNE;
              return failed;
            }
        }
    }
  return failed;
}

/**
  for debugging
 */
int checkGlobalElementBoundaryAndExteriorElementBoundaryArraysSame(proteus_index_t nExteriorElementBoundaries_global,
                                                                   int nQuadraturePoints_elementBoundary,
                                                                   int nValuesPerQuadraturePoint,
                                                                   double tolerance,
                                                                   const proteus_index_t * exteriorElementBoundariesArray,
                                                                   const proteus_index_t * elementBoundaryElementsArray,
                                                                   const int * elementBoundaryLocalElementBoundariesArray,
                                                                   const double * ebq_global_val,
                                                                   const double * ebqe_val,
                                                                   proteus_index_t* firstBadIndex)
{
  proteus_index_t eN, ebN;  int ebNE, k, i;
  double val1,val2;
  int failed = 0;
  *firstBadIndex = -1;

  for (ebNE = 0; ebNE < nExteriorElementBoundaries_global; ebNE++)
    {
      ebN = exteriorElementBoundariesArray[ebNE];
      eN  = elementBoundaryElementsArray[ebN*2+0];

      for (i=0; i < nValuesPerQuadraturePoint; i++)
        {
          val1 = ebq_global_val[ebN*nQuadraturePoints_elementBoundary*nValuesPerQuadraturePoint +
                                k*nValuesPerQuadraturePoint +
                                i];
          val2 = ebqe_val[ebNE*nQuadraturePoints_elementBoundary*nValuesPerQuadraturePoint +
                          k*nValuesPerQuadraturePoint +
                          i];
          if (fabs(val1-val2) > tolerance)
            {
              failed = 1;
              *firstBadIndex = ebNE;
              return failed;
            }
        }
    }
  return failed;
}

void copyFreeUnknownsToGlobalUnknowns(proteus_index_t nDOF2set,
                                      proteus_index_t offset,
                                      proteus_index_t stride,
                                      const proteus_index_t* globalDOFids,
                                      const proteus_index_t* freeDOFids,
                                      const double * free_u,
                                      double * u)
{
  proteus_index_t free_dofN;  proteus_index_t dofN; int i;
  for (i = 0; i < nDOF2set; i++)
    {
      dofN = globalDOFids[i]; free_dofN = freeDOFids[i];
      u[dofN] = free_u[offset + stride*free_dofN];
    }

}

void copyGlobalUnknownsToFreeUnknowns(proteus_index_t nDOF2set,
                                      proteus_index_t offset,
                                      proteus_index_t stride,
                                      const proteus_index_t* globalDOFids,
                                      const proteus_index_t* freeDOFids,
                                      const double * u,
                                      double * free_u)
{
  proteus_index_t free_dofN;  proteus_index_t dofN; int i;
  for (i = 0; i < nDOF2set; i++)
    {
      dofN = globalDOFids[i]; free_dofN = freeDOFids[i];
      free_u[offset + stride*free_dofN] = u[dofN]; /*hit  multiple times*/
    }

}

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
                                                   double* residual)
{
  proteus_index_t nSpace2=nSpace*nSpace;  proteus_index_t ebN, left_eN_global, right_eN_global, I, J; int ebNI, left_ebN_element, right_ebN_element, i, k;
  for(ebNI=0;ebNI<nInteriorElementBoundaries_global;ebNI++)
    {
      ebN = interiorElementBoundaries[ebNI];
      left_eN_global = elementBoundaryElements[ebN*2+0];
      right_eN_global = elementBoundaryElements[ebN*2+1];
      left_ebN_element = elementBoundaryLocalElementBoundaries[ebN*2+0];
      right_ebN_element = elementBoundaryLocalElementBoundaries[ebN*2+1];
      for(i=0;i<nDOF_test_element;i++)
        for(k=0;k<nQuadraturePoints_elementBoundary;k++)
          {
            for (I=0;I<nSpace;I++)
              for (J=0;J<nSpace;J++)
                {
                  residual[left_eN_global*nDOF_test_element+
                           i]
                    -=
                    sigma*0.5*((u[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary+
                                  left_ebN_element*nQuadraturePoints_elementBoundary+
                                  k]
                                -u[right_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary+
                                   right_ebN_element*nQuadraturePoints_elementBoundary+
                                   k])
                               *n[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                                  left_ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                                  k*nSpace+I]
                               *a[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace2+
                                  left_ebN_element*nQuadraturePoints_elementBoundary*nSpace2+
                                  k*nSpace2+
                                  J*nSpace+
                                  I]
                               *grad_w[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_test_element*nSpace+
                                       left_ebN_element*nQuadraturePoints_elementBoundary*nDOF_test_element*nSpace+
                                       k*nDOF_test_element*nSpace+
                                       i*nSpace+
                                       I]
                               *dS[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary+
                                   left_ebN_element*nQuadraturePoints_elementBoundary+
                                   k]);
                  residual[right_eN_global*nDOF_test_element+
                           i]
                    -=
                    sigma*0.5*((u[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary+
                                  left_ebN_element*nQuadraturePoints_elementBoundary+
                                  k]
                                -u[right_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary+
                                   right_ebN_element*nQuadraturePoints_elementBoundary+
                                   k])
                               *n[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                                  left_ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                                  k*nSpace+I]
                               *a[right_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace2+
                                  right_ebN_element*nQuadraturePoints_elementBoundary*nSpace2+
                                  k*nSpace2+
                                  J*nSpace+
                                  I]
                               *grad_w[right_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_test_element*nSpace+
                                       right_ebN_element*nQuadraturePoints_elementBoundary*nDOF_test_element*nSpace+
                                       k*nDOF_test_element*nSpace+
                                       i*nSpace+
                                       I]
                               *dS[right_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary+
                                   right_ebN_element*nQuadraturePoints_elementBoundary+
                                   k]);
                }
          }
    }
}

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
                                                   double* residual)
{
  proteus_index_t nSpace2=nSpace*nSpace;  proteus_index_t ebN, eN_global, I, J; int ebNE, i, k;
  for(ebNE=0;ebNE<nExteriorElementBoundaries_global;ebNE++)
    {
      ebN = exteriorElementBoundaries[ebNE];
      eN_global = elementBoundaryElements[ebN*2+0];
      for(i=0;i<nDOF_test_element;i++)
        for(k=0;k<nQuadraturePoints_elementBoundary;k++)
          {
            if (isDOFBoundary[ebNE*nQuadraturePoints_elementBoundary+k])
              {
                for (I=0;I<nSpace;I++)
                  for (J=0;J<nSpace;J++)
                    {
                      residual[eN_global*nDOF_test_element+
                               i]
                        -=
                        sigma*((u[ebNE*nQuadraturePoints_elementBoundary+
                                      k]
                                    -ub[ebNE*nQuadraturePoints_elementBoundary+
                                        k])
                                   *n[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                                      k*nSpace+
                                      I]
                                   *a[ebNE*nQuadraturePoints_elementBoundary*nSpace2+
                                      k*nSpace2+
                                      J*nSpace+
                                      I]
                                   *grad_w[ebNE*nQuadraturePoints_elementBoundary*nDOF_test_element*nSpace+
                                           k*nDOF_test_element*nSpace+
                                           i*nSpace+
                                           I]
                                   *dS[ebNE*nQuadraturePoints_elementBoundary+
                                       k]);
                    }
              }
          }
    }
}

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
                                                                           double* jac)
{
  proteus_index_t nSpace2=nSpace*nSpace;  proteus_index_t ebN, left_eN_global, right_eN_global, jacIndex, I, J; int ebNI, left_ebN_element, right_ebN_element, ii, i, k, jj, j, II, JJ;
  for (ebNI=0;ebNI<nInteriorElementBoundaries_global;ebNI++)
    {
      ebN = interiorElementBoundaries[ebNI];
      left_eN_global   = elementBoundaryElements[ebN*2+0];
      right_eN_global  = elementBoundaryElements[ebN*2+1];
      left_ebN_element  = elementBoundaryLocalElementBoundaries[ebN*2+0];
      right_ebN_element = elementBoundaryLocalElementBoundaries[ebN*2+1];
      /*left  flux*/
      for(ii=0;ii<nFreeDOF_element_r[left_eN_global];ii++)
        {
          i = freeLocal_r[left_eN_global*nDOF_test_element+
                          ii];
          I = offset_r + stride_r*freeGlobal_r[left_eN_global*nDOF_test_element+
                                               ii];
          for(k=0;k<nQuadraturePoints_elementBoundary;k++)
            {
              for(jj=0;jj<nFreeDOF_element_u[left_eN_global];jj++)
                {
                  j = freeLocal_u[left_eN_global*nDOF_trial_element+
                                  jj];
                  J = offset_u + stride_u*freeGlobal_u[left_eN_global*nDOF_trial_element+
                                                       jj];
                  jacIndex = I+J*nFreeVDOF_global;
                  for (II=0;II<nSpace;II++)
                    for (JJ=0;JJ<nSpace;JJ++)
                      jac[jacIndex]
                        -= sigma*0.5*(v[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                                        left_ebN_element*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                                        k*nDOF_trial_element+
                                        j]
                                      *n[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                                         left_ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                                         k*nSpace+
                                         II]
                                      *a[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace2+
                                         left_ebN_element*nQuadraturePoints_elementBoundary*nSpace2+k*nSpace2+
                                         II*nSpace+
                                         JJ]
                                      *grad_w[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_trial_element*nSpace+
                                              left_ebN_element*nQuadraturePoints_elementBoundary*nDOF_trial_element*nSpace+
                                              k*nDOF_trial_element*nSpace+
                                              i*nSpace+
                                              II]
                                      *dS[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary+
                                          left_ebN_element*nQuadraturePoints_elementBoundary+
                                          k]);
                }
              for(jj=0;jj<nFreeDOF_element_u[right_eN_global];jj++)
                {
                  j = freeLocal_u[right_eN_global*nDOF_trial_element+
                                  jj];
                  J = offset_u + stride_u*freeGlobal_u[right_eN_global*nDOF_trial_element+
                                                       jj];
                  jacIndex = I+J*nFreeVDOF_global;
                  for (II=0;II<nSpace;II++)
                    for (JJ=0;JJ<nSpace;JJ++)
                      jac[jacIndex] += sigma*0.5*(v[right_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                                                    right_ebN_element*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                                                    k*nDOF_trial_element+
                                                    j]
                                                  *n[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                                                     left_ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                                                     k*nSpace+
                                                     II]
                                                  *a[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace2+
                                                     left_ebN_element*nQuadraturePoints_elementBoundary*nSpace2+
                                                     k*nSpace2+
                                                     II*nSpace+
                                                     JJ]
                                                  *grad_w[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_trial_element*nSpace+
                                                          left_ebN_element*nQuadraturePoints_elementBoundary*nDOF_trial_element*nSpace+
                                                          k*nDOF_trial_element*nSpace+
                                                          i*nSpace+
                                                          II]
                                                  *dS[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary+
                                                      left_ebN_element*nQuadraturePoints_elementBoundary+
                                                      k]);
                }
            }/*k*/
        }/*ii*/
      /*right flux*/
      for(ii=0;ii<nFreeDOF_element_r[right_eN_global];ii++)
        {
          i = freeLocal_r[right_eN_global*nDOF_test_element+
                        ii];
          I = offset_r + stride_r*freeGlobal_r[right_eN_global*nDOF_test_element+
                                             ii];
          for(k=0;k<nQuadraturePoints_elementBoundary;k++)
            {
              for(jj=0;jj<nFreeDOF_element_u[left_eN_global];jj++)
                {
                  j = freeLocal_u[left_eN_global*nDOF_trial_element+
                                jj];
                  J = offset_u + stride_u*freeGlobal_u[left_eN_global*nDOF_trial_element+
                                                     jj];
                  jacIndex = I+J*nFreeVDOF_global;
                  for (II=0;II<nSpace;II++)
                    for (JJ=0;JJ<nSpace;JJ++)
                      jac[jacIndex] -= sigma*0.5*(v[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                                                    left_ebN_element*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                                                    k*nDOF_trial_element+
                                                    j]
                                                  *n[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                                                     left_ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                                                     k*nSpace+
                                                     II]
                                                  *a[right_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace2+
                                                     right_ebN_element*nQuadraturePoints_elementBoundary*nSpace2+
                                                     k*nSpace2+
                                                     II*nSpace+
                                                     JJ]
                                                  *grad_w[right_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_trial_element*nSpace+
                                                          right_ebN_element*nQuadraturePoints_elementBoundary*nDOF_trial_element*nSpace+
                                                          k*nDOF_trial_element*nSpace+
                                                          i*nSpace+
                                                          II]
                                                  *dS[right_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary+
                                                      right_ebN_element*nQuadraturePoints_elementBoundary+
                                                      k]);
                }
              for(jj=0;jj<nFreeDOF_element_u[right_eN_global];jj++)
                {
                  j = freeLocal_u[right_eN_global*nDOF_trial_element+
                                jj];
                  J = offset_u + stride_u*freeGlobal_u[right_eN_global*nDOF_trial_element+
                                                     jj];
                  jacIndex = I+J*nFreeVDOF_global;
                  for (II=0;II<nSpace;II++)
                    for (JJ=0;JJ<nSpace;JJ++)
                      jac[jacIndex] += sigma*0.5*(v[right_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                                                    right_ebN_element*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                                                    k*nDOF_trial_element+j]
                                                  *n[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                                                     left_ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                                                     k*nSpace+
                                                     II]
                                                  *a[right_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace2+
                                                     right_ebN_element*nQuadraturePoints_elementBoundary*nSpace2+
                                                     k*nSpace2+
                                                     II*nSpace+
                                                     JJ]
                                                  *grad_w[right_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_trial_element*nSpace+
                                                          right_ebN_element*nQuadraturePoints_elementBoundary*nDOF_trial_element*nSpace+
                                                          k*nDOF_trial_element*nSpace+
                                                          i*nSpace+
                                                          II]
                                                  *dS[right_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary+
                                                      right_ebN_element*nQuadraturePoints_elementBoundary+
                                                      k]);
                }
            }
        }
    }
}

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
                                                                           double* jac)
{
  proteus_index_t nSpace2=nSpace*nSpace;  proteus_index_t ebN, eN_global, jacIndex, I, J; int ebNE, ii, i, k, jj, j, II, JJ;
  for (ebNE=0;ebNE<nExteriorElementBoundaries_global;ebNE++)
    {
      ebN = exteriorElementBoundaries[ebNE];
      eN_global   = elementBoundaryElements[ebN*2+0];
      for(ii=0;ii<nFreeDOF_element_r[eN_global];ii++)
        {
          i = freeLocal_r[eN_global*nDOF_test_element+
                          ii];
          I = offset_r + stride_r*freeGlobal_r[eN_global*nDOF_test_element+
                                               ii];
          for(k=0;k<nQuadraturePoints_elementBoundary;k++)
            {
              for(jj=0;jj<nFreeDOF_element_u[eN_global];jj++)
                {
                  j = freeLocal_u[eN_global*nDOF_trial_element+
                                  jj];
                  J = offset_u + stride_u*freeGlobal_u[eN_global*nDOF_trial_element+
                                                       jj];
                  jacIndex = I+J*nFreeVDOF_global;
                  if (isDOFBoundary[ebNE*nQuadraturePoints_elementBoundary+k])
                    {
                      for (II=0;II<nSpace;II++)
                        for (JJ=0;JJ<nSpace;JJ++)
                          jac[jacIndex]
                            -= sigma*(v[ebNE*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                                        k*nDOF_trial_element+
                                        j]
                                      *n[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                                         k*nSpace+
                                         II]
                                      *a[ebNE*nQuadraturePoints_elementBoundary*nSpace2+
                                         II*nSpace+
                                         JJ]
                                      *grad_w[ebNE*nQuadraturePoints_elementBoundary*nDOF_trial_element*nSpace+
                                              k*nDOF_trial_element*nSpace+
                                              i*nSpace+
                                              II]
                                      *dS[ebNE*nQuadraturePoints_elementBoundary+
                                          k]);
                    }
                }
            }
        }
    }
}

void updateInteriorElementBoundaryDiffusionAdjoint_sd(proteus_index_t nInteriorElementBoundaries_global,
                                                      int nElementBoundaries_element,
                                                      int nQuadraturePoints_elementBoundary,
                                                      int nDOF_test_element,
                                                      int nSpace,
                                                      proteus_index_t* rowptr,
                                                      proteus_index_t* colind,
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
{
  proteus_index_t m, nnz=rowptr[nSpace];  proteus_index_t ebN, left_eN_global, right_eN_global, I; int ebNI, left_ebN_element, right_ebN_element, i, k;
  for(ebNI=0;ebNI<nInteriorElementBoundaries_global;ebNI++)
    {
      ebN = interiorElementBoundaries[ebNI];
      left_eN_global = elementBoundaryElements[ebN*2+0];
      right_eN_global = elementBoundaryElements[ebN*2+1];
      left_ebN_element = elementBoundaryLocalElementBoundaries[ebN*2+0];
      right_ebN_element = elementBoundaryLocalElementBoundaries[ebN*2+1];
      for(i=0;i<nDOF_test_element;i++)
        for(k=0;k<nQuadraturePoints_elementBoundary;k++)
          {
            for (I=0;I<nSpace;I++)
              for (m=rowptr[I];m<rowptr[I+1];m++)
                {
                  residual[left_eN_global*nDOF_test_element+
                           i]
                    -=
                    sigma*0.5*((u[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary+
                                  left_ebN_element*nQuadraturePoints_elementBoundary+
                                  k]
                                -u[right_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary+
                                   right_ebN_element*nQuadraturePoints_elementBoundary+
                                   k])
                               *n[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                                  left_ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                                  k*nSpace+
                                  colind[m]]
                               *a[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nnz+
                                  left_ebN_element*nQuadraturePoints_elementBoundary*nnz+
                                  k*nnz+
                                  m]
                               *grad_w[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_test_element*nSpace+
                                       left_ebN_element*nQuadraturePoints_elementBoundary*nDOF_test_element*nSpace+
                                       k*nDOF_test_element*nSpace+
                                       i*nSpace+
                                       I]
                               *dS[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary+
                                   left_ebN_element*nQuadraturePoints_elementBoundary+
                                   k]);
                  residual[right_eN_global*nDOF_test_element+
                           i]
                    -=
                    sigma*0.5*((u[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary+
                                  left_ebN_element*nQuadraturePoints_elementBoundary+
                                  k]
                                -u[right_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary+
                                   right_ebN_element*nQuadraturePoints_elementBoundary+
                                   k])
                               *n[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                                  left_ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                                  k*nSpace+
                                  colind[m]]
                               *a[right_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nnz+
                                  right_ebN_element*nQuadraturePoints_elementBoundary*nnz+
                                  k*nnz+
                                  m]
                               *grad_w[right_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_test_element*nSpace+
                                       right_ebN_element*nQuadraturePoints_elementBoundary*nDOF_test_element*nSpace+
                                       k*nDOF_test_element*nSpace+
                                       i*nSpace+
                                       I]
                               *dS[right_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary+
                                   right_ebN_element*nQuadraturePoints_elementBoundary+
                                   k]);
                }
          }
    }
}

void updateExteriorElementBoundaryDiffusionAdjoint_sd(proteus_index_t nExteriorElementBoundaries_global,
                                                      int nQuadraturePoints_elementBoundary,
                                                      int nDOF_test_element,
                                                      int nSpace,
                                                      proteus_index_t* rowptr,
                                                      proteus_index_t* colind,
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
{
  proteus_index_t m, nnz=rowptr[nSpace];  proteus_index_t ebN, eN_global, I; int ebNE, i, k;
  for(ebNE=0;ebNE<nExteriorElementBoundaries_global;ebNE++)
    {
      ebN = exteriorElementBoundaries[ebNE];
      eN_global = elementBoundaryElements[ebN*2+0];
      for(i=0;i<nDOF_test_element;i++)
        for(k=0;k<nQuadraturePoints_elementBoundary;k++)
          {
            if (isDOFBoundary[ebNE*nQuadraturePoints_elementBoundary+k])
              {
                for (I=0;I<nSpace;I++)
                  for (m=rowptr[I];m<rowptr[I+1];m++)
                    {
                      residual[eN_global*nDOF_test_element+
                               i]
                        -=
                        sigma*((u[ebNE*nQuadraturePoints_elementBoundary+
                                      k]
                                    -ub[ebNE*nQuadraturePoints_elementBoundary+
                                        k])
                                   *n[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                                      k*nSpace+
                                      colind[m]]
                                   *a[ebNE*nQuadraturePoints_elementBoundary*nnz+
                                      k*nnz+
                                      m]
                                   *grad_w[ebNE*nQuadraturePoints_elementBoundary*nDOF_test_element*nSpace+
                                           k*nDOF_test_element*nSpace+
                                           i*nSpace+
                                           I]
                                   *dS[ebNE*nQuadraturePoints_elementBoundary+
                                       k]);
                    }
              }
          }
    }
}

void updateGlobalJacobianFromInteriorElementBoundaryDiffusionAdjoint_dense_sd(proteus_index_t nInteriorElementBoundaries_global,
                                                                              int nElementBoundaries_element,
                                                                              int nQuadraturePoints_elementBoundary,
                                                                              int nDOF_test_element,
                                                                              int nDOF_trial_element,
                                                                              int nSpace,
                                                                              proteus_index_t* rowptr,
                                                                              proteus_index_t* colind,
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
{
  proteus_index_t m, nnz=rowptr[nSpace];  proteus_index_t ebN, left_eN_global, right_eN_global, jacIndex, I, J; int ebNI, left_ebN_element, right_ebN_element, ii, i, k, jj, j, II;
  for (ebNI=0;ebNI<nInteriorElementBoundaries_global;ebNI++)
    {
      ebN = interiorElementBoundaries[ebNI];
      left_eN_global   = elementBoundaryElements[ebN*2+0];
      right_eN_global  = elementBoundaryElements[ebN*2+1];
      left_ebN_element  = elementBoundaryLocalElementBoundaries[ebN*2+0];
      right_ebN_element = elementBoundaryLocalElementBoundaries[ebN*2+1];
      /*left  flux*/
      for(ii=0;ii<nFreeDOF_element_r[left_eN_global];ii++)
        {
          i = freeLocal_r[left_eN_global*nDOF_test_element+
                          ii];
          I = offset_r + stride_r*freeGlobal_r[left_eN_global*nDOF_test_element+
                                               ii];
          for(k=0;k<nQuadraturePoints_elementBoundary;k++)
            {
              for(jj=0;jj<nFreeDOF_element_u[left_eN_global];jj++)
                {
                  j = freeLocal_u[left_eN_global*nDOF_trial_element+
                                  jj];
                  J = offset_u + stride_u*freeGlobal_u[left_eN_global*nDOF_trial_element+
                                                       jj];
                  jacIndex = I+J*nFreeVDOF_global;
                  for (II=0;II<nSpace;II++)
                    for(m=rowptr[II];m<rowptr[II+1];m++)
                      {
                        jac[jacIndex]
                          -= sigma*0.5*(v[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                                          left_ebN_element*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                                          k*nDOF_trial_element+
                                          j]
                                        *n[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                                           left_ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                                           k*nSpace+
                                           colind[II]]
                                        *a[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nnz+
                                           left_ebN_element*nQuadraturePoints_elementBoundary*nnz+
                                           k*nnz+
                                           m]
                                        *grad_w[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_trial_element*nSpace+
                                                left_ebN_element*nQuadraturePoints_elementBoundary*nDOF_trial_element*nSpace+
                                                k*nDOF_trial_element*nSpace+
                                                i*nSpace+
                                                II]
                                        *dS[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary+
                                            left_ebN_element*nQuadraturePoints_elementBoundary+
                                            k]);
                      }
                }
              for(jj=0;jj<nFreeDOF_element_u[right_eN_global];jj++)
                {
                  j = freeLocal_u[right_eN_global*nDOF_trial_element+
                                  jj];
                  J = offset_u + stride_u*freeGlobal_u[right_eN_global*nDOF_trial_element+
                                                       jj];
                  jacIndex = I+J*nFreeVDOF_global;
                  for (II=0;II<nSpace;II++)
                    for (m=rowptr[II];m<rowptr[II+1];m++)
                      {
                        jac[jacIndex] += sigma*0.5*(v[right_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                                                      right_ebN_element*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                                                      k*nDOF_trial_element+
                                                      j]
                                                    *n[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                                                       left_ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                                                       k*nSpace+
                                                       colind[m]]
                                                    *a[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nnz+
                                                       left_ebN_element*nQuadraturePoints_elementBoundary*nnz+
                                                       k*nnz+
                                                       m]
                                                    *grad_w[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_trial_element*nSpace+
                                                            left_ebN_element*nQuadraturePoints_elementBoundary*nDOF_trial_element*nSpace+
                                                            k*nDOF_trial_element*nSpace+
                                                            i*nSpace+
                                                            II]
                                                    *dS[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary+
                                                        left_ebN_element*nQuadraturePoints_elementBoundary+
                                                        k]);
                      }
                }
            }/*k*/
        }/*ii*/
      /*right flux*/
      for(ii=0;ii<nFreeDOF_element_r[right_eN_global];ii++)
        {
          i = freeLocal_r[right_eN_global*nDOF_test_element+
                        ii];
          I = offset_r + stride_r*freeGlobal_r[right_eN_global*nDOF_test_element+
                                             ii];
          for(k=0;k<nQuadraturePoints_elementBoundary;k++)
            {
              for(jj=0;jj<nFreeDOF_element_u[left_eN_global];jj++)
                {
                  j = freeLocal_u[left_eN_global*nDOF_trial_element+
                                jj];
                  J = offset_u + stride_u*freeGlobal_u[left_eN_global*nDOF_trial_element+
                                                     jj];
                  jacIndex = I+J*nFreeVDOF_global;
                  for (II=0;II<nSpace;II++)
                    for (m=rowptr[II];m<rowptr[II+1];m++)
                      {
                        jac[jacIndex] -= sigma*0.5*(v[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                                                      left_ebN_element*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                                                      k*nDOF_trial_element+
                                                      j]
                                                    *n[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                                                       left_ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                                                       k*nSpace+
                                                       colind[m]]
                                                    *a[right_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nnz+
                                                       right_ebN_element*nQuadraturePoints_elementBoundary*nnz+
                                                       k*nnz+
                                                       m]
                                                    *grad_w[right_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_trial_element*nSpace+
                                                            right_ebN_element*nQuadraturePoints_elementBoundary*nDOF_trial_element*nSpace+
                                                            k*nDOF_trial_element*nSpace+
                                                            i*nSpace+
                                                            II]
                                                    *dS[right_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary+
                                                        right_ebN_element*nQuadraturePoints_elementBoundary+
                                                        k]);
                      }
                }
              for(jj=0;jj<nFreeDOF_element_u[right_eN_global];jj++)
                {
                  j = freeLocal_u[right_eN_global*nDOF_trial_element+
                                jj];
                  J = offset_u + stride_u*freeGlobal_u[right_eN_global*nDOF_trial_element+
                                                     jj];
                  jacIndex = I+J*nFreeVDOF_global;
                  for (II=0;II<nSpace;II++)
                    for (m=rowptr[II];m<rowptr[II+1];m++)
                      {
                        jac[jacIndex] += sigma*0.5*(v[right_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                                                      right_ebN_element*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                                                      k*nDOF_trial_element+j]
                                                    *n[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                                                       left_ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                                                       k*nSpace+
                                                       colind[m]]
                                                    *a[right_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nnz+
                                                       right_ebN_element*nQuadraturePoints_elementBoundary*nnz+
                                                       k*nnz+
                                                       m]
                                                    *grad_w[right_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_trial_element*nSpace+
                                                            right_ebN_element*nQuadraturePoints_elementBoundary*nDOF_trial_element*nSpace+
                                                            k*nDOF_trial_element*nSpace+
                                                            i*nSpace+
                                                            II]
                                                    *dS[right_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary+
                                                        right_ebN_element*nQuadraturePoints_elementBoundary+
                                                        k]);
                      }
                }
            }
        }
    }
}

void updateGlobalJacobianFromExteriorElementBoundaryDiffusionAdjoint_dense_sd(proteus_index_t nExteriorElementBoundaries_global,
                                                                              int nQuadraturePoints_elementBoundary,
                                                                              int nDOF_test_element,
                                                                              int nDOF_trial_element,
                                                                              int nSpace,
                                                                              proteus_index_t* rowptr,
                                                                              proteus_index_t* colind,
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
{
  proteus_index_t m, nnz=rowptr[nSpace];  proteus_index_t ebN, eN_global, jacIndex, I, J; int ebNE, ii, i, k, jj, j, II;
  for (ebNE=0;ebNE<nExteriorElementBoundaries_global;ebNE++)
    {
      ebN = exteriorElementBoundaries[ebNE];
      eN_global   = elementBoundaryElements[ebN*2+0];
      for(ii=0;ii<nFreeDOF_element_r[eN_global];ii++)
        {
          i = freeLocal_r[eN_global*nDOF_test_element+
                          ii];
          I = offset_r + stride_r*freeGlobal_r[eN_global*nDOF_test_element+
                                               ii];
          for(k=0;k<nQuadraturePoints_elementBoundary;k++)
            {
              for(jj=0;jj<nFreeDOF_element_u[eN_global];jj++)
                {
                  j = freeLocal_u[eN_global*nDOF_trial_element+
                                  jj];
                  J = offset_u + stride_u*freeGlobal_u[eN_global*nDOF_trial_element+
                                                       jj];
                  jacIndex = I+J*nFreeVDOF_global;
                  if (isDOFBoundary[ebNE*nQuadraturePoints_elementBoundary+k])
                    {
                      for (II=0;II<nSpace;II++)
                        for (m=rowptr[II];m<rowptr[II+1];m++)
                          jac[jacIndex]
                            -= sigma*(v[ebNE*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                                        k*nDOF_trial_element+
                                        j]
                                      *n[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                                         k*nSpace+
                                         colind[m]]
                                      *a[ebNE*nQuadraturePoints_elementBoundary*nnz+
                                         k*nnz+
                                         m]
                                      *grad_w[ebNE*nQuadraturePoints_elementBoundary*nDOF_trial_element*nSpace+
                                              k*nDOF_trial_element*nSpace+
                                              i*nSpace+
                                              II]
                                      *dS[ebNE*nQuadraturePoints_elementBoundary+
                                          k]);
                    }
                }
            }
        }
    }
}
void updateGlobalJacobianFromInteriorElementBoundaryDiffusionAdjoint_CSR_sd(proteus_index_t nInteriorElementBoundaries_global,
                                                                            int nElementBoundaries_element,
                                                                            int nQuadraturePoints_elementBoundary,
                                                                            int nDOF_test_element,
                                                                            int nDOF_trial_element,
                                                                            int nSpace,
                                                                            proteus_index_t* rowptr,
                                                                            proteus_index_t* colind,
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
                                                                            proteus_index_t* csrRowIndeces_ru,
                                                                            proteus_index_t* csrColumnOffsets_eb_ru,
                                                                            double sigma,
                                                                            double* v,
                                                                            double* n,
                                                                            double* a,
                                                                            double* grad_w,
                                                                            double* dS,
                                                                            double* jac)
{
  proteus_index_t m, nnz=rowptr[nSpace], nDOF_test_X_trial_element=nDOF_trial_element*nDOF_test_element;  proteus_index_t ebN, left_eN_global, right_eN_global, jacIndex; int ebNI, left_ebN_element, right_ebN_element, ii, i, k, jj, j, II;
  for (ebNI=0;ebNI<nInteriorElementBoundaries_global;ebNI++)
    {
      ebN = interiorElementBoundaries[ebNI];
      left_eN_global   = elementBoundaryElements[ebN*2+0];
      right_eN_global  = elementBoundaryElements[ebN*2+1];
      left_ebN_element  = elementBoundaryLocalElementBoundaries[ebN*2+0];
      right_ebN_element = elementBoundaryLocalElementBoundaries[ebN*2+1];
      /*left  flux*/
      for(ii=0;ii<nFreeDOF_element_r[left_eN_global];ii++)
        {
          i = freeLocal_r[left_eN_global*nDOF_test_element+
                          ii];
          for(k=0;k<nQuadraturePoints_elementBoundary;k++)
            {
              for(jj=0;jj<nFreeDOF_element_u[left_eN_global];jj++)
                {
                  j = freeLocal_u[left_eN_global*nDOF_trial_element+
                                  jj];
                  jacIndex = csrRowIndeces_ru[left_eN_global*nDOF_test_element+
                                              ii]
                    +
                    csrColumnOffsets_eb_ru[ebN*4*nDOF_test_X_trial_element +
                                           0*2*nDOF_test_X_trial_element +
                                           0*nDOF_test_X_trial_element +
                                           ii*nDOF_trial_element +
                                           jj];
                  for (II=0;II<nSpace;II++)
                    for(m=rowptr[II];m<rowptr[II+1];m++)
                      jac[jacIndex] -= sigma*0.5*(v[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                                                    left_ebN_element*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                                                    k*nDOF_trial_element+
                                                    j]
                                                  *n[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                                                     left_ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                                                     k*nSpace+
                                                     colind[m]]
                                                  *a[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nnz+
                                                     left_ebN_element*nQuadraturePoints_elementBoundary*nnz+
                                                     k*nnz+
                                                     m]
                                                  *grad_w[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_trial_element*nSpace+
                                                          left_ebN_element*nQuadraturePoints_elementBoundary*nDOF_trial_element*nSpace+
                                                          k*nDOF_trial_element*nSpace+
                                                          i*nSpace+
                                                          II]
                                                  *dS[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary+
                                                      left_ebN_element*nQuadraturePoints_elementBoundary+
                                                      k]);
                }
              for(jj=0;jj<nFreeDOF_element_u[right_eN_global];jj++)
                {
                  j = freeLocal_u[right_eN_global*nDOF_trial_element+
                                  jj];
                  jacIndex = csrRowIndeces_ru[left_eN_global*nDOF_test_element+
                                              ii]
                    +
                    csrColumnOffsets_eb_ru[ebN*4*nDOF_test_X_trial_element+
                                           0*2*nDOF_test_X_trial_element+
                                           1*nDOF_test_X_trial_element+
                                           ii*nDOF_trial_element+
                                           jj];
                  for (II=0;II<nSpace;II++)
                    for (m=rowptr[II];m<rowptr[II+1];m++)
                      jac[jacIndex] += sigma*0.5*(v[right_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                                                    right_ebN_element*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                                                    k*nDOF_trial_element+
                                                    j]
                                                  *n[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                                                     left_ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                                                     k*nSpace+
                                                     colind[m]]
                                                  *a[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nnz+
                                                     left_ebN_element*nQuadraturePoints_elementBoundary*nnz+
                                                     k*nnz+
                                                     m]
                                                  *grad_w[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_trial_element*nSpace+
                                                          left_ebN_element*nQuadraturePoints_elementBoundary*nDOF_trial_element*nSpace+
                                                          k*nDOF_trial_element*nSpace+
                                                          i*nSpace+
                                                          II]
                                                  *dS[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary+
                                                      left_ebN_element*nQuadraturePoints_elementBoundary+
                                                      k]);
                }
            }/*k*/
        }/*ii*/
      /*right flux*/
      for(ii=0;ii<nFreeDOF_element_r[right_eN_global];ii++)
        {
          i = freeLocal_r[right_eN_global*nDOF_test_element+
                          ii];
          for(k=0;k<nQuadraturePoints_elementBoundary;k++)
            {
              for(jj=0;jj<nFreeDOF_element_u[left_eN_global];jj++)
                {
                  j = freeLocal_u[left_eN_global*nDOF_trial_element+
                                jj];
                  jacIndex = csrRowIndeces_ru[right_eN_global*nDOF_test_element+
                                              ii]
                    +
                    csrColumnOffsets_eb_ru[ebN*4*nDOF_test_X_trial_element+
                                           1*2*nDOF_test_X_trial_element+
                                           0*nDOF_test_X_trial_element+
                                           ii*nDOF_trial_element+
                                           jj];
                  for (II=0;II<nSpace;II++)
                    for (m=rowptr[II];m<rowptr[II+1];m++)
                      jac[jacIndex] -= sigma*0.5*(v[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                                                    left_ebN_element*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                                                    k*nDOF_trial_element+
                                                    j]
                                                  *n[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                                                     left_ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                                                     k*nSpace+
                                                     colind[m]]
                                                  *a[right_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nnz+
                                                     right_ebN_element*nQuadraturePoints_elementBoundary*nnz+
                                                     k*nnz+
                                                     m]
                                                  *grad_w[right_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_trial_element*nSpace+
                                                          right_ebN_element*nQuadraturePoints_elementBoundary*nDOF_trial_element*nSpace+
                                                          k*nDOF_trial_element*nSpace+
                                                          i*nSpace+
                                                          II]
                                                  *dS[right_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary+
                                                      right_ebN_element*nQuadraturePoints_elementBoundary+
                                                      k]);
                }
              for(jj=0;jj<nFreeDOF_element_u[right_eN_global];jj++)
                {
                  j = freeLocal_u[right_eN_global*nDOF_trial_element+
                                  jj];
                  jacIndex = csrRowIndeces_ru[right_eN_global*nDOF_test_element+
                                              ii]
                    +
                    csrColumnOffsets_eb_ru[ebN*4*nDOF_test_X_trial_element+
                                           1*2*nDOF_test_X_trial_element+
                                           1*nDOF_test_X_trial_element+
                                           ii*nDOF_trial_element+
                                           jj];
                  for (II=0;II<nSpace;II++)
                    for (m=rowptr[II];m<rowptr[II+1];m++)
                      jac[jacIndex] += sigma*0.5*(v[right_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                                                    right_ebN_element*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                                                    k*nDOF_trial_element+j]
                                                  *n[left_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace+
                                                     left_ebN_element*nQuadraturePoints_elementBoundary*nSpace+
                                                     k*nSpace+
                                                     colind[m]]
                                                  *a[right_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nnz+
                                                     right_ebN_element*nQuadraturePoints_elementBoundary*nnz+
                                                     k*nnz+
                                                     m]
                                                  *grad_w[right_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nDOF_trial_element*nSpace+
                                                          right_ebN_element*nQuadraturePoints_elementBoundary*nDOF_trial_element*nSpace+
                                                          k*nDOF_trial_element*nSpace+
                                                          i*nSpace+
                                                          II]
                                                  *dS[right_eN_global*nElementBoundaries_element*nQuadraturePoints_elementBoundary+
                                                      right_ebN_element*nQuadraturePoints_elementBoundary+
                                                      k]);
                }
            }
        }
    }
}

void updateGlobalJacobianFromExteriorElementBoundaryDiffusionAdjoint_CSR_sd(proteus_index_t nExteriorElementBoundaries_global,
                                                                            int nQuadraturePoints_elementBoundary,
                                                                            int nDOF_test_element,
                                                                            int nDOF_trial_element,
                                                                            int nSpace,
                                                                            proteus_index_t* rowptr,
                                                                            proteus_index_t* colind,
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
                                                                            proteus_index_t* csrRowIndeces_ru,
                                                                            proteus_index_t* csrColumnOffsets_eb_ru,
                                                                            int* isDOFBoundary,
                                                                            double sigma,
                                                                            double* v,
                                                                            double* n,
                                                                            double* a,
                                                                            double* grad_w,
                                                                            double* dS,
                                                                            double* jac)
{
  proteus_index_t m, nnz=rowptr[nSpace], nDOF_test_X_trial_element=nDOF_trial_element*nDOF_test_element;  proteus_index_t ebN, eN_global, jacIndex; int ebNE, ii, i, k, jj, j, II;
  for (ebNE=0;ebNE<nExteriorElementBoundaries_global;ebNE++)
    {
      ebN = exteriorElementBoundaries[ebNE];
      eN_global   = elementBoundaryElements[ebN*2+0];
      for(ii=0;ii<nFreeDOF_element_r[eN_global];ii++)
        {
          i = freeLocal_r[eN_global*nDOF_test_element+
                          ii];
          for(k=0;k<nQuadraturePoints_elementBoundary;k++)
            {
              for(jj=0;jj<nFreeDOF_element_u[eN_global];jj++)
                {
                  j = freeLocal_u[eN_global*nDOF_trial_element+
                                  jj];
                  jacIndex = csrRowIndeces_ru[eN_global*nDOF_test_element+
                                              ii]
                    +
                    csrColumnOffsets_eb_ru[ebN*4*nDOF_test_X_trial_element +
                                           0*2*nDOF_test_X_trial_element +
                                           0*nDOF_test_X_trial_element +
                                           ii*nDOF_trial_element +
                                           jj];
                  if (isDOFBoundary[ebNE*nQuadraturePoints_elementBoundary+k])
                    {
                      for (II=0;II<nSpace;II++)
                        for (m=rowptr[II];m<rowptr[II+1];m++)
                          jac[jacIndex]
                            -= sigma*(v[ebNE*nQuadraturePoints_elementBoundary*nDOF_trial_element+
                                        k*nDOF_trial_element+
                                        j]
                                      *n[ebNE*nQuadraturePoints_elementBoundary*nSpace+
                                         k*nSpace+
                                         colind[m]]
                                      *a[ebNE*nQuadraturePoints_elementBoundary*nnz+
                                         k*nnz+
                                         m]
                                      *grad_w[ebNE*nQuadraturePoints_elementBoundary*nDOF_trial_element*nSpace+
                                              k*nDOF_trial_element*nSpace+
                                              i*nSpace+
                                              II]
                                      *dS[ebNE*nQuadraturePoints_elementBoundary+
                                          k]);
                    }
                }
            }
        }
    }
}

void update_f_movingDomain_q(proteus_index_t nElements_global,
                             int nQuadraturePoints_element,
                             int nSpace,
                             double* xt,
                             double* m,
                             double* f)
{
  proteus_index_t eN, I;  int k;
  for(eN=0;eN<nElements_global;eN++)
    for (k=0;k<nQuadraturePoints_element;k++)
      for (I=0;I<nSpace;I++)
        f[eN*nQuadraturePoints_element*nSpace +
          k*nSpace +
          I]
          -=
          m[eN*nQuadraturePoints_element +
            k]
          *
          xt[eN*nQuadraturePoints_element*3 +
             k*3 +
             I];
}

void update_f_movingDomain_constantMass_q(proteus_index_t nElements_global,
                                          int nQuadraturePoints_element,
                                          int nSpace,
                                          double* xt,
                                          double* f)
{
  proteus_index_t eN, I;  int k;
  for(eN=0;eN<nElements_global;eN++)
    for (k=0;k<nQuadraturePoints_element;k++)
      for (I=0;I<nSpace;I++)
        f[eN*nQuadraturePoints_element*nSpace +
          k*nSpace +
          I]
          -=
          xt[eN*nQuadraturePoints_element*3 +
             k*3 +
             I];
}

void update_f_movingDomain_ebq(proteus_index_t nElements_global,
                               int nElementBoundaries_element,
                               int nQuadraturePoints_elementBoundary,
                               int nSpace,
                               double* xt,
                               double* m,
                               double* f)
{
  proteus_index_t eN, ebN, I;  int k;
  for(eN=0;eN<nElements_global;eN++)
    for(ebN=0;ebN<nElementBoundaries_element;ebN++)
      for (k=0;k<nQuadraturePoints_elementBoundary;k++)
        for (I=0;I<nSpace;I++)
          f[eN*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace +
            ebN*nQuadraturePoints_elementBoundary*nSpace+
            k*nSpace +
            I]
            -=
            m[eN*nElementBoundaries_element*nQuadraturePoints_elementBoundary+
              ebN*nQuadraturePoints_elementBoundary+
              k]
            *
            xt[eN*nElementBoundaries_element*nQuadraturePoints_elementBoundary*3+
               ebN*nQuadraturePoints_elementBoundary*3+
               k*3 +
               I];
}

void update_f_movingDomain_constantMass_ebq(proteus_index_t nElements_global,
                                            int nElementBoundaries_element,
                                            int nQuadraturePoints_elementBoundary,
                                            int nSpace,
                                            double* xt,
                                            double* f)
{
  proteus_index_t eN, ebN, I;  int k;
  for(eN=0;eN<nElements_global;eN++)
    for(ebN=0;ebN<nElementBoundaries_element;ebN++)
      for (k=0;k<nQuadraturePoints_elementBoundary;k++)
        for (I=0;I<nSpace;I++)
          f[eN*nElementBoundaries_element*nQuadraturePoints_elementBoundary*nSpace +
            ebN*nQuadraturePoints_elementBoundary*nSpace+
            k*nSpace +
            I]
            -=
            xt[eN*nElementBoundaries_element*nQuadraturePoints_elementBoundary*3+
               ebN*nQuadraturePoints_elementBoundary*3+
               k*3 +
               I];
}

void updateStress_weak(proteus_index_t nElements_global,
                       int nQuadraturePoints_element,
                       int nDOF_test_element,
                       int nSpace,
                       double* sigma,
                       double* grad_w_dV,
                       double* weak_residual_x,
                       double* weak_residual_y,
                       double* weak_residual_z)
{
  proteus_index_t nSpace2=nSpace*nSpace;  proteus_index_t I, J; int eN, i, k;
  for(eN=0;eN<nElements_global;eN++)
    for (i=0;i<nDOF_test_element;i++)
      for (k=0;k<nQuadraturePoints_element;k++)
        for (J=0;J<nSpace;J++)
          {
            I=0;
            weak_residual_x[eN*nDOF_test_element + i]
              +=
              sigma[eN*nQuadraturePoints_element*nSpace2+
                    k*nSpace2 +
                    I*nSpace+
                    J]
              *
              grad_w_dV[eN*nQuadraturePoints_element*nDOF_test_element*nSpace +
                        k*nDOF_test_element*nSpace +
                        i*nSpace+
                        J];
            I=1;
            weak_residual_y[eN*nDOF_test_element + i]
              +=
              sigma[eN*nQuadraturePoints_element*nSpace2+
                    k*nSpace2 +
                    I*nSpace+
                    J]
              *
              grad_w_dV[eN*nQuadraturePoints_element*nDOF_test_element*nSpace +
                        k*nDOF_test_element*nSpace +
                        i*nSpace+
                        J];
            I=2;
            weak_residual_z[eN*nDOF_test_element + i]
              +=
              sigma[eN*nQuadraturePoints_element*nSpace2+
                    k*nSpace2 +
                    I*nSpace+
                    J]
              *
              grad_w_dV[eN*nQuadraturePoints_element*nDOF_test_element*nSpace +
                        k*nDOF_test_element*nSpace +
                        i*nSpace+
                        J];
          }
}

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
                               double* jacobian_weak_residual_zz)
{
  proteus_index_t nDOF_test_X_trial_element=nDOF_test_element*nDOF_trial_element, nSpace2=nSpace*nSpace;  proteus_index_t I, J; int eN, i, j, k;
  for(eN=0;eN<nElements_global;eN++)
    for (i=0;i<nDOF_test_element;i++)
      for (k=0;k<nQuadraturePoints_element;k++)
        for (j=0;j<nDOF_trial_element;j++)
          for (I=0;I<nSpace;I++)
            for(J=0;J<nSpace;J++)
              {
                //x
                jacobian_weak_residual_xx[eN*nDOF_test_X_trial_element +
                                          i*nDOF_trial_element +
                                          j]
                  +=
                  dsigma_xx[eN*nQuadraturePoints_element*nSpace2 +
                            k*nSpace2+
                            I*nSpace+
                            J]
                  *
                  grad_v[eN*nQuadraturePoints_element*nDOF_trial_element*nSpace +
                         k*nDOF_trial_element*nSpace +
                         j*nSpace +
                         J]
                  *
                  grad_w_dV[eN*nQuadraturePoints_element*nDOF_test_element*nSpace +
                            k*nDOF_test_element*nSpace +
                            i*nSpace +
                            I];
                jacobian_weak_residual_xy[eN*nDOF_test_X_trial_element +
                                          i*nDOF_trial_element +
                                          j]
                  +=
                  dsigma_xy[eN*nQuadraturePoints_element*nSpace2 +
                            k*nSpace2+
                            I*nSpace+
                            J]
                  *
                  grad_v[eN*nQuadraturePoints_element*nDOF_trial_element*nSpace +
                         k*nDOF_trial_element*nSpace +
                         j*nSpace +
                         J]
                  *
                  grad_w_dV[eN*nQuadraturePoints_element*nDOF_test_element*nSpace +
                            k*nDOF_test_element*nSpace +
                            i*nSpace +
                            I];
                jacobian_weak_residual_xz[eN*nDOF_test_X_trial_element +
                                          i*nDOF_trial_element +
                                          j]
                  +=
                  dsigma_xz[eN*nQuadraturePoints_element*nSpace2 +
                            k*nSpace2+
                            I*nSpace+
                            J]
                  *
                  grad_v[eN*nQuadraturePoints_element*nDOF_trial_element*nSpace +
                         k*nDOF_trial_element*nSpace +
                         j*nSpace +
                         J]
                  *
                  grad_w_dV[eN*nQuadraturePoints_element*nDOF_test_element*nSpace +
                            k*nDOF_test_element*nSpace +
                            i*nSpace +
                            I];
                //y
                jacobian_weak_residual_yx[eN*nDOF_test_X_trial_element +
                                          i*nDOF_trial_element +
                                          j]
                  +=
                  dsigma_yx[eN*nQuadraturePoints_element*nSpace2 +
                            k*nSpace2+
                            I*nSpace+
                            J]
                  *
                  grad_v[eN*nQuadraturePoints_element*nDOF_trial_element*nSpace +
                         k*nDOF_trial_element*nSpace +
                         j*nSpace +
                         J]
                  *
                  grad_w_dV[eN*nQuadraturePoints_element*nDOF_test_element*nSpace +
                            k*nDOF_test_element*nSpace +
                            i*nSpace +
                            I];
                jacobian_weak_residual_yy[eN*nDOF_test_X_trial_element +
                                          i*nDOF_trial_element +
                                          j]
                  +=
                  dsigma_yy[eN*nQuadraturePoints_element*nSpace2 +
                            k*nSpace2+
                            I*nSpace+
                            J]
                  *
                  grad_v[eN*nQuadraturePoints_element*nDOF_trial_element*nSpace +
                         k*nDOF_trial_element*nSpace +
                         j*nSpace +
                         J]
                  *
                  grad_w_dV[eN*nQuadraturePoints_element*nDOF_test_element*nSpace +
                            k*nDOF_test_element*nSpace +
                            i*nSpace +
                            I];
                jacobian_weak_residual_yz[eN*nDOF_test_X_trial_element +
                                          i*nDOF_trial_element +
                                          j]
                  +=
                  dsigma_yz[eN*nQuadraturePoints_element*nSpace2 +
                            k*nSpace2+
                            I*nSpace+
                            J]
                  *
                  grad_v[eN*nQuadraturePoints_element*nDOF_trial_element*nSpace +
                         k*nDOF_trial_element*nSpace +
                         j*nSpace +
                         J]
                  *
                  grad_w_dV[eN*nQuadraturePoints_element*nDOF_test_element*nSpace +
                            k*nDOF_test_element*nSpace +
                            i*nSpace +
                            I];
                //z
                jacobian_weak_residual_zx[eN*nDOF_test_X_trial_element +
                                          i*nDOF_trial_element +
                                          j]
                  +=
                  dsigma_zx[eN*nQuadraturePoints_element*nSpace2 +
                            k*nSpace2+
                            I*nSpace+
                            J]
                  *
                  grad_v[eN*nQuadraturePoints_element*nDOF_trial_element*nSpace +
                         k*nDOF_trial_element*nSpace +
                         j*nSpace +
                         J]
                  *
                  grad_w_dV[eN*nQuadraturePoints_element*nDOF_test_element*nSpace +
                            k*nDOF_test_element*nSpace +
                            i*nSpace +
                            I];
                jacobian_weak_residual_zy[eN*nDOF_test_X_trial_element +
                                          i*nDOF_trial_element +
                                          j]
                  +=
                  dsigma_zy[eN*nQuadraturePoints_element*nSpace2 +
                            k*nSpace2+
                            I*nSpace+
                            J]
                  *
                  grad_v[eN*nQuadraturePoints_element*nDOF_trial_element*nSpace +
                         k*nDOF_trial_element*nSpace +
                         j*nSpace +
                         J]
                  *
                  grad_w_dV[eN*nQuadraturePoints_element*nDOF_test_element*nSpace +
                            k*nDOF_test_element*nSpace +
                            i*nSpace +
                            I];
                jacobian_weak_residual_zz[eN*nDOF_test_X_trial_element +
                                          i*nDOF_trial_element +
                                          j]
                  +=
                  dsigma_zz[eN*nQuadraturePoints_element*nSpace2 +
                            k*nSpace2+
                            I*nSpace+
                            J]
                  *
                  grad_v[eN*nQuadraturePoints_element*nDOF_trial_element*nSpace +
                         k*nDOF_trial_element*nSpace +
                         j*nSpace +
                         J]
                  *
                  grad_w_dV[eN*nQuadraturePoints_element*nDOF_test_element*nSpace +
                            k*nDOF_test_element*nSpace +
                            i*nSpace +
                            I];
              }
}

/*load nodal degrees of freedom from an element based set of interpolation values*/
void projectFromNodalInterpolationConditions(proteus_index_t nElements_global,
                                             int nDOF_element,
                                             int dim_dof,
                                             const proteus_index_t * l2g,
                                             const int * functional_map_element,
                                             const double * interpolationValues,
                                             double * dofs)
{
  proteus_index_t eN;  int i, j, k;

  //  printf("dim_dof = %d, nDOF_element = %d\n",dim_dof,nDOF_element);
  for (eN = 0; eN < nElements_global; eN++)
    {
      for (i=0; i < nDOF_element; i++)
        {
          k = functional_map_element[i]; /*dof i maps to interp value k locally*/
          //  printf("eN=%d, k=%d, i=%d\n",eN,k,i);
          for (j=0; j < dim_dof; j++)
            {
              //     printf("l2g[%d]= %d \n", eN*nDOF_element+i, l2g[eN*nDOF_element+i]);
              //     printf("l2g[%d]*%d+%d = %d  \n", eN*nDOF_element+i, dim_dof, j, l2g[eN*nDOF_element+i]*dim_dof + j);
              //     printf("interpolationValues[eN*nDOF_element*dim_dof + k*dim_dof + j] = %.2f \n", interpolationValues[eN*nDOF_element*dim_dof + k*dim_dof + j]);
              dofs[l2g[eN*nDOF_element+i]*dim_dof + j] = interpolationValues[eN*nDOF_element*dim_dof + k*dim_dof + j];

            }
        }
    }
}
/** @}*/
