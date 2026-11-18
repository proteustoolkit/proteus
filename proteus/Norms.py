"""
Tools for calculating norms on function spaces.
"""
from .FemTools import *
from .Quadrature import *
from .Comm import globalSum,globalMax
from . import  cfemIntegrals

useC = True

"""
"""
def L2errorSFEMvsAF(analyticalFunction,quadraturePointArray,quadratureWeightArray,functionValueArray,T=None):
    error=0.0
    range_nQuadraturePoints_element = list(range(quadraturePointArray.shape[1]))
    for eN in range(quadraturePointArray.shape[0]):
        for k in range_nQuadraturePoints_element:
            AF = analyticalFunction.uOfXT(quadraturePointArray[eN,k],T)
            error += ((functionValueArray[eN,k] - AF)**2)*quadratureWeightArray[eN,k]
    error = sqrt(abs(globalSum(error)))
    return error

def L1errorSFEMvsAF(analyticalFunction,quadraturePointArray,quadratureWeightArray,functionValueArray,T=None):
    error=0.0
    range_nQuadraturePoints_element = list(range(quadraturePointArray.shape[1]))
    for eN in range(quadraturePointArray.shape[0]):
        for k in range_nQuadraturePoints_element:
            AF = analyticalFunction.uOfXT(quadraturePointArray[eN,k],T)
            error += abs(functionValueArray[eN,k] - AF)*quadratureWeightArray[eN,k]
    return globalSum(error)

def L2errorVFEMvsAF(analyticalFunction,quadraturePointArray,quadratureWeightArray,functionValueArray,T=None):
    error=0.0
    range_nQuadraturePoints_element = list(range(quadraturePointArray.shape[1]))
    for eN in range(quadraturePointArray.shape[0]):
        for k in range_nQuadraturePoints_element:
            AF = analyticalFunction.uOfXT(quadraturePointArray[eN,k],T)
            e2 = numpy.inner(functionValueArray[eN,k,:] - AF,
                             functionValueArray[eN,k,:] - AF)
            error += e2*quadratureWeightArray[eN,k]
    error = sqrt(abs(globalSum(error)))
    return error

def L1errorVFEMvsAF(analyticalFunction,quadraturePointArray,quadratureWeightArray,functionValueArray,T=None):
    error=0.0
    range_nQuadraturePoints_element = list(range(quadraturePointArray.shape[1]))
    for eN in range(quadraturePointArray.shape[0]):
        for k in range_nQuadraturePoints_element:
            AF = analyticalFunction.uOfXT(quadraturePointArray[eN,k],T)
            e1 = numpy.sum(numpy.absolute(functionValueArray[eN,k,:] - AF))
            error += e1*quadratureWeightArray[eN,k]
    return globalSum(error)



def L2errorSFEMvsAF2(analyticalFunction,quadraturePointArray,abs_det_J,
                     quadratureWeightArray,functionValueArray,T=None):
    error=0.0
    range_nQuadraturePoints_element = list(range(quadraturePointArray.shape[1]))
    for eN in range(quadraturePointArray.shape[0]):
        for k in range_nQuadraturePoints_element:
            AF = analyticalFunction.uOfXT(quadraturePointArray[eN,k],T)
            error += ((functionValueArray[eN,k] - AF)**2)*quadratureWeightArray[k]*abs_det_J[eN,k]
    error = sqrt(abs(globalSum(error)))
    return error

def L1errorSFEMvsAF2(analyticalFunction,quadraturePointArray,abs_det_J,
                     quadratureWeightArray,functionValueArray,T=None):
    error=0.0
    range_nQuadraturePoints_element = list(range(quadraturePointArray.shape[1]))
    for eN in range(quadraturePointArray.shape[0]):
        for k in range_nQuadraturePoints_element:
            AF = analyticalFunction.uOfXT(quadraturePointArray[eN,k],T)
            error += abs(functionValueArray[eN,k] - AF)*quadratureWeightArray[k]*abs_det_J[eN,k]
    return globalSum(error)

def L2errorVFEMvsAF2(analyticalFunction,quadraturePointArray,abs_det_J,
                     quadratureWeightArray,functionValueArray,T=None):
    error=0.0
    range_nQuadraturePoints_element = list(range(quadraturePointArray.shape[1]))
    for eN in range(quadraturePointArray.shape[0]):
        for k in range_nQuadraturePoints_element:
            AF = analyticalFunction.uOfXT(quadraturePointArray[eN,k],T)
            e2 = numpy.inner(functionValueArray[eN,k,:] - AF,
                             functionValueArray[eN,k,:] - AF)
            error += e2*quadratureWeightArray[k]*abs_det_J[eN,k]
    error = sqrt(abs(globalSum(error)))
    return error
def L1errorVFEMvsAF2(analyticalFunction,quadraturePointArray,quadratureWeightArray,functionValueArray,T=None):
    error=0.0
    range_nQuadraturePoints_element = list(range(quadraturePointArray.shape[1]))
    for eN in range(quadraturePointArray.shape[0]):
        for k in range_nQuadraturePoints_element:
            AF = analyticalFunction.uOfXT(quadraturePointArray[eN,k],T)
            e1 = numpy.sum(numpy.absolute(functionValueArray[eN,k,:] - AF))
            error += e1*quadratureWeightArray[k]*abs_det_J[eN,k]
    return globalSum(error)

def L2errorSFEM(quadratureWeightArray,aSolutionValueArray,nSolutionValueArray,T=None):
    error=0.0
    range_nQuadraturePoints_element = list(range(aSolutionValueArray.shape[1]))
    for eN in range(aSolutionValueArray.shape[0]):
        for k in range_nQuadraturePoints_element:
            error += ((aSolutionValueArray[eN,k] - nSolutionValueArray[eN,k])**2)*quadratureWeightArray[eN,k]
    error = sqrt(abs(globalSum(error)))
    return error
def L2errorSFEM_local(quadratureWeightArray,aSolutionValueArray,nSolutionValueArray,elementError,T=None):
    error=0.0
    elementError.flat[:]=0.0
    range_nQuadraturePoints_element = list(range(nSolutionValueArray.shape[1]))
    for eN in range(nSolutionValueArray.shape[0]):
        for k in range_nQuadraturePoints_element:
            elementError[eN] += ((aSolutionValueArray[eN,k] - nSolutionValueArray[eN,k])**2)*quadratureWeightArray[eN,k]
        error += elementError[eN]
        elementError[eN] = sqrt(abs(elementError[eN]))
    error = sqrt(abs(globalSum(error)))
    return error
def L2normSFEM(quadratureWeightArray,nSolutionValueArray):
    error=0.0
    range_nQuadraturePoints_element = list(range(nSolutionValueArray.shape[1]))
    for eN in range(nSolutionValueArray.shape[0]):
        for k in range_nQuadraturePoints_element:
            error += (nSolutionValueArray[eN,k]**2)*quadratureWeightArray[eN,k]
    error = sqrt(abs(globalSum(error)))
    return error

def L1errorSFEM(quadratureWeightArray,aSolutionValueArray,nSolutionValueArray,T=None):
    error=0.0
    range_nQuadraturePoints_element = list(range(aSolutionValueArray.shape[1]))
    for eN in range(aSolutionValueArray.shape[0]):
        for k in range_nQuadraturePoints_element:
            error += abs(aSolutionValueArray[eN,k] - nSolutionValueArray[eN,k])*quadratureWeightArray[eN,k]
    return globalSum(error)

def L2errorVFEM(quadratureWeightArray,aSolutionValueArray,nSolutionValueArray,T=None):
    error=0.0
    range_nQuadraturePoints_element = list(range(aSolutionValueArray.shape[1]))
    for eN in range(aSolutionValueArray.shape[0]):
        for k in range_nQuadraturePoints_element:
            e2    = numpy.inner(aSolutionValueArray[eN,k,:] - nSolutionValueArray[eN,k,:],
                                aSolutionValueArray[eN,k,:] - nSolutionValueArray[eN,k,:])
            error += e2*quadratureWeightArray[eN,k]
    error = sqrt(abs(globalSum(error)))
    return error

def L1errorVFEM(quadratureWeightArray,aSolutionValueArray,nSolutionValueArray,T=None):
    error=0.0
    range_nQuadraturePoints_element = list(range(aSolutionValueArray.shape[1]))
    for eN in range(aSolutionValueArray.shape[0]):
        for k in range_nQuadraturePoints_element:
            e      = numpy.sum(numpy.absolute(aSolutionValueArray[eN,k,:] - nSolutionValueArray[eN,k,:]))
            error += e*quadratureWeightArray[eN,k]
    return globalSum(error)

def L2errorSFEM2(abs_det_J,quadratureWeightArray,aSolutionValueArray,nSolutionValueArray,T=None):
    error=0.0
    range_nQuadraturePoints_element = list(range(aSolutionValueArray.shape[1]))
    for eN in range(aSolutionValueArray.shape[0]):
        for k in range_nQuadraturePoints_element:
            error += ((aSolutionValueArray[eN,k] - nSolutionValueArray[eN,k])**2)*quadratureWeightArray[k]*abs_det_J[eN,k]
    error = sqrt(abs(globalSum(error)))
    return error

def L1errorSFEM2(abs_det_J,quadratureWeightArray,aSolutionValueArray,nSolutionValueArray,T=None):
    error=0.0
    range_nQuadraturePoints_element = list(range(aSolutionValueArray.shape[1]))
    for eN in range(aSolutionValueArray.shape[0]):
        for k in range_nQuadraturePoints_element:
            error += abs(aSolutionValueArray[eN,k] - nSolutionValueArray[eN,k])*quadratureWeightArray[k]*abs_det_J[eN,k]
    error = sqrt(abs(globalSum(error)))
    return error

def L2errorVFEM2(quadratureWeightArray,aSolutionValueArray,nSolutionValueArray,T=None):
    error=0.0
    range_nQuadraturePoints_element = list(range(aSolutionValueArray.shape[1]))
    for eN in range(aSolutionValueArray.shape[0]):
        for k in range_nQuadraturePoints_element:
            e2    = numpy.inner(aSolutionValueArray[eN,k,:] - nSolutionValueArray[eN,k,:],
                                aSolutionValueArray[eN,k,:] - nSolutionValueArray[eN,k,:])
            error += e2*quadratureWeightArray[k]*abs_det_J[eN,k]
    error = sqrt(abs(globalSum(error)))
    return error

def L1errorVFEM2(quadratureWeightArray,aSolutionValueArray,nSolutionValueArray,T=None):
    error=0.0
    range_nQuadraturePoints_element = list(range(aSolutionValueArray.shape[1]))
    for eN in range(aSolutionValueArray.shape[0]):
        for k in range_nQuadraturePoints_element:
            e      = numpy.sum(numpy.absolute(aSolutionValueArray[eN,k,:] - nSolutionValueArray[eN,k,:]))
            error += e*quadratureWeightArray[k]*abs_det_J[eN,k]
    return globalSum(error)


#just compute the mass in the domain
def scalarDomainIntegral(dV,nValueArray,nElements=None):
    if useC:
        if nElements is None:
            partialSum = cfemIntegrals.scalarDomainIntegral(dV,nValueArray,nValueArray.shape[0])
        else:
            partialSum = cfemIntegrals.scalarDomainIntegral(dV,nValueArray,nElements)
    else:
        partialSum=0.0
        range_nQuadraturePoints_element = list(range(nValueArray.shape[1]))
        for eN in range(nValueArray.shape[0]):
            for k in range_nQuadraturePoints_element:
                partialSum += nValueArray[eN,k]*dV[eN,k]
    return globalSum(partialSum)

def scalarHeavisideDomainIntegral(dV,nValueArray,nElements=None):
    #unlike its neighbours, this kernel takes the counts first
    if nElements is None:
        nElements = nValueArray.shape[0]
    partialSum = cfemIntegrals.scalarHeavisideDomainIntegral(nElements,dV.shape[1],dV,nValueArray)
    return globalSum(partialSum)
def scalarSmoothedHeavisideDomainIntegral(epsFact,elementDiameters,dV,nValueArray,nElements=None):
    if nElements is None:
        partialSum = cfemIntegrals.scalarSmoothedHeavisideDomainIntegral(epsFact,elementDiameters,dV,nValueArray,nValueArray.shape[0])
    else:
        partialSum = cfemIntegrals.scalarSmoothedHeavisideDomainIntegral(epsFact,elementDiameters,dV,nValueArray,nElements)
    return globalSum(partialSum)
#compute the mass in the domain, but make global across processors
def globalScalarDomainIntegral(abs_det_J,quadratureWeightArray,nValueArray):
    integral = 0.0
    if useC:
        integral = cfemIntegrals.scalarDomainIntegral(abs_det_J,quadratureWeightArray,nValueArray)
    else:
        integral=0.0
        range_nQuadraturePoints_element = list(range(nValueArray.shape[1]))
        for eN in range(nValueArray.shape[0]):
            for k in range_nQuadraturePoints_element:
                integral += nValueArray[eN,k]*quadratureWeightArray[k]*abs_det_J[eN,k]

    return globalSum(integral)

#just compute the mass in the domain
def fluxDomainBoundaryIntegral(dS,nValueArray,mesh):
    partialSum = cfemIntegrals.fluxDomainBoundaryIntegral(mesh.nElementBoundaries_owned,
                                                          mesh.elementBoundaryUserFlags,
                                                          mesh.exteriorElementBoundariesArray,
                                                          dS,
                                                          nValueArray)
    return globalSum(partialSum)
def fluxDomainBoundaryIntegralFromVector(dS,nValueArray,normal,mesh):
    partialSum = cfemIntegrals.fluxDomainBoundaryIntegralFromVector(mesh.nElementBoundaries_owned,
                                                                    mesh.elementBoundaryUserFlags,
                                                                    mesh.exteriorElementBoundariesArray,
                                                                    dS,
                                                                    nValueArray,
                                                                    normal)
    return globalSum(partialSum)

def LIerrorSFEMvsAF(analyticalFunction,quadraturePointArray,functionValueArray,T=None):
    error=0.0
    range_nQuadraturePoints_element = list(range(quadraturePointArray.shape[1]))
    for eN in range(quadraturePointArray.shape[0]):
        for k in range_nQuadraturePoints_element:
            AF = analyticalFunction.uOfXT(quadraturePointArray[eN,k],T)
            error = max(error,abs(functionValueArray[eN,k] - AF))
    return globalMax(error)

def LIerrorVFEMvsAF(analyticalFunction,quadraturePointArray,quadratureWeightArray,functionValueArray,T=None):
    error=0.0
    range_nQuadraturePoints_element = list(range(quadraturePointArray.shape[1]))
    for eN in range(quadraturePointArray.shape[0]):
        for k in range_nQuadraturePoints_element:
            AF = analyticalFunction.uOfXT(quadraturePointArray[eN,k],T)
            e = numpy.absolute(functionValueArray[eN,k,:] - AF)

            error = max(error,max(e.flat))
    return globalMax(error)

def LIerrorSFEM(quadratureWeightArray,aSolutionValueArray,nSolutionValueArray,T=None):
    error=0.0
    range_nQuadraturePoints_element = list(range(aSolutionValueArray.shape[1]))
    error = max(numpy.absolute(aSolutionValueArray.flat - nSolutionValueArray.flat))

    return globalMax(error)
def TVseminormSFEM(dofArray,l2gMap):
    tv = 0.0
    nElements_global = l2gMap.shape[0]; nDOF_element = l2gMap.shape[1]
    for eN in range(nElements_global):
        for i in range(nDOF_element):
            I = l2gMap[eN,i]
            for j in range(i+1,nDOF_element):
                jj = int(fmod(j,nDOF_element))
                JJ = l2gMap[eN,jj]
                tv+= abs(dofArray[I]-dofArray[JJ])
            #neighbors on element
        #local dofs
    #elements
    return globalSum(tv)
## @}
