# cython: language_level=3
import numpy
cimport numpy
cdef extern from "math.h":
   double fabs(double x)
   double sqrt(double x)
   double pow(double x, double y)
   double exp(double x)
   double cos(double x)
   double sin(double x)
   double M_PI
cdef inline double double_max(double a, double b): return a if a >= b else b
cdef inline double double_min(double a, double b): return a if a <= b else b

ctypedef numpy.double_t DTYPE_t
#numpy.intc_t not in cython's numpy.pxd 
ctypedef int ITYPE_t
cdef extern from "index_type.h":
    ctypedef int proteus_index_t  # mesh entity numbers; its true width comes from the header

def setExteriorElementBoundaryTypes(int nExteriorElementBoundaries_global, 
                                    numpy.ndarray[proteus_index_t,ndim=1] exteriorElementBoundariesArray,
                                    numpy.ndarray[proteus_index_t,ndim=2] elementBoundaryElementsArray,
                                    numpy.ndarray[ITYPE_t,ndim=1] elementMaterialTypes,
                                    numpy.ndarray[ITYPE_t,ndim=1] exteriorElementBoundaryMaterialTypes):
    cdef proteus_index_t ebNE,ebN,eN
    for ebNE in range(nExteriorElementBoundaries_global):
        ebN = exteriorElementBoundariesArray[ebNE]
        eN  = elementBoundaryElementsArray[ebN,0]
        exteriorElementBoundaryMaterialTypes[ebNE] = elementMaterialTypes[eN]

def setElementBoundariesArray(int nElementBoundaries_global,
                              numpy.ndarray[proteus_index_t,ndim=2] elementBoundaryElementsArray,
                              numpy.ndarray[ITYPE_t,ndim=1] elementMaterialTypes,
                              numpy.ndarray[ITYPE_t,ndim=2] elementBoundaryMaterialTypes):
    cdef proteus_index_t ebN,eN_left,eN_right
    for ebN in range(nElementBoundaries_global):
        eN_left = elementBoundaryElementsArray[ebN,0]
        eN_right= elementBoundaryElementsArray[ebN,1]
        elementBoundaryMaterialTypes[ebN,0] = elementMaterialTypes[eN_left]
        elementBoundaryMaterialTypes[ebN,1] = elementMaterialTypes[eN_left]
        if eN_right >= 0:
            elementBoundaryMaterialTypes[ebN,1] = elementMaterialTypes[eN_right]

###
def setScalarMaterialFunctionOverElements(numpy.ndarray[ITYPE_t,ndim=1] elementMaterialTypes,
                                          numpy.ndarray[DTYPE_t,ndim=2] q_vals,
                                          dict material_functions):
    """
    loop over quadrature array and set is material j
    likely little improvement right now without correct typing of material_functions
    """
    cdef proteus_index_t eN
    cdef int k,material
    for eN in range(q_vals.shape[0]):
        material = elementMaterialTypes[eN]
        for k in range(q_vals.shape[1]):
            q_vals[eN,k] = material_functions[material]

def setVectorMaterialFunctionOverElements(numpy.ndarray[ITYPE_t,ndim=1] elementMaterialTypes,
                                          numpy.ndarray[DTYPE_t,ndim=3] q_vals,
                                          dict material_functions):
    """
    loop over quadrature array and set \vec f_j assuming element is material j
    """
    cdef proteus_index_t eN
    cdef int k,material
    for eN in range(q_vals.shape[0]):
        material = elementMaterialTypes[eN]
        for k in range(q_vals.shape[1]):
            q_vals[eN,k,:] = material_functions[material].flat
                                    

def setScalarMaterialFunctionOverElementBoundaries_arithmeticAverage(numpy.ndarray[proteus_index_t,ndim=2] elementBoundariesArray,
                                                                     numpy.ndarray[ITYPE_t,ndim=2] elementBoundaryTypes,
                                                                     numpy.ndarray[DTYPE_t,ndim=3] ebq_vals,
                                                                     dict material_functions):
    """
    loop over quadrature array and set f = 0.5(f^L_j+f^R_k) assuming element on left 
    is material j and element on right is material k

    likely little improvement right now without correct typing of material_functions
 
    """
    cdef proteus_index_t eN,ebN
    cdef int ebN_local,k,material_left,material_right
    
    for eN in range(ebq_vals.shape[0]):
        for ebN_local in range(ebq_vals.shape[1]):
            ebN = elementBoundariesArray[eN,ebN_local]
            material_left = elementBoundaryTypes[ebN,0]
            material_right= elementBoundaryTypes[ebN,1]
            for k in range(ebq_vals.shape[2]):
                ebq_vals[eN,ebN_local,k] = 0.5*(material_functions[material_left]+
                                                material_functions[material_right])

def setSparseTensorMaterialFunctionOverElementBoundaries_harmonicAverage(int nd,
                                                                         numpy.ndarray[proteus_index_t,ndim=2] elementBoundariesArray,
                                                                         numpy.ndarray[ITYPE_t,ndim=2] elementBoundaryTypes,
                                                                         numpy.ndarray[DTYPE_t,ndim=4] ebq_vals,
                                                                         dict material_functions):
    """
    loop over quadrature array and evaluate function \ten f_{mn} = f^L_{j,mn} f^R_{k,mn}/(f^L_{j,mn}+f^R_{k,mn})
     assuming element on left is material j and element on right is material k

    likely little improvement right now without correct typing of material_functions
 
    """
    cdef proteus_index_t eN,ebN
    cdef int ebN_local,k,material_left,material_right,I,J
    cdef double numer,denom

    for eN in range(ebq_vals.shape[0]):
        for ebN_local in range(ebq_vals.shape[1]):
            ebN = elementBoundariesArray[eN,ebN_local]
            material_left = elementBoundaryTypes[ebN,0]
            material_right= elementBoundaryTypes[ebN,1]
            for k in range(ebq_vals.shape[2]):
                for I in range(nd):
                    for J in range(nd):
                        numer = 2.0*material_functions[material_left][I,J]*material_functions[material_right][I,J]
                        denom = material_functions[material_left][I,J] + material_functions[material_right][I,J] + 1.0e-20
                        ebq_vals[eN,ebN_local,k,I*nd+J] = numer/denom

def setScalarMaterialFunctionOverGlobalElementBoundaries_arithmeticAverage(numpy.ndarray[proteus_index_t,ndim=2] elementBoundariesArray,
                                                                           numpy.ndarray[ITYPE_t,ndim=2] elementBoundaryTypes,
                                                                           numpy.ndarray[DTYPE_t,ndim=2] ebq_global_vals,
                                                                           dict material_functions):
    """
    loop over quadrature array and evaluate function f = 0.5(f^L_j+f^R_k) assuming element on left 
    is material j and element on right is material k

    likely little improvement right now without correct typing of material_functions
 
    """
    cdef proteus_index_t ebN
    cdef int material_left,material_right
    
    for ebN in range(ebq_global_vals.shape[0]):
        material_left = elementBoundaryTypes[ebN,0]
        material_right= elementBoundaryTypes[ebN,1]
        for k in range(ebq_global_vals.shape[1]):
            ebq_global_vals[ebN,k] = 0.5*(material_functions[material_left]+
                                          material_functions[material_right])


def setSparseTensorMaterialFunctionOverGlobalElementBoundaries_harmonicAverage(int nd,
                                                                               numpy.ndarray[proteus_index_t,ndim=2] elementBoundariesArray,
                                                                               numpy.ndarray[ITYPE_t,ndim=2] elementBoundaryTypes,
                                                                               numpy.ndarray[DTYPE_t,ndim=3] ebq_global_vals,
                                                                               dict material_functions):
    """
    loop over quadrature array and evaluate function \ten f_{mn} = f^L_{j,mn}f^R_{k,mn}/(f^L_{j,mn}+f^R_{k,mn})
     assuming element on left is material j and element on right is material k

    likely little improvement right now without correct typing of material_functions
 
    """
    cdef proteus_index_t ebN
    cdef int k,material_left,material_right,I,J
    cdef double numer,denom

    for ebN in range(ebq_global_vals.shape[0]):
        material_left = elementBoundaryTypes[ebN,0]
        material_right= elementBoundaryTypes[ebN,1]
        for k in range(ebq_global_vals.shape[1]):
            for I in range(nd):
                for J in range(nd):
                    numer = 2.0*material_functions[material_left][I,J]*material_functions[material_right][I,J]
                    denom = material_functions[material_left][I,J] + material_functions[material_right][I,J] + 1.0e-20
                    ebq_global_vals[ebN,k,I*nd+J] = numer/denom

###
def evaluateScalarMaterialFunctionOverElements(double t,
                                               numpy.ndarray[ITYPE_t,ndim=1] elementMaterialTypes,
                                               numpy.ndarray[DTYPE_t,ndim=3] x,
                                               numpy.ndarray[DTYPE_t,ndim=2] q_vals,
                                               dict material_functions):
    """
    loop over quadrature array and evaluate function f_j(x,t) assuming element is material j
    likely little improvement right now without correct typing of material_functions
    """
    cdef proteus_index_t eN
    cdef int k,material
    for eN in range(x.shape[0]):
        material = elementMaterialTypes[eN]
        for k in range(x.shape[1]):
            q_vals[eN,k] = material_functions[material](x[eN,k],t)
                                    
def evaluateVectorMaterialFunctionOverElements(double t,
                                               numpy.ndarray[ITYPE_t,ndim=1] elementMaterialTypes,
                                               numpy.ndarray[DTYPE_t,ndim=3] x,
                                               numpy.ndarray[DTYPE_t,ndim=3] q_vals,
                                               dict material_functions):
    """
    loop over quadrature array and evaluate function \vec f_j(x,t) assuming element is material j
    """
    cdef proteus_index_t eN
    cdef int k,material
    for eN in range(x.shape[0]):
        material = elementMaterialTypes[eN]
        for k in range(x.shape[1]):
            q_vals[eN,k,:] = material_functions[material](x[eN,k],t).flat
                                    

def evaluateScalarMaterialFunctionOverElementBoundaries_arithmeticAverage(double t,
                                                                          numpy.ndarray[proteus_index_t,ndim=2] elementBoundariesArray,
                                                                          numpy.ndarray[ITYPE_t,ndim=2] elementBoundaryTypes,
                                                                          numpy.ndarray[DTYPE_t,ndim=4] x,
                                                                          numpy.ndarray[DTYPE_t,ndim=3] ebq_vals,
                                                                          dict material_functions):
    """
    loop over quadrature array and evaluate function f(x,t) = 0.5(f^L_j(x,t)+f^R_k(x,t)) assuming element on left 
    is material j and element on right is material k

    likely little improvement right now without correct typing of material_functions
 
    """
    cdef proteus_index_t eN,ebN
    cdef int ebN_local,k,material_left,material_right
    
    for eN in range(x.shape[0]):
        for ebN_local in range(x.shape[1]):
            ebN = elementBoundariesArray[eN,ebN_local]
            material_left = elementBoundaryTypes[ebN,0]
            material_right= elementBoundaryTypes[ebN,1]
            for k in range(x.shape[2]):
                ebq_vals[eN,ebN_local,k] = 0.5*(material_functions[material_left](x[eN,ebN_local,k],t)+
                                                material_functions[material_right](x[eN,ebN_local,k],t))

def evaluateSparseTensorMaterialFunctionOverElementBoundaries_harmonicAverage(int nd,
                                                                              double t,
                                                                              numpy.ndarray[proteus_index_t,ndim=2] elementBoundariesArray,
                                                                              numpy.ndarray[ITYPE_t,ndim=2] elementBoundaryTypes,
                                                                              numpy.ndarray[DTYPE_t,ndim=4] x,
                                                                              numpy.ndarray[DTYPE_t,ndim=4] ebq_vals,
                                                                              dict material_functions):
    """
    loop over quadrature array and evaluate function \ten f_{mn}(x,t) = f^L_{j,mn}(x,t)f^R_{k,mn}(x,t)/(f^L_{j,mn}(x,t)+f^R_{k,mn})
     assuming element on left is material j and element on right is material k

    likely little improvement right now without correct typing of material_functions
 
    """
    cdef proteus_index_t eN,ebN
    cdef int ebN_local,k,material_left,material_right,I,J
    cdef double numer,denom

    for eN in range(x.shape[0]):
        for ebN_local in range(x.shape[1]):
            ebN = elementBoundariesArray[eN,ebN_local]
            material_left = elementBoundaryTypes[ebN,0]
            material_right= elementBoundaryTypes[ebN,1]
            for k in range(x.shape[2]):
                for I in range(nd):
                    for J in range(nd):
                        numer = 2.0*material_functions[material_left](x[eN,ebN_local,k],t)[I,J]*material_functions[material_right](x[eN,ebN_local,k],t)[I,J]
                        denom = material_functions[material_left](x[eN,ebN_local,k],t)[I,J] + material_functions[material_right](x[eN,ebN_local,k],t)[I,J] + 1.0e-20
                        ebq_vals[eN,ebN_local,k,I*nd+J] = numer/denom

def evaluateScalarMaterialFunctionOverGlobalElementBoundaries_arithmeticAverage(double t,
                                                                                numpy.ndarray[proteus_index_t,ndim=2] elementBoundariesArray,
                                                                                numpy.ndarray[ITYPE_t,ndim=2] elementBoundaryTypes,
                                                                                numpy.ndarray[DTYPE_t,ndim=3] x,
                                                                                numpy.ndarray[DTYPE_t,ndim=2] ebq_global_vals,
                                                                                dict material_functions):
    """
    loop over quadrature array and evaluate function f(x,t) = 0.5(f^L_j(x,t)+f^R_k(x,t)) assuming element on left 
    is material j and element on right is material k

    likely little improvement right now without correct typing of material_functions
 
    """
    cdef proteus_index_t ebN
    cdef int material_left,material_right
    
    for ebN in range(x.shape[0]):
        material_left = elementBoundaryTypes[ebN,0]
        material_right= elementBoundaryTypes[ebN,1]
        for k in range(x.shape[1]):
            ebq_global_vals[ebN,k] = 0.5*(material_functions[material_left](x[ebN,k],t)+
                                          material_functions[material_right](x[ebN,k],t))


def evaluateSparseTensorMaterialFunctionOverGlobalElementBoundaries_harmonicAverage(int nd,
                                                                                    double t,
                                                                                    numpy.ndarray[proteus_index_t,ndim=2] elementBoundariesArray,
                                                                                    numpy.ndarray[ITYPE_t,ndim=2] elementBoundaryTypes,
                                                                                    numpy.ndarray[DTYPE_t,ndim=4] x,
                                                                                    numpy.ndarray[DTYPE_t,ndim=3] ebq_global_vals,
                                                                                    dict material_functions):
    """
    loop over quadrature array and evaluate function \ten f_{mn}(x,t) = f^L_{j,mn}(x,t)f^R_{k,mn}(x,t)/(f^L_{j,mn}(x,t)+f^R_{k,mn})
     assuming element on left is material j and element on right is material k

    likely little improvement right now without correct typing of material_functions
 
    """
    cdef proteus_index_t ebN
    cdef int k,material_left,material_right,I,J
    cdef double numer,denom

    for ebN in range(x.shape[0]):
        material_left = elementBoundaryTypes[ebN,0]
        material_right= elementBoundaryTypes[ebN,1]
        for k in range(x.shape[1]):
            for I in range(nd):
                for J in range(nd):
                    numer = 2.0*material_functions[material_left](x[ebN,k],t)[I,J]*material_functions[material_right](x[ebN,k],t)[I,J]
                    denom = material_functions[material_left](x[ebN,k],t)[I,J] + material_functions[material_right](x[ebN,k],t)[I,J] + 1.0e-20
                    ebq_global_vals[ebN,k,I*nd+J] = numer/denom



##################################################
def RE_NCP1_evaluateElementCoefficients_Linear(double rho,
                                               numpy.ndarray[DTYPE_t,ndim=1] gravity,#physical quantities
                                               numpy.ndarray[ITYPE_t,ndim=1] rowptr,
                                               numpy.ndarray[ITYPE_t,ndim=1] colind,
                                               numpy.ndarray[DTYPE_t,ndim=2] KWs,
                                               #mesh info
                                               int nSpace,
                                               int nElements_global,                  
                                               int nElementBoundaries_element,
                                               numpy.ndarray[ITYPE_t,ndim=2] elementNeighborsArray,
                                               numpy.ndarray[ITYPE_t,ndim=1] elementMaterialTypes,
                                               numpy.ndarray[DTYPE_t,ndim=3] q_flin,
                                               numpy.ndarray[DTYPE_t,ndim=3] q_alin):
    """
    routine for evaluating linaer interface (nodal) coefficients in NCP1 approximation for Darcy Flow

    Approximation:
         uses  nodal quadrature where the nodes are face barycenters
         uses harmonic average for intrinsic permeability/ hydraulic conductivity
         assumes slight compressiblity for now

    TODO:
         everything
         
    """
    #temporaries
    cdef proteus_index_t eN,ebN
    cdef int eN_neighbor,ii,I,matID,matID_neig
    cdef int nSpace2 = nSpace*nSpace
    cdef int nnz = rowptr[nSpace]

    cdef numpy.ndarray[DTYPE_t,ndim=1] a_eN   = numpy.zeros(nnz,'d')
    cdef numpy.ndarray[DTYPE_t,ndim=1] a_neig = numpy.zeros(nnz,'d')
    cdef numpy.ndarray[DTYPE_t,ndim=1] a_avg  = numpy.zeros(nnz,'d')
 
    #loop through and evaluate
    for eN in range(nElements_global):
        matID = elementMaterialTypes[eN]
        for ii in range(nnz):
            a_eN[ii]  = rho*KWs[matID,ii]
        for ebN in range(nElementBoundaries_element):
            eN_neighbor = elementNeighborsArray[eN,ebN]
            for ii in range(nnz):
                a_neig[ii] = a_eN[ii]
            if eN_neighbor >= 0:
                matID_neig = elementMaterialTypes[eN_neighbor]
                for ii in range(nnz):
                    a_neig[ii]  = rho*KWs[matID_neig,ii]
            for ii in range(nnz):
                a_avg[ii] = 2.0*a_eN[ii]*a_neig[ii]/(a_eN[ii]+a_neig[ii]+1.0e-20)
                q_alin[eN,ebN,ii] = a_avg[ii]
            for I in range(nSpace):
                q_flin[eN,ebN,I] = 0.0
                for ii in range(rowptr[I],rowptr[I+1]):
                    q_flin[eN,ebN,I] += rho*a_avg[ii]*gravity[colind[ii]]
        #ebN
    #eN

                
def RE_NCP1_evaluateElementCoefficients_VGM(double rho,
                                            double beta,
                                            numpy.ndarray[DTYPE_t,ndim=1] gravity,#physical quantities
                                            numpy.ndarray[DTYPE_t,ndim=1] alpha,
                                            numpy.ndarray[DTYPE_t,ndim=1] n,
                                            numpy.ndarray[DTYPE_t,ndim=1] thetaR,
                                            numpy.ndarray[DTYPE_t,ndim=1] thetaSR,
                                            #mesh info
                                            int nSpace,
                                            int nElements_global,                  
                                            int nElementBoundaries_element,
                                            numpy.ndarray[ITYPE_t,ndim=2] elementNeighborsArray,
                                            numpy.ndarray[DTYPE_t,ndim=2] elementBarycentersArray,
                                            numpy.ndarray[ITYPE_t,ndim=1] elementMaterialTypes,
                                            #solution info
                                            int nDOF_trial_element,
                                            numpy.ndarray[ITYPE_t,ndim=2] u_l2g,
                                            numpy.ndarray[DTYPE_t,ndim=1] u_dof,
                                            #element quadrature arrays, must be nodal
                                            numpy.ndarray[DTYPE_t,ndim=3] q_x,
                                            numpy.ndarray[DTYPE_t,ndim=2] q_u,
                                            numpy.ndarray[DTYPE_t,ndim=2] q_mass,
                                            numpy.ndarray[DTYPE_t,ndim=2] q_dmass,
                                            numpy.ndarray[DTYPE_t,ndim=2] q_r,
                                            numpy.ndarray[DTYPE_t,ndim=2] q_kr,
                                            numpy.ndarray[DTYPE_t,ndim=2] q_dkr,
                                            numpy.ndarray[DTYPE_t,ndim=2] q_kr_up):
    """
    routine for evaluating nodal coefficients in NCP1 approximation for conservative head formulation of Richards equation 

    Approximation:
         uses  nodal quadrature where the nodes are face barycenters
         uses harmonic average for intrinsic permeability/ hydraulic conductivity
         assumes slight compressiblity for now

    TODO:
         everything
         
    """
    #check some sizes
    for q in [q_u,q_mass,q_r,q_kr,q_dkr]:
        assert q.shape[1] == nSpace+1
    assert nDOF_trial_element == nSpace + 1
    #temporaries
    cdef double psiC,pcBar,pcBar_n,pcBar_nM1,pcBar_nM2,onePlus_pcBar_n,sBar,sqrt_sBar,DsBar_DpsiC,thetaW,DthetaW_DpsiC
    cdef double vBar,vBar2,DvBar_DpsiC,KWr,DKWr_DpsiC,rho2=rho*rho,thetaS,rhom,drhom,m
    
    cdef double u_j,u_eN,u_neig,kr_eN,kr_neig,phi_eN,phi_neig
    cdef proteus_index_t eN,ebN
    cdef int eN_neighbor,ii,I,j,matID
    #for averaging/integration weights
    cdef double nAvgWeight = 1.0/(nSpace+1.)

    #loop through and evaluate
    for eN in range(nElements_global):
        matID = elementMaterialTypes[eN]
        for j in range(nDOF_trial_element):
            u_j = u_dof[u_l2g[eN,j]]
            q_u[eN,j] = u_j

            #VGM evaluation
            psiC = -u_j
            m = 1.0 - 1.0/n[matID]
            thetaS = thetaR[matID] + thetaSR[matID]
            if psiC > 0.0:
                pcBar = alpha[matID]*psiC
                pcBar_nM2 = pow(pcBar,n[matID]-2)
                pcBar_nM1 = pcBar_nM2*pcBar
                pcBar_n   = pcBar_nM1*pcBar
                onePlus_pcBar_n = 1.0 + pcBar_n
                
                sBar = pow(onePlus_pcBar_n,-m)
	        # using -mn = 1-n
                DsBar_DpsiC = alpha[matID]*(1.0-n[matID])*(sBar/onePlus_pcBar_n)*pcBar_nM1
                
                vBar = 1.0-pcBar_nM1*sBar
                vBar2 = vBar*vBar
                DvBar_DpsiC = -alpha[matID]*(n[matID]-1.0)*pcBar_nM2*sBar - pcBar_nM1*DsBar_DpsiC

                thetaW = thetaSR[matID]*sBar + thetaR[matID]
                DthetaW_DpsiC = thetaSR[matID] * DsBar_DpsiC 
                
                sqrt_sBar = sqrt(sBar)
                KWr= sqrt_sBar*vBar2
                DKWr_DpsiC= ((0.5/sqrt_sBar)*DsBar_DpsiC*vBar2 + 2.0*sqrt_sBar*vBar*DvBar_DpsiC)
            else:
                thetaW        = thetaS
                DthetaW_DpsiC = 0.0
                KWr           = 1.0
                DKWr_DpsiC    = 0.0
            #
            rhom = rho*exp(beta*u_j)
            drhom= beta*rhom
            q_mass[eN,j] = rhom*thetaW
            q_dmass[eN,j]=-rhom*DthetaW_DpsiC+drhom*thetaW

            q_kr[eN,j] = KWr
            q_dkr[eN,j]= -DKWr_DpsiC
            
        #j
    #eN

    #now upwind kr
    for eN in range(nElements_global):
        u_eN  = numpy.sum(q_u[eN])*nAvgWeight
        kr_eN = numpy.sum(q_kr[eN])*nAvgWeight
        #potential assumes slight compressibility
        phi_eN= u_eN - numpy.dot(gravity,elementBarycentersArray[eN])


        #loop over faces to upwind
        for ebN in range(nElementBoundaries_element):
            #by default eN is upwind
            q_kr_up[eN,ebN] = kr_eN
            eN_neighbor = elementNeighborsArray[eN,ebN]
            if eN_neighbor >= 0:
                u_neig  = numpy.sum(q_u[eN_neighbor])*nAvgWeight
                kr_neig = numpy.sum(q_kr[eN_neighbor])*nAvgWeight
                #potential, assumes slight compressibility
                phi_neig= u_neig - numpy.dot(gravity,elementBarycentersArray[eN_neighbor])

                if phi_eN < phi_neig: #neighbor  is upwind
                    q_kr_up[eN,ebN] = kr_neig

            #neighbor
        #local boundaries
    #eN for upwind
    
def RE_NCP1_getElementResidual(numpy.ndarray[DTYPE_t,ndim=1] gravity,#physical quantities
                               numpy.ndarray[ITYPE_t,ndim=1] rowptr,
                               numpy.ndarray[ITYPE_t,ndim=1] colind,
                               #mesh info
                               int nSpace,
                               int nElements_global,                  
                               int nElementBoundaries_element,
                               numpy.ndarray[ITYPE_t,ndim=2] elementNeighborsArray,
                               numpy.ndarray[DTYPE_t,ndim=2] elementBarycentersArray,
                               #solution information
                               int nDOF_test_element,
                               numpy.ndarray[DTYPE_t,ndim=2] q_u,
                               numpy.ndarray[DTYPE_t,ndim=3] q_grad_u,
                               numpy.ndarray[DTYPE_t,ndim=4] q_grad_w,
                               #element quadrature arrays, must be nodal
                               numpy.ndarray[DTYPE_t,ndim=2] q_detJ,
                               numpy.ndarray[DTYPE_t,ndim=2] q_m,
                               numpy.ndarray[DTYPE_t,ndim=2] q_mt,
                               numpy.ndarray[DTYPE_t,ndim=2] q_r,
                               numpy.ndarray[DTYPE_t,ndim=2] q_kr,
                               numpy.ndarray[DTYPE_t,ndim=2] q_kr_up,
                               #linear parts of f and a assumed to be evaluated
                               #already at interfaces using harmonic averages
                               numpy.ndarray[DTYPE_t,ndim=3] q_flin,
                               numpy.ndarray[DTYPE_t,ndim=3] q_alin,
                               #element residual information
                               numpy.ndarray[DTYPE_t,ndim=2] elementResidual
                               ):
    """
    residual routine for NCP1 approximation for conservative head formulation of Richards equation 

    Approximation:
         uses  nodal quadrature where the nodes are face barycenters
         uses harmonic average for intrinsic permeability/ hydraulic conductivity
         upwinds relative permeability based on element averages
         applies dirichlet boundary conditions strongly but keeps dofs in system

    TODO:
         everything
         
    """
    cdef int upwindFlag = 1
    #check some sizes
    for q in [q_u,q_m,q_mt,q_r,q_kr]:
        assert q.shape[1] == nSpace+1
    assert nDOF_test_element == nSpace+1
    cdef int nnz = rowptr[nSpace]
    #temporaries
    cdef double u_eN,kr_eN,phi_eN,u_neig,kr_neig,phi_neig
    cdef proteus_index_t eN,ebN
    cdef int eN_neighbor,ii,I,i
    cdef numpy.ndarray[DTYPE_t,ndim=1] a_up= numpy.zeros(nnz,'d')
    cdef numpy.ndarray[DTYPE_t,ndim=1] f_up= numpy.zeros(nSpace,'d')
    #for averaging/integration weights
    cdef double nAvgWeight = 1.0/(nSpace+1.)
    cdef double weight=1.0,volFactor = 1.0
    if nSpace == 2:
        volFactor = 0.5
    if nSpace == 3:
        volFactor = 1.0/6.0
    #
    for eN in range(nElements_global):
        volume = volFactor*fabs(q_detJ[eN,0]) #affine transformation
        weight = nAvgWeight*volume

        for i in range(nDOF_test_element):
            #nodal quadrature so diagonal
            ##mass
            elementResidual[eN,i] += weight*q_mt[eN,i] 
            ##sources
            elementResidual[eN,i] += weight*q_r[eN,i] 

            #have to actually compute loop over other nodes for stiffness terms
            for ebN in range(nElementBoundaries_element): #same as nDOF_trial , nElementBoundaries_
                #assumes linear parts of f and a and upwind k_r have already been evaluated correctly for interface
                for ii in range(nnz):
                    a_up[ii] = q_alin[eN,ebN,ii]*q_kr_up[eN,ebN]
                for I in range(nSpace):
                    f_up[I] = q_flin[eN,ebN,I]*q_kr_up[eN,ebN]
                #
                #accumulate advection and stiffness contributions
                for I in range(nSpace):
                    elementResidual[eN,i] -= weight*f_up[I]*q_grad_w[eN,ebN,i,I]
                    for ii in range(rowptr[I],rowptr[I+1]):
                        elementResidual[eN,i] += weight*a_up[ii]*q_grad_u[eN,ebN,colind[ii]]*q_grad_w[eN,ebN,i,I]
                #I
            #j
        #i
    #eN
def RE_NCP1_getElementJacobian(numpy.ndarray[DTYPE_t,ndim=1] gravity,#physical quantities
                               numpy.ndarray[ITYPE_t,ndim=1] rowptr,
                               numpy.ndarray[ITYPE_t,ndim=1] colind,
                               #mesh info
                               int nSpace,
                               int nElements_global,                  
                               int nElementBoundaries_element,
                               numpy.ndarray[ITYPE_t,ndim=2] elementNeighborsArray,
                               numpy.ndarray[DTYPE_t,ndim=2] elementBarycentersArray,
                               #solution information
                               int nDOF_test_element,
                               int nDOF_trial_element,
                               numpy.ndarray[DTYPE_t,ndim=2] q_u,
                               numpy.ndarray[DTYPE_t,ndim=3] q_grad_u,
                               numpy.ndarray[DTYPE_t,ndim=4] q_grad_w,
                               numpy.ndarray[DTYPE_t,ndim=4] q_grad_v,
                               #element quadrature arrays, must be nodal
                               numpy.ndarray[DTYPE_t,ndim=2] q_detJ,
                               numpy.ndarray[DTYPE_t,ndim=2] q_m,
                               numpy.ndarray[DTYPE_t,ndim=2] q_dm,
                               numpy.ndarray[DTYPE_t,ndim=2] q_mt,
                               numpy.ndarray[DTYPE_t,ndim=2] q_dmt,
                               numpy.ndarray[DTYPE_t,ndim=2] q_r,
                               numpy.ndarray[DTYPE_t,ndim=2] q_kr,
                               numpy.ndarray[DTYPE_t,ndim=2] q_dkr,
                               numpy.ndarray[DTYPE_t,ndim=2] q_kr_up,
                               #linear parts of f and a assumed to be evaluated
                               #already at interfaces using harmonic averages
                               numpy.ndarray[DTYPE_t,ndim=3] q_flin,
                               numpy.ndarray[DTYPE_t,ndim=3] q_alin,
                               #element residual information
                               numpy.ndarray[DTYPE_t,ndim=3] elementJacobian
                               ):
    """
    residual routine for NCP1 approximation for conservative head formulation of Richards equation 

    Approximation:
         uses  nodal quadrature where the nodes are face barycenters
         uses harmonic average for intrinsic permeability/ hydraulic conductivity
         upwinds relative permeability based on element averages
         applies dirichlet boundary conditions strongly but keeps dofs in system

    TODO:
         everything
         
    """
    cdef int upwindFlag = 1
    cdef int picard = 1
    #check some sizes
    for q in [q_u,q_m,q_mt,q_r,q_kr]:
        assert q.shape[1] == nSpace+1
    assert nDOF_test_element == nSpace+1
    cdef int nnz = rowptr[nSpace]
    #temporaries
    cdef double u_eN,kr_eN,phi_eN,u_neig,kr_neig,phi_neig,dkr_up
    cdef proteus_index_t eN
    cdef int eN_neighbor,ii,I
    cdef numpy.ndarray[DTYPE_t,ndim=1] a_up= numpy.zeros(nnz,'d')
    cdef numpy.ndarray[DTYPE_t,ndim=1] f_up= numpy.zeros(nSpace,'d')
    #for averaging/integration weights
    cdef double nAvgWeight = 1.0/(nSpace+1.)
    cdef double weight=1.0,volFactor = 1.0
    cdef int thisElementIsUpwind = 1
    if nSpace == 2:
        volFactor = 0.5
    if nSpace == 3:
        volFactor = 1.0/6.0
    #
    for eN in range(nElements_global):
        volume = volFactor*fabs(q_detJ[eN,0]) #affine transformation
        weight = nAvgWeight*volume

        
        for i in range(nDOF_test_element):
            #nodal quadrature so diagonal
            ##mass
            #elementResidual[eN,i] += weight*q_m[eN,i] 
            elementJacobian[eN,i,i] += weight*q_dmt[eN,i]

            #have to actually compute loop over other nodes for stiffness terms
            for ebN in range(nElementBoundaries_element): #same as nDOF_trial , nElementBoundaries_
                #assumes linear parts of f and a and upwind k_r have already been evaluated correctly for interface
                for ii in range(nnz):
                    a_up[ii] = q_alin[eN,ebN,ii]*q_kr_up[eN,ebN]
                for I in range(nSpace):
                    f_up[I] = q_flin[eN,ebN,I]*q_kr_up[eN,ebN]
                #

                #Picard part first
                for j in range(nDOF_trial_element):
                    for I in range(nSpace):
                        for ii in range(rowptr[I],rowptr[I+1]):
                            elementJacobian[eN,i,j] += weight*a_up[ii]*q_grad_v[eN,ebN,j,colind[ii]]*q_grad_w[eN,ebN,i,I]
                #j picard
        #i
    #eN
##################################################
#methods for optimized saturation equation 
##################################################

def updateMass_weakAvg(numpy.ndarray[DTYPE_t,ndim=2] mt,
                       numpy.ndarray[DTYPE_t,ndim=3] w,
                       numpy.ndarray[DTYPE_t,ndim=2] dV,
                       numpy.ndarray[DTYPE_t,ndim=2] weak_residual):
   """
   approximate element mass term as (\bar{c}_e,w_{h,i})_e
   """
   cdef proteus_index_t eN
   cdef int i,k
   cdef mt_avg,vol
   for eN in range(mt.shape[0]):
      mt_avg = 0.0
      vol    = 0.0
      for k in range(mt.shape[1]):
         mt_avg += dV[eN,k]*mt[eN,k]
         vol += dV[eN,k]
      mt_avg /= vol
      for i in range(weak_residual.shape[1]):
          #cek hack, seems like cython needed some help figuring out that the rvalue was a float
          weak_residual[eN,i] += float(mt_avg*w[eN,k,i]*dV[eN,k])
def updateMassJacobian_weakAvg(numpy.ndarray[DTYPE_t,ndim=2] dmt,
                               numpy.ndarray[DTYPE_t,ndim=3] w,
                               numpy.ndarray[DTYPE_t,ndim=3] v,
                               numpy.ndarray[DTYPE_t,ndim=2] dV,
                               numpy.ndarray[DTYPE_t,ndim=3] jacobian_weak_residual):
   """
   approximate element mass Jacobian term as (\pd{\bar{c}_e}{u_j},w_{h,i})_e
   """
   cdef proteus_index_t eN
   cdef int i,j,k
   cdef double dmtj_avg,vol
   for eN in range(dmt.shape[0]):
      vol = 0.0 #should I save a loop? 
      for k in range(dmt.shape[1]):
         vol += dV[eN,k]
      for i in range(w.shape[2]):
         for j in range(v.shape[2]):
            dmtj_avg = 0.0
            for k in range(dmt.shape[1]):
               dmtj_avg += dV[eN,k]*dmt[eN,k]*v[eN,k,j]
            dmtj_avg /= vol
            jacobian_weak_residual[eN,i,j] += dmtj_avg*w[eN,k,i]*dV[eN,k]

########################################################################
#ELLAM
########################################################################
def calculateNormalFlux(numpy.ndarray[DTYPE_t,ndim=4] v,
                        numpy.ndarray[DTYPE_t,ndim=4] n,
                        numpy.ndarray[DTYPE_t,ndim=3] dS,
                        numpy.ndarray[DTYPE_t,ndim=2] flux):
   
   cdef proteus_index_t eN,ebN
   cdef int kb
   cdef double integral
   for eN in range(n.shape[0]):
      for ebN in range(n.shape[1]):
         integral = 0.0
         for kb in range(n.shape[2]):
            for I in range(n.shape[3]):
               integral += v[eN,ebN,kb,I]*n[eN,ebN,kb,I]*dS[eN,ebN,kb]
         flux[eN,ebN] = integral

def computeSimpleCharacteristicVelocityFromElementVelocity(numpy.ndarray[DTYPE_t,ndim=3] df,
                                                           numpy.ndarray[DTYPE_t,ndim=3] characteristic_velocity,
                                                           numpy.ndarray[DTYPE_t,ndim=2] dm,
                                                           numpy.ndarray[DTYPE_t,ndim=2] dV):

   """
   simple approximation for \lambda = df/\bar{dm} using \bar{dm} = \frac{1}{\Omega_e} \int_{\Omega_e} dm dV
   """
   cdef proteus_index_t eN
   cdef int k,I
   cdef double omega_e, vol_e

   for eN in range(dm.shape[0]):
      omega_e = 0.0
      vol_e = 0.0
      for k in range(dm.shape[1]):
         vol_e   += dV[eN,k]
         omega_e += dV[eN,k]*dm[eN,k]
      for k in range(df.shape[1]):
         for I in range(df.shape[2]):
            characteristic_velocity[eN,k,I] = df[eN,k,I]*vol_e/(omega_e+1.0e-12)
            
def computeSimpleCharacteristicVelocityFromVelocityDOFs(numpy.ndarray[DTYPE_t,ndim=1] df_dofs,
                                                        numpy.ndarray[DTYPE_t,ndim=1] characteristic_velocity_dofs,
                                                        numpy.ndarray[proteus_index_t,ndim=2] l2g,
                                                        numpy.ndarray[DTYPE_t,ndim=2] dm,
                                                        numpy.ndarray[DTYPE_t,ndim=2] dV):

   """
   simple approximation for \lambda = df/\bar{dm} using \bar{dm} = \frac{1}{\Omega_e} \int_{\Omega_e} dm dV
   """
   cdef proteus_index_t eN
   cdef int k,j,J
   cdef double omega_e, vol_e

   for eN in range(dm.shape[0]):
      omega_e = 0.0
      vol_e = 0.0
      for k in range(dm.shape[1]):
         vol_e   += dV[eN,k]
         omega_e += dV[eN,k]*dm[eN,k]
      for j in range(l2g.shape[1]):
         J = l2g[eN,j]
         characteristic_velocity_dofs[J] = df_dofs[J]*vol_e/(omega_e+1.0e-12)

#problem specific velocity evaluation
def rotatingGaussianElementVelocityEval3(int transient,
                                         double t,
                                         double tForReversal,
                                         double clock,
                                         double xc, double yc,
                                         numpy.ndarray[DTYPE_t,ndim=3] x,
                                         numpy.ndarray[DTYPE_t,ndim=3] v,
                                         double zvelocity=0.0):
   cdef proteus_index_t eN
   cdef int k
   cdef double pi
   pi = M_PI
   if v.shape[2] == 3:
      if transient == 1:
         for eN in range(x.shape[0]):
            for k in range(x.shape[1]):
               v[eN,k,0]=2.0*pi*(x[eN,k,1]-xc)
               v[eN,k,1]=2.0*pi*(yc-x[eN,k,0])
               v[eN,k,2]=zvelocity
               v[eN,k,:]*=(tForReversal-t)/(tForReversal-0.0)*clock
      else:
         for eN in range(x.shape[0]):
            for k in range(x.shape[1]):
               v[eN,k,0]=2.0*pi*(x[eN,k,1]-xc)
               v[eN,k,1]=2.0*pi*(yc-x[eN,k,0])
               v[eN,k,2]=zvelocity
   else:
      assert v.shape[2] == 2
      if transient == 1:
         for eN in range(x.shape[0]):
            for k in range(x.shape[1]):
               v[eN,k,0]=2.0*pi*(x[eN,k,1]-xc)
               v[eN,k,1]=2.0*pi*(yc-x[eN,k,0])
               v[eN,k,:]*=(tForReversal-t)/(tForReversal-0.0)*clock
      else:
         for eN in range(x.shape[0]):
            for k in range(x.shape[1]):
               v[eN,k,0]=2.0*pi*(x[eN,k,1]-xc)
               v[eN,k,1]=2.0*pi*(yc-x[eN,k,0])

      
def rotatingGaussianElementVelocityEval4(int transient,
                                         double t,
                                         double tForReversal,
                                         double clock,
                                         double xc, double yc,
                                         numpy.ndarray[DTYPE_t,ndim=4] x,
                                         numpy.ndarray[DTYPE_t,ndim=4] v,
                                         double zvelocity=0.0):
   cdef proteus_index_t eN,ebN
   cdef int k
   cdef double pi
   pi = M_PI
   if v.shape[v.ndim-1] == 3:
      if transient == 1:
         for eN in range(x.shape[0]):
            for ebN in range(x.shape[1]):
               for k in range(x.shape[2]):
                  v[eN,ebN,k,0]=2.0*pi*(x[eN,ebN,k,1]-xc)
                  v[eN,ebN,k,1]=2.0*pi*(yc-x[eN,ebN,k,0])
                  v[eN,ebN,k,2]=zvelocity
                  v[eN,ebN,k,:]*=(tForReversal-t)/(tForReversal-0.0)*clock
      else:
         for eN in range(x.shape[0]):
            for ebN in range(x.shape[1]):
               for k in range(x.shape[2]):
                  v[eN,ebN,k,0]=2.0*pi*(x[eN,ebN,k,1]-xc)
                  v[eN,ebN,k,1]=2.0*pi*(yc-x[eN,ebN,k,0])
                  v[eN,ebN,k,2]=zvelocity
   else:
      assert v.shape[v.ndim-1] == 2
      if transient == 1:
         for eN in range(x.shape[0]):
            for ebN in range(x.shape[1]):
               for k in range(x.shape[2]):
                  v[eN,ebN,k,0]=2.0*pi*(x[eN,ebN,k,1]-xc)
                  v[eN,ebN,k,1]=2.0*pi*(yc-x[eN,ebN,k,0])
                  v[eN,ebN,k,:]*=(tForReversal-t)/(tForReversal-0.0)*clock
      else:
         for eN in range(x.shape[0]):
            for ebN in range(x.shape[1]):
               for k in range(x.shape[2]):
                  v[eN,ebN,k,0]=2.0*pi*(x[eN,ebN,k,1]-xc)
                  v[eN,ebN,k,1]=2.0*pi*(yc-x[eN,ebN,k,0])
      
def helicalElementVelocityEval3(int transient,
                                double t,
                                double tForReversal,
                                double clock,
                                double zVelocity,
                                double xc, double yc,
                                numpy.ndarray[DTYPE_t,ndim=3] x,
                                numpy.ndarray[DTYPE_t,ndim=3] v):
   cdef proteus_index_t eN
   cdef int k
   cdef double pi
   pi = M_PI
   if transient == 1:
      for eN in range(x.shape[0]):
         for k in range(x.shape[1]):
            v[eN,k,0]=2.0*pi*(x[eN,k,1]-xc)
            v[eN,k,1]=2.0*pi*(yc-x[eN,k,0])
            v[eN,k,2]=zVelocity
            v[eN,k,:]*=clock*cos(pi*t/(tForReversal*2.0))
   else:
      for eN in range(x.shape[0]):
         for k in range(x.shape[1]):
            v[eN,k,0]=2.0*pi*(x[eN,k,1]-xc)
            v[eN,k,1]=2.0*pi*(yc-x[eN,k,0])
            v[eN,k,2]=zVelocity
      
def helicalElementVelocityEval4(int transient,
                                double t,
                                double tForReversal,
                                double clock,
                                double zVelocity,
                                double xc, double yc,
                                numpy.ndarray[DTYPE_t,ndim=4] x,
                                numpy.ndarray[DTYPE_t,ndim=4] v):
   cdef proteus_index_t eN,ebN
   cdef int k
   cdef double pi
   pi = M_PI
   if transient == 1:
      for eN in range(x.shape[0]):
         for ebN in range(x.shape[1]):
            for k in range(x.shape[2]):
               v[eN,ebN,k,0]=2.0*pi*(x[eN,ebN,k,1]-xc)
               v[eN,ebN,k,1]=2.0*pi*(yc-x[eN,ebN,k,0])
               v[eN,ebN,k,2]=zVelocity
               v[eN,ebN,k,:]*=clock*cos(pi*t/(tForReversal*2.0))
   else:
      for eN in range(x.shape[0]):
         for ebN in range(x.shape[1]):
            for k in range(x.shape[2]):
               v[eN,ebN,k,0]=2.0*pi*(x[eN,ebN,k,1]-xc)
               v[eN,ebN,k,1]=2.0*pi*(yc-x[eN,ebN,k,0])
               v[eN,ebN,k,2]=zVelocity
      
def vortexElementVelocityEval3(double t,
                               numpy.ndarray[DTYPE_t,ndim=3] x,
                               numpy.ndarray[DTYPE_t,ndim=3] v):
   cdef proteus_index_t eN
   cdef int k
   cdef double pi,one8
   pi = M_PI
   one8 = 1.0/8.0
   for eN in range(x.shape[0]):
      for k in range(x.shape[1]):
         v[eN,k,0]= cos(pi*one8*t)*sin(2.0*pi*x[eN,k,1])*sin(pi*x[eN,k,0])*sin(pi*x[eN,k,0]);
         v[eN,k,1]=-cos(pi*one8*t)*sin(2.0*pi*x[eN,k,0])*sin(pi*x[eN,k,1])*sin(pi*x[eN,k,1]);

      
def vortexElementVelocityEval4(double t,
                               numpy.ndarray[DTYPE_t,ndim=4] x,
                               numpy.ndarray[DTYPE_t,ndim=4] v):
   cdef proteus_index_t eN,ebN
   cdef int k
   cdef double pi,one8
   pi = M_PI
   one8 = 1.0/8.0
   for eN in range(x.shape[0]):
      for ebN in range(x.shape[1]):
         for k in range(x.shape[2]):
            v[eN,ebN,k,0]= cos(pi*one8*t)*sin(2.0*pi*x[eN,ebN,k,1])*sin(pi*x[eN,ebN,k,0])*sin(pi*x[eN,ebN,k,0]);
            v[eN,ebN,k,1]=-cos(pi*one8*t)*sin(2.0*pi*x[eN,ebN,k,0])*sin(pi*x[eN,ebN,k,1])*sin(pi*x[eN,ebN,k,1]);

      
