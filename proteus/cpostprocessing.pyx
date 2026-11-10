# A type of -*- python -*- file
# cython: language_level=3
import numpy as np
cimport numpy as np
cimport postprocessing as pp
    
def postProcessRT0velocityFromP1nc(np.ndarray nFreeDOF_element,
                                   np.ndarray freeLocal_element,
                                   np.ndarray detJ,
                                   np.ndarray sqrt_det_g,
                                   np.ndarray n,
                                   np.ndarray elementBarycenters,
                                   np.ndarray quad_a,
                                   np.ndarray quad_f,
                                   np.ndarray w_dV_r,
                                   np.ndarray u,
                                   np.ndarray gradu,
                                   np.ndarray a,
                                   np.ndarray f,
                                   np.ndarray r,
                                   np.ndarray rt0vdofs,
                                   np.ndarray w_dV_m=None,
                                   np.ndarray mt=None):
    if mt is not None and w_dV_m is not None:
        pp.postProcessRT0velocityFromP1nc(gradu.shape[0],#/*nElements_global*/
                                          gradu.shape[1],#/*nQuadraturePoints_element*/
                                          w_dV_r.shape[2],#/*nDOF_test_element*/
                                          n.shape[1],    #/*nElementBoundaries_element*/
                                          n.shape[2],    #/*nQuadraturePoints_elementBoundary*/
                                          gradu.shape[2], #/*spaceDim*/
                                          <int*>(nFreeDOF_element.data),
                                          <int*>(freeLocal_element.data),
                                          <double*>(detJ.data),
                                          <double*>(sqrt_det_g.data),
                                          <double*>(n.data),
                                          <double*>(elementBarycenters.data),
                                          <double*>(quad_a.data),
                                          <double*>(quad_f.data),
                                          <double*>(w_dV_r.data),
                                          <double*>(w_dV_m.data),
                                          <double*>(u.data),
                                          <double*>(gradu.data),
                                          <double*>(a.data),
                                          <double*>(f.data),
                                          <double*>(r.data),
                                          <double*>(mt.data),
                                          <double*>(rt0vdofs.data))
    else:
        pp.postProcessRT0velocityFromP1ncNoMass(gradu.shape[0],#/*nElements_global*/
                                                gradu.shape[1],#/*nQuadraturePoints_element*/
                                                w_dV_r.shape[2],#/*nDOF_test_element*/
                                                n.shape[1],    #/*nElementBoundaries_element*/
                                                n.shape[2],    #/*nQuadraturePoints_elementBoundary*/
                                                gradu.shape[2], #/*spaceDim*/
                                                <int*>(nFreeDOF_element.data),
                                                <int*>(freeLocal_element.data),
                                                <double*>(detJ.data),
                                                <double*>(sqrt_det_g.data),
                                                <double*>(n.data),
                                                <double*>(elementBarycenters.data),
                                                <double*>(quad_a.data),
                                                <double*>(quad_f.data),
                                                <double*>(w_dV_r.data),
                                                <double*>(u.data),
                                                <double*>(gradu.data),
                                                <double*>(a.data),
                                                <double*>(f.data),
                                                <double*>(r.data),
                                                <double*>(rt0vdofs.data))

def postProcessRT0velocityFromP1nc_sd(np.ndarray rowptr,
                                      np.ndarray colind,
                                      np.ndarray nFreeDOF_element,
                                      np.ndarray freeLocal_element,
                                      np.ndarray detJ,
                                      np.ndarray sqrt_det_g,
                                      np.ndarray n,
                                      np.ndarray elementBarycenters,
                                      np.ndarray quad_a,
                                      np.ndarray quad_f,
                                      np.ndarray w_dV_r,
                                      np.ndarray u,
                                      np.ndarray gradu,
                                      np.ndarray a,
                                      np.ndarray f,
                                      np.ndarray r,
                                      np.ndarray rt0vdofs,
                                      np.ndarray w_dV_m=None,
                                      np.ndarray mt=None):
    if mt is None and w_dV_m is not None:
        pp.postProcessRT0velocityFromP1nc_sd(gradu.shape[0],#/*nElements_global*/
                                             gradu.shape[1],#/*nQuadraturePoints_element*/
                                             w_dV_r.shape[2],#/*nDOF_test_element*/
                                             n.shape[1],    #/*nElementBoundaries_element*/
                                             n.shape[2],    #/*nQuadraturePoints_elementBoundary*/
                                             gradu.shape[2], #/*spaceDim*/
                                             <pp.proteus_index_t*>(rowptr.data),
                                             <pp.proteus_index_t*>(colind.data),
                                             <int*>(nFreeDOF_element.data),
                                             <int*>(freeLocal_element.data),
                                             <double*>(detJ.data),
                                             <double*>(sqrt_det_g.data),
                                             <double*>(n.data),
                                             <double*>(elementBarycenters.data),
                                             <double*>(quad_a.data),
                                             <double*>(quad_f.data),
                                             <double*>(w_dV_r.data),
                                             <double*>(w_dV_m.data),
                                             <double*>(u.data),
                                             <double*>(gradu.data),
                                             <double*>(a.data),
                                             <double*>(f.data),
                                             <double*>(r.data),
                                             <double*>(mt.data),
                                             <double*>(rt0vdofs.data))
    else:
        pp.postProcessRT0velocityFromP1ncNoMass_sd(gradu.shape[0],#/*nElements_global*/
                                                   gradu.shape[1],#/*nQuadraturePoints_element*/
                                                   w_dV_r.shape[2],#/*nDOF_test_element*/
                                                   n.shape[1],    #/*nElementBoundaries_element*/
                                                   n.shape[2],    #/*nQuadraturePoints_elementBoundary*/
                                                   gradu.shape[2], #/*spaceDim*/
                                                   <pp.proteus_index_t*>(rowptr.data),
                                                   <pp.proteus_index_t*>(colind.data),
                                                   <int*>(nFreeDOF_element.data),
                                                   <int*>(freeLocal_element.data),
                                                   <double*>(detJ.data),
                                                   <double*>(sqrt_det_g.data),
                                                   <double*>(n.data),
                                                   <double*>(elementBarycenters.data),
                                                   <double*>(quad_a.data),
                                                   <double*>(quad_f.data),
                                                   <double*>(w_dV_r.data),
                                                   <double*>(u.data),
                                                   <double*>(gradu.data),
                                                   <double*>(a.data),
                                                   <double*>(f.data),
                                                   <double*>(r.data),
                                                   <double*>(rt0vdofs.data))

def updateRT0velocityWithAveragedPotentialP1nc(np.ndarray detJ,
                                               np.ndarray quad_a,
                                               np.ndarray phi,
                                               np.ndarray gradphi,
                                               np.ndarray a,
                                               np.ndarray rt0vdofs_element):
    pp.updateRT0velocityWithAveragedPotentialP1nc(a.shape[0],
                                             a.shape[1],
                                             gradphi.shape[gradphi.ndim-1],
                                             <double*>(detJ.data),
                                             <double*>(quad_a.data),
                                             <double*>(phi.data),
                                             <double*>(gradphi.data),
                                             <double*>(a.data), 
                                             <double*>(rt0vdofs_element.data))

def updateRT0velocityWithAveragedPotentialP1nc_sd(np.ndarray rowptr,
                                                  np.ndarray colind,
                                                  np.ndarray detJ,
                                                  np.ndarray quad_a,
                                                  np.ndarray phi,
                                                  np.ndarray gradphi,
                                                  np.ndarray a,
                                                  np.ndarray rt0vdofs_element):
    pp.updateRT0velocityWithAveragedPotentialP1nc_sd(a.shape[0],
                                                     a.shape[1],
                                                     gradphi.shape[gradphi.ndim-1],
                                                     <pp.proteus_index_t*>(rowptr.data),
                                                     <pp.proteus_index_t*>(colind.data),
                                                     <double*>(detJ.data),
                                                     <double*>(quad_a.data),
                                                     <double*>(phi.data),
                                                     <double*>(gradphi.data),
                                                     <double*>(a.data), 
                                                     <double*>(rt0vdofs_element.data))

def getElementRT0velocityValues(np.ndarray x_element,
                                np.ndarray rt0vdofs_element,
                                np.ndarray v_element):
    pp.getElementRT0velocityValues(v_element.shape[0],
                                   v_element.shape[1],
                                   v_element.shape[2],
                                   <double*>(x_element.data),
                                   <double*>(rt0vdofs_element.data),
                                   <double*>(v_element.data))

def getElementBoundaryRT0velocityValues(np.ndarray x_elementBoundary,
                                        np.ndarray rt0vdofs_element,
                                        np.ndarray v_elementBoundary):
    pp.getElementBoundaryRT0velocityValues(v_elementBoundary.shape[0],
                                           v_elementBoundary.shape[1],
                                           v_elementBoundary.shape[2],
                                           v_elementBoundary.shape[3],
                                           <double*>(x_elementBoundary.data),
                                           <double*>(rt0vdofs_element.data),
                                           <double*>(v_elementBoundary.data))

def getGlobalElementBoundaryRT0velocityValues(np.ndarray elementBoundaryElementsArray,
                                              np.ndarray x_elementBoundary_global,
                                              np.ndarray rt0vdofs_element,
                                              np.ndarray v_elementBoundary_global):
    pp.getGlobalElementBoundaryRT0velocityValues(v_elementBoundary_global.shape[0],
                                                 v_elementBoundary_global.shape[1],
                                                 v_elementBoundary_global.shape[2],
                                                 <pp.proteus_index_t*>(elementBoundaryElementsArray.data),
                                                 <double*>(x_elementBoundary_global.data),
                                                 <double*>(rt0vdofs_element.data),
                                                 <double*>(v_elementBoundary_global.data))
    
def getGlobalExteriorElementBoundaryRT0velocityValues(np.ndarray exteriorElementBoundariesArray,
                                                      np.ndarray elementBoundaryElementsArray,
                                                      np.ndarray x_elementBoundary_global,
                                                      np.ndarray rt0vdofs_element,
                                                      np.ndarray v_elementBoundary_global):
    pp.getGlobalExteriorElementBoundaryRT0velocityValues(exteriorElementBoundariesArray.shape[0],
                                                         v_elementBoundary_global.shape[1],
                                                         v_elementBoundary_global.shape[2],
                                                         <pp.proteus_index_t*>(elementBoundaryElementsArray.data),
                                                         <pp.proteus_index_t*>(exteriorElementBoundariesArray.data),
                                                         <double*>(x_elementBoundary_global.data),
                                                         <double*>(rt0vdofs_element.data),
                                                         <double*>(v_elementBoundary_global.data))

def postProcessRT0potentialFromP1nc(np.ndarray uQuadratureWeights_element,
                                    np.ndarray elementBarycenters,
                                    np.ndarray aElementQuadWeights,
                                    np.ndarray detJ,
                                    np.ndarray uQuadratureWeights_elementBoundary,
                                    np.ndarray x,
                                    np.ndarray u,
                                    np.ndarray gradu,
                                    np.ndarray x_elementBoundary,
                                    np.ndarray u_elementBoundary,
                                    np.ndarray n,
                                    np.ndarray a,
                                    np.ndarray f,
                                    np.ndarray r,
                                    np.ndarray rt0vdofs,
                                    np.ndarray rt0potential):
    pp.postProcessRT0potentialFromP1nc(u.shape[0],
                                       u.shape[1],
                                       n.shape[1],
                                       n.shape[2],
                                       n.shape[3],
                                       <double*>(uQuadratureWeights_element.data),
                                       <double*>(elementBarycenters.data),
                                       <double*>(aElementQuadWeights.data),
                                       <double*>(detJ.data),
                                       <double*>(uQuadratureWeights_element.data),
                                       <double*>(x.data),
                                       <double*>(u.data),
                                       <double*>(gradu.data),
                                       <double*>(x_elementBoundary.data),
                                       <double*>(u_elementBoundary.data),
                                       <double*>(n.data),
                                       <double*>(a.data),
                                       <double*>(f.data),
                                       <double*>(r.data),
                                       <double*>(rt0vdofs.data),
                                       <double*>(rt0potential.data))

def postProcessRT0potentialFromP1nc_sd(np.ndarray rowptr,
                                       np.ndarray colind,
                                       np.ndarray uQuadratureWeights_element,
                                       np.ndarray elementBarycenters,
                                       np.ndarray aElementQuadWeights,
                                       np.ndarray detJ,
                                       np.ndarray uQuadratureWeights_elementBoundary,
                                       np.ndarray x,
                                       np.ndarray u,
                                       np.ndarray gradu,
                                       np.ndarray x_elementBoundary,
                                       np.ndarray u_elementBoundary,
                                       np.ndarray n,
                                       np.ndarray a,
                                       np.ndarray f,
                                       np.ndarray r,
                                       np.ndarray rt0vdofs,
                                       np.ndarray rt0potential):
    pp.postProcessRT0potentialFromP1nc_sd(u.shape[0],
                                          u.shape[1],
                                          n.shape[1],
                                          n.shape[2],
                                          n.shape[3],
                                          <pp.proteus_index_t*>(rowptr.data),
                                          <pp.proteus_index_t*>(colind.data),
                                          <double*>(uQuadratureWeights_element.data),
                                          <double*>(elementBarycenters.data),
                                          <double*>(aElementQuadWeights.data),
                                          <double*>(detJ.data),
                                          <double*>(uQuadratureWeights_element.data),
                                          <double*>(x.data),
                                          <double*>(u.data),
                                          <double*>(gradu.data),
                                          <double*>(x_elementBoundary.data),
                                          <double*>(u_elementBoundary.data),
                                          <double*>(n.data),
                                          <double*>(a.data),
                                          <double*>(f.data),
                                          <double*>(r.data),
                                          <double*>(rt0vdofs.data),
                                          <double*>(rt0potential.data))

def buildLocalBDM1projectionMatrices(np.ndarray w_dS_f,
                                     np.ndarray ebq_n,
                                     np.ndarray ebq_v,
                                     np.ndarray BDMmat_element):
    pp.buildLocalBDM1projectionMatrices(ebq_n.shape[0],
                                        ebq_n.shape[1],
                                        ebq_n.shape[2],
                                        ebq_n.shape[3],
                                        w_dS_f.shape[3],
                                        ebq_v.shape[3],
                                        BDMmat_element.shape[1],
                                        <double*>(w_dS_f.data),
                                        <double*>(ebq_n.data),
                                        <double*>(ebq_v.data),
                                        <double*>(BDMmat_element.data))

def buildLocalBDM2projectionMatrices(int degree,
                                     np.ndarray w_dS_f,
                                     np.ndarray ebq_n,
                                     np.ndarray ebq_v,
                                     np.ndarray q_basis_vals,
                                     np.ndarray w_int_test_grads,
                                     np.ndarray w_int_div_free,
                                     np.ndarray piola_trial_fun,
                                     np.ndarray edgeFlags,
                                     np.ndarray BDMmat_element):
    pp.buildLocalBDM2projectionMatrices(degree,
                                        ebq_n.shape[0],
                                        ebq_n.shape[1],
                                        ebq_n.shape[2],
                                        q_basis_vals.shape[1],
                                        ebq_n.shape[3],
                                        w_dS_f.shape[3],
                                        ebq_v.shape[3],
                                        w_int_test_grads.shape[2],
                                        BDMmat_element.shape[1],
                                        <int*>(edgeFlags.data),
                                        <double*>(w_dS_f.data),
                                        <double*>(ebq_n.data),
                                        <double*>(ebq_v.data),
                                        <double*>(BDMmat_element.data),
                                        <double*>(q_basis_vals.data),
                                        <double*>(w_int_test_grads.data),
                                        <double*>(w_int_div_free.data),
                                        <double*>(piola_trial_fun.data))

def factorLocalBDM1projectionMatrices(np.ndarray BDMmat_element,
                                      np.ndarray BDMmatPivots_element):
    pp.factorLocalBDM1projectionMatrices(BDMmat_element.shape[0],
                                         BDMmat_element.shape[1],
                                         <double*>(BDMmat_element.data),
                                         <int*>(BDMmatPivots_element.data))

def factorLocalBDM2projectionMatrices(np.ndarray BDMmat_element,
                                      np.ndarray BDMmatPivots_element):
    pp.factorLocalBDM2projectionMatrices(BDMmat_element.shape[0],
                                         BDMmat_element.shape[1],
                                         <double*>(BDMmat_element.data),
                                         <int*>(BDMmatPivots_element.data))

def solveLocalBDM1projection(np.ndarray BDMmat_element,
                             np.ndarray BDMmatPivots_element,
                             np.ndarray w_dS_f,
                             np.ndarray ebq_n,
                             np.ndarray ebq_velocity,
                             np.ndarray q_vdofs):
    pp.solveLocalBDM1projection(ebq_n.shape[0],
                                ebq_n.shape[1],
                                ebq_n.shape[2],
                                ebq_n.shape[3],
                                w_dS_f.shape[3],
                                BDMmat_element.shape[1],
                                <double*>(BDMmat_element.data),
                                <int*>(BDMmatPivots_element.data),
                                <double*>(w_dS_f.data),
                                <double*>(ebq_n.data),
                                <double*>(ebq_velocity.data),
                                <double*>(q_vdofs.data))

def solveLocalBDM2projection(np.ndarray BDMmat_element,
                             np.ndarray BDMmatPivots_element,
                             np.ndarray w_dS_f,
                             np.ndarray ebq_n,
                             np.ndarray w_interior_gradients,
                             np.ndarray ebq_velocity,
                             np.ndarray q_velocity,
                             np.ndarray q_vdofs):
    pp.solveLocalBDM2projection(ebq_n.shape[0],
                                ebq_n.shape[1],
                                ebq_n.shape[2],
                                ebq_n.shape[3],
                                w_dS_f.shape[3],
                                BDMmat_element.shape[1],
                                <double*>(BDMmat_element.data),
                                <int*>(BDMmatPivots_element.data),
                                <double*>(w_dS_f.data),
                                <double*>(ebq_n.data),
                                <double*>(w_interior_gradients.data),
                                <double*>(q_velocity.data),
                                <double*>(ebq_velocity.data),
                                <double*>(q_vdofs.data))

def buildBDM2rhs(np.ndarray BDMmat_element,
                 np.ndarray BDMmatPivots_element,
                 np.ndarray w_dS_f,
                 np.ndarray ebq_n,
                 np.ndarray w_interior_gradients,
                 np.ndarray w_interior_divfree,
                 np.ndarray ebq_velocity,
                 np.ndarray q_velocity,
                 np.ndarray q_vdofs,
                 np.ndarray edgeFlags):
    pp.buildBDM2rhs(ebq_n.shape[0],
                    ebq_n.shape[1],
                    ebq_n.shape[2],
                    q_velocity.shape[1],
                    ebq_n.shape[3],
                    w_dS_f.shape[3],
                    BDMmat_element.shape[1],
                    w_interior_gradients.shape[2],
                    <double*>(BDMmat_element.data),
                    <int*>(BDMmatPivots_element.data),
                    <int*>(edgeFlags.data),
                    <double*>(w_dS_f.data),
                    <double*>(ebq_n.data),
                    <double*>(w_interior_gradients.data),
                    <double*>(w_interior_divfree.data),
                    <double*>(ebq_velocity.data),
                    <double*>(q_velocity.data),               
                    <double*>(q_vdofs.data))

def solveLocalBDM1projectionFromFlux(np.ndarray BDMmat_element,
                                     np.ndarray BDMmatPivots_element,
                                     np.ndarray elementBoundaryElementsArray,
                                     np.ndarray elementBoundariesArray,
                                     np.ndarray w_dS_f,
                                     np.ndarray ebq_global_flux,
                                     np.ndarray q_vdofs):
    pp.solveLocalBDM1projectionFromFlux(w_dS_f.shape[0],
                                        w_dS_f.shape[1],
                                        w_dS_f.shape[2],
                                        w_dS_f.shape[3],
                                        BDMmat_element.shape[1],
                                        <double*>(BDMmat_element.data),
                                        <int*>(BDMmatPivots_element.data),
                                        <pp.proteus_index_t*>(elementBoundaryElementsArray.data),
                                        <pp.proteus_index_t*>(elementBoundariesArray.data),
                                        <double*>(w_dS_f.data),
                                        <double*>(ebq_global_flux.data),
                                        <double*>(q_vdofs.data))

def getElementBDM1velocityValuesLagrangeRep(np.ndarray q_v,
                                            np.ndarray p1_vdofs,
                                            np.ndarray q_velocity):
    cdef int nVDOFs_element = q_velocity.shape[2]*(q_velocity.shape[2]+1)
    
    if p1_vdofs.ndim > 1:
        assert nVDOFs_element == p1_vdofs.shape[1]
    pp.getElementBDM1velocityValuesLagrangeRep(q_velocity.shape[0],
                                            q_velocity.shape[1],
                                            q_velocity.shape[2],
                                            q_v.shape[2],
                                            nVDOFs_element,
                                            <double*>(q_v.data),
                                            <double*>(p1_vdofs.data),
                                            <double*>(q_velocity.data))

def getElementBDM2velocityValuesLagrangeRep(np.ndarray q_v,
                                            np.ndarray p1_vdofs,
                                            np.ndarray q_velocity):
    if q_velocity.shape[2] == 2:
        #dimension of bdm2 in 2d
        nVDOFs_element = 12
    elif q_velocity.shape[2] == 3:
        #dimension of bdm2 in 3d
        nVDOFs_element = 30
        
    if p1_vdofs.ndim > 1:
        assert nVDOFs_element == p1_vdofs.shape[1]
        
    pp.getElementBDM2velocityValuesLagrangeRep(q_velocity.shape[0],
                                               q_velocity.shape[1],
                                               q_velocity.shape[2],
                                               q_v.shape[2],
                                               nVDOFs_element,
                                               <double*>(q_v.data),
                                               <double*>(p1_vdofs.data),
                                               <double*>(q_velocity.data))
    
def getGlobalExteriorElementBoundaryBDM1velocityValuesLagrangeRep(np.ndarray elementBoundaryElementsArray,
                                                                  np.ndarray exteriorElementBoundariesArray,
                                                                  np.ndarray ebqe_v,
                                                                  np.ndarray p1_vdofs,
                                                                  np.ndarray ebqe_velocity):
    cdef int nVDOFs_element = ebqe_velocity.shape[2]*(ebqe_velocity.shape[2]+1)
    if p1_vdofs.ndim > 1:
        assert nVDOFs_element == p1_vdofs.shape[1]
    pp.getGlobalExteriorElementBoundaryBDM1velocityValuesLagrangeRep(ebqe_velocity.shape[0],
                                                                     ebqe_velocity.shape[1],
                                                                     ebqe_velocity.shape[2],
                                                                     ebqe_v.shape[2],
                                                                     nVDOFs_element,
                                                                     <pp.proteus_index_t*>(elementBoundaryElementsArray.data),
                                                                     <pp.proteus_index_t*>(exteriorElementBoundariesArray.data),
                                                                     <double*>(ebqe_v.data),
                                                                     <double*>(p1_vdofs.data),
                                                                     <double*>(ebqe_velocity.data))

def getGlobalElementBoundaryBDM1velocityValuesLagrangeRep(np.ndarray elementBoundaryElementsArray,
                                                          np.ndarray exteriorElementBoundariesArray,
                                                          np.ndarray ebqe_v,
                                                          np.ndarray p1_vdofs,
                                                          np.ndarray ebq_global_velocity):
    cdef int nVDOFs_element = ebq_global_velocity.shape[2]*(ebq_global_velocity.shape[2]+1)
    if p1_vdofs.ndim > 1:
        assert nVDOFs_element == p1_vdofs.shape[1]
    pp.getGlobalExteriorElementBoundaryBDM1velocityValuesLagrangeRep(ebqe_v.shape[0],
                                                                     ebq_global_velocity.shape[1],
                                                                     ebq_global_velocity.shape[2],
                                                                     ebqe_v.shape[2],
                                                                     nVDOFs_element,
                                                                     <pp.proteus_index_t*>(elementBoundaryElementsArray.data),
                                                                     <pp.proteus_index_t*>(exteriorElementBoundariesArray.data),
                                                                     <double*>(ebqe_v.data),
                                                                     <double*>(p1_vdofs.data),
                                                                     <double*>(ebq_global_velocity.data))

def getElementBoundaryBDM1velocityValuesLagrangeRep(np.ndarray elementBoundaryElementsArray,
                                                    np.ndarray exteriorElementBoundariesArray,
                                                    np.ndarray ebq_v,
                                                    np.ndarray p1_vdofs,
                                                    np.ndarray ebq_velocity):
    cdef int nVDOFs_element = ebq_velocity.shape[3]*(ebq_velocity.shape[3]+1)
    if p1_vdofs.ndim > 1:
        assert nVDOFs_element == p1_vdofs.shape[1]
        pp.getElementBoundaryBDM1velocityValuesLagrangeRep(ebq_velocity.shape[0],
                                                           ebq_velocity.shape[1],
                                                           ebq_velocity.shape[2],
                                                           ebq_velocity.shape[3],
                                                           ebq_v.shape[3], # shape functions per element; shape[2] is the face quadrature points
                                                           nVDOFs_element,
                                                           <pp.proteus_index_t*>(elementBoundaryElementsArray.data),
                                                           <pp.proteus_index_t*>(exteriorElementBoundariesArray.data), #/*need the correct mapping here */
                                                           <double*>(ebq_v.data),
                                                           <double*>(p1_vdofs.data),
                                                           <double*>(ebq_velocity.data))

def projectElementBoundaryVelocityToRT0fluxRep(np.ndarray elementBoundaryQuadratureWeights,
                                               np.ndarray n,
                                               np.ndarray v_elementBoundary,
                                               np.ndarray rt0vdofs_element):
    pp.projectElementBoundaryVelocityToRT0fluxRep(n.shape[0],
                                                  n.shape[1],
                                                  n.shape[2],
                                                  n.shape[3],
                                                  <double*>(elementBoundaryQuadratureWeights.data),
                                                  <double*>(n.data),
                                                  <double*>(v_elementBoundary.data),
                                                  <double*>(rt0vdofs_element.data))

def projectElementBoundaryFluxToRT0fluxRep(np.ndarray elementBoundaryElementsArray,
                                           np.ndarray elementBoundariesArray,
                                           np.ndarray elementBoundaryQuadratureWeights,
                                           np.ndarray flux_elementBoundary,
                                           np.ndarray rt0vdofs_element):
    pp.projectElementBoundaryFluxToRT0fluxRep(elementBoundaryQuadratureWeights.shape[0],
                                              elementBoundaryQuadratureWeights.shape[1],
                                              elementBoundaryQuadratureWeights.shape[2],
                                              rt0vdofs_element.shape[1],
                                              <pp.proteus_index_t*>(elementBoundaryElementsArray.data),
                                              <pp.proteus_index_t*>(elementBoundariesArray.data),
                                              <double*>(elementBoundaryQuadratureWeights.data),
                                              <double*>(flux_elementBoundary.data),
                                              <double*>(rt0vdofs_element.data))

def getElementRT0velocityValuesFluxRep(np.ndarray nodeArray,
                                       np.ndarray elementNodesArray,
                                       np.ndarray abs_det_J,
                                       np.ndarray x_element,
                                       np.ndarray rt0vdofs_element,
                                       np.ndarray v_element):
    pp.getElementRT0velocityValuesFluxRep(v_element.shape[0],
                                          elementNodesArray.shape[1],
                                          v_element.shape[1],
                                          v_element.shape[2],
                                          abs_det_J.shape[1],
                                          <double*>(nodeArray.data),
                                          <pp.proteus_index_t*>(elementNodesArray.data),
                                          <double*>(abs_det_J.data),
                                          <double*>(x_element.data),
                                          <double*>(rt0vdofs_element.data),
                                          <double*>(v_element.data))

def getElementBoundaryRT0velocityValuesFluxRep(np.ndarray nodeArray,
                                               np.ndarray elementNodesArray,
                                               np.ndarray abs_det_J,
                                               np.ndarray x_elementBoundary,
                                               np.ndarray rt0vdofs_element,
                                               np.ndarray v_elementBoundary):
    pp.getElementBoundaryRT0velocityValuesFluxRep(v_elementBoundary.shape[0],
                                                  v_elementBoundary.shape[1],
                                                  v_elementBoundary.shape[2],
                                                  v_elementBoundary.shape[3],
                                                  abs_det_J.shape[1],
                                                  <double*>(nodeArray.data),
                                                  <pp.proteus_index_t*>(elementNodesArray.data),
                                                  <double*>(abs_det_J.data),
                                                  <double*>(x_elementBoundary.data),
                                                  <double*>(rt0vdofs_element.data),
                                                  <double*>(v_elementBoundary.data))
    
def getGlobalElementBoundaryRT0velocityValuesFluxRep(np.ndarray nodeArray,
                                                     np.ndarray elementNodesArray,
                                                     np.ndarray elementBoundaryElementsArray,
                                                     np.ndarray abs_det_J,
                                                     np.ndarray x_elementBoundary_global,
                                                     np.ndarray rt0vdofs_element,
                                                     np.ndarray v_elementBoundary_global):
    pp.getGlobalElementBoundaryRT0velocityValuesFluxRep(v_elementBoundary_global.shape[0],
                                                        v_elementBoundary_global.shape[1],
                                                        v_elementBoundary_global.shape[2],
                                                        abs_det_J.shape[1],
                                                        <double*>(nodeArray.data),
                                                        <pp.proteus_index_t*>(elementNodesArray.data),
                                                        <pp.proteus_index_t*>(elementBoundaryElementsArray.data),
                                                        <double*>(abs_det_J.data),
                                                        <double*>(x_elementBoundary_global.data),
                                                        <double*>(rt0vdofs_element.data),
                                                        <double*>(v_elementBoundary_global.data))

def getGlobalExteriorElementBoundaryRT0velocityValuesFluxRep(np.ndarray nodeArray,
                                                             np.ndarray elementNodesArray,
                                                             np.ndarray elementBoundaryElementsArray,
                                                             np.ndarray exteriorElementBoundariesArray,
                                                             np.ndarray abs_det_J,
                                                             np.ndarray x_ebqe,
                                                             np.ndarray rt0vdofs_element,
                                                             np.ndarray v_ebqe):
    pp.getGlobalExteriorElementBoundaryRT0velocityValuesFluxRep(v_ebqe.shape[0],
                                                                v_ebqe.shape[1],
                                                                v_ebqe.shape[2],
                                                                abs_det_J.shape[1],
                                                                <double*>(nodeArray.data),
                                                                <pp.proteus_index_t*>(elementNodesArray.data),
                                                                <pp.proteus_index_t*>(elementBoundaryElementsArray.data),
                                                                <pp.proteus_index_t*>(exteriorElementBoundariesArray.data),
                                                                <double*>(abs_det_J.data),
                                                                <double*>(x_ebqe.data),
                                                                <double*>(rt0vdofs_element.data),
                                                                <double*>(v_ebqe.data))

def getRT0velocityValuesFluxRep_arbitraryElementMembership(np.ndarray nodeArray,
                                                           np.ndarray elementNodesArray,
                                                           np.ndarray abs_det_J,
                                                           np.ndarray x,
                                                           np.ndarray element_locations,
                                                           np.ndarray rt0vdofs_element,
                                                           np.ndarray v_element):
    cdef int nPoints = x.size//x.shape[x.ndim-1]
    pp.getRT0velocityValuesFluxRep_arbitraryElementMembership(elementNodesArray.shape[0],
                                                              elementNodesArray.shape[1],
                                                              nPoints,
                                                              v_element.shape[1],
                                                              abs_det_J.shape[1],
                                                              <double*>(nodeArray.data),
                                                              <pp.proteus_index_t*>(elementNodesArray.data),
                                                              <double*>(abs_det_J.data),
                                                              <double*>(x.data),
                                                              <int*>(element_locations.data),
                                                              <double*>(rt0vdofs_element.data),
                                                              <double*>(v_element.data))

def fluxCorrectionVelocityUpdate(np.ndarray interiorElementBoundaries,
                                 np.ndarray exteriorElementBoundaries,
                                 np.ndarray elementBoundaryElements,
                                 np.ndarray elementBoundaryLocalElementBoundaries,
                                 np.ndarray dS,
                                 np.ndarray n,
                                 np.ndarray fluxCorrection,
                                 np.ndarray velocity,
                                 np.ndarray velocity_element):
    pp.fluxCorrectionVelocityUpdate(dS.shape[0],
                                    velocity.shape[0],
                                    interiorElementBoundaries.shape[0],
                                    exteriorElementBoundaries.shape[0],
                                    dS.shape[1],
                                    dS.shape[2],
                                    velocity.shape[2],
                                    <pp.proteus_index_t*>(interiorElementBoundaries.data),
                                    <pp.proteus_index_t*>(exteriorElementBoundaries.data),
                                    <pp.proteus_index_t*>(elementBoundaryElements.data),
                                    <int*>(elementBoundaryLocalElementBoundaries.data),
                                    <double*>(dS.data),
                                    <double*>(n.data),
                                    <double*>(fluxCorrection.data),
                                    <double*>(velocity.data),
                                    <double*>(velocity_element.data))

def computeFluxCorrectionPWC(np.ndarray interiorElementBoundaries,
                             np.ndarray exteriorElementBoundaries,
                             np.ndarray elementBoundaryElements,
                             np.ndarray pwcW,
                             np.ndarray pwcV,
                             np.ndarray fluxCorrection):
    pp.computeFluxCorrectionPWC(elementBoundaryElements.shape[0],
                                interiorElementBoundaries.shape[0],
                                exteriorElementBoundaries.shape[0],
                                <pp.proteus_index_t*>(interiorElementBoundaries.data),
                                <pp.proteus_index_t*>(exteriorElementBoundaries.data),
                                <pp.proteus_index_t*>(elementBoundaryElements.data),
                                <double*>(pwcW.data),
                                <double*>(pwcV.data),
                                <double*>(fluxCorrection.data))

def sunWheelerGSsweep(np.ndarray interiorElementBoundaries,
                      np.ndarray exteriorElementBoundaries,
                      np.ndarray elementBoundaryElements,
                      np.ndarray elementBoundaryLocalElementBoundaries,
                      np.ndarray dS,
                      np.ndarray n,
                      np.ndarray sqrt_det_g,
                      np.ndarray alphaFactor,
                      np.ndarray fluxCorrection,
                      np.ndarray conservationResidual):
    pp.sunWheelerGSsweep(conservationResidual.shape[0],
                         elementBoundaryElements.shape[0],
                         interiorElementBoundaries.shape[0],
                         exteriorElementBoundaries.shape[0],
                         dS.shape[1],
                         dS.shape[2],
                         n.shape[2],
                         <pp.proteus_index_t*>(interiorElementBoundaries.data),
                         <pp.proteus_index_t*>(exteriorElementBoundaries.data),
                         <pp.proteus_index_t*>(elementBoundaryElements.data),
                         <int*>(elementBoundaryLocalElementBoundaries.data),
                         <double*>(dS.data),
                         <double*>(n.data),
                         <double*>(sqrt_det_g.data),
                         <double*>(alphaFactor.data),
                         <double*>(fluxCorrection.data),
                         <double*>(conservationResidual.data))

#  put in data structure for node star solves now
cdef class NodeStarFactor:
    cdef pp.NodeStarFactorStruct nsf
    def __cinit__(self,
                  np.ndarray nElements_node,
                  np.ndarray nodeStarElementsArray,
                  np.ndarray nodeStarElementsNeighborsArray):
        cdef int rval=0
        rval = pp.nodeStar_init(nodeStarElementsArray.shape[0],
                                nodeStarElementsArray.shape[1],
                                nElements_node.shape[0],
                                <int*>(nElements_node.data),
                                <int*>(nodeStarElementsArray.data),
                                <int*>(nodeStarElementsNeighborsArray.data),
                                &self.nsf.N,
                                &self.nsf.subdomain_dim,
                                &self.nsf.subdomain_L,
                                &self.nsf.subdomain_R,
                                &self.nsf.subdomain_U,
                                &self.nsf.subdomain_pivots,
                                &self.nsf.subdomain_column_pivots)
        assert rval == 0

    def __dealloc__(self):
        pp.nodeStar_free(self.nsf.N, 
                         self.nsf.subdomain_dim,
                         self.nsf.subdomain_L,
                         self.nsf.subdomain_R,
                         self.nsf.subdomain_U,
                         self.nsf.subdomain_pivots,
                         self.nsf.subdomain_column_pivots)
    
    def setU(self,
             double val):
        cdef int I
        cdef int i
        for I in range(self.nsf.N):
            for i in range(self.nsf.subdomain_dim[I]):
                self.nsf.subdomain_U[I][i] = val

    def getSubdomainU(self, int I):
        """A copy of the solution of node I's star problem, one value per element of the star."""
        assert 0 <= I < self.nsf.N
        cdef int i
        cdef np.ndarray[np.double_t, ndim=1] U = np.empty(self.nsf.subdomain_dim[I], dtype=np.double)
        for i in range(self.nsf.subdomain_dim[I]):
            U[i] = self.nsf.subdomain_U[I][i]
        return U

    def setSubdomainU(self, int I, np.ndarray[np.double_t, ndim=1] U):
        """Overwrite the solution of node I's star problem."""
        assert 0 <= I < self.nsf.N
        assert U.shape[0] == self.nsf.subdomain_dim[I]
        cdef int i
        for i in range(self.nsf.subdomain_dim[I]):
            self.nsf.subdomain_U[I][i] = U[i]

    def copyData(self,
                 NodeStarFactor other):
        pp.nodeStar_copy(other.nsf.N, 
                         other.nsf.subdomain_dim,
                         other.nsf.subdomain_L,
                         other.nsf.subdomain_R,
                         other.nsf.subdomain_U,
                         other.nsf.subdomain_pivots,
                         other.nsf.subdomain_column_pivots,
                         &self.nsf.N, 
                         &self.nsf.subdomain_dim,
                         &self.nsf.subdomain_L,
                         &self.nsf.subdomain_R,
                         &self.nsf.subdomain_U,
                         &self.nsf.subdomain_pivots,
                         &self.nsf.subdomain_column_pivots)

def calculateConservationResidualPWL(np.ndarray interiorElementBoundaries,
                                     np.ndarray exteriorElementBoundaries,
                                     np.ndarray elementBoundaryElements,
                                     np.ndarray elementBoundaryLocalElementBoundaries,
                                     np.ndarray elementNodes,
                                     np.ndarray dofMapl2g,
                                     np.ndarray nodeStarElements,
                                     np.ndarray nodeStarElementNeighbors,
                                     np.ndarray nElements_node,
                                     np.ndarray fluxElementBoundaries,
                                     np.ndarray elementResidual,
                                     np.ndarray vAverage,
                                     np.ndarray dx,
                                     np.ndarray w,
                                     np.ndarray n,
                                     NodeStarFactor nodeStarFactor,
                                     np.ndarray conservationResidual,
                                     np.ndarray vConservative,
                                     np.ndarray vConservative_element):
    pp.calculateConservationResidualPWL(w.shape[0],
                                        #/* nElements_global*/, 
                                        interiorElementBoundaries.shape[0], 
                                        #/*nInteriorElementBoundaries_global*/,
                                        exteriorElementBoundaries.shape[0],
                                        #/*nExteriorElementBoundaries_global*/,
                                        w.shape[1],
                                        #/*nElementBoundaries_element*/,
                                        w.shape[2],
                                        #/*nQuadraturePoints_elementBoundary*/,
                                        w.shape[3],
                                        #/*nNodes_element*/,
                                        dofMapl2g.shape[1],
                                        #/*nDOF_element*/,
                                        n.shape[2],
                                        #/*nSpace */,
                                        <pp.proteus_index_t*>(interiorElementBoundaries.data),
                                        <pp.proteus_index_t*>(exteriorElementBoundaries.data),
                                        <pp.proteus_index_t*>(elementBoundaryElements.data),
                                        <int*>(elementBoundaryLocalElementBoundaries.data),
                                        <pp.proteus_index_t*>(elementNodes.data),
                                        <int*>(dofMapl2g.data),
                                        <pp.proteus_index_t*>(nodeStarElements.data),
                                        <pp.proteus_index_t*>(nodeStarElementNeighbors.data),
                                        <int*>(nElements_node.data),
                                        <pp.proteus_index_t*>(fluxElementBoundaries.data),
                                        <double*>(elementResidual.data),
                                        <double*>(vAverage.data),
                                        <double*>(dx.data),
                                        <double*>(w.data),
                                        <double*>(n.data),
                                        &nodeStarFactor.nsf,
                                        <double*>(conservationResidual.data),
                                        <double*>(vConservative.data),
                                        <double*>(vConservative_element.data))

def calculateConservationJacobianPWL(np.ndarray interiorElementBoundaries,
                                     np.ndarray exteriorElementBoundaries,
                                     np.ndarray elementBoundaryElements,
                                     np.ndarray elementBoundaryLocalElementBoundaries,
                                     np.ndarray elementNodes,
                                     np.ndarray dofMapl2g,
                                     np.ndarray dofStarElements,
                                     np.ndarray dofStarElementNeighbors,
                                     np.ndarray nElements_dof,
                                     np.ndarray internalNodes,
                                     np.ndarray fluxElementBoundaries,
                                     np.ndarray fluxBoundaryNodes,
                                     np.ndarray w,
                                     np.ndarray n,
                                     NodeStarFactor nodeStarFactor):
    pp.calculateConservationJacobianPWL(nElements_dof.shape[0],
                                        internalNodes.shape[0],
                                        w.shape[0],
                                        interiorElementBoundaries.shape[0],
                                        exteriorElementBoundaries.shape[0],
                                        w.shape[1],
                                        w.shape[2],
                                        w.shape[3],
                                        dofMapl2g.shape[1],
                                        n.shape[2],
                                        <pp.proteus_index_t*>(interiorElementBoundaries.data),
                                        <pp.proteus_index_t*>(exteriorElementBoundaries.data),
                                        <pp.proteus_index_t*>(elementBoundaryElements.data),
                                        <int*>(elementBoundaryLocalElementBoundaries.data),
                                        <pp.proteus_index_t*>(elementNodes.data),
                                        <int*>(dofMapl2g.data),
                                        <int*>(dofStarElements.data),
                                        <int*>(dofStarElementNeighbors.data),
                                        <int*>(nElements_dof.data),
                                        <pp.proteus_index_t*>(internalNodes.data),
                                        <pp.proteus_index_t*>(fluxElementBoundaries.data),
                                        <int*>(fluxBoundaryNodes.data),
                                        <double*>(w.data),
                                        <double*>(n.data),
                                        &nodeStarFactor.nsf)

def calculateConservationFluxPWL(np.ndarray nElements_node,
                                 np.ndarray internalNodes,
                                 np.ndarray fluxBoundaryNodes,
                                 NodeStarFactor nodeStarFactor):
    pp.calculateConservationFluxPWL(nElements_node.shape[0],
                                    internalNodes.shape[0],
                                    <int*>(nElements_node.data),
                                    <pp.proteus_index_t*>(internalNodes.data),
                                    <int*>(fluxBoundaryNodes.data),
                                    &nodeStarFactor.nsf)

def calculateConservationFluxPWL_noNeumannFix(np.ndarray nElements_node,
                                              NodeStarFactor nodeStarFactor):
    pp.calculateConservationFluxPWL_noNeumannFix(nElements_node.shape[0],
                                                 <int*>(nElements_node.data),
                                                 &nodeStarFactor.nsf)

def calculateConservationResidualPWL_opt(pp.proteus_index_t nNodes_owned,
                                         np.ndarray interiorElementBoundaries,
                                         np.ndarray exteriorElementBoundaries,
                                         np.ndarray elementBoundaryElements,
                                         np.ndarray elementBoundaryLocalElementBoundaries,
                                         np.ndarray elementNodes,
                                         np.ndarray nodeStarElements,
                                         np.ndarray nodeStarElementNeighbors,
                                         np.ndarray nElements_node,
                                         np.ndarray fluxElementBoundaries,
                                         np.ndarray elementResidual,
                                         np.ndarray vAverage,
                                         np.ndarray dx,
                                         np.ndarray w,
                                         np.ndarray n,
                                         NodeStarFactor nodeStarFactor,
                                         np.ndarray conservationResidual,
                                         np.ndarray vConservative,
                                         np.ndarray vConservative_element):
    pp.calculateConservationResidualPWL_opt(nNodes_owned,
                                            w.shape[0],
                                            interiorElementBoundaries.shape[0],
                                            exteriorElementBoundaries.shape[0],
                                            w.shape[1],
                                            w.shape[2],
                                            w.shape[3],
                                            n.shape[2],
                                            <pp.proteus_index_t*>(interiorElementBoundaries.data),
                                            <pp.proteus_index_t*>(exteriorElementBoundaries.data),
                                            <pp.proteus_index_t*>(elementBoundaryElements.data),
                                            <int*>(elementBoundaryLocalElementBoundaries.data),
                                            <pp.proteus_index_t*>(elementNodes.data),
                                            <pp.proteus_index_t*>(nodeStarElements.data),
                                            <pp.proteus_index_t*>(nodeStarElementNeighbors.data),
                                            <int*>(nElements_node.data),
                                            <pp.proteus_index_t*>(fluxElementBoundaries.data),
                                            <double*>(elementResidual.data),
                                            <double*>(vAverage.data),
                                            <double*>(dx.data),
                                            <double*>(w.data),
                                            <double*>(n.data),
                                            &nodeStarFactor.nsf,
                                            <double*>(conservationResidual.data),
                                            <double*>(vConservative.data),
                                            <double*>(vConservative_element.data))

def calculateConservationResidualPWL_primative(np.ndarray interiorElementBoundaries,
                                               np.ndarray exteriorElementBoundaries,
                                               np.ndarray elementBoundaryElements,
                                               np.ndarray elementBoundaryLocalElementBoundaries,
                                               np.ndarray skipflag_elementBoundaries,
                                               np.ndarray elementResidual,
                                               np.ndarray dx,
                                               np.ndarray n,
                                               np.ndarray conservationResidual,
                                               np.ndarray vConservative):
    pp.calculateConservationResidualPWL_primative(dx.shape[0],
                                                  interiorElementBoundaries.shape[0],
                                                  exteriorElementBoundaries.shape[0],
                                                  dx.shape[1],
                                                  dx.shape[2],
                                                  elementResidual.shape[1],
                                                  n.shape[2],
                                                  <pp.proteus_index_t*>(interiorElementBoundaries.data),
                                                  <pp.proteus_index_t*>(exteriorElementBoundaries.data),
                                                  <pp.proteus_index_t*>(elementBoundaryElements.data),
                                                  <int*>(elementBoundaryLocalElementBoundaries.data),
                                                  <int*>(skipflag_elementBoundaries.data),
                                                  <double*>(elementResidual.data),
                                                  <double*>(dx.data),
                                                  <double*>(n.data),
                                                  <double*>(conservationResidual.data),
                                                  <double*>(vConservative.data))

def calculateConservationJacobianPWL_opt(pp.proteus_index_t nNodes_owned,
                                         np.ndarray interiorElementBoundaries,
                                         np.ndarray exteriorElementBoundaries,
                                         np.ndarray elementBoundaryElements,
                                         np.ndarray elementBoundaryLocalElementBoundaries,
                                         np.ndarray elementNodes,
                                         np.ndarray nodeStarElements,
                                         np.ndarray nodeStarElementNeighbors,
                                         np.ndarray nElements_node,
                                         np.ndarray internalNodes,
                                         np.ndarray fluxElementBoundaries,
                                         np.ndarray fluxBoundaryNodes,
                                         np.ndarray w,
                                         np.ndarray n,
                                         NodeStarFactor nodeStarFactor):
    pp.calculateConservationJacobianPWL_opt(nNodes_owned,
                                            nElements_node.shape[0],
                                            internalNodes.shape[0],
                                            w.shape[0],
                                            interiorElementBoundaries.shape[0],
                                            exteriorElementBoundaries.shape[0],
                                            w.shape[1],
                                            w.shape[2],
                                            w.shape[3],
                                            n.shape[2],
                                            <pp.proteus_index_t*>(interiorElementBoundaries.data),
                                            <pp.proteus_index_t*>(exteriorElementBoundaries.data),
                                            <pp.proteus_index_t*>(elementBoundaryElements.data),
                                            <int*>(elementBoundaryLocalElementBoundaries.data),
                                            <pp.proteus_index_t*>(elementNodes.data),
                                            <pp.proteus_index_t*>(nodeStarElements.data),
                                            <pp.proteus_index_t*>(nodeStarElementNeighbors.data),
                                            <int*>(nElements_node.data),
                                            <pp.proteus_index_t*>(internalNodes.data),
                                            <pp.proteus_index_t*>(fluxElementBoundaries.data),
                                            <int*>(fluxBoundaryNodes.data),
                                            <double*>(w.data),
                                            <double*>(n.data),
                                            &nodeStarFactor.nsf)

def calculateConservationFluxPWL_opt(pp.proteus_index_t nNodes_owned,
                                     np.ndarray nElements_node,
                                     np.ndarray internalNodes,
                                     np.ndarray fluxBoundaryNodes,
                                     NodeStarFactor nodeStarFactor):
    pp.calculateConservationFluxPWL_opt(nNodes_owned,
                                        nElements_node.shape[0],
                                        internalNodes.shape[0],
                                        <int*>(nElements_node.data),
                                        <pp.proteus_index_t*>(internalNodes.data),
                                        <int*>(fluxBoundaryNodes.data),
                                        &nodeStarFactor.nsf)

def calculateConservationResidualPWL_interiorBoundaries(np.ndarray interiorElementBoundaries,
                                                        np.ndarray exteriorElementBoundaries,
                                                        np.ndarray elementBoundaryElements,
                                                        np.ndarray elementBoundaryLocalElementBoundaries,
                                                        np.ndarray elementNodes,
                                                        np.ndarray nodeStarElements,
                                                        np.ndarray nodeStarElementNeighbors,
                                                        np.ndarray nElements_node,
                                                        np.ndarray fluxElementBoundaries,
                                                        np.ndarray elementResidual,
                                                        np.ndarray vAverage,
                                                        np.ndarray dx,
                                                        np.ndarray w,
                                                        np.ndarray n,
                                                        NodeStarFactor nodeStarFactor,
                                                        np.ndarray conservationResidual,
                                                        np.ndarray vConservative,
                                                        np.ndarray vConservative_element):
    assert fluxElementBoundaries.shape[0] == elementBoundaryLocalElementBoundaries.shape[0]
    pp.calculateConservationResidualPWL_interiorBoundaries(w.shape[0],
                                                           interiorElementBoundaries.shape[0],
                                                           exteriorElementBoundaries.shape[0],
                                                           w.shape[1],
                                                           w.shape[2],
                                                           w.shape[3],
                                                           n.shape[2],
                                                           <pp.proteus_index_t*>(interiorElementBoundaries.data),
                                                           <pp.proteus_index_t*>(exteriorElementBoundaries.data),
                                                           <pp.proteus_index_t*>(elementBoundaryElements.data),
                                                           <int*>(elementBoundaryLocalElementBoundaries.data),
                                                           <pp.proteus_index_t*>(elementNodes.data),
                                                           <pp.proteus_index_t*>(nodeStarElements.data),
                                                           <pp.proteus_index_t*>(nodeStarElementNeighbors.data),
                                                           <int*>(nElements_node.data),
                                                           <pp.proteus_index_t*>(fluxElementBoundaries.data),
                                                           <double*>(elementResidual.data),
                                                           <double*>(vAverage.data),
                                                           <double*>(dx.data),
                                                           <double*>(w.data),
                                                           <double*>(n.data),
                                                           &nodeStarFactor.nsf,
                                                           <double*>(conservationResidual.data),
                                                           <double*>(vConservative.data),
                                                           <double*>(vConservative_element.data))

def calculateConservationJacobianPWL_interiorBoundaries(np.ndarray interiorElementBoundaries,
                                                        np.ndarray exteriorElementBoundaries,
                                                        np.ndarray elementBoundaryElements,
                                                        np.ndarray elementBoundaryLocalElementBoundaries,
                                                        np.ndarray elementNodes,
                                                        np.ndarray nodeStarElements,
                                                        np.ndarray nodeStarElementNeighbors,
                                                        np.ndarray nElements_node,
                                                        np.ndarray fluxElementBoundaries,
                                                        np.ndarray w,
                                                        np.ndarray n,
                                                        NodeStarFactor nodeStarFactor):
    assert fluxElementBoundaries.shape[0] == elementBoundaryLocalElementBoundaries.shape[0]
    pp.calculateConservationJacobianPWL_interiorBoundaries(nElements_node.shape[0],
                                                           w.shape[0],
                                                           interiorElementBoundaries.shape[0],
                                                           exteriorElementBoundaries.shape[0],
                                                           w.shape[1],
                                                           w.shape[2],
                                                           w.shape[3],
                                                           n.shape[2],
                                                           <pp.proteus_index_t*>(interiorElementBoundaries.data),
                                                           <pp.proteus_index_t*>(exteriorElementBoundaries.data),
                                                           <pp.proteus_index_t*>(elementBoundaryElements.data),
                                                           <int*>(elementBoundaryLocalElementBoundaries.data),
                                                           <pp.proteus_index_t*>(elementNodes.data),
                                                           <pp.proteus_index_t*>(nodeStarElements.data),
                                                           <pp.proteus_index_t*>(nodeStarElementNeighbors.data),
                                                           <int*>(nElements_node.data),
                                                           <pp.proteus_index_t*>(fluxElementBoundaries.data),
                                                           <double*>(w.data),
                                                           <double*>(n.data),
                                                           &nodeStarFactor.nsf)

def _subdomain_U_copy_global2local(int max_nN_owned,
                                   np.ndarray elementNodes,
                                   np.ndarray nodeStarElements,
                                   NodeStarFactor nodeStarFactor,
                                   np.ndarray subdomain_U):
    pp.subdomain_U_copy_global2local(max_nN_owned,
                                     subdomain_U.shape[0],
                                     subdomain_U.shape[1],
                                     <pp.proteus_index_t*>(elementNodes.data),
                                     <pp.proteus_index_t*>(nodeStarElements.data),
                                     &nodeStarFactor.nsf,
                                     <double*>(subdomain_U.data))

def _subdomain_U_copy_local2global(int max_nN_owned,
                                   np.ndarray elementNodes,
                                   np.ndarray nodeStarElements,
                                   NodeStarFactor nodeStarFactor,
                                   np.ndarray subdomain_U):
    pp.subdomain_U_copy_local2global(max_nN_owned,
                                     subdomain_U.shape[0],
                                     subdomain_U.shape[1],
                                     <pp.proteus_index_t*>(elementNodes.data),
                                     <pp.proteus_index_t*>(nodeStarElements.data),
                                     &nodeStarFactor.nsf,
                                     <double*>(subdomain_U.data))

def calculateConservationResidualGlobalBoundaries(np.ndarray interiorElementBoundaries,
                                                  np.ndarray exteriorElementBoundaries,
                                                  np.ndarray elementBoundaryElements,
                                                  np.ndarray elementBoundaryLocalElementBoundaries,
                                                  np.ndarray exteriorElementBoundariesToSkip,
                                                  np.ndarray dS,
                                                  np.ndarray n,
                                                  np.ndarray elementResidual,
                                                  np.ndarray velocity,
                                                  np.ndarray conservationResidual):
    pp.calculateConservationResidualGlobalBoundaries(conservationResidual.shape[0],
                                                     interiorElementBoundaries.shape[0],
                                                     exteriorElementBoundaries.shape[0],
                                                     dS.shape[1],
                                                     dS.shape[2],
                                                     elementResidual.shape[1],
                                                     velocity.shape[2],
                                                     <pp.proteus_index_t*>(interiorElementBoundaries.data),
                                                     <pp.proteus_index_t*>(exteriorElementBoundaries.data),
                                                     <pp.proteus_index_t*>(elementBoundaryElements.data),
                                                     <int*>(elementBoundaryLocalElementBoundaries.data),
                                                     <int*>(exteriorElementBoundariesToSkip.data),
                                                     <double*>(dS.data),
                                                     <double*>(n.data),
                                                     <double*>(elementResidual.data),
                                                     <double*>(velocity.data),
                                                     <double*>(conservationResidual.data))

def updateSelectedExteriorElementBoundaryFlux(np.ndarray exteriorElementBoundaries,
                                              np.ndarray elementBoundaryElements,
                                              np.ndarray elementBoundaryLocalElementBoundaries,
                                              np.ndarray skipflag_elementBoundaries,
                                              np.ndarray flux,
                                              np.ndarray w,
                                              np.ndarray residual):
    pp.updateSelectedExteriorElementBoundaryFlux(exteriorElementBoundaries.shape[0],
                                                 w.shape[1],
                                                 w.shape[2],
                                                 w.shape[3],
                                                 <pp.proteus_index_t*>(exteriorElementBoundaries.data),
                                                 <pp.proteus_index_t*>(elementBoundaryElements.data),
                                                 <int*>(elementBoundaryLocalElementBoundaries.data),
                                                 <int*>(skipflag_elementBoundaries.data),
                                                 <double*>(flux.data),
                                                 <double*>(w.data),
                                                 <double*>(residual.data))

def updateAdvectiveVelocityPointEval(double updateCoef,
                                     np.ndarray advectiveVelocity,
                                     np.ndarray velocity):
    cdef int nPoints = velocity.size//velocity.shape[velocity.ndim-1]
    pp.postprocessAdvectiveVelocityPointEval(nPoints,
                                             velocity.shape[velocity.ndim-1],
                                             updateCoef,
                                             <double*>(advectiveVelocity.data),
                                             <double*>(velocity.data))

def updateDiffusiveVelocityPointEval(double updateCoef,
                                     np.ndarray diffusionTensor,
                                     np.ndarray grad_phi,
                                     np.ndarray velocity):
    cdef int nPoints = velocity.size//velocity.shape[velocity.ndim-1]
    pp.postprocessDiffusiveVelocityPointEval(nPoints,
                                             velocity.shape[velocity.ndim-1],
                                             updateCoef,
                                             <double*>(diffusionTensor.data),
                                             <double*>(grad_phi.data),
                                             <double*>(velocity.data))

def updateDiffusiveVelocityPointEval_sd(double updateCoef,
                                        np.ndarray rowptr,
                                        np.ndarray colind,
                                        np.ndarray diffusionTensor,
                                        np.ndarray grad_phi,
                                        np.ndarray velocity):
    cdef int nPoints = velocity.size//velocity.shape[velocity.ndim-1]
    pp.postprocessDiffusiveVelocityPointEval_sd(nPoints,
                                                velocity.shape[velocity.ndim-1],
                                                updateCoef,
                                                <pp.proteus_index_t*>(rowptr.data),
                                                <pp.proteus_index_t*>(colind.data),
                                                <double*>(diffusionTensor.data),
                                                <double*>(grad_phi.data),
                                                <double*>(velocity.data))

def calculateElementResidualPWL(np.ndarray alpha,
                                np.ndarray elementResidual,
                                np.ndarray elementResidualPWL):
    pp.calculateElementResidualPWL(elementResidual.shape[0],
                                   elementResidual.shape[1],
                                   elementResidualPWL.shape[1],
                                   <double*>(alpha.data),
                                   <double*>(elementResidual.data),
                                   <double*>(elementResidualPWL.data))

def copyElementBoundaryVelocityToParVec(np.ndarray ebq_velocity, 
                                        np.ndarray[int, ndim=2] permutations, 
                                        np.ndarray ebq_v_par_local):
    cdef int ebN
    cdef int k
    cdef int I
    cdef int nElementBoundaries=ebq_v_par_local.shape[0]
    cdef int nQuadraturePoints=ebq_v_par_local.shape[1]
    cdef int nSpace=ebq_v_par_local.shape[2]
    for ebN in range(nElementBoundaries):
        for k in range(nQuadraturePoints):
            for I in range(nSpace):
                ebq_v_par_local[ebN,permutations[ebN,k],I] = ebq_velocity[ebN,k,I]

def addAverageToParVec(np.ndarray ebq_velocityAverage, 
                       np.ndarray[int, ndim=2] permutations, 
                       np.ndarray ebq_v_par_local):
    cdef int ebN
    cdef int k
    cdef int I
    cdef int nElementBoundaries=ebq_v_par_local.shape[0]
    cdef int nQuadraturePoints=ebq_v_par_local.shape[1]
    cdef int nSpace=ebq_v_par_local.shape[2]
    for  ebN in range(nElementBoundaries):
        for k in range(nQuadraturePoints):
            for I in range(nSpace):
                ebq_v_par_local[ebN, permutations[ebN,k], I] += ebq_velocityAverage[ebN, k, I]

def copyParVecToElementBoundaryVelocity(np.ndarray ebq_velocity, 
                                        np.ndarray[int, ndim=2] permutations, 
                                        np.ndarray ebq_v_par_local):
    cdef int ebN
    cdef int k
    cdef int I
    cdef int nElementBoundaries=ebq_v_par_local.shape[0]
    cdef int nQuadraturePoints=ebq_v_par_local.shape[1]
    cdef int nSpace=ebq_v_par_local.shape[2]
    for ebN in range(nElementBoundaries):
        for k in range(nQuadraturePoints):
            for I in range(nSpace):
                ebq_velocity[ebN, k, I] = ebq_v_par_local[ebN, permutations[ebN, k], I]
