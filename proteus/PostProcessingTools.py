#! /usr/bin/env python
"""
Collect classes and routines for postprocessing solution to get
 quantities like conservative velocities, higher accuracy, etc

.. inheritance-diagram:: proteus.PostProcessingTools
   :parts: 1
"""
import numpy
from . import FemTools
from . import LinearSolvers
from .LinearAlgebraTools import Mat,Vec,SparseMatFromDict
from . import cfemIntegrals
from . import cpostprocessing
from .Profiling import logEvent
from . import Norms
from . import Archiver
from warnings import warn

def VelocityPostProcessingChooser(transport):
    """
    pick acceptable velocity postprocessing based on input
    """
    velocityPostProcessor = None
    if transport.conservativeFlux is not None:
        if (transport.mesh.parallelPartitioningType == 0 and transport.mesh.nLayersOfOverlap==0): #element-based partition
            # an owned element's faces on the rank interface have no neighbour there, so no variant has their flux;
            # this used to call exit(), which ended every rank with status 0
            raise RuntimeError("conservativeFlux needs an overlap with element partitioning: set "
                               "nLayersOfOverlapForParallel = 1 (or use node partitioning)")
        ppcomps = []
        pptypes = {}
        for ci in list(transport.conservativeFlux.keys()):
            if (transport.conservativeFlux[ci] == 'p1-nc' and
                isinstance(transport.u[ci].femSpace,FemTools.NC_AffineLinearOnSimplexWithNodalBasis)):
                ppcomps.append(ci)
                pptypes[ci] = 'p1-nc'
            #end p1-nc for comp ci
            elif 'pwl' in transport.conservativeFlux[ci]:
                ppcomps.append(ci)
                pptypes[ci] = transport.conservativeFlux[ci]
            elif transport.conservativeFlux[ci] in ['point-eval','dg-point-eval','point-eval-gwvd']: #tjp addin for gwvd
                ppcomps.append(ci)
                pptypes[ci] = transport.conservativeFlux[ci]
            elif transport.conservativeFlux[ci] == 'pwc':
                ppcomps.append(ci)
                pptypes[ci] = 'pwc'
            elif 'sun-' in transport.conservativeFlux[ci]:
                ppcomps.append(ci)
                pptypes[ci] = transport.conservativeFlux[ci]
            elif transport.conservativeFlux[ci] in ['dg','dg-bdm']:
                ppcomps.append(ci)
                pptypes[ci] = transport.conservativeFlux[ci]
            else:
                logEvent("Unrecognized conservative flux", transport.conservativeFlux[ci])
        #for ci
        velocityPostProcessor = AggregateVelocityPostProcessor(pptypes,transport)
    #conservative flux specified
    return velocityPostProcessor


def _requireOneRank(postProcessingType):
    """The global-correction variants solve one system over the mesh they are given, which on more than one rank is
    the rank's subdomain, not the domain."""
    from . import Comm
    if Comm.get().size() > 1:
        raise RuntimeError("conservativeFlux '%s' runs on one rank only: it solves one global correction problem, "
                           "and on %d ranks each would solve its own subdomain's; use 'pwl' or 'pwl-bdm'"
                           % (postProcessingType, Comm.get().size()))


#####################################################################################################
#begin pulling out different velocity cases into separate classes to make this more manageable
#####################################################################################################

class VelocityPostProcessingAlgorithmBase(object):
    """ Base class for velocity post processing algorithms
    
    Applies same algorithm to all components in vtComponents

    TODO
    make velocity representation type an enum

    Attributes
    ----------
    q[('velocity',ci)] : array type
        Stores the velocity values on element quadrature points.
    ebq[('velocity',ci)] : array type
        Stores the velocity values on element boundary quadrature points.  
        Designed to reference the element number first.
    ebq_global[('velocity',ci)] : array
        Stores the velocity values on the boundary.  Designed to
        reference the global edge number first.    
    """
    def __init__(self,postProcessingType=None,vectorTransport=None,vtComponents=[0]):
        """ VelocityPostProcessingAlgorithmBase Base Class Constructor

        Parameters
        ----------
        postProcessingType : str
            String value of the post-processing type to be applied
        vectorTransport : :class:`proteus.Transport.OneLevelTransport`
            Vector Transport class object describing the problem.
        vtComponents : list
            List of indices for different vector transport components.
        """

        self.postProcessingType=postProcessingType
        self.vt = vectorTransport
        self.vtComponents = vtComponents
        self.q = None
        self.ebq_global = None
        assert self.vt is not None, "vt is None not allowed for %s " %  self.postProcessingType

        #quadrature arrays
        self.q = self.vt.q
        self.ebq_global = self.vt.ebq_global
        self.ebq = self.vt.ebq
        self.ebqe = self.vt.ebqe

        #test spaces and weights
        self.qv={}
        self.w={}
        self.w_dS={}


        #information for enforcing Neumann boundaries directly,
        #start with original approach then modify to be cleaner
        self.fluxElementBoundaries = {}
        self.fluxBoundaryNodes     = {}

        #how is the local velocity represented
        # -1 -- no local space (e.g. just quad values)
        #  0 -- BDM P^1 Lagrange rep
        #  1 -- RT0, local rep is \vec a + b \vec x
        #  2 -- RT0, local rep is \sum^d_{i=0}V^i\vec N_{T,i},
        #           \vec N_{T,i} = \frac{1}{d|E|}(\vec x - p_i), i=0,...,d
        self.localVelocityRepresentationFlag = -1
        #information for
        for ci in self.vtComponents:
            ##make sure velocity entries are in the transport quadrature dictionaries
            if ('velocity',ci) not in self.q:
                self.q[('velocity',ci)]      = numpy.zeros((self.vt.mesh.nElements_global,
                                                            self.vt.nQuadraturePoints_element,
                                                            self.vt.nSpace_global),'d')
            if ('velocity',ci) not in self.ebq:
                self.ebq[('velocity',ci)]     = numpy.zeros((self.vt.mesh.nElements_global,
                                                             self.vt.mesh.nElementBoundaries_element,
                                                             self.vt.nElementBoundaryQuadraturePoints_elementBoundary,
                                                             self.vt.nSpace_global),'d')
            if ('velocity',ci) not in self.ebq_global:
                self.ebq_global[('velocity',ci)] = numpy.zeros((self.vt.mesh.nElementBoundaries_global,
                                                                self.vt.nElementBoundaryQuadraturePoints_elementBoundary,
                                                                self.vt.nSpace_global),'d')

            ##
            self.q[('conservationResidual',ci)] = numpy.zeros((self.vt.mesh.nElements_global,),'d')


            ##setup test space information to be weak reference to transport class by default
            ##only necessary to change for now if the approximation is > p1
            self.w[ci] = self.ebq[('w',ci)]
            self.qv[ci] = self.q[('v',ci)]
            if ('w*dS_f',ci) not in self.ebq:
                self.w_dS[ci] = numpy.zeros(
                    (self.vt.mesh.nElements_global,
                     self.vt.mesh.nElementBoundaries_element,
                     self.vt.nElementBoundaryQuadraturePoints_elementBoundary,
                     self.vt.nDOF_test_element[ci]),
                    'd')
                        #since self.ebq = self.vt.ebq this gets updated by vt when the mesh changes
                cfemIntegrals.calculateWeightedShapeTrace(self.vt.elementBoundaryQuadratureWeights[('u',ci)],
                                                          self.ebq['sqrt(det(g))'],
                                                          self.ebq[('w',ci)],
                                                          self.w_dS[ci])
            else:
                self.w_dS[ci] = self.ebq[('w*dS_f',ci)]


            ##determine flux boundary information as before to start?
            if self.vt.numericalFlux is not None and self.vt.numericalFlux.useWeakDirichletConditions:
                self.fluxElementBoundaries[ci] = numpy.ones((self.vt.mesh.nExteriorElementBoundaries_global,),'i')
            else:
                self.fluxElementBoundaries[ci] = numpy.zeros((self.vt.mesh.nExteriorElementBoundaries_global,),'i')

            for cj,fbcObject  in self.vt.fluxBoundaryConditionsObjectsDict.items():
                for t,g in fbcObject.advectiveFluxBoundaryConditionsDict.items():
                    if cj == ci:
                        self.fluxElementBoundaries[cj][t[0]] = 1
                #repeat for diffusive flux boundary conditions too
                for ck,diffusiveFluxBoundaryConditionsDict in fbcObject.diffusiveFluxBoundaryConditionsDictDict.items():
                    for t,g in diffusiveFluxBoundaryConditionsDict.items():
                        if ck == ci:
                            self.fluxElementBoundaries[ck][t[0]]=1
                        #diag
                     #
                #
            #end setting flux boundary elements
            # a face the partition cut (flag -1) is interior to the domain: its flux is not a boundary condition,
            # and the transport's boundary flux there is the subdomain's, not the solution's
            cut = self.vt.mesh.elementBoundaryMaterialTypes[self.vt.mesh.exteriorElementBoundariesArray] == -1
            self.fluxElementBoundaries[ci][cut] = 0
        #
        self.velocityWriter = Archiver.XdmfWriter()
    def computeGeometricInfo(self):
        pass
    def postprocess(self,verbose=0):
        """generate post-processed velocity field for attached VectorTransport
        object and store them in local dictionaries

        """
        for ci in self.vtComponents:
            self.postprocess_component(ci,verbose=verbose)

    def postprocess_component(self,ci,verbose=0):
        """
        main functionality
        generate post-processed velocity field component ci

        """
        assert False, "must override postprocess_component"
    def evaluateElementVelocityField(self,x,ci):
        """evaluate velocity field assuming velocity_dofs already calculated
        for now assumes x shaped like nE x nq x 3

        """
        assert False, "must override postprocess"
        return None
    def removeBoundaryFluxesFromResidual(self,ci,flag_elementBoundaries=None):
#        TODO:
#         get rid of temporary creation, just add a scalar argument like daxpy
        flux = -1.0*self.vt.ebq_global[('totalFlux',ci)]
        if flag_elementBoundaries is None:
            cfemIntegrals.updateExteriorElementBoundaryFlux(self.vt.mesh.exteriorElementBoundariesArray,
                                                            self.vt.mesh.elementBoundaryElementsArray,
                                                            self.vt.mesh.elementBoundaryLocalElementBoundariesArray,
                                                            flux,
                                                            self.w_dS[ci],
                                                            self.elementResidual[ci])
        else:
            cpostprocessing.updateSelectedExteriorElementBoundaryFlux(self.vt.mesh.exteriorElementBoundariesArray,
                                                                      self.vt.mesh.elementBoundaryElementsArray,
                                                                      self.vt.mesh.elementBoundaryLocalElementBoundariesArray,
                                                                      flag_elementBoundaries,
                                                                      flux,
                                                                      self.w_dS[ci],
                                                                      self.elementResidual[ci])

    def addBoundaryFluxesBackToResidual(self,ci,flag_elementBoundaries=None):
#        TODO
#           add scalar multiple for call like remove boundary fluxes
        flux = self.vt.ebq_global[('totalFlux',ci)]
        if flag_elementBoundaries is None:
            cfemIntegrals.updateExteriorElementBoundaryFlux(self.vt.mesh.exteriorElementBoundariesArray,
                                                            self.vt.mesh.elementBoundaryElementsArray,
                                                            self.vt.mesh.elementBoundaryLocalElementBoundariesArray,
                                                            flux,
                                                            self.w_dS[ci],
                                                            self.elementResidual[ci])
        else:
            cpostprocessing.updateSelectedExteriorElementBoundaryFlux(self.vt.mesh.exteriorElementBoundariesArray,
                                                                      self.vt.mesh.elementBoundaryElementsArray,
                                                                      self.vt.mesh.elementBoundaryLocalElementBoundariesArray,
                                                                      flag_elementBoundaries,
                                                                      flux,
                                                                      self.w_dS[ci],
                                                                      self.elementResidual[ci])


    #
    def archiveVelocityValues(self,archive,t,tCount,initialPhase=False,meshChanged=False):
        """
        write out post processed velocity values as a finite element space

        """
#        TODO
#           test
#           reuse data arrays
        self.velocityWriter.writeMeshXdmf_LowestOrderMixed(archive,
                                                           self.vt.mesh,
                                                           self.vt.nSpace_global,
                                                           t=t,init=initialPhase,
                                                           meshChanged=meshChanged,arGrid=None,
                                                           tCount=tCount)
        dgnodes = self.vt.mesh.nodeArray[self.vt.mesh.elementNodesArray]
        for ci in self.vtComponents:
            velocity= self.evaluateElementVelocityField(dgnodes,ci)
            self.velocityWriter.writeVectorFunctionXdmf_LowestOrderMixed(archive,velocity,tCount,init=initialPhase,name="velocity_vpp"+"_%s" % ci)

    def getElementwiseFlux(self,ci):
        """ 
        Calculates elementwise fluxes given the boundary velocities.
        
        Arguments
        ---------
        ci : int
            The component number
        
        Notes
        -----
        This function has not been extensively tested and it does not account for the 
        source terms.
        """
        num_elements        = self.vt.mesh.nElements_global
        edge_values         = self.ebq[('velocity',ci)]
        norm_values         = self.ebq['n']
        int_weights         = self.ebq[('dS_u',0)]
        self.element_flux   = numpy.zeros(shape = (num_elements,1))
        # vectorised over elements but summed in the order of the old per-element loop (face, point, coordinate), so
        # each element's flux is bit-identical; the loop took 99% of a pwl call (0.30 s per call on 8,192 elements)
        for edge in range(edge_values.shape[1]):
            for k in range(edge_values.shape[2]):
                for t in range(edge_values.shape[3]):
                    self.element_flux[:,0] += (edge_values[:num_elements,edge,k,t] *
                                               norm_values[:num_elements,edge,k,t] *
                                               int_weights[:num_elements,edge,k])


class VPP_P1nc_RT0(VelocityPostProcessingAlgorithmBase):
    """
    P1-nonconforming velocity postprocessing

    Flagged for likely removal (campaign C10a, decision deferred 2026-11-02): it applies only to a Crouzeix-Raviart
    (P1 nonconforming) potential, no deck or test uses it, and two branches name misspelled objects
    (``cpostprocesssing``, ``f_ci_wieght``) that would raise NameError if reached.
    """
    from .cpostprocessing import postProcessRT0potentialFromP1nc,postProcessRT0potentialFromP1nc_sd
    from .cpostprocessing import postProcessRT0velocityFromP1nc,postProcessRT0velocityFromP1nc_sd
    from .cpostprocessing import updateRT0velocityWithAveragedPotentialP1nc,updateRT0velocityWithAveragedPotentialP1nc_sd

    from .cpostprocessing import getElementRT0velocityValues
    from .cpostprocessing import getElementBoundaryRT0velocityValues
    from .cpostprocessing import getGlobalElementBoundaryRT0velocityValues

    def __init__(self,vectorTransport=None,vtComponents=[0]):
        VelocityPostProcessingAlgorithmBase.__init__(self,postProcessingType='p1-nc',
                                                     vectorTransport=vectorTransport,
                                                     vtComponents=vtComponents)

        #how is the local velocity represented
        #  1 -- RT0, local rep is \vec a + b \vec x
        self.localVelocityRepresentationFlag = 1
        #
        self.nDOFs_element = {}
        #local element residuals
        self.elementResidual = self.vt.elementResidual
        self.elementBarycenters = None
        self.potentials = {} #map ci to indexes for potentials
        #begin building more features of velocity space
        for ci in self.vtComponents:
            try:
                self.elementBarycenters = self.vt.mesh.elementBarycentersArray
            except:
                warn("Element Barcyenters should be in mesh, this code will not work with moving mesh")
                self.elementBarycenters = numpy.zeros((self.vt.mesh.nElements_global,3),
                                                    'd')
                for eN in range(self.vt.mesh.nElements_global):
                    for nN in range(self.vt.mesh.nNodes_element):
                        nN_global = self.vt.mesh.elementNodesArray[eN,nN]
                        self.elementBarycenters[eN,:] += self.vt.mesh.nodeArray[nN_global,:]
                    self.elementBarycenters[eN,:] /= float(self.vt.mesh.nNodes_element)
            #
            assert isinstance(self.vt.u[ci].femSpace,FemTools.NC_AffineLinearOnSimplexWithNodalBasis)
            self.nDOFs_element[ci] = self.vt.nSpace_global+1
            self.q[('velocity_dofs',ci)] = numpy.zeros((self.vt.mesh.nElements_global,self.nDOFs_element[ci]),'d')
            if ('velocity_l2g',self.vtComponents[0]) in self.q:
                self.q[('velocity_l2g',ci)]  = self.q[('velocity_l2g',self.vtComponents[0])]
            else:
                self.q[('velocity_l2g',ci)]  = numpy.arange((self.vt.mesh.nElements_global*self.vt.mesh.nElementBoundaries_element),dtype='i').reshape((self.vt.mesh.nElements_global,self.vt.mesh.nElementBoundaries_element))
            #determine what terms are in equation
            self.potentials[ci] = []
            for cj in list(self.vt.coefficients.diffusion[ci].keys()):
                self.potentials[ci].append(cj)
                assert cj in self.vt.coefficients.potential, "ci=%s cj=%s diffusion but no potential" % (ci,cj)
            assert len(self.potentials[ci]) > 0, "ci=%s no diffusion coefficient found" % ci
            for cj in self.potentials[ci]:
                assert ('a',ci,cj) in self.vt.q, "ci=%s cj=%s diffusion but no key for a" % (ci,cj)
            #in case some terms missing from expected equations
            if ('f',ci) not in self.vt.q:
                if not hasattr(self,'dummy_p1nc'):
                    self.dummy_p1nc = {}
                self.dummy_p1nc[('f',ci)] = numpy.zeros((self.vt.q[('u',ci)].shape[0],
                                                         self.vt.q[('u',ci)].shape[1],
                                                         self.vt.nSpace_global),'d')
                self.dummy_p1nc[('f-weights',ci)]=numpy.array(self.vt.elementQuadratureWeights[('a',ci,self.potentials[ci][0])])

    #init

    def postprocess_component(self,ci,verbose=0):
        """compute velocity field in RT_0 assuming a potential field in
        :math:`P^1_{nc}` has already been solved for

        Uses Chou and Tang approach for now, so averaged pressure is not correct

        """
#        TODO:
#           cleanup use of mt,r
        #mwf hack
        if ('f',ci) not in self.vt.q:
            f_ci = self.dummy_p1nc[('f',ci)]
            f_ci_weight = self.dummy_p1nc[('f-weights',ci)]
        else:
            f_ci = self.vt.q[('f',ci)]
            f_ci_weight = self.vt.elementQuadratureWeights[('f',ci)]
        if ('mt',ci) in self.vt.q:
            assert ('w*dV_m',ci) in self.vt.q, "missing ('w*dV_m',ci) when have mt,ci "
            if self.vt.sd:
                cpostprocessing.postProcessRT0velocityFromP1nc_sd(self.vt.coefficients.sdInfo[(ci,self.potentials[ci][0])][0],
                                                                  self.vt.coefficients.sdInfo[(ci,self.potentials[ci][0])][1],
                                                                  self.vt.l2g[ci]['nFreeDOF'],
                                                                  self.vt.l2g[ci]['freeLocal'],
                                                                  self.vt.q['det(J)'],
                                                                  self.vt.ebq['sqrt(det(g))'],
                                                                  self.vt.ebq['n'],
                                                                  self.elementBarycenters,
                                                                  self.vt.elementQuadratureWeights[('a',ci,self.potentials[ci][0])],
                                                                  f_ci_weight,
                                                                  self.vt.q[('w*dV_r',ci)],
                                                                  self.vt.q[('phi',self.potentials[ci][0])],
                                                                  self.vt.q[('grad(phi)',self.potentials[ci][0])],
                                                                  self.vt.q[('a',ci,self.potentials[ci][0])],
                                                                  f_ci,
                                                                  self.vt.q[('r',ci)],
                                                                  self.q[('velocity_dofs',ci)],
                                                                  self.vt.q[('w*dV_m',ci)],
                                                                  self.vt.q[('mt',ci)])
            else:
                cpostprocessing.postProcessRT0velocityFromP1nc(self.vt.l2g[ci]['nFreeDOF'],
                                                               self.vt.l2g[ci]['freeLocal'],
                                                               self.vt.q['det(J)'],
                                                               self.vt.ebq['sqrt(det(g))'],
                                                               self.vt.ebq['n'],
                                                               self.elementBarycenters,
                                                               self.vt.elementQuadratureWeights[('a',ci,self.potentials[ci][0])],
                                                               f_ci_weight,
                                                               self.vt.q[('w*dV_r',ci)],
                                                               self.vt.q[('phi',self.potentials[ci][0])],
                                                               self.vt.q[('grad(phi)',self.potentials[ci][0])],
                                                               self.vt.q[('a',ci,self.potentials[ci][0])],
                                                               f_ci,
                                                               self.vt.q[('r',ci)],
                                                               self.q[('velocity_dofs',ci)],
                                                               self.vt.q[('w*dV_m',ci)],
                                                               self.vt.q[('mt',ci)])
        else:
            if self.vt.sd:
                cpostprocessing.postProcessRT0velocityFromP1nc_sd(self.vt.coefficients.sdInfo[(ci,self.potentials[ci][0])][0],
                                                                  self.vt.coefficients.sdInfo[(ci,self.potentials[ci][0])][1],
                                                                  self.vt.l2g[ci]['nFreeDOF'],
                                                                  self.vt.l2g[ci]['freeLocal'],
                                                                  self.vt.q['det(J)'],
                                                                  self.vt.ebq['sqrt(det(g))'],
                                                                  self.vt.ebq['n'],
                                                                  self.elementBarycenters,
                                                                  self.vt.elementQuadratureWeights[('a',ci,self.potentials[ci][0])],
                                                                  f_ci_weight,#self.vt.elementQuadratureWeights[('f',ci)],
                                                                  self.vt.q[('w*dV_r',ci)],
                                                                  self.vt.q[('phi',self.potentials[ci][0])],
                                                                  self.vt.q[('grad(phi)',self.potentials[ci][0])],
                                                                  self.vt.q[('a',ci,self.potentials[ci][0])],
                                                                  f_ci,#self.vt.q[('f',ci)],
                                                                  self.vt.q[('r',ci)],
                                                                  self.q[('velocity_dofs',ci)])
            else:
                cpostprocessing.postProcessRT0velocityFromP1nc(self.vt.l2g[ci]['nFreeDOF'],
                                                               self.vt.l2g[ci]['freeLocal'],
                                                               self.vt.q['det(J)'],
                                                               self.vt.ebq['sqrt(det(g))'],
                                                               self.vt.ebq['n'],
                                                               self.elementBarycenters,
                                                               self.vt.elementQuadratureWeights[('a',ci,self.potentials[ci][0])],
                                                               f_ci_wieght,#self.vt.elementQuadratureWeights[('f',ci)],
                                                               self.vt.q[('w*dV_r',ci)],
                                                               self.vt.q[('phi',self.potentials[ci][0])],
                                                               self.vt.q[('grad(phi)',self.potentials[ci][0])],
                                                               self.vt.q[('a',ci,self.potentials[ci][0])],
                                                               f_ci,#self.vt.q[('f',ci)],
                                                               self.vt.q[('r',ci)],
                                                               self.q[('velocity_dofs',ci)])

        for cj in self.potentials[ci][1:-1]:
            if self.vt.sd:
                cpostprocesssing.updateRT0velocityWithAveragedPotentialP1nc_sd(self.vt.coefficients.sdInfo[(ci,cj)][0],self.vt.coefficients.sdInfo[(ci,cj)][1],
                                                                               self.vt.q['det(J)'],
                                                                               self.vt.elementQuadratureWeights[('a',ci,self.potentials[ci][0])],
                                                                               self.vt.q[('phi',ci)],
                                                                               self.vt.q[('grad(phi)',cj)],
                                                                               self.vt.q[('a',ci,cj)],
                                                                               self.q[('velocity_dofs',ci)])
            else:
                cpostprocessing.updateRT0velocityWithAveragedPotentialP1nc(self.vt.q['det(J)'],
                                                                           self.vt.elementQuadratureWeights[('a',ci,self.potentials[ci][0])],
                                                                           self.vt.q[('phi',ci)],
                                                                           self.vt.q[('grad(phi)',cj)],
                                                                           self.vt.q[('a',ci,cj)],
                                                                           self.q[('velocity_dofs',ci)])

        cpostprocessing.getElementRT0velocityValues(self.vt.q['x'],
                                                    self.q[('velocity_dofs',ci)],
                                                    self.q[('velocity',ci)])
        cpostprocessing.getElementBoundaryRT0velocityValues(self.vt.ebq['x'],
                                                            self.q[('velocity_dofs',ci)],
                                                            self.ebq[('velocity',ci)])
        cpostprocessing.getGlobalElementBoundaryRT0velocityValues(self.vt.mesh.elementBoundaryElementsArray,
                                                                  self.vt.ebq_global['x'],
                                                                  self.q[('velocity_dofs',ci)],
                                                                  self.ebq_global[('velocity',ci)])

        cfemIntegrals.copyExteriorElementBoundaryValuesFromGlobalElementBoundaryValues(self.vt.mesh.exteriorElementBoundariesArray,
                                                                                       self.vt.mesh.elementBoundaryElementsArray,
                                                                                       self.vt.mesh.elementBoundaryLocalElementBoundariesArray,
                                                                                       self.vt.ebq_global[('velocity',ci)],
                                                                                       self.vt.ebqe[('velocity',ci)])

        #end ci
    #per component step
    def evaluateElementVelocityField(self,x,ci):
        """
        evaluate velocity field assuming velocity_dofs already calculated
        for now assumes x shaped like nE x nq x 3
        """
        assert len(x.shape) == 3, "wrong shape for x= %s " % x.shape
        nE = x.shape[0]; nq = x.shape[1]; nd = self.vt.nSpace_global
        vx = numpy.zeros((nE,nq,nd),'d')
        cpostprocessing.getElementRT0velocityValues(x,
                                         self.q[('velocity_dofs',ci)],
                                         vx)
        return vx


class VPP_PWL_RT0(VelocityPostProcessingAlgorithmBase):
    r"""Local Larson-Niklasson method with RT_0 representation for element
    velocities This is not the correct postprocessing for optimized NS
    codes

    uses RT_0 representation in terms of local basis

    .. math:

      \vec N_i = \frac{1}{d|E|}(\vec x - p_i)

    where :math:`p_i` is the vertex across from face i, :math:`|E|` is the volume of
    the element, and d is the space dimension the degrees of freedom
    are :math:`V^i = \int_{e_i}\vec v\dot n_{i}\ds`

    """
#    TODO:
#      remove python loops in building mesh infrastructure or get rid of dependence
#        on these forms of node star info and use data structures in cmesh
    def __init__(self,vectorTransport=None,vtComponents=[0],omitFluxBoundaryNodes=True):
        VelocityPostProcessingAlgorithmBase.__init__(self,postProcessingType='pwl',
                                                     vectorTransport=vectorTransport,
                                                     vtComponents=vtComponents)
        #how is the local velocity represented
        #  2 -- RT0, local rep is \sum^d_{i=0}V^i\vec N_{T,i},
        #           \vec N_{T,i} = \frac{1}{d|E|}(\vec x - p_i), i=0,...,d
        if self.vt.mesh.meshType() != 'simplex':
            raise Exception('Proteus currently only supports conservative '\
                'flux post-processing on triangular and tetrahedral meshes.  ' \
                'Try removing the post-processing flag or changing your ' \
                'mesh/finite element type.')

        self.localVelocityRepresentationFlag = 2

        self.omitFluxBoundaryNodes=omitFluxBoundaryNodes

        #to handle some spaces being P^k k > 1
        self.alpha = {}
        self.solutionTestSpaceIsNotPWL = {}
        self.elementResidual = {}
        self.testSpace = None
        #mesh information needed
        #building mesh infrastructure
        self.globalNode2globalElementList = [[] for nN in range(self.vt.mesh.nNodes_global)]
        for eN in range(self.vt.mesh.nElements_global):
            for nN in range(self.vt.mesh.nNodes_element):
                nN_global = self.vt.mesh.elementNodesArray[eN,nN]
                self.globalNode2globalElementList[nN_global].append(eN)
        self.globalNodeGlobalElement2StarElement = []
        self.nElements_node = numpy.zeros((self.vt.mesh.nNodes_global,),'i')
        self.nodeStarElementsArray = numpy.ones((self.vt.mesh.nElements_global,
                                                 self.vt.mesh.nNodes_element),'i')
        self.nodeStarElementsArray[:]=-1
        self.nodeStarElementNeighborsArray = numpy.zeros((self.vt.mesh.nElements_global,
                                                            self.vt.mesh.nNodes_element,
                                                            self.vt.mesh.nElementBoundaries_element),
                                                           'i')
        for I in range(self.vt.mesh.nNodes_global):
            self.globalNode2globalElementList[I].sort()
            self.nElements_node[I] = len(self.globalNode2globalElementList[I])
            self.globalNodeGlobalElement2StarElement.append(dict([(eN_global,eN_node) for eN_node,eN_global in enumerate(self.globalNode2globalElementList[I])]))
        for ebNI in range(self.vt.mesh.nInteriorElementBoundaries_global):
            ebN = self.vt.mesh.interiorElementBoundariesArray[ebNI]
            left_eN_global   = self.vt.mesh.elementBoundaryElementsArray[ebN,0]
            right_eN_global  = self.vt.mesh.elementBoundaryElementsArray[ebN,1]
            left_ebN_element  = self.vt.mesh.elementBoundaryLocalElementBoundariesArray[ebN,0]
            right_ebN_element = self.vt.mesh.elementBoundaryLocalElementBoundariesArray[ebN,1]
            for i in range(self.vt.mesh.nNodes_element):
                left_I = self.vt.mesh.elementNodesArray[left_eN_global,i]
                self.nodeStarElementsArray[left_eN_global,i] = self.globalNodeGlobalElement2StarElement[left_I][left_eN_global]
                if i != left_ebN_element:
                    self.nodeStarElementNeighborsArray[left_eN_global,i,left_ebN_element] = self.globalNodeGlobalElement2StarElement[left_I][right_eN_global]
                #if
                right_I=self.vt.mesh.elementNodesArray[right_eN_global,i]
                self.nodeStarElementsArray[right_eN_global,i] = self.globalNodeGlobalElement2StarElement[right_I][right_eN_global]
                if i != right_ebN_element:
                    self.nodeStarElementNeighborsArray[right_eN_global,i,right_ebN_element] = self.globalNodeGlobalElement2StarElement[right_I][left_eN_global]

            #i
        #ebNi
        for ebNE in range(self.vt.mesh.nExteriorElementBoundaries_global):
            ebN = self.vt.mesh.exteriorElementBoundariesArray[ebNE]
            eN_global   = self.vt.mesh.elementBoundaryElementsArray[ebN,0]
            ebN_element  = self.vt.mesh.elementBoundaryLocalElementBoundariesArray[ebN,0]
            for i in range(self.vt.mesh.nNodes_element):
                I = self.vt.mesh.elementNodesArray[eN_global,i]
                self.nodeStarElementsArray[eN_global,i] = self.globalNodeGlobalElement2StarElement[I][eN_global]
            #i
        #ebNE
        self.nodeStarFactors = {}
        for ci in self.vtComponents:
            self.nodeStarFactors[ci] = cpostprocessing.NodeStarFactor(self.nElements_node,
                                                                      self.nodeStarElementsArray,
                                                                      self.nodeStarElementNeighborsArray)

        self.nDOFs_element = {}
        self.updateConservationJacobian = {}
        for ci in self.vtComponents:
            ##Start with original approach to "fix" local node star problems around boundaries
            #when one or more faces may be omitted because of direct enforcement of Neumann boundary
            #conditions
            #start with no nodes to be "fixed" for Neumann boundaries
            self.fluxBoundaryNodes[ci] = numpy.zeros((self.vt.mesh.nNodes_global,),'i')
            if self.omitFluxBoundaryNodes == True:
                #find
                # 1) boundary nodes,
                # 2) boundary nodes that have an adjoining boundary face w/o
                #      flux boundary conditions specified
                # 3) boundary nodes that are "free" ie not strong Dirichlet nodes
                boundaryNodes = set()
                nodesWithDirichletBoundaryFace = set()
                freeBoundaryNodes = set()

                for ebNE in range(self.vt.mesh.nExteriorElementBoundaries_global):
                    ebN = self.vt.mesh.exteriorElementBoundariesArray[ebNE]
                    for nN_ebN in range(self.vt.mesh.nNodes_elementBoundary):
                        nN = self.vt.mesh.elementBoundaryNodesArray[ebN,nN_ebN]
                        boundaryNodes.add(nN)
                        if not self.fluxElementBoundaries[ci][ebNE]:
                            nodesWithDirichletBoundaryFace.add(nN) #Dirichlet Node connected to at least one Dirichlet face
                        #
                    #local elementBoundary nodes
                    eN  = self.vt.mesh.elementBoundaryElementsArray[ebN,0]
                    for i in range(self.vt.l2g[ci]['nFreeDOF'][eN]):
                        j = self.vt.l2g[ci]['freeLocal'][eN,i]
                        if j < self.vt.mesh.nNodes_element:
                            J = self.vt.mesh.elementNodesArray[eN,j]
                            freeBoundaryNodes.add(J)
                    #i
                #ebNE
                boundaryNodesNoDirichletFace = set.difference(boundaryNodes,nodesWithDirichletBoundaryFace)
                dirichletNodesNoDirichletFace= set.difference(boundaryNodesNoDirichletFace,freeBoundaryNodes)
                fluxNodesNoDirichletFace     = set.intersection(boundaryNodesNoDirichletFace,freeBoundaryNodes)
                fluxElementBoundariesRemoved = set()
                for ebNE in range(self.vt.mesh.nExteriorElementBoundaries_global):
                    ebN = self.vt.mesh.exteriorElementBoundariesArray[ebNE]
                    if self.fluxElementBoundaries[ci][ebNE] == 1:
                        for nN in self.vt.mesh.elementBoundaryNodesArray[ebN]:
                            if nN in dirichletNodesNoDirichletFace:
                                self.fluxElementBoundaries[ci][ebNE] = 0
                                fluxElementBoundariesRemoved.add(ebNE)

                #update "free" nodes without a "Dirichlet" boundary collection
                for ebNE in fluxElementBoundariesRemoved:
                    ebN = self.vt.mesh.exteriorElementBoundariesArray[ebNE]
                    for nN_ebN in range(self.vt.mesh.nNodes_elementBoundary):#nodes on elementBoundary
                        nN = self.vt.mesh.elementBoundaryNodesArray[ebN,nN_ebN]
                        fluxNodesNoDirichletFace.discard(nN)
                #
                ##\todo need to make this only be size of number exterior boundary nodes
                #now done above self.fluxBoundaryNodes[ci] = numpy.zeros((self.vt.mesh.nNodes_global,),'i')
                for nN in fluxNodesNoDirichletFace:
                    #these  nodes are "overconstrained" for correction equation like interior nodes
                    self.fluxBoundaryNodes[ci][nN] = 1
                #
            #end omitting flux boundary nodes
            if not isinstance(self.vt.u[ci].femSpace,FemTools.C0_AffineLinearOnSimplexWithNodalBasis):
                self.solutionTestSpaceIsNotPWL[ci] = True
                logEvent("pwl post-processing for finite element space"+str(self.vt.u[ci].femSpace))
                #need to set up w and w*dS_f and weights to build residuals for linears out of R_{e,i}
                #Point1
                if self.testSpace is None:
                    self.testSpace = FemTools.C0_AffineLinearOnSimplexWithNodalBasis(self.vt.mesh,self.vt.nSpace_global)
                assert isinstance(self.testSpace,FemTools.C0_AffineLinearOnSimplexWithNodalBasis)
                self.alpha[ci] = numpy.zeros((self.testSpace.referenceFiniteElement.localFunctionSpace.dim,
                                          self.vt.u[ci].femSpace.referenceFiniteElement.localFunctionSpace.dim),'d')
                for j,w in enumerate(self.testSpace.referenceFiniteElement.localFunctionSpace.basis):
                    for i,f in enumerate(self.vt.u[ci].femSpace.referenceFiniteElement.interpolationConditions.functionals):
                        self.alpha[ci][j,i] = f(w)
                self.elementResidual[ci] = numpy.zeros((self.vt.mesh.nElements_global,
                                                        self.testSpace.referenceFiniteElement.localFunctionSpace.dim),
                                                       'd')
                self.qv[ci] = numpy.zeros(
                    (self.vt.mesh.nElements_global,
                     self.vt.nQuadraturePoints_element,
                     self.testSpace.referenceFiniteElement.localFunctionSpace.dim),
                    'd')
                self.w[ci] = numpy.zeros(
                    (self.vt.mesh.nElements_global,
                     self.vt.mesh.nElementBoundaries_element,
                     self.vt.nElementBoundaryQuadraturePoints_elementBoundary,
                     self.testSpace.referenceFiniteElement.localFunctionSpace.dim),
                    'd')
                self.w_dS[ci] = numpy.zeros(
                    (self.vt.mesh.nElements_global,
                     self.vt.mesh.nElementBoundaries_element,
                     self.vt.nElementBoundaryQuadraturePoints_elementBoundary,
                     self.testSpace.referenceFiniteElement.localFunctionSpace.dim),
                    'd')
                self.testSpace.getBasisValues(self.vt.elementQuadraturePoints,
                                              self.qv[ci])
                self.testSpace.getBasisValuesTrace(self.vt.u[0].femSpace.elementMaps.permutations,
                                                   self.vt.ebq['hat(x)'],
                                                   self.w[ci])
                cfemIntegrals.calculateWeightedShapeTrace(self.vt.elementBoundaryQuadratureWeights[('u',ci)],
                                                          self.vt.ebq['sqrt(det(g))'],
                                                          self.w[ci],
                                                          self.w_dS[ci])
            else:
                #local element residuals
                self.elementResidual[ci] = self.vt.elementResidual[ci]
            #solution space not P^1
            self.updateConservationJacobian[ci] = True
            #RT0 specific information
            self.nDOFs_element[ci] = self.vt.nSpace_global+1
            self.q[('velocity_dofs',ci)] = numpy.zeros((self.vt.mesh.nElements_global,
                                                        self.nDOFs_element[ci]),'d')
            if ('velocity_l2g',self.vtComponents[0]) in self.q:
                self.q[('velocity_l2g',ci)]  = self.q[('velocity_l2g',self.vtComponents[0])]
            else:
                self.q[('velocity_l2g',ci)]  = numpy.arange((self.vt.mesh.nElements_global*self.vt.mesh.nElementBoundaries_element),dtype='i').reshape((self.vt.mesh.nElements_global,self.vt.mesh.nElementBoundaries_element))
        #ci
        if self.testSpace is None:
            #all the solution spaces must be C0P1
            self.testSpace = self.vt.u[self.vtComponents[0]].femSpace
    #init
    def postprocess_component(self,ci,verbose=0):
        """compute mass conservative velocity field following Larson and
        Niklasson assuming a :math:`P^k C_0` Galerkin solution has already been
        found

        """
        #must zero first time for average velocity
        self.nodeStarFactors[ci].setU(0.0)
        #correct first time through, in case there are Flux boundaries that
        #are not enforced directly in postprocessed flux
        #self.fluxElementBoundaries[ci] determines which boundaries have fluxes
        #enforced directly
        if self.solutionTestSpaceIsNotPWL:
            useC=True
            if useC:
                cpostprocessing.calculateElementResidualPWL(self.alpha[ci],self.vt.elementResidual[ci],self.elementResidual[ci])
            else:
                self.elementResidual[ci].fill(0.0)
                for eN in range(self.vt.mesh.nElements_global):
                    for i in range(self.testSpace.referenceFiniteElement.localFunctionSpace.dim):
                        for j in range(self.vt.u[ci].femSpace.referenceFiniteElement.localFunctionSpace.dim):
                            self.elementResidual[ci][eN,i] += self.alpha[ci][i,j]*self.vt.elementResidual[ci][eN,j]
        self.getConservationResidualPWL(ci,correctFlux=True)
        if verbose > 0:
            logEvent("""velpp Max local conservation (average velocity) = %12.5e""" % max(numpy.absolute(self.q[('conservationResidual',ci)].flat[0:self.vt.mesh.subdomainMesh.nElements_owned])))

        if self.updateConservationJacobian[ci]:
            self.getConservationJacobianPWL(ci)
            self.updateConservationJacobian[ci] = False #only depends on mesh need to resignal if mesh adapts
        cpostprocessing.calculateConservationFluxPWL(self.nElements_node,
                                                     self.vt.internalNodesArray,
                                                     self.fluxBoundaryNodes[ci],
                                                     self.nodeStarFactors[ci])
        self.sendStarSolutionsToGhosts(ci)

        self.getConservationResidualPWL(ci,correctFlux=False)

        #add back fluxes for elementBoundaries that were Neumann but
        #not enforced directly
        self.addBoundaryFluxesBackToResidual(ci,self.fluxElementBoundaries[ci])

        if verbose > 0:
            logEvent("Max local conservation (dgp1 enriched) = %12.5e" % max(numpy.absolute(self.q[('conservationResidual',ci)].flat[0:self.vt.mesh.nElements_owned])))


    def takeCutFaceVelocitiesFromInteriorRanks(self,ci):
        """Replace the velocity on faces the partition cut by its value on a rank where the face is interior.

        An owned element can have a face whose other element is not on the rank (with node partitioning the rank
        that owns the elements around a cut need not hold their neighbours). There the face is exterior: its average
        velocity is one-sided and the correction lacks the neighbour's side, so the flux is wrong. A cut face has a
        ghost node, and the owner of that node holds every element around it, so the face is interior there. Each rank
        asks those owners for the face velocity at the face quadrature points (matched by position) and corrects the
        conservation residual of its element by the change in outward flux. Cut faces are the exterior faces the
        partitioner flagged -1. Call after sendStarSolutionsToGhosts; serial runs return at once.
        """
        from . import Comm
        comm = Comm.get()
        if comm.size() == 1:
            return
        mpi = comm.comm.tompi4py()
        mesh = self.vt.mesh
        globalMesh = mesh.globalMesh
        rank, size = mpi.rank, mpi.size
        nodeOffsets = numpy.asarray(globalMesh.nodeOffsets_subdomain_owned)
        node_l2g = numpy.asarray(globalMesh.nodeNumbering_subdomain2global)
        face_l2g = numpy.asarray(globalMesh.elementBoundaryNumbering_subdomain2global)
        nOwned = mesh.nNodes_owned
        velocity = self.ebq_global[('velocity',ci)]
        x = self.vt.ebq_global['x']
        requests = [[] for r in range(size)]
        asked_for = {}
        for ebN in mesh.exteriorElementBoundariesArray:
            if mesh.elementBoundaryMaterialTypes[ebN] != -1:
                continue  # the partitioner flags the faces it cut -1; every other exterior face is the domain's
            ghosts = [nN for nN in mesh.elementBoundaryNodesArray[ebN] if nN >= nOwned]
            assert ghosts, "a cut face with only owned nodes: the owned nodes' stars would be incomplete"
            owner = int(numpy.searchsorted(nodeOffsets, node_l2g[ghosts[0]], side='right') - 1)
            requests[owner].append(int(face_l2g[ebN]))
            asked_for[int(face_l2g[ebN])] = ebN
        asked = mpi.alltoall(requests)
        face_g2l = dict((int(g), ebN) for ebN, g in enumerate(face_l2g[:mesh.nElementBoundaries_global]))
        interior = numpy.zeros((mesh.nElementBoundaries_global,), 'bool')
        interior[mesh.interiorElementBoundariesArray] = True
        replies = []
        for r in range(size):
            reply = []
            for ebN_global in asked[r]:
                ebN = face_g2l.get(ebN_global)
                if ebN is not None and interior[ebN]:
                    reply.append((ebN_global, x[ebN].copy(), velocity[ebN].copy()))
            replies.append(reply)
        received = mpi.alltoall(replies)
        dS = self.vt.ebq[('dS_u',ci)]
        normal = self.vt.ebq_global['n']
        for reply in received:
            for ebN_global, x_other, v_other in reply:
                ebN = asked_for[ebN_global]
                eN = mesh.elementBoundaryElementsArray[ebN,0]
                ebN_element = mesh.elementBoundaryLocalElementBoundariesArray[ebN,0]
                old = velocity[ebN].copy()
                for k in range(velocity.shape[1]):
                    j = numpy.argmin(numpy.linalg.norm(x_other - x[ebN,k], axis=1))
                    assert numpy.linalg.norm(x_other[j] - x[ebN,k]) <= 1.0e-10*max(1.0, numpy.abs(x[ebN,k]).max())
                    velocity[ebN,k] = v_other[j]
                dflux = ((velocity[ebN] - old)*normal[ebN]).sum(axis=-1)
                self.q[('conservationResidual',ci)][eN] += (dflux*dS[eN,ebN_element]).sum()

    def sendStarSolutionsToGhosts(self,ci):
        """Replace the star solutions of ghost nodes by their owners'.

        A face flux is corrected by the star problems of the face's nodes. On a rank, the star of a node it owns is
        complete, but the star of a ghost node is cut by the partition unless the overlap is two layers deep, so its
        local solution is wrong and so are the fluxes of the owned elements around it. Each rank asks the owners of
        its ghost nodes for their star solutions, keyed by global element number, and takes the values for the star
        elements it has. Serial runs return at once.
        """
        from . import Comm
        comm = Comm.get()
        if comm.size() == 1:
            return
        mpi = comm.comm.tompi4py()
        mesh = self.vt.mesh
        globalMesh = mesh.globalMesh
        rank, size = mpi.rank, mpi.size
        nodeOffsets = numpy.asarray(globalMesh.nodeOffsets_subdomain_owned)
        node_l2g = numpy.asarray(globalMesh.nodeNumbering_subdomain2global)
        element_l2g = numpy.asarray(globalMesh.elementNumbering_subdomain2global)
        nOwned = mesh.nNodes_owned
        # owned nodes come first and are numbered contiguously from the rank's offset
        assert (node_l2g[:nOwned] == numpy.arange(nodeOffsets[rank], nodeOffsets[rank + 1])).all()
        ghosts = node_l2g[nOwned:mesh.nNodes_global]
        owners = numpy.searchsorted(nodeOffsets, ghosts, side='right') - 1
        requests = [ghosts[owners == r].tolist() for r in range(size)]
        asked = mpi.alltoall(requests)
        nsf = self.nodeStarFactors[ci]
        replies = []
        for r in range(size):
            reply = []
            for nN_global in asked[r]:
                I = int(nN_global - nodeOffsets[rank])
                reply.append((nN_global, element_l2g[self.globalNode2globalElementList[I]], nsf.getSubdomainU(I)))
            replies.append(reply)
        received = mpi.alltoall(replies)
        ghost_local = dict((int(g), nOwned + i) for i, g in enumerate(ghosts))
        for reply in received:
            for nN_global, elements, U_owner in reply:
                I = ghost_local[int(nN_global)]
                value = dict(zip(elements.tolist(), U_owner.tolist()))
                star = self.globalNode2globalElementList[I]
                U = nsf.getSubdomainU(I)
                assert len(U) == len(star)
                for k, eN in enumerate(star):
                    U[k] = value[int(element_l2g[eN])]  # the owner's star is complete, so it has every element
                nsf.setSubdomainU(I, U)

    def getConservationResidualPWL(self,ci,correctFlux=False):
        """
        compute conservation resiudal using current guess for element boundary flux
        """
        if correctFlux == True:
            self.removeBoundaryFluxesFromResidual(ci,self.fluxElementBoundaries[ci])
        cpostprocessing.calculateConservationResidualPWL(self.vt.mesh.interiorElementBoundariesArray,
                                                         self.vt.mesh.exteriorElementBoundariesArray,
                                                         self.vt.mesh.elementBoundaryElementsArray,
                                                         self.vt.mesh.elementBoundaryLocalElementBoundariesArray,
                                                         self.vt.mesh.elementNodesArray,
                                                         self.vt.mesh.elementNodesArray, # node numbers index the stars; a P2 vertex dof is the node only in serial
                                                         self.nodeStarElementsArray,
                                                         self.nodeStarElementNeighborsArray,
                                                         self.nElements_node,
                                                         self.fluxElementBoundaries[ci],
                                                         self.elementResidual[ci],
                                                         self.vt.ebq_global[('velocityAverage',ci)],
                                                         self.vt.ebq[('dS_u',ci)],
                                                         self.w[ci],
                                                         self.vt.ebq_global['n'],
                                                         self.nodeStarFactors[ci],
                                                         self.q[('conservationResidual',ci)], # output
                                                         self.ebq_global[('velocity',ci)],    # output
                                                         self.ebq[('velocity',ci)])           # output
        if not correctFlux:
            self.takeCutFaceVelocitiesFromInteriorRanks(ci)
        #set boundary flux
        updateCoef = 0.0 #overwrite first
        cfemIntegrals.loadBoundaryFluxIntoGlobalElementBoundaryVelocity(self.vt.mesh.exteriorElementBoundariesArray,
                                                                        self.fluxElementBoundaries[ci],
                                                                        self.vt.ebq_global['n'],
                                                                        self.vt.ebq_global[('totalFlux',ci)],
                                                                        updateCoef,
                                                                        self.vt.ebq_global[('velocity',ci)])

        cfemIntegrals.copyGlobalElementBoundaryVelocityToElementBoundary(self.vt.mesh.interiorElementBoundariesArray,
                                                                         self.vt.mesh.exteriorElementBoundariesArray,
                                                                         self.vt.mesh.elementBoundaryElementsArray,
                                                                         self.vt.mesh.elementBoundaryLocalElementBoundariesArray,
                                                                         self.vt.ebq_global[('velocity',ci)],
                                                                         self.vt.ebq[('velocity',ci)])

        cfemIntegrals.copyExteriorElementBoundaryValuesFromGlobalElementBoundaryValues(self.vt.mesh.exteriorElementBoundariesArray,
                                                                                       self.vt.mesh.elementBoundaryElementsArray,
                                                                                       self.vt.mesh.elementBoundaryLocalElementBoundariesArray,
                                                                                       self.vt.ebq_global[('velocity',ci)],
                                                                                       self.vt.ebqe[('velocity',ci)])

        #end set boundary flux
        #go from boundary flux to local element boundary representation
        self.evaluateLocalVelocityRepresentation(ci)
        self.getElementwiseFlux(ci)

    #
    def getConservationJacobianPWL(self,ci):
        """
        Build local systems for post-processing solve
        """
        cpostprocessing.calculateConservationJacobianPWL(self.vt.mesh.interiorElementBoundariesArray,
                                                         self.vt.mesh.exteriorElementBoundariesArray,
                                                         self.vt.mesh.elementBoundaryElementsArray,
                                                         self.vt.mesh.elementBoundaryLocalElementBoundariesArray,
                                                         self.vt.mesh.elementNodesArray,
                                                         self.vt.mesh.elementNodesArray, # node numbers index the stars; a P2 vertex dof is the node only in serial
                                                         self.nodeStarElementsArray,
                                                         self.nodeStarElementNeighborsArray,
                                                         self.nElements_node,
                                                         self.vt.internalNodesArray,
                                                         self.fluxElementBoundaries[ci],
                                                         self.fluxBoundaryNodes[ci],
                                                         self.w_dS[ci],
                                                         self.vt.ebq_global['n'],
                                                         self.nodeStarFactors[ci])

    def evaluateLocalVelocityRepresentation(self,ci):
        """
        project to :math:`RT_0` velocity from element boundary fluxes
        """
        cpostprocessing.projectElementBoundaryVelocityToRT0fluxRep(self.vt.ebq[('dS_u',ci)],
                                                                   self.vt.ebq['n'],
                                                                   self.ebq[('velocity',ci)],
                                                                   self.q[('velocity_dofs',ci)])
        cpostprocessing.getElementRT0velocityValuesFluxRep(self.vt.mesh.nodeArray,
                                                           self.vt.mesh.elementNodesArray,
                                                           self.vt.q['abs(det(J))'],
                                                           self.vt.q['x'],
                                                           self.q[('velocity_dofs',ci)],
                                                           self.q[('velocity',ci)])
    def evaluateElementVelocityField(self,x,ci):
        """
        evaluate velocity field assuming velocity_dofs already calculated
        for now assumes x shaped like nE x nq x 3
        """
        assert len(x.shape) == 3, "wrong shape for x= %s " % x.shape
        nE = x.shape[0]; nq = x.shape[1]; nd = self.vt.nSpace_global
        vx = numpy.zeros((nE,nq,nd),'d')
        cpostprocessing.getElementRT0velocityValuesFluxRep(self.vt.mesh.nodeArray,
                                                           self.vt.mesh.elementNodesArray,
                                                           self.vt.q['abs(det(J))'],
                                                           x,
                                                           self.q[('velocity_dofs',ci)],
                                                           vx)
        return vx
class VPP_PWL_BDM(VPP_PWL_RT0):
    r"""Local Larson-Niklasson method with BDM1 representation for element
    velocities This is not the correct postprocessing for optimized NS
    codes

    Only difference from VPP_PWL_RT0 should be steps for local velocity representation
    This one uses  BDM_1 space which is :math:`[P^1]^d` locally with continuous
    linear fluxes on each face
    use standard basis

    .. math::
      
      \vec N_i = \lambda_{i/d}\vec e_{i \bmod d}
    
    That is the dofs are locally (say in 2d) :math:`[v^x_0,v^y_0,v^x_1,v^y_1,v^x_2,v^y_2]`

    Have to use BDM projection to get degrees of freedom

    """
#    TODO:
#      need additional code to compute velocities at ebq and ebq_global if desired
#      looks like ebq_global, ebqe may not be getting a good approximation if v.n=0 on boundary?
#        for example try LN_Example1 with no heterogeneity
#
#      Check what the problem is when running with numerical flux. mass balances are right
#        but getting strange convergence and error values. May need to check what is getting
#        added to ebq[('velocity',ci)] from numerical flux
    def __init__(self,vectorTransport=None,vtComponents=[0]):
        VPP_PWL_RT0.__init__(self,vectorTransport=vectorTransport,vtComponents=vtComponents)
        #have to directly modify the type now to show bdm
        self.postProcessingType = 'pwl-bdm'
        #how is the local velocity represented
        #  0 -- BDM P^1 Lagrange rep
        self.localVelocityRepresentationFlag = 0

        #
        for ci in self.vtComponents:
            self.nDOFs_element[ci] = self.vt.nSpace_global*(self.vt.nSpace_global+1)
            self.q[('velocity_dofs',ci)] = numpy.zeros((self.vt.mesh.nElements_global,self.nDOFs_element[ci]),'d')
            if ci != self.vtComponents[0] and ('velocity_l2g',self.vtComponents[0]) in self.q:
                self.q[('velocity_l2g',ci)]  = self.q[('velocity_l2g',self.vtComponents[0])]
            else:
                self.q[('velocity_l2g',ci)]  = numpy.arange((self.vt.mesh.nElements_global*self.nDOFs_element[ci]),dtype='i').reshape((self.vt.mesh.nElements_global,self.nDOFs_element[ci]))
        #

        self.BDMcomponent=self.vtComponents[0]
        self.BDMprojectionMat_element = numpy.zeros((self.vt.mesh.nElements_global,
                                                     self.nDOFs_element[self.BDMcomponent],
                                                     self.nDOFs_element[self.BDMcomponent]),
                                                    'd')
        self.BDMprojectionMatPivots_element = numpy.zeros((self.vt.mesh.nElements_global,
                                                           self.nDOFs_element[self.BDMcomponent]),
                                                          'i')
        self.computeBDM1projectionMatrices()

    def computeBDM1projectionMatrices(self):
        cpostprocessing.buildLocalBDM1projectionMatrices(self.w_dS[self.BDMcomponent],#vt.ebq[('w*dS_u',self.BDMcomponent)],
                                                         self.vt.ebq['n'],
                                                         self.w[self.BDMcomponent],#self.vt.ebq[('v',self.BDMcomponent)],
                                                         self.BDMprojectionMat_element)
        cpostprocessing.factorLocalBDM1projectionMatrices(self.BDMprojectionMat_element,
                                                          self.BDMprojectionMatPivots_element)
    def computeGeometricInfo(self):
        if self.BDMcomponent is not None:
            self.computeBDM1projectionMatrices()

    def evaluateLocalVelocityRepresentation(self,ci):
        """
        project to BDM velocity from element boundary fluxes
        """
        assert self.nDOFs_element[ci] == self.vt.nSpace_global*(self.vt.nSpace_global+1), "wrong size for BDM"

        cpostprocessing.solveLocalBDM1projection(self.BDMprojectionMat_element,
                                 self.BDMprojectionMatPivots_element,
                                 self.w_dS[ci],
                                 self.vt.ebq['n'],
                                 self.ebq[('velocity',ci)],
                                 self.q[('velocity_dofs',ci)])

        cpostprocessing.getElementBDM1velocityValuesLagrangeRep(self.qv[ci],
                                                                self.q[('velocity_dofs',ci)],
                                                                self.vt.q[('velocity',ci)])

    def evaluateElementVelocityField(self,x,ci):
        """
        evaluate velocity field assuming velocity_dofs already calculated
        for now assumes x shaped like nE x nq x 3
        TODO:
           put python loops in c
        """
        assert len(x.shape) == 3, "wrong shape for x= %s " % x.shape
        nE = x.shape[0]; nq = x.shape[1]; nd = self.vt.nSpace_global
        vx = numpy.zeros((nE,nq,nd),'d')
        #have to evaluate shape functions at new points. This is painful
        #uci     = self.vt.u[ci]
        xiArray = numpy.zeros(x.shape,'d')
        vArray  = numpy.zeros((nE,nq,self.testSpace.max_nDOF_element),'d')
        invJ    = numpy.zeros((nE,nq,self.vt.q['inverse(J)'].shape[2],self.vt.q['inverse(J)'].shape[3]),
                                'd')
        for ie in range(nE):
            for iq in range(nq):
                invJ[ie,iq,:,:] = self.vt.q['inverse(J)'][ie,0,:,:] #assume affine
            #iq
        #iq
        self.testSpace.elementMaps.getInverseValues(invJ,x,xiArray)
        self.testSpace.getBasisValuesAtArray(xiArray,vArray)

        cpostprocessing.getElementBDM1velocityValuesLagrangeRep(vArray,
                                                                self.q[('velocity_dofs',ci)],
                                                                vx)

        return vx

class VPP_PWL_BDM2(VPP_PWL_RT0):
    """
    This class is intended to implement BDM2 elements in proteus

    """
    def __init__(self,vectorTransport=None,vtComponents=[0]):
        VPP_PWL_RT0.__init__(self,vectorTransport=vectorTransport,vtComponents=vtComponents)
        self.postProcessingType = 'pwl-bdm2'
        self.localVelocityRepresentationFlag = 0
        self.degree = 2
        self.set_BDM_dimensions(self.degree)

        # Question - the space is set indepenedently for the interior test functions.  Should this be the case for the trial functions as well?
        # at a minimum there should be an assert preventing the user from using Linear functions in the BDM2 projection.
        
        for ci in self.vtComponents:
            self.nDOFs_element[ci] = self.dim  # BDM2 requires quadratic elements
            self.q[('velocity_dofs',ci)] = numpy.zeros((self.vt.mesh.nElements_global,self.nDOFs_element[ci]),'d')
            if ci != self.vtComponents[0] and ('velocity_l2g',self.vtComponents[0]) in self.q:
                self.q[('velocity_l2g',ci)]  = self.q[('velocity_l2g',self.vtComponents[0])]
            else:
                self.q[('velocity_l2g',ci)]  = numpy.arange((self.vt.mesh.nElements_global*self.nDOFs_element[ci]),dtype='i').reshape((self.vt.mesh.nElements_global,self.nDOFs_element[ci]))

        self.BDMcomponent=self.vtComponents[0]
        self.BDMprojectionMat_element = numpy.zeros((self.vt.mesh.nElements_global,
                                                     self.nDOFs_element[self.BDMcomponent],
                                                     self.nDOFs_element[self.BDMcomponent]),
                                                    'd')
        self.BDMprojectionMatPivots_element = numpy.zeros((self.vt.mesh.nElements_global,
                                                           self.nDOFs_element[self.BDMcomponent]),
                                                          'i')

        self.interiorTestSpace = None
        self.setInteriorTestSpace(self.degree)        
        self.getInteriorTestGradients()
        self.getInteriorDivFreeElement()
        self.setEdgeFlags()
        self.computeBDM2projectionMatrices()
        

    def setInteriorVelocityValues(self,ci):
        """ This function sets the interior velocity values based on the solution to the vt-problem. """
        if ('a',ci,ci) in self.vt.q:
            assert ('grad(phi)',ci) in self.vt.q
            updateCoef = 0.0 #overwrite first time
            if self.vt.sd:
                cpostprocessing.updateDiffusiveVelocityPointEval_sd(updateCoef,
                                                                    self.vt.coefficients.sdInfo[(ci,ci)][0],
                                                                    self.vt.coefficients.sdInfo[(ci,ci)][1],
                                                                    self.vt.q[('a',ci,ci)],
                                                                    self.vt.q[('grad(phi)',ci)],
                                                                    self.q[('velocity',ci)])
            else:
                cpostprocessing.updateDiffusiveVelocityPointEval(updateCoef,
                                                                 self.vt.q[('a',ci,ci)],
                                                                 self.vt.q[('grad(phi)',ci)],
                                                                 self.q[('velocity',ci)])
        if ('f',ci) in self.vt.q:
            updateCoef = 1.0
            cpostprocessing.updateAdvectiveVelocityPointEval(updateCoef,
                                                             self.vt.q[('f',ci)],
                                                             self.q[('velocity',ci)])

    def set_BDM_dimensions(self,degree):
        ''' Calculate and set the BDM polynomial dimension
            input - degree
        '''
        if self.vt.nSpace_global == 2:
            self.dim = (degree+1)*(degree+2)
            self.boundary_dim_per_edge = degree+1
            self.boundary_dim = 3*self.boundary_dim_per_edge
            self.interior_dim = self.dim - self.boundary_dim

        elif self.vt.nSpace_global == 3:
            self.dim = (degree+1)*(degree+2)*(degree+3) // 2
            self.boundary_dim_per_edge = degree*(degree+1)
            self.boundary_dim = 4 * self.boundary_dim_per_edge
            self.interior_dim = self.dim - self.boundary_dim
            
        else:
            # Need to raise exception here.
            pass

    def get_num_sigmaBasisElements(self):
        if self.vt.nSpace_global==2:
            return 1
        if self.vt.nSpace_global==3:
            return 3

    def setInteriorTestSpace(self,degree):
        ''' This function sets the interior test space corresponding to the dimension
        of the BDM space        
        '''
        if degree==2:
            self.interiorTestSpace = FemTools.C0_AffineLinearOnSimplexWithNodalBasis(self.vt.mesh,self.vt.nSpace_global)
        else:
            pass

    def setEdgeFlags(self):
        """This function sets the edge flags for the bdm2 loops """
        if self.vt.nSpace_global == 2:
            self.edgeFlags = numpy.array([1,2,4,0,2,5,0,1,3],'i')
        elif self.vt.nSpace_global == 3:
            self.edgeFlags = numpy.array([1,2,3,5,6,8, 0,2,3,6,7,9, 0,1,3,4,8,9, 0,1,2,4,5,7],'i')
            # Note - the numbers above should be correct but have not been tested.
        else:
            pass # This should be an assert.

    def getInteriorTestGradients(self):
        '''
        Calculate and return the gradient values for polynomials of degree 1
        '''
        # Initialze some arrays to store the results...
        self.interiorTestGradients = numpy.zeros((self.vt.mesh.nElements_global,
                                              self.vt.nQuadraturePoints_element,
                                              self.interiorTestSpace.referenceFiniteElement.localFunctionSpace.dim,
                                              self.interiorTestSpace.referenceFiniteElement.referenceElement.dim),
                                              'd')
        self.weightedInteriorTestGradients = numpy.zeros((self.vt.mesh.nElements_global,
                                              self.vt.nQuadraturePoints_element,
                                              self.interiorTestSpace.referenceFiniteElement.localFunctionSpace.dim,
                                              self.interiorTestSpace.referenceFiniteElement.referenceElement.dim),
                                              'd') 
        self.interiorTestSpace.getBasisGradientValues(self.vt.elementQuadraturePoints,
                                                      self.vt.q['inverse(J)'],
                                                      self.interiorTestGradients)
        # TODO - remove weighted shape PIOLa
        cfemIntegrals.calculateWeightedShapeGradients(self.vt.elementQuadratureWeights[('f',0)],
                                                      self.vt.q['abs(det(J))'],
                                                      self.interiorTestGradients,
                                                      self.weightedInteriorTestGradients)

    def sigmaBasisElement(self,dim,component,point,i=0):
        if dim == 2:
            if component == 0:
                return point[0] - point[0]**2 - 2*point[0]*point[1]
            if component == 1:
                return -point[1] + point[1]**2 + 2*point[0]*point[1]
        if dim == 3:
            if i == 0:
                if component == 0:
                    # -xy + xz
                    return point[0]*(1. - point[0] - 2.*point[1] - point[2])
                if component == 1:
                    # xy - yz
                    return point[1]*(-1. + 2.*point[0] + point[1] + point[2])
                if component == 2:
                    # -xz + yz
                    return 0.
            if i == 1:
                if component == 0:
                    # 2xy - 2xz
                    return 2*point[0]*(point[1] - point[2])
                if component == 1:
                    # y - 3xy - y*y
                    return point[1]*(1 - 3*point[0] - point[1])
                if component == 2:
                    # -z + 3xz + z*z
                    return point[2]*(3*point[0] - 1)
            if i == 2:
                if component == 0:
                    # x - x*x - 2xy - xz
                    return point[0]*(1 - point[0] - point[1] )
                if component == 1:
                    # -y + 2xy + y*y + y*z
                    return point[1]*(-1. + 3*point[0] + point[1])
                if component == 2:
                    return -point[0]*point[2]

    def getInteriorDivFreeElement(self):
        '''
        Calculate and return the values of the divergence free interior test function.
        Note - calculating this integral requires the use of the Piola transformation.
        '''

        n_xi = self.vt.nQuadraturePoints_element
        psi = numpy.zeros((n_xi, self.vt.nSpace_global, self.get_num_sigmaBasisElements() ), 'd')
        
        # start by calculating the interior DivFreeElements
        self.interiorDivFreeElement = numpy.zeros((self.vt.mesh.nElements_global,
                                                   self.vt.nQuadraturePoints_element,
                                                   self.vt.nSpace_global,
                                                   self.get_num_sigmaBasisElements() ),'d')
        
        self.weightedInteriorDivFreeElement = numpy.zeros((self.vt.mesh.nElements_global,
                                                           self.vt.nQuadraturePoints_element,
                                                           self.vt.nSpace_global,
                                                           self.get_num_sigmaBasisElements() ),'d')
        
        for k in range(n_xi):
            for j in range(self.vt.nSpace_global):
                for i in range(self.get_num_sigmaBasisElements() ):
                    psi[k,j,i] = self.sigmaBasisElement(self.vt.nSpace_global,
                                                        j,
                                                        self.vt.elementQuadraturePoints[k],
                                                        i)
        # TODO - add C routines for the following functions (see FemTools.py getBasisValues for an example)

        # Populate interiorDivFreeElement 
        for eN in range(self.vt.mesh.nElements_global):
            for k in range(n_xi):
                for j in range(self.vt.nSpace_global):
                    for i in range(self.get_num_sigmaBasisElements()):
                        self.interiorDivFreeElement[eN,k,j,i] = psi[k,j,i]
        
        # Next, populate weightedInteriorDivFreeElement
        for eN in range(self.vt.mesh.nElements_global):
            for k in range(n_xi):
                for j in range(self.vt.nSpace_global):
                    # Apply Jacobian matrix
                    for i in range(self.get_num_sigmaBasisElements()):
                        for h in range(self.vt.nSpace_global):
                            self.weightedInteriorDivFreeElement[eN,k,j,i] += self.interiorDivFreeElement[eN,k,h,i]*self.q['J'][eN][k][j][h]
                    for i in range(self.get_num_sigmaBasisElements()):
                        # scale by Jacobian
                        self.weightedInteriorDivFreeElement[eN,k,j,i] *= 1./self.vt.q['abs(det(J))'][eN][k]
                        # scale with quadrature weight
                        self.weightedInteriorDivFreeElement[eN,k,j,i] *= self.vt.q['dV'][eN][k]


        self.piola_trial_function = numpy.zeros((self.vt.mesh.nElements_global,
                                                  self.vt.nQuadraturePoints_element,
                                                  self.dim,
                                                  self.vt.nSpace_global),'d')

        for eN in range(self.vt.mesh.nElements_global):
            for k in range(self.vt.nQuadraturePoints_element):
                for i in range(self.dim//self.vt.nSpace_global):
                    for j in range(self.vt.nSpace_global):
                        self.piola_trial_function[eN,k,i*self.vt.nSpace_global+j,j] = self.q[('w',self.BDMcomponent)][eN][k][i]

    def computeBDM2projectionMatrices(self):

        cpostprocessing.buildLocalBDM2projectionMatrices(self.degree,
                                                         self.vt.ebq[('w*dS_u',self.BDMcomponent)],
#                                                         self.w_dS[self.BDMcomponent],#vt.ebq[('w*dS_u',self.BDMcomponent)],
                                                         self.vt.ebq['n'],
                                                         self.vt.ebq[('v',self.BDMcomponent)],#self.w[self.BDMcomponent]
                                                         self.vt.q[('w',self.BDMcomponent)],          # interior integrals - gradient part
                                                         self.weightedInteriorTestGradients,       # interior integrals - gradient part
                                                         self.weightedInteriorDivFreeElement,      # interior integrals - divFree part
                                                         self.piola_trial_function,                # interior integrals - divFree part
                                                         self.edgeFlags,
                                                         self.BDMprojectionMat_element)            # projection matrix

        cpostprocessing.factorLocalBDM2projectionMatrices(self.BDMprojectionMat_element,
                                                          self.BDMprojectionMatPivots_element)



    def flagNeumannBoundaryEdges(self):
        ''' This function flags neumann boundaries.

        TODO - WIP - part of some experimental functions
        designed for LN approximations.
        NOTE - Currently this function is not operational and
        it returns a dictionary which suggests all edges
        are Dirichlet.
        '''
        self.neumann_edge_dict = {}
        for edge in self.vt.mesh.exteriorElementBoundariesArray:
            self.neumann_edge_dict[edge] = False

    def conservativeVelocityCalculation(self):
        '''Serial implimentation of LN algorithm.

        TODO - This function is experimental and should be
        refactor once completed.
        '''
        # Initialize the matrix - serial LN requires a system
        # the size of the number of elements in the mesh.
        num_elements = self.vt.mesh.nElements_global
        num_element_quadpts = self.vt.q['dV'].shape[1]
        num_bdy_quadpts = self.vt.ebq_global['n'][0].shape[0]
        num_edges = self.vt.mesh.nElementBoundaries_global
        dim = self.vt.ebq_global['n'][0].shape[1]
        h = self.vt.mesh.h

        # Flag Neumann terms
        self.flagNeumannBoundaryEdges()
        self.getAverageFlux()

        # Allocate space for matrices
        A = numpy.zeros(shape=(num_elements,num_elements))
        b = numpy.zeros(shape=(num_elements,1))
        flux_approx = numpy.zeros(shape=(num_edges,1))

        # loop over all elements in the mesh
        for k in range(num_elements):
            # Need to calculate the source term integral
            # Eg. - int_{K} (f, 1)_{K}
            for pt in range(num_element_quadpts):
                # Q for ck - is q[('r',0)][k][pt] effectively f 
                b[k] += (self.vt.q[('r',0)][k][pt] *  
                         self.vt.q['dV'][k][pt])
            # Need to loop over edges
            # Need to find Neumann edges and Dirichlet edges
            for local_edge_num , global_edge_num in enumerate(self.vt.mesh.elementBoundariesArray[k]):
                print('element : ' + repr(k))
                # Diagonal element
                A[k][k] += 1/h * ( sum(self.vt.ebq[('dS_u',0)][k][local_edge_num]) )
                # Calculate A's off-diagonal terms
                if global_edge_num in self.vt.mesh.interiorElementBoundariesArray:
                    elements = self.vt.mesh.elementBoundaryElementsArray[global_edge_num]
                    if k == elements[0]:
                        j = elements[1]
                    else:
                        j = elements[0]
                    # off diagonal elements
                    A[k][j] += ( sum(self.vt.ebq[('dS_u',0)][k][local_edge_num]) )
                # Calculate Flux Inner Products
                for pt in range(num_bdy_quadpts):
                    for comp in range(dim):
                        b[k] += ( self.flux_average[global_edge_num] *
                                  self.vt.ebq[('dS_u',0)][k][local_edge_num][pt] )
        print('loop done')
        V = numpy.linalg.solve(A,b)
#       self.CorrectedFlux = self.flux_average - V

    def getAverageFlux(self,ci):
        '''Helper function to cacluate the average flux along the mesh boundaries. '''
        # Step 1 : calculate the normal component of all velocities

        num_elements = self.vt.mesh.nElements_global
        num_edges = self.vt.mesh.nElementBoundaries_global
        num_bdy_quadpts = self.vt.ebq_global['n'][0].shape[0]
        dim = self.vt.ebq_global['n'][0].shape[1]

        flux_array = numpy.zeros(shape=(num_edges,2))
        self.flux_average = numpy.zeros(shape=(num_edges,1))

        for element in range(num_elements):
            for local_edge_num , global_edge_num in enumerate(self.vt.mesh.elementBoundariesArray[element]):
                elements = self.vt.mesh.elementBoundaryElementsArray[global_edge_num]
                j = 0
                if element == elements[1]:
                    j = 1
                for pt in range(num_bdy_quadpts):
                    for comp in range(dim):
                        flux_array[global_edge_num][j] += (self.vt.ebq['n'][element][local_edge_num][pt][comp] *
                                                           self.vt.ebq[('velocity',ci)][element][local_edge_num][pt][comp] )
        for edge in range(num_edges):
            if edge in self.vt.mesh.interiorElementBoundariesArray:
                self.flux_average[edge] = (flux_array[edge][0]+flux_array[edge][1])/2.0
            else:
                self.flux_average[edge] = flux_array[edge][0]

    def computeGeometricInfo(self):
        if self.BDMcomponent is not None:
            self.computeBDM2projectionMatrices()

    def evaluateLocalVelocityRepresentation(self,ci,velocity_field_set=False):
        """
        This function projects the solution of the vt transport problem into the
        BDM2 space.

        Parameters
        ----------
        ci : int
            The solution component being projected.
        velocity_field_set: boolean
            Set to true when the quadrature values for the velocity function are 
            already defined (this is typically used for testing).
        """
        assert self.nDOFs_element[ci] == self.dim , "wrong size for BDM"
            
        if self.degree >= 2 and velocity_field_set==False:
            for ci in self.vtComponents:
                self.setInteriorVelocityValues(ci)

        cpostprocessing.buildBDM2rhs(self.BDMprojectionMat_element,
                                     self.BDMprojectionMatPivots_element,
                                     self.ebq[('w*dS_u',self.BDMcomponent)],
                                     #                          self.w_dS[ci],
                                     self.vt.ebq['n'],
                                     self.weightedInteriorTestGradients,
                                     self.weightedInteriorDivFreeElement,
                                     self.ebq[('velocity',ci)],
                                     self.q[('velocity',ci)],
                                     self.q[('velocity_dofs',ci)],
                                     self.edgeFlags)
        
        cpostprocessing.solveLocalBDM2projection(self.BDMprojectionMat_element,
                                      self.BDMprojectionMatPivots_element,
#                                      self.w_dS[ci],
                                      self.ebq[('w*dS_u',self.BDMcomponent)],
                                      self.vt.ebq['n'],
                                      self.weightedInteriorTestGradients,
                                      self.ebq[('velocity',ci)],
                                      self.q[('velocity',ci)],
                                      self.q[('velocity_dofs',ci)])

        cpostprocessing.getElementBDM2velocityValuesLagrangeRep(self.vt.q[('w',ci)],
                                                                self.q[('velocity_dofs',ci)],
                                                                self.vt.q[('velocity',ci)])


    def evaluateElementVelocityField(self,x,ci):
        """
        evaluate velocity field assuming velocity_dofs already calculated
        for now assumes x shaped like nE x nq x 3
        TODO:
           put python loops in c
        """
        assert len(x.shape) == 3, "wrong shape for x= %s " % x.shape
        nE = x.shape[0]; nq = x.shape[1]; nd = self.vt.nSpace_global
        vx = numpy.zeros((nE,nq,nd),'d')
        xiArray = numpy.zeros(x.shape,'d')
        vArray  = numpy.zeros((nE,nq,self.testSpace.max_nDOF_element),'d')
        invJ    = numpy.zeros((nE,nq,self.vt.q['inverse(J)'].shape[2],self.vt.q['inverse(J)'].shape[3]),
                                'd')
        for ie in range(nE):
            for iq in range(nq):
                invJ[ie,iq,:,:] = self.vt.q['inverse(J)'][ie,0,:,:] #assume affine
            #iq
        #iq
        self.testSpace.elementMaps.getInverseValues(invJ,x,xiArray)
        self.testSpace.getBasisValuesAtArray(xiArray,vArray)

        cpostprocessing.getElementBDM2velocityValuesLagrangeRep(vArray,
                                                                self.q[('velocity_dofs',ci)],
                                                                vx)

        return vx


class VPP_PWL_RT0_OPT(VPP_PWL_RT0):
    """
    Version of local Larson-Niklasson scheme to use with optimized
    codes Assumes that all exterior boundaries are either flux or weak
    dirichlet boundaries and that exterior velocity has been
    calculated correctly (i.e., it's consistent with either flux or
    weak dirichlet approximation)

    Refused unless the numerical flux uses weak (Nitsche-type) Dirichlet
    conditions. The mprans models (RANS2P, RANS3PF) put the weak boundary
    flux into ebq_global['velocityAverage']; the classic Transport does not
    (its exterior value is one-sided), so with it only the interior
    elements balance.

    Assumes layout the same across components
    """
    def __init__(self,vectorTransport=None,vtComponents=[0]):
        VPP_PWL_RT0.__init__(self,vectorTransport,vtComponents=vtComponents)
        #have to directly modify the type now to set to optimized
        self.postProcessingType = 'pwl-opt'
        self.setupOptimizedAssembly()

    def setupOptimizedAssembly(self):
        """What the optimized variants add to the PWL setup: the weak-condition check, and the parallel vectors that
        sum the owned-node star corrections on each face across ranks."""
        # The optimized variants take the boundary flux from the face velocity the model computed and never load a
        # boundary condition themselves, so every exterior face must carry the weak (Nitsche-type) boundary flux.
        if self.vt.numericalFlux is None or not self.vt.numericalFlux.useWeakDirichletConditions:
            raise RuntimeError("conservativeFlux 'pwl-opt', 'pwl-bdm-opt' and 'pwl-bdm2-opt' need weak (Nitsche-type) "
                               "Dirichlet conditions: the boundary flux must come from the model's face velocity; "
                               "use 'pwl', 'pwl-bdm' or 'pwl-bdm2' with strong Dirichlet conditions")

        from .LinearAlgebraTools import ParVec_petsc4py as ParVec

        self.ebq_v_par_local = self.vt.ebq_global[('velocity',self.vtComponents[0])].copy()
        self.ebq_v_par = ParVec(self.ebq_v_par_local,
                                self.vt.nElementBoundaryQuadraturePoints_elementBoundary*self.vt.nSpace_global,#block size
                                self.vt.mesh.nElementBoundaries_owned,#number of local element boundaries owned
                                self.vt.mesh.globalMesh.nElementBoundaries_global,#number of global element boundaries
                                self.vt.mesh.nElementBoundaries_global - self.vt.mesh.nElementBoundaries_owned,#number of ghost element boundaries
                                self.vt.mesh.globalMesh.elementBoundaryNumbering_subdomain2global)
        self.ebq_x_par_local = self.vt.ebq_global['x'].copy()
        self.x_par = ParVec(self.ebq_x_par_local,
                            self.vt.nElementBoundaryQuadraturePoints_elementBoundary*3,#block size
                            self.vt.mesh.nElementBoundaries_owned,#number of local element boundaries owned
                            self.vt.mesh.globalMesh.nElementBoundaries_global,#number of global element boundaries
                            self.vt.mesh.nElementBoundaries_global - self.vt.mesh.nElementBoundaries_owned,#number of ghost element boundaries
                            self.vt.mesh.globalMesh.elementBoundaryNumbering_subdomain2global)
        self.x_par.scatter_forward_insert()
        self.permutations = numpy.zeros((self.vt.mesh.nElementBoundaries_global,self.vt.nElementBoundaryQuadraturePoints_elementBoundary),'i')
        cfemIntegrals.getPermutationsGlobal(self.vt.ebq_global['x'],self.ebq_x_par_local,self.permutations)

    def postprocess_component(self,ci,verbose=0):
        """
        compute mass conservative velocity field following Larson and Niklasson assuming a P^k C0
        Galerkin solution has already been found
        Setup for optimized parallel codes
        """
        from .Comm import globalSum,globalMax

        self.nodeStarFactors[ci].setU(0.0)
        if self.solutionTestSpaceIsNotPWL:
            cpostprocessing.calculateElementResidualPWL(self.alpha[ci],self.vt.elementResidual[ci],self.elementResidual[ci])
        self.getConservationResidualPWL(ci)
        if self.updateConservationJacobian[ci]:
            self.getConservationJacobianPWL(ci)
            self.updateConservationJacobian[ci] = False #only depends on mesh need to resignal if mesh adapts
        #calculation partial corrections on owned node stars
        cpostprocessing.calculateConservationFluxPWL_opt(self.vt.mesh.nNodes_owned,
                                                         self.nElements_node,
                                                         self.vt.internalNodesArray,
                                                         self.fluxBoundaryNodes[ci],
                                                         self.nodeStarFactors[ci])
        self.getConservationResidualPWL(ci)

        #do parallel assembly of correction
        useC=True
        if useC:
            cpostprocessing.copyElementBoundaryVelocityToParVec(self.vt.ebq_global[('velocity',ci)],self.permutations,self.ebq_v_par_local)
            self.ebq_v_par.scatter_reverse_add()
            cpostprocessing.addAverageToParVec(self.vt.ebq_global[('velocityAverage',ci)],self.permutations,self.ebq_v_par_local)
            self.ebq_v_par.scatter_forward_insert()
            cpostprocessing.copyParVecToElementBoundaryVelocity(self.vt.ebq_global[('velocity',ci)],self.permutations,self.ebq_v_par_local)
        else:
            #copy partial corrections to global ordering and sum on owning processors
            for ebN in range(self.ebq_v_par_local.shape[0]):
                for k in range(self.ebq_v_par_local.shape[1]):
                    for I in range(self.ebq_v_par_local.shape[2]):
                        self.ebq_v_par_local[ebN,self.permutations[ebN,k],I] = self.vt.ebq_global[('velocity',ci)][ebN,k,I]
            self.ebq_v_par.scatter_reverse_add()
            #add correction to average velocity
            for ebN in range(self.ebq_v_par_local.shape[0]):
                for k in range(self.ebq_v_par_local.shape[1]):
                    for I in range(self.ebq_v_par_local.shape[2]):
                        self.ebq_v_par_local[ebN,self.permutations[ebN,k],I] += self.vt.ebq_global[('velocityAverage',ci)][ebN,k,I]
            #send to ghost boundaries
            self.ebq_v_par.scatter_forward_insert()
            for ebN in range(self.ebq_v_par_local.shape[0]):
                for k in range(self.ebq_v_par_local.shape[1]):
                    for I in range(self.ebq_v_par_local.shape[2]):
                        self.vt.ebq_global[('velocity',ci)][ebN,k,I] = self.ebq_v_par_local[ebN,self.permutations[ebN,k],I]

        #copy to ebq storage
        cfemIntegrals.copyGlobalElementBoundaryVelocityToElementBoundary(self.vt.mesh.interiorElementBoundariesArray,
                                                                         self.vt.mesh.exteriorElementBoundariesArray,
                                                                         self.vt.mesh.elementBoundaryElementsArray,
                                                                         self.vt.mesh.elementBoundaryLocalElementBoundariesArray,
                                                                         self.vt.ebq_global[('velocity',ci)],
                                                                         self.vt.ebq[('velocity',ci)])
        cfemIntegrals.copyExteriorElementBoundaryValuesFromGlobalElementBoundaryValues(self.vt.mesh.exteriorElementBoundariesArray,
                                                                                       self.vt.mesh.elementBoundaryElementsArray,
                                                                                       self.vt.mesh.elementBoundaryLocalElementBoundariesArray,
                                                                                       self.vt.ebq_global[('velocity',ci)],
                                                                                       self.vt.ebqe[('velocity',ci)])
        #go from boundary flux to local element boundary representation
        self.evaluateLocalVelocityRepresentation(ci)

        cpostprocessing.calculateConservationResidualPWL_primative(self.vt.mesh.interiorElementBoundariesArray,
                                                                   self.vt.mesh.exteriorElementBoundariesArray,
                                                                   self.vt.mesh.elementBoundaryElementsArray,
                                                                   self.vt.mesh.elementBoundaryLocalElementBoundariesArray,
                                                                   self.fluxElementBoundaries[ci],
                                                                   self.elementResidual[ci],
                                                                   self.vt.ebq[('dS_u',ci)],
                                                                   self.vt.ebq_global['n'],
                                                                   self.q[('conservationResidual',ci)],
                                                                   self.ebq_global[('velocity',ci)])

        logEvent("Max local conservation (dgp1 enriched all elements) = %12.5e" % globalMax(max(numpy.absolute(self.q[('conservationResidual',ci)].flat))))
        divergence = Norms.fluxDomainBoundaryIntegralFromVector(self.vt.ebqe['dS'],
                                                                self.vt.ebqe[('velocity',ci)],
                                                                self.vt.ebqe['n'],
                                                                self.vt.mesh)
        logEvent("Global divergence = %12.5e" % (divergence,),level=2)

    def getConservationJacobianPWL(self,ci):
        """
        Build local systems for post-processing solve
        """
        cpostprocessing.calculateConservationJacobianPWL_opt(self.vt.mesh.nNodes_owned,
                                                             self.vt.mesh.interiorElementBoundariesArray,
                                                             self.vt.mesh.exteriorElementBoundariesArray,
                                                             self.vt.mesh.elementBoundaryElementsArray,
                                                             self.vt.mesh.elementBoundaryLocalElementBoundariesArray,
                                                             self.vt.mesh.elementNodesArray,
                                                             self.nodeStarElementsArray,
                                                             self.nodeStarElementNeighborsArray,
                                                             self.nElements_node,
                                                             self.vt.internalNodesArray,
                                                             self.fluxElementBoundaries[ci],
                                                             self.fluxBoundaryNodes[ci],
                                                             self.w_dS[ci],
                                                             self.vt.ebq_global['n'],
                                                             self.nodeStarFactors[ci])
    def getConservationResidualPWL(self,ci,correctFlux=False):
        """
        compute conservation resiudal using current guess for element boundary flux
        """
        cpostprocessing.calculateConservationResidualPWL_opt(self.vt.mesh.nNodes_owned,
                                                         self.vt.mesh.interiorElementBoundariesArray,
                                                         self.vt.mesh.exteriorElementBoundariesArray,
                                                         self.vt.mesh.elementBoundaryElementsArray,
                                                         self.vt.mesh.elementBoundaryLocalElementBoundariesArray,
                                                         self.vt.mesh.elementNodesArray,
                                                         self.nodeStarElementsArray,
                                                         self.nodeStarElementNeighborsArray,
                                                         self.nElements_node,
                                                         self.fluxElementBoundaries[ci],
                                                         self.elementResidual[ci],
                                                         self.vt.ebq_global[('velocityAverage',ci)],
                                                         self.vt.ebq[('dS_u',ci)],
                                                         self.w[ci],
                                                         self.vt.ebq_global['n'],
                                                         self.nodeStarFactors[ci],
                                                         self.q[('conservationResidual',ci)],
                                                         self.vt.ebq_global[('velocity',ci)],
                                                         self.ebq[('velocity',ci)])


    def evaluateElementVelocityField(self,x,ci):
        """
        evaluate velocity field assuming velocity_dofs already calculated
        for now assumes x shaped like nE x nq x 3
        """
        assert len(x.shape) == 3, "wrong shape for x= %s " % x.shape
        nE = x.shape[0]; nq = x.shape[1]; nd = self.vt.nSpace_global
        vx = numpy.zeros((nE,nq,nd),'d')
        cpostprocessing.getElementRT0velocityValuesFluxRep(self.vt.mesh.nodeArray,
                                                           self.vt.mesh.elementNodesArray,
                                                           self.vt.q['abs(det(J))'],
                                                           x,
                                                           self.q[('velocity_dofs',ci)],
                                                           vx)
        return vx

class VPP_PWL_BDM_OPT(VPP_PWL_RT0_OPT):
    r"""Local Larson-Niklasson method with BDM1 representation for element
    velocities for optimized parallel codes

    Only difference from VPP_PWL_RT0_OPT should be steps for local
    velocity representation This one uses BDM_1 space which is :math:`[P^1]^d`
    locally with continuous linear fluxes on each face use standard
    basis

    .. math::

      \vec N_i = \lambda_{i \bmod (d+1)} \vec e_{i/(d+1)}

    Have to use BDM projection to get degrees of freedom

    """
    def __init__(self,vectorTransport=None,vtComponents=[0]):
        VPP_PWL_RT0_OPT.__init__(self,vectorTransport=vectorTransport,vtComponents=vtComponents)
        #have to directly modify the type now to show bdm
        self.postProcessingType = 'pwl-bdm-opt'
        #how is the local velocity represented
        #  0 -- BDM P^1 Lagrange rep
        self.localVelocityRepresentationFlag = 0

        #
        for ci in self.vtComponents:
            self.nDOFs_element[ci] = self.vt.nSpace_global*(self.vt.nSpace_global+1)
            self.q[('velocity_dofs',ci)] = numpy.zeros((self.vt.mesh.nElements_global,self.nDOFs_element[ci]),'d')
            if ('velocity_l2g',self.vtComponents[0]) in self.q:
                self.q[('velocity_l2g',ci)]  = self.q[('velocity_l2g',self.vtComponents[0])]
            else:
                self.q[('velocity_l2g',ci)]  = numpy.arange((self.vt.mesh.nElements_global*self.nDOFs_element[ci]),dtype='i').reshape((self.vt.mesh.nElements_global,self.nDOFs_element[ci]))
        #

        self.BDMcomponent=self.vtComponents[0]
        self.BDMprojectionMat_element = numpy.zeros((self.vt.mesh.nElements_global,
                                                     self.nDOFs_element[self.BDMcomponent],
                                                     self.nDOFs_element[self.BDMcomponent]),
                                                    'd')
        self.BDMprojectionMatPivots_element = numpy.zeros((self.vt.mesh.nElements_global,
                                                           self.nDOFs_element[self.BDMcomponent]),
                                                          'i')
        self.computeBDM1projectionMatrices()

    def computeBDM1projectionMatrices(self):
        cpostprocessing.buildLocalBDM1projectionMatrices(self.w_dS[self.BDMcomponent],#vt.ebq[('w*dS_u',self.BDMcomponent)],
                                                         self.vt.ebq['n'],
                                                         self.w[self.BDMcomponent],#self.vt.ebq[('v',self.BDMcomponent)],
                                                         self.BDMprojectionMat_element)
        cpostprocessing.factorLocalBDM1projectionMatrices(self.BDMprojectionMat_element,
                                                          self.BDMprojectionMatPivots_element)
    def updateWeights(self):
        for ci in self.vtComponents:
            if not isinstance(self.vt.u[ci].femSpace,FemTools.C0_AffineLinearOnSimplexWithNodalBasis):
                self.testSpace.getBasisValues(self.vt.elementQuadraturePoints,
                                              self.qv[ci])
                self.testSpace.getBasisValuesTrace(self.vt.u[0].femSpace.elementMaps.permutations,
                                                   self.vt.ebq['hat(x)'],
                                                   self.w[ci])
                cfemIntegrals.calculateWeightedShapeTrace(self.vt.elementBoundaryQuadratureWeights[('u',ci)],
                                                          self.vt.ebq['sqrt(det(g))'],
                                                          self.w[ci],
                                                          self.w_dS[ci])
            else:
                if ('w*dS_f',ci) not in self.ebq:
                    self.w_dS[ci] = numpy.zeros(
                        (self.vt.mesh.nElements_global,
                         self.vt.mesh.nElementBoundaries_element,
                         self.vt.nElementBoundaryQuadraturePoints_elementBoundary,
                         self.vt.nDOF_test_element[ci]),
                        'd')
                    #since self.ebq = self.vt.ebq this gets updated by vt when the mesh changes
                    cfemIntegrals.calculateWeightedShapeTrace(self.vt.elementBoundaryQuadratureWeights[('u',ci)],
                                                              self.ebq['sqrt(det(g))'],
                                                              self.ebq[('w',ci)],
                                                              self.w_dS[ci])
    def computeGeometricInfo(self):
        if self.BDMcomponent is not None:
            self.computeBDM1projectionMatrices()

    def evaluateLocalVelocityRepresentation(self,ci):
        """
        project to BDM velocity from element boundary fluxes
        """
        assert self.nDOFs_element[ci] == self.vt.nSpace_global*(self.vt.nSpace_global+1), "wrong size for BDM"

        cpostprocessing.solveLocalBDM1projection(self.BDMprojectionMat_element,
                                                 self.BDMprojectionMatPivots_element,
                                                 self.w_dS[ci],
                                                 self.vt.ebq['n'],
                                                 self.ebq[('velocity',ci)],
                                                 self.q[('velocity_dofs',ci)])

        cpostprocessing.getElementBDM1velocityValuesLagrangeRep(self.qv[ci],
                                                                self.q[('velocity_dofs',ci)],
                                                                self.vt.q[('velocity',ci)])



    def evaluateElementVelocityField(self,x,ci):
        """
        evaluate velocity field assuming velocity_dofs already calculated
        for now assumes x shaped like nE x nq x 3
        """
        assert len(x.shape) == 3, "wrong shape for x= %s " % x.shape
        nE = x.shape[0]; nq = x.shape[1]; nd = self.vt.nSpace_global
        vx = numpy.zeros((nE,nq,nd),'d')
        #have to evaluate shape functions at new points. This is painful
        #uci     = self.vt.u[ci]
        xiArray = numpy.zeros(x.shape,'d')
        vArray  = numpy.zeros((nE,nq,self.testSpace.max_nDOF_element),'d')
        invJ    = numpy.zeros((nE,nq,self.vt.q['inverse(J)'].shape[2],self.vt.q['inverse(J)'].shape[3]),
                                'd')
        for ie in range(nE):
            for iq in range(nq):
                invJ[ie,iq,:,:] = self.vt.q['inverse(J)'][ie,0,:,:] #assume affine
            #iq
        #iq
        self.testSpace.elementMaps.getInverseValues(invJ,x,xiArray)
        self.testSpace.getBasisValuesAtArray(xiArray,vArray)

        cpostprocessing.getElementBDM1velocityValuesLagrangeRep(vArray,
                                                                self.q[('velocity_dofs',ci)],
                                                                vx)

        return vx



class VPP_PWL_BDM2_OPT(VPP_PWL_BDM2):
    r"""Local Larson-Niklasson method with the BDM2 representation, for optimized parallel codes.

    The face fluxes come from the pwl-opt pipeline: star problems are solved for owned nodes only, the corrections are
    summed on each face across ranks, and the boundary flux is the model's face velocity, so weak (Nitsche-type)
    Dirichlet conditions are required. The element velocity is then the BDM2 projection of pwl-bdm2: the face
    fluxes plus interior moments of the transport's pointwise velocity. Needs a C0 P2 solution, as pwl-bdm2 does.
    """
    def __init__(self,vectorTransport=None,vtComponents=[0]):
        VPP_PWL_BDM2.__init__(self,vectorTransport=vectorTransport,vtComponents=vtComponents)
        self.postProcessingType = 'pwl-bdm2-opt'
        self.setupOptimizedAssembly()

    setupOptimizedAssembly = VPP_PWL_RT0_OPT.setupOptimizedAssembly
    postprocess_component = VPP_PWL_RT0_OPT.postprocess_component
    getConservationJacobianPWL = VPP_PWL_RT0_OPT.getConservationJacobianPWL
    getConservationResidualPWL = VPP_PWL_RT0_OPT.getConservationResidualPWL


class VPP_PWC_RT0(VelocityPostProcessingAlgorithmBase):
    r"""Global piecewise constant Larson-Niklasson method with :math:`RT_0`
    representation for element velocities

    uses RT_0 representation in terms of local basis

    .. math::

      \vec N_i = \frac{1}{d|E|}(\vec x - p_i)

    where :math:`p_i` is the vertex across from face :math:`i`,
    :math:`|E|` is the volume of the element, and :math:`d` is the
    space dimension the degrees of freedom are :math:`V^i =
    \int_{e_i}\vec v\dot n_{i}\ds`

    """
#    TOOD:
#       make sure works with weak dirichlet conditions
#       remove python loops in building global Jacobian
    def __init__(self,vectorTransport=None,vtComponents=[0]):
        _requireOneRank('pwc')
        VelocityPostProcessingAlgorithmBase.__init__(self,postProcessingType='pwc',
                                                     vectorTransport=vectorTransport,
                                                     vtComponents=vtComponents)

        #how is the local velocity represented
        #  2 -- RT0, local rep is \sum^d_{i=0}V^i\vec N_{T,i},
        #           \vec N_{T,i} = \frac{1}{d|E|}(\vec x - p_i), i=0,...,d
        self.localVelocityRepresentationFlag = 2

        self.nDOFs_element = {}
        #local element residuals
        self.elementResidual = self.vt.elementResidual
        #data structures for solving global problem
        self.pwcMat = {}; self.pwcMat_zval = {}; self.pwcLU = {}; self.pwcV = {}
        #could just compute correction factors once since going to overwrite Neumann fluxes anyway
        self.ebq_global['pwc-corr'] = numpy.zeros((self.vt.mesh.nElementBoundaries_global,2),'d')
        volFact = 1.0; areaFact = 1.0
        if self.vt.nSpace_global == 2:
            volFact = 0.5
        if self.vt.nSpace_global == 3:
            volFact = 1.0/6.0; areaFact = 0.5
        for ebNI in range(self.vt.mesh.nInteriorElementBoundaries_global):
            ebN = self.vt.mesh.interiorElementBoundariesArray[ebNI]
            eN_left = self.vt.mesh.elementBoundaryElementsArray[ebN,0]
            eN_right= self.vt.mesh.elementBoundaryElementsArray[ebN,1]
            ebN_element_left = self.vt.mesh.elementBoundaryLocalElementBoundariesArray[ebN,0]
            area_face= areaFact*self.vt.ebq['sqrt(det(g))'][eN_left,ebN_element_left,0]
            self.ebq_global['pwc-corr'][ebN,0] = 1.0/area_face
            self.ebq_global['pwc-corr'][ebN,1] =-1.0/area_face
        for ebNE in range(self.vt.mesh.nExteriorElementBoundaries_global):
            ebN = self.vt.mesh.exteriorElementBoundariesArray[ebNE]
            eN_left = self.vt.mesh.elementBoundaryElementsArray[ebN,0]
            ebN_element_left = self.vt.mesh.elementBoundaryLocalElementBoundariesArray[ebN,0]
            area_face= areaFact*self.vt.ebq['sqrt(det(g))'][eN_left,ebN_element_left,0]
            self.ebq_global['pwc-corr'][ebN,0] = 1.0/area_face
        #end ebNE

        for ci in self.vtComponents:
            #correction
            self.ebq_global[('flux_correction',ci)] = numpy.zeros((self.vt.mesh.nElementBoundaries_global),'d')
            #velocity representation
            self.nDOFs_element[ci] = self.vt.nSpace_global+1
            self.q[('velocity_dofs',ci)] = numpy.zeros((self.vt.mesh.nElements_global,self.nDOFs_element[ci]),'d')
            if ('velocity_l2g',self.vtComponents[0]) in self.q:
                self.q[('velocity_l2g',ci)]  = self.q[('velocity_l2g',self.vtComponents[0])]
            else:
                self.q[('velocity_l2g',ci)]  = numpy.arange((self.vt.mesh.nElements_global*self.vt.mesh.nElementBoundaries_element),dtype='i').reshape((self.vt.mesh.nElements_global,self.vt.mesh.nElementBoundaries_element))

            #build jacobian, may have differences in components because of differences in boundary conditions
            #could use one version for all components in an optimized code (but need to address flux bc's someway)
            pwcMatGlobalDict = {}
            #mwf now build system to enforce Neumann bc's explicitly
            #build system assuming conforming mesh
            for ebNI in range(self.vt.mesh.nInteriorElementBoundaries_global):
                ebN = self.vt.mesh.interiorElementBoundariesArray[ebNI]
                eN_left = self.vt.mesh.elementBoundaryElementsArray[ebN,0]
                eN_right= self.vt.mesh.elementBoundaryElementsArray[ebN,1]
                ebN_element_left = self.vt.mesh.elementBoundaryLocalElementBoundariesArray[ebN,0]
                if (eN_left,eN_left) in pwcMatGlobalDict:
                    pwcMatGlobalDict[(eN_left,eN_left)] += 1.
                else:
                    pwcMatGlobalDict[(eN_left,eN_left)] = 1.
                if (eN_right,eN_right) in pwcMatGlobalDict:
                    pwcMatGlobalDict[(eN_right,eN_right)] += 1.
                else:
                    pwcMatGlobalDict[(eN_right,eN_right)] = 1.
                pwcMatGlobalDict[(eN_left,eN_right)] = -1.
                pwcMatGlobalDict[(eN_right,eN_left)] = -1.
            #interior
            #any chance an element didn't get hit by interior loop?
            for ebNE in range(self.vt.mesh.nExteriorElementBoundaries_global):
                ebN = self.vt.mesh.exteriorElementBoundariesArray[ebNE]
                if not self.fluxElementBoundaries[ci][ebNE]:
                    eN_left = self.vt.mesh.elementBoundaryElementsArray[ebN,0]
                    if (eN_left,eN_left) in pwcMatGlobalDict:
                        pwcMatGlobalDict[(eN_left,eN_left)] += 1.
                    else:
                        pwcMatGlobalDict[(eN_left,eN_left)] = 1.
            #exterior
            (self.pwcMat[ci],self.pwcMat_zval[ci]) = SparseMatFromDict(self.vt.mesh.nElements_global,
                                                                       self.vt.mesh.nElements_global,
                                                                       pwcMatGlobalDict)
            self.pwcLU[ci] = LinearSolvers.LU(self.pwcMat[ci])
            self.pwcLU[ci].prepare()
            #make this the number of entries in system (if had element with no correction?
            self.pwcV[ci] = numpy.zeros((self.vt.mesh.nElements_global,),'d')


        #
    def postprocess_component(self,ci,verbose=0):
        """compute mass conservative velocity field following Larson and
        Niklasson Global piecewise constant correcction

        """
        #compute initial element conservation residual for average velocity
        self.q[('conservationResidual',ci)].fill(0.0)
        self.ebq_global[('flux_correction',ci)].fill(0.0)
        self.ebq_global[('velocity',ci)].flat[:]=self.vt.ebq_global[('velocityAverage',ci)].flat

        #need to add back in after solves, now skip Neumann boundaries since enforcing them explicitly
        self.removeBoundaryFluxesFromResidual(ci,self.fluxElementBoundaries[ci])

        cpostprocessing.calculateConservationResidualGlobalBoundaries(self.vt.mesh.interiorElementBoundariesArray,
                                                                      self.vt.mesh.exteriorElementBoundariesArray,
                                                                      self.vt.mesh.elementBoundaryElementsArray,
                                                                      self.vt.mesh.elementBoundaryLocalElementBoundariesArray,
                                                                      self.fluxElementBoundaries[ci],
                                                                      self.vt.ebq[('dS_u',ci)],
                                                                      self.vt.ebq_global['n'],
                                                                      self.elementResidual[ci],
                                                                      self.ebq_global[('velocity',ci)],
                                                                      self.q[('conservationResidual',ci)])

        self.pwcLU[ci].solve(self.pwcV[ci],b=self.q[('conservationResidual',ci)])
        #now generate flux correction from element V's
        #correction on face f = \partial \Omega_l \cap \partial \Omega_{r}
        # \Delta_f = (V_l - V_r)/|\gamma_f|
        #
        #
        #for now proceed through correction step including Neumann boundaries, then overwrite below
        cpostprocessing.computeFluxCorrectionPWC(self.vt.mesh.interiorElementBoundariesArray,
                                                 self.vt.mesh.exteriorElementBoundariesArray,
                                                 self.vt.mesh.elementBoundaryElementsArray,
                                                 self.ebq_global['pwc-corr'],
                                                 self.pwcV[ci],
                                                 self.ebq_global[('flux_correction',ci)])

        cpostprocessing.fluxCorrectionVelocityUpdate(self.vt.mesh.interiorElementBoundariesArray,
                                                   self.vt.mesh.exteriorElementBoundariesArray,
                                                   self.vt.mesh.elementBoundaryElementsArray,
                                                   self.vt.mesh.elementBoundaryLocalElementBoundariesArray,
                                                   self.vt.ebq[('dS_u',ci)],
                                                   self.vt.ebq_global['n'],
                                                   self.ebq_global[('flux_correction',ci)],
                                                   self.ebq_global[('velocity',ci)],
                                                   self.ebq[('velocity',ci)])

        #set boundary flux
        updateCoef = 0.0 #overwrite first
        cfemIntegrals.loadBoundaryFluxIntoGlobalElementBoundaryVelocity(self.vt.mesh.exteriorElementBoundariesArray,
                                                                        self.fluxElementBoundaries[ci],
                                                                        self.vt.ebq_global['n'],
                                                                        self.vt.ebq_global[('totalFlux',ci)],
                                                                        updateCoef,
                                                                        self.vt.ebq_global[('velocity',ci)])


        cfemIntegrals.copyGlobalElementBoundaryVelocityToElementBoundary(self.vt.mesh.interiorElementBoundariesArray,
                                                                         self.vt.mesh.exteriorElementBoundariesArray,
                                                                         self.vt.mesh.elementBoundaryElementsArray,
                                                                         self.vt.mesh.elementBoundaryLocalElementBoundariesArray,
                                                                         self.vt.ebq_global[('velocity',ci)],
                                                                         self.vt.ebq[('velocity',ci)])

        #go from boundary flux to local element boundary representation
        self.evaluateLocalVelocityRepresentation(ci)

        cfemIntegrals.copyExteriorElementBoundaryValuesFromGlobalElementBoundaryValues(self.vt.mesh.exteriorElementBoundariesArray,
                                                                                       self.vt.mesh.elementBoundaryElementsArray,
                                                                                       self.vt.mesh.elementBoundaryLocalElementBoundariesArray,
                                                                                       self.vt.ebq_global[('velocity',ci)],
                                                                                       self.vt.ebqe[('velocity',ci)])
        self.addBoundaryFluxesBackToResidual(ci,self.fluxElementBoundaries[ci])
        if verbose > 0:
            logEvent("Max local conservation (dgp0 enriched) = %12.5e" % max(numpy.absolute(self.q[('conservationResidual',ci)].flat[0:self.vt.mesh.subdomainMesh.nElements_owned])))


    def evaluateLocalVelocityRepresentation(self,ci):
        """project to :math:`RT_0` velocity from element boundary fluxes

        """
        cpostprocessing.projectElementBoundaryVelocityToRT0fluxRep(self.vt.ebq[('dS_u',ci)],
                                                                   self.vt.ebq['n'],
                                                                   self.ebq[('velocity',ci)],
                                                                   self.q[('velocity_dofs',ci)])
        cpostprocessing.getElementRT0velocityValuesFluxRep(self.vt.mesh.nodeArray,
                                                           self.vt.mesh.elementNodesArray,
                                                           self.vt.q['abs(det(J))'],
                                                           self.vt.q['x'],
                                                           self.q[('velocity_dofs',ci)],
                                                           self.q[('velocity',ci)])
    def evaluateElementVelocityField(self,x,ci):
        """
        evaluate velocity field assuming velocity_dofs already calculated
        for now assumes x shaped like nE x nq x 3
        """
        assert len(x.shape) == 3, "wrong shape for x= %s " % x.shape
        nE = x.shape[0]; nq = x.shape[1]; nd = self.vt.nSpace_global
        vx = numpy.zeros((nE,nq,nd),'d')
        cpostprocessing.getElementRT0velocityValuesFluxRep(self.vt.mesh.nodeArray,
                                                           self.vt.mesh.elementNodesArray,
                                                           self.vt.q['abs(det(J))'],
                                                           x,
                                                           self.q[('velocity_dofs',ci)],
                                                           vx)
        return vx


class VPP_SUN_RT0(VelocityPostProcessingAlgorithmBase):
    r"""Global piecewise constant Sun-Wheeler method with :math:`RT_0`
    representation for element velocities

    uses RT_0 representation in terms of local basis

    .. math::

      \vec N_i = \frac{1}{d|E|}(\vec x - p_i)

    where :math:`p_i` is the vertex across from face i, :math:`|E|` is
    the volume of the element, and d is the space dimension the
    degrees of freedom are :math:`V^i = \int_{e_i}\vec v\dot n_{i}\ds`

    """
#    TOOD:
#       make sure works with weak dirichlet conditions
#       remove python loops in building global Jacobian
    def __init__(self,vectorTransport=None,vtComponents=[0]):
        _requireOneRank('sun-rt0')
        VelocityPostProcessingAlgorithmBase.__init__(self,postProcessingType='sun-rt0',
                                                     vectorTransport=vectorTransport,
                                                     vtComponents=vtComponents)

        #how is the local velocity represented
        #  2 -- RT0, local rep is \sum^d_{i=0}V^i\vec N_{T,i},
        #           \vec N_{T,i} = \frac{1}{d|E|}(\vec x - p_i), i=0,...,d
        self.localVelocityRepresentationFlag = 2

        self.nDOFs_element = {}
        #local element residuals
        self.elementResidual = self.vt.elementResidual
        #data structures for solving global problem
        self.sunWheelerMat = {}; self.sunWheelerMat_zval = {}; self.sunWheelerLS = {}; self.sunWheelerV= {}
        self.sunWheelerConnectionList = {} #for GaussSeidel
        globalSolverFlag = 'default' #'Gauss-Seidel', default--> LU

        #could just compute correction factors once since going to overwrite Neumann fluxes anyway
        self.ebq_global['sun-glob-corr'] = numpy.zeros((self.vt.mesh.nElementBoundaries_global,2),'d')
        volFact = 1.0; areaFact = 1.0
        if self.vt.nSpace_global == 2:
            volFact = 0.5
        if self.vt.nSpace_global == 3:
            volFact = 1.0/6.0; areaFact = 0.5

        for ebNI in range(self.vt.mesh.nInteriorElementBoundaries_global):
            ebN = self.vt.mesh.interiorElementBoundariesArray[ebNI]
            eN_left = self.vt.mesh.elementBoundaryElementsArray[ebN,0]
            eN_right= self.vt.mesh.elementBoundaryElementsArray[ebN,1]
            ebN_element_left = self.vt.mesh.elementBoundaryLocalElementBoundariesArray[ebN,0]
            vol_left = volFact*self.vt.q['abs(det(J))'][eN_left,0]#assume affine
            vol_right= volFact*self.vt.q['abs(det(J))'][eN_right,0]#assume affine
            area_face= areaFact*self.vt.ebq['sqrt(det(g))'][eN_left,ebN_element_left,0]
            #this is valid for
            # w_e = -\frac{|\Omega_e|}{|\gamma|_f}\vec n_f \cdot \vec n_e
            self.ebq_global['sun-glob-corr'][ebN,0] = -vol_left/area_face
            self.ebq_global['sun-glob-corr'][ebN,1] =  vol_right/area_face
        for ebNE in range(self.vt.mesh.nExteriorElementBoundaries_global):
            ebN = self.vt.mesh.exteriorElementBoundariesArray[ebNE]
            eN_left = self.vt.mesh.elementBoundaryElementsArray[ebN,0]
            ebN_element_left = self.vt.mesh.elementBoundaryLocalElementBoundariesArray[ebN,0]
            vol_left = volFact*self.vt.q['abs(det(J))'][eN_left,0]#assume affine
            area_face= areaFact*self.vt.ebq['sqrt(det(g))'][eN_left,ebN_element_left,0]
            #this is valid for
            # w_e = -\frac{|\Omega_e|}{|\gamma|_f}\vec n_f \cdot \vec n_e
            self.ebq_global['sun-glob-corr'][ebN,0] = -vol_left/area_face
        #end ebNE

        for ci in self.vtComponents:
            #correction
            self.ebq_global[('flux_correction',ci)] = numpy.zeros((self.vt.mesh.nElementBoundaries_global),'d')
            #velocity representation
            self.nDOFs_element[ci] = self.vt.nSpace_global+1
            self.q[('velocity_dofs',ci)] = numpy.zeros((self.vt.mesh.nElements_global,self.nDOFs_element[ci]),'d')
            if ('velocity_l2g',self.vtComponents[0]) in self.q:
                self.q[('velocity_l2g',ci)]  = self.q[('velocity_l2g',self.vtComponents[0])]
            else:
                self.q[('velocity_l2g',ci)]  = numpy.arange((self.vt.mesh.nElements_global*self.vt.mesh.nElementBoundaries_element),dtype='i').reshape((self.vt.mesh.nElements_global,self.vt.mesh.nElementBoundaries_element))

            #repeat build because of flux boundary condition enforcement
            sunWheelerGlobalDict = {}
            #build system assuming conforming mesh
            for ebNI in range(self.vt.mesh.nInteriorElementBoundaries_global):
                ebN = self.vt.mesh.interiorElementBoundariesArray[ebNI]
                eN_left = self.vt.mesh.elementBoundaryElementsArray[ebN,0]
                eN_right= self.vt.mesh.elementBoundaryElementsArray[ebN,1]
                ebN_element_left = self.vt.mesh.elementBoundaryLocalElementBoundariesArray[ebN,0]
                vol_left = volFact*self.vt.q['abs(det(J))'][eN_left,0]#assume affine
                vol_right= volFact*self.vt.q['abs(det(J))'][eN_right,0]#assume affine
                area_face= areaFact*self.vt.ebq['sqrt(det(g))'][eN_left,ebN_element_left,0]
                #We're solving with right hand side= integrated residual not just residual
                #as in Sun_Wheeler
                #this is valid for
                # w_e = -\frac{|\Omega_e|}{|\gamma|_f}\vec n_f \cdot \vec n_e
                if (eN_left,eN_left) in sunWheelerGlobalDict:
                    sunWheelerGlobalDict[(eN_left,eN_left)] -= vol_left
                else:
                    sunWheelerGlobalDict[(eN_left,eN_left)] = -vol_left
                if (eN_right,eN_right) in sunWheelerGlobalDict:
                    sunWheelerGlobalDict[(eN_right,eN_right)] -= vol_right
                else:
                    sunWheelerGlobalDict[(eN_right,eN_right)]  = -vol_right
                #
                sunWheelerGlobalDict[(eN_left,eN_right)] = 1.*vol_right
                sunWheelerGlobalDict[(eN_right,eN_left)] = 1.*vol_left
            #interior
            #any chance an element didn't get hit by interior loop?
            for ebNE in range(self.vt.mesh.nExteriorElementBoundaries_global):
                ebN = self.vt.mesh.exteriorElementBoundariesArray[ebNE]
                if not self.fluxElementBoundaries[ci][ebNE]:
                    eN_left = self.vt.mesh.elementBoundaryElementsArray[ebN,0]
                    vol_left = volFact*self.vt.q['abs(det(J))'][eN_left,0]#assume affine
                    #this is valid for
                    # w_e = -\frac{|\Omega_e|}{|\gamma|_f}\vec n_f \cdot \vec n_e
                    if (eN_left,eN_left) in sunWheelerGlobalDict:
                        sunWheelerGlobalDict[(eN_left,eN_left)]  -=vol_left
                    else:
                        sunWheelerGlobalDict[(eN_left,eN_left)]   =-vol_left
            #exterior
            (self.sunWheelerMat[ci],self.sunWheelerMat_zval[ci]) = SparseMatFromDict(self.vt.mesh.nElements_global,
                                                                                     self.vt.mesh.nElements_global,
                                                                                     sunWheelerGlobalDict)


            if globalSolverFlag == 'Gauss-Seidel':
                #try to build connection list manually, doesn't really need if
                #sparse mat used? move outside ci loop
                self.sunWheelerConnectionList[ci] = [[] for I in range(mesh.nElements_global)]
                for IJ in list(sunWheelerGlobalDict.keys()):
                    self.sunWheelerConnectionList[ci][IJ[0]].append(IJ[1])

                self.sunWheelerLS[ci] = LinearSolvers.GaussSeidel(connectionList=self.sunWheelerConnectionList[ci],
                                                                  L=self.sunWheelerMat[ci],
                                                                  weight=1.0,
                                                                  rtol_r = 1.0e-6,
                                                                  atol_r = 1.0e-6,
                                                                  rtol_du= 1.0e-6,
                                                                  atol_du= 1.0e-6,
                                                                  maxIts = 1000,
                                                                  printInfo=True)
                self.sunWheelerLS[ci].prepare()
            else:
                self.sunWheelerLS[ci] = LinearSolvers.LU(self.sunWheelerMat[ci])
                self.sunWheelerLS[ci].prepare()
            self.sunWheelerV[ci] = numpy.zeros((self.vt.mesh.nElements_global,),'d')
        #ci


        #
    def postprocess_component(self,ci,verbose=0):
        """compute mass conservative velocity field following Sun and Wheeler
        Global piecewise constant correcction

        """
        #compute initial element conservation residual for average velocity
        self.q[('conservationResidual',ci)].fill(0.0)
        self.ebq_global[('flux_correction',ci)].fill(0.0)
        self.ebq_global[('velocity',ci)].flat[:]=self.vt.ebq_global[('velocityAverage',ci)].flat

        #need to add back in after solves, now skip Neumann boundaries since enforcing them explicitly
        self.removeBoundaryFluxesFromResidual(ci,self.fluxElementBoundaries[ci])


        cpostprocessing.calculateConservationResidualGlobalBoundaries(self.vt.mesh.interiorElementBoundariesArray,
                                                                      self.vt.mesh.exteriorElementBoundariesArray,
                                                                      self.vt.mesh.elementBoundaryElementsArray,
                                                                      self.vt.mesh.elementBoundaryLocalElementBoundariesArray,
                                                                      self.fluxElementBoundaries[ci],
                                                                      self.vt.ebq[('dS_u',ci)],
                                                                      self.vt.ebq_global['n'],
                                                                      self.elementResidual[ci],
                                                                      self.ebq_global[('velocity',ci)],
                                                                      self.q[('conservationResidual',ci)])

        self.sunWheelerLS[ci].solve(self.sunWheelerV[ci],b=self.q[('conservationResidual',ci)])
        #now generate flux correction from element V's
        #basis : on \gamma_f
        #  w_e = -\frac{|\Omega_e|}{|\gamma_f|}\vec n_f \cdot \vec n_e
        #
        #so correction on face f = \partial \Omega_l \cap \partial \Omega_{r}
        # \Delta_f = V_l w_l + V_{r} w_{r}
        #          = (|\Omega_r|V_r + |\Omega_l|V_l)/|\gamma_f|
        #
        #
        #overwrite Neumann flux boundaries below
        cpostprocessing.computeFluxCorrectionPWC(self.vt.mesh.interiorElementBoundariesArray,
                                                 self.vt.mesh.exteriorElementBoundariesArray,
                                                 self.vt.mesh.elementBoundaryElementsArray,
                                                 self.ebq_global['sun-glob-corr'],
                                                 self.sunWheelerV[ci],
                                                 self.ebq_global[('flux_correction',ci)])
        cpostprocessing.fluxCorrectionVelocityUpdate(self.vt.mesh.interiorElementBoundariesArray,
                                                     self.vt.mesh.exteriorElementBoundariesArray,
                                                     self.vt.mesh.elementBoundaryElementsArray,
                                                     self.vt.mesh.elementBoundaryLocalElementBoundariesArray,
                                                     self.vt.ebq[('dS_u',ci)],
                                                     self.vt.ebq_global['n'],
                                                     self.ebq_global[('flux_correction',ci)],
                                                     self.ebq_global[('velocity',ci)],
                                                     self.ebq[('velocity',ci)])

        #set boundary flux
        updateCoef = 0.0 #overwrite first
        cfemIntegrals.loadBoundaryFluxIntoGlobalElementBoundaryVelocity(self.vt.mesh.exteriorElementBoundariesArray,
                                                                        self.fluxElementBoundaries[ci],
                                                                        self.vt.ebq_global['n'],
                                                                        self.vt.ebq_global[('totalFlux',ci)],
                                                                        updateCoef,
                                                                        self.vt.ebq_global[('velocity',ci)])


        cfemIntegrals.copyGlobalElementBoundaryVelocityToElementBoundary(self.vt.mesh.interiorElementBoundariesArray,
                                                                         self.vt.mesh.exteriorElementBoundariesArray,
                                                                         self.vt.mesh.elementBoundaryElementsArray,
                                                                         self.vt.mesh.elementBoundaryLocalElementBoundariesArray,
                                                                         self.vt.ebq_global[('velocity',ci)],
                                                                         self.vt.ebq[('velocity',ci)])

        #go from boundary flux to local element boundary representation
        self.evaluateLocalVelocityRepresentation(ci)

        cfemIntegrals.copyExteriorElementBoundaryValuesFromGlobalElementBoundaryValues(self.vt.mesh.exteriorElementBoundariesArray,
                                                                                       self.vt.mesh.elementBoundaryElementsArray,
                                                                                       self.vt.mesh.elementBoundaryLocalElementBoundariesArray,
                                                                                       self.vt.ebq_global[('velocity',ci)],
                                                                                       self.vt.ebqe[('velocity',ci)])
        self.addBoundaryFluxesBackToResidual(ci,self.fluxElementBoundaries[ci])
        if verbose > 0:
            logEvent("Max local conservation (Sun global enriched) = %12.5e" % max(numpy.absolute(self.q[('conservationResidual',ci)].flat[0:self.vt.mesh.subdomainMesh.nElements_owned])))


    def evaluateLocalVelocityRepresentation(self,ci):
        """project to :math:`RT_0` velocity from element boundary fluxes

        """
        cpostprocessing.projectElementBoundaryVelocityToRT0fluxRep(self.vt.ebq[('dS_u',ci)],
                                                                   self.vt.ebq['n'],
                                                                   self.ebq[('velocity',ci)],
                                                                   self.q[('velocity_dofs',ci)])
        cpostprocessing.getElementRT0velocityValuesFluxRep(self.vt.mesh.nodeArray,
                                                           self.vt.mesh.elementNodesArray,
                                                           self.vt.q['abs(det(J))'],
                                                           self.vt.q['x'],
                                                           self.q[('velocity_dofs',ci)],
                                                           self.q[('velocity',ci)])
    def evaluateElementVelocityField(self,x,ci):
        """evaluate velocity field assuming velocity_dofs already calculated
        for now assumes x shaped like nE x nq x 3

        """
        assert len(x.shape) == 3, "wrong shape for x= %s " % x.shape
        nE = x.shape[0]; nq = x.shape[1]; nd = self.vt.nSpace_global
        vx = numpy.zeros((nE,nq,nd),'d')
        cpostprocessing.getElementRT0velocityValuesFluxRep(self.vt.mesh.nodeArray,
                                                           self.vt.mesh.elementNodesArray,
                                                           self.vt.q['abs(det(J))'],
                                                           x,
                                                           self.q[('velocity_dofs',ci)],
                                                           vx)
        return vx


class VPP_SUN_GS_RT0(VelocityPostProcessingAlgorithmBase):
    r"""Local Sun-Wheeler "Gauss-Seidel" method with :math:`RT_0` representation
    for element velocities

    uses :math:`RT_0` representation in terms of local basis

    .. math::

      \vec N_i = \frac{1}{d|E|}(\vec x - p_i)

    where p_i is the vertex across from face i, :math:`|E|` is the
    volume of the element, and d is the space dimension the degrees of
    freedom are :math:`V^i = \int_{e_i}\vec v\dot n_{i}\ds`

    """
    def __init__(self,vectorTransport=None,vtComponents=[0]):
        _requireOneRank('sun-gs-rt0')
        VelocityPostProcessingAlgorithmBase.__init__(self,postProcessingType='sun-gs-rt0',
                                                     vectorTransport=vectorTransport,
                                                     vtComponents=vtComponents)

        #how is the local velocity represented
        #  2 -- RT0, local rep is \sum^d_{i=0}V^i\vec N_{T,i},
        #           \vec N_{T,i} = \frac{1}{d|E|}(\vec x - p_i), i=0,...,d
        self.localVelocityRepresentationFlag = 2

        #need to allow these as parameters
        self.GSmaxIts = 10000
        self.GStol    = 1.0e-6
        self.nDOFs_element = {}
        #local element residuals
        self.elementResidual = self.vt.elementResidual
        #make some minor changes to Sun-Wheeler formalation because our residual is already integrated
        self.ebq_global['sun-gs-alpha'] = numpy.zeros((self.vt.mesh.nElementBoundaries_global,2),'d')

        volFact = 1.0; areaFact = 1.0
        if self.vt.nSpace_global == 2:
            volFact = 0.5
        if self.vt.nSpace_global == 3:
            volFact = 1.0/6.0; areaFact = 0.5
        for ebNI in range(self.vt.mesh.nInteriorElementBoundaries_global):
            ebN = self.vt.mesh.interiorElementBoundariesArray[ebNI]
            eN_left = self.vt.mesh.elementBoundaryElementsArray[ebN,0]
            eN_right= self.vt.mesh.elementBoundaryElementsArray[ebN,1]
            ebN_element_left = self.vt.mesh.elementBoundaryLocalElementBoundariesArray[ebN,0]
            vol_left = volFact*self.vt.q['abs(det(J))'][eN_left,0]#assume affine
            vol_right= volFact*self.vt.q['abs(det(J))'][eN_right,0]#assume affine
            area_face= areaFact*self.vt.ebq['sqrt(det(g))'][eN_left,ebN_element_left,0]
            #if not using integral of residual?
            #weighted harmonic average, should be what Sun-Wheeler use but doesn't seem to work as well as
            # the weighted arithmetic average does
            #note signs are opposite of our paper formulation
            self.ebq_global['sun-gs-alpha'][ebN,0]=  vol_right/(area_face*(vol_left+vol_right))
            self.ebq_global['sun-gs-alpha'][ebN,1]= -vol_left/(area_face*(vol_left+vol_right))
            #weighted arithmetic average basically
            #self.ebq_global['sun-gs-alpha'][ebN,0]=  vol_left/(area_face*(vol_left+vol_right))
            #self.ebq_global['sun-gs-alpha'][ebN,1]= -vol_right/(area_face*(vol_left+vol_right))

        #ebNI
        for ci in self.vtComponents:
            #correction
            self.ebq_global[('flux_correction',ci)] = numpy.zeros((self.vt.mesh.nElementBoundaries_global),'d')
            #velocity representation
            self.nDOFs_element[ci] = self.vt.nSpace_global+1
            self.q[('velocity_dofs',ci)] = numpy.zeros((self.vt.mesh.nElements_global,self.nDOFs_element[ci]),'d')
            if ('velocity_l2g',self.vtComponents[0]) in self.q:
                self.q[('velocity_l2g',ci)]  = self.q[('velocity_l2g',self.vtComponents[0])]
            else:
                self.q[('velocity_l2g',ci)]  = numpy.arange((self.vt.mesh.nElements_global*self.vt.mesh.nElementBoundaries_element),dtype='i').reshape((self.vt.mesh.nElements_global,self.vt.mesh.nElementBoundaries_element))

    def postprocess_component(self,ci,verbose=0):
        """compute mass conservative velocity field following Sun and Wheeler
        local Gauss-Seidel

        """
        #compute initial element conservation residual for average velocity
        self.q[('conservationResidual',ci)].fill(0.0)
        self.ebq_global[('flux_correction',ci)].fill(0.0)
        self.ebq_global[('velocity',ci)].flat[:]=self.vt.ebq_global[('velocityAverage',ci)].flat

        #need to add back in after solves, now skip Neumann boundaries since enforcing them explicitly
        if self.vt.numericalFlux is not None:#In general need numericalFlux since exterior boundary velocity assumed correct
            #have to set boundary flux into velocity data structure from advective and diffusive flux arrays
            #in case transport didn't to this?
            for ebNE in range(self.vt.mesh.nExteriorElementBoundaries_global):
                ebN = self.vt.mesh.exteriorElementBoundariesArray[ebNE]
                for k in range(self.vt.nElementBoundaryQuadraturePoints_elementBoundary):
                    vdotn = numpy.dot(self.ebq_global[('velocity',ci)][ebN,k,:],
                                      self.vt.ebq_global['n'][ebN,k,:])
                    self.ebq_global[('velocity',ci)][ebN,k,:] += (self.vt.ebq_global[('totalFlux',ci)][ebN,k]-vdotn)*self.vt.ebq_global['n'][ebN,k,:]

        self.removeBoundaryFluxesFromResidual(ci,self.fluxElementBoundaries[ci])

        cpostprocessing.calculateConservationResidualGlobalBoundaries(self.vt.mesh.interiorElementBoundariesArray,
                                                                      self.vt.mesh.exteriorElementBoundariesArray,
                                                                      self.vt.mesh.elementBoundaryElementsArray,
                                                                      self.vt.mesh.elementBoundaryLocalElementBoundariesArray,
                                                                      self.fluxElementBoundaries[ci],
                                                                      self.vt.ebq[('dS_u',ci)],
                                                                      self.vt.ebq_global['n'],
                                                                      self.elementResidual[ci],
                                                                      self.ebq_global[('velocity',ci)],
                                                                      self.q[('conservationResidual',ci)])


        converged = False
        GSits = 0
        if verbose > 0:
            logEvent("Sun-Wheeler GS initial residual= %s" % max(abs(self.q[('conservationResidual',ci)].flat[:])))

        while GSits < self.GSmaxIts and not converged:
            cpostprocessing.sunWheelerGSsweep(self.vt.mesh.interiorElementBoundariesArray,
                                              self.vt.mesh.exteriorElementBoundariesArray,
                                              self.vt.mesh.elementBoundaryElementsArray,
                                              self.vt.mesh.elementBoundaryLocalElementBoundariesArray,
                                              self.vt.ebq[('dS_u',ci)],
                                              self.vt.ebq_global['n'],
                                              self.vt.ebq['sqrt(det(g))'],
                                              self.ebq_global['sun-gs-alpha'],
                                              self.ebq_global[('flux_correction',ci)],
                                              self.q[('conservationResidual',ci)])
            maxRes = numpy.max(numpy.absolute(self.q[('conservationResidual',ci)].flat))
            converged = maxRes < self.GStol
            GSits += 1
            if verbose > 0:
                logEvent("SunWheelerSweep %d maxRes=%s " % (GSits,maxRes))
        #end while

        cpostprocessing.fluxCorrectionVelocityUpdate(self.vt.mesh.interiorElementBoundariesArray,
                                                     self.vt.mesh.exteriorElementBoundariesArray,
                                                     self.vt.mesh.elementBoundaryElementsArray,
                                                     self.vt.mesh.elementBoundaryLocalElementBoundariesArray,
                                                     self.vt.ebq[('dS_u',ci)],
                                                     self.vt.ebq_global['n'],
                                                     self.ebq_global[('flux_correction',ci)],
                                                     self.ebq_global[('velocity',ci)],
                                                     self.ebq[('velocity',ci)])


        #go from boundary flux to local element boundary representation
        self.evaluateLocalVelocityRepresentation(ci)

        cfemIntegrals.copyExteriorElementBoundaryValuesFromGlobalElementBoundaryValues(self.vt.mesh.exteriorElementBoundariesArray,
                                                                                       self.vt.mesh.elementBoundaryElementsArray,
                                                                                       self.vt.mesh.elementBoundaryLocalElementBoundariesArray,
                                                                                       self.vt.ebq_global[('velocity',ci)],
                                                                                       self.vt.ebqe[('velocity',ci)])
        self.addBoundaryFluxesBackToResidual(ci,self.fluxElementBoundaries[ci])
        if verbose > 0:
            logEvent("Max local conservation (Sun global enriched) = %12.5e" % max(numpy.absolute(self.q[('conservationResidual',ci)].flat[0:self.vt.mesh.subdomainMesh.nElements_owned])))


    def evaluateLocalVelocityRepresentation(self,ci):
        """project to :math:`RT_0` velocity from element boundary fluxes

        """
        cpostprocessing.projectElementBoundaryVelocityToRT0fluxRep(self.vt.ebq[('dS_u',ci)],
                                                                   self.vt.ebq['n'],
                                                                   self.ebq[('velocity',ci)],
                                                                   self.q[('velocity_dofs',ci)])
        cpostprocessing.getElementRT0velocityValuesFluxRep(self.vt.mesh.nodeArray,
                                                           self.vt.mesh.elementNodesArray,
                                                           self.vt.q['abs(det(J))'],
                                                           self.vt.q['x'],
                                                           self.q[('velocity_dofs',ci)],
                                                           self.q[('velocity',ci)])
    def evaluateElementVelocityField(self,x,ci):
        """
        evaluate velocity field assuming velocity_dofs already calculated
        for now assumes x shaped like nE x nq x 3
        """
        assert len(x.shape) == 3, "wrong shape for x= %s " % x.shape
        nE = x.shape[0]; nq = x.shape[1]; nd = self.vt.nSpace_global
        vx = numpy.zeros((nE,nq,nd),'d')
        cpostprocessing.getElementRT0velocityValuesFluxRep(self.vt.mesh.nodeArray,
                                                           self.vt.mesh.elementNodesArray,
                                                           self.vt.q['abs(det(J))'],
                                                           x,
                                                           self.q[('velocity_dofs',ci)],
                                                           vx)
        return vx


class VPP_DG_RT0(VelocityPostProcessingAlgorithmBase):
    """Post process DG solution velocity by projecting boundary fluxes
    onto local :math:`RT_0` space

    """
    def __init__(self,vectorTransport=None,vtComponents=[0]):
        VelocityPostProcessingAlgorithmBase.__init__(self,postProcessingType='dg',
                                                     vectorTransport=vectorTransport,
                                                     vtComponents=vtComponents)

        #how is the local velocity represented
        #  2 -- RT0, local rep is \sum^d_{i=0}V^i\vec N_{T,i},
        #           \vec N_{T,i} = \frac{1}{d|E|}(\vec x - p_i), i=0,...,d
        self.localVelocityRepresentationFlag = 2
        self.nDOFs_element = {}
        #local element residuals
        self.elementResidual = self.vt.elementResidual
        for ci in self.vtComponents:
            #for RT0
            self.nDOFs_element[ci] = self.vt.nSpace_global+1
            self.q[('velocity_dofs',ci)] = numpy.zeros((self.vt.mesh.nElements_global,self.nDOFs_element[ci]),'d')
            if ('velocity_l2g',self.vtComponents[0]) in self.q:
                self.q[('velocity_l2g',ci)]  = self.q[('velocity_l2g',self.vtComponents[0])]
            else:
                self.q[('velocity_l2g',ci)]  = numpy.arange((self.vt.mesh.nElements_global*self.vt.mesh.nElementBoundaries_element),dtype='i').reshape((self.vt.mesh.nElements_global,self.vt.mesh.nElementBoundaries_element))
        #
    def postprocess_component(self,ci,verbose=0):
        """
        """
        self.evaluateLocalVelocityRepresentation(ci)
    def evaluateLocalVelocityRepresentation(self,ci):
        """
        project to :math:`RT_0` velocity from element boundary fluxes

        """
#        TODO:
#          Check if need to evaluate element boundary and global element boundary velocities (and exterior global element boundary  too?)
#          or if they are already evaluated with DG fluxes

        cpostprocessing.projectElementBoundaryFluxToRT0fluxRep(self.vt.mesh.elementBoundaryElementsArray,
                                                               self.vt.mesh.elementBoundariesArray,
                                                               self.vt.ebq[('dS_u',ci)],
                                                               self.vt.ebq_global[('totalFlux',ci)],
                                                               self.q[('velocity_dofs',ci)])

        cpostprocessing.getElementRT0velocityValuesFluxRep(self.vt.mesh.nodeArray,
                                                           self.vt.mesh.elementNodesArray,
                                                           self.vt.q['abs(det(J))'],
                                                           self.vt.q['x'],
                                                           self.q[('velocity_dofs',ci)],
                                                           self.q[('velocity',ci)])

        cpostprocessing.getElementBoundaryRT0velocityValuesFluxRep(self.vt.mesh.nodeArray,
                                                                   self.vt.mesh.elementNodesArray,
                                                                   self.vt.q['abs(det(J))'],
                                                                   self.vt.ebq['x'],
                                                                   self.q[('velocity_dofs',ci)],
                                                                   self.ebq[('velocity',ci)])

        cpostprocessing.getGlobalElementBoundaryRT0velocityValuesFluxRep(self.vt.mesh.nodeArray,
                                                                         self.vt.mesh.elementNodesArray,
                                                                         self.vt.mesh.elementBoundaryElementsArray,
                                                                         self.vt.q['abs(det(J))'],
                                                                         self.vt.ebq_global['x'],
                                                                         self.q[('velocity_dofs',ci)],
                                                                         self.ebq_global[('velocity',ci)])

    def evaluateElementVelocityField(self,x,ci):
        """
        evaluate velocity field assuming velocity_dofs already calculated
        for now assumes x shaped like nE x nq x 3
        """
        assert len(x.shape) == 3, "wrong shape for x= %s " % x.shape
        nE = x.shape[0]; nq = x.shape[1]; nd = self.vt.nSpace_global
        vx = numpy.zeros((nE,nq,nd),'d')
        cpostprocessing.getElementRT0velocityValuesFluxRep(self.vt.mesh.nodeArray,
                                                           self.vt.mesh.elementNodesArray,
                                                           self.vt.q['abs(det(J))'],
                                                           x,
                                                           self.q[('velocity_dofs',ci)],
                                                           vx)
        return vx

class VPP_DG_BDM(VPP_DG_RT0):
    r"""Project DG solution onto BDM1 local representation which is
    :math:`[P^1]^d` locally with continuous linear fluxes on each face

    use standard basis

    .. math::

      \vec N_i = \lambda_{i \bmod (d+1)} \vec e_{i/(d+1)}

    Have to use BDM projection to get degrees of freedom

    """
    def __init__(self,vectorTransport=None,vtComponents=[0]):
        VPP_DG_RT0.__init__(self,vectorTransport=vectorTransport,vtComponents=vtComponents)
        #how is the local velocity represented
        #  0 -- BDM P^1 Lagrange rep
        self.localVelocityRepresentationFlag = 0
        #have to directly modify the type now to show bdm
        self.postProcessingType = 'dg-bdm'
        #
        self.testSpace = None
        for ci in self.vtComponents:
            self.nDOFs_element[ci] = self.vt.nSpace_global*(self.vt.nSpace_global+1)
            self.q[('velocity_dofs',ci)] = numpy.zeros((self.vt.mesh.nElements_global,self.nDOFs_element[ci]),'d')
            if ('velocity_l2g',self.vtComponents[0]) in self.q:
                self.q[('velocity_l2g',ci)]  = self.q[('velocity_l2g',self.vtComponents[0])]
            else:
                self.q[('velocity_l2g',ci)]  = numpy.arange((self.vt.mesh.nElements_global*self.nDOFs_element[ci]),dtype='i').reshape((self.vt.mesh.nElements_global,self.nDOFs_element[ci]))
            #build linear test space for BMD projection if test space isn't P1

            if not isinstance(self.vt.u[ci].femSpace,FemTools.DG_AffineLinearOnSimplexWithNodalBasis):
                if self.testSpace is None:
                    self.testSpace = FemTools.C0_AffineLinearOnSimplexWithNodalBasis(self.vt.mesh,self.vt.nSpace_global)
                assert isinstance(self.testSpace,FemTools.C0_AffineLinearOnSimplexWithNodalBasis)
                self.qv[ci] = numpy.zeros(
                    (self.vt.mesh.nElements_global,
                     self.vt.nQuadraturePoints_element,
                     self.testSpace.referenceFiniteElement.localFunctionSpace.dim),
                    'd')
                self.w[ci] = numpy.zeros(
                    (self.vt.mesh.nElements_global,
                     self.vt.mesh.nElementBoundaries_element,
                     self.vt.nElementBoundaryQuadraturePoints_elementBoundary,
                     self.testSpace.referenceFiniteElement.localFunctionSpace.dim),
                    'd')
                self.w_dS[ci] = numpy.zeros(
                    (self.vt.mesh.nElements_global,
                     self.vt.mesh.nElementBoundaries_element,
                     self.vt.nElementBoundaryQuadraturePoints_elementBoundary,
                     self.testSpace.referenceFiniteElement.localFunctionSpace.dim),
                    'd')

                self.testSpace.getBasisValues(self.vt.elementQuadraturePoints,
                                              self.qv[ci])
                self.testSpace.getBasisValuesTrace(self.vt.u[0].femSpace.elementMaps.permutations,
                                                   self.vt.ebq['hat(x)'],
                                                   self.w[ci])
                cfemIntegrals.calculateWeightedShapeTrace(self.vt.elementBoundaryQuadratureWeights[('u',ci)],
                                                          self.vt.ebq['sqrt(det(g))'],
                                                          self.w[ci],
                                                          self.w_dS[ci])

        if self.testSpace is None:
            #all the solution spaces must be C0P1
            self.testSpace = self.vt.u[self.vtComponents[0]].femSpace
        self.BDMcomponent=self.vtComponents[0]

        self.BDMprojectionMat_element = numpy.zeros((self.vt.mesh.nElements_global,
                                                     self.nDOFs_element[self.BDMcomponent],
                                                     self.nDOFs_element[self.BDMcomponent]),
                                                    'd')
        self.BDMprojectionMatPivots_element = numpy.zeros((self.vt.mesh.nElements_global,
                                                           self.nDOFs_element[self.BDMcomponent]),
                                                          'i')
        self.computeBDM1projectionMatrices()

    def computeBDM1projectionMatrices(self):
        cpostprocessing.buildLocalBDM1projectionMatrices(self.w_dS[self.BDMcomponent],#vt.ebq[('w*dS_u',self.BDMcomponent)],
                                                         self.vt.ebq['n'],
                                                         self.w[self.BDMcomponent],#self.vt.ebq[('v',self.BDMcomponent)],
                                                         self.BDMprojectionMat_element)
        cpostprocessing.factorLocalBDM1projectionMatrices(self.BDMprojectionMat_element,
                                                          self.BDMprojectionMatPivots_element)
    def computeGeometricInfo(self):
        if self.BDMcomponent is not None:
            self.computeBDM1projectionMatrices()

    def evaluateLocalVelocityRepresentation(self,ci):
        """project to BDM velocity from element boundary fluxes

        """
#TODO:
#          evaluate element boundary and global element boundary velocities (and exterior global element boundary  too?)
#          Check that they aren't already evaluated with DG fluxes
        assert self.nDOFs_element[ci] == self.vt.nSpace_global*(self.vt.nSpace_global+1), "wrong size for BDM"
        cpostprocessing.solveLocalBDM1projectionFromFlux(self.BDMprojectionMat_element,
                                                         self.BDMprojectionMatPivots_element,
                                                         self.vt.mesh.elementBoundaryElementsArray,
                                                         self.vt.mesh.elementBoundariesArray,
                                                         self.w_dS[ci],
                                                         self.ebq_global[('totalFlux',ci)],
                                                         self.q[('velocity_dofs',ci)])

        cpostprocessing.getElementBDM1velocityValuesLagrangeRep(self.qv[ci],
                                                                self.q[('velocity_dofs',ci)],
                                                                self.vt.q[('velocity',ci)])

        # the BDM1 representation is in the linear basis, so evaluate it with the P1 test space's traces (self.w), not
        # the solution space's (quadratic for DG P2)
        cpostprocessing.getElementBoundaryBDM1velocityValuesLagrangeRep(self.vt.mesh.elementBoundaryElementsArray,
                                                                        self.vt.mesh.exteriorElementBoundariesArray,
                                                                        self.w[ci],
                                                                        self.q[('velocity_dofs',ci)],
                                                                        self.vt.ebq[('velocity',ci)])
        # every face takes its left element's trace (ebq_global's points are in the left element's order); the normal
        # component is the projected numerical flux, the same from either side
        mesh = self.vt.mesh
        self.ebq_global[('velocity',ci)][:] = self.ebq[('velocity',ci)][mesh.elementBoundaryElementsArray[:,0],
                                                                       mesh.elementBoundaryLocalElementBoundariesArray[:,0]]
        if ('velocity',ci) in self.ebqe:
            cfemIntegrals.copyExteriorElementBoundaryValuesFromGlobalElementBoundaryValues(mesh.exteriorElementBoundariesArray,
                                                                                           mesh.elementBoundaryElementsArray,
                                                                                           mesh.elementBoundaryLocalElementBoundariesArray,
                                                                                           self.ebq_global[('velocity',ci)],
                                                                                           self.ebqe[('velocity',ci)])




    def evaluateElementVelocityField(self,x,ci):
        """evaluate velocity field assuming velocity_dofs already calculated
        for now assumes x shaped like nE x nq x 3 

        """
#        TODO: put python loops in c
        assert len(x.shape) == 3, "wrong shape for x= %s " % x.shape
        nE = x.shape[0]; nq = x.shape[1]; nd = self.vt.nSpace_global
        vx = numpy.zeros((nE,nq,nd),'d')
        #have to evaluate shape functions at new points. This is painful
        #uci     = self.vt.u[ci]
        xiArray = numpy.zeros(x.shape,'d')
        vArray  = numpy.zeros((nE,nq,self.testSpace.max_nDOF_element),'d')
        invJ    = numpy.zeros((nE,nq,self.vt.q['inverse(J)'].shape[2],self.vt.q['inverse(J)'].shape[3]),
                                'd')
        for ie in range(nE):
            for iq in range(nq):
                invJ[ie,iq,:,:] = self.vt.q['inverse(J)'][ie,0,:,:] #assume affine
            #iq
        #iq
        self.testSpace.elementMaps.getInverseValues(invJ,x,xiArray)
        self.testSpace.getBasisValuesAtArray(xiArray,vArray)

        cpostprocessing.getElementBDM1velocityValuesLagrangeRep(vArray,
                                                                self.q[('velocity_dofs',ci)],
                                                                vx)

        return vx

class VPP_POINT_EVAL(VelocityPostProcessingAlgorithmBase):
    """
    Velocity calculation from just directly evaluating flux formula at intergration points using
    trial solution
    """
    def __init__(self,vectorTransport=None,vtComponents=[0]):
        VelocityPostProcessingAlgorithmBase.__init__(self,postProcessingType='point-eval',
                                                     vectorTransport=vectorTransport,
                                                     vtComponents=vtComponents)

    def postprocess_component(self,ci,verbose=0):
        r"""compute velocity field by just using point evaluation of trial
        solution and coefficients

        .. math::

          v = -\ten{a}_h\grad \phi_h + \vec f_h

        """
#         TODO
#           Include off diagonal potentials
        self.q[('velocity',ci)].fill(0.0)
        if ('a',ci,ci) in self.vt.q:
            assert ('grad(phi)',ci) in self.vt.q
            updateCoef = 0.0 #overwrite first time
            if self.vt.sd:
                cpostprocessing.updateDiffusiveVelocityPointEval_sd(updateCoef,
                                                                    self.vt.coefficients.sdInfo[(ci,ci)][0],self.vt.coefficients.sdInfo[(ci,ci)][1],
                                                                    self.vt.q[('a',ci,ci)],
                                                                    self.vt.q[('grad(phi)',ci)],
                                                                    self.q[('velocity'),ci])
            else:
                cpostprocessing.updateDiffusiveVelocityPointEval(updateCoef,
                                                              self.vt.q[('a',ci,ci)],
                                                              self.vt.q[('grad(phi)',ci)],
                                                              self.q[('velocity'),ci])
        if ('f',ci) in self.vt.q:
            updateCoef = 1.0
            cpostprocessing.updateAdvectiveVelocityPointEval(updateCoef,
                                                             self.vt.q[('f',ci)],
                                                             self.q[('velocity'),ci])
        #
        if ('a',ci,ci) in self.vt.ebq:
            assert ('grad(phi)',ci) in self.vt.ebq
            updateCoef = 0.0 #overwrite first time
            if self.vt.sd:
                cpostprocessing.updateDiffusiveVelocityPointEval_sd(updateCoef,
                                                                    self.vt.coefficients.sdInfo[(ci,ci)][0],self.vt.coefficients.sdInfo[(ci,ci)][1],
                                                                    self.vt.ebq[('a',ci,ci)],
                                                                    self.vt.ebq[('grad(phi)',ci)],
                                                                    self.ebq[('velocity'),ci])
            else:
                cpostprocessing.updateDiffusiveVelocityPointEval(updateCoef,
                                                                 self.vt.ebq[('a',ci,ci)],
                                                                 self.vt.ebq[('grad(phi)',ci)],
                                                                 self.ebq[('velocity'),ci])
        if ('f',ci) in self.vt.ebq:
            updateCoef = 1.0
            cpostprocessing.updateAdvectiveVelocityPointEval(updateCoef,
                                                             self.vt.ebq[('f',ci)],
                                                             self.ebq[('velocity'),ci])

        if ('a',ci,ci) in self.vt.ebqe:
            assert ('grad(phi)',ci) in self.vt.ebqe
            updateCoef = 0.0 #overwrite first time
            if self.vt.sd:
                cpostprocessing.updateDiffusiveVelocityPointEval_sd(updateCoef,
                                                                    self.vt.coefficients.sdInfo[(ci,ci)][0],self.vt.coefficients.sdInfo[(ci,ci)][1],
                                                                    self.vt.ebqe[('a',ci,ci)],
                                                                    self.vt.ebqe[('grad(phi)',ci)],
                                                                    self.ebqe[('velocity'),ci])
            else:
                cpostprocessing.updateDiffusiveVelocityPointEval(updateCoef,
                                                             self.vt.ebqe[('a',ci,ci)],
                                                             self.vt.ebqe[('grad(phi)',ci)],
                                                             self.ebqe[('velocity'),ci])
        if ('f',ci) in self.vt.ebqe:
            updateCoef = 1.0
            cpostprocessing.updateAdvectiveVelocityPointEval(updateCoef,
                                                             self.vt.ebqe[('f',ci)],
                                                             self.ebqe[('velocity'),ci])
    def evaluateElementVelocityField(self,x,ci):
        """
        evaluate velocity field assuming velocity_dofs already calculated
        for now assumes x shaped like nE x nq x 3

        TODO
          Decide if should fail or throw exception here
        """
        assert len(x.shape) == 3, "wrong shape for x= %s " % x.shape
        nE = x.shape[0]; nq = x.shape[1]; nd = self.vt.nSpace_global
        vx = numpy.zeros((nE,nq,nd),'d')

        logEvent("""WARNING evalute ElementVelocity Field point-eval not implemented """)
        #
        return vx

#########################
#  GWVD Post Processing Classes; gwvd addin tjp
#########################
class VPP_POINT_EVAL_GWVD(VelocityPostProcessingAlgorithmBase):
    """
    Velocity calculation for evaluating coarse grid velocity solution
    at fine grid quadrature points; assumes that the coarse velocity degrees of
    freedom from the LDG solution are known

    Currently only working for P=1 and P=2
    """
    from .cpostprocessing import getElementLDGvelocityValuesLagrangeRep
    def __init__(self,vectorTransport=None,vtComponents=[0]):
        VelocityPostProcessingAlgorithmBase.__init__(self,postProcessingType='point-eval-gwvd',
                                                     vectorTransport=vectorTransport,
                                                     vtComponents=vtComponents)
        #how is the local velocity represented, 1=P1 Lagrange, 2=P2 Lagrange

        self.testSpace=None
        for ci in self.vtComponents:
            self.testSpace=self.vt.u[ci].femSpace



    def postprocess_component(self,ci,verbose=0):
        """
       may need to compute velocity field at the desired velocity dofs if this step is not
       included in the getResidual step of the solution
        """
        pass


    def evaluateElementVelocityField(self,x,ci):
        """
        evaluate velocity field assuming velocity_dofs already calculated
        for now assumes x shaped like nE x nq x 3
        TODO:
           put python loops in c
        """
        assert len(x.shape) == 3, "wrong shape for x= %s " % x.shape
        nE = x.shape[0]; nq = x.shape[1]; nd = self.vt.nSpace_global
        vx = numpy.zeros((nE,nq,nd),'d')
        #have to evaluate shape functions at new points. This is painful
        xiArray = numpy.zeros(x.shape,'d')
        vArray  = numpy.zeros((nE,nq,self.testSpace.max_nDOF_element),'d')
        invJ    = numpy.zeros((nE,nq,self.vt.q['inverse(J)'].shape[2],self.vt.q['inverse(J)'].shape[3]),
                                'd')
        for ie in range(nE):
            for iq in range(nq):
                invJ[ie,iq,:,:] = self.vt.q['inverse(J)'][ie,0,:,:] #assume affine
            #iq
        #iq

        self.testSpace.elementMaps.getInverseValues(invJ,x,xiArray)
        self.testSpace.getBasisValuesAtArray(xiArray,vArray)

        cpostprocessing.getElementLDGvelocityValuesLagrangeRep(vArray,
                                                                self.q[('velocity_dofs',ci)],
                                                                vx)

        return vx

    def archiveVelocityValues(self,archive,t,tCount,initialPhase=False,meshChanged=False):
        pass

#########################################################################
#Begin trying "fixes" for material interface jumps
class VPP_LOW_K_IB_PWL_RT0(VelocityPostProcessingAlgorithmBase):
    """Local Larson-Niklasson method with RT_0 representation for element
    velocities This version tries to address flux through low
    permeability interfaces by manually setting boundaries with a
    'large' jump in permeability and one of these boundaries being
    'small' to no flux

    """
#    TODO:
#      get tolerances from coefficients?
#      remove python loops in building mesh infrastructure or get rid of dependence
#        on these forms of node star info and use data structures in cmesh
    def __init__(self,vectorTransport=None,vtComponents=[0],jump_tol=1.0e4,min_tol=1.0e-5):
        VelocityPostProcessingAlgorithmBase.__init__(self,postProcessingType='pwl-ib-fix-0',
                                                     vectorTransport=vectorTransport,
                                                     vtComponents=vtComponents)
        #how is the local velocity represented
        #  2 -- RT0, local rep is \sum^d_{i=0}V^i\vec N_{T,i},
        #           \vec N_{T,i} = \frac{1}{d|E|}(\vec x - p_i), i=0,...,d
        self.localVelocityRepresentationFlag = 2
        #
        self.jump_tol = jump_tol; self.min_tol = min_tol

        #to handle some spaces being P^k k > 1
        self.alpha = {}
        self.solutionTestSpaceIsNotPWL = {}
        self.elementResidual = {}
        self.testSpace = None
        #mesh information needed
        #building mesh infrastructure
        self.globalNode2globalElementList = [[] for nN in range(self.vt.mesh.nNodes_global)]
        for eN in range(self.vt.mesh.nElements_global):
            for nN in range(self.vt.mesh.nNodes_element):
                nN_global = self.vt.mesh.elementNodesArray[eN,nN]
                self.globalNode2globalElementList[nN_global].append(eN)
        self.globalNodeGlobalElement2StarElement = []
        self.nElements_node = numpy.zeros((self.vt.mesh.nNodes_global,),'i')
        self.nodeStarElementsArray = numpy.ones((self.vt.mesh.nElements_global,
                                                 self.vt.mesh.nNodes_element),'i')
        self.nodeStarElementsArray[:]=-1
        self.nodeStarElementNeighborsArray = numpy.zeros((self.vt.mesh.nElements_global,
                                                            self.vt.mesh.nNodes_element,
                                                            self.vt.mesh.nElementBoundaries_element),
                                                           'i')
        for I in range(self.vt.mesh.nNodes_global):
            self.globalNode2globalElementList[I].sort()
            self.nElements_node[I] = len(self.globalNode2globalElementList[I])
            self.globalNodeGlobalElement2StarElement.append(dict([(eN_global,eN_node) for eN_node,eN_global in enumerate(self.globalNode2globalElementList[I])]))
        for ebNI in range(self.vt.mesh.nInteriorElementBoundaries_global):
            ebN = self.vt.mesh.interiorElementBoundariesArray[ebNI]
            left_eN_global   = self.vt.mesh.elementBoundaryElementsArray[ebN,0]
            right_eN_global  = self.vt.mesh.elementBoundaryElementsArray[ebN,1]
            left_ebN_element  = self.vt.mesh.elementBoundaryLocalElementBoundariesArray[ebN,0]
            right_ebN_element = self.vt.mesh.elementBoundaryLocalElementBoundariesArray[ebN,1]
            for i in range(self.vt.mesh.nNodes_element):
                left_I = self.vt.mesh.elementNodesArray[left_eN_global,i]
                self.nodeStarElementsArray[left_eN_global,i] = self.globalNodeGlobalElement2StarElement[left_I][left_eN_global]
                if i != left_ebN_element:
                    self.nodeStarElementNeighborsArray[left_eN_global,i,left_ebN_element] = self.globalNodeGlobalElement2StarElement[left_I][right_eN_global]
                #if
                right_I=self.vt.mesh.elementNodesArray[right_eN_global,i]
                self.nodeStarElementsArray[right_eN_global,i] = self.globalNodeGlobalElement2StarElement[right_I][right_eN_global]
                if i != right_ebN_element:
                    self.nodeStarElementNeighborsArray[right_eN_global,i,right_ebN_element] = self.globalNodeGlobalElement2StarElement[right_I][left_eN_global]

            #i
        #ebNi
        for ebNE in range(self.vt.mesh.nExteriorElementBoundaries_global):
            ebN = self.vt.mesh.exteriorElementBoundariesArray[ebNE]
            eN_global   = self.vt.mesh.elementBoundaryElementsArray[ebN,0]
            ebN_element  = self.vt.mesh.elementBoundaryLocalElementBoundariesArray[ebN,0]
            for i in range(self.vt.mesh.nNodes_element):
                I = self.vt.mesh.elementNodesArray[eN_global,i]
                self.nodeStarElementsArray[eN_global,i] = self.globalNodeGlobalElement2StarElement[I][eN_global]
            #i
        #ebNE
        self.nodeStarFactors = {}
        for ci in self.vtComponents:
            self.nodeStarFactors[ci] = cpostprocessing.NodeStarFactor(self.nElements_node,
                                                                      self.nodeStarElementsArray,
                                                                      self.nodeStarElementNeighborsArray)


        #
        self.nDOFs_element = {}
        self.updateConservationJacobian = {}
        for ci in self.vtComponents:
            if not isinstance(self.vt.u[ci].femSpace,FemTools.C0_AffineLinearOnSimplexWithNodalBasis):
                self.solutionTestSpaceIsNotPWL[ci] = True
                logEvent("pwl post-processing for finite element space"+str(self.vt.u[ci].femSpace))
                #need to set up w and w*dS_f and weights to build residuals for linears out of R_{e,i}
                if self.testSpace is None:
                    self.testSpace = FemTools.C0_AffineLinearOnSimplexWithNodalBasis(self.vt.mesh,self.vt.nSpace_global)
                assert isinstance(self.testSpace,FemTools.C0_AffineLinearOnSimplexWithNodalBasis)
                self.alpha[ci] = numpy.zeros((self.testSpace.referenceFiniteElement.localFunctionSpace.dim,
                                          self.vt.u[ci].femSpace.referenceFiniteElement.localFunctionSpace.dim),'d')
                for j,w in enumerate(self.testSpace.referenceFiniteElement.localFunctionSpace.basis):
                    for i,f in enumerate(self.vt.u[ci].femSpace.referenceFiniteElement.interpolationConditions.functionals):
                        self.alpha[ci][j,i] = f(w)
                self.elementResidual[ci] = numpy.zeros((self.vt.mesh.nElements_global,
                                                        self.testSpace.referenceFiniteElement.localFunctionSpace.dim),
                                                       'd')
                self.qv[ci] = numpy.zeros(
                    (self.vt.mesh.nElements_global,
                     self.vt.nQuadraturePoints_element,
                     self.testSpace.referenceFiniteElement.localFunctionSpace.dim),
                    'd')
                self.w[ci] = numpy.zeros(
                    (self.vt.mesh.nElements_global,
                     self.vt.mesh.nElementBoundaries_element,
                     self.vt.nElementBoundaryQuadraturePoints_elementBoundary,
                     self.testSpace.referenceFiniteElement.localFunctionSpace.dim),
                    'd')
                self.w_dS[ci] = numpy.zeros(
                    (self.vt.mesh.nElements_global,
                     self.vt.mesh.nElementBoundaries_element,
                     self.vt.nElementBoundaryQuadraturePoints_elementBoundary,
                     self.testSpace.referenceFiniteElement.localFunctionSpace.dim),
                    'd')
                self.testSpace.getBasisValues(self.vt.elementQuadraturePoints,
                                              self.qv[ci])
                self.testSpace.getBasisValuesTrace(self.vt.u[0].femSpace.elementMaps.permutations,
                                                   self.vt.ebq['hat(x)'],
                                                   self.w[ci])
                cfemIntegrals.calculateWeightedShapeTrace(self.vt.elementBoundaryQuadratureWeights[('u',ci)],
                                                          self.vt.ebq['sqrt(det(g))'],
                                                          self.w[ci],
                                                          self.w_dS[ci])
            else:
                #local element residuals
                self.elementResidual[ci] = self.vt.elementResidual[ci]

            #solution space not P^1
            self.updateConservationJacobian[ci] = True

            #RT0 specific information
            self.nDOFs_element[ci] = self.vt.nSpace_global+1
            self.q[('velocity_dofs',ci)] = numpy.zeros((self.vt.mesh.nElements_global,
                                                        self.nDOFs_element[ci]),'d')

            if ('velocity_l2g',self.vtComponents[0]) in self.q:
                self.q[('velocity_l2g',ci)]  = self.q[('velocity_l2g',self.vtComponents[0])]
            else:
                self.q[('velocity_l2g',ci)]  = numpy.arange((self.vt.mesh.nElements_global*self.vt.mesh.nElementBoundaries_element),dtype='i').reshape((self.vt.mesh.nElements_global,self.vt.mesh.nElementBoundaries_element))
        #ci
        if self.testSpace is None:
            #all the solution spaces must be C0P1
            self.testSpace = self.vt.u[self.vtComponents[0]].femSpace

        ##now setup flux element boundaries including interior information
        self.fluxElementBoundaries_global = {}
        #mwf debug
        self.fluxElementBoundaries_global_tmp = {}
        for ci in self.vtComponents:
            self.fluxElementBoundaries_global[ci] = numpy.zeros((self.vt.mesh.nElementBoundaries_global,),'i')
            #copy over exterior information
            for ebNE in range(self.vt.mesh.nExteriorElementBoundaries_global):
                ebN = self.vt.mesh.exteriorElementBoundariesArray[ebNE]
                self.fluxElementBoundaries_global[ci][ebN] = self.fluxElementBoundaries[ci][ebNE]
            #
            #mwf for debugging some things
            self.fluxElementBoundaries_global_tmp[ci] = numpy.copy(self.fluxElementBoundaries_global[ci])
            self.setInteriorFluxBoundaries(ci)
    #init
    def setInteriorFluxBoundaries(self,ci):
        """Flag interior flux boundaries

        this version flags boundaries where
        :math:`|max(|a|_L)-max(|a|_R)|_{ebN} > jump_{tol}` and
        :math:`min(max(|a|_L),max(|a|_R)) < min_{tol}`

        """
#        TODO:
#          put in off-diagional potentials (ie ('a',ci,cj) cj != ci)
#          put in c (once it works)
        if ('a',ci,ci) in self.vt.q:
            for ebNI in range(self.vt.mesh.nInteriorElementBoundaries_global):
                ebN = self.vt.mesh.interiorElementBoundariesArray[ebNI]
                left_eN_global  = self.vt.mesh.elementBoundaryElementsArray[ebN,0]
                right_eN_global = self.vt.mesh.elementBoundaryElementsArray[ebN,1]
                max_a_left = numpy.max(numpy.absolute(self.vt.q[('a',ci,ci)][left_eN_global].flat))
                max_a_right= numpy.max(numpy.absolute(self.vt.q[('a',ci,ci)][right_eN_global].flat))
                min_a_LR  = min(max_a_left,max_a_right)
                jump_a = abs(max_a_left-max_a_right)
                if min_a_LR > 0.0: jump_a /= min_a_LR
                #mwf debug
                #print "vpp-ib-fix-0 ebN=%s, max_a_left=%s max_a_right= %s " % (ebN,max_a_left,max_a_right)
                if jump_a > self.jump_tol and min_a_LR < self.min_tol:
                    self.fluxElementBoundaries_global[ci][ebN] = 1
                    #mwf debug
                    print("vpp-ib-fix-0 setting ebN=%s ebN = 1, max_a_left=%s max_a_right= %s " % (ebN,max_a_left,max_a_right))

    def setInteriorFluxBoundaryValues(self,ci,ebq_global_velocity):
        """
        set interior flux boundary values
        this version sets interior flux boundaries to no flux

        """
#        TODO:
#          put in c (once it works)
        for ebNI in range(self.vt.mesh.nInteriorElementBoundaries_global):
            ebN = self.vt.mesh.interiorElementBoundariesArray[ebNI]
            if self.fluxElementBoundaries_global[ci][ebN] == 1:
                ebq_global_velocity[ebN].fill(0.0)


    def postprocess_component(self,ci,verbose=0):
        """
        compute mass conservative velocity field following Larson and Niklasson assuming a :math:`P^k C_0`
        Galerkin solution has already been found
        """
        #must zero first time for average velocity
        self.nodeStarFactors[ci].setU(0.0)
        #set interior boundarie values in velocity_average
        self.setInteriorFluxBoundaryValues(ci,self.vt.ebq_global[('velocityAverage',ci)])
        #correct first time through, in case there are Flux boundaries that
        #are not enforced directly in postprocessed flux
        #self.fluxElementBoundaries[ci] determines which boundaries have fluxes
        #enforced directly
        if self.solutionTestSpaceIsNotPWL:
            useC=True
            if useC:
                cpostprocessing.calculateElementResidualPWL(self.alpha[ci],self.vt.elementResidual[ci],self.elementResidual[ci])
            else:
                self.elementResidual[ci].fill(0.0)
                for eN in range(self.vt.mesh.nElements_global):
                    for i in range(self.testSpace.referenceFiniteElement.localFunctionSpace.dim):
                        for j in range(self.vt.u[ci].femSpace.referenceFiniteElement.localFunctionSpace.dim):
                            self.elementResidual[ci][eN,i] += self.alpha[ci][i,j]*self.vt.elementResidual[ci][eN,j]
        self.getConservationResidualPWL(ci,correctFlux=True)
        if verbose > 0:
            logEvent("""velpp Max local conservation (average velocity) = %12.5e""" % max(numpy.absolute(self.q[('conservationResidual',ci)].flat[0:self.vt.mesh.subdomainMesh.nElements_owned])))

        if self.updateConservationJacobian[ci]:
            self.getConservationJacobianPWL(ci)
            self.updateConservationJacobian[ci] = False #only depends on mesh need to resignal if mesh adapts

        cpostprocessing.calculateConservationFluxPWL_noNeumannFix(self.nElements_node,
                                                                  self.nodeStarFactors[ci])
        #
        self.getConservationResidualPWL(ci,correctFlux=False)

        #add back fluxes for elementBoundaries that were Neumann but
        #not enforced directly
        #in this version can still do this just for exterior boundaries since
        #interior boundaries don't have boundary integral term
        #if the interior boundaries are set somewhere that they shouldn't be
        #should show up as mass error
        self.addBoundaryFluxesBackToResidual(ci,self.fluxElementBoundaries[ci])

        if verbose > 0:
            logEvent("Max local conservation (dgp1 enriched) = %12.5e" % max(numpy.absolute(self.q[('conservationResidual',ci)].flat[0:self.vt.mesh.nElements_owned])))

    def getConservationResidualPWL(self,ci,correctFlux=False):
        """
        compute conservation resiudal using current guess for element boundary flux
        """
        if correctFlux == True:
            #exterior boundaries
            self.removeBoundaryFluxesFromResidual(ci,self.fluxElementBoundaries[ci])
            #in this version there's no interior boundary fluxes in residual
            #but in general would have to do this for interior as well
        cpostprocessing.calculateConservationResidualPWL_interiorBoundaries(self.vt.mesh.interiorElementBoundariesArray,
                                                                            self.vt.mesh.exteriorElementBoundariesArray,
                                                                            self.vt.mesh.elementBoundaryElementsArray,
                                                                            self.vt.mesh.elementBoundaryLocalElementBoundariesArray,
                                                                            self.vt.mesh.elementNodesArray,
                                                                            self.nodeStarElementsArray,
                                                                            self.nodeStarElementNeighborsArray,
                                                                            self.nElements_node,
                                                                            self.fluxElementBoundaries_global[ci],
                                                                            self.elementResidual[ci],
                                                                            self.vt.ebq_global[('velocityAverage',ci)],
                                                                            self.vt.ebq[('dS_u',ci)],
                                                                            self.w[ci],
                                                                            self.vt.ebq_global['n'],
                                                                            self.nodeStarFactors[ci],
                                                                            self.q[('conservationResidual',ci)],
                                                                            self.ebq_global[('velocity',ci)],
                                                                            self.ebq[('velocity',ci)])
        #set boundary flux
        updateCoef = 0.0 #overwrite first
        #handle exterior boundaries
        cfemIntegrals.loadBoundaryFluxIntoGlobalElementBoundaryVelocity(self.vt.mesh.exteriorElementBoundariesArray,
                                                                        self.fluxElementBoundaries[ci],
                                                                        self.vt.ebq_global['n'],
                                                                        self.vt.ebq_global[('totalFlux',ci)],
                                                                        updateCoef,
                                                                        self.vt.ebq_global[('velocity',ci)])
        #set interior boundaries
        self.setInteriorFluxBoundaryValues(ci,self.vt.ebq_global[('velocity',ci)])

        cfemIntegrals.copyGlobalElementBoundaryVelocityToElementBoundary(self.vt.mesh.interiorElementBoundariesArray,
                                                                         self.vt.mesh.exteriorElementBoundariesArray,
                                                                         self.vt.mesh.elementBoundaryElementsArray,
                                                                         self.vt.mesh.elementBoundaryLocalElementBoundariesArray,
                                                                         self.vt.ebq_global[('velocity',ci)],
                                                                         self.vt.ebq[('velocity',ci)])


        cfemIntegrals.copyExteriorElementBoundaryValuesFromGlobalElementBoundaryValues(self.vt.mesh.exteriorElementBoundariesArray,
                                                                                       self.vt.mesh.elementBoundaryElementsArray,
                                                                                       self.vt.mesh.elementBoundaryLocalElementBoundariesArray,
                                                                                       self.vt.ebq_global[('velocity',ci)],
                                                                                       self.vt.ebqe[('velocity',ci)])


        #end set boundary flux
        #go from boundary flux to local element boundary representation
        self.evaluateLocalVelocityRepresentation(ci)
    #
    def getConservationJacobianPWL(self,ci):
        """
        Build local systems for post-processing solve
        """
        cpostprocessing.calculateConservationJacobianPWL_interiorBoundaries(self.vt.mesh.interiorElementBoundariesArray,
                                                                            self.vt.mesh.exteriorElementBoundariesArray,
                                                                            self.vt.mesh.elementBoundaryElementsArray,
                                                                            self.vt.mesh.elementBoundaryLocalElementBoundariesArray,
                                                                            self.vt.mesh.elementNodesArray,
                                                                            self.nodeStarElementsArray,
                                                                            self.nodeStarElementNeighborsArray,
                                                                            self.nElements_node,
                                                                            self.fluxElementBoundaries_global[ci],
                                                                            self.w_dS[ci],
                                                                            self.vt.ebq_global['n'],
                                                                            self.nodeStarFactors[ci])

    def evaluateLocalVelocityRepresentation(self,ci):
        """
        project to RT_0 velocity from element boundary fluxes
        """
        cpostprocessing.projectElementBoundaryVelocityToRT0fluxRep(self.vt.ebq[('dS_u',ci)],
                                                                   self.vt.ebq['n'],
                                                                   self.ebq[('velocity',ci)],
                                                                   self.q[('velocity_dofs',ci)])
        cpostprocessing.getElementRT0velocityValuesFluxRep(self.vt.mesh.nodeArray,
                                                           self.vt.mesh.elementNodesArray,
                                                           self.vt.q['abs(det(J))'],
                                                           self.vt.q['x'],
                                                           self.q[('velocity_dofs',ci)],
                                                           self.q[('velocity',ci)])
    def evaluateElementVelocityField(self,x,ci):
        """
        evaluate velocity field assuming velocity_dofs already calculated
        for now assumes x shaped like nE x nq x 3
        """
        assert len(x.shape) == 3, "wrong shape for x= %s " % x.shape
        nE = x.shape[0]; nq = x.shape[1]; nd = self.vt.nSpace_global
        vx = numpy.zeros((nE,nq,nd),'d')
        cpostprocessing.getElementRT0velocityValuesFluxRep(self.vt.mesh.nodeArray,
                                                           self.vt.mesh.elementNodesArray,
                                                           self.vt.q['abs(det(J))'],
                                                           x,
                                                           self.q[('velocity_dofs',ci)],
                                                           vx)
        return vx


####################################################
#class to wrap different algorithms
#all the types need to be defined for vpp_types declaration
class AggregateVelocityPostProcessor(object):
    """
    collect different velocity postprocessing algorithms for a (possibly) multicomponent system
    """
    vpp_types = {'p1-nc':VPP_P1nc_RT0,    #P^1 non-conforming potential, RT_0 velocity repres. (e.g., Chou Tang 02)
                 'pwc':VPP_PWC_RT0,       #Larson Niklasson (global) piecewise constant correction, RT_0 velocity repres.
                 'pwl':VPP_PWL_RT0,       #Larson Niklasson (local) piecewise linear correction, RT_0 velocity repres.
                 'pwl-bdm':VPP_PWL_BDM,   #Larson Niklasson (local) piecewise linear correction, Brezzi Douglas Marini linear velocity repres.
                 'pwl-bdm2':VPP_PWL_BDM2,        # Initial changes to adding BDM2 projection capabilities
                 'pwl-opt':VPP_PWL_RT0_OPT,      #Larson Niklasson (local) piecewise linear correction, RT_0 velocity repres. optimized code
                 'pwl-bdm-opt':VPP_PWL_BDM_OPT,  #Larson Niklasson (local) piecewise linear correction, Brezzi Douglas Marini linear velocity repres. optimized code
                 'pwl-bdm2-opt':VPP_PWL_BDM2_OPT,  #as pwl-bdm-opt, with the BDM2 (quadratic) velocity representation; weak Dirichlet conditions only
                 'sun-rt0':VPP_SUN_RT0,          #Sun and Wheeler global correction, RT_0 velocity repres.
                 'sun-gs-rt0':VPP_SUN_GS_RT0,    #Sun and Wheeler local Gauss-Seidel, RT_0 velocity repres.
                 'point-eval':VPP_POINT_EVAL,    #direct pointwise evaluation of fluxes
                 'dg-point-eval':VPP_POINT_EVAL, #pointwise evaluation for dg
                 'point-eval-gwvd':VPP_POINT_EVAL_GWVD, #pointwise evaluation for dg using LDG method vel_DoFs, add in tjp
                 'dg':VPP_DG_RT0,            #dg scheme, RT_0 local velocity repres
                 'dg-bdm':VPP_DG_BDM,        #dg scheme,  Brezzi Douglas Marini linear velocity repres
                 'pwl-ib-fix-0':VPP_LOW_K_IB_PWL_RT0} #hack to enforce zero flux manually around low perm regions
    def __init__(self,postProcessingTypes=None,transport=None):
        self.postProcessingTypes = postProcessingTypes
        self.vpp_algorithms = []
        self.vt = transport
        self.vpp_components = {}
        if self.postProcessingTypes is not None:
            assert self.vt is not None, "must pass in vectorTransport if doing velocity postprocessing"
            #collect components that share an algorithm
            for ci in list(transport.conservativeFlux.keys()):
                assert transport.conservativeFlux[ci] in list(self.vpp_types.keys()), "invalid postprocessing string"
                if transport.conservativeFlux[ci] in self.vpp_components:
                    self.vpp_components[transport.conservativeFlux[ci]].append(ci)
                else:
                    self.vpp_components[transport.conservativeFlux[ci]] = [ci]
            for algs in list(self.vpp_components.keys()):
                self.vpp_algorithms.append(self.vpp_types[algs](transport,self.vpp_components[algs]))

    def postprocess(self,verbose=0):
        """
        main functionality
        generate post-processed velocity field for attached
        VectorTransport object and store them in local dictionaries

        """
        if self.postProcessingTypes is None:
            return
        for vpp in self.vpp_algorithms:
            vpp.postprocess(verbose=verbose)
    def evaluateElementVelocityField(self,x,ci):
        """
        evaluate velocity field assuming velocity_dofs already calculated
        for now assumes x shaped like nE x nq x 3
        """
        if self.postProcessingTypes is None:
            return None
        for vpp in self.vpp_algorithms:
            return vpp.evaluateElementVelocityField(x,ci)
    def archiveVelocityValues(self,archive,t,tCount,initialPhase=False,meshChanged=False):
        """
        write out post processed velocity values as a finite element space
        """
        if self.postProcessingTypes is None:
            return None
        for vpp in self.vpp_algorithms:
            return vpp.archiveVelocityValues(archive,t,tCount,initialPhase=initialPhase,meshChanged=meshChanged)
