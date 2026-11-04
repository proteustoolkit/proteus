import proteus
import numpy as np
from proteus.Transport import OneLevelTransport
import os
from proteus import cfemIntegrals, Quadrature, Norms, Comm
from proteus.NonlinearSolvers import NonlinearEquation
from proteus.FemTools import (DOFBoundaryConditions,
                              FluxBoundaryConditions,
                              C0_AffineLinearOnSimplexWithNodalBasis)
from proteus.Comm import (globalMax,
                          globalSum)
from proteus.Profiling import memory
from proteus.Profiling import logEvent as log
from proteus.Transport import OneLevelTransport
from proteus.TransportCoefficients import TC_base
from proteus.SubgridError import SGE_base
from proteus.ShockCapturing import ShockCapturing_base
from . import cPresInc
from . import cArgumentsDict
import proteus.NumericalFlux
import proteus.Transport


class NumericalFlux(proteus.NumericalFlux.ConstantAdvection_Diffusion_SIPG_exterior):
    def __init__(self,
                 vt,
                 getPointwiseBoundaryConditions,
                 getAdvectiveFluxBoundaryConditions,
                 getDiffusiveFluxBoundaryConditions):
        proteus.NumericalFlux.ConstantAdvection_Diffusion_SIPG_exterior.__init__(self, vt, getPointwiseBoundaryConditions,
                                                                                 getAdvectiveFluxBoundaryConditions,
                                                                                 getDiffusiveFluxBoundaryConditions)


class Coefficients(TC_base):
    r"""
    The coefficients for pressure increment solution

    Update is given by

    .. math::

       \nabla\cdot( -a \nabla \phi^{k+1} - \mathbf{q^t}^{k+1} ) = 0
       a = \frac{\tau (1-\theta_s)}{\rho_f} + \frac{\tau \theta_s}{\rho_s}
       q^t = (1-\theta_s) v_f + \theta_s v_s
    """
    #: a closed domain fixes this pressure only up to a constant: NumericalSolution.postStep shifts it to zero
    #: integral mean when no Dirichlet condition fixes it (proteus.mprans.PressureNormalization)
    zeroMean = True

    def __init__(self,
                 rho_f_min=998.0,
                 rho_s_min=998.0,
                 nd=2,
                 VOS_model=0,
                 VOF_model=1,
                 modelIndex = None,
                 fluidModelIndex = None, 
                 sedModelIndex = None, 
                 fixNullSpace=False,
                 INTEGRATE_BY_PARTS_DIV_U=True,
                 nullSpace="NoNullSpace",
                 initialize=True):
        """Construct a coefficients object

        :param modelIndex: This model's index into the model list
        :param fluidModelIndex: The fluid momentum model's index
        """
        """
        TODO
        """
        self.nullSpace = nullSpace
        self.fixNullSpace=fixNullSpace
        self.INTEGRATE_BY_PARTS_DIV_U=INTEGRATE_BY_PARTS_DIV_U
        self.VOS_model=VOS_model
        self.VOF_model=VOF_model
        self.nd = nd
        self.rho_f_min = rho_f_min
        self.rho_s_min = rho_s_min
        self.modelIndex = modelIndex
        self.fluidModelIndex = fluidModelIndex
        self.variableNames=['pInc']
        self.sedModelIndex = sedModelIndex
        if initialize:
            self.initialize()

    def initialize(self):
        assert(self.nd in [2,3])        
        if self.nd == 2:
            sdInfo = {(0, 0): (np.array([0, 1, 2], dtype='i'),
                               np.array([0, 1], dtype='i'))}
        else:
            sdInfo = {(0, 0): (np.array([0, 1, 2, 3], dtype='i'),
                               np.array([0, 1, 2], dtype='i'))}
        TC_base.__init__(self,
                         nc=1,
                         variableNames=['pInc'],
                         diffusion={0: {0: {0: 'constant'}}},
                         potential={0: {0: 'u'}},
                         advection={0: {0: 'constant'}},
                         sparseDiffusionTensors=sdInfo,
                         useSparseDiffusion=True)

    def attachModels(self, modelList):
        """
        Attach the model for velocity and density to PresureIncrement model
        """
        self.model = modelList[self.modelIndex]
        self.fluidModel = modelList[self.fluidModelIndex]
        try:
            self.model.coefficients.phi_sp = self.fluidModel.coefficients.phi_s
        except:
            pass
        if self.sedModelIndex is not None:
            self.sedModel = modelList[self.sedModelIndex]
        if self.VOS_model is not None:
            self.model.q_vos = modelList[self.VOS_model].q[('u',0)]
            self.model.ebqe_vos = modelList[self.VOS_model].ebqe[('u',0)]
        else:
            self.model.q_vos = np.zeros_like(self.model.q[('u',0)])
            self.model.ebqe_vos = np.zeros_like(self.model.ebqe[('u',0)])

    def initializeMesh(self, mesh):
        """
        Give the TC object access to the mesh for any mesh-dependent information.
        """
        pass

    def initializeElementQuadrature(self, t, cq):
        """
        Give the TC object access to the element quadrature storage
        """
        pass

    def initializeElementBoundaryQuadrature(self, t, cebq, cebq_global):
        """
        Give the TC object access to the element boundary quadrature storage
        """
        pass

    def initializeGlobalExteriorElementBoundaryQuadrature(self, t, cebqe):
        """
        Give the TC object access to the exterior element boundary quadrature storage
        """
        pass

    def initializeGeneralizedInterpolationPointQuadrature(self, t, cip):
        """
        Give the TC object access to the generalized interpolation point storage. These points are used  to project nonlinear potentials (phi).
        """
        pass

    def preStep(self, t, firstStep=False):
        """
        Move the current values to values_last to keep cached set of values for bdf1 algorithm
        """
        copyInstructions = {}
        return copyInstructions

    def postStep(self, t, firstStep=False):
        """
        Update the fluid velocities
        """
        if self.fluidModel.KILL_PRESSURE_TERM is False and self.fluidModel.coefficients.CORRECT_VELOCITY is True:
            assert self.INTEGRATE_BY_PARTS_DIV_U, "INTEGRATE_BY_PARTS the div(U) must be set to true to correct the velocity"
            alphaBDF = self.fluidModel.timeIntegration.alpha_bdf
            q_vos = self.model.q_vos
            ebqe_vos = self.model.ebqe_vos
            q_a = 1.0/(q_vos*self.rho_s_min + (1.0-q_vos)*self.rho_f_min)/alphaBDF
            ebqe_a = 1.0/(ebqe_vos*self.rho_s_min + (1.0-ebqe_vos)*self.rho_f_min)/alphaBDF
            if self.sedModelIndex is not None:
                for i in range(self.fluidModel.q[('velocity',0)].shape[-1]):
                    self.fluidModel.q[('velocity',0)][...,i] -= self.model.q[('grad(u)',0)][...,i] * q_a
                    self.fluidModel.ebqe[('velocity',0)][...,i] += (self.model.ebqe[('advectiveFlux', 0)] +
                                                                    self.model.ebqe[('diffusiveFlux', 0, 0)] -
                                                                    self.fluidModel.ebqe[('velocity', 0)][..., i]) * self.model.ebqe['n'][..., i]
                    self.fluidModel.coefficients.q_velocity_solid[...,i] -= self.model.q[('grad(u)',0)][...,i] * q_a
                    self.fluidModel.coefficients.ebqe_velocity_solid[...,i] += (self.model.ebqe[('advectiveFlux', 0)] +
                                                                                self.model.ebqe[('diffusiveFlux', 0, 0)] -
                                                                                self.fluidModel.coefficients.ebqe_velocity_solid[..., i]) * self.model.ebqe['n'][..., i]
# end of loop
                self.fluidModel.stabilization.v_last[:] = self.fluidModel.q[('velocity',0)]
                self.fluidModel.coefficients.ebqe_velocity_last[:] = self.fluidModel.ebqe[('velocity',0)]
                self.sedModel.q[('velocity',0)][:] = self.fluidModel.coefficients.q_velocity_solid
                self.sedModel.ebqe[('velocity',0)][:] = self.fluidModel.coefficients.ebqe_velocity_solid
                self.sedModel.stabilization.v_last[:] = self.sedModel.q[('velocity',0)]
                self.sedModel.coefficients.ebqe_velocity_last[:] = self.sedModel.ebqe[('velocity',0)]
                assert(self.fluidModel.coefficients.q_velocity_solid is self.sedModel.q[('velocity',0)])
                assert(self.fluidModel.coefficients.ebqe_velocity_solid is self.sedModel.ebqe[('velocity',0)])
                # This is for disabling the motion of sediment when it's needed
                if self.sedModel.coefficients.staticSediment:
                    for i in range(self.sedModel.q[('velocity',0)].shape[-1]):
                        self.sedModel.q[('velocity',0)][...,i] = 0.0
                        self.sedModel.ebqe[('velocity',0)][...,i] = 0.0  
                        self.fluidModel.coefficients.q_velocity_solid[...,i] = 0.0
                        self.fluidModel.coefficients.ebqe_velocity_solid[...,i] = 0.0 
                #for eN in range(self.model.q_vos.shape[0]):
                #    for k in range(self.model.q_vos.shape[1]):
                #        if self.model.q_vos[eN,k] >= self.sedModel.coefficients.maxFraction:
                #            self.sedModel.q[('velocity',0)][eN,k] = 0.0
                #for eN in range(self.model.ebqe_vos.shape[0]):
                #    for k in range(self.model.ebqe_vos.shape[1]):
                #        if self.model.ebqe_vos[eN,k] >= self.sedModel.coefficients.maxFraction:
                #            self.sedModel.ebqe[('velocity',0)][eN,k] = 0.0 
            else:
                for i in range(self.fluidModel.q[('velocity', 0)].shape[-1]):
                    self.fluidModel.q[('velocity', 0)][..., i] -= self.model.q[('grad(u)',0)][...,i]/(self.rho_f_min*alphaBDF)
                    self.fluidModel.ebqe[('velocity', 0)][..., i] += (self.model.ebqe[('advectiveFlux', 0)] +
                                                                      self.model.ebqe[('diffusiveFlux', 0, 0)] -
                                                                      self.fluidModel.ebqe[('velocity', 0)][..., i]) * self.model.ebqe['n'][..., i]
                    self.fluidModel.coefficients.q_velocity_solid[..., i] -= self.model.q[('grad(u)',0)][...,i]/(self.rho_s_min*alphaBDF)
                    self.fluidModel.coefficients.ebqe_velocity_solid[..., i] -= self.model.ebqe[('grad(u)',0)][...,i]/(self.rho_s_min*alphaBDF)
                self.fluidModel.stabilization.v_last[:] = self.fluidModel.q[('velocity', 0)]
                self.fluidModel.coefficients.ebqe_velocity_last[:] = self.fluidModel.ebqe[('velocity', 0)]
                

        copyInstructions = {}
        return copyInstructions

    def evaluate(self, t, c):
        self.evaluatePresureIncrement(t, c)

    def evaluatePresureIncrement(self, t, c):
        """
        Evaluate the coefficients after getting the velocities and densities
        """
        u_shape = c[('u', 0)].shape
        alphaBDF = self.fluidModel.timeIntegration.alpha_bdf

        vs = None
        vos = None

        if  u_shape == self.fluidModel.q[('u',0)].shape:
            vf = self.fluidModel.q[('velocity',0)]
            rho_f = self.fluidModel.coefficients.q_rho
            rho_s =  self.rho_s_min
            if self.sedModelIndex is not None:
                vs = self.sedModel.q[('velocity',0)]
                vos = self.sedModel.coefficients.q_vos
                rho_s = self.sedModel.coefficients.rho_s
        if  u_shape == self.fluidModel.ebqe[('u',0)].shape:
            vf = self.fluidModel.ebqe[('velocity',0)]
            rho_f = self.fluidModel.coefficients.ebqe_rho
            rho_s =  self.rho_s_min
            if self.sedModelIndex is not None:
                vs = self.sedModel.ebqe[('velocity',0)]
                vos = self.sedModel.coefficients.ebqe_vos
                rho_s = self.sedModel.coefficients.rho_s
        
        assert rho_s >= self.rho_s_min, "solid density out of bounds"
        assert (rho_f >= self.rho_f_min).all(), "fluid density out of bounds"

        if self.sedModelIndex is not None:        
            for i in range(vs.shape[-1]):
                c[('f',0)][...,i] = (1.0-vos)*vf[...,i] + vos*vs[...,i]
        else:
            for i in range(vf.shape[-1]):
                c[('f',0)][...,i] = vf[...,i] 

                #a is really a scalar diffusion but defining it as diagonal tensor
        #if we push phase momentum interchange (drag) to correction
        #then a may become a full  tensor
        if self.sedModelIndex is not None:
            a_penalty = (1.0-vos)*self.rho_f_min*alphaBDF + vos*self.rho_s_min*alphaBDF
            c['a_f'] = (1.0-vos)/a_penalty
            c['a_s'] = (vos)/a_penalty
            c[('a',0,0)][...,0] = c['a_f'] + c['a_s']
        else:
            c['a_f'] = 1.0/(self.rho_f_min*alphaBDF)
            c[('a', 0, 0)][..., 0] =  c['a_f'] 
        for i in range(1,c[('a',0,0)].shape[-1]):
            c[('a',0,0)][...,i] = c[('a',0,0)][...,0]

class LevelModel(proteus.Transport.OneLevelTransport):
    nCalls = 0

    def __init__(self,
                 uDict,
                 phiDict,
                 testSpaceDict,
                 matType,
                 dofBoundaryConditionsDict,
                 dofBoundaryConditionsSetterDict,
                 coefficients,
                 elementQuadrature,
                 elementBoundaryQuadrature,
                 fluxBoundaryConditionsDict=None,
                 advectiveFluxBoundaryConditionsSetterDict=None,
                 diffusiveFluxBoundaryConditionsSetterDictDict=None,
                 stressTraceBoundaryConditionsSetterDict=None,
                 stabilization=None,
                 shockCapturing=None,
                 conservativeFluxDict=None,
                 numericalFluxType=None,
                 TimeIntegrationClass=None,
                 massLumping=False,
                 reactionLumping=False,
                 options=None,
                 name='defaultName',
                 reuse_trial_and_test_quadrature=True,
                 sd=True,
                 movingDomain=False):
        #
        # set the objects describing the method and boundary conditions
        #
        self.movingDomain = movingDomain
        self.tLast_mesh = None
        #
        self.name = name
        self.sd = sd
        self.Hess = False
        self.lowmem = True
        self.timeTerm = True  # allow turning off  the  time derivative
        # self.lowmem=False
        self.testIsTrial = True
        self.phiTrialIsTrial = True
        self.u = uDict
        self.ua = {}  # analytical solutions
        self.phi = phiDict
        self.dphi = {}
        self.matType = matType
        # mwf try to reuse test and trial information across components if
        # spaces are the same
        self.reuse_test_trial_quadrature = reuse_trial_and_test_quadrature  # True#False
        if self.reuse_test_trial_quadrature:
            for ci in range(1, coefficients.nc):
                assert self.u[ci].femSpace.__class__.__name__ == self.u[
                    0].femSpace.__class__.__name__, "to reuse_test_trial_quad all femSpaces must be the same!"
        # Simplicial Mesh
        # assume the same mesh for  all components for now
        self.mesh = self.u[0].femSpace.mesh
        self.testSpace = testSpaceDict
        self.dirichletConditions = dofBoundaryConditionsDict
        # explicit Dirichlet  conditions for now, no Dirichlet BC constraints
        self.dirichletNodeSetList = None
        self.coefficients = coefficients
        self.coefficients.initializeMesh(self.mesh)
        self.nc = self.coefficients.nc
        self.stabilization = stabilization
        self.shockCapturing = shockCapturing
        # no velocity post-processing for now
        self.conservativeFlux = conservativeFluxDict
        self.fluxBoundaryConditions = fluxBoundaryConditionsDict
        self.advectiveFluxBoundaryConditionsSetterDict = advectiveFluxBoundaryConditionsSetterDict
        self.diffusiveFluxBoundaryConditionsSetterDictDict = diffusiveFluxBoundaryConditionsSetterDictDict
        # determine whether  the stabilization term is nonlinear
        self.stabilizationIsNonlinear = False
        # cek come back
        if self.stabilization is not None:
            for ci in range(self.nc):
                if ci in coefficients.mass:
                    for flag in list(coefficients.mass[ci].values()):
                        if flag == 'nonlinear':
                            self.stabilizationIsNonlinear = True
                if ci in coefficients.advection:
                    for flag in list(coefficients.advection[ci].values()):
                        if flag == 'nonlinear':
                            self.stabilizationIsNonlinear = True
                if ci in coefficients.diffusion:
                    for diffusionDict in list(coefficients.diffusion[ci].values()):
                        for flag in list(diffusionDict.values()):
                            if flag != 'constant':
                                self.stabilizationIsNonlinear = True
                if ci in coefficients.potential:
                    for flag in list(coefficients.potential[ci].values()):
                        if flag == 'nonlinear':
                            self.stabilizationIsNonlinear = True
                if ci in coefficients.reaction:
                    for flag in list(coefficients.reaction[ci].values()):
                        if flag == 'nonlinear':
                            self.stabilizationIsNonlinear = True
                if ci in coefficients.hamiltonian:
                    for flag in list(coefficients.hamiltonian[ci].values()):
                        if flag == 'nonlinear':
                            self.stabilizationIsNonlinear = True
        # determine if we need element boundary storage
        self.elementBoundaryIntegrals = {}
        for ci in range(self.nc):
            self.elementBoundaryIntegrals[ci] = (
                (self.conservativeFlux is not None) or (
                    numericalFluxType is not None) or (
                    self.fluxBoundaryConditions[ci] == 'outFlow') or (
                    self.fluxBoundaryConditions[ci] == 'mixedFlow') or (
                    self.fluxBoundaryConditions[ci] == 'setFlow'))
        #
        # calculate some dimensions
        #
        # assume same space dim for all variables
        self.nSpace_global = self.u[0].femSpace.nSpace_global
        self.nDOF_trial_element = [
            u_j.femSpace.max_nDOF_element for u_j in list(self.u.values())]
        self.nDOF_phi_trial_element = [
            phi_k.femSpace.max_nDOF_element for phi_k in list(self.phi.values())]
        self.n_phi_ip_element = [
            phi_k.femSpace.referenceFiniteElement.interpolationConditions.nQuadraturePoints for phi_k in list(self.phi.values())]
        self.nDOF_test_element = [
            femSpace.max_nDOF_element for femSpace in list(self.testSpace.values())]
        self.nFreeDOF_global = [
            dc.nFreeDOF_global for dc in list(self.dirichletConditions.values())]
        self.nVDOF_element = sum(self.nDOF_trial_element)
        self.nFreeVDOF_global = sum(self.nFreeDOF_global)
        #
        NonlinearEquation.__init__(self, self.nFreeVDOF_global)
        #
        # build the quadrature point dictionaries from the input (this
        # is just for convenience so that the input doesn't have to be
        # complete)
        #
        elementQuadratureDict = {}
        elemQuadIsDict = isinstance(elementQuadrature, dict)
        if elemQuadIsDict:  # set terms manually
            for I in self.coefficients.elementIntegralKeys:
                if I in elementQuadrature:
                    elementQuadratureDict[I] = elementQuadrature[I]
                else:
                    elementQuadratureDict[I] = elementQuadrature['default']
        else:
            for I in self.coefficients.elementIntegralKeys:
                elementQuadratureDict[I] = elementQuadrature
        if self.stabilization is not None:
            for I in self.coefficients.elementIntegralKeys:
                if elemQuadIsDict:
                    if I in elementQuadrature:
                        elementQuadratureDict[
                            ('stab',) + I[1:]] = elementQuadrature[I]
                    else:
                        elementQuadratureDict[
                            ('stab',) + I[1:]] = elementQuadrature['default']
                else:
                    elementQuadratureDict[
                        ('stab',) + I[1:]] = elementQuadrature
        if self.shockCapturing is not None:
            for ci in self.shockCapturing.components:
                if elemQuadIsDict:
                    if ('numDiff', ci, ci) in elementQuadrature:
                        elementQuadratureDict[('numDiff', ci, ci)] = elementQuadrature[
                            ('numDiff', ci, ci)]
                    else:
                        elementQuadratureDict[('numDiff', ci, ci)] = elementQuadrature[
                            'default']
                else:
                    elementQuadratureDict[
                        ('numDiff', ci, ci)] = elementQuadrature
        if massLumping:
            for ci in list(self.coefficients.mass.keys()):
                elementQuadratureDict[('m', ci)] = Quadrature.SimplexLobattoQuadrature(
                    self.nSpace_global, 1)
            for I in self.coefficients.elementIntegralKeys:
                elementQuadratureDict[
                    ('stab',) + I[1:]] = Quadrature.SimplexLobattoQuadrature(self.nSpace_global, 1)
        if reactionLumping:
            for ci in list(self.coefficients.mass.keys()):
                elementQuadratureDict[('r', ci)] = Quadrature.SimplexLobattoQuadrature(
                    self.nSpace_global, 1)
            for I in self.coefficients.elementIntegralKeys:
                elementQuadratureDict[
                    ('stab',) + I[1:]] = Quadrature.SimplexLobattoQuadrature(self.nSpace_global, 1)
        elementBoundaryQuadratureDict = {}
        if isinstance(elementBoundaryQuadrature, dict):  # set terms manually
            for I in self.coefficients.elementBoundaryIntegralKeys:
                if I in elementBoundaryQuadrature:
                    elementBoundaryQuadratureDict[
                        I] = elementBoundaryQuadrature[I]
                else:
                    elementBoundaryQuadratureDict[
                        I] = elementBoundaryQuadrature['default']
        else:
            for I in self.coefficients.elementBoundaryIntegralKeys:
                elementBoundaryQuadratureDict[I] = elementBoundaryQuadrature
        #
        # find the union of all element quadrature points and
        # build a quadrature rule for each integral that has a
        # weight at each point in the union
        # mwf include tag telling me which indices are which quadrature rule?
        (self.elementQuadraturePoints, self.elementQuadratureWeights,
         self.elementQuadratureRuleIndeces) = Quadrature.buildUnion(elementQuadratureDict)
        self.nQuadraturePoints_element = self.elementQuadraturePoints.shape[0]
        self.nQuadraturePoints_global = self.nQuadraturePoints_element * \
            self.mesh.nElements_global
        #
        # Repeat the same thing for the element boundary quadrature
        #
        (self.elementBoundaryQuadraturePoints, self.elementBoundaryQuadratureWeights,
         self.elementBoundaryQuadratureRuleIndeces) = Quadrature.buildUnion(elementBoundaryQuadratureDict)
        self.nElementBoundaryQuadraturePoints_elementBoundary = self.elementBoundaryQuadraturePoints.shape[
            0]
        self.nElementBoundaryQuadraturePoints_global = (
            self.mesh.nElements_global *
            self.mesh.nElementBoundaries_element *
            self.nElementBoundaryQuadraturePoints_elementBoundary)
        #
        # simplified allocations for test==trial and also check if space is mixed or not
        #
        self.q = {}
        self.ebq = {}
        self.ebq_global = {}
        self.ebqe = {}
        self.phi_ip = {}
        # mesh
        #self.q['x'] = np.zeros((self.mesh.nElements_global,self.nQuadraturePoints_element,3),'d')
        self.ebqe['x'] = np.zeros(
            (self.mesh.nExteriorElementBoundaries_global,
             self.nElementBoundaryQuadraturePoints_elementBoundary,
             3),
            'd')
        self.q[('u', 0)] = np.zeros(
            (self.mesh.nElements_global, self.nQuadraturePoints_element), 'd')
        self.q['a'] = np.zeros(
            (self.mesh.nElements_global, self.nQuadraturePoints_element), 'd')
        self.q[
            ('grad(u)',
             0)] = np.zeros(
            (self.mesh.nElements_global,
             self.nQuadraturePoints_element,
             self.nSpace_global),
            'd')
        self.ebqe[
            ('u',
             0)] = np.zeros(
            (self.mesh.nExteriorElementBoundaries_global,
             self.nElementBoundaryQuadraturePoints_elementBoundary),
            'd')
        self.ebqe['a'] = np.zeros(
            (self.mesh.nExteriorElementBoundaries_global,
             self.nElementBoundaryQuadraturePoints_elementBoundary),
            'd')
        self.ebqe[
            ('grad(u)',
             0)] = np.zeros(
            (self.mesh.nExteriorElementBoundaries_global,
             self.nElementBoundaryQuadraturePoints_elementBoundary,
             self.nSpace_global),
            'd')

        self.q[('v', 0)] = np.zeros(
            (self.mesh.nElements_global,
             self.nQuadraturePoints_element,
             self.nDOF_trial_element[0]),
            'd')
        self.q['J'] = np.zeros(
            (self.mesh.nElements_global,
             self.nQuadraturePoints_element,
             self.nSpace_global,
             self.nSpace_global),
            'd')
        self.q['det(J)'] = np.zeros(
            (self.mesh.nElements_global,
             self.nQuadraturePoints_element),
            'd')
        self.q['inverse(J)'] = np.zeros(
            (self.mesh.nElements_global,
             self.nQuadraturePoints_element,
             self.nSpace_global,
             self.nSpace_global),
            'd')
        self.ebq[('v', 0)] = np.zeros(
            (self.mesh.nElements_global,
             self.mesh.nElementBoundaries_element,
             self.nElementBoundaryQuadraturePoints_elementBoundary,
             self.nDOF_trial_element[0]),
            'd')
        self.ebq[('w', 0)] = np.zeros(
            (self.mesh.nElements_global,
             self.mesh.nElementBoundaries_element,
             self.nElementBoundaryQuadraturePoints_elementBoundary,
             self.nDOF_trial_element[0]),
            'd')
        self.ebq['x'] = np.zeros(
            (self.mesh.nElements_global,
             self.mesh.nElementBoundaries_element,
             self.nElementBoundaryQuadraturePoints_elementBoundary,
             3),
            'd')
        self.ebq['hat(x)'] = np.zeros(
            (self.mesh.nElements_global,
             self.mesh.nElementBoundaries_element,
             self.nElementBoundaryQuadraturePoints_elementBoundary,
             3),
            'd')
        self.ebq['inverse(J)'] = np.zeros(
            (self.mesh.nElements_global,
             self.mesh.nElementBoundaries_element,
             self.nElementBoundaryQuadraturePoints_elementBoundary,
             self.nSpace_global,
             self.nSpace_global),
            'd')
        self.ebq['g'] = np.zeros(
            (self.mesh.nElements_global,
             self.mesh.nElementBoundaries_element,
             self.nElementBoundaryQuadraturePoints_elementBoundary,
             self.nSpace_global - 1,
             self.nSpace_global - 1),
            'd')
        self.ebq['sqrt(det(g))'] = np.zeros(
            (self.mesh.nElements_global,
             self.mesh.nElementBoundaries_element,
             self.nElementBoundaryQuadraturePoints_elementBoundary),
            'd')
        self.ebq['n'] = np.zeros(
            (self.mesh.nElements_global,
             self.mesh.nElementBoundaries_element,
             self.nElementBoundaryQuadraturePoints_elementBoundary,
             self.nSpace_global),
            'd')
        self.ebq[('dS_u', 0)] = np.zeros(
            (self.mesh.nElements_global,
             self.mesh.nElementBoundaries_element,
             self.nElementBoundaryQuadraturePoints_elementBoundary),
            'd')
        self.ebqe['dS'] = np.zeros(
            (self.mesh.nExteriorElementBoundaries_global,
             self.nElementBoundaryQuadraturePoints_elementBoundary),
            'd')
        self.ebqe[('dS_u', 0)] = self.ebqe['dS']
        self.ebqe['n'] = np.zeros(
            (self.mesh.nExteriorElementBoundaries_global,
             self.nElementBoundaryQuadraturePoints_elementBoundary,
             self.nSpace_global),
            'd')
        self.ebqe['inverse(J)'] = np.zeros(
            (self.mesh.nExteriorElementBoundaries_global,
             self.nElementBoundaryQuadraturePoints_elementBoundary,
             self.nSpace_global,
             self.nSpace_global),
            'd')
        self.ebqe['g'] = np.zeros(
            (self.mesh.nExteriorElementBoundaries_global,
             self.nElementBoundaryQuadraturePoints_elementBoundary,
             self.nSpace_global - 1,
             self.nSpace_global - 1),
            'd')
        self.ebqe['sqrt(det(g))'] = np.zeros(
            (self.mesh.nExteriorElementBoundaries_global,
             self.nElementBoundaryQuadraturePoints_elementBoundary),
            'd')
        self.ebq_global['n'] = np.zeros(
            (self.mesh.nElementBoundaries_global,
             self.nElementBoundaryQuadraturePoints_elementBoundary,
             self.nSpace_global),
            'd')
        self.ebq_global['x'] = np.zeros(
            (self.mesh.nElementBoundaries_global,
             self.nElementBoundaryQuadraturePoints_elementBoundary,
             3),
            'd')
        self.ebqe[('advectiveFlux_bc_flag', 0)] = np.zeros(
            (self.mesh.nExteriorElementBoundaries_global, self.nElementBoundaryQuadraturePoints_elementBoundary), 'i')
        self.ebqe[('diffusiveFlux_bc_flag', 0, 0)] = np.zeros(
            (self.mesh.nExteriorElementBoundaries_global, self.nElementBoundaryQuadraturePoints_elementBoundary), 'i')
        self.ebqe[('advectiveFlux_bc', 0)] = np.zeros((self.mesh.nExteriorElementBoundaries_global, self.nElementBoundaryQuadraturePoints_elementBoundary), 'd')
        self.ebqe[('diffusiveFlux_bc', 0, 0)] = np.zeros(
            (self.mesh.nExteriorElementBoundaries_global, self.nElementBoundaryQuadraturePoints_elementBoundary), 'd')
        self.ebqe[('advectiveFlux', 0)] = np.zeros((self.mesh.nExteriorElementBoundaries_global, self.nElementBoundaryQuadraturePoints_elementBoundary), 'd')
        self.ebqe[('diffusiveFlux', 0, 0)] = np.zeros((self.mesh.nExteriorElementBoundaries_global, self.nElementBoundaryQuadraturePoints_elementBoundary), 'd')
        self.points_elementBoundaryQuadrature = set()
        self.scalars_elementBoundaryQuadrature = set(
            [('u', ci) for ci in range(self.nc)])
        self.vectors_elementBoundaryQuadrature = set()
        self.tensors_elementBoundaryQuadrature = set()
        log(memory("element and element boundary Jacobians",
                   "OneLevelTransport"), level=4)
        self.inflowBoundaryBC = {}
        self.inflowBoundaryBC_values = {}
        self.inflowFlux = {}
        for cj in range(self.nc):
            self.inflowBoundaryBC[cj] = np.zeros(
                (self.mesh.nExteriorElementBoundaries_global,), 'i')
            self.inflowBoundaryBC_values[cj] = np.zeros(
                (self.mesh.nExteriorElementBoundaries_global, self.nDOF_trial_element[cj]), 'd')
            self.inflowFlux[cj] = np.zeros(
                (self.mesh.nExteriorElementBoundaries_global,
                 self.nElementBoundaryQuadraturePoints_elementBoundary),
                'd')
        self.internalNodes = set(range(self.mesh.nNodes_global))
        # identify the internal nodes this is ought to be in mesh
        # \todo move this to mesh
        for ebNE in range(self.mesh.nExteriorElementBoundaries_global):
            ebN = self.mesh.exteriorElementBoundariesArray[ebNE]
            eN_global = self.mesh.elementBoundaryElementsArray[ebN, 0]
            ebN_element = self.mesh.elementBoundaryLocalElementBoundariesArray[
                ebN, 0]
            for i in range(self.mesh.nNodes_element):
                if i != ebN_element:
                    I = self.mesh.elementNodesArray[eN_global, i]
                    self.internalNodes -= set([I])
        self.nNodes_internal = len(self.internalNodes)
        self.internalNodesArray = np.zeros((self.nNodes_internal,), 'i')
        for nI, n in enumerate(self.internalNodes):
            self.internalNodesArray[nI] = n
        #
        del self.internalNodes
        self.internalNodes = None
        log("Updating local to global mappings", 2)
        self.updateLocal2Global()
        log("Building time integration object", 2)
        log(memory("inflowBC, internalNodes,updateLocal2Global",
                   "OneLevelTransport"), level=4)
        # mwf for interpolating subgrid error for gradients etc
        if self.stabilization and self.stabilization.usesGradientStabilization:
            self.timeIntegration = TimeIntegrationClass(
                self, integrateInterpolationPoints=True)
        else:
            self.timeIntegration = TimeIntegrationClass(self)

        if options is not None:
            self.timeIntegration.setFromOptions(options)
        log(memory("TimeIntegration", "OneLevelTransport"), level=4)
        log("Calculating numerical quadrature formulas", 2)
        self.calculateQuadrature()
        self.setupFieldStrides()

        comm = Comm.get()
        self.comm = comm
        if comm.size() > 1:
            assert numericalFluxType is not None and numericalFluxType.useWeakDirichletConditions, "You must use a numerical flux to apply weak boundary conditions for parallel runs"
            if self.coefficients.fixNullSpace:
                # fixNullSpace pins dof 0 on rank 0 by rewriting its row. Under a
                # partition without overlap that dof can be shared, and the other
                # rank adds its partial row back, so the pin is wrong. The
                # increment is determined up to a constant, which is what the
                # constant null space of the KSP solve removes.
                self.coefficients.fixNullSpace = False
                self.coefficients.nullSpace = 'ConstantComponentsNullSpace'
                self.coefficients.nullSpaceComponents = [0]

        log(memory("stride+offset", "OneLevelTransport"), level=4)
        if numericalFluxType is not None:
            if options is None or options.periodicDirichletConditions is None:
                self.numericalFlux = numericalFluxType(
                    self,
                    dofBoundaryConditionsSetterDict,
                    advectiveFluxBoundaryConditionsSetterDict,
                    diffusiveFluxBoundaryConditionsSetterDictDict)
            else:
                self.numericalFlux = numericalFluxType(
                    self,
                    dofBoundaryConditionsSetterDict,
                    advectiveFluxBoundaryConditionsSetterDict,
                    diffusiveFluxBoundaryConditionsSetterDictDict,
                    options.periodicDirichletConditions)
        else:
            self.numericalFlux = None
        # set penalty terms
        # cek todo move into numerical flux initialization
        if 'penalty' in self.ebq_global:
            for ebN in range(self.mesh.nElementBoundaries_global):
                for k in range(
                        self.nElementBoundaryQuadraturePoints_elementBoundary):
                    self.ebq_global['penalty'][ebN, k] = self.numericalFlux.penalty_constant/(self.mesh.elementBoundaryDiametersArray[ebN]**self.numericalFlux.penalty_power)
        # penalty term
        # cek move  to Numerical flux initialization
        if 'penalty' in self.ebqe:
            for ebNE in range(self.mesh.nExteriorElementBoundaries_global):
                ebN = self.mesh.exteriorElementBoundariesArray[ebNE]
                for k in range(
                        self.nElementBoundaryQuadraturePoints_elementBoundary):
                    self.ebqe['penalty'][ebNE, k] = self.numericalFlux.penalty_constant/self.mesh.elementBoundaryDiametersArray[ebN]**self.numericalFlux.penalty_power
        log(memory("numericalFlux", "OneLevelTransport"), level=4)
        self.elementEffectiveDiametersArray = self.mesh.elementInnerDiametersArray
        # use post processing tools to get conservative fluxes, None by default
        from proteus import PostProcessingTools
        self.velocityPostProcessor = PostProcessingTools.VelocityPostProcessingChooser(
            self)
        log(memory("velocity postprocessor", "OneLevelTransport"), level=4)
        # helper for writing out data storage
        from proteus import Archiver
        self.elementQuadratureDictionaryWriter = Archiver.XdmfWriter()
        self.elementBoundaryQuadratureDictionaryWriter = Archiver.XdmfWriter()
        self.exteriorElementBoundaryQuadratureDictionaryWriter = Archiver.XdmfWriter()
        self.globalResidualDummy = None
        log("flux bc objects")
        for ci, fbcObject in list(self.fluxBoundaryConditionsObjectsDict.items()):
            self.ebqe[('advectiveFlux_bc_flag', ci)] = np.zeros(self.ebqe[('advectiveFlux_bc', ci)].shape, 'i')
            for t, g in list(fbcObject.advectiveFluxBoundaryConditionsDict.items()):
                if ci in self.coefficients.advection:
                    self.ebqe[('advectiveFlux_bc', ci)][t[0], t[1]] = g(self.ebqe[('x')][t[0], t[1]], self.timeIntegration.t)
                    self.ebqe[('advectiveFlux_bc_flag', ci)][t[0], t[1]] = 1
            for ck, diffusiveFluxBoundaryConditionsDict in list(fbcObject.diffusiveFluxBoundaryConditionsDictDict.items()):
                self.ebqe[('diffusiveFlux_bc_flag', ck, ci)] = np.zeros(self.ebqe[('diffusiveFlux_bc', ck, ci)].shape, 'i')
                for t, g in list(diffusiveFluxBoundaryConditionsDict.items()):
                    self.ebqe[('diffusiveFlux_bc', ck, ci)][t[0], t[1]] = g(self.ebqe[('x')][t[0], t[1]], self.timeIntegration.t)
                    self.ebqe[('diffusiveFlux_bc_flag', ck, ci)][t[0], t[1]] = 1
        self.numericalFlux.setDirichletValues(self.ebqe)
        compKernelFlag = 0
        self.presinc = cPresInc.PresInc(
            self.nSpace_global,
            self.nQuadraturePoints_element,
            self.u[0].femSpace.elementMaps.localFunctionSpace.dim,
            self .u[0].femSpace.referenceFiniteElement.localFunctionSpace.dim,
            self.testSpace[0].referenceFiniteElement.localFunctionSpace.dim,
            self.nElementBoundaryQuadraturePoints_elementBoundary,
            compKernelFlag)

    def calculateCoefficients(self):
        pass

    def calculateElementResidual(self):
        if self.globalResidualDummy is not None:
            self.getResidual(self.u[0].dof, self.globalResidualDummy)

    def getResidual(self, u, r):
        import pdb
        import copy
        """
        Calculate the element residuals and add in to the global residual
        """
        r.fill(0.0)
        # Load the unknowns into the finite element dof
        self.setUnknowns(u)
        self.numericalFlux.setDirichletValues(self.ebqe)

        # flux boundary conditions
        for t, g in list(self.fluxBoundaryConditionsObjectsDict[
                0].advectiveFluxBoundaryConditionsDict.items()):
            self.ebqe[('advectiveFlux_bc', 0)][t[0], t[1]] = g(self.ebqe[('x')][t[0], t[1]], self.timeIntegration.t)
            self.ebqe[('advectiveFlux_bc_flag', 0)][t[0], t[1]] = 1
        for t, g in list(self.fluxBoundaryConditionsObjectsDict[
                0].diffusiveFluxBoundaryConditionsDictDict[0].items()):
            self.ebqe[('diffusiveFlux_bc',0,0)][t[0],t[1]] = g(self.ebqe[('x')][t[0],t[1]],self.timeIntegration.t)
            self.ebqe[('diffusiveFlux_bc_flag',0,0)][t[0],t[1]] = 1 
        if self.coefficients.fixNullSpace and self.comm.rank() == 0:
            self.u[0].dof[0] = 0
        argsDict = cArgumentsDict.ArgumentsDict()
        argsDict["mesh_trial_ref"] = self.u[0].femSpace.elementMaps.psi
        argsDict["mesh_grad_trial_ref"] = self.u[0].femSpace.elementMaps.grad_psi
        argsDict["mesh_dof"] = self.mesh.nodeArray
        argsDict["mesh_l2g"] = self.mesh.elementNodesArray
        argsDict["dV_ref"] = self.elementQuadratureWeights[('u', 0)]
        argsDict["u_trial_ref"] = self.u[0].femSpace.psi
        argsDict["u_grad_trial_ref"] = self.u[0].femSpace.grad_psi
        argsDict["u_test_ref"] = self.u[0].femSpace.psi
        argsDict["u_grad_test_ref"] = self.u[0].femSpace.grad_psi
        argsDict["mesh_trial_trace_ref"] = self.u[0].femSpace.elementMaps.psi_trace
        argsDict["mesh_grad_trial_trace_ref"] = self.u[0].femSpace.elementMaps.grad_psi_trace
        argsDict["dS_ref"] = self.elementBoundaryQuadratureWeights[('u', 0)]
        argsDict["u_trial_trace_ref"] = self.u[0].femSpace.psi_trace
        argsDict["u_grad_trial_trace_ref"] = self.u[0].femSpace.grad_psi_trace
        argsDict["u_test_trace_ref"] = self.u[0].femSpace.psi_trace
        argsDict["u_grad_test_trace_ref"] = self.u[0].femSpace.grad_psi_trace
        argsDict["normal_ref"] = self.u[0].femSpace.elementMaps.boundaryNormals
        argsDict["boundaryJac_ref"] = self.u[0].femSpace.elementMaps.boundaryJacobians
        argsDict["nElements_global"] = self.mesh.nElements_global
        argsDict["isDOFBoundary"] = self.numericalFlux.isDOFBoundary[0]
        argsDict["isFluxBoundary"] = self.ebqe[('diffusiveFlux_bc_flag', 0, 0)]
        argsDict["u_l2g"] = self.u[0].femSpace.dofMap.l2g
        argsDict["u_dof"] = self.u[0].dof
        argsDict["alphaBDF"] = self.coefficients.fluidModel.timeIntegration.alpha_bdf
        argsDict["q_vf"] = self.coefficients.fluidModel.q[('velocity', 0)]
        argsDict["q_divU"] = self.coefficients.fluidModel.q['divU']
        argsDict["q_vs"] = self.coefficients.fluidModel.coefficients.q_velocity_solid
        argsDict["q_vos"] = self.coefficients.fluidModel.coefficients.q_vos
        argsDict["rho_s"] = self.coefficients.fluidModel.coefficients.rho_s
        argsDict["q_rho_f"] = self.coefficients.fluidModel.coefficients.q_rho
        argsDict["rho_s_min"] = self.coefficients.rho_s_min
        argsDict["rho_f_min"] = self.coefficients.rho_f_min
        argsDict["ebqe_vf"] = self.coefficients.fluidModel.ebqe[('velocity', 0)]
        argsDict["ebqe_vs"] = self.coefficients.fluidModel.coefficients.ebqe_velocity_solid
        argsDict["ebqe_vos"] = self.coefficients.fluidModel.coefficients.ebqe_vos
        argsDict["ebqe_rho_f"] = self.coefficients.fluidModel.coefficients.ebqe_rho
        argsDict["q_u"] = self.q[('u', 0)]
        argsDict["q_grad_u"] = self.q[('grad(u)', 0)]
        argsDict["ebqe_u"] = self.ebqe[('u', 0)]
        argsDict["ebqe_grad_u"] = self.ebqe[('grad(u)', 0)]
        argsDict["ebqe_bc_u_ext"] = self.numericalFlux.ebqe[('u', 0)]
        argsDict["ebqe_adv_flux"] = self.ebqe[('advectiveFlux', 0)]
        argsDict["ebqe_diff_flux"] = self.ebqe[('diffusiveFlux', 0, 0)]
        argsDict["bc_adv_flux"] = self.ebqe[('advectiveFlux_bc', 0)]
        argsDict["bc_diff_flux"] = self.ebqe[('diffusiveFlux_bc', 0, 0)]
        argsDict["offset_u"] = self.offset[0]
        argsDict["stride_u"] = self.stride[0]
        argsDict["globalResidual"] = r
        argsDict["nExteriorElementBoundaries_global"] = self.mesh.nExteriorElementBoundaries_global
        argsDict["exteriorElementBoundariesArray"] = self.mesh.exteriorElementBoundariesArray
        argsDict["elementBoundaryElementsArray"] = self.mesh.elementBoundaryElementsArray
        argsDict["elementBoundaryLocalElementBoundariesArray"] = self.mesh.elementBoundaryLocalElementBoundariesArray
        argsDict["INTEGRATE_BY_PARTS_DIV_U"] = self.coefficients.INTEGRATE_BY_PARTS_DIV_U
        argsDict["q_a"] = self.q['a']
        argsDict["ebqe_a"] = self.ebqe['a']
        self.presinc.calculateResidual(argsDict)

        if self.coefficients.fixNullSpace and self.comm.rank() == 0:
            r[0] = 0.
        log("Global residual", level=9, data=r)
        # turn this on to view the global conservation residual
        # it should be the same as the linear solver residual tolerance
        # self.coefficients.massConservationError = fabs(
        #    globalSum(r[:self.mesh.nNodes_owned].sum()))
        # log("   Mass Conservation Error", level=3,
        #    data=self.coefficients.massConservationError)
        self.nonlinear_function_evaluations += 1

    def getJacobian(self, jacobian):
        cfemIntegrals.zeroJacobian_CSR(self.nNonzerosInJacobian, jacobian)
        argsDict = cArgumentsDict.ArgumentsDict()
        argsDict["mesh_trial_ref"] = self.u[0].femSpace.elementMaps.psi
        argsDict["mesh_grad_trial_ref"] = self.u[0].femSpace.elementMaps.grad_psi
        argsDict["mesh_dof"] = self.mesh.nodeArray
        argsDict["mesh_l2g"] = self.mesh.elementNodesArray
        argsDict["dV_ref"] = self.elementQuadratureWeights[('u', 0)]
        argsDict["u_trial_ref"] = self.u[0].femSpace.psi
        argsDict["u_grad_trial_ref"] = self.u[0].femSpace.grad_psi
        argsDict["u_test_ref"] = self.u[0].femSpace.psi
        argsDict["u_grad_test_ref"] = self.u[0].femSpace.grad_psi
        argsDict["mesh_trial_trace_ref"] = self.u[0].femSpace.elementMaps.psi_trace
        argsDict["mesh_grad_trial_trace_ref"] = self.u[0].femSpace.elementMaps.grad_psi_trace
        argsDict["dS_ref"] = self.elementBoundaryQuadratureWeights[('u', 0)]
        argsDict["u_trial_trace_ref"] = self.u[0].femSpace.psi_trace
        argsDict["u_grad_trial_trace_ref"] = self.u[0].femSpace.grad_psi_trace
        argsDict["u_test_trace_ref"] = self.u[0].femSpace.psi_trace
        argsDict["u_grad_test_trace_ref"] = self.u[0].femSpace.grad_psi_trace
        argsDict["normal_ref"] = self.u[0].femSpace.elementMaps.boundaryNormals
        argsDict["boundaryJac_ref"] = self.u[0].femSpace.elementMaps.boundaryJacobians
        argsDict["nElements_global"] = self.mesh.nElements_global
        argsDict["isDOFBoundary"] = self.numericalFlux.isDOFBoundary[0]
        argsDict["isFluxBoundary"] = self.ebqe[('diffusiveFlux_bc_flag', 0, 0)]
        argsDict["u_l2g"] = self.u[0].femSpace.dofMap.l2g
        argsDict["u_dof"] = self.u[0].dof
        argsDict["alphaBDF"] = self.coefficients.fluidModel.timeIntegration.alpha_bdf
        argsDict["q_vf"] = self.coefficients.fluidModel.q[('velocity', 0)]
        argsDict["q_vs"] = self.coefficients.fluidModel.coefficients.q_velocity_solid
        argsDict["q_vos"] = self.coefficients.fluidModel.coefficients.q_vos
        argsDict["rho_s"] = self.coefficients.fluidModel.coefficients.rho_s
        argsDict["q_rho_f"] = self.coefficients.fluidModel.coefficients.q_rho
        argsDict["rho_s_min"] = self.coefficients.rho_s_min
        argsDict["rho_f_min"] = self.coefficients.rho_f_min
        argsDict["ebqe_vf"] = self.coefficients.fluidModel.ebqe[('velocity', 0)]
        argsDict["ebqe_vs"] = self.coefficients.fluidModel.coefficients.ebqe_velocity_solid
        argsDict["ebqe_vos"] = self.coefficients.fluidModel.coefficients.ebqe_vos
        argsDict["ebqe_rho_f"] = self.coefficients.fluidModel.coefficients.ebqe_rho
        argsDict["csrRowIndeces_u_u"] = self.csrRowIndeces[(0, 0)]
        argsDict["csrColumnOffsets_u_u"] = self.csrColumnOffsets[(0, 0)]
        argsDict["globalJacobian"] = jacobian.getCSRrepresentation()[2]
        argsDict["nExteriorElementBoundaries_global"] = self.mesh.nExteriorElementBoundaries_global
        argsDict["exteriorElementBoundariesArray"] = self.mesh.exteriorElementBoundariesArray
        argsDict["elementBoundaryElementsArray"] = self.mesh.elementBoundaryElementsArray
        argsDict["elementBoundaryLocalElementBoundariesArray"] = self.mesh.elementBoundaryLocalElementBoundariesArray
        argsDict["csrColumnOffsets_eb_u_u"] = self.csrColumnOffsets_eb[(0, 0)]
        self.presinc.calculateJacobian(argsDict)

        if self.coefficients.fixNullSpace and self.comm.rank() == 0:
            dofN = 0
            global_dofN = self.offset[0] + self.stride[0] * dofN
            for i in range(self.rowptr[global_dofN], self.rowptr[global_dofN + 1]):
                if (self.colind[i] == global_dofN):
                    self.nzval[i] = 1.0
                else:
                    self.nzval[i] = 0.0

        log("Jacobian ", level=10, data=jacobian)
        # mwf decide if this is reasonable for solver statistics
        self.nonlinear_function_jacobian_evaluations += 1
        return jacobian

    def calculateElementQuadrature(self):
        """
        Calculate the physical location and weights of the quadrature rules
        and the shape information at the quadrature points.

        This function should be called only when the mesh changes.
        """
        # self.u[0].femSpace.elementMaps.getValues(self.elementQuadraturePoints,
        #                                          self.q['x'])
        self.u[0].femSpace.elementMaps.getBasisValuesIP(
            self.u[0].femSpace.referenceFiniteElement.interpolationConditions.quadraturePointArray)
        self.u[0].femSpace.elementMaps.getBasisValuesRef(
            self.elementQuadraturePoints)
        self.u[0].femSpace.elementMaps.getBasisGradientValuesRef(
            self.elementQuadraturePoints)
        self.u[0].femSpace.getBasisValuesRef(self.elementQuadraturePoints)
        self.u[0].femSpace.getBasisGradientValuesRef(
            self.elementQuadraturePoints)
        self.coefficients.initializeElementQuadrature(
            self.timeIntegration.t, self.q)
        if self.stabilization is not None:
            self.stabilization.initializeElementQuadrature(
                self.mesh, self.timeIntegration.t, self.q)
            self.stabilization.initializeTimeIntegration(self.timeIntegration)
        if self.shockCapturing is not None:
            self.shockCapturing.initializeElementQuadrature(
                self.mesh, self.timeIntegration.t, self.q)

    def calculateElementBoundaryQuadrature(self):
        pass

    def calculateExteriorElementBoundaryQuadrature(self):
        log("initalizing ebqe vectors for post-procesing velocity")
        self.u[0].femSpace.elementMaps.getValuesGlobalExteriorTrace(self.elementBoundaryQuadraturePoints,
                                                                    self.ebqe['x'])
        self.u[0].femSpace.elementMaps.getJacobianValuesGlobalExteriorTrace(self.elementBoundaryQuadraturePoints,
                                                                            self.ebqe['inverse(J)'],
                                                                            self.ebqe['g'],
                                                                            self.ebqe['sqrt(det(g))'],
                                                                            self.ebqe['n'])
        cfemIntegrals.calculateIntegrationWeights(self.ebqe['sqrt(det(g))'],
                                                  self.elementBoundaryQuadratureWeights[('u', 0)],
                                                  self.ebqe[('dS_u', 0)])
        self.u[0].femSpace.elementMaps.getBasisValuesTraceRef(
            self.elementBoundaryQuadraturePoints)
        self.u[0].femSpace.elementMaps.getBasisGradientValuesTraceRef(
            self.elementBoundaryQuadraturePoints)
        self.u[0].femSpace.getBasisValuesTraceRef(
            self.elementBoundaryQuadraturePoints)
        self.u[0].femSpace.getBasisGradientValuesTraceRef(
            self.elementBoundaryQuadraturePoints)
        log("setting flux boundary conditions")
        self.fluxBoundaryConditionsObjectsDict = dict([(cj, FluxBoundaryConditions(self.mesh,
                                                                                   self.nElementBoundaryQuadraturePoints_elementBoundary,
                                                                                   self.ebqe[('x')],
                                                                                   self.advectiveFluxBoundaryConditionsSetterDict[cj],
                                                                                   self.diffusiveFluxBoundaryConditionsSetterDictDict[cj]))
                                                       for cj in list(self.advectiveFluxBoundaryConditionsSetterDict.keys())])
        log("initializing coefficients ebqe")
        self.coefficients.initializeGlobalExteriorElementBoundaryQuadrature(self.timeIntegration.t, self.ebqe)

    def estimate_mt(self):
        pass

    def calculateAuxiliaryQuantitiesAfterStep(self):
        pass

    def calculateSolutionAtQuadrature(self):
        pass

    def updateAfterMeshMotion(self):
        pass
