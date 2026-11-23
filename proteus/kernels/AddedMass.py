import proteus
import numpy as np
from proteus import cmeshTools
from proteus.Transport import OneLevelTransport
import os
from proteus import cfemIntegrals, Quadrature, Norms, Comm
from proteus.NonlinearSolvers import NonlinearEquation
from proteus.FemTools import (DOFBoundaryConditions,
                              FluxBoundaryConditions,
                              C0_AffineLinearOnSimplexWithNodalBasis)
from proteus.Comm import globalMax, globalSum
from proteus.Profiling import memory
from proteus.Profiling import logEvent
from proteus.Transport import OneLevelTransport
from proteus.TransportCoefficients import TC_base
from proteus.SubgridError import SGE_base
from proteus.ShockCapturing import ShockCapturing_base
from . import cAddedMass
from proteus import cArgumentsDict
from mpi4py import MPI
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
    """
    Parameters
    ----------
    nd: int
        Number of space dimensions.
    V_model: int
        Index of Navier-Stokes model.
    barycenters: array_like
        List of domain barycenters.
    flags_rigidbody: array_like
        Array of integers of length at least as long as maximum domain flag,
        with 0 at indices corresponding to flags for fixed walls, and 1 at
        indices corresponding to flags for rigid bodies.
        e.g. if only flag 2 is rigid body, and the max value of flag is 5,
        the array will look like np.array([0,0,1,0,0,0])
    """

    def __init__(self,
                 nd=2,
                 V_model=None,
                 barycenters=None,
                 flags_rigidbody=None,
                 solve_rate=0.,
                 nullSpace="ConstantNullSpace",
                 initialize=True):
        """
        TODO
        """
        self.solve_rate = solve_rate
        self.updated_global = False
        self.next_solve = 0.
        self.nullSpace = nullSpace
        self.nd = nd
        self.flowModelIndex=V_model
        self.barycenters = barycenters
        self.flags_rigidbody = flags_rigidbody
        self.variableNames=['addedMass']
        if initialize:
            self.initialize()

    def initialize(self):
        assert(self.nd in [2, 3])
        if self.nd == 2:
            sdInfo = {(0, 0): (np.array([0, 1, 2], dtype='i'),
                               np.array([0, 1], dtype='i'))}
        else:
            sdInfo = {(0, 0): (np.array([0, 1, 2, 3], dtype='i'),
                               np.array([0, 1, 2], dtype='i'))}
        TC_base.__init__(self,
                         nc=1,
                         variableNames=['am'],
                         diffusion={0: {0: {0: 'constant'}}},
                         potential={0: {0: 'u'}},
                         sparseDiffusionTensors=sdInfo,
                         useSparseDiffusion=True)
        if self.flags_rigidbody is None:
            Profiling.logEvent("Warning: flags_rigidbody was not set for"+
                               "AddedMass model, using a zero array of length"+
                               "1000, this sets all boundaries as fixed walls"+
                               "for the added mass model and might cause"+
                               " issues if the domain contains more flags")
            self.flags_rigidbody = np.zeros(1000, dtype='int32')

    def attachModels(self, modelList):
        """
        Attach the model for velocity and density to PresureIncrement model
        """
        self.flowModel = modelList[self.flowModelIndex]
        self.q_rho = self.flowModel.q['rho']

    def initializeMesh(self, mesh):
        """
        Give the TC object access to the mesh for any mesh-dependent information.
        """
        self.mesh = mesh

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
        copyInstructions = {}
        return copyInstructions

    def evaluate(self, t, c):
        if c[('u',0)].shape == self.q_rho.shape:
            c[('a', 0, 0)][..., 0] = 1.0/self.q_rho
            for i in range(1, c[('a', 0, 0)].shape[-1]):
                c[('a', 0, 0)][..., i] = c[('a', 0, 0)][..., 0]

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
        from proteus import Comm
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
                assert self.u[ci].femSpace.sameSpaceAs(self.u[0].femSpace), "to reuse_test_trial_quad all femSpaces must be the same!"
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
        self.added_mass_i = 0;
        self.Aij = np.zeros((self.coefficients.flags_rigidbody.shape[0], 6, 6), 'd')
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
        self.ebqe[('diffusiveFlux_bc_flag', 0, 0)] = np.zeros(
            (self.mesh.nExteriorElementBoundaries_global, self.nElementBoundaryQuadraturePoints_elementBoundary), 'i')
        self.ebqe[('diffusiveFlux_bc', 0, 0)] = np.zeros(
            (self.mesh.nExteriorElementBoundaries_global, self.nElementBoundaryQuadraturePoints_elementBoundary), 'd')
        self.ebqe[('diffusiveFlux', 0, 0)] = np.zeros((self.mesh.nExteriorElementBoundaries_global, self.nElementBoundaryQuadraturePoints_elementBoundary), 'd')
        self.points_elementBoundaryQuadrature = set()
        self.scalars_elementBoundaryQuadrature = set(
            [('u', ci) for ci in range(self.nc)])
        self.vectors_elementBoundaryQuadrature = set()
        self.tensors_elementBoundaryQuadrature = set()
        logEvent(memory("element and element boundary Jacobians",
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
        self.internalNodesArray = np.zeros((self.nNodes_internal,), cmeshTools.INDEX_DTYPE)
        for nI, n in enumerate(self.internalNodes):
            self.internalNodesArray[nI] = n
        #
        del self.internalNodes
        self.internalNodes = None
        logEvent("Updating local to global mappings", 2)
        self.updateLocal2Global()
        logEvent("Building time integration object", 2)
        logEvent(memory("inflowBC, internalNodes,updateLocal2Global",
                   "OneLevelTransport"), level=4)
        # mwf for interpolating subgrid error for gradients etc
        if self.stabilization and self.stabilization.usesGradientStabilization:
            self.timeIntegration = TimeIntegrationClass(
                self, integrateInterpolationPoints=True)
        else:
            self.timeIntegration = TimeIntegrationClass(self)

        if options is not None:
            self.timeIntegration.setFromOptions(options)
        logEvent(memory("TimeIntegration", "OneLevelTransport"), level=4)
        logEvent("Calculating numerical quadrature formulas", 2)
        self.calculateQuadrature()
        self.setupFieldStrides()

        comm = Comm.get()
        self.comm = comm
        if comm.size() > 1:
            assert numericalFluxType is not None and numericalFluxType.useWeakDirichletConditions, "You must use a numerical flux to apply weak boundary conditions for parallel runs"

        logEvent(memory("stride+offset", "OneLevelTransport"), level=4)
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
        # strong Dirichlet
        self.dirichletConditionsForceDOF = {0: DOFBoundaryConditions(self.u[cj].femSpace, dofBoundaryConditionsSetterDict[cj], weakDirichletConditions=False)}
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
        logEvent(memory("numericalFlux", "OneLevelTransport"), level=4)
        self.elementEffectiveDiametersArray = self.mesh.elementInnerDiametersArray
        # helper for writing out data storage
        from proteus import Archiver
        self.elementQuadratureDictionaryWriter = Archiver.XdmfWriter()
        self.elementBoundaryQuadratureDictionaryWriter = Archiver.XdmfWriter()
        self.exteriorElementBoundaryQuadratureDictionaryWriter = Archiver.XdmfWriter()
        self.globalResidualDummy = None
        compKernelFlag = 0
        self.addedMass = cAddedMass.newAddedMass(
            self.nSpace_global,
            self.nQuadraturePoints_element,
            self.u[0].femSpace.elementMaps.localFunctionSpace.dim,
            self .u[0].femSpace.referenceFiniteElement.localFunctionSpace.dim,
            self.testSpace[0].referenceFiniteElement.localFunctionSpace.dim,
            self.nElementBoundaryQuadraturePoints_elementBoundary,
            compKernelFlag)
        self.barycenters = self.coefficients.barycenters
        self.flags_rigidbody = self.coefficients.flags_rigidbody

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
        for dofN, g in list(self.dirichletConditionsForceDOF[0].DOFBoundaryConditionsDict.items()):
            # load the BC valu        # Load the unknowns into the finite element dof
            u[self.offset[0] + self.stride[0] * dofN] = g(self.dirichletConditionsForceDOF[0].DOFBoundaryPointDict[dofN], self.timeIntegration.t)
        self.setUnknowns(u)
        self.Aij[:,:,self.added_mass_i]=0.0

        argDict = cArgumentsDict.ArgumentsDict()
        argDict['mesh_trial_ref'] = self.u[0].femSpace.elementMaps.psi
        argDict['mesh_grad_trial_ref'] = self.u[0].femSpace.elementMaps.grad_psi
        argDict['mesh_dof'] = self.mesh.nodeArray
        argDict['mesh_l2g'] = self.mesh.elementNodesArray
        argDict['dV_ref'] = self.elementQuadratureWeights[('u', 0)]
        argDict['u_trial_ref'] = self.u[0].femSpace.psi
        argDict['u_grad_trial_ref'] = self.u[0].femSpace.grad_psi
        argDict['u_test_ref'] = self.u[0].femSpace.psi
        argDict['u_grad_test_ref'] = self.u[0].femSpace.grad_psi
            # element boundary
        argDict['mesh_trial_trace_ref'] = self.u[0].femSpace.elementMaps.psi_trace
        argDict['mesh_grad_trial_trace_ref'] = self.u[0].femSpace.elementMaps.grad_psi_trace
        argDict['dS_ref'] = self.elementBoundaryQuadratureWeights[('u', 0)]
        argDict['u_trial_trace_ref'] = self.u[0].femSpace.psi_trace
        argDict['u_grad_trial_trace_ref'] = self.u[0].femSpace.grad_psi_trace
        argDict['u_test_trace_ref'] = self.u[0].femSpace.psi_trace
        argDict['u_grad_test_trace_ref'] = self.u[0].femSpace.grad_psi_trace
        argDict['normal_ref'] = self.u[0].femSpace.elementMaps.boundaryNormals
        argDict['boundaryJac_ref'] = self.u[0].femSpace.elementMaps.boundaryJacobians
            # physics
        argDict['nElements_global'] = self.mesh.nElements_global
        argDict['nElementBoundaries_owned'] = self.mesh.nElementBoundaries_owned
        argDict['u_l2g'] = self.u[0].femSpace.dofMap.l2g
        argDict['u_dof'] = self.u[0].dof
        argDict['q_rho'] = self.coefficients.q_rho
        argDict['offset_u'] = self.offset[0]
        argDict['stride_u'] = self.stride[0]
        argDict['globalResidual'] = r
        argDict['nExteriorElementBoundaries_global'] = self.mesh.nExteriorElementBoundaries_global
        argDict['exteriorElementBoundariesArray'] = self.mesh.exteriorElementBoundariesArray
        argDict['elementBoundaryElementsArray'] = self.mesh.elementBoundaryElementsArray
        argDict['elementBoundaryLocalElementBoundariesArray'] = self.mesh.elementBoundaryLocalElementBoundariesArray
        argDict['elementBoundaryMaterialTypesArray'] = self.mesh.elementBoundaryUserFlags
        argDict['Aij'] = self.Aij
        argDict['added_mass_i'] = self.added_mass_i
        argDict['barycenters'] = self.barycenters
        argDict['flags_rigidbody'] = self.flags_rigidbody
        self.addedMass.calculateResidual(argDict)
        # sum of residual should be zero
        # but it is sometimes not exactly zero with certain meshes in parallel
        # hack
        comm = Comm.get().comm.tompi4py()
        if comm.size > 1:
            r_sum_local = np.sum(r[:self.mesh.nNodes_owned])
            r_sum_global = comm.reduce(r_sum_local, op=MPI.SUM, root=0)
            if comm.rank == 0:
                r[0] -= r_sum_global
                logEvent('R SUM: local: {r_sum_local} global: {r_sum_global}'.format(r_sum_local=r_sum_local, r_sum_global=r_sum_global))
            logEvent('ACCELERATION INDEX: {id}'.format(id=self.added_mass_i))
        # end hack
        for k in range(self.Aij.shape[0]):
            for j in range(self.Aij.shape[2]):
                self.Aij[k,j,self.added_mass_i] = globalSum(
                    self.Aij[k,j,self.added_mass_i])
        for i,flag in enumerate(self.flags_rigidbody):
            if flag==1:
                np.set_printoptions(precision=2, linewidth=160)
                logEvent("Added Mass Tensor for rigid body i" + repr(i))
                logEvent("Aij = \n"+str(self.Aij[i]))
        for dofN, g in list(self.dirichletConditionsForceDOF[0].DOFBoundaryConditionsDict.items()):
            r[self.offset[0] + self.stride[0] * dofN] = self.u[0].dof[dofN] - \
                g(self.dirichletConditionsForceDOF[0].DOFBoundaryPointDict[dofN], self.timeIntegration.t)
        logEvent("Global residual", level=9, data=r)
        self.nonlinear_function_evaluations += 1

    def getJacobian(self, jacobian):
        cfemIntegrals.zeroJacobian_CSR(self.nNonzerosInJacobian, jacobian)
        self.addedMass.calculateJacobian(  # element
            self.u[0].femSpace.elementMaps.psi,
            self.u[0].femSpace.elementMaps.grad_psi,
            self.mesh.nodeArray,
            self.mesh.elementNodesArray,
            self.elementQuadratureWeights[('u', 0)],
            self.u[0].femSpace.psi,
            self.u[0].femSpace.grad_psi,
            self.u[0].femSpace.psi,
            self.u[0].femSpace.grad_psi,
            # element boundary
            self.u[0].femSpace.elementMaps.psi_trace,
            self.u[0].femSpace.elementMaps.grad_psi_trace,
            self.elementBoundaryQuadratureWeights[('u', 0)],
            self.u[0].femSpace.psi_trace,
            self.u[0].femSpace.grad_psi_trace,
            self.u[0].femSpace.psi_trace,
            self.u[0].femSpace.grad_psi_trace,
            self.u[0].femSpace.elementMaps.boundaryNormals,
            self.u[0].femSpace.elementMaps.boundaryJacobians,
            self.mesh.nElements_global,
            self.u[0].femSpace.dofMap.l2g,
            self.u[0].dof,
            self.coefficients.q_rho,
            self.csrRowIndeces[(0, 0)], self.csrColumnOffsets[(0, 0)],
            jacobian.getCSRrepresentation()[2],
            self.mesh.nExteriorElementBoundaries_global,
            self.mesh.exteriorElementBoundariesArray,
            self.mesh.elementBoundaryElementsArray,
            self.mesh.elementBoundaryLocalElementBoundariesArray,
            self.csrColumnOffsets_eb[(0, 0)])
        for dofN in list(self.dirichletConditionsForceDOF[0].DOFBoundaryConditionsDict.keys()):
            global_dofN = self.offset[0] + self.stride[0] * dofN
            self.nzval[np.where(self.colind == global_dofN)] = 0.0  # column
            self.nzval[self.rowptr[global_dofN]:self.rowptr[global_dofN + 1]] = 0.0  # row
            zeroRow = True
            for i in range(self.rowptr[global_dofN], self.rowptr[global_dofN + 1]):  # row
                if (self.colind[i] == global_dofN):
                    self.nzval[i] = 1.0
                    zeroRow = False
            if zeroRow:
                raise RuntimeError("Jacobian has a zero row because sparse matrix has no diagonal entry at row " +
                                   repr(global_dofN)+". You probably need add diagonal mass or reaction term")
        logEvent("Jacobian ", level=10, data=jacobian)
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
        logEvent("initalizing ebqe vectors for post-procesing velocity")
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
        logEvent("initializing coefficients ebqe")
        self.coefficients.initializeGlobalExteriorElementBoundaryQuadrature(self.timeIntegration.t, self.ebqe)

    def estimate_mt(self):
        pass

    def calculateAuxiliaryQuantitiesAfterStep(self):
        pass

    def calculateSolutionAtQuadrature(self):
        pass

    def updateAfterMeshMotion(self):
        pass
