import proteus
from proteus.mprans.cMoveMesh2D import cMoveMesh2D_base
from proteus.mprans.cMoveMesh import cMoveMesh_base
import numpy as np
from math import sqrt
from proteus.Transport import OneLevelTransport, TC_base, NonlinearEquation
from proteus import Quadrature, FemTools, Comm, Archiver, cfemIntegrals
from proteus.Profiling import logEvent, memory
from proteus import cmeshTools
from . import cArgumentsDict
import proteus.Transport
import proteus.TransportCoefficients

class Coefficients(proteus.TransportCoefficients.TC_base):
    def __init__(self,
                 modelType_block,
                 modelParams_block,
                 g=[0.0, 0.0, -9.8],  # gravitational acceleration
                 rhow=998.2,  # kg/m^3 water density (used if pore pressures specified)
                 nd=3,
                 meIndex=0,
                 V_model=0,
                 nullSpace='NoNullSpace',
                 initialize=True):
        self.flowModelIndex = V_model
        self.modelType_block = modelType_block
        self.modelParams_block = modelParams_block
        self.g = np.array(g)
        self.gmag = sqrt(sum([gi**2 for gi in g]))
        self.rhow = rhow
        self.nd = nd
        self.firstCall = True
        self.gravityStep = True
        self.meIndex = meIndex
        self.dt_last = None
        self.dt_last_last = None
        self.solidsList = []
        self.nullSpace = nullSpace
        if self.nd == 2:
            self.variableNames = ['hx', 'hy']
        else:
            self.variableNames = ['hx', 'hy', 'hz']

        if initialize:
            self.initialize()

    def initialize(self):
        self.materialProperties = self.modelParams_block
        self.nMaterialProperties = len(self.materialProperties[-1])
        nd = self.nd
        mass = {}
        advection = {}
        diffusion = {}
        potential = {}
        reaction = {}
        hamiltonian = {}
        stress = {}
        if nd == 2:
            variableNames = ['hx', 'hy']
            mass = {0: {0: 'linear'},
                    1: {1: 'linear'}}
            reaction = {0: {0: 'constant'},
                        1: {1: 'constant'}}
            stress = {0: {0: 'linear', 1: 'linear'},
                      1: {0: 'linear', 1: 'linear'}}
            TC_base.__init__(self,
                             2,
                             mass,
                             advection,
                             diffusion,
                             potential,
                             reaction,
                             hamiltonian,
                             variableNames,
                             stress=stress)
            self.vectorComponents = [0, 1]
            self.vectorName = "displacement"
        else:
            assert(nd == 3)
            variableNames = ['hx', 'hy', 'hz']
            mass = {0: {0: 'linear'},
                    1: {1: 'linear'},
                    2: {2: 'linear'}}
            reaction = {0: {0: 'constant'},
                        1: {1: 'constant'},
                        2: {2: 'constant'}}
            stress = {0: {0: 'linear', 1: 'linear', 2: 'linear'},
                      1: {0: 'linear', 1: 'linear', 2: 'linear'},
                      2: {0: 'linear', 1: 'linear', 2: 'linear'}}
            TC_base.__init__(self,
                             3,
                             mass,
                             advection,
                             diffusion,
                             potential,
                             reaction,
                             hamiltonian,
                             variableNames,
                             stress=stress)
            self.vectorComponents = [0, 1, 2]
            self.vectorName = "displacement"

    def attachModels(self, modelList):
        self.model = modelList[self.meIndex]
        self.flowModel = modelList[self.flowModelIndex]

    def initializeElementQuadrature(self, t, cq):
        """
        Give the TC object access to the element quadrature storage
        """
        if self.firstCall:
            self.firstCall = False
        self.cq = cq
        self.bodyForce = cq['bodyForce']
        # for eN in range(self.bodyForce.shape[0]):
        #     for k in range(self.bodyForce.shape[1]):
        #         self.bodyForce[eN,k,:] = self.g

    def initializeGlobalExteriorElementBoundaryQuadrature(self, t, cebqe):
        """
        Give the TC object access to the element quadrature storage
        """
        pass

    def initializeMesh(self, mesh):
        self.mesh = mesh

    def state(self):
        """What postStep moves forward that the model's dofs do not hold: the mesh nodes (postStep adds the
        displacement to them and clears the dofs), their velocity, and the step sizes postStep keeps. A retried
        system step restores it, or the retry adds the displacement a second time."""
        mesh = getattr(self, "mesh", None)
        if mesh is None:
            return None
        st = {"nodeArray": mesh.nodeArray.copy()}
        if getattr(mesh, "nodeVelocityArray", None) is not None:
            st["nodeVelocityArray"] = mesh.nodeVelocityArray.copy()
        for name in ("dt_last", "dt_last_last"):
            if getattr(self, name) is not None:
                st[name] = float(getattr(self, name))
        return st

    def set_state(self, state):
        self.mesh.nodeArray[:] = state["nodeArray"]
        if "nodeVelocityArray" in state:
            self.mesh.nodeVelocityArray[:] = state["nodeVelocityArray"]
        for name in ("dt_last", "dt_last_last"):
            setattr(self, name, float(state[name]) if name in state else None)

    def postStep(self, t, firstStep=False):
        self.model.postStep()
        self.mesh.nodeArray[:, 0] += self.model.u[0].dof
        self.mesh.nodeVelocityArray[:, 0] = self.model.u[0].dof
        self.model.u[0].dof[:] = 0.0
        self.mesh.nodeArray[:, 1] += self.model.u[1].dof
        self.mesh.nodeVelocityArray[:, 1] = self.model.u[1].dof
        self.model.u[1].dof[:] = 0.0
        if self.nd == 3:
            self.mesh.nodeArray[:, 2] += self.model.u[2].dof
            self.mesh.nodeVelocityArray[:, 2] = self.model.u[2].dof
            self.model.u[2].dof[:] = 0.0
        if self.dt_last is None:
            dt = self.model.timeIntegration.dt
        else:
            dt = self.dt_last
        self.mesh.nodeVelocityArray /= dt
        #this is needed for proper restarting
        if(self.dt_last is not None):
            self.dt_last_last = self.dt_last

        self.dt_last = self.model.timeIntegration.dt

        #update nodal/element diameters:
        #TODO: unclear if this needs to apply to all mesh types, look into parallel communication of geometric info
        #if self.nd == 2:  
        #   #cmeshTools.computeGeometricInfo_triangle(self.mesh.cmesh)
        #   cmeshTools.computeGeometricInfo_triangle(self.mesh.subdomainMesh.cmesh)
        #if self.nd == 3:  
        #   cmeshTools.computeGeometricInfo_tetrahedron(self.mesh.subdomainMesh.cmesh)

        copyInstructions = {'clear_uList': True}
        return copyInstructions

    def preStep(self, t, firstStep=False):
        logEvent("MoveMesh preStep")
        self.model.preStep()
        for s in self.solidsList:
            logEvent("Calling step on solids")
            logEvent(repr(s))
            s.step()

    def evaluate(self, t, c):
        pass


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
                 stressFluxBoundaryConditionsSetterDict=None,
                 stabilization=None,
                 shockCapturing=None,
                 conservativeFluxDict=None,
                 numericalFluxType=None,
                 TimeIntegrationClass=None,
                 massLumping=False,
                 reactionLumping=False,
                 options=None,
                 name='Plasticity',
                 reuse_trial_and_test_quadrature=True,
                 sd=True,
                 movingDomain=False,
                 bdyNullSpace=False):
        #
        # set the objects describing the method and boundary conditions
        #
        self.bdyNullSpace=bdyNullSpace
        self.moveCalls = 0
        self.movingDomain = movingDomain
        self.tLast_mesh = None
        self.bdyNullSpace = bdyNullSpace
        #
        # cek todo clean up these flags in the optimized version
        self.bcsTimeDependent = options.bcsTimeDependent
        self.bcsSet = False
        self.name = name
        self.sd = sd
        self.lowmem = True
        self.timeTerm = True  # allow turning off  the  time derivative
        self.testIsTrial = True
        self.phiTrialIsTrial = True
        self.u = uDict
        self.Hess = False
        if isinstance(self.u[0].femSpace, FemTools.C0_AffineQuadraticOnSimplexWithNodalBasis):
            self.Hess = True
        self.ua = {}  # analytical solutions
        self.phi = phiDict
        self.dphi = {}
        self.matType = matType
        # mwf try to reuse test and trial information across components if spaces are the same
        self.reuse_test_trial_quadrature = reuse_trial_and_test_quadrature  # True#False
        if self.reuse_test_trial_quadrature:
            for ci in range(1, coefficients.nc):
                assert self.u[ci].femSpace.__class__.__name__ == self.u[0].femSpace.__class__.__name__, "to reuse_test_trial_quad all femSpaces must be the same!"
        # Simplicial Mesh
        self.mesh = self.u[0].femSpace.mesh  # assume the same mesh for  all components for now
        self.testSpace = testSpaceDict
        self.dirichletConditions = dofBoundaryConditionsDict
        self.dirichletNodeSetList = None  # explicit Dirichlet  conditions for now, no Dirichlet BC constraints
        self.coefficients = coefficients
        self.coefficients.initializeMesh(self.mesh)
        self.nc = self.coefficients.nc
        self.stabilization = stabilization
        self.shockCapturing = shockCapturing
        self.conservativeFlux = conservativeFluxDict  # no velocity post-processing for now
        self.fluxBoundaryConditions = fluxBoundaryConditionsDict
        self.stressFluxBoundaryConditionsSetterDict = stressFluxBoundaryConditionsSetterDict
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
            self.elementBoundaryIntegrals[ci] = ((self.conservativeFlux is not None) or
                                                 (numericalFluxType is not None) or
                                                 (self.fluxBoundaryConditions[ci] == 'outFlow') or
                                                 (self.fluxBoundaryConditions[ci] == 'mixedFlow') or
                                                 (self.fluxBoundaryConditions[ci] == 'setFlow'))
        #
        # calculate some dimensions
        #
        self.nSpace_global = self.u[0].femSpace.nSpace_global  # assume same space dim for all variables
        self.nDOF_trial_element = [u_j.femSpace.max_nDOF_element for u_j in list(self.u.values())]
        self.nDOF_phi_trial_element = [phi_k.femSpace.max_nDOF_element for phi_k in list(self.phi.values())]
        self.n_phi_ip_element = [phi_k.femSpace.referenceFiniteElement.interpolationConditions.nQuadraturePoints for phi_k in list(self.phi.values())]
        self.nDOF_test_element = [femSpace.max_nDOF_element for femSpace in list(self.testSpace.values())]
        self.nFreeDOF_global = [dc.nFreeDOF_global for dc in list(self.dirichletConditions.values())]
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
                        elementQuadratureDict[('stab',) + I[1:]] = elementQuadrature[I]
                    else:
                        elementQuadratureDict[('stab',) + I[1:]] = elementQuadrature['default']
                else:
                    elementQuadratureDict[('stab',) + I[1:]] = elementQuadrature
        if self.shockCapturing is not None:
            for ci in self.shockCapturing.components:
                if elemQuadIsDict:
                    if ('numDiff', ci, ci) in elementQuadrature:
                        elementQuadratureDict[('numDiff', ci, ci)] = elementQuadrature[('numDiff', ci, ci)]
                    else:
                        elementQuadratureDict[('numDiff', ci, ci)] = elementQuadrature['default']
                else:
                    elementQuadratureDict[('numDiff', ci, ci)] = elementQuadrature
        if massLumping:
            for ci in list(self.coefficients.mass.keys()):
                elementQuadratureDict[('m', ci)] = Quadrature.SimplexLobattoQuadrature(self.nSpace_global, 1)
            for I in self.coefficients.elementIntegralKeys:
                elementQuadratureDict[('stab',) + I[1:]] = Quadrature.SimplexLobattoQuadrature(self.nSpace_global, 1)
        if reactionLumping:
            for ci in list(self.coefficients.mass.keys()):
                elementQuadratureDict[('r', ci)] = Quadrature.SimplexLobattoQuadrature(self.nSpace_global, 1)
            for I in self.coefficients.elementIntegralKeys:
                elementQuadratureDict[('stab',) + I[1:]] = Quadrature.SimplexLobattoQuadrature(self.nSpace_global, 1)
        elementBoundaryQuadratureDict = {}
        if isinstance(elementBoundaryQuadrature, dict):  # set terms manually
            for I in self.coefficients.elementBoundaryIntegralKeys:
                if I in elementBoundaryQuadrature:
                    elementBoundaryQuadratureDict[I] = elementBoundaryQuadrature[I]
                else:
                    elementBoundaryQuadratureDict[I] = elementBoundaryQuadrature['default']
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
        self.nQuadraturePoints_global = self.nQuadraturePoints_element * self.mesh.nElements_global
        #
        # Repeat the same thing for the element boundary quadrature
        #
        (self.elementBoundaryQuadraturePoints,
         self.elementBoundaryQuadratureWeights,
         self.elementBoundaryQuadratureRuleIndeces) = Quadrature.buildUnion(elementBoundaryQuadratureDict)
        self.nElementBoundaryQuadraturePoints_elementBoundary = self.elementBoundaryQuadraturePoints.shape[0]
        self.nElementBoundaryQuadraturePoints_global = (self.mesh.nElements_global *
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
        self.ebqe['x'] = np.zeros((self.mesh.nExteriorElementBoundaries_global, self.nElementBoundaryQuadraturePoints_elementBoundary, 3), 'd')
        self.q['bodyForce'] = np.zeros((self.mesh.nElements_global, self.nQuadraturePoints_element, self.nSpace_global), 'd')
        self.ebqe[('u', 0)] = np.zeros((self.mesh.nExteriorElementBoundaries_global, self.nElementBoundaryQuadraturePoints_elementBoundary), 'd')
        self.ebqe[('u', 1)] = np.zeros((self.mesh.nExteriorElementBoundaries_global, self.nElementBoundaryQuadraturePoints_elementBoundary), 'd')
        self.ebqe[('u', 2)] = np.zeros((self.mesh.nExteriorElementBoundaries_global, self.nElementBoundaryQuadraturePoints_elementBoundary), 'd')
        self.ebqe[('stressFlux_bc_flag', 0)] = np.zeros(
            (self.mesh.nExteriorElementBoundaries_global, self.nElementBoundaryQuadraturePoints_elementBoundary), 'i')
        self.ebqe[('stressFlux_bc_flag', 1)] = np.zeros(
            (self.mesh.nExteriorElementBoundaries_global, self.nElementBoundaryQuadraturePoints_elementBoundary), 'i')
        self.ebqe[('stressFlux_bc_flag', 2)] = np.zeros(
            (self.mesh.nExteriorElementBoundaries_global, self.nElementBoundaryQuadraturePoints_elementBoundary), 'i')
        self.ebqe[('stressFlux_bc', 0)] = np.zeros((self.mesh.nExteriorElementBoundaries_global, self.nElementBoundaryQuadraturePoints_elementBoundary), 'd')
        self.ebqe[('stressFlux_bc', 1)] = np.zeros((self.mesh.nExteriorElementBoundaries_global, self.nElementBoundaryQuadraturePoints_elementBoundary), 'd')
        self.ebqe[('stressFlux_bc', 2)] = np.zeros((self.mesh.nExteriorElementBoundaries_global, self.nElementBoundaryQuadraturePoints_elementBoundary), 'd')
        self.points_elementBoundaryQuadrature = set()
        self.scalars_elementBoundaryQuadrature = set([('u', ci) for ci in range(self.nc)])
        self.vectors_elementBoundaryQuadrature = set()
        self.tensors_elementBoundaryQuadrature = set()
        #
        # show quadrature
        #
        logEvent("Dumping quadrature shapes for model %s" % self.name, level=9)
        logEvent("Element quadrature array (q)", level=9)
        for (k, v) in list(self.q.items()):
            logEvent(str((k, v.shape)), level=9)
        logEvent("Element boundary quadrature (ebq)", level=9)
        for (k, v) in list(self.ebq.items()):
            logEvent(str((k, v.shape)), level=9)
        logEvent("Global element boundary quadrature (ebq_global)", level=9)
        for (k, v) in list(self.ebq_global.items()):
            logEvent(str((k, v.shape)), level=9)
        logEvent("Exterior element boundary quadrature (ebqe)", level=9)
        for (k, v) in list(self.ebqe.items()):
            logEvent(str((k, v.shape)), level=9)
        logEvent("Interpolation points for nonlinear diffusion potential (phi_ip)", level=9)
        for (k, v) in list(self.phi_ip.items()):
            logEvent(str((k, v.shape)), level=9)
        #
        # allocate residual and Jacobian storage
        #
        self.elementResidual = [np.zeros(
            (self.mesh.nElements_global,
             self.nDOF_test_element[ci]),
            'd') for ci in range(self.nc)]
        self.elementSpatialResidual = [np.zeros(
            (self.mesh.nElements_global,
             self.nDOF_test_element[ci]),
            'd') for ci in range(self.nc)]
        self.inflowBoundaryBC = {}
        self.inflowBoundaryBC_values = {}
        self.inflowFlux = {}
        for cj in range(self.nc):
            self.inflowBoundaryBC[cj] = np.zeros((self.mesh.nExteriorElementBoundaries_global,), 'i')
            self.inflowBoundaryBC_values[cj] = np.zeros((self.mesh.nExteriorElementBoundaries_global, self.nDOF_trial_element[cj]), 'd')
            self.inflowFlux[cj] = np.zeros((self.mesh.nExteriorElementBoundaries_global, self.nElementBoundaryQuadraturePoints_elementBoundary), 'd')
        self.internalNodes = set(range(self.mesh.nNodes_global))
        # identify the internal nodes this is ought to be in mesh
        # \todo move this to mesh
        for ebNE in range(self.mesh.nExteriorElementBoundaries_global):
            ebN = self.mesh.exteriorElementBoundariesArray[ebNE]
            eN_global = self.mesh.elementBoundaryElementsArray[ebN, 0]
            ebN_element = self.mesh.elementBoundaryLocalElementBoundariesArray[ebN, 0]
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
        logEvent("Updating local to global mappings", 2)
        self.updateLocal2Global()
        logEvent("Building time integration object", 2)
        logEvent(memory("inflowBC, internalNodes,updateLocal2Global", "OneLevelTransport"), level=4)
        # mwf for interpolating subgrid error for gradients etc
        if self.stabilization and self.stabilization.usesGradientStabilization:
            self.timeIntegration = TimeIntegrationClass(self, integrateInterpolationPoints=True)
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
                self.numericalFlux = numericalFluxType(self,
                                                       dofBoundaryConditionsSetterDict,
                                                       advectiveFluxBoundaryConditionsSetterDict,
                                                       diffusiveFluxBoundaryConditionsSetterDictDict)
            else:
                self.numericalFlux = numericalFluxType(self,
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
                for k in range(self.nElementBoundaryQuadraturePoints_elementBoundary):
                    self.ebq_global['penalty'][ebN, k] = self.numericalFlux.penalty_constant/(self.mesh.elementBoundaryDiametersArray[ebN]**self.numericalFlux.penalty_power)
        # penalty term
        # cek move  to Numerical flux initialization
        if 'penalty' in self.ebqe:
            for ebNE in range(self.mesh.nExteriorElementBoundaries_global):
                ebN = self.mesh.exteriorElementBoundariesArray[ebNE]
                for k in range(self.nElementBoundaryQuadraturePoints_elementBoundary):
                    self.ebqe['penalty'][ebNE, k] = self.numericalFlux.penalty_constant/self.mesh.elementBoundaryDiametersArray[ebN]**self.numericalFlux.penalty_power
        logEvent(memory("numericalFlux", "OneLevelTransport"), level=4)
        self.elementEffectiveDiametersArray = self.mesh.elementInnerDiametersArray
        # use post processing tools to get conservative fluxes, None by default
        # helper for writing out data storage
        import proteus.Archiver
        self.elementQuadratureDictionaryWriter = Archiver.XdmfWriter()
        self.elementBoundaryQuadratureDictionaryWriter = Archiver.XdmfWriter()
        self.exteriorElementBoundaryQuadratureDictionaryWriter = Archiver.XdmfWriter()
        for ci, sbcObject in list(self.stressFluxBoundaryConditionsObjectsDict.items()):
            self.ebqe[('stressFlux_bc_flag', ci)] = np.zeros(self.ebqe[('stressFlux_bc', ci)].shape, 'i')
            for t, g in list(sbcObject.stressFluxBoundaryConditionsDict.items()):
                self.ebqe[('stressFlux_bc', ci)][t[0], t[1]] = g(self.ebqe[('x')][t[0], t[1]], self.timeIntegration.t)
                self.ebqe[('stressFlux_bc_flag', ci)][t[0], t[1]] = 1
        self.numericalFlux.setDirichletValues(self.ebqe)
        if self.mesh.nodeVelocityArray is None:
            self.mesh.nodeVelocityArray = np.zeros(self.mesh.nodeArray.shape, 'd')
        compKernelFlag = 0
        if self.nSpace_global == 2:
            import copy
            self.u[2] = self.u[1].copy()
            self.u[2].name = 'hz'
            self.offset.append(self.offset[1])
            self.stride.append(self.stride[1])
            self.numericalFlux.isDOFBoundary[2] = self.numericalFlux.isDOFBoundary[1].copy()
            self.numericalFlux.ebqe[('u', 2)] = self.numericalFlux.ebqe[('u', 1)].copy()
            logEvent("calling cMoveMesh2D_base ctor")
            self.moveMesh = cMoveMesh2D_base(self.nSpace_global,
                                             self.nQuadraturePoints_element,
                                             self.u[0].femSpace.elementMaps.localFunctionSpace.dim,
                                             self.u[0].femSpace.referenceFiniteElement.localFunctionSpace.dim,
                                             self.testSpace[0].referenceFiniteElement.localFunctionSpace.dim,
                                             self.nElementBoundaryQuadraturePoints_elementBoundary,
                                             compKernelFlag)
        else:
            logEvent("calling cMoveMesh_base ctor")
            self.moveMesh = cMoveMesh_base(self.nSpace_global,
                                           self.nQuadraturePoints_element,
                                           self.u[0].femSpace.elementMaps.localFunctionSpace.dim,
                                           self.u[0].femSpace.referenceFiniteElement.localFunctionSpace.dim,
                                           self.testSpace[0].referenceFiniteElement.localFunctionSpace.dim,
                                           self.nElementBoundaryQuadraturePoints_elementBoundary,
                                           compKernelFlag)

        self.disp0 = np.zeros(self.nSpace_global, 'd')
        self.disp1 = np.zeros(self.nSpace_global, 'd')
        self.vel0 = np.zeros(self.nSpace_global, 'd')
        self.vel1 = np.zeros(self.nSpace_global, 'd')
        self.rot0 = np.eye(self.nSpace_global, dtype=float)
        self.rot1 = np.eye(self.nSpace_global, dtype=float)
        self.angVel0 = np.zeros(self.nSpace_global, 'd')
        self.angVel1 = np.zeros(self.nSpace_global, 'd')

        self.forceStrongConditions = True  # False#True
        self.dirichletConditionsForceDOF = {}
        if self.forceStrongConditions:
            for cj in range(self.nc):
                self.dirichletConditionsForceDOF[cj] = FemTools.DOFBoundaryConditions(
                    self.u[cj].femSpace, dofBoundaryConditionsSetterDict[cj], weakDirichletConditions=False)
        from proteus import PostProcessingTools
        self.velocityPostProcessor = PostProcessingTools.VelocityPostProcessingChooser(self)
        logEvent(memory("velocity postprocessor", "OneLevelTransport"), level=4)

    def getResidual(self, u, r):
        """
        Calculate the element residuals and add in to the global residual
        """
        # Load the unknowns into the finite element dof
        self.timeIntegration.calculateCoefs()
        self.timeIntegration.calculateU(u)
        self.setUnknowns(self.timeIntegration.u)
        if self.bcsTimeDependent or not self.bcsSet:
            self.bcsSet = True
            # Dirichlet boundary conditions
            self.numericalFlux.setDirichletValues(self.ebqe)
            # Flux boundary conditions
            for ci, fbcObject in list(self.stressFluxBoundaryConditionsObjectsDict.items()):
                for t, g in list(fbcObject.stressFluxBoundaryConditionsDict.items()):
                    self.ebqe[('stressFlux_bc', ci)][t[0], t[1]] = g(self.ebqe[('x')][t[0], t[1]], self.timeIntegration.t)
                    self.ebqe[('stressFlux_bc_flag', ci)][t[0], t[1]] = 1
        r.fill(0.0)
        self.elementResidual[0].fill(0.0)
        self.elementResidual[1].fill(0.0)
        if self.nSpace_global == 3:
            self.elementResidual[2].fill(0.0)
        if self.forceStrongConditions:
            for cj in range(self.nc):
                for dofN, g in list(self.dirichletConditionsForceDOF[cj].DOFBoundaryConditionsDict.items()):
                    self.u[cj].dof[dofN] = g(self.dirichletConditionsForceDOF[cj].DOFBoundaryPointDict[dofN], self.timeIntegration.t)
        argsDict = cArgumentsDict.ArgumentsDict()
        argsDict["mesh_trial_ref"] = self.u[0].femSpace.elementMaps.psi
        argsDict["mesh_grad_trial_ref"] = self.u[0].femSpace.elementMaps.grad_psi
        argsDict["mesh_dof"] = self.mesh.nodeArray
        argsDict["mesh_l2g"] = self.mesh.elementNodesArray
        argsDict["dV_ref"] = self.elementQuadratureWeights[('u', 0)]
        argsDict["disp_trial_ref"] = self.u[0].femSpace.psi
        argsDict["disp_grad_trial_ref"] = self.u[0].femSpace.grad_psi
        argsDict["disp_test_ref"] = self.u[0].femSpace.psi
        argsDict["disp_grad_test_ref"] = self.u[0].femSpace.grad_psi
        argsDict["mesh_trial_trace_ref"] = self.u[0].femSpace.elementMaps.psi_trace
        argsDict["mesh_grad_trial_trace_ref"] = self.u[0].femSpace.elementMaps.grad_psi_trace
        argsDict["dS_ref"] = self.elementBoundaryQuadratureWeights[('u', 0)]
        argsDict["disp_trial_trace_ref"] = self.u[0].femSpace.psi_trace
        argsDict["disp_grad_trial_trace_ref"] = self.u[0].femSpace.grad_psi_trace
        argsDict["disp_test_trace_ref"] = self.u[0].femSpace.psi_trace
        argsDict["disp_grad_test_trace_ref"] = self.u[0].femSpace.grad_psi_trace
        argsDict["normal_ref"] = self.u[0].femSpace.elementMaps.boundaryNormals
        argsDict["boundaryJac_ref"] = self.u[0].femSpace.elementMaps.boundaryJacobians
        argsDict["nElements_global"] = self.mesh.nElements_global
        argsDict["materialTypes"] = self.mesh.elementMaterialTypes
        argsDict["nMaterialProperties"] = self.coefficients.nMaterialProperties
        argsDict["materialProperties"] = self.coefficients.materialProperties
        argsDict["disp_l2g"] = self.u[0].femSpace.dofMap.l2g
        argsDict["u_dof"] = self.u[0].dof
        argsDict["v_dof"] = self.u[1].dof
        argsDict["w_dof"] = self.u[2].dof
        argsDict["bodyForce"] = self.coefficients.bodyForce
        argsDict["offset_u"] = self.offset[0]
        argsDict["offset_v"] = self.offset[1]
        argsDict["offset_w"] = self.offset[2]
        argsDict["stride_u"] = self.stride[0]
        argsDict["stride_v"] = self.stride[1]
        argsDict["stride_w"] = self.stride[2]
        argsDict["globalResidual"] = r
        argsDict["nExteriorElementBoundaries_global"] = self.mesh.nExteriorElementBoundaries_global
        argsDict["exteriorElementBoundariesArray"] = self.mesh.exteriorElementBoundariesArray
        argsDict["elementBoundaryElementsArray"] = self.mesh.elementBoundaryElementsArray
        argsDict["elementBoundaryLocalElementBoundariesArray"] = self.mesh.elementBoundaryLocalElementBoundariesArray
        argsDict["isDOFBoundary_u"] = self.numericalFlux.isDOFBoundary[0]
        argsDict["isDOFBoundary_v"] = self.numericalFlux.isDOFBoundary[1]
        argsDict["isDOFBoundary_w"] = self.numericalFlux.isDOFBoundary[2]
        argsDict["isStressFluxBoundary_u"] = self.ebqe[('stressFlux_bc_flag', 0)]
        argsDict["isStressFluxBoundary_v"] = self.ebqe[('stressFlux_bc_flag', 1)]
        argsDict["isStressFluxBoundary_w"] = self.ebqe[('stressFlux_bc_flag', 2)]
        argsDict["ebqe_bc_u_ext"] = self.numericalFlux.ebqe[('u', 0)]
        argsDict["ebqe_bc_v_ext"] = self.numericalFlux.ebqe[('u', 1)]
        argsDict["ebqe_bc_w_ext"] = self.numericalFlux.ebqe[('u', 2)]
        argsDict["ebqe_bc_stressFlux_u_ext"] = self.ebqe[('stressFlux_bc', 0)]
        argsDict["ebqe_bc_stressFlux_v_ext"] = self.ebqe[('stressFlux_bc', 1)]
        argsDict["ebqe_bc_stressFlux_w_ext"] = self.ebqe[('stressFlux_bc', 2)]
        self.moveMesh.calculateResidual(argsDict)
        if self.forceStrongConditions:
            for cj in range(self.nc):
                for dofN, g in list(self.dirichletConditionsForceDOF[cj].DOFBoundaryConditionsDict.items()):
                    r[self.offset[cj] + self.stride[cj] * dofN] = self.u[cj].dof[dofN] - \
                        g(self.dirichletConditionsForceDOF[cj].DOFBoundaryPointDict[dofN], self.timeIntegration.t)
        logEvent("Global residual", level=9, data=r)
        self.nonlinear_function_evaluations += 1

    def getJacobian(self, jacobian):
        cfemIntegrals.zeroJacobian_CSR(self.nNonzerosInJacobian,
                                       jacobian)
        if self.nSpace_global == 2:
            self.csrRowIndeces[(0, 2)] = self.csrRowIndeces[(0, 1)]
            self.csrColumnOffsets[(0, 2)] = self.csrColumnOffsets[(0, 1)]
            self.csrRowIndeces[(1, 2)] = self.csrRowIndeces[(0, 1)]
            self.csrColumnOffsets[(1, 2)] = self.csrColumnOffsets[(0, 1)]
            self.csrRowIndeces[(2, 0)] = self.csrRowIndeces[(1, 0)]
            self.csrColumnOffsets[(2, 0)] = self.csrColumnOffsets[(1, 0)]
            self.csrRowIndeces[(2, 1)] = self.csrRowIndeces[(1, 0)]
            self.csrColumnOffsets[(2, 1)] = self.csrColumnOffsets[(1, 0)]
            self.csrRowIndeces[(2, 2)] = self.csrRowIndeces[(1, 0)]
            self.csrColumnOffsets[(2, 2)] = self.csrColumnOffsets[(1, 0)]
            self.csrColumnOffsets_eb[(0, 2)] = self.csrColumnOffsets[(0, 1)]
            self.csrColumnOffsets_eb[(1, 2)] = self.csrColumnOffsets[(0, 1)]
            self.csrColumnOffsets_eb[(2, 2)] = self.csrColumnOffsets[(0, 1)]
            self.csrColumnOffsets_eb[(2, 0)] = self.csrColumnOffsets[(0, 1)]
            self.csrColumnOffsets_eb[(2, 1)] = self.csrColumnOffsets[(0, 1)]
            self.csrColumnOffsets_eb[(2, 2)] = self.csrColumnOffsets[(0, 1)]
        argsDict = cArgumentsDict.ArgumentsDict()
        argsDict["mesh_trial_ref"] = self.u[0].femSpace.elementMaps.psi
        argsDict["mesh_grad_trial_ref"] = self.u[0].femSpace.elementMaps.grad_psi
        argsDict["mesh_dof"] = self.mesh.nodeArray
        argsDict["mesh_l2g"] = self.mesh.elementNodesArray
        argsDict["dV_ref"] = self.elementQuadratureWeights[('u', 0)]
        argsDict["disp_trial_ref"] = self.u[0].femSpace.psi
        argsDict["disp_grad_trial_ref"] = self.u[0].femSpace.grad_psi
        argsDict["disp_test_ref"] = self.u[0].femSpace.psi
        argsDict["disp_grad_test_ref"] = self.u[0].femSpace.grad_psi
        argsDict["mesh_trial_trace_ref"] = self.u[0].femSpace.elementMaps.psi_trace
        argsDict["mesh_grad_trial_trace_ref"] = self.u[0].femSpace.elementMaps.grad_psi_trace
        argsDict["dS_ref"] = self.elementBoundaryQuadratureWeights[('u', 0)]
        argsDict["disp_trial_trace_ref"] = self.u[0].femSpace.psi_trace
        argsDict["disp_grad_trial_trace_ref"] = self.u[0].femSpace.grad_psi_trace
        argsDict["disp_test_trace_ref"] = self.u[0].femSpace.psi_trace
        argsDict["disp_grad_test_trace_ref"] = self.u[0].femSpace.grad_psi_trace
        argsDict["normal_ref"] = self.u[0].femSpace.elementMaps.boundaryNormals
        argsDict["boundaryJac_ref"] = self.u[0].femSpace.elementMaps.boundaryJacobians
        argsDict["nElements_global"] = self.mesh.nElements_global
        argsDict["materialTypes"] = self.mesh.elementMaterialTypes
        argsDict["nMaterialProperties"] = self.coefficients.nMaterialProperties
        argsDict["materialProperties"] = self.coefficients.materialProperties
        argsDict["disp_l2g"] = self.u[0].femSpace.dofMap.l2g
        argsDict["u_dof"] = self.u[0].dof
        argsDict["v_dof"] = self.u[1].dof
        argsDict["w_dof"] = self.u[2].dof
        argsDict["bodyForce"] = self.coefficients.bodyForce
        argsDict["csrRowIndeces_u_u"] = self.csrRowIndeces[(0, 0)]
        argsDict["csrColumnOffsets_u_u"] = self.csrColumnOffsets[(0, 0)]
        argsDict["csrRowIndeces_u_v"] = self.csrRowIndeces[(0, 1)]
        argsDict["csrColumnOffsets_u_v"] = self.csrColumnOffsets[(0, 1)]
        argsDict["csrRowIndeces_u_w"] = self.csrRowIndeces[(0, 2)]
        argsDict["csrColumnOffsets_u_w"] = self.csrColumnOffsets[(0, 2)]
        argsDict["csrRowIndeces_v_u"] = self.csrRowIndeces[(1, 0)]
        argsDict["csrColumnOffsets_v_u"] = self.csrColumnOffsets[(1, 0)]
        argsDict["csrRowIndeces_v_v"] = self.csrRowIndeces[(1, 1)]
        argsDict["csrColumnOffsets_v_v"] = self.csrColumnOffsets[(1, 1)]
        argsDict["csrRowIndeces_v_w"] = self.csrRowIndeces[(1, 2)]
        argsDict["csrColumnOffsets_v_w"] = self.csrColumnOffsets[(1, 2)]
        argsDict["csrRowIndeces_w_u"] = self.csrRowIndeces[(2, 0)]
        argsDict["csrColumnOffsets_w_u"] = self.csrColumnOffsets[(2, 0)]
        argsDict["csrRowIndeces_w_v"] = self.csrRowIndeces[(2, 1)]
        argsDict["csrColumnOffsets_w_v"] = self.csrColumnOffsets[(2, 1)]
        argsDict["csrRowIndeces_w_w"] = self.csrRowIndeces[(2, 2)]
        argsDict["csrColumnOffsets_w_w"] = self.csrColumnOffsets[(2, 2)]
        argsDict["globalJacobian"] = jacobian.getCSRrepresentation()[2]
        argsDict["nExteriorElementBoundaries_global"] = self.mesh.nExteriorElementBoundaries_global
        argsDict["exteriorElementBoundariesArray"] = self.mesh.exteriorElementBoundariesArray
        argsDict["elementBoundaryElementsArray"] = self.mesh.elementBoundaryElementsArray
        argsDict["elementBoundaryLocalElementBoundariesArray"] = self.mesh.elementBoundaryLocalElementBoundariesArray
        argsDict["isDOFBoundary_u"] = self.numericalFlux.isDOFBoundary[0]
        argsDict["isDOFBoundary_v"] = self.numericalFlux.isDOFBoundary[1]
        argsDict["isDOFBoundary_w"] = self.numericalFlux.isDOFBoundary[2]
        argsDict["isStressFluxBoundary_u"] = self.ebqe[('stressFlux_bc_flag', 0)]
        argsDict["isStressFluxBoundary_v"] = self.ebqe[('stressFlux_bc_flag', 1)]
        argsDict["isStressFluxBoundary_w"] = self.ebqe[('stressFlux_bc_flag', 2)]
        argsDict["csrColumnOffsets_eb_u_u"] = self.csrColumnOffsets_eb[(0, 0)]
        argsDict["csrColumnOffsets_eb_u_v"] = self.csrColumnOffsets_eb[(0, 1)]
        argsDict["csrColumnOffsets_eb_u_w"] = self.csrColumnOffsets_eb[(0, 2)]
        argsDict["csrColumnOffsets_eb_v_u"] = self.csrColumnOffsets_eb[(1, 0)]
        argsDict["csrColumnOffsets_eb_v_v"] = self.csrColumnOffsets_eb[(1, 1)]
        argsDict["csrColumnOffsets_eb_v_w"] = self.csrColumnOffsets_eb[(1, 2)]
        argsDict["csrColumnOffsets_eb_w_u"] = self.csrColumnOffsets_eb[(2, 0)]
        argsDict["csrColumnOffsets_eb_w_v"] = self.csrColumnOffsets_eb[(2, 1)]
        argsDict["csrColumnOffsets_eb_w_w"] = self.csrColumnOffsets_eb[(2, 2)]
        self.moveMesh.calculateJacobian(argsDict)
        # Load the Dirichlet conditions directly into residual
        if self.forceStrongConditions:
            scaling = 1.0  # probably want to add some scaling to match non-dirichlet diagonals in linear system
            for cj in range(self.nc):
                for dofN in list(self.dirichletConditionsForceDOF[cj].DOFBoundaryConditionsDict.keys()):
                    global_dofN = self.offset[cj] + self.stride[cj] * dofN
                    for i in range(self.rowptr[global_dofN], self.rowptr[global_dofN + 1]):
                        if (self.colind[i] == global_dofN):
                            self.nzval[i] = scaling
                        else:
                            self.nzval[i] = 0.0
        logEvent("Jacobian ", level=10, data=jacobian)
        # mwf decide if this is reasonable for solver statistics
        self.nonlinear_function_jacobian_evaluations += 1
        # jacobian.fwrite("jacobian_p"+`self.nonlinear_function_jacobian_evaluations`)
        return jacobian

    def calculateElementQuadrature(self):
        """
        Calculate the physical location and weights of the quadrature rules
        and the shape information at the quadrature points.

        This function should be called only when the mesh changes.
        """
        self.u[0].femSpace.elementMaps.getBasisValuesRef(self.elementQuadraturePoints)
        self.u[0].femSpace.elementMaps.getBasisGradientValuesRef(self.elementQuadraturePoints)
        self.u[0].femSpace.getBasisValuesRef(self.elementQuadraturePoints)
        self.u[0].femSpace.getBasisGradientValuesRef(self.elementQuadraturePoints)
        self.coefficients.initializeElementQuadrature(self.timeIntegration.t, self.q)

    def calculateElementBoundaryQuadrature(self):
        """
        Calculate the physical location and weights of the quadrature rules
        and the shape information at the quadrature points on element boundaries.

        This function should be called only when the mesh changes.
        """
        pass

    def calculateExteriorElementBoundaryQuadrature(self):
        """
        Calculate the physical location and weights of the quadrature rules
        and the shape information at the quadrature points on global element boundaries.

        This function should be called only when the mesh changes.
        """
        #
        # get physical locations of element boundary quadrature points
        #
        # assume all components live on the same mesh
        self.u[0].femSpace.elementMaps.getBasisValuesTraceRef(self.elementBoundaryQuadraturePoints)
        self.u[0].femSpace.elementMaps.getBasisGradientValuesTraceRef(self.elementBoundaryQuadraturePoints)
        self.u[0].femSpace.getBasisValuesTraceRef(self.elementBoundaryQuadraturePoints)
        self.u[0].femSpace.getBasisGradientValuesTraceRef(self.elementBoundaryQuadraturePoints)
        self.u[0].femSpace.elementMaps.getValuesGlobalExteriorTrace(self.elementBoundaryQuadraturePoints,
                                                                    self.ebqe['x'])
        self.stressFluxBoundaryConditionsObjectsDict = dict([(cj, FemTools.FluxBoundaryConditions(self.mesh,
                                                                                                  self.nElementBoundaryQuadraturePoints_elementBoundary,
                                                                                                  self.ebqe[('x')],
                                                                                                  self.stressFluxBoundaryConditionsSetterDict[cj]))
                                                             for cj in list(self.stressFluxBoundaryConditionsSetterDict.keys())])
        self.coefficients.initializeGlobalExteriorElementBoundaryQuadrature(self.timeIntegration.t, self.ebqe)

    def estimate_mt(self):
        pass

    def calculateSolutionAtQuadrature(self):
        pass

    def calculateAuxiliaryQuantitiesAfterStep(self):
        OneLevelTransport.calculateAuxiliaryQuantitiesAfterStep(self)

    def preStep(self):
        pass

    def postStep(self):
        pass

    def updateAfterMeshMotion(self):
        # cek todo: this needs to be cleaned up and generalized for other models under moving conditions
        # few  models  actually use  the ebqe['x'] for boundary conditions, but  we need  to make it
        # consistent  (the  physical coordinates and not the reference domain coordinates)
        self.calculateElementQuadrature()
        self.ebqe_old_x = self.ebqe['x'].copy()
        self.calculateExteriorElementBoundaryQuadrature()  # pass
        for cj in range(self.nc):
            self.u[cj].femSpace.updateInterpolationPoints()
            for dofN, g in list(self.dirichletConditionsForceDOF[cj].DOFBoundaryConditionsDict.items()):
                self.dirichletConditionsForceDOF[cj].DOFBoundaryPointDict[dofN] = self.mesh.nodeArray[dofN]
