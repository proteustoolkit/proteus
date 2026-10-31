import proteus
import proteus.MeshTools
from proteus.mprans.cNCLS import *
import numpy as np
from proteus.Transport import OneLevelTransport, cfemIntegrals, SparseMat
from proteus.Transport import TC_base, logEvent, NonlinearEquation, Quadrature, Comm
from proteus.Transport import memory, FluxBoundaryConditions, ExplicitLumpedMassMatrix
from proteus.Transport import globalMax, SSP, ExplicitConsistentMassMatrixWithRedistancing
from proteus import Norms
from . import cArgumentsDict
import proteus.NumericalFlux
import proteus.ShockCapturing
import proteus.SubgridError
import proteus.TimeIntegration
import proteus.TransportCoefficients

class SubgridError(proteus.SubgridError.SGE_base):
    def __init__(self, coefficients, nd):
        proteus.SubgridError.SGE_base.__init__(self, coefficients, nd, False)

    def initializeElementQuadrature(self, mesh, t, cq):
        for ci in range(self.nc):
            cq[('dH_sge', ci, ci)] = cq[('dH', ci, ci)]

    def calculateSubgridError(self, q):
        pass

    def updateSubgridErrorHistory(self, initializationPhase=False):
        pass


class ShockCapturing(proteus.ShockCapturing.DelayedLagState, proteus.ShockCapturing.ShockCapturing_base):
    def __init__(self, coefficients, nd, shockCapturingFactor=0.25, lag=True, nStepsToDelay=None):
        proteus.ShockCapturing.ShockCapturing_base.__init__(self, coefficients, nd, shockCapturingFactor, lag)
        self.nStepsToDelay = nStepsToDelay
        self.nSteps = 0
        if self.lag:
            logEvent("NCLS.ShockCapturing: lagging requested but must lag the first step; switching lagging off and delaying")
            self.nStepsToDelay = 1
            self.lag = False

    def initializeElementQuadrature(self, mesh, t, cq):
        self.mesh = mesh
        self.numDiff = []
        self.numDiff_last = []
        for ci in range(self.nc):
            self.numDiff.append(cq[('numDiff', ci, ci)])
            self.numDiff_last.append(cq[('numDiff', ci, ci)])

    def updateShockCapturingHistory(self):
        self.nSteps += 1
        if self.lag:
            for ci in range(self.nc):
                self.numDiff_last[ci][:] = self.numDiff[ci]
        if self.lag == False and self.nStepsToDelay is not None and self.nSteps > self.nStepsToDelay:
            logEvent("NCLS.ShockCapturing: switched to lagged shock capturing")
            self.lag = True
            self.numDiff_last = []
            for ci in range(self.nc):
                self.numDiff_last.append(self.numDiff[ci].copy())
        logEvent("NCLS: max numDiff %e" % (globalMax(self.numDiff_last[0].max()),))


class NumericalFlux(proteus.NumericalFlux.HamiltonJacobi_DiagonalLesaintRaviart):
    def __init__(self, vt, getPointwiseBoundaryConditions,
                 getAdvectiveFluxBoundaryConditions,
                 getDiffusiveFluxBoundaryConditions,
                 getPeriodicBoundaryConditions=None):
        proteus.NumericalFlux.HamiltonJacobi_DiagonalLesaintRaviart.__init__(self, vt, getPointwiseBoundaryConditions,
                                                                             getAdvectiveFluxBoundaryConditions,
                                                                             getDiffusiveFluxBoundaryConditions)


class RKEV(proteus.TimeIntegration.SSP):
    """
    Wrapper for SSPRK time integration using EV
    ... more to come ...
    """

    def __init__(self, transport, timeOrder=1, runCFL=0.1, integrateInterpolationPoints=False):
        SSP.__init__(self, transport, integrateInterpolationPoints=integrateInterpolationPoints)
        self.runCFL = runCFL
        self.dtLast = None
        self.isAdaptive = True
        # About the cfl
        assert transport.coefficients.STABILIZATION_TYPE > 0, "SSP method just works for edge based EV methods; i.e., STABILIZATION_TYPE>0"
        assert hasattr(transport, 'edge_based_cfl'), "No edge based cfl defined"
        self.cfl = transport.edge_based_cfl
        # Stuff particular for SSP
        self.timeOrder = timeOrder  # order of approximation
        self.nStages = timeOrder  # number of stages total
        self.lstage = 0  # last stage completed
        # storage vectors
        self.u_dof_last = {}
        # per component stage values, list with array at each stage
        self.u_dof_stage = {}
        for ci in range(self.nc):
            if ('m', ci) in transport.q:
                self.u_dof_last[ci] = transport.u[ci].dof.copy()
                self.u_dof_stage[ci] = []
                for k in range(self.nStages + 1):
                    self.u_dof_stage[ci].append(transport.u[ci].dof.copy())

    def choose_dt(self):
        maxCFL = 1.0e-6
        maxCFL = max(maxCFL, globalMax(self.cfl.max()))
        self.dt = self.runCFL/maxCFL
        if self.dtLast is None:
            self.dtLast = self.dt
        self.t = self.tLast + self.dt
        self.substeps = [self.t for i in range(self.nStages)]  # Manuel is ignoring different time step levels for now

    def initialize_dt(self, t0, tOut, q):
        """
        Modify self.dt
        """
        self.tLast = t0
        self.choose_dt()
        self.t = t0 + self.dt

    def setCoefficients(self):
        """
        beta are all 1's here
        mwf not used right now
        """
        self.alpha = np.zeros((self.nStages, self.nStages), 'd')
        self.dcoefs = np.zeros((self.nStages), 'd')

    def updateStage(self):
        """
        Need to switch to use coefficients
        """
        self.lstage += 1
        assert self.timeOrder in [1, 2, 3]
        assert self.lstage > 0 and self.lstage <= self.timeOrder
        if self.timeOrder == 3:
            if self.lstage == 1:
                logEvent("First stage of SSP33 method", level=4)
                for ci in range(self.nc):
                    self.u_dof_stage[ci][self.lstage][:] = self.transport.u[ci].dof
                    # update u_dof_old
                    self.transport.u_dof_old[:] = self.u_dof_stage[ci][self.lstage]
            elif self.lstage == 2:
                logEvent("Second stage of SSP33 method", level=4)
                for ci in range(self.nc):
                    self.u_dof_stage[ci][self.lstage][:] = self.transport.u[ci].dof
                    self.u_dof_stage[ci][self.lstage] *= 1./4.
                    self.u_dof_stage[ci][self.lstage] += 3. / 4. * self.u_dof_last[ci]
                    # Update u_dof_old
                    self.transport.u_dof_old[:] = self.u_dof_stage[ci][self.lstage]
            elif self.lstage == 3:
                logEvent("Third stage of SSP33 method", level=4)
                for ci in range(self.nc):
                    self.u_dof_stage[ci][self.lstage][:] = self.transport.u[ci].dof
                    self.u_dof_stage[ci][self.lstage] *= 2.0/3.0
                    self.u_dof_stage[ci][self.lstage] += 1.0 / 3.0 * self.u_dof_last[ci]
                    # update u_dof_old
                    self.transport.u_dof_old[:] = self.u_dof_last[ci]
                    # update solution to u[0].dof
                    self.transport.u[ci].dof[:] = self.u_dof_stage[ci][self.lstage]
        elif self.timeOrder == 2:
            if self.lstage == 1:
                logEvent("First stage of SSP22 method", level=4)
                for ci in range(self.nc):
                    self.u_dof_stage[ci][self.lstage][:] = self.transport.u[ci].dof
                    # Update u_dof_old
                    self.transport.u_dof_old[:] = self.transport.u[ci].dof
            elif self.lstage == 2:
                logEvent("Second stage of SSP22 method", level=4)
                for ci in range(self.nc):
                    self.u_dof_stage[ci][self.lstage][:] = self.transport.u[ci].dof
                    self.u_dof_stage[ci][self.lstage][:] *= 1./2.
                    self.u_dof_stage[ci][self.lstage][:] += 1. / 2. * self.u_dof_last[ci]
                    # update u_dof_old
                    self.transport.u_dof_old[:] = self.u_dof_last[ci]
                    # update solution to u[0].dof
                    self.transport.u[ci].dof[:] = self.u_dof_stage[ci][self.lstage]
        else:
            assert self.timeOrder == 1
            for ci in range(self.nc):
                self.u_dof_stage[ci][self.lstage][:] = self.transport.u[ci].dof[:]

    def initializeTimeHistory(self, resetFromDOF=True):
        """
        Push necessary information into time history arrays
        """
        for ci in range(self.nc):
            self.u_dof_last[ci][:] = self.transport.u[ci].dof[:]

    def updateTimeHistory(self, resetFromDOF=False):
        """
        assumes successful step has been taken
        """
        self.t = self.tLast + self.dt
        for ci in range(self.nc):
            self.u_dof_last[ci][:] = self.transport.u[ci].dof[:]
            for k in range(self.nStages):
                self.u_dof_stage[ci][k][:] = self.transport.u[ci].dof[:]
        self.lstage = 0
        self.dtLast = self.dt
        self.tLast = self.t

    def generateSubsteps(self, tList):
        """
        create list of substeps over time values given in tList. These correspond to stages
        """
        self.substeps = []
        tLast = self.tLast
        for t in tList:
            dttmp = t - tLast
            self.substeps.extend([tLast + dttmp for i in range(self.nStages)])
            tLast = t

    def resetOrder(self, order):
        """
        initialize data structures for stage updges
        """
        self.timeOrder = order  # order of approximation
        self.nStages = order  # number of stages total
        self.lstage = 0  # last stage completed
        # storage vectors
        # per component stage values, list with array at each stage
        self.u_dof_stage = {}
        for ci in range(self.nc):
            if ('m', ci) in self.transport.q:
                self.u_dof_stage[ci] = []
                for k in range(self.nStages + 1):
                    self.u_dof_stage[ci].append(self.transport.u[ci].dof.copy())
        self.substeps = [self.t for i in range(self.nStages)]

    def setFromOptions(self, nOptions):
        """
        allow classes to set various numerical parameters
        """
        if 'runCFL' in dir(nOptions):
            self.runCFL = nOptions.runCFL
        flags = ['timeOrder']
        for flag in flags:
            if flag in dir(nOptions):
                val = getattr(nOptions, flag)
                setattr(self, flag, val)
                if flag == 'timeOrder':
                    self.resetOrder(self.timeOrder)


class Coefficients(proteus.TransportCoefficients.TC_base):
    from proteus.ctransportCoefficients import ncLevelSetCoefficientsEvaluate
    def __init__(self,
                 V_model=0,
                 RD_model=None,
                 ME_model=1,
                 checkMass=True,
                 epsFact=1.5,
                 useMetrics=0.0,
                 sc_uref=1.0,
                 sc_beta=1.0,
                 waterline_interval=-1,
                 movingDomain=False,
                 PURE_BDF=False,
                 # PARAMETERS FOR EV
                 STABILIZATION_TYPE=0,
                 LUMPED_MASS_MATRIX=False,
                 ENTROPY_TYPE=1,  # polynomial, u=0.5*u^2
                 cE=1.0,
                 # COUPEZ AND REDISTANCING PARAMETERS
                 DO_SMOOTHING=False,
                 DO_REDISTANCING=False,
                 pure_redistancing=False,
                 COUPEZ=False,
                 SATURATED_LEVEL_SET=False,
                 epsCoupez=0.1,
                 epsFactRedistancing=0.33,  # For the signed distance function
                 redistancing_tolerance=0.1,
                 maxIter_redistancing=3,
                 lambda_coupez=0.1,
                 cfl_redistancing=1.0,
                 # OUTPUT quantDOFs
                 outputQuantDOFs=False,
                 # NULLSPACE Info
                 nullSpace='NoNullSpace',
                 initialize=True):

        self.PURE_BDF=PURE_BDF
        self.DO_SMOOTHING = DO_SMOOTHING
        self.COUPEZ = COUPEZ
        self.SATURATED_LEVEL_SET = SATURATED_LEVEL_SET
        self.DO_REDISTANCING = DO_REDISTANCING
        self.ENTROPY_TYPE = ENTROPY_TYPE
        self.cE = cE
        self.LUMPED_MASS_MATRIX = LUMPED_MASS_MATRIX
        self.STABILIZATION_TYPE = STABILIZATION_TYPE
        self.epsFactRedistancing = epsFactRedistancing
        self.pure_redistancing = pure_redistancing
        self.maxIter_redistancing = maxIter_redistancing
        self.redistancing_tolerance = redistancing_tolerance
        self.cfl_redistancing = cfl_redistancing
        self.epsCoupez = epsCoupez
        self.lambda_coupez = lambda_coupez
        self.outputQuantDOFs = outputQuantDOFs
        self.movingDomain = movingDomain
        self.useMetrics = useMetrics
        self.epsFact = epsFact
        self.variableNames = ['phi']
        self.flowModelIndex = V_model
        self.modelIndex = ME_model
        self.RD_modelIndex = RD_model
        self.checkMass = checkMass
        self.sc_uref = sc_uref
        self.sc_beta = sc_beta
        self.waterline_interval = waterline_interval
        self.nullSpace = nullSpace
        if initialize:
            self.initialize()

    def initialize(self):
        nc = 1
        mass = {0: {0: 'linear'}}
        hamiltonian = {0: {0: 'linear'}}
        advection = {}
        diffusion = {}
        potential = {}
        reaction = {}
        TC_base.__init__(self,
                         nc,
                         mass,
                         advection,
                         diffusion,
                         potential,
                         reaction,
                         hamiltonian,
                         ['phi'],
                         movingDomain=self.movingDomain)

    def attachModels(self, modelList):
        # the level set model
        self.model = modelList[self.modelIndex]
        # the velocity
        if self.flowModelIndex >= 0:
            self.flowModel = modelList[self.flowModelIndex]
            self.q_v = modelList[self.flowModelIndex].q[('velocity', 0)]
            self.ebqe_v = modelList[self.flowModelIndex].ebqe[('velocity', 0)]
            if ('velocity', 0) in modelList[self.flowModelIndex].ebq:
                self.ebq_v = modelList[self.flowModelIndex].ebq[('velocity', 0)]
            else:
                self.ebq_v = None
            if ('u', 0) not in self.model.ebq and ('u', 0) in self.flowModel.ebq:
                self.model.ebq[('u', 0)] = np.zeros(self.flowModel.ebq[('u', 0)].shape, 'd')
                self.model.ebq[('grad(u)', 0)] = np.zeros(self.flowModel.ebq[('grad(u)', 0)].shape, 'd')
            if ('v', 1) in self.flowModel.ebq:
                self.model.u[0].getValuesTrace(self.flowModel.ebq[('v', 1)], self.model.ebq[('u', 0)])
                self.model.u[0].getGradientValuesTrace(self.flowModel.ebq[('grad(v)', 1)], self.model.ebq[('grad(u)', 0)])
        if self.RD_modelIndex is not None:
            # print self.RD_modelIndex,len(modelList)
            self.rdModel = modelList[self.RD_modelIndex]
            self.ebqe_rd_u = self.rdModel.ebqe[('u',0)]

    def initializeElementQuadrature(self, t, cq):
        if self.flowModelIndex is None:
            self.q_v = np.ones(cq[('grad(u)', 0)].shape, 'd')

    def initializeElementBoundaryQuadrature(self, t, cebq, cebq_global):
        if self.flowModelIndex is None:
            self.ebq_v = np.ones(cebq[('grad(u)', 0)].shape, 'd')

    def initializeGlobalExteriorElementBoundaryQuadrature(self, t, cebqe):
        if self.flowModelIndex is None:
            self.ebqe_v = np.ones(cebqe[('grad(u)', 0)].shape, 'd')
        if self.RD_modelIndex is None:
            self.ebqe_rd_u = cebqe[('u',0)]

    def preStep(self, t, firstStep=False):
        # SAVE OLD SOLUTION #
        self.model.u_dof_old[:] = self.model.u[0].dof

        # COMPUTE NEW VELOCITY (if given by user) #
        if self.model.hasVelocityFieldAsFunction:
            self.model.updateVelocityFieldAsFunction()

        if self.checkMass:
            self.m_pre = Norms.scalarSmoothedHeavisideDomainIntegral(self.epsFact,
                                                                     self.model.mesh.elementDiametersArray,
                                                                     self.flowModel.coefficients.q_porosity*self.model.q['dV'],
                                                                     self.model.q[('m',0)],
                                                                     self.model.mesh.nElements_owned)
            logEvent("Phase  0 mass before NCLS step = %12.5e" % (self.m_pre,),level=2)
            self.m_last = Norms.scalarSmoothedHeavisideDomainIntegral(self.epsFact,
                                                                      self.model.mesh.elementDiametersArray,
                                                                      self.flowModel.coefficients.q_porosity*self.model.q['dV'],
                                                                      self.model.timeIntegration.m_last[0],
                                                                      self.model.mesh.nElements_owned)
            logEvent("Phase  0 mass before NCLS step (m_last) = %12.5e" % (self.m_last,),level=2)
        copyInstructions = {}
        return copyInstructions

    def postStep(self, t, firstStep=False):
        self.model.q['dV_last'][:] = self.model.q['dV']
        if self.checkMass:
            self.m_post = Norms.scalarSmoothedHeavisideDomainIntegral(self.epsFact,
                                                                      self.model.mesh.elementDiametersArray,
                                                                      self.flowModel.coefficients.q_porosity*self.model.q['dV'],
                                                                      self.model.q[('u',0)],
                                                                      self.model.mesh.nElements_owned)
            logEvent("Phase  0 mass after NCLS step = %12.5e" % (self.m_post,),level=2)
            # #need a flux here not a velocity
            # self.fluxIntegral = Norms.fluxDomainBoundaryIntegralFromVector(self.flowModel.ebqe['dS'],
            #                                                                self.flowModel.ebqe[('velocity',0)],
            #                                                                self.flowModel.ebqe['n'],
            #                                                                self.model.mesh)
            # logEvent("Flux integral = %12.5e" % (self.fluxIntegral,),level=2)
            # logEvent("Phase  0 mass conservation after NCLS step = %12.5e" % (self.m_post - self.m_last + self.model.timeIntegration.dt*self.fluxIntegral,),level=2)
            # self.lsGlobalMass = self.m_post
            # self.fluxGlobal = self.fluxIntegral*self.model.timeIntegration.dt
            # self.totalFluxGlobal += self.fluxGlobal
            # self.lsGlobalMassArray.append(self.lsGlobalMass)
            # self.lsGlobalMassErrorArray.append(self.lsGlobalMass - self.lsGlobalMassArray[0] + self.totalFluxGlobal)
            # self.fluxArray.append(self.fluxIntegral)
            # self.timeArray.append(self.model.timeIntegration.t)
        copyInstructions = {}
        return copyInstructions

    def updateToMovingDomain(self, t, c):
        # in a moving domain simulation the velocity coming in is already for the moving domain
        pass

    def evaluate(self, t, c):
        v = None
        if c[('dH', 0, 0)].shape == self.q_v.shape:
            v = self.q_v
        elif c[('dH', 0, 0)].shape == self.ebqe_v.shape:
            v = self.ebqe_v
        elif self.ebq_v is not None and c[('dH', 0, 0)].shape == self.ebq_v.shape:
            v = self.ebq_v
        else:
            raise RuntimeError("don't have v for NC Level set of shape = " + repr(c[('dH', 0, 0)].shape))
        if v is not None:
            self.ncLevelSetCoefficientsEvaluate(v,
                                                c[('u', 0)],
                                                c[('grad(u)', 0)],
                                                c[('m', 0)],
                                                c[('dm', 0, 0)],
                                                c[('H', 0)],
                                                c[('dH', 0, 0)])


class LevelModel(OneLevelTransport):
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
                 movingDomain=False,
                 bdyNullSpace=False):

        self.L2_norm_redistancing = 0.
        self.redistancing_L2_norm_history = []
        self.auxiliaryCallCalculateResidual = False
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
        # mwf try to reuse test and trial information across components if spaces are the same
        self.reuse_test_trial_quadrature = reuse_trial_and_test_quadrature  # True#False
        if self.reuse_test_trial_quadrature:
            for ci in range(1, coefficients.nc):
                assert self.u[ci].femSpace.__class__.__name__ == self.u[0].femSpace.__class__.__name__, "to reuse_test_trial_quad all femSpaces must be the same!"
        self.u_dof_old = None
        self.free_u = None
        # Simplicial Mesh
        self.mesh = self.u[0].femSpace.mesh  # assume the same mesh for  all components for now
        self.testSpace = testSpaceDict
        self.dirichletConditions = dofBoundaryConditionsDict
        self.dirichletNodeSetList = None  # explicit Dirichlet  conditions for now, no Dirichlet BC constraint
        self.bdyNullSpace = bdyNullSpace
        self.coefficients = coefficients
        self.coefficients.initializeMesh(self.mesh)
        self.nc = self.coefficients.nc
        self.stabilization = stabilization
        self.shockCapturing = shockCapturing
        self.conservativeFlux = conservativeFluxDict  # no velocity post-processing for now
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
#        if isinstance(self.u[0].femSpace,C0_AffineLinearOnSimplexWithNodalBasis):
#            print self.nQuadraturePoints_element
#            if self.nSpace_global == 3:
#                assert(self.nQuadraturePoints_element == 5)
#            elif self.nSpace_global == 2:
#                assert(self.nQuadraturePoints_element == 6)
#            elif self.nSpace_global == 1:
#                assert(self.nQuadraturePoints_element == 3)
#
#            print self.nElementBoundaryQuadraturePoints_elementBoundary
#            if self.nSpace_global == 3:
#                assert(self.nElementBoundaryQuadraturePoints_elementBoundary == 4)
#            elif self.nSpace_global == 2:
#                assert(self.nElementBoundaryQuadraturePoints_elementBoundary == 4)
#            elif self.nSpace_global == 1:
#                assert(self.nElementBoundaryQuadraturePoints_elementBoundary == 1)
        #
        # simplified allocations for test==trial and also check if space is mixed or not
        #
        self.q = {}
        self.ebq = {}
        self.ebq_global = {}
        self.ebqe = {}
        self.phi_ip = {}
        self.edge_based_cfl = np.zeros(self.u[0].dof.shape)
        # mesh
        self.q['x'] = np.zeros((self.mesh.nElements_global, self.nQuadraturePoints_element, 3), 'd')
        self.ebqe['x'] = np.zeros((self.mesh.nExteriorElementBoundaries_global, self.nElementBoundaryQuadraturePoints_elementBoundary, 3), 'd')
        self.q[('dV_u', 0)] = (1.0/self.mesh.nElements_global) * np.ones((self.mesh.nElements_global, self.nQuadraturePoints_element), 'd')
        self.q[('u', 0)] = np.zeros((self.mesh.nElements_global, self.nQuadraturePoints_element), 'd')
        self.q[('grad(u)', 0)] = np.zeros((self.mesh.nElements_global, self.nQuadraturePoints_element, self.nSpace_global), 'd')
        self.q[('m_last', 0)] = np.zeros((self.mesh.nElements_global, self.nQuadraturePoints_element), 'd')
        self.q[('mt', 0)] = np.zeros((self.mesh.nElements_global, self.nQuadraturePoints_element), 'd')
        self.q['dV'] = np.zeros((self.mesh.nElements_global, self.nQuadraturePoints_element), 'd')
        self.q['dV_last'] = -1000 * np.ones((self.mesh.nElements_global, self.nQuadraturePoints_element), 'd')
        self.q[('m_tmp', 0)] = self.q[('u', 0)]  # np.zeros((self.mesh.nElements_global,self.nQuadraturePoints_element),'d')
        self.q[('m', 0)] = self.q[('u', 0)]  # np.zeros((self.mesh.nElements_global,self.nQuadraturePoints_element),'d')
        # cek todo for NCLS we really don't need dH because it's just q_v from the flow model
        self.q[('dH', 0, 0)] = np.zeros((self.mesh.nElements_global, self.nQuadraturePoints_element, self.nSpace_global), 'd')
        self.q[('dH_sge', 0, 0)] = np.zeros((self.mesh.nElements_global, self.nQuadraturePoints_element, self.nSpace_global), 'd')
        self.q[('cfl', 0)] = np.zeros((self.mesh.nElements_global, self.nQuadraturePoints_element), 'd')
        self.q[('numDiff', 0, 0)] = np.zeros((self.mesh.nElements_global, self.nQuadraturePoints_element), 'd')
        self.ebqe[('u', 0)] = np.zeros((self.mesh.nExteriorElementBoundaries_global, self.nElementBoundaryQuadraturePoints_elementBoundary), 'd')
        self.ebqe[('grad(u)', 0)] = np.zeros((self.mesh.nExteriorElementBoundaries_global,
                                                 self.nElementBoundaryQuadraturePoints_elementBoundary, self.nSpace_global), 'd')
        # mwf for running as standalone
        self.ebqe[('dH', 0, 0)] = np.zeros((self.mesh.nExteriorElementBoundaries_global,
                                               self.nElementBoundaryQuadraturePoints_elementBoundary, self.nSpace_global), 'd')
        self.q[('dm', 0, 0)] = np.zeros((self.mesh.nElements_global, self.nQuadraturePoints_element), 'd')
        self.q[('H', 0)] = np.zeros((self.mesh.nElements_global, self.nQuadraturePoints_element), 'd')
        self.points_elementBoundaryQuadrature = set()
        self.scalars_elementBoundaryQuadrature = set([('u', ci) for ci in range(self.nc)])
        self.vectors_elementBoundaryQuadrature = set()
        self.tensors_elementBoundaryQuadrature = set()
        # mql. Allow the user to provide functions to define the velocity field
        self.hasVelocityFieldAsFunction = False
        if ('velocityField') in dir(options):
            self.velocityField = options.velocityField
            self.hasVelocityFieldAsFunction = True
        #
        # allocate residual and Jacobian storage
        #
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

        # mql. Some ASSERTS to restrict the combination of the methods
        if self.coefficients.STABILIZATION_TYPE > 0:
            assert self.timeIntegration.isSSP == True, "If STABILIZATION_TYPE>0, use RKEV timeIntegration within NCLS model"
            cond = 'levelNonlinearSolver' in dir(options) and (options.levelNonlinearSolver ==
                                                               ExplicitLumpedMassMatrix or options.levelNonlinearSolver == ExplicitConsistentMassMatrixWithRedistancing)
            assert cond, "If STABILIZATION_TYPE>0, use levelNonlinearSolver=ExplicitLumpedMassMatrix or ExplicitConsistentMassMatrixWithRedistancing"
        if 'levelNonlinearSolver' in dir(options) and options.levelNonlinearSolver == ExplicitLumpedMassMatrix:
            assert self.coefficients.LUMPED_MASS_MATRIX, "If levelNonlinearSolver=ExplicitLumpedMassMatrix, use LUMPED_MASS_MATRIX=True"
        if self.coefficients.LUMPED_MASS_MATRIX == True:
            cond = self.coefficients.STABILIZATION_TYPE == 2
            assert cond, "Use lumped mass matrix just with: STABILIZATION_TYPE=2 (smoothness based stab.)"
            cond = 'levelNonlinearSolver' in dir(options) and options.levelNonlinearSolver == ExplicitLumpedMassMatrix
            assert cond, "Use levelNonlinearSolver=ExplicitLumpedMassMatrix when the mass matrix is lumped"
        if self.coefficients.DO_REDISTANCING:
            assert self.coefficients.STABILIZATION_TYPE > 0, "If DO_REDISTANCING=True, use: STABILIZATION_TYPE>0"
            assert self.coefficients.LUMPED_MASS_MATRIX == False, "If DO_REDISTANCING=True, use: LUMPED_MASS_MATRIX=False"
            cond = 'levelNonlinearSolver' in dir(options) and options.levelNonlinearSolver == ExplicitConsistentMassMatrixWithRedistancing
            assert cond, "If DO_REDISTANCING=True, use: levelNonlinearSolver=ExplicitConsistentMassMatrixWithRedistancing"
            assert self.timeIntegration.isSSP, "If DO_REDISTANCING=True, use RKEV timeIntegration within NCLS. timeOrder=2 is recommended"
        # END OF ASSERTS

        # Smoothing matrix
        self.SmoothingMatrix = None  # Mass-epsilon^2*Laplacian
        self.SmoothingMatrix_a = None
        self.SmoothingMatrix_sparseFactor = None
        self.Jacobian_sparseFactor = None
        self.uStar_dof = np.copy(self.u[0].dof)
        # Mass matrices
        self.ML = None  # lumped mass matrix
        self.MC_global = None  # consistent mass matrix
        # C-Matrices
        self.cterm_global = None

        # interface locator
        self.interface_locator = np.zeros(self.u[0].dof.shape,'d')        
        # Aux quantity at DOFs to be filled by optimized code (MQL)
        self.quantDOFs = np.zeros(self.u[0].dof.shape, 'd')

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
        from proteus import PostProcessingTools
        self.velocityPostProcessor = PostProcessingTools.VelocityPostProcessingChooser(self)
        logEvent(memory("velocity postprocessor", "OneLevelTransport"), level=4)
        # helper for writing out data storage
        from proteus import Archiver
        self.elementQuadratureDictionaryWriter = Archiver.XdmfWriter()
        self.elementBoundaryQuadratureDictionaryWriter = Archiver.XdmfWriter()
        self.exteriorElementBoundaryQuadratureDictionaryWriter = Archiver.XdmfWriter()
        # TODO get rid of this
#        for ci,fbcObject  in self.fluxBoundaryConditionsObjectsDict.iteritems():
#            self.ebqe[('advectiveFlux_bc_flag',ci)] = np.zeros(self.ebqe[('advectiveFlux_bc',ci)].shape,'i')
#            for t,g in fbcObject.advectiveFluxBoundaryConditionsDict.iteritems():
#                if self.coefficients.advection.has_key(ci):
#                    self.ebqe[('advectiveFlux_bc',ci)][t[0],t[1]] = g(self.ebqe[('x')][t[0],t[1]],self.timeIntegration.t)
#                    self.ebqe[('advectiveFlux_bc_flag',ci)][t[0],t[1]] = 1

        if hasattr(self.numericalFlux, 'setDirichletValues'):
            self.numericalFlux.setDirichletValues(self.ebqe)
        if not hasattr(self.numericalFlux, 'isDOFBoundary'):
            self.numericalFlux.isDOFBoundary = {0: np.zeros(self.ebqe[('u', 0)].shape, 'i')}
        if not hasattr(self.numericalFlux, 'ebqe'):
            self.numericalFlux.ebqe = {('u', 0): np.zeros(self.ebqe[('u', 0)].shape, 'd')}
        # TODO how to handle redistancing calls for calculateCoefficients,calculateElementResidual etc
        self.globalResidualDummy = None
        compKernelFlag = 0
        self.ncls = cNCLS_base(self.nSpace_global,
                               self.nQuadraturePoints_element,
                               self.u[0].femSpace.elementMaps.localFunctionSpace.dim,
                               self.u[0].femSpace.referenceFiniteElement.localFunctionSpace.dim,
                               self.testSpace[0].referenceFiniteElement.localFunctionSpace.dim,
                               self.nElementBoundaryQuadraturePoints_elementBoundary,
                               compKernelFlag)

        self.forceStrongConditions = False
        if self.forceStrongConditions:
            self.dirichletConditionsForceDOF = DOFBoundaryConditions(self.u[0].femSpace, dofBoundaryConditionsSetterDict[0], weakDirichletConditions=False)

        if self.movingDomain:
            self.MOVING_DOMAIN = 1.0
        else:
            self.MOVING_DOMAIN = 0.0
        if self.mesh.nodeVelocityArray is None:
            self.mesh.nodeVelocityArray = np.zeros(self.mesh.nodeArray.shape, 'd')

        self.waterline_calls = 0
        self.waterline_prints = 0

    # mwf these are getting called by redistancing classes,
    def calculateCoefficients(self):
        pass

    def updateVelocityFieldAsFunction(self):
        X = {0: self.q[('x')][:, :, 0],
             1: self.q[('x')][:, :, 1],
             2: self.q[('x')][:, :, 2]}
        t = self.timeIntegration.t
        self.coefficients.q_v[..., 0] = self.velocityField[0](X, t)
        self.coefficients.q_v[..., 1] = self.velocityField[1](X, t)
        if (self.nSpace_global == 3):
            self.coefficients.q_v[..., 2] = self.velocityField[2](X, t)

        # BOUNDARY
        ebqe_X = {0: self.ebqe['x'][:, :, 0],
                  1: self.ebqe['x'][:, :, 1],
                  2: self.ebqe['x'][:, :, 2]}
        self.coefficients.ebqe_v[..., 0] = self.velocityField[0](ebqe_X, t)
        self.coefficients.ebqe_v[..., 1] = self.velocityField[1](ebqe_X, t)
        if (self.nSpace_global == 3):
            self.coefficients.ebqe_v[..., 2] = self.velocityField[2](ebqe_X, t)

    ######################################
    ######## GET REDISTANCING RHS ########
    ######################################
    def getRedistancingResidual(self, u, r):
        import pdb
        import copy
        """
        Calculate the element residuals and add in to the global residual
        """

        r.fill(0.0)
        # Load the unknowns into the finite element dof
        self.timeIntegration.calculateCoefs()
        self.timeIntegration.calculateU(u)
        self.setUnknowns(self.timeIntegration.u)

        rowptr, colind, nzval = self.jacobian.getCSRrepresentation()
        edge_based_cfl = np.zeros(len(rowptr) - 1)

        assert (self.cterm_global is not None), "C matrices have not been computed"
        rowptr, colind, Cx = self.cterm_global[0].getCSRrepresentation()
        rowptr, colind, Cy = self.cterm_global[1].getCSRrepresentation()
        if (self.nSpace_global == 3):
            rowptr, colind, Cz = self.cterm_global[2].getCSRrepresentation()
        else:
            Cz = np.zeros(Cx.shape, 'd')

        if (self.coefficients.pure_redistancing == True):
            u_dof_old = np.copy(self.u_dof_old)
        else:
            u_dof_old = np.copy(self.u[0].dof)

        argsDict = cArgumentsDict.ArgumentsDict()
        argsDict["dt"] = self.timeIntegration.dt * (self.coefficients.cfl_redistancing if self.coefficients.pure_redistancing == False else 1.)
        argsDict["mesh_trial_ref"] = self.u[0].femSpace.elementMaps.psi
        argsDict["mesh_grad_trial_ref"] = self.u[0].femSpace.elementMaps.grad_psi
        argsDict["mesh_dof"] = self.mesh.nodeArray
        argsDict["mesh_l2g"] = self.mesh.elementNodesArray
        argsDict["dV_ref"] = self.elementQuadratureWeights[('u', 0)]
        argsDict["u_trial_ref"] = self.u[0].femSpace.psi
        argsDict["u_grad_trial_ref"] = self.u[0].femSpace.grad_psi
        argsDict["u_test_ref"] = self.u[0].femSpace.psi
        argsDict["nElements_global"] = self.mesh.nElements_global
        argsDict["u_l2g"] = self.u[0].femSpace.dofMap.l2g
        argsDict["r_l2g"] = self.l2g[0]['freeGlobal']
        argsDict["elementDiameter"] = self.mesh.elementDiametersArray
        argsDict["nodeDiametersArray"] = self.mesh.nodeDiametersArray
        argsDict["u_dof"] = self.u[0].dof
        argsDict["u_dof_old"] = u_dof_old
        argsDict["uStar_dof"] = self.uStar_dof
        argsDict["offset_u"] = self.offset[0]
        argsDict["stride_u"] = self.stride[0]
        argsDict["globalResidual"] = r
        argsDict["numDOFs"] = len(rowptr) - 1
        argsDict["NNZ"] = self.nnz
        argsDict["csrRowIndeces_DofLoops"] = rowptr
        argsDict["csrColumnOffsets_DofLoops"] = colind
        argsDict["csrRowIndeces_CellLoops"] = self.csrRowIndeces[(0, 0)]
        argsDict["csrColumnOffsets_CellLoops"] = self.csrColumnOffsets[(0, 0)]
        argsDict["lambda_coupez"] = self.coefficients.lambda_coupez
        argsDict["epsCoupez"] = self.coefficients.epsCoupez
        argsDict["epsFactRedistancing"] = self.coefficients.epsFactRedistancing * self.mesh.h
        argsDict["edge_based_cfl"] = edge_based_cfl
        argsDict["SATURATED_LEVEL_SET"] = self.coefficients.SATURATED_LEVEL_SET
        argsDict["Cx"] = Cx
        argsDict["Cy"] = Cy
        argsDict["Cz"] = Cz
        argsDict["ML"] = self.ML
        L2_norm = self.ncls.calculateRedistancingResidual(argsDict)

        if (self.coefficients.pure_redistancing == True):
            self.edge_based_cfl[:] = edge_based_cfl

        return L2_norm
    ######################################
    ######################################
    ######################################

    ######################################
    ######## GET SMOOTHING RHS ########
    ######################################
    def getRhsSmoothing(self, u, r):
        import pdb
        import copy
        """
        Calculate the element residuals and add in to the global residual
        """
        r.fill(0.0)
        # Load the unknowns into the finite element dof
        self.timeIntegration.calculateCoefs()
        self.timeIntegration.calculateU(u)
        self.setUnknowns(self.timeIntegration.u)

        rowptr, colind, nzval = self.jacobian.getCSRrepresentation()
        argsDict = cArgumentsDict.ArgumentsDict()
        argsDict["mesh_trial_ref"] = self.u[0].femSpace.elementMaps.psi
        argsDict["mesh_grad_trial_ref"] = self.u[0].femSpace.elementMaps.grad_psi
        argsDict["mesh_dof"] = self.mesh.nodeArray
        argsDict["mesh_l2g"] = self.mesh.elementNodesArray
        argsDict["dV_ref"] = self.elementQuadratureWeights[('u', 0)]
        argsDict["u_trial_ref"] = self.u[0].femSpace.psi
        argsDict["u_grad_trial_ref"] = self.u[0].femSpace.grad_psi
        argsDict["u_test_ref"] = self.u[0].femSpace.psi
        argsDict["nElements_global"] = self.mesh.nElements_global
        argsDict["u_l2g"] = self.u[0].femSpace.dofMap.l2g
        argsDict["r_l2g"] = self.l2g[0]['freeGlobal']
        argsDict["elementDiameter"] = self.mesh.elementDiametersArray
        argsDict["nodeDiametersArray"] = self.mesh.nodeDiametersArray
        argsDict["u_dof_old"] = self.u_dof_old
        argsDict["offset_u"] = self.offset[0]
        argsDict["stride_u"] = self.stride[0]
        argsDict["globalResidual"] = r
        self.ncls.calculateRhsSmoothing(argsDict)

    ######################################
    ######################################
    ######################################

    def calculateElementResidual(self):
        fromFreeToGlobal=0 #direction copying
        cfemIntegrals.copyBetweenFreeUnknownsAndGlobalUnknowns(fromFreeToGlobal,
                                                               self.offset[0],
                                                               self.stride[0],
                                                               self.dirichletConditions[0].global2freeGlobal_global_dofs,
                                                               self.dirichletConditions[0].global2freeGlobal_free_dofs,
                                                               self.free_u,
                                                               self.u[0].dof)

        if self.globalResidualDummy is not None:
            self.getResidual(self.free_u, self.globalResidualDummy)

    def ebqe_rd_u_for_kernel(self):
        """The redistanced level set on the exterior faces, as the kernel should see it.

        On a face with no Dirichlet condition the kernel's exterior flux is
        -flow*(u - rd_u) where the flow is inward, and 0 where it is outward.
        That is the right boundary treatment on a domain boundary. On a face
        shared with another rank (an exterior face of the subdomain that is
        interior to the global mesh) the neighbour's outward side contributes
        nothing, so the term does not cancel and pulls u toward the
        redistanced value there, which differs from u. Use the current u on
        those faces so the term is zero, as it is in serial.
        """
        rd = self.coefficients.ebqe_rd_u
        mask = proteus.MeshTools.rankInterfaceExteriorMask(self.mesh)
        if mask is None or not mask.any():
            return rd
        faces = np.nonzero(mask)[0]
        ebN = self.mesh.exteriorElementBoundariesArray[faces]
        eN = self.mesh.elementBoundaryElementsArray[ebN, 0]
        ebN_local = self.mesh.elementBoundaryLocalElementBoundariesArray[ebN, 0]
        dofs = self.u[0].femSpace.dofMap.l2g[eN]
        psi = self.u[0].femSpace.psi_trace[ebN_local]
        rd = rd.copy()
        rd[faces] = np.einsum('fkj,fj->fk', psi, self.u[0].dof[dofs])
        return rd

    def getResidual(self, u, r):
        import pdb
        import copy
        """
        Calculate the element residuals and add in to the global residual
        """
        if self.free_u is None:
            self.free_u = u.copy()
        if self.u_dof_old is None:
            # Pass initial condition to u_dof_old
            self.u_dof_old = np.copy(self.u[0].dof)
        ########################
        ### COMPUTE C MATRIX ###
        ########################
        if self.cterm_global is None:
            # since we only need cterm_global to persist, we can drop the other self.'s
            self.cterm = {}
            self.cterm_a = {}
            self.cterm_global = {}
            rowptr, colind, nzval = self.jacobian.getCSRrepresentation()
            nnz = nzval.shape[-1]  # number of non-zero entries in sparse matrix
            di = self.q[('grad(u)', 0)].copy()  # direction of derivative
            # JACOBIANS (FOR ELEMENT TRANSFORMATION)
            self.q[('J')] = np.zeros((self.mesh.nElements_global,
                                      self.nQuadraturePoints_element,
                                      self.nSpace_global,
                                      self.nSpace_global),
                                     'd')
            self.q[('inverse(J)')] = np.zeros((self.mesh.nElements_global,
                                               self.nQuadraturePoints_element,
                                               self.nSpace_global,
                                               self.nSpace_global),
                                              'd')
            self.q[('det(J)')] = np.zeros((self.mesh.nElements_global,
                                           self.nQuadraturePoints_element),
                                          'd')
            self.u[0].femSpace.elementMaps.getJacobianValues(self.elementQuadraturePoints,
                                                             self.q['J'],
                                                             self.q['inverse(J)'],
                                                             self.q['det(J)'])
            self.q['abs(det(J))'] = np.abs(self.q['det(J)'])
            # SHAPE FUNCTIONS
            self.q[('w', 0)] = np.zeros((self.mesh.nElements_global,
                                         self.nQuadraturePoints_element,
                                         self.nDOF_test_element[0]),
                                        'd')
            self.q[('w*dV_m', 0)] = self.q[('w', 0)].copy()
            self.u[0].femSpace.getBasisValues(self.elementQuadraturePoints, self.q[('w', 0)])
            cfemIntegrals.calculateWeightedShape(self.elementQuadratureWeights[('u', 0)],
                                                 self.q['abs(det(J))'],
                                                 self.q[('w', 0)],
                                                 self.q[('w*dV_m', 0)])
            # GRADIENT OF TEST FUNCTIONS
            self.q[('grad(w)', 0)] = np.zeros((self.mesh.nElements_global,
                                               self.nQuadraturePoints_element,
                                               self.nDOF_test_element[0],
                                               self.nSpace_global),
                                              'd')
            self.u[0].femSpace.getBasisGradientValues(self.elementQuadraturePoints,
                                                      self.q['inverse(J)'],
                                                      self.q[('grad(w)', 0)])
            self.q[('grad(w)*dV_f', 0)] = np.zeros((self.mesh.nElements_global,
                                                    self.nQuadraturePoints_element,
                                                    self.nDOF_test_element[0],
                                                    self.nSpace_global),
                                                   'd')
            cfemIntegrals.calculateWeightedShapeGradients(self.elementQuadratureWeights[('u', 0)],
                                                          self.q['abs(det(J))'],
                                                          self.q[('grad(w)', 0)],
                                                          self.q[('grad(w)*dV_f', 0)])
            ##########################
            ### LUMPED MASS MATRIX ###
            ##########################
            # assume a linear mass term
            dm = np.ones(self.q[('u', 0)].shape, 'd')
            elementMassMatrix = np.zeros((self.mesh.nElements_global,
                                          self.nDOF_test_element[0],
                                          self.nDOF_trial_element[0]), 'd')
            cfemIntegrals.updateMassJacobian_weak_lowmem(dm,
                                                         self.q[('w', 0)],
                                                         self.q[('w*dV_m', 0)],
                                                         elementMassMatrix)
            self.MC_a = nzval.copy()
            self.MC_global = SparseMat(self.nFreeDOF_global[0],
                                       self.nFreeDOF_global[0],
                                       nnz,
                                       self.MC_a,
                                       colind,
                                       rowptr)
            cfemIntegrals.zeroJacobian_CSR(self.nnz, self.MC_global)
            cfemIntegrals.updateGlobalJacobianFromElementJacobian_CSR(self.l2g[0]['nFreeDOF'],
                                                                      self.l2g[0]['freeLocal'],
                                                                      self.l2g[0]['nFreeDOF'],
                                                                      self.l2g[0]['freeLocal'],
                                                                      self.csrRowIndeces[(0, 0)],
                                                                      self.csrColumnOffsets[(0, 0)],
                                                                      elementMassMatrix,
                                                                      self.MC_global)
            self.ML = np.zeros((self.nFreeDOF_global[0],), 'd')
            for i in range(self.nFreeDOF_global[0]):
                self.ML[i] = self.MC_a[rowptr[i]:rowptr[i + 1]].sum()
            #################################
            ### END OF LUMPED MASS MATRIX ###
            #################################

            for d in range(self.nSpace_global):  # spatial dimensions
                # C matrices
                self.cterm[d] = np.zeros((self.mesh.nElements_global,
                                          self.nDOF_test_element[0],
                                          self.nDOF_trial_element[0]), 'd')
                self.cterm_a[d] = nzval.copy()
                #self.cterm_a[d] = np.zeros(nzval.size)
                self.cterm_global[d] = SparseMat(self.nFreeDOF_global[0],
                                                 self.nFreeDOF_global[0],
                                                 nnz,
                                                 self.cterm_a[d],
                                                 colind,
                                                 rowptr)
                cfemIntegrals.zeroJacobian_CSR(self.nnz, self.cterm_global[d])
                di[:] = 0.0
                di[..., d] = 1.0
                cfemIntegrals.updateHamiltonianJacobian_weak_lowmem(di,
                                                                    self.q[('grad(w)*dV_f', 0)],
                                                                    self.q[('w', 0)],
                                                                    self.cterm[d])  # int[(di*grad(wj))*wi*dV]
                cfemIntegrals.updateGlobalJacobianFromElementJacobian_CSR(self.l2g[0]['nFreeDOF'],
                                                                          self.l2g[0]['freeLocal'],
                                                                          self.l2g[0]['nFreeDOF'],
                                                                          self.l2g[0]['freeLocal'],
                                                                          self.csrRowIndeces[(0, 0)],
                                                                          self.csrColumnOffsets[(0, 0)],
                                                                          self.cterm[d],
                                                                          self.cterm_global[d])
        rowptr, colind, Cx = self.cterm_global[0].getCSRrepresentation()
        rowptr, colind, Cy = self.cterm_global[1].getCSRrepresentation()
        if (self.nSpace_global == 3):
            rowptr, colind, Cz = self.cterm_global[2].getCSRrepresentation()
        else:
            Cz = np.zeros(Cx.shape, 'd')

        # zero out residual
        r.fill(0.0)
        self.interface_locator.fill(0.0) 
        # Load the unknowns into the finite element dof
        self.timeIntegration.calculateCoefs()
        self.timeIntegration.calculateU(u)
        self.setUnknowns(self.timeIntegration.u)

        # Dirichlet boundary conditions
        self.numericalFlux.setDirichletValues(self.ebqe)
        # flux boundary conditions, SHOULDN'T HAVE

        if self.forceStrongConditions:
            for dofN, g in list(self.dirichletConditionsForceDOF.DOFBoundaryConditionsDict.items()):
                self.u[0].dof[dofN] = g(self.dirichletConditionsForceDOF.DOFBoundaryPointDict[dofN], self.timeIntegration.t)

        degree_polynomial = 1
        try:
            degree_polynomial = self.u[0].femSpace.order
        except:
            pass

        if (self.coefficients.STABILIZATION_TYPE == 0):  # SUPG
            self.calculateResidual = self.ncls.calculateResidual
            self.calculateJacobian = self.ncls.calculateJacobian
        else:
            self.calculateResidual = self.ncls.calculateResidual_entropy_viscosity
            self.calculateJacobian = self.ncls.calculateMassMatrix

        #if(self.timeIntegration.t > 0.002):
        #  import pdb; pdb.set_trace()

        argsDict = cArgumentsDict.ArgumentsDict()
        argsDict["dt"] = self.timeIntegration.dt
        # FLCBDF shrinks dt until the BDF time derivative dominates the lagged shock-capturing
        # residual (numDiff*dt ~ const); keep the time term out of that residual for FLCBDF only
        argsDict["sc_drop_time"] = 1.0 if isinstance(self.timeIntegration, proteus.TimeIntegration.FLCBDF) else 0.0
        argsDict["mesh_trial_ref"] = self.u[0].femSpace.elementMaps.psi
        argsDict["mesh_grad_trial_ref"] = self.u[0].femSpace.elementMaps.grad_psi
        argsDict["mesh_dof"] = self.mesh.nodeArray
        argsDict["mesh_velocity_dof"] = self.mesh.nodeVelocityArray
        argsDict["MOVING_DOMAIN"] = self.MOVING_DOMAIN
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
        argsDict["useMetrics"] = self.coefficients.useMetrics
        argsDict["alphaBDF"] = self.timeIntegration.alpha_bdf
        argsDict["lag_shockCapturing"] = self.shockCapturing.lag
        argsDict["shockCapturingDiffusion"] = self.shockCapturing.shockCapturingFactor
        argsDict["sc_uref"] = self.coefficients.sc_uref
        argsDict["sc_alpha"] = self.coefficients.sc_beta
        argsDict["u_l2g"] = self.u[0].femSpace.dofMap.l2g
        argsDict["r_l2g"] = self.l2g[0]['freeGlobal']
        argsDict["elementDiameter"] = self.mesh.elementDiametersArray
        argsDict["nodeDiametersArray"] = self.mesh.nodeDiametersArray
        argsDict["degree_polynomial"] = degree_polynomial
        argsDict["u_dof"] = self.u[0].dof
        argsDict["u_dof_old"] = self.u_dof_old
        argsDict["uStar_dof"] = self.uStar_dof
        argsDict["velocity"] = self.coefficients.q_v
        argsDict["q_m"] = self.timeIntegration.m_tmp[0]
        argsDict["q_u"] = self.q[('u', 0)]
        argsDict["q_n"] = self.q[('grad(u)', 0)]
        argsDict["q_dH"] = self.q[('dH_sge', 0, 0)]
        argsDict["q_m_betaBDF"] = self.timeIntegration.beta_bdf[0]
        argsDict["q_dV"] = self.q['dV']
        argsDict["q_dV_last"] = self.q['dV_last']
        argsDict["cfl"] = self.q[('cfl', 0)]
        argsDict["edge_based_cfl"] = self.edge_based_cfl
        argsDict["q_numDiff_u"] = self.shockCapturing.numDiff[0]
        argsDict["q_numDiff_u_last"] = self.shockCapturing.numDiff_last[0]
        argsDict["offset_u"] = self.offset[0]
        argsDict["stride_u"] = self.stride[0]
        argsDict["globalResidual"] = r
        argsDict["nExteriorElementBoundaries_global"] = self.mesh.nExteriorElementBoundaries_global
        argsDict["exteriorElementBoundariesArray"] = self.mesh.exteriorElementBoundariesArray
        argsDict["elementBoundaryElementsArray"] = self.mesh.elementBoundaryElementsArray
        argsDict["elementBoundaryLocalElementBoundariesArray"] = self.mesh.elementBoundaryLocalElementBoundariesArray
        argsDict["ebqe_velocity_ext"] = self.coefficients.ebqe_v
        argsDict["isDOFBoundary_u"] = self.numericalFlux.isDOFBoundary[0]
        argsDict["ebqe_rd_u_ext"] = self.ebqe_rd_u_for_kernel()
        argsDict["ebqe_bc_u_ext"] = self.numericalFlux.ebqe[('u', 0)]
        argsDict["ebqe_u"] = self.ebqe[('u', 0)]
        argsDict["ebqe_grad_u"] = self.ebqe[('grad(u)', 0)]
        argsDict["interface_locator"] = self.interface_locator
        argsDict["PURE_BDF"] = self.coefficients.PURE_BDF
        argsDict["numDOFs"] = len(rowptr) - 1
        argsDict["NNZ"] = self.nnz
        argsDict["csrRowIndeces_DofLoops"] = rowptr
        argsDict["csrColumnOffsets_DofLoops"] = colind
        argsDict["csrRowIndeces_CellLoops"] = self.csrRowIndeces[(0, 0)]
        argsDict["csrColumnOffsets_CellLoops"] = self.csrColumnOffsets[(0, 0)]
        argsDict["csrColumnOffsets_eb_CellLoops"] = self.csrColumnOffsets_eb[(0, 0)]
        argsDict["LUMPED_MASS_MATRIX"] = self.coefficients.LUMPED_MASS_MATRIX
        argsDict["quantDOFs"] = self.quantDOFs
        argsDict["lambda_coupez"] = self.coefficients.lambda_coupez
        argsDict["epsCoupez"] = self.coefficients.epsCoupez
        argsDict["epsFactRedistancing"] = self.coefficients.epsFactRedistancing * self.mesh.h
        argsDict["COUPEZ"] = self.coefficients.COUPEZ
        argsDict["SATURATED_LEVEL_SET"] = self.coefficients.SATURATED_LEVEL_SET
        argsDict["Cx"] = Cx
        argsDict["Cy"] = Cy
        argsDict["Cz"] = Cz
        argsDict["ML"] = self.ML
        argsDict["STABILIZATION_TYPE"] = self.coefficients.STABILIZATION_TYPE
        argsDict["ENTROPY_TYPE"] = self.coefficients.ENTROPY_TYPE
        argsDict["cE"] = self.coefficients.cE
        self.calculateResidual(argsDict)
        
        self.quantDOFs[:] = self.interface_locator

        if self.forceStrongConditions:
            for dofN, g in list(self.dirichletConditionsForceDOF.DOFBoundaryConditionsDict.items()):
                r[dofN] = 0

        if (self.auxiliaryCallCalculateResidual == False):
            edge_based_cflMax = globalMax(self.edge_based_cfl.max()) * self.timeIntegration.dt
            cell_based_cflMax = globalMax(self.q[('cfl', 0)].max()) * self.timeIntegration.dt
            logEvent("...   Current dt = " + str(self.timeIntegration.dt), level=4)
            logEvent("...   Maximum Cell Based CFL = " + str(cell_based_cflMax), level=2)
            logEvent("...   Maximum Edge Based CFL = " + str(edge_based_cflMax), level=2)

        # print "velocity in ncls",self.coefficients.q_v,
        # print "cfl",self.q[('cfl',0)]
        if self.stabilization:
            self.stabilization.accumulateSubgridMassHistory(self.q)
        logEvent("Global residual", level=9, data=r)
        self.nonlinear_function_evaluations += 1
        if self.globalResidualDummy is None:
            self.globalResidualDummy = np.zeros(r.shape, 'd')

    def getSmoothingMatrix(self):
        #import superluWrappers
        #import np
        import pdb

        if (self.SmoothingMatrix is None):
            rowptr, colind, nzval = self.jacobian.getCSRrepresentation()
            self.SmoothingMatrix_a = nzval.copy()
            nnz = nzval.shape[-1]  # number of non-zero entries in sparse matrix
            self.SmoothingMatrix = LinearAlgebraTools.SparseMat(self.nFreeDOF_global[0],
                                                                self.nFreeDOF_global[0],
                                                                nnz,
                                                                self.SmoothingMatrix_a,
                                                                colind,
                                                                rowptr)
        cfemIntegrals.zeroJacobian_CSR(self.nNonzerosInJacobian,
                                       self.SmoothingMatrix)
        degree_polynomial = 1
        try:
            degree_polynomial = self.u[0].femSpace.order
        except:
            pass

        argsDict = cArgumentsDict.ArgumentsDict()
        argsDict["dt"] = self.timeIntegration.dt
        argsDict["mesh_trial_ref"] = self.u[0].femSpace.elementMaps.psi
        argsDict["mesh_grad_trial_ref"] = self.u[0].femSpace.elementMaps.grad_psi
        argsDict["mesh_dof"] = self.mesh.nodeArray
        argsDict["mesh_velocity_dof"] = self.mesh.nodeVelocityArray
        argsDict["MOVING_DOMAIN"] = self.MOVING_DOMAIN
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
        argsDict["useMetrics"] = self.coefficients.useMetrics
        argsDict["alphaBDF"] = self.timeIntegration.alpha_bdf
        argsDict["lag_shockCapturing"] = self.shockCapturing.lag
        argsDict["shockCapturingDiffusion"] = self.shockCapturing.shockCapturingFactor
        argsDict["u_l2g"] = self.u[0].femSpace.dofMap.l2g
        argsDict["r_l2g"] = self.l2g[0]['freeGlobal']
        argsDict["elementDiameter"] = self.mesh.elementDiametersArray
        argsDict["degree_polynomial"] = degree_polynomial
        argsDict["u_dof"] = self.u[0].dof
        argsDict["velocity"] = self.coefficients.q_v
        argsDict["q_m_betaBDF"] = self.timeIntegration.beta_bdf[0]
        argsDict["cfl"] = self.q[('cfl', 0)]
        argsDict["q_numDiff_u_last"] = self.shockCapturing.numDiff_last[0]
        argsDict["csrRowIndeces_u_u"] = self.csrRowIndeces[(0, 0)]
        argsDict["csrColumnOffsets_u_u"] = self.csrColumnOffsets[(0, 0)]
        argsDict["globalJacobian"] = self.SmoothingMatrix.getCSRrepresentation()[2]
        argsDict["nExteriorElementBoundaries_global"] = self.mesh.nExteriorElementBoundaries_global
        argsDict["exteriorElementBoundariesArray"] = self.mesh.exteriorElementBoundariesArray
        argsDict["elementBoundaryElementsArray"] = self.mesh.elementBoundaryElementsArray
        argsDict["elementBoundaryLocalElementBoundariesArray"] = self.mesh.elementBoundaryLocalElementBoundariesArray
        argsDict["ebqe_velocity_ext"] = self.coefficients.ebqe_v
        argsDict["isDOFBoundary_u"] = self.numericalFlux.isDOFBoundary[0]
        argsDict["ebqe_rd_u_ext"] = self.ebqe_rd_u_for_kernel()
        argsDict["ebqe_bc_u_ext"] = self.numericalFlux.ebqe[('u', 0)]
        argsDict["csrColumnOffsets_eb_u_u"] = self.csrColumnOffsets_eb[(0, 0)]
        argsDict["he"] = self.mesh.h
        self.ncls.calculateSmoothingMatrix(argsDict)

    def getJacobian(self, jacobian):
        #import superluWrappers
        #import np
        import pdb
        cfemIntegrals.zeroJacobian_CSR(self.nNonzerosInJacobian,
                                       jacobian)

        degree_polynomial = 1
        try:
            degree_polynomial = self.u[0].femSpace.order
        except:
            pass

        argsDict = cArgumentsDict.ArgumentsDict()
        argsDict["dt"] = self.timeIntegration.dt
        argsDict["mesh_trial_ref"] = self.u[0].femSpace.elementMaps.psi
        argsDict["mesh_grad_trial_ref"] = self.u[0].femSpace.elementMaps.grad_psi
        argsDict["mesh_dof"] = self.mesh.nodeArray
        argsDict["mesh_velocity_dof"] = self.mesh.nodeVelocityArray
        argsDict["MOVING_DOMAIN"] = self.MOVING_DOMAIN
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
        argsDict["useMetrics"] = self.coefficients.useMetrics
        argsDict["alphaBDF"] = self.timeIntegration.alpha_bdf
        argsDict["lag_shockCapturing"] = self.shockCapturing.lag
        argsDict["shockCapturingDiffusion"] = self.shockCapturing.shockCapturingFactor
        argsDict["u_l2g"] = self.u[0].femSpace.dofMap.l2g
        argsDict["r_l2g"] = self.l2g[0]['freeGlobal']
        argsDict["elementDiameter"] = self.mesh.elementDiametersArray
        argsDict["degree_polynomial"] = degree_polynomial
        argsDict["u_dof"] = self.u[0].dof
        argsDict["velocity"] = self.coefficients.q_v
        argsDict["q_m_betaBDF"] = self.timeIntegration.beta_bdf[0]
        argsDict["cfl"] = self.q[('cfl', 0)]
        argsDict["q_numDiff_u_last"] = self.shockCapturing.numDiff_last[0]
        argsDict["csrRowIndeces_u_u"] = self.csrRowIndeces[(0, 0)]
        argsDict["csrColumnOffsets_u_u"] = self.csrColumnOffsets[(0, 0)]
        argsDict["globalJacobian"] = jacobian.getCSRrepresentation()[2]
        argsDict["nExteriorElementBoundaries_global"] = self.mesh.nExteriorElementBoundaries_global
        argsDict["exteriorElementBoundariesArray"] = self.mesh.exteriorElementBoundariesArray
        argsDict["elementBoundaryElementsArray"] = self.mesh.elementBoundaryElementsArray
        argsDict["elementBoundaryLocalElementBoundariesArray"] = self.mesh.elementBoundaryLocalElementBoundariesArray
        argsDict["ebqe_velocity_ext"] = self.coefficients.ebqe_v
        argsDict["isDOFBoundary_u"] = self.numericalFlux.isDOFBoundary[0]
        argsDict["ebqe_rd_u_ext"] = self.ebqe_rd_u_for_kernel()
        argsDict["ebqe_bc_u_ext"] = self.numericalFlux.ebqe[('u', 0)]
        argsDict["csrColumnOffsets_eb_u_u"] = self.csrColumnOffsets_eb[(0, 0)]
        argsDict["PURE_BDF"] = self.coefficients.PURE_BDF
        argsDict["LUMPED_MASS_MATRIX"] = self.coefficients.LUMPED_MASS_MATRIX
        self.calculateJacobian(argsDict)

        # Load the Dirichlet conditions directly into residual
        if self.forceStrongConditions:
            scaling = 1.0  # probably want to add some scaling to match non-dirichlet diagonals in linear system
            for dofN in list(self.dirichletConditionsForceDOF.DOFBoundaryConditionsDict.keys()):
                global_dofN = dofN
                for i in range(self.rowptr[global_dofN], self.rowptr[global_dofN + 1]):
                    if (self.colind[i] == global_dofN):
                            # print "RBLES forcing residual cj = %s dofN= %s global_dofN= %s was self.nzval[i]= %s now =%s " % (cj,dofN,global_dofN,self.nzval[i],scaling)
                        self.nzval[i] = scaling
                    else:
                        self.nzval[i] = 0.0
                        # print "RBLES zeroing residual cj = %s dofN= %s global_dofN= %s " % (cj,dofN,global_dofN)

        logEvent("Jacobian ", level=10, data=jacobian)
        # mwf decide if this is reasonable for solver statistics
        self.nonlinear_function_jacobian_evaluations += 1
        return jacobian

    def calculateElementQuadrature(self):
        """
        Calculate the physical location and weights of the quadrature rules
        and the shape information at the quadrature points.

        This function should be called only when the mesh changes.
        """
        self.u[0].femSpace.elementMaps.getValues(self.elementQuadraturePoints,
                                                 self.q['x'])
        self.u[0].femSpace.elementMaps.getBasisValuesRef(self.elementQuadraturePoints)
        self.u[0].femSpace.elementMaps.getBasisGradientValuesRef(self.elementQuadraturePoints)
        self.u[0].femSpace.getBasisValuesRef(self.elementQuadraturePoints)
        self.u[0].femSpace.getBasisGradientValuesRef(self.elementQuadraturePoints)
        self.coefficients.initializeElementQuadrature(self.timeIntegration.t, self.q)
        if self.stabilization is not None:
            self.stabilization.initializeElementQuadrature(self.mesh, self.timeIntegration.t, self.q)
            self.stabilization.initializeTimeIntegration(self.timeIntegration)
        if self.shockCapturing is not None:
            self.shockCapturing.initializeElementQuadrature(self.mesh, self.timeIntegration.t, self.q)

    def calculateElementBoundaryQuadrature(self):
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
        self.fluxBoundaryConditionsObjectsDict = dict([(cj, FluxBoundaryConditions(self.mesh,
                                                                                   self.nElementBoundaryQuadraturePoints_elementBoundary,
                                                                                   self.ebqe[('x')],
                                                                                   self.advectiveFluxBoundaryConditionsSetterDict[cj],
                                                                                   self.diffusiveFluxBoundaryConditionsSetterDictDict[cj]))
                                                       for cj in list(self.advectiveFluxBoundaryConditionsSetterDict.keys())])
        self.coefficients.initializeGlobalExteriorElementBoundaryQuadrature(self.timeIntegration.t, self.ebqe)

    def estimate_mt(self):
        pass

    def calculateSolutionAtQuadrature(self):
        pass

    def calculateAuxiliaryQuantitiesAfterStep(self):
        pass

    def computeWaterline(self, t):
        self.waterline_calls += 1
        if self.coefficients.waterline_interval > 0 and self.waterline_calls % self.coefficients.waterline_interval == 0:
            self.waterline_npoints = np.zeros((1,), 'i')
            self.waterline_data = np.zeros((self.mesh.nExteriorElementBoundaries_global, self.nSpace_global), 'd')
            argsDict = cArgumentsDict.ArgumentsDict()
            argsDict["wlc"] = self.waterline_npoints
            argsDict["waterline"] = self.waterline_data
            argsDict["mesh_trial_ref"] = self.u[0].femSpace.elementMaps.psi
            argsDict["mesh_grad_trial_ref"] = self.u[0].femSpace.elementMaps.grad_psi
            argsDict["mesh_dof"] = self.mesh.nodeArray
            argsDict["mesh_velocity_dof"] = self.mesh.nodeVelocityArray
            argsDict["MOVING_DOMAIN"] = self.MOVING_DOMAIN
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
            argsDict["useMetrics"] = self.coefficients.useMetrics
            argsDict["alphaBDF"] = self.timeIntegration.alpha_bdf
            argsDict["lag_shockCapturing"] = self.shockCapturing.lag
            argsDict["shockCapturingDiffusion"] = self.shockCapturing.shockCapturingFactor
            argsDict["sc_uref"] = self.coefficients.sc_uref
            argsDict["sc_alpha"] = self.coefficients.sc_beta
            argsDict["u_l2g"] = self.u[0].femSpace.dofMap.l2g
            argsDict["r_l2g"] = self.l2g[0]['freeGlobal']
            argsDict["elementDiameter"] = self.mesh.elementDiametersArray
            argsDict["u_dof"] = self.u[0].dof
            argsDict["u_dof_old"] = self.u_dof_old
            argsDict["velocity"] = self.coefficients.q_v
            argsDict["q_m"] = self.timeIntegration.m_tmp[0]
            argsDict["q_u"] = self.q[('u', 0)]
            argsDict["q_n"] = self.q[('grad(u)', 0)]
            argsDict["q_dH"] = self.q[('dH_sge', 0, 0)]
            argsDict["q_m_betaBDF"] = self.timeIntegration.beta_bdf[0]
            argsDict["cfl"] = self.q[('cfl', 0)]
            argsDict["q_numDiff_u"] = self.shockCapturing.numDiff[0]
            argsDict["q_numDiff_u_last"] = self.shockCapturing.numDiff_last[0]
            argsDict["offset_u"] = self.offset[0]
            argsDict["stride_u"] = self.stride[0]
            argsDict["nExteriorElementBoundaries_global"] = self.mesh.nExteriorElementBoundaries_global
            argsDict["exteriorElementBoundariesArray"] = self.mesh.exteriorElementBoundariesArray
            argsDict["elementBoundaryElementsArray"] = self.mesh.elementBoundaryElementsArray
            argsDict["elementBoundaryLocalElementBoundariesArray"] = self.mesh.elementBoundaryLocalElementBoundariesArray
            argsDict["elementBoundaryMaterialTypes"] = self.mesh.elementBoundaryUserFlags
            argsDict["ebqe_velocity_ext"] = self.coefficients.ebqe_v
            argsDict["isDOFBoundary_u"] = self.numericalFlux.isDOFBoundary[0]
            argsDict["ebqe_bc_u_ext"] = self.numericalFlux.ebqe[('u', 0)]
            argsDict["ebqe_u"] = self.ebqe[('u', 0)]
            self.ncls.calculateWaterline(argsDict)
            from proteus import Comm
            comm = Comm.get()
            filename = os.path.join(self.coefficients.opts.dataDir,  "waterline." + str(comm.rank()) + "." + str(self.waterline_prints))
            np.save(filename, self.waterline_data[0:self.waterline_npoints[0]])
            self.waterline_prints += 1

    def updateAfterMeshMotion(self):
        pass
