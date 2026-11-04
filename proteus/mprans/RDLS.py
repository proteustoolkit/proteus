import proteus
from proteus import TimeIntegration
from proteus.mprans.cRDLS import *
from proteus.Comm import globalMax
import numpy as np
from proteus.Transport import OneLevelTransport, memory
from proteus.Transport import TC_base, NonlinearEquation, Quadrature, logEvent
from proteus.Transport import cfemIntegrals
from proteus.LinearAlgebraTools import ParVec_petsc4py
from . import cArgumentsDict
import proteus.FemTools
import proteus.NonlinearSolvers
import proteus.ShockCapturing
import proteus.StepControl
import proteus.SubgridError
import proteus.Transport
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


class ShockCapturing(proteus.ShockCapturing.ShockCapturing_base):
    def __init__(self, coefficients, nd, shockCapturingFactor=0.25, lag=True, nStepsToDelay=None):
        proteus.ShockCapturing.ShockCapturing_base.__init__(self, coefficients, nd, shockCapturingFactor, lag)
        self.nStepsToDelay = nStepsToDelay
        self.nSteps = 0

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
            self.lag = True
            self.numDiff_last = []
            for ci in range(self.nc):
                self.numDiff_last.append(self.numDiff[ci].copy())
        logEvent("RDLS: max numDiff %e" % (globalMax(self.numDiff_last[0].max()),))


class PsiTC(proteus.StepControl.SC_base):
    def __init__(self, model, nOptions):
        proteus.StepControl.SC_base.__init__(self, model, nOptions)
        for ci in list(nOptions.atol_res.keys()):
            self.atol = nOptions.atol_res[ci]
            self.rtol = nOptions.rtol_res[ci]
        self.stepExact = True
        for m in model.levelModelList:
            m.timeIntegration.isAdaptive = False
        self.nSteps = 0
        self.nStepsOsher = nOptions.psitc['nStepsForce']
        self.red_ratio = nOptions.psitc['reduceRatio']
        self.start_ratio = nOptions.psitc['startRatio']
        self.nStepsMax = nOptions.psitc['nStepsMax']

    def stepExact_model(self, tOut):
        # pseudo time step
        for m in self.model.levelModelList:
            m.timeIntegration.choose_dt()
        self.dt_model = self.start_ratio * m.timeIntegration.dt
        self.set_dt_allLevels()
        # physical time step
        self.t_model = tOut
        self.setSubsteps([tOut])

    def initialize_dt_model(self, t0, tOut):
        self.saveSolution()
        self.t_model_last = t0
        self.t_model = tOut
        # pseudo time step
        for m in self.model.levelModelList:
            m.timeIntegration.initialize_dt(t0, tOut, m.q)
        # set the starting time steps
        self.dt_model = self.start_ratio * m.timeIntegration.dt
        self.set_dt_allLevels()
        # physical time step
        self.t = tOut
        self.setSubsteps([tOut])
        self.nSteps = 0
        logEvent("Initializing time step on model %s to dt = %12.5e" % (self.model.name,
                                                                        self.dt_model),
                 level=1)

    def updateSubstep(self):
        # choose  a new dt and add a substep without increasing t
        # if the steady state has been reached then append the new  t to the  substeps
        self.solverFailures = 0
        self.errorFailures = 0
        self.saveSolution()
        # here we just need to test the error and set to tOut if steady state
        if self.nSteps == 0:
            self.res0 = self.model.solver.solverList[-1].norm_r0
            for m in self.model.levelModelList:
                m.timeIntegration.choose_dt()
            self.dt_model = self.start_ratio * self.model.levelModelList[0].timeIntegration.dt
        res = self.model.solver.solverList[-1].norm_r0
        ssError = res/(self.res0*self.rtol+self.atol)
        for m in self.model.levelModelList:
            m.updateTimeHistory(self.t_model)
            m.timeIntegration.updateTimeHistory()
        if ((self.nSteps < self.nStepsOsher or
             ssError >= 1.0) and
                self.nSteps < self.nStepsMax):
            self.nSteps += 1
            self.dt_model = m.timeIntegration.dt
            if self.nSteps >= self.nStepsOsher:  # start ramping up the time step
                self.dt_model = self.dt_model * self.red_ratio
            #logEvent("Osher-PsiTC dt %12.5e" %(self.dt_model),level=1)
            self.set_dt_allLevels()
            # physical time step
            self.t_model = self.substeps[0]
            self.substeps.append(self.substeps[0])
            logEvent("Osher-PsiTC iteration %d  dt = %12.5e  |res| = %12.5e %g  " % (self.nSteps, self.dt_model, res, (res/self.res0) * 100.0), level=1)
        elif self.nSteps >= self.nStepsMax:
            logEvent("Osher-PsiTC DID NOT Converge |res| = %12.5e but quitting anyway" % (res,))
            logEvent("Osher-PsiTC tolerance                %12.5e " % (self.res0 * self.rtol + self.atol,))
            self.nSteps = 0
        else:
            logEvent("Osher-PsiTC converged |res| = %12.5e %12.5e" % (res, ssError * 100.0))
            logEvent("Osher-PsiTC tolerance                %12.5e " % (self.res0 * self.rtol + self.atol,))
            self.nSteps = 0

    def choose_dt_model(self):
        # don't modify dt_model
        self.solverFailures = 0
        self.errorFailures = 0
        self.saveSolution()
        # pseudo time step
        for m in self.model.levelModelList:
            m.timeIntegration.choose_dt()
        self.dt_model = self.start_ratio * m.timeIntegration.dt
        self.set_dt_allLevels()
        # physical time step
        self.t_model = self.substeps[0]
        self.setSubsteps([self.substeps[0]])
        logEvent("Osher-PsiTC choosing dt = %12.5e " % (self.dt_model,))


class Coefficients(proteus.TransportCoefficients.TC_base):
    from proteus.ctransportCoefficients import redistanceLevelSetCoefficientsEvaluate

    def __init__(self,
                 applyRedistancing=True,
                 epsFact=2.0,
                 nModelId=None,
                 u0=None,
                 rdModelId=0,
                 penaltyParameter=0.0,
                 useMetrics=0.0,
                 useConstantH=False,
                 weakDirichletFactor=10.0,
                 backgroundDiffusionFactor=0.01,
                 # Parameters for elliptic re-distancing
                 computeMetrics = False,
                 ELLIPTIC_REDISTANCING=0, #Linear elliptic re-distancing by default
                 alpha=1.0E9,
                 backgroundDissipationEllipticRedist=1.0,
                 # OUTPUT quantDOFs
                 outputQuantDOFs=False,
                 # NULLSpace info
                 nullSpace='NoNullSpace', #penalization param for elliptic re-distancing
                 useExact=False,
                 copyList=True,
                 initialize=True):
        self.copyList=copyList
        self.useExact=useExact
        self.useConstantH = useConstantH
        self.useMetrics = useMetrics
        self.variableNames = ['phid']
        nc = 1
        mass = {0: {0: 'linear'}}
        hamiltonian = {0: {0: 'nonlinear'}}
        advection = {}
        diffusion = {}
        potential = {}
        reaction = {0: {0: 'constant'}}
        TC_base.__init__(self,
                         nc,
                         mass,
                         advection,
                         diffusion,
                         potential,
                         reaction,
                         hamiltonian,
                         self.variableNames)
        self.nModelId = nModelId
        self.rdModelId = rdModelId
        self.epsFact = epsFact
        self.q_u0 = None
        self.ebq_u0 = None
        self.ebqe_u0 = None
        self.dof_u0 = None
        self.u0 = u0
        self.applyRedistancing = applyRedistancing
        self.weakBC_on = True
        self.penaltyParameter = penaltyParameter
        self.backgroundDiffusionFactor = backgroundDiffusionFactor
        self.weakDirichletFactor = weakDirichletFactor
        self.outputQuantDOFs=outputQuantDOFs
        self.computeMetrics=computeMetrics
        self.ELLIPTIC_REDISTANCING=ELLIPTIC_REDISTANCING
        assert (ELLIPTIC_REDISTANCING >= 0 and ELLIPTIC_REDISTANCING <= 3), "ELLIPTIC_REDISTANCING=0,1,2 or 3."
        #ELLIPTIC_REDISTANCING:
        #0: Use default re-distancing in Proteus
        #1: Non-linear elliptic re-distancing via single or doble pot.
        #   Uses single/double pot. with(out) |grad(u)| reconstructed. See c++ code
        #2: Linear elliptic re-distancing via C0 normal reconstruction and single pot.
        #3: Non-linear elliptic re-distancing via C0 normal reconstruction and single pot.
        self.alpha=alpha
        self.backgroundDissipationEllipticRedist=backgroundDissipationEllipticRedist
        self.freeze_interface_within_elliptic_redist = False
        if alpha=='inf':
            self.alpha = 0
            self.freeze_interface_within_elliptic_redist = True
        self.nullSpace = nullSpace
        if initialize:
            self.initialize()

    def initialize(self):
        pass

    def attachModels(self, modelList):
        if self.nModelId is not None:
            self.nModel = modelList[self.nModelId]
            self.q_u0 = self.nModel.q[('u', 0)]
            if ('u', 0) in self.nModel.ebq:
                self.ebq_u0 = self.nModel.ebq[('u', 0)]
            self.ebqe_u0 = self.nModel.ebqe[('u', 0)]
            self.dof_u0 = self.nModel.u[0].dof
        else:
            self.nModel = None
        self.rdModel = modelList[self.rdModelId]

    def initializeMesh(self, mesh):
        self.h = mesh.h
        self.eps = self.epsFact * mesh.h

    def initializeElementQuadrature(self, t, cq):
        if self.nModelId is None:
            if self.q_u0 is None:
                self.q_u0 = np.zeros(cq[('u', 0)].shape, 'd')
            if self.u0 is not None:
                for i in range(len(cq[('u', 0)].flat)):
                    self.q_u0.flat[i] = self.u0.uOfXT(cq['x'].flat[3 * i:3 * (i + 1)], 0.)

    def initializeElementBoundaryQuadrature(self, t, cebq, cebq_global):
        if self.nModelId is None:
            if self.ebq_u0 is None:
                self.ebq_u0 = np.zeros(cebq[('u', 0)].shape, 'd')
            if self.u0 is not None:
                for i in range(len(cebq[('u', 0)].flat)):
                    self.ebq_u0.flat[i] = self.u0.uOfXT(cebq['x'].flat[3 * i:3 * (i + 1)], 0.)

    def initializeGlobalExteriorElementBoundaryQuadrature(self, t, cebqe):
        if self.nModelId is None:
            if self.ebqe_u0 is None:
                self.ebqe_u0 = np.zeros(cebqe[('u', 0)].shape, 'd')
            if self.u0 is not None:
                for i in range(len(cebqe[('u', 0)].flat)):
                    self.ebqe_u0.flat[i] = self.u0.uOfXT(cebqe['x'].flat[3 * i:3 * (i + 1)], 0.)

    def preStep(self, t, firstStep=False):
        # reset ellipticStage flag
        self.rdModel.stage = 1
        self.rdModel.auxEllipticFlag = 1
        # COMPUTE NORMAL RECONSTRUCTION
        if self.ELLIPTIC_REDISTANCING == 2: # linear via C0 normal reconstruction
            self.rdModel.getNormalReconstruction()
        # END OF NORMAL RECONSTRUCTION #

        if self.nModel is not None:
            logEvent("resetting signed distance level set to current level set", level=2)
            self.rdModel.u[0].dof[:] = self.nModel.u[0].dof[:]
            self.rdModel.calculateCoefficients()
            self.rdModel.calculateElementResidual()
            self.rdModel.timeIntegration.updateTimeHistory(resetFromDOF=True)
            self.rdModel.timeIntegration.resetTimeHistory(resetFromDOF=True)
            self.rdModel.updateTimeHistory(t, resetFromDOF=True)
            # now do again because of subgrid error lagging
            #\todo modify subgrid error lagging so this won't be necessary
            self.rdModel.calculateCoefficients()
            self.rdModel.calculateElementResidual()
            self.rdModel.timeIntegration.updateTimeHistory(resetFromDOF=True)
            self.rdModel.timeIntegration.resetTimeHistory(resetFromDOF=True)
            self.rdModel.updateTimeHistory(t, resetFromDOF=True)
            #copyInstructions = {'copy_uList': False,
            #                     'uList_model': self.nModelId}
            #copyInstructions = {'reset_uList': False}
            copyInstructions = {'copy_uList': self.copyList,
                                'uList_model': self.nModelId}
            copyInstructions = {'reset_uList': self.copyList}
            return copyInstructions
        else:
            return {}

    def postStep(self, t, firstStep=False):
        self.rdModel.quantDOFs[:] = self.rdModel.interface_locator
        if self.nModel is not None:
            if self.applyRedistancing == True:
                logEvent("resetting level set to signed distance")
                self.nModel.u[0].dof.flat[:] = self.rdModel.u[0].dof.flat[:]
                self.nModel.calculateCoefficients()
                self.nModel.calculateElementResidual()
                # save the boundary level set in the numerical flux to use for
                self.nModel.numericalFlux.ebqe[('u', 0)][:] = self.rdModel.ebqe[('u', 0)]
            copyInstructions = {}
            return copyInstructions
        else:
            return {}

    def getICDofs(self, cj):
        return self.dof_u0

    def updateToMovingDomain(self, t, c):
        # the redistancing equations is not physical so it just needs the updated mesh
        pass

    def evaluate(self, t, c):
        if c[('H', 0)].shape == self.q_u0.shape:
            u0 = self.q_u0
            # print "numdiff",c[('numDiff',0,0)].min(),c[('numDiff',0,0)].max()
            # print "tau",self.rdModel.stabilization.tau[0].min(),self.rdModel.stabilization.tau[0].max(),
        elif c[('H', 0)].shape == self.ebqe_u0.shape:
            u0 = self.ebqe_u0
        else:
            u0 = self.ebq_u0
        assert u0 is not None
        # \todo make redistancing epsilon depend on local element diamater instead of global max
        self.redistanceLevelSetCoefficientsEvaluate(self.eps,
                                                    u0,
                                                    c[('u', 0)],
                                                    c[('grad(u)', 0)],
                                                    c[('m', 0)],
                                                    c[('dm', 0, 0)],
                                                    c[('H', 0)],
                                                    c[('dH', 0, 0)],
                                                    c[('r', 0)])
    # weak Dirichlet conditions on level set (boundary conditions of Eikonal equation)
    # \todo clean up weak Dirichlet conditions for Eikonal equation in transport coefficents

    def setZeroLSweakDirichletBCs(vt):
        if vt.coefficients.weakBC_on:
            #print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!=========================Setting new weak BC's=======================!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            vt.dirichletNodeSetList[0] = []
            vt.dirichletGlobalNodeSet[0] = set()
            vt.dirichletValues[0] = {}
            for eN in range(vt.mesh.nElements_global):
                vt.dirichletNodeSetList[0].append(set())
                signU = 0
                j0 = 0
                eps = vt.u[0].femSpace.mesh.elementDiametersArray[eN]
                # loop over nodes looking for a node not within eps of zero
                while ((signU == 0) and
                       (j0 < vt.nDOF_trial_element[0])):
                    J0 = vt.u[0].femSpace.dofMap.l2g[eN, j0]
                    if vt.u[0].dof[J0] < -eps:
                        signU = -1
                    elif vt.u[0].dof[J0] > eps:
                        signU = 1
                    else:  # freeze this node within eps of zero
                        vt.dirichletNodeSetList[0][eN].add(j0)
                        vt.dirichletValues[0][(eN, j0)] = vt.u[0].dof[J0]
                        vt.dirichletGlobalNodeSet[0].add(J0)
                    j0 += 1
                # loop over remaining nodes to see if the zero level set cuts element
                for j in range(j0, vt.nDOF_trial_element[0]):
                    J = vt.u[0].femSpace.dofMap.l2g[eN, j]
                    if (((vt.u[0].dof[J] < -eps) and
                         (signU == 1)) or
                        ((vt.u[0].dof[J] > eps) and
                         (signU == -1))):  # level set cuts element, freeze whole element
                        for jj in range(vt.nDOF_trial_element[0]):
                            JJ = vt.u[0].femSpace.dofMap.l2g[eN, jj]
                            vt.dirichletNodeSetList[0][eN].add(jj)
                            vt.dirichletValues[0][(eN, jj)] = float(vt.u[0].dof[JJ])
                            vt.dirichletGlobalNodeSet[0].add(JJ)
                        break
                    elif (fabs(vt.u[0].dof[J]) < eps):  # freeze this node within eps of zero
                        vt.dirichletNodeSetList[0][eN].add(j)
                        vt.dirichletValues[0][(eN, j)] = float(vt.u[0].dof[J])
                        vt.dirichletGlobalNodeSet[0].add(J)
            # get all frozen dof and make sure they're frozen on each element
            for eN in range(vt.mesh.nElements_global):
                for j in range(vt.nDOF_trial_element[0]):
                    J = vt.u[0].femSpace.dofMap.l2g[eN, j]
                    if J in vt.dirichletGlobalNodeSet[0]:
                        vt.dirichletNodeSetList[0][eN].add(j)
                        vt.dirichletValues[0][(eN, j)] = float(vt.u[0].dof[J])
        else:
            #print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!=========================Unsetting weak BC's=======================!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            vt.dirichletNodeSetList[0] = []
            vt.dirichletGlobalNodeSet[0] = set()
            vt.dirichletValues[0] = {}
            for eN in range(vt.mesh.nElements_global):
                vt.dirichletNodeSetList[0].append(set())

    def setZeroLSweakDirichletBCs2(vt):
        # just look for cut edges and nodes
        vt.dirichletNodeSetList[0] = []
        vt.dirichletGlobalNodeSet[0] = set()
        vt.dirichletValues[0] = {}
        for eN in range(vt.mesh.nElements_global):
            vt.dirichletNodeSetList[0].append(set())
            signU = 0
            j0 = 0
            eps = .1 * vt.u[0].femSpace.mesh.elementDiametersArray[eN]
            eps = 0.0
            for j0 in range(vt.nDOF_trial_element[0]):
                J0 = vt.u[0].femSpace.dofMap.l2g[eN, j0]
                if vt.u[0].dof[J0] < -eps:
                    signU = -1
                elif vt.u[0].dof[J0] > eps:
                    signU = 1
                else:
                    vt.dirichletNodeSetList[0][eN].add(j0)
                    vt.dirichletValues[0][(eN, j0)] = float(vt.u[0].dof[J0])
                    vt.dirichletGlobalNodeSet[0].add(J0)
                if signU != 0:
                    for j in (list(range(0, j0)) + list(range(j0 + 1, vt.nDOF_trial_element[0]))):
                        J = vt.u[0].femSpace.dofMap.l2g[eN, j]
                        if (((vt.u[0].dof[J] < -eps) and
                             (signU == 1)) or
                            ((vt.u[0].dof[J] > eps) and
                             (signU == -1))):
                            vt.dirichletNodeSetList[0][eN].add(j)
                            vt.dirichletValues[0][(eN, j)] = float(vt.u[0].dof[J])
                            vt.dirichletGlobalNodeSet[0].add(J)
                            vt.dirichletNodeSetList[0][eN].add(j0)
                            vt.dirichletValues[0][(eN, j0)] = float(vt.u[0].dof[j0])
                            vt.dirichletGlobalNodeSet[0].add(j0)
        for eN in range(vt.mesh.nElements_global):
            for j in range(vt.nDOF_trial_element[0]):
                J = vt.u[0].femSpace.dofMap.l2g[eN, j]
                if J in vt.dirichletGlobalNodeSet[0]:
                    vt.dirichletNodeSetList[0][eN].add(j)
                    vt.dirichletValues[0][(eN, j)] = float(vt.u[0].dof[J])

    def setZeroLSweakDirichletBCs3(vt):
        if vt.coefficients.weakBC_on:
            #print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!=========================Setting new weak BC's=======================!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            vt.dirichletNodeSetList[0] = []
            vt.dirichletGlobalNodeSet[0] = set()
            vt.dirichletValues[0] = {}
            for eN in range(vt.mesh.nElements_global):
                vt.dirichletNodeSetList[0].append(set())
                eps = vt.coefficients.epsFact * vt.u[0].femSpace.mesh.elementDiametersArray[eN]
                #eps = 1.5*vt.u[0].femSpace.mesh.elementDiametersArray[eN]
                # loop over nodes looking for a node not within eps of zero
                for j in range(vt.nDOF_trial_element[0]):
                    J = vt.u[0].femSpace.dofMap.l2g[eN, j]
                    if (fabs(vt.u[0].dof[J]) < eps):  # freeze this node within eps of zero
                        vt.dirichletNodeSetList[0][eN].add(j)
                        vt.dirichletValues[0][(eN, j)] = float(vt.u[0].dof[J])
                        vt.dirichletGlobalNodeSet[0].add(J)
        else:
            #print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!=========================Unsetting weak BC's=======================!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            vt.dirichletNodeSetList[0] = []
            vt.dirichletGlobalNodeSet[0] = set()
            vt.dirichletValues[0] = {}
            for eN in range(vt.mesh.nElements_global):
                vt.dirichletNodeSetList[0].append(set())

    # def
    setZeroLSweakDirichletBCs = staticmethod(setZeroLSweakDirichletBCs)
    setZeroLSweakDirichletBCs2 = staticmethod(setZeroLSweakDirichletBCs2)
    setZeroLSweakDirichletBCs3 = staticmethod(setZeroLSweakDirichletBCs3)


debugRDLS = False  # True


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
                 movingDomain=False,
                 bdyNullSpace=False):
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
        # mesh
        self.q['x'] = np.zeros((self.mesh.nElements_global,self.nQuadraturePoints_element,3),'d')
        self.ebqe['x'] = np.zeros((self.mesh.nExteriorElementBoundaries_global, self.nElementBoundaryQuadraturePoints_elementBoundary, 3), 'd')
        self.q[('u', 0)] = np.zeros((self.mesh.nElements_global, self.nQuadraturePoints_element), 'd')
        self.q[('grad(u)', 0)] = np.zeros((self.mesh.nElements_global, self.nQuadraturePoints_element, self.nSpace_global), 'd')
        self.q[('m_last', 0)] = np.zeros((self.mesh.nElements_global, self.nQuadraturePoints_element), 'd')
        self.q[('m_tmp', 0)] = np.zeros((self.mesh.nElements_global, self.nQuadraturePoints_element), 'd')
        self.q[('m', 0)] = self.q[('u', 0)]  # for time integration by VBDF and probably FLCBDF
        # needed by PsiTCtte
        self.q[('mt', 0)] = np.zeros(self.q[('u', 0)].shape, 'd')
        self.q[('dH', 0, 0)] = np.zeros((self.mesh.nElements_global, self.nQuadraturePoints_element, self.nSpace_global), 'd')
        self.q[('dH_sge', 0, 0)] = np.zeros((self.mesh.nElements_global, self.nQuadraturePoints_element, self.nSpace_global), 'd')
        self.q[('cfl', 0)] = np.zeros((self.mesh.nElements_global, self.nQuadraturePoints_element), 'd')
        self.q[('numDiff', 0, 0)] = np.zeros((self.mesh.nElements_global, self.nQuadraturePoints_element), 'd')
        self.ebqe[('u', 0)] = np.zeros((self.mesh.nExteriorElementBoundaries_global, self.nElementBoundaryQuadraturePoints_elementBoundary), 'd')
        self.ebqe[('grad(u)', 0)] = np.zeros((self.mesh.nExteriorElementBoundaries_global,
                                                 self.nElementBoundaryQuadraturePoints_elementBoundary, self.nSpace_global), 'd')
        self.points_elementBoundaryQuadrature = set()
        self.scalars_elementBoundaryQuadrature = set([('u', ci) for ci in range(self.nc)])
        self.vectors_elementBoundaryQuadrature = set()
        self.tensors_elementBoundaryQuadrature = set()
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

        # add some structures for elliptic re-distancing
        self.quantDOFs = np.zeros(self.u[0].dof.shape, 'd')
        self.interface_locator = None
        self.lumped_qx = np.zeros(self.u[0].dof.shape,'d')
        self.lumped_qy = np.zeros(self.u[0].dof.shape,'d')
        self.lumped_qz = np.zeros(self.u[0].dof.shape,'d')
        self._dofVecs = {}
        self.stage = 1
        self.auxEllipticFlag = 1
        if self.coefficients.ELLIPTIC_REDISTANCING==2:
            self.useTwoStageNewton = True

        # ASSERT: If ELLIPTIC_REDISTANCING>0 make sure that a two stage newton solver is used
        if self.coefficients.ELLIPTIC_REDISTANCING>0:
            assert options.levelNonlinearSolver == proteus.NonlinearSolvers.TwoStageNewton, "If ELLIPTIC REDISTANCING>0, use levelNonlinearSolver=TwoStageNewton"
        # ASSERT: if order>1 and ELLIPTIC_REDISTANCING>0 then use bernstein polynomials
        if self.coefficients.ELLIPTIC_REDISTANCING>0 and self.u[0].femSpace.order>1:
            isBernsteinOnCube = isinstance(self.u[0].femSpace,
                                           proteus.FemTools.C0_AffineBernsteinOnCube)
            isBernsteinOnSimplex = isinstance(self.u[0].femSpace,
                                              proteus.FemTools.C0_AffineBernsteinOnSimplex)
            isBernstein = isBernsteinOnCube or isBernsteinOnSimplex
            assert isBernstein==True, "If order>1 and ELLIPTIC_REDISTANCING=True, use Bernstein polynomials"

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
        for ci, fbcObject in list(self.fluxBoundaryConditionsObjectsDict.items()):
            self.ebqe[('advectiveFlux_bc_flag', ci)] = np.zeros(self.ebqe[('advectiveFlux_bc', ci)].shape, 'i')
            for t, g in list(fbcObject.advectiveFluxBoundaryConditionsDict.items()):
                if ci in self.coefficients.advection:
                    self.ebqe[('advectiveFlux_bc', ci)][t[0], t[1]] = g(self.ebqe[('x')][t[0], t[1]], self.timeIntegration.t)
                    self.ebqe[('advectiveFlux_bc_flag', ci)][t[0], t[1]] = 1
        # reduced quad
        if hasattr(self.numericalFlux, 'setDirichletValues'):
            self.numericalFlux.setDirichletValues(self.ebqe)
        if not hasattr(self.numericalFlux, 'isDOFBoundary'):
            self.numericalFlux.isDOFBoundary = {0: np.zeros(self.ebqe[('u', 0)].shape, 'i')}
        if not hasattr(self.numericalFlux, 'ebqe'):
            self.numericalFlux.ebqe = {('u', 0): np.zeros(self.ebqe[('u', 0)].shape, 'd')}
        # TODO how to handle redistancing calls for calculateCoefficients,calculateElementResidual etc
        self.globalResidualDummy = None
        # create storage for enforcing weak dirichlet boundary conditions around zero level set contour
        self.freezeLevelSet = 1  # True
        self.u_dof_last = np.zeros(self.u[0].dof.shape, 'd')
        self.weakDirichletConditionFlags = np.zeros(self.u[0].dof.shape, 'i')
        self.dofFlag_element = np.zeros((self.nDOF_trial_element[0],), 'i')
        # allow Newton solves for redistancing
        if self.timeIntegration.__class__ == TimeIntegration.NoIntegration:
            self.timeIntegration.m_tmp = {0: np.zeros(self.q[('m_tmp', 0)].shape, 'd')}
            self.timeIntegration.m_last = {0: np.zeros(self.q[('m_tmp', 0)].shape, 'd')}
        compKernelFlag = 0
        if self.coefficients.useConstantH:
            self.elementDiameter = self.mesh.elementDiametersArray.copy()
            self.elementDiameter[:] = max(self.mesh.elementDiametersArray)
        else:
            self.elementDiameter = self.mesh.elementDiametersArray
        self.rdls = cRDLS_base(self.nSpace_global,
                               self.nQuadraturePoints_element,
                               self.u[0].femSpace.elementMaps.localFunctionSpace.dim,
                               self.u[0].femSpace.referenceFiniteElement.localFunctionSpace.dim,
                               self.testSpace[0].referenceFiniteElement.localFunctionSpace.dim,
                               self.nElementBoundaryQuadraturePoints_elementBoundary,
                               compKernelFlag)

        ###########
        # METRICS #
        ###########
        self.hasExactSolution = False
        if ('exactSolution') in dir (options):
            self.hasExactSolution = True
            self.exactSolution = options.exactSolution

        # metrics
        self.global_I_err = 0.0
        self.global_V_err = 0.0
        self.global_D_err = 0.0
        if self.coefficients.computeMetrics:
            self.metricsAtEOS = open(self.name+"_metricsAtEOS.csv","w")
            self.metricsAtEOS.write('global_I_err'+","+
                                    'global_V_err'+","+
                                    'global_D_err'+"\n")

    ####################################3
    def runAtEOS(self):
        if self.coefficients.computeMetrics==True and self.hasExactSolution==True:
            # Get exact solution at quad points
            u_exact = np.zeros(self.q[('u',0)].shape,'d')
            X = {0:self.q[('x')][:,:,0],
                 1:self.q[('x')][:,:,1],
                 2:self.q[('x')][:,:,2]}
            t = self.timeIntegration.t
            u_exact[:] = self.exactSolution[0](X,t)
            self.getMetricsAtEOS(u_exact)
            self.metricsAtEOS.write(repr(self.global_I_err)+","+
                                    repr(self.global_V_err)+","+
                                    repr(self.global_D_err)+"\n")
            self.metricsAtEOS.flush()

    def getMetricsAtEOS(self,u_exact):
        import copy
        """
        Calculate the element residuals and add in to the global residual
        """
        degree_polynomial = 1
        try:
            degree_polynomial = self.u[0].femSpace.order
        except:
            pass

        argsDict = cArgumentsDict.ArgumentsDict()
        argsDict["mesh_trial_ref"] = self.u[0].femSpace.elementMaps.psi
        argsDict["mesh_grad_trial_ref"] = self.u[0].femSpace.elementMaps.grad_psi
        argsDict["mesh_dof"] = self.mesh.nodeArray
        argsDict["mesh_l2g"] = self.mesh.elementNodesArray
        argsDict["dV_ref"] = self.elementQuadratureWeights[('u',0)]
        argsDict["u_trial_ref"] = self.u[0].femSpace.psi
        argsDict["u_grad_trial_ref"] = self.u[0].femSpace.grad_psi
        argsDict["u_test_ref"] = self.u[0].femSpace.psi
        argsDict["nElements_global"] = self.mesh.nElements_global
        argsDict["u_l2g"] = self.u[0].femSpace.dofMap.l2g
        argsDict["elementDiameter"] = self.mesh.elementDiametersArray
        argsDict["degree_polynomial"] = degree_polynomial
        argsDict["epsFact_redist"] = self.coefficients.epsFact
        argsDict["u_dof"] = self.u[0].dof
        argsDict["u_exact"] = u_exact
        argsDict["offset_u"] = self.offset[0]
        argsDict["stride_u)"] = self.stride[0]
        (self.global_I_err,
         self.global_V_err,
         self.global_D_err) = self.rdls.calculateMetricsAtEOS(argsDict)

    ###############################################

    def calculateCoefficients(self):
        pass

    def calculateElementResidual(self):
        pass

    def getNormalReconstruction(self):
        argsDict = cArgumentsDict.ArgumentsDict()
        argsDict["mesh_trial_ref"] = self.u[0].femSpace.elementMaps.psi
        argsDict["mesh_grad_trial_ref"] = self.u[0].femSpace.elementMaps.grad_psi
        argsDict["mesh_dof"] = self.mesh.nodeArray
        argsDict["mesh_l2g"] = self.mesh.elementNodesArray
        argsDict["dV_ref"] = self.elementQuadratureWeights[('u',0)]
        argsDict["u_trial_ref"] = self.u[0].femSpace.psi
        argsDict["u_grad_trial_ref"] = self.u[0].femSpace.grad_psi
        argsDict["u_test_ref"] = self.u[0].femSpace.psi
        argsDict["nElements_global"] = self.mesh.nElements_global
        argsDict["u_l2g"] = self.u[0].femSpace.dofMap.l2g
        argsDict["elementDiameter"] = self.mesh.elementDiametersArray
        argsDict["u_dof"] = self.u[0].dof
        argsDict["offset_u"] = self.offset[0]
        argsDict["stride_u"] = self.stride[0]
        argsDict["numDOFs"] = self.nFreeDOF_global[0]
        argsDict["lumped_qx"] = self.lumped_qx
        argsDict["lumped_qy"] = self.lumped_qy
        argsDict["lumped_qz"] = self.lumped_qz
        self.rdls.normalReconstruction(argsDict)
        # a lumped projection: a ghost node sums only the elements this rank has, and owned rows read it
        for q in (self.lumped_qx, self.lumped_qy, self.lumped_qz):
            self.scatterDOFArray(q)

    def scatterDOFArray(self, a):
        """Replace the ghost entries of a dof array by the owners' values (in place)."""
        if self.comm.size() == 1:
            return
        cached = self._dofVecs.get(a.shape)
        if cached is None:
            dofMap = self.u[0].femSpace.dofMap
            rank = self.comm.rank()
            nOwned = dofMap.dof_offsets_subdomain_owned[rank + 1] - dofMap.dof_offsets_subdomain_owned[rank]
            buf = a.copy()
            cached = self._dofVecs[a.shape] = (buf, ParVec_petsc4py(
                buf, 1, nOwned, dofMap.nDOF_all_processes, dofMap.nDOF_subdomain - nOwned, dofMap.subdomain2global))
        buf, par = cached
        buf[:] = a
        par.scatter_forward_insert()
        a[:] = buf

    def getResidual(self, u, r):
        import pdb
        import copy

        if self.interface_locator is None:
            if self.coefficients.nModel is not None:
                self.interface_locator = self.coefficients.nModel.interface_locator
            else:
                self.interface_locator = np.zeros(self.u[0].dof.shape,'d')

        # try to use 1d,2d,3d specific modules
        r.fill(0.0)
        # Load the unknowns into the finite element dof
        self.timeIntegration.calculateCoefs()
        self.timeIntegration.calculateU(u)
        self.setUnknowns(self.timeIntegration.u)
        # cek can put in logic to skip of BC's don't depend on t or u
        # Dirichlet boundary conditions
        if hasattr(self.numericalFlux, 'setDirichletValues'):
            self.numericalFlux.setDirichletValues(self.ebqe)

        # flux boundary conditions, SHOULDN'T HAVE
        # for  now force time integration
        useTimeIntegration = 1
        #assert self.timeIntegration.__class__ != TimeIntegration.NoIntegration
        if self.timeIntegration.__class__ == TimeIntegration.NoIntegration or not self.timeTerm:
            useTimeIntegration = 0

        if useTimeIntegration:
            alpha_bdf = self.timeIntegration.alpha_bdf
            beta_bdf = self.timeIntegration.beta_bdf
        else:
            alpha_bdf = self.timeIntegration.dt
            beta_bdf = self.timeIntegration.m_last
        # self.elementResidual[0].fill(0.0)
        # cRDLS.calculateResidual(self.mesh.nElements_global,
        # print "beta_bdf",beta_bdf

        if self.coefficients.ELLIPTIC_REDISTANCING>0:
            self.calculateResidual = self.rdls.calculateResidual_ellipticRedist
            self.calculateJacobian = self.rdls.calculateJacobian_ellipticRedist
            # COMPUTE RECONSTRUCTIONS #
            if self.coefficients.ELLIPTIC_REDISTANCING == 3: #nlinear via C0 normal rec.
                self.getNormalReconstruction()
            if (self.coefficients.ELLIPTIC_REDISTANCING == 2 # linear via C0 normal rec.
                and self.stage == 2 and self.auxEllipticFlag == 1):
                self.getNormalReconstruction()
                self.auxEllipticFlag = 0
        else:
            self.calculateResidual = self.rdls.calculateResidual
            self.calculateJacobian = self.rdls.calculateJacobian

        # FREEZE INTERFACE #
        if self.coefficients.freeze_interface_within_elliptic_redist==True:
            for gi in range(len(self.u[0].dof)):
                if self.interface_locator[gi] == 1.0:
                    self.u[0].dof[gi] = self.coefficients.dof_u0[gi]
        # END OF FREEZING INTERFACE #
        argsDict = cArgumentsDict.ArgumentsDict()
        argsDict["mesh_trial_ref"] = self.u[0].femSpace.elementMaps.psi
        argsDict["mesh_grad_trial_ref"] = self.u[0].femSpace.elementMaps.grad_psi
        argsDict["mesh_dof"] = self.mesh.nodeArray
        argsDict["mesh_l2g"] = self.mesh.elementNodesArray
        argsDict["x_ref"] = self.elementQuadraturePoints
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
        argsDict["alphaBDF"] = alpha_bdf
        argsDict["epsFact_redist"] = self.coefficients.epsFact
        argsDict["backgroundDiffusionFactor"] = self.coefficients.backgroundDiffusionFactor
        argsDict["weakDirichletFactor"] = self.coefficients.weakDirichletFactor
        argsDict["freezeLevelSet"] = self.freezeLevelSet
        argsDict["useTimeIntegration"] = useTimeIntegration
        argsDict["lag_shockCapturing"] = self.shockCapturing.lag
        argsDict["lag_subgridError"] = self.stabilization.lag
        argsDict["shockCapturingDiffusion"] = float(self.shockCapturing.shockCapturingFactor)
        argsDict["u_l2g"] = self.u[0].femSpace.dofMap.l2g
        argsDict["elementDiameter"] = self.elementDiameter
        argsDict["nodeDiametersArray"] = self.mesh.nodeDiametersArray
        argsDict["u_dof"] = self.u[0].dof
        argsDict["phi_dof"] = self.coefficients.dof_u0
        argsDict["phi_ls"] = self.coefficients.q_u0
        argsDict["q_m"] = self.timeIntegration.m_tmp[0]
        argsDict["q_u"] = self.q[('u', 0)]
        argsDict["q_n"] = self.q[('grad(u)', 0)]
        argsDict["q_dH"] = self.q[('dH', 0, 0)]
        argsDict["u_weak_internal_bc_dofs"] = self.u_dof_last
        argsDict["q_m_betaBDF"] = beta_bdf[0]
        argsDict["q_dH_last"] = self.q[('dH_sge', 0, 0)]
        argsDict["q_cfl"] = self.q[('cfl', 0)]
        argsDict["q_numDiff_u"] = self.shockCapturing.numDiff[0]
        argsDict["q_numDiff_u_last"] = self.shockCapturing.numDiff_last[0]
        argsDict["weakDirichletConditionFlags"] = self.weakDirichletConditionFlags
        argsDict["offset_u"] = self.offset[0]
        argsDict["stride_u"] = self.stride[0]
        argsDict["globalResidual"] = r
        argsDict["nExteriorElementBoundaries_global"] = self.mesh.nExteriorElementBoundaries_global
        argsDict["exteriorElementBoundariesArray"] = self.mesh.exteriorElementBoundariesArray
        argsDict["elementBoundaryElementsArray"] = self.mesh.elementBoundaryElementsArray
        argsDict["elementBoundaryLocalElementBoundariesArray"] = self.mesh.elementBoundaryLocalElementBoundariesArray
        argsDict["ebqe_phi_ls_ext"] = self.coefficients.ebqe_u0
        argsDict["isDOFBoundary_u"] = self.numericalFlux.isDOFBoundary[0]
        argsDict["ebqe_bc_u_ext"] = self.numericalFlux.ebqe[('u', 0)]
        argsDict["ebqe_u"] = self.ebqe[('u', 0)]
        argsDict["ebqe_n"] = self.ebqe[('grad(u)', 0)]
        argsDict["ELLIPTIC_REDISTANCING"] = self.coefficients.ELLIPTIC_REDISTANCING
        argsDict["backgroundDissipationEllipticRedist"] = self.coefficients.backgroundDissipationEllipticRedist
        argsDict["lumped_qx"] = self.lumped_qx
        argsDict["lumped_qy"] = self.lumped_qy
        argsDict["lumped_qz"] = self.lumped_qz
        argsDict["alpha"] = self.coefficients.alpha/self.elementDiameter.min()
        self.calculateResidual(argsDict, self.coefficients.useExact)



        # FREEZE INTERFACE #
        if self.coefficients.freeze_interface_within_elliptic_redist==True:
            for gi in range(len(self.u[0].dof)):
                if self.interface_locator[gi] == 1.0:
                    r[gi] = 0
        # END OF FREEZING INTERFACE #

        # print "m_tmp",self.timeIntegration.m_tmp[0]
        # print "dH",self.q[('dH',0,0)]
        # print "dH_sge",self.q[('dH_sge',0,0)]
        if self.stabilization:
            self.stabilization.accumulateSubgridMassHistory(self.q)
        logEvent("Global residual", level=9, data=r)
        # mwf decide if this is reasonable for keeping solver statistics
        self.nonlinear_function_evaluations += 1
        if self.globalResidualDummy is None:
            self.globalResidualDummy = np.zeros(r.shape, 'd')

    def getJacobian(self, jacobian):
        if not debugRDLS:
            return self.getJacobianNew(jacobian)
        OneLevelTransport.getJacobian(self, jacobian)

    def getJacobianNew(self, jacobian):
        #import superluWrappers
        #import np
        import pdb
        cfemIntegrals.zeroJacobian_CSR(self.nNonzerosInJacobian,jacobian)

        # for  now force time integration
        useTimeIntegration = 1
        if self.timeIntegration.__class__ == TimeIntegration.NoIntegration or not self.timeTerm:
            useTimeIntegration = 0
        if useTimeIntegration:
            alpha_bdf = self.timeIntegration.alpha_bdf
            beta_bdf = self.timeIntegration.beta_bdf
        else:
            alpha_bdf = self.timeIntegration.dt
            beta_bdf = self.timeIntegration.m_last

        argsDict = cArgumentsDict.ArgumentsDict()
        argsDict["mesh_trial_ref"] = self.u[0].femSpace.elementMaps.psi
        argsDict["mesh_grad_trial_ref"] = self.u[0].femSpace.elementMaps.grad_psi
        argsDict["mesh_dof"] = self.mesh.nodeArray
        argsDict["mesh_l2g"] = self.mesh.elementNodesArray
        argsDict["x_ref"] = self.elementQuadraturePoints
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
        argsDict["alphaBDF"] = alpha_bdf
        argsDict["epsFact_redist"] = self.coefficients.epsFact
        argsDict["backgroundDiffusionFactor"] = self.coefficients.backgroundDiffusionFactor
        argsDict["weakDirichletFactor"] = self.coefficients.weakDirichletFactor
        argsDict["freezeLevelSet"] = self.freezeLevelSet
        argsDict["useTimeIntegration"] = useTimeIntegration
        argsDict["lag_shockCapturing"] = self.shockCapturing.lag
        argsDict["lag_subgridError"] = int(self.stabilization.lag)
        argsDict["shockCapturingDiffusion"] = float(self.shockCapturing.shockCapturingFactor)
        argsDict["u_l2g"] = self.u[0].femSpace.dofMap.l2g
        argsDict["elementDiameter"] = self.elementDiameter
        argsDict["nodeDiametersArray"] = self.mesh.nodeDiametersArray
        argsDict["u_dof"] = self.u[0].dof
        argsDict["phi_dof"] = self.coefficients.dof_u0
        argsDict["u_weak_internal_bc_dofs"] = self.u_dof_last
        argsDict["phi_ls"] = self.coefficients.q_u0
        argsDict["q_m_betaBDF"] = beta_bdf[0]
        argsDict["q_dH_last"] = self.q[('dH_sge', 0, 0)]
        argsDict["q_cfl"] = self.q[('cfl', 0)]
        argsDict["q_numDiff_u"] = self.shockCapturing.numDiff[0]
        argsDict["q_numDiff_u_last"] = self.shockCapturing.numDiff_last[0]
        argsDict["weakDirichletConditionFlags"] = self.weakDirichletConditionFlags
        argsDict["csrRowIndeces_u_u"] = self.csrRowIndeces[(0, 0)]
        argsDict["csrColumnOffsets_u_u"] = self.csrColumnOffsets[(0, 0)]
        argsDict["globalJacobian"] = jacobian.getCSRrepresentation()[2]
        argsDict["nExteriorElementBoundaries_global"] = self.mesh.nExteriorElementBoundaries_global
        argsDict["exteriorElementBoundariesArray"] = self.mesh.exteriorElementBoundariesArray
        argsDict["elementBoundaryElementsArray"] = self.mesh.elementBoundaryElementsArray
        argsDict["elementBoundaryLocalElementBoundariesArray"] = self.mesh.elementBoundaryLocalElementBoundariesArray
        argsDict["ebqe_phi_ls_ext"] = self.coefficients.ebqe_u0
        argsDict["isDOFBoundary_u"] = self.numericalFlux.isDOFBoundary[0]
        argsDict["ebqe_bc_u_ext"] = self.numericalFlux.ebqe[('u', 0)]
        argsDict["csrColumnOffsets_eb_u_u"] = self.csrColumnOffsets_eb[(0, 0)]
        argsDict["ELLIPTIC_REDISTANCING"] = self.coefficients.ELLIPTIC_REDISTANCING
        argsDict["backgroundDissipationEllipticRedist"] = self.coefficients.backgroundDissipationEllipticRedist
        argsDict["alpha"] = float(self.coefficients.alpha/self.elementDiameter.min())
        self.calculateJacobian(argsDict, self.coefficients.useExact)

        # FREEZING INTERFACE #
        if self.coefficients.freeze_interface_within_elliptic_redist==True:
            for gi in range(len(self.u[0].dof)):
                if self.interface_locator[gi] == 1.0:
                    for i in range(self.rowptr[gi], self.rowptr[gi + 1]):
                        if (self.colind[i] == gi):
                            self.nzval[i] = 1.0
                        else:
                            self.nzval[i] = 0.0
        # END OF FREEZING INTERFACE #

        logEvent("Jacobian ", level=10, data=jacobian)
        # mwf decide if this is reasonable for solver statistics
        self.nonlinear_function_jacobian_evaluations += 1
        return jacobian
    # jacobian

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

    def updateAfterMeshMotion(self):
        pass


# OneLevelRDLS
from proteus import ctransportCoefficients


def setZeroLSweakDirichletBCs(RDLSvt):
    assert hasattr(RDLSvt, 'freezeLevelSet')
    assert hasattr(RDLSvt, 'u_dof_last')
    assert hasattr(RDLSvt, 'weakDirichletConditionFlags')
    assert hasattr(RDLSvt.coefficients, 'epsFact')
    assert hasattr(RDLSvt, 'dofFlag_element')
    RDLSvt.freezeLevelSet = 1
    useC = True  # True
    if debugRDLS:
        useC = False
    RDLSvt.u_dof_last[:] = RDLSvt.u[0].dof
    RDLSvt.weakDirichletConditionFlags.fill(0)
    # to debug use original implementation and then copy over
    if not useC:
        RDLSvt.coefficients.setZeroLSweakDirichletBCs(RDLSvt)
        for eN in range(RDLSvt.mesh.nElements_global):
            for j in range(RDLSvt.nDOF_trial_element[0]):
                J = RDLSvt.u[0].femSpace.dofMap.l2g[eN, j]
                if J in RDLSvt.dirichletGlobalNodeSet[0]:
                    RDLSvt.weakDirichletConditionFlags[J] = 1
    else:
        #
        # use c directly
        ctransportCoefficients.setWeakDirichletConditionsForLevelSet(RDLSvt.mesh.nElements_global,
                                                                     RDLSvt.nDOF_trial_element[0],
                                                                     RDLSvt.coefficients.epsFact,
                                                                     RDLSvt.elementDiameter,  # RDLSvt.mesh.elementDiametersArray,
                                                                     RDLSvt.u[0].femSpace.dofMap.l2g,
                                                                     RDLSvt.u[0].dof,
                                                                     RDLSvt.dofFlag_element,  # temporary storage
                                                                     RDLSvt.weakDirichletConditionFlags)


def setZeroLSweakDirichletBCsSimple(RDLSvt):
    assert hasattr(RDLSvt, 'freezeLevelSet')
    assert hasattr(RDLSvt, 'u_dof_last')
    assert hasattr(RDLSvt, 'weakDirichletConditionFlags')
    assert hasattr(RDLSvt.coefficients, 'epsFact')
    assert hasattr(RDLSvt, 'dofFlag_element')
    RDLSvt.freezeLevelSet = 1
    useC = True  # True
    if debugRDLS:
        useC = False
    RDLSvt.u_dof_last[:] = RDLSvt.u[0].dof
    RDLSvt.weakDirichletConditionFlags.fill(0)
    # to debug use original implementation and then copy over
    if not useC:
        RDLSvt.coefficients.setZeroLSweakDirichletBCs(RDLSvt)
        for eN in range(RDLSvt.mesh.nElements_global):
            for j in range(RDLSvt.nDOF_trial_element[0]):
                J = RDLSvt.u[0].femSpace.dofMap.l2g[eN, j]
                if J in RDLSvt.dirichletGlobalNodeSet[0]:
                    RDLSvt.weakDirichletConditionFlags[J] = 1
    else:
        #
        # use c directly
        ctransportCoefficients.setSimpleWeakDirichletConditionsForLevelSet(RDLSvt.mesh.nElements_global,
                                                                           RDLSvt.nDOF_trial_element[0],
                                                                           RDLSvt.coefficients.epsFact,
                                                                           RDLSvt.elementDiameter,  # RDLSvt.mesh.elementDiametersArray,
                                                                           RDLSvt.u[0].femSpace.dofMap.l2g,
                                                                           RDLSvt.u[0].dof,
                                                                           RDLSvt.dofFlag_element,  # temporary storage
                                                                           RDLSvt.weakDirichletConditionFlags)
