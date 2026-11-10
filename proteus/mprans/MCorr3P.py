import proteus
import numpy
from proteus import *
from proteus.Transport import *
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
from . import cMCorr3P
from . import cArgumentsDict
import proteus.NonlinearSolvers
import proteus.StepControl
import proteus.Transport
import proteus.TransportCoefficients


class Coefficients(proteus.TransportCoefficients.TC_base):
    from proteus.ctransportCoefficients import levelSetConservationCoefficientsEvaluate
    from proteus.ctransportCoefficients import levelSetConservationCoefficientsEvaluate_sd
    # staticmethod: Cython 3 functions bind self like Python functions (C47)
    levelSetConservationCoefficientsEvaluate = staticmethod(levelSetConservationCoefficientsEvaluate)
    levelSetConservationCoefficientsEvaluate_sd = staticmethod(levelSetConservationCoefficientsEvaluate_sd)

    def __init__(
            self,
            applyCorrection=True,
            epsFactHeaviside=0.0,
            epsFactDirac=1.0,
            epsFactDiffusion=2.0,
            LS_model=None,
            V_model=None,
            ME_model=None,
            VOS_model=None,
            VOF_model=None,
            checkMass=True,
            sd=True,
            nd=None,
            applyCorrectionToDOF=True,
            useMetrics=0.0,
            useConstantH=False,
            set_vos=None,
            fastAssembly=False):
        # fastAssembly=False (default): the consistent element Jacobian (exact derivative of the residual).  True: the LUMPED interface mass matrix
        # precomputed in the residual (faster, M-matrix, but only approximately the derivative: ~1e-3 relative error).
        # Measured 2026-11-10 (RANS3PSed bed_sediment, refinement 50, 8134 dofs, 41 steps, MUMPS, macOS arm64): the consistent
        # assembly is ~5.5x the lumped one per call (1.3 vs 0.22 ms serial) but < 0.5% of the run; at the deck's tolerance both
        # take one Newton iteration per step, at nl_atol_res=1e-11 the consistent Jacobian takes 82 iterations against 186.
        self.fastAssembly = fastAssembly
        self.set_vos = set_vos
        self.useConstantH = useConstantH
        self.useMetrics = useMetrics
        self.sd = sd
        self.checkMass = checkMass
        self.variableNames = ['phiCorr']
        nc = 1
        mass = {}
        advection = {}
        hamiltonian = {}
        diffusion = {0: {0: {0: 'constant'}}}
        potential = {0: {0: 'u'}}
        reaction = {0: {0: 'nonlinear'}}
        # reaction={}
        if self.sd:
            assert nd is not None, "You must set the number of dimensions to use sparse diffusion in LevelSetConservationCoefficients"
            sdInfo = {(0,
                       0): (numpy.arange(start=0,
                                         stop=nd + 1,
                                         step=1,
                                         dtype='i'),
                            numpy.arange(start=0,
                                         stop=nd,
                                         step=1,
                                         dtype='i'))}
        else:
            sdInfo = {}
        TC_base.__init__(self,
                         nc,
                         mass,
                         advection,
                         diffusion,
                         potential,
                         reaction,
                         hamiltonian,
                         self.variableNames,
                         sparseDiffusionTensors=sdInfo,
                         useSparseDiffusion=sd)
        self.LS_model = LS_model
        self.V_model = V_model
        self.epsFactHeaviside = epsFactHeaviside
        self.epsFactDirac = epsFactDirac
        self.epsFactDiffusion = epsFactDiffusion
        self.ME_model = ME_model
        self.VOF_model = VOF_model
        self.VOS_model = VOS_model
        self.useC = True
        self.applyCorrection = applyCorrection
        if self.applyCorrection:
            self.applyCorrectionToDOF = applyCorrectionToDOF
        else:
            self.applyCorrectionToDOF = False
        self.massConservationError = 0.0

    def initializeMesh(self, mesh):
        self.h = mesh.h
        self.epsHeaviside = self.epsFactHeaviside * mesh.h
        self.epsDirac = self.epsFactDirac * mesh.h
        self.epsDiffusion = self.epsFactDiffusion * mesh.h

    def attachModels(self, modelList):
        import copy
        log("Attaching models in LevelSetConservation")
        # level set
        self.lsModel = modelList[self.LS_model]
        self.q_u_ls = modelList[self.LS_model].q[('u', 0)]
        self.q_n_ls = modelList[self.LS_model].q[('grad(u)', 0)]

        self.ebqe_u_ls = modelList[self.LS_model].ebqe[('u', 0)]
        self.ebqe_n_ls = modelList[
            self.LS_model].ebqe[
            ('grad(u)', 0)]

        if ('u', 0) in modelList[self.LS_model].ebq:
            self.ebq_u_ls = modelList[self.LS_model].ebq[('u', 0)]
        else:
            self.ebq_u_ls = None
        # volume of fluid
        self.vofModel = modelList[self.VOF_model]
        self.q_H_vof = modelList[self.VOF_model].q[('u', 0)]
        if self.VOS_model is not None:
            self.q_vos = modelList[self.VOS_model].q[('u', 0)]
        else:
            self.q_vos = self.vofModel.coefficients.q_vos
        self.ebqe_H_vof = modelList[self.VOF_model].ebqe[('u', 0)]
        if ('u', 0) in modelList[self.VOF_model].ebq:
            self.ebq_H_vof = modelList[self.VOF_model].ebq[('u', 0)]
        else:
            self.ebq_H_vof = None
        # correction
        self.massCorrModel = modelList[self.ME_model]
        self.vofModel.q[('m_last', 0)][:] = self.vofModel.q[('m', 0)]
        if self.checkMass:
            self.m_tmp = copy.deepcopy(self.massCorrModel.q[('r', 0)])
            if self.checkMass:
                # self.vofGlobalMass = Norms.scalarDomainIntegral(self.vofModel.q['dV'],
                #                                                 self.vofModel.q[('u',0)],
                #                                                 self.massCorrModel.mesh.nElements_owned)
                # self.lsGlobalMass = Norms.scalarHeavisideDomainIntegral(self.vofModel.q['dV'],
                #                                                         self.lsModel.q[('u',0)],
                #                                                         self.massCorrModel.mesh.nElements_owned)
                #self.vofGlobalMass = 0.0
                #self.lsGlobalMass = self.massCorrModel.calculateMass(self.lsModel.q[('u',0)])
                # log("Attach Models MCorr3P: mass correction %12.5e" % (Norms.scalarDomainIntegral(self.vofModel.q['dV'],
                #                                                                                self.massCorrModel.q[('r',0)],
                # self.massCorrModel.mesh.nElements_owned),),level=2)
                self.fluxGlobal = 0.0
                self.totalFluxGlobal = 0.0
                self.vofGlobalMassArray = []  # self.vofGlobalMass]
                self.lsGlobalMassArray = []  # self.lsGlobalMass]
                # self.vofGlobalMass - self.vofGlobalMassArray[0]]# +
                # self.vofModel.timeIntegration.dt*self.vofModel.coefficients.fluxIntegral]
                self.vofGlobalMassErrorArray = []
                # self.lsGlobalMass - self.lsGlobalMassArray[0]]# +
                # self.vofModel.timeIntegration.dt*self.vofModel.coefficients.fluxIntegral]
                self.lsGlobalMassErrorArray = []
                self.fluxArray = []  # 0.0]#self.vofModel.coefficients.fluxIntegral]
                self.timeArray = []  # self.vofModel.timeIntegration.t]
                #log("Attach Models MCorr3P: Phase 0 mass after mass correction (VOF) %12.5e" % (self.vofGlobalMass,),level=2)
                #log("Attach Models MCorr3P: Phase 0 mass after mass correction (LS) %12.5e" % (self.lsGlobalMass,),level=2)
                #log("Attach Models MCorr3P: Phase  0 mass conservation (VOF) after step = %12.5e" % (self.vofGlobalMass - self.vofModel.coefficients.m_pre + self.vofModel.timeIntegration.dt*self.vofModel.coefficients.fluxIntegral,),level=2)
                #log("Attach Models MCorr3P: Phase  0 mass conservation (LS) after step = %12.5e" % (self.lsGlobalMass - self.lsModel.coefficients.m_pre + self.vofModel.timeIntegration.dt*self.vofModel.coefficients.fluxIntegral,),level=2)

    def initializeElementQuadrature(self, t, cq):
        if self.sd and ('a', 0, 0) in cq:
            cq[('a', 0, 0)].fill(self.epsDiffusion)

    def initializeElementBoundaryQuadrature(self, t, cebq, cebq_global):
        if self.sd and ('a', 0, 0) in cebq:
            cebq[('a', 0, 0)].fill(self.epsDiffusion)

    def initializeGlobalExteriorElementBoundaryQuadrature(self, t, cebqe):
        if self.sd and ('a', 0, 0) in cebqe:
            cebqe[('a', 0, 0)].fill(self.epsDiffusion)

    def preStep(self, t, firstStep=False):
        if self.checkMass:
            log(
                "Phase 0 mass before mass correction (VOF) %12.5e" %
                (Norms.scalarDomainIntegral(
                    self.vofModel.q['dV'], self.vofModel.q[
                        ('m', 0)], self.massCorrModel.mesh.nElements_owned),), level=2)
            log(
                "Phase 0 mass (primitive) before mass correction (LS) %12.5e" %
                (Norms.scalarSmoothedHeavisideDomainIntegral(
                    self.epsFactHeaviside,
                    self.massCorrModel.elementDiameter,
                    self.vofModel.q['dV'],
                    self.lsModel.q[
                        ('m',
                         0)],
                    self.massCorrModel.mesh.nElements_owned),
                 ),
                level=2)
            log("Phase 0 mass (consistent) before mass correction (LS) %12.5e" % (
                self.massCorrModel.calculateMass(self.lsModel.q[('m', 0)]),), level=2)
        copyInstructions = {'clear_uList': True}
        return copyInstructions

    def postStep(self, t, firstStep=False):
        if self.applyCorrection:
            # ls
            self.lsModel.u[0].dof += self.massCorrModel.u[0].dof
            self.lsModel.q[('u', 0)] += self.massCorrModel.q[('u', 0)]
            self.lsModel.ebqe[('u', 0)] += self.massCorrModel.ebqe[('u', 0)]
            self.lsModel.q[
                ('grad(u)', 0)] += self.massCorrModel.q[('grad(u)', 0)]
            self.lsModel.ebqe[
                ('grad(u)', 0)] += self.massCorrModel.ebqe[('grad(u)', 0)]
            # vof
            self.massCorrModel.setMassQuadrature()
            self.vofModel.q[('m_tmp',0)][:] = (1.0 - self.vofModel.coefficients.q_vos)*self.vofModel.q[('u',0)]
            #self.vofModel.q[('u',0)] += self.massCorrModel.q[('r',0)]
            # print "********************max
            # VOF************************",max(self.vofModel.q[('u',0)].flat[:])
        if self.checkMass:
            log(
                "Phase 0 mass after mass correction (VOF) %12.5e" %
                (Norms.scalarDomainIntegral(
                    self.vofModel.q['dV'], self.vofModel.q[
                        ('m', 0)], self.massCorrModel.mesh.nElements_owned),), level=2)
            log(
                "Phase 0 mass (primitive) after mass correction (LS) %12.5e" %
                (Norms.scalarSmoothedHeavisideDomainIntegral(
                    self.epsFactHeaviside,
                    self.massCorrModel.elementDiameter,
                    self.vofModel.q['dV'],
                    self.lsModel.q[
                        ('m',
                         0)],
                    self.massCorrModel.mesh.nElements_owned),
                 ),
                level=2)
            log("Phase 0 mass (consistent) after mass correction (LS) %12.5e" % (
                self.massCorrModel.calculateMass(self.lsModel.q[('m', 0)]),), level=2)
        copyInstructions = {}
        # get the waterline on the obstacle if option set in NCLS (boundary==7)
        self.lsModel.computeWaterline(t)
        return copyInstructions

    def evaluate(self, t, c):
        import math
        if c[('u', 0)].shape == self.q_u_ls.shape:
            u_ls = self.q_u_ls
            H_vof = self.q_H_vof
        elif c[('u', 0)].shape == self.ebqe_u_ls.shape:
            u_ls = self.ebqe_u_ls
            H_vof = self.ebqe_H_vof
        elif self.ebq_u_ls is not None and c[('u', 0)].shape == self.ebq_u_ls.shape:
            u_ls = self.ebq_u_ls
            H_vof = self.ebq_H_vof
        else:
            #\todo trap errors in TransportCoefficients.py
            u_ls = None
            H_vof = None
        if u_ls is not None and H_vof is not None:
            if self.useC:
                if self.sd:
                    self.levelSetConservationCoefficientsEvaluate_sd(
                        self.epsHeaviside, self.epsDirac, u_ls, H_vof, c[
                            ('u', 0)], c[
                            ('r', 0)], c[
                            ('dr', 0, 0)])
                else:
                    self.levelSetConservationCoefficientsEvaluate(
                        self.epsHeaviside, self.epsDirac, self.epsDiffusion, u_ls, H_vof, c[
                            ('u', 0)], c[
                            ('r', 0)], c[
                            ('dr', 0, 0)], c[
                            ('a', 0, 0)])
        if (self.checkMass and c[('u', 0)].shape == self.q_u_ls.shape):
            self.m_tmp[:] = H_vof
            self.m_tmp += self.massCorrModel.q[('r', 0)]
            log("mass correction during Newton %12.5e" % (Norms.scalarDomainIntegral(self.vofModel.q[
                'dV'], self.massCorrModel.q[('r', 0)], self.massCorrModel.mesh.nElements_owned),), level=2)
            log("Phase 0 mass during Newton %12.5e" % (Norms.scalarDomainIntegral(
                self.vofModel.q['dV'], self.m_tmp, self.massCorrModel.mesh.nElements_owned),), level=2)


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
        self.bdyNullSpace = bdyNullSpace
        self.useConstantH = coefficients.useConstantH
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
        #self.q['x'] = numpy.zeros((self.mesh.nElements_global,self.nQuadraturePoints_element,3),'d')
        self.q[('u', 0)] = numpy.zeros(
            (self.mesh.nElements_global, self.nQuadraturePoints_element), 'd')
        self.q[
            ('grad(u)',
             0)] = numpy.zeros(
            (self.mesh.nElements_global,
             self.nQuadraturePoints_element,
             self.nSpace_global),
            'd')
        self.q[('r', 0)] = numpy.zeros(
            (self.mesh.nElements_global, self.nQuadraturePoints_element), 'd')

        self.ebqe[
            ('u',
             0)] = numpy.zeros(
            (self.mesh.nExteriorElementBoundaries_global,
             self.nElementBoundaryQuadraturePoints_elementBoundary),
            'd')
        self.ebqe[
            ('grad(u)',
             0)] = numpy.zeros(
            (self.mesh.nExteriorElementBoundaries_global,
             self.nElementBoundaryQuadraturePoints_elementBoundary,
             self.nSpace_global),
            'd')

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
            self.inflowBoundaryBC[cj] = numpy.zeros(
                (self.mesh.nExteriorElementBoundaries_global,), 'i')
            self.inflowBoundaryBC_values[cj] = numpy.zeros(
                (self.mesh.nExteriorElementBoundaries_global, self.nDOF_trial_element[cj]), 'd')
            self.inflowFlux[cj] = numpy.zeros(
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
        self.internalNodesArray = numpy.zeros((self.nNodes_internal,), 'i')
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

        # STIFFNESS MATRIX #
        self.interface_lumpedMassMatrix = numpy.zeros(self.u[0].dof.shape,'d')
        self.stiffness_matrix_array = None
        self.stiffness_matrix = None
        
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
        compKernelFlag = 0
        if self.coefficients.useConstantH:
            self.elementDiameter = self.mesh.elementDiametersArray.copy()
            self.elementDiameter[:] = max(self.mesh.elementDiametersArray)
        else:
            self.elementDiameter = self.mesh.elementDiametersArray
        self.mcorr3p = cMCorr3P.newMCorr3P(
            self.nSpace_global,
            self.nQuadraturePoints_element,
            self.u[0].femSpace.elementMaps.localFunctionSpace.dim,
            self .u[0].femSpace.referenceFiniteElement.localFunctionSpace.dim,
            self.testSpace[0].referenceFiniteElement.localFunctionSpace.dim,
            self.nElementBoundaryQuadraturePoints_elementBoundary,
            compKernelFlag)
    # mwf these are getting called by redistancing classes,

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
        if self.coefficients.set_vos:
            self.coefficients.set_vos(self.q['x'], self.coefficients.q_vos)
        r.fill(0.0)
        self.interface_lumpedMassMatrix.fill(0.0)
        # Load the unknowns into the finite element dof
        self.setUnknowns(u)

        # no flux boundary conditions
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
        argsDict["useMetrics"] = self.coefficients.useMetrics
        argsDict["epsFactHeaviside"] = self.coefficients.epsFactHeaviside
        argsDict["epsFactDirac"] = self.coefficients.epsFactDirac
        argsDict["epsFactDiffusion"] = self.coefficients.epsFactDiffusion
        argsDict["u_l2g"] = self.u[0].femSpace.dofMap.l2g
        argsDict["elementDiameter"] = self.elementDiameter
        argsDict["nodeDiametersArray"] = self.mesh.nodeDiametersArray
        argsDict["u_dof"] = self.u[0].dof
        argsDict["q_phi"] = self.coefficients.q_u_ls
        argsDict["q_normal_phi"] = self.coefficients.q_n_ls
        argsDict["ebqe_phi"] = self.coefficients.ebqe_u_ls
        argsDict["ebqe_normal_phi"] = self.coefficients.ebqe_n_ls
        argsDict["q_H"] = self.coefficients.q_H_vof
        argsDict["q_u"] = self.q[('u', 0)]
        argsDict["q_n"] = self.q[('grad(u)', 0)]
        argsDict["ebqe_u"] = self.ebqe[('u', 0)]
        argsDict["ebqe_n"] = self.ebqe[('grad(u)', 0)]
        argsDict["q_r"] = self.q[('r', 0)]
        argsDict["q_vos"] = self.coefficients.q_vos
        argsDict["offset_u"] = self.offset[0]
        argsDict["stride_u"] = self.stride[0]
        argsDict["globalResidual"] = r
        argsDict["nExteriorElementBoundaries_global"] = self.mesh.nExteriorElementBoundaries_global
        argsDict["exteriorElementBoundariesArray"] = self.mesh.exteriorElementBoundariesArray
        argsDict["elementBoundaryElementsArray"] = self.mesh.elementBoundaryElementsArray
        argsDict["elementBoundaryLocalElementBoundariesArray"] = self.mesh.elementBoundaryLocalElementBoundariesArray
        argsDict["interface_lumpedMassMatrix"] = self.interface_lumpedMassMatrix
        self.mcorr3p.calculateResidual(argsDict)

        log("Global residual", level=9, data=r)
        self.coefficients.massConservationError = fabs(globalSum(r[:self.mesh.nNodes_owned].sum()))
        log("   Mass Conservation Error", level=3,
            data=self.coefficients.massConservationError)
        self.nonlinear_function_evaluations += 1
        if self.globalResidualDummy is None:
            self.globalResidualDummy = numpy.zeros(r.shape, 'd')

    def getStiffnessMatrix(self):
        rowptr, colind, nzval = self.jacobian.getCSRrepresentation()
        nnz = nzval.shape[-1]  # number of non-zero entries in sparse matrix
        self.stiffness_matrix_array = nzval.copy()
        self.stiffness_matrix = SparseMat(self.nFreeDOF_global[0],
                                          self.nFreeDOF_global[0],
                                          nnz,
                                          self.stiffness_matrix_array,
                                          colind,
                                          rowptr)
        cfemIntegrals.zeroJacobian_CSR(self.nNonzerosInJacobian,
                                       self.stiffness_matrix)
        argsDict = cArgumentsDict.ArgumentsDict()
        argsDict["mesh_trial_ref"] = self.u[0].femSpace.elementMaps.psi
        argsDict["mesh_grad_trial_ref"] = self.u[0].femSpace.elementMaps.grad_psi
        argsDict["mesh_dof"] = self.mesh.nodeArray
        argsDict["mesh_l2g"] = self.mesh.elementNodesArray
        argsDict["dV_ref"] = self.elementQuadratureWeights[('u', 0)]
        argsDict["u_grad_trial_ref"] = self.u[0].femSpace.grad_psi
        argsDict["nElements_global"] = self.mesh.nElements_global
        argsDict["csrRowIndeces_u_u"] = self.csrRowIndeces[(0, 0)]
        argsDict["csrColumnOffsets_u_u"] = self.csrColumnOffsets[(0, 0)]
        argsDict["globalJacobian"] = self.stiffness_matrix.getCSRrepresentation()[2]
        argsDict["useMetrics"] = self.coefficients.useMetrics
        argsDict["epsFactDiffusion"] = self.coefficients.epsFactDiffusion
        argsDict["elementDiameter"] = self.elementDiameter
        argsDict["nodeDiametersArray"] = self.mesh.nodeDiametersArray
        self.mcorr3p.calculateStiffnessMatrix(argsDict)
        
    def getJacobian(self, jacobian):
        cfemIntegrals.zeroJacobian_CSR(self.nNonzerosInJacobian, jacobian)

        if self.stiffness_matrix is None:
            self.getStiffnessMatrix()
        
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
        argsDict["useMetrics"] = self.coefficients.useMetrics
        argsDict["epsFactHeaviside"] = self.coefficients.epsFactHeaviside
        argsDict["epsFactDirac"] = self.coefficients.epsFactDirac
        argsDict["epsFactDiffusion"] = self.coefficients.epsFactDiffusion
        argsDict["u_l2g"] = self.u[0].femSpace.dofMap.l2g
        argsDict["elementDiameter"] = self.elementDiameter
        argsDict["nodeDiametersArray"] = self.mesh.nodeDiametersArray
        argsDict["u_dof"] = self.u[0].dof
        argsDict["q_phi"] = self.coefficients.q_u_ls
        argsDict["q_normal_phi"] = self.coefficients.q_n_ls
        argsDict["q_H"] = self.coefficients.q_H_vof
        argsDict["q_vos"] = self.coefficients.q_vos
        argsDict["csrRowIndeces_u_u"] = self.csrRowIndeces[(0, 0)]
        argsDict["csrColumnOffsets_u_u"] = self.csrColumnOffsets[(0, 0)]
        argsDict["globalJacobian"] = jacobian.getCSRrepresentation()[2]
        argsDict["numDOFs"] = len(self.u[0].dof)
        argsDict["csrRowIndeces_DofLoops"] = self.rowptr
        argsDict["csrColumnOffsets_DofLoops"] = self.colind
        argsDict["stiffness_matrix"] = self.stiffness_matrix_array
        argsDict["interface_lumpedMassMatrix"] = self.interface_lumpedMassMatrix
        argsDict["fast_assembly"] = int(self.coefficients.fastAssembly)
        self.mcorr3p.calculateJacobian(argsDict)
        
        log("Jacobian ", level=10, data=jacobian)
        # mwf decide if this is reasonable for solver statistics
        self.nonlinear_function_jacobian_evaluations += 1
        return jacobian

    def elementSolve(self, u, r):
        import pdb
        import copy
        """
        Calculate the element residuals and add in to the global residual
        """
        r.fill(0.0)
        # Load the unknowns into the finite element dof
        self.setUnknowns(u)

        # no flux boundary conditions
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
        argsDict["useMetrics"] = self.coefficients.useMetrics
        argsDict["epsFactHeaviside"] = self.coefficients.epsFactHeaviside
        argsDict["epsFactDirac"] = self.coefficients.epsFactDirac
        argsDict["epsFactDiffusion"] = self.coefficients.epsFactDiffusion
        argsDict["u_l2g"] = self.u[0].femSpace.dofMap.l2g
        argsDict["elementDiameter"] = self.elementDiameter
        argsDict["nodeDiametersArray"] = self.mesh.nodeDiametersArray
        argsDict["u_dof"] = self.u[0].dof
        argsDict["q_phi"] = self.coefficients.q_u_ls
        argsDict["q_normal_phi"] = self.coefficients.q_n_ls
        argsDict["ebqe_phi"] = self.coefficients.ebqe_u_ls
        argsDict["ebqe_normal_phi"] = self.coefficients.ebqe_n_ls
        argsDict["q_H"] = self.coefficients.q_H_vof
        argsDict["q_u"] = self.q[('u', 0)]
        argsDict["q_n"] = self.q[('grad(u)', 0)]
        argsDict["ebqe_u"] = self.ebqe[('u', 0)]
        argsDict["ebqe_n"] = self.ebqe[('grad(u)', 0)]
        argsDict["q_r"] = self.q[('r', 0)]
        argsDict["q_vos"] = self.coefficients.q_vos
        argsDict["offset_u"] = self.offset[0]
        argsDict["stride_u"] = self.stride[0]
        argsDict["globalResidual"] = r
        argsDict["nExteriorElementBoundaries_global"] = self.mesh.nExteriorElementBoundaries_global
        argsDict["exteriorElementBoundariesArray"] = self.mesh.exteriorElementBoundariesArray
        argsDict["elementBoundaryElementsArray"] = self.mesh.elementBoundaryElementsArray
        argsDict["elementBoundaryLocalElementBoundariesArray"] = self.mesh.elementBoundaryLocalElementBoundariesArray
        argsDict["maxIts"] = self.maxIts
        argsDict["atol"] = self.atol
        self.mcorr3p.elementSolve(argsDict)

    def elementConstantSolve(self, u, r):
        import pdb
        import copy
        """
        Calculate the element residuals and add in to the global residual
        """
        r.fill(0.0)
        # Load the unknowns into the finite element dof
        self.setUnknowns(u)

        # no flux boundary conditions
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
        argsDict["useMetrics"] = self.coefficients.useMetrics
        argsDict["epsFactHeaviside"] = self.coefficients.epsFactHeaviside
        argsDict["epsFactDirac"] = self.coefficients.epsFactDirac
        argsDict["epsFactDiffusion"] = self.coefficients.epsFactDiffusion
        argsDict["u_l2g"] = self.u[0].femSpace.dofMap.l2g
        argsDict["elementDiameter"] = self.elementDiameter
        argsDict["nodeDiametersArray"] = self.mesh.nodeDiametersArray
        argsDict["u_dof"] = self.u[0].dof
        argsDict["q_phi"] = self.coefficients.q_u_ls
        argsDict["q_normal_phi"] = self.coefficients.q_n_ls
        argsDict["ebqe_phi"] = self.coefficients.ebqe_u_ls
        argsDict["ebqe_normal_phi"] = self.coefficients.ebqe_n_ls
        argsDict["q_H"] = self.coefficients.q_H_vof
        argsDict["q_u"] = self.q[('u', 0)]
        argsDict["q_n"] = self.q[('grad(u)', 0)]
        argsDict["ebqe_u"] = self.ebqe[('u', 0)]
        argsDict["ebqe_n"] = self.ebqe[('grad(u)', 0)]
        argsDict["q_r"] = self.q[('r', 0)]
        argsDict["q_vos"] = self.coefficients.q_vos
        argsDict["offset_u"] = self.offset[0]
        argsDict["stride_u"] = self.stride[0]
        argsDict["globalResidual"] = r
        argsDict["nExteriorElementBoundaries_global"] = self.mesh.nExteriorElementBoundaries_global
        argsDict["exteriorElementBoundariesArray"] = self.mesh.exteriorElementBoundariesArray
        argsDict["elementBoundaryElementsArray"] = self.mesh.elementBoundaryElementsArray
        argsDict["elementBoundaryLocalElementBoundariesArray"] = self.mesh.elementBoundaryLocalElementBoundariesArray
        argsDict["maxIts"] = self.maxIts
        argsDict["atol"] = self.atol
        self.mcorr3p.elementConstantSolve(argsDict)

    def globalConstantRJ(self, u, r, U):
        import pdb
        import copy
        """
        Calculate the element residuals and add in to the global residual
        """
        r.fill(0.0)
        # Load the unknowns into the finite element dof
        self.setUnknowns(u)

        # no flux boundary conditions
        R = np.array(dtype=float)
        J = np.array(dtype=float)
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
        argsDict["nElements_owned"] = self.mesh.nElements_owned
        argsDict["useMetrics"] = self.coefficients.useMetrics
        argsDict["epsFactHeaviside"] = self.coefficients.epsFactHeaviside
        argsDict["epsFactDirac"] = self.coefficients.epsFactDirac
        argsDict["epsFactDiffusion"] = self.coefficients.epsFactDiffusion
        argsDict["u_l2g"] = self.u[0].femSpace.dofMap.l2g
        argsDict["elementDiameter"] = self.elementDiameter
        argsDict["nodeDiametersArray"] = self.mesh.nodeDiametersArray
        argsDict["u_dof"] = self.u[0].dof
        argsDict["q_phi"] = self.coefficients.q_u_ls
        argsDict["q_normal_phi"] = self.coefficients.q_n_ls
        argsDict["ebqe_phi"] = self.coefficients.ebqe_u_ls
        argsDict["ebqe_normal_phi"] = self.coefficients.ebqe_n_ls
        argsDict["q_H"] = self.coefficients.q_H_vof
        argsDict["q_u"] = self.q[('u', 0)]
        argsDict["q_n"] = self.q[('grad(u)', 0)]
        argsDict["ebqe_u"] = self.ebqe[('u', 0)]
        argsDict["ebqe_n"] = self.ebqe[('grad(u)', 0)]
        argsDict["q_r"] = self.q[('r', 0)]
        argsDict["q_vos"] = self.offset[0]
        argsDict["offset_u"] = self.stride[0]
        argsDict["stride_u"] = r
        argsDict["globalResidual"] = self.coefficients.q_vos
        argsDict["nExteriorElementBoundaries_global"] = self.mesh.nExteriorElementBoundaries_global
        argsDict["exteriorElementBoundariesArray"] = self.mesh.exteriorElementBoundariesArray
        argsDict["elementBoundaryElementsArray"] = self.mesh.elementBoundaryElementsArray
        argsDict["elementBoundaryLocalElementBoundariesArray"] = self.mesh.elementBoundaryLocalElementBoundariesArray
        argsDict["maxIts"] = self.maxIts
        argsDict["atol"] = self.atol
        argsDict["constant_u"] = U
        (R, J) = self.mcorr3p.globalConstantRJ(argsDict)
        R = globalSum(R)
        J = globalSum(J)
        self.coefficients.massConservationError = fabs(R)
        return (R, J)

    def globalConstantSolve(self, u, r):
        U = 0.0
        R = 0.0
        J = 0.0
        (R, J) = self.globalConstantRJ(u, r, U)
        its = 0
        log("   Mass Conservation Residual 0 ", level=3, data=R)
        RNORM_OLD = fabs(R)
        while ((fabs(R) > self.atol and its < self.maxIts) or its < 1):
            U -= R/(J+1.0e-8)
            (R, J) = self.globalConstantRJ(u, r, U)
            lsits = 0
            while(fabs(R) > 0.99 * RNORM_OLD and lsits < self.maxLSits):
                lsits += 1
                U += (0.5)**lsits * (R/(J+1.0e-8))
                (R, J) = self.globalConstantRJ(u, r, U)
            its += 1
            log("   Mass Conservation Residual " + repr(its)+" ", level=3, data=R)
        self.u[0].dof.flat[:] = U

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
        self.u[0].femSpace.elementMaps.getBasisValuesTraceRef(
            self.elementBoundaryQuadraturePoints)
        self.u[0].femSpace.elementMaps.getBasisGradientValuesTraceRef(
            self.elementBoundaryQuadraturePoints)
        self.u[0].femSpace.getBasisValuesTraceRef(
            self.elementBoundaryQuadraturePoints)
        self.u[0].femSpace.getBasisGradientValuesTraceRef(
            self.elementBoundaryQuadraturePoints)

    def estimate_mt(self):
        pass

    def calculateAuxiliaryQuantitiesAfterStep(self):
        pass

    def calculateMass(self, q_phi):
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
        argsDict["nElements_owned"] = self.mesh.nElements_owned
        argsDict["useMetrics"] = self.coefficients.useMetrics
        argsDict["epsFactHeaviside"] = self.coefficients.epsFactHeaviside
        argsDict["epsFactDirac"] = self.coefficients.epsFactDirac
        argsDict["epsFactDiffusion"] = self.coefficients.epsFactDiffusion
        argsDict["u_l2g"] = self.u[0].femSpace.dofMap.l2g
        argsDict["elementDiameter"] = self.elementDiameter
        argsDict["nodeDiametersArray"] = self.mesh.nodeDiametersArray
        argsDict["u_dof"] = self.u[0].dof
        argsDict["q_phi"] = q_phi
        argsDict["q_normal_phi"] = self.coefficients.q_n_ls
        argsDict["ebqe_phi"] = self.coefficients.ebqe_u_ls
        argsDict["ebqe_normal_phi"] = self.coefficients.ebqe_n_ls
        argsDict["q_H"] = self.coefficients.q_H_vof
        argsDict["q_u"] = self.q[('u', 0)]
        argsDict["q_n"] = self.q[('grad(u)', 0)]
        argsDict["ebqe_u"] = self.ebqe[('u', 0)]
        argsDict["ebqe_n"] = self.ebqe[('grad(u)', 0)]
        argsDict["q_r"] = self.q[('r', 0)]
        argsDict["q_vos"] = self.coefficients.q_vos
        argsDict["offset_u"] = self.offset[0]
        argsDict["stride_u"] = self.stride[0]
        argsDict["globalResidual"] = self.u[0].dof
        argsDict["nExteriorElementBoundaries_global"] = self.mesh.nExteriorElementBoundaries_global
        argsDict["exteriorElementBoundariesArray"] = self.mesh.exteriorElementBoundariesArray
        argsDict["elementBoundaryElementsArray"] = self.mesh.elementBoundaryElementsArray
        argsDict["elementBoundaryLocalElementBoundariesArray"] = self.mesh.elementBoundaryLocalElementBoundariesArray
        return globalSum(self.mcorr3p.calculateMass(argsDict))

    def setMassQuadrature(self):
        argsDict = cArgumentsDict.ArgumentsDict()
        argsDict["mesh_trial_ip"] = self.u[0].femSpace.elementMaps.psi_ip
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
        argsDict["useMetrics"] = self.coefficients.useMetrics
        argsDict["epsFactHeaviside"] = self.coefficients.epsFactHeaviside
        argsDict["epsFactDirac"] = self.coefficients.epsFactDirac
        argsDict["epsFactDiffusion"] = self.coefficients.epsFactDiffusion
        argsDict["phi_l2g"] = self.coefficients.lsModel.u[0].femSpace.dofMap.l2g
        argsDict["elementDiameter"] = self.elementDiameter
        argsDict["nodeDiametersArray"] = self.mesh.nodeDiametersArray
        argsDict["phi_dof"] = self.coefficients.lsModel.u[0].dof
        argsDict["q_phi"] = self.coefficients.q_u_ls
        argsDict["q_normal_phi"] = self.coefficients.q_n_ls
        argsDict["ebqe_phi"] = self.coefficients.ebqe_u_ls
        argsDict["ebqe_normal_phi"] = self.coefficients.ebqe_n_ls
        argsDict["q_H"] = self.coefficients.q_H_vof
        argsDict["q_u"] = self.q[('u', 0)]
        argsDict["q_n"] = self.q[('grad(u)', 0)]
        argsDict["ebqe_u"] = self.ebqe[('u', 0)]
        argsDict["ebqe_n"] = self.ebqe[('grad(u)', 0)]
        argsDict["q_r"] = self.q[('r', 0)]
        argsDict["q_vos"] = self.coefficients.q_vos
        argsDict["offset_u"] = self.offset[0]
        argsDict["stride_u"] = self.stride[0]
        argsDict["globalResidual"] = self.u[0].dof
        argsDict["nExteriorElementBoundaries_global"] = self.mesh.nExteriorElementBoundaries_global
        argsDict["exteriorElementBoundariesArray"] = self.mesh.exteriorElementBoundariesArray
        argsDict["elementBoundaryElementsArray"] = self.mesh.elementBoundaryElementsArray
        argsDict["elementBoundaryLocalElementBoundariesArray"] = self.mesh.elementBoundaryLocalElementBoundariesArray
        argsDict["H_dof"] = self.coefficients.vofModel.u[0].dof
        self.mcorr3p.setMassQuadrature(argsDict)
        # self.coefficients.q_H_vof.flat[:]=777.0

    def calculateSolutionAtQuadrature(self):
        pass

    def updateAfterMeshMotion(self):
        pass


class DummyNewton(proteus.NonlinearSolvers.NonlinearSolver):

    def __init__(self,
                 linearSolver,
                 F, J=None, du=None, par_du=None,
                 rtol_r=1.0e-4,
                 atol_r=1.0e-16,
                 rtol_du=1.0e-4,
                 atol_du=1.0e-16,
                 maxIts=100,
                 norm=l2Norm,
                 convergenceTest='r',
                 computeRates=True,
                 printInfo=True,
                 fullNewton=True,
                 directSolver=False,
                 EWtol=True,
                 maxLSits=100):
        import copy
        self.par_du = par_du
        if par_du is not None:
            F.dim_proc = par_du.dim_proc
        NonlinearSolver.__init__(self, F, J, du,
                                 rtol_r,
                                 atol_r,
                                 rtol_du,
                                 atol_du,
                                 maxIts,
                                 norm,
                                 convergenceTest,
                                 computeRates,
                                 printInfo)
        self.updateJacobian = True
        self.fullNewton = fullNewton
        self.linearSolver = linearSolver
        self.directSolver = directSolver
        self.lineSearch = True
        # mwf turned back on self.lineSearch = False
        self.EWtol = EWtol
        # mwf added
        self.maxLSits = maxLSits
        if self.linearSolver.computeEigenvalues:
            self.JLast = copy.deepcopy(self.J)
            self.J_t_J = copy.deepcopy(self.J)
            self.dJ_t_dJ = copy.deepcopy(self.J)
            self.JLsolver = LU(self.J_t_J, computeEigenvalues=True)
            self.dJLsolver = LU(self.dJ_t_dJ, computeEigenvalues=True)
            self.u0 = numpy.zeros(self.F.dim, 'd')

    def info(self):
        return "Not Implemented"

    def solve(self, u, r=None, b=None, par_u=None, par_r=None):
        self.F.q[('r', 0)].flat[:] = 0.0
        self.F.q[('u', 0)].flat[:] = 0.0
        self.failedFlag = False
        return self.failedFlag


class ElementNewton(proteus.NonlinearSolvers.NonlinearSolver):

    def __init__(self,
                 linearSolver,
                 F, J=None, du=None, par_du=None,
                 rtol_r=1.0e-4,
                 atol_r=1.0e-16,
                 rtol_du=1.0e-4,
                 atol_du=1.0e-16,
                 maxIts=100,
                 norm=l2Norm,
                 convergenceTest='r',
                 computeRates=True,
                 printInfo=True,
                 fullNewton=True,
                 directSolver=False,
                 EWtol=True,
                 maxLSits=100):
        import copy
        self.par_du = par_du
        if par_du is not None:
            F.dim_proc = par_du.dim_proc
        NonlinearSolver.__init__(self, F, J, du,
                                 rtol_r,
                                 atol_r,
                                 rtol_du,
                                 atol_du,
                                 maxIts,
                                 norm,
                                 convergenceTest,
                                 computeRates,
                                 printInfo)
        self.updateJacobian = True
        self.fullNewton = fullNewton
        self.linearSolver = linearSolver
        self.directSolver = directSolver
        self.lineSearch = True
        # mwf turned back on self.lineSearch = False
        self.EWtol = EWtol
        # mwf added
        self.maxLSits = maxLSits
        if self.linearSolver.computeEigenvalues:
            self.JLast = copy.deepcopy(self.J)
            self.J_t_J = copy.deepcopy(self.J)
            self.dJ_t_dJ = copy.deepcopy(self.J)
            self.JLsolver = LU(self.J_t_J, computeEigenvalues=True)
            self.dJLsolver = LU(self.dJ_t_dJ, computeEigenvalues=True)
            self.u0 = numpy.zeros(self.F.dim, 'd')

    def info(self):
        return "Not Implemented"

    def solve(self, u, r=None, b=None, par_u=None, par_r=None):
        self.F.maxIts = self.maxIts
        self.F.maxLSits = self.maxLSits
        self.F.atol = self.atol_r
        self.F.elementSolve(u, r)
        self.failedFlag = False
        return self.failedFlag


class ElementConstantNewton(proteus.NonlinearSolvers.NonlinearSolver):

    def __init__(self,
                 linearSolver,
                 F, J=None, du=None, par_du=None,
                 rtol_r=1.0e-4,
                 atol_r=1.0e-16,
                 rtol_du=1.0e-4,
                 atol_du=1.0e-16,
                 maxIts=100,
                 norm=l2Norm,
                 convergenceTest='r',
                 computeRates=True,
                 printInfo=True,
                 fullNewton=True,
                 directSolver=False,
                 EWtol=True,
                 maxLSits=100):
        import copy
        self.par_du = par_du
        if par_du is not None:
            F.dim_proc = par_du.dim_proc
        NonlinearSolver.__init__(self, F, J, du,
                                 rtol_r,
                                 atol_r,
                                 rtol_du,
                                 atol_du,
                                 maxIts,
                                 norm,
                                 convergenceTest,
                                 computeRates,
                                 printInfo)
        self.updateJacobian = True
        self.fullNewton = fullNewton
        self.linearSolver = linearSolver
        self.directSolver = directSolver
        self.lineSearch = True
        # mwf turned back on self.lineSearch = False
        self.EWtol = EWtol
        # mwf added
        self.maxLSits = maxLSits
        if self.linearSolver.computeEigenvalues:
            self.JLast = copy.deepcopy(self.J)
            self.J_t_J = copy.deepcopy(self.J)
            self.dJ_t_dJ = copy.deepcopy(self.J)
            self.JLsolver = LU(self.J_t_J, computeEigenvalues=True)
            self.dJLsolver = LU(self.dJ_t_dJ, computeEigenvalues=True)
            self.u0 = numpy.zeros(self.F.dim, 'd')

    def info(self):
        return "Not Implemented"

    def solve(self, u, r=None, b=None, par_u=None, par_r=None):
        self.F.maxIts = self.maxIts
        self.F.maxLSits = self.maxLSits
        self.F.atol = self.atol_r
        self.F.elementConstantSolve(u, r)
        self.failedFlag = False
        return self.failedFlag


class GlobalConstantNewton(proteus.NonlinearSolvers.NonlinearSolver):

    def __init__(self,
                 linearSolver,
                 F, J=None, du=None, par_du=None,
                 rtol_r=1.0e-4,
                 atol_r=1.0e-16,
                 rtol_du=1.0e-4,
                 atol_du=1.0e-16,
                 maxIts=100,
                 norm=l2Norm,
                 convergenceTest='r',
                 computeRates=True,
                 printInfo=True,
                 fullNewton=True,
                 directSolver=False,
                 EWtol=True,
                 maxLSits=100):
        import copy
        self.par_du = par_du
        if par_du is not None:
            F.dim_proc = par_du.dim_proc
        NonlinearSolver.__init__(self, F, J, du,
                                 rtol_r,
                                 atol_r,
                                 rtol_du,
                                 atol_du,
                                 maxIts,
                                 norm,
                                 convergenceTest,
                                 computeRates,
                                 printInfo)
        self.updateJacobian = True
        self.fullNewton = fullNewton
        self.linearSolver = linearSolver
        self.directSolver = directSolver
        self.lineSearch = True
        # mwf turned back on self.lineSearch = False
        self.EWtol = EWtol
        # mwf added
        self.maxLSits = maxLSits
        if self.linearSolver.computeEigenvalues:
            self.JLast = copy.deepcopy(self.J)
            self.J_t_J = copy.deepcopy(self.J)
            self.dJ_t_dJ = copy.deepcopy(self.J)
            self.JLsolver = LU(self.J_t_J, computeEigenvalues=True)
            self.dJLsolver = LU(self.dJ_t_dJ, computeEigenvalues=True)
            self.u0 = numpy.zeros(self.F.dim, 'd')

    def info(self):
        return "Not Implemented"

    def solve(self, u, r=None, b=None, par_u=None, par_r=None):
        self.F.maxIts = self.maxIts
        self.F.maxLSits = self.maxLSits
        self.F.atol = self.atol_r
        self.F.globalConstantSolve(u, r)
        self.failedFlag = False
        return self.failedFlag


def conservationNorm(x):
    return fabs(globalSum(sum(x.flat)))


class Newton_controller(proteus.StepControl.Newton_controller):

    def __init__(self, model, nOptions):
        proteus.StepControl.Newton_controller.__init__(self, model, nOptions)

    def initializeTimeHistory(self):
        proteus.StepControl.Newton_controller.initializeTimeHistory(self)
        for m, u, r in zip(self.model.levelModelList,
                           self.model.uList,
                           self.model.rList):
            # pass
            # m.setMassQuadrature()
            u.flat[:] = 0.0
            m.getResidual(u, r)
            m.coefficients.postStep(self.t_model)
            m.coefficients.vofModel.updateTimeHistory(
                self.t_model, resetFromDOF=False)
            m.coefficients.vofModel.timeIntegration.updateTimeHistory(
                resetFromDOF=False)
