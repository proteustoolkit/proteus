import proteus
import proteus.mprans.RANS2P as RANS2P
from proteus.mprans.cRANS2P_IB import *
import numpy as np
from . import beamFEM
from .ArchiveBeams import *
from proteus.Comm import (globalMax,
                          globalSum)
import numpy
from proteus import *
from proteus.Transport import *
from proteus.Transport import OneLevelTransport
from . import cArgumentsDict

class Coefficients(proteus.mprans.RANS2P.Coefficients):
    def __init__(self,
                 epsFact=1.5,
                 sigma=72.8,
                 rho_0=998.2, nu_0=1.004e-6,
                 rho_1=1.205, nu_1=1.500e-5,
                 g=[0.0, 0.0, -9.8],
                 nd=3,
                 LS_model=None,
                 VF_model=None,
                 KN_model=None,
                 Closure_0_model=None,  # Turbulence closure model
                 Closure_1_model=None,  # Second possible Turbulence closure model
                 epsFact_density=None,
                 stokes=False,
                 sd=True,
                 movingDomain=False,
                 useVF=0.0,
                 useRBLES=0.0,
                 useMetrics=0.0,
                 useConstant_he=False,
                 dragAlpha=0.01,
                 dragBeta=0.0,
                 setParamsFunc=None,  # uses setParamsFunc if given
                 dragAlphaTypes=None,  # otherwise can use element constant values
                 dragBetaTypes=None,  # otherwise can use element constant values
                 porosityTypes=None,
                 killNonlinearDrag=False,
                 epsFact_source=1.,
                 epsFact_solid=None,
                 eb_adjoint_sigma=1.0,
                 eb_penalty_constant=10.0,
                 forceStrongDirichlet=False,
                 turbulenceClosureModel=0,  # 0=No Model, 1=Smagorinksy, 2=Dynamic Smagorinsky, 3=K-Epsilon, 4=K-Omega
                 smagorinskyConstant=0.1,
                 barycenters=None,
                 beamLocation=[],
                 beamLength=[],
                 beamRadius=[],
                 EI=[],
                 GJ=[],
                 nBeamElements=4,
                 beam_quadOrder=3,
                 beamFilename="Beams",
                 beam_useSparse=False,
                 beam_Cd=1.2,
                 beam_nlTol=1.0e-5,
                 beamRigid=True):

        # forward the arguments (this call passed the defaults, so every setting above was dropped)
        RANS2P.Coefficients.__init__(self,
                                     epsFact=epsFact,
                                     sigma=sigma,
                                     rho_0=rho_0,
                                     nu_0=nu_0,
                                     rho_1=rho_1,
                                     nu_1=nu_1,
                                     g=g,
                                     nd=nd,
                                     LS_model=LS_model,
                                     VF_model=VF_model,
                                     KN_model=KN_model,
                                     Closure_0_model=Closure_0_model,
                                     Closure_1_model=Closure_1_model,
                                     epsFact_density=epsFact_density,
                                     stokes=stokes,
                                     sd=sd,
                                     movingDomain=movingDomain,
                                     useVF=useVF,
                                     useRBLES=useRBLES,
                                     useMetrics=useMetrics,
                                     useConstant_he=useConstant_he,
                                     dragAlpha=dragAlpha,
                                     dragBeta=dragBeta,
                                     setParamsFunc=setParamsFunc,
                                     dragAlphaTypes=dragAlphaTypes,
                                     dragBetaTypes=dragBetaTypes,
                                     porosityTypes=porosityTypes,
                                     killNonlinearDrag=killNonlinearDrag,
                                     epsFact_solid=epsFact_solid,
                                     eb_adjoint_sigma=eb_adjoint_sigma,
                                     eb_penalty_constant=eb_penalty_constant,
                                     forceStrongDirichlet=forceStrongDirichlet,
                                     turbulenceClosureModel=turbulenceClosureModel,
                                     smagorinskyConstant=smagorinskyConstant,
                                     barycenters=barycenters)
        self.epsFact_source = epsFact_source
        self.beamFilename = beamFilename
        self.beamLength = beamLength
        self.beamRadius = np.array(beamRadius)
        self.beamLocation = beamLocation
        self.beamVolume = math.pi * self.beamRadius * self.beamRadius * self.beamLength
        self.nBeamElements = nBeamElements
        self.EI = EI
        self.GJ = GJ
        self.beam_quadOrder = beam_quadOrder
        self.beam_nlTol = beam_nlTol
        self.beam_useSparse = beam_useSparse
        self.beam_Cd = beam_Cd
        self.beam_nlTol = beam_nlTol
        self.beamRigid = beamRigid

    def attachModels(self, modelList):
        RANS2P.Coefficients.attachModels(self, modelList)
        self.initializeBeams()
        self.netBeamDrag = np.array([0.0])

    def initializeElementQuadrature(self, t, cq):
        RANS2P.Coefficients.initializeElementQuadrature(self, t, cq)
        self.q_dragBeam1 = numpy.zeros(cq[('u', 1)].shape, 'd')
        self.q_dragBeam2 = numpy.zeros(cq[('u', 1)].shape, 'd')
        self.q_dragBeam3 = numpy.zeros(cq[('u', 1)].shape, 'd')

    def initializeGlobalExteriorElementBoundaryQuadrature(self, t, cebqe):
        RANS2P.Coefficients.initializeGlobalExteriorElementBoundaryQuadrature(self, t, cebqe)
        self.ebqe_dragBeam1 = numpy.zeros(cebqe[('u', 1)].shape, 'd')
        self.ebqe_dragBeam2 = numpy.zeros(cebqe[('u', 1)].shape, 'd')
        self.ebqe_dragBeam3 = numpy.zeros(cebqe[('u', 1)].shape, 'd')

    def initializeMesh(self, mesh):
        RANS2P.Coefficients.initializeMesh(self, mesh)
        if self.comm.isMaster():
            self.netForceHistory = open("netForceHistory.txt", "w")
            self.forceHistory_drag = open("forceHistory_drag.txt", "w")
            self.velocityAverage = open("velocityAverage.txt", "w")
            self.dragCoefficientHistory = open("dragCoefficientHistory.txt", "w")
            if self.beamRigid == False:
                self.deflectionHistory = open("deflectionHistory.txt", "w")

    def postStep(self, t, firstStep=False):
        # Not RANS2P.Coefficients.postStep: this LevelModel is an older copy of RANS2P's, without the lagged dV and
        # active-element arrays that one updates (and the RANS2P_IB kernel does not read them).
        self.beamDrag = np.array([0.0, 0.0, 0.0])
        if not self.beamRigid:
            self.updateBeams(t)
        else:
            self.updateBeamLoad()
        if self.comm.isMaster():
            # the rows RANS2P writes (these went to stdout instead, and the history files RANS2P opened stayed empty),
            # and the beam drag
            if self.legacyHistoryFiles:
                self.deferWrite(self.timeHistory, "%21.16e\n" % (t,))
                for bN in range(self.wettedAreas.shape[0]):
                    self.deferWrite(self.wettedAreaHistory, "%21.16e\n" % (self.wettedAreas[bN],))
                    self.deferWrite(self.forceHistory_p, "%21.16e %21.16e %21.16e\n" % tuple(self.netForces_p[bN, :]))
                    self.deferWrite(self.forceHistory_v, "%21.16e %21.16e %21.16e\n" % tuple(self.netForces_v[bN, :]))
                    self.deferWrite(self.momentHistory, "%21.15e %21.16e %21.16e\n" % tuple(self.netMoments[bN, :]))
            self.deferWrite(self.forceHistory_drag, "%21.16e %21.16e %21.16e %21.16e\n" % ((t,) + tuple(self.beamDrag)))
            if self.beamRigid == False:
                self.deferWrite(self.deflectionHistory, "%21.16e %21.16e %21.16e %21.16e\n"
                                % (t, self.avgHeight, self.avgDeflection, self.avgAngle))

    def delta_h(self, r):
        if r <= -2.0:
            return 0.0
        elif r <= -1.0:
            return 1.0 / 8.0 * (5.0 + 2.0 * r - (-7.0 - 12.0 * r - 4 * r * r)**0.5)
        elif r <= 0.0:
            return 1.0 / 8.0 * (3.0 + 2.0 * r + (1.0 - 4.0 * r - 4.0 * r * r)**0.5)
        elif r <= 1.0:
            return 1.0 / 8.0 * (3.0 - 2.0 * r + (1.0 + 4.0 * r - 4.0 * r * r)**0.5)
        elif r <= 2.0:
            return 1.0 / 8.0 * (5.0 - 2.0 * r - (-7.0 + 12.0 * r - 4.0 * r * r)**0.5)
        else:
            return 0.0

    def updateBeamLoad(self):
        for I in range(self.nBeams):
            if I % self.comm.size() == self.comm.rank():
                if self.nd == 3:
                    self.Beam_Solver[I].updateLoads(self.q1[I, :], self.q2[I, :], self.q3[I, :])
                elif self.nd == 2:
                    self.Beam_Solver[I].updateLoads(self.q3[I, :], self.q2[I, :], self.q1[I, :])
                self.Beam_Solver[I].updateQs(endLoad=[0.0, 0.0, 0.0], scale=1.0)
                if self.centerBox[I] == True:
                    if self.nd == 3:
                        self.beamDrag[0] += self.Beam_Solver[I].Q1[0]
                        self.beamDrag[1] += self.Beam_Solver[I].Q2[0]
                        self.beamDrag[2] += self.Beam_Solver[I].Q3[0]
                    elif self.nd == 2:
                        self.beamDrag[0] += self.Beam_Solver[I].Q3[0]
                        self.beamDrag[1] += self.Beam_Solver[I].Q2[0]
                        self.beamDrag[2] += self.Beam_Solver[I].Q1[0]
        for i in range(3):
            self.beamDrag.flat[i] = globalSum(self.beamDrag.flat[i])

    def updateBeams(self, t):
        self.beamDrag = np.array([0.0, 0.0, 0.0])
        loadSteps = 20
        xv_hold = np.copy(self.xv)
        yv_hold = np.copy(self.yv)
        zv_hold = np.copy(self.zv)
        xq_hold = np.copy(self.xq)
        yq_hold = np.copy(self.yq)
        zq_hold = np.copy(self.zq)
        for I in range(self.nBeams):
            self.xv[I].flat = 0.0
            self.yv[I].flat = 0.0
            self.zv[I].flat = 0.0
            self.xq[I].flat = 0.0
            self.yq[I].flat = 0.0
            self.zq[I].flat = 0.0
            if I % self.comm.size() == self.comm.rank():
                if self.nd == 3:
                    self.Beam_Solver[I].updateLoads(self.q1[I, :], self.q2[I, :], self.q3[I, :])
                elif self.nd == 2:
                    self.Beam_Solver[I].updateLoads(self.q3[I, :], self.q2[I, :], self.q1[I, :])
                for j in range(loadSteps):
                    self.Beam_Solver[I].Phi.flat[:] = 0.0
                    self.Beam_Solver[I].updateQs(endLoad=[0.0, 0.0, 0.0], scale=float(j)/float(loadSteps))
                    counter = 0
                    go = True
                    Phi = np.copy(self.Beam_Solver[I].Phi)
                    while go == True:
                        self.Beam_Solver[I].calculateGradient_Hessian()
                        self.Beam_Solver[I].setBCs()
                        self.Beam_Solver[I].reduceOrder()
                        error = self.Beam_Solver[I].calculateResidual()
                        self.Beam_Solver[I].updateSolution()
                        go = self.Beam_Solver[I].checkConvergence()
                        counter += 1
                        if counter >= 100:
                            go = False
                            self.Beam_Solver[I].Phi = np.copy(Phi)
                if self.nd == 3:
                    xv, yv, zv = self.Beam_Solver[I].updateCoords()
                    if np.min(zv) > -1.0e-10:
                        self.xv[I, :] = xv
                        self.yv[I, :] = yv
                        self.zv[I, :] = zv
                        self.xq[I, :].flat[:], self.yq[I, :].flat[:], self.zq[I, :].flat[:] = self.Beam_Solver[I].getCoords_at_Quad()
                    else:
                        self.xv[I, :] = xv_hold[I, :]
                        self.yv[I, :] = yv_hold[I, :]
                        self.zv[I, :] = zv_hold[I, :]
                        self.xq[I, :] = xq_hold[I, :]
                        self.yq[I, :] = yq_hold[I, :]
                        self.zq[I, :] = zq_hold[I, :]
                    self.beamDrag[0] += self.Beam_Solver[I].Q1[0]
                    self.beamDrag[1] += self.Beam_Solver[I].Q2[0]
                    self.beamDrag[2] += self.Beam_Solver[I].Q3[0]
                elif self.nd == 2:
                    self.zv[I, :], self.yv[I, :], self.xv[I, :] = self.Beam_Solver[I].updateCoords()
                    self.zq[I, :].flat[:], self.yq[I, :].flat[:], self.xq[I, :].flat[:] = self.Beam_Solver[I].getCoords_at_Quad()
                    self.xq[I, :] += 0.2
                    self.xv[I, :] += 0.2
                    self.beamDrag[0] += self.Beam_Solver[I].Q3[0]
                    self.beamDrag[1] += self.Beam_Solver[I].Q2[0]
                    self.beamDrag[2] += self.Beam_Solver[I].Q1[0]
        self.beamDrag[0] = globalSum(self.beamDrag[0])
        self.beamDrag[1] = globalSum(self.beamDrag[1])
        self.beamDrag[2] = globalSum(self.beamDrag[2])

        for i in range(self.nBeams * (self.nBeamElements + 1)):
            self.xv.flat[i] = globalSum(self.xv.flat[i])
            self.yv.flat[i] = globalSum(self.yv.flat[i])
            self.zv.flat[i] = globalSum(self.zv.flat[i])
        for i in range(self.nBeams * self.nBeamElements * self.beam_quadOrder):
            self.xq.flat[i] = globalSum(self.xq.flat[i])
            self.yq.flat[i] = globalSum(self.yq.flat[i])
            self.zq.flat[i] = globalSum(self.zq.flat[i])

        if self.nBeams > 0 and self.comm.isMaster():
            Archive_time_step(Beam_x=self.xv,
                              Beam_y=self.yv,
                              Beam_z=self.zv,
                              nBeams=self.nBeams,
                              filename=self.beamFilename,
                              t=t,
                              tStep=self.tStep)
            self.tStep += 1
            # self.avgHeight=0.0
            # self.avgDeflection=0.0
            # self.avgAngle=0.0
            # nBeamsLocal=0
            # for I in range(self.nBeams):
            #    if self.centerBox[I]==True:
            #        nBeamsLocal+=1
            #        self.avgHeight+=self.zv[I,-1]
            #        self.avgDeflection += abs(self.xv[I,-1]-self.xv[I,0])
            #        self.avgAngle+= math.degrees(math.atan((self.zv[I,-1]-self.zv[I,-2])/(self.xv[I,-1]-self.xv[I,-2])))#self.Beam_Solver[I].Phi[-3]
            self.avgHeight = np.sum(self.zv[:,-1])/float(self.nBeams)
            self.avgDeflection = np.sum(np.abs(self.xv[:,-1]-self.xv[:,0]))/float(self.nBeams)
            self.avgAngle = np.sum(np.rad2deg(np.arctan((self.zv[:, -1] - self.zv[:, -2])/(self.xv[:, -1] - self.xv[:, -2]))))/float(self.nBeams)

    def initializeBeams(self):
        comm = Comm.get()
        self.comm = comm

        bBox = [.25 * .69, .69 - .25 * .69]
        self.centerBox = []

        self.nBeams = len(self.beamLocation)
        # eulerian coords for beam at mesh vertices
        self.xv = np.zeros((self.nBeams, self.nBeamElements + 1))
        self.yv = np.zeros((self.nBeams, self.nBeamElements + 1))
        self.zv = np.zeros((self.nBeams, self.nBeamElements + 1))

        # distributed loads for beam at quadrature points #mesh vertices
        self.q1 = np.zeros((self.nBeams, self.nBeamElements, self.beam_quadOrder))  # np.zeros((self.nBeams, self.nBeamElements+1))
        self.q2 = np.zeros((self.nBeams, self.nBeamElements, self.beam_quadOrder))  # np.zeros((self.nBeams, self.nBeamElements+1))
        self.q3 = np.zeros((self.nBeams, self.nBeamElements, self.beam_quadOrder))  # np.zeros((self.nBeams, self.nBeamElements+1))

        # element diameters
        self.Beam_h = np.zeros((self.nBeams, self.nBeamElements))

        # eulerian coords for beam at quadrataure points
        self.xq = np.zeros((self.nBeams, self.nBeamElements, self.beam_quadOrder))
        self.yq = np.zeros((self.nBeams, self.nBeamElements, self.beam_quadOrder))
        self.zq = np.zeros((self.nBeams, self.nBeamElements, self.beam_quadOrder))

        # integration constant for beams
        self.dV_beam = np.zeros((self.nBeams, self.nBeamElements, self.beam_quadOrder))

        self.Beam_mesh = np.zeros((self.nBeams, self.nBeamElements + 1))

        self.Beam_Solver = []
        boxCount = 0
        for i in range(self.nBeams):
            if self.beamLocation[i][1] >= bBox[0] and self.beamLocation[i][1] <= bBox[1]:
                self.centerBox.append(True)
                boxCount += 1
            else:
                self.centerBox.append(False)
            self.Beam_Solver.append(beamFEM.FEMTools(L=self.beamLength[i],
                                                     nElements=self.nBeamElements,
                                                     quadOrder=self.beam_quadOrder,
                                                     EI=self.EI[i],
                                                     GJ=self.GJ[i],
                                                     nlTol=self.beam_nlTol,
                                                     useSparse=self.beam_useSparse,
                                                     beamLocation=self.beamLocation[i]))
            self.Beam_mesh[i, :], self.Beam_h[i, :] = self.Beam_Solver[i].structuredMesh()
            self.Beam_Solver[i].GaussQuad()
            self.Beam_Solver[i].basisFunctions()
            self.Beam_Solver[i].initializePhi()
            self.Beam_Solver[i].initializeCoords()
            if self.nd == 3:
                self.xv[i, :], self.yv[i, :], self.zv[i, :] = self.Beam_Solver[i].updateCoords()
            elif self.nd == 2:
                self.zv[i, :], self.yv[i, :], self.xv[i, :] = self.Beam_Solver[i].updateCoords()
                self.xv[i, :] += 0.2
            #self.xv[i,:], self.yv[i,:], self.zv[i,:] = self.Beam_Solver[i].updateCoords()
            for j in range(self.nBeamElements):
                for k in range(self.beam_quadOrder):
                    self.dV_beam[i, j, k] = 0.5 * self.Beam_h[i, j] * self.Beam_Solver[i].w[k]

            if self.nd == 3:
                self.xq[i, :].flat[:], self.yq[i, :].flat[:], self.zq[i, :].flat[:] = self.Beam_Solver[i].getCoords_at_Quad()
            elif self.nd == 2:
                self.zq[i, :].flat[:], self.yq[i, :].flat[:], self.xq[i, :].flat[:] = self.Beam_Solver[i].getCoords_at_Quad()
                self.xq[i, :] += 0.2

            self.tStep = 0
            if self.nBeams > 0 and self.comm.isMaster():
                Archive_time_step(Beam_x=self.xv,
                                  Beam_y=self.yv,
                                  Beam_z=self.zv,
                                  nBeams=self.nBeams,
                                  filename=self.beamFilename,
                                  t=0.0,
                                  tStep=self.tStep)
            self.tStep += 1
        print(boxCount)

    pass


class LevelModel(proteus.mprans.RANS2P.LevelModel):
    # def __init__(self,
    #              uDict,
    #              phiDict,
    #              testSpaceDict,
    #              matType,
    #              dofBoundaryConditionsDict,
    #              dofBoundaryConditionsSetterDict,
    #              coefficients,
    #              elementQuadrature,
    #              elementBoundaryQuadrature,
    #              fluxBoundaryConditionsDict=None,
    #              advectiveFluxBoundaryConditionsSetterDict=None,
    #              diffusiveFluxBoundaryConditionsSetterDictDict=None,
    #              stressTraceBoundaryConditionsSetterDictDict=None,
    #              stabilization=None,
    #              shockCapturing=None,
    #              conservativeFluxDict=None,
    #              numericalFluxType=None,
    #              TimeIntegrationClass=None,
    #              massLumping=False,
    #              reactionLumping=False,
    #              options=None,
    #              name='RANS2P',
    #              reuse_trial_and_test_quadrature=True,
    #              sd = True,
    #              movingDomain=False):
    #      RANS2P.LevelModel.__init__(self,
    #                                       uDict,
    #                                       phiDict,
    #                                       testSpaceDict,
    #                                       matType,
    #                                       dofBoundaryConditionsDict,
    #                                       dofBoundaryConditionsSetterDict,
    #                                       coefficients,
    #                                       elementQuadrature,
    #                                       elementBoundaryQuadrature,
    #                                       fluxBoundaryConditionsDict=None,
    #                                       advectiveFluxBoundaryConditionsSetterDict=None,
    #                                       diffusiveFluxBoundaryConditionsSetterDictDict=None,
    #                                       stressTraceBoundaryConditionsSetterDictDict=None,
    #                                       stabilization=None,
    #                                       shockCapturing=None,
    #                                       conservativeFluxDict=None,
    #                                       numericalFluxType=None,
    #                                       TimeIntegrationClass=None,
    #                                       massLumping=False,
    #                                       reactionLumping=False,
    #                                       options=None,
    #                                       name='RANS2P',
    #                                       reuse_trial_and_test_quadrature=True,
    #                                       sd = True,
    #                                       movingDomain=False)
    #      self.rans2p_ib = cRANS2P_IB_base(self.nSpace_global,
    #                                    self.nQuadraturePoints_element,
    #                                    self.u[0].femSpace.elementMaps.localFunctionSpace.dim,
    #                                    self.u[0].femSpace.referenceFiniteElement.localFunctionSpace.dim,
    #                                    self.testSpace[0].referenceFiniteElement.localFunctionSpace.dim,
    #                                    self.nElementBoundaryQuadraturePoints_elementBoundary,
    #                                    compKernelFlag)

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
                 stressTraceBoundaryConditionsSetterDictDict=None,
                 stabilization=None,
                 shockCapturing=None,
                 conservativeFluxDict=None,
                 numericalFluxType=None,
                 TimeIntegrationClass=None,
                 massLumping=False,
                 reactionLumping=False,
                 options=None,
                 name='RANS2P',
                 reuse_trial_and_test_quadrature=True,
                 sd=True,
                 movingDomain=False):
        self.eb_adjoint_sigma = coefficients.eb_adjoint_sigma
        useConstant_he = coefficients.useConstant_he  # this is a hack to test the effect of using a constant smoothing width
        self.postProcessing = True
        #
        # set the objects describing the method and boundary conditions
        #
        self.movingDomain = coefficients.movingDomain
        self.tLast_mesh = None
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
        if isinstance(self.u[0].femSpace, C0_AffineQuadraticOnSimplexWithNodalBasis):
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
        self.ebqe['x'] = numpy.zeros((self.mesh.nExteriorElementBoundaries_global, self.nElementBoundaryQuadraturePoints_elementBoundary, 3), 'd')
        self.ebq_global[('totalFlux', 0)] = numpy.zeros((self.mesh.nElementBoundaries_global, self.nElementBoundaryQuadraturePoints_elementBoundary), 'd')
        self.ebq_global[('velocityAverage', 0)] = numpy.zeros((self.mesh.nElementBoundaries_global,
                                                               self.nElementBoundaryQuadraturePoints_elementBoundary, self.nSpace_global), 'd')
        self.q[('u', 1)] = numpy.zeros((self.mesh.nElements_global, self.nQuadraturePoints_element), 'd')
        self.q[('u', 2)] = numpy.zeros((self.mesh.nElements_global, self.nQuadraturePoints_element), 'd')
        self.q[('u', 3)] = numpy.zeros((self.mesh.nElements_global, self.nQuadraturePoints_element), 'd')
        self.q[('m', 1)] = self.q[('u', 1)]
        self.q[('m', 2)] = self.q[('u', 2)]
        self.q[('m', 3)] = self.q[('u', 3)]
        self.q['KE'] = numpy.zeros((self.mesh.nElements_global, self.nQuadraturePoints_element), 'd')
        self.q['PE'] = numpy.zeros((self.mesh.nElements_global, self.nQuadraturePoints_element), 'd')
        self.q['speed'] = numpy.zeros((self.mesh.nElements_global, self.nQuadraturePoints_element), 'd')
        self.q['rho'] = numpy.zeros((self.mesh.nElements_global, self.nQuadraturePoints_element), 'd')
        self.q[('m_last', 1)] = numpy.zeros((self.mesh.nElements_global, self.nQuadraturePoints_element), 'd')
        self.q[('m_last', 2)] = numpy.zeros((self.mesh.nElements_global, self.nQuadraturePoints_element), 'd')
        self.q[('m_last', 3)] = numpy.zeros((self.mesh.nElements_global, self.nQuadraturePoints_element), 'd')
        self.q[('m_tmp', 1)] = numpy.zeros((self.mesh.nElements_global, self.nQuadraturePoints_element), 'd')
        self.q[('m_tmp', 2)] = numpy.zeros((self.mesh.nElements_global, self.nQuadraturePoints_element), 'd')
        self.q[('m_tmp', 3)] = numpy.zeros((self.mesh.nElements_global, self.nQuadraturePoints_element), 'd')
        self.q[('mt', 1)] = numpy.zeros((self.mesh.nElements_global, self.nQuadraturePoints_element), 'd')
        self.q[('mt', 2)] = numpy.zeros((self.mesh.nElements_global, self.nQuadraturePoints_element), 'd')
        self.q[('mt', 3)] = numpy.zeros((self.mesh.nElements_global, self.nQuadraturePoints_element), 'd')
        self.q[('dV_u', 1)] = (1.0/self.mesh.nElements_global) * numpy.ones((self.mesh.nElements_global, self.nQuadraturePoints_element), 'd')
        self.q[('dV_u', 2)] = (1.0/self.mesh.nElements_global) * numpy.ones((self.mesh.nElements_global, self.nQuadraturePoints_element), 'd')
        self.q[('dV_u', 3)] = (1.0/self.mesh.nElements_global) * numpy.ones((self.mesh.nElements_global, self.nQuadraturePoints_element), 'd')
        self.q['dV'] = self.q[('dV_u', 1)]
        self.q[('f', 0)] = numpy.zeros((self.mesh.nElements_global, self.nQuadraturePoints_element, self.nSpace_global), 'd')
        self.q[('velocity', 0)] = numpy.zeros((self.mesh.nElements_global, self.nQuadraturePoints_element, self.nSpace_global), 'd')
        self.q['velocity_solid'] = numpy.zeros((self.mesh.nElements_global, self.nQuadraturePoints_element, self.nSpace_global), 'd')
        self.q['phi_solid'] = numpy.zeros((self.mesh.nElements_global, self.nQuadraturePoints_element), 'd')
        self.q['x'] = numpy.zeros((self.mesh.nElements_global, self.nQuadraturePoints_element, 3), 'd')
        self.q[('cfl', 0)] = numpy.zeros((self.mesh.nElements_global, self.nQuadraturePoints_element), 'd')
        self.q[('numDiff', 1, 1)] = numpy.zeros((self.mesh.nElements_global, self.nQuadraturePoints_element), 'd')
        self.q[('numDiff', 2, 2)] = numpy.zeros((self.mesh.nElements_global, self.nQuadraturePoints_element), 'd')
        self.q[('numDiff', 3, 3)] = numpy.zeros((self.mesh.nElements_global, self.nQuadraturePoints_element), 'd')
        self.ebqe[('u', 0)] = numpy.zeros((self.mesh.nExteriorElementBoundaries_global, self.nElementBoundaryQuadraturePoints_elementBoundary), 'd')
        self.ebqe[('u', 1)] = numpy.zeros((self.mesh.nExteriorElementBoundaries_global, self.nElementBoundaryQuadraturePoints_elementBoundary), 'd')
        self.ebqe[('u', 2)] = numpy.zeros((self.mesh.nExteriorElementBoundaries_global, self.nElementBoundaryQuadraturePoints_elementBoundary), 'd')
        self.ebqe[('u', 3)] = numpy.zeros((self.mesh.nExteriorElementBoundaries_global, self.nElementBoundaryQuadraturePoints_elementBoundary), 'd')
        self.ebqe[('advectiveFlux_bc_flag', 0)] = numpy.zeros(
            (self.mesh.nExteriorElementBoundaries_global, self.nElementBoundaryQuadraturePoints_elementBoundary), 'i')
        self.ebqe[('advectiveFlux_bc_flag', 1)] = numpy.zeros(
            (self.mesh.nExteriorElementBoundaries_global, self.nElementBoundaryQuadraturePoints_elementBoundary), 'i')
        self.ebqe[('advectiveFlux_bc_flag', 2)] = numpy.zeros(
            (self.mesh.nExteriorElementBoundaries_global, self.nElementBoundaryQuadraturePoints_elementBoundary), 'i')
        self.ebqe[('advectiveFlux_bc_flag', 3)] = numpy.zeros(
            (self.mesh.nExteriorElementBoundaries_global, self.nElementBoundaryQuadraturePoints_elementBoundary), 'i')
        self.ebqe[('diffusiveFlux_bc_flag', 1, 1)] = numpy.zeros(
            (self.mesh.nExteriorElementBoundaries_global, self.nElementBoundaryQuadraturePoints_elementBoundary), 'i')
        self.ebqe[('diffusiveFlux_bc_flag', 2, 2)] = numpy.zeros(
            (self.mesh.nExteriorElementBoundaries_global, self.nElementBoundaryQuadraturePoints_elementBoundary), 'i')
        self.ebqe[('diffusiveFlux_bc_flag', 3, 3)] = numpy.zeros(
            (self.mesh.nExteriorElementBoundaries_global, self.nElementBoundaryQuadraturePoints_elementBoundary), 'i')
        self.ebqe[('advectiveFlux_bc', 0)] = numpy.zeros((self.mesh.nExteriorElementBoundaries_global, self.nElementBoundaryQuadraturePoints_elementBoundary), 'd')
        self.ebqe[('advectiveFlux_bc', 1)] = numpy.zeros((self.mesh.nExteriorElementBoundaries_global, self.nElementBoundaryQuadraturePoints_elementBoundary), 'd')
        self.ebqe[('advectiveFlux_bc', 2)] = numpy.zeros((self.mesh.nExteriorElementBoundaries_global, self.nElementBoundaryQuadraturePoints_elementBoundary), 'd')
        self.ebqe[('advectiveFlux_bc', 3)] = numpy.zeros((self.mesh.nExteriorElementBoundaries_global, self.nElementBoundaryQuadraturePoints_elementBoundary), 'd')
        self.ebqe[('diffusiveFlux_bc', 1, 1)] = numpy.zeros(
            (self.mesh.nExteriorElementBoundaries_global, self.nElementBoundaryQuadraturePoints_elementBoundary), 'd')
        self.ebqe['penalty'] = numpy.zeros((self.mesh.nExteriorElementBoundaries_global, self.nElementBoundaryQuadraturePoints_elementBoundary), 'd')
        self.ebqe[('diffusiveFlux_bc', 2, 2)] = numpy.zeros(
            (self.mesh.nExteriorElementBoundaries_global, self.nElementBoundaryQuadraturePoints_elementBoundary), 'd')
        self.ebqe[('diffusiveFlux_bc', 3, 3)] = numpy.zeros(
            (self.mesh.nExteriorElementBoundaries_global, self.nElementBoundaryQuadraturePoints_elementBoundary), 'd')
        self.ebqe[('velocity', 0)] = numpy.zeros((self.mesh.nExteriorElementBoundaries_global,
                                                  self.nElementBoundaryQuadraturePoints_elementBoundary, self.nSpace_global), 'd')
        self.ebqe[('velocity', 1)] = numpy.zeros((self.mesh.nExteriorElementBoundaries_global,
                                                  self.nElementBoundaryQuadraturePoints_elementBoundary, self.nSpace_global), 'd')
        self.ebqe[('velocity', 2)] = numpy.zeros((self.mesh.nExteriorElementBoundaries_global,
                                                  self.nElementBoundaryQuadraturePoints_elementBoundary, self.nSpace_global), 'd')
        self.ebqe[('velocity', 3)] = numpy.zeros((self.mesh.nExteriorElementBoundaries_global,
                                                  self.nElementBoundaryQuadraturePoints_elementBoundary, self.nSpace_global), 'd')
        # VRANS start, defaults to RANS
        self.q[('r', 0)] = numpy.zeros((self.mesh.nElements_global, self.nQuadraturePoints_element), 'd')
        self.q['eddy_viscosity'] = numpy.zeros((self.mesh.nElements_global, self.nQuadraturePoints_element), 'd')
        # VRANS end
        # RANS 2eq Models start
        self.q[('grad(u)', 1)] = numpy.zeros((self.mesh.nElements_global, self.nQuadraturePoints_element, self.nSpace_global), 'd')
        self.q[('grad(u)', 2)] = numpy.zeros((self.mesh.nElements_global, self.nQuadraturePoints_element, self.nSpace_global), 'd')
        self.q[('grad(u)', 3)] = numpy.zeros((self.mesh.nElements_global, self.nQuadraturePoints_element, self.nSpace_global), 'd')
        # probably don't need ebqe gradients
        self.ebqe[('grad(u)', 1)] = numpy.zeros((self.mesh.nExteriorElementBoundaries_global,
                                                 self.nElementBoundaryQuadraturePoints_elementBoundary, self.nSpace_global), 'd')
        self.ebqe[('grad(u)', 2)] = numpy.zeros((self.mesh.nExteriorElementBoundaries_global,
                                                 self.nElementBoundaryQuadraturePoints_elementBoundary, self.nSpace_global), 'd')
        self.ebqe[('grad(u)', 3)] = numpy.zeros((self.mesh.nExteriorElementBoundaries_global,
                                                 self.nElementBoundaryQuadraturePoints_elementBoundary, self.nSpace_global), 'd')
        # RANS 2eq Models end
        self.points_elementBoundaryQuadrature = set()
        self.scalars_elementBoundaryQuadrature = set([('u', ci) for ci in range(self.nc)])
        self.vectors_elementBoundaryQuadrature = set()
        self.tensors_elementBoundaryQuadrature = set()
        # use post processing tools to get conservative fluxes, None by default
        if self.postProcessing:
            self.q[('v', 0)] = numpy.zeros(
                (self.mesh.nElements_global,
                 self.nQuadraturePoints_element,
                 self.nDOF_trial_element[0]),
                'd')
            self.q['J'] = numpy.zeros(
                (self.mesh.nElements_global,
                 self.nQuadraturePoints_element,
                 self.nSpace_global,
                 self.nSpace_global),
                'd')
            self.q['det(J)'] = numpy.zeros(
                (self.mesh.nElements_global,
                 self.nQuadraturePoints_element),
                'd')
            self.q['inverse(J)'] = numpy.zeros(
                (self.mesh.nElements_global,
                 self.nQuadraturePoints_element,
                 self.nSpace_global,
                 self.nSpace_global),
                'd')
            self.ebq[('v', 0)] = numpy.zeros(
                (self.mesh.nElements_global,
                 self.mesh.nElementBoundaries_element,
                 self.nElementBoundaryQuadraturePoints_elementBoundary,
                 self.nDOF_trial_element[0]),
                'd')
            self.ebq[('w', 0)] = numpy.zeros(
                (self.mesh.nElements_global,
                 self.mesh.nElementBoundaries_element,
                 self.nElementBoundaryQuadraturePoints_elementBoundary,
                 self.nDOF_trial_element[0]),
                'd')
            self.ebq['x'] = numpy.zeros(
                (self.mesh.nElements_global,
                 self.mesh.nElementBoundaries_element,
                 self.nElementBoundaryQuadraturePoints_elementBoundary,
                 3),
                'd')
            self.ebq['hat(x)'] = numpy.zeros(
                (self.mesh.nElements_global,
                 self.mesh.nElementBoundaries_element,
                 self.nElementBoundaryQuadraturePoints_elementBoundary,
                 3),
                'd')
            self.ebq['inverse(J)'] = numpy.zeros(
                (self.mesh.nElements_global,
                 self.mesh.nElementBoundaries_element,
                 self.nElementBoundaryQuadraturePoints_elementBoundary,
                 self.nSpace_global,
                 self.nSpace_global),
                'd')
            self.ebq['g'] = numpy.zeros(
                (self.mesh.nElements_global,
                 self.mesh.nElementBoundaries_element,
                 self.nElementBoundaryQuadraturePoints_elementBoundary,
                 self.nSpace_global - 1,
                 self.nSpace_global - 1),
                'd')
            self.ebq['sqrt(det(g))'] = numpy.zeros(
                (self.mesh.nElements_global,
                 self.mesh.nElementBoundaries_element,
                 self.nElementBoundaryQuadraturePoints_elementBoundary),
                'd')
            self.ebq['n'] = numpy.zeros(
                (self.mesh.nElements_global,
                 self.mesh.nElementBoundaries_element,
                 self.nElementBoundaryQuadraturePoints_elementBoundary,
                 self.nSpace_global),
                'd')
            self.ebq[('dS_u', 0)] = numpy.zeros(
                (self.mesh.nElements_global,
                 self.mesh.nElementBoundaries_element,
                 self.nElementBoundaryQuadraturePoints_elementBoundary),
                'd')
            self.ebqe['dS'] = numpy.zeros(
                (self.mesh.nExteriorElementBoundaries_global,
                 self.nElementBoundaryQuadraturePoints_elementBoundary),
                'd')
            self.ebqe[('dS_u', 0)] = self.ebqe['dS']
            self.ebqe['n'] = numpy.zeros(
                (self.mesh.nExteriorElementBoundaries_global,
                 self.nElementBoundaryQuadraturePoints_elementBoundary,
                 self.nSpace_global),
                'd')
            self.ebqe['inverse(J)'] = numpy.zeros(
                (self.mesh.nExteriorElementBoundaries_global,
                 self.nElementBoundaryQuadraturePoints_elementBoundary,
                 self.nSpace_global,
                 self.nSpace_global),
                'd')
            self.ebqe['g'] = numpy.zeros(
                (self.mesh.nExteriorElementBoundaries_global,
                 self.nElementBoundaryQuadraturePoints_elementBoundary,
                 self.nSpace_global - 1,
                 self.nSpace_global - 1),
                'd')
            self.ebqe['sqrt(det(g))'] = numpy.zeros(
                (self.mesh.nExteriorElementBoundaries_global,
                 self.nElementBoundaryQuadraturePoints_elementBoundary),
                'd')
            self.ebq_global['n'] = numpy.zeros(
                (self.mesh.nElementBoundaries_global,
                 self.nElementBoundaryQuadraturePoints_elementBoundary,
                 self.nSpace_global),
                'd')
            self.ebq_global['x'] = numpy.zeros(
                (self.mesh.nElementBoundaries_global,
                 self.nElementBoundaryQuadraturePoints_elementBoundary,
                 3),
                'd')
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
        #
        # allocate residual and Jacobian storage
        #
        self.elementResidual = [numpy.zeros(
            (self.mesh.nElements_global,
             self.nDOF_test_element[ci]),
            'd')]
        self.inflowBoundaryBC = {}
        self.inflowBoundaryBC_values = {}
        self.inflowFlux = {}
        for cj in range(self.nc):
            self.inflowBoundaryBC[cj] = numpy.zeros((self.mesh.nExteriorElementBoundaries_global,), 'i')
            self.inflowBoundaryBC_values[cj] = numpy.zeros((self.mesh.nExteriorElementBoundaries_global, self.nDOF_trial_element[cj]), 'd')
            self.inflowFlux[cj] = numpy.zeros((self.mesh.nExteriorElementBoundaries_global, self.nElementBoundaryQuadraturePoints_elementBoundary), 'd')
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
        self.internalNodesArray = numpy.zeros((self.nNodes_internal,), 'i')
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

        logEvent("initalizing numerical flux")
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
        logEvent("initializing numerical flux penalty")
        self.numericalFlux.penalty_constant = self.coefficients.eb_penalty_constant
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
        logEvent("setting up post-processing")
        from proteus import PostProcessingTools
        self.velocityPostProcessor = PostProcessingTools.VelocityPostProcessingChooser(self)
        logEvent(memory("velocity postprocessor", "OneLevelTransport"), level=4)
        # helper for writing out data storage
        logEvent("initializing archiver")
        from proteus import Archiver
        self.elementQuadratureDictionaryWriter = Archiver.XdmfWriter()
        self.elementBoundaryQuadratureDictionaryWriter = Archiver.XdmfWriter()
        self.exteriorElementBoundaryQuadratureDictionaryWriter = Archiver.XdmfWriter()
        logEvent("flux bc objects")
        for ci, fbcObject in list(self.fluxBoundaryConditionsObjectsDict.items()):
            self.ebqe[('advectiveFlux_bc_flag', ci)] = numpy.zeros(self.ebqe[('advectiveFlux_bc', ci)].shape, 'i')
            for t, g in list(fbcObject.advectiveFluxBoundaryConditionsDict.items()):
                if ci in self.coefficients.advection:
                    self.ebqe[('advectiveFlux_bc', ci)][t[0], t[1]] = g(self.ebqe[('x')][t[0], t[1]], self.timeIntegration.t)
                    self.ebqe[('advectiveFlux_bc_flag', ci)][t[0], t[1]] = 1
            for ck, diffusiveFluxBoundaryConditionsDict in list(fbcObject.diffusiveFluxBoundaryConditionsDictDict.items()):
                self.ebqe[('diffusiveFlux_bc_flag', ck, ci)] = numpy.zeros(self.ebqe[('diffusiveFlux_bc', ck, ci)].shape, 'i')
                for t, g in list(diffusiveFluxBoundaryConditionsDict.items()):
                    self.ebqe[('diffusiveFlux_bc', ck, ci)][t[0], t[1]] = g(self.ebqe[('x')][t[0], t[1]], self.timeIntegration.t)
                    self.ebqe[('diffusiveFlux_bc_flag', ck, ci)][t[0], t[1]] = 1
        self.numericalFlux.setDirichletValues(self.ebqe)
        if self.movingDomain:
            self.MOVING_DOMAIN = 1.0
        else:
            self.MOVING_DOMAIN = 0.0
        if self.mesh.nodeVelocityArray is None:
            self.mesh.nodeVelocityArray = numpy.zeros(self.mesh.nodeArray.shape, 'd')
        # cek/ido todo replace python loops in modules with optimized code if possible/necessary
        logEvent("dirichlet conditions")
        self.forceStrongConditions = coefficients.forceStrongDirichlet
        self.dirichletConditionsForceDOF = {}
        if self.forceStrongConditions:
            for cj in range(self.nc):
                self.dirichletConditionsForceDOF[cj] = DOFBoundaryConditions(
                    self.u[cj].femSpace, dofBoundaryConditionsSetterDict[cj], weakDirichletConditions=False)
        logEvent("final allocations")
        compKernelFlag = 0
        if self.coefficients.useConstant_he:
            self.elementDiameter = self.mesh.elementDiametersArray.copy()
            self.elementDiameter[:] = max(self.mesh.elementDiametersArray)
        else:
            self.elementDiameter = self.mesh.elementDiametersArray
        if self.nSpace_global == 2:
            import copy
            self.u[3] = self.u[2].copy()
            self.u[3].name = 'w'
            self.timeIntegration.m_tmp[3] = self.timeIntegration.m_tmp[2].copy()
            self.timeIntegration.beta_bdf[3] = self.timeIntegration.beta_bdf[2].copy()
            self.coefficients.sdInfo[(1, 3)] = (numpy.array([0, 1, 2], dtype='i'),
                                                numpy.array([0, 1], dtype='i'))
            self.coefficients.sdInfo[(2, 3)] = (numpy.array([0, 1, 2], dtype='i'),
                                                numpy.array([0, 1], dtype='i'))
            self.coefficients.sdInfo[(3, 0)] = (numpy.array([0, 1, 2], dtype='i'),
                                                numpy.array([0, 1], dtype='i'))
            self.coefficients.sdInfo[(3, 1)] = (numpy.array([0, 1, 2], dtype='i'),
                                                numpy.array([0, 1], dtype='i'))
            self.coefficients.sdInfo[(3, 2)] = (numpy.array([0, 1, 2], dtype='i'),
                                                numpy.array([0, 1], dtype='i'))
            self.coefficients.sdInfo[(3, 3)] = (numpy.array([0, 1, 2], dtype='i'),
                                                numpy.array([0, 1], dtype='i'))
            self.offset.append(self.offset[2])
            self.stride.append(self.stride[2])
            self.numericalFlux.isDOFBoundary[3] = self.numericalFlux.isDOFBoundary[2].copy()
            self.numericalFlux.ebqe[('u', 3)] = self.numericalFlux.ebqe[('u', 2)].copy()
            logEvent("calling cRANS2P2D_base ctor")
            self.rans2p = cRANS2P2D_base(self.nSpace_global,
                                         self.nQuadraturePoints_element,
                                         self.u[0].femSpace.elementMaps.localFunctionSpace.dim,
                                         self.u[0].femSpace.referenceFiniteElement.localFunctionSpace.dim,
                                         self.testSpace[0].referenceFiniteElement.localFunctionSpace.dim,
                                         self.nElementBoundaryQuadraturePoints_elementBoundary,
                                         compKernelFlag)
        else:
            logEvent("calling  cRANS2P_base ctor")
            self.rans2p = cRANS2P_IB_base(self.nSpace_global,
                                          self.nQuadraturePoints_element,
                                          self.u[0].femSpace.elementMaps.localFunctionSpace.dim,
                                          self.u[0].femSpace.referenceFiniteElement.localFunctionSpace.dim,
                                          self.testSpace[0].referenceFiniteElement.localFunctionSpace.dim,
                                          self.nElementBoundaryQuadraturePoints_elementBoundary,
                                          compKernelFlag)

    def getResidual(self, u, r):
        """
        Calculate the element residuals and add in to the global residual
        """

        # Load the unknowns into the finite element dof
        self.timeIntegration.calculateCoefs()
        self.timeIntegration.calculateU(u)
        self.setUnknowns(self.timeIntegration.u)
        # cek todo put in logic to skip if BC's don't depend on t or u
        # hack
        if self.bcsTimeDependent or not self.bcsSet:
            self.bcsSet = True
            # Dirichlet boundary conditions
            self.numericalFlux.setDirichletValues(self.ebqe)
            # Flux boundary conditions
            for ci, fbcObject in list(self.fluxBoundaryConditionsObjectsDict.items()):
                for t, g in list(fbcObject.advectiveFluxBoundaryConditionsDict.items()):
                    if ci in self.coefficients.advection:
                        self.ebqe[('advectiveFlux_bc', ci)][t[0], t[1]] = g(self.ebqe[('x')][t[0], t[1]], self.timeIntegration.t)
                        self.ebqe[('advectiveFlux_bc_flag', ci)][t[0], t[1]] = 1
                for ck, diffusiveFluxBoundaryConditionsDict in list(fbcObject.diffusiveFluxBoundaryConditionsDictDict.items()):
                    for t, g in list(diffusiveFluxBoundaryConditionsDict.items()):
                        self.ebqe[('diffusiveFlux_bc', ck, ci)][t[0], t[1]] = g(self.ebqe[('x')][t[0], t[1]], self.timeIntegration.t)
                        self.ebqe[('diffusiveFlux_bc_flag', ck, ci)][t[0], t[1]] = 1
        r.fill(0.0)
        self.Ct_sge = 4.0
        self.Cd_sge = 36.0
        # TODO how to request problem specific evaluations from coefficient class
        if 'evaluateForcingTerms' in dir(self.coefficients):
            self.coefficients.evaluateForcingTerms(self.timeIntegration.t, self.q, self.mesh,
                                                   self.u[0].femSpace.elementMaps.psi, self.mesh.elementNodesArray)
        self.coefficients.wettedAreas[:] = 0.0
        self.coefficients.netForces_p[:, :] = 0.0
        self.coefficients.netForces_v[:, :] = 0.0
        self.coefficients.netMoments[:, :] = 0.0

        if self.forceStrongConditions:
            for cj in range(len(self.dirichletConditionsForceDOF)):
                for dofN, g in list(self.dirichletConditionsForceDOF[cj].DOFBoundaryConditionsDict.items()):
                    self.u[cj].dof[dofN] = g(self.dirichletConditionsForceDOF[cj].DOFBoundaryPointDict[dofN], self.timeIntegration.t)
        self.r = r
        # self.beamStep()
        # self.coefficients.beamDrag=np.array([0.0,0.0,0.0])

        # self.coefficients.updateBeamLoad()
        # print self.coefficients.beamDrag
        # import pdb
        # pdb.set_trace()

        argsDict = cArgumentsDict.ArgumentsDict()
        argsDict["mesh_trial_ref"] = self.u[0].femSpace.elementMaps.psi
        argsDict["mesh_grad_trial_ref"] = self.u[0].femSpace.elementMaps.grad_psi
        argsDict["mesh_dof"] = self.mesh.nodeArray
        argsDict["mesh_velocity_dof"] = self.mesh.nodeVelocityArray
        argsDict["MOVING_DOMAIN"] = self.MOVING_DOMAIN
        argsDict["mesh_l2g"] = self.mesh.elementNodesArray
        argsDict["dV_ref"] = self.elementQuadratureWeights[('u', 0)]
        argsDict["p_trial_ref"] = self.u[0].femSpace.psi
        argsDict["p_grad_trial_ref"] = self.u[0].femSpace.grad_psi
        argsDict["p_test_ref"] = self.u[0].femSpace.psi
        argsDict["p_grad_test_ref"] = self.u[0].femSpace.grad_psi
        argsDict["vel_trial_ref"] = self.u[1].femSpace.psi
        argsDict["vel_grad_trial_ref"] = self.u[1].femSpace.grad_psi
        argsDict["vel_test_ref"] = self.u[1].femSpace.psi
        argsDict["vel_grad_test_ref"] = self.u[1].femSpace.grad_psi
        argsDict["mesh_trial_trace_ref"] = self.u[0].femSpace.elementMaps.psi_trace
        argsDict["mesh_grad_trial_trace_ref"] = self.u[0].femSpace.elementMaps.grad_psi_trace
        argsDict["dS_ref"] = self.elementBoundaryQuadratureWeights[('u', 0)]
        argsDict["p_trial_trace_ref"] = self.u[0].femSpace.psi_trace
        argsDict["p_grad_trial_trace_ref"] = self.u[0].femSpace.grad_psi_trace
        argsDict["p_test_trace_ref"] = self.u[0].femSpace.psi_trace
        argsDict["p_grad_test_trace_ref"] = self.u[0].femSpace.grad_psi_trace
        argsDict["vel_trial_trace_ref"] = self.u[1].femSpace.psi_trace
        argsDict["vel_grad_trial_trace_ref"] = self.u[1].femSpace.grad_psi_trace
        argsDict["vel_test_trace_ref"] = self.u[1].femSpace.psi_trace
        argsDict["vel_grad_test_trace_ref"] = self.u[1].femSpace.grad_psi_trace
        argsDict["normal_ref"] = self.u[0].femSpace.elementMaps.boundaryNormals
        argsDict["boundaryJac_ref"] = self.u[0].femSpace.elementMaps.boundaryJacobians
        argsDict["eb_adjoint_sigma"] = self.eb_adjoint_sigma
        argsDict["elementDiameter"] = self.elementDiameter
        argsDict["nodeDiametersArray"] = self.mesh.nodeDiametersArray
        argsDict["hFactor"] = self.stabilization.hFactor
        argsDict["nElements_global"] = self.mesh.nElements_global
        argsDict["nElementBoundaries_owned"] = self.mesh.nElementBoundaries_owned
        argsDict["useRBLES"] = self.coefficients.useRBLES
        argsDict["useMetrics"] = self.coefficients.useMetrics
        argsDict["alphaBDF"] = self.timeIntegration.alpha_bdf
        argsDict["epsFact_rho"] = self.coefficients.epsFact_density
        argsDict["epsFact_mu"] = self.coefficients.epsFact
        argsDict["sigma"] = self.coefficients.sigma
        argsDict["rho_0"] = self.coefficients.rho_0
        argsDict["nu_0"] = self.coefficients.nu_0
        argsDict["rho_1"] = self.coefficients.rho_1
        argsDict["nu_1"] = self.coefficients.nu_1
        argsDict["smagorinskyConstant"] = self.coefficients.smagorinskyConstant
        argsDict["turbulenceClosureModel"] = self.coefficients.turbulenceClosureModel
        argsDict["Ct_sge"] = self.Ct_sge
        argsDict["Cd_sge"] = self.Cd_sge
        argsDict["C_dc"] = self.shockCapturing.shockCapturingFactor
        argsDict["C_b"] = self.numericalFlux.penalty_constant
        argsDict["eps_solid"] = self.coefficients.epsFact_solid
        argsDict["phi_solid"] = self.coefficients.q_phi_solid
        argsDict["q_velocity_solid"] = self.coefficients.q_velocity_solid
        argsDict["q_porosity"] = self.coefficients.q_porosity
        argsDict["q_dragAlpha"] = self.coefficients.q_dragAlpha
        argsDict["q_dragBeta"] = self.coefficients.q_dragBeta
        argsDict["q_mass_source"] = self.q[('r', 0)]
        argsDict["q_turb_var_0"] = self.coefficients.q_turb_var[0]
        argsDict["q_turb_var_1"] = self.coefficients.q_turb_var[1]
        argsDict["q_turb_var_grad_0"] = self.coefficients.q_turb_var_grad[0]
        argsDict["p_l2g"] = self.u[0].femSpace.dofMap.l2g
        argsDict["vel_l2g"] = self.u[1].femSpace.dofMap.l2g
        argsDict["p_dof"] = self.u[0].dof
        argsDict["u_dof"] = self.u[1].dof
        argsDict["v_dof"] = self.u[2].dof
        argsDict["w_dof"] = self.u[3].dof
        argsDict["g"] = self.coefficients.g
        argsDict["useVF"] = self.coefficients.useVF
        argsDict["vf"] = self.coefficients.q_vf
        argsDict["phi"] = self.coefficients.q_phi
        argsDict["normal_phi"] = self.coefficients.q_n
        argsDict["kappa_phi"] = self.coefficients.q_kappa
        argsDict["q_mom_u_acc"] = self.timeIntegration.m_tmp[1]
        argsDict["q_mom_v_acc"] = self.timeIntegration.m_tmp[2]
        argsDict["q_mom_w_acc"] = self.timeIntegration.m_tmp[3]
        argsDict["q_mass_adv"] = self.q[('f', 0)]
        argsDict["q_mom_u_acc_beta_bdf"] = self.timeIntegration.beta_bdf[1]
        argsDict["q_mom_v_acc_beta_bdf"] = self.timeIntegration.beta_bdf[2]
        argsDict["q_mom_w_acc_beta_bdf"] = self.timeIntegration.beta_bdf[3]
        argsDict["q_velocity_sge"] = self.stabilization.v_last
        argsDict["q_cfl"] = self.q[('cfl', 0)]
        argsDict["q_numDiff_u"] = self.q[('numDiff', 1, 1)]
        argsDict["q_numDiff_v"] = self.q[('numDiff', 2, 2)]
        argsDict["q_numDiff_w"] = self.q[('numDiff', 3, 3)]
        argsDict["q_numDiff_u_last"] = self.shockCapturing.numDiff_last[1]
        argsDict["q_numDiff_v_last"] = self.shockCapturing.numDiff_last[2]
        argsDict["q_numDiff_w_last"] = self.shockCapturing.numDiff_last[3]
        argsDict["sdInfo_u_u_rowptr"] = self.coefficients.sdInfo[(1, 1)][0]
        argsDict["sdInfo_u_u_colind"] = self.coefficients.sdInfo[(1, 1)][1]
        argsDict["sdInfo_u_v_rowptr"] = self.coefficients.sdInfo[(1, 2)][0]
        argsDict["sdInfo_u_v_colind"] = self.coefficients.sdInfo[(1, 2)][1]
        argsDict["sdInfo_u_w_rowptr"] = self.coefficients.sdInfo[(1, 3)][0]
        argsDict["sdInfo_u_w_colind"] = self.coefficients.sdInfo[(1, 3)][1]
        argsDict["sdInfo_v_v_rowptr"] = self.coefficients.sdInfo[(2, 2)][0]
        argsDict["sdInfo_v_v_colind"] = self.coefficients.sdInfo[(2, 2)][1]
        argsDict["sdInfo_v_u_rowptr"] = self.coefficients.sdInfo[(2, 1)][0]
        argsDict["sdInfo_v_u_colind"] = self.coefficients.sdInfo[(2, 1)][1]
        argsDict["sdInfo_v_w_rowptr"] = self.coefficients.sdInfo[(2, 3)][0]
        argsDict["sdInfo_v_w_colind"] = self.coefficients.sdInfo[(2, 3)][1]
        argsDict["sdInfo_w_w_rowptr"] = self.coefficients.sdInfo[(3, 3)][0]
        argsDict["sdInfo_w_w_colind"] = self.coefficients.sdInfo[(3, 3)][1]
        argsDict["sdInfo_w_u_rowptr"] = self.coefficients.sdInfo[(3, 1)][0]
        argsDict["sdInfo_w_u_colind"] = self.coefficients.sdInfo[(3, 1)][1]
        argsDict["sdInfo_w_v_rowptr"] = self.coefficients.sdInfo[(3, 2)][0]
        argsDict["sdInfo_w_v_colind"] = self.coefficients.sdInfo[(3, 2)][1]
        argsDict["offset_p"] = self.offset[0]
        argsDict["offset_u"] = self.offset[1]
        argsDict["offset_v"] = self.offset[2]
        argsDict["offset_w"] = self.offset[3]
        argsDict["stride_p"] = self.stride[0]
        argsDict["stride_u"] = self.stride[1]
        argsDict["stride_v"] = self.stride[2]
        argsDict["stride_w"] = self.stride[3]
        argsDict["globalResidual"] = r
        argsDict["nExteriorElementBoundaries_global"] = self.mesh.nExteriorElementBoundaries_global
        argsDict["exteriorElementBoundariesArray"] = self.mesh.exteriorElementBoundariesArray
        argsDict["elementBoundaryElementsArray"] = self.mesh.elementBoundaryElementsArray
        argsDict["elementBoundaryLocalElementBoundariesArray"] = self.mesh.elementBoundaryLocalElementBoundariesArray
        argsDict["ebqe_vf_ext"] = self.coefficients.ebqe_vf
        argsDict["bc_ebqe_vf_ext"] = self.coefficients.bc_ebqe_vf
        argsDict["ebqe_phi_ext"] = self.coefficients.ebqe_phi
        argsDict["bc_ebqe_phi_ext"] = self.coefficients.bc_ebqe_phi
        argsDict["ebqe_normal_phi_ext"] = self.coefficients.ebqe_n
        argsDict["ebqe_kappa_phi_ext"] = self.coefficients.ebqe_kappa
        argsDict["ebqe_porosity_ext"] = self.coefficients.ebqe_porosity
        argsDict["ebqe_turb_var_0"] = self.coefficients.ebqe_turb_var[0]
        argsDict["ebqe_turb_var_1"] = self.coefficients.ebqe_turb_var[1]
        argsDict["isDOFBoundary_p"] = self.numericalFlux.isDOFBoundary[0]
        argsDict["isDOFBoundary_u"] = self.numericalFlux.isDOFBoundary[1]
        argsDict["isDOFBoundary_v"] = self.numericalFlux.isDOFBoundary[2]
        argsDict["isDOFBoundary_w"] = self.numericalFlux.isDOFBoundary[3]
        argsDict["isAdvectiveFluxBoundary_p"] = self.ebqe[('advectiveFlux_bc_flag', 0)]
        argsDict["isAdvectiveFluxBoundary_u"] = self.ebqe[('advectiveFlux_bc_flag', 1)]
        argsDict["isAdvectiveFluxBoundary_v"] = self.ebqe[('advectiveFlux_bc_flag', 2)]
        argsDict["isAdvectiveFluxBoundary_w"] = self.ebqe[('advectiveFlux_bc_flag', 3)]
        argsDict["isDiffusiveFluxBoundary_u"] = self.ebqe[('diffusiveFlux_bc_flag', 1, 1)]
        argsDict["isDiffusiveFluxBoundary_v"] = self.ebqe[('diffusiveFlux_bc_flag', 2, 2)]
        argsDict["isDiffusiveFluxBoundary_w"] = self.ebqe[('diffusiveFlux_bc_flag', 3, 3)]
        argsDict["ebqe_bc_p_ext"] = self.numericalFlux.ebqe[('u', 0)]
        argsDict["ebqe_bc_flux_mass_ext"] = self.ebqe[('advectiveFlux_bc', 0)]
        argsDict["ebqe_bc_flux_mom_u_adv_ext"] = self.ebqe[('advectiveFlux_bc', 1)]
        argsDict["ebqe_bc_flux_mom_v_adv_ext"] = self.ebqe[('advectiveFlux_bc', 2)]
        argsDict["ebqe_bc_flux_mom_w_adv_ext"] = self.ebqe[('advectiveFlux_bc', 3)]
        argsDict["ebqe_bc_u_ext"] = self.numericalFlux.ebqe[('u', 1)]
        argsDict["ebqe_bc_flux_u_diff_ext"] = self.ebqe[('diffusiveFlux_bc', 1, 1)]
        argsDict["ebqe_penalty_ext"] = self.ebqe['penalty']
        argsDict["ebqe_bc_v_ext"] = self.numericalFlux.ebqe[('u', 2)]
        argsDict["ebqe_bc_flux_v_diff_ext"] = self.ebqe[('diffusiveFlux_bc', 2, 2)]
        argsDict["ebqe_bc_w_ext"] = self.numericalFlux.ebqe[('u', 3)]
        argsDict["ebqe_bc_flux_w_diff_ext"] = self.ebqe[('diffusiveFlux_bc', 3, 3)]
        argsDict["q_x"] = self.q['x']
        argsDict["q_velocity"] = self.q[('velocity', 0)]
        argsDict["ebqe_velocity"] = self.ebqe[('velocity', 0)]
        argsDict["flux"] = self.ebq_global[('totalFlux', 0)]
        argsDict["elementResidual_p_save"] = self.elementResidual[0]
        argsDict["boundaryFlags"] = self.mesh.elementBoundaryUserFlags
        argsDict["barycenters"] = self.coefficients.barycenters
        argsDict["wettedAreas"] = self.coefficients.wettedAreas
        argsDict["netForces_p"] = self.coefficients.netForces_p
        argsDict["netForces_v"] = self.coefficients.netForces_v
        argsDict["netMoments"] = self.coefficients.netMoments
        argsDict["q_dragBeam1"] = self.coefficients.q_dragBeam1
        argsDict["q_dragBeam2"] = self.coefficients.q_dragBeam2
        argsDict["q_dragBeam3"] = self.coefficients.q_dragBeam3
        argsDict["ebqe_dragBeam1"] = self.coefficients.ebqe_dragBeam1
        argsDict["ebqe_dragBeam2"] = self.coefficients.ebqe_dragBeam2
        argsDict["ebqe_dragBeam3"] = self.coefficients.ebqe_dragBeam3
        self.rans2p.calculateResidual(argsDict)
        for i in range(self.coefficients.netForces_p.shape[0]):
            self.coefficients.wettedAreas[i] = globalSum(self.coefficients.wettedAreas[i])
            for I in range(3):
                self.coefficients.netForces_p[i, I] = globalSum(self.coefficients.netForces_p[i, I])
                self.coefficients.netForces_v[i, I] = globalSum(self.coefficients.netForces_v[i, I])
                self.coefficients.netMoments[i, I] = globalSum(self.coefficients.netMoments[i, I])
        if self.forceStrongConditions:
            for cj in range(len(self.dirichletConditionsForceDOF)):
                for dofN, g in list(self.dirichletConditionsForceDOF[cj].DOFBoundaryConditionsDict.items()):
                    r[self.offset[cj] + self.stride[cj] * dofN] = 0
        cflMax = globalMax(self.q[('cfl', 0)].max()) * self.timeIntegration.dt
        logEvent("Maximum CFL = " + str(cflMax), level=2)
        if self.stabilization:
            self.stabilization.accumulateSubgridMassHistory(self.q)
        logEvent("Global residual", level=9, data=r)
        # mwf decide if this is reasonable for keeping solver statistics
        self.nonlinear_function_evaluations += 1

    def getJacobian(self, jacobian):
        cfemIntegrals.zeroJacobian_CSR(self.nNonzerosInJacobian,
                                       jacobian)
        if self.nSpace_global == 2:
            self.csrRowIndeces[(0, 3)] = self.csrRowIndeces[(0, 2)]
            self.csrColumnOffsets[(0, 3)] = self.csrColumnOffsets[(0, 2)]
            self.csrRowIndeces[(1, 3)] = self.csrRowIndeces[(0, 2)]
            self.csrColumnOffsets[(1, 3)] = self.csrColumnOffsets[(0, 2)]
            self.csrRowIndeces[(2, 3)] = self.csrRowIndeces[(0, 2)]
            self.csrColumnOffsets[(2, 3)] = self.csrColumnOffsets[(0, 2)]
            self.csrRowIndeces[(3, 0)] = self.csrRowIndeces[(2, 0)]
            self.csrColumnOffsets[(3, 0)] = self.csrColumnOffsets[(2, 0)]
            self.csrRowIndeces[(3, 1)] = self.csrRowIndeces[(2, 0)]
            self.csrColumnOffsets[(3, 1)] = self.csrColumnOffsets[(2, 0)]
            self.csrRowIndeces[(3, 2)] = self.csrRowIndeces[(2, 0)]
            self.csrColumnOffsets[(3, 2)] = self.csrColumnOffsets[(2, 0)]
            self.csrRowIndeces[(3, 3)] = self.csrRowIndeces[(2, 0)]
            self.csrColumnOffsets[(3, 3)] = self.csrColumnOffsets[(2, 0)]
            self.csrColumnOffsets_eb[(0, 3)] = self.csrColumnOffsets[(0, 2)]
            self.csrColumnOffsets_eb[(1, 3)] = self.csrColumnOffsets[(0, 2)]
            self.csrColumnOffsets_eb[(2, 3)] = self.csrColumnOffsets[(0, 2)]
            self.csrColumnOffsets_eb[(3, 0)] = self.csrColumnOffsets[(0, 2)]
            self.csrColumnOffsets_eb[(3, 1)] = self.csrColumnOffsets[(0, 2)]
            self.csrColumnOffsets_eb[(3, 2)] = self.csrColumnOffsets[(0, 2)]
            self.csrColumnOffsets_eb[(3, 3)] = self.csrColumnOffsets[(0, 2)]

        argsDict = cArgumentsDict.ArgumentsDict()
        argsDict["mesh_trial_ref"] = self.u[0].femSpace.elementMaps.psi
        argsDict["mesh_grad_trial_ref"] = self.u[0].femSpace.elementMaps.grad_psi
        argsDict["mesh_dof"] = self.mesh.nodeArray
        argsDict["mesh_velocity_dof"] = self.mesh.nodeVelocityArray
        argsDict["MOVING_DOMAIN"] = self.MOVING_DOMAIN
        argsDict["mesh_l2g"] = self.mesh.elementNodesArray
        argsDict["dV_ref"] = self.elementQuadratureWeights[('u', 0)]
        argsDict["p_trial_ref"] = self.u[0].femSpace.psi
        argsDict["p_grad_trial_ref"] = self.u[0].femSpace.grad_psi
        argsDict["p_test_ref"] = self.u[0].femSpace.psi
        argsDict["p_grad_test_ref"] = self.u[0].femSpace.grad_psi
        argsDict["vel_trial_ref"] = self.u[1].femSpace.psi
        argsDict["vel_grad_trial_ref"] = self.u[1].femSpace.grad_psi
        argsDict["vel_test_ref"] = self.u[1].femSpace.psi
        argsDict["vel_grad_test_ref"] = self.u[1].femSpace.grad_psi
        argsDict["mesh_trial_trace_ref"] = self.u[0].femSpace.elementMaps.psi_trace
        argsDict["mesh_grad_trial_trace_ref"] = self.u[0].femSpace.elementMaps.grad_psi_trace
        argsDict["dS_ref"] = self.elementBoundaryQuadratureWeights[('u', 0)]
        argsDict["p_trial_trace_ref"] = self.u[0].femSpace.psi_trace
        argsDict["p_grad_trial_trace_ref"] = self.u[0].femSpace.grad_psi_trace
        argsDict["p_test_trace_ref"] = self.u[0].femSpace.psi_trace
        argsDict["p_grad_test_trace_ref"] = self.u[0].femSpace.grad_psi_trace
        argsDict["vel_trial_trace_ref"] = self.u[1].femSpace.psi_trace
        argsDict["vel_grad_trial_trace_ref"] = self.u[1].femSpace.grad_psi_trace
        argsDict["vel_test_trace_ref"] = self.u[1].femSpace.psi_trace
        argsDict["vel_grad_test_trace_ref"] = self.u[1].femSpace.grad_psi_trace
        argsDict["normal_ref"] = self.u[0].femSpace.elementMaps.boundaryNormals
        argsDict["boundaryJac_ref"] = self.u[0].femSpace.elementMaps.boundaryJacobians
        argsDict["eb_adjoint_sigma"] = self.eb_adjoint_sigma
        argsDict["elementDiameter"] = self.elementDiameter  # mesh.elementDiametersArray
        argsDict["nodeDiametersArray"] = self.mesh.nodeDiametersArray
        argsDict["hFactor"] = self.stabilization.hFactor
        argsDict["nElements_global"] = self.mesh.nElements_global
        argsDict["useRBLES"] = self.coefficients.useRBLES
        argsDict["useMetrics"] = self.coefficients.useMetrics
        argsDict["alphaBDF"] = self.timeIntegration.alpha_bdf
        argsDict["epsFact_rho"] = self.coefficients.epsFact_density
        argsDict["epsFact_mu"] = self.coefficients.epsFact
        argsDict["sigma"] = self.coefficients.sigma
        argsDict["rho_0"] = self.coefficients.rho_0
        argsDict["nu_0"] = self.coefficients.nu_0
        argsDict["rho_1"] = self.coefficients.rho_1
        argsDict["nu_1"] = self.coefficients.nu_1
        argsDict["smagorinskyConstant"] = self.coefficients.smagorinskyConstant
        argsDict["turbulenceClosureModel"] = self.coefficients.turbulenceClosureModel
        argsDict["Ct_sge"] = self.Ct_sge
        argsDict["Cd_sge"] = self.Cd_sge
        argsDict["C_dg"] = self.shockCapturing.shockCapturingFactor
        argsDict["C_b"] = self.numericalFlux.penalty_constant
        argsDict["eps_solid"] = self.coefficients.epsFact_solid
        argsDict["phi_solid"] = self.coefficients.q_phi_solid
        argsDict["q_velocity_solid"] = self.coefficients.q_velocity_solid
        argsDict["q_porosity"] = self.coefficients.q_porosity
        argsDict["q_dragAlpha"] = self.coefficients.q_dragAlpha
        argsDict["q_dragBeta"] = self.coefficients.q_dragBeta
        argsDict["q_mass_source"] = self.q[('r', 0)]
        argsDict["q_turb_var_0"] = self.coefficients.q_turb_var[0]
        argsDict["q_turb_var_1"] = self.coefficients.q_turb_var[1]
        argsDict["q_turb_var_grad_0"] = self.coefficients.q_turb_var_grad[0]
        argsDict["p_l2g"] = self.u[0].femSpace.dofMap.l2g
        argsDict["vel_l2g"] = self.u[1].femSpace.dofMap.l2g
        argsDict["p_dof"] = self.u[0].dof
        argsDict["u_dof"] = self.u[1].dof
        argsDict["v_dof"] = self.u[2].dof
        argsDict["w_dof"] = self.u[3].dof
        argsDict["g"] = self.coefficients.g
        argsDict["useVF"] = self.coefficients.useVF
        argsDict["vf"] = self.coefficients.q_vf
        argsDict["phi"] = self.coefficients.q_phi
        argsDict["normal_phi"] = self.coefficients.q_n
        argsDict["kappa_phi"] = self.coefficients.q_kappa
        argsDict["q_mom_u_acc_beta_bdf"] = self.timeIntegration.beta_bdf[1]
        argsDict["q_mom_v_acc_beta_bdf"] = self.timeIntegration.beta_bdf[2]
        argsDict["q_mom_w_acc_beta_bdf"] = self.timeIntegration.beta_bdf[3]
        argsDict["q_velocity_sge"] = self.stabilization.v_last
        argsDict["q_cfl"] = self.q[('cfl', 0)]
        argsDict["q_numDiff_u_last"] = self.shockCapturing.numDiff_last[1]
        argsDict["q_numDiff_v_last"] = self.shockCapturing.numDiff_last[2]
        argsDict["q_numDiff_w_last"] = self.shockCapturing.numDiff_last[3]
        argsDict["sdInfo_u_u_rowptr"] = self.coefficients.sdInfo[(1, 1)][0]
        argsDict["sdInfo_u_u_colind"] = self.coefficients.sdInfo[(1, 1)][1]
        argsDict["sdInfo_u_v_rowptr"] = self.coefficients.sdInfo[(1, 2)][0]
        argsDict["sdInfo_u_v_colind"] = self.coefficients.sdInfo[(1, 2)][1]
        argsDict["sdInfo_u_w_rowptr"] = self.coefficients.sdInfo[(1, 3)][0]
        argsDict["sdInfo_u_w_colind"] = self.coefficients.sdInfo[(1, 3)][1]
        argsDict["sdInfo_v_v_rowptr"] = self.coefficients.sdInfo[(2, 2)][0]
        argsDict["sdInfo_v_v_colind"] = self.coefficients.sdInfo[(2, 2)][1]
        argsDict["sdInfo_v_u_rowptr"] = self.coefficients.sdInfo[(2, 1)][0]
        argsDict["sdInfo_v_u_colind"] = self.coefficients.sdInfo[(2, 1)][1]
        argsDict["sdInfo_v_w_rowptr"] = self.coefficients.sdInfo[(2, 3)][0]
        argsDict["sdInfo_v_w_colind"] = self.coefficients.sdInfo[(2, 3)][1]
        argsDict["sdInfo_w_w_rowptr"] = self.coefficients.sdInfo[(3, 3)][0]
        argsDict["sdInfo_w_w_colind"] = self.coefficients.sdInfo[(3, 3)][1]
        argsDict["sdInfo_w_u_rowptr"] = self.coefficients.sdInfo[(3, 1)][0]
        argsDict["sdInfo_w_u_colind"] = self.coefficients.sdInfo[(3, 1)][1]
        argsDict["sdInfo_w_v_rowptr"] = self.coefficients.sdInfo[(3, 2)][0]
        argsDict["sdInfo_w_v_colind"] = self.coefficients.sdInfo[(3, 2)][1]
        argsDict["csrRowIndeces_p_p"] = self.csrRowIndeces[(0, 0)]
        argsDict["csrColumnOffsets_p_p"] = self.csrColumnOffsets[(0, 0)]
        argsDict["csrRowIndeces_p_u"] = self.csrRowIndeces[(0, 1)]
        argsDict["csrColumnOffsets_p_u"] = self.csrColumnOffsets[(0, 1)]
        argsDict["csrRowIndeces_p_v"] = self.csrRowIndeces[(0, 2)]
        argsDict["csrColumnOffsets_p_v"] = self.csrColumnOffsets[(0, 2)]
        argsDict["csrRowIndeces_p_w"] = self.csrRowIndeces[(0, 3)]
        argsDict["csrColumnOffsets_p_w"] = self.csrColumnOffsets[(0, 3)]
        argsDict["csrRowIndeces_u_p"] = self.csrRowIndeces[(1, 0)]
        argsDict["csrColumnOffsets_u_p"] = self.csrColumnOffsets[(1, 0)]
        argsDict["csrRowIndeces_u_u"] = self.csrRowIndeces[(1, 1)]
        argsDict["csrColumnOffsets_u_u"] = self.csrColumnOffsets[(1, 1)]
        argsDict["csrRowIndeces_u_v"] = self.csrRowIndeces[(1, 2)]
        argsDict["csrColumnOffsets_u_v"] = self.csrColumnOffsets[(1, 2)]
        argsDict["csrRowIndeces_u_w"] = self.csrRowIndeces[(1, 3)]
        argsDict["csrColumnOffsets_u_w"] = self.csrColumnOffsets[(1, 3)]
        argsDict["csrRowIndeces_v_p"] = self.csrRowIndeces[(2, 0)]
        argsDict["csrColumnOffsets_v_p"] = self.csrColumnOffsets[(2, 0)]
        argsDict["csrRowIndeces_v_u"] = self.csrRowIndeces[(2, 1)]
        argsDict["csrColumnOffsets_v_u"] = self.csrColumnOffsets[(2, 1)]
        argsDict["csrRowIndeces_v_v"] = self.csrRowIndeces[(2, 2)]
        argsDict["csrColumnOffsets_v_v"] = self.csrColumnOffsets[(2, 2)]
        argsDict["csrRowIndeces_v_w"] = self.csrRowIndeces[(2, 3)]
        argsDict["csrColumnOffsets_v_w"] = self.csrColumnOffsets[(2, 3)]
        argsDict["csrRowIndeces_w_p"] = self.csrRowIndeces[(3, 0)]
        argsDict["csrColumnOffsets_w_p"] = self.csrColumnOffsets[(3, 0)]
        argsDict["csrRowIndeces_w_u"] = self.csrRowIndeces[(3, 1)]
        argsDict["csrColumnOffsets_w_u"] = self.csrColumnOffsets[(3, 1)]
        argsDict["csrRowIndeces_w_v"] = self.csrRowIndeces[(3, 2)]
        argsDict["csrColumnOffsets_w_v"] = self.csrColumnOffsets[(3, 2)]
        argsDict["csrRowIndeces_w_w"] = self.csrRowIndeces[(3, 3)]
        argsDict["csrColumnOffsets_w_w"] = self.csrColumnOffsets[(3, 3)]
        argsDict["globalJacobian"] = jacobian.getCSRrepresentation()[2]
        argsDict["nExteriorElementBoundaries_global"] = self.mesh.nExteriorElementBoundaries_global
        argsDict["exteriorElementBoundariesArray"] = self.mesh.exteriorElementBoundariesArray
        argsDict["elementBoundaryElementsArray"] = self.mesh.elementBoundaryElementsArray
        argsDict["elementBoundaryLocalElementBoundariesArray"] = self.mesh.elementBoundaryLocalElementBoundariesArray
        argsDict["ebqe_vf_ext"] = self.coefficients.ebqe_vf
        argsDict["bc_ebqe_vf_ext"] = self.coefficients.bc_ebqe_vf
        argsDict["ebqe_phi_ext"] = self.coefficients.ebqe_phi
        argsDict["bc_ebqe_phi_ext"] = self.coefficients.bc_ebqe_phi
        argsDict["ebqe_normal_phi_ext"] = self.coefficients.ebqe_n
        argsDict["ebqe_kappa_phi_ext"] = self.coefficients.ebqe_kappa
        argsDict["ebqe_porosity_ext"] = self.coefficients.ebqe_porosity
        argsDict["ebqe_turb_var_0"] = self.coefficients.ebqe_turb_var[0]
        argsDict["ebqe_turb_var_1"] = self.coefficients.ebqe_turb_var[1]
        argsDict["isDOFBoundary_p"] = self.numericalFlux.isDOFBoundary[0]
        argsDict["isDOFBoundary_u"] = self.numericalFlux.isDOFBoundary[1]
        argsDict["isDOFBoundary_v"] = self.numericalFlux.isDOFBoundary[2]
        argsDict["isDOFBoundary_w"] = self.numericalFlux.isDOFBoundary[3]
        argsDict["isAdvectiveFluxBoundary_p"] = self.ebqe[('advectiveFlux_bc_flag', 0)]
        argsDict["isAdvectiveFluxBoundary_u"] = self.ebqe[('advectiveFlux_bc_flag', 1)]
        argsDict["isAdvectiveFluxBoundary_v"] = self.ebqe[('advectiveFlux_bc_flag', 2)]
        argsDict["isAdvectiveFluxBoundary_w"] = self.ebqe[('advectiveFlux_bc_flag', 3)]
        argsDict["isDiffusiveFluxBoundary_u"] = self.ebqe[('diffusiveFlux_bc_flag', 1, 1)]
        argsDict["isDiffusiveFluxBoundary_v"] = self.ebqe[('diffusiveFlux_bc_flag', 2, 2)]
        argsDict["isDiffusiveFluxBoundary_w"] = self.ebqe[('diffusiveFlux_bc_flag', 3, 3)]
        argsDict["ebqe_bc_p_ext"] = self.numericalFlux.ebqe[('u', 0)]
        argsDict["ebqe_bc_flux_mass_ext"] = self.ebqe[('advectiveFlux_bc', 0)]
        argsDict["ebqe_bc_flux_mom_u_adv_ext"] = self.ebqe[('advectiveFlux_bc', 1)]
        argsDict["ebqe_bc_flux_mom_v_adv_ext"] = self.ebqe[('advectiveFlux_bc', 2)]
        argsDict["ebqe_bc_flux_mom_w_adv_ext"] = self.ebqe[('advectiveFlux_bc', 3)]
        argsDict["ebqe_bc_u_ext"] = self.numericalFlux.ebqe[('u', 1)]
        argsDict["ebqe_bc_flux_u_diff_ext"] = self.ebqe[('diffusiveFlux_bc', 1, 1)]
        argsDict["ebqe_penalty_ext"] = self.ebqe['penalty']
        argsDict["ebqe_bc_v_ext"] = self.numericalFlux.ebqe[('u', 2)]
        argsDict["ebqe_bc_flux_v_diff_ext"] = self.ebqe[('diffusiveFlux_bc', 2, 2)]
        argsDict["ebqe_bc_w_ext"] = self.numericalFlux.ebqe[('u', 3)]
        argsDict["ebqe_bc_flux_w_diff_ext"] = self.ebqe[('diffusiveFlux_bc', 3, 3)]
        argsDict["csrColumnOffsets_eb_p_p"] = self.csrColumnOffsets_eb[(0, 0)]
        argsDict["csrColumnOffsets_eb_p_u"] = self.csrColumnOffsets_eb[(0, 1)]
        argsDict["csrColumnOffsets_eb_p_v"] = self.csrColumnOffsets_eb[(0, 2)]
        argsDict["csrColumnOffsets_eb_p_w"] = self.csrColumnOffsets_eb[(0, 3)]
        argsDict["csrColumnOffsets_eb_u_p"] = self.csrColumnOffsets_eb[(1, 0)]
        argsDict["csrColumnOffsets_eb_u_u"] = self.csrColumnOffsets_eb[(1, 1)]
        argsDict["csrColumnOffsets_eb_u_v"] = self.csrColumnOffsets_eb[(1, 2)]
        argsDict["csrColumnOffsets_eb_u_w"] = self.csrColumnOffsets_eb[(1, 3)]
        argsDict["csrColumnOffsets_eb_v_p"] = self.csrColumnOffsets_eb[(2, 0)]
        argsDict["csrColumnOffsets_eb_v_u"] = self.csrColumnOffsets_eb[(2, 1)]
        argsDict["csrColumnOffsets_eb_v_v"] = self.csrColumnOffsets_eb[(2, 2)]
        argsDict["csrColumnOffsets_eb_v_w"] = self.csrColumnOffsets_eb[(2, 3)]
        argsDict["csrColumnOffsets_eb_w_p"] = self.csrColumnOffsets_eb[(3, 0)]
        argsDict["csrColumnOffsets_eb_w_u"] = self.csrColumnOffsets_eb[(3, 1)]
        argsDict["csrColumnOffsets_eb_w_v"] = self.csrColumnOffsets_eb[(3, 2)]
        argsDict["csrColumnOffsets_eb_w_w"] = self.csrColumnOffsets_eb[(3, 3)]
        argsDict["q_dragBeam1"] = self.coefficients.q_dragBeam1
        argsDict["q_dragBeam2"] = self.coefficients.q_dragBeam2
        argsDict["q_dragBeam3"] = self.coefficients.q_dragBeam3
        argsDict["ebqe_dragBeam1"] = self.coefficients.ebqe_dragBeam1
        argsDict["ebqe_dragBeam2"] = self.coefficients.ebqe_dragBeam2
        argsDict["ebqe_dragBeam3"] = self.coefficients.ebqe_dragBeam3
        self.rans2p.calculateJacobian(argsDict)

        if not self.forceStrongConditions and max(numpy.linalg.norm(self.u[1].dof, numpy.inf), numpy.linalg.norm(self.u[2].dof, numpy.inf), numpy.linalg.norm(self.u[3].dof, numpy.inf)) < 1.0e-8:
            self.pp_hasConstantNullSpace = True
        else:
            self.pp_hasConstantNullSpace = False

        # Load the Dirichlet conditions directly into residual
        if self.forceStrongConditions:
            scaling = 1.0  # probably want to add some scaling to match non-dirichlet diagonals in linear system
            for cj in range(self.nc):
                for dofN in list(self.dirichletConditionsForceDOF[cj].DOFBoundaryConditionsDict.keys()):
                    global_dofN = self.offset[cj] + self.stride[cj] * dofN
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

    def calculateAuxiliaryQuantitiesAfterStep(self):
        # RANS2P.LevelModel.calculateAuxiliaryQuantitiesAfterStep(self)
        self.beamStep()

    def beamStep(self):
        self.coefficients.vel_avg = np.array([0.0, 0.0, 0.0])
        self.coefficients.q_dragBeam1.flat[:] = 0.0
        self.coefficients.q_dragBeam2.flat[:] = 0.0
        self.coefficients.q_dragBeam3.flat[:] = 0.0
        self.coefficients.ebqe_dragBeam1.flat[:] = 0.0
        self.coefficients.ebqe_dragBeam2.flat[:] = 0.0
        self.coefficients.ebqe_dragBeam3.flat[:] = 0.0
        self.q1 = numpy.zeros(self.coefficients.q1.shape, 'd')  # .flat[:]
        self.q2 = numpy.zeros(self.coefficients.q1.shape, 'd')  # .flat[:]
        self.q3 = numpy.zeros(self.coefficients.q1.shape, 'd')  # .flat[:]
        self.coefficients.netBeamDrag.flat[:] = 0.0

        argsDict = cArgumentsDict.ArgumentsDict()
        argsDict["mesh_trial_ref"] = self.u[0].femSpace.elementMaps.psi
        argsDict["mesh_grad_trial_ref"] = self.u[0].femSpace.elementMaps.grad_psi
        argsDict["mesh_dof"] = self.mesh.nodeArray
        argsDict["mesh_velocity_dof"] = self.mesh.nodeVelocityArray
        argsDict["MOVING_DOMAIN"] = self.MOVING_DOMAIN
        argsDict["mesh_l2g"] = self.mesh.elementNodesArray
        argsDict["dV_ref"] = self.elementQuadratureWeights[('u', 0)]
        argsDict["p_trial_ref"] = self.u[0].femSpace.psi
        argsDict["p_grad_trial_ref"] = self.u[0].femSpace.grad_psi
        argsDict["p_test_ref"] = self.u[0].femSpace.psi
        argsDict["p_grad_test_ref"] = self.u[0].femSpace.grad_psi
        argsDict["vel_trial_ref"] = self.u[1].femSpace.psi
        argsDict["vel_grad_trial_ref"] = self.u[1].femSpace.grad_psi
        argsDict["vel_test_ref"] = self.u[1].femSpace.psi
        argsDict["vel_grad_test_ref"] = self.u[1].femSpace.grad_psi
        argsDict["mesh_trial_trace_ref"] = self.u[0].femSpace.elementMaps.psi_trace
        argsDict["mesh_grad_trial_trace_ref"] = self.u[0].femSpace.elementMaps.grad_psi_trace
        argsDict["dS_ref"] = self.elementBoundaryQuadratureWeights[('u', 0)]
        argsDict["p_trial_trace_ref"] = self.u[0].femSpace.psi_trace
        argsDict["p_grad_trial_trace_ref"] = self.u[0].femSpace.grad_psi_trace
        argsDict["p_test_trace_ref"] = self.u[0].femSpace.psi_trace
        argsDict["p_grad_test_trace_ref"] = self.u[0].femSpace.grad_psi_trace
        argsDict["vel_trial_trace_ref"] = self.u[1].femSpace.psi_trace
        argsDict["vel_grad_trial_trace_ref"] = self.u[1].femSpace.grad_psi_trace
        argsDict["vel_test_trace_ref"] = self.u[1].femSpace.psi_trace
        argsDict["vel_grad_test_trace_ref"] = self.u[1].femSpace.grad_psi_trace
        argsDict["normal_ref"] = self.u[0].femSpace.elementMaps.boundaryNormals
        argsDict["boundaryJac_ref"] = self.u[0].femSpace.elementMaps.boundaryJacobians
        argsDict["eb_adjoint_sigma"] = self.eb_adjoint_sigma
        argsDict["elementDiameter"] = self.elementDiameter  # mesh.elementDiametersArray
        argsDict["nodeDiametersArray"] = self.mesh.nodeDiametersArray
        argsDict["hFactor"] = self.stabilization.hFactor
        argsDict["nElements_global"] = self.mesh.nElements_global
        argsDict["nElements_owned"] = self.mesh.nElements_owned
        argsDict["nElementBoundaries_owned"] = self.mesh.nElementBoundaries_owned
        argsDict["useRBLES"] = self.coefficients.useRBLES
        argsDict["useMetrics"] = self.coefficients.useMetrics
        argsDict["alphaBDF"] = self.timeIntegration.alpha_bdf
        argsDict["epsFact_rho"] = self.coefficients.epsFact_density
        argsDict["epsFact_mu"] = self.coefficients.epsFact
        argsDict["sigma"] = self.coefficients.sigma
        argsDict["rho_0"] = self.coefficients.rho_0
        argsDict["nu_0"] = self.coefficients.nu_0
        argsDict["rho_1"] = self.coefficients.rho_1
        argsDict["nu_1"] = self.coefficients.nu_1
        argsDict["smagorinskyConstant"] = self.coefficients.smagorinskyConstant
        argsDict["turbulenceClosureModel"] = self.coefficients.turbulenceClosureModel
        argsDict["Ct_sge"] = self.Ct_sge
        argsDict["Cd_sge"] = self.Cd_sge
        argsDict["C_dc"] = self.shockCapturing.shockCapturingFactor
        argsDict["C_b"] = self.numericalFlux.penalty_constant
        argsDict["eps_solid"] = self.coefficients.epsFact_solid
        argsDict["phi_solid"] = self.coefficients.q_phi_solid
        argsDict["q_velocity_solid"] = self.coefficients.q_velocity_solid
        argsDict["q_porosity"] = self.coefficients.q_porosity
        argsDict["q_dragAlpha"] = self.coefficients.q_dragAlpha
        argsDict["q_dragBeta"] = self.coefficients.q_dragBeta
        argsDict["q_mass_source"] = self.q[('r', 0)]
        argsDict["q_turb_var_0"] = self.coefficients.q_turb_var[0]
        argsDict["q_turb_var_1"] = self.coefficients.q_turb_var[1]
        argsDict["q_turb_var_grad_0"] = self.coefficients.q_turb_var_grad[0]
        argsDict["p_l2g"] = self.u[0].femSpace.dofMap.l2g
        argsDict["vel_l2g"] = self.u[1].femSpace.dofMap.l2g
        argsDict["p_dof"] = self.u[0].dof
        argsDict["u_dof"] = self.u[1].dof
        argsDict["v_dof"] = self.u[2].dof
        argsDict["w_dof"] = self.u[3].dof
        argsDict["g"] = self.coefficients.g
        argsDict["useVF"] = self.coefficients.useVF
        argsDict["vf"] = self.coefficients.q_vf
        argsDict["phi"] = self.coefficients.q_phi
        argsDict["normal_phi"] = self.coefficients.q_n
        argsDict["kappa_phi"] = self.coefficients.q_kappa
        argsDict["q_mom_u_acc"] = self.timeIntegration.m_tmp[1]
        argsDict["q_mom_v_acc"] = self.timeIntegration.m_tmp[2]
        argsDict["q_mom_w_acc"] = self.timeIntegration.m_tmp[3]
        argsDict["q_mass_adv"] = self.q[('f', 0)]
        argsDict["q_mom_u_acc_beta_bdf"] = self.timeIntegration.beta_bdf[1]
        argsDict["q_mom_v_acc_beta_bdf"] = self.timeIntegration.beta_bdf[2]
        argsDict["q_mom_w_acc_beta_bdf"] = self.timeIntegration.beta_bdf[3]
        argsDict["q_velocity_sge"] = self.stabilization.v_last
        argsDict["q_cfl"] = self.q[('cfl', 0)]
        argsDict["q_numDiff_u"] = self.q[('numDiff', 1, 1)]
        argsDict["q_numDiff_v"] = self.q[('numDiff', 2, 2)]
        argsDict["q_numDiff_w"] = self.q[('numDiff', 3, 3)]
        argsDict["q_numDiff_u_last"] = self.shockCapturing.numDiff_last[1]
        argsDict["q_numDiff_v_last"] = self.shockCapturing.numDiff_last[2]
        argsDict["q_numDiff_w_last"] = self.shockCapturing.numDiff_last[3]
        argsDict["sdInfo_u_u_rowptr"] = self.coefficients.sdInfo[(1, 1)][0]
        argsDict["sdInfo_u_u_colind"] = self.coefficients.sdInfo[(1, 1)][1]
        argsDict["sdInfo_u_v_rowptr"] = self.coefficients.sdInfo[(1, 2)][0]
        argsDict["sdInfo_u_v_colind"] = self.coefficients.sdInfo[(1, 2)][1]
        argsDict["sdInfo_u_w_rowptr"] = self.coefficients.sdInfo[(1, 3)][0]
        argsDict["sdInfo_u_w_colind"] = self.coefficients.sdInfo[(1, 3)][1]
        argsDict["sdInfo_v_v_rowptr"] = self.coefficients.sdInfo[(2, 2)][0]
        argsDict["sdInfo_v_v_colind"] = self.coefficients.sdInfo[(2, 2)][1]
        argsDict["sdInfo_v_u_rowptr"] = self.coefficients.sdInfo[(2, 1)][0]
        argsDict["sdInfo_v_u_colind"] = self.coefficients.sdInfo[(2, 1)][1]
        argsDict["sdInfo_v_w_rowptr"] = self.coefficients.sdInfo[(2, 3)][0]
        argsDict["sdInfo_v_w_colind"] = self.coefficients.sdInfo[(2, 3)][1]
        argsDict["sdInfo_w_w_rowptr"] = self.coefficients.sdInfo[(3, 3)][0]
        argsDict["sdInfo_w_w_colind"] = self.coefficients.sdInfo[(3, 3)][1]
        argsDict["sdInfo_w_u_rowptr"] = self.coefficients.sdInfo[(3, 1)][0]
        argsDict["sdInfo_w_u_colind"] = self.coefficients.sdInfo[(3, 1)][1]
        argsDict["sdInfo_w_v_rowptr"] = self.coefficients.sdInfo[(3, 2)][0]
        argsDict["sdInfo_w_v_colind"] = self.coefficients.sdInfo[(3, 2)][1]
        argsDict["offset_p"] = self.offset[0]
        argsDict["offset_u"] = self.offset[1]
        argsDict["offset_v"] = self.offset[2]
        argsDict["offset_w"] = self.offset[3]
        argsDict["stride_p"] = self.stride[0]
        argsDict["stride_u"] = self.stride[1]
        argsDict["stride_v"] = self.stride[2]
        argsDict["stride_w"] = self.stride[3]
        argsDict["globalResidual"] = self.r
        argsDict["nExteriorElementBoundaries_global"] = self.mesh.nExteriorElementBoundaries_global
        argsDict["exteriorElementBoundariesArray"] = self.mesh.exteriorElementBoundariesArray
        argsDict["elementBoundaryElementsArray"] = self.mesh.elementBoundaryElementsArray
        argsDict["elementBoundaryLocalElementBoundariesArray"] = self.mesh.elementBoundaryLocalElementBoundariesArray
        argsDict["ebqe_vf_ext"] = self.coefficients.ebqe_vf
        argsDict["bc_ebqe_vf_ext"] = self.coefficients.bc_ebqe_vf
        argsDict["ebqe_phi_ext"] = self.coefficients.ebqe_phi
        argsDict["bc_ebqe_phi_ext"] = self.coefficients.bc_ebqe_phi
        argsDict["ebqe_normal_phi_ext"] = self.coefficients.ebqe_n
        argsDict["ebqe_kappa_phi_ext"] = self.coefficients.ebqe_kappa
        argsDict["ebqe_porosity_ext"] = self.coefficients.ebqe_porosity
        argsDict["ebqe_turb_var_0"] = self.coefficients.ebqe_turb_var[0]
        argsDict["ebqe_turb_var_1"] = self.coefficients.ebqe_turb_var[1]
        argsDict["isDOFBoundary_p"] = self.numericalFlux.isDOFBoundary[0]
        argsDict["isDOFBoundary_u"] = self.numericalFlux.isDOFBoundary[1]
        argsDict["isDOFBoundary_v"] = self.numericalFlux.isDOFBoundary[2]
        argsDict["isDOFBoundary_w"] = self.numericalFlux.isDOFBoundary[3]
        argsDict["isAdvectiveFluxBoundary_p"] = self.ebqe[('advectiveFlux_bc_flag', 0)]
        argsDict["isAdvectiveFluxBoundary_u"] = self.ebqe[('advectiveFlux_bc_flag', 1)]
        argsDict["isAdvectiveFluxBoundary_v"] = self.ebqe[('advectiveFlux_bc_flag', 2)]
        argsDict["isAdvectiveFluxBoundary_w"] = self.ebqe[('advectiveFlux_bc_flag', 3)]
        argsDict["isDiffusiveFluxBoundary_u"] = self.ebqe[('diffusiveFlux_bc_flag', 1, 1)]
        argsDict["isDiffusiveFluxBoundary_v"] = self.ebqe[('diffusiveFlux_bc_flag', 2, 2)]
        argsDict["isDiffusiveFluxBoundary_w"] = self.ebqe[('diffusiveFlux_bc_flag', 3, 3)]
        argsDict["ebqe_bc_p_ext"] = self.numericalFlux.ebqe[('u', 0)]
        argsDict["ebqe_bc_flux_mass_ext"] = self.ebqe[('advectiveFlux_bc', 0)]
        argsDict["ebqe_bc_flux_mom_u_adv_ext"] = self.ebqe[('advectiveFlux_bc', 1)]
        argsDict["ebqe_bc_flux_mom_v_adv_ext"] = self.ebqe[('advectiveFlux_bc', 2)]
        argsDict["ebqe_bc_flux_mom_w_adv_ext"] = self.ebqe[('advectiveFlux_bc', 3)]
        argsDict["ebqe_bc_u_ext"] = self.numericalFlux.ebqe[('u', 1)]
        argsDict["ebqe_bc_flux_u_diff_ext"] = self.ebqe[('diffusiveFlux_bc', 1, 1)]
        argsDict["ebqe_penalty_ext"] = self.ebqe['penalty']
        argsDict["ebqe_bc_v_ext"] = self.numericalFlux.ebqe[('u', 2)]
        argsDict["ebqe_bc_flux_v_diff_ext"] = self.ebqe[('diffusiveFlux_bc', 2, 2)]
        argsDict["ebqe_bc_w_ext"] = self.numericalFlux.ebqe[('u', 3)]
        argsDict["ebqe_bc_flux_w_diff_ext"] = self.ebqe[('diffusiveFlux_bc', 3, 3)]
        argsDict["q_x"] = self.q['x']
        argsDict["q_velocity"] = self.q[('velocity', 0)]
        argsDict["ebqe_velocity"] = self.ebqe[('velocity', 0)]
        argsDict["flux"] = self.ebq_global[('totalFlux', 0)]
        argsDict["elementResidual_p_save"] = self.elementResidual[0]
        argsDict["boundaryFlags"] = self.mesh.elementBoundaryUserFlags
        argsDict["barycenters"] = self.coefficients.barycenters
        argsDict["wettedAreas"] = self.coefficients.wettedAreas
        argsDict["netForces_p"] = self.coefficients.netForces_p
        argsDict["netForces_v"] = self.coefficients.netForces_v
        argsDict["netMoments"] = self.coefficients.netMoments
        argsDict["q_dragBeam1"] = self.coefficients.q_dragBeam1
        argsDict["q_dragBeam2"] = self.coefficients.q_dragBeam2
        argsDict["q_dragBeam3"] = self.coefficients.q_dragBeam3
        argsDict["ebqe_dragBeam1"] = self.coefficients.ebqe_dragBeam1
        argsDict["ebqe_dragBeam2"] = self.coefficients.ebqe_dragBeam2
        argsDict["ebqe_dragBeam3"] = self.coefficients.ebqe_dragBeam3
        argsDict["nBeams"] = self.coefficients.nBeams
        argsDict["nBeamElements"] = self.coefficients.nBeamElements
        argsDict["beam_quadOrder"] = self.coefficients.beam_quadOrder
        argsDict["beam_Cd"] = self.coefficients.beam_Cd
        argsDict["beamRadius"] = self.coefficients.beamRadius
        argsDict["xq"] = self.coefficients.xq
        argsDict["yq"] = self.coefficients.yq
        argsDict["zq"] = self.coefficients.zq
        argsDict["Beam_h"] = self.coefficients.Beam_h
        argsDict["dV_beam"] = self.coefficients.dV_beam
        argsDict["q1"] = self.q1
        argsDict["q2"] = self.q2
        argsDict["q3"] = self.q3
        argsDict["vel_avg"] = self.coefficients.vel_avg
        argsDict["netBeamDrag"] = self.coefficients.netBeamDrag
        self.rans2p.calculateBeams(argsDict)
        for i in range(self.coefficients.nBeams):
            for j in range(self.coefficients.nBeamElements):
                for k in range(self.coefficients.beam_quadOrder):
                    # globalSum(self.q1[i*(self.coefficients.nBeamElements*self.coefficients.beam_quadOrder)+j*self.coefficients.beam_quadOrder+k])
                    self.coefficients.q1[i, j, k] = globalSum(self.q1[i, j, k])
                    # globalSum(self.q2[i*(self.coefficients.nBeamElements*self.coefficients.beam_quadOrder)+j*self.coefficients.beam_quadOrder+k])
                    self.coefficients.q2[i, j, k] = globalSum(self.q2[i, j, k])
                    # globalSum(self.q3[i*(self.coefficients.nBeamElements*self.coefficients.beam_quadOrder)+j*self.coefficients.beam_quadOrder+k])
                    self.coefficients.q3[i, j, k] = globalSum(self.q3[i, j, k])
        for i in range(3):
            self.coefficients.vel_avg[i] = globalSum(self.coefficients.vel_avg[i])

        self.coefficients.vel_avg = self.coefficients.vel_avg/0.1472
        self.coefficients.netBeamDrag[0] = globalSum(self.coefficients.netBeamDrag[0])
