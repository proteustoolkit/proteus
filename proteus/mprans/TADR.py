# A type of -*- python -*- file
"""
An optimized Advection-Diffusion-Reaction  transport module
"""
import os
import numpy as np
from proteus import cfemIntegrals, Quadrature, Norms, Comm, TimeIntegration
from proteus.NonlinearSolvers import NonlinearEquation
from proteus.FemTools import (DOFBoundaryConditions,
                              FluxBoundaryConditions,
                              C0_AffineLinearOnSimplexWithNodalBasis)

from proteus.Comm import globalMax, globalSum
from proteus.Profiling import (memory, logEvent)
from proteus.Transport import OneLevelTransport
from proteus.TransportCoefficients import TC_base
from proteus.SubgridError import SGE_base
from proteus.ShockCapturing import ShockCapturing_base
from proteus.NumericalFlux import Advection_DiagonalUpwind_Diffusion_IIPG_exterior
from proteus.LinearAlgebraTools import SparseMat, ParVec_petsc4py
from proteus.NonlinearSolvers import ExplicitLumpedMassMatrix,ExplicitConsistentMassMatrixForVOF,TwoStageNewton
from proteus.mprans.cTADR import cTADR_base

from . import cArgumentsDict

#: TADR_BC_DBG=1 logs the exterior-boundary solute-flux split once per time
#: step; see LevelModel._logBoundarySoluteFlux.  Off by default: the reduction
#: costs a handful of MPI allreduces per step.
_TADR_BC_DBG = os.environ.get("TADR_BC_DBG", "0") not in ("0", "false", "False")


class SubgridError(SGE_base):
    def __init__(self, coefficients, nd):
        SGE_base.__init__(self, coefficients, nd, lag=False)

    def initializeElementQuadrature(self, mesh, t, cq):
        pass

    def updateSubgridErrorHistory(self, initializationPhase=False):
        pass

    def calculateSubgridError(self, q):
        pass

class ShockCapturing(ShockCapturing_base):
    def __init__(self,
                 coefficients,
                 nd,
                 shockCapturingFactor=0.25,
                 lag=True,
                 nStepsToDelay=None):
        ShockCapturing_base.__init__(self,
                         coefficients,
                         nd,
                         shockCapturingFactor=shockCapturingFactor,
                         lag=lag)
        self.nStepsToDelay = nStepsToDelay
        self.nSteps = 0
        if self.lag:
            logEvent("TADR.ShockCapturing: lagging requested but must delay the first step; switching lagging off and delaying")
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
            logEvent("TADR.ShockCapturing: switched to lagged shock capturing")
            self.lag = True
            self.numDiff_last = []
            for ci in range(self.nc):
                self.numDiff_last.append(self.numDiff[ci].copy())
        logEvent("TADR: max numDiff %e" % (globalMax(self.numDiff_last[0].max()),))

class NumericalFlux(Advection_DiagonalUpwind_Diffusion_IIPG_exterior):
    def __init__(self,
                 vt,
                 getPointwiseBoundaryConditions,
                 getAdvectiveFluxBoundaryConditions,
                 getDiffusiveFluxBoundaryConditions):#,
                 #getPeriodicBoundaryConditions=None):
        Advection_DiagonalUpwind_Diffusion_IIPG_exterior.__init__(
            self,
            vt,
            getPointwiseBoundaryConditions,
            getAdvectiveFluxBoundaryConditions,
            getDiffusiveFluxBoundaryConditions)#,
            #getPeriodicBoundaryConditions)#
  
class RKEV(TimeIntegration.SSP):
    """
    Wrapper for SSPRK time integration using EV

    ... more to come ...
    """

    def __init__(self, transport, timeOrder=1, runCFL=0.1, integrateInterpolationPoints=False):
        TimeIntegration.SSP.__init__(self,
                                     transport,
                                     integrateInterpolationPoints=integrateInterpolationPoints)
        self.runCFL = runCFL
        self.dtLast = None
        self.isAdaptive = True
        assert transport.coefficients.STABILIZATION_TYPE>1, "SSP method just works for edge based EV methods; i.e., STABILIZATION_TYPE>1"
        assert hasattr(transport, 'edge_based_cfl'), "No edge based cfl defined"
        # About the cfl
        self.cfl = transport.edge_based_cfl
        # Stuff particular for SSP
        self.timeOrder = timeOrder  # order of approximation
        self.nStages = timeOrder  # number of stages total
        self.lstage = 0  # last stage completed
        # storage vectors
        self.u_dof_last = {}
        self.m_old = {}
        # per component stage values, list with array at each stage
        for ci in range(self.nc):
            self.m_last[ci] = transport.q[('u',ci)].copy()
            self.m_old[ci] = transport.q[('u',ci)].copy()
            self.u_dof_last[ci] = transport.u[ci].dof.copy()

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
                    # save stage at quad points
                    self.m_last[ci][:] = self.transport.q[('u',ci)]
                    # DOFs
                    self.transport.u_dof_old[:] = self.transport.u[ci].dof
            elif self.lstage == 2:
                logEvent("Second stage of SSP33 method", level=4)
                for ci in range(self.nc):
                    # Quad points
                    self.m_last[ci][:] = 1./4*self.transport.q[('u',ci)]
                    self.m_last[ci][:] += 3./4*self.m_old[ci]
                    # DOFs
                    self.transport.u_dof_old[:] = 1./4*self.transport.u[ci].dof
                    self.transport.u_dof_old[:] += 3./4* self.u_dof_last[ci]                
            elif self.lstage == 3:
                logEvent("Third stage of SSP33 method", level=4)
                for ci in range(self.nc):
                    # Quad points
                    self.m_last[ci][:] = 2./3*self.transport.q[('u',ci)]
                    self.m_last[ci][:] += 1./3*self.m_old[ci]
                    # DOFs
                    self.transport.u[0].dof[:] = 2./3*self.transport.u[ci].dof
                    self.transport.u[0].dof[:] += 1./3* self.u_dof_last[ci]
                    # update u_dof_old
                    self.transport.u_dof_old[:] = self.u_dof_last[ci]                    
        elif self.timeOrder == 2:
            if self.lstage == 1:
                logEvent("First stage of SSP22 method", level=4)
                for ci in range(self.nc):
                    # save stage at quad points
                    self.m_last[ci][:] = self.transport.q[('u',ci)]
                    # DOFs
                    self.transport.u_dof_old[:] = self.transport.u[ci].dof
            elif self.lstage == 2:
                logEvent("Second stage of SSP22 method", level=4)
                for ci in range(self.nc):
                    # Quad points
                    self.m_last[ci][:] = 1./2*self.transport.q[('u',ci)]
                    self.m_last[ci][:] += 1./2*self.m_old[ci]
                    # DOFs
                    self.transport.u[0].dof[:]  = 1./2*self.transport.u[ci].dof
                    self.transport.u[0].dof[:] += 1./2*self.u_dof_last[ci]
                    # update u_dof_old
                    self.transport.u_dof_old[:] = self.u_dof_last[ci]                    
        else:
            assert self.timeOrder == 1
            for ci in range(self.nc):
                self.m_last[ci][:] = self.transport.q[('u',ci)]

    def initializeTimeHistory(self, resetFromDOF=True):
        """
        Push necessary information into time history arrays
        """
        for ci in range(self.nc):
            self.m_old[ci][:] = self.transport.q[('u',ci)]
            self.m_last[ci][:] = self.transport.q[('u',ci)]
            self.u_dof_last[ci][:] = self.transport.u[ci].dof[:]

    def updateTimeHistory(self, resetFromDOF=False):
        """
        assumes successful step has been taken
        """
        self.t = self.tLast + self.dt
        for ci in range(self.nc):
            self.m_old[ci][:] = self.m_last[ci][:]
            self.u_dof_last[ci][:] = self.transport.u[ci].dof[:]
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
                    
class CompatibleBackwardEuler(TimeIntegration.BackwardEuler):
    """Backward Euler for STABILIZATION_TYPE='Compatible'. The transport takes one step per flow
    step, so it proposes no step of its own: the split-operator controller's minimum is the flow's."""

    def choose_dt(self):
        self.dt = 1.0e300
        self.t = self.tLast + self.dt



def checkCoupledQuadrature(modelA, nameA, modelB, nameB):
    """Verify two coupled models share a mesh and one quadrature rule.

    Coupled fields (velocity, water content, density) are passed between
    models as bare quadrature arrays indexed by eN*nQuadraturePoints_element+k,
    so both sides must agree on the quadrature or the receiving kernel reads
    the wrong entries -- or past the end of the array.

    nQuadraturePoints_element is NOT a shared setting: OneLevelTransport builds
    it per model in Quadrature.buildUnion over that model's own integral keys,
    plus any ('stab',..)/('numDiff',..) rules and the Lobatto points that
    massLumping/reactionLumping inject.  Two models handed the identical
    elementQuadrature object can therefore still end up with different point
    counts, and unions of equal size can order their points differently, which
    scrambles the coupled fields within each element without changing a shape.
    So compare the reference points themselves, not just their count.

    Called from attachModels on both sides of the Richards <-> TADR coupling,
    where the aliases are formed and before any kernel indexes them.
    """
    if modelA.mesh.nElements_global != modelB.mesh.nElements_global:
        raise ValueError(
            "Coupled models {} and {} are on different meshes "
            "({} vs {} elements); coupled quadrature fields cannot be "
            "exchanged.".format(nameA, nameB,
                                modelA.mesh.nElements_global,
                                modelB.mesh.nElements_global))
    for attr, label in (('elementQuadraturePoints', 'element'),
                        ('elementBoundaryQuadraturePoints', 'element boundary')):
        pA = getattr(modelA, attr, None)
        pB = getattr(modelB, attr, None)
        if pA is None or pB is None:
            continue
        if pA.shape != pB.shape or not np.allclose(pA, pB, rtol=0.0, atol=1e-14):
            raise ValueError(
                "Coupled models {} and {} do not share an {} quadrature rule "
                "({} vs {} points per element). Coupled fields (velocity, "
                "theta, rho) are exchanged as raw quadrature arrays, so both "
                "models must be given the same elementQuadrature and "
                "elementBoundaryQuadrature in their _n.py files, and the same "
                "massLumping/reactionLumping/stabilization/shockCapturing "
                "settings (each of these changes the quadrature union).".format(
                    nameA, nameB, label, pA.shape[0], pB.shape[0]))


class Coefficients(TC_base):
    from proteus.ctransportCoefficients import VOFCoefficientsEvaluate
    from proteus.ctransportCoefficients import VolumeAveragedVOFCoefficientsEvaluate
    from proteus.cfemIntegrals import copyExteriorElementBoundaryValuesFromElementBoundaryValues
    
    def __init__(self,
                 aOfX=None,
                 LS_model=None,
                 nd=2,
                 V_model=0,
                 RD_model=None,
                 ME_model=1,
                 VOS_model=None,
                 checkMass=True,
                 epsFact=0.0,
                 useMetrics=0.0,
                 sc_uref=1.0,
                 sc_beta=1.0,
                 movingDomain=False,
                 forceStrongConditions=False,
                 STABILIZATION_TYPE='VMS',        
                 ENTROPY_TYPE='POWER',
                 diagonal_conductivity=True,
                 # 0: quadratic
                 # 1: logarithmic
                 # FOR ENTROPY VISCOSITY
                 cE=1.0,
                 cMax=1.0,
                 uL=0.0,
                 uR=1.0,
                 # FOR ARTIFICIAL COMPRESSION
                 cK=0.0,
                 LUMPED_MASS_MATRIX=False,
                 FCT=True,
                 outputQuantDOFs=False,
                 #NULLSPACE INFO
                 nullSpace='NoNullSpace',
                 initialize=True,
                 physicalDiffusion=0.0,
                 alpha_L=0.0,
                 alpha_T=0.0,
                 Dm=None,
                 porosity=1.0,
                 dispersion_type=0,
                 theta_s=1.0,
                 theta_r=0.0,
                 power_law_exponent=0.0,
                 velocity_exponent=1.0,
                 rho_f=1.0,
                 rho_s=1.0,
                 specified_velocity=True,
                 # Kinetic dissolution source: R_diss = k_d * S_n * S_w * (c_sat - c)
                 # k_d : volumetric mass-transfer rate coefficient [1/s].  Default 0
                 #       disables dissolution (Richards <-> TADR coupling is unchanged).
                 # c_sat: saturation concentration (Henry's law equilibrium).
                 # S_n : taken from vModel.u[1].dof if the flow model exposes a
                 #       gas saturation (mphase_co2); else treated as zero.
                 k_d=0.0,
                 c_sat=1.0):
        self.variableNames = ['u']
        self.LS_modelIndex = LS_model
        self.V_model = V_model
        self.RD_modelIndex = RD_model
        self.modelIndex = ME_model
        self.VOS_model=VOS_model
        self.checkMass = checkMass
        self.epsFact = epsFact
        self.flowModelIndex = V_model
        self.modelIndex = ME_model
        self.RD_modelIndex = RD_model
        self.LS_modelIndex = LS_model
        self.V_model = V_model
        self.RD_modelIndex = RD_model
        self.modelIndex = ME_model
        self.VOS_model=VOS_model
        self.checkMass = checkMass
        self.epsFact = epsFact
        self.useMetrics = useMetrics
        self.sc_uref = sc_uref
        self.sc_beta = sc_beta
        self.movingDomain = movingDomain
        self.forceStrongConditions = forceStrongConditions
        self.nd=nd
        self.aOfX = aOfX
        self.diagonal_conductivity = diagonal_conductivity

        if self.diagonal_conductivity:
            sdInfo= {(0,0):(np.arange(self.nd+1,dtype='i'),
                            np.arange(self.nd,dtype='i'))}
        else:
            sdInfo = {(0, 0): (np.arange(start=0, stop=nd**2 + 1, step=nd, dtype='int32'),
                   np.array([range(nd) for _ in range(nd)], dtype='int32'))}
        
        self.cE = cE
        self.cMax = cMax
        self.uL = uL
        self.uR = uR
        self.cK = cK
        self.LUMPED_MASS_MATRIX = LUMPED_MASS_MATRIX
        self.FCT = FCT
        self.outputQuantDOFs = outputQuantDOFs
        self.nullSpace = nullSpace
        self.flowCoefficients = None
        self.physicalDiffusion=physicalDiffusion
        self.alpha_L = alpha_L
        self.alpha_T = alpha_T
        self.Dm = physicalDiffusion if Dm is None else Dm
        # aOfX (the old interface) is the default: the kernel builds its tensor
        # from (Dm, alpha_L, alpha_T), so when the caller supplies aOfX and none
        # of those, LevelModel.__init__ back-fills Dm from the aOfX tensor.  A
        # caller that names any of them has specified the new dispersion model,
        # and it wins over aOfX.
        self.dispersion_specified = (Dm is not None or
                                     alpha_L != 0.0 or
                                     alpha_T != 0.0 or
                                     physicalDiffusion != 0.0)
        self.porosity = porosity
        dispersion_types = {"Constant":0,
                            "PowerLawSaturation":1,
                            "VelocityBased":2}
        try:
            if isinstance(dispersion_type, int):
                if dispersion_type not in [0, 1, 2]:
                    raise ValueError
                self.dispersion_type = dispersion_type
            else:
                self.dispersion_type = dispersion_types[dispersion_type]
        except:
            raise ValueError("dispersion_type must be one of "+str(dispersion_types.keys())+" or an int in [0,1,2], not "+str(dispersion_type))
        self.theta_s = theta_s
        self.theta_r = theta_r
        self.power_law_exponent = power_law_exponent
        self.velocity_exponent = velocity_exponent
        self.rho_f = rho_f
        self.rho_s = rho_s
        self.specified_velocity = specified_velocity
        self.k_d = k_d
        self.c_sat = c_sat
        self.sparseDiffusionTensors = sdInfo
        if initialize:
            self.initialize()


        #must keep synchronized with TADR.h enums
        stabilization_types = {"Galerkin":-1,
                               "VMS":0,
                               "TaylorGalerkinEV":1,
                               "EntropyViscosity":2,
                               "SmoothnessIndicator":3,
                               "Kuzmin":4,
                               "ImplicitEV":5,
                               "Compatible":6}
        entropy_types = {'POWER':0 , 
                         'LOG':1}
        try:
            if isinstance(STABILIZATION_TYPE, int):
                STABILIZATION_TYPE = [key for key, value in stabilization_types.items() if value == STABILIZATION_TYPE][0]
            
            self.STABILIZATION_TYPE = stabilization_types[STABILIZATION_TYPE]
        except:
            raise ValueError("STABILIZATION_TYPE must be one of "+str(stabilization_types.keys())+" not "+STABILIZATION_TYPE)
        try:
            if isinstance(ENTROPY_TYPE, int):
                ENTROPY_TYPE = [key for key, value in entropy_types.items() if value == ENTROPY_TYPE][0]

            self.ENTROPY_TYPE = entropy_types[ENTROPY_TYPE]
        except:
            raise ValueError("ENTROPY_TYPE must be one of "+str(entropy_types.keys())+" not "+str(ENTROPY_TYPE))
        self.cE = cE
        self.cMax = cMax
        self.uL = uL
        self.uR = uR
        self.cK = cK
        self.LUMPED_MASS_MATRIX = LUMPED_MASS_MATRIX
        self.FCT = FCT
        self.outputQuantDOFs = outputQuantDOFs
        self.nullSpace = nullSpace
        self.flowCoefficients = None
        self.physicalDiffusion=physicalDiffusion
        self.alpha_L = alpha_L
        self.alpha_T = alpha_T
        self.Dm = physicalDiffusion if Dm is None else Dm
        self.porosity = porosity
        self.theta_s = theta_s
        self.theta_r = theta_r
        self.power_law_exponent = power_law_exponent
        self.velocity_exponent = velocity_exponent
        self.rho_f = rho_f
        self.rho_s = rho_s
        self.specified_velocity = specified_velocity
        if initialize:
            self.initialize()
   

    def initialize(self):
        nc = 1
        mass = {0: {0: 'linear'}}
        advection = {0: {0: 'linear'}}
        hamiltonian = {}
        diffusion = {0: {0: {0: 'constant'}}}
        potential = {0: {0: 'u'} }
        reaction = {0: {0: 'linear'}}
        sparseDiffusionTensors = {(0,0):(np.arange(self.nd+1,dtype='i'),
                                         np.arange(self.nd,dtype='i'))}

        

        TC_base.__init__(self,
                         nc,
                         mass,
                         advection,
                         diffusion,
                         potential,
                         reaction,
                         hamiltonian,
                         self.variableNames,
                         movingDomain=self.movingDomain,
                         sparseDiffusionTensors=sparseDiffusionTensors)
                

    def initializeMesh(self, mesh):
        self.eps = self.epsFact * mesh.h
        
    def attachModels(self, modelList):
        # self
        self.model = modelList[self.modelIndex]
        # redistanced level set
        if self.RD_modelIndex is not None:
            self.rdModel = modelList[self.RD_modelIndex]
        # level set
        if self.LS_modelIndex is not None:
            self.lsModel = modelList[self.LS_modelIndex]
            self.q_phi = modelList[self.LS_modelIndex].q[('u', 0)]
            self.ebqe_phi = modelList[self.LS_modelIndex].ebqe[('u', 0)]
            if ('u', 0) in modelList[self.LS_modelIndex].ebq:
                self.ebq_phi = modelList[self.LS_modelIndex].ebq[('u', 0)]
        else:
            self.ebqe_phi = np.zeros(self.model.ebqe[('u', 0)].shape, 'd') # cek hack, we don't need this
        # flow model
        if self.V_model is not None:
            self.vModel = modelList[self.V_model]
            # q_v/q_porosity below are bare aliases (or shape-inheriting copies)
            # of the flow model's quadrature arrays, which this model's kernels
            # then index with their OWN nQuadraturePoints_element.  Nothing
            # downstream would catch a mismatch -- the copies keep the donor's
            # shape, so even the [:] refresh in preStep stays silent -- so check
            # the rules agree here, before the first alias is taken.
            checkCoupledQuadrature(self.model, 'TADR', self.vModel, 'flow model')
            if self.specified_velocity:
                if ('velocity', 0) in self.vModel.q:
                    self.q_v = self.vModel.q[('velocity', 0)]
                    self.ebqe_v = self.vModel.ebqe[('velocity', 0)]
                else:
                    self.q_v = self.vModel.q[('f', 0)]
                    self.ebqe_v = self.vModel.ebqe[('f', 0)]
                if ('velocity', 0) in self.vModel.ebq:
                    self.ebq_v = self.vModel.ebq[('velocity', 0)]
                else:
                    if ('f', 0) in self.vModel.ebq:
                        self.ebq_v = self.vModel.ebq[('f', 0)]
            else:
                # Use coupled velocity produced by Richards: ('velocity_couple', 0)
                self.q_v = self.vModel.q[('velocity_couple', 0)].copy()
                self.ebqe_v = self.vModel.ebqe[('velocity_couple', 0)].copy()
                self.ebq_v = None
            if (not self.specified_velocity and
                    ('theta', 0) in self.vModel.q and
                    ('theta', 0) in self.vModel.ebqe):
                # Use Richards water content in the transport mass/storage term.
                self.q_porosity = self.vModel.q[('theta', 0)]
                self.ebqe_porosity = self.vModel.ebqe[('theta', 0)]
            else:
                self.q_porosity = np.full(self.model.q[('u', 0)].shape, self.porosity, 'd')
                self.ebqe_porosity = np.full(self.model.ebqe[('u', 0)].shape, self.porosity, 'd')
            # q_porosity is an ALIAS of the flow model's theta, but the lagged
            # q_porosity_old is a COPY refreshed only in postStep.  At
            # attachModels the flow model has not evaluated a residual yet, so
            # its q[('theta',0)] is still the zeros allocated in
            # initializeElementQuadrature -- copying it hands the FIRST TADR
            # residual a porosity of exactly 0.  That is not merely inaccurate:
            # theta_dof_proj is built from q_porosity_old (TADR.h:637) and the
            # group variable divides by it, so a zero floors to 1e-14 while the
            # BOUNDARY loop reads the live ebqe_porosity alias (correctly 0.1).
            # The interior mass is then 0 against a non-zero boundary integral
            # and uLow = mLow/fmax(theta*rho_f,1e-14) blows up (measured
            # 1.49e6 on Elder, first call only).  Fall back to the configured
            # porosity while the donor is unpopulated; a porosity that is
            # identically zero everywhere is not physical.
            self.q_porosity_old = self.q_porosity.copy()
            if not self.q_porosity_old.any():
                self.q_porosity_old[:] = self.porosity
            self.q_rho = np.full(self.model.q[('u', 0)].shape, self.rho_f, 'd')
            self.ebqe_rho = np.full(self.model.ebqe[('u', 0)].shape, self.rho_f, 'd')
            self.q_v_old = self.q_v.copy()
            self.q_rho_old = self.q_rho.copy()
            # Sn_dof: gas saturation DOFs from a two-phase flow model.  When
            # vModel has a second primary variable (mphase_co2: u[1] = S_n),
            # alias it -- C0 P1 nodal DOFs are shared with TADR on the same
            # mesh, so this is direct DOF-to-DOF transfer (no projection).
            # When vModel has only one primary variable (Richards), Sn stays
            # zero and the dissolution source R_diss = k_d * S_n * S_w *
            # (c_sat - c) vanishes -- Richards <-> TADR coupling is unchanged.
            if len(self.vModel.u) >= 2 and hasattr(self.vModel.u[1], 'dof'):
                self.Sn_dof = self.vModel.u[1].dof
            else:
                self.Sn_dof = np.zeros_like(self.model.u[0].dof)
        else:
            self.q_v = np.ones(self.model.q[('u',0)].shape+(self.model.nSpace_global,),'d')
            self.ebqe_v = np.ones(self.model.ebqe[('u',0)].shape+(self.model.nSpace_global,),'d')
            self.q_porosity = np.full(self.model.q[('u', 0)].shape, self.porosity, 'd')
            self.ebqe_porosity = np.full(self.model.ebqe[('u', 0)].shape, self.porosity, 'd')
            self.q_porosity_old = self.q_porosity.copy()
            self.q_rho = np.full(self.model.q[('u', 0)].shape, self.rho_f, 'd')
            self.ebqe_rho = np.full(self.model.ebqe[('u', 0)].shape, self.rho_f, 'd')
            self.q_v_old = self.q_v.copy()
            self.q_rho_old = self.q_rho.copy()
            self.Sn_dof = np.zeros_like(self.model.u[0].dof)
        # VRANS
        if self.V_model is not None:
            self.flowCoefficients = modelList[self.V_model].coefficients
        else:
            self.flowCoefficients = None
        if self.STABILIZATION_TYPE == 6:
            # Compatible coupling: the flow model exports its realized exchanges every step
            # and takes rho_T(c) at the nodes from this model (syncCompatibleDensity).
            assert self.V_model is not None and hasattr(self.flowCoefficients, 'exportTransportFluxes'), \
                "STABILIZATION_TYPE='Compatible' needs V_model to be a Richards model"
            assert self.flowCoefficients.STABILIZATION_TYPE > 0, \
                "STABILIZATION_TYPE='Compatible' needs an edge-based Richards model (STABILIZATION_TYPE > 0)"
            self.flowCoefficients.exportTransportFluxes = True
            self.nodal_rho = np.full(self.model.u[0].dof.shape, self.rho_f, 'd')

    def syncCompatibleDensity(self):
        """rho_T(c) = rho_f (1 + eps c) from the current DOFs: at the nodes (nodal_rho, the flow's
        rho_T,i) and at the element and boundary quadrature points (q_rho, ebqe_rho)."""
        model = self.model
        c = model.u[0].dof
        eps = (self.rho_s - self.rho_f) / self.rho_f
        self.nodal_rho[:] = self.rho_f * (1.0 + eps * c)
        fs = model.u[0].femSpace
        l2g = fs.dofMap.l2g
        self.q_rho[:] = self.rho_f * (1.0 + eps * np.einsum('ej,kj->ek', c[l2g], fs.psi))
        mesh = model.mesh
        ebN = mesh.exteriorElementBoundariesArray
        eN = mesh.elementBoundaryElementsArray[ebN, 0]
        ebN_local = mesh.elementBoundaryLocalElementBoundariesArray[ebN, 0]
        psi_trace = np.asarray(fs.psi_trace).reshape((mesh.nElementBoundaries_element,
                                                      self.ebqe_rho.shape[1],
                                                      l2g.shape[1]))
        self.ebqe_rho[:] = self.rho_f * (1.0 + eps * np.einsum('ej,ekj->ek', c[l2g[eN]], psi_trace[ebN_local]))
        

    def preStep(self, t, firstStep=False):
        # SAVE OLD SOLUTION #
        self.model.u_dof_old[:] = self.model.u[0].dof
        self.q_v_old[:] = self.q_v
        self.q_rho_old[:] = self.q_rho

        # Restart flags for stages of taylor galerkin
        self.model.stage = 1
        self.model.auxTaylorGalerkinFlag = 1
        
        # COMPUTE NEW VELOCITY (if given by user) #
        if self.model.hasVelocityFieldAsFunction and self.model.coefficients.specified_velocity:
            self.model.updateVelocityFieldAsFunction()
        elif (not self.model.coefficients.specified_velocity and
              self.V_model is not None):
            # Refresh coupled velocity every step from Richards.
            self.q_v[:] = self.vModel.q[('velocity_couple', 0)]
            self.ebqe_v[:] = self.vModel.ebqe[('velocity_couple', 0)]

        if firstStep:
            # No previous step to lag against, and the lag above ran before the
            # velocity existed.  Re-seed so velocity_old == velocity on step 1;
            # STABILIZATION_TYPE 2/3/4 read velocity_old in the low-order flux.
            self.q_v_old[:] = self.q_v

        if self.checkMass:
            self.m_pre = Norms.scalarDomainIntegral(self.model.q['dV_last'],
                                                    self.model.q[('m', 0)],
                                                    self.model.mesh.nElements_owned)
            logEvent("Phase  0 mass before TADR step = %12.5e" % (self.m_pre,), level=2)
        copyInstructions = {}
        return copyInstructions

    def postStep(self, t, firstStep=False):
        # OPTION C: ImplicitEV no longer adopts an advected fluid mass here --
        # it re-reads M from Richards at the top of every residual, so there is
        # nothing to adopt.  Report the conserved solute mass instead: that is
        # the quantity this trade puts at risk, and it is the number to watch.
        if self.STABILIZATION_TYPE == 5:
            self.model._reportSoluteMass("postStep")
            # The defect that says how much of the above is real.  Here, not in
            # _reportSoluteMass: it needs Richards' M^{n+1}, and under
            # Sequential_MinModelStep the flow model has finished its step by now.
            self.model._reportContinuityDefect("postStep")
        # q_porosity aliases the coupled Richards theta field, so cache the
        # lagged copy here before Richards advances it on the next split step.
        self.q_porosity_old[:] = self.q_porosity
        self.model.q['dV_last'][:] = self.model.q['dV']
        if self.checkMass:
            self.m_post = Norms.scalarDomainIntegral(self.model.q['dV'],
                                                     self.model.q[('m', 0)],
                                                     self.model.mesh.nElements_owned)
            logEvent("Phase  0 mass after TADR step = %12.5e" % (self.m_post,), level=2)

        # ---- coupling diagnostic: MPI-reduced, print on rank 0 ----
        from mpi4py import MPI
        comm = MPI.COMM_WORLD

        def _global_stats(local):
            a = np.asarray(local)
            lo = comm.allreduce(float(a.min()) if a.size else float('inf'), op=MPI.MIN)
            hi = comm.allreduce(float(a.max()) if a.size else float('-inf'), op=MPI.MAX)
            ssum = comm.allreduce(float(a.sum()), op=MPI.SUM)
            n = comm.allreduce(int(a.size), op=MPI.SUM)
            return lo, hi, (ssum / n if n > 0 else float('nan'))

        q_rho = getattr(self, 'q_rho', None)
        ebqe_rho = getattr(self, 'ebqe_rho', None)
        # OWNED ELEMENTS ONLY for the element-quadrature fields.  _global_stats
        # forms its mean as allreduce(sum)/allreduce(size) over the WHOLE local
        # array, and the tail of that array is the overlap elements -- which the
        # owning rank also counts.  The duplicated fraction grows with the
        # partition surface, so the reported mean drifts with RANK COUNT for a
        # reason that has nothing to do with the solution: measured here, the
        # 1-vs-8-rank spread in mean(u) read 1.9e-2 against 1.2e-3 on the
        # owned-DOF solute mass, i.e. the diagnostic was ~20x noisier than what
        # it was reporting on.  min/max are unaffected (duplicates cannot change
        # an extremum), so only the slice changes here.
        _no = self.model.mesh.nElements_owned
        u_field = self.model.q[('u', 0)][:_no]
        if q_rho is not None:
            u_lo, u_hi, u_mn = _global_stats(u_field)
            r_lo, r_hi, r_mn = _global_stats(q_rho[:_no])
            if comm.Get_rank() == 0:
                logEvent(
                    "[Coupling rho q] TADR.postStep t={:.6e} "
                    "u (min,max,mean)=({:.6e},{:.6e},{:.6e}) "
                    "TADR.q_rho (min,max,mean)=({:.6e},{:.6e},{:.6e})".format(
                        float(t), u_lo, u_hi, u_mn, r_lo, r_hi, r_mn),
                    level=2)
        if ebqe_rho is not None:
            e_lo, e_hi, e_mn = _global_stats(ebqe_rho)
            if comm.Get_rank() == 0:
                logEvent(
                    "[Coupling rho ebqe] TADR.postStep t={:.6e} "
                    "TADR.ebqe_rho (min,max,mean)=({:.6e},{:.6e},{:.6e})".format(
                        float(t), e_lo, e_hi, e_mn),
                    level=2)

        #self._write_velocity_output(t)
        copyInstructions = {}
        return copyInstructions

    # def _write_velocity_output(self, t):
    #     comm = Comm.get()
    #     rank = comm.rank()
    #     mpicomm = comm.comm.tompi4py() if hasattr(comm.comm, "tompi4py") else comm.comm
    #     nSpace = int(getattr(self.model, "nSpace_global", getattr(self.model, "nSpace", self.nd)))

    #     # Avoid duplicate writes if postStep is revisited at the same time.
    #     if hasattr(self, "_last_velocity_output_time") and abs(t - self._last_velocity_output_time) <= 1.0e-12:
    #         return

    #     if not hasattr(self, "_wrote_velocity_coords_once"):
    #         qcoords_local = np.asarray(self.model.q['x']).reshape((-1, 3))
    #         qcoords_all = mpicomm.gather(qcoords_local, root=0)
    #         if rank == 0:
    #             Q = np.vstack(qcoords_all)
    #             np.savetxt("RE_q_coords_all.txt",
    #                        Q,
    #                        fmt="%.16e",
    #                        header=f"columns: x y z | total_rows={Q.shape[0]}")
    #             logEvent(f"[TADR.postStep] wrote RE_q_coords_all.txt rows={Q.shape[0]}")
    #         self._wrote_velocity_coords_once = True

    #     qv_local = np.asarray(self.q_v).reshape((-1, nSpace))
    #     qv_all = mpicomm.gather(qv_local, root=0)
    #     if rank == 0:
    #         V = np.vstack(qv_all)
    #         header_cols = "vx vy" if nSpace == 2 else "vx vy vz"
    #         np.savetxt(f"RE_q_vel_all_t{t:.8e}.txt",
    #                    V,
    #                    fmt="%.16e",
    #                    header=f"columns: {header_cols} | t={t:.16e} | total_rows={V.shape[0]}")
    #         logEvent(f"[TADR.postStep] wrote RE_q_vel_all_t{t:.8e}.txt rows={V.shape[0]}")
    #     self._last_velocity_output_time = t

    def evaluate(self, t, c):
        if c[('f', 0)].shape == self.q_v.shape:
            v = self.q_v
            phi = self.q_phi


        elif c[('f', 0)].shape == self.ebqe_v.shape:
            v = self.ebqe_v
            phi = self.ebqe_phi

        elif ((self.ebq_v is not None and self.ebq_phi is not None) and c[('f', 0)].shape == self.ebq_v.shape):
            v = self.ebq_v
            phi = self.ebq_phi

        else:
            v = None
            phi = None
        if v is not None:
            c[('m',0)] = c[('u',0)]
            c[('dm',0,0)] = np.ones_like(c[('u',0)])
            c[('f',0)][:] = v*c[('u',0)]
            c[('df',0,0)][:] = v
            #c[('a',0,0)][:] = self.physicalDiffusion
            nd = self.nd
            if self.aOfX is not None:
                for i in range(len(c[('r', 0)].flat)):
                    x_i = c['x'].flat[3*i:3*(i+1)]
                    c[('a', 0, 0)].flat[nd*nd*i:nd*nd*(i+1)] = self.aOfX[0](x_i).flat
                #print(c[('a', 0, 0)])
            
            # Compute diffusion coefficients at each quadrature point
            #for i in range(len(c['x'])):
            #    x_i = c['x'][i]
            #    diffusion_matrix = self.aOfX[0](x_i)
            #    c[('a', 0, 0)][i] = diffusion_matrix


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

        self.auxiliaryCallCalculateResidual = False
        #
        # set the objects describing the method and boundary conditions
        #
        self.bdyNullSpace = bdyNullSpace
        self.movingDomain = movingDomain
        self.tLast_mesh = None
        #
        self.name = name
        self.sd = sd
        self.Hess = False
        self.timeTerm = True  # allow turning off  the  time derivative
        self.testIsTrial = True
        self.phiTrialIsTrial = True
        self.u = uDict
        self.ua = {}  # analytical solutions
        self.phi = phiDict
        self.dphi = {}
        self.matType = matType
        self.reuse_test_trial_quadrature = reuse_trial_and_test_quadrature
        if self.reuse_test_trial_quadrature:
            for ci in range(1, coefficients.nc):
                assert self.u[ci].femSpace.__class__.__name__ == self.u[0].femSpace.__class__.__name__, "to reuse_test_trial_quad all femSpaces must be the same!"
        self.u_dof_old = None
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
        # storage dictionaries
        self.scalars_element = set()
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
        self.q[('u', 0)] = np.zeros((self.mesh.nElements_global, self.nQuadraturePoints_element), 'd')
        self.q[('dV_u', 0)] = (1.0/self.mesh.nElements_global) * np.ones((self.mesh.nElements_global, self.nQuadraturePoints_element), 'd')
        self.q[('grad(u)', 0)] = np.zeros((self.mesh.nElements_global, self.nQuadraturePoints_element, self.nSpace_global), 'd')
        self.q[('m_last', 0)] = np.zeros((self.mesh.nElements_global, self.nQuadraturePoints_element), 'd')
        self.q[('mt', 0)] = np.zeros((self.mesh.nElements_global, self.nQuadraturePoints_element), 'd')
        self.q['dV'] = np.zeros((self.mesh.nElements_global, self.nQuadraturePoints_element), 'd')
        self.q['dV_last'] = -1000 * np.ones((self.mesh.nElements_global, self.nQuadraturePoints_element), 'd')
        self.q[('m_tmp', 0)] = self.q[('u', 0)].copy()
        self.q[('m', 0)] = self.q[('m_tmp', 0)]
        self.q[('cfl', 0)] = np.zeros((self.mesh.nElements_global, self.nQuadraturePoints_element), 'd')
        self.q[('numDiff', 0, 0)] = np.zeros((self.mesh.nElements_global, self.nQuadraturePoints_element), 'd')
        ###################################################
        self.q[('a',0,0)] = np.zeros((self.mesh.nElements_global,self.nQuadraturePoints_element,self.coefficients.sdInfo[(0,0)][0][-1]),'d')
        nd = self.coefficients.nd

        
        
        self.q[('r',0)] = np.zeros((self.mesh.nElements_global,self.nQuadraturePoints_element),'d')

        ###################################################
        #self.calculateQuadrature()



        self.ebqe[('u', 0)] = np.zeros((self.mesh.nExteriorElementBoundaries_global, self.nElementBoundaryQuadraturePoints_elementBoundary), 'd')
        self.ebqe[('grad(u)', 0)] = np.zeros((self.mesh.nExteriorElementBoundaries_global,
                                                 self.nElementBoundaryQuadraturePoints_elementBoundary, self.nSpace_global), 'd')
        self.ebqe[('advectiveFlux_bc_flag', 0)] = np.zeros(
            (self.mesh.nExteriorElementBoundaries_global, self.nElementBoundaryQuadraturePoints_elementBoundary), 'i')
        self.ebqe[('advectiveFlux_bc', 0)] = np.zeros((self.mesh.nExteriorElementBoundaries_global, self.nElementBoundaryQuadraturePoints_elementBoundary), 'd')
        self.ebqe[('advectiveFlux', 0)] = np.zeros((self.mesh.nExteriorElementBoundaries_global, self.nElementBoundaryQuadraturePoints_elementBoundary), 'd')
        # BOUNDARY-FLUX DIAGNOSTIC, see TADR_BC_DBG in getResidual.
        # ('advectiveFlux',0) holds advection PLUS dispersion; these hold the
        # advective half on its own and the quadrature weight, so the two can be
        # separated and integrated.
        self.ebqe[('advectiveFluxOnly', 0)] = np.zeros((self.mesh.nExteriorElementBoundaries_global, self.nElementBoundaryQuadraturePoints_elementBoundary), 'd')
        self.ebqe['dS'] = np.zeros((self.mesh.nExteriorElementBoundaries_global, self.nElementBoundaryQuadraturePoints_elementBoundary), 'd')
        #######################################################
        self.ebqe[('a',0,0)] = np.zeros((self.mesh.nExteriorElementBoundaries_global,self.nElementBoundaryQuadraturePoints_elementBoundary,self.coefficients.sdInfo[(0,0)][0][-1]),'d')
        self.ebqe[('df',0,0)] = np.zeros((self.mesh.nExteriorElementBoundaries_global,self.nElementBoundaryQuadraturePoints_elementBoundary,self.nSpace_global),'d')
        self.ebqe[('r',0)] = np.zeros((self.mesh.nExteriorElementBoundaries_global,self.nElementBoundaryQuadraturePoints_elementBoundary),'d')
        self.ebqe['penalty'] = np.zeros((self.mesh.nExteriorElementBoundaries_global,self.nElementBoundaryQuadraturePoints_elementBoundary),'d')
        self.ebqe[('diffusiveFlux_bc_flag',0,0)] = np.zeros((self.mesh.nExteriorElementBoundaries_global,self.nElementBoundaryQuadraturePoints_elementBoundary),'i')
        self.ebqe[('diffusiveFlux_bc',0,0)] = np.zeros((self.mesh.nExteriorElementBoundaries_global,self.nElementBoundaryQuadraturePoints_elementBoundary),'d')
        
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

        for ebNE in range(self.mesh.nExteriorElementBoundaries_global):
            ebN = self.mesh.exteriorElementBoundariesArray[ebNE]
            eN_global   = self.mesh.elementBoundaryElementsArray[ebN,0]
            ebN_element  = self.mesh.elementBoundaryLocalElementBoundariesArray[ebN,0]
        
            for i in range(self.mesh.nNodes_element):
                if i != ebN_element:
                    I = self.mesh.elementNodesArray[eN_global,i]
                    self.internalNodes -= set([I])
        self.nNodes_internal = len(self.internalNodes)
        self.internalNodesArray=np.zeros((self.nNodes_internal,),'i')
        for nI,n in enumerate(self.internalNodes):
            self.internalNodesArray[nI]=n
            
        del self.internalNodes
        self.internalNodes = None
        

        logEvent("Updating local to global mappings", 2)
        self.updateLocal2Global()
        logEvent("Building time integration object", 2)
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
        ############################################

        nd = self.coefficients.nd
        a_rowptr = self.coefficients.sdInfo[(0, 0)][0]
        a_colind = self.coefficients.sdInfo[(0, 0)][1]

        def _pack_diffusion_to_sd(a_full):
            """
            Pack dense diffusion tensor a_full into the sparse diffusion
            layout defined by (a_rowptr, a_colind).
            """
            a_arr = np.asarray(a_full, dtype='d')
            if a_arr.ndim == 0:
                a_arr = a_arr.reshape((1, 1))
            elif a_arr.ndim == 1 and a_arr.size == 1:
                a_arr = a_arr.reshape((1, 1))

            n_rows = a_rowptr.shape[0] - 1
            if a_arr.ndim != 2 or a_arr.shape[0] < n_rows:
                raise ValueError(
                    f"Unexpected aOfX shape {a_arr.shape}; expected at least ({n_rows}, {n_rows})"
                )

            a_sd = np.zeros((a_rowptr[-1],), dtype='d')
            for row in range(n_rows):
                for offset in range(a_rowptr[row], a_rowptr[row + 1]):
                    col = a_colind[offset]
                    a_sd[offset] = a_arr[row, col]
            return a_sd

        if self.coefficients.aOfX is not None:
         #########Dealing with element diffusion coefficient
         for eN in range(self.mesh.nElements_global):
            for k in range(self.nQuadraturePoints_element):
                x = self.q['x'][eN, k, :]
                a_full = self.coefficients.aOfX[0](x)
                a_val = _pack_diffusion_to_sd(a_full)

                if a_val.shape == (a_rowptr[-1],):
                    self.q[('a', 0, 0)][eN, k, :] = a_val
                else:
                    raise ValueError(f"Unexpected shape {a_val.shape} for a_val. Expected ({a_rowptr[-1]},)")
         #self.ebqe[('a',0,0)] = np.zeros((,,self.coefficients.sdInfo[(0,0)][0][-1]),'d')

         #########Dealing with boundary diffusion coefficient

         for ebNE in range(self.mesh.nExteriorElementBoundaries_global):
            for kb in range(self.nElementBoundaryQuadraturePoints_elementBoundary):
                x = self.ebqe['x'][ebNE, kb, :]
                a_full = self.coefficients.aOfX[0](x)
                a_val = _pack_diffusion_to_sd(a_full)
                if a_val.shape == (a_rowptr[-1],):
                    self.ebqe[('a', 0, 0)][ebNE, kb, :] = a_val
                else:
                    raise ValueError(f"Unexpected shape {a_val.shape} for a_val. Expected ({a_rowptr[-1]},)")

         # evaluateCoefficients builds the tensor from (Dm,alpha_L,alpha_T) and no
         # longer reads q[('a',0,0)], so carry aOfX across into Dm when the caller
         # gave aOfX alone.  Naming any of the three selects the new dispersion
         # model instead, and it wins.  Only a constant isotropic aOfX has an
         # exact (Dm,0,0) equivalent -- the dispersion form is isotropic plus a
         # rank-1 term along v, so it cannot represent a general tensor.
         if not self.coefficients.dispersion_specified:
            a_q = self.q[('a', 0, 0)]
            a0 = float(a_q.flat[0]) if a_q.size else 0.0
            if not np.allclose(a_q, a0, rtol=0.0, atol=1.0e-14):
                raise ValueError(
                    "TADR: aOfX is not constant isotropic (entries span [%r, %r]) and the "
                    "kernel builds its tensor from (Dm, alpha_L, alpha_T) only. Pass those "
                    "explicitly for this case." % (float(a_q.min()), float(a_q.max())))
            self.coefficients.Dm = a0
            logEvent("TADR: aOfX given without Dm/alpha_L/alpha_T; taking Dm=%r from it" % (a0,))




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

            # Initialize the diffusive flux boundary condition flags
            #self.ebqe[('diffusiveFlux_bc_flag', ci)] = np.zeros(self.ebqe[('diffusiveFlux_bc', ci)].shape, 'i')
    
        for ck,diffusiveFluxBoundaryConditionsDict in fbcObject.diffusiveFluxBoundaryConditionsDictDict.items():
            self.ebqe[('diffusiveFlux_bc_flag',ck,ci)] = np.zeros(self.ebqe[('diffusiveFlux_bc',ck,ci)].shape,'i')
            for t,g in diffusiveFluxBoundaryConditionsDict.items():
                self.ebqe[('diffusiveFlux_bc',ck,ci)][t[0],t[1]] = g(self.ebqe[('x')][t[0],t[1]],self.timeIntegration.t)
                self.ebqe[('diffusiveFlux_bc_flag',ck,ci)][t[0],t[1]] = 1

        
        self.numericalFlux.setDirichletValues(self.ebqe)
        if hasattr(self.numericalFlux, 'setDirichletValues'):
            self.numericalFlux.setDirichletValues(self.ebqe)
        if not hasattr(self.numericalFlux, 'isDOFBoundary'):
            self.numericalFlux.isDOFBoundary = {0: np.zeros(self.ebqe[('u', 0)].shape, 'i')}
        if not hasattr(self.numericalFlux, 'ebqe'):
            self.numericalFlux.ebqe = {('u', 0): np.zeros(self.ebqe[('u', 0)].shape, 'd')}

        # Build an explicit mask of physical exterior boundaries.
        # A face is physical if it has no "outside" element; processor
        # interfaces typically have a neighboring (ghost) element.
        self.isExteriorBoundaryPhysical = np.zeros((self.mesh.nExteriorElementBoundaries_global,), 'i')
        ebe = self.mesh.elementBoundaryElementsArray
        for ebNE in range(self.mesh.nExteriorElementBoundaries_global):
            ebN = self.mesh.exteriorElementBoundariesArray[ebNE]
            if ebe.ndim == 2:
                outside_eN = ebe[ebN, 1]
            else:
                outside_eN = ebe[ebN * 2 + 1]
            self.isExteriorBoundaryPhysical[ebNE] = 1 if outside_eN < 0 else 0
        self.elementBoundaryMaterialTypes = self.mesh.elementBoundaryMaterialTypes
        #TODO how to handle redistancing calls for calculateCoefficients,calculateElementResidual etc
        self.globalResidualDummy = None
        compKernelFlag = 0
        self.adr = cTADR_base(self.nSpace_global,
                             self.nQuadraturePoints_element,
                             self.u[0].femSpace.elementMaps.localFunctionSpace.dim,
                             self.u[0].femSpace.referenceFiniteElement.localFunctionSpace.dim,
                             self.testSpace[0].referenceFiniteElement.localFunctionSpace.dim,
                             self.nElementBoundaryQuadraturePoints_elementBoundary,
                             compKernelFlag)

        self.forceStrongConditions = bool(getattr(self.coefficients,"forceStrongConditions",False))
        if self.forceStrongConditions:
            self.dirichletConditionsForceDOF = DOFBoundaryConditions(self.u[0].femSpace, dofBoundaryConditionsSetterDict[0], weakDirichletConditions=False)
        if self.movingDomain:
            self.MOVING_DOMAIN = 1.0
        else:
            self.MOVING_DOMAIN = 0.0
        if self.mesh.nodeVelocityArray is None:
            self.mesh.nodeVelocityArray = np.zeros(self.mesh.nodeArray.shape, 'd')

        # Stuff added by mql.
        # Some ASSERTS to restrict the combination of the methods
        # STABILIZATION_TYPE 2,3,4 are the EXPLICIT edge-based schemes (SSP +
        # ExplicitLumpedMassMatrix).  Type 5 (ImplicitEV) is the IMPLICIT
        # edge-based scheme: backward-Euler + Newton, so it is exempt from the
        # SSP/Explicit-solver/LUMPED_MASS_MATRIX constraints below.
        if self.coefficients.STABILIZATION_TYPE > 1 and self.coefficients.STABILIZATION_TYPE not in (5, 6):
            assert self.timeIntegration.isSSP == True, "If STABILIZATION_TYPE>1, use RKEV timeIntegration within TADR model"
            cond = 'levelNonlinearSolver' in dir(options) and (options.levelNonlinearSolver ==
                                                               ExplicitLumpedMassMatrix or options.levelNonlinearSolver == ExplicitConsistentMassMatrixForVOF)
            assert cond, "If STABILIZATION_TYPE>1, use levelNonlinearSolver=ExplicitLumpedMassMatrix or ExplicitConsistentMassMatrixForVOF"
        if self.coefficients.STABILIZATION_TYPE == 6:
            assert isinstance(self.timeIntegration, CompatibleBackwardEuler), \
                "STABILIZATION_TYPE='Compatible' takes the flow's steps: use timeIntegration = TADR.CompatibleBackwardEuler"
            assert 'levelNonlinearSolver' not in dir(options) or \
                options.levelNonlinearSolver not in (ExplicitLumpedMassMatrix, ExplicitConsistentMassMatrixForVOF), \
                "STABILIZATION_TYPE='Compatible' is implicit: use levelNonlinearSolver = Newton"
            # Concentrations on the Dirichlet boundary are imposed strongly
            self.forceStrongConditions = True
            self.dirichletConditionsForceDOF = DOFBoundaryConditions(self.u[0].femSpace, dofBoundaryConditionsSetterDict[0],
                                                                     weakDirichletConditions=False)
            self.compatibleBudget = []
            self.carriedStorage = {}  # M^{n+1} keyed by t^{n+1}
            self._compatStepKey = None
        if self.coefficients.STABILIZATION_TYPE == 5:
            assert getattr(self.timeIntegration, 'isSSP', False) == False, \
                "If STABILIZATION_TYPE==5 (ImplicitEV), use an implicit (non-SSP) timeIntegration, e.g. BackwardEuler"
            if 'levelNonlinearSolver' in dir(options):
                assert options.levelNonlinearSolver not in (ExplicitLumpedMassMatrix,
                                                            ExplicitConsistentMassMatrixForVOF), \
                    "If STABILIZATION_TYPE==5 (ImplicitEV), use an implicit Newton solver " \
                    "(levelNonlinearSolver=Newton); ExplicitLumpedMassMatrix / " \
                    "ExplicitConsistentMassMatrixForVOF only do a single lumped update and " \
                    "leave the implicit system unsolved."
        if 'levelNonlinearSolver' in dir(options) and options.levelNonlinearSolver == ExplicitLumpedMassMatrix:
            assert self.coefficients.LUMPED_MASS_MATRIX, "If levelNonlinearSolver=ExplicitLumpedMassMatrix, use LUMPED_MASS_MATRIX=True"
        if self.coefficients.LUMPED_MASS_MATRIX == True and self.coefficients.STABILIZATION_TYPE not in (5, 6):
            cond = 'levelNonlinearSolver' in dir(options) and options.levelNonlinearSolver == ExplicitLumpedMassMatrix
            assert cond, "Use levelNonlinearSolver=ExplicitLumpedMassMatrix when the mass matrix is lumped"
        if self.coefficients.FCT is True:
            assert self.coefficients.STABILIZATION_TYPE > 1, \
                "Use FCT just with STABILIZATION_TYPE>1; i.e., edge based stabilization"
        if self.coefficients.STABILIZATION_TYPE==1:
            cond = 'levelNonlinearSolver' in dir(options) and  options.levelNonlinearSolver == TwoStageNewton
            assert cond, "If STABILIZATION_TYPE==1, use levelNonlinearSolver=TwoStageNewton"
        if self.coefficients.STABILIZATION_TYPE==1:
            self.useTwoStageNewton = True
            assert isinstance(self.timeIntegration, TimeIntegration.BackwardEuler_cfl), "If STABILIZATION_TYPE=1, use BackwardEuler_cfl"
            assert options.levelNonlinearSolver == TwoStageNewton, "If STABILIZATION_TYPE=1, use levelNonlinearSolver=TwoStageNewton"
        assert self.coefficients.ENTROPY_TYPE in [0,1], "Set ENTROPY_TYPE={0,1}"
        assert self.coefficients.STABILIZATION_TYPE in [-1,0,1,2,3,4,5,6], "Set STABILIZATION_TYPE={-1,0,1,2,3,4,5,6}"
        if self.coefficients.STABILIZATION_TYPE==4:
            assert self.coefficients.FCT==True, "If STABILIZATION_TYPE=4, use FCT=True"
            
        # mql. Allow the user to provide functions to define the velocity field
        self.hasVelocityFieldAsFunction = False
        if ('velocityFieldAsFunction') in dir(options):
            self.velocityFieldAsFunction = options.velocityFieldAsFunction
            self.hasVelocityFieldAsFunction = True

        # For edge based methods
        self.ML = None  # lumped mass matrix
        self.MC_global = None  # consistent mass matrix
        self.uLow = None
        self.dt_times_dC_minus_dL = None
        self.dLow = None
        self.min_u_bc = None
        self.max_u_bc = None
        self.quantDOFs = np.zeros(self.u[0].dof.shape, 'd')

        # bc_mask: 1.0 at free DOFs, 0.0 at Dirichlet DOFs.  Mirrors mphase_co2's
        # pattern.  The FCT step uses this to zero antidiffusive corrections at
        # Dirichlet DOFs so they stay at the low-order solution, which already
        # carries the Nitsche-weak BC contribution from the boundary kernel.
        # Together with the existing consistent-flux + penalty boundary integral
        # this gives bounded + mass-conservative Dirichlet BCs (matches the
        # mphase_co2 architecture; no forceStrongConditions row-replacement
        # needed for boundedness).
        self.bc_mask = np.ones_like(self.u[0].dof)
        if 0 in self.dirichletConditions and self.dirichletConditions[0] is not None:
            for dofN in self.dirichletConditions[0].DOFBoundaryConditionsDict.keys():
                self.bc_mask[dofN] = 0.0

        # For Taylor Galerkin methods
        self.stage = 1
        self.auxTaylorGalerkinFlag = 1        
        self.uTilde_dof = np.zeros(self.u[0].dof.shape)
        self.degree_polynomial = 1
        try:
            self.degree_polynomial = self.u[0].femSpace.order
        except:
            pass
        # Defaults for the coefficient-owned auxiliary fields calculateResidual
        # reads.  Coefficients.attachModels() sets them, but a user subclass that
        # overrides attachModels() without calling super -- the pattern used
        # throughout test/ -- never runs it, and the residual then dies on a
        # missing attribute.  Seed them here (attachModels() runs later and
        # assigns unconditionally, so it still wins).  q_v/ebqe_v are not seeded:
        # every attachModels(), overridden or not, sets those.
        c = self.coefficients
        if not hasattr(c, 'q_v_old'):
            c.q_v_old = np.zeros(self.q[('grad(u)', 0)].shape, 'd')
        if not hasattr(c, 'q_porosity'):
            c.q_porosity = np.full(self.q[('u', 0)].shape, c.porosity, 'd')
        if not hasattr(c, 'q_porosity_old'):
            c.q_porosity_old = c.q_porosity.copy()
        if not hasattr(c, 'q_rho'):
            c.q_rho = np.full(self.q[('u', 0)].shape, c.rho_f, 'd')
        if not hasattr(c, 'q_rho_old'):
            c.q_rho_old = c.q_rho.copy()
        if not hasattr(c, 'ebqe_porosity'):
            c.ebqe_porosity = np.full(self.ebqe[('u', 0)].shape, c.porosity, 'd')
        if not hasattr(c, 'ebqe_rho'):
            c.ebqe_rho = np.full(self.ebqe[('u', 0)].shape, c.rho_f, 'd')
        if not hasattr(c, 'Sn_dof'):
            c.Sn_dof = np.zeros_like(self.u[0].dof)

    # def calculateQuadrature(self):
    #     self.coefficients.initializeElementQuadrature(self.timeIntegration.t, self.q)
                
    def _refreshFluidMassFromFlow(self):
        """SCHEME B / OPTION C: take the fluid mass M = theta*rho STRAIGHT FROM
        RICHARDS instead of advecting a second copy of it here.

        Richards already carries exactly this quantity.  Its conserved variable
        is  m = rhom*thetaW  with  rhom = q_rho*exp(beta*psi)  and q_rho the
        density TADR itself pushes across in Richards.preStep -- so Richards' mn
        is theta*rho(salinity), DOF-sized, ghost-scattered, and it already
        respects every Dirichlet condition on u (rho is a function of u, so a
        pinned u pins rho, and Richards sees the pinned value through q_rho).

        What this buys over the sub-cycled fluid solve it replaces:

        * COST.  The sub-cycler needed S = dt/dt_fluid explicit sub-steps, each
          one matvec plus one ghost scatter (an MPI collective).  On the 20 yr
          Elder run S reached 100+ on 12602 of 13116 steps.  Here S = 0.
        * NO SEPARATE DIRICHLET SET.  The whole strong_bc_flag/strong_bc_u
          apparatus existed only because M was a free interior row at the source
          strip and the dispersion inside K drained it (to ~4% of theta*rho,
          which is what made the 20 yr benchmark take up 3x too little salt).
          A fluid mass read from Richards cannot drain: Richards imposes its own
          flow BCs there.
        * NO COLLECTIVE-COUNT HAZARD.  The per-rank sub-step count was an MPI
          collective trip count and deadlocked a 64-rank run when it went
          rank-local.  There is no loop left to get that wrong.

        What it COSTS, and this is the honest part: exact discrete conservation.
        The cancelled solute residual is algebraically equivalent to the
        conservative pair only if M satisfies TADR's OWN discrete fluid equation
        (same K, same d, same B^f).  Richards' m satisfies RICHARDS' discrete
        continuity equation instead, and the two operators are not the same.  The
        gap is exactly the flow/transport continuity defect that _adoptFluidMass
        used to log: on the 20 yr Elder run it grew monotonically from 1.4e-16 to
        3.6e-2.  So conservation here is NOT to round-off; it is to that defect.
        _reportSoluteMass below measures the consequence directly.

        BOUNDEDNESS is unaffected and remains unconditional: the cancelled
        residual is an M-matrix for ANY strictly positive M (the off-diagonals
        are -gamma_ij*rho_j with gamma_ij >= 0 and rho_j = M_j/theta_j > 0), and
        it is Richards' job to keep theta*rho positive.

        Returns 1 if the kernel must seed M from the EOS instead (no flow model
        -- test_tadr and any single-model deck), 0 if M was taken from the flow.
        """
        mn_flow = getattr(getattr(self.coefficients, 'vModel', None), 'mn', None)
        # COLLECTIVE.  This test picks init_M_dof, which selects a different
        # BRANCH IN THE KERNEL (seed M from the EOS vs take Richards' M), so
        # every rank has to reach the same answer.  Evaluated rank-locally -- as
        # it was -- a single non-positive entry on one rank, including one in a
        # ghost slot that rank does not own, makes that rank alone reseed M from
        # the EOS while the others keep Richards'.  That is an O(1) divergence
        # between ranks, not a round-off one, and it is silent: both branches
        # produce a plausible M.  globalMin over the per-rank verdict means the
        # fallback is taken everywhere or nowhere.
        _mn_ok_local = 1 if (mn_flow is not None
                             and np.shape(mn_flow) == np.shape(self.M_dof)
                             and np.all(np.asarray(mn_flow) > 0.0)) else 0
        if int(Comm.globalMin(_mn_ok_local)) == 1:
            # mn is theta*rho at Richards' OLD time level.  Under
            # Sequential_MinModelStep the flow model steps FIRST, so by the time
            # TADR evaluates a residual Richards has completed its step and mn
            # holds the value at the start of THIS step -- i.e. M^n, which is
            # what both the storage coefficient and the flux density want.
            self.M_dof[:] = mn_flow
            # ...and Richards' NODAL theta(psi^n), which replaces TADR's L2-projected
            # quadrature theta at a material interface.  It no longer serves to
            # recover rho = M/theta under ImplicitEV: mn is Richards' old-time mass
            # with the OLD density, theta(psi^n)*rho(c^{n-1}), and the kernel takes the flux
            # density from the EOS at the node instead.  Left at zero when there is
            # no flow model, which is how the kernel knows to keep its projection.
            self.theta_nodal[:] = self.coefficients.vModel.thetaW_dof
            if not self._M_dof_initialized:
                logEvent("TADR SCHEME B (option C): fluid mass M read from "
                         "Richards mn every step (min=%.6e max=%.6e); no fluid "
                         "sub-cycling, no separate fluid Dirichlet set"
                         % (self.M_dof.min(), self.M_dof.max()))
                self._M_dof_initialized = True
            return 0
        if not self._M_dof_initialized:
            logEvent("TADR SCHEME B (option C): no usable flow-model mn; "
                     "seeding fluid mass M from the EOS in the kernel")
            self._M_dof_initialized = True
            return 1
        return 0

    def _reportSoluteMass(self, tag):
        """Global solute mass sum_i ML_i*M_i*u_i, the quantity the conservative
        form conserves.  Under option C M no longer satisfies TADR's own fluid
        equation, so this is the direct measure of what that costs -- and it
        replaces the [TADR-CONT] defect log, which becomes vacuously zero once M
        IS Richards' m.  globalSum, because rank 0 owns only its own slice.
        """
        if getattr(self, 'ML', None) is None:
            return
        # Owned DOFs only, then one allreduce: proteus numbers the subdomain
        # with owned DOFs first and ghosts last, so summing the whole array
        # would double-count every shared node (cf. MCorr.py:710).
        no = self.mesh.nNodes_owned
        local = float(np.sum(np.asarray(self.ML)[:no]
                             * np.asarray(self.M_dof)[:no]
                             * np.asarray(self.u[0].dof)[:no]))
        total = globalSum(local)
        if self._solute_mass_0 is None:
            self._solute_mass_0 = total
        ref = self._solute_mass_0 if abs(self._solute_mass_0) > 1.0e-30 else 1.0
        # NOT a conservation error: Elder is an OPEN system (salt enters through
        # the Dirichlet source strip), so this total is SUPPOSED to grow.  Turning
        # it into a conservation measure needs the time-integrated boundary influx
        # subtracted, including what enters through the strongly-pinned rows --
        # which this does not do.  Labelled rel_change, not rel_drift, so the log
        # does not claim more than it measures.
        logEvent("[TADR-MASS] %s t=%.6e sum(ML*M*u)=%.10e rel_change=%.6e"
                 % (tag, self.timeIntegration.t, total, (total - ref)/ref))
        # Is the fluid mass actually tracking Richards?  M is the one coupled
        # field with no live diagnostic -- it is a DOF array, so it is not
        # covered by the quadrature coupling check, and it is seeded once and
        # then refreshed silently.  If theta is constant (Elder: 0.1) this range
        # must span [theta*rho_f, theta*rho_s] = [0.100, 0.120] once salt is in
        # the domain; a range pinned at exactly 0.100 means M is FROZEN at its
        # seed and the transport has lost its density feedback entirely.
        # globalMin/globalMax, because rank 0 owns a near-empty partition.
        Mo = np.asarray(self.M_dof)[:no]
        logEvent("[TADR-M] %s t=%.6e M=theta*rho (min,max,mean)=(%.6e,%.6e,%.6e)"
                 % (tag, self.timeIntegration.t,
                    Comm.globalMin(float(Mo.min())),
                    globalMax(float(Mo.max())),
                    globalSum(float(Mo.sum()))/max(1.0, globalSum(float(Mo.size)))))

    def _reportContinuityDefect(self, tag):
        """OPTION C (STAB=5) flow/transport continuity defect, per node:

            delta_i = ML_i (M^{n+1}_i - M^n_i)/dt + Phi_i + B^f_i - ML_i R_diss_i

        The cancelled solute residual is algebraically equivalent to the conservative
        pair ONLY where delta_i = 0 -- i.e. only if the M handed over satisfies TADR's
        OWN discrete fluid equation.  Under option C it satisfies RICHARDS' instead,
        and delta is exactly the gap.  Each c_i*delta_i is a solute source that no
        limiter bounds and no boundary flux explains, so this is the honest
        denominator for [TADR-MASS], which reports rel_change and not a defect.

        The kernel supplies the flux part (self.fluidResidual = Phi + B^f - ML*R_diss)
        assembled with TADR's own K, dLow and boundary loop.  The storage part needs
        M^{n+1}, which only the flow model has: Richards' `mLow` is theta(psi^{n+1})*rho
        (Richards.h, `mLow.data()[i] = m` in the DOF loop).  M_dof is still M^n here --
        option C never advances it, since _adoptFluidMass early-returns on M_low.

        RESULTS ARE STORED, not just logged, so a test can assert on them:
          self.continuityDefect  -- per-node delta (owned rows meaningful)
          self.continuityStats   -- dict: net, abs, rel, spurious, solute, t, dt
        Returns that dict, or None if the defect is not measurable.

        M^{n+1} comes from Richards' `mAccepted` (the limited mass when its FCT stood,
        mLow when the veto threw the step away), falling back to `mLow` when the flow
        model runs without FCT and mLow IS the accepted mass.  Accurate to Richards'
        invert() error on the Newton path; for an exact value read the NEXT step's mn,
        at the cost of reporting in arrears and folding in the density refresh.
        """
        if getattr(self, 'ML', None) is None:
            return None
        vModel = getattr(self.coefficients, 'vModel', None)
        # mAccepted is set by Richards.FCTStep and tracks its per-step accept/reject
        # veto; it is None when the flow model runs without FCT, and then mLow (the
        # unlimited predictor) IS the accepted mass.  Preferring mAccepted makes this
        # probe valid in both modes.
        M_new = getattr(vModel, 'mAccepted', None)
        if M_new is None:
            M_new = getattr(vModel, 'mLow', None)
        if M_new is None or np.shape(M_new) != np.shape(self.M_dof):
            return None
        dt = self.timeIntegration.dt
        if dt <= 0.0:
            return None
        # Owned DOFs only, then one allreduce -- ghosts would double-count every
        # shared node (same convention as _reportSoluteMass).
        no = self.mesh.nNodes_owned
        ML = np.asarray(self.ML)[:no]
        u = np.asarray(self.u[0].dof)[:no]
        M_old = np.asarray(self.M_dof)[:no]
        self.continuityDefect = (ML*(np.asarray(M_new)[:no] - M_old)/dt
                                 + np.asarray(self.fluidResidual)[:no])
        d = self.continuityDefect
        # ROW 2 (history): self._M_prev_new is Richards' M^{n+1} as the PREVIOUS
        # step's probe saw it, and M_old is the mn this step read back at its
        # start.  Under Sequential_MinModelStep those are the same instant, so a
        # difference is stored history rewritten with no flux to explain it --
        # theta*rho re-evaluated after TADR moved c.  delta cannot see this: it is
        # built with rho^n on both sides by construction, which is what makes it a
        # clean continuity measure and also what makes it incomplete.  Weighted
        # with the c of that instant (stashed together with M), because the
        # carried solute mass is sum ML*M*c and the jump moves M underneath it.
        # It is a JUMP, not a rate: accumulated directly, never scaled by dt.
        if (self._M_prev_new is not None
                and np.shape(self._M_prev_new) == np.shape(M_old)):
            jump = ML*(M_old - self._M_prev_new)
            jump_solute, jump_mag = (jump*self._u_prev).sum(), np.abs(jump).sum()
        else:
            jump_solute, jump_mag = 0.0, 0.0
        # ONE reduction for all six, as [TADR-BC] does: a reduction per quantity
        # is six collectives per step, and desynchronises the ranks if any of the
        # numpy above ever throws on only some of them.
        # c_i*delta_i is the spurious solute source the defect injects; compared
        # with the solute present, it is a per-step relative creation rate.
        local = np.array([d.sum(), np.abs(d).sum(), (d*u).sum(), (ML*M_old*u).sum(),
                          jump_solute, jump_mag], dtype='d')
        comm = Comm.get()
        if comm.size() > 1:
            comm.comm.tompi4py().Allreduce(local.copy(), local)
        net, mag, spurious, solute, jump_solute, jump_mag = (float(x) for x in local)
        # Cumulative, because neither error is visible per step: both are tiny and
        # one-signed, and it is the TIME-INTEGRAL that reaches percent level.
        self._cont_spurious_int += dt*spurious
        self._cont_jump_int += jump_solute
        self.continuityStats = {
            't': self.timeIntegration.t, 'dt': dt,
            'net': net, 'abs': mag, 'rel': net/max(1.0e-30, mag),
            'spurious': spurious, 'solute': solute,
            'spurious_frac': dt*spurious/max(1.0e-30, abs(solute)),
            'jump': jump_solute, 'jump_abs': jump_mag,
            'spurious_int': self._cont_spurious_int,
            'jump_int': self._cont_jump_int,
            'total_frac': ((self._cont_spurious_int + self._cont_jump_int)
                           / max(1.0e-30, abs(solute))),
        }
        logEvent("[TADR-CONT] %s t=%.6e defect net=%+.6e sum|.|=%.6e rel=%.3e "
                 "spurious_solute=%+.6e (dt*that/mass=%+.3e)"
                 % (tag, self.continuityStats['t'], net, mag,
                    self.continuityStats['rel'], spurious,
                    self.continuityStats['spurious_frac']))
        logEvent("[TADR-CONT2] %s t=%.6e history_jump=%+.6e sum|.|=%.6e | "
                 "INT: flux=%+.6e history=%+.6e total=%+.6e (%.4f%% of mass)"
                 % (tag, self.continuityStats['t'], jump_solute, jump_mag,
                    self._cont_spurious_int, self._cont_jump_int,
                    self._cont_spurious_int + self._cont_jump_int,
                    100.0*self.continuityStats['total_frac']))
        # Stash AFTER reporting: next step compares its M^n against this M^{n+1}.
        self._M_prev_new = np.array(M_new)[:no]
        self._u_prev = u.copy()
        return self.continuityStats

    def _adoptFluidMass(self):
        """SCHEME B: the fluid mass advances once per step, with the low-order
        operator, and becomes the new carried state.  Call this exactly ONCE per
        completed step -- calling it per Newton iteration would advance M every
        iteration and destroy the m/M pairing that makes u bounded.

        Also publishes the flow/transport INCOMPATIBILITY: Richards carries the
        same quantity (its conserved variable m = theta*rho), so the difference
        between TADR's advected fluid mass and Richards' is exactly the discrete
        continuity defect between the two models.  If this is small, reading M
        straight from Richards would be enough and the per-edge flux export
        (Stage 6) is unnecessary; if it is O(1), that export is mandatory.
        """
        if not np.any(self.M_low):
            return
        mLow_flow = getattr(getattr(self.coefficients, 'vModel', None), 'mLow', None)
        if (mLow_flow is not None
                and np.shape(mLow_flow) == np.shape(self.M_low)):
            mf = np.asarray(mLow_flow)
            scale = max(1.0e-30, float(np.max(np.abs(mf))))
            defect = float(np.max(np.abs(self.M_low - mf)))/scale
            logEvent("[TADR-CONT] max|M_tadr - m_richards|/scale = %.6e "
                     "(flow/transport continuity defect; rank-local)" % defect)
        self.M_dof[:] = self.M_low

    def FCTStep(self):
        if self.coefficients.STABILIZATION_TYPE == 6:
            return self.FCTStep_compatible()
        # Explicit solvers call FCTStep() unconditionally.
        # If FCT is disabled, advance with low-order solution (uLow).
        if not self.coefficients.FCT:
            if self.forceStrongConditions:
                for dofN, g in list(self.dirichletConditionsForceDOF.DOFBoundaryConditionsDict.items()):
                    self.uLow[dofN] = g(self.dirichletConditionsForceDOF.DOFBoundaryPointDict[dofN],
                                        self.timeIntegration.t)
            fromFreeToGlobal = 0
            cfemIntegrals.copyBetweenFreeUnknownsAndGlobalUnknowns(
                fromFreeToGlobal,
                self.offset[0],
                self.stride[0],
                self.dirichletConditions[0].global2freeGlobal_global_dofs,
                self.dirichletConditions[0].global2freeGlobal_free_dofs,
                self.timeIntegration.u,
                self.uLow,
            )
            # Keep model DOFs in sync with the low-order update.
            self.u[0].dof[:] = self.uLow
            self._adoptFluidMass()   # SCHEME B
            return

        rowptr, colind, MassMatrix = self.MC_global.getCSRrepresentation()
        rowptr, colind, MassMatrix = self.MC_global.getCSRrepresentation()
        limited_mass = np.zeros(self.u[0].dof.shape)
        limited_solution = np.zeros(self.u[0].dof.shape)
        numDOFs = len(rowptr) - 1
        theta_dof_out_arr = np.zeros(numDOFs, 'd')

        # theta AS THE RESIDUAL USED IT, not as it stands now.  See the capture
        # in getResidual: for STAB=5 the FCT hook runs from modelStepTaken,
        # after Coefficients.postStep has already advanced q_porosity_old to
        # theta^{n+1}.  The explicit branches (STAB 2/3/4) call FCTStep during
        # the step, before postStep, so their live copy is already theta^n and
        # the fallback is correct for them.
        _theta_fct = getattr(self, "_q_porosity_conv", None)
        if _theta_fct is None or _theta_fct.shape != self.coefficients.q_porosity_old.shape:
            _theta_fct = self.coefficients.q_porosity_old
        q_porosity_old_arr = np.ascontiguousarray(_theta_fct)
        q_rho_arr = np.ascontiguousarray(self.coefficients.q_rho)
        q_dV_arr = np.ascontiguousarray(self.q['dV_last'])
        u_l2g_arr = np.ascontiguousarray(self.u[0].femSpace.dofMap.l2g)
        u_test_ref_arr = np.ascontiguousarray(self.u[0].femSpace.psi)

        argsDict = cArgumentsDict.ArgumentsDict()
        argsDict["dt"] = self.timeIntegration.dt
        argsDict["NNZ"] = self.nnz
        argsDict["numDOFs"] = numDOFs
        argsDict["lumped_mass_matrix"] = self.ML
        argsDict["soln"] = self.u_dof_old
        argsDict["solH"] = self.timeIntegration.u
        argsDict["uLow"] = self.uLow
        argsDict["dLow"] = self.dLow
        argsDict["limited_solution"] = limited_mass
        argsDict["csrRowIndeces_DofLoops"] = rowptr
        argsDict["csrColumnOffsets_DofLoops"] = colind
        argsDict["MassMatrix"] = MassMatrix
        argsDict["dt_times_dH_minus_dL"] = self.dt_times_dC_minus_dL
        argsDict["min_u_bc"] = self.min_u_bc
        argsDict["max_u_bc"] = self.max_u_bc
        argsDict["bc_mask"] = self.bc_mask
        argsDict["LUMPED_MASS_MATRIX"] = self.coefficients.LUMPED_MASS_MATRIX
        argsDict["STABILIZATION_TYPE"] = self.coefficients.STABILIZATION_TYPE
        argsDict["q_porosity_old_fct"] = q_porosity_old_arr
        argsDict["q_rho_fct"] = q_rho_arr
        argsDict["q_dV_fct"] = q_dV_arr
        argsDict["u_l2g_fct"] = u_l2g_arr
        argsDict["u_test_ref_fct"] = u_test_ref_arr
        argsDict["theta_dof_out"] = theta_dof_out_arr
        argsDict["M_dof_fct"] = self.M_dof   # M^n
        argsDict["M_low_fct"] = self.M_low   # M^{n+1}, explicit branches only
        argsDict["theta_nodal"] = self.theta_nodal
        argsDict["nElements_global_fct"] = self.mesh.nElements_global
        argsDict["nQuadraturePoints_element_fct"] = self.nQuadraturePoints_element
        argsDict["nDOF_trial_element_fct"] = self.nDOF_trial_element[0]
        argsDict["rho_f"] = self.coefficients.rho_f
        argsDict["rho_s"] = self.coefficients.rho_s
        self.adr.FCTStep(argsDict)
        argsDict["mIn"] = limited_mass
        argsDict["uOut"] = limited_solution
        argsDict["nodal_porosity"] = theta_dof_out_arr
        argsDict["rho_nodal"] = self.rho_nodal
        self.adr.invert(argsDict)
        if self.forceStrongConditions:
            for dofN, g in list(self.dirichletConditionsForceDOF.DOFBoundaryConditionsDict.items()):
                limited_solution[dofN] = g(self.dirichletConditionsForceDOF.DOFBoundaryPointDict[dofN],
                                           self.timeIntegration.t)
        #self.timeIntegration.u[:] = limited_solution
        fromFreeToGlobal=0 #direction copying
        cfemIntegrals.copyBetweenFreeUnknownsAndGlobalUnknowns(fromFreeToGlobal,
                                                               self.offset[0],
                                                               self.stride[0],
                                                               self.dirichletConditions[0].global2freeGlobal_global_dofs,
                                                               self.dirichletConditions[0].global2freeGlobal_free_dofs,
                                                               self.timeIntegration.u,
                                                               limited_solution)
        self.u[0].dof[:] = limited_solution
        self._adoptFluidMass()   # SCHEME B

    def updateVelocityFieldAsFunction(self):
        import pdb
        X = {0: self.q[('x')][:, :, 0],
             1: self.q[('x')][:, :, 1],
             2: self.q[('x')][:, :, 2]}
        #pdb.set_trace()
        t = self.timeIntegration.t
        self.coefficients.q_v[..., 0] = self.velocityFieldAsFunction[0](X, t)
        if (self.nSpace_global == 2):
            self.coefficients.q_v[..., 1] = self.velocityFieldAsFunction[1](X, t)
        if (self.nSpace_global == 3):
            self.coefficients.q_v[..., 2] = self.velocityFieldAsFunction[2](X, t)

        # BOUNDARY
        ebqe_X = {0: self.ebqe['x'][:, :, 0],
                  1: self.ebqe['x'][:, :, 1],
                  2: self.ebqe['x'][:, :, 2]}
        self.coefficients.ebqe_v[..., 0] = self.velocityFieldAsFunction[0](ebqe_X, t)
        if (self.nSpace_global == 2):
            self.coefficients.ebqe_v[..., 1] = self.velocityFieldAsFunction[1](ebqe_X, t)
        if (self.nSpace_global == 3):
            self.coefficients.ebqe_v[..., 2] = self.velocityFieldAsFunction[2](ebqe_X, t)
            
    def calculateCoefficients(self):
        pass

    def calculateElementResidual(self):
        if self.globalResidualDummy is not None:
            self.getResidual(self.u[0].dof, self.globalResidualDummy)

    def getMassMatrix(self):
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
        self.q[('w',0)] = np.zeros((self.mesh.nElements_global,
                                    self.nQuadraturePoints_element,
                                    self.nDOF_test_element[0]),
                                   'd')
        self.q[('w*dV_m',0)] = self.q[('w',0)].copy()
        self.u[0].femSpace.getBasisValues(self.elementQuadraturePoints, self.q[('w',0)])
        cfemIntegrals.calculateWeightedShape(self.elementQuadratureWeights[('u',0)],
                                             self.q['abs(det(J))'],
                                             self.q[('w',0)],
                                             self.q[('w*dV_m',0)])
        # assume a linear mass term
        dm = np.ones(self.q[('u', 0)].shape, 'd')
        elementMassMatrix = np.zeros((self.mesh.nElements_global,
                                      self.nDOF_test_element[0],
                                      self.nDOF_trial_element[0]), 'd')
        cfemIntegrals.updateMassJacobian_weak_lowmem(dm,
                                                     self.q[('w', 0)],
                                                     self.q[('w*dV_m', 0)],
                                                     elementMassMatrix)
        self.MC_a = self.nzval.copy()
        self.MC_global = SparseMat(self.nFreeDOF_global[0],
                                   self.nFreeDOF_global[0],
                                   self.nnz,
                                   self.MC_a,
                                   self.colind,
                                   self.rowptr)
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
            self.ML[i] = self.MC_a[self.rowptr[i]:self.rowptr[i + 1]].sum()
        np.testing.assert_almost_equal(self.ML.sum(),
                                       self.mesh.volume,
                                       err_msg="Trace of lumped mass matrix should be the domain volume", verbose=True)
        
    def initVectors(self):
        self.u_dof_old = np.copy(self.u[0].dof)
        self.par_u_dof_old = None
        par_u = None
        if hasattr(self, "par_uList") and len(self.par_uList) > 0 and self.par_uList[0] is not None:
            par_u = self.par_uList[0]
        elif hasattr(self.u[0], "par_dof") and self.u[0].par_dof is not None:
            par_u = self.u[0].par_dof
        if par_u is not None:
            self.par_u_dof_old = ParVec_petsc4py(self.u_dof_old,
                                                 1,
                                                 par_u.dim_proc,
                                                 par_u.getSize(),
                                                 par_u.nghosts,
                                                 par_u.subdomain2global,
                                                 ghosts=None,
                                                 proteus2petsc_subdomain=par_u.proteus2petsc_subdomain,
                                                 petsc2proteus_subdomain=par_u.petsc2proteus_subdomain)
        rowptr, colind, MC = self.MC_global.getCSRrepresentation()
        # This is dummy. I just care about the csr structure of the sparse matrix
        self.dt_times_dC_minus_dL = np.zeros(MC.shape, 'd')
        self.uLow = np.zeros(self.u[0].dof.shape, 'd')
        self.dLow = np.zeros(MC.shape, 'd')
        # SCHEME B: the FLUID mass M = theta*rho is a state variable paired with
        # the solute mass, and the concentration is the ratio u = m/M -- a convex
        # combination for ANY velocity, including the divergent one that
        # variable-density flow necessarily produces.
        #
        # Where M comes from depends on the branch:
        #  * EXPLICIT edge-based (STAB 2/3/4): M is CARRIED and advected by the
        #    same edge operator as the solute (M_low in the kernel, adopted in
        #    _adoptFluidMass).  That costs nothing extra there -- it is one more
        #    accumulation inside a loop that already runs -- and it is exactly
        #    conservative, so those branches keep it.  Seeded once, never
        #    re-seeded: re-seeding each step would put the flow/transport
        #    continuity defect back in as a per-step projection.
        #  * ImplicitEV (STAB 5), OPTION C: M is READ FROM RICHARDS every step
        #    (_refreshFluidMassFromFlow).  Carrying it here would need the fluid
        #    equation solved implicitly too, and the sub-cycler that did that
        #    cost 100+ MPI collectives per step on Elder.  See that method for
        #    what the trade buys and what it costs.
        self.M_dof = np.zeros(self.u[0].dof.shape, 'd')
        self.M_low = np.zeros(self.u[0].dof.shape, 'd')
        # Fluid-equation flux for the continuity-defect probe; see
        # _reportContinuityDefect.  Written by the ImplicitEV branch only.
        self.fluidResidual = np.zeros(self.u[0].dof.shape, 'd')
        # NODAL density rho(c_i) at the end of the step, for the flow model's edge
        # loop (Richards.Coefficients.preStep copies it).  Written by the
        # ImplicitEV branch only: the residual at the converged iterate, then
        # invert() for the limited c when FCT is on.
        self.rho_nodal = np.zeros(self.u[0].dof.shape, 'd')
        # Richards' NODAL theta(psi^n), the partner of M_dof.  Zero means "no flow
        # model", and the kernel then keeps its own quadrature projection.
        self.theta_nodal = np.zeros(self.u[0].dof.shape, 'd')
        self._M_dof_initialized = False
        # Baseline for the [TADR-MASS] conservation diagnostic, set on first use.
        self._solute_mass_0 = None
        # Filled by _reportContinuityDefect; None until it has run successfully.
        self.continuityDefect = None
        self.continuityStats = None
        # Previous step's M^{n+1} and c^{n+1}, stashed by _reportContinuityDefect
        # so it can compare them against the M^n and c^n this step reads back --
        # same instant, so any difference is the row-2 history jump.
        self._M_prev_new = None
        self._u_prev = None
        # Running time-integrals of the two conservation errors, so the log
        # carries the cumulative number instead of needing post-processing.
        self._cont_spurious_int = 0.0
        self._cont_jump_int = 0.0

    def _logBoundarySoluteFlux(self):
        """TADR_BC_DBG=1: split the exterior solute boundary flux.

        ebqe_flux is advection + dispersion summed, which cannot distinguish a
        boundary that is delivering solute from one that is merely leaking it by
        spreading.  On the Simmons source strip that is the whole question: a
        Dirichlet u = 1 face with (v.n) ~ 0 supplies nothing however hard it is
        pinned, and the advective/dispersive split says so directly.

        Reported per time step (not per Newton iteration), area-weighted, with
        the outward normal, so positive leaves the domain.  Dirichlet faces are
        split again by the sign of the advective flux, because on a Dirichlet
        strip the inflow and outflow parts are physically different boundaries
        and their sum hides both.
        """
        if not _TADR_BC_DBG:
            return
        t = self.timeIntegration.t
        if getattr(self, '_tadr_bc_dbg_t', None) == t:
            return
        self._tadr_bc_dbg_t = t
        try:
            dS = self.ebqe['dS']
            tot = self.ebqe[('advectiveFlux', 0)]
            adv = self.ebqe[('advectiveFluxOnly', 0)]
            dif = tot - adv
            isdb = (self.numericalFlux.isDOFBoundary[0] == 1)

            # An exterior face of an overlap element lives in this rank's ebqe
            # arrays AND in its owner's, so a bare allreduce double-counts it.
            nE_owned = getattr(self.mesh, 'nElements_owned',
                               self.mesh.nElements_global)
            if nE_owned < self.mesh.nElements_global:
                ebN = self.mesh.exteriorElementBoundariesArray
                owned = (self.mesh.elementBoundaryElementsArray[ebN, 0] < nE_owned)
                keep = np.zeros(dS.shape, dtype=bool)
                keep[owned] = True
            else:
                keep = np.ones(dS.shape, dtype=bool)
            isdb = isdb & keep

            d_in = isdb & (adv < 0.0)          # solute carried INTO the domain
            d_out = isdb & (adv >= 0.0)
            nbc = keep & (~isdb)
            # ONE collective for the whole probe; see the note on the matching
            # reduction in Richards._logBoundaryFluxSplit.  A reduction per
            # quantity inside a try/except can desynchronise the ranks and
            # deadlock the job if it ever throws on one of them.
            local = np.array([
                d_in.sum(), (adv[d_in] * dS[d_in]).sum(), (dif[d_in] * dS[d_in]).sum(),
                d_out.sum(), (adv[d_out] * dS[d_out]).sum(), (dif[d_out] * dS[d_out]).sum(),
                nbc.sum(), (adv[nbc] * dS[nbc]).sum(), (dif[nbc] * dS[nbc]).sum(),
                (tot[keep] * dS[keep]).sum(),
            ], dtype='d')
            comm = Comm.get()
            if comm.size() > 1:
                comm.comm.tompi4py().Allreduce(local.copy(), local)
            logEvent(
                "[TADR-BC] t=%.6e  DIRICHLET in(%d qp): adv=%+.6e disp=%+.6e | "
                "DIRICHLET out(%d qp): adv=%+.6e disp=%+.6e | NON-DIRICHLET(%d qp): "
                "adv=%+.6e disp=%+.6e | net out=%+.6e  [kg/s, outward normal positive]"
                % (t, int(local[0]), local[1], local[2],
                   int(local[3]), local[4], local[5],
                   int(local[6]), local[7], local[8], local[9]), level=2)
        except Exception as e:
            logEvent("[TADR-BC] probe failed: %r" % (e,), level=2)

    def getResidual(self, u, r):
        import copy
        """
        Calculate the element residuals and add in to the global residual
        """
        if self.coefficients.STABILIZATION_TYPE == 6:
            return self.getResidual_compatible(u, r)

        if self.MC_global is None:
            self.getMassMatrix()
            self.initVectors()

        # ------------------------------------------------------------------
        # SCHEME B: supply the fluid mass M = theta*rho.
        #
        # Richards ALREADY carries exactly this quantity -- its conserved
        # variable is m = rhom*thetaW with rhom = q_rho*exp(beta*psi), and q_rho
        # is the density THIS model pushes across in Richards.preStep -- so it
        # is theta*rho(salinity), nodal, DOF-sized and ghost-scattered.
        #
        # ImplicitEV takes it from there EVERY step (option C).  The explicit
        # branches seed once and then advect their own copy, which is exactly
        # conservative and costs them nothing.
        #
        # Reading from Richards also sidesteps a trap: on the first residual
        # call q_porosity_old is still zeros (the copy is taken in attachModels,
        # before Richards has evaluated a residual), and an EOS seed built on
        # theta = 0 would be wrong for the WHOLE RUN.  The EOS path is the
        # fallback for decks with no flow model at all (test_tadr: porosity = 1,
        # rho_f = rho_s = 1, so it is exact there).
        # ------------------------------------------------------------------
        init_M_in_kernel = 0
        if self.coefficients.STABILIZATION_TYPE == 5 or not self._M_dof_initialized:
            init_M_in_kernel = self._refreshFluidMassFromFlow()

        # Reset some vectors for FCT
        self.min_u_bc = np.zeros(self.u[0].dof.shape, 'd') + 1E10
        self.max_u_bc = np.zeros(self.u[0].dof.shape, 'd') - 1E10
        self.dt_times_dC_minus_dL.fill(0.0)
        self.uLow.fill(0.0)
        self.dLow.fill(0.0)
        
        r.fill(0.0)
        # Load the unknowns into the finite element dof
        self.timeIntegration.calculateCoefs()
        self.timeIntegration.calculateU(u)
        self.setUnknowns(self.timeIntegration.u)
        # cek can put in logic to skip if BC's don't depend on t or u
        # cek todo: faster implementation of boundary conditions
        # Dirichlet boundary conditions
        # if hasattr(self.numericalFlux,'setDirichletValues'):
        if (self.stage!=2):
            self.numericalFlux.setDirichletValues(self.ebqe)
        # flux boundary conditions
        for t, g in list(self.fluxBoundaryConditionsObjectsDict[0].advectiveFluxBoundaryConditionsDict.items()):
            self.ebqe[('advectiveFlux_bc', 0)][t[0], t[1]] = g(self.ebqe[('x')][t[0], t[1]], self.timeIntegration.t)
            self.ebqe[('advectiveFlux_bc_flag', 0)][t[0], t[1]] = 1


        # Flux boundary conditions for diffusive terms
        for ck, diffusiveFluxBoundaryConditionsDict in self.fluxBoundaryConditionsObjectsDict[0].diffusiveFluxBoundaryConditionsDictDict.items():
            #self.ebqe[('diffusiveFlux_bc_flag', ck, 0)] = np.zeros(self.ebqe[('diffusiveFlux_bc', ck, 0)].shape, 'i')
            for t, g in diffusiveFluxBoundaryConditionsDict.items():
                self.ebqe[('diffusiveFlux_bc', ck, 0)][t[0], t[1]] = g(self.ebqe[('x')][t[0], t[1]], self.timeIntegration.t)
                self.ebqe[('diffusiveFlux_bc_flag', ck, 0)][t[0], t[1]] = 1


        if self.forceStrongConditions:
            for dofN, g in list(self.dirichletConditionsForceDOF.DOFBoundaryConditionsDict.items()):
                self.u[0].dof[dofN] = g(self.dirichletConditionsForceDOF.DOFBoundaryPointDict[dofN], self.timeIntegration.t)

        if (self.stage==2 and self.auxTaylorGalerkinFlag==1):
            self.uTilde_dof[:] = self.u[0].dof
            self.auxTaylorGalerkinFlag=0

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
        argsDict["shockCapturingDiffusion"] = float(self.shockCapturing.shockCapturingFactor)
        argsDict["sc_uref"] = self.coefficients.sc_uref
        argsDict["sc_alpha"] = self.coefficients.sc_beta
        argsDict["u_l2g"] = self.u[0].femSpace.dofMap.l2g
        argsDict["r_l2g"] = self.l2g[0]['freeGlobal']
        argsDict["elementDiameter"] = self.mesh.elementDiametersArray
        argsDict["degree_polynomial"] = float(self.degree_polynomial)
        argsDict["u_dof"] = self.u[0].dof
        argsDict["u_dof_old"] = self.u_dof_old
        argsDict["velocity"] = self.coefficients.q_v
        argsDict["velocity_old"] = self.coefficients.q_v_old
        argsDict["q_m"] = self.timeIntegration.m_tmp[0]
        argsDict["q_u"] = self.q[('u', 0)]
        argsDict["q_porosity"] = self.coefficients.q_porosity
        argsDict["q_porosity_old"] = self.coefficients.q_porosity_old
        argsDict["q_rho"] = self.coefficients.q_rho
        argsDict["q_rho_old"] = self.coefficients.q_rho_old
        ###########################################
        argsDict["q_a"] = self.q[('a',0,0)]
        argsDict["q_r"] = self.q[('r',0)]

        argsDict["ebq_a"] = self.ebqe[('a',0,0)]
        argsDict["ebq_r"] = self.ebqe[('r',0)]     

        ###########################################
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
        argsDict["elementBoundaryMaterialTypes"] = self.elementBoundaryMaterialTypes
        argsDict["isExteriorBoundaryPhysical"] = self.isExteriorBoundaryPhysical
        argsDict["elementBoundaryElementsArray"] = self.mesh.elementBoundaryElementsArray
        argsDict["elementBoundariesArray"] = self.mesh.elementBoundariesArray
        argsDict["elementBoundaryLocalElementBoundariesArray"] = self.mesh.elementBoundaryLocalElementBoundariesArray
        argsDict["ebqe_velocity_ext"] = self.coefficients.ebqe_v
        argsDict["isDOFBoundary_u"] = self.numericalFlux.isDOFBoundary[0]
        argsDict["ebqe_bc_u_ext"] = self.numericalFlux.ebqe[('u', 0)]
        argsDict["isFluxBoundary_u"] = self.ebqe[('advectiveFlux_bc_flag', 0)]
        argsDict["ebqe_bc_advectiveFlux_u_ext"] = self.ebqe[('advectiveFlux_bc',0)]
        argsDict["ebqe_bc_flux_u_ext"] = self.ebqe[('advectiveFlux_bc', 0)]
        argsDict["isDiffusiveFluxBoundary_u"] = self.ebqe[('diffusiveFlux_bc_flag', 0, 0)]
        argsDict["ebqe_bc_diffusiveFlux_u_ext"] = self.ebqe[('diffusiveFlux_bc', 0, 0)]
        argsDict["ebqe_porosity"] = self.coefficients.ebqe_porosity
        argsDict["ebqe_rho"] = self.coefficients.ebqe_rho
        


        #argsDict["ebqe_bc_flux_u_ext"] = self.ebqe[('diffusiveFlux_bc',0,0)]
        argsDict["epsFact"] = self.coefficients.epsFact
        argsDict["ebqe_u"] = self.ebqe[('u', 0)]
        argsDict["ebqe_flux"] = self.ebqe[('advectiveFlux', 0)]
        argsDict["ebqe_flux_adv"] = self.ebqe[('advectiveFluxOnly', 0)]
        argsDict["ebqe_dS_ext"] = self.ebqe['dS']
        argsDict["stage"] = self.stage
        argsDict["uTilde_dof"] = self.uTilde_dof
        argsDict["cE"] = self.coefficients.cE
        argsDict["cMax"] = self.coefficients.cMax
        argsDict["cK"] = self.coefficients.cK
        argsDict["uL"] = self.coefficients.uL
        argsDict["uR"] = self.coefficients.uR
        argsDict["numDOFs"] = len(self.rowptr) - 1
        argsDict["NNZ"] = self.nnz
        argsDict["csrRowIndeces_DofLoops"] = self.rowptr
        argsDict["csrColumnOffsets_DofLoops"] = self.colind
        argsDict["csrRowIndeces_CellLoops"] = self.csrRowIndeces[(0, 0)]
        argsDict["csrColumnOffsets_CellLoops"] = self.csrColumnOffsets[(0, 0)]
        argsDict["csrColumnOffsets_eb_CellLoops"] = self.csrColumnOffsets_eb[(0, 0)]
        argsDict["ML"] = self.ML
        argsDict["LUMPED_MASS_MATRIX"] = self.coefficients.LUMPED_MASS_MATRIX
        argsDict["STABILIZATION_TYPE"] = self.coefficients.STABILIZATION_TYPE
        argsDict["ENTROPY_TYPE"] = self.coefficients.ENTROPY_TYPE
        argsDict["uLow"] = self.uLow
        argsDict["dLow"] = self.dLow
        # SCHEME B: carried fluid mass (in) and its low-order update (out)
        argsDict["M_dof"] = self.M_dof
        argsDict["M_low"] = self.M_low
        argsDict["init_M_dof"] = init_M_in_kernel
        argsDict["fluidResidual"] = self.fluidResidual
        argsDict["rho_nodal"] = self.rho_nodal
        argsDict["theta_nodal"] = self.theta_nodal
        argsDict["dt_times_dH_minus_dL"] = self.dt_times_dC_minus_dL
        argsDict["min_u_bc"] = self.min_u_bc
        argsDict["max_u_bc"] = self.max_u_bc
        argsDict["quantDOFs"] = self.quantDOFs
        argsDict["physicalDiffusion"] = self.coefficients.physicalDiffusion
        argsDict["alpha_L"] = self.coefficients.alpha_L
        argsDict["alpha_T"] = self.coefficients.alpha_T
        argsDict["Dm"] = self.coefficients.Dm
        argsDict["dispersion_type"] = self.coefficients.dispersion_type
        argsDict["power_law_exponent"] = self.coefficients.power_law_exponent
        argsDict["velocity_exponent"] = self.coefficients.velocity_exponent
        argsDict["rho_f"] = self.coefficients.rho_f
        argsDict["rho_s"] = self.coefficients.rho_s
        argsDict["theta_s"] = self.coefficients.theta_s
        argsDict["theta_r"] = self.coefficients.theta_r
        # Stage 3 (kinetic dissolution): R_diss = k_d * S_n * S_w * (c_sat - c)
        # added to the per-DOF mass update in the C++ residual.  Sn_dof comes
        # from the flow model's u[1].dof when it's two-phase, else zeros.
        argsDict["Sn_dof"] = self.coefficients.Sn_dof
        argsDict["k_d"] = float(self.coefficients.k_d)
        argsDict["c_sat"] = float(self.coefficients.c_sat)
        argsDict["forceStrongConditions"] = int(self.forceStrongConditions)
        #argsDict["D"] = self.coefficients.DTypes
        argsDict["isDiffusiveFluxBoundary_u"] = self.ebqe[('diffusiveFlux_bc_flag',0,0)]
        argsDict["isAdvectiveFluxBoundary_u"] = self.ebqe[('advectiveFlux_bc_flag',0)]
        argsDict["ebqe_bc_advectiveFlux_u_ext"] = self.ebqe[('advectiveFlux_bc',0)]
        argsDict["ebqe_penalty_ext"] = self.ebqe['penalty']
        argsDict["eb_adjoint_sigma"] = self.numericalFlux.boundaryAdjoint_sigma
        #print("sigma",self.numericalFlux.boundaryAdjoint_sigma)
        ###################################################################################
        sdInfo = self.coefficients.sdInfo
    
        argsDict["a_rowptr"] = sdInfo[(0, 0)][0]
        argsDict["a_colind"] = sdInfo[(0, 0)][1]

        # Keep ghost values synchronized for edge-based MPI couplings.
        par_u = None
        if hasattr(self, "par_uList") and len(self.par_uList) > 0 and self.par_uList[0] is not None:
            par_u = self.par_uList[0]
        elif hasattr(self.u[0], "par_dof") and self.u[0].par_dof is not None:
            par_u = self.u[0].par_dof
        if par_u is not None:
            par_u.scatter_forward_insert()
            if self.par_u_dof_old is not None:
                self.par_u_dof_old.scatter_forward_insert()

        #argsDict["a_rowptr"] = self.coefficients.sdInfo[(0,0)][0]
        #argsDict["a_colind"] = self.coefficients.sdInfo[(0,0)][1]
        self.adr.calculateResidual(argsDict)
        self._logBoundarySoluteFlux()
        # OPTION C: no second residual pass here.  Under the sub-cycled fluid
        # solve this call was needed because the kernel assembled the fluid
        # system, Python solved it, and the iterate then had to re-see the new
        # rho.  M now comes from Richards before the kernel ever runs, so one
        # pass is all there is -- which also halves the residual cost per Newton
        # iteration.

        # --- DMP probe (env TADR_DMP_DBG=1) -------------------------------
        # Python-side on purpose: std::cout from the MPI ranks does not reach
        # the redirected run.log, but logEvent does reach ls_CCS_so.log.
        # For the explicit edge-based types with LUMPED_MASS_MATRIX=1 the
        # kernel writes the HIGH-order solution into r and the low-order one
        # into self.uLow, so this separates them definitively:
        #   uLow > 1  -> the low-order operator is not a DMP
        #   r    > 1  -> the high-order solution (expected; that is why FCT
        #                exists), and the question becomes why the limited
        #                result does not survive to the archive.
        # globalMax is essential: rank 0 owns only its own slice.
        logEvent("TADR_DMP_GATE env=%r aux=%r" % (os.environ.get("TADR_DMP_DBG"),
                 getattr(self, "auxiliaryCallCalculateResidual", "MISSING")), level=1)
        if True:
            try:
                _lo = globalMax(float(self.uLow.max()))
                _hi = globalMax(float(r.max()))
                _u0 = globalMax(float(self.u[0].dof.max()))
                _od = globalMax(float(self.u_dof_old.max()))
                # UNCONDITIONAL: log every main residual call.  A conditional
                # probe that prints nothing is indistinguishable from a probe
                # that never runs -- which is exactly how the C++ std::cout
                # version misled me.
                logEvent("TADR_DMP t=%g dt=%g max(u_dof_old)=%.8f max(uLow)=%.8f max(r=solH)=%.8f max(u[0].dof)=%.8f"
                         % (self.timeIntegration.t, self.timeIntegration.dt,
                            _od, _lo, _hi, _u0),
                         level=1)
                # theta_dof_proj is built from q_porosity_old (TADR.h:637) and
                # the group variable divides by it (w = m/fmax(theta,1e-14),
                # and inversevaluateCoefficients divides by theta*rho_f).  A
                # theta that is still zero -- q_porosity_old is a .copy() taken
                # in attachModels, when Richards' q[('theta',0)] has not been
                # evaluated yet, and it is only refreshed in postStep -- floors
                # to 1e-14 and blows the division up.  Log BOTH the alias and
                # the lagged copy: if old is 0 while the alias is 0.1, that is
                # the bug; if both are 0.1, the uLow spike is something else.
                _co = self.coefficients
                _po = np.asarray(getattr(_co, 'q_porosity_old', np.zeros(1)))
                _pn = np.asarray(getattr(_co, 'q_porosity', np.zeros(1)))
                _rh = np.asarray(getattr(_co, 'q_rho', np.zeros(1)))
                _vv = np.asarray(getattr(_co, 'q_v', np.zeros(1)))
                logEvent("TADR_DMP_COUP q_porosity_old[min,max]=[%.6e,%.6e] "
                         "q_porosity[min,max]=[%.6e,%.6e] q_rho[min,max]=[%.6e,%.6e] "
                         "max|q_v|=%.6e ML[min]=%.6e"
                         % (_po.min(), _po.max(), _pn.min(), _pn.max(),
                            _rh.min(), _rh.max(), float(np.abs(_vv).max()),
                            float(np.asarray(self.ML).min()) if getattr(self, 'ML', None) is not None else -1.0),
                         level=1)
                # ROW-SUM PROBE (env TADR_ROWSUM_DBG=1).  The kernel parks the
                # RELATIVE row sum  sum_j (T+D)_ij / sum_j |T+D|_ij  in quantDOFs.
                # The full-speed Kuzmin form assumes that is 0; where it is not,
                # the residue (sum_j (T+D)_ij)*w_i is an additive source that no
                # graph viscosity bounds.  Split by boundary vs interior:
                # min_u_bc is left at +1e10 on every node that no exterior face
                # touched, so it is an exact boundary mask.  globalMax across
                # ranks -- rank 0 owns only its own slice.
                if os.environ.get("TADR_ROWSUM_DBG"):
                    _rs = np.abs(np.asarray(self.quantDOFs))
                    _isb = np.asarray(self.min_u_bc) < 1.0e9
                    _all = globalMax(float(_rs.max()) if _rs.size else 0.0)
                    _bnd = globalMax(float(_rs[_isb].max()) if _isb.any() else 0.0)
                    _int = globalMax(float(_rs[~_isb].max()) if (~_isb).any() else 0.0)
                    logEvent("TADR_ROWSUM max|rowsum/scale| all=%.3e boundary=%.3e interior=%.3e "
                             "(nboundary=%d ninterior=%d)"
                             % (_all, _bnd, _int, int(_isb.sum()), int((~_isb).sum())),
                             level=1)
            except Exception as _e:
                logEvent("TADR_DMP probe failed: %r" % (_e,), level=1)

        if self.forceStrongConditions:
            for dofN, g in list(self.dirichletConditionsForceDOF.DOFBoundaryConditionsDict.items()):
                r[dofN] = 0

        if (self.auxiliaryCallCalculateResidual == False):
            if self.coefficients.STABILIZATION_TYPE == 5:
                # Capture the converged DOF solution WHILE u[0].dof is still valid.
                # calculateAuxiliaryQuantitiesAfterStep (where FCT runs) is called
                # from modelStepTaken AFTER solveMultilevel with NO scatter back
                # into u[0].dof, so u[0].dof/timeIntegration.u are stale (zero)
                # there.  FCT must read uLow from this saved copy, not u[0].dof.
                if (getattr(self, "_u_dof_conv", None) is None
                        or self._u_dof_conv.shape != self.u[0].dof.shape):
                    self._u_dof_conv = np.zeros_like(self.u[0].dof)
                self._u_dof_conv[:] = self.u[0].dof
                # Capture the porosity THIS residual projected theta_dof_proj
                # from (TADR.h:729) for the same reason.  Coefficients.postStep
                # advances q_porosity_old to theta^{n+1} BEFORE modelStepTaken
                # runs the FCT hook, so FCTStep would rebuild its own theta_dof
                # (TADR.h:3171) from a donor that has already moved on -- and
                # then reference uLow to a DIFFERENT mass than the residual
                # produced it with, which is exactly the consistency FCT's
                # boundedness argument rests on.  Inert while theta is constant
                # (the saturated SHD/SMD cases, theta = 0.36 everywhere); with a
                # moving water table -- UHD, UMD -- theta spans 0.05 to 0.36 and
                # the two disagree by the full step change in theta.
                _po = self.coefficients.q_porosity_old
                if (getattr(self, "_q_porosity_conv", None) is None
                        or self._q_porosity_conv.shape != _po.shape):
                    self._q_porosity_conv = np.zeros_like(_po)
                self._q_porosity_conv[:] = _po
                # ImplicitEV is solved implicitly by Newton (which already logs
                # per-iteration norm(r)); the CFL is NOT the time-step limiter
                # here, so report the residual norm like the implicit flow model
                # instead of the misleading CFL=0 lines.
                res_inf = globalMax(np.abs(r).max())
                logEvent("...   Current dt = " + str(self.timeIntegration.dt), level=2)
                logEvent("...   TADR ImplicitEV max|residual| = " + str(res_inf), level=2)
            else:
                edge_based_cflMax = globalMax(self.edge_based_cfl.max()) * self.timeIntegration.dt
                cell_based_cflMax = globalMax(self.q[('cfl', 0)].max()) * self.timeIntegration.dt
                logEvent("...   Current dt = " + str(self.timeIntegration.dt), level=4)
                logEvent("...   Maximum Cell Based CFL = " + str(cell_based_cflMax), level=2)
                logEvent("...   Maximum Edge Based CFL = " + str(edge_based_cflMax), level=2)

        if self.stabilization:
            self.stabilization.accumulateSubgridMassHistory(self.q)
        logEvent("Global residual", level=9, data=r)
        if self.globalResidualDummy is None:
            self.globalResidualDummy = np.zeros(r.shape, 'd')

        #Debugging statements
        # print("Mesh details:")
        # print(f"Number of elements: {self.mesh.nElements_global}")
        # print(f"Number of exterior element boundaries: {self.mesh.nExteriorElementBoundaries_global}")

        # #For each boundary condition setting
        # for ebNE in range(self.mesh.nExteriorElementBoundaries_global):
        #     for kb in range(self.nElementBoundaryQuadraturePoints_elementBoundary):
        #         if self.ebqe['x'][ebNE, kb, 0] == 0:
        #             #self.ebqe[('advectiveFlux_bc_flag', 0)][ebNE, kb] = 4.0
        #             #self.ebqe[('u', 0)][ebNE, kb] += 1.0
        #             print(f"Boundary {ebNE}, point {kb}:")
        #             print(f"  Coordinates: {self.ebqe['x'][ebNE, kb, :]}")
        #             print(f"  u: {self.ebqe[('u', 0)][ebNE, kb]}")
        #             print(f"  grad(u): {self.ebqe[('grad(u)', 0)][ebNE, kb, :]}")
        #             print(f"  advective flux bc flag: {self.ebqe[('advectiveFlux_bc_flag', 0)][ebNE, kb]}")
        #             print(f"  advective flux bc: {self.ebqe[('advectiveFlux_bc', 0)][ebNE, kb]}")


    def getJacobian(self, jacobian):
        if self.coefficients.STABILIZATION_TYPE == 6:
            return self.getJacobian_compatible(jacobian)
        cfemIntegrals.zeroJacobian_CSR(self.nNonzerosInJacobian,
                                       jacobian)
        argsDict = cArgumentsDict.ArgumentsDict()
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
        argsDict["shockCapturingDiffusion"] = float(self.shockCapturing.shockCapturingFactor)
        argsDict["u_l2g"] = self.u[0].femSpace.dofMap.l2g
        argsDict["r_l2g"] = self.l2g[0]['freeGlobal']
        argsDict["elementDiameter"] = self.mesh.elementDiametersArray
        argsDict["u_dof"] = self.u[0].dof
        argsDict["velocity"] = self.coefficients.q_v
        argsDict["q_porosity"] = self.coefficients.q_porosity
        argsDict["q_rho"] = self.coefficients.q_rho
        argsDict["q_m_betaBDF"] = self.timeIntegration.beta_bdf[0]
        argsDict["cfl"] = self.q[('cfl', 0)]
        argsDict["q_numDiff_u_last"] = self.shockCapturing.numDiff_last[0]
        argsDict["csrRowIndeces_u_u"] = self.csrRowIndeces[(0, 0)]
        argsDict["csrColumnOffsets_u_u"] = self.csrColumnOffsets[(0, 0)]
        argsDict["globalJacobian"] = jacobian.getCSRrepresentation()[2]
        argsDict["nExteriorElementBoundaries_global"] = self.mesh.nExteriorElementBoundaries_global
        argsDict["exteriorElementBoundariesArray"] = self.mesh.exteriorElementBoundariesArray
        argsDict["elementBoundaryMaterialTypes"] = self.elementBoundaryMaterialTypes
        argsDict["isExteriorBoundaryPhysical"] = self.isExteriorBoundaryPhysical
        argsDict["elementBoundaryElementsArray"] = self.mesh.elementBoundaryElementsArray
        argsDict["elementBoundaryLocalElementBoundariesArray"] = self.mesh.elementBoundaryLocalElementBoundariesArray
        argsDict["ebqe_velocity_ext"] = self.coefficients.ebqe_v
        argsDict["isDOFBoundary_u"] = self.numericalFlux.isDOFBoundary[0]
        argsDict["ebqe_bc_u_ext"] = self.numericalFlux.ebqe[('u', 0)]
        argsDict["isFluxBoundary_u"] = self.ebqe[('advectiveFlux_bc_flag', 0)]
        argsDict["ebqe_bc_flux_u_ext"] = self.ebqe[('advectiveFlux_bc', 0)]
        argsDict["ebqe_porosity"] = self.coefficients.ebqe_porosity
        argsDict["ebqe_rho"] = self.coefficients.ebqe_rho

        argsDict["isDiffusiveFluxBoundary_u"] = self.ebqe[('diffusiveFlux_bc_flag', 0, 0)]
        argsDict["ebqe_bc_diffusiveFlux_u_ext"] = self.ebqe[('diffusiveFlux_bc', 0, 0)]
        

        argsDict["csrColumnOffsets_eb_u_u"] = self.csrColumnOffsets_eb[(0, 0)]
        argsDict["STABILIZATION_TYPE"] = self.coefficients.STABILIZATION_TYPE
        argsDict["physicalDiffusion"] = self.coefficients.physicalDiffusion   
        argsDict["alpha_L"] = self.coefficients.alpha_L
        argsDict["alpha_T"] = self.coefficients.alpha_T
        argsDict["Dm"] = self.coefficients.Dm
        argsDict["dispersion_type"] = self.coefficients.dispersion_type
        argsDict["power_law_exponent"] = self.coefficients.power_law_exponent
        argsDict["velocity_exponent"] = self.coefficients.velocity_exponent
        argsDict["rho_f"] = self.coefficients.rho_f
        argsDict["rho_s"] = self.coefficients.rho_s
        argsDict["theta_s"] = self.coefficients.theta_s
        argsDict["theta_r"] = self.coefficients.theta_r
        # Stage 3 (kinetic dissolution): R_diss = k_d * S_n * S_w * (c_sat - c)
        # added to the per-DOF mass update in the C++ residual.  Sn_dof comes
        # from the flow model's u[1].dof when it's two-phase, else zeros.
        argsDict["Sn_dof"] = self.coefficients.Sn_dof
        argsDict["k_d"] = float(self.coefficients.k_d)
        argsDict["c_sat"] = float(self.coefficients.c_sat)
        # Extra keys consumed only by the STABILIZATION_TYPE==ImplicitEV (5)
        # branch of calculateJacobian, which assembles the implicit edge-based
        # Jacobian using the DOF-loop CSR, lumped mass, time step and the
        # low-order graph viscosity dLow the residual stored.
        argsDict["dt"] = self.timeIntegration.dt
        argsDict["numDOFs"] = len(self.rowptr) - 1
        argsDict["ML"] = self.ML
        argsDict["dLow"] = self.dLow
        # SCHEME B: the ImplicitEV Jacobian differentiates the cancelled form,
        # whose coefficients are gamma_ij*rho^n_j with rho^n = M_dof/theta.
        argsDict["M_dof"] = self.M_dof
        argsDict["csrRowIndeces_DofLoops"] = self.rowptr
        argsDict["csrColumnOffsets_DofLoops"] = self.colind
        argsDict["forceStrongConditions"] = int(self.forceStrongConditions)
        argsDict["ebq_a"] = self.ebqe[('a',0,0)]
        #argsDict["D"] = self.coefficients.DTypes

        sdInfo = self.coefficients.sdInfo
    
        argsDict["a_rowptr"] = sdInfo[(0, 0)][0]
        argsDict["a_colind"] = sdInfo[(0, 0)][1]
        argsDict["q_a"] = self.q[('a',0,0)]
        argsDict["eb_adjoint_sigma"] = self.numericalFlux.boundaryAdjoint_sigma
        argsDict["ebqe_penalty_ext"] = self.ebqe['penalty']

        #argsDict["a_rowptr"] = self.coefficients.sdInfo[(0,0)][0]
        #argsDict["a_colind"] = self.coefficients.sdInfo[(0,0)][1]

        self.adr.calculateJacobian(argsDict)

        # Load the Dirichlet conditions directly into residual
        if self.forceStrongConditions:
            scaling = 1.0  # probably want to add some scaling to match non-dirichlet diagonals in linear system
            for dofN in list(self.dirichletConditionsForceDOF.DOFBoundaryConditionsDict.keys()):
                global_dofN = dofN
                for i in range(self.rowptr[global_dofN],
                               self.rowptr[global_dofN + 1]):
                    if (self.colind[i] == global_dofN):
                        self.nzval[i] = scaling
                    else:
                        self.nzval[i] = 0.0
        logEvent("Jacobian ", level=10, data=jacobian)
        return jacobian

    def calculateElementQuadrature(self):
        """
        Calculate the physical location and weights of the quadrature rules
        and the shape information at the quadrature points.

        This function should be called only when the mesh changes.
        """
        self.u[0].femSpace.elementMaps.getValues(self.elementQuadraturePoints,
                                                 self.q['x'])
        self.u[0].femSpace.elementMaps.getBasisValuesRef(
            self.elementQuadraturePoints)
        self.u[0].femSpace.elementMaps.getBasisGradientValuesRef(
            self.elementQuadraturePoints)
        self.u[0].femSpace.getBasisValuesRef(self.elementQuadraturePoints)
        self.u[0].femSpace.getBasisGradientValuesRef(self.elementQuadraturePoints)
        #self.coefficients.initializeElementQuadrature(self.timeIntegration.t,
        #                                              self.q)
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
        """
        Calculate the physical location and weights of the quadrature rules
        and the shape information at the quadrature points on global element boundaries.

        This function should be called only when the mesh changes.
        """
        #
        # get physical locations of element boundary quadrature points
        #
        # assume all components live on the same mesh
        self.u[0].femSpace.elementMaps.getBasisValuesTraceRef(
            self.elementBoundaryQuadraturePoints)
        self.u[0].femSpace.elementMaps.getBasisGradientValuesTraceRef(
            self.elementBoundaryQuadraturePoints)
        self.u[0].femSpace.getBasisValuesTraceRef(
            self.elementBoundaryQuadraturePoints)
        self.u[0].femSpace.getBasisGradientValuesTraceRef(
            self.elementBoundaryQuadraturePoints)
        self.u[0].femSpace.elementMaps.getValuesGlobalExteriorTrace(
            self.elementBoundaryQuadraturePoints, self.ebqe['x'])
        self.fluxBoundaryConditionsObjectsDict = dict([(cj, FluxBoundaryConditions(self.mesh,
                                          self.nElementBoundaryQuadraturePoints_elementBoundary,
                                          self.ebqe[('x')],
                                          self.advectiveFluxBoundaryConditionsSetterDict[cj],
                                          self.diffusiveFluxBoundaryConditionsSetterDictDict[cj]))
              for cj in list(self.advectiveFluxBoundaryConditionsSetterDict.keys())])
        self.coefficients.initializeGlobalExteriorElementBoundaryQuadrature(
            self.timeIntegration.t, self.ebqe)

    def estimate_mt(self):
        pass

    def calculateSolutionAtQuadrature(self):
        pass

    def prepareCompatibleStep(self):
        """Per-step data of STABILIZATION_TYPE='Compatible' from the flow model's export.

        Fhat_ij = E_ij, the carried storage M^n -> M^{n+1} = M^n + (m^{n+1} - m^n) (M^0 = m^0),
        B_i = M_L,i (M^{n+1}_i - M^n_i)/dt - sum_j Fhat_ij, and the dispersion velocity
        v_h = q_h / rho^F with q_h reconstructed from the spatial exchanges only.
        """
        ti = self.timeIntegration
        key = (ti.tLast, ti.t)
        if self._compatStepKey == key:
            return self._compat
        flow = self.coefficients.vModel
        ex = flow.transportExport
        tol = 1.0e-12 * max(1.0, abs(ti.t))
        assert ex is not None and abs(ex['t0'] - ti.tLast) <= tol and abs(ex['t1'] - ti.t) <= tol, \
            "STABILIZATION_TYPE='Compatible' takes exactly one transport step per flow step " \
            "(flow step [%s, %s], transport step [%s, %s])" % (None if ex is None else ex['t0'],
                                                              None if ex is None else ex['t1'], ti.tLast, ti.t)
        if self._compatStepKey is None:
            if self.comm.size() > 1:
                raise NotImplementedError("STABILIZATION_TYPE='Compatible' is serial for now: the edge exchanges, "
                                          "the dispersion matrix and the limiter are not ghost-synchronized")
            assert (np.array_equal(ex['rowptr'], self.rowptr) and np.array_equal(ex['colind'], self.colind)), \
                "STABILIZATION_TYPE='Compatible' needs the flow and transport models on the same DOFs and sparsity"
            assert self.nFreeDOF_global[0] == self.u[0].dof.shape[0], \
                "STABILIZATION_TYPE='Compatible' imposes Dirichlet data strongly; use a weak numerical flux " \
                "(e.g. TADR.NumericalFlux) so every DOF is free"
            self._exportRows = np.repeat(np.arange(len(self.rowptr) - 1), np.diff(self.rowptr))
            nnz = len(self.colind)
            self._edgeWeight = np.zeros(nnz, 'd')
            self._edgeCount = np.zeros(nnz, 'd')
            self._dDispersion = np.zeros(nnz, 'd')
            self._FluxCorrectionMatrix = np.zeros(nnz, 'd')
            self._limitedFluxMatrix = np.zeros(nnz, 'd')
            self._Rpos = np.zeros(self.u[0].dof.shape, 'd')
            self._Rneg = np.zeros(self.u[0].dof.shape, 'd')
            self._element_q = np.zeros((self.mesh.nElements_global, self.nSpace_global), 'd')
            self._q_velocity_dispersion = np.zeros(self.q[('u', 0)].shape + (self.nSpace_global,), 'd')
            self._cConverged = np.zeros(self.u[0].dof.shape, 'd')
        dt = ti.dt
        Mn = None
        for tk, Mk in self.carriedStorage.items():
            if abs(tk - ti.tLast) <= 1.0e-12 * max(1.0, abs(ti.tLast)):
                Mn = Mk
        if Mn is None:
            Mn = ex['mOld'].copy()  # M^0 = m^0
        Mnp1 = Mn + (ex['mNew'] - ex['mOld'])
        assert Mnp1.min() > 0.0, "STABILIZATION_TYPE='Compatible': nonpositive carried storage %g" % Mnp1.min()
        Fhat = ex['E']
        rowSum = np.zeros(Mn.shape)
        np.add.at(rowSum, self._exportRows, Fhat)
        B = self.ML * (Mnp1 - Mn) / dt - rowSum
        # strong Dirichlet data at t^{n+1}
        bc_mask = np.ones(self.u[0].dof.shape, 'd')
        cD = np.zeros(self.u[0].dof.shape, 'd')
        for dofN, g in self.dirichletConditionsForceDOF.DOFBoundaryConditionsDict.items():
            bc_mask[dofN] = 0.0
            cD[dofN] = g(self.dirichletConditionsForceDOF.DOFBoundaryPointDict[dofN], ti.t)
        # Inflow through a node without a concentration condition carries the local concentration,
        # i.e. no prescribed c_in there; Binflow/cin hold the nodes that have one (none at present:
        # the concentration boundary contains the inflow boundary and is imposed strongly).
        Binflow = np.zeros(Mn.shape)
        cin = np.zeros(Mn.shape)
        # dispersion velocity from the spatial exchanges
        argsDict = cArgumentsDict.ArgumentsDict()
        argsDict["nElements_global"] = self.mesh.nElements_global
        argsDict["mesh_trial_ref"] = self.u[0].femSpace.elementMaps.psi
        argsDict["mesh_grad_trial_ref"] = self.u[0].femSpace.elementMaps.grad_psi
        argsDict["mesh_dof"] = self.mesh.nodeArray
        argsDict["mesh_l2g"] = self.mesh.elementNodesArray
        argsDict["dV_ref"] = self.elementQuadratureWeights[('u', 0)]
        argsDict["u_grad_trial_ref"] = self.u[0].femSpace.grad_psi
        argsDict["csrRowIndeces_CellLoops"] = self.csrRowIndeces[(0, 0)]
        argsDict["csrColumnOffsets_CellLoops"] = self.csrColumnOffsets[(0, 0)]
        argsDict["Espatial"] = ex['Espatial']
        argsDict["edgeWeight"] = self._edgeWeight
        argsDict["edgeCount"] = self._edgeCount
        argsDict["element_q"] = self._element_q
        self.adr.reconstructFlux_compatible(argsDict)
        rho_F = flow.q['rho']
        self._q_velocity_dispersion[:] = self._element_q[:, None, :] / rho_F[:, :, None]
        inflowFree = float((np.maximum(B, 0.0) * bc_mask).sum())
        self._compat = {'Mn': Mn, 'Mnp1': Mnp1, 'Fhat': Fhat, 'B': B, 'Binflow': Binflow, 'cin': cin,
                        'bc_mask': bc_mask, 'cD': cD, 'dt': dt, 'export': ex, 'inflowFree': inflowFree}
        self._compatStepKey = key
        return self._compat

    def _compatibleArgs(self, r, computeJacobian, jacobianValues):
        d = self.prepareCompatibleStep()
        argsDict = cArgumentsDict.ArgumentsDict()
        argsDict["dt"] = d['dt']
        argsDict["numDOFs"] = len(self.rowptr) - 1
        argsDict["nElements_global"] = self.mesh.nElements_global
        argsDict["mesh_trial_ref"] = self.u[0].femSpace.elementMaps.psi
        argsDict["mesh_grad_trial_ref"] = self.u[0].femSpace.elementMaps.grad_psi
        argsDict["mesh_dof"] = self.mesh.nodeArray
        argsDict["mesh_l2g"] = self.mesh.elementNodesArray
        argsDict["dV_ref"] = self.elementQuadratureWeights[('u', 0)]
        argsDict["u_trial_ref"] = self.u[0].femSpace.psi
        argsDict["u_grad_trial_ref"] = self.u[0].femSpace.grad_psi
        argsDict["u_l2g"] = self.u[0].femSpace.dofMap.l2g
        argsDict["r_l2g"] = self.l2g[0]['freeGlobal']
        argsDict["a_rowptr"] = self.coefficients.sdInfo[(0, 0)][0]
        argsDict["a_colind"] = self.coefficients.sdInfo[(0, 0)][1]
        argsDict["alpha_L"] = float(self.coefficients.alpha_L)
        argsDict["alpha_T"] = float(self.coefficients.alpha_T)
        argsDict["Dm"] = float(self.coefficients.Dm)
        argsDict["rho_f"] = self.coefficients.rho_f
        argsDict["rho_s"] = self.coefficients.rho_s
        argsDict["q_porosity"] = self.coefficients.vModel.q[('theta', 0)]
        argsDict["q_velocity_dispersion"] = self._q_velocity_dispersion
        argsDict["u_dof"] = self.u[0].dof
        argsDict["u_dof_old"] = self.u_dof_old
        argsDict["csrRowIndeces_DofLoops"] = self.rowptr
        argsDict["csrColumnOffsets_DofLoops"] = self.colind
        argsDict["csrRowIndeces_CellLoops"] = self.csrRowIndeces[(0, 0)]
        argsDict["csrColumnOffsets_CellLoops"] = self.csrColumnOffsets[(0, 0)]
        argsDict["ML"] = self.ML
        argsDict["Fhat"] = d['Fhat']
        argsDict["Mn"] = d['Mn']
        argsDict["B"] = d['Binflow']
        argsDict["cin"] = d['cin']
        argsDict["bc_mask"] = d['bc_mask']
        argsDict["cD"] = d['cD']
        argsDict["dDispersion"] = self._dDispersion
        argsDict["globalResidual"] = r
        argsDict["computeJacobian"] = int(computeJacobian)
        argsDict["globalJacobian"] = jacobianValues
        return argsDict

    def getResidual_compatible(self, u, r):
        if self.MC_global is None:
            self.getMassMatrix()
            self.initVectors()
        r.fill(0.0)
        self.timeIntegration.calculateCoefs()
        self.timeIntegration.calculateU(u)
        self.setUnknowns(self.timeIntegration.u)
        if self.coefficients.vModel.transportExport is None:
            # NumericalSolution evaluates every residual once before the first step; there is no
            # flow step to be compatible with yet
            return
        self.adr.calculateResidual_compatible(self._compatibleArgs(r, False, self.nzval))
        if not self.auxiliaryCallCalculateResidual:
            self._cConverged[:] = self.u[0].dof
            self._compatResidual = r.copy()
        logEvent("Global residual", level=9, data=r)
        if self.globalResidualDummy is None:
            self.globalResidualDummy = np.zeros(r.shape, 'd')

    def getJacobian_compatible(self, jacobian):
        cfemIntegrals.zeroJacobian_CSR(self.nNonzerosInJacobian, jacobian)
        r = np.zeros(self.u[0].dof.shape, 'd')
        self.adr.calculateResidual_compatible(self._compatibleArgs(r, True, jacobian.getCSRrepresentation()[2]))
        return jacobian

    def FCTStep_compatible(self):
        """Zalesak limiting of S = M c toward the high-order target (Algorithm 3); called by Newton
        after the low-order solve has converged, with c^L in u[0].dof."""
        d = self.prepareCompatibleStep()
        cL = self.u[0].dof.copy()
        c_new = np.zeros_like(cL)
        rowptr, colind, MassMatrix = self.MC_global.getCSRrepresentation()
        argsDict = cArgumentsDict.ArgumentsDict()
        argsDict["dt"] = d['dt']
        argsDict["numDOFs"] = len(self.rowptr) - 1
        argsDict["csrRowIndeces_DofLoops"] = self.rowptr
        argsDict["csrColumnOffsets_DofLoops"] = self.colind
        argsDict["ML"] = self.ML
        argsDict["MC"] = MassMatrix
        argsDict["Mn"] = d['Mn']
        argsDict["Mnp1"] = d['Mnp1']
        argsDict["cn"] = self.u_dof_old
        argsDict["cL"] = cL
        argsDict["Fhat"] = d['Fhat']
        argsDict["dDispersion"] = self._dDispersion
        argsDict["bc_mask"] = d['bc_mask']
        argsDict["Rpos"] = self._Rpos
        argsDict["Rneg"] = self._Rneg
        argsDict["FluxCorrectionMatrix"] = self._FluxCorrectionMatrix
        argsDict["limitedFluxMatrix"] = self._limitedFluxMatrix
        argsDict["c_new"] = c_new
        self.adr.FCTStep_compatible(argsDict)
        d['cL'] = cL
        d['c1'] = c_new.copy()
        d['fctApplied'] = True
        self.u[0].dof[:] = c_new
        self.timeIntegration.u[:] = c_new

    def compatibleStepBudget(self):
        """Fluid and solute budgets of the step just taken (STABILIZATION_TYPE='Compatible').

        fluid:  sum_i M_L,i (m^{n+1}_i - m^n_i) - dt sum_i (sum_j E~_ij + b_i) = dt sum_i r_i, the
                flow solver residual (free rows), from the flow model's export.
        solute: sum_free M_L,i (S^{n+1}_i - S^n_i) minus dt sum_free B^c_i, minus the exchange
                with Dirichlet nodes, dt sum F^L_ij(c^L) + alpha_ij f_ij (i free, j Dirichlet);
                dt times the sum of the transport residual at c^L, up to round-off.
        """
        d = self._compat
        ex = d['export']
        dt = d['dt']
        free = d['bc_mask'] == 1.0
        cn = self.u_dof_old
        cL = d.get('cL', self._cConverged)
        c1 = d.get('c1', self._cConverged)
        Mn, Mnp1, F = d['Mn'], d['Mnp1'], d['Fhat']
        rows, cols = self._exportRows, self.colind
        dD = self._dDispersion
        nu = np.maximum(0.0, 0.5 * np.abs(F) + dD)
        FL = 0.5 * F * (cL[rows] + cL[cols]) - dD * (cL[cols] - cL[rows]) + nu * (cL[cols] - cL[rows])
        FL[rows == cols] = 0.0
        toDirichlet = free[rows] & ~free[cols]
        Bc = d['Binflow'] * d['cin'] + (d['B'] - d['Binflow']) * cL
        dS = float((self.ML * (Mnp1 * c1 - Mn * cn))[free].sum())
        exchange = dt * float(Bc[free].sum()) + dt * float(FL[toDirichlet].sum())
        if d.get('fctApplied', False):
            exchange += float(self._limitedFluxMatrix[toDirichlet].sum())
        solute = dS - exchange
        fluid = dt * float(ex['fluidResidual'].sum())
        rec = {'t': self.timeIntegration.t, 'dt': dt,
               'fluid_budget_error': fluid,
               'fluid_mass': float((self.ML * Mnp1).sum()),
               'flow_residual_l1': float(np.abs(ex['fluidResidual']).sum()),
               'solute_budget_error': solute,
               'solute_mass': float((self.ML * Mnp1 * c1).sum()),
               'transport_residual_l1': float(np.abs(self._compatResidual[free]).sum()),
               'carried_minus_flow_mass': float(np.abs(Mnp1 - ex['mNew']).max()),
               'inflow_without_concentration': d['inflowFree'],
               'c_min': float(c1.min()), 'c_max': float(c1.max())}
        self.compatibleBudget.append(rec)
        logEvent("TADR compatible t=%12.5e: fluid budget %12.5e (flow |r|_1 dt %12.5e), solute budget %12.5e "
                 "(transport |r|_1 dt %12.5e), c in [%g, %g]"
                 % (rec['t'], fluid, dt * rec['flow_residual_l1'], solute, dt * rec['transport_residual_l1'],
                    rec['c_min'], rec['c_max']), level=2)
        return rec

    def calculateAuxiliaryQuantitiesAfterStep(self):
        if self.coefficients.STABILIZATION_TYPE == 6:
            self.compatibleStepBudget()
            t1 = self.timeIntegration.t
            self.carriedStorage = {tk: Mk for tk, Mk in self.carriedStorage.items() if tk < t1}
            if len(self.carriedStorage) > 2:
                for tk in sorted(self.carriedStorage)[:-2]:
                    del self.carriedStorage[tk]
            self.carriedStorage[t1] = self._compat['Mnp1'].copy()
            self.coefficients.syncCompatibleDensity()
            return
        # STAB=5 (ImplicitEV) is solved implicitly by Newton, so the explicit
        # solver never calls FCTStep().  When FCT is on, apply it here (this
        # hook is called by SplitOperator after the model's solve): the
        # converged Newton solution IS the implicit, CFL-free LOW-ORDER
        # solution uLow, and FCTStep adds the Zalesak-limited symmetric
        # antidiffusion (Kuzmin branch) to recover a bounded high-order
        # solution.  The residual stored the symmetric dLow and the min/max_u_bc
        # bounds at the converged iterate, so FCTStep is consistent.
        if self.coefficients.STABILIZATION_TYPE == 5 and self.coefficients.FCT:
            # This hook (from modelStepTaken) runs AFTER solveMultilevel with no
            # scatter back into u[0].dof for the FCT limiter, so the DOF interior
            # can be stale here.  Restore the converged field captured at the end
            # of getResidual, then feed it to FCTStep as uLow.  Without this the
            # limiter operates on a stale field and freezes the solution.
            _conv = getattr(self, "_u_dof_conv", None)
            if _conv is not None:
                self.u[0].dof[:] = _conv
            self.uLow[:] = self.u[0].dof
            # In-situ FCT sharpness diagnostic (set env TADR_FCT_DBG=1).  sum(u^2)
            # rises when mass concentrates (front sharpens) and falls when it
            # spreads (diffuses), at fixed mass.  Compares the low-order field
            # (uLow) with the FCT-limited result to settle, on the REAL 2D run,
            # whether the STAB=5 FCTStep sharpens or diffuses.  MAX/SUM-reduced
            # across ranks (rank 0 owns a near-empty slice in parallel).
            import os as _os
            _dbg = bool(_os.environ.get("TADR_FCT_DBG"))
            if _dbg:
                _s_low = float(np.sum(self.uLow * self.uLow))
            self.FCTStep()
            # --------------------------------------------------------------
            # HAND THE LIMITED FIELD TO THE FLOW MODEL.
            #
            # q_rho/ebqe_rho are written by the RESIDUAL kernel (TADR.h, in the
            # element and boundary quadrature loops) from whatever iterate it
            # was called on, so when the step ends they hold rho(uLow) -- the
            # UNLIMITED low-order solution.  FCTStep runs after the last
            # residual call and rewrites only u[0].dof, and
            # Richards.Coefficients.preStep copies q_rho/ebqe_rho straight into
            # its own q['rho']/ebqe['rho'] at the top of the next step.
            #
            # Without this refresh the limiter sharpens the concentration while
            # the BUOYANCY the flow model integrates stays the diffused field,
            # so FCT cannot change fingering at all and would read as "the
            # limiter does nothing" when in fact it was never coupled.
            #
            # rho is pointwise in u (rho = rho_f(1 + eps_rho*u)), so this is
            # just interpolating the limited DOFs onto the two quadrature sets
            # the kernel uses -- one einsum each, once per step.
            #
            # FIXED (2026-09-22): q_porosity_old is advanced to theta^{n+1} by
            # Coefficients.postStep, which runs BEFORE this hook, so FCTStep's
            # theta projection used to be one step ahead of the residual that
            # produced uLow.  getResidual now snapshots the porosity it
            # projected into self._q_porosity_conv and FCTStep reads that.
            # Was inert for the saturated cases (theta constant) and wrong for
            # UHD/UMD.
            # --------------------------------------------------------------
            _rho_f = self.coefficients.rho_f
            _eps_rho = (self.coefficients.rho_s - _rho_f)/_rho_f
            _l2g = self.u[0].femSpace.dofMap.l2g
            _u_q = np.einsum('kj,ej->ek',
                             self.u[0].femSpace.psi, self.u[0].dof[_l2g])
            self.coefficients.q_rho[:] = _rho_f*(1.0 + _eps_rho*_u_q)
            _ebN = self.mesh.exteriorElementBoundariesArray
            _eN = self.mesh.elementBoundaryElementsArray[_ebN, 0]
            _ebN_local = self.mesh.elementBoundaryLocalElementBoundariesArray[_ebN, 0]
            _u_ext = np.einsum('ekj,ej->ek',
                               self.u[0].femSpace.psi_trace[_ebN_local],
                               self.u[0].dof[_l2g[_eN]])
            self.coefficients.ebqe_rho[:] = _rho_f*(1.0 + _eps_rho*_u_ext)
            logEvent("TADR FCT: refreshed q_rho/ebqe_rho from the LIMITED u "
                     "(q_rho min=%.6e max=%.6e)"
                     % (float(self.coefficients.q_rho.min()),
                        float(self.coefficients.q_rho.max())), level=2)
            if _dbg:
                _s_fct = float(np.sum(self.u[0].dof * self.u[0].dof))
                try:
                    from proteus import Comm
                    _c = Comm.get().comm.tompi4py()
                    _s_low = _c.allreduce(_s_low)
                    _s_fct = _c.allreduce(_s_fct)
                except Exception:
                    pass
                logEvent("TADR FCT sharpness sum(u^2): low=%.6e fct=%.6e d=%+.3e %s"
                         % (_s_low, _s_fct, _s_fct - _s_low,
                            "(FCT SHARPENS)" if _s_fct > _s_low
                            else "(FCT DIFFUSES!)"), level=1)

    def updateAfterMeshMotion(self):
        pass
